import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, TILE_1D
from triwarp.kernels.array import (
    declare_map_signatures,
    map_probe,
    map_probe_single,
    mat33_column,
    pack_farthest_key,
    unpack_ranked_index,
)
from triwarp.kernels.grouping import hash_slot_words3, next_slot
from triwarp.kernels.neighbors import ACCEL_HASHGRID, ball_count_in_radius
from triwarp.kernels.predicates import point_plane_dot, triangle_normal
from triwarp.kernels.reduce import (
    block_chunk_1d,
    block_max,
    block_sum,
    commit_sum_and_count,
    outer_sum_chunk,
    tile_chunk,
)


@wp.func
def point_plane_distance(
    point: wp.vec3, plane_normal: wp.vec3, plane_origin: wp.vec3
) -> wp.float32:
    # Signed perpendicular distance: the shared unnormalized plane dot divided by the normal
    # length, so a non-unit ``plane_normal`` behaves like trimesh's reference. This is the only
    # caller that needs the division, so the dot stays the primitive.
    return point_plane_dot(point, plane_normal, plane_origin) / wp.length(plane_normal)


@wp.func
def is_in_half_space(point: wp.vec3, plane_normal: wp.vec3, plane_origin: wp.vec3) -> wp.bool:
    # Strictly on the normal's side, so a point exactly on the plane is excluded. Only the sign of
    # the dot matters, which is why this reads the unnormalized primitive rather than
    # ``point_plane_distance``: a non-unit normal cannot change the answer and the division cannot
    # change the sign, but it can turn a large dot into an infinity.
    return point_plane_dot(point, plane_normal, plane_origin) > 0.0


@wp.func
def centroid_from_sum(point_sum: wp.array[wp.vec3], n: wp.float32) -> wp.vec3:
    # ``point_sum[0] / n`` -- the same ``wp.div`` ``points.centroid``'s map applies, so a consumer
    # reading the raw sum gets the centroid's exact bits -- and the zero vector for a length-zero
    # array, which is how the uncentred Gram matrix asks for no center.
    if point_sum.shape[0] == 0:
        return wp.vec3(0.0, 0.0, 0.0)
    return point_sum[0] / n


@wp.kernel
def centered_covariance(
    points: wp.array[wp.vec3],
    center: wp.array[wp.vec3],
    center_divisor: wp.float32,
    out_cov: wp.array[wp.mat33],
) -> None:
    # Scatter matrix C = sum_k outer(x_k - c, x_k - c) with ``c = center[0] / center_divisor``, so
    # a caller holding the point *sum* passes it with ``n`` and pays no division launch of its own
    # (``centroid_from_sum``), and a caller holding a centroid passes ``1.0``, which divides
    # exactly. A length-zero ``center`` is the zero center, i.e. the uncentred Gram matrix
    # G = sum_k outer(x_k, x_k).
    #
    # Stays in ``points`` rather than moving to ``reduce`` with the rest of the chunked-accumulate
    # family: the reusable half -- the tile skeleton and ``outer_sum_chunk`` -- is already in
    # ``reduce`` and imported from there, and what is left is a point-cloud statistic whose only
    # callers are ``points``' own ``gram_matrix`` / ``fit_line`` / ``fit_plane``. ``triwarp.reduce``
    # is axis-parametrized array reductions in NumPy's vocabulary; a mat33 of second moments is not
    # one of those (section 11, the machinery half outranks the subject half).
    #
    # Launched ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: one block per
    # ``ITEMS_PER_BLOCK_1D`` points, lanes striding that block's own chunk, nine ``wp.tile_sum``
    # tree reductions, one ``wp.mat33`` atomic per block.
    #
    # **The redundant lane arithmetic is not the cost -- the block count is**, the same finding as
    # ``registration.accumulate_procrustes_moments``: what costs is one add per block on each of
    # nine hot addresses, so the lever is the fold width. Flat at small ``n``, an order of magnitude
    # at a million points. Nine reductions is the cheapest accumulator in this family to amortize,
    # which is why it turns over sooner than the 25- and 43-slot ``registration`` pair. The tree is
    # also the more accurate of the two, by about an order of magnitude at large ``n``.
    #
    # No ``prefers_tiled_reduction`` branch, for the reason in ``.claude/CLAUDE.md`` section 2.2:
    # the lanes partition a chunk the block already owns and stride by ``wp.block_dim()``, which
    # reads 1 on CPU. ``measures.centroid_tiled`` needs a device pair because *its* lanes partition
    # the outer work at a constant stride; this is the other form.
    chunk, lane = wp.tid()
    offset, remaining = block_chunk_1d(points.shape[0], chunk)
    if remaining <= 0:
        return

    m = outer_sum_chunk(
        points, centroid_from_sum(center, center_divisor), offset, remaining, lane, wp.block_dim()
    )

    # Block-collective, so it runs outside the ``lane == 0`` guard.
    total = block_sum(m)

    if lane == 0:
        wp.atomic_add(out_cov, 0, total)


@wp.kernel
def finalize_fit_line(m: wp.array[wp.mat33], out_axis: wp.array[wp.vec3]) -> None:
    # gram matrix of the (uncentred) points: M = sum_j outer(x_j, x_j)
    # the singular values / right singular vectors of M match the squared
    # singular values / right singular vectors of the (n, 3) point matrix.
    u, sigma, _v = wp.svd3(m[0])
    # axis = sum_i S_i * rsv_i, where S_i = sqrt(sigma_i) are the point singular values and the
    # right singular vectors are the columns of u -- i.e. exactly the matrix-vector product
    # u * (S_0, S_1, S_2). ``wp.sqrt`` is scalar-only, so the weight vector is built explicitly.
    axis = u * wp.vec3(wp.sqrt(sigma[0]), wp.sqrt(sigma[1]), wp.sqrt(sigma[2]))
    out_axis[0] = wp.normalize(axis)


@wp.kernel
def finalize_fit_plane(
    point_sum: wp.array[wp.vec3], n: wp.float32, m: wp.array[wp.mat33], out_plane: wp.array[wp.vec3]
) -> None:
    # One output buffer rather than two, because both entries cross to the host together and a
    # readback costs far more than the row of it that is read: slot 0 is the normal, slot 1 the
    # centroid, in the order ``fit_plane`` returns them.
    #
    # plane origin is the centroid of the point set; the covariance matrix of
    # the centred points was accumulated into m.
    u, _sigma, _v = wp.svd3(m[0])
    # normal is the singular vector with the smallest singular value
    # (svd3 returns singular values in descending order: last column of u).
    out_plane[0] = wp.normalize(mat33_column(u, 2))
    out_plane[1] = centroid_from_sum(point_sum, n)


# Orientation modes for estimate_point_normals (mirror Open3D's orient methods).
ORIENT_CENTROID = wp.constant(wp.int32(0))  # outward from the cloud centroid (the default)
ORIENT_DIRECTION = wp.constant(wp.int32(1))  # align with a fixed direction
ORIENT_CAMERA = wp.constant(wp.int32(2))  # point toward a camera location


@wp.kernel
def finalize_principal_axes(
    point_sum: wp.array[wp.vec3], n: wp.float32, m: wp.array[wp.mat33], out_frame: wp.array[wp.vec3]
) -> None:
    # One ``(5,)`` output buffer rather than three, because all three results cross to the host
    # together: rows 0-2 are the rotation's rows, row 3 the eigenvalues, row 4 the centroid --
    # the order ``principal_axes`` returns them in. One readback of five rows is several times
    # cheaper than three readbacks of one.
    # ``wp.svd3`` returns singular values in descending order, so the columns of ``u`` are the
    # principal axes from widest to narrowest. The rows of the result are those axes, which is the
    # convention that makes ``rotation * p`` the coordinates of ``p`` in the principal frame.
    u, sigma, _v = wp.svd3(m[0])
    axis0 = wp.normalize(mat33_column(u, 0))
    axis1 = wp.normalize(mat33_column(u, 1))
    # Take the third axis from the cross product rather than from ``u``: that forces a proper
    # rotation (determinant +1) whatever sign convention the SVD chose, so the frame is always
    # right-handed and only the first two signs are free.
    axis2 = wp.cross(axis0, axis1)
    out_frame[0] = axis0
    out_frame[1] = axis1
    out_frame[2] = axis2
    out_frame[3] = sigma
    out_frame[4] = centroid_from_sum(point_sum, n)


@wp.kernel
def estimate_point_normals(
    points: wp.array[wp.vec3],
    neighbor_idx: wp.array2d[wp.int32],
    point_sum: wp.array[wp.vec3],
    n_points: wp.float32,
    orient_mode: wp.int32,
    orient_reference: wp.vec3,
    out_normals: wp.array[wp.vec3],
) -> None:
    # Per-point normal = eigenvector of the smallest eigenvalue of the neighbourhood
    # covariance (the same choice Open3D's FastEigen3x3 makes).
    #
    # ``neighbor_idx``'s shape is validated by the caller, but a value ``>= points.shape[0]`` is
    # not -- indexing ``points[nb]`` below then reads out of bounds with no exception on CUDA and
    # corrupts the host heap on CPU (CLAUDE.md section 12.1). Every in-repo caller
    # (``points.estimate_normals``) builds this table from a *self*-query
    # (``query_nearest(points, points, k)``), which can only ever produce an in-range index or the
    # documented ``-1`` sentinel already handled below, so the gap is latent rather than
    # demonstrated. A guard would cost an unconditional device-wide min/max reduction over the
    # whole table on every call, to protect against an input shape nothing currently constructs --
    # the speculative-generality case CLAUDE.md section 4.2 asks to leave unbuilt. Revisit if a
    # caller ever builds an asymmetric ``neighbor_idx`` (e.g. from
    # ``query_nearest(other_cloud, points, k)``) for this kernel.
    v = wp.int32(wp.tid())
    k = neighbor_idx.shape[1]

    # Local neighbourhood mean over the valid entries of the table. A self-query table
    # contains the point itself once, so it is naturally included (matching Open3D KNN).
    mean = wp.vec3(0.0, 0.0, 0.0)
    count = wp.float32(0.0)
    for i in range(k):
        nb = neighbor_idx[v, i]
        if nb >= 0:
            mean += points[nb]
            count += 1.0

    if count < 2.0:
        out_normals[v] = wp.vec3(0.0, 0.0, 1.0)  # too few neighbours (Open3D fallback)
        return
    mean = mean / count

    cov = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(k):
        nb = neighbor_idx[v, i]
        if nb >= 0:
            e = points[nb] - mean
            cov += wp.outer(e, e)

    if wp.ddot(cov, cov) <= 0.0:
        out_normals[v] = wp.vec3(0.0, 0.0, 1.0)  # coincident neighbours (zero covariance)
        return

    u, _sigma, _vt = wp.svd3(cov)
    normal = wp.normalize(mat33_column(u, 2))

    # Orientation: flip so the normal points along a per-point reference vector.
    ref = wp.vec3(0.0, 0.0, 0.0)
    if orient_mode == ORIENT_CENTROID:
        # Outward from the cloud centroid (star-shaped assumption). ``point_sum`` is only read in
        # this mode; the others pass a null array.
        ref = points[v] - centroid_from_sum(point_sum, n_points)
    elif orient_mode == ORIENT_DIRECTION:
        ref = orient_reference
    else:
        ref = orient_reference - points[v]  # toward the camera location
    if wp.dot(normal, ref) < 0.0:
        normal = -normal
    out_normals[v] = normal


# Rows per block of ``neighbor_distance_moments``: four tiles, so a block of ``TILE_1D`` lanes walks
# four rows each. One row per lane (64) keeps the grid widest but puts one float64 commit per 64
# rows on the two accumulator slots, and at a million rows that contention made the fused kernel
# 1.4x the unfused pair it replaced; 1024 starves the grid below ~100 k rows (2.6x slower at 41 k).
# 256 measured at or below the unfused pair at both ends (0.54x at 41 k rows, 0.93x at 1 M, k = 16).
MOMENT_ROWS_PER_BLOCK = wp.constant(256)


@wp.kernel
def neighbor_distance_moments(
    neighbor_distance: wp.array2d[wp.float32],
    out_mean: wp.array[wp.float32],
    out_rms: wp.array[wp.float32],
    out_count: wp.array[wp.int32],
    out_totals: wp.array[wp.float64],
) -> None:
    # First and second moments of each point's neighbour distances, over the *filled* slots only:
    # ``query_nearest`` leaves unused slots at ``inf`` (index -1), and a row can be short when
    # ``max_radius`` bites or the cloud is smaller than ``k``. An empty row reports zeros with a
    # zero count, which is how both callers detect it.
    #
    # The mean feeds Open3D's statistical criterion and the RMS is the LoOP "standard distance".
    # Each caller wants only its own moments, so a length-zero output is skipped: the statistical
    # mask passes no ``out_rms`` and ``outlier_probability`` only ``out_rms``.
    #
    # The statistical mask also passes ``out_totals``, the first two slots of its
    # ``(count, total, deviation)`` accumulator, and gets ``(rows with at least one neighbour, sum
    # of their mean distances)`` folded into them here: a separate fold over this launch's own
    # outputs would re-read them for one more launch. ``MOMENT_ROWS_PER_BLOCK`` rows per block, a
    # narrower fold than the reduce module's, because the kernel writes per row and the wide fold
    # would collapse the grid (CLAUDE.md section 13.2); lane-strided by ``wp.block_dim()`` so the
    # CPU device's single lane covers the block's rows. ``float64`` slots: slot 0 is a *count*,
    # which a float32 stops representing exactly at 2 ** 24.
    #
    # An empty row contributes zero to both sums, which is what lets the cloud mean be a plain
    # reduction.
    chunk, lane = wp.tid()
    k = neighbor_distance.shape[1]
    offset, remaining = tile_chunk(neighbor_distance.shape[0], chunk, MOMENT_ROWS_PER_BLOCK)
    if remaining <= 0:
        return
    remaining = wp.min(remaining, MOMENT_ROWS_PER_BLOCK)
    counted = wp.float64(0.0)
    distance_total = wp.float64(0.0)
    for r in range(lane, remaining, wp.block_dim()):
        i = offset + r
        total = wp.float32(0.0)
        total_sq = wp.float32(0.0)
        count = wp.int32(0)
        for s in range(k):
            d = neighbor_distance[i, s]
            if not wp.isinf(d):
                total += d
                total_sq += d * d
                count += 1
        mean = wp.float32(0.0)
        rms = wp.float32(0.0)
        if count > 0:
            inverse = 1.0 / wp.float32(count)
            mean = total * inverse
            rms = wp.sqrt(total_sq * inverse)
            counted = counted + wp.float64(1.0)
            distance_total = distance_total + wp.float64(mean)
        if out_mean.shape[0] > 0:
            out_mean[i] = mean
        if out_rms.shape[0] > 0:
            out_rms[i] = rms
        if out_count.shape[0] > 0:
            out_count[i] = count
    # Launch-uniform, so the block-collective commit stays on one side of it for every lane. The
    # count goes in the *sum* slot of ``reduce.commit_sum_and_count``'s pair: this buffer's order
    # is ``(count, total)``.
    if out_totals.shape[0] > 0:
        commit_sum_and_count(lane, counted, distance_total, out_totals)


@wp.func
def local_outlier_factor(
    standard_distance: wp.array[wp.float32], neighbor_idx: wp.array2d[wp.int32], i: wp.int32
) -> wp.float32:
    """LoOP's probabilistic local outlier factor of point ``i``."""
    # How far this point's standard distance sits above the mean standard distance of its own
    # neighbourhood. Zero when the neighbourhood is empty or collapsed, so such a point never reads
    # as an outlier on this term alone.
    #
    # The LoOP normalization factor lambda cancels here (it scales numerator and denominator
    # alike); it only enters through the cloud-wide nplof the caller divides by.
    #
    # Same latent gap as ``estimate_point_normals`` above: ``neighbor_idx[i, s]`` is trusted to be
    # either ``-1`` or a valid row of ``standard_distance``, and nothing here bounds-checks it. The
    # one in-repo caller (``points.outlier_probability``) only ever builds this table from a
    # self-query, so this stays a documented, unguarded assumption rather than a demonstrated bug —
    # see the longer note there for why a guard is not added speculatively.
    k = neighbor_idx.shape[1]
    total = wp.float32(0.0)
    count = wp.int32(0)
    for s in range(k):
        j = neighbor_idx[i, s]
        if j >= 0:
            total += standard_distance[j]
            count += 1
    plof = wp.float32(0.0)
    if count > 0 and total > 0.0:
        plof = standard_distance[i] * wp.float32(count) / total - 1.0
    return plof


@wp.kernel
def local_outlier_factors(
    standard_distance: wp.array[wp.float32],
    neighbor_idx: wp.array2d[wp.int32],
    out_plof: wp.array[wp.float32],
    out_sum_squares: wp.array[wp.float32],
) -> None:
    # Every point's ``local_outlier_factor``, and the cloud-wide sum of their squares the LoOP
    # normalizer needs, folded as they are written. One tile of points per block (the grid keeps
    # its per-point width, since every point is written), lanes striding it by ``wp.block_dim()``
    # so the one CPU lane walks it all, and one atomic per block.
    block, lane = wp.tid()
    offset, remaining = tile_chunk(neighbor_idx.shape[0], block, TILE_1D)
    squares = wp.float32(0.0)
    for k in range(lane, wp.min(remaining, TILE_1D), wp.block_dim()):
        plof = local_outlier_factor(standard_distance, neighbor_idx, offset + k)
        out_plof[offset + k] = plof
        squares += plof * plof
    squares = block_sum(squares)
    if lane == 0:
        wp.atomic_add(out_sum_squares, 0, squares)


@wp.func
def outlier_probability(plof: wp.float32, inverse_normalizer: wp.float32) -> wp.float32:
    # LoOP score: the error function of the normalized factor, clamped at zero so an
    # inlier-or-better point reads exactly 0 rather than a negative "probability".
    return wp.max(wp.float32(0.0), wp.erf(plof * inverse_normalizer))


SQRT2_F64 = wp.constant(wp.float64(1.4142135623730951))


@wp.kernel
def outlier_probabilities(
    plof: wp.array[wp.float32],
    plof_sum_squares: wp.array[wp.float32],
    scale: wp.float64,
    n: wp.float32,
    out_probability: wp.array[wp.float32],
) -> None:
    # ``outlier_probability`` with the cloud-wide normalizer ``nplof = scale * sqrt(E[plof^2])``
    # formed here from the device sum rather than on the host, so no readback sits between the two
    # passes. Each thread repeats the host form's arithmetic operation for operation -- the mean in
    # ``float32``, then square root, product and reciprocal in ``float64``, each correctly rounded
    # on both devices -- so the scale it narrows to ``float32`` is the host's to the bit. A cloud
    # with no spread (every plof zero) scores zero.
    i = wp.int32(wp.tid())
    normalizer = scale * wp.sqrt(wp.float64(plof_sum_squares[0] / n))
    probability = wp.float32(0.0)
    if normalizer > wp.float64(0.0):
        inverse = wp.float32(wp.float64(1.0) / (normalizer * SQRT2_F64))
        probability = outlier_probability(plof[i], inverse)
    out_probability[i] = probability


@wp.func
def centered_square_if_counted(
    value: wp.float32, count: wp.int32, center: wp.float32
) -> wp.float32:
    # ``(value - center)^2`` for a counted row, 0 for an empty one: the masked variance term behind
    # the statistical threshold, so empty rows neither shift the mean nor inflate the deviation.
    if count == 0:
        return 0.0
    d = value - center
    return d * d


@wp.func
def is_statistical_outlier(
    mean_distance: wp.float32, count: wp.int32, threshold: wp.float32
) -> wp.bool:
    # Open3D's ``remove_statistical_outlier`` keeps a point when its mean neighbour distance is
    # strictly positive and strictly below the cloud threshold; everything else -- an empty
    # neighbourhood, a coincident one, or a far one -- is an outlier.
    return count == 0 or mean_distance <= 0.0 or mean_distance >= threshold


@wp.func
def counted_cloud_mean(totals: wp.array[wp.float64]) -> wp.float64:
    # The cloud mean over the counted rows, off ``neighbor_distance_moments``' ``(count, total)``
    # slots -- the division the host used to do after reading them back, in the same ``float64``.
    return totals[1] / totals[0]


@wp.func
def has_cloud_deviation(totals: wp.array[wp.float64]) -> wp.bool:
    # A (ddof = 1) cloud deviation needs two counted rows. The one rule both the deviation pass and
    # the mask launch branch on: below it the first adds nothing and the second disables the
    # threshold, so the two must never disagree about which side a cloud is on.
    return totals[0] >= wp.float64(2.0)


@wp.kernel
def accumulate_counted_deviation(
    count: wp.array[wp.int32], mean_distance: wp.array[wp.float32], out_totals: wp.array[wp.float64]
) -> None:
    # The squared-deviation sum behind ``statistical_outlier_mask``'s cloud deviation, into slot 2
    # of the ``(count, total, deviation)`` buffer whose first two slots
    # ``neighbor_distance_moments`` has already filled -- so the mean it centres on never leaves
    # the device. Each term is the ``float32`` ``centered_square_if_counted`` around the
    # ``float32`` cloud mean, as before; the terms are summed in ``float64``, which is also what the
    # Open3D reference accumulates in. A cloud with fewer than two counted rows has no deviation
    # and adds nothing (the mask kernel disables the threshold there).
    chunk, lane = wp.tid()
    offset, n_rows = block_chunk_1d(count.shape[0], chunk)
    if n_rows <= 0 or not has_cloud_deviation(out_totals):
        return
    center = wp.float32(counted_cloud_mean(out_totals))
    total = wp.float64(0.0)
    for k in range(lane, n_rows, wp.block_dim()):
        term = centered_square_if_counted(mean_distance[offset + k], count[offset + k], center)
        total = total + wp.float64(term)
    block = block_sum(total)
    if lane == 0:
        wp.atomic_add(out_totals, 2, block)


@wp.kernel
def statistical_outlier_from_totals(
    mean_distance: wp.array[wp.float32],
    count: wp.array[wp.int32],
    totals: wp.array[wp.float64],
    std_ratio: wp.float64,
    out_mask: wp.array[wp.bool],
) -> None:
    # ``is_statistical_outlier`` at the cloud threshold ``mean + std_ratio * std`` (ddof = 1), with
    # the threshold formed from the ``(count, total, deviation)`` slots in ``float64`` and rounded
    # once to ``float32``. Fewer than two counted rows give no deviation: the threshold is then
    # ``inf``, which disables only the distance third of the predicate and keeps the empty- and
    # coincident-neighbourhood halves.
    i = wp.int32(wp.tid())
    threshold = wp.float32(FLOAT32_INF_CONSTANT)
    if has_cloud_deviation(totals):
        cloud_std = wp.sqrt(totals[2] / (totals[0] - wp.float64(1.0)))
        threshold = wp.float32(counted_cloud_mean(totals) + std_ratio * cloud_std)
    out_mask[i] = is_statistical_outlier(mean_distance[i], count[i], threshold)


@wp.kernel
def radius_outlier_flags(
    points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    min_neighbors: wp.int32,
    out_mask: wp.array[wp.bool],
) -> None:
    # A point is a radius outlier when its ball holds at most ``min_neighbors`` points, itself
    # included: ``neighbors.query_ball_count``'s count, compared in the launch that forms it.
    i = wp.int32(wp.tid())
    count = ball_count_in_radius(points, ACCEL_HASHGRID, grid_id, points[i], radius)
    out_mask[i] = count <= min_neighbors


@wp.func
def is_finite_point(point: wp.vec3) -> wp.bool:
    # All three coordinates finite -- the row predicate behind ``point_finite_mask``. Any one NaN
    # or infinity condemns the point, which is what a downstream tree build or covariance fit
    # needs: a single non-finite coordinate poisons every reduction the point enters.
    return wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])


@wp.func
def zero_normalized_bits(value: wp.float32) -> wp.int32:
    # The coordinate's ``float32`` bit pattern, with ``-0.0`` folded onto ``+0.0``. ``wp.cast`` is a
    # bit reinterpretation, so it separates the two zeros -- and IEEE-754 equality does not, which
    # is the rule an exact-equality dedup has to reproduce. Every other value is its own bits.
    if value == 0.0:
        return wp.int32(0)
    return wp.cast(value, wp.int32)


@wp.func
def same_position_bits(a: wp.vec3, b: wp.vec3) -> wp.bool:
    # Exact position equality under the dedup's rule: equal ``float32`` bit patterns with the two
    # zeros folded together, so two identical ``NaN`` rows are one position and ``-0.0`` is
    # ``+0.0``. The one spelling ``point_duplicate_first`` compares candidates with.
    return (
        zero_normalized_bits(a[0]) == zero_normalized_bits(b[0])
        and zero_normalized_bits(a[1]) == zero_normalized_bits(b[1])
        and zero_normalized_bits(a[2]) == zero_normalized_bits(b[2])
    )


@wp.func
def position_hash_slot(point: wp.vec3, mask: wp.int32) -> wp.int32:
    # The home slot of a position's bit pattern. A colliding slot is resolved by
    # ``same_position_bits``, never trusted.
    return hash_slot_words3(
        zero_normalized_bits(point[0]),
        zero_normalized_bits(point[1]),
        zero_normalized_bits(point[2]),
        mask,
    )


@wp.kernel
def point_duplicate_first(
    points: wp.array[wp.vec3],
    mask: wp.int32,
    out_first: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
) -> None:
    # One open-addressing table of point indices, ``-1`` empty. A point claims an empty slot with
    # ``atomic_cas``; an occupied slot holds *some* point of its class, which is enough to compare
    # against, because every member of a class has the same position. Once a point has found its
    # class slot it lowers the slot to its own index with ``atomic_min`` -- a member's index never
    # reads as empty, so the probe of a concurrent thread is unaffected -- and after the launch
    # every class slot holds its smallest index: the first occurrence, whatever the arrival order.
    # ``out_first`` must arrive filled with ``-1``; it leaves holding the minima.
    i = wp.int32(wp.tid())
    point = points[i]
    h = position_hash_slot(point, mask)
    while True:
        prev = wp.atomic_cas(out_first, h, wp.int32(-1), i)
        if prev == wp.int32(-1):
            break
        if same_position_bits(points[prev], point):
            wp.atomic_min(out_first, h, i)
            break
        h = next_slot(h, mask)
    out_slot[i] = h


@wp.kernel
def point_duplicate_from_first(
    first: wp.array[wp.int32], slot: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # A point is a duplicate unless it is its class's smallest index.
    i = wp.int32(wp.tid())
    out_mask[i] = first[slot[i]] != i


# Lanes per block for ``farthest_point_sample_block``, keyed on the cloud size. The kernel is one
# persistent block, so this is the whole launch's width and there is nothing else to tune.
#
# **Three brackets, and the middle one is what a Warp 1.17 re-probe added.** The original reading
# was taken on Warp 1.16 against the capture-and-replay loop this kernel replaced and sampled only
# two widths, which is why it read as two brackets. A finer sweep -- interleaved, with the selected
# index buffer **identical at every width in every row**, so this is a pure cost choice -- shows a
# third bracket in between that the coarse one straddled: a small cloud wants a narrow block, a
# large one wants the widest, and there is a real middle band that wants 512.
#
# The brackets are **not** an artefact of ``count``: re-run at a quarter of the sample count, the
# best width is the same at every size. Against the two-bracket dispatch this replaces the win is
# largest exactly at the old crossover, where a cloud that wants 512 lanes was handed 1 024. 2 048
# is where 256 and 512 measure a tie, which is what makes it a safe boundary in either direction.
#
# A constant whose sweep sampled only two values has not been shown to be a two-bracket problem.
FARTHEST_BLOCK_SMALL = 256
FARTHEST_BLOCK_MID = 512
FARTHEST_BLOCK_MID_FROM = 2048
FARTHEST_BLOCK_LARGE = 1024
FARTHEST_BLOCK_LARGE_FROM = 6144


@wp.kernel
def farthest_point_sample_block(
    points: wp.array[wp.vec3],
    start: wp.int32,
    count: wp.int32,
    min_distance_sq: wp.array[wp.float32],
    out_selected: wp.array[wp.int32],
) -> None:
    # The whole greedy sweep in one persistent block: ``count - 1`` rounds, each folding the point
    # just selected into every point's running distance to the chosen set and taking the global
    # arg-max, with ``wp.tile_max`` as both the reduction and the round barrier. Lane ``t`` owns
    # the points ``t, t + block_dim, ...`` throughout, so no other synchronization is needed --
    # including none between the initialization below and the first round.
    #
    # Why a single block wins here where it loses elsewhere (``kernels/holes.py::
    # fill_dp_span_tiled`` measured the opposite): a round is ``n`` distance updates, which one SM
    # finishes in about a microsecond, and the alternative -- one launch per round, even replayed
    # from a captured graph -- paid more than that in launch cost with the device idle. So the
    # per-round cost is what moved. The tie-break is the shipped one: ``pack_farthest_key`` orders
    # equal distances towards the lower index, and the block max over those keys is the same maximum
    # whatever lane holds it, so the selection is reproducible and matches the reference's strict
    # ``>`` arg-max.
    #
    # ``min_distance_sq`` is caller-allocated scratch, initialized here rather than by ``wp.full``
    # so the wrapper is one launch. The stride is ``wp.block_dim()`` rather than the constant so the
    # kernel is also correct on the CPU device, where ``wp.launch_tiled`` runs one lane per block
    # and ``wp.block_dim()`` reads 1: that lane walks the whole cloud each round and the tile max
    # returns its own key.
    _block, t = wp.tid()
    n = points.shape[0]
    stride = wp.block_dim()
    for i in range(t, n, stride):
        min_distance_sq[i] = wp.inf
    chosen_index = start
    if t == 0:
        out_selected[0] = start
    for step in range(1, count):
        chosen = points[chosen_index]
        best = wp.int64(-1)
        for i in range(t, n, stride):
            distance_sq = wp.min(min_distance_sq[i], wp.length_sq(points[i] - chosen))
            min_distance_sq[i] = distance_sq
            best = wp.max(best, pack_farthest_key(distance_sq, i))
        chosen_index = unpack_ranked_index(block_max(best))
        if t == 0:
            out_selected[step] = chosen_index


# Directions one ``hull_support_extremes`` thread may reduce together: the point it loads is dotted
# with each of them from registers, so the cloud is streamed ``n_directions / width`` times rather
# than ``n_directions`` times -- the kernel is bound by that traffic, not by its arithmetic. Each
# launch takes the widest that still leaves ``SUPPORT_MIN_THREADS`` threads (``support_width``),
# since grouping directions divides the grid: 8 directions a thread is 2.7-3.1x on a 0.5 M-point
# cloud and 4.5-11x on 14 M against one, 16 is slower than 8, and a 36 k-point cloud has too few
# slices to group at all.
SUPPORT_DIRECTION_WIDTHS = (8, 4, 2, 1)
SUPPORT_MIN_THREADS = 1 << 17


def _hull_support_extremes_kernel(width: int) -> wp.Kernel:
    """Build ``hull_support_extremes`` over ``width`` directions per thread."""
    extremes_t = wp.types.vector(length=width, dtype=wp.float32)
    directions_t = wp.types.matrix(shape=(width, 3), dtype=wp.float32)

    def hull_support_extremes(
        points: wp.array[wp.vec3],
        directions: wp.array[wp.vec3],
        n_slices: wp.int32,
        out_best_max: wp.array[wp.float32],
        out_best_min: wp.array[wp.float32],
        out_slice_max: wp.array2d[wp.float32],
        out_slice_min: wp.array2d[wp.float32],
    ) -> None:
        # Thread ``(b, j)`` reduces directions ``b * width ..`` over slice ``j``. Max and min are
        # exact, so the grouping changes no extreme.
        #
        # ``out_slice_max`` / ``out_slice_min`` keep a bound on each ``(direction, slice)`` extreme
        # for the filtered sweeps after this one (``support_indices``, ``mark_hull_support``): a
        # slice whose bound does not reach a direction's threshold holds no point that does, so
        # only the few that do are walked again. Both may be null descriptors (``None``).
        b, j = wp.tid()
        n_p = points.shape[0]
        n_dir = directions.shape[0]
        k0 = b * width
        # Strided slice, NOT a contiguous chunk: consecutive threads read consecutive points, so the
        # loads coalesce, and each thread contributes one atomic instead of one per point.
        #
        # Lane-free because the threads partition the **outer** work -- the cloud this reduction is
        # over -- rather than a sequence one block owns, so there is no `wp.block_dim()` for them to
        # stride by and a `wp.tile_max(wp.tile(...))` cannot be reached from here without changing
        # the launch. `wp.launch_tiled` runs one lane per block on the CPU device through Warp 1.17,
        # and that lane would then cover `1/block_dim` of the slice. See `.claude/CLAUDE.md` section
        # 2.2; `farthest_point_sample_block` below is the other side of the rule, and reduces with
        # `wp.tile_max` on both devices because its stride *is* `wp.block_dim()`.
        #
        # **Converting this to one block per direction was measured and refuted**, which is worth
        # recording because the analogous rewrite of `kernels/visibility.py::obscurance` was a large
        # win and the shapes look alike. They are not: `obscurance` launched `dim = n_points` with
        # no second dimension, so the outer dimension alone was starving the device, while this
        # kernel's *slice* dimension is what fills it. Collapsing that into `block_dim` lanes leaves
        # one block per direction: measured with answers bit-identical, that is a win on a small
        # cloud -- where the call is already microseconds -- and a several-fold **loss** on a large
        # one, which is section 13's decline shape exactly. Do not re-propose it from the comment
        # above.
        dirs = directions_t()
        for d in range(width):
            if k0 + d < n_dir:
                dirs[d] = directions[k0 + d]
        local_max = extremes_t(-FLOAT32_INF_CONSTANT)
        local_min = extremes_t(FLOAT32_INF_CONSTANT)
        local_scale = extremes_t(0.0)
        tables = out_slice_max.shape[0] > 0
        for i in range(j, n_p, n_slices):
            point = points[i]
            for d in range(width):
                distance = wp.dot(dirs[d], point)
                local_max[d] = wp.max(local_max[d], distance)
                local_min[d] = wp.min(local_min[d], distance)
            if tables:
                magnitude = wp.abs(point)
                for d in range(width):
                    local_scale[d] = wp.max(local_scale[d], wp.dot(wp.abs(dirs[d]), magnitude))

        for d in range(width):
            k = k0 + d
            if k < n_dir:
                if tables:
                    # The sweeps that read these recompute each dot product in other code, where
                    # FMA contraction may round it differently (see ``SUPPORT_TIE_SLACK``), so a
                    # bare extreme could sit a ULP inside a point's recomputed value and skip it.
                    # Widened by a bound on that rounding difference -- a few units in the last
                    # place of ``sum |d_i p_i|``, the magnitude every term of the dot carries --
                    # the filter is conservative, never exact-only.
                    margin = wp.float32(1e-6) * local_scale[d]
                    out_slice_max[k, j] = local_max[d] + margin
                    out_slice_min[k, j] = local_min[d] - margin
                # A slice past the end of the cloud contributes nothing.
                if local_max[d] > -FLOAT32_INF_CONSTANT:
                    wp.atomic_max(out_best_max, k, local_max[d])
                    wp.atomic_min(out_best_min, k, local_min[d])

    return wp.kernel(hull_support_extremes, name=f"hull_support_extremes_{width}")


HULL_SUPPORT_EXTREMES = {w: _hull_support_extremes_kernel(w) for w in SUPPORT_DIRECTION_WIDTHS}


def support_width(n_directions: int, n_slices: int) -> int:
    """Return the widest ``SUPPORT_DIRECTION_WIDTHS`` entry keeping ``SUPPORT_MIN_THREADS``."""
    for width in SUPPORT_DIRECTION_WIDTHS:
        if -(-n_directions // width) * n_slices >= SUPPORT_MIN_THREADS:
            return width
    return 1


@wp.func
def support_slack(best_max: wp.float32, best_min: wp.float32, tolerance: wp.float32) -> wp.float32:
    # Slack scales with the per-direction support extent so the test is scale-invariant and stays
    # above the float32 dot-product noise floor. Shared by `mark_hull_support` and
    # `support_indices`, which both tie-break against it: a future change to the formula (e.g. a
    # different scale-invariance fix) has one definition to change rather than two that can drift.
    return tolerance * (best_max - best_min)


@wp.kernel
def mark_hull_support(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    stride: wp.int32,
    slice_max: wp.array2d[wp.float32],
    slice_min: wp.array2d[wp.float32],
    out_mask: wp.array[wp.bool],
) -> None:
    # Thread ``(k, j)`` tests points ``j, j + stride, ...`` against direction ``k``. Two launch
    # shapes, one rule: one thread per point (``stride`` the point count, the slice tables
    # ``None``), or one per ``(direction, slice)`` of ``hull_support_extremes``' launch with its
    # slice tables, walking a slice only if the slice's own extreme reaches one of the direction's
    # two thresholds -- every point the exhaustive test would mark lies in such a slice, since its
    # support value is the same ``wp.dot`` the slice extreme was folded from.
    k, j = wp.tid()
    # A hemisphere direction n covers both +n (max, supports the vertex farthest
    # along n) and -n (min, supports the vertex farthest along -n).
    direction = directions[k]
    slack = support_slack(best_max[k], best_min[k], tolerance)
    upper = best_max[k] - slack
    lower = best_min[k] + slack
    if slice_max.shape[0] > 0:
        if slice_max[k, j] < upper and slice_min[k, j] > lower:
            return
    for i in range(j, points.shape[0], stride):
        distance = wp.dot(direction, points[i])
        if distance >= upper or distance <= lower:
            out_mask[i] = True


@wp.kernel
def support_indices(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    stride: wp.int32,
    slice_max: wp.array2d[wp.float32],
    out_support: wp.array[wp.int32],
) -> None:
    # ``mark_hull_support``'s two launch shapes, for the upper threshold alone, stopping at a
    # thread's first attaining point -- its points ascend, so that is its lowest.
    k, j = wp.tid()
    upper = best_max[k] - support_slack(best_max[k], best_min[k], tolerance)
    if slice_max.shape[0] > 0:
        if slice_max[k, j] < upper:
            return
    direction = directions[k]
    for i in range(j, points.shape[0], stride):
        if wp.dot(direction, points[i]) >= upper:
            # Lowest attaining index wins, so the shell is identical across launches even when
            # several points tie for the support along a direction.
            wp.atomic_min(out_support, k, i)
            break


@wp.kernel
def shell_bounds(
    shell_vertices: wp.indexedarray[wp.vec3],
    out_centroid: wp.array[wp.vec3],
    out_radius: wp.array[wp.float32],
) -> None:
    # One thread: the shell has a few hundred vertices at most, and reducing on device keeps the
    # support sweep and the tetrahedron build in one launch chain with no host readback between.
    # ``shell_vertices`` is the points gathered through the support indices as a Python-scope
    # ``wp.indexedarray`` view, so neither this kernel nor ``tetrahedron_planes`` needs a copy.
    n = shell_vertices.shape[0]
    total = wp.vec3(0.0, 0.0, 0.0)
    for i in range(n):
        total = total + shell_vertices[i]
    center = total / wp.float32(n)

    # The radius is the length scale the interior margin is measured against, so that the margin is
    # a fraction of the construction's own size rather than of a tetrahedron's aspect ratio.
    radius = wp.float32(0.0)
    for i in range(n):
        radius = wp.max(radius, wp.length(shell_vertices[i] - center))

    out_centroid[0] = center
    out_radius[0] = radius


@wp.func
def outward_plane(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, interior: wp.vec3) -> wp.vec4:
    """Unit-normal plane through the triangle, oriented so ``interior`` has negative offset."""
    # `predicates.triangle_normal` is exactly this triangle's normalize(cross(...)), including the
    # degenerate-triangle convention (the zero vector) this function's own early return needs.
    normal = triangle_normal(p0, p1, p2)
    if wp.length_sq(normal) <= 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    offset = wp.dot(normal, p0)
    if wp.dot(normal, interior) > offset:
        return wp.vec4(-normal[0], -normal[1], -normal[2], -offset)
    return wp.vec4(normal[0], normal[1], normal[2], offset)


@wp.kernel
def tetrahedron_planes(
    shell_vertices: wp.indexedarray[wp.vec3],
    shell_faces: wp.array[wp.int32],
    centroid: wp.array[wp.vec3],
    flatness: wp.float32,
    out_planes: wp.array2d[wp.vec4],
    out_valid: wp.array[wp.bool],
) -> None:
    t = wp.int32(wp.tid())
    apex = centroid[0]
    a = shell_vertices[shell_faces[t * 3 + 0]]
    b = shell_vertices[shell_faces[t * 3 + 1]]
    c = shell_vertices[shell_faces[t * 3 + 2]]

    # Scale-free flatness test: the determinant of the three apex edges against their length
    # product. A sliver's face normals are ill-conditioned cross products of nearly parallel edges,
    # and that error is the only one that can cost the superset guarantee, so reject generously --
    # neighbouring well-shaped tetrahedra cover the same region. A wholly degenerate shell (a
    # coplanar or collinear cloud) rejects every tetrahedron, and the filter then keeps all points.
    ea = a - apex
    eb = b - apex
    ec = c - apex
    m = wp.matrix_from_cols(ea, eb, ec)
    if wp.abs(wp.determinant(m)) <= flatness * wp.length(ea) * wp.length(eb) * wp.length(ec):
        out_valid[t] = False
        return

    # Half-space form with *unit* normals, so the interior test below is a true distance and its
    # margin can be a length. Barycentric coordinates would be the cheaper test but their margin is
    # meaningless as a distance: these tetrahedra run from the centroid out to the shell, so a fixed
    # barycentric slack cuts a thick layer off the base and nothing off the sides.
    inner = (apex + a + b + c) / 4.0
    out_planes[t, 0] = outward_plane(a, b, c, inner)
    out_planes[t, 1] = outward_plane(apex, a, b, inner)
    out_planes[t, 2] = outward_plane(apex, b, c, inner)
    out_planes[t, 3] = outward_plane(apex, c, a, inner)
    out_valid[t] = True


@wp.kernel
def mark_hull_superset(
    points: wp.array[wp.vec3],
    planes: wp.array2d[wp.vec4],
    valid: wp.array[wp.bool],
    radius: wp.array[wp.float32],
    margin: wp.float32,
    out_mask: wp.array[wp.bool],
) -> None:
    i = wp.int32(wp.tid())
    point = points[i]
    # Strict interior only, by a real distance. A point on a tetrahedron's boundary can still be a
    # hull vertex, and requiring it to clear every face by `slack` means float32 error in the plane
    # evaluation can only keep a point that could have been dropped -- never drop a hull vertex.
    slack = margin * radius[0]
    n_tetra = valid.shape[0]
    keep = wp.int32(1)
    for t in range(n_tetra):
        if valid[t]:
            inside = wp.int32(1)
            for f in range(4):
                plane = planes[t, f]
                normal = wp.vec3(plane[0], plane[1], plane[2])
                if wp.dot(normal, point) - plane[3] >= -slack:
                    inside = wp.int32(0)
                    break
            if inside != 0:
                keep = wp.int32(0)
                break
    out_mask[i] = keep != 0


@wp.func
def radial_sort_key(point: wp.vec3, origin: wp.vec3, axis0: wp.vec3, axis1: wp.vec3) -> wp.float32:
    v = point - origin
    # Negated angle: an ascending radix sort of these keys reproduces trimesh's
    # descending-angle order (`angles.argsort()[::-1]`).
    return -wp.atan2(wp.dot(v, axis0), wp.dot(v, axis1))


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).

    Three ops are ``wp.map``'d from ``points.py``; all three are declared. ``is_finite_point``
    (``point_finite_mask``) and ``radial_sort_key`` (``radial_sort``) are reachable from a length-1
    point cloud exactly as ``is_in_half_space`` is, and each forks its own module the first time a
    call at the other length is seen.
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (is_in_half_space, (dense(wp.vec3), wp.vec3(), wp.vec3()), wp.bool),
            (is_in_half_space, (single(wp.vec3), wp.vec3(), wp.vec3()), wp.bool),
            (is_finite_point, (dense(wp.vec3),), wp.bool),
            (is_finite_point, (single(wp.vec3),), wp.bool),
            (radial_sort_key, (dense(wp.vec3), wp.vec3(), wp.vec3(), wp.vec3()), wp.float32),
            (radial_sort_key, (single(wp.vec3), wp.vec3(), wp.vec3(), wp.vec3()), wp.float32),
        ]
    )


_declare_map_kernels()
