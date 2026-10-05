import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT
from ordito.kernels.array import LOOP_CONDITION, LOOP_ROUND, binary_search_index_left
from ordito.kernels.neighbors import mesh_nearest_point
from ordito.kernels.predicates import normalize_or_zero
from ordito.kernels.proximity import closest_point_query
from ordito.kernels.reduce import (
    ITEMS_PER_BLOCK_1D,
    block_chunk_1d,
    block_sum,
    commit_block_sum,
    commit_block_total,
    commit_sum_and_count,
)
from ordito.kernels.transform import transform_point_mat44

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Packed Procrustes accumulator
#
# Every moment the fit needs lives in one ``float32`` buffer, so the whole thing costs a single
# allocation and a single memset instead of six of each — which is most of the host-side latency
# at the sizes this is called at (``icp`` calls it once per iteration, and the cost was flat in
# ``n`` from 8k points upward).
#
# Slots 1..24 are the *shifted* moments: sums of ``a - p`` and ``b - q`` where ``p = a[0]`` and
# ``q = b[0]``. Shifting by a point *of the cloud* is what lets one pass do the work of two. The
# textbook single-pass identity shifts by the origin, which makes the cancellation in
# ``E[x^2] - E[x]^2`` scale as ``(|centroid| / spread)^2`` — unbounded for a cloud far from the
# origin. Shifting by a sample point bounds it at ``(diameter / spread)^2``, a handful of bits.
# ---------------------------------------------------------------------------
ACC_W_SUM = wp.constant(0)  # sum w
ACC_A_SUM = wp.constant(1)  # vec3, slots 1..3:   sum w (a - p)
ACC_B_SUM = wp.constant(4)  # vec3, slots 4..6:   sum w (b - q)
ACC_A_SQ = wp.constant(7)  # sum w |a - p|^2
ACC_B_SQ = wp.constant(8)  # sum w |b - q|^2
ACC_COV = wp.constant(9)  # mat33, slots 9..17: sum_{w>0} outer(b - q, a - p), row-major
ACC_MASK_A = wp.constant(18)  # vec3, slots 18..20: sum_{w>0} (a - p)
ACC_MASK_B = wp.constant(21)  # vec3, slots 21..23: sum_{w>0} (b - q)
ACC_MASK_N = wp.constant(24)  # count of w > 0
# The moment slots, ``ACC_W_SUM`` through ``ACC_MASK_N``: what one block of
# ``accumulate_procrustes_moments`` folds. A plain constant, since a vector length must be one.
PROCRUSTES_MOMENT_SLOTS = wp.constant(25)
ACC_COST = wp.constant(25)  # weighted mean squared residual
PROCRUSTES_ACC_SIZE = 26
# ``icp``'s loop carries one slot more: the cost of the fit it has kept so far, which
# ``point_to_point_round`` writes and never zeroes (``inf`` if round 0 was weightless).
ACC_KEPT_COST = wp.constant(26)
ICP_POINT_ACC_SIZE = 27

# Point-to-plane accumulator, one ``float32`` buffer for the same reason the moments above share
# one: the normal matrix, its right-hand side and the two scalars ``accumulate_point_to_plane``
# folds as **one** packed block reduction and commits, then the previous kept cost the round's
# convergence test reads. One allocation and one zero-fill for the whole loop state.
ICP_JTJ = wp.constant(0)  # spatial_matrix, slots 0..35, row-major
ICP_JTR = wp.constant(36)  # spatial_vector, slots 36..41
ICP_COST = wp.constant(42)  # sum robust_loss(r); see ``robust_weight_and_loss``
ICP_WEIGHT_SUM = wp.constant(43)  # sum w
# What one block of ``accumulate_point_to_plane`` folds, ``ICP_JTJ`` through ``ICP_WEIGHT_SUM``.
ICP_MOMENT_SLOTS = wp.constant(44)
# The previous kept fit's cost, the ``float32`` it was read from. Never zeroed and never read before
# the first kept fit: ``point_to_plane_round`` reads ``inf`` while the round counter is 0.
ICP_OLD_COST = wp.constant(44)
ICP_ACC_SIZE = 45


@wp.func
def procrustes_shifts(
    a: wp.array[wp.vec3], b: wp.array[wp.vec3], use_translation: wp.bool
) -> tuple[wp.vec3, wp.vec3]:
    # Read the shift origins straight off the device (a broadcast load, no host round trip).
    # ``use_translation=False`` must shift by nothing at all, so the arithmetic stays bit-for-bit
    # the uncentered form the caller asked for.
    shift_a = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    shift_b = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation and a.shape[0] > 0:
        shift_a = a[0]
        shift_b = b[0]
    return shift_a, shift_b


@wp.func
def sample_weight(weights: wp.array[wp.float32], i: wp.int32) -> wp.float32:
    # Sample ``i``'s weight, a zero-length ``weights`` meaning uniform -- which is what ``icp``
    # passes: the alternative is a ``wp.full(n, 1.0)`` allocation *and* fill on every iteration,
    # for a value the kernel can just assume.
    w = wp.float32(1.0)
    if weights.shape[0] > 0:
        w = weights[i]
    return w


@wp.func
def make_affine44(rotation: wp.mat33, translation: wp.vec3) -> wp.mat44:
    """Pack a 3x3 linear part and a translation into a homogeneous transform."""
    return wp.mat44(
        rotation[0, 0],
        rotation[0, 1],
        rotation[0, 2],
        translation[0],
        rotation[1, 0],
        rotation[1, 1],
        rotation[1, 2],
        translation[1],
        rotation[2, 0],
        rotation[2, 1],
        rotation[2, 2],
        translation[2],
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(1.0),
    )


@wp.kernel
def accumulate_procrustes_moments(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    use_translation: wp.bool,
    out_acc: wp.array[wp.float32],
) -> None:
    # Every moment the fit needs, in one pass over ``a`` and ``b``. The two-pass form this replaces
    # could not be fused because the covariance needs the centroid the first pass computes; the
    # shifted-moment identity in ``build_procrustes_matrix`` removes that dependency.
    #
    # Launched ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: one block per
    # ``ITEMS_PER_BLOCK_1D`` points, lanes striding that block's own chunk, and one packed
    # 25-slot tree reduction committing **one** atomic set per block.
    #
    # **The 64-fold redundant lane arithmetic is not the cost, and never was**: all lanes read the
    # same ``a[offset + k]`` and the loads broadcast out of one cache line, so giving each lane
    # exactly one element instead is worth a few percent at every size. What costs is the *atomic
    # contention* -- 25 hot addresses taking one add per block -- so the lever is the fold width.
    # Folding ``TILES_PER_BLOCK_1D`` tiles into each block cuts the block count sixteenfold: flat at
    # small ``n``, an order of magnitude at a million points.
    #
    # The tree is also *more* accurate than the serialized form, as in ``accumulate_cost``.
    #
    # No ``prefers_tiled_reduction`` branch: the lanes partition a chunk the block already owns and
    # stride by ``wp.block_dim()``, which reads 1 on CPU, so lane 0 walks the whole chunk and the
    # one-element tiles hold its true totals. The CPU clock is flat and the accuracy win holds
    # there too.
    #
    # A zero-length ``weights`` means uniform weights (``sample_weight``).
    chunk, lane = wp.tid()
    offset, count = block_chunk_1d(a.shape[0], chunk)
    if count <= 0:
        return
    shift_a, shift_b = procrustes_shifts(a, b, use_translation)

    w_sum = wp.float32(0.0)
    a_sum = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    b_sum = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    a_sq = wp.float32(0.0)
    b_sq = wp.float32(0.0)
    cov = wp.mat33(wp.float32(0.0))
    mask_a = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    mask_b = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    mask_n = wp.float32(0.0)

    for k in range(lane, count, wp.block_dim()):
        index = offset + k
        w = sample_weight(weights, index)
        av = a[index] - shift_a
        bv = b[index] - shift_b
        w_sum += w
        a_sum += w * av
        b_sum += w * bv
        a_sq += w * wp.length_sq(av)
        b_sq += w * wp.length_sq(bv)
        # The covariance is weighted by *membership*, not magnitude — trimesh's convention, and
        # what ``test_procrustes_matches_trimesh[binary_weights]`` pins.
        if w > wp.float32(0.0):
            cov += wp.outer(bv, av)
            mask_a += av
            mask_b += bv
            mask_n += wp.float32(1.0)

    # Packed into the accumulator's own slot layout and folded by **one** block reduction, which
    # is block-collective, so it runs outside the ``lane == 0`` guard and only the commit is
    # guarded. The default constructor, not a zero-fill: every slot is written just below.
    packed = wp.vector(length=PROCRUSTES_MOMENT_SLOTS, dtype=wp.float32)
    packed[ACC_W_SUM] = w_sum
    packed[ACC_A_SQ] = a_sq
    packed[ACC_B_SQ] = b_sq
    packed[ACC_MASK_N] = mask_n
    for c in range(3):
        packed[ACC_A_SUM + c] = a_sum[c]
        packed[ACC_B_SUM + c] = b_sum[c]
        packed[ACC_MASK_A + c] = mask_a[c]
        packed[ACC_MASK_B + c] = mask_b[c]
        for r in range(3):
            packed[ACC_COV + c * 3 + r] = cov[c, r]
    commit_block_sum(lane, packed, out_acc, 0)


@wp.func
def acc_vec3(acc: wp.array[wp.float32], base: wp.int32) -> wp.vec3:
    """Read a three-slot vector out of the packed accumulator."""
    return wp.vec3(acc[base], acc[base + 1], acc[base + 2])


@wp.func
def procrustes_matrix(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    acc: wp.array[wp.float32],
    use_reflection: wp.bool,
    use_translation: wp.bool,
    use_scale: wp.bool,
) -> wp.mat44:
    ws = acc[ACC_W_SUM]
    shift_a, shift_b = procrustes_shifts(a, b, use_translation)

    # Centroids relative to the shift origins, then un-shifted back into world coordinates.
    a_rel = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    b_rel = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        a_rel = acc_vec3(acc, ACC_A_SUM) / ws
        b_rel = acc_vec3(acc, ACC_B_SUM) / ws
    acenter = shift_a + a_rel
    bcenter = shift_b + b_rel

    ascale = wp.float32(1.0)
    bscale = wp.float32(1.0)
    if use_scale:
        # Shifted second-moment identity: sum w |a - centroid|^2 / S_w = sum w |a - p|^2 / S_w
        # minus |centroid - p|^2. Mathematically non-negative, but computed as a difference of two
        # independently tile-reduced sums (CLAUDE.md section 12.4, non-associative float32
        # accumulation), so a near-degenerate cloud (tightly clustered, or exactly duplicated
        # points) can land it at a tiny negative value from cancellation alone -- floored the same
        # way ``solve_spd6`` below floors its own Cholesky pivot for the identical reason, rather
        # than feeding ``wp.sqrt`` a negative argument and propagating NaN through the SVD.
        ascale = wp.sqrt(wp.max(acc[ACC_A_SQ] / ws - wp.length_sq(a_rel), wp.float32(1e-20)))
        bscale = wp.sqrt(wp.max(acc[ACC_B_SQ] / ws - wp.length_sq(b_rel), wp.float32(1e-20)))

    # Shifted cross-moment identity, over the membership-masked subset:
    # H = sum_m outer(b - bc, a - ac)
    #   = M - outer(Sm_b, ac') - outer(bc', Sm_a) + N_m outer(bc', ac')
    cov = wp.matrix_from_rows(
        acc_vec3(acc, ACC_COV + 0), acc_vec3(acc, ACC_COV + 3), acc_vec3(acc, ACC_COV + 6)
    )
    cov = (
        cov
        - wp.outer(acc_vec3(acc, ACC_MASK_B), a_rel)
        - wp.outer(b_rel, acc_vec3(acc, ACC_MASK_A))
        + acc[ACC_MASK_N] * wp.outer(b_rel, a_rel)
    )

    # Normalise cross-covariance by scale product
    inv_scales = wp.float32(1.0) / (bscale * ascale)
    target = cov * inv_scales

    U, sigma, V = wp.svd3(target)  # noqa: N806

    Vt = wp.transpose(V)  # noqa: N806

    # wp.svd3 may return negative singular values; absorb their signs into a
    # diagonal correction matrix so R = U @ D @ V^T matches the numpy convention
    # (all-positive sigma) and correctly handles reflective optimal solutions.
    # wp.sign is -1 for negative components and +1 otherwise (including at exactly 0).
    d = wp.sign(sigma)
    R = U * wp.diag(d) * Vt  # noqa: N806

    if not use_reflection:
        # Ensure det(R) = 1 by flipping the last correction factor when needed. Rebuilding ``R``
        # only on the branch that actually flips ``d`` -- the common, no-flip case reuses the
        # matrix just computed above instead of recomputing the identical product from the same,
        # unchanged ``d``.
        if wp.determinant(R) < wp.float32(0.0):
            d = wp.vec3(d[0], d[1], -d[2])
            R = U * wp.diag(d) * Vt  # noqa: N806

    s = wp.float32(1.0)
    if use_scale:
        s = bscale / ascale

    sR = s * R  # noqa: N806

    t = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        t = bcenter - sR * acenter

    return make_affine44(sR, t)


@wp.kernel
def build_procrustes_matrix(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    acc: wp.array[wp.float32],
    use_reflection: wp.bool,
    use_translation: wp.bool,
    use_scale: wp.bool,
    out_matrix: wp.array[wp.mat44],
) -> None:
    out_matrix[0] = procrustes_matrix(a, b, acc, use_reflection, use_translation, use_scale)


@wp.kernel
def fit_transform_and_accumulate_cost(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    acc: wp.array[wp.float32],
    use_reflection: wp.bool,
    use_translation: wp.bool,
    use_scale: wp.bool,
    out_matrix: wp.array[wp.mat44],
    out_transformed: wp.array[wp.vec3],
) -> None:
    # Form the fit from the finished moments, apply it to ``a``, publish the moved points, and
    # reduce the weighted mean squared residual against ``b`` into the packed accumulator's cost
    # slot -- one pass, one launch. A zero-length ``weights`` means uniform (``sample_weight``), and
    # a zero-length ``out_transformed`` publishes nothing: ``icp``'s device loop wants only the cost
    # of a fit it may yet reject, and moves the points once, after the loop, by the transform it
    # kept.
    #
    # Every lane forms the fit itself (``procrustes_matrix`` on the same ``acc``, so every lane
    # holds the one matrix, bit for bit) and the first lane of block 0 publishes it, which is what
    # ``build_procrustes_matrix`` writes when no cost is wanted. That replaced a ``dim=1`` fit
    # launch between the moments and this pass: 1.04-1.07x on ``procrustes(return_cost=True)``
    # from 16 k to 0.87 M points, level at 14 M, and 1.00-1.02x on ``icp``'s recorded rounds. The
    # cost slot the reduction commits is not one the fit reads.
    #
    # The reduction is the lane-strided single-slot form of CLAUDE.md section 13.2, striding by
    # ``wp.block_dim()`` so it needs no ``prefers_tiled_reduction`` branch. The transform only
    # became worth fusing in once that reduction was flat: as one ``wp.atomic_add`` per thread to a
    # constant slot it dominated, and at a launch floor removing a launch is most of the pair.
    # ``apply_transform_mat44`` keeps its other callers; the shared arithmetic is its own
    # ``transform_point_mat44`` helper, called here rather than copied.
    chunk, lane = wp.tid()
    transform = procrustes_matrix(a, b, acc, use_reflection, use_translation, use_scale)
    if chunk == 0 and lane == 0:
        out_matrix[0] = transform
    offset, count = block_chunk_1d(a.shape[0], chunk)
    if count <= 0:
        return
    publish = out_transformed.shape[0] > 0
    local = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        moved = transform_point_mat44(a[i], transform)
        if publish:
            out_transformed[i] = moved
        w = sample_weight(weights, i)
        local += w * wp.length_sq(b[i] - moved)
    total = block_sum(local)
    if lane == 0:
        wp.atomic_add(acc, ACC_COST, total / acc[ACC_W_SUM])


# --- Iterative closest point (ICP) -----------------------------------------


@wp.func
def point_to_plane_residual(current: wp.vec3, closest: wp.vec3, normal: wp.vec3) -> wp.float32:
    """Signed point-to-plane residual ``dot(current - closest, normal)``."""
    return wp.dot(current - closest, normal)


@wp.func
def residual_valid(
    triangle_id: wp.int32, distance: wp.float32, max_distance: wp.float32
) -> wp.bool:
    """Whether a correspondence is a valid, in-range hit."""
    return triangle_id >= wp.int32(0) and distance <= max_distance


@wp.func
def target_unit_normal(normals: wp.array[wp.vec3], index: wp.int32) -> wp.vec3:
    """
    Return the target's normal at correspondence ``index``, normalized, or zero for a zero entry.

    The one reading of the normal table both the point-to-plane fit and its robust scale use, so
    the two cannot measure residuals differently: a caller's ``target_normals`` need not be unit.
    A mesh target's normals come from ``face_normals_and_areas``, which writes an exact zero vector
    for a degenerate face (CLAUDE.md section 12.4) -- a plain ``wp.normalize`` there is ``0/0``,
    and one poisoned lane's NaN would spread to a whole block through a tile reduction. Same guard,
    same zero tolerance, as ``transform.transform_normal_mat33``'s identical hazard.
    """
    return normalize_or_zero(normals[index], wp.float32(0.0))


@wp.func
def distance_threshold_weight(
    distance: wp.float32, triangle_id: wp.int32, max_distance: wp.float32
) -> wp.float32:
    """Binary correspondence mask: 1 for a valid, in-range hit, 0 otherwise."""
    # Same predicate as ``residual_valid`` -- called rather than re-derived, so the "valid
    # correspondence" rule has one definition instead of two that a future edit could desync.
    # ``distance``/``triangle_id`` are transposed relative to ``residual_valid``'s own parameter
    # order because this one's ``wp.map`` call sites already pass them that way.
    return wp.where(
        residual_valid(triangle_id, distance, max_distance), wp.float32(1.0), wp.float32(0.0)
    )


@wp.func
def icp_match(
    mesh_id: wp.uint64,
    target_points: wp.array[wp.vec3],
    cloud: wp.bool,
    q: wp.vec3,
    query_max: wp.float32,
) -> tuple[wp.vec3, wp.float32, wp.int32]:
    # One moved source point's match on an ICP target: the matched point, its distance and its
    # index (``-1`` for a miss). ``cloud`` selects the target, warp-uniformly: a point-cloud
    # target's ``neighbors.mesh_from_points`` answers the nearest vertex (unbounded, so it always
    # hits), whose position is the match; a mesh target's closest point on the surface within
    # ``query_max``, whose index is the face. The one target dispatch both ICP loops share.
    if cloud:
        index, distance = mesh_nearest_point(mesh_id, target_points, q, FLOAT32_INF_CONSTANT)
        return target_points[wp.max(index, wp.int32(0))], distance, index
    return closest_point_query(mesh_id, q, query_max)


@wp.kernel
def point_to_plane_correspondence_pass(
    mesh_id: wp.uint64,
    target_points: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    step: wp.array[wp.mat44],
    cloud: wp.bool,
    query_max: wp.float32,
    out_points: wp.array[wp.vec3],
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_index: wp.array[wp.int32],
) -> None:
    # One ``icp_point_to_plane`` iteration's whole correspondence step: the previous iteration's
    # rigid step applied to each source point and published, and its match (``icp_match``). The
    # target normal is not gathered here: the accumulation reads it at ``out_index``, and the index
    # is all a valid correspondence needs to find it.
    #
    # Differs from ``point_to_point_correspondence_pass`` by what it publishes: this loop moves
    # its points incrementally and reads the distance and index back in the accumulation, where
    # that one re-derives each point from the source and keeps only the match and a weight.
    #
    # The step is ``apply_transform_mat44``'s own ``transform_point_mat44`` on the same operands,
    # so ``out_points`` holds the bits that kernel wrote when it ran as its own launch at the tail
    # of the previous iteration; the loop now applies the last step once after it exits instead.
    tid = wp.int32(wp.tid())
    q = transform_point_mat44(points[tid], step[0])
    out_points[tid] = q
    closest, distance, index = icp_match(mesh_id, target_points, cloud, q, query_max)
    out_closest[tid] = closest
    out_distance[tid] = distance
    out_index[tid] = index


@wp.kernel
def point_to_point_correspondence_pass(
    mesh_id: wp.uint64,
    target_points: wp.array[wp.vec3],
    source: wp.array[wp.vec3],
    total: wp.array[wp.mat44],
    cloud: wp.bool,
    query_max: wp.float32,
    max_distance: wp.float32,
    out_closest: wp.array[wp.vec3],
    out_weights: wp.array[wp.float32],
) -> None:
    # ``registration.icp``'s whole correspondence step: the source point moved by the transform the
    # loop has kept so far, its match on the target (``icp_match``), and -- when ``out_weights`` is
    # not zero-length -- the binary distance gate its Procrustes fit weights by. The match, the
    # distance and the index stay in registers.
    #
    # The move is ``transform_point_mat44`` on the operands the host loop's fit applied when it
    # published the points this search then read, so the query sees the same bits.
    tid = wp.int32(wp.tid())
    q = transform_point_mat44(source[tid], total[0])
    closest, distance, index = icp_match(mesh_id, target_points, cloud, q, query_max)
    out_closest[tid] = closest
    if out_weights.shape[0] > 0:
        out_weights[tid] = distance_threshold_weight(distance, index, max_distance)


@wp.func
def continue_icp_loop(
    old_cost: wp.float64,
    cost: wp.float64,
    threshold: wp.float64,
    max_iterations: wp.int32,
    out_state: wp.array[wp.int32],
) -> None:
    # Both ICP loops' shared stopping rule, run once a round has kept a fit of cost ``cost``: count
    # the round, and continue while fewer than ``max_iterations`` have run and the cost fell from
    # ``old_cost`` by at least ``threshold``. ``old_cost`` is ``inf`` before the first kept fit,
    # which no finite threshold's test reads as convergence -- the ``float64`` test of two
    # ``float32`` values the host loops formed from their Python floats.
    iteration = out_state[LOOP_ROUND] + 1
    out_state[LOOP_ROUND] = iteration
    keep_going = not (old_cost - cost < threshold) and iteration < max_iterations
    out_state[LOOP_CONDITION] = wp.where(keep_going, wp.int32(1), wp.int32(0))


@wp.kernel
def point_to_point_round(
    threshold: wp.float64,
    max_iterations: wp.int32,
    fitted: wp.array[wp.mat44],
    out_acc: wp.array[wp.float32],
    out_total: wp.array[wp.mat44],
    out_state: wp.array[wp.int32],
) -> None:
    """
    Close one ``icp`` iteration on the device and decide whether another runs.

    ``dim=1``, the point-to-point sibling of ``point_to_plane_round``: the host loop's tail after
    its fit, so the loop body is one fixed sequence ``_device.run_device_loop`` records once.

    - **weightless** (the fit's weight sum is exactly zero -- every correspondence rejected by the
      distance gate): the fit ``fitted`` is a matrix of NaN and is *not* kept, and the loop stops.
      The host loop broke before rebinding its result, which is what this reproduces without a
      ping-pong of the fit's buffers: the fit lands in ``fitted`` and only an accepted one is
      copied into ``out_total``;
    - otherwise ``fitted`` becomes ``out_total``, its cost ``out_acc[ACC_KEPT_COST]``, and the loop
      continues while fewer than ``max_iterations`` have run and the cost fell by at least
      ``threshold``.

    ``ACC_KEPT_COST`` doubles as the previous iteration's cost. The round counter advances exactly
    when a fit is kept, so while it reads 0 there is no previous cost and the test reads ``inf``,
    the host loop's seed -- and a weightless round 0 writes ``inf`` there, the answer of a call
    that kept nothing. The moments are zeroed after they are read, by this one thread, so no memset
    precedes the next round's fit.
    """
    w_sum = out_acc[ACC_W_SUM]
    cost = out_acc[ACC_COST]
    for slot in range(PROCRUSTES_ACC_SIZE):
        out_acc[slot] = wp.float32(0.0)
    first = out_state[LOOP_ROUND] == 0
    if w_sum == wp.float32(0.0):
        if first:
            out_acc[ACC_KEPT_COST] = wp.float32(FLOAT32_INF_CONSTANT)
        out_state[LOOP_CONDITION] = 0
        return
    old_cost = wp.float64(FLOAT32_INF_CONSTANT)
    if not first:
        old_cost = wp.float64(out_acc[ACC_KEPT_COST])
    out_total[0] = fitted[0]
    out_acc[ACC_KEPT_COST] = cost
    continue_icp_loop(old_cost, wp.float64(cost), threshold, max_iterations, out_state)


@wp.func
def abs_deviation(value: wp.float32, center: wp.float32) -> wp.float32:
    """Absolute deviation ``|value - center|`` (median-absolute-deviation building block)."""
    return wp.abs(value - center)


# --- The robust scale's median absolute deviation, on the device ------------------------------
#
# ``registration._robust_scale_from_residuals`` needs the median of the in-range point-to-plane
# residuals and the median of their absolute deviations from it. Each median is a radix sort of a
# ``2n`` key buffer whose first ``n`` keys these kernels write: an out-of-range correspondence
# writes ``+inf``, which sorts last, so the in-range values are the sorted prefix and its length
# is one search, where compacting them first took a scan, a count readback and a gather. The
# medians are formed in ``float64`` exactly as the host formed them from the ``float32`` values it
# read back -- the middle value, or the mean of the two middle values -- so the scale is the same
# double; ``MAD_STATS`` is ``(center, then sigma; in-range count)``.
MAD_STATS_VALUE = wp.constant(0)
MAD_STATS_COUNT = wp.constant(1)
MAD_STATS_SIZE = 2


@wp.func
def sorted_prefix_median(values: wp.array[wp.float32], count: wp.int32) -> wp.float64:
    """Median of ``values[:count]``, sorted and non-empty, as ``numpy.median`` forms it."""
    upper = wp.float64(values[count // 2])
    if count % 2 == 1:
        return upper
    return (wp.float64(values[count // 2 - 1]) + upper) / wp.float64(2.0)


@wp.func
def in_range_residual(
    current: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    triangle_id: wp.array[wp.int32],
    distance: wp.array[wp.float32],
    max_distance: wp.float32,
    i: wp.int32,
) -> wp.float32:
    """Correspondence ``i``'s point-to-plane residual where it is in range, ``+inf`` where not."""
    # ``normals`` is the target's normal table, read at the correspondence's index through
    # ``target_unit_normal`` exactly as ``point_to_plane_tile`` reads it -- behind the range test,
    # which is what keeps a miss's ``-1`` from indexing it (a ``wp.where`` evaluates both arms).
    residual = wp.float32(FLOAT32_INF_CONSTANT)
    index = triangle_id[i]
    if residual_valid(index, distance[i], max_distance):
        residual = point_to_plane_residual(
            current[i], closest[i], target_unit_normal(normals, index)
        )
    return residual


@wp.kernel
def robust_residual_keys(
    current: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    triangle_id: wp.array[wp.int32],
    distance: wp.array[wp.float32],
    max_distance: wp.float32,
    out_keys: wp.array[wp.float32],
) -> None:
    # The medians' sort keys: ``in_range_residual``, whose ``+inf`` sorts last.
    i = wp.int32(wp.tid())
    out_keys[i] = in_range_residual(
        current, closest, normals, triangle_id, distance, max_distance, i
    )


@wp.kernel
def residual_moment(
    current: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    triangle_id: wp.array[wp.int32],
    distance: wp.array[wp.float32],
    max_distance: wp.float32,
    centered: wp.int32,
    out_sums: wp.array[wp.float64],
) -> None:
    # One pass of the in-range residuals' two-pass standard deviation, folded as the residuals are
    # formed (``in_range_residual``; the out-of-range ``+inf`` is skipped). With ``centered == 0``
    # it commits their sum and count into ``out_sums[0:2]``; with ``centered == 1`` the sum of
    # squared deviations from that mean, ``out_sums[0] / out_sums[1]``, into ``out_sums[2]``. The
    # mean is read on the device, so the two passes need no readback between them.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(current.shape[0], block)
    mean = wp.float64(0.0)
    if centered != 0:
        mean = out_sums[0] / out_sums[1]
    total = wp.float64(0.0)
    kept = wp.float64(0.0)
    for k in range(lane, count, wp.block_dim()):
        r = in_range_residual(
            current, closest, normals, triangle_id, distance, max_distance, offset + k
        )
        if r < wp.float32(FLOAT32_INF_CONSTANT):
            d = wp.float64(r) - mean
            total += wp.where(centered != 0, d * d, d)
            kept += wp.float64(1.0)
    if centered != 0:
        commit_block_total(lane, total, out_sums, 2)
    else:
        commit_sum_and_count(lane, total, kept, out_sums)


@wp.kernel
def robust_residual_center(
    sorted_residuals: wp.array[wp.float32], n: wp.int32, out_stats: wp.array[wp.float64]
) -> None:
    # ``dim=1``: the in-range count, and the median of the in-range residuals.
    count = binary_search_index_left(sorted_residuals[:n], FLOAT32_INF_CONSTANT)
    out_stats[MAD_STATS_COUNT] = wp.float64(count)
    if count > 0:
        out_stats[MAD_STATS_VALUE] = sorted_prefix_median(sorted_residuals, count)


@wp.kernel
def robust_deviation_keys(stats: wp.array[wp.float64], out_keys: wp.array[wp.float32]) -> None:
    # In place over the sorted residuals, each thread reading and rewriting its own slot: each
    # in-range residual's absolute deviation from the median, which ``abs_deviation`` takes as the
    # ``float32`` the host used to marshal; ``+inf`` past the in-range prefix. The next sort is the
    # only reader, which is why the buffer wears the ``out_`` prefix although it is also read.
    i = wp.int32(wp.tid())
    count = wp.int32(stats[MAD_STATS_COUNT])
    out_keys[i] = wp.where(
        i < count,
        abs_deviation(out_keys[i], wp.float32(stats[MAD_STATS_VALUE])),
        FLOAT32_INF_CONSTANT,
    )


@wp.kernel
def robust_sigma(sorted_deviations: wp.array[wp.float32], out_stats: wp.array[wp.float64]) -> None:
    # ``dim=1``: ``1.4826 * MAD``, the normal-consistent scale, over the value slot.
    count = wp.int32(out_stats[MAD_STATS_COUNT])
    if count > 0:
        mad = sorted_prefix_median(sorted_deviations, count)
        out_stats[MAD_STATS_VALUE] = wp.float64(1.4826) * mad


@wp.func
def robust_weight_and_loss(
    residual: wp.float32, scale: wp.float32, kind: wp.int32
) -> tuple[wp.float32, wp.float32]:
    """
    IRLS weight of a residual under an M-estimator, and its term of the objective.

    ``kind``: 0 = none (unit weight), 1 = Huber (``k = scale``),
    2 = Tukey biweight (``c = scale``). A non-positive ``scale`` yields unit weight. One dispatch
    for both halves, so the weight a residual is fitted with and the loss the loop stops on
    cannot disagree about which estimator is in force.

    The loss is ``weight * residual^2`` for no kernel and for Huber, where it grows with
    ``|residual|`` like the loss itself, and twice the Tukey biweight loss ``rho`` for Tukey:
    ``c^2 / 3 * (1 - (1 - u^2)^3)`` with ``u = |r| / c``, saturating at ``c^2 / 3`` once
    ``|r| >= c``. The factor two makes every kind read ``r^2`` for a small residual.

    **Tukey cannot use ``weight * residual^2``**, because the biweight redescends: a residual beyond
    ``c`` carries zero weight, so its term is zero, and as the fit pulls it inside the kernel the
    term *rises*. Registering from a start whose residuals mostly exceed ``c`` -- an explicit
    ``robust_scale`` below the initial misalignment, which is the normal way to pass one -- makes
    that sum climb for several iterations while the pose improves, and a decrease test reads the
    first climb as convergence. ``rho`` is nondecreasing in ``|r|`` and saturates rather than
    vanishing, so it falls as the fit improves. Measured on a 5-degree start with ``c`` at 1 % of
    the bounding-box diagonal: the weighted sum rose for five iterations and the loop stopped after
    two, 4.8 degrees from the answer; the loss falls monotonically and the loop runs to it.
    """
    if kind == wp.int32(0) or scale <= wp.float32(0.0):
        return wp.float32(1.0), residual * residual
    r = wp.abs(residual)
    if kind == wp.int32(1):
        # Huber
        w = wp.float32(1.0)
        if r > scale:
            w = scale / r
        return w, w * residual * residual
    # Tukey biweight: ``t * t * t`` is ``w * t``, the same product in the same order.
    if r >= scale:
        return wp.float32(0.0), scale * scale / wp.float32(3.0)
    u = r / scale
    t = wp.float32(1.0) - u * u
    w = t * t
    return w, scale * scale / wp.float32(3.0) * (wp.float32(1.0) - w * t)


@wp.func
def point_to_plane_tile(
    source: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: wp.float32,
    robust_kind: wp.int32,
    robust_scale: wp.float32,
    offset: wp.int32,
    remaining: wp.int32,
    lane: wp.int32,
    stride: wp.int32,
) -> tuple[wp.spatial_matrix, wp.spatial_vector, wp.float32, wp.float32]:
    # ``lane`` / ``stride`` rather than ``wp.block_dim()`` because this is a ``@wp.func``: the
    # caller is the kernel that knows its own launch shape, and passing the stride in keeps this
    # usable from a serial caller too.
    #
    # ``target`` is the per-correspondence matched point, and ``normals`` the target's own normal
    # table -- per face for a mesh, per vertex for a cloud -- read at the correspondence's index,
    # which a valid correspondence has non-negative.
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
    jtj = wp.spatial_matrix(wp.float32(0.0))
    jtr = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    cost = wp.float32(0.0)
    # Sum of the robust weight itself, not the "in-range hit" count ``residual_valid`` already
    # gates on: a Tukey kernel can drive *every* in-range correspondence's weight to exactly zero
    # (``r >= scale``, e.g. a ``robust_scale`` too tight for the residual distribution, or a
    # previous iteration's step overshooting far past what a first-iteration-derived scale
    # anticipated), which leaves ``jtj``/``jtr``/``cost`` all zero without a single rejection ever
    # firing. The caller uses this to tell that case apart from genuine convergence.
    weight_sum = wp.float32(0.0)
    for k in range(lane, count, stride):
        idx = offset + k
        if not residual_valid(triangle_id[idx], distance[idx], max_distance):
            continue
        nrm = target_unit_normal(normals, triangle_id[idx])
        x = source[idx]
        r = point_to_plane_residual(x, target[idx], nrm)
        w, loss = robust_weight_and_loss(r, robust_scale, robust_kind)
        # Jacobian of the point-to-plane residual: [x x n ; n]
        j = wp.spatial_vector(wp.cross(x, nrm), nrm)
        jtj += w * wp.outer(j, j)
        jtr += (w * r) * j
        cost += loss
        weight_sum += w
    return jtj, jtr, cost, weight_sum


@wp.kernel
def accumulate_point_to_plane(
    source: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: wp.float32,
    robust_kind: wp.int32,
    robust_scale: wp.float32,
    out_acc: wp.array[wp.float32],
) -> None:
    # Launched ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: one block per
    # ``ITEMS_PER_BLOCK_1D`` correspondences, lanes striding that block's own chunk, and one packed
    # block reduction of all 44 moment slots (the normal matrix, the right-hand side, the two
    # scalars) committing one atomic set per block.
    #
    # It was every lane walking a ``TILE_1D`` chunk with lane 0 publishing, which put one add per
    # block on each of 44 hot addresses at ``n / TILE_1D`` blocks. Same finding as
    # ``accumulate_procrustes_moments``: the redundant lanes were nearly free and the atomic
    # contention was the cost, worth an order of magnitude at a million correspondences.
    #
    # One packed fold rather than three (matrix, vector, scalar pair): ``block_sum`` of a vector is
    # componentwise the per-quantity sums, bit for bit, so this is two barriers per block fewer and
    # the three accumulators are one buffer. The tree is the *more* accurate arm on every
    # component, which matters here because the normal matrix is what ``point_to_plane_round``
    # factorizes.
    i, lane = wp.tid()
    offset, remaining = block_chunk_1d(source.shape[0], i)
    if remaining <= 0:
        return

    tile_jtj, tile_jtr, tile_cost, tile_weight_sum = point_to_plane_tile(
        source,
        target,
        normals,
        distance,
        triangle_id,
        max_distance,
        robust_kind,
        robust_scale,
        offset,
        remaining,
        lane,
        wp.block_dim(),
    )

    # The default constructor, not a zero-fill: every slot is written just below.
    packed = wp.vector(length=ICP_MOMENT_SLOTS, dtype=wp.float32)
    for r in range(6):
        packed[ICP_JTR + r] = tile_jtr[r]
        for c in range(6):
            packed[ICP_JTJ + r * 6 + c] = tile_jtj[r, c]
    packed[ICP_COST] = tile_cost
    packed[ICP_WEIGHT_SUM] = tile_weight_sum
    commit_block_sum(lane, packed, out_acc, 0)


@wp.func
def solve_spd6(a: wp.spatial_matrix, b: wp.spatial_vector) -> wp.spatial_vector:
    """
    Solve the SPD 6x6 system ``a x = b`` via Cholesky (``a = L L^T``).

    Hand-written rather than routed through ``wp.dense_chol`` / ``wp.dense_subs`` /
    ``wp.dense_solve``, which look like exactly this function and are not usable here. Three
    independent reasons, any one sufficient: they are ``hidden=True`` with ``doc="WIP"``, so they
    are undocumented and unannounced in the installed Warp; they take ``wp.array[wp.float32]``
    rather than a register value, so adopting them means per-thread global scratch for ``A``,
    ``L``, ``b`` and ``x``, which is the shape measured at a 2x loss against registers; and they
    are ``float32``-only, so ``linalg``'s ``float64`` systems could not use them either.

    The two zero ``wp.spatial_vector``s below are written out longhand although the
    ``wp.spatial_matrix`` one line up is the broadcast ``wp.spatial_matrix(wp.float32(0.0))``. That
    asymmetry is Warp's, not a style slip: a one-argument ``wp.spatial_vector(x)`` binds ``x`` to
    the *templated* ``vec_t``'s ``dtype`` parameter and fails to parse -- *"Remove the extraneous
    ``dtype`` parameter when calling the templated version of ``wp.vec_t()``"*, still true in Warp
    1.18. There is no broadcast-fill spelling for it, so do not collapse these.
    """
    lower = wp.spatial_matrix(wp.float32(0.0))
    for j in range(6):
        s = a[j, j]
        for k in range(j):
            s -= lower[j, k] * lower[j, k]
        if s < wp.float32(1e-20):
            s = wp.float32(1e-20)
        ljj = wp.sqrt(s)
        lower[j, j] = ljj
        for i in range(j + 1, 6):
            v = a[i, j]
            for k in range(j):
                v -= lower[i, k] * lower[j, k]
            lower[i, j] = v / ljj
    # Forward substitution: L y = b
    y = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    for i in range(6):
        v = b[i]
        for k in range(i):
            v -= lower[i, k] * y[k]
        y[i] = v / lower[i, i]
    # Back substitution: L^T x = y
    x = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    for ii in range(6):
        i = 5 - ii
        v = y[i]
        for k in range(i + 1, 6):
            v -= lower[k, i] * x[k]
        x[i] = v / lower[i, i]
    return x


@wp.kernel
def point_to_plane_round(
    damping: wp.float32,
    threshold: wp.float64,
    max_iterations: wp.int32,
    out_acc: wp.array[wp.float32],
    out_total: wp.array[wp.mat44],
    out_step: wp.array[wp.mat44],
    out_state: wp.array[wp.int32],
) -> None:
    """
    Close one ``icp_point_to_plane`` iteration on the device and decide whether another runs.

    ``dim=1``. The host loop's whole per-iteration tail -- the weightless test, the 6x6 solve, the
    composition into the running transform and the convergence test -- so that the loop body is a
    fixed sequence ``_device.run_device_loop`` can record once and replay, with no readback:

    - **weightless**: no correspondence carried weight, so nothing is solved; the step is set to
      the identity and the loop stops, and the host reports ``inf``;
    - otherwise the damped system is solved, ``out_step`` written, ``out_total`` composed in place
      (``step * total``, the product and order the host ping-pong formed) and the accumulated
      moments zeroed for the next iteration -- after they are read, by this one thread, so no
      memset;
    - the loop continues by ``continue_icp_loop``, the point-to-point loop's own rule, against the
      previous kept cost in ``ICP_OLD_COST`` -- ``inf`` while the round counter is 0, i.e. before
      the first kept fit, so the first round never reads as converged. The counter advances
      exactly when a fit is kept, so a zero-initialized slot needs no ``inf`` seed.

    Every ``out_`` argument but ``out_step`` is loop state, read *and* rewritten every round; they
    wear the prefix as ``kernels/array.loop_advance``'s ``out_state`` does.
    """
    if out_acc[ICP_WEIGHT_SUM] <= wp.float32(0.0):
        # The step becomes the identity so the host's closing correspondence pass, which applies
        # ``out_step`` like every other search, leaves the pose where the weightless search found
        # it; its accumulation then sums zero weight again and the host reports ``inf``. That is
        # what lets the host learn "weightless" from the one readback it takes anyway.
        out_step[0] = wp.identity(n=4, dtype=wp.float32)
        out_state[LOOP_CONDITION] = 0
        return
    cost = out_acc[ICP_COST]
    a = wp.spatial_matrix(wp.float32(0.0))
    # Longhand for the reason ``solve_spd6`` gives: ``wp.spatial_vector`` has no broadcast fill.
    b = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    for r in range(6):
        b[r] = out_acc[ICP_JTR + r]
        for c in range(6):
            a[r, c] = out_acc[ICP_JTJ + r * 6 + c]

    # Levenberg-style diagonal damping, scaled by the mean diagonal magnitude,
    # keeps the system positive-definite for planar / rank-deficient targets.
    reg = damping * wp.trace(a) / wp.float32(6.0) + wp.float32(1e-12)
    for i in range(6):
        a[i, i] += reg

    delta = -solve_spd6(a, b)
    omega = wp.spatial_top(delta)
    tvec = wp.spatial_bottom(delta)

    angle = wp.length(omega)
    rot = wp.identity(n=3, dtype=wp.float32)
    if angle > wp.float32(1e-12):
        rot = wp.quat_to_matrix(wp.quat_from_axis_angle(wp.normalize(omega), angle))

    step = make_affine44(rot, tvec)
    out_step[0] = step
    out_total[0] = wp.mul(step, out_total[0])

    for slot in range(ICP_MOMENT_SLOTS):
        out_acc[slot] = wp.float32(0.0)

    old_cost = wp.float64(FLOAT32_INF_CONSTANT)
    if out_state[LOOP_ROUND] > 0:
        old_cost = wp.float64(out_acc[ICP_OLD_COST])
    out_acc[ICP_OLD_COST] = cost
    continue_icp_loop(old_cost, wp.float64(cost), threshold, max_iterations, out_state)
