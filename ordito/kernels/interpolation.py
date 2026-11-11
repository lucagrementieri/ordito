from typing import Any

import warp as wp

from ordito.kernels.array import (
    OverloadTable,
    declare_map_signatures,
    divide_if_positive,
    map_probe,
    ordered_item,
    trilinear_cell,
    trilinear_corner,
    trilinear_weight,
)
from ordito.kernels.neighbors import ball_cell_box, ball_cell_probe, in_ball
from ordito.kernels.proximity import closest_point_query
from ordito.kernels.reduce import peer_counted_add
from ordito.kernels.triangles import face_vertices, point_barycentric

wp.set_module_options({"enable_backward": False})


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    x0, x1, x2 = face_vertices(vertex_values, faces, f)
    out_face_values[f] = (x0 + x1 + x2) / wp.float32(3.0)


@wp.func
def mean_rounded(total: wp.float64, count: wp.float32) -> wp.float32:
    # A vertex's exact ``float64`` sum over its count, rounded once: ``average_onto_vertices``
    # under the deterministic setting. Unguarded, so a zero count is ``nan`` as in the
    # arrival-order ``wp.div``.
    return wp.float32(total / wp.float64(count))


@wp.func
def mean_rounded_if_counted(total: wp.float64, count: wp.float32) -> wp.float32:
    # ``mean_rounded`` guarded as ``array.divide_if_positive``: a zero count keeps the zero sum,
    # ``average_from_edges_onto_vertices``' convention.
    return wp.float32(divide_if_positive(total, wp.float64(count)))


# ``transfer_onto_vertices`` takes its targets in ``_common.search_order``'s Hilbert order
# (2026-11-11, CUDA, a dragon source, new/old, under a shared GPU, three interleaved rounds):
# lucy's 14 M vertices scrambled 0.18-0.21x (235-283 -> 48-51 ms), where the order is always taken;
# dragon's own jittered vertices, at 437 k between the gates, scrambled 0.93-1.03x and in scan
# order 1.05-1.10x (the sampled test's cost), so on CUDA it takes an order only from the
# always-gate. The CPU keeps the sampled test: 437 k scrambled 0.49-0.52x, scan order at par.
# Answers bit-identical.
TRANSFER_SAMPLED_ORDER_ON_CUDA = False


@wp.kernel
def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[Any],
    mesh_id: wp.uint64,
    target_vertices: wp.array[wp.vec3],
    max_dist: wp.float32,
    order: wp.array[wp.int32],
    out_values: wp.array[Any],
    out_distance: wp.array[wp.float32],
) -> None:
    # Barycentric resample of a source per-vertex field at each target vertex's closest point, the
    # query fused in through ``proximity.closest_point_query`` (the same helper
    # ``closest_point_on_mesh`` launches), so no closest-point or face buffer is materialized.
    #
    # ``face_id < 0`` is the ``max_dist`` miss case. The value slot keeps whatever the wrapper
    # pre-filled (so a caller who narrowed the search still gets a deterministic buffer), and the
    # distance is promoted to ``inf``: ``wp.mesh_query_point_no_sign`` reports ``max_dist`` there,
    # which is indistinguishable from a genuine hit at exactly that range.
    #
    # The wrapper's mesh is Warp's default build (an LBVH on CUDA, leaf size 4), which is about
    # half the call from bunny to lucy. Declined, build plus query against it on CUDA: leaf sizes
    # 1, 2 and 8 0.98-1.05x on sphere_med, bunny and dragon; the SAH, median and cuBQL builders
    # query 0.4-0.9x as long but build 10-300x as long (dragon: 0.70 ms LBVH, 16-400 ms the
    # others). A zero-filled value buffer in place of an empty one written on a miss costs ~3 us
    # of a ~190 us small call (the mesh build ~150 us): not taken.
    #
    # Thread ``t`` answers target vertex ``order[t]`` (``_common.search_order``, the order
    # ``proximity.closest_point_on_mesh`` takes; index order for ``None``).
    i = ordered_item(order, wp.int32(wp.tid()))
    closest, distance, f = closest_point_query(mesh_id, target_vertices[i], max_dist)
    if f < 0:
        out_distance[i] = wp.float32(wp.INF)
        return
    out_distance[i] = distance
    #
    # ``point_barycentric``'s conditioning is load-bearing here, not incidental: a source face thin
    # enough for the Gram-determinant form to cancel to zero in float32 would write ``nan`` into the
    # field while ``out_distance[i]`` stayed finite, so the confidence measure this function returns
    # would report a healthy hit on a poisoned value. Slivers are ordinary in a decimated or
    # reconstructed source mesh, which is exactly the input this transfer exists for -- that is the
    # defect that got the Gram form removed from the package outright.
    v0, v1, v2 = face_vertices(source_vertices, source_faces, f)
    bary = point_barycentric(v0, v1, v2, closest)
    a0, a1, a2 = face_vertices(source_values, source_faces, f)
    out_values[i] = a0 * bary[0] + a1 * bary[1] + a2 * bary[2]


@wp.kernel
def apply_transfer_operator(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    source_values: wp.array[Any],
    out_values: wp.array[Any],
) -> None:
    # One CSR row per output element: the field's value there is the weighted sum of the source
    # values the row references. Written rather than delegated to ``warp.sparse.bsr_mv``, which
    # requires the vector's dtype to match the matrix's 1x1 ``float32`` block, so it can move a
    # scalar field and nothing else -- where an attribute is as often a ``wp.vec3`` colour or a
    # ``wp.vec2`` UV, and those are exactly what a topology edit drops today.
    #
    # The accumulator is seeded as ``source_values[0] - source_values[0]``, which is how a generic
    # ``Any`` kernel spells "the additive identity of this dtype": there is no ``dtype(0)``
    # constructor to reach for, and ``x * wp.float32(0.0)`` would not parse for a dtype whose scalar
    # is not float32 (Warp requires both operands' scalars to match). It also gives an empty row --
    # an output element no source reaches -- the right answer rather than uninitialized memory.
    #
    # The weights being ``float32`` is what bounds the dtype set: the product below needs the
    # field's own scalar to be float32 too, so ``wp.float32`` / ``wp.vec2`` / ``wp.vec3`` work and a
    # ``float64`` field does not. That is not a gap -- these operators are assembled from float32
    # vertex data, so a float64 field carried through one would advertise precision the weights do
    # not have.
    row = wp.int32(wp.tid())
    total = source_values[0] - source_values[0]
    for k in range(offsets[row], offsets[row + 1]):
        total = total + source_values[columns[k]] * weights[k]
    out_values[row] = total


@wp.kernel
def gather_or_zero(
    values: wp.array[Any], index: wp.array[wp.int32], zero: Any, out_values: wp.array[Any]
) -> None:
    # ``values[index[i]]``, or the dtype's zero where ``index[i]`` is negative: an output element a
    # topology edit created from nothing. ``zero`` is an argument because a generic kernel has no
    # spelling of a zero that also covers ``wp.bool``.
    i = wp.int32(wp.tid())
    source = index[i]
    if source >= 0:
        out_values[i] = values[source]
    else:
        out_values[i] = zero


@wp.func
def first_row_entry(sources: wp.array2d[wp.int32], row: wp.int32, slot: wp.int32) -> wp.bool:
    # Whether slot ``slot`` of a padded provenance row names a source no earlier slot of the row
    # names: the slot an entry's merged weight is written from. A negative source is padding.
    source = sources[row, slot]
    first = source >= 0
    for earlier in range(slot):
        if sources[row, earlier] == source:
            first = False
    return first


@wp.func
def merged_row_weight(
    sources: wp.array2d[wp.int32], weights: wp.array2d[wp.float32], row: wp.int32, slot: wp.int32
) -> wp.float32:
    # The summed weight of every slot of ``row`` naming the same source as ``slot``, in slot order.
    total = wp.float32(0.0)
    for other in range(sources.shape[1]):
        if sources[row, other] == sources[row, slot]:
            total = total + weights[row, other]
    return total


@wp.kernel
def count_provenance_rows(
    sources: wp.array2d[wp.int32], weights: wp.array2d[wp.float32], out_counts: wp.array[wp.int32]
) -> None:
    # The distinct sources of a padded row whose merged weight is not zero, written into the
    # ``offsets[1:]`` the in-place scan turns into row bounds.
    row = wp.int32(wp.tid())
    count = wp.int32(0)
    for slot in range(sources.shape[1]):
        if first_row_entry(sources, row, slot):
            if merged_row_weight(sources, weights, row, slot) != wp.float32(0.0):
                count += 1
    out_counts[row + 1] = count


@wp.kernel
def write_provenance_rows(
    sources: wp.array2d[wp.int32],
    weights: wp.array2d[wp.float32],
    offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_weights: wp.array[wp.float32],
) -> None:
    # A padded row as one CSR row: duplicate sources merged, zero weights dropped, columns
    # ascending (an entry's place is the number of kept entries with a smaller source, so no
    # in-thread sort).
    row = wp.int32(wp.tid())
    start = offsets[row]
    for slot in range(sources.shape[1]):
        if first_row_entry(sources, row, slot):
            weight = merged_row_weight(sources, weights, row, slot)
            if weight != wp.float32(0.0):
                source = sources[row, slot]
                place = start
                for other in range(sources.shape[1]):
                    if first_row_entry(sources, row, other) and sources[row, other] < source:
                        if merged_row_weight(sources, weights, row, other) != wp.float32(0.0):
                            place += 1
                out_columns[place] = source
                out_weights[place] = weight


@wp.kernel
def edge_point_rows(
    n_vertices: wp.int32,
    edges: wp.array2d[wp.int32],
    t: wp.array[wp.float32],
    out_sources: wp.array2d[wp.int32],
    out_weights: wp.array2d[wp.float32],
) -> None:
    # The padded rows of a mesh whose vertices are the ``n_vertices`` originals followed by one
    # point per edge row at ``(1 - t) * a + t * b``: a copy each, then two weights each. An empty
    # ``t`` places every point at its edge's midpoint.
    row = wp.int32(wp.tid())
    if row < n_vertices:
        out_sources[row, 0] = row
        out_sources[row, 1] = -1
        out_weights[row, 0] = wp.float32(1.0)
        out_weights[row, 1] = wp.float32(0.0)
        return
    e = row - n_vertices
    along = wp.float32(0.5)
    if t.shape[0] > 0:
        along = t[e]
    out_sources[row, 0] = edges[e, 0]
    out_sources[row, 1] = edges[e, 1]
    out_weights[row, 0] = wp.float32(1.0) - along
    out_weights[row, 1] = along


@wp.kernel
def count_cluster_members(cluster: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # Each cluster's member count into ``offsets[1:]``; an input in no cluster (``-1``) counts
    # nowhere. Integer atomics, so the counts are exact in any order.
    i = wp.int32(wp.tid())
    target = cluster[i]
    if target >= 0:
        peer_counted_add(out_counts, target + 1)


@wp.kernel
def cluster_keys_and_identity(
    cluster: wp.array[wp.int32],
    n_clusters: wp.int32,
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # Sort keys putting an input in no cluster (``-1``) after every cluster, and the identity
    # payload, into the leading halves of the sort's double-width buffers.
    i = wp.int32(wp.tid())
    target = cluster[i]
    out_keys[i] = wp.where(target >= 0, target, n_clusters)
    out_order[i] = i


@wp.kernel
def write_cluster_rows(
    sorted_clusters: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    n_clusters: wp.int32,
    out_weights: wp.array[wp.float32],
) -> None:
    # Over the members stably sorted by cluster, which are already the CSR columns (ascending in a
    # row): each entry weighs one over its cluster's size. The trailing entries of unclustered
    # inputs are not read.
    k = wp.int32(wp.tid())
    target = sorted_clusters[k]
    if target >= n_clusters:
        return
    size = offsets[target + 1] - offsets[target]
    out_weights[k] = wp.float32(1.0) / wp.float32(size)


@wp.kernel
def row_representatives(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    out_representative: wp.array[wp.int32],
) -> None:
    # Each CSR row's column of largest weight, the smaller column on a tie (whatever the row's
    # column order), or -1 for an empty row: where an attribute that cannot be averaged is copied
    # from.
    row = wp.int32(wp.tid())
    best = wp.int32(-1)
    best_weight = wp.float32(0.0)
    for k in range(offsets[row], offsets[row + 1]):
        column = columns[k]
        weight = weights[k]
        if best < 0 or weight > best_weight or (weight == best_weight and column < best):
            best = column
            best_weight = weight
    out_representative[row] = best


@wp.kernel
def count_gather_rows(index: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # A gather map as a CSR row each: one entry where the source exists, none where it is -1.
    i = wp.int32(wp.tid())
    out_counts[i + 1] = wp.where(index[i] >= 0, 1, 0)


@wp.kernel
def write_gather_rows(
    index: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_weights: wp.array[wp.float32],
) -> None:
    i = wp.int32(wp.tid())
    if index[i] >= 0:
        out_columns[offsets[i]] = index[i]
        out_weights[offsets[i]] = wp.float32(1.0)


@wp.kernel
def count_composed_rows(
    first_offsets: wp.array[wp.int32],
    second_offsets: wp.array[wp.int32],
    second_columns: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The product ``second @ first`` of two CSR operators, row ``row`` of ``second`` expanded
    # through ``first``: one triplet per pair of entries, duplicates summed by the assembly.
    row = wp.int32(wp.tid())
    count = wp.int32(0)
    for k in range(second_offsets[row], second_offsets[row + 1]):
        middle = second_columns[k]
        count += first_offsets[middle + 1] - first_offsets[middle]
    out_counts[row + 1] = count


@wp.kernel
def write_composed_rows(
    first_offsets: wp.array[wp.int32],
    first_columns: wp.array[wp.int32],
    first_weights: wp.array[wp.float32],
    second_offsets: wp.array[wp.int32],
    second_columns: wp.array[wp.int32],
    second_weights: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_weights: wp.array[wp.float32],
) -> None:
    row = wp.int32(wp.tid())
    slot = offsets[row]
    for k in range(second_offsets[row], second_offsets[row + 1]):
        middle = second_columns[k]
        for j in range(first_offsets[middle], first_offsets[middle + 1]):
            out_rows[slot] = row
            out_columns[slot] = first_columns[j]
            out_weights[slot] = second_weights[k] * first_weights[j]
            slot += 1


@wp.kernel
def interpolate_barycentric(
    faces: wp.array[wp.int32],
    face_index: wp.array[wp.int32],
    barycentric: wp.array[wp.vec3],
    vertex_values: wp.array[Any],
    out_values: wp.array[Any],
) -> None:
    # A per-vertex field at a point given by its face and barycentric coordinates: the three
    # corner values weighted in corner order. A negative face is a point on no face (a missed
    # query) and reads the dtype's zero, spelled as ``apply_transfer_operator`` spells it.
    i = wp.int32(wp.tid())
    f = face_index[i]
    if f < 0:
        out_values[i] = vertex_values[0] - vertex_values[0]
        return
    x0, x1, x2 = face_vertices(vertex_values, faces, f)
    w = barycentric[i]
    out_values[i] = x0 * w[0] + x1 * w[1] + x2 * w[2]


@wp.func
def gaussian_kernel_weight(distance: wp.float32, inverse_scale: wp.float32) -> wp.float32:
    # ``exp(-(sharpness * d / radius) ** 2)``, VTK's ``vtkGaussianKernel``, with the caller having
    # folded ``sharpness / radius`` into one reciprocal length. An unused neighbour slot arrives at
    # distance ``inf`` and weighs exactly 0.
    scaled = distance * inverse_scale
    return wp.exp(-scaled * scaled)


@wp.func
def accumulate_gaussian_neighbor(
    source_values: wp.array[Any],
    index: wp.int32,
    distance: wp.float32,
    inverse_scale: wp.float32,
    accumulated: Any,
    total: wp.float32,
    coincident: wp.int32,
) -> tuple[Any, wp.float32, wp.int32]:
    # Fold one neighbour into a query's Gaussian-weighted mean. Both interpolation kernels call it,
    # the one reading a neighbour CSR and the one walking the hash grid itself, so the two agree
    # to the bit whenever they visit the same neighbours in the same order.
    if distance <= wp.float32(0.0):
        coincident = index
    weight = gaussian_kernel_weight(distance, inverse_scale)
    return accumulated + source_values[index] * weight, total + weight, coincident


@wp.func
def write_gaussian_mean(
    source_values: wp.array[Any],
    q: wp.int32,
    accumulated: Any,
    total: wp.float32,
    coincident: wp.int32,
    out_values: wp.array[Any],
) -> None:
    # VTK returns the source's own value where a query sits on a data point, rather than blending
    # its neighbours in; the interpolant is then exact at the data. A query with no neighbours, or
    # whose every weight underflowed, is left at the null value the wrapper pre-filled -- so the
    # miss case needs no per-dtype null argument here. Writing the null here instead would trade
    # the fill (~3 us more than an empty allocation, of a ~50 us small call whose hash-grid build
    # is ~28 us) for a launch argument (~1 us) on every kernel: declined.
    if coincident >= 0:
        out_values[q] = source_values[coincident]
    elif total > wp.float32(0.0):
        out_values[q] = accumulated / total


@wp.kernel
def interpolate_from_points_nearest(
    source_values: wp.array[Any],
    neighbor_indices: wp.array[wp.int32],
    neighbor_distances: wp.array[wp.float32],
    k: wp.int32,
    inverse_scale: wp.float32,
    out_values: wp.array[Any],
) -> None:
    # Gaussian-weighted mean of one query's ``k`` nearest sources, over the flattened ``(m, k)``
    # rows ``query_nearest`` returns; a slot no neighbour was found for carries index -1 and is
    # skipped. The radius footprint walks the grid in ``interpolate_from_points_in_ball`` instead,
    # which differs from this only in where the neighbours come from.
    q = wp.int32(wp.tid())
    # A zero of the field's own dtype, which a generic kernel cannot spell any other way.
    accumulated = source_values[0] * wp.float32(0.0)
    total = wp.float32(0.0)
    coincident = wp.int32(-1)
    for slot in range(q * k, (q + 1) * k):
        index = neighbor_indices[slot]
        if index < 0:
            continue
        accumulated, total, coincident = accumulate_gaussian_neighbor(
            source_values,
            index,
            neighbor_distances[slot],
            inverse_scale,
            accumulated,
            total,
            coincident,
        )
    write_gaussian_mean(source_values, q, accumulated, total, coincident, out_values)


@wp.kernel
def interpolate_from_points_in_ball(
    source_points: wp.array[wp.vec3],
    source_values: wp.array[Any],
    query_points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    inverse_scale: wp.float32,
    out_values: wp.array[Any],
) -> None:
    # The Gaussian-weighted mean over every source within ``radius``, reduced during the hash-grid
    # walk instead of over a ball-query CSR. The walk, the predicate (``in_ball``) and the distance
    # are ``kernels.neighbors.ball_collect``'s hash-grid branch, so a query folds exactly the
    # neighbours, in exactly the order, that CSR would have listed -- with no count pass, scan,
    # readback or two neighbour-sized buffers in between. What still differs is the body: the
    # collect writes each neighbour out, this folds it in. ``interpolate_from_points_in_ball_tiled``
    # is the same mean with the ball's cells walked by a block's lanes, for few queries.
    q = wp.int32(wp.tid())
    p = query_points[q]
    accumulated = source_values[0] * wp.float32(0.0)
    total = wp.float32(0.0)
    coincident = wp.int32(-1)
    j = wp.int32(0)
    query = wp.hash_grid_query(grid_id, p, radius)
    while wp.hash_grid_query_next(query, j):
        offset = source_points[j] - p
        if in_ball(offset, radius):
            accumulated, total, coincident = accumulate_gaussian_neighbor(
                source_values, j, wp.length(offset), inverse_scale, accumulated, total, coincident
            )
    write_gaussian_mean(source_values, q, accumulated, total, coincident, out_values)


@wp.kernel(enable_backward=False, module="unique")
def interpolate_from_points_in_ball_tiled(
    source_points: wp.array[wp.vec3],
    source_values: wp.array[Any],
    query_points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    inverse_scale: wp.float32,
    cell_width: wp.float32,
    cell_width_inv: wp.float32,
    resolution: wp.int32,
    out_values: wp.array[Any],
) -> None:
    # ``interpolate_from_points_in_ball`` a block a query, for few queries: each lane folds the
    # sources of its share of the ball's hash-grid cells (``neighbors.ball_cell_box`` /
    # ``ball_cell_probe``) with ``accumulate_gaussian_neighbor``, and ``wp.tile_sum`` adds the
    # lanes' weighted values and weights in a fixed tree, so the mean is the one-thread walk's up
    # to ``float32`` summation order and the same on every run. Of several sources sitting on the
    # query, each lane keeps the last it meets and the block the highest index (``wp.tile_max``),
    # where the one-thread walk keeps the last it meets; one such source reads the same either
    # way. Its own module, launched at ``neighbors.TILED_BALL_WALK_BLOCK_DIM`` lanes.
    q, lane = wp.tid()
    p = query_points[q]
    accumulated = source_values[0] * wp.float32(0.0)
    total = wp.float32(0.0)
    coincident = wp.int32(-1)
    first, counts = ball_cell_box(p, radius, cell_width_inv, resolution)
    for c in range(lane, counts[0] * counts[1] * counts[2], wp.block_dim()):
        probe = ball_cell_probe(first, counts, c, cell_width)
        j = wp.int32(0)
        query = wp.hash_grid_query(grid_id, probe, cell_width * wp.float32(0.25))
        while wp.hash_grid_query_next(query, j):
            offset = source_points[j] - p
            if in_ball(offset, radius):
                accumulated, total, coincident = accumulate_gaussian_neighbor(
                    source_values,
                    j,
                    wp.length(offset),
                    inverse_scale,
                    accumulated,
                    total,
                    coincident,
                )
    summed = wp.tile_sum(wp.tile(accumulated, preserve_type=True))[0]
    weight = wp.tile_sum(wp.tile(total))[0]
    nearest = wp.tile_max(wp.tile(coincident))[0]
    if lane == 0:
        write_gaussian_mean(source_values, q, summed, weight, nearest, out_values)


@wp.kernel(enable_backward=False)
def sample_grid_trilinear(
    field: wp.array3d[Any],
    lower: wp.vec3,
    inverse_spacing: wp.vec3,
    points: wp.array[wp.vec3],
    out_values: wp.array[Any],
) -> None:
    # Read a dense lattice at one arbitrary position, as the trilinearly weighted sum of the eight
    # corners around it -- the transpose of ``kernels/scatter.splat_grid_trilinear``. The wrapper's
    # ``Notes`` records which round trips through the pair are exact; averaging makes the two
    # adjoint rather than inverse.
    #
    # Summed rather than nested as ``wp.lerp``. ``kernels/reconstruction.poisson_sample_grid`` is
    # the lerp spelling over a flat ``res**3`` float32 buffer; the two are the same function and
    # differ only in float32 summation order, and they are deliberately kept apart because that
    # one's arithmetic is what the Poisson iso-value was measured against. All three share
    # ``trilinear_cell``.
    #
    # A position outside the lattice reads the nearest cell's stencil rather than a null value,
    # because ``trilinear_cell`` clamps: the field is extended by its boundary cells, which is what
    # a signed-distance or density lattice wants and is the convention ``poisson_sample_grid``
    # already had.
    s = wp.int32(wp.tid())
    shape = wp.vec3i(field.shape[0], field.shape[1], field.shape[2])
    base, next_corner, fractions = trilinear_cell(
        wp.cw_mul(points[s] - lower, inverse_spacing), shape
    )
    accumulator = out_values.dtype(0.0)
    for offset_x in range(2):
        for offset_y in range(2):
            for offset_z in range(2):
                corner = trilinear_corner(base, next_corner, wp.vec3i(offset_x, offset_y, offset_z))
                accumulator += (
                    trilinear_weight(fractions, offset_x, offset_y, offset_z)
                    * field[corner[0], corner[1], corner[2]]
                )
    out_values[s] = accumulator


# Concrete overloads, registered at import -- rationale in ``ordito/kernels/reduce.py``.
#
# Field dtypes the three transfer kernels admit, and the whole set: every one of them scales the
# field by a ``float32`` weight and sums, and Warp requires both operands of that product to share a
# scalar type, so the admissible fields are exactly the ``float32``-scalar ones this package
# transfers -- a scalar, a ``wp.vec2`` UV, a ``wp.vec3`` position / normal / colour, a ``wp.vec4``
# RGBA colour. The three
# tables register the same set because a caller who can move a UV through
# ``transfer_through_operator`` must be able to move it through the closest-point transfer too; the
# operator path is documented as the preferred one *when an operator exists*, which makes the other
# two its fallback rather than a narrower API. A ``float64`` field is deliberately absent -- these
# operators are assembled from ``float32`` vertex data, so carrying one would advertise precision
# the weights do not have.
FIELD_DTYPES = (wp.float32, wp.vec2, wp.vec3, wp.vec4)

# Attribute dtypes a provenance gather copies (``gather_or_zero``): a copy needs no arithmetic, so
# labels, masks, colours stored as bytes and ``float64`` fields are in as well as the field dtypes.
ATTRIBUTE_DTYPES = (
    wp.bool,
    wp.uint8,
    wp.int32,
    wp.int64,
    wp.float32,
    wp.float64,
    wp.vec2,
    wp.vec3,
    wp.vec4,
    wp.vec2d,
    wp.vec3d,
    wp.vec2i,
    wp.vec3i,
    wp.types.vector(4, wp.uint8),
)

# Queries below which ``interpolate_from_points`` walks a block a query
# (``interpolate_from_points_in_ball_tiled``), lower than the counts' shared
# ``neighbors.TILED_BALL_WALK_MAX_QUERIES``: its fold costs more a candidate. Call time against
# the one-thread walk over cells of twice the radius: 0.51-0.54x at 2.5 k and 10 k queries,
# 0.59x at 10 k over 41 k sources, but 1.2x at 40 k queries over 160 k.
INTERPOLATE_TILED_MAX_QUERIES = 1 << 14

# The concrete handles keyed by the caller's value dtype -- see
# [`OverloadTable`][ordito.kernels.array.OverloadTable].
APPLY_TRANSFER_OPERATOR: OverloadTable
GATHER_OR_ZERO: OverloadTable
INTERPOLATE_BARYCENTRIC: OverloadTable
SAMPLE_GRID_TRILINEAR: OverloadTable
TRANSFER_ONTO_VERTICES: OverloadTable
INTERPOLATE_FROM_POINTS_NEAREST: OverloadTable
INTERPOLATE_FROM_POINTS_IN_BALL: OverloadTable
INTERPOLATE_FROM_POINTS_IN_BALL_TILED: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global APPLY_TRANSFER_OPERATOR, GATHER_OR_ZERO, INTERPOLATE_BARYCENTRIC, SAMPLE_GRID_TRILINEAR
    global TRANSFER_ONTO_VERTICES, INTERPOLATE_FROM_POINTS_NEAREST, INTERPOLATE_FROM_POINTS_IN_BALL
    global INTERPOLATE_FROM_POINTS_IN_BALL_TILED
    APPLY_TRANSFER_OPERATOR = OverloadTable(
        apply_transfer_operator,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.float32],
                wp.array[d],
                wp.array[d],
            ]
            for d in FIELD_DTYPES
        },
    )
    GATHER_OR_ZERO = OverloadTable(
        gather_or_zero,
        {d: [wp.array[d], wp.array[wp.int32], d, wp.array[d]] for d in ATTRIBUTE_DTYPES},
    )
    INTERPOLATE_BARYCENTRIC = OverloadTable(
        interpolate_barycentric,
        {
            d: [wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.vec3], wp.array[d], wp.array[d]]
            for d in FIELD_DTYPES
        },
    )
    SAMPLE_GRID_TRILINEAR = OverloadTable(
        sample_grid_trilinear,
        {
            d: [wp.array3d[d], wp.vec3, wp.vec3, wp.array[wp.vec3], wp.array[d]]
            for d in (wp.float32, wp.vec3)
        },
    )
    TRANSFER_ONTO_VERTICES = OverloadTable(
        transfer_onto_vertices,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[d],
                wp.uint64,
                wp.array[wp.vec3],
                wp.float32,
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.float32],
            ]
            for d in FIELD_DTYPES
        },
    )
    INTERPOLATE_FROM_POINTS_NEAREST = OverloadTable(
        interpolate_from_points_nearest,
        {
            d: [
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.float32],
                wp.int32,
                wp.float32,
                wp.array[d],
            ]
            for d in FIELD_DTYPES
        },
    )
    INTERPOLATE_FROM_POINTS_IN_BALL = OverloadTable(
        interpolate_from_points_in_ball,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[d],
                wp.array[wp.vec3],
                wp.uint64,
                wp.float32,
                wp.float32,
                wp.array[d],
            ]
            for d in FIELD_DTYPES
        },
    )
    INTERPOLATE_FROM_POINTS_IN_BALL_TILED = OverloadTable(
        interpolate_from_points_in_ball_tiled,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[d],
                wp.array[wp.vec3],
                wp.uint64,
                wp.float32,
                wp.float32,
                wp.float32,
                wp.float32,
                wp.int32,
                wp.array[d],
            ]
            for d in FIELD_DTYPES
        },
    )


_register_overloads()


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's ``wp.map`` signatures so each builds one module.

    See ``kernels/array.py::declare_map_signatures`` for why this exists.
    """
    dense = map_probe
    declare_map_signatures(
        [
            (mean_rounded, (dense(wp.float64), dense(wp.float32)), wp.float32),
            (mean_rounded_if_counted, (dense(wp.float64), dense(wp.float32)), wp.float32),
        ]
    )


_declare_map_kernels()
