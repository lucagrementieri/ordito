"""
Differentiable kernels for Chamfer distance losses.

The nearest-neighbor / closest-face assignment is a non-differentiable ``argmin``
and is computed by the proximity primitives *outside* the autodiff tape (see
[`ordito.metrics`][]). These kernels consume that fixed assignment and compute
per-element squared distances as pure arithmetic of the input coordinates, so
Warp's reverse-mode autodiff (``wp.Tape``) flows gradients back to the point
positions (and, for the surface terms, the mesh vertices).

Each term accumulates a *scaled* contribution into a length-1 loss accumulator via
``wp.atomic_add`` (which has a well-defined adjoint), so ``"sum"`` and ``"mean"``
reductions differ only by the ``scale`` passed from Python scope. Under ``ordito.config``'s
deterministic setting the CUDA device takes a fixed-order form instead (gathered operands pulled
per query, terms written and folded by ``sum_in_blocks``), so the loss and its gradients repeat.
Each term is the Euclidean distance by default and its square under ``squared`` (``pytorch3d``'s
convention), both formed by [`chamfer_term`][ordito.kernels.metrics.chamfer_term] from the squared
distance the geometry computes.
"""

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT, TILE_1D
from ordito.kernels.array import atomic_min_packed_box, binary_search_index_left
from ordito.kernels.reduce import (
    block_chunk,
    block_chunk_1d,
    block_max,
    block_sum,
    store_block_sum,
    tile_chunk,
)
from ordito.kernels.triangles import face_vertices

# Relative coplanarity tolerance for the triangle-interior test, matching
# warp.fem's ``project_on_tri_at_origin``.
_TRI_DET_TOLERANCE = wp.constant(wp.float32(1.0e-6))


@wp.func
def mark_first_hit(hits: wp.array[wp.int32], target: wp.int32) -> wp.float32:
    # ``1`` the first time ``target`` is marked and ``0`` after, so a sum over the markers counts
    # the distinct targets whatever order the threads reach them in.
    return wp.where(wp.atomic_exch(hits, target, 1) == 0, wp.float32(1.0), wp.float32(0.0))


@wp.kernel(enable_backward=False)
def pair_search_scale(
    points: wp.array[wp.vec3],
    distances: wp.array[wp.float32],
    assignment: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    per_face: wp.int32,
    hits: wp.array[wp.int32],
    out_scale: wp.array[wp.float32],
) -> None:
    # What the backward search of a cloud pair is chosen from, in one launch into one buffer, from
    # the forward half's answer at each point of the cloud that search will run over (``points``):
    # that cloud's bounding box, the largest forward distance, and how many of the backward
    # *queries* are some point's forward answer -- ``assignment`` indexes them directly, or, when
    # ``per_face`` is set, names a triangle whose three corners are the queries (the forward half
    # of a point-to-mesh pair is point-to-surface; ``faces`` is ``None`` otherwise). ``hits`` is a
    # zeroed mark per query.
    #
    # ``out_scale`` is ``reduce.minmax_vec3_chunked``'s packed box in slots 0-5, the *negated*
    # largest distance in slot 6 (both reduced by ``atomic_min`` from an ``inf`` seed) and the hit
    # count in slot 7 (``atomic_add`` from zero). Launched with ``reduce.chunks_1d``: one plain
    # thread per ``TILE_1D`` points, as that kernel is, and runs on either device.
    offset, remaining = tile_chunk(points.shape[0], wp.int32(wp.tid()), TILE_1D)
    if remaining <= 0:
        return
    count = wp.min(remaining, TILE_1D)

    lower = points[offset]
    upper = points[offset]
    largest = wp.float32(-FLOAT32_INF_CONSTANT)
    first_hits = wp.float32(0.0)
    for k in range(count):
        i = offset + k
        p = points[i]
        lower = wp.min(lower, p)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, p)
        largest = wp.max(largest, distances[i])
        target = assignment[i]
        if target >= 0:
            if per_face != 0:
                for corner in range(3):
                    first_hits += mark_first_hit(hits, faces[target * 3 + corner])
            else:
                first_hits += mark_first_hit(hits, target)

    atomic_min_packed_box(out_scale, wp.int32(0), lower, upper)
    wp.atomic_min(out_scale, 6, -largest)
    wp.atomic_add(out_scale, 7, first_hits)


@wp.func
def project_segment_sq_dist(q: wp.vec3, segment: wp.vec3, length_sq: wp.float32) -> wp.float32:
    """
    Squared distance from ``q`` to the segment ``[0, segment]`` (clamped projection).

    ``s`` is computed only for a segment of positive length. A zero-length one is the single point
    at the origin, so ``q - s * segment`` is ``q`` for *every* ``s`` and the answer is
    ``length_sq(q)`` however the parameter is chosen -- the guard is therefore exact rather than a
    tolerance, and it changes no value this function ever returned. What it changes is the
    *derivative*: the unguarded form evaluates ``0 / 0`` on a triangle with two coincident
    vertices, and while the forward value survives that (``wp.min`` over a ``wp.vec3`` reduces with
    ``<``, which a NaN always loses) the adjoint does not, so the caller's whole accumulated
    gradient came back NaN from one such face. This module is the one differentiated through
    ``wp.Tape``.
    """
    # Initialized before the branch: Warp leaves a branch-local variable readable but uninitialized
    # when the branch is not taken.
    s = wp.float32(0.0)
    if length_sq > wp.float32(0.0):
        s = wp.clamp(wp.dot(q, segment) / length_sq, wp.float32(0.0), wp.float32(1.0))
    return wp.length_sq(q - s * segment)


@wp.func
def point_triangle_sq_dist(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    """
    Squared Euclidean distance from point ``p`` to triangle ``(a, b, c)``.

    Adapts warp.fem's ``project_on_tri_at_origin``: the closest point lies either in
    the triangle interior (barycentric ``s, t`` both non-negative with ``s + t <= 1``)
    or on one of the three edges. Every branch is a smooth function of ``p, a, b, c``,
    so Warp differentiates whichever branch is taken. Degenerate (near-zero-area)
    triangles fall through to the edge projections.
    """
    q = p - a
    e1 = b - a
    e2 = c - a
    e1e1 = wp.length_sq(e1)
    e1e2 = wp.dot(e1, e2)
    e2e2 = wp.length_sq(e2)
    det = e1e1 * e2e2 - e1e2 * e1e2

    # Initialize before branching (Warp leaves branch-local variables uninitialized
    # when the branch is not taken).
    dist_sq = wp.float32(0.0)
    inside = wp.int32(0)
    if det > e1e1 * e2e2 * _TRI_DET_TOLERANCE:
        e1p = wp.dot(e1, q)
        e2p = wp.dot(e2, q)
        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det
        if s >= wp.float32(0.0) and t >= wp.float32(0.0) and s + t <= wp.float32(1.0):
            dist_sq = wp.length_sq(q - s * e1 - t * e2)
            inside = wp.int32(1)

    if inside == wp.int32(0):
        d_e1 = project_segment_sq_dist(q, e1, e1e1)
        d_e2 = project_segment_sq_dist(q, e2, e2e2)
        d_e12 = project_segment_sq_dist(q - e1, e2 - e1, wp.length_sq(e2 - e1))
        dist_sq = wp.min(wp.vec3(d_e1, d_e2, d_e12))

    return dist_sq


@wp.func
def chamfer_term(dist_sq: wp.float32, squared: wp.int32) -> wp.float32:
    """
    One Chamfer term from a squared distance: ``dist_sq`` itself under ``squared``, else its root.

    The root is taken only for a positive ``dist_sq``. At zero the forward value is the same either
    way, but ``sqrt``'s adjoint there is ``0.5 / sqrt(0)``, which would turn the zero gradient of a
    coincident pair into a NaN for the whole accumulated loss; inside the branch the adjoint runs
    only where it is finite, and a coincident pair keeps the zero subgradient ``dist_sq``'s own
    adjoint gives it. ``squared`` is uniform across the launch, so the branch costs no divergence.
    """
    term = dist_sq
    if squared == 0 and dist_sq > wp.float32(0.0):
        term = wp.sqrt(dist_sq)
    return term


@wp.func
def point_pair_term(p: wp.vec3, q: wp.vec3, scale: wp.float32, squared: wp.int32) -> wp.float32:
    """``scale`` times the Chamfer term of the pair ``(p, q)``: every point-to-point kernel's."""
    return scale * chamfer_term(wp.length_sq(p - q), squared)


@wp.func
def point_triangle_term(
    p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, scale: wp.float32, squared: wp.int32
) -> wp.float32:
    """``scale`` times the Chamfer term of ``p`` and triangle ``abc``: every surface kernel's."""
    return scale * chamfer_term(point_triangle_sq_dist(p, a, b, c), squared)


@wp.kernel
def chamfer_nn_term_tiled(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: wp.float32,
    squared: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * ||x[i] - y[nearest[i]]||`` into ``out_loss[0]``, squared under ``squared``.

    ``nearest[i]`` is the (fixed, non-differentiable) index in ``y`` closest to ``x[i]``.
    CUDA path, launched via ``wp.launch_tiled`` (block ``TILE_1D``): each block reduces its lanes
    cooperatively and commits one atomic; out-of-range lanes contribute zero. Tile ops carry
    adjoints, so the kernel stays differentiable under ``wp.Tape``.

    The CPU device MUST use
    [`chamfer_nn_term_sliced`][ordito.kernels.metrics.chamfer_nn_term_sliced] instead:
    ``wp.launch_tiled`` runs exactly one lane per block there, so this block reduction would see
    one point per tile and the loss would come out a whole tile-width too small.

    The lane-strided ``ITEMS_PER_BLOCK_1D`` fold that would make this portable is declined: it
    measured flat on the centroid reduction this kernel was modelled on, because one accumulator
    slot at ``n / TILE_1D`` blocks is nowhere near the contention where the fold starts paying.
    """
    i, t = wp.tid()
    idx = i * TILE_1D + t
    contrib = wp.float32(0.0)
    if idx < x.shape[0]:
        contrib = point_pair_term(x[idx], y[nearest[idx]], scale, squared)
    total = block_sum(contrib)
    if t == 0:
        wp.atomic_add(out_loss, 0, total)


@wp.kernel
def chamfer_nn_term_sliced(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: wp.float32,
    squared: wp.int32,
    n_slices: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * ||x[i] - y[nearest[i]]||`` into ``out_loss[0]``, squared under ``squared``.

    Portable path, correct on both devices: one thread per slice walks a strided slice of ``x``,
    accumulates locally and commits one ``wp.atomic_add``. Both the dynamic loop and the atomic
    carry adjoints, so the kernel stays differentiable under ``wp.Tape``. Lane-free, so it is
    correct on the CPU device where
    [`chamfer_nn_term_tiled`][ordito.kernels.metrics.chamfer_nn_term_tiled] is not; it gives up
    the block shuffle-reduce and is measurably slower on CUDA, which is why both exist.

    A slice that owns no element returns before its loop. On Warp 1.18 the backward pass of a
    dynamic ``range(start, end, step)`` walks ``iter_reverse`` of it, which for an *empty* range
    whose ``end - start - 1`` truncates to zero against ``step`` is not empty but one iteration at
    ``start`` -- past the end of ``x`` -- so an empty slice read ``x`` and ``nearest`` out of bounds
    and scattered a gradient through them. ``slice_count`` never launches more slices than points,
    so no public call reached it; the guard makes the kernel safe under any slice count.
    """
    j = wp.int32(wp.tid())
    if j >= x.shape[0]:
        return
    total = wp.float32(0.0)
    for idx in range(j, x.shape[0], n_slices):
        total = total + point_pair_term(x[idx], y[nearest[idx]], scale, squared)
    wp.atomic_add(out_loss, 0, total)


@wp.kernel
def chamfer_surface_term_tiled(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    scale: wp.float32,
    squared: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * d(points[i], triangle face_id[i])`` into ``out_loss[0]``, squared or not.

    ``face_id[i]`` is the (fixed, non-differentiable) index of the triangle of the mesh
    closest to ``points[i]``. Gradients flow to both ``points`` and ``vertices``. Points
    with ``face_id[i] < 0`` (no face within the search radius) contribute nothing. Same tiled
    CUDA reduction as [`chamfer_nn_term_tiled`][ordito.kernels.metrics.chamfer_nn_term_tiled],
    and CPU-unsafe for the same reason.
    """
    i, t = wp.tid()
    idx = i * TILE_1D + t
    contrib = wp.float32(0.0)
    if idx < points.shape[0]:
        f = face_id[idx]
        if f >= 0:
            a, b, c = face_vertices(vertices, faces, f)
            contrib = point_triangle_term(points[idx], a, b, c, scale, squared)
    total = block_sum(contrib)
    if t == 0:
        wp.atomic_add(out_loss, 0, total)


@wp.kernel
def chamfer_surface_term_sliced(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    scale: wp.float32,
    squared: wp.int32,
    n_slices: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * d(points[i], triangle face_id[i])`` into ``out_loss[0]``, squared or not.

    ``face_id[i]`` is the (fixed, non-differentiable) index of the triangle of the mesh
    closest to ``points[i]``. Gradients flow to both ``points`` and ``vertices``. Points
    with ``face_id[i] < 0`` (no face within the search radius) contribute nothing. Same lane-free
    sliced reduction as
    [`chamfer_nn_term_sliced`][ordito.kernels.metrics.chamfer_nn_term_sliced], for the same
    reason, and guarding an empty slice for the same reason too.
    """
    j = wp.int32(wp.tid())
    if j >= points.shape[0]:
        return
    total = wp.float32(0.0)
    for idx in range(j, points.shape[0], n_slices):
        f = face_id[idx]
        if f >= 0:
            a, b, c = face_vertices(vertices, faces, f)
            total = total + point_triangle_term(points[idx], a, b, c, scale, squared)
    wp.atomic_add(out_loss, 0, total)


# The fixed-order form of the four kernels above, taken on CUDA under ``ordito.config``'s
# deterministic setting. Those commit one float atomic per block into the loss, and their adjoints
# scatter a gathered operand's gradient (``y[nearest[i]]``, a face's corners) from every query
# that picked it, both in arrival order. Here every array a kernel reads by a gathered index is
# first copied out per query by ``gather_by_sorted_key``, a pull whose adjoint sums each source's
# gradient in ascending query order on one thread; the per-query terms are written, not added;
# and ``sum_in_blocks`` folds them in two fixed-order passes. Every adjoint is then a gather or a
# write with one writer per slot. The per-query arithmetic is ``point_pair_term`` /
# ``point_triangle_term``, shared with the arrival-order kernels, so they differ by the order of
# the sums alone.


@wp.kernel(enable_backward=False)
def surface_corner_keys(
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # The vertex under each corner ``3 i + k`` of query ``i``'s closest face, ``n_vertices`` (past
    # every vertex) for a query with no face, into the leading half of a radix sort's
    # double-width key buffer beside the identity permutation.
    i = wp.int32(wp.tid())
    f = face_id[i]
    for k in range(3):
        corner = 3 * i + k
        out_keys[corner] = wp.where(f >= 0, faces[3 * wp.max(f, 0) + k], n_vertices)
        out_order[corner] = corner


@wp.kernel
def gather_by_sorted_key(
    values: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_gathered: wp.array[wp.vec3],
) -> None:
    """
    ``out_gathered[order[s]] = values[sorted_keys[s]]``, one thread per entry of ``values``.

    ``sorted_keys`` / ``order`` are a stable sort of the gather's index array and its permutation,
    so thread ``v`` writes every slot that reads ``values[v]``, in ascending slot order. The adjoint
    is then ``values.grad[v]`` summed over those slots on that one thread, in a fixed order, where
    the adjoint of a plain ``values[index[s]]`` read adds into it from every slot's thread.

    An entry no slot reads returns before its loop: the adjoint of an empty dynamic ``range`` runs
    one iteration at its start, which would scatter a slot's gradient into this entry.
    """
    v = wp.int32(wp.tid())
    start = binary_search_index_left(sorted_keys, v)
    end = binary_search_index_left(sorted_keys, v + 1)
    if start == end:
        return
    value = values[v]
    for s in range(start, end):
        out_gathered[order[s]] = value


@wp.kernel
def chamfer_nn_terms(
    x: wp.array[wp.vec3],
    y_nearest: wp.array[wp.vec3],
    scale: wp.float32,
    squared: wp.int32,
    offset: wp.int32,
    out_terms: wp.array[wp.float32],
) -> None:
    """
    ``out_terms[offset + i] = scale * ||x[i] - y_nearest[i]||``, squared under ``squared``.

    ``y_nearest[i]`` is ``x[i]``'s nearest point, gathered by
    [`gather_by_sorted_key`][ordito.kernels.metrics.gather_by_sorted_key].
    """
    i = wp.int32(wp.tid())
    out_terms[offset + i] = point_pair_term(x[i], y_nearest[i], scale, squared)


@wp.kernel
def chamfer_surface_terms(
    points: wp.array[wp.vec3],
    corners: wp.array[wp.vec3],
    face_id: wp.array[wp.int32],
    scale: wp.float32,
    squared: wp.int32,
    offset: wp.int32,
    out_terms: wp.array[wp.float32],
) -> None:
    """
    ``out_terms[offset + i] = scale * d(points[i], triangle face_id[i])``, squared or not.

    ``corners[3 i + k]`` is corner ``k`` of that triangle, gathered by
    [`gather_by_sorted_key`][ordito.kernels.metrics.gather_by_sorted_key]; a point with
    ``face_id[i] < 0`` contributes zero.
    """
    i = wp.int32(wp.tid())
    term = wp.float32(0.0)
    if face_id[i] >= 0:
        term = point_triangle_term(
            points[i], corners[3 * i], corners[3 * i + 1], corners[3 * i + 2], scale, squared
        )
    out_terms[offset + i] = term


@wp.kernel
def sum_in_blocks(
    values: wp.array[wp.float32], width: wp.int32, out_sums: wp.array[wp.float32]
) -> None:
    """
    ``out_sums[b]`` = the sum of block ``b``'s ``width`` elements of ``values``, in a fixed order.

    Lanes stride the chunk by ``wp.block_dim()`` and the block folds them with ``wp.tile_sum``, so
    the order depends on the block width alone. Launched twice, at ``ITEMS_PER_BLOCK_1D`` into
    per-block partials and as one block over those into the loss. It writes rather than adds, so
    its adjoint gives every element the gradient of its block's slot. A lane with nothing to fold
    skips its loop (the empty-range adjoint, above) but still joins the block reduction.
    """
    block, lane = wp.tid()
    offset, count = block_chunk(values.shape[0], block, width)
    partial = wp.float32(0.0)
    if lane < count:
        for k in range(lane, count, wp.block_dim()):
            partial += values[offset + k]
    total = block_sum(partial)
    if lane == 0:
        out_sums[block] = total


@wp.func
def block_max_of_chunk(values: wp.array[wp.float32], chunk: wp.int32, lane: wp.int32) -> wp.float32:
    # The largest of block ``chunk``'s ``ITEMS_PER_BLOCK_1D`` chunk of ``values``, on every lane;
    # ``-inf`` for an empty chunk. Lanes stride the chunk by ``wp.block_dim()``, so the CPU
    # device's single lane covers it. Block-collective.
    offset, count = block_chunk_1d(values.shape[0], chunk)
    best = wp.float32(-FLOAT32_INF_CONSTANT)
    for k in range(lane, count, wp.block_dim()):
        best = wp.max(best, values[offset + k])
    return block_max(best)


@wp.kernel(enable_backward=False)
def max_of_pair(
    first: wp.array[wp.float32],
    second: wp.array[wp.float32],
    first_blocks: wp.int32,
    out_max: wp.array[wp.float32],
) -> None:
    # The largest element of two arrays into ``out_max[0]``, seeded ``-inf``: one launch
    # ``blocks_1d(n_first) + blocks_1d(n_second)`` blocks wide, the leading ``first_blocks`` folding
    # ``first`` and the rest ``second``, one ``atomic_max`` per block. A maximum is exact in any
    # order, so this is the float two ``reduce.max`` calls would return, at one launch.
    chunk, lane = wp.tid()
    best = wp.float32(0.0)
    if chunk < first_blocks:
        best = block_max_of_chunk(first, chunk, lane)
    else:
        best = block_max_of_chunk(second, chunk - first_blocks, lane)
    if lane == 0:
        wp.atomic_max(out_max, 0, best)


@wp.func
def lane_sum_of_chunk(
    values: wp.array[wp.float32], chunk: wp.int32, lane: wp.int32, squared: wp.int32
) -> wp.float32:
    # This lane's share of the sum of block ``chunk``'s ``ITEMS_PER_BLOCK_1D`` chunk of ``values``
    # (of their squares when ``squared`` is non-zero), lanes striding the chunk by
    # ``wp.block_dim()`` as in ``block_max_of_chunk``.
    offset, count = block_chunk_1d(values.shape[0], chunk)
    total = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        value = values[offset + k]
        total += wp.where(squared != 0, value * value, value)
    return total


@wp.kernel(enable_backward=False)
def totals_of_pair(
    first: wp.array[wp.float32],
    second: wp.array[wp.float32],
    first_blocks: wp.int32,
    squared: wp.int32,
    stride: wp.int32,
    out_totals: wp.array[wp.float32],
) -> None:
    # The totals of two arrays (of their squares under ``squared``) into slots 0 and 1 of a
    # ``_launch.block_partials`` buffer of width 2: ``max_of_pair``'s block split, the leading
    # ``first_blocks`` blocks summing ``first`` and the rest ``second``, committed by
    # ``store_block_sum`` (a fixed-order fold under ``ordito.config``'s deterministic setting,
    # one ``atomic_add`` per slot and block otherwise).
    chunk, lane = wp.tid()
    total = wp.vec2(0.0, 0.0)
    if chunk < first_blocks:
        total = wp.vec2(lane_sum_of_chunk(first, chunk, lane, squared), 0.0)
    else:
        total = wp.vec2(0.0, lane_sum_of_chunk(second, chunk - first_blocks, lane, squared))
    store_block_sum(chunk, lane, total, out_totals, 0, stride)
