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
reductions differ only by the ``scale`` passed from Python scope. Following the
``pytorch3d`` convention the distances are **squared** Euclidean distances.
"""

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT, TILE_1D
from ordito.kernels.array import atomic_min_packed_box
from ordito.kernels.reduce import block_sum, tile_chunk
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


@wp.kernel
def chamfer_nn_term_tiled(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: wp.float32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * ||x[i] - y[nearest[i]]||^2`` into ``out_loss[0]``.

    ``nearest[i]`` is the (fixed, non-differentiable) index in ``y`` closest to ``x[i]``.
    CUDA path, launched via ``wp.launch_tiled`` (block ``TILE_1D``): each block reduces its lanes
    cooperatively and commits one atomic; out-of-range lanes contribute zero. Tile ops carry
    adjoints, so the kernel stays differentiable under ``wp.Tape``.

    The CPU device MUST use
    [`chamfer_nn_term_sliced`][ordito.kernels.metrics.chamfer_nn_term_sliced] instead:
    ``wp.launch_tiled`` runs exactly one lane per block there, so this block reduction would see
    one point per tile and the loss would come out a whole tile-width too small.

    The lane-strided ``ITEMS_PER_BLOCK_1D`` fold that would make this portable is declined for the
    reason measured on its sibling [`centroid_tiled`][ordito.kernels.measures.centroid_tiled]: it
    is flat, because one accumulator slot at ``n / TILE_1D`` blocks is nowhere near the contention
    where the fold starts paying.
    """
    i, t = wp.tid()
    idx = i * TILE_1D + t
    contrib = wp.float32(0.0)
    if idx < x.shape[0]:
        diff = x[idx] - y[nearest[idx]]
        contrib = scale * wp.length_sq(diff)
    total = block_sum(contrib)
    if t == 0:
        wp.atomic_add(out_loss, 0, total)


@wp.kernel
def chamfer_nn_term_sliced(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: wp.float32,
    n_slices: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * ||x[i] - y[nearest[i]]||^2`` into ``out_loss[0]``.

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
        diff = x[idx] - y[nearest[idx]]
        total = total + scale * wp.length_sq(diff)
    wp.atomic_add(out_loss, 0, total)


@wp.kernel
def chamfer_surface_term_tiled(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    scale: wp.float32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * d(points[i], triangle face_id[i])^2`` into ``out_loss[0]``.

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
            contrib = scale * point_triangle_sq_dist(points[idx], a, b, c)
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
    n_slices: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * d(points[i], triangle face_id[i])^2`` into ``out_loss[0]``.

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
            total = total + scale * point_triangle_sq_dist(points[idx], a, b, c)
    wp.atomic_add(out_loss, 0, total)
