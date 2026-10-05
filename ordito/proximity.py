"""
Queries that ask where a point stands relative to a triangle mesh.

Three answers, in increasing order of what they need from the mesh:
[`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] and
[`normals_at_closest_faces`][ordito.proximity.normals_at_closest_faces] need only a surface;
[`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] needs a consistent winding
to give the distance a sign; and [`winding_number`][ordito.proximity.winding_number] needs neither
watertightness nor manifoldness, which is why it is the robust inside test on damaged input.
[`containing_faces_2d`][ordito.proximity.containing_faces_2d] is the planar case -- point location
in a 2D triangulation -- and
[`query_mesh_aabb_with_offsets`][ordito.proximity.query_mesh_aabb_with_offsets] is
the low-level box query the others are built over.

Everything here takes raw ``(vertices, faces)`` buffers. The queries phrased the other way round --
*how far away* the surface is rather than *where* it is, all taking a prebuilt ``wp.Mesh`` and a set
of points to measure at -- live in [`ordito.visibility`][ordito.visibility]: ambient occlusion and
obscurance outward, shape diameter and thickness inward, and the maximal tangent sphere in every
direction at once.

Everything signed here, plus [`contains_points`][ordito.ray.contains_points], follows Warp's SDF
sign convention: outside positive, inside negative. Trimesh's ``signed_distance`` uses the opposite
sign.

Point-set acceleration structures (``wp.Bvh`` / ``wp.HashGrid``) and raw neighbor queries live in
[`ordito.neighbors`][ordito.neighbors]; axis-aligned bounding boxes in
[`ordito.bounds`][ordito.bounds].
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import (
    prefers_tiled_reduction,
    read_scalar,
    require_nonempty_mesh,
    require_same_device,
)
from ordito.constants import INT32_MAX, INT64_MAX, TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import edges as kernel_edges
from ordito.kernels import neighbors as kernel_neighbors
from ordito.kernels import proximity as kernel_proximity
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import triangles as kernel_triangles

# Elements per thread for the two *per-query* lane-free reductions here (solid-angle sum, packed
# support argmax). Unlike a global reduction, these have one accumulator per query, so the query
# dimension already supplies the parallelism and a short slice only multiplies the atomic traffic.
# This is not a sensitive knob: the optimum is flat over a fairly wide range.
ITEMS_PER_QUERY_SLICE = 128

# First search radius for [`closest_point_on_edges`][ordito.proximity.closest_point_on_edges], as a
# fraction of the ``max_dist`` its deepening loop is capped at. The loop doubles from here and jumps
# straight to the certified radius as soon as it holds any candidate, so this only decides how many
# empty scans a query far from every edge pays; too *large* a start is the expensive mistake, since
# the first scan then enumerates the whole edge set.
_EDGE_INITIAL_RADIUS_SCALE = 0.01

# Smallest positive normal float32, used as the floor on
# [`mesh_to_mesh_distance`][ordito.proximity.mesh_to_mesh_distance]'s seeded running minimum: two
# touching meshes give an ``upper_bound`` of ``0``, and a limit of ``0`` would prune every candidate
# including the zero-gap pair that achieves the answer.
_MIN_POSITIVE_FLOAT32 = 1.1754943508222875e-38

# Broad-phase candidates a single thread walks in
# [`mesh_to_mesh_distance`][ordito.proximity.mesh_to_mesh_distance]'s first pass before handing its
# face to the block-cooperative second one.
#
# The split exists because that traversal is badly unbalanced: the vast majority of query faces
# return no candidate at all, a small fraction carry most of the cost, and the busiest single face
# can walk a BVH sequentially for far longer than typical, so the launch's wall time is set by a
# handful of threads while the rest of the machine idles.
#
# 64 is chosen to sit well above the typical floor and far below the tail, so only a small fraction
# of faces overflow into the second launch. The value is not sharp -- it trades first-pass work
# against second-pass launches, and both ends are cheap -- but do not raise it far: the point of the
# cap is that a thread stops *before* it becomes the launch's critical path.
_QUERY_CANDIDATE_CAP = 64

# Query points to draw from mesh A when deriving ``mesh_to_mesh_distance``'s own upper bound: the
# first corner of every ``n_faces_a // _BOUND_SAMPLE_TARGET``-th face. The bound only has to be an
# upper bound -- it seeds the broad phase's prune limit and nothing else -- so a *subsample* of A's
# surface is as correct as all of it and merely looser, and the whole question is what a looser
# bound costs the traversal it is paying for.
#
# Face corners rather than vertices, because a vertex no face references is not a point of A's
# surface: a stray one near B gave a bound below the answer, and the walk then pruned the pair that
# achieves it and returned ``inf``. A stride, not a random draw: it is deterministic and needs no
# RNG, and a pathological face ordering can only make the bound looser, never wrong.
_BOUND_SAMPLE_TARGET = 16_384

# ...but never sparser than one face in this many. Once the queries are capped (next constant) a
# sample point far from B costs next to nothing, and what a sparse sample costs instead is a looser
# bound -- roughly the sample spacing -- which grows every face's query box in the walk. On the
# largest scan mesh the count alone left a stride of ~1 700 faces and the walk paid ~1.6x for it;
# on the meshes where the count already gives a stride under this, the sample is unchanged.
_BOUND_SAMPLE_MAX_STRIDE = 128

# Points per side of the brute-force corner sample that caps those bound queries' ``max_dist``.
# Its only job is to be *some* surface-to-surface distance not far above the answer: an unbounded
# closest-point query from a point far from B walks most of B's BVH, and on the largest scan mesh
# that was nearly half the call. The brute force is ``_SEED_SAMPLE_TARGET ** 2`` pair tests, which
# is negligible here, and a larger sample tightens a cap that is already within a few tens of
# percent of the answer.
_SEED_SAMPLE_TARGET = 1024

# Block width for that second pass: one warp per straggler face. Wider blocks did not help, and a
# warp is what [`ball_pivoting`][ordito.reconstruction.ball_pivoting]'s pivot search settled on for
# the same ``wp.tile_bvh_query_aabb`` walk.
_QUERY_TILE_WIDTH = 32

# Ray-origin offset *below* the surface along the inward normal, as a fraction of the query AABB
# diagonal: without it the cone's own starting triangle is the nearest hit for every ray.
_SDF_SURFACE_OFFSET = 1e-4

# [`containing_faces_2d`][ordito.proximity.containing_faces_2d]'s candidate search radius, as a
# fraction of the triangulation's bounding-box diagonal, and the barycentric slack that then decides
# containment. The radius only has to exceed the float32 rounding of a closest-point query on a flat
# mesh, and being generous costs only BVH descent on queries that land outside; the sign test
# classifies, so the two are not a precision trade-off.
_CONTAINMENT_SEARCH_SCALE = 1e-3
_CONTAINMENT_BARYCENTRIC_EPS = wp.float32(1e-6)


def closest_point_on_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    mesh: wp.Mesh | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    For each query point, find the closest point on any triangle of the mesh.

    Uses ``wp.mesh_query_point_no_sign`` via ``wp.Mesh``. Distances are unsigned
    Euclidean lengths in ``float32``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index array.
        The internally built ``wp.Mesh`` aliases these buffers rather than copying them;
        do not mutate them for the duration of the call.
    points
        ``(m,)`` query positions in space.
    max_dist
        Maximum search radius per query. Faces farther than this are ignored.
        When ``None``, the search is
        unbounded, which finds the same closest point as any radius at least as long as the
        diagonal of the box enclosing the mesh and the query points.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the clone and BVH
        build this otherwise pays on every call. Purely an optimization: the answer is identical
        either way, and it is not checked against ``vertices`` / ``faces`` -- passing a mesh over
        *different* geometry silently answers for that geometry, since only ``mesh`` is queried.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is a cached property and is what to
        pass.

    Returns
    -------
    closest
        ``(m,)`` closest point on the mesh surface for each query.
    distance
        ``(m,)`` unsigned distance from each query to its closest surface point.
    triangle_id
        ``(m,)`` index of the triangle containing each closest point, or ``-1``
        when no face lies within ``max_dist``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``points`` and ``mesh`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces, points=points, mesh=mesh)
    device = vertices.device
    m = points.size
    n_faces = faces.size // 3
    if m == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
        )
    if n_faces == 0:
        nan = float("nan")
        out_closest = _launch.full(m, wp.vec3(nan, nan, nan), dtype=wp.vec3, device=device)
        out_distance = _launch.full(m, float("inf"), dtype=wp.float32, device=device)
        out_face = _launch.full(m, -1, dtype=wp.int32, device=device)
        return out_closest, out_distance, out_face

    if mesh is None:
        require_nonempty_mesh(faces, "closest_point_on_mesh")
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
        mesh = wp.Mesh(points=vertices, indices=faces)
    if max_dist is None:
        max_dist = math.inf

    out_closest = _launch.empty(m, dtype=wp.vec3, device=device)
    out_distance = _launch.empty(m, dtype=wp.float32, device=device)
    out_face = _launch.empty(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_proximity.closest_point_on_mesh,
        dim=m,
        inputs=[mesh.id, points, wp.float32(max_dist), out_closest, out_distance, out_face],
        device=device,
    )
    return out_closest, out_distance, out_face


def closest_point_on_edges(
    vertices: wp.array[wp.vec3],
    edges: odt.Array2dInt32,
    queries: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    bvh: wp.Bvh | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    For each query point, find the closest point on any edge of an edge set.

    The **wireframe** counterpart of
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]: same return triple, same
    ``max_dist`` semantics, but the geometry is a set of segments rather than a surface. That is the
    query a crease set, a seam, a boundary rim or a feature curve wants -- all four are already
    produced as an edge array by [`ordito.seams`][ordito.seams],
    [`ordito.boundary`][ordito.boundary] and [`ordito.edges`][ordito.edges].

    Parameters
    ----------
    vertices
        ``(n,)`` positions the edges index.
    edges
        ``(n_edges, 2)`` vertex-index pairs. Order within a pair is irrelevant, and
        edges may share vertices or repeat.
    queries
        ``(m,)`` query positions in space.
    max_dist
        Maximum search distance per query; an edge farther than this is ignored and the query
        reports a miss. When ``None``, derived from the box enclosing ``vertices`` and ``queries``,
        which no real query can exceed.
    bvh
        A ``wp.Bvh`` already built over this edge set's per-edge boxes, to spare the bounds pass and
        the build. Purely an optimization -- and, as with
        [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]'s ``mesh``, it is not
        checked against ``vertices`` / ``edges``: a BVH over a *different* edge set silently answers
        for the boxes it holds while the distances are computed from these vertices.

    Returns
    -------
    closest
        ``(m,)`` closest point on the edge set for each query.
    distance
        ``(m,)`` unsigned distance from each query to that point.
    edge_id
        ``(m,)`` row of ``edges`` the closest point lies on, or ``-1`` when no edge lies within
        ``max_dist``.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``edges`` does not have two columns.
    RuntimeError
        If ``vertices``, ``edges``, ``queries`` and ``bvh`` are not all on one device.

    Notes
    -----
    A miss reports the query point itself and ``max_dist``, alongside the ``-1`` id -- the same
    convention [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] uses, so the two
    are interchangeable in a caller that only reads the id.

    The traversal is iterative deepening over a per-edge-box BVH and is **exact**, not a broad-phase
    approximation: a scan of the cube of half-extent ``r`` about a query enumerates every edge whose
    closest point lies within ``r``, so a best distance under ``r`` certifies the answer. The
    tempting shortcut -- one degenerate ``(a, b, b)`` triangle per edge, queried with
    ``wp.mesh_query_point_no_sign`` -- does **not** work: Warp's mesh BVH rejects a zero-area
    triangle outright, on both devices.

    See Also
    --------
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
        The surface form. On a closed mesh its answer is never farther than this one's.
    [`ordito.polyline.polyline_point_distance`][ordito.polyline.polyline_point_distance]
        The same computation for an *ordered* chain, where the segments are consecutive vertices and
        no index structure is built.
    """
    require_same_device(vertices=vertices, edges=edges, queries=queries, bvh=bvh)
    device = vertices.device
    odt.ensure_edge_pairs(edges, "edges")
    m = queries.size
    n_edges = int(edges.shape[0])

    if m == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
        )
    if n_edges == 0:
        return (
            _launch.clone(queries),
            _launch.full(m, float("inf"), dtype=wp.float32, device=device),
            _launch.full(m, -1, dtype=wp.int32, device=device),
        )

    if bvh is None:
        lower = _launch.empty(n_edges, dtype=wp.vec3, device=device)
        upper = _launch.empty(n_edges, dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_edges.edge_aabb_bounds,
            dim=n_edges,
            inputs=[vertices, edges, lower, upper],
            device=device,
        )
        bvh = od.neighbors.bvh_from_bounds(lower, upper)
    # The scene box bounds each query's *complete* search radius, so a query outside the geometry
    # still terminates exactly rather than growing to ``max_dist``.
    min_bound, max_bound = od.bounds.aabb(vertices)
    if max_dist is None:
        # ``enclosing_diagonal(vertices, queries)`` from the scene box already in hand plus the
        # queries' own box: one reduction of ``vertices`` rather than two. The union and the norm
        # are taken in float32 NumPy, exactly as ``enclosing_diagonal`` takes them, so the radius
        # is the same float.
        query_lower, query_upper = od.bounds.aabb(queries)
        union_lower = np.minimum(np.array(min_bound, np.float32), np.array(query_lower, np.float32))
        union_upper = np.maximum(np.array(max_bound, np.float32), np.array(query_upper, np.float32))
        max_dist = float(np.linalg.norm(union_upper - union_lower))

    out_closest = _launch.empty(m, dtype=wp.vec3, device=device)
    out_distance = _launch.empty(m, dtype=wp.float32, device=device)
    out_edge = _launch.empty(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_proximity.closest_point_on_edges,
        dim=m,
        inputs=[
            vertices,
            edges,
            queries,
            bvh.id,
            wp.float32(max_dist),
            wp.float32(_EDGE_INITIAL_RADIUS_SCALE * max_dist),
            min_bound,
            max_bound,
            out_closest,
            out_distance,
            out_edge,
        ],
        device=device,
    )
    return out_closest, out_distance, out_edge


def mesh_to_mesh_distance(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    upper_bound: float | None = None,
) -> tuple[float, int, int]:
    """
    Smallest distance between two triangle meshes, with the pair of faces that achieves it.

    The clearance between two parts, and zero when they touch or overlap. Unlike a vertex-to-mesh
    query this is the true minimum over the *surfaces*: two boxes edge to edge realise their
    clearance between edge interiors, and every vertex of each is further from the other than
    that.

    Two phases. An upper bound comes first -- the smallest distance from a *sample* of points on
    ``A``'s faces to ``B``, which is a real distance between the surfaces and therefore an upper
    bound on their minimum. Then every face of ``A`` queries a BVH over ``B``'s faces with its own
    bounding box grown by that bound, and each candidate pair gets the exact triangle-triangle
    distance. The bound is what makes the broad phase sound rather than heuristic: the true minimum
    is at most the bound, so the pair achieving it has boxes within that distance and cannot be
    culled -- and that argument needs an upper bound rather than a *tight* one, which is why
    sampling is sound and why supplying your own coarse ``upper_bound`` is too. The returned
    distance is exact either way; a looser bound only leaves more candidates for the narrow phase to
    reject.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` positions and ``(3 * n_faces_a,)`` index buffer of the first mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` positions and ``(3 * n_faces_b,)`` index buffer of the second mesh.
    upper_bound
        A distance known to be at least the answer, which prunes the broad phase. Supply one when
        you have it -- from a previous frame, or from a bounding-volume gap -- and the sampled query
        that would otherwise derive it is skipped. **Too small a bound gives a wrong answer**, not a
        slow one: it culls the pair that would have won. ``None`` derives a sound bound.

    Returns
    -------
    distance : float
        The minimum distance. ``0.0`` exactly when some pair of faces crosses.
    face_a : int
        The face of the first mesh achieving it, or ``-1`` if either mesh is empty.
    face_b : int
        The face of the second mesh achieving it, or ``-1``.

    Raises
    ------
    ValueError
        If ``upper_bound`` is negative.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.

    See Also
    --------
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
        Point-to-mesh, which is the query this derives its bound from.
    [`mesh_with_mesh`][ordito.intersection.mesh_with_mesh]
        The zero-distance case in detail: every intersecting pair and the segments they cross on.
    [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]
        The one-mesh analogue of that.

    Notes
    -----
    !!! note "Witness faces are ambiguous under ties"
        Two parallel plates have a continuum of closest pairs and any of them is a correct answer,
        and ties are the common case rather than the exotic one: whenever the closest approach is
        realised at a *vertex*, every face around that vertex achieves the minimum exactly. The
        tie-break on ``face_a`` is the lowest index; ``face_b`` is **unspecified** among the faces
        attaining it, and which one comes back can differ between runs of the identical input even
        though the squared distance is bit-identical. Compare *distances* against another
        implementation, and faces only where the configuration is generic.
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    if upper_bound is not None and upper_bound < 0.0:
        raise ValueError(f"upper_bound must be non-negative, got {upper_bound}")
    device = faces_a.device
    n_faces_a = faces_a.size // 3
    n_faces_b = faces_b.size // 3
    if n_faces_a == 0 or n_faces_b == 0:
        return math.inf, -1, -1

    # One structure over B, shared by both phases. It is bound to a name for the whole call because
    # the kernels read its BVH by id through ``wp.mesh_get_bvh``, and a collected ``wp.Mesh`` would
    # leave them a dangling id.
    require_nonempty_mesh(faces_b, "mesh_to_mesh_distance")
    mesh_b = wp.Mesh(points=vertices_b, indices=faces_b)

    # The walk's three running scalars in one buffer (``kernels.proximity.*_SLOT``): the published
    # prune limit, the corner seed and the upper bound that grows every face's query box.
    if upper_bound is None:
        # The smallest distance from a *sample* of A's surface to B is a real distance between the
        # surfaces, so it bounds the answer from above; it is what lets the broad phase cull at
        # all. A subsample because a minimum over a subset is still an upper bound and this query
        # can otherwise dominate the call on a large mesh -- see ``_BOUND_SAMPLE_TARGET``; face
        # corners rather than vertices, because a vertex no face references is not a point of A's
        # surface and would bound nothing. The bound and the prune limit it seeds are written on
        # the device, so the walk below is issued behind them with no readback.
        sample_stride = min(max(1, n_faces_a // _BOUND_SAMPLE_TARGET), _BOUND_SAMPLE_MAX_STRIDE)
        n_samples = (n_faces_a + sample_stride - 1) // sample_stride
        # Unbounded, a closest-point query from a point far from B walks most of B's BVH. So the
        # queries are capped by a cheap, loose upper bound first -- the closest pair between two
        # small corner samples, brute force -- and a point farther than it is a miss that stops at
        # the top of the tree. See ``_SEED_SAMPLE_TARGET``.
        seed_step = max(1, n_samples // _SEED_SAMPLE_TARGET)
        target_stride = max(1, n_faces_b // _SEED_SAMPLE_TARGET)
        bounds = _launch.full(3, math.inf, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_proximity.sampled_corner_gap_sq,
            dim=(kernel_proximity.SEED_SLICES, (n_samples + seed_step - 1) // seed_step),
            inputs=[
                vertices_a,
                faces_a,
                sample_stride * seed_step,
                vertices_b,
                faces_b,
                target_stride,
                (n_faces_b + target_stride - 1) // target_stride,
                bounds,
            ],
            device=device,
        )
        _launch.launch(
            kernel_proximity.sampled_corner_distance_min,
            dim=n_samples,
            inputs=[mesh_b.id, vertices_a, faces_a, sample_stride, bounds],
            device=device,
        )
    else:
        # Seeded at the caller's bound, so every thread prunes against it from its first candidate
        # instead of waiting for some other thread to publish one.
        #
        # Seeded at *exactly* ``upper_bound ** 2`` this is wrong: the prune skips a candidate whose
        # box gap is ``>=`` the limit, so when the bound *is* the answer -- two spheres whose
        # closest points are vertices -- the very pair achieving it is skipped and the result comes
        # back ``inf``. The relative bump is what keeps that pair, and it is a relative ``1e-4``
        # rather than an ulp because a bound from ``mesh_query_point_no_sign`` is itself accurate
        # to a few times 1e-5; the margin clears that by several times and weakens the prune by
        # nothing measurable. ``max`` covers touching meshes, where the bound is ``0`` and any
        # positive limit keeps the exactly-zero-gap pair. The sampled bound above is published
        # into the limit by the identical rule, on the device (``publish_best_sq``).
        bounds = _launch.array(
            np.array(
                [
                    max(upper_bound * upper_bound * (1.0 + 1e-4), _MIN_POSITIVE_FLOAT32),
                    math.inf,
                    upper_bound,
                ],
                dtype=np.float32,
            ),
            dtype=wp.float32,
            device=device,
        )

    # The per-face AABBs stay: the kernel's box-gap prune reads them, so they are not merely the
    # input to a build. There is no second acceleration structure built over them -- the kernels
    # read the ``wp.Mesh`` built above's own BVH directly with ``wp.mesh_get_bvh``.
    lower = _launch.empty(n_faces_b, dtype=wp.vec3, device=device)
    upper = _launch.empty(n_faces_b, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_triangles.face_aabb_bounds,
        dim=n_faces_b,
        inputs=[vertices_b, faces_b, lower, upper],
        device=device,
    )
    distance_sq = _launch.empty(n_faces_a, dtype=wp.float32, device=device)
    witness = _launch.empty(n_faces_a, dtype=wp.int32, device=device)
    # The broad phase is wildly unbalanced -- most query faces return no candidate at all, while a
    # few carry most of the traversal -- so the walk runs in two passes on CUDA: a thread per face,
    # capped, then a *block* per face that exceeded the cap. See ``_QUERY_CANDIDATE_CAP``. On the
    # cpu device ``wp.launch_tiled`` runs one lane per block, so the second pass would be a serial
    # re-walk; there the cap is disabled and the first pass settles every face instead.
    tiled = prefers_tiled_reduction(device)
    candidate_cap = _QUERY_CANDIDATE_CAP if tiled else INT32_MAX
    overflow = _launch.empty(n_faces_a if tiled else 1, dtype=wp.int32, device=device)
    counter = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_proximity.face_to_mesh_distance,
        dim=n_faces_a,
        inputs=[
            vertices_a,
            faces_a,
            vertices_b,
            faces_b,
            lower,
            upper,
            mesh_b.id,
            wp.int32(candidate_cap),
            bounds,
            distance_sq,
            witness,
            counter,
            overflow,
        ],
        device=device,
    )
    if tiled:
        # One readback, and it is what sizes the second launch. Skipping it by launching
        # ``n_faces_a`` blocks would put an empty block on nearly all of them.
        n_overflow = int(read_scalar(counter, 0))
        if n_overflow > 0:
            _launch.launch_tiled(
                kernel_proximity.face_to_mesh_distance_tiled,
                dim=n_overflow,
                inputs=[
                    vertices_a,
                    faces_a,
                    vertices_b,
                    faces_b,
                    lower,
                    upper,
                    mesh_b.id,
                    overflow,
                    bounds,
                    distance_sq,
                    witness,
                ],
                device=device,
                block_dim=_QUERY_TILE_WIDTH,
            )
    # One launch reduces the packed ``(distance_sq, face_a)`` key -- low half the winning face, high
    # half its squared distance's own float32 bits -- and a second writes that face's witness beside
    # it, so all three answers come back in one 16-byte readback.
    result = _launch.full(2, INT64_MAX, dtype=wp.int64, device=device)
    _launch.launch_tiled(
        kernel_neighbors.nearest_key_argmin,
        dim=kernel_reduce.blocks_1d(n_faces_a),
        inputs=[distance_sq, result],
        block_dim=TILE_1D,
        device=device,
    )
    _launch.launch(
        kernel_neighbors.nearest_key_partner, dim=1, inputs=[result, witness, result], device=device
    )
    key, face_b = (int(value) for value in result.numpy())
    best_distance_sq = np.array([key >> 32], dtype=np.uint32).view(np.float32)[0]
    return math.sqrt(float(best_distance_sq)), key & 0xFFFFFFFF, face_b


def normals_at_closest_faces(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.vec3]:
    """
    Return unit face normals at the closest mesh triangle for each query point.

    For each position in ``points``, runs an unsigned closest-point query on
    ``mesh`` and returns the normal of the hit triangle.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` query positions.
    max_dist
        Maximum search radius per query. When ``None``, the search is
        unbounded, which finds the same closest point as any radius at least as long as the
        diagonal of the box enclosing the mesh and the query points.
    face_normals
        ``(n_faces,)`` unit face normals of ``mesh``
        ([`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]); when ``None``,
        each query's hit face normal is computed from its corners instead, to the same value.
        [`Trimesh.face_normals`][ordito.mesh.Trimesh.face_normals] has them cached.

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` face normals at the closest triangle for each query. A query with no face
        within ``max_dist`` reports the first face's normal; use
        [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] directly, whose
        ``triangle_id`` is ``-1`` there, when a miss has to be detected. A mesh with no faces
        reports an all-NaN normal for every query, matching
        [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]'s own zero-face
        convention.

    Raises
    ------
    RuntimeError
        If ``mesh``, ``points`` and ``face_normals`` are not all on one device.

    See Also
    --------
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
    """
    require_same_device(mesh=mesh, points=points, face_normals=face_normals)
    device = points.device
    m = points.size
    if m == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device)
    if mesh.indices.size == 0:
        # Every query misses a mesh with no faces, and the kernel below maps a miss to face 0 -- of
        # a face_normals array that has no face 0, an out-of-bounds read that segfaults rather than
        # raising. Match closest_point_on_mesh's own zero-face convention instead.
        nan = float("nan")
        return _launch.full(m, wp.vec3(nan, nan, nan), dtype=wp.vec3, device=device)

    if max_dist is None:
        max_dist = math.inf
    out_normals = _launch.empty(m, dtype=wp.vec3, device=device)
    if face_normals is None:
        # No table: the hit face's normal is formed in the query's thread, the same value the
        # per-face table would hold.
        _launch.launch(
            kernel_proximity.normals_at_closest_faces_computed,
            dim=m,
            inputs=[mesh.id, points, wp.float32(max_dist), mesh.points, mesh.indices, out_normals],
            device=device,
        )
        return out_normals
    _launch.launch(
        kernel_proximity.normals_at_closest_faces,
        dim=m,
        inputs=[mesh.id, points, wp.float32(max_dist), face_normals, out_normals],
        device=device,
    )
    return out_normals


def signed_distance_on_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    sign_mode: Literal["parity", "winding"] = "parity",
    n_sample: int = 5,
    perturbation_scale: float = 0.1,
    accuracy: float = 2.0,
    winding_threshold: float = 0.5,
    mesh: wp.Mesh | None = None,
) -> wp.array[wp.float32]:
    """
    Signed distance from each query point to a triangle mesh (Warp SDF convention).

    Distances follow Warp's signed-distance field convention:

    * Points **outside** the mesh have **positive** distance.
    * Points **inside** have **negative** distance.
    * Points within [`TOLERANCE_MERGE`][ordito.constants.TOLERANCE_MERGE] of the surface
      return positive unsigned distance.

    Trimesh ``signed_distance`` uses the opposite sign; negate its output to compare.
    See also [`contains_points`][ordito.ray.contains_points] (inside iff signed distance is
    negative, except on the on-surface tolerance band).

    The **unsigned** distance is identical in both ``sign_mode`` values — only the sign differs.

    !!! note "Choosing a `sign_mode`"

        ``"parity"`` (default) uses ``wp.mesh_query_point_sign_parity``: it casts ``n_sample``
        perturbed rays and votes on the crossing parity. Exact on a watertight mesh, cheap, but
        it has no principled answer on an open or holed surface — a ray that escapes through a
        hole flips the verdict.

        ``"winding"`` uses ``wp.mesh_query_point_sign_winding_number``, which evaluates the
        *generalized winding number* on the mesh BVH (a Barnes-Hut style traversal governed by
        ``accuracy``) and compares it against ``winding_threshold``. This is the
        Jacobson et al. robust inside/outside criterion and it degrades gracefully on
        non-watertight input, which is why it is the mode to reach for on raw scan data.

        The two modes agree on watertight meshes, but on a surface with an open patch ``"winding"``
        reproduces the exact generalized winding number's sign far more often than ``"parity"``
        does. The cost is a slower query and a larger ``wp.Mesh``: ``support_winding_number=True``
        stores a solid-angle expansion per BVH node, using noticeably more device memory.

        ``"winding"`` is still much cheaper than thresholding
        [`winding_number`][ordito.proximity.winding_number] yourself, because that sums the exact
        solid angle over *every* face for *every* query rather than using the BVH's Barnes-Hut
        traversal, and the gap widens with the face count. Reach for
        [`winding_number`][ordito.proximity.winding_number] only when you need the winding
        *value* — Warp exposes no builtin for the approximated value, only its sign.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index array.
        The internally built ``wp.Mesh`` aliases these buffers rather than copying them;
        do not mutate them for the duration of the call.
    points
        ``(m,)`` query positions in space.
    max_dist
        Maximum search radius per query. When ``None``, the search is
        unbounded, which finds the same closest point as any radius at least as long as the
        diagonal of the box enclosing the mesh and the query points.
    sign_mode
        ``"parity"`` (default) for ray-parity sign, ``"winding"`` for the generalized
        winding-number sign. See the note above.
    n_sample
        Perturbed rays for parity voting (off-triangle sign branch). ``"parity"`` only.
    perturbation_scale
        Uniform perturbation scale for parity rays. ``"parity"`` only.
    accuracy
        Barnes-Hut accuracy for the winding-number traversal: a node is expanded when the query
        point is within ``accuracy`` times the node's radius, so larger values are more accurate
        and slower. ``"winding"`` only; Warp's default is ``2.0``.
    winding_threshold
        Winding number above which a point counts as inside. ``"winding"`` only; ``0.5`` is the
        standard choice for a once-wound closed surface.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the clone and BVH
        build. With ``sign_mode="winding"`` it must have been built with
        ``support_winding_number=True``; without that flag the winding builtin silently answers with
        ray parity, so such a mesh is refused.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` signed distances.

    Raises
    ------
    ValueError
        If ``sign_mode`` is not ``"parity"`` or ``"winding"``, or if ``mesh`` is supplied with
        ``sign_mode="winding"`` but was built without ``support_winding_number=True``.
    RuntimeError
        If ``vertices``, ``faces``, ``points`` and ``mesh`` are not all on one device.

    See Also
    --------
    [`winding_number`][ordito.proximity.winding_number]
    [`contains_points`][ordito.ray.contains_points]
    """
    require_same_device(vertices=vertices, faces=faces, points=points, mesh=mesh)
    if sign_mode not in ("parity", "winding"):
        raise ValueError(f"sign_mode must be 'parity' or 'winding', got {sign_mode!r}")

    device = vertices.device
    m = points.size
    n_faces = faces.size // 3
    if m == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)
    if n_faces == 0:
        return _launch.full(m, float("inf"), dtype=wp.float32, device=device)

    if mesh is not None and sign_mode == "winding" and not mesh.support_winding_number:
        # Without the per-node solid-angle expansion the winding builtin silently degrades to ray
        # parity, so an unflagged mesh is refused; the parity mode accepts any mesh.
        raise ValueError(
            "sign_mode='winding' needs a mesh built with wp.Mesh(support_winding_number=True); "
            "the supplied mesh was not. Rebuild it with the flag, omit mesh=, or use "
            "sign_mode='parity'."
        )
    if mesh is None:
        require_nonempty_mesh(faces, "signed_distance_on_mesh")
        # The winding-number builtin silently degrades to ray parity unless the mesh carries the
        # per-node solid-angle expansion, so the flag is bound to sign_mode here rather than
        # exposed.
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
        mesh = wp.Mesh(
            points=vertices, indices=faces, support_winding_number=sign_mode == "winding"
        )
    if max_dist is None:
        max_dist = math.inf
    out_distance = _launch.empty(m, dtype=wp.float32, device=device)
    if sign_mode == "winding":
        _launch.launch(
            kernel_proximity.signed_distance_on_mesh_winding,
            dim=m,
            inputs=[
                mesh.id,
                points,
                wp.float32(max_dist),
                wp.float32(accuracy),
                wp.float32(winding_threshold),
                out_distance,
            ],
            device=device,
        )
        return out_distance
    _launch.launch(
        kernel_proximity.signed_distance_on_mesh,
        dim=m,
        inputs=[
            mesh.id,
            points,
            wp.float32(max_dist),
            wp.int32(n_sample),
            wp.float32(perturbation_scale),
            out_distance,
        ],
        device=device,
    )
    return out_distance


def signed_distance_grid(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    voxel_size: float | None = None,
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
    pad: int = 2,
    sign_mode: Literal["parity", "winding"] = "parity",
    mesh: wp.Mesh | None = None,
) -> tuple[odt.Array3dFloat32, tuple[wp.vec3, wp.vec3]]:
    """
    Sample the signed distance to a mesh on a regular lattice, as a field and the box it spans.

    The bridge from a surface to a **level set**, and the missing half of the implicit round trip:
    [`ordito.voxels.to_field`][ordito.voxels.to_field] already gives
    [`ordito.levelset.marching_cubes`][ordito.levelset.marching_cubes] an
    *occupancy* lattice, but occupancy is ``0`` or ``1`` and thresholding it at anything other than
    ``0.5`` does not move the surface anywhere. A distance field does, which is what makes
    ``marching_cubes(*signed_distance_grid(...), iso=d)`` an offset surface at distance ``d`` --
    see [`ordito.levelset.offset_mesh`][ordito.levelset.offset_mesh], the named entry point for
    it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    voxel_size
        Lattice spacing, isotropic. ``None`` takes
        [`ordito.voxels.resolve_voxel_grid`][ordito.voxels.resolve_voxel_grid]'s default of 1 % of
        the bounding-box diagonal, which is this package's one definition of an unspecified grid.
    bounds
        ``(lower, upper)`` box to sample, **before** padding. ``None`` uses the mesh's own
        axis-aligned box, which is what an offset wants -- an inward offset needs no more, and an
        outward one needs ``pad`` to cover it.
    pad
        Cells of margin added on every side, so the lattice extends ``pad * voxel_size`` beyond the
        box. Two is enough for the surface itself to be enclosed; an **outward offset of ``d``
        needs ``pad >= d / voxel_size + 1``** or its level set is clipped by the lattice boundary.
    sign_mode
        How the sign is decided, forwarded to
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]: ``"parity"`` counts
        ray crossings, ``"winding"`` sums solid angles and is the one that survives a mesh with open
        rims.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the build. Forwarded
        as-is, so ``signed_distance_on_mesh``'s rule applies unchanged: with ``sign_mode="winding"``
        it must have been built with ``support_winding_number=True``.

    Returns
    -------
    field : odt.Array3dFloat32
        ``(nx, ny, nz)`` signed distances, negative inside. Exactly the first argument
        [`ordito.levelset.marching_cubes`][ordito.levelset.marching_cubes] takes.
    bounds : tuple[wp.vec3, wp.vec3]
        The ``(lower, upper)`` corners the lattice actually spans, padded and snapped so the spacing
        is exactly ``voxel_size`` on every axis. Pass it straight through as that function's
        ``bounds``.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, ``pad`` is negative, ``faces`` is empty, or ``mesh`` is
        supplied with ``sign_mode="winding"`` but was built without ``support_winding_number=True``.
    RuntimeError
        If ``vertices``, ``faces`` and ``mesh`` are not all on one device.

    Examples
    --------
    ```python
    field, box = od.proximity.signed_distance_grid(v, f, voxel_size=0.05, pad=4)
    shell_v, shell_f = od.levelset.marching_cubes(field, 0.1, bounds=box)
    ```

    Notes
    -----
    **The whole lattice is sampled, so size it deliberately**: the field costs
    ``4 * nx * ny * nz`` bytes and one closest-point query per sample, which is 16.7 M queries and
    67 MB at ``256 ** 3``. There is no narrow band, and that is deliberate rather than missing --
    ``signed_distance_on_mesh`` reports ``+max_dist`` for a query that finds no face within the
    limit, so a banded field would carry a **positive** value deep inside the solid and silently
    invert the level set. Reduce the resolution instead.

    The lattice is a *corner* lattice: ``field[0, 0, 0]`` sits exactly on the returned ``lower``.
    That is [`ordito.voxels.grid_points`][ordito.voxels.grid_points]'s convention and
    ``marching_cubes``'s, and it is **not** the voxel-centre convention the rest of
    [`ordito.voxels`][ordito.voxels] uses.

    See Also
    --------
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]
        The per-query form this samples, and where the sign conventions are documented.
    [`ordito.levelset.offset_mesh`][ordito.levelset.offset_mesh]
        What to call instead when the answer wanted is the offset surface rather than the field.
    [`ordito.voxels.to_field`][ordito.voxels.to_field]
        The occupancy lattice, when a binary inside test is all that is needed.
    """
    require_same_device(vertices=vertices, faces=faces, mesh=mesh)
    if faces.size == 0:
        raise ValueError("signed_distance_grid needs at least one face")
    shape, box = signed_distance_lattice(vertices, voxel_size, bounds=bounds, pad=pad)
    samples = od.voxels.grid_points(shape, bounds=box, device=vertices.device)
    distances = signed_distance_on_mesh(vertices, faces, samples, sign_mode=sign_mode, mesh=mesh)
    return odt.as_array3d(distances.reshape(shape), wp.float32), box


def signed_distance_lattice(
    vertices: wp.array[wp.vec3],
    voxel_size: float | None = None,
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
    pad: int = 2,
) -> tuple[tuple[int, int, int], tuple[wp.vec3, wp.vec3]]:
    """
    Shape and bounds of the corner lattice a signed-distance grid samples, without sampling it.

    The lattice is exactly [`signed_distance_grid`][ordito.proximity.signed_distance_grid]'s,
    for a caller that evaluates the field somewhere other than on every node -- a sparse level-set
    extraction that queries only the cells near the surface, as
    [`ordito.levelset.offset_mesh`][ordito.levelset.offset_mesh] does on a large lattice -- and
    must land on exactly the nodes the dense field would have.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions; only their box is read, and only when ``bounds``
        is ``None`` or ``voxel_size`` is.
    voxel_size, bounds, pad
        As in [`signed_distance_grid`][ordito.proximity.signed_distance_grid].

    Returns
    -------
    shape : tuple[int, int, int]
        ``(nx, ny, nz)`` samples per axis, each at least 2.
    bounds : tuple[wp.vec3, wp.vec3]
        The ``(lower, upper)`` corners the lattice spans, padded and snapped so the spacing is
        exactly ``voxel_size`` on every axis.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive or ``pad`` is negative.
    """
    if pad < 0:
        raise ValueError("pad must be non-negative")
    spacing, _origin = od.voxels.resolve_voxel_grid(
        vertices, voxel_size, None, caller="signed_distance_grid"
    )

    lower, upper = bounds if bounds is not None else od.bounds.aabb(vertices)
    margin = float(pad) * spacing
    lower = wp.vec3(*(component - margin for component in odt.vec3_floats(lower)))
    # One sample per spacing, and at least the two marching cubes needs to have a cell at all.
    shape_x, shape_y, shape_z = (
        max(2, math.floor((float(upper[axis]) + margin - float(lower[axis])) / spacing) + 1)
        for axis in range(3)
    )
    shape = (shape_x, shape_y, shape_z)
    snapped_upper = wp.vec3(
        *(float(lower[axis]) + (shape[axis] - 1) * spacing for axis in range(3))
    )
    return shape, (lower, snapped_upper)


def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    tiled: bool = True,
) -> wp.array[wp.float32]:
    """
    Generalized winding number at each query point.

    Sums the signed solid angle subtended by each oriented triangle. For a
    closed, consistently oriented watertight mesh, interior points have
    winding number near ``1`` and exterior points near ``0``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index array.
    points
        ``(m,)`` query positions in space.
    tiled
        When ``True`` (default), sum solid angles with the face list partitioned across
        threads: one thread per ``(query, face slice)`` walks a strided slice of
        [`ITEMS_PER_QUERY_SLICE`][ordito.proximity.ITEMS_PER_QUERY_SLICE] faces and accumulates
        one ``wp.atomic_add`` per slice, so the summation order is nondeterministic and the result
        can differ in the last float32 digits between runs. When ``False``, each query thread
        loops over all faces serially — orders of magnitude slower on large meshes,
        but the fixed left-to-right summation makes it the exact-sum reference.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` winding numbers.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``points`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces, points=points)
    device = points.device
    n_queries = points.size
    n_faces = faces.size // 3
    if n_queries == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)
    if n_faces == 0:
        return _launch.zeros(n_queries, dtype=wp.float32, device=device)

    out_winding = (
        _launch.zeros(n_queries, dtype=wp.float32, device=device)
        if tiled
        else _launch.empty(n_queries, dtype=wp.float32, device=device)
    )
    if tiled:
        n_face_slices = max(1, (n_faces + ITEMS_PER_QUERY_SLICE - 1) // ITEMS_PER_QUERY_SLICE)
        kernel, dim = kernel_proximity.WINDING_NUMBER_TILED.launch_shape(n_queries, n_face_slices)
        _launch.launch(
            kernel,
            dim=dim,
            inputs=[
                vertices,
                faces,
                wp.int32(n_faces),
                wp.int32(n_face_slices),
                points,
                out_winding,
            ],
            device=device,
        )
    else:
        _launch.launch(
            kernel_proximity.winding_number,
            dim=n_queries,
            inputs=[vertices, faces, wp.int32(n_faces), points, out_winding],
            device=device,
        )
    return out_winding


def query_mesh_aabb_with_offsets(
    mesh: wp.Mesh,
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    *,
    max_hits: int = 16,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Low-level mesh AABB query with per-query axis-aligned bounds.

    For each query primitive ``k``, tests intersection of ``[query_lower[k],
    query_upper[k]]`` against every triangle in ``mesh`` via ``wp.mesh_query_aabb``.
    At most ``max_hits`` candidate face indices are recorded per query.

    Requires the default Warp mesh BVH backend; ``bvh_constructor="cubql"`` meshes
    do not support AABB queries.

    Parameters
    ----------
    mesh
        Target ``warp.Mesh`` built with the default BVH backend.
    query_lower
        ``(m,)`` lower corners of the query boxes, on the target device.
    query_upper
        ``(m,)`` upper corners of the query boxes.
    max_hits
        Maximum candidate faces recorded per query. Hits past this cap are dropped, so the
        result is a bounded sample rather than the full candidate set when a query straddles
        more than ``max_hits`` triangles.

    Returns
    -------
    candidate_indices_flat, offsets
        ``(n_hits,)`` candidate face indices and their ``(m + 1,)`` total-terminated ``offsets``,
        the prefix sum of per-query hit counts, ``n_hits == offsets[-1]``: query ``k`` owns
        ``candidate_indices_flat[offsets[k] : offsets[k + 1]]``. When ``m == 0`` the candidates are
        empty and ``offsets == [0]``.

    Raises
    ------
    ValueError
        If ``query_lower`` and ``query_upper`` have different lengths, or ``max_hits < 1``.
    RuntimeError
        If ``mesh``, ``query_lower`` and ``query_upper`` are not all on one device.
    """
    require_same_device(mesh=mesh, query_lower=query_lower, query_upper=query_upper)
    device = query_lower.device
    m = query_lower.size
    if query_upper.size != m:
        raise ValueError("query_lower and query_upper must have the same length")
    if max_hits < 1:
        raise ValueError("max_hits must be >= 1")

    if m == 0:
        return _launch.empty(0, dtype=wp.int32, device=device), _launch.zeros(
            1, dtype=wp.int32, device=device
        )

    # The counts are written behind the leading zero of the ``m + 1`` offsets buffer and scanned
    # there in place, so no separate count buffer is allocated; an all-zero count scans to all-zero
    # offsets, which is exactly what the empty case wants to return.
    offsets = _launch.zeros(m + 1, dtype=wp.int32, device=device)
    hit_counts = odt.as_dense(offsets[1:])
    _launch.launch(
        kernel_proximity.query_mesh_aabb_count,
        dim=m,
        inputs=[query_lower, query_upper, mesh.id, wp.int32(max_hits), hit_counts],
        device=device,
    )
    _launch.array_scan(hit_counts, out_array=hit_counts, inclusive=True)
    # One 4-byte read of the total sizes the candidate buffer.
    total_hits = int(read_scalar(offsets))
    if total_hits == 0:
        return _launch.empty(0, dtype=wp.int32, device=device), offsets

    candidate_indices_flat = _launch.empty(total_hits, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_proximity.query_mesh_aabb_neighbors,
        dim=m,
        inputs=[
            query_lower,
            query_upper,
            mesh.id,
            wp.int32(max_hits),
            offsets,
            candidate_indices_flat,
        ],
        device=device,
    )

    return candidate_indices_flat, offsets


def containing_faces_2d(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], points: wp.array[wp.vec2]
) -> wp.array[wp.int32]:
    """
    For each 2D query point, the triangle of a planar triangulation that contains it.

    Point location in the plane -- the primitive an inverse UV lookup needs, and the counterpart of
    [`ordito.texture`][]'s forward direction: ``remap_attribute_from_uv`` *samples an image* at a
    UV coordinate, where this answers which triangle of a UV atlas a coordinate falls in, and so
    which surface point it corresponds to. Pair it with
    [`points_to_barycentric`][ordito.triangles.points_to_barycentric] on the returned face to
    finish the inverse map.

    The triangulation must not overlap itself -- a UV atlas, a Delaunay triangulation, or the
    ``xy`` projection of a height field. Where triangles do overlap, each query gets one of the
    containing faces and which one is unspecified.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` planar vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    points
        ``(m,)`` planar query positions.

    Returns
    -------
    wp.array[wp.int32]
        ``(m,)`` containing-triangle index per query on ``vertices.device``, or ``-1`` where the
        query lies outside the triangulation. A query exactly on a shared edge is inside *both* its
        triangles and which one is returned is not specified.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``points`` are not all on one device.

    Notes
    -----
    Two stages: a closest-point query against the triangulation lifted to the ``z = 0`` plane picks
    a candidate face -- sufficient because a point inside any triangle is at distance zero from it,
    so the *closest* triangle contains it whenever one does -- and a barycentric sign test on the
    query's own coordinates decides. That reuses the BVH ``wp.Mesh`` already builds, at the cost of
    one lifted ``wp.vec3`` copy of ``vertices``.

    The second stage is not redundant: deciding on the query radius alone misclassifies some
    queries, because an in-plane point's closest-point distance is not exactly zero in
    ``float32`` and no radius threshold separates every interior point from every exterior one.
    The barycentric test is much sharper, so the radius is only a search bound and there is no
    tolerance to tune.

    Building that BVH happens on every call, so a caller locating several point sets against the
    same triangulation pays for it each time -- there is no prebuilt-index entry point.

    See Also
    --------
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
    [`points_to_barycentric`][ordito.triangles.points_to_barycentric]
    [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv]
    """
    require_same_device(vertices=vertices, faces=faces, points=points)
    device = vertices.device
    m = points.size
    n_faces = faces.size // 3
    if m == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)
    if n_faces == 0:
        return _launch.full(m, -1, dtype=wp.int32, device=device)

    lifted = _launch.empty(vertices.size, dtype=wp.vec3, device=device)
    _launch.map(kernel_array.lift_vec2, vertices, wp.float32(0.0), out=lifted)
    # One readback: the search radius has to be in the triangulation's own units and nothing else
    # knows its scale.
    search_radius = _CONTAINMENT_SEARCH_SCALE * od.bounds.enclosing_diagonal(lifted)
    require_nonempty_mesh(faces, "containing_faces_2d")
    # Both buffers are local to this call (``lifted`` is built above), so no copy is needed.
    mesh = wp.Mesh(points=lifted, indices=faces)
    out_face = _launch.empty(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_proximity.face_containing_point_2d,
        dim=m,
        inputs=[
            mesh.id,
            vertices,
            faces,
            points,
            wp.float32(search_radius),
            _CONTAINMENT_BARYCENTRIC_EPS,
            out_face,
        ],
        device=device,
    )
    return out_face
