"""Mesh diagnostics: topological/geometric validity predicates and their per-element masks."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import (
    read_scalar,
    read_values,
    require_nonempty_mesh,
    require_same_device,
    require_valid_faces,
)
from triwarp.constants import INDEX_RADIX_PAIR
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import array as kernel_array
from triwarp.kernels import intersection as kernel_intersection
from triwarp.kernels import triangles as kernel_triangles
from triwarp.kernels import validation as kernel_validation
from triwarp.kernels.algorithms import connected_components as kernel_connected_components


def is_edge_manifold(
    faces: wp.array[wp.int32],
    allow_boundary_edges: bool = True,
    *,
    n_vertices: int | None = None,
    validate: bool = True,
) -> bool:
    """
    Whether every undirected mesh edge is shared by a manifold number of faces.

    Counts how many faces share each undirected edge. With ``allow_boundary_edges=True`` an edge is
    manifold when it belongs to one or two faces; with ``allow_boundary_edges=False`` it must belong
    to exactly two faces (no boundary or non-manifold edges), matching Open3D's ``IsEdgeManifold``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    allow_boundary_edges
        When ``True`` (default) boundary edges (used by a single face) are allowed. When ``False``
        every edge must be shared by exactly two faces.
    n_vertices
        Optional vertex count (the edge-hash base and the upper bound checked). When ``None``, the
        keys pack against a bound on every ``int32`` and only the negative half is checked.
    validate
        Whether to range-check the face indices. The check rides on the pass that packs the keys
        and shares the verdict's readback. Pass ``False`` only where both bounds are structurally
        guaranteed -- a face buffer this package produced itself, or one an entry point has
        already validated.

    Returns
    -------
    bool
        ``True`` when all edges satisfy the manifold condition. Vacuously ``True`` for an empty
        mesh.

    Raises
    ------
    ValueError
        If ``validate`` is ``True`` and a face index is negative or reaches the vertex count.

    See Also
    --------
    [`is_vertex_manifold`][triwarp.validation.is_vertex_manifold]
    [`is_watertight`][triwarp.validation.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_edge_manifold``; libigl ``is_edge_manifold``
    corresponds to the ``allow_boundary_edges=True`` case.

    Deliberately **not** ``all(edge_manifold_mask(faces))``, unlike the other ``is_*`` / ``*_mask``
    pairs here: this counts the edges in a hash table and reads it in place, where the mask needs
    each halfedge's own count and so sorts the keys. Delegating would add the sort to the cheap
    path.
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    device = faces.device
    # ``flags[0]`` is the verdict and ``flags[1]`` the range check, read back together.
    flags = _launch.zeros(2, dtype=wp.int32, device=device)
    # The keys come straight off ``faces``: packing them there costs no more than hashing the
    # sorted edge rows, and the range check rides on the same pass, so validating costs no
    # reduction and no readback of its own; unvalidated, its flag is simply not read.
    keys = _launch.empty(3 * n_faces, dtype=wp.uint64, device=device)
    _launch.launch(
        kernel_validation.face_edge_keys_checked,
        dim=n_faces,
        inputs=[
            faces,
            wp.uint64(INDEX_RADIX_PAIR if n_vertices is None else n_vertices),
            twt.dtype_max(wp.int32) if n_vertices is None else n_vertices,
        ],
        outputs=[keys, flags],
        device=device,
    )
    # Counted with ``grouping.hashed_occurrence_counts``, the table ``grouping.unique_1d`` builds,
    # and tested in place: the answer is
    # order-free and needs no per-edge output, so the compaction, the sort and the host read of
    # the unique count that sizes them are skipped.
    slot_counts = tw.grouping.hashed_occurrence_counts(keys)
    _launch.launch(
        kernel_validation.edge_share_count_violation,
        dim=slot_counts.size,
        inputs=[slot_counts, allow_boundary_edges, flags],
        device=device,
    )
    # One readback for both flags.
    violated, out_of_range = read_values(flags, 0, 2)
    if validate and out_of_range:
        bound = "" if n_vertices is None else f" below {n_vertices}"
        raise ValueError(f"is_edge_manifold: faces must be non-negative indices{bound}")
    return violated == 0


def edge_manifold_mask(
    faces: wp.array[wp.int32],
    allow_boundary_edges: bool = True,
    *,
    n_vertices: int | None = None,
    validate: bool = True,
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether all three of each face's undirected edges are edge-manifold.

    An edge is manifold when shared by one or two faces (``allow_boundary_edges=True``) or by
    exactly two faces (``allow_boundary_edges=False``); a face is flagged ``True`` only when all
    three of its edges qualify. This is a per-face collapse of libigl's ``BF`` (its per-corner
    ``is_edge_manifold`` matrix), and
    [`is_edge_manifold`][triwarp.validation.is_edge_manifold] is ``True`` iff every entry of
    this mask is ``True``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    allow_boundary_edges
        When ``True`` (default) boundary edges (used by a single face) count as manifold. When
        ``False`` every edge of a face must be shared by exactly two faces.
    n_vertices
        Optional vertex count (the edge-hash base). When ``None`` and ``validate`` is ``True``,
        inferred from ``faces`` by the same reduction that checks them; unvalidated, no bound is
        needed and none is inferred.
    validate
        Whether to range-check the edge indices before packing them. The check is a
        ``triwarp.reduce.minmax`` whose host readback serialises the device pipeline. Pass
        ``False`` only where both bounds are structurally guaranteed -- a face buffer this package
        produced itself, or one an entry point has already validated.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If ``validate`` is ``True`` and a face index is negative or reaches the vertex count.

    See Also
    --------
    [`is_edge_manifold`][triwarp.validation.is_edge_manifold]
    [`vertex_manifold_mask`][triwarp.validation.vertex_manifold_mask]
    """
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    # The share count of each edge is the length of its run of sorted keys, read in place: no
    # unique-edge table, inverse or count array, and no host read of the unique count.
    keys, order = _sorted_halfedge_keys(
        faces, _validated_vertex_bound(faces, n_vertices, validate, "edge_manifold_mask")
    )
    out_mask = _launch.full(n_faces, True, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_validation.sorted_run_manifold_mask,
        dim=keys.size,
        inputs=[keys, order, wp.bool(allow_boundary_edges), out_mask],
        device=device,
    )
    return out_mask


def _validated_vertex_bound(
    faces: wp.array[wp.int32], n_vertices: int | None, validate: bool, name: str
) -> int | None:
    """
    Resolve the edge-key radix of the two edge-manifold predicates, checking ``faces`` on request.

    What ``validate`` adds is one reduction over the face buffer, never over the ``(3F, 2)`` rows.
    Without ``n_vertices`` that reduction also supplies the radix, and only its negative half is
    informative; unvalidated and without ``n_vertices`` the radix stays ``None`` (so the keys pack
    against ``INDEX_RADIX_PAIR``) and nothing is reduced.
    """
    if validate:
        if n_vertices is None:
            return tw.array.index_bound(faces, require_non_negative=True)
        require_valid_faces(faces, n_vertices, name)
    return n_vertices


def is_vertex_manifold(
    faces: wp.array[wp.int32],
    *,
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> bool:
    """
    Whether every referenced vertex has a single edge-connected fan of faces.

    A vertex is manifold when the faces incident to it form one connected group under
    "shares an edge through this vertex" adjacency; two fans meeting only at the vertex (a bow-tie)
    are non-manifold. Unreferenced vertices in ``[0, max(faces)]`` are treated as non-manifold,
    matching libigl's ``is_vertex_manifold`` and Open3D's ``IsVertexManifold``.

    The check builds the corner graph (node ``3 * f + k`` per face corner), links the corresponding
    corners of edge-adjacent faces, and requires all corners of each vertex to fall in one connected
    component.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    face_adjacency
        ``(m, 2)`` precomputed face-adjacency pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. Must be given together with
        ``face_adjacency_edges``.
    face_adjacency_edges
        ``(m, 2)`` shared-edge vertex pairs aligned with ``face_adjacency``
        (``return_edges=True``). Must be given together with ``face_adjacency``.
    n_vertices
        Optional length of the vertex buffer ``faces`` indexes, which must be at least
        ``max(faces) + 1`` and is not checked. It sizes the working tables only and does not change
        the answer: vertices past ``max(faces)`` are still not considered. When ``None``,
        ``max(faces) + 1`` is inferred with a device-host sync.

    Returns
    -------
    bool
        ``True`` when every referenced vertex is manifold. Vacuously ``True`` for an empty mesh.

    Raises
    ------
    ValueError
        If exactly one of ``face_adjacency`` / ``face_adjacency_edges`` is provided.
    RuntimeError
        If ``faces``, ``face_adjacency`` and ``face_adjacency_edges`` are not all on one device.

    See Also
    --------
    [`vertex_manifold_mask`][triwarp.validation.vertex_manifold_mask]
    [`is_edge_manifold`][triwarp.validation.is_edge_manifold]
    [`is_watertight`][triwarp.validation.is_watertight]

    Notes
    -----
    Corner adjacency is derived from
    [`face_adjacency`][triwarp.adjacency.face_adjacency], which pairs faces across edges shared by
    exactly two faces; on edge-non-manifold meshes (an edge shared by three or more faces) read the
    result together with [`is_edge_manifold`][triwarp.validation.is_edge_manifold]. Equivalent
    to libigl ``is_vertex_manifold``. The output is sized to ``max(faces) + 1`` so unreferenced
    vertices in that range count as non-manifold; use
    [`vertex_manifold_mask`][triwarp.validation.vertex_manifold_mask] for a per-vertex flag
    sized to a caller-provided vertex buffer.
    """
    require_same_device(
        faces=faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
    )
    # Checked before the empty-mesh guard so a caller passing only one half of the pair is told
    # about it whatever the mesh is; resolved after it, because on an empty buffer the resolve is
    # two empty tables nothing reads. Every wrapper taking this pair does it in this order.
    tw.adjacency.require_paired_adjacency(face_adjacency, face_adjacency_edges)
    if faces.size // 3 == 0:
        return True
    if n_vertices is None:
        # The corner graph derived below trusts this bound, so it carries the range check the
        # edge-key packing would otherwise have made itself.
        n_vertices = tw.array.index_bound(faces, require_non_negative=face_adjacency is None)
    violation = _launch.zeros(1, dtype=wp.int32, device=faces.device)
    if face_adjacency is None:
        parents = _corner_parents_from_keys(
            faces, _sorted_halfedge_keys(faces, n_vertices), False, None
        )
    else:
        assert face_adjacency_edges is not None
        parents = _corner_parents_from_adjacency(faces, face_adjacency, face_adjacency_edges)
    _vertex_manifold_check(faces, n_vertices, parents, None, violation)
    return int(read_scalar(violation)) == 0


def _corner_parents_from_adjacency(
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32,
    face_adjacency_edges: twt.Array2dInt32,
) -> wp.array[wp.int32]:
    """
    Union-find forest of the corner graph over a face-adjacency table.

    Two corner-graph edges per adjacency row, one per shared-edge endpoint, formed in the thread
    and hooked by ``connected_components``' pair pre-hook and hook: the forest
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    flattens, left unflattened for the fused labelling pass that follows.
    """
    device = faces.device
    parents = tw.array.arange(faces.size // 3 * 3, device=device)
    m = int(face_adjacency.shape[0])
    if m == 0:
        return parents
    for kernel in (
        kernel_validation.adjacency_corner_prehook,
        kernel_validation.adjacency_corner_hook,
    ):
        _launch.launch(
            kernel,
            dim=m,
            inputs=[faces, face_adjacency, face_adjacency_edges, parents],
            device=device,
        )
    return parents


def vertex_manifold_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    """
    Per-vertex flag: whether each vertex has a single edge-connected fan of faces.

    A vertex is manifold when the faces incident to it form one connected group under
    "shares an edge through this vertex" adjacency; two fans meeting only at the vertex (a bow-tie)
    are non-manifold. Unreferenced vertices are flagged ``False``, matching libigl's
    ``is_vertex_manifold`` per-vertex output ``B`` and Open3D's ``GetNonManifoldVertices``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions. Only the length is used; it sets the output size so that
        trailing unreferenced vertices are reported (``False``).
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_vertices,)`` mask on ``faces.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`is_vertex_manifold`][triwarp.validation.is_vertex_manifold]
    [`edge_manifold_mask`][triwarp.validation.edge_manifold_mask]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_vertices = vertices.size
    if faces.size // 3 == 0:
        return _launch.zeros(n_vertices, dtype=wp.bool, device=faces.device)
    # The corner graph straight off the sorted halfedge keys, with its pair check off: no thread
    # indexes the violation flag, so none is allocated.
    parents = _corner_parents_from_keys(
        faces, _sorted_halfedge_keys(faces, n_vertices), False, None
    )
    mask = _launch.zeros(n_vertices, dtype=wp.bool, device=faces.device)
    _vertex_manifold_check(faces, n_vertices, parents, mask, None)
    return mask


def is_self_intersecting(mesh: wp.Mesh, *, max_triangle_collisions: int = 32) -> bool:
    """
    Whether any two non-adjacent triangles of the mesh intersect.

    Broad phase queries each triangle's AABB against ``mesh``'s BVH for candidate overlaps;
    narrow phase runs Moller's interval test on each candidate pair, skipping pairs that share a
    vertex. Mirrors Open3D's ``IsSelfIntersecting`` (AABB pre-test followed by a triangle-triangle
    test on non-neighbouring faces). Coplanar and merely touching pairs are **not** intersections
    -- contact alone does not count.

    Parameters
    ----------
    mesh
        Mesh to test; ``mesh.points`` and ``mesh.indices`` are used directly, so its BVH is
        always in sync with the triangles being tested (a separately-passed vertex/face pair
        could otherwise be misaligned with a caller's ``mesh``).
    max_triangle_collisions
        Broad-phase candidate cap per query triangle. Raise this for meshes with many triangles
        packed into overlapping bounding boxes.

    Returns
    -------
    bool
        ``True`` when at least one intersecting non-adjacent triangle pair exists. ``False`` for
        meshes with fewer than two faces.

    See Also
    --------
    [`face_self_intersecting_mask`][triwarp.validation.face_self_intersecting_mask]
    [`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh]
    [`is_watertight`][triwarp.validation.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_self_intersecting``.
    """
    found = _launch.zeros(1, dtype=wp.bool, device=mesh.points.device)
    _mark_self_intersections(mesh, mesh.points, mesh.indices, max_triangle_collisions, False, found)
    return bool(read_scalar(found))


def face_self_intersecting_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 32,
    mesh: wp.Mesh | None = None,
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether each triangle intersects some non-adjacent triangle.

    Broad phase builds a ``warp.Mesh`` and queries each triangle's AABB for candidate overlaps;
    narrow phase runs Moller's interval test on each candidate pair (skipping pairs that share a
    vertex), and both faces of every intersecting pair are flagged.
    [`is_self_intersecting`][triwarp.validation.is_self_intersecting] is ``True`` iff any entry of
    this mask is ``True``. A pair that is coplanar, or that touches without crossing, is not an
    intersection.

    **The narrow phase runs in ``float64`` on the ``float32`` vertices**, and it is load-bearing
    rather than caution. Widening is lossless, so the geometry is unchanged; what the precision buys
    is the decisions -- the sign of a plane distance and the overlap of two intervals. Audited face
    for face against an independent ``float64`` Moller implementation over seven self-intersecting
    surfaces, it is exact on five, and on the other two it matches a second library and differs from
    the reference by at most four faces, every one of them a vertex lying *on* the other triangle's
    plane -- a tolerance choice rather than a fact. In ``float32`` the same audit read dozens of
    false positives and false negatives on one surface alone. It costs a few percent of the call.

    Where a comparison against another library is wanted on that tangency class, compare **counts,
    not sets**: which faces sit exactly on a tangency is the part no two implementations agree on.
    And a library whose loader repairs connectivity first may cut the orientation-reversing seam of
    a non-orientable surface, leaving two coincident sheets where the surface had one -- after which
    every face along the cut genuinely does intersect. Establish that the load changed nothing
    before reading such a comparison.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_triangle_collisions
        Broad-phase candidate cap per query triangle. Raise this for meshes with many triangles
        packed into overlapping bounding boxes.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the BVH build the
        broad phase otherwise pays on every call. Purely an optimization, and not checked against
        ``vertices`` / ``faces``.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``. All-``False`` for meshes with fewer than two faces.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``mesh`` are not all on one device.

    See Also
    --------
    [`is_self_intersecting`][triwarp.validation.is_self_intersecting]
    """
    require_same_device(vertices=vertices, faces=faces, mesh=mesh)
    device = vertices.device
    n_faces = faces.size // 3
    mask = _launch.zeros(n_faces, dtype=wp.bool, device=device)
    if n_faces < 2:
        return mask

    if mesh is None:
        require_nonempty_mesh(faces, "face_self_intersecting_mask")
        mesh = wp.Mesh(points=vertices, indices=faces)

    _mark_self_intersections(mesh, vertices, faces, max_triangle_collisions, True, mask)
    return mask


def _mark_self_intersections(
    mesh: wp.Mesh,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_triangle_collisions: int,
    mark_faces: bool,
    out_marks: wp.array[wp.bool],
) -> None:
    """
    Broad and narrow phase of the self-intersection test, shared by the predicate and the mask.

    The two differ only in ``mark_faces``: the mask marks both faces of every intersecting
    candidate in ``out_marks``, the predicate raises ``out_marks[0]``. The broad phase queries each
    triangle's AABB against ``mesh``'s BVH once, writing at most ``max_triangle_collisions``
    candidates into a fixed-stride block per triangle, so no candidate count has to be read back
    to size a packed list; the narrow phase is Moller's interval test on every live slot, skipping
    pairs that share a vertex (``kernels/intersection.candidate_pair_intersects``), on
    ``vertices`` / ``faces``, which must be what ``mesh`` was built over.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions < 1``.
    """
    if max_triangle_collisions < 1:
        raise ValueError("max_triangle_collisions must be >= 1")
    n_faces = faces.size // 3
    if n_faces == 0:
        return
    device = vertices.device
    lower = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    upper = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_triangles.face_aabb_bounds,
        dim=n_faces,
        inputs=[vertices, faces, lower, upper],
        device=device,
    )
    targets = _launch.empty(n_faces * max_triangle_collisions, dtype=wp.int32, device=device)
    counts = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersection.collect_face_box_candidates,
        dim=n_faces,
        inputs=[mesh.id, lower, upper, max_triangle_collisions],
        outputs=[targets, counts],
        device=device,
    )
    _launch.launch(
        kernel_validation.self_intersection_marks,
        dim=n_faces * max_triangle_collisions,
        inputs=[vertices, faces, targets, counts, max_triangle_collisions, mark_faces, out_marks],
        device=device,
    )


def is_winding_consistent(faces: wp.array[wp.int32]) -> bool:
    """
    Whether every shared edge is traversed in opposite directions by its two faces.

    A mesh has consistent winding when, for each edge shared by two faces, the two faces list the
    edge's endpoints in opposite order (so their normals agree locally). This is a property of the
    current winding, unlike [`is_orientable`][triwarp.validation.is_orientable], which allows
    faces to be flipped. Matches [`trimesh.Trimesh.is_winding_consistent`][].

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when winding is consistent across all shared edges. Vacuously ``True`` for an empty
        mesh or a mesh with no shared edges.

    See Also
    --------
    [`edge_winding_consistent_mask`][triwarp.validation.edge_winding_consistent_mask]
    [`is_orientable`][triwarp.validation.is_orientable]
    [`is_volume`][triwarp.validation.is_volume]
    [`trimesh.Trimesh.is_winding_consistent`][]
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    # ``edge_winding_consistent_mask``'s per-pair test read straight off the sorted keys, so the
    # verdict needs no group table, no host read of its length and no reduction over a mask.
    keys, order = _sorted_halfedge_keys(faces, None)
    violation = _launch.zeros(1, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_validation.sorted_pair_winding_violation,
        dim=keys.size,
        inputs=[faces, None, keys, order, False, violation],
        device=faces.device,
    )
    return int(read_scalar(violation)) == 0


def edge_winding_consistent_mask(
    faces: wp.array[wp.int32], edges: twt.Array2dInt32 | None = None
) -> wp.array[wp.bool]:
    """
    Per shared-edge flag: whether an undirected edge's two faces traverse it in opposite directions.

    Built over the undirected edges shared by exactly two faces (edge groups of length two). An
    entry is ``True`` when the two directed half-edges are reversed (locally consistent normals);
    boundary and non-manifold edges have no entry.
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent] is ``True`` iff every
    entry is ``True``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    edges
        ``(3 * n_faces, 2)`` precomputed directed edges in
        [`faces_to_edges`][triwarp.edges.faces_to_edges] row order. Built from ``faces`` when
        ``None``.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_shared_edges,)`` mask on ``faces.device``. Empty when the mesh has no shared edges.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``edges`` are not all on one device.

    See Also
    --------
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]
    [`face_flip_mask`][triwarp.validation.face_flip_mask]
    """
    require_same_device(faces=faces, edges=edges)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    # Neither edge table is built when not given: the keys come straight off ``faces`` and the
    # kernel reads each directed edge from its face corner, the row ``faces_to_edges`` would hold.
    edge_groups = tw.grouping.group(_halfedge_keys(faces), 2)
    n_groups = int(edge_groups.shape[0])
    if n_groups == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    consistent = _launch.empty(n_groups, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_validation.edge_pair_winding_mask,
        dim=n_groups,
        inputs=[faces, edges, edge_groups, consistent],
        device=device,
    )
    return consistent


def face_orientation_bits(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], twt.Array2dInt32, wp.array[wp.int32], int]:
    """
    Per-face Z2 orientation bits (relative to each component seed) plus signed face-adjacency edges.

    ``orient[f]`` is ``0`` for a face that agrees with its connected component's seed and ``1`` for
    a face that must be flipped to agree — the flip mask consumed by
    [`face_flip_mask`][triwarp.validation.face_flip_mask] and
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]. ``m`` is the number of
    face-adjacency rows; when ``m == 0`` the returned ``signed_edges`` / ``signs`` are empty.
    Assumes ``n_faces > 0`` (callers guard).

    This is a lower-level primitive (the underlying orientation engine behind
    [`is_orientable`][triwarp.validation.is_orientable] and
    [`face_flip_mask`][triwarp.validation.face_flip_mask]) exposed for callers that
    need the raw flip bits together with the propagation edges, such as
    [`triwarp.repair.make_winding_consistent`][triwarp.repair.make_winding_consistent].

    The bits are solved, not propagated: ``orient`` is the Z2 potential of the signed
    face-adjacency graph, computed by a parity-carrying union-find
    ([`connected_component_parity_from_edges`]
    [triwarp.graph.connected_component_parity_from_edges]) in a fixed three launches. Cost
    therefore depends on the mesh's *size*, not on the diameter of its face-adjacency graph.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer. ``n_faces`` must be positive.

    Returns
    -------
    orient : wp.array[wp.int32]
        ``(n_faces,)`` flip bits (``0`` or ``1``) on ``faces.device``.
    signed_edges : twt.Array2dInt32
        ``(m, 2)`` face-adjacency pairs, one row per face-adjacency edge.
    signs : wp.array[wp.int32]
        ``(m,)`` Z2 sign per adjacency edge.
    m : int
        Number of face-adjacency rows.

    See Also
    --------
    [`is_orientable`][triwarp.validation.is_orientable]
    [`face_flip_mask`][triwarp.validation.face_flip_mask]
    """
    device = faces.device
    n_faces = faces.size // 3
    adjacency, adjacency_edges = tw.adjacency.face_adjacency(faces, return_edges=True)
    m = int(adjacency.shape[0])

    if m == 0:
        signed_edges = twt.empty_2d((0, 2), wp.int32, device=device)
        signs = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.zeros(n_faces, dtype=wp.int32, device=device), signed_edges, signs, m

    signed_edges = twt.empty_2d((m, 2), wp.int32, device=device)
    signs = _launch.empty(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_validation.build_signed_face_edges,
        dim=m,
        inputs=[faces, adjacency, adjacency_edges, signed_edges, signs],
        device=device,
    )

    # The flip bits *are* a Z2 potential over the signed face-adjacency graph, so a parity-carrying
    # union-find produces them directly in three launches with no host synchronization
    # ([`connected_component_parity_from_edges`]
    # [triwarp.graph.connected_component_parity_from_edges]). The bits are unchanged from the
    # edge-by-edge flood fill this replaces: that seeded ``0`` at each component's smallest face id
    # and XOR-ed signs outward, and the union-find's root is that same smallest id at that same
    # parity 0. What changes is the cost — the fill ran one launch per graph level, so a mesh whose
    # face adjacency is a long path (a ribbon) paid tens of thousands of launches and thousands of
    # readbacks for work bounded by a few milliseconds of bandwidth.
    # ``signed_edges`` holds face ids this function just built from ``adjacency`` (itself bounded
    # by ``n_faces`` by construction) and every sign is ``0`` or ``1``, so nothing is range-checked.
    orient = _solve_orientation(
        n_faces, kernel_connected_components.ecl_hook_parity, m, [signed_edges, signs], device
    )
    return orient, signed_edges, signs, m


def is_orientable(faces: wp.array[wp.int32]) -> bool:
    """
    Whether the faces admit a consistent orientation (allowing per-face flips).

    A mesh is orientable when each face can be assigned a flip bit so that every pair of
    edge-adjacent faces traverses their shared edge in opposite directions. This is a topological
    property independent of the current winding, matching Open3D's ``IsOrientable``.

    Each face-adjacency edge carries a Z2 flip bit; the check seeds one bit per connected component,
    propagates bits across adjacencies, then verifies that no adjacency violates its constraint (a
    contradiction means non-orientable).

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when a consistent orientation exists. Vacuously ``True`` for an empty mesh.

    See Also
    --------
    [`face_flip_mask`][triwarp.validation.face_flip_mask]
    [`is_watertight`][triwarp.validation.is_watertight]
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_orientable``. Unlike
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent], orientability allows
    individual faces to be flipped, so a consistently-orientable mesh with inconsistent winding
    still returns ``True``.
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    orient, keys, order = _orientation_bits_from_keys(faces)
    device = faces.device
    conflict = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_validation.sorted_pair_orientation_conflict,
        dim=keys.size,
        inputs=[faces, keys, order, orient, conflict],
        device=device,
    )
    return int(read_scalar(conflict, 0)) == 0


def face_flip_mask(faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a face must be flipped to make winding consistent within its patch.

    Runs the Z2 orientation propagation over the face-adjacency graph (seeding one arbitrary
    reference face per connected component) and returns ``True`` for every face whose winding
    disagrees with its component's seed. Applying these flips yields a consistently wound mesh, so
    this is the per-face flip mask consumed by
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]. A mesh that is already
    consistently wound yields an all-``False`` mask.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``. Empty for an empty mesh.

    See Also
    --------
    [`is_orientable`][triwarp.validation.is_orientable]
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`face_flipped_mask`][triwarp.parametrization.face_flipped_mask]

    Notes
    -----
    The reference orientation is arbitrary per connected component, so on a non-orientable patch the
    mask is still a best-effort flood-fill (matching ``trimesh.repair.fix_winding``).

    This is a different flip from
    [`face_flipped_mask`][triwarp.parametrization.face_flipped_mask], which is about the *UV
    domain*: this mask flags a triangle whose index order must be reversed to agree with its patch,
    a property of the 3D connectivity that ignores geometry entirely, while that one flags a
    triangle whose UV image has negative signed area. Neither implies the other.
    """
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    orient, _, _ = _orientation_bits_from_keys(faces)
    mask = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.map(kernel_array.greater, orient, wp.int32(0), out=mask)
    return mask


def _orientation_bits_from_keys(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Flip bits, with the signed edges formed straight off the sorted halfedge keys.

    The bits [`face_orientation_bits`][triwarp.validation.face_orientation_bits] returns.

    One signed edge per halfedge: each adjacency pair's row at its first member, a sign-``0``
    self-loop everywhere else, which the parity union-find skips and every orientation satisfies.
    The pair rows keep their adjacency order, so the bits are the ones the adjacency table gives,
    and neither that table nor the host read of its length is built. For callers that consume the
    bits, not the table: [`is_orientable`][triwarp.validation.is_orientable] and
    [`face_flip_mask`][triwarp.validation.face_flip_mask]. ``faces`` must be non-empty. The sorted
    keys are returned too, so a caller can re-form the same edges.
    """
    keys, order = _sorted_halfedge_keys(faces, None)
    # Every endpoint is a face id derived in the thread from a halfedge index, bounded by
    # ``n_faces`` by construction, and every sign is ``0`` or ``1``.
    orient = _solve_orientation(
        faces.size // 3,
        kernel_validation.sorted_pair_hook_parity,
        keys.size,
        [faces, keys, order],
        faces.device,
    )
    return orient, keys, order


def _solve_orientation(
    n_faces: int,
    hook: twt.Kernel,
    n_edges: int,
    hook_inputs: Sequence[object],
    device: wp.DeviceLike,
) -> wp.array[wp.int32]:
    """
    Z2 potential of a signed face graph: the parity union-find, returning the parity alone.

    [`connected_component_parity_from_edges`][triwarp.graph.connected_component_parity_from_edges]'s
    three launches with ``hook`` in place of its edge-list hook and no label table, which neither
    orientation caller reads. ``hook_inputs`` precede the packed words in ``hook``'s signature.
    """
    words = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_connected_components.ecl_init_parent_parity,
        dim=n_faces,
        inputs=[words],
        device=device,
    )
    if n_edges > 0:
        _launch.launch(hook, dim=n_edges, inputs=[*hook_inputs, words], device=device)
    orient = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_connected_components.ecl_flatten_parity,
        dim=n_faces,
        inputs=[words, None, orient],
        device=device,
    )
    return orient


def is_watertight(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    mesh: wp.Mesh | Callable[[], wp.Mesh] | None = None,
) -> bool:
    """
    Whether the mesh bounds a closed volume with no self-intersections.

    Follows Open3D's ``IsWatertight``: the mesh must be edge-manifold with no boundary edges,
    vertex-manifold, and free of self-intersections.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, forwarded to the
        self-intersection broad phase so its BVH is not rebuilt -- or a zero-argument callable
        returning one, called only once both manifold tests have passed, so a caller that caches
        the mesh builds it only when the answer needs it. Purely an optimization, and not checked
        against ``vertices`` / ``faces``.

    Returns
    -------
    bool
        ``True`` when the mesh is edge-manifold (no boundary), vertex-manifold, and not
        self-intersecting.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and a ``wp.Mesh`` ``mesh`` are not all on one device.

    See Also
    --------
    [`face_watertight_mask`][triwarp.validation.face_watertight_mask]
    [`is_edge_manifold`][triwarp.validation.is_edge_manifold]
    [`is_vertex_manifold`][triwarp.validation.is_vertex_manifold]
    [`is_self_intersecting`][triwarp.validation.is_self_intersecting]
    [`is_volume`][triwarp.validation.is_volume]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_watertight``. For the cheaper "every edge shared
    by exactly two faces" test (trimesh semantics) use
    [`is_edge_manifold`][triwarp.validation.is_edge_manifold] with
    ``allow_boundary_edges=False``.

    **This is not the predicate form of
    [`face_watertight_mask`][triwarp.validation.face_watertight_mask]**, despite the names: that
    mask is trimesh's per-face edge test, while this is Open3D's three-part composition, so
    ``all(face_watertight_mask(faces))`` is the *first* of the three conditions and not this
    function. Every other ``is_*`` / ``*_mask`` pair in this module does relate that way.
    """
    require_same_device(
        vertices=vertices, faces=faces, mesh=mesh if isinstance(mesh, wp.Mesh) else None
    )
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    # One sort of the halfedge keys serves the first two checks: the runs of equal keys are the
    # edge-share counts the edge test reads and the face pairs the vertex test's corner graph
    # links, so neither a face-adjacency table nor a second packing is built. Both tests raise one
    # flag and it is read once; only the self-intersection test, which needs a BVH, waits for it.
    # Reading the edge test's flag first, to skip the vertex test on an open mesh, costs a closed
    # mesh more than it saves an open one: the vertex test's launches then no longer overlap the
    # sort on the device.
    n_vertices = vertices.size
    violation = _launch.zeros(1, dtype=wp.int32, device=faces.device)
    parents = _corner_parents_from_keys(
        faces, _sorted_halfedge_keys(faces, n_vertices), True, violation
    )
    _vertex_manifold_check(faces, n_vertices, parents, None, violation)
    if int(read_scalar(violation)) != 0:
        return False
    if mesh is None:
        require_nonempty_mesh(faces, "is_watertight")
        mesh = wp.Mesh(points=vertices, indices=faces)
    elif not isinstance(mesh, wp.Mesh):
        mesh = mesh()
    return not is_self_intersecting(mesh)


def face_watertight_mask(
    faces: wp.array[wp.int32], *, n_vertices: int | None = None
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether all three of a face's undirected edges are shared by exactly two faces.

    This is [`edge_manifold_mask`][triwarp.validation.edge_manifold_mask] with
    ``allow_boundary_edges=False``: a face is ``True`` only when none of its edges is a boundary
    edge (used once) or a non-manifold edge (used three or more times). The faces that break
    watertightness (trimesh's ``broken_faces``) are ``flatnonzero(~face_watertight_mask(faces))``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Optional vertex count forwarded as the edge-hash base. When ``None``, inferred from
        ``faces`` with a device-host sync.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If a face index is negative or reaches the vertex count.

    See Also
    --------
    [`is_watertight`][triwarp.validation.is_watertight]
    [`edge_manifold_mask`][triwarp.validation.edge_manifold_mask]
    """
    return edge_manifold_mask(faces, allow_boundary_edges=False, n_vertices=n_vertices)


def is_volume(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, edges: twt.Array2dInt32 | None = None
) -> bool:
    """
    Whether the mesh is a valid closed volume with outward-facing normals.

    Follows [`trimesh.Trimesh.is_volume`][]: the mesh must be watertight (every undirected edge
    shared by exactly two faces), winding-consistent, and enclose a positive signed volume (normals
    point outward). A mesh with inward-facing normals encloses a negative signed volume and is
    reported ``False``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    edges
        ``(3 * n_faces, 2)`` precomputed directed edges in
        [`faces_to_edges`][triwarp.edges.faces_to_edges] row order. When ``None``, built from
        ``faces``.

    Returns
    -------
    bool
        ``True`` when the mesh is watertight, winding-consistent, and has outward normals (positive
        signed volume). ``False`` for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``edges`` are not all on one device.

    See Also
    --------
    [`make_volume`][triwarp.repair.make_volume]
        Produce one: flip whatever fails this test.
    [`volume`][triwarp.measures.volume]
        Measure one, once this returns ``True``.
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]
    [`is_watertight`][triwarp.validation.is_watertight]
    [`is_orientable`][triwarp.validation.is_orientable]
    [`trimesh.Trimesh.is_volume`][]

    Notes
    -----
    Watertightness here is trimesh's "every undirected edge shared by exactly two faces" (each
    sorted edge forms a group of two), and winding consistency requires the two directed copies of
    every shared edge to be reversed. The signed volume is the sum of per-face signed tetrahedron
    volumes ``dot(v0, cross(v1, v2)) / 6`` measured from the origin; for a closed surface this is
    independent of the reference point and its sign encodes the normal orientation.
    """
    require_same_device(vertices=vertices, faces=faces, edges=edges)
    n_faces = faces.size // 3
    if n_faces == 0:
        return False

    # Watertightness and winding consistency are one pass over the sorted halfedge keys: a run
    # that is not exactly two keys is an edge not shared by exactly two faces, and each pair's two
    # directed copies must be reversed. One flag, one readback, and the volume only if it passes.
    keys, order = _sorted_halfedge_keys(faces, vertices.size)
    violation = _launch.zeros(1, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_validation.sorted_pair_winding_violation,
        dim=keys.size,
        inputs=[faces, edges, keys, order, True, violation],
        device=faces.device,
    )
    if int(read_scalar(violation)) != 0:
        return False

    return tw.measures.volume(vertices, faces) > 0.0


def face_defective_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_quality: float | None = 0.02,
    max_normal_angle: float | None = None,
    max_fold_angle: float | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.bool]:
    """
    Flag faces that are thin, misoriented relative to their neighbourhood, or folded over it.

    The three criteria are independent and a face is flagged if *any* enabled one fires; each is
    disabled by passing ``None``. This is the detector behind
    [`repair.remove_folded_faces`][triwarp.repair.remove_folded_faces] -- unlike the rest of this
    module it needs geometry rather than topology, which is why it takes ``vertices`` and has no
    ``is_*`` counterpart: "defective" is a threshold question, not a property of the mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    min_quality
        Flag a face whose ``radius_ratio``
        ([`face_quality`][triwarp.triangles.face_quality]) is below this. ``0`` is fully degenerate
        and ``1`` is equilateral, so this is a *thinness* gate; MeshLab's ``aratio``, whose default
        of ``0.02`` is this one. ``None`` disables it.
    max_normal_angle
        Flag a face whose normal is more than this many **degrees** from the direction of the sum of
        its edge-neighbours' normals — the local consensus. This catches a single face inserted the
        wrong way round in an otherwise consistent patch. MeshLab's ``nfratio`` (default ``60``,
        off by default). ``None`` disables it.
    max_fold_angle
        Flag a face that meets *some* neighbour at more than this many **degrees** — a fold, where
        the two triangles lie almost on top of each other with opposing normals. Of the two faces at
        such an edge only the one facing *against* its own wider neighbourhood is flagged, since
        only one of them is the mistake; a face whose neighbours give it no consensus (an isolated
        face, or a strip of exactly three) is therefore never flagged on this criterion alone.
        MeshLab's ``folded_faces_angle_threshold`` (default ``160``, off by default). Must be in
        ``(0, 180]``. ``None`` disables it.
    face_normals
        ``(n_faces,)`` unit face normals from
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]; recomputed when
        ``None``. Only the two angle criteria read them, so this is the one geometry pass a caller
        already holding normals can skip.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``; ``True`` marks a defective face. All-``False``
        when every criterion is disabled.

    Raises
    ------
    ValueError
        If ``max_normal_angle`` or ``max_fold_angle`` is outside ``(0, 180]``.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_normals`` are not all on one device.

    Notes
    -----
    Matches MeshLab's ``compute_selection_bad_faces``; the three parameters are its ``aratio``,
    ``nfratio`` and ``folded_faces_angle_threshold``.

    See Also
    --------
    [`repair.remove_folded_faces`][triwarp.repair.remove_folded_faces]
        Deletes the folded ones.
    [`Trimesh.face_normals`][triwarp.mesh.Trimesh.face_normals]
        Caches what ``face_normals`` wants.
    [`repair.remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
        Deletes the fully degenerate ones, on an exact test rather than a threshold.
    [`triwarp.triangles.face_quality`][triwarp.triangles.face_quality]
        The thinness measure ``min_quality`` gates on.
    """
    require_same_device(vertices=vertices, faces=faces, face_normals=face_normals)
    for name, angle in (("max_normal_angle", max_normal_angle), ("max_fold_angle", max_fold_angle)):
        if angle is not None and not 0.0 < angle <= 180.0:
            raise ValueError(f"{name} must be in (0, 180] degrees, got {angle}")

    device = faces.device
    n_faces = faces.size // 3
    # ``wp.empty``: the mask kernel writes every face on every path.
    out_bad = _launch.empty(n_faces, dtype=wp.bool, device=device)
    if n_faces == 0:
        return out_bad

    # A disabled criterion's tables are ``None``: the kernel never reads them, see its comment.
    quality = (
        tw.triangles.face_quality(vertices, faces, metric="radius_ratio")
        if min_quality is not None
        else None
    )
    neighbor_sum = None
    max_angle = None
    if max_normal_angle is not None or max_fold_angle is not None:
        if face_normals is None:
            face_normals, _areas = tw.triangles.face_normals_and_areas(vertices, faces)
        neighbor_sum = _launch.zeros(n_faces, dtype=wp.vec3, device=device)
        max_angle = _launch.zeros(n_faces, dtype=wp.float32, device=device)
        # Over the sorted halfedge keys, each adjacency pair taken at its first member: the pairs
        # ``face_adjacency`` would emit, with no table compacted and no host read of its length.
        keys, order = _sorted_halfedge_keys(faces, vertices.size)
        _launch.launch(
            kernel_validation.accumulate_neighbor_normals,
            dim=keys.size,
            inputs=[face_normals, keys, order, neighbor_sum, max_angle],
            device=device,
        )

    # -2 is unreachable for a cosine and -1 for the normalized quality, so a disabled criterion
    # simply never fires and the kernel needs no per-criterion flag.
    _launch.launch(
        kernel_validation.face_defective_mask,
        dim=n_faces,
        inputs=[
            quality,
            face_normals,
            neighbor_sum,
            max_angle,
            wp.float32(min_quality if min_quality is not None else -1.0),
            wp.float32(
                math.cos(math.radians(max_normal_angle)) if max_normal_angle is not None else -2.0
            ),
            wp.float32(
                math.cos(math.radians(max_fold_angle)) if max_fold_angle is not None else -2.0
            ),
            out_bad,
        ],
        device=device,
    )
    return out_bad


def _corner_parents_from_keys(
    faces: wp.array[wp.int32],
    sorted_keys: tuple[wp.array[wp.uint64], wp.array[wp.int32]],
    require_pairs: bool,
    violation: wp.array[wp.int32] | None,
) -> wp.array[wp.int32]:
    """
    Union-find forest of the corner graph, hooked straight off the sorted halfedge keys.

    The pairs [`face_adjacency`][triwarp.adjacency.face_adjacency] would emit are the runs of
    exactly two equal keys, so no adjacency table, run scan or host read of its length is needed,
    and each corner-graph edge is formed inside the pre-hook and the hook rather than written to a
    table first; every other halfedge links its own corner to itself. With ``require_pairs`` the
    pre-hook raises ``violation[0]`` on any edge not shared by exactly two faces; without it
    ``violation`` is never read and may be ``None``.
    """
    keys, order = sorted_keys
    n = keys.size
    parents = tw.array.arange(n, device=faces.device)
    _launch.launch(
        kernel_validation.sorted_corner_prehook,
        dim=n,
        inputs=[faces, keys, order, require_pairs, parents, violation],
        device=faces.device,
    )
    _launch.launch(
        kernel_validation.sorted_corner_hook,
        dim=n,
        inputs=[faces, keys, order, parents],
        device=faces.device,
    )
    return parents


def _vertex_manifold_check(
    faces: wp.array[wp.int32],
    n_vertices: int,
    parents: wp.array[wp.int32],
    mask: wp.array[wp.bool] | None,
    violation: wp.array[wp.int32] | None,
) -> None:
    """
    Per-vertex manifold test over a hooked corner-graph forest, shared by the predicate and mask.

    Labels the corner graph (node ``3 * f + k`` per face corner) from ``parents`` and flags a
    vertex unless all of its corners land in one component. Exactly one of ``mask`` and
    ``violation`` is given. A ``mask`` arrives zeroed, length ``n_vertices``, and leaves ``True``
    at every referenced vertex whose corners share one component. A ``violation`` flag is raised
    unless every vertex in ``[0, max(faces)]`` is referenced and manifold -- libigl's predicate --
    which holds over any ``n_vertices`` at least ``max(faces) + 1``, so a caller's vertex count
    serves as well as the bound. The flag is the caller's, so it can already carry another test's
    verdict and be read back once for both.

    ``n_vertices`` is a parameter rather than derived because the two public callers size their
    answer differently on purpose:
    [`is_vertex_manifold`][triwarp.validation.is_vertex_manifold] uses ``max(faces) + 1`` (libigl's
    convention, so unreferenced vertices in that range count as non-manifold) while
    [`vertex_manifold_mask`][triwarp.validation.vertex_manifold_mask] uses the caller's vertex
    buffer length.
    """
    device = faces.device
    n_corners = faces.size // 3 * 3
    labels = _launch.empty(n_corners, dtype=wp.int32, device=device)
    min_label = _launch.full(n_vertices, twt.dtype_max(wp.int32), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_validation.corner_labels_and_vertex_min,
        dim=n_corners,
        inputs=[faces, parents, labels, min_label, mask],
        device=device,
    )
    _launch.launch(
        kernel_validation.corner_vertex_check,
        dim=n_corners,
        inputs=[faces, labels, min_label, mask, violation],
        device=device,
    )


def _sorted_halfedge_keys(
    faces: wp.array[wp.int32], n_vertices: int | None
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Sort the ``3 * n_faces`` packed undirected halfedge keys, returning the sorting permutation too.

    Halfedge ``3f + k`` is corner ``k`` of face ``f``, the ``faces_to_edges`` row order, so the
    permutation maps each sorted position back to its face and corner. The keys come straight off
    ``faces`` in one launch into the sort's own buffer
    ([`sorted_face_edge_keys`][triwarp.adjacency.sorted_face_edge_keys]), which packs a caller's
    ``edges_sorted`` rows identically and costs less than hashing them: no row read, no staging
    copy, and only the bits ``n_vertices`` leaves a key are sorted. So no caller here reads a
    precomputed row table for its keys. The radix is ``n_vertices``, or ``INDEX_RADIX_PAIR`` when
    it is ``None`` -- injective on any ``int32`` pair and order-preserving on non-negative indices,
    so the sort, and every run of equal keys, is the same whichever radix packed them and no
    reduction has to find the bound. The radix sort is stable, so equal keys stay in halfedge
    order: the order [`face_adjacency`][triwarp.adjacency.face_adjacency] groups them in.
    """
    return tw.adjacency.sorted_face_edge_keys(faces, n_vertices=n_vertices)


def _halfedge_keys(faces: wp.array[wp.int32]) -> wp.array[wp.uint64]:
    """
    Pack the undirected key of every halfedge, in halfedge order, against ``INDEX_RADIX_PAIR``.

    The unsorted counterpart of ``_sorted_halfedge_keys``, read straight off ``faces`` for the
    same reason: it is the rows ``hash_indices_rows`` of ``edges_sorted`` would give, for less.
    """
    n_faces = faces.size // 3
    keys = _launch.empty(3 * n_faces, dtype=wp.uint64, device=faces.device)
    _launch.launch(
        kernel_adjacency.face_edge_keys,
        dim=n_faces,
        inputs=[faces, wp.uint64(INDEX_RADIX_PAIR)],
        outputs=[keys],
        device=faces.device,
    )
    return keys
