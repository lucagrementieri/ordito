"""Mesh diagnostics: topological/geometric validity predicates and their per-element masks."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import (
    read_scalar,
    read_values,
    require_nonempty_mesh,
    require_same_device,
    require_valid_faces,
)
from ordito.config import is_deterministic
from ordito.constants import INDEX_RADIX_PAIR, TILE_1D
from ordito.kernels import adjacency as kernel_adjacency
from ordito.kernels import intersection as kernel_intersections
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import validation as kernel_validation
from ordito.kernels.algorithms import connected_components as kernel_connected_components


def is_edge_manifold(
    faces: wp.array[wp.int32],
    allow_boundary_edges: bool = True,
    *,
    n_vertices: int | None = None,
    validate: bool = True,
    mates: wp.array[wp.int32] | None = None,
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
    mates
        ``(3 * n_faces,)`` [`halfedge_mates`][ordito.halfedge.halfedge_mates] of ``faces`` when the
        caller holds them ([`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates]). Each
        halfedge's mate code is its edge's share count, so the verdict is then one pass over them
        and no edge is counted. Trusted, not checked; ``faces`` is not read, so neither
        ``n_vertices`` nor ``validate`` has an effect.

    Returns
    -------
    bool
        ``True`` when all edges satisfy the manifold condition. Vacuously ``True`` for an empty
        mesh.

    Raises
    ------
    ValueError
        If ``validate`` is ``True``, ``mates`` is ``None`` and a face index is negative or reaches
        the vertex count.
    RuntimeError
        If ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`is_vertex_manifold`][ordito.validation.is_vertex_manifold]
    [`is_watertight`][ordito.validation.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_edge_manifold``; libigl ``is_edge_manifold``
    corresponds to the ``allow_boundary_edges=True`` case.

    Deliberately **not** ``all(edge_manifold_mask(faces))``, unlike the other ``is_*`` / ``*_mask``
    pairs here: this counts the edges in a hash table and reads it in place, where the mask needs
    each halfedge's own count and so pairs every halfedge. Delegating would add the pairing to the
    cheap path.

    Examples
    --------
    An open grid is edge-manifold only while boundary edges are allowed; a fin makes an edge with
    three faces, which is never manifold:

    ```python
    import warp as wp

    import ordito as od

    vertices, grid_faces = od.creation.grid(count=(3, 3))
    print(od.validation.is_edge_manifold(grid_faces))  # True
    print(od.validation.is_edge_manifold(grid_faces, allow_boundary_edges=False))  # False
    # A strip of three triangles with a fourth "fin" on the edge (1, 2): three faces share it.
    faces = wp.array([0, 1, 2, 2, 1, 3, 2, 3, 4, 1, 2, 5], dtype=wp.int32)
    print(od.validation.is_edge_manifold(faces))  # False
    ```
    """
    require_same_device(faces=faces, mates=mates)
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    device = faces.device
    if mates is not None:
        violation = _launch.zeros(1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_validation.mate_manifold_violation,
            dim=mates.size,
            inputs=[mates, allow_boundary_edges, violation],
            device=device,
        )
        return int(read_scalar(violation)) == 0
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
            odt.dtype_max(wp.int32) if n_vertices is None else n_vertices,
        ],
        outputs=[keys, flags],
        device=device,
    )
    # Counted with ``grouping.hashed_occurrence_counts``, the table ``grouping.unique_1d`` builds,
    # and tested in place: the answer is
    # order-free and needs no per-edge output, so the compaction, the sort and the host read of
    # the unique count that sizes them are skipped.
    slot_counts = od.grouping.hashed_occurrence_counts(keys)
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
    mates: wp.array[wp.int32] | None = None,
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether all three of each face's undirected edges are edge-manifold.

    An edge is manifold when shared by one or two faces (``allow_boundary_edges=True``) or by
    exactly two faces (``allow_boundary_edges=False``); a face is flagged ``True`` only when all
    three of its edges qualify. This is a per-face collapse of libigl's ``BF`` (its per-corner
    ``is_edge_manifold`` matrix), and
    [`is_edge_manifold`][ordito.validation.is_edge_manifold] is ``True`` iff every entry of
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
        ``ordito.reduce.minmax`` whose host readback serialises the device pipeline. Pass
        ``False`` only where both bounds are structurally guaranteed -- a face buffer this package
        produced itself, or one an entry point has already validated.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If ``validate`` is ``True`` and a face index is negative or reaches the vertex count.
    RuntimeError
        If ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`is_edge_manifold`][ordito.validation.is_edge_manifold]
    [`vertex_manifold_mask`][ordito.validation.vertex_manifold_mask]

    Examples
    --------
    The three faces on the over-shared edge are flagged ``False``:

    ```python
    import warp as wp

    import ordito as od

    # A strip of three triangles with a fourth "fin" on the edge (1, 2): three faces share it.
    faces = wp.array([0, 1, 2, 2, 1, 3, 2, 3, 4, 1, 2, 5], dtype=wp.int32)
    print(od.validation.edge_manifold_mask(faces).numpy().tolist())  # [False, False, True, False]
    ```
    """
    require_same_device(faces=faces, mates=mates)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    # The share count of each edge is its halfedges' mate code, read in place: no unique-edge
    # table, inverse or count array, and no host read of the unique count.
    mates = (
        mates
        if mates is not None
        else od.halfedge.halfedge_mates(
            faces, _validated_vertex_bound(faces, n_vertices, validate, "edge_manifold_mask")
        )
    )
    # ``wp.empty``: the pass writes every face's entry.
    out_mask = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_validation.mate_manifold_mask,
        dim=n_faces,
        inputs=[mates, wp.bool(allow_boundary_edges), out_mask],
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
            return od.array.index_bound(faces, require_non_negative=True)
        require_valid_faces(faces, n_vertices, name)
    return n_vertices


def is_vertex_manifold(
    faces: wp.array[wp.int32],
    *,
    face_adjacency: odt.Array2dInt32 | None = None,
    face_adjacency_edges: odt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
    mates: wp.array[wp.int32] | None = None,
) -> bool:
    """
    Whether every vertex up to ``max(faces)`` is used and has a single edge-connected fan of faces.

    A vertex is manifold when the faces incident to it form one connected group under
    "shares an edge through this vertex" adjacency; two fans meeting only at the vertex (a bow-tie)
    are non-manifold. Unreferenced vertices in ``[0, max(faces)]`` are treated as non-manifold,
    matching libigl's ``is_vertex_manifold``; Open3D's ``is_vertex_manifold`` skips them, and so
    does [`is_watertight`][ordito.validation.is_watertight].

    The check builds the corner graph (node ``3 * f + k`` per face corner), links the corresponding
    corners of edge-adjacent faces, and requires all corners of each vertex to fall in one connected
    component.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    face_adjacency
        ``(m, 2)`` precomputed face-adjacency pairs from
        [`face_adjacency`][ordito.adjacency.face_adjacency]. Must be given together with
        ``face_adjacency_edges``.
    face_adjacency_edges
        ``(m, 2)`` shared-edge vertex pairs aligned with ``face_adjacency``
        (``return_edges=True``). Must be given together with ``face_adjacency``.
    n_vertices
        Optional length of the vertex buffer ``faces`` indexes, which must be at least
        ``max(faces) + 1`` and is not checked. It sizes the working tables only and does not change
        the answer: vertices past ``max(faces)`` are still not considered. When ``None``,
        ``max(faces) + 1`` is inferred with a device-host sync.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    bool
        ``True`` when every vertex in ``[0, max(faces)]`` is referenced and manifold. Vacuously
        ``True`` for an empty mesh.

    Raises
    ------
    ValueError
        If exactly one of ``face_adjacency`` / ``face_adjacency_edges`` is provided.
    RuntimeError
        If ``faces``, ``face_adjacency``, ``face_adjacency_edges`` and ``mates`` are not all on
        one device.

    See Also
    --------
    [`vertex_manifold_mask`][ordito.validation.vertex_manifold_mask]
    [`is_edge_manifold`][ordito.validation.is_edge_manifold]
    [`is_watertight`][ordito.validation.is_watertight]

    Notes
    -----
    Corner adjacency is derived from
    [`face_adjacency`][ordito.adjacency.face_adjacency], which pairs faces across edges shared by
    exactly two faces; on edge-non-manifold meshes (an edge shared by three or more faces) read the
    result together with [`is_edge_manifold`][ordito.validation.is_edge_manifold]. Equivalent
    to libigl ``is_vertex_manifold``. The output is sized to ``max(faces) + 1`` so unreferenced
    vertices in that range count as non-manifold; use
    [`vertex_manifold_mask`][ordito.validation.vertex_manifold_mask] for a per-vertex flag
    sized to a caller-provided vertex buffer.

    Examples
    --------
    Two triangles meeting only at vertex 0 form a bow-tie, whose vertex 0 has two separate fans:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    print(od.validation.is_vertex_manifold(faces))  # True
    bow_tie = wp.array([0, 1, 2, 0, 3, 4], dtype=wp.int32)
    print(od.validation.is_vertex_manifold(bow_tie))  # False
    ```
    """
    require_same_device(
        faces=faces,
        face_adjacency=face_adjacency,
        face_adjacency_edges=face_adjacency_edges,
        mates=mates,
    )
    # Checked before the empty-mesh guard so a caller passing only one half of the pair is told
    # about it whatever the mesh is; resolved after it, because on an empty buffer the resolve is
    # two empty tables nothing reads. Every wrapper taking this pair does it in this order.
    od.adjacency.require_paired_adjacency(face_adjacency, face_adjacency_edges)
    if faces.size // 3 == 0:
        return True
    if n_vertices is None:
        # The corner graph derived below trusts this bound, so it carries the range check the
        # edge-key packing would otherwise have made itself.
        n_vertices = od.array.index_bound(faces, require_non_negative=face_adjacency is None)
    violation = _launch.zeros(1, dtype=wp.int32, device=faces.device)
    if face_adjacency is None:
        parents = _corner_parents_from_mates(
            faces,
            (mates if mates is not None else od.halfedge.halfedge_mates(faces, n_vertices)),
            False,
            None,
        )
    else:
        assert face_adjacency_edges is not None
        parents = _corner_parents_from_adjacency(faces, face_adjacency, face_adjacency_edges)
    _vertex_manifold_check(faces, n_vertices, parents, None, violation)
    return int(read_scalar(violation)) == 0


def _corner_parents_from_adjacency(
    faces: wp.array[wp.int32],
    face_adjacency: odt.Array2dInt32,
    face_adjacency_edges: odt.Array2dInt32,
) -> wp.array[wp.int32]:
    """
    Union-find forest of the corner graph over a face-adjacency table.

    Two corner-graph edges per adjacency row, one per shared-edge endpoint, formed in the thread
    and hooked by ``connected_components``' pair pre-hook and hook: the forest
    [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
    flattens, left unflattened for the fused labelling pass that follows.
    """
    device = faces.device
    parents = od.array.arange(faces.size // 3 * 3, device=device)
    m = int(face_adjacency.shape[0])
    if m == 0:
        return parents
    for kernel in (
        kernel_validation.adjacency_corner_prehook,
        kernel_validation.adjacency_corner_hook,
    ):
        if kernel is kernel_validation.adjacency_corner_hook:
            n_corners = parents.size
            if n_corners >= kernel_connected_components.ECL_COMPRESS_FROM:
                _launch.launch(
                    kernel_connected_components.ecl_compress,
                    dim=n_corners,
                    inputs=[parents],
                    device=device,
                )
        _launch.launch(
            kernel,
            dim=m,
            inputs=[faces, face_adjacency, face_adjacency_edges, parents],
            device=device,
        )
    return parents


def vertex_manifold_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    mates: wp.array[wp.int32] | None = None,
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
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_vertices,)`` mask on ``faces.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`is_vertex_manifold`][ordito.validation.is_vertex_manifold]
    [`edge_manifold_mask`][ordito.validation.edge_manifold_mask]

    Examples
    --------
    In a bow-tie only the shared vertex is non-manifold; a vertex no face uses is flagged too:

    ```python
    import warp as wp

    import ordito as od

    vertices = wp.zeros(6, dtype=wp.vec3)
    faces = wp.array([0, 1, 2, 0, 3, 4], dtype=wp.int32)
    mask = od.validation.vertex_manifold_mask(vertices, faces)
    print(mask.numpy().tolist())  # [False, True, True, True, True, False]
    ```
    """
    require_same_device(vertices=vertices, faces=faces, mates=mates)
    n_vertices = vertices.size
    if faces.size // 3 == 0:
        return _launch.zeros(n_vertices, dtype=wp.bool, device=faces.device)
    # The corner graph straight off the halfedge mates, with its pair check off: no thread indexes
    # the violation flag, so none is allocated.
    parents = _corner_parents_from_mates(
        faces,
        (mates if mates is not None else od.halfedge.halfedge_mates(faces, n_vertices)),
        False,
        None,
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
        Broad-phase slots reserved per triangle. Not a cap: a triangle with more candidates walks
        the rest again on the device, so the answer does not depend on it.

    Returns
    -------
    bool
        ``True`` when at least one intersecting non-adjacent triangle pair exists. ``False`` for
        meshes with fewer than two faces.

    See Also
    --------
    [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]
    [`mesh_with_mesh`][ordito.intersection.mesh_with_mesh]
    [`is_watertight`][ordito.validation.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_self_intersecting``.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    # Two triangles piercing each other, which share no vertex.
    vertices = wp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
         [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 1.0]],
        dtype=wp.vec3,
    )
    faces = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32)
    print(od.validation.is_self_intersecting(wp.Mesh(points=vertices, indices=faces)))  # True
    vertices, faces = od.creation.box()
    print(od.validation.is_self_intersecting(wp.Mesh(points=vertices, indices=faces)))  # False
    ```
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
    [`is_self_intersecting`][ordito.validation.is_self_intersecting] is ``True`` iff any entry of
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
        Broad-phase slots reserved per triangle. Not a cap: a triangle with more candidates walks
        the rest again on the device, so the answer does not depend on it.
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
    [`is_self_intersecting`][ordito.validation.is_self_intersecting]

    Examples
    --------
    Both faces of an intersecting pair are flagged:

    ```python
    import warp as wp

    import ordito as od

    # Two triangles piercing each other, which share no vertex.
    vertices = wp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
         [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 1.0]],
        dtype=wp.vec3,
    )
    faces = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32)
    mask = od.validation.face_self_intersecting_mask(vertices, faces)
    print(mask.numpy().tolist())  # [True, True]
    ```
    """
    require_same_device(vertices=vertices, faces=faces, mesh=mesh)
    return _face_self_intersecting_mask(vertices, faces, max_triangle_collisions, mesh)


def _face_self_intersecting_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_triangle_collisions: int = 32,
    mesh: wp.Mesh | None = None,
    any_marked: tuple[int, wp.array[wp.int32]] | None = None,
) -> wp.array[wp.bool]:
    """
    [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask], trusting.

    With ``any_marked`` as ``(stamp, state)``, ``stamp`` is also stored into ``state[0]`` when the
    mask marks any face, by the mask's own last launch
    (``kernels/intersection.mark_overflow_crossings_flag_any``), for a caller deciding on that
    alone with a four-byte read.
    """
    device = vertices.device
    n_faces = faces.size // 3
    mask = _launch.zeros(n_faces, dtype=wp.bool, device=device)
    if n_faces < 2:
        return mask

    if mesh is None:
        require_nonempty_mesh(faces, "face_self_intersecting_mask")
        mesh = wp.Mesh(points=vertices, indices=faces)

    _mark_self_intersections(
        mesh, vertices, faces, max_triangle_collisions, True, mask, any_marked=any_marked
    )
    return mask


def _mark_self_intersections(
    mesh: wp.Mesh,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_triangle_collisions: int,
    mark_faces: bool,
    out_marks: wp.array[wp.bool],
    *,
    any_marked: tuple[int, wp.array[wp.int32]] | None = None,
) -> None:
    """
    Broad and narrow phase of the self-intersection test, shared by the predicate and the mask.

    The two differ only in ``mark_faces``: the mask marks both faces of every intersecting
    candidate in ``out_marks``, the predicate raises ``out_marks[0]``. The broad phase queries each
    triangle's AABB against ``mesh``'s BVH
    ([`intersection.face_box_candidates`][ordito.intersection.face_box_candidates], complete
    whatever ``max_triangle_collisions`` is); the narrow phase is Moller's interval test on every
    live slot, skipping
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
    # No host read: the faces whose candidates overflow the stride are re-walked on the device
    # by ``mark_overflow_crossings``, so the table's stride stays ``max_triangle_collisions``.
    targets, counts, stride = od.intersection.face_box_candidates(
        vertices, faces, mesh, capacity=max_triangle_collisions, grow=False
    )
    flag_only = not mark_faces
    inputs = [vertices, faces, vertices, faces]
    _launch.launch(
        kernel_intersections.mark_crossing_slot_masks,
        dim=n_faces * stride,
        inputs=[*inputs, targets, counts, stride, flag_only, out_marks, out_marks],
        device=vertices.device,
    )
    if any_marked is not None:
        stamp, state = any_marked
        _launch.launch(
            kernel_intersections.mark_overflow_crossings_flag_any,
            dim=n_faces,
            inputs=[mesh.id, *inputs, counts, stride, stamp],
            outputs=[out_marks, state],
            device=vertices.device,
        )
        return
    _launch.launch(
        kernel_intersections.mark_overflow_crossings,
        dim=n_faces,
        inputs=[mesh.id, *inputs, counts, stride, flag_only, out_marks, out_marks],
        device=vertices.device,
    )


def is_winding_consistent(
    faces: wp.array[wp.int32],
    *,
    n_vertices: int | None = None,
    mates: wp.array[wp.int32] | None = None,
) -> bool:
    """
    Whether every shared edge is traversed in opposite directions by its two faces.

    A mesh has consistent winding when, for each edge shared by two faces, the two faces list the
    edge's endpoints in opposite order (so their normals agree locally). This is a property of the
    current winding, unlike [`is_orientable`][ordito.validation.is_orientable], which allows
    faces to be flipped. Matches [`trimesh.Trimesh.is_winding_consistent`][].

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Optional exclusive bound on the vertex indices, forwarded to
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], which pairs faster with it. It does
        not change the answer, and it is trusted, not checked: it must exceed every index.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    bool
        ``True`` when winding is consistent across all shared edges. Vacuously ``True`` for an empty
        mesh or a mesh with no shared edges.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`edge_winding_consistent_mask`][ordito.validation.edge_winding_consistent_mask]
    [`is_orientable`][ordito.validation.is_orientable]
    [`is_volume`][ordito.validation.is_volume]
    [`trimesh.Trimesh.is_winding_consistent`][]

    Examples
    --------
    The cube is consistently wound; two triangles running their shared edge the same way are not,
    although flipping one would fix them (see [`is_orientable`][ordito.validation.is_orientable]):

    ```python
    import warp as wp

    import ordito as od

    vertices, box_faces = od.creation.box()
    print(od.validation.is_winding_consistent(box_faces))  # True
    # Two triangles sharing the edge (1, 2), the second wound against the first.
    faces = wp.array([0, 1, 2, 1, 2, 3], dtype=wp.int32)
    print(od.validation.is_winding_consistent(faces))  # False
    print(od.validation.is_orientable(faces))  # True
    ```
    """
    require_same_device(faces=faces, mates=mates)
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    # ``edge_winding_consistent_mask``'s per-pair test read straight off the halfedge mates, so
    # the verdict needs no group table, no host read of its length and no reduction over a mask.
    mates = mates if mates is not None else od.halfedge.halfedge_mates(faces, n_vertices)
    violation = _launch.zeros(1, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_validation.mate_pair_winding_violation,
        dim=mates.size,
        inputs=[faces, None, mates, False, violation],
        device=faces.device,
    )
    return int(read_scalar(violation)) == 0


def edge_winding_consistent_mask(
    faces: wp.array[wp.int32], edges: odt.Array2dInt32 | None = None
) -> wp.array[wp.bool]:
    """
    Per shared-edge flag: whether an undirected edge's two faces traverse it in opposite directions.

    Built over the undirected edges shared by exactly two faces (edge groups of length two). An
    entry is ``True`` when the two directed half-edges are reversed (locally consistent normals);
    boundary and non-manifold edges have no entry.
    [`is_winding_consistent`][ordito.validation.is_winding_consistent] is ``True`` iff every
    entry is ``True``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    edges
        ``(3 * n_faces, 2)`` precomputed directed edges in
        [`faces_to_edges`][ordito.edges.faces_to_edges] row order. Built from ``faces`` when
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
    [`is_winding_consistent`][ordito.validation.is_winding_consistent]
    [`face_flip_mask`][ordito.validation.face_flip_mask]

    Examples
    --------
    One entry per edge shared by two faces; the cube's 18 are all consistent:

    ```python
    import warp as wp

    import ordito as od

    vertices, box_faces = od.creation.box()
    print(od.validation.edge_winding_consistent_mask(box_faces).shape)  # (18,)
    # Two triangles sharing the edge (1, 2), the second wound against the first.
    faces = wp.array([0, 1, 2, 1, 2, 3], dtype=wp.int32)
    print(od.validation.edge_winding_consistent_mask(faces).numpy().tolist())  # [False]
    ```
    """
    require_same_device(faces=faces, edges=edges)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    # Neither edge table is built when not given: the keys come straight off ``faces`` and the
    # kernel reads each directed edge from its face corner, the row ``faces_to_edges`` would hold.
    edge_groups = od.grouping.group(_halfedge_keys(faces), 2)
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
    *,
    n_vertices: int | None = None,
    face_adjacency: odt.Array2dInt32 | None = None,
    face_adjacency_edges: odt.Array2dInt32 | None = None,
) -> tuple[wp.array[wp.int32], odt.Array2dInt32, wp.array[wp.int32], int]:
    """
    Per-face Z2 orientation bits (relative to each component seed) plus signed face-adjacency edges.

    ``orient[f]`` is ``0`` for a face that agrees with its connected component's seed and ``1`` for
    a face that must be flipped to agree — the flip mask consumed by
    [`face_flip_mask`][ordito.validation.face_flip_mask] and
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]. ``m`` is the number of
    face-adjacency rows; when ``m == 0`` the returned ``signed_edges`` / ``signs`` are empty.
    Assumes ``n_faces > 0`` (callers guard).

    This is a lower-level primitive (the underlying orientation engine behind
    [`is_orientable`][ordito.validation.is_orientable] and
    [`face_flip_mask`][ordito.validation.face_flip_mask]) exposed for callers that
    need the raw flip bits together with the propagation edges, such as
    [`ordito.repair.make_winding_consistent`][ordito.repair.make_winding_consistent].

    The bits are solved, not propagated: ``orient`` is the Z2 potential of the signed
    face-adjacency graph, computed by a parity-carrying union-find
    ([`connected_component_parity_from_edges`]
    [ordito.graph.connected_component_parity_from_edges]) in a fixed three launches. Cost
    therefore depends on the mesh's *size*, not on the diameter of its face-adjacency graph.

    On a non-orientable component no potential exists, and some adjacency edges are left
    unsatisfied. Which ones depends on the setting of
    [`is_deterministic`][ordito.config.is_deterministic]: on, the bits are the parity over the
    minimum spanning forest of the adjacency rows keyed by row index, each tree's lowest-indexed
    face at ``0``, the same on every run and on either device, for one more pass and a host read
    on every input, and a few passes more on a non-orientable one; off, they follow whichever
    spanning forest the parallel union-find builds, which on CUDA can differ between runs. On an
    orientable mesh the bits are the same either way.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer. ``n_faces`` must be positive.
    n_vertices
        Optional vertex count, forwarded to
        [`face_adjacency`][ordito.adjacency.face_adjacency]: it narrows the sort that pairs the
        faces. Trusted, not checked: it must exceed every index in ``faces``.
    face_adjacency
        ``(m, 2)`` precomputed face-adjacency pairs from
        [`face_adjacency`][ordito.adjacency.face_adjacency]
        ([`Trimesh.face_adjacency`][ordito.mesh.Trimesh.face_adjacency]), which spares the sort
        that pairs the faces; ``signed_edges`` is then this array itself. Must be given together
        with ``face_adjacency_edges``.
    face_adjacency_edges
        ``(m, 2)`` shared-edge vertex pairs aligned with ``face_adjacency``
        (``return_edges=True``). Must be given together with ``face_adjacency``.

    Returns
    -------
    orient : wp.array[wp.int32]
        ``(n_faces,)`` flip bits (``0`` or ``1``) on ``faces.device``.
    signed_edges : odt.Array2dInt32
        ``(m, 2)`` face-adjacency pairs, one row per face-adjacency edge.
    signs : wp.array[wp.int32]
        ``(m,)`` Z2 sign per adjacency edge.
    m : int
        Number of face-adjacency rows.

    Raises
    ------
    ValueError
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is given.
    RuntimeError
        If ``faces``, ``face_adjacency`` and ``face_adjacency_edges`` are not all on one device.

    See Also
    --------
    [`is_orientable`][ordito.validation.is_orientable]
    [`face_flip_mask`][ordito.validation.face_flip_mask]

    Examples
    --------
    The second face disagrees with the first, its component's seed:

    ```python
    import warp as wp

    import ordito as od

    # Two triangles sharing the edge (1, 2), the second wound against the first.
    faces = wp.array([0, 1, 2, 1, 2, 3], dtype=wp.int32)
    orient, signed_edges, signs, m = od.validation.face_orientation_bits(faces)
    print(orient.numpy().tolist())  # [0, 1]
    print(signed_edges.numpy().tolist(), m)  # [[0, 1]] 1
    ```
    """
    require_same_device(
        faces=faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
    )
    od.adjacency.require_paired_adjacency(face_adjacency, face_adjacency_edges)
    device = faces.device
    n_faces = faces.size // 3
    if face_adjacency is None or face_adjacency_edges is None:
        adjacency, adjacency_edges = od.adjacency.face_adjacency(
            faces, return_edges=True, n_vertices=n_vertices
        )
    else:
        adjacency, adjacency_edges = face_adjacency, face_adjacency_edges
    m = int(adjacency.shape[0])

    if m == 0:
        signed_edges = odt.empty_2d((0, 2), wp.int32, device=device)
        signs = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.zeros(n_faces, dtype=wp.int32, device=device), signed_edges, signs, m

    # The signed edges are the adjacency rows themselves, so only the signs are written, by the
    # launch that also lays down the parity forest's identity words.
    signed_edges = adjacency
    signs = _launch.empty(m, dtype=wp.int32, device=device)
    words = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_validation.face_pair_signs_and_roots,
        dim=max(m, n_faces),
        inputs=[faces, adjacency, adjacency_edges, n_faces],
        outputs=[signs, words],
        device=device,
    )

    # The flip bits *are* a Z2 potential over the signed face-adjacency graph, so a parity-carrying
    # union-find produces them directly, in a hook and a flatten after the signs, with no host sync
    # ([`connected_component_parity_from_edges`]
    # [ordito.graph.connected_component_parity_from_edges]). The bits are unchanged from the
    # edge-by-edge flood fill this replaces: that seeded ``0`` at each component's smallest face id
    # and XOR-ed signs outward, and the union-find's root is that same smallest id at that same
    # parity 0. What changes is the cost — the fill ran one launch per graph level, so a mesh whose
    # face adjacency is a long path (a ribbon) paid tens of thousands of launches and thousands of
    # readbacks for work bounded by a few milliseconds of bandwidth.
    # ``signed_edges`` holds face ids this function just built from ``adjacency`` (itself bounded
    # by ``n_faces`` by construction) and every sign is ``0`` or ``1``, so nothing is range-checked.
    _launch.launch(
        kernel_connected_components.ecl_hook_parity,
        dim=m,
        inputs=[signed_edges, signs, words],
        device=device,
    )
    if is_deterministic():
        words = _reproducible_orientation_words(faces, None, signed_edges, signs, words)
    return _flatten_orientation(words), signed_edges, signs, m


def is_orientable(
    faces: wp.array[wp.int32],
    *,
    n_vertices: int | None = None,
    mates: wp.array[wp.int32] | None = None,
) -> bool:
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
    n_vertices
        Optional exclusive bound on the vertex indices, forwarded to
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], which pairs faster with it. It does
        not change the answer, and it is trusted, not checked: it must exceed every index.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    bool
        ``True`` when a consistent orientation exists. Vacuously ``True`` for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`face_flip_mask`][ordito.validation.face_flip_mask]
    [`is_watertight`][ordito.validation.is_watertight]
    [`is_winding_consistent`][ordito.validation.is_winding_consistent]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_orientable``. Unlike
    [`is_winding_consistent`][ordito.validation.is_winding_consistent], orientability allows
    individual faces to be flipped, so a consistently-orientable mesh with inconsistent winding
    still returns ``True``.

    Examples
    --------
    A Mobius strip, three quads joined with a half twist, cannot be wound consistently whichever
    faces are flipped:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    print(od.validation.is_orientable(faces))  # True
    mobius = wp.array([0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 0, 2, 0, 3], dtype=wp.int32)
    print(od.validation.is_orientable(mobius))  # False
    ```
    """
    require_same_device(faces=faces, mates=mates)
    n_faces = faces.size // 3
    if n_faces == 0:
        return True

    orient, mates = _orientation_bits_from_mates(faces, n_vertices, mates)
    device = faces.device
    conflict = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_validation.mate_orientation_conflict,
        dim=mates.size,
        inputs=[faces, mates, orient, conflict],
        device=device,
    )
    return int(read_scalar(conflict, 0)) == 0


def face_flip_mask(
    faces: wp.array[wp.int32],
    *,
    n_vertices: int | None = None,
    mates: wp.array[wp.int32] | None = None,
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a face must be flipped to make winding consistent within its patch.

    Runs the Z2 orientation propagation over the face-adjacency graph and returns ``True`` for
    every face whose winding disagrees with its connected component's seed, the component's
    lowest-indexed face (whose own entry is therefore always ``False``). Applying these flips
    yields a consistently wound mesh, so this is the per-face flip mask consumed by
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]. A mesh that is already
    consistently wound yields an all-``False`` mask.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Optional exclusive bound on the vertex indices, forwarded to
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], which pairs faster with it. It does
        not change the answer, and it is trusted, not checked: it must exceed every index.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``. Empty for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`is_orientable`][ordito.validation.is_orientable]
    [`is_winding_consistent`][ordito.validation.is_winding_consistent]
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]
    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask]

    Notes
    -----
    On an orientable mesh the mask is a deterministic function of ``faces``: each component keeps
    its lowest-indexed face's winding, on either device. On a non-orientable patch no consistent
    winding exists and the mask is a best-effort flood-fill (matching
    ``trimesh.repair.fix_winding``) that leaves the contradiction on a seam. Which seam depends
    on the setting of [`is_deterministic`][ordito.config.is_deterministic]:

    - on, the mask is the parity over the minimum spanning forest of the edge-adjacent face pairs,
      each pair keyed by the lower of its two halfedge indices (``3 * face + corner``): each tree's
      lowest-indexed face keeps its winding, every other face agrees with its tree path, and the
      seam is the pairs the forest leaves out. The same mask on every run and on either device,
      for one more pass and a host read on every input, and a few passes more on a non-orientable
      one;
    - off, it is whichever spanning forest the parallel union-find builds, which on CUDA can differ
      between runs.

    This is a different flip from
    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask], which is about the *UV
    domain*: this mask flags a triangle whose index order must be reversed to agree with its patch,
    a property of the 3D connectivity that ignores geometry entirely, while that one flags a
    triangle whose UV image has negative signed area. Neither implies the other.

    Examples
    --------
    Reversing one cube face makes it disagree with all the others. The mask keeps the winding of
    the lowest-indexed face, so here it flags the eleven faces that were *not* changed; flipping
    them gives a consistent, inward-facing cube:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    rows = faces.numpy().reshape(-1, 3)
    rows[0] = rows[0][::-1]
    faces = wp.array(rows.ravel(), dtype=wp.int32)
    flips = od.validation.face_flip_mask(faces).numpy()
    print(int(flips.sum()), bool(flips[0]))  # 11 False
    ```
    """
    require_same_device(faces=faces, mates=mates)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    # The flip mask is the parity itself, so the flatten writes it as a ``bool`` directly.
    words = _face_flip_words(faces, n_vertices, mates)
    mask = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_validation.flatten_parity_flip_mask, dim=n_faces, inputs=[words, mask], device=device
    )
    return mask


def _face_flip_words(
    faces: wp.array[wp.int32], n_vertices: int | None, mates: wp.array[wp.int32] | None
) -> wp.array[wp.int32]:
    """
    ``face_flip_mask``'s parity words, before the flatten that reads each face's flip off them.

    For a caller whose own per-face kernel reads the flip in place
    (``kernels/validation.face_parity_flip``), so the mask is never materialized. Nonempty
    ``faces``; reproducible under the deterministic setting exactly as the mask is.
    """
    words, mates = _orientation_words_from_mates(faces, n_vertices, mates)
    if is_deterministic():
        words = _reproducible_orientation_words(faces, mates, None, None, words)
    return words


def _reproducible_orientation_words(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32] | None,
    edges: odt.Array2dInt32 | None,
    signs: wp.array[wp.int32] | None,
    words: wp.array[wp.int32],
) -> wp.array[wp.int32]:
    """
    Parity words that are a function of ``faces`` alone, for the deterministic setting.

    The signed face graph is the halfedge ``mates``' pairs (``face_flip_mask``) or, given
    ``edges`` and ``signs``, that edge list (``face_orientation_bits``); the other pair is
    ``None``. ``words`` from the racing parity union-find over it already are on a consistent
    mesh, which one conflict test and a readback confirm. Otherwise the parity is re-solved over
    the graph's unique minimum spanning forest, keyed by edge index (``kernels/validation``'s
    Boruvka rounds, one readback each to stop), whose parity is the same on every run and, on each
    consistent component, the racing answer.
    """
    device = faces.device
    n_faces = words.size
    conflict = _launch.zeros(1, dtype=wp.int32, device=device)
    orient = _flatten_orientation(words)
    if edges is None:
        assert mates is not None
        n_edges = mates.size
        _launch.launch(
            kernel_validation.mate_orientation_conflict,
            dim=n_edges,
            inputs=[faces, mates, orient, conflict],
            device=device,
        )
    else:
        n_edges = int(edges.shape[0])
        _launch.launch(
            kernel_validation.signed_edge_orientation_conflict,
            dim=n_edges,
            inputs=[edges, signs, orient, conflict],
            device=device,
        )
    # Unavoidable: whether any pair contradicts the parity decides whether the forest is needed.
    if int(read_scalar(conflict, 0)) == 0:
        return words

    graph = [faces, mates, edges, signs]
    parents = od.array.arange(n_faces, device=device)
    labels = _launch.empty(n_faces, dtype=wp.int32, device=device)
    best = _launch.empty(n_faces, dtype=wp.int32, device=device)
    in_forest = _launch.zeros(n_edges, dtype=wp.int32, device=device)
    hooked = conflict
    # Each round hooks every component with an outgoing edge, so at least halves their number:
    # ``log2(n_faces) + 1`` rounds bound it, and a round that hooks nothing ends it sooner.
    for _ in range(n_faces.bit_length() + 1):
        _launch.launch(
            kernel_validation.forest_flatten,
            dim=n_faces,
            inputs=[parents, labels, best],
            device=device,
        )
        _launch.launch(
            kernel_validation.signed_forest_candidates,
            dim=n_edges,
            inputs=[*graph, labels, best],
            device=device,
        )
        _launch.zero_(hooked)
        _launch.launch(
            kernel_validation.signed_forest_hook,
            dim=n_faces,
            inputs=[*graph, labels, best, parents, in_forest, hooked],
            device=device,
        )
        # Unavoidable: the round count is the components' merge depth, known only on the device.
        if int(read_scalar(hooked, 0)) == 0:
            break
    forest_words = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_connected_components.ecl_init_parent_parity,
        dim=n_faces,
        inputs=[forest_words],
        device=device,
    )
    _launch.launch(
        kernel_validation.signed_forest_hook_parity,
        dim=n_edges,
        inputs=[*graph, in_forest, forest_words],
        device=device,
    )
    return forest_words


def _orientation_bits_from_mates(
    faces: wp.array[wp.int32], n_vertices: int | None, mates: wp.array[wp.int32] | None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Flip bits, with the signed edges formed straight off the halfedge mates.

    The bits [`face_orientation_bits`][ordito.validation.face_orientation_bits] returns, for
    [`is_orientable`][ordito.validation.is_orientable], which consumes the bits and not the table.
    ``faces`` must be non-empty; ``n_vertices`` is ``halfedge_mates``' optional bound. The mates
    are returned too, so a caller can re-form the same edges; given ``mates`` are used as they are.
    """
    words, mates = _orientation_words_from_mates(faces, n_vertices, mates)
    return _flatten_orientation(words), mates


def _orientation_words_from_mates(
    faces: wp.array[wp.int32], n_vertices: int | None, mates: wp.array[wp.int32] | None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Build the parity union-find's hooked words over the signed edges of the halfedge mates.

    One signed edge per halfedge: each adjacency pair's row at its lower halfedge, a sign-``0``
    self-loop everywhere else, which the parity union-find skips and every orientation satisfies.
    So neither the adjacency table nor the host read of its length is built. Flattened, the words
    give [`face_orientation_bits`][ordito.validation.face_orientation_bits]' bits
    (``_flatten_orientation``) or [`face_flip_mask`][ordito.validation.face_flip_mask]'s mask.
    Returned with the mates.
    """
    if mates is None:
        mates = od.halfedge.halfedge_mates(faces, n_vertices)
    # Every endpoint is a face id derived in the thread from a halfedge index, bounded by
    # ``n_faces`` by construction, and every sign is ``0`` or ``1``. The pre-hook lays the forest
    # down already hooked to each face's smallest neighbour, so no identity forest is written first.
    n_faces = faces.size // 3
    words = _launch.empty(n_faces, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_validation.mate_prehook_parity,
        dim=n_faces,
        inputs=[faces, mates],
        outputs=[words],
        device=faces.device,
    )
    _launch.launch(
        kernel_validation.mate_hook_parity,
        dim=mates.size,
        inputs=[faces, mates, words],
        device=faces.device,
    )
    return words, mates


def _flatten_orientation(words: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """Each face's parity against its component root, from hooked parity words."""
    orient = _launch.empty(words.size, dtype=wp.int32, device=words.device)
    _launch.launch(
        kernel_connected_components.ecl_flatten_parity,
        dim=words.size,
        inputs=[words, None, orient],
        device=words.device,
    )
    return orient


def is_watertight(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    mesh: wp.Mesh | Callable[[], wp.Mesh] | None = None,
    mates: wp.array[wp.int32] | None = None,
) -> bool:
    """
    Whether the mesh bounds a closed volume with no self-intersections.

    Follows Open3D's ``IsWatertight``: the mesh must be edge-manifold with no boundary edges,
    vertex-manifold, and free of self-intersections. Vertices no face uses are ignored, as in
    Open3D: they are not part of the surface. (The stricter
    [`is_vertex_manifold`][ordito.validation.is_vertex_manifold] counts one below ``max(faces)``
    as non-manifold, as libigl does.)

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
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    bool
        ``True`` when the mesh is edge-manifold (no boundary), vertex-manifold, and not
        self-intersecting.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, a ``wp.Mesh`` ``mesh`` and ``mates`` are not all on one device.

    See Also
    --------
    [`face_watertight_mask`][ordito.validation.face_watertight_mask]
    [`is_edge_manifold`][ordito.validation.is_edge_manifold]
    [`is_vertex_manifold`][ordito.validation.is_vertex_manifold]
    [`is_self_intersecting`][ordito.validation.is_self_intersecting]
    [`is_volume`][ordito.validation.is_volume]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_watertight``. For the cheaper "every edge shared
    by exactly two faces" test (trimesh semantics) use
    [`is_edge_manifold`][ordito.validation.is_edge_manifold] with
    ``allow_boundary_edges=False``.

    **This is not the predicate form of
    [`face_watertight_mask`][ordito.validation.face_watertight_mask]**, despite the names: that
    mask is trimesh's per-face edge test, while this is Open3D's three-part composition, so
    ``all(face_watertight_mask(faces))`` is the *first* of the three conditions and not this
    function. Every other ``is_*`` / ``*_mask`` pair in this module does relate that way.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    print(od.validation.is_watertight(vertices, faces))  # True
    open_box = wp.array(faces.numpy()[3:], dtype=wp.int32)  # face 0 removed
    print(od.validation.is_watertight(vertices, open_box))  # False
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        mesh=mesh if isinstance(mesh, wp.Mesh) else None,
        mates=mates,
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
    parents = _corner_parents_from_mates(
        faces,
        (mates if mates is not None else od.halfedge.halfedge_mates(faces, n_vertices)),
        True,
        violation,
    )
    _vertex_manifold_check(faces, n_vertices, parents, None, violation, require_referenced=False)
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

    This is [`edge_manifold_mask`][ordito.validation.edge_manifold_mask] with
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
    [`is_watertight`][ordito.validation.is_watertight]
    [`edge_manifold_mask`][ordito.validation.edge_manifold_mask]

    Examples
    --------
    Removing one face of a cube opens a hole whose rim runs through the faces next to it:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    open_box = wp.array(faces.numpy()[3:], dtype=wp.int32)  # face 0 removed
    mask = od.validation.face_watertight_mask(open_box)
    print(mask.shape)  # (11,)
    print(np.flatnonzero(~mask.numpy()).tolist())  # [0, 1, 3]
    ```
    """
    return edge_manifold_mask(faces, allow_boundary_edges=False, n_vertices=n_vertices)


def is_volume(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    edges: odt.Array2dInt32 | None = None,
    mates: wp.array[wp.int32] | None = None,
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
        [`faces_to_edges`][ordito.edges.faces_to_edges] row order. When ``None``, built from
        ``faces``.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.

    Returns
    -------
    bool
        ``True`` when the mesh is watertight, winding-consistent, and has outward normals (positive
        signed volume). ``False`` for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``edges`` and ``mates`` are not all on one device.

    See Also
    --------
    [`make_volume`][ordito.repair.make_volume]
        Produce one: flip whatever fails this test.
    [`volume`][ordito.measures.volume]
        Measure one, once this returns ``True``.
    [`is_winding_consistent`][ordito.validation.is_winding_consistent]
    [`is_watertight`][ordito.validation.is_watertight]
    [`is_orientable`][ordito.validation.is_orientable]
    [`trimesh.Trimesh.is_volume`][]

    Notes
    -----
    Watertightness here is trimesh's "every undirected edge shared by exactly two faces" (each
    sorted edge forms a group of two), and winding consistency requires the two directed copies of
    every shared edge to be reversed. The signed volume is the sum of per-face signed tetrahedron
    volumes ``dot(v0, cross(v1, v2)) / 6`` measured from the origin; for a closed surface this is
    independent of the reference point and its sign encodes the normal orientation.

    Examples
    --------
    Reversing every face keeps the cube watertight and consistent but turns its normals inward:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    print(od.validation.is_volume(vertices, faces))  # True
    inverted = wp.array(faces.numpy().reshape(-1, 3)[:, ::-1].ravel(), dtype=wp.int32)
    print(od.validation.is_volume(vertices, inverted))  # False
    ```
    """
    require_same_device(vertices=vertices, faces=faces, edges=edges, mates=mates)
    n_faces = faces.size // 3
    if n_faces == 0:
        return False

    # Watertightness and winding consistency are a count over the halfedge mates -- a halfedge
    # without a mate is on an edge not shared by exactly two faces, and each pair's two directed
    # copies must be reversed -- taken in the launch that folds the volume, read back with it.
    mates = mates if mates is not None else od.halfedge.halfedge_mates(faces, vertices.size)
    volume, closed = _closed_signed_volume(vertices, faces, mates, edges, True)
    return closed and volume > 0.0


def _closed_signed_volume(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    edges: odt.Array2dInt32 | None,
    require_opposite: bool,
) -> tuple[float, bool]:
    """
    [`measures.volume`][ordito.measures.volume], and whether every edge has two faces.

    With ``require_opposite`` the two faces must also traverse their edge both ways. ``mates`` are
    ``faces``' [`halfedge_mates`][ordito.halfedge.halfedge_mates]; ``edges`` as
    [`is_volume`][ordito.validation.is_volume] takes it. Nonempty ``faces``; one launch and one
    readback (``kernels/validation.closed_signed_volume``).
    """
    device = faces.device
    blocks = kernel_reduce.blocks_1d(faces.size // 3)
    total, stride = _launch.block_partials(blocks, 1, wp.vec2, device)
    _launch.launch_tiled(
        kernel_validation.closed_signed_volume,
        dim=[blocks],
        inputs=[vertices, faces, edges, mates, require_opposite, stride, total],
        block_dim=TILE_1D,
        device=device,
    )
    volume, broken = read_scalar(total, 0)
    return float(volume), float(broken) == 0.0


def face_defective_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_quality: float | None = 0.02,
    max_normal_angle: float | None = None,
    max_fold_angle: float | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
    mates: wp.array[wp.int32] | None = None,
    normal_criterion: Literal["consensus", "meshlab"] = "consensus",
) -> wp.array[wp.bool]:
    """
    Flag faces that are thin, misoriented relative to their neighbourhood, or folded over it.

    The three criteria are independent and a face is flagged if *any* enabled one fires; each is
    disabled by passing ``None``. This is the detector behind
    [`repair.remove_folded_faces`][ordito.repair.remove_folded_faces] -- unlike the rest of this
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
        ([`face_quality`][ordito.triangles.face_quality]) is below this. ``0`` is fully degenerate
        and ``1`` is equilateral, so this is a *thinness* gate; MeshLab's ``aratio``, whose default
        of ``0.02`` is this one. ``None`` disables it.
    max_normal_angle
        Flag a face whose normal is more than this many **degrees** from the direction of the sum of
        its edge-neighbours' normals — the local consensus. This catches a single face inserted the
        wrong way round in an otherwise consistent patch. Same default (``60``) as MeshLab's
        ``nfratio`` but not the same test unless ``normal_criterion="meshlab"`` (see Notes).
        ``None`` disables it.
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
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]; formed from the
        corners where they are read when ``None``. Only the two angle criteria read them.
    mates
        ``(3 * n_faces,)`` halfedge mate codes of ``faces``, from
        [`halfedge_mates`][ordito.halfedge.halfedge_mates], to spare the sort that otherwise pairs
        the halfedges. Not checked against ``faces``.
        [`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates] is the cached one to pass.
    normal_criterion
        What ``max_normal_angle`` compares a face's normal with.

        - ``"consensus"`` (default) — the direction of the sum of its edge-neighbours' normals.
        - ``"meshlab"`` — each edge-neighbour's normal in turn, flagging the face when any one
          differs by more than the angle: MeshLab's ``nfratio``.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``faces.device``; ``True`` marks a defective face. All-``False``
        when every criterion is disabled.

    Raises
    ------
    ValueError
        If ``max_normal_angle`` or ``max_fold_angle`` is outside ``(0, 180]``, or
        ``normal_criterion`` is not ``"consensus"`` or ``"meshlab"``.
    RuntimeError
        If ``vertices``, ``faces``, ``face_normals`` and ``mates`` are not all on one device.

    Notes
    -----
    The thinness and fold criteria match MeshLab's ``compute_selection_bad_faces`` (its ``aratio``
    and ``folded_faces_angle_threshold``). The default normal criterion does not match its
    ``nfratio``: MeshLab selects *both* faces of every edge whose normals differ by more than the
    angle (``normal_criterion="meshlab"``), so it flags a reversed face together with all its
    neighbours, and every face of a cube at ``60``. Comparing each face with its neighbours'
    consensus instead flags the reversed face alone on a smooth surface. The consensus needs
    neighbours that mostly agree: on a coarse mesh, where a face has three neighbours and they meet
    at sharp angles, a correct face next to a reversed one can fail it too (a cube with one
    triangle reversed flags that triangle and one neighbour).

    See Also
    --------
    [`repair.remove_folded_faces`][ordito.repair.remove_folded_faces]
        Deletes the folded ones.
    [`Trimesh.face_normals`][ordito.mesh.Trimesh.face_normals]
        Caches what ``face_normals`` wants.
    [`repair.remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
        Deletes the fully degenerate ones, on an exact test rather than a threshold.
    [`ordito.triangles.face_quality`][ordito.triangles.face_quality]
        The thinness measure ``min_quality`` gates on.

    Examples
    --------
    The second triangle is a sliver, far below the default thinness gate:

    ```python
    import warp as wp

    import ordito as od

    vertices = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.001, 0.0]], dtype=wp.vec3
    )
    faces = wp.array([0, 1, 2, 1, 3, 2], dtype=wp.int32)
    print(od.validation.face_defective_mask(vertices, faces).numpy().tolist())  # [False, True]
    ```

    One face of a sphere reversed against its neighbours is caught by the normal criterion:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    rows = faces.numpy().reshape(-1, 3)
    rows[5] = rows[5][::-1]
    faces = wp.array(rows.ravel(), dtype=wp.int32)
    mask = od.validation.face_defective_mask(vertices, faces, max_normal_angle=60.0)
    print(np.flatnonzero(mask.numpy()).tolist())  # [5]
    ```
    """
    require_same_device(vertices=vertices, faces=faces, face_normals=face_normals, mates=mates)
    for name, angle in (("max_normal_angle", max_normal_angle), ("max_fold_angle", max_fold_angle)):
        if angle is not None and not 0.0 < angle <= 180.0:
            raise ValueError(f"{name} must be in (0, 180] degrees, got {angle}")
    if normal_criterion not in ("consensus", "meshlab"):
        raise ValueError(
            f"normal_criterion must be 'consensus' or 'meshlab', got {normal_criterion!r}"
        )

    device = faces.device
    n_faces = faces.size // 3
    # ``wp.empty``: the mask kernel writes every face on every path.
    out_bad = _launch.empty(n_faces, dtype=wp.bool, device=device)
    if n_faces == 0:
        return out_bad

    # Both angle criteria read the face normals and the halfedge mates; with both off the kernel
    # reads neither, see its comment. Normals not given are formed in the kernel, from the
    # corners, where each is read: no normal table and no launch to write one.
    if max_normal_angle is not None or max_fold_angle is not None:
        mates = mates if mates is not None else od.halfedge.halfedge_mates(faces, vertices.size)
    else:
        face_normals = None
        mates = None

    # -2 is unreachable for a cosine, -1 for the normalized quality and ``FOLD_ANGLE_OFF`` (past
    # pi) for a dihedral, so a disabled criterion simply never fires and the kernel needs no
    # per-criterion flag. The normal criterion is the consensus cosine or the pairwise angle.
    normal_cos = -2.0
    pair_angle = kernel_validation.FOLD_ANGLE_OFF
    if max_normal_angle is not None and normal_criterion == "meshlab":
        pair_angle = wp.float32(math.radians(max_normal_angle))
    elif max_normal_angle is not None:
        normal_cos = math.cos(math.radians(max_normal_angle))
    _launch.launch(
        kernel_validation.face_defective_mask,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            face_normals,
            mates,
            wp.float32(min_quality if min_quality is not None else -1.0),
            wp.float32(normal_cos),
            pair_angle,
            wp.float32(math.radians(max_fold_angle))
            if max_fold_angle is not None
            else kernel_validation.FOLD_ANGLE_OFF,
            out_bad,
        ],
        device=device,
    )
    return out_bad


def _corner_parents_from_mates(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    require_pairs: bool,
    violation: wp.array[wp.int32] | None,
    *,
    oriented: bool = False,
) -> wp.array[wp.int32]:
    """
    Union-find forest of the corner graph, hooked straight off the halfedge mates.

    The pairs [`face_adjacency`][ordito.adjacency.face_adjacency] would emit are the halfedges
    with a mate, so no adjacency table, run scan or host read of its length is needed, and each
    corner-graph edge is formed inside the pre-hook and the hook rather than written to a table
    first; every other halfedge links its own corner to itself. The pre-hook writes every
    corner's parent itself, so no identity forest is laid down before it. With ``require_pairs``
    the pre-hook raises ``violation[0]`` on any edge not shared by exactly two faces; without it
    ``violation`` is never read and may be ``None``. With ``oriented`` two corners are joined only
    across a pair wound both ways, which is
    [`split_non_manifold_vertices`][ordito.repair.split_non_manifold_vertices]' graph rather than
    the vertex-manifold one.
    """
    n = mates.size
    parents = _launch.empty(n, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_validation.mate_corner_prehook,
        dim=n // 3,
        inputs=[faces, mates, require_pairs, oriented],
        outputs=[parents, violation],
        device=faces.device,
    )
    if n >= kernel_connected_components.ECL_COMPRESS_FROM:
        _launch.launch(
            kernel_connected_components.ecl_compress, dim=n, inputs=[parents], device=faces.device
        )
    _launch.launch(
        kernel_validation.mate_corner_hook,
        dim=n,
        inputs=[faces, mates, oriented, parents],
        device=faces.device,
    )
    return parents


def _vertex_manifold_check(
    faces: wp.array[wp.int32],
    n_vertices: int,
    parents: wp.array[wp.int32],
    mask: wp.array[wp.bool] | None,
    violation: wp.array[wp.int32] | None,
    *,
    require_referenced: bool = True,
) -> None:
    """
    Per-vertex manifold test over a hooked corner-graph forest, shared by the predicate and mask.

    Labels the corner graph (node ``3 * f + k`` per face corner) from ``parents`` and flags a
    vertex unless all of its corners land in one component. Exactly one of ``mask`` and
    ``violation`` is given. A ``mask`` arrives zeroed, length ``n_vertices``, and leaves ``True``
    at every referenced vertex whose corners share one component. A ``violation`` flag is raised
    unless every vertex in ``[0, max(faces)]`` is referenced and manifold -- libigl's predicate --
    which holds over any ``n_vertices`` at least ``max(faces) + 1``, so a caller's vertex count
    serves as well as the bound. ``require_referenced=False`` drops the "referenced" half, which is
    Open3D's predicate and the one [`is_watertight`][ordito.validation.is_watertight] composes.
    The flag is the caller's, so it can already carry another test's verdict and be read back once
    for both.

    ``n_vertices`` is a parameter rather than derived because the two public callers size their
    answer differently on purpose:
    [`is_vertex_manifold`][ordito.validation.is_vertex_manifold] uses ``max(faces) + 1`` (libigl's
    convention, so unreferenced vertices in that range count as non-manifold) while
    [`vertex_manifold_mask`][ordito.validation.vertex_manifold_mask] uses the caller's vertex
    buffer length.
    """
    device = faces.device
    n_corners = faces.size // 3 * 3
    labels = _launch.empty(n_corners, dtype=wp.int32, device=device)
    min_label = _launch.full(n_vertices, odt.dtype_max(wp.int32), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_validation.corner_labels_and_vertex_min,
        dim=n_corners,
        inputs=[faces, parents, labels, min_label, mask],
        device=device,
    )
    _launch.launch(
        kernel_validation.corner_vertex_check,
        dim=n_corners,
        inputs=[faces, labels, min_label, wp.int32(require_referenced), mask, violation],
        device=device,
    )


def _halfedge_keys(faces: wp.array[wp.int32]) -> wp.array[wp.uint64]:
    """
    Pack the undirected key of every halfedge, in halfedge order, against ``INDEX_RADIX_PAIR``.

    Read straight off ``faces``: it is the rows ``hash_indices_rows`` of ``edges_sorted`` would
    give, for less.
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
