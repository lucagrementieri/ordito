"""Face-subset extraction and vertex-selection morphology."""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, read_values, require_same_device
from triwarp.constants import INDEX_RADIX_PAIR
from triwarp.halfedge import halfedge_twins, require_matching_twins
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import grouping as kernel_grouping
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import selection as kernel_selection
from triwarp.kernels.algorithms import connected_components as kernel_connected_components


def region_boundary_edges(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    n_vertices: int | None = None,
    *,
    oriented: bool = False,
) -> twt.Array2dInt32:
    """
    Interior edges on the boundary of a face region.

    Returns the edges that have exactly two incident faces, exactly one of which is in
    ``face_mask`` — i.e. the interior seam separating the region from the rest of the mesh (mesh
    boundary edges, with a single incident face, are excluded).

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` region mask.
    n_vertices
        Optional exclusive bound on the vertex indices, used as the edge-key radix so the sort
        orders only the bits a key can occupy. It is trusted, not checked. Without it the keys pack
        against [`constants.INDEX_RADIX_PAIR`][triwarp.constants.INDEX_RADIX_PAIR], which needs no
        bound and orders them identically.
    oriented
        Return each row as the **directed** pair belonging to its region face, instead of the
        ascending pair. That puts the region on the left of the contour, which is the orientation
        [`faces_left_of_contour`][triwarp.selection.faces_left_of_contour] reads — the two are
        inverse and round-trip exactly. The default ascending form carries no orientation at all,
        so feeding it to that function seeds *both* sides and returns the whole mesh.

    Returns
    -------
    twt.Array2dInt32
        ``(k, 2)`` vertex pairs on ``faces.device``, sorted ascending per row unless ``oriented``.

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.
    RuntimeError
        If ``faces`` and ``face_mask`` are not all on one device.

    See Also
    --------
    [`faces_left_of_contour`][triwarp.selection.faces_left_of_contour]
        The inverse: turns this seam back into the region it bounds, given ``oriented=True``.
    [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]
    """
    require_same_device(faces=faces, face_mask=face_mask)
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )
    if n_faces == 0:
        return twt.empty_2d((0, 2), wp.int32, device=device)
    # One radix sort of every halfedge's edge key, payload its halfedge index: a seam edge is a run
    # of exactly two keys whose faces straddle the region, and it is emitted from its region
    # halfedge in ascending key order -- no unique-edge table, and one readback, of the seam size.
    n = 3 * n_faces
    radix = n_vertices if n_vertices else INDEX_RADIX_PAIR
    keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
    order = wp.empty(2 * n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_adjacency.face_edge_keys_and_order,
        dim=n_faces,
        inputs=[faces, wp.uint64(radix), keys, order],
        device=device,
    )
    wp.utils.radix_sort_pairs(
        keys, order, count=n, end_bit=min(64, max(1, (radix * radix - 1).bit_length()))
    )
    # The sort leaves its result in the leading halves, so the payload buffer's upper half is free
    # scratch: the seam flags are written and scanned there instead of in a fresh ``n`` buffer.
    inclusive = twt.as_dense(order[n:])
    wp.launch(
        kernel_selection.mark_region_seam,
        dim=n,
        inputs=[keys, order, face_mask, wp.int32(n), inclusive],
        device=device,
    )
    wp.utils.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # Sizes the output: the one host readback.
    n_seam = int(read_scalar(inclusive))
    out_edges = twt.empty_2d((n_seam, 2), wp.int32, device=device)
    if n_seam > 0:
        wp.launch(
            kernel_selection.emit_region_seam,
            dim=n,
            inputs=[inclusive, order, faces, face_mask, oriented, out_edges],
            device=device,
        )
    return out_edges


def faces_left_of_contour(
    faces: wp.array[wp.int32],
    contour_edges: twt.Array2dInt32,
    *,
    n_vertices: int | None = None,
    twins: wp.array[wp.int32] | None = None,
) -> wp.array[wp.bool]:
    """
    Flood-fill the faces on the left of a directed contour of mesh edges.

    The dual of [`region_boundary_edges`][triwarp.selection.region_boundary_edges]: that one turns a
    face region into the seam around it, this one turns a seam back into a region. Which of the two
    sides comes back is decided **only** by the contour's direction: the seeds are the faces that
    own the halfedges ``a -> b``, and a face's corners run counter-clockwise, so reversing the rows
    returns the complement. Nothing else about the contour matters, and a closed contour on a closed
    mesh therefore partitions it exactly.

    The fill is a connected-component labelling of the face-adjacency graph with the contour's dual
    edges removed, so a contour that does *not* separate the mesh returns everything reachable --
    which is the whole surface. That is the honest answer rather than a failure, but it means the
    result is worth checking against the input's face count when the contour is meant to close.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    contour_edges
        ``(k, 2)`` ``wp.int32`` **directed** vertex pairs, each an edge of the mesh. Consecutive
        rows need not be connected: the fill only cares about which edges are blocked and which
        halfedges seed it, so several disjoint contours can be passed at once. A row that is not a
        mesh edge blocks nothing and seeds nothing.
    n_vertices
        Optional total vertex count, forwarded to
        [`halfedge_twins`][triwarp.halfedge.halfedge_twins] as its key radix when ``twins`` is
        built here. Never inferred: every key this function packs itself uses
        [`constants.INDEX_RADIX_PAIR`][triwarp.constants.INDEX_RADIX_PAIR], which needs no bound.
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins]. Building it is the
        single largest cost here, so pass it when several contours are filled on one mesh.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_faces`` mask, ``True`` for the faces on the left of the contour, on
        ``faces.device``. All-``False`` only when no contour row is a mesh edge whose left face
        exists.

    Raises
    ------
    TypeError
        If ``contour_edges`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``contour_edges`` does not have two columns, or if ``twins`` is given and is not a
        twin table for ``faces``
        ([`require_matching_twins`][triwarp.halfedge.require_matching_twins] states what that
        means, and checking it costs one launch and one readback).
    RuntimeError
        If ``faces``, ``contour_edges`` and ``twins`` are not all on one device.

    Examples
    --------
    ```python
    seam = tw.selection.region_boundary_edges(f, face_mask, oriented=True)
    left = tw.selection.faces_left_of_contour(f, seam)
    assert np.array_equal(left.numpy(), face_mask.numpy())
    ```

    See Also
    --------
    [`region_boundary_edges`][triwarp.selection.region_boundary_edges]
        The dual: the seam around a region, which this turns back into a region.
    [`cut_along_edges`][triwarp.seams.cut_along_edges]
        Splits the mesh along the same kind of edge set instead of labelling its sides.
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        Turns the returned mask into a mesh.
    """
    require_same_device(faces=faces, contour_edges=contour_edges, twins=twins)
    require_matching_twins(faces, twins)
    twt.ensure_edge_pairs(contour_edges, "contour_edges")
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    n_contour = int(contour_edges.shape[0])
    if n_faces == 0 or n_contour == 0:
        return wp.zeros(n_faces, dtype=wp.bool, device=device)
    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    # The directed keys pack against the pair radix, which needs no vertex bound and cannot alias a
    # contour row whose index is past the mesh onto a real edge.
    base = wp.uint64(INDEX_RADIX_PAIR)

    # Directed contour keys, sorted, and small: the kernels probe this once or twice per halfedge,
    # so a `k`-entry binary search replaces sorting a key per halfedge to look 320 of them up. The
    # keys are packed straight into the sort's double-width buffer; the payload is never read.
    contour_keys = wp.empty(2 * n_contour, dtype=wp.uint64, device=device)
    wp.launch(
        kernel_grouping.pack_directed_index_keys,
        dim=n_contour,
        inputs=[contour_edges, base, contour_keys],
        device=device,
    )
    wp.utils.radix_sort_pairs(
        contour_keys, wp.empty(2 * n_contour, dtype=wp.int32, device=device), count=n_contour
    )
    contour_keys = twt.as_dense(contour_keys[:n_contour])

    # The fill is a union-find over the face-adjacency graph with the contour's dual edges removed,
    # formed across ``twins`` in the thread rather than listed: a pre-hook (with the seeding), a
    # hook, and a flatten that also flags each seeded root.
    n_halfedges = 3 * n_faces
    seeds = wp.zeros(n_faces, dtype=wp.bool, device=device)
    parents = tw.array.arange(n_faces, device=device)
    wp.launch(
        kernel_selection.seed_and_prehook_dual,
        dim=n_halfedges,
        inputs=[faces, twins, contour_keys, base, parents, seeds],
        device=device,
    )
    wp.launch(
        kernel_selection.hook_dual,
        dim=n_halfedges,
        inputs=[faces, twins, contour_keys, base, parents],
        device=device,
    )
    # The root flags overwrite ``seeds``: only a seeded component's root is ever written, and to
    # ``True``, so a thread reading a written entry is that root reading its own component's
    # answer, and every entry the gather below reads -- a root's -- ends ``True`` exactly when its
    # component holds a seed. The labels cannot overwrite ``parents`` the same way: another
    # thread's path halving may write a stale ancestor over a label already stored.
    labels = wp.empty(n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.label_flagged_components,
        dim=n_faces,
        inputs=[parents, seeds, True, labels, seeds],
        device=device,
    )
    # A label names a representative face, so the per-face answer is a gather of the per-label flag,
    # and it writes every entry.
    left = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.copy(left, seeds[labels])
    return left


def exclude_fully_selected_components(
    faces: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    n_vertices: int,
    unique_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.bool]:
    """
    Drop selected vertices whose entire connected component is selected.

    A vertex-connected component that is wholly inside ``mask`` would make a region-smoothing
    Dirichlet system singular (no fixed anchor), so those components are removed from the
    selection; components with at least one unselected vertex are kept intact.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    mask
        Length-``n_vertices`` ``wp.bool`` selection.
    n_vertices
        Vertex count (component labels span ``0 .. n_vertices - 1``).
    unique_edges
        Optional precomputed ``(m, 2)`` edges to label the components over; the faces' own
        edges when ``None``, which give the same components.

    Returns
    -------
    wp.array[wp.bool]
        Selection with fully-selected components removed, on ``mask.device``.

    Raises
    ------
    ValueError
        If ``mask`` does not have one entry per vertex.
    RuntimeError
        If ``faces``, ``mask`` and ``unique_edges`` are not all on one device.
    """
    require_same_device(faces=faces, mask=mask, unique_edges=unique_edges)
    if int(mask.shape[0]) != n_vertices:
        raise ValueError(
            f"mask must have one entry per vertex, got {mask.shape[0]} for n_vertices={n_vertices}"
        )
    device = mask.device
    if n_vertices == 0:
        return wp.clone(mask)
    # A union-find labels every component by its smallest vertex whatever the order and
    # multiplicity of the unions, so the faces' own edges, formed in the thread, give the identical
    # labels without an edge table. The endpoints are the face buffer's indices, which this
    # function trusts to be below ``n_vertices`` as the rest of its caller's pipeline does.
    parents = tw.array.arange(n_vertices, device=device)
    if unique_edges is None:
        n_faces = int(faces.shape[0]) // 3
        if n_faces > 0:
            for kernel in (kernel_selection.prehook_face_edges, kernel_selection.hook_face_edges):
                wp.launch(kernel, dim=n_faces, inputs=[faces, parents], device=device)
    elif int(unique_edges.shape[0]) > 0:
        for kernel in (
            kernel_connected_components.ecl_init_parent_edges,
            kernel_connected_components.ecl_hook_edges,
        ):
            wp.launch(
                kernel,
                dim=int(unique_edges.shape[0]),
                inputs=[unique_edges, parents],
                device=device,
            )
    # One flatten that also flags each component holding an unselected vertex -- one that is not
    # fully selected, and so keeps its selection.
    labels = wp.empty(n_vertices, dtype=wp.int32, device=device)
    keep = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    wp.launch(
        kernel_selection.label_flagged_components,
        dim=n_vertices,
        inputs=[parents, mask, False, labels, keep],
        device=device,
    )
    out = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.launch(
        kernel_selection.keep_selected_by_component,
        dim=n_vertices,
        inputs=[mask, labels, keep, out],
        device=device,
    )
    return out


@overload
def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Extract a face subset by index and reindex vertices from zero.

    Gathers the selected face triplets, compacts referenced vertices, and remaps
    face indices into the compact vertex buffer. Matches the core geometry step of
    [`trimesh.util.submesh`][] (without visuals, repair, watertight filtering, or
    append).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_indices
        1D ``wp.int32`` array of face indices into the source mesh
        (``0 .. n_faces - 1``), on the same device as ``vertices``. May repeat a face: each
        occurrence keeps its own row of ``sub_faces``.
    return_index
        If ``True``, also return the vertex map below -- which the extraction computes anyway, so it
        costs nothing.

    Returns
    -------
    sub_vertices : wp.array[wp.vec3]
        Compact positions on ``vertices.device``, length ``0`` when ``face_indices`` is empty.
    sub_faces : wp.array[wp.int32]
        Flat triangle index buffer into ``sub_vertices``.
    vertex_index : wp.array[wp.int32]
        Only when ``return_index`` is ``True``: length ``n_sub_vertices``, the **input** vertex each
        output vertex came from, ascending. That direction makes it a gather, so a per-vertex
        attribute follows the submesh with ``tw.array.gather(attribute, vertex_index)``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``face_indices`` are not all on one device.

    See Also
    --------
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
    [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]
    [`concatenate`][triwarp.combine.concatenate]
    [`trimesh.util.submesh`][]
    """
    require_same_device(vertices=vertices, faces=faces, face_indices=face_indices)
    device = vertices.device
    k = int(face_indices.shape[0])
    if k == 0:
        empty_vertices = wp.empty(0, dtype=wp.vec3, device=device)
        empty_faces = wp.empty(0, dtype=wp.int32, device=device)
        if return_index:
            return empty_vertices, empty_faces, wp.empty(0, dtype=wp.int32, device=device)
        return empty_vertices, empty_faces

    # The referenced vertices are a mask over the vertex count, and its in-place scan is both the
    # sorted unique set and every corner's compact rank: no dedup of ``face_indices`` (a duplicated
    # face reaches the vertices it already reached, and keeps its own row), no face gather and no
    # ``unique_1d``. One readback, of the vertex count.
    n_vertices = int(vertices.shape[0])
    inclusive = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.mark_indexed_face_vertices,
        dim=k,
        inputs=[faces, face_indices, inclusive],
        device=device,
    )
    wp.utils.array_scan(inclusive, out_array=inclusive, inclusive=True)
    n_sub = int(read_scalar(inclusive))
    sub_vertices = wp.empty(n_sub, dtype=wp.vec3, device=device)
    sub_faces = wp.empty(3 * k, dtype=wp.int32, device=device)
    vertex_index = wp.empty(n_sub, dtype=wp.int32, device=device) if return_index else None
    wp.launch(
        kernel_selection.compact_indexed_submesh,
        dim=max(k, n_vertices),
        inputs=[vertices, faces, face_indices, inclusive, sub_faces, sub_vertices, vertex_index],
        device=device,
    )
    if vertex_index is not None:
        return sub_vertices, sub_faces, vertex_index
    return sub_vertices, sub_faces


def submeshes_from_face_groups(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    group_face_indices: wp.array[wp.int32],
    group_offsets: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Extract many face subsets at once: CSR groups in, CSR submeshes out.

    The batched form of
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]. Every launch count
    is ``O(1)`` in the number of groups, so extracting ten thousand components costs the same
    number of kernel launches and host synchronisations as extracting one — which is the whole
    point, since the per-group loop it replaces pays a sort, a scan and a readback *each*.

    How it stays batched: each corner of each selected face is packed into the single ``int64`` key
    ``group * n_vertices + vertex``, and one global
    [`unique_1d`][triwarp.grouping.unique_1d] deduplicates all groups together. Because the key is
    ``group * radix + vertex``, ascending key order is exactly ``(group, vertex)`` lexicographic
    order, so the unique slots come out already partitioned by group **and** ascending by original
    vertex index within each group — bit-identical to running ``unique_1d`` per group.

    A vertex shared by two groups is duplicated into both, matching
    [`trimesh.util.submesh`][] and the per-group loop.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    group_face_indices
        Concatenated face indices of every group, as ``wp.int32``. Indices must be unique within a
        group (duplicates would produce duplicate output faces).
    group_offsets
        Length-``k + 1`` total-terminated ``wp.int32`` offsets of the groups in
        ``group_face_indices`` -- the package's one convention, stated at
        [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]. Groups must be non-empty.

    Returns
    -------
    vertices_all : wp.array[wp.vec3]
        Every group's compacted vertices, concatenated.
    vertex_offsets : wp.array[wp.int32]
        Length-``k + 1`` total-terminated offsets; group ``g`` owns
        ``vertices_all[vertex_offsets[g] : vertex_offsets[g + 1]]``.
    faces_all : wp.array[wp.int32]
        Every group's reindexed flat faces, concatenated in the same group order. Group ``g`` owns
        ``faces_all[3 * group_offsets[g] : 3 * group_offsets[g + 1]]`` — the *input* offsets,
        scaled by three, because a group keeps exactly the faces it was given.

    Raises
    ------
    ValueError
        If ``k * n_vertices`` overflows the ``int64`` packing budget of ``2 ** 62``.
    RuntimeError
        If ``vertices``, ``faces``, ``group_face_indices`` and ``group_offsets`` are not all on one
        device.

    See Also
    --------
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`split_with_offsets`][triwarp.combine.split_with_offsets]
    [`unique_1d`][triwarp.grouping.unique_1d]
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        group_face_indices=group_face_indices,
        group_offsets=group_offsets,
    )
    device = vertices.device
    k = max(int(group_offsets.shape[0]) - 1, 0)
    n_selected = int(group_face_indices.shape[0])
    n_vertices = int(vertices.shape[0])
    if k == 0 or n_selected == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.zeros(k + 1, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )
    radix = max(n_vertices, 1)
    if k * radix >= 1 << 62:
        raise ValueError(
            f"submeshes_from_face_groups cannot pack {k} groups x {radix} vertices into int64"
        )

    n_corners = 3 * n_selected
    keys = wp.empty(n_corners, dtype=wp.int64, device=device)
    wp.launch(
        kernel_selection.pack_group_vertex_keys,
        dim=n_corners,
        inputs=[faces, group_face_indices, group_offsets, wp.int64(radix), keys],
        device=device,
    )

    # A key is ``group * radix + vertex``, below ``k * radix``.
    unique_keys, inverse = tw.grouping.unique_1d(keys, return_inverse=True, max_value=k * radix - 1)
    n_slots = int(unique_keys.shape[0])
    # One decode per unique slot answers everything its key is needed for: the slot's group, the
    # group's vertex count (a histogram, so an empty group still gets a zero-length entry and no
    # host synchronisation is needed to size it), and the source position it names.
    slot_groups = wp.empty(n_slots, dtype=wp.int32, device=device)
    # The counts land behind the leading zero of the ``k + 1`` offsets buffer and are scanned there
    # in place, which leaves the total-terminated vertex offsets with no second buffer.
    vertex_offsets = wp.zeros(k + 1, dtype=wp.int32, device=device)
    group_counts = twt.as_dense(vertex_offsets[1:])
    vertices_all = wp.empty(n_slots, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_selection.decode_group_vertex_keys,
        dim=n_slots,
        inputs=[unique_keys, wp.int64(radix), vertices, slot_groups, group_counts, vertices_all],
        device=device,
    )
    wp.utils.array_scan(group_counts, out_array=group_counts, inclusive=True)

    faces_all = wp.empty(n_corners, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.local_corner_indices,
        dim=n_corners,
        inputs=[inverse, slot_groups, vertex_offsets, faces_all],
        device=device,
    )
    return vertices_all, vertex_offsets, faces_all


@overload
def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Extract a face subset selected by a per-face boolean mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` array on the same device as ``vertices``.
    return_index
        If ``True``, also return the output-to-input vertex map, as
        [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices] documents.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]] | tuple[..., wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``, plus ``vertex_index`` when
        ``return_index`` is ``True``.

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_mask`` are not all on one device.

    See Also
    --------
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]
        The complement: keep everything *outside* a region, and report the rims that opens.
    """
    require_same_device(vertices=vertices, faces=faces, face_mask=face_mask)
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )
    sub_vertices, sub_faces, vertex_index, _ = _submesh_from_mask(
        vertices, faces, face_mask, keep_masked=True
    )
    if return_index:
        return sub_vertices, sub_faces, vertex_index
    return sub_vertices, sub_faces


def delete_region_keep_boundary(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_mask: wp.array[wp.bool]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], list[wp.array[wp.int32]]]:
    """
    Remove a face region and report the boundary loops the removal opened.

    The complement of
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]: it keeps everything the
    mask does *not* select, and it answers the question that makes the result usable -- **which rims
    are new**. That is the distinction a caller needs and cannot recover afterwards: the surviving
    mesh's boundary loops are the rims the deletion made *plus* whatever rims the input already had,
    and only the former should be filled. It is the natural input to
    [`triwarp.holes.fill_min_weight`][triwarp.holes.fill_min_weight], and the first half of
    [`triwarp.holes.refill_region`][triwarp.holes.refill_region].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` array, ``True`` for each face to **delete**.

    Returns
    -------
    kept_vertices : wp.array[wp.vec3]
        Compact positions of the surviving submesh.
    kept_faces : wp.array[wp.int32]
        Flat triangle index buffer into ``kept_vertices``.
    new_loops : list[wp.array[wp.int32]]
        The boundary loops the deletion opened, as vertex-index cycles **into** ``kept_vertices`` --
        the same form [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops] returns.
        A loop the input already had is excluded, so an open input's original rims do not appear.

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_mask`` are not all on one device.

    Examples
    --------
    ```python
    doomed = tw.validation.face_self_intersecting_mask(v, f)
    kept_v, kept_f, rims = tw.selection.delete_region_keep_boundary(v, f, doomed)
    ```

    Notes
    -----
    A loop counts as pre-existing when **every** one of its edges was already a boundary edge of the
    input. That is the right test rather than "any": deleting a face that touches an existing rim
    extends that rim rather than opening a new one, and the extended loop has to be reported --
    otherwise a caller filling only the new loops would leave the extension open.

    The test is answered from the deletion rather than from the input's boundary: a loop edge the
    survivor holds in exactly one face was an input boundary edge exactly when no deleted face
    contains it, so the classification sorts only the deleted faces' edges, never the mesh's. That
    holds for every loop [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops] can
    return, since each of its consecutive pairs is a boundary edge on any input. On a mesh that is
    not edge-manifold a rim can be missing from that trace altogether, and then it is not reported
    either. The call's cost is dominated by the loop trace and the submesh extraction, so anything
    spent optimizing this function further belongs in
    [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops] rather than here.

    See Also
    --------
    [`delete_region_keep_boundary_with_offsets`][triwarp.selection.delete_region_keep_boundary_with_offsets]
        The same, with the new loops packed into one buffer and their offsets.
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        The same extraction with the mask's polarity reversed and no loop report.
    [`triwarp.holes.refill_region`][triwarp.holes.refill_region]
        Delete and immediately fill, which is what this is usually the first half of.
    """
    kept_vertices, kept_faces, _, _, views = _delete_region_loops(
        vertices, faces, face_mask, host_bounds=True
    )
    assert views is not None
    return kept_vertices, kept_faces, views


def delete_region_keep_boundary_with_offsets(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_mask: wp.array[wp.bool]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Remove a face region and report the boundary loops it opened, packed with their offsets.

    The packed form of
    [`delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]: the same
    surviving mesh and the same new loops in the same order, the loops concatenated into one buffer
    with no per-loop Python object. It is the form
    [`triwarp.boundary.boundary_loops_with_offsets`][triwarp.boundary.boundary_loops_with_offsets]
    returns, so a packed hole filler can take it directly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` array, ``True`` for each face to **delete**.

    Returns
    -------
    kept_vertices : wp.array[wp.vec3]
        Compact positions of the surviving submesh.
    kept_faces : wp.array[wp.int32]
        Flat triangle index buffer into ``kept_vertices``.
    loops : wp.array[wp.int32]
        Every new boundary loop's vertex cycle **into** ``kept_vertices``, loop after loop.
    loop_offsets : wp.array[wp.int32]
        ``(n_loops + 1,)`` total-terminated offsets: loop ``i`` is
        ``loops[loop_offsets[i] : loop_offsets[i + 1]]``, and ``[0]`` when the deletion opened no
        rim.

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_mask`` are not all on one device.

    Notes
    -----
    Which loops count as new is decided exactly as in
    [`delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]'s Notes. The kept
    loops are compacted on the device, and the one readback is the scan's tail, which sizes the
    buffer.

    See Also
    --------
    [`delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]
        The same, with one array per new loop.
    [`triwarp.boundary.boundary_loops_with_offsets`][triwarp.boundary.boundary_loops_with_offsets]
    """
    kept_vertices, kept_faces, loops, loop_offsets, _ = _delete_region_loops(
        vertices, faces, face_mask, host_bounds=False
    )
    return kept_vertices, kept_faces, loops, loop_offsets


def _delete_region_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    host_bounds: bool,
) -> tuple[
    wp.array[wp.vec3],
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.int32],
    list[wp.array[wp.int32]] | None,
]:
    """
    Both ``delete_region_keep_boundary`` forms: ``(vertices, faces, loops, offsets, views)``.

    With ``host_bounds`` the one readback is every loop's verdict and extent, and ``views`` cuts
    the kept loops straight out of the traced ones (``loops`` / ``offsets`` are then the traced,
    unfiltered pair and not an answer). Without it the verdicts are scanned on the device, the kept
    loops compacted into ``loops`` / ``offsets``, the one readback is the scan's tail, and
    ``views`` is ``None``. Both forms share the classification, so they cannot disagree about
    which loops are new.
    """
    require_same_device(vertices=vertices, faces=faces, face_mask=face_mask)
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )

    kept_vertices, kept_faces, vertex_index, kept_ranks = _submesh_from_mask(
        vertices, faces, face_mask, keep_masked=False
    )

    def no_loops() -> tuple[
        wp.array[wp.vec3],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        list[wp.array[wp.int32]] | None,
    ]:
        # Built only when returned: two allocations the common path must not pay.
        return (
            kept_vertices,
            kept_faces,
            wp.empty(0, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            [] if host_bounds else None,
        )

    if int(kept_faces.shape[0]) == 0:
        return no_loops()
    # Nothing deleted opens no rim: every loop the submesh has is the input's own. The face count
    # says so without a readback, and it skips the loop extraction as well as the classification.
    if int(kept_faces.shape[0]) == 3 * n_faces:
        return no_loops()

    flat_loops, loop_offsets = tw.boundary.boundary_loops_with_offsets(kept_vertices, kept_faces)
    n_loops = int(loop_offsets.shape[0]) - 1
    if n_loops == 0:
        return no_loops()

    # Classify every loop at once on the device, from the deletion rather than from the input's
    # boundary, so no table here is mesh-sized: the deleted faces' edges are sorted, and each loop
    # edge is searched in them (see ``loops_are_input_rims`` for why that agrees with the
    # input-boundary test on every loop the trace can return).
    #
    # Done on the host this was a readback per loop plus a Python membership test per rim edge, so
    # its cost grew with the *loop count* as much as with the mesh.
    n_kept = int(kept_faces.shape[0]) // 3
    # The deleted count is the face count's complement, and a deleted face's rank among the
    # deleted ones follows from the extraction's kept-face ranks, so the region needs neither a
    # readback nor a scan of its own.
    base = wp.uint64(int(vertices.shape[0]))
    n_deleted_keys = 3 * (n_faces - n_kept)
    # The keys are written into the leading half of the radix sort's double-width scratch, and the
    # payload is never read, so neither is seeded: only the sorted keys are wanted.
    key_buffer = wp.empty(2 * n_deleted_keys, dtype=wp.uint64, device=device)
    wp.launch(
        kernel_selection.deleted_face_edge_keys,
        dim=n_faces,
        inputs=[faces, face_mask, kept_ranks, base, key_buffer],
        device=device,
    )
    payload = wp.empty(2 * n_deleted_keys, dtype=wp.int32, device=device)
    # A key is ``min + max * n_vertices``, below ``n_vertices ** 2``: only those bits are sorted.
    n_vertices = int(vertices.shape[0])
    wp.utils.radix_sort_pairs(
        key_buffer,
        payload,
        count=n_deleted_keys,
        end_bit=max(1, (n_vertices * n_vertices - 1).bit_length()),
    )
    deleted_keys = twt.as_dense(key_buffer[:n_deleted_keys])

    kept_counts = wp.empty(n_loops, dtype=wp.vec3i, device=device)
    wp.launch(
        kernel_selection.loops_are_input_rims,
        dim=n_loops,
        inputs=[flat_loops, loop_offsets, vertex_index, deleted_keys, base, kept_counts],
        device=device,
    )
    if host_bounds:
        # The list form's one readback: each loop's verdict and extent, from which its views are
        # cut straight out of the traced loops -- no scan and no compaction.
        counts_np = kept_counts.numpy()
        views = [
            twt.as_dense(flat_loops[int(start) : int(start) + int(size)])
            for kept, size, start in counts_np
            if kept
        ]
        return kept_vertices, kept_faces, flat_loops, loop_offsets, views
    wp.utils.array_scan(kept_counts, out_array=kept_counts, inclusive=True)
    # The packed form's one readback: the scan's tail sizes the kept loops.
    tail = read_scalar(kept_counts)
    n_new, n_values = int(tail[0]), int(tail[1])
    if n_new == 0:
        return no_loops()
    loops = wp.empty(n_values, dtype=wp.int32, device=device)
    offsets = wp.zeros(n_new + 1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.compact_kept_loops,
        dim=n_loops,
        inputs=[flat_loops, loop_offsets, kept_counts, loops, offsets],
        device=device,
    )
    return kept_vertices, kept_faces, loops, offsets, None


def _submesh_from_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    keep_masked: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Extract the faces whose mask entry equals ``keep_masked``, reindexed from zero.

    Returns ``(sub_vertices, sub_faces, vertex_index, ranks)``, the first three as
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices] returns them, and
    ``ranks`` the ``(n_faces + n_vertices,)`` inclusive scan whose leading ``n_faces`` entries rank
    the kept faces. One scan orders the kept faces and the vertices they reference, and a single
    two-value readback sizes both outputs.
    """
    device = vertices.device
    n_faces = int(face_mask.shape[0])
    n_vertices = int(vertices.shape[0])
    if n_faces == 0:
        empty_index = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=device), empty_index, empty_index, empty_index

    ranks = wp.zeros(n_faces + n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.mark_submesh_faces_and_vertices,
        dim=n_faces,
        inputs=[faces, face_mask, keep_masked, ranks],
        device=device,
    )
    wp.utils.array_scan(ranks, out_array=ranks, inclusive=True)
    counts = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.submesh_counts, dim=1, inputs=[ranks, n_faces, counts], device=device
    )
    # One readback sizes both outputs.
    n_kept_faces, n_kept_vertices = read_values(counts, 0, 2)

    sub_vertices = wp.empty(n_kept_vertices, dtype=wp.vec3, device=device)
    sub_faces = wp.empty(3 * n_kept_faces, dtype=wp.int32, device=device)
    vertex_index = wp.empty(n_kept_vertices, dtype=wp.int32, device=device)
    if n_kept_faces > 0:
        wp.launch(
            kernel_selection.compact_submesh,
            dim=max(n_faces, n_vertices),
            inputs=[
                vertices,
                faces,
                face_mask,
                keep_masked,
                ranks,
                sub_faces,
                sub_vertices,
                vertex_index,
            ],
            device=device,
        )
    return sub_vertices, sub_faces, vertex_index, ranks


def submesh_from_vertex_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_indices: wp.array[wp.int32],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract faces incident on the given vertices and reindex from zero.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    vertex_indices
        1D ``wp.int32`` array of vertex indices on the same device as ``vertices``.
    face_mode
        ``"all"`` selects faces whose three vertex indices all lie in ``vertex_indices``;
        ``"any"`` selects faces with at least one vertex index in ``vertex_indices``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``vertex_indices`` are not all on one device.

    See Also
    --------
    [`face_indices_from_vertex_indices`][triwarp.selection.face_indices_from_vertex_indices]
    [`submesh_from_vertex_mask`][triwarp.selection.submesh_from_vertex_mask]
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    """
    require_same_device(vertices=vertices, faces=faces, vertex_indices=vertex_indices)
    face_indices = face_indices_from_vertex_indices(
        faces, vertex_indices, face_mode=face_mode, n_vertices=int(vertices.shape[0])
    )
    return submesh_from_face_indices(vertices, faces, face_indices)


def submesh_from_vertex_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract faces incident on vertices selected by a boolean mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    vertex_mask
        Length-``n_vertices`` ``wp.bool`` array on the same device as ``vertices``.
    face_mode
        ``"all"`` selects faces whose three corners are all selected; ``"any"`` selects faces with
        at least one corner selected. Same rule as
        [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]'s.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``vertex_mask`` length does not equal ``n_vertices``, or ``face_mode`` is not
        ``"all"`` or ``"any"``.
    RuntimeError
        If ``vertices``, ``faces`` and ``vertex_mask`` are not all on one device.

    Notes
    -----
    A mask is a *cheaper* input than the equivalent index list, not merely a more convenient one:
    the face reduction reads it directly, where the index form has to rebuild membership through
    [`isin`][triwarp.array.isin] first. The index form still pays that cost, because it starts from
    indices and has nothing else to go on.

    See Also
    --------
    [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]
        The index form, for a caller that holds a list rather than a mask.
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        The per-face selection this reduces onto.
    """
    require_same_device(vertices=vertices, faces=faces, vertex_mask=vertex_mask)
    n_vertices = int(vertices.shape[0])
    if int(vertex_mask.shape[0]) != n_vertices:
        raise ValueError(
            f"vertex_mask length must equal n_vertices={n_vertices}, got {vertex_mask.shape[0]}"
        )

    face_mask = _face_mask_from_vertex_mask(faces, vertex_mask, face_mode=face_mode)
    return submesh_from_face_mask(vertices, faces, face_mask)


def _face_mask_from_vertex_mask(
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> wp.array[wp.bool]:
    """
    Reduce a per-vertex mask onto a per-face mask under the all/any rule.

    The mask counterpart of
    [`face_indices_from_vertex_indices`][triwarp.selection.face_indices_from_vertex_indices], and
    the reason a mask input is cheaper: one kernel with three O(1) mask lookups per face, against
    that function's ``isin`` (two min/max reductions with a host readback each, then either a
    radix sort or a span-sized lookup table) plus a row reduction plus a ``flatnonzero``.
    """
    if face_mode not in ("all", "any"):
        raise ValueError(f'face_mode must be "all" or "any", got {face_mode!r}')

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    out_face_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    if n_faces == 0:
        return out_face_mask

    wp.launch(
        kernel_selection.face_mask_from_vertex_mask,
        dim=n_faces,
        inputs=[faces, vertex_mask, wp.bool(face_mode == "all"), out_face_mask],
        device=device,
    )
    return out_face_mask


def expand_vertex_mask(
    faces: wp.array[wp.int32], mask: wp.array[wp.bool], hops: int
) -> wp.array[wp.bool]:
    """
    Grow a vertex selection by ``hops`` one-ring layers.

    Each round adds every vertex sharing a mesh edge with a currently-selected vertex, so after
    ``hops`` rounds the mask covers all vertices within graph distance ``hops`` of the input
    selection.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    mask
        Length-``n_vertices`` ``wp.bool`` selection to dilate.
    hops
        Number of one-ring dilation rounds (``0`` returns a copy).

    Returns
    -------
    wp.array[wp.bool]
        Dilated mask on ``mask.device``.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``mask`` are not on one device.

    See Also
    --------
    [`shrink_vertex_mask`][triwarp.selection.shrink_vertex_mask]

    Notes
    -----
    Dilation is vertex-based: a vertex enters the mask when any 1-ring neighbour is in it. On a
    triangle mesh two vertices are one-ring neighbours exactly when they share a face, so a round
    marks the corners of every face with a selected corner and no edge table is built. Vertices no
    face references are neither reached nor removed.
    """
    require_same_device(faces=faces, mask=mask)
    n = int(mask.shape[0])
    if hops <= 0 or n == 0:
        return wp.clone(mask)
    return _dilate_vertex_mask(faces, mask, hops, value=True)


def shrink_vertex_mask(
    faces: wp.array[wp.int32], mask: wp.array[wp.bool], hops: int
) -> wp.array[wp.bool]:
    """
    Erode a vertex selection by ``hops`` one-ring layers.

    Implemented as the complement of an [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]
    of the complement: a vertex is removed if any vertex within ``hops`` hops is unselected.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    mask
        Length-``n_vertices`` ``wp.bool`` selection to erode.
    hops
        Number of one-ring erosion rounds.

    Returns
    -------
    wp.array[wp.bool]
        Eroded mask on ``mask.device``.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``mask`` are not on one device.

    See Also
    --------
    [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]

    Notes
    -----
    Erosion here is vertex-based: a vertex survives when every 1-ring neighbour is also in the
    mask. MeshLab's Erode Selection is a *face*-based operation and gives a different answer.
    """
    require_same_device(faces=faces, mask=mask)
    n = int(mask.shape[0])
    if hops <= 0 or n == 0:
        return wp.clone(mask)
    # The dilation of the complement, run on the mask itself with the polarity flipped: a face with
    # an unselected corner clears all three. No complement is formed on either side.
    return _dilate_vertex_mask(faces, mask, hops, value=False)


def _dilate_vertex_mask(
    faces: wp.array[wp.int32], mask: wp.array[wp.bool], hops: int, *, value: bool
) -> wp.array[wp.bool]:
    """
    Spread ``value`` through ``mask`` by ``hops`` one-ring rounds, into a buffer of its own.

    ``True`` dilates the selection and ``False`` erodes it.

    A round reads one mask and writes a copy of it, so two buffers alternate: every round after
    the second refills the buffer the round before last read, rather than allocating a fresh one.
    The caller's ``mask`` is never written. ``hops`` must be positive.
    """
    device = mask.device
    n_faces = int(faces.shape[0]) // 3
    current = mask
    spare = None
    owned = False
    for _ in range(hops):
        if spare is None:
            nxt = wp.clone(current)
        else:
            wp.copy(spare, current)
            nxt = spare
        if n_faces > 0:
            wp.launch(
                kernel_selection.dilate_vertex_mask,
                dim=n_faces,
                inputs=[faces, current, value, nxt],
                device=device,
            )
        spare = current if owned else None
        owned = True
        current = nxt
    return current


def face_indices_from_vertex_indices(
    faces: wp.array[wp.int32],
    vertex_indices: wp.array[wp.int32],
    *,
    face_mode: Literal["all", "any"] = "all",
    n_vertices: int | None = None,
) -> wp.array[wp.int32]:
    """
    Face indices whose vertex indices match a set under an all/any rule.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    vertex_indices
        1D ``wp.int32`` array of vertex indices on the same device as ``faces``.
    face_mode
        ``"all"`` keeps faces whose three vertex indices all lie in ``vertex_indices``;
        ``"any"`` keeps faces with at least one vertex index in ``vertex_indices``.
    n_vertices
        Optional vertex count, the length of the membership mask over the vertices. Supplying it
        skips the reduction and host readback that would otherwise size the mask from
        ``vertex_indices``. Must be greater than every index in ``vertex_indices``.

    Returns
    -------
    wp.array[wp.int32]
        Selected face indices on ``faces.device``. Empty when no faces match or
        ``vertex_indices`` is empty.

    Raises
    ------
    ValueError
        If ``face_mode`` is not ``"all"`` or ``"any"``.
    RuntimeError
        If ``faces`` and ``vertex_indices`` are not all on one device.
    """
    require_same_device(faces=faces, vertex_indices=vertex_indices)
    if face_mode not in ("all", "any"):
        raise ValueError(f'face_mode must be "all" or "any", got {face_mode!r}')

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if int(vertex_indices.shape[0]) == 0 or n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    # A membership mask over the vertices, read per corner, is the ``isin`` over the index list;
    # its reads are range-guarded, so a mask only as long as the listed indices need is exact.
    if n_vertices is None:
        n_vertices = max(tw.array.index_bound(vertex_indices), 0)
    vertex_mask = tw.array.indices_to_mask(vertex_indices, n_vertices, device=device)
    inclusive = wp.empty(n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.face_flags_from_vertex_mask,
        dim=n_faces,
        inputs=[faces, vertex_mask, wp.bool(face_mode == "all"), inclusive],
        device=device,
    )
    wp.utils.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # Sizes the output: the one host readback when ``n_vertices`` is supplied.
    n_selected = int(read_scalar(inclusive))
    selected = wp.empty(n_selected, dtype=wp.int32, device=device)
    if n_selected > 0:
        wp.launch(
            kernel_scatter.scatter_index_where_scanned,
            dim=n_faces,
            inputs=[inclusive, selected],
            device=device,
        )
    return selected
