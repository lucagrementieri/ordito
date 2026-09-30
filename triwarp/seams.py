"""
Feature edges and the topological cut along them.

Two operations that are only useful together. [`crease_edges`][triwarp.seams.crease_edges] finds the
edges where the surface bends sharply — the ones a modeller would call hard edges — and
[`cut_along_edges`][triwarp.seams.cut_along_edges] *splits* the mesh along a given edge set,
duplicating vertices so that the two sides stop sharing them.

The cut is the operation that was missing. Marking edges is a threshold; separating the two sides of
a marked edge means rebuilding the vertex array, because the identity of a vertex is exactly what
has to change. Three things need it:

- **Hard normals.** A crease vertex shared by both sides averages a normal that belongs to neither.
- **UV seams.** A parametrization cannot be continuous across a closed surface, so a disk cut is a
  precondition rather than a nicety — [`lscm`][triwarp.parametrization.lscm] and
  [`harmonic`][triwarp.parametrization.harmonic] both require the caller to supply a mesh that
  already has a boundary. Going the other way,
  [`uv_seam_edges`][triwarp.seams.uv_seam_edges] *recovers* the cut an existing atlas already
  implies, so a textured mesh can be reopened along exactly the edges its texcoords tear at.
- **Part separation.** Cutting every crease of an assembly and then splitting components
  ([`split`][triwarp.combine.split]) recovers the pieces.

Nothing here moves a vertex: [`cut_along_edges`][triwarp.seams.cut_along_edges] changes only which
vertex indices the faces name, so the surface is geometrically identical and topologically opened.
"""

from __future__ import annotations

import math
from typing import Literal

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import seams as kernel_seams

# Whether the seam predicate compares coordinates rather than texcoord indices. A lookup rather than
# a chain of comparisons so an unrecognised mode cannot fall into a mode -- but the *membership*
# test at the call site is what turns it into a ``ValueError`` naming the argument and its options,
# which is the whole package's answer for an off-menu value; the bare ``KeyError('bogus')`` the
# lookup raises on its own names neither.
_UV_MATCH_MODES: dict[str, bool] = {"index": False, "uv": True}


def crease_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    angle: float = 30.0,
    *,
    include_boundary: bool = False,
) -> twt.Array2dInt32:
    """
    Edges where the surface bends by more than ``angle``.

    MeshLab's ``compute_selection_crease_per_edge``, returned as an explicit edge list rather than a
    selection so it feeds straight into [`cut_along_edges`][triwarp.seams.cut_along_edges].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    angle
        Dihedral threshold in **degrees**. An edge is a crease when its two faces meet at *strictly*
        more than this, so ``0`` selects every interior edge whose faces are not exactly coplanar
        (which excludes the diagonals of a flat quad, and is usually what a caller wanting "all of
        them" means) and ``180`` selects none. Must be in ``[0, 180]``.
    include_boundary
        When ``True``, boundary edges are included in the result. A boundary edge has no dihedral
        angle at all, so it is neither a crease nor not one — but it *is* already a seam, and a
        caller building a cut set usually wants it. Defaults to ``False``, which is the pure
        dihedral answer.

    Returns
    -------
    twt.Array2dInt32
        ``(k, 2)`` vertex-index pairs on ``faces.device``, one row per selected edge, smaller index
        first. The creases come first, in [`face_adjacency`][triwarp.adjacency.face_adjacency]'s
        row order, and the boundary edges, when asked for, after them in
        [`boundary_edges`][triwarp.boundary.boundary_edges]' order; neither block is sorted.

    Raises
    ------
    ValueError
        If ``angle`` is outside ``[0, 180]``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`cut_along_edges`][triwarp.seams.cut_along_edges]
    [`triwarp.adjacency.face_adjacency_angles`][triwarp.adjacency.face_adjacency_angles]
    [`triwarp.boundary.boundary_edges`][triwarp.boundary.boundary_edges]
    """
    require_same_device(vertices=vertices, faces=faces)
    if not 0.0 <= angle <= 180.0:
        raise ValueError(f"angle must be in [0, 180] degrees, got {angle}")

    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return twt.empty_2d((0, 2), wp.int32, device=device)

    # One radix sort of every halfedge's edge key, payload its halfedge index: an interior edge is
    # a run of exactly two keys -- ``face_adjacency``'s row, in its order -- and a boundary edge a
    # run of one, so both classes are flagged off the one sort, numbered by one scan of the flag
    # table and emitted by one launch, creases first. A key is below ``n_vertices ** 2``, so only
    # those bits are sorted.
    n_vertices = vertices.size
    n = 3 * n_faces
    keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
    order = wp.empty(2 * n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_adjacency.face_edge_keys_and_order,
        dim=n_faces,
        inputs=[faces, wp.uint64(n_vertices), keys, order],
        device=device,
    )
    wp.utils.radix_sort_pairs(
        keys, order, count=n, end_bit=min(64, max(1, (n_vertices * n_vertices - 1).bit_length()))
    )
    flags = twt.empty_2d((2 if include_boundary else 1, n), wp.int32, device=device)
    wp.launch(
        kernel_seams.crease_flags,
        dim=n,
        inputs=[vertices, faces, keys, order, n, wp.float32(math.radians(angle)), flags],
        device=device,
    )
    inclusive = flags.flatten()
    wp.utils.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # Sizes the output: the one host readback.
    n_edges = int(read_scalar(inclusive))
    edges = twt.empty_2d((n_edges, 2), wp.int32, device=device)
    if n_edges > 0:
        wp.launch(
            kernel_seams.emit_crease_edges,
            dim=inclusive.size,
            inputs=[inclusive, order, n, faces, edges],
            device=device,
        )
    return edges


def cut_along_edges(
    vertices: wp.array[wp.vec3] | wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    edges: twt.Array2dInt32,
    *,
    twins: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.vec3] | wp.array[wp.vec3d], wp.array[wp.int32]]:
    """
    Split the mesh along an edge set, duplicating vertices so the two sides no longer share them.

    Each *corner* (a face-vertex incidence) becomes a node, two corners at the same vertex are
    joined when the mesh edge between them is **not** in ``edges``, and each connected component of
    that graph becomes one output vertex. So a vertex whose whole fan is uncut survives as one
    vertex, a vertex crossed by a single marked edge on an open fan survives as one (the fan is
    still connected the long way round), and a vertex crossed by two survives as two. That is the
    correct local rule, and it is why the answer cannot be computed edge by edge.

    MeshLab's ``meshing_cut_along_crease_edges``. Geometrically a no-op — every output vertex sits
    exactly where its input did — and topologically the operation that turns a marked edge set into
    a boundary.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions, ``wp.vec3`` or ``wp.vec3d``. The output dtype
        follows.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer. Must be **edge-manifold**:
        the cut is defined through halfedge twins, and an edge with three faces has no well-defined
        "other side" (see [`halfedge_twins`][triwarp.halfedge.halfedge_twins]).
    edges
        ``(k, 2)`` vertex-index pairs to cut along, in either order per row. Rows that are not mesh
        edges are ignored, and boundary edges are already cuts so marking them changes nothing.
        Get a crease set from [`crease_edges`][triwarp.seams.crease_edges].
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins], length
        ``3 * n_faces``. Depends on the connectivity alone, so a caller running several passes over
        one topology builds it once -- and
        [`Trimesh.halfedge_twins`][triwarp.mesh.Trimesh.halfedge_twins] has it cached. Passing it
        also skips the edge-manifold check and the host readback that check costs.

    Returns
    -------
    vertices : wp.array[wp.vec3] | wp.array[wp.vec3d]
        Positions of the cut mesh, one per surviving corner component, in ``vertices``' own dtype.
        Longer than the input's wherever a vertex was split, and **shorter** when the input had
        unreferenced vertices — only corners produce output vertices, so an unused vertex
        disappears.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer over the new vertices. Same length, same winding
        and same face order as the input; only the indices change.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` does not have two columns, or, when ``twins`` is not given, ``faces`` is not
        edge-manifold or not consistently wound.
    RuntimeError
        If ``vertices``, ``faces``, ``edges`` and ``twins`` are not all on one device.

    See Also
    --------
    [`crease_edges`][triwarp.seams.crease_edges]
    [`triwarp.combine.split`][triwarp.combine.split]
    [`triwarp.repair.remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]

    Notes
    -----
    The inverse operation is
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]: welding coincident
    positions back together closes every cut this makes, which is a useful round-trip check and a
    warning — a cut mesh must not be passed through a position-based weld if the seams are meant to
    survive.
    """
    require_same_device(vertices=vertices, faces=faces, edges=edges, twins=twins)
    twt.ensure_edge_pairs(edges, "edges")

    device = faces.device
    n_halfedges = faces.size // 3 * 3
    if n_halfedges == 0:
        return wp.clone(vertices), wp.clone(faces)

    n_vertices = vertices.size
    if twins is None:
        twins = tw.halfedge.halfedge_twins(faces, n_vertices=n_vertices)

    # Marked edges as a sorted key set, so the kernel tests membership with a binary search rather
    # than a per-halfedge scan. Keys come from the same builder the kernel uses, which is what makes
    # a row given in either order match.
    marked_keys = tw.grouping.sorted_undirected_edge_keys(edges, n_vertices)

    # Components over the corners, by union-find straight over the halfedges: each halfedge names
    # one join of the corner graph (a self-loop where its edge is a boundary or marked), formed in
    # the thread, so no edge list is materialised or compacted. Every root is its component's
    # smallest corner.
    key_base = wp.uint64(n_vertices)
    union_inputs = [faces, twins, marked_keys, key_base]
    parents = tw.array.arange(n_halfedges, device=device)
    for kernel in (kernel_seams.corner_union_prehook, kernel_seams.corner_union_hook):
        wp.launch(kernel, dim=n_halfedges, inputs=[*union_inputs, parents], device=device)
    roots = wp.empty(n_halfedges, dtype=wp.int32, device=device)
    root_ranks = wp.empty(n_halfedges, dtype=wp.int32, device=device)
    wp.launch(
        kernel_seams.corner_roots,
        dim=n_halfedges,
        inputs=[parents, roots, root_ranks],
        device=device,
    )
    # Scanned in place, a root's flag becomes its rank plus one among the roots in ascending order,
    # which numbers the output vertices by their smallest corner; the total is the last entry.
    wp.utils.array_scan(root_ranks, out_array=root_ranks, inclusive=True)
    out_vertices = wp.empty(int(read_scalar(root_ranks)), dtype=vertices.dtype, device=device)
    # The union-find is done with ``parents``, so it takes the new face buffer: corner ``h`` of the
    # flat layout is entry ``h``.
    corner_index = parents
    wp.launch(
        kernel_seams.SCATTER_CORNER_VALUES[vertices.dtype],
        dim=n_halfedges,
        inputs=[faces, roots, root_ranks, vertices, corner_index, out_vertices],
        device=device,
    )
    return out_vertices, corner_index


def uv_seam_edges(
    faces: wp.array[wp.int32],
    texcoords: wp.array[wp.vec2],
    face_texcoords: wp.array[wp.int32] | None = None,
    *,
    match: Literal["index", "uv"] | None = None,
    tolerance: float = 0.0,
    n_vertices: int | None = None,
    twins: wp.array[wp.int32] | None = None,
) -> tuple[twt.Array2dInt32, twt.Array2dInt32, twt.Array2dInt32]:
    """
    Classify every mesh edge as a UV seam, a boundary or a UV-space foldover.

    ``igl::seam_edges``. Where [`crease_edges`][triwarp.seams.crease_edges] asks how the *surface*
    bends, this asks where the *atlas* tears: an edge is a seam when its two triangles disagree
    about where it lands in texture space, so the two sides cannot be sampled from one continuous
    patch. Positions are not read at all.

    Every result is ``(face, corner)`` provenance rather than a vertex pair, because the two sides
    of a seam name *different* texcoords at the same positions and both sides are usually wanted.
    Corner ``c`` of face ``f`` denotes the edge ``faces[3f + c] -> faces[3f + (c + 1) % 3]``;
    [`seam_edge_vertices`][triwarp.seams.seam_edge_vertices] converts any of the three blocks to the
    vertex pairs [`cut_along_edges`][triwarp.seams.cut_along_edges] takes.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer. Must be **edge-manifold**
        (see [`halfedge_twins`][triwarp.halfedge.halfedge_twins]).
    texcoords
        UV coordinates. With ``face_texcoords`` this is igl's ``TC``, an arbitrary-length pool the
        corners index into; without it, this is the per-corner (*wedge*) buffer of length
        ``3 * n_faces``, which is how MeshLab and the OBJ format store an atlas.
    face_texcoords
        igl's ``FTC``: a length-``3 * n_faces`` ``wp.int32`` buffer giving each corner's index into
        ``texcoords``. ``None`` means ``texcoords`` is already per-corner.
    match
        How two corners are judged to carry the same texcoord. ``"index"`` compares indices into
        ``texcoords`` (igl); ``"uv"`` compares the coordinates themselves (MeshLab). Defaults to
        ``"index"`` when ``face_texcoords`` is given and ``"uv"`` when it is not. The two differ on
        an atlas that stores the same coordinate twice: ``"index"`` calls that a seam and ``"uv"``
        does not.
    tolerance
        Distance below which two texcoords count as equal under ``match="uv"``. The ``0.0`` default
        is exact comparison, which is what MeshLab does. Ignored under ``match="index"``.
    n_vertices
        Total vertex count, used as the edge-pairing radix. When ``None`` it is inferred with
        [`array.index_bound`][triwarp.array.index_bound], which costs a host readback.
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins], length
        ``3 * n_faces``. Depends on the connectivity alone, so one table serves this call and every
        other halfedge walk over the same mesh --
        [`Trimesh.halfedge_twins`][triwarp.mesh.Trimesh.halfedge_twins] has it cached.

    Returns
    -------
    seams : twt.Array2dInt32
        ``(n_seams, 4)`` rows ``(forward_face, forward_corner, backward_face, backward_corner)``.
        The two halves name the same undirected edge from opposite sides; the *forward* one is the
        one running from the smaller vertex index to the larger.
    boundaries : twt.Array2dInt32
        ``(n_boundaries, 2)`` rows ``(face, corner)``, one per edge with a single incident triangle.
        A boundary is already a tear, so a caller reproducing MeshLab's notion of a seam wants these
        unioned with ``seams`` — which is what
        [`uv_seam_vertex_mask`][triwarp.seams.uv_seam_vertex_mask] does.
    foldovers : twt.Array2dInt32
        ``(n_foldovers, 4)``, same layout as ``seams``. Edges whose texcoords *do* match but whose
        two triangles land on the same side of the shared edge in UV space, so the map folds back
        over itself. Detected only where there is no seam.

    Raises
    ------
    ValueError
        If ``match="index"`` is asked for without ``face_texcoords``; if ``face_texcoords`` is
        present but not the same length as ``faces``, or has an entry outside
        ``[0, texcoords.shape[0])``; if ``face_texcoords`` is ``None`` and ``texcoords`` is not
        length ``3 * n_faces``; if ``faces`` is not edge-manifold or not consistently wound (when
        ``twins`` is not given); or if ``match`` is neither ``"index"`` nor ``"uv"``.
    RuntimeError
        If ``faces``, ``texcoords``, ``face_texcoords`` and ``twins`` are not all on one device.

    See Also
    --------
    [`seam_edge_vertices`][triwarp.seams.seam_edge_vertices]
    [`uv_seam_vertex_mask`][triwarp.seams.uv_seam_vertex_mask]
    [`crease_edges`][triwarp.seams.crease_edges]
    [`cut_along_edges`][triwarp.seams.cut_along_edges]
    [`triwarp.parametrization.face_flipped_indices`][triwarp.parametrization.face_flipped_indices]

    Notes
    -----
    Divergences from ``igl::seam_edges``, none of which change a row's contents on a consistently
    wound manifold mesh:

    - **Row order.** igl iterates an ``unordered_set`` and so has no defined order; rows here come
      out in ascending canonical-halfedge order, which is face-major and deterministic.
    - **Non-manifold edges** raise, because "the other side" of an edge with three triangles is not
      defined. igl's hash map silently keeps whichever half-edge it saw last.
    - **Inconsistent winding** raises, like a non-manifold edge, unless ``twins`` is supplied;
      igl's directed map loses one of two same-direction half-edges and reports the edge as a
      boundary. A supplied table that pairs same-direction half-edges is classified correctly.
    - **Precision.** The foldover orientation test runs in ``float32``; igl uses the input scalar
      type.
    - igl also takes ``V``, but reads only its row count — the ``n_vertices`` argument here.
    """
    require_same_device(
        faces=faces, texcoords=texcoords, face_texcoords=face_texcoords, twins=twins
    )
    match_uv = _validate_uv_inputs(faces, texcoords, face_texcoords, match)
    device = faces.device
    n_halfedges = faces.size // 3 * 3
    if n_halfedges == 0:
        return (
            twt.empty_2d((0, 4), wp.int32, device=device),
            twt.empty_2d((0, 2), wp.int32, device=device),
            twt.empty_2d((0, 4), wp.int32, device=device),
        )
    if twins is None:
        twins = tw.halfedge.halfedge_twins(faces, n_vertices=n_vertices)

    # Row ``k`` flags class ``k + 1`` (seam, boundary, foldover). One inclusive scan of the
    # flattened table, in place, numbers all three blocks, and its last column holds the running
    # totals that size them -- read back together in one copy.
    flags = twt.empty_2d((3, n_halfedges), wp.int32, device=device)
    wp.launch(
        kernel_seams.classify_uv_halfedges,
        dim=n_halfedges,
        inputs=[
            faces,
            twins,
            face_texcoords,
            face_texcoords is not None,
            texcoords,
            match_uv,
            wp.float32(tolerance * tolerance),
            flags,
        ],
        device=device,
    )
    inclusive = flags.flatten()
    wp.utils.array_scan(inclusive, out_array=inclusive, inclusive=True)
    totals = flags[:, n_halfedges - 1].numpy()
    n_seams = int(totals[0])
    n_boundaries = int(totals[1]) - n_seams
    n_foldovers = int(totals[2]) - int(totals[1])
    seams = twt.empty_2d((n_seams, 4), wp.int32, device=device)
    boundaries = twt.empty_2d((n_boundaries, 2), wp.int32, device=device)
    foldovers = twt.empty_2d((n_foldovers, 4), wp.int32, device=device)
    if int(totals[2]) > 0:
        wp.launch(
            kernel_seams.scatter_uv_halfedges,
            dim=n_halfedges,
            inputs=[faces, twins, inclusive, seams, boundaries, foldovers],
            device=device,
        )
    return seams, boundaries, foldovers


def seam_edge_vertices(
    faces: wp.array[wp.int32], face_corners: twt.Array2dInt32
) -> twt.Array2dInt32:
    """
    Vertex-index pairs of ``(face, corner)`` rows, the form the cut and the edge helpers take.

    Accepts any block [`uv_seam_edges`][triwarp.seams.uv_seam_edges] returns: only columns 0 and 1
    are read, which is the whole of a ``(n, 2)`` boundary row and the *forward* half of a
    ``(n, 4)`` seam or foldover row. The backward half traverses the same undirected edge, so it
    would produce the same pair reversed.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer, the same one the rows came
        from.
    face_corners
        ``(n, 2)`` or ``(n, 4)`` ``int32`` array of ``(face, corner, ...)`` rows.

    Returns
    -------
    twt.Array2dInt32
        ``(n, 2)`` vertex-index pairs on ``faces.device``, in the input's row order. Seam and
        foldover pairs come out **smaller index first**, since the forward half-edge is by
        definition the one running that way; boundary pairs keep their face's winding direction,
        matching [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges].

    Raises
    ------
    TypeError
        If ``face_corners`` is not a rank-2 ``int32`` array.
    ValueError
        If ``face_corners`` does not have at least two columns.
    RuntimeError
        If ``faces`` and ``face_corners`` are not all on one device.

    See Also
    --------
    [`uv_seam_edges`][triwarp.seams.uv_seam_edges]
    [`cut_along_edges`][triwarp.seams.cut_along_edges]
    """
    require_same_device(faces=faces, face_corners=face_corners)
    twt.ensure_ndim(face_corners, 2, dtype=wp.int32)
    if int(face_corners.shape[1]) < 2:
        raise ValueError(f"face_corners must have at least two columns, got {face_corners.shape}")

    device = faces.device
    n_rows = int(face_corners.shape[0])
    edges = twt.empty_2d((n_rows, 2), wp.int32, device=device)
    if n_rows > 0:
        wp.launch(
            kernel_seams.face_corner_edge_vertices,
            dim=n_rows,
            inputs=[faces, face_corners, edges],
            device=device,
        )
    return twt.as_array2d(edges, wp.int32)


def uv_seam_vertex_mask(
    faces: wp.array[wp.int32],
    texcoords: wp.array[wp.vec2],
    face_texcoords: wp.array[wp.int32] | None = None,
    *,
    include_boundary: bool = True,
    match: Literal["index", "uv"] | None = None,
    tolerance: float = 0.0,
    n_vertices: int | None = None,
) -> wp.array[wp.bool]:
    """
    Per-vertex mask marking the endpoints of every UV seam.

    MeshLab's ``compute_selection_by_texture_seams_per_vertex``, which folds boundaries into the
    seam set rather than reporting them apart — hence the ``include_boundary`` default. The
    per-edge answer, and foldovers, are in [`uv_seam_edges`][triwarp.seams.uv_seam_edges].

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    texcoords
        UV coordinates, per-corner or pooled; see
        [`uv_seam_edges`][triwarp.seams.uv_seam_edges].
    face_texcoords
        Optional length-``3 * n_faces`` corner-to-texcoord index buffer.
    include_boundary
        Whether an edge with a single incident triangle counts as a seam. ``True`` reproduces
        MeshLab; ``False`` restricts the mask to interior texcoord mismatches.
    match
        ``"index"`` or ``"uv"``; see [`uv_seam_edges`][triwarp.seams.uv_seam_edges].
    tolerance
        Coordinate tolerance under ``match="uv"``.
    n_vertices
        Total vertex count. When ``None`` it is inferred with
        [`array.index_bound`][triwarp.array.index_bound], which costs a host readback.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_vertices`` mask on ``faces.device``. Unreferenced vertices are always ``False``.

    Raises
    ------
    ValueError
        As for [`uv_seam_edges`][triwarp.seams.uv_seam_edges]: if ``match="index"`` is asked for
        without ``face_texcoords``, if ``face_texcoords`` is present but not the same length as
        ``faces`` or has an entry outside ``[0, texcoords.shape[0])``, if ``face_texcoords`` is
        ``None`` and ``texcoords`` is not length ``3 * n_faces``, if ``faces`` is not
        edge-manifold or not consistently wound, or if ``match`` is neither ``"index"`` nor
        ``"uv"``.
    RuntimeError
        If ``faces``, ``texcoords`` and ``face_texcoords`` are not all on one device.

    See Also
    --------
    [`uv_seam_edges`][triwarp.seams.uv_seam_edges]
    [`seam_edge_vertices`][triwarp.seams.seam_edge_vertices]
    [`triwarp.array.indices_to_mask`][triwarp.array.indices_to_mask]
    """
    require_same_device(faces=faces, texcoords=texcoords, face_texcoords=face_texcoords)
    match_uv = _validate_uv_inputs(faces, texcoords, face_texcoords, match)
    device = faces.device
    if n_vertices is None:
        n_vertices = tw.array.index_bound(faces)
    mask = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    n_halfedges = faces.size // 3 * 3
    if n_halfedges == 0 or n_vertices == 0:
        return mask
    twins = tw.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
    # The same per-halfedge classification ``uv_seam_edges`` compacts, marked straight onto the
    # vertices: a seam row's endpoints are its halfedge's own, so no row needs to exist.
    wp.launch(
        kernel_seams.mark_uv_seam_vertices,
        dim=n_halfedges,
        inputs=[
            faces,
            twins,
            face_texcoords,
            face_texcoords is not None,
            texcoords,
            match_uv,
            wp.float32(tolerance * tolerance),
            include_boundary,
            mask,
        ],
        device=device,
    )
    return mask


def _validate_uv_inputs(
    faces: wp.array[wp.int32],
    texcoords: wp.array[wp.vec2],
    face_texcoords: wp.array[wp.int32] | None,
    match: Literal["index", "uv"] | None,
) -> bool:
    """
    Check the texcoord arguments both UV entry points take, and resolve ``match`` to a bool.

    Returns whether the seam predicate compares coordinates (``True``) rather than texcoord
    indices. Shared by [`uv_seam_edges`][triwarp.seams.uv_seam_edges] and
    [`uv_seam_vertex_mask`][triwarp.seams.uv_seam_vertex_mask], whose ``Raises`` blocks document
    what this rejects.
    """
    n_faces = faces.size // 3
    if match is None:
        match = "index" if face_texcoords is not None else "uv"
    if match not in _UV_MATCH_MODES:
        raise ValueError(f"match must be one of {list(_UV_MATCH_MODES)}, got {match!r}")
    match_uv = _UV_MATCH_MODES[match]
    if not match_uv and face_texcoords is None:
        raise ValueError(
            "match='index' needs face_texcoords: without it every corner has its own texcoord "
            "index and every interior edge would be reported as a seam. Pass match='uv'."
        )
    if face_texcoords is not None and face_texcoords.size != faces.size:
        raise ValueError(
            f"face_texcoords must have one entry per face corner, got "
            f"{face_texcoords.size} for {faces.size} corners"
        )
    if face_texcoords is not None and face_texcoords.size > 0:
        # A caller-supplied pool index, unlike the length check above -- ``classify_uv_halfedge``
        # indexes ``texcoords`` with it directly, and an out-of-range entry (a stale ``FTC`` after
        # ``texcoords`` was trimmed, an off-by-one building the pool) is a device-side out-of-bounds
        # read rather than a Python exception. Caught here rather than left to the kernel.
        min_index, max_index = tw.reduce.minmax(face_texcoords)
        if min_index < 0 or max_index >= texcoords.size:
            raise ValueError(
                f"face_texcoords entries must be in [0, {texcoords.size}) (texcoords' "
                f"length), got a range of [{min_index}, {max_index}]"
            )
    if face_texcoords is None and texcoords.size != 3 * n_faces:
        raise ValueError(
            f"without face_texcoords, texcoords must be per-corner (length {3 * n_faces}), got "
            f"{texcoords.size}"
        )

    return match_uv
