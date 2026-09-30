"""
Mesh face-adjacency graph: which faces share an edge, and what each adjacent pair looks like.

[`face_adjacency`][triwarp.adjacency.face_adjacency] is the table every other function here reads:
one row per edge-adjacent face pair. The rest are per-adjacency-row quantities over that table, all
row-aligned with it, so a caller derives the pairs once and passes them in --
[`require_paired_adjacency`][triwarp.adjacency.require_paired_adjacency] states the one rule those
keywords carry, and is public because the functions taking the pair span three modules.

- [`face_adjacency_unshared`][triwarp.adjacency.face_adjacency_unshared] gives the two opposite
  corners of each pair, and [`face_adjacency_angles`][triwarp.adjacency.face_adjacency_angles] the
  dihedral between the two faces.
- [`face_adjacency_convex`][triwarp.adjacency.face_adjacency_convex] answers whether a pair is
  *convex* -- each face's third vertex on the inner side of the other's plane -- and
  [`face_adjacency_projections`][triwarp.adjacency.face_adjacency_projections] returns the signed
  distances it thresholds, for callers that want the margin rather than the verdict.

[`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels] is the one
whole-graph answer: face-level connected components over the same adjacency.
"""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.constants import INDEX_RADIX_PAIR, TOLERANCE_MERGE_CONSTANT
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import array as kernel_array
from triwarp.kernels import grouping as kernel_grouping
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels.algorithms import connected_components as kernel_connected_components


@overload
def face_adjacency(
    faces: wp.array[wp.int32],
    *,
    return_edges: Literal[False] = False,
    n_vertices: int | None = None,
    edges_paired: bool = False,
) -> twt.Array2dInt32: ...
@overload
def face_adjacency(
    faces: wp.array[wp.int32],
    *,
    return_edges: Literal[True],
    n_vertices: int | None = None,
    edges_paired: bool = False,
) -> tuple[twt.Array2dInt32, twt.Array2dInt32]: ...
def face_adjacency(
    faces: wp.array[wp.int32],
    *,
    return_edges: bool = False,
    n_vertices: int | None = None,
    edges_paired: bool = False,
) -> twt.Array2dInt32 | tuple[twt.Array2dInt32, twt.Array2dInt32]:
    """
    Face index pairs that share an undirected mesh edge.

    Each output row lists two face indices whose triangles share an edge (vertex
    pair). On a closed manifold mesh every interior edge appears exactly twice in
    the edge list, so only edges with duplicate sorted rows are kept—boundary edges
    that appear once are omitted.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer of triangle vertex indices, the
        same flat layout as [`triwarp.triangles`][triwarp.triangles] and
        [`faces_to_edges`][triwarp.edges.faces_to_edges].
    return_edges
        If ``True``, also return the shared vertex indices for each adjacency row.
    n_vertices
        Optional vertex count, used as the row-hashing radix. It does not change the answer:
        when ``None`` a fixed radix above every ``int32`` index packs the edges instead, in the
        same order, so no reduction or host readback is needed either way. When given it must be
        greater than every index in ``faces``; see the warning on
        [`hash_indices_rows`][triwarp.grouping.hash_indices_rows].
    edges_paired
        Promise that every undirected edge is shared by exactly two faces -- what
        [`is_edge_manifold`][triwarp.validation.is_edge_manifold] with
        ``allow_boundary_edges=False`` establishes. The adjacency then has exactly
        ``3 * n_faces / 2`` rows and they are consecutive in the sorted edge-key order, so the
        run detection, its scan and the host read of the pair count are skipped. The answer is
        byte-identical to the default path on such a mesh. The promise is **not checked** beyond
        the face count's parity: on a mesh with a boundary or a non-manifold edge the rows are
        still valid face indices but pair faces that need not share an edge.

    Returns
    -------
    twt.Array2dInt32 or tuple of two such arrays
        **adjacency** — shape ``(m, 2)`` on ``faces.device``. Row ``k`` gives face
        indices ``(f0, f1)`` with ``f0 <= f1`` (rows sorted in-place). Faces
        ``faces[3*f0:3*f0+3]`` and ``faces[3*f1:3*f1+3]`` share an edge.

        When ``return_edges`` is ``True``, also returns **adjacency_edges** —
        shape ``(m, 2)`` with the sorted vertex pair for that shared edge (one row
        per adjacency pair, taken from the first matching edge row).

    Raises
    ------
    ValueError
        If ``edges_paired`` is given for an odd face count, where no pairing exists.

    Notes
    -----
    Duplicate-edge grouping uses
    [`group_int_rows`][triwarp.grouping.group_int_rows] with
    ``length=2``, equivalent to [`trimesh.grouping.group_rows`][] with
    ``require_count=2``. An empty mesh yields shape ``(0, 2)``.

    See Also
    --------
    [`faces_to_edges`][triwarp.edges.faces_to_edges]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    [`trimesh.graph.face_adjacency`][]
    """
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        empty_array = twt.empty_2d((0, 2), wp.int32, device=device)
        if return_edges:
            return empty_array, twt.empty_2d((0, 2), wp.int32, device=device)
        return empty_array

    if edges_paired and n_faces % 2 != 0:
        raise ValueError(f"edges_paired needs an even face count, got {n_faces} faces")
    # Edge ``e`` belongs to face ``e // 3``, so the owning faces need no ``edges_face`` table, no
    # gather through it, and no row sort — one kernel does the division and orders the pair. With
    # ``return_edges`` the same kernel writes each pair's shared edge too, straight off ``faces``,
    # so no edge table is built just to be gathered from. Unpaired, that kernel also compacts the
    # pairs off the sorted keys, so no intermediate group table is written.
    if edges_paired:
        edge_groups = _paired_edge_groups(faces, n_vertices)
        n_pairs = int(edge_groups.shape[0])
        kernel, dim, sources = kernel_adjacency.edge_pairs_to_face_pairs, n_pairs, [edge_groups]
    else:
        order, inclusive, n_pairs = _sorted_pair_scan(faces, n_vertices)
        kernel, dim = kernel_adjacency.emit_sorted_face_pairs, inclusive.size
        sources = [inclusive, order]
    adjacency = twt.empty_2d((n_pairs, 2), wp.int32, device=device)
    adjacency_edges = twt.empty_2d((n_pairs, 2), wp.int32, device=device) if return_edges else None
    if n_pairs > 0:
        wp.launch(
            kernel,
            dim=dim,
            inputs=[faces, None, *sources, adjacency, adjacency_edges],
            device=device,
        )
    if adjacency_edges is None:
        return twt.as_array2d(adjacency, wp.int32)
    return twt.as_array2d(adjacency, wp.int32), twt.as_array2d(adjacency_edges, wp.int32)


def require_paired_adjacency(
    face_adjacency: twt.Array2dInt32 | None, face_adjacency_edges: twt.Array2dInt32 | None
) -> None:
    """
    Raise unless a precomputed face-adjacency pair is either wholly given or wholly omitted.

    The contract behind every ``face_adjacency=`` / ``face_adjacency_edges=`` keyword in the
    package: the two tables are row-aligned halves of one answer and are useless apart, so a
    function that accepts them accepts both or neither. Public because the functions that take the
    pair live in three modules -- here, [`triwarp.curvature`][triwarp.curvature] and
    [`triwarp.validation`][triwarp.validation] -- and they must all reject the same call with the
    same message; a caller that validates its own arguments before dispatching between them can
    use it for that too.

    **Call it before an empty-mesh guard, not after.** A half-supplied pair is a caller bug
    whatever the mesh is, and the four wrappers that take the pair once disagreed about this, so
    the same wrong call raised or returned an empty answer depending on the input. The *derivation*
    goes the other way round -- below the guard -- because on an empty mesh
    [`face_adjacency`][triwarp.adjacency.face_adjacency] would build two empty tables nothing reads.

    Parameters
    ----------
    face_adjacency
        Candidate ``(m, 2)`` face pairs, or ``None``.
    face_adjacency_edges
        Candidate ``(m, 2)`` shared-edge endpoints, or ``None``.

    Raises
    ------
    ValueError
        If exactly one of the two is given.
    RuntimeError
        If ``face_adjacency`` and ``face_adjacency_edges`` are not all on one device.

    See Also
    --------
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
        Produces the pair, with ``return_edges=True``.
    [`face_adjacency_unshared`][triwarp.adjacency.face_adjacency_unshared]
    [`face_adjacency_projections`][triwarp.adjacency.face_adjacency_projections]
    [`face_adjacency_convex`][triwarp.adjacency.face_adjacency_convex]
    """
    require_same_device(face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges)
    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )


def _paired_edge_groups(faces: wp.array[wp.int32], n_vertices: int | None) -> twt.Array2dInt32:
    """
    Pair the halfedges of a mesh whose every edge is shared by exactly two faces, as ``(m, 2)``.

    Every key then occurs exactly twice, so each run of the stable sort starts at an even
    position and the run detection of [`group`][triwarp.grouping.group] would emit sorted slots
    ``2k, 2k + 1`` as row ``k``: the sort's permutation, read two to a row, is that answer already.
    """
    _, order, n = _sorted_face_edge_buffers(faces, n_vertices)
    return twt.as_array2d(twt.as_dense(order[:n]).reshape((n // 2, 2)), wp.int32)


def _sorted_pair_scan(
    faces: wp.array[wp.int32], n_vertices: int | None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int]:
    """
    Sort the ``3 * n_faces`` undirected halfedge keys and locate the runs of exactly two.

    Returns ``(order, inclusive, m)``: the sorting permutation's double-width buffer (sorted
    position ``i`` is halfedge ``order[i]``, which belongs to face ``order[i] // 3``), the in-place
    inclusive scan of the positions that start a run of exactly two equal keys -- one edge shared
    by exactly two faces, i.e. one adjacency pair, whose row is the scan's exclusive offset at its
    first position -- and the pair count ``m``, read back because it sizes the answer. The scan
    lives in the payload buffer's upper half, which the sort leaves free.

    The keys hash the edge rows over the vertex-index range (the caller's ``n_vertices``, or
    ``INDEX_RADIX_PAIR``); using ``n_faces`` as the base is wrong whenever the largest vertex
    index is >= n_faces. The partition is invariant to any sufficiently large base, and the keys
    come straight off ``faces`` in one launch, so the ``(3 * n_faces, 2)`` edge rows are never
    written or read back; hashing those rows would give byte-identical keys, hence identical pair
    order, for more work.
    """
    keys, order, n = _sorted_face_edge_buffers(faces, n_vertices)
    inclusive = twt.as_dense(order[n:])
    wp.launch(
        kernel_grouping.MARK_GROUP_STARTS[keys.dtype],
        dim=n,
        inputs=[keys, n, 2, inclusive],
        device=faces.device,
    )
    wp.utils.array_scan(inclusive, inclusive, inclusive=True)
    # The pair count sizes the output, so it has to come back to the host.
    return order, inclusive, int(read_scalar(inclusive))


def vertex_face_adjacency(
    faces: wp.array[wp.int32], *, n_vertices: int | None = None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Incidence CSR of the faces touching each vertex, as ``(vertex_faces, offsets)``.

    Row ``v`` is ``vertex_faces[offsets[v] : offsets[v + 1]]`` and lists every face that references
    vertex ``v``, once per reference. ``offsets`` has length ``n_vertices + 1``, so its last entry
    is the total ``3 * n_faces`` and no caller needs a sentinel appended. Values first and offsets
    second is the package's packed-buffer convention -- see
    [`array.pack_1d_arrays`][triwarp.array.pack_1d_arrays], which states it.

    Built by counting sort rather than from halfedge twins, and that is a deliberate limitation
    rather than an omission: **each row is a set, not a rotation**. Ordering a row would require the
    halfedge structure, which does not exist at a vertex-non-manifold vertex — and the decimator and
    normal-flip guards that consume this must run on exactly such meshes. Use
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] when the rotational order is what is
    wanted; it requires an edge-manifold mesh in exchange.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    n_vertices
        Number of vertices, i.e. the number of CSR rows. When ``None`` it is inferred from
        ``faces`` with [`array.index_bound`][triwarp.array.index_bound], which costs one host
        readback; pass it when the caller already knows it. Rows for vertices no face references
        come out empty.

    Returns
    -------
    vertex_faces : wp.array[wp.int32]
        Length ``3 * n_faces`` face indices, grouped by vertex, arbitrary order within a row.
    offsets : wp.array[wp.int32]
        Length ``n_vertices + 1`` row offsets on ``faces.device``.

    See Also
    --------
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings]
    ``igl.vertex_triangle_adjacency``
    """
    device = faces.device
    n_faces = faces.size // 3
    row_count = tw.array.index_bound(faces) if n_vertices is None else int(n_vertices)

    offsets = wp.zeros(row_count + 1, dtype=wp.int32, device=device)
    if n_faces == 0 or row_count == 0:
        # ``row_count == 0`` with faces present (reachable only via an explicit ``n_vertices=0``)
        # would otherwise hand back an unwritten ``3 * n_faces`` buffer of allocator garbage.
        return wp.zeros(3 * n_faces, dtype=wp.int32, device=device), offsets

    vertex_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)

    counts = wp.zeros(row_count, dtype=wp.int32, device=device)
    # A flat face buffer *is* the corner -> vertex map, so one launch over all 3 * n_faces corners
    # gives each vertex its incident-face count.
    wp.launch(
        kernel_scatter.count_occurrences, dim=3 * n_faces, inputs=[faces, counts], device=device
    )
    # Deliberately NOT tw.array.counts_to_offsets: that helper always reads the total back, and
    # this function never needs it (it is 3 * n_faces, known on the host). Converting for
    # symmetry would add a device synchronization where there is currently none.
    wp.utils.array_scan(counts, out_array=offsets[1:], inclusive=True)
    # The counts are spent once scanned, so their buffer becomes the scatter's per-row cursor:
    # re-zeroing it in stream order is a memset where a second zeroed buffer was an allocation too.
    cursor = counts
    cursor.zero_()
    wp.launch(
        kernel_adjacency.scatter_vertex_faces,
        dim=n_faces,
        inputs=[faces, offsets, cursor, vertex_faces],
        device=device,
    )
    return vertex_faces, offsets


def face_adjacency_unshared(
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    *,
    n_vertices: int | None = None,
) -> twt.Array2dInt32:
    """
    Vertex on each adjacent face that is not on their shared edge.

    For each row of ``face_adjacency``, column 0 is the unshared vertex index on
    the first face and column 1 on the second face. When a face does not have
    exactly one vertex off the shared edge (degenerate case), that entry is ``-1``.

    The answer is defined by the **recorded shared edge** of each adjacency row, matching
    [`trimesh.graph.face_adjacency_unshared`][] exactly. This matters only for duplicate faces:
    two coincident triangles meet along all three of their edges, so they produce three adjacency
    rows and each one reports the corner off *its own* edge (``[[2, 2], [1, 1], [0, 0]]`` for two
    copies of ``(0, 1, 2)``). A cheaper "vertex of one face absent from the other" rule would
    return ``-1`` for all three, and would not be trimesh.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer of triangle vertex indices, the
        same flat layout as [`face_adjacency`][triwarp.adjacency.face_adjacency].
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When
        ``None``, adjacency and shared edges are derived from ``faces`` directly and neither table
        is materialized.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        [`face_adjacency`][triwarp.adjacency.face_adjacency] with ``return_edges=True``).
        Must be supplied
        together with ``face_adjacency`` or omitted with it.
    n_vertices
        Optional vertex count used as the row-hashing radix, forwarded to
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. Ignored when the adjacency tables are
        supplied. It does not change the answer; see that function's note.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(m, 2)`` on ``faces.device``. Row ``k`` gives vertex indices into
        ``faces`` for the corners not on ``face_adjacency_edges[k]``, or ``-1``
        when degenerate. Rows are in the same order
        [`face_adjacency`][triwarp.adjacency.face_adjacency] returns for the same ``faces``, so the
        two line up row-for-row whether or not the tables were passed in.

    Raises
    ------
    ValueError
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided,
        or if their row counts differ.
    RuntimeError
        If ``faces``, ``face_adjacency`` and ``face_adjacency_edges`` are not all on one device.

    See Also
    --------
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    [`trimesh.graph.face_adjacency_unshared`][]
    """
    require_same_device(
        faces=faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
    )
    # The derive branch below deliberately does *not* call face_adjacency, recovering both
    # owning faces and the shared edge from the sorted halfedge pairs instead. Only the pairing
    # rule is shared with the other wrappers that take this pair.
    require_paired_adjacency(face_adjacency, face_adjacency_edges)
    device = faces.device
    # Both branches end in one launch over ``m`` adjacency rows writing one ``(m, 2)`` buffer, so
    # the two differ only in the kernel and the table it reads. Resolving that first, and letting a
    # single ``m == 0`` return cover an empty mesh, an empty adjacency and an empty supplied table
    # alike, is what keeps the empty case to *one* allocation rather than building an empty
    # ``edge_groups`` only to size an empty output off it.
    if face_adjacency is None:
        m, dim, tables = 0, 0, ()
        if faces.size >= 3:
            order, inclusive, m = _sorted_pair_scan(faces, n_vertices)
            dim, tables = inclusive.size, (inclusive, order)
        kernel = kernel_adjacency.emit_sorted_unshared
    else:
        assert face_adjacency_edges is not None
        if face_adjacency.shape[0] != face_adjacency_edges.shape[0]:
            raise ValueError(
                "face_adjacency and face_adjacency_edges row counts must match, "
                f"got {face_adjacency.shape[0]} and {face_adjacency_edges.shape[0]}"
            )
        m = dim = int(face_adjacency.shape[0])
        kernel, tables = (
            kernel_adjacency.face_adjacency_unshared,
            (face_adjacency, face_adjacency_edges),
        )

    unshared = twt.empty_2d((m, 2), wp.int32, device=device)
    if m > 0:
        wp.launch(kernel, dim=dim, inputs=[faces, *tables, unshared], device=device)
    return twt.as_array2d(unshared, wp.int32)


def face_adjacency_angles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.float32]:
    """
    Unsigned angle in radians between each pair of adjacent faces.

    For each row of ``face_adjacency``, the angle is computed from the two
    corresponding face normals (unit vectors). Pair it with
    [`face_adjacency_convex`][triwarp.adjacency.face_adjacency_convex] for the sign: that function
    reports which side of each shared edge the pair folds towards, which is exactly the sign this
    unsigned magnitude is missing.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When
        ``None``, adjacency is computed from ``faces``.
    face_normals
        Optional length-``n_faces`` unit face normals. When ``None``, normals
        are computed from ``vertices`` and ``faces`` via
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas].

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` unsigned angles in radians on ``faces.device``, one per
        ``face_adjacency`` row. Empty when there are no faces or no adjacency pairs.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``face_adjacency`` and ``face_normals`` are not all on one
        device.

    See Also
    --------
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    [`face_adjacency_convex`][triwarp.adjacency.face_adjacency_convex]
        The sign this magnitude omits: convex or concave, per adjacency row.
    [`vector_angle`][triwarp.points.vector_angle]
    [`trimesh.Trimesh.face_adjacency_angles`][]
    """
    require_same_device(
        vertices=vertices, faces=faces, face_adjacency=face_adjacency, face_normals=face_normals
    )
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    if face_adjacency is None:
        face_adjacency = tw.adjacency.face_adjacency(faces, n_vertices=vertices.size)
    if face_normals is None:
        face_normals, _ = tw.triangles.face_normals_and_areas(vertices, faces)

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_angles = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_adjacency.face_adjacency_angles,
        dim=m,
        inputs=[face_normals, face_adjacency, out_angles],
        device=device,
    )
    return out_angles


def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    face_adjacency_unshared: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.float32]:
    """
    Project each adjacent face pair's non-shared vertex onto the first face plane.

    For each row of ``face_adjacency``, the dot product is taken between the
    normal of face ``face_adjacency[k, 0]`` and the vector from one endpoint of
    the shared edge to the unshared vertex on ``face_adjacency[k, 1]``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        [`face_adjacency`][triwarp.adjacency.face_adjacency] with ``return_edges=True``).
        Must be supplied together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair from
        [`face_adjacency_unshared`][triwarp.adjacency.face_adjacency_unshared]. When ``None``,
        computed from ``faces`` and the adjacency data.
    face_normals
        Optional length-``n_faces`` unit face normals. When ``None``, normals
        are computed from ``vertices`` and ``faces`` via
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas].

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` projections on ``faces.device``, one per ``face_adjacency``
        row. Empty when there are no faces or no adjacency pairs. A row whose second face is
        degenerate (its
        [`face_adjacency_unshared`][triwarp.adjacency.face_adjacency_unshared] entry is ``-1``)
        reads as ``+inf``, so it never registers as convex.

    Raises
    ------
    ValueError
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided, or if a
        supplied ``face_adjacency_unshared`` has a different row count from ``face_adjacency``.
    RuntimeError
        If ``vertices``, ``faces``, ``face_adjacency``, ``face_adjacency_edges``,
        ``face_adjacency_unshared`` and ``face_normals`` are not all on one device.

    See Also
    --------
    [`face_adjacency_convex`][triwarp.adjacency.face_adjacency_convex]
    [`trimesh.Trimesh.face_adjacency_projections`][]
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        face_adjacency=face_adjacency,
        face_adjacency_edges=face_adjacency_edges,
        face_adjacency_unshared=face_adjacency_unshared,
        face_normals=face_normals,
    )
    tables = _projection_tables(
        vertices, faces, face_adjacency, face_adjacency_edges, face_adjacency_unshared, face_normals
    )
    if tables is None:
        return wp.empty(0, dtype=wp.float32, device=faces.device)
    out_projections = wp.empty(int(tables[1].shape[0]), dtype=wp.float32, device=faces.device)
    wp.launch(
        kernel_adjacency.face_adjacency_projections,
        dim=out_projections.size,
        inputs=[vertices, *tables, out_projections],
        device=faces.device,
    )
    return out_projections


def face_adjacency_convex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    face_adjacency_unshared: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.bool]:
    """
    Return face pairs that are adjacent and locally convex.

    A pair is locally convex when the unshared vertex of the second face,
    projected onto the plane of the first face, has a projection less than
    [`TOLERANCE_MERGE`][triwarp.constants.TOLERANCE_MERGE].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs. Must be supplied
        together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair.
    face_normals
        Optional length-``n_faces`` unit face normals.

    Returns
    -------
    wp.array[wp.bool]
        Length ``m`` boolean mask on ``faces.device``, one per
        ``face_adjacency`` row. Empty when there are no faces or no adjacency
        pairs.

    Raises
    ------
    ValueError
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided, or if a
        supplied ``face_adjacency_unshared`` has a different row count from ``face_adjacency``.
    RuntimeError
        If ``vertices``, ``faces``, ``face_adjacency``, ``face_adjacency_edges``,
        ``face_adjacency_unshared`` and ``face_normals`` are not all on one device.

    See Also
    --------
    [`face_adjacency_projections`][triwarp.adjacency.face_adjacency_projections]
    [`trimesh.Trimesh.face_adjacency_convex`][]
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        face_adjacency=face_adjacency,
        face_adjacency_edges=face_adjacency_edges,
        face_adjacency_unshared=face_adjacency_unshared,
        face_normals=face_normals,
    )
    tables = _projection_tables(
        vertices, faces, face_adjacency, face_adjacency_edges, face_adjacency_unshared, face_normals
    )
    if tables is None:
        return wp.empty(0, dtype=wp.bool, device=faces.device)
    # The projection and its threshold in one launch rather than the projections array plus a
    # ``wp.map`` comparison over it; the shared ``adjacency_projection`` keeps the two answers
    # row-for-row consistent.
    out_convex = wp.empty(int(tables[1].shape[0]), dtype=wp.bool, device=faces.device)
    wp.launch(
        kernel_adjacency.face_adjacency_convex,
        dim=out_convex.size,
        inputs=[vertices, *tables, TOLERANCE_MERGE_CONSTANT, out_convex],
        device=faces.device,
    )
    return out_convex


def _projection_tables(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None,
    face_adjacency_edges: twt.Array2dInt32 | None,
    face_adjacency_unshared: twt.Array2dInt32 | None,
    face_normals: wp.array[wp.vec3] | None,
) -> tuple[wp.array[wp.vec3], twt.Array2dInt32, twt.Array2dInt32, twt.Array2dInt32] | None:
    """
    Resolve the four per-row tables a projection reads, or ``None`` when there is no row.

    The shared front of [`face_adjacency_projections`][triwarp.adjacency.face_adjacency_projections]
    and [`face_adjacency_convex`][triwarp.adjacency.face_adjacency_convex], which launch different
    kernels over the same ``(face_normals, face_adjacency, face_adjacency_edges,
    face_adjacency_unshared)`` -- returned in that order, the kernels' own.

    The pairing check runs *before* the empty-mesh guard, so a caller who passed only one half of
    the pair is told about it whatever the mesh is, and the four wrappers that take this pair agree
    about that. The *resolve* stays below the guard, because on an empty mesh it would allocate two
    empty tables nothing reads.
    """
    require_paired_adjacency(face_adjacency, face_adjacency_edges)
    if faces.size // 3 == 0:
        return None
    if face_adjacency is None:
        # ``tw.adjacency.`` rather than a bare call: the parameter shadows the module-level
        # ``face_adjacency`` it derives from.
        face_adjacency, face_adjacency_edges = tw.adjacency.face_adjacency(
            faces, return_edges=True, n_vertices=vertices.size
        )
    assert face_adjacency_edges is not None
    m = int(face_adjacency.shape[0])
    if m == 0:
        return None

    if face_adjacency_unshared is None:
        face_adjacency_unshared = tw.adjacency.face_adjacency_unshared(
            faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
        )
    elif int(face_adjacency_unshared.shape[0]) != m:
        # A caller-supplied table is otherwise trusted as-is; the kernel indexes it at every row up
        # to ``m``, so a shorter table is an out-of-bounds read rather than a wrong answer.
        raise ValueError(
            "face_adjacency_unshared row count must match face_adjacency, got "
            f"{face_adjacency_unshared.shape[0]} and {m}."
        )
    if face_normals is None:
        face_normals, _ = tw.triangles.face_normals_and_areas(vertices, faces)
    return face_normals, face_adjacency, face_adjacency_edges, face_adjacency_unshared


def face_connected_component_labels(
    faces: wp.array[wp.int32], *, n_vertices: int | None = None
) -> wp.array[wp.int32]:
    """
    Connected-component label per face (face-adjacency graph).

    The labels are those
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    gives over [`face_adjacency`][triwarp.adjacency.face_adjacency] with
    ``node_count = n_faces`` -- each component named by its smallest face id -- and run through
    the same union-find, fed straight from the sorted edge keys instead of a compacted pair table.
    [`Trimesh.face_connected_component_labels`][triwarp.mesh.Trimesh] calls the edge-list form
    directly when it already holds a cached adjacency.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    n_vertices
        Optional exclusive bound on the vertex indices, forwarded to
        [`sorted_face_edge_keys`][triwarp.adjacency.sorted_face_edge_keys] so the key sort orders
        only the bits a key can occupy. It does not change the answer, and it is trusted, not
        checked.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces`` on ``faces.device``.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=faces.device)
    # The adjacency pairs straight off the sorted edge keys, one union-find edge per halfedge (a
    # self-loop where no pair starts), formed inside the pre-hook and hook kernels: no compacted
    # table, no host read of its length, and no per-halfedge edge table written only to be read
    # back twice. The labels are each component's smallest face id whatever edges are hooked.
    # The forest lives in the payload buffer's upper half, which the sort leaves free.
    device = faces.device
    keys, order, n = _sorted_face_edge_buffers(faces, n_vertices)
    # Trimmed: the run classification reads its length off the keys' shape.
    sorted_keys = twt.as_dense(keys[:n])
    parents = twt.as_dense(order[n : n + n_faces])
    wp.launch(kernel_array.ARANGE[wp.int32], dim=n_faces, inputs=[parents], device=device)
    for kernel in (kernel_adjacency.sorted_pair_prehook, kernel_adjacency.sorted_pair_hook):
        wp.launch(kernel, dim=n, inputs=[sorted_keys, order, parents], device=device)
    labels = wp.empty(n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_connected_components.ecl_flatten,
        dim=n_faces,
        inputs=[parents, labels],
        device=device,
    )
    return labels


def sorted_face_edge_keys(
    faces: wp.array[wp.int32], *, n_vertices: int | None = None
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Every halfedge's undirected edge key, sorted, and the permutation that sorted them.

    Halfedge ``3f + k`` is corner ``k`` of face ``f``, the
    [`faces_to_edges`][triwarp.edges.faces_to_edges] row order, and its key packs the sorted
    endpoint pair ``(lo, hi)`` as ``lo + hi * radix`` -- the key
    [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] gives those rows. The sort is stable,
    so equal keys keep halfedge order: an edge's halfedges are one run, the pairs
    [`face_adjacency`][triwarp.adjacency.face_adjacency] groups are its runs of exactly two, and
    ``order[i] // 3`` is the face owning sorted position ``i``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    n_vertices
        Optional exclusive bound on the vertex indices, used as the packing radix. It lets the sort
        order only the low bits the keys can occupy, which is cheaper; without it the keys pack
        against [`constants.INDEX_RADIX_PAIR`][triwarp.constants.INDEX_RADIX_PAIR], which orders
        them identically and needs no bound. It is trusted, not checked.

    Returns
    -------
    sorted_keys : wp.array[wp.uint64]
        Length ``3 * n_faces`` ascending keys.
    order : wp.array[wp.int32]
        Length ``3 * n_faces`` halfedge index of each sorted key.

    Raises
    ------
    ValueError
        If ``n_vertices`` is not positive.
    """
    keys, order, n = _sorted_face_edge_buffers(faces, n_vertices)
    return twt.as_dense(keys[:n]), twt.as_dense(order[:n])


def _sorted_face_edge_buffers(
    faces: wp.array[wp.int32], n_vertices: int | None
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32], int]:
    """
    Untrimmed sorted edge keys and permutation, as ``(keys, order, n)``.

    [`sorted_face_edge_keys`][triwarp.adjacency.sorted_face_edge_keys]' sort, returning its two
    ``2 * n`` buffers, sorted in their leading halves. The upper halves are free scratch once
    the sort has run, which the internal callers borrow for a scan or a union-find forest instead
    of allocating one.
    """
    device = faces.device
    n = faces.size // 3 * 3
    radix = _hash_radix(n_vertices)
    # The keys and the identity payload are written straight into the leading halves of the radix
    # sort's double-width buffers in one launch, so no staging copy of every halfedge key is made
    # and the scratch halves are never filled; a known radix bounds the sorted bits.
    keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
    order = wp.empty(2 * n, dtype=wp.int32, device=device)
    if n == 0:
        return keys, order, 0
    wp.launch(
        kernel_adjacency.face_edge_keys_and_order,
        dim=n // 3,
        inputs=[faces, wp.uint64(radix), keys, order],
        device=device,
    )
    wp.utils.radix_sort_pairs(
        keys, order, count=n, end_bit=min(64, max(1, (radix * radix - 1).bit_length()))
    )
    return keys, order, n


def _hash_radix(n_vertices: int | None) -> int:
    """
    Resolve the row-hash base for the edge keys: ``n_vertices``, or ``INDEX_RADIX_PAIR``.

    Any base above every index packs a sorted pair injectively and in lexicographic order, so the
    radix sort orders the keys -- and ``group`` emits the pairs -- identically whichever base is
    used. ``INDEX_RADIX_PAIR`` exceeds every ``int32`` reinterpreted as ``uint32``, which is what
    ``pack_edge_key`` packs, so no reduction has to find the bound and no host readback serialises
    the call.
    """
    if n_vertices is None:
        return INDEX_RADIX_PAIR
    if n_vertices <= 0:
        raise ValueError(f"n_vertices must be positive, got {n_vertices}")
    return n_vertices
