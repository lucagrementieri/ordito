"""
Triangle edges: the directed corner edges, the deduplicated undirected set, and their lengths.

Three representations, and which one a caller wants depends on what an edge means to them:

- **directed corner edges** ([`faces_to_edges`][triwarp.edges.faces_to_edges]) — three per face, in
  corner order, so index ``3 * f + k`` is face ``f``'s edge ``k``. This is the halfedge indexing
  [`triwarp.halfedge`][triwarp.halfedge] builds on.
- **unique undirected edges** ([`edges_unique`][triwarp.edges.edges_unique]) — one per mesh edge,
  with [`edges_unique_inverse`][triwarp.edges.edges_unique_inverse] mapping each corner back to it.
- **per-face length tables** ([`face_edge_lengths`][triwarp.edges.face_edge_lengths]) — an
  ``(n_faces, 3)`` table in the *opposite-corner* column order the cotangent formulas want, which is
  the intrinsic description [`triwarp.laplacian`][triwarp.laplacian] consumes.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_same_device
from triwarp.array import arange_repeat
from triwarp.constants import INDEX_RADIX_PAIR
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import edges as kernel_edges


def faces_to_edges(
    faces: wp.array[wp.int32],
    sorted: bool = False,  # noqa: A002
) -> twt.Array2dInt32:
    """
    Directed triangle edges from a flat ``(i0, i1, i2)`` index buffer.

    For each face emits three directed edges ``(i0, i1)``, ``(i1, i2)``, ``(i2, i0)`` in
    row-major order, matching [`trimesh.geometry.faces_to_edges`][].

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer of consecutive vertex-index triples.
    sorted
        If ``True``, each output row has its smaller vertex index first (undirected edges).

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_faces * 3, 2)``. Empty ``(0, 2)`` array when ``n_faces == 0``.

    See Also
    --------
    [`trimesh.geometry.faces_to_edges`][]
    """
    n_faces = faces.size // 3
    edges = twt.empty_2d((n_faces * 3, 2), wp.int32, device=faces.device)
    wp.launch(
        kernel_edges.faces_to_edges,
        dim=n_faces,
        inputs=[faces, wp.bool(sorted), edges],
        device=faces.device,
    )
    return twt.as_array2d(edges, wp.int32)


def edges_face(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Face index for each directed edge produced by [`faces_to_edges`][triwarp.edges.faces_to_edges].

    Returns an array of length ``n_faces * 3`` where entry ``3*f + k`` equals ``f``,
    matching [`trimesh.Trimesh.edges_face`][].

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces * 3`` face indices on ``faces.device``.

    See Also
    --------
    [`trimesh.Trimesh.edges_face`][]
    """
    n_faces = faces.size // 3
    return arange_repeat(n_faces * 3, 3, faces.device)


def edges_unique(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
    *,
    validate: bool = True,
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]:
    """
    Return unique undirected edges and their inverse mapping into the sorted edge list.

    Equivalent to [`trimesh.Trimesh.edges_unique`][] and
    [`trimesh.Trimesh.edges_unique_inverse`][] computed together.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first).
        When ``None``, built from ``faces``.
    n_vertices
        Total number of vertices (used as the hash base, and as the radix the unique rows are
        unpacked with). When ``None`` and ``validate`` is ``True`` it is inferred from the edge
        indices with a device-host sync -- the same reduction that runs the range check.
        When ``None`` and ``validate`` is ``False`` the keys pack against
        [`constants.INDEX_RADIX_PAIR`][triwarp.constants.INDEX_RADIX_PAIR] instead, which bounds
        every ``int32`` index with no reduction at all and leaves the row order unchanged.
    validate
        Whether to range-check the edge indices before packing them. The check is a
        ``triwarp.reduce.minmax`` whose host readback serialises the device pipeline, and it is a
        large share of this call because everything else here is launch overhead. Pass ``False``
        only where both bounds are structurally guaranteed -- a face buffer this package produced
        itself, or one an entry point has already validated. A ``False`` that is wrong does not
        raise: indices at or above ``n_vertices`` collide in the packing and silently group two
        different edges as one, and a negative index is read as a huge unsigned digit.

    Returns
    -------
    unique_edges : twt.Array2dInt32
        Shape ``(m, 2)`` unique undirected vertex pairs, ``m <= n_faces * 3``.
    inverse : wp.array[wp.int32]
        Length ``n_faces * 3``. ``unique_edges[inverse[i]] == edges_sorted[i]``.

    Raises
    ------
    TypeError
        If ``edges_sorted`` is given and is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``edges_sorted`` is given and does not have exactly two columns, if ``n_vertices`` is
        not positive, or if ``validate`` is ``True`` and an edge index is negative or reaches
        ``n_vertices``.
    RuntimeError
        If ``faces`` and ``edges_sorted`` are not all on one device.

    See Also
    --------
    [`trimesh.Trimesh.edges_unique`][]
    [`trimesh.Trimesh.edges_unique_inverse`][]
    """
    require_same_device(faces=faces, edges_sorted=edges_sorted)
    if edges_sorted is not None:
        twt.ensure_edge_pairs(edges_sorted, "edges_sorted")
    n_faces = faces.size // 3
    device = faces.device

    if n_faces == 0:
        empty_edges = twt.empty_2d((0, 2), wp.int32, device=device)
        empty_inv = wp.empty(0, dtype=wp.int32, device=device)
        return empty_edges, empty_inv

    if edges_sorted is None:
        # The keys are built straight off ``faces``: ``face_edge_keys`` packs each corner's sorted
        # pair exactly as ``hash_indices_rows`` packs the corresponding ``faces_to_edges`` row, so
        # the ``(3 * n_faces, 2)`` table is never written or read back, and any range check reduces
        # the ``3 * n_faces`` face buffer -- the same values, half the entries.
        keys, radix = _face_edge_keys(faces, n_faces, n_vertices, validate)
        return _unique_edges_from_keys(keys, radix, device)

    if n_vertices is None:
        if validate:
            # Inferring the bound from this very buffer already reduces it, and the packing's own
            # check would reduce it again to re-test a bound derived from it -- only the negative
            # half could ever fire. One reduction answers both.
            n_vertices = tw.array.index_bound(edges_sorted, require_non_negative=True)
            validate = False
        else:
            # Nothing is being checked, so the only thing the count would be used for is the
            # radix -- and the pair radix bounds every ``int32`` without reducing the rows to find
            # out. Row order is unchanged; ``edges_from_keys`` below unpacks against whichever
            # radix packed them.
            n_vertices = INDEX_RADIX_PAIR

    keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=validate)
    return _unique_edges_from_keys(keys, n_vertices, device)


def _face_edge_keys(
    faces: wp.array[wp.int32], n_faces: int, n_vertices: int | None, validate: bool
) -> tuple[wp.array[wp.uint64], int]:
    """
    Packed sorted-pair key of every corner edge, plus the radix they were packed against.

    The ``edges_sorted is None`` half of [`edges_unique`][triwarp.edges.edges_unique], resolving
    ``n_vertices`` and ``validate`` exactly as the composed ``faces_to_edges`` +
    ``hash_indices_rows`` path does: an inferred bound checks the negative half, a supplied one is
    checked against, and ``validate=False`` with no bound packs against ``INDEX_RADIX_PAIR``. The
    reduction reads only the ``3 * n_faces`` indices a face actually owns, which is the set the
    edge rows would have held.
    """
    if n_vertices is not None and n_vertices <= 0:
        raise ValueError(f"n_vertices must be positive, got {n_vertices}")
    if validate:
        owned = faces if faces.size == 3 * n_faces else twt.as_dense(faces[: 3 * n_faces])
        low, high = tw.reduce.minmax(owned)
        if low < 0:
            raise ValueError(f"edge indices must be non-negative, got a minimum of {low}")
        if n_vertices is None:
            n_vertices = int(high) + 1
        elif high >= n_vertices:
            raise ValueError(
                f"edge indices must be less than n_vertices {n_vertices}, got a maximum of {high}"
            )
    elif n_vertices is None:
        n_vertices = INDEX_RADIX_PAIR
    keys = wp.empty(3 * n_faces, dtype=wp.uint64, device=faces.device)
    wp.launch(
        kernel_adjacency.face_edge_keys,
        dim=n_faces,
        inputs=[faces, wp.uint64(n_vertices), keys],
        device=faces.device,
    )
    return keys, n_vertices


def _unique_edges_from_keys(
    keys: wp.array[wp.uint64], n_vertices: int, device: wp.DeviceLike
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]:
    """Deduplicate packed edge keys into ``(unique_edges, inverse)``, unpacked by ``n_vertices``."""
    # A key is ``min + max * n_vertices``, below ``n_vertices ** 2``, which bounds the sort's bits.
    unique_keys, inverse = tw.grouping.unique_1d(
        keys, return_inverse=True, max_value=n_vertices * n_vertices - 1
    )

    # The deduplicated rows are recovered from the keys, not by gathering the corner that first
    # produced each one: the packing is exactly invertible for two columns, so a
    # ``first_occurrence_indices`` scatter, an ``array.gather`` and the first-occurrence buffer
    # between them all disappear. Row order is unchanged -- both forms index by the same unique id.
    unique_edges_out = twt.empty_2d((unique_keys.size, 2), wp.int32, device=device)
    wp.launch(
        kernel_edges.edges_from_keys,
        dim=unique_edges_out.shape[0],
        inputs=[unique_keys, wp.uint64(n_vertices), unique_edges_out],
        device=device,
    )
    return twt.as_array2d(unique_edges_out, wp.int32), inverse


def edges_unique_inverse(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> wp.array[wp.int32]:
    """
    Inverse mapping from sorted edges into [`edges_unique`][triwarp.edges.edges_unique].

    Maps [`faces_to_edges`][triwarp.edges.faces_to_edges] (sorted) into `edges_unique`:
    ``edges_unique(faces)[edges_unique_inverse(faces)] == faces_to_edges(faces, sorted=True)``

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed sorted edges. When ``None``, built from ``faces``.
    n_vertices
        Total vertex count. When ``None``, inferred from ``edges_sorted`` with a device-host sync.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces * 3`` inverse indices.

    Raises
    ------
    TypeError
        If ``edges_sorted`` is given and is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``edges_sorted`` is given and does not have exactly two columns.
    RuntimeError
        If ``faces`` and ``edges_sorted`` are not all on one device.

    See Also
    --------
    [`trimesh.Trimesh.edges_unique_inverse`][]
    """
    require_same_device(faces=faces, edges_sorted=edges_sorted)
    return edges_unique(faces, edges_sorted=edges_sorted, n_vertices=n_vertices)[1]


def edges_unique_length(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
    *,
    validate: bool = True,
) -> wp.array[wp.float32]:
    """
    Euclidean length of each unique undirected edge.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    unique_edges
        Optional precomputed unique edges ``(m, 2)``. When ``None``, computed from ``faces``.
    n_vertices
        Total vertex count passed to [`edges_unique`][triwarp.edges.edges_unique]. Ignored when
        ``unique_edges`` is already provided. When ``None`` it is taken from ``vertices`` rather
        than inferred from ``faces``, which would cost a host readback for a count this function
        was already handed.
    validate
        Forwarded to [`edges_unique`][triwarp.edges.edges_unique] when ``unique_edges`` is built
        here; ignored when ``unique_edges`` is supplied. Pass ``False`` only where the face
        indices' range is structurally guaranteed.

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` edge lengths on ``faces.device``.

    Raises
    ------
    TypeError
        If ``unique_edges`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``unique_edges`` does not have exactly two columns, or if ``validate`` is ``True``,
        ``unique_edges`` is omitted and a face index is negative or reaches ``n_vertices``.
    RuntimeError
        If ``vertices``, ``faces`` and ``unique_edges`` are not all on one device.

    See Also
    --------
    [`trimesh.Trimesh.edges_unique_length`][]
    """
    require_same_device(vertices=vertices, faces=faces, unique_edges=unique_edges)
    if unique_edges is None:
        # ``vertices.shape[0]`` when the caller gave no count: ``edges_unique`` would otherwise
        # infer it from ``faces`` with ``array.index_bound``, a device reduction plus a host
        # readback, for a number every caller of *this* function already holds in the array it
        # passed. It is also the bound ``validate`` is documented against.
        if n_vertices is None:
            n_vertices = vertices.size
        unique_edges, _ = edges_unique(faces, n_vertices=n_vertices, validate=validate)

    return _edge_lengths(vertices, unique_edges, "unique_edges")


def edges_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], edges_in: twt.Array2dInt32 | None = None
) -> wp.array[wp.float32]:
    """
    Euclidean length of every directed edge (one per face half-edge).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_in
        Optional precomputed directed edges ``(n_faces * 3, 2)``. When ``None``, computed
        from ``faces``.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_faces * 3`` edge lengths on ``faces.device``.

    Raises
    ------
    TypeError
        If ``edges_in`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``edges_in`` does not have exactly two columns.
    RuntimeError
        If ``vertices``, ``faces`` and ``edges_in`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces, edges_in=edges_in)
    if edges_in is not None:
        return _edge_lengths(vertices, edges_in, "edges_in")
    # Without a table the rows are the halfedges, so the lengths come straight off ``faces``.
    n_halfedges = faces.size // 3 * 3
    out = wp.empty(n_halfedges, dtype=wp.float32, device=vertices.device)
    if n_halfedges > 0:
        wp.launch(
            kernel_edges.halfedge_lengths,
            dim=n_halfedges,
            inputs=[vertices, faces, out],
            device=vertices.device,
        )
    return out


def _edge_lengths(
    vertices: wp.array[wp.vec3], edges: twt.Array2dInt32, name: str
) -> wp.array[wp.float32]:
    """
    Euclidean length of every row of an ``(m, 2)`` edge table.

    The shared body of [`edges_unique_length`][triwarp.edges.edges_unique_length] and
    [`edges_length`][triwarp.edges.edges_length], which differ only in which edge table they
    obtain first. Stays a kernel rather than a ``wp.map`` over gathered endpoints: the columns
    of ``edges`` are strided views, and Warp's Python-scope gather ignores a view's stride.

    ``name`` is the *caller's* parameter name, so the shape guard names the keyword the caller
    actually passed. The guard runs here rather than at the two entry points because the kernel
    reads columns 0 and 1 unconditionally: a wider table is accepted silently and its third column
    ignored, where a rank-1 buffer raises at launch -- so only the wide case needs catching, and it
    needs catching on both paths.
    """
    twt.ensure_edge_pairs(edges, name)
    m = int(edges.shape[0])
    device = vertices.device
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(kernel_edges.edge_lengths, dim=m, inputs=[vertices, edges, out], device=device)
    return out


def face_edge_lengths(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> twt.Array2dFloat32:
    """
    Per-face edge lengths, in the intrinsic column order the cotangent formulas expect.

    Column ``e`` holds the length of the edge *opposite* corner ``e``, matching
    [`cotmatrix_entries_intrinsic`][triwarp.laplacian.cotmatrix_entries_intrinsic] and
    ``igl::cotmatrix_entries``' intrinsic overload. Unlike
    [`edges_unique_length`][triwarp.edges.edges_unique_length] this is a per-*corner* table: an
    interior edge appears twice, which is what lets the entries be perturbed per face.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    twt.Array2dFloat32
        ``(n_faces, 3)`` edge lengths on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]
    [`edges_unique_length`][triwarp.edges.edges_unique_length]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    lengths = twt.empty_2d((n_faces, 3), wp.float32, device=device)
    if n_faces == 0:
        return twt.as_array2d(lengths, wp.float32)
    wp.launch(
        kernel_edges.face_edge_lengths,
        dim=n_faces,
        inputs=[vertices, faces, lengths],
        device=device,
    )
    return twt.as_array2d(lengths, wp.float32)


def mean_edge_length(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Mean length of the ``3 * n_faces`` per-face edges, counting a shared edge once per face.

    Every face contributes all three of its edges, so an edge shared by two faces is counted twice
    and a boundary edge once -- which is what separates this from
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length], the same average over each
    edge once. That weighting is deliberate, not an oversight: it is the average
    libigl's ``CurvatureCalculator::getAverageEdge`` computes, which is what
    ``igl::principal_curvature`` calls to set its sphere-search radius, and what
    [`principal_curvature`][triwarp.curvature.principal_curvature] therefore uses here.

    !!! note "There are two edge averages, and they are not interchangeable"
        [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] counts each edge once,
        matching ``igl::avg_edge_length`` and MeshLab's ``avg_edge_length``. The two agree exactly
        on a closed manifold mesh -- every edge has two incident faces there, so the doubling is
        uniform -- and diverge by a fraction of a percent on anything with a boundary or a
        non-manifold edge.

        Reach for this one when reproducing libigl's curvature; reach for the unique-edge one when
        reproducing anything else, including the heat method's timestep.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.

    Returns
    -------
    float
        Mean length over the ``3 * n_faces`` per-face edges. ``0.0`` when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length]
        The same average taken over the unique edges instead.
    [`edges_length`][triwarp.edges.edges_length]
        The per-face lengths this averages.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0
    return tw.reduce.mean(edges_length(vertices, faces))


def mean_unique_edge_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, validate: bool = True
) -> float:
    """
    Mean length of the **unique** undirected edges.

    Each edge is counted once, however many faces share it. This is the definition
    ``igl::avg_edge_length`` and MeshLab's ``avg_edge_length`` both use, and the one
    ``igl::heat_geodesics`` picks its diffusion timestep from -- so it is what
    [`heat_geodesic`][triwarp.heat.heat_geodesic] and
    [`vector_heat_operators`][triwarp.heat.vector_heat_operators] use here.

    See [`mean_edge_length`][triwarp.edges.mean_edge_length] for the per-face average, how far the
    two diverge on an open mesh, and why libigl carries both.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    validate
        Forwarded to [`edges_unique`][triwarp.edges.edges_unique]. Pass ``False`` only where the
        face indices' range is structurally guaranteed; it removes a host readback.

    Returns
    -------
    float
        Mean length over the unique undirected edges. ``0.0`` when ``n_faces == 0``.

    Raises
    ------
    ValueError
        If ``validate`` is ``True`` and a face index is negative or reaches ``n_vertices``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`mean_edge_length`][triwarp.edges.mean_edge_length]
        The same average taken over the per-face edges instead.
    [`edges_unique_length`][triwarp.edges.edges_unique_length]
        The per-edge lengths this averages.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0
    return tw.reduce.mean(
        edges_unique_length(vertices, faces, n_vertices=vertices.size, validate=validate)
    )
