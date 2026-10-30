"""
Triangle edges: the directed corner edges, the deduplicated undirected set, and their lengths.

Three representations, and which one a caller wants depends on what an edge means to them:

- **directed corner edges** ([`faces_to_edges`][ordito.edges.faces_to_edges]) — three per face, in
  corner order, so index ``3 * f + k`` is face ``f``'s edge ``k``. This is the halfedge indexing
  [`ordito.halfedge`][ordito.halfedge] builds on.
- **unique undirected edges** ([`edges_unique`][ordito.edges.edges_unique]) — one per mesh edge,
  with [`edges_unique_inverse`][ordito.edges.edges_unique_inverse] mapping each corner back to it.
- **per-face length tables** ([`face_edge_lengths`][ordito.edges.face_edge_lengths]) — an
  ``(n_faces, 3)`` table in the *opposite-corner* column order the cotangent formulas want, which is
  the intrinsic description [`ordito.laplacian`][ordito.laplacian] consumes.
"""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.array import arange_repeat
from ordito.constants import INDEX_RADIX_PAIR, TILE_1D
from ordito.kernels import adjacency as kernel_adjacency
from ordito.kernels import edges as kernel_edges
from ordito.kernels import grouping as kernel_grouping
from ordito.kernels import reduce as kernel_reduce


def faces_to_edges(
    faces: wp.array[wp.int32],
    sorted: bool = False,  # noqa: A002
) -> odt.Array2dInt32:
    """
    Directed triangle edges from a flat ``(i0, i1, i2)`` index buffer.

    For each face emits three directed edges ``(i0, i1)``, ``(i1, i2)``, ``(i2, i0)`` in
    row-major order, matching [`trimesh.geometry.faces_to_edges`][].

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` buffer of consecutive vertex-index triples.
    sorted
        If ``True``, each output row has its smaller vertex index first (undirected edges).

    Returns
    -------
    odt.Array2dInt32
        ``(3 * n_faces, 2)`` directed edges. Empty ``(0, 2)`` array when ``n_faces == 0``.

    See Also
    --------
    [`trimesh.geometry.faces_to_edges`][]

    Examples
    --------
    Three directed edges per face, in corner order; ``sorted=True`` puts the smaller index first:

    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([0, 1, 2, 2, 1, 3], dtype=wp.int32)
    directed = od.edges.faces_to_edges(faces)
    print(directed.numpy().tolist())  # [[0, 1], [1, 2], [2, 0], [2, 1], [1, 3], [3, 2]]
    undirected = od.edges.faces_to_edges(faces, sorted=True)
    print(undirected.numpy().tolist())  # [[0, 1], [1, 2], [0, 2], [1, 2], [1, 3], [2, 3]]
    ```
    """
    n_faces = faces.size // 3
    edges = odt.empty_2d((n_faces * 3, 2), wp.int32, device=faces.device)
    _launch.launch(
        kernel_edges.faces_to_edges,
        dim=n_faces,
        inputs=[faces, wp.bool(sorted), edges],
        device=faces.device,
    )
    return odt.as_array2d(edges, wp.int32)


def edges_face(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Face index for each directed edge produced by [`faces_to_edges`][ordito.edges.faces_to_edges].

    Returns an array of length ``n_faces * 3`` where entry ``3*f + k`` equals ``f``,
    matching [`trimesh.Trimesh.edges_face`][].

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` face indices on ``faces.device``.

    See Also
    --------
    [`trimesh.Trimesh.edges_face`][]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([0, 1, 2, 2, 1, 3], dtype=wp.int32)
    print(od.edges.edges_face(faces).numpy().tolist())  # [0, 0, 0, 1, 1, 1]
    ```
    """
    n_faces = faces.size // 3
    return arange_repeat(n_faces * 3, 3, faces.device)


@overload
def edges_unique(
    faces: wp.array[wp.int32],
    edges_sorted: odt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
    *,
    validate: bool = True,
    return_counts: Literal[False] = False,
) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def edges_unique(
    faces: wp.array[wp.int32],
    edges_sorted: odt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
    *,
    validate: bool = True,
    return_counts: Literal[True],
) -> tuple[odt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32]]: ...
def edges_unique(
    faces: wp.array[wp.int32],
    edges_sorted: odt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
    *,
    validate: bool = True,
    return_counts: bool = False,
) -> (
    tuple[odt.Array2dInt32, wp.array[wp.int32]]
    | tuple[odt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Return unique undirected edges and their inverse mapping into the sorted edge list.

    Equivalent to [`trimesh.Trimesh.edges_unique`][] and
    [`trimesh.Trimesh.edges_unique_inverse`][] computed together.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` face index buffer.
    edges_sorted
        ``(3 * n_faces, 2)`` precomputed sorted edges (each row min-first), or ``None`` to build
        them from ``faces``.
    n_vertices
        Total number of vertices (used as the hash base, and as the radix the unique rows are
        unpacked with). When ``None`` and ``validate`` is ``True`` it is inferred from the edge
        indices with a device-host sync -- the same reduction that runs the range check.
        When ``None`` and ``validate`` is ``False`` the keys pack against
        [`constants.INDEX_RADIX_PAIR`][ordito.constants.INDEX_RADIX_PAIR] instead, which bounds
        every ``int32`` index with no reduction at all and leaves the row order unchanged.
    validate
        Whether to range-check the edge indices. With ``n_vertices`` given (and no
        ``edges_sorted``) the check is folded into the key pass and read back with the edge count,
        so it costs no extra synchronization; otherwise it is a ``ordito.reduce.minmax`` whose host
        readback serialises the device pipeline. Pass ``False``
        only where both bounds are structurally guaranteed -- a face buffer this package produced
        itself, or one an entry point has already validated. A ``False`` that is wrong does not
        raise: indices at or above ``n_vertices`` collide in the packing and silently group two
        different edges as one, and a negative index is read as a huge unsigned digit.
    return_counts
        Also return how many rows of ``edges_sorted`` each unique edge has (``numpy.unique``'s
        ``return_counts``): its face count on a mesh, ``1`` on a boundary edge. Counted in the pass
        that writes the inverse.

    Returns
    -------
    unique_edges : odt.Array2dInt32
        ``(m, 2)`` unique undirected vertex pairs, ``m <= 3 * n_faces``.
    inverse : wp.array[wp.int32]
        ``(3 * n_faces,)`` inverse indices, ``unique_edges[inverse[i]] == edges_sorted[i]``.
    counts : wp.array[wp.int32]
        ``(m,)`` occurrences of each unique edge, ``numpy.bincount(inverse)``. Only with
        ``return_counts``.

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

    Examples
    --------
    A cube has 12 triangles, 36 corner edges and 18 unique edges (12 sides and 6 face diagonals);
    each unique edge of a closed mesh is reached from exactly two corners:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    unique_edges, inverse = od.edges.edges_unique(faces)
    print(unique_edges.shape)  # (18, 2)
    print(inverse.shape)  # (36,)
    print(np.bincount(inverse.numpy()).tolist() == [2] * 18)  # True
    unique_edges, inverse, counts = od.edges.edges_unique(faces, return_counts=True)
    print(counts.numpy().tolist() == [2] * 18)  # True
    ```
    """
    require_same_device(faces=faces, edges_sorted=edges_sorted)
    if edges_sorted is not None:
        odt.ensure_edge_pairs(edges_sorted, "edges_sorted")
    n_faces = faces.size // 3
    device = faces.device

    if n_faces == 0:
        empty_edges = odt.empty_2d((0, 2), wp.int32, device=device)
        empty_inv = _launch.empty(0, dtype=wp.int32, device=device)
        if return_counts:
            return empty_edges, empty_inv, _launch.empty(0, dtype=wp.int32, device=device)
        return empty_edges, empty_inv

    if edges_sorted is None:
        # The keys are built straight off ``faces``: ``face_edge_keys`` packs each corner's sorted
        # pair exactly as ``hash_indices_rows`` packs the corresponding ``faces_to_edges`` row, so
        # the ``(3 * n_faces, 2)`` table is never written or read back, and any range check reduces
        # the ``3 * n_faces`` face buffer -- the same values, half the entries.
        radix, check = _face_edge_radix(faces, n_faces, n_vertices, validate)
        unique_edges, inverse, counts = _unique_edges_from_faces(
            faces, n_faces, radix, check, return_counts
        )
    else:
        if n_vertices is None:
            if validate:
                # Inferring the bound from this very buffer already reduces it, and the packing's
                # own check would reduce it again to re-test a bound derived from it -- only the
                # negative half could ever fire. One reduction answers both.
                n_vertices = od.array.index_bound(edges_sorted, require_non_negative=True)
                validate = False
            else:
                # Nothing is being checked, so the only thing the count would be used for is the
                # radix -- and the pair radix bounds every ``int32`` without reducing the rows to
                # find out. Row order is unchanged; the emit below unpacks against whichever radix
                # packed them.
                n_vertices = INDEX_RADIX_PAIR
        elif n_vertices <= 0:
            raise ValueError(f"n_vertices must be positive, got {n_vertices}")
        elif validate:
            # The packing's range check: the reduction's readback, then the same two raises as
            # the face path's key-pass check.
            low, high = od.reduce.minmax(edges_sorted)
            if low < 0:
                raise ValueError(f"edge indices must be non-negative, got a minimum of {low}")
            if high >= n_vertices:
                raise ValueError(
                    f"edge indices must be less than n_vertices {n_vertices}, "
                    f"got a maximum of {high}"
                )
        unique_edges, inverse, counts = _unique_edges_from_sorted_rows(
            edges_sorted, n_vertices, return_counts
        )
    if return_counts:
        assert counts is not None
        return unique_edges, inverse, counts
    return unique_edges, inverse


def unique_edge_count(
    faces: wp.array[wp.int32], n_vertices: int | None = None, *, validate: bool = True
) -> int:
    """
    Count the unique undirected edges, without building the edge table.

    Equals ``edges_unique(faces, n_vertices=n_vertices)[0].shape[0]``: the same sort and run scan,
    stopped before the rows and the inverse map are written.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Vertex count bounding the indices, as in [`edges_unique`][ordito.edges.edges_unique];
        inferred from ``faces`` when ``None``.
    validate
        As in [`edges_unique`][ordito.edges.edges_unique]: check the index range. With
        ``n_vertices`` given the check rides in the count's own readback; inferring the bound
        costs one more.

    Returns
    -------
    int
        Unique undirected edge count. ``0`` for an empty mesh.

    Raises
    ------
    ValueError
        If ``n_vertices`` is not positive, or ``validate`` is ``True`` and a face index is negative
        or reaches ``n_vertices``.

    See Also
    --------
    [`edges_unique`][ordito.edges.edges_unique]
        The table this counts.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(od.edges.unique_edge_count(faces))  # 18
    ```
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0
    radix, check = _face_edge_radix(faces, n_faces, n_vertices, validate)
    return _sorted_edge_run_ranks(faces, n_faces, radix, check)[3]


def _face_edge_radix(
    faces: wp.array[wp.int32], n_faces: int, n_vertices: int | None, validate: bool
) -> tuple[int, bool]:
    """
    Return the radix every corner edge's sorted-pair key packs against, and whether to check it.

    The ``edges_sorted is None`` half of [`edges_unique`][ordito.edges.edges_unique], resolving
    ``n_vertices`` and ``validate`` exactly as the composed ``faces_to_edges`` +
    ``hash_indices_rows`` path does: an inferred bound checks the negative half, a supplied one is
    checked against, and ``validate=False`` with no bound packs against ``INDEX_RADIX_PAIR``. Only
    the inferred bound needs a reduction here, since the radix depends on it; a supplied bound is
    returned with ``True``, and the key kernel checks every index against it in the same pass,
    its flag read back with the edge count.
    """
    if n_vertices is not None and n_vertices <= 0:
        raise ValueError(f"n_vertices must be positive, got {n_vertices}")
    if n_vertices is not None:
        return n_vertices, validate
    if validate:
        owned = faces if faces.size == 3 * n_faces else odt.as_dense(faces[: 3 * n_faces])
        low, high = od.reduce.minmax(owned)
        if low < 0:
            raise ValueError(f"edge indices must be non-negative, got a minimum of {low}")
        return int(high) + 1, False
    return INDEX_RADIX_PAIR, False


def _unique_edges_from_faces(
    faces: wp.array[wp.int32], n_faces: int, radix: int, check: bool, return_counts: bool
) -> tuple[odt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32] | None]:
    """
    ``(unique_edges, inverse)`` from one radix sort of every corner's packed edge key.

    The unique edges are the runs of the sorted keys, so a run-start mark, an inclusive scan and
    one emit give exactly what ``grouping.unique_1d`` over the same keys returns -- ascending-key
    rows and the corner map into them -- without its open-addressing table, whose ~``2 * 3 *
    n_faces`` slots take random atomics where the sort streams. The emit re-derives each run start
    from the sorted keys; the edge count (and, with ``check``, the range flag) comes back in one
    read before it. With ``return_counts`` the emit also sums each run's length into the counts.
    """
    device = faces.device
    n = 3 * n_faces
    keys, order, ranks, n_edges = _sorted_edge_run_ranks(faces, n_faces, radix, check)
    unique_edges = odt.empty_2d((n_edges, 2), wp.int32, device=device)
    inverse = _launch.empty(n, dtype=wp.int32, device=device)
    if not return_counts:
        _launch.launch(
            kernel_edges.emit_sorted_unique_edges,
            dim=n,
            inputs=[faces, keys, order, ranks, unique_edges, inverse],
            device=device,
        )
        return unique_edges, inverse, None
    counts = _launch.zeros(n_edges, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_edges.emit_sorted_unique_edges_counts,
        dim=n,
        inputs=[faces, keys, order, ranks, unique_edges, inverse, counts],
        device=device,
    )
    return unique_edges, inverse, counts


def _sorted_edge_run_ranks(
    faces: wp.array[wp.int32], n_faces: int, radix: int, check: bool
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32], wp.array[wp.int32], int]:
    """
    ``(keys, order, ranks, n_edges)``: sorted edge keys, their 1-based unique-edge ranks, the count.

    The part of [`edges_unique`][ordito.edges.edges_unique] that both the table emit and
    [`unique_edge_count`][ordito.edges.unique_edge_count] need. The run marks and their scan live
    in the payload's upper half, which is free scratch once the sort has run. With ``check`` the
    key pass range-checks every index against ``radix`` and its flag sits in the slot after the
    scan, so the count and the validation come back in one read, raising before any emit.
    """
    n = 3 * n_faces
    keys, order = _sorted_face_edge_keys(faces, n_faces, radix, radix if check else None)
    ranks, n_edges = _sorted_run_ranks(keys, order, n, radix if check else None)
    return keys, order, ranks, n_edges


def _sorted_face_edge_keys(
    faces: wp.array[wp.int32], n_faces: int, radix: int, bound: int | None
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Every corner's packed edge key, radix-sorted with its corner index as the payload.

    The keys and the identity payload are written straight into the sort's double-width buffers
    (the launch ``adjacency.sorted_face_edge_keys`` makes). With a ``bound`` the key kernel also
    range-checks every face index against it and raises the zeroed slot ``order[6 * n_faces]`` to
    a nonzero code on a miss; [`_raise_on_range_code`][ordito.edges._raise_on_range_code] reads it.
    """
    device = faces.device
    n = 3 * n_faces
    if bound is None:
        keys, order = _launch.sort_buffers(n, wp.uint64, device)
        _launch.launch(
            kernel_adjacency.face_edge_keys_and_order,
            dim=n_faces,
            inputs=[faces, wp.uint64(radix), keys, order],
            device=device,
        )
    else:
        keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
        order = _launch.zeros(2 * n + 1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_edges.face_edge_keys_checked,
            dim=n_faces,
            inputs=[faces, wp.uint64(radix), bound, 2 * n, keys, order],
            device=device,
        )
    _launch.radix_sort_pairs(
        keys, order, count=n, end_bit=min(64, max(1, (radix * radix - 1).bit_length()))
    )
    return keys, order


def _unique_edges_from_sorted_rows(
    edges_sorted: odt.Array2dInt32, n_vertices: int, return_counts: bool
) -> tuple[odt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32] | None]:
    """
    ``(unique_edges, inverse)`` of precomputed min-first rows, packed by ``n_vertices``.

    The rows are packed as ``grouping.hash_indices_rows`` packs them, straight into the radix
    sort's double-width buffers with the identity payload, and sorted over only the bits a key
    below ``n_vertices ** 2`` can set; the runs are then emitted as
    [`_unique_edges_from_faces`][ordito.edges._unique_edges_from_faces] emits them, each unique row
    unpacked from its key.
    """
    device = edges_sorted.device
    n = int(edges_sorted.shape[0])
    if n == 0:
        nothing = _launch.empty(0, dtype=wp.int32, device=device)
        return odt.empty_2d((0, 2), wp.int32, device=device), nothing, nothing
    keys, order = _launch.sort_buffers(n, wp.uint64, device)
    _launch.launch(
        kernel_grouping.pack_indices_and_order,
        dim=n,
        inputs=[edges_sorted, wp.uint64(n_vertices)],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(
        keys, order, count=n, end_bit=min(64, max(1, (n_vertices * n_vertices - 1).bit_length()))
    )
    ranks, n_edges = _sorted_run_ranks(keys, order, n, None)
    unique_edges = odt.empty_2d((n_edges, 2), wp.int32, device=device)
    inverse = _launch.empty(n, dtype=wp.int32, device=device)
    counts = _launch.zeros(n_edges, dtype=wp.int32, device=device) if return_counts else None
    _launch.launch(
        kernel_edges.emit_sorted_unique_edges_from_keys,
        dim=n,
        inputs=[keys, order, ranks, wp.uint64(n_vertices)],
        outputs=[unique_edges, inverse, counts],
        device=device,
    )
    return odt.as_array2d(unique_edges, wp.int32), inverse, counts


def _sorted_run_ranks(
    keys: wp.array[wp.uint64], order: wp.array[wp.int32], n: int, bound: int | None
) -> tuple[wp.array[wp.int32], int]:
    """
    ``(ranks, n_edges)`` of sorted double-width edge-key buffers: 1-based run ranks, the run count.

    The run marks and their inclusive scan live in the payload's upper half, which is free scratch
    once the sort has run. With a ``bound`` the key pass left its range code in the slot after the
    payload, read in the same copy as the count and raised on before any emit.
    """
    ranks = _launch.view(order, n, 2 * n)
    _launch.launch(
        kernel_grouping.MARK_SORTED_RUN_STARTS[wp.uint64],
        dim=n,
        inputs=[keys, ranks],
        device=keys.device,
    )
    _launch.array_scan(ranks, ranks, inclusive=True)
    # The edge count sizes the caller's table; the range flag (when checked) rides in the same copy.
    if bound is not None:
        n_edges, code = read_values(order, 2 * n - 1, 2)
        _raise_on_range_code(int(code), bound)
    else:
        n_edges = read_scalar(ranks, n - 1)
    return ranks, int(n_edges)


def _raise_on_range_code(code: int, n_vertices: int) -> None:
    """Raise the ``ValueError`` a ``face_edge_keys_checked`` range code stands for, if any."""
    if code == int(kernel_edges.INDEX_NEGATIVE):
        raise ValueError("edge indices must be non-negative, got a negative index")
    if code == int(kernel_edges.INDEX_ABOVE_BOUND):
        raise ValueError(
            f"edge indices must be less than n_vertices {n_vertices}, got one at or above it"
        )


def edges_unique_inverse(
    faces: wp.array[wp.int32],
    edges_sorted: odt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> wp.array[wp.int32]:
    """
    Inverse mapping from sorted edges into [`edges_unique`][ordito.edges.edges_unique].

    Maps [`faces_to_edges`][ordito.edges.faces_to_edges] (sorted) into `edges_unique`:
    ``edges_unique(faces)[edges_unique_inverse(faces)] == faces_to_edges(faces, sorted=True)``

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` face index buffer.
    edges_sorted
        ``(3 * n_faces, 2)`` precomputed sorted edges, or ``None`` to build them from ``faces``.
    n_vertices
        Total vertex count. When ``None``, inferred from ``edges_sorted`` with a device-host sync.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` inverse indices.

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

    Examples
    --------
    Indexing the unique edges with the inverse reproduces the sorted corner edges:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    unique_edges, _inverse = od.edges.edges_unique(faces)
    inverse = od.edges.edges_unique_inverse(faces)
    corner_edges = od.edges.faces_to_edges(faces, sorted=True)
    print(np.array_equal(unique_edges.numpy()[inverse.numpy()], corner_edges.numpy()))  # True
    ```
    """
    require_same_device(faces=faces, edges_sorted=edges_sorted)
    return edges_unique(faces, edges_sorted=edges_sorted, n_vertices=n_vertices)[1]


def edges_unique_length(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: odt.Array2dInt32 | None = None,
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
        ``(3 * n_faces,)`` face index buffer.
    unique_edges
        ``(m, 2)`` precomputed unique edges, or ``None`` to compute them from ``faces``.
    n_vertices
        Total vertex count passed to [`edges_unique`][ordito.edges.edges_unique]. Ignored when
        ``unique_edges`` is already provided. When ``None`` it is taken from ``vertices`` rather
        than inferred from ``faces``, which would cost a host readback for a count this function
        was already handed.
    validate
        Forwarded to [`edges_unique`][ordito.edges.edges_unique] when ``unique_edges`` is built
        here; ignored when ``unique_edges`` is supplied. Pass ``False`` only where the face
        indices' range is structurally guaranteed.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` edge lengths on ``faces.device``.

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

    Examples
    --------
    The unit cube's 18 unique edges are 12 sides of length 1 and 6 diagonals of length ``sqrt(2)``:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    lengths = od.edges.edges_unique_length(vertices, faces)
    print(lengths.shape)  # (18,)
    print(f"{lengths.numpy().min():.4f} {lengths.numpy().max():.4f}")  # 1.0000 1.4142
    print(int((lengths.numpy() > 1.1).sum()))  # 6
    ```
    """
    require_same_device(vertices=vertices, faces=faces, unique_edges=unique_edges)
    if unique_edges is not None:
        return _edge_lengths(vertices, unique_edges, "unique_edges")
    # ``vertices.shape[0]`` when the caller gave no count: ``edges_unique`` would otherwise infer it
    # from ``faces`` with ``array.index_bound``, a device reduction plus a host readback, for a
    # number every caller of *this* function already holds in the array it passed. It is also the
    # bound ``validate`` is documented against.
    if n_vertices is None:
        n_vertices = vertices.size
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.float32, device=vertices.device)
    # ``edges_unique``'s sort and run scan, then the lengths written at each run's rank: the edge
    # table and the corner inverse it would emit are never built.
    radix, check = _face_edge_radix(faces, n_faces, n_vertices, validate)
    keys, order, ranks, n_edges = _sorted_edge_run_ranks(faces, n_faces, radix, check)
    lengths = _launch.empty(n_edges, dtype=wp.float32, device=vertices.device)
    _launch.launch(
        kernel_edges.emit_sorted_unique_edge_lengths,
        dim=3 * n_faces,
        inputs=[vertices, faces, keys, order, ranks, lengths],
        device=vertices.device,
    )
    return lengths


def edges_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], edges_in: odt.Array2dInt32 | None = None
) -> wp.array[wp.float32]:
    """
    Euclidean length of every directed edge (one per face half-edge).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` face index buffer.
    edges_in
        ``(3 * n_faces, 2)`` precomputed directed edges, or ``None`` to compute them from
        ``faces``.

    Returns
    -------
    wp.array[wp.float32]
        ``(3 * n_faces,)`` edge lengths on ``faces.device``.

    Raises
    ------
    TypeError
        If ``edges_in`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``edges_in`` does not have exactly two columns.
    RuntimeError
        If ``vertices``, ``faces`` and ``edges_in`` are not all on one device.

    Examples
    --------
    One length per corner edge, so a shared edge appears once from each of its faces:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    lengths = od.edges.edges_length(vertices, faces)
    print(lengths.shape)  # (36,)
    print(f"{lengths.numpy().sum():.4f}")  # 40.9706
    ```
    """
    require_same_device(vertices=vertices, faces=faces, edges_in=edges_in)
    if edges_in is not None:
        return _edge_lengths(vertices, edges_in, "edges_in")
    # Without a table the rows are the halfedges, so the lengths come straight off ``faces``.
    n_halfedges = faces.size // 3 * 3
    out = _launch.empty(n_halfedges, dtype=wp.float32, device=vertices.device)
    if n_halfedges > 0:
        _launch.launch(
            kernel_edges.halfedge_lengths,
            dim=n_halfedges,
            inputs=[vertices, faces, out],
            device=vertices.device,
        )
    return out


def _edge_lengths(
    vertices: wp.array[wp.vec3], edges: odt.Array2dInt32, name: str
) -> wp.array[wp.float32]:
    """
    Euclidean length of every row of an ``(m, 2)`` edge table.

    The shared body of [`edges_unique_length`][ordito.edges.edges_unique_length] and
    [`edges_length`][ordito.edges.edges_length], which differ only in which edge table they
    obtain first. Stays a kernel rather than a ``wp.map`` over gathered endpoints: the columns
    of ``edges`` are strided views, and Warp's Python-scope gather ignores a view's stride.

    ``name`` is the *caller's* parameter name, so the shape guard names the keyword the caller
    actually passed. The guard runs here rather than at the two entry points because the kernel
    reads columns 0 and 1 unconditionally: a wider table is accepted silently and its third column
    ignored, where a rank-1 buffer raises at launch -- so only the wide case needs catching, and it
    needs catching on both paths.
    """
    odt.ensure_edge_pairs(edges, name)
    m = int(edges.shape[0])
    device = vertices.device
    if m == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    out = _launch.empty(m, dtype=wp.float32, device=device)
    _launch.launch(kernel_edges.edge_lengths, dim=m, inputs=[vertices, edges, out], device=device)
    return out


def face_edge_lengths(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> odt.Array2dFloat32:
    """
    Per-face edge lengths, in the intrinsic column order the cotangent formulas expect.

    Column ``e`` holds the length of the edge *opposite* corner ``e``, matching
    [`cotmatrix_entries_intrinsic`][ordito.laplacian.cotmatrix_entries_intrinsic] and
    ``igl::cotmatrix_entries``' intrinsic overload. Unlike
    [`edges_unique_length`][ordito.edges.edges_unique_length] this is a per-*corner* table: an
    interior edge appears twice, which is what lets the entries be perturbed per face.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    odt.Array2dFloat32
        ``(n_faces, 3)`` edge lengths on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`mollify_intrinsic`][ordito.laplacian.mollify_intrinsic]
    [`edges_unique_length`][ordito.edges.edges_unique_length]

    Examples
    --------
    Column ``k`` is the edge opposite corner ``k``; on a cube every face has two sides of length 1
    and one diagonal:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    lengths = od.edges.face_edge_lengths(vertices, faces)
    print(lengths.shape)  # (12, 3)
    print(np.sort(lengths.numpy()[0]).round(4))  # [1.     1.     1.4142]
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    lengths = odt.empty_2d((n_faces, 3), wp.float32, device=device)
    if n_faces == 0:
        return odt.as_array2d(lengths, wp.float32)
    _launch.launch(
        kernel_edges.face_edge_lengths,
        dim=n_faces,
        inputs=[vertices, faces, lengths],
        device=device,
    )
    return odt.as_array2d(lengths, wp.float32)


def mean_edge_length(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Mean length of the ``3 * n_faces`` per-face edges, counting a shared edge once per face.

    Every face contributes all three of its edges, so an edge shared by two faces is counted twice
    and a boundary edge once -- which is what separates this from
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length], the same average over each
    edge once. That weighting is deliberate, not an oversight: it is the average
    libigl's ``CurvatureCalculator::getAverageEdge`` computes, which is what
    ``igl::principal_curvature`` calls to set its sphere-search radius, and what
    [`principal_curvature`][ordito.curvature.principal_curvature] therefore uses here.

    !!! note "There are two edge averages, and they are not interchangeable"
        [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length] counts each edge once,
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
        ``(3 * n_faces,)`` face index buffer.

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
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length]
        The same average taken over the unique edges instead.
    [`edges_length`][ordito.edges.edges_length]
        The per-face lengths this averages.

    Examples
    --------
    On a closed mesh this equals the unique-edge mean; on an open grid the boundary edges count once
    and the interior ones twice, so the two averages differ:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(f"{od.edges.mean_edge_length(vertices, faces):.4f}")  # 1.1381
    vertices, faces = od.creation.grid(count=(2, 2))
    print(f"{od.edges.mean_edge_length(vertices, faces):.4f}")  # 1.1381
    print(f"{od.edges.mean_unique_edge_length(vertices, faces):.4f}")  # 1.0828
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    n_halfedges = faces.size // 3 * 3
    if n_halfedges == 0:
        return 0.0
    # The total is ``edge_length_sum``'s, read back once.
    return float(read_scalar(edge_length_sum(vertices, faces), 0)) / n_halfedges


def edge_length_sum(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.float32]:
    """
    Total length of the ``3 * n_faces`` per-face edges, left on the device.

    The sum [`mean_edge_length`][ordito.edges.mean_edge_length] divides by ``3 * n_faces``, for a
    caller whose kernels consume it on the device and would otherwise wait on a readback to pass
    the mean in as a scalar. A shared edge counts once per face, as in the mean. Summed in
    [`sum`][ordito.reduce.sum]'s order: fixed under
    [`is_deterministic`][ordito.config.is_deterministic], arrival order otherwise.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    wp.array[wp.float32]
        ``(1,)`` the total on ``vertices.device``; ``0`` when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`mean_edge_length`][ordito.edges.mean_edge_length]
        The same total over the edge count, read back.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    total = od.edges.edge_length_sum(vertices, faces)
    print(total.shape, f"{total.numpy()[0] / faces.size:.4f}")  # (1,) 1.1381
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_halfedges = faces.size // 3 * 3
    if n_halfedges == 0:
        return _launch.zeros(1, dtype=wp.float32, device=device)
    # The lengths are formed inside the sum's own blocks: no ``(3 * n_faces,)`` length buffer.
    terms = kernel_edges.HalfedgeLengthTerms()
    terms.n = n_halfedges
    terms.vertices = vertices
    terms.faces = faces
    return od.reduce.sum_of_terms(
        kernel_edges.HALFEDGE_LENGTH_SUM, [terms], n_halfedges, device=device
    )


def mean_unique_edge_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, validate: bool = True
) -> float:
    """
    Mean length of the **unique** undirected edges.

    Each edge is counted once, however many faces share it. This is the definition
    ``igl::avg_edge_length`` and MeshLab's ``avg_edge_length`` both use, and the one
    ``igl::heat_geodesics`` picks its diffusion timestep from -- so it is what
    [`heat_geodesic`][ordito.heat.heat_geodesic] and
    [`vector_heat_operators`][ordito.heat.vector_heat_operators] use here.

    See [`mean_edge_length`][ordito.edges.mean_edge_length] for the per-face average, how far the
    two diverge on an open mesh, and why libigl carries both.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` face index buffer.
    validate
        Forwarded to [`edges_unique`][ordito.edges.edges_unique]. Pass ``False`` only where the
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
    [`mean_edge_length`][ordito.edges.mean_edge_length]
        The same average taken over the per-face edges instead.
    [`edges_unique_length`][ordito.edges.edges_unique_length]
        The per-edge lengths this averages.

    Examples
    --------
    The cube's 18 unique edges average ``(12 + 6 * sqrt(2)) / 18``:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(f"{od.edges.mean_unique_edge_length(vertices, faces):.4f}")  # 1.1381
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0
    # The unique edges are the sorted corner keys' run starts, so their lengths and count fold
    # straight off the sort into per-block ``(sum, count, range flag)`` partials: no emitted edge
    # table, no length buffer, and one readback carrying the total, the count and the validation.
    n_vertices = vertices.size
    n = 3 * n_faces
    keys, order = _sorted_face_edge_keys(faces, n_faces, n_vertices, n_vertices)
    partials = _launch.empty(kernel_reduce.blocks_1d(n), dtype=wp.vec3, device=vertices.device)
    _launch.launch_tiled(
        kernel_edges.unique_edge_length_partials,
        dim=[partials.size],
        inputs=[vertices, keys, n, wp.uint64(n_vertices), order, 2 * n, partials],
        block_dim=TILE_1D,
        device=vertices.device,
    )
    total, count, code = od.reduce.sum(partials)
    if validate:
        _raise_on_range_code(int(code), n_vertices)
    return float(total) / float(count)
