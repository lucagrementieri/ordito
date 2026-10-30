"""
Equality of meshes, point sets, row sets and partitions up to how their elements are numbered.

Two triangle meshes describe the same surface when one is the other with its vertices renumbered,
its faces listed in another order, and each face started at another corner. None of those choices
is part of the answer: a parallel construction and a serial one, or two libraries, settle them
differently and are both right. The functions here compare what remains.

- [`meshes_allclose`][ordito.equality.meshes_allclose] is the whole comparison: positions within a
  tolerance, each vertex matched to exactly one of the other mesh's, and the same faces under that
  match.
- [`faces_equal`][ordito.equality.faces_equal] is the comparison of two face buffers over one
  vertex buffer, where the vertex numbering is already shared.
- [`canonical_faces`][ordito.equality.canonical_faces] is the form both compare through: every
  face rotated to start at its smallest corner, the faces in lexicographic order.

The same question asked of the other unordered answers a mesh library returns:

- [`points_allclose`][ordito.equality.points_allclose]: two point sets, up to order (the vertex
  half of [`meshes_allclose`][ordito.equality.meshes_allclose]).
- [`rows_equal`][ordito.equality.rows_equal]: two tables of integer rows (edges, segments, index
  pairs), up to row order.
- [`partitions_equal`][ordito.equality.partitions_equal]: two labellings, up to the names of
  their labels.

A rotation keeps a face's winding and a reflection does not, so a face and its reversed twin stay
distinct throughout: a mesh with one face flipped is not equal to the original.
"""

from __future__ import annotations

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.constants import ALLCLOSE_ATOL, ALLCLOSE_RTOL
from ordito.kernels import equality as kernel_equality


def meshes_allclose(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    rtol: float = ALLCLOSE_RTOL,
    atol: float = ALLCLOSE_ATOL,
) -> bool:
    """
    Test whether two meshes are equal up to vertex numbering, face order and face rotation.

    Each vertex of ``a`` is matched to its nearest vertex of ``b``. The meshes are equal when every
    match lies within ``atol + rtol * |b_vertex|`` (the Euclidean distance, ``numpy.allclose``'s
    asymmetric convention applied to points), no vertex of ``b`` is matched twice, and the faces of
    ``a`` written through the match are the faces of ``b`` as a multiset of rows, each compared from
    its smallest corner.

    Parameters
    ----------
    vertices_a
        ``(n_vertices,)`` positions of the first mesh.
    faces_a
        ``(3 * n_faces,)`` flat triangle index buffer of the first mesh.
    vertices_b
        ``(n_vertices,)`` positions of the second mesh.
    faces_b
        ``(3 * n_faces,)`` flat triangle index buffer of the second mesh.
    rtol
        Relative tolerance on the distance between matched vertices, against the ``b`` vertex's
        distance from the origin. Defaults to ``1e-05``.
    atol
        Absolute tolerance on the distance between matched vertices. Defaults to ``1e-08``.

    Returns
    -------
    bool
        ``True`` when the meshes are equal. ``False`` when their vertex or face counts differ, or
        when a face names a vertex outside ``[0, n_vertices)``.

    Raises
    ------
    ValueError
        If ``faces_a`` or ``faces_b`` does not hold a whole number of triangles.
    RuntimeError
        If the four arrays are not all on one device.

    Notes
    -----
    The match is by nearest neighbour, not by sorting coordinates, so it does not depend on how two
    nearly equal coordinates of different vertices round on either side. It does require the
    tolerance to be smaller than half the distance between any two vertices of ``b``: two vertices
    of ``a`` within tolerance of one vertex of ``b`` both match it, and the meshes compare unequal.
    Weld coincident vertices
    ([`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices]) before comparing.

    Vertices no face uses take part in the match like any other.

    See Also
    --------
    [`faces_equal`][ordito.equality.faces_equal]
        The same comparison of the faces alone, over one shared vertex buffer.
    [`allclose`][ordito.array.allclose]
        Element-wise comparison, for two meshes in the same numbering.

    Examples
    --------
    A tetrahedron against itself with its vertices renumbered and its faces reordered and rotated:

    ```python
    import warp as wp

    import ordito as od

    vertices_np = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    faces_np = [0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2]
    vertices = wp.array(vertices_np, dtype=wp.vec3)
    faces = wp.array(faces_np, dtype=wp.int32)

    # The second mesh lists old vertices 2, 0, 3, 1, so old vertex 0 is new vertex 1, and so on.
    renumbered = wp.array([vertices_np[k] for k in (2, 0, 3, 1)], dtype=wp.vec3)
    reordered = wp.array([2, 1, 3, 0, 3, 1, 2, 0, 1, 3, 0, 2], dtype=wp.int32)
    print(od.equality.meshes_allclose(vertices, faces, renumbered, reordered))  # True

    # One face flipped is a different mesh.
    flipped = wp.array([3, 1, 2, 0, 3, 1, 2, 0, 1, 3, 0, 2], dtype=wp.int32)
    print(od.equality.meshes_allclose(vertices, faces, renumbered, flipped))  # False
    ```
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    _require_triangles(faces_a=faces_a, faces_b=faces_b)
    n_vertices = vertices_a.size
    if vertices_b.size != n_vertices or faces_b.size != faces_a.size:
        return False
    if n_vertices == 0:
        return faces_a.size == 0
    unequal = _launch.zeros(1, dtype=wp.int32, device=vertices_a.device)
    match = _flag_unmatched_points(vertices_a, vertices_b, rtol, atol, unequal)
    # A failed match still leaves every entry of ``match`` a valid ``b`` index, so the faces are
    # compared either way and the one readback below answers both.
    _flag_unequal_faces(faces_a, match, faces_b, n_vertices, unequal)
    # The answer is a host ``bool``.
    return int(read_scalar(unequal, 0)) == 0


def points_allclose(
    points_a: wp.array[wp.vec3],
    points_b: wp.array[wp.vec3],
    *,
    rtol: float = ALLCLOSE_RTOL,
    atol: float = ALLCLOSE_ATOL,
) -> bool:
    """
    Test whether two point sets are equal within a tolerance, up to the order of their points.

    Each point of ``a`` is matched to its nearest point of ``b``. The sets are equal when every
    match lies within ``atol + rtol * |b_point|`` (the Euclidean distance) and no point of ``b`` is
    matched twice, so the match is a bijection.

    Parameters
    ----------
    points_a
        ``(n,)`` points.
    points_b
        ``(n,)`` points.
    rtol
        Relative tolerance on the distance between matched points, against the ``b`` point's
        distance from the origin. Defaults to ``1e-05``.
    atol
        Absolute tolerance on the distance between matched points. Defaults to ``1e-08``.

    Returns
    -------
    bool
        ``True`` when the sets are equal. ``False`` when their sizes differ.

    Raises
    ------
    RuntimeError
        If ``points_a`` and ``points_b`` are not on one device.

    Notes
    -----
    As in [`meshes_allclose`][ordito.equality.meshes_allclose], the tolerance must be smaller than
    half the distance between any two points of ``b``: two points of ``a`` within tolerance of one
    point of ``b`` both match it, and the sets compare unequal. A set holding a point twice equals
    only another holding it twice, and only at a tolerance the duplicates themselves pass.

    See Also
    --------
    [`meshes_allclose`][ordito.equality.meshes_allclose]
        The same match, with the faces carried through it.
    [`allclose`][ordito.array.allclose]
        Element-wise comparison, for two sets in the same order.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=wp.vec3)
    shuffled = wp.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    moved = wp.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=wp.vec3)
    print(od.equality.points_allclose(points, shuffled))  # True
    print(od.equality.points_allclose(points, moved))  # False
    ```
    """
    require_same_device(points_a=points_a, points_b=points_b)
    if points_b.size != points_a.size:
        return False
    if points_a.size == 0:
        return True
    unequal = _launch.zeros(1, dtype=wp.int32, device=points_a.device)
    _flag_unmatched_points(points_a, points_b, rtol, atol, unequal)
    # The answer is a host ``bool``.
    return int(read_scalar(unequal, 0)) == 0


def _flag_unmatched_points(
    points_a: wp.array[wp.vec3],
    points_b: wp.array[wp.vec3],
    rtol: float,
    atol: float,
    unequal: wp.array[wp.int32],
) -> odt.Array1dInt32:
    """
    Match each of the ``n > 0`` points of ``a`` to its nearest of ``b``; return the match.

    ``unequal[0]`` is raised unless the match is a bijection within tolerance. Every entry of the
    returned match is a valid ``b`` index either way.
    """
    # The hash-grid backend, though the BVH's ``k == 1`` descent skips its two readbacks: on a
    # cloud queried against itself the two ran at par at 100 k points and the grid in 0.36x the
    # BVH's time at 2 M (CUDA).
    match, _distance = od.neighbors.query_nearest(points_b, points_a)
    hits = _launch.zeros(points_b.size, dtype=wp.int32, device=points_a.device)
    _launch.launch(
        kernel_equality.flag_unmatched_vertices,
        dim=points_a.size,
        inputs=[points_a, points_b, match, wp.float32(rtol), wp.float32(atol), hits, unequal],
        device=points_a.device,
    )
    return match


def faces_equal(
    faces_a: wp.array[wp.int32], faces_b: wp.array[wp.int32], *, n_vertices: int | None = None
) -> bool:
    """
    Test whether two face buffers hold the same faces, up to face order and rotation.

    The two buffers index one vertex buffer. They are equal when they hold the same multiset of
    triangles, each compared from its smallest corner: a face may be listed anywhere and start at
    any of its three corners, but its winding counts.

    Parameters
    ----------
    faces_a
        ``(3 * n_faces,)`` flat triangle index buffer.
    faces_b
        ``(3 * n_faces,)`` flat triangle index buffer over the same vertices.
    n_vertices
        Exclusive upper bound on every vertex index, typically the vertex count. A face naming an
        index outside ``[0, n_vertices)`` makes the buffers unequal. When ``None`` (default) the
        bound is read off the two buffers.

    Returns
    -------
    bool
        ``True`` when the faces are equal. ``False`` when the face counts differ.

    Raises
    ------
    ValueError
        If either buffer does not hold a whole number of triangles, or -- when ``n_vertices`` is
        ``None`` -- holds a negative index.
    RuntimeError
        If ``faces_a`` and ``faces_b`` are not on one device.

    See Also
    --------
    [`meshes_allclose`][ordito.equality.meshes_allclose]
        The comparison of two meshes that number their vertices differently.
    [`canonical_faces`][ordito.equality.canonical_faces]
        The form the two buffers are compared in.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([0, 1, 2, 2, 1, 3], dtype=wp.int32)
    rotated = wp.array([1, 3, 2, 2, 0, 1], dtype=wp.int32)
    flipped = wp.array([1, 3, 2, 0, 2, 1], dtype=wp.int32)
    print(od.equality.faces_equal(faces, rotated))  # True
    print(od.equality.faces_equal(faces, flipped))  # False
    ```
    """
    require_same_device(faces_a=faces_a, faces_b=faces_b)
    _require_triangles(faces_a=faces_a, faces_b=faces_b)
    if faces_a.size != faces_b.size:
        return False
    if faces_a.size == 0:
        return True
    if n_vertices is None:
        # The bound sizes the sort keys, so it is read back before anything is keyed.
        n_vertices = max(_index_bound(faces_a), _index_bound(faces_b))
    unequal = _launch.zeros(1, dtype=wp.int32, device=faces_a.device)
    _flag_unequal_faces(faces_a, None, faces_b, n_vertices, unequal)
    # The answer is a host ``bool``.
    return int(read_scalar(unequal, 0)) == 0


def _flag_unequal_faces(
    faces_a: wp.array[wp.int32],
    vertex_map: odt.ArrayNdInt32 | None,
    faces_b: wp.array[wp.int32],
    n_vertices: int,
    unequal: wp.array[wp.int32],
) -> None:
    """
    Raise ``unequal[0]`` unless ``faces_a`` through ``vertex_map`` and ``faces_b`` are equal.

    Both buffers hold ``n_faces > 0`` faces; an out-of-range corner on either side raises the flag.
    """
    if faces_a.size == 0:
        return
    # Only the two sorted orders are kept: the comparison forms each side's canonical rows where it
    # reads them, so neither canonical buffer is written.
    n_faces = faces_a.size // 3
    keys_a, payload_a, keys_b, payload_b = _launch.carve_typed(
        ((wp.uint64, 2 * n_faces), (wp.int32, 2 * n_faces)) * 2, faces_a.device
    )
    order_a = _canonical_order(faces_a, vertex_map, n_vertices, unequal, (keys_a, payload_a))
    order_b = _canonical_order(faces_b, None, n_vertices, unequal, (keys_b, payload_b))
    _launch.launch(
        kernel_equality.flag_unequal_canonical_rows,
        dim=n_faces,
        inputs=[faces_a, vertex_map, order_a, faces_b, None, order_b, n_vertices, unequal],
        device=faces_a.device,
    )


def canonical_faces(
    faces: wp.array[wp.int32], *, n_vertices: int | None = None
) -> wp.array[wp.int32]:
    """
    Faces rotated to start at their smallest corner, in lexicographic order.

    Two face buffers over one vertex buffer hold the same faces, up to face order and the corner
    each face starts at, exactly when their canonical forms are equal element for element. The
    rotation keeps each face's winding, so a face and its reversed twin stay distinct.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Exclusive upper bound on every vertex index, typically the vertex count. When given it is
        trusted rather than checked, which skips a reduction over ``faces`` and its host readback:
        a bound at or below an index leaves that face out of order. When ``None`` (default) the
        indices are validated and the bound read off them.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` flat buffer of the same faces, each rotated to start at its smallest
        corner, in ascending lexicographic order of their rows.

    Raises
    ------
    ValueError
        If ``faces`` does not hold a whole number of triangles, or -- when ``n_vertices`` is
        ``None`` -- holds a negative index.

    See Also
    --------
    [`faces_equal`][ordito.equality.faces_equal]
        The comparison of two buffers through this form.
    [`unique_faces`][ordito.grouping.unique_faces]
        Deduplication of faces regardless of winding.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([3, 1, 2, 2, 0, 1], dtype=wp.int32)
    print(od.equality.canonical_faces(faces).numpy().tolist())  # [0, 1, 2, 1, 2, 3]
    ```
    """
    _require_triangles(faces=faces)
    if faces.size == 0:
        return _launch.empty(0, dtype=wp.int32, device=faces.device)
    if n_vertices is None:
        # The bound sizes the sort keys, so it is read back before anything is keyed.
        n_vertices = _index_bound(faces)
    return _canonical_faces(faces, None, n_vertices, None)


def _canonical_faces(
    faces: wp.array[wp.int32],
    vertex_map: odt.ArrayNdInt32 | None,
    n_vertices: int,
    invalid: wp.array[wp.int32] | None,
) -> wp.array[wp.int32]:
    """
    Return [`canonical_faces`][ordito.equality.canonical_faces] of ``faces`` through a map.

    ``vertex_map`` is ``None`` for the identity. ``invalid`` (or ``None``) has ``invalid[0]`` raised
    by a corner outside ``[0, n_vertices)``; through a map such a corner is read as ``0``.
    ``faces`` is non-empty.
    """
    device = faces.device
    n_faces = faces.size // 3
    order = _canonical_order(faces, vertex_map, n_vertices, invalid)
    out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_equality.emit_canonical_faces,
        dim=n_faces,
        inputs=[faces, vertex_map, n_vertices, order, out_faces],
        device=device,
    )
    return out_faces


def _canonical_order(
    faces: wp.array[wp.int32],
    vertex_map: odt.ArrayNdInt32 | None,
    n_vertices: int,
    invalid: wp.array[wp.int32] | None,
    buffers: tuple[wp.array[wp.uint64], wp.array[wp.int32]] | None = None,
) -> wp.array[wp.int32]:
    """
    Return the faces in canonical order: sorted position ``i`` holds canonical row ``i``'s face.

    ``vertex_map``, ``invalid`` and ``faces`` as in ``_canonical_faces``, whose rows are
    ``kernels/equality.canonical_corner_triple`` of these faces. One radix sort while three
    indices below ``n_vertices`` pack into 64 bits (``n_vertices`` up to ``2 ** 21``). Past that,
    a least-significant-digit pair of stable sorts: the last two corners packed into one key, then
    the leading corner alone. ``buffers`` is the first sort's ``(keys, payload)`` when the caller
    carved them, shaped as ``_launch.sort_buffers``' are.
    """
    device = faces.device
    n_faces = faces.size // 3
    base = max(n_vertices, 1)
    whole_row = base**3 <= 1 << 64
    index_bits = max(1, (base - 1).bit_length())
    keys, order = (
        buffers if buffers is not None else _launch.sort_buffers(n_faces, wp.uint64, device)
    )
    _launch.launch(
        kernel_equality.canonical_face_keys,
        dim=n_faces,
        inputs=[faces, vertex_map, n_vertices, int(whole_row), keys, order, invalid],
        device=device,
    )
    if whole_row:
        end_bit = max(1, (base**3 - 1).bit_length())
        _launch.radix_sort_pairs(keys, order, count=n_faces, end_bit=end_bit)
        return order
    _launch.radix_sort_pairs(keys, order, count=n_faces, end_bit=32 + index_bits)
    leading, leading_order = _launch.sort_buffers(n_faces, wp.int32, device)
    _launch.launch(
        kernel_equality.leading_corner_keys,
        dim=n_faces,
        inputs=[faces, vertex_map, n_vertices, order, leading, leading_order],
        device=device,
    )
    _launch.radix_sort_pairs(leading, leading_order, count=n_faces, end_bit=index_bits)
    return leading_order


def rows_equal(rows_a: odt.Array2dInt32, rows_b: odt.Array2dInt32) -> bool:
    """
    Test whether two tables hold the same integer rows, up to the order of the rows.

    The tables are equal when they hold the same multiset of rows; the order of the entries within
    a row counts. For a row whose entries are unordered (an undirected edge), sort each row first
    ([`sort_rows`][ordito.array.sort_rows]).

    Parameters
    ----------
    rows_a
        ``(n, k)`` table of ``int32`` rows.
    rows_b
        ``(n, k)`` table of ``int32`` rows.

    Returns
    -------
    bool
        ``True`` when the tables hold the same rows. ``False`` when their shapes differ.

    Raises
    ------
    RuntimeError
        If ``rows_a`` and ``rows_b`` are not on one device.

    See Also
    --------
    [`faces_equal`][ordito.equality.faces_equal]
        The triangle form, which also ignores the corner each face starts at.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [2, 0]], dtype=wp.int32)
    reordered = wp.array([[2, 0], [0, 1], [1, 2]], dtype=wp.int32)
    reversed_edge = wp.array([[0, 2], [0, 1], [1, 2]], dtype=wp.int32)
    print(od.equality.rows_equal(edges, reordered))  # True
    print(od.equality.rows_equal(edges, reversed_edge))  # False
    ```
    """
    require_same_device(rows_a=rows_a, rows_b=rows_b)
    if tuple(rows_a.shape) != tuple(rows_b.shape):
        return False
    n_rows, width = int(rows_a.shape[0]), int(rows_a.shape[1])
    if n_rows == 0 or width == 0:
        return True
    order_a = _lexicographic_row_order(rows_a)
    order_b = _lexicographic_row_order(rows_b)
    unequal = _launch.zeros(1, dtype=wp.int32, device=rows_a.device)
    _launch.launch(
        kernel_equality.flag_unequal_ordered_rows,
        dim=(n_rows, width),
        inputs=[rows_a, order_a, rows_b, order_b, unequal],
        device=rows_a.device,
    )
    # The answer is a host ``bool``.
    return int(read_scalar(unequal, 0)) == 0


def _lexicographic_row_order(rows: odt.Array2dInt32) -> wp.array[wp.int32]:
    """
    Return the permutation listing the non-empty ``rows`` in ascending lexicographic order.

    One stable radix sort per column, last column first (least significant digit first), each
    carrying the permutation so far; ``int32`` keys sort by sign, so negative entries order too.
    """
    device = rows.device
    n_rows, width = int(rows.shape[0]), int(rows.shape[1])
    keys, order = _launch.sort_buffers(n_rows, wp.int32, device)
    previous: wp.array[wp.int32] | None = None
    spare = _launch.empty(2 * n_rows, dtype=wp.int32, device=device)
    for column in range(width - 1, -1, -1):
        _launch.launch(
            kernel_equality.column_keys,
            dim=n_rows,
            inputs=[rows, column, previous, keys, order],
            device=device,
        )
        _launch.radix_sort_pairs(keys, order, count=n_rows)
        previous, order, spare = order, spare, order
    assert previous is not None
    return previous


def partitions_equal(labels_a: wp.array[wp.int32], labels_b: wp.array[wp.int32]) -> bool:
    """
    Test whether two labellings induce the same partition, up to the names of the labels.

    Item ``i`` carries label ``labels_a[i]`` in one labelling and ``labels_b[i]`` in the other.
    The partitions are equal when two items share a label in one exactly when they share a label in
    the other: the labels correspond one to one, whatever their values.

    Parameters
    ----------
    labels_a
        ``(n,)`` label of each item.
    labels_b
        ``(n,)`` label of each item.

    Returns
    -------
    bool
        ``True`` when the partitions are equal. ``False`` when the lengths differ.

    Raises
    ------
    RuntimeError
        If ``labels_a`` and ``labels_b`` are not on one device.

    Notes
    -----
    The partitions are equal exactly when every label of one labelling meets a single label of
    the other, both ways: two sorts of the item's label pairs and one host readback.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    labels = wp.array([0, 0, 1, 2, 2], dtype=wp.int32)
    renamed = wp.array([7, 7, 3, 5, 5], dtype=wp.int32)
    merged = wp.array([7, 7, 3, 3, 3], dtype=wp.int32)
    print(od.equality.partitions_equal(labels, renamed))  # True
    print(od.equality.partitions_equal(labels, merged))  # False
    ```
    """
    require_same_device(labels_a=labels_a, labels_b=labels_b)
    if labels_b.size != labels_a.size:
        return False
    if labels_a.size == 0:
        return True
    device = labels_a.device
    n = labels_a.size
    # The partitions are equal when every ``a`` label meets one ``b`` label and every ``b`` label
    # one ``a`` label: each holds when no run of one label in the pairs sorted by it holds two
    # distinct pairs. Two sorts of packed pairs, ``a`` major then ``b`` major, each checked by a
    # pass over its neighbours, raise one flag, read once; nothing reads the sorts' payloads.
    keys_ab, order_ab, keys_ba, order_ba, unequal = _launch.carve_typed(
        (
            (wp.uint64, 2 * n),
            (wp.int32, 2 * n),
            (wp.uint64, 2 * n),
            (wp.int32, 2 * n),
            (wp.int32, 1),
        ),
        device,
    )
    _launch.launch(
        kernel_equality.label_pair_keys,
        dim=n,
        inputs=[labels_a, labels_b],
        outputs=[keys_ab, unequal],
        device=device,
    )
    _launch.radix_sort_pairs(keys_ab, order_ab, count=n)
    _launch.launch(
        kernel_equality.check_pairs_and_swap_keys,
        dim=n,
        inputs=[keys_ab, labels_a, labels_b, keys_ba, unequal],
        device=device,
    )
    _launch.radix_sort_pairs(keys_ba, order_ba, count=n)
    _launch.launch(kernel_equality.check_pairs, dim=n, inputs=[keys_ba, unequal], device=device)
    # The answer is a host ``bool``.
    return int(read_scalar(unequal, 0)) == 0


def _require_triangles(**named: wp.array[wp.int32]) -> None:
    """Raise ``ValueError`` unless every named buffer holds a whole number of triangles."""
    for name, faces in named.items():
        if faces.size % 3:
            raise ValueError(
                f"{name} must hold a whole number of triangles, got {faces.size} indices"
            )


def _index_bound(faces: wp.array[wp.int32]) -> int:
    """One past the largest index of the non-empty ``faces``, which must hold no negative one."""
    # A reduction with a host readback: the bound sizes the sort keys.
    min_index, max_index = od.reduce.minmax(faces)
    if min_index < 0:
        raise ValueError(f"faces must be non-negative, got a minimum of {min_index}")
    return max_index + 1
