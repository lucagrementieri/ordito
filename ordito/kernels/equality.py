import warp as wp

from ordito.kernels.array import pack_directed_key, pack_index_triple, rotate_min_first
from ordito.kernels.triangles import corner_triple

wp.set_module_options({"enable_backward": False})


@wp.func
def mapped_corner(
    vertex_map: wp.array[wp.int32], index: wp.int32, n_vertices: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # A corner index read through ``vertex_map`` (the identity when it is ``None``), with ``1`` in
    # the second slot when the index is outside ``[0, n_vertices)``. Through a map such a corner is
    # read as ``0``, so the gather stays inside the map; the identity keeps it as it is. Either way
    # a caller holding a flag answers ``False`` whatever key the face then packs.
    bad = wp.where(index < 0 or index >= n_vertices, wp.int32(1), wp.int32(0))
    if vertex_map.shape[0] == 0:
        return index, bad
    if bad != 0:
        return wp.int32(0), bad
    return vertex_map[index], bad


@wp.func
def canonical_corner_triple(
    faces: wp.array[wp.int32], vertex_map: wp.array[wp.int32], f: wp.int32, n_vertices: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # Face ``f``'s corners mapped through ``vertex_map`` and rotated to start at the smallest, which
    # keeps the winding; the fourth slot is non-zero when a corner was out of range. Shared by the
    # key pass and the emit, which must write exactly the row the key ordered.
    i0, i1, i2 = corner_triple(faces, f)
    a, bad_a = mapped_corner(vertex_map, i0, n_vertices)
    b, bad_b = mapped_corner(vertex_map, i1, n_vertices)
    c, bad_c = mapped_corner(vertex_map, i2, n_vertices)
    r0, r1, r2 = rotate_min_first(a, b, c)
    return r0, r1, r2, bad_a + bad_b + bad_c


@wp.kernel
def canonical_face_keys(
    faces: wp.array[wp.int32],
    vertex_map: wp.array[wp.int32],
    n_vertices: wp.int32,
    whole_row: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_invalid: wp.array[wp.int32],
) -> None:
    # The sort key of face ``f``'s canonical row, with the identity payload, into the leading halves
    # of the radix sort's double-width buffers. ``whole_row`` (warp-uniform) packs the three corners
    # into one key, which fits while ``n_vertices ** 3 <= 2 ** 64``; otherwise the key is the last
    # two corners only and ``leading_corner_keys`` orders the first in a second, stable pass.
    # ``out_invalid`` (``None`` when the caller trusts its faces) is raised by an out-of-range
    # corner: a plain store, since every writer stores the same value.
    f = wp.int32(wp.tid())
    a, b, c, bad = canonical_corner_triple(faces, vertex_map, f, n_vertices)
    if bad != 0 and out_invalid.shape[0] > 0:
        out_invalid[0] = wp.int32(1)
    base = wp.uint64(n_vertices)
    if whole_row != 0:
        out_keys[f] = pack_index_triple(c, b, a, base)
    else:
        out_keys[f] = pack_directed_key(c, b, wp.uint64(1) << wp.uint64(32))
    out_order[f] = f


@wp.kernel
def leading_corner_keys(
    faces: wp.array[wp.int32],
    vertex_map: wp.array[wp.int32],
    n_vertices: wp.int32,
    order: wp.array[wp.int32],
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # The second pass of a two-pass lexicographic sort: the leading corner of the face at sorted
    # position ``i`` as the key, and that face as the payload, into the second sort's buffers.
    i = wp.int32(wp.tid())
    f = order[i]
    a, _b, _c, _bad = canonical_corner_triple(faces, vertex_map, f, n_vertices)
    out_keys[i] = a
    out_order[i] = f


@wp.kernel
def emit_canonical_faces(
    faces: wp.array[wp.int32],
    vertex_map: wp.array[wp.int32],
    n_vertices: wp.int32,
    order: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # Row ``i`` of the canonical buffer: the face at sorted position ``i``, rotated as it was keyed.
    i = wp.int32(wp.tid())
    a, b, c, _bad = canonical_corner_triple(faces, vertex_map, order[i], n_vertices)
    base = i * wp.int32(3)
    out_faces[base] = a
    out_faces[base + wp.int32(1)] = b
    out_faces[base + wp.int32(2)] = c


@wp.kernel
def flag_unequal(
    a: wp.array[wp.int32], b: wp.array[wp.int32], out_unequal: wp.array[wp.int32]
) -> None:
    # Raise ``out_unequal[0]`` where the two buffers differ. A plain store: every writer stores the
    # same value, and only a mismatch writes at all.
    i = wp.int32(wp.tid())
    if a[i] != b[i]:
        out_unequal[0] = wp.int32(1)


@wp.kernel
def flag_unmatched_vertices(
    vertices_a: wp.array[wp.vec3],
    vertices_b: wp.array[wp.vec3],
    match: wp.array[wp.int32],
    rtol: wp.float32,
    atol: wp.float32,
    hits: wp.array[wp.int32],
    out_unequal: wp.array[wp.int32],
) -> None:
    # Vertex ``v`` of ``a`` is matched when its nearest ``b`` vertex lies within
    # ``atol + rtol * |b|`` and no other ``a`` vertex took it: the integer ticket on ``hits`` makes
    # the match a bijection, whatever order the threads arrive in.
    v = wp.int32(wp.tid())
    m = match[v]
    if m < 0:
        out_unequal[0] = wp.int32(1)
        return
    target = vertices_b[m]
    if wp.length(vertices_a[v] - target) > atol + rtol * wp.length(target):
        out_unequal[0] = wp.int32(1)
        return
    if wp.atomic_add(hits, m, wp.int32(1)) != wp.int32(0):
        out_unequal[0] = wp.int32(1)


@wp.kernel
def column_keys(
    rows: wp.array2d[wp.int32],
    column: wp.int32,
    order: wp.array[wp.int32],
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # One least-significant-digit pass of a lexicographic row sort: the ``column`` entry of the row
    # at position ``i`` of the order so far (the identity when ``order`` is ``None``), with that row
    # as the payload, into the leading halves of the radix sort's double-width buffers.
    i = wp.int32(wp.tid())
    row = i
    if order.shape[0] > 0:
        row = order[i]
    out_keys[i] = rows[row, column]
    out_order[i] = row


@wp.kernel
def flag_unequal_ordered_rows(
    rows_a: wp.array2d[wp.int32],
    order_a: wp.array[wp.int32],
    rows_b: wp.array2d[wp.int32],
    order_b: wp.array[wp.int32],
    out_unequal: wp.array[wp.int32],
) -> None:
    # Raise ``out_unequal[0]`` where the two tables, each read in its own sorted order, differ.
    i, j = wp.tid()
    if rows_a[order_a[i], j] != rows_b[order_b[i], j]:
        out_unequal[0] = wp.int32(1)


@wp.kernel
def label_pair_keys(
    labels_a: wp.array[wp.int32], labels_b: wp.array[wp.int32], out_keys: wp.array[wp.uint64]
) -> None:
    # Item ``i``'s two labels packed into one key, injective over all ``int32`` pairs.
    i = wp.int32(wp.tid())
    out_keys[i] = pack_directed_key(labels_b[i], labels_a[i], wp.uint64(1) << wp.uint64(32))
