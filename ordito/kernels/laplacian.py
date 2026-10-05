from typing import Any

import warp as wp

from ordito.kernels.array import OverloadTable, csr_key, csr_run_start, sorted_run_end
from ordito.kernels.halfedge import halfedge_destination
from ordito.kernels.predicates import doublearea_from_lengths, squared_edge_lengths
from ordito.kernels.triangles import face_vertices, row_triple

wp.set_module_options({"enable_backward": False})

# ``cot_entries_from_l2`` and ``cot_entries_from_edge_lengths`` stay here rather than joining
# ``squared_edge_lengths`` / ``doublearea_from_lengths`` in ``kernels/predicates.py``: a cotangent
# weight is not a general triangle quantity, it is this operator's own entry, and the zero-area
# reasoning below is numerical defence *of the Laplacian* that belongs beside the thing it defends.


@wp.func
def cot_entries_from_l2(
    l2_0: wp.float32, l2_1: wp.float32, l2_2: wp.float32, dbl_area: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    # A zero-area triangle contributes nothing rather than an infinity. Its angles are 0 or pi, so
    # it has no finite cotangent, and ``doublearea_from_lengths`` deliberately reports 0.0 for one:
    # without this guard that 0 divides straight through to +-inf, and a *single* collapsed face
    # poisons the whole assembled operator -- and every solve against it -- with NaN.
    #
    # The test is against exact zero, not a tolerance. ``dbl_area`` is already clamped
    # non-negative, so this changes results only where they used to be non-finite; a merely
    # sliver triangle still yields its (huge, finite) weight, because that is ill-conditioning
    # rather than a division by zero and the fix for it is mollification -- see
    # ``laplacian.robust_laplacian``.
    denominator = wp.float32(4.0) * dbl_area
    if denominator <= wp.float32(0.0):
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    c0 = (l2_1 + l2_2 - l2_0) / denominator
    c1 = (l2_2 + l2_0 - l2_1) / denominator
    c2 = (l2_0 + l2_1 - l2_2) / denominator
    return c0, c1, c2


@wp.func
def cot_entries_from_edge_lengths(
    l0: wp.float32, l1: wp.float32, l2: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    l2_0 = l0 * l0
    l2_1 = l1 * l1
    l2_2 = l2 * l2
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    return cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)


@wp.func
def face_half_cotangents(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    # Face ``f``'s three half-cotangent weights, column ``e`` for the edge opposite corner ``e``:
    # one row of ``cotmatrix_entries``' table, for a kernel that needs a few faces' weights and not
    # the whole mesh's (``smoothing.band_dirichlet_values``).
    v0, v1, v2 = face_vertices(vertices, faces, f)
    l2_0, l2_1, l2_2 = squared_edge_lengths(v0, v1, v2)
    l0 = wp.sqrt(l2_0)
    l1 = wp.sqrt(l2_1)
    l2 = wp.sqrt(l2_2)
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    return cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)


@wp.kernel
def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cot: wp.array2d[wp.Float]
) -> None:
    # ``out_cot`` is generic: the half-cotangent weights are computed in float32 (the vertex
    # precision) and cast to the requested output dtype (float32 or float64) at store time.
    f = wp.int32(wp.tid())
    c0, c1, c2 = face_half_cotangents(vertices, faces, f)
    out_cot[f, 0] = type(out_cot[f, 0])(c0)
    out_cot[f, 1] = type(out_cot[f, 1])(c1)
    out_cot[f, 2] = type(out_cot[f, 2])(c2)


@wp.kernel
def cotmatrix_entries_intrinsic(
    edge_lengths: wp.array2d[wp.float32], out_cot: wp.array2d[wp.Float]
) -> None:
    f = wp.int32(wp.tid())
    l0, l1, l2 = row_triple(edge_lengths, f)
    c0, c1, c2 = cot_entries_from_edge_lengths(l0, l1, l2)
    out_cot[f, 0] = type(out_cot[f, 0])(c0)
    out_cot[f, 1] = type(out_cot[f, 1])(c1)
    out_cot[f, 2] = type(out_cot[f, 2])(c2)


@wp.func
def edge_weight(
    a: wp.int32, b: wp.int32, vertices: wp.array[wp.vec3], equal_weight: wp.int32
) -> wp.float32:
    if equal_weight != 0:
        return wp.float32(1.0)
    return wp.float32(1.0) / (wp.length(vertices[a] - vertices[b]) + wp.float32(1.0e-12))


@wp.kernel
def laplacian_triplets_directed(
    edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # One triplet per directed triangle edge, matching trimesh's ``mesh.edges`` adjacency.
    # ``out_vals`` is generic: the float32 edge weight is cast to the requested output dtype.
    e = wp.int32(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    out_rows[e] = a
    out_cols[e] = b
    out_vals[e] = type(out_vals[e])(edge_weight(a, b, vertices, equal_weight))


@wp.kernel
def laplacian_triplets_symmetric(
    edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # Emits both directed pairs (a, b) and (b, a) from each unique undirected edge so the
    # adjacency is symmetric, matching trimesh's ``vertex_neighbors``. Duplicate multiplicity
    # cancels under row-normalization. ``out_vals`` is generic (float32 or float64).
    e = wp.int32(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    w = type(out_vals[e * 2])(edge_weight(a, b, vertices, equal_weight))
    base = e * 2
    out_rows[base + 0] = a
    out_cols[base + 0] = b
    out_vals[base + 0] = w
    out_rows[base + 1] = b
    out_cols[base + 1] = a
    out_vals[base + 1] = w


@wp.kernel
def row_normalize(offsets: wp.array[wp.int32], out_values: wp.array[wp.Float]) -> None:
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    # ``out_values[0]`` is always valid: the launcher only runs this kernel when nnz > 0. It is
    # read solely to source the generic scalar type for the accumulator / zero literals.
    total = type(out_values[0])(0.0)
    for k in range(start, end):
        total += out_values[k]
    if total > type(out_values[0])(0.0):
        for k in range(start, end):
            out_values[k] = out_values[k] / total


# One row of the row-stochastic averaging operator, as a ``@wp.func`` so a consumer can apply it
# and use the result in the same thread instead of round-tripping an intermediate buffer through
# global memory and a second launch. Every smoothing filter that iterates ``L`` does exactly that
# (see ``ordito/smoothing.py``), so the row apply is the shared run rather than the kernel.
#
# Note the precision: the float32 weight is promoted to float64 because the accumulator is a
# ``wp.vec3d``. ``kernels/smoothing.diffuse_scalar_pass`` walks the same row on a float32 scalar
# field and accumulates in float32, so it cannot call this -- ``float64 * float32`` does not parse,
# and the float64-field form that would let one generic serve both was measured as a loss. That
# comment carries the reasoning.


@wp.func
def operator_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[Any],
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of the operator applied to a ``wp.vec3d`` field -- or a ``wp.vec3`` one, widened
    # exactly as each entry is read, so a caller holding ``float32`` positions gets the same sum as
    # from their ``float64`` copy without materializing it.
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        # Isolated vertex (empty row): the averaging operator acts as the identity so the
        # vertex does not drift toward the origin.
        return wp.vec3d(field[i])
    acc = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(start, end):
        w = wp.float64(values[k])
        acc += w * wp.vec3d(field[columns[k]])
    return acc


@wp.kernel
def mesh_operator_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    diagonal: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The sparsity of every vertex operator on a triangle mesh, as ``kernels/array.csr_key`` keys,
    # for the *directed* pattern build (small meshes): for each face and each corner ``e``, the
    # edge opposite it both ways -- slots ``6 * f + 2 * e`` and ``+ 1`` hold ``(i, j)`` and
    # ``(j, i)``, where ``i`` is corner ``e + 1`` and ``j`` corner ``e + 2`` -- and one diagonal
    # slot per vertex in the tail ``[6 * n_faces, 6 * n_faces + n_vertices)``. ``diagonal`` chooses
    # what the tail holds: nothing (0: the pattern has no diagonal slots), the diagonal of every
    # vertex a face references (1), or of every vertex (2). For 1 the caller fills the tail with the
    # sentinel first, so a vertex no face references keeps an empty row; every face writes the same
    # key into a shared vertex's slot, so the race is benign. For 2 the launch covers
    # ``max(n_faces, n_vertices)`` threads and thread ``v`` writes vertex ``v``'s key. Both
    # directions carry the corner slot ``3 * f + e`` as payload -- the convention
    # ``mesh_edge_keys`` shares, so one value kernel serves both builds; it reads the direction off
    # the row. A diagonal slot's payload is never read, so it is not written. A degenerate face's
    # self-edge keeps its keys and lands on the diagonal entry; each value kernel decides what that
    # means.
    t = wp.int32(wp.tid())
    n_faces = faces.shape[0] // 3
    tail = 6 * n_faces
    if diagonal == 2 and t < n_vertices:
        out_keys[tail + t] = csr_key(t, t, n_vertices, n_vertices)
    if t >= n_faces:
        return
    for e in range(3):
        i = faces[t * 3 + (e + 1) % 3]
        j = faces[t * 3 + (e + 2) % 3]
        out_keys[t * 6 + e * 2] = csr_key(i, j, n_vertices, n_vertices)
        out_keys[t * 6 + e * 2 + 1] = csr_key(j, i, n_vertices, n_vertices)
        out_order[t * 6 + e * 2] = t * 3 + e
        out_order[t * 6 + e * 2 + 1] = t * 3 + e
        # Range-checked at the write (CLAUDE.md 12.1): an index past the vertex count must not
        # become a store past the tail.
        vertex = faces[t * 3 + e]
        if diagonal == 1 and vertex >= 0 and vertex < n_vertices:
            out_keys[tail + vertex] = csr_key(vertex, vertex, n_vertices, n_vertices)


@wp.kernel
def mesh_halfedge_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # One key per halfedge ``faces[h] -> faces[next(h)]``, the directed adjacency trimesh's
    # ``edges_to_coo(mesh.edges)`` builds -- a self-edge included, as an entry on the diagonal.
    # The payload is the halfedge index.
    h = wp.int32(wp.tid())
    out_keys[h] = csr_key(faces[h], halfedge_destination(faces, h), n_vertices, n_vertices)
    out_order[h] = h


# The *undirected* pattern build (large meshes) sorts one key per corner instead of two plus one per
# vertex: a run of equal ``(min, max)`` keys is one unique edge, and in that order every row's
# upper half -- the edges whose smaller endpoint it is -- is contiguous and sorted by column. The
# lower half is the transpose, laid out by a second, stable sort of the unique edges by their larger
# endpoint over 32-bit keys. Each row is ``[lower | diagonal | upper]``, and an entry's value comes
# from its edge's run in the first sort, so both builds hand the value kernels the same thing.


@wp.kernel
def mesh_edge_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    mark: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_referenced: wp.array[wp.int32],
) -> None:
    # Corner ``e``'s opposite edge as an undirected ``(min, max)`` key with the corner slot
    # ``3 * f + e`` as payload. A degenerate face's self-edge never gets a key; instead ``mark``
    # chooses which vertices ``out_referenced`` flags for a diagonal slot: every in-range corner
    # (1), the endpoint of a self-edge (2) -- the entry the directed build keeps for it -- or every
    # vertex (3), for which the launch covers ``max(n_faces, n_vertices)`` threads and thread ``v``
    # flags vertex ``v``.
    t = wp.int32(wp.tid())
    if mark == 3 and t < n_vertices:
        out_referenced[t] = 1
    if t >= faces.shape[0] // 3:
        return
    for e in range(3):
        i = faces[t * 3 + (e + 1) % 3]
        j = faces[t * 3 + (e + 2) % 3]
        key = csr_key(wp.min(i, j), wp.max(i, j), n_vertices, n_vertices)
        if i == j:
            key = wp.uint64(n_vertices) * wp.uint64(n_vertices)
            if mark == 2 and i >= 0 and i < n_vertices:
                out_referenced[i] = 1
        out_keys[t * 3 + e] = key
        out_order[t * 3 + e] = t * 3 + e
        vertex = faces[t * 3 + e]
        if mark == 1 and vertex >= 0 and vertex < n_vertices:
            out_referenced[vertex] = 1


@wp.kernel
def mesh_edge_runs(
    keys: wp.array[wp.uint64],
    sentinel: wp.uint64,
    n_vertices: wp.int32,
    out_flags: wp.array[wp.int32],
    out_upper: wp.array[wp.int32],
    out_lower: wp.array[wp.int32],
    out_second_keys: wp.array[wp.int32],
    out_second_order: wp.array[wp.int32],
) -> None:
    # Over the sorted corner keys: each run start is one unique edge ``(lo, hi)``. Flag it (the
    # flags' scan ranks the edges), count it into ``lo``'s upper half and ``hi``'s lower half, and
    # key it by ``hi`` for the second sort, whose payload is its position here.
    i = wp.int32(wp.tid())
    flag = 0
    second = n_vertices
    if csr_run_start(keys, i, sentinel):
        lo = wp.int32(keys[i] // wp.uint64(n_vertices))
        hi = wp.int32(keys[i] % wp.uint64(n_vertices))
        wp.atomic_add(out_upper, lo, 1)
        wp.atomic_add(out_lower, hi, 1)
        flag = 1
        second = hi
    out_flags[i] = flag
    out_second_keys[i] = second
    out_second_order[i] = i


@wp.func
def mesh_row_start(before: wp.array2d[wp.int32], r: wp.int32) -> wp.int32:
    # Row ``r``'s first entry in the undirected build's ``[lower | diagonal | upper]`` rows.
    # ``before`` is the exclusive scan of the ``(3, n_vertices)`` tallies (upper-half count,
    # lower-half count, diagonal flag) taken as one flat array, so its second and third rows carry
    # the whole upper total and the upper-plus-lower total ahead of them; ``before[1, 0]`` and
    # ``before[2, 0]`` are exactly those totals, and removing them leaves each part's own prefix.
    return before[0, r] + (before[1, r] - before[1, 0]) + (before[2, r] - before[2, 0])


@wp.func
def mesh_place_upper(
    keys: wp.array[wp.uint64],
    inclusive: wp.array[wp.int32],
    sentinel: wp.uint64,
    n_vertices: wp.int32,
    tallies: wp.array2d[wp.int32],
    before: wp.array2d[wp.int32],
    i: wp.int32,
    out_columns: wp.array[wp.int32],
    out_run_start: wp.array[wp.int32],
) -> None:
    # Edge ``(lo, hi)`` of sorted key ``i``, in row ``lo``: after the lower half and the diagonal,
    # at its rank among the edges whose smaller endpoint is ``lo`` -- its global rank less the edges
    # of earlier rows.
    if not csr_run_start(keys, i, sentinel):
        return
    lo = wp.int32(keys[i] // wp.uint64(n_vertices))
    rank = inclusive[i] - 1 - before[0, lo]
    slot = mesh_row_start(before, lo) + tallies[1, lo] + tallies[2, lo] + rank
    out_columns[slot] = wp.int32(keys[i] % wp.uint64(n_vertices))
    out_run_start[slot] = i


@wp.kernel
def mesh_place_columns(
    keys: wp.array[wp.uint64],
    inclusive: wp.array[wp.int32],
    sentinel: wp.uint64,
    second_keys: wp.array[wp.int32],
    second_order: wp.array[wp.int32],
    count: wp.int32,
    n_vertices: wp.int32,
    tallies: wp.array2d[wp.int32],
    before: wp.array2d[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_run_start: wp.array[wp.int32],
) -> None:
    # Launched over ``max(count, n_vertices)``; every write lands in a slot no other thread writes.
    # As a vertex: its row's offset (the last also closes the total) and its diagonal, which has no
    # contributors. As a position of the first sort: its upper-half entry (``mesh_place_upper``).
    # As a sorted position of the second sort: edge ``(lo, hi)`` in row ``hi``, at its rank among
    # the edges whose larger endpoint is ``hi`` -- stable, so in ascending ``lo``.
    t = wp.int32(wp.tid())
    if t < n_vertices:
        start = mesh_row_start(before, t)
        out_offsets[t] = start
        if t == n_vertices - 1:
            out_offsets[n_vertices] = start + tallies[0, t] + tallies[1, t] + tallies[2, t]
        if tallies[2, t] != 0:
            slot = start + tallies[1, t]
            out_columns[slot] = t
            out_run_start[slot] = 0
    if t < count:
        mesh_place_upper(
            keys, inclusive, sentinel, n_vertices, tallies, before, t, out_columns, out_run_start
        )
        hi = second_keys[t]
        if hi < n_vertices:
            i = second_order[t]
            slot = mesh_row_start(before, hi) + t - (before[1, hi] - before[1, 0])
            out_columns[slot] = wp.int32(keys[i] // wp.uint64(n_vertices))
            out_run_start[slot] = i


@wp.kernel
def cotmatrix_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    out_values: wp.array[wp.Float],
) -> None:
    # One thread per row of a mesh operator pattern (``laplacian._mesh_operator_pattern``). An
    # off-diagonal is the sum of its contributing half-cotangents -- the corner slots ``order[p]``
    # over its run of equal sorted ``keys`` from ``run_start[k]``, in face order since the sort is
    # stable -- and the
    # diagonal is minus the row's off-diagonal sum, in column order, so every row sums to zero up to
    # that one sum's rounding. A degenerate face's self-edge never reaches a row: its four
    # contributions ``w + w - w - w`` cancelled exactly in the 12-triplet form too. The
    # half-cotangents may be float32 or float64 whatever the matrix precision; each is cast once.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    total = type(out_values[start])(0.0)
    slot = wp.int32(-1)
    for k in range(start, end):
        if columns[k] == i:
            slot = k
            continue
        value = type(out_values[k])(0.0)
        for p in range(run_start[k], sorted_run_end(keys, run_start[k], count)):
            q = order[p]
            value += type(out_values[k])(cot_entries[q // 3, q % 3])
        out_values[k] = value
        total += value
    if slot >= 0:
        out_values[slot] = -total


@wp.kernel
def laplacian_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    symmetric: wp.int32,
    out_values: wp.array[wp.Float],
) -> None:
    # The row-normalized umbrella operator over a mesh pattern with no diagonal slots. An entry's
    # weight is ``edge_weight`` of its row and column (cast once). Directed (``symmetric == 0``,
    # trimesh's ``mesh.edges``): once per halfedge in its run, so a duplicated halfedge counts
    # twice. Symmetric (trimesh's ``vertex_neighbors``): once per unique edge whatever its run --
    # but twice for a degenerate face's self-edge, whose unique ``(a, a)`` edge emits both of its
    # directions onto one entry. The row is then divided by its sum, in column order, if positive.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    total = type(out_values[start])(0.0)
    for k in range(start, end):
        c = columns[k]
        w = type(out_values[k])(edge_weight(i, c, vertices, equal_weight))
        value = type(out_values[k])(0.0)
        if symmetric != 0:
            value = w
            if c == i:
                value += w
        else:
            for _p in range(run_start[k], sorted_run_end(keys, run_start[k], count)):
                value += w
        out_values[k] = value
        total += value
    if total > type(total)(0.0):
        for k in range(start, end):
            out_values[k] = out_values[k] / total


@wp.kernel
def graph_laplacian_rows(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], out_values: wp.array[wp.Float]
) -> None:
    # ``A - diag(deg)`` over a mesh pattern with every vertex's diagonal: one per neighbour, the
    # diagonal minus their count (``igl::adjacency_matrix``'s unit weights). A degenerate face's
    # self-edge is not a neighbour.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    degree = wp.int32(0)
    slot = wp.int32(-1)
    for k in range(start, end):
        if columns[k] == i:
            slot = k
        else:
            out_values[k] = type(out_values[k])(1.0)
            degree += 1
    if slot >= 0:
        out_values[slot] = -type(out_values[slot])(degree)


@wp.func
def rotation22(angle: wp.float32) -> wp.mat22d:
    # Real 2x2 form of the unit complex number ``exp(i * angle)``: the rotation that re-expresses a
    # tangent vector in a neighbour's frame.
    c = wp.float64(wp.cos(angle))
    s = wp.float64(wp.sin(angle))
    return wp.mat22d(c, -s, s, c)


@wp.kernel
def connection_laplacian_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    transport_angles: wp.array[wp.float32],
    out_values: wp.array[wp.mat22d],
) -> None:
    # The vector Laplacian over a mesh operator pattern, one thread per row: each off-diagonal
    # weight becomes a rotation, since the two endpoints of an edge measure tangent directions from
    # different reference directions and a difference between them is only meaningful after
    # transporting one into the other's frame. Corner ``e``'s opposite halfedge ``h`` runs
    # ``i -> j``; it contributes ``-w R(-rho_h)`` to row ``i`` and ``-w R(rho_h)`` to row ``j``,
    # summed over the contributing faces in face order. The diagonal is ``sum w * I`` over the
    # row's entries in column order. Positive semi-definite (positive diagonal), unlike
    # ``cotmatrix``'s igl sign convention, because every consumer here feeds it straight to a
    # conjugate-gradient solve. A degenerate face's self-edge never reaches a row.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    weight = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(start, end):
        if columns[k] == i:
            slot = k
            continue
        value = wp.mat22d()
        for p in range(run_start[k], sorted_run_end(keys, run_start[k], count)):
            q = order[p]
            h = (q // 3) * 3 + (q % 3 + 1) % 3
            w = wp.float64(cot_entries[q // 3, q % 3])
            rho = transport_angles[h]
            value += -w * rotation22(wp.where(faces[h] == i, -rho, rho))
            weight += w
        out_values[k] = value
    if slot >= 0:
        out_values[slot] = weight * wp.mat22d(1.0, 0.0, 0.0, 1.0)


@wp.kernel
def triangle_inequality_slack(
    edge_lengths: wp.array2d[wp.float32], epsilon: wp.float32, out_slack: wp.array[wp.float32]
) -> None:
    # How far this triangle is from satisfying the strict triangle inequality with margin
    # ``epsilon``, expressed as the constant that would have to be added to all three of its edges.
    # Adding ``delta`` to all three grows each slack ``a + b - c`` by exactly ``delta`` -- the two
    # short sides gain ``2 * delta`` and the long side gives ``delta`` of it back -- so the
    # shortfall is the constant, undivided. Sharp & Crane (2020) eq. 3 states the same rule, and
    # an independent port of it (kentechx/HoleFillingPy, MIT) computes the identical quantity a
    # different way -- ``max(2 * max(L) - sum(L)) + delta``, which is ``epsilon - min_f slack_f``
    # rearranged -- also undivided.
    #
    # **``TOLERANCE_MOLLIFY = 1e-5`` sits between two arms, both of which were swept.** Below a few
    # ULP of the mean edge length the added constant does not survive the float32 store and the
    # mollification silently does nothing, which is the failure the whole function exists to
    # prevent -- so the left arm is a *storage* floor rather than a numerical-quality one, and it
    # moves with the mesh's length distribution. On the right, the perturbation this introduces into
    # rows no degenerate face touches grows linearly with epsilon and reaches the tolerance the igl
    # ``intrinsic_delaunay_cotmatrix`` parity test runs at around 1e-4. ``1e-5`` is more than a
    # decade clear of both, and a decade more conservative than that port's own 1e-4, which sits on
    # the right arm. A clean mesh yields ``delta == 0`` at every epsilon, so none of this is paid
    # where nothing is degenerate.
    f = wp.int32(wp.tid())
    a, b, c = row_triple(edge_lengths, f)
    worst = wp.max(wp.max(epsilon - (a + b - c), epsilon - (b + c - a)), epsilon - (c + a - b))
    out_slack[f] = wp.max(worst, 0.0)


@wp.func
def add_constant(length: wp.float32, delta: wp.float32) -> wp.float32:
    return length + delta


# Concrete overloads, registered at import -- see the long-form rationale in
# ``ordito/kernels/reduce.py`` and the rule in CLAUDE.md section 2.5. In short: these kernels are
# generic, Warp instantiates an overload on the first launch at each new dtype, and a module's hash
# covers the instantiated set -- so a lazily-created overload rebuilds the whole module.
#
# ``ordito.laplacian`` exposes the precision as a public ``dtype`` keyword documented as "may be
# float32 or float64", so both are reachable for every kernel here.
_MATRIX_DTYPES = (wp.float32, wp.float64)


# The concrete handles keyed by the caller's dtype -- see
# [`OverloadTable`][ordito.kernels.array.OverloadTable]; worth about a tenth of
# ``laplacian.cotmatrix`` / ``laplacian.laplacian``, from removing one generic launch each.
# ``COTMATRIX_ROWS`` keys on the pair ``(entry dtype, matrix dtype)`` because those two
# templates are independent, exactly as the registration already was.
COTMATRIX_ENTRIES: OverloadTable
COTMATRIX_ENTRIES_INTRINSIC: OverloadTable
ROW_NORMALIZE: OverloadTable
LAPLACIAN_TRIPLETS_SYMMETRIC: OverloadTable
LAPLACIAN_TRIPLETS_DIRECTED: OverloadTable
COTMATRIX_ROWS: OverloadTable
LAPLACIAN_ROWS: OverloadTable
GRAPH_LAPLACIAN_ROWS: OverloadTable
CONNECTION_LAPLACIAN_ROWS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global COTMATRIX_ENTRIES, COTMATRIX_ENTRIES_INTRINSIC, ROW_NORMALIZE
    global LAPLACIAN_TRIPLETS_SYMMETRIC, LAPLACIAN_TRIPLETS_DIRECTED
    global COTMATRIX_ROWS, CONNECTION_LAPLACIAN_ROWS, LAPLACIAN_ROWS, GRAPH_LAPLACIAN_ROWS
    COTMATRIX_ENTRIES = OverloadTable(
        cotmatrix_entries,
        {d: [wp.array[wp.vec3], wp.array[wp.int32], wp.array2d[d]] for d in _MATRIX_DTYPES},
    )
    COTMATRIX_ENTRIES_INTRINSIC = OverloadTable(
        cotmatrix_entries_intrinsic,
        {d: [wp.array2d[wp.float32], wp.array2d[d]] for d in _MATRIX_DTYPES},
    )
    ROW_NORMALIZE = OverloadTable(
        row_normalize, {d: [wp.array[wp.int32], wp.array[d]] for d in _MATRIX_DTYPES}
    )
    triplet_signature = {
        d: [
            wp.array2d[wp.int32],
            wp.array[wp.vec3],
            wp.int32,
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[d],
        ]
        for d in _MATRIX_DTYPES
    }
    LAPLACIAN_TRIPLETS_SYMMETRIC = OverloadTable(laplacian_triplets_symmetric, triplet_signature)
    LAPLACIAN_TRIPLETS_DIRECTED = OverloadTable(laplacian_triplets_directed, triplet_signature)
    # ``cot_entries`` and the matrix precision are *independent* templates: cotmatrix's
    # docstring says the entries "may be float32 or float64 regardless of dtype: the assembly
    # kernel casts them to the matrix precision", so this is a genuine 2x2, not a diagonal.
    COTMATRIX_ROWS = OverloadTable(
        cotmatrix_rows,
        {
            (entry_dtype, dtype): [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.int32],
                wp.array2d[entry_dtype],
                wp.array[dtype],
            ]
            for dtype in _MATRIX_DTYPES
            for entry_dtype in _MATRIX_DTYPES
        },
    )
    # The connection Laplacian's values are always ``wp.mat22d``; only its cotangent entries
    # follow the caller, who may pass their own in place of the float64 default.
    LAPLACIAN_ROWS = OverloadTable(
        laplacian_rows,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.vec3],
                wp.int32,
                wp.int32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    GRAPH_LAPLACIAN_ROWS = OverloadTable(
        graph_laplacian_rows,
        {d: [wp.array[wp.int32], wp.array[wp.int32], wp.array[d]] for d in _MATRIX_DTYPES},
    )
    CONNECTION_LAPLACIAN_ROWS = OverloadTable(
        connection_laplacian_rows,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array2d[d],
                wp.array[wp.float32],
                wp.array[wp.mat22d],
            ]
            for d in _MATRIX_DTYPES
        },
    )


_register_overloads()
