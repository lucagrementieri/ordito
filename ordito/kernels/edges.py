import warp as wp

from ordito.kernels.adjacency import edge_endpoints, write_edge_row, write_face_edge_keys
from ordito.kernels.array import sorted_run_start, unpack_edge_key
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.predicates import segment_aabb, side_lengths
from ordito.kernels.reduce import block_chunk_1d, block_sum, term_sum_1d

wp.set_module_options({"enable_backward": False})


@wp.func
def _write_edge(
    out_edges: wp.array2d[wp.int32], row: wp.int32, a: wp.int32, b: wp.int32, sort: wp.bool
):
    if sort:
        out_edges[row, 0] = wp.min(a, b)
        out_edges[row, 1] = wp.max(a, b)
    else:
        out_edges[row, 0] = a
        out_edges[row, 1] = b


@wp.kernel
def faces_to_edges(
    faces: wp.array[wp.int32], sort: wp.bool, out_edges: wp.array2d[wp.int32]
) -> None:
    # Three directed edges per face; ``sort`` puts the smaller vertex index first per row.
    tid = wp.int32(wp.tid())
    f = tid * 3
    i0 = faces[f + 0]
    i1 = faces[f + 1]
    i2 = faces[f + 2]
    _write_edge(out_edges, f, i0, i1, sort)
    _write_edge(out_edges, f + 1, i1, i2, sort)
    _write_edge(out_edges, f + 2, i2, i0, sort)


@wp.kernel
def emit_sorted_unique_edges_from_keys(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    base: wp.uint64,
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
) -> None:
    # ``emit_sorted_unique_edges`` for ``edges.edges_unique(edges_sorted=)``, whose rows were
    # packed by ``grouping.pack_indices_and_order`` and so have no face buffer to read a row from:
    # a run's first position writes its row by unpacking its key instead. ``pack_indices`` packs a
    # sorted ``(lo, hi)`` row as ``lo + hi * base``, which ``array.unpack_edge_key`` inverts
    # exactly, so the unique rows equal those a gather of each run's first row would give.
    i = wp.int32(wp.tid())
    e = ranks[i] - 1
    out_inverse[order[i]] = e
    if sorted_run_start(sorted_keys, i):
        lo, hi = unpack_edge_key(sorted_keys[i], base)
        out_unique_edges[e, 0] = lo
        out_unique_edges[e, 1] = hi


# Codes ``face_edge_keys_checked`` raises its range flag to; the larger wins, so a negative index is
# reported ahead of one past the bound, the order the reduction-based check tests them in.
INDEX_ABOVE_BOUND = wp.int32(1)
INDEX_NEGATIVE = wp.int32(2)


@wp.func
def face_index_range_code(faces: wp.array[wp.int32], f: wp.int32, bound: wp.int32) -> wp.int32:
    # ``0`` when face ``f``'s three corners lie in ``[0, bound)``, else the worse of
    # ``INDEX_NEGATIVE`` / ``INDEX_ABOVE_BOUND``.
    code = wp.int32(0)
    for k in range(3):
        v = faces[3 * f + k]
        if v < 0:
            code = INDEX_NEGATIVE
        elif v >= bound:
            code = wp.max(code, INDEX_ABOVE_BOUND)
    return code


@wp.kernel
def face_edge_keys_checked(
    faces: wp.array[wp.int32],
    base: wp.uint64,
    bound: wp.int32,
    flag_slot: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # ``adjacency.face_edge_keys_and_order`` plus the index range check ``edges.edges_unique``
    # would otherwise run as a ``minmax`` reduction and its own readback: a face with a corner
    # outside ``[0, bound)`` raises ``out_order[flag_slot]`` (a zeroed slot past the sort's
    # double-width buffer) to its ``face_index_range_code``, and the wrapper reads that slot in the
    # same copy as the scan's tail. Keys of a bad face are garbage, which only reorders the sort
    # the wrapper then discards. The identity payload is the only other statement the two share.
    f = wp.int32(wp.tid())
    write_face_edge_keys(faces, f, 3 * f, base, out_keys)
    for k in range(3):
        out_order[3 * f + k] = 3 * f + k
    code = face_index_range_code(faces, f, bound)
    if code > 0:
        wp.atomic_max(out_order, flag_slot, code)


@wp.kernel
def emit_sorted_unique_edges(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
) -> None:
    # ``edges.edges_unique``'s two returns from a sort of every corner's edge key: ``ranks`` is the
    # inclusive scan of ``grouping.mark_sorted_run_starts``, so ``ranks[i] - 1`` is the
    # ascending-key unique index ``grouping.unique_1d`` assigns the key at sorted position ``i``,
    # and a run's first position writes its row from the corner that sorted there. The run start
    # is re-derived from the keys rather than read from the marks, so the marks may be scanned in
    # place.
    # ``remesh._FlipTopology.edges_unique`` launches it over the flip loop's own sort;
    # ``kernels/remesh.emit_pass_edges`` is the decimation pass's form: live corners and a capacity
    # bound, a chunked exclusive scan in place of the inclusive one, and the incidence and
    # adjacency slots in the same launch.
    i = wp.int32(wp.tid())
    corner = order[i]
    e = ranks[i] - 1
    out_inverse[corner] = e
    if sorted_run_start(sorted_keys, i):
        write_edge_row(faces, corner, e, out_unique_edges)


@wp.func
def edge_row_length(vertices: wp.array[wp.vec3], lo: wp.int32, hi: wp.int32) -> wp.float32:
    # Length of the edge row ``(lo, hi)``, second endpoint minus first: the one spelling the edge
    # table's lengths (``edge_lengths``), the unique edges' lengths emitted off the sort
    # (``emit_sorted_unique_edge_lengths``) and their mean's partials share, so the three agree bit
    # for bit.
    return wp.length(vertices[hi] - vertices[lo])


@wp.kernel
def emit_sorted_unique_edge_lengths(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_lengths: wp.array[wp.float32],
) -> None:
    # ``edges.edges_unique_length`` straight off the sort ``emit_sorted_unique_edges`` reads: a
    # run's first position writes the length of the row that kernel would write there, so neither
    # the ``(m, 2)`` table nor the corner inverse is materialized only for ``edge_lengths`` to read
    # the table back. Differs from ``emit_sorted_unique_edges`` only in what it writes.
    i = wp.int32(wp.tid())
    if sorted_run_start(sorted_keys, i):
        lo, hi = edge_endpoints(faces, order[i])
        out_lengths[ranks[i] - 1] = edge_row_length(vertices, lo, hi)


@wp.kernel
def edge_lengths(
    vertices: wp.array[wp.vec3], edges: wp.array2d[wp.int32], out_lengths: wp.array[wp.float32]
) -> None:
    i = wp.int32(wp.tid())
    out_lengths[i] = edge_row_length(vertices, edges[i, 0], edges[i, 1])


@wp.func
def halfedge_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], h: wp.int32
) -> wp.float32:
    # Length of halfedge ``h``, tip minus tail: the one spelling ``halfedge_lengths`` writes and
    # ``halfedge_length_term`` folds, so the stored lengths and their mean cannot drift apart.
    tail, tip = halfedge_endpoints(faces, h)
    return wp.length(vertices[tip] - vertices[tail])


@wp.kernel
def halfedge_lengths(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_lengths: wp.array[wp.float32]
) -> None:
    # ``edge_lengths`` over the ``faces_to_edges`` rows, read straight off the faces: row ``h`` of
    # that table is halfedge ``h``, so the ``(3 * n_faces, 2)`` table is never written only to be
    # gathered through once.
    h = wp.int32(wp.tid())
    out_lengths[h] = halfedge_length(vertices, faces, h)


@wp.struct
class HalfedgeLengthTerms:
    """``edges.edge_length_sum``'s halfedge count and the mesh its lengths are read off."""

    n: wp.int32
    vertices: wp.array[wp.vec3]
    faces: wp.array[wp.int32]


@wp.func
def halfedge_length_term(
    local: wp.int32, offset: wp.int32, terms: HalfedgeLengthTerms
) -> wp.float32:
    # ``halfedge_length`` of halfedge ``offset + local``, the term ``HALFEDGE_LENGTH_SUM`` folds on
    # the fly, so the ``(3 * n_faces,)`` length buffer is never written.
    return halfedge_length(terms.vertices, terms.faces, offset + local)


# ``edges.edge_length_sum``'s ``kernels/reduce.TermSum``: ``reduce.sum`` of the halfedge lengths in
# either determinism setting, one launch while it is off.
HALFEDGE_LENGTH_SUM = term_sum_1d(halfedge_length_term, HalfedgeLengthTerms, "halfedge_length_sum")


@wp.kernel
def unique_edge_length_partials(
    vertices: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.uint64],
    n_keys: wp.int32,
    base: wp.uint64,
    flags: wp.array[wp.int32],
    flag_slot: wp.int32,
    out_partials: wp.array[wp.vec3],
) -> None:
    # ``(length sum, edge count, range flag)`` of the unique edges, one stored triple per block for
    # ``reduce.sum``'s fixed-order tree, straight off the sorted corner keys: a run start is one
    # unique edge and its key unpacks to the endpoints, so ``edges.mean_unique_edge_length`` needs
    # no run scan, no emitted edge table, no length buffer and no edge-count readback. Block 0 adds
    # ``flags[flag_slot]`` (``face_edge_keys_checked``'s range code, or a zero slot) as the third
    # component, so the one readback of the total also carries the validation.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(n_keys, block)
    # A raised flag means some key unpacks to an index outside ``vertices``: read none of them.
    if flags[flag_slot] != 0:
        count = 0
    acc = wp.vec3(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        if sorted_run_start(sorted_keys, i):
            lo, hi = unpack_edge_key(sorted_keys[i], base)
            acc += wp.vec3(edge_row_length(vertices, lo, hi), 1.0, 0.0)
    if block == 0 and lane == 0:
        acc += wp.vec3(0.0, 0.0, wp.float32(flags[flag_slot]))
    block_total = block_sum(acc)
    if lane == 0:
        out_partials[block] = block_total


@wp.kernel
def edge_aabb_bounds(
    vertices: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    # Per-edge AABB, the input a segment BVH is built from -- the edge counterpart of
    # ``triangles.face_aabb_bounds``, and the one spelling of it: this lived twice, as
    # ``curvature.edge_aabb_from_endpoints`` (for the mean-curvature ball measure over the
    # face-adjacency edges) and ``proximity.edge_bounds`` (for closest-point-on-edges), which is
    # what a per-edge geometric quantity parked in two algorithm modules looks like.
    e = wp.int32(wp.tid())
    lower, upper = segment_aabb(vertices[edges[e, 0]], vertices[edges[e, 1]])
    out_lower[e] = lower
    out_upper[e] = upper


@wp.kernel
def face_edge_lengths(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_lengths: wp.array2d[wp.float32]
) -> None:
    # Column ``e`` is the edge *opposite* corner ``e``, the igl intrinsic convention that
    # ``laplacian.cotmatrix_entries_intrinsic`` reads.
    f = wp.int32(wp.tid())
    l0, l1, l2 = side_lengths(
        vertices[faces[f * 3 + 0]], vertices[faces[f * 3 + 1]], vertices[faces[f * 3 + 2]]
    )
    out_lengths[f, 0] = l0
    out_lengths[f, 1] = l1
    out_lengths[f, 2] = l2
