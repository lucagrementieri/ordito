import warp as wp

from ordito.kernels.adjacency import write_edge_row
from ordito.kernels.array import sorted_run_start, unpack_edge_key
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.predicates import segment_aabb, side_lengths


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
def edges_from_keys(
    keys: wp.array[wp.uint64], base: wp.uint64, out_edges: wp.array2d[wp.int32]
) -> None:
    # ``grouping.hash_indices_rows`` packs a sorted ``(lo, hi)`` row as ``lo + hi * base``, which
    # ``array.unpack_edge_key`` inverts exactly -- the docstring there names this packing as the
    # shared convention. So the deduplicated *rows* need not be recovered by gathering the first
    # corner that produced each key: they are already in the key. That replaces a
    # ``first_occurrence_indices`` scatter plus an ``array.gather`` with one launch, and drops the
    # first-occurrence buffer with them.
    i = wp.int32(wp.tid())
    lo, hi = unpack_edge_key(keys[i], base)
    out_edges[i, 0] = lo
    out_edges[i, 1] = hi


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


@wp.kernel
def edge_lengths(
    vertices: wp.array[wp.vec3], edges: wp.array2d[wp.int32], out_lengths: wp.array[wp.float32]
) -> None:
    i = wp.int32(wp.tid())
    out_lengths[i] = wp.length(vertices[edges[i, 1]] - vertices[edges[i, 0]])


@wp.kernel
def halfedge_lengths(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_lengths: wp.array[wp.float32]
) -> None:
    # ``edge_lengths`` over the ``faces_to_edges`` rows, read straight off the faces: row ``h`` of
    # that table is halfedge ``h``, so the ``(3 * n_faces, 2)`` table is never written only to be
    # gathered through once.
    h = wp.int32(wp.tid())
    tail, tip = halfedge_endpoints(faces, h)
    out_lengths[h] = wp.length(vertices[tip] - vertices[tail])


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
