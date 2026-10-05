import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT
from ordito.kernels.algorithms.connected_components import ecl_hook_pair, ecl_prehook_pair
from ordito.kernels.array import pack_edge_key, scanned_count
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.predicates import vector_angle
from ordito.kernels.triangles import corner_triple

wp.set_module_options({"enable_backward": False})


@wp.func
def write_face_edge_keys(
    faces: wp.array[wp.int32],
    f: wp.int32,
    slot: wp.int32,
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
) -> None:
    # The three undirected edge keys of face ``f``, in corner order, at ``slot .. slot + 2``.
    # Every caller here but ``selection.deleted_face_edge_keys`` passes ``slot = 3f``: edge
    # ``3f + k`` then belongs to face ``f``, which is what lets ``edge_pairs_to_face_pairs`` and
    # ``edge_endpoints`` recover everything else from an edge index alone.
    #
    # ``pack_edge_key`` is byte-identical to what ``pack_indices`` produces for the sorted edge row
    # ``[min, max]``, which is what keeps the radix sort's key order -- and so the adjacency row
    # order -- the same as the composed ``faces_to_edges`` + ``pack_indices`` path.
    c = f * 3
    i0 = faces[c + 0]
    i1 = faces[c + 1]
    i2 = faces[c + 2]
    out_keys[slot + 0] = pack_edge_key(i0, i1, base)
    out_keys[slot + 1] = pack_edge_key(i1, i2, base)
    out_keys[slot + 2] = pack_edge_key(i2, i0, base)


@wp.kernel
def face_edge_keys(
    faces: wp.array[wp.int32], base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # One launch in place of ``faces_to_edges`` + ``pack_indices``, so the intermediate
    # ``(3F, 2)`` edge rows are never materialized. ``remesh.begin_decimation_pass`` writes the same
    # keys over a fixed-capacity buffer, differing only in a sentinel key past the live faces.
    f = wp.int32(wp.tid())
    write_face_edge_keys(faces, f, 3 * f, base, out_keys)


@wp.kernel
def face_edge_keys_and_order(
    faces: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # ``face_edge_keys`` plus the identity payload ``radix_sort_pairs`` carries, written
    # straight into the first half of the sort's double-width buffers: no key copy, and no
    # separate identity fill. The upper halves are sort scratch and are never initialized.
    f = wp.int32(wp.tid())
    write_face_edge_keys(faces, f, 3 * f, base, out_keys)
    for k in range(3):
        out_order[3 * f + k] = 3 * f + k


@wp.func
def sorted_pair_slot(sorted_keys: wp.array[wp.uint64], i: wp.int32) -> tuple[wp.int32, wp.bool]:
    """
    Classify sorted position ``i`` of the packed halfedge keys by the run of equal keys it is in.

    Returns ``(first, unpaired_start)``: ``first`` is the sorted position of the first member when
    ``i`` belongs to a run of *exactly* two keys -- one undirected edge shared by two faces, the
    rows ``grouping.group(keys, 2)`` emits and so the rows of ``face_adjacency`` -- and ``-1``
    otherwise; ``unpaired_start`` is ``True`` on the first position of every other run (a boundary
    edge, or one shared by three or more faces), so a kernel can flag a non-paired edge exactly
    once. A kernel launched over every sorted position can therefore act on each adjacency pair
    with no compacted pair table and no host read of its length. Every neighbour read sits behind
    its own bounds test.
    """
    n = sorted_keys.shape[0]
    key = sorted_keys[i]
    prev_same = wp.bool(False)
    if i > 0:
        prev_same = sorted_keys[i - 1] == key
    next_same = wp.bool(False)
    if i + 1 < n:
        next_same = sorted_keys[i + 1] == key
    first = wp.int32(-1)
    unpaired_start = wp.bool(False)
    if not prev_same:
        third_same = wp.bool(False)
        if i + 2 < n:
            third_same = sorted_keys[i + 2] == key
        if next_same and not third_same:
            first = i
        else:
            unpaired_start = True
    elif not next_same:
        before_same = wp.bool(False)
        if i > 1:
            before_same = sorted_keys[i - 2] == key
        if not before_same:
            first = i - 1
    return first, unpaired_start


@wp.func
def edge_endpoints(faces: wp.array[wp.int32], edge_index: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The sorted endpoints of edge ``3f + c``, recovered from the edge index alone: halfedge
    # ``3f + c`` (``halfedge.halfedge_endpoints``) spans ``(v[c], v[(c + 1) % 3])``, matching
    # ``kernels/edges.py:faces_to_edges``, so this reproduces exactly the row
    # ``faces_to_edges(sorted=True)`` would have written there.
    a, b = halfedge_endpoints(faces, edge_index)
    return wp.min(a, b), wp.max(a, b)


@wp.func
def write_edge_row(
    faces: wp.array[wp.int32], edge_index: wp.int32, row: wp.int32, out_edges: wp.array2d[wp.int32]
) -> None:
    # Write edge ``edge_index``'s sorted endpoints (``edge_endpoints``) as row ``row`` of an
    # ``(m, 2)`` edge table: how every kernel emitting a unique-edge row from a representative
    # corner fills it, with no ``(3 * n_faces, 2)`` table to gather from.
    a, b = edge_endpoints(faces, edge_index)
    out_edges[row, 0] = a
    out_edges[row, 1] = b


@wp.func
def write_adjacency_row(
    faces: wp.array[wp.int32],
    edges_sorted: wp.array2d[wp.int32],
    edge_0: wp.int32,
    edge_1: wp.int32,
    row: wp.int32,
    out_adjacency: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # Two halfedge indices sharing a key -> row ``row`` of ``face_adjacency``: the ascending pair
    # of faces owning them, and with ``out_edges`` (a null descriptor otherwise) the shared edge.
    # The owning face of edge ``e`` is just ``e // 3``, and ordering two values needs no sort
    # kernel. The shared edge is the first halfedge's sorted endpoints: from a caller-supplied
    # ``edges_sorted`` row when one is given (a null descriptor otherwise), else recovered from
    # the halfedge index through ``edge_endpoints``, so no ``(3 * n_faces, 2)`` edge table has to
    # exist to gather it from. Both selectors are warp-uniform.
    f0 = edge_0 // 3
    f1 = edge_1 // 3
    out_adjacency[row, 0] = wp.min(f0, f1)
    out_adjacency[row, 1] = wp.max(f0, f1)
    if out_edges.shape[0] > 0:
        if edges_sorted.shape[0] > 0:
            out_edges[row, 0] = edges_sorted[edge_0, 0]
            out_edges[row, 1] = edges_sorted[edge_0, 1]
        else:
            write_edge_row(faces, edge_0, row, out_edges)


@wp.kernel
def edge_pairs_to_face_pairs(
    faces: wp.array[wp.int32],
    edges_sorted: wp.array2d[wp.int32],
    edge_groups: wp.array2d[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # ``write_adjacency_row`` for every row of an ``(m, 2)`` halfedge-pair table.
    tid = wp.int32(wp.tid())
    write_adjacency_row(
        faces, edges_sorted, edge_groups[tid, 0], edge_groups[tid, 1], tid, out_adjacency, out_edges
    )


@wp.kernel
def emit_sorted_face_pairs(
    faces: wp.array[wp.int32],
    edges_sorted: wp.array2d[wp.int32],
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # ``write_adjacency_row`` straight off the sorted halfedge keys, launched over the sorted
    # positions with ``inclusive`` the in-place inclusive scan of the exact-pair starts
    # (``grouping.mark_group_starts`` at length two): a position starts a pair exactly where the
    # scan steps, and the exclusive offset is the pair's row -- ``grouping.emit_groups``'
    # compaction, with the pair's faces and edge written in the same launch rather than from an
    # intermediate ``(m, 2)`` group table.
    i = wp.int32(wp.tid())
    row, count = scanned_count(inclusive, i)
    if count == 0:
        return
    write_adjacency_row(faces, edges_sorted, order[i], order[i + 1], row, out_adjacency, out_edges)


@wp.func
def mate_pair_faces(mates: wp.array[wp.int32], h: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The face-adjacency graph as one edge per halfedge, for a union-find: the lower halfedge of a
    # pair (``halfedge.halfedge_mates``) gives the ``edge_pairs_to_face_pairs`` pair, every other
    # halfedge a self-loop on its own face, which the union-find ignores. No pair table is
    # compacted, so nothing is sized by a host read, and the labels -- each component's smallest
    # face id -- are the ones the compacted adjacency gives.
    f0 = h // 3
    f1 = f0
    mate = mates[h]
    if mate > h:
        f1 = mate // 3
    return f0, f1


@wp.kernel
def mate_pair_prehook(mates: wp.array[wp.int32], parents: wp.array[wp.int32]) -> None:
    # ``connected_components.ecl_init_parent_edges`` over ``mate_pair_faces``' edges, formed in
    # the thread: no ``(3 n_faces, 2)`` edge table is written only to be read back twice.
    f0, f1 = mate_pair_faces(mates, wp.int32(wp.tid()))
    ecl_prehook_pair(parents, f0, f1)


@wp.kernel
def mate_pair_hook(mates: wp.array[wp.int32], parents: wp.array[wp.int32]) -> None:
    # ``connected_components.ecl_hook_edges`` over the same edges, after ``mate_pair_prehook``.
    f0, f1 = mate_pair_faces(mates, wp.int32(wp.tid()))
    ecl_hook_pair(parents, f0, f1)


@wp.func
def unshared_vertex(
    v0: wp.int32, v1: wp.int32, v2: wp.int32, e0: wp.int32, e1: wp.int32
) -> wp.int32:
    result = wp.int32(-1)
    count = wp.int32(0)
    if v0 != e0 and v0 != e1:
        result = v0
        count = count + wp.int32(1)
    if v1 != e0 and v1 != e1:
        if count == wp.int32(0):
            result = v1
        count = count + wp.int32(1)
    if v2 != e0 and v2 != e1:
        if count == wp.int32(0):
            result = v2
        count = count + wp.int32(1)
    if count != wp.int32(1):
        return wp.int32(-1)
    return result


@wp.kernel
def face_adjacency_unshared(
    faces: wp.array[wp.int32],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    f0 = face_adjacency[tid, 0] * 3
    f1 = face_adjacency[tid, 1] * 3
    e0 = face_adjacency_edges[tid, 0]
    e1 = face_adjacency_edges[tid, 1]
    out_unshared[tid, 0] = unshared_vertex(faces[f0 + 0], faces[f0 + 1], faces[f0 + 2], e0, e1)
    out_unshared[tid, 1] = unshared_vertex(faces[f1 + 0], faces[f1 + 1], faces[f1 + 2], e0, e1)


@wp.func
def edge_pair_topology(
    faces: wp.array[wp.int32], edge_0: wp.int32, edge_1: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32, wp.int32, wp.int32]:
    # Everything a manifold edge's two grouped halfedge indices determine: the sorted endpoints of
    # the edge they share, the two faces owning them, and each face's opposite apex. No edge table
    # and no adjacency table is read -- a halfedge index already encodes its face as ``edge // 3``.
    #
    # Shared with ``remesh.emit_flip_topology``, which wants the identical six values and differs
    # only in where it puts them: a scan-derived slot rather than a thread row, plus the adjacency
    # pair itself. It lives here because the quantity is edge topology, not a remeshing step.
    shared_a, shared_b = edge_endpoints(faces, edge_0)
    face_0 = edge_0 // 3
    face_1 = edge_1 // 3
    i0, i1, i2 = corner_triple(faces, face_0)
    j0, j1, j2 = corner_triple(faces, face_1)
    unshared_0 = unshared_vertex(i0, i1, i2, shared_a, shared_b)
    unshared_1 = unshared_vertex(j0, j1, j2, shared_a, shared_b)
    return shared_a, shared_b, face_0, face_1, unshared_0, unshared_1


@wp.kernel
def emit_sorted_unshared(
    faces: wp.array[wp.int32],
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_unshared: wp.array2d[wp.int32],
) -> None:
    # Same answer as ``face_adjacency_unshared`` with no edge table and no adjacency table,
    # compacted off the sorted halfedge keys exactly as ``emit_sorted_face_pairs`` compacts the
    # pairs, so the rows align with ``face_adjacency``'s. Column order follows that kernel, which
    # emits the face pair ascending.
    i = wp.int32(wp.tid())
    row, count = scanned_count(inclusive, i)
    if count == 0:
        return
    _shared_a, _shared_b, face_0, face_1, unshared_0, unshared_1 = edge_pair_topology(
        faces, order[i], order[i + 1]
    )
    if face_0 <= face_1:
        out_unshared[row, 0] = unshared_0
        out_unshared[row, 1] = unshared_1
    else:
        out_unshared[row, 0] = unshared_1
        out_unshared[row, 1] = unshared_0


@wp.kernel
def face_adjacency_angles(
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    out_angles: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    normal_a = face_normals[face_adjacency[tid, 0]]
    normal_b = face_normals[face_adjacency[tid, 1]]
    out_angles[tid] = vector_angle(normal_a, normal_b)


@wp.kernel
def scatter_vertex_faces(
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_vertex_faces: wp.array[wp.int32],
) -> None:
    # Vertex-to-face CSR payload. Row order is thread-order and therefore arbitrary, which is what
    # the wrapper documents: a rotational order would need halfedge twins and would refuse a
    # vertex-non-manifold mesh, which the decimator that consumes this must not do.
    f = wp.int32(wp.tid())
    for k in range(3):
        v = faces[f * 3 + k]
        out_vertex_faces[offsets[v] + wp.atomic_add(cursor, v, 1)] = f


@wp.func
def unshared_projection(
    vertices: wp.array[wp.vec3], normal: wp.vec3, origin: wp.int32, other: wp.int32
) -> wp.float32:
    # Signed distance of a face pair's second unshared vertex ``other`` above the first face's
    # plane, measured from the shared edge's first endpoint ``origin``: the convexity rule
    # ``face_adjacency_convex`` thresholds and ``curvature.face_pair_dihedrals`` signs its angle by.
    #
    # ``unshared_vertex`` returns -1 for a degenerate second face (it does not have exactly one
    # vertex off the shared edge), and that sentinel is not a valid index into ``vertices`` --
    # reading it would be the out-of-bounds access CLAUDE.md's memory-safety rule forbids. There is
    # no meaningful projection for a degenerate face, so it reports as never locally convex
    # (+inf is never < TOLERANCE_MERGE) rather than being read as an arbitrary finite value.
    if other < wp.int32(0):
        return FLOAT32_INF_CONSTANT
    return wp.dot(vertices[other] - vertices[origin], normal)


@wp.func
def adjacency_projection(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    row: wp.int32,
) -> wp.float32:
    # ``unshared_projection`` for adjacency row ``row``, read from the precomputed tables: the
    # quantity ``face_adjacency_projections`` returns and ``face_adjacency_convex`` thresholds.
    return unshared_projection(
        vertices,
        face_normals[face_adjacency[row, 0]],
        face_adjacency_edges[row, 0],
        face_adjacency_unshared[row, 1],
    )


@wp.kernel
def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    out_projections: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    out_projections[tid] = adjacency_projection(
        vertices, face_normals, face_adjacency, face_adjacency_edges, face_adjacency_unshared, tid
    )


@wp.kernel
def face_adjacency_convex(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    tolerance: wp.float32,
    out_convex: wp.array[wp.bool],
) -> None:
    # The projection and its ``< tolerance`` threshold in one launch, where the wrapper used to
    # write the ``(m,)`` projections and ``wp.map`` a comparison over them.
    tid = wp.int32(wp.tid())
    projection = adjacency_projection(
        vertices, face_normals, face_adjacency, face_adjacency_edges, face_adjacency_unshared, tid
    )
    out_convex[tid] = projection < tolerance
