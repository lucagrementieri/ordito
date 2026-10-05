import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels.adjacency import edge_endpoints, sorted_pair_slot, write_face_edge_keys
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_hook_pair_parity,
    ecl_prehook_pair,
    find_representative,
)
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.intersection import candidate_pair_intersects, candidate_slot_query
from ordito.kernels.predicates import vector_angle

wp.set_module_options({"enable_backward": False})


@wp.func
def directed_edge(
    faces: wp.array[wp.int32], edges: wp.array2d[wp.int32], e: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """
    Endpoints of directed edge ``e`` in ``faces_to_edges`` row order.

    The caller's table row when ``edges`` is given, else halfedge ``e`` read off ``faces``
    (``halfedge.halfedge_endpoints``) -- the row that table would hold. ``edges`` is a null
    descriptor (shape 0) when the caller passed none.
    """
    if edges.shape[0] > 0:
        return edges[e, 0], edges[e, 1]
    return halfedge_endpoints(faces, e)


@wp.kernel
def edge_pair_winding_mask(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    edge_groups: wp.array2d[wp.int32],
    out_consistent: wp.array[wp.bool],
) -> None:
    tid = wp.int32(wp.tid())
    _a0, b0 = directed_edge(faces, edges, edge_groups[tid, 0])
    a1, _b1 = directed_edge(faces, edges, edge_groups[tid, 1])
    out_consistent[tid] = b0 == a1


@wp.kernel
def sorted_pair_winding_violation(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    require_pairs: wp.bool,
    out_violation: wp.array[wp.int32],
) -> None:
    """
    Raise ``out_violation[0]`` when a shared edge's two faces traverse it the same way.

    ``edge_pair_winding_mask``'s per-pair test, read straight off the sorted keys (each pair's
    first member does it), so a predicate needs no compacted group table and no host read of its
    length. With ``require_pairs`` a run that is not exactly two keys also raises the flag: the
    "every edge shared by exactly two faces" half of ``is_volume``.
    """
    i = wp.int32(wp.tid())
    first, unpaired_start = sorted_pair_slot(sorted_keys, i)
    if unpaired_start and require_pairs:
        out_violation[0] = 1
    if first == i:
        _a0, b0 = directed_edge(faces, edges, order[i])
        a1, _b1 = directed_edge(faces, edges, order[i + 1])
        if b0 != a1:
            out_violation[0] = 1


@wp.func
def local_index(faces: wp.array[wp.int32], f: wp.int32, v: wp.int32) -> wp.int32:
    """Position (0, 1, 2) of vertex ``v`` within face ``f``."""
    base = f * wp.int32(3)
    if faces[base] == v:
        return wp.int32(0)
    if faces[base + wp.int32(1)] == v:
        return wp.int32(1)
    return wp.int32(2)


@wp.func
def edge_forward_in_face(
    faces: wp.array[wp.int32], f: wp.int32, a: wp.int32, b: wp.int32
) -> wp.int32:
    """1 if face ``f`` traverses the shared edge as ``a -> b``, else 0 (``b -> a``)."""
    la = local_index(faces, f, a)
    nxt = faces[f * wp.int32(3) + (la + wp.int32(1)) % wp.int32(3)]
    if nxt == b:
        return wp.int32(1)
    return wp.int32(0)


@wp.func
def edge_manifold(count: wp.int32, allow_boundary: wp.bool) -> wp.bool:
    """Per-unique-edge manifold flag from its face-share count."""
    if allow_boundary:
        return count <= wp.int32(2)
    return count == wp.int32(2)


@wp.kernel
def face_edge_keys_checked(
    faces: wp.array[wp.int32],
    base: wp.uint64,
    bound: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_flags: wp.array[wp.int32],
) -> None:
    """
    ``adjacency.face_edge_keys`` plus the range check of the indices it packs.

    Raises ``out_flags[1]`` when a corner index is negative or reaches ``bound``, so the check
    rides on the pass that reads the faces anyway and shares the verdict's readback, where a
    separate ``reduce.minmax`` would add a launch and a readback of its own. ``out_flags[0]`` is
    left to ``edge_share_count_violation``.
    """
    f = wp.int32(wp.tid())
    write_face_edge_keys(faces, f, 3 * f, base, out_keys)
    for k in range(3):
        v = faces[3 * f + k]
        if v < 0 or v >= bound:
            out_flags[1] = 1


@wp.kernel
def edge_share_count_violation(
    slot_counts: wp.array[wp.int32], allow_boundary: wp.bool, out_violation: wp.array[wp.int32]
) -> None:
    """
    Raise ``out_violation[0]`` when an occupied hash-table slot holds a non-manifold share count.

    ``slot_counts`` is a ``hash_insert`` occurrence table over the packed edge keys, so an empty
    slot reads ``0`` and every occupied one is exactly one undirected edge's face-share count. The
    predicate needs only whether *some* edge fails, so the table is read in place: no compaction,
    no sort and no count of the unique edges. The store is unsynchronized because every writer
    stores the same ``1``, and it fires only on a violating slot, so it contends with nothing on a
    manifold mesh.
    """
    s = wp.int32(wp.tid())
    count = slot_counts[s]
    if count > 0 and not edge_manifold(count, allow_boundary):
        out_violation[0] = 1


@wp.kernel
def sorted_run_manifold_mask(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    allow_boundary: wp.bool,
    out_mask: wp.array[wp.bool],
) -> None:
    """
    Clear the face flag of every halfedge whose undirected edge is not edge-manifold.

    One thread per sorted halfedge key. An edge's face-share count is the length of its run of
    equal keys, and [`edge_manifold`][ordito.kernels.validation.edge_manifold] only asks whether
    that is one, two, or more -- which the neighbouring keys answer (``sorted_pair_slot`` for an
    exact pair, then one more equal neighbour means three or more). So no unique-edge table, no
    inverse and no count array is built, and ``out_mask`` (arriving all ``True``) needs no gather.
    The stores are unsynchronized because every writer stores the same ``False``.
    """
    i = wp.int32(wp.tid())
    first, _unpaired_start = sorted_pair_slot(sorted_keys, i)
    count = wp.int32(2)
    if first < 0:
        count = 1
        key = sorted_keys[i]
        if i > 0:
            if sorted_keys[i - 1] == key:
                count = 3
        if i + 1 < sorted_keys.shape[0]:
            if sorted_keys[i + 1] == key:
                count = 3
    if not edge_manifold(count, allow_boundary):
        out_mask[order[i] // 3] = False


@wp.func
def corner_link(
    faces: wp.array[wp.int32], f0: wp.int32, f1: wp.int32, v: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """Link the ``v`` corners of edge-adjacent faces ``f0`` and ``f1`` in the corner graph."""
    return wp.int32(3) * f0 + local_index(faces, f0, v), wp.int32(3) * f1 + local_index(
        faces, f1, v
    )


@wp.func
def adjacency_corner_links(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    r: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    """
    Return the two corner-graph edges of face-adjacency row ``r``, one per shared-edge endpoint.

    Corner node ``3 * f + k`` is the corner of face ``f`` at its local vertex ``k``. Two faces
    sharing edge ``(a, b)`` fan-connect their ``a`` corners and their ``b`` corners, so the
    connected components of the corner graph are exactly the edge-connected fans around each
    vertex. Shared by ``adjacency_corner_prehook`` and ``adjacency_corner_hook``, which form the
    edges in the thread so no ``(2 m, 2)`` corner-edge table is written only to be read back
    twice.
    """
    f0 = adjacency[r, 0]
    f1 = adjacency[r, 1]
    a0, a1 = corner_link(faces, f0, f1, adjacency_edges[r, 0])
    b0, b1 = corner_link(faces, f0, f1, adjacency_edges[r, 1])
    return a0, a1, b0, b1


@wp.kernel
def adjacency_corner_prehook(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_init_parent_edges`` over ``adjacency_corner_links``' two edges.
    a0, a1, b0, b1 = adjacency_corner_links(faces, adjacency, adjacency_edges, wp.int32(wp.tid()))
    ecl_prehook_pair(parents, a0, a1)
    ecl_prehook_pair(parents, b0, b1)


@wp.kernel
def adjacency_corner_hook(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_hook_edges`` over the same edges, after the pre-hook.
    a0, a1, b0, b1 = adjacency_corner_links(faces, adjacency, adjacency_edges, wp.int32(wp.tid()))
    ecl_hook_pair(parents, a0, a1)
    ecl_hook_pair(parents, b0, b1)


@wp.func
def sorted_corner_link(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    i: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.bool]:
    """
    Corner-graph edge of sorted halfedge position ``i``, plus ``sorted_pair_slot``'s run flag.

    ``adjacency_corner_links`` with no face-adjacency table between them. Both members of an
    exact pair give one link each -- the pair's first shared-edge endpoint from its first member,
    the second from its second -- which are the two links the adjacency row would have produced,
    taken from the same first-member edge. Every other position gives the self-loop ``(i, i)``, a
    no-op to the union-find. The component labels are each component's smallest corner id whatever
    order or multiplicity the unions come in, so they equal the adjacency path's exactly. Shared
    by ``sorted_corner_prehook`` and ``sorted_corner_hook``, which form the edge in the thread so
    no ``(3 n_faces, 2)`` corner-edge table is written only to be read back twice.
    """
    first, unpaired_start = sorted_pair_slot(sorted_keys, i)
    if first < 0:
        return i, i, unpaired_start
    e0 = order[first]
    a, b = edge_endpoints(faces, e0)
    c0, c1 = corner_link(faces, e0 // 3, order[first + 1] // 3, wp.where(i == first, a, b))
    return c0, c1, unpaired_start


@wp.kernel
def sorted_corner_prehook(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    require_pairs: wp.bool,
    parents: wp.array[wp.int32],
    out_violation: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_init_parent_edges`` over ``sorted_corner_link``'s edges. With
    # ``require_pairs`` a run that is not exactly two keys raises ``out_violation[0]``: the edge
    # half of ``is_watertight``, folded into the same pass; without it the flag is never indexed.
    i = wp.int32(wp.tid())
    c0, c1, unpaired_start = sorted_corner_link(faces, sorted_keys, order, i)
    if unpaired_start and require_pairs:
        out_violation[0] = 1
    ecl_prehook_pair(parents, c0, c1)


@wp.kernel
def sorted_corner_hook(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_hook_edges`` over the same edges, after ``sorted_corner_prehook``.
    c0, c1, _unpaired_start = sorted_corner_link(faces, sorted_keys, order, wp.int32(wp.tid()))
    ecl_hook_pair(parents, c0, c1)


@wp.kernel
def corner_labels_and_vertex_min(
    faces: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    out_labels: wp.array[wp.int32],
    out_min_label: wp.array[wp.int32],
    out_referenced: wp.array[wp.bool],
) -> None:
    """
    ``connected_components.ecl_flatten`` fused with the per-vertex reduction of its labels.

    Each corner's label is its component root, and every vertex keeps the smallest label over its
    corners. ``out_referenced`` (a null descriptor when the caller has no per-vertex mask) marks
    the vertices a corner reaches.
    """
    c = wp.int32(wp.tid())
    label = find_representative(parents, c)
    out_labels[c] = label
    v = faces[c]
    wp.atomic_min(out_min_label, v, label)
    if out_referenced.shape[0] > 0:
        out_referenced[v] = True


@wp.kernel
def corner_vertex_check(
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    min_label: wp.array[wp.int32],
    out_mask: wp.array[wp.bool],
    out_violation: wp.array[wp.int32],
) -> None:
    """
    Flag a vertex one of whose corners is in a different fan.

    The flag clears the vertex's ``out_mask`` entry or raises ``out_violation[0]``, whichever of
    the two the caller passed (the other is a null descriptor).

    The predicate form also answers libigl's range half -- every vertex in ``[0, max(faces)]``
    must be referenced -- from the corners: an unreferenced vertex lies below ``max(faces)``
    exactly when some unreferenced vertex is followed by a referenced one, and that referenced
    vertex has a corner here that sees its predecessor's ``min_label`` still at the ``int32``
    maximum it was seeded with. So the bound itself is never needed, and no vertex-sized pass is.
    """
    c = wp.int32(wp.tid())
    v = faces[c]
    if out_mask.shape[0] > 0:
        if labels[c] != min_label[v]:
            out_mask[v] = False
        return
    violation = labels[c] != min_label[v]
    if v > 0:
        if min_label[v - 1] == INT32_MAX_CONSTANT:
            violation = True
    if violation:
        out_violation[0] = 1


@wp.kernel
def self_intersection_marks(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    max_hits: wp.int32,
    mark_faces: wp.bool,
    out_marks: wp.array[wp.bool],
) -> None:
    """
    Narrow phase over ``collect_face_box_candidates``' slot blocks, one thread per slot.

    ``intersection.candidate_pair_intersects`` decides each live candidate. ``mark_faces`` selects
    the answer: the per-face mask marks both faces of every intersecting candidate (idempotent
    ``True`` stores); the predicate stores ``True`` at ``out_marks[0]``, and a thread finding the
    flag already raised skips its test. One kernel serves both, so the mask and the predicate
    cannot disagree about a pair. The narrow phase stays one thread per pair rather than inside
    the traversal: fused, a thread runs up to ``max_hits`` ``float64`` tests serially while holding
    the traversal state, and measured several times slower.
    """
    s = wp.int32(wp.tid())
    f = candidate_slot_query(counts, max_hits, s)
    if f < 0:
        return
    if not mark_faces:
        if out_marks[0]:
            return
    target = targets[s]
    if candidate_pair_intersects(vertices, faces, vertices, faces, f, target):
        if mark_faces:
            out_marks[f] = True
            out_marks[target] = True
        else:
            out_marks[0] = True


@wp.func
def pair_flip_sign(
    faces: wp.array[wp.int32], f0: wp.int32, f1: wp.int32, a: wp.int32, b: wp.int32
) -> wp.int32:
    """
    Z2 flip bit of an edge-adjacent face pair sharing edge ``(a, b)``.

    ``0`` when the two faces traverse it in opposite directions (compatible orientations), ``1``
    when a flip is required.
    """
    d0 = edge_forward_in_face(faces, f0, a, b)
    d1 = edge_forward_in_face(faces, f1, a, b)
    return wp.where(d0 == d1, wp.int32(1), wp.int32(0))


@wp.kernel
def build_signed_face_edges(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
    out_sign: wp.array[wp.int32],
) -> None:
    """Face-pair edges with a Z2 flip bit (``pair_flip_sign``) for orientability propagation."""
    r = wp.int32(wp.tid())
    f0 = adjacency[r, 0]
    f1 = adjacency[r, 1]
    out_edges[r, 0] = f0
    out_edges[r, 1] = f1
    out_sign[r] = pair_flip_sign(faces, f0, f1, adjacency_edges[r, 0], adjacency_edges[r, 1])


@wp.func
def sorted_pair_signed_edge(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    i: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.int32]:
    """
    ``build_signed_face_edges``' row for sorted halfedge position ``i``, with no adjacency table.

    A pair's first member gives the row the adjacency table would hold -- the ascending face pair
    and the flip bit over its first member's edge -- and every other position a self-loop with
    sign ``0``, which the parity union-find skips and every orientation satisfies. The pair rows
    keep their adjacency order, so the unions run in the same sequence as over the compacted table.
    Shared by ``sorted_pair_hook_parity`` and ``sorted_pair_orientation_conflict``, which form the
    edge in the thread rather than read a ``(3 n_faces, 2)`` table and its signs.
    """
    first, _unpaired_start = sorted_pair_slot(sorted_keys, i)
    e0 = order[i]
    if first != i:
        return e0 // 3, e0 // 3, wp.int32(0)
    g0 = e0 // 3
    g1 = order[i + 1] // 3
    f0 = wp.min(g0, g1)
    f1 = wp.max(g0, g1)
    a, b = edge_endpoints(faces, e0)
    return f0, f1, pair_flip_sign(faces, f0, f1, a, b)


@wp.kernel
def sorted_pair_hook_parity(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    words: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_hook_parity`` over ``sorted_pair_signed_edge``'s edges.
    f0, f1, sign = sorted_pair_signed_edge(faces, sorted_keys, order, wp.int32(wp.tid()))
    ecl_hook_pair_parity(words, f0, f1, sign)


@wp.kernel
def sorted_pair_orientation_conflict(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    out_conflict: wp.array[wp.int32],
) -> None:
    # Flag any signed edge of ``sorted_pair_signed_edge``'s whose faces violate its flip constraint.
    f0, f1, sign = sorted_pair_signed_edge(faces, sorted_keys, order, wp.int32(wp.tid()))
    if ((orient[f0] + orient[f1]) & wp.int32(1)) != sign:
        out_conflict[0] = wp.int32(1)


@wp.kernel
def accumulate_neighbor_normals(
    face_normals: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    out_neighbor_sum: wp.array[wp.vec3],
    out_max_angle: wp.array[wp.float32],
) -> None:
    # One pass over the adjacency pairs, scattering both per-face quantities the bad-face criteria
    # need: the sum of the neighbouring face normals (whose direction is the local consensus) and
    # the sharpest dihedral angle to any neighbour (which is what a fold looks like).
    #
    # The dihedral angle is ``adjacency.face_adjacency_angles``' -- ``vector_angle`` of the pair's
    # two normals -- taken from the two normals this thread loads anyway, rather than read from a
    # ``(m,)`` table a launch of its own would have written for this pass alone.
    #
    # Launched over the sorted halfedge keys rather than a compacted ``face_adjacency`` table: each
    # pair's first member (``sorted_pair_slot``) does the pair's work, in the pair's adjacency-row
    # order, so no table is built and no host read of its length is needed.
    i = wp.int32(wp.tid())
    first, _unpaired_start = sorted_pair_slot(sorted_keys, i)
    if first != i:
        return
    e0 = order[i] // 3
    e1 = order[i + 1] // 3
    f0 = wp.min(e0, e1)
    f1 = wp.max(e0, e1)
    normal_0 = face_normals[f0]
    normal_1 = face_normals[f1]
    angle = vector_angle(normal_0, normal_1)
    wp.atomic_add(out_neighbor_sum, f0, normal_1)
    wp.atomic_add(out_neighbor_sum, f1, normal_0)
    wp.atomic_max(out_max_angle, f0, angle)
    wp.atomic_max(out_max_angle, f1, angle)


@wp.kernel
def face_defective_mask(
    quality: wp.array[wp.float32],
    face_normals: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    max_angle: wp.array[wp.float32],
    min_quality: wp.float32,
    max_normal_cos: wp.float32,
    max_fold_cos: wp.float32,
    out_bad: wp.array[wp.bool],
) -> None:
    # A face is bad if it is too thin, too far from its neighbourhood's consensus normal, or folded
    # back onto its own ring. Each criterion is disabled by passing a cosine of -2 / a quality of
    # -1, which no real value can reach, so the three gates compose without a separate flag
    # argument.
    #
    # The same sentinels also say which tables exist: a disabled criterion's inputs are never read,
    # so the wrapper passes ``None`` for them rather than allocating a table of constants (the two
    # branches are warp-uniform). A disabled pair of angle criteria reaches ``False`` below exactly
    # as their zeroed tables did: the zero consensus fails the normal test and ``cos(0)`` the fold.
    f = wp.int32(wp.tid())
    if min_quality > -1.0:
        if quality[f] < min_quality:
            out_bad[f] = wp.bool(True)
            return
    if max_normal_cos <= -2.0 and max_fold_cos <= -2.0:
        out_bad[f] = wp.bool(False)
        return

    # Direction of the sum of the neighbouring normals: the local consensus. ``wp.normalize`` of a
    # zero vector is zero in Warp, which is how an isolated face -- and a face whose neighbours
    # cancel each other exactly -- ends up with no consensus to disagree with.
    consensus = wp.normalize(neighbor_sum[f])
    agreement = wp.dot(face_normals[f], consensus)
    if wp.length_sq(consensus) > 0.0 and agreement < max_normal_cos:
        out_bad[f] = wp.bool(True)
        return

    # Fold: some neighbour meets this face at nearly pi (an unsigned dihedral, so the gate is on its
    # cosine) *and* this face is the one facing against its own neighbourhood. Both halves matter --
    # a fold has two faces and only one of them is the mistake, so flagging both would delete a good
    # triangle along with it.
    out_bad[f] = wp.cos(max_angle[f]) < max_fold_cos and agreement < 0.0
