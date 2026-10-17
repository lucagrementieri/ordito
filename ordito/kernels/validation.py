import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels.adjacency import edge_endpoints, write_face_edge_keys
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_hook_pair_parity,
    ecl_prehook_pair,
    find_representative,
    find_representative_parity,
)
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.predicates import vector_angle
from ordito.kernels.triangles import face_vertices, triangle_radius_ratio

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
def mate_pair_winding_violation(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    mates: wp.array[wp.int32],
    require_pairs: wp.bool,
    out_violation: wp.array[wp.int32],
) -> None:
    """
    Raise ``out_violation[0]`` when a shared edge's two faces traverse it the same way.

    ``edge_pair_winding_mask``'s per-pair test, read off the halfedge mates (each pair's lower
    halfedge does it), so a predicate needs no compacted group table and no host read of its
    length. With ``require_pairs`` a halfedge whose edge is not shared by exactly two faces also
    raises the flag: the "every edge shared by exactly two faces" half of ``is_volume``.
    """
    h = wp.int32(wp.tid())
    mate = mates[h]
    if mate < 0 and require_pairs:
        out_violation[0] = 1
    if mate > h:
        _a0, b0 = directed_edge(faces, edges, h)
        a1, _b1 = directed_edge(faces, edges, mate)
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
def mate_manifold_mask(
    mates: wp.array[wp.int32], allow_boundary: wp.bool, out_mask: wp.array[wp.bool]
) -> None:
    """
    Clear the face flag of every halfedge whose undirected edge is not edge-manifold.

    One thread per halfedge. [`edge_manifold`][ordito.kernels.validation.edge_manifold] only asks
    whether an edge's face-share count is one, two, or more, which the halfedge's mate code
    (``halfedge.halfedge_mates``) says outright: ``-1`` one, a partner two, ``<= -2`` three or
    more. So no unique-edge table, no inverse and no count array is built, and ``out_mask``
    (arriving all ``True``) needs no gather. The stores are unsynchronized because every writer
    stores the same ``False``.
    """
    h = wp.int32(wp.tid())
    mate = mates[h]
    count = wp.where(mate >= 0, wp.int32(2), wp.where(mate == -1, wp.int32(1), wp.int32(3)))
    if not edge_manifold(count, allow_boundary):
        out_mask[h // 3] = False


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
def mate_corner_link(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], h: wp.int32
) -> tuple[wp.int32, wp.int32, wp.bool]:
    """
    Corner-graph edge of halfedge ``h``, plus whether its edge is not shared by exactly two faces.

    ``adjacency_corner_links`` with no face-adjacency table between them. Both halfedges of a pair
    give one link each -- the pair's first shared-edge endpoint from its lower halfedge, the
    second from its upper -- which are the two links the adjacency row would have produced, taken
    from the lower halfedge's edge. Every other halfedge gives the self-loop ``(h, h)``, a no-op to
    the union-find. The component labels are each component's smallest corner id whatever order
    or multiplicity the unions come in, so they equal the adjacency path's exactly. Shared by
    ``mate_corner_prehook`` and ``mate_corner_hook``, which form the edge in the thread so no
    ``(3 n_faces, 2)`` corner-edge table is written only to be read back twice.
    """
    mate = mates[h]
    if mate < 0:
        return h, h, wp.bool(True)
    e0 = wp.min(h, mate)
    a, b = edge_endpoints(faces, e0)
    c0, c1 = corner_link(faces, e0 // 3, wp.max(h, mate) // 3, wp.where(h == e0, a, b))
    return c0, c1, wp.bool(False)


@wp.kernel
def mate_corner_prehook(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    require_pairs: wp.bool,
    parents: wp.array[wp.int32],
    out_violation: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_init_parent_edges`` over ``mate_corner_link``'s edges. With
    # ``require_pairs`` an edge not shared by exactly two faces raises ``out_violation[0]``: the
    # edge half of ``is_watertight``, folded into the same pass; without it the flag is never
    # indexed.
    h = wp.int32(wp.tid())
    c0, c1, unpaired = mate_corner_link(faces, mates, h)
    if unpaired and require_pairs:
        out_violation[0] = 1
    ecl_prehook_pair(parents, c0, c1)


@wp.kernel
def mate_corner_hook(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], parents: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_hook_edges`` over the same edges, after ``mate_corner_prehook``.
    c0, c1, _unpaired = mate_corner_link(faces, mates, wp.int32(wp.tid()))
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
    require_referenced: wp.int32,
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
    ``require_referenced == 0`` drops that half (Open3D's convention, which ``is_watertight``
    follows: an unused vertex is not part of the surface).
    """
    c = wp.int32(wp.tid())
    v = faces[c]
    if out_mask.shape[0] > 0:
        if labels[c] != min_label[v]:
            out_mask[v] = False
        return
    violation = labels[c] != min_label[v]
    if require_referenced != 0 and v > 0:
        if min_label[v - 1] == INT32_MAX_CONSTANT:
            violation = True
    if violation:
        out_violation[0] = 1


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
def mate_signed_edge(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], h: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32]:
    """
    ``build_signed_face_edges``' row for halfedge ``h``, with no adjacency table.

    A pair's lower halfedge gives the row the adjacency table would hold -- the ascending face
    pair and the flip bit over that halfedge's edge -- and every other halfedge a self-loop with
    sign ``0``, which the parity union-find skips and every orientation satisfies. Shared by
    ``mate_hook_parity`` and ``mate_orientation_conflict``, which form the edge in the thread
    rather than read a ``(3 n_faces, 2)`` table and its signs.
    """
    mate = mates[h]
    if mate <= h:
        return h // 3, h // 3, wp.int32(0)
    g0 = h // 3
    g1 = mate // 3
    f0 = wp.min(g0, g1)
    f1 = wp.max(g0, g1)
    a, b = edge_endpoints(faces, h)
    return f0, f1, pair_flip_sign(faces, f0, f1, a, b)


@wp.kernel
def mate_hook_parity(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], words: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_hook_parity`` over ``mate_signed_edge``'s edges.
    f0, f1, sign = mate_signed_edge(faces, mates, wp.int32(wp.tid()))
    ecl_hook_pair_parity(words, f0, f1, sign)


@wp.kernel
def flatten_parity_flip_mask(words: wp.array[wp.int32], out_flip: wp.array[wp.bool]) -> None:
    # ``connected_components.ecl_flatten_parity`` with the parity written as the flip mask
    # ``face_flip_mask`` returns -- a parity is ``0`` or ``1``, so ``parity != 0`` is
    # ``orient > 0`` -- and no label, for the caller that wants the mask alone rather than the
    # ``int32`` bits.
    v = wp.int32(wp.tid())
    _root, parity = find_representative_parity(words, v)
    out_flip[v] = parity != 0


@wp.kernel
def mate_orientation_conflict(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    out_conflict: wp.array[wp.int32],
) -> None:
    # Flag any signed edge of ``mate_signed_edge``'s whose faces violate its flip constraint.
    f0, f1, sign = mate_signed_edge(faces, mates, wp.int32(wp.tid()))
    if ((orient[f0] + orient[f1]) & wp.int32(1)) != sign:
        out_conflict[0] = wp.int32(1)


# A spanning forest of the mate pairs that is a function of the input alone, for ``face_flip_mask``
# on a non-orientable mesh under the deterministic setting. There the parity union-find's answer
# is whichever spanning forest its racing hooks built: no potential exists, and the edges left out
# of the forest are where the contradiction lands. Every pair keyed by its lower halfedge (the
# keys are distinct), the minimum spanning forest is unique, and Boruvka's rounds find it with
# integer atomics only, which are exact: ``mate_forest_candidates`` takes each component's lightest
# outgoing pair by ``atomic_min``, ``mate_forest_hook`` adds it to the forest and hooks the
# component, and ``forest_flatten`` relabels. The parity union-find over the forest's pairs alone
# (``mate_hook_parity_forest``) is then order-free: each tree's root is its smallest face at parity
# ``0`` and every other face's parity is its tree path's. On a consistent component that is the
# fast path's answer, whatever the tree.


@wp.kernel
def mate_forest_candidates(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    best: wp.array[wp.int32],
) -> None:
    # One thread per halfedge: a pair joining two components offers its key to both.
    h = wp.int32(wp.tid())
    f0, f1, _sign = mate_signed_edge(faces, mates, h)
    if f0 == f1:
        return
    c0 = labels[f0]
    c1 = labels[f1]
    if c0 != c1:
        wp.atomic_min(best, c0, h)
        wp.atomic_min(best, c1, h)


@wp.kernel
def mate_forest_hook(
    mates: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    best: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    in_forest: wp.array[wp.int32],
    hooked: wp.array[wp.int32],
) -> None:
    # One thread per face; a component root with an outgoing pair takes it into the forest and
    # hooks onto the component across it. Two roots that chose the same pair (the only cycle
    # distinct keys allow) would hook onto each other, so the smaller stays the root.
    r = wp.int32(wp.tid())
    if labels[r] != r:
        return
    h = best[r]
    if h == INT32_MAX_CONSTANT:
        return
    other = labels[h // 3]
    if other == r:
        other = labels[mates[h] // 3]
    in_forest[h] = wp.int32(1)
    if best[other] == h and r < other:
        return
    parents[r] = other
    hooked[0] = wp.int32(1)


@wp.kernel
def forest_flatten(
    parents: wp.array[wp.int32], out_labels: wp.array[wp.int32], out_best: wp.array[wp.int32]
) -> None:
    # Each face's root, by path halving: a concurrent halving writes an ancestor over an ancestor,
    # so the trees race but the roots do not. Also clears the face's slot for the next round.
    v = wp.int32(wp.tid())
    x = v
    p = parents[x]
    while p != x:
        grandparent = parents[p]
        parents[x] = grandparent
        x = grandparent
        p = parents[x]
    out_labels[v] = x
    out_best[v] = INT32_MAX_CONSTANT


@wp.kernel
def mate_hook_parity_forest(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    in_forest: wp.array[wp.int32],
    words: wp.array[wp.int32],
) -> None:
    # ``mate_hook_parity`` over the forest's pairs only; every other halfedge is a self-loop.
    h = wp.int32(wp.tid())
    if in_forest[h] != 0:
        f0, f1, sign = mate_signed_edge(faces, mates, h)
        ecl_hook_pair_parity(words, f0, f1, sign)


@wp.func
def mate_pair_key(mates: wp.array[wp.int32], h: wp.int32) -> wp.int32:
    # The lower halfedge of ``h``'s adjacency pair, or ``INT32_MAX`` for a halfedge with no mate:
    # the order in which a launch over the pairs (each at its lower halfedge) visits them.
    mate = mates[h]
    return wp.where(mate >= 0, wp.min(h, mate), INT32_MAX_CONSTANT)


@wp.func
def neighbor_normals(
    face_normals: wp.array[wp.vec3], mates: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.vec3, wp.float32]:
    # Both per-face quantities the bad-face criteria need, gathered through the face's three
    # halfedge mates: the sum of the neighbouring face normals (whose direction is the local
    # consensus) and the sharpest dihedral angle to any neighbour (which is what a fold looks like).
    # The dihedral angle is ``adjacency.face_adjacency_angles``' -- ``vector_angle`` of the pair's
    # two normals, lower face first.
    #
    # The sum is formed in ascending pair-key order (each pair keyed by its lower halfedge), the
    # order a serial scatter over the pairs adds them in, so the result does not depend on which
    # corner of the face a neighbour sits at.
    normal = face_normals[f]
    keys = wp.vec3i(
        mate_pair_key(mates, 3 * f),
        mate_pair_key(mates, 3 * f + 1),
        mate_pair_key(mates, 3 * f + 2),
    )
    slots = wp.vec3i(0, 1, 2)
    # Three-element sorting network on (key, slot): compare-exchange (0, 1), (1, 2), (0, 1).
    for step in range(3):
        lo = wp.where(step == 1, 1, 0)
        if keys[lo + 1] < keys[lo]:
            key = keys[lo]
            keys[lo] = keys[lo + 1]
            keys[lo + 1] = key
            slot = slots[lo]
            slots[lo] = slots[lo + 1]
            slots[lo + 1] = slot
    total = wp.vec3(0.0)
    max_angle = wp.float32(0.0)
    for k in range(3):
        if keys[k] != INT32_MAX_CONSTANT:
            g = mates[3 * f + slots[k]] // 3
            other = face_normals[g]
            total = total + other
            angle = wp.where(f < g, vector_angle(normal, other), vector_angle(other, normal))
            max_angle = wp.max(max_angle, angle)
    return total, max_angle


@wp.kernel
def face_defective_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    mates: wp.array[wp.int32],
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
    # One thread per face does all of it: the thinness is ``triangles.face_quality``'s
    # ``radius_ratio`` formed from the corners in registers, and the neighbourhood is gathered
    # through the face's halfedge mates (``neighbor_normals``) rather than scattered into per-face
    # tables. ``face_normals`` and ``mates`` are never read when both angle criteria are off, so
    # the wrapper passes ``None`` for them then (the branch is warp-uniform).
    f = wp.int32(wp.tid())
    if min_quality > -1.0:
        v0, v1, v2 = face_vertices(vertices, faces, f)
        if triangle_radius_ratio(v0, v1, v2) < min_quality:
            out_bad[f] = wp.bool(True)
            return
    if max_normal_cos <= -2.0 and max_fold_cos <= -2.0:
        out_bad[f] = wp.bool(False)
        return

    neighbor_sum, max_angle = neighbor_normals(face_normals, mates, f)
    # Direction of the sum of the neighbouring normals: the local consensus. ``wp.normalize`` of a
    # zero vector is zero in Warp, which is how an isolated face -- and a face whose neighbours
    # cancel each other exactly -- ends up with no consensus to disagree with.
    consensus = wp.normalize(neighbor_sum)
    agreement = wp.dot(face_normals[f], consensus)
    if wp.length_sq(consensus) > 0.0 and agreement < max_normal_cos:
        out_bad[f] = wp.bool(True)
        return

    # Fold: some neighbour meets this face at nearly pi (an unsigned dihedral, so the gate is on its
    # cosine) *and* this face is the one facing against its own neighbourhood. Both halves matter --
    # a fold has two faces and only one of them is the mistake, so flagging both would delete a good
    # triangle along with it.
    out_bad[f] = wp.cos(max_angle) < max_fold_cos and agreement < 0.0
