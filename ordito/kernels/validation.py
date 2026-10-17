import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels.adjacency import edge_endpoints, write_face_edge_keys
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_hook_pair_parity,
    ecl_prehook_pair,
    find_representative,
    find_representative_parity,
    parity_root_word,
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


@wp.func
def lower_face_corner_parents(
    p0: wp.int32, p1: wp.int32, p2: wp.int32, base: wp.int32, c0: wp.int32, c1: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # ``ecl_prehook_pair``'s ``atomic_min`` of link ``(c0, c1)``, kept in the registers of the face
    # whose corners are ``base`` to ``base + 2``: the larger end, when it is one of them, points at
    # the smaller if that is lower than where it points already.
    hi = wp.max(c0, c1)
    lo = wp.min(c0, c1)
    if hi == base:
        p0 = wp.min(p0, lo)
    elif hi == base + 1:
        p1 = wp.min(p1, lo)
    elif hi == base + 2:
        p2 = wp.min(p2, lo)
    return p0, p1, p2


@wp.kernel
def mate_corner_prehook(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    require_pairs: wp.bool,
    out_parents: wp.array[wp.int32],
    out_violation: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_init_parent_edges`` over ``mate_corner_link``'s edges, from no
    # identity forest: one thread per face writes its three corners' parents. A link lies between
    # the faces of a halfedge and of its mate, so the links touching this face's corners are those
    # of its three halfedges and of their mates (every one, as ``mates[mates[h]] == h``); folding
    # each into the corner it lowers gives exactly the ``atomic_min``s an identity forest would
    # have taken, with no ``arange`` launch before it. Every link folded is one ``mate_corner_hook``
    # also hooks, so the forest stays valid whatever ``mates`` holds. With ``require_pairs`` an
    # edge not shared by exactly two faces raises ``out_violation[0]``: the edge half of
    # ``is_watertight``, folded into the same pass; without it the flag is never indexed.
    f = wp.int32(wp.tid())
    base = wp.int32(3) * f
    p0 = base
    p1 = base + 1
    p2 = base + 2
    unpaired_any = wp.bool(False)
    for k in range(3):
        h = base + k
        c0, c1, unpaired = mate_corner_link(faces, mates, h)
        if unpaired:
            unpaired_any = True
        p0, p1, p2 = lower_face_corner_parents(p0, p1, p2, base, c0, c1)
        mate = mates[h]
        if mate >= 0:
            m0, m1, _unpaired = mate_corner_link(faces, mates, mate)
            p0, p1, p2 = lower_face_corner_parents(p0, p1, p2, base, m0, m1)
    if unpaired_any and require_pairs:
        out_violation[0] = 1
    out_parents[base] = p0
    out_parents[base + 1] = p1
    out_parents[base + 2] = p2


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
def face_pair_signs_and_roots(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    n_faces: wp.int32,
    out_sign: wp.array[wp.int32],
    out_words: wp.array[wp.int32],
) -> None:
    """
    Z2 flip bit (``pair_flip_sign``) of every face-adjacency row, and the parity forest's roots.

    The signed face graph is the adjacency table itself, whose rows are already ascending face
    pairs, so only the signs are written. Launched over ``max(m, n_faces)``: the parity
    union-find's identity words (``connected_components.ecl_init_parent_parity``) are laid down
    by the same launch rather than one of their own.
    """
    i = wp.int32(wp.tid())
    if i < adjacency.shape[0]:
        out_sign[i] = pair_flip_sign(
            faces, adjacency[i, 0], adjacency[i, 1], adjacency_edges[i, 0], adjacency_edges[i, 1]
        )
    if i < n_faces:
        out_words[i] = parity_root_word(i)


@wp.func
def mate_signed_edge(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], h: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32]:
    """
    ``face_pair_signs_and_roots``' signed row for halfedge ``h``, with no adjacency table.

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


@wp.func
def face_parity_flip(words: wp.array[wp.int32], v: wp.int32) -> wp.bool:
    # Does face ``v`` flip, read off the parity union-find's ``words``: a parity is ``0`` or ``1``,
    # so ``parity != 0`` is ``orient > 0``. Shared by ``flatten_parity_flip_mask`` and
    # ``repair.flip_faces_by_parity``, which applies the flip where this one stores it.
    _root, parity = find_representative_parity(words, v)
    return parity != 0


@wp.kernel
def flatten_parity_flip_mask(words: wp.array[wp.int32], out_flip: wp.array[wp.bool]) -> None:
    # ``connected_components.ecl_flatten_parity`` with the parity written as the flip mask
    # ``face_flip_mask`` returns and no label, for the caller that wants the mask alone rather than
    # the ``int32`` bits.
    v = wp.int32(wp.tid())
    out_flip[v] = face_parity_flip(words, v)


@wp.func
def signed_edge_violated(
    orient: wp.array[wp.int32], f0: wp.int32, f1: wp.int32, sign: wp.int32
) -> wp.int32:
    # ``1`` when the flip bits of ``f0`` and ``f1`` break their signed edge's constraint, else
    # ``0``. Shared by ``mate_orientation_conflict`` and ``signed_edge_orientation_conflict``,
    # which differ only in where the edge comes from.
    return wp.where(((orient[f0] + orient[f1]) & wp.int32(1)) != sign, wp.int32(1), wp.int32(0))


@wp.kernel
def mate_orientation_conflict(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    out_conflict: wp.array[wp.int32],
) -> None:
    # Flag any signed edge of ``mate_signed_edge``'s whose faces violate its flip constraint.
    f0, f1, sign = mate_signed_edge(faces, mates, wp.int32(wp.tid()))
    if signed_edge_violated(orient, f0, f1, sign) != 0:
        out_conflict[0] = wp.int32(1)


@wp.kernel
def signed_edge_orientation_conflict(
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    out_conflict: wp.array[wp.int32],
) -> None:
    # ``mate_orientation_conflict`` over a signed edge list (``face_pair_signs_and_roots``'), for
    # ``face_orientation_bits`` under the deterministic setting.
    e = wp.int32(wp.tid())
    if signed_edge_violated(orient, edges[e, 0], edges[e, 1], signs[e]) != 0:
        out_conflict[0] = wp.int32(1)


# A spanning forest of a signed face graph that is a function of the input alone, for the
# orientation under the deterministic setting on a non-orientable mesh: ``face_flip_mask``'s over
# the mate pairs, ``face_orientation_bits``' over its signed edge list. There the parity
# union-find's answer is whichever spanning forest its racing hooks built: no potential exists,
# and the edges left out of the forest are where the contradiction lands. Every edge keyed by its
# index (the mate pairs by their lower halfedge, the list's rows by row; the keys are distinct),
# the minimum spanning forest is unique, and Boruvka's rounds find it with integer atomics only,
# which are exact: ``signed_forest_candidates`` takes each component's lightest outgoing edge by
# ``atomic_min``, ``signed_forest_hook`` adds it to the forest and hooks the component, and
# ``forest_flatten`` relabels. The parity union-find over the forest's edges alone
# (``signed_forest_hook_parity``) is then order-free: each tree's root is its smallest face at
# parity ``0`` and every other face's parity is its tree path's. On a consistent component that
# is the fast path's answer, whatever the tree.
#
# Each kernel reads its edge through ``forest_signed_edge``: from ``edges`` / ``signs`` when they
# are non-empty, else from ``mate_signed_edge`` (the branch is warp-uniform). That is all the two
# graphs differ by.


@wp.func
def forest_signed_edge(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    e: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # Signed edge ``e`` of the graph the forest spans: row ``e`` of a non-empty edge list, else
    # halfedge ``e``'s ``mate_signed_edge`` (a self-loop for all but a pair's lower halfedge).
    if edges.shape[0] > 0:
        return edges[e, 0], edges[e, 1], signs[e]
    return mate_signed_edge(faces, mates, e)


@wp.kernel
def signed_forest_candidates(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    best: wp.array[wp.int32],
) -> None:
    # One thread per edge: an edge joining two components offers its key to both.
    e = wp.int32(wp.tid())
    f0, f1, _sign = forest_signed_edge(faces, mates, edges, signs, e)
    if f0 == f1:
        return
    c0 = labels[f0]
    c1 = labels[f1]
    if c0 != c1:
        wp.atomic_min(best, c0, e)
        wp.atomic_min(best, c1, e)


@wp.kernel
def signed_forest_hook(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    best: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    in_forest: wp.array[wp.int32],
    hooked: wp.array[wp.int32],
) -> None:
    # One thread per face; a component root with an outgoing edge takes it into the forest and
    # hooks onto the component across it. Two roots that chose the same edge (the only cycle
    # distinct keys allow) would hook onto each other, so the smaller stays the root.
    r = wp.int32(wp.tid())
    if labels[r] != r:
        return
    e = best[r]
    if e == INT32_MAX_CONSTANT:
        return
    f0, f1, _sign = forest_signed_edge(faces, mates, edges, signs, e)
    other = labels[f0]
    if other == r:
        other = labels[f1]
    in_forest[e] = wp.int32(1)
    if best[other] == e and r < other:
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
def signed_forest_hook_parity(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    in_forest: wp.array[wp.int32],
    words: wp.array[wp.int32],
) -> None:
    # The parity hook over the forest's edges only; every other edge is skipped.
    e = wp.int32(wp.tid())
    if in_forest[e] != 0:
        f0, f1, sign = forest_signed_edge(faces, mates, edges, signs, e)
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


@wp.func
def face_defective(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    mates: wp.array[wp.int32],
    f: wp.int32,
    min_quality: wp.float32,
    max_normal_cos: wp.float32,
    max_fold_cos: wp.float32,
) -> wp.bool:
    # Is face ``f`` bad: too thin, too far from its neighbourhood's consensus normal, or folded
    # back onto its own ring? Each criterion is disabled by passing a cosine of -2 / a quality of
    # -1, which no real value can reach, so the three gates compose without a separate flag
    # argument. The decision rule behind ``face_defective_mask`` and ``repair.remove_folded_faces``'
    # marking pass (``repair.mark_unfolded_faces``).
    #
    # The thinness is ``triangles.face_quality``'s ``radius_ratio`` formed from the corners in
    # registers, and the neighbourhood is gathered through the face's halfedge mates
    # (``neighbor_normals``) rather than scattered into per-face tables. ``face_normals`` and
    # ``mates`` are never read when both angle criteria are off, so a caller passes ``None`` for
    # them then (the branch is warp-uniform).
    if min_quality > -1.0:
        v0, v1, v2 = face_vertices(vertices, faces, f)
        if triangle_radius_ratio(v0, v1, v2) < min_quality:
            return True
    if max_normal_cos <= -2.0 and max_fold_cos <= -2.0:
        return False

    neighbor_sum, max_angle = neighbor_normals(face_normals, mates, f)
    # Direction of the sum of the neighbouring normals: the local consensus. ``wp.normalize`` of a
    # zero vector is zero in Warp, which is how an isolated face -- and a face whose neighbours
    # cancel each other exactly -- ends up with no consensus to disagree with.
    consensus = wp.normalize(neighbor_sum)
    agreement = wp.dot(face_normals[f], consensus)
    if wp.length_sq(consensus) > 0.0 and agreement < max_normal_cos:
        return True

    # Fold: some neighbour meets this face at nearly pi (an unsigned dihedral, so the gate is on its
    # cosine) *and* this face is the one facing against its own neighbourhood. Both halves matter --
    # a fold has two faces and only one of them is the mistake, so flagging both would delete a good
    # triangle along with it.
    return wp.cos(max_angle) < max_fold_cos and agreement < 0.0


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
    # One thread per face does all of ``face_defective``.
    f = wp.int32(wp.tid())
    out_bad[f] = face_defective(
        vertices, faces, face_normals, mates, f, min_quality, max_normal_cos, max_fold_cos
    )
