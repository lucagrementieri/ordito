import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels.adjacency import edge_endpoints
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_hook_pair_parity,
    ecl_prehook_pair,
    find_representative,
    find_representative_parity,
    parity_root_word,
)
from ordito.kernels.array import scanned_count
from ordito.kernels.grouping import count_edge_key
from ordito.kernels.halfedge import (
    halfedge_destination,
    halfedge_endpoints,
    halfedge_next,
    mate_twin_defect,
)
from ordito.kernels.predicates import vector_angle
from ordito.kernels.reduce import (
    block_chunk,
    block_chunk_1d,
    commit_block_sum,
    last_ticket_vector,
    store_block_total,
)
from ordito.kernels.triangles import (
    corner_triple,
    face_normals_and_area,
    face_signed_volume,
    face_vertices,
    known_local_corner,
    triangle_radius_ratio,
)

wp.set_module_options({"enable_backward": False})

# ``face_defective``'s disabled fold gate: an angle in radians past pi, which no dihedral exceeds.
FOLD_ANGLE_OFF = wp.float32(4.0)


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


@wp.func
def pair_traversed_opposite(
    faces: wp.array[wp.int32], edges: wp.array2d[wp.int32], h0: wp.int32, h1: wp.int32
) -> wp.bool:
    # Whether the two halfedges of a shared edge run it in opposite directions: the per-pair test
    # of ``emit_sorted_pair_winding`` and ``mate_pair_violated``, which differ in where the pair
    # comes from (the sorted key runs, the halfedge mates).
    _a0, b0 = directed_edge(faces, edges, h0)
    a1, _b1 = directed_edge(faces, edges, h1)
    return b0 == a1


@wp.kernel
def emit_sorted_pair_winding(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_consistent: wp.array[wp.bool],
) -> None:
    # ``adjacency.emit_sorted_face_pairs``' compaction over the scanned exact-pair starts, writing
    # each pair's winding flag where that kernel writes its face pair, so no group table exists.
    i = wp.int32(wp.tid())
    row, count = scanned_count(inclusive, i)
    if count == 0:
        return
    out_consistent[row] = pair_traversed_opposite(faces, edges, order[i], order[i + 1])


@wp.func
def mate_pair_violated(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    mates: wp.array[wp.int32],
    require_pairs: wp.bool,
    require_opposite: wp.bool,
    h: wp.int32,
) -> wp.bool:
    """
    Whether halfedge ``h`` breaks "every edge shared by two faces, traversed both ways".

    With ``require_opposite``, a pair's lower halfedge fails when the pair's two faces traverse
    their edge the same way (``pair_traversed_opposite``, read off the halfedge
    mates); with ``require_pairs`` a halfedge fails when its edge is not shared by exactly two
    faces. Shared by ``mate_pair_winding_violation`` and ``closed_signed_volume``.
    """
    mate = mates[h]
    if mate < 0:
        return require_pairs
    if require_opposite and mate > h:
        return not pair_traversed_opposite(faces, edges, h, mate)
    return False


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

    ``mate_pair_violated`` per halfedge, so a predicate needs no compacted group table and no host
    read of its length. With ``require_pairs`` a halfedge whose edge is not shared by exactly two
    faces also raises the flag.
    """
    if mate_pair_violated(faces, edges, mates, require_pairs, True, wp.int32(wp.tid())):
        out_violation[0] = 1


@wp.kernel(module="unique")
def closed_signed_volume(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    mates: wp.array[wp.int32],
    require_opposite: wp.bool,
    stride: wp.int32,
    out_total: wp.array[wp.vec2],
) -> None:
    """
    Fold the mesh's signed volume from the origin and count the halfedges breaking closedness.

    ``measures.mesh_signed_volume``'s fold -- the same chunks, lane stride and fixed-order commit,
    so component 0 is ``measures.volume`` bit for bit -- with ``mate_pair_violated`` counted over
    each face's three halfedges in component 1 (every edge shared by two faces, and with
    ``require_opposite`` traversed both ways), so a closed-volume test is one launch and one
    readback where the edge test and the volume were a launch and a readback each. Its own module
    (``module="unique"``): the only tiled launch here, at ``TILE_1D``, so the module launched at
    the default width is not built a second time.
    """
    block, lane = wp.tid()
    offset, count = block_chunk_1d(faces.shape[0] // 3, block)
    total = wp.vec2(0.0)
    origin = wp.vec3()
    for k in range(lane, count, wp.block_dim()):
        f = offset + k
        broken = wp.float32(0.0)
        for corner in range(3):
            if mate_pair_violated(faces, edges, mates, True, require_opposite, 3 * f + corner):
                broken += wp.float32(1.0)
        total += wp.vec2(face_signed_volume(vertices, faces, f, origin), broken)
    store_block_total(block, lane, total, out_total, 0, stride, last_ticket_vector)


@wp.func
def edge_forward_in_face(
    faces: wp.array[wp.int32], f: wp.int32, a: wp.int32, b: wp.int32
) -> wp.int32:
    """1 if face ``f`` traverses the shared edge as ``a -> b``, else 0 (``b -> a``)."""
    nxt = halfedge_destination(faces, f * wp.int32(3) + known_local_corner(faces, f, a))
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
def edge_share_violations(
    faces: wp.array[wp.int32],
    base: wp.uint64,
    per: wp.int32,
    global_mask: wp.int32,
    width: wp.int32,
    bound: wp.int32,
    table: wp.array[wp.uint64],
    out_flags: wp.array[wp.int32],
) -> None:
    """
    ``is_edge_manifold``'s one pass: every corner's edge counted into a zeroed edge-key set.

    ``out_flags[0]`` counts the third arrivals on an edge (``grouping.count_edge_key`` reads 2:
    some edge has three faces or more), ``out_flags[1]`` the faces with a corner outside
    ``[0, bound)``, and ``out_flags[2]`` the first arrivals, the unique edges, so with boundary
    edges disallowed the wrapper tests ``2 * unique == 3 * n_faces`` (no edge above two faces and
    the shares summing to the corner count leave every edge at exactly two). No key table, no
    count table beside the set and no pass over its slots. A block owns ``width`` faces
    (``reduce.spanned_blocks_1d``) and commits the three counts once.
    """
    block, lane = wp.tid()
    offset, count = block_chunk(faces.shape[0] // 3, block, width)
    local = wp.vec3i(0, 0, 0)
    for k in range(lane, count, wp.block_dim()):
        f = offset + k
        i0, i1, i2 = corner_triple(faces, f)
        before0 = count_edge_key(table, i0, i1, base, per, global_mask)
        before1 = count_edge_key(table, i1, i2, base, per, global_mask)
        before2 = count_edge_key(table, i2, i0, base, per, global_mask)
        third = wp.where(before0 == 2, 1, 0) + wp.where(before1 == 2, 1, 0)
        third += wp.where(before2 == 2, 1, 0)
        first = wp.where(before0 == 0, 1, 0) + wp.where(before1 == 0, 1, 0)
        first += wp.where(before2 == 0, 1, 0)
        out_of_range = wp.int32(0)
        for c in range(3):
            v = faces[3 * f + c]
            if v < 0 or v >= bound:
                out_of_range = 1
        local += wp.vec3i(third, out_of_range, first)
    commit_block_sum(lane, local, out_flags, 0)


@wp.func
def mate_share_count(mate: wp.int32) -> wp.int32:
    """
    Return the face-share count ``edge_manifold`` asks of a halfedge's edge, from its mate code.

    ``halfedge.halfedge_mates`` says it outright: ``-1`` one, a partner two, ``<= -2`` three or
    more (reported as three, all ``edge_manifold`` distinguishes).
    """
    return wp.where(mate >= 0, wp.int32(2), wp.where(mate == -1, wp.int32(1), wp.int32(3)))


@wp.func
def mate_edge_manifold(mate: wp.int32, allow_boundary: wp.bool) -> wp.bool:
    """Whether a halfedge's undirected edge is edge-manifold, from its mate code alone."""
    # The one decision both mate-driven kernels below make: the mask clears a face on it and the
    # verdict raises a flag.
    return edge_manifold(mate_share_count(mate), allow_boundary)


@wp.func
def face_mates_edge_manifold(
    mates: wp.array[wp.int32], f: wp.int32, allow_boundary: wp.bool
) -> wp.bool:
    """Whether all three edges of face ``f`` are edge-manifold, from its halfedges' mate codes."""
    # The per-face verdict both face-mask kernels write: ``mate_manifold_mask`` here and
    # ``repair.face_mate_manifold_flags``, which also writes it as a ``0`` / ``1`` flag.
    keep = wp.bool(True)
    for k in range(3):
        if not mate_edge_manifold(mates[3 * f + k], allow_boundary):
            keep = False
    return keep


@wp.kernel
def mate_manifold_mask(
    mates: wp.array[wp.int32], allow_boundary: wp.bool, out_mask: wp.array[wp.bool]
) -> None:
    """
    Write every face's edge-manifold flag from its three halfedges' mate codes.

    One thread per face. [`edge_manifold`][ordito.kernels.validation.edge_manifold] only asks
    whether an edge's face-share count is one, two, or more, which the halfedge's mate code
    (``halfedge.halfedge_mates``) says outright: ``-1`` one, a partner two, ``<= -2`` three or
    more. So no unique-edge table, no inverse and no count array is built, and since each thread
    writes its own face's entry, ``out_mask`` arrives unfilled.
    """
    f = wp.int32(wp.tid())
    out_mask[f] = face_mates_edge_manifold(mates, f, allow_boundary)


@wp.kernel
def mate_manifold_violation(
    mates: wp.array[wp.int32], allow_boundary: wp.bool, out_violation: wp.array[wp.int32]
) -> None:
    """
    Raise ``out_violation[0]`` when some halfedge's undirected edge is not edge-manifold.

    ``mate_manifold_mask``'s test reduced to the verdict, for a caller holding the mates: one
    thread per halfedge, and an unsynchronized store because every writer stores the same ``1``.
    """
    h = wp.int32(wp.tid())
    if not mate_edge_manifold(mates[h], allow_boundary):
        out_violation[0] = 1


@wp.func
def corner_link(
    faces: wp.array[wp.int32], f0: wp.int32, f1: wp.int32, v: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """Link the ``v`` corners of edge-adjacent faces ``f0`` and ``f1`` in the corner graph."""
    return wp.int32(3) * f0 + known_local_corner(faces, f0, v), wp.int32(
        3
    ) * f1 + known_local_corner(faces, f1, v)


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
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], oriented: wp.bool, h: wp.int32
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

    ``oriented`` (warp-uniform) joins corners only across a pair that runs the edge both ways, its
    halfedge twin (``halfedge.mate_twin_defect``): ``repair.split_non_manifold_vertices``' graph,
    where an edge wound the same way by both faces separates them. The link is then positional,
    corner ``h`` with the twin's next corner, the two corners at ``h``'s origin, so a face repeating
    a vertex is never searched for it.
    """
    mate = mates[h]
    if mate < 0:
        return h, h, wp.bool(True)
    if oriented:
        twin, _defect = mate_twin_defect(faces, mate, h)
        if twin < 0:
            return h, h, wp.bool(False)
        return h, halfedge_next(twin), wp.bool(False)
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
    oriented: wp.bool,
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
        c0, c1, unpaired = mate_corner_link(faces, mates, oriented, h)
        if unpaired:
            unpaired_any = True
        p0, p1, p2 = lower_face_corner_parents(p0, p1, p2, base, c0, c1)
        mate = mates[h]
        if mate >= 0:
            m0, m1, _unpaired = mate_corner_link(faces, mates, oriented, mate)
            p0, p1, p2 = lower_face_corner_parents(p0, p1, p2, base, m0, m1)
    if unpaired_any and require_pairs:
        out_violation[0] = 1
    out_parents[base] = p0
    out_parents[base + 1] = p1
    out_parents[base + 2] = p2


@wp.kernel
def mate_corner_hook(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    oriented: wp.bool,
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_hook_edges`` over the same edges, after ``mate_corner_prehook``.
    c0, c1, _unpaired = mate_corner_link(faces, mates, oriented, wp.int32(wp.tid()))
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
    ``mate_hook_parity`` and ``mate_hook_parity_flagged``, which form the edge in the thread
    rather than read a ``(3 n_faces, 2)`` table and its signs.
    """
    mate = mates[h]
    g0 = h // 3
    g1 = mate // 3
    # A face paired with itself (``(a, b, a)``: ``a -> b`` and ``b -> a``) runs that edge both
    # ways, which any flip satisfies, so it is a sign-``0`` self-loop like an unpaired halfedge.
    if mate <= h or g0 == g1:
        return g0, g0, wp.int32(0)
    f0 = wp.min(g0, g1)
    f1 = wp.max(g0, g1)
    a, b = edge_endpoints(faces, h)
    return f0, f1, pair_flip_sign(faces, f0, f1, a, b)


@wp.kernel
def mate_prehook_parity(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], out_words: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_init_parent_parity`` with each face pre-hooked to its smallest
    # edge neighbour, at the sign of the pair between them (``mate_signed_edge`` of the pair's
    # lower halfedge), the pre-hook ``ecl_init_parent`` makes and the parity form could not without
    # a signed CSR: a face's three halfedge mates are that CSR row. One thread per face writes its
    # own word, so no identity forest is laid down first; a face with no smaller neighbour is its
    # own root at parity 0. Every link taken is one ``mate_hook_parity`` also hooks, so the forest
    # it starts from is consistent with the signed edges.
    f = wp.int32(wp.tid())
    parent = f
    parity = wp.int32(0)
    for k in range(3):
        h = 3 * f + k
        mate = mates[h]
        if mate >= 0:
            g = mate // 3
            if g < parent:
                _f0, _f1, sign = mate_signed_edge(faces, mates, wp.min(h, mate))
                parent = g
                parity = sign
    out_words[f] = (parent << 1) | parity


@wp.kernel
def mate_hook_parity(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], words: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_hook_parity`` over ``mate_signed_edge``'s edges.
    f0, f1, sign = mate_signed_edge(faces, mates, wp.int32(wp.tid()))
    ecl_hook_pair_parity(words, f0, f1, sign)


@wp.kernel
def mate_hook_parity_flagged(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    words: wp.array[wp.int32],
    out_conflict: wp.array[wp.int32],
) -> None:
    # ``mate_hook_parity`` raising ``out_conflict[0]`` on a pair the forest contradicts
    # (``connected_components.ecl_hook_parity_flagged``'s test): whether the mesh is orientable,
    # with no flatten and no re-test of the pairs.
    f0, f1, sign = mate_signed_edge(faces, mates, wp.int32(wp.tid()))
    if ecl_hook_pair_parity(words, f0, f1, sign) != 0:
        out_conflict[0] = wp.int32(1)


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
def stored_or_face_normal(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    f: wp.int32,
) -> wp.vec3:
    # Face ``f``'s unit normal: the caller's table entry when ``face_normals`` is given, else formed
    # from the corners by ``triangles.face_normals_and_area``, the value
    # ``triangles.face_normals_and_areas`` would have stored, so the bad-face criteria need no
    # normal table written by a launch of its own. ``face_normals`` is a null descriptor (shape 0)
    # when the caller passed none, so the branch is warp-uniform.
    if face_normals.shape[0] > 0:
        return face_normals[f]
    normal, _area = face_normals_and_area(vertices, faces, f)
    return normal


@wp.func
def neighbor_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    mates: wp.array[wp.int32],
    f: wp.int32,
) -> tuple[wp.vec3, wp.vec3, wp.float32]:
    # Both per-face quantities the bad-face criteria need, gathered through the face's three
    # halfedge mates: the sum of the neighbouring face normals (whose direction is the local
    # consensus) and the sharpest dihedral angle to any neighbour (which is what a fold looks like).
    # The dihedral angle is ``adjacency.face_adjacency_angles``' -- ``vector_angle`` of the pair's
    # two normals, lower face first.
    #
    # The sum is formed in ascending pair-key order (each pair keyed by its lower halfedge), the
    # order a serial scatter over the pairs adds them in, so the result does not depend on which
    # corner of the face a neighbour sits at. Returns the face's own normal first, which every
    # criterion compares against. Normals come from ``stored_or_face_normal``.
    normal = stored_or_face_normal(vertices, faces, face_normals, f)
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
            other = stored_or_face_normal(vertices, faces, face_normals, g)
            total = total + other
            angle = wp.where(f < g, vector_angle(normal, other), vector_angle(other, normal))
            max_angle = wp.max(max_angle, angle)
    return normal, total, max_angle


@wp.func
def face_defective(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    mates: wp.array[wp.int32],
    f: wp.int32,
    min_quality: wp.float32,
    max_normal_cos: wp.float32,
    max_pair_angle: wp.float32,
    max_fold_angle: wp.float32,
) -> wp.bool:
    # Is face ``f`` bad: too thin, too far from its neighbourhood's consensus normal (or, the
    # pairwise form, from any one neighbour's), or folded back onto its own ring? Each criterion is
    # disabled by passing a cosine of -2 / a quality of -1 / an angle of ``FOLD_ANGLE_OFF``
    # (radians, past pi), which no real value can reach, so the gates compose without a separate
    # flag argument. The decision rule behind ``face_defective_mask`` and
    # ``repair.remove_folded_faces``' marking pass (``repair.mark_unfolded_faces``).
    #
    # The thinness is ``triangles.face_quality``'s ``radius_ratio`` formed from the corners in
    # registers, and the neighbourhood is gathered through the face's halfedge mates
    # (``neighbor_normals``) rather than scattered into per-face tables. ``face_normals`` and
    # ``mates`` are never read when both angle criteria are off, so a caller passes ``None`` for
    # them then (the branch is warp-uniform); with a criterion on, ``face_normals`` may still be
    # ``None``, and each normal is then formed from the corners where it is read.
    if min_quality > -1.0:
        v0, v1, v2 = face_vertices(vertices, faces, f)
        if triangle_radius_ratio(v0, v1, v2) < min_quality:
            return True
    if (
        max_normal_cos <= -2.0
        and max_pair_angle >= FOLD_ANGLE_OFF
        and max_fold_angle >= FOLD_ANGLE_OFF
    ):
        return False

    normal, neighbor_sum, max_angle = neighbor_normals(vertices, faces, face_normals, mates, f)
    # Direction of the sum of the neighbouring normals: the local consensus. ``wp.normalize`` of a
    # zero vector is zero in Warp, which is how an isolated face -- and a face whose neighbours
    # cancel each other exactly -- ends up with no consensus to disagree with.
    consensus = wp.normalize(neighbor_sum)
    agreement = wp.dot(normal, consensus)
    if wp.length_sq(consensus) > 0.0 and agreement < max_normal_cos:
        return True
    # Pairwise: the sharpest dihedral to any one neighbour, the test MeshLab's ``nfratio`` makes.
    # It flags both faces of every such edge, so a reversed face drags its neighbours with it.
    if max_angle > max_pair_angle:
        return True

    # Fold: some neighbour meets this face at nearly pi *and* this face is the one facing against
    # its own neighbourhood. Both halves matter -- a fold has two faces and only one of them is the
    # mistake, so flagging both would delete a good triangle along with it. The gate is on the angle
    # itself, strict, not on its cosine: the cosine is flat at pi, so in ``float32`` every threshold
    # above ~179.98 degrees rounded to ``cos == -1`` and an exactly folded-back pair (``atan2``
    # gives pi) was never flagged, where MeshLab's ``folded_faces_angle_threshold`` flags it.
    return max_angle > max_fold_angle and agreement < 0.0


@wp.kernel
def face_defective_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    mates: wp.array[wp.int32],
    min_quality: wp.float32,
    max_normal_cos: wp.float32,
    max_pair_angle: wp.float32,
    max_fold_angle: wp.float32,
    out_bad: wp.array[wp.bool],
) -> None:
    # One thread per face does all of ``face_defective``.
    f = wp.int32(wp.tid())
    out_bad[f] = face_defective(
        vertices,
        faces,
        face_normals,
        mates,
        f,
        min_quality,
        max_normal_cos,
        max_pair_angle,
        max_fold_angle,
    )
