from typing import Any

import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels.adjacency import edge_endpoints
from ordito.kernels.array import (
    launched_set_bit,
    loop_owner,
    loop_rim_edge,
    pack_ranked_key,
    scanned_count,
    store_counted_ballot_bit,
)
from ordito.kernels.halfedge import (
    HALFEDGE_MATE_BOUNDARY,
    halfedge_endpoints,
    key_ordered_halfedge,
    next_boundary_halfedge_from_mates,
    sorted_run_mate,
)
from ordito.kernels.predicates import newell_term
from ordito.kernels.reduce import block_max, block_min, block_sum

wp.set_module_options({"enable_backward": False})

# Lanes per loop of the fixed-order segmented sums (``loop_perimeters_ordered`` and the
# ``*_ordered`` kernels after it). A loop is one block, so the width only has to cover a typical
# rim; a long one is walked ``LOOP_SUM_BLOCK`` edges at a time.
LOOP_SUM_BLOCK = 64

# Every boundary query here reads the halfedge mates (``halfedge.halfedge_mates``): a boundary
# edge is a halfedge with no mate, which names the halfedge -- row ``h`` of
# ``edges.faces_to_edges`` -- so the edge's endpoints (sorted or directed) are read straight off
# ``faces`` and no ``(3F, 2)`` edge table is ever built. The kernels below write the bits the
# caller ranks and read the verdict back out of the ranks.


@wp.func
def boundary_halfedge_pair(
    faces: wp.array[wp.int32], sort_pair: wp.bool, h: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Halfedge ``h``'s row of ``edges.faces_to_edges``: its directed pair, ascending when
    # ``sort_pair`` -- read off ``faces``, so no ``(3F, 2)`` table is built or read.
    if sort_pair:
        return edge_endpoints(faces, h)
    return halfedge_endpoints(faces, h)


@wp.func
def is_boundary_mate(mate: wp.int32) -> wp.bool:
    # A halfedge alone on its edge: the boundary rule every kernel here reads off the mates.
    return mate == HALFEDGE_MATE_BOUNDARY


@wp.func
def flag_boundary_halfedge(
    boundary: wp.bool,
    h: wp.int32,
    i: wp.int32,
    faces: wp.array[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_defects: wp.array[wp.int32],
    out_words: wp.array[wp.uint32],
    out_word_counts: wp.array[wp.int32],
):
    # Bit ``i`` of the zeroed ``out_words`` set where a halfedge spans its edge alone -- a boundary
    # edge -- bit ``i`` standing for halfedge ``key_ordered_halfedge(key_order, i)``, and each
    # word's population added into the zeroed ``out_word_counts``, which the caller scans in place
    # into the words' ranks. Every thread of the launch reaches the ballot
    # (``store_counted_ballot_bit``), thread ``i`` holding bit ``i``. With ``out_degrees`` (else
    # ``None``), ``boundary_loops_with_offsets``' degree census of the directed rows
    # (``halfedge_endpoints``) rides in the same launch, its two defect bits stamped into the
    # zeroed ``out_defects[0]`` and ``out_defects[1]``, which the caller lays right after the
    # ranks so the total and both bits come back in one readback.
    #
    # Degree column 0: how many boundary edges *leave* each vertex. Column 1: how many touch it at
    # all. One out-edge and two incidences is the well-behaved case. Two out-edges is the seam of a
    # non-orientable surface, where ``succ[tail] = head`` silently drops an edge (bit 0). Four
    # incidences is a pinch point, where two loops meet and no 2-regular walk over vertices exists
    # at all (bit 1) -- the two are different defects with different walks: the seam's is
    # undirected, the pinch's is over halfedges (``boundary_halfedge_successors``).
    #
    # The bits are stamped here rather than by a second pass over the degree table:
    # ``wp.atomic_add`` returns the value the slot held *before* the increment, so the thread that
    # pushes a vertex past the threshold is the one that knows it. Boundary vertices are the only
    # ones whose degrees are ever non-zero, so no pass over the vertices is needed.
    store_counted_ballot_bit(out_words, out_word_counts, i, boundary)
    if boundary and out_degrees.shape[0] > 0:
        tail, head = halfedge_endpoints(faces, h)
        if wp.atomic_add(out_degrees, tail, 0, 1) >= 1:
            out_defects[0] = 1
        if wp.atomic_add(out_degrees, tail, 1, 1) >= 2:
            out_defects[1] = 1
        if wp.atomic_add(out_degrees, head, 1, 1) >= 2:
            out_defects[1] = 1


@wp.kernel
def mark_boundary_halfedges(
    mates: wp.array[wp.int32],
    key_order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_defects: wp.array[wp.int32],
    out_words: wp.array[wp.uint32],
    out_word_counts: wp.array[wp.int32],
) -> None:
    # ``flag_boundary_halfedge`` over a mate table the caller holds.
    i = wp.int32(wp.tid())
    h = key_ordered_halfedge(key_order, i)
    flag_boundary_halfedge(
        is_boundary_mate(mates[h]),
        h,
        i,
        faces,
        out_degrees,
        out_defects,
        out_words,
        out_word_counts,
    )


@wp.kernel
def mark_sorted_boundary_halfedges(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_defects: wp.array[wp.int32],
    out_mates: wp.array[wp.int32],
    out_words: wp.array[wp.uint32],
    out_word_counts: wp.array[wp.int32],
) -> None:
    # ``mark_boundary_halfedges`` straight from the edge-key sort, one thread per sorted position:
    # the mate ``halfedge.sorted_halfedge_mates`` would store is formed here (``sorted_run_mate``),
    # stored for the module's later reads, and flagged in the same thread, so the pairing and the
    # flagging are one launch. The two kernels differ only in where the mate code comes from.
    i = wp.int32(wp.tid())
    h = order[i]
    mate = sorted_run_mate(sorted_keys, order, i)
    out_mates[h] = mate
    flag_boundary_halfedge(
        is_boundary_mate(mate), h, i, faces, out_degrees, out_defects, out_words, out_word_counts
    )


@wp.func
def store_boundary_row(
    faces: wp.array[wp.int32],
    sort_pair: wp.bool,
    h: wp.int32,
    row: wp.int32,
    out_edges: wp.array2d[wp.int32],
):
    # Boundary halfedge ``h``'s ``boundary_halfedge_pair`` as row ``row`` of ``out_edges``.
    a, b = boundary_halfedge_pair(faces, sort_pair, h)
    out_edges[row, 0] = a
    out_edges[row, 1] = b


@wp.kernel
def emit_boundary_edges(
    halfedges: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    sort_pair: wp.bool,
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One thread per boundary halfedge, already in ascending key order
    # (``halfedge.key_ordered_halfedges``) -- the order ``grouping.group`` emitted the rows in --
    # each row read from ``faces`` (see ``boundary_halfedge_pair``).
    g = wp.int32(wp.tid())
    store_boundary_row(faces, sort_pair, halfedges[g], g, out_edges)


@wp.kernel
def emit_key_ordered_boundary_edges(
    words: wp.array[wp.uint32],
    word_ranks: wp.array[wp.int32],
    by_rank: wp.bool,
    key_order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    sort_pair: wp.bool,
    out_edges: wp.array2d[wp.int32],
) -> None:
    # ``emit_boundary_edges`` over ``halfedge.key_ordered_halfedges``' rank compaction in one
    # launch, for bits ranked in the key sort's order (``mark_boundary_halfedges`` given its
    # ``key_order``): over the set bits (``launched_set_bit``), each halfedge's row stored at its
    # rank, its row of the ascending-key table. The two kernels differ only in where a row's
    # halfedge and slot come from.
    j, slot, flagged = launched_set_bit(words, word_ranks, wp.int32(wp.tid()), by_rank)
    if flagged:
        store_boundary_row(faces, sort_pair, key_ordered_halfedge(key_order, j), slot, out_edges)


@wp.func
def boundary_halfedge_endpoints(
    mates: wp.array[wp.int32], faces: wp.array[wp.int32], h: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Halfedge ``h``'s two endpoints when it is a boundary edge, else ``(-1, -1)``.
    if not is_boundary_mate(mates[h]):
        return -1, -1
    return halfedge_endpoints(faces, h)


@wp.kernel
def mark_boundary_vertices(
    mates: wp.array[wp.int32], faces: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # Flag both endpoints of every boundary edge in a zeroed per-vertex ``int32`` array, which the
    # caller scans in place: the sorted unique boundary vertices then come out of the scan's steps
    # with no edge list and no ``unique_1d``. Concurrent writers all store 1, so no atomic.
    h = wp.int32(wp.tid())
    a, b = boundary_halfedge_endpoints(mates, faces, h)
    if a >= 0:
        out_flags[a] = 1
        out_flags[b] = 1


@wp.kernel
def mark_sorted_boundary_vertices(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``mark_boundary_vertices`` straight from the edge-key sort, one thread per sorted position: a
    # run of one key is a boundary halfedge (``sorted_run_mate``), so no mate table is stored.
    i = wp.int32(wp.tid())
    if is_boundary_mate(sorted_run_mate(sorted_keys, order, i)):
        a, b = halfedge_endpoints(faces, order[i])
        out_flags[a] = 1
        out_flags[b] = 1


@wp.kernel
def clear_boundary_vertices(
    mates: wp.array[wp.int32], faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # ``mark_boundary_vertices``' endpoints cleared in a vertex mask instead: the mask less the
    # boundary vertices, with no scan, no index list and no readback.
    h = wp.int32(wp.tid())
    a, b = boundary_halfedge_endpoints(mates, faces, h)
    if a >= 0:
        out_mask[a] = False
        out_mask[b] = False


@wp.kernel
def set_boundary_vertices(
    mates: wp.array[wp.int32], faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # ``clear_boundary_vertices``' dual: the boundary vertices set in a ``False``-filled mask, the
    # fixed set ``smoothing._dirichlet_state`` pins, with no scan, no index list and no readback.
    h = wp.int32(wp.tid())
    a, b = boundary_halfedge_endpoints(mates, faces, h)
    if a >= 0:
        out_mask[a] = True
        out_mask[b] = True


@wp.kernel
def zero_at_boundary_vertices(
    mates: wp.array[wp.int32], faces: wp.array[wp.int32], out_values: wp.array[wp.float64]
) -> None:
    # ``out_values`` zeroed in place at both endpoints of every boundary halfedge
    # (``energies._zero_at_boundary``); a closed mesh pays one launch that writes nothing.
    h = wp.int32(wp.tid())
    a, b = boundary_halfedge_endpoints(mates, faces, h)
    if a >= 0:
        out_values[a] = wp.float64(0.0)
        out_values[b] = wp.float64(0.0)


@wp.func
def ear_interior_corner(mates: wp.array[wp.int32], f: wp.int32) -> wp.int32:
    # The local index of face ``f``'s one interior edge when exactly two of its three edges are
    # boundary edges -- an ear -- else ``-1``.
    b0 = is_boundary_mate(mates[3 * f])
    b1 = is_boundary_mate(mates[3 * f + 1])
    b2 = is_boundary_mate(mates[3 * f + 2])
    if wp.int32(b0) + wp.int32(b1) + wp.int32(b2) != 2:
        return -1
    if not b0:
        return 0
    if not b1:
        return 1
    return 2


@wp.kernel
def mark_ears(mates: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    # 0/1 per face: is it an ear? The ``int32`` flag the caller scans in place.
    f = wp.int32(wp.tid())
    out_flags[f] = wp.where(ear_interior_corner(mates, f) >= 0, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_ears(
    mates: wp.array[wp.int32],
    inclusive: wp.array[wp.int32],
    out_ear: wp.array[wp.int32],
    out_ear_opp: wp.array[wp.int32],
) -> None:
    # Where ``mark_ears``' in-place scan steps, write the ear and the corner opposite its interior
    # edge at its rank: ascending face order on both devices, with no atomic cursor and no trim
    # copy. Interior edge ``i`` runs from corner ``i`` to ``i + 1``, so the corner facing it is
    # ``i + 2`` (libigl's ``ear_opp`` numbering).
    f = wp.int32(wp.tid())
    slot, flag = scanned_count(inclusive, f)
    if flag != 0:
        out_ear[slot] = f
        out_ear_opp[slot] = (ear_interior_corner(mates, f) + 2) % 3


@wp.kernel
def scatter_boundary_neighbors(
    boundary_edges: wp.array2d[wp.int32],
    slot_count: wp.array[wp.int32],
    out_neighbors: wp.array2d[wp.int32],
) -> None:
    # Fill each boundary vertex's two neighbour slots from the *undirected* boundary edges. Every
    # boundary vertex of a manifold boundary has exactly two, so the atomic counter never exceeds
    # 2 -- a third increment would mean a pinch point and is dropped rather than corrupting memory.
    e = wp.int32(wp.tid())
    a = boundary_edges[e, 0]
    b = boundary_edges[e, 1]
    slot_a = wp.atomic_add(slot_count, a, 1)
    if slot_a < 2:
        out_neighbors[a, slot_a] = b
    slot_b = wp.atomic_add(slot_count, b, 1)
    if slot_b < 2:
        out_neighbors[b, slot_b] = a


@wp.func
def sorted_boundary_neighbors(
    neighbors: wp.array2d[wp.int32], v: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Vertex ``v``'s two boundary-neighbour slots in ascending order, so the dart numbering below
    # does not depend on the order the scatter's atomics happened to run in. This is what makes the
    # non-orientable answer reproducible: with no face winding to follow, slot 0 is the smaller
    # neighbour by definition. Read sorted rather than sorted in place by a pass of its own; a lone
    # neighbour (the other slot ``-1``) stays in slot 0.
    first = neighbors[v, 0]
    second = neighbors[v, 1]
    if first >= 0 and second >= 0 and second < first:
        first, second = second, first
    return first, second


@wp.kernel
def dart_successors(
    boundary_edges: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    out_tails: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
) -> None:
    # One successor per *dart*, where dart ``2 * v + s`` means "at vertex v, arrived from
    # neighbour slot s". Its successor leaves by the other slot: the next vertex is
    # ``w = neighbors[v, 1 - s]``, and the arriving slot at w is whichever of w's two slots holds
    # v. Every dart therefore has exactly one out-edge -- which is the property the directed
    # boundary edges lose on a non-orientable surface, and the whole reason this path exists.
    #
    # One thread per boundary edge end: end ``v`` of edge ``(v, u)`` is the dart that arrived at
    # ``v`` from ``u``, so the edge ends are the darts, each once, with no list of boundary
    # vertices to build first. ``out_tails`` is that dart list, in edge order; ranking is by dart
    # index, so the order is free.
    e, end = wp.tid()
    v = boundary_edges[e, end]
    u = boundary_edges[e, 1 - end]
    slot0, slot1 = sorted_boundary_neighbors(neighbors, v)
    s = wp.where(slot0 == u, wp.int32(0), wp.int32(1))
    w = wp.where(s == 0, slot1, slot0)
    dart = 2 * v + s
    out_tails[2 * e + end] = dart
    if w < 0:
        # ``v`` has one boundary edge (a non-manifold edge ends the rim there): a chain end, which
        # the ranking drops.
        out_next[dart] = -1
    else:
        w_slot0, _w_slot1 = sorted_boundary_neighbors(neighbors, w)
        arriving = wp.where(w_slot0 == v, wp.int32(0), wp.int32(1))
        out_next[dart] = 2 * w + arriving


@wp.kernel
def emit_boundary_successors(
    words: wp.array[wp.uint32],
    word_ranks: wp.array[wp.int32],
    by_rank: wp.bool,
    key_order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    out_tails: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
) -> None:
    # The boundary as a successor graph, straight off ``mark_boundary_halfedges``' ranked bits
    # with no edge rows in between: the ranked nodes in ``out_tails`` and their successors in the
    # node-sized ``out_next``. Without ``mates`` (``None``) the nodes are the vertices, the
    # directed rows' tails (``halfedge_endpoints``). With them they are the boundary
    # *halfedges*, each followed by the boundary halfedge leaving its tip in its own sector
    # (``next_boundary_halfedge_from_mates``, each twin formed from the mates as the rotation
    # reads it, so no twin table is built for the few halfedges walked): over vertices a pinch
    # point has two successors, over halfedges every node has one. A fan that does not close (a
    # mesh that is not edge-manifold) maps to a self-loop, so the walk stays bounded. The ranking
    # names each cycle by its smallest node, so the order of ``out_tails`` is free. Launched over
    # the set bits (``launched_set_bit``): a thread a set bit took ``lucy``'s launch from 84 M
    # threads, 252 us, to 20 us and ``dragon``'s 7.1 -> 4.9 us. Declined: a thread a word walking
    # its set bits, 1.4-2.5x the set-bit launch except on ``lucy`` and 9x on ``holes_pinched``,
    # whose rotations it runs one after another.
    i, g, flagged = launched_set_bit(words, word_ranks, wp.int32(wp.tid()), by_rank)
    if not flagged:
        return
    h = key_ordered_halfedge(key_order, i)
    if mates.shape[0] > 0:
        following = next_boundary_halfedge_from_mates(faces, mates, h)
        out_tails[g] = h
        out_next[h] = wp.where(following >= 0, following, h)
        if following >= 0 and not is_boundary_mate(mates[following]):
            # A dead end, on a mesh that is not edge-manifold: a halfedge with no twin that is no
            # node, where the chain must end. Written here, every entry the ranking reads -- a
            # node's or a node's successor's -- is written by this launch, so the
            # halfedge-sized table takes no fill. Its threads all store ``-1``.
            out_next[following] = -1
    else:
        a, b = halfedge_endpoints(faces, h)
        out_tails[g] = a
        out_next[a] = b


@wp.kernel
def record_pinch_out_halfedges(
    tails: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_slots: wp.array[wp.int32],
) -> None:
    # Each boundary halfedge registers at its origin: the first two of a vertex in its slots, the
    # count beside them. ``out_counts`` arrives zeroed; only a pinch reaches two.
    h = tails[wp.int32(wp.tid())]
    v = faces[h]
    slot = wp.atomic_add(out_counts, v, 1)
    if slot < 2:
        out_slots[2 * v + slot] = h


@wp.kernel
def cross_pinch_gaps(
    tails: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
) -> None:
    # In place over ``emit_boundary_successors``' halfedge table: at a vertex with two fans the
    # walk leaves by the other fan's boundary halfedge, crossing the gap rather than staying in its
    # own fan, so a loop follows the empty region. With two out-halfedges the swap is the whole
    # rule; each thread rewrites only its own entry. A vertex with more fans keeps the in-fan
    # successor, and so does a walk whose successor is no boundary halfedge of ``v`` (a dead end on
    # a mesh that is not edge-manifold): sent to one of ``v``'s two it would share it with the walk
    # already leading there, and the ranking needs every successor to have one predecessor.
    h = tails[wp.int32(wp.tid())]
    s = next_node[h]
    v = faces[s]
    if counts[v] == 2 and (slots[2 * v] == s or slots[2 * v + 1] == s):
        next_node[h] = wp.where(slots[2 * v] == s, slots[2 * v + 1], slots[2 * v])


# ``boundary._split_at_pinches`` on the device: every loop that passes a vertex twice cut into
# simple cycles, each rotated to start at its smallest vertex, the cycles in ascending order of
# their first two vertices (ties in the order the cuts made them). The cut is a stack walk, one
# thread a loop (``split_pinched_loops``); what it needs from outside its loop is each entry's
# previous occurrence of the same vertex in the same loop, which a stable sort of the entries by
# vertex lays side by side (``pinch_occurrence_keys``, ``previous_occurrences``).
#
# Against the host walk it replaced (a readback, a Python stack walk a loop, an upload), new/old:
# ``holes_pinched`` (10 460 cycles) 0.0057x CUDA and 0.11x CPU, an icosphere(7) with every
# seventh face removed 0.0032x CUDA. Its five launches and two sorts are a fixed cost the host
# form did not pay: on a five-by-five grid with two touching holes CUDA reads 0.93x but the CPU
# 1.35x (+0.12 ms), even at about a hundred boundary entries. Kept on both devices: a host path
# below a size would be a second implementation of one answer.


@wp.kernel
def pinch_occurrence_keys(
    flat_loops: wp.array[wp.int32], out_keys: wp.array[wp.int32], out_positions: wp.array[wp.int32]
) -> None:
    # Each packed entry's vertex as its sort key, its position as the payload: a stable sort then
    # leaves every vertex's occurrences adjacent and in position order.
    p = wp.int32(wp.tid())
    out_keys[p] = flat_loops[p]
    out_positions[p] = p


@wp.kernel
def previous_occurrences(
    sorted_vertices: wp.array[wp.int32],
    sorted_positions: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_previous: wp.array[wp.int32],
    out_totals: wp.array[wp.int32],
) -> None:
    # Over the sorted entries: each position's previous occurrence of its vertex within its own
    # loop, else ``-1``. Thread 0 also zeroes the walk's two totals, which arrive unwritten.
    s = wp.int32(wp.tid())
    if s == 0:
        out_totals[0] = 0
        out_totals[1] = 0
    p = sorted_positions[s]
    previous = wp.int32(-1)
    if s > 0 and sorted_vertices[s - 1] == sorted_vertices[s]:
        q = sorted_positions[s - 1]
        if q >= offsets[loop_owner(loop_id, offsets, p)]:
            previous = q
    out_previous[p] = previous


@wp.func
def store_pinch_cycle(
    flat_loops: wp.array[wp.int32],
    stack: wp.array[wp.int32],
    base: wp.int32,
    first: wp.int32,
    depth: wp.int32,
    value_slot: wp.int32,
    record_slot: wp.int32,
    out_values: wp.array[wp.int32],
    out_records: wp.array[wp.vec3i],
    out_keys: wp.array[wp.int64],
) -> wp.int32:
    # The path entries ``first .. depth - 1`` (positions on ``stack`` from ``base``) as one cycle:
    # their vertices stored in walk order from ``value_slot``, and at ``record_slot`` the record
    # ``(value_slot, size, rotation)`` -- ``rotation`` the offset of the smallest vertex, its first
    # occurrence as ``np.argmin`` takes it -- and the sort key ``(smallest << 32) | (next + 1)``,
    # ``next`` the vertex after the smallest or ``-1`` for a cycle of one. Returns the size.
    size = depth - first
    smallest = wp.int32(INT32_MAX_CONSTANT)
    rotation = wp.int32(0)
    for k in range(size):
        vertex = flat_loops[stack[base + first + k]]
        out_values[value_slot + k] = vertex
        if vertex < smallest:
            smallest = vertex
            rotation = k
    following = wp.int32(-1)
    if size > 1:
        following = out_values[value_slot + (rotation + 1) % size]
    out_records[record_slot] = wp.vec3i(value_slot, size, rotation)
    out_keys[record_slot] = (wp.int64(smallest) << wp.int64(32)) | wp.int64(following + 1)
    return size


# Lanes a loop of ``split_pinched_loops``: a warp, at its own launch width (``module="unique"``).
PINCH_SPLIT_LANES = 32


@wp.kernel(enable_backward=False, module="unique")
def split_pinched_loops(
    flat_loops: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    previous: wp.array[wp.int32],
    sentinel: wp.int64,
    out_stack: wp.array[wp.int32],
    out_path_index: wp.array[wp.int32],
    out_representative: wp.array[wp.int32],
    out_values: wp.array[wp.int32],
    out_records: wp.array[wp.vec3i],
    out_keys: wp.array[wp.int64],
    out_slots: wp.array[wp.int32],
    out_totals: wp.array[wp.int32],
) -> None:
    # One block per loop. A loop passing no vertex twice is one cycle in walk order: its lanes
    # copy it, take its smallest vertex by a block minimum of ``(vertex, offset)`` (its vertices
    # are distinct, so no tie) and fill its slots, striding by ``wp.block_dim()``. Most loops of
    # a mesh with a pinch somewhere have none of their own, and one thread walking each of them
    # in turn was the launch (dragon's deletion rims: 0.07x this kernel's device time).
    #
    # A loop with a repeat takes the host walk's stack decomposition over positions, on the
    # first lane: an entry whose vertex is on the path cuts the path back to that vertex's entry
    # and the cut-off run, from that entry on, is a cycle; any other entry is pushed. What is left
    # at the end is the last cycle. The scratch tables are indexed by packed position, inside the
    # loop's own range: ``out_stack`` the path (from the loop's start), ``out_path_index`` a
    # pushed entry's depth or ``-1`` once cut off, ``out_representative`` the pushed entry an
    # occurrence stands for. An occurrence's vertex is on the path exactly when its previous
    # occurrence's representative still is: when that one was handled no other entry of the
    # vertex was on the path.
    #
    # Cycles take value and record slots from the loop's start in cut order (a loop yields at
    # most one value and one cycle per entry); the loop's other record slots get ``sentinel``,
    # which sorts after every key. ``out_totals`` counts the cycles and their entries.
    ell, lane = wp.tid()
    start = offsets[ell]
    end = offsets[ell + 1]
    size = end - start
    repeats = wp.int32(0)
    smallest = wp.int64(INT32_MAX_CONSTANT) << wp.int64(32)
    for p in range(start + lane, end, wp.block_dim()):
        if previous[p] >= 0:
            repeats = wp.int32(1)
        vertex = flat_loops[p]
        out_values[p] = vertex
        smallest = wp.min(smallest, (wp.int64(vertex) << wp.int64(32)) | wp.int64(p - start))
    repeats = block_max(repeats)
    cycles = wp.int32(0)
    if repeats == 0:
        if size > 0:
            smallest = block_min(smallest)
            cycles = wp.int32(1)
            if lane == 0:
                rotation = wp.int32(smallest & wp.int64(0xFFFFFFFF))
                following = wp.int32(-1)
                if size > 1:
                    following = flat_loops[start + (rotation + 1) % size]
                out_records[start] = wp.vec3i(start, size, rotation)
                out_keys[start] = (smallest - wp.int64(rotation)) | wp.int64(following + 1)
                wp.atomic_add(out_totals, 0, 1)
                wp.atomic_add(out_totals, 1, size)
    else:
        if lane == 0:
            depth = wp.int32(0)
            stored = wp.int32(0)
            for p in range(start, end):
                q = previous[p]
                representative = wp.int32(-1)
                on_path = wp.int32(-1)
                if q >= 0:
                    representative = out_representative[q]
                    on_path = out_path_index[representative]
                if on_path >= 0:
                    stored += store_pinch_cycle(
                        flat_loops,
                        out_stack,
                        start,
                        on_path,
                        depth,
                        start + stored,
                        start + cycles,
                        out_values,
                        out_records,
                        out_keys,
                    )
                    cycles += 1
                    for k in range(on_path + 1, depth):
                        out_path_index[out_stack[start + k]] = -1
                    depth = on_path + 1
                    out_representative[p] = representative
                else:
                    out_stack[start + depth] = p
                    out_path_index[p] = depth
                    out_representative[p] = p
                    depth += 1
            if depth > 0:
                stored += store_pinch_cycle(
                    flat_loops,
                    out_stack,
                    start,
                    0,
                    depth,
                    start + stored,
                    start + cycles,
                    out_values,
                    out_records,
                    out_keys,
                )
                cycles += 1
            wp.atomic_add(out_totals, 0, cycles)
            wp.atomic_add(out_totals, 1, stored)
        # The first lane's cycle count, to every lane.
        cycles = block_max(cycles)
    for slot in range(start + lane, end, wp.block_dim()):
        out_slots[slot] = slot
        if slot >= start + cycles:
            out_keys[slot] = sentinel


@wp.kernel
def pinch_cycle_sizes(
    slots: wp.array[wp.int32], records: wp.array[wp.vec3i], out_offsets: wp.array[wp.int32]
) -> None:
    # Cycle ``g`` of the sorted order's size into ``out_offsets[g + 1]``, which the caller scans
    # in place; thread 0 writes the leading ``0``.
    g = wp.int32(wp.tid())
    if g == 0:
        out_offsets[0] = 0
    out_offsets[g + 1] = records[slots[g]][1]


@wp.kernel
def emit_pinch_cycles(
    slots: wp.array[wp.int32],
    records: wp.array[wp.vec3i],
    values: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_flat: wp.array[wp.int32],
) -> None:
    # One thread per sorted cycle: its vertices from the smallest on, at its scanned offset.
    g = wp.int32(wp.tid())
    record = records[slots[g]]
    base = offsets[g]
    for j in range(record[1]):
        out_flat[base + j] = values[record[0] + (record[2] + j) % record[1]]


@wp.func
def add_rim_edge_length(
    out_perimeter: wp.array[wp.float32], ell: wp.int32, a: wp.vec3, c: wp.vec3
) -> None:
    # One rim edge's arc length into its loop's perimeter, in arrival order: the atomic perimeter
    # of ``loop_perimeters`` and of ``holes.loop_rim_metrics``, whose fixed-order form is
    # ``loop_perimeter_lanes``.
    wp.atomic_add(out_perimeter, ell, wp.length(c - a))


@wp.kernel
def loop_perimeters(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_perimeter: wp.array[wp.float32],
) -> None:
    # Segmented ``polyline_length(closed=True)``: the arc length of every loop in one launch, so
    # ``preserve_largest_hole`` costs one readback instead of two per loop.
    t = wp.int32(wp.tid())
    ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_offsets, vertices, t)
    add_rim_edge_length(out_perimeter, ell, a, c)


@wp.func
def rim_edge_length(a: wp.vec3, c: wp.vec3, _origin: wp.vec3) -> wp.float32:
    # ``loop_rim_lanes_sum``'s perimeter term: the rim edge's arc length (``origin`` unread).
    return wp.length(c - a)


@wp.func
def loop_rim_lanes_sum(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    ell: wp.int32,
    lane: wp.int32,
    origin: wp.vec3,
    local: Any,
    term: wp.Function,
):
    # Loop ``ell``'s sum of ``term(a, c, origin)`` over its rim edges ``a -> c``, added onto
    # ``local`` by the block's lanes in one fixed order: lane ``l`` folds the rim edges
    # ``l, l + block_dim, ...`` and ``block_sum`` adds the lanes, so the bits are the same on every
    # run. ``loop_perimeter_lanes`` and ``loop_newell_lanes`` differ only in the term.
    slot = loop_offsets[ell] + lane
    end = loop_offsets[ell + 1]
    while slot < end:
        _ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_offsets, vertices, slot)
        local += term(a, c, origin)
        slot += wp.block_dim()
    return block_sum(local)


@wp.func
def loop_perimeter_lanes(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    ell: wp.int32,
    lane: wp.int32,
) -> wp.float32:
    # Loop ``ell``'s perimeter in ``loop_rim_lanes_sum``'s fixed lane order, where
    # ``loop_perimeters``' atomics commit in arrival order. Every lane returns the total.
    return loop_rim_lanes_sum(
        flat_loops,
        loop_id,
        loop_offsets,
        vertices,
        ell,
        lane,
        wp.vec3(0.0),
        wp.float32(0.0),
        rim_edge_length,
    )


@wp.kernel
def loop_perimeters_ordered(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_perimeter: wp.array[wp.float32],
) -> None:
    # ``loop_perimeters`` in deterministic mode: one block per loop (``LOOP_SUM_BLOCK`` lanes,
    # one on the CPU device), written once instead of accumulated, so ``out_perimeter`` needs no
    # zeroing.
    ell, lane = wp.tid()
    total = loop_perimeter_lanes(flat_loops, loop_id, loop_offsets, vertices, ell, lane)
    if lane == 0:
        out_perimeter[ell] = total


@wp.func
def loop_newell_lanes(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    ell: wp.int32,
    lane: wp.int32,
) -> wp.vec3:
    # Loop ``ell``'s Newell sum about its first vertex (``newell_term``), in
    # ``loop_perimeter_lanes``' fixed lane order. Shared by ``loop_directed_areas_ordered`` (half
    # of it) and ``holes.finalize_rim_metrics_ordered`` (the hole plane), whose atomic forms
    # differ the same way: only in the scale. Every lane returns the total.
    origin = vertices[flat_loops[loop_offsets[ell]]]
    return loop_rim_lanes_sum(
        flat_loops, loop_id, loop_offsets, vertices, ell, lane, origin, wp.vec3(0.0), newell_term
    )


@wp.kernel
def loop_directed_areas(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_directed_area: wp.array[wp.vec3],
) -> None:
    # Half the sum of ``(p_i - p_0) x (p_{i+1} - p_0)`` around the loop: the directed area vector,
    # whose norm is the area of the planar polygon spanning the loop and whose direction is that
    # polygon's normal. Taken about the loop's first vertex (``newell_term``), which every thread
    # reads without a centroid pass -- accumulated per segment in one launch, like the perimeter.
    t = wp.int32(wp.tid())
    ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_offsets, vertices, t)
    origin = vertices[flat_loops[loop_offsets[ell]]]
    wp.atomic_add(out_directed_area, ell, wp.float32(0.5) * newell_term(a, c, origin))


@wp.kernel
def loop_directed_areas_ordered(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_directed_area: wp.array[wp.vec3],
) -> None:
    # ``loop_directed_areas`` in deterministic mode, as ``loop_perimeters_ordered`` is for the
    # perimeter. Halving the sum rather than each term changes no bit: the scale is a power of two.
    ell, lane = wp.tid()
    total = loop_newell_lanes(flat_loops, loop_id, loop_offsets, vertices, ell, lane)
    if lane == 0:
        out_directed_area[ell] = wp.float32(0.5) * total


@wp.kernel
def longest_loop_key(loop_offsets: wp.array[wp.int32], out_best: wp.array[wp.int64]) -> None:
    # The longest packed loop, as one ``wp.atomic_max`` over ``pack_ranked_key``. The low half
    # carries the loop's *start* rather than its index, which is what lets a single readback of
    # this key give the caller both halves of the answer -- a second read of ``loop_offsets`` at
    # the winning index would otherwise cost as much again as the reduction. Starts increase with
    # the loop index, so "lowest start on a tie" is "lowest index on a tie" and the packer's
    # tie-break is the one a host-side first-maximum scan would have produced.
    #
    # Against unpacking the loops and scanning them on the host, output identical: a wash at 7
    # rims -- where the reduction's launch costs about what the handful of array views it removes
    # did -- 1.8x on the whole public call at 384, and 4.1x on a mesh with several thousand. The
    # win grows with the rim count because the host form was linear in it and this is not.
    ell = wp.int32(wp.tid())
    start = loop_offsets[ell]
    wp.atomic_max(out_best, 0, pack_ranked_key(loop_offsets[ell + 1] - start, start))
