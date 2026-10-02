import warp as wp

from ordito.kernels.adjacency import edge_endpoints
from ordito.kernels.array import (
    loop_rim_edge,
    merge_window_minimum,
    pack_ranked_key,
    scanned_count,
    wrap_index,
)
from ordito.kernels.grouping import sorted_run_of_length
from ordito.kernels.halfedge import halfedge_endpoints, next_boundary_halfedge

# Every boundary query here is one radix sort of the halfedges' undirected edge keys, carrying each
# halfedge's index as the payload: a boundary edge is then a run of exactly one key, and its
# payload names the halfedge -- which is row ``h`` of ``edges.faces_to_edges``, so the edge's
# endpoints (sorted or directed) are read straight off ``faces`` and no ``(3F, 2)`` edge table is
# ever built. The kernels below write the sort's input and read its verdict.


@wp.func
def boundary_halfedge_pair(
    faces: wp.array[wp.int32], sort_pair: wp.bool, h: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Halfedge ``h``'s row of ``edges.faces_to_edges``: its directed pair, ascending when
    # ``sort_pair`` -- read off ``faces``, so no ``(3F, 2)`` table is built or read.
    if sort_pair:
        return edge_endpoints(faces, h)
    return halfedge_endpoints(faces, h)


@wp.kernel
def mark_boundary_runs(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    faces: wp.array[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 where a sorted position holds a key occurring exactly once -- a boundary edge -- as the
    # ``int32`` flag ``wp.utils.array_scan`` then scans in place, over ``out_flags``' first ``n``
    # entries. With ``out_degrees`` (else ``None``), ``boundary_loops_with_offsets``' degree census
    # of the directed rows (``halfedge_endpoints``) rides in the same launch, its two defect
    # bits stamped into the zeroed ``out_flags[n]`` and ``out_flags[n + 1]`` -- the scan does not
    # reach them, so the total and both bits come back in one readback.
    #
    # Degree column 0: how many boundary edges *leave* each vertex. Column 1: how many touch it at
    # all. One out-edge and two incidences is the well-behaved case. Two out-edges is the seam of a
    # non-orientable surface, where ``succ[tail] = head`` silently drops an edge (bit ``n``). Four
    # incidences is a pinch point, where two loops meet and no 2-regular walk over vertices exists
    # at all (bit ``n + 1``) -- the two are different defects with different walks: the seam's is
    # undirected, the pinch's is over halfedges (``boundary_halfedge_successors``).
    #
    # The bits are stamped here rather than by a second pass over the degree table:
    # ``wp.atomic_add`` returns the value the slot held *before* the increment, so the thread that
    # pushes a vertex past the threshold is the one that knows it. Boundary vertices are the only
    # ones whose degrees are ever non-zero, so no pass over the vertices is needed.
    i = wp.int32(wp.tid())
    boundary = sorted_run_of_length(sorted_keys, n, i, 1)
    out_flags[i] = wp.where(boundary, wp.int32(1), wp.int32(0))
    if boundary and out_degrees.shape[0] > 0:
        tail, head = halfedge_endpoints(faces, order[i])
        if wp.atomic_add(out_degrees, tail, 0, 1) >= 1:
            out_flags[n] = 1
        if wp.atomic_add(out_degrees, tail, 1, 1) >= 2:
            out_flags[n + 1] = 1
        if wp.atomic_add(out_degrees, head, 1, 1) >= 2:
            out_flags[n + 1] = 1


@wp.kernel
def emit_boundary_edges(
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    sort_pair: wp.bool,
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One thread per sorted position; ``inclusive`` is ``mark_boundary_runs``' flags scanned in
    # place, so a boundary edge is where the scan steps and its rank is the step's start. Rows come
    # out in ascending key order -- the order ``grouping.group`` emitted them in -- read from
    # ``faces`` (see ``boundary_halfedge_pair``).
    i = wp.int32(wp.tid())
    g, count = scanned_count(inclusive, i)
    if count == 0:
        return
    a, b = boundary_halfedge_pair(faces, sort_pair, order[i])
    out_edges[g, 0] = a
    out_edges[g, 1] = b


@wp.kernel
def mark_boundary_vertices(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    faces: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Flag both endpoints of every boundary edge in a zeroed per-vertex ``int32`` array, which the
    # caller scans in place: the sorted unique boundary vertices then come out of the scan's steps
    # with no edge list and no ``unique_1d``. Concurrent writers all store 1, so no atomic.
    i = wp.int32(wp.tid())
    if not sorted_run_of_length(sorted_keys, n, i, 1):
        return
    a, b = halfedge_endpoints(faces, order[i])
    out_flags[a] = 1
    out_flags[b] = 1


@wp.kernel
def boundary_halfedge_mask(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    out_mask: wp.array[wp.bool],
) -> None:
    # Per halfedge: is its edge a boundary edge? ``order`` is a permutation of ``0 .. n - 1``, so
    # every entry of ``out_mask`` is written exactly once and it needs no zero fill -- and there is
    # no scan and no readback, since the mask's size is known.
    i = wp.int32(wp.tid())
    out_mask[order[i]] = sorted_run_of_length(sorted_keys, n, i, 1)


@wp.func
def ear_interior_corner(edge_boundary: wp.array[wp.bool], f: wp.int32) -> wp.int32:
    # The local index of face ``f``'s one interior edge when exactly two of its three edges are
    # boundary edges -- an ear -- else ``-1``.
    b0 = edge_boundary[3 * f]
    b1 = edge_boundary[3 * f + 1]
    b2 = edge_boundary[3 * f + 2]
    if wp.int32(b0) + wp.int32(b1) + wp.int32(b2) != 2:
        return -1
    if not b0:
        return 0
    if not b1:
        return 1
    return 2


@wp.kernel
def mark_ears(edge_boundary: wp.array[wp.bool], out_flags: wp.array[wp.int32]) -> None:
    # 0/1 per face: is it an ear? The ``int32`` flag the caller scans in place.
    f = wp.int32(wp.tid())
    out_flags[f] = wp.where(ear_interior_corner(edge_boundary, f) >= 0, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_ears(
    edge_boundary: wp.array[wp.bool],
    inclusive: wp.array[wp.int32],
    out_ear: wp.array[wp.int32],
    out_ear_opp: wp.array[wp.int32],
) -> None:
    # Where ``mark_ears``' in-place scan steps, write the ear and its interior corner at its rank:
    # ascending face order on both devices, with no atomic cursor and no trim copy.
    f = wp.int32(wp.tid())
    slot, flag = scanned_count(inclusive, f)
    if flag != 0:
        out_ear[slot] = f
        out_ear_opp[slot] = ear_interior_corner(edge_boundary, f)


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
def boundary_neighbor(neighbors: wp.array2d[wp.int32], v: wp.int32, slot: wp.int32) -> wp.int32:
    # Vertex ``v``'s boundary neighbour in ``slot``, with the two slots read in ascending order, so
    # the dart numbering below does not depend on the order the scatter's atomics happened to run
    # in. This is what makes the non-orientable answer reproducible: with no face winding to
    # follow, slot 0 is the smaller neighbour by definition. Read sorted rather than sorted in
    # place by a pass of its own; a lone neighbour (the other slot ``-1``) stays in slot 0.
    first = neighbors[v, 0]
    second = neighbors[v, 1]
    if first >= 0 and second >= 0 and second < first:
        first, second = second, first
    return wp.where(slot == 0, first, second)


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
    s = wp.where(boundary_neighbor(neighbors, v, 0) == u, wp.int32(0), wp.int32(1))
    w = boundary_neighbor(neighbors, v, 1 - s)
    dart = 2 * v + s
    out_tails[2 * e + end] = dart
    if w < 0:
        # ``v`` has one boundary edge (a non-manifold edge ends the rim there): a chain end, which
        # the ranking drops.
        out_next[dart] = -1
    else:
        arriving = wp.where(boundary_neighbor(neighbors, w, 0) == v, wp.int32(0), wp.int32(1))
        out_next[dart] = 2 * w + arriving


@wp.kernel
def emit_boundary_successors(
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    out_tails: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
) -> None:
    # The boundary as a successor graph, straight off the sorted keys (``emit_boundary_edges``'
    # scan) with no edge rows in between: the ranked nodes in ``out_tails`` and their successors in
    # the node-sized ``out_next``. Without ``twins`` (``None``) the nodes are the vertices, the
    # directed rows' tails (``halfedge_endpoints``). With it they are the boundary
    # *halfedges*, each followed by the boundary halfedge leaving its tip in its own sector: over
    # vertices a pinch point has two successors, over halfedges every node has one. A fan that
    # does not close (a malformed twin table) maps to a self-loop, so the walk stays bounded.
    i = wp.int32(wp.tid())
    g, count = scanned_count(inclusive, i)
    if count == 0:
        return
    h = order[i]
    if twins.shape[0] > 0:
        following = next_boundary_halfedge(faces, twins, h)
        out_tails[g] = h
        out_next[h] = wp.where(following >= 0, following, h)
    else:
        a, b = halfedge_endpoints(faces, h)
        out_tails[g] = a
        out_next[a] = b


# ---------------------------------------------------------------------------------------------
# Closed-cycle ranking
#
# Every boundary walk hands its ranking a successor graph that is a union of cycles by
# construction: each node is the tail of exactly one edge and the head of at most one. So the
# nodes are the edge tails, the cycle a node belongs to is named by its smallest node, and that
# node is also where the cycle is cut to rank it -- one quantity, which pointer jumping finds on its
# own. A node's *window* of length ``W`` is itself and its next ``W - 1`` successors; the table
# holds, per node, ``(successor^W, smallest node in the window, hops to its first occurrence)``.
# Merging two adjacent windows keeps the earlier minimum on a tie, and once ``W`` covers the cycle
# the minimum is the cycle's and the hop count is the distance to it. No union-find and no cut: the
# connected-component labelling ``graph.successor_cycles`` runs for an arbitrary graph gives each
# cycle the same label, its smallest node, and ranks from the same start.
#
# That holds on an edge-manifold mesh. On one that is not, a walk can reach a node that is not a
# tail -- a vertex where the rim runs into a three-faced edge, a halfedge whose rotation dead-ends
# on one that is not a boundary halfedge -- and several chains can end on the same one. Such a
# successor makes the window's successor ``-1``: the node is on a chain, not a cycle, and is dropped
# with its whole chain -- exactly the component ``successor_cycles`` excludes. No two *tails* ever
# share a successor (that would be a pinch or a seam, which take their own walks), so the cycles
# that remain are exact.

# What a ranked node is written out as.
CYCLE_NODES = 0  # the node itself: a vertex of the vertex walk
CYCLE_DARTS = 1  # dart ``2 * v + s`` -> vertex ``v``; keep only cycles starting at an even dart
CYCLE_HALFEDGES = 2  # halfedge ``h`` -> its origin vertex ``faces[h]``


@wp.func
def closed_cycle_is_kept(successor: wp.int32, start: wp.int32, mode: wp.int32) -> wp.bool:
    # A node is ranked when its window closed (no chain end reached) and, for the dart walk, when
    # its cycle is the even-starting one of its mirror pair (``_unoriented_boundary_cycles``).
    return successor >= 0 and (mode != CYCLE_DARTS or start % 2 == 0)


@wp.func
def count_closed_cycle_node(
    out_cycle_counts: wp.array[wp.vec2i],
    v: wp.int32,
    successor: wp.int32,
    start: wp.int32,
    mode: wp.int32,
) -> None:
    # Last round only (``out_cycle_counts`` is ``None`` before it): per cycle start, ``(1, length)``
    # accumulated as ``(v == start, 1)`` per ranked node, for the caller's inclusive scan.
    if out_cycle_counts.shape[0] > 0 and closed_cycle_is_kept(successor, start, mode):
        wp.atomic_add(out_cycle_counts, start, wp.vec2i(wp.where(v == start, 1, 0), wp.int32(1)))


@wp.kernel
def closed_cycle_windows(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    hops: wp.int32,
    mode: wp.int32,
    out_windows: wp.array[wp.vec3i],
    out_cycle_counts: wp.array[wp.vec2i],
) -> None:
    # Round one: every window of length ``hops``, straight off the successor table. A successor is
    # a tail exactly when its own successor is set: ``next_node`` is ``-1`` off the tails wherever
    # a walk can reach a non-tail, and a tail whose successor is ``-1`` ends its chain at once.
    v = tails[wp.tid()]
    start = v
    dist = wp.int32(0)
    current = v
    following = next_node[v]
    for k in range(1, hops + 1):
        if current >= 0:
            after = wp.int32(-1)
            if following >= 0:
                after = next_node[following]
            if after < 0:
                current = -1
            else:
                current = following
                if k < hops and current < start:
                    start = current
                    dist = k
                following = after
    out_windows[v] = wp.vec3i(current, start, dist)
    count_closed_cycle_node(out_cycle_counts, v, current, start, mode)


@wp.kernel
def closed_cycle_jump(
    tails: wp.array[wp.int32],
    windows: wp.array[wp.vec3i],
    window: wp.int32,
    hops: wp.int32,
    mode: wp.int32,
    out_windows: wp.array[wp.vec3i],
    out_cycle_counts: wp.array[wp.vec2i],
) -> None:
    # One further round: ``hops`` adjacent windows of length ``window`` merged into one, chased
    # through the previous round's table (ping-ponged, so no thread reads a window merged this
    # round), under ``array.merge_window_minimum``'s first-occurrence rule.
    v = tails[wp.tid()]
    own = windows[v]
    current = own[0]
    start = own[1]
    dist = own[2]
    offset = window
    for _ in range(hops - 1):
        if current >= 0:
            other = windows[current]
            start, dist = merge_window_minimum(start, dist, other[1], other[2], offset)
            offset += window
            current = other[0]
    out_windows[v] = wp.vec3i(current, start, dist)
    count_closed_cycle_node(out_cycle_counts, v, current, start, mode)


@wp.kernel
def scatter_closed_cycles(
    tails: wp.array[wp.int32],
    windows: wp.array[wp.vec3i],
    cycle_counts: wp.array[wp.vec2i],
    mode: wp.int32,
    faces: wp.array[wp.int32],
    out_flat: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
) -> None:
    # ``cycle_counts`` is scanned inclusively in place over node space, so the entry below a cycle's
    # start is ``(its rank, its offset)`` and the difference is ``(1, its length)``: every cycle is
    # placed in ascending start order with no rank table and no second scan. The start writes the
    # cycle's own offset and its end into the total-terminated ``out_offsets``; the end is the next
    # cycle's offset, so every interior entry is written twice with the same value and the last
    # entry (the total) is written by the last cycle. ``faces`` is read by the halfedge walk only.
    v = tails[wp.tid()]
    own = windows[v]
    start = own[1]
    if not closed_cycle_is_kept(own[0], start, mode):
        return
    inclusive = cycle_counts[start]
    exclusive = wp.vec2i(0, 0)
    if start > 0:
        exclusive = cycle_counts[start - 1]
    length = inclusive[1] - exclusive[1]
    node = v
    if mode == CYCLE_DARTS:
        node = v // 2
    elif mode == CYCLE_HALFEDGES:
        node = faces[v]
    out_flat[exclusive[1] + wrap_index(length - own[2], length)] = node
    if v == start:
        out_offsets[exclusive[0]] = exclusive[1]
        out_offsets[inclusive[0]] = inclusive[1]


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
    wp.atomic_add(out_perimeter, ell, wp.length(c - a))


@wp.kernel
def loop_directed_areas(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_directed_area: wp.array[wp.vec3],
) -> None:
    # Half the sum of ``p_i x p_{i+1}`` around the loop: the directed area vector, whose norm is the
    # area of the planar polygon spanning the loop and whose direction is that polygon's normal.
    # Origin-independent because the cross products of a *closed* ring cancel the shift, so no
    # centroid pass is needed -- and accumulated per segment in one launch, like the perimeter.
    t = wp.int32(wp.tid())
    ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_offsets, vertices, t)
    wp.atomic_add(out_directed_area, ell, wp.float32(0.5) * wp.cross(a, c))


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
