import warp as wp

from ordito.kernels.adjacency import edge_endpoints
from ordito.kernels.array import loop_rim_edge, pack_ranked_key, scanned_count
from ordito.kernels.halfedge import (
    HALFEDGE_MATE_BOUNDARY,
    halfedge_endpoints,
    key_ordered_halfedge,
    next_boundary_halfedge,
)
from ordito.kernels.predicates import newell_term
from ordito.kernels.reduce import block_sum

wp.set_module_options({"enable_backward": False})

# Lanes per loop of the fixed-order segmented sums (``loop_perimeters_ordered`` and the
# ``*_ordered`` kernels after it). A loop is one block, so the width only has to cover a typical
# rim; a long one is walked ``LOOP_SUM_BLOCK`` edges at a time.
LOOP_SUM_BLOCK = 64

# Every boundary query here reads the halfedge mates (``halfedge.halfedge_mates``): a boundary
# edge is a halfedge with no mate, which names the halfedge -- row ``h`` of
# ``edges.faces_to_edges`` -- so the edge's endpoints (sorted or directed) are read straight off
# ``faces`` and no ``(3F, 2)`` edge table is ever built. The kernels below write the flags the
# caller scans and read the verdict back out of the scan.


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


@wp.kernel
def mark_boundary_halfedges(
    mates: wp.array[wp.int32],
    key_order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 where a halfedge spans its edge alone -- a boundary edge -- as the ``int32`` flag, entry
    # ``i`` standing for halfedge ``key_ordered_halfedge(key_order, i)``
    # ``wp.utils.array_scan`` then scans in place, over ``out_flags``' first ``n`` entries. With
    # ``out_degrees`` (else ``None``), ``boundary_loops_with_offsets``' degree census of the
    # directed rows (``halfedge_endpoints``) rides in the same launch, its two defect bits stamped
    # into the zeroed ``out_flags[n]`` and ``out_flags[n + 1]`` -- the scan does not reach them, so
    # the total and both bits come back in one readback.
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
    n = mates.shape[0]
    h = key_ordered_halfedge(key_order, i)
    boundary = is_boundary_mate(mates[h])
    out_flags[i] = wp.where(boundary, wp.int32(1), wp.int32(0))
    if boundary and out_degrees.shape[0] > 0:
        tail, head = halfedge_endpoints(faces, h)
        if wp.atomic_add(out_degrees, tail, 0, 1) >= 1:
            out_flags[n] = 1
        if wp.atomic_add(out_degrees, tail, 1, 1) >= 2:
            out_flags[n + 1] = 1
        if wp.atomic_add(out_degrees, head, 1, 1) >= 2:
            out_flags[n + 1] = 1


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
    inclusive: wp.array[wp.int32],
    key_order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    sort_pair: wp.bool,
    out_edges: wp.array2d[wp.int32],
) -> None:
    # ``emit_boundary_edges`` over ``halfedge.key_ordered_halfedges``' rank compaction in one
    # launch, for flags scanned in the key sort's order (``mark_boundary_halfedges`` given its
    # ``key_order``): one thread per flag, and where the scan steps, the halfedge's row is stored at
    # its rank, which is its row of the ascending-key table. The two kernels differ only in where
    # a row's halfedge and slot come from.
    j = wp.int32(wp.tid())
    slot, count = scanned_count(inclusive, j)
    if count != 0:
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
    inclusive: wp.array[wp.int32],
    key_order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    out_tails: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
) -> None:
    # The boundary as a successor graph, straight off ``mark_boundary_halfedges``' scanned flags
    # with no edge rows in between: the ranked nodes in ``out_tails`` and their successors in the
    # node-sized ``out_next``. Without ``twins`` (``None``) the nodes are the vertices, the
    # directed rows' tails (``halfedge_endpoints``). With it they are the boundary
    # *halfedges*, each followed by the boundary halfedge leaving its tip in its own sector: over
    # vertices a pinch point has two successors, over halfedges every node has one. A fan that
    # does not close (a malformed twin table) maps to a self-loop, so the walk stays bounded. The
    # ranking names each cycle by its smallest node, so the order of ``out_tails`` is free.
    i = wp.int32(wp.tid())
    g, count = scanned_count(inclusive, i)
    if count == 0:
        return
    h = key_ordered_halfedge(key_order, i)
    if twins.shape[0] > 0:
        following = next_boundary_halfedge(faces, twins, h)
        out_tails[g] = h
        out_next[h] = wp.where(following >= 0, following, h)
    else:
        a, b = halfedge_endpoints(faces, h)
        out_tails[g] = a
        out_next[a] = b


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


@wp.func
def loop_perimeter_lanes(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    ell: wp.int32,
    lane: wp.int32,
) -> wp.float32:
    # Loop ``ell``'s perimeter, summed by the block's lanes in one fixed order: lane ``l`` folds
    # the rim edges ``l, l + block_dim, ...`` and ``block_sum`` adds the lanes, so the bits are the
    # same on every run, where ``loop_perimeters``' atomics commit in arrival order. Every lane
    # returns the total.
    local = wp.float32(0.0)
    slot = loop_offsets[ell] + lane
    end = loop_offsets[ell + 1]
    while slot < end:
        _ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_offsets, vertices, slot)
        local += wp.length(c - a)
        slot += wp.block_dim()
    return block_sum(local)


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
    local = wp.vec3(0.0)
    slot = loop_offsets[ell] + lane
    end = loop_offsets[ell + 1]
    while slot < end:
        _ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_offsets, vertices, slot)
        local += newell_term(a, c, origin)
        slot += wp.block_dim()
    return block_sum(local)


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
