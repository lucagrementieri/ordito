import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels import array as kernel_array
from ordito.kernels.reduce import block_chunk_1d, block_min

wp.set_module_options({"enable_backward": False})


@wp.func
def write_adjacency_keys(
    edges: wp.array2d[wp.int32],
    e: wp.int32,
    n: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # One undirected edge becomes two directed entry keys in ``array.csr_key_buffers``' layout:
    # slot ``2e`` is ``(a, b)`` and ``2e + 1`` is ``(b, a)``, each carrying its own slot as the
    # payload -- what ``array.coo_keys`` would write from a two-column triplet table, without the
    # table. Both kernels below emit in exactly this layout, which is what lets the weighted one
    # duplicate the weight into the matching slots.
    a = edges[e, 0]
    b = edges[e, 1]
    base = e * 2
    out_keys[base] = kernel_array.csr_key(a, b, n, n)
    out_order[base] = base
    out_keys[base + 1] = kernel_array.csr_key(b, a, n, n)
    out_order[base + 1] = base + 1


@wp.kernel
def edges_to_adjacency_keys(
    edges: wp.array2d[wp.int32],
    n: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The unweighted path, whose values are a ``wp.ones`` fill rather than a launch.
    write_adjacency_keys(edges, wp.int32(wp.tid()), n, out_keys, out_order)


@wp.kernel
def edges_to_adjacency_keys_weighted(
    edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    n: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # The weighted path: the keys and the two duplicated values in one launch.
    e = wp.int32(wp.tid())
    write_adjacency_keys(edges, e, n, out_keys, out_order)
    out_values[e * 2] = weights[e]
    out_values[e * 2 + 1] = weights[e]


@wp.kernel
def scatter_neighbor_lists(
    edges: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_neighbors: wp.array[wp.int32],
) -> None:
    # Both directed entries of each undirected edge, into the CSR row the counting pass sized --
    # the payload half of ``graph.edges_to_neighbor_lists``. The same counting-sort fill as
    # [`scatter_vertex_faces`][ordito.kernels.adjacency.scatter_vertex_faces]; what differs is the
    # payload, which is the *other endpoint* here and the owning face there, so the two are
    # siblings rather than one kernel.
    #
    # ``cursor`` is the counting pass's per-node degree, not an output, and it counts *down*: the
    # atomic returns how many slots of the row are still free, so ``offsets[row + 1]`` minus it is
    # the next free slot from the front -- the slot a zeroed cursor counting up hands out, without
    # zeroing the cursor first. The column order within a row is thread order and **not sorted**
    # -- nor stable, so two runs return different permutations. That is deliberate and is what
    # this build costs less than ``bsr_from_triplets`` for.
    # ``graph.edges_to_neighbor_lists(sort_rows=True)`` pins it with one ``array.sort_segments``
    # launch, which makes the buffer identical to ``graph.edges_to_csr``'s; that is what
    # ``neighbors.geodesic_ball`` asks for, because it emits its BFS queue in visit order.
    e = wp.int32(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    out_neighbors[offsets[a + 1] - wp.atomic_sub(cursor, a, 1)] = b
    out_neighbors[offsets[b + 1] - wp.atomic_sub(cursor, b, 1)] = a


@wp.kernel
def signed_edge_bounds(
    edges: wp.array2d[wp.int32], signs: wp.array[wp.int32], out_bounds: wp.array[wp.int32]
) -> None:
    # ``graph.connected_component_parity_from_edges``' two range checks in one fold and one read:
    # ``out_bounds`` (four slots seeded ``INT32_MAX``) receives ``min(edges)``, ``min(~edges)``,
    # ``min(signs)`` and ``min(~signs)``. ``~v`` is ``-v - 1``, so its minimum is ``~max(v)``
    # with no overflow at ``INT32_MIN``, and all four commits are the same ``atomic_min``.
    # Lane-strided by ``wp.block_dim()`` over the block's ``ITEMS_PER_BLOCK_1D`` rows.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(edges.shape[0], block)
    low = wp.int32(INT32_MAX_CONSTANT)
    low_flip = wp.int32(INT32_MAX_CONSTANT)
    low_sign = wp.int32(INT32_MAX_CONSTANT)
    low_sign_flip = wp.int32(INT32_MAX_CONSTANT)
    for k in range(lane, count, wp.block_dim()):
        e = offset + k
        a = edges[e, 0]
        b = edges[e, 1]
        s = signs[e]
        low = wp.min(low, wp.min(a, b))
        low_flip = wp.min(low_flip, wp.min(~a, ~b))
        low_sign = wp.min(low_sign, s)
        low_sign_flip = wp.min(low_sign_flip, ~s)
    low = block_min(low)
    low_flip = block_min(low_flip)
    low_sign = block_min(low_sign)
    low_sign_flip = block_min(low_sign_flip)
    if lane == 0 and count > 0:
        wp.atomic_min(out_bounds, 0, low)
        wp.atomic_min(out_bounds, 1, low_flip)
        wp.atomic_min(out_bounds, 2, low_sign)
        wp.atomic_min(out_bounds, 3, low_sign_flip)


@wp.kernel
def shortest_path_envelope_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    labels: wp.array[wp.float32],
    out_labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # One Bellman-Ford relaxation of ``q_i <= q_j + w(i, j)``, one thread per node, pulling from the
    # neighbours' labels of the *previous* round -- so the pass is a pure function of ``labels`` and
    # the result does not depend on how the threads interleave. Labels only ever go down, so the
    # iteration is monotone and converges in at most (graph diameter) passes.
    # (VCG ``UpdateQuality::VertexSaturate`` is this loop.)
    #
    # **Two passes per round, written into each other's buffer, would remove the wrapper's
    # ``wp.copy``** -- it is a whole device pass over the node array and worth a real fraction of
    # that call -- and it is **not portable**: the two devices then disagree. Measured with the
    # unrolled body, the CPU device relaxed strictly fewer nodes per round than CUDA at every cap,
    # because the recorded body did not replay as two passes per round there; even caps agreed
    # exactly. A ``wp.capture_while`` body is not guaranteed to execute as an indivisible unit
    # across devices, so a loop whose *result buffer* depends on the body running whole cannot rely
    # on it. A Python-level ping-pong cannot help either: the body is recorded once and replayed,
    # so rebinding the names would only take effect at record time.
    #
    # ``out_state`` is the shared round-loop word of ``kernels/array.py``; raising
    # ``LOOP_PROGRESS`` is this pass's "something moved", which the closing ``array.loop_advance``
    # turns into the next round's condition and clears. A plain store, not an atomic: the slot only
    # ever goes 0 -> 1 within a round, so every writer writes the same value.
    i = wp.int32(wp.tid())
    best = labels[i]
    for k in range(offsets[i], offsets[i + 1]):
        relaxed = labels[columns[k]] + weights[k]
        if relaxed < best:
            best = relaxed
    out_labels[i] = best
    if best < labels[i]:
        out_state[kernel_array.LOOP_PROGRESS] = 1


# ---------------------------------------------------------------------------------------------
# Successor-cycle ranking
#
# ``graph.successor_cycles`` and every boundary walk rank a successor graph given as a node-sized
# successor table plus its *tails*, the nodes with an edge out. Where each node also has at most
# one edge in, every component is a cycle or a chain, so the cycle a node belongs to is named by
# its smallest node, and that node is also where the cycle is cut to rank it -- one quantity, which
# pointer jumping finds on its own. A node's *window* of length ``W`` is itself and its next
# ``W - 1`` successors; the table holds, per node, ``(successor^W, smallest node in the window,
# hops to its first occurrence)``. Merging two adjacent windows keeps the earlier minimum on a tie,
# and once ``W`` covers the cycle the minimum is the cycle's and the hop count is the distance to
# it. No union-find and no cut: the start is the smallest node of the cycle's connected component,
# the label a component labelling gives it.
#
# A chain is a walk that reaches a node with no edge out, which the table holds as ``-1``: the end
# of an open chain handed to ``successor_cycles``, a vertex where a boundary rim runs into a
# three-faced edge, a halfedge whose rotation dead-ends on one that is not a boundary halfedge.
# Such a successor makes the window's successor ``-1``: the node is on a chain, not a cycle, and is
# dropped with its whole chain. Where no two tails share a successor the cycles that remain are
# exact; where some do (malformed input, which the boundary walks never build) every write stays
# in range.

# What a ranked node is written out as, before the optional ``node_values`` gather.
CYCLE_NODES = 0  # the node itself
CYCLE_PAIRED = 1  # node ``2 * v + s`` -> ``v``; keep only cycles starting at an even node


@wp.kernel
def scatter_successor_tails(
    directed_edges: wp.array2d[wp.int32],
    out_next: wp.array[wp.int32],
    out_tails: wp.array[wp.int32],
) -> None:
    # ``successor_cycles``' successor table and its tails in one pass over the edges, both
    # node-sized and seeded ``-1``. The tails are listed in node space -- entry ``a`` is ``a`` when
    # ``a`` has an edge out, ``-1`` (skipped by the ranking) otherwise -- so the ranking's threads
    # walk the nodes in ascending order, which keeps a mesh-ordered graph's reads and writes local
    # (listed in edge order, a shuffled edge list made the ranking 4x slower at 3 M nodes), and a
    # node with several edges out (malformed input) is listed once. Indexed by the raw endpoint
    # unguarded: the wrapper's range check before this launch is what this relies on.
    e = wp.int32(wp.tid())
    a = directed_edges[e, 0]
    out_next[a] = directed_edges[e, 1]
    out_tails[a] = a


# The widest pointer jump one launch takes. A launch chasing ``h`` pointers through the previous
# round's table multiplies every node's window by ``h``, so ``L`` nodes need ``ceil(log_h L)``
# launches rather than doubling's ``ceil(log2 L)``. The chase is ``h - 1`` dependent loads per
# thread, a few microseconds against a launch's host cost, so the widest hop that still bounds the
# round count is the one to take; ``pointer_jump_schedule`` then narrows it to the smallest hop that
# reaches the same round count.
POINTER_JUMP_MAX_HOPS = 16


def pointer_jump_schedule(n: int) -> tuple[int, int]:
    """
    ``(hops, rounds)`` for ranking chains of at most ``n`` nodes: ``hops ** rounds >= n``.

    ``rounds`` is the fewest launches at ``POINTER_JUMP_MAX_HOPS``, and ``hops`` the smallest
    width reaching ``n`` in that many -- the same launch count for the fewest dependent loads.
    """
    rounds = 1
    window = POINTER_JUMP_MAX_HOPS
    while window < n:
        window *= POINTER_JUMP_MAX_HOPS
        rounds += 1
    hops = 2
    while hops**rounds < n:
        hops += 1
    return hops, rounds


@wp.func
def closed_cycle_is_kept(successor: wp.int32, start: wp.int32, mode: wp.int32) -> wp.bool:
    # A node is ranked when its window closed (no chain end reached) and, for paired nodes, when
    # its cycle is the even-starting one of its mirror pair
    # (``boundary._unoriented_boundary_cycles``).
    return successor >= 0 and (mode != CYCLE_PAIRED or start % 2 == 0)


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
    # a walk can reach a non-tail, and a tail whose successor is ``-1`` ends its chain at once. A
    # ``-1`` tail is a skipped edge (``scatter_successor_tails``).
    v = tails[wp.tid()]
    if v < 0:
        return
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
    if v < 0:
        return
    own = windows[v]
    current = own[0]
    start = own[1]
    dist = own[2]
    offset = window
    for _ in range(hops - 1):
        if current >= 0:
            other = windows[current]
            start, dist = kernel_array.merge_window_minimum(start, dist, other[1], other[2], offset)
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
    node_values: wp.array[wp.int32],
    out_flat: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
) -> None:
    # ``cycle_counts`` is scanned inclusively in place over node space, so the entry below a cycle's
    # start is ``(its rank, its offset)`` and the difference is ``(1, its length)``: every cycle is
    # placed in ascending start order with no rank table and no second scan. The start writes the
    # cycle's own offset and its end into the total-terminated ``out_offsets``; the end is the next
    # cycle's offset, so every interior entry is written twice with the same value and the last
    # entry (the total) is written by the last cycle. ``node_values``, when not empty, maps what is
    # written (the halfedge walk's origin vertices).
    v = tails[wp.tid()]
    if v < 0:
        return
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
    if mode == CYCLE_PAIRED:
        node = v // 2
    if node_values.shape[0] > 0:
        node = node_values[node]
    out_flat[exclusive[1] + kernel_array.wrap_index(length - own[2], length)] = node
    if v == start:
        out_offsets[exclusive[0]] = exclusive[1]
        out_offsets[inclusive[0]] = inclusive[1]
