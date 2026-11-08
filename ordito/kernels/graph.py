import warp as wp

from ordito.constants import INT32_MAX_CONSTANT
from ordito.kernels import array as kernel_array
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair_parity,
    ecl_prehook_pair,
    parity_root_word,
)
from ordito.kernels.reduce import block_chunk_1d, block_min
from ordito.kernels.scatter import claim_row_slot

wp.set_module_options({"enable_backward": False})


@wp.kernel
def edges_to_adjacency_keys(
    edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    n: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # One undirected edge becomes two directed entry keys in ``array.csr_key_buffers``' layout:
    # slot ``2e`` is ``(a, b)`` and ``2e + 1`` is ``(b, a)``, each carrying its own slot as the
    # payload -- what ``array.coo_keys`` would write from a two-column triplet table, without the
    # table -- and both slots' value, the edge's weight, or ``1`` where ``weights`` is empty. The
    # unit values were a fill of their own before the weighted and unweighted kernels became this
    # one: unweighted ``graph.edges_to_csr`` 0.96-0.97x on ``sphere_small`` and ``bunny``, at par
    # from ``dragon`` on, where the sort is the call (2026-11-08).
    e = wp.int32(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    base = e * 2
    out_keys[base] = kernel_array.csr_key(a, b, n, n)
    out_order[base] = base
    out_keys[base + 1] = kernel_array.csr_key(b, a, n, n)
    out_order[base + 1] = base + 1
    weight = wp.float32(1.0)
    if weights.shape[0] > 0:
        weight = weights[e]
    out_values[base] = weight
    out_values[base + 1] = weight


@wp.func
def edge_in_range(edges: wp.array2d[wp.int32], e: wp.int32, n: wp.int32) -> wp.int32:
    # ``1`` when both endpoints of edge row ``e`` lie in ``[0, n)``, else ``0``: the range check
    # the validated edge-list entry points of ``graph`` fold into the first kernel that indexes a
    # node table by the endpoints, which then skips the row and raises a flag the wrapper reads
    # back -- where a separate ``minmax`` of the edges was a launch, an allocation and a readback
    # of its own. One unsigned compare per endpoint covers both ends of the range.
    a = wp.uint32(edges[e, 0])
    b = wp.uint32(edges[e, 1])
    bound = wp.uint32(n)
    return wp.where(a < bound and b < bound, wp.int32(1), wp.int32(0))


@wp.kernel
def count_edge_endpoints_checked(
    edges: wp.array2d[wp.int32], out_flag_and_counts: wp.array[wp.int32]
) -> None:
    # ``graph.edges_to_neighbor_lists``' degree count under ``validate``, node ``v``'s count in
    # entry ``v + 1``: each row in range adds one to both endpoints' counts
    # (``scatter.count_occurrences`` over the flattened edges, the unchecked form), and a row
    # outside sets the flag in entry ``0`` instead of writing past the counts. Launched on the
    # offsets buffer from its entry 1, whose zero (the flag clear) is the first row's start for
    # ``scatter.claim_row_slot``. Launch over the edge rows.
    e = wp.int32(wp.tid())
    n = out_flag_and_counts.shape[0] - 1
    if edge_in_range(edges, e, n) == 0:
        out_flag_and_counts[0] = 1
        return
    wp.atomic_add(out_flag_and_counts, edges[e, 0] + 1, 1)
    wp.atomic_add(out_flag_and_counts, edges[e, 1] + 1, 1)


@wp.kernel
def scatter_neighbor_lists(
    edges: wp.array2d[wp.int32], out_offsets: wp.array[wp.int32], out_neighbors: wp.array[wp.int32]
) -> None:
    # Both directed entries of each undirected edge, into the CSR row the counting pass sized --
    # the payload half of ``graph.edges_to_neighbor_lists``, over an offsets buffer holding each
    # row's start in entry ``v + 1`` (``scatter.claim_row_slot``), which ends as the offsets. Entry
    # 0, which no claim touches, is written here: ``graph.edges_to_neighbor_lists_from_counts``
    # scans the starts in behind it and zeroes nothing. The same counting-sort fill as
    # ``adjacency.scatter_vertex_faces``; what differs is the payload, which is the
    # *other endpoint* here and the owning face there.
    #
    # The column order within a row is arrival order and **not sorted** -- nor stable, so two runs
    # return different permutations. That is deliberate and is what this build costs less than
    # ``bsr_from_triplets`` for. ``graph.edges_to_neighbor_lists(sort_rows=True)`` pins it with one
    # ``array.sort_segments`` launch, which makes the buffer identical to ``graph.edges_to_csr``'s;
    # that is what ``neighbors.geodesic_ball`` asks for, because it emits its BFS queue in visit
    # order.
    e = wp.int32(wp.tid())
    if e == 0:
        out_offsets[0] = 0
    a = edges[e, 0]
    b = edges[e, 1]
    out_neighbors[claim_row_slot(out_offsets, a)] = b
    out_neighbors[claim_row_slot(out_offsets, b)] = a


@wp.kernel
def ecl_init_parent_edges_checked(edges: wp.array2d[wp.int32], parents: wp.array[wp.int32]) -> None:
    # ``connected_components.ecl_init_parent_edges`` under ``validate``, over an identity
    # ``parents`` one entry longer than the node count: a row in range pre-hooks as there
    # (``ecl_prehook_pair``), and a row outside sets that last entry to ``-1`` -- the identity put
    # the node count there -- instead of writing past the forest. The wrapper reads the entry back
    # before the hook, which indexes by the raw endpoints.
    e = wp.int32(wp.tid())
    n = parents.shape[0] - 1
    if edge_in_range(edges, e, n) == 0:
        parents[n] = -1
        return
    ecl_prehook_pair(parents, edges[e, 0], edges[e, 1])


@wp.kernel
def ecl_init_parent_parity_checked(out_words: wp.array[wp.int32]) -> None:
    # ``connected_components.ecl_init_parent_parity`` over one more word than the node count: the
    # flag ``ecl_hook_parity_checked`` raises, laid down clear (``0``) in the same launch.
    v = wp.int32(wp.tid())
    n = out_words.shape[0] - 1
    out_words[v] = wp.where(v < n, parity_root_word(v), wp.int32(0))


@wp.kernel
def ecl_hook_parity_checked(
    edges: wp.array2d[wp.int32], signs: wp.array[wp.int32], words: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_hook_parity`` under ``validate``: a row whose endpoints lie in the
    # forest and whose sign is ``0`` or ``1`` hooks as there (``ecl_hook_pair_parity``), and any
    # other row is skipped and sets the flag word past the forest, read back after the flatten. The
    # skip is what keeps the forest in range; the flag is what makes the wrapper raise.
    e = wp.int32(wp.tid())
    n = words.shape[0] - 1
    sign = signs[e]
    if edge_in_range(edges, e, n) == 0 or wp.uint32(sign) > wp.uint32(1):
        words[n] = 1
        return
    ecl_hook_pair_parity(words, edges[e, 0], edges[e, 1], sign)


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


# ``graph.shortest_path_envelope``'s slot past the shared round-loop word (``kernels/array.py``):
# raised by a pass that met a negative weight, read back with the first pass's condition.
ENVELOPE_NEGATIVE_WEIGHT = wp.int32(3)
ENVELOPE_STATE_SIZE = 4  # plain: a Python-scope read length


@wp.func
def envelope_relaxed_label(
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    labels: wp.array[wp.float32],
) -> tuple[wp.float32, wp.float32]:
    # Node ``i``'s label after one Bellman-Ford relaxation against ``labels``, and the smallest
    # weight in its row (the negative-weight guard): the per-pass kernel's entry and the serial
    # loop's, so the two agree to the bit.
    best = labels[i]
    lowest = wp.float32(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        weight = weights[k]
        lowest = wp.min(lowest, weight)
        relaxed = labels[columns[k]] + weight
        if relaxed < best:
            best = relaxed
    return best, lowest


@wp.kernel
def shortest_path_envelope_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    labels: wp.array[wp.float32],
    out_labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # One Bellman-Ford relaxation of ``q_i <= q_j + w(i, j)``, one thread per node, storing a label
    # only where it went down. (VCG ``UpdateQuality::VertexSaturate`` is this loop.)
    #
    # Two forms, by what ``out_labels`` is. **In place** (``out_labels`` *is* ``labels``, the
    # uncapped call): a thread reads whatever its neighbours hold, this pass's labels or the last's,
    # and only it writes its own. Every value read is an upper bound the iteration reaches anyway,
    # and ``min(x, y + w)`` is monotone in rounded ``float32`` for ``w >= 0``, so the labels fall
    # to the same greatest fixpoint as the Jacobi passes, bit for bit, in no more passes; what the
    # interleaving moves is only how many. It drops the Jacobi form's copy a pass and its buffer:
    # 0.70-0.88x the call from ``sphere_small`` to ``lucy``, a scrambled ``dragon`` included
    # (2026-11-08). **Jacobi** (a second buffer holding ``labels``' values, copied back after each
    # pass, under an explicit pass cap): a pass is a pure function of the previous labels, so a
    # capped answer is exactly that many passes on every device.
    #
    # Declined, a block a range of nodes relaxing in place behind a barrier until its range stops
    # moving (``wp.tile_sum`` of the moved flags, at most 8 rounds a pass): passes fell 668 -> 168
    # on ``dragon`` and the call 0.58x there and 0.50x on ``lucy``, but it was 1.2-1.7x from
    # ``sphere_small`` to ``bunny`` and 1.18x on a scrambled ``dragon``, where a front block's
    # rounds each move a few nodes while it holds its SM (2026-11-08). Two rounds: 0.51x ``lucy``,
    # 0.75x ``dragon``, 1.03-1.14x on icospheres.
    #
    # Declined, the rounds after a first full pass pushing from a frontier, the nodes the last
    # round lowered (``atomic_min``, listed once a round by a round stamp, a fixed grid striding
    # over a ping-pong list): the rounds equal these passes (667 / 668 on ``dragon``), but on
    # Euclidean weights each round re-lowers a tenth of the graph (31.8 M pushes for 437 k nodes,
    # fronts of 47 k), so a scattered atomic round cost 1.4-1.7x a coalesced pull (0.81x only on
    # a scrambled ``dragon``). Near-far on top (push only within ``c`` times the largest edge
    # weight above the front's least label, carry the rest) multiplied the rounds instead:
    # 1.3-3.8x for ``c`` in 0.5-4, 0.67-0.88x on the scrambled ``dragon`` (2026-11-08). Both were
    # byte-identical to these passes.
    #
    # ``out_state`` is the shared round-loop word of ``kernels/array.py``; raising
    # ``LOOP_PROGRESS`` is this pass's "something moved", which the closing ``array.loop_advance``
    # turns into the next round's condition and clears. A plain store, not an atomic: the slot only
    # ever goes 0 -> 1 within a round, so every writer writes the same value. A negative weight
    # raises ``ENVELOPE_NEGATIVE_WEIGHT`` the same way, for the wrapper's read after the first pass.
    i = wp.int32(wp.tid())
    best, lowest = envelope_relaxed_label(i, offsets, columns, weights, labels)
    if lowest < 0.0:
        out_state[ENVELOPE_NEGATIVE_WEIGHT] = 1
    if best < labels[i]:
        out_labels[i] = best
        out_state[kernel_array.LOOP_PROGRESS] = 1


# Rounds a block of ``shortest_path_envelope_patch_pass`` relaxes its patch in one pass, at most
# (plain: a launch argument). Swept against the plain passes on CUDA (2026-11-08, Hilbert order),
# a distance from one source: 4 rounds 0.60-0.92x but 1.07x on ``dragon``, 8 0.62-0.92x, 16
# 0.51-0.75x on the dragons but 1.08x on ``bunny_decimated``; a spike and a random field lost at
# every count, least at 4-8 (1.0-1.5x).
ENVELOPE_PATCH_ROUNDS = 8


@wp.kernel
def shortest_path_envelope_patch_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    order: wp.array[wp.int32],
    rounds: wp.int32,
    labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # The uncapped CUDA pass in place over a caller's node order, tiled: when consecutive nodes of
    # ``order`` are near each other in the graph (a Hilbert order of the vertices), a block's
    # nodes are a patch, and the block relaxes them round after round (``wp.tile_sum`` the barrier)
    # until none moves or ``rounds`` ran, so a pass carries a label across the patch rather than an
    # edge. Any schedule of in-place relaxations falls to the passes' fixpoint, bit for bit. A
    # distance from one source, CUDA new/old with the wrapper's plain first pass: 0.63-0.81x from
    # ``sphere_small`` to a scrambled ``dragon`` (0.97x on ``bunny_decimated``); a spike or a random
    # field on ``dragon`` 1.2-1.37x, where most blocks move and run every round. In
    # the graph's own order a block is no patch and the rounds lost (declined at
    # ``shortest_path_envelope_pass``); ``flip_geodesics.relax_distances`` is the same pass over a
    # halfedge mesh, which won there in Hilbert order.
    block, lane = wp.tid()
    i = block * wp.block_dim() + lane
    v = wp.int32(-1)
    if i < labels.shape[0]:
        v = order[i]
    moved_any = wp.int32(0)
    for _round in range(rounds):
        moved = wp.int32(0)
        if v >= 0:
            best, lowest = envelope_relaxed_label(v, offsets, columns, weights, labels)
            if lowest < 0.0:
                out_state[ENVELOPE_NEGATIVE_WEIGHT] = 1
            if best < labels[v]:
                labels[v] = best
                moved = wp.int32(1)
        if wp.tile_sum(wp.tile(moved))[0] == 0:
            break
        moved_any = wp.int32(1)
    if moved_any != 0 and lane == 0:
        out_state[kernel_array.LOOP_PROGRESS] = 1


@wp.kernel
def shortest_path_envelope_serial(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    max_pass_count: wp.int32,
    labels: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # Every capped ``shortest_path_envelope_pass`` in one launch and one thread: the CPU device's
    # form, where a launch's threads run one after another anyway, so a launch and a progress
    # readback a pass buy no parallelism. Each pass reads one buffer whole and writes the other
    # (``labels`` -> ``scratch`` on even passes), each entry ``envelope_relaxed_label``; the loop
    # stops after the first pass that lowered nothing, or at the cap. These are the per-pass Jacobi
    # loop's labels to the bit, and a result left in ``scratch`` is copied back. A negative weight
    # stops the first pass and raises ``ENVELOPE_NEGATIVE_WEIGHT``. Uncapped, the CPU device takes
    # ``shortest_path_envelope_dijkstra``; before it, the passes ran in place there, a Gauss-Seidel
    # sweep in node order (0.5-0.85x these passes; sweeping odd passes down the nodes was 0.55x on
    # ``bunny`` but 1.7-1.8x on icospheres).
    n = labels.shape[0]
    parity = wp.int32(0)
    for _pass in range(max_pass_count):
        moved = wp.int32(0)
        for i in range(n):
            if parity == 0:
                best, lowest = envelope_relaxed_label(i, offsets, columns, weights, labels)
                if best < labels[i]:
                    moved = wp.int32(1)
                scratch[i] = best
            else:
                best, lowest = envelope_relaxed_label(i, offsets, columns, weights, scratch)
                if best < scratch[i]:
                    moved = wp.int32(1)
                labels[i] = best
            if lowest < 0.0:
                out_state[ENVELOPE_NEGATIVE_WEIGHT] = 1
                return
        parity = 1 - parity
        if moved == 0:
            break
    if parity == 1:
        for i in range(n):
            labels[i] = scratch[i]


# In-place sweeps ``shortest_path_envelope_dijkstra`` makes before its heap (plain: a launch
# argument). A field that settles in a few sweeps (a spike, a random field) stops there; one that
# needs many (a distance from one source) hands the heap labels that are already lower. Swept on
# the CPU device against the in-place sweeps alone, a distance / a random field / a spike: 4 gives
# 0.04-0.43x / 0.39-0.65x / 0.96-1.09x from ``sphere_small`` to a scrambled ``dragon``, with
# ``sphere_small``'s random field at 1.08x; 2 gave 0.03-0.35x / 0.41-1.08x / 0.94-1.11x, 8 lost
# on every distance (2026-11-08).
ENVELOPE_CPU_SWEEPS = 4


@wp.kernel
def shortest_path_envelope_dijkstra(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    sweeps: wp.int32,
    labels: wp.array[wp.float32],
    heap: wp.array[wp.int32],
    position: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # The uncapped envelope on the CPU device, in one launch and one thread, lowering ``labels`` in
    # place: up to ``sweeps`` Gauss-Seidel sweeps in node order, and if those leave a label still
    # moving, Dijkstra keyed by the labels (``array.indexed_heap_*``). The heap starts with the
    # nodes that can lower a neighbour, found in one sweep, and takes another node only when it is
    # lowered: labels only fall, so a node that lowers no neighbour at the start never will.
    # Settling each node once where a sweep visits every node, it took a distance from one source
    # to 0.03-0.31x the sweeps alone, but heaping every node lost 2.7-4.3x on a spike, and a
    # random field on an icosphere, which a few sweeps settle, 1.7-1.9x with the improvers alone:
    # hence the sweeps first (``ENVELOPE_CPU_SWEEPS``).
    #
    # The labels are the passes' to the bit. The sweeps leave labels between the fixpoint and the
    # input, so the fixpoint below them is the input's; with ``w >= 0`` a path's rounded sums
    # never fall along it, so the node popped next holds the least sum over its paths, which is
    # that fixpoint. A node holding NaN, which no relaxation lowers or propagates, never enters. A
    # negative weight raises ``ENVELOPE_NEGATIVE_WEIGHT`` in the first sweep. ``heap`` and
    # ``position`` are scratch, one entry a node; ``position`` reads ``-2`` off the heap and ``-1``
    # once popped.
    n = labels.shape[0]
    for _sweep in range(sweeps):
        moved = wp.int32(0)
        for i in range(n):
            best, lowest = envelope_relaxed_label(i, offsets, columns, weights, labels)
            if lowest < 0.0:
                out_state[ENVELOPE_NEGATIVE_WEIGHT] = 1
                return
            if best < labels[i]:
                labels[i] = best
                moved = wp.int32(1)
        if moved == 0:
            return
    size = wp.int32(0)
    for v in range(n):
        position[v] = -2
    for u in range(n):
        own = labels[u]
        for k in range(offsets[u], offsets[u + 1]):
            if own + weights[k] < labels[columns[k]]:
                heap[size] = u
                kernel_array.indexed_heap_sift_up(heap, position, labels, size)
                size += 1
                break
    while size > 0:
        u = kernel_array.indexed_heap_pop(heap, position, labels, size)
        size -= 1
        settled = labels[u]
        for k in range(offsets[u], offsets[u + 1]):
            v = columns[k]
            reach = settled + weights[k]
            if reach < labels[v]:
                labels[v] = reach
                if position[v] >= 0:
                    kernel_array.indexed_heap_sift_up(heap, position, labels, position[v])
                else:
                    heap[size] = v
                    kernel_array.indexed_heap_sift_up(heap, position, labels, size)
                    size += 1


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
    out_flag: wp.array[wp.vec2i],
) -> None:
    # ``successor_cycles``' successor table and its tails in one pass over the edges, both
    # node-sized and seeded ``-1``. The tails are listed in node space -- entry ``a`` is ``a`` when
    # ``a`` has an edge out, ``-1`` (skipped by the ranking) otherwise -- so the ranking's threads
    # walk the nodes in ascending order, which keeps a mesh-ordered graph's reads and writes local
    # (listed in edge order, a shuffled edge list made the ranking 4x slower at 3 M nodes), and a
    # node with several edges out (malformed input) is listed once. A row with an endpoint outside
    # the table is skipped and raises ``out_flag[0]`` (the zeroed entry just past the ranking's
    # cycle counts, read back with their totals), so the ranking runs on an in-range table and the
    # range check costs neither a launch nor a readback of its own.
    e = wp.int32(wp.tid())
    if edge_in_range(directed_edges, e, out_next.shape[0]) == 0:
        out_flag[0] = wp.vec2i(1, 0)
        return
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
def closed_cycle_windows_serial(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    mode: wp.int32,
    out_windows: wp.array[wp.vec3i],
    out_scratch: wp.array[wp.vec3i],
    out_cycle_counts: wp.array[wp.vec2i],
) -> None:
    # Every round of ``closed_cycle_windows`` / ``closed_cycle_jump``, the last round's counts and
    # the caller's inclusive scan of them, in one launch and one thread: the CPU device's form,
    # where a launch's threads run one after another, so the rounds buy no parallelism and the
    # pointer chases do ``hops`` times the work of one walk. Each node is resolved once, as the
    # finished windows of the parallel rounds describe it (their window covers every node a walk
    # can reach, ``pointer_jump_schedule``): a node whose walk reaches a node with no successor is
    # on a chain (``current = -1``); otherwise ``start`` is the smallest node of its walk -- its
    # cycle's, or a smaller node on its way in -- and ``dist`` the hops to its first occurrence.
    # Only the sign of ``current`` is read after the rounds, so a live window stores ``0`` there.
    # ``out_scratch`` holds, per node, its state (``0`` unseen, ``1`` on the current walk, ``2``
    # resolved) in component 0, and the current walk's nodes in order in component 1.
    n = next_node.shape[0]
    for i in range(n):
        out_scratch[i] = wp.vec3i(0, 0, 0)
    for t in range(tails.shape[0]):
        v = tails[t]
        if v < 0:
            continue
        if out_scratch[v][0] == 2:
            continue
        depth = wp.int32(0)
        u = v
        cycle_at = wp.int32(-1)
        walking = wp.int32(1)
        while walking == 1:
            state = out_scratch[u][0]
            if state == 2:
                walking = wp.int32(0)
            elif state == 1:
                cycle_at = u
                walking = wp.int32(0)
            elif next_node[u] < 0:
                out_windows[u] = wp.vec3i(-1, u, 0)
                out_scratch[u] = wp.vec3i(2, out_scratch[u][1], 0)
                walking = wp.int32(0)
            else:
                out_scratch[u] = wp.vec3i(1, out_scratch[u][1], 0)
                out_scratch[depth] = wp.vec3i(out_scratch[depth][0], u, 0)
                depth += 1
                u = next_node[u]
        if cycle_at >= 0:
            # The walk closed on ``cycle_at``: the cycle is the walk from there, every node of it
            # starting at its smallest node.
            first = wp.int32(0)
            while out_scratch[first][1] != cycle_at:
                first += 1
            length = depth - first
            lowest = cycle_at
            lowest_at = first
            for j in range(first, depth):
                node = out_scratch[j][1]
                if node < lowest:
                    lowest = node
                    lowest_at = j
            for j in range(first, depth):
                node = out_scratch[j][1]
                out_windows[node] = wp.vec3i(0, lowest, (lowest_at - j + length) % length)
                out_scratch[node] = wp.vec3i(2, out_scratch[node][1], 0)
            depth = first
        # The nodes on the way in, last first: each takes its successor's window one hop longer,
        # or starts itself where it is smaller than everything after it.
        k = depth - 1
        while k >= 0:
            node = out_scratch[k][1]
            after = out_windows[next_node[node]]
            if after[0] < 0:
                out_windows[node] = wp.vec3i(-1, node, 0)
            elif node < after[1]:
                out_windows[node] = wp.vec3i(0, node, 0)
            else:
                out_windows[node] = wp.vec3i(0, after[1], after[2] + 1)
            out_scratch[node] = wp.vec3i(2, out_scratch[node][1], 0)
            k -= 1
    for t in range(tails.shape[0]):
        v = tails[t]
        if v >= 0:
            own = out_windows[v]
            if closed_cycle_is_kept(own[0], own[1], mode):
                start = own[1]
                out_cycle_counts[start] = out_cycle_counts[start] + wp.vec2i(
                    wp.where(v == start, 1, 0), wp.int32(1)
                )
    running = wp.vec2i(0, 0)
    for i in range(out_cycle_counts.shape[0]):
        running = running + out_cycle_counts[i]
        out_cycle_counts[i] = running


@wp.kernel
def mark_tail_bits(
    tails: wp.array[wp.int32], out_bits: wp.array[wp.uint32], out_word_counts: wp.array[wp.int32]
) -> None:
    # A bit a node of a successor table, set on the nodes ``tails`` lists (``out_bits`` zeroed),
    # each word's new bits counted into ``out_word_counts`` (zeroed) for the caller to scan in place
    # into ``bit_rank``'s table. A node may be listed more than once, so the set is
    # ``or_counted_bit``'s, which counts a bit once however often it is set; every lane calls it,
    # an unlisted one with ``bit`` false, so that a warp's lanes stay converged.
    v = tails[wp.int32(wp.tid())]
    kernel_array.or_counted_bit(out_bits, out_word_counts, wp.max(v, 0), v >= 0)


@wp.kernel
def compact_successor_table(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    bits: wp.array[wp.uint32],
    word_ranks: wp.array[wp.int32],
    node_values: wp.array[wp.int32],
    out_tails: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
    out_values: wp.array[wp.int32],
) -> None:
    # A successor table restated over its listed nodes alone: a node's new number is its rank
    # among them (``kernel_array.bit_rank`` over ``mark_tail_bits``' bits), which is monotone in the
    # node, so every cycle keeps its smallest node as its start and the cycles their order. A
    # successor off the listed nodes (a halfedge walk's rotation can dead-end on a halfedge that is
    # not a boundary one, whose own successor is ``-1``) becomes ``-1`` itself: the chain ends a
    # step sooner and is dropped all the same. ``out_values`` is what a ranked node is written out
    # as: ``node_values`` of it when given, else the node itself.
    k = wp.int32(wp.tid())
    v = tails[k]
    if v < 0:
        out_tails[k] = -1
        return
    r = kernel_array.bit_rank(bits, word_ranks, v)
    out_tails[k] = r
    successor = next_node[v]
    compact = wp.int32(-1)
    if successor >= 0:
        if kernel_array.bit_is_set(bits, successor):
            compact = kernel_array.bit_rank(bits, word_ranks, successor)
    out_next[r] = compact
    if node_values.shape[0] > 0:
        out_values[r] = node_values[v]
    else:
        out_values[r] = v


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
