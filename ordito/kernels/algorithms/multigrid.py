"""
Smoothed-aggregation algebraic multigrid: the setup passes and the V-cycle.

**Why a hierarchy at all.** Jacobi-preconditioned conjugate gradient's iteration count grows with
the mesh, roughly in proportion to the unknown count on an ill-conditioned operator. A multigrid
V-cycle attacks the low-frequency error the smoother cannot see, so the count stops growing; the
whole item is whether one cycle costs less than the iterations it removes, which is a *launch*
question on this hardware and not an arithmetic one. See ``ordito.linalg`` for the gate. The
least-squares operator ``smoothing.smooth_region`` builds was this hierarchy's first customer and
is no longer one: a *squared* operator is better served by the square of a second-order
preconditioner (``linalg.squared_laplacian_preconditioner``) than by aggregating the square itself.

**Aggregation is the only part that is not a library call**, and it is a parallel maximal
independent set, which this package already runs twice on device (``sample.dart_select_minima``'s
randomized-priority selection and ``remesh``'s hashed-key independent set). The roots of a
**distance-2** MIS on the operator's off-diagonal graph are at least three hops apart, so their
one-rings are disjoint and every remaining node is within two hops of exactly one candidate root;
spreading the root's label two hops therefore tiles the graph into aggregates. Distance-2 is reached
without building the squared graph: propagating the lexicographic maximum of
``(state, priority, index)`` over one-hop neighbours **twice** gives every node the maximum over its
two-hop ball, which is the Bell/Dalton/Olson formulation.

Measured against ``pyamg``'s serial ``standard_aggregation`` on the same operator, same smoother,
same cycle and same coarse solve -- the comparison that decides whether the parallel aggregation
gives anything up -- it costs about a tenth more iterations. That is the price of the parallelism
and it is small.

The tentative prolongator is one entry per row (the constant near-nullspace vector, normalized
per aggregate), so the smoothed prolongator ``P = (I - w D^-1 A) P0`` is ``A``'s own entries
re-keyed by their column's aggregate and summed by key; the coarse operator is the Galerkin
product ``P^T A P``, ``A P`` summed from its ``A_ij P_jl`` triplets and ``P^T (A P)`` from the
outer product of each fine row's two sparse rows, every contribution written as a ``csr_key``
(no ``(row, col)`` triplet buffers). No ``bsr_mm`` runs in the setup. Against the
earlier ``bsr_mm`` / ``bsr_axpy`` chain the prolongator is 1.9-4.0x faster on the dragon and bunny
cotangent Laplacians, with the same patterns and values to round-off; see ``product_keys`` and
``galerkin_keys`` for the products. Operator complexity comes out just above 1, so the coarse
levels are nearly free and a cycle's cost is its fine level.

The per-level inverse diagonal is ``array.inverse_or_one`` of the operator's diagonal, applied
where it is read rather than written to a buffer first. It is not a copy of the conjugate
gradient's: the quantity is the same one the Jacobi preconditioner needs, down to mapping a zero
diagonal to 1 rather than to infinity -- which the least-squares operators here rely on, since they
carry empty rows for free vertices no equation reaches. The strength test's ``sqrt(|A_ii|)`` is
``array.sqrt_abs`` over the same diagonal.
"""

from typing import Any

import warp as wp

from ordito.constants import TILE_1D
from ordito.kernels.array import (
    OverloadTable,
    csr_key_nonzero,
    csr_row_long,
    declare_block_dims,
    element_priority,
    inverse_or_one,
    sqrt_abs,
)
from ordito.kernels.linalg import (
    csr_row_diagonal,
    csr_row_diagonal_block,
    csr_row_dot,
    listed_long_pieces,
    long_row_piece_total,
)
from ordito.kernels.reduce import block_barrier, block_chunk, block_max, block_sum

wp.set_module_options({"enable_backward": False})

# Node states for the distance-2 maximal independent set. The encoding is ordered rather than
# arbitrary: a root must win any maximum (it vetoes every node in its two-hop ball) and an excluded
# node must lose to every undecided one (it can no longer veto anything).
MG_EXCLUDED = wp.int32(0)
MG_UNDECIDED = wp.int32(1)
MG_ROOT = wp.int32(2)

# Layout of the packed comparison key: state in bits 60-61, priority in bits 32-59, index in bits
# 0-31. Two hops of a plain integer ``max`` then implement the lexicographic order, and the key
# stays positive so a signed ``wp.int64`` compares correctly. 28 bits of priority is plenty -- the
# index breaks any tie, which is what makes the aggregation independent of thread order.
_MG_STATE_SHIFT = wp.int64(60)
_MG_PRIORITY_SHIFT = wp.int64(32)
_MG_PRIORITY_MASK = wp.uint32(0x0FFFFFFF)

MG_UNAGGREGATED = wp.int32(-1)


@wp.func
def mg_is_strong(
    value: wp.float64,
    scaled_diagonal_row: wp.float64,
    scaled_diagonal_column: wp.float64,
    theta: wp.float64,
) -> wp.bool:
    """Whether ``A_ij`` is a strong connection: ``|A_ij| >= theta * sqrt(A_ii * A_jj)``."""
    # ``scaled_diagonal`` carries ``sqrt(|A_ii|)``, so the product is the geometric mean and no
    # square root runs per edge. At ``theta = 0`` this is unconditionally true -- including for an
    # explicit zero, since ``0 >= 0`` -- which is what makes the unfiltered aggregation the exact
    # ``theta = 0`` case of this one rather than a separate code path.
    return wp.abs(value) >= theta * scaled_diagonal_row * scaled_diagonal_column


@wp.func
def mg_strong_neighbor(
    i: wp.int32,
    j: wp.int32,
    value: wp.float64,
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
) -> wp.bool:
    # Whether row ``i``'s entry ``value`` at column ``j`` is an edge of the *strong off-diagonal*
    # graph: the diagonal is skipped because a node is not its own neighbour, and a weak entry
    # (``mg_is_strong``) is not an edge. The one definition of the graph both walks run on -- the
    # independent set's propagation (``mis_ball_max``) and the label spread
    # (``spread_aggregate_labels``) -- whose agreement is what makes the aggregates tile the graph.
    return j != i and mg_is_strong(value, scaled_diagonal[i], scaled_diagonal[j], theta)


@wp.func
def mis_key(state: wp.int32, priority: wp.uint32, index: wp.int32) -> wp.int64:
    masked = priority & _MG_PRIORITY_MASK
    return (
        (wp.int64(state) << _MG_STATE_SHIFT)
        | (wp.int64(masked) << _MG_PRIORITY_SHIFT)
        | wp.int64(index)
    )


@wp.func
def mis_root_flag(state: wp.int32) -> wp.int32:
    """Whether ``state`` claims an aggregate, as the 0/1 flag ``wp.utils.array_scan`` wants."""
    # An inclusive scan of these numbers the aggregates consecutively and its last element is the
    # aggregate count.
    #
    # The test is "not excluded" rather than "is a root" so that a node still undecided when the
    # round cap is reached becomes an aggregate of its own instead of an unaggregated hole. The
    # selection normally settles well inside the cap and the two readings then coincide.
    #
    # ``mis_propagate_decide`` writes it as it decides. It returns the flag directly rather than
    # composing an inequality with an ``array_cast(bool -> int32)``, which would be two device
    # passes and a second buffer.
    return wp.where(state != MG_EXCLUDED, wp.int32(1), wp.int32(0))


@wp.func
def mis_entry_key(
    j: wp.int32,
    from_state: wp.int32,
    key: wp.array[wp.int64],
    state: wp.array[wp.int32],
    priority: wp.array[wp.uint32],
) -> wp.int64:
    # Node ``j``'s packed key as a hop reads it: formed from its state on a round's first hop, read
    # from the first hop's output on the second. ``from_state`` is a literal at each call site, so
    # the branch folds at compile time.
    if from_state != 0:
        return mis_key(state[j], priority[j], j)
    return key[j]


@wp.func
def mis_ball_max(
    i: wp.int32,
    from_state: wp.int32,
    key: wp.array[wp.int64],
    state: wp.array[wp.int32],
    priority: wp.array[wp.uint32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
) -> wp.int64:
    # One hop of the lexicographic maximum over the operator's *strong off-diagonal* graph
    # (``mg_strong_neighbor``) -- the node's own key is folded in separately so the reduction is
    # over the closed neighbourhood. Two hops give the maximum
    # over the two-hop ball, which is the distance-2 test without a squared graph.
    #
    # The strength test is applied here rather than by materializing a filtered graph, so a level
    # pays no extra allocation and ``theta = 0`` is bit-exactly the unfiltered aggregation. Shared
    # by the round's two kernels below, which differ in where a neighbour's key comes from
    # (``mis_entry_key``) and in what the second does with the maximum.
    #
    # Declined (2026-11-11): the walk four entries a step (``array.row_chunk_slots``, every gather
    # of a step issued together, the strength test masking them) took the round kernels 0.80x on
    # bunny but cost the wide fine level 1.07x on dragon (the unconditional key loads), and
    # ``multigrid_preconditioner`` calls read par (0.98-1.01x) on sphere_med to dragon; pooling the
    # undecided and root counts (``reduce.peer_counted_add``) was 0.96-0.97x on the decide kernel
    # and par on the call too.
    best = mis_entry_key(i, from_state, key, state, priority)
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        if mg_strong_neighbor(i, j, values[k], scaled_diagonal, theta):
            best = wp.max(best, mis_entry_key(j, from_state, key, state, priority))
    return best


# Rows a block of a multigrid level's setup kernels holds, a row a lane on CUDA: a lane takes its
# rows in order, and a long row (``array.csr_row_long``, a fan's hub) is left to the whole block
# after, its lanes striding the row, so a block without one pays one reduction. On the CPU device
# the block's one lane walks them all.
MG_ROWS_PER_BLOCK = 256


@wp.func
def mis_ball_max_block(
    i: wp.int32,
    from_state: wp.int32,
    key: wp.array[wp.int64],
    state: wp.array[wp.int32],
    priority: wp.array[wp.uint32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
    lane: wp.int32,
) -> wp.int64:
    # ``mis_ball_max`` by a whole block, its lanes striding the row (every lane must call it). A
    # maximum, so the same value whatever the order.
    best = mis_entry_key(i, from_state, key, state, priority)
    for k in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        j = columns[k]
        if mg_strong_neighbor(i, j, values[k], scaled_diagonal, theta):
            best = wp.max(best, mis_entry_key(j, from_state, key, state, priority))
    return block_max(best)


@wp.kernel
def mis_propagate_states(
    state: wp.array[wp.int32],
    priority: wp.array[wp.uint32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
    out_key: wp.array[wp.int64],
    out_undecided: wp.array[wp.int32],
) -> None:
    # A round's first hop, reading every key straight from ``(state, priority, index)``: the packed
    # key is an elementwise function of the state, so no seeding pass writes it first.
    # ``mis_propagate_decide`` is the round's second hop; the undecided counter it increments is
    # cleared here, the launch before, so no memset runs per round. Launch tiled over
    # ``MG_ROWS_PER_BLOCK`` rows a block.
    blk, lane = wp.tid()
    first, rows = block_chunk(offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if i == 0:
            out_undecided[0] = 0
        if csr_row_long(offsets, i):
            found = 1
            continue
        out_key[i] = mis_ball_max(
            i, wp.int32(1), out_key, state, priority, offsets, columns, values, scaled_diagonal,
            theta,
        )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(offsets, i):
            best = mis_ball_max_block(
                i, wp.int32(1), out_key, state, priority, offsets, columns, values,
                scaled_diagonal, theta, lane,
            )  # fmt: skip
            if lane == 0:
                out_key[i] = best


@wp.func
def mis_decide(
    i: wp.int32,
    current: wp.int32,
    best: wp.int64,
    priority: wp.array[wp.uint32],
    out_state: wp.array[wp.int32],
    out_root_flags: wp.array[wp.int32],
    out_undecided: wp.array[wp.int32],
) -> None:
    # An undecided node's verdict from its two-hop maximum ``best`` (``mis_propagate_decide``).
    verdict = MG_UNDECIDED
    if wp.int32(best >> _MG_STATE_SHIFT) == MG_ROOT:
        verdict = MG_EXCLUDED
    elif best == mis_key(current, priority[i], i):
        verdict = MG_ROOT
    out_state[i] = verdict
    out_root_flags[i] = mis_root_flag(verdict)
    if verdict == MG_UNDECIDED:
        wp.atomic_add(out_undecided, 0, wp.int32(1))
    elif verdict == MG_ROOT:
        # The roots so far, in the counter's second slot (cleared by ``mis_level_setup``): with the
        # nodes still undecided, which become aggregates of their own, it is the aggregate count,
        # read with the undecided count rather than off the flags' scan.
        wp.atomic_add(out_undecided, 1, wp.int32(1))


@wp.kernel
def mis_propagate_decide(
    key: wp.array[wp.int64],
    priority: wp.array[wp.uint32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
    state: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
    out_root_flags: wp.array[wp.int32],
    out_undecided: wp.array[wp.int32],
) -> None:
    # A round's second hop and its verdict in one pass: the two-hop maximum is only ever read at
    # the node that formed it, so it never needs to reach memory. A root anywhere in the two-hop
    # ball excludes the node, and otherwise being the maximum of that ball makes it a root.
    # Anything else waits for the next round, and ``out_undecided`` is what tells the host whether
    # there is one. A node already decided keeps its state and skips the hop, whose maximum it
    # would not read.
    #
    # ``out_root_flags`` is ``mis_root_flag`` of the new state, written by the round that decides it
    # (and by every round a node stays undecided): a decided node's flag never changes, so after
    # whichever round ends the loop it holds every node's final flag, with no map over the final
    # state.
    #
    # Launch tiled over ``MG_ROWS_PER_BLOCK`` rows a block.
    blk, lane = wp.tid()
    first, rows = block_chunk(offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        current = state[i]
        if current != MG_UNDECIDED:
            out_state[i] = current
            continue
        if csr_row_long(offsets, i):
            found = 1
            continue
        best = mis_ball_max(
            i, wp.int32(0), key, state, priority, offsets, columns, values, scaled_diagonal, theta
        )
        mis_decide(i, current, best, priority, out_state, out_root_flags, out_undecided)
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        current = state[i]
        if current == MG_UNDECIDED and csr_row_long(offsets, i):
            best = mis_ball_max_block(
                i, wp.int32(0), key, state, priority, offsets, columns, values,
                scaled_diagonal, theta, lane,
            )  # fmt: skip
            if lane == 0:
                mis_decide(i, current, best, priority, out_state, out_root_flags, out_undecided)


@wp.func
def aggregate_label(state: wp.int32, scan_pos: wp.int32) -> wp.int32:
    # An excluded node has no aggregate; everything else takes the (0-based) index its inclusive
    # scan position names.
    return wp.where(state != MG_EXCLUDED, scan_pos - wp.int32(1), MG_UNAGGREGATED)


@wp.func
def aggregate_entry_label(
    j: wp.int32,
    from_state: wp.int32,
    label: wp.array[wp.int32],
    state: wp.array[wp.int32],
    scan_pos: wp.array[wp.int32],
) -> wp.int32:
    # Node ``j``'s label as a spread hop reads it: the first hop forms it from ``(state, scan_pos)``
    # (``aggregate_label``), the second reads the first's output. ``from_state`` is a literal at
    # each call site.
    if from_state != 0:
        return aggregate_label(state[j], scan_pos[j])
    return label[j]


@wp.func
def node_weight(weights: wp.array[wp.int32], i: wp.int32) -> wp.int32:
    # How many finest-level rows node ``i`` of a level stands for: ``weights[i]`` below the finest
    # level, ``1`` on it, where ``weights`` is empty. Smoothed aggregation interpolates the
    # near-nullspace vector, the constant on the finest level; the tentative prolongator's
    # normalized columns carry it to the next level as ``sqrt(weight)``, not as a constant, so
    # every level reads it from the weights (``tentative_scale``) rather than assuming it.
    if weights.shape[0] == 0:
        return wp.int32(1)
    return weights[i]


@wp.func
def tentative_scale(weights: wp.array[wp.int32], i: wp.int32) -> wp.float64:
    # Row ``i``'s near-nullspace entry, ``sqrt(node_weight)``: exactly ``1`` on the finest level,
    # so its prolongator is the constant one to the bit.
    if weights.shape[0] == 0:
        return wp.float64(1.0)
    return wp.sqrt(wp.float64(weights[i]))


@wp.kernel
def spread_aggregate_labels(
    label: wp.array[wp.int32],
    state: wp.array[wp.int32],
    scan_pos: wp.array[wp.int32],
    from_state: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
    weights: wp.array[wp.int32],
    out_label: wp.array[wp.int32],
    out_sizes: wp.array[wp.int32],
) -> None:
    # An unlabelled node adopts a neighbour's aggregate, largest id winning so the choice does not
    # depend on thread order. Two launches cover the two hops the MIS guarantees are enough; the
    # first (warp-uniform ``from_state != 0``) reads every label straight from the root flags'
    # scan, so no pass writes the labels first, and ``label`` is not read. The second hop, whose
    # labels are final, also counts each aggregate's rows into the zeroed ``out_sizes`` (the first
    # hop passes ``None`` and never reads it), so no separate counting pass runs over the labels.
    # A row counts its ``node_weight``: the finest rows it stands for, so ``out_sizes`` holds each
    # aggregate's, the next level's weights.
    #
    # The spread walks the same *strong* graph the independent set was selected on
    # (``mg_strong_neighbor``), which is what keeps the tiling consistent: every excluded node has
    # a root within two strong hops precisely because the exclusion came from a strong-graph
    # propagation.
    i = wp.int32(wp.tid())
    best = aggregate_entry_label(i, from_state, label, state, scan_pos)
    if best == MG_UNAGGREGATED:
        for k in range(offsets[i], offsets[i + 1]):
            j = columns[k]
            if mg_strong_neighbor(i, j, values[k], scaled_diagonal, theta):
                best = wp.max(best, aggregate_entry_label(j, from_state, label, state, scan_pos))
    out_label[i] = best
    if from_state == 0 and best != MG_UNAGGREGATED:
        wp.atomic_add(out_sizes, best, node_weight(weights, i))


@wp.func
def tentative_norm(sizes: wp.array[wp.int32], aggregate: wp.int32) -> wp.float64:
    # ``sqrt(sizes[aggregate])``, the norm of the near-nullspace vector over an aggregate's column,
    # ``sizes`` holding each aggregate's finest-row count (``spread_aggregate_labels``): the
    # tentative prolongator ``P0``'s entry in a row of that aggregate is ``tentative_scale`` over
    # it. Returned undivided so a caller scaling by it divides once, as it always has.
    return wp.sqrt(wp.float64(sizes[aggregate]))


@wp.func
def emit_entry(
    row: wp.int32,
    col: wp.int32,
    value: wp.float64,
    slot: wp.int32,
    n_rows: wp.int32,
    n_cols: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # One contribution to a pruned key assembly (``array.csr_from_coo_keys``): its key, the sentinel
    # when the value is exactly zero, the identity payload and the value, all at ``slot``. The three
    # setup assemblies below write nothing else, so no ``(row, col)`` buffers exist to be packed.
    out_keys[slot] = csr_key_nonzero(row, col, value, wp.float64(0.0), n_rows, n_cols)
    out_order[slot] = slot
    out_values[slot] = value


@wp.func
def emit_scaled_row(
    row: wp.int32,
    weight: wp.float64,
    begin: wp.int32,
    end: wp.int32,
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    slot: wp.int32,
    n_rows: wp.int32,
    n_cols: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> wp.int32:
    # ``weight`` times the sparse row ``[begin, end)`` of a CSR, as contributions to output row
    # ``row`` from ``slot`` on; returns the slot after the last. The emission both row-local
    # products below are made of: ``A P`` emits ``A_ij`` times ``P``'s row ``j`` into row ``i``,
    # ``P^T (A P)`` emits ``P_ik`` times ``A P``'s row ``i`` into row ``k``.
    for b in range(begin, end):
        emit_entry(
            row,
            columns[b],
            weight * values[b],
            slot,
            n_rows,
            n_cols,
            out_keys,
            out_order,
            out_values,
        )
        slot += 1
    return slot


@wp.func
def smoothed_prolongator_row(
    i: wp.int32,
    start: wp.int32,
    step: wp.int32,
    with_identity: wp.bool,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    label: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    weights: wp.array[wp.int32],
    damped_inverse_diagonal: wp.array[wp.float64],
    n_aggregates: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # ``smoothed_prolongator_keys``' contributions of row ``i``'s entries ``start``, ``start +
    # step``, ... (each at its own slot), and with ``with_identity`` the identity's.
    n = offsets.shape[0] - 1
    scale = -damped_inverse_diagonal[i]
    for e in range(start, offsets[i + 1], step):
        j = columns[e]
        aggregate = label[j]
        # An unaggregated node (``mis_level_setup``) has no column in ``P0``: its slot emits an
        # exact zero, which the assembly drops.
        contribution = wp.float64(0.0)
        # An off-diagonal entry of a long row or column (``csr_row_long``, a fan's hub) does not
        # smooth the prolongator. A hub's entries are weak (each about ``1 / sqrt(valence)`` of its
        # diagonals' geometric mean, far under ``linalg._MULTIGRID_THETA``); smoothing through
        # them, ``fan_hub``'s 1 %-pinned solve ran 1 000 rounds without converging, and 18 with
        # them left out (13 on ``sphere_med``).
        hub = j != i and (csr_row_long(offsets, i) or csr_row_long(offsets, j))
        if aggregate != MG_UNAGGREGATED and not hub:
            contribution = (
                scale * values[e] * tentative_scale(weights, j) / tentative_norm(sizes, aggregate)
            )
        emit_entry(
            i,
            wp.max(aggregate, wp.int32(0)),
            contribution,
            e,
            n,
            n_aggregates,
            out_keys,
            out_order,
            out_values,
        )
    if not with_identity:
        return
    own = label[i]
    identity = wp.float64(0.0)
    if own != MG_UNAGGREGATED:
        identity = tentative_scale(weights, i) / tentative_norm(sizes, own)
    emit_entry(
        i,
        wp.max(own, wp.int32(0)),
        identity,
        offsets[n] + i,
        n,
        n_aggregates,
        out_keys,
        out_order,
        out_values,
    )


@wp.kernel
def smoothed_prolongator_keys(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    label: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    weights: wp.array[wp.int32],
    damped_inverse_diagonal: wp.array[wp.float64],
    n_aggregates: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # The smoothed prolongator ``P = (I - w D^-1 A) P0`` as keyed contributions, one thread a row.
    # ``P0`` holds one entry a row -- node ``j``'s aggregate, ``tentative_scale(j) /
    # tentative_norm(aggregate)``, the near-nullspace vector normalized per column -- so
    # ``(A P0)_ik`` is the sum of ``A_ij P0_j`` over the row's columns ``j`` in aggregate ``k``:
    # every stored ``A_ij`` is one contribution at ``(i, label[j])``, at the slot ``A`` stores it
    # in, and the identity's ``P0_i`` one more, after all of ``A``'s. The key sum that follows is
    # the whole product, where ``bsr_mm`` against ``P0`` then a row scale and a ``bsr_axpy``
    # against ``P0`` was three sparse passes, each allocating its result.
    # Launch tiled over ``MG_ROWS_PER_BLOCK`` rows a block.
    blk, lane = wp.tid()
    n = offsets.shape[0] - 1
    first, rows = block_chunk(n, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if csr_row_long(offsets, i):
            found = 1
            continue
        smoothed_prolongator_row(
            i, offsets[i], 1, True, offsets, columns, values, label, sizes, weights,
            damped_inverse_diagonal, n_aggregates, out_keys, out_order, out_values,
        )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(offsets, i):
            smoothed_prolongator_row(
                i, offsets[i] + lane, wp.block_dim(), lane == 0, offsets, columns, values, label,
                sizes, weights, damped_inverse_diagonal, n_aggregates, out_keys, out_order,
                out_values,
            )  # fmt: skip


@wp.func
def product_row_count(
    i: wp.int32,
    start: wp.int32,
    step: wp.int32,
    a_offsets: wp.array[wp.int32],
    a_columns: wp.array[wp.int32],
    p_offsets: wp.array[wp.int32],
) -> wp.int64:
    # ``product_triplet_counts``' count over row ``i``'s entries ``start``, ``start + step``, ...
    count = wp.int64(0)
    for e in range(start, a_offsets[i + 1], step):
        j = a_columns[e]
        count += wp.int64(p_offsets[j + 1] - p_offsets[j])
    return count


@wp.kernel
def product_triplet_counts(
    a_offsets: wp.array[wp.int32],
    a_columns: wp.array[wp.int32],
    p_offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int64],
) -> None:
    # Row ``i`` of ``A P`` has one product ``A_ij P_jl`` per stored ``A_ij`` per entry of ``P``'s
    # row ``j``. Counted in 64 bits, as ``galerkin_triplet_counts`` is: on a level that coarsened
    # badly the total passes ``2^31`` and an ``int32`` scan wraps, sometimes to a small positive
    # total that a budget test cannot tell from a real one. Launch tiled over
    # ``MG_ROWS_PER_BLOCK`` rows a block.
    blk, lane = wp.tid()
    first, rows = block_chunk(a_offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if i == 0:
            out_counts[0] = wp.int64(0)
        if csr_row_long(a_offsets, i):
            found = 1
            continue
        out_counts[i + 1] = product_row_count(i, a_offsets[i], 1, a_offsets, a_columns, p_offsets)
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(a_offsets, i):
            count = block_sum(
                product_row_count(
                    i, a_offsets[i] + lane, wp.block_dim(), a_offsets, a_columns, p_offsets
                )
            )
            if lane == 0:
                out_counts[i + 1] = count


@wp.kernel
def product_keys(
    a_offsets: wp.array[wp.int32],
    a_columns: wp.array[wp.int32],
    a_values: wp.array[wp.float64],
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    starts: wp.array[wp.int64],
    n_rows: wp.int32,
    n_cols: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # ``A P`` as keyed contributions, one thread a row, at ``starts[i]``: the key sum that follows
    # replaces ``bsr_mm`` (1.7x on the product at the bunny Laplacians, 1.04x at dragon's 9 M
    # contributions) and leaves no structural zeros behind. Launch tiled over
    # ``MG_ROWS_PER_BLOCK`` rows a block; a long row's block places each entry's run by an
    # exclusive scan of the runs' lengths, a chunk of ``block_dim`` entries at a time.
    blk, lane = wp.tid()
    first, rows = block_chunk(a_offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if csr_row_long(a_offsets, i):
            found = 1
            continue
        slot = wp.int32(starts[i])  # the wrapper capped the total below ``2^30``
        for e in range(a_offsets[i], a_offsets[i + 1]):
            j = a_columns[e]
            slot = emit_scaled_row(
                i, a_values[e], p_offsets[j], p_offsets[j + 1], p_columns, p_values, slot,
                n_rows, n_cols, out_keys, out_order, out_values,
            )  # fmt: skip
    if block_sum(found) == 0:
        return
    lanes = wp.block_dim()
    for r in range(rows):
        i = first + r
        if not csr_row_long(a_offsets, i):
            continue
        slot = wp.int32(starts[i])
        end = a_offsets[i + 1]
        for chunk in range(a_offsets[i], end, lanes):
            e = chunk + lane
            run = wp.int32(0)
            j = wp.int32(0)
            if e < end:
                j = a_columns[e]
                run = p_offsets[j + 1] - p_offsets[j]
            before = wp.untile(wp.tile_scan_exclusive(wp.tile(run)))
            if e < end:
                emit_scaled_row(
                    i, a_values[e], p_offsets[j], p_offsets[j + 1], p_columns, p_values,
                    slot + before, n_rows, n_cols, out_keys, out_order, out_values,
                )  # fmt: skip
            slot += block_sum(run)


@wp.kernel
def galerkin_triplet_counts(
    p_offsets: wp.array[wp.int32], ap_offsets: wp.array[wp.int32], out_counts: wp.array[wp.int64]
) -> None:
    # Row ``i`` of ``A P`` meets row ``i`` of ``P`` in every product ``P_ik (A P)_il``. Written at
    # ``i + 1`` of an ``n + 1`` buffer, so the inclusive scan of it is the starts and the total.
    # Launch tiled over ``MG_ROWS_PER_BLOCK`` rows a block, as ``galerkin_keys`` is.
    blk, lane = wp.tid()
    first, rows = block_chunk(p_offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        out_counts[i + 1] = wp.int64(p_offsets[i + 1] - p_offsets[i]) * wp.int64(
            ap_offsets[i + 1] - ap_offsets[i]
        )
        if i == 0:
            out_counts[0] = wp.int64(0)


@wp.kernel
def galerkin_keys(
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    ap_offsets: wp.array[wp.int32],
    ap_columns: wp.array[wp.int32],
    ap_values: wp.array[wp.float64],
    starts: wp.array[wp.int64],
    n_rows: wp.int32,
    n_cols: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # ``P^T (A P)`` as keyed contributions, one thread a fine row: ``(P^T A P)_kl`` is the sum over
    # fine rows ``i`` of ``P_ik (A P)_il``, so row ``i`` emits the outer product of its two sparse
    # rows at ``starts[i]``. No transpose is formed and no second product runs. Launch tiled over
    # ``MG_ROWS_PER_BLOCK`` rows a block; a row whose ``A P`` row is long (a hub's) is left to the
    # whole block, its lanes striding that row (every slot is positional).
    blk, lane = wp.tid()
    first, rows = block_chunk(p_offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if csr_row_long(ap_offsets, i):
            found = 1
            continue
        slot = wp.int32(starts[i])  # the wrapper capped the total below ``2^30``
        for a in range(p_offsets[i], p_offsets[i + 1]):
            slot = emit_scaled_row(
                p_columns[a], p_values[a], ap_offsets[i], ap_offsets[i + 1], ap_columns,
                ap_values, slot, n_rows, n_cols, out_keys, out_order, out_values,
            )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if not csr_row_long(ap_offsets, i):
            continue
        width = ap_offsets[i + 1] - ap_offsets[i]
        base = wp.int32(starts[i])
        for a in range(p_offsets[i], p_offsets[i + 1]):
            row = p_columns[a]
            weight = p_values[a]
            for b in range(lane, width, wp.block_dim()):
                k = ap_offsets[i] + b
                emit_entry(
                    row, ap_columns[k], weight * ap_values[k], base + b, n_rows, n_cols,
                    out_keys, out_order, out_values,
                )  # fmt: skip
            base += width


@wp.func
def galerkin_dense_row(
    i: wp.int32,
    start: wp.int32,
    step: wp.int32,
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    ap_offsets: wp.array[wp.int32],
    ap_columns: wp.array[wp.int32],
    ap_values: wp.array[wp.float64],
    n_coarse: wp.int32,
    out_dense: wp.array[wp.float64],
) -> None:
    # ``galerkin_dense``' products of fine row ``i``, its ``A P`` entries ``start``, ``start +
    # step``, ... against every entry of its ``P`` row.
    for a in range(p_offsets[i], p_offsets[i + 1]):
        row = p_columns[a] * n_coarse
        weight = p_values[a]
        for b in range(start, ap_offsets[i + 1], step):
            wp.atomic_add(out_dense, row + ap_columns[b], weight * ap_values[b])


@wp.kernel
def galerkin_dense(
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    ap_offsets: wp.array[wp.int32],
    ap_columns: wp.array[wp.int32],
    ap_values: wp.array[wp.float64],
    n_coarse: wp.int32,
    out_dense: wp.array[wp.float64],
) -> None:
    # ``P^T (A P)`` added straight into the zeroed row-major ``(n_coarse, n_coarse)`` matrix, one
    # thread a fine row: ``galerkin_keys``' outer product of its two sparse rows, summed by atomics
    # rather than keyed and sorted, for the coarsest level, which only the dense inverse reads
    # (``linalg._multigrid_galerkin``). Arrival order decides the sums' rounding, so it serves
    # the deterministic setting's off mode only. Launch tiled over ``MG_ROWS_PER_BLOCK`` rows a
    # block; a row whose ``A P`` row is long is the whole block's, its lanes striding that row.
    blk, lane = wp.tid()
    first, rows = block_chunk(p_offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if csr_row_long(ap_offsets, i):
            found = 1
            continue
        galerkin_dense_row(
            i, ap_offsets[i], 1, p_offsets, p_columns, p_values, ap_offsets, ap_columns,
            ap_values, n_coarse, out_dense,
        )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(ap_offsets, i):
            galerkin_dense_row(
                i, ap_offsets[i] + lane, wp.block_dim(), p_offsets, p_columns, p_values,
                ap_offsets, ap_columns, ap_values, n_coarse, out_dense,
            )  # fmt: skip


@wp.kernel
def csr_matvec(
    n_rows: wp.int32,
    x_stride: wp.int32,
    y_stride: wp.int32,
    accumulate: wp.int32,
    alpha: wp.float64,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
) -> None:
    # ``y = alpha * A x`` (or ``y += alpha * A x``) for **every column at once**, which is the whole
    # reason this exists rather than ``warp.sparse.bsr_mv``: that takes one vector, so a cycle over
    # three right-hand sides pays three launches per mat-vec, and the V-cycle is launch-bound at
    # these sizes -- measured at less than half the launches and less than half the time per cycle,
    # with the same answer.
    #
    # ``x_stride`` and ``y_stride`` differ whenever the operator is rectangular -- the restriction
    # reads at the fine pitch and writes at the coarse one -- and either may exceed its own row
    # count, since the top level's vectors carry the conjugate-gradient state's tile padding.
    # ``accumulate`` is warp-uniform and folds the prolongation's correction into the same kernel.
    # ``alpha`` mirrors ``bsr_mv``'s own scale argument -- every caller here passes a compile-time
    # constant (``1.0`` or ``-1.0``), so it costs one multiply per row rather than a second kernel.
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    total = alpha * csr_row_dot(row, column * x_stride, offsets, columns, values, x)
    slot = column * y_stride + row
    if accumulate != wp.int32(0):
        out_y[slot] += total
    else:
        out_y[slot] = total


@wp.func
def chebyshev_update(
    coefficients: Any, x: wp.Float, x_previous: wp.Float, source: wp.Float, ax: wp.Float
) -> Any:
    # One entry of a Chebyshev semi-iteration step in its three-term form,
    # ``x' = s x + momentum (s x - s_prev x_previous) + step (source - s A x)``, with
    # ``coefficients = (s, s_prev, momentum, step)`` and ``ax`` the entry's ``(A x)``. Shared by
    # ``chebyshev_step`` and ``conjugate_gradient.one_block_chebyshev``, which differ only in where
    # the vectors live and in the precision (the latter's coefficients are narrowed).
    current = coefficients[0] * x
    return (
        current
        + coefficients[2] * (current - coefficients[1] * x_previous)
        + coefficients[3] * (source - coefficients[0] * ax)
    )


@wp.func
def chebyshev_scaled_update(
    row: wp.int32,
    slot: wp.int32,
    step: wp.vec4d,
    ax: wp.float64,
    row_scaled: wp.int32,
    row_scale: wp.array[wp.float64],
    source: wp.array[wp.float64],
    x: wp.array[wp.float64],
    x_previous: wp.array[wp.float64],
) -> wp.float64:
    # What a Chebyshev step does with row ``row``'s product ``ax`` at ``slot``: scale it by
    # ``row_scale`` under a nonzero ``row_scaled``, then the three-term update. Shared by
    # ``chebyshev_row`` (one column a thread) and ``chebyshev_step_rows`` (every column a thread),
    # which differ only in how many columns one walk of the row feeds.
    scaled = ax
    if row_scaled != wp.int32(0):
        scaled = ax * row_scale[row]
    return chebyshev_update(step, x[slot], x_previous[slot], source[slot], scaled)


@wp.func
def chebyshev_row(
    row: wp.int32,
    base: wp.int32,
    step: wp.vec4d,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    row_scaled: wp.int32,
    row_scale: wp.array[wp.float64],
    source: wp.array[wp.float64],
    x: wp.array[wp.float64],
    x_previous: wp.array[wp.float64],
) -> wp.float64:
    # One row's entry of a Chebyshev step over the column at ``base``: the row dot of ``x``,
    # scaled by ``row_scale`` under a nonzero ``row_scaled``, then the three-term update.
    slot = base + row
    return chebyshev_scaled_update(
        row, slot, step, csr_row_dot(row, base, offsets, columns, values, x), row_scaled,
        row_scale, source, x, x_previous,
    )  # fmt: skip


@wp.kernel
def chebyshev_step(
    n_rows: wp.int32,
    stride: wp.int32,
    skip_long: wp.int32,
    steps: wp.array[wp.vec4d],
    index: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    row_scaled: wp.int32,
    row_scale: wp.array[wp.float64],
    source: wp.array[wp.float64],
    x: wp.array[wp.float64],
    x_previous: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # A nonzero ``row_scaled`` (warp uniform) iterates on ``diag(row_scale) A`` rather than on the
    # matrix as stored, scaling each row's product after the dot: the Jacobi-scaled operator the
    # polynomial preconditioner in ``linalg`` wants, with no scaled copy of ``A`` to build. Off,
    # ``row_scale`` is never read and may be ``None``.
    #
    # One step of the Chebyshev semi-iteration for ``A x = source``, every column at once, in its
    # three-term form: ``x' = s x + momentum (s x - s_prev x_previous) + step (source - s A x)``,
    # with ``(s, s_prev, momentum, step)`` read from ``steps[index]``, which
    # ``linalg.chebyshev_steps`` writes on the device.
    # With both scales ``1`` that is the textbook recurrence. The semi-iteration's first iterate is
    # ``source / theta``, which no launch writes: the first step passes ``x = source`` with
    # ``s = 1 / theta`` (and ``s_prev = 0``, the zero iterate before it), and the second passes
    # ``x_previous = source`` with ``s_prev = 1 / theta``. Reading ``source`` there unscaled is
    # a different polynomial -- still a polynomial, but one that can change sign inside the
    # interval once ``theta`` is not close to 1. Written in ``x`` rather than in the correction so
    # that the one mat-vec and both updates are a single launch, which is what a step costs at these
    # sizes. ``out_x`` must not alias ``x`` (the row dot reads it at other rows); it may alias
    # ``x_previous``, which each row reads at its own slot before writing it. Shares
    # ``csr_row_dot`` with ``csr_matvec``, and its row with ``chebyshev_polynomial_serial``.
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    if skip_long != 0 and csr_row_long(offsets, row):
        # Stepped by ``chebyshev_long_rows``.
        return
    base = column * stride
    out_x[base + row] = chebyshev_row(
        row, base, steps[index], offsets, columns, values, row_scaled, row_scale, source, x,
        x_previous,
    )  # fmt: skip


# Launched at its own width (``linalg._LONG_ROW_LANES``), so in a module of its own (§12.4).
@wp.kernel(module="unique")
def chebyshev_long_rows(
    stride: wp.int32,
    launch_blocks: wp.int32,
    pieces: wp.array[wp.vec3i],
    count: wp.array[wp.int32],
    steps: wp.array[wp.vec4d],
    index: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    row_scaled: wp.int32,
    row_scale: wp.array[wp.float64],
    source: wp.array[wp.float64],
    x: wp.array[wp.float64],
    x_previous: wp.array[wp.float64],
    partials: wp.array[wp.float64],
    tickets: wp.array[wp.int32],
    out_x: wp.array[wp.float64],
) -> None:
    # ``chebyshev_step`` at the long rows ``linalg.list_long_rows`` listed, which the thread-a-row
    # steps skip under ``skip_long``: a block a piece and column sums the row
    # (``long_row_piece_total``), and the block folding it updates. Launch tiled over
    # ``(n_columns, launch_blocks)``; the blocks stride the listed pieces, so a launch with none
    # listed returns at once (block uniform). Same aliasing rules.
    column, blk, lane = wp.tid()
    step = steps[index]
    base = column * stride
    slots = column * pieces.shape[0]
    for b in range(blk, listed_long_pieces(pieces, count), launch_blocks):
        piece = pieces[b]
        last, ax = long_row_piece_total(
            piece, base, slots, lane, offsets, columns, values, x, partials, tickets
        )
        if last != 0:
            row = piece[0]
            slot = base + row
            out_x[slot] = chebyshev_scaled_update(
                row, slot, step, ax, row_scaled, row_scale, source, x, x_previous
            )


# Row count from which a Chebyshev step over several columns runs one thread a row
# (``chebyshev_step_rows``) rather than one a row and column (``chebyshev_step``). The thread-a-slot
# form walks each row once per column; at a few columns and enough rows the extra walks of
# ``offsets`` / ``columns`` / ``values`` are the step. CUDA, per step, rows/slots at three
# (two) columns: 1.38-1.47x (1.08x) at 4-9 k rows, 1.13x (1.06x) at 22 k, 0.82-1.09x (0.90-0.98x)
# at 36-43 k, then 0.48-0.69x (0.60-0.67x) from 109 k to 5.6 M, bit-identical everywhere.
# The CPU device runs ``chebyshev_polynomial_serial`` instead.
CHEBYSHEV_STEP_ROWS_FROM = 1 << 16

# The column counts with a ``chebyshev_step_rows`` kernel: the solvers' coordinate columns (three)
# and the parametrizations' UV pair. Any other count keeps ``chebyshev_step``.
CHEBYSHEV_STEP_ROWS_COLUMNS = (2, 3)


def _csr_row_dots(n_columns: int, value_dtype: Any = wp.float64) -> wp.Function:
    """``csr_row_dot`` against ``n_columns`` columns of ``x`` at once, off one walk of the row."""
    column_values = wp.types.vector(length=n_columns, dtype=wp.float64)

    def _func(
        row: wp.int32,
        stride: wp.int32,
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.float64],
        x: wp.array[wp.float64],
    ) -> column_values:
        # The row walk every thread-a-row kernel here shares: each column's sum, into registers (a
        # constant ``n_columns``, so the column loop unrolls and the vector is never indexed at run
        # time), adds the row's entries in ``csr_row_dot``'s order, so a thread-a-row kernel's
        # result is its thread-a-slot sibling's to the bit.
        # ``float32`` values are widened as they are read (``csr_row_dot``'s convention); at
        # ``float64`` the cast is the identity.
        total = column_values()
        for k in range(offsets[row], offsets[row + 1]):
            value = wp.float64(values[k])
            entry = columns[k]
            for column in range(n_columns):
                total[column] += value * x[column * stride + entry]
        return total

    _func.__annotations__["values"] = wp.array[value_dtype]
    suffix = "" if value_dtype == wp.float64 else "_f32"
    return wp.func(_func, name=f"csr_row_dots{n_columns}{suffix}")


# The column counts with thread-a-row kernels (``csr_row_dots``): one column (the scalar solves,
# where only the cycle's fused sweep differs from the thread-a-slot form), the parametrizations' UV
# pair and the solvers' coordinate columns.
ROWS_COLUMNS = (1, 2, 3)

CSR_ROW_DOTS = {n_columns: _csr_row_dots(n_columns) for n_columns in ROWS_COLUMNS}
CSR_ROW_DOTS_NARROW = {
    n_columns: _csr_row_dots(n_columns, wp.float32) for n_columns in ROWS_COLUMNS
}


def _chebyshev_step_rows(n_columns: int) -> wp.Kernel:
    """``chebyshev_step`` over ``n_columns`` columns, one thread a row walking it once."""
    row_dots = CSR_ROW_DOTS[n_columns]

    def _kernel(
        stride: wp.int32,
        skip_long: wp.int32,
        steps: wp.array[wp.vec4d],
        index: wp.int32,
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.float64],
        row_scaled: wp.int32,
        row_scale: wp.array[wp.float64],
        source: wp.array[wp.float64],
        x: wp.array[wp.float64],
        x_previous: wp.array[wp.float64],
        out_x: wp.array[wp.float64],
    ) -> None:
        # ``chebyshev_step``'s contract and aliasing rules, with every column's product of the row
        # summed off one walk of it (``csr_row_dots``), so the result is ``chebyshev_step``'s to
        # the bit.
        row = wp.int32(wp.tid())
        if skip_long != 0 and csr_row_long(offsets, row):
            return
        ax = row_dots(row, stride, offsets, columns, values, x)
        step = steps[index]
        for column in range(n_columns):
            slot = column * stride + row
            out_x[slot] = chebyshev_scaled_update(
                row, slot, step, ax[column], row_scaled, row_scale, source, x, x_previous
            )

    return wp.kernel(_kernel, name=f"chebyshev_step_rows{n_columns}")


CHEBYSHEV_STEP_ROWS = {
    n_columns: _chebyshev_step_rows(n_columns) for n_columns in CHEBYSHEV_STEP_ROWS_COLUMNS
}


@wp.func
def jacobi_update(
    x: wp.float64, scale: wp.float64, rhs: wp.float64, operator_x: wp.float64
) -> wp.float64:
    # One entry of a damped-Jacobi sweep, ``x + scale (b - A x)`` with ``scale = w / A_ii``.
    # Shared by ``jacobi_sweep`` (``A x`` read from a buffer a mat-vec wrote) and
    # ``cycle_sweep_rows`` (``A x`` summed in the same thread), which differ only in that.
    return x + scale * (rhs - operator_x)


# Row count from which a multigrid cycle level on CUDA runs its passes one thread a row
# (``CYCLE_ROWS``) rather than one a row and column (``CYCLE_SLOTS``). Either way a smoothing sweep
# and the residual take their mat-vec in the same thread, so ``A x`` is never stored; a thread a
# row also walks each row once for all its columns. Bit-identical to the slot forms.
# CUDA, ``smooth_region``'s squared V-cycle (three columns), rows/slots: 0.78x a call on dragon's
# 109 k free vertices, 0.51-0.53x on lucy's 3.5 M; ``multigrid_preconditioner`` solves on
# ``icosphere(8 / 9)`` 0.90-0.93x at one column, 0.61-0.80x at three. The gate, a call at 2**14
# against 2**16: 0.96-0.98x at 21-43 k free vertices, level from 109 k; at zero, 1.03-1.10x (a
# thread a row starves the small levels).
CYCLE_ROWS_FROM = 1 << 14


def _cycle_rows(
    n_columns: int, value_dtype: Any = wp.float64
) -> tuple[wp.Kernel, wp.Kernel, wp.Kernel]:
    """Return a cycle level's sweep, residual and transfer mat-vec over ``n_columns`` columns."""
    narrow = value_dtype != wp.float64
    row_dots = (CSR_ROW_DOTS_NARROW if narrow else CSR_ROW_DOTS)[n_columns]
    suffix = "_f32" if narrow else ""

    def _sweep(
        stride: wp.int32,
        skip_long: wp.int32,
        inv_diag: wp.array[wp.float64],
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.float64],
        rhs: wp.array[wp.float64],
        x: wp.array[wp.float64],
        out_x: wp.array[wp.float64],
    ) -> None:
        # ``csr_matvec`` then ``jacobi_sweep`` (the damping folded into ``inv_diag``) in one pass,
        # to the bit. ``out_x`` must not alias ``x``: the row dot reads ``x`` at other rows, so the
        # cycle ping-pongs a level's iterate between two buffers. Under ``skip_long`` a long row
        # (``csr_row_long``) is ``cycle_long_rows``'.
        row = wp.int32(wp.tid())
        if skip_long != 0 and csr_row_long(offsets, row):
            return
        ax = row_dots(row, stride, offsets, columns, values, x)
        scale = inv_diag[row]
        for column in range(n_columns):
            slot = column * stride + row
            out_x[slot] = jacobi_update(x[slot], scale, rhs[slot], ax[column])

    def _residual(
        stride: wp.int32,
        skip_long: wp.int32,
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.float64],
        rhs: wp.array[wp.float64],
        x: wp.array[wp.float64],
        out_residual: wp.array[wp.float64],
    ) -> None:
        # ``csr_matvec`` then ``residual`` in one pass, to the bit; a long row under ``skip_long``
        # is ``cycle_long_rows``'.
        row = wp.int32(wp.tid())
        if skip_long != 0 and csr_row_long(offsets, row):
            return
        ax = row_dots(row, stride, offsets, columns, values, x)
        for column in range(n_columns):
            slot = column * stride + row
            out_residual[slot] = rhs[slot] - ax[column]

    def _matvec(
        x_stride: wp.int32,
        y_stride: wp.int32,
        accumulate: wp.int32,
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.float64],
        x: wp.array[wp.float64],
        addend: wp.array[wp.float64],
        out_y: wp.array[wp.float64],
    ) -> None:
        # ``csr_matvec`` at ``alpha = 1``, a row each: the restriction (``accumulate`` zero) and the
        # prolongation, which under a nonzero ``accumulate`` (warp uniform) writes ``addend + P e``.
        # ``addend`` may alias ``out_y`` (each slot is read before it is written), which is
        # ``csr_matvec``'s in-place accumulation; a distinct one lets the cycle land the corrected
        # iterate in the other buffer of the level's pair.
        row = wp.int32(wp.tid())
        product = row_dots(row, x_stride, offsets, columns, values, x)
        for column in range(n_columns):
            slot = column * y_stride + row
            if accumulate != wp.int32(0):
                out_y[slot] = addend[slot] + product[column]
            else:
                out_y[slot] = product[column]

    for kernel in (_sweep, _residual, _matvec):
        kernel.__annotations__["values"] = wp.array[value_dtype]
    return (
        wp.kernel(_sweep, name=f"cycle_sweep_rows{n_columns}{suffix}"),
        wp.kernel(_residual, name=f"cycle_residual_rows{n_columns}{suffix}"),
        wp.kernel(_matvec, name=f"cycle_matvec_rows{n_columns}{suffix}"),
    )


CYCLE_ROWS = {n_columns: _cycle_rows(n_columns) for n_columns in ROWS_COLUMNS}
# The same kernels over ``float32`` values: a level whose operators the cycle keeps a narrow copy
# of (``linalg._MultigridCycle``).
CYCLE_ROWS_NARROW = {n_columns: _cycle_rows(n_columns, wp.float32) for n_columns in ROWS_COLUMNS}


# Launched at its own width (``linalg._LONG_ROW_LANES``), so in a module of its own (§12.4).
@wp.kernel(module="unique")
def cycle_long_rows(
    stride: wp.int32,
    launch_blocks: wp.int32,
    residual: wp.int32,
    pieces: wp.array[wp.vec3i],
    count: wp.array[wp.int32],
    inv_diag: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    rhs: wp.array[wp.float64],
    x: wp.array[wp.float64],
    partials: wp.array[wp.float64],
    tickets: wp.array[wp.int32],
    out_x: wp.array[wp.float64],
) -> None:
    # A fused cycle sweep (``residual`` zero) or residual at the long rows ``linalg.list_long_rows``
    # listed, which the fused passes skip under ``skip_long``: a block a piece and column sums the
    # row (``linalg.long_row_piece_total``) and the block folding it writes the slot, the sweep's
    # ``jacobi_update`` or ``rhs - A x``. Launch tiled over ``(n_columns, launch_blocks)``; the
    # blocks stride the listed pieces. Same aliasing rules as the fused passes.
    column, blk, lane = wp.tid()
    base = column * stride
    slots = column * pieces.shape[0]
    for b in range(blk, listed_long_pieces(pieces, count), launch_blocks):
        piece = pieces[b]
        last, ax = long_row_piece_total(
            piece, base, slots, lane, offsets, columns, values, x, partials, tickets
        )
        if last != 0:
            slot = base + piece[0]
            if residual != 0:
                out_x[slot] = rhs[slot] - ax
            else:
                out_x[slot] = jacobi_update(x[slot], inv_diag[piece[0]], rhs[slot], ax)


# A cycle level below ``CYCLE_ROWS_FROM`` rows on CUDA runs ``CYCLE_ROWS``' fused passes one thread
# a row and column instead (a thread a row would leave such a level too few threads), still with
# the mat-vec in the sweep's and the residual's thread, so a sweep is one launch, not two. Same
# contracts, aliasing rules and arithmetic as the thread-a-row kernels, to the bit.


@wp.kernel
def cycle_sweep_slots(
    n_rows: wp.int32,
    stride: wp.int32,
    skip_long: wp.int32,
    inv_diag: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    if skip_long != 0 and csr_row_long(offsets, row):
        return
    base = column * stride
    slot = base + row
    ax = csr_row_dot(row, base, offsets, columns, values, x)
    out_x[slot] = jacobi_update(x[slot], inv_diag[row], rhs[slot], ax)


@wp.kernel
def cycle_residual_slots(
    n_rows: wp.int32,
    stride: wp.int32,
    skip_long: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_residual: wp.array[wp.float64],
) -> None:
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    if skip_long != 0 and csr_row_long(offsets, row):
        return
    base = column * stride
    slot = base + row
    out_residual[slot] = rhs[slot] - csr_row_dot(row, base, offsets, columns, values, x)


@wp.kernel
def cycle_matvec_slots(
    n_rows: wp.int32,
    x_stride: wp.int32,
    y_stride: wp.int32,
    accumulate: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    addend: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
) -> None:
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    product = csr_row_dot(row, column * x_stride, offsets, columns, values, x)
    slot = column * y_stride + row
    if accumulate != wp.int32(0):
        out_y[slot] = addend[slot] + product
    else:
        out_y[slot] = product


CYCLE_SLOTS = (cycle_sweep_slots, cycle_residual_slots, cycle_matvec_slots)


@wp.func
def chebyshev_polynomial_column(
    column: wp.int32,
    n_rows: wp.int32,
    stride: wp.int32,
    steps: wp.array[wp.vec4d],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    row_scaled: wp.int32,
    row_scale: wp.array[wp.float64],
    source: wp.array[wp.float64],
    iterate_0: wp.array[wp.float64],
    iterate_1: wp.array[wp.float64],
    iterate_2: wp.array[wp.float64],
    x: wp.array[wp.float64],
) -> None:
    # Column ``column``'s every ``chebyshev_step`` of one polynomial, walking each step's rows in
    # turn: ``chebyshev_polynomial_serial``'s thread, and the polynomial the CPU device's one-launch
    # conjugate-gradient solve (``conjugate_gradient.cg_solve_serial``) applies every round.
    base = column * stride
    n_steps = steps.shape[0]
    for index in range(n_steps):
        step = steps[index]
        target = 2 - index % 3
        if index == n_steps - 1:
            target = -1
        phase = (index + 1) % 3
        for row in range(n_rows):
            value = wp.float64(0.0)
            if index == 0:
                value = chebyshev_row(
                    row, base, step, offsets, columns, values, row_scaled, row_scale, source,
                    source, source,
                )  # fmt: skip
            elif index == 1:
                value = chebyshev_row(
                    row, base, step, offsets, columns, values, row_scaled, row_scale, source,
                    iterate_2, source,
                )  # fmt: skip
            elif phase == 0:
                value = chebyshev_row(
                    row, base, step, offsets, columns, values, row_scaled, row_scale, source,
                    iterate_1, iterate_2,
                )  # fmt: skip
            elif phase == 1:
                value = chebyshev_row(
                    row, base, step, offsets, columns, values, row_scaled, row_scale, source,
                    iterate_0, iterate_1,
                )  # fmt: skip
            else:
                value = chebyshev_row(
                    row, base, step, offsets, columns, values, row_scaled, row_scale, source,
                    iterate_2, iterate_0,
                )  # fmt: skip
            slot = base + row
            if target == 0:
                iterate_0[slot] = value
            elif target == 1:
                iterate_1[slot] = value
            elif target == 2:
                iterate_2[slot] = value
            else:
                x[slot] = value


@wp.kernel
def chebyshev_polynomial_serial(
    n_rows: wp.int32,
    stride: wp.int32,
    steps: wp.array[wp.vec4d],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    row_scaled: wp.int32,
    row_scale: wp.array[wp.float64],
    source: wp.array[wp.float64],
    out_iterate_0: wp.array[wp.float64],
    out_iterate_1: wp.array[wp.float64],
    out_iterate_2: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # Every ``chebyshev_step`` of one polynomial in one launch, one thread a column walking each
    # step's rows in turn: the CPU device's form, where a launch's threads run one after another
    # anyway, so the per-step launches buy no parallelism. The rotation is ``linalg``'s: step
    # ``k`` writes ``out_iterate_{2 - k % 3}`` (the last one ``out_x``), reading the iterates steps
    # ``k - 1`` and ``k - 2`` wrote, ``source`` in place of those that do not exist. Each entry is
    # ``chebyshev_row``, the step kernel's own, so the result is that sequence's to the bit.
    chebyshev_polynomial_column(
        wp.int32(wp.tid()), n_rows, stride, steps, offsets, columns, values, row_scaled,
        row_scale, source, out_iterate_0, out_iterate_1, out_iterate_2, out_x,
    )  # fmt: skip


@wp.kernel
def cycle_chebyshev_update(
    n_rows: wp.int32,
    stride: wp.int32,
    momentum: wp.float64,
    step: wp.float64,
    first: wp.int32,
    cycled: wp.array[wp.float64],
    direction: wp.array[wp.float64],
    iterate: wp.array[wp.float64],
) -> None:
    # One step of the Chebyshev iteration for ``L z = r`` preconditioned by a V-cycle ``V``,
    # every column at once: ``direction = momentum * direction + step * cycled`` with
    # ``cycled = V res``, then ``iterate += direction`` -- or, on the ``first`` step (warp
    # uniform), ``direction = step * cycled`` and ``iterate = direction``, since the iteration
    # starts from zero and no launch clears it. ``linalg._SquaredMultigridApply`` forms the
    # residual between steps with ``csr_matvec``. Both arrays are updated in place.
    t = wp.int32(wp.tid())
    slot = (t // n_rows) * stride + t % n_rows
    if first != wp.int32(0):
        value = step * cycled[slot]
        direction[slot] = value
        iterate[slot] = value
    else:
        value = momentum * direction[slot] + step * cycled[slot]
        direction[slot] = value
        iterate[slot] += value


# Rows a block of ``power_step_partials`` / ``damped_inverse_diagonal`` owns, its ``TILE_1D``
# lanes striding them. Four rows a lane rather than the ``reduce`` fold width
# (``ITEMS_PER_BLOCK_1D``, sixteen): a level of 40 k rows then launches 160 blocks rather than 40
# on a 170-SM part (the narrower block measured level at 40 k rows where sixteen read ~2 % slower).
# Every ``damped_inverse_diagonal`` block folds all the partials, so the wrapper pre-folds them with
# ``reduce.SUM1D_PARTIALS`` past ``4 * ITEMS_PER_BLOCK_1D`` (about 1 M rows): folding 2 560 per
# block at 655 k rows was 1.03x the old inner-product pass, one pre-fold launch 1.02x.
POWER_ROWS_PER_BLOCK = 4 * TILE_1D


@wp.func
def power_block_rows(n: wp.int32, block: wp.int32) -> tuple[wp.int32, wp.int32]:
    # ``(offset, count)`` of the ``POWER_ROWS_PER_BLOCK`` rows block ``block`` owns, clamped to
    # its own share (``tile_chunk``'s ``remaining`` runs to the end of the array).
    return block_chunk(n, block, POWER_ROWS_PER_BLOCK)


@wp.kernel
def damped_inverse_diagonal(
    growth_partials: wp.array[wp.float64],
    start: wp.float64,
    exponent: wp.float64,
    factor: wp.float64,
    diagonal: wp.array[wp.float64],
    out_scaled: wp.array[wp.float64],
) -> None:
    # ``omega D^-1`` for a level's smoother and prolongator, with ``omega = factor / rho`` formed
    # on the device from the power iteration's growth rather than read back to the host. ``D^-1``
    # is ``array.inverse_or_one`` of the operator's diagonal, applied as it is read, as
    # ``power_step`` does.
    #
    # The growth arrives as ``power_step_partials``' per-block squared norms, and every block
    # folds all of them itself (one ``block_sum`` over lanes striding the partials) before writing
    # its ``power_block_rows``: the same fixed tree in every block, so the total is identical
    # across blocks and runs, and no separate inner-product pass or ``dim=1`` launch runs. The
    # wrapper first folds the partials with ``reduce.SUM1D_PARTIALS`` while there are more than
    # ``4 * ITEMS_PER_BLOCK_1D``, so each block reads at most that many (``POWER_ROWS_PER_BLOCK``).
    # ``rho = sqrt(growth / start) ** exponent``, ``exponent`` being one over the step count, and
    # ``rho = 1`` when the growth is not a positive finite number (an operator the iteration cannot
    # measure), which is the host form's guard verbatim.
    #
    # Pre-scaling changes no consumer's arithmetic: ``jacobi_sweep`` and ``scaled_diagonal_apply``
    # form ``factor * inv[row] * r`` left to right, so ``1 * (omega * inv[row]) * r`` is the same
    # product, and the prolongator's ``(-omega) * inv[row]`` is ``(-1) * (omega * inv[row])``.
    block, lane = wp.tid()
    offset, count = power_block_rows(out_scaled.shape[0], block)
    if count <= 0:
        return
    partial = wp.float64(0.0)
    for k in range(lane, growth_partials.shape[0], wp.block_dim()):
        partial += growth_partials[k]
    end = block_sum(partial)
    rho = wp.float64(1.0)
    if start > wp.float64(0.0) and end > wp.float64(0.0) and wp.isfinite(end):
        rho = wp.pow(wp.sqrt(end / start), exponent)
    omega = factor / rho
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        out_scaled[i] = omega * inverse_or_one(diagonal[i])


@wp.func
def random_sign(seed: wp.int32, index: wp.int32) -> wp.float64:
    # Entry ``index`` of the power iteration's start vector.
    #
    # Random rather than constant, because on a Laplacian-like operator the dominant eigenvector is
    # the highest-frequency mode and a constant vector is nearly orthogonal to it. And *signs*
    # rather than uniform values, because then the squared norm is exactly ``n`` and the growth
    # needs no measurement of the start.
    #
    # Deliberately *not* ``array.element_priority``, which has the same shape over ``wp.randu``:
    # that one draws a total order on the elements, this one draws a start vector whose norm is
    # known in closed form. Same tokens, different quantities.
    return wp.where(
        wp.randi(wp.rand_init(seed, index)) < wp.int32(0), wp.float64(-1.0), wp.float64(1.0)
    )


@wp.func
def power_row(
    seed: wp.int32,
    from_signs: wp.int32,
    diagonal: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    i: wp.int32,
) -> wp.float64:
    # Row ``i`` of ``y = D^-1 A x``, one whole step of the power iteration that estimates the
    # spectral radius. It differs from ``csr_matvec`` above only in folding the diagonal scale in
    # and in being single-column -- and that fusion is the point, because the setup is
    # launch-bound: as a ``bsr_mv`` plus a ``scaled_diagonal_apply`` it is two launches, and an
    # uncaptured ``bsr_mv`` costs a fixed host price whatever its nnz.
    #
    # ``D^-1`` is ``array.inverse_or_one`` of the operator's diagonal, applied as it is read, so no
    # pass writes the inverse first. The first step (warp-uniform ``from_signs != 0``) draws its
    # ``x`` from ``random_sign`` as it reads it and does not read ``x``; its row dot is
    # ``csr_row_dot``'s, term for term.
    total = wp.float64(0.0)
    if from_signs != 0:
        for k in range(offsets[i], offsets[i + 1]):
            total += values[k] * random_sign(seed, columns[k])
    else:
        total = csr_row_dot(i, wp.int32(0), offsets, columns, values, x)
    return inverse_or_one(diagonal[i]) * total


@wp.func
def power_row_block(
    seed: wp.int32,
    from_signs: wp.int32,
    diagonal: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    i: wp.int32,
    lane: wp.int32,
) -> wp.float64:
    # ``power_row`` by a whole block, its lanes striding the row and the block summing it in a
    # fixed tree (every lane must call it; every lane gets the row).
    total = wp.float64(0.0)
    for k in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        if from_signs != 0:
            total += values[k] * random_sign(seed, columns[k])
        else:
            total += values[k] * x[columns[k]]
    return inverse_or_one(diagonal[i]) * block_sum(total)


@wp.func
def mis_row_setup(
    i: wp.int32,
    seed: wp.int32,
    l1_smoother: wp.int32,
    diagonal: wp.float64,
    off_sum: wp.float64,
    out_diagonal: wp.array[wp.float64],
    out_scaled_diagonal: wp.array[wp.float64],
    out_priority: wp.array[wp.uint32],
    out_state: wp.array[wp.int32],
    out_root_flags: wp.array[wp.int32],
) -> None:
    # ``mis_level_setup``'s per-row writes from row ``i``'s diagonal and off-diagonal magnitude
    # sum, all but the power iteration's first step.
    #
    # The l1 diagonal keeps the true one's sign, so a negative-definite operator (a cotangent
    # Laplacian as ``cotmatrix`` returns it) smooths toward its solution rather than away from it.
    l1 = wp.abs(diagonal) + off_sum
    out_diagonal[i] = wp.where(
        l1_smoother != 0, wp.where(diagonal < wp.float64(0.0), -l1, l1), diagonal
    )
    out_scaled_diagonal[i] = sqrt_abs(diagonal)
    out_priority[i] = element_priority(seed, i)
    # A row with no nonzero off-diagonal entry is left out of every aggregate: the smoother solves
    # it exactly, and as an aggregate of its own it would stay an isolated row on every coarser
    # level. Where much of a system is pinned (``min_quad_with_fixed`` with half its unknowns
    # fixed at random) such rows multiply level by level -- 2 % of dragon's free rows, 81 % six
    # levels down -- so coarsening stalls just under ``linalg._MULTIGRID_MIN_COARSENING`` while the
    # rest densifies, and on lucy a level's product overran memory. It starts excluded, which no
    # round revisits, with its root flag written here since no round writes it.
    isolated = off_sum == wp.float64(0.0)
    out_state[i] = wp.where(isolated, MG_EXCLUDED, MG_UNDECIDED)
    if isolated:
        out_root_flags[i] = wp.int32(0)


@wp.kernel
def mis_level_setup(
    seed: wp.int32,
    l1_smoother: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_diagonal: wp.array[wp.float64],
    out_scaled_diagonal: wp.array[wp.float64],
    out_priority: wp.array[wp.uint32],
    out_state: wp.array[wp.int32],
    out_root_flags: wp.array[wp.int32],
    out_power: wp.array[wp.float64],
    out_counts: wp.array[wp.int32],
) -> None:
    # Everything one level reads per row before its aggregation's first round, in one launch: the
    # operator's diagonal (``linalg.csr_row_diagonal``, 0 for a row without one, as
    # ``bsr_get_diag`` gives it), the strength test's ``sqrt(|A_ii|)`` (``array.sqrt_abs``), the
    # priority order (``array.element_priority``, so the draw is exactly
    # ``array.random_priorities``'), the undecided start state, and the power iteration's first
    # step from its sign vector (``power_row``, reading the diagonal this thread just wrote), which
    # needs nothing but the row. Independent elementwise writes at one ``dim``; launched apart they
    # were four launches, a fill and a memset. Placed after ``power_row``, which it calls.
    #
    # Under ``l1_smoother`` (warp-uniform) the diagonal the smoother, the prolongator and the power
    # iteration read is the row's absolute sum ``|A_ii| + sum_{j != i} |A_ij|`` instead: the l1
    # Jacobi scaling, a convergent smoother for any symmetric positive-definite operator, where
    # plain Jacobi under one global damping over-relaxes the rows a non-M-matrix (a clamped
    # cotangent Laplacian) leaves far from diagonal dominance. The strength test keeps the true
    # diagonal.
    #
    # Launch tiled over ``MG_ROWS_PER_BLOCK`` rows a block.
    blk, lane = wp.tid()
    first, rows = block_chunk(offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if i == 0:
            # The rounds' root count (``mis_propagate_decide``); each round clears the undecided
            # slot.
            out_counts[1] = 0
        if csr_row_long(offsets, i):
            found = 1
            continue
        diagonal, off_sum = csr_row_diagonal(offsets, columns, values, i)
        mis_row_setup(
            i, seed, l1_smoother, diagonal, off_sum, out_diagonal, out_scaled_diagonal,
            out_priority, out_state, out_root_flags,
        )  # fmt: skip
        out_power[i] = power_row(
            seed, wp.int32(1), out_diagonal, offsets, columns, values, out_power, i
        )
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(offsets, i):
            diagonal, off_sum = csr_row_diagonal_block(offsets, columns, values, i, lane)
            if lane == 0:
                mis_row_setup(
                    i, seed, l1_smoother, diagonal, off_sum, out_diagonal, out_scaled_diagonal,
                    out_priority, out_state, out_root_flags,
                )  # fmt: skip
            # Every lane reads the diagonal lane 0 wrote.
            block_barrier()
            power = power_row_block(
                seed, wp.int32(1), out_diagonal, offsets, columns, values, out_power, i, lane
            )
            if lane == 0:
                out_power[i] = power


@wp.kernel
def power_step(
    seed: wp.int32,
    from_signs: wp.int32,
    diagonal: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
) -> None:
    # One ``power_row`` per thread. Writing a second buffer rather than updating ``x`` in place is
    # what lets a step be one launch; the caller swaps the two. The last step is
    # ``power_step_partials``, which also measures the growth. Launch tiled over
    # ``MG_ROWS_PER_BLOCK`` rows a block.
    blk, lane = wp.tid()
    first, rows = block_chunk(offsets.shape[0] - 1, blk, MG_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if csr_row_long(offsets, i):
            found = 1
            continue
        out_y[i] = power_row(seed, from_signs, diagonal, offsets, columns, values, x, i)
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(offsets, i):
            y = power_row_block(seed, from_signs, diagonal, offsets, columns, values, x, i, lane)
            if lane == 0:
                out_y[i] = y


@wp.kernel
def power_step_partials(
    seed: wp.int32,
    from_signs: wp.int32,
    diagonal: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
    out_partials: wp.array[wp.float64],
) -> None:
    # ``power_step``'s last step, which also stores its block's share of ``|y|^2`` in the block's
    # own slot: the growth ``damped_inverse_diagonal`` folds, in a fixed order, with no separate
    # inner-product pass over ``y``. Launched tiled over ``POWER_ROWS_PER_BLOCK`` rows a block,
    # lanes striding them by ``wp.block_dim()`` (so the one CPU lane walks them all).
    block, lane = wp.tid()
    offset, count = power_block_rows(out_y.shape[0], block)
    partial = wp.float64(0.0)
    found = wp.int32(0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        if csr_row_long(offsets, i):
            found = 1
            continue
        y = power_row(seed, from_signs, diagonal, offsets, columns, values, x, i)
        out_y[i] = y
        partial += y * y
    # A long row (``csr_row_long``) is the whole block's, after the lanes' pass.
    if block_sum(found) != 0:
        for k in range(count):
            i = offset + k
            if csr_row_long(offsets, i):
                y = power_row_block(
                    seed, from_signs, diagonal, offsets, columns, values, x, i, lane
                )
                if lane == 0:
                    out_y[i] = y
                    partial += y * y
    total = block_sum(partial)
    if lane == 0:
        out_partials[block] = total


@wp.kernel
def jacobi_sweep(
    n: wp.int32,
    stride: wp.int32,
    inv_diag: wp.array[wp.float64],
    omega: wp.float64,
    rhs: wp.array[wp.float64],
    operator_x: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # One damped-Jacobi sweep, ``x += w D^-1 (b - A x)``, with ``A x`` already in hand. ``out_x`` is
    # both the input and the result, which is what the sweep means.
    #
    # ``row = t % stride`` then ``if row >= n: return`` is the padded-row guard this file's cycle
    # kernels share; what the pad is and why it must stay unwritten is written out once, on
    # ``algorithms/conjugate_gradient.scaled_diagonal_apply``, which owns the padding contract. Not
    # factored into a ``@wp.func``: the caller still needs the flat ``t`` for its own indexing, so
    # a helper returning ``-1`` for the pad renames the three lines rather than removing any.
    t = wp.int32(wp.tid())
    row = t % stride
    if row >= n:
        return
    out_x[t] = jacobi_update(out_x[t], omega * inv_diag[row], rhs[t], operator_x[t])


@wp.kernel
def residual(
    n: wp.int32,
    stride: wp.int32,
    rhs: wp.array[wp.float64],
    operator_x: wp.array[wp.float64],
    out_residual: wp.array[wp.float64],
) -> None:
    t = wp.int32(wp.tid())
    row = t % stride
    if row >= n:
        return
    out_residual[t] = rhs[t] - operator_x[t]


@wp.kernel
def dense_solve(
    n: wp.int32,
    stride: wp.int32,
    inverse: wp.array2d[wp.float64],
    slots: wp.array[wp.int32],
    active: wp.array[wp.int32],
    rhs: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # The coarsest level, solved against a pseudo-inverse factored on the host at setup: a few dozen
    # unknowns, where a dense row dot is cheaper than any iteration and -- unlike an iteration whose
    # count depends on the data -- it is a single launch inside the captured cycle. The *pseudo*
    # inverse because the coarse operator inherits the fine one's null space. It covers only the
    # active rows (``slots`` maps a row to its position among them, ``-1`` for an inactive row,
    # which solves to zero; both maps are empty when every row is active): the terms an ``(n, n)``
    # inverse would add for the inactive columns are exact zeros.
    #
    # Launched tiled, a block an output slot, its lanes striding the dot by ``wp.block_dim()`` (the
    # one CPU lane walks it whole, in ascending order) and summing with ``block_sum``: one thread a
    # slot walked a few hundred dependent loads and was latency-bound (30 us a launch on dragon's
    # 203-row level, coalesced or not). The inverse is read down a column, ``inverse[k, slot]``,
    # which is row ``slot``'s value: it is exactly symmetric (``linalg._symmetric_inverse``).
    t, lane = wp.tid()
    column = t // stride
    row = t % stride
    if row >= n:
        return
    mapped = slots.shape[0] > 0
    slot = row
    if mapped:
        slot = slots[row]
    if slot < 0:
        if lane == 0:
            out_x[t] = wp.float64(0.0)
        return
    partial = wp.float64(0.0)
    for k in range(lane, inverse.shape[0], wp.block_dim()):
        source = k
        if mapped:
            source = active[k]
        partial += inverse[k, slot] * rhs[column * stride + source]
    total = block_sum(partial)
    if lane == 0:
        out_x[t] = total


CYCLE_LONG_ROWS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global CYCLE_LONG_ROWS
    i32 = wp.array[wp.int32]
    f64 = wp.array[wp.float64]
    # Keyed on the operator values' dtype: a narrowed level reads ``float32`` copies.
    CYCLE_LONG_ROWS = OverloadTable(
        cycle_long_rows,
        {
            v: [
                *(wp.int32, wp.int32, wp.int32, wp.array[wp.vec3i], i32, f64, i32, i32),
                *(wp.array[v], f64, f64, f64, i32, f64),
            ]
            for v in (wp.float32, wp.float64)
        },
    )


_register_overloads()


def _declare_block_dims() -> None:
    """Declare the CUDA launch widths of this module's kernels besides the default (precompile)."""
    declare_block_dims(__name__, (TILE_1D,))


_declare_block_dims()
