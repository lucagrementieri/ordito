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
re-keyed by their column's aggregate and summed as triplets; the coarse operator is the Galerkin
product ``P^T A P``, ``A P`` summed from its ``A_ij P_jl`` triplets and ``P^T (A P)`` from the
outer product of each fine row's two sparse rows. No ``bsr_mm`` runs in the setup. Against the
earlier ``bsr_mm`` / ``bsr_axpy`` chain the prolongator is 1.9-4.0x faster on the dragon and bunny
cotangent Laplacians, with the same patterns and values to round-off; see ``product_triplets`` and
``galerkin_triplets`` for the products. Operator complexity comes out just above 1, so the coarse
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
from ordito.kernels.array import element_priority, inverse_or_one, sqrt_abs
from ordito.kernels.reduce import block_chunk, block_sum

# Node states for the distance-2 maximal independent set. The encoding is ordered rather than
# arbitrary: a root must win any maximum (it vetoes every node in its two-hop ball) and an excluded
# node must lose to every undecided one (it can no longer veto anything).
MG_EXCLUDED = wp.constant(wp.int32(0))
MG_UNDECIDED = wp.constant(wp.int32(1))
MG_ROOT = wp.constant(wp.int32(2))

# Layout of the packed comparison key: state in bits 60-61, priority in bits 32-59, index in bits
# 0-31. Two hops of a plain integer ``max`` then implement the lexicographic order, and the key
# stays positive so a signed ``wp.int64`` compares correctly. 28 bits of priority is plenty -- the
# index breaks any tie, which is what makes the aggregation independent of thread order.
_MG_STATE_SHIFT = wp.constant(wp.int64(60))
_MG_PRIORITY_SHIFT = wp.constant(wp.int64(32))
_MG_PRIORITY_MASK = wp.constant(wp.uint32(0x0FFFFFFF))

MG_UNAGGREGATED = wp.constant(wp.int32(-1))


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


@wp.kernel
def mis_level_setup(
    seed: wp.int32,
    diagonal: wp.array[wp.float64],
    out_scaled_diagonal: wp.array[wp.float64],
    out_priority: wp.array[wp.uint32],
    out_state: wp.array[wp.int32],
) -> None:
    # Everything one level's aggregation reads per row before its first round, in one launch: the
    # strength test's ``sqrt(|A_ii|)`` (``array.sqrt_abs``), the priority order
    # (``array.element_priority``, so the draw is exactly ``array.random_priorities``') and the
    # undecided start state. Three independent elementwise writes at one ``dim``; launched apart
    # they were two launches and a fill.
    i = wp.int32(wp.tid())
    out_scaled_diagonal[i] = sqrt_abs(diagonal[i])
    out_priority[i] = element_priority(seed, i)
    out_state[i] = MG_UNDECIDED


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
    best = mis_entry_key(i, from_state, key, state, priority)
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        if mg_strong_neighbor(i, j, values[k], scaled_diagonal, theta):
            best = wp.max(best, mis_entry_key(j, from_state, key, state, priority))
    return best


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
    # cleared here, the launch before, so no memset runs per round.
    i = wp.int32(wp.tid())
    if i == 0:
        out_undecided[0] = 0
    out_key[i] = mis_ball_max(
        i, wp.int32(1), out_key, state, priority, offsets, columns, values, scaled_diagonal, theta
    )


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
    i = wp.int32(wp.tid())
    current = state[i]
    if current != MG_UNDECIDED:
        out_state[i] = current
        return
    best = mis_ball_max(
        i, wp.int32(0), key, state, priority, offsets, columns, values, scaled_diagonal, theta
    )
    verdict = MG_UNDECIDED
    if wp.int32(best >> _MG_STATE_SHIFT) == MG_ROOT:
        verdict = MG_EXCLUDED
    elif best == mis_key(current, priority[i], i):
        verdict = MG_ROOT
    out_state[i] = verdict
    out_root_flags[i] = mis_root_flag(verdict)
    if verdict == MG_UNDECIDED:
        wp.atomic_add(out_undecided, 0, wp.int32(1))


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
    out_label: wp.array[wp.int32],
    out_sizes: wp.array[wp.int32],
) -> None:
    # An unlabelled node adopts a neighbour's aggregate, largest id winning so the choice does not
    # depend on thread order. Two launches cover the two hops the MIS guarantees are enough; the
    # first (warp-uniform ``from_state != 0``) reads every label straight from the root flags'
    # scan, so no pass writes the labels first, and ``label`` is not read. The second hop, whose
    # labels are final, also counts each aggregate's rows into the zeroed ``out_sizes`` (the first
    # hop passes ``None`` and never reads it), so no separate counting pass runs over the labels.
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
        wp.atomic_add(out_sizes, best, wp.int32(1))


@wp.func
def tentative_norm(sizes: wp.array[wp.int32], aggregate: wp.int32) -> wp.float64:
    # ``sqrt(|aggregate|)``, the norm of the constant near-nullspace vector over an aggregate's
    # column: the tentative prolongator ``P0``'s entry in a row of that aggregate is its reciprocal.
    # Returned undivided so a caller scaling by it divides once, as it always has.
    return wp.sqrt(wp.float64(sizes[aggregate]))


@wp.func
def emit_scaled_row(
    row: wp.int32,
    weight: wp.float64,
    begin: wp.int32,
    end: wp.int32,
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    slot: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> wp.int32:
    # ``weight`` times the sparse row ``[begin, end)`` of a CSR, as triplets in output row ``row``
    # from ``slot`` on; returns the slot after the last. The emission both row-local products below
    # are made of: ``A P`` emits ``A_ij`` times ``P``'s row ``j`` into row ``i``, ``P^T (A P)``
    # emits ``P_ik`` times ``A P``'s row ``i`` into row ``k``.
    for b in range(begin, end):
        out_rows[slot] = row
        out_cols[slot] = columns[b]
        out_values[slot] = weight * values[b]
        slot += 1
    return slot


@wp.kernel
def smoothed_prolongator_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    label: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    damped_inverse_diagonal: wp.array[wp.float64],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # The smoothed prolongator ``P = (I - w D^-1 A) P0`` as coordinate triplets, one thread a row.
    # ``P0`` holds one entry a row -- node ``j``'s aggregate, ``1 / sqrt(|aggregate|)``, the
    # constant near-nullspace vector normalized per column -- so ``(A P0)_ik`` is the sum of
    # ``A_ij P0_j`` over the row's columns ``j`` in aggregate ``k``: every stored ``A_ij`` is one
    # triplet at ``(i, label[j])``, at the slot ``A`` stores it in, and the identity's ``P0_i`` one
    # more, after all of ``A``'s. The triplet sum that follows is the whole product, where
    # ``bsr_mm`` against ``P0`` then a row scale and a ``bsr_axpy`` against ``P0`` was three
    # sparse passes, each allocating its result.
    i = wp.int32(wp.tid())
    n = offsets.shape[0] - 1
    scale = -damped_inverse_diagonal[i]
    for e in range(offsets[i], offsets[i + 1]):
        aggregate = label[columns[e]]
        out_rows[e] = i
        out_cols[e] = aggregate
        out_values[e] = scale * values[e] / tentative_norm(sizes, aggregate)
    own = offsets[n] + i
    out_rows[own] = i
    out_cols[own] = label[i]
    out_values[own] = wp.float64(1.0) / tentative_norm(sizes, label[i])


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
    # total that a budget test cannot tell from a real one.
    i = wp.int32(wp.tid())
    count = wp.int64(0)
    for e in range(a_offsets[i], a_offsets[i + 1]):
        j = a_columns[e]
        count += wp.int64(p_offsets[j + 1] - p_offsets[j])
    out_counts[i + 1] = count
    if i == 0:
        out_counts[0] = wp.int64(0)


@wp.kernel
def product_triplets(
    a_offsets: wp.array[wp.int32],
    a_columns: wp.array[wp.int32],
    a_values: wp.array[wp.float64],
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    starts: wp.array[wp.int64],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # ``A P`` as coordinate triplets, one thread a row, at ``starts[i]``: the triplet sum that
    # follows replaces ``bsr_mm`` (1.7x on the product at the bunny Laplacians, 1.04x at dragon's
    # 9 M triplets) and leaves no structural zeros behind.
    i = wp.int32(wp.tid())
    slot = wp.int32(starts[i])  # the wrapper capped the total below ``2^30``
    for e in range(a_offsets[i], a_offsets[i + 1]):
        j = a_columns[e]
        slot = emit_scaled_row(
            i,
            a_values[e],
            p_offsets[j],
            p_offsets[j + 1],
            p_columns,
            p_values,
            slot,
            out_rows,
            out_cols,
            out_values,
        )


@wp.kernel
def galerkin_triplet_counts(
    p_offsets: wp.array[wp.int32], ap_offsets: wp.array[wp.int32], out_counts: wp.array[wp.int64]
) -> None:
    # Row ``i`` of ``A P`` meets row ``i`` of ``P`` in every product ``P_ik (A P)_il``. Written at
    # ``i + 1`` of an ``n + 1`` buffer, so the inclusive scan of it is the starts and the total.
    i = wp.int32(wp.tid())
    out_counts[i + 1] = wp.int64(p_offsets[i + 1] - p_offsets[i]) * wp.int64(
        ap_offsets[i + 1] - ap_offsets[i]
    )
    if i == 0:
        out_counts[0] = wp.int64(0)


@wp.kernel
def galerkin_triplets(
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    ap_offsets: wp.array[wp.int32],
    ap_columns: wp.array[wp.int32],
    ap_values: wp.array[wp.float64],
    starts: wp.array[wp.int64],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # ``P^T (A P)`` as coordinate triplets, one thread a fine row: ``(P^T A P)_kl`` is the sum over
    # fine rows ``i`` of ``P_ik (A P)_il``, so row ``i`` emits the outer product of its two sparse
    # rows at ``starts[i]``. No transpose is formed and no second product runs.
    i = wp.int32(wp.tid())
    slot = wp.int32(starts[i])  # the wrapper capped the total below ``2^30``
    for a in range(p_offsets[i], p_offsets[i + 1]):
        slot = emit_scaled_row(
            p_columns[a],
            p_values[a],
            ap_offsets[i],
            ap_offsets[i + 1],
            ap_columns,
            ap_values,
            slot,
            out_rows,
            out_cols,
            out_values,
        )


@wp.func
def csr_row_dot(
    row: wp.int32,
    x_offset: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    x: wp.array[wp.Float],
) -> wp.Float:
    # One CSR row against one column of ``x``. Shared by the cycle's mat-vec and the power
    # iteration's step below, which differ only in what they do with the result -- and the row bound
    # comes from ``offsets`` alone, so neither depends on the matrix's ``nnz`` field being fresh.
    #
    # Generic over the storage precision, and accumulated at ``x``'s: the conjugate-gradient
    # mat-vec also runs on ``float32`` systems, whose rows ``warp.fem`` assembles a hundred entries
    # long, where a ``float64`` accumulator measured well behind ``warp.sparse.bsr_mv``'s own. The
    # values may be narrower than ``x`` -- the heat diffusions' ``float32`` copy of their operator
    # (``linalg._BatchedCg``'s ``narrow_values``) -- and are widened as they are read; at one
    # precision the cast is the identity and the arithmetic is unchanged.
    total = x.dtype(0.0)
    for k in range(offsets[row], offsets[row + 1]):
        total += x.dtype(values[k]) * x[x_offset + columns[k]]
    return total


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


@wp.kernel
def chebyshev_step(
    n_rows: wp.int32,
    stride: wp.int32,
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
    # sizes. ``out_x`` must alias neither ``x`` (the row dot reads it at other rows) nor
    # ``x_previous``. Shares ``csr_row_dot`` with ``csr_matvec``.
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    base = column * stride
    slot = base + row
    ax = csr_row_dot(row, base, offsets, columns, values, x)
    if row_scaled != wp.int32(0):
        ax = ax * row_scale[row]
    out_x[slot] = chebyshev_update(steps[index], x[slot], x_previous[slot], source[slot], ax)


# Rows a block of ``power_step_partials`` / ``damped_inverse_diagonal`` owns, its ``TILE_1D``
# lanes striding them. Four rows a lane rather than the ``reduce`` fold width
# (``ITEMS_PER_BLOCK_1D``, sixteen): a level of 40 k rows then launches 160 blocks rather than 40
# on a 170-SM part (the narrower block measured level at 40 k rows where sixteen read ~2 % slower).
# Every ``damped_inverse_diagonal`` block folds all the partials, so the wrapper pre-folds them with
# ``reduce.SUM1D_PARTIALS`` past ``4 * ITEMS_PER_BLOCK_1D`` (about 1 M rows): folding 2 560 per
# block at 655 k rows was 1.03x the old inner-product pass, one pre-fold launch 1.02x.
POWER_ROWS_PER_BLOCK = wp.constant(4 * TILE_1D)


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
    # ``power_step_partials``, which also measures the growth.
    i = wp.int32(wp.tid())
    out_y[i] = power_row(seed, from_signs, diagonal, offsets, columns, values, x, i)


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
    # lanes striding them by ``wp.block_dim()`` (so the one CPU lane walks them all, CLAUDE.md
    # section 2.2).
    block, lane = wp.tid()
    offset, count = power_block_rows(out_y.shape[0], block)
    partial = wp.float64(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        y = power_row(seed, from_signs, diagonal, offsets, columns, values, x, i)
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
    out_x[t] += omega * inv_diag[row] * (rhs[t] - operator_x[t])


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
    rhs: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # The coarsest level, solved against a pseudo-inverse factored on the host at setup: a few dozen
    # unknowns, where a dense row dot is cheaper than any iteration and -- unlike an iteration whose
    # count depends on the data -- it is a single launch inside the captured cycle. The *pseudo*
    # inverse because the coarse operator inherits the fine one's null space.
    t = wp.int32(wp.tid())
    column = t // stride
    row = t % stride
    if row >= n:
        return
    total = wp.float64(0.0)
    for k in range(n):
        total += inverse[row, k] * rhs[column * stride + k]
    out_x[t] = total
