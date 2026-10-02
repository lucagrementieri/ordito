"""
Smoothed-aggregation algebraic multigrid: the setup passes and the V-cycle.

**Why a hierarchy at all.** Jacobi-preconditioned conjugate gradient's iteration count grows with
the mesh, roughly in proportion to the unknown count on an ill-conditioned operator. A multigrid
V-cycle attacks the low-frequency error the smoother cannot see, so the count stops growing; the
whole item is whether one cycle costs less than the iterations it removes, which is a *launch*
question on this hardware and not an arithmetic one. See ``triwarp.linalg`` for the gate. The
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
product ``P^T A P``, one ``bsr_mm`` for ``A P`` and the outer product of each fine row's two
sparse rows, summed as triplets, for ``P^T (A P)``. Against the earlier ``bsr_mm`` / ``bsr_axpy``
chain (and a ``bsr_transposed`` multiplied in) the prolongator is 1.9-4.0x and the Galerkin product
1.14-1.8x faster on the dragon and bunny cotangent Laplacians, the hierarchy's whole setup
1.23-1.51x, with the same patterns and values to round-off. Operator complexity comes out just
above 1, so the coarse levels are nearly free and a cycle's cost is its fine level.

The per-level inverse diagonal is ``array.inverse_or_one`` of the operator's diagonal, applied
where it is read rather than written to a buffer first. It is not a copy of the conjugate
gradient's: the quantity is the same one the Jacobi preconditioner needs, down to mapping a zero
diagonal to 1 rather than to infinity -- which the least-squares operators here rely on, since they
carry empty rows for free vertices no equation reaches. The strength test's ``sqrt(|A_ii|)`` is
``array.sqrt_abs`` over the same diagonal.
"""

from typing import Any

import warp as wp

from triwarp.kernels.array import element_priority, inverse_or_one, sqrt_abs

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
    # One hop of the lexicographic maximum over the operator's *strong off-diagonal* graph -- the
    # diagonal is skipped because a node is not its own neighbour, and the node's own key is folded
    # in separately so the reduction is over the closed neighbourhood. Two hops give the maximum
    # over the two-hop ball, which is the distance-2 test without a squared graph.
    #
    # The strength test is applied here rather than by materializing a filtered graph, so a level
    # pays no extra allocation and ``theta = 0`` is bit-exactly the unfiltered aggregation. Shared
    # by the round's two kernels below, which differ in where a neighbour's key comes from
    # (``mis_entry_key``) and in what the second does with the maximum.
    best = mis_entry_key(i, from_state, key, state, priority)
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        if j != i and mg_is_strong(values[k], scaled_diagonal[i], scaled_diagonal[j], theta):
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
) -> None:
    # A round's first hop, reading every key straight from ``(state, priority, index)``: the packed
    # key is an elementwise function of the state, so no seeding pass writes it first.
    # ``mis_propagate_decide`` is the round's second hop.
    i = wp.int32(wp.tid())
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
) -> None:
    # An unlabelled node adopts a neighbour's aggregate, largest id winning so the choice does not
    # depend on thread order. Two launches cover the two hops the MIS guarantees are enough; the
    # first (warp-uniform ``from_state != 0``) reads every label straight from the root flags'
    # scan, so no pass writes the labels first, and ``label`` is not read.
    #
    # The spread walks the same *strong* graph the independent set was selected on, which is what
    # keeps the tiling consistent: every excluded node has a root within two strong hops precisely
    # because the exclusion came from a strong-graph propagation.
    i = wp.int32(wp.tid())
    best = aggregate_entry_label(i, from_state, label, state, scan_pos)
    if best != MG_UNAGGREGATED:
        out_label[i] = best
        return
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        if j != i and mg_is_strong(values[k], scaled_diagonal[i], scaled_diagonal[j], theta):
            best = wp.max(best, aggregate_entry_label(j, from_state, label, state, scan_pos))
    out_label[i] = best


@wp.kernel
def aggregate_sizes(label: wp.array[wp.int32], out_sizes: wp.array[wp.int32]) -> None:
    i = wp.int32(wp.tid())
    wp.atomic_add(out_sizes, label[i], wp.int32(1))


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
        out_values[e] = scale * values[e] / wp.sqrt(wp.float64(sizes[aggregate]))
    own = offsets[n] + i
    out_rows[own] = i
    out_cols[own] = label[i]
    out_values[own] = wp.float64(1.0) / wp.sqrt(wp.float64(sizes[label[i]]))


@wp.kernel
def galerkin_triplet_counts(
    p_offsets: wp.array[wp.int32], ap_offsets: wp.array[wp.int32], out_counts: wp.array[wp.int32]
) -> None:
    # Row ``i`` of ``A P`` meets row ``i`` of ``P`` in every product ``P_ik (A P)_il``.
    i = wp.int32(wp.tid())
    out_counts[i] = (p_offsets[i + 1] - p_offsets[i]) * (ap_offsets[i + 1] - ap_offsets[i])


@wp.kernel
def galerkin_triplets(
    p_offsets: wp.array[wp.int32],
    p_columns: wp.array[wp.int32],
    p_values: wp.array[wp.float64],
    ap_offsets: wp.array[wp.int32],
    ap_columns: wp.array[wp.int32],
    ap_values: wp.array[wp.float64],
    starts: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # ``P^T (A P)`` as coordinate triplets, one thread a fine row: ``(P^T A P)_kl`` is the sum over
    # fine rows ``i`` of ``P_ik (A P)_il``, so row ``i`` emits the outer product of its two sparse
    # rows at ``starts[i]``. No transpose is formed and no second ``bsr_mm`` runs. ``bsr_mm``'s
    # structural zeros in ``A P`` become zero-valued triplets, which the triplet assembly skips.
    i = wp.int32(wp.tid())
    slot = starts[i]
    for a in range(p_offsets[i], p_offsets[i + 1]):
        k = p_columns[a]
        weight = p_values[a]
        for b in range(ap_offsets[i], ap_offsets[i + 1]):
            out_rows[slot] = k
            out_cols[slot] = ap_columns[b]
            out_values[slot] = weight * ap_values[b]
            slot += 1


@wp.kernel
def scale_rows(
    offsets: wp.array[wp.int32],
    row_scale: wp.array[wp.float64],
    factor: wp.float64,
    invert: wp.int32,
    values: wp.array[wp.float64],
) -> None:
    # Row-scale a matrix in place: the prolongation smoother needs ``-w D^-1 (A P0)``, and scaling
    # the product's values is one pass where ``bsr_mm`` against a diagonal matrix would be another
    # sparse product. ``values`` is both the input and the result. A nonzero ``invert`` (warp
    # uniform) scales by ``inverse_or_one(row_scale[i])`` instead, so a caller holding ``D`` rather
    # than ``D^-1`` (``linalg.squared_laplacian_preconditioner``) needs no map and no buffer first.
    i = wp.int32(wp.tid())
    raw = row_scale[i]
    scale = factor * wp.where(invert != 0, inverse_or_one(raw), raw)
    for k in range(offsets[i], offsets[i + 1]):
        values[k] = values[k] * scale


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


@wp.kernel
def damped_inverse_diagonal(
    growth: wp.array[wp.float64],
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
    # ``rho = sqrt(growth / start) ** exponent``, ``exponent`` being one over the step count, and
    # ``rho = 1`` when the growth is not a positive finite number (an operator the iteration cannot
    # measure), which is the host form's guard verbatim. Every thread forms the same two scalars;
    # that is cheaper than a ``dim=1`` launch and a second buffer.
    #
    # Pre-scaling changes no consumer's arithmetic: ``jacobi_sweep`` and ``scaled_diagonal_apply``
    # form ``factor * inv[row] * r`` left to right, so ``1 * (omega * inv[row]) * r`` is the same
    # product, and the prolongator's ``(-omega) * inv[row]`` is ``(-1) * (omega * inv[row])``.
    i = wp.int32(wp.tid())
    end = growth[0]
    rho = wp.float64(1.0)
    if start > wp.float64(0.0) and end > wp.float64(0.0) and wp.isfinite(end):
        rho = wp.pow(wp.sqrt(end / start), exponent)
    out_scaled[i] = (factor / rho) * inverse_or_one(diagonal[i])


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
    # ``y = D^-1 A x``: one whole step of the power iteration that estimates the spectral radius, in
    # one launch. It differs from ``csr_matvec`` above only in folding the diagonal scale in and in
    # being single-column -- and that fusion is the point, because the setup is launch-bound: as a
    # ``bsr_mv`` plus a ``scaled_diagonal_apply`` it is two launches, and an uncaptured ``bsr_mv``
    # costs a fixed host price whatever its nnz. Writing a second buffer rather than updating ``x``
    # in place is what lets it be one launch; the caller swaps the two.
    #
    # ``D^-1`` is ``array.inverse_or_one`` of the operator's diagonal, applied as it is read, so no
    # pass writes the inverse first. The first step (warp-uniform ``from_signs != 0``) draws its
    # ``x`` from ``random_sign`` as it reads it and does not read ``x``; its row dot is
    # ``csr_row_dot``'s, term for term.
    i = wp.int32(wp.tid())
    total = wp.float64(0.0)
    if from_signs != 0:
        for k in range(offsets[i], offsets[i + 1]):
            total += values[k] * random_sign(seed, columns[k])
    else:
        total = csr_row_dot(i, wp.int32(0), offsets, columns, values, x)
    out_y[i] = inverse_or_one(diagonal[i]) * total


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
