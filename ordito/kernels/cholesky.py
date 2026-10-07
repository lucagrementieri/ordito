"""
Kernels for ``ordito/cholesky.py``: a multifrontal sparse Cholesky and its triangular solves.

**Numeric factorization.** The fronts are dense ``float64`` matrices, one per supernode of the
nested-dissection tree, stored row-major in one arena and factored level by level from the leaves:
``scatter_entries`` writes the operator's lower entries into their fronts, ``factor_panel``,
``panel_rows`` and ``update_panel`` (three launches per ``PANEL`` columns, every front of the
level at once, the rows and the update spread over the device) factor each front's own columns,
and ``extend_add``
folds the Schur complement left in the front's trailing block into its parent. Alongside the factor
each front writes its **solve block** ``[L11^-1; -L21 L11^-1]`` (column-major, leading dimension
the front's size), so the triangular solves are matrix-vector products rather than substitutions:
the *partitioned inverse* of Alvarado, Pothen and Schreiber (1993).

**Triangular solves.** Level by level as well: ``forward_gather`` forms a supernode's right-hand
side from the original one and every descendant's contribution, ``forward_partial`` multiplies a
row of a solve block by ``CHUNK`` gathered entries, ``forward_finalize`` sums each row's chunks;
then ``backward_partial`` / ``backward_finalize`` run the transposed product from the root down.
Every partial product lands in its own slot and every sum reads its slots in a fixed order, so a
solve is bit-reproducible on each device (no float atomics).

``ordito.cholesky`` refines every solve against the operator itself (``residual_test``, one more
solve, whose last level adds the correction) until the residual passes its test.

Every kernel that cooperates strides its lanes by ``wp.block_dim()``, so the CPU device's
one-lane blocks run the same code.
"""

import warp as wp

from ordito.kernels.array import binary_search_index, ordered_float_bits, scanned_count
from ordito.kernels.reduce import block_barrier, block_sum

wp.set_module_options({"enable_backward": False})

# Entries of a solve-block row (forward) or column (backward) one partial product covers.
CHUNK = wp.constant(8)
# Columns factored per panel launch pair in the multi-block path.
PANEL = wp.constant(16)
panel_vector = wp.types.vector(length=int(PANEL), dtype=wp.float64)

# ``refine_state`` slots: loop condition, "some entry has not converged", round counter.
REFINE_CONDITION = wp.constant(0)
REFINE_PENDING = wp.constant(1)
REFINE_ROUND = wp.constant(2)


# --------------------------------------------------------------------------------------
# Singular components
# --------------------------------------------------------------------------------------


@wp.kernel
def mark_nonsingular_components(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    labels: wp.array[wp.int32],
    tolerance: wp.float64,
    out_singular: wp.array[wp.int32],
    out_pinned: wp.array[wp.int32],
    out_dropped: wp.array[wp.int32],
) -> None:
    # A component is singular when every one of its rows sums to zero (a Laplacian's constant null
    # space); ``out_singular`` is seeded with ones and any row that does not sum to zero clears its
    # component. An empty row (an unreferenced vertex) sums to zero: a component of one. Row ``i``
    # also clears its two pin flags, which ``pin_singular_components`` then sets.
    i = wp.int32(wp.tid())
    out_pinned[i] = 0
    out_dropped[i] = 0
    total = wp.float64(0.0)
    diagonal = wp.float64(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        total += values[e]
        if columns[e] == i:
            diagonal = values[e]
    if wp.abs(total) > tolerance * wp.abs(diagonal):
        out_singular[labels[i]] = 0


@wp.kernel
def pin_singular_components(
    component_label: wp.array[wp.int32],
    component_last: wp.array[wp.int32],
    perm: wp.array[wp.int32],
    singular: wp.array[wp.int32],
    component_diagonal: wp.array[wp.int64],
    out_pinned: wp.array[wp.int32],
    out_dropped: wp.array[wp.int32],
    out_fronts: wp.array[wp.float64],
) -> None:
    # One pin per singular component, at its last-eliminated row: that row's equation is dropped
    # and its unknown held at zero, which leaves a consistent system solvable. ``out_pinned`` is
    # indexed in elimination order, ``out_dropped`` in the operator's own. A pinned row's diagonal
    # is the identity's even where the operator stores none (an unreferenced vertex's empty row),
    # so it is set here in the zeroed fronts; ``scatter_entries``, which runs after, writes the
    # same one wherever the operator does store it.
    c = wp.int32(wp.tid())
    pin = singular[component_label[c]]
    out_pinned[component_last[c]] = pin
    out_dropped[perm[component_last[c]]] = pin
    if pin != 0:
        out_fronts[component_diagonal[c]] = wp.float64(1.0)


# --------------------------------------------------------------------------------------
# Numeric factorization
# --------------------------------------------------------------------------------------


@wp.kernel
def scatter_entries(
    values: wp.array[wp.float64],
    entry_source: wp.array[wp.int32],
    entry_target: wp.array[wp.int64],
    entry_row: wp.array[wp.int32],
    entry_column: wp.array[wp.int32],
    pinned: wp.array[wp.int32],
    sign: wp.float64,
    out_fronts: wp.array[wp.float64],
) -> None:
    # One stored lower entry into its front, times ``sign`` (``-1`` factors a negative
    # semi-definite operator's negation). A pinned row or column becomes the identity's.
    e = wp.int32(wp.tid())
    r = entry_row[e]
    c = entry_column[e]
    value = sign * values[entry_source[e]]
    if pinned[r] != 0 or pinned[c] != 0:
        value = wp.where(r == c, wp.float64(1.0), wp.float64(0.0))
    out_fronts[entry_target[e]] = value


@wp.func
def front_index(base: wp.int64, size: wp.int64, row: wp.int32, column: wp.int32) -> wp.int64:
    return base + wp.int64(row) * size + wp.int64(column)


@wp.kernel
def extend_add(
    children: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    parent_local: wp.array[wp.int32],
    fronts: wp.array[wp.float64],
) -> None:
    # A child's lower Schur complement (its front's trailing block) into its parent's front. The
    # launch holds at most one child per parent, so no two threads add into one entry.
    c, t = wp.tid()
    k = children[c]
    nrow = front_size[k] - ncol[k]
    a = t // wp.max(nrow, 1)
    b = t % wp.max(nrow, 1)
    if a >= nrow or b > a:
        return
    p = parent[k]
    size = wp.int64(front_size[k])
    psize = wp.int64(front_size[p])
    rows = row_offsets[k]
    source = front_index(front_offset[k], size, ncol[k] + a, ncol[k] + b)
    target = front_index(front_offset[p], psize, parent_local[rows + a], parent_local[rows + b])
    fronts[target] = fronts[target] + fronts[source]


@wp.func
def factor_columns(
    base: wp.int64,
    size: wp.int32,
    rows: wp.int32,
    first: wp.int32,
    last: wp.int32,
    lane: wp.int32,
    width: wp.int32,
    fronts: wp.array[wp.float64],
    status: wp.array[wp.int32],
) -> None:
    # Right-looking Cholesky of columns ``[first, last)`` of a row-major front of ``size``,
    # through its first ``rows`` rows, updating only the columns of that panel; a pivot that is
    # not positive flags ``status`` and is replaced by one.
    m64 = wp.int64(size)
    for j in range(first, last):
        pivot = fronts[front_index(base, m64, j, j)]
        if not (pivot > wp.float64(0.0)):
            if lane == 0:
                status[0] = 1
            pivot = wp.float64(1.0)
        diagonal = wp.sqrt(pivot)
        inverse = wp.float64(1.0) / diagonal
        i = j + 1 + lane
        while i < rows:
            fronts[front_index(base, m64, i, j)] = fronts[front_index(base, m64, i, j)] * inverse
            i += width
        block_barrier()
        if lane == 0:
            fronts[front_index(base, m64, j, j)] = diagonal
        span = last - j - 1
        t = lane
        while t < (rows - j - 1) * span:
            row = j + 1 + t // span
            column = j + 1 + t % span
            if column <= row:
                target = front_index(base, m64, row, column)
                fronts[target] = (
                    fronts[target]
                    - fronts[front_index(base, m64, row, j)]
                    * fronts[front_index(base, m64, column, j)]
                )
            t += width
        block_barrier()


@wp.func
def panel_span(ncol: wp.int32, panel: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Columns ``[first, last)`` of a front's panel ``panel``; empty (``last <= first``) once the
    # front's ``ncol`` columns are exhausted, which a level's narrower fronts reach first.
    first = panel * PANEL
    return first, wp.min(first + PANEL, ncol)


@wp.func
def substitute_panel(
    base: wp.int64,
    m64: wp.int64,
    first: wp.int32,
    last: wp.int32,
    start: wp.int32,
    fronts: wp.array[wp.float64],
    target: wp.array[wp.float64],
    target_base: wp.int64,
) -> None:
    # Forward substitution of ``target[target_base + j]``, ``j`` in ``[first, last)``, against the
    # factored panel's own lower triangle, in place, the solved entries held in registers. Entries
    # below ``start`` are left alone and enter the later rows as zero: a solve-block column ``c``
    # of ``L11^-1`` is zero above its diagonal. ``panel_rows`` calls it for a row of the front
    # below the panel (``target`` the front itself) and for a column of the solve block.
    values = panel_vector()
    for q1 in range(PANEL):
        j = first + wp.int32(q1)
        if j < last and j >= start:
            total = target[target_base + wp.int64(j)]
            for q2 in range(PANEL):
                if wp.int32(q2) < wp.int32(q1):
                    total -= values[q2] * fronts[front_index(base, m64, j, first + wp.int32(q2))]
            values[q1] = total / fronts[front_index(base, m64, j, j)]
            target[target_base + wp.int64(j)] = values[q1]


@wp.kernel
def seed_blocks(
    block_offset: wp.array[wp.int64],
    front_size: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    blocks: wp.array[wp.float64],
) -> None:
    # Every solve block starts as ``[I; 0]``: the zeroed arena's diagonal set, one row per thread.
    i = wp.int32(wp.tid())
    k = owner[i]
    local = wp.int64(i - c0[k])
    blocks[block_offset[k] + local * wp.int64(front_size[k]) + local] = wp.float64(1.0)


@wp.kernel
def factor_panel(
    level: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    panel: wp.int32,
    fronts: wp.array[wp.float64],
    status: wp.array[wp.int32],
) -> None:
    # A large front's panel ``panel``: the Cholesky of its ``PANEL``-square diagonal block, one
    # block per front. ``panel_rows`` then forms the panel's rows below it and its rows of the solve
    # block, and ``update_panel`` spreads the panel's rank-``PANEL`` update.
    b, lane = wp.tid()
    k = level[b]
    first, last = panel_span(ncol[k], panel)
    if last <= first:
        return
    base = front_offset[k]
    size = front_size[k]
    width = wp.block_dim()
    factor_columns(base, size, last, first, last, lane, width, fronts, status)


@wp.kernel
def panel_rows(
    level: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    block_offset: wp.array[wp.int64],
    panel: wp.int32,
    rows_below: wp.int32,
    fronts: wp.array[wp.float64],
    blocks: wp.array[wp.float64],
) -> None:
    # After ``factor_panel``, one thread per row below the panel's diagonal block (``L[i, panel] =
    # A[i, panel] L_pp^-T``, forward substitution against the factored block) and, past
    # ``rows_below``, one per solve-block column (the panel rows' own triangle). Both are
    # ``substitute_panel``.
    b, t = wp.tid()
    k = level[b]
    first, last = panel_span(ncol[k], panel)
    if last <= first:
        return
    size = front_size[k]
    base = front_offset[k]
    m64 = wp.int64(size)
    if t >= rows_below:
        c = t - rows_below
        if c < last:
            substitute_panel(base, m64, first, last, c, fronts, blocks,
                             block_offset[k] + wp.int64(c) * m64)  # fmt: skip
        return
    row = last + t
    if row >= size:
        return
    substitute_panel(base, m64, first, last, first, fronts, fronts, front_index(base, m64, row, 0))


@wp.kernel
def update_panel(
    level: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    block_offset: wp.array[wp.int64],
    panel: wp.int32,
    fronts: wp.array[wp.float64],
    blocks: wp.array[wp.float64],
) -> None:
    # The rank-sixteen update after ``factor_panel``: one thread per updated entry, of the front's
    # trailing lower triangle and of the solve block's rows below the panel.
    b, t = wp.tid()
    k = level[b]
    first, last = panel_span(ncol[k], panel)
    if last <= first:
        return
    base = front_offset[k]
    size = front_size[k]
    m64 = wp.int64(size)
    span = size - last
    if t < span * span:
        row = last + t // span
        column = last + t % span
        if column <= row:
            row_base = front_index(base, m64, row, first)
            column_base = front_index(base, m64, column, first)
            total = wp.float64(0.0)
            for q1 in range(PANEL):
                if first + wp.int32(q1) < last:
                    total += fronts[row_base + wp.int64(q1)] * fronts[column_base + wp.int64(q1)]
            target = front_index(base, m64, row, column)
            fronts[target] = fronts[target] - total
        return
    t2 = t - span * span
    if t2 < span * last:
        block = block_offset[k]
        row = last + t2 % span
        column = t2 // span
        row_base = front_index(base, m64, row, first)
        column_base = block + wp.int64(column) * m64 + wp.int64(first)
        total = wp.float64(0.0)
        for q2 in range(PANEL):
            if first + wp.int32(q2) < last:
                total += fronts[row_base + wp.int64(q2)] * blocks[column_base + wp.int64(q2)]
        target = block + wp.int64(column) * m64 + wp.int64(row)
        blocks[target] = blocks[target] - total


# --------------------------------------------------------------------------------------
# Triangular solves
# --------------------------------------------------------------------------------------


@wp.func
def chunk_dot(
    blocks: wp.array[wp.float64],
    block: wp.int64,
    stride: wp.int64,
    values: wp.array[wp.float64],
    source: wp.int32,
    count: wp.int32,
) -> wp.float64:
    # ``sum_j blocks[block + j stride] values[source + j]`` over ``j < count <= CHUNK``, in four
    # independent accumulators: a dependent ``float64`` FMA is slow on consumer parts.
    acc = wp.vec4d()
    for jj in range(CHUNK):
        j = wp.int32(jj)
        if j < count:
            acc[j % 4] += blocks[block + wp.int64(j) * stride] * values[source + j]
    return (acc[0] + acc[1]) + (acc[2] + acc[3])


@wp.kernel
def forward_gather(
    level_rows: wp.array[wp.int32],
    rhs: wp.array[wp.float64],
    perm: wp.array[wp.int32],
    pinned: wp.array[wp.int32],
    contribution_offsets: wp.array[wp.int32],
    contribution_pair: wp.array[wp.int32],
    pair_values: wp.array[wp.float64],
    n: wp.int32,
    n_pairs: wp.int32,
    out_gathered: wp.array[wp.float64],
) -> None:
    # A level's rows of the forward right-hand side: the original entry plus every descendant's
    # contribution (their solve blocks' lower rows carry the sign), summed in a fixed order.
    t, column = wp.tid()
    i = level_rows[t]
    value = rhs[column * n + perm[i]]
    pairs = wp.int64(column) * wp.int64(n_pairs)
    for e in range(contribution_offsets[t], contribution_offsets[t + 1]):
        value += pair_values[pairs + wp.int64(contribution_pair[e])]
    if pinned[i] != 0:
        value = wp.float64(0.0)
    out_gathered[column * n + i] = value


@wp.kernel
def forward_partial(
    task_block: wp.array[wp.int64],
    task_stride: wp.array[wp.int32],
    task_count: wp.array[wp.int32],
    task_source: wp.array[wp.int32],
    task_slot: wp.array[wp.int32],
    blocks: wp.array[wp.float64],
    gathered: wp.array[wp.float64],
    n: wp.int32,
    n_parts: wp.int32,
    out_parts: wp.array[wp.float64],
) -> None:
    # One row of a solve block against ``CHUNK`` of its supernode's gathered entries.
    t, column = wp.tid()
    out_parts[wp.int64(column) * wp.int64(n_parts) + wp.int64(task_slot[t])] = chunk_dot(
        blocks, task_block[t], wp.int64(task_stride[t]), gathered, column * n + task_source[t],
        task_count[t],
    )  # fmt: skip


@wp.func
def slot_sum(
    parts: wp.array[wp.float64],
    base: wp.int64,
    slot: wp.int32,
    stride: wp.int32,
    begin: wp.int32,
    end: wp.int32,
) -> wp.float64:
    # A row's partial products ``begin <= q < end``, at ``slot + q * stride``, summed in order:
    # the fixed order that makes a solve bit-reproducible.
    value = wp.float64(0.0)
    for q in range(begin, end):
        value += parts[base + wp.int64(slot + q * stride)]
    return value


@wp.kernel
def forward_finalize(
    row_slot: wp.array[wp.int32],
    row_stride: wp.array[wp.int32],
    row_count: wp.array[wp.int32],
    row_target: wp.array[wp.int32],
    parts: wp.array[wp.float64],
    n: wp.int32,
    n_parts: wp.int32,
    n_pairs: wp.int32,
    out_forward: wp.array[wp.float64],
    out_pair_values: wp.array[wp.float64],
) -> None:
    # A level's solve-block rows: their chunks summed in order. A diagonal-block row is the
    # forward solution's entry; a lower row is a contribution to an ancestor (``-1 - pair``).
    t, column = wp.tid()
    value = slot_sum(
        parts, wp.int64(column) * wp.int64(n_parts), row_slot[t], row_stride[t], 0, row_count[t]
    )
    target = row_target[t]
    if target >= 0:
        out_forward[column * n + target] = value
    else:
        out_pair_values[wp.int64(column) * wp.int64(n_pairs) + wp.int64(-1 - target)] = value


@wp.kernel
def backward_partial(
    task_block: wp.array[wp.int64],
    task_first: wp.array[wp.int32],
    task_ncol: wp.array[wp.int32],
    task_size: wp.array[wp.int32],
    task_start: wp.array[wp.int32],
    task_rows: wp.array[wp.int32],
    task_slot: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    blocks: wp.array[wp.float64],
    forward: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    n: wp.int32,
    n_parts: wp.int32,
    out_parts: wp.array[wp.float64],
) -> None:
    # ``CHUNK`` entries of one solve-block column, transposed, against its supernode's forward
    # entries and its ancestors' finished solution entries.
    t, column = wp.tid()
    block = task_block[t]
    first = task_first[t]
    ncol = task_ncol[t]
    size = task_size[t]
    start = task_start[t]
    row_start = task_rows[t]
    offset = column * n
    acc = wp.vec4d()
    for kk in range(CHUNK):
        k = first + wp.int32(kk)
        if k < ncol:
            acc[wp.int32(kk) % 4] += blocks[block + wp.int64(k)] * forward[offset + start + k]
        elif k < size:
            acc[wp.int32(kk) % 4] += (
                blocks[block + wp.int64(k)] * solution[offset + rows[row_start + k - ncol]]
            )
    out_parts[wp.int64(column) * wp.int64(n_parts) + wp.int64(task_slot[t])] = (acc[0] + acc[1]) + (
        acc[2] + acc[3]
    )


@wp.kernel
def backward_finalize(
    row_slot: wp.array[wp.int32],
    row_stride: wp.array[wp.int32],
    row_first: wp.array[wp.int32],
    row_count: wp.array[wp.int32],
    row_target: wp.array[wp.int32],
    pinned: wp.array[wp.int32],
    parts: wp.array[wp.float64],
    perm: wp.array[wp.int32],
    refined: wp.array[wp.float64],
    n: wp.int32,
    n_parts: wp.int32,
    out_correction: wp.array[wp.float64],
) -> None:
    # A level's entries of the correction (in elimination order, which the next level's
    # ``backward_partial`` reads): their chunks summed in order. Each is also added to the refined
    # solution at its operator row: every row is finished by exactly one level, so this is the
    # whole correction's scatter, one launch fewer per refinement round.
    t, column = wp.tid()
    value = slot_sum(
        parts, wp.int64(column) * wp.int64(n_parts), row_slot[t], row_stride[t], row_first[t],
        row_count[t],
    )  # fmt: skip
    i = row_target[t]
    if pinned[i] != 0:
        value = wp.float64(0.0)
    out_correction[column * n + i] = value
    target = column * n + perm[i]
    refined[target] = refined[target] + value


# --------------------------------------------------------------------------------------
# Iterative refinement
# --------------------------------------------------------------------------------------


@wp.kernel
def refine_start(
    rhs: wp.array[wp.float64],
    n: wp.int32,
    state: wp.array[wp.int32],
    rhs_scale: wp.array[wp.float64],
    initial: wp.array[wp.float64],
    out_solution: wp.array[wp.float64],
) -> None:
    # Arm the refinement loop, record each column's largest right-hand-side entry, and start the
    # solution from the caller's initial guess.
    i, column = wp.tid()
    out_solution[column * n + i] = initial[column * n + i]
    if i == 0 and column == 0:
        state[REFINE_CONDITION] = 0
        state[REFINE_PENDING] = 0
        state[REFINE_ROUND] = 0
    wp.atomic_max(rhs_scale, column, wp.abs(rhs[column * n + i]))


@wp.func
def residual_and_scale(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    offset: wp.int32,
    i: wp.int32,
) -> tuple[wp.float64, wp.float64]:
    # Row ``i`` of ``b - A x`` in ``float64`` and its Oettli-Prager scale ``(|A| |x| + |b|)_i``,
    # for the column starting at ``offset``: the componentwise backward error is their ratio.
    b = rhs[offset + i]
    total = b
    scale = wp.abs(b)
    for e in range(offsets[i], offsets[i + 1]):
        term = values[e] * solution[offset + columns[e]]
        total -= term
        scale += wp.abs(term)
    return total, scale


@wp.kernel
def residual_test(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    rhs_scale: wp.array[wp.float64],
    dropped: wp.array[wp.int32],
    sign: wp.float64,
    n: wp.int32,
    tolerance: wp.float64,
    componentwise: wp.int32,
    state: wp.array[wp.int32],
    out_residual: wp.array[wp.float64],
) -> None:
    # ``rhs - A x`` in ``float64``, and whether it is still above ``tolerance``: against each
    # row's own scale ``(|A| |x| + |b|)_i`` (componentwise, the Oettli-Prager backward error; for
    # a solution spanning many orders of magnitude) or against the larger of that and the column's
    # largest ``|b|`` (normwise; the row scale keeps a row the residual's own rounding dominates
    # from never passing). Stored times ``sign``: the factored operator's right-hand side.
    i, column = wp.tid()
    offset = column * n
    total, scale = residual_and_scale(offsets, columns, values, solution, rhs, offset, i)
    out_residual[offset + i] = sign * total
    if componentwise == 0:
        scale = wp.max(scale, rhs_scale[column])
    # A dropped row (a singular component's pin) holds whatever the right-hand side's own
    # inconsistency left there, which no refinement removes.
    if dropped[i] == 0 and wp.abs(total) > tolerance * scale:
        state[REFINE_PENDING] = 1


@wp.kernel
def refine_advance(cap: wp.int32, state: wp.array[wp.int32]) -> None:
    # Close a refinement round: continue while some row is above tolerance, up to ``cap`` solves.
    state[REFINE_CONDITION] = wp.where(
        state[REFINE_PENDING] != 0 and state[REFINE_ROUND] < cap, 1, 0
    )
    state[REFINE_PENDING] = 0
    state[REFINE_ROUND] = state[REFINE_ROUND] + 1


@wp.kernel
def project_null_space(
    component_offsets: wp.array[wp.int32],
    component_rows: wp.array[wp.int32],
    component_label: wp.array[wp.int32],
    singular: wp.array[wp.int32],
    initial: wp.array[wp.float64],
    n: wp.int32,
    solution: wp.array[wp.float64],
) -> None:
    # A singular component's constant offset: the solve held one row at zero, so shift the
    # component to the initial guess's mean, the answer conjugate gradient returns from it.
    c, lane = wp.tid()
    n_components = component_offsets.shape[0] - 1
    component = c % n_components
    column = c // n_components
    if singular[component_label[component]] == 0:
        return
    offset = column * n
    width = wp.block_dim()
    start = component_offsets[component]
    count = component_offsets[component + 1] - start
    local = wp.float64(0.0)
    k = lane
    while k < count:
        r = component_rows[start + k]
        local += solution[offset + r] - initial[offset + r]
        k += width
    shift = block_sum(local) / wp.float64(count)
    block_barrier()
    k = lane
    while k < count:
        r = component_rows[start + k]
        solution[offset + r] = solution[offset + r] - shift
        k += width


@wp.kernel
def backward_error_exceeds(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    n: wp.int32,
    tolerance: wp.float64,
    residual: wp.array[wp.float64],
    threshold: wp.array[wp.float64],
    out_flag: wp.array[wp.int32],
) -> None:
    # The componentwise backward error of a solution (Oettli and Prager): ``|b - A x|_i`` against
    # ``(|A| |x| + |b|)_i``. Above ``tolerance`` anywhere, the solution is not the system's to
    # within its own entries' size, whatever its residual norm says. Given a conjugate-gradient
    # result's per-column ``residual`` and ``threshold`` (or ``None``), row 0 of each column also
    # flags a column that did not converge, so one four-byte read answers both questions.
    i, column = wp.tid()
    offset = column * n
    total, scale = residual_and_scale(offsets, columns, values, solution, rhs, offset, i)
    if wp.abs(total) > tolerance * scale:
        out_flag[0] = 1
    if i == 0 and residual.shape[0] > 0:
        if residual[column] > threshold[column]:
            out_flag[0] = 1


# --------------------------------------------------------------------------------------
# Symbolic analysis: nested dissection
# --------------------------------------------------------------------------------------
#
# ``ordito.cholesky`` bisects every subset of one depth at once. The live vertices sit in
# ``order``, grouped by subset (a *segment*, numbered within its depth); per depth the device
# forms each segment's principal axis, sorts the segment along it, marks the pattern edges that
# cross the median and splits off the separator, while the host keeps the tree's bookkeeping
# (node ids, sizes, intervals of the elimination order), one small array per segment.

# Positions of ``order`` one ``dissection_partials`` thread sums.
DISSECTION_CHUNK = wp.constant(256)
moment_vector = wp.types.vector(length=9, dtype=wp.float64)


@wp.func
def shifted_point(
    coordinates: wp.array[wp.vec3], vertex: wp.int32, reference: wp.vec3d
) -> wp.vec3d:
    p = coordinates[vertex]
    return wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2])) - reference


@wp.func
def segment_reference(
    order: wp.array[wp.int32], coordinates: wp.array[wp.vec3], start: wp.int32
) -> wp.vec3d:
    # The point the segment's moments are taken about: its first vertex, so the second moments
    # carry the segment's own extent rather than its distance from the origin.
    p = coordinates[order[start]]
    return wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))


@wp.kernel
def dissection_partials(
    order: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    segment_start: wp.array[wp.int32],
    task_segment: wp.array[wp.int32],
    task_begin: wp.array[wp.int32],
    task_end: wp.array[wp.int32],
    out_partials: wp.array[moment_vector],
) -> None:
    # One chunk of one segment: the first and second moments of its points about the segment's
    # reference point, summed serially (the segment sums them in task order, so the axis is the
    # same on every run).
    t = wp.int32(wp.tid())
    reference = segment_reference(order, coordinates, segment_start[task_segment[t]])
    m = moment_vector()
    for p in range(task_begin[t], task_end[t]):
        d = shifted_point(coordinates, order[p], reference)
        m[0] += d[0]
        m[1] += d[1]
        m[2] += d[2]
        m[3] += d[0] * d[0]
        m[4] += d[0] * d[1]
        m[5] += d[0] * d[2]
        m[6] += d[1] * d[1]
        m[7] += d[1] * d[2]
        m[8] += d[2] * d[2]
    out_partials[t] = m


@wp.kernel
def dissection_axes(
    order: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    segment_start: wp.array[wp.int32],
    task_offsets: wp.array[wp.int32],
    partials: wp.array[moment_vector],
    out_centre: wp.array[wp.vec3d],
    out_axis: wp.array[wp.vec3d],
) -> None:
    # A segment's centroid and the principal axis of its covariance: the eigenvector of the
    # largest eigenvalue.
    s = wp.int32(wp.tid())
    m = moment_vector()
    for t in range(task_offsets[s], task_offsets[s + 1]):
        m += partials[t]
    count = wp.float64(segment_start[s + 1] - segment_start[s])
    mean = wp.vec3d(m[0], m[1], m[2]) / count
    covariance = wp.mat33d(
        m[3], m[4], m[5],
        m[4], m[6], m[7],
        m[5], m[7], m[8],
    ) / count - wp.outer(mean, mean)  # fmt: skip
    vectors, values = wp.eig3(covariance)
    best = wp.int32(0)
    for q in range(1, 3):
        if values[q] > values[best]:
            best = wp.int32(q)
    out_centre[s] = segment_reference(order, coordinates, segment_start[s]) + mean
    out_axis[s] = wp.vec3d(vectors[0, best], vectors[1, best], vectors[2, best])


@wp.kernel
def dissection_keys(
    order: wp.array[wp.int32],
    order_segment: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    centre: wp.array[wp.vec3d],
    axis: wp.array[wp.vec3d],
    out_keys: wp.array[wp.uint64],
    out_vertices: wp.array[wp.int32],
) -> None:
    # The sort key of a live vertex: its segment, then its projection on the segment's axis.
    p = wp.int32(wp.tid())
    s = order_segment[p]
    v = order[p]
    projection = wp.dot(shifted_point(coordinates, v, centre[s]), axis[s])
    bits = ordered_float_bits(wp.float32(projection))
    out_keys[p] = (wp.uint64(s) << wp.uint64(32)) | wp.uint64(bits)
    out_vertices[p] = v


@wp.func
def sorted_segment(keys: wp.array[wp.uint64], p: wp.int32) -> wp.int32:
    return wp.int32(keys[p] >> wp.uint64(32))


@wp.kernel
def dissection_sides(
    keys: wp.array[wp.uint64],
    vertices: wp.array[wp.int32],
    segment_start: wp.array[wp.int32],
    base: wp.int32,
    out_tag: wp.array[wp.int32],
) -> None:
    # Each sorted vertex's side of its segment's median, as ``base + 2 segment + upper``: tags of
    # earlier depths are all below ``base``, so the array needs no reset between depths.
    p = wp.int32(wp.tid())
    s = sorted_segment(keys, p)
    start = segment_start[s]
    count = segment_start[s + 1] - start
    upper = wp.where(p - start >= count // 2, 1, 0)
    out_tag[vertices[p]] = base + 2 * s + upper


@wp.kernel
def dissection_crossings(
    keys: wp.array[wp.uint64],
    vertices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    tag: wp.array[wp.int32],
    stamp: wp.int32,
    out_lower_mark: wp.array[wp.int32],
    out_upper_mark: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The pattern edges from a segment's lower half to its upper half: both endpoints marked with
    # this depth's ``stamp``, and each segment's count of marked endpoints per side (an upper
    # endpoint counted by the first thread to mark it).
    p = wp.int32(wp.tid())
    u = vertices[p]
    t = tag[u]
    s = sorted_segment(keys, p)
    if (t & 1) != 0:
        return
    crossing = wp.int32(0)
    for e in range(offsets[u], offsets[u + 1]):
        v = columns[e]
        if tag[v] == t + 1:
            crossing = 1
            if wp.atomic_max(out_upper_mark, v, stamp) < stamp:
                wp.atomic_add(out_counts, 2 * s + 1, 1)
    if crossing != 0:
        out_lower_mark[u] = stamp
        wp.atomic_add(out_counts, 2 * s, 1)


@wp.kernel
def dissection_relabel(
    keys: wp.array[wp.uint64],
    vertices: wp.array[wp.int32],
    segment_start: wp.array[wp.int32],
    separator_side: wp.array[wp.int32],
    separator_node: wp.array[wp.int32],
    child_next: wp.array[wp.int32],
    child_leaf: wp.array[wp.int32],
    lower_mark: wp.array[wp.int32],
    upper_mark: wp.array[wp.int32],
    stamp: wp.int32,
    out_node: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
    out_next_segment: wp.array[wp.int32],
) -> None:
    # A sorted vertex leaves the live set as its segment's separator (the side with fewer marked
    # endpoints) or as a leaf's row, or stays, in the next depth's segment of its half. The halves
    # of a segment are consecutive in the sort and the host numbers the next segments in child
    # order, so the survivors, compacted in place, are grouped by next segment.
    p = wp.int32(wp.tid())
    s = sorted_segment(keys, p)
    v = vertices[p]
    start = segment_start[s]
    upper = wp.where(p - start >= (segment_start[s + 1] - start) // 2, 1, 0)
    side = separator_side[s]
    separator = (side == 1 and upper == 0 and lower_mark[v] == stamp) or (
        side == 2 and upper == 1 and upper_mark[v] == stamp
    )
    keep = wp.int32(0)
    next_segment = wp.int32(-1)
    if separator:
        out_node[v] = separator_node[s]
    else:
        child = 2 * s + upper
        leaf = child_leaf[child]
        if leaf >= 0:
            out_node[v] = leaf
        else:
            keep = 1
            next_segment = child_next[child]
    out_keep[p] = keep
    out_next_segment[p] = next_segment


@wp.kernel
def dissection_compact(
    vertices: wp.array[wp.int32],
    kept_through: wp.array[wp.int32],
    next_segment: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
    out_order_segment: wp.array[wp.int32],
) -> None:
    # The survivors in sorted order: ``kept_through`` is the keep flags' inclusive scan, done in
    # place, from which each position recovers its own flag.
    p = wp.int32(wp.tid())
    q, kept = scanned_count(kept_through, p)
    if kept != 0:
        out_order[q] = vertices[p]
        out_order_segment[q] = next_segment[p]


# --------------------------------------------------------------------------------------
# Symbolic analysis: elimination order and row structure
# --------------------------------------------------------------------------------------


@wp.kernel
def node_order_keys(
    node: wp.array[wp.int32],
    node_start: wp.array[wp.int32],
    out_keys: wp.array[wp.int32],
    out_vertices: wp.array[wp.int32],
) -> None:
    # Each vertex keyed by its node's first position: a stable sort of the vertices in index order
    # is the elimination order, every node's rows in ascending vertex index.
    v = wp.int32(wp.tid())
    out_keys[v] = node_start[node[v]]
    out_vertices[v] = v


@wp.kernel
def scatter_positions(
    perm: wp.array[wp.int32],
    node: wp.array[wp.int32],
    node_supernode: wp.array[wp.int32],
    out_pos: wp.array[wp.int32],
    out_owner: wp.array[wp.int32],
) -> None:
    # The inverse permutation, and each position's supernode.
    i = wp.int32(wp.tid())
    v = perm[i]
    out_pos[v] = i
    out_owner[i] = node_supernode[node[v]]


@wp.func
def owns_entry(
    pos: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    column_end: wp.array[wp.int32],
    row: wp.int32,
    column: wp.int32,
) -> wp.int32:
    # The supernode an operator entry adds to the row structure of (``-1`` for none): the
    # supernode of its column, when its row lies past that supernode's columns.
    r = pos[row]
    k = owner[pos[column]]
    return wp.where(r >= column_end[k], k, -1)


@wp.kernel
def count_own_entries(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    column_end: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    count = wp.int32(0)
    for e in range(offsets[i], offsets[i + 1]):
        if owns_entry(pos, owner, column_end, i, columns[e]) >= 0:
            count += 1
    out_counts[i] = count


@wp.kernel
def write_own_entries(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    column_end: wp.array[wp.int32],
    height: wp.array[wp.int32],
    n: wp.int32,
    shift: wp.int32,
    ends: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_values: wp.array[wp.int32],
) -> None:
    # Each own entry as ``(height, supernode, row)`` packed for one sort: the entries of one
    # height become one contiguous run, each supernode's rows ascending within it.
    i = wp.int32(wp.tid())
    q = wp.where(i > 0, ends[wp.max(i - 1, 0)], 0)
    r = wp.uint64(pos[i])
    for e in range(offsets[i], offsets[i + 1]):
        k = owns_entry(pos, owner, column_end, i, columns[e])
        if k >= 0:
            out_keys[q] = (wp.uint64(height[k]) << wp.uint64(shift)) | (
                wp.uint64(k) * wp.uint64(n) + r
            )
            out_values[q] = q
            q += 1


@wp.kernel
def height_starts(
    keys: wp.array[wp.uint64], shift: wp.int32, out_starts: wp.array[wp.int32]
) -> None:
    # The first sorted position of each height (``out_starts`` is filled with the count first, so
    # a height with no entry starts where the next one does).
    p = wp.int32(wp.tid())
    h = wp.int32(keys[p] >> wp.uint64(shift))
    previous = wp.int32(-1)
    if p > 0:
        previous = wp.int32(keys[p - 1] >> wp.uint64(shift))
    for hh in range(previous + 1, h + 1):
        out_starts[hh] = p


# ``structure_cursors`` slots: pending rows kept for a later height, rows passed up, and the
# height's distinct rows.
CURSOR_KEPT = wp.constant(0)
CURSOR_PASSED = wp.constant(1)
CURSOR_UNIQUE = wp.constant(2)


@wp.kernel
def gather_candidates(
    own_keys: wp.array[wp.uint64],
    own_begin: wp.int32,
    own_count: wp.int32,
    key_mask: wp.uint64,
    pending: wp.array[wp.uint64],
    height: wp.array[wp.int32],
    n: wp.int32,
    level: wp.int32,
    cursors: wp.array[wp.int32],
    out_candidates: wp.array[wp.uint64],
    out_values: wp.array[wp.int32],
    out_kept: wp.array[wp.uint64],
) -> None:
    # The row candidates of one height's supernodes: their own entries, then the rows their
    # descendants passed up (``pending``). A pending row another height takes is moved to
    # ``out_kept`` and its candidate slot holds ``key_mask``, above every real key (rows below
    # ``2^key_bits - 1``), so the sort puts those slots last and the host need not count them
    # first. The candidates are sorted and deduplicated next, so the cursor's arrival order does
    # not matter.
    t = wp.int32(wp.tid())
    out_values[t] = t
    if t < own_count:
        out_candidates[t] = own_keys[own_begin + t] & key_mask
        return
    key = pending[t - own_count]
    if height[wp.int32(key // wp.uint64(n))] == level:
        out_candidates[t] = key
    else:
        out_candidates[t] = key_mask
        out_kept[wp.atomic_add(cursors, CURSOR_KEPT, 1)] = key


@wp.kernel
def emit_rows(
    candidates: wp.array[wp.uint64],
    run_ends: wp.array[wp.int32],
    key_mask: wp.uint64,
    n: wp.int32,
    level_base: wp.int32,
    parent: wp.array[wp.int32],
    column_end: wp.array[wp.int32],
    cursors: wp.array[wp.int32],
    out_unique: wp.array[wp.uint64],
    out_row_start: wp.array[wp.int32],
    out_nrow: wp.array[wp.int32],
    out_pending: wp.array[wp.uint64],
) -> None:
    # One height's row structure is final, from its sorted candidates and the in-place inclusive
    # scan of their run starts: each distinct row is stored, each supernode's run recorded, and
    # every row past the parent's columns passed up to the parent, after the rows
    # ``gather_candidates`` kept. The last thread (``run_ends`` is exactly the height's capacity;
    # ``candidates`` is twice it, the sort's scratch) publishes the distinct count: the scan's
    # total, less the run of placeholder keys if there is one.
    p = wp.int32(wp.tid())
    key = candidates[p]
    if p == run_ends.shape[0] - 1:
        cursors[CURSOR_UNIQUE] = run_ends[p] - wp.where(key == key_mask, 1, 0)
    if key == key_mask:
        return
    i, starts = scanned_count(run_ends, p)
    if starts == 0:
        return
    out_unique[i] = key
    k = wp.int32(key // wp.uint64(n))
    row = wp.int32(key % wp.uint64(n))
    # ``candidates[p - 1]`` is the previous distinct row: placeholders sort last.
    if p == 0 or wp.int32(candidates[wp.max(p - 1, 0)] // wp.uint64(n)) != k:
        out_row_start[k] = level_base + i
    wp.atomic_add(out_nrow, k, 1)
    q = parent[k]
    if q >= 0 and row >= column_end[q]:
        slot = cursors[CURSOR_KEPT] + wp.atomic_add(cursors, CURSOR_PASSED, 1)
        out_pending[slot] = wp.uint64(q) * wp.uint64(n) + wp.uint64(row)


@wp.kernel
def split_rows(
    unique: wp.array[wp.uint64],
    n: wp.int32,
    base: wp.int32,
    out_rows: wp.array[wp.int32],
    out_row_node: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    key = unique[i]
    out_rows[base + i] = wp.int32(key % wp.uint64(n))
    out_row_node[base + i] = wp.int32(key // wp.uint64(n))


# --------------------------------------------------------------------------------------
# Symbolic analysis: factorization and solve maps
# --------------------------------------------------------------------------------------


@wp.func
def front_local(
    k: wp.int32,
    p: wp.int32,
    c0: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
) -> wp.int32:
    # Position ``p``'s index in supernode ``k``'s front: a column, or a row of its (ascending) row
    # structure.
    first = c0[k]
    width = ncol[k]
    if p < first + width:
        return p - first
    low = row_offsets[k]
    high = low + front_size[k] - width
    while low < high:
        mid = (low + high) // 2
        if rows[mid] < p:
            low = mid + 1
        else:
            high = mid
    return width + low - row_offsets[k]


@wp.kernel
def count_lower_entries(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    count = wp.int32(0)
    r = pos[i]
    for e in range(offsets[i], offsets[i + 1]):
        if r >= pos[columns[e]]:
            count += 1
    out_counts[i] = count


@wp.kernel
def write_lower_entries(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    row_offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    out_source: wp.array[wp.int32],
    out_target: wp.array[wp.int64],
    out_row: wp.array[wp.int32],
    out_column: wp.array[wp.int32],
) -> None:
    # The lower triangle of the permuted operator, in stored order: each entry's front slot.
    i = wp.int32(wp.tid())
    q = wp.where(i > 0, ends[wp.max(i - 1, 0)], 0)
    r = pos[i]
    for e in range(offsets[i], offsets[i + 1]):
        c = pos[columns[e]]
        if r >= c:
            k = owner[c]
            local = front_local(k, r, c0, ncol, front_size, row_offsets, rows)
            out_source[q] = e
            out_target[q] = front_index(front_offset[k], wp.int64(front_size[k]), local, c - c0[k])
            out_row[q] = r
            out_column[q] = c
            q += 1


@wp.kernel
def parent_locals(
    rows: wp.array[wp.int32],
    row_node: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    out_parent_local: wp.array[wp.int32],
) -> None:
    # Each row-structure entry's index in the parent's front: where ``extend_add`` adds it.
    e = wp.int32(wp.tid())
    p = parent[row_node[e]]
    if p < 0:
        out_parent_local[e] = 0
        return
    out_parent_local[e] = front_local(p, rows[e], c0, ncol, front_size, row_offsets, rows)


@wp.func
def level_item(offsets: wp.array[wp.int32], t: wp.int32) -> wp.int32:
    # The supernode (index into the level-major order) whose run of items holds item
    # ``t < offsets[-1]``: the last ``j`` with ``offsets[j] <= t``, so empty runs are skipped.
    return binary_search_index(offsets, t) - 1


@wp.func
def forward_chunk_start(c: wp.int32, size: wp.int32) -> wp.int32:
    # Tasks before chunk ``c`` of a supernode's forward product: chunk ``c'`` keeps its rows from
    # ``CHUNK c'`` (the diagonal block's rows above it are zero in ``L11^-1``).
    return c * size - (CHUNK // 2) * c * (c - 1)


@wp.kernel
def forward_task_map(
    level_node: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    block_offset: wp.array[wp.int64],
    forward_base: wp.array[wp.int32],
    out_block: wp.array[wp.int64],
    out_stride: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
    out_source: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
) -> None:
    # Forward tasks of every level, level-major: supernode by supernode, chunk-major, each chunk's
    # rows from ``CHUNK`` times the chunk down.
    t = wp.int32(wp.tid())
    j = level_item(offsets, t)
    k = level_node[j]
    local = t - offsets[j]
    size = front_size[k]
    width = ncol[k]
    low = wp.int32(0)
    high = (width + CHUNK - 1) // CHUNK
    while high - low > 1:
        mid = (low + high) // 2
        if forward_chunk_start(mid, size) <= local:
            low = mid
        else:
            high = mid
    c = low
    row = CHUNK * c + local - forward_chunk_start(c, size)
    out_block[t] = block_offset[k] + wp.int64(c * CHUNK) * wp.int64(size) + wp.int64(row)
    out_stride[t] = size
    out_count[t] = wp.min(CHUNK, width - c * CHUNK)
    out_source[t] = c0[k] + c * CHUNK
    out_slot[t] = forward_base[k] + c * size + row


@wp.kernel
def forward_row_map(
    level_node: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    forward_base: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
    out_stride: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
    out_target: wp.array[wp.int32],
) -> None:
    # Every solve-block row of every level: where its chunks sit and where their sum goes (a
    # column's forward entry, or ``-1 - pair`` for a lower row's contribution).
    t = wp.int32(wp.tid())
    j = level_item(offsets, t)
    k = level_node[j]
    r = t - offsets[j]
    size = front_size[k]
    width = ncol[k]
    chunks = (width + CHUNK - 1) // CHUNK
    out_slot[t] = forward_base[k] + r
    out_stride[t] = size
    if r >= width:
        out_count[t] = chunks
        out_target[t] = -1 - (row_offsets[k] + r - width)
    else:
        out_count[t] = r // CHUNK + 1
        out_target[t] = c0[k] + r


@wp.func
def backward_chunk_start(c: wp.int32, width: wp.int32) -> wp.int32:
    # Tasks before chunk ``c`` of a supernode's backward product: chunk ``c'`` keeps the columns
    # below ``CHUNK (c' + 1)``.
    full = width // CHUNK
    if c <= full:
        return (CHUNK // 2) * c * (c + 1)
    return (CHUNK // 2) * full * (full + 1) + (c - full) * width


@wp.kernel
def backward_task_map(
    level_node: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    block_offset: wp.array[wp.int64],
    row_offsets: wp.array[wp.int32],
    backward_base: wp.array[wp.int32],
    out_block: wp.array[wp.int64],
    out_first: wp.array[wp.int32],
    out_ncol: wp.array[wp.int32],
    out_size: wp.array[wp.int32],
    out_start: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
) -> None:
    t = wp.int32(wp.tid())
    j = level_item(offsets, t)
    k = level_node[j]
    local = t - offsets[j]
    size = front_size[k]
    width = ncol[k]
    low = wp.int32(0)
    high = (size + CHUNK - 1) // CHUNK
    while high - low > 1:
        mid = (low + high) // 2
        if backward_chunk_start(mid, width) <= local:
            low = mid
        else:
            high = mid
    c = low
    column = local - backward_chunk_start(c, width)
    out_block[t] = block_offset[k] + wp.int64(column) * wp.int64(size)
    out_first[t] = c * CHUNK
    out_ncol[t] = width
    out_size[t] = size
    out_start[t] = c0[k]
    out_rows[t] = row_offsets[k]
    out_slot[t] = backward_base[k] + c * width + column


@wp.kernel
def backward_row_map(
    level_node: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    backward_base: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
    out_stride: wp.array[wp.int32],
    out_first: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
    out_target: wp.array[wp.int32],
) -> None:
    # Every column of every level, level-major: its backward chunks and its position (also the
    # forward solve's gather rows, in the same order).
    t = wp.int32(wp.tid())
    j = level_item(offsets, t)
    k = level_node[j]
    i = t - offsets[j]
    width = ncol[k]
    out_slot[t] = backward_base[k] + i
    out_stride[t] = width
    out_first[t] = i // CHUNK
    out_count[t] = (front_size[k] + CHUNK - 1) // CHUNK
    out_target[t] = c0[k] + i


@wp.kernel
def contribution_keys(
    rows: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    column_base: wp.array[wp.int32],
    out_keys: wp.array[wp.int32],
    out_pairs: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Each row-structure entry keyed by its row's level-major column index, so one stable sort
    # lists every column's contributions, level by level, in pair order.
    e = wp.int32(wp.tid())
    v = rows[e]
    k = owner[v]
    key = column_base[k] + v - c0[k]
    out_keys[e] = key
    out_pairs[e] = e
    wp.atomic_add(out_counts, key, 1)


# --------------------------------------------------------------------------------------
# Symbolic analysis: components
# --------------------------------------------------------------------------------------


@wp.kernel
def component_tables(
    sorted_labels: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    vertices: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    out_label: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_last: wp.array[wp.int32],
) -> None:
    # Per component (labels ascending): its label, where its vertices start in label order, and
    # its last-eliminated position.
    p = wp.int32(wp.tid())
    c = ends[p] - 1
    if flags[p] != 0:
        out_label[c] = sorted_labels[p]
        out_offsets[c] = p
    wp.atomic_max(out_last, c, pos[vertices[p]])


@wp.kernel
def component_diagonals(
    last: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    c0: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    out_diagonal: wp.array[wp.int64],
) -> None:
    c = wp.int32(wp.tid())
    i = last[c]
    k = owner[i]
    local = i - c0[k]
    out_diagonal[c] = front_index(front_offset[k], wp.int64(front_size[k]), local, local)


# --------------------------------------------------------------------------------------
# Symbolic analysis: landmark coordinates (an ordering from the pattern alone)
# --------------------------------------------------------------------------------------

# ``breadth_first_level`` launches per recorded round of the breadth-first loop.
BFS_LEVELS_PER_ROUND = wp.constant(16)
# ``bfs_state`` slots: loop condition, the round's first level, the last level that reached a row.
BFS_CONDITION = wp.constant(0)
BFS_LEVEL = wp.constant(1)
BFS_REACHED = wp.constant(2)


@wp.kernel
def component_sizes(
    labels: wp.array[wp.int32], out_size: wp.array[wp.int32], out_first: wp.array[wp.int32]
) -> None:
    # Per label slot: the component's row count and its lowest row.
    v = wp.int32(wp.tid())
    wp.atomic_add(out_size, labels[v], 1)
    wp.atomic_min(out_first, labels[v], v)


@wp.kernel
def seed_breadth_first(
    labels: wp.array[wp.int32],
    seeds: wp.array[wp.int32],
    out_distance: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Distance zero at each component's seed (``-1`` for a component without one), and the
    # breadth-first loop armed at level one.
    v = wp.int32(wp.tid())
    out_distance[v] = wp.where(seeds[labels[v]] == v, 0, -1)
    if v == 0:
        out_state[BFS_CONDITION] = 1
        out_state[BFS_LEVEL] = 1
        out_state[BFS_REACHED] = 0


@wp.kernel
def breadth_first_level(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    step: wp.int32,
    state: wp.array[wp.int32],
    distance: wp.array[wp.int32],
) -> None:
    # One level of every component's search at once: an unreached row next to a row of the
    # previous level is at this one. A row reached in this launch reads as neither, so the levels
    # stay exact whatever the thread order.
    v = wp.int32(wp.tid())
    if distance[v] >= 0:
        return
    level = state[BFS_LEVEL] + step
    for e in range(offsets[v], offsets[v + 1]):
        if distance[columns[e]] == level - 1:
            distance[v] = level
            state[BFS_REACHED] = level
            return


@wp.kernel
def breadth_first_advance(state: wp.array[wp.int32]) -> None:
    # Continue while the round's last level still reached a row.
    last = state[BFS_LEVEL] + BFS_LEVELS_PER_ROUND - 1
    state[BFS_CONDITION] = wp.where(state[BFS_REACHED] == last, 1, 0)
    state[BFS_LEVEL] = last + 1


@wp.kernel
def farthest_rows(
    labels: wp.array[wp.int32],
    first: wp.array[wp.float32],
    second: wp.array[wp.float32],
    out_key: wp.array[wp.uint64],
) -> None:
    # Per component, the row farthest by ``min(first, second)``, the lowest such row on a tie:
    # the larger distance in the high word, the complemented row in the low one.
    v = wp.int32(wp.tid())
    value = wp.min(first[v], second[v])
    if value < wp.float32(0.0):
        return
    key = (wp.uint64(value) << wp.uint64(32)) | wp.uint64(wp.uint32(0xFFFFFFFF) - wp.uint32(v))
    wp.atomic_max(out_key, labels[v], key)


@wp.kernel
def farthest_seeds(
    key: wp.array[wp.uint64],
    size: wp.array[wp.int32],
    first: wp.array[wp.int32],
    min_size: wp.int32,
    use_first: wp.int32,
    out_seed: wp.array[wp.int32],
) -> None:
    # Each label slot's next seed: its lowest row on the first search, else its farthest row;
    # ``-1`` for a component of at most ``min_size`` rows (it keeps zero coordinates).
    c = wp.int32(wp.tid())
    seed = wp.int32(-1)
    if size[c] > min_size:
        if use_first != 0:
            seed = first[c]
        else:
            seed = wp.int32(wp.uint32(0xFFFFFFFF) - wp.uint32(key[c] & wp.uint64(0xFFFFFFFF)))
    out_seed[c] = seed


@wp.kernel
def store_distance(distance: wp.array[wp.int32], out_coordinate: wp.array[wp.float32]) -> None:
    # One landmark's distances as a coordinate (zero off the searched components).
    v = wp.int32(wp.tid())
    out_coordinate[v] = wp.float32(wp.max(distance[v], 0))
