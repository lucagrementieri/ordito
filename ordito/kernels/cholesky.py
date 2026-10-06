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
solve, ``scatter_correction``) until the residual passes its test.

Every kernel that cooperates strides its lanes by ``wp.block_dim()``, so the CPU device's
one-lane blocks run the same code.
"""

import warp as wp

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
) -> None:
    # A component is singular when every one of its rows sums to zero (a Laplacian's constant null
    # space); ``out_singular`` is seeded with ones and any row that does not sum to zero clears its
    # component. An empty row (an unreferenced vertex) sums to zero: a component of one.
    i = wp.int32(wp.tid())
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
    out_pinned: wp.array[wp.int32],
    out_dropped: wp.array[wp.int32],
) -> None:
    # One pin per singular component, at its last-eliminated row: that row's equation is dropped
    # and its unknown held at zero, which leaves a consistent system solvable. ``out_pinned`` is
    # indexed in elimination order, ``out_dropped`` in the operator's own.
    c = wp.int32(wp.tid())
    pin = singular[component_label[c]]
    out_pinned[component_last[c]] = pin
    out_dropped[perm[component_last[c]]] = pin


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
    out_fronts: wp.array[wp.float64],
) -> None:
    # One stored lower entry into its front. A pinned row or column becomes the identity's.
    e = wp.int32(wp.tid())
    r = entry_row[e]
    c = entry_column[e]
    value = values[entry_source[e]]
    if pinned[r] != 0 or pinned[c] != 0:
        value = wp.where(r == c, wp.float64(1.0), wp.float64(0.0))
    out_fronts[entry_target[e]] = value


@wp.func
def front_index(base: wp.int64, size: wp.int64, row: wp.int32, column: wp.int32) -> wp.int64:
    return base + wp.int64(row) * size + wp.int64(column)


@wp.kernel
def set_pinned_diagonals(
    component_label: wp.array[wp.int32],
    component_diagonal: wp.array[wp.int64],
    singular: wp.array[wp.int32],
    out_fronts: wp.array[wp.float64],
) -> None:
    # A pinned row's diagonal is the identity's even where the operator stores none (an
    # unreferenced vertex's empty row).
    c = wp.int32(wp.tid())
    if singular[component_label[c]] != 0:
        out_fronts[component_diagonal[c]] = wp.float64(1.0)


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
def invert_column(
    base: wp.int64,
    size: wp.int32,
    column: wp.int64,
    c: wp.int32,
    first: wp.int32,
    last: wp.int32,
    fronts: wp.array[wp.float64],
    blocks: wp.array[wp.float64],
) -> None:
    # Forward substitution of one solve-block column's rows ``[first, last)`` -- the panel's own
    # triangle -- the panel's entries held in registers.
    m64 = wp.int64(size)
    values = panel_vector()
    for q1 in range(PANEL):
        j = first + wp.int32(q1)
        if j < last and j >= c:
            total = blocks[column + wp.int64(j)]
            for q2 in range(PANEL):
                if wp.int32(q2) < wp.int32(q1):
                    total -= fronts[front_index(base, m64, j, first + wp.int32(q2))] * values[q2]
            values[q1] = total / fronts[front_index(base, m64, j, j)]
            blocks[column + wp.int64(j)] = values[q1]


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
    block_offset: wp.array[wp.int64],
    panel: wp.int32,
    fronts: wp.array[wp.float64],
    status: wp.array[wp.int32],
) -> None:
    # A large front's panel ``panel``: the Cholesky of its ``PANEL``-square diagonal block, one
    # block per front. ``panel_rows`` then forms the panel's rows below it and its rows of the solve
    # block, and ``update_panel`` spreads the panel's rank-``PANEL`` update.
    b, lane = wp.tid()
    k = level[b]
    ns = ncol[k]
    first = panel * PANEL
    if first >= ns:
        return
    last = wp.min(first + PANEL, ns)
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
    # ``rows_below``, one per solve-block column (the panel rows' own triangle, ``invert_column``).
    b, t = wp.tid()
    k = level[b]
    ns = ncol[k]
    first = panel * PANEL
    if first >= ns:
        return
    last = wp.min(first + PANEL, ns)
    size = front_size[k]
    base = front_offset[k]
    m64 = wp.int64(size)
    if t >= rows_below:
        c = t - rows_below
        if c < last:
            invert_column(base, size, block_offset[k] + wp.int64(c) * m64, c, first, last,
                          fronts, blocks)  # fmt: skip
        return
    row = last + t
    if row >= size:
        return
    values = panel_vector()
    for q1 in range(PANEL):
        j = first + wp.int32(q1)
        if j < last:
            total = fronts[front_index(base, m64, row, j)]
            for q2 in range(PANEL):
                if wp.int32(q2) < wp.int32(q1):
                    total -= values[q2] * fronts[front_index(base, m64, j, first + wp.int32(q2))]
            values[q1] = total / fronts[front_index(base, m64, j, j)]
            fronts[front_index(base, m64, row, j)] = values[q1]


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
    ns = ncol[k]
    first = panel * PANEL
    if first >= ns:
        return
    last = wp.min(first + PANEL, ns)
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
    base = wp.int64(column) * wp.int64(n_parts)
    slot = row_slot[t]
    stride = row_stride[t]
    value = wp.float64(0.0)
    for q in range(row_count[t]):
        value += parts[base + wp.int64(slot + q * stride)]
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
    n: wp.int32,
    n_parts: wp.int32,
    out_solution: wp.array[wp.float64],
) -> None:
    # A level's solution entries: their chunks summed in order.
    t, column = wp.tid()
    base = wp.int64(column) * wp.int64(n_parts)
    slot = row_slot[t]
    stride = row_stride[t]
    value = wp.float64(0.0)
    for q in range(row_first[t], row_count[t]):
        value += parts[base + wp.int64(slot + q * stride)]
    i = row_target[t]
    if pinned[i] != 0:
        value = wp.float64(0.0)
    out_solution[column * n + i] = value


# --------------------------------------------------------------------------------------
# Iterative refinement
# --------------------------------------------------------------------------------------


@wp.kernel
def refine_start(
    rhs: wp.array[wp.float64],
    n: wp.int32,
    state: wp.array[wp.int32],
    rhs_scale: wp.array[wp.float64],
) -> None:
    # Arm the refinement loop and record each column's largest right-hand-side entry.
    i, column = wp.tid()
    if i == 0 and column == 0:
        state[REFINE_CONDITION] = 0
        state[REFINE_PENDING] = 0
        state[REFINE_ROUND] = 0
    wp.atomic_max(rhs_scale, column, wp.abs(rhs[column * n + i]))


@wp.kernel
def residual_test(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    rhs_scale: wp.array[wp.float64],
    dropped: wp.array[wp.int32],
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
    # from never passing).
    i, column = wp.tid()
    offset = column * n
    b = rhs[offset + i]
    total = b
    scale = wp.abs(b)
    for e in range(offsets[i], offsets[i + 1]):
        term = values[e] * solution[offset + columns[e]]
        total -= term
        scale += wp.abs(term)
    out_residual[offset + i] = total
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
def scatter_correction(
    correction: wp.array[wp.float64], perm: wp.array[wp.int32], n: wp.int32,
    solution: wp.array[wp.float64],
) -> None:  # fmt: skip
    # Add a permuted correction to the solution.
    i, column = wp.tid()
    target = column * n + perm[i]
    solution[target] = solution[target] + correction[column * n + i]


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
    out_flag: wp.array[wp.int32],
) -> None:
    # The componentwise backward error of a solution (Oettli and Prager): ``|b - A x|_i`` against
    # ``(|A| |x| + |b|)_i``. Above ``tolerance`` anywhere, the solution is not the system's to
    # within its own entries' size, whatever its residual norm says.
    i, column = wp.tid()
    offset = column * n
    b = rhs[offset + i]
    total = b
    scale = wp.abs(b)
    for e in range(offsets[i], offsets[i + 1]):
        term = values[e] * solution[offset + columns[e]]
        total -= term
        scale += wp.abs(term)
    if wp.abs(total) > tolerance * scale:
        out_flag[0] = 1
