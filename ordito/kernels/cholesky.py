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

from ordito.kernels.array import (
    binary_search_index,
    binary_search_index_left,
    ordered_float_bits,
    scanned_count,
)
from ordito.kernels.reduce import block_barrier, block_sum, last_ticket, take_ticket

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
def mark_components_and_seed_blocks(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    labels: wp.array[wp.int32],
    tolerance: wp.float64,
    block_offset: wp.array[wp.int64],
    front_size: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    blocks: wp.array[wp.float64],
    out_singular: wp.array[wp.int32],
    out_pinned: wp.array[wp.int32],
    out_dropped: wp.array[wp.int32],
) -> None:
    # A component is singular when every one of its rows sums to zero (a Laplacian's constant null
    # space); ``out_singular`` is seeded with ones and any row that does not sum to zero clears its
    # component. An empty row (an unreferenced vertex) sums to zero: a component of one. Row ``i``
    # also clears its two pin flags, which ``pin_singular_components`` then sets.
    #
    # Every solve block starts as ``[I; 0]``: thread ``i`` also sets the diagonal of elimination
    # position ``i`` in the zeroed ``blocks`` arena.
    i = wp.int32(wp.tid())
    k = owner[i]
    local = wp.int64(i - c0[k])
    blocks[block_offset[k] + local * wp.int64(front_size[k]) + local] = wp.float64(1.0)
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
def leading_diagonal_sum(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    n_rows: wp.int32,
    out_sum: wp.array[wp.float64],
) -> None:
    # The diagonal of the first ``n_rows`` rows summed by one thread: its sign is the definite
    # operator's, which ``sparse_cholesky`` tries first when either sign is acceptable.
    total = wp.float64(0.0)
    for i in range(n_rows):
        for e in range(offsets[i], offsets[i + 1]):
            if columns[e] == i:
                total += values[e]
    out_sum[0] = total


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
    out_status: wp.array[wp.int32],
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
        # Some component is singular: the solves project its null space (``status[1]``).
        out_status[1] = 1


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


@wp.func
def unit_entry(i: wp.int32, j: wp.int32) -> wp.float64:
    return wp.where(i == j, wp.float64(1.0), wp.float64(0.0))


@wp.kernel
def factor_panel_tiled(
    level: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    panel: wp.int32,
    fronts: wp.array[wp.float64],
    status: wp.array[wp.int32],
    scratch: wp.array2d[wp.float64],
    out_inverse: wp.array2d[wp.float64],
) -> None:
    # ``factor_panel`` on CUDA: the ``PANEL``-square diagonal block staged in ``scratch`` (a tile a
    # front of the level, padded with the identity past the panel's width) and factored as one
    # shared-memory tile. Its Cholesky goes back to the front and its inverse to
    # ``out_inverse``, which ``panel_rows_inverse`` multiplies by instead of substituting, so
    # every row forms its entries independently. A pivot that is not positive flags ``status``
    # (the factorization is refused; the rest of the front is not used).
    b, lane = wp.tid()
    k = level[b]
    first, last = panel_span(ncol[k], panel)
    if last <= first:
        return
    base = front_offset[k]
    m64 = wp.int64(front_size[k])
    width = last - first
    row0 = b * PANEL
    for e in range(lane, PANEL * PANEL, wp.block_dim()):
        i = e // PANEL
        j = e % PANEL
        value = unit_entry(i, j)
        if i < width and j < width:
            # The lower triangle, mirrored: a front's strict upper one is not kept current.
            value = fronts[front_index(base, m64, first + wp.max(i, j), first + wp.min(i, j))]
        scratch[row0 + i, j] = value
    block_barrier()
    factor = wp.tile_cholesky(wp.tile_load(scratch, shape=(PANEL, PANEL), offset=(row0, 0)))
    identity = wp.tile_diag_add(
        wp.tile_zeros(shape=(PANEL, PANEL), dtype=wp.float64),
        wp.tile_ones(shape=(PANEL,), dtype=wp.float64),
    )
    wp.tile_store(out_inverse, wp.tile_lower_solve(factor, identity), offset=(row0, 0))
    wp.tile_store(scratch, factor, offset=(row0, 0))
    block_barrier()
    for e in range(lane, PANEL * PANEL, wp.block_dim()):
        i = e // PANEL
        j = e % PANEL
        if i < width and j <= i:
            entry = scratch[row0 + i, j]
            if i == j and not (entry > wp.float64(0.0)):
                status[0] = 1
            fronts[front_index(base, m64, first + i, first + j)] = entry


@wp.kernel
def panel_rows_inverse(
    level: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    block_offset: wp.array[wp.int64],
    panel: wp.int32,
    rows_below: wp.int32,
    inverse: wp.array[wp.float64],
    fronts: wp.array[wp.float64],
    blocks: wp.array[wp.float64],
) -> None:
    # ``panel_rows`` after ``factor_panel_tiled``: a row below the panel is ``A[i, panel]
    # L_pp^-T`` and a solve-block column's panel rows ``L_pp^-1 x``, both products with the
    # panel's inverse (its upper triangle zero). One lane an entry, ``PANEL`` rows a block of
    # ``PANEL * PANEL`` lanes (launched over ``(front, row group)``): every lane reads its row
    # and forms its dot, the block synchronizes, then each writes its entry back in place -- a
    # row's entries are read by all its lanes before any is overwritten.
    b, g, lane = wp.tid()
    k = level[b]
    first, last = panel_span(ncol[k], panel)
    if last <= first:
        return
    width = last - first
    size = front_size[k]
    base = front_offset[k]
    m64 = wp.int64(size)
    q1 = lane % PANEL
    t = g * PANEL + lane // PANEL
    target = blocks
    target_base = wp.int64(0)
    valid = q1 < width
    if t >= rows_below:
        c = t - rows_below
        valid = valid and c < last
        target_base = block_offset[k] + wp.int64(wp.min(c, last - 1)) * m64
    else:
        row = last + t
        valid = valid and row < size
        target = fronts
        target_base = front_index(base, m64, wp.min(row, size - 1), 0)
    total = wp.float64(0.0)
    if valid:
        tile_row = b * PANEL * PANEL + q1 * PANEL
        for q2 in range(PANEL):
            if wp.int32(q2) <= q1:
                total += inverse[tile_row + q2] * target[target_base + wp.int64(first + q2)]
    block_barrier()
    if valid:
        target[target_base + wp.int64(first + q1)] = total


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
    start: wp.int32,
    out_gathered: wp.array[wp.float64],
) -> None:
    # A level's rows of the forward right-hand side: the original entry plus every descendant's
    # contribution (their solve blocks' lower rows carry the sign), summed in a fixed order. The
    # level's entries of the level-major maps begin at ``start``, as in every solve kernel.
    t0, column = wp.tid()
    t = start + t0
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
    start: wp.int32,
    out_parts: wp.array[wp.float64],
) -> None:
    # One row of a solve block against ``CHUNK`` of its supernode's gathered entries.
    t0, column = wp.tid()
    t = start + t0
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
    start: wp.int32,
    out_forward: wp.array[wp.float64],
    out_pair_values: wp.array[wp.float64],
) -> None:
    # A level's solve-block rows: their chunks summed in order. A diagonal-block row is the
    # forward solution's entry; a lower row is a contribution to an ancestor (``-1 - pair``).
    t0, column = wp.tid()
    t = start + t0
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
    level_start: wp.int32,
    out_parts: wp.array[wp.float64],
) -> None:
    # ``CHUNK`` entries of one solve-block column, transposed, against its supernode's forward
    # entries and its ancestors' finished solution entries.
    t0, column = wp.tid()
    t = level_start + t0
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
    start: wp.int32,
    out_correction: wp.array[wp.float64],
) -> None:
    # A level's entries of the correction (in elimination order, which the next level's
    # ``backward_partial`` reads): their chunks summed in order. Each is also added to the refined
    # solution at its operator row: every row is finished by exactly one level, so this is the
    # whole correction's scatter, one launch fewer per refinement round.
    t0, column = wp.tid()
    t = start + t0
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
    componentwise: wp.int32,
    initial: wp.array[wp.float64],
    out_solution: wp.array[wp.float64],
) -> None:
    # Arm the refinement loop, record each column's largest right-hand-side entry (the normwise
    # test's scale; the componentwise one never reads it), and start the solution from the
    # caller's initial guess.
    i, column = wp.tid()
    out_solution[column * n + i] = initial[column * n + i]
    if i == 0 and column == 0:
        state[REFINE_CONDITION] = 0
        state[REFINE_PENDING] = 0
        state[REFINE_ROUND] = 0
    if componentwise == 0:
        magnitude = wp.abs(rhs[column * n + i])
        # Read first: the column's maximum only grows, so most rows skip the contended atomic.
        if magnitude > rhs_scale[column]:
            wp.atomic_max(rhs_scale, column, magnitude)


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
    # inconsistency left there, which no refinement removes. Spelled as "not within": a NaN
    # residual or scale fails the test rather than passing it.
    if dropped[i] == 0 and not (wp.abs(total) <= tolerance * scale):
        state[REFINE_PENDING] = 1


@wp.kernel
def refine_advance(cap: wp.int32, state: wp.array[wp.int32]) -> None:
    # Close a refinement round: continue while some row is above tolerance, up to ``cap`` solves.
    state[REFINE_CONDITION] = wp.where(
        state[REFINE_PENDING] != 0 and state[REFINE_ROUND] < cap, 1, 0
    )
    state[REFINE_PENDING] = 0
    state[REFINE_ROUND] = state[REFINE_ROUND] + 1


# Positions of a component's rows one block of ``null_space_partials`` sums.
NULL_SPACE_CHUNK = wp.constant(1024)


@wp.func
def chunk_component(chunk_offsets: wp.array[wp.int32], chunk: wp.int32) -> wp.int32:
    # The component owning ``chunk``: the last whose first chunk is at or before it.
    return binary_search_index(chunk_offsets, chunk) - 1


@wp.kernel
def null_space_partials(
    component_offsets: wp.array[wp.int32],
    component_rows: wp.array[wp.int32],
    component_label: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    chunk_bound: wp.int32,
    singular: wp.array[wp.int32],
    initial: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    n: wp.int32,
    out_partials: wp.array[wp.float64],
) -> None:
    # A singular component's constant offset, first half: the solve held one row at zero, so the
    # component is shifted to the initial guess's mean, the answer conjugate gradient returns from
    # it. One block per ``NULL_SPACE_CHUNK`` of a component's rows sums its ``solution - initial``;
    # ``null_space_means`` adds a component's chunks in order, so the shift is reproducible.
    b, lane = wp.tid()
    chunk = b % chunk_bound
    column = b // chunk_bound
    if chunk >= chunk_offsets[chunk_offsets.shape[0] - 1]:
        return
    component = chunk_component(chunk_offsets, chunk)
    if singular[component_label[component]] == 0:
        return
    offset = column * n
    start = component_offsets[component] + (chunk - chunk_offsets[component]) * NULL_SPACE_CHUNK
    stop = wp.min(start + NULL_SPACE_CHUNK, component_offsets[component + 1])
    local = wp.float64(0.0)
    k = start + lane
    while k < stop:
        r = component_rows[k]
        local += solution[offset + r] - initial[offset + r]
        k += wp.block_dim()
    total = block_sum(local)
    if lane == 0:
        out_partials[column * chunk_bound + chunk] = total


@wp.kernel
def null_space_means(
    component_offsets: wp.array[wp.int32],
    component_label: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    chunk_bound: wp.int32,
    singular: wp.array[wp.int32],
    partials: wp.array[wp.float64],
    out_means: wp.array[wp.float64],
) -> None:
    # Second step: a singular component's mean offset, its chunk partials summed in chunk order by
    # one thread a component and column (a non-singular component's is never read).
    component, column = wp.tid()
    if singular[component_label[component]] == 0:
        return
    total = wp.float64(0.0)
    for k in range(chunk_offsets[component], chunk_offsets[component + 1]):
        total += partials[column * chunk_bound + k]
    count = component_offsets[component + 1] - component_offsets[component]
    out_means[column * component_label.shape[0] + component] = total / wp.float64(count)


@wp.kernel
def null_space_shift(
    component_offsets: wp.array[wp.int32],
    component_rows: wp.array[wp.int32],
    component_label: wp.array[wp.int32],
    singular: wp.array[wp.int32],
    means: wp.array[wp.float64],
    n: wp.int32,
    solution: wp.array[wp.float64],
) -> None:
    # Last step: each row of a singular component subtracts its component's mean offset.
    p, column = wp.tid()
    component = binary_search_index(component_offsets, p) - 1
    if singular[component_label[component]] == 0:
        return
    r = column * n + component_rows[p]
    solution[r] = solution[r] - means[column * component_label.shape[0] + component]


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
    # flags a column that did not converge, so one four-byte read answers both questions. Both
    # tests are spelled "not within", so a NaN anywhere flags.
    i, column = wp.tid()
    offset = column * n
    total, scale = residual_and_scale(offsets, columns, values, solution, rhs, offset, i)
    if not (wp.abs(total) <= tolerance * scale):
        out_flag[0] = 1
    if i == 0 and residual.shape[0] > 0:
        if not (residual[column] <= threshold[column]):
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

# Positions of ``order`` one ``dissection_partials`` block sums, its lanes striding them. CUDA takes
# 4096 a block: a thread summing a 256-position chunk serially left a 41 k-row depth on 160
# threads, latency-bound. The CPU's one-lane blocks keep the 256-position serial sums, so its
# orderings are the ones they always were.
DISSECTION_CHUNK_CUDA = 4096
DISSECTION_CHUNK_CPU = 256
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


@wp.func
def sorted_segment(keys: wp.array[wp.uint64], p: wp.int32) -> wp.int32:
    return wp.int32(keys[p] >> wp.uint64(32))


@wp.func
def live_count(segment_start: wp.array[wp.int32]) -> wp.int32:
    # The depth's live vertices: its segment starts end in their total, every slot past its
    # segments being an empty one.
    return segment_start[segment_start.shape[0] - 1]


# ``dissection_state`` slots: the tree's node count, and the depth being split.
DISSECTION_NODES = wp.constant(0)
DISSECTION_DEPTH = wp.constant(1)


@wp.func
def dissection_task(
    segment_start: wp.array[wp.int32], tasks: wp.array[wp.int32], chunk: wp.int32, t: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # Task ``t``'s segment and its ``[begin, end)`` positions: each segment's positions in
    # ``chunk``-sized tasks, numbered segment by segment. ``tasks`` is the second half of the
    # depth's table, scanned with the starts, so each entry carries the live count in front.
    shifted = t + live_count(segment_start)
    s = chunk_component(tasks, shifted)
    begin = segment_start[s] + chunk * (shifted - tasks[s])
    return s, begin, wp.min(begin + chunk, segment_start[s + 1])


@wp.func
def segment_axis(
    order: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    segment_start: wp.array[wp.int32],
    tasks: wp.array[wp.int32],
    partials: wp.array[moment_vector],
    s: wp.int32,
) -> tuple[wp.vec3d, wp.vec3d]:
    # A non-empty segment's centroid and the principal axis of its covariance (the eigenvector of
    # the largest eigenvalue), from its tasks' partial moments summed in task order.
    live = live_count(segment_start)
    m = moment_vector()
    for t in range(tasks[s], tasks[s + 1]):
        m += partials[t - live]
    count = wp.float64(segment_start[s + 1] - segment_start[s])
    mean = wp.vec3d(m[0], m[1], m[2]) / count
    covariance = wp.mat33d(
        m[3], m[4], m[5],
        m[4], m[6], m[7],
        m[5], m[7], m[8],
    ) / count - wp.outer(mean, mean)  # fmt: skip
    # The eigenvectors in ``float32`` from the ``float64`` covariance: the axis only orients the
    # split, and a ``float64`` ``eig3`` -- an iteration of square roots and divisions at a
    # sixty-fourth of the ``float32`` rate on this class of device -- was most of a depth.
    scale = wp.max(wp.max(covariance[0, 0], covariance[1, 1]), covariance[2, 2])
    reduced = wp.mat33(covariance / wp.where(scale > wp.float64(0.0), scale, wp.float64(1.0)))
    vectors, values = wp.eig3(reduced)
    best = wp.int32(0)
    for q in range(1, 3):
        if values[q] > values[best]:
            best = wp.int32(q)
    centre = segment_reference(order, coordinates, segment_start[s]) + mean
    axis = wp.vec3d(
        wp.float64(vectors[0, best]), wp.float64(vectors[1, best]), wp.float64(vectors[2, best])
    )
    return centre, axis


@wp.kernel
def dissection_partials(
    order: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    segment_start: wp.array[wp.int32],
    tasks: wp.array[wp.int32],
    chunk: wp.int32,
    tickets: wp.array[wp.int32],
    out_partials: wp.array[moment_vector],
    out_centre: wp.array[wp.vec3d],
    out_axis: wp.array[wp.vec3d],
) -> None:
    # One chunk of one segment, one block: the first and second moments of its points about the
    # segment's reference point, each lane summing a strided share and the block reducing them
    # (the block reduction and the segment's sum in task order are both fixed, so the axis is the
    # same on every run). Launched over a bound on the depth's task count; a block past the depth's
    # tasks returns whole.
    #
    # The block storing a segment's last partial (``take_ticket`` on the segment's slot of the
    # zeroed ``tickets``, which it clears again for the next depth) forms the segment's centre and
    # axis from them in task order (``segment_axis``), so the axes need no launch of their own. An
    # empty segment slot has no task, so its centre and axis are not written.
    t, lane = wp.tid()
    if t + live_count(segment_start) >= tasks[tasks.shape[0] - 1]:
        return
    s, begin, end = dissection_task(segment_start, tasks, chunk, t)
    reference = segment_reference(order, coordinates, segment_start[s])
    m = moment_vector()
    p = begin + lane
    while p < end:
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
        p += wp.block_dim()
    total = block_sum(m)
    if lane != 0:
        return
    out_partials[t] = total
    ticket = take_ticket(tickets, s, wp.int32(0))
    if last_ticket(ticket, tasks[s + 1] - tasks[s]) == 0:
        return
    tickets[s] = 0
    if segment_start[s + 1] > segment_start[s]:
        centre, axis = segment_axis(order, coordinates, segment_start, tasks, out_partials, s)
        out_centre[s] = centre
        out_axis[s] = axis


@wp.kernel
def dissection_keys(
    order: wp.array[wp.int32],
    order_segment: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    centre: wp.array[wp.vec3d],
    axis: wp.array[wp.vec3d],
    segment_start: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_vertices: wp.array[wp.int32],
) -> None:
    # The sort key of a live vertex: its segment, then its projection on the segment's axis. A
    # position past the depth's live vertices takes a key above every segment's
    # (``segment_start``'s slot count), so the sort over the whole buffer leaves it last.
    p = wp.int32(wp.tid())
    if p >= live_count(segment_start):
        out_keys[p] = (wp.uint64(segment_start.shape[0]) << wp.uint64(32)) | wp.uint64(0xFFFFFFFF)
        out_vertices[p] = 0
        return
    s = order_segment[p]
    v = order[p]
    projection = wp.dot(shifted_point(coordinates, v, centre[s]), axis[s])
    bits = ordered_float_bits(wp.float32(projection))
    out_keys[p] = (wp.uint64(s) << wp.uint64(32)) | wp.uint64(bits)
    out_vertices[p] = v


@wp.func
def lower_half_rows(count: wp.int32) -> wp.int32:
    # The rows of a ``count``-row segment below its median: the first half of its sorted order.
    return count // 2


@wp.func
def median_side(p: wp.int32, start: wp.int32, count: wp.int32) -> wp.int32:
    # Which half of its segment (``count`` rows from ``start``) sorted position ``p`` falls in:
    # ``0`` the lower, ``1`` the upper. ``dissection_sides``, ``dissection_relabel`` and
    # ``dissection_children`` split by this one rule.
    return wp.where(p - start >= lower_half_rows(count), 1, 0)


@wp.kernel
def dissection_sides(
    keys: wp.array[wp.uint64],
    vertices: wp.array[wp.int32],
    segment_start: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_tag: wp.array[wp.int32],
) -> None:
    # Each sorted vertex's side of its segment's median, as ``base + 2 segment + upper`` with
    # ``base`` the depth times twice the segment slots: tags of earlier depths are all below it, so
    # the array needs no reset between depths.
    p = wp.int32(wp.tid())
    if p >= live_count(segment_start):
        return
    base = 2 * (segment_start.shape[0] - 1) * state[DISSECTION_DEPTH]
    s = sorted_segment(keys, p)
    start = segment_start[s]
    upper = median_side(p, start, segment_start[s + 1] - start)
    out_tag[vertices[p]] = base + 2 * s + upper


@wp.kernel
def dissection_crossings(
    keys: wp.array[wp.uint64],
    vertices: wp.array[wp.int32],
    segment_start: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    tag: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_lower_mark: wp.array[wp.int32],
    out_upper_mark: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The pattern edges from a segment's lower half to its upper half: both endpoints marked with
    # this depth's ``stamp``, and each segment's count of marked endpoints per side (an upper
    # endpoint counted by the first thread to mark it).
    p = wp.int32(wp.tid())
    if p >= live_count(segment_start):
        return
    stamp = state[DISSECTION_DEPTH] + 1
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
    state: wp.array[wp.int32],
    out_node: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
    out_next_segment: wp.array[wp.int32],
) -> None:
    # A sorted vertex leaves the live set as its segment's separator (the side with fewer marked
    # endpoints) or as a leaf's row, or stays, in the next depth's segment of its half. The halves
    # of a segment are consecutive in the sort and the host numbers the next segments in child
    # order, so the survivors, compacted in place, are grouped by next segment. A position past the
    # live vertices keeps nothing, so the scan over the whole buffer counts only survivors.
    p = wp.int32(wp.tid())
    if p >= live_count(segment_start):
        out_keep[p] = 0
        return
    stamp = state[DISSECTION_DEPTH] + 1
    s = sorted_segment(keys, p)
    v = vertices[p]
    start = segment_start[s]
    upper = median_side(p, start, segment_start[s + 1] - start)
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
def seed_dissection(
    n: wp.int32, chunk: wp.int32, out_table: wp.array[wp.int32], out_segments: wp.array2d[wp.int32]
) -> None:
    # The first depth's one segment: every row, at the start of the order, with no parent node.
    # ``out_table`` holds the depth's segment starts then its live-shifted task offsets -- the
    # inclusive scan ``dissection_bookkeeping`` leaves for every later depth -- a thread an entry.
    p = wp.int32(wp.tid())
    slots = out_table.shape[0] // 2 - 1
    if p > 0:
        out_table[p] = wp.where(p <= slots + 1, n, n + (n + chunk - 1) // chunk)
    if p == 0:
        out_segments[0, 0] = 0
        out_segments[1, 0] = -1
        out_segments[2, 0] = -1
        out_segments[3, 0] = -1


@wp.func
def dissection_children(
    segment_start: wp.array[wp.int32], counts: wp.array[wp.int32], s: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # Segment ``s``'s split: its separator side (``0`` none, ``1`` the lower half's crossing
    # endpoints, ``2`` the upper's, whichever set is smaller), the separator's size, and the two
    # halves' remaining sizes.
    count = segment_start[s + 1] - segment_start[s]
    lower_ends = counts[2 * s]
    upper_ends = counts[2 * s + 1]
    take_lower = lower_ends <= upper_ends
    separator = wp.where(take_lower, lower_ends, upper_ends)
    half = lower_half_rows(count)
    lower = half - wp.where(take_lower, separator, 0)
    upper = count - half - wp.where(take_lower, 0, separator)
    side = wp.where(separator > 0, wp.where(take_lower, 1, 2), 0)
    return side, separator, lower, upper


@wp.func
def write_tree_node(
    node: wp.int32,
    start: wp.int32,
    size: wp.int32,
    parent: wp.int32,
    depth: wp.int32,
    root: wp.int32,
    tree: wp.array2d[wp.int32],
) -> None:
    # One node of the dissection tree, a row of ``tree``: its first position, rows, parent, depth
    # and root (its own id when it roots its tree). Row-major by node, so the nodes the
    # dissection made are one contiguous prefix to read back.
    tree[node, 0] = start
    tree[node, 1] = size
    tree[node, 2] = parent
    tree[node, 3] = depth
    tree[node, 4] = wp.where(root >= 0, root, node)


# ``segment_tallies`` packs a slot's separator, leaf and next-segment counts into one ``uint32``
# (bits 0-9, 10-20, 21-31): summed over a block-wide chunk of at most 512 slots no field
# overflows, so a chunk ranks its slots by one scan of them, and two of the next segments' rows
# and tasks.
TALLY_LEAF = wp.uint32(1 << 10)
TALLY_NEXT = wp.uint32(1 << 21)


@wp.func
def segment_tallies(
    segment_start: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    leaf_rows: wp.int32,
    chunk: wp.int32,
    s: wp.int32,
    live: wp.int32,
) -> tuple[wp.uint32, wp.int32, wp.int32]:
    # What segment slot ``s`` adds to its depth: a separator node (``0`` or ``1``), leaf nodes and
    # next-depth segments (``0`` to ``2`` each), packed, and its next segments' rows and task
    # counts. A slot past ``live`` adds nothing.
    nodes = wp.uint32(0)
    rows_on = wp.int32(0)
    tasks_on = wp.int32(0)
    if s < live:
        side, _separator, lower, upper = dissection_children(segment_start, counts, s)
        nodes = wp.where(side != 0, wp.uint32(1), wp.uint32(0))
        for k in range(2):
            rows = wp.where(k == 0, lower, upper)
            leaf = rows > 0 and rows <= leaf_rows
            goes_on = rows > leaf_rows
            nodes += wp.where(leaf, TALLY_LEAF, wp.uint32(0))
            nodes += wp.where(goes_on, TALLY_NEXT, wp.uint32(0))
            rows_on += wp.where(goes_on, rows, 0)
            tasks_on += wp.where(goes_on, (rows + chunk - 1) // chunk, 0)
    return nodes, rows_on, tasks_on


@wp.func
def tally_counts(packed: wp.uint32) -> wp.vec3i:
    # The separator, leaf and next-segment counts a ``segment_tallies`` sum packs.
    return wp.vec3i(
        wp.int32(packed & wp.uint32(0x3FF)),
        wp.int32((packed >> wp.uint32(10)) & wp.uint32(0x7FF)),
        wp.int32(packed >> wp.uint32(21)),
    )


@wp.func
def dissection_link(
    segment_start: wp.array[wp.int32],
    segments: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    leaf_rows: wp.int32,
    chunk: wp.int32,
    counts: wp.array[wp.int32],
    s: wp.int32,
    nodes_before: wp.vec3i,
    rows_before: wp.int32,
    tasks_before: wp.int32,
    n_separators: wp.int32,
    total_rows: wp.int32,
    out_tree: wp.array2d[wp.int32],
    out_separator_side: wp.array[wp.int32],
    out_separator_node: wp.array[wp.int32],
    out_child_next: wp.array[wp.int32],
    out_child_leaf: wp.array[wp.int32],
    out_next_table: wp.array[wp.int32],
    out_next_segments: wp.array2d[wp.int32],
) -> None:
    # Segment ``s``'s nodes and next-depth segments: its separator node, each half that is a leaf
    # as a leaf node, and each half that goes on as a next-depth segment (its start and task
    # offset into the next depth's table, already scanned). ``nodes_before``, ``rows_before``
    # and ``tasks_before`` are the ``segment_tallies`` of the slots before ``s``, and
    # ``n_separators`` / ``total_rows`` the depth's totals, so the nodes are numbered as the
    # host-side tree numbered them: the depth's separators in segment order, then its leaves in
    # child order. ``segments`` rows are each segment's first position, parent node, that node's
    # depth and root; ``out_tree`` holds ``write_tree_node``'s. The crossing counts are cleared
    # once read, for the next depth.
    if segment_start[s + 1] == segment_start[s]:
        return
    side, separator, lower, upper = dissection_children(segment_start, counts, s)
    counts[2 * s] = 0
    counts[2 * s + 1] = 0
    slots = segment_start.shape[0] - 1
    first = segments[0, s]
    parent = segments[1, s]
    parent_depth = segments[2, s]
    root = segments[3, s]
    base = state[DISSECTION_NODES]
    separator_node = wp.int32(-1)
    me = parent
    me_depth = parent_depth
    me_root = root
    if side != 0:
        separator_node = base + nodes_before[0]
        count = segment_start[s + 1] - segment_start[s]
        write_tree_node(
            separator_node, first + count - separator, separator, parent, parent_depth + 1, root,
            out_tree,
        )  # fmt: skip
        me = separator_node
        me_depth = parent_depth + 1
        me_root = wp.where(root >= 0, root, separator_node)
    out_separator_side[s] = side
    out_separator_node[s] = separator_node
    leaves = nodes_before[1]
    nexts = nodes_before[2]
    rows_through = rows_before
    tasks_through = total_rows + tasks_before
    for k in range(2):
        c = 2 * s + k
        rows = wp.where(k == 0, lower, upper)
        child_first = first + wp.where(k == 0, 0, lower)
        leaf = wp.int32(-1)
        next_segment = wp.int32(-1)
        if rows > 0 and rows <= leaf_rows:
            leaf = base + n_separators + leaves
            leaves += 1
            write_tree_node(leaf, child_first, rows, me, me_depth + 1, me_root, out_tree)
        elif rows > leaf_rows:
            next_segment = nexts
            nexts += 1
            rows_through += rows
            tasks_through += (rows + chunk - 1) // chunk
            out_next_table[next_segment + 1] = rows_through
            out_next_table[slots + 2 + next_segment] = tasks_through
            out_next_segments[0, next_segment] = child_first
            out_next_segments[1, next_segment] = me
            out_next_segments[2, next_segment] = me_depth
            out_next_segments[3, next_segment] = me_root
        out_child_leaf[c] = leaf
        out_child_next[c] = next_segment


@wp.kernel
def dissection_bookkeeping(
    segment_start: wp.array[wp.int32],
    segments: wp.array2d[wp.int32],
    live: wp.int32,
    leaf_rows: wp.int32,
    chunk: wp.int32,
    counts: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_tree: wp.array2d[wp.int32],
    out_separator_side: wp.array[wp.int32],
    out_separator_node: wp.array[wp.int32],
    out_child_next: wp.array[wp.int32],
    out_child_leaf: wp.array[wp.int32],
    out_next_table: wp.array[wp.int32],
    out_next_segments: wp.array2d[wp.int32],
) -> None:
    # A depth's per-segment bookkeeping in one block, over its ``live`` first segment slots (a
    # depth has at most ``2^depth`` segments): the depth's totals, then the slots a block-wide
    # chunk at a time, each ranked by an exclusive scan of ``segment_tallies`` carried across
    # chunks and linked (``dissection_link``), then the next depth's table past its last segment
    # filled with the totals -- the inclusive scan its readers expect -- and the node count
    # advanced. The tables are a few entries a segment, so one block walks them in less time
    # than separate flag, scan and link launches took to issue.
    _block, lane = wp.tid()
    lanes = wp.block_dim()
    slots = segment_start.shape[0] - 1
    nodes_total = wp.vec3i(0, 0, 0)
    rows_total = wp.int32(0)
    tasks_total = wp.int32(0)
    for start in range(0, live, lanes):
        nodes, rows, tasks = segment_tallies(segment_start, counts, leaf_rows, chunk,
                                             start + lane, live)  # fmt: skip
        nodes_total += tally_counts(block_sum(nodes))
        rows_total += block_sum(rows)
        tasks_total += block_sum(tasks)
    nodes_carry = wp.vec3i(0, 0, 0)
    rows_carry = wp.int32(0)
    tasks_carry = wp.int32(0)
    for start in range(0, live, lanes):
        s = start + lane
        nodes, rows, tasks = segment_tallies(segment_start, counts, leaf_rows, chunk, s, live)
        nodes_before = nodes_carry + tally_counts(wp.untile(wp.tile_scan_exclusive(wp.tile(nodes))))
        rows_before = rows_carry + wp.untile(wp.tile_scan_exclusive(wp.tile(rows)))
        tasks_before = tasks_carry + wp.untile(wp.tile_scan_exclusive(wp.tile(tasks)))
        nodes_carry += tally_counts(block_sum(nodes))
        rows_carry += block_sum(rows)
        tasks_carry += block_sum(tasks)
        if s < live:
            dissection_link(segment_start, segments, state, leaf_rows, chunk, counts, s,
                            nodes_before, rows_before, tasks_before, nodes_total[0], rows_total,
                            out_tree, out_separator_side, out_separator_node, out_child_next,
                            out_child_leaf, out_next_table, out_next_segments)  # fmt: skip
    for j in range(nodes_total[2] + lane, slots, lanes):
        out_next_table[j + 1] = rows_total
        out_next_table[slots + 2 + j] = rows_total + tasks_total
    block_barrier()
    if lane == 0:
        out_next_table[slots + 1] = rows_total
        state[DISSECTION_NODES] = state[DISSECTION_NODES] + nodes_total[0] + nodes_total[1]


@wp.kernel
def dissection_compact(
    vertices: wp.array[wp.int32],
    kept_through: wp.array[wp.int32],
    next_segment: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
    out_order_segment: wp.array[wp.int32],
) -> None:
    # The survivors in sorted order: ``kept_through`` is the keep flags' inclusive scan, done in
    # place, from which each position recovers its own flag. The depth closes here: it advances.
    p = wp.int32(wp.tid())
    if p == 0:
        state[DISSECTION_DEPTH] = state[DISSECTION_DEPTH] + 1
    q, kept = scanned_count(kept_through, p)
    if kept != 0:
        out_order[q] = vertices[p]
        out_order_segment[q] = next_segment[p]


# --------------------------------------------------------------------------------------
# Symbolic analysis: elimination order and row structure
# --------------------------------------------------------------------------------------


# ``analysis_header`` slots, read back together at the analysis' first sync: the supernode and
# level counts, the structure pairs, the operator's lower entries, the components, the stored
# entries (``offsets[n]``) and whether some stored entry's transpose is not stored.
HEADER_SUPERNODES = wp.constant(0)
HEADER_LEVELS = wp.constant(1)
HEADER_PAIRS = wp.constant(2)
HEADER_ENTRIES = wp.constant(3)
HEADER_COMPONENTS = wp.constant(4)
HEADER_STORED = wp.constant(5)
HEADER_ASYMMETRIC = wp.constant(6)
HEADER_SLOTS = 7


@wp.kernel
def top_depth_rows(
    tree: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    width: wp.int32,
    out_rows: wp.array[wp.int32],
) -> None:
    # Each tree's rows at each depth, ``root * width + depth``: the top merge's cumulative counts.
    t = wp.int32(wp.tid())
    if t >= state[DISSECTION_NODES]:
        return
    wp.atomic_add(out_rows, tree[t, 4] * width + tree[t, 3], tree[t, 1])


@wp.kernel
def supernode_keys(
    tree: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    depth_rows: wp.array[wp.int32],
    width: wp.int32,
    top_rows: wp.int32,
    n: wp.int32,
    sentinel: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_values: wp.array[wp.int32],
) -> None:
    # Each node keyed by its group's first position, then its own: a tree's top -- its depths
    # while their cumulative rows stay within ``top_rows``, the root's at least -- is merged into
    # the root and takes the root's place, every other node is a group of its own. Sorted, the
    # nodes run in elimination order (a node's first position orders its subtree in postorder)
    # and each group is one run. A slot past the tree's nodes sorts last.
    t = wp.int32(wp.tid())
    out_values[t] = t
    if t >= state[DISSECTION_NODES]:
        out_keys[t] = sentinel
        return
    root = tree[t, 4]
    depth = tree[t, 3]
    rows = wp.int32(0)
    for d in range(depth + 1):
        rows += depth_rows[root * width + d]
    group_start = tree[t, 0]
    if depth == 0 or rows <= top_rows:
        group_start = tree[root, 0]
    out_keys[t] = wp.uint64(group_start) * wp.uint64(n + 1) + wp.uint64(tree[t, 0])


@wp.func
def key_group(keys: wp.array[wp.uint64], p: wp.int32, n: wp.int32) -> wp.int64:
    return wp.int64(keys[p] // wp.uint64(n + 1))


@wp.kernel
def supernode_flags(
    keys: wp.array[wp.uint64],
    nodes: wp.array[wp.int32],
    tree: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    n: wp.int32,
    out_counts: wp.array[wp.vec2i],
) -> None:
    # Per sorted node: whether it starts a group, and its rows -- scanned together into each
    # node's supernode and first position.
    p = wp.int32(wp.tid())
    if p >= state[DISSECTION_NODES]:
        out_counts[p] = wp.vec2i(0, 0)
        return
    first = p == 0 or key_group(keys, p, n) != key_group(keys, wp.max(p - 1, 0), n)
    out_counts[p] = wp.vec2i(wp.where(first, 1, 0), tree[nodes[p], 1])


@wp.kernel
def supernode_tables(
    keys: wp.array[wp.uint64],
    nodes: wp.array[wp.int32],
    ends: wp.array[wp.vec2i],
    tree: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    n: wp.int32,
    out_node_start: wp.array[wp.int32],
    out_node_supernode: wp.array[wp.int32],
    out_c0: wp.array[wp.int32],
    out_ncol: wp.array[wp.int32],
    out_kept: wp.array[wp.int32],
    out_header: wp.array[wp.int32],
) -> None:
    # Every node's first position and supernode; each supernode's first column, column count and
    # the node it keeps (its group's last: a merged top's root, whose interval ends the tree's).
    p = wp.int32(wp.tid())
    count = state[DISSECTION_NODES]
    if p >= count:
        return
    node = nodes[p]
    size = tree[node, 1]
    sn = ends[p][0] - 1
    start = ends[p][1] - size
    out_node_start[node] = start
    out_node_supernode[node] = sn
    group = key_group(keys, p, n)
    if p == 0 or key_group(keys, wp.max(p - 1, 0), n) != group:
        out_c0[sn] = start
    wp.atomic_add(out_ncol, sn, size)
    if p == count - 1 or key_group(keys, wp.min(p + 1, count - 1), n) != group:
        out_kept[sn] = node
    if p == count - 1:
        out_header[HEADER_SUPERNODES] = sn + 1


@wp.kernel
def supernode_links(
    kept: wp.array[wp.int32],
    tree: wp.array2d[wp.int32],
    node_supernode: wp.array[wp.int32],
    c0: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    header: wp.array[wp.int32],
    out_parent: wp.array[wp.int32],
    out_column_end: wp.array[wp.int32],
) -> None:
    # A supernode's parent (its kept node's parent's supernode, ``-1`` for a root) and the end of
    # its columns.
    sn = wp.int32(wp.tid())
    if sn >= header[HEADER_SUPERNODES]:
        return
    parent = tree[kept[sn], 2]
    out_parent[sn] = wp.where(parent >= 0, node_supernode[wp.max(parent, 0)], -1)
    out_column_end[sn] = c0[sn] + ncol[sn]


@wp.kernel
def supernode_heights(
    parent: wp.array[wp.int32], out_height: wp.array[wp.int32], out_header: wp.array[wp.int32]
) -> None:
    # Each supernode's height -- its longest path down to a leaf -- by every supernode walking up
    # to its root and raising each ancestor's to its distance; the walks' longest is the number
    # of levels.
    sn = wp.int32(wp.tid())
    if sn >= out_header[HEADER_SUPERNODES]:
        return
    a = sn
    k = wp.int32(0)
    while a >= 0:
        wp.atomic_max(out_height, a, k)
        a = parent[a]
        k += 1
    wp.atomic_max(out_header, HEADER_LEVELS, k)


# One supernode's prefix-summed quantities in supernode order: front entries, solve-block entries,
# forward and backward chunk slots, forward and backward tasks.
front_terms = wp.types.vector(length=6, dtype=wp.int64)
# ``analysis_schedule`` layout: a header (``SCHEDULE_ROWS``, ``SCHEDULE_GROUPS``,
# ``SCHEDULE_PANELS``, the terms' totals from ``SCHEDULE_TERMS``), one row of ``LEVEL_FIELDS`` per
# level and a closing row, every level's panel work, then four entries per extend-add group.
SCHEDULE_ROWS = wp.constant(0)
SCHEDULE_GROUPS = wp.constant(1)
SCHEDULE_PANELS = wp.constant(2)
# The ``front_terms`` totals, each as two ``int32`` digits in base ``2^31`` (high, then low).
SCHEDULE_TERMS = wp.constant(3)
SCHEDULE_HEADER = wp.constant(15)
LEVEL_FIELDS = wp.constant(10)


@wp.func
def forward_chunk_start(c: wp.Int, size: wp.Int) -> wp.Int:
    # Tasks before chunk ``c`` of a supernode's forward product: chunk ``c'`` keeps its rows from
    # ``CHUNK c'`` (the diagonal block's rows above it are zero in ``L11^-1``). Generic: the task
    # maps index in ``int32``, the totals ``solve_task_counts`` forms (its value at the chunk
    # count) in ``int64``.
    one = type(c)(1)
    half = type(c)(CHUNK) // type(c)(2)
    return c * size - half * c * (c - one)


@wp.func
def backward_chunk_start(c: wp.Int, width: wp.Int) -> wp.Int:
    # Tasks before chunk ``c`` of a supernode's backward product: chunk ``c'`` keeps the columns
    # below ``CHUNK (c' + 1)``. Generic as ``forward_chunk_start`` is.
    one = type(c)(1)
    chunk = type(c)(CHUNK)
    half = chunk // type(c)(2)
    full = width // chunk
    if c <= full:
        return half * c * (c + one)
    return half * full * (full + one) + (c - full) * width


@wp.func
def solve_task_counts(
    ncol: wp.int64, size: wp.int64
) -> tuple[wp.int64, wp.int64, wp.int64, wp.int64]:
    # A supernode's forward and backward chunk counts and task counts: a forward task is a solve
    # block row times ``CHUNK`` of its columns, a backward one ``CHUNK`` of a column's rows; a
    # chunk skips the rows (forward) or columns (backward) where ``L11^-1`` is zero. The task
    # counts are the chunk-start rules at the chunk counts, so the maps and the totals agree.
    chunk = wp.int64(CHUNK)
    f_chunks = (ncol + chunk - wp.int64(1)) // chunk
    b_chunks = (size + chunk - wp.int64(1)) // chunk
    f_count = forward_chunk_start(f_chunks, size)
    b_count = backward_chunk_start(b_chunks, ncol)
    return f_chunks, b_chunks, f_count, b_count


@wp.kernel
def front_terms_and_keys(
    ncol: wp.array[wp.int32],
    nrow: wp.array[wp.int32],
    height: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    sentinel: wp.uint64,
    out_front_size: wp.array[wp.int32],
    out_parent: wp.array[wp.int32],
    out_terms: wp.array[front_terms],
    out_level_keys: wp.array[wp.uint64],
    out_level_values: wp.array[wp.int32],
    out_child_keys: wp.array[wp.uint64],
    out_child_values: wp.array[wp.int32],
    out_level_stats: wp.array2d[wp.int32],
) -> None:
    # A supernode's front size, its clamped parent and its prefix-summed terms (after a zero slot),
    # its key in the
    # level order (height, then supernode), its key among its parent's children when it extends
    # a parent (parent, then supernode; otherwise last), and its level's front count, widest
    # column count and most rows below a first panel.
    sn = wp.int32(wp.tid())
    n_sn = ncol.shape[0]
    c = ncol[sn]
    size = c + nrow[sn]
    out_front_size[sn] = size
    # The parent as the extend-add reads it: a root's own ``-1`` as ``0``, never read.
    out_parent[sn] = wp.max(parent[sn], 0)
    c64 = wp.int64(c)
    s64 = wp.int64(size)
    f_chunks, b_chunks, f_count, b_count = solve_task_counts(c64, s64)
    terms = front_terms()
    terms[0] = s64 * s64
    terms[1] = s64 * c64
    terms[2] = f_chunks * s64
    terms[3] = b_chunks * c64
    terms[4] = f_count
    terms[5] = b_count
    out_terms[sn + 1] = terms
    h = height[sn]
    out_level_keys[sn] = wp.uint64(h) * wp.uint64(n_sn) + wp.uint64(sn)
    out_level_values[sn] = sn
    out_child_values[sn] = sn
    out_child_keys[sn] = sentinel
    if parent[sn] >= 0 and nrow[sn] > 0:
        out_child_keys[sn] = wp.uint64(parent[sn]) * wp.uint64(n_sn) + wp.uint64(sn)
    wp.atomic_add(out_level_stats, h, 0, 1)
    wp.atomic_max(out_level_stats, h, 1, c)
    wp.atomic_max(out_level_stats, h, 2, size - wp.min(PANEL, c))


@wp.kernel
def level_terms(
    level_nodes: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    out_terms: wp.array[wp.vec4i],
) -> None:
    # The level-major counts the solve maps are laid out by -- forward tasks, front rows, backward
    # tasks, columns -- in level order, after a zero slot, for one scan.
    r = wp.int32(wp.tid())
    sn = level_nodes[r]
    _f_chunks, _b_chunks, f_count, b_count = solve_task_counts(
        wp.int64(ncol[sn]), wp.int64(front_size[sn])
    )
    out_terms[r + 1] = wp.vec4i(wp.int32(f_count), front_size[sn], wp.int32(b_count), ncol[sn])


@wp.kernel
def split_terms(
    terms: wp.array[front_terms],
    level_sums: wp.array[wp.vec4i],
    level_nodes: wp.array[wp.int32],
    out_front_offset: wp.array[wp.int64],
    out_block_offset: wp.array[wp.int64],
    out_f_base: wp.array[wp.int32],
    out_b_base: wp.array[wp.int32],
    out_level_offsets: wp.array2d[wp.int32],
    out_column_base: wp.array[wp.int32],
) -> None:
    # The scanned terms as the tables the maps read: every supernode's front and solve-block
    # offsets and chunk bases, the four level-major prefixes, and each supernode's first column in
    # level order.
    i = wp.int32(wp.tid())
    t = terms[i]
    out_front_offset[i] = t[0]
    out_block_offset[i] = t[1]
    out_f_base[i] = wp.int32(t[2])
    out_b_base[i] = wp.int32(t[3])
    sums = level_sums[i]
    for k in range(4):
        out_level_offsets[k, i] = sums[k]
    if i < level_nodes.shape[0]:
        out_column_base[level_nodes[i]] = sums[3]


@wp.kernel
def sibling_keys(
    child_keys: wp.array[wp.uint64],
    child_values: wp.array[wp.int32],
    height: wp.array[wp.int32],
    sentinel: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_values: wp.array[wp.int32],
) -> None:
    # A child's extend-add key: its height, then its slot among its parent's children (in child
    # order), then the child -- one launch per ``(height, slot)`` group keeps two children of a
    # launch from sharing a parent.
    p = wp.int32(wp.tid())
    n_sn = child_keys.shape[0]
    key = child_keys[p]
    out_values[p] = child_values[p]
    if key == sentinel:
        out_keys[p] = sentinel
        return
    parent = key // wp.uint64(n_sn)
    slot = p - binary_search_index_left(child_keys, parent * wp.uint64(n_sn))
    child = child_values[p]
    group = wp.uint64(height[child]) * wp.uint64(n_sn + 1) + wp.uint64(slot)
    out_keys[p] = group * wp.uint64(n_sn) + wp.uint64(child)


@wp.func
def extend_group_key(keys: wp.array[wp.uint64], p: wp.int32) -> wp.uint64:
    return keys[p] // wp.uint64(keys.shape[0])


@wp.kernel
def extend_group_flags(
    keys: wp.array[wp.uint64], sentinel: wp.uint64, out_ends: wp.array[wp.int32]
) -> None:
    # Whether a sorted child starts an extend-add group, scanned in place into the group ranks.
    p = wp.int32(wp.tid())
    first = keys[p] != sentinel and (
        p == 0 or extend_group_key(keys, p) != extend_group_key(keys, wp.max(p - 1, 0))
    )
    out_ends[p] = wp.where(first, 1, 0)


@wp.kernel
def extend_group_tables(
    keys: wp.array[wp.uint64],
    children: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    nrow: wp.array[wp.int32],
    sentinel: wp.uint64,
    out_groups: wp.array2d[wp.int32],
) -> None:
    # Each extend-add group's first and end position among the sorted children, its key's level
    # and its widest member's rows (``out_groups`` rows: level, start, end, width; zeroed).
    p = wp.int32(wp.tid())
    if keys[p] == sentinel:
        return
    g = ends[p] - 1
    group = extend_group_key(keys, p)
    count = keys.shape[0]
    if p == 0 or extend_group_key(keys, wp.max(p - 1, 0)) != group:
        out_groups[g, 0] = wp.int32(group // wp.uint64(count + 1))
        out_groups[g, 1] = p
    last = p == count - 1
    if not last:
        nxt = keys[wp.min(p + 1, count - 1)]
        last = nxt == sentinel or nxt // wp.uint64(count) != group
    if last:
        out_groups[g, 2] = p + 1
    wp.atomic_max(out_groups, g, 3, nrow[children[p]])


@wp.kernel
def schedule_levels(
    level_stats: wp.array2d[wp.int32],
    level_offsets: wp.array2d[wp.int32],
    row_ends: wp.array[wp.int32],
    group_ends: wp.array[wp.int32],
    terms: wp.array[front_terms],
    out_schedule: wp.array[wp.int32],
) -> None:
    # One thread: each level's row of the schedule -- its first supernode in level order, its
    # front count, panel count, most rows below a first panel, widest column count, the four
    # level-major starts and its first panel's slot -- then a closing row with the totals, and
    # every panel's work seeded at one. The header also carries the scanned terms' totals, split
    # into base-``2^31`` digits, so the header's one read returns them too.
    totals = terms[terms.shape[0] - 1]
    radix = wp.int64(1) << wp.int64(31)
    for k in range(6):
        out_schedule[SCHEDULE_TERMS + 2 * k] = wp.int32(totals[k] // radix)
        out_schedule[SCHEDULE_TERMS + 2 * k + 1] = wp.int32(totals[k] % radix)
    n_levels = level_stats.shape[0]
    start = wp.int32(0)
    panels = wp.int32(0)
    for h in range(n_levels + 1):
        row = SCHEDULE_HEADER + h * LEVEL_FIELDS
        out_schedule[row] = start
        for k in range(4):
            out_schedule[row + 5 + k] = level_offsets[k, start]
        out_schedule[row + 9] = panels
        if h < n_levels:
            count = level_stats[h, 0]
            level_panels = (level_stats[h, 1] + PANEL - 1) // PANEL
            out_schedule[row + 1] = count
            out_schedule[row + 2] = level_panels
            out_schedule[row + 3] = level_stats[h, 2]
            out_schedule[row + 4] = level_stats[h, 1]
            start += count
            panels += level_panels
    out_schedule[SCHEDULE_ROWS] = row_ends[row_ends.shape[0] - 1]
    out_schedule[SCHEDULE_GROUPS] = group_ends[group_ends.shape[0] - 1]
    out_schedule[SCHEDULE_PANELS] = panels
    work = SCHEDULE_HEADER + (n_levels + 1) * LEVEL_FIELDS
    for p in range(panels):
        out_schedule[work + p] = 1


@wp.kernel
def schedule_work(
    level_nodes: wp.array[wp.int32],
    ncol: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    height: wp.array[wp.int32],
    groups: wp.array2d[wp.int32],
    n_levels: wp.int32,
    out_schedule: wp.array[wp.int32],
) -> None:
    # Every panel's update work: the largest trailing triangle and solve-block rows any front of
    # the level updates after it, a panel past a narrower front's columns counting that front's
    # last. Thread ``r`` also copies extend-add group ``r`` after the panel work, four entries
    # each: both read only the header ``schedule_levels`` wrote and write disjoint ranges, over
    # the same supernode-count grid, so they share one launch.
    r = wp.int32(wp.tid())
    tail = SCHEDULE_HEADER + (n_levels + 1) * LEVEL_FIELDS
    if r < out_schedule[SCHEDULE_GROUPS]:
        base = tail + out_schedule[SCHEDULE_PANELS] + 4 * r
        for k in range(4):
            out_schedule[base + k] = groups[r, k]
    sn = level_nodes[r]
    row = SCHEDULE_HEADER + height[sn] * LEVEL_FIELDS
    work = tail + out_schedule[row + 9]
    c = ncol[sn]
    size = front_size[sn]
    for p in range(out_schedule[row + 2]):
        last = wp.min((p + 1) * PANEL, c)
        span = size - last
        wp.atomic_max(out_schedule, work + p, span * span + span * last)


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


@wp.func
def structure_parent(
    parent: wp.array[wp.int32], column_end: wp.array[wp.int32], k: wp.int32, row: wp.int32
) -> wp.int32:
    # The supernode a row of ``k``'s structure is passed up to (``-1`` for none): ``k``'s parent,
    # unless the row is one of the parent's own columns -- there the path ends at its owner.
    q = parent[k]
    return wp.where(q >= 0 and row >= column_end[q], q, -1)


@wp.kernel
def count_structure_paths(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    column_end: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The row structure as paths: an operator entry's row belongs to the structure of the
    # supernode of its column and of every ancestor up to the row's owner (exclusive). Row ``i``'s
    # count of those ``(supernode, row)`` pairs over its entries.
    i = wp.int32(wp.tid())
    r = pos[i]
    count = wp.int32(0)
    for e in range(offsets[i], offsets[i + 1]):
        k = owns_entry(pos, owner, column_end, i, columns[e])
        while k >= 0:
            count += 1
            k = structure_parent(parent, column_end, k, r)
    out_counts[i] = count


@wp.kernel
def write_structure_paths(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    column_end: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    height: wp.array[wp.int32],
    n: wp.int32,
    shift: wp.int32,
    ends: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_values: wp.array[wp.int32],
) -> None:
    # ``count_structure_paths``' pairs, each packed as ``(height, supernode, row)`` for one sort
    # and deduplication: the rows come out level-major, each supernode's ascending, at the slots
    # the counts' inclusive scan ``ends`` gives row ``i``.
    i = wp.int32(wp.tid())
    q = wp.where(i > 0, ends[wp.max(i - 1, 0)], 0)
    r = pos[i]
    for e in range(offsets[i], offsets[i + 1]):
        k = owns_entry(pos, owner, column_end, i, columns[e])
        while k >= 0:
            out_keys[q] = (wp.uint64(height[k]) << wp.uint64(shift)) | (
                wp.uint64(k) * wp.uint64(n) + wp.uint64(r)
            )
            out_values[q] = q
            q += 1
            k = structure_parent(parent, column_end, k, r)


@wp.kernel
def emit_structure_rows(
    keys: wp.array[wp.uint64],
    run_ends: wp.array[wp.int32],
    key_mask: wp.uint64,
    n: wp.int32,
    out_rows: wp.array[wp.int32],
    out_row_node: wp.array[wp.int32],
    out_row_start: wp.array[wp.int32],
    out_nrow: wp.array[wp.int32],
) -> None:
    # Each distinct sorted pair once (``run_ends`` the in-place inclusive scan of the run starts):
    # its row and supernode stored at its rank, each supernode's first rank and its row count.
    p = wp.int32(wp.tid())
    i, starts = scanned_count(run_ends, p)
    if starts == 0:
        return
    key = keys[p] & key_mask
    k = wp.int32(key // wp.uint64(n))
    out_rows[i] = wp.int32(key % wp.uint64(n))
    out_row_node[i] = k
    if p == 0 or wp.int32((keys[wp.max(p - 1, 0)] & key_mask) // wp.uint64(n)) != k:
        out_row_start[k] = i
    wp.atomic_add(out_nrow, k, 1)


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


@wp.func
def permuted_lower(pos: wp.array[wp.int32], r: wp.int32, column: wp.int32) -> wp.int32:
    # The permuted column of an operator entry of permuted row ``r`` when the entry lies in the
    # permuted lower triangle (diagonal included), ``-1`` otherwise: the one rule that picks the
    # entries the factorization scatters.
    c = pos[column]
    return wp.where(r >= c, c, -1)


@wp.func
def stores_entry(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], row: wp.int32, column: wp.int32
) -> wp.int32:
    # Whether row ``row`` stores column ``column`` (``1`` or ``0``): a binary search of the row's
    # columns, sorted in every ``warp.sparse`` matrix, then a scan of the row should it not be.
    begin = offsets[row]
    end = offsets[row + 1]
    if begin >= end:
        return 0
    if columns[wp.lower_bound(columns, begin, end, column)] == column:
        return 1
    for e in range(begin, end):
        if columns[e] == column:
            return 1
    return 0


@wp.kernel
def count_lower_entries(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    pos: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_header: wp.array[wp.int32],
) -> None:
    # Row ``i``'s entries in the permuted lower triangle, and the pattern's structural symmetry:
    # every stored ``(i, j)`` must have its ``(j, i)`` stored too. The ordering, the row structure
    # and the entry maps read one triangle each, so a one-sided entry would leave the factor's
    # fill short of it and its extend-add out of its parent's front.
    i = wp.int32(wp.tid())
    count = wp.int32(0)
    r = pos[i]
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        if permuted_lower(pos, r, j) >= 0:
            count += 1
        if j != i and stores_entry(offsets, columns, j, i) == 0:
            out_header[HEADER_ASYMMETRIC] = 1
    out_counts[i] = count


@wp.kernel
def structure_counts(
    path_ends: wp.array[wp.int32],
    lower_ends: wp.array[wp.int32],
    component_ends: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_header: wp.array[wp.int32],
) -> None:
    # One thread: the scanned totals into the analysis header, for its one read.
    n = lower_ends.shape[0]
    out_header[HEADER_PAIRS] = path_ends[n - 1]
    out_header[HEADER_ENTRIES] = lower_ends[n - 1]
    out_header[HEADER_COMPONENTS] = component_ends[n - 1]
    out_header[HEADER_STORED] = offsets[n]


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
        c = permuted_lower(pos, r, columns[e])
        if c >= 0:
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
def component_chunks_and_diagonals(
    component_offsets: wp.array[wp.int32],
    last: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    front_offset: wp.array[wp.int64],
    c0: wp.array[wp.int32],
    front_size: wp.array[wp.int32],
    out_chunk_offsets: wp.array[wp.int32],
    out_diagonal: wp.array[wp.int64],
) -> None:
    # A component's ``NULL_SPACE_CHUNK`` count, into the tail of its chunk offsets (scanned after;
    # component 0 writes the leading zero), and where its last-eliminated row's diagonal sits in
    # the fronts arena.
    c = wp.int32(wp.tid())
    if c == 0:
        out_chunk_offsets[0] = 0
    count = component_offsets[c + 1] - component_offsets[c]
    out_chunk_offsets[c + 1] = (count + NULL_SPACE_CHUNK - 1) // NULL_SPACE_CHUNK
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
    distance: wp.array[wp.int32],
    first: wp.array[wp.float32],
    second: wp.array[wp.float32],
    store: wp.int32,
    search: wp.int32,
    out_coordinate: wp.array[wp.float32],
    out_key: wp.array[wp.uint64],
) -> None:
    # With ``store`` set, the last search's ``distance`` as a landmark coordinate (zero off the
    # searched components); with ``search`` set, per component the row farthest by
    # ``min(first, second)``, the lowest such row on a tie: the larger distance in the high word,
    # the complemented row in the low one. ``first`` or ``second`` may be ``out_coordinate``
    # itself: row ``v`` reads it only at ``v``, after its own store. An array a flag leaves off may
    # be ``None``.
    v = wp.int32(wp.tid())
    if store != 0:
        out_coordinate[v] = wp.float32(wp.max(distance[v], 0))
    if search == 0:
        return
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
