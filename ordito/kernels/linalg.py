"""
Kernels for the shared sparse linear-algebra layer (``ordito/linalg.py``).

The **reduced-system assembly** used by every fixed-value quadratic solve (``harmonic`` / ``tutte``
/ ``lscm`` / ``arap``). It is a **CSR-to-CSR** extraction in two passes -- count the surviving
entries per free row, then fill them -- rather than a COO emission handed to
``warp.sparse.bsr_from_triplets``. The input rows come out of a CSR, so they are already row-major,
already column-sorted and already duplicate-free, and ``free_map`` is monotone, so the extracted
row is sorted by construction: the sort and duplicate-accumulation a triplet build performs are pure
waste here, and that build was the overwhelming majority of ``min_quad_with_fixed``'s device time.

The module's other half, the batched conjugate-gradient iteration that solves the system this
assembles, lives in ``ordito.kernels.algorithms.conjugate_gradient``.

!!! note
    ``smoothing``'s region assembly (``dirichlet_system_values`` / ``least_squares_rows`` /
    ``normal_equations_*``) is deliberately *not* merged in here. It looks similar but solves a
    different problem: it **builds** the operator from per-edge weights over a region's own
    pattern, whereas these kernels **extract** the free-free block of an operator that already
    exists. Routing it through here would force an extra full matrix build.
"""

from typing import Any

import warp as wp

# Warp's own Householder QR. From ``warp._src.fem.linalg`` rather than the public
# ``warp.fem.linalg``: both bind the same two ``@wp.func``s, which inline here and trigger no fem
# codegen, but the public path executes ``warp/fem/__init__.py`` and eagerly loads the whole fem
# package, which is two orders of magnitude dearer to import and shows up in ``import ordito``.
# ``kernels/reduce.py`` reaches into ``warp._src`` on the same terms.
from warp._src.fem.linalg import householder_qr_decomposition, solve_triangular
from warp._src.sparse import make_bsr_mv_kernel, make_bsr_mv_tiled_kernel
from warp._src.types import array as array_type
from warp._src.types import type_is_generic

from ordito.kernels.array import OverloadTable, inverse_or_one
from ordito.kernels.reduce import block_chunk, block_max, block_sum, thread_fence

wp.set_module_options({"enable_backward": False})


@wp.func
def solve_normal_equations(matrix: Any, rhs: Any) -> tuple[Any, wp.bool]:
    """
    Solve a small dense symmetric system ``A x = b`` by Householder QR, at any rank.

    Returns ``(solution, ok)``; ``ok`` is False when the system is singular to ``1e-14``, in which
    case the returned vector is ``rhs`` unchanged. Reporting rather than raising, because the caller
    is a kernel: a per-vertex least-squares fit that fails on a degenerate 1-ring has to fall back,
    not abort the launch.

    ``|R[k, k]|`` is the norm of column k after the preceding reflections -- the QR analogue of the
    partial-pivot magnitude a Gaussian elimination would test, to within a ``sqrt(n)`` factor -- so
    a single threshold carries across ranks.

    **The threshold is absolute, so the caller owns the system's scale.** A ``@wp.func`` cannot see
    the units its caller works in, and a relative test is not available either: the normal matrix of
    a polynomial fit is scaled *inhomogeneously* by its design row (``[u^2, u v, v^2, u, v]``
    spreads a mesh scale ``h`` over ``h^8`` to ``h^2`` on the diagonal alone), so
    ``min|R| / max|R|`` moves with ``h`` exactly as ``min|R|`` does. Both callers divide their local
    coordinates by the neighbourhood radius before accumulating, which makes the matrix they hand
    over ``O(1)`` whatever the mesh's units; each says so at the site. A new caller that skips it
    gets a silent ``ok=False`` on a perfectly good system.

    **The rank is nowhere in this function, and that is the point.** It was written twice, as a 5x5
    for ``curvature``'s quadric fit and a 6x6 for ``smoothing``'s area-equalizing solve, because the
    singularity test was a ``for k in range(5)`` / ``range(6)`` loop and a generic matrix has no
    readable rank in kernel scope -- ``r.shape[0]`` is a ``WarpCodegenAttributeError`` at parse time
    on Warp 1.19. ``wp.min(wp.abs(wp.get_diag(r)))`` asks the identical question ("is some
    diagonal below tolerance") with no loop and no rank, which is what let the two collapse into
    one. Verified against ``numpy.linalg.solve`` at both ranks, and the two singularity predicates
    compared directly over hundreds of matrices per rank spanning fourteen orders of conditioning:
    they agree on every finite one, and part only on a matrix carrying a ``nan`` or an ``inf``,
    which the old loop passed (``wp.abs(nan) < tol`` is False, so no iteration rejected it) and this
    reports ``ok=False`` -- the safe direction, and the fallback both callers already take.
    """
    q, r = householder_qr_decomposition(matrix)
    if wp.min(wp.abs(wp.get_diag(r))) < wp.float64(1e-14):
        return rhs, False
    return solve_triangular(r, wp.transpose(q) * rhs), True


@wp.func
def free_row(fixed_mask: wp.array[wp.bool], free_map: wp.array[wp.int32], i: wp.int32) -> wp.int32:
    # The compact row index degree of freedom ``i`` occupies in the reduced system, or ``-1`` when
    # it is pinned and has no row at all. The sentinel rather than an early return, because a
    # ``@wp.func`` cannot return for its caller.
    #
    # This does not shorten the three lines it replaces -- ``if ri < 0: return`` costs what
    # ``if fixed_mask[i]: return`` cost. What it buys is that the mask/compact-index convention is
    # written down *once*, in the module that owns the elimination, instead of being re-inferred at
    # every site: ``free_map`` is defined only where ``fixed_mask`` is False, it is monotone
    # non-decreasing over the free indices (which is what lets ``interior_row_entries`` skip a
    # triplet sort), and reading it at a pinned index gives a stale or out-of-range value rather
    # than an error.
    #
    # **The mask's polarity is the argument's name and nothing else**, which is why ``selected_row``
    # below exists: both ask the identical question of a (mask, index map) pair, this one of a mask
    # marking what is *excluded* and that one of a mask marking what is *kept*. Two polarities are
    # in the tree because two families of caller legitimately hold different masks -- the Dirichlet
    # eliminations pin a boundary (``linalg``, ``parametrization``, ``heat/signed`` and three
    # kernels in ``smoothing``), while the region solves take "which vertices may move" straight
    # from their public signature -- and inverting one to reach the other costs a ``wp.map`` and an
    # ``(n,)`` buffer per call. What must not happen again is a *third* site inferring the
    # convention from scratch: ``smoothing.gather_free_positions`` was written that way, six lines
    # from a kernel of the opposite sense in a file that already imported this function.
    if fixed_mask[i]:
        return wp.int32(-1)
    return free_map[i]


@wp.func
def selected_row(
    selected_mask: wp.array[wp.bool], index_map: wp.array[wp.int32], i: wp.int32
) -> wp.int32:
    # ``free_row`` for a mask of the opposite sense: the compact row index element ``i`` occupies in
    # the reduced system, or ``-1`` when the mask does not keep it. See ``free_row`` for why there
    # are two and for the contract on ``index_map`` (which is ``array.mask_to_compact_ranks``'s
    # output, exactly as ``free_map`` is ``linalg.free_partition``'s -- the same array, built from
    # the two complementary masks).
    #
    # Two of ``smoothing``'s region-solve kernels ask this of *two different* partitions at once --
    # the free vertices and the wider set of rows the least-squares system carries -- so having one
    # name for the question is what keeps those readable; and asking it once per neighbour replaces
    # a mask test followed by a separate map read.
    if not selected_mask[i]:
        return wp.int32(-1)
    return index_map[i]


@wp.func
def interior_kept(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    fixed_mask: wp.array[wp.bool],
    i: wp.int32,
) -> wp.int32:
    # How many of row ``i``'s entries survive the elimination: those whose column is free. Shared
    # by the two extractions' counting passes (``interior_row_counts``,
    # ``interior_partition_counts``).
    kept = wp.int32(0)
    for e in range(offsets[i], offsets[i + 1]):
        if not fixed_mask[columns[e]]:
            kept += 1
    return kept


@wp.func
def interior_rhs_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: wp.array2d[wp.float64],
    scale: wp.float64,
    load: wp.array2d[wp.float64],
    i: wp.int32,
    ri: wp.int32,
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Column ``ri`` of the reduced right-hand side, from free row ``i`` of ``Q``: the load's entry
    # less the pinned columns' ``scale * Q_ij bc_j``, for every right-hand-side row. A row past
    # ``load``'s or ``fixed_values``' own row count reads that array as zero. Shared by both
    # extractions (``interior_row_counts``, ``interior_partition_system``).
    for c in range(out_rhs.shape[0]):
        acc = wp.float64(0.0)
        if c < load.shape[0]:
            acc = load[c, i]
        if c < fixed_values.shape[0]:
            for e in range(offsets[i], offsets[i + 1]):
                j = columns[e]
                if fixed_mask[j]:
                    acc -= scale * values[e] * fixed_values[c, j]
        out_rhs[c, ri] = acc


# Rows a block of the free-free extraction's kernels holds, a row a lane on CUDA.
INTERIOR_ROWS_PER_BLOCK = 256


@wp.func
def interior_kept_block(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    fixed_mask: wp.array[wp.bool],
    i: wp.int32,
    lane: wp.int32,
) -> wp.int32:
    # ``interior_kept`` by a whole block, its lanes striding the row (every lane must call it).
    kept = wp.int32(0)
    for e in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        if not fixed_mask[columns[e]]:
            kept += 1
    return block_sum(kept)


@wp.func
def interior_rhs_row_block(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: wp.array2d[wp.float64],
    scale: wp.float64,
    load: wp.array2d[wp.float64],
    i: wp.int32,
    ri: wp.int32,
    lane: wp.int32,
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # ``interior_rhs_row`` by a whole block, its lanes striding the row and the block summing the
    # pinned columns in a fixed tree, lane 0 writing (every lane must call it).
    for c in range(out_rhs.shape[0]):
        acc = wp.float64(0.0)
        if c < fixed_values.shape[0]:
            for e in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
                j = columns[e]
                if fixed_mask[j]:
                    acc -= scale * values[e] * fixed_values[c, j]
        acc = block_sum(acc)
        if lane == 0:
            seed = wp.float64(0.0)
            if c < load.shape[0]:
                seed = load[c, i]
            out_rhs[c, ri] = seed + acc


@wp.func
def emit_interior_row_block(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    scale: wp.float64,
    i: wp.int32,
    slot: wp.int32,
    rank_table: wp.array[Any],
    rank_of: wp.Function,
    lane: wp.int32,
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
):
    # ``emit_interior_row`` by a whole block (every lane must call it): the row a chunk of
    # ``block_dim`` entries at a time, each kept entry's slot its rank among the chunk's kept
    # entries (an exclusive scan) past those of the chunks before, so the row keeps its order.
    lanes = wp.block_dim()
    end = offsets[i + 1]
    for chunk in range(offsets[i], end, lanes):
        e = chunk + lane
        kept = wp.int32(0)
        if e < end:
            if not fixed_mask[columns[e]]:
                kept = 1
        before = wp.untile(wp.tile_scan_exclusive(wp.tile(kept)))
        if kept != 0:
            out_columns[slot + before] = rank_of(rank_table, columns[e])
            out_values[slot + before] = scale * values[e]
        slot += block_sum(kept)


@wp.func
def interior_row_count_and_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: wp.array2d[wp.float64],
    scale: wp.float64,
    load: wp.array2d[wp.float64],
    i: wp.int32,
    ri: wp.int32,
    out_counts: wp.array[wp.int32],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Free row ``i``'s (compact row ``ri``) count and right-hand side in ``interior_row_counts``,
    # by one lane.
    #
    # A free row every one of whose columns is pinned counts 0 and becomes an *empty* CSR row, not a
    # dropped one -- the scan of these counts is what keeps the row offsets in step.
    out_counts[ri] = interior_kept(offsets, columns, fixed_mask, i)
    # The thread owns row ``ri`` of ``out_rhs`` exclusively, so a single register accumulator and
    # write suffice per right-hand-side column.
    #
    # The nest re-reads the row once per right-hand side, which looks like an ``n_rhs``-fold read
    # amplification worth inverting (scan the row once into an ``n_rhs``-wide register vector).
    # Measured before building it, and it is not: this kernel is launched **once** per
    # ``min_quad_with_fixed``, one of fifteen launches in the solve, and ``n_rhs`` is **1** for the
    # default call -- so the inner loop runs a single iteration and there is nothing to invert. The
    # callers that pass more are ``lscm`` (2) and a vector-valued ``harmonic`` (3), where the whole
    # pass is still one launch against a conjugate-gradient solve whose iteration count dominates. A
    # register vector would also need a compile-time ``MAX_RHS`` cap, which is a new documented
    # limitation bought for nothing.
    #
    # ``scale`` multiplies ``Q`` as it is read (``-1`` extracts ``-L`` from a Laplacian with no
    # negated copy; ``scale * q`` is exact for that), and an optional full-length ``load`` row --
    # ``None`` reads as zero rows -- seeds the accumulator, so a linear term ``b_u`` is compacted in
    # this launch rather than a scatter of its own. A right-hand-side row past either array's own
    # row count reads that array as zero.
    interior_rhs_row(
        offsets, columns, values, fixed_mask, fixed_values, scale, load, i, ri, out_rhs
    )


@wp.kernel
def interior_row_counts(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    fixed_values: wp.array2d[wp.float64],
    scale: wp.float64,
    load: wp.array2d[wp.float64],
    out_counts: wp.array[wp.int32],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Pass 1 of the free-free extraction: one thread per row ``i`` of the operator ``Q`` (positive
    # semi-definite), counting the entries of that row whose column also survives. The right-hand
    # side rides along in the same launch because it needs the identical row scan and depends on
    # nothing this pass produces: fixed-column contributions move to the right-hand side as
    # ``Q_uu x_u = -Q_ub bc``, generalized to ``n_rhs`` columns (``fixed_values`` is
    # ``(n_rhs, n_dofs)``, ``out_rhs`` is ``(n_rhs, n_free)``). Assembled in float64: the biharmonic
    # (k > 1) operator squares the Laplacian condition number, beyond float32 CG's reach; LSCM's
    # coupled u/v system is likewise ill-conditioned.
    #
    # Launch tiled over ``INTERIOR_ROWS_PER_BLOCK`` rows a block: a lane takes its rows in order; a
    # long row (``csr_row_long``) is left to the whole block after, so a block without one pays
    # one reduction.
    blk, lane = wp.tid()
    first, rows = block_chunk(fixed_mask.shape[0], blk, INTERIOR_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        ri = free_row(fixed_mask, free_map, i)
        if ri < 0:
            continue
        if csr_row_long(offsets, i):
            found = 1
            continue
        interior_row_count_and_rhs(
            offsets, columns, values, fixed_mask, fixed_values, scale, load, i, ri, out_counts,
            out_rhs,
        )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        ri = free_row(fixed_mask, free_map, i)
        if ri >= 0 and csr_row_long(offsets, i):
            kept = interior_kept_block(offsets, columns, fixed_mask, i, lane)
            if lane == 0:
                out_counts[ri] = kept
            interior_rhs_row_block(
                offsets, columns, values, fixed_mask, fixed_values, scale, load, i, ri, lane,
                out_rhs,
            )  # fmt: skip


@wp.func
def mapped_rank(free_map: wp.array[wp.int32], j: wp.int32) -> wp.int32:
    # ``emit_interior_row``'s ``rank_of`` from a free map handed in (``interior_system_csr``).
    return free_map[j]


@wp.func
def scanned_rank(ranks: wp.array[wp.vec2i], j: wp.int32) -> wp.int32:
    # ``emit_interior_row``'s ``rank_of`` from the scanned ``(free, kept)`` pairs
    # (``interior_partition_system``).
    return ranks[j][0]


@wp.func
def emit_interior_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    scale: wp.float64,
    i: wp.int32,
    slot: wp.int32,
    rank_table: wp.array[Any],
    rank_of: wp.Function,
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
):
    # Row ``i``'s surviving (unpinned-column) entries of ``Q``, scaled, from ``slot`` on, each
    # column renumbered to ``rank_of(rank_table, j)``: the emit both extractions share.
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        if not fixed_mask[j]:
            out_columns[slot] = rank_of(rank_table, j)
            out_values[slot] = scale * values[e]
            slot += 1


@wp.kernel
def interior_system_csr(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    scale: wp.float64,
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # Pass 2: one thread per row ``i`` of ``Q``, writing that row's surviving entries into the slots
    # ``row_offsets`` reserved for it. ``row_offsets`` is read-only here -- it is the exclusive scan
    # of ``interior_row_counts``' output, produced by ``counts_to_offsets`` in the wrapper -- so the
    # destination range is exactly the right size and no two threads overlap. Column order is
    # inherited from ``Q``'s row and ``free_map`` is monotone non-decreasing, so the emitted row is
    # column-sorted by construction -- which is the whole reason this can skip a triplet sort.
    #
    # Launch tiled over ``INTERIOR_ROWS_PER_BLOCK`` rows a block: a lane takes its rows in order; a
    # long row (``csr_row_long``) is left to the whole block after, so a block without one pays
    # one reduction.
    blk, lane = wp.tid()
    first, rows = block_chunk(fixed_mask.shape[0], blk, INTERIOR_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        ri = free_row(fixed_mask, free_map, i)
        if ri < 0:
            continue
        if csr_row_long(offsets, i):
            found = 1
            continue
        emit_interior_row(
            offsets, columns, values, fixed_mask, scale, i, row_offsets[ri], free_map,
            mapped_rank, out_columns, out_values,
        )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        ri = free_row(fixed_mask, free_map, i)
        if ri >= 0 and csr_row_long(offsets, i):
            emit_interior_row_block(
                offsets, columns, values, fixed_mask, scale, i, row_offsets[ri], free_map,
                mapped_rank, lane, out_columns, out_values,
            )  # fmt: skip


@wp.kernel
def interior_partition_counts(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    fixed_mask: wp.array[wp.bool],
    out_ranks: wp.array[wp.vec2i],
) -> None:
    # The partitioned extraction's one counting pass (``linalg.partitioned_interior_system``): row
    # ``i``'s ``(free, kept entries)`` at ``i + 1`` of an ``n + 1`` buffer, ``(0, 0)`` for a pinned
    # row, and the leading zero by thread 0, so one inclusive scan of it holds every row's free
    # rank and output offset at once, and its last element both totals for one read.
    #
    # Launch tiled over ``INTERIOR_ROWS_PER_BLOCK`` rows a block: a lane takes its rows in order; a
    # long row (``csr_row_long``) is left to the whole block after, so a block without one pays
    # one reduction.
    blk, lane = wp.tid()
    first, rows = block_chunk(fixed_mask.shape[0], blk, INTERIOR_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if i == 0:
            out_ranks[0] = wp.vec2i(0, 0)
        if fixed_mask[i]:
            out_ranks[i + 1] = wp.vec2i(0, 0)
        elif csr_row_long(offsets, i):
            found = 1
        else:
            out_ranks[i + 1] = wp.vec2i(1, interior_kept(offsets, columns, fixed_mask, i))
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if not fixed_mask[i] and csr_row_long(offsets, i):
            kept = interior_kept_block(offsets, columns, fixed_mask, i, lane)
            if lane == 0:
                out_ranks[i + 1] = wp.vec2i(1, kept)


@wp.kernel
def interior_partition_system(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    ranks: wp.array[wp.vec2i],
    fixed_values: wp.array2d[wp.float64],
    scale: wp.float64,
    load: wp.array2d[wp.float64],
    out_free_map: wp.array[wp.int32],
    out_row_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Everything the extraction writes, from the scanned ``(free, kept)`` pairs: row ``i``'s free
    # rank (``free_partition``'s map, defined at pinned rows too), and for a free row its CSR
    # offset, its surviving entries -- columns renumbered by the scan, so no map is read back --
    # and its right-hand-side column (``interior_rhs_row``). ``interior_system_csr`` emits the same
    # entries from a map it is handed; the two differ only in where a column's rank comes from
    # (``emit_interior_row``'s ``rank_of``).
    #
    # Launch tiled over ``INTERIOR_ROWS_PER_BLOCK`` rows a block: a lane takes its rows in order; a
    # long row (``csr_row_long``) is left to the whole block after, so a block without one pays
    # one reduction.
    blk, lane = wp.tid()
    n = fixed_mask.shape[0]
    first, rows = block_chunk(n, blk, INTERIOR_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        rank = ranks[i][0]
        out_free_map[i] = rank
        if i == 0:
            out_row_offsets[ranks[n][0]] = ranks[n][1]
        if fixed_mask[i]:
            continue
        out_row_offsets[rank] = ranks[i][1]
        if csr_row_long(offsets, i):
            found = 1
            continue
        emit_interior_row(
            offsets, columns, values, fixed_mask, scale, i, ranks[i][1], ranks, scanned_rank,
            out_columns, out_values,
        )  # fmt: skip
        interior_rhs_row(
            offsets, columns, values, fixed_mask, fixed_values, scale, load, i, rank, out_rhs
        )
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if not fixed_mask[i] and csr_row_long(offsets, i):
            emit_interior_row_block(
                offsets, columns, values, fixed_mask, scale, i, ranks[i][1], ranks, scanned_rank,
                lane, out_columns, out_values,
            )  # fmt: skip
            interior_rhs_row_block(
                offsets, columns, values, fixed_mask, fixed_values, scale, load, i, ranks[i][0],
                lane, out_rhs,
            )  # fmt: skip


@wp.kernel
def masks_differ(
    first: wp.array[wp.bool],
    second: wp.array[wp.bool],
    base: wp.int32,
    out_flag: wp.array[wp.int32],
) -> None:
    # Whether a call pins the degrees of freedom a precomputed factorization was built for: the
    # reduced operator depends on the pinned *set*, so two masks are compared, not index lists.
    # A difference writes ``base + 1``, every writer the same value; a word left at or below
    # ``base`` reads as equal, so the caller reuses it above a fresh base instead of clearing it
    # (``linalg._StampedFlag``).
    i = wp.int32(wp.tid())
    if first[i] != second[i]:
        out_flag[0] = base + 1


@wp.kernel
def free_coordinates(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    coordinates: wp.array[wp.vec3],
    out_coordinates: wp.array[wp.vec3],
) -> None:
    # The reduced system's row positions, for its factorization's geometric ordering: degree of
    # freedom ``i`` sits at point ``i mod n_points``, so stacked components (``lscm``'s ``[u; v]``)
    # share their vertex's position with no stacked copy of the coordinates.
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    out_coordinates[ri] = coordinates[i % coordinates.shape[0]]


@wp.kernel
def stacked_positions(
    vertices: wp.array[wp.vec3], n_scalars: wp.int32, out_positions: wp.array[wp.vec3]
) -> None:
    # The vertex of every row of a ``[vector system; scalar system x n_scalars]`` stack, for a
    # factorization's geometric ordering: the vector system's interleaved two rows (a
    # ``wp.mat22d`` operator's scalar expansion), then one row per scalar system.
    v = wp.int32(wp.tid())
    n = vertices.shape[0]
    position = vertices[v]
    out_positions[2 * v] = position
    out_positions[2 * v + 1] = position
    for k in range(n_scalars):
        out_positions[2 * n + k * n + v] = position


# Entries from which a CSR row is *long*: on CUDA the row-a-thread solver kernels (Chebyshev steps,
# the conjugate-gradient mat-vec) skip it under a nonzero ``skip_long`` and blocks sum it instead, a
# piece of ``CSR_LONG_ROW_PIECE`` entries each (``long_row_piece_total``). A row a thread is one
# thread's chain of dependent loads, so a hub row (a fan's apex, 41 k entries) held every step for
# as long as the rest of the step by a factor of ~1 000. Mesh and ``warp.fem`` rows stay far below
# it, so the threshold was not swept. ``linalg._LongRows`` decides on the host, from one count read
# per listing, whether a round launches the long-row kernels; launching them every step with no read
# instead (their blocks returning at once with none listed) cost 1.0-1.31x where no row is long
# (geomean 1.09-1.12 over 13 meshes and the solver rows of six benchmark files) against the read's
# 0.98-1.13x (geomean 1.01-1.02; the worst a 1.7 ms call on 1 024 components): a small solve is
# mostly steps, and the read is once a solve.
# ``heat_geodesic`` on ``fan_hub``: 0.009x, the same 1 005 rounds.
CSR_LONG_ROW_ENTRIES = wp.int32(1024)


# Entries of a long row one block sums: its piece. A row a block put a 41 k-entry hub row's
# 820 KB of values, columns and gathers through one SM, which held a launch at 49-55 us (the
# thread-a-row step beside it: 5.5 us), and four interleaved accumulators a lane took it only to
# 37 us; pieces spread the row over SMs and a ticket folds them.
CSR_LONG_ROW_PIECE = wp.int32(1024)


@wp.func
def csr_row_long(offsets: wp.array[wp.int32], row: wp.int32) -> wp.bool:
    return offsets[row + 1] - offsets[row] >= CSR_LONG_ROW_ENTRIES


@wp.func
def long_row_piece_count(offsets: wp.array[wp.int32], row: wp.int32) -> wp.int32:
    return (offsets[row + 1] - offsets[row] + CSR_LONG_ROW_PIECE - 1) // CSR_LONG_ROW_PIECE


@wp.kernel
def list_long_rows(
    offsets: wp.array[wp.int32], out_pieces: wp.array[wp.vec3i], out_count: wp.array[wp.int32]
) -> None:
    # The pieces of the rows ``csr_row_long`` names, a row's pieces in consecutive slots taken in
    # arrival order: ``(row, its first piece's slot, piece index)``. A row's sum does not depend
    # on its slots. Launch over the row count; ``out_pieces`` holds ``columns.size //
    # CSR_LONG_ROW_PIECE + columns.size // CSR_LONG_ROW_ENTRIES``, which bounds their number, so a
    # row is listed whole or (past that, never) not at all.
    row = wp.int32(wp.tid())
    if csr_row_long(offsets, row):
        n_pieces = long_row_piece_count(offsets, row)
        first = wp.atomic_add(out_count, 0, n_pieces)
        if first + n_pieces <= out_pieces.shape[0]:
            for k in range(n_pieces):
                out_pieces[first + k] = wp.vec3i(row, first, k)


@wp.func
def listed_long_pieces(pieces: wp.array[wp.vec3i], count: wp.array[wp.int32]) -> wp.int32:
    # How many of ``list_long_rows``' slots hold a piece.
    return wp.min(count[0], pieces.shape[0])


@wp.func
def long_row_piece_total(
    piece: wp.vec3i,
    x_offset: wp.int32,
    slot_offset: wp.int32,
    lane: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: Any,
    x: Any,
    partials: wp.array[wp.float64],
    tickets: wp.array[wp.int32],
) -> tuple[wp.int32, Any]:
    # A block's piece of a long row against one column of ``x`` (every lane must call it): its
    # lanes stride the piece and the block sums them, lane 0 stores the sum in its slot of
    # ``partials`` (``slot_offset`` past the column's) and takes a ticket on the row's counter.
    # The block drawing the row's last ticket folds its pieces in a fixed tree and resets the
    # counter: returns 1 and the row's sum in that block's lane 0, 0 elsewhere. A fixed order, so
    # the result repeats; it differs from the one-thread order in its last bits. Block-collective:
    # every lane of every block reaches it.
    row = piece[0]
    first = slot_offset + piece[1]
    start = offsets[row] + piece[2] * CSR_LONG_ROW_PIECE
    end = wp.min(start + CSR_LONG_ROW_PIECE, offsets[row + 1])
    partial = x.dtype(0.0)
    for k in range(start + lane, end, wp.block_dim()):
        partial += x.dtype(values[k]) * x[x_offset + columns[k]]
    total = block_sum(partial)
    n_pieces = long_row_piece_count(offsets, row)
    last = wp.int32(0)
    if lane == 0:
        partials[first + piece[2]] = wp.float64(total)
        thread_fence()
        if wp.atomic_add(tickets, first, 1) == n_pieces - 1:
            last = 1
    # Every lane of the last block folds: its lanes load the pieces at once (one lane walking them
    # waited on each load in turn, 9.5 us a launch at 41 pieces) and the block sums them in a fixed
    # tree.
    last = block_sum(last)
    if last != 0:
        thread_fence()
        folded = wp.float64(0.0)
        for k in range(lane, n_pieces, wp.block_dim()):
            folded += partials[first + k]
        total = x.dtype(block_sum(folded))
        if lane == 0:
            tickets[first] = 0
    return last, total


# Launched at its own width (``linalg._LONG_ROW_LANES``), so in a module of its own (§12.4).
@wp.kernel(module="unique")
def csr_long_rows_matvec(
    stride: wp.int32,
    launch_blocks: wp.int32,
    pieces: wp.array[wp.vec3i],
    count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    u: wp.array[wp.Float],
    partials: wp.array[wp.float64],
    tickets: wp.array[wp.int32],
    out_w: wp.array[wp.Float],
) -> None:
    # ``w = A u`` at the long rows ``list_long_rows`` listed, a block a piece and column
    # (``long_row_piece_total``). Launch tiled over ``(n_columns, launch_blocks)``; the blocks
    # stride the listed pieces, so a launch with none listed returns at once (block uniform).
    c, blk, lane = wp.tid()
    base = c * stride
    slots = c * pieces.shape[0]
    for b in range(blk, listed_long_pieces(pieces, count), launch_blocks):
        piece = pieces[b]
        last, total = long_row_piece_total(
            piece, base, slots, lane, offsets, columns, values, u, partials, tickets
        )
        if last != 0:
            out_w[base + piece[0]] = total


@wp.func
def csr_row_dot(
    row: wp.int32,
    x_offset: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    x: wp.array[wp.Float],
) -> wp.Float:
    # One CSR row against one column of ``x``: the multigrid cycle's mat-vec and power iteration,
    # the conjugate-gradient rounds and ``smoothing.diffused_scalar``, which differ only in what
    # they do with the result -- and the row bound comes from ``offsets`` alone, so none depends on
    # the matrix's ``nnz`` field being fresh.
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


@wp.func
def csr_row_diagonal(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    i: wp.int32,
) -> tuple[wp.Float, wp.Float]:
    # Row ``i`` of a CSR operator split into its diagonal entry (0 where the row has none) and
    # ``sum_{j != i} |A_ij|``: what a Jacobi scaling and a Gershgorin disc both read off a row, so a
    # launch that wants both walks the row once.
    diagonal = values.dtype(0.0)
    off_sum = values.dtype(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        if columns[e] == i:
            diagonal += values[e]
        else:
            off_sum += wp.abs(values[e])
    return diagonal, off_sum


@wp.func
def dominance_ratio(diagonal: wp.Float, off_sum: wp.Float) -> wp.Float:
    # A row's Gershgorin ratio ``sum_{j != i} |A_ij| / |A_ii|``, 0 for a zero diagonal -- see
    # ``jacobi_dominance_bound`` for why both halves of that are load-bearing.
    magnitude = wp.abs(diagonal)
    if magnitude == type(magnitude)(0.0):
        return type(magnitude)(0.0)
    return off_sum / magnitude


@wp.func
def jacobi_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    i: wp.int32,
) -> tuple[wp.Float, wp.Float]:
    # Row ``i``'s Jacobi inverse diagonal (``array.inverse_or_one``, so an empty row scales by 1)
    # and its Gershgorin ratio, from one walk of the row: every kernel that scales by the Jacobi
    # diagonal or fits an interval to the discs reads them here, so the diagonal's convention is
    # written once. A caller wanting one of the two leaves the other dead, and it compiles away.
    diagonal, off_sum = csr_row_diagonal(offsets, columns, values, i)
    return inverse_or_one(diagonal), dominance_ratio(diagonal, off_sum)


@wp.func
def csr_row_diagonal_block(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    i: wp.int32,
    lane: wp.int32,
) -> tuple[wp.Float, wp.Float]:
    # ``csr_row_diagonal`` by a whole block, its lanes striding the row and the block summing the
    # diagonal and the off-diagonal magnitudes in a fixed tree (every lane must call it; every lane
    # gets the sums).
    diagonal = values.dtype(0.0)
    off_sum = values.dtype(0.0)
    for e in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        if columns[e] == i:
            diagonal += values[e]
        else:
            off_sum += wp.abs(values[e])
    return block_sum(diagonal), block_sum(off_sum)


@wp.func
def jacobi_row_block(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    i: wp.int32,
    lane: wp.int32,
) -> tuple[wp.Float, wp.Float]:
    # ``jacobi_row`` by a whole block (``csr_row_diagonal_block``; every lane must call it).
    diagonal, off_sum = csr_row_diagonal_block(offsets, columns, values, i, lane)
    return inverse_or_one(diagonal), dominance_ratio(diagonal, off_sum)


@wp.kernel
def jacobi_dominance_bound(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    write_inverse: wp.int32,
    rows_per_block: wp.int32,
    out_inverse_diagonal: wp.array[wp.float64],
    out_bound: wp.array[wp.float64],
) -> None:
    # The maximum over the rows of a CSR operator of the Gershgorin ratio
    # ``sum_{j != i} |A_ij| / A_ii``, and with ``write_inverse`` set each row's Jacobi inverse
    # diagonal (``array.inverse_or_one``), from one walk of each row (``jacobi_row``); unset,
    # ``out_inverse_diagonal`` is never touched and may be ``None``. The maximum is what
    # ``linalg._offdiagonal_dominance`` reads to decide whether a system reaches a multigrid
    # hierarchy -- see ``linalg.CG_PROBE_ITERATIONS`` for the 29 systems behind that gate -- and
    # what the Jacobi-Chebyshev interval is fitted to, which also wants the scaling.
    #
    # Lanes stride the block's ``rows_per_block`` rows (``reduce.spanned_blocks_1d``: a row is a
    # walk, so on CUDA a block spans as few as fill the device) by ``wp.block_dim()`` (right on the
    # CPU device too, where a block is one lane), and lane 0 commits the block's maximum with one
    # ``atomic_max`` into ``out_bound[0]``, which the caller seeds below any ratio. A maximum is
    # exact in any order, so the bound repeats bit for bit, and no per-row ratio is stored.
    #
    # A row whose diagonal is *absent or zero* is a free vertex no face refers to: the least-squares
    # system does not constrain it, so it contributes no row to the operator's *conditioning* and
    # ratio 0 keeps it out of the maximum. Dividing by it instead would make the reduction read
    # ``inf`` on every scan mesh, which is what a first version of this did.
    #
    # The magnitude is taken on **both** sides, which is not cosmetic: a Laplacian written in the
    # negative-semi-definite convention has a negative diagonal, and a second version of this
    # rejected every such row as "zero or negative" and reduced to a flat **0.000** -- a number
    # that reads like a well-conditioned operator and silently declines the gate. Caught on
    # ``min_quad_with_fixed`` driven with a raw ``cotmatrix``.
    #
    # A long row (``csr_row_long``, a fan's hub) is left to the whole block after the lanes' pass
    # (``jacobi_row_block``), so a block without one pays one reduction.
    block, lane = wp.tid()
    base, count = block_chunk(offsets.shape[0] - 1, block, rows_per_block)
    if count <= 0:
        return
    local = wp.float64(-1.0e308)
    found = wp.int32(0)
    for k in range(lane, count, wp.block_dim()):
        i = base + k
        if csr_row_long(offsets, i):
            found = 1
            continue
        inverse, ratio = jacobi_row(offsets, columns, values, i)
        if write_inverse != 0:
            out_inverse_diagonal[i] = inverse
        local = wp.max(local, ratio)
    if block_sum(found) != 0:
        for k in range(count):
            i = base + k
            if csr_row_long(offsets, i):
                inverse, ratio = jacobi_row_block(offsets, columns, values, i, lane)
                if lane == 0:
                    if write_inverse != 0:
                        out_inverse_diagonal[i] = inverse
                    local = wp.max(local, ratio)
    bound = block_max(local)
    if lane == 0:
        wp.atomic_max(out_bound, 0, bound)


@wp.kernel
def jacobi_inverse_diagonal(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    out_inverse_diagonal: wp.array[wp.Float],
) -> None:
    # The Jacobi inverse diagonal of a scalar CSR operator in one launch, where
    # ``wps.bsr_get_diag`` followed by a ``wp.map`` of ``array.inverse_or_one`` is two and an
    # intermediate buffer. The row's ratio is dead code here and compiles away. Launch tiled over
    # ``INTERIOR_ROWS_PER_BLOCK`` rows a block; a long row (``csr_row_long``) is left to the whole
    # block after (``jacobi_row_block``), so a block without one pays one reduction.
    blk, lane = wp.tid()
    first, rows = block_chunk(offsets.shape[0] - 1, blk, INTERIOR_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = first + r
        if csr_row_long(offsets, i):
            found = 1
            continue
        inverse, _ratio = jacobi_row(offsets, columns, values, i)
        out_inverse_diagonal[i] = inverse
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = first + r
        if csr_row_long(offsets, i):
            inverse, _ratio = jacobi_row_block(offsets, columns, values, i, lane)
            if lane == 0:
                out_inverse_diagonal[i] = inverse


@wp.func
def refresh_pooled_entries(
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    narrow: wp.int32,
    i: wp.int32,
    start: wp.int32,
    end: wp.int32,
    step: wp.int32,
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_narrowed: wp.array[wp.float32],
) -> wp.vec2d:
    # Entries ``start``, ``start + step``, ... below ``end`` of row ``i`` of a pooled
    # conjugate-gradient state's refresh (``linalg._cached_solver``): copied into the state's own
    # storage, with the ``float32`` copy the rounds read (``narrow``). Returns their share of the
    # row's diagonal and ``sum_{j != i} |A_ij|`` (``csr_row_diagonal``), for ``refresh_pooled_row``.
    sums = wp.vec2d()
    for e in range(start, end, step):
        value = values[e]
        column = columns[e]
        out_columns[e] = column
        out_values[e] = value
        if narrow != 0:
            out_narrowed[e] = wp.float32(value)
        if column == i:
            sums[0] = sums[0] + value
        else:
            sums[1] = sums[1] + wp.abs(value)
    return sums


@wp.func
def refresh_pooled_row(
    derive: wp.int32, i: wp.int32, sums: wp.vec2d, out_inverse_diagonal: wp.array[wp.float64]
) -> wp.float64:
    # Row ``i``'s refresh from its whole sums (``refresh_pooled_entries``): what the state derived
    # from its operator when it was built -- the Jacobi inverse diagonal (``derive >= 1``) -- and
    # the row's Gershgorin ratio, which the Jacobi-Chebyshev interval is refitted to
    # (``derive == 2``; ``0`` otherwise). Each is the arithmetic of the kernel that built it
    # (``jacobi_inverse_diagonal`` / ``jacobi_dominance_bound``), so a refreshed state holds what a
    # new one would.
    ratio = wp.float64(0.0)
    if derive != 0:
        out_inverse_diagonal[i] = inverse_or_one(sums[0])
        ratio = dominance_ratio(sums[0], sums[1])
    return ratio


@wp.kernel
def refresh_pooled_operator(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    narrow: wp.int32,
    derive: wp.int32,
    width: wp.int32,
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_narrowed: wp.array[wp.float32],
    out_inverse_diagonal: wp.array[wp.float64],
    out_bound: wp.array[wp.float64],
) -> None:
    # A pooled state's refresh, ``width`` rows a block (the arrays a flag leaves off may be
    # ``None``). With a non-empty ``out_bound`` (``derive == 2``) the rows' largest Gershgorin
    # ratio is committed into ``out_bound[0]``, which the caller holds at ``-inf``
    # (``chebyshev_steps`` leaves it there), so the refit needs no reduction launch of its own: the
    # ``atomic_max`` commit of ``jacobi_dominance_bound``, a maximum being exact in any order.
    # A lane refreshes its rows in order; a long row (``csr_row_long``) is left to the whole block
    # after, its lanes striding the entries and the block summing them in a fixed tree, so a block
    # without one pays one reduction. Launch tiled.
    block, lane = wp.tid()
    n = offsets.shape[0] - 1
    base, rows = block_chunk(n, block, width)
    if rows <= 0:
        return
    local = wp.float64(-1.0e308)
    found = wp.int32(0)
    for k in range(lane, rows, wp.block_dim()):
        i = base + k
        start = offsets[i]
        end = offsets[i + 1]
        out_offsets[i] = start
        if i == n - 1:
            out_offsets[n] = end
        if csr_row_long(offsets, i):
            found = 1
            continue
        sums = refresh_pooled_entries(
            columns, values, narrow, i, start, end, 1, out_columns, out_values, out_narrowed
        )
        local = wp.max(local, refresh_pooled_row(derive, i, sums, out_inverse_diagonal))
    if block_sum(found) != 0:
        for k in range(rows):
            i = base + k
            if csr_row_long(offsets, i):
                sums = refresh_pooled_entries(
                    columns, values, narrow, i, offsets[i] + lane, offsets[i + 1],
                    wp.block_dim(), out_columns, out_values, out_narrowed,
                )  # fmt: skip
                sums = block_sum(sums)
                if lane == 0:
                    local = wp.max(local, refresh_pooled_row(derive, i, sums, out_inverse_diagonal))
    if out_bound.shape[0] > 0:
        bound = block_max(local)
        if lane == 0:
            wp.atomic_max(out_bound, 0, bound)


@wp.kernel
def scaled_rows_and_bound(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    weight_sums: wp.array[wp.float64],
    rows_per_block: wp.int32,
    out_scaled: wp.array[wp.float64],
    out_scaled_t: wp.array[wp.float64],
    out_bound: wp.array[wp.float64],
) -> None:
    # In one walk of each row of a symmetric CSR ``L``: the values of ``D^-1 L`` (each row scaled
    # by ``array.inverse_or_one`` of ``D_i``, so a zero weight scales by 1), those of its transpose
    # ``L D^-1`` over the same pattern (each entry scaled by its column's weight: ``L_ij = L_ji``),
    # and the maximum over the rows of ``sum_j |L_ij| / D_i``, the Gershgorin radius plus centre of
    # row ``i`` of ``D^-1 L``, which bounds that operator's spectrum from above.
    # ``linalg.SquaredLaplacianPreconditioner`` fits its polynomial to that bound. With ``D`` the
    # diagonal of ``L`` it is ``1 +`` the ``jacobi_dominance_bound`` ratio; with any other positive
    # ``D`` -- ``sqrt(M)`` for the k = 2 harmonic operator ``L M^-1 L`` -- it is the bound that
    # ratio is not, and an interval fitted short of the spectrum amplifies what it should invert.
    # The block shape (``rows_per_block``) and the ``atomic_max`` commit into the seeded
    # ``out_bound[0]`` are ``jacobi_dominance_bound``'s.
    block, lane = wp.tid()
    base, count = block_chunk(offsets.shape[0] - 1, block, rows_per_block)
    if count <= 0:
        return
    local = wp.float64(-1.0e308)
    for k in range(lane, count, wp.block_dim()):
        i = base + k
        weight = weight_sums[i]
        scale = inverse_or_one(weight)
        total = wp.float64(0.0)
        for e in range(offsets[i], offsets[i + 1]):
            value = values[e]
            total += wp.abs(value)
            out_scaled[e] = value * scale
            out_scaled_t[e] = value * inverse_or_one(weight_sums[columns[e]])
        local = wp.max(local, total / weight)
    bound = block_max(local)
    if lane == 0:
        wp.atomic_max(out_bound, 0, bound)


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


# ``backward_error_exceeds``' verdicts, past ``0`` (verified).
BACKWARD_ERROR_FAILED = wp.int32(1)
BACKWARD_ERROR_UNCONVERGED = wp.int32(2)

# Rows a block of ``backward_error_exceeds`` holds, a row a lane on CUDA.
BACKWARD_ERROR_ROWS = 256


@wp.func
def raise_backward_error_flag(out_flag: wp.array[wp.int32], verdict: wp.int32) -> None:
    # Raise ``backward_error_exceeds``' flag to ``verdict``, the plain read first sparing the atomic
    # once it is set.
    if out_flag[0] < verdict:
        wp.atomic_max(out_flag, 0, verdict)


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
    base: wp.int32,
    out_flag: wp.array[wp.int32],
) -> None:
    # The componentwise backward error of a solution (Oettli and Prager): ``|b - A x|_i`` against
    # ``(|A| |x| + |b|)_i``. Above ``tolerance`` anywhere, the solution is not the system's to
    # within its own entries' size, whatever its residual norm says. Given a conjugate-gradient
    # result's per-column ``residual`` and ``threshold`` (or ``None``), row 0 of each column also
    # flags a column that did not converge, so one four-byte read answers both questions. Both
    # tests are spelled "not within", so a NaN anywhere flags. The flag is raised to ``base`` plus
    # the worst verdict: ``BACKWARD_ERROR_FAILED`` for an entry over the bar,
    # ``BACKWARD_ERROR_UNCONVERGED`` for a column outside its threshold, which takes precedence (a
    # maximum, so the read is the same whatever order the rows land in; the plain read first
    # spares the atomic once it is set). A word left at or below ``base`` reads as verified, so the
    # caller reuses it with a higher ``base`` instead of clearing it (``linalg._StampedFlag``).
    #
    # Launch tiled over ``(blocks of BACKWARD_ERROR_ROWS rows, columns)``: a lane tests its rows in
    # order; a long row (``csr_row_long``) is left to the whole block after, its lanes striding the
    # entries and the block summing them in a fixed tree, so a block without one pays one
    # reduction.
    blk, column, lane = wp.tid()
    offset = column * n
    first, rows = block_chunk(offsets.shape[0] - 1, blk, BACKWARD_ERROR_ROWS)
    if rows <= 0:
        return
    found = wp.int32(0)
    for k in range(lane, rows, wp.block_dim()):
        i = first + k
        if i == 0 and residual.shape[0] > 0:
            if not (residual[column] <= threshold[column]):
                raise_backward_error_flag(out_flag, base + BACKWARD_ERROR_UNCONVERGED)
        if csr_row_long(offsets, i):
            found = 1
            continue
        total, scale = residual_and_scale(offsets, columns, values, solution, rhs, offset, i)
        if not (wp.abs(total) <= tolerance * scale):
            raise_backward_error_flag(out_flag, base + BACKWARD_ERROR_FAILED)
    if block_sum(found) == 0:
        return
    for k in range(rows):
        i = first + k
        if csr_row_long(offsets, i):
            sums = wp.vec2d()
            for e in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
                term = values[e] * solution[offset + columns[e]]
                sums += wp.vec2d(term, wp.abs(term))
            sums = block_sum(sums)
            b = rhs[offset + i]
            if lane == 0 and not (wp.abs(b - sums[0]) <= tolerance * (wp.abs(b) + sums[1])):
                raise_backward_error_flag(out_flag, base + BACKWARD_ERROR_FAILED)


@wp.kernel
def chebyshev_steps(
    bound: wp.array[wp.float64],
    n: wp.int32,
    squared: wp.int32,
    interval: wp.float64,
    cap: wp.float64,
    out_steps: wp.array[wp.vec4d],
) -> None:
    # ``(scale, previous_scale, momentum, step)`` for each launch of a Chebyshev semi-iteration, one
    # thread, from a Gershgorin bound left on the device -- so a preconditioner is built without
    # reading its interval back, which would drain everything queued ahead of it. ``bound[0]`` is
    # ``max_i sum_j |L_ij| / D_i`` for the squared-Laplacian polynomial (``squared`` set) and the
    # largest off-diagonal ratio for the Jacobi-Chebyshev one; ``-inf`` for an empty operator.
    # Every scalar is the host arithmetic ``linalg`` documents at each preconditioner, in the same
    # order, and the steps are ``linalg._chebyshev_steps``': the degree is one more than the
    # length of ``out_steps``, and the first iterate ``source / theta`` is folded into the first
    # two steps' scales (see ``multigrid.chebyshev_step``).
    #
    # The bound is consumed: it is left at ``-inf``, the seed a pooled state's next refresh commits
    # its maximum into (``refresh_pooled_operator_bound``), so a refit clears nothing.
    maximum = bound[0]
    bound[0] = wp.float64(-wp.inf)
    relative = wp.min(interval / wp.float64(wp.max(n, 1)), cap)
    lower = wp.float64(0.0)
    upper = wp.float64(0.0)
    if squared != 0:
        upper = wp.max(maximum, wp.float64(2.0))
        lower = relative * (upper / wp.float64(2.0))
    else:
        dominance = wp.max(maximum, wp.float64(1e-3))
        lower = wp.max(relative, wp.float64(1.0) - dominance)
        upper = wp.float64(1.0) + dominance
    theta = wp.float64(0.5) * (upper + lower)
    delta = wp.float64(0.5) * (upper - lower)
    sigma = theta / delta
    rho = wp.float64(1.0) / sigma
    for index in range(out_steps.shape[0]):
        rho_next = wp.float64(1.0) / (wp.float64(2.0) * sigma - rho)
        scale = wp.float64(1.0)
        previous_scale = wp.float64(1.0)
        if index == 0:
            scale = wp.float64(1.0) / theta
            previous_scale = wp.float64(0.0)
        elif index == 1:
            previous_scale = wp.float64(1.0) / theta
        out_steps[index] = wp.vec4d(
            scale, previous_scale, rho_next * rho, wp.float64(2.0) * rho_next / delta
        )
        rho = rho_next


@wp.func
def shift_factor(
    scale: wp.float64, edge_sum_and_count: wp.array[wp.float64], from_edges: wp.int32
) -> wp.float64:
    # The factor ``s`` of a shifted system ``s A + M``: ``scale`` itself, or with ``from_edges`` set
    # ``scale`` times the default timestep ``h ** 2``, ``h`` the mean unique-edge length from
    # ``heat.upper_edge_length_sum_and_count``'s two sums (``0`` for no edges), formed on the device
    # so the host never waits for them; ``scale`` is then the sign. Unset, ``edge_sum_and_count``
    # is never read and may be ``None``. Shared by ``shifted_system_values`` and the heat systems'
    # row kernels (``kernels/heat.heat_system_rows`` / ``vector_heat_system_rows``), which form
    # ``A``'s row in the same thread instead of reading it.
    factor = scale
    if from_edges != 0:
        h = wp.float64(0.0)
        if edge_sum_and_count[1] > wp.float64(0.0):
            h = edge_sum_and_count[0] / edge_sum_and_count[1]
        factor = scale * (h * h)
    return factor


@wp.func
def shifted_scalar_entry(
    value: wp.float64, factor: wp.float64, diagonal: wp.bool, mass: wp.float64
) -> wp.float64:
    # One entry of ``factor * A + M`` for scalar blocks: the mass joins the diagonal entry.
    shifted = factor * value
    if diagonal:
        shifted = shifted + mass
    return shifted


@wp.func
def shifted_block_entry(
    value: wp.mat22d, factor: wp.float64, diagonal: wp.bool, mass: wp.float64
) -> wp.mat22d:
    # ``shifted_scalar_entry`` for ``wp.mat22d`` blocks: a vertex's two tangent unknowns share its
    # area weight, so its diagonal block gains ``mass * I``, formed here as the mass is read.
    shifted = factor * value
    if diagonal:
        shifted = shifted + wp.mat22d(mass, wp.float64(0.0), wp.float64(0.0), mass)
    return shifted


def _shifted_system_values_kernel(dtype: Any) -> Any:
    """``shifted_system_values`` for one block dtype: ``wp.float64`` or ``wp.mat22d``."""
    entry = shifted_block_entry if dtype == wp.mat22d else shifted_scalar_entry

    def shifted_system_values(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[Any],
        mass: wp.array[wp.float64],
        scale: wp.float64,
        edge_sum_and_count: wp.array[wp.float64],
        from_edges: wp.int32,
        negate: wp.int32,
        out_values: wp.array[Any],
        out_negated: wp.array[Any],
    ) -> None:
        # One row of ``scale * A + M``, written over ``A``'s own pattern -- implicit fairing's
        # ``M - lamb L`` -- where ``warp.sparse.bsr_axpy`` against a ``bsr_diag`` would merge two
        # patterns, sorting both, for a result whose pattern is ``A``'s: every vertex a face
        # references has a diagonal entry (``cotmatrix`` and ``connection_laplacian`` keep every
        # triplet), and one no face references has no mass. The heat systems ``M - t L`` and
        # ``M + t L_connection`` take the same entries from ``kernels/heat.heat_system_rows`` and
        # ``vector_heat_system_rows``, which form ``A``'s row in the same thread instead.
        # With ``negate`` set, ``-A`` rides along into ``out_negated`` in the same pass;
        # unset, ``out_negated`` is never touched and may be ``None``. Scalar ``float64`` or
        # ``wp.mat22d`` blocks; ``M`` is the scalar lumped mass at either (``shifted_block_entry``).
        # The factor is ``shift_factor``'s.
        row = wp.int32(wp.tid())
        factor = shift_factor(scale, edge_sum_and_count, from_edges)
        m = mass[row]
        for e in range(offsets[row], offsets[row + 1]):
            value = values[e]
            out_values[e] = entry(value, factor, columns[e] == row, m)
            if negate != 0:
                out_negated[e] = -value

    for name in ("values", "out_values", "out_negated"):
        shifted_system_values.__annotations__[name] = wp.array[dtype]
    return wp.kernel(shifted_system_values, name=f"shifted_system_values_{dtype.__name__}")


# Keyed by the block dtype: the scalar heat system and the vector one's ``wp.mat22d`` blocks. A
# factory rather than one generic kernel because the diagonal block is formed from the scalar mass
# differently at the two dtypes.
SHIFTED_SYSTEM_VALUES = {d: _shifted_system_values_kernel(d) for d in (wp.float64, wp.mat22d)}


def _register_overloads() -> None:
    """Instantiate the generic solver kernels at the precisions ``_BatchedCg`` solves in."""
    global JACOBI_INVERSE_DIAGONAL, CSR_LONG_ROWS_MATVEC
    # Keyed ``(vector dtype, value dtype)`` like ``conjugate_gradient.CG_MATVEC_DOTS``.
    CSR_LONG_ROWS_MATVEC = OverloadTable(
        csr_long_rows_matvec,
        {
            (d, v): [wp.int32, wp.int32, wp.array[wp.vec3i]]
            + [wp.array[wp.int32]] * 3
            + [wp.array[v], wp.array[d], wp.array[wp.float64], wp.array[wp.int32], wp.array[d]]
            for d, v in (
                (wp.float32, wp.float32),
                (wp.float64, wp.float64),
                (wp.float64, wp.float32),
            )
        },
    )
    JACOBI_INVERSE_DIAGONAL = OverloadTable(
        jacobi_inverse_diagonal,
        {
            d: [wp.array[wp.int32], wp.array[wp.int32], wp.array[d], wp.array[d]]
            for d in (wp.float32, wp.float64)
        },
    )


def register_warp_overload(kernel: Any, scalar: Any, **arguments: Any) -> None:
    """
    Register ``kernel``'s overload with every generic argument at ``scalar``.

    For Warp's *own* generic kernels, whose modules ordito cannot give a ``_register_overloads``:
    each new dtype a launch meets adds an overload, changes the module's hash and recompiles it,
    exactly as for ordito's. Registered before the module first loads, all
    of them compile together once. A generic scalar argument becomes ``scalar``, a generic array
    an array of ``scalar`` at its own rank; ``arguments`` overrides a named argument's scalar
    (a copy between precisions). The concrete signature is derived from the kernel's own
    template, so an argument added or renamed upstream is followed rather than guessed.
    """
    concrete = {}
    for name, template in kernel.adj.arg_types.items():
        dtype = arguments.get(name, scalar)
        if template is Any:
            concrete[name] = dtype
        elif isinstance(template, array_type) and type_is_generic(template.dtype):
            concrete[name] = array_type(dtype=dtype, ndim=template.ndim)
        else:
            concrete[name] = template
    wp.overload(kernel, concrete)


def register_bsr_mv_overloads(tile_size: int) -> None:
    """
    Register the ``warp.sparse.bsr_mv`` kernels a heavy-row CG round launches, at both precisions.

    The lane-per-row kernel for scalar blocks and the block-per-row one at ``tile_size``. Warp
    builds each from a cached factory, so the arguments are spelled as ``bsr_mv`` spells them
    (``block_cols=`` as a keyword, the tile positionally) to reach the kernel it launches.
    """
    for scalar in (wp.float32, wp.float64):
        register_warp_overload(make_bsr_mv_kernel(block_cols=1), scalar)
        register_warp_overload(make_bsr_mv_tiled_kernel(tile_size), scalar)


CSR_LONG_ROWS_MATVEC: OverloadTable
_register_overloads()
