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
    on Warp 1.18. ``wp.min(wp.abs(wp.get_diag(r)))`` asks the identical question ("is some
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
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    start = offsets[i]
    end = offsets[i + 1]
    kept = wp.int32(0)
    for e in range(start, end):
        if not fixed_mask[columns[e]]:
            kept += 1
    # A free row every one of whose columns is pinned counts 0 and becomes an *empty* CSR row, not a
    # dropped one -- the scan of these counts is what keeps the row offsets in step.
    out_counts[ri] = kept
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
    for c in range(out_rhs.shape[0]):
        acc = wp.float64(0.0)
        if c < load.shape[0]:
            acc = load[c, i]
        if c < fixed_values.shape[0]:
            for e in range(start, end):
                j = columns[e]
                if fixed_mask[j]:
                    acc -= scale * values[e] * fixed_values[c, j]
        out_rhs[c, ri] = acc


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
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    slot = row_offsets[ri]
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        if not fixed_mask[j]:
            out_columns[slot] = free_map[j]
            out_values[slot] = scale * values[e]
            slot += 1


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
def flag_unconverged_columns(
    residual: wp.array[wp.float64], threshold: wp.array[wp.float64], out_flag: wp.array[wp.int32]
) -> None:
    # A conjugate-gradient result's verdict, on the device: a column whose last residual is above
    # its threshold did not converge. Writes into the flag the backward-error test shares, so one
    # four-byte read answers both.
    column = wp.int32(wp.tid())
    if residual[column] > threshold[column]:
        out_flag[0] = 1


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
    # ``offdiagonal_dominance_rows`` for why both halves of that are load-bearing.
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


@wp.kernel
def offdiagonal_dominance_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_ratio: wp.array[wp.float64],
) -> None:
    # One thread per row of a CSR operator, writing that row's Gershgorin ratio
    # ``sum_{j != i} |A_ij| / A_ii``. The maximum over the rows is what
    # ``linalg._offdiagonal_dominance`` reduces to, and what decides whether a system reaches a
    # multigrid hierarchy -- see ``linalg.CG_PROBE_ITERATIONS`` for the 29 systems behind that gate.
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
    i = wp.int32(wp.tid())
    _inverse, ratio = jacobi_row(offsets, columns, values, i)
    out_ratio[i] = ratio


@wp.kernel
def jacobi_dominance_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_inverse_diagonal: wp.array[wp.float64],
    out_ratio: wp.array[wp.float64],
) -> None:
    # ``offdiagonal_dominance_rows`` and the Jacobi inverse diagonal (``array.inverse_or_one``) from
    # one walk of each row: the Jacobi-Chebyshev preconditioner wants the scaling and the
    # Gershgorin interval of one operator, which ``wps.bsr_get_diag``, a map and a second row walk
    # otherwise took three launches and a buffer to produce.
    i = wp.int32(wp.tid())
    inverse, ratio = jacobi_row(offsets, columns, values, i)
    out_inverse_diagonal[i] = inverse
    out_ratio[i] = ratio


@wp.kernel
def jacobi_inverse_diagonal(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    out_inverse_diagonal: wp.array[wp.Float],
) -> None:
    # The Jacobi inverse diagonal of a scalar CSR operator in one launch, where
    # ``wps.bsr_get_diag`` followed by a ``wp.map`` of ``array.inverse_or_one`` is two and an
    # intermediate buffer. The row's ratio is dead code here and compiles away.
    i = wp.int32(wp.tid())
    inverse, _ratio = jacobi_row(offsets, columns, values, i)
    out_inverse_diagonal[i] = inverse


@wp.kernel
def refresh_pooled_operator(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    narrow: wp.int32,
    derive: wp.int32,
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_narrowed: wp.array[wp.float32],
    out_inverse_diagonal: wp.array[wp.float64],
    out_ratio: wp.array[wp.float64],
) -> None:
    # One thread per row: copy a scalar CSR operator into a pooled conjugate-gradient state's own
    # storage (``linalg._cached_solver``) and re-derive, from the same walk of the row, what the
    # state derived from its operator when it was built -- the ``float32`` copy of the values the
    # rounds read (``narrow``), the Jacobi inverse diagonal (``derive >= 1``) and the Gershgorin
    # ratio the Jacobi-Chebyshev interval is fitted to (``derive == 2``). Each is the same
    # arithmetic as the kernel that built it (``jacobi_inverse_diagonal`` /
    # ``jacobi_dominance_rows``), so a refreshed state holds what a new one would. The arrays a
    # flag leaves off may be ``None``. The last thread also writes the terminating offset.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    out_offsets[i] = start
    if i == out_offsets.shape[0] - 2:
        out_offsets[i + 1] = end
    for e in range(start, end):
        value = values[e]
        out_columns[e] = columns[e]
        out_values[e] = value
        if narrow != 0:
            out_narrowed[e] = wp.float32(value)
    if derive != 0:
        inverse, ratio = jacobi_row(offsets, columns, values, i)
        out_inverse_diagonal[i] = inverse
        if derive == 2:
            out_ratio[i] = ratio


@wp.kernel
def scaled_rows_and_abs_sums(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.float64],
    weight_sums: wp.array[wp.float64],
    out_scaled: wp.array[wp.float64],
    out_ratio: wp.array[wp.float64],
) -> None:
    # One thread per row of a CSR ``L``, in one walk of the row: the values of ``D^-1 L`` (each
    # row scaled by ``array.inverse_or_one`` of ``D_i``, so a zero weight scales by 1) and
    # ``sum_j |L_ij| / D_i``, the Gershgorin radius plus centre of row ``i`` of ``D^-1 L``, whose
    # maximum bounds that operator's spectrum from above.
    # ``linalg.SquaredLaplacianPreconditioner`` fits its polynomial to that bound. With ``D`` the
    # diagonal of ``L`` it is ``1 + offdiagonal_dominance_rows``' ratio; with any other positive
    # ``D`` -- ``sqrt(M)`` for the k = 2 harmonic operator ``L M^-1 L`` -- it is the bound that
    # ratio is not, and an interval fitted short of the spectrum amplifies what it should invert.
    i = wp.int32(wp.tid())
    weight = weight_sums[i]
    scale = inverse_or_one(weight)
    total = wp.float64(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        value = values[e]
        total += wp.abs(value)
        out_scaled[e] = value * scale
    out_ratio[i] = total / weight


@wp.kernel
def expand_block_csr_2x2(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat22d],
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # One thread per block row ``i`` of a compact ``wp.mat22d`` CSR, writing scalar rows ``2 i`` and
    # ``2 i + 1`` of the same operator over interleaved unknowns (``2 j + b`` is component ``b`` of
    # block unknown ``j``), the memory layout of a ``wp.vec2d`` array viewed as ``float64``.
    # Each scalar row holds ``2 * count`` entries, so its start is a closed form of the block
    # offsets and needs no scan; columns stay sorted because the block row's are. The last thread
    # also writes the terminating offset.
    i = wp.int32(wp.tid())
    start = offsets[i]
    count = offsets[i + 1] - start
    for a in range(2):
        row_start = 4 * start + a * 2 * count
        out_offsets[2 * i + a] = row_start
        for e in range(count):
            block = values[start + e]
            j = columns[start + e]
            for b in range(2):
                out_columns[row_start + 2 * e + b] = 2 * j + b
                out_values[row_start + 2 * e + b] = block[a, b]
    if i == offsets.shape[0] - 2:
        out_offsets[2 * i + 2] = 4 * offsets[i + 1]


@wp.struct
class CsrBlock:
    """One diagonal block of ``stack_block_diagonal``: a square scalar ``float64`` CSR."""

    offsets: wp.array[wp.int32]
    columns: wp.array[wp.int32]
    values: wp.array[wp.float64]
    n_rows: wp.int32


@wp.kernel
def stack_block_diagonal(
    blocks: wp.array[CsrBlock],
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # One thread per row of the block-diagonal stack of ``blocks``, copying its block's row with
    # the columns shifted by the rows above it. Each block's stored count is its own last offset,
    # read here rather than off ``nnz`` (a capacity, CLAUDE.md 3.7), so a row's start needs no
    # scan and no readback: a handful of blocks, a handful of reads. The last thread also writes
    # the terminating offset.
    r = wp.int32(wp.tid())
    row_base = wp.int32(0)
    entry_base = wp.int32(0)
    b = wp.int32(0)
    while r - row_base >= blocks[b].n_rows:
        entry_base += blocks[b].offsets[blocks[b].n_rows]
        row_base += blocks[b].n_rows
        b += 1
    block = blocks[b]
    i = r - row_base
    start = block.offsets[i]
    end = block.offsets[i + 1]
    destination = entry_base + start
    out_offsets[r] = destination
    for e in range(end - start):
        out_columns[destination + e] = block.columns[start + e] + row_base
        out_values[destination + e] = block.values[start + e]
    if r == out_offsets.shape[0] - 2:
        out_offsets[r + 1] = entry_base + end


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
    relative = wp.min(interval / wp.float64(wp.max(n, 1)), cap)
    lower = wp.float64(0.0)
    upper = wp.float64(0.0)
    if squared != 0:
        upper = wp.max(bound[0], wp.float64(2.0))
        lower = relative * (upper / wp.float64(2.0))
    else:
        dominance = wp.max(bound[0], wp.float64(1e-3))
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


def _shifted_system_values_kernel(dtype: Any) -> Any:
    """``shifted_system_values`` for one block dtype: ``wp.float64`` or ``wp.mat22d``."""

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
        # One row of ``scale * A + M``, written over ``A``'s own pattern -- the heat method's
        # ``M - t L`` and ``M + t L_connection``, implicit fairing's ``M - lamb L`` -- where
        # ``warp.sparse.bsr_axpy`` against a ``bsr_diag`` would merge two patterns, sorting both,
        # for a result whose pattern is ``A``'s: every vertex a face references has a diagonal
        # entry (``cotmatrix`` and ``connection_laplacian`` keep every triplet), and one no face
        # references has no mass.
        # With ``negate`` set, ``-A`` rides along into ``out_negated`` in the same pass: the scalar
        # method's Poisson operator; unset, ``out_negated`` is never touched and may be ``None``.
        # Scalar ``float64`` or ``wp.mat22d`` blocks.
        #
        # ``M`` is the scalar lumped mass at either block dtype: a ``wp.mat22d`` system carries
        # two unknowns per vertex and the same area weight applies to both, so its diagonal block
        # is formed here, as it is read, rather than mapped into a block buffer first.
        #
        # With ``from_edges`` set the scale is ``scale`` times the default timestep ``h ** 2``,
        # ``h`` the mean unique-edge length from ``upper_edge_length_sum_and_count``'s two sums
        # (``0`` for no edges), formed on the device so the host never waits for them; ``scale``
        # is then the sign. Unset, ``edge_sum_and_count`` is never read and may be ``None``.
        row = wp.int32(wp.tid())
        factor = scale
        if from_edges != 0:
            h = wp.float64(0.0)
            if edge_sum_and_count[1] > wp.float64(0.0):
                h = edge_sum_and_count[0] / edge_sum_and_count[1]
            factor = scale * (h * h)
        for e in range(offsets[row], offsets[row + 1]):
            value = values[e]
            shifted = factor * value
            if columns[e] == row:
                m = mass[row]
                if wp.static(dtype == wp.mat22d):
                    shifted = shifted + wp.mat22d(m, wp.float64(0.0), wp.float64(0.0), m)
                else:
                    shifted = shifted + m
            out_values[e] = shifted
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
    """Instantiate ``jacobi_inverse_diagonal`` at the precisions ``_BatchedCg`` solves in."""
    global JACOBI_INVERSE_DIAGONAL
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
    exactly as for ordito's (CLAUDE.md section 2.5). Registered before the module first loads, all
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


_register_overloads()
