"""
Sparse linear-algebra helpers shared by the parametrization and smoothing solvers.

Two layers:

- [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] and its two steps
  ([`free_partition`][ordito.linalg.free_partition],
  [`assemble_interior_system`][ordito.linalg.assemble_interior_system]) port
  ``igl::min_quad_with_fixed``: minimize a quadratic form with some degrees of freedom pinned, by
  eliminating the fixed rows/columns into a reduced symmetric system.
- [`solve_spd_columns`][ordito.linalg.solve_spd_columns] and
  [`spd_column_solver`][ordito.linalg.spd_column_solver] solve one symmetric positive-definite
  operator against several right-hand-side columns in a *single* conjugate-gradient call.

**Why batch the columns.** Solving a ``k``-column system as a Python loop of ``k`` independent
``cg`` calls costs the ``sum`` of the per-column iteration counts, and every CG iteration then
launches ``k`` separate sets of reduction and AXPY kernels.
[`replicated_operator`][ordito.linalg.replicated_operator] instead presents the *same* operator as
``k`` independent subproblems over one flat ``k * n`` vector: the solver advances all columns
together and stops on the worst-case residual, so the cost becomes ``max`` of the per-column
iteration counts with one set of vector kernels per iteration. The sparse matrix is never replicated
in memory -- the ``matvec`` issues ``k`` ``bsr_mv`` calls against the single operator.

**Whose conjugate gradient.** Every scalar ``float64`` solve runs this module's own ``_BatchedCg``
-- the column solvers at any column count, and [`solve_spd`][ordito.linalg.solve_spd] whenever its
preconditioner is one this module built -- with Warp's stopping rule; only a block operator or a
caller's own ``LinearOperator`` goes to ``warp.optim.linear.cg``. Two costs of Warp's solver decide
it. Batching is expressed to Warp as ``batch_offsets``, which makes its dot-product reduction take
a per-column path whose cost grows with the vector length; and at ``check_every=0`` it records and
instantiates a fresh conditional graph on every call, which on the small and medium systems here
costs more than the iterations do. ``_BatchedCg`` reduces per column with a two-stage tree, runs
the Chronopoulos-Gear form of the iteration -- one mat-vec with all three of its dots, then one
update, two launches a round -- and records its loop once per state, where a solve through
[`solve_spd`][ordito.linalg.solve_spd] or [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
keeps one state per operator, so a later solve against the same operator only replays it.

**Determinism.** Build each operator natively at its final dtype in a *single* sparse build and
never recast or rebuild it. A rebuild re-sorts and duplicate-accumulates an order the CSR already
has; and if the rebuild's triplet buffers are ever sized off ``BsrMatrix.nnz`` (a stale *capacity*,
not the true entry count -- see ``nnz_sync()``) the buffers' tail reaches the build
uninitialized. ``Q_uu`` is assembled as a CSR
*directly*, without any triplet build, so it carries an exact ``nnz``.

**Why Jacobi.** Every solve here preconditions with ``warp.optim.linear.preconditioner(A, "diag")``.
On a cotangent Laplacian a preconditioner costing ``k`` mat-vecs per iteration cuts the iteration
count by only about ``sqrt(k)``, so total work scales as ``sqrt(k)`` -- single-level preconditioning
loses on this operator class, and only a multilevel method escapes it. IC(0) would help if it were
available, but Warp has no sparse triangular solve, and no parallel substitute keeps its advantage:
a Jacobi-sweep approximation, an exact apply parallelized by graph coloring (whose uncoalesced
access and extra launches eat most of its iteration win), and natural-ordering level sets (whose
level count swings wildly with the input's vertex numbering) are all worse than Jacobi in practice.

**The exception is a polynomial**, because the ``sqrt(k)`` argument counts mat-vecs and a solve here
is bound by *launches*: an iteration is half a dozen launches and two reductions, while one step of
a Chebyshev polynomial in ``D⁻¹ A`` is a single fused mat-vec launch.
[`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner] trades the iterations for
those steps and wins several-fold on long, ill-conditioned Laplacian solves -- the Poisson half of
the heat method, a graded patch -- while a short, well-conditioned one (a heat system, whose mass
term keeps it near diagonal) is faster under plain Jacobi. So it is an opt-in, taken where a caller
knows its solve is long. Chebyshev as the V-cycle's *smoother* is a separate question -- see
``_MULTIGRID_SWEEPS`` -- and so is a *squared* operator, where the polynomial is in its second-order
square root: [`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner].

Two further obstacles are specific to this repository. Obtuse triangles give negative cotangent
weights, so ``-L`` is often not the M-matrix IC(0) existence requires; and both
[`heat_geodesic`][ordito.heat.heat_geodesic] and
[`heat_signed_distance`][ordito.heat.heat_signed_distance] solve a ``-L`` with a genuine constant
null space, where IC(0) hits a zero pivot on the last row of every connected component.

**The multilevel option** is [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner]:
smoothed aggregation, the one scheme that breaks the iteration count's growth with problem size
rather than paying it down by a constant factor. It is not the default, and the reason is the
*setup* rather than the cycle: building the hierarchy (one aggregation, one power iteration, a
transpose and two sparse products per level) has a real fixed cost, so the V-cycle wins
exactly where the solve it replaces is long enough to amortize that setup and loses on a
well-conditioned or already-fast-converging system. See
[`CG_MULTIGRID_DOMINANCE`][ordito.linalg.CG_MULTIGRID_DOMINANCE] for how the gate decides.

**A direct factorization** is the other half of the module, through
[`sparse_cholesky`][ordito.cholesky.sparse_cholesky]: its cost is a symbolic analysis and a numeric
factorization, flat regardless of conditioning, so it loses to the iteration where conjugate
gradient converges quickly and wins where it does not, or where one operator is solved many times.
A caller keeps one in an [`OperatorFactorization`][ordito.linalg.OperatorFactorization], and the
verified solves -- [`solve_spd_columns`][ordito.linalg.solve_spd_columns] given one,
[`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] always,
[`solve_spd_settled`][ordito.linalg.solve_spd_settled] -- buy it exactly when the iteration's result
fails its check. Nothing in this module keeps a factorization across calls on its own.

Nothing cheap predicts in advance which side of the multigrid-versus-Jacobi line a system falls on,
which is why ``preconditioner="auto"`` uses a capped Jacobi probe rather than a heuristic predictor
of the iteration count; see [`CG_PROBE_ITERATIONS`][ordito.linalg.CG_PROBE_ITERATIONS]. The
aggregation keeps a strength-of-connection threshold (``_MULTIGRID_THETA``) rather than every
off-diagonal, because on a graded (anisotropic) patch keeping every off-diagonal aggregates across
the weak direction and the hierarchy converges far more slowly.

**Routing through ``"auto"`` decides from the operator, not the caller**, and the axis that matters
is conditioning: only a system whose off-diagonal dominance clears
[`CG_MULTIGRID_DOMINANCE`][ordito.linalg.CG_MULTIGRID_DOMINANCE] benefits from a hierarchy. A plain
Laplacian (``tutte``, ``min_quad_with_fixed`` on a raw cotangent matrix) and ``lscm``'s coupled u/v
system sit below that bar and a forced hierarchy regresses them, where a *squared* operator
(``harmonic`` at ``k >= 3``) sits comfortably above it. So
[`harmonic`][ordito.parametrization.harmonic] passes ``"auto"`` at ``k >= 3`` for the iteration it
falls back to when the system cannot be factored (it factors first there), and
[`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] defaults to ``"adaptive"``. At ``k = 2``
the operator is ``L M^-1 L``, which
[`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner] inverts as the
square of a Laplacian, and ``harmonic`` takes that instead.

A ``k=2`` biharmonic operator on a strongly graded patch is outside what any preconditioned
conjugate gradient reaches in ``float64``: its condition number nears the reciprocal of the unit
round-off, and the iteration stalls at a residual of a few percent. That is the case the verified
solves exist for -- the stalled iterate fails its check after
[`CG_FACTOR_AFTER_ROUNDS`][ordito.linalg.CG_FACTOR_AFTER_ROUNDS] rounds and the factorization
answers.

The heat system ``M - tL`` that [`heat_geodesic`][ordito.heat.heat_geodesic] solves needs no
multilevel help of its own: it converges in a small, size-independent number of iterations, because
``t = h**2`` makes it a small perturbation of the mass matrix.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import glob
import math
import os
import warnings
import weakref
from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast, overload

import numpy as np
import warp as wp
import warp.optim.linear as wpl

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.cholesky import NotPositiveDefiniteError, SparseCholesky, sparse_cholesky
from ordito.constants import TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import cholesky as kernel_cholesky
from ordito.kernels import linalg as kernel_linalg
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels.algorithms import conjugate_gradient as kernel_cg
from ordito.kernels.algorithms import multigrid as kernel_mg

if TYPE_CHECKING:
    import numpy.typing as npt

# Relative residual tolerance used when a caller does not supply one.
CG_TOLERANCE = 1e-10

# Iteration cap as a multiple of the per-column system size, matching the callers' previous
# hand-rolled ``maxiter=10 * n``.
CG_MAXITER_FACTOR = 10

# The componentwise backward error ``|b - A x|_i / (|A| |x| + |b|)_i`` a settled iterate of
# ``solve_spd_settled`` may carry anywhere before the solve falls back to a direct factorization.
# A genuinely settled iterate sits orders of magnitude below it; one that "settled" on round-off
# (a strongly graded mesh, a scan with slivers) sits at one.
SETTLE_BACKWARD_ERROR = 1e-6

# Componentwise backward error the direct fallback refines to.
SETTLE_DIRECT_TOLERANCE = 1e-12

# Rounds a solve holding a fallback factorization (``solve_spd_columns(factorization=...)``, which
# every ``min_quad_with_fixed`` solve does) iterates before it stops and factors instead: the rent
# of a rent-or-buy choice. A factorization of the reduced systems the parametrizations solve costs
# about what 200-400 conjugate-gradient rounds do, and every well-posed solve measured converges
# well inside the budget (18-855 rounds; ``lscm`` on a graded patch is the longest), so the budget
# stops a solve that is not converging -- a biharmonic system with a condition number near 1e17
# stalls at a few percent residual for its whole ``10 n`` cap -- long before the cap would.
CG_FACTOR_AFTER_ROUNDS = 1000

# The componentwise backward error ``|b - A x|_i / (|A| |x| + |b|)_i`` a verified solve's iterate
# may carry anywhere -- or 100 times the solve's tolerance, if that is larger -- before the solve
# falls back to its factorization. A converged iterate at a relative residual of ``1e-8`` sits at
# 5e-9 to 2.2e-7 on the parametrizations' systems; one that is not the system's answer at 1e-2.
VERIFIED_BACKWARD_ERROR = 1e-6

# How often the conjugate-gradient loop tests the residual against the tolerance. ``0`` means
# "every iteration, on device": ``warp.optim.linear`` then drives the loop with ``wp.capture_while``
# and an on-device condition kernel, with no host readback at all. See the ``check_every`` parameter
# docs for the measurements and for the return-type consequence.
CG_CHECK_EVERY = 0

# Batching several iterations per conditional-graph test was tried and removed; do not
# reintroduce it without a *short* solve in the sweep. ``wp.capture_while`` evaluates its condition
# on device at several times the cost of a replayed launch, so running a run of iterations per test
# amortizes that -- but it also overshoots by up to ``K - 1`` iterations past the point the residual
# crossed, and that overshoot is a fixed number of launches whose *share* is set by how long the
# solve is. Long solves therefore win a little and short ones lose a lot, and every summary
# statistic put the unbatched loop ahead. A converged iteration here is a pure no-op --
# ``cg_update`` takes no step on a column inside its tolerance -- so the overshoot buys nothing,
# unlike the equivalent batching in a breadth-first level loop, where an extra level still does
# useful work and a small batch is kept.

# Jacobi rounds ``preconditioner="adaptive"`` runs before it escalates to the Chebyshev polynomial.
# The polynomial costs ``CHEBYSHEV_DEGREE - 1`` launches a round and a setup that reads a bound
# back, which a short, well-conditioned solve -- a warm-started ``arap`` step, a system pinned along
# half its unknowns -- does not earn back, while a long one gains several-fold. The probe cap sits
# between the two classes the suite's solves fall into: the short ones finish in under a hundred
# Jacobi rounds, the long ones take several hundred to a few thousand.
CG_CHEBYSHEV_PROBE_ITERATIONS = 150

# Tiles per column up to which a conjugate-gradient round's update folds the dots' second stage
# itself -- each block re-reading its column's ``blocks`` partials through the same fixed tree, so
# every block agrees on ``alpha`` and ``beta`` with no launch between it and the mat-vec -- and
# above which a finalize launch does it once. The redundant fold costs ``blocks`` reads per block,
# so ``blocks`` squared per round: at 256 tiles (65 536 unknowns a column at ``CG_TILE`` 256) that
# is under two megabytes out of L2, against the one launch a round it removes. Warp exposes no
# grid-wide fence, so the last-block-done pattern that would make the fold free is not available.
CG_FOLD_MAX_BLOCKS = 256

# Mean stored entries a row above which a conjugate-gradient round forms ``A u`` with
# ``warp.sparse.bsr_mv`` and reduces its dots in a launch of their own, instead of one lane a row
# fused with the reduction. Measured on 400 000-row random-pattern ``float32`` operators, against
# ``bsr_mv``'s lane-per-row kernel: the fused kernel wins at 8 entries a row (0.042 ms against
# 0.064), ties at 16 (0.072 / 0.070) and loses from there (0.22 against 0.13 at 32, 0.82 against
# 0.64 at 128) -- its block-wide reduction holds every lane until the block's longest row is done.
# Mesh Laplacians sit near 7 and keep the fused kernel; ``warp.fem``'s Poisson systems sit near 26.
# Applies only past ``CG_FOLD_MAX_BLOCKS``, where a round is bound by bytes rather than launches.
CG_HEAVY_ROW_ENTRIES = 16

# Mean entries a row above which that ``bsr_mv`` takes Warp's block-per-row kernel, 64 lanes a
# row, rather than its lane-per-row one: the two tie at 64 (0.25 / 0.27 ms) and the block-per-row
# kernel is 2x ahead at 128. Chosen here rather than by ``bsr_mv``'s own heuristic, which reads the
# ``nnz`` *capacity* -- 4.5x the true count on the ``warp.fem`` system that motivated this.
_HEAVY_ROW_TILED_ENTRIES = 64
_HEAVY_ROW_TILE = 64
# Both kernels those rounds launch are Warp's generic ones, and a process solving in both
# precisions would otherwise rebuild each kernel's module at its second dtype.
kernel_linalg.register_bsr_mv_overloads(_HEAVY_ROW_TILE)

# Rows per column up to which a solve under Jacobi or the squared-Laplacian polynomial runs as one
# launch, one block per column (``kernels/algorithms/conjugate_gradient.cg_one_block``), instead of
# a recorded ``_BatchedCg``: a round then costs block barriers rather than replayed graph nodes,
# and a fresh system pays no recording. The whole column lives in one SM, so the round grows with
# the column where a replayed round stays flat. Measured on Jacobi-preconditioned heat systems
# against an already-recorded ``_BatchedCg`` (its best case), 30 and 150 rounds: 1.5-1.8x at 642
# rows and one column, 1.8-2.1x at three, and a loss from ~1 200 rows at one column (0.85x at
# 1 589) and ~1 900 at three. A fresh system's recording, which this skips, moves the crossover
# up; 1 024 keeps every measured single-column case a win. ``CG_ONE_BLOCK_WIDE_FROM`` rows and up
# take the wider block (512 against 256 lanes: 0.84 against 0.97 ms at 1 000 rows).
CG_ONE_BLOCK_MAX_ROWS = 1024
CG_ONE_BLOCK_WIDE_FROM = 768
_CG_ONE_BLOCK_NARROW = 256
_CG_ONE_BLOCK_WIDE = 512

# Solver states kept per operator by ``_cached_solver``. One operator is normally solved under one
# configuration; the bound is there so a long-lived operator solved under many cannot accumulate
# device memory without limit.
_SOLVER_CACHE_ENTRIES = 4

# Cadence substituted for ``check_every=0`` on a device without conditional CUDA graphs, where Warp
# cannot test the residual on device and would otherwise run every solve to ``maxiter``. Warp's own
# default, and what this module shipped before the device-side check became the default.
CG_CHECK_EVERY_FALLBACK = 10

# Jacobi iterations ``preconditioner="auto"`` runs before it escalates to a multigrid hierarchy, on
# the systems its gate (``CG_MULTIGRID_DOMINANCE`` below) did not already send straight to one. This
# cap is the *fallback* branch and runs unchanged on every system the gate declines.
#
# It is a cap, not a predictor: nothing cheap distinguishes in advance a system that benefits from a
# hierarchy from one that does not, on axes like size, iteration count or the probe's own
# convergence rate, so a system that converges inside the cap never pays for a hierarchy setup and
# runs at exactly Jacobi's speed. That is conservative -- it forgoes the hierarchy's benefit on
# systems that converge just inside the cap -- and deliberately so: escalating early or not at all
# is better than escalating from a mid-range cap, which pays for both the probe and the hierarchy
# setup without benefiting from either. The escalated solve does not meaningfully reuse the probe's
# iterates either, since Jacobi leaves a residual a V-cycle still has to work down from scratch. The
# axis that actually separates the two classes is a property of the operator, not of the probe --
# see ``CG_MULTIGRID_DOMINANCE``.
CG_PROBE_ITERATIONS = 2000

# The gate ``preconditioner="auto"`` applies *before* the probe above: a system that clears it goes
# straight to a multigrid hierarchy and never runs a Jacobi iteration, and one that does not falls
# through to ``CG_PROBE_ITERATIONS`` unchanged. So the gate can only ever *remove* a probe, which is
# what makes it safe: on a system it declines, ``"auto"`` behaves exactly as it did before it
# existed.
#
# The axis is the operator's off-diagonal dominance, not its size: ``max_i sum_{j != i} |A_ij| /
# A_ii``, one kernel and one max-reduction over the assembled system (``_offdiagonal_dominance``).
# Smoothed aggregation's advantage over Jacobi grows with how much weight a row carries off its
# diagonal, so this is the operator property the V-cycle is actually paid for.
CG_MULTIGRID_MIN_UNKNOWNS = 1500

# Free unknowns above which a merely *moderate* dominance is enough to skip the probe -- the
# well-shaped regular meshes, where conjugate gradient's growth in the mesh size is the whole
# problem. Paired with ``CG_MULTIGRID_SIZE_FLOOR`` below, which keeps this branch from firing on an
# operator class it was not calibrated on.
CG_MULTIGRID_LARGE_UNKNOWNS = 7000

# Dominance floor under which the size branch above does **not** fire, however large the system.
#
# The size branch is the operator-class-specific half of the gate: it transfers to well-shaped
# regular meshes but not to every large system. A plain Laplacian's rows nearly sum to zero (a
# uniform-weight parametrization, a raw cotangent system) and does not need a hierarchy at any size
# -- it needs iterations -- while a squared or coupled operator does. Without this floor the size
# branch alone would fire on any large system regardless of conditioning.
CG_MULTIGRID_SIZE_FLOOR = 1.9

# Off-diagonal dominance above which ``"auto"`` skips its probe, given at least
# ``CG_MULTIGRID_MIN_UNKNOWNS`` unknowns.
CG_MULTIGRID_DOMINANCE = 2.15


def min_quad_with_fixed(
    q: odt.SparseMatrix,
    fixed_mask: wp.array[wp.bool],
    fixed_values: odt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    check_every: int = CG_CHECK_EVERY,
    preconditioner: str = "adaptive",
    solver: str = "iterative",
    coordinates: wp.array[wp.vec3] | None = None,
) -> tuple[odt.Array2dFloat, wp.array[wp.int32], int]:
    """
    Minimize a quadratic form with pinned degrees of freedom.

    Eliminates the fixed degrees of freedom from ``0.5 x' Q x`` and solves the reduced symmetric
    system ``Q_uu x_u = -Q_ub bc`` for every right-hand-side column at once. Handles both shapes the
    parametrization solvers need: several *independent* columns over ``n_vertices`` unknowns
    (``harmonic`` / ``tutte``, one column per UV coordinate) and a single *coupled* column over
    ``2 * n_vertices`` unknowns (``lscm``).

    The solve is verified: it runs as
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns] does with a ``factorization``, so a
    conjugate-gradient result that has not converged within
    [`CG_FACTOR_AFTER_ROUNDS`][ordito.linalg.CG_FACTOR_AFTER_ROUNDS] rounds, or fails the
    componentwise backward-error test, is replaced by a solve through a
    [`sparse_cholesky`][ordito.cholesky.sparse_cholesky] factorization of ``Q_uu``, built for the
    call and dropped.

    Parameters
    ----------
    q
        ``(n_dofs, n_dofs)`` symmetric positive-semi-definite operator, ``float64``.
    fixed_mask
        ``(n_dofs,)`` mask: ``True`` marks a pinned degree of freedom.
    fixed_values
        ``(n_rhs, n_dofs)`` prescribed values; only the entries where ``fixed_mask`` is ``True`` are
        read.
    tol
        Relative residual tolerance for the conjugate-gradient solve.
    check_every
        Iterations between residual tests, forwarded to
        [`solve_spd_columns`][ordito.linalg.solve_spd_columns]. The default tests on device every
        iteration; see there for the tradeoff. This entry point's own return type does not depend
        on it — the solver's ``(iterations, residual, atol)`` triple is not surfaced here.
    preconditioner
        Forwarded to [`solve_spd_columns`][ordito.linalg.solve_spd_columns]. ``"adaptive"`` (the
        default) because the eliminated operator is usually a plain Laplacian, whose rows nearly sum
        to zero and which a V-cycle does not help, and whose solve is short when many degrees of
        freedom are pinned and long when few are -- the one case Jacobi wins and the other the
        polynomial does, with a probe to tell them apart.
    solver
        ``"iterative"`` (the default) runs the verified conjugate gradient above. ``"direct"``
        factors ``Q_uu`` first and solves by the factorization, iterating (verified, to the full
        cap) only when it cannot be factored: the choice for a system whose iteration is known to
        be long or inaccurate, such as a ``k >= 3`` polyharmonic operator.
    coordinates
        ``(n_points,)`` positions for the factorization's geometric ordering, degree of freedom
        ``i`` at point ``i mod n_points`` -- a mesh's vertices for a vertex operator, or for one
        stacking per-vertex components as ``lscm`` does. ``None`` orders by the pattern alone.

    Returns
    -------
    solution : odt.Array2dFloat
        ``(n_rhs, n_free)`` solved values for the unpinned degrees of freedom.
    free_map : wp.array[wp.int32]
        ``(n_dofs,)`` compact remap of the unpinned degrees of freedom.
    n_free : int
        Number of unpinned degrees of freedom.

    Raises
    ------
    ValueError
        If ``solver`` is not ``"iterative"`` or ``"direct"``, or ``preconditioner`` is not a name
        [`solve_spd_columns`][ordito.linalg.solve_spd_columns] accepts.
    RuntimeError
        If ``fixed_mask``, ``fixed_values`` and ``coordinates`` are not all on one device.

    Warns
    -----
    UserWarning
        When neither the iteration nor a factorization reaches a verified solution: ``Q_uu`` is
        over [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET] or not definite in
        ``float64``, and conjugate gradient fails to converge within ``CG_MAXITER_FACTOR * n_free``
        iterations.

    Notes
    -----
    The elimination is ``igl::min_quad_with_fixed``'s, without its ``Aeq`` linear-equality block --
    no caller here needs one.

    When every degree of freedom is pinned (``n_free == 0``) the prescribed values are the whole
    answer: an empty solution is returned without a solve, so this case also works on the CPU.
    Reconstructing the full field from ``solution`` / ``free_map`` is left to the caller's scatter.

    See Also
    --------
    [`free_partition`][ordito.linalg.free_partition]
    [`assemble_interior_system`][ordito.linalg.assemble_interior_system]
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    """
    require_same_device(fixed_mask=fixed_mask, fixed_values=fixed_values, coordinates=coordinates)
    if solver not in ("iterative", "direct"):
        raise ValueError(
            f'min_quad_with_fixed: unknown solver {solver!r}, expected "iterative" or "direct".'
        )
    _require_preconditioner_name(preconditioner, "min_quad_with_fixed")
    device = fixed_mask.device
    n_rhs = int(fixed_values.shape[0])
    free_map, n_free = free_partition(fixed_mask)

    solution = odt.as_array2d(
        _launch.zeros((n_rhs, n_free), dtype=wp.float64, device=device), wp.float64
    )
    if n_free == 0:
        return solution, free_map, n_free

    q_uu, rhs = assemble_interior_system(q, fixed_mask, free_map, fixed_values, n_free)
    factorization = OperatorFactorization(
        q_uu, free_coordinates(coordinates, fixed_mask, free_map, n_free)
    )
    if solver == "direct":
        factorization.factor()
    solve_spd_columns(
        q_uu,
        rhs,
        solution,
        tol=tol,
        check_every=check_every,
        preconditioner=preconditioner,
        factorization=factorization,
    )
    return solution, free_map, n_free


def free_partition(fixed_mask: wp.array[wp.bool]) -> tuple[wp.array[wp.int32], int]:
    """
    Compact remap of the *unpinned* degrees of freedom, plus their count.

    Thin inversion of [`mask_to_compact_ranks`][ordito.array.mask_to_compact_ranks]:
    ``fixed_mask`` marks the constrained degrees of freedom, and the returned map indexes the
    reduced system built over the complement.

    Parameters
    ----------
    fixed_mask
        ``(n_dofs,)`` mask: ``True`` marks a pinned degree of freedom.

    Returns
    -------
    free_map : wp.array[wp.int32]
        ``(n_dofs,)`` compact remap; ``free_map[i]`` is the reduced-system index of degree of
        freedom ``i``, meaningful only where ``fixed_mask[i]`` is ``False``.
    n_free : int
        Number of unpinned degrees of freedom.

    See Also
    --------
    [`mask_to_compact_ranks`][ordito.array.mask_to_compact_ranks]
    """
    return od.array.mask_to_compact_ranks(fixed_mask, invert=True)


def free_coordinates(
    coordinates: wp.array[wp.vec3] | None,
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    n_free: int,
) -> wp.array[wp.vec3] | None:
    """
    Positions of the reduced system's rows, for its factorization's ordering.

    Parameters
    ----------
    coordinates
        ``(n_points,)`` positions, degree of freedom ``i`` at point ``i mod n_points``: a mesh's
        vertices, for a vertex operator or one stacking per-vertex components. ``None`` passes
        through.
    fixed_mask
        ``(n_dofs,)`` mask: ``True`` marks a pinned degree of freedom.
    free_map
        ``(n_dofs,)`` compact remap from [`free_partition`][ordito.linalg.free_partition].
    n_free
        Number of unpinned degrees of freedom.

    Returns
    -------
    wp.array[wp.vec3] | None
        ``(n_free,)`` position of each free degree of freedom, or ``None`` for ``None``.

    Raises
    ------
    RuntimeError
        If ``coordinates``, ``fixed_mask`` and ``free_map`` are not all on one device.

    See Also
    --------
    [`OperatorFactorization`][ordito.linalg.OperatorFactorization]
    [`free_partition`][ordito.linalg.free_partition]
    """
    require_same_device(coordinates=coordinates, fixed_mask=fixed_mask, free_map=free_map)
    if coordinates is None or coordinates.size == 0:
        return None
    device = fixed_mask.device
    out = _launch.empty(n_free, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_linalg.free_coordinates,
        dim=fixed_mask.size,
        inputs=[fixed_mask, free_map, coordinates],
        outputs=[out],
        device=device,
    )
    return out


def assemble_interior_system(
    q: odt.SparseMatrix,
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    fixed_values: odt.Array2dFloat,
    n_free: int,
    *,
    scale: float = 1.0,
    load: odt.Array2dFloat | None = None,
) -> tuple[odt.BsrMatrix[wp.float64], odt.Array2dFloat]:
    """
    Extract the free-free block ``Q_uu`` and the constant right-hand side ``b_u - Q_ub bc``.

    A **CSR-to-CSR** extraction in two launches over the ``n_dofs`` rows of ``q``: the first counts
    each free row's surviving entries (and accumulates the pinned-column contributions into the
    ``n_rhs`` right-hand-side rows, which needs the same row scan), a prefix sum turns those counts
    into row offsets, and the second fills ``columns`` / ``values`` in place, remapped through
    ``free_map``. The result is handed to a ``warp.sparse.BsrMatrix`` directly.

    !!! note "Why not ``bsr_from_triplets``"
        Because the input is *already* a CSR. Emitting ``q.nnz`` triplets and letting
        ``bsr_from_triplets`` sort and duplicate-accumulate them re-derives an order the input
        already had, at real cost for a large operator. Row order comes from the launch index and
        column order from ``q``'s own rows through the monotone ``free_map``, so the extracted
        matrix is sorted by construction.

        It also avoids a second, quieter cost: ``bsr_from_triplets`` leaves ``nnz`` at the
        *triplet* count, which is ``q.nnz`` — an upper bound on the true entry count of a
        lightly-pinned ``Q_uu`` — so every downstream ``bsr_mv`` would be dimensioned for the
        unreduced matrix. The count here is exact.

    Parameters
    ----------
    q
        ``(n_dofs, n_dofs)`` symmetric positive-semi-definite operator, ``float64``.
    fixed_mask
        ``(n_dofs,)`` mask: ``True`` marks a pinned degree of freedom.
    free_map
        ``(n_dofs,)`` compact remap from [`free_partition`][ordito.linalg.free_partition].
    fixed_values
        ``(n_rhs, n_dofs)`` prescribed values.
    n_free
        Number of unpinned degrees of freedom.
    scale
        Factor applied to ``q`` as it is read, so the extraction is of ``scale * q`` (both the
        block and the pinned-column term). ``-1.0`` extracts ``-L`` from a Laplacian without a
        negated copy of it.
    load
        ``(n_rhs, n_dofs)`` linear term ``b``, compacted to its free entries and added to the
        right-hand side. ``fixed_values`` may then have zero rows, which reads as every pinned value
        zero. ``None`` (the default) is ``b = 0``.

    Returns
    -------
    q_uu : ``warp.sparse.BsrMatrix``
        ``(n_free, n_free)`` reduced operator.
    rhs : odt.Array2dFloat
        ``(n_rhs, n_free)`` constant right-hand side. ``arap`` adds its per-iteration rotation term
        to this buffer rather than reassembling it.

    Raises
    ------
    RuntimeError
        If ``fixed_mask``, ``free_map``, ``fixed_values`` and ``load`` are not all on one device.
    ValueError
        If ``load`` and ``fixed_values`` both have rows and their row counts differ.

    See Also
    --------
    [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]
    """
    require_same_device(
        fixed_mask=fixed_mask, free_map=free_map, fixed_values=fixed_values, load=load
    )
    device = fixed_mask.device
    n_dofs = fixed_mask.size
    n_rhs = int(fixed_values.shape[0])
    if load is not None:
        n_load = int(load.shape[0])
        if n_rhs not in (0, n_load):
            raise ValueError(
                f"load has {n_load} rows but fixed_values has {n_rhs}; they must match, or "
                "fixed_values must have none."
            )
        n_rhs = n_load
    # Every free row writes its own count exactly once (``free_map`` is a bijection onto
    # ``[0, n_free)``), so there is nothing to pre-zero.
    counts = _launch.empty(n_free, dtype=wp.int32, device=device)
    rhs = _launch.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    _launch.launch(
        kernel_linalg.interior_row_counts,
        dim=n_dofs,
        inputs=[
            q.offsets,
            q.columns,
            q.values,
            fixed_mask,
            free_map,
            fixed_values,
            wp.float64(scale),
            load,
            counts,
            rhs,
        ],
        device=device,
    )
    # The total-terminated form *is* the CSR offsets array, and the one host read it costs is what
    # sizes ``columns`` / ``values`` for their final use at allocation time.
    row_offsets, nnz_uu = od.array.counts_to_offsets(counts)
    columns = _launch.empty(nnz_uu, dtype=wp.int32, device=device)
    values = _launch.empty(nnz_uu, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_linalg.interior_system_csr,
        dim=n_dofs,
        inputs=[
            q.offsets,
            q.columns,
            q.values,
            fixed_mask,
            free_map,
            row_offsets,
            wp.float64(scale),
            columns,
            values,
        ],
        device=device,
    )
    # Hand the finished CSR to a compact ``BsrMatrix`` directly (``_bsr_over``).
    q_uu = _bsr_over(n_free, n_free, row_offsets, columns, values, nnz_uu)
    return q_uu, odt.as_array2d(rhs, wp.float64)


def solve_spd(
    matrix: odt.SparseMatrix,
    rhs: odt.ArrayNd,
    solution: odt.ArrayNd,
    *,
    tol: float = CG_TOLERANCE,
    maxiter: int | None = None,
    check_every: int = CG_CHECK_EVERY_FALLBACK,
    preconditioner: wpl.LinearOperator | None = None,
    name: str = "solve_spd",
) -> tuple[int, float, float]:
    """
    Solve one symmetric positive-definite system by preconditioned conjugate gradient.

    The single-right-hand-side form of
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns], and deliberately dtype-agnostic: it
    carries scalar ``float64`` systems as well as the ``wp.mat22d``-block operators the
    tangent-field solvers build, whose right-hand side is an array of ``wp.vec2d``. ``solution`` is
    the initial guess and is overwritten in place, so warm starting is free.

    This exists so the Jacobi preconditioner and the iteration cap are decided in one place rather
    than re-derived at every call site.

    !!! note "``check_every`` defaults to a positive cadence here, which means host scalars"
        Unlike [`solve_spd_columns`][ordito.linalg.solve_spd_columns], which defaults to
        [`CG_CHECK_EVERY`][ordito.linalg.CG_CHECK_EVERY], this defaults to
        [`CG_CHECK_EVERY_FALLBACK`][ordito.linalg.CG_CHECK_EVERY_FALLBACK], so the return values
        are host scalars and a non-convergence warning can be raised. On a system this module
        solves itself (see ``preconditioner``) the loop is tested on device either way wherever the
        device allows it, and a positive cadence costs one readback of the result after the solve;
        ``check_every=0`` skips that and returns device arrays.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-(semi-)definite operator. Scalar or block dtype.
    rhs
        ``(n,)`` right-hand side, with as many rows as ``matrix``.
    solution
        ``(n,)`` initial guess, overwritten with the result. Same shape and dtype as ``rhs``.
    tol
        Relative residual tolerance.
    maxiter
        Iteration cap. When ``None``, uses
        [`CG_MAXITER_FACTOR`][ordito.linalg.CG_MAXITER_FACTOR] times the number of rows.
    check_every
        Residual-test cadence; ``0`` tests on device every iteration and returns device arrays.
    preconditioner
        Optional preconditioner for ``matrix``; Jacobi when ``None``. For a scalar ``float64`` or
        ``wp.mat22d`` operator, ``None`` and the operators
        [`jacobi_preconditioner`][ordito.linalg.jacobi_preconditioner] and
        [`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner] build for ``matrix``
        are solved by this module's own conjugate gradient through a state kept for ``matrix``'s
        lifetime, so a later solve against the same operator replays its recorded loop instead of
        recording one -- and an operator rebuilt per call, of a shape already solved, is copied
        into a kept state of that shape, which replays too. Any other operator is applied as given
        through ``warp.optim.linear.cg``.
    name
        Caller name, used in the non-convergence warning.

    Returns
    -------
    tuple[int, float, float]
        Whatever ``warp.optim.linear.cg`` returns: iteration count, residual and tolerance. Device
        1-element arrays instead of host scalars when ``check_every=0``, which belong to the kept
        state and are overwritten by its next solve.

    Raises
    ------
    RuntimeError
        If ``rhs`` and ``solution`` are not all on one device.

    Warns
    -----
    UserWarning
        When the solve exhausts ``maxiter`` without reaching ``tol``. The returned ``solution`` is
        then whatever the last iterate happened to be, not an answer -- silence here is how a
        diverging solve reaches a caller looking like a converged one. Only detectable when
        ``check_every > 0``; under ``check_every=0`` the counts stay on device and testing them
        would reintroduce the readback that setting exists to avoid.

    See Also
    --------
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    [`spd_column_solver`][ordito.linalg.spd_column_solver]
    """
    require_same_device(rhs=rhs, solution=solution)
    n_rows = int(rhs.shape[0])
    iteration_cap = CG_MAXITER_FACTOR * n_rows if maxiter is None else maxiter
    kind = _batched_preconditioner_kind(matrix, rhs, preconditioner)
    if kind is not None:
        result = _solve_spd_batched(
            matrix, rhs, solution, tol=tol, cap=iteration_cap, check_every=check_every, kind=kind
        )
        # ``check_every=0`` never warns, as documented, even on a device that had to test the loop
        # from the host and so has host scalars to hand: a caller asking for no readback is often
        # one running a fixed budget on purpose (``reconstruction``'s capped Poisson solves).
        if check_every > 0:
            _warn_if_not_converged(result, iteration_cap, name)
        return result
    result = wpl.cg(
        matrix,
        rhs,
        solution,
        tol=tol,
        # ``atol=0.0`` rather than omitted: ``warp.optim.linear``'s own tolerance resolution sets
        # ``atol := tol`` whenever ``atol`` is left ``None``, silently turning this "relative
        # residual tolerance" into an *absolute* floor of the same numeric value -- a right-hand
        # side whose own norm falls below ``tol`` then "converges" at the untouched initial guess
        # in zero iterations, which looks like a correct answer rather than a failure (CLAUDE.md
        # section 12.7). ``_BatchedCg`` below already passes an explicit zero for the same
        # reason.
        atol=0.0,
        maxiter=iteration_cap,
        M=wpl.preconditioner(matrix, "diag") if preconditioner is None else preconditioner,
        check_every=_supported_check_every(check_every),
    )
    converged = cast("tuple[int, float, float]", result)
    _warn_if_not_converged(converged, iteration_cap, name)
    return converged


def _batched_preconditioner_kind(
    matrix: odt.SparseMatrix, rhs: odt.ArrayNd, preconditioner: wpl.LinearOperator | None
) -> str | None:
    """
    Name the preconditioner ``solve_spd`` can hand ``_BatchedCg``, or ``None`` to keep ``wpl.cg``.

    ``_BatchedCg`` runs compact scalar ``float64`` and ``float32`` CSR systems (the latter under
    Jacobi only), and builds its preconditioner itself from the operator, so it can stand in only
    for ``None`` and for the operators this module's own
    [`jacobi_preconditioner`][ordito.linalg.jacobi_preconditioner] and
    [`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner] built *for this
    matrix*. Anything else -- a block operator, a caller's own ``LinearOperator`` -- goes to Warp.
    """
    scalar = matrix.values.dtype in (wp.float32, wp.float64) and rhs.dtype == matrix.values.dtype
    block = matrix.values.dtype == wp.mat22d and rhs.dtype == wp.vec2d
    # The kernels bound a row by ``offsets[row + 1]``, which is the compact topology only.
    if not (scalar or block) or rhs.ndim != 1 or matrix.row_counts is not None:
        return None
    if preconditioner is None:
        return "diag"
    try:
        tag = _PRECONDITIONER_TAGS.get(preconditioner)
    except TypeError:  # a caller's operator that cannot be weakly referenced is not one of ours
        tag = None
    if tag is None or tag[1]() is not matrix:
        return None
    # A ``float32`` system stores its vectors at that precision and has only the Jacobi apply.
    if matrix.values.dtype == wp.float32 and tag[0] != "diag":
        return None
    return tag[0]


def _solve_spd_batched(
    matrix: odt.SparseMatrix,
    rhs: wp.array[wp.float64],
    solution: wp.array[wp.float64],
    *,
    tol: float,
    cap: int,
    check_every: int,
    kind: str,
) -> tuple[int, float, float]:
    """
    ``solve_spd`` on a scalar system, through the operator's cached ``_BatchedCg``.

    The loop is tested on device whenever the device allows it, whatever ``check_every`` asked
    for: a positive cadence is a request for host scalars (and the non-convergence warning), which
    reading the three results back once after the replay answers, where running the cadence from
    the host would issue every iteration's launches from Python.
    """
    if odt.has_blocks(matrix, wp.mat22d):
        # A ``wp.mat22d`` operator is solved as the scalar CSR over its interleaved unknowns, kept
        # per operator like the solver: the right-hand side and solution are the same bytes viewed
        # as ``float64``. Warp's blocked Jacobi inverts each diagonal block's diagonal
        # *coefficients*, which is scalar Jacobi on this expansion, so the preconditioner is the
        # one the caller asked for.
        matrix = _scalar_expansion(matrix)
        rhs = _as_scalar_view(rhs)
        solution = _as_scalar_view(solution)
    if _one_block_eligible(matrix, rhs.size, kind):
        result = _cg_one_block(
            matrix,
            rhs,
            solution,
            1,
            tol=tol,
            maxiter=cap,
            check_every=check_every,
            preconditioner=kind,
        )
        return cast("tuple[int, float, float]", result)
    solver = _cached_solver(
        matrix,
        1,
        tol=tol,
        maxiter=cap,
        check_every=_supported_check_every(0),
        preconditioner=kind,
        pooled=True,
    )
    result = solver.solve(rhs, solution)
    if check_every == 0 or not isinstance(result[0], wp.array):
        return cast("tuple[int, float, float]", result)
    return solver.host_result()


# ``_scalar_expansion``'s results, keyed weakly by the block operator they expand.
_EXPANSION_CACHE: weakref.WeakKeyDictionary[
    odt.SparseMatrix,
    tuple[tuple[tuple[int, int, int], tuple[object, ...]], odt.BsrMatrix[wp.float64]],
] = weakref.WeakKeyDictionary()


def _scalar_expansion(matrix: odt.SparseMatrix) -> odt.BsrMatrix[wp.float64]:
    """
    Expand a compact ``wp.mat22d`` operator into its ``float64`` CSR, once per operator.

    The key carries the block arrays' identities, as ``_cached_solver``'s does, and the entry holds
    them, so a matrix whose storage is replaced is expanded again. Sized off the ``values``
    capacity rather than ``nnz``: the row offsets alone bound every row, so a stale capacity's
    tail is allocated and never read.
    """
    identity = (id(matrix.offsets), id(matrix.columns), id(matrix.values))
    cached = _EXPANSION_CACHE.get(matrix)
    if cached is not None and cached[0][0] == identity:
        return cached[1]
    device = matrix.device
    n_rows = int(matrix.nrow)
    capacity = 4 * matrix.values.size
    offsets = _launch.empty(2 * n_rows + 1, dtype=wp.int32, device=device)
    columns = _launch.empty(capacity, dtype=wp.int32, device=device)
    values = _launch.empty(capacity, dtype=wp.float64, device=device)
    if n_rows == 0:
        _launch.zero_(offsets)
    else:
        _launch.launch(
            kernel_linalg.expand_block_csr_2x2,
            dim=n_rows,
            inputs=[matrix.offsets, matrix.columns, matrix.values],
            outputs=[offsets, columns, values],
            device=device,
        )
    expanded = _bsr_over(
        2 * n_rows, 2 * int(matrix.ncol), offsets, columns, values, 4 * int(matrix.nnz)
    )
    _EXPANSION_CACHE[matrix] = (
        (identity, (matrix.offsets, matrix.columns, matrix.values)),
        expanded,
    )
    return expanded


def _as_scalar_view(vectors: odt.ArrayNd) -> wp.array[wp.float64]:
    """View a contiguous ``wp.vec2d`` array's bytes as a flat ``float64`` array, with no copy."""
    return vectors.view(wp.float64).flatten()


def jacobi_preconditioner(matrix: odt.SparseMatrix) -> wpl.LinearOperator:
    """
    Jacobi preconditioner for a symmetric positive-definite operator.

    ``warp.optim.linear.preconditioner(matrix, "diag")``, recognized by
    [`solve_spd`][ordito.linalg.solve_spd] as such: on a scalar ``float64`` operator it then
    solves through this module's own conjugate gradient, whose recorded device loop is kept for
    the operator's lifetime and replayed on every later solve against it, where Warp's solver
    records a new one per call. Any other operator handed to ``solve_spd`` is honoured as given,
    through ``warp.optim.linear.cg``.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-(semi-)definite operator. Scalar or block dtype.

    Returns
    -------
    ``warp.optim.linear.LinearOperator``
        ``(n, n)`` inverse-diagonal operator.

    See Also
    --------
    [`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner]
    [`solve_spd`][ordito.linalg.solve_spd]
    """
    # Built on the first apply rather than here: every solve that recognizes the tag reads the
    # diagonal itself, so the inverse diagonal ``warp.optim.linear`` would extract -- a diagonal
    # gather, an allocation and a launch -- is needed only by a caller applying the operator.
    built: list[wpl.LinearOperator] = []

    def matvec(x: odt.ArrayNd, y: odt.ArrayNd, z: odt.ArrayNd, alpha: float, beta: float) -> None:
        if not built:
            built.append(wpl.preconditioner(matrix, "diag"))
        built[0].matvec(x, y, z, alpha, beta)

    operator = wpl.LinearOperator(matrix.shape, matrix.scalar_type, matrix.device, matvec)
    _PRECONDITIONER_TAGS[operator] = ("diag", weakref.ref(matrix))
    return operator


def solve_spd_columns(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    maxiter: int | None = None,
    check_every: int = CG_CHECK_EVERY,
    preconditioner: str | SquaredLaplacianPreconditioner = "diag",
    factorization: OperatorFactorization | None = None,
) -> tuple[int, float, float]:
    """
    Solve one symmetric positive-definite operator against several right-hand-side columns.

    Single batched conjugate-gradient call: all ``n_rhs`` columns advance together and convergence
    uses the worst-case residual, so the iteration count is the *maximum* over columns rather than
    their sum. ``solution`` is used as the initial guess (warm starting is therefore free) and is
    overwritten in place.

    !!! warning "``check_every=0`` returns device arrays, not host scalars"
        Under the default ``check_every=0`` nothing is ever read back, so the three returned values
        are 1-element **device arrays** — and the second and third are the *squared* residual norm
        and *squared* absolute tolerance, matching ``warp.optim.linear.cg``. Code that formats or
        compares them must call ``.numpy()`` first, which reintroduces the host sync the setting
        exists to avoid. Pass a positive ``check_every`` to get host scalars back. ``solution`` is
        unaffected either way.

        On a device without conditional CUDA graphs there is nowhere to put the device-side test,
        so ``check_every=0`` is replaced by
        [`CG_CHECK_EVERY_FALLBACK`][ordito.linalg.CG_CHECK_EVERY_FALLBACK] and host scalars come
        back instead. Warp's own behaviour in that case is to run every solve to ``maxiter`` —
        ``CG_MAXITER_FACTOR * n`` iterations of guaranteed waste — so substituting the cadence is
        the only usable reading of the request.


    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-definite operator.
    rhs
        ``(n_rhs, n)`` right-hand sides, one per row. Must be contiguous.
    solution
        ``(n_rhs, n)`` initial guess, overwritten with the result. Must be contiguous.
    tol
        Relative residual tolerance, as a ratio of the right-hand-side norm.
    maxiter
        Iteration cap. Defaults to ``CG_MAXITER_FACTOR * n``.
    check_every
        How many iterations run between residual tests. ``0`` (the default) tests every iteration
        on device; see Notes for the tradeoff and the warning above for what it does to the return
        type.
    preconditioner
        ``"diag"`` (the default) for the Jacobi preconditioner, ``"chebyshev"`` for the
        Jacobi-Chebyshev polynomial [`chebyshev_preconditioner`]
        [ordito.linalg.chebyshev_preconditioner] applies -- the choice for a long solve of a
        symmetric Laplacian-like system, and a loss on a short one -- ``"adaptive"`` for Jacobi
        under a probe of
        [`CG_CHEBYSHEV_PROBE_ITERATIONS`][ordito.linalg.CG_CHEBYSHEV_PROBE_ITERATIONS] rounds that
        escalates to that polynomial, warm-started, only if the probe has not converged -- the
        choice for a caller whose solves may be short or long -- ``"multigrid"`` for the
        smoothed-aggregation V-cycle [`multigrid_preconditioner`]
        [ordito.linalg.multigrid_preconditioner] builds, or ``"auto"`` to let the operator decide.
        ``"auto"`` first tests the assembled system against
        [`CG_MULTIGRID_DOMINANCE`][ordito.linalg.CG_MULTIGRID_DOMINANCE] and its size floor: a
        system that clears the gate builds the V-cycle outright, and one that does not runs Jacobi
        under [`CG_PROBE_ITERATIONS`][ordito.linalg.CG_PROBE_ITERATIONS] and escalates only if
        that has not converged. The V-cycle costs a setup pass and pays for itself only where the
        solve dominates the call, so ``"auto"`` is the setting for a caller whose systems vary --
        it cannot regress a solve that was already short, though it can forgo a small win on a
        small ill-conditioned system that the size floor declines. See that function's Notes, and
        both constants. For normal equations whose square block is a scaled Laplacian, pass the
        operator
        [`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]
        builds instead of a name.
    factorization
        ``None``, or an [`OperatorFactorization`][ordito.linalg.OperatorFactorization] of
        ``matrix``, which makes the solve *verified*. One that already holds a factorization is
        solved by directly, with no iteration. Otherwise the iteration runs at most
        [`CG_FACTOR_AFTER_ROUNDS`][ordito.linalg.CG_FACTOR_AFTER_ROUNDS] rounds (or ``maxiter``,
        if fewer), and its result must have converged and pass a componentwise backward-error
        test ([`VERIFIED_BACKWARD_ERROR`][ordito.linalg.VERIFIED_BACKWARD_ERROR]); one that does
        not is solved again by factoring ``matrix`` into ``factorization``, where the caller keeps
        it. An operator that cannot be factored keeps iterating, warm-started, to the full
        ``maxiter`` -- unless it is not definite in ``float64``, which no iteration solves either.
        The verification reads back one flag whatever ``check_every`` is.

    Returns
    -------
    tuple[int, float, float]
        ``(iterations, residual_norm, absolute_tolerance)`` on ``warp.optim.linear.cg``'s terms,
        with the residual taken over the worst column. Device arrays rather than host scalars under
        ``check_every=0``; see the warning above. Those arrays belong to the solver state kept for
        ``matrix`` and are overwritten by the next solve against it, so read them before then.

    Raises
    ------
    ValueError
        If ``preconditioner`` is not one of the five names above. It is rejected rather than
        treated as ``"diag"``, so a misspelling cannot turn into a silently slower solve. Or if
        ``factorization`` holds another operator than ``matrix``.
    RuntimeError
        If ``rhs`` and ``solution`` are not all on one device.

    Warns
    -----
    UserWarning
        When the solve exhausts ``maxiter`` without reaching ``tol``, on the same terms as
        [`solve_spd`][ordito.linalg.solve_spd]. A batched solve converges on its *worst* column,
        so hitting the cap here means at least one column is unsolved. Only detectable when
        ``check_every > 0`` -- or, under ``factorization``, whenever the result fails its
        verification and the operator cannot be factored.

    Notes
    -----
    ``check_every`` is a pure performance knob on every path but one -- it cannot change the
    converged answer, only how far past the tolerance the solver may overshoot before it notices.
    The exception is the exactly-two-column, ``"diag"``-preconditioned path: it shares one Krylov
    subspace across both columns, so a positive ``check_every`` can let a batch of iterations apply
    a real, coupled update to a column that already crossed its own tolerance. Its device-side
    default (``0``) scales with how much work a single ``cg`` call does, and the regimes disagree,
    so read the one that matches the caller:

    - **Cold single solves** -- one ``cg`` call from a zero initial guess, the shape ``harmonic`` /
      ``tutte`` / ``smooth_region`` take. The regime the default is set for, and a clear win there.
    - **Warm-started solves inside an iteration loop** -- ``arap``, whose right-hand side changes
      every
      iteration so each solve still runs tens of CG iterations from the previous answer. Neutral to
      mildly positive.
    - **Repeated near-converged solves over one
      [`spd_column_solver`][ordito.linalg.spd_column_solver] state** -- the same right-hand side
      re-solved back to back, so every call after the first converges in one or two iterations. Here
      the device-side check's fixed per-call cost dominates a solve that short, so it is a real
      loss; pass a positive ``check_every`` to a state driven that way.
    - **Raising ``check_every`` well above the default is a loss in every regime**: the readback it
      saves is cheap, while the extra iterations it can cause are real work -- the smaller the
      solve, the worse the trade.

    The *preconditioner* is not a knob worth turning beyond these four: IC(0) comes out a wash or a
    loss against Jacobi, and ``"chebyshev"`` wins only where the solve is long. See "Why Jacobi" in
    the [`ordito.linalg`][ordito.linalg] module documentation. The *per-iteration* cost of
    batching was a separate lever and has been taken: with more than one column this runs ordito's
    own conjugate gradient rather than ``warp.optim.linear``'s, because Warp's reduction degrades on
    precisely the batched input the worst-case stopping rule needs -- see "Whose conjugate gradient"
    there.

    See Also
    --------
    [`spd_column_solver`][ordito.linalg.spd_column_solver]
    [`replicated_operator`][ordito.linalg.replicated_operator]
    """
    require_same_device(rhs=rhs, solution=solution)
    if factorization is not None:
        if factorization.matrix is not matrix:
            raise ValueError("solve_spd_columns: factorization holds another operator than matrix")
        return _solve_columns_verified(
            matrix,
            rhs,
            solution,
            factorization,
            tol=tol,
            maxiter=maxiter,
            check_every=check_every,
            preconditioner=preconditioner,
        )
    result = _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=maxiter,
        check_every=check_every,
        preconditioner=preconditioner,
        run=True,
        caller="solve_spd_columns",
    )
    # ``check_every=0`` never warns, as in ``solve_spd``: not even on a device that tested the loop
    # from the host and so holds host scalars.
    if check_every > 0:
        _warn_if_not_converged(
            result,
            maxiter if maxiter is not None else CG_MAXITER_FACTOR * int(rhs.shape[1]),
            "solve_spd_columns",
        )
    return result


def _solve_columns_verified(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    factorization: OperatorFactorization,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str | SquaredLaplacianPreconditioner,
) -> tuple[Any, Any, Any]:
    """
    ``solve_spd_columns`` under a ``factorization``: iterate under a budget, verify, else factor.

    Rent or buy: the iteration may spend up to ``CG_FACTOR_AFTER_ROUNDS`` rounds, about what the
    factorization would cost, and its result is kept only if it converged and passes the
    componentwise backward-error test. Otherwise the factorization is bought and solves the
    system from zero. An operator that cannot be factored has only the iteration left, which then
    continues from its iterate to the full cap and warns if it still fails -- unless the refusal was
    the operator's definiteness, when it warns at once.
    """
    n = int(rhs.shape[1])
    cap = maxiter if maxiter is not None else CG_MAXITER_FACTOR * n
    if factorization.factorization is not None:
        factorization.solve(rhs, solution, tol=tol)
        return _direct_result(tol, check_every, rhs.device)
    rent = min(cap, CG_FACTOR_AFTER_ROUNDS)
    result = _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=rent,
        check_every=check_every,
        preconditioner=preconditioner,
        run=True,
        caller="solve_spd_columns",
    )
    if _verified(matrix, rhs, solution, result, tol):
        return result
    if factorization.factor():
        # From zero, not from the rejected iterate: refinement corrects an initial guess through
        # its residual, and the error of that correction scales with the residual's size, so a bad
        # guess leaves a backward error the refinement's step test can stop above (5.9e-7 against
        # 1.9e-12 from zero, on a graded biharmonic system).
        solution.zero_()
        factorization.solve(rhs, solution, tol=tol)
        return _direct_result(tol, check_every, rhs.device)
    # An operator too large to factor may still converge under the iteration; one the
    # factorization finds indefinite either way round is past what conjugate gradient reaches too.
    if cap > rent and not factorization._indefinite:  # pyright: ignore[reportPrivateUsage]
        result = _cg_columns(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=cap - rent,
            check_every=check_every,
            preconditioner=preconditioner,
            run=True,
            caller="solve_spd_columns",
        )
        if _verified(matrix, rhs, solution, result, tol):
            return result
    warnings.warn(
        "solve_spd_columns: conjugate gradient did not reach a verified solution in "
        f"{cap} iterations and the operator cannot be factored (over "
        "CHOLESKY_MEMORY_BUDGET, or not definite in float64); the result is the last iterate, "
        "not a solution.",
        stacklevel=3,
    )
    return result


def _verified(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    result: tuple[Any, Any, Any],
    tol: float,
) -> bool:
    """
    Whether a conjugate-gradient result converged and its solution passes the backward-error test.

    The convergence half reads the solver's own residual against its threshold; the other half
    recomputes the residual from ``matrix``, entry by entry against ``(|A| |x| + |b|)_i``, which a
    recurrence residual drifting from the true one -- an operator ill-conditioned in places --
    cannot fool. Both write one device flag, read back once.
    """
    _iterations, residual, threshold = result
    if not isinstance(residual, wp.array) and float(residual) > float(threshold):
        return False
    device = rhs.device
    n_columns, n = int(rhs.shape[0]), int(rhs.shape[1])
    flag = _launch.zeros(1, dtype=wp.int32, device=device)
    if isinstance(residual, wp.array):
        _launch.launch(
            kernel_linalg.flag_unconverged_columns,
            dim=residual.shape[0],
            inputs=[residual, threshold],
            outputs=[flag],
            device=device,
        )
    _launch.launch(
        kernel_cholesky.backward_error_exceeds,
        dim=(n, n_columns),
        inputs=[matrix.offsets, matrix.columns, matrix.values, solution.flatten(), rhs.flatten(),
                wp.int32(n), wp.float64(max(VERIFIED_BACKWARD_ERROR, 100.0 * tol))],
        outputs=[flag],
        device=device,
    )  # fmt: skip
    return read_scalar(flag, 0) == 0


def spd_column_solver(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    maxiter: int | None = None,
    check_every: int = CG_CHECK_EVERY,
    preconditioner: str = "diag",
    factorize: bool = True,
    coordinates: wp.array[wp.vec3] | None = None,
) -> _BatchedCg | _AdaptiveCg | _FactoringColumns:
    """
    Pre-allocated batched conjugate-gradient state, for repeated solves of one operator.

    Same solve as [`solve_spd_columns`][ordito.linalg.solve_spd_columns], but the temporary buffers
    and the batch layout are allocated once and the returned state is callable. Build it *outside* a
    per-iteration loop and call it inside: ``solve_spd_columns`` would otherwise reallocate its
    temporaries and rebuild the operator wrappers on every iteration.

    ``rhs`` and ``solution`` are captured at construction, so a loop that overwrites those same
    buffers in place can call the state with no arguments.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-definite operator. Held by the returned state.
    rhs
        ``(n_rhs, n)`` right-hand sides; re-read from this buffer on every call.
    solution
        ``(n_rhs, n)`` initial guess and result; warm-started from its current contents each call.
    tol
        Relative residual tolerance.
    maxiter
        Iteration cap. Defaults to ``CG_MAXITER_FACTOR * n``.
    check_every
        Iterations between residual tests; see
        [`solve_spd_columns`][ordito.linalg.solve_spd_columns] for the tradeoff and for what the
        default ``0`` does to the values each call returns.
    preconditioner
        ``"diag"``, ``"chebyshev"``, ``"adaptive"`` or ``"multigrid"``, as in
        [`solve_spd_columns`][ordito.linalg.solve_spd_columns]. Built once here and reused by every
        call against this state, which is the shape the V-cycle's setup cost wants; ``"adaptive"``
        decides on every call, and builds its polynomial on the first call that escalates.
        ``"auto"`` is **not** accepted: it decides on the first solve -- from that system's
        operator, and from the probe when the operator does not settle it -- and a hoisted state
        exists to be driven many times.
    factorize
        From the second call on, solve by a [`sparse_cholesky`][ordito.cholesky.sparse_cholesky]
        factorization of ``matrix`` that the returned state builds and owns, refined to ``tol``,
        rather than iterating. The first call runs the conjugate gradient: one solve does not repay
        a factorization. A factorized call reports zero iterations. ``matrix``'s values must not be
        rewritten in place between calls under ``factorize``, which the factorization would not
        see; a ``float64`` operator only (any other iterates whatever this says).
    coordinates
        ``(n,)`` positions of ``matrix``'s rows, for that factorization's ordering.

    Returns
    -------
    internal batched state
        Callable solver state, this module's own ``_BatchedCg`` at every column count, kept
        internal because a caller never has to name it (see the
        [`ordito.linalg`][ordito.linalg] module's "Whose conjugate gradient").
        Substituted operands must match the construction-time shape, dtype, device and batch
        layout. Each call returns what
        [`solve_spd_columns`][ordito.linalg.solve_spd_columns] returns, including its
        ``check_every=0`` device arrays.

    Raises
    ------
    ValueError
        If ``preconditioner`` is ``"auto"``, for the reason above, or is not one of the four names
        this accepts.
    RuntimeError
        If ``rhs``, ``solution`` and ``coordinates`` are not all on one device.

    Notes
    -----
    The factorization lives as long as the returned state, and holds device memory of the order
    of the operator's fill-in until the state is dropped.

    Examples
    --------
    ```python
    solver = spd_column_solver(q_uu, rhs, solution, tol=1e-10)
    for _ in range(max_iterations):
        ...  # rewrite rhs in place
        solver()
    ```

    See Also
    --------
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    [`OperatorFactorization`][ordito.linalg.OperatorFactorization]
    """
    require_same_device(rhs=rhs, solution=solution, coordinates=coordinates)
    state = _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=maxiter,
        check_every=check_every,
        preconditioner=preconditioner,
        run=False,
        caller="spd_column_solver",
    )
    if not factorize or matrix.values.dtype != wp.float64:
        return state
    return _FactoringColumns(
        state, OperatorFactorization(matrix, coordinates), rhs, solution, tol, check_every
    )


class _FactoringColumns:
    """
    ``spd_column_solver``'s state under ``factorize``: the iteration, then the factorization.

    The first call runs the conjugate-gradient state; the second factors the operator into the
    ``OperatorFactorization`` this state owns, and it and every later one solve by it, reading
    ``rhs`` and writing ``solution`` as the iteration does. An operator that cannot be factored
    keeps iterating.
    """

    def __init__(
        self,
        state: _BatchedCg | _AdaptiveCg,
        factorization: OperatorFactorization,
        rhs: odt.Array2dFloat,
        solution: odt.Array2dFloat,
        tol: float,
        check_every: int,
    ) -> None:
        self._state = state
        self._factorization = factorization
        self._rhs = rhs
        self._solution = solution
        self._tol = tol
        self._check_every = check_every
        self._calls = 0

    def __call__(self) -> tuple[Any, Any, Any]:
        """Solve, by the iteration on the first call and by the factorization after it."""
        self._calls += 1
        if self._calls == 1 or not self._factorization.factor():
            return self._state()
        self._factorization.solve(self._rhs, self._solution, tol=self._tol)
        return _direct_result(self._tol, self._check_every, self._rhs.device)


class OperatorFactorization:
    """
    A sparse Cholesky factorization of one operator, built when asked and owned by its holder.

    The place a caller keeps the factorization of an operator it solves repeatedly: built by
    [`factor`][ordito.linalg.OperatorFactorization.factor] (once; an operator that cannot be
    factored is remembered as such), used by [`solve`][ordito.linalg.OperatorFactorization.solve]
    and by [`solve_spd_settled`][ordito.linalg.solve_spd_settled] when passed there, and dropped by
    [`release`][ordito.linalg.OperatorFactorization.release]. Nothing in this module keeps one
    across calls on its own: whoever holds this object decides how long the factorization lives.

    A ``wp.mat22d``-block operator is factored as its scalar expansion, and a negative
    semi-definite one (a cotangent Laplacian) negated.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive- or negative-(semi-)definite operator, scalar ``float64`` or
        ``wp.mat22d``-block. Held by this object; its values must not be rewritten in place once
        factored.
    coordinates
        ``(n,)`` positions of ``matrix``'s (block) rows -- a mesh's vertices for a vertex operator
        -- for the factorization's ordering. ``None`` orders by the pattern alone.

    Raises
    ------
    ValueError
        If ``matrix`` is neither ``float64`` nor ``wp.mat22d``-block.
    RuntimeError
        If ``matrix`` and ``coordinates`` are not on one device.

    Notes
    -----
    A built factorization holds device memory of the order of the operator's fill-in, reported by
    [`nbytes`][ordito.linalg.OperatorFactorization.nbytes], until it is released or this object
    is dropped.

    See Also
    --------
    [`sparse_cholesky`][ordito.cholesky.sparse_cholesky]
    [`solve_spd_settled`][ordito.linalg.solve_spd_settled]
    """

    def __init__(
        self, matrix: odt.SparseMatrix, coordinates: wp.array[wp.vec3] | None = None
    ) -> None:
        """Hold ``matrix`` (and the ordering's ``coordinates``) with nothing factored yet."""
        require_same_device(matrix=matrix, coordinates=coordinates)
        if matrix.values.dtype not in (wp.float64, wp.mat22d):
            raise ValueError(
                "OperatorFactorization factors a float64 or wp.mat22d operator, got "
                f"{matrix.values.dtype}"
            )
        self._matrix = matrix
        self._coordinates = coordinates
        self._factorization: SparseCholesky | None = None
        self._refused = False
        # Whether the refusal was the operator's definiteness, not its size.
        self._indefinite = False

    @property
    def matrix(self) -> odt.SparseMatrix:
        """The operator this factors."""
        return self._matrix

    @property
    def factorization(self) -> SparseCholesky | None:
        """The built factorization; ``None`` before ``factor`` and after ``release``."""
        return self._factorization

    @property
    def nbytes(self) -> int:
        """Device memory the built factorization holds, in bytes; ``0`` when none is."""
        return 0 if self._factorization is None else self._factorization.nbytes

    def factor(self) -> bool:
        """
        Factor the operator unless it already is, or could not be.

        Returns
        -------
        bool
            Whether a factorization is now held: ``False`` when the operator is over
            [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET] or not definite,
            which is remembered so the attempt is not repeated.
        """
        if self._factorization is not None:
            return True
        if self._refused:
            return False
        blocks = odt.has_blocks(self._matrix, wp.mat22d)
        scalar = (
            _scalar_expansion(self._matrix)
            if blocks
            else cast("odt.BsrMatrix[wp.float64]", self._matrix)
        )
        points = _expanded_coordinates(self._coordinates, 2 if blocks else 1)
        indefinite = True
        for negated in (False, True):
            try:
                self._factorization = sparse_cholesky(
                    _storage_alias(scalar), points, negated=negated
                )
            except NotPositiveDefiniteError:
                continue
            except ValueError:
                indefinite = False
                continue
            return True
        self._refused = True
        self._indefinite = indefinite
        return False

    def solve(
        self, rhs: odt.ArrayNd, solution: odt.ArrayNd, *, tol: float, componentwise: bool = False
    ) -> None:
        """
        Solve by the built factorization, refined to ``tol``.

        Parameters
        ----------
        rhs
            ``(n,)`` right-hand side (``wp.vec2d`` against a block operator), or ``(n_columns, n)``
            ``float64`` columns against a scalar one.
        solution
            Same shape and dtype as ``rhs``: the initial guess, overwritten with the answer.
        tol
            Refinement tolerance, as [`SparseCholesky.solve`][ordito.cholesky.SparseCholesky.solve]
            reads it.
        componentwise
            Refine every entry to its own size.

        Raises
        ------
        ValueError
            If nothing is factored.
        """
        if self._factorization is None:
            raise ValueError("OperatorFactorization.solve: nothing is factored; call factor()")
        if rhs.dtype == wp.vec2d:
            rhs, solution = _as_scalar_view(rhs), _as_scalar_view(solution)
        self._factorization.solve(rhs, solution, tol=tol, componentwise=componentwise)

    def release(self) -> None:
        """Drop the factorization and forget a refused attempt; the next ``factor`` builds anew."""
        self._factorization = None
        self._refused = False
        self._indefinite = False


def solve_spd_settled(
    matrix: odt.SparseMatrix,
    rhs: odt.ArrayNd,
    solution: odt.ArrayNd,
    *,
    check_rounds: int,
    change_tolerance: float,
    settle_rounds: int,
    maxiter: int | None = None,
    preconditioner: wpl.LinearOperator | None = None,
    factorization: OperatorFactorization | None = None,
) -> wp.array[wp.int32]:
    """
    Run Jacobi conjugate gradient until every entry of the solution has settled relative to itself.

    A stopping rule for systems whose solution spans many orders of magnitude, where a residual
    tolerance says nothing about the small entries: the heat method's diffusions, whose far field
    sits hundreds of orders of magnitude below the source and is still zero, or noise, long after
    the residual has converged. The solve runs at a zero residual tolerance, one continuous
    iteration, and every ``check_rounds`` rounds compares the iterate with the one ``check_rounds``
    earlier: it stops once no entry has newly become non-zero and none moved by
    ``change_tolerance`` or more of its own value -- or ``settle_rounds`` rounds after the count of
    non-zero entries last changed, the bound for an entry that never settles because it cancels
    toward zero. Or at ``maxiter``.

    The iteration reads a ``float64`` operator's values in ``float32``, widened as they are read
    and accumulated in ``float64``: past a few tens of thousands of rows a round is bound by its
    bytes and the values are its largest stream. The system solved therefore differs from
    ``matrix`` by one rounding per stored entry, which on the heat method's operators is orders of
    magnitude below the method's own discretization error. The residual, the iterate and every
    reduction stay ``float64``.

    The check is tested on the device, inside the solve's recorded loop. As with
    [`solve_spd`][ordito.linalg.solve_spd], the solver state is kept for ``matrix``'s lifetime and
    a later solve against it replays the recorded loop.

    The settled iterate is then verified: where its componentwise backward error exceeds
    [`SETTLE_BACKWARD_ERROR`][ordito.linalg.SETTLE_BACKWARD_ERROR] -- conjugate gradient's
    round-off swamping entries far below the peak, on an operator ill-conditioned in places -- the
    system is solved again by a [`sparse_cholesky`][ordito.cholesky.sparse_cholesky]
    factorization, refined componentwise. The verification reads back one flag. The factorization
    is built in ``factorization`` when one is passed, where the caller keeps it; and a
    ``factorization`` that already holds one is solved by directly, with no iteration.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-definite operator, scalar ``float64`` or ``wp.mat22d``-block.
    rhs
        ``(n,)`` right-hand side, ``float64`` or ``wp.vec2d`` matching ``matrix``, or
        ``(n_columns, n)`` ``float64`` columns against a scalar ``matrix``, solved together.
    solution
        ``(n,)`` or ``(n_columns, n)`` initial guess, overwritten with the result. Same shape and
        dtype as ``rhs``, contiguous.
    check_rounds
        Rounds between checks.
    change_tolerance
        Largest relative change per entry over ``check_rounds`` rounds that counts as settled.
    settle_rounds
        Rounds after the non-zero count last changed at which the solve stops regardless.
    maxiter
        Iteration cap. When ``None``, uses
        [`CG_MAXITER_FACTOR`][ordito.linalg.CG_MAXITER_FACTOR] times the number of rows.
    preconditioner
        ``None``, or the operator [`jacobi_preconditioner`][ordito.linalg.jacobi_preconditioner]
        built for ``matrix``: the solve is Jacobi-preconditioned either way.
    factorization
        ``None``, or an [`OperatorFactorization`][ordito.linalg.OperatorFactorization] of
        ``matrix``: the caller-owned place for the fallback's factorization (built there, and so
        kept for the caller's next solve), and the direct solve when it already holds one. Without
        it a fallback's factorization is built for this solve and dropped.

    Returns
    -------
    wp.array[wp.int32]
        ``(1,)`` device array holding the rounds that took a step; zero when the solve went to a
        held factorization directly. It belongs to the solver
        state kept for ``matrix`` and is overwritten by the next solve against it -- or, for an
        operator rebuilt per call, against the next operator of its shape, whose solve takes the
        same state (see [`solve_spd`][ordito.linalg.solve_spd]).

    Raises
    ------
    ValueError
        If ``matrix`` and ``rhs`` are not one of the three pairings above, ``preconditioner`` is
        not ``None`` or ``matrix``'s own Jacobi preconditioner, or ``factorization`` is not
        ``matrix``'s.
    RuntimeError
        If ``rhs`` and ``solution`` are not all on one device.

    Warns
    -----
    UserWarning
        When the settled iterate fails the backward-error check and no factorization can be built
        (over [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], or not positive
        definite); the iterate is returned.

    See Also
    --------
    [`solve_spd`][ordito.linalg.solve_spd]
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    [`OperatorFactorization`][ordito.linalg.OperatorFactorization]
    """
    require_same_device(rhs=rhs, solution=solution)
    if factorization is not None and factorization.matrix is not matrix:
        raise ValueError("solve_spd_settled: factorization holds another operator than matrix")
    columns = rhs.ndim == 2
    if columns:
        if (
            not odt.has_blocks(matrix, wp.float64)
            or rhs.dtype != wp.float64
            or preconditioner is not None
        ):
            raise ValueError(
                "solve_spd_settled solves (n_columns, n) float64 columns against a float64 matrix "
                "with no preconditioner argument"
            )
        n_columns, n_rows = int(rhs.shape[0]), int(rhs.shape[1])
        rhs_flat, solution_flat = rhs.flatten(), solution.flatten()
    else:
        dtype_ok = (matrix.values.dtype, rhs.dtype) in (
            (wp.float64, wp.float64),
            (wp.mat22d, wp.vec2d),
        )
        if not dtype_ok or _batched_preconditioner_kind(matrix, rhs, preconditioner) != "diag":
            raise ValueError(
                "solve_spd_settled solves a float64 or wp.mat22d system under its own Jacobi "
                "preconditioner"
            )
        n_columns, n_rows = 1, int(rhs.shape[0])
        rhs_flat, solution_flat = rhs, solution
    if odt.has_blocks(matrix, wp.mat22d):
        # Solved as its scalar expansion, as ``solve_spd`` does (``_solve_spd_batched``).
        scalar = _scalar_expansion(matrix)
        rhs_flat, solution_flat = _as_scalar_view(rhs), _as_scalar_view(solution)
    else:
        scalar = cast("odt.BsrMatrix[wp.float64]", matrix)
    if factorization is not None and factorization.factorization is not None:
        # The caller holds this operator's factorization: solve by it directly.
        factorization.solve(
            _columns_view(rhs_flat, n_columns), _columns_view(solution_flat, n_columns),
            tol=SETTLE_DIRECT_TOLERANCE, componentwise=True,
        )  # fmt: skip
        return _launch.zeros(1, dtype=wp.int32, device=rhs.device)
    solver = _cached_solver(
        scalar,
        n_columns,
        tol=0.0,
        maxiter=CG_MAXITER_FACTOR * n_rows if maxiter is None else maxiter,
        check_every=_supported_check_every(0),
        preconditioner="diag",
        settle=(int(check_rounds), float(change_tolerance), int(settle_rounds)),
        narrow_values=True,
        pooled=True,
    )
    solver.solve(rhs_flat, solution_flat)
    # The settle rule decides *when to stop*, not whether the iterate is right: on an operator
    # whose conditioning is extreme in places (a strongly graded mesh, a scan's slivers) conjugate
    # gradient's round-off swamps the far field's entries long before they settle, and the
    # iterate "settles" on garbage. The componentwise backward error of the result catches it
    # exactly (a settled solve sits orders of magnitude below the bar, a failed one at 1), and the
    # direct factorization -- which reaches every entry with its own relative precision -- takes
    # over. One four-byte read: the decision to factor is the host's.
    flag = _launch.zeros(1, dtype=wp.int32, device=rhs.device)
    _launch.launch(
        kernel_cholesky.backward_error_exceeds,
        dim=(int(scalar.nrow), n_columns),
        inputs=[scalar.offsets, scalar.columns, scalar.values, solution_flat, rhs_flat,
                wp.int32(int(scalar.nrow)), wp.float64(SETTLE_BACKWARD_ERROR)],
        outputs=[flag],
        device=rhs.device,
    )  # fmt: skip
    if read_scalar(flag, 0) == 0:
        return solver.iterations
    fallback = factorization if factorization is not None else OperatorFactorization(matrix)
    if not fallback.factor():
        warnings.warn(
            "solve_spd_settled: the settled iterate fails the componentwise backward-error check "
            "and no factorization of the operator can be built; returning the iterate",
            stacklevel=2,
        )
        return solver.iterations
    fallback.solve(
        _columns_view(rhs_flat, n_columns), _columns_view(solution_flat, n_columns),
        tol=SETTLE_DIRECT_TOLERANCE, componentwise=True,
    )  # fmt: skip
    return solver.iterations


def _direct_result(tol: float, check_every: int, device: wp.DeviceLike) -> tuple[Any, Any, Any]:
    """Return a factorized solve's report on ``solve_spd``'s terms: no iterations, at ``tol``."""
    if check_every > 0:
        return (0, tol, tol)
    return (
        _launch.zeros(1, dtype=wp.int32, device=device),
        _launch.full(1, tol * tol, dtype=wp.float64, device=device),
        _launch.full(1, tol * tol, dtype=wp.float64, device=device),
    )


def _columns_view(vector: wp.array[wp.float64], n_columns: int) -> odt.ArrayNd:
    """Return a flat ``float64`` vector as ``n_columns`` columns, or itself for one."""
    if n_columns == 1:
        return vector
    return vector.reshape((n_columns, vector.size // n_columns))


def _expanded_coordinates(
    coordinates: wp.array[wp.vec3] | None, rows_per_point: int
) -> wp.array[wp.vec3] | None:
    """Return one position per scalar row: each block row's, ``rows_per_point`` times."""
    if coordinates is None or rows_per_point == 1:
        return coordinates
    # Host-side: ``sparse_cholesky`` reads its coordinates back once anyway.
    expanded = np.repeat(coordinates.numpy(), rows_per_point, axis=0)
    return _launch.array(expanded, dtype=wp.vec3, device=coordinates.device)


@overload
def _cg_columns(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str | SquaredLaplacianPreconditioner,
    run: Literal[True],
    caller: str,
) -> tuple[int, float, float]: ...
@overload
def _cg_columns(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str | SquaredLaplacianPreconditioner,
    run: Literal[False],
    caller: str,
) -> _BatchedCg | _AdaptiveCg: ...
def _cg_columns(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str | SquaredLaplacianPreconditioner,
    run: bool,
    caller: str,
) -> tuple[int, float, float] | _BatchedCg | _AdaptiveCg:
    """
    Batched conjugate gradient over the columns of ``rhs`` -- the shared body of the two solvers.

    [`solve_spd_columns`][ordito.linalg.solve_spd_columns] and
    [`spd_column_solver`][ordito.linalg.spd_column_solver] build the same replicated operator and
    the same replicated Jacobi preconditioner and differ only in ``run``: the first drives the
    solve, the second hands back the un-run state for a caller to drive repeatedly.
    """
    n_columns, n = int(rhs.shape[0]), int(rhs.shape[1])
    iteration_cap = maxiter if maxiter is not None else CG_MAXITER_FACTOR * n
    if preconditioner == "auto":
        if not run:
            raise ValueError(
                f'{caller} cannot take preconditioner="auto": it decides on the *first* solve, '
                'and a hoisted state is built to be driven many times. Pass "diag" or '
                '"multigrid".'
            )
        return _cg_columns_auto(
            matrix,
            rhs,
            solution,
            tol=tol,
            cap=iteration_cap,
            check_every=check_every,
            caller=caller,
        )
    # Validated rather than left to fall through. Every branch below tests for one name and treats
    # anything else as ``"diag"``, so without this an unrecognised string -- ``"jacobi"``,
    # ``"amg"``, a capitalised ``"Multigrid"`` -- runs a Jacobi solve silently, bit-identically to
    # ``"diag"`` and with nothing but the clock to tell the caller.
    if not isinstance(preconditioner, SquaredLaplacianPreconditioner):
        _require_preconditioner_name(preconditioner, caller)
    if preconditioner == "adaptive":
        state = _AdaptiveCg(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=iteration_cap,
            check_every=_supported_check_every(check_every),
            cached=run,
        )
        return cast("tuple[int, float, float]", state()) if run else state
    # Every column count goes to ``_BatchedCg``, one column included. Warp's own ``cg`` reaches its
    # fast unbatched reduction for a single column, but at ``check_every=0`` it records,
    # instantiates and launches a fresh conditional graph on every call, which on the small and
    # medium systems here costs more than the iterations do; ``_BatchedCg`` records once per state,
    # and the solve (``run=True``) goes through a state kept per operator, so a later solve against
    # the same operator replays it (``_cached_solver``).
    #
    # A block-CG variant that shares one Krylov subspace across exactly two columns
    # (O'Leary 1980) was built for this branch, shipped, and removed again. It does reduce the
    # iteration count on a *uniform* mesh, but its iteration costs two more launches than this
    # one's, so the wall clock is a wash there, and on every other system reached in this
    # package it is neutral or a loss -- worst on a *graded* mesh, where the count goes the
    # other way by a factor of two. Do not reintroduce it without deflation and a measurement on
    # a graded mesh.
    supported = _supported_check_every(check_every)
    if run and _one_block_eligible(matrix, n, preconditioner):
        return _cg_one_block(
            matrix,
            rhs.flatten(),
            solution.flatten(),
            n_columns,
            tol=tol,
            maxiter=iteration_cap,
            check_every=supported,
            preconditioner=preconditioner,
        )
    if not run:
        return _BatchedCg(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=iteration_cap,
            check_every=supported,
            preconditioner=preconditioner,
        )
    state = _cached_solver(
        matrix,
        n_columns,
        tol=tol,
        maxiter=iteration_cap,
        check_every=supported,
        preconditioner=preconditioner,
        pooled=True,
    )
    return cast("tuple[int, float, float]", state.solve(rhs.flatten(), solution.flatten()))


def _require_preconditioner_name(preconditioner: str, caller: str) -> None:
    """
    Raise unless ``preconditioner`` names one the column solvers build.

    Validated rather than left to fall through: every branch of ``_cg_columns`` tests for one name
    and treats anything else as ``"diag"``, so an unrecognised string -- ``"jacobi"``, ``"amg"``, a
    capitalised ``"Multigrid"`` -- would otherwise run a Jacobi solve silently, bit-identically to
    ``"diag"`` and with nothing but the clock to tell the caller.
    """
    if preconditioner not in ("diag", "chebyshev", "adaptive", "multigrid", "auto"):
        raise ValueError(
            f'{caller}: unknown preconditioner {preconditioner!r}, expected "diag", "chebyshev", '
            '"adaptive", "multigrid" or "auto".'
        )


def _one_block_eligible(
    matrix: odt.SparseMatrix, n: int, preconditioner: str | SquaredLaplacianPreconditioner
) -> TypeGuard[odt.BsrMatrix[wp.float64]]:
    """Whether a solve can run as ``_cg_one_block``: small, compact ``float64``, a fitting one."""
    fitting = isinstance(preconditioner, SquaredLaplacianPreconditioner) or preconditioner == "diag"
    return (
        fitting
        and 0 < n <= CG_ONE_BLOCK_MAX_ROWS
        and odt.has_blocks(matrix, wp.float64)
        and matrix.row_counts is None
    )


def _cg_one_block(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.ArrayNdFloat,
    solution: odt.ArrayNdFloat,
    n_columns: int,
    *,
    tol: float,
    maxiter: int,
    check_every: int,
    preconditioner: str | SquaredLaplacianPreconditioner,
) -> tuple[Any, Any, Any]:
    """
    Solve a small system in one launch, one block per column (``kernel_cg.cg_one_block``).

    ``rhs`` and ``solution`` are the flat ``n_columns * n`` views of the caller's contiguous
    buffers, and ``solution`` is the initial guess, overwritten. Returns ``_BatchedCg``'s three
    values on the same terms: device arrays -- the round count and each column's squared residual
    and threshold -- under a zero ``check_every`` on a CUDA device, host scalars otherwise.
    """
    device = matrix.device
    n = int(matrix.nrow)
    total = n_columns * n
    polynomial = isinstance(preconditioner, SquaredLaplacianPreconditioner)
    factor = factor_t = None
    factor_values = factor_t_values = steps = narrow = None
    if isinstance(preconditioner, SquaredLaplacianPreconditioner):
        factor, factor_t = preconditioner._factor, preconditioner._factor_t  # pyright: ignore[reportPrivateUsage]
        factor_values, factor_t_values = preconditioner.narrowed()
        steps = preconditioner._steps  # pyright: ignore[reportPrivateUsage]
        narrow = _launch.empty(
            kernel_cg.ONE_BLOCK_NARROW_SLOTS * total, dtype=wp.float32, device=device
        )
    # One buffer for the working vectors and the three per-column results -- residual, threshold
    # and round count -- which the kernel writes before anything reads them. The integer count is
    # only needed as a device array, so a host caller allocates none and reads one buffer.
    scratch = _launch.empty(
        kernel_cg.ONE_BLOCK_SLOTS * total + 3 * n_columns, dtype=wp.float64, device=device
    )
    results = odt.as_dense(scratch[kernel_cg.ONE_BLOCK_SLOTS * total :])
    residual = odt.as_dense(results[:n_columns])
    threshold = odt.as_dense(results[n_columns : 2 * n_columns])
    steps_out = odt.as_dense(results[2 * n_columns :])
    device_result = check_every == 0 and device.is_cuda
    iterations = _launch.zeros(1, dtype=wp.int32, device=device) if device_result else None
    _launch.launch_tiled(
        kernel_cg.cg_one_block,
        dim=[n_columns],
        inputs=[
            wp.int32(n),
            wp.int32(n_columns),
            wp.int32(maxiter),
            wp.float64(tol * tol),
            matrix.offsets,
            matrix.columns,
            matrix.values,
            wp.int32(1 if polynomial else 0),
            None if factor is None else factor.offsets,
            None if factor is None else factor.columns,
            factor_values,
            None if factor_t is None else factor_t.offsets,
            None if factor_t is None else factor_t.columns,
            factor_t_values,
            steps,
            rhs,
            scratch,
            narrow,
            wp.int32(1 if device_result else 0),
        ],
        outputs=[solution, iterations, residual, threshold, steps_out],
        block_dim=_CG_ONE_BLOCK_WIDE if n >= CG_ONE_BLOCK_WIDE_FROM else _CG_ONE_BLOCK_NARROW,
        device=device,
    )
    if device_result:
        return iterations, residual, threshold
    results_np = results.numpy()
    return (
        int(results_np[2 * n_columns :].max()),
        math.sqrt(float(results_np[:n_columns].max())),
        math.sqrt(float(results_np[n_columns : 2 * n_columns].max())),
    )


def _cg_columns_auto(
    matrix: odt.BsrMatrix[wp.float64],
    rhs: odt.Array2dFloat,
    solution: odt.Array2dFloat,
    *,
    tol: float,
    cap: int,
    check_every: int,
    caller: str,
) -> tuple[int, float, float]:
    """
    Run Jacobi under ``CG_PROBE_ITERATIONS``, then escalate if it has not converged.

    A system that finishes inside the probe pays nothing at all for the option -- the probe *is* the
    solve. One that does not is the ill-conditioned kind
    [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner] is for.

    The escalated solve does warm-start from the iterate the probe left in ``solution``, but **do
    not read that as the probe being cheap**: its cost is very nearly additive on top of the
    V-cycle's, because Jacobi leaves a residual whose low-frequency part is exactly what the
    V-cycle then has to work down, so the probe's iterates carry over close to nothing.

    See [`CG_PROBE_ITERATIONS`][ordito.linalg.CG_PROBE_ITERATIONS] for why this is a cap rather
    than a predictor, and [`CG_MULTIGRID_DOMINANCE`][ordito.linalg.CG_MULTIGRID_DOMINANCE] for the
    gate that runs first and decides, on the operator alone, whether to skip the probe entirely.
    """
    if _wants_multigrid(matrix):
        return _cg_columns(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=cap,
            check_every=check_every,
            preconditioner="multigrid",
            run=True,
            caller=caller,
        )
    if cap <= CG_PROBE_ITERATIONS:
        return _cg_columns(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=cap,
            check_every=check_every,
            preconditioner="diag",
            run=True,
            caller=caller,
        )
    probe = _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=CG_PROBE_ITERATIONS,
        check_every=check_every,
        preconditioner="diag",
        run=True,
        caller=caller,
    )
    residual, tolerance = _cg_residual_and_tolerance(probe)
    if residual <= tolerance:
        return probe
    # ``cap`` is the caller's total iteration budget, and the probe above already spent
    # ``CG_PROBE_ITERATIONS`` of it (this branch is only reached once ``cap > CG_PROBE_ITERATIONS``,
    # by the early return above) -- so the escalated call, which is a fresh ``wpl.cg`` call with its
    # own iteration counter, gets what remains rather than the full ``cap`` again.
    return _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=cap - CG_PROBE_ITERATIONS,
        check_every=check_every,
        preconditioner="multigrid",
        run=True,
        caller=caller,
    )


def _wants_multigrid(matrix: odt.BsrMatrix[wp.float64]) -> bool:
    """
    Whether ``preconditioner="auto"`` should skip its probe and build a hierarchy outright.

    The gate is a size floor crossed with the operator's off-diagonal dominance -- see
    [`CG_MULTIGRID_DOMINANCE`][ordito.linalg.CG_MULTIGRID_DOMINANCE]. Declining is always safe: the
    caller then runs the Jacobi probe it would have run anyway.
    """
    n_rows = int(matrix.nrow)
    if n_rows < CG_MULTIGRID_MIN_UNKNOWNS:
        return False
    dominance = _offdiagonal_dominance(matrix)
    if dominance > CG_MULTIGRID_DOMINANCE:
        return True
    return n_rows >= CG_MULTIGRID_LARGE_UNKNOWNS and dominance > CG_MULTIGRID_SIZE_FLOOR


def _offdiagonal_dominance(matrix: odt.BsrMatrix[wp.float64]) -> float:
    """
    ``max_i sum_{j != i} |A_ij| / A_ii`` over the rows of a scalar-block CSR operator.

    One launch and one max-reduction, so the whole thing is cheap relative to the solve it is
    deciding about, essentially independent of the operator's size. The only host readback is the
    reduced scalar. The diagonal is taken in magnitude, so either sign convention reads the same,
    and a row with a zero diagonal contributes 0 rather than an infinity -- see the kernel.
    """
    n_rows = int(matrix.nrow)
    device = matrix.values.device
    ratios = odt.empty_1d(n_rows, wp.float64, device=device)
    _launch.launch(
        kernel_linalg.offdiagonal_dominance_rows,
        dim=n_rows,
        # ``nnz`` is a stale capacity, but ``offsets``/``columns``/``values`` are indexed through
        # the row offsets here rather than sliced by it, so no count is read off the matrix.
        inputs=[matrix.offsets, matrix.columns, matrix.values, ratios],
        device=device,
    )
    return float(od.reduce.max(ratios))


def _cg_residual_and_tolerance(result: tuple[Any, ...]) -> tuple[float, float]:
    """
    Unwrap a conjugate-gradient result's worst-column residual norm and tolerance as host floats.

    ``check_every=0`` returns 1-element *device* arrays holding the **squared** residual norm and
    squared absolute tolerance, one entry per column, and a positive cadence returns their square
    roots as host scalars already reduced over the columns -- so the unwrapping differs and the
    quantity does not. One readback in the device case when the two arrays are adjacent views of
    one buffer -- both of this module's solvers lay them out so -- and two otherwise.
    """
    _iterations, residual, tolerance = result
    if not isinstance(residual, wp.array):
        return float(residual), float(tolerance)
    residual_np, tolerance_np = _read_pair(residual, tolerance)
    worst = int(np.argmax(residual_np / np.maximum(tolerance_np, 1e-300)))
    return math.sqrt(float(residual_np[worst])), math.sqrt(float(tolerance_np[worst]))


def _read_pair(
    first: wp.array[wp.float64], second: wp.array[wp.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Read two same-length ``float64`` device arrays back, in one copy when they are adjacent.

    The solvers here return a column's residual and threshold as neighbouring views of one buffer
    (in either order), so the span covering both is one contiguous array and one readback.
    """
    n = int(first.shape[0])
    first_ptr = first.ptr or 0
    second_ptr = second.ptr or 0
    itemsize = 8
    if (
        first.dtype == wp.float64
        and second.dtype == wp.float64
        and int(second.shape[0]) == n
        and first.device == second.device
        and first.is_contiguous
        and second.is_contiguous
        and abs(second_ptr - first_ptr) == n * itemsize
    ):
        low = min(first_ptr, second_ptr)
        both = wp.array(ptr=low, shape=(2 * n,), dtype=wp.float64, device=first.device).numpy()
        if first_ptr == low:
            return both[:n], both[n:]
        return both[n:], both[:n]
    return first.numpy(), second.numpy()


class _AdaptiveCg:
    """
    ``preconditioner="adaptive"``: Jacobi under a probe, escalating to the Chebyshev polynomial.

    Runs up to [`CG_CHEBYSHEV_PROBE_ITERATIONS`][ordito.linalg.CG_CHEBYSHEV_PROBE_ITERATIONS]
    Jacobi rounds; a solve that converges there never pays for the polynomial, and one that does
    not continues under ``"chebyshev"`` from the iterate the probe left, with the rest of the
    budget. Callable like ``_BatchedCg``, returning the last solve's three values; as host scalars
    an escalated count includes the probe's rounds, so it is comparable with the caller's cap and
    a non-convergence warning can fire. Device arrays are returned as the escalated solve left
    them.

    With ``cached`` the two states are ``_cached_solver``'s, kept per operator; without, they are
    bound to the caller's ``rhs`` / ``solution`` for a hoisted state to drive repeatedly.
    """

    def __init__(
        self,
        matrix: odt.BsrMatrix[wp.float64],
        rhs: odt.Array2dFloat,
        solution: odt.Array2dFloat,
        *,
        tol: float,
        maxiter: int,
        check_every: int,
        cached: bool,
    ) -> None:
        self._matrix = matrix
        self._rhs = rhs
        self._solution = solution
        self._tol = tol
        self._maxiter = maxiter
        self._check_every = check_every
        self._cached = cached
        self._probe_cap = min(maxiter, CG_CHEBYSHEV_PROBE_ITERATIONS)
        self._probe = self._state("diag", self._probe_cap)
        # Built on the first escalation: a solve that never escalates never pays the polynomial's
        # setup, which reads a bound back.
        self._escalation: _BatchedCg | None = None

    def _state(self, preconditioner: str, maxiter: int) -> _BatchedCg:
        if self._cached:
            return _cached_solver(
                self._matrix,
                int(self._solution.shape[0]),
                tol=self._tol,
                maxiter=maxiter,
                check_every=self._check_every,
                preconditioner=preconditioner,
            )
        return _BatchedCg(
            self._matrix,
            self._rhs,
            self._solution,
            tol=self._tol,
            maxiter=maxiter,
            check_every=self._check_every,
            preconditioner=preconditioner,
        )

    def _run(self, state: _BatchedCg):
        if not self._cached:
            return state()
        return state.solve(self._rhs.flatten(), self._solution.flatten())

    def __call__(self):
        """Run the probe, and the escalation if the probe did not converge."""
        result = self._run(self._probe)
        if self._probe_cap >= self._maxiter:
            return result
        # One readback: the probe's residual against its tolerance. It is the price of deciding,
        # and a solve short enough to finish inside the probe pays nothing else.
        residual, tolerance = _cg_residual_and_tolerance(result)
        if residual <= tolerance:
            return result
        if self._escalation is None:
            self._escalation = self._state("chebyshev", self._maxiter - self._probe_cap)
        iterations, residual, tolerance = self._run(self._escalation)
        if isinstance(iterations, wp.array):
            return iterations, residual, tolerance
        return int(iterations) + self._probe_cap, residual, tolerance


class _BatchedCg:
    """

    Preconditioned conjugate gradient over the columns of one operator.

    The same iterates as ``warp.optim.linear.cg`` in exact arithmetic and the same stopping rule --
    every column runs until *its own* residual is under ``max(atol, tol * ||b_c||)`` -- so it is a
    drop-in for it, down to the ``(iterations, residual, tolerance)`` return contract. Callable
    like ``warp.optim.linear``'s solver state: the buffers are allocated once and every call
    re-reads ``rhs`` and warm-starts from whatever ``solution`` currently holds.


    Notes
    -----
    It is here because of what the batching costs inside Warp's solver rather than because of the
    arithmetic. ``replicated_operator`` attaches ``batch_offsets`` to get the per-column stopping
    rule, and that is precisely the input for which Warp's ``TiledDot`` selects a per-column
    reduction path whose cost grows with the vector length, where the reduction it uses for an
    unbatched vector stays flat -- and a CG iteration runs two of these dot products. Reducing per
    column with a real two-stage tree keeps the batching and drops that cost back down.

    A round is two launches, not six. It is the Chronopoulos-Gear form (1989), which recovers
    ``alpha`` from ``r.u`` and ``w.u`` with ``w = A u`` and carries ``s = A p`` by recurrence, so
    the mat-vec and all three of the round's dots are one launch and the update another -- where
    standard CG needs ``p.Ap`` before it can update ``r``, and ``r.z`` after, two reductions with a
    dependency between them. The Jacobi apply is an elementwise multiply of the ``r`` the update
    has just written, so it rides in that launch too. And the reductions' second stage needs no
    launch of its own: every block of the consumer folds its column's partials itself, through one
    fixed tree, so every block agrees (``CG_FOLD_MAX_BLOCKS`` bounds the column length where that
    pays). The conditional-graph test is then the largest single cost of a round; see
    ``CG_CHECK_EVERY``'s note for why it is not batched.

    **What was tried and is not here.** Solving the columns as separate unbatched ``cg`` calls also
    reaches the tree reduction, but it pays a second copy of every other kernel in the iteration and
    loses. Dropping ``batch_offsets`` to get the tree from a single call is not a tuning change at
    all -- ``alpha`` and ``beta`` would then be global rather than per column, which is CG on the
    block system and a different iteration. And **sharing the Krylov subspace across columns was
    tried and removed**: this class batches the *launches* but runs each column's iteration
    mathematically independently, so the counts are unchanged versus separate solves, where a
    classical block conjugate gradient (O'Leary 1980) over two columns reduces them on a uniform
    mesh at two more launches per iteration -- a wash there, and worse on a *graded* mesh, where the
    two search directions go nearly parallel and the count goes the other way.
    """

    def __init__(
        self,
        matrix: odt.CsrMatrix,
        rhs: odt.Array2dFloat | None,
        solution: odt.Array2dFloat,
        *,
        tol: float,
        maxiter: int,
        check_every: int,
        preconditioner: str | SquaredLaplacianPreconditioner = "diag",
        settle: tuple[int, float, int] | None = None,
        narrow_values: bool = False,
        own_storage: bool = False,
    ) -> None:
        device = matrix.device
        self._device = device
        # A pooled state (``_cached_solver``) holds a copy of its operator's storage rather than the
        # caller's, so ``refresh`` can write the next operator of the same shape into the same
        # arrays the recorded graph reads.
        if own_storage:
            matrix = _owned_copy(matrix)
            if isinstance(preconditioner, SquaredLaplacianPreconditioner):
                preconditioner = preconditioner._owned_copy()  # pyright: ignore[reportPrivateUsage]
        self._owns_storage = own_storage
        self._matrix = matrix
        self._n_columns, self._n = int(solution.shape[0]), int(solution.shape[1])
        self._tol = float(tol)
        self._maxiter = int(maxiter)
        self._check_every = int(check_every)

        # Every vector the iteration reduces is one this class owns, so the column pitch is ours to
        # choose: pad it to a whole number of tiles and zero the gap, and the dot has no ragged
        # block at all. See ``cg_norm_partials`` for what the ragged block cost when it existed.
        # ``span`` entries a block, ``blocks`` blocks a column, and whether the update folds the
        # dots' second stage itself rather than ``cg_coefficients`` doing it once: two launches a
        # round under Jacobi instead of three. See ``CG_FOLD_MAX_BLOCKS`` and ``cg_layout``.
        self._span, self._blocks, self._fold = kernel_cg.cg_layout(self._n, CG_FOLD_MAX_BLOCKS)
        # Whether a round forms ``A u`` with ``bsr_mv`` (see ``CG_HEAVY_ROW_ENTRIES``), decided on
        # the true entry count -- ``nnz`` is a capacity -- and only for a column too long to fold:
        # below that a round is bound by its launches, and one ``bsr_mv`` a column plus a dots
        # launch lose to the one fused launch (measured 0.91-0.94x on ``smooth_region``'s
        # three-column, twenty-entry-a-row systems). One four-byte read per state.
        self._heavy, self._mv_tile = _row_path(matrix, self._n, self._fold)
        self._stride = self._blocks * self._span
        self._dofs = self._n_columns * self._stride
        # The values the fused mat-vec reads: the operator's own, or -- ``narrow_values``, taken by
        # the heat diffusions (``solve_spd_settled``) -- a ``float32`` copy of a ``float64``
        # operator's, widened as each is read (``multigrid.csr_row_dot``). Past a few tens of
        # thousands of rows a round is bound by its bytes, and the values are the largest stream;
        # the operator it then solves differs by one rounding per entry, which moved the heat
        # method's distance by 2e-8 of its range and its transported directions by 1e-4 degrees.
        # A copy, so values rewritten in place after construction are not seen: the settle solves
        # never rewrite theirs. The ``bsr_mv`` path of a heavy-row column and every
        # preconditioner read the operator as stored.
        self._round_values = matrix.values
        if narrow_values and matrix.values.dtype == wp.float64 and not self._heavy:
            self._round_values = _launch.empty(matrix.values.shape, dtype=wp.float32, device=device)
            od.array.copyto(self._round_values, matrix.values)

        self._rhs = rhs
        self._solution = solution
        self._solution_flat = solution.flatten()
        # ``wp.zeros`` rather than ``wp.empty``: the pad between each column's ``n`` entries and
        # ``stride`` must read as zero, and it stays zero because every kernel writing there writes
        # a multiple of it.
        # The vectors are stored at the operator's precision and every reduction and scalar is
        # ``float64`` whatever it is (``kernels/algorithms/conjugate_gradient.cg_update``).
        dtype = matrix.values.dtype
        if dtype != wp.float64 and preconditioner != "diag":
            raise ValueError(
                f"only the Jacobi preconditioner solves a {dtype.__name__} system; got "
                f"{preconditioner!r}"
            )
        self._dtype = dtype
        # ``r``, ``u = M^-1 r``, ``w = A u``, the search direction ``p`` and ``s = A p``, in one
        # allocation: a view is a third of a ``wp.zeros``.
        dofs = self._dofs
        vectors = _launch.zeros(5 * dofs, dtype=dtype, device=device)
        self._r, self._u, self._w, self._p, self._s = (
            odt.as_dense(vectors[k * dofs : (k + 1) * dofs]) for k in range(5)
        )
        # First-stage partials of the three dots ``cg_matvec_dots`` reduces: ``r.u``, ``w.u``,
        # ``r.r`` (and of ``||b||^2`` in row 0 at the start of a solve). ``wp.empty``: every entry a
        # fold reads, ``[0, blocks)`` of each row, is written by the partials launch before it.
        self._partials = _launch.empty(
            (3, self._n_columns, self._blocks), dtype=wp.float64, device=device
        )
        # Every per-column ``float64`` scalar in one ``(10, n_columns)`` allocation, rows viewed
        # once (re-taking a slice costs a few microseconds of ``wp.array.__getitem__`` each time):
        #
        # * row 0, each column's squared threshold (``cg_seed``);
        # * rows 1-2, ``(r.r, r.u)`` as of the last round, published by ``cg_update`` -- row 1 is
        #   the residual the solve returns, so rows 0-1 are the whole host result in one readback;
        # * rows 3-5, ``(alpha, beta, stepping)``, written by ``cg_coefficients`` on the unfolded
        #   path only;
        # * rows 6-9, the recurrence's scalars, double-buffered: ``cg_update`` reads the ``old``
        #   rows in every block and writes the ``new`` ones, and ``cg_matvec_dots`` carries them.
        #
        # The loop's integer state leads the same allocation, two ``float64`` words holding the
        # ``int32`` words (``_INT_WORDS``), so the loop condition, the count and rows 0-1 are one
        # contiguous span a host result reads in one readback (``host_result``).
        head = _launch.zeros(_INT_WORDS + 10 * self._n_columns, dtype=wp.float64, device=device)
        self._head = odt.as_dense(head[: _INT_WORDS + 2 * self._n_columns])
        scalars = odt.as_dense(head[_INT_WORDS:]).reshape((10, self._n_columns))
        self._residual_tolerance = odt.as_array2d(odt.as_dense(scalars[0:2]), wp.float64)
        self._atol_sq = odt.as_dense(scalars[0])
        self._dots = odt.as_array2d(odt.as_dense(scalars[1:3]), wp.float64)
        self._dots_rz = odt.as_dense(scalars[1])
        self._coefficients = odt.as_array2d(odt.as_dense(scalars[3:6]), wp.float64)
        self._gamma_old, self._alpha_old, self._gamma_new, self._alpha_new = (
            odt.as_dense(scalars[row]) for row in range(6, 10)
        )
        # The round-loop state (``kernels/array.py``'s ``LOOP_ROUND`` / ``LOOP_CONDITION``, seeded
        # by ``cg_seed`` at the start of every solve) and, after it, the rounds that took a step --
        # the iteration count a solve reports -- so the host cadence's condition read carries it.
        self._ints = wp.array(
            ptr=head.ptr, dtype=wp.int32, shape=(kernel_array.LOOP_STATE_SIZE + 1,), device=device
        )
        # A view by pointer holds no reference to its storage; this one keeps ``head`` alive for as
        # long as it -- or any slice of it handed to a caller -- is.
        self._ints._ref = head  # pyright: ignore[reportPrivateUsage]
        self._state = odt.as_dense(self._ints[: kernel_array.LOOP_STATE_SIZE])
        self.iterations = odt.as_dense(self._ints[kernel_array.LOOP_STATE_SIZE :])
        # The live rows of ``u`` and ``w`` per column, for ``bsr_mv`` on the heavy-row path.
        self._uw_columns = [
            (
                odt.as_dense(self._u[c * self._stride : c * self._stride + self._n]),
                odt.as_dense(self._w[c * self._stride : c * self._stride + self._n]),
            )
            for c in range(self._n_columns if self._heavy else 0)
        ]

        # A multigrid V-cycle cannot be fused into a register the way the Jacobi apply is, so it
        # runs as its own launches over the same padded vectors and the x/r update drops its ``z``
        # write -- one extra launch per iteration, which is the whole cost of un-fusing. The
        # hierarchy is batched over the columns at *this* solver's pitch, so the cycle's elementwise
        # passes stay one launch each rather than one per column.
        self._cycle = None
        if isinstance(preconditioner, SquaredLaplacianPreconditioner):
            self._cycle = preconditioner.bind(self._n_columns, self._stride)
        elif preconditioner == "chebyshev":
            # Only ``float64`` systems get a non-Jacobi one (``_batched_preconditioner_kind``).
            assert odt.has_blocks(matrix, wp.float64)
            self._cycle = _JacobiChebyshev(matrix).bind(self._n_columns, self._stride)
        elif preconditioner == "multigrid":
            assert odt.has_blocks(matrix, wp.float64)
            hierarchy = _multigrid_hierarchy(matrix, 0)
            if hierarchy is not None:
                self._cycle = _MultigridCycle(
                    *hierarchy, n_columns=self._n_columns, stride=self._stride
                )
        # The Jacobi inverse diagonal, which the update applies in its own register (see
        # ``cg_update``): into ``u`` under Jacobi, and into the Jacobi-Chebyshev polynomial's input
        # under that polynomial, which then reuses the diagonal it already holds. A V-cycle or the
        # squared-Laplacian polynomial owns its scaling, and none is built.
        self._inv_diag = None
        self._scaled = self._u
        if self._cycle is None:
            self._inv_diag = _launch.empty(self._n, dtype=dtype, device=device)
            _launch.launch(
                kernel_linalg.JACOBI_INVERSE_DIAGONAL[dtype],
                dim=self._n,
                inputs=[matrix.offsets, matrix.columns, matrix.values],
                outputs=[self._inv_diag],
                device=device,
            )
        elif isinstance(self._cycle, _JacobiChebyshevApply):
            self._inv_diag = self._cycle.inverse_diagonal
            self._scaled = self._cycle.scaled
        # The caller's right-hand side at pitch ``n``, re-read by ``_initialize`` on every call. A
        # solver built without one (``_cached_solver``'s) is handed it by ``solve``.
        self._rhs_flat: odt.ArrayNdFloat | None = None if rhs is None else rhs.flatten()
        # The device-side loop's recorded graph: ``_seed`` -- which resets the loop condition --
        # then the loop. Every launch in both reads buffers this state owns and never rebinds, so
        # one recording serves every call: re-recording it per call would repay the capture of
        # every launch in the body, plus the graph instantiation, for an identical graph. Only
        # ``_initialize``, which reads the caller's operands, is issued per call.
        self._graph = None

        # The settle monitor ``solve_spd_settled`` runs (``check_rounds``, ``change_tolerance``,
        # ``settle_rounds``): every ``check_rounds`` rounds ``cg_settle_change`` compares the
        # iterate with the one ``check_rounds`` earlier and ``cg_settle_decide`` may clear the
        # loop condition -- two launches a check, inside the same recorded loop, so a solve that
        # stops on it never reads back. ``_settle`` is reset by one memset per solve; ``previous``
        # needs none, since the first check never stops (``SETTLE_PREVIOUS``).
        self._settle = settle
        if settle is not None:
            self._settle_previous = _launch.zeros(
                self._n_columns * self._n, dtype=dtype, device=device
            )
            self._settle_state = _launch.zeros(
                kernel_cg.SETTLE_STATE_SIZE, dtype=wp.float64, device=device
            )

    def refresh(
        self, matrix: odt.CsrMatrix, preconditioner: str | SquaredLaplacianPreconditioner = "diag"
    ) -> None:
        """
        Write ``matrix`` into this pooled state's own operator storage, and re-derive from it.

        One launch (``kernels/linalg.refresh_pooled_operator``) copies the pattern and the values
        and rewrites what construction derived from them -- the narrowed values, the Jacobi
        inverse diagonal -- and, under the Jacobi-Chebyshev polynomial, the Gershgorin ratios its
        interval is refitted to on the device. A squared-Laplacian ``preconditioner`` is the
        caller's own, so its arrays are copied into the state's copy of it. The recorded graph
        reads these same arrays, so the state then solves ``matrix`` with no new recording. Only for
        a state built with ``own_storage``, whose operator and preconditioner have these shapes.
        """
        assert self._owns_storage
        owned = self._matrix
        chebyshev = isinstance(self._cycle, _JacobiChebyshevApply)
        narrowed = None if self._round_values is owned.values else self._round_values
        ratios = None
        if chebyshev:
            assert isinstance(self._cycle, _JacobiChebyshevApply)
            ratios = self._cycle.owner.ratios
        if self._n > 0:
            _launch.launch(
                kernel_linalg.refresh_pooled_operator,
                dim=self._n,
                inputs=[
                    matrix.offsets,
                    matrix.columns,
                    matrix.values,
                    wp.int32(0 if narrowed is None else 1),
                    wp.int32(2 if chebyshev else (0 if self._inv_diag is None else 1)),
                ],
                outputs=[
                    owned.offsets,
                    owned.columns,
                    owned.values,
                    narrowed,
                    self._inv_diag,
                    ratios,
                ],
                device=self._device,
            )
        if chebyshev:
            assert isinstance(self._cycle, _JacobiChebyshevApply)
            self._cycle.owner.refit()
        if isinstance(preconditioner, SquaredLaplacianPreconditioner):
            assert isinstance(self._cycle, _SquaredLaplacianApply)
            self._cycle.owner._assign(preconditioner)  # pyright: ignore[reportPrivateUsage]

    def _settle_check(self) -> None:
        """Issue the settle monitor's two launches, after a block of ``check_rounds`` rounds."""
        assert self._settle is not None
        check_rounds, change_tolerance, settle_rounds = self._settle
        n = self._n_columns * self._n
        _launch.launch_tiled(
            kernel_cg.cg_settle_change,
            dim=[kernel_reduce.blocks_1d(n)],
            inputs=[self._solution_flat, self._settle_previous],
            outputs=[self._settle_state],
            block_dim=TILE_1D,
            device=self._device,
        )
        _launch.launch(
            kernel_cg.cg_settle_decide,
            dim=1,
            inputs=[
                wp.int32(check_rounds),
                wp.float64(change_tolerance),
                wp.int32(settle_rounds),
                self._settle_state,
                self._state,
            ],
            device=self._device,
        )

    def _settle_block(self) -> None:
        """Run the recorded loop body under a settle monitor: ``check_rounds`` rounds, a check."""
        assert self._settle is not None
        for _ in range(self._settle[0]):
            self._iteration()
        self._settle_check()

    def _iteration(self) -> None:
        """
        One Chronopoulos-Gear round, none of which reads back to the host.

        Two launches under Jacobi and ``fold``: the mat-vec with its three dots, then the update
        with the preconditioner apply fused in. A preconditioner with launches of its own adds
        them after the update, and a column long enough to lose ``fold`` adds a finalize between
        the two. See ``kernels/algorithms/conjugate_gradient.py``.
        """
        tile = int(kernel_cg.CG_TILE)
        grid = (self._n_columns, self._blocks)
        if self._heavy:
            for u_column, w_column in self._uw_columns:
                odt.bsr_mv(self._matrix, u_column, w_column, tile_size=self._mv_tile)
            _launch.launch_tiled(
                kernel_cg.CG_ROUND_DOTS[self._dtype],
                dim=grid,
                inputs=[
                    wp.int32(self._stride),
                    wp.int32(self._span),
                    self._r,
                    self._u,
                    self._w,
                    self._gamma_new,
                    self._alpha_new,
                ],
                outputs=[self._partials, self._gamma_old, self._alpha_old, self._state],
                block_dim=tile,
                device=self._device,
            )
        else:
            _launch.launch_tiled(
                kernel_cg.CG_MATVEC_DOTS[self._dtype, self._round_values.dtype],
                dim=grid,
                inputs=[
                    wp.int32(self._n),
                    wp.int32(self._stride),
                    wp.int32(self._span),
                    self._matrix.offsets,
                    self._matrix.columns,
                    self._round_values,
                    self._r,
                    self._u,
                    self._gamma_new,
                    self._alpha_new,
                ],
                outputs=[self._w, self._partials, self._gamma_old, self._alpha_old, self._state],
                block_dim=tile,
                device=self._device,
            )
        if not self._fold:
            _launch.launch_tiled(
                kernel_cg.cg_coefficients,
                dim=(self._n_columns,),
                inputs=[
                    wp.int32(self._n_columns),
                    wp.int32(self._maxiter),
                    wp.int32(self._blocks),
                    self._partials,
                    self._gamma_old,
                    self._alpha_old,
                    self._atol_sq,
                    self._state,
                ],
                outputs=[
                    self._coefficients,
                    self._gamma_new,
                    self._alpha_new,
                    self._dots,
                    self.iterations,
                ],
                block_dim=tile,
                device=self._device,
            )
        jacobi = self._inv_diag is not None
        # One tile a block whatever ``span`` is: the update reduces nothing unless it folds (and it
        # folds only at one tile a block), so it is a pure stream, which wants the most blocks.
        _launch.launch_tiled(
            kernel_cg.CG_UPDATE[self._dtype],
            dim=(self._n_columns, self._stride // tile),
            inputs=[
                wp.int32(self._stride),
                wp.int32(tile),
                wp.int32(self._n),
                wp.int32(self._n_columns),
                wp.int32(self._maxiter),
                wp.int32(1 if self._fold else 0),
                wp.int32(self._blocks),
                wp.int32(1 if jacobi else 0),
                self._partials,
                self._coefficients,
                self._gamma_old,
                self._alpha_old,
                self._atol_sq,
                self._inv_diag,
                self._w,
                self._p,
                self._s,
                self._r,
                self._u,
                self._state,
            ],
            outputs=[
                self._solution_flat,
                self._scaled,
                self._gamma_new,
                self._alpha_new,
                self._dots,
                self.iterations,
            ],
            block_dim=tile,
            device=self._device,
        )
        if isinstance(self._cycle, _JacobiChebyshevApply):
            self._cycle.apply_scaled(self._u)
        elif self._cycle is not None:
            self._cycle.apply(self._r, self._u)

    def _initialize(self, initial: odt.ArrayNdFloat | None) -> None:
        """
        Seed the residual from the caller's operands: the one launch that reads them.

        ``initial``, when given, is the caller's initial guess, which the launch copies into this
        state's own solution buffer as it reads it (``solve``); otherwise that buffer already holds
        it. Whenever the state holds a Jacobi inverse diagonal the initial scaling rides in the same
        launch, into ``_scaled`` -- ``u`` itself under Jacobi, the polynomial's input under
        Jacobi-Chebyshev -- exactly as ``cg_update`` does it every round.
        """
        tile = int(kernel_cg.CG_TILE)
        jacobi = self._inv_diag is not None
        _launch.launch_tiled(
            kernel_cg.CG_INITIAL[self._dtype, self._round_values.dtype],
            dim=(self._n_columns, self._blocks),
            inputs=[
                wp.int32(self._n),
                wp.int32(self._stride),
                wp.int32(self._span),
                wp.int32(1 if jacobi else 0),
                wp.int32(0 if initial is None else 1),
                self._matrix.offsets,
                self._matrix.columns,
                self._round_values,
                self._rhs_flat,
                self._solution_flat if initial is None else initial,
                self._inv_diag,
            ],
            outputs=[
                None if initial is None else self._solution_flat,
                self._r,
                self._scaled,
                self._p,
                self._s,
                self._partials,
            ],
            block_dim=tile,
            device=self._device,
        )

    def _seed(self) -> None:
        """
        Everything a solve does after the residual and before its first round.

        The tolerances and the recurrence's seeds, the preconditioner's first apply and the settle
        monitor's reset. None of it reads the caller's operands, so on the device-side path it is
        recorded into the replayed graph ahead of the loop rather than issued on every call -- under
        the Jacobi-Chebyshev polynomial that is a dozen launches a solve.
        """
        tile = int(kernel_cg.CG_TILE)
        _launch.launch_tiled(
            kernel_cg.cg_seed,
            dim=(self._n_columns,),
            inputs=[
                wp.float64(self._tol * self._tol),
                wp.float64(0.0),
                wp.int32(self._blocks),
                self._partials,
            ],
            outputs=[self._atol_sq, self._gamma_new, self._alpha_new, self.iterations, self._state],
            block_dim=tile,
            device=self._device,
        )
        if isinstance(self._cycle, _JacobiChebyshevApply):
            self._cycle.apply_scaled(self._u)
        elif self._cycle is not None:
            self._cycle.apply(self._r, self._u)
        if self._settle is not None:
            _launch.zero_(self._settle_state)

    def __call__(self, initial: odt.ArrayNdFloat | None = None):
        """
        Run the solve, returning ``warp.optim.linear.cg``'s three values on its own terms.

        Device arrays under ``check_every == 0``, host scalars otherwise, exactly as
        [`solve_spd_columns`][ordito.linalg.solve_spd_columns] documents. ``initial`` is
        ``_initialize``'s.
        """
        self._initialize(initial)
        # The device-side loop needs a conditional CUDA graph, so a CPU-resident system takes the
        # host cadence even where the machine has a GPU -- ``_supported_check_every`` can only see
        # the *machine*, not which device these arrays are on.
        check_every = self._check_every
        if check_every == 0 and not self._device.is_cuda:
            check_every = CG_CHECK_EVERY_FALLBACK
        if check_every > 0:
            self._seed()
            return self._run_with_host_checks(check_every)
        if self._graph is None:
            condition = self._state[kernel_array.LOOP_CONDITION_VIEW]
            # One iteration per conditional-graph test, not a batched run of them: see the note
            # above ``CG_CHECK_EVERY`` for the sweep that removed the batching. A settle monitor is
            # the exception: its check needs a block of rounds between tests anyway, and a round
            # after the loop's own stop takes no step (``cg_step_scalars``), so the block costs at
            # most ``check_rounds - 1`` idle rounds at the very end.
            body = self._iteration if self._settle is None else self._settle_block
            with wp.ScopedCapture(self._device) as capture:
                self._seed()
                wp.capture_while(cast("wp.array[int]", condition), body)
            self._graph = capture.graph
        assert self._graph is not None
        wp.capture_launch(self._graph)
        return self.iterations, self._dots_rz, self._atol_sq

    def solve(self, rhs: odt.ArrayNdFloat, solution: odt.ArrayNdFloat):
        """
        Run against a caller's operands through this solver's own solution buffer.

        The recorded graph writes ``x`` through a pointer taken at record time, so a solver that
        outlives one call cannot hand the graph the caller's buffer; the initial guess is copied in
        by the residual's own launch and the answer copied out, against a recording. ``rhs`` is
        read only by ``_initialize``, which runs outside the graph, so it is taken as given. Both
        are the flat ``n_columns * n`` views of the caller's contiguous buffers.
        """
        self._rhs_flat = rhs
        try:
            result = self(solution)
        finally:
            # Not held past the call: the caller's right-hand side is not this solver's to keep.
            self._rhs_flat = None
        _launch.copy(solution, self._solution_flat)
        return result

    def _run_with_host_checks(self, check_every: int) -> tuple[int, float, float]:
        """
        Drive the loop from the host: issue a block of rounds, then read the loop condition.

        The condition is the device-side loop's own, so this stops exactly where that loop would,
        at most ``check_every - 1`` idle rounds later -- a round after the one that stopped the
        loop takes no step, because every column is then inside its tolerance or past the cap.
        The block is trimmed against ``maxiter``, so no more than ``maxiter`` rounds are issued.

        Returns ``warp.optim.linear.cg``'s three host scalars. The count is the rounds that took a
        step, and the residual is the one the last round measured.
        """
        if self._settle is not None:
            # The monitor's check is the cadence: it is what can stop the loop early.
            check_every = self._settle[0]
        done = 0
        head = None
        while done < self._maxiter:
            block = min(check_every, self._maxiter - done)
            for _ in range(block):
                self._iteration()
            done += block
            if self._settle is not None:
                self._settle_check()
            # The condition, the iteration count and the result rows share one span, so the last
            # check's read is also the result's.
            head = self._head.numpy()
            if not int(_head_ints(head)[int(kernel_array.LOOP_CONDITION)]):
                break
        return self.host_result(head)

    def host_result(self, head: npt.NDArray[np.float64] | None = None) -> tuple[int, float, float]:
        """
        Read the last solve's ``(iterations, residual, tolerance)`` back as host scalars.

        One readback of the span holding the integer state and the two scalar rows, where reading
        the three returned device arrays one at a time takes three; ``head`` is a read of that span
        the caller already holds, which saves it.
        """
        if head is None:
            head = self._head.numpy()
        rows = np.asarray(head[_INT_WORDS:]).reshape((2, self._n_columns))
        return (
            int(_head_ints(head)[kernel_array.LOOP_STATE_SIZE]),
            math.sqrt(float(rows[1].max())),
            math.sqrt(float(rows[0].max())),
        )


# ``float64`` words at the head of a ``_BatchedCg``'s scalar buffer that hold its ``int32`` loop
# state and iteration count.
_INT_WORDS = (kernel_array.LOOP_STATE_SIZE + 2) // 2


def _head_ints(head: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """Reinterpret the head of a host read of a ``_BatchedCg``'s scalars as its ``int32`` words."""
    return np.ascontiguousarray(head[:_INT_WORDS]).view(np.int32)


# ``_cached_solver``'s states, keyed weakly by the operator they solve, so a state -- its buffers,
# its preconditioner and its recorded graph -- lives exactly as long as its operator does.
_SOLVER_CACHE: weakref.WeakKeyDictionary[odt.SparseMatrix, dict[tuple[object, ...], _BatchedCg]] = (
    weakref.WeakKeyDictionary()
)


def _cached_solver(
    matrix: odt.CsrMatrix,
    n_columns: int,
    *,
    tol: float,
    maxiter: int,
    check_every: int,
    preconditioner: str | SquaredLaplacianPreconditioner,
    settle: tuple[int, float, int] | None = None,
    narrow_values: bool = False,
    pooled: bool = False,
) -> _BatchedCg:
    """
    Return the ``_BatchedCg`` state for this operator and configuration, built on first use.

    A state records its device-side loop once and replays it on every later call, so keeping one
    per operator removes the recording from every solve after the first against that operator: a
    hoisted ``heat_operators`` bundle, an ``arap`` global step, any caller that solves one system
    for several right-hand sides. The state owns its solution buffer (``_BatchedCg.solve``) and
    reads the right-hand side outside the graph, so any caller's operands are valid.

    The state holds the operator's *storage* through a second ``BsrMatrix`` sharing its arrays
    rather than the operator itself, which would keep its own weak key alive; holding the arrays
    also keeps every pointer the graph recorded valid. The key carries the arrays' identities, so
    a matrix whose storage is replaced gets a new state. **Values rewritten in place are not
    detected**, and need not be for correctness: the mat-vec reads them live, so the solve still
    converges to the new system's answer, and only the preconditioner -- derived from the values at
    construction -- goes stale, which changes the rate and not the fixed point.

    **A fresh operator of a shape already seen takes a pooled state instead** (``pooled``, for a
    caller that runs the state at once and keeps nothing of it). A recorded graph holds pointers
    into its operator's storage and nothing rebinds them, so every fresh operator recorded anew --
    most of a solve's cost on small and medium systems whenever the operators are rebuilt per call.
    A pooled state owns a copy of its operator's storage, and ``_BatchedCg.refresh`` writes the new
    operator into it in one launch, so the graph replays. It is refreshed on every use, whether the
    operator changed or not, which is what keeps it right under in-place rewrites and under two
    live operators of one shape alternating. The first operator of a shape keeps a state of its
    own as above, so an operator solved once, or a single hoisted one, pays no copy.

    A ``SquaredLaplacianPreconditioner`` is the caller's own, built fresh against each fresh system,
    so a state keyed by the operator would never be hit again: its solve takes a pooled state, whose
    own copy of the preconditioner ``refresh`` rewrites with the operator, or else a state of its
    own that is not kept.

    Two solves against one state must run on one stream, as every caller here does.
    """
    squared = isinstance(preconditioner, SquaredLaplacianPreconditioner)
    config = (
        n_columns,
        float(tol),
        int(maxiter),
        int(check_every),
        preconditioner._pool_shape() if squared else preconditioner,  # pyright: ignore[reportPrivateUsage]
        settle,
        narrow_values,
    )
    key = (*config, id(matrix.offsets), id(matrix.columns), id(matrix.values))
    entries = {} if squared else _SOLVER_CACHE.setdefault(matrix, {})
    state = entries.pop(key, None)
    # Pooled only where ``_BatchedCg.refresh`` can rebuild everything the state derives from
    # ``matrix``.
    if (
        state is None
        and pooled
        and matrix.values.dtype == wp.float64
        and (
            isinstance(preconditioner, SquaredLaplacianPreconditioner)
            or preconditioner in ("diag", "chebyshev")
        )
        and int(matrix.nrow) > 0
    ):
        n = int(matrix.nrow)
        fold = kernel_cg.cg_layout(n, CG_FOLD_MAX_BLOCKS)[2]
        shape = (
            *config,
            n,
            matrix.offsets.size,
            matrix.columns.size,
            matrix.values.size,
            _row_path(matrix, n, fold),
            str(matrix.device),
        )
        if shape in _SHAPES_SEEN:
            pooled_state = _SOLVER_POOL.pop(shape, None)
            if pooled_state is None:
                solution = odt.empty_2d((n_columns, n), wp.float64, device=matrix.device)
                pooled_state = _BatchedCg(
                    matrix,
                    None,
                    solution,
                    tol=tol,
                    maxiter=maxiter,
                    check_every=check_every,
                    preconditioner=preconditioner,
                    settle=settle,
                    narrow_values=narrow_values,
                    own_storage=True,
                )
            else:
                pooled_state.refresh(matrix, preconditioner)
            while len(_SOLVER_POOL) >= _SOLVER_POOL_ENTRIES:
                _SOLVER_POOL.pop(next(iter(_SOLVER_POOL)))
            _SOLVER_POOL[shape] = pooled_state
            return pooled_state
        if len(_SHAPES_SEEN) >= _SHAPES_SEEN_ENTRIES:
            _SHAPES_SEEN.clear()
        _SHAPES_SEEN.add(shape)
    if state is None:
        n = int(matrix.nrow)
        solution = odt.empty_2d((n_columns, n), matrix.values.dtype, device=matrix.device)
        state = _BatchedCg(
            _storage_alias(matrix),
            None,
            solution,
            tol=tol,
            maxiter=maxiter,
            check_every=check_every,
            preconditioner=preconditioner,
            settle=settle,
            narrow_values=narrow_values,
        )
        while len(entries) >= _SOLVER_CACHE_ENTRIES:
            entries.pop(next(iter(entries)))
    # Re-inserted last, so the eviction above drops the least recently used. A squared-Laplacian
    # state's ``entries`` is a throwaway, so it is never kept.
    entries[key] = state
    return state


# ``_cached_solver``'s pooled states, one per operator shape and configuration, least recently
# used first, and the shapes it has seen once (``_SHAPES_SEEN``), which a second fresh operator of
# the same shape then pools on. Both bounded: a pooled state holds a copy of its operator.
_SOLVER_POOL: dict[tuple[object, ...], _BatchedCg] = {}
_SOLVER_POOL_ENTRIES = 8
_SHAPES_SEEN: set[tuple[object, ...]] = set()
_SHAPES_SEEN_ENTRIES = 256


def _row_path(matrix: odt.SparseMatrix, n: int, fold: bool) -> tuple[bool, int]:
    """
    ``_BatchedCg``'s mat-vec path for ``matrix``: whether it takes ``bsr_mv``, and at what tile.

    ``bsr_mv`` (see ``CG_HEAVY_ROW_ENTRIES``) is decided on the true entry count -- ``nnz`` is a
    capacity -- and only for a column too long to fold: below that a round is bound by its
    launches, and one ``bsr_mv`` a column plus a dots launch lose to the one fused launch (measured
    0.91-0.94x on ``smooth_region``'s three-column, twenty-entry-a-row systems). One four-byte
    read for a column that does not fold.

    The block-per-row kernel is a ``launch_tiled`` kernel whose lanes split a row, and the CPU
    device runs one lane per block (CLAUDE.md section 2.2), so there it would form ``A u`` from
    one lane's share and the solve would stop at zero iterations; the CPU keeps the lane-per-row
    kernel, as ``bsr_mv``'s own heuristic does.
    """
    entries = read_scalar(matrix.offsets, n) if n > 0 and not fold else 0
    heavy = entries > CG_HEAVY_ROW_ENTRIES * n
    tiled = matrix.device.is_cuda and entries > _HEAVY_ROW_TILED_ENTRIES * n
    return heavy, _HEAVY_ROW_TILE if tiled else -1


@overload
def _owned_copy(matrix: odt.BsrMatrix[wp.float32]) -> odt.BsrMatrix[wp.float32]: ...
@overload
def _owned_copy(matrix: odt.BsrMatrix[wp.float64]) -> odt.BsrMatrix[wp.float64]: ...
def _owned_copy(matrix: odt.BsrMatrix[odt.Block]) -> odt.BsrMatrix[odt.Block]:
    """Build a ``BsrMatrix`` over copies of ``matrix``'s arrays, for a pooled state to own."""
    return _bsr_over(
        int(matrix.nrow),
        int(matrix.ncol),
        _launch.clone(matrix.offsets),
        _launch.clone(matrix.columns),
        _launch.clone(matrix.values),
        int(matrix.nnz),
    )


@overload
def _storage_alias(matrix: odt.BsrMatrix[wp.float32]) -> odt.BsrMatrix[wp.float32]: ...
@overload
def _storage_alias(matrix: odt.BsrMatrix[wp.float64]) -> odt.BsrMatrix[wp.float64]: ...
def _storage_alias(matrix: odt.BsrMatrix[odt.Block]) -> odt.BsrMatrix[odt.Block]:
    """Build a second ``BsrMatrix`` over ``matrix``'s arrays, holding them and not ``matrix``."""
    return _bsr_over(
        int(matrix.nrow),
        int(matrix.ncol),
        matrix.offsets,
        matrix.columns,
        matrix.values,
        int(matrix.nnz),
        row_counts=matrix.row_counts,
    )


def bsr_with_values(
    matrix: odt.SparseMatrix, values: wp.array[odt.Block, Any]
) -> odt.BsrMatrix[odt.Block]:
    """
    Build a sparse matrix with ``matrix``'s sparsity pattern and the given block values.

    The pattern is **shared**, not copied: the result holds ``matrix``'s ``offsets``, ``columns``
    and ``row_counts`` arrays, and ``matrix`` itself is not referenced. It is how a system whose
    pattern is known to equal an existing operator's -- a mass-plus-stiffness system over a
    Laplacian that already stores every diagonal -- is built without merging two patterns.

    Parameters
    ----------
    matrix
        ``(n_rows, n_cols)`` operator, in blocks, whose pattern the result takes. Rewriting that
        pattern in place afterwards rewrites the result's too.
    values
        ``(nnz,)`` one block per stored entry of ``matrix``, laid out as ``matrix.values`` is; its
        dtype is the result's block type.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_rows, n_cols)`` blocks, as ``matrix``, with ``matrix``'s stored entry count.

    Raises
    ------
    ValueError
        If ``values`` does not hold one block per stored entry of ``matrix``.

    See Also
    --------
    [`replicated_operator`][ordito.linalg.replicated_operator]
    """
    if values.shape[0] != matrix.values.size:
        raise ValueError(
            f"values must hold one block per stored entry ({matrix.values.shape[0]}), got "
            f"{values.shape[0]}"
        )
    return _bsr_over(
        int(matrix.nrow),
        int(matrix.ncol),
        matrix.offsets,
        matrix.columns,
        values,
        int(matrix.nnz),
        row_counts=matrix.row_counts,
    )


# The preconditioners this module built, keyed weakly: ``(kind, weak reference to the matrix they
# were built for)``, which ``_batched_preconditioner_kind`` reads to recognize one.
_PRECONDITIONER_TAGS: weakref.WeakKeyDictionary[
    wpl.LinearOperator, tuple[str, weakref.ref[odt.SparseMatrix]]
] = weakref.WeakKeyDictionary()


# ``block_diag``'s results, keyed weakly by the first operator of the stack.
_BLOCK_DIAG_CACHE: weakref.WeakKeyDictionary[
    odt.SparseMatrix, dict[tuple[int, ...], tuple[tuple[object, ...], odt.BsrMatrix[wp.float64]]]
] = weakref.WeakKeyDictionary()


def block_diag(matrices: Sequence[odt.SparseMatrix]) -> odt.BsrMatrix[wp.float64]:
    """
    Stack square sparse operators into one block-diagonal ``float64`` operator.

    The operator ``scipy.sparse.block_diag`` builds. Its use here is to solve several independent
    systems as one: a single conjugate-gradient iteration over the stack advances every block at
    once, in the launches of one solve rather than several, and stops when the slowest block does.
    A ``wp.mat22d`` operator enters as its scalar expansion over interleaved unknowns -- the memory
    of a ``wp.vec2d`` array viewed as ``float64`` -- so a tangent-field system and scalar systems
    stack together.

    The result is **kept per set of operators**: a later call with the same operators, over the
    same storage, returns the same matrix, so a solve against it replays its recorded loop (see
    [`solve_spd`][ordito.linalg.solve_spd]). The stack is a copy, so values rewritten in place
    in one of ``matrices`` afterwards do not reach it.

    Parameters
    ----------
    matrices
        One or more square operators, ``float64`` or ``wp.mat22d`` blocks, on one device.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n, n)`` operator whose row count ``n`` is the sum of the blocks' scalar row counts, in
        the order given.

    Raises
    ------
    ValueError
        If ``matrices`` is empty, or holds an operator that is not square or has another block type.
    RuntimeError
        If ``matrices`` are not all on one device.

    See Also
    --------
    [`solve_spd_settled`][ordito.linalg.solve_spd_settled]
    [`replicated_operator`][ordito.linalg.replicated_operator]
    """
    require_same_device(matrices=list(matrices))
    if len(matrices) == 0:
        raise ValueError("block_diag needs at least one operator")
    for matrix in matrices:
        if matrix.values.dtype not in (wp.float64, wp.mat22d) or matrix.nrow != matrix.ncol:
            raise ValueError(
                "block_diag stacks square float64 or wp.mat22d operators, got a "
                f"{matrix.nrow} x {matrix.ncol} operator of {matrix.values.dtype}"
            )
    identity = tuple(
        i
        for matrix in matrices
        for i in (id(matrix.offsets), id(matrix.columns), id(matrix.values))
    )
    entries = _BLOCK_DIAG_CACHE.setdefault(matrices[0], {})
    cached = entries.get(identity)
    if cached is not None:
        return cached[1]
    blocks = [
        _scalar_expansion(matrix) if matrix.values.dtype == wp.mat22d else matrix
        for matrix in matrices
    ]
    device = blocks[0].device
    n_rows = sum(int(block.nrow) for block in blocks)
    capacity = sum(block.values.size for block in blocks)
    descriptors = []
    for block in blocks:
        descriptor = kernel_linalg.CsrBlock()
        descriptor.offsets = block.offsets
        descriptor.columns = block.columns
        descriptor.values = block.values
        descriptor.n_rows = int(block.nrow)
        descriptors.append(descriptor)
    offsets = _launch.empty(n_rows + 1, dtype=wp.int32, device=device)
    columns = _launch.empty(capacity, dtype=wp.int32, device=device)
    values = _launch.empty(capacity, dtype=wp.float64, device=device)
    if n_rows == 0:
        _launch.zero_(offsets)
    else:
        _launch.launch(
            kernel_linalg.stack_block_diagonal,
            dim=n_rows,
            inputs=[_launch.array(descriptors, dtype=kernel_linalg.CsrBlock, device=device)],
            outputs=[offsets, columns, values],
            device=device,
        )
    # A capacity, as the blocks' own counts are: the offsets bound every row.
    stacked = _bsr_over(n_rows, n_rows, offsets, columns, values, capacity)
    # The entry holds every block's arrays, so the identities in its key cannot be reused.
    held = tuple((matrix.offsets, matrix.columns, matrix.values) for matrix in matrices)
    entries[identity] = (held, stacked)
    return stacked


def replicated_operator(
    matrix: odt.BsrMatrix[wp.float64] | wpl.LinearOperator, n_columns: int
) -> wpl.LinearOperator:
    """
    Present one ``(n, n)`` operator as ``n_columns`` independent subproblems over a flat vector.

    The returned operator acts on length-``n_columns * n`` vectors laid out as ``n_columns``
    contiguous blocks — exactly the memory of a contiguous ``(n_columns, n)`` array — and carries
    the ``batch_offsets`` that make ``warp.optim.linear``'s solvers iterate all blocks together and
    converge on the worst-case residual. ``matrix`` is **not** replicated in memory: the ``matvec``
    applies the single operator to each block in turn.

    Parameters
    ----------
    matrix
        ``(n, n)`` operator: a ``warp.sparse.BsrMatrix``, a 1-D array (read as a diagonal), or an
        existing ``warp.optim.linear.LinearOperator`` such as a preconditioner.
    n_columns
        Number of independent right-hand-side columns.

    Returns
    -------
    ``warp.optim.linear.LinearOperator``
        ``(n_columns * n, n_columns * n)`` batched operator.

    Notes
    -----
    A block-diagonal *matrix* would give the same batching at ``n_columns`` times the operator
    memory, and 2x2 blocks holding a scalar multiple of the identity would cost four times the
    storage and flops per scalar entry. Replicating only the ``matvec`` avoids both.

    ``batch_offsets`` is attached and ``max_batch_length`` deliberately is not, although this
    function knows the value exactly (every subproblem is ``n`` scalars): passing it to skip
    redundant reduction levels for a known maximum subproblem size made no measurable difference
    here, so it would be one more argument to explain for nothing. ``warp.optim.linear``'s
    ``restart`` option was also considered and is worse: resetting the search direction throws away
    the Krylov space CG has built, costing more iterations rather than buying better-conditioned
    ones.

    See Also
    --------
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    """
    base = wpl.aslinearoperator(matrix)
    if n_columns == 1:
        # Nothing to batch: hand back the operator itself rather than wrapping every matvec in a
        # one-iteration Python loop. ``lscm`` solves a single coupled column and would otherwise pay
        # the wrapper cost for no benefit.
        return base
    n = int(base.shape[0])

    # Per-block views, cached by buffer address. A solver calls ``matvec`` once per iteration with
    # the *same* handful of temporaries, so slicing them fresh every time would add thousands of
    # Python-level array constructions to a solve — enough to erase the batching win outright, and
    # enough to keep the iteration from being captured as a CUDA graph. The cache is keyed on the
    # full memory layout, so a different buffer (or a re-slice of one) never aliases a stale view.
    block_views: dict[tuple[object, ...], list[odt.ArrayNd]] = {}

    def blocks(array: odt.ArrayNd) -> list[odt.ArrayNd]:
        key = (array.ptr, tuple(array.shape), tuple(array.strides))
        views = block_views.get(key)
        if views is None:
            views = [
                odt.as_dense(array[column * n : (column + 1) * n]) for column in range(n_columns)
            ]
            block_views[key] = views
        return views

    def matvec(x: odt.ArrayNd, y: odt.ArrayNd, z: odt.ArrayNd, alpha: float, beta: float) -> None:
        x_blocks, y_blocks, z_blocks = blocks(x), blocks(y), blocks(z)
        for column in range(n_columns):
            base.matvec(x_blocks[column], y_blocks[column], z_blocks[column], alpha, beta)

    offsets = _launch.array(
        [column * n for column in range(n_columns + 1)], dtype=wp.int32, device=base.device
    )
    total = n_columns * n
    return wpl.LinearOperator(
        (total, total), base.dtype, base.device, matvec, batch_offsets=offsets
    )


def multigrid_preconditioner(
    matrix: odt.BsrMatrix[wp.float64], n_columns: int = 1, *, seed: int = 0
) -> wpl.LinearOperator:
    """

    Smoothed-aggregation multigrid preconditioner for a symmetric positive-semi-definite operator.

    A single-level preconditioner cannot break conjugate gradient's growth in the mesh size -- see
    "Why Jacobi" in the [`ordito.linalg`][ordito.linalg] module documentation for the alternatives
    that were tried and rejected -- because cutting the iteration count by ``sqrt(k)`` at ``k``
    mat-vecs per apply leaves the total work growing. A multigrid V-cycle attacks the low-frequency
    error the smoother cannot see, so the iteration count grows much more slowly with the problem
    size than it does under Jacobi.

    The returned operator acts on length-``n_columns * n`` vectors laid out as ``n_columns``
    contiguous blocks, exactly as [`replicated_operator`][ordito.linalg.replicated_operator] does,
    so it drops into ``M=`` beside a Jacobi preconditioner without any other change.


    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-(semi-)definite operator. Zero diagonal entries are allowed:
        such a row is identically zero for a semi-definite operator, and the whole chain --
        smoother, coarse solve and all -- leaves those unknowns at zero.
    n_columns
        Number of independent right-hand-side columns the operator will be applied to.
    seed
        Seeds the aggregation's randomized priorities and the power iteration's start vector, so a
        hierarchy is a deterministic function of the operator and this number.

    Returns
    -------
    ``warp.optim.linear.LinearOperator``
        ``(n_columns * n, n_columns * n)`` V-cycle. A **Jacobi** preconditioner instead when the
        operator does not coarsen -- see Notes.

    Raises
    ------
    ValueError
        If the returned operator's ``matvec`` is called with ``alpha != 1`` or ``beta != 0``. A
        preconditioner apply is always ``z = M x``, and the general form would cost a pass that no
        caller needs.

    Notes
    -----
    Setup is not free and it is not amortized over anything: building the hierarchy costs one
    aggregation, one power iteration, two sparse products and one transpose per level. **Ask for
    this where the solve dominates the call**, and pass the same operator's preconditioner into a
    loop rather than rebuilding it -- [`solve_spd`][ordito.linalg.solve_spd]'s ``preconditioner``
    parameter exists for exactly that.

    **The setup is host cost and it is per *level*, not per unknown, so it cannot be cut by making
    the operator smaller.** A level's cost is dominated by host-side overhead in
    ``warp.sparse.bsr_mm``, which makes device-to-host readbacks to size its output and so costs
    roughly the same regardless of the level's size. That is Warp's, not this package's, so the
    reachable lever is to build **fewer levels**, which is what ``_MULTIGRID_MAX_COARSE`` is set
    for.

    Coarsening stops at 128 rows, or earlier if a level fails to shrink; the coarsest operator is
    then inverted densely on the host, which is exact and is a single launch inside the cycle where
    an iterative coarse solve would be a data-dependent loop. When coarsening stalls while the level
    is still too large to factor, there is no usable hierarchy and this hands back
    ``warp.optim.linear.preconditioner(matrix, "diag")`` rather than a cycle whose coarse solve is a
    guess -- so a caller never has to branch on the operator's shape.

    !!! warning "That fallback is silent, and the strength threshold can trigger it"
        A stalled hierarchy is indistinguishable from a weak one at the call site: the solve simply
        runs at its Jacobi iteration count. ``_MULTIGRID_THETA`` decides how easily it happens --
        raising it makes more off-diagonals weak, and past some threshold an operator can stop
        coarsening entirely and come back at *exactly* the Jacobi count. So an aggregation change
        that "did nothing" should be checked against the level count before it is read as a change
        that did not help.

    See Also
    --------
    [`solve_spd`][ordito.linalg.solve_spd]
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    [`replicated_operator`][ordito.linalg.replicated_operator]
    """
    base = wpl.aslinearoperator(matrix)
    n = int(matrix.nrow)
    hierarchy = _multigrid_hierarchy(matrix, seed)
    if hierarchy is None:
        return replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    levels, coarse_inverse = hierarchy
    cycle = _MultigridCycle(levels, coarse_inverse, n_columns=n_columns, stride=n)

    def matvec(x: odt.ArrayNd, _y: odt.ArrayNd, z: odt.ArrayNd, alpha: float, beta: float) -> None:
        if alpha != 1.0 or beta != 0.0:
            raise ValueError(
                "multigrid_preconditioner's operator only implements z = M x "
                f"(got alpha={alpha}, beta={beta})"
            )
        cycle.apply(x, z)

    total = n_columns * n
    return wpl.LinearOperator((total, total), base.dtype, base.device, matvec)


def chebyshev_preconditioner(
    matrix: odt.BsrMatrix[wp.float64], n_columns: int = 1
) -> wpl.LinearOperator:
    """
    Jacobi-Chebyshev polynomial preconditioner for a second-order symmetric definite operator.

    Applies ``z = p(D⁻¹ A) D⁻¹ r``, with ``D`` the diagonal of ``A`` and ``p`` the fixed
    degree-[`CHEBYSHEV_DEGREE`][ordito.linalg.CHEBYSHEV_DEGREE] Chebyshev polynomial that
    approximates ``1 / x`` over an interval covering ``D⁻¹ A``'s spectrum -- the Chebyshev
    semi-iteration on the Jacobi-scaled system, run a fixed number of steps from zero. That is
    ``D^-1/2 p(D^-1/2 A D^-1/2) D^-1/2``, symmetric, and definite with the sign of ``A`` whenever
    ``p`` is positive on the spectrum, which the interval guarantees: its upper end is ``D⁻¹ A``'s
    Gershgorin bound and its lower end,
    [`CHEBYSHEV_INTERVAL`][ordito.linalg.CHEBYSHEV_INTERVAL] over the unknown count, only sets how
    well the low end is resolved -- ``p`` stays positive below it.

    A conjugate-gradient solve on this hardware is bound by its launches rather than its flops, and
    each step of the polynomial is one fused mat-vec launch, where each iteration it removes is
    half a dozen launches and two reductions. So the iteration count falls several-fold and the
    solve with it, on well- and ill-conditioned Laplacian systems alike. It needs no setup beyond
    one copy of the matrix and one reduction for the bound: unlike
    [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner] there is no hierarchy.

    The setup -- one copy of the matrix and the bound's reduction, which reads one scalar back --
    runs at the operator's **first apply**, so building one that is never applied is free. That
    first apply therefore cannot sit inside a CUDA graph capture; ``warp.optim.linear.cg`` makes it
    before its own captured loop.

    !!! warning "The operator must be symmetric"
        ``p(D⁻¹ A) D⁻¹`` is symmetric only when ``A`` is, and conjugate gradient needs a symmetric
        preconditioner. Jacobi is symmetric whatever ``A`` is, which is why a mildly asymmetric
        system -- a row-normalized Laplacian's boundary rows -- still converges under it; do not
        hand such a system to this.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive- or negative-(semi-)definite operator, whose diagonal
        carries its sign -- a cotangent Laplacian in either convention, a mass-plus-
        stiffness heat system. An empty row is allowed: the unknown no equation touches stays at
        zero.
    n_columns
        Number of independent right-hand-side columns the operator will be applied to.

    Returns
    -------
    ``warp.optim.linear.LinearOperator``
        ``(n_columns * n, n_columns * n)`` preconditioner over ``n_columns`` contiguous blocks, as
        [`replicated_operator`][ordito.linalg.replicated_operator] lays them out.

    Raises
    ------
    ValueError
        If the returned operator's ``matvec`` is called with ``alpha != 1`` or ``beta != 0``. A
        preconditioner apply is always ``z = M x``.

    See Also
    --------
    [`solve_spd`][ordito.linalg.solve_spd]
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    [`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]
    """
    n = int(matrix.nrow)
    # Built on the first apply rather than here, so an operator handed out and never applied costs
    # nothing: the bound's reduction reads a scalar back, which would drain whatever the caller
    # has queued.
    cycle: list[_JacobiChebyshevApply] = []

    def matvec(x: odt.ArrayNd, _y: odt.ArrayNd, z: odt.ArrayNd, alpha: float, beta: float) -> None:
        if alpha != 1.0 or beta != 0.0:
            raise ValueError(
                "chebyshev_preconditioner's operator only implements z = M x "
                f"(got alpha={alpha}, beta={beta})"
            )
        if not cycle:
            cycle.append(_JacobiChebyshev(matrix).bind(n_columns, n))
        cycle[0].apply(x, z)

    total = n_columns * n
    operator = wpl.LinearOperator((total, total), wp.float64, matrix.values.device, matvec)
    if n_columns == 1:
        _PRECONDITIONER_TAGS[operator] = ("chebyshev", weakref.ref(matrix))
    return operator


def squared_laplacian_preconditioner(
    laplacian: odt.BsrMatrix[wp.float64], weight_sums: wp.array[wp.float64]
) -> SquaredLaplacianPreconditioner:
    """
    Preconditioner for normal equations ``(MᵀM) x = Mᵀ b`` whose square block is ``D⁻¹ L``.

    A least-squares umbrella system (``smoothing.smooth_region``'s) is a *squared*, fourth-order
    operator: its condition number is roughly the square of the Laplacian's, so Jacobi needs
    several times the iterations a Laplacian solve takes on the same unknowns, and smoothed
    aggregation -- built for second-order operators -- recovers only part of that. The standard
    remedy for a biharmonic-type system is to precondition it with the square of a second-order
    one. When the free rows of ``M`` form the square block ``M_ff = D⁻¹ L`` for a symmetric
    positive-definite ``L`` and a positive diagonal ``D``, ``M_ffᵀ M_ff`` is spectrally close to
    ``MᵀM`` -- the rows ``M`` adds beyond ``M_ff`` are one ring of the boundary -- so its inverse
    ``M_ff⁻¹ M_ff⁻ᵀ`` preconditions ``MᵀM`` well.

    It is applied as ``B Bᵀ``, with ``B`` a fixed Chebyshev polynomial in ``M_ff`` that
    approximates its inverse over ``[a, b]``. ``b`` is ``M_ff``'s Gershgorin bound -- ``2`` when
    every weight is non-negative, more when some are negative, as clamped cotangent weights can
    be -- which keeps the polynomial positive on every eigenvalue, so ``B`` is nonsingular and
    ``B Bᵀ`` symmetric positive-definite -- what conjugate gradient needs -- whatever ``a`` is.
    ``a`` only sets how well the low-frequency end is resolved; it is taken as
    [`SQUARED_LAPLACIAN_INTERVAL`][ordito.linalg.SQUARED_LAPLACIAN_INTERVAL] over the unknown
    count, the rate at which a disc-like region's smallest eigenvalue falls. The setup is two
    copies of the matrix and one reduction for ``b``: a polynomial needs no hierarchy.

    Parameters
    ----------
    laplacian
        ``(n, n)`` symmetric positive-definite ``L``. An empty row is allowed and
        is left at zero, as it is in the system: an unknown no equation touches.
    weight_sums
        ``(n,)`` diagonal of ``D``, positive. ``M_ff = D⁻¹ L``; give ``1`` for an empty row.

    Returns
    -------
    SquaredLaplacianPreconditioner
        Hand to [`solve_spd_columns`][ordito.linalg.solve_spd_columns] as ``preconditioner=``.

    Raises
    ------
    RuntimeError
        If ``laplacian`` and ``weight_sums`` are not all on one device.

    See Also
    --------
    [`solve_spd_columns`][ordito.linalg.solve_spd_columns]
    [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner]
    """
    require_same_device(laplacian=laplacian.values, weight_sums=weight_sums)
    return SquaredLaplacianPreconditioner(laplacian, weight_sums)


# Degree of ``squared_laplacian_preconditioner``'s Chebyshev polynomial: ``degree - 1`` sparse
# steps, one launch each, per factor, and two factors per apply. The cost that matters is the total
# step count, iterations times launches per iteration, and 12 keeps it near its minimum from hole
# patches of a thousand unknowns up to regions of ten thousand.
SQUARED_LAPLACIAN_DEGREE = 12

# The polynomial's interval starts at ``SQUARED_LAPLACIAN_INTERVAL / n`` for ``n`` unknowns, capped
# at ``SQUARED_LAPLACIAN_INTERVAL_CAP`` so a tiny system still has an interval. The best lower end
# falls as ``1 / n``, like a disc-like region's smallest eigenvalue, and the iteration count is
# flat to within a few percent over a factor of three either side of it.
SQUARED_LAPLACIAN_INTERVAL = 40.0
SQUARED_LAPLACIAN_INTERVAL_CAP = 0.5

# Degree and interval of ``chebyshev_preconditioner``'s polynomial: ``degree - 1`` fused mat-vec
# launches per apply, and a lower end of ``CHEBYSHEV_INTERVAL / n`` under the same cap. Both sit
# where the solve's total launch count is flat, from a few thousand unknowns to twenty thousand and
# across the uniform and graded members of the saddle pair; a lower end several times smaller
# costs up to twice the iterations, a larger one is flat.
CHEBYSHEV_DEGREE = 12
CHEBYSHEV_INTERVAL = 80.0


class SquaredLaplacianPreconditioner:
    """
    The preconditioner a normal-equations solve is handed.

    Built by [`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner].

    Holds the square block ``M_ff = D⁻¹ L``, its transpose and the polynomial's coefficients, and
    binds a set of working vectors to the column layout of each solve that uses it.
    """

    def __init__(
        self, laplacian: odt.BsrMatrix[wp.float64], weight_sums: wp.array[wp.float64]
    ) -> None:
        """Build ``M_ff`` and its transpose from ``L`` and ``D``, and fix the interval."""
        self._n = int(laplacian.nrow)
        self._device = laplacian.values.device
        # ``M_ff``'s values and the Gershgorin ratios below come from one walk of ``L``'s rows, and
        # ``M_ff`` takes ``L``'s pattern rather than a copy of it.
        factor_values = _launch.empty_like(laplacian.values)
        ratios = odt.empty_1d(self._n, wp.float64, device=self._device)
        _launch.launch(
            kernel_linalg.scaled_rows_and_abs_sums,
            dim=self._n,
            inputs=[laplacian.offsets, laplacian.values, weight_sums],
            outputs=[factor_values, ratios],
            device=self._device,
        )
        self._factor = bsr_with_values(laplacian, factor_values)
        self._factor_t = od.array.csr_transpose(self._factor)
        # The upper end is ``M_ff``'s Gershgorin bound, ``max_i sum_j |L_ij| / D_i``, never below
        # 2. With ``D`` the diagonal of ``L`` it is exactly 2 when every weight is non-negative and
        # exceeds it when a clamped cotangent weight is negative, as on a regular grid's near-right
        # triangles; with any other ``D`` (``sqrt(M)`` for the k = 2 harmonic operator) it is
        # nowhere near 2 -- and a Chebyshev polynomial grows without bound past the end of its
        # interval, so an eigenvalue beyond a fixed 2 would be amplified rather than inverted. The
        # lower end scales with it, so the interval is the same *relative* one whatever ``D``'s
        # units: the polynomial in ``c M_ff`` is ``c`` times the one in ``M_ff``, and conjugate
        # gradient's iterates do not see a constant factor on the preconditioner.
        # The bound, the interval and the steps stay on the device (``_device_chebyshev_steps``):
        # the interval is ``[min(SQUARED_LAPLACIAN_INTERVAL / n, SQUARED_LAPLACIAN_INTERVAL_CAP) *
        # upper / 2, upper]`` with ``upper = max(bound, 2)``.
        self._steps = _device_chebyshev_steps(
            ratios, SQUARED_LAPLACIAN_INTERVAL, SQUARED_LAPLACIAN_DEGREE, squared=True
        )
        self._narrowed: tuple[wp.array[wp.float32], wp.array[wp.float32]] | None = None

    @classmethod
    def from_factors(
        cls,
        factor: odt.BsrMatrix[wp.float64],
        factor_t: odt.BsrMatrix[wp.float64],
        ratios: wp.array[wp.float64],
        narrowed: tuple[wp.array[wp.float32], wp.array[wp.float32]] | None = None,
    ) -> SquaredLaplacianPreconditioner:
        """
        Build the preconditioner from a caller-assembled ``M_ff = D⁻¹ L`` and its transpose.

        For a caller that writes the factor while assembling its system, as
        ``smoothing.smooth_region``'s normal equations do: the constructor's copy, row scaling,
        transpose and Gershgorin reduction are then work already done.

        Parameters
        ----------
        factor
            ``(n, n)`` ``D⁻¹ L``.
        factor_t
            ``(n, n)`` transpose of ``factor``. It may share ``factor``'s pattern, holding explicit
            zeros.
        ratios
            ``(n,)`` ``sum_j |L_ij| / D_i``, ``0`` for an empty row: what the constructor's
            ``scaled_rows_and_abs_sums`` writes.
        narrowed
            ``factor``'s and ``factor_t``'s values in ``float32``, when the caller wrote them
            too; otherwise ``narrowed()``
            casts them on first use.

        Returns
        -------
        SquaredLaplacianPreconditioner
            Hand to [`solve_spd_columns`][ordito.linalg.solve_spd_columns] as ``preconditioner=``.
        """
        self = cls.__new__(cls)
        self._n = int(factor.nrow)
        self._device = factor.values.device
        self._factor = factor
        self._factor_t = factor_t
        self._steps = _device_chebyshev_steps(
            ratios, SQUARED_LAPLACIAN_INTERVAL, SQUARED_LAPLACIAN_DEGREE, squared=True
        )
        self._narrowed = narrowed
        return self

    def bind(self, n_columns: int, stride: int) -> _SquaredLaplacianApply:
        """Working vectors for ``n_columns`` blocks at column pitch ``stride``."""
        return _SquaredLaplacianApply(self, n_columns, stride)

    def _shares_pattern(self) -> bool:
        """Whether ``Bᵀ`` is stored over ``B``'s own pattern arrays, as region solves write it."""
        return (
            self._factor_t.offsets is self._factor.offsets
            and self._factor_t.columns is self._factor.columns
        )

    def _pool_shape(self) -> tuple[object, ...]:
        """Return what a pooled solver state's copy of this must match to be rewritten in place."""
        return (
            "squared",
            self._shares_pattern(),
            *(
                array.size
                for factor in (self._factor, self._factor_t)
                for array in (factor.offsets, factor.columns, factor.values)
            ),
            self._steps.size,
        )

    def _owned_copy(self) -> SquaredLaplacianPreconditioner:
        """Copy onto storage of its own, for a pooled solver state to hold and ``_assign`` to."""
        copy = SquaredLaplacianPreconditioner.__new__(SquaredLaplacianPreconditioner)
        copy._n = self._n
        copy._device = self._device
        copy._factor = _owned_copy(self._factor)
        copy._factor_t = (
            bsr_with_values(copy._factor, _launch.clone(self._factor_t.values))
            if self._shares_pattern()
            else _owned_copy(self._factor_t)
        )
        copy._steps = _launch.clone(self._steps)
        copy._narrowed = None
        return copy

    def _assign(self, other: SquaredLaplacianPreconditioner) -> None:
        """Copy ``other``'s arrays into this one's, which must have ``other``'s ``_pool_shape``."""
        factors = [(self._factor, other._factor), (self._factor_t, other._factor_t)]
        for index, (mine, theirs) in enumerate(factors):
            if index == 0 or not other._shares_pattern():
                _launch.copy(mine.offsets, theirs.offsets)
                _launch.copy(mine.columns, theirs.columns)
            _launch.copy(mine.values, theirs.values)
        _launch.copy(self._steps, other._steps)
        self._narrowed = None

    def narrowed(self) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
        """
        Return ``B``'s and ``Bᵀ``'s values in ``float32``, built once.

        What ``_cg_one_block`` applies the polynomial with: see
        ``kernels/algorithms/conjugate_gradient.one_block_precondition`` for why in ``float32``.
        """
        if self._narrowed is None:
            factor = _launch.empty(self._factor.values.size, dtype=wp.float32, device=self._device)
            factor_t = _launch.empty(
                self._factor_t.values.size, dtype=wp.float32, device=self._device
            )
            od.array.copyto(factor, self._factor.values.flatten())
            od.array.copyto(factor_t, self._factor_t.values.flatten())
            self._narrowed = (factor, factor_t)
        return self._narrowed


class _ChebyshevApply:
    """
    Working vectors for Chebyshev polynomials over the padded column blocks of one CG state.

    Shared by both polynomial preconditioners, which differ only in what they wrap the polynomial
    in: [`SquaredLaplacianPreconditioner`][ordito.linalg.SquaredLaplacianPreconditioner] applies
    two factors, ``B Bᵀ``, and ``_JacobiChebyshev`` one, after a Jacobi scaling.
    """

    def __init__(
        self, n: int, n_columns: int, stride: int, steps: wp.array[wp.vec4d], device: wp.DeviceLike
    ) -> None:
        self._n = n
        self._stride = stride
        self._steps = steps
        self._device = device
        self._dim = n_columns * n
        # Zeroed: every step writes the ``n`` live rows of a column and nothing in the pad, which
        # the conjugate-gradient state reduces over.
        self._dofs = n_columns * stride
        self._spare = [_launch.zeros(self._dofs, dtype=wp.float64, device=device) for _ in range(3)]

    def _polynomial(
        self,
        factor: odt.BsrMatrix[wp.float64],
        source: wp.array[wp.float64],
        destination: wp.array[wp.float64],
        row_scale: wp.array[wp.float64] | None = None,
    ) -> None:
        """
        ``destination = p(factor) source``, one launch per Chebyshev step.

        With ``row_scale``, ``p(diag(row_scale) factor) source`` instead, scaled inside each step.
        """
        steps = self._steps
        # Three rotating iterates: a step reads the current and the previous one and writes a
        # third, so none of the three may be ``source`` or ``destination`` until the last write.
        # The first two steps read ``source`` in the place of the iterates no launch writes; see
        # ``chebyshev_step``.
        previous, current = source, source
        free = list(self._spare)
        n_steps = steps.size
        for index in range(n_steps):
            target = destination if index == n_steps - 1 else free.pop()
            _launch.launch(
                kernel_mg.chebyshev_step,
                dim=self._dim,
                inputs=[
                    wp.int32(self._n),
                    wp.int32(self._stride),
                    steps,
                    wp.int32(index),
                    factor.offsets,
                    factor.columns,
                    factor.values,
                    wp.int32(0 if row_scale is None else 1),
                    row_scale,
                    source,
                    current,
                    previous,
                ],
                outputs=[target],
                device=self._device,
            )
            # The buffer two steps back is free again; ``source`` never is.
            if previous is not source:
                free.append(previous)
            previous, current = current, target


class _SquaredLaplacianApply(_ChebyshevApply):
    """``z = B Bᵀ r`` over the padded column blocks of one conjugate-gradient state."""

    def __init__(self, owner: SquaredLaplacianPreconditioner, n_columns: int, stride: int) -> None:
        super().__init__(owner._n, n_columns, stride, owner._steps, owner._device)  # pyright: ignore[reportPrivateUsage]
        self._owner = owner
        self._middle = _launch.zeros(self._dofs, dtype=wp.float64, device=self._device)

    @property
    def owner(self) -> SquaredLaplacianPreconditioner:
        """The preconditioner whose arrays the recorded polynomial reads."""
        return self._owner

    def apply(self, source: wp.array[wp.float64], destination: wp.array[wp.float64]) -> None:
        """``destination = B Bᵀ source``: ``Bᵀ`` first, then ``B``."""
        self._polynomial(self._owner._factor_t, source, self._middle)  # pyright: ignore[reportPrivateUsage]
        self._polynomial(self._owner._factor, self._middle, destination)  # pyright: ignore[reportPrivateUsage]


class _JacobiChebyshev:
    """``A``, ``D⁻¹`` and the polynomial's steps in ``D⁻¹ A``, for one operator."""

    def __init__(self, matrix: odt.BsrMatrix[wp.float64]) -> None:
        self.n = int(matrix.nrow)
        self.device = matrix.values.device
        # ``A`` itself, not a row-scaled copy: each step scales its row's product by ``D⁻¹``, which
        # is the same arithmetic without the copy -- and ``bsr_copy`` is most of what a setup would
        # otherwise cost.
        self.matrix = matrix
        self.inverse_diagonal = _launch.empty(self.n, dtype=wp.float64, device=self.device)
        self.ratios = odt.empty_1d(self.n, wp.float64, device=self.device)
        _launch.launch(
            kernel_linalg.jacobi_dominance_rows,
            dim=self.n,
            inputs=[matrix.offsets, matrix.columns, matrix.values],
            outputs=[self.inverse_diagonal, self.ratios],
            device=self.device,
        )
        # Gershgorin's discs for ``D⁻¹ A`` are centred on 1 with radius ``r_i = sum_j |A_ij| /
        # |A_ii|``, whatever the diagonal's sign. ``1 + max r`` bounds the spectrum from above, and
        # a polynomial fitted short of it changes sign past its end, as a clamped negative
        # cotangent weight takes a regular grid's spectrum past 2. A diagonally dominant operator
        # (a heat system's mass-plus-stiffness) also has ``1 - max r`` as a lower bound, which is
        # far tighter than the ``1 / n`` scale a Laplacian's smallest eigenvalue falls at.
        # The radius is floored so that a diagonal operator still has an interval to fit.
        # So the interval is ``[max(min(CHEBYSHEV_INTERVAL / n, SQUARED_LAPLACIAN_INTERVAL_CAP),
        # 1 - r), 1 + r]`` with ``r = max(max_i r_i, 1e-3)``, computed on the device
        # (``_device_chebyshev_steps``).
        self.steps = _device_chebyshev_steps(
            self.ratios, CHEBYSHEV_INTERVAL, CHEBYSHEV_DEGREE, squared=False
        )

    def refit(self) -> None:
        """Refit the steps, in place, to ``ratios`` as a pooled state's ``refresh`` rewrote them."""
        _device_chebyshev_steps(
            self.ratios, CHEBYSHEV_INTERVAL, CHEBYSHEV_DEGREE, squared=False, out=self.steps
        )

    def bind(self, n_columns: int, stride: int) -> _JacobiChebyshevApply:
        """Working vectors for ``n_columns`` blocks at column pitch ``stride``."""
        return _JacobiChebyshevApply(self, n_columns, stride)


class _JacobiChebyshevApply(_ChebyshevApply):
    """``z = p(D⁻¹ A) D⁻¹ r`` over the padded column blocks of one conjugate-gradient state."""

    def __init__(self, owner: _JacobiChebyshev, n_columns: int, stride: int) -> None:
        super().__init__(owner.n, n_columns, stride, owner.steps, owner.device)
        self._owner = owner
        self._scaled = _launch.zeros(self._dofs, dtype=wp.float64, device=self._device)

    @property
    def owner(self) -> _JacobiChebyshev:
        """The operator-side half, which a pooled state's ``refresh`` rewrites."""
        return self._owner

    @property
    def inverse_diagonal(self) -> wp.array[wp.float64]:
        """``D⁻¹``, length ``n``: what a caller writing ``scaled`` itself applies."""
        return self._owner.inverse_diagonal

    @property
    def scaled(self) -> wp.array[wp.float64]:
        """The polynomial's input, ``D⁻¹ source``, which ``apply_scaled`` reads."""
        return self._scaled

    def apply_scaled(self, destination: wp.array[wp.float64]) -> None:
        """``destination = p(D⁻¹ A) scaled``, for a caller that has already written ``scaled``."""
        self._polynomial(
            self._owner.matrix, self._scaled, destination, row_scale=self._owner.inverse_diagonal
        )

    def apply(self, source: wp.array[wp.float64], destination: wp.array[wp.float64]) -> None:
        """``destination = p(D⁻¹ A) D⁻¹ source``: the Jacobi scaling, then the polynomial."""
        _launch.launch(
            kernel_cg.scaled_diagonal_apply,
            dim=self._dofs,
            inputs=[
                wp.int32(self._n),
                wp.int32(self._stride),
                self._owner.inverse_diagonal,
                wp.float64(1.0),
                source,
                self._scaled,
            ],
            device=self._device,
        )
        self.apply_scaled(destination)


def _device_chebyshev_steps(
    ratios: wp.array[wp.float64, Any],
    interval: float,
    degree: int,
    *,
    squared: bool,
    out: wp.array[wp.vec4d] | None = None,
) -> wp.array[wp.vec4d]:
    """
    Fit a degree-``degree`` Chebyshev semi-iteration's steps on the device.

    ``ratios`` holds each row's Gershgorin quantity; its maximum is reduced into a device scalar and
    ``kernels/linalg.chebyshev_steps`` fits the interval and writes one ``(scale, previous_scale,
    momentum, step)`` per launch of the polynomial, so building a preconditioner reads nothing
    back. ``squared`` picks the squared-Laplacian interval over the Jacobi-Chebyshev one; see
    that kernel for both. ``out``, when given, is written in place of a new array.
    """
    device = ratios.device
    n = ratios.size
    bound = _launch.full(1, -math.inf, dtype=wp.float64, device=device)
    if n > 0:
        _launch.launch_tiled(
            kernel_reduce.MAX1D_TILED[wp.float64],
            dim=[kernel_reduce.blocks_1d(n)],
            inputs=[ratios, bound],
            block_dim=TILE_1D,
            device=device,
        )
    steps = _launch.empty(degree - 1, dtype=wp.vec4d, device=device) if out is None else out
    _launch.launch(
        kernel_linalg.chebyshev_steps,
        dim=1,
        inputs=[
            bound,
            wp.int32(n),
            wp.int32(1 if squared else 0),
            wp.float64(interval),
            wp.float64(SQUARED_LAPLACIAN_INTERVAL_CAP),
        ],
        outputs=[steps],
        device=device,
    )
    return steps


# ---------------------------------------------------------------------------
# Smoothed-aggregation multigrid preconditioner
# ---------------------------------------------------------------------------

# Rows below which a level is solved exactly instead of coarsened further. The coarse solve is a
# dense pseudo-inverse factored on the host, so this is also the size of that factorization.
#
# The last level is expensive twice over: it costs a level's setup (one aggregation, one power
# iteration, a prolongator, a transpose and two ``bsr_mm``) and then a smoothing sweep, a
# restriction and a prolongation in *every* cycle, where solving that level densely instead is one
# matvec. 384 keeps headroom under ``_MULTIGRID_MAX_DENSE`` and the host ``pinv`` that guard sizes.
_MULTIGRID_MAX_COARSE = 384

# Hard cap on the hierarchy depth, and on the dense coarse solve. Coarsening stops early whenever a
# level fails to shrink by ``1 - _MULTIGRID_MIN_COARSENING``, which is what happens once a level is
# mostly isolated rows -- the least-squares operators here can carry a fair number of them.
_MULTIGRID_MAX_LEVELS = 12
_MULTIGRID_MIN_COARSENING = 0.9
_MULTIGRID_MAX_DENSE = 512

# Damped-Jacobi sweeps per level per half-cycle, and the damping as a multiple of ``1 / rho`` where
# ``rho`` is the spectral radius of ``D^-1 A``. 4/3 is the classical smoothed-aggregation choice and
# is used both for the smoother and for the prolongation smoother.
#
# Two sweeps is close to the fewest cycles needed for the fastest solve across the systems this
# hierarchy is built for; more sweeps reduce the iteration count further but at a cost per cycle
# that offsets the gain, so the curve in solve time is fairly flat around this value.
#
# A Chebyshev smoother was built to try to do better and is refuted -- do not rebuild it. A
# polynomial smoother of the same degree looked attractive on a synthetic system but did not
# transfer to the real ones this hierarchy solves: a solve cycle here is launch-bound rather than
# flop-bound, and a Chebyshev step costs more launches per iteration than a Jacobi sweep, so trading
# launches for a modest iteration-count reduction is the wrong side of this cost model. No single
# Chebyshev degree and interval was best across every system tried, either.
_MULTIGRID_SWEEPS = 2
_MULTIGRID_JACOBI_FACTOR = 4.0 / 3.0

# Power iterations for that spectral radius, and the round cap for the aggregation's independent
# set. The power iteration is unnormalized -- ``rho`` is recovered from the growth over all the
# steps -- so it costs one mat-vec and one elementwise pass per step, and two inner products. Eight
# rather than fifteen: the extra steps move the iteration count negligibly at real setup cost.
_MULTIGRID_POWER_STEPS = 8
_MULTIGRID_MIS_ROUNDS = 32

# Strength-of-connection threshold for the aggregation: an off-diagonal counts as an edge only when
# ``|A_ij| >= theta sqrt(A_ii A_jj)``. ``0.0`` keeps every off-diagonal, which is the aggregation
# this package shipped first and is bit-exactly what a zero threshold reduces to.
#
# A larger threshold removes weak edges from the strength graph, so aggregates are smaller and the
# hierarchy can grow a level -- each of which costs real setup -- while the iteration count falls.
# ``0.05`` sits in a basin where that trade is favourable on the systems that reach the hierarchy
# today: it buys enough of a fall in iteration count to pay back the extra setup, without pushing
# the coarsening into the range where it stalls (which, at a large enough threshold, leaves a level
# unable to coarsen at all and a caller silently falling back to a plain Jacobi preconditioner). The
# optimum is flat rather than sharp, so this should be moved only on evidence spanning multiple
# systems and both the setup and the solve, not a single system's clock.
_MULTIGRID_THETA = 0.05

# Triplet budget of one Galerkin-product assembly, as a multiple of the operator's stored entries
# plus rows, and an absolute cap that keeps the ``int32`` scan of the counts clear of overflow. The
# well-coarsened levels this hierarchy builds by default emit 1-3x their operator's size; a level
# whose coarsening stalled goes nearly dense and the per-row products become quadratic in the row
# length (a billion triplets on ``icosphere(6)`` at ``theta = 0.2``), so such a level falls back to
# ``bsr_mm``, whose memory follows its output.
_MULTIGRID_TRIPLET_FACTOR = 16
_MULTIGRID_TRIPLET_CAP = 1 << 30


class _MultigridLevel:
    """One level of the hierarchy: its operator, its smoother, and its link to the next."""

    __slots__ = (
        "ax",
        "b",
        "dim",
        "inverse_diagonal",
        "matvec_dim",
        "n",
        "operator",
        "prolong_dim",
        "prolongator",
        "r",
        "restrict_dim",
        "restrictor",
        "stride",
        "x",
    )

    def __init__(self, operator: odt.BsrMatrix[wp.float64]) -> None:
        self.operator = operator
        self.n = int(operator.nrow)
        self.prolongator: odt.BsrMatrix[wp.float64] | None = None
        self.restrictor: odt.BsrMatrix[wp.float64] | None = None
        # ``omega D^-1``, the smoother's damping folded in (``_multigrid_damped_diagonal``).
        self.inverse_diagonal: wp.array[wp.float64] | None = None


def _multigrid_hierarchy(
    matrix: odt.BsrMatrix[wp.float64], seed: int
) -> tuple[list[_MultigridLevel], odt.ArrayNd] | None:
    """
    Coarsen ``matrix`` until a level is small enough to invert densely.

    ``None`` when no usable hierarchy exists -- the coarsening stalled above the dense cap -- which
    is the one case [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner] hands back
    a Jacobi preconditioner instead.
    """
    levels: list[_MultigridLevel] = []
    operator = matrix
    while True:
        level = _MultigridLevel(operator)
        levels.append(level)
        if level.n <= _MULTIGRID_MAX_COARSE or len(levels) >= _MULTIGRID_MAX_LEVELS:
            break
        # One extraction per level, read by both consumers: the aggregation's strength test wants
        # ``sqrt(|A_ii|)`` and the smoother wants ``1 / A_ii``. ``wps.bsr_get_diag`` is an
        # allocation *and* a launch, so extracting it once here rather than once per consumer saves
        # a redundant diagonal extraction per level.
        diag = odt.bsr_get_diag(operator)
        label, sizes, n_aggregates = _multigrid_aggregate(operator, diag, seed)
        if n_aggregates >= _MULTIGRID_MIN_COARSENING * level.n:
            break
        # ``omega D^-1``, the damping already folded in, so the smoother and the prolongator read
        # one scaled diagonal and no level reads its spectral radius back to the host.
        level.inverse_diagonal = _multigrid_damped_diagonal(operator, diag, seed)
        level.prolongator = _multigrid_prolongator(
            operator, label, sizes, n_aggregates, level.inverse_diagonal
        )
        level.restrictor = od.array.csr_transpose(level.prolongator)
        operator = _multigrid_galerkin(operator, level.prolongator)
    coarse_inverse = _multigrid_dense_inverse(levels[-1].operator)
    if coarse_inverse is None:
        return None
    return levels, coarse_inverse


def _multigrid_aggregate(
    matrix: odt.BsrMatrix[wp.float64], diagonal: wp.array[wp.float64], seed: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int]:
    """
    Aggregate label per row, from a distance-2 maximal independent set on the off-diagonal graph.

    Returns ``(label, sizes, n_aggregates)``, ``sizes`` being each aggregate's row count, counted
    by the spread's final hop.

    The selection is the randomized-priority pattern ``sample.dart_select_minima`` runs, lifted to
    distance 2 by propagating the packed ``(state, priority, index)`` maximum over one-hop
    neighbours *twice* per round -- so no squared graph is built. Roots then spread their label two
    hops, which tiles the graph because the MIS keeps them at least three hops apart.

    Both walks are over the operator's **strong** off-diagonal graph
    (``|A_ij| >= _MULTIGRID_THETA sqrt(A_ii A_jj)``), tested per edge rather than materialized, so
    that a level pays no extra allocation for the filter and ``theta = 0`` is bit-exactly the
    unfiltered aggregation. ``diagonal`` is ``matrix``'s, passed in because the caller has already
    extracted it for the smoother.
    """
    device = matrix.device
    n = int(matrix.nrow)
    offsets, columns, values = matrix.offsets, matrix.columns, matrix.values
    theta = wp.float64(_MULTIGRID_THETA)
    # ``sqrt(|A_ii|)`` per row, so the strength test below is a product rather than a square root
    # per edge, read by both walks; the priority order; and the undecided start state -- one
    # launch. The ``diagonal`` argument is the caller's, because the hierarchy needs the same
    # extraction for the smoother and must not repeat it here.
    scaled_diagonal = _launch.empty(n, dtype=wp.float64, device=device)
    priority = _launch.empty(n, dtype=wp.uint32, device=device)
    state = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_mg.mis_level_setup,
        dim=n,
        inputs=[wp.int32(seed), diagonal],
        outputs=[scaled_diagonal, priority, state],
        device=device,
    )
    next_state = _launch.empty(n, dtype=wp.int32, device=device)
    key = _launch.empty(n, dtype=wp.int64, device=device)
    # Every node's root flag, written by the round that decides it (and every round it is still
    # undecided), so the scan below needs no map over the final state.
    flags = _launch.empty(n, dtype=wp.int32, device=device)
    # Cleared by every round's first launch.
    undecided = _launch.empty(1, dtype=wp.int32, device=device)
    for _ in range(_MULTIGRID_MIS_ROUNDS):
        # Two launches a round: the first hop forms every key from the state as it reads it, the
        # second hop decides as it reduces.
        _launch.launch(
            kernel_mg.mis_propagate_states,
            dim=n,
            inputs=[state, priority, offsets, columns, values, scaled_diagonal, theta],
            outputs=[key, undecided],
            device=device,
        )
        _launch.launch(
            kernel_mg.mis_propagate_decide,
            dim=n,
            inputs=[key, priority, offsets, columns, values, scaled_diagonal, theta, state],
            outputs=[next_state, flags, undecided],
            device=device,
        )
        state, next_state = next_state, state
        # One 4-byte read per round, and there are a handful of rounds: the loop cannot be a
        # device-side one because the *number of aggregates* sizes every buffer downstream.
        if int(read_scalar(undecided, 0)) == 0:
            break

    # The root flags are scanned in place: only their scan is read from here on.
    scan_pos = flags
    _launch.array_scan(flags, scan_pos, inclusive=True)
    n_aggregates = int(read_scalar(scan_pos))

    # Two hops of the spread. The first reads each label straight from ``(state, scan_pos)`` (its
    # ``label`` argument is not read); the second reads the first's output.
    strong_graph = [offsets, columns, values, scaled_diagonal, theta]
    # ``next_state`` is the loop's spent ping-pong buffer, free to hold the first hop.
    hop = next_state
    _launch.launch(
        kernel_mg.spread_aggregate_labels,
        dim=n,
        inputs=[hop, state, scan_pos, wp.int32(1), *strong_graph],
        outputs=[hop, None],
        device=device,
    )
    label = _launch.empty(n, dtype=wp.int32, device=device)
    sizes = _launch.zeros(n_aggregates, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_mg.spread_aggregate_labels,
        dim=n,
        inputs=[hop, state, scan_pos, wp.int32(0), *strong_graph],
        outputs=[label, sizes],
        device=device,
    )
    return label, sizes, n_aggregates


def _multigrid_damped_diagonal(
    matrix: odt.BsrMatrix[wp.float64], diagonal: wp.array[wp.float64], seed: int
) -> wp.array[wp.float64]:
    """
    ``omega D^-1``, with ``omega = 4/3 / rho`` and ``rho`` the spectral radius of ``D^-1 A``.

    ``diagonal`` is ``matrix``'s own; ``D^-1`` is ``array.inverse_or_one`` of it, which the kernels
    apply as they read it.

    ``rho`` comes from an unnormalized power iteration. Normalizing every step would cost a host
    readback per step; leaving the iterate to grow and taking the geometric mean of the growth over
    all the steps costs none at all, because the start vector is a sign vector whose squared norm is
    exactly ``n`` and the growth is folded into the diagonal on the device
    (``kernels/algorithms/multigrid.damped_inverse_diagonal``). ``rho`` is around 3 here, so eight
    steps grow the vector by about ``3 ** 8`` and ``float64`` has room to spare. The estimate
    approaches ``rho`` from below, which is the safe side: it makes the damping *smaller* than the
    stability limit rather than larger.

    Everything about the arithmetic here is chosen against the launch count, because the hierarchy
    build is launch-bound. A step is one fused ``power_step`` launch rather than a ``bsr_mv`` plus
    an elementwise scale, and the two buffers are ping-ponged rather than updated in place, which is
    what allows the single kernel. The first step draws the sign vector as it reads it.
    """
    device = matrix.device
    n = int(matrix.nrow)
    x = _launch.empty(n, dtype=wp.float64, device=device)
    y = _launch.empty(n, dtype=wp.float64, device=device)
    for step in range(_MULTIGRID_POWER_STEPS - 1):
        _launch.launch(
            kernel_mg.power_step,
            dim=n,
            inputs=[
                wp.int32(seed),
                wp.int32(step == 0),
                diagonal,
                matrix.offsets,
                matrix.columns,
                matrix.values,
                x,
            ],
            outputs=[y],
            device=device,
        )
        x, y = y, x
    # How much the iterate grew over ``K`` steps of ``D^-1 A`` is ``rho ** K`` to the accuracy this
    # needs, and it stays on the device. The start is exact: every entry of a sign vector is +-1,
    # so its squared norm is exactly n. The last step stores each block's share of the squared norm
    # in its own slot; past ``4 * ITEMS_PER_BLOCK_1D`` of them they are folded in a fixed tree,
    # and ``damped_inverse_diagonal`` folds the rest in every block, so the total is the same
    # number on every run with no inner-product pass of its own.
    blocks = -(-n // int(kernel_mg.POWER_ROWS_PER_BLOCK))
    partials = _launch.empty(blocks, dtype=wp.float64, device=device)
    _launch.launch_tiled(
        kernel_mg.power_step_partials,
        dim=[blocks],
        inputs=[
            wp.int32(seed),
            # Never the first step: ``_MULTIGRID_POWER_STEPS`` is well above one.
            wp.int32(0),
            diagonal,
            matrix.offsets,
            matrix.columns,
            matrix.values,
            x,
        ],
        outputs=[y, partials],
        block_dim=TILE_1D,
        device=device,
    )
    while partials.size > 4 * kernel_reduce.ITEMS_PER_BLOCK_1D:
        folded = _launch.empty(
            kernel_reduce.blocks_1d(partials.size), dtype=wp.float64, device=device
        )
        _launch.launch_tiled(
            kernel_reduce.SUM1D_PARTIALS[wp.float64],
            dim=[folded.size],
            inputs=[partials, folded],
            block_dim=TILE_1D,
            device=device,
        )
        partials = folded
    # The last step read ``x`` and wrote ``y``, so ``x`` is free to hold the result.
    damped = x
    _launch.launch_tiled(
        kernel_mg.damped_inverse_diagonal,
        dim=[blocks],
        inputs=[
            partials,
            wp.float64(n),
            wp.float64(1.0 / _MULTIGRID_POWER_STEPS),
            wp.float64(_MULTIGRID_JACOBI_FACTOR),
            diagonal,
        ],
        outputs=[damped],
        block_dim=TILE_1D,
        device=device,
    )
    return damped


def _multigrid_prolongator(
    matrix: odt.BsrMatrix[wp.float64],
    label: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    n_aggregates: int,
    damped_inverse_diagonal: wp.array[wp.float64],
) -> odt.BsrMatrix[wp.float64]:
    """
    Smoothed prolongator ``(I - omega D^-1 A) P0``, given ``omega D^-1``, for ``P0``.

    ``P0`` has one entry a row, so the product is ``A``'s entries re-keyed by their column's
    aggregate and summed: one triplet per stored entry plus the identity's, one assembly
    (``kernels/algorithms/multigrid.smoothed_prolongator_triplets``), with no sparse product.
    Entries that sum to exactly zero are pruned.
    """
    device = matrix.device
    n = int(matrix.nrow)
    # The stored count, not ``matrix.nnz``: that field is a capacity after a triplet build, and the
    # triplets are laid out at the slots the entries occupy.
    count = int(read_scalar(matrix.offsets, n)) + n
    rows = _launch.empty(count, dtype=wp.int32, device=device)
    cols = _launch.empty(count, dtype=wp.int32, device=device)
    values = _launch.empty(count, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_mg.smoothed_prolongator_triplets,
        dim=n,
        inputs=[
            matrix.offsets,
            matrix.columns,
            matrix.values,
            label,
            sizes,
            damped_inverse_diagonal,
        ],
        outputs=[rows, cols, values],
        device=device,
    )
    return _synced(
        od.array.csr_from_triplets(n, n_aggregates, rows, cols, values, prune_numerical_zeros=True)
    )


def _multigrid_galerkin(
    matrix: odt.BsrMatrix[wp.float64], prolongator: odt.BsrMatrix[wp.float64]
) -> odt.BsrMatrix[wp.float64]:
    """
    Form the coarse operator ``P^T A P``, with its exactly-zero entries pruned.

    Two triplet assemblies and no sparse product: ``A P`` from its ``A_ij P_jl`` products
    (``kernels/algorithms/multigrid.product_triplets``), then ``P^T (A P)`` from each fine row's
    outer product of its two sparse rows (``galerkin_triplets``), so no transpose is multiplied.

    Pruning is part of the algorithm, not tidying: an explicit zero at ``(i, c)`` would make coarse
    column ``c`` see fine row ``i``, so the next level's product would inherit every aggregate
    reachable from it and operator complexity would blow up. Zero-valued triplets are skipped and
    an entry whose triplets cancel exactly is dropped -- two summation orders can disagree on that,
    for a handful of entries per million.
    """
    n = int(matrix.nrow)
    n_coarse = int(prolongator.ncol)
    p = prolongator
    # A level that coarsens badly goes nearly dense, and there the per-row products are quadratic
    # in the row length where the product itself is not; past the budget a product takes
    # ``bsr_mm`` instead, whose memory follows the output.
    budget = min(_MULTIGRID_TRIPLET_FACTOR * (int(matrix.nnz) + n), _MULTIGRID_TRIPLET_CAP)
    product = _row_triplet_assembly(
        matrix.device,
        n,
        n_coarse,
        kernel_mg.product_triplet_counts,
        [matrix.offsets, matrix.columns, p.offsets],
        kernel_mg.product_triplets,
        [matrix.offsets, matrix.columns, matrix.values, p.offsets, p.columns, p.values],
        budget=budget,
    )
    if product is None:
        product = odt.bsr_mm(matrix, p)
    galerkin = _row_triplet_assembly(
        matrix.device,
        n,
        n_coarse,
        kernel_mg.galerkin_triplet_counts,
        [p.offsets, product.offsets],
        kernel_mg.galerkin_triplets,
        [p.offsets, p.columns, p.values, product.offsets, product.columns, product.values],
        budget=budget,
        out_rows=n_coarse,
    )
    if galerkin is None:
        # ``bsr_mm``'s structural zeros are what ``bsr_compress`` drops here.
        galerkin = _synced(
            odt.bsr_compress(
                odt.bsr_mm(od.array.csr_transpose(p), product), prune_numerical_zeros=True
            )
        )
    return galerkin


def _row_triplet_assembly(
    device: wp.DeviceLike,
    n: int,
    n_cols: int,
    count_kernel: odt.Kernel,
    count_inputs: Sequence[object],
    triplet_kernel: odt.Kernel,
    triplet_inputs: Sequence[object],
    *,
    budget: int,
    out_rows: int | None = None,
) -> odt.BsrMatrix[wp.float64] | None:
    """
    Sum the triplets ``n`` fine rows emit into a pruned, ``nnz``-synced CSR.

    Each row's thread counts its triplets (``count_kernel`` writes ``counts[i + 1]`` and a leading
    zero, in ``int64``), the counts are scanned in place into starts, and ``triplet_kernel`` writes
    row ``i``'s triplets from ``starts[i]``. The result has ``out_rows`` rows (``n`` by default)
    and ``n_cols`` columns. ``None``, with nothing allocated past the counts, when the triplets
    would number more than ``budget``.
    """
    # Counted and scanned in 64 bits: the budget test must see the true total, which an ``int32``
    # scan wraps on a dense level.
    starts = _launch.empty(n + 1, dtype=wp.int64, device=device)
    _launch.launch(count_kernel, dim=n, inputs=count_inputs, outputs=[starts], device=device)
    _launch.array_scan(starts, starts, inclusive=True)
    # The total sizes the triplet buffers, and decides the fallback.
    total = int(read_scalar(starts, n))
    if total > budget:
        return None
    rows = _launch.empty(total, dtype=wp.int32, device=device)
    cols = _launch.empty(total, dtype=wp.int32, device=device)
    values = _launch.empty(total, dtype=wp.float64, device=device)
    _launch.launch(
        triplet_kernel,
        dim=n,
        inputs=[*triplet_inputs, starts],
        outputs=[rows, cols, values],
        device=device,
    )
    n_out = n if out_rows is None else out_rows
    return _synced(
        od.array.csr_from_triplets(n_out, n_cols, rows, cols, values, prune_numerical_zeros=True)
    )


def _synced(matrix: odt.BsrMatrix[wp.float64]) -> odt.BsrMatrix[wp.float64]:
    """
    ``matrix`` with its ``nnz`` field repaired to the stored count.

    A triplet assembly leaves ``nnz`` at the triplet count, and ``csr_transpose``, ``bsr_mm`` and
    the V-cycle's launches read the field rather than the offsets, so every hierarchy matrix pays
    the one readback here.
    """
    matrix.nnz_sync()
    return matrix


def _multigrid_dense_inverse(matrix: odt.BsrMatrix[wp.float64]) -> odt.ArrayNd | None:
    """
    Pseudo-inverse of the coarsest operator as a dense ``(n, n)`` device array, or ``None``.

    ``None`` when the level is too large to factor, which is what makes the caller fall back to the
    Jacobi preconditioner rather than ship a cycle whose coarse solve is a guess. Rows whose
    diagonal is zero are dropped before the factorization and left as zero in the result: a positive
    semi-definite operator with ``A_ii == 0`` has an identically zero row and column, so those
    unknowns are already solved, and skipping them is what keeps the factorization small when
    coarsening stalls on isolated rows.
    """
    n = int(matrix.nrow)
    nnz = int(matrix.nnz_sync())
    offsets = cast("npt.NDArray[np.int32]", matrix.offsets.numpy())[: n + 1]
    columns = cast("npt.NDArray[np.int32]", matrix.columns.numpy())[:nnz]
    values = matrix.values.numpy()[:nnz].astype(np.float64).reshape(nnz)
    dense = np.zeros((n, n), dtype=np.float64)
    # A triplet build coalesces duplicates, so the CSR has one entry per position and a plain
    # scatter is exact -- ``np.add.at`` would be the same answer several times slower.
    dense[np.repeat(np.arange(n), np.diff(offsets)), columns] = values
    active = np.flatnonzero(dense.diagonal() != 0.0)
    if active.size > _MULTIGRID_MAX_DENSE:
        return None
    inverse = np.zeros((n, n), dtype=np.float64)
    if active.size:
        inverse[np.ix_(active, active)] = _symmetric_inverse(dense[np.ix_(active, active)])
    return _launch.array(inverse, dtype=wp.float64, device=matrix.device)


# Largest infinity-norm condition number at which ``_symmetric_inverse`` trusts a plain inverse.
# ``kappa_2 <= n kappa_inf``, so at ``n <= _MULTIGRID_MAX_DENSE`` this keeps ``kappa_2`` under
# ``1e12``, the cut ``pinv(rcond=1e-12)`` would make: below it the pseudo-inverse truncates nothing
# and the two agree to rounding.
_DENSE_INVERSE_MAX_CONDITION = 1e9


def _symmetric_inverse(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Inverse of a symmetric matrix, or its pseudo-inverse when it is numerically singular.

    The coarsest levels this factors are non-singular in practice, where an LU inverse is the same
    answer as ``pinv(hermitian=True)`` (an eigendecomposition) to rounding at a fraction of its
    cost; a singular or ill-conditioned level -- a semi-definite operator whose constants survived
    coarsening -- falls back to the pseudo-inverse. Both run with the calling thread's BLAS
    limited to one thread: at these sizes a threaded factorization is slower, and its worker
    threads keep spinning afterwards, slowing the launches that follow.
    """
    with _single_threaded_blas():
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            inverse = None
        if inverse is not None and not (
            np.linalg.norm(block, np.inf) * np.linalg.norm(inverse, np.inf)
            <= _DENSE_INVERSE_MAX_CONDITION
        ):
            inverse = None
        if inverse is None:
            # ``hermitian=True`` factors through ``eigh`` rather than a general SVD, which the
            # operator being symmetric makes exact and considerably cheaper.
            inverse = np.linalg.pinv(block, rcond=1e-12, hermitian=True).astype(
                np.float64, copy=False
            )
    # Neither factorization returns an exactly symmetric matrix, and the V-cycle must stay a
    # symmetric preconditioner.
    return 0.5 * (inverse + inverse.T)


@contextlib.contextmanager
def _single_threaded_blas() -> Generator[None]:
    """
    Limit OpenBLAS to one thread for the calling thread, inside a ``with`` block.

    Uses OpenBLAS's thread-local ``openblas_set_num_threads_local``, so no other thread's BLAS
    calls are affected. A no-op when NumPy is not linked against an OpenBLAS that exports it.
    """
    setter = _openblas_thread_setter()
    if setter is None:
        yield
        return
    previous = setter(1)
    try:
        yield
    finally:
        setter(previous)


@functools.cache
def _openblas_thread_setter() -> Callable[[int], int] | None:
    """``openblas_set_num_threads_local`` from the OpenBLAS NumPy ships with, or ``None``."""
    root = os.path.dirname(os.path.abspath(np.__file__))
    candidates = [
        *glob.glob(os.path.join(root + ".libs", "*openblas*")),
        *glob.glob(os.path.join(root, ".dylibs", "*openblas*")),
    ]
    for path in candidates:
        try:
            setter = ctypes.CDLL(path).openblas_set_num_threads_local
        except (OSError, AttributeError):
            continue
        setter.restype = ctypes.c_int
        setter.argtypes = [ctypes.c_int]
        return setter
    return None


class _MultigridCycle:
    """
    One V-cycle of a smoothed-aggregation hierarchy, over ``n_columns`` blocks of a flat vector.

    Batched over the columns: *every* pass, the sparse mat-vecs included, is one launch over the
    whole flat vector. That is not a tidiness choice -- a cycle at these sizes is launch-bound, and
    ``warp.sparse.bsr_mv`` takes one vector, so routing several right-hand sides through it would
    cost one launch per column per mat-vec. A hand-written batched CSR mat-vec
    (``kernels/algorithms/multigrid.csr_matvec``) gives the same answer in about half the launches.

    The vectors of the *top* level are the caller's own, at the caller's column pitch, so the cycle
    adds no copy at its boundary; every level below allocates its own at pitch ``n``. Every buffer
    is built here and the top level's is bound by ``apply``, so ``apply`` issues launches and
    nothing else -- which is what lets the whole preconditioned iteration be captured as one graph.
    """

    def __init__(
        self,
        levels: list[_MultigridLevel],
        coarse_inverse: wp.array[wp.float64],
        *,
        n_columns: int,
        stride: int,
        sweeps: int = _MULTIGRID_SWEEPS,
    ) -> None:
        self._levels = levels
        self._coarse_inverse = coarse_inverse
        self._device = levels[0].operator.device
        self._n_columns = n_columns
        self._sweeps = max(1, int(sweeps))

        for depth, level in enumerate(levels):
            top = depth == 0
            level.stride = stride if top else level.n
            level.dim = n_columns * level.stride
            level.matvec_dim = n_columns * level.n
            level.ax = _launch.zeros(level.dim, dtype=wp.float64, device=self._device)
            level.r = _launch.zeros(level.dim, dtype=wp.float64, device=self._device)
            if top:
                continue
            level.b = _launch.zeros(level.dim, dtype=wp.float64, device=self._device)
            level.x = _launch.zeros(level.dim, dtype=wp.float64, device=self._device)
        for depth, level in enumerate(levels[:-1]):
            child = levels[depth + 1]
            level.restrict_dim = n_columns * child.n
            level.prolong_dim = n_columns * level.n

    @property
    def grid(self) -> list[int]:
        """Rows per level, coarsest last -- the hierarchy's shape."""
        return [level.n for level in self._levels]

    def apply(self, source: wp.array[wp.float64], destination: wp.array[wp.float64]) -> None:
        """One V-cycle: ``destination = M^-1 source``, both at the caller's column pitch."""
        top = self._levels[0]
        top.b, top.x = source, destination
        self._cycle(0)

    def _matvec(
        self,
        matrix: odt.BsrMatrix[wp.float64] | None,
        dim: int,
        n_rows: int,
        x_stride: int,
        y_stride: int,
        x: wp.array[wp.float64],
        y: wp.array[wp.float64],
        accumulate: bool = False,
    ) -> None:
        assert matrix is not None
        _launch.launch(
            kernel_mg.csr_matvec,
            dim=dim,
            inputs=[
                wp.int32(n_rows),
                wp.int32(x_stride),
                wp.int32(y_stride),
                wp.int32(1 if accumulate else 0),
                wp.float64(1.0),
                matrix.offsets,
                matrix.columns,
                matrix.values,
                x,
                y,
            ],
            device=self._device,
        )

    def _smooth(self, level: _MultigridLevel, sweeps: int) -> None:
        for _ in range(sweeps):
            self._matvec(
                level.operator,
                level.matvec_dim,
                level.n,
                level.stride,
                level.stride,
                level.x,
                level.ax,
            )
            _launch.launch(
                kernel_mg.jacobi_sweep,
                dim=level.dim,
                inputs=[
                    wp.int32(level.n),
                    wp.int32(level.stride),
                    level.inverse_diagonal,
                    wp.float64(1.0),  # the damping is in the diagonal already
                    level.b,
                    level.ax,
                    level.x,
                ],
                device=self._device,
            )

    def _cycle(self, depth: int) -> None:
        level = self._levels[depth]
        if level.prolongator is None:
            _launch.launch(
                kernel_mg.dense_solve,
                dim=level.dim,
                inputs=[
                    wp.int32(level.n),
                    wp.int32(level.stride),
                    self._coarse_inverse,
                    level.b,
                    level.x,
                ],
                device=self._device,
            )
            return
        # The first sweep from a zero initial guess is a *write*, so nothing has to be zeroed and
        # that sweep costs no mat-vec.
        _launch.launch(
            kernel_cg.scaled_diagonal_apply,
            dim=level.dim,
            inputs=[
                wp.int32(level.n),
                wp.int32(level.stride),
                level.inverse_diagonal,
                wp.float64(1.0),  # the damping is in the diagonal already
                level.b,
                level.x,
            ],
            device=self._device,
        )
        self._smooth(level, self._sweeps - 1)
        self._matvec(
            level.operator, level.matvec_dim, level.n, level.stride, level.stride, level.x, level.ax
        )
        _launch.launch(
            kernel_mg.residual,
            dim=level.dim,
            inputs=[wp.int32(level.n), wp.int32(level.stride), level.b, level.ax, level.r],
            device=self._device,
        )
        child = self._levels[depth + 1]
        self._matvec(
            level.restrictor,
            level.restrict_dim,
            child.n,
            level.stride,
            child.stride,
            level.r,
            child.b,
        )
        self._cycle(depth + 1)
        # Prolong and correct in one launch: ``accumulate`` adds into the fine iterate.
        self._matvec(
            level.prolongator,
            level.prolong_dim,
            level.n,
            child.stride,
            level.stride,
            child.x,
            level.x,
            accumulate=True,
        )
        self._smooth(level, self._sweeps)


def _warn_if_not_converged(result: tuple[int, float, float], iteration_cap: int, name: str) -> None:
    """
    Warn when conjugate gradient stopped because it ran out of iterations, not because it converged.

    Skipped under ``check_every=0``, where the three values are 1-element *device* arrays and
    inspecting them would cost the host sync that setting exists to avoid.

    That early return is also why the message says "residual norm" and not "squared residual": the
    squared form is what the *device* path carries, and this never reaches it. Every value this
    formats has already had its square root taken -- by ``warp.optim.linear.cg`` on the one-column
    path, and by ``_BatchedCg`` on the batched ones.
    """
    iterations, residual, atol = result
    if isinstance(iterations, wp.array):
        return
    if int(iterations) >= iteration_cap and float(residual) > float(atol):
        warnings.warn(
            f"{name}: conjugate gradient hit its {iteration_cap}-iteration cap with residual norm "
            f"{float(residual):.3e} against tolerance {float(atol):.3e}; the result is "
            f"the last iterate, not a solution. The operator is likely ill-conditioned or "
            f"singular — consider ordito.laplacian.robust_laplacian, or a smaller step.",
            stacklevel=3,
        )


def _supported_check_every(check_every: int) -> int:
    """
    Substitute a host-side cadence for ``check_every=0`` where the device cannot test on device.

    ``warp.optim.linear`` implements ``check_every=0`` with ``wp.capture_while``; without
    conditional CUDA graphs it has nowhere to put the test and runs every solve to ``maxiter``
    instead. Falling back to [`CG_CHECK_EVERY_FALLBACK`][ordito.linalg.CG_CHECK_EVERY_FALLBACK]
    keeps the request's *meaning* (stop when converged) at the cost of its return type.
    """
    if check_every == 0 and not wp.is_conditional_graph_supported():
        return CG_CHECK_EVERY_FALLBACK
    return check_every


def _bsr_over(
    n_rows: int,
    n_cols: int,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[odt.Block, Any],
    nnz: int,
    *,
    row_counts: wp.array[wp.int32] | None = None,
) -> odt.BsrMatrix[odt.Block]:
    """
    Build a ``BsrMatrix`` over arrays that already hold its storage, allocating nothing.

    ``values``' dtype is the block type, and ``row_counts`` is ``None`` for the compact topology
    every caller here describes. The typed class is built directly rather than through
    ``wps.bsr_zeros``, which allocates three placeholder arrays and zeroes the offsets only for
    every one of them to be replaced; ``notify_nnz_changed`` is still Warp's documented entry
    point for storage assigned from outside ``warp.sparse``, and records ``nnz``.
    """
    matrix = odt.bsr_matrix_t(values.dtype)()
    matrix.nrow = n_rows
    matrix.ncol = n_cols
    matrix.offsets = offsets
    matrix.columns = columns
    matrix.values = values
    matrix.row_counts = row_counts
    matrix.notify_nnz_changed(nnz=nnz)
    return matrix
