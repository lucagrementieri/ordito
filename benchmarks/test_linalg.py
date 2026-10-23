"""
Benchmarks for ``ordito.linalg``: the conjugate-gradient solvers behind every smoother and map.

Axis: **quality**. This module has no size story worth telling -- ``nnz`` sets the cost of one
mat-vec and that is arithmetic -- but it has a conditioning story, and conditioning is the whole
reason an iterative solver is unpredictable. ``saddle`` and ``saddle_graded`` are the same mesh with
the same connectivity, the same ``nnz`` and the same right-hand side; only the spacing along one
axis differs, pushing the worst triangle aspect ratio from 1.6 to 4 719. Everything these groups
measure beyond a fixed per-iteration cost is the iteration count that difference buys.

Three knobs are swept on top of it, each isolating a different lever:

* **fixed fraction** -- ``min_quad_with_fixed`` extracts the free-free block, whose ``nnz`` goes as
  ``(n_free / n)^2``, and a larger fixed set is also a better-conditioned system. Pinning 1% of the
  vertices against 50% moves both at once, which is what a caller actually chooses between. The
  returned free-degree-of-freedom count is asserted on, so a run that accidentally pins everything
  -- and therefore skips the solve entirely -- fails rather than reporting a suspiciously good
  time. It is *not* an iteration count: ``min_quad_with_fixed`` returns
  ``(solution, free_map, n_free)`` and never surfaces one.
* **check_every** -- how often ``solve_spd_columns`` tests the residual. ``1`` checks every
  iteration with a host sync each time; ``0`` (now the default) tests every iteration on device via
  ``wp.capture_while``, with no readback at all. Skipping checks trades syncs for possibly-wasted
  iterations, and which side wins is a measurement, not a derivation -- read this sweep together
  with the mesh pair, because on a well-conditioned system the syncs dominate and on a badly
  conditioned one they should not. It is this group that set the default: against a fixed check
  interval the on-device row is markedly cheaper on both meshes.
* **repeat count** -- ``spd_column_solver`` preallocates its temporaries and batch layout for reuse,
  which is what ``parametrization.arap`` builds outside its iteration loop. One solve against fifty
  is the amortization question: if ``x50`` lands near 50x ``once``, the preallocation is not earning
  its API surface.

Everything runs in **float64**, the operator dtype these entry points require. ``ordito-cpu`` is
timed as well as ``ordito-cuda``: ``warp.optim.linear.cg`` converges on the Warp CPU backend,
verified against ``numpy.linalg.solve`` on a small SPD system.

References
----------
**pymeshlab** is the only reference this module has, and it covers exactly one of the three groups:
``compute_scalar_by_scalar_harmonic_field_per_vertex`` (MeshLab's Generate Scalar Harmonic Field) is
a Dirichlet-constrained solve of the same cotangent system ``min_quad_with_fixed`` solves, so the
two answer the same question by opposite means -- MeshLab factors the free-free block directly,
ordito runs batched CG on it.

That makes it worth more than a timing row: it is the **conditioning control**. It is flat across
the axis meshes where ordito spreads several-fold at 1% pinned, so ordito wins on the
well-conditioned mesh and loses on the graded one, and the entire spread is its CG iteration count
rather than anything intrinsic about the problem: a direct factorization of the identical system
does not care. It is the same observation the potpourri3d rows make in
[`test_heat_distance.py`](test_heat_distance.py) and libigl's LDLT makes in
[`test_parametrization.py`](test_parametrization.py).

``assemble_interior_system`` deliberately does **not** route its already-sorted CSR through
``bsr_from_triplets``: that build costs ``q.nnz`` regardless of how much of ``Q`` survives and
leaves ``q_uu.nnz`` at ``q.nnz``, an upper bound well above the true count on a lightly-pinned
system, so every CG mat-vec is dimensioned for the *unreduced* matrix too. Avoiding it is worth
around 2x at 1% pinned and an order of magnitude at 50%, where the free block is smallest and so the
overcounting is worst.

Two limits on it, both structural. MeshLab pins **exactly two vertices** (``point1`` / ``point2``
with scalar values), so it has no fixed-fraction axis at all and appears only in the ``pin1pct``
row, the closer and harder of the two. And it is *only* the harmonic solve: nothing in the reference
group exposes a raw iterative solver, so ``solve_spd_columns`` and ``spd_column_solver_amortized``
remain before/after self-comparisons. scipy's ``cg`` would be a host solver over a scipy matrix
(timing SciPy's sparse layer, not this one), libigl solves only through its own direct
factorizations, and neither trimesh nor open3d has a linear solver at all.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase

# Fraction of vertices pinned as Dirichlet boundary conditions.
_FIXED_FRACTIONS = [0.01, 0.5]

# Residual-check cadence: every iteration (a host sync each) against the on-device test.
_CHECK_EVERY = [1, 0]

# Solve counts for the amortization group.
_REPEATS = [1, 50]

# Right-hand sides solved together, matching the two UV columns the parametrization solvers batch.
_N_RHS = 2

_SEED = 23

_operator_cache: dict[tuple[str, str], odt.BsrMatrix[wp.float64]] = {}
_fixed_cache: dict[tuple[str, str, float], tuple[wp.array[wp.bool], odt.Array2dFloat64]] = {}
_rhs_cache: dict[tuple[str, str], wp.array[wp.float64]] = {}
_heat_cache: dict[tuple[str, str], tuple[odt.BsrMatrix[wp.float64], wp.array[wp.float64]]] = {}


def _operator(bench_case: BenchCase) -> odt.BsrMatrix[wp.float64]:
    """Float64 cotangent stiffness -- the *input*; its assembly is timed in test_laplacian."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_cache:
        _operator_cache[key] = od.laplacian.cotmatrix(
            bench_case.vertices_wp, bench_case.faces_wp, dtype=wp.float64
        )
    return _operator_cache[key]


def _fixed_mask_np(bench_case: BenchCase, fraction: float) -> np.ndarray:
    """
    Draw the pinned-vertex mask on the host, so a ``cpu_bound`` reference row gets the same one.

    Split out of ``_fixed`` rather than duplicated: the igl row solves the identical problem and a
    second draw from the same seed would still be a second place for the seed to drift.
    """
    rng = np.random.default_rng(_SEED)
    n = bench_case.n_vertices
    mask_np = np.zeros(n, dtype=bool)
    mask_np[rng.choice(n, size=max(1, int(n * fraction)), replace=False)] = True
    return mask_np


def _fixed(bench_case: BenchCase, fraction: float) -> tuple[wp.array[wp.bool], odt.Array2dFloat64]:
    """``(fixed_mask, fixed_values)`` pinning ``fraction`` of the vertices, at a fixed seed."""
    key = (bench_case.mesh_name, str(bench_case.device), fraction)
    if key not in _fixed_cache:
        mask_np = _fixed_mask_np(bench_case, fraction)
        values_np = np.tile(bench_case.vertices_np[:, 2], (_N_RHS, 1))
        _fixed_cache[key] = (
            wp.array(mask_np, dtype=wp.bool, device=bench_case.device),
            odt.as_array2d(
                wp.array(
                    np.ascontiguousarray(values_np), dtype=wp.float64, device=bench_case.device
                ),
                wp.float64,
            ),
        )
    return _fixed_cache[key]


def _rhs(bench_case: BenchCase) -> wp.array[wp.float64]:
    """``(n_rhs, n)`` float64 right-hand sides taken from the vertex coordinates."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _rhs_cache:
        rhs_np = np.ascontiguousarray(bench_case.vertices_np[:, :_N_RHS].T)
        _rhs_cache[key] = wp.array(rhs_np, dtype=wp.float64, device=bench_case.device)
    return _rhs_cache[key]


def _heat_system(bench_case: BenchCase) -> odt.BsrMatrix[wp.float64]:
    """Return the heat system ``M - tL``: a definite operator, as the solve entry points take."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _heat_cache:
        system = od.heat.heat_operators(bench_case.vertices_wp, bench_case.faces_wp)[0]
        mass = od.laplacian.mass_matrix_entries(
            bench_case.vertices_wp, bench_case.faces_wp, dtype=wp.float64
        )
        _heat_cache[key] = (system, mass)
    return _heat_cache[key][0]


def _heat_rhs(bench_case: BenchCase) -> wp.array[wp.float64]:
    """Return the lumped masses, whose heat solve is the constant field."""
    _heat_system(bench_case)
    return _heat_cache[(bench_case.mesh_name, str(bench_case.device))][1]


def _source_rhs(bench_case: BenchCase) -> wp.array[wp.float64]:
    """Return one vertex's indicator: the heat method's diffusion right-hand side."""
    rhs_np = np.zeros(bench_case.n_vertices)
    rhs_np[0] = 1.0
    return wp.array(rhs_np, dtype=wp.float64, device=bench_case.device)


def _interior_laplacian(
    bench_case: BenchCase, fraction: float
) -> tuple[odt.BsrMatrix[wp.float64], int]:
    """Return ``-L``'s free-free block for ``fraction`` pinned, and its row count."""
    fixed_mask, _ = _fixed(bench_case, fraction)
    free_map, n_free = od.linalg.free_partition(fixed_mask)
    none = odt.as_array2d(
        wp.zeros((0, bench_case.n_vertices), dtype=wp.float64, device=bench_case.device), wp.float64
    )
    l_ff, _ = od.linalg.assemble_interior_system(
        _operator(bench_case), fixed_mask, free_map, none, n_free, scale=-1.0
    )
    return l_ff, n_free


def _harmonic_endpoints_pml(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Return the two extremal-``x`` vertices, the Dirichlet pair MeshLab's harmonic field takes."""
    vertices_np = bench_case.vertices_np
    order = np.argsort(vertices_np[:, 0])
    return vertices_np[order[0]], vertices_np[order[-1]]


@pytest.mark.benchmark(group="min_quad_with_fixed")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab", "igl")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_min_quad_with_fixed(bench_case: BenchCase, fixed_fraction: float) -> None:
    """
    Dirichlet-constrained quadratic minimization: submatrix extraction plus a batched CG solve.

    Four rows per table. Across the mesh pair the gap is conditioning; across the pin pair it is
    both a smaller free block and a better-conditioned one. The interesting cell is
    ``saddle_graded`` at 1% pinned -- the worst-conditioned, largest free system.

    The pymeshlab row is the control for exactly that cell: its direct factorization of the same
    system is flat across the mesh pair, so whatever spread ordito shows is the CG iteration count
    and not the problem.

    **libigl binds the function this one is named after**, and it is the closer control of the two:
    ``igl.min_quad_with_fixed(A, B, known, Y, Aeq, Beq, pd)`` minimizes ``0.5 x' A x + x' B`` under
    ``x[known] = Y``, which at ``B = 0`` and no equality constraints is exactly this problem -- so
    unlike MeshLab it takes the *same* pinned fraction and follows the whole ``fixed_fraction``
    axis. The named transform is one sign: ``igl.cotmatrix`` is negative semi-definite, so ``A`` is
    ``-L``.

    It is a direct factorization too, so read the two reference rows together: both should be flat
    where ordito's CG is not, and igl's should additionally be flat in ``fixed_fraction`` where
    ordito's is not -- a smaller free block is less work for CG and roughly the same amount of
    fill-reducing ordering for a factorization.
    """
    if bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        operator_igl = -igl.cotmatrix(vertices_np, np.ascontiguousarray(faces_np, dtype=np.int64))
        n_vertices = bench_case.n_vertices
        known_igl = np.ascontiguousarray(
            np.flatnonzero(_fixed_mask_np(bench_case, fixed_fraction)).astype(np.int64)
        )
        # The same pinned values ordito's row is given: the z coordinate at the pinned vertices.
        values_igl = np.ascontiguousarray(vertices_np[known_igl, 2]).reshape(-1, 1)
        zeros_igl = np.zeros((n_vertices, 1))
        equality_igl = sp.csr_matrix((0, n_vertices))
        rhs_igl = np.zeros((0, 1))
        solution_igl = bench_case.run(
            # The binding converts any scipy sparse matrix; its stub spells ``csc_matrix``.
            lambda: igl.min_quad_with_fixed(
                operator_igl,
                zeros_igl,
                known_igl,
                values_igl,
                equality_igl,  # pyright: ignore[reportArgumentType]
                rhs_igl,
                True,
            )
        )
        assert np.asarray(solution_igl).shape[0] == n_vertices
        return
    if bench_case.kind == "pymeshlab":
        if fixed_fraction != min(_FIXED_FRACTIONS):
            pytest.skip("MeshLab's harmonic field pins exactly two vertices: no fraction axis")
        # Geometry-preserving with ``colorize=False`` (it writes only the vertex scalar), so the
        # shared MeshSet is sound and its build stays out of the row.
        meshset_pml = bench_case.meshset_pml
        point1, point2 = _harmonic_endpoints_pml(bench_case)
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_scalar_harmonic_field_per_vertex(
                point1=point1, point2=point2, colorize=False
            )
        )
        assert meshset_pml.current_mesh().vertex_scalar_array().size == bench_case.n_vertices
        return

    operator = _operator(bench_case)
    fixed_mask, fixed_values = _fixed(bench_case, fixed_fraction)
    solution, _free_map, n_free = bench_case.run(
        lambda: od.linalg.min_quad_with_fixed(operator, fixed_mask, fixed_values)
    )
    assert solution.shape[0] == _N_RHS
    assert n_free > 0, "a fully-pinned system returns without solving, so there is nothing to time"


@pytest.mark.benchmark(group="min_quad_with_fixed_precompute")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_min_quad_with_fixed_precompute(bench_case: BenchCase, fixed_fraction: float) -> None:
    """
    The reduced system's extraction and factorization, for a caller that solves it repeatedly.

    Read against ``min_quad_with_fixed`` at the same pin fraction: the factorization does not care
    about conditioning, so this row should be flat across the mesh pair where the CG solve is not.
    """
    operator = _operator(bench_case)
    fixed_mask, _ = _fixed(bench_case, fixed_fraction)
    data = bench_case.run(
        lambda: od.linalg.min_quad_with_fixed_precompute(
            operator, fixed_mask, coordinates=bench_case.vertices_wp
        ),
        rounds=3,
    )
    assert data.n_free > 0


@pytest.mark.benchmark(group="free_partition")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_free_partition(bench_case: BenchCase, fixed_fraction: float) -> None:
    """The free degrees of freedom's compact remap: one scan and its total's readback."""
    fixed_mask, _ = _fixed(bench_case, fixed_fraction)
    free_map, n_free = bench_case.run(lambda: od.linalg.free_partition(fixed_mask))
    assert free_map.shape[0] == bench_case.n_vertices
    assert 0 < n_free < bench_case.n_vertices


@pytest.mark.benchmark(group="free_coordinates")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_free_coordinates(bench_case: BenchCase, fixed_fraction: float) -> None:
    """The free rows' positions, gathered for a reduced factorization's ordering."""
    fixed_mask, _ = _fixed(bench_case, fixed_fraction)
    free_map, n_free = od.linalg.free_partition(fixed_mask)
    coordinates = bench_case.run(
        lambda: od.linalg.free_coordinates(bench_case.vertices_wp, fixed_mask, free_map, n_free)
    )
    assert coordinates is not None
    assert coordinates.shape[0] == n_free


@pytest.mark.benchmark(group="assemble_interior_system")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_assemble_interior_system(bench_case: BenchCase, fixed_fraction: float) -> None:
    """
    The free-free block and its constant right-hand side, the extraction ahead of every solve.

    The pin pair is the axis: the free block's ``nnz`` goes as ``(n_free / n)^2``. Not timed
    against NumPy, whose dense partition is the test oracle and not an implementation of it.
    """
    operator = _operator(bench_case)
    fixed_mask, fixed_values = _fixed(bench_case, fixed_fraction)
    free_map, n_free = od.linalg.free_partition(fixed_mask)
    q_uu, rhs = bench_case.run(
        lambda: od.linalg.assemble_interior_system(
            operator, fixed_mask, free_map, fixed_values, n_free
        )
    )
    assert q_uu.nrow == n_free
    assert rhs.shape == (_N_RHS, n_free)


@pytest.mark.benchmark(group="partitioned_interior_system")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_partitioned_interior_system(bench_case: BenchCase, fixed_fraction: float) -> None:
    """
    ``free_partition`` and ``assemble_interior_system`` in one call.

    Read against the two rows above at the same point: their sum is the two-step form's cost.
    """
    operator = _operator(bench_case)
    fixed_mask, fixed_values = _fixed(bench_case, fixed_fraction)
    q_uu, rhs, _free_map, n_free = bench_case.run(
        lambda: od.linalg.partitioned_interior_system(operator, fixed_mask, fixed_values)
    )
    assert q_uu.nrow == n_free
    assert rhs.shape == (_N_RHS, n_free)


@pytest.mark.benchmark(group="solve_spd")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_solve_spd(bench_case: BenchCase) -> None:
    """
    One Jacobi-preconditioned CG solve of the heat system against the lumped masses.

    The single-right-hand-side form ``solve_spd_columns`` batches; the heat system's mass term
    keeps it well conditioned, so the quality pair reads the per-iteration cost more than the count.
    """
    system, rhs = _heat_system(bench_case), _heat_rhs(bench_case)
    solution = wp.zeros_like(rhs)

    def run() -> tuple[int, float, float]:
        solution.zero_()
        return od.linalg.solve_spd(system, rhs, solution)

    bench_case.run(run)
    assert solution.shape[0] == bench_case.n_vertices
    assert np.isfinite(solution.numpy()).all()


@pytest.mark.benchmark(group="jacobi_preconditioner")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_jacobi_preconditioner(bench_case: BenchCase) -> None:
    """Building the inverse diagonal: the setup every Jacobi-preconditioned solve pays first."""
    system = _heat_system(bench_case)
    preconditioner = bench_case.run(lambda: od.linalg.jacobi_preconditioner(system))
    assert preconditioner.shape[0] == bench_case.n_vertices


@pytest.mark.benchmark(group="solve_spd_columns")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("check_every", _CHECK_EVERY, ids=["every", "ondevice"])
def test_solve_spd_columns(bench_case: BenchCase, check_every: int) -> None:
    """Residual-check cadence: host syncs traded against possibly-wasted iterations."""
    operator, rhs = _operator(bench_case), _rhs(bench_case)
    solution = wp.zeros_like(rhs)
    solution_2d = odt.as_array2d(solution, wp.float64)
    rhs_2d = odt.as_array2d(rhs, wp.float64)

    def run() -> tuple[int, float, float]:
        solution.zero_()
        return od.linalg.solve_spd_columns(operator, rhs_2d, solution_2d, check_every=check_every)

    # ``check_every=0`` reports its iteration count in a device array (the residual test runs
    # inside ``wp.capture_while``), while a positive cadence reports a host int -- so only the
    # solution is asserted on here, and the iteration count is asserted in the group above.
    assert bench_case.run(run) is not None
    assert solution.shape[0] == _N_RHS


@pytest.mark.benchmark(group="multigrid_preconditioner")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_multigrid_preconditioner(bench_case: BenchCase) -> None:
    """
    Hierarchy *setup* only: the cost that decides whether a call site should ask for the V-cycle.

    The solve-side win is not timed here -- it lands in the ``smooth_region`` group, the one
    operator class in the package where it pays. What this row
    measures is the other half of that trade, and why the switch is per call site and not a default:
    building the hierarchy is one aggregation, one power iteration, a transpose and two sparse
    products per level, and at these sizes every one of those is a *fixed* per-call cost rather
    than a function of the operator -- which is why ``smoothing.smooth_region_fixed_rim`` and
    ``parametrization.harmonic`` at ``k=1`` decline it: their whole solve is shorter than this row.

    Read against the ``solve_spd_columns`` group above, whose two rows are the *same* operator's
    solve. A setup that grew with the mesh rather than sitting near-flat would change the decision
    at every call site, so the shape of this row across the quality axis is the thing to watch.

    **This group gets worse as the aggregation gets better, so never read a loss here alone.** It
    times the setup and nothing else, and ``linalg._MULTIGRID_THETA`` trades setup for iterations:
    raising it removes weak edges, which makes aggregates smaller and can add a level, and a level
    costs a fixed amount whatever its size. At the shipped theta against zero, the setup is dearer
    by one level while the same operator's ``smooth_region`` solve is cheaper by more than that, so
    the extra level is bought twice over. That constant's comment carries the setup-plus-solve
    reasoning; a round that reads this row's regression without it will re-propose lowering theta,
    which is a total loss.
    """
    operator = _operator(bench_case)
    preconditioner = bench_case.run(lambda: od.linalg.multigrid_preconditioner(operator))
    assert preconditioner.shape[0] == int(operator.nrow)


@pytest.mark.benchmark(group="spd_column_solver_amortized")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("repeats", _REPEATS, ids=["once", "x50"])
def test_spd_column_solver_amortized(bench_case: BenchCase, repeats: int) -> None:
    """
    One solve against fifty over the same preallocated state.

    Each call warm-starts from whatever ``solution`` already holds, which is exactly how
    ``parametrization.arap`` drives it -- so the later solves in the ``x50`` row are converging from
    a good guess and should be much cheaper than the first. That is the effect being measured; a
    ``x50`` row near 50x ``once`` would mean the reuse is buying nothing.

    Unlike ``arap``, this group re-solves the *same* right-hand side, so calls 2..50 are essentially
    no-ops -- which makes it the sharpest probe in the suite of per-call solver overhead. It is what
    caught the one regime where the default ``check_every=0`` loses: the conditional-graph loop
    costs a fixed fraction of a millisecond a call whatever it does, so this row runs twice as slow
    as at ``check_every=10`` while every other solver group got faster. See the ``check_every``
    notes on
    ``ordito.linalg.solve_spd_columns``.
    """
    operator, rhs = _operator(bench_case), _rhs(bench_case)
    solution = wp.zeros_like(rhs)
    solver = od.linalg.spd_column_solver(
        operator, odt.as_array2d(rhs, wp.float64), odt.as_array2d(solution, wp.float64)
    )

    def run() -> None:
        solution.zero_()
        for _ in range(repeats):
            solver()

    bench_case.run(run, rounds=3)
    assert solution.shape[0] == _N_RHS


@pytest.mark.benchmark(group="chebyshev_preconditioner")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_chebyshev_preconditioner(bench_case: BenchCase) -> None:
    """
    Building the Jacobi-Chebyshev polynomial for the Poisson operator ``-L``.

    The polynomial's interval is fitted on its first apply, so this row is the construction a
    caller pays per operator, not the fit; the fit lands in the first solve that uses it.
    """
    system = _heat_system(bench_case)
    preconditioner = bench_case.run(lambda: od.linalg.chebyshev_preconditioner(system))
    assert preconditioner.shape[0] == bench_case.n_vertices


@pytest.mark.benchmark(group="operator_factorization_factor")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_operator_factorization_factor(bench_case: BenchCase) -> None:
    """
    ``OperatorFactorization.factor`` on a fresh holder: a whole ``sparse_cholesky`` build.

    A factored holder returns at once, so every round starts from a new one (built outside the
    timed region). Read against ``sparse_cholesky`` in ``test_cholesky.py``.
    """
    system = _heat_system(bench_case)
    factored = bench_case.run(
        lambda holder: holder.factor(),
        setup=lambda: od.linalg.OperatorFactorization(system, bench_case.vertices_wp),
        rounds=3,
    )
    assert factored


@pytest.mark.benchmark(group="solve_spd_settled")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_solve_spd_settled(bench_case: BenchCase) -> None:
    """
    A settle-ruled heat diffusion from one vertex, at the heat method's own settle constants.

    The quality pair is the axis: the graded mesh is where the settled iterate fails its backward
    error check and the fallback factorization is paid (built per call, as no holder is passed).
    """
    system, rhs = _heat_system(bench_case), _source_rhs(bench_case)
    solution = wp.zeros_like(rhs)

    def run() -> wp.array[wp.int32]:
        solution.zero_()
        return od.linalg.solve_spd_settled(
            system, rhs, solution, check_rounds=16, change_tolerance=1e-6, settle_rounds=192
        )

    rounds = bench_case.run(run, rounds=3)
    assert rounds.shape == (1,)
    assert np.isfinite(solution.numpy()).all()


@pytest.mark.benchmark(group="bsr_with_values")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito")
def test_bsr_with_values(bench_case: BenchCase) -> None:
    """A matrix over another's pattern with new values: host bookkeeping only, no launch."""
    operator = _operator(bench_case)
    values = wp.zeros_like(operator.values)
    matrix = bench_case.run(lambda: od.linalg.bsr_with_values(operator, values))
    assert matrix.offsets is operator.offsets


def _scipy_csr(operator: odt.BsrMatrix[wp.float64], n: int) -> sp.csr_matrix:
    """Read a scalar operator back as a SciPy CSR matrix, for the reference row's input."""
    offsets_np = operator.offsets.numpy()[: n + 1]
    count = int(offsets_np[-1])
    return sp.csr_matrix(
        (operator.values.numpy()[:count], operator.columns.numpy()[:count], offsets_np),
        shape=(n, n),
    )


@pytest.mark.benchmark(group="block_diag")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "sphere_large")
@pytest.mark.benchlibs("ordito", "scipy")
def test_block_diag(bench_case: BenchCase) -> None:
    """
    Three copies of the heat system stacked block-diagonally, across the scale sweep.

    ``block_diag`` keeps its result per set of operators, so the ordito row passes a fresh view of
    the first operator (``bsr_with_values`` over its own arrays, built outside the timed region) to
    time the build, not the cache hit. SciPy's ``block_diag`` of the same three CSR matrices is the
    test oracle and the reference.
    """
    if bench_case.kind == "scipy":
        # The reference's input is the same operator, assembled by ordito and read back.
        device = wp.get_preferred_device()
        system_np = _scipy_csr(
            od.heat.heat_operators(
                wp.array(bench_case.vertices_np, dtype=wp.vec3, device=device),
                wp.array(bench_case.faces_np.ravel(), dtype=wp.int32, device=device),
            )[0],
            bench_case.n_vertices,
        )
        stacked_np = bench_case.run(
            lambda: sp.block_diag((system_np, system_np, system_np), format="csr")
        )
        assert stacked_np.shape[0] == 3 * bench_case.n_vertices
        return
    system = _heat_system(bench_case)
    stacked = bench_case.run(
        lambda first: od.linalg.block_diag((first, system, system)),
        setup=lambda: od.linalg.bsr_with_values(system, system.values),
    )
    assert stacked.nrow == 3 * bench_case.n_vertices


@pytest.mark.benchmark(group="replicated_operator")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito")
def test_replicated_operator(bench_case: BenchCase) -> None:
    """One operator presented as two batched columns: a host wrapper, no copy of the matrix."""
    system = _heat_system(bench_case)
    batched = bench_case.run(lambda: od.linalg.replicated_operator(system, _N_RHS))
    assert batched.shape[0] == _N_RHS * bench_case.n_vertices


@pytest.mark.benchmark(group="squared_laplacian_preconditioner")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_squared_laplacian_preconditioner(bench_case: BenchCase) -> None:
    """
    The squared-Laplacian preconditioner over ``-L``'s interior block at 1% pinned.

    One row-scaling walk, a transpose and the polynomial's interval, all on the device.
    """
    l_ff, n_free = _interior_laplacian(bench_case, _FIXED_FRACTIONS[0])
    weight_sums = wp.full(n_free, 1.0, dtype=wp.float64, device=bench_case.device)
    preconditioner = bench_case.run(
        lambda: od.linalg.squared_laplacian_preconditioner(l_ff, weight_sums)
    )
    assert preconditioner.narrowed()[0].shape[0] == l_ff.values.size


def _squared_laplacian_factors(
    bench_case: BenchCase,
) -> tuple[odt.BsrMatrix[wp.float64], odt.BsrMatrix[wp.float64], wp.array[wp.float64]]:
    """``(M_ff, M_ffᵀ, ratios)`` at ``D = 1``: ``-L``'s interior block, its transpose, row sums."""
    l_ff, n_free = _interior_laplacian(bench_case, _FIXED_FRACTIONS[0])
    ratios_np = np.asarray(abs(_scipy_csr(l_ff, n_free)).sum(axis=1)).ravel()
    ratios = wp.array(ratios_np, dtype=wp.float64, device=bench_case.device)
    return l_ff, od.array.csr_transpose(l_ff), ratios


@pytest.mark.benchmark(group="squared_laplacian_from_factors")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_squared_laplacian_from_factors(bench_case: BenchCase) -> None:
    """
    ``SquaredLaplacianPreconditioner.from_factors``: the constructor without its scaling walk.

    Read against ``squared_laplacian_preconditioner``: the gap is the copy, row scaling and
    transpose a caller that wrote the factor while assembling does not pay again.
    """
    factor, factor_t, ratios = _squared_laplacian_factors(bench_case)
    preconditioner = bench_case.run(
        lambda: od.linalg.SquaredLaplacianPreconditioner.from_factors(factor, factor_t, ratios)
    )
    assert preconditioner.bind(1, int(factor.nrow)) is not None


@pytest.mark.benchmark(group="squared_laplacian_bind")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_squared_laplacian_bind(bench_case: BenchCase) -> None:
    """``SquaredLaplacianPreconditioner.bind``: the working vectors for a two-column solve."""
    factor, factor_t, ratios = _squared_laplacian_factors(bench_case)
    preconditioner = od.linalg.SquaredLaplacianPreconditioner.from_factors(factor, factor_t, ratios)
    n_free = int(factor.nrow)
    applied = bench_case.run(lambda: preconditioner.bind(_N_RHS, n_free))
    assert applied is not None


@pytest.mark.benchmark(group="squared_laplacian_narrowed")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_squared_laplacian_narrowed(bench_case: BenchCase) -> None:
    """
    ``SquaredLaplacianPreconditioner.narrowed``: the factor's two ``float32`` value casts.

    Built once per preconditioner, so every round takes a fresh one from ``from_factors``
    (outside the timed region) and times the first, building call.
    """
    factor, factor_t, ratios = _squared_laplacian_factors(bench_case)
    narrowed = bench_case.run(
        lambda preconditioner: preconditioner.narrowed(),
        setup=lambda: od.linalg.SquaredLaplacianPreconditioner.from_factors(
            factor, factor_t, ratios
        ),
    )
    assert narrowed[0].shape[0] == factor.values.size
