"""
Regression tests for ``ordito.cholesky`` against ``scipy.sparse.linalg.spsolve``.

The references are SciPy's direct solvers on the very operator handed to
[`sparse_cholesky`][ordito.cholesky.sparse_cholesky]; none of the nine reference libraries binds
a reusable sparse factorization of an arbitrary operator, so this module carries no ``parity``
marker.
"""

from __future__ import annotations

import weakref
from typing import Any, cast

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito.kernels import cholesky as kernel_cholesky
from ordito.kernels import linalg as kernel_linalg
from tests.conversions import bsr_to_csr, numpy_to_warp, scipy_to_bsr, warp_empty


def _heat_system(mesh_wp: wp.Mesh) -> odt.BsrMatrix[wp.float64]:
    return od.heat.heat_operators(mesh_wp.points, mesh_wp.indices)[0]


def _poisson_system(mesh_wp: wp.Mesh) -> odt.BsrMatrix[wp.float64]:
    return od.heat.heat_operators(mesh_wp.points, mesh_wp.indices)[3]


@pytest.mark.parametrize("ordering", ["geometric", "pattern"])
@pytest.mark.parametrize(
    ("mesh_name", "rtol"), [("sphere_irregular", 1e-12), ("saddle_graded", 1e-11)]
)
def test_sparse_cholesky_matches_spsolve(
    request: pytest.FixtureRequest, mesh_name: str, rtol: float, ordering: str
) -> None:
    """
    Class A at 1e-12 componentwise: the heat system ``M - tL`` against SciPy's direct solve.

    The right-hand side is one source's indicator, so the solution decays by orders of magnitude
    away from it and a componentwise comparison is the meaningful one. Both orderings (from the
    vertices, and from the pattern alone) must give the same system's answer.

    On ``saddle_graded`` the bar is 1e-11, set by the system rather than the solver: there both
    answers have a componentwise backward error of 5e-16 (ordito 4.3e-16 to 5.0e-16, SciPy 5.2e-16
    to 5.3e-16), and the operator's componentwise conditioning turns that into forward differences
    up to 1.67e-12. So the backward error is asserted too, against SciPy's own.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    system = _heat_system(mesh_wp)
    n = int(system.nrow)
    rhs_np = np.zeros(n)
    rhs_np[0] = 1.0
    expected = spla.spsolve(bsr_to_csr(system).tocsc(), rhs_np)
    coordinates = mesh_wp.points if ordering == "geometric" else None
    factor = od.cholesky.sparse_cholesky(system, coordinates)
    solution = wp.zeros(n, dtype=wp.float64, device=mesh_wp.device)
    factor.solve(
        wp.array(rhs_np, dtype=wp.float64, device=mesh_wp.device), solution, componentwise=True
    )
    # Magnitudes: on ``sphere_irregular``'s obtuse faces the system is not an M-matrix, so some
    # entries of the decaying solution are negative.
    assert np.ptp(np.log10(np.abs(expected[expected != 0.0]))) > 3.0
    assert np.allclose(solution.numpy(), expected, rtol=rtol, atol=0.0)
    system_sp = bsr_to_csr(system)
    abs_sp = abs(system_sp)

    def backward_error(x: np.ndarray) -> float:
        return float((np.abs(rhs_np - system_sp @ x) / (abs_sp @ np.abs(x) + np.abs(rhs_np))).max())

    # ``tol`` bounds the componentwise backward error refinement stops at, so the default 1e-12
    # stops above SciPy's (``sphere_irregular``: 2.2e-15 against 3.7e-16), and 1e-15 is not enough
    # either: the pattern ordering stops there at 8.2e-16 / 9.5e-16 (CUDA / CPU). At 1e-16, below
    # what rounding reaches, every ordering on both fixtures and devices refines to 1.9e-16 to
    # 2.8e-16, below SciPy's 3.7e-16 to 5.4e-16.
    tight = wp.zeros(n, dtype=wp.float64, device=mesh_wp.device)
    factor.solve(
        wp.array(rhs_np, dtype=wp.float64, device=mesh_wp.device),
        tight,
        tol=1e-16,
        componentwise=True,
    )
    assert backward_error(tight.numpy()) <= 2.0 * backward_error(expected) + 1e-16


def test_sparse_cholesky_solves_columns_together(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Ordito against ordito: three columns solved together equal three solved one at a time."""
    _, mesh_wp = sphere_irregular
    system = _heat_system(mesh_wp)
    n = int(system.nrow)
    rhs_np = np.random.default_rng(3).standard_normal((3, n))
    factor = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    together = wp.zeros((3, n), dtype=wp.float64, device=device)
    factor.solve(wp.array(rhs_np, dtype=wp.float64, device=device), together)
    for column in range(3):
        alone = wp.zeros(n, dtype=wp.float64, device=device)
        factor.solve(wp.array(rhs_np[column], dtype=wp.float64, device=device), alone)
        assert np.allclose(together.numpy()[column], alone.numpy(), rtol=1e-12, atol=1e-14)


def test_sparse_cholesky_is_reproducible(
    device: str, saddle_graded: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Ordito against ordito: sums are reduced in a fixed order, so solves repeat bit for bit."""
    _, mesh_wp = saddle_graded
    system = _heat_system(mesh_wp)
    n = int(system.nrow)
    rhs = wp.array(np.random.default_rng(5).standard_normal(n), dtype=wp.float64, device=device)
    first = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    second = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    a = wp.zeros(n, dtype=wp.float64, device=device)
    b = wp.zeros(n, dtype=wp.float64, device=device)
    first.solve(rhs, a)
    second.solve(rhs, b)
    assert np.array_equal(a.numpy(), b.numpy())


def test_singular_component_takes_the_initial_guess_mean(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class B at 1e-10: the pure-Neumann Poisson system against SciPy's solve of its regularized form.

    ``-L`` has the constants for null space. On a consistent right-hand side conjugate gradient
    returns the solution whose mean is the initial guess's; the factorization pins one row and
    restores that constant. The reference solves ``-L + 1 1^T / n``, whose solution is the
    zero-mean one, shifted by the initial guess's mean (the named transform).
    """
    offset = 2.5
    _, mesh_wp = sphere_irregular
    system = _poisson_system(mesh_wp)
    n = int(system.nrow)
    rhs_np = np.random.default_rng(7).standard_normal(n)
    rhs_np -= rhs_np.mean()
    regularized = bsr_to_csr(system) + sp.csr_matrix(np.full((n, n), 1.0 / n))
    expected = spla.spsolve(regularized.tocsc(), rhs_np) + offset
    factor = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    solution = wp.full(n, offset, dtype=wp.float64, device=device)
    factor.solve(wp.array(rhs_np, dtype=wp.float64, device=device), solution, tol=1e-12)
    assert np.allclose(solution.numpy(), expected, rtol=0.0, atol=1e-10 * np.abs(expected).max())


def test_refactor_rejects_another_operator_shape(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: ``refactor`` and ``with_operator`` refuse an operator of another size.

    No reference reuses an analysis. An operator of another row count, or of the same rows storing
    fewer entries than the analysed pattern, made the factorization's kernels read past its
    storage (on the CPU a crash); both are now a ``ValueError`` before any launch.
    """
    _, mesh_wp = sphere_irregular
    system = _heat_system(mesh_wp)
    n = int(system.nrow)
    factor = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    smaller = scipy_to_bsr(sp.identity(n - 1, format="csr"), device)
    diagonal = scipy_to_bsr(sp.identity(n, format="csr"), device)
    for operator, match in ((smaller, "expected an"), (diagonal, "fewer entries")):
        with pytest.raises(ValueError, match=match):
            factor.refactor(operator)
        with pytest.raises(ValueError, match=match):
            factor.with_operator(operator)
    # The refused calls left the factorization as it was.
    rhs_np = np.random.default_rng(17).standard_normal(n)
    solution = wp.zeros(n, dtype=wp.float64, device=device)
    factor.solve(wp.array(rhs_np, dtype=wp.float64, device=device), solution)
    expected = spla.spsolve(bsr_to_csr(system).tocsc(), rhs_np)
    assert np.allclose(solution.numpy(), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_a_failed_refactor_leaves_no_factor_until_the_next(
    device: str, saddle_graded: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A at 1e-12: after a refused ``refactor``, ``solve`` raises; a later good one solves.

    The refused operator overwrote the factor, and the solves used to return its NaNs silently.
    Two refactors run, so the second goes through the recorded factorization path as well.
    """
    _, mesh_wp = saddle_graded
    system_np = bsr_to_csr(_heat_system(mesh_wp))
    n = system_np.shape[0]
    rhs = wp.array(np.random.default_rng(19).standard_normal(n), dtype=wp.float64, device=device)
    factor = od.cholesky.sparse_cholesky(scipy_to_bsr(system_np, device), mesh_wp.points)
    solution = wp.zeros(n, dtype=wp.float64, device=device)
    for _ in range(2):
        with pytest.raises(od.cholesky.NotPositiveDefiniteError):
            factor.refactor(scipy_to_bsr(sp.csr_matrix(-system_np), device))
        with pytest.raises(RuntimeError, match="refactor failed"):
            factor.solve(rhs, solution)
        factor.refactor(scipy_to_bsr(sp.csr_matrix(3.0 * system_np), device))
        solution.zero_()
        factor.solve(rhs, solution)
        expected = spla.spsolve((3.0 * system_np).tocsc(), rhs.numpy())
        assert np.allclose(
            solution.numpy(), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
        )


def test_either_sign_frees_the_refused_attempt_first(
    monkeypatch: pytest.MonkeyPatch, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a parity assert: ``negated=None`` drops its first, refused attempt before the second.

    The sign probe is forced wrong, so the first attempt fails; its solve blocks must be released
    by the time the second attempt allocates its own (the retry used to run inside the exception
    handler, whose traceback kept the first attempt's arena alive: two at peak).
    """
    _, mesh_wp = sphere_irregular
    laplacian = od.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float64)
    plan_type = od.cholesky._Plan  # pyright: ignore[reportPrivateUsage]
    numeric = plan_type.numeric
    attempts: list[weakref.ref[wp.array[wp.float64]]] = []
    alive_at_retry: list[bool] = []

    def recording(plan: object) -> object:
        alive_at_retry.extend(attempt() is not None for attempt in attempts)
        state = numeric(cast("Any", plan))
        attempts.append(weakref.ref(state.blocks))
        return state

    monkeypatch.setattr(plan_type, "numeric", recording)
    plan_cholesky = od.cholesky._plan_cholesky  # pyright: ignore[reportPrivateUsage]

    def wrong_sign(*args: Any, **kwargs: Any) -> object:
        # The analysis reads the operator's sign; the first attempt takes the other one.
        plan = plan_cholesky(*args, **kwargs)
        plan.leading_negative = False
        return plan

    monkeypatch.setattr(od.cholesky, "_plan_cholesky", wrong_sign)
    factor = od.cholesky.sparse_cholesky(laplacian, mesh_wp.points, negated=None)
    assert factor.negated
    assert len(attempts) == 2
    assert alive_at_retry == [False]


def test_an_empty_row_keeps_its_initial_value(device: str) -> None:
    """Class A: an unreferenced row (no entries) is a singular component of one, left untouched."""
    block = sp.csr_matrix(np.array([[4.0, -1.0], [-1.0, 4.0]]))
    full = scipy_to_bsr(sp.csr_matrix(sp.block_diag([block, sp.csr_matrix((1, 1))])), device)
    factor = od.cholesky.sparse_cholesky(full)
    solution = wp.array(np.array([0.0, 0.0, 7.0]), dtype=wp.float64, device=device)
    factor.solve(wp.array(np.array([3.0, 3.0, 0.0]), dtype=wp.float64, device=device), solution)
    assert np.allclose(solution.numpy(), [1.0, 1.0, 7.0], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("with_coordinates", [False, True], ids=["landmarks", "coordinates"])
def test_an_empty_operator_factors_and_solves_nothing(device: str, with_coordinates: bool) -> None:
    """
    Not a library comparison: a ``0 x 0`` operator factors with no levels and solves to nothing.

    The coordinate-free path computes landmark coordinates first, which raised on an empty
    operator; both paths now return an empty factorization.
    """
    empty = scipy_to_bsr(sp.csr_matrix((0, 0)), device)
    coordinates = warp_empty(0, wp.vec3, device) if with_coordinates else None
    factor = od.cholesky.sparse_cholesky(empty, coordinates)
    solution = warp_empty(0, wp.float64, device)
    factor.solve(warp_empty(0, wp.float64, device), solution)
    assert factor.n == 0
    assert solution.shape == (0,)


def test_sparse_cholesky_rejects_invalid_inputs(
    device: str, monkeypatch: pytest.MonkeyPatch, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: each guard raises before a factor is used.

    An indefinite operator, coordinates of another length, a ``float32`` or mis-sized solve, and a
    factorization past ``CHOLESKY_MEMORY_BUDGET`` are each refused with their message.
    """
    _, mesh_wp = sphere_irregular
    system = _heat_system(mesh_wp)
    negated = scipy_to_bsr(-bsr_to_csr(system), device)
    with pytest.raises(od.cholesky.NotPositiveDefiniteError, match="not positive definite"):
        od.cholesky.sparse_cholesky(negated, mesh_wp.points)
    with pytest.raises(ValueError, match="coordinates"):
        od.cholesky.sparse_cholesky(system, odt.as_dense(mesh_wp.points[:3]))
    factor = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    n = int(system.nrow)
    for dtype, rows, match in ((wp.float32, n, "float64"), (wp.float64, n + 1, "rows")):
        with pytest.raises(ValueError, match=match):
            factor.solve(
                wp.zeros(rows, dtype=dtype, device=device),
                wp.zeros(rows, dtype=dtype, device=device),
            )
    monkeypatch.setattr(od.cholesky, "CHOLESKY_MEMORY_BUDGET", 1024)
    with pytest.raises(ValueError, match="CHOLESKY_MEMORY_BUDGET"):
        od.cholesky.sparse_cholesky(system, mesh_wp.points)


def test_sparse_cholesky_rejects_a_one_sided_entry(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: a stored entry whose transpose is not stored raises ``ValueError``.

    SciPy's solvers take any pattern. The factorization reads one triangle of the pattern, so an
    entry stored on one side only (here an explicit zero, numerically symmetric) made its
    extend-add write outside the fronts; the analysis now rejects it. Both an entry in the upper
    and one in the lower triangle are tried, between rows the mesh does not join.
    """
    _, mesh_wp = sphere_irregular
    system_np = bsr_to_csr(_heat_system(mesh_wp)).tolil()
    n = system_np.shape[0]
    far = int(np.flatnonzero(system_np.getrow(0).toarray()[0] == 0.0)[-1])
    assert far > 0
    for row, column in ((0, far), (far, 0)):
        one_sided = system_np.copy()
        one_sided[row, column] = 1.0
        one_sided = one_sided.tocsr()
        one_sided.data[one_sided.indptr[row] : one_sided.indptr[row + 1]][
            one_sided.indices[one_sided.indptr[row] : one_sided.indptr[row + 1]] == column
        ] = 0.0
        assert one_sided.nnz == system_np.nnz + 1
        with pytest.raises(ValueError, match="not symmetric"):
            od.cholesky.sparse_cholesky(scipy_to_bsr(one_sided, device), mesh_wp.points)
    assert n == int(mesh_wp.points.shape[0])


def test_refinement_never_passes_a_nan_residual(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: a NaN solution fails both backward-error tests instead of passing.

    Both tests used ``error > tol * scale``, false for NaN, so a NaN initial guess passed the
    refinement's first test (the solve returned at once) and ``backward_error_exceeds`` (the
    verified solves' check) cleared it. The refinement now runs to its round cap, and the check
    flags; the same solution with its NaN replaced passes both.
    """
    _, mesh_wp = sphere_irregular
    system = _heat_system(mesh_wp)
    n = int(system.nrow)
    rhs_np = np.random.default_rng(23).standard_normal(n)
    exact_np = spla.spsolve(bsr_to_csr(system).tocsc(), rhs_np)
    rhs = wp.array(rhs_np, dtype=wp.float64, device=device)
    factor = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    for guess_np, rounds, flagged in (
        (np.where(np.arange(n) == 5, np.nan, exact_np), od.cholesky.CHOLESKY_REFINE_ROUNDS + 1, 1),
        (exact_np, 1, 0),
    ):
        solution = wp.array(guess_np, dtype=wp.float64, device=device)
        factor.solve(rhs, solution)
        state = factor._solves[1].state.numpy()  # pyright: ignore[reportPrivateUsage]
        assert state[int(kernel_cholesky.REFINE_ROUND)] == rounds
        flag = wp.zeros(1, dtype=wp.int32, device=device)
        nothing = wp.zeros(0, dtype=wp.float64, device=device)
        wp.launch(
            kernel_linalg.backward_error_exceeds,
            dim=(n, 1),
            inputs=[system.offsets, system.columns, system.values,
                    wp.array(guess_np, dtype=wp.float64, device=device), rhs, wp.int32(n),
                    wp.float64(1e-10), nothing, nothing, wp.int32(0)],
            outputs=[flag],
            device=device,
        )  # fmt: skip
        assert int(flag.numpy()[0]) == flagged
        # A NaN residual or threshold of a conjugate-gradient result flags its column too, as
        # unconverged; a later check reuses the word uncleared, raised above a higher base.
        residual = wp.array([np.nan], dtype=wp.float64, device=device)
        threshold = wp.array([1.0], dtype=wp.float64, device=device)
        for base in (0, 4):
            if base == 0:
                flag.zero_()
            wp.launch(
                kernel_linalg.backward_error_exceeds,
                dim=(n, 1),
                inputs=[system.offsets, system.columns, system.values,
                        wp.array(exact_np, dtype=wp.float64, device=device), rhs, wp.int32(n),
                        wp.float64(1e-10), residual, threshold, wp.int32(base)],
                outputs=[flag],
                device=device,
            )  # fmt: skip
            assert int(flag.numpy()[0]) == base + int(kernel_linalg.BACKWARD_ERROR_UNCONVERGED)


def test_negated_factorization_solves_a_negative_definite_operator(
    device: str, saddle_graded: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Class A at 1e-12: ``negated=True`` factors ``-A`` and still answers ``A x = b``."""
    _, mesh_wp = saddle_graded
    system_np = -bsr_to_csr(_heat_system(mesh_wp))
    n = system_np.shape[0]
    rhs_np = np.random.default_rng(13).standard_normal(n)
    factor = od.cholesky.sparse_cholesky(
        scipy_to_bsr(sp.csr_matrix(system_np), device), mesh_wp.points, negated=True
    )
    solution = wp.zeros(n, dtype=wp.float64, device=device)
    factor.solve(wp.array(rhs_np, dtype=wp.float64, device=device), solution)
    expected = spla.spsolve(system_np.tocsc(), rhs_np)
    assert factor.negated
    assert np.allclose(solution.numpy(), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def _vector_heat_systems(
    mesh_wp: wp.Mesh,
) -> tuple[odt.BsrMatrix[wp.float64], odt.BsrMatrix[wp.float64]]:
    """Return the heat system and the vector heat system's scalar expansion, one pattern apart."""
    vector_system, scalar_operators = od.heat.vector_heat_operators(
        mesh_wp.points, mesh_wp.indices
    )[:2]
    return scalar_operators[0], od.array.scalar_expansion(vector_system)


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
def test_with_expanded_operator_matches_spsolve(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A at 1e-11: the vector heat system, its analysis derived, against SciPy's direct solve.

    The expansion's analysis is the heat system's doubled, with no ordering of its own; its fill
    must not exceed an analysis of the expansion's own rows (ordito against ordito, the solve
    blocks' entries within 2 %: measured 0.99x on ``saddle_graded`` and 0.96x on
    ``sphere_irregular``, both devices). Mutation probe: deriving from a heat analysis at the
    default granularity (``expandable=False``), whose merged top holds either fixture whole, holds
    2.0x and 5.0x, and fails the fill bound.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    heat_system, expanded = _vector_heat_systems(mesh_wp)
    device = mesh_wp.device
    n = int(expanded.nrow)
    heat = od.cholesky.sparse_cholesky(heat_system, mesh_wp.points, expandable=True)
    assert heat.expandable
    derived = heat.with_expanded_operator(expanded)
    assert derived.n == n == 2 * heat.n
    assert not derived.expandable
    rhs_np = np.random.default_rng(23).standard_normal(n)
    solution = wp.zeros(n, dtype=wp.float64, device=device)
    derived.solve(wp.array(rhs_np, dtype=wp.float64, device=device), solution)
    expected = spla.spsolve(bsr_to_csr(expanded).tocsc(), rhs_np)
    assert np.allclose(solution.numpy(), expected, rtol=1e-11, atol=1e-11 * np.abs(expected).max())
    points = wp.array(np.repeat(mesh_wp.points.numpy(), 2, axis=0), dtype=wp.vec3, device=device)
    own = od.cholesky._plan_cholesky(expanded, points)  # pyright: ignore[reportPrivateUsage]
    blocks = derived._plan.block_entries  # pyright: ignore[reportPrivateUsage]
    assert blocks <= 1.02 * own.block_entries


def test_with_expanded_operator_keeps_each_empty_row(device: str) -> None:
    """
    Class A: an unreferenced vertex's two expanded rows are two components, each left untouched.

    The expansion's components are the scalar ones doubled except there: the two rows of an empty
    scalar row share no entry, so each is pinned on its own. Mutation probe: labelling both rows
    as one component leaves one zero pivot unpinned, and the factorization raises
    ``NotPositiveDefiniteError``.
    """
    scalar = scipy_to_bsr(
        sp.csr_matrix(
            (np.array([4.0, -1.0, -1.0, 4.0]), np.array([0, 1, 0, 1]), np.array([0, 2, 4, 4])),
            shape=(3, 3),
        ),
        device,
    )
    factor = od.cholesky.sparse_cholesky(scalar, expandable=True)
    identity = np.eye(2)
    blocks = np.stack([4.0 * identity, -identity, -identity, 4.0 * identity])
    block_system = od.array.bsr_from_csr(
        3,
        3,
        wp.array([0, 2, 4, 4], dtype=wp.int32, device=device),
        wp.array([0, 1, 0, 1], dtype=wp.int32, device=device),
        wp.array(blocks, dtype=wp.mat22d, device=device),
    )
    derived = factor.with_expanded_operator(od.array.scalar_expansion(block_system))
    solution = wp.array([0.0, 0.0, 0.0, 0.0, 7.0, 9.0], dtype=wp.float64, device=device)
    rhs = wp.array([2.0, 2.0, 7.0, 7.0, 0.0, 0.0], dtype=wp.float64, device=device)
    derived.solve(rhs, solution)
    assert np.allclose(solution.numpy(), [1.0, 1.0, 2.0, 2.0, 7.0, 9.0], rtol=1e-14, atol=0.0)


def test_with_expanded_operator_rejects_another_operator_shape(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Not a library comparison: an operator not of the expansion's size or storage is refused."""
    _, mesh_wp = sphere_irregular
    heat_system, expanded = _vector_heat_systems(mesh_wp)
    factor = od.cholesky.sparse_cholesky(heat_system, mesh_wp.points)
    n = 2 * factor.n
    smaller = scipy_to_bsr(sp.identity(n - 1, format="csr"), device)
    diagonal = scipy_to_bsr(sp.identity(n, format="csr"), device)
    for operator, match in ((heat_system, "expected an"), (smaller, "expected an"),
                            (diagonal, "fewer entries")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            factor.with_expanded_operator(operator)
    assert factor.with_expanded_operator(expanded).n == n


def _two_component_system(device: str) -> tuple[sp.csr_matrix, np.ndarray]:
    sphere = tm.creation.icosphere(3)
    shell = tm.creation.icosphere(2)
    shell.vertices += 3.0
    vertices_np = np.vstack([sphere.vertices, shell.vertices])
    blocks = []
    for part in (sphere, shell):
        points, faces = numpy_to_warp(part.vertices, part.faces.astype(np.int32).ravel(), device)
        blocks.append(bsr_to_csr(od.heat.heat_operators(points, faces)[0]))
    return sp.csr_matrix(sp.block_diag(blocks)), vertices_np


@pytest.mark.parametrize("ordering", ["geometric", "pattern"])
def test_sparse_cholesky_orders_two_components(device: str, ordering: str) -> None:
    """
    Class A at 1e-12 componentwise: two disconnected surfaces in one operator against ``spsolve``.

    The first bisection crosses no pattern edge, so it makes no separator and the two halves are
    the roots of two dissection trees: the branch a single connected mesh never reaches.
    """
    system_np, vertices_np = _two_component_system(device)
    n = system_np.shape[0]
    rhs_np = np.zeros(n)
    rhs_np[[0, n - 1]] = 1.0
    expected = spla.spsolve(system_np.tocsc(), rhs_np)
    coordinates = (
        wp.array(vertices_np, dtype=wp.vec3, device=device) if ordering == "geometric" else None
    )
    factor = od.cholesky.sparse_cholesky(scipy_to_bsr(system_np, device), coordinates)
    solution = wp.zeros(n, dtype=wp.float64, device=device)
    factor.solve(wp.array(rhs_np, dtype=wp.float64, device=device), solution, componentwise=True)
    # Magnitudes: on ``sphere_irregular``'s obtuse faces the system is not an M-matrix, so some
    # entries of the decaying solution are negative.
    assert np.ptp(np.log10(np.abs(expected[expected != 0.0]))) > 3.0
    assert np.allclose(solution.numpy(), expected, rtol=1e-12, atol=0.0)


def _fill_case(case: str, device: str) -> tuple[sp.csr_matrix, wp.array[wp.vec3]]:
    if case == "two_components":
        system_np, vertices_np = _two_component_system(device)
        return system_np, wp.array(vertices_np, dtype=wp.vec3, device=device)
    mesh = tm.creation.icosphere(4)
    vertices_np, faces_np = mesh.vertices, mesh.faces
    if case == "cap":
        kept, faces_np = np.unique(
            faces_np[mesh.triangles_center[:, 2] > -0.3], return_inverse=True
        )
        vertices_np, faces_np = vertices_np[kept], faces_np.reshape(-1, 3)
    points, faces = numpy_to_warp(vertices_np, faces_np.astype(np.int32).ravel(), device)
    return bsr_to_csr(od.heat.heat_operators(points, faces)[0]), points


@pytest.mark.parametrize("case", ["sphere", "cap", "two_components"])
def test_symbolic_structure_is_the_factors_fill(device: str, case: str) -> None:
    """
    Not a library comparison: each supernode's rows are its columns' fill in the exact factor.

    No reference exposes a supernodal row structure. Each supernode's row structure must equal
    the union of its columns' rows in the exact Cholesky factor of the permuted pattern, found by
    a dense boolean elimination.

    Refinement would hide a missing fill entry (the factor would only be approximate and the
    solve would iterate to the same answer), so the structure is checked directly: a dropped
    row or one too many both fail.
    """
    system_np, points = _fill_case(case, device)
    n = system_np.shape[0]
    plan = od.cholesky._plan_cholesky(scipy_to_bsr(system_np, device), points)  # pyright: ignore[reportPrivateUsage]
    perm = plan.perm.numpy()[:n]
    assert np.array_equal(np.sort(perm), np.arange(n))
    pattern = system_np[perm][:, perm].toarray() != 0.0
    for j in range(n):
        below = j + 1 + np.flatnonzero(pattern[j + 1 :, j])
        pattern[np.ix_(below, below)] = True
    ncol = plan.ncol.numpy()
    c0 = plan.c0.numpy()
    front_size = plan.front_size.numpy()
    row_offsets = plan.row_offsets.numpy()
    rows = plan.rows.numpy()
    assert ncol.size > 16
    for k in range(ncol.size):
        last = c0[k] + ncol[k]
        expected = last + np.flatnonzero(pattern[last:, c0[k] : last].any(axis=1))
        got = rows[row_offsets[k] : row_offsets[k] + front_size[k] - ncol[k]]
        assert np.array_equal(got, expected)


def _sinuous_ribbon() -> tuple[np.ndarray, np.ndarray]:
    """Return a 801 x 21 grid strip bent into four sine periods: coordinate cuts slice it apart."""
    u, w = np.meshgrid(np.linspace(0.0, 8.0 * np.pi, 801), np.linspace(0.0, 1.0, 21), indexing="ij")
    vertices_np = np.stack([u, 3.0 * np.sin(u), w], axis=-1).reshape(-1, 3)
    iu, iw = np.meshgrid(np.arange(800), np.arange(20), indexing="ij")
    a = (iu * 21 + iw).ravel()
    faces_np = np.concatenate(
        [np.stack([a, a + 21, a + 1], 1), np.stack([a + 1, a + 21, a + 22], 1)]
    )
    return vertices_np, faces_np.astype(np.int32)


def test_symbolic_structure_follows_the_elimination_tree(device: str) -> None:
    """
    Not a library comparison: rows pass up the supernodal elimination tree, not the dissection's.

    No reference exposes a supernodal structure. The oracle is the supernodal symbolic
    factorization written out in NumPy over the plan's own order and supernodes: in elimination
    order, a supernode's rows are its operator entries' rows past its columns and its
    children's rows past its columns, and its parent is the owner of its first row. On the
    dense-elimination cases above it agrees to the row.

    A coordinate cut slices the sinuous ribbon into pieces, so some dissection child is not
    connected to its dissection parent. Passing rows up the dissection tree gave two of its
    supernodes rows their columns never fill, on both devices (and +1.5 % factor entries on
    ``dragon``); the dense elimination is too slow to check a mesh this size, so the test above
    could not see it. Mutation probe: walking the dissection tree instead (the code before the
    elimination tree) fails it on CUDA and CPU, at a supernode the dissection hangs one level
    too low.
    """
    vertices_np, faces_np = _sinuous_ribbon()
    points, faces = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    system_np = bsr_to_csr(od.heat.heat_operators(points, faces)[0])
    n = system_np.shape[0]
    plan = od.cholesky._plan_cholesky(scipy_to_bsr(system_np, device), points)  # pyright: ignore[reportPrivateUsage]
    perm = plan.perm.numpy()[:n]
    ncol, c0, front_size = plan.ncol.numpy(), plan.c0.numpy(), plan.front_size.numpy()
    row_offsets, rows, parent = plan.row_offsets.numpy(), plan.rows.numpy(), plan.parent.numpy()
    lower = sp.tril(system_np[perm][:, perm], format="csc")
    owner = np.repeat(np.arange(ncol.size), ncol)
    expected: dict[int, np.ndarray] = {}
    children: list[list[int]] = [[] for _ in range(ncol.size)]
    for k in np.argsort(c0):
        last = c0[k] + ncol[k]
        own = lower.indices[lower.indptr[c0[k]] : lower.indptr[last]]
        parts = [own[own >= last], *(expected[c][expected[c] >= last] for c in children[k])]
        expected[k] = np.unique(np.concatenate(parts))
        got = rows[row_offsets[k] : row_offsets[k] + front_size[k] - ncol[k]]
        assert np.array_equal(got, expected[k])
        if expected[k].size:
            children[owner[expected[k][0]]].append(int(k))
            assert parent[k] == owner[expected[k][0]]


def _held_bytes(*objects: object) -> int:
    """
    Return the summed capacity of every distinct allocation reachable from ``objects``.

    A view counts its owner's allocation, once however many views share it. The operator a
    factorization was handed (``matrix``, ``_matrix``) is the caller's and is skipped; the copy a
    repeated ``refactor`` keeps is reached through its own attribute.
    """
    seen: set[int] = set()
    owners: dict[int, int] = {}

    def walk(item: object) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, wp.array):
            owner: Any = item
            while isinstance(getattr(owner, "_ref", None), wp.array):
                owner = owner._ref
            owners[id(owner)] = int(owner.capacity)
        elif isinstance(item, (list, tuple)):
            for value in cast("list[object] | tuple[object, ...]", item):
                walk(value)
        elif isinstance(item, dict):
            for value in cast("dict[object, object]", item).values():
                walk(value)
        elif type(item).__module__.startswith(("ordito", "warp._src.codegen")):
            # Ordito's objects, and Warp structs: the kept operator copy is a ``BsrMatrix`` one.
            for name, value in vars(item).items():
                if name not in ("matrix", "_matrix"):
                    walk(value)

    for item in objects:
        walk(item)
    return sum(owners.values())


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "empty", "one_leaf"])
def test_nbytes_is_the_memory_the_factorization_holds(
    request: pytest.FixtureRequest, device: str, monkeypatch: pytest.MonkeyPatch, mesh_name: str
) -> None:
    """
    Not a library comparison: ``nbytes`` against the allocations the factorization holds.

    No reference reports a factorization's memory. ``nbytes`` must equal, to the byte, the summed
    capacity of every distinct device allocation reachable from the factorization: once built,
    after a solve at one right-hand-side count and at another, after one replaced by another
    tolerance, and after a repeated ``refactor`` keeps its operator copy and fronts arena. The
    memory budget is the first factorization's peak -- what it holds plus the larger of the
    transient fronts arena and a one-column solve's scratch, each read off the measured
    allocations -- so a budget of exactly that admits it and one byte less refuses it. A
    second factorization sharing the analysis holds only its own factor:
    ``held_nbytes`` counts the analysis once. An estimate pricing ``float64`` entries at 4 bytes,
    or leaving out the solve scratch or the refactor storage, fails the first equality.
    """
    if mesh_name == "empty":
        system = scipy_to_bsr(sp.csr_matrix((0, 0)), device)
        coordinates = None
    elif mesh_name == "one_leaf":
        # Twelve rows: one leaf, one supernode, no extend-add.
        _, mesh_wp = request.getfixturevalue("icosahedron")
        system, coordinates = _heat_system(mesh_wp), mesh_wp.points
    else:
        _, mesh_wp = request.getfixturevalue(mesh_name)
        system, coordinates = _heat_system(mesh_wp), mesh_wp.points
    n = int(system.nrow)
    factor = od.cholesky.sparse_cholesky(system, coordinates)
    built = factor.nbytes
    assert built == _held_bytes(factor) > 0
    if n == 0:
        return
    rhs = wp.ones(n, dtype=wp.float64, device=device)
    factor.solve(rhs, wp.zeros_like(rhs))
    solve_scratch = factor.nbytes - built
    assert factor.nbytes == _held_bytes(factor)
    assert solve_scratch > 0
    columns = wp.ones(3 * n, dtype=wp.float64, device=device).reshape((3, n))
    factor.solve(columns, wp.zeros_like(columns), componentwise=True)
    assert factor.nbytes == _held_bytes(factor)
    factor.solve(rhs, wp.zeros_like(rhs), tol=1e-10)  # replaces the one-column configuration
    assert factor.nbytes == _held_bytes(factor)
    before = factor.nbytes
    factor.refactor(system)
    fronts = factor._fronts  # pyright: ignore[reportPrivateUsage]
    assert fronts is not None
    assert factor.nbytes == _held_bytes(factor) > before + fronts.capacity
    # The budget: the peak of a first factorization, to the byte.
    peak = built + max(fronts.capacity, solve_scratch)
    monkeypatch.setattr(od.cholesky, "CHOLESKY_MEMORY_BUDGET", peak - 1)
    with pytest.raises(ValueError, match=f"needs {peak} bytes"):
        od.cholesky.sparse_cholesky(system, coordinates)
    monkeypatch.setattr(od.cholesky, "CHOLESKY_MEMORY_BUDGET", peak)
    fresh = od.cholesky.sparse_cholesky(system, coordinates)
    assert fresh.nbytes == built
    # A shared analysis, counted once.
    sibling = fresh.with_operator(system)
    together = od.cholesky.held_nbytes([fresh, sibling, fresh])
    assert together == _held_bytes(fresh, sibling) < fresh.nbytes + sibling.nbytes


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "icosahedron"])
def test_nbytes_of_a_derived_analysis_is_the_memory_it_holds(
    request: pytest.FixtureRequest, device: str, mesh_name: str
) -> None:
    """
    Not a library comparison: a derived analysis' ``nbytes`` against the allocations it holds.

    ``with_expanded_operator`` sizes its analysis' tables by the expansion (no sort width, columns
    by supernode), which ``nbytes`` must count to the byte, built and after a solve; the derived
    factorization holds none of the scalar one's analysis, so together they hold their sum. The
    icosahedron is one leaf: one supernode, no extend-add.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    heat_system, expanded = _vector_heat_systems(mesh_wp)
    heat = od.cholesky.sparse_cholesky(heat_system, mesh_wp.points, expandable=True)
    derived = heat.with_expanded_operator(expanded)
    assert derived.nbytes == _held_bytes(derived) > 0
    rhs = wp.ones(derived.n, dtype=wp.float64, device=device)
    derived.solve(rhs, wp.zeros_like(rhs))
    assert derived.nbytes == _held_bytes(derived)
    together = od.cholesky.held_nbytes([heat, derived])
    assert together == _held_bytes(heat, derived) == heat.nbytes + derived.nbytes


def test_landmark_coordinates_match_breadth_first_distances(device: str) -> None:
    """
    Class A: the landmark coordinates of many components against SciPy's breadth-first distances.

    A pattern of 300 components -- paths of 40 to 79 rows with chords inside each, and paths of
    at most ``CHOLESKY_LEAF_ROWS`` rows that keep zero coordinates -- so the host's simultaneous
    searches (the CPU path, linear in the pattern) and the device's both meet ties and small
    components. The reference runs each landmark's search per component with
    [`scipy.sparse.csgraph.shortest_path`][] and picks the next landmark as the sequence does:
    farthest by the nearer of the last two, the lowest row on a tie.
    """
    rng = np.random.default_rng(29)
    sizes = np.where(np.arange(300) % 3 == 0, rng.integers(5, 33, 300), rng.integers(40, 80, 300))
    starts = np.concatenate([[0], np.cumsum(sizes)])
    n = int(starts[-1])
    rows, cols = [np.arange(n)], [np.arange(n)]
    for start, size in zip(starts[:-1], sizes, strict=True):
        path = np.arange(start, start + size - 1)
        chords = start + rng.integers(0, size, (size // 10, 2))
        rows += [path, chords[:, 0]]
        cols += [path + 1, chords[:, 1]]
    row_np, col_np = np.concatenate(rows), np.concatenate(cols)
    pattern = sp.csr_matrix(
        (np.ones(2 * row_np.size), (np.r_[row_np, col_np], np.r_[col_np, row_np])), shape=(n, n)
    )
    pattern.sum_duplicates()
    system = scipy_to_bsr(pattern, device)
    labels = od.graph.connected_component_labels(system)
    coordinates = od.cholesky._landmark_coordinates(system, labels)  # pyright: ignore[reportPrivateUsage]
    coordinates_np = coordinates.numpy().view(np.float32).reshape(-1, 3)
    expected = np.zeros((n, 3))
    searched = 0
    for start, size in zip(starts[:-1], sizes, strict=True):
        if size <= od.cholesky.CHOLESKY_LEAF_ROWS:
            continue
        searched += 1
        block = pattern[start : start + size][:, start : start + size]

        def hops(seed: int, block: sp.csr_matrix = block) -> np.ndarray:
            return sp.csgraph.shortest_path(block, unweighted=True, indices=seed)

        first = hops(0)
        a = hops(int(np.argmax(first)))
        b = hops(int(np.argmax(a)))
        c = hops(int(np.argmax(np.minimum(a, b))))
        expected[start : start + size] = np.stack([a, b, c], 1)
    assert searched > 150
    assert np.ptp(expected[:, 2]) > 20.0
    assert np.array_equal(coordinates_np, expected.astype(np.float32))
