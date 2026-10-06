"""
Regression tests for ``ordito.cholesky`` against ``scipy.sparse.linalg.spsolve``.

The references are SciPy's direct solvers on the very operator handed to
[`sparse_cholesky`][ordito.cholesky.sparse_cholesky]; none of the nine reference libraries binds
a reusable sparse factorization of an arbitrary operator, so this module carries no ``parity``
marker.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from tests.conversions import bsr_to_csr, scipy_to_bsr


def _heat_system(mesh_wp: wp.Mesh) -> odt.BsrMatrix[wp.float64]:
    return od.heat.heat_operators(mesh_wp.points, mesh_wp.indices)[0]


def _poisson_system(mesh_wp: wp.Mesh) -> odt.BsrMatrix[wp.float64]:
    return od.heat.heat_operators(mesh_wp.points, mesh_wp.indices)[3]


@pytest.mark.parametrize("ordering", ["geometric", "pattern"])
@pytest.mark.parametrize("mesh_name", ["icosphere", "hemisphere"])
def test_sparse_cholesky_matches_spsolve(
    request: pytest.FixtureRequest, mesh_name: str, ordering: str
) -> None:
    """
    Class A at 1e-12 componentwise: the heat system ``M - tL`` against SciPy's direct solve.

    The right-hand side is one source's indicator, so the solution decays by orders of magnitude
    away from it and a componentwise comparison is the meaningful one. Both orderings (from the
    vertices, and from the pattern alone) must give the same system's answer.
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
    assert np.ptp(np.log10(expected)) > 3.0
    assert np.allclose(solution.numpy(), expected, rtol=1e-12, atol=0.0)


def test_sparse_cholesky_solves_columns_together(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Ordito against ordito: three columns solved together equal three solved one at a time."""
    _, mesh_wp = icosphere
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
    device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Ordito against ordito: sums are reduced in a fixed order, so solves repeat bit for bit."""
    _, mesh_wp = hemisphere
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


@pytest.mark.parametrize("offset", [0.0, 2.5])
def test_singular_component_takes_the_initial_guess_mean(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh], offset: float
) -> None:
    """
    Class B at 1e-10: the pure-Neumann Poisson system against SciPy's solve of its regularized form.

    ``-L`` has the constants for null space. On a consistent right-hand side conjugate gradient
    returns the solution whose mean is the initial guess's; the factorization pins one row and
    restores that constant. The reference solves ``-L + 1 1^T / n``, whose solution is the
    zero-mean one, shifted by the initial guess's mean (the named transform).
    """
    _, mesh_wp = icosphere
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


def test_refactor_takes_new_values_of_the_same_pattern(
    device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Class A at 1e-12: after ``refactor`` with the operator doubled, the solution halves."""
    _, mesh_wp = hemisphere
    system_np = bsr_to_csr(_heat_system(mesh_wp))
    n = system_np.shape[0]
    rhs_np = np.random.default_rng(9).standard_normal(n)
    factor = od.cholesky.sparse_cholesky(scipy_to_bsr(system_np, device), mesh_wp.points)
    doubled = scipy_to_bsr(sp.csr_matrix(2.0 * system_np), device)
    factor.refactor(doubled)
    solution = wp.zeros(n, dtype=wp.float64, device=device)
    factor.solve(wp.array(rhs_np, dtype=wp.float64, device=device), solution)
    expected = spla.spsolve((2.0 * system_np).tocsc(), rhs_np)
    assert np.allclose(solution.numpy(), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_an_empty_row_keeps_its_initial_value(device: str) -> None:
    """Class A: an unreferenced row (no entries) is a singular component of one, left untouched."""
    block = sp.csr_matrix(np.array([[4.0, -1.0], [-1.0, 4.0]]))
    full = scipy_to_bsr(sp.csr_matrix(sp.block_diag([block, sp.csr_matrix((1, 1))])), device)
    factor = od.cholesky.sparse_cholesky(full)
    solution = wp.array(np.array([0.0, 0.0, 7.0]), dtype=wp.float64, device=device)
    factor.solve(wp.array(np.array([3.0, 3.0, 0.0]), dtype=wp.float64, device=device), solution)
    assert np.allclose(solution.numpy(), [1.0, 1.0, 7.0], rtol=1e-14, atol=0.0)


def test_sparse_cholesky_rejects_an_indefinite_operator(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    _, mesh_wp = icosahedron
    negated = scipy_to_bsr(-bsr_to_csr(_heat_system(mesh_wp)), device)
    with pytest.raises(ValueError, match="not positive definite"):
        od.cholesky.sparse_cholesky(negated, mesh_wp.points)


def test_sparse_cholesky_honours_its_memory_budget(
    monkeypatch: pytest.MonkeyPatch, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    _, mesh_wp = icosahedron
    monkeypatch.setattr(od.cholesky, "CHOLESKY_MEMORY_BUDGET", 1024)
    with pytest.raises(ValueError, match="CHOLESKY_MEMORY_BUDGET"):
        od.cholesky.sparse_cholesky(_heat_system(mesh_wp), mesh_wp.points)


def test_sparse_cholesky_rejects_mismatched_inputs(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    _, mesh_wp = icosahedron
    system = _heat_system(mesh_wp)
    with pytest.raises(ValueError, match="coordinates"):
        od.cholesky.sparse_cholesky(system, odt.as_dense(mesh_wp.points[:3]))
    factor = od.cholesky.sparse_cholesky(system, mesh_wp.points)
    n = int(system.nrow)
    with pytest.raises(ValueError, match="float64"):
        factor.solve(
            wp.zeros(n, dtype=wp.float32, device=device),
            wp.zeros(n, dtype=wp.float32, device=device),
        )
    with pytest.raises(ValueError, match="rows"):
        factor.solve(
            wp.zeros(n + 1, dtype=wp.float64, device=device),
            wp.zeros(n + 1, dtype=wp.float64, device=device),
        )
