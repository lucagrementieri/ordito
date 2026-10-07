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
from tests.conversions import bsr_to_csr, numpy_to_warp, scipy_to_bsr


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
    with pytest.raises(od.cholesky.NotPositiveDefiniteError, match="not positive definite"):
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


def test_negated_factorization_solves_a_negative_definite_operator(
    device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Class A at 1e-12: ``negated=True`` factors ``-A`` and still answers ``A x = b``."""
    _, mesh_wp = hemisphere
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
    assert np.ptp(np.log10(expected)) > 3.0
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


def test_landmark_coordinates_match_the_host_search(device: str) -> None:
    """
    Ordito against ordito: the device's breadth-first searches give the host's coordinates.

    The simultaneous searches of every component equal the host's frontier searches exactly, on
    a pattern of two surfaces and an isolated row.

    The host search is the oracle (it is the CPU device's path, so on the CPU this compares it
    with itself); the isolated row is a component too small to search, which keeps zeros.
    """
    system_np, _ = _two_component_system(device)
    system_np = sp.csr_matrix(sp.block_diag([system_np, sp.csr_matrix(np.ones((1, 1)))]))
    system = scipy_to_bsr(system_np, device)
    labels = od.graph.connected_component_labels(system)
    coordinates = od.cholesky._landmark_coordinates(system, labels)  # pyright: ignore[reportPrivateUsage]
    host = od.cholesky._host_landmark_coordinates(system, labels)  # pyright: ignore[reportPrivateUsage]
    assert np.ptp(host[:, 2]) > 4.0
    assert np.array_equal(coordinates.numpy().view(np.float32).reshape(-1, 3), host)
