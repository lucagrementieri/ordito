"""
Benchmarks for ``ordito.cholesky``: building a sparse factorization and solving with it.

Meshes: the **quality** pair and two of the **scale** sweep. A factorization's cost does not
depend on the operator's condition number -- that is its whole point against the conjugate-gradient
solvers of [`test_linalg.py`](test_linalg.py) -- so ``saddle`` and ``saddle_graded`` should read
flat where CG spreads several-fold; the scale meshes show how the factorization and a solve grow
with the mesh. The operator is the heat method's ``M - tL`` (``heat_operators``' first system),
the one the heat diffusions fall back to factoring.

Two groups split the two costs a caller pays at different rates: ``sparse_cholesky`` times the
whole build (ordering, symbolic structure and the numeric factorization on the device) once per
operator; ``sparse_cholesky_solve`` times one refined solve against a built factorization, which
a caller pays per right-hand side.

No reference row: none of the nine reference libraries exposes a reusable sparse factorization
of an arbitrary operator (potpourri3d and libigl factor internally and are timed against the
calls that use them, in [`test_heat.py`](test_heat.py)); SciPy is the test oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase

_systems: dict[tuple[str, str], odt.BsrMatrix[wp.float64]] = {}


def _system(bench_case: BenchCase) -> odt.BsrMatrix[wp.float64]:
    """Return the heat system ``M - tL`` -- the input; its assembly is timed in test_heat."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _systems:
        _systems[key] = od.heat.heat_operators(bench_case.vertices_wp, bench_case.faces_wp)[0]
    return _systems[key]


@pytest.mark.benchmark(group="sparse_cholesky")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "saddle", "saddle_graded")
@pytest.mark.benchlibs("ordito")
def test_sparse_cholesky(bench_case: BenchCase) -> None:
    """Ordering, symbolic structure and numeric factorization of the heat system."""
    system = _system(bench_case)
    factor = bench_case.run(
        lambda: od.cholesky.sparse_cholesky(system, bench_case.vertices_wp), rounds=3
    )
    assert factor.n == bench_case.n_vertices


@pytest.mark.benchmark(group="sparse_cholesky_solve")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "saddle", "saddle_graded")
@pytest.mark.benchlibs("ordito")
def test_sparse_cholesky_solve(bench_case: BenchCase) -> None:
    """One componentwise-refined solve from a single source's indicator."""
    system = _system(bench_case)
    factor = od.cholesky.sparse_cholesky(system, bench_case.vertices_wp)
    rhs_np = np.zeros(bench_case.n_vertices)
    rhs_np[0] = 1.0
    rhs = wp.array(rhs_np, dtype=wp.float64, device=bench_case.device)
    solution = wp.zeros_like(rhs)
    bench_case.run(lambda: factor.solve(rhs, solution, componentwise=True))
    assert solution.shape[0] == bench_case.n_vertices
