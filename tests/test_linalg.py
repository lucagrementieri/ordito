# This file tests ordito.linalg's private solver states and helpers directly.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import gc
import warnings
import weakref
from typing import Any, TypedDict, cast

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import trimesh as tm
import warp as wp
import warp.sparse as wps

import ordito as od
import ordito.typing as odt
from tests.comparisons import assert_nonconstant
from tests.conversions import bsr_to_dense, trimesh_to_pymeshlab


class _CgOptions(TypedDict):
    tol: float
    maxiter: int
    check_every: int
    preconditioner: str


def _spd_system(
    device: str, n: int = 64, n_rhs: int = 3, seed: int = 11
) -> tuple[odt.BsrMatrix[wp.float64], wp.array[wp.float64, Any], np.ndarray, np.ndarray]:
    """
    Build an SPD operator with ``n_rhs`` right-hand sides, plus its NumPy form to solve.

    The operator is **dense** -- ``n ** 2`` triplets, 263 169 of them at the ``n = 513`` the
    boundary test reaches -- and a banded rewrite was measured and declined. Once ``conftest.py``
    caps OpenBLAS's thread pool the whole eight-parametrization family of
    [`test_solve_spd_columns_across_the_reduction_tile_boundary`] is dominated by the first launch's
    kernel load, so there is nothing left to win and a banded operator would only make the reference
    solve less obviously right. Before that cap the same family was 9 of the CPU suite's 80 slowest
    rows -- but the cost was OpenBLAS at 48 threads (``np.linalg.solve`` is orders of magnitude
    slower there), not the triplet count.
    """
    rng = np.random.default_rng(seed)
    dense_np = rng.standard_normal((n, n))
    dense_np = dense_np @ dense_np.T + n * np.eye(n)
    rows_np, cols_np = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    matrix_wp = wps.bsr_from_triplets(
        n,
        n,
        wp.array(rows_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(cols_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(dense_np.ravel()), dtype=wp.float64, device=device),
    )
    assert odt.has_blocks(matrix_wp, wp.float64)
    rhs_np = rng.standard_normal((n_rhs, n))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    return matrix_wp, odt.as_array2d(rhs_wp, wp.float64), dense_np, rhs_np


def _grid_laplacian_system(
    device: str, k: int = 24, n_rhs: int = 3, shift: float = 1e-3, seed: int = 5
) -> tuple[odt.BsrMatrix[wp.float64], wp.array[wp.float64, Any], np.ndarray, np.ndarray]:
    """
    Build a ``k x k`` five-point Laplacian plus a small shift: sparse, SPD, and it coarsens.

    Assembled here rather than assembled from a mesh so the operator's definiteness and sparsity are
    the test's own, not a cotangent-sign convention's. ``k = 24`` puts it above
    ``linalg``'s dense-coarse-solve threshold, so the hierarchy really has a level.
    """
    index = np.arange(k * k).reshape(k, k)
    rows = [index.ravel()]
    columns = [index.ravel()]
    values = [np.full(k * k, 4.0 + shift)]
    for a, b in (
        (index[:-1, :], index[1:, :]),
        (index[1:, :], index[:-1, :]),
        (index[:, :-1], index[:, 1:]),
        (index[:, 1:], index[:, :-1]),
    ):
        rows.append(a.ravel())
        columns.append(b.ravel())
        values.append(np.full(a.size, -1.0))
    rows_np = np.concatenate(rows).astype(np.int32)
    columns_np = np.concatenate(columns).astype(np.int32)
    values_np = np.concatenate(values)
    matrix_wp = wps.bsr_from_triplets(
        k * k,
        k * k,
        wp.array(rows_np, dtype=wp.int32, device=device),
        wp.array(columns_np, dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(values_np), dtype=wp.float64, device=device),
    )
    assert odt.has_blocks(matrix_wp, wp.float64)
    dense_np = np.zeros((k * k, k * k))
    np.add.at(dense_np, (rows_np, columns_np), values_np)
    rng = np.random.default_rng(seed)
    rhs_np = rng.standard_normal((n_rhs, k * k))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    return matrix_wp, odt.as_array2d(rhs_wp, wp.float64), dense_np, rhs_np


@pytest.mark.parity("min_quad_with_fixed", "pymeshlab", "igl")
def test_min_quad_with_fixed_matches_pymeshlab_harmonic_field(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A twice: MeshLab's harmonic field, and the libigl function this one is named after.

    **libigl binds ``min_quad_with_fixed``**, which makes it the direct comparison and not merely a
    library that solves a similar system: it minimizes ``0.5 x' A x + x' B`` under ``x[known] = Y``,
    so at ``B = 0`` with no equality constraints it is this function's problem exactly. The named
    transform is one sign -- ``igl.cotmatrix`` is negative semi-definite, so ``A`` is ``-L`` --
    plus its ``(n_vertices, 1)`` dense return against ordito's free-block-only one, which the
    ``free_map`` scatter already resolves for the MeshLab half.

    Two conventions worth knowing before using it: it returns a **plain array** here rather than the
    tuple its C++ signature suggests, and ``Aeq`` / ``Beq`` are not optional -- an empty
    ``csr_matrix((0, n))`` and an ``(0, 1)`` array are what "no equality constraints" looks like.
    ``igl.min_quad_with_fixed_precompute`` / ``_solve`` are bound too, which is what gives the
    benchmark group a real amortized axis on the reference side.

    MeshLab's harmonic field pins exactly two vertices and solves the same cotangent system
    directly, so pinning the same two to 0 and 1 makes the two answers the same field -- measured to
    1.4e-8. igl is given the identical two pins, so all three agree on one field.
    """
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    n_vertices = vertices_np.shape[0]

    # The two poles, so the field spans the whole mesh rather than a local patch.
    low, high = int(np.argmin(vertices_np[:, 2])), int(np.argmax(vertices_np[:, 2]))
    fixed_np = np.zeros(n_vertices, dtype=bool)
    fixed_np[[low, high]] = True
    values_np = np.zeros((1, n_vertices), dtype=np.float64)
    values_np[0, high] = 1.0

    operator_wp = od.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float64)
    solution_wp, free_map_wp, n_free = od.linalg.min_quad_with_fixed(
        operator_wp,
        wp.array(fixed_np, dtype=wp.bool, device=device),
        odt.as_array2d(
            wp.array(np.ascontiguousarray(values_np), dtype=wp.float64, device=device), wp.float64
        ),
    )
    assert n_free == n_vertices - 2

    # ``min_quad_with_fixed`` returns the free degrees of freedom only; the scatter back through
    # ``free_map`` is the caller's, as its docstring says.
    field_wp = values_np[0].copy()
    free_np = np.flatnonzero(~fixed_np)
    field_wp[free_np] = solution_wp.numpy()[0][free_map_wp.numpy()[free_np]]

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_scalar_harmonic_field_per_vertex(
        point1=np.ascontiguousarray(vertices_np[low]),
        point2=np.ascontiguousarray(vertices_np[high]),
        value1=0.0,
        value2=1.0,
        colorize=False,
    )
    assert np.allclose(
        field_wp, meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-5, atol=1e-5
    )

    # igl: the same problem, with -L for the sign convention and no equality constraints.
    faces_igl = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    field_igl = np.asarray(
        igl.min_quad_with_fixed(
            -igl.cotmatrix(vertices_np, faces_igl),
            np.zeros((n_vertices, 1)),
            np.ascontiguousarray(np.array([low, high], dtype=np.int64)),
            np.array([[0.0], [1.0]]),
            sp.csr_matrix((0, n_vertices)),  # pyright: ignore[reportArgumentType]  # binding says csc
            np.zeros((0, 1)),
            True,
        )
    ).ravel()
    # non-vacuity: the reference produced a real field, not a constant
    assert_nonconstant(field_igl, tol=0.5)
    assert np.allclose(field_wp, field_igl, rtol=1e-5, atol=1e-5)


# --- free_partition / assemble_interior_system ---------------------------------------------


def _pinned_system(device: str, n: int = 12, n_rhs: int = 2, seed: int = 3):
    """Build an SPD operator with three pinned degrees of freedom, plus its NumPy form."""
    matrix_wp, fixed_values_wp, dense_np, values_np = _spd_system(
        device, n=n, n_rhs=n_rhs, seed=seed
    )
    fixed_np = np.zeros(n, dtype=bool)
    fixed_np[[0, 3, 7]] = True
    fixed_wp = wp.array(fixed_np, dtype=wp.bool, device=device)
    return matrix_wp, fixed_wp, fixed_values_wp, dense_np, values_np, fixed_np


def test_free_partition_ranks_the_unpinned_degrees_of_freedom(device: str) -> None:
    """Class A: the map is the rank of each free degree of freedom among the free ones."""
    _matrix_wp, fixed_wp, _values_wp, _dense_np, _rhs_np, fixed_np = _pinned_system(device)

    free_map_wp, n_free = od.linalg.free_partition(fixed_wp)

    assert n_free == int((~fixed_np).sum())
    free_np = np.flatnonzero(~fixed_np)
    # Only the free entries carry meaning, which is what the docstring promises; the pinned slots
    # hold whatever the compaction left there.
    assert np.array_equal(free_map_wp.numpy()[free_np], np.arange(n_free, dtype=np.int32))


def test_assemble_interior_system_matches_a_numpy_partition(device: str) -> None:
    """
    Class A: ``Q_uu`` and ``-Q_ub bc`` equal the dense row/column partition, and ``nnz`` is exact.

    The CSR-to-CSR extraction is the one that replaced a ``bsr_from_triplets`` round trip, and the
    ``nnz`` assert is the quieter half of that change: ``bsr_from_triplets`` would have left the
    count at the *triplet* count (here ``q.nnz``, 144 against the true 81), so every downstream
    ``bsr_mv`` was dimensioned for the unreduced matrix. The count here is exact.

    ``Q_uu`` is compared with ``np.array_equal`` deliberately: the extraction *copies* entries
    rather than recomputing them, so it is bit-exact on both devices (measured ``0.0`` on each). The
    right-hand side accumulates the pinned-column contributions, and there the devices differ --
    exact on CUDA, one ulp out on the CPU backend (8.9e-16 absolute, 1.1e-16 relative), because the
    row scan sums them in a different order. Asserting equality on that half passed here and failed
    the CPU gate.
    """
    matrix_wp, fixed_wp, fixed_values_wp, dense_np, values_np, fixed_np = _pinned_system(device)
    free_map_wp, n_free = od.linalg.free_partition(fixed_wp)

    q_uu, rhs_wp = od.linalg.assemble_interior_system(
        matrix_wp, fixed_wp, free_map_wp, fixed_values_wp, n_free
    )

    free_np = np.flatnonzero(~fixed_np)
    pinned_np = np.flatnonzero(fixed_np)
    assert q_uu.nnz_sync() == n_free * n_free
    assert np.array_equal(bsr_to_dense(q_uu, n_free), dense_np[np.ix_(free_np, free_np)])
    assert np.allclose(
        rhs_wp.numpy(),
        -(dense_np[np.ix_(free_np, pinned_np)] @ values_np[:, pinned_np].T).T,
        rtol=1e-14,
        atol=1e-14,
    )


def test_assemble_interior_system_scales_and_adds_a_load(device: str) -> None:
    """
    Class A: ``scale`` and ``load`` extract ``scale * Q`` and ``b_u - scale * Q_ub bc``.

    ``scale = -1`` is how the parametrization solvers take ``-L``'s block without a negated copy,
    and ``load`` how the pinned Poisson solve compacts its divergence in the extraction's own row
    pass. Both arms are checked against the dense partition: pinned values present (rows must
    match) and absent (zero ``fixed_values`` rows read as every pinned value zero).
    """
    matrix_wp, fixed_wp, fixed_values_wp, dense_np, values_np, fixed_np = _pinned_system(device)
    free_map_wp, n_free = od.linalg.free_partition(fixed_wp)
    n = fixed_np.shape[0]
    rng = np.random.default_rng(11)
    load_np = rng.standard_normal((values_np.shape[0], n))
    load_wp = odt.as_array2d(wp.array(load_np, dtype=wp.float64, device=device), wp.float64)
    free_np = np.flatnonzero(~fixed_np)
    pinned_np = np.flatnonzero(fixed_np)

    q_uu, rhs_wp = od.linalg.assemble_interior_system(
        matrix_wp, fixed_wp, free_map_wp, fixed_values_wp, n_free, scale=-1.0, load=load_wp
    )

    assert np.array_equal(bsr_to_dense(q_uu, n_free), -dense_np[np.ix_(free_np, free_np)])
    expected_np = (
        load_np[:, free_np] + (dense_np[np.ix_(free_np, pinned_np)] @ values_np[:, pinned_np].T).T
    )
    assert np.allclose(rhs_wp.numpy(), expected_np, rtol=1e-14, atol=1e-14)

    no_values = odt.empty_2d((0, n), wp.float64, device=device)
    _q_uu, rhs_wp = od.linalg.assemble_interior_system(
        matrix_wp, fixed_wp, free_map_wp, no_values, n_free, load=load_wp
    )
    assert np.array_equal(rhs_wp.numpy(), load_np[:, free_np])

    one_row = odt.as_array2d(wp.array(load_np[:1], dtype=wp.float64, device=device), wp.float64)
    with pytest.raises(ValueError, match="load has 1 rows"):
        od.linalg.assemble_interior_system(
            matrix_wp, fixed_wp, free_map_wp, fixed_values_wp, n_free, load=one_row
        )


@pytest.mark.parity(
    "solve_spd_columns",
    "numpy",
    benchmarked=False,
    reason="numpy.linalg.solve is a dense LU on the host and this is a batched preconditioned CG "
    "on the device, so a row would compare an O(n^3) factorization with an iterative solve and "
    "report the crossover as a speedup. The *answer* is what is comparable, and a dense solve "
    "is the strongest oracle available for it -- exact up to conditioning.",
)
def test_solve_spd_columns_matches_numpy(device: str) -> None:
    """
    Class A against ``numpy.linalg.solve``, on the default single-column path.

    The reference is a dense LU of the same operator, so this is an exact oracle up to conditioning
    rather than a tolerance dictated by two approximations meeting. The ``1e-5`` bound is the CG
    tolerance's, not the reference's.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "solve_spd_columns",
    "numpy",
    benchmarked=False,
    reason="numpy.linalg.solve is a dense LU on the host and this is a batched preconditioned CG "
    "on the device, so a row would compare an O(n^3) factorization with an iterative solve and "
    "report the crossover as a speedup. The *answer* is what is comparable, and a dense solve "
    "is the strongest oracle available for it -- exact up to conditioning.",
)
@pytest.mark.parametrize("n", [1, 2, 255, 256, 257, 511, 512, 513])
@pytest.mark.parametrize("engine", ["one_block", "batched"])
def test_solve_spd_columns_across_the_reduction_tile_boundary(
    device: str, n: int, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``numpy.linalg.solve``, through both engines a system this small can take.

    The batched solver pads each column to a whole number of
    ``kernels.algorithms.conjugate_gradient.CG_TILE`` entries so its dot has no ragged block, and
    those pad lanes ride through the dot, the fused Jacobi apply and the ``x`` update -- the last
    of which writes the caller's *unpadded* buffer and so must skip them. How much padding there
    is, and therefore which of those three a mistake shows up in, is decided entirely by
    ``n mod CG_TILE``: a value that passes at 256 proves nothing about 257. These straddle the
    boundary in both directions. Every one of them is small enough for the one-block solve
    (``linalg.CG_ONE_BLOCK_MAX_ROWS``), whose lanes stride the rows by the block width instead,
    so the batched arm lowers the gate to reach the padded solver at all.
    """
    if engine == "batched":
        monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n=n)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parity(
    "solve_spd_columns",
    "numpy",
    benchmarked=False,
    reason="numpy.linalg.solve is a dense LU on the host and this is a batched preconditioned CG "
    "on the device, so a row would compare an O(n^3) factorization with an iterative solve and "
    "report the crossover as a speedup. The *answer* is what is comparable, and a dense solve "
    "is the strongest oracle available for it -- exact up to conditioning.",
)
@pytest.mark.parametrize("n_rhs", [1, 2, 5])
def test_solve_spd_columns_agrees_across_column_counts(device: str, n_rhs: int) -> None:
    """
    Class A. One column takes ``warp.optim.linear.cg``; more than one takes ordito's own solver.

    The split is an implementation detail -- a single column has nothing to batch, so Warp already
    reduces it with a tiled tree -- and this is what keeps the two paths answering the same
    question. Without it the batched solver could drift from the reference and only the
    multi-column callers would notice.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n_rhs=n_rhs)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


def test_solve_spd_columns_two_columns_uses_batched_cg(device: str) -> None:
    """
    Ordito against ordito: the dispatch itself, not just the answer it produces.

    Every multi-column solve under ``"diag"`` builds a ``linalg._BatchedCg``, the two-column case
    included. A block conjugate gradient sharing one Krylov subspace across exactly two columns
    was gated in here and removed again (see ``linalg._cg_columns``); a wrong gate would still
    converge to the right answer, so it needs its own assert rather than relying on a value check.
    """
    matrix_wp, rhs_wp, _dense, _rhs = _spd_system(device, n_rhs=2)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = od.linalg.spd_column_solver(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
    assert isinstance(solver, od.linalg._BatchedCg)


@pytest.mark.parametrize("n", [255, 256, 257, 511, 512, 513])
def test_solve_spd_columns_two_columns_across_the_tile_boundary(device: str, n: int) -> None:
    """
    Class A, against ``numpy.linalg.solve``.

    Same boundary ``test_solve_spd_columns_across_the_reduction_tile_boundary`` pins, run again at
    ``n_rhs=2``: the reduction's padded ``stride`` is derived from the column count as well as from
    ``n``, so an even column count exercises a different padding than the ``n_rhs=3`` default every
    other tile-boundary case uses.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n=n, n_rhs=2)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("shift", [1e-3, 1e-7])
def test_two_column_solve_costs_no_more_iterations_than_its_worst_column(
    device: str, shift: float
) -> None:
    """
    Not a library comparison: no reference exposes a solver's iteration count.

    The guard is against a two-column mechanism that *couples* the columns. ``_BatchedCg`` shares
    the launches and leaves each column's Krylov subspace its own, so a two-column solve costs
    exactly the worse of the two single-column solves and can never cost more. A block conjugate
    gradient over a shared subspace does not have that property -- it was measured at 4 774
    iterations against 2 230 on an ill-conditioned system while winning on a well-conditioned one,
    which is why it is gone (``linalg._cg_columns``). The small ``shift`` arm is the
    ill-conditioned member of the pair and is the one that would catch a reintroduction: a
    mechanism validated on the well-conditioned arm alone is exactly the failure this pins.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device, k=40, n_rhs=2, shift=shift)
    kwargs: _CgOptions = {
        "tol": 1e-10,
        "maxiter": 40_000,
        "check_every": 1,
        "preconditioner": "diag",
    }

    both_solution = wp.zeros_like(rhs_wp)
    both_iterations, _residual, _tol = od.linalg._BatchedCg(
        matrix_wp, rhs_wp, odt.as_array2d(both_solution, wp.float64), **kwargs
    )()

    worst_single = 0
    for column in range(2):
        single_rhs = wp.array(
            np.ascontiguousarray(rhs_np[column : column + 1]), dtype=wp.float64, device=device
        )
        single_solution = wp.zeros_like(single_rhs)
        iterations, _residual, _tol = od.linalg._BatchedCg(
            matrix_wp,
            odt.as_array2d(single_rhs, wp.float64),
            odt.as_array2d(single_solution, wp.float64),
            **kwargs,
        )()
        worst_single = max(worst_single, int(cast("int", iterations)))

    assert np.allclose(
        both_solution.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )
    both_iterations = cast("int", both_iterations)  # a host scalar: ``check_every`` is 1
    assert int(both_iterations) == worst_single, (
        f"a two-column solve took {int(both_iterations)} iterations where its worst column alone "
        f"takes {worst_single}: the columns are no longer independent"
    )


def test_two_identical_columns_do_not_diverge(device: str) -> None:
    """
    Not a library comparison: the degenerate two-column block is the input to pin, not a value.

    Two identical right-hand-side columns are the worst case any two-column mechanism can be
    handed. Under ``_BatchedCg`` they are simply the same solve run twice, so this must return the
    single-column answer in both rows and nothing may go non-finite.
    """
    matrix_wp, _rhs_wp, dense_np, rhs_np = _spd_system(device, n_rhs=1)
    rhs_two_np = np.concatenate([rhs_np, rhs_np], axis=0)
    rhs_two_wp = wp.array(np.ascontiguousarray(rhs_two_np), dtype=wp.float64, device=device)
    solution_wp = wp.zeros((2, dense_np.shape[0]), dtype=wp.float64, device=device)
    od.linalg.solve_spd_columns(
        matrix_wp, odt.as_array2d(rhs_two_wp, wp.float64), odt.as_array2d(solution_wp, wp.float64)
    )
    solution_np = solution_wp.numpy()
    assert np.all(np.isfinite(solution_np))
    expected_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_np[0], expected_np[0], rtol=1e-4, atol=1e-4)
    assert np.allclose(solution_np[1], expected_np[0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("check_every", [1, 25, 0])
def test_solve_spd_columns_check_every_is_solution_invariant(device: str, check_every: int) -> None:
    # ``check_every`` only changes how often the residual is tested (``0`` tests it on device via
    # ``wp.capture_while``), never the system being solved, so every setting converges to the same
    # answer. It is a performance knob only.
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), check_every=check_every
    )
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "spd_column_solver_amortized",
    "numpy",
    benchmarked=False,
    reason="same dense-LU-against-iterative-CG mismatch as the solve_spd_columns claims above, and "
    "this group additionally measures *reuse* across calls -- numpy has no hoisted-state form to "
    "amortize, so there is nothing on its side for the amortization axis to time. The answer each "
    "reused call converges to is what is comparable.",
)
def test_spd_column_solver_check_every_reused_across_calls(device: str) -> None:
    # The hoisted functor keeps its ``check_every`` across calls and warm-starts from ``solution``.
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = od.linalg.spd_column_solver(
        matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), check_every=0
    )
    solver()
    solver()
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("check_every", [0, 5])
def test_spd_column_solver_reads_a_rewritten_rhs_on_every_call(
    device: str, check_every: int
) -> None:
    """
    Class A, against ``numpy.linalg.solve``, for a right-hand side rewritten between two calls.

    The state records its device-side loop once and replays it on every later call, and it
    captures ``rhs`` at construction; both are only correct if each call re-reads the buffer. A
    second call against the *same* right-hand side cannot show that -- it is already converged,
    and a replay that ignored the new values would still return the old, correct answer -- so the
    second right-hand side here differs, and the call must run iterations to reach it.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = od.linalg.spd_column_solver(
        matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), check_every=check_every
    )
    solver()
    second_np = np.roll(rhs_np, 1, axis=1) - 0.5 * rhs_np
    rhs_wp.assign(np.ascontiguousarray(second_np))
    iterations, _, _ = solver()
    # Device arrays under ``check_every=0``, host scalars otherwise.
    assert int(iterations.numpy()[0] if isinstance(iterations, wp.array) else iterations) > 0
    solution_np = np.linalg.solve(dense_np, second_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


def test_solve_spd_warns_when_it_runs_out_of_iterations(device: str) -> None:
    """
    A conjugate gradient that stops on ``maxiter`` rather than on ``tol`` says so.

    Without this the returned array is the last iterate and looks exactly like a solution --
    which is how a diverging smoothing pass used to reach a caller silently.
    """
    # A 1-D Laplacian: SPD, but Jacobi-preconditioned CG needs O(n) iterations on it, so a budget
    # of two cannot converge. (A *diagonal* system would be solved exactly in one, and a singular
    # one makes the Jacobi preconditioner itself infinite, which CG bails out of instead.)
    n = 64
    rows, cols, values = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        values.append(2.0)
        if i + 1 < n:
            rows += [i, i + 1]
            cols += [i + 1, i]
            values += [-1.0, -1.0]
    matrix = wps.bsr_from_triplets(
        n,
        n,
        wp.array(np.array(rows, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array(cols, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array(values, dtype=np.float64), dtype=wp.float64, device=device),
    )
    assert odt.has_blocks(matrix, wp.float64)
    rhs = wp.array(np.ones(n, dtype=np.float64), dtype=wp.float64, device=device)
    solution = wp.zeros(n, dtype=wp.float64, device=device)

    with pytest.warns(UserWarning, match="iteration cap"):
        iterations, _, _ = od.linalg.solve_spd(
            matrix, rhs, solution, tol=1e-14, maxiter=2, name="test_solve_spd"
        )
    assert int(iterations) >= 2


def test_solve_spd_is_quiet_when_it_converges(device: str) -> None:
    """The warning is specific to non-convergence: a well-posed solve emits nothing."""
    n = 8
    indices = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device)
    values = wp.array(np.full(n, 2.0, dtype=np.float64), dtype=wp.float64, device=device)
    matrix = wps.bsr_from_triplets(n, n, indices, wp.clone(indices), values)
    assert odt.has_blocks(matrix, wp.float64)
    rhs = wp.array(np.ones(n, dtype=np.float64), dtype=wp.float64, device=device)
    solution = wp.zeros(n, dtype=wp.float64, device=device)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        od.linalg.solve_spd(matrix, rhs, solution, maxiter=10 * n)
    assert np.allclose(solution.numpy(), np.full(n, 0.5))


def test_solve_spd_keeps_one_state_per_operator(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``numpy.linalg.solve``, for two right-hand sides solved against one operator.

    The second solve replays the state the first one built and recorded (``linalg._cached_solver``),
    so what has to hold is that a replayed state reads the *new* right-hand side and starts from
    the *new* initial guess -- a replay that reused the first call's would return the first answer,
    which is why the two right-hand sides differ. A system this small would take the one-block
    solve, which keeps no state, so the gate is lowered to reach the cache.
    """
    monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n_rhs=2)
    for column in range(2):
        solution_wp = wp.zeros(dense_np.shape[0], dtype=wp.float64, device=device)
        od.linalg.solve_spd(matrix_wp, odt.as_dense(rhs_wp[column]), solution_wp)
        assert np.allclose(
            solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np[column]), rtol=1e-5, atol=1e-5
        )
    assert len(od.linalg._SOLVER_CACHE[matrix_wp]) == 1


def test_solver_cache_keys_on_the_operator(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Class A, against ``numpy.linalg.solve``: two operators of one shape do not share a state.

    A state records pointers into its operator's arrays, so one handed a second matrix of the same
    shape would solve the first matrix's system and converge to a plausible wrong answer. The
    one-block gate is lowered so these small systems reach the cache.
    """
    monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    first_wp, rhs_wp, first_np, rhs_np = _spd_system(device, n_rhs=1, seed=3)
    second_wp, _rhs_wp, second_np, _rhs_np = _spd_system(device, n_rhs=1, seed=4)
    for matrix_wp, dense_np in ((first_wp, first_np), (second_wp, second_np)):
        solution_wp = wp.zeros_like(rhs_wp)
        od.linalg.solve_spd_columns(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
        assert np.allclose(
            solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("kind", ["diag", "chebyshev"])
def test_pooled_solver_state_follows_each_operator(
    device: str, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """
    Class A, against ``numpy.linalg.solve``: operators of one shape share a pooled state correctly.

    A fresh operator of a shape already solved is copied into a kept state of that shape rather
    than recorded (``linalg._cached_solver``), so the claims are that three operators alternating
    through it each get their own answer -- a state that kept the previous operator's values or
    preconditioner would converge to that operator's system -- that a value rewritten in place is
    seen on the next solve, and that no solve after the first two records a graph. The
    Jacobi-Chebyshev arm also refits the polynomial on the device. The one-block gate is lowered
    so these small systems reach the cache.
    """
    monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    systems = [_spd_system(device, n_rhs=1, seed=seed) for seed in (31, 32, 33)]

    def solve(
        matrix_wp: odt.BsrMatrix[wp.float64], rhs_wp: wp.array[wp.float64, Any]
    ) -> np.ndarray:
        solution_wp = wp.zeros(int(rhs_wp.shape[1]), dtype=wp.float64, device=device)
        preconditioner = None
        if kind == "chebyshev":
            preconditioner = od.linalg.chebyshev_preconditioner(matrix_wp)
        od.linalg.solve_spd(
            matrix_wp, odt.as_dense(rhs_wp[0]), solution_wp, preconditioner=preconditioner
        )
        return solution_wp.numpy()

    for matrix_wp, rhs_wp, _dense_np, _rhs_np in systems[:2]:
        solve(matrix_wp, rhs_wp)
    captures = 0
    enter = wp.ScopedCapture.__enter__

    def counting_enter(self: wp.ScopedCapture) -> wp.ScopedCapture:
        nonlocal captures
        captures += 1
        return enter(self)

    monkeypatch.setattr(wp.ScopedCapture, "__enter__", counting_enter)
    for matrix_wp, rhs_wp, dense_np, rhs_np in systems + systems:
        expected_np = np.linalg.solve(dense_np, rhs_np[0])
        assert np.allclose(solve(matrix_wp, rhs_wp), expected_np, rtol=1e-6, atol=1e-6)
    matrix_wp, rhs_wp, dense_np, rhs_np = systems[2]
    matrix_wp.values.assign(2.0 * matrix_wp.values.numpy())
    expected_np = np.linalg.solve(2.0 * dense_np, rhs_np[0])
    assert np.allclose(solve(matrix_wp, rhs_wp), expected_np, rtol=1e-6, atol=1e-6)
    if wp.get_device(device).is_cuda and wp.is_conditional_graph_supported():
        assert captures == 0
    if kind == "chebyshev":
        # The refit is a rate, not a fixed point, so the answers above cannot see it: compare the
        # pooled polynomial with the one a new state would fit to the last operator.
        pooled = list(od.linalg._SOLVER_POOL.values())[-1]
        fresh = od.linalg._JacobiChebyshev(matrix_wp).steps.numpy()
        assert isinstance(pooled._cycle, od.linalg._JacobiChebyshevApply)
        assert np.array_equal(pooled._cycle.owner.steps.numpy(), fresh)


def test_pooled_squared_laplacian_state_follows_each_system(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``numpy.linalg.solve``: a squared-Laplacian pooled state follows its callers.

    It must follow both the operator and the caller's preconditioner.
    The preconditioner is built fresh per call against each fresh system, as the region solves do,
    so a kept state is only reached through the pool, which copies the operator *and* the
    preconditioner into the state's own arrays (``_BatchedCg.refresh``). Three systems alternating
    through it must each get their own answer, no solve after the first two may record a graph, and
    -- since a stale preconditioner changes only the rate, which no answer can see -- the pooled
    copy of the polynomial must be bit-for-bit the last caller's. The one-block gate is lowered so
    these small systems reach the pool.
    """
    monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    systems = [_spd_system(device, n_rhs=2, seed=seed) for seed in (41, 42, 43)]
    preconditioners = []

    def solve(matrix_wp: odt.BsrMatrix[wp.float64], rhs_wp: odt.Array2dFloat64) -> np.ndarray:
        weights_wp = wp.full(int(matrix_wp.nrow), 2.0, dtype=wp.float64, device=device)
        preconditioner = od.linalg.squared_laplacian_preconditioner(matrix_wp, weights_wp)
        preconditioners.append(preconditioner)
        solution_wp = odt.as_array2d(
            wp.zeros(rhs_wp.shape, dtype=rhs_wp.dtype, device=rhs_wp.device), wp.float64
        )
        od.linalg.solve_spd_columns(matrix_wp, rhs_wp, solution_wp, preconditioner=preconditioner)
        return solution_wp.numpy()

    for matrix_wp, rhs_wp, _dense_np, _rhs_np in systems[:2]:
        solve(matrix_wp, rhs_wp)
    captures = 0
    enter = wp.ScopedCapture.__enter__

    def counting_enter(self: wp.ScopedCapture) -> wp.ScopedCapture:
        nonlocal captures
        captures += 1
        return enter(self)

    monkeypatch.setattr(wp.ScopedCapture, "__enter__", counting_enter)
    for matrix_wp, rhs_wp, dense_np, rhs_np in systems + systems:
        expected_np = np.linalg.solve(dense_np, rhs_np.T).T
        assert np.allclose(solve(matrix_wp, rhs_wp), expected_np, rtol=1e-6, atol=1e-6)
    if wp.get_device(device).is_cuda and wp.is_conditional_graph_supported():
        assert captures == 0
    pooled = list(od.linalg._SOLVER_POOL.values())[-1]
    assert isinstance(pooled._cycle, od.linalg._SquaredLaplacianApply)
    owned, last = pooled._cycle.owner, preconditioners[-1]
    assert owned is not last
    assert np.array_equal(owned._steps.numpy(), last._steps.numpy())
    for mine, theirs in ((owned._factor, last._factor), (owned._factor_t, last._factor_t)):
        for field in ("offsets", "columns", "values"):
            assert np.array_equal(getattr(mine, field).numpy(), getattr(theirs, field).numpy())


def test_solver_cache_does_not_outlive_its_operator(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: a lifetime is the claim, and no reference caches a solver.

    The cache is keyed weakly and its states hold the operator's arrays rather than the operator,
    so dropping the last reference to the matrix drops its state -- device buffers, preconditioner
    and recorded graph. A state that held the matrix itself would keep its own key alive for ever.
    The one-block gate is lowered so this small system reaches the cache.
    """
    monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    matrix_wp, rhs_wp, _dense_np, _rhs_np = _spd_system(device, n_rhs=1)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64))
    alive = weakref.ref(matrix_wp)
    assert alive() in od.linalg._SOLVER_CACHE
    del matrix_wp
    gc.collect()
    assert alive() is None


def test_solve_spd_block_operator_matches_numpy(device: str) -> None:
    """
    Class A, against ``numpy.linalg.solve`` of the same operator written out densely.

    A ``wp.mat22d`` operator is solved as its scalar expansion over interleaved unknowns, through
    the ``float64`` view of the ``wp.vec2d`` right-hand side; a wrong interleaving convention would
    still converge, to the solution of a permuted system.
    """
    n_blocks = 40
    rng = np.random.default_rng(17)
    dense_np = rng.standard_normal((2 * n_blocks, 2 * n_blocks))
    dense_np = dense_np @ dense_np.T + 2 * n_blocks * np.eye(2 * n_blocks)
    blocks_np = dense_np.reshape(n_blocks, 2, n_blocks, 2).transpose(0, 2, 1, 3)
    rows_np, cols_np = np.meshgrid(np.arange(n_blocks), np.arange(n_blocks), indexing="ij")
    matrix_wp = wps.bsr_from_triplets(
        n_blocks,
        n_blocks,
        wp.array(rows_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(cols_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(blocks_np.reshape(-1, 2, 2)), dtype=wp.mat22d, device=device),
    )
    assert odt.has_blocks(matrix_wp, wp.mat22d)
    rhs_np = rng.standard_normal((n_blocks, 2))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.vec2d, device=device)
    solution_wp = wp.zeros(n_blocks, dtype=wp.vec2d, device=device)
    od.linalg.solve_spd(
        matrix_wp, rhs_wp, solution_wp, preconditioner=od.linalg.jacobi_preconditioner(matrix_wp)
    )
    expected_np = np.linalg.solve(dense_np, rhs_np.ravel()).reshape(n_blocks, 2)
    assert np.allclose(solution_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)


def test_jacobi_preconditioner_applies_the_inverse_diagonal(device: str) -> None:
    """
    Class A, against NumPy's inverse diagonal of the same dense operator.

    The operator builds Warp's inverse diagonal on its first apply rather than when it is built,
    because the solves that recognize it read the diagonal themselves; so applying it directly --
    as a caller handing it to ``warp.optim.linear`` does -- is the one path that reaches the build.
    Applied twice, so the second apply reuses the first one's diagonal.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n_rhs=1)
    operator = od.linalg.jacobi_preconditioner(matrix_wp)
    x_wp = odt.as_dense(rhs_wp[0])
    expected_np = rhs_np[0] / np.diag(dense_np)
    for _ in range(2):
        z_wp = wp.zeros_like(x_wp)
        operator.matvec(x_wp, z_wp, z_wp, 1.0, 0.0)
        assert np.allclose(z_wp.numpy(), expected_np, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("preconditioner", ["diag", "chebyshev"])
def test_unfolded_iteration_matches_numpy(
    device: str, preconditioner: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``numpy.linalg.solve``, on the path a column of over 65 536 unknowns takes.

    Past ``CG_FOLD_MAX_BLOCKS`` tiles a round launches a finalize between the mat-vec and the
    update instead of every update block folding the partials itself. No fixture here is that
    large, so the threshold is lowered to reach it.
    """
    monkeypatch.setattr(od.linalg, "CG_FOLD_MAX_BLOCKS", 0)
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device, n_rhs=2)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(
        matrix_wp,
        rhs_wp,
        odt.as_array2d(solution_wp, wp.float64),
        preconditioner=preconditioner,
        check_every=5,
    )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("heavy", [False, True])
def test_solve_spd_float32_system_matches_numpy(
    device: str, heavy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``numpy.linalg.solve``, for a ``float32`` operator solved in place.

    A ``float32`` system keeps ``float32`` vectors and reduces its dots in ``float64``
    (``reconstruction``'s ``warp.fem`` Poisson system is the caller). The ``heavy`` arm lowers the
    thresholds so the round forms ``A u`` with ``warp.sparse.bsr_mv`` and reduces the dots in a
    launch of their own -- the path for long rows on a long column, which no fixture here reaches.
    The tolerance is ``float32``'s.
    """
    if heavy:
        monkeypatch.setattr(od.linalg, "CG_FOLD_MAX_BLOCKS", 0)
        monkeypatch.setattr(od.linalg, "CG_HEAVY_ROW_ENTRIES", 0)
    matrix64_wp, _rhs, dense_np, _rhs_np = _grid_laplacian_system(device, n_rhs=1)
    matrix_wp = wps.bsr_copy(
        matrix64_wp,
        scalar_type=wp.float32,  # pyright: ignore[reportArgumentType]  # Warp types it an instance
    )
    assert odt.has_blocks(matrix_wp, wp.float32)
    rhs_np = np.random.default_rng(31).standard_normal(dense_np.shape[0]).astype(np.float32)
    rhs_wp = wp.array(rhs_np, dtype=wp.float32, device=device)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd(matrix_wp, rhs_wp, solution_wp, tol=1e-6)
    expected_np = np.linalg.solve(dense_np, rhs_np.astype(np.float64))
    assert np.allclose(solution_wp.numpy(), expected_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shift", [1.0, 1e-4])
def test_adaptive_preconditioner_escalates_only_past_its_probe(device: str, shift: float) -> None:
    """
    Class A, against ``numpy.linalg.solve``, on both branches of ``preconditioner="adaptive"``.

    A heavily shifted grid Laplacian converges inside the Jacobi probe and must never build the
    polynomial; a barely shifted one does not, and must escalate and still reach the answer. The
    iteration counts are the operators' own (tens against thousands of Jacobi rounds), so the
    parametrization reaches each branch by construction; the asserts on ``_escalation`` say so.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device, k=64, n_rhs=2, shift=shift)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = od.linalg.spd_column_solver(
        matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner="adaptive"
    )
    solver()
    assert isinstance(solver, od.linalg._AdaptiveCg)
    assert (solver._escalation is not None) == (shift < 1e-2)
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


# --- replicated_operator ------------------------------------------------------------------


@pytest.mark.parity(
    "multigrid_preconditioner",
    "numpy",
    benchmarked=False,
    reason="nothing installed here binds smoothed-aggregation multigrid -- pyamg is not a "
    "dependency -- so there is no reference *preconditioner* to time. What a preconditioner "
    "cannot do is change the converged answer, and a dense numpy.linalg.solve of the same "
    "operator is the oracle for that; timing it would be the O(n^3)-against-iterative "
    "mismatch the sibling claims record.",
)
def test_multigrid_preconditioner_solves_the_same_system(device: str) -> None:
    """
    Class A, against ``numpy.linalg.solve``: the V-cycle changes the path, not the answer.

    A preconditioner cannot change a converged solution, only how many iterations reach it -- so the
    thing to check is that the multigrid path really does converge to the reference rather than
    stalling somewhere plausible. The Jacobi arm is the one carrying an external oracle (every other
    solver test in this file), and this pins the new path to the same reference.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device)
    reference_np = np.linalg.solve(dense_np, rhs_np.T).T
    for mode in ("diag", "multigrid"):
        solution_wp = wp.zeros_like(rhs_wp)
        od.linalg.solve_spd_columns(
            matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner=mode
        )
        assert np.allclose(solution_wp.numpy(), reference_np, rtol=1e-5, atol=1e-5), mode


def test_multigrid_preconditioner_is_symmetric(device: str) -> None:
    """
    Not a library comparison: no reference here builds a multigrid preconditioner.

    The invariant is the one conjugate gradient actually depends on. ``cg`` is only a valid
    iteration when its preconditioner is symmetric positive definite, and a V-cycle is symmetric
    only if its pre- and post-smoothing are balanced -- drop the post-smoothing to save a mat-vec
    and the solver silently stops being conjugate gradient. So ``<M r1, r2> == <r1, M r2>``, which
    this checks, is what licenses the whole approach; the positive part shows up as convergence in
    the test above. This excludes an unbalanced cycle and a restrictor that is not the
    prolongator's transpose; it does not exclude a hierarchy that is merely a bad one.
    """
    matrix_wp, _rhs, dense_np, _rhs_np = _grid_laplacian_system(device)
    n = dense_np.shape[0]
    operator = od.linalg.multigrid_preconditioner(matrix_wp)
    rng = np.random.default_rng(3)
    left_np, right_np = rng.standard_normal((2, n))
    left = wp.array(np.ascontiguousarray(left_np), dtype=wp.float64, device=device)
    right = wp.array(np.ascontiguousarray(right_np), dtype=wp.float64, device=device)
    applied_left = wp.zeros(n, dtype=wp.float64, device=device)
    applied_right = wp.zeros(n, dtype=wp.float64, device=device)
    operator.matvec(left, applied_left, applied_left, 1.0, 0.0)
    operator.matvec(right, applied_right, applied_right, 1.0, 0.0)
    cross_a = float(applied_left.numpy() @ right_np)
    cross_b = float(left_np @ applied_right.numpy())
    assert np.isclose(cross_a, cross_b, rtol=1e-9, atol=1e-12), (cross_a, cross_b)
    # Non-vacuity: an operator that returned zero, or the identity, would pass the line above.
    assert abs(cross_a) > 1e-6
    assert not np.allclose(applied_left.numpy(), left_np)


@pytest.mark.parametrize("k", [24, 40])
def test_multigrid_galerkin_fallback_builds_the_same_hierarchy(
    device: str, monkeypatch: pytest.MonkeyPatch, k: int
) -> None:
    """
    Ordito against ordito: the triplet products and their ``bsr_mm`` fallback agree.

    The fallback runs only on a level whose coarsening stalled, which no default-threshold system
    in this suite reaches, so it is forced by a zero budget. The triplet path carries the oracle
    (the convergence tests above). Equal shapes and coarse operators to rounding, level by level,
    exclude a fallback that drops the pruning (the coarse levels would gain explicit zeros and
    the shapes would still match, so the patterns are compared too) or multiplies the wrong
    pair.
    """
    matrix_wp, _rhs, _dense_np, _rhs_np = _grid_laplacian_system(device, k=k)
    triplet = od.linalg._multigrid_hierarchy(matrix_wp, 0)
    monkeypatch.setattr(od.linalg, "_MULTIGRID_TRIPLET_FACTOR", 0)
    fallback = od.linalg._multigrid_hierarchy(matrix_wp, 0)
    assert triplet is not None
    assert fallback is not None
    assert len(triplet[0]) == len(fallback[0]) >= 2
    for level_triplet, level_fallback in zip(triplet[0][1:], fallback[0][1:], strict=True):
        dense_triplet = bsr_to_dense(level_triplet.operator, level_triplet.n)
        dense_fallback = bsr_to_dense(level_fallback.operator, level_fallback.n)
        assert dense_triplet.shape == dense_fallback.shape
        assert np.allclose(dense_triplet, dense_fallback, rtol=1e-12, atol=1e-12)
        assert level_triplet.operator.nnz_sync() == level_fallback.operator.nnz_sync()


def test_multigrid_preconditioner_needs_fewer_iterations(device: str) -> None:
    """
    Not a library comparison: this is the *reason* the mode exists, stated as an assertion.

    ordito against ordito -- the Jacobi arm is the reference implementation and carries the
    oracle. Without this the mode could silently degrade to something that still converges (the
    test above would pass) while costing a hierarchy for nothing. The margin is deliberately loose:
    the measured factor on this operator is far above 2x, and the assertion is only meant to catch
    a hierarchy that has stopped working.
    """
    matrix_wp, rhs_wp, _dense, _rhs_np = _grid_laplacian_system(device)
    counts = {}
    for mode in ("diag", "multigrid"):
        solution_wp = wp.zeros_like(rhs_wp)
        counts[mode] = od.linalg.solve_spd_columns(
            matrix_wp,
            rhs_wp,
            odt.as_array2d(solution_wp, wp.float64),
            check_every=1,
            preconditioner=mode,
        )[0]
    assert counts["multigrid"] * 2 < counts["diag"], counts


def test_multigrid_preconditioner_auto_matches_the_forced_modes(device: str) -> None:
    """
    Class A, against ``numpy.linalg.solve``, for the third mode.

    ``"auto"`` is a *policy* over the other two, so what has to hold is that whichever branch it
    takes still ends in a converged solve at the caller's own cap -- the bug it is written against
    is returning the probe's unconverged iterate. This operator converges inside the probe, so it
    exercises the branch that never builds a hierarchy; the escalating branch is what
    ``smoothing.smooth_region`` runs on every ill-conditioned region and what its own tests cover.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
    )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


def test_multigrid_preconditioner_auto_converges_past_the_probe(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: this pins ``"auto"``'s escalating branch, which is the risky one.

    A cap of one iteration forces the probe to end unconverged, so the escalation runs -- and the
    assertion is that the *answer* is the converged one rather than the probe's iterate. That was a
    real defect in the first version of this mode: it returned the capped probe when it decided not
    to escalate, so an ill-conditioned system came back silently wrong by 1.3 in absolute terms.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    monkeypatch.setattr(od.linalg, "CG_PROBE_ITERATIONS", 1)
    od.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
    )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


def test_offdiagonal_dominance_matches_a_numpy_reduction(device: str) -> None:
    """
    Not a library comparison: no reference exposes a Gershgorin ratio, so the oracle is the formula.

    ``linalg._offdiagonal_dominance`` is what ``preconditioner="auto"`` gates on, so a wrong answer
    here does not fail anything -- it silently picks the other preconditioner and costs time. The
    row with a **zero** diagonal is the case worth pinning: it is a free vertex no face refers to,
    and it must contribute ``0`` rather than an infinity, or every scan mesh reads ``inf`` and the
    gate degenerates to "always multigrid".
    """
    rows_np = np.array([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int32)
    columns_np = np.array([0, 1, 2, 1, 0, 2, 0, 2], dtype=np.int32)
    values_np = np.array([2.0, -1.0, -3.0, 4.0, 1.0, 0.5, -1.0, -2.0], dtype=np.float64)
    matrix_wp = wps.bsr_from_triplets(
        4,
        4,
        wp.array(rows_np, dtype=wp.int32, device=device),
        wp.array(columns_np, dtype=wp.int32, device=device),
        wp.array(values_np, dtype=wp.float64, device=device),
    )
    assert odt.has_blocks(matrix_wp, wp.float64)
    dense_np = np.zeros((4, 4))
    np.add.at(dense_np, (rows_np, columns_np), values_np)
    diagonal_np = np.diag(dense_np)
    off_np = np.abs(dense_np).sum(axis=1) - np.abs(diagonal_np)
    # Row 3 has no diagonal entry at all, so it is excluded rather than divided by zero.
    ratios_np = np.where(
        diagonal_np > 0.0, off_np / np.where(diagonal_np > 0.0, diagonal_np, 1.0), 0.0
    )
    assert diagonal_np[3] == 0.0, "the zero-diagonal row is the point of this fixture"
    assert np.allclose(od.linalg._offdiagonal_dominance(matrix_wp), ratios_np.max(), rtol=1e-12)


def test_multigrid_preconditioner_auto_gate_takes_both_branches(device: str) -> None:
    """
    Ordito against ordito: ``"auto"``'s gate against the forced modes it chooses between.

    The oracle is ``numpy.linalg.solve`` on the small system and forced ``"multigrid"`` on the
    large one, which is too big to densify. What this pins is that the gate is *answer-neutral*:
    it only decides which preconditioner runs, and a converged solve is a converged solve either
    way.

    It also runs every branch of the gate, which no other ``"auto"`` test does. All three matter and
    the middle one is the subtle one:

    * **too small** -- the ``k = 24`` grid Laplacian falls under ``CG_MULTIGRID_MIN_UNKNOWNS`` and
      goes to the probe.
    * **large but not dominant enough** -- at ``k = 90`` it is over
      ``CG_MULTIGRID_LARGE_UNKNOWNS`` and still declined, because a five-point Laplacian's rows
      nearly sum to zero (dominance ~1.0, under ``CG_MULTIGRID_SIZE_FLOOR``). That floor exists
      because the size branch was measured **0.41x** on exactly this shape of operator.
    * **dominant** -- squaring that Laplacian squares its condition number and lifts the dominance
      past ``CG_MULTIGRID_DOMINANCE``, which is the ``harmonic`` at ``k = 2`` case the branch is
      for.
    """
    small_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device, k=24)
    assert not od.linalg._wants_multigrid(small_wp), "576 unknowns must fall through to the probe"
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(
        small_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
    )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )

    large_wp, large_rhs_wp, _dense, _rhs = _grid_laplacian_system(device, k=90, n_rhs=1)
    assert od.linalg._offdiagonal_dominance(large_wp) < od.linalg.CG_MULTIGRID_SIZE_FLOOR
    assert not od.linalg._wants_multigrid(large_wp), (
        "a large but weakly-dominant operator must still be declined"
    )

    squared_wp = odt.bsr_mm(large_wp, large_wp)
    assert od.linalg._offdiagonal_dominance(squared_wp) > od.linalg.CG_MULTIGRID_DOMINANCE
    assert od.linalg._wants_multigrid(squared_wp), "the squared operator must clear the gate"
    answers = {}
    for mode in ("auto", "multigrid"):
        answers[mode] = wp.zeros_like(large_rhs_wp)
        od.linalg.solve_spd_columns(
            squared_wp, large_rhs_wp, odt.as_array2d(answers[mode], wp.float64), preconditioner=mode
        )
    assert np.allclose(answers["auto"].numpy(), answers["multigrid"].numpy(), rtol=1e-5, atol=1e-5)


def test_spd_column_solver_rejects_the_auto_preconditioner(device: str) -> None:
    # A hoisted state is built to be driven many times, and the probe decides on the first solve --
    # so "auto" has no meaning there and says so rather than quietly picking one of the two.
    matrix_wp, rhs_wp, _dense, _rhs_np = _grid_laplacian_system(device, k=8)
    solution_wp = wp.zeros_like(rhs_wp)
    with pytest.raises(ValueError, match="auto"):
        od.linalg.spd_column_solver(
            matrix_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
        )


def test_multigrid_preconditioner_falls_back_when_the_operator_does_not_coarsen(
    device: str,
) -> None:
    """
    Not a library comparison: a fallback has no counterpart to compare against.

    A diagonal operator has no off-diagonal graph, so every row is its own aggregate and coarsening
    stalls at the first level. Above the dense-factorization cap there is then no usable hierarchy,
    and the documented behaviour is to hand back a Jacobi preconditioner rather than a cycle whose
    coarse solve is a guess -- which for a diagonal operator is the *exact* inverse, so the solve
    converges in one iteration. That is what makes this case checkable at all.
    """
    n = 1024
    rng = np.random.default_rng(7)
    diagonal_np = rng.uniform(1.0, 4.0, size=n)
    index_np = np.arange(n, dtype=np.int32)
    matrix_wp = wps.bsr_from_triplets(
        n,
        n,
        wp.array(index_np, dtype=wp.int32, device=device),
        wp.array(index_np, dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(diagonal_np), dtype=wp.float64, device=device),
    )
    assert odt.has_blocks(matrix_wp, wp.float64)
    rhs_np = rng.standard_normal((2, n))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    solution_wp = wp.zeros_like(rhs_wp)
    iterations = od.linalg.solve_spd_columns(
        matrix_wp,
        odt.as_array2d(rhs_wp, wp.float64),
        odt.as_array2d(solution_wp, wp.float64),
        check_every=1,
        preconditioner="multigrid",
    )[0]
    assert iterations == 1, iterations
    assert np.allclose(solution_wp.numpy(), rhs_np / diagonal_np, rtol=1e-8, atol=1e-10)


def _normal_equations_system(
    device: str, *, negative: bool, k: int = 28, n_rhs: int = 3, seed: int = 13
) -> tuple[
    odt.BsrMatrix[wp.float64],
    wp.array[wp.float64, Any],
    odt.BsrMatrix[wp.float64],
    wp.array[wp.float64],
    np.ndarray,
    np.ndarray,
]:
    """
    Build ``smoothing.smooth_region``'s least-squares umbrella system on a ``k x k`` grid graph.

    Rows are the free vertices (the grid's interior below its top two rows) plus their first fixed
    ring; a row is ``p_v - sum_d w_vd p_d / sum_d w_vd`` with the fixed terms moved to the right, so
    the free rows alone are ``D^-1 L`` for ``L`` the free block of the weighted graph Laplacian.
    ``negative`` gives one diagonal of every grid square a weight of ``-0.2``, the shape a clamped
    cotangent weight takes on a near-right triangle, which pushes ``D^-1 L``'s spectrum past 2.

    Returns the assembled ``M^T M``, the right-hand side, ``L``, the weight sums, and the dense
    forms of the operator and the right-hand side for a reference solve.
    """
    index = np.arange(k * k).reshape(k, k)
    edges, weights = [], []
    for a, b, w in (
        (index[:-1, :], index[1:, :], 1.0),
        (index[:, :-1], index[:, 1:], 1.0),
        (index[:-1, :-1], index[1:, 1:], -0.2 if negative else 0.5),
    ):
        edges.append(np.stack([a.ravel(), b.ravel()], axis=1))
        weights.append(np.full(a.size, w))
    edges_np, weights_np = np.concatenate(edges), np.concatenate(weights)
    n_nodes = k * k
    w_np = sp.coo_matrix(
        (np.concatenate([weights_np] * 2), (edges_np.ravel("F"), edges_np[:, ::-1].ravel("F"))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    sums_np = np.asarray(w_np.sum(axis=1)).ravel()
    free_np = np.zeros((k, k), dtype=bool)
    free_np[2:, :] = True
    free_np = free_np.ravel()
    ring_np = ~free_np & (np.asarray(w_np[:, free_np].astype(bool).sum(axis=1)).ravel() > 0)
    free_index = np.flatnonzero(free_np)
    rank = np.full(n_nodes, -1)
    rank[free_index] = np.arange(free_index.size)
    rows_np = np.flatnonzero(free_np | ring_np)
    umbrella = sp.identity(n_nodes, format="csr") - sp.diags(1.0 / sums_np) @ w_np
    m_np = umbrella[rows_np][:, free_index].toarray()
    positions_np = np.random.default_rng(seed).standard_normal((n_nodes, n_rhs))
    positions_np[free_np] = 0.0
    b_np = cast("np.ndarray", -(umbrella[rows_np] @ positions_np))
    system_np, rhs_np = m_np.T @ m_np, (m_np.T @ b_np).T
    laplacian_np = cast("sp.csr_matrix", sp.diags(sums_np) - w_np)[free_index][
        :, free_index
    ].tocoo()

    def upload(matrix_np: sp.coo_matrix) -> odt.BsrMatrix[wp.float64]:
        matrix_wp = wps.bsr_from_triplets(
            matrix_np.shape[0],
            matrix_np.shape[1],
            wp.array(matrix_np.row.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(matrix_np.col.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(np.ascontiguousarray(matrix_np.data), dtype=wp.float64, device=device),
        )
        assert odt.has_blocks(matrix_wp, wp.float64)
        return matrix_wp

    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    sums_wp = wp.array(np.ascontiguousarray(sums_np[free_index]), dtype=wp.float64, device=device)
    return (
        upload(sp.coo_matrix(system_np)),
        odt.as_array2d(rhs_wp, wp.float64),
        upload(laplacian_np),
        sums_wp,
        system_np,
        rhs_np,
    )


@pytest.mark.parametrize("negative", [False, True], ids=["positive_weights", "negative_weights"])
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize("engine", ["one_block", "batched"])
def test_squared_laplacian_preconditioner_solves_the_same_system(
    device: str, negative: bool, n_columns: int, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``numpy.linalg.solve``: the preconditioner changes the path, not the answer.

    Parametrized over the weight sign because a negative weight is what takes ``D^-1 L``'s spectrum
    past 2, the end of a fixed Chebyshev interval, over the column count because this
    preconditioner routes a single column through the batched solver too, and over the two engines
    a system this small can take: the one-block solve applies the same polynomial in-block, and
    lowering its gate reaches the batched one.
    """
    if engine == "batched":
        monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    system_wp, rhs_wp, laplacian_wp, sums_wp, system_np, rhs_np = _normal_equations_system(
        device, negative=negative, n_rhs=n_columns
    )
    reference_np = np.linalg.solve(system_np, rhs_np.T).T
    assert np.ptp(reference_np) > 1e-3  # non-vacuity: a zero solve would pass trivially
    solution_wp = wp.zeros_like(rhs_wp)
    od.linalg.solve_spd_columns(
        system_wp,
        rhs_wp,
        odt.as_array2d(solution_wp, wp.float64),
        preconditioner=od.linalg.squared_laplacian_preconditioner(laplacian_wp, sums_wp),
    )
    assert np.allclose(solution_wp.numpy(), reference_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("negative", [False, True], ids=["positive_weights", "negative_weights"])
@pytest.mark.parametrize("engine", ["one_block", "batched"])
def test_squared_laplacian_preconditioner_needs_far_fewer_iterations(
    device: str, negative: bool, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: the reason the preconditioner exists, stated as an assertion.

    ordito against ordito; the Jacobi arm carries the oracle through the test above. Measured
    19x and 27x fewer iterations than Jacobi. The negative-weight arm is the one that matters: with
    the polynomial's interval ending at a fixed 2 rather than at the operator's Gershgorin bound it
    still converges -- the preconditioner stays positive definite -- but in 340 iterations against
    37, a third of Jacobi's count rather than a twenty-seventh, so only a count catches it. Run
    through both engines, since the one-block solve applies the polynomial with code of its own.
    """
    if engine == "batched":
        monkeypatch.setattr(od.linalg, "CG_ONE_BLOCK_MAX_ROWS", 0)
    system_wp, rhs_wp, laplacian_wp, sums_wp, _system_np, _rhs_np = _normal_equations_system(
        device, negative=negative
    )
    counts = {}
    for name, preconditioner in (
        ("diag", "diag"),
        ("squared", od.linalg.squared_laplacian_preconditioner(laplacian_wp, sums_wp)),
    ):
        solution_wp = wp.zeros_like(rhs_wp)
        counts[name] = od.linalg.solve_spd_columns(
            system_wp,
            rhs_wp,
            odt.as_array2d(solution_wp, wp.float64),
            check_every=1,
            preconditioner=preconditioner,
        )[0]
    assert counts["squared"] * 8 < counts["diag"], counts


def _chebyshev_reference(dense_np: np.ndarray, rhs_np: np.ndarray) -> np.ndarray:
    """
    ``p(D^-1 A) D^-1 rhs`` in closed form, for the interval ``chebyshev_preconditioner`` picks.

    The semi-iteration's degree-``d`` iterate from zero is ``x_d = p(D^-1 A) D^-1 b`` with
    ``1 - lambda p(lambda) = T_d((theta - lambda) / delta) / T_d(theta / delta)``. Evaluated here on
    the eigendecomposition of the symmetric ``D^-1/2 A D^-1/2``, which shares ``D^-1 A``'s spectrum,
    so nothing of the three-term recurrence is reused.
    """
    diagonal = np.diag(dense_np)
    n = diagonal.size
    radius = max(
        float(np.max((np.abs(dense_np).sum(axis=1) - np.abs(diagonal)) / np.abs(diagonal))), 1e-3
    )
    lower = max(
        min(od.linalg.CHEBYSHEV_INTERVAL / n, od.linalg.SQUARED_LAPLACIAN_INTERVAL_CAP),
        1.0 - radius,
    )
    upper = 1.0 + radius
    theta, delta = 0.5 * (upper + lower), 0.5 * (upper - lower)
    degree = od.linalg.CHEBYSHEV_DEGREE
    root = 1.0 / np.sqrt(np.abs(diagonal))
    sign = np.sign(diagonal[0])
    eigenvalues, vectors = np.linalg.eigh(sign * root[:, None] * dense_np * root[None, :])
    chebyshev = np.polynomial.chebyshev.Chebyshev.basis(degree)
    residual = chebyshev((theta - eigenvalues) / delta) / chebyshev(theta / delta)
    polynomial = (1.0 - residual) / eigenvalues
    # ``D^-1/2 p(S) D^-1/2`` with the diagonal's sign restored.
    return (
        sign
        * root[:, None]
        * (vectors @ (polynomial[:, None] * (vectors.T @ (root[:, None] * rhs_np.T))))
    )


@pytest.mark.parametrize("negative", [False, True], ids=["positive_weights", "negative_weights"])
def test_chebyshev_preconditioner_applies_the_chebyshev_polynomial(
    device: str, negative: bool
) -> None:
    """
    Not a library comparison: no reference builds this preconditioner, so the oracle is its formula.

    The apply is checked against ``p(D^-1 A) D^-1`` evaluated in closed form on an eigenbasis
    (``_chebyshev_reference``), which shares nothing with the three-term recurrence the kernel runs.
    That is what catches a recurrence that computes *some* polynomial. A first version read the
    semi-iteration's first iterate as ``source`` rather than ``source / theta``: put back, it fails
    both arms here and passes every solve and iteration-count test in this file, because the wrong
    polynomial still preconditions -- it only goes negative inside the interval once the Gershgorin
    bound moves away from 2, which is when a solve stalls. The negative-weight arm is the one whose
    bound exceeds 2. ``<r, M r> > 0`` is the definiteness conjugate gradient needs.
    """
    _system, _rhs, laplacian_wp, _sums, _system_np, _rhs_np = _normal_equations_system(
        device, negative=negative
    )
    dense_np = bsr_to_dense(laplacian_wp, int(laplacian_wp.nrow))
    n = dense_np.shape[0]
    rhs_np = np.random.default_rng(11).standard_normal((1, n))
    reference_np = _chebyshev_reference(dense_np, rhs_np).T
    operator = od.linalg.chebyshev_preconditioner(laplacian_wp)
    source = wp.array(np.ascontiguousarray(rhs_np[0]), dtype=wp.float64, device=device)
    applied = wp.zeros(n, dtype=wp.float64, device=device)
    operator.matvec(source, applied, applied, 1.0, 0.0)
    assert np.allclose(applied.numpy(), reference_np[0], rtol=1e-8, atol=1e-10)
    assert np.ptp(reference_np) > 1e-3  # non-vacuity
    # Symmetric, and definite with the operator's sign: <r, M r> > 0 for a positive diagonal.
    assert float(rhs_np[0] @ applied.numpy()) > 0.0


@pytest.mark.parametrize("negative", [False, True], ids=["positive_weights", "negative_weights"])
@pytest.mark.parametrize("n_columns", [1, 3])
def test_chebyshev_preconditioner_solves_the_same_system(
    device: str, negative: bool, n_columns: int
) -> None:
    """
    Class A, against ``numpy.linalg.solve``: the preconditioner changes the path, not the answer.

    Through both entry points -- the ``"chebyshev"`` name the column solvers take, which routes a
    single column through the batched solver too, and the operator
    [`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner] returns for
    ``solve_spd``.
    """
    _system, _rhs, laplacian_wp, _sums, _system_np, _rhs_np = _normal_equations_system(
        device, negative=negative
    )
    dense_np = bsr_to_dense(laplacian_wp, int(laplacian_wp.nrow))
    n = dense_np.shape[0]
    rhs_np = np.random.default_rng(12).standard_normal((n_columns, n))
    reference_np = np.linalg.solve(dense_np, rhs_np.T).T
    rhs_wp = odt.as_array2d(
        wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device), wp.float64
    )
    solution_wp = wp.zeros(rhs_wp.shape, dtype=rhs_wp.dtype, device=rhs_wp.device)
    od.linalg.solve_spd_columns(
        laplacian_wp, rhs_wp, odt.as_array2d(solution_wp, wp.float64), preconditioner="chebyshev"
    )
    assert np.allclose(solution_wp.numpy(), reference_np, rtol=1e-5, atol=1e-5)
    single = wp.zeros(n, dtype=wp.float64, device=device)
    od.linalg.solve_spd(
        laplacian_wp,
        odt.as_dense(rhs_wp[0]),
        single,
        preconditioner=od.linalg.chebyshev_preconditioner(laplacian_wp),
    )
    assert np.allclose(single.numpy(), reference_np[0], rtol=1e-5, atol=1e-5)


def test_chebyshev_preconditioner_needs_far_fewer_iterations(device: str) -> None:
    """
    Not a library comparison: the reason the preconditioner exists, stated as an assertion.

    ordito against ordito; the Jacobi arm carries the oracle through the test above. Each
    iteration costs ``CHEBYSHEV_DEGREE`` extra launches, so a preconditioner that did not cut the
    count several-fold would be a loss on every system, and a solve test would not notice.
    """
    matrix_wp, rhs_wp, _dense_np, _rhs_np = _grid_laplacian_system(device)
    counts = {}
    for name in ("diag", "chebyshev"):
        solution_wp = wp.zeros_like(rhs_wp)
        counts[name] = od.linalg.solve_spd_columns(
            matrix_wp,
            rhs_wp,
            odt.as_array2d(solution_wp, wp.float64),
            check_every=1,
            preconditioner=name,
        )[0]
    assert counts["chebyshev"] * 4 < counts["diag"], counts


def test_block_diag_matches_scipy(device: str) -> None:
    """
    Class A, against ``scipy.sparse.block_diag`` of the same operators written out densely.

    The stack mixes a ``wp.mat22d`` operator -- entering as its scalar expansion over interleaved
    unknowns -- with two scalar ones, one of them a duplicate-accumulating triplet build whose
    ``nnz`` is a stale capacity (CLAUDE.md 3.7): a stack sized off that field would place the
    second block's rows at the wrong entries. A repeated call returns the same matrix, which is what
    lets a solve against it replay its recorded loop.
    """
    rng = np.random.default_rng(29)
    n_blocks = 6
    dense_block_np = rng.standard_normal((2 * n_blocks, 2 * n_blocks))
    blocks_np = dense_block_np.reshape(n_blocks, 2, n_blocks, 2).transpose(0, 2, 1, 3)
    rows_np, cols_np = np.meshgrid(np.arange(n_blocks), np.arange(n_blocks), indexing="ij")
    vector_wp = wps.bsr_from_triplets(
        n_blocks,
        n_blocks,
        wp.array(rows_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(cols_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(blocks_np.reshape(-1, 2, 2)), dtype=wp.mat22d, device=device),
    )
    assert odt.has_blocks(vector_wp, wp.mat22d)
    n = 9
    rows = rng.integers(0, n, 40).astype(np.int32)
    cols = rng.integers(0, n, 40).astype(np.int32)
    vals = rng.standard_normal(40)
    scalar_wp = wps.bsr_from_triplets(
        n,
        n,
        wp.array(rows, dtype=wp.int32, device=device),
        wp.array(cols, dtype=wp.int32, device=device),
        wp.array(vals, dtype=wp.float64, device=device),
    )
    assert odt.has_blocks(scalar_wp, wp.float64)
    # Duplicates, so the triplet build's ``nnz`` really is a capacity.
    assert len(set(zip(rows.tolist(), cols.tolist(), strict=True))) < rows.size
    scalar_np = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()

    stacked_wp = od.linalg.block_diag((vector_wp, scalar_wp, scalar_wp))

    expected_np = sp.block_diag((dense_block_np, scalar_np, scalar_np)).toarray()
    assert stacked_wp.nrow == expected_np.shape[0]
    assert np.allclose(
        bsr_to_dense(stacked_wp, expected_np.shape[0]), expected_np, rtol=1e-12, atol=1e-12
    )
    assert od.linalg.block_diag((vector_wp, scalar_wp, scalar_wp)) is stacked_wp


def test_block_diag_rejects_a_rectangular_operator(device: str) -> None:
    """Not a library comparison: a rectangular block has no place on a diagonal."""
    rectangular_wp = wps.bsr_zeros(
        3,
        4,
        wp.float64,  # pyright: ignore[reportArgumentType]  # Warp types it a BlockType instance
        device=device,
    )
    assert odt.has_blocks(rectangular_wp, wp.float64)
    with pytest.raises(ValueError, match="square"):
        od.linalg.block_diag((rectangular_wp,))
    with pytest.raises(ValueError, match="at least one"):
        od.linalg.block_diag(())


@pytest.mark.parametrize("n_columns", [1, 3])
def test_replicated_operator_applies_the_base_to_each_block(device: str, n_columns: int) -> None:
    """
    Class A: the batched ``matvec`` equals applying the single operator to each contiguous block.

    Nothing is replicated in memory, so what has to be pinned is that the flat layout really is
    ``n_columns`` independent subproblems -- and that ``n_columns == 1`` hands back the unwrapped
    operator, which the docstring calls out as the ``lscm`` path. Measured agreement 1.4e-14.
    """
    n = 12
    matrix_wp, _values_wp, dense_np, _rhs_np = _spd_system(device, n=n, n_rhs=1, seed=5)

    operator = od.linalg.replicated_operator(matrix_wp, n_columns)

    assert operator.shape == (n_columns * n, n_columns * n)
    # A single column is not batched at all: no ``batch_offsets``, so no per-block residual.
    assert (operator.batch_offsets is None) == (n_columns == 1)

    vector_np = np.random.default_rng(23).standard_normal(n_columns * n)
    vector_wp = wp.array(np.ascontiguousarray(vector_np), dtype=wp.float64, device=device)
    result_wp = wp.zeros(n_columns * n, dtype=wp.float64, device=device)
    operator.matvec(vector_wp, result_wp, result_wp, alpha=1.0, beta=0.0)

    expected_np = np.concatenate(
        [dense_np @ vector_np[column * n : (column + 1) * n] for column in range(n_columns)]
    )
    assert np.allclose(result_wp.numpy(), expected_np, rtol=1e-10, atol=1e-10)
