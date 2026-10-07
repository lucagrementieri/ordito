"""
Benchmarks for ``ordito.parametrization``.

Two axes, because these solvers have two independent cost drivers and mesh size is neither outright:

* **patch** — ``saddle_small`` / ``saddle`` / ``hemisphere``, the disk-topology inputs these solvers
  are actually for, flat and curved. This is the size sweep, restricted to meshes that *have* a
  single boundary loop to pin.
* **quality** — ``saddle`` against ``saddle_graded``: same vertices, faces and boundary, differing
  only in the spacing along one axis, which pushes the worst aspect ratio from 1.6 to 4 719.
  Cotangent weights go large and the free-block condition number with them, so the CG iteration
  count moves while nothing else does. That is the cleanest available measurement of conditioning
  cost, and it is invisible to any face-count sweep.

Setup that is *not* part of the measured operation is precomputed and cached: the boundary loop, its
circle map, and the harmonic warm start ARAP iterates from. What stays inside the timed callable is
operator assembly plus the CG solve, because that is what the batched-CG work targets.

References
----------
**libigl** runs on every mesh in both axes. It goes through a direct LDLT factorization of the
cotangent system, which fails outright on the scanned registry meshes (``Failed to compute harmonic
map`` / ``igl::lscm failed``): they are not disk topology and their cotangent Laplacian is not
positive definite on the free set. The patch and quality meshes are disk topology by construction,
so the comparison is drawn on exactly the meshes ordito is measured on. That also makes the quality
axis a genuine A/B between an iterative and a direct solver: CG pays for conditioning in iterations,
LDLT in fill-in, and they need not move together.

``rim_long`` is deliberately **not** used. It is an annulus, and pinning only one of its rims leaves
ARAP free to fold: ordito returns 32 767 flipped faces out of 131 072, disagrees with libigl by
0.35 regardless of CG tolerance (the two land on different local minima of a non-convex energy), and
burns 8 200 CG iterations per solve on the resulting near-singular system. Timing that measures a
pathology, not the algorithm.

**open3d** has no mesh parametrization at all. **pymeshlab** covers ``harmonic`` and ``lscm``, and
its own filter descriptions say why it is a *second* reference rather than a third implementation:
both state that they use "the original code provided in the libigl library". So they wrap the solver
the ``igl`` rows call directly, and the gap between them is MeshLab's boundary detection, attribute
plumbing and MeshSet build rather than a different algorithm. Worth having — it prices what a
*library wrapper* adds over the bare call — but not independent evidence, and it should not be read
as such.

**Its ``harm_function`` parameter is documented as ordito's ``k`` (1 harmonic, 2 biharmonic) and is
a no-op in pymeshlab 2025.7**: ``harm_function=1``, ``2`` and ``3`` return **bit-identical** texture
coordinates at identical cost, where libigl's own ``k=2`` costs several times its ``k=1``. So the
harmonic order axis does not map and the pymeshlab row appears at ``k=1`` only. A row tracking
ordito's ``k=2`` would be silently reporting the ``k=1`` solve.

LSCM takes no parameters at all: MeshLab pins the boundary condition itself rather than accepting a
pin set, so unlike ordito's two-pin call there is nothing to match. And **it rejects closed meshes
outright** — a boundary loop is required — which is why a pymeshlab parametrization row can never
move to the scan sweep or the ``scale`` axis. Both filters rewrite the per-vertex texture
coordinates, so the MeshSet is rebuilt per round; on the ``patch`` axis that build is a small share.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp

import ordito as od
from conftest import BenchCase

# Local/global alternations for ARAP. The pair brackets "converged early" against "ran the full
# schedule": each iteration is a per-face SVD pass plus a CG solve, so the slope is the per-
# iteration cost and the intercept is the operator assembly.
_ARAP_ITERATIONS = [3, 10]

# Harmonic order. k=2 is the bilaplacian: the operator is squared, so it is both denser and far
# worse conditioned than k=1 at identical mesh size -- a second conditioning handle alongside the
# quality axis, and one that moves nnz as well.
_HARMONIC_ORDERS = [1, 2]

_boundary_cache: dict[tuple[str, str], tuple[wp.array[wp.int32], wp.array[wp.vec2]]] = {}
_warm_start_cache: dict[tuple[str, str], wp.array[wp.vec2]] = {}


def _boundary(bench_case: BenchCase) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Longest boundary loop and its unit-circle UV, cached per ``(mesh, device)``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _boundary_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop = od.boundary.longest_boundary_loop(vertices, faces)
        _boundary_cache[key] = (loop, od.parametrization.map_vertices_to_circle(vertices, loop))
    return _boundary_cache[key]


def _warm_start(bench_case: BenchCase) -> wp.array[wp.vec2]:
    """Harmonic UV used as the ARAP initial guess, cached per ``(mesh, device)``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _warm_start_cache:
        loop, loop_uv = _boundary(bench_case)
        _warm_start_cache[key] = od.parametrization.harmonic(
            bench_case.vertices_wp, bench_case.faces_wp, loop, loop_uv
        )
    return _warm_start_cache[key]


def _igl_boundary(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices, faces_i64, loop)`` for the libigl branches."""
    faces_np = bench_case.faces_np.astype(np.int64)
    return bench_case.vertices_np, faces_np, np.asarray(igl.boundary_loop(faces_np))


@pytest.mark.benchmark(group="map_vertices_to_circle")
@pytest.mark.benchaxis("patch")
@pytest.mark.benchlibs("ordito", "igl")
def test_map_vertices_to_circle(bench_case: BenchCase) -> None:
    """Arc-length parametrization of the rim: driven by loop length, not by mesh size."""
    if bench_case.kind == "ordito":
        vertices = bench_case.vertices_wp
        loop, _loop_uv = _boundary(bench_case)
        circle = bench_case.run(lambda: od.parametrization.map_vertices_to_circle(vertices, loop))
        assert circle.size == loop.size
    else:
        vertices_np, _faces_np, loop_np = _igl_boundary(bench_case)
        circle_igl = bench_case.run(lambda: igl.map_vertices_to_circle(vertices_np, loop_np))
        assert circle_igl.shape[0] == loop_np.size


def _run_harmonic_pml(bench_case: BenchCase, order: int) -> None:
    """
    Time MeshLab's harmonic parametrization at ``k = 1``, the only order it actually honours.

    ``harm_function`` is a no-op in pymeshlab 2025.7 -- see the module docstring for the
    measurement. Higher orders are skipped rather than run, so the table cannot show a ``k=2`` row
    that is really solving ``k=1``.
    """
    if order != min(_HARMONIC_ORDERS):
        pytest.skip("MeshLab's harm_function is a no-op in 2025.7: identical output at every order")
    bench_case.run(
        lambda: bench_case.new_meshset_pml().compute_texcoord_parametrization_harmonic(
            harm_function=order
        )
    )


@pytest.mark.noparity(
    "pymeshlab",
    oracle="igl",
    reason="D1 not an independent implementation: MeshLab documents "
    "compute_texcoord_parametrization_harmonic as using the original code from the libigl "
    "library, so it wraps the very solver the igl row calls directly. Asserting against it would "
    "re-check libigl through a second wrapper. Its harm_function parameter is additionally a "
    "no-op in pymeshlab 2025.7, so the harmonic order axis does not map either.",
)
@pytest.mark.benchmark(group="harmonic")
@pytest.mark.benchaxis("patch")
@pytest.mark.benchlibs("ordito", "igl", "pymeshlab")
@pytest.mark.parametrize("order", _HARMONIC_ORDERS)
def test_harmonic(bench_case: BenchCase, order: int) -> None:
    """
    Fixed-boundary harmonic map, at the Laplacian and the much stiffer bilaplacian.

    **Declined (2026-10-07): solving ``k=2`` by a sparse Cholesky of the reduced system.** It read
    1.5-2.7x faster than the iteration only while ``ordito.cholesky`` kept each pattern's analysis
    across calls (a cache ``.claude/CLAUDE.md`` section 16.16 rejects); with the analysis paid by
    the call a factored solve is 4.9x / 7.4x / 6.6x slower than iterating on ``saddle_small`` /
    ``saddle`` / ``hemisphere``. Its pattern depends on the pinned set, so a ``Trimesh`` could reuse
    it only for a repeated call with the same boundary. Open lead: on ``saddle_graded`` (not on
    this axis) the ``k=2`` iteration is 114x slower than a factorization with its analysis.
    """
    if bench_case.kind == "pymeshlab":
        _run_harmonic_pml(bench_case, order)
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, loop_uv = _boundary(bench_case)
        uv = bench_case.run(
            lambda: od.parametrization.harmonic(vertices, faces, loop, loop_uv, k=order)
        )
        assert uv.size == vertices.size
    else:
        vertices_np, faces_np, loop_np = _igl_boundary(bench_case)
        circle_np = igl.map_vertices_to_circle(vertices_np, loop_np)
        uv_igl = bench_case.run(
            lambda: igl.harmonic(vertices_np, faces_np, loop_np, circle_np, order)
        )
        assert uv_igl.shape[0] == vertices_np.shape[0]


@pytest.mark.noparity(
    "pymeshlab",
    oracle="igl",
    reason="D1 not an independent implementation: the same libigl-wrapping harmonic filter as the "
    "harmonic group above, so asserting against it would re-check libigl through a second "
    "wrapper rather than add evidence. igl is the oracle for the conditioning axis too.",
)
@pytest.mark.benchmark(group="harmonic_conditioning")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "igl", "pymeshlab")
def test_harmonic_conditioning(bench_case: BenchCase) -> None:
    """
    The same harmonic solve on the same connectivity, well- and ill-conditioned.

    ``saddle`` and ``saddle_graded`` have identical vertex counts, face arrays and boundary loops;
    only the spacing differs. Any gap between these two rows is conditioning and nothing else --
    for ordito, CG iterations; for libigl, LDLT fill-in. The pymeshlab row wraps libigl's own
    solver, so it should track the ``igl`` row's *shape* and differ only by a constant; a pair that
    diverges here would mean MeshLab's boundary detection is doing something size-dependent.
    """
    if bench_case.kind == "pymeshlab":
        _run_harmonic_pml(bench_case, 1)
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, loop_uv = _boundary(bench_case)
        uv = bench_case.run(lambda: od.parametrization.harmonic(vertices, faces, loop, loop_uv))
        assert uv.size == vertices.size
    else:
        vertices_np, faces_np, loop_np = _igl_boundary(bench_case)
        circle_np = igl.map_vertices_to_circle(vertices_np, loop_np)
        uv_igl = bench_case.run(lambda: igl.harmonic(vertices_np, faces_np, loop_np, circle_np, 1))
        assert uv_igl.shape[0] == vertices_np.shape[0]


@pytest.mark.benchmark(group="arap")
@pytest.mark.benchaxis("patch")
@pytest.mark.benchlibs("ordito", "igl")
@pytest.mark.parametrize("iterations", _ARAP_ITERATIONS)
def test_arap(bench_case: BenchCase, iterations: int) -> None:
    """Local/global ARAP from a harmonic warm start: per-face SVD plus a CG solve per iteration."""
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, loop_uv = _boundary(bench_case)
        uv_init = _warm_start(bench_case)
        uv = bench_case.run(
            lambda: od.parametrization.arap(
                vertices, faces, loop, loop_uv, uv_init, max_iterations=iterations
            )
        )
        assert uv.size == vertices.size
    else:
        vertices_np, faces_np, loop_np = _igl_boundary(bench_case)
        circle_np = igl.map_vertices_to_circle(vertices_np, loop_np)
        uv_init_np = np.ascontiguousarray(
            igl.harmonic(vertices_np, faces_np, loop_np, circle_np, 1)
        )

        # ordito's ``arap`` rebuilds its operator on every call, so the igl side includes
        # ``arap_precomputation`` for a like-for-like comparison rather than solve-only.
        def run() -> np.ndarray:
            data = igl.ARAPData()
            data.max_iter = iterations
            igl.arap_precomputation(vertices_np, faces_np, 2, loop_np.astype(np.int32), data)
            return np.asarray(igl.arap_solve(circle_np, data, uv_init_np))

        uv_igl = bench_case.run(run)
        assert uv_igl.shape[0] == vertices_np.shape[0]


@pytest.mark.noparity(
    "pymeshlab",
    oracle="igl",
    reason="D1 not an independent implementation: MeshLab's least-squares conformal maps filter "
    "also wraps libigl, the same solver the igl row calls. It additionally pins the boundary "
    "condition itself rather than accepting a pin set, so ordito's two-pin call has nothing to "
    "match there.",
)
@pytest.mark.benchmark(group="lscm")
@pytest.mark.benchaxis("patch")
@pytest.mark.benchlibs("ordito", "igl", "pymeshlab")
def test_lscm(bench_case: BenchCase) -> None:
    """
    Free-boundary conformal map: two pins, so the free block is nearly the whole system.

    **Declined (2026-10-07): a sparse Cholesky of the reduced system**, for the reason
    ``test_harmonic`` gives: 1.6-2.8x only with the analysis cached across calls, and 7.7x / 10x /
    17x slower than the iteration on ``saddle_small`` / ``saddle`` / ``hemisphere`` (4.5x on
    ``saddle_graded``) with the analysis paid by the call.
    """
    if bench_case.kind == "pymeshlab":  # MeshLab picks its own pins; there is no pin set to pass
        bench_case.run(
            lambda: (
                bench_case.new_meshset_pml()
            ).compute_texcoord_parametrization_least_squares_conformal_maps()
        )
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, _loop_uv = _boundary(bench_case)
        # LSCM needs only enough pins to kill the similarity freedom: two opposite loop vertices.
        loop_np = loop.numpy()
        pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32)
        pins = wp.array(pins_np, dtype=wp.int32, device=bench_case.device)
        pins_uv = wp.array(
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            dtype=wp.vec2,
            device=bench_case.device,
        )
        uv = bench_case.run(lambda: od.parametrization.lscm(vertices, faces, pins, pins_uv))
        assert uv.size == vertices.size
    else:
        vertices_np, faces_np, loop_np = _igl_boundary(bench_case)
        pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int64)
        pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        uv_igl, _hessian = bench_case.run(
            lambda: igl.lscm(vertices_np, faces_np, pins_np, pins_uv_np)
        )
        assert uv_igl.shape[0] == vertices_np.shape[0]
