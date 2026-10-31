"""
Benchmarks for ``ordito.smoothing``.

Four groups over two axes, because this module has two genuinely different cost regimes and the
switch between them is a keyword argument rather than a mesh property:

* ``filter_mut_dif_laplacian`` on **scale** — the explicit branch. One sparse mat-vec per iteration
  plus a diffusion-coefficient recomputation, so cost is ``iterations * nnz``: linear, predictable,
  and the reason this group carries the reference comparison.
* ``filter_laplacian`` on **quality**, sweeping ``implicit_time_integration`` — the regime change.
  Explicit is the same cheap SpMV loop; **implicit is a full preconditioned CG solve per
  iteration**, whose count depends on the cotangent system's conditioning. Running that on
  ``saddle`` against ``saddle_graded`` (identical connectivity, worst aspect ratio 1.6 against
  4 719) puts both regimes and both conditionings in one table, which is where the cost of choosing
  implicit becomes visible.
* ``filter_taubin`` and ``filter_humphrey`` on **scale** — the two shrinkage-controlled variants of
  the same loop, each a small constant multiple of ``filter_laplacian``.

``iterations`` is deliberately not swept: in the explicit branch it is exactly linear by
construction, and in the implicit branch the interesting variation is *within* an iteration, which
the quality axis already supplies. The Laplacian operator is precomputed outside the timed callable
wherever the signature accepts one, so the timing isolates the loop from the assembly that
[`test_laplacian.py`](test_laplacian.py) covers.

References
----------
**open3d**'s ``filter_smooth_laplacian`` runs the same number of uniform-weight iterations, so it is
the reference for the ``novol`` case; it has no volume-constraint variant, so ``vol`` stays
ordito/trimesh only. Open3D returns a new mesh so the shared one is reusable; **trimesh** mutates
in place and is rebuilt inside the timed callable. Neither has an implicit smoother, so that half of
the ``quality`` group is a ordito-only before/after.

**pymeshlab** carries four of MeshLab's ``apply_coord_*`` smoothers, giving each explicit scheme a
second independent implementation: ``apply_coord_laplacian_smoothing_scale_dependent`` is the same
Desbrun scale-dependent umbrella as ``filter_mut_dif_laplacian``;
``apply_coord_laplacian_smoothing(cotangentweight=False)`` the same uniform explicit loop (MeshLab
has no implicit variant, so it appears in the ``explicit`` row only);
``apply_coord_taubin_smoothing`` the same lambda-mu alternation at MeshLab's ``mu=-0.53`` against
ordito's ``nu=0.5``; and ``apply_coord_hc_laplacian_smoothing`` MeshLab's HC, in its own
``filter_humphrey_meshlab`` group against ``method="meshlab"``.

Two caveats govern every row:

- **All of them mutate the coordinates**, so the MeshSet is rebuilt inside the timed callable and
  the row carries the build — about a third of a ten-step Laplacian row on a scan mesh, so subtract
  it before quoting a ratio.
- **HC Laplacian exposes no parameters at all** — no step count, no ``alpha``/``beta`` — so its row
  calls the filter once per pass. It matches the default ``filter_humphrey`` at none of 8 x 11 x
  11 ``(iterations, alpha, beta)`` combinations, because it adds the neighbours' mean correction
  where Vollmer et al. subtract it; ``method="meshlab"`` is that update, at ``alpha=0`` and
  ``beta=0.5``.

``filter_two_step`` is on **quality** for a different reason from ``filter_laplacian``: it is
*three* nested loops (outer passes x normal diffusion x vertex fitting, 3 x 20 x 20 at MeshLab's
defaults), so its cost is a fixed 1 200 passes over the adjacency whatever the mesh, and the axis
confirms that triangle shape does not change it. ``filter_normals`` times the inner half alone,
separating the normal diffusion from the fitting solve. Both have pymeshlab references at the same
four parameters.

The last group leaves positions alone and runs over a per-vertex **scalar** field:
``filter_scalar_laplacian`` against ``apply_scalar_smoothing_per_vertex``, on **scale** at a fixed
pass count. Its old neighbour, the Lipschitz projection of the same kind of field, moved to
[`test_graph.py`](test_graph.py) as ``shortest_path_envelope`` with the function — a weighted graph
relaxation whose pass count is data-dependent is a different cost shape from a fixed budget.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, face_bitset_ml, mesh_ml_from_numpy, skip_larger_than

if TYPE_CHECKING:
    from typing_extensions import Buffer

_ITERATIONS = 10

_operator_cache: dict[tuple[str, str, bool], odt.BsrMatrix[wp.float32]] = {}
_scalar_cache: dict[tuple[str, str], wp.array[wp.float32]] = {}


def _laplacian_operator(
    bench_case: BenchCase, *, symmetric: bool = False
) -> odt.BsrMatrix[wp.float32]:
    """Assemble the uniform Laplacian once per (mesh, device) -- an *input*, not the operation."""
    key = (bench_case.mesh_name, str(bench_case.device), symmetric)
    if key not in _operator_cache:
        _operator_cache[key] = od.laplacian.laplacian(
            bench_case.vertices_wp, bench_case.faces_wp, symmetric=symmetric
        )
    return _operator_cache[key]


def _skip_pml_beyond_bunny(bench_case: BenchCase) -> None:
    """Cap the pymeshlab smoothers at ``bunny``, the way the trimesh rows are capped."""
    skip_larger_than(bench_case, "bunny", "MeshLab's serial smoothers take seconds beyond bunny")


@pytest.mark.noparity(
    "open3d",
    oracle="trimesh",
    reason="D2 a different diffusion scheme with no mutable coefficient: filter_smooth_laplacian "
    "diffuses at a fixed rate toward the inverse-distance-weighted 1-ring mean, where "
    "filter_mut_dif_laplacian scales each vertex's rate by how much of its Laplacian residual lies "
    "along the normal. Measured 0.0428 max-coordinate deviation on a noisy icosphere(2) of extent "
    "2.02 at 10 iterations. trimesh is the oracle for this group, in "
    "tests/test_smoothing.py::test_filter_mut_dif_laplacian. Recorded so it is "
    "not re-derived: open3d's filter IS ordito's filter_laplacian under an inverse-distance "
    "operator, matching it to 6.6e-08 at *one* iteration and diverging to 0.032 by ten only "
    "because open3d re-derives the edge weights from the current positions every pass while "
    "ordito holds the assembled operator fixed.",
)
@pytest.mark.noparity(
    "pymeshlab",
    oracle="trimesh",
    reason="D2 Desbrun et al.'s scale-dependent umbrella, not Barroqueiro et al.'s mutable "
    "diffusion: apply_coord_laplacian_smoothing_scale_dependent weights the 1-ring by edge length "
    "and exposes no per-vertex rate at all, and its step count is its only parameter. Measured "
    "0.171 max-coordinate deviation on a noisy icosphere(2) of extent 2.02 at 10 iterations -- 4x "
    "further from ordito than open3d's row is. trimesh is the oracle for this group, in "
    "tests/test_smoothing.py::test_filter_mut_dif_laplacian.",
)
@pytest.mark.benchmark(group="filter_mut_dif_laplacian")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parametrize("volume_constraint", [False, True], ids=["novol", "vol"])
def test_filter_mut_dif_laplacian(bench_case: BenchCase, volume_constraint: bool) -> None:
    """The explicit SpMV loop on the scan sweep: linear in iterations, linear in nnz."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pymeshlab":
        if volume_constraint:
            pytest.skip("MeshLab's scale-dependent Laplacian has no volume-constraint variant")
        _skip_pml_beyond_bunny(bench_case)
        # Desbrun et al.'s scale-dependent umbrella: the same scheme, mutating coordinates, so the
        # MeshSet is rebuilt per round and the row carries the build.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_laplacian_smoothing_scale_dependent(
                stepsmoothnum=_ITERATIONS
            )
        )
        return
    if bench_case.kind == "open3d":
        if volume_constraint:
            pytest.skip("open3d has no volume-constrained Laplacian smoother")
        mesh_o3d = bench_case.mesh_o3d
        smoothed = bench_case.run(
            lambda: mesh_o3d.filter_smooth_laplacian(number_of_iterations=_ITERATIONS)
        )
        assert len(smoothed.vertices) == bench_case.n_vertices
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: od.smoothing.filter_mut_dif_laplacian(
                vertices,
                faces,
                iterations=_ITERATIONS,
                volume_constraint=volume_constraint,
                laplacian_operator=operator,
            )
        )
        assert result.shape == vertices.shape
    else:  # trimesh mutates the mesh in place: rebuild it inside the timed callable
        skip_larger_than(bench_case, "bunny", "trimesh CPU loop takes tens of seconds on dragon")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            # trimesh's reference normalizes vertex normals via a scipy-sparse divide that
            # hits 1/0 on degenerate (zero-length) normals; the result is finite and
            # unused, so silence the third-party warning rather than let it leak.
            with np.errstate(divide="ignore", invalid="ignore"):
                tm.smoothing.filter_mut_dif_laplacian(
                    mesh, iterations=_ITERATIONS, volume_constraint=volume_constraint
                )
            return mesh

        result = bench_case.run(run)
        assert result.vertices.shape == vertices.shape


@pytest.mark.benchmark(group="filter_laplacian_integration")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab", "meshlib")
@pytest.mark.parametrize("implicit", [False, True], ids=["explicit", "implicit"])
def test_filter_laplacian_integration(bench_case: BenchCase, implicit: bool) -> None:
    """
    Explicit SpMV against a per-iteration CG solve, on well- and ill-conditioned connectivity.

    Four rows per table: the explicit pair should be flat across the two meshes (an SpMV does not
    care about aspect ratio) and the implicit pair should not (CG does). A flat implicit pair would
    mean the solve is not actually conditioning-bound, which is worth knowing either way.

    pymeshlab confirms the explicit half independently: it is flat across the mesh pair to within
    noise where ordito is orders of magnitude faster and equally flat -- the same statement its
    harmonic-field row makes in [`test_linalg.py`](test_linalg.py) about where the conditioning cost
    actually lives.
    """
    if bench_case.kind == "meshlib":
        if implicit:
            pytest.skip("relax is an explicit per-iteration pass: no backward-Euler variant")
        # ``relax`` mutates the mesh and returns a status, so the mesh is rebuilt per round. Its
        # ``force`` is ordito's ``lamb``, both 0.5 by default, so on a closed mesh the two rows
        # compute the same positions; on these open patches a rim vertex averages both rim
        # neighbours here and one of them under the ordito row's directed operator.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        iterations = _ITERATIONS

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            params_ml = mm.MeshRelaxParams()
            params_ml.iterations = iterations
            mm.relax(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml) > 0
        return
    if bench_case.kind == "pymeshlab":
        if implicit:
            pytest.skip("MeshLab has no implicit / backward-Euler Laplacian smoother")
        # ``cotangentweight=False`` to match ordito's uniform-weight operator.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_laplacian_smoothing(
                stepsmoothnum=_ITERATIONS, cotangentweight=False
            )
        )
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    operator = _laplacian_operator(bench_case)
    result = bench_case.run(
        lambda: od.smoothing.filter_laplacian(
            vertices,
            faces,
            iterations=_ITERATIONS,
            implicit_time_integration=implicit,
            volume_constraint=False,
            laplacian_operator=operator,
        )
    )
    assert result.shape == vertices.shape


@pytest.mark.benchmark(group="filter_laplacian_pyvista")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "pyvista")
def test_filter_laplacian_pyvista(bench_case: BenchCase) -> None:
    """
    VTK's in-place sweep: one thread in index order there, a dataflow wavefront on CUDA here.

    The wave count follows the numbering, not the size: about 15 waves a sweep on a scrambled
    numbering, a scan's longest increasing neighbour path on top (bunny 977, dragon 1 301 at 10
    sweeps), each a launch; the CPU device runs VTK's serial loop. ``relaxation_factor=0.5`` is
    ordito's default ``lamb``, so both rows compute the same positions.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        smoothed_pv = bench_case.run(
            lambda: mesh_pv.smooth(
                n_iter=_ITERATIONS,
                relaxation_factor=0.5,
                boundary_smoothing=True,
                feature_smoothing=False,
            )
        )
        assert smoothed_pv.n_points == bench_case.n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: od.smoothing.filter_laplacian(
            vertices, faces, iterations=_ITERATIONS, volume_constraint=False, method="pyvista"
        )
    )
    assert result.shape == vertices.shape


@pytest.mark.benchmark(group="filter_taubin")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pymeshlab")
def test_filter_taubin(bench_case: BenchCase) -> None:
    """
    The lambda-nu alternation: two SpMVs per iteration instead of one.

    Against ``filter_laplacian_integration``'s explicit row this measures exactly the second pass,
    so the two rows should sit at a ratio near 2 and nothing else should separate them. All four
    libraries implement Taubin's 1995 scheme; MeshLab's and open3d's inflating step is ``mu=-0.53``
    against ordito's and trimesh's ``nu=0.53``, which changes the fixed point but not the work per
    pass.

    **MeshLab's ``stepsmoothnum`` and open3d's ``number_of_iterations`` count lambda-mu pairs, not
    half-steps**, where ordito and trimesh do one half-step per ``iterations`` and alternate. So
    both get ``_ITERATIONS // 2``: passing ``_ITERATIONS`` to both times twice the passes. The
    MeshLab mapping is pinned exactly in
    ``tests/test_smoothing.py::test_filter_taubin_matches_pymeshlab``. open3d's filter re-derives
    its inverse-distance weights from the current positions every half-pass, so its tested pair is
    ``filter_taubin(recompute=True)`` (``test_filter_taubin_recompute_matches_open3d``), while this
    row times the fixed operator: read the open3d ratio as a per-pass cost of one 1-ring sweep, not
    as the same filter.

    **pytorch3d is deliberately absent, and it is the one reference here that agrees with ordito.**
    ``ops.taubin_smoothing`` rebuilds its inverse-distance operator from the current positions every
    half-pass, and ``filter_taubin(recompute=True)`` matches it closely -- so this is not a
    D2 exemption, it is a pair whose *tested* configuration is not this row's. Timing it here would
    race ten sparse assemblies against one under a single group name: the recompute path is an order
    of magnitude dearer than this row, which is the whole ratio rather than a caveat on it. The
    comparison lives in
    ``tests/test_smoothing.py::test_filter_taubin_recompute_matches_pytorch3d`` as a
    ``benchmarked=False`` claim.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: od.smoothing.filter_taubin(
                vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
            )
        )
        assert result.shape == vertices.shape
    elif bench_case.kind == "open3d":
        _skip_pml_beyond_bunny(bench_case)
        mesh_o3d = bench_case.mesh_o3d
        smoothed_o3d = bench_case.run(
            lambda: mesh_o3d.filter_smooth_taubin(
                number_of_iterations=_ITERATIONS // 2, lambda_filter=0.5, mu=-0.53
            )
        )
        assert len(smoothed_o3d.vertices) == bench_case.n_vertices
    elif bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_taubin_smoothing(
                stepsmoothnum=_ITERATIONS // 2
            )
        )
    else:  # trimesh mutates in place: rebuild inside the timed callable
        _skip_pml_beyond_bunny(bench_case)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.smoothing.filter_taubin(mesh, iterations=_ITERATIONS)
            return mesh

        assert bench_case.run(run).vertices.shape == vertices.shape


@pytest.mark.benchmark(group="filter_windowed_sinc")
@pytest.mark.benchlibs("ordito", "pyvista")
def test_filter_windowed_sinc(bench_case: BenchCase) -> None:
    """
    VTK's windowed-sinc filter (``pyvista``'s ``smooth_taubin``) at its defaults: 20 passes.

    One umbrella-Laplacian pass per polynomial degree, plus the stencil analysis (a unique-edge
    sort and the neighbour lists) once per call. VTK's filter is multi-threaded.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        smoothed_pv = bench_case.run(lambda: mesh_pv.smooth_taubin(n_iter=20, pass_band=0.1))
        assert smoothed_pv.n_points == bench_case.n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: od.smoothing.filter_windowed_sinc(vertices, faces, iterations=20, pass_band=0.1)
    )
    assert result.shape == vertices.shape


@pytest.mark.benchmark(group="filter_neighborhood_average")
@pytest.mark.benchlibs("ordito", "open3d")
def test_filter_neighborhood_average(bench_case: BenchCase) -> None:
    """
    The plain 1-ring mean, ``_ITERATIONS`` passes, operator assembly included on both sides.

    open3d's ``filter_smooth_simple`` is the same filter (each neighbour weighted equally, the
    vertex itself not at all) and its test oracle; it builds its adjacency per call as ordito
    assembles its operator per call.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "open3d":
        mesh_o3d = bench_case.mesh_o3d
        smoothed_o3d = bench_case.run(
            lambda: mesh_o3d.filter_smooth_simple(number_of_iterations=_ITERATIONS)
        )
        assert len(smoothed_o3d.vertices) == bench_case.n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    smoothed = bench_case.run(
        lambda: od.smoothing.filter_neighborhood_average(vertices, faces, iterations=_ITERATIONS)
    )
    assert smoothed.shape == vertices.shape


@pytest.mark.benchmark(group="filter_humphrey")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_filter_humphrey(bench_case: BenchCase) -> None:
    """
    HC filtering: a Laplacian pass plus a push-back toward the original positions.

    trimesh is the parameter-comparable reference of the default ``method="vollmer"``; MeshLab's
    update has the opposite sign on one term and is timed in ``filter_humphrey_meshlab``.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: od.smoothing.filter_humphrey(
                vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
            )
        )
        assert result.shape == vertices.shape
    else:  # trimesh mutates in place: rebuild inside the timed callable
        _skip_pml_beyond_bunny(bench_case)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.smoothing.filter_humphrey(mesh, iterations=_ITERATIONS)
            return mesh

        assert bench_case.run(run).vertices.shape == vertices.shape


@pytest.mark.benchmark(group="filter_humphrey_meshlab")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_filter_humphrey_meshlab(bench_case: BenchCase) -> None:
    """
    ``method="meshlab"`` against MeshLab's HC Laplacian, the same number of passes on each side.

    MeshLab's filter is one parameterless pass (``alpha=0``, ``beta=0.5``), so its row calls it
    ``_ITERATIONS`` times on one MeshSet and the ordito row runs that many passes.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        new_meshset_pml = bench_case.new_meshset_pml

        def hc_pml() -> None:
            meshset_pml = new_meshset_pml()
            for _ in range(_ITERATIONS):
                meshset_pml.apply_coord_hc_laplacian_smoothing()

        bench_case.run(hc_pml)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    operator = _laplacian_operator(bench_case, symmetric=True)
    result = bench_case.run(
        lambda: od.smoothing.filter_humphrey(
            vertices,
            faces,
            alpha=0.0,
            beta=0.5,
            iterations=_ITERATIONS,
            laplacian_operator=operator,
            method="meshlab",
        )
    )
    assert result.shape == vertices.shape


@pytest.mark.benchmark(group="filter_implicit_fairing")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_filter_implicit_fairing(bench_case: BenchCase) -> None:
    """
    Implicit fairing, whose operator is rebuilt every pass rather than reused.

    The distinguishing cost against ``filter_laplacian_integration``'s implicit row: the cotangent
    weights depend on the positions the step moves, so each pass re-assembles the stiffness and mass
    matrices before solving. Assembly is therefore *inside* the timed loop by construction, not a
    setup cost that could be hoisted.

    Both meshes on this axis are open patches, so this measures the default ``pin_boundary=True``:
    the rim is held and each pass is a Dirichlet problem over the interior, which converges however
    many passes are applied. Hence the assertion of *no* non-convergence warning below -- if this
    group ever starts warning, the flow has stopped being well posed.

    ``pin_boundary=False`` is deliberately **not** a row here. The unconstrained flow pulls the rim
    inward until the triangles there collapse, and past two passes the system is effectively
    singular: the conjugate gradient runs to its ``maxiter`` cap and returns its last iterate, so
    timing it measures a failed solve -- orders of magnitude dearer than the pinned row, and
    minutes of suite time spent measuring divergence at high precision. A collapsing free case can
    drive ``cot_entries_from_l2``'s division by ``4 * dbl_area`` to ``inf`` and return NaN.

    **A sparse Cholesky refactored per pass, chosen by the first pass (2026-10-07).** Every pass's
    system shares one pattern, so once the first pass's Jacobi-Chebyshev solve takes at least
    ``_FAIRING_FACTOR_ITERATIONS`` (60) rounds the call does one analysis and refactors per pass.
    Nothing is kept across calls. CUDA, 10 passes, min of 5 interleaved: 2.02x on
    ``saddle_graded`` (115 -> 57 ms; the first pass takes ~170 rounds there), 0.99-1.00x on
    ``saddle_small`` / ``saddle`` / ``hemisphere``, where the rule never fires (~20 rounds).
    Factoring from the first pass instead is 2.58x on ``saddle_graded`` and 0.29-0.43x elsewhere.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a non-converged pass fails the benchmark
        result = bench_case.run(
            lambda: od.smoothing.filter_implicit_fairing(vertices, faces, iterations=_ITERATIONS),
            rounds=3,
        )
    assert result.shape == vertices.shape
    assert np.isfinite(result.numpy()).all()


def _spike_field_np(bench_case: BenchCase) -> np.ndarray:
    """Build a delta at vertex 0: the steepest field there is, so saturation travels furthest."""
    values_np = np.zeros(bench_case.n_vertices, dtype=np.float64)
    values_np[0] = 10.0
    return values_np


def _scalar_field_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    """Upload the same spike as a device ``float32`` buffer, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _scalar_cache:
        _scalar_cache[key] = wp.array(
            _spike_field_np(bench_case).astype(np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _scalar_cache[key]


def _new_scalar_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """Build a fresh MeshSet carrying the spike as its vertex scalar attribute."""
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            bench_case.vertices_np,
            np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
            v_scalar_array=_spike_field_np(bench_case),
        )
    )
    return meshset_pml


@pytest.mark.benchmark(group="filter_scalar_laplacian")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_filter_scalar_laplacian(bench_case: BenchCase) -> None:
    """Ten diffusion passes over a scalar field, against MeshLab's single full-step pass."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)

        def smooth_pml() -> int:
            meshset_pml = _new_scalar_meshset_pml(bench_case)
            meshset_pml.apply_scalar_smoothing_per_vertex()
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(smooth_pml) == n_vertices
        return
    values, vertices, faces = (
        _scalar_field_wp(bench_case),
        bench_case.vertices_wp,
        bench_case.faces_wp,
    )
    operator = _laplacian_operator(bench_case)
    smoothed = bench_case.run(
        lambda: od.smoothing.filter_scalar_laplacian(
            values, vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
        )
    )
    assert smoothed.shape == (n_vertices,)


# MeshLab's two-step defaults, used verbatim on both sides: 3 outer passes, a 60-degree crease
# threshold, 20 normal-diffusion steps and 20 fitting steps.
_TWO_STEP_OUTER = 3
_TWO_STEP_NORMAL_THRESHOLD = 60.0
_TWO_STEP_NORMAL_STEPS = 20
_TWO_STEP_FIT_STEPS = 20


@pytest.mark.benchmark(group="filter_normals")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab", "meshlib")
def test_filter_normals(bench_case: BenchCase) -> None:
    """
    The crease-gated normal diffusion alone: 20 scatter passes over the face adjacency.

    meshlib's ``denoiseNormals`` is a **different formulation** of the same job -- an L1
    minimization over the face graph regularized by ``gamma``, against ordito's gated diffusion --
    so the two are not iteration-for-iteration comparable and neither parameter maps to the other.
    What ``tests/test_smoothing.py`` establishes is that both recover the same clean normal field;
    this row prices the two routes to it. Its per-edge weight array is the *input* and is built
    outside the timed callable, but the normals it mutates have to be rebuilt per round -- it edits
    them in place, so rounds 2..n would otherwise denoise an already-denoised field.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        weights_ml = mm.UndirectedEdgeScalars()
        weights_ml.resize(mesh_ml.topology.undirectedEdgeSize(), 1.0)

        def denoise_ml() -> mm.FaceNormals:
            normals_ml = mm.computePerFaceNormals(mesh_ml)
            mm.denoiseNormals(mesh_ml, normals_ml, weights_ml, 20.0)
            return normals_ml

        assert bench_case.run(denoise_ml).size() == n_faces
        return
    if bench_case.kind == "pymeshlab":
        # ``apply_normal_smoothing_per_face`` is one pass with no crease gate, ordito's pass at
        # ``threshold=180`` (``tests/test_smoothing.py``), so its row calls it as many times as
        # the ordito row runs passes; a gated pass costs what an ungated one does. It writes the
        # face normal attribute and leaves the coordinates alone, so the MeshSet is shared.
        meshset_pml = bench_case.meshset_pml

        def smooth_pml() -> None:
            for _ in range(_TWO_STEP_NORMAL_STEPS):
                meshset_pml.apply_normal_smoothing_per_face()

        bench_case.run(smooth_pml)
        assert meshset_pml.current_mesh().face_normal_matrix().shape == (n_faces, 3)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    normals = bench_case.run(
        lambda: od.smoothing.filter_normals(
            vertices, faces, iterations=_TWO_STEP_NORMAL_STEPS, threshold=_TWO_STEP_NORMAL_THRESHOLD
        )
    )
    assert normals.shape == (n_faces,)


@pytest.mark.benchmark(group="filter_two_step")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_filter_two_step(bench_case: BenchCase) -> None:
    """Normal diffusion plus vertex fitting, at MeshLab's own 3 x 20 x 20 defaults on both sides."""
    _bench_two_step(bench_case, "edge_neighbors")


@pytest.mark.benchmark(group="filter_two_step_meshlab")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_filter_two_step_meshlab(bench_case: BenchCase) -> None:
    """
    ``method="meshlab"``: the same answer as the pymeshlab row, vertex for vertex.

    Its normal pass walks the vertex -> face rows (every face sharing a vertex, of lower index)
    where the default reads three edge mates.
    """
    _bench_two_step(bench_case, "meshlab")


def _bench_two_step(bench_case: BenchCase, method: Literal["edge_neighbors", "meshlab"]) -> None:
    """Time ``filter_two_step`` at ``method``, or MeshLab's filter, at MeshLab's defaults."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        new_meshset_pml = bench_case.new_meshset_pml  # mutates the coordinates

        def two_step_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.apply_coord_two_steps_smoothing(
                stepsmoothnum=_TWO_STEP_OUTER,
                normalthr=_TWO_STEP_NORMAL_THRESHOLD,
                stepnormalnum=_TWO_STEP_NORMAL_STEPS,
                stepfitnum=_TWO_STEP_FIT_STEPS,
            )
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(two_step_pml, rounds=3) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    smoothed = bench_case.run(
        lambda: od.smoothing.filter_two_step(
            vertices,
            faces,
            iterations=_TWO_STEP_OUTER,
            threshold=_TWO_STEP_NORMAL_THRESHOLD,
            normal_iterations=_TWO_STEP_NORMAL_STEPS,
            fit_iterations=_TWO_STEP_FIT_STEPS,
            method=method,
        ),
        rounds=3,
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.benchmark(group="filter_normals_meshlib")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_filter_normals_meshlib(bench_case: BenchCase) -> None:
    """
    ``method="meshlib"`` at its defaults: ten face-system solves with the indicator between.

    meshlib's side is the same loop through its two public steps, ``denoiseNormals`` from the
    original normals then ``updateIndicatorFast``, which is ``meshDenoiseViaNormals``' normal half
    (Class A in ``tests/test_smoothing.py``). Its normals and indicator are rebuilt per round: both
    are mutated in place.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()  # read only

        def denoise_ml() -> mm.FaceNormals:
            weights_ml = mm.UndirectedEdgeScalars()
            weights_ml.resize(mesh_ml.topology.undirectedEdgeSize(), 1.0)
            normals_ml = mm.computePerFaceNormals(mesh_ml)
            for _ in range(10):
                normals_ml = mm.computePerFaceNormals(mesh_ml)
                mm.denoiseNormals(mesh_ml, normals_ml, weights_ml, 5.0)
                mm.updateIndicatorFast(mesh_ml.topology, weights_ml, normals_ml, 0.01, 5.0)
            return normals_ml

        assert bench_case.run(denoise_ml, rounds=3).size() == n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    normals = bench_case.run(
        lambda: od.smoothing.filter_normals(vertices, faces, method="meshlib"), rounds=3
    )
    assert normals.shape == (n_faces,)


@pytest.mark.benchmark(group="filter_two_step_meshlib")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_filter_two_step_meshlib(bench_case: BenchCase) -> None:
    """
    ``method="meshlib"`` against ``meshDenoiseViaNormals``, both at the reference's defaults.

    Ten normal solves, then twenty least-squares vertex solves over one factorization. meshlib
    moves its mesh in place, so each round denoises a fresh copy.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "meshlib":
        new_mesh_ml = bench_case.new_mesh_ml
        settings_ml = mm.DenoiseViaNormalsSettings()

        def denoise_ml() -> int:
            mesh_ml = new_mesh_ml()
            mm.meshDenoiseViaNormals(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(denoise_ml, rounds=3) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    smoothed = bench_case.run(
        lambda: od.smoothing.filter_two_step(vertices, faces, method="meshlib"), rounds=3
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.benchmark(group="filter_sharpen")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_filter_sharpen(bench_case: BenchCase) -> None:
    """Five Laplacian passes plus one blend: the cheapest thing in the module, on the scan sweep."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        new_meshset_pml = bench_case.new_meshset_pml

        def unsharp_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.apply_coord_unsharp_mask(weight=0.3, weightorig=1.0, iterations=5)
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(unsharp_pml) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    # The default operator, built inside the call as MeshLab builds its own: the same filter.
    sharpened = bench_case.run(
        lambda: od.smoothing.filter_sharpen(vertices, faces, weight=0.3, iterations=5)
    )
    assert sharpened.shape == (n_vertices,)


@pytest.mark.benchmark(group="filter_sharpen_graph_laplacian")
@pytest.mark.benchlibs("ordito", "open3d")
def test_filter_sharpen_graph_laplacian(bench_case: BenchCase) -> None:
    """
    Open3D's sharpener: five steps along the unnormalized graph Laplacian, operator included.

    A small strength, since the update adds the *unnormalized* residual, degree times the unsharp
    mask's, and so amplifies noise fast. Both rows compute the same positions
    (``tests/test_smoothing.py::test_filter_sharpen_graph_laplacian_matches_open3d``).
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "open3d":
        _skip_pml_beyond_bunny(bench_case)
        mesh_o3d = bench_case.mesh_o3d
        sharpened_o3d = bench_case.run(
            lambda: mesh_o3d.filter_sharpen(number_of_iterations=5, strength=0.05)
        )
        assert len(sharpened_o3d.vertices) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    sharpened = bench_case.run(
        lambda: od.smoothing.filter_sharpen(
            vertices, faces, weight=0.05, iterations=5, method="graph_laplacian"
        )
    )
    assert sharpened.shape == (n_vertices,)


_REGION_FRACTION = 0.25  # free vertices as a fraction of the mesh, by a coordinate cut


def _free_mask_np(bench_case: BenchCase) -> np.ndarray:
    """
    Cut a contiguous free region: the top quarter by z, so its rim is one closed curve.

    A random mask would give the solver a shredded region with an enormous rim and would measure
    something else -- the region *boundary* is what both libraries fold into the right-hand side, so
    its length is the thing to hold fixed across rows.
    """
    z_np = bench_case.vertices_np[:, 2]
    threshold = float(np.quantile(z_np, 1.0 - _REGION_FRACTION))
    return np.ascontiguousarray(z_np > threshold)


@pytest.mark.benchmark(group="smooth_region")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_smooth_region(bench_case: BenchCase) -> None:
    """
    Solve the umbrella-Laplacian Dirichlet system on a free region: a sparse solve, not a filter.

    Read against the ``filter_*`` groups above, which apply a fixed number of explicit passes: this
    one solves to a fixpoint, so its cost is a linear solve over the free set and is driven by that
    set's size and shape rather than by an iteration count. The free region is the top quarter of
    the mesh by z on both sides, so the two solve the same system.

    meshlib's ``positionVertsSmoothly`` is that same system with the same unit edge weights
    (``EdgeWeights.Unit``, ``VertexMass.Unit``), factorized where ordito iterates -- the answers
    agree to 1e-4. ordito's row passes ``edge_weights="unit"`` for that reason: the default
    cotangent weights are another system, ten times the CG iterations on dragon. meshlib mutates
    the mesh in place and returns nothing, so its mesh is rebuilt inside the timed callable and the
    row carries the build.

    **Two preconditioners, switched by size.** Below ``smoothing._SMOOTH_REGION_MULTIGRID_FROM``
    free vertices the normal equations take ``linalg.squared_laplacian_preconditioner``'s Chebyshev
    polynomial in the free rows' square block ``D^-1 L``, applied as ``B B^T``, with no setup; its
    degree and interval (``SQUARED_LAPLACIAN_DEGREE``, ``SQUARED_LAPLACIAN_INTERVAL``) move the
    bunny row. Its iteration count grows in proportion to the free count, so from the gate both
    weightings take ``method="multigrid"``, ``V D^2 V`` with ``V`` a smoothed-aggregation V-cycle
    on ``L_ff`` under l1-Jacobi smoothing (``linalg._SQUARED_MULTIGRID_SWEEPS``): unit weights
    dragon 1 035 -> 99 iterations, lucy past 600 s -> 48 s; the default cotangent weights dragon
    10 948 -> 339 (6.4 -> 0.58 s), lucy 153 s where the polynomial did not finish. On the largest
    regions each V-cycle is further wrapped in a degree-3 Chebyshev iteration
    (``smoothing._SMOOTH_REGION_CHEBYSHEV_FROM_*``): lucy cotangent 153 -> 99 s, unit 48 -> 42 s.
    ``linalg._MULTIGRID_THETA`` moves the scan rows.
    """
    skip_larger_than(
        bench_case,
        "happy_buddha",
        "one ordito round on lucy is 42 s (350 CG iterations over 3.5 M free vertices)",
    )
    n_vertices = bench_case.n_vertices
    free_np = _free_mask_np(bench_case)
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def smooth_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mm.positionVertsSmoothly(
                mesh_ml,
                mn.vertBitSetFromBools(cast("Buffer", free_np)),
                mm.EdgeWeights.Unit,
                mm.VertexMass.Unit,
            )
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(smooth_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    free_wp = wp.array(free_np, dtype=wp.bool, device=bench_case.device)
    smoothed = bench_case.run(
        lambda: od.smoothing.smooth_region(vertices, faces, free_wp, "unit"), rounds=3
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.benchmark(group="smooth_region_fixed_rim")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_smooth_region_fixed_rim(bench_case: BenchCase) -> None:
    """
    The same solve with the region rim pinned: the variant hole filling actually calls.

    Read against ``smooth_region`` on the identical region -- the only difference is whether the rim
    is a hard C0 constraint, which changes the system's size rather than its kind, so the two rows
    should track each other closely. A large gap would mean one of the two is not solving what it
    says.

    meshlib's ``positionVertsSmoothlySharpBd`` is the matching variant and takes the region through
    ``PositionVertsSmoothlyParams``; it agrees with ordito to 1e-5 (``tests/test_smoothing.py``).
    Same in-place mutation, so same rebuild inside the timed callable.
    """
    skip_larger_than(
        bench_case,
        "bunny",
        "the Dirichlet solve over a quarter of the mesh runs into tens of seconds past bunny "
        "(15.8 s on dragon, 24.9 s on happy_buddha for one ordito round)",
    )
    n_vertices = bench_case.n_vertices
    free_np = _free_mask_np(bench_case)
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def smooth_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            params_ml = mm.PositionVertsSmoothlyParams()
            params_ml.region = mn.vertBitSetFromBools(cast("Buffer", free_np))
            mm.positionVertsSmoothlySharpBd(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(smooth_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    free_wp = wp.array(free_np, dtype=wp.bool, device=bench_case.device)
    smoothed = bench_case.run(
        lambda: od.smoothing.smooth_region_fixed_rim(vertices, faces, free_wp), rounds=3
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.benchmark(group="refine_and_smooth_region")
@pytest.mark.benchmeshes("saddle_small", "hemisphere")
@pytest.mark.benchlibs("ordito")
def test_refine_and_smooth_region(bench_case: BenchCase) -> None:
    """
    The refine-and-smooth half of ``holes.fill_smooth``: the rim's fill subdivided and faired.

    The rim is filled outside the timed region (``fill_smooth(triangulate_only=True)``), so the row
    is the subdivision to the mean edge length plus the cotangent smooth of the new vertices. The
    two disk patches give one rim of 268 and of 384 vertices: the patch size is the axis.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    filled_vertices, filled_faces, patch = od.holes.fill_smooth(
        vertices, faces, triangulate_only=True, return_patch=True
    )
    max_edge = float(od.edges.mean_edge_length(vertices, faces))
    n_vertices = bench_case.n_vertices
    refined, refined_faces, refined_patch = bench_case.run(
        lambda: od.smoothing.refine_and_smooth_region(
            filled_vertices,
            filled_faces,
            n_vertices,
            patch,
            max_edge=max_edge,
            max_edge_splits=1000,
            max_angle_change_after_flip=math.radians(30.0),
            smooth_curvature=True,
            smooth_boundary=True,
            natural_smooth=False,
            edge_weights="cotan",
        ),
        rounds=3,
    )
    assert refined.shape[0] > n_vertices
    assert refined_patch.shape[0] == refined_faces.shape[0] // 3
    assert np.isfinite(refined.numpy()).all()


@pytest.mark.benchmark(group="inflate")
@pytest.mark.benchlibs("ordito")
def test_inflate(bench_case: BenchCase) -> None:
    """
    The balloon flow: per pass, a vertex-normal build, one displacement map and one Laplacian pass.

    Read against ``filter_laplacian`` at the same iteration count -- the difference between the two
    groups is what the normals and the displacement cost, and it should be roughly the normal build,
    since the displacement is one ``wp.map`` with the kernel hoisted out of the loop.

    ordito-only, and not by omission. MeshLib's ``inflate`` is the only reference that has one and
    it cannot be timed here: with every vertex selected -- the operation this performs -- it
    collapses the mesh to a point at every pressure probed, because its implicit solve takes the
    *unselected* vertices as its boundary condition. The evidence is in ``tests/test_smoothing.py``.

    Restricting it to a *region* does not rescue the row, and the reason is sharper than "it solves
    a different problem": **its displacement is insensitive to the parameter that should drive it.**
    Over two orders of magnitude of pressure and a sign flip, with a cap selected, the surface moves
    less than the runs differ from each other and the volume *falls* where positive pressure must
    raise it. What the row would be timing is the ``preSmooth`` pass.

    ordito's own row is flat from a feature mesh up to a million faces, so the three passes are
    launch-bound rather than data-bound at this scale -- and ``lucy``'s jump at a comparable face
    count is the same unexplained outlier ``split_faces_along_field`` records on that mesh.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    pressure = 0.1 * bench_case.mean_edge
    inflated = bench_case.run(lambda: od.smoothing.inflate(vertices, faces, pressure), rounds=3)
    assert int(inflated.shape[0]) == bench_case.n_vertices


@pytest.mark.benchmark(group="filter_spikes")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_filter_spikes(bench_case: BenchCase) -> None:
    """
    Detect and flatten needle vertices: per pass, the corner angles, a defect scatter and one map.

    This is **not** a detector that stops after one pass on a clean mesh, and reading it that way
    hid a real win. Two things to know:

    * **The loop runs its full pass budget on any mesh with *unreferenced* vertices.** A vertex with
      no incident face has an angle sum of ``0``, so it is a spike under any threshold, and
      averaging its (empty) 1-ring never moves it -- the per-pass spike count converges to exactly
      the unreferenced count and stays there. So the row is a full-budget row everywhere in this
      registry, and ``flattened`` over-reports by the pass count times the unreferenced count.
    * **Most of it was rebuilding an operator that never changes.** Each pass called
      ``filter_neighborhood_average``, which builds the uniform 1-ring operator per call -- and at
      ``equal_weight=True`` that operator reads no positions, so every pass produced a
      byte-identical matrix. The build being flat in the mesh is what made the *row* flat in the
      mesh, which is the signature that named this a work item.

    Hoisting the build turned both small rows from losses into wins and is worth several-fold on the
    large ones. The residual is still nearly flat below a million faces, which is the passes'
    remaining fixed cost -- read it as floor now rather than as an algorithm.

    meshlib mutates in place, so its mesh is rebuilt per round the way the other ``repair`` rows do.
    Both sides are pinned against each other on a genuinely spiky mesh in
    ``tests/test_smoothing.py``; no scan mesh in this registry has a spike, which is why that test
    needs a generated fixture and this row does not.
    """
    threshold = 0.5 * math.pi
    if bench_case.kind == "meshlib":

        def remove_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            mm.removeSpikes(mesh_ml, 10, threshold)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(remove_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    repaired, flattened = bench_case.run(
        lambda: od.smoothing.filter_spikes(vertices, faces, threshold, return_count=True), rounds=3
    )
    assert flattened >= 0
    assert int(repaired.shape[0]) == bench_case.n_vertices


@pytest.mark.benchmark(group="equalize_triangle_areas")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_equalize_triangle_areas(bench_case: BenchCase) -> None:
    """
    A 3x3 float64 solve per vertex per pass, over the incident-face CSR.

    Read against ``filter_laplacian``, which is the same shape of pass over the same 1-ring and does
    a weighted average where this solves: the gap between the two rows is what the area objective
    costs, and it is a fixed per-vertex constant rather than anything that grows.

    ``no_shrinkage`` is deliberately left off. It adds a vertex-normal pass and a 2x2 solve per
    vertex per iteration, which is a second measurement rather than a variation of this one, and the
    two agree with meshlib either way (``tests/test_smoothing.py``, exactly without it and to
    3.6e-07 with).

    meshlib's ``equalizeTriAreas`` is the same solve, threaded across vertices and mutating in
    place, so its mesh is rebuilt per round -- and the positions agree **exactly** on a closed mesh
    (``tests/test_smoothing.py``), which is what makes the ratio a fair one. On an open mesh ordito
    holds the boundary vertices where meshlib solves for them too (collapsing the rim's fans), so
    meshlib does a few hundred more per-vertex solves on the scans: a negligible share of a pass.
    ordito wins it by more than an order of magnitude at every size.

    The first mesh in a selection carries the module's compile and therefore reads slower than it
    is; read the large rows against each other, where the scaling is clean.
    """
    if bench_case.kind == "meshlib":

        def equalize_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshEqualizeTriAreasParams()
            params_ml.iterations = _ITERATIONS
            mm.equalizeTriAreas(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(equalize_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: od.smoothing.equalize_triangle_areas(vertices, faces, _ITERATIONS), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="equalize_triangle_areas_no_shrinkage")
@pytest.mark.benchlibs("ordito")
def test_equalize_triangle_areas_no_shrinkage(bench_case: BenchCase) -> None:
    """
    ``equalize_triangle_areas`` confined to each vertex's tangent plane.

    Read against the plain row: the difference is the area-weighted normal each vertex gathers
    over its incident faces per pass, plus the 2x2 tangent solve. ordito-only: meshlib's row
    already covers the agreement, which ``tests/test_smoothing.py`` checks with and without it.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: od.smoothing.equalize_triangle_areas(
            vertices, faces, _ITERATIONS, no_shrinkage=True
        ),
        rounds=3,
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="relax_keep_volume")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_relax_keep_volume(bench_case: BenchCase) -> None:
    """
    Two launches per pass over the 1-ring CSR, against ``filter_laplacian``'s one.

    That ratio is the whole cost model: the volume correction is a second pass over the same
    adjacency reading a displacement field instead of positions, so this row should sit near twice
    ``filter_laplacian``'s and anything else means the adjacency build (hoisted out of the loop
    here, as there) has moved.

    meshlib's ``relaxKeepVolume`` is the same two-pass formulation, and the positions agree closely
    (``tests/test_smoothing.py``). It mutates in place, so its mesh is rebuilt per round. ordito
    wins it by most of an order of magnitude, by more the larger the mesh.

    Against ``equalize_triangle_areas`` on the same passes and meshes the two are within a third of
    each other once the mesh is large. That is the cost model working out: two cheap ring passes
    here against one ring pass with a 3x3 float64 solve there, so neither the solve nor the extra
    launch dominates and both rows are bandwidth on the adjacency.
    """
    if bench_case.kind == "meshlib":

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshRelaxParams()
            params_ml.iterations = _ITERATIONS
            mm.relaxKeepVolume(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: od.smoothing.relax_keep_volume(vertices, faces, _ITERATIONS), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="relax_approx")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_relax_approx(bench_case: BenchCase) -> None:
    """
    Neighbourhood fitting, where the neighbourhood build dominates the fit.

    The geodesic balls are built **once** and reused across passes, so this row is one
    ``geodesic_ball`` plus a per-vertex plane fit per pass -- read it against
    ``curvature.principal_curvature``, which builds the same balls and then does a strictly larger
    5x5 solve on each. A change here that does not show there is in the fit; one that shows in both
    is in the ball.

    The radius is 3 % of the bounding-box diagonal, which is the scale at which a ball holds enough
    vertices to fit on every mesh in the registry. It is not scale-free and there is no default:
    meshlib's own ``surfaceDilateRadius`` default of ``0`` is a no-op (``tests``), so the two rows
    would otherwise not be timing the same work at all.

    ``fit="planar"`` on both sides. The quadric adds a 6x6 QR per vertex per pass and is a separate
    measurement, not a variation of this one.

    **This is the only row in the module whose cost is superlinear**, and the ball is why: it grows
    faster than the face count, because a fixed 3 % radius holds more vertices as the mesh refines.
    That is the thing to watch on any change here -- a regression in the *fit* would move all four
    rows together, and one in the ball would move only the large two. ordito wins the row at every
    size, by more the larger the mesh.
    """
    skip_larger_than(
        bench_case,
        "happy_buddha",
        "a ball of 3 % of lucy's diagonal holds about 57 000 vertices by area, so every ordito "
        "ball is clipped at its 512-entry row and the two rows would not time the same fit; the "
        "call itself runs (streamed balls, about 5 GB of device memory)",
    )
    radius = 0.03 * float(
        np.linalg.norm(bench_case.vertices_np.max(0) - bench_case.vertices_np.min(0))
    )
    if bench_case.kind == "meshlib":

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshApproxRelaxParams()
            params_ml.iterations = 1
            params_ml.surfaceDilateRadius = radius
            params_ml.type = mm.RelaxApproxType.Planar
            mm.relaxApprox(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: od.smoothing.relax_approx(vertices, faces, radius, 1), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="relax_approx_meshlib")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_relax_approx_meshlib(bench_case: BenchCase) -> None:
    """
    ``relax_approx(method="meshlib")``: shortest-edge-path balls rebuilt every pass, float64 fit.

    Read against ``relax_approx``, which builds its balls once: this row walks a bounded Dijkstra
    per vertex on every pass, so at several passes the walk is most of the call. Three planar
    passes, radius 3 % of the bounding-box diagonal as in ``relax_approx``, and the same cap for
    the same reason (lucy's balls clip at their 512-entry row).

    Measured on CUDA (noisy unit icospheres, this radius, planar, device time), ``method="meshlib"``
    against the default: 4.5x at three passes and 1.7x at one on ``icosphere(6)`` (41 k vertices),
    3.5x and 1.26x on ``icosphere(7)`` (164 k). One walk costs about 1.14x the default's
    breadth-first walk, and the default walks once whatever the pass count, so the rebuild per pass
    is now the gap. The walk expands each ball vertex once (counted: one pop a vertex, six probes),
    so its cost is the probe: one read of a ``(vertex, distance)`` table entry, where a key table, a
    rank table and a distance row took three dependent reads (0.68x walk), and a memset of the
    table per chunk instead of probing every key again to clear it (0.88x call). Chunk re-swept
    after both: ``2**15`` sources a launch (``neighbors._EDGE_PATH_BALL_CHUNK``).
    """
    skip_larger_than(
        bench_case,
        "happy_buddha",
        "a ball of 3 % of lucy's diagonal holds about 57 000 vertices by area, so every ordito "
        "ball is clipped at its 512-entry row and the two rows would not time the same fit",
    )
    radius = 0.03 * float(
        np.linalg.norm(bench_case.vertices_np.max(0) - bench_case.vertices_np.min(0))
    )
    if bench_case.kind == "meshlib":

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshApproxRelaxParams()
            params_ml.iterations = 3
            params_ml.surfaceDilateRadius = radius
            params_ml.type = mm.RelaxApproxType.Planar
            mm.relaxApprox(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: od.smoothing.relax_approx(vertices, faces, radius, 3, method="meshlib"), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="smooth_region_boundary")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_smooth_region_boundary(bench_case: BenchCase) -> None:
    """
    A harmonic solve per pass, on the same region as ``smooth_region`` -- but a *scalar* one.

    Read against ``smooth_region``: same region, same cotangent operator, one right-hand side
    instead of three, and the free set is the region's rim band rather than its whole interior. So
    this row should be much the cheaper of the two whenever the region is fat, and the gap closing
    would mean the band had stopped being thin.

    The band's system is assembled once and its values rewritten each pass from the moved
    positions: the sparsity is the topology and does not change, the cotangent weights do. Keeping
    the one operator also keeps its conjugate-gradient state, so every pass after the first replays
    a recorded loop.

    meshlib's ``smoothRegionBoundary`` additionally flips the band's interior edges before each
    solve, which this port does not do -- so its row carries connectivity work ordito's does not,
    and the two are pinned on the *moved set* and the rim length rather than element-wise. It
    mutates in place, so its mesh is rebuilt per round.

    The module's one **loss** against meshlib, and the reason is visible in the shape: ordito is
    flat across the mesh pair while meshlib tracks the mesh, so the row is four conjugate-gradient
    solves and their fixed per-call cost rather than anything proportional.

    **What the row actually prices is the conjugate-gradient launch floor on a tiny system.** The
    free set is the band, a couple of percent of the mesh, so the reduced system is a few hundred
    unknowns and solving it takes a couple of dozen iterations whose cost is per-iteration *launch*
    rather than arithmetic. That is why the row is flat in a mesh several times larger: the band
    barely grows. The rest of the call is the prologue, which is likewise band-sized or built once.

    The preconditioner work does not reach it either: ``linalg``'s hierarchy setup alone is several
    times this entire solve, and ``"auto"``'s probe would spend its Jacobi iterations before paying
    it (see ``linalg.py``, where routing the fixed-boundary solves that way is refuted at systems
    orders larger than this one). Anything that moves this row has to remove *launches per
    iteration* or iterations, not operator builds.
    """
    skip_larger_than(
        bench_case,
        "bunny",
        "four harmonic solves over the region rim run into tens of seconds past bunny, the same "
        "wall smooth_region hits on the same region",
    )
    region_np = np.ascontiguousarray(_free_mask_np(bench_case)[bench_case.faces_np].any(axis=1))
    if bench_case.kind == "meshlib":

        def smooth_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            region_ml = face_bitset_ml(region_np)
            mm.smoothRegionBoundary(mesh_ml, region_ml, 4)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(smooth_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    region_wp = wp.array(region_np, dtype=wp.bool, device=bench_case.device)
    smoothed = bench_case.run(
        lambda: od.smoothing.smooth_region_boundary(vertices, faces, region_wp, 4), rounds=3
    )
    assert smoothed.shape == (bench_case.n_vertices,)
