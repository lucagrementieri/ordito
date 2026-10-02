"""
Benchmarks for ``ordito.curvature``.

Axis: **scale**, plus a **radius** sweep on every group — and the second matters more. All three
functions gather a neighborhood before computing anything, so their cost is ``V * k_bar``, and
``k_bar`` grows with the *square* of the radius on a surface: doubling it is roughly four times the
work at an unchanged vertex count. That is why the radius is swept explicitly rather than pinned to
one value that would hide the exponent.

Three functions, three cost profiles:

* ``principal_curvature`` — a per-vertex 5x5 float64 least-squares quadric fit over a geodesic-ball
  neighborhood. The collection ([`geodesic_ball`][ordito.neighbors.geodesic_ball], timed on its own
  in [`test_proximity.py`](test_proximity.py)) is part of the call, so this number is "ball + fit"
  and the fit is what a change to the 5x5 solver moves.
* ``discrete_gaussian_curvature`` — a hash-grid ball query plus a segmented scatter-sum of vertex
  defects. Cheap, memory-bound, dominated by the query.
* ``discrete_mean_curvature`` — the same query shape over *edges*, so it also pays face adjacency
  and the per-edge dihedral angles.

Why the clean spheres rather than the scan meshes
-------------------------------------------------
``igl.principal_curvature`` **segfaults** on every scan mesh — they have non-manifold vertices and
libigl's vertex-ring walk assumes manifoldness, so it takes the pytest process with it rather than
raising. The ``scale`` axis is manifold by construction, so libigl runs on all three points and the
comparison is drawn on the same meshes ordito is measured on. **Not** on the ``valence`` axis: that
ring walk is quadratic in valence, and ``fan_hub``'s forty-thousand-face hub takes minutes. The
guard is explicit rather than implied by the axis, because the failure mode is a wall-clock blowout
that looks like a hang.

References
----------
**libigl**: ``igl.principal_curvature(..., useKring=False)`` is the sphere-neighborhood variant
ordito reproduces — the ``useKring=True`` default collects a combinatorial k-ring instead, a
different neighborhood and so a different amount of work — timed at the same radius multiplier.

**trimesh** is the reference for the two Cohen-Steiner / Morvan measures, given the same query
points and radius. **open3d** has no curvature estimation at all, and does not even expose vertex
angle defects.

**pymeshlab** covers all three groups and is the only reference surviving the *whole* scale axis on
the two measures, where trimesh is capped at ``sphere_small`` by its per-point cKDTree ball. Two
filters:

``compute_curvature_principal_directions_per_vertex`` is the quadric-fit analogue, and its
``method`` menu matters: the two expensive variants are more than an order of magnitude dearer than
the one ordito implements and are deliberately not benchmarked. **The radius sweep does not map** —
MeshLab exposes no neighborhood size for the fit — so its row appears at the narrow radius only.
That internal neighborhood is what the row actually measures: against the analytic ``H = 1`` of a
unit icosphere it is equivalent to ``radius = 2``, so the benchmarked ``radius = 3`` row is already
doing more work than the reference. Read the ratio with that in mind.

It is **not** a test oracle, and that is a deliberate rejection. On ``torus`` at ``radius = 2`` the
two correlate at 0.982 but carry a ~7 % systematic level offset — the neighborhood-size effect above
— where ``igl.principal_curvature(useKring=False)`` agrees element-wise to 1e-3, so libigl is
strictly the better oracle and pymeshlab would only loosen it. The other three ``method=`` variants
are worse still: ``'Normal Cycles'`` is area-integrated rather than pointwise, ``'Taubin
approximation'`` produces outliers orders of magnitude out, and ``'Scale Dependent Quadric
Fitting'`` reproduces plain Quadric Fitting at many times the cost.

``compute_scalar_by_discrete_curvature_per_vertex`` gives Gaussian and Mean from one filter, but it
is the Meyer / Desbrun **pointwise 1-ring** operator rather than the Cohen-Steiner / Morvan ball
measure ordito and trimesh compute. So it has no radius axis and its absolute value is not
comparable — it is a *throughput* reference for a per-vertex curvature pass over the same mesh,
which is what makes it useful where trimesh cannot run.

Both mutate only an attribute, but ``compute_curvature_principal_directions_per_vertex`` defaults to
``autoclean=True`` and would delete unreferenced vertices under a shared MeshSet, so both build
inside the timed callable; the build is a small share across the whole scale axis.

The timed callable holds everything the public function does. For the Gaussian measure that includes
the vertex-defect recomputation, so the trimesh reference builds its ``tm.Trimesh`` inside the timed
region too — otherwise its cached ``vertex_defects`` / ``face_adjacency`` / ``kdtree`` properties
would make rounds 2..n measure only the ball query. The precomputed inputs are the ones ordito's
signature asks the caller for.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm

import ordito as od
import ordito.typing as odt
from conftest import BenchCase

# Sphere radius of the quadric fit, as a multiple of the average edge length. libigl defaults to 5;
# the pair brackets it so the quadratic growth in neighborhood size is visible.
_QUADRIC_RADII = [3, 8]

# Radius of the normal-cycle curvature measures, as a multiple of the mean edge length.
_MEASURE_RADII = [2.0, 4.0]

# The quadric fit runs into hundreds of milliseconds a call at the wide radius; ten rounds of that
# on three meshes and two libraries would dominate the suite.
_HEAVY_ROUNDS = 3

_face_angles_cache: dict[tuple[str, str], odt.Array2dFloat32] = {}


def _face_angles(bench_case: BenchCase) -> odt.Array2dFloat32:
    """``(n_faces, 3)`` interior angles -- an *input* of ``discrete_gaussian_curvature``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _face_angles_cache:
        _face_angles_cache[key] = od.triangles.face_angles(
            bench_case.vertices_wp, bench_case.faces_wp
        )
    return _face_angles_cache[key]


@pytest.mark.noparity(
    "pyvista",
    oracle="igl",
    reason="D1 not an independent implementation: VTK's curvature('maximum') and ('minimum') are "
    "pure algebra on its own Gauss and mean curvature -- measured *exactly* H +- sqrt(H^2 - K), "
    "max abs "
    "difference 0.0 in float64 on icosphere(3) -- so they estimate no principal curvature of their "
    "own and inherit vtkCurvatures' 1-ring stencil. They also go complex where H^2 < K, which is "
    "300 of 642 vertices on that same sphere, and VTK returns the clamped real part rather than "
    "raising. igl.principal_curvature(useKring=False) is the quadric fit ordito implements and is "
    "the oracle, in tests/test_curvature.py::test_principal_curvature.",
)
@pytest.mark.benchmark(group="principal_curvature")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl", "pymeshlab", "pyvista")
@pytest.mark.parametrize("radius", _QUADRIC_RADII)
def test_principal_curvature(bench_case: BenchCase, radius: int) -> None:
    """Per-vertex quadric fit over a geodesic ball: the 5x5 solve in bulk, at two radii."""
    if bench_case.kind == "igl" and bench_case.mesh_name == "sphere_large":
        pytest.skip("igl.principal_curvature is ~2 s a call at this size; capped at sphere_med")
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        if radius != min(_QUADRIC_RADII):
            pytest.skip("VTK's curvature is a fixed 1-ring stencil: no radius axis")
        mesh_pv = bench_case.mesh_pv
        curvature_pv = bench_case.run(lambda: mesh_pv.curvature("maximum"), rounds=_HEAVY_ROUNDS)
        assert np.asarray(curvature_pv).shape == (n_vertices,)
        return
    if bench_case.kind == "pymeshlab":
        if radius != min(_QUADRIC_RADII):
            pytest.skip("MeshLab derives the fit neighborhood itself: no radius axis")
        # ``autoclean=True`` (the default) deletes unreferenced vertices, so this one cannot share
        # a MeshSet even though it writes only curvature attributes.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().compute_curvature_principal_directions_per_vertex(
                method="Quadric Fitting"
            ),
            rounds=_HEAVY_ROUNDS,
        )
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        *_, pv1, _pv2 = bench_case.run(
            lambda: od.curvature.principal_curvature(vertices, faces, radius=radius),
            rounds=_HEAVY_ROUNDS,
        )
        assert pv1.shape == (n_vertices,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        *_, pv1_igl, _pv2, _bad = bench_case.run(
            lambda: igl.principal_curvature(vertices_np, faces_np, radius=radius, useKring=False),
            rounds=_HEAVY_ROUNDS,
        )
        assert pv1_igl.shape == (n_vertices,)


def _run_discrete_curvature_pml(
    bench_case: BenchCase, radius_scale: float, curvature_type: str
) -> None:
    """
    Time MeshLab's pointwise Meyer / Desbrun curvature, which has no radius to sweep.

    Shared by both measure groups because one filter serves both through ``curvaturetype``. It
    writes only the vertex scalar, but the MeshSet is rebuilt anyway so the two curvature groups in
    this module stay consistent with the quadric-fit row above, which has no choice.
    """
    if radius_scale != min(_MEASURE_RADII):
        pytest.skip("MeshLab's discrete curvature is a pointwise 1-ring operator: no radius axis")
    bench_case.run(
        lambda: bench_case.new_meshset_pml().compute_scalar_by_discrete_curvature_per_vertex(
            curvaturetype=curvature_type
        )
    )


@pytest.mark.noparity(
    "pymeshlab",
    oracle="trimesh",
    reason="D2 different operator with a measured offset: MeshLab computes the Meyer / Desbrun "
    "pointwise 1-ring curvature, not the Cohen-Steiner / Morvan ball measure ordito and trimesh "
    "integrate over a radius, so its absolute value is not comparable and it has no radius axis "
    "at all. It is a throughput reference; trimesh is the oracle, in "
    "tests/test_curvature.py::test_discrete_gaussian_curvature.",
)
@pytest.mark.benchmark(group="discrete_gaussian_curvature")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "trimesh", "pymeshlab")
@pytest.mark.parametrize("radius_scale", _MEASURE_RADII)
def test_discrete_gaussian_curvature(bench_case: BenchCase, radius_scale: float) -> None:
    """Summed vertex defects inside a ball around every vertex (Cohen-Steiner / Morvan)."""
    n_vertices = bench_case.n_vertices
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        _run_discrete_curvature_pml(bench_case, radius_scale, "Gaussian Curvature")
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        face_angles = _face_angles(bench_case)
        curvature = bench_case.run(
            lambda: od.curvature.discrete_gaussian_curvature(
                vertices, vertices, faces, face_angles, radius
            )
        )
        assert curvature.shape == (n_vertices,)
    else:  # rebuild inside: vertex_defects and kdtree are cached Trimesh properties
        if bench_case.mesh_name != "sphere_small":
            # Measured at tens of seconds a call on ``sphere_med`` -- one cKDTree ball query per
            # point. Two rows of that were nearly all of this module's wall clock, for a ratio the
            # smaller mesh already establishes at three orders of magnitude.
            pytest.skip("trimesh queries one cKDTree ball per point; capped at sphere_small")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        curvature_tm = bench_case.run(
            lambda: tm.curvature.discrete_gaussian_curvature_measure(
                tm.Trimesh(vertices_np, faces_np, process=False), vertices_np, radius
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert curvature_tm.shape == (n_vertices,)


@pytest.mark.noparity(
    "pymeshlab",
    oracle="trimesh",
    reason="D2 different operator with a measured offset: the same Meyer / Desbrun pointwise "
    "1-ring measure as the Gaussian row above, against ordito's Cohen-Steiner / Morvan ball "
    "integral. Not comparable in absolute value and it exposes no radius; trimesh is the oracle, "
    "in tests/test_curvature.py::test_discrete_mean_curvature.",
)
@pytest.mark.benchmark(group="discrete_mean_curvature")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "trimesh", "pymeshlab")
@pytest.mark.parametrize("radius_scale", _MEASURE_RADII)
def test_discrete_mean_curvature(bench_case: BenchCase, radius_scale: float) -> None:
    """Summed edge dihedral angles inside a ball around every vertex: adjacency plus the query."""
    n_vertices = bench_case.n_vertices
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        _run_discrete_curvature_pml(bench_case, radius_scale, "Mean Curvature")
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        curvature = bench_case.run(
            lambda: od.curvature.discrete_mean_curvature(vertices, vertices, faces, radius)
        )
        assert curvature.shape == (n_vertices,)
    else:  # rebuild inside: face_adjacency and kdtree are cached Trimesh properties
        if bench_case.mesh_name != "sphere_small":
            # Measured at tens of seconds a call on ``sphere_med`` -- one cKDTree ball query per
            # point. Two rows of that were nearly all of this module's wall clock, for a ratio the
            # smaller mesh already establishes at three orders of magnitude.
            pytest.skip("trimesh queries one cKDTree ball per point; capped at sphere_small")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        curvature_tm = bench_case.run(
            lambda: tm.curvature.discrete_mean_curvature_measure(
                tm.Trimesh(vertices_np, faces_np, process=False), vertices_np, radius
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert np.asarray(curvature_tm).shape == (n_vertices,)
