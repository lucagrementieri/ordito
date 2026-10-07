"""
Regression tests for ``ordito.heat`` against igl, potpourri3d, pymeshlab and pyvista.

Three solvers in one file, in the module's own source order: the scalar heat method, the signed
heat method, then the vector-valued family. Each section's references and its comparison hazards
are described where it begins.

Signed heat method: the source curves
-------------------------------------
Every curve here is a **vertex one-ring cycle**, for a reason that is easy to trip over:
``potpourri3d.MeshSignedHeatSolver`` rejects a curve whose consecutive points do not share a face
("Each curve segment must be contained within a single face"), so an arbitrary vertex list is
not a valid source for it. A one-ring cycle is the smallest curve that is genuinely edge-connected,
closed, *and* separating — which is what makes the sign meaningful — and it comes straight out of
[`vertex_one_rings`][ordito.halfedge.vertex_one_rings].

The strongest check is not against the reference at all: the *magnitude* of the signed field has to
agree with the unsigned [`heat_geodesic`][ordito.heat.heat_geodesic] distance to the same
curve, which is a completely different solve.

Vector heat method: comparing tangent fields
--------------------------------------------
Comparing tangent fields across libraries needs care in two places, and both are load-bearing here:

* **The source vector.** ``(1, 0)`` means "along *my* reference direction", and the two libraries
  choose different ones, so handing both the literal ``(1, 0)`` transports two different world
  vectors. Every comparison below converts the source vector into the reference library's frame
  first.
* **The result.** 2D components are meaningless across libraries; the fields are compared after
  expanding them to 3D through each library's own frames.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from benchmarks.meshes import BUILDERS
from tests.conftest import MESHES
from tests.conversions import (
    bsr_to_dense,
    meshlib_scalars_to_numpy,
    numpy_to_warp,
    points_to_warp,
    points_to_warp_uv,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    trimesh_to_warp,
    warp_empty,
)

# Not ``conftest.MESHES``: both predate that constant and neither has ever carried ``cave_cube``.
_HEAT_MESHES = ["icosahedron", "hemisphere", "half_torus"]
_HEAT_MESHES_SMALL = ["icosahedron", "hemisphere"]

Icosphere5 = tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]]


@functools.cache
def _icosphere5_tm() -> tm.Trimesh:
    return tm.creation.icosphere(subdivisions=5)


@pytest.fixture
def icosphere5(device: str) -> Icosphere5:
    """
    Return ``icosphere(5)`` (10 242 vertices, 96 rings pole to pole) and its device buffers.

    The far-field tests' mesh: every ``conftest`` fixture is too small to have a far field, and the
    defects those tests pin are invisible below about a dozen rings.
    """
    mesh_tm = _icosphere5_tm()
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, np.ravel(mesh_tm.faces), device)
    return mesh_tm, vertices_wp, faces_wp


GradedSaddle = tuple[np.ndarray, np.ndarray, wp.array[wp.vec3], wp.array[wp.int32]]


@functools.cache
def _graded_saddle_np() -> tuple[np.ndarray, np.ndarray]:
    """
    Return a ``68 x 68`` grid lifted onto a saddle, its spacing cubed along one axis.

    ``benchmarks``' ``saddle_graded`` at a quarter of the vertices: worst aspect ratio in the
    thousands.
    """
    k = 68
    step = np.linspace(-1.0, 1.0, k)
    u, v = np.meshgrid(np.sign(step) * np.abs(step) ** 3, step, indexing="ij")
    vertices = np.column_stack((u.ravel(), v.ravel(), 0.35 * (u * u - 0.6 * v * v).ravel()))
    i, j = np.meshgrid(np.arange(k - 1), np.arange(k - 1), indexing="ij")
    corner = (i * k + j).ravel()
    faces = np.vstack(
        (
            np.column_stack((corner, corner + k, corner + k + 1)),
            np.column_stack((corner, corner + k + 1, corner + 1)),
        )
    )
    return vertices, faces.astype(np.int64)


@pytest.fixture
def graded_saddle(device: str) -> GradedSaddle:
    """
    Return the graded saddle and its device buffers.

    The heat system there is ill-conditioned in places (Jacobi-scaled condition number in the
    millions), which is what the settle rule cannot see.
    """
    vertices_np, faces_np = _graded_saddle_np()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, np.ravel(faces_np), device)
    return vertices_np, faces_np, vertices_wp, faces_wp


# --------------------------------------------------------------------------
# heat_operators / heat_geodesic
# --------------------------------------------------------------------------

# --- heat_operators -----------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosphere_coarse", "hemisphere"])
def test_heat_operators_are_the_matrices_its_docstring_names(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: every member is checked against the function that defines it, and all agree exactly.

    The whole point of this entry point is that its results depend on the mesh alone, so each one
    has an independent definition to be held to: ``heat_system`` is ``M - t L`` for the documented
    default ``t`` (the squared mean edge length), ``poisson_system`` is ``-L``, ``laplacian`` is the
    ``float64`` cotangent matrix, and the face quantities are trimesh's. Measured on
    ``icosphere_coarse``: all three matrix identities hold to **0.0**, the normals to 2.0e-07 and
    the areas to 7.8e-09 (ordito's float32 vertex buffer against trimesh's float64). On an open
    mesh the ``M - t L`` residual is 5.6e-17 rather than zero -- ``bsr_axpy`` accumulates in a
    different order than the numpy expression -- so that one comparison carries a tolerance.

    ``t`` is checked in two halves. The system must be ``M - t L`` for *one* ``t`` to 1e-12, so
    ``t`` is read back off an off-diagonal entry rather than recomputed; and that ``t`` must be the
    squared mean of trimesh's unique edge lengths, which is the convention, to 1e-6 -- the float32
    vertex buffer puts the two a few 1e-7 apart. Recomputing ``t`` with
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length] instead would pin one
    summation order rather than the definition.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])
    n_faces = int(mesh_tm.faces.shape[0])
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    (
        heat_system,
        _heat_preconditioner,
        laplacian,
        poisson_system,
        _poisson_preconditioner,
        cot_entries_wp,
        face_normals_wp,
        face_areas_wp,
    ) = od.heat.heat_operators(vertices_wp, faces_wp)

    laplacian_np = bsr_to_dense(laplacian, n_vertices)
    cotmatrix_np = bsr_to_dense(
        od.laplacian.cotmatrix(vertices_wp, faces_wp, dtype=wp.float64), n_vertices
    )
    # The lumped diagonal ``heat_operators`` builds the system from, not the assembled
    # ``mass_matrix`` -- its own See Also names ``mass_matrix_entries``, and on an open mesh the two
    # differ by a float rounding (1.4e-17).
    mass_np = np.diag(
        od.laplacian.mass_matrix_entries(vertices_wp, faces_wp, dtype=wp.float64).numpy()
    )
    heat_np = bsr_to_dense(heat_system, n_vertices)
    row, column = np.argwhere(np.triu(laplacian_np, k=1) != 0.0)[0]
    diffusion_time = float(-heat_np[row, column] / laplacian_np[row, column])
    unique_length_np = mesh_tm.edges_unique_length.mean()
    assert np.isclose(diffusion_time, unique_length_np**2, rtol=1e-6, atol=0.0)

    assert np.array_equal(laplacian_np, cotmatrix_np)
    assert np.array_equal(bsr_to_dense(poisson_system, n_vertices), -laplacian_np)
    assert np.allclose(heat_np, mass_np - diffusion_time * laplacian_np, rtol=1e-12, atol=1e-15)
    assert cot_entries_wp.shape == (n_faces, 3)
    assert np.allclose(face_normals_wp.numpy(), mesh_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(face_areas_wp.numpy(), mesh_tm.area_faces, rtol=1e-5, atol=1e-5)


def test_heat_operators_reject_cot_entries_with_use_robust(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    The two arguments ask for different half-cotangent tables, so together they are an error.

    Not a library comparison: no reference exposes the mollified operator as an option, and this is
    about the signature rather than the numbers. Passing both silently would let ``use_robust``
    look like it was honoured while the extrinsic table was used, which is exactly the failure a
    caller reaches for ``use_robust`` to avoid.
    """
    _mesh_tm, mesh_wp = icosphere_coarse
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    cot_entries_wp = od.laplacian.cotmatrix_entries(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="mutually exclusive"):
        od.heat.heat_operators(vertices_wp, faces_wp, cot_entries=cot_entries_wp, use_robust=True)


def test_heat_operators_honour_an_explicit_diffusion_time(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """``t`` reaches the assembled system rather than being recomputed from the edge lengths."""
    mesh_tm, mesh_wp = icosphere_coarse
    n_vertices = int(mesh_tm.vertices.shape[0])

    heat_system, *_rest, _cot, _normals, _areas = od.heat.heat_operators(
        mesh_wp.points, mesh_wp.indices, t=0.05
    )

    laplacian_np = bsr_to_dense(
        od.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float64), n_vertices
    )
    mass_np = np.diag(
        od.laplacian.mass_matrix_entries(mesh_wp.points, mesh_wp.indices, dtype=wp.float64).numpy()
    )
    assert np.allclose(
        bsr_to_dense(heat_system, n_vertices), mass_np - 0.05 * laplacian_np, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("mesh_name", ["icosphere_coarse", "hemisphere"])
def test_heat_geodesic_on_a_trimesh_matches_vertices_and_faces(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the two forms of the call are one computation.

    Not a parity assert; the oracle for the distances themselves is
    [`test_heat_geodesic_matches_igl`]. The first `Trimesh` call runs the same solves as the
    ``vertices, faces`` form (equal to 1e-12); the second goes to the factorizations the mesh kept,
    which agree to the Poisson solve's tolerance (1e-6 of the range here; measured below 1e-9).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(
        np.array([0, mesh_tm.vertices.shape[0] // 3], dtype=np.int32),
        dtype=wp.int32,
        device=mesh_wp.points.device,
    )
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)

    internal_np = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp).numpy()
    first_np = od.heat.heat_geodesic(mesh, sources_wp).numpy()
    second_np = od.heat.heat_geodesic(mesh, sources_wp).numpy()

    assert internal_np.max() > 0.0
    assert np.allclose(first_np, internal_np, rtol=1e-12, atol=1e-12)
    assert mesh.heat_solver().nbytes > 0
    assert np.abs(second_np - internal_np).max() < 1e-6 * np.ptp(internal_np)


def _heat_geodesic_igl(
    vertices_np: np.ndarray, faces_np: np.ndarray, sources_np: np.ndarray
) -> np.ndarray:
    data = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(vertices_np, faces_np, data)
    return np.asarray(igl.heat_geodesics_solve(data, sources_np))


@pytest.mark.parametrize(
    ("mesh_name", "n_sources"),
    [("icosahedron", 1), ("hemisphere", 1), ("icosahedron", 2)],
    ids=["icosahedron", "hemisphere", "icosahedron_two_sources"],
)
@pytest.mark.parity("heat_geodesic", "igl")
@pytest.mark.parity("heat_geodesic_conditioning", "igl")
def test_heat_geodesic_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, n_sources: int
) -> None:
    """
    Class A at 5e-2: the same method, but igl factorizes where ordito runs conjugate gradient.

    The tolerance is the discretization the two share plus each solver's own stopping point,
    not a disagreement about the method -- both are Crane et al. on the identical
    triangulation, which is why ``_heat_geodesic_igl`` is built with matching defaults rather
    than igl's own.

    The two-source arm is the multi-source form, where the answer is the distance to the
    *nearest* source: a single-source implementation passes the other arms while getting the
    reduction over several sources wrong -- the field is not a sum but a minimum.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    sources_np = np.array([0, len(vertices_np) // 2][:n_sources], dtype=np.int64)
    sources_wp = wp.array(sources_np.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)

    distance_igl = _heat_geodesic_igl(vertices_np, faces_np, sources_np)
    distance_wp = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp)

    assert np.allclose(distance_wp.numpy(), distance_igl, rtol=5e-2, atol=5e-2)


@pytest.mark.parity(
    "heat_geodesic",
    "meshlib",
    benchmarked=False,
    reason="computeSurfaceDistances is fast marching, not the heat method -- it advances a serial "
    "front where ordito solves two sparse systems -- so its row belongs in the "
    "fast_marching_distance group beside potpourri3d, pymeshlab and igl's exact_geodesic, and that "
    "is where it is timed. Putting it in heat_geodesic would price a different algorithm under "
    "that group's name. The values are nonetheless comparable, which is what this test checks.",
)
def test_heat_geodesic_matches_meshlib(device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (a bound against the truth): two approximations of the same geodesic field.

    ``computeSurfaceDistances`` is fast marching -- a serial front over the edge graph solving the
    local Eikonal update -- where ``heat_geodesic`` is Crane et al.'s heat method. Neither is exact
    on a triangulation, so comparing them to each other alone would be a claim about two errors. The
    sphere is the fixture that fixes that: its geodesics are great-circle arcs, so the *exact*
    answer is known and each method can be measured against it.

    Measured from one source on ``icosphere(3)``, as the worst absolute deviation from the exact
    arc length: ordito **0.0515** and meshlib **0.0442**, i.e. 1.6 % and 1.4 % of the field's
    range. The two agree with each other to **0.067** with a correlation of **0.99983** -- so they
    differ from each other by about as much as each differs from the truth, which is the honest
    statement about two first-order methods and is what the asserts encode.

    Its ``startVertices`` is a ``VertBitSet`` over the vertex domain -- there is no index-list
    overload for the single-source case -- and the result is a ``VertScalars``, read through
    [`meshlib_scalars_to_numpy`][tests.conversions.meshlib_scalars_to_numpy] since ``np.asarray``
    on one silently yields a 0-d object array.

    **Mutation probe**, and what it says is that the three asserts divide the work unevenly. The
    measured errors against the exact field are 0.0515 (ordito) and 0.0442 (meshlib) against a
    0.1571 bar, and the correlation is 0.99983 against a 0.999 bar. Mutating the ordito field:

    | mutation      | error vs the 5 % bar | correlation vs the 0.999 bar |
    |---------------|----------------------|------------------------------|
    | shuffle       | 2.753 -- **fails**   | -0.014 -- **fails**          |
    | scale x1.10   | 0.273 -- **fails**   | 0.99983 -- passes            |
    | scale x1.05   | 0.124 -- passes      | 0.99983 -- passes            |

    So the **correlation cannot see a scale error at all** (it is scale-invariant), and the error
    bar is what carries that half -- but only past ~5 %, because the bar *is* 5 % and the field
    already spends a third of it on discretization. That is the honest limit of this comparison and
    not a threshold to tighten: 5 % is where two first-order methods genuinely sit, which the
    paragraph above already argues. A finer claim needs a finer mesh, not a smaller number.
    """
    mesh_tm, mesh_wp = icosphere
    source = 0
    centre_np = mesh_tm.vertices.mean(axis=0)
    directions_np = mesh_tm.vertices - centre_np
    directions_np /= np.linalg.norm(directions_np, axis=1, keepdims=True)
    radius = float(np.linalg.norm(mesh_tm.vertices[source] - centre_np))
    exact_np = radius * np.arccos(np.clip(directions_np @ directions_np[source], -1.0, 1.0))

    distance_wp = od.heat.heat_geodesic(
        mesh_wp.points, mesh_wp.indices, wp.array([source], dtype=wp.int32, device=device)
    ).numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    starts_ml = mm.VertBitSet()
    starts_ml.resize(mesh_ml.points.size(), False)
    starts_ml.set(mm.VertId(source), True)
    distance_ml = meshlib_scalars_to_numpy(mm.computeSurfaceDistances(mesh_ml, starts_ml))

    assert distance_ml.shape == distance_wp.shape
    assert distance_ml.max() > 0.9 * exact_np.max()  # non-vacuity: it really propagated
    # Each within 5 % of the exact great-circle field, and closer to each other than that.
    assert np.abs(distance_wp - exact_np).max() < 0.05 * exact_np.max()
    assert np.abs(distance_ml - exact_np).max() < 0.05 * exact_np.max()
    assert np.corrcoef(distance_wp, distance_ml)[0, 1] > 0.999


@pytest.mark.parametrize("n_sources", [1, 3])
def test_heat_geodesic_matches_igl_far_from_the_sources(
    device: str, icosphere5: Icosphere5, n_sources: int
) -> None:
    """
    Class A at 5e-3 of the distance range, on a mesh large enough to have a far field.

    ``icosphere(5)`` (10 242 vertices, 96 rings from pole to pole) rather than a ``conftest``
    fixture, because the defect this pins is invisible below about a dozen rings, which is every
    fixture: a conjugate-gradient heat solve stopped on its residual is exactly zero more than its
    iteration count away from the sources, and it used to stop after ~30 rounds -- leaving 92 % of
    this sphere without heat and the distance up to 2.3 off, against igl's 0.019. Measured
    agreement now 1e-4 (one source) and 1.3e-3 (three) of the range, igl factorizing where
    ordito iterates until every vertex has converged relative to itself.
    """
    mesh_tm, vertices_wp, faces_wp = icosphere5
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    n_vertices = len(vertices_np)
    sources_np = np.array([0, n_vertices // 2, n_vertices // 3][:n_sources], dtype=np.int64)
    distance_igl = _heat_geodesic_igl(vertices_np, faces_np, sources_np)
    distance_wp = od.heat.heat_geodesic(
        vertices_wp, faces_wp, wp.array(sources_np.astype(np.int32), dtype=wp.int32, device=device)
    ).numpy()
    assert np.ptp(distance_igl) > 1.0
    assert np.abs(distance_wp - distance_igl).max() < 5e-3 * np.ptp(distance_igl)


@pytest.mark.parametrize("reuse", [False, True], ids=["one_shot", "trimesh"])
def test_heat_geodesic_on_a_graded_mesh_matches_potpourri3d(
    device: str, graded_saddle: GradedSaddle, reuse: bool
) -> None:
    """
    Class A at 1e-3 of the distance range against geometry-central's factorized heat method.

    The settle rule stopped the heat solve on round-off: on this mesh conjugate gradient's
    iterate "settles" with the far field wrong by up to 1e30 relative, and the distance was 38 % of
    the range off potpourri3d (and 4.4 % mean, 38 % worst off ``igl.exact_geodesic``, against the
    method's own 1.1 % / 6.2 %). The settled iterate's componentwise backward error flags it and
    the solve falls back to a direct factorization. Measured agreement 2.6e-4 of the range, both
    one-shot and on a `Trimesh` (whose second call goes to the kept factorizations directly).
    """
    vertices_np, faces_np, vertices_wp, faces_wp = graded_saddle
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    mesh = od.Trimesh(vertices_wp, faces_wp)
    for _ in range(2 if reuse else 1):
        distance_wp = (
            od.heat.heat_geodesic(mesh, sources_wp)
            if reuse
            else od.heat.heat_geodesic(vertices_wp, faces_wp, sources_wp)
        ).numpy()
        distance_pp = pp3d.MeshHeatMethodDistanceSolver(
            vertices_np, faces_np.astype(np.int32), use_robust=False
        ).compute_distance(0)
        assert np.ptp(distance_pp) > 1.0
        assert np.abs(distance_wp - distance_pp).max() < 1e-3 * np.ptp(distance_pp)


def test_heat_geodesic_approximates_exact(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A at 8e-2 relative and 1e-1 absolute: ``igl.exact_geodesic`` is the true distance.

    The heat method only approximates it, so the tolerance is the method's own first-order error
    on a 12-vertex mesh rather than a solver disagreement.
    """
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    n_vertices = len(vertices_np)
    empty = np.array([], dtype=np.int64)
    sources_np = np.array([0], dtype=np.int64)

    distance_exact = igl.exact_geodesic(
        vertices_np, faces_np, sources_np, empty, np.arange(n_vertices, dtype=np.int64), empty
    )
    sources_wp = wp.array(sources_np.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)
    distance_wp = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp)

    # Heat method is an approximation of the true geodesic distance; a loose tolerance.
    assert np.allclose(distance_wp.numpy(), distance_exact, rtol=8e-2, atol=1e-1)


def test_heat_geodesic_source_is_zero_and_nonnegative(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = hemisphere
    sources_np = np.array([0], dtype=np.int32)
    sources_wp = wp.array(sources_np, dtype=wp.int32, device=mesh_wp.device)

    distance = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp).numpy()

    assert np.all(distance >= -1e-6)
    assert np.allclose(distance[sources_np], 0.0, atol=1e-4)


@pytest.mark.parametrize("mesh_name", _HEAT_MESHES_SMALL)
def test_heat_geodesic_cpu_matches_cuda(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the CPU solve is the CUDA solve. Pins the removal of the old CUDA-only guard.

    ``warp.optim.linear.cg`` returned NaN on the Warp CPU device through 1.15, so every entry point
    reaching a solve refused to run there. Fixed in 1.16, verified here rather than only in a probe.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    mesh_tm, _ = request.getfixturevalue(mesh_name)

    distances = {}
    for device in ("cpu", "cuda:0"):
        mesh_wp = trimesh_to_warp(mesh_tm, device)
        sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
        distances[device] = od.heat.heat_geodesic(
            mesh_wp.points, mesh_wp.indices, sources_wp
        ).numpy()

    assert np.isfinite(distances["cpu"]).all()
    assert distances["cpu"].max() > 0.0
    assert np.allclose(distances["cpu"], distances["cuda:0"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _HEAT_MESHES)
@pytest.mark.parity("heat_geodesic", "potpourri3d", "pymeshlab")
@pytest.mark.parity("heat_geodesic_conditioning", "potpourri3d", "pymeshlab")
def test_heat_geodesic_matches_potpourri3d_plain_and_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The plain heat method against geometry-central and MeshLab on the *same* discretization.

    Class C for both: an error norm against the mesh diameter rather than element-wise, because the
    sides still differ in the *solver* -- ordito runs conjugate gradient to a tolerance where
    geometry-central and MeshLab factor directly, so the residuals differ even though the systems
    match. Mean and maximum are held separately so a single blown-up vertex cannot hide inside a
    mean taken over thousands.

    **potpourri3d.** Distinct from ``test_robust_heat_geodesic_matches_potpourri3d``, which runs
    both sides with ``use_robust=True``. That is a different configuration from the one
    ``benchmarks/test_heat.py`` times, and it is the looser of the two comparisons: potpourri3d's
    robust path additionally flips to an intrinsic Delaunay triangulation, so the two solve on
    different triangulations and can only agree to the heat method's own accuracy. Passing
    ``use_robust=False`` on both sides removes that difference -- same mesh, same cotangent
    weights, same lumped mass -- which is what makes this the honest oracle for the benchmark row
    and lets the tolerance be far tighter than the robust comparison's ``0.1 * scale``. Measured
    across the three fixtures: mean error **0.00 / 0.00 / 0.01%** of the diameter and maximum
    **0.00 / 0.00 / 0.99%**, the worst being ``half_torus``, whose non-uniform scaling gives it the
    widest triangle-quality spread. The bounds sit 10x and 5x off those, which is the margin rule;
    they are this tight *because* both sides discretize identically, and a regression to the robust
    comparison's ``0.1 * scale`` would mean the two are no longer solving the same system.

    **pymeshlab**, a fourth independent implementation of the same PDE and the cheapest strong
    check on it. The named transform is how the source is specified: MeshLab has no source argument
    at all and takes the current *selection*, so ``compute_selection_by_condition_per_vertex`` with
    ``"(vi == 0)"`` is what pins it to vertex 0, and the answer is read off
    ``vertex_scalar_array()`` rather than returned. Exactly what the benchmark does. Measured:
    mean error **0.000 / 0.002 / 0.019%** of the diameter and maximum **0.000 / 0.006 / 0.966%**
    -- within a factor of two of the potpourri3d numbers, on a completely separate codebase, under
    the same bounds (>50x and 5x off the measured values).

    **Bug class excluded:** a wrong timestep or a wrong mass lumping. Both leave the field smooth,
    monotone and zero at the source -- so they pass every self-consistency test in this module --
    and both shift the whole field by several percent, which two independent references pin down.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)

    distance_wp = od.heat.heat_geodesic(
        mesh_wp.points, mesh_wp.indices, sources_wp, use_robust=False
    ).numpy()
    distance_pp = np.asarray(
        pp3d.MeshHeatMethodDistanceSolver(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            use_robust=False,
        ).compute_distance(0)
    )
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_by_condition_per_vertex(condselect="(vi == 0)")
    meshset_pml.compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex()
    distance_pml = np.asarray(meshset_pml.current_mesh().vertex_scalar_array())

    diameter = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    for distance_ref in (distance_pp, distance_pml):
        assert distance_ref.max() > 0.1 * diameter  # non-vacuity: the reference propagated
        error = np.abs(distance_wp - distance_ref)
        assert error.mean() < 0.001 * diameter
        assert error.max() < 0.05 * diameter


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
@pytest.mark.parity("heat_geodesic", "pyvista")
def test_heat_geodesic_is_bounded_by_the_graph_distance(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C, a two-sided bound on a value with no correspondence to compare against.

    VTK's ``geodesic_distance`` is ``vtkDijkstraGraphGeodesicPath`` -- the shortest path along mesh
    *edges*, so it is an **upper** bound on the true geodesic (a path confined to edges cannot beat
    one free to cross faces), while the straight-line distance is a **lower** bound on it. The heat
    method returns a smoothed approximation of the quantity in between, which is why this is a bound
    and not an ``allclose``: nothing here is the same number.

    **Bug class excluded:** a field on the wrong *scale*. A wrong timestep, a missing mass lumping
    or a lost square root all leave the field smooth, monotone and zero at the source -- so every
    self-consistency test in this module still passes -- while moving its magnitude, and the upper
    bound is sharp enough to see that. Measured ``heat / graph`` over 20 random targets per fixture:
    max **0.866 / 0.983 / 0.934 / 0.969**, so a field inflated by more than ~2% on ``hemisphere``
    fails it (a 1.5x scaling fails on all four).

    The lower bound is checked only at the Euclidean-**farthest** vertex, and that restriction is
    the finding rather than a convenience: at short range the heat method *undershoots* the
    straight-line distance, measured 0.911 against 1.051 between adjacent-ish vertices of the
    icosahedron and 0.728 against 0.829 on ``half_torus``. So "heat >= Euclidean" is false in
    general and true where the two are far apart -- margins there are 36% / 53% / 0.5% / 30%, the
    thin one being ``half_torus``, whose farthest pair is nearly straight-line reachable across its
    opening.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    heat_np = od.heat.heat_geodesic(
        mesh_wp.points, mesh_wp.indices, sources_wp, use_robust=False
    ).numpy()

    rng = np.random.default_rng(3)
    targets_np = rng.choice(np.arange(1, n_vertices), size=min(20, n_vertices - 1), replace=False)
    for target in targets_np:
        graph_pv = float(mesh_pv.geodesic_distance(0, int(target)))
        assert graph_pv > 0.0  # the reference answered, so the bound is not vacuous
        assert heat_np[int(target)] <= graph_pv

    # The lower bound bites only at range -- see the docstring.
    farthest = int(np.argmax(np.linalg.norm(mesh_tm.vertices - mesh_tm.vertices[0], axis=1)))
    euclidean_np = float(np.linalg.norm(mesh_tm.vertices[farthest] - mesh_tm.vertices[0]))
    assert euclidean_np <= heat_np[farthest] <= float(mesh_pv.geodesic_distance(0, farthest))


# --- the heat method's robust path (potpourri3d use_robust=True reference) -------------
@pytest.mark.parametrize("mesh_name", _HEAT_MESHES)
def test_robust_heat_geodesic_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the ``use_robust=True`` path, against potpourri3d's identically-named flag.

    The two flags are *not* the same operation and the module docstring says so: geometry-
    central's includes an intrinsic Delaunay retriangulation that ordito's deliberately omits.
    On a clean mesh mollification changes nothing, so the comparison holds there -- which is
    why the fixtures are clean ones and the degenerate case is a separate invariant test.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)

    distance_wp = od.heat.heat_geodesic(
        mesh_wp.points, mesh_wp.indices, sources_wp, use_robust=True
    )
    distance_pp = np.asarray(
        pp3d.MeshHeatMethodDistanceSolver(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            use_robust=True,
        ).compute_distance(0)
    )

    # potpourri3d's robust path also flips to an intrinsic Delaunay triangulation, which this does
    # not (see the module docstring), so the two agree to the heat method's own accuracy rather than
    # tightly. The comparison is still worth making: it is the configuration potpourri3d ships.
    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert np.abs(distance_wp.numpy() - distance_pp).mean() < 0.1 * scale


def test_robust_heat_geodesic_survives_a_degenerate_triangle(
    device: str, sliver_patch: tuple[np.ndarray, np.ndarray, wp.array[wp.vec3], wp.array[wp.int32]]
) -> None:
    _, _, vertices_wp, faces_wp = sliver_patch
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)

    plain = od.heat.heat_geodesic(vertices_wp, faces_wp, sources_wp).numpy()
    robust = od.heat.heat_geodesic(vertices_wp, faces_wp, sources_wp, use_robust=True).numpy()

    # Both are finite: the cotangent assembly refuses to divide by a degenerate face's zero area.
    # What ``use_robust`` buys is *accuracy* -- the plain operator loses that face's edge couplings
    # (see ``test_robust_laplacian_keeps_couplings_the_plain_one_drops``), so its distance across
    # the collapsed edge is worse. Vertices 0 and 1 are one unit apart in a straight line.
    assert np.isfinite(plain).all()
    assert np.isfinite(robust).all()
    assert robust[0] == pytest.approx(0.0, abs=1e-6)
    assert abs(robust[1] - 1.0) < abs(plain[1] - 1.0)


# --------------------------------------------------------------------------
# heat_signed_distance
# --------------------------------------------------------------------------


def _one_ring_cycle(
    mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh, which: int = 0
) -> tuple[int, np.ndarray, wp.array[wp.int32]]:
    """
    Return an interior vertex and the counter-clockwise cycle of its neighbours, host and device.

    The centre has to be an *interior* vertex: a boundary vertex's fan is open, so its neighbours do
    not close into a cycle and the last "segment" would be a chord across the surface rather than an
    edge — which both this method and the reference read as a different curve entirely.
    """
    ring, offsets, is_boundary = (
        array.numpy()
        for array in od.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))
    )
    interior = np.flatnonzero(~is_boundary)
    center = int(interior[which % len(interior)])
    faces_np = np.asarray(mesh_tm.faces)
    halfedges = ring[offsets[center] : offsets[center + 1]]
    curve_np = np.array([faces_np[h // 3][(h % 3 + 1) % 3] for h in halfedges], dtype=np.int32)
    return center, curve_np, wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)


def _distance_pp(
    mesh_tm: tm.Trimesh, curve: np.ndarray, level_set_constraint: str = "ZeroSet"
) -> np.ndarray:
    solver_pp = pp3d.MeshSignedHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
    )
    return np.asarray(
        solver_pp.compute_distance(
            [[(int(vertex), []) for vertex in curve]], level_set_constraint=level_set_constraint
        )
    )


# ---------------------------------------------------------------------------
# heat_signed_distance
# ---------------------------------------------------------------------------


# ``cave_cube`` and ``half_torus`` are left out, for the same reason as in the vector heat section
# below: every one of their faces carries an edge whose two opposite angles are right angles, so
# that edge's cotangent weight is exactly zero. potpourri3d cannot even factor ``cave_cube`` (it
# returns NaN), and on ``half_torus`` the two libraries agree only to a correlation of 0.39 — both
# are solving a degenerate problem there and degrade differently. The structural tests below still
# cover ``half_torus``, and they pass.
@pytest.mark.parity(
    "heat_signed_distance_constraint",
    "potpourri3d",
    benchmarked=False,
    reason="potpourri3d is already timed in the heat_signed_distance group and this is the same "
    "compute_distance call with one keyword changed, so a second row would time the identical "
    "solve under a second name. What the constraint costs is the delta between the two ids, which "
    "is a ordito-internal question; whether it is *honoured* is what this checks.",
)
@pytest.mark.parametrize(
    ("level_set_constraint", "constraint_pp"), [("zero_set", "ZeroSet"), ("none", "None")]
)
def test_heat_signed_distance_level_set_constraint_matches_potpourri3d(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], level_set_constraint: str, constraint_pp: str
) -> None:
    """
    Class C (correlation plus a mean-error bound), on both settings of the constraint.

    Same statistic and same reasoning as
    [`test_heat_signed_distance_matches_potpourri3d`] -- the two boundary handlings differ at the
    curve, so no vertex-wise tolerance exists -- but parametrized over the keyword the
    ``heat_signed_distance_constraint`` benchmark group is an axis on. potpourri3d spells the same
    two settings ``"ZeroSet"`` and ``"None"`` and honours them the same way.

    **The constraint is observable, which is what keeps this from being one comparison run twice.**
    Under ``zero_set`` both libraries pin the curve to exactly ``0.0``; under ``none`` neither does,
    and the measured on-curve magnitudes on this fixture are 5.57e-02 (ordito) and 6.94e-02
    (potpourri3d). So the last two asserts fail if either side silently ignores the keyword -- the
    bug this exists for, and one a correlation bound alone would pass, since the two fields
    correlate 0.940 and 0.936 respectively, i.e. the *statistic does not separate the two modes at
    all*. The 1e-3 floor is 18x below the smaller measured magnitude.

    **Mutation probe.** The paragraph above already *is* one for the last two asserts -- it names
    the bug (a silently ignored keyword), shows the correlation cannot see it, and records the 18x
    floor. For the two statistical asserts the probe is the sibling's, on this same fixture and
    these same two bounds: shuffling one side gives a mean error of **0.478 against the 0.150 bar**
    and negating it gives ``r = -0.94``, so both fail. What that probe also found and this test
    inherits: the ``hemisphere`` error bound has only **1.19x** headroom on unshuffled input.
    """
    mesh_tm, mesh_wp = hemisphere
    _, curve_np, curve_wp = _one_ring_cycle(mesh_tm, mesh_wp)

    distance_wp = od.heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, level_set_constraint=level_set_constraint
    ).numpy()
    distance_pp = _distance_pp(mesh_tm, curve_np, level_set_constraint=constraint_pp)

    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert np.corrcoef(distance_wp, distance_pp)[0, 1] > 0.9
    assert np.abs(distance_wp - distance_pp).mean() < 0.05 * scale

    # The keyword really took, on both sides: pinned to zero, or demonstrably not pinned.
    on_curve_wp = np.abs(distance_wp[curve_np]).max()
    on_curve_pp = np.abs(distance_pp[curve_np]).max()
    if level_set_constraint == "zero_set":
        assert on_curve_wp == 0.0
        assert on_curve_pp == 0.0
    else:
        assert on_curve_wp > 1e-3
        assert on_curve_pp > 1e-3


@pytest.mark.parametrize("mesh_name", ["icosahedron"])
@pytest.mark.parity("heat_signed_distance", "potpourri3d")
@pytest.mark.parity("heat_signed_distance_conditioning", "potpourri3d")
def test_heat_signed_distance_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C (correlation plus a mean-error bound): the two fields have no correspondence to assert.

    Both sides solve the same continuous problem but through different boundary handling at the
    curve, so a vertex-wise ``allclose`` is not available at any tolerance -- what *is* shared is
    the sign convention and the shape of the field. The bug class this excludes is the one that
    matters here: a sign flip, a scale error, or a field ignoring the curve would each break a 0.9
    correlation and a mean error under 5 % of the mesh diagonal. Measured margins are in the comment
    below, and the correlation is the tighter constraint (0.996 and 0.940 against the 0.9 bar).

    **Mutation probe** (30 permutations, both fixtures), and it shows the two statistics are
    complementary rather than redundant -- neither alone excludes the whole bug class:

    | mutation             | correlation      | mean error vs its bar        |
    |----------------------|------------------|------------------------------|
    | shuffle one side     | 0.73 / 0.21      | 0.487 / 0.478 vs 0.147 / 0.150 -- **fails** |
    | negate one side      | -0.996 / -0.940 -- **fails** | unchanged in magnitude |
    | scale one side x1.5  | unchanged (scale-invariant) | 0.284 / 0.533 -- **fails** |

    So the correlation catches the sign flip and the error bound catches the scale error and the
    shuffle, which is the division of labour the docstring above claims. One caveat worth carrying:
    on ``hemisphere`` the measured mean error is **0.1259 against a 0.1500 bar -- 1.19x**, well
    inside section 7.4's 3x preference, because that curve sits one ring from the rim where the two
    boundary handlings diverge most. Do not tighten that bar without re-measuring both fixtures.

    The ``hemisphere`` comparison is the ``zero_set`` arm of
    [`test_heat_signed_distance_level_set_constraint_matches_potpourri3d`] -- the default
    constraint, the same curve and the same two bounds -- so it runs there, once.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np, curve_wp = _one_ring_cycle(mesh_tm, mesh_wp)

    distance_wp = od.heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, curve_wp).numpy()
    distance_pp = _distance_pp(mesh_tm, curve_np)

    # Same sign convention (a counter-clockwise curve encloses the positive side) and the same field
    # to a few percent of the mesh scale: measured 1.6% on ``icosahedron`` and 4.0% on
    # ``hemisphere``.
    # The correlation is the looser of the two statistics here -- 0.996 and 0.940 -- because this
    # curve sits one ring from the hemisphere's rim, where the reference's boundary handling and
    # this
    # one's diverge; taking the centre-most interior vertex instead gives 0.990.
    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert np.corrcoef(distance_wp, distance_pp)[0, 1] > 0.9
    assert np.abs(distance_wp - distance_pp).mean() < 0.05 * scale


@pytest.mark.parametrize("mesh_name", _HEAT_MESHES_SMALL)
def test_signed_distance_magnitude_is_the_unsigned_distance(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, _, curve_wp = _one_ring_cycle(mesh_tm, mesh_wp)

    signed = od.heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, curve_wp).numpy()
    unsigned = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, curve_wp).numpy()

    # Two independent solves — a vector diffusion plus Poisson against a scalar diffusion plus
    # Poisson — that have to agree about *how far* the curve is, whatever they say about which side.
    # Measured 2.6% of the span on ``icosahedron`` and 9.9% on ``hemisphere``, where the unsigned
    # method's Neumann boundary and the signed field's behaviour at the rim pull apart.
    span = float(unsigned.max())
    assert np.abs(np.abs(signed) - unsigned).mean() < 0.15 * span


@pytest.mark.parametrize("mesh_name", _HEAT_MESHES)
def test_signed_distance_is_positive_inside_the_curve(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    center, curve_np, curve_wp = _one_ring_cycle(mesh_tm, mesh_wp)

    distance = od.heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, curve_wp).numpy()

    # The ring encloses exactly one vertex, and the sign convention puts that side positive; every
    # other vertex is outside it and must come out negative.
    assert distance[center] > 0.0
    outside = np.ones(len(mesh_tm.vertices), dtype=bool)
    outside[curve_np] = False
    outside[center] = False
    assert (distance[outside] < 0.0).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_reversing_the_curve_negates_the_field(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np, _ = _one_ring_cycle(mesh_tm, mesh_wp)

    forward = od.heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)
    ).numpy()
    backward = od.heat.heat_signed_distance(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(curve_np[::-1].copy(), dtype=wp.int32, device=mesh_wp.device),
    ).numpy()

    # Orientation *is* the sign: nothing else about the source changed.
    assert np.allclose(forward, -backward, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_zero_set_constraint_pins_the_curve(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np, curve_wp = _one_ring_cycle(mesh_tm, mesh_wp)

    pinned = od.heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, level_set_constraint="zero_set"
    ).numpy()
    shifted = od.heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, level_set_constraint="none"
    ).numpy()

    # ``zero_set`` pins exactly; ``none`` only centres the curve's mean, so individual curve
    # vertices
    # sit slightly off zero.
    assert np.array_equal(pinned[curve_np], np.zeros(len(curve_np)))
    assert abs(shifted[curve_np].mean()) < 1e-9
    # Both agree away from the curve, where the constraint does not act.
    span = float(np.abs(pinned).max())
    assert np.abs(pinned - shifted).mean() < 0.1 * span


def test_multiple_curves_via_offsets(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    _, first, first_wp = _one_ring_cycle(mesh_tm, mesh_wp, which=0)
    _, second, _ = _one_ring_cycle(mesh_tm, mesh_wp, which=5)
    packed = wp.array(np.concatenate([first, second]), dtype=wp.int32, device=mesh_wp.device)
    offsets = wp.array(
        np.array([0, len(first), len(first) + len(second)], dtype=np.int32),
        dtype=wp.int32,
        device=mesh_wp.device,
    )

    both = od.heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, packed, offsets).numpy()
    only_first = od.heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, first_wp).numpy()

    # Two sources pin two zero sets, so the combined field differs from either alone but still
    # vanishes on both curves.
    assert np.array_equal(both[np.concatenate([first, second])], np.zeros(len(first) + len(second)))
    assert not np.allclose(both, only_first, rtol=1e-2, atol=1e-2)


def test_open_curve_still_changes_sign_across_itself(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = icosahedron
    _, curve_np, _ = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_np = curve_np[:3]
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)

    # An open curve has no inside, so the far field means nothing — but the method does not need a
    # closed curve to run, and the result must still be finite and vanish on the source.
    distance = od.heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, closed=False
    ).numpy()
    assert np.isfinite(distance).all()
    assert np.array_equal(distance[curve_np], np.zeros(len(curve_np)))
    assert distance.min() < 0.0 < distance.max()


def test_invalid_level_set_constraint(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="level_set_constraint"):
        od.heat.heat_signed_distance(
            mesh_wp.points,
            mesh_wp.indices,
            wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
            level_set_constraint="Multiple",
        )


# --------------------------------------------------------------------------
# vector_heat_operators / extend_scalar / transport_tangent_vectors / log_map
# --------------------------------------------------------------------------


def _solver_pp(mesh_tm: tm.Trimesh) -> pp3d.MeshVectorHeatSolver:
    return pp3d.MeshVectorHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
        use_intrinsic_delaunay=False,
    )


def _frames(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_x, basis_y, normal = od.tangent_space.vertex_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )
    return basis_x.numpy(), basis_y.numpy(), normal.numpy()


def _to_world(tangent: np.ndarray, basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
    return tangent[:, 0:1] * basis_x + tangent[:, 1:2] * basis_y


def _unit_transport_source(device: wp.DeviceLike) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Vertex 0 as the single source, carrying the unit vector ``(1, 0)`` of its own frame."""
    return (
        wp.array([0], dtype=wp.int32, device=device),
        wp.array([[1.0, 0.0]], dtype=wp.vec2, device=device),
    )


# ---------------------------------------------------------------------------
# extend_scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("extend_scalar", "potpourri3d")
def test_extend_scalar_matches_potpourri3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: a scalar field carries no gauge, so this is the one vector-heat comparison that is.

    Everything else in this module measures a *tangent* quantity, which agrees only up to a rotation
    about the normal (section 6). ``extend_scalar`` returns numbers, so it is compared elementwise,
    which makes it the test pinning the shared diffusion machinery all the others build on.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    sources_np = np.array([0, n_vertices // 3, 2 * n_vertices // 3], dtype=np.int32)
    values_np = np.array([1.0, 2.0, 5.0])

    extended_wp = od.heat.extend_scalar(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(sources_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(values_np, dtype=wp.float64, device=mesh_wp.device),
    )
    extended_pp = np.asarray(
        _solver_pp(mesh_tm).extend_scalar(sources_np.tolist(), values_np.tolist())
    )

    # Nothing here depends on a frame, so this one is a direct comparison. The two libraries solve
    # the same system differently (conjugate gradient against a Cholesky factorization).
    #
    # ``icosahedron`` and ``hemisphere`` agree to the last bit. ``half_torus`` has exactly one
    # vertex of 544 outside the band, at 0.065 on a source range of [1, 5]; the rest agree to a mean
    # of 3.3e-4. That one vertex is *not* a regression from the timestep convention, which is worth
    # recording because it looks like one: sweeping ``t`` around the default shows the mean error
    # has a clean minimum exactly at the ``mean_unique_edge_length`` value used here --
    # 0.00085 / 0.00059 / **0.00033** / 0.00051 / 0.00153 at t x 0.995 / 0.998 / 1.000 / 1.002 /
    # 1.011 of it -- so this is the timestep geometry-central uses, and the per-face average
    # (x1.011) is 4.6x worse on the mean. The *maximum* falls monotonically across that whole
    # sweep and so tracks nothing: it is one vertex in a steep part of the field that smooths out
    # as t grows.
    #
    # Hence a mean-and-fraction bound rather than a max-only one, and no widening of the band for
    # all 544 to absorb a single point.
    within_band = np.abs(extended_wp.numpy() - extended_pp) <= 2e-2 + 1e-2 * np.abs(extended_pp)
    assert within_band.mean() > 0.99
    assert np.abs(extended_wp.numpy() - extended_pp).max() < 0.1
    assert np.abs(extended_wp.numpy() - extended_pp).mean() < 1e-3
    # The extension interpolates: it never leaves the range of its sources.
    assert extended_wp.numpy().min() >= values_np.min() - 1e-6
    assert extended_wp.numpy().max() <= values_np.max() + 1e-6


def test_extend_scalar_matches_potpourri3d_far_from_the_sources(
    device: str, icosphere5: Icosphere5
) -> None:
    """
    Class A on ``icosphere(5)``: the extension has a value everywhere, not only near its sources.

    The far-field counterpart of the comparison above, which runs on fixtures too small to have a
    far field (see ``test_heat_geodesic_matches_igl_far_from_the_sources`` for why). The extension
    used to return zero wherever the diffused indicator fell below ``1e-12`` of its maximum, which
    on this sphere was most of it: mean error 1.39 on a source range of [1, 3]. Measured agreement
    now below 1e-6.
    """
    mesh_tm, vertices_wp, faces_wp = icosphere5
    sources_np = np.array([0, len(mesh_tm.vertices) // 2], dtype=np.int32)
    values_np = np.array([1.0, 3.0])
    extended_wp = od.heat.extend_scalar(
        vertices_wp,
        faces_wp,
        wp.array(sources_np, dtype=wp.int32, device=device),
        wp.array(values_np, dtype=wp.float64, device=device),
    ).numpy()
    extended_pp = np.asarray(
        _solver_pp(mesh_tm).extend_scalar(sources_np.tolist(), values_np.tolist())
    )
    assert np.ptp(extended_pp) > 1.9
    assert np.abs(extended_wp - extended_pp).max() < 1e-3


def test_extend_scalar_single_source_is_constant(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    extended = od.heat.extend_scalar(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(np.array([3.5]), dtype=wp.float64, device=mesh_wp.device),
    )
    # One source has nothing to blend against, so its value must fill the surface.
    assert np.allclose(extended.numpy(), 3.5, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# transport_tangent_vectors
# ---------------------------------------------------------------------------


def test_transport_on_a_flat_patch_is_constant(device: str) -> None:
    # Parallel transport across a plane is the identity, so the transported field must be one
    # constant
    # world vector. Away from the rim, where the angle sum is exactly 2*pi and the intrinsic
    # flattening is trivial, that is exact.
    size = 9
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size))
    vertices_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(size * size)], axis=1)
    faces_np = np.array(
        [
            quad
            for row in range(size - 1)
            for column in range(size - 1)
            for quad in (
                [row * size + column, row * size + column + 1, (row + 1) * size + column + 1],
                [row * size + column, (row + 1) * size + column + 1, (row + 1) * size + column],
            )
        ],
        dtype=np.int32,
    )
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)

    transported, _ = od.heat.transport_tangent_vectors(
        vertices_wp,
        faces_wp,
        wp.array(np.array([size * size // 2], dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device),
    )
    basis_x, basis_y, _ = (
        basis.numpy() for basis in od.tangent_space.vertex_tangent_frames(vertices_wp, faces_wp)
    )
    world = _to_world(transported.numpy(), basis_x, basis_y)
    world /= np.linalg.norm(world, axis=1, keepdims=True)

    interior = ~od.halfedge.vertex_one_rings(faces_wp, n_vertices=len(vertices_np))[2].numpy()
    reference = world[interior][0]
    assert np.allclose(world[interior] @ reference, 1.0, rtol=1e-4, atol=1e-4)
    # Magnitude is carried by the scalar extension, so it is preserved everywhere.
    assert np.allclose(
        np.linalg.norm(_to_world(transported.numpy(), basis_x, basis_y), axis=1),
        1.0,
        rtol=1e-4,
        atol=1e-4,
    )


# ``cave_cube`` is left out: potpourri3d returns NaN on it. Its quad faces are split by diagonals
# whose two opposite angles are both right angles, so those cotangent weights are exactly zero and
# geometry-central's direct factorization of the connection Laplacian fails ("factorization
# failed").
# It is covered by the disconnected-component test below instead.
@pytest.mark.parametrize("mesh_name", _HEAT_MESHES)
@pytest.mark.parity("transport_tangent_vectors", "potpourri3d")
@pytest.mark.parity("vector_heat_scale", "potpourri3d")
def test_transport_tangent_vectors_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (gauge fix): both 2-D fields are pushed to 3-D world vectors before comparing.

    A tangent component is meaningless across libraries -- each measures from its own reference
    direction -- so the named transform expresses both answers in each library's *own* frames and
    compares the resulting world vectors, which are gauge-invariant. Comparing the raw ``vec2``
    components instead is exactly what section 6 forbids here.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solver_pp = _solver_pp(mesh_tm)
    basis_x_pp, basis_y_pp, _ = (np.asarray(basis) for basis in solver_pp.get_tangent_frames())
    basis_x, basis_y, _ = _frames(mesh_wp)

    source = 0
    vector = np.array([[1.0, 0.0]], dtype=np.float32)
    # The same *world* vector for both libraries: ordito's basis_x at the source, re-expressed in
    # potpourri3d's frame there.
    vector_pp = [
        [float(basis_x[source] @ basis_x_pp[source]), float(basis_x[source] @ basis_y_pp[source])]
    ]

    transported_wp, _ = od.heat.transport_tangent_vectors(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([source], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        points_to_warp_uv(vector, mesh_wp.device),
    )
    transported_pp = np.asarray(solver_pp.transport_tangent_vectors([source], vector_pp))

    world_wp = _to_world(transported_wp.numpy(), basis_x, basis_y)
    world_pp = _to_world(transported_pp, basis_x_pp, basis_y_pp)
    assert np.allclose(np.linalg.norm(world_wp, axis=1), 1.0, rtol=5e-2, atol=5e-2)
    cosine = (world_wp * world_pp).sum(axis=1) / (
        np.linalg.norm(world_wp, axis=1) * np.linalg.norm(world_pp, axis=1) + 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    # Measured: 0.93 degrees median on ``hemisphere``, 0.03 on ``half_torus``. The *median* is the
    # statistic to use: on the cut locus (and on a 12-vertex icosahedron, most of which is cut
    # locus)
    # the transported direction is genuinely undefined and the two libraries disagree freely there.
    assert np.median(angle) < 2.0


def test_transport_on_a_graded_mesh_matches_a_direct_solve(
    device: str, graded_saddle: GradedSaddle
) -> None:
    """
    Class A at 1e-3 degrees: transported directions against ``scipy``'s solve of the same system.

    The direction of a transported vector is the direction of the diffused vector field at that
    vertex (the magnitude extension only rescales it), so the reference is ``spsolve`` on
    ``vector_heat_operators``' own system. The settle rule used to stop this solve on round-off:
    the median direction was 4.8e-3 degrees off and the worst 180 degrees. Measured agreement
    2.4e-6 degrees after the backward-error fallback.
    """
    _, _, vertices_wp, faces_wp = graded_saddle
    n_vertices = vertices_wp.size
    mesh = od.Trimesh(vertices_wp, faces_wp)
    operators = mesh.vector_heat_operators
    offsets = operators[0].offsets.numpy()[: n_vertices + 1]
    system_sp = sp.bsr_matrix(
        (
            operators[0].values.numpy()[: offsets[-1]],
            operators[0].columns.numpy()[: offsets[-1]],
            offsets,
        ),
        shape=(2 * n_vertices, 2 * n_vertices),
    )
    rhs_np = np.zeros(2 * n_vertices)
    rhs_np[0] = 1.0
    field_np = spla.spsolve(system_sp.tocsc(), rhs_np)
    expected = field_np[0::2] + 1j * field_np[1::2]
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    vectors_wp = wp.array(np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device)
    transported_wp, resolved_wp = od.heat.transport_tangent_vectors(mesh, sources_wp, vectors_wp)
    transported = transported_wp.numpy()
    resolved = resolved_wp.numpy()
    angle = np.degrees(np.abs(np.angle((transported[:, 0] + 1j * transported[:, 1]) / expected)))
    assert resolved.all()
    assert angle.max() < 1e-3


@pytest.mark.parametrize("mesh_name", _HEAT_MESHES_SMALL)
def test_transport_preserves_source_magnitudes(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    magnitude = 2.5
    transported, _ = od.heat.transport_tangent_vectors(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(
            np.array([[0.0, magnitude]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
        ),
    )
    # A single source's magnitude is extended as a constant, so every transported vector has it.
    assert np.allclose(np.linalg.norm(transported.numpy(), axis=1), magnitude, rtol=1e-4, atol=1e-4)


def test_transport_does_not_cross_components(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: potpourri3d returns NaN on this mesh (see the note above).

    ``cave_cube`` is a cube shell around a smaller cube shell: two components. Nothing can be
    transported across the gap, so the cavity's vertices must come back at zero rather than with a
    smeared value. The validity mask is then exactly "reachable, and not the antipodal corner":
    those are the two ways a direction fails to exist, and the vectors alone distinguish neither
    from an ordinary answer -- the cavity's eight vertices and the antipode all read zero on CPU,
    while on CUDA the antipode reads *full length* in a direction that is pure round-off. The mask
    is the same array on both.
    """
    _, mesh_wp = cave_cube
    transported, resolved = od.heat.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, *_unit_transport_source(mesh_wp.device)
    )

    magnitude = np.linalg.norm(transported.numpy(), axis=1)
    labels = od.graph.connected_component_labels_from_edges(
        od.edges.edges_unique(mesh_wp.indices)[0], mesh_wp.points.size
    ).numpy()
    reachable = labels == labels[0]

    # The outer shell's corner diagonally opposite the source is excluded, and not as a tolerance
    # dodge: it is a point of the cut locus with three-fold symmetry about the body diagonal, so the
    # three shortest paths deliver copies of the source vector 120 degrees apart whose sum is
    # exactly zero. Measured on this fixture: exactly 0.0 at diffusion times 1e-3, 1e-2 and 1e-1 (so
    # it is not short-time underflow), unchanged by rotating the source vector (so it is not a bad
    # input direction), and restored in proportion to a symmetry-breaking jitter of the vertices
    # (1.4e-04 at 1e-3, 1.2e-02 at 1e-1).
    #
    # What survives the cancellation is round-off, and the two devices round differently: CPU
    # returns exactly zero and is scaled to zero, CUDA returns 8.7e-09 of the field maximum and is
    # scaled to unit length. Neither is more correct, and no threshold can pick the round-off out --
    # ``half_torus`` resolves genuine directions down to 8.0e-10 of its maximum, below this noise.
    # ``test_transport_cancels_at_a_symmetric_cut_locus_point`` pins the cancellation itself.
    positions_np = mesh_wp.points.numpy()
    antipode = int(np.argmin(np.linalg.norm(positions_np + positions_np[0], axis=1)))
    assert reachable[antipode]
    resolvable = reachable & (np.arange(magnitude.size) != antipode)

    assert np.isfinite(magnitude).all()
    # Seven of the outer shell's eight corners, so the comparison below is not vacuous.
    assert resolvable.sum() == 7
    assert np.allclose(magnitude[resolvable], 1.0, rtol=1e-4, atol=1e-4)
    assert np.allclose(magnitude[~reachable], 0.0, rtol=1e-6, atol=1e-6)
    assert np.array_equal(resolved.numpy(), resolvable)
    assert not ((magnitude == 0.0) & resolved.numpy()).any()


def test_transport_cancels_at_a_symmetric_cut_locus_point(
    cave_cube: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C: the transported direction at a symmetric cut-locus point is a cancellation.

    There is nothing to compare the answer against — the quantity *is* the cancellation — so the
    statistic is a stability one: how far the unit direction at the antipodal corner moves when the
    symmetry causing the cancellation is broken by a vertex jitter, against how far every other
    vertex of the same shell moves under the identical perturbation. It excludes "that corner is
    merely the farthest from the source", which would move it no more than its neighbours move.

    Margins, measured over five jitter seeds on both devices: the antipode moves 1.000 (CPU, where
    the unperturbed answer is the zero vector) to 1.99 (CUDA), against a 0.25 threshold — 4x. Every
    other vertex moves a median 0.0007-0.0034 and at most 0.0089, against a 0.02 threshold — 5.9x.
    The two populations are 112x apart at their closest.
    """
    _, mesh_wp = cave_cube
    sources_wp, vectors_wp = _unit_transport_source(mesh_wp.device)
    positions_np = mesh_wp.points.numpy()
    antipode = int(np.argmin(np.linalg.norm(positions_np + positions_np[0], axis=1)))

    def _directions(points_np: np.ndarray) -> np.ndarray:
        transported, _ = od.heat.transport_tangent_vectors(
            points_to_warp(points_np, mesh_wp.device), mesh_wp.indices, sources_wp, vectors_wp
        )
        norm = np.linalg.norm(transported.numpy(), axis=1, keepdims=True)
        return transported.numpy() / np.where(norm > 0.0, norm, 1.0)

    jitter = np.random.default_rng(20260810).normal(scale=1e-3, size=positions_np.shape)
    moved = np.linalg.norm(_directions(positions_np) - _directions(positions_np + jitter), axis=1)
    labels = od.graph.connected_component_labels_from_edges(
        od.edges.edges_unique(mesh_wp.indices)[0], mesh_wp.points.size
    ).numpy()
    others = (labels == labels[0]) & (np.arange(moved.size) != antipode)

    assert others.sum() == 7
    assert np.median(moved[others]) < 0.02
    assert moved[antipode] > 0.25


@pytest.mark.parametrize("scale", [1e-3, 1e5])
def test_transport_is_invariant_to_mesh_scale(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], scale: float
) -> None:
    """
    Class A: the same surface in different units transports to the same tangent field.

    Both fields the solver divides by — the direction field and the source indicator — carry the
    mesh's scale as ~1/scale^2, so the cutoff that decides "has this vanished?" has to be relative
    to the field. Against an absolute cutoff the failure is a *silent* zero field rather than an
    error: 43 of this fixture's 97 vertices came back at zero magnitude at scale 1e5, and 0 do now.
    """
    _, mesh_wp = hemisphere
    sources_wp, vectors_wp = _unit_transport_source(mesh_wp.device)
    unit, _ = od.heat.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources_wp, vectors_wp
    )
    rescaled, _ = od.heat.transport_tangent_vectors(
        points_to_warp(mesh_wp.points.numpy() * scale, mesh_wp.device),
        mesh_wp.indices,
        sources_wp,
        vectors_wp,
    )

    # Non-vacuous on both sides: every vertex of the unit-scale field is resolved, so a zeroed
    # rescaled field cannot pass the comparison by matching zeros against zeros.
    assert np.allclose(np.linalg.norm(unit.numpy(), axis=1), 1.0, rtol=1e-4, atol=1e-4)
    assert np.allclose(rescaled.numpy(), unit.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("mesh_name", "n_resolved"), [("icosahedron", 11), ("hemisphere", 97)])
def test_transport_validity_mask_flags_the_unresolvable(
    request: pytest.FixtureRequest, mesh_name: str, n_resolved: int
) -> None:
    """
    Class A against resolved counts measured on both devices.

    The counts are the point of the mask: they are identical on CPU and CUDA (11 / 97) where
    the *vectors* are not, because at a cancelling vertex CPU returns the zero vector and CUDA
    returns a full-length one. ``hemisphere`` resolves every vertex, so the boolean assert is
    parametrized over inputs producing both answers rather than only the interesting one.
    ``cave_cube``'s mask (7 resolved) is pinned exactly in
    ``test_transport_does_not_cross_components``.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp, vectors_wp = _unit_transport_source(mesh_wp.device)
    transported, resolved = od.heat.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources_wp, vectors_wp
    )

    assert resolved.numpy().sum() == n_resolved
    # A vanished vector is never called resolved. The converse fails on ``cave_cube``'s antipode.
    assert not ((np.linalg.norm(transported.numpy(), axis=1) == 0.0) & resolved.numpy()).any()


# ---------------------------------------------------------------------------
# log_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_log_map_radius_is_the_geodesic_distance(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    distance = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp).numpy()
    logarithm = od.heat.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()

    # By construction the radius *is* the distance field: this pins the assembly, not the accuracy.
    assert np.allclose(np.linalg.norm(logarithm, axis=1), distance, rtol=1e-4, atol=1e-4)


# Only the better-resolved fixtures: ``potpourri3d.compute_log_map`` raises "factorization failed"
# on
# ``cave_cube`` (zero cotangent weights, as above), and a 12-vertex icosahedron is too coarse for
# either library's log map to mean much.
@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
@pytest.mark.parity("log_map", "potpourri3d")
def test_log_map_matches_potpourri3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (gauge fix by an explicit rotation): both log maps live in the source's tangent plane.

    Unlike the transport test, the gauge here is a single rotation for the whole field -- both maps
    measure angles in *one* tangent plane, the source vertex's -- so the transform recovers that
    angle from the two ``basis_x`` directions and rotates potpourri3d's answer by it. The fixture
    choice is explained in the comment above: neither ``cave_cube`` nor ``icosahedron`` can serve.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solver_pp = _solver_pp(mesh_tm)
    basis_x_pp, basis_y_pp, _ = (np.asarray(basis) for basis in solver_pp.get_tangent_frames())
    basis_x, _, _ = _frames(mesh_wp)

    logarithm_wp = od.heat.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()
    logarithm_pp = np.asarray(solver_pp.compute_log_map(0, "VectorHeat"))

    # Both maps live in the source vertex's tangent plane but measure angles from their own
    # reference
    # direction, so potpourri3d's has to be rotated into ordito's before the two can be compared.
    cosine = float(basis_x[0] @ basis_x_pp[0])
    sine = float(basis_x[0] @ basis_y_pp[0])
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    aligned_pp = logarithm_pp @ rotation.T

    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    error = np.linalg.norm(logarithm_wp - aligned_pp, axis=1)
    # The two constructions differ (this one reads the angle off the distance gradient), so they
    # agree
    # only to a few percent of the mesh scale, improving with resolution: measured 10% of the
    # bounding
    # diagonal on ``hemisphere`` (97 vertices) and 2.8% on ``half_torus`` (544).
    assert np.median(error) < 0.15 * scale


def test_log_map_radius_matches_potpourri3d_far_from_the_sources(icosphere5: Icosphere5) -> None:
    """
    Class B (a norm, which removes the tangent-plane gauge): the radius is the geodesic distance.

    The far-field counterpart of the comparison above, on ``icosphere(5)`` (see
    ``test_heat_geodesic_matches_igl_far_from_the_sources`` for why a ``conftest`` sphere has no
    far field). Taking ``|log|`` on both sides removes the rotation between the two reference
    directions. The exact great-circle distance is asserted alongside as an invariant: measured
    0.021 against potpourri3d and 0.019 against the exact field (potpourri3d's own is 0.040), on
    both devices; the bounds sit 3x above. Before the heat solve ran to full reach the radius was
    2.30 off potpourri3d's and 2.34 off the exact field at worst, which fails both bounds.
    """
    mesh_tm, vertices_wp, faces_wp = icosphere5
    radius_wp = np.linalg.norm(od.heat.log_map(vertices_wp, faces_wp, 0).numpy(), axis=1)
    radius_pp = np.linalg.norm(
        np.asarray(_solver_pp(mesh_tm).compute_log_map(0, "VectorHeat")), axis=1
    )
    exact_np = np.arccos(np.clip(mesh_tm.vertices @ mesh_tm.vertices[0], -1.0, 1.0))
    assert radius_pp.max() > 3.0
    assert np.abs(radius_wp - radius_pp).max() < 0.065
    assert np.abs(radius_wp - exact_np).max() < 0.06


def test_log_map_is_zero_at_its_source(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    logarithm = od.heat.log_map(mesh_wp.points, mesh_wp.indices, 3).numpy()
    assert np.allclose(logarithm[3], 0.0, rtol=1e-6, atol=1e-6)


def test_log_map_is_invariant_to_mesh_scale(icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: the same surface in different units maps to the same angles, scaled.

    ``world_to_tangent_unit``'s vertex-gradient field is an area-weighted *sum* of unit vectors
    (see ``kernels/heat.py::scatter_unit_gradient_to_vertices``), so its magnitude carries the
    mesh's coordinate scale *squared* -- against a fixed absolute floor the failure is not an error
    but a silent collapse of the whole log map to angle zero, because ``log_map_from_angles``
    keeps the (correct) radius and reports an arbitrary angle whenever it reads the field as
    vanished. Every
    one of this fixture's 162 vertices reported angle zero at a scale of 1e-7, and 0 do now (the
    source's own row aside, which is forced to angle zero by construction at every scale).

    Excludes vertex 3, this mesh's genuine cut-locus antipode (confirmed against
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]'s own ``resolved``
    mask): there the transported directions arriving from either side genuinely cancel, and which
    way that cancellation rounds is device-dependent -- on CUDA enough round-off survives it to be
    renormalized into a full-length but arbitrary direction, while on CPU the same point can cancel
    to exactly zero (see ``transport_tangent_vectors``'s own ``Notes``) -- so its angle is not
    expected to agree between the two scales, or to read as zero on one device and not the other,
    independently of this bug.
    """
    _, mesh_wp = icosphere_coarse
    cut_locus = 3  # this mesh's antipode of vertex 0 -- see the docstring above
    unit = od.heat.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()
    rescaled = od.heat.log_map(
        points_to_warp(mesh_wp.points.numpy() * 1e-7, mesh_wp.device), mesh_wp.indices, 0
    ).numpy()

    # Non-vacuous on both sides: the unit-scale angles spread widely away from the source and the
    # cut locus, and no row *other than* the source and the cut locus reads as vanished at either
    # scale -- the source always does, by construction; the cut locus may or may not, by device.
    excluded = (0, cut_locus)
    unit_angles = np.arctan2(unit[:, 1], unit[:, 0])
    assert np.std(np.delete(unit_angles, excluded)) > 0.5
    rescaled_angles = np.arctan2(rescaled[:, 1], rescaled[:, 0])
    assert set(np.flatnonzero(np.abs(unit[:, 1]) < 1e-30).tolist()) <= {0, cut_locus}
    assert set(np.flatnonzero(np.abs(rescaled[:, 1]) < 1e-30).tolist()) <= {0, cut_locus}

    # Compare angles rather than raw coordinates: at 1e-7 the *radius* already carries seven orders
    # of magnitude of scale, so an absolute tolerance on the coordinates themselves would either
    # miss a real angular error (loose) or fail on float32 noise alone (tight). 0.02 rad (~1.1
    # degrees) is well above the ~7e-3 rad of float32 noise measured at this scale and far below
    # the near-pi error the collapse produces.
    angle_diff = np.abs(np.angle(np.exp(1j * (rescaled_angles - unit_angles))))
    assert np.delete(angle_diff, excluded).max() < 0.02


# ---------------------------------------------------------------------------
# vector_heat_operators (the amortized path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _HEAT_MESHES)
def test_reused_operators_give_the_same_transport(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: a `Trimesh`, solved twice, changes no answer.

    Not a parity assert: the ``vertices, faces`` calls carry the potpourri3d oracle. Covers the
    three consumers of the vector bundle -- transport, the log map, and the signed heat method,
    which assembles three matrices -- each called twice on one mesh, so the second call goes to the
    factorizations the first one's solves earned.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, _, curve_wp = _one_ring_cycle(mesh_tm, mesh_wp)
    sources, vectors = _unit_transport_source(mesh_wp.device)
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)

    # Reusing the mesh must be an optimization and nothing else: the same assembly, so the same
    # matrices, so the same answer. Not *bit* for bit, though — conjugate gradient reduces with
    # atomics, and the factorization answers to its own tolerance. Measured: transport agrees to
    # 2e-15, and the log map to 4e-7 absolute, which is float32 epsilon on its own output. Off the
    # cut locus only, where the direction is round-off and the two solves' round-off differs (the
    # icosahedron's antipode: 0.06 apart; see ``transport_tangent_vectors``' Notes); the log map's
    # radius is compared everywhere. The signed field integrates the normalized diffused field,
    # whose direction at that antipode is round-off, so on the icosahedron alone the two solves'
    # signed fields differ there by 3.9e-3 of their span.
    resolved = od.heat.transport_tangent_vectors(mesh_wp.points, mesh_wp.indices, sources, vectors)[
        1
    ].numpy()
    assert resolved.any()
    signed_tolerance = 1e-2 if mesh_name == "icosahedron" else 1e-5
    points, indices = mesh_wp.points, mesh_wp.indices
    everywhere = np.ones_like(resolved)
    # Each row: the field from ``vertices, faces``, the same field from the mesh, where to compare.
    calls: list[tuple[Callable[[], np.ndarray], Callable[[], np.ndarray], np.ndarray, float]] = [
        (
            lambda: od.heat.transport_tangent_vectors(points, indices, sources, vectors)[0].numpy(),
            lambda: od.heat.transport_tangent_vectors(mesh, sources, vectors)[0].numpy(),
            resolved,
            1e-5,
        ),
        (
            lambda: od.heat.log_map(points, indices, 0).numpy(),
            lambda: od.heat.log_map(mesh, 0).numpy(),
            resolved,
            1e-5,
        ),
        (
            lambda: np.linalg.norm(od.heat.log_map(points, indices, 0).numpy(), axis=1),
            lambda: np.linalg.norm(od.heat.log_map(mesh, 0).numpy(), axis=1),
            everywhere,
            1e-5,
        ),
        (
            lambda: od.heat.heat_signed_distance(points, indices, curve_wp).numpy(),
            lambda: od.heat.heat_signed_distance(mesh, curve_wp).numpy(),
            everywhere,
            signed_tolerance,
        ),
    ]
    for on_vertices, on_mesh, rows, tolerance in calls:
        fresh = on_vertices()
        for _ in range(2):
            reused = on_mesh()
            # Compared against the *field's* magnitude rather than per element: a component that is
            # near-zero in a field of size one carries no information about the solve's agreement.
            span = float(np.abs(fresh).max())
            assert np.allclose(fresh[rows], reused[rows], rtol=0.0, atol=tolerance * span)
    assert mesh.heat_solver().nbytes > 0


def test_trimesh_keeps_one_heat_solver_per_diffusion_time(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: a `Trimesh` solves at the ``t`` it is asked for, with a solver per ``t``.

    Not a parity assert. ``t`` lives in the assembled system, so a solver kept for one ``t`` must
    never answer for another: ``t`` differs by six orders of magnitude between the two calls, so a
    wrong key would not be subtle.
    """
    _, mesh_wp = icosahedron
    sources, vectors = _unit_transport_source(mesh_wp.device)
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    od.heat.transport_tangent_vectors(mesh, sources, vectors, t=1e-6)
    on_mesh, _ = od.heat.transport_tangent_vectors(mesh, sources, vectors, t=1.0)
    direct, _ = od.heat.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources, vectors, t=1.0
    )
    span = float(np.abs(direct.numpy()).max())
    assert np.allclose(on_mesh.numpy(), direct.numpy(), rtol=0.0, atol=1e-5 * span)
    assert mesh.heat_solver(1.0) is mesh.heat_solver(1.0)
    assert mesh.heat_solver(1.0) is not mesh.heat_solver(1e-6)


# ---------------------------------------------------------------------------
# diffuse_tangent_field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("on_trimesh", [False, True], ids=["system", "trimesh"])
@pytest.mark.parametrize("mesh_name", _HEAT_MESHES_SMALL)
def test_diffuse_tangent_field_solves_its_own_system(
    request: pytest.FixtureRequest, mesh_name: str, on_trimesh: bool
) -> None:
    """
    Class A: the result satisfies ``(M + t L_connection) X = source`` to the solver's tolerance.

    This entry point is public because the *source* is where the vector-valued methods differ while
    the solve is shared, so what has to be pinned is the equation rather than any particular field.
    Applying the operator back to the answer is the direct check, and it is independent of the CG
    path that produced it. Measured residual 1.5e-09 against a unit source. Both forms: the
    system, and a `Trimesh` whose own vector heat system the solve must be.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    vector_system = mesh.vector_heat_operators[0]

    source_np = np.zeros((n_vertices, 2))
    source_np[0] = [1.0, 0.0]
    source_np[n_vertices // 3] = [0.0, -1.0]
    source_wp = wp.array(
        np.ascontiguousarray(source_np), dtype=wp.vec2d, device=mesh_wp.points.device
    )

    diffused_wp = (
        od.heat.diffuse_tangent_field(mesh, source_wp)
        if on_trimesh
        else od.heat.diffuse_tangent_field(vector_system, source_wp)
    )

    # Non-trivial: diffusion reaches every vertex, so this is not solving for zero.
    assert np.all(np.linalg.norm(diffused_wp.numpy(), axis=1) > 0.0)
    residual_wp = wp.zeros(n_vertices, dtype=wp.vec2d, device=mesh_wp.points.device)
    odt.bsr_mv(vector_system, diffused_wp, residual_wp, alpha=1.0, beta=0.0)
    assert np.abs(residual_wp.numpy() - source_np).max() < 1e-7


# ---------------------------------------------------------------------------
# HeatSolver: what a Trimesh keeps between calls
# ---------------------------------------------------------------------------


def _count_factorizations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the sparse Cholesky factorizations built from here on, by every caller."""
    built: list[int] = []
    factor = od.linalg.sparse_cholesky

    def counted(*args: Any, **kwargs: Any) -> od.cholesky.SparseCholesky:
        result = factor(*args, **kwargs)
        built.append(result.n)
        return result

    monkeypatch.setattr(od.linalg, "sparse_cholesky", counted)
    return built


def test_trimesh_factors_on_the_second_solve_and_reuses_it(
    icosphere5: Icosphere5, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: a solver-state invariant.

    On a well-conditioned mesh the first call iterates and builds nothing (it costs what a one-shot
    call costs); the second factors each system it solves -- the heat system and the Poisson
    system, two -- and the third builds none and reuses both. The ``vertices, faces`` form never
    factors here, however often it is called. The answers carry their oracle in
    ``test_heat_geodesic_matches_igl_far_from_the_sources``; here they only agree with each other
    to the solves' tolerances.
    """
    _, vertices_wp, faces_wp = icosphere5
    built = _count_factorizations(monkeypatch)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=vertices_wp.device)
    for _ in range(2):
        od.heat.heat_geodesic(vertices_wp, faces_wp, sources_wp)
    assert built == []

    mesh = od.Trimesh(vertices_wp, faces_wp)
    distances = []
    for expected in ([], [vertices_wp.size] * 2, [vertices_wp.size] * 2):
        distances.append(od.heat.heat_geodesic(mesh, sources_wp).numpy())
        assert built == expected
    span = np.ptp(distances[0])
    assert span > 1.0
    assert np.abs(distances[1] - distances[0]).max() < 1e-6 * span
    assert np.abs(distances[2] - distances[1]).max() < 1e-12 * span


@pytest.fixture
def saddle_graded(device: str) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Return ``benchmarks``' ``saddle_graded`` (17 689 vertices) on ``device``."""
    vertices_np, faces_np = BUILDERS["saddle_graded"]()
    return numpy_to_warp(vertices_np, np.ravel(faces_np), device)


def test_graded_heat_diffusion_falls_back_to_a_factorization_that_matches_scipy(
    saddle_graded: tuple[wp.array[wp.vec3], wp.array[wp.int32]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A at 1e-10 of the range: the diffused heat against ``scipy``'s solve of the same system.

    On ``saddle_graded`` the settled conjugate-gradient iterate fails its backward-error check, so
    the first solve on a `Trimesh` falls back to a factorization at once -- one build -- which the
    mesh keeps: the second solve builds nothing. The reference is ``spsolve`` on the solver's own
    heat system, so the comparison is of the linear solve alone.
    """
    vertices_wp, faces_wp = saddle_graded
    mesh = od.Trimesh(vertices_wp, faces_wp)
    built = _count_factorizations(monkeypatch)
    solver = mesh.heat_solver()
    n_vertices = vertices_wp.size
    rhs_np = np.zeros(n_vertices)
    rhs_np[0] = 1.0
    heat_wp = wp.zeros(n_vertices, dtype=wp.float64, device=vertices_wp.device)
    solver.diffuse(wp.array(rhs_np, dtype=wp.float64, device=vertices_wp.device), heat_wp)
    assert built == [n_vertices]
    assert solver.nbytes > 0

    system = solver.operators[0]
    offsets = system.offsets.numpy()[: n_vertices + 1]
    system_sp = sp.csr_matrix(
        (system.values.numpy()[: offsets[-1]], system.columns.numpy()[: offsets[-1]], offsets),
        shape=(n_vertices, n_vertices),
    )
    heat_np = spla.spsolve(system_sp.tocsc(), rhs_np)
    assert np.ptp(heat_np) > 0.0
    assert np.abs(heat_wp.numpy() - heat_np).max() < 1e-10 * np.ptp(heat_np)

    heat_wp.zero_()
    solver.diffuse(wp.array(rhs_np, dtype=wp.float64, device=vertices_wp.device), heat_wp)
    assert built == [n_vertices]
    assert np.abs(heat_wp.numpy() - heat_np).max() < 1e-10 * np.ptp(heat_np)


@pytest.mark.parametrize("on_trimesh", [False, True], ids=["vertices_faces", "trimesh"])
def test_a_discarded_factorization_is_logged(
    saddle_graded: tuple[wp.array[wp.vec3], wp.array[wp.int32]],
    caplog: pytest.LogCaptureFixture,
    on_trimesh: bool,
) -> None:
    """
    Not a library comparison: the ``INFO`` record says exactly when a factorization is dropped.

    On the graded saddle the heat solve falls back to a factorization. The ``vertices, faces``
    call then drops it and says so; the `Trimesh` call keeps it and says nothing.
    """
    vertices_wp, faces_wp = saddle_graded
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=vertices_wp.device)
    mesh = od.Trimesh(vertices_wp, faces_wp)
    with caplog.at_level(logging.INFO, logger="ordito.heat"):
        if on_trimesh:
            od.heat.heat_geodesic(mesh, sources_wp)
        else:
            od.heat.heat_geodesic(vertices_wp, faces_wp, sources_wp)
    discarded = [r for r in caplog.records if r.levelno == logging.INFO and "discard" in r.message]
    assert len(discarded) == (0 if on_trimesh else 1)
    assert (mesh.heat_solver().nbytes > 0) == on_trimesh


def test_nothing_is_logged_where_nothing_was_factored(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], caplog: pytest.LogCaptureFixture
) -> None:
    """
    Not a library comparison: a one-shot call on a well-conditioned mesh builds nothing to drop.

    So the ``vertices, faces`` form logs nothing at ``INFO``.
    """
    _, mesh_wp = icosahedron
    sources, vectors = _unit_transport_source(mesh_wp.device)
    with caplog.at_level(logging.INFO, logger="ordito.heat"):
        od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources)
        od.heat.transport_tangent_vectors(mesh_wp.points, mesh_wp.indices, sources, vectors)
        od.heat.log_map(mesh_wp.points, mesh_wp.indices, 0)
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []


def test_release_factorizations_frees_them(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: ``Trimesh.release_factorizations`` drops what the solvers kept.

    After two calls the mesh holds two factorizations; released, it holds none, and the next call
    starts counting afresh (it iterates, building nothing).
    """
    _, mesh_wp = icosahedron
    sources, _ = _unit_transport_source(mesh_wp.device)
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    built = _count_factorizations(monkeypatch)
    for _ in range(2):
        od.heat.heat_geodesic(mesh, sources)
    solver = mesh.heat_solver()
    assert len(built) == 2
    assert solver.nbytes > 0
    mesh.release_factorizations()
    assert solver.nbytes == 0
    assert mesh.heat_solver() is not solver
    assert mesh.heat_solver().nbytes == 0
    od.heat.heat_geodesic(mesh, sources)
    assert len(built) == 2


# ---------------------------------------------------------------------------
# tangent_to_world and edge cases
# ---------------------------------------------------------------------------


def test_tangent_to_world_reproduces_the_frames(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    basis_x_wp, basis_y_wp, _ = od.tangent_space.vertex_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )
    n_vertices = mesh_wp.points.size
    tangent = wp.array(
        np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n_vertices, 1)),
        dtype=wp.vec2,
        device=mesh_wp.device,
    )
    world = od.heat.tangent_to_world(tangent, basis_x_wp, basis_y_wp)
    assert np.allclose(world.numpy(), basis_y_wp.numpy(), rtol=1e-6, atol=1e-6)


# A returned field and the number of values it must hold, every one of them zero.
_Sized = tuple["wp.array[Any]", int]


def _empty_mesh(device: str) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    return warp_empty(0, wp.vec3, device), warp_empty(0, wp.int32, device)


def _icosahedron_wp(device: str) -> wp.Mesh:
    return trimesh_to_warp(tm.creation.icosahedron(), device)


def _no_faces_distance(device: str) -> list[_Sized]:
    vertices_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    sources_wp = wp.array([0], dtype=wp.int32, device=device)
    distance_wp = od.heat.heat_geodesic(vertices_wp, warp_empty(0, wp.int32, device), sources_wp)
    return [(distance_wp, 4)]


def _no_sources_distance(device: str) -> list[_Sized]:
    mesh_wp = _icosahedron_wp(device)
    distance_wp = od.heat.heat_geodesic(
        mesh_wp.points, mesh_wp.indices, warp_empty(0, wp.int32, device)
    )
    return [(distance_wp, mesh_wp.points.size)]


def _empty_signed_distance(device: str) -> list[_Sized]:
    vertices_wp, faces_wp = _empty_mesh(device)
    curve_wp = warp_empty(0, wp.int32, device)
    return [(od.heat.heat_signed_distance(vertices_wp, faces_wp, curve_wp), 0)]


def _empty_vector_heat(device: str) -> list[_Sized]:
    vertices_wp, faces_wp = _empty_mesh(device)
    empty_int = warp_empty(0, wp.int32, device)
    extended_wp = od.heat.extend_scalar(
        vertices_wp, faces_wp, empty_int, warp_empty(0, wp.float64, device)
    )
    transported_wp, resolved_wp = od.heat.transport_tangent_vectors(
        vertices_wp, faces_wp, empty_int, warp_empty(0, wp.vec2, device)
    )
    log_wp = od.heat.log_map(vertices_wp, faces_wp, 0)
    return [(extended_wp, 0), (transported_wp, 0), (resolved_wp, 0), (log_wp, 0)]


def _empty_tangent_source(device: str) -> list[_Sized]:
    mesh_wp = _icosahedron_wp(device)
    vector_system, _scalar, _frames, _preconditioner = od.heat.vector_heat_operators(
        mesh_wp.points, mesh_wp.indices
    )
    diffused_wp = od.heat.diffuse_tangent_field(vector_system, warp_empty(0, wp.vec2d, device))
    return [(diffused_wp, 0)]


_EMPTY_CASES: list[tuple[str, Callable[[str], list[_Sized]]]] = [
    ("heat_geodesic_no_faces", _no_faces_distance),
    ("heat_geodesic_no_sources", _no_sources_distance),
    ("heat_signed_distance_empty_mesh", _empty_signed_distance),
    ("vector_heat_empty_mesh", _empty_vector_heat),
    ("diffuse_tangent_field_empty_source", _empty_tangent_source),
]


@pytest.mark.parametrize("case", [pytest.param(case, id=case_id) for case_id, case in _EMPTY_CASES])
def test_empty_inputs_return_zero_fields(device: str, case: Callable[[str], list[_Sized]]) -> None:
    """
    Not a library comparison: an empty mesh, face set or source has an all-zero answer.

    Each entry point returns one value per vertex -- zero where there is no heat to diffuse, and
    nothing on an empty mesh -- and ``diffuse_tangent_field`` returns an empty field for an empty
    source without entering the solver.
    """
    for result_wp, n_values in case(device):
        assert result_wp.shape == (n_values,)
        assert not result_wp.numpy().any()
