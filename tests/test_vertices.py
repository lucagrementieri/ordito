"""Regression tests for ``triwarp.vertices`` against Trimesh (CPU reference)."""

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.conversions import (
    faces_igl,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
)


@pytest.mark.parity("vertex_normals", "open3d", "pymeshlab")
@pytest.mark.parity("mean_vertex_normals", "pymeshlab")
def test_vertex_normal_weightings_match_open3d_and_pymeshlab(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    The two unweighted-and-area weightings against the libraries that implement the same ones.

    Class A for all three, and they agree to **3.5e-7**, tighter than the 1e-5 tolerance, because
    ``compute_vertex_normals`` and ``compute_normal_per_vertex(weightmode="By Area")`` are the
    same scheme triwarp implements, and ``"Simple Average"`` is the unweighted one.

    **trimesh is deliberately absent**, and that is the finding worth recording:
    ``Trimesh.vertex_normals`` is *angle*-weighted, not area-weighted. It matches
    ``vertex_normals`` at ``weighting="angle"`` to 4.3e-7 (see ``test_vertex_normals_angle``)
    and differs from the area-weighted answer by up to **0.072** on this fixture. The
    ``vertex_normals`` benchmark group timed it as though it were the same quantity; it is now
    exempted there with a redirect to open3d.
    """
    mesh_tm, mesh_wp = half_torus
    n_vertices = mesh_wp.points.size

    area_wp = tw.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)

    mesh_o3d = trimesh_to_open3d(mesh_tm)
    mesh_o3d.compute_vertex_normals()
    assert np.allclose(area_wp.numpy(), np.asarray(mesh_o3d.vertex_normals), rtol=1e-5, atol=1e-5)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_normal_per_vertex(weightmode="By Area")
    normals_pml = meshset_pml.current_mesh().vertex_normal_matrix()
    assert np.allclose(area_wp.numpy(), normals_pml, rtol=1e-5, atol=1e-5)

    # The unweighted scheme, from the same filter under a different weightmode.
    face_normals_wp, _areas_wp = tw.triangles.face_normals_and_areas(
        mesh_wp.points, mesh_wp.indices
    )
    mean_wp = tw.vertices.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)
    mean_meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    mean_meshset_pml.compute_normal_per_vertex(weightmode="Simple Average")
    mean_pml = mean_meshset_pml.current_mesh().vertex_normal_matrix()
    assert np.allclose(mean_wp.numpy(), mean_pml, rtol=1e-5, atol=1e-5)

    # The two weightings are genuinely different, so neither assert above is weightless.
    assert not np.allclose(area_wp.numpy(), mean_wp.numpy(), atol=1e-3)


@pytest.mark.parity("mean_vertex_normals", "pyvista")
def test_mean_vertex_normals_match_pyvista(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A, and the row names ``mean_vertex_normals`` for a measured reason.

    ``compute_normals``' point ``Normals`` is the *unweighted* sum of incident face normals, so it
    is this function and not one of ``vertex_normals``' three weightings: measured 5.1e-07 here
    against 4.5e-03 (area), 6.8e-03 (angle) on the same mesh. A row pointed at
    ``vertex_normals`` would therefore fail at ``1e-5`` rather than merely be loose, and the last
    assert keeps that separation live.

    The flags are the ones ``tests/test_triangles.py`` explains: no re-winding, no vertex splitting.
    The array comes back **float32**, which is the tolerance floor on VTK's side rather than
    triwarp's.
    """
    mesh_tm, mesh_wp = half_torus
    n_vertices = mesh_wp.points.size
    face_normals_wp, _areas_wp = tw.triangles.face_normals_and_areas(
        mesh_wp.points, mesh_wp.indices
    )
    mean_wp = tw.vertices.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)

    normals_pv = trimesh_to_pyvista(mesh_tm).compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    normals_pv_np = np.asarray(normals_pv.point_data["Normals"])
    assert np.allclose(mean_wp.numpy(), normals_pv_np, rtol=1e-5, atol=1e-5)

    area_wp = tw.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    assert not np.allclose(area_wp.numpy(), normals_pv_np, atol=1e-4)


def test_mean_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the unweighted 1-ring mean against ``trimesh.geometry.mean_vertex_normals``.

    Fed trimesh's own face normals, so the comparison isolates the accumulation and
    normalization from [`triangles.face_normals_and_areas`], which has its own oracle.
    """
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    vertex_normals_tm = tm.geometry.mean_vertex_normals(n_vertices, mesh_tm.faces, face_normals_tm)

    face_normals_wp = points_to_warp(face_normals_tm, mesh_wp.device)
    vertex_normals_wp = tw.vertices.mean_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the general weighted form against trimesh's, with both inputs supplied.

    trimesh's ``weighted_vertex_normals`` takes the face normals *and* the corner angles, so
    passing both in pins the weighting rule alone -- which is where the libraries in section 6
    differ most.
    """
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    face_angles_tm = mesh_tm.face_angles
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, face_normals_tm, face_angles_tm
    )

    face_normals_wp = points_to_warp(face_normals_tm, mesh_wp.device)
    face_weights_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.weighted_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp, face_weights_wp
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_vertex_normals_area(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: area weighting against ``igl.per_vertex_normals``' area-weighted mode.

    igl is the reference here rather than trimesh, because trimesh has no area-weighted mode --
    the angle-weighted one is [`test_vertex_normals_angle`].
    """
    mesh_tm, mesh_wp = half_torus

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    vertex_normals_igl = igl.per_vertex_normals(
        vertices_np, faces_np, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
    )

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    vertex_normals_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices)
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("vertex_normals", "pytorch3d")
def test_vertex_normals_match_pytorch3d(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: ``Meshes.verts_normals_packed`` is triwarp's **area**-weighted vertex normal.

    pytorch3d sums the *unnormalized* face cross products into each incident vertex and normalizes
    once at the end, which is area weighting by construction -- so this pins the default
    ``weighting="area"`` and nothing else. Measured 1.19e-07 against it, and **1.14e-02** against
    ``weighting="angle"``, five orders apart: the second assert is what makes the first a claim
    about the weighting convention rather than about vertex normals in general.

    A closed, fully-referenced fixture on purpose. pytorch3d does not drop unreferenced vertices
    and leaves their normals at the ``torch.zeros`` initial value rather than ``NaN``, so on a mesh
    with spares this would be comparing zeros with whatever triwarp writes.
    """
    mesh_tm, mesh_wp = icosphere
    device = str(mesh_wp.points.device)
    normals_t = trimesh_to_pytorch3d(mesh_tm, device).verts_normals_packed()
    assert normals_t is not None
    normals_p3d = normals_t.cpu().numpy()
    normals_wp = tw.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    angle_wp = tw.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices, weighting="angle")

    assert normals_p3d.shape == mesh_tm.vertices.shape
    assert np.allclose(normals_wp.numpy(), normals_p3d, rtol=1e-5, atol=1e-6)
    assert not np.allclose(angle_wp.numpy(), normals_p3d, rtol=1e-5, atol=1e-6)


@pytest.mark.parity("vertex_normals", "meshlib")
def test_vertex_normal_weightings_match_meshlib(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A, and the reason to have it: MeshLib pins *which* weighting each name means.

    The pairing is not guessable from the names and was found by measuring all six combinations.
    ``computePerVertNormals`` is the **area**-weighted sum and ``computePerVertPseudoNormals`` is
    the **angle**-weighted one; each matches its triwarp partner to **1.19e-07** and sits
    **6.8e-03** from the other's, which is four orders of magnitude of separation and far more than
    any tolerance argument. ``mean_vertex_normals`` matches neither (4.5e-03 from the nearer), so
    it keeps its own oracles and is asserted here only to be *different* -- without that, a
    regression collapsing all three weightings to one would still pass the two positive asserts.

    No other reference in the suite distinguishes these: igl exposes an area mode and trimesh an
    angle-weighted one, but neither can say what the other's name would mean.
    """
    mesh_tm, mesh_wp = half_torus
    n_vertices = mesh_tm.vertices.shape[0]
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    area_ml = mn.toNumpyArray(mm.computePerVertNormals(mesh_ml))
    angle_ml = mn.toNumpyArray(mm.computePerVertPseudoNormals(mesh_ml))

    area_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices)
    angle_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices, weighting="angle")
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(vertices_wp, mesh_wp.indices)
    mean_wp = tw.vertices.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)

    assert area_ml.shape == (n_vertices, 3)  # non-vacuity: the converter kept every vertex
    assert np.allclose(area_wp.numpy(), area_ml, rtol=1e-5, atol=1e-5)
    assert np.allclose(angle_wp.numpy(), angle_ml, rtol=1e-5, atol=1e-5)

    # The separation, without which the two asserts above would not pin a weighting at all.
    assert not np.allclose(area_wp.numpy(), angle_ml, atol=1e-4)
    assert not np.allclose(angle_wp.numpy(), area_ml, atol=1e-4)
    assert not np.allclose(mean_wp.numpy(), area_ml, atol=1e-4)


def test_vertex_normals_area_precomputed(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)

    face_normals_wp, face_areas_wp = tw.triangles.face_normals_and_areas(
        vertices_wp, mesh_wp.indices
    )
    vertex_normals_precomputed_wp = tw.vertices.vertex_normals(
        vertices_wp, mesh_wp.indices, face_normals=face_normals_wp, face_weights=face_areas_wp
    )
    vertex_normals_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices)
    assert np.allclose(
        vertex_normals_precomputed_wp.numpy(), vertex_normals_wp.numpy(), rtol=1e-5, atol=1e-5
    )


def test_vertex_normals_angle(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: angle weighting against trimesh's, which is the same rule under a different name.

    Completes the three weightings the wrapper exposes; each is a different accumulation rather
    than a scaling of one, so each needs its own comparison.
    """
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals, mesh_tm.face_angles
    )

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    vertex_normals_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices, weighting="angle")
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_vertex_normals_angle_precomputed(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)

    face_normals_wp, _ = tw.triangles.face_normals_and_areas(vertices_wp, mesh_wp.indices)
    face_angles_wp = tw.triangles.face_angles(vertices_wp, mesh_wp.indices)
    vertex_normals_precomputed_wp = tw.vertices.vertex_normals(
        vertices_wp,
        mesh_wp.indices,
        weighting="angle",
        face_normals=face_normals_wp,
        face_weights=face_angles_wp,
    )
    vertex_normals_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices, weighting="angle")
    assert np.allclose(
        vertex_normals_precomputed_wp.numpy(), vertex_normals_wp.numpy(), rtol=1e-5, atol=1e-5
    )


def _compute_max_vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Nelson Max MWSELR vertex normals: (e1 x e2) / (||e1||^2 * ||e2||^2) per corner."""
    corner = np.arange(3)
    i0 = faces
    i1 = faces[:, np.roll(corner, -1)]
    i2 = faces[:, np.roll(corner, -2)]
    e1 = vertices[i1] - vertices[i0]
    e2 = vertices[i2] - vertices[i0]
    cross = np.cross(e1, e2)
    len_sq = np.sum(e1**2, axis=-1) * np.sum(e2**2, axis=-1)
    contrib = cross / np.where(len_sq[..., None] == 0, 1.0, len_sq[..., None])
    vertex_normals_np = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vertex_normals_np, i0.ravel(), contrib.reshape(-1, 3))
    norms = np.linalg.norm(vertex_normals_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vertex_normals_np / norms


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])
def test_vertex_normals_mwselr(scale: float, half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A against a NumPy transcription of Nelson Max's MWSELR weight.

    **Parametrized over the mesh scale, which is where the weight used to fail silently.** The
    weight divides by ``||e1||^2 * ||e2||^2``, a length^4 quantity, and the kernel guarded that
    denominator against the absolute ``TOLERANCE_ZERO`` (1e-12) rather than against zero -- so
    every corner of a mesh with edges shorter than ~1e-3 was read as degenerate and weighted zero.
    Measured before the fix at ``scale=1e-3``: all 162 vertex normals of a scaled ``icosphere(2)``
    came back as the zero vector, while ``"area"`` and ``"angle"`` returned correct unit normals
    from the same buffers -- so nothing but this parametrization separates the defect from the
    documented zero-row contract (an unreferenced vertex, or a fan whose contributions cancel).
    The oracle keys on ``len_sq == 0`` for the same reason, so it is scale-free by construction and
    the two agree at every scale. ``1e3`` covers the other direction, where the denominator grows
    rather than shrinks.

    Both entry points are checked at every scale: the derived-normals path and the
    supplied-``face_normals`` path multiply in the cross-product magnitude at different points, so
    a scale that broke one need not break the other.
    """
    mesh_tm, mesh_wp = half_torus

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64) * scale
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    vertex_normals_np = _compute_max_vertex_normals_np(vertices_np, faces_np)
    # Non-vacuity: the oracle itself must be unit normals, not the zero rows the defect produced.
    assert np.allclose(np.linalg.norm(vertex_normals_np, axis=1), 1.0)

    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    vertex_normals_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices, weighting="mwselr")
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_np, rtol=1e-5, atol=1e-5)

    # Face normals are unit vectors, so they are the same buffer at every scale.
    face_normals_wp = points_to_warp(mesh_tm.face_normals, mesh_wp.device)
    vertex_normals_explicit_wp = tw.vertices.vertex_normals(
        vertices_wp, mesh_wp.indices, weighting="mwselr", face_normals=face_normals_wp
    )
    assert np.allclose(vertex_normals_explicit_wp.numpy(), vertex_normals_np, rtol=1e-5, atol=1e-5)


def test_vertex_normals_are_reproducible(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Triwarp against triwarp: the identical call, eight times, must return the identical buffer.

    No reference library can carry this -- it is a claim about *this* implementation's summation
    order, not about the quantity -- so the oracle for the values themselves is
    ``test_vertex_normals_area``, and this only pins repeatability on top of it. A float atomic's
    order is the scheduler's and float addition is not associative, so the ``float32`` accumulator
    this used to carry moved by one ULP (1.19e-07) between runs on CUDA while the CPU device was
    exact. The accumulator is ``float64`` now, which drops the disagreement between two orderings
    below what the ``float32`` answer can represent.

    Asserted as exact equality deliberately: the old behaviour fails it, a tolerance of 1e-6 would
    not, and the whole point of the change is that there is nothing left to tolerate. The
    downstream consumer this was found through is
    ``test_principal_curvature_is_reproducible``.
    """
    _, mesh_wp = half_torus

    runs = [
        tw.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices).numpy().copy() for _ in range(8)
    ]

    # Non-vacuity: an all-zero or constant buffer would compare equal to itself for free.
    assert np.allclose(np.linalg.norm(runs[0], axis=1), 1.0)
    assert np.ptp(runs[0], axis=0).max() > 1.0
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)


@pytest.mark.parametrize("scale", [3e-6, 1e-9])
def test_vertex_normals_survive_a_small_mesh_scale(
    icosphere: tuple[tm.Trimesh, wp.Mesh], scale: float
) -> None:
    """
    Class A: a normal is a direction, not a length, so shrinking the mesh must not move it.

    Against ``igl.per_vertex_normals``' area mode, for the reason ``test_vertex_normals_area``
    gives -- trimesh has no area-weighted mode -- and taken at unit scale, which is where the rest
    of this file pins it. ``|cross|`` scales as ``h^2``, so an absolute floor on it inside
    ``kernels.triangles.face_normals_and_area`` put every face of a mesh at ``h <= 3e-6`` below the
    floor and returned the raw cross product where a unit normal was promised; area-weighting then
    squared that and ``wp.normalize`` saw a ``float32`` ``length_sq`` underflowed to zero.
    Measured before the fix: **every** vertex normal came back exactly zero at both scales here,
    with nothing raised.
    """
    mesh_tm, mesh_wp = icosphere
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    normals_igl = igl.per_vertex_normals(
        vertices_np,
        np.asarray(mesh_tm.faces, dtype=np.int32),
        igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
    )
    vertices_wp = points_to_warp(vertices_np * scale, mesh_wp.device)

    normals_wp = tw.vertices.vertex_normals(vertices_wp, mesh_wp.indices)

    # Non-vacuity: the reference is unit everywhere, so nothing here is comparing two zeros.
    assert np.allclose(np.linalg.norm(normals_igl, axis=1), 1.0)
    assert np.allclose(normals_wp.numpy(), normals_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["half_torus", "icosahedron"])
@pytest.mark.parity("vertex_defects", "trimesh", "igl", "meshlib")
def test_vertex_defects(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A on all three references, on an open fixture and a closed one.

    ``igl.gaussian_curvature`` and MeshLib's ``mn.getNumpyGaussianCurvature`` are both the
    *pointwise* angle defect ``2π - Σθ`` -- the same quantity ``tm.curvature.vertex_defects``
    returns and emphatically **not** the ball-integrated Cohen-Steiner/Morvan measure
    ``curvature.discrete_gaussian_curvature`` computes, which is exempted from parity against
    MeshLab for exactly that reason. Sharing a name with a different measure is the whole hazard
    here, so the comparison is worth having three times over -- and MeshLib is the one to be
    careful with, because it is the reference whose *name* says curvature while its value is the
    defect.

    MeshLib's batched ``mn.getNumpyGaussianCurvature`` is used rather than the per-vertex
    ``mm.discreteGaussianCurvature``; the two are bit-identical (measured 0.0) and the batched form
    is 49-67x faster, which is section 6's rule about its per-vertex entry points.

    Both fixtures are needed because the interesting disagreement would be at the **boundary**: a
    reference could reasonably use ``π - Σθ`` there. Measured, none of the three does --
    ``half_torus``'s 56 boundary vertices agree element-wise with the closed ``icosahedron``'s
    interior ones -- so the assert is a plain ``allclose`` over every vertex.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    n_vertices = mesh_tm.vertices.shape[0]
    vertex_defects_tm = tm.curvature.vertex_defects(mesh_tm)
    vertex_defects_igl = igl.gaussian_curvature(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
    ).ravel()

    face_angles_wp = wp.array(mesh_tm.face_angles, dtype=wp.float32, device=mesh_wp.device)
    vertex_defects_wp = tw.vertices.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp)

    vertex_defects_ml = mn.getNumpyGaussianCurvature(trimesh_to_meshlib(mesh_tm))

    # Non-vacuity: an implementation returning zeros passes any allclose against another one, and
    # on a nearly-flat patch that is what the true answer looks like. Neither the spread nor the
    # minimum is the check -- an icosahedron is regular so all 12 read pi/3, and half_torus has
    # genuinely near-flat vertices at 4e-05 -- so the claim is that the answer is not all zero.
    assert np.abs(vertex_defects_tm).max() > 1e-2
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_ml, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["half_torus", "icosahedron", "hemisphere"])
@pytest.mark.parity("vertex_defects", "pyvista")
def test_vertex_defects_against_pyvista_gaussian_curvature(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: VTK's ``curvature('gaussian')`` is this defect divided by the lumped area.

    ``vtkCurvatures`` returns a *density* -- the angle defect over the barycentric lumped area,
    ``sum of incident face areas / 3`` -- so the named transform is a multiplication by that area,
    which is read off ``compute_cell_sizes`` on the same mesh rather than recomputed. Measured
    element-wise agreement 2.7e-07 / 1.2e-06 / 5.1e-07 on the three fixtures.

    ``atol`` carries the comparison rather than ``rtol``: a flat vertex has zero defect, so a
    relative tolerance is meaningless there. The residual is triwarp's ``float32`` vertex buffer and
    not the reference's -- pyvista's curvature comes back float64, and the same comparison against
    vedo's float32 points measures the same 5e-05 on a larger mesh.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_tm.vertices.shape[0]

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    gaussian_pv = np.asarray(mesh_pv.curvature("gaussian"))
    areas_pv = np.asarray(
        mesh_pv.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
    )
    lumped_pv = np.zeros(n_vertices)
    np.add.at(lumped_pv, mesh_tm.faces.ravel(), np.repeat(areas_pv / 3.0, 3))

    face_angles_wp = wp.array(mesh_tm.face_angles, dtype=wp.float32, device=mesh_wp.device)
    vertex_defects_wp = tw.vertices.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp)

    assert np.allclose(vertex_defects_wp.numpy(), gaussian_pv * lumped_pv, rtol=1e-4, atol=1e-4)
    # Non-vacuous on every fixture: a mesh whose defects were all zero would pass trivially.
    assert np.abs(vertex_defects_wp.numpy()).max() > 1e-2


@pytest.mark.parametrize(
    ("mesh_name", "chi"), [("icosahedron", 2), ("boy_surface", 1), ("bohemian_dome", 0)]
)
def test_vertex_defects_satisfy_gauss_bonnet(
    request: pytest.FixtureRequest, mesh_name: str, chi: int
) -> None:
    """
    Not a library comparison: the defects of a closed mesh sum to ``2π χ``, at any genus.

    No reference: this is the discrete Gauss-Bonnet theorem, and it is a stronger statement about
    the defects than a per-vertex comparison because it couples every vertex at once. The point of
    running it on these three fixtures is the **odd** characteristic: Boy's surface is the only
    input in the suite with χ = 1, so it is the only one that can catch a defect convention that is
    right up to a factor of two, or a sign that is right only on an orientable mesh.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    assert mesh_tm.is_watertight
    assert tw.measures.euler_characteristic(mesh_wp.indices) == chi

    face_angles_wp = wp.array(mesh_tm.face_angles, dtype=wp.float32, device=mesh_wp.device)
    defects_wp = tw.vertices.vertex_defects(
        mesh_tm.vertices.shape[0], mesh_wp.indices, face_angles_wp
    )
    assert np.isclose(defects_wp.numpy().sum(), 2.0 * np.pi * chi, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(
    not wp.is_cuda_available(),
    reason="needs a second device to make the current device differ from the arrays' device",
)
def test_scatter_wrappers_ignore_the_current_device(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the three ``vertices`` scatter wrappers answer on their inputs' device, not Warp's.

    Every other test runs with the arrays' device *as* the current device, so a ``wp.launch`` that
    forgets to forward ``device=`` resolves to the right answer by accident and the suite stays
    green. Pinning a different current device around the call is what makes the omission
    observable -- it was a real defect in all three of these functions.
    """
    mesh_tm, mesh_wp = half_torus
    n_vertices = mesh_tm.vertices.shape[0]

    face_normals_wp = points_to_warp(mesh_tm.face_normals, mesh_wp.device)
    face_angles_wp = wp.array(mesh_tm.face_angles, dtype=wp.float32, device=mesh_wp.device)

    mean_normals_tm = tm.geometry.mean_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals
    )
    weighted_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals, mesh_tm.face_angles
    )
    defects_tm = tm.curvature.vertex_defects(mesh_tm)

    with wp.ScopedDevice("cpu"):
        mean_normals_wp = tw.vertices.mean_vertex_normals(
            n_vertices, mesh_wp.indices, face_normals_wp
        )
        weighted_normals_wp = tw.vertices.weighted_vertex_normals(
            n_vertices, mesh_wp.indices, face_normals_wp, face_angles_wp
        )
        defects_wp = tw.vertices.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp)

    assert str(mean_normals_wp.device) == str(mesh_wp.device)
    assert str(weighted_normals_wp.device) == str(mesh_wp.device)
    assert str(defects_wp.device) == str(mesh_wp.device)
    assert np.allclose(mean_normals_wp.numpy(), mean_normals_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(weighted_normals_wp.numpy(), weighted_normals_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(defects_wp.numpy(), defects_tm, rtol=1e-5, atol=1e-5)
