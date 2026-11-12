"""Regression tests for ``ordito.normals`` against Trimesh (CPU reference)."""

import contextlib
from typing import Literal

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.conversions import (
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("vertex_normals", "open3d", "pymeshlab")
@pytest.mark.parity("mean_vertex_normals", "pymeshlab")
def test_vertex_normal_weightings_match_open3d_and_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The two unweighted-and-area weightings against the libraries that implement the same ones.

    Class A for all three, and they agree to **3.5e-7**, tighter than the 1e-5 tolerance, because
    ``compute_vertex_normals`` and ``compute_normal_per_vertex(weightmode="By Area")`` are the
    same scheme ordito implements, and ``"Simple Average"`` is the unweighted one.

    **trimesh is deliberately absent**, and that is the finding worth recording:
    ``Trimesh.vertex_normals`` is *angle*-weighted, not area-weighted. It matches
    ``vertex_normals`` at ``weighting="angle"`` to 4.3e-7 and differs from the area-weighted answer
    by up to **0.072** on this fixture; ``test_vertex_normals_angle_match_trimesh`` pins that.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size

    area_wp = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices)

    mesh_o3d = trimesh_to_open3d(mesh_tm)
    mesh_o3d.compute_vertex_normals()
    assert np.allclose(area_wp.numpy(), np.asarray(mesh_o3d.vertex_normals), rtol=1e-5, atol=1e-5)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_normal_per_vertex(weightmode="By Area")
    normals_pml = meshset_pml.current_mesh().vertex_normal_matrix()
    assert np.allclose(area_wp.numpy(), normals_pml, rtol=1e-5, atol=1e-5)

    # The unweighted scheme, from the same filter under a different weightmode.
    face_normals_wp, _areas_wp = od.triangles.face_normals_and_areas(
        mesh_wp.points, mesh_wp.indices
    )
    mean_wp = od.normals.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)
    mean_meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    mean_meshset_pml.compute_normal_per_vertex(weightmode="Simple Average")
    mean_pml = mean_meshset_pml.current_mesh().vertex_normal_matrix()
    assert np.allclose(mean_wp.numpy(), mean_pml, rtol=1e-5, atol=1e-5)

    # The two weightings are genuinely different, so neither assert above is weightless.
    assert not np.allclose(area_wp.numpy(), mean_wp.numpy(), atol=1e-3)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular"])
@pytest.mark.parity("vertex_normals", "trimesh")
def test_vertex_normals_angle_match_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: ``Trimesh.vertex_normals`` is ``vertex_normals(weighting="angle")``.

    trimesh weights each incident face normal by the corner angle, so it is compared against that
    weighting; the benchmark row times the default area weighting, which two of the group's other
    references compute (``test_vertex_normal_weightings_match_open3d_and_pymeshlab``). The second
    assert keeps the first a claim about the weighting: the area-weighted answer is measurably
    elsewhere on both fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    normals_tm = np.asarray(mesh_tm.vertex_normals)
    angle_wp = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices, weighting="angle")
    area_wp = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(angle_wp.numpy(), normals_tm, rtol=1e-5, atol=1e-5)
    assert not np.allclose(area_wp.numpy(), normals_tm, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("mean_vertex_normals", "pyvista")
def test_mean_vertex_normals_match_pyvista(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A, and the row names ``mean_vertex_normals`` for a measured reason.

    ``compute_normals``' point ``Normals`` is the *unweighted* sum of incident face normals, so it
    is this function and not one of ``vertex_normals``' three weightings: measured 5.1e-07 here
    against 4.5e-03 (area), 6.8e-03 (angle) on the same mesh. A row pointed at
    ``vertex_normals`` would therefore fail at ``1e-5`` rather than merely be loose, and the last
    assert keeps that separation live.

    The flags are the ones ``tests/test_triangles.py`` explains: no re-winding, no vertex splitting.
    The array comes back **float32**, which is the tolerance floor on VTK's side rather than
    ordito's.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size
    face_normals_wp, _areas_wp = od.triangles.face_normals_and_areas(
        mesh_wp.points, mesh_wp.indices
    )
    mean_wp = od.normals.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)

    normals_pv = trimesh_to_pyvista(mesh_tm).compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    normals_pv_np = np.asarray(normals_pv.point_data["Normals"])
    assert np.allclose(mean_wp.numpy(), normals_pv_np, rtol=1e-5, atol=1e-5)

    area_wp = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices)
    assert not np.allclose(area_wp.numpy(), normals_pv_np, atol=1e-4)


@pytest.mark.parametrize(
    ("mesh_name", "scale"), [("sphere_irregular", 3e-6), ("sphere_irregular", 1e-9)]
)
def test_vertex_normals_area_matches_igl_at_any_scale(
    request: pytest.FixtureRequest, mesh_name: str, scale: float
) -> None:
    """
    Class A: area weighting against ``igl.per_vertex_normals``' area-weighted mode, at any scale.

    igl is the reference here rather than trimesh, because trimesh has no area-weighted mode. The
    reference is taken at unit scale: a normal is a direction, not a length, so shrinking the mesh
    must not move it. The two small scales pin the magnitude handling. ``|cross|`` scales as
    ``h^2``, so an absolute floor on it inside ``kernels.triangles.face_normals_and_area`` put every
    face of a mesh at ``h <= 3e-6`` below the floor and returned the raw cross product where a unit
    normal was promised; area-weighting then squared that and ``wp.normalize`` saw a ``float32``
    ``length_sq`` underflowed to zero.
    Measured before the fix: **every** vertex normal came back exactly zero at both small scales,
    with nothing raised.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    normals_igl = igl.per_vertex_normals(
        vertices_np,
        np.asarray(mesh_tm.faces, dtype=np.int32),
        igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
    )
    vertices_wp = points_to_warp(vertices_np * scale, mesh_wp.device)

    normals_wp = od.normals.vertex_normals(vertices_wp, mesh_wp.indices)

    # Non-vacuity: the reference is unit everywhere, so nothing here is comparing two zeros.
    assert np.allclose(np.linalg.norm(normals_igl, axis=1), 1.0)
    assert np.allclose(normals_wp.numpy(), normals_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("vertex_normals", "pytorch3d")
def test_vertex_normals_match_pytorch3d(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: ``Meshes.verts_normals_packed`` is ordito's **area**-weighted vertex normal.

    pytorch3d sums the *unnormalized* face cross products into each incident vertex and normalizes
    once at the end, which is area weighting by construction -- so this pins the default
    ``weighting="area"`` and nothing else. Measured 1.19e-07 against it, and **1.14e-02** against
    ``weighting="angle"``, five orders apart: the second assert is what makes the first a claim
    about the weighting convention rather than about vertex normals in general.

    A closed, fully-referenced fixture on purpose. pytorch3d does not drop unreferenced vertices
    and leaves their normals at the ``torch.zeros`` initial value rather than ``NaN``, so on a mesh
    with spares this would be comparing zeros with whatever ordito writes.
    """
    mesh_tm, mesh_wp = sphere_irregular
    device = str(mesh_wp.points.device)
    normals_t = trimesh_to_pytorch3d(mesh_tm, device).verts_normals_packed()
    assert normals_t is not None
    normals_p3d = normals_t.cpu().numpy()
    normals_wp = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices)
    angle_wp = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices, weighting="angle")

    assert normals_p3d.shape == mesh_tm.vertices.shape
    assert np.allclose(normals_wp.numpy(), normals_p3d, rtol=1e-5, atol=1e-6)
    assert not np.allclose(angle_wp.numpy(), normals_p3d, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("vertex_normals", "meshlib")
def test_vertex_normal_weightings_match_meshlib(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A, and the reason to have it: MeshLib pins *which* weighting each name means.

    The pairing is not guessable from the names and was found by measuring all six combinations.
    ``computePerVertNormals`` is the **area**-weighted sum and ``computePerVertPseudoNormals`` is
    the **angle**-weighted one; each matches its ordito partner to **1.19e-07** and sits
    **6.8e-03** from the other's, which is four orders of magnitude of separation and far more than
    any tolerance argument. ``mean_vertex_normals`` matches neither (4.5e-03 from the nearer), so
    it keeps its own oracles and is asserted here only to be *different* -- without that, a
    regression collapsing all three weightings to one would still pass the two positive asserts.

    No other reference in the suite distinguishes these: igl exposes an area mode and trimesh an
    angle-weighted one, but neither can say what the other's name would mean.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_tm.vertices.shape[0]
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    area_ml = mn.toNumpyArray(mm.computePerVertNormals(mesh_ml))
    angle_ml = mn.toNumpyArray(mm.computePerVertPseudoNormals(mesh_ml))

    area_wp = od.normals.vertex_normals(vertices_wp, mesh_wp.indices)
    angle_wp = od.normals.vertex_normals(vertices_wp, mesh_wp.indices, weighting="angle")
    face_normals_wp, _ = od.triangles.face_normals_and_areas(vertices_wp, mesh_wp.indices)
    mean_wp = od.normals.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)

    assert area_ml.shape == (n_vertices, 3)  # non-vacuity: the converter kept every vertex
    assert np.allclose(area_wp.numpy(), area_ml, rtol=1e-5, atol=1e-5)
    assert np.allclose(angle_wp.numpy(), angle_ml, rtol=1e-5, atol=1e-5)

    # The separation, without which the two asserts above would not pin a weighting at all.
    assert not np.allclose(area_wp.numpy(), angle_ml, atol=1e-4)
    assert not np.allclose(angle_wp.numpy(), area_ml, atol=1e-4)
    assert not np.allclose(mean_wp.numpy(), area_ml, atol=1e-4)


@pytest.mark.parametrize("weighting", ["area", "angle"])
def test_vertex_normals_precomputed_face_quantities(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], weighting: Literal["area", "angle"]
) -> None:
    """
    Ordito against ordito: supplied face normals and weights equal the ones the call derives.

    The derived path carries the oracle (open3d's and igl's area mode, MeshLib's angle weighting
    above); this pins that the ``face_normals=`` / ``face_weights=`` path accumulates them the same
    way.
    """
    mesh_tm, mesh_wp = saddle_graded
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)

    face_normals_wp, face_areas_wp = od.triangles.face_normals_and_areas(
        vertices_wp, mesh_wp.indices
    )
    face_weights_wp = (
        face_areas_wp
        if weighting == "area"
        else od.triangles.face_angles(vertices_wp, mesh_wp.indices)
    )
    vertex_normals_precomputed_wp = od.normals.vertex_normals(
        vertices_wp,
        mesh_wp.indices,
        weighting=weighting,
        face_normals=face_normals_wp,
        face_weights=face_weights_wp,
    )
    vertex_normals_wp = od.normals.vertex_normals(vertices_wp, mesh_wp.indices, weighting=weighting)
    assert np.allclose(
        vertex_normals_precomputed_wp.numpy(), vertex_normals_wp.numpy(), rtol=1e-5, atol=1e-5
    )
    # Half supplied: the call derives the other table, and the result is the fully supplied one's
    # exactly, since both scatter the same per-face values.
    normals_only_wp = od.normals.vertex_normals(
        vertices_wp, mesh_wp.indices, weighting=weighting, face_normals=face_normals_wp
    )
    weights_only_wp = od.normals.vertex_normals(
        vertices_wp, mesh_wp.indices, weighting=weighting, face_weights=face_weights_wp
    )
    assert np.array_equal(normals_only_wp.numpy(), vertex_normals_precomputed_wp.numpy())
    assert np.array_equal(weights_only_wp.numpy(), vertex_normals_precomputed_wp.numpy())


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
def test_vertex_normals_mwselr(scale: float, saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
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
    mesh_tm, mesh_wp = saddle_graded

    # Both sides see the same ``float32``-rounded scaled vertices: scaling by 1e-3 is inexact, and
    # on ``saddle_graded``'s aspect-4 800 needles one rounding of the inputs moves a normal past
    # the 1e-5 compared here.
    vertices_np = (np.array(mesh_tm.vertices, dtype=np.float64) * scale).astype(np.float32)
    vertices_np = vertices_np.astype(np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    vertex_normals_np = _compute_max_vertex_normals_np(vertices_np, faces_np)
    # Non-vacuity: the oracle itself must be unit normals, not the zero rows the defect produced.
    assert np.allclose(np.linalg.norm(vertex_normals_np, axis=1), 1.0)

    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    vertex_normals_wp = od.normals.vertex_normals(vertices_wp, mesh_wp.indices, weighting="mwselr")
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_np, rtol=1e-5, atol=1e-5)

    # The face normals of the same rounded vertices: unit vectors, but on a needle the rounding of
    # the scaled corners moves them (3.8e-5 at 1e-3 if taken from the unscaled mesh).
    face_normals_wp = points_to_warp(
        tm.Trimesh(vertices_np, faces_np, process=False).face_normals, mesh_wp.device
    )
    vertex_normals_explicit_wp = od.normals.vertex_normals(
        vertices_wp, mesh_wp.indices, weighting="mwselr", face_normals=face_normals_wp
    )
    assert np.allclose(vertex_normals_explicit_wp.numpy(), vertex_normals_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("weighting", ["area", "angle", "mwselr", "mean"])
@pytest.mark.parametrize("mesh_name", ["saddle_graded", "icosphere_poles"])
def test_vertex_normals_are_reproducible(
    request: pytest.FixtureRequest,
    device: str,
    mesh_name: str,
    weighting: Literal["area", "angle", "mwselr", "mean"],
) -> None:
    """
    Ordito against ordito: under the deterministic setting the identical call repeats bit for bit.

    No reference library can carry this -- it is a claim about *this* implementation's summation
    order, not about the quantity -- so the values' oracles are the library comparisons above, and
    this pins repeatability on top of them. Off, a vertex's ``float32`` terms sum in a ``float64``
    accumulator in the atomics' arrival order, which is not exact when one vertex's terms span
    more than ``2^29``: on ``od.creation.icosphere(5)`` the poles' x terms are ``+-1.76e-2`` and
    ``+-5.1e-11`` and their sum came out ``-3.5e-18``, ``0`` or ``3.5e-18`` by arrival order (11 of
    40 angle-weighted calls differed on CUDA, so 32 runs miss it with odds under ``1e-4``). On,
    the terms take the exact group sum. Off agrees with on to rounding.

    Exact equality on purpose: the arrival-order form fails it on the icosphere, a tolerance of
    1e-6 would not. ``test_principal_curvature_is_reproducible`` is the downstream consumer.
    """
    if mesh_name == "icosphere_poles":
        vertices_wp, faces_wp = od.creation.icosphere(subdivisions=5, device=device)
    else:
        _, mesh_wp = request.getfixturevalue(mesh_name)
        vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    def normals() -> np.ndarray:
        if weighting == "mean":
            face_normals_wp, _ = od.triangles.face_normals_and_areas(vertices_wp, faces_wp)
            return od.normals.mean_vertex_normals(
                vertices_wp.size, faces_wp, face_normals_wp
            ).numpy()
        return od.normals.vertex_normals(vertices_wp, faces_wp, weighting=weighting).numpy()

    with od.config.deterministic(True):
        runs = [normals() for _ in range(32)]
    with od.config.deterministic(False):
        arrival_np = normals()

    # Non-vacuity: an all-zero or constant buffer would compare equal to itself for free.
    assert np.allclose(np.linalg.norm(runs[0], axis=1), 1.0)
    assert np.ptp(runs[0], axis=0).max() > 1.0
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)
    assert np.allclose(arrival_np, runs[0], rtol=1e-5, atol=1e-5)


def test_vertex_normals_guards_and_faceless_mwselr(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the argument guards and the face-free ``"mwselr"`` contract.

    No reference raises on these inputs. An unknown weighting, weights with ``"mwselr"`` and a
    per-face table one row short each raise the documented ``ValueError``; a mesh with no faces has
    no incident normal to average, so every vertex keeps the documented zero row.
    """
    mesh_tm, mesh_wp = saddle_graded
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    with pytest.raises(ValueError, match="weighting must be"):
        od.normals.vertex_normals(vertices_wp, mesh_wp.indices, weighting="cotangent")  # pyright: ignore[reportArgumentType]
    face_weights_wp = wp.ones(len(mesh_tm.faces), dtype=wp.float32, device=mesh_wp.device)
    with pytest.raises(ValueError, match='not accepted with weighting="mwselr"'):
        od.normals.vertex_normals(
            vertices_wp, mesh_wp.indices, weighting="mwselr", face_weights=face_weights_wp
        )
    short_normals_wp = points_to_warp(mesh_tm.face_normals[:-1], mesh_wp.device)
    with pytest.raises(ValueError, match="face_normals must have one row per triangle"):
        od.normals.vertex_normals(
            vertices_wp, mesh_wp.indices, weighting="mwselr", face_normals=short_normals_wp
        )

    no_faces_wp = wp.zeros(0, dtype=wp.int32, device=mesh_wp.device)
    normals_np = od.normals.vertex_normals(vertices_wp, no_faces_wp, weighting="mwselr").numpy()
    assert np.array_equal(normals_np, np.zeros((len(mesh_tm.vertices), 3), dtype=np.float32))


@pytest.mark.parity("weighted_vertex_normals", "trimesh")
def test_scatter_wrappers_match_trimesh_on_their_inputs_device(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class A: the two scatter wrappers against trimesh, on their inputs' device.

    ``mean_vertex_normals`` and ``weighted_vertex_normals`` are compared with
    ``trimesh.geometry``'s functions of the same names, fed trimesh's own face normals and corner
    angles, so the comparison isolates the accumulation and the weighting rule from
    [`triangles.face_normals_and_areas`], which has its own oracle.

    Every other test runs with the arrays' device *as* the current device, so a ``wp.launch`` that
    forgets to forward ``device=`` resolves to the right answer by accident and the suite stays
    green. Where the arrays are on CUDA the calls run under a CPU current device, which is what
    makes the omission observable -- it was a real defect in both of these functions. On the CPU
    device there is no second device to switch to, and the comparisons alone remain.
    """
    mesh_tm, mesh_wp = saddle_graded
    n_vertices = mesh_tm.vertices.shape[0]

    face_normals_wp = points_to_warp(mesh_tm.face_normals, mesh_wp.device)
    face_angles_wp = _face_angles_wp(mesh_tm, mesh_wp.device)

    mean_normals_tm = tm.geometry.mean_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals
    )
    weighted_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals, mesh_tm.face_angles
    )

    other_device = (
        wp.ScopedDevice("cpu")
        if wp.get_device(mesh_wp.device).is_cuda
        else contextlib.nullcontext()
    )
    with other_device:
        mean_normals_wp = od.normals.mean_vertex_normals(
            n_vertices, mesh_wp.indices, face_normals_wp
        )
        weighted_normals_wp = od.normals.weighted_vertex_normals(
            n_vertices, mesh_wp.indices, face_normals_wp, face_angles_wp
        )

    assert str(mean_normals_wp.device) == str(mesh_wp.device)
    assert str(weighted_normals_wp.device) == str(mesh_wp.device)
    assert np.allclose(mean_normals_wp.numpy(), mean_normals_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(weighted_normals_wp.numpy(), weighted_normals_tm, rtol=1e-5, atol=1e-5)


def _face_angles_wp(mesh_tm: tm.Trimesh, device: wp.DeviceLike) -> odt.Array2dFloat32:
    """Upload trimesh's ``(n_faces, 3)`` corner angles as the ``face_angles`` table ordito takes."""
    return odt.as_array2d(
        wp.array(mesh_tm.face_angles, dtype=wp.float32, device=device), wp.float32
    )
