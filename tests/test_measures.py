"""
Regression tests for ``ordito.measures``: the whole-mesh reductions.

Against ``trimesh``'s cached mass properties, ``igl.moments`` and pymeshlab's
``get_geometric_measures``, which answers six of these questions in one call. Mirrors the module's
source order -- volume, surface centroid, moments, Euler characteristic.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.conftest import MESHES
from tests.conversions import (
    faces_igl,
    numpy_to_warp,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    trimesh_to_warp,
)


@pytest.mark.parity("moments", "meshlib")
def test_volume(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the divergence-theorem volume against ``Trimesh.volume`` and MeshLib's ``volume``.

    A closed mesh is required for the integral to mean anything, which is why the fixture is
    closed and the open-mesh behaviour is not asserted here.

    The parity marker names the ``moments`` group rather than a group of its own, because that is
    where the integral family is timed and MeshLib's ``volume`` sits in it exactly as pyvista's
    ``PolyData.volume`` does -- the volume alone, no centre of mass and no inertia tensor. Its
    ``region`` argument defaults to the whole mesh, which is the comparison; ordito has no
    per-region form, so nothing is left untested by passing ``None``.
    """
    mesh_tm, mesh_wp = icosahedron
    mesh_ml = trimesh_to_meshlib(mesh_tm)

    volume_wp = od.measures.volume(mesh_wp.points, mesh_wp.indices)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5, atol=1e-5)
    assert np.isclose(volume_wp, mm.volume(mesh_ml.topology, mesh_ml.points), rtol=1e-5, atol=1e-5)


def test_volume_inward_normals_negative(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: reversing every face negates the volume, and trimesh agrees on the sign.

    The *signed* result is the claim -- an implementation taking the absolute value would pass
    [`test_volume`] and fail this, and ``validation.is_volume`` depends on the sign.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_np = mesh_tm.faces[:, ::-1].reshape(-1).astype(np.int32)
    faces_flipped_wp = wp.array(
        np.ascontiguousarray(faces_np), dtype=wp.int32, device=mesh_wp.device
    )
    volume_wp = od.measures.volume(mesh_wp.points, faces_flipped_wp)
    assert np.isclose(volume_wp, -mesh_tm.volume, rtol=1e-5, atol=1e-5)


def test_volume_empty(device: str):
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.array([], dtype=wp.int32, device=device)
    assert od.measures.volume(vertices, faces) == 0.0


@pytest.mark.parity("surface_centroid", "trimesh", "meshlib")
def test_surface_centroid(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the area-weighted surface centroid against ``Trimesh.centroid`` and MeshLib's.

    An open fixture, because on a closed symmetric mesh the centroid sits at the origin and a
    wrongly-weighted sum lands there too -- and this fixture is translated as well, so a function
    returning the origin fails rather than passing by luck.

    MeshLib carries **two** centres and picking the wrong one would still pass a symmetric fixture:
    ``findCenterFromFaces`` is the area-weighted surface centroid ordito computes, while
    ``findCenterFromPoints`` is the plain vertex mean that [`centroid`][ordito.points.centroid]
    computes. The final assert holds them apart -- measured 2.63 apart on ``half_torus`` -- so a
    future rebinding to the other one cannot pass quietly.
    """
    mesh_tm, mesh_wp = hemisphere
    mesh_ml = trimesh_to_meshlib(mesh_tm)

    centroid_tm = mesh_tm.centroid
    centroid_ml = np.array([*mm.findCenterFromFaces(mesh_ml.topology, mesh_ml.points)])

    centroid_wp = od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)
    centroid_wp = np.array(odt.vec3_floats(centroid_wp))
    assert np.allclose(centroid_wp, centroid_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(centroid_wp, centroid_ml, rtol=1e-5, atol=1e-5)

    # The vertex mean is a different point, and is ordito's points.centroid rather than this.
    vertex_mean_ml = np.array([*mm.findCenterFromPoints(mesh_ml.topology, mesh_ml.points)])
    assert not np.allclose(centroid_wp, vertex_mean_ml, atol=1e-3)
    assert np.allclose(
        od.points.centroid(mesh_wp.points).numpy()[0], vertex_mean_ml, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_surface_centroid_matches_trimesh_on_a_skewed_mesh_on_both_devices(kernel_device: str):
    """
    Class A on both devices: the area-weighted centroid sum on an asymmetric mesh.

    Three things this guards. ``wp.launch_tiled`` runs exactly one lane per block on Warp 1.18's CPU
    backend, so the block-wide ``wp.tile_sum`` this reduction used to perform accumulated one face
    per 64-face tile there. A *symmetric* mesh hides that completely -- the centroid of every 64th
    face of a sphere is still the sphere's centre -- which is why the mesh is stretched and sheared
    first. Measured: the sub-sampled sum was off by 1.1e-2 on this shape and by 4e-8 on the
    unmodified sphere. And the reduction now has two implementations (``centroid_tiled`` on CUDA,
    ``centroid_sliced`` on CPU), so the parametrization is what covers both of them -- running this
    on one device only would leave a whole kernel untested.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    mesh_tm = tm.creation.icosphere(subdivisions=3)
    mesh_tm.vertices[:, 0] *= 6.0
    mesh_tm.vertices[:, 2] += 0.4 * mesh_tm.vertices[:, 1] ** 3
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)

    centroid_wp = od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        np.array(odt.vec3_floats(centroid_wp)), mesh_tm.centroid, rtol=1e-4, atol=1e-4
    )


def test_surface_centroid_empty(device: str):
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.array([], dtype=wp.int32, device=device)
    centroid_wp = od.measures.surface_centroid(vertices, faces)
    centroid_wp = np.array(odt.vec3_floats(centroid_wp))
    assert np.isnan(centroid_wp).all()


def test_surface_centroid_all_degenerate(device: str):
    """
    Not a library comparison: the other input whose area weights sum to zero.

    An area-weighted mean of nothing is the empty mesh's answer, and a mesh of collinear faces is
    the same statement with faces present -- it used to be a bare ``ZeroDivisionError``, which the
    docstring documented nowhere and which no other reduction here raises (``moments`` already
    guards its zero volume the same way).
    """
    vertices_wp, faces_wp = numpy_to_warp(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        device,
    )

    # Non-vacuity: every face really is degenerate, so the weights really do all vanish.
    _, areas_wp = od.triangles.face_normals_and_areas(vertices_wp, faces_wp)
    assert float(areas_wp.numpy().max()) == 0.0
    centroid_wp = od.measures.surface_centroid(vertices_wp, faces_wp)
    assert np.isnan(odt.vec3_floats(centroid_wp)).all()


@pytest.mark.parity("surface_centroid", "pymeshlab")
def test_surface_centroid_matches_pymeshlab_shell_barycenter(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
):
    """
    The area-weighted centroid against MeshLab's ``shell_barycenter``.

    ``get_geometric_measures`` answers six questions in one call -- which is why
    ``benchmarks/test_triangles.py`` reads its row as an *upper* bound on the centroid alone -- but
    "upper bound" is a statement about cost, not about the value. Indexing the dict is the whole
    transform (Class B).

    The distinction that matters here is that the same call also returns ``barycenter``, the plain
    *vertex* mean, and the two differ on any mesh with uneven triangle areas. Asserting against
    ``shell_barycenter`` while checking that ``barycenter`` is measurably different is what makes
    this a real test of the area weighting rather than of a centre of mass in general: on
    ``hemisphere`` they sit about 0.02 apart, some 400x the 1e-5 tolerance.
    """
    mesh_tm, mesh_wp = hemisphere
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_geometric_measures()

    centroid_wp = od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)
    centroid_np = np.array(odt.vec3_floats(centroid_wp))

    assert np.allclose(centroid_np, measures_pml["shell_barycenter"], rtol=1e-5, atol=1e-5)
    # The vertex mean is a different quantity; if it were not, the assert above would be weightless.
    assert not np.allclose(measures_pml["barycenter"], measures_pml["shell_barycenter"], atol=1e-3)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("moments", "igl", "trimesh")
def test_moments(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B on igl (its first moment is un-normalised), Class A on trimesh.

    ``igl.moments`` returns ``(m0, m1, m2)`` where ``m1`` is the centre of mass **times the mass**,
    so the named transform is ``m1 / m0``. ``m2`` needs no transform: it is already referred to the
    centre of mass rather than the origin, which was verified against a *translated* mesh rather
    than assumed -- on a mesh centred at the origin the two references coincide and the check would
    be vacuous.

    trimesh answers all three too (``volume`` / ``center_mass`` / ``moment_inertia``), so this is a
    three-way comparison of the same integrals.

    The inertia tolerance is relative to the tensor's own scale: ordito integrates ``float32``
    positions in ``float64``, so the deviation tracks the positions' precision (measured 3.5e-8
    relative on ``icosahedron``, and 3.6e-15 on an axis-aligned box whose coordinates are exact in
    ``float32``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)

    volume_igl, first_moment_igl, inertia_igl = igl.moments(vertices_np, faces_igl(mesh_tm))
    volume_wp, center_wp, inertia_wp = od.measures.moments(mesh_wp.points, mesh_wp.indices)

    assert np.isclose(volume_wp, volume_igl, rtol=1e-5)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5)
    assert np.allclose(
        odt.vec3_floats(center_wp), np.asarray(first_moment_igl) / volume_igl, rtol=1e-4, atol=1e-5
    )
    assert np.allclose(odt.vec3_floats(center_wp), mesh_tm.center_mass, rtol=1e-4, atol=1e-5)
    scale = float(np.abs(np.asarray(inertia_igl)).max())
    assert (
        np.abs(np.asarray(inertia_wp).reshape(3, 3) - np.asarray(inertia_igl)).max() < 1e-5 * scale
    )
    assert (
        np.abs(np.asarray(inertia_wp).reshape(3, 3) - mesh_tm.moment_inertia).max() < 1e-5 * scale
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("moments", "pyvista")
def test_volume_matches_pyvista(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A on the volume half of ``moments``: ``PolyData.volume`` is the same divergence integral.

    VTK computes it as this module does -- a signed sum of per-face tetrahedra -- so no transform
    applies and the agreement is float32-limited (measured 2.5e-07 on ``icosahedron``, where ordito
    reads 2.536150455 against 2.536150710 and trimesh's float64 answer is 2.536150710).

    ``cave_cube`` is the fixture that makes the *sign* convention matter: it is a shell whose inner
    box is wound inwards, so a volume taken without signs would report the outer box alone.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    volume_pv = float(trimesh_to_pyvista(mesh_tm).volume)
    assert volume_pv > 0.0

    assert np.isclose(od.measures.volume(mesh_wp.points, mesh_wp.indices), volume_pv, rtol=1e-5)
    # Non-vacuous on cave_cube: the outer box alone is a measurably different number.
    assert np.isclose(volume_pv, mesh_tm.volume, rtol=1e-5)


def test_moments_center_of_mass_differs_from_the_surface_centroid(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
):
    """
    The two centres are different quantities, which is why both functions exist.

    ``surface_centroid`` is the area-weighted centre of the *shell* and ``moments``' is the volume
    centre of the solid. On a closed uniform sphere they coincide; on anything asymmetric they do
    not, and
    asserting they differ is what keeps ``moments`` from being a synonym.
    """
    _mesh_tm, mesh_wp = half_torus
    surface_centroid = od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)
    _volume, center_of_mass, _inertia = od.measures.moments(mesh_wp.points, mesh_wp.indices)
    assert not np.allclose(
        odt.vec3_floats(surface_centroid), odt.vec3_floats(center_of_mass), atol=1e-3
    )


def test_moments_translation_shifts_only_the_center(device: str):
    """
    Translation moves the centre of mass and leaves the volume and inertia tensor alone.

    The parallel-axis shift is what makes that true, and getting it wrong is invisible on any mesh
    centred at the origin -- which is why this fixture is deliberately offset.
    """
    box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0])
    offset_np = np.array([1.5, -2.0, 0.5])

    def moments_of(vertices_np: np.ndarray):
        vertices_wp = points_to_warp(vertices_np, device)
        faces_wp = wp.array(
            np.ascontiguousarray(box_tm.faces.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        return od.measures.moments(vertices_wp, faces_wp)

    volume_a, center_a, inertia_a = moments_of(box_tm.vertices)
    volume_b, center_b, inertia_b = moments_of(box_tm.vertices + offset_np)

    assert np.isclose(volume_a, volume_b, rtol=1e-5)
    assert np.allclose(
        odt.vec3_floats(center_b), np.array(odt.vec3_floats(center_a)) + offset_np, atol=1e-5
    )
    assert np.abs(np.asarray(inertia_a) - np.asarray(inertia_b)).max() < 1e-4 * float(
        np.abs(np.asarray(inertia_a)).max()
    )


def test_moments_empty(device: str):
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    volume, center, inertia = od.measures.moments(vertices_wp, faces_wp)
    assert volume == 0.0
    assert np.isnan(odt.vec3_floats(center)).all()
    assert np.array_equal(np.asarray(inertia).reshape(3, 3), np.zeros((3, 3)))


@pytest.mark.parametrize("mesh_name", MESHES)
def test_euler_characteristic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on an integer, against ``Trimesh.euler_number`` across the fixture set.

    Non-vacuous by construction: the parametrisation spans chi = 2, 1 and 0, so a constant
    answer cannot pass.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    euler_wp = od.measures.euler_characteristic(mesh_wp.indices)
    assert euler_wp == int(mesh_tm.euler_number)


def test_euler_characteristic_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    assert od.measures.euler_characteristic(mesh_wp.indices) == 2
