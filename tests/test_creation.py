"""Regression tests for ``ordito.creation`` against Trimesh (CPU reference)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias, cast

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.utils as p3d_utils
import pyvista as pv
import shapely.geometry as sg
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import cKDTree

import ordito as od
from ordito.creation import ParametricSurfaceKind
from tests.comparisons import (
    assert_unordered_rows_equal,
    euler_characteristic,
    lexsort_rows,
    open_edge_count,
)
from tests.conversions import (
    meshlib_to_trimesh,
    open3d_to_trimesh,
    points_to_warp,
    points_to_warp_uv,
    pytorch3d_to_numpy,
    warp_empty,
    warp_to_trimesh,
)

# Every surface ``_build_parametric`` builds: the lattice kinds and the two superquadric builders.
_Surface: TypeAlias = ParametricSurfaceKind | Literal["super_ellipsoid", "super_toroid"]


def _assert_same_faces(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], mesh_tm: tm.Trimesh
) -> None:
    """
    Assert the ordito result is the same mesh as ``mesh_tm`` up to vertex and face order.

    Vertex order genuinely differs — ordito compacts its buffers rather than reproducing trimesh's
    merge order — so the comparison matches face *centroid* sets and requires the match to be a
    bijection. A plain lexsort is too brittle here: ordito works in ``float32`` and trimesh in
    ``float64``, so near-tied coordinates sort differently on the two sides.
    """
    vertices_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    assert len(vertices_np) == mesh_tm.vertices.shape[0], (
        f"vertex count mismatch: got {len(vertices_np)}, expected {mesh_tm.vertices.shape[0]}"
    )
    assert faces_np.shape[0] == mesh_tm.faces.shape[0], (
        f"face count mismatch: got {faces_np.shape[0]}, expected {mesh_tm.faces.shape[0]}"
    )
    centroids_wp = vertices_np[faces_np].mean(axis=1)
    centroids_tm = mesh_tm.vertices[mesh_tm.faces].mean(axis=1)
    distance_np, match_np = cKDTree(centroids_tm).query(centroids_wp)
    assert np.max(distance_np) < 1e-5, f"face centroids differ by up to {np.max(distance_np):.3e}"
    assert len(set(np.asarray(match_np).tolist())) == len(np.asarray(match_np)), (
        "face centroid match is not a bijection"
    )


def _assert_same_solid(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], mesh_tm: tm.Trimesh
) -> None:
    """
    Assert the ordito result is the same solid as ``mesh_tm`` without pinning its triangulation.

    Used wherever the result contains an ear-clipped cap: ordito's clipper and trimesh's earcut
    pick different (equally valid) diagonals, so the face *count* and the enclosed volume agree
    while individual cap triangles do not.
    """
    assert vertices_wp.size == mesh_tm.vertices.shape[0]
    assert faces_wp.size // 3 == mesh_tm.faces.shape[0]
    assert np.isclose(warp_to_trimesh(vertices_wp, faces_wp).volume, mesh_tm.volume, rtol=1e-4)
    assert np.allclose(
        warp_to_trimesh(vertices_wp, faces_wp).bounds, mesh_tm.bounds, rtol=1e-5, atol=1e-5
    )


def _assert_closed(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> None:
    """
    Assert the mesh bounds a volume with no cracks: trimesh's ``is_watertight`` semantics.

    Deliberately not [`ordito.validation.is_watertight`][], which follows Open3D and also
    requires the mesh not to self-intersect. That extra check reports the coplanar edge-adjacent
    triangles of a flat cap as intersecting — it says ``True`` for the output of
    ``trimesh.creation.annulus`` too — so it cannot tell a correct cap from a broken one here.
    """
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert od.validation.is_winding_consistent(faces_wp)
    assert warp_to_trimesh(vertices_wp, faces_wp).volume > 0.0


def _assert_same_vertices_and_faces(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], mesh_tm: tm.Trimesh
) -> None:
    """Assert an exact face-set match after remapping ordito's vertices onto trimesh's."""
    faces_np = faces_wp.numpy().reshape(-1, 3)
    distance_np, remap_np = cKDTree(mesh_tm.vertices).query(vertices_wp.numpy().astype(np.float64))
    assert np.max(distance_np) < 1e-5, f"vertices differ by up to {np.max(distance_np):.3e}"
    mapped_np = np.sort(np.asarray(remap_np)[faces_np], axis=1)
    reference_np = np.sort(mesh_tm.faces, axis=1)
    assert np.array_equal(lexsort_rows(mapped_np), lexsort_rows(reference_np))


def _mat44(matrix_np: np.ndarray) -> wp.mat44:
    return wp.mat44(*matrix_np.flatten().tolist())


def _face_frames(points_np: np.ndarray, faces_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-face centroid and unit normal, for matching two triangulations of the same surface."""
    corners_np = points_np[faces_np]
    normal_np = np.cross(corners_np[:, 1] - corners_np[:, 0], corners_np[:, 2] - corners_np[:, 0])
    return corners_np.mean(axis=1), normal_np / np.maximum(
        np.linalg.norm(normal_np, axis=1, keepdims=True), 1e-30
    )


def _build_parametric(
    surface: _Surface, resolution: int, device: str
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Build one surface at ``resolution`` in both directions, whichever builder owns it."""
    if surface == "super_ellipsoid":
        return od.creation.super_ellipsoid(
            u_resolution=resolution, v_resolution=resolution, device=device
        )
    if surface == "super_toroid":
        return od.creation.super_toroid(
            u_resolution=resolution, v_resolution=resolution, device=device
        )
    return od.creation.parametric_surface(surface, resolution, resolution, device=device)


class _Topology(NamedTuple):
    """The measured topology of one parametric surface at ``u_res = v_res = 40``."""

    points: int
    faces: int
    open_edges: int
    chi: int
    loops: int
    orientable: bool
    watertight: bool
    pole_cells: int
    """Triangles the pole rows remove, in units of ``resolution - 1``."""


# Every column measured against pyvista at ``clean=True`` and trimesh, and each row is a class of
# input the rest of the suite has no other example of. ``klein`` is deliberately here despite its
# name: as VTK parameterizes it, it welds to two boundary loops and is *orientable*, so only
# ``figure8_klein`` is the closed non-orientable chi = 0 surface.
_PARAMETRIC_TABLE: dict[str, _Topology] = {
    "bohemian_dome": _Topology(1521, 3042, 0, 0, 0, True, True, 0),
    "bour": _Topology(1522, 3003, 39, 1, 1, True, False, 1),
    "boy": _Topology(1483, 2964, 0, 1, 0, False, True, 2),
    "catalan_minimal": _Topology(1600, 3042, 156, 1, 1, True, False, 0),
    "conic_spiral": _Topology(1522, 3003, 39, 1, 1, True, False, 1),
    "cross_cap": _Topology(1483, 2964, 0, 1, 0, False, True, 2),
    "dini": _Topology(1600, 3042, 156, 1, 1, True, False, 0),
    "enneper": _Topology(1600, 3042, 156, 1, 1, True, False, 0),
    "figure8_klein": _Topology(1521, 3042, 0, 0, 0, False, True, 0),
    "henneberg": _Topology(1560, 3042, 78, 0, 1, False, False, 0),
    "klein": _Topology(1560, 3042, 78, 0, 2, True, False, 0),
    "kuen": _Topology(1561, 3003, 117, 1, 1, True, False, 1),
    "mobius": _Topology(1560, 3042, 78, 0, 1, False, False, 0),
    "plucker_conoid": _Topology(1560, 3042, 78, 0, 2, True, False, 0),
    "pseudosphere": _Topology(1560, 3042, 78, 0, 2, True, False, 0),
    "roman": _Topology(1521, 3042, 0, 0, 0, False, True, 0),
    "super_ellipsoid": _Topology(1484, 2964, 0, 2, 0, True, True, 2),
    "super_toroid": _Topology(1521, 3042, 0, 0, 0, True, True, 0),
}

_PARAMETRIC_PYVISTA = {
    name: "Parametric" + "".join(part.capitalize() for part in name.split("_"))
    for name in _PARAMETRIC_TABLE
}
_PARAMETRIC_PYVISTA["figure8_klein"] = "ParametricFigure8Klein"

# A rectangle and a non-convex L, as counter-clockwise 2D rings.
_SQUARE_RING = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
_L_RING = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])


# --- open3d primitives ------------------------------------------------------------------


@pytest.mark.parity("box", "open3d")
@pytest.mark.parity("cylinder", "open3d")
@pytest.mark.parity("cone", "open3d")
@pytest.mark.parity("torus", "open3d")
def test_primitives_match_open3d(device: str) -> None:
    """
    Four primitives against Open3D's, which agree exactly -- same counts, same volume, same area.

    Worth asserting rather than assuming: these are table-driven generators, so a wrong cap winding
    or a dropped seam ring changes the enclosed volume while leaving the face count intact, and a
    count check alone would miss it. Class A on the counts, Class B on volume and area (both are
    functions of the mesh, not of its vertex ordering, which the two libraries do not share).

    ``uv_sphere`` is deliberately not here -- its tessellation parameter does not map
    one-for-one, and it gets its own test below.
    """
    for name, (vertices_wp, faces_wp), mesh_o3d in (
        (
            "box",
            od.creation.box(extents=(1.0, 2.0, 3.0), device=device),
            o3d.geometry.TriangleMesh.create_box(1.0, 2.0, 3.0),
        ),
        (
            "cylinder",
            od.creation.cylinder(radius=1.0, height=2.0, sections=32, device=device),
            o3d.geometry.TriangleMesh.create_cylinder(1.0, 2.0, resolution=32, split=1),
        ),
        (
            "cone",
            od.creation.cone(radius=1.0, height=2.0, sections=32, device=device),
            o3d.geometry.TriangleMesh.create_cone(1.0, 2.0, resolution=32, split=1),
        ),
        (
            "torus",
            od.creation.torus(1.0, 0.25, major_sections=32, minor_sections=32, device=device),
            o3d.geometry.TriangleMesh.create_torus(
                1.0, 0.25, radial_resolution=32, tubular_resolution=32
            ),
        ),
    ):
        mesh_ref = open3d_to_trimesh(mesh_o3d)
        assert vertices_wp.size == len(mesh_ref.vertices), name
        assert faces_wp.size // 3 == len(mesh_ref.faces), name
        mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)
        assert np.isclose(mesh_wp.volume, mesh_ref.volume, rtol=1e-4), name
        assert np.isclose(mesh_wp.area, mesh_ref.area, rtol=1e-4), name


@pytest.mark.parity("box", "meshlib")
@pytest.mark.parity("cylinder", "meshlib")
@pytest.mark.parity("cone", "meshlib")
@pytest.mark.parity("torus", "meshlib")
def test_primitives_match_meshlib(device: str) -> None:
    """
    Class A on the counts and Class B on the invariants -- and the four agree **exactly**.

    Stronger than the plan for this pairing predicted, which was a rigid-motion comparison after
    scaling: measured on all four, the vertex and face counts, the enclosed volume, the surface
    area *and* the bounding-box **extent** match to float32. The extent is asserted because volume
    and area are both invariant under a rotation and would not notice a differently oriented cone.

    Where the two do differ is *where the origin sits*, and only for one of the four: MeshLib's
    ``makeCylinder`` is **base-anchored** (z from 0 to the length) where ordito's is centred on
    z = 0, while ``makeCone`` is base-anchored on **both** sides and the box and torus are centred
    on both. That is a convention rather than a disagreement, so it is pinned per primitive below
    rather than absorbed into a tolerance.

    Two more parameter conventions have to be crossed, both silent if got wrong. ``makeCube`` takes
    a ``size`` and a **base corner** rather than a centre, so a centred box needs
    ``base = -size / 2``; and ``makeCylinder`` / ``makeCone`` default to a radius of **0.1**, not 1,
    so a call that omits it builds something ten times too thin rather than failing.

    ``uv_sphere`` has its own test below rather than a row here, because it needs a parameter
    mapping first: at the same nominal resolution the two tessellate differently -- 450 vertices and
    896 faces against MeshLib's 258 and 512 at ``16 x 16`` -- and deriving the mapping is what that
    test is for.
    """
    # Which primitives share an origin, and which only share a shape.
    centred_on_both = {"box", "cone", "torus"}
    for name, (vertices_wp, faces_wp), mesh_ml in (
        (
            "box",
            od.creation.box(extents=(1.0, 2.0, 3.0), device=device),
            mm.makeCube(mm.Vector3f(1.0, 2.0, 3.0), mm.Vector3f(-0.5, -1.0, -1.5)),
        ),
        (
            "cylinder",
            od.creation.cylinder(radius=0.5, height=2.0, sections=16, device=device),
            mm.makeCylinder(0.5, 2.0, 16),
        ),
        (
            "cone",
            od.creation.cone(radius=0.5, height=2.0, sections=32, device=device),
            mm.makeCone(0.5, 2.0, 32),
        ),
        (
            "torus",
            od.creation.torus(1.0, 0.3, major_sections=16, minor_sections=16, device=device),
            mm.makeTorus(1.0, 0.3, 16, 16),
        ),
    ):
        mesh_ref = meshlib_to_trimesh(mesh_ml)
        mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)
        assert mesh_ref.faces.shape[0] > 0, name  # non-vacuity: the reference built something
        assert vertices_wp.size == mesh_ref.vertices.shape[0], name
        assert faces_wp.size // 3 == mesh_ref.faces.shape[0], name
        assert np.isclose(mesh_wp.volume, mesh_ref.volume, rtol=1e-4), name
        assert np.isclose(mesh_wp.area, mesh_ref.area, rtol=1e-4), name
        extent_wp = mesh_wp.bounds[1] - mesh_wp.bounds[0]
        assert np.allclose(extent_wp, mesh_ref.bounds[1] - mesh_ref.bounds[0], atol=1e-5), name
        if name in centred_on_both:
            assert np.allclose(mesh_wp.bounds, mesh_ref.bounds, atol=1e-5), name
        else:  # the cylinder, and the whole of the difference: MeshLib bases it at z = 0
            assert np.isclose(mesh_wp.bounds[0, 2], -mesh_ref.bounds[1, 2] / 2.0, atol=1e-5), name
            assert np.isclose(mesh_ref.bounds[0, 2], 0.0, atol=1e-5), name


@pytest.mark.parametrize("sections", [16, 32, 64])
@pytest.mark.parity("uv_sphere", "meshlib")
def test_uv_sphere_matches_meshlib(device: str, sections: int) -> None:
    """
    Class B (a named parameter mapping), and then the *same mesh* -- vertex for vertex.

    ``makeUVSphere``'s ``verticalResolution`` counts interior latitude **rings** where
    ``uv_sphere``'s ``count[0]`` counts profile points, poles included, while its
    ``horisontalResolution`` is the section count, exactly as ``count[1]`` is. So the mapping is
    ``makeUVSphere(r, h, v) == uv_sphere(radius=r, count=(v + 2, h))``, which the row here inverts
    to hold ``sections`` fixed.

    At that pairing the two agree far past a count check: measured at 16 / 32 / 64 sections, the
    vertex and face counts are equal, the areas and volumes agree to **1.5e-08 relative**, and a
    nearest-neighbour match between the two vertex sets is a **bijection** whose worst displacement
    is **4.7e-07** -- the float32 floor, since MeshLib stores points in float32 too. The positions
    are matched through a KD-tree rather than sorted, because the two emit their rings in different
    orders and a ``lexsort`` on float coordinates is not reliable at ties.

    Contrast the open3d pairing below, which needs a *different* mapping (``2 * r`` and ``r``)
    and is only equal in the counts -- its latitude rings sit elsewhere, so its volume differs by up
    to 1.22%. Two references, two mappings, and only one of them is the same mesh; that is worth
    pinning in both directions so neither mapping drifts onto the other.
    """
    vertices_wp, faces_wp = od.creation.uv_sphere(
        radius=1.0, count=(2 * sections, sections), device=device
    )
    mesh_ref = meshlib_to_trimesh(mm.makeUVSphere(1.0, sections, 2 * sections - 2))

    assert len(mesh_ref.faces) == 2 * sections * (2 * sections - 2) > 0  # non-vacuity
    assert vertices_wp.size == len(mesh_ref.vertices)
    assert faces_wp.size // 3 == len(mesh_ref.faces)

    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)
    assert np.isclose(mesh_wp.area, mesh_ref.area, rtol=1e-6)
    assert np.isclose(mesh_wp.volume, mesh_ref.volume, rtol=1e-6)

    # The same vertex set, matched by proximity: a bijection, at the float32 floor.
    distance_np, index_np = cKDTree(mesh_ref.vertices).query(mesh_wp.vertices, k=1)
    assert len(set(np.asarray(index_np).tolist())) == len(np.asarray(index_np))
    assert np.max(distance_np) < 1e-5


@pytest.mark.parity("revolve", "meshlib")
def test_revolve_matches_meshlib(device: str) -> None:
    """
    Class A, exactly: ``makeSolidOfRevolution`` sweeps the same profile into the same mesh.

    Same vertex count, face count, area and bounding box on a four-point profile at 16 sections --
    49 vertices, 80 faces, area 5.0059, box ``[-0.5, -0.5, 0] .. [0.5, 0.5, 2]``. Both revolve about
    ``+z`` with the profile in the ``(radius, height)`` plane and both close the seam, which is what
    the vertex count pins: a sweep that duplicated the seam ring would give 52.

    The reference is the only one in this module for ``revolve`` -- trimesh's ``revolve`` is
    ``creation.revolve``'s own model but is not benchmarked here, and open3d and pyvista have no
    solid-of-revolution generator -- so this is the pairing that says the sweep is right rather than
    merely self-consistent.
    """
    profile_np = np.array([[0.5, 0.0], [0.5, 1.0], [0.3, 1.5], [0.0, 2.0]])
    profile_wp = points_to_warp_uv(profile_np, device)
    vertices_wp, faces_wp = od.creation.revolve(profile_wp, sections=16)
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)

    profile_ml = mm.std_vector_Vector2_float()
    for point_np in profile_np:
        profile_ml.append(mm.Vector2f(float(point_np[0]), float(point_np[1])))
    mesh_ref = meshlib_to_trimesh(mm.makeSolidOfRevolution(profile_ml, 16))

    assert mesh_ref.faces.shape[0] > 0  # non-vacuity
    assert vertices_wp.size == mesh_ref.vertices.shape[0]
    assert faces_wp.size // 3 == mesh_ref.faces.shape[0]
    assert np.isclose(mesh_wp.area, mesh_ref.area, rtol=1e-4)
    assert np.allclose(mesh_wp.bounds, mesh_ref.bounds, atol=1e-5)


@pytest.mark.parametrize("sections", [16, 32, 64])
@pytest.mark.parity("uv_sphere", "open3d")
def test_uv_sphere_matches_open3d(device: str, sections: int) -> None:
    """
    Class B (a named index mapping): the two UV spheres tessellate differently than expected.

    ``create_sphere(resolution=r)`` is neither ``count=(r, r)`` nor ``count=(32, r)``: measured
    exactly at r = 16, 32, 64, 128 and 256, it equals ``uv_sphere(count=(2 * r, r))`` in both
    vertex and face count. ``benchmarks/test_creation.py`` paired it with ``count=(32, r)`` and so
    timed a linear sweep against a quadratic one -- 15 360 faces against 65 024 at ``sections=256``.
    Pinning the count mapping, so that fix cannot drift back, is the substance of this test.

    The two are **not** the same mesh, and the assertions say so rather than pretending otherwise.
    Both are exact unit spheres -- every vertex sits at radius 1 to float32 -- but they distribute
    their latitude rings differently, so at equal tessellation the enclosed volumes differ by
    **1.22 / 0.30 / 0.08%** at ``sections`` 16 / 32 / 64. That gap is a discretization difference,
    not an error in either, and the shape of it is the check worth making: it must *shrink* as the
    tessellation refines, and both must converge on ``4 pi / 3``. A sphere generator with a wrong
    ring placement would hold a constant offset instead.
    """
    vertices_wp, faces_wp = od.creation.uv_sphere(
        radius=1.0, count=(2 * sections, sections), device=device
    )
    mesh_ref = open3d_to_trimesh(o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=sections))

    assert vertices_wp.size == len(mesh_ref.vertices)
    assert faces_wp.size // 3 == len(mesh_ref.faces)

    # Both are unit spheres: every vertex on the surface, not merely near it.
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.linalg.norm(mesh_ref.vertices, axis=1), 1.0, rtol=1e-5, atol=1e-5)

    # Both inscribe the true sphere and converge on it; the tolerance tracks the tessellation.
    exact_volume = 4.0 / 3.0 * np.pi
    tolerance = {16: 0.03, 32: 0.01, 64: 0.005}[sections]
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)
    for volume in (mesh_wp.volume, mesh_ref.volume):
        assert volume < exact_volume
        assert abs(volume - exact_volume) / exact_volume < tolerance


# --- pymeshlab primitives ---------------------------------------------------------------


def _pymeshlab_mesh(meshset_pml: ml.MeshSet) -> tm.Trimesh:
    """Read MeshLab's current mesh back as a `trimesh.Trimesh`, unprocessed."""
    mesh_pml = meshset_pml.current_mesh()
    return tm.Trimesh(
        np.asarray(mesh_pml.vertex_matrix(), dtype=np.float64),
        np.asarray(mesh_pml.face_matrix()),
        process=False,
    )


def _sorted_edge_lengths(mesh_tm: tm.Trimesh) -> np.ndarray:
    """
    Sorted multiset of unique-edge lengths: a rotation- and ordering-invariant mesh fingerprint.

    Two generators of the same primitive that seat their seam at a different angle produce the same
    multiset while sharing no vertex position, so this pins the tessellation where a
    position-by-position compare cannot.
    """
    edges_np = mesh_tm.edges_unique
    return np.sort(
        np.linalg.norm(mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1)
    )


@pytest.mark.parity("box", "pymeshlab")
@pytest.mark.parity("icosphere", "pymeshlab")
@pytest.mark.parity("cone", "pymeshlab")
@pytest.mark.parity("torus", "pymeshlab")
def test_primitives_match_pymeshlab(device: str) -> None:
    """
    Four primitives against MeshLab's ``create_*`` generators, which agree exactly on the shape.

    Class B throughout, with one named transform per primitive and one shared one. Each
    ``create_*`` **pushes a new mesh onto the MeshSet** rather than returning it, so every call gets
    a fresh set and the answer is read off ``current_mesh()``.

    Two of the four are the *same mesh*: ``create_cube`` and ``create_torus`` match ordito's vertex
    set as a bijection (to 0 and 4.4e-07). The other two are the same mesh under a rotation about
    the axis -- MeshLab seats the icosahedron base and the cone seam at a different angle, so the
    vertex positions differ by up to 0.163 and 1.43 while every measure agrees -- and
    ``create_cone`` is additionally *centred* on the origin where ordito's cone stands on ``z =
    0``, hence the ``h / 2`` translation. Those two are therefore compared through
    [`_sorted_edge_lengths`][tests.test_creation._sorted_edge_lengths], which is invariant to both,
    matching to 8.0e-08 and 4.0e-07.

    ``create_cube`` takes one ``size``, so the box is compared as a unit cube rather than the
    ``1x2x3`` box the trimesh and open3d rows use; ``create_sphere`` caps ``subdiv`` at 8.
    """
    cube_pml = ml.MeshSet()
    cube_pml.create_cube(size=1.0)
    sphere_pml = ml.MeshSet()
    sphere_pml.create_sphere(radius=1.0, subdiv=2)
    cone_pml = ml.MeshSet()
    cone_pml.create_cone(r0=1.0, r1=0.0, h=2.0, subdiv=32)
    torus_pml = ml.MeshSet()
    torus_pml.create_torus(hradius=1.0, vradius=0.25, hsubdiv=32, vsubdiv=32)

    for name, (vertices_wp, faces_wp), mesh_ref, exact_vertices in (
        ("box", od.creation.box(extents=(1.0, 1.0, 1.0), device=device), cube_pml, True),
        ("icosphere", od.creation.icosphere(subdivisions=2, device=device), sphere_pml, False),
        (
            "cone",
            od.creation.cone(radius=1.0, height=2.0, sections=32, device=device),
            cone_pml,
            False,
        ),
        (
            "torus",
            od.creation.torus(1.0, 0.25, major_sections=32, minor_sections=32, device=device),
            torus_pml,
            True,
        ),
    ):
        mesh_pml = _pymeshlab_mesh(mesh_ref)
        if name == "cone":  # MeshLab centres the cone on the origin; ordito bases it at z = 0.
            mesh_pml.vertices = mesh_pml.vertices + np.array([0.0, 0.0, 1.0])
        mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)

        assert vertices_wp.size == mesh_pml.vertices.shape[0], name
        assert faces_wp.size // 3 == mesh_pml.faces.shape[0], name
        assert np.isclose(mesh_wp.volume, mesh_pml.volume, rtol=1e-4), name
        assert np.isclose(mesh_wp.area, mesh_pml.area, rtol=1e-4), name
        assert np.allclose(mesh_wp.bounds, mesh_pml.bounds, rtol=1e-5, atol=1e-5), name
        assert np.allclose(
            _sorted_edge_lengths(mesh_wp), _sorted_edge_lengths(mesh_pml), rtol=1e-5, atol=1e-5
        ), name
        if exact_vertices:
            distance_np, match_np = cKDTree(mesh_pml.vertices).query(mesh_wp.vertices)
            assert np.max(distance_np) < 1e-5, name
            assert len(set(np.asarray(match_np).tolist())) == match_np.size, name


# --- table primitives -------------------------------------------------------------------


@pytest.mark.parity("box", "trimesh")
@pytest.mark.parametrize("case", ["default", "extents", "bounds", "transform"])
def test_box(device: str, case: str) -> None:
    """
    Class A: vertices, faces and bounds against ``trimesh.creation.box``, per way of sizing it.

    The default and a non-cube box, because the extent scaling is applied after the unit table
    -- a transposed scale passes the first and fails the second -- then explicit bounds and a
    rotation with a translation, each handed to both sides.
    """
    if case == "default":
        vertices_wp, faces_wp = od.creation.box(device=device)
        box_tm = tm.creation.box()
    elif case == "extents":
        vertices_wp, faces_wp = od.creation.box(extents=(1.0, 2.0, 3.0), device=device)
        box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0])
    elif case == "bounds":
        bounds_np = np.array([[-1.0, 0.0, 2.0], [3.0, 1.0, 5.0]])
        vertices_wp, faces_wp = od.creation.box(
            bounds=cast("Sequence[Sequence[float]]", bounds_np), device=device
        )
        box_tm = tm.creation.box(bounds=bounds_np)
    else:
        matrix_np = np.asarray(
            tm.transformations.rotation_matrix(np.deg2rad(37.0), [1.0, 2.0, 3.0])
        )
        matrix_np[:3, 3] = np.array([1.0, -2.0, 0.5])
        vertices_wp, faces_wp = od.creation.box(
            extents=(1.0, 2.0, 3.0), transform=_mat44(matrix_np), device=device
        )
        box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0], transform=matrix_np)
    _assert_same_vertices_and_faces(vertices_wp, faces_wp, box_tm)
    assert np.allclose(
        warp_to_trimesh(vertices_wp, faces_wp).bounds, box_tm.bounds, rtol=1e-5, atol=1e-5
    )


def test_box_mirror_transform_keeps_outward_winding(device: str) -> None:
    mirror_np = np.eye(4)
    mirror_np[0, 0] = -1.0
    vertices_wp, faces_wp = od.creation.box(
        extents=(1.0, 2.0, 3.0), transform=_mat44(mirror_np), device=device
    )
    # A negative-determinant transform reverses winding, so the faces have to be flipped back.
    assert warp_to_trimesh(vertices_wp, faces_wp).volume > 0.0
    assert od.validation.is_volume(vertices_wp, faces_wp)


def test_box_invalid(device: str) -> None:
    bounds_np = np.zeros((2, 3))
    with pytest.raises(ValueError, match="bounds overrides"):
        od.creation.box(
            extents=(1.0, 1.0, 1.0),
            bounds=cast("Sequence[Sequence[float]]", bounds_np),
            device=device,
        )
    with pytest.raises(ValueError, match="bounds must be"):
        od.creation.box(bounds=cast("Sequence[Sequence[float]]", np.zeros((3, 3))), device=device)
    with pytest.raises(ValueError, match="extents must be"):
        od.creation.box(extents=np.zeros(4), device=device)  # pyright: ignore[reportArgumentType]


@pytest.mark.parity("platonic_solids", "trimesh", "igl")
def test_icosahedron(device: str) -> None:
    """
    Class A against trimesh; Class B against igl, whose icosahedron sits in a **rotated frame**.

    ``igl.icosahedron`` is libigl's only Platonic generator, and it is the same solid in a different
    orientation: ordito and trimesh use the ``(0, ±1, ±φ)`` form (every coordinate ±0.851 or 0)
    where igl puts a vertex at the pole (coordinates ±0.894 / ±0.447 / ±1). Positions therefore
    cannot be matched at all, and a comparison that tried would be reporting the frame.

    The named transform is *compare the rigid-motion invariants*: counts, unit circumradius, the
    single edge length shared by all 30 edges, surface area and enclosed volume. Together those pin
    the solid uniquely up to a rotation, which is exactly the equivalence igl's output sits in. A
    wrong vertex table -- the bug class this excludes -- moves the edge-length spread off zero or
    the area off its exact value, and both are asserted.
    """
    vertices_wp, faces_wp = od.creation.icosahedron(device=device)
    _assert_same_vertices_and_faces(vertices_wp, faces_wp, tm.creation.icosahedron())

    vertices_igl, faces_igl = map(np.asarray, igl.icosahedron())
    mesh_igl = tm.Trimesh(vertices_igl, faces_igl, process=False)
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)

    assert vertices_igl.shape == (12, 3)
    assert faces_igl.shape == (20, 3)
    assert np.allclose(np.linalg.norm(vertices_igl, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    # One edge length, the same on both sides (a regular icosahedron on the unit sphere).
    edges_igl = np.linalg.norm(np.diff(vertices_igl[mesh_igl.edges_unique], axis=1), axis=2).ravel()
    edges_wp = np.linalg.norm(
        np.diff(mesh_wp.vertices[mesh_wp.edges_unique], axis=1), axis=2
    ).ravel()
    assert np.allclose(edges_igl, edges_igl[0], rtol=1e-5, atol=1e-5)
    assert np.allclose(edges_wp.mean(), edges_igl.mean(), rtol=1e-5, atol=1e-5)
    assert np.isclose(mesh_wp.area, mesh_igl.area, rtol=1e-5)
    assert np.isclose(abs(mesh_wp.volume), abs(mesh_igl.volume), rtol=1e-5)


@pytest.mark.parametrize(
    ("builder", "filter_name", "n_vertices", "n_faces"),
    [
        ("tetrahedron", "create_tetrahedron", 4, 4),
        ("octahedron", "create_octahedron", 6, 8),
        ("dodecahedron", "create_dodecahedron", 20, 36),
    ],
)
@pytest.mark.parity("platonic_solids", "pymeshlab")
def test_platonic_solids_match_pymeshlab(
    device: str, builder: str, filter_name: str, n_vertices: int, n_faces: int
) -> None:
    """
    Class B (scale to unit circumradius): the three vertex tables came from MeshLab.

    trimesh has no tetrahedron / octahedron / dodecahedron, and libigl's only Platonic generator is
    ``igl.icosahedron`` (compared in
    [`test_icosahedron`][tests.test_creation.test_icosahedron]), so pymeshlab is the only reference
    for these three.
    """
    vertices_wp, faces_wp = getattr(od.creation, builder)(device=device)
    assert vertices_wp.size == n_vertices
    assert faces_wp.size // 3 == n_faces
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)
    _assert_closed(vertices_wp, faces_wp)

    meshset_pml = ml.MeshSet()
    getattr(meshset_pml, filter_name)()
    vertices_pml = meshset_pml.current_mesh().vertex_matrix()
    vertices_pml = vertices_pml / np.linalg.norm(vertices_pml, axis=1, keepdims=True)
    _assert_same_vertices_and_faces(
        vertices_wp,
        faces_wp,
        tm.Trimesh(vertices_pml, meshset_pml.current_mesh().face_matrix(), process=False),
    )


@pytest.mark.parametrize(
    ("builder", "creator_name"),
    [
        ("tetrahedron", "create_tetrahedron"),
        ("octahedron", "create_octahedron"),
        ("icosahedron", "create_icosahedron"),
    ],
)
@pytest.mark.parity("platonic_solids", "open3d")
def test_platonic_solids_match_open3d(device: str, builder: str, creator_name: str) -> None:
    """
    Class B via rigid-motion invariants, the ``test_icosahedron`` transform against a third table.

    Probed before this test was written: open3d's octahedron matches ordito's vertex set exactly,
    but its tetrahedron sits in a rotated frame (nearest-vertex distance 0.92 after scaling) and
    its icosahedron is the raw ``(0, ±1, ±phi)`` table at circumradius 1.902 in yet another
    orientation -- so positions cannot be compared across the family and the invariants are the
    honest common ground: counts, one shared edge length, surface area and enclosed volume, all
    after scaling open3d's solid to ordito's unit circumradius. open3d has no dodecahedron, which
    is why the parametrization stops at three where the pymeshlab test above has four.
    """
    vertices_wp, faces_wp = getattr(od.creation, builder)(device=device)
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)

    mesh_o3d = getattr(o3d.geometry.TriangleMesh, creator_name)()
    vertices_o3d = np.asarray(mesh_o3d.vertices)
    radii_o3d = np.linalg.norm(vertices_o3d, axis=1)
    assert np.allclose(radii_o3d, radii_o3d[0], rtol=1e-5)  # centred: circumradius well-defined
    mesh_o3d_unit = tm.Trimesh(
        vertices_o3d / radii_o3d[0], np.asarray(mesh_o3d.triangles), process=False
    )

    assert len(mesh_o3d_unit.vertices) == len(mesh_wp.vertices)
    assert len(mesh_o3d_unit.faces) == len(mesh_wp.faces)
    edges_o3d = np.linalg.norm(
        np.diff(mesh_o3d_unit.vertices[mesh_o3d_unit.edges_unique], axis=1), axis=2
    ).ravel()
    edges_wp = np.linalg.norm(
        np.diff(mesh_wp.vertices[mesh_wp.edges_unique], axis=1), axis=2
    ).ravel()
    assert np.allclose(edges_o3d, edges_o3d[0], rtol=1e-5, atol=1e-5)
    assert np.allclose(edges_wp.mean(), edges_o3d.mean(), rtol=1e-5, atol=1e-5)
    assert np.isclose(mesh_wp.area, mesh_o3d_unit.area, rtol=1e-5)
    assert np.isclose(abs(mesh_wp.volume), abs(mesh_o3d_unit.volume), rtol=1e-5)


@pytest.mark.parametrize("count", [(2, 2), (3, 7), (10, 10)])
def test_grid(device: str, count: tuple[int, int]) -> None:
    """
    Not a library comparison: the lattice's counts, bounds and area, which are closed-form.

    MeshLab supplies the elementwise comparison in [`test_grid_matches_pymeshlab`]. This is the
    test that pins the *shape* of the answer -- a transposed ``count`` gives the same vertex
    total and a different face total, which only the face formula catches.
    """
    vertices_wp, faces_wp = od.creation.grid(count=count, extents=(2.0, 3.0), device=device)
    assert vertices_wp.size == count[0] * count[1]
    assert faces_wp.size // 3 == 2 * (count[0] - 1) * (count[1] - 1)

    mesh_tm = warp_to_trimesh(vertices_wp, faces_wp)
    assert np.allclose(mesh_tm.bounds, [[-1.0, -1.5, 0.0], [1.0, 1.5, 0.0]], rtol=1e-5, atol=1e-5)
    assert np.isclose(mesh_tm.area, 6.0, rtol=1e-5)
    # Flat, wound outward along +Z, and one boundary loop around the rim.
    assert np.allclose(mesh_tm.face_normals, [0.0, 0.0, 1.0], rtol=1e-5, atol=1e-5)
    assert od.validation.is_winding_consistent(faces_wp)
    assert len(od.boundary.boundary_loops(vertices_wp, faces_wp)) == 1


@pytest.mark.parity("grid", "igl")
def test_grid_matches_igl(device: str) -> None:
    """
    Class B: ``igl.triangulated_grid`` is the same lattice in 2D over the unit square.

    Two named transforms, both exact: the reference's ``(n, 2)`` vertices gain a zero third column,
    and ordito is asked for the matching patch (``extents=(1, 1)``, ``center=False``) so the two
    cover the same square. The **vertex sets then agree exactly**, which is the assert.

    **The two triangulate each cell along the opposite diagonal**, and that is measured rather than
    assumed: ordito's first two face centroids are ``(0.222, 0.111)`` and ``(0.111, 0.222)`` where
    igl's are ``(0.111, 0.111)`` and ``(0.222, 0.222)`` on a 4x4 grid. Both are valid grids, so a
    face-centroid comparison is *not* available here -- it fails by the cell size -- and the
    triangulation is instead pinned by the invariants both must satisfy: the same face count, the
    same total area, and every triangle right-angled with legs one cell wide.
    """
    count = 10
    vertices_wp, faces_wp = od.creation.grid(
        count=(count, count), extents=(1.0, 1.0), center=False, device=device
    )
    vertices_igl, faces_igl = map(np.asarray, igl.triangulated_grid(count, count))

    assert faces_wp.size // 3 == faces_igl.shape[0]
    padded_igl = np.column_stack([vertices_igl, np.zeros(vertices_igl.shape[0])])
    assert np.allclose(
        np.sort(vertices_wp.numpy().astype(np.float64), axis=0),
        np.sort(padded_igl, axis=0),
        rtol=1e-5,
        atol=1e-6,
    )

    # The diagonals differ, so compare what both triangulations must satisfy.
    mesh_wp, mesh_igl = (
        warp_to_trimesh(vertices_wp, faces_wp),
        tm.Trimesh(padded_igl, faces_igl, process=False),
    )
    assert np.isclose(mesh_wp.area, 1.0, rtol=1e-5)
    assert np.isclose(mesh_igl.area, mesh_wp.area, rtol=1e-5)
    cell = 1.0 / (count - 1)
    for mesh in (mesh_wp, mesh_igl):
        sides = np.sort(
            np.linalg.norm(np.diff(mesh.vertices[mesh.faces[:, [0, 1, 2, 0]]], axis=1), axis=2),
            axis=1,
        )
        assert np.allclose(sides[:, :2], cell, rtol=1e-5, atol=1e-6)
        assert np.allclose(sides[:, 2], cell * np.sqrt(2.0), rtol=1e-5, atol=1e-6)


@pytest.mark.parity("grid", "pymeshlab")
def test_grid_matches_pymeshlab(device: str) -> None:
    """Class B (recentre): MeshLab's ``create_grid`` is the same lattice, uncentered."""
    vertices_wp, faces_wp = od.creation.grid(
        count=(10, 8), extents=(0.3, 0.5), center=False, device=device
    )
    meshset_pml = ml.MeshSet()
    meshset_pml.create_grid(numvertx=10, numverty=8, absscalex=0.3, absscaley=0.5, center=False)
    mesh_pml = meshset_pml.current_mesh()
    assert vertices_wp.size == mesh_pml.vertex_number()
    assert faces_wp.size // 3 == mesh_pml.face_number()
    # MeshLab lays the patch out along -X; compare the shape after mirroring that back.
    vertices_pml = mesh_pml.vertex_matrix() * np.array([-1.0, 1.0, 1.0])
    assert np.allclose(
        np.sort(vertices_wp.numpy().astype(np.float64), axis=0),
        np.sort(vertices_pml, axis=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grid_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="count must be at least 2"):
        od.creation.grid(count=(1, 4), device=device)
    with pytest.raises(ValueError, match="extents must be non-negative"):
        od.creation.grid(extents=(-1.0, 1.0), device=device)


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3, 4])
def test_sphere_cap(device: str, subdivisions: int) -> None:
    angle, radius = np.deg2rad(35.0), 1.5
    vertices_wp, faces_wp = od.creation.sphere_cap(
        angle=angle, subdivisions=subdivisions, radius=radius, device=device
    )
    n_rings = 2**subdivisions
    assert vertices_wp.size == 1 + 3 * n_rings * (n_rings + 1)
    assert faces_wp.size // 3 == 6 * n_rings**2

    # Every vertex on the sphere, the apex at the pole, and the rim at exactly ``angle``.
    vertices_np = vertices_wp.numpy().astype(np.float64)
    assert np.allclose(np.linalg.norm(vertices_np, axis=1), radius, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertices_np[0], [0.0, 0.0, radius], rtol=1e-5, atol=1e-5)
    polar_np = np.arccos(np.clip(vertices_np[:, 2] / radius, -1.0, 1.0))
    assert np.isclose(polar_np.max(), angle, rtol=1e-5, atol=1e-5)

    # An open disc: one boundary loop, of exactly the rim's ``6 * n_rings`` vertices.
    loops = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops) == 1
    assert loops[0].size == 6 * n_rings
    assert od.validation.is_winding_consistent(faces_wp)
    # Wound outward: the area-weighted normal of a cap around +Z points along +Z.
    normals_wp, areas_wp = od.triangles.face_normals_and_areas(vertices_wp, faces_wp)
    assert (normals_wp.numpy() * areas_wp.numpy()[:, None]).sum(axis=0)[2] > 0.0

    # Area of a spherical cap of half-angle ``angle``, approached from below by the inscribed mesh.
    exact_area = 2.0 * np.pi * radius**2 * (1.0 - np.cos(angle))
    assert warp_to_trimesh(vertices_wp, faces_wp).area <= exact_area * (1.0 + 1e-6)
    if subdivisions >= 3:
        assert warp_to_trimesh(vertices_wp, faces_wp).area > exact_area * 0.99


@pytest.mark.parity("sphere_cap", "pymeshlab")
def test_sphere_cap_matches_pymeshlab_size(device: str) -> None:
    """Class B (halve the angle): ``create_sphere_cap`` is the same lattice, by aperture."""
    vertices_wp, faces_wp = od.creation.sphere_cap(
        angle=np.deg2rad(30.0), subdivisions=3, device=device
    )
    meshset_pml = ml.MeshSet()
    meshset_pml.create_sphere_cap(angle=60.0, subdiv=3)
    mesh_pml = meshset_pml.current_mesh()
    assert vertices_wp.size == mesh_pml.vertex_number()
    assert faces_wp.size // 3 == mesh_pml.face_number()
    # MeshLab puts the rim plane at z = 0 rather than centering the sphere; shift it back and the
    # two caps are the same surface (its lattice rings are rotated in azimuth, so compare radii).
    vertices_pml = mesh_pml.vertex_matrix() + np.array([0.0, 0.0, np.cos(np.deg2rad(30.0))])
    assert np.allclose(
        np.sort(np.linalg.norm(vertices_wp.numpy().astype(np.float64), axis=1)),
        np.sort(np.linalg.norm(vertices_pml, axis=1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sphere_cap_invalid(device: str) -> None:
    with pytest.raises(ValueError, match=r"angle must be in \(0, pi\)"):
        od.creation.sphere_cap(angle=0.0, device=device)
    with pytest.raises(ValueError, match=r"angle must be in \(0, pi\)"):
        od.creation.sphere_cap(angle=np.pi, device=device)
    with pytest.raises(ValueError, match="subdivisions must be non-negative"):
        od.creation.sphere_cap(subdivisions=-1, device=device)


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3, 4, 5])
@pytest.mark.parity("icosphere", "trimesh")
def test_icosphere(device: str, subdivisions: int) -> None:
    """
    Class A: the same vertex set and the same face set as trimesh, at every refinement level.

    This is the gate on the closed-form vertex numbering, which is why it runs past
    ``subdivisions=2``: a base edge carries ``2 ** subdivisions - 1`` interior points, so at levels
    0 and 1 there are none or one and an edge walked in the *wrong direction* still lands on the
    same index. From level 2 up, any error in the shared numbering shows as a different face set.

    The closed-form numbering also has to be crack-free: a point on a base edge gets the same index
    from both faces holding it. A seam is exactly what that failing looks like, and it is invisible
    in a vertex *count* -- the count is closed-form too -- so the mesh is asserted closed and its
    vertices distinct.
    """
    vertices_wp, faces_wp = od.creation.icosphere(subdivisions=subdivisions, device=device)
    _assert_same_vertices_and_faces(
        vertices_wp, faces_wp, tm.creation.icosphere(subdivisions=subdivisions)
    )
    assert faces_wp.size // 3 == 20 * 4**subdivisions
    assert vertices_wp.size == 10 * 4**subdivisions + 2
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)
    _assert_closed(vertices_wp, faces_wp)
    assert vertices_wp.size == len(np.unique(vertices_wp.numpy(), axis=0))


@pytest.mark.parametrize("subdivisions", [0, 1, 2])
@pytest.mark.parity("icosphere", "pytorch3d")
def test_icosphere_matches_pytorch3d(device: str, subdivisions: int) -> None:
    """
    Class B: ``utils.ico_sphere`` is the *same* construction, to its base table's 4 decimal places.

    Not the rotated-frame situation open3d's Platonic solids -- pytorch3d
    starts from the identical ``(+-0.5257, +-0.8507, 0)`` vertex table ordito uses and subdivides
    the same way, so the positions correspond one-to-one and the residual is pytorch3d's table
    being *written* to four decimals: measured 5.8e-05 at level 0 and 5.2e-05 at level 1 by nearest
    vertex, with the sorted pairwise-distance spectrum agreeing to 9.8e-05.

    So the transform is the correspondence, not a gauge fix: a ``cKDTree`` nearest-neighbour match
    plus a bijection check. Counts are asserted first (12/20, 42/80, 162/320), which is what makes
    the bijection meaningful rather than a statement about a subset.
    """
    sphere_p3d = p3d_utils.ico_sphere(subdivisions)
    vertices_p3d, faces_p3d = pytorch3d_to_numpy(sphere_p3d)
    vertices_wp, faces_wp = od.creation.icosphere(subdivisions=subdivisions, device=device)
    vertices_np = vertices_wp.numpy().astype(np.float64)

    assert vertices_p3d.shape[0] == 10 * 4**subdivisions + 2
    assert faces_p3d.shape[0] == 20 * 4**subdivisions
    assert len(vertices_np) == vertices_p3d.shape[0]
    assert faces_wp.size // 3 == faces_p3d.shape[0]

    distances_np, indices_np = cKDTree(vertices_p3d).query(vertices_np)
    assert float(np.max(distances_np)) < 1e-4
    assert np.unique(indices_np).size == len(vertices_np)


@pytest.mark.parametrize("levels_per_launch", [1, 3])
def test_icosphere_launch_schedule_is_value_neutral(
    monkeypatch: pytest.MonkeyPatch, device: str, levels_per_launch: int
) -> None:
    """
    Ordito against ordito: every launch schedule builds the same mesh, bit for bit.

    ``test_icosphere`` carries the trimesh oracle; this pins the schedules to each other. At five
    levels the shipped schedule is one launch descending every level per vertex; one level per
    launch refines exactly as the recursive build does, and three levels then one per launch mixes
    a multi-level launch reading the vertices an earlier one wrote.
    """
    vertices_wp, faces_wp = od.creation.icosphere(subdivisions=5, radius=1.7, device=device)
    monkeypatch.setattr(od.creation, "_ICOSPHERE_LEVELS_PER_LAUNCH", levels_per_launch)
    other_vertices_wp, other_faces_wp = od.creation.icosphere(
        subdivisions=5, radius=1.7, device=device
    )
    assert np.array_equal(
        vertices_wp.numpy().view(np.uint32), other_vertices_wp.numpy().view(np.uint32)
    )
    assert np.array_equal(faces_wp.numpy(), other_faces_wp.numpy())


def test_icosphere_radius(device: str) -> None:
    vertices_wp, faces_wp = od.creation.icosphere(subdivisions=3, radius=2.5, device=device)
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 2.5, rtol=1e-5, atol=1e-5)
    exact_volume = 4.0 / 3.0 * np.pi * 2.5**3
    assert abs(warp_to_trimesh(vertices_wp, faces_wp).volume - exact_volume) / exact_volume < 0.01


# --- revolution primitives --------------------------------------------------------------


@pytest.mark.parity("uv_sphere", "trimesh")
def test_uv_sphere(device: str) -> None:
    """
    Class A on the faces, plus a radius check the reference cannot supply.

    Only the connectivity is compared against trimesh, because
    [`test_uv_sphere_matches_open3d`] documents that the two libraries tessellate differently;
    the radius assert is what pins the positions here.
    """
    _assert_same_faces(*od.creation.uv_sphere(device=device), tm.creation.uv_sphere())
    vertices_wp, _ = od.creation.uv_sphere(radius=3.0, device=device)
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 3.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("uv_sphere", "trimesh")
def test_uv_sphere_explicit_count(device: str) -> None:
    """
    Class B (a named parameter mapping): ``count[1]`` is the section count, given or defaulted.

    trimesh doubles an explicit ``count[1]`` and not its default, so its ``count=[16, 16]`` is
    this module's ``count=(16, 32)``. The second half pins that the default is not special: an
    explicit ``(32, 64)`` is the default mesh.
    """
    vertices_wp, faces_wp = od.creation.uv_sphere(count=(16, 32), device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.uv_sphere(count=[16, 16]))
    assert vertices_wp.size == 14 * 32 + 2
    default_wp, _ = od.creation.uv_sphere(device=device)
    explicit_wp, _ = od.creation.uv_sphere(count=(32, 64), device=device)
    assert np.array_equal(default_wp.numpy(), explicit_wp.numpy())


def test_uv_sphere_does_not_mutate_the_caller_s_count(device: str) -> None:
    """
    Ordito against ordito: the odd-to-even rounding must not write through the caller's array.

    ``np.asanyarray`` does not copy an ``int64`` ndarray, so an in-place ``counts += counts % 2``
    reaches back into the caller's own buffer. The sibling ``capsule`` never had the defect, and
    ``trimesh`` uses ``np.array(...)``, which always copies -- so this pins the one spelling that
    differed rather than a behaviour any reference defines.
    """
    count_np = np.array([31, 63], dtype=np.int64)
    # An array rather than the annotated tuple, deliberately: it is the input that could be written.
    od.creation.uv_sphere(count=count_np, device=device)  # pyright: ignore[reportArgumentType]
    assert np.array_equal(count_np, [31, 63])
    # A tuple cannot be written through, so it is the control: both spellings must round the same.
    from_tuple_wp, _ = od.creation.uv_sphere(count=(31, 63), device=device)
    from_array_wp, _ = od.creation.uv_sphere(count=count_np, device=device)  # pyright: ignore[reportArgumentType]
    assert from_tuple_wp.size == from_array_wp.size


def test_capsule(device: str) -> None:
    vertices_wp, faces_wp = od.creation.capsule(height=2.0, radius=0.5, device=device)
    mesh_tm = tm.creation.capsule(height=2.0, radius=0.5)
    _assert_same_faces(vertices_wp, faces_wp, mesh_tm)
    # Centered on the origin, spanning +-(height / 2 + radius) along Z.
    assert np.allclose(
        warp_to_trimesh(vertices_wp, faces_wp).bounds[:, 2], [-1.5, 1.5], rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize(
    ("name", "kwargs", "expect_faces"),
    [
        ("annulus", {"r_min": 0.5, "r_max": 1.0, "height": 1.0, "sections": 16}, True),
        ("annulus", {"r_min": 0.5, "r_max": 1.0, "height": 1.0, "sections": 2}, True),
        ("annulus", {"r_min": 1e-5, "r_max": 2e-5, "height": 1.0, "sections": 8}, True),
        (
            "torus",
            {"major_radius": 1.0, "minor_radius": 0.3, "major_sections": 16, "minor_sections": 12},
            True,
        ),
        (
            "torus",
            {"major_radius": 1.0, "minor_radius": 0.3, "major_sections": 2, "minor_sections": 5},
            True,
        ),
        ("uv_sphere", {"radius": 1.0, "count": (8, 12)}, True),
        ("uv_sphere", {"radius": 1e-5, "count": (8, 16)}, False),
        ("capsule", {"radius": 0.5, "height": 2.0, "count": (8, 6)}, True),
        ("cone", {"radius": 1.0, "height": 2.0, "sections": 16}, True),
        ("cone", {"radius": 1.0, "height": 2.0, "sections": 2}, True),
        ("cylinder", {"radius": 1.0, "height": 2.0, "sections": 16}, True),
        ("cylinder", {"radius": 1.0, "height": 2.0, "sections": 1}, False),
    ],
)
def test_solids_of_revolution_agree_with_the_general_engine(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, object],
    expect_faces: bool,
) -> None:
    """
    Ordito against ordito: the closed-form path against ``revolve``, which carries the oracle.

    Every solid here has two implementations — a single closed-form launch, and the general
    profile-revolving engine the reference comparisons elsewhere in this file are written against.
    They must be **bit-identical**, not merely close: the closed-form path exists only to remove
    host work, and any drift in the last float32 bit would be a second definition of the geometry
    rather than a faster route to the same one.

    The parameters deliberately straddle the gate. ``sections`` of 1 and 2 collapse the triangles
    touching the axis, and a radius of ``1e-5`` puts whole quads under the absolute area tolerance;
    both make ``creation._revolve_regular`` decline and fall back, which is why they are here — a
    parametrization that only covered ordinary shapes would never execute the fallback at all.

    ``expect_faces`` says which of those degenerate cases legitimately produce *no* faces, so the
    comparison cannot pass by both sides being empty without that being the stated intent.
    """
    builder = getattr(od.creation, name)
    fast_vertices, fast_faces = builder(device=device, **kwargs)

    # Force the general engine for the same call, so both answers come from one process and one
    # device rather than from a remembered table.
    monkeypatch.setattr(od.creation, "_revolve_regular", lambda *a, **k: None)
    slow_vertices, slow_faces = builder(device=device, **kwargs)

    assert (fast_faces.size > 0) == expect_faces
    assert fast_vertices.size > 0
    assert np.array_equal(fast_vertices.numpy(), slow_vertices.numpy())
    assert np.array_equal(fast_faces.numpy(), slow_faces.numpy())


@pytest.mark.parity("cylinder", "trimesh")
def test_cylinder(device: str) -> None:
    """
    Class A on the faces, with the *inscribed prism* volume as the closed-form check.

    The volume reference is deliberately not ``pi r^2 h``: 32 sections truncate the circle, so
    the exact answer is the inscribed prism's and a test against the smooth formula would need
    a loose tolerance that hides real error.
    """
    vertices_wp, faces_wp = od.creation.cylinder(radius=1.0, height=2.0, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.cylinder(radius=1.0, height=2.0))
    # 32 sections truncate the circle, so the volume is the inscribed prism's, not pi * r^2 * h.
    inscribed = 0.5 * 32 * np.sin(2.0 * np.pi / 32) * 2.0
    assert np.isclose(warp_to_trimesh(vertices_wp, faces_wp).volume, inscribed, rtol=1e-4)


def test_cylinder_segment(device: str) -> None:
    """
    Class A on the faces and the bounds, for the arbitrary-axis form.

    The segment is off-axis, so this is the branch where the frame construction matters; the
    bounds comparison is what catches a rotation applied in the wrong order.
    """
    segment_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_wp, faces_wp = od.creation.cylinder(
        radius=0.5, segment=cast("Sequence[Sequence[float]]", segment_np), device=device
    )
    mesh_tm = tm.creation.cylinder(radius=0.5, segment=segment_np)
    _assert_same_faces(vertices_wp, faces_wp, mesh_tm)
    assert np.allclose(
        warp_to_trimesh(vertices_wp, faces_wp).bounds, mesh_tm.bounds, rtol=1e-5, atol=1e-5
    )


def test_cylinder_requires_height_or_segment(device: str) -> None:
    with pytest.raises(ValueError, match="height or segment"):
        od.creation.cylinder(radius=1.0, device=device)
    with pytest.raises(ValueError, match="segment must be"):
        od.creation.cylinder(
            radius=1.0, segment=cast("Sequence[Sequence[float]]", np.zeros((3, 3))), device=device
        )


@pytest.mark.parity("cone", "trimesh")
def test_cone(device: str) -> None:
    """
    Class A on the faces, plus the exact vertex count the fan collapse must produce.

    34 = 32 rim + apex + base centre. The count is the claim: without the vertex collapse the
    two fans each carry their own copy of the rim and the mesh is not closed, which the closure
    assert then catches independently.
    """
    vertices_wp, faces_wp = od.creation.cone(radius=1.0, height=2.0, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.cone(radius=1.0, height=2.0))
    # 32 rim vertices plus the apex and the base center; the two fans need the vertex collapse.
    assert vertices_wp.size == 34
    _assert_closed(vertices_wp, faces_wp)


@pytest.mark.parity("annulus", "trimesh")
def test_annulus(device: str) -> None:
    """
    Class A on the faces, plus closure and Euler characteristic 0 -- a torus-like shell.

    The profile's closing point has to collapse or the inner-wall seam stays open, and ``chi ==
    0`` is what detects that: a mesh with the seam open is still watertight-looking by face
    count.
    """
    vertices_wp, faces_wp = od.creation.annulus(0.5, 1.0, height=2.0, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.annulus(0.5, 1.0, height=2.0))
    # The closing point of the annulus profile has to collapse, or the inner-wall seam stays open.
    _assert_closed(vertices_wp, faces_wp)
    assert od.measures.euler_characteristic(faces_wp) == 0


def test_annulus_zero_inner_radius_is_a_cylinder(device: str) -> None:
    _assert_same_faces(
        *od.creation.annulus(0.0, 1.0, height=2.0, device=device),
        tm.creation.cylinder(radius=1.0, height=2.0),
    )


def test_annulus_requires_height_or_segment(device: str) -> None:
    with pytest.raises(ValueError, match="height or segment"):
        od.creation.annulus(0.5, 1.0, device=device)


@pytest.mark.parity("torus", "trimesh")
def test_torus(device: str) -> None:
    """
    Class A on the faces, with the analytic volume ``2 pi^2 R r^2`` to 2 %.

    The 2 % band is the tessellation error at 32x32 sections, not slack: the polygonal torus
    genuinely encloses less than the smooth one, and the face-count assert pins the resolution
    the band assumes.
    """
    vertices_wp, faces_wp = od.creation.torus(1.0, 0.25, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.torus(1.0, 0.25))
    assert faces_wp.size // 3 == 2 * 32 * 32
    exact_volume = 2.0 * np.pi**2 * 1.0 * 0.25**2
    assert abs(warp_to_trimesh(vertices_wp, faces_wp).volume - exact_volume) / exact_volume < 0.02


_CLOSED_BUILDERS = {
    "uv_sphere": lambda device: od.creation.uv_sphere(device=device),
    "capsule": lambda device: od.creation.capsule(device=device),
    "cylinder": lambda device: od.creation.cylinder(radius=1.0, height=2.0, device=device),
    "cone": lambda device: od.creation.cone(radius=1.0, height=2.0, device=device),
    "annulus": lambda device: od.creation.annulus(0.5, 1.0, height=2.0, device=device),
    "torus": lambda device: od.creation.torus(1.0, 0.25, device=device),
    "icosphere": lambda device: od.creation.icosphere(subdivisions=2, device=device),
    "box": lambda device: od.creation.box(device=device),
}


@pytest.mark.parity("torus", "pytorch3d")
def test_torus_matches_pytorch3d(device: str) -> None:
    """
    Class B: ``utils.torus(r, R, sides, rings)`` is ordito's torus under a parameter swap.

    Three things have to be lined up and none is a tolerance. pytorch3d takes the **minor** radius
    first and ordito the major; its ``sides`` is the minor loop's section count and its ``rings``
    the major loop's, which is the reverse order of ordito's ``(major_sections,
    minor_sections)``. At the matching mapping both build 96 vertices and 192 faces.

    The comparison is then the surface rather than the buffers -- pytorch3d walks its own Python
    double loop and numbers vertices in its own order -- so it is a nearest-vertex bijection plus
    the two radii recovered from the point set, which is what actually distinguishes a swapped
    ``r``/``R`` from a correct one.
    """
    major, minor, major_sections, minor_sections = 1.0, 0.3, 12, 8
    torus_p3d = p3d_utils.torus(minor, major, minor_sections, major_sections)
    vertices_p3d, faces_p3d = pytorch3d_to_numpy(torus_p3d)
    vertices_wp, faces_wp = od.creation.torus(
        major, minor, major_sections, minor_sections, device=device
    )
    vertices_np = vertices_wp.numpy().astype(np.float64)

    assert vertices_p3d.shape[0] == major_sections * minor_sections
    assert faces_p3d.shape[0] == 2 * major_sections * minor_sections
    assert len(vertices_np) == vertices_p3d.shape[0]
    assert faces_wp.size // 3 == faces_p3d.shape[0]

    # Distance from the major circle recovers the minor radius on both sides.
    for points_np in (vertices_np, vertices_p3d):
        radial_np = np.linalg.norm(points_np[:, :2], axis=1)
        tube_np = np.sqrt((radial_np - major) ** 2 + points_np[:, 2] ** 2)
        assert np.allclose(tube_np, minor, rtol=1e-5, atol=1e-6)

    distances_np, indices_np = cKDTree(vertices_p3d).query(vertices_np)
    assert float(np.max(distances_np)) < 1e-6
    assert np.unique(indices_np).size == len(vertices_np)


@pytest.mark.parametrize("name", sorted(_CLOSED_BUILDERS))
def test_closed_primitives_are_deterministic_volumes(device: str, name: str) -> None:
    """
    Not a library comparison: every closed builder bounds a volume, and builds it the same twice.

    The closedness is the single assertion that pins the vertex collapse: without it the apex, pole
    and closed-profile vertices stay duplicated per slice and nothing here is watertight.
    """
    vertices_wp, faces_wp = _CLOSED_BUILDERS[name](device)
    _assert_closed(vertices_wp, faces_wp)
    assert od.validation.is_volume(vertices_wp, faces_wp)
    second_v, second_f = _CLOSED_BUILDERS[name](device)
    assert np.array_equal(vertices_wp.numpy(), second_v.numpy())
    assert np.array_equal(faces_wp.numpy(), second_f.numpy())


@pytest.mark.parity("revolve", "trimesh")
def test_revolve_matches_trimesh(device: str) -> None:
    """
    Class A on the faces: the same profile revolved into the same connectivity.

    The profile closes on itself, so this also pins the seam handling -- an implementation
    duplicating the closing vertex gets a different face table, not merely different positions.
    """
    profile_np = np.array([[0.25, 0.0], [1.0, 0.0], [1.0, 1.0], [0.25, 1.0], [0.25, 0.0]])
    _assert_same_faces(
        *od.creation.revolve(points_to_warp_uv(profile_np, device), sections=24),
        tm.creation.revolve(profile_np, sections=24),
    )


@pytest.mark.parametrize("cap", [False, True])
def test_revolve_partial_revolution(device: str, cap: bool) -> None:
    profile_np = np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0], [0.5, 0.0]])
    vertices_wp, faces_wp = od.creation.revolve(
        points_to_warp_uv(profile_np, device), angle=np.pi, cap=cap, sections=16
    )
    mesh_tm = tm.creation.revolve(profile_np, angle=np.pi, cap=cap, sections=16)
    closed = od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert closed is cap
    if cap:
        # The cap winding has to come out facing away from the solid, not into it.
        _assert_same_solid(vertices_wp, faces_wp, mesh_tm)
        _assert_closed(vertices_wp, faces_wp)
        assert od.validation.is_volume(vertices_wp, faces_wp)
    else:
        _assert_same_faces(vertices_wp, faces_wp, mesh_tm)


def test_revolve_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        od.creation.revolve(points_to_warp_uv(np.zeros((1, 2)), device))
    with pytest.raises(ValueError, match="sections must be at least 1"):
        od.creation.revolve(points_to_warp_uv(_SQUARE_RING, device), sections=0)


def test_revolve_absolute_tolerance_is_scale_dependent(device: str) -> None:
    # Documented limitation: the degenerate-triangle filter compares an absolute area against
    # 1e-8, so a large enough sphere keeps its (near-)zero-area polar triangles. Recorded here so
    # the threshold in revolve's Notes stays honest rather than asserted as desirable.
    small_v, small_f = od.creation.uv_sphere(radius=1.0, count=(8, 16), device=device)
    large_v, large_f = od.creation.uv_sphere(radius=1.0e5, count=(8, 16), device=device)
    _assert_closed(small_v, small_f)
    assert large_f.size >= small_f.size
    assert np.allclose(np.linalg.norm(large_v.numpy(), axis=1), 1.0e5, rtol=1e-5, atol=1.0)


# --- extrusion and polygons -------------------------------------------------------------


@pytest.mark.parity("extrude_polygon", "pyvista", "open3d")
@pytest.mark.parametrize("ring_size", [8, 32])
def test_extrude_polygon_matches_pyvista_and_open3d(device: str, ring_size: int) -> None:
    """
    Class A on the counts and the solid: a convex ring extrudes to the same mesh in all three.

    A convex ring is used deliberately rather than the L-shape the trimesh comparison below runs on.
    The cap admits only one triangulation up to a rotation of the fan there, so the *counts* are
    comparable exactly -- ``2 * (n - 2)`` cap triangles plus ``2 * n`` wall triangles, measured
    **64** vertices and **124** faces from all three on a 32-gon, watertight with chi = 2.

    The split between the two references is who triangulates the cap, and it decides how each one is
    handed the input:

    * **pyvista** ``extrude((0, 0, h), capping=True)`` takes a ``PolyData`` whose single polygon
      *cell* is the cap, so VTK triangulates on the way out and ``.triangulate()`` is required --
      without it the result carries polygons, not triangles, and the cell count is not comparable.
    * **open3d** ``extrude_linear`` walls an **already triangulated** mesh; it does not triangulate
      a ring, so the cap fan is built here. Its faces must be ``Int32``/``Int64`` -- a ``UInt32``
      tensor raises ``Tensor has dtype UInt32, but is expected to have dtype among {Int32, Int64}``,
      although ``RaycastingScene.add_triangles`` accepts one. Two conventions inside one API.

    Both also want 3-D points where ordito's signature takes ``wp.vec2``, which is the only other
    transform.

    **Bug class excluded:** a wall band that skips or doubles a quad, which the face count catches,
    and an unclosed solid, which the watertightness and chi asserts catch on every side.
    """
    angle_np = 2.0 * np.pi * np.arange(ring_size) / ring_size
    ring_np = np.column_stack([np.cos(angle_np), np.sin(angle_np)])
    ring3_np = np.column_stack([ring_np, np.zeros(ring_size)])
    n_expected = 2 * (ring_size - 2) + 2 * ring_size

    vertices_wp, faces_wp = od.creation.extrude_polygon(points_to_warp_uv(ring_np, device), 1.0)
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)
    assert vertices_wp.size == 2 * ring_size
    assert faces_wp.size // 3 == n_expected
    assert mesh_wp.is_watertight
    assert mesh_wp.euler_number == 2

    polygon_pv = pv.PolyData(ring3_np, faces=np.hstack([[ring_size], np.arange(ring_size)]))
    extruded_pv = polygon_pv.extrude((0.0, 0.0, 1.0), capping=True).triangulate()
    mesh_pv = tm.Trimesh(
        np.asarray(extruded_pv.points), np.asarray(extruded_pv.regular_faces), process=False
    )

    fan_np = np.array([[0, i, i + 1] for i in range(1, ring_size - 1)], dtype=np.int32)
    disc_o3d = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(ring3_np, dtype=np.float64)),  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
        o3d.core.Tensor(np.ascontiguousarray(fan_np)),  # pyright: ignore[reportCallIssue]
    )
    extruded_o3d = disc_o3d.extrude_linear([0.0, 0.0, 1.0])  # pyright: ignore[reportArgumentType]  # takes any sequence
    mesh_o3d = tm.Trimesh(
        extruded_o3d.vertex.positions.numpy(), extruded_o3d.triangle.indices.numpy(), process=False
    )

    for reference_tm in (mesh_pv, mesh_o3d):
        assert reference_tm.vertices.shape[0] == 2 * ring_size
        assert reference_tm.faces.shape[0] == n_expected
        assert reference_tm.is_watertight
        assert reference_tm.euler_number == 2
        assert np.isclose(abs(reference_tm.volume), abs(mesh_wp.volume), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ring_name", ["square", "L"])
@pytest.mark.parametrize("height", [0.5, -0.5])
@pytest.mark.parity("extrude_polygon", "trimesh")
def test_extrude_polygon(device: str, ring_name: str, height: float) -> None:
    """
    Class C (same solid, not the same triangles): trimesh triangulates the caps differently.

    Both signs of ``height`` are run, and the comparison is on the enclosed volume and outward
    orientation rather than the face table -- an L-shaped ring admits several valid cap
    triangulations, so equality is not available.
    """
    ring_np = _SQUARE_RING if ring_name == "square" else _L_RING
    vertices_wp, faces_wp = od.creation.extrude_polygon(points_to_warp_uv(ring_np, device), height)
    mesh_tm = tm.creation.extrude_polygon(sg.Polygon(ring_np), height)
    assert vertices_wp.size == 2 * ring_np.shape[0]
    # Both signs of height must give an outward-facing solid of the same volume.
    _assert_same_solid(vertices_wp, faces_wp, mesh_tm)
    _assert_closed(vertices_wp, faces_wp)


def test_extrude_polygon_mid_plane(device: str) -> None:
    vertices_wp, faces_wp = od.creation.extrude_polygon(
        points_to_warp_uv(_SQUARE_RING, device), 1.0, mid_plane=True
    )
    assert np.allclose(warp_to_trimesh(vertices_wp, faces_wp).bounds[:, 2], [-0.5, 0.5], atol=1e-5)


def _star_ring(n: int, clockwise: bool) -> np.ndarray:
    """Return a non-convex ``n``-vertex star, which ``triangulate_polygon`` ear-clips."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = np.where(np.arange(n) % 2 == 0, 1.0, 0.4)
    ring_np = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    return ring_np[::-1].copy() if clockwise else ring_np


@pytest.mark.parametrize("ring_name", ["square", "L", "star", "star_cw"])
@pytest.mark.parametrize("height", [0.5, -0.5])
def test_extrude_polygon_ring_walls_match_the_derived_boundary(
    device: str, ring_name: str, height: float
) -> None:
    """
    Ordito against ordito: the ring-edge walls against the derived boundary.

    ``extrude_polygon`` walls a full ``n - 2`` triangulation from its ring edges and never derives
    the boundary; ``extrude_triangulation`` derives it from the same triangulation. The derived path
    carries the oracle (``test_extrude_polygon`` compares it with trimesh), so the two must build
    the same mesh -- and, on the CPU device, the same buffers row for row, since the ring walls are
    emitted in the derived boundary's order. On CUDA the ear clip's face order is not reproducible
    between calls, so there the rows are compared as a set. The star rings reach the ear clip in
    both orientations, and both height signs re-wind the caps.
    """
    rings = {"square": _SQUARE_RING, "L": _L_RING}
    ring_np = rings.get(ring_name, _star_ring(12, ring_name == "star_cw"))
    ring_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    assert faces_wp.size // 3 == ring_np.shape[0] - 2
    derived_v, derived_f = od.creation.extrude_triangulation(ring_wp, faces_wp, height)
    ring_v, ring_f = od.creation.extrude_polygon(points_to_warp_uv(ring_np, device), height)
    _assert_closed(ring_v, ring_f)
    assert np.array_equal(ring_v.numpy(), derived_v.numpy())
    if device == "cpu":
        assert np.array_equal(ring_f.numpy(), derived_f.numpy())
    else:
        assert_unordered_rows_equal(ring_f.numpy().reshape(-1, 3), derived_f.numpy().reshape(-1, 3))


def test_extrude_triangulation_recovers_subdivided_boundary(device: str) -> None:
    # A boundary edge split by an extra collinear vertex still has to become two wall quads, which
    # is why the boundary is recovered from the triangulation rather than taken from the input ring.
    ring_np = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    solid_v, solid_f = od.creation.extrude_triangulation(vertices_wp, faces_wp, 0.5)
    _assert_closed(solid_v, solid_f)
    assert solid_f.size // 3 == 2 * 3 + 2 * 5
    assert np.isclose(warp_to_trimesh(solid_v, solid_f).volume, 1.0, rtol=1e-4)


def test_extrude_triangulation_invalid(device: str) -> None:
    ring_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(_SQUARE_RING, device))
    with pytest.raises(ValueError, match="height must be nonzero"):
        od.creation.extrude_triangulation(ring_wp, faces_wp, 0.0)
    with pytest.raises(ValueError, match="multiple of 3"):
        od.creation.extrude_triangulation(ring_wp, faces_wp[:2].contiguous(), 1.0)


_SWEEP_PATHS = {
    "straight": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
    "bent": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 2.0, 2.0]]),
    "closed": np.column_stack(
        (
            np.cos(2.0 * np.pi * np.arange(9) / 8),
            np.sin(2.0 * np.pi * np.arange(9) / 8),
            np.zeros(9),
        )
    )
    * 2.0,
    # Doubles back on itself at the middle vertex, so the two segment tangents there sum to the
    # zero vector -- ``sweep_plane_normals``' own comment names this cancellation. Included so the
    # degenerate-normal path always runs under the ordinary sweep suite, not only in the dedicated
    # position regression below (which needs an asymmetric profile the square one here can't give).
    "reversing": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]]),
}


@pytest.mark.parametrize("path_name", sorted(_SWEEP_PATHS))
@pytest.mark.parity("sweep_polygon", "trimesh")
def test_sweep_polygon(device: str, path_name: str) -> None:
    """
    Class C (same solid): the swept volume, since the frame carried along the path is a gauge.

    Two libraries can sweep the same profile with different twist about the path and produce
    the same solid, so the volume and the closure are what compare; the profile is square,
    which makes a twist difference invisible in the volume by construction.
    """
    ring_np = np.array([[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]])
    path_np = _SWEEP_PATHS[path_name]
    vertices_wp, faces_wp = od.creation.sweep_polygon(
        points_to_warp_uv(ring_np, device), points_to_warp(path_np, device)
    )
    mesh_tm = tm.creation.sweep_polygon(sg.Polygon(ring_np), path_np)
    _assert_same_solid(vertices_wp, faces_wp, mesh_tm)
    _assert_closed(vertices_wp, faces_wp)
    assert warp_to_trimesh(vertices_wp, faces_wp).body_count == 1


def test_sweep_polygon_reversing_path_matches_trimesh_at_the_reversal(device: str) -> None:
    """
    Regression: a sharp path reversal used to twist the cross-section ~90 degrees at that vertex.

    ``sweep_plane_normals`` legitimately produces the exact zero vector at an interior vertex where
    two consecutive path tangents cancel (the "reversing" path in ``_SWEEP_PATHS``, and the module's
    own comment already names the case). ``sweep_transforms`` used to compute
    ``phi = acos(0) == pi/2`` for that degenerate normal, where trimesh's ``vector_to_spherical``
    (which the kernel's docstring says it is unrolled from) leaves a near-zero vector's angles at
    their zero default instead — the identity mapping for local +Z, not a quarter turn about it.

    ``test_sweep_polygon``'s volume-based Class C check cannot catch this: rotating a straight
    prism's cross-section about its own axis doesn't change the swept volume, and that test's own
    profile is a square, invariant under a 90 degree rotation in any case. This uses an asymmetric
    rectangle instead, and matches by nearest point (Class B) since trimesh's ear-clipped caps use a
    different, equally valid diagonal choice than ordito's — only the *positions* are the shared
    claim, and this profile has no interior cap point for that choice to add or move.
    """
    ring_np = np.array([[-0.5, -0.1], [0.5, -0.1], [0.5, 0.1], [-0.5, 0.1]])
    path_np = _SWEEP_PATHS["reversing"]
    vertices_wp, faces_wp = od.creation.sweep_polygon(
        points_to_warp_uv(ring_np, device), points_to_warp(path_np, device), cap=True, connect=False
    )
    mesh_tm = tm.creation.sweep_polygon(sg.Polygon(ring_np), path_np, cap=True, connect=False)
    _assert_closed(vertices_wp, faces_wp)
    assert vertices_wp.size == mesh_tm.vertices.shape[0]
    distance_np, _ = cKDTree(mesh_tm.vertices).query(vertices_wp.numpy().astype(np.float64))
    assert np.max(distance_np) < 1e-4, f"vertices differ by up to {np.max(distance_np):.3e}"


def test_sweep_polygon_angles_roll_the_profile(device: str) -> None:
    ring_np = np.array([[-0.5, -0.1], [0.5, -0.1], [0.5, 0.1], [-0.5, 0.1]])
    # A quarter turn spread over four segments. Concentrating the same twist in a single segment
    # sweeps the profile through itself, and both libraries then report a negative volume for the
    # self-intersecting result — so the roll is kept gentle enough for the solid to stay valid.
    path_np = np.column_stack((np.zeros(5), np.zeros(5), np.linspace(0.0, 1.0, 5)))
    path_wp = points_to_warp(path_np, device)
    angles_np = np.linspace(0.0, np.pi / 2.0, 5)
    straight_v, _ = od.creation.sweep_polygon(points_to_warp_uv(ring_np, device), path_wp)
    twisted_v, twisted_f = od.creation.sweep_polygon(
        points_to_warp_uv(ring_np, device),
        path_wp,
        angles=wp.array(angles_np.astype(np.float32), device=device),
    )
    assert not np.allclose(straight_v.numpy(), twisted_v.numpy(), atol=1e-3)
    _assert_closed(twisted_v, twisted_f)
    _assert_same_solid(
        twisted_v,
        twisted_f,
        tm.creation.sweep_polygon(sg.Polygon(ring_np), path_np, angles=angles_np),
    )


def test_sweep_polygon_open_path_without_caps(device: str) -> None:
    ring_np = np.array([[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]])
    path_wp = points_to_warp(_SWEEP_PATHS["straight"], device)
    _, faces_wp = od.creation.sweep_polygon(points_to_warp_uv(ring_np, device), path_wp, cap=False)
    assert not od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert faces_wp.size // 3 == 2 * 2 * 4


def test_sweep_polygon_invalid(device: str) -> None:
    ring_wp = points_to_warp_uv(_SQUARE_RING, device)
    single_wp = wp.array(np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="at least 2 points"):
        od.creation.sweep_polygon(ring_wp, single_wp)
    path_wp = points_to_warp(_SWEEP_PATHS["straight"], device)
    with pytest.raises(ValueError, match="one entry per path point"):
        od.creation.sweep_polygon(
            ring_wp, path_wp, angles=wp.zeros(2, dtype=wp.float32, device=device)
        )


def test_sweep_polygon_rejects_a_non_simple_ring(device: str) -> None:
    """
    A self-intersecting ring triangulates partially, so the sweep derives its boundary and raises.

    The ring-edge walls are only taken for a full ``n - 2`` triangulation; a bow-tie gets one
    triangle for four vertices, which is the branch where the boundary is derived and checked.
    """
    bowtie_np = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    ring_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(bowtie_np, device))
    assert faces_wp.size // 3 < ring_wp.size - 2
    path_wp = points_to_warp(_SWEEP_PATHS["straight"], device)
    with pytest.raises(ValueError, match="simple ring"):
        od.creation.sweep_polygon(points_to_warp_uv(bowtie_np, device), path_wp)


# --- composites -------------------------------------------------------------------------


def _triangle_soup(
    device: str, seed: int = 7
) -> tuple[np.ndarray, wp.array[wp.vec3], wp.array[wp.int32]]:
    triangles_np = np.random.default_rng(seed).random((5, 3, 3)) + np.array([0.0, 0.0, 1.0])
    vertices_wp = points_to_warp(triangles_np.reshape(-1, 3), device)
    faces_wp = wp.array(np.arange(15, dtype=np.int32), dtype=wp.int32, device=device)
    return triangles_np, vertices_wp, faces_wp


@pytest.mark.parity("truncated_prisms", "trimesh")
@pytest.mark.parametrize("case", ["z_zero", "plane", "reversed_winding"])
def test_truncated_prisms(device: str, case: str) -> None:
    """
    Class C (volume and body count): trimesh emits one prism per triangle in its own vertex order.

    The counts are exact -- 6 vertices and 8 faces per input triangle -- and the volume is
    compared to trimesh's; ``body_count == 5`` is what catches prisms welded together, which
    the volume alone would not show. Three inputs: the default ``z = 0`` plane; an explicit plane,
    which changes only where the prisms stop; and every source triangle reversed to face the plane,
    whose prisms need their winding reversed or the bodies come out inside-out with negative volume.
    """
    triangles_np, vertices_wp, faces_wp = _triangle_soup(device)
    if case == "reversed_winding":
        triangles_np = np.ascontiguousarray(triangles_np[:, ::-1, :])
        vertices_wp = points_to_warp(triangles_np.reshape(-1, 3), device)
    if case == "plane":
        origin_np, normal_np = np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0])
        prism_v, prism_f = od.creation.truncated_prisms(
            vertices_wp,
            faces_wp,
            origin=wp.vec3(*origin_np.tolist()),
            normal=wp.vec3(*normal_np.tolist()),
        )
        mesh_tm = tm.creation.truncated_prisms(triangles_np, origin=origin_np, normal=normal_np)
    else:
        prism_v, prism_f = od.creation.truncated_prisms(vertices_wp, faces_wp)
        mesh_tm = tm.creation.truncated_prisms(triangles_np)
    prism_tm = warp_to_trimesh(prism_v, prism_f)
    assert prism_v.size == 6 * 5
    assert prism_f.size // 3 == 8 * 5
    assert prism_tm.volume > 0.0
    assert np.isclose(prism_tm.volume, mesh_tm.volume, rtol=1e-4)
    assert prism_tm.body_count == 5


def test_truncated_prisms_requires_normal_with_origin(device: str) -> None:
    _, vertices_wp, faces_wp = _triangle_soup(device)
    with pytest.raises(ValueError, match="normal is required"):
        od.creation.truncated_prisms(vertices_wp, faces_wp, origin=wp.vec3(0.0, 0.0, 0.0))


def test_axis(device: str) -> None:
    vertices_wp, faces_wp = od.creation.axis(device=device)
    ball_v, ball_f = od.creation.icosphere(radius=0.04, device=device)
    shaft_v, shaft_f = od.creation.cylinder(radius=0.008, height=0.4, device=device)
    assert vertices_wp.size == ball_v.size + 3 * shaft_v.size
    assert faces_wp.size == ball_f.size + 3 * shaft_f.size
    # One shaft runs out to axis_length along each of X, Y and Z.
    assert np.allclose(warp_to_trimesh(vertices_wp, faces_wp).bounds[1], 0.4, rtol=1e-3, atol=1e-3)
    assert np.allclose(
        warp_to_trimesh(vertices_wp, faces_wp).bounds[0], -0.04, rtol=1e-3, atol=1e-3
    )


def test_axis_transform(device: str) -> None:
    matrix_np = np.asarray(tm.transformations.rotation_matrix(np.deg2rad(90.0), [1.0, 0.0, 0.0]))
    matrix_np[:3, 3] = np.array([1.0, 0.0, 0.0])
    vertices_wp, faces_wp = od.creation.axis(transform=_mat44(matrix_np), device=device)
    expected_np = tm.transform_points(
        od.creation.axis(device=device)[0].numpy().astype(np.float64), matrix_np
    )
    assert np.allclose(vertices_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)
    assert faces_wp.size > 0


@pytest.mark.parametrize("surface", sorted(_PARAMETRIC_TABLE))
@pytest.mark.parity("parametric_surface", "pyvista")
@pytest.mark.parity("super_ellipsoid", "pyvista")
@pytest.mark.parity("super_toroid", "pyvista")
def test_parametric_surface_matches_pyvista(device: str, surface: _Surface) -> None:
    """
    Class A on the topology, Class B on the geometry, against ``pv.Parametric*(clean=True)``.

    The counts and the Euler characteristic are integers and compare directly. The vertex *order*
    differs — VTK welds its raw lattice by distance and drops whichever duplicate it meets second —
    so the positions are compared as point **sets**, by a two-sided nearest-neighbour query rather
    than by ``lexsort_rows``, which these surfaces' exact symmetric ties make unusable.

    ``clean=True`` is passed explicitly on every surface: pyvista sets it on only 9 of the 21, so at
    its own defaults twelve of these arrive as topological disks and every assertion below would
    compare ordito's closed answer against an accidentally open one.

    The face normals are compared as well as the positions, which is what pins the winding: a point
    set alone cannot tell the two orientations of a surface apart.

    Each surface also has the topology it exists to provide, with open3d reading orientability:
    Class A, ``o3d.geometry.TriangleMesh.is_orientable`` is the oracle ordito's own
    [`is_orientable`][ordito.validation.is_orientable] was written against, and the six
    non-orientable surfaces here are the first inputs in the suite for which it answers ``False`` —
    without them the comparison is one-sided and a predicate returning a constant would pass it.
    """
    vertices_wp, faces_wp = _build_parametric(surface, 40, device)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)

    reference_pv = getattr(pv, _PARAMETRIC_PYVISTA[surface])(u_res=40, v_res=40, clean=True)
    points_pv = np.asarray(reference_pv.points)
    # Anti-vacuity: a reference that returned nothing would pass every comparison below.
    assert reference_pv.n_points > 1_000, "pyvista returned a degenerate surface"

    expected = _PARAMETRIC_TABLE[surface]
    if surface == "catalan_minimal":
        # The one documented divergence: two sheets of the immersion cross, and VTK's distance weld
        # merges 40 lattice points the parameterization does not identify -- dropping the 2
        # triangles that thereby became degenerate. ordito keeps the sheets apart.
        assert (reference_pv.n_points, reference_pv.n_faces) == (1560, 3040)
        assert np.max(cKDTree(vertices_np).query(points_pv)[0]) < 1e-5
    else:
        assert (reference_pv.n_points, reference_pv.n_faces) == (expected.points, expected.faces)
        assert euler_characteristic(np.asarray(reference_pv.regular_faces)) == expected.chi
        assert np.max(cKDTree(points_pv).query(vertices_np)[0]) < 1e-5
        assert np.max(cKDTree(vertices_np).query(points_pv)[0]) < 1e-5
    assert (len(vertices_np), len(faces_np)) == (expected.points, expected.faces)

    centroid_np, normal_np = _face_frames(vertices_np, faces_np)
    centroid_pv, normal_pv = _face_frames(points_pv, np.asarray(reference_pv.regular_faces))
    distance_np, match_np = cKDTree(centroid_pv).query(centroid_np)
    matched = np.asarray(distance_np) < 1e-5
    assert matched.mean() > 0.9, "face centroids do not correspond"
    aligned_np = np.einsum("ij,ij->i", normal_np[matched], normal_pv[np.asarray(match_np)[matched]])
    # A handful of triangles are degenerate enough that their normal is float noise; every other
    # one must agree in *direction*, not just in plane.
    assert (aligned_np > 0.9).mean() > 0.999

    mesh_tm = warp_to_trimesh(vertices_wp, faces_wp)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_np),
        o3d.utility.Vector3iVector(faces_np.astype(np.int32)),
    )
    assert mesh_o3d.is_orientable() == expected.orientable
    assert bool(od.validation.is_orientable(faces_wp)) == expected.orientable
    # The sharpest statement about the index arithmetic: the seam-glued template winds consistently
    # exactly where a consistent winding exists at all, with no repair pass.
    assert bool(od.validation.is_winding_consistent(faces_wp)) == expected.orientable
    assert euler_characteristic(faces_np) == expected.chi
    assert open_edge_count(faces_np) == expected.open_edges
    assert mesh_tm.is_watertight == expected.watertight
    assert len(mesh_tm.outline().entities if expected.open_edges else []) == expected.loops


@pytest.mark.parametrize("surface", ["boy", "mobius", "dini", "super_toroid"])
def test_parametric_surface_topology_is_resolution_independent(
    device: str, surface: _Surface
) -> None:
    """
    The identification is combinatorial, so the topology cannot move with the resolution.

    This is the property a distance weld would not have, and it needs no reference: a tolerance
    applied to a float32 vertex buffer glues a different set of points at 20, 40 and 80 samples.
    """
    invariants = set()
    for resolution in (20, 40, 80):
        vertices_wp, faces_wp = _build_parametric(surface, resolution, device)
        faces_np = faces_wp.numpy().reshape(-1, 3)
        assert len(faces_np) == 2 * (resolution - 1) ** 2 - _PARAMETRIC_TABLE[
            surface
        ].pole_cells * (resolution - 1)
        assert vertices_wp.size == int(faces_np.max()) + 1
        invariants.add(
            (
                euler_characteristic(faces_np),
                bool(od.validation.is_orientable(faces_wp)),
                open_edge_count(faces_np) // (resolution - 1),
            )
        )
    assert len(invariants) == 1, f"topology moved with the resolution: {invariants}"


def test_parametric_surface_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        od.creation.parametric_surface("klein_bottle", device=device)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="at least 2"):
        od.creation.parametric_surface("mobius", 1, 40, device=device)
    with pytest.raises(ValueError, match="at least 2"):
        od.creation.parametric_surface("mobius", 40, 1, device=device)


@pytest.mark.parametrize(
    ("surface", "u_resolution", "v_resolution"),
    [("klein", 40, 2), ("pseudosphere", 40, 2), ("bohemian_dome", 2, 40)],
)
def test_parametric_surface_rejects_resolution_2_on_a_wrapped_axis(
    device: str, surface: ParametricSurfaceKind, u_resolution: int, v_resolution: int
) -> None:
    """
    Not a library comparison: no reference builds these lattices combinatorially.

    There is nothing to compare a *rejection* against, so the claim is the rejection itself.

    A wrapped, untwisted axis identifies its last row with its first, so at resolution 2 every cell
    along it has two equal corners and the degeneracy filter removes the entire face buffer -- a
    vertices-only "mesh" that contradicts the documented face count and, handed to
    ``Trimesh.warp_mesh``, builds the zero-triangle ``wp.Mesh`` that corrupts CUDA allocator state.
    What the invariant excludes is only that silent case; it says nothing about the face *values*
    at any resolution, which the topology tests above cover.
    """
    with pytest.raises(ValueError, match="at least 3"):
        od.creation.parametric_surface(surface, u_resolution, v_resolution, device=device)
    # The twisted wrap is the control: a flip keeps the two rows distinct, so 2 stays admissible.
    _vertices_wp, faces_wp = od.creation.parametric_surface("mobius", 2, 40, device=device)
    assert faces_wp.size > 0

    with pytest.raises(ValueError, match="at least 3"):
        od.creation.super_toroid(u_resolution=2, device=device)
    with pytest.raises(ValueError, match="at least 3"):
        od.creation.super_ellipsoid(u_resolution=2, device=device)


def test_super_ellipsoid_unit_exponents_are_a_sphere(device: str) -> None:
    """``n1 = n2 = 1`` is the ellipsoid, which is why no separate ``creation.ellipsoid`` exists."""
    vertices_wp, faces_wp = od.creation.super_ellipsoid(device=device)
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)
    _assert_closed(vertices_wp, faces_wp)

    scaled_wp, _ = od.creation.super_ellipsoid(radii=(2.0, 1.0, 0.5), device=device)
    axes_np = np.abs(scaled_wp.numpy()).max(axis=0)
    assert np.allclose(axes_np, [2.0, 1.0, 0.5], rtol=1e-2, atol=1e-2)
    # The squareness axis is live: at n1 = n2 = 0.4 the surface bulges out towards its box.
    boxy_wp, _ = od.creation.super_ellipsoid(n1=0.4, n2=0.4, device=device)
    assert np.linalg.norm(boxy_wp.numpy(), axis=1).max() > 1.3


def test_super_ellipsoid_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="radii must be"):
        od.creation.super_ellipsoid(radii=(1.0, 1.0), device=device)  # pyright: ignore[reportArgumentType]


def test_super_toroid_unit_exponents_are_a_torus(device: str) -> None:
    """``n1 = n2 = 1`` is VTK's (1, 0.5) torus, the genus-1 counterpart of the sphere above."""
    vertices_wp, faces_wp = od.creation.super_toroid(device=device)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    ring_np = np.linalg.norm(vertices_np[:, :2], axis=1) - 1.0
    tube_np = np.hypot(ring_np, vertices_np[:, 2])
    assert np.allclose(tube_np, 0.5, rtol=1e-5, atol=1e-5)
    _assert_closed(vertices_wp, faces_wp)
    assert od.measures.euler_characteristic(faces_wp) == 0


def test_random_hills(device: str) -> None:
    vertices_wp, faces_wp = od.creation.random_hills(seed=3, device=device)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    assert vertices_wp.size == 1_600
    assert faces_wp.size // 3 == 2 * 39 * 39
    assert od.measures.euler_characteristic(faces_wp) == 1
    # The lattice is the plain grid over [-10, 10]^2, and only the height is random.
    assert np.allclose(vertices_np[:, :2].min(axis=0), -10.0)
    assert np.allclose(vertices_np[:, :2].max(axis=0), 10.0)
    assert 0.0 < vertices_np[:, 2].max() <= 30 * 2.0

    assert np.array_equal(
        vertices_np, od.creation.random_hills(seed=3, device=device)[0].numpy().astype(np.float64)
    )
    assert not np.array_equal(
        vertices_np, od.creation.random_hills(seed=4, device=device)[0].numpy().astype(np.float64)
    )
    # Amplitude scales the height field linearly, and no hills leaves it flat.
    doubled_np = od.creation.random_hills(amplitude=4.0, seed=3, device=device)[0].numpy()
    assert np.allclose(doubled_np[:, 2], 2.0 * vertices_np[:, 2], rtol=1e-5, atol=1e-5)
    assert not od.creation.random_hills(n_hills=0, device=device)[0].numpy()[:, 2].any()


def test_random_hills_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="variances must be positive"):
        od.creation.random_hills(x_variance=0.0, device=device)
    with pytest.raises(ValueError, match="at least 2"):
        od.creation.random_hills(u_resolution=1, device=device)


def test_random_soup(device: str) -> None:
    vertices_wp, faces_wp = od.creation.random_soup(50, seed=3, device=device)
    assert vertices_wp.size == 150
    assert np.array_equal(faces_wp.numpy(), np.arange(150, dtype=np.int32))
    assert vertices_wp.numpy().min() >= -0.5
    assert vertices_wp.numpy().max() <= 0.5
    assert np.array_equal(
        vertices_wp.numpy(), od.creation.random_soup(50, seed=3, device=device)[0].numpy()
    )
    assert not np.array_equal(
        vertices_wp.numpy(), od.creation.random_soup(50, seed=4, device=device)[0].numpy()
    )


def test_empty_results(device: str) -> None:
    vertices_wp, faces_wp = od.creation.random_soup(0, seed=1, device=device)
    assert vertices_wp.size == 0
    assert faces_wp.size == 0

    empty_v = warp_empty(0, wp.vec3, device)
    empty_f = warp_empty(0, wp.int32, device)
    prism_v, prism_f = od.creation.truncated_prisms(empty_v, empty_f)
    assert prism_v.size == 0
    assert prism_f.size == 0

    solid_v, solid_f = od.creation.extrude_triangulation(
        warp_empty(0, wp.vec2, device), empty_f, 1.0
    )
    assert solid_v.size == 0
    assert solid_f.size == 0


@pytest.mark.parametrize("surface", ["boy", "cross_cap", "klein", "mobius", "dini", "conic_spiral"])
@pytest.mark.parametrize(("u_resolution", "v_resolution"), [(7, 5), (17, 33)])
def test_parametric_lattice_paths_agree(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    surface: ParametricSurfaceKind,
    u_resolution: int,
    v_resolution: int,
) -> None:
    """
    Ordito against ordito: the device lattice against the numpy one, which carries the oracle.

    Not a library comparison at this level — the reference comparisons for these surfaces are the
    pyvista tests elsewhere in this file, and they exercise whichever path the size gate selects.
    What this pins is that the gate is a *performance* switch and nothing else: the two lattices
    must be **bit-identical**, because the device path exists only to remove host work and any drift
    in the last float32 bit would be a second definition of the geometry rather than a faster route
    to the same one.

    The gate is at ``_PARAMETRIC_LATTICE_DEVICE_FROM`` lattice samples, and each path is forced
    both ways rather than left to whichever side of the gate a resolution happens to land on; the
    two lattices are a tiny one and a non-square one, so a transposed u/v index shows. The
    surfaces straddle the gluing rules the two paths have to agree on: a twisted wrap with two
    collapsed pole rows (``boy``, ``cross_cap``), a wrap in each direction (``klein``), a twist with
    a boundary (``mobius``), a plain open patch (``dini``) and a pole on one end only
    (``conic_spiral``).
    """
    forced = od.creation._PARAMETRIC_LATTICE_DEVICE_FROM  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(od.creation, "_PARAMETRIC_LATTICE_DEVICE_FROM", 1 << 30)
    host_v, host_f = od.creation.parametric_surface(
        surface, u_resolution, v_resolution, device=device
    )
    host_v_np, host_f_np = host_v.numpy(), host_f.numpy()

    monkeypatch.setattr(od.creation, "_PARAMETRIC_LATTICE_DEVICE_FROM", 0)
    device_v, device_f = od.creation.parametric_surface(
        surface, u_resolution, v_resolution, device=device
    )
    assert forced > 0, "the gate must be a positive sample count"
    # Not vacuous: the lattice really did produce a surface on both sides.
    assert len(host_v_np) > 0
    assert host_f_np.size > 0
    assert np.array_equal(device_v.numpy(), host_v_np)
    assert np.array_equal(device_f.numpy(), host_f_np)
