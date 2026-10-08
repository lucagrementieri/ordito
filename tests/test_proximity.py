"""
Regression tests for ``ordito.proximity`` mesh-query APIs.

Mesh AABB queries against a brute-force reference; closest-on-mesh tests compare against
``trimesh.proximity.closest_point``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.loss as p3d_loss
import pyvista as pv
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.spatial import Delaunay

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito.constants import TOLERANCE_MERGE
from ordito.kernels import proximity as kernel_proximity
from tests.conversions import (
    mesh_igl,
    meshlib_scalars_to_numpy,
    numpy_to_warp,
    points_to_pytorch3d,
    points_to_pyvista,
    points_to_warp,
    points_to_warp_uv,
    trimesh_to_meshlib,
    trimesh_to_open3d_t,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
    trimesh_to_warp,
    warp_empty,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Buffer

_SIGNED_DISTANCE_MESHES = ["sphere_irregular", "sphere_irregular_hollow"]
_SIGNED_DISTANCE_MESHES_TORUS = ["sphere_irregular", "sphere_irregular_hollow", "torus_irregular"]


def _queries_in_bounds_np(mesh_tm: tm.Trimesh, n: int, seed: int) -> np.ndarray:
    """
    Draw ``n`` queries from the mesh's own bounding box, grown 20 %.

    A fixed cube of queries is not usable across fixtures: ``cave_cube`` is a unit box with a
    0.1-wide cavity removed, so its *interior* is a thin shell and 200 points drawn from
    ``[-2, 2]**3`` land outside it every time -- which turns a signed-distance or winding-number
    comparison into a test of the exterior branch alone. Scaling to the fixture puts points on both
    sides of the surface for every mesh in this module, which the non-vacuity asserts then state.
    """
    lower_np, upper_np = mesh_tm.bounds
    margin_np = 0.2 * (upper_np - lower_np)
    return np.random.default_rng(seed).uniform(lower_np - margin_np, upper_np + margin_np, (n, 3))


def _random_queries_np(seed: int = 42, n: int = 200) -> np.ndarray:
    """Draw ``n`` queries uniformly from the cube ``[-2, 2]**3``."""
    return np.random.default_rng(seed).random((n, 3), dtype=np.float64) * 4.0 - 2.0


def _queries_in_box_np(mesh_tm: tm.Trimesh, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` queries uniformly from the mesh's own (ungrown) bounding box."""
    lower_np, upper_np = mesh_tm.bounds
    return lower_np + np.random.default_rng(seed).random((n, 3)) * (upper_np - lower_np)


def _upload(
    mesh_tm: tm.Trimesh, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Upload a trimesh mesh as ordito's ``(vertices, flat faces)`` pair on ``device``."""
    return numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces).ravel().astype(np.int32), device
    )


def _meshlib_distance(a_tm: tm.Trimesh, b_tm: tm.Trimesh) -> float:
    """MeshLib's exact minimum distance between two meshes, ``findDistance`` with no transform."""
    result_ml = mm.findDistance(
        mm.MeshPart(trimesh_to_meshlib(a_tm)),
        mm.MeshPart(trimesh_to_meshlib(b_tm)),
        None,  # pyright: ignore[reportArgumentType]  # nullptr is the identity; the stub drops Optional
        float(np.finfo(np.float32).max),
    )
    return float(np.sqrt(result_ml.distSq))


def test_query_mesh_aabb_with_offsets(device: str) -> None:
    rng = np.random.default_rng(11)
    n_faces = 8
    vertices_np = rng.random((n_faces * 3, 3), dtype=np.float32)
    faces_np = np.arange(n_faces * 3, dtype=np.int32).reshape(n_faces, 3)

    lower_np = np.empty((n_faces, 3), dtype=np.float32)
    upper_np = np.empty((n_faces, 3), dtype=np.float32)
    for face_idx in range(n_faces):
        tri = vertices_np[faces_np[face_idx]]
        lower_np[face_idx] = tri.min(axis=0)
        upper_np[face_idx] = tri.max(axis=0)

    query_lower_np = lower_np[:4].copy()
    query_upper_np = upper_np[:4].copy()

    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=vertices_wp, indices=faces_wp)
    query_lower_wp = points_to_warp(query_lower_np, device)
    query_upper_wp = points_to_warp(query_upper_np, device)

    indices_wp, offsets_wp = od.proximity.query_mesh_aabb_with_offsets(
        mesh, query_lower_wp, query_upper_wp, max_hits=16
    )

    indices_np = indices_wp.numpy()
    bounds_np = offsets_wp.numpy()
    assert bounds_np.shape == (query_lower_np.shape[0] + 1,)
    assert bounds_np[0] == 0
    assert bounds_np[-1] == indices_np.size

    for query_idx in range(query_lower_np.shape[0]):
        q_lower = query_lower_np[query_idx]
        q_upper = query_upper_np[query_idx]
        mask_np = np.all(lower_np <= q_upper, axis=1) & np.all(upper_np >= q_lower, axis=1)
        expected_np = np.sort(np.flatnonzero(mask_np).astype(np.int32))
        got_np = np.sort(indices_np[bounds_np[query_idx] : bounds_np[query_idx + 1]])
        assert np.array_equal(got_np, expected_np)


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
@pytest.mark.parity("closest_point_on_mesh", "meshlib", "pyvista")
def test_closest_point_on_mesh_matches_references(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    200 random queries against trimesh, MeshLib and pyvista: distance, point and face index.

    **trimesh, Class A on the distance, Class C on the point**: ties make the closest *point*
    ambiguous. The distance is the well-defined quantity and is compared directly; a query
    equidistant from two faces has two valid closest points, and section 6 records the same
    divergence against Open3D at ~2e-4. Warp's own ``mesh_query_point_no_sign`` is off by up to
    2.1e-5, which sets the floor here.

    **MeshLib, Class A on the distance, Class C on the point, and the face index is a tie-break
    gauge.** ``findProjection`` returns a ``MeshProjectionResult`` carrying the squared distance,
    the projected point and the ``FaceId`` it landed on -- so it is the only reference in this
    module that reports all three. It is per query, so this loops on the MeshLib side and batches on
    ordito's; the benchmark rows do the same, which is why the row prices a Python loop and says so.
    The distances agree to **1.2e-07** and the points to **1.6e-04** on a unit-radius fixture, the
    latter being Warp's own ``mesh_query_point_no_sign`` floor rather than a disagreement about
    geometry (section 6 records the same magnitude against Open3D). The face index is the
    interesting part and it is why this pair is worth having. It differs on **39 %** of 200 random
    queries, and every one of those is a genuine tie: the two faces always share at least one corner
    (57 of 78 share two, i.e. an edge) and the distances differ by at most 1.2e-07. So the assert is
    not "the same face" -- which would be wrong to demand -- but "any disagreement is a tie", which
    is a real constraint a mis-indexed lookup would fail.

    **pyvista, Class A on the distance and the point; the cell index is not comparable.**
    ``find_closest_cell(..., return_closest_point=True)`` walks a ``vtkStaticCellLocator`` and is
    exact in float64 -- against ``igl.point_mesh_squared_distance`` on these same queries it agrees
    to **4.4e-16** (icosahedron) and **8.9e-16** (hemisphere) on both the distance and the point --
    so it is the most accurate closest-point reference in this module. Against ordito: distances to
    1.8e-07 / 2.7e-07 and points to 1.6e-07 / 9.0e-07, the ``float32`` floor. The point residual is
    fixture-dependent and reaches **2.2e-04** on a subdivided icosphere with queries drawn from a
    wider box, which is Warp's own ``mesh_query_point_no_sign`` limit and the same magnitude the
    MeshLib and Open3D pairings record -- so a comparison on another fixture must not tighten this
    tolerance without re-measuring. igl and pyvista agree on the geometry to 1e-16 and still
    disagree on the cell for 37% / 42% of these queries, and on ``icosphere(3)`` every disagreeing
    query's closest point has a minimum barycentric coordinate of ~1e-16: it lies exactly on a
    shared edge. That is structural rather than a fixture accident -- for a query far outside a
    convex mesh the nearest point is a *vertex*, since the vertex normal fans exhaust the sphere of
    directions (their angular defects sum to 4 pi) -- so the tie fraction grows with the query
    radius. The index is therefore left to the MeshLib pair, whose ``findProjection`` reports the
    face on ordito's own soup numbering.

    On ``saddle_graded`` the *point* is compared at 6e-4: on the CPU device ordito's closest point
    sits 1.9e-4 from the float64-exact one (trimesh and pyvista agree to 4e-16) while the distance
    agrees to 1.5e-7 -- the query lands on a needle, where Warp's ``mesh_query_point_no_sign``
    places the point along the needle no better than that (on CUDA, 1.6e-7); on
    ``sphere_irregular`` 2.1e-7 on both devices (5.9e-5 on its 1 500-vertex form). The distances
    keep their 1e-5.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np = _random_queries_np()
    points_wp = points_to_warp(points_np, mesh_wp.device)
    closest_wp, distances_wp, faces_wp = od.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    )
    closest_np, distances_np, faces_np = closest_wp.numpy(), distances_wp.numpy(), faces_wp.numpy()

    # trimesh.
    point_tolerance = 6e-4  # both fixtures have needles; see the docstring
    closest_tm, distance_tm, _triangle_id_tm = tm.proximity.closest_point(mesh_tm, points_np)
    assert np.allclose(closest_np, closest_tm, rtol=point_tolerance, atol=point_tolerance)
    assert np.allclose(distances_np, distance_tm, rtol=1e-5, atol=1e-5)

    # MeshLib.
    mesh_part_ml = mm.MeshPart(trimesh_to_meshlib(mesh_tm))
    projections_ml = [
        mm.findProjection(mm.Vector3f(*point_np.tolist()), mesh_part_ml) for point_np in points_np
    ]
    points_ml = np.array(
        [
            [result.proj.point.x, result.proj.point.y, result.proj.point.z]
            for result in projections_ml
        ]
    )
    distances_ml = np.sqrt(np.array([result.distSq for result in projections_ml]))
    faces_ml = np.array([int(result.proj.face) for result in projections_ml], dtype=np.int32)
    assert np.allclose(distances_np, distances_ml, rtol=1e-5, atol=1e-5)
    assert np.allclose(closest_np, points_ml, rtol=1e-4, atol=max(1e-4, point_tolerance))

    # Every face disagreement is a tie: the same distance, on a face sharing a corner or an edge.
    disagree_np = faces_np != faces_ml
    assert np.allclose(distances_np[disagree_np], distances_ml[disagree_np], atol=1e-5)
    shared_np = np.array(
        [
            len(set(mesh_tm.faces[face_wp]) & set(mesh_tm.faces[face_ml]))
            for face_wp, face_ml in zip(faces_np[disagree_np], faces_ml[disagree_np], strict=True)
        ]
    )
    assert (faces_np == faces_ml).any()  # non-vacuity: the indices do line up in general
    assert shared_np.size == 0 or shared_np.min() >= 1

    # pyvista.
    _cells_pv, closest_pv = cast(
        "tuple[object, NDArray[np.float64]]",
        trimesh_to_pyvista(mesh_tm).find_closest_cell(points_np, return_closest_point=True),
    )
    closest_pv = np.asarray(closest_pv)
    distances_pv = np.linalg.norm(points_np - closest_pv, axis=1)
    assert distances_pv.min() > 0.0  # non-vacuity: no query sits on the surface
    assert np.allclose(distances_np, distances_pv, rtol=1e-5, atol=1e-5)
    assert np.allclose(closest_np, closest_pv, rtol=point_tolerance, atol=point_tolerance)


def test_closest_point_on_mesh_ambiguous_edge(device: str) -> None:
    """
    Class B: at an exact tie, the *distance* is the answer and the winning face is not.

    Two right triangles meet along the y-axis, one in ``z = 0`` and one in ``x = 0``. The query sits
    at ``(-0.25, 0, -0.25)``, which projects into the interior of each at exactly 0.25 -- the shared
    edge is 0.354 away, so the two face projections are the tied minima and nothing separates them.

    The transform is "up to the tie": which of the two a correct implementation returns is a
    traversal order, and ordito's differs between the CPU and CUDA BVH. This used to assert the
    point elementwise against trimesh and pass only because the query carried a ``-1e-9`` nudge
    toward the first face -- which does nothing, since ordito's vertex buffer is ``float32`` and
    ``-0.25 - 1e-9`` rounds to exactly ``-0.25`` there. So the nudge disambiguated trimesh's
    ``float64`` answer and not ordito's, and the test was asserting a coincidence. It failed on the
    CPU device from the day it was written and nobody ran that device.

    Still non-vacuous: the distance is exact, and the point must be one of the two tied projections
    -- the shared-edge answer ``(0, 0, 0)`` at 0.354 and any point off the surface both fail.
    """
    mesh_tm = tm.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        faces=[[0, 1, 2], [0, 1, 3]],
        process=False,
    )
    query_np = np.array([[-0.25, 0.0, -0.25]], dtype=np.float64)
    _closest_tm, distance_tm, _triangle_id_tm = tm.proximity.closest_point(mesh_tm, query_np)

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    query_wp = points_to_warp(query_np, device)
    closest_wp, distance_wp, _triangle_id_wp = od.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, query_wp
    )

    assert np.allclose(distance_tm, 0.25, rtol=1e-5, atol=1e-5)  # the reference is at the tie too
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)
    tied_np = np.array([[-0.25, 0.0, 0.0], [0.0, 0.0, -0.25]])
    assert np.isclose(tied_np, closest_wp.numpy(), rtol=1e-5, atol=1e-5).all(axis=1).any()


def test_closest_point_on_mesh_unreferenced_vertex(device: str) -> None:
    """
    Class A: an unreferenced vertex must not attract the query, on a one-face mesh.

    The stray vertex sits closer to the query than the single face does, so an implementation
    searching *vertices* rather than triangles returns it and fails -- which is exactly the bug
    this exists for. trimesh ignores it too, so the comparison is elementwise.
    """
    query_np = np.array([[-1.0, -1.0, -1.0]], dtype=np.float64)
    mesh_tm = tm.Trimesh(
        vertices=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-0.5, -0.5, -0.5]],
        faces=[[0, 1, 2]],
        process=False,
    )
    closest_tm, distance_tm, triangle_id_tm = tm.proximity.closest_point(mesh_tm, query_np)

    vertices_wp = points_to_warp(mesh_tm.vertices, device)
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), device=device
    )
    query_wp = points_to_warp(query_np, device)
    closest_wp, distance_wp, triangle_id_wp = od.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, query_wp
    )

    assert np.allclose(closest_wp.numpy(), closest_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)
    assert np.array_equal(triangle_id_wp.numpy(), triangle_id_tm)


def _closest_on_edges_np(
    vertices_np: np.ndarray, edges_np: np.ndarray, queries_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exhaustive point-to-segment distance over every edge: the oracle both references match."""
    a_np, b_np = vertices_np[edges_np[:, 0]], vertices_np[edges_np[:, 1]]
    direction_np = b_np - a_np
    coordinate_np = np.clip(
        ((queries_np[:, None, :] - a_np[None]) * direction_np[None]).sum(-1)
        / (direction_np * direction_np).sum(-1)[None],
        0.0,
        1.0,
    )
    projection_np = a_np[None] + coordinate_np[..., None] * direction_np[None]
    distance_np = np.linalg.norm(queries_np[:, None, :] - projection_np, axis=-1)
    nearest_np = distance_np.argmin(1)
    return distance_np, nearest_np, projection_np[np.arange(queries_np.shape[0]), nearest_np]


def _crossed_bars(angle_degrees: float, gap: float = 1.0) -> tuple[tm.Trimesh, tm.Trimesh]:
    """Two thin bars crossing at a height, whose closest points are interior to skew edges."""
    lower_tm = tm.creation.box(extents=[4.0, 0.2, 0.2])
    upper_tm = tm.creation.box(extents=[4.0, 0.2, 0.2])
    upper_tm.apply_transform(
        tm.transformations.rotation_matrix(np.radians(angle_degrees), [0.0, 0.0, 1.0])
    )
    upper_tm.apply_translation([0.0, 0.0, gap])
    return lower_tm, upper_tm


@pytest.mark.parity("mesh_to_mesh_distance", "meshlib")
@pytest.mark.parametrize("angle_degrees", [20.0, 45.0, 60.0])
def test_mesh_to_mesh_distance_matches_meshlib_where_no_vertex_wins(
    device: str, angle_degrees: float
) -> None:
    """
    Class A on the distance, on the configuration that separates this from a vertex query.

    Two thin bars crossing above one another: their closest points lie in the **interiors** of two
    skew edges, so no vertex of either mesh achieves the minimum. Measured at a 1.0 gap and 0.2
    thickness, the true distance is **0.800000** while the smallest vertex-to-surface distance is
    0.938 at 20 degrees, 1.479 at 45 and 1.773 at 60 -- up to **2.2x** too large. A closest-point
    query over vertices would return those numbers, so this fixture is what says the triangle-to-
    triangle leaf test is doing its job rather than being an expensive way to reach the same answer.

    ``findDistance`` agrees to all eight decimals printed on every one of the three angles. The
    non-vacuity assert is the vertex bound itself: the test fails if the fixture ever stops being
    vertex-free, which is the only way these asserts could become trivial.
    """
    lower_tm, upper_tm = _crossed_bars(angle_degrees)
    lower_vertices_wp, lower_faces_wp = _upload(lower_tm, device)
    upper_vertices_wp, upper_faces_wp = _upload(upper_tm, device)
    distance, face_a, face_b = od.proximity.mesh_to_mesh_distance(
        lower_vertices_wp, lower_faces_wp, upper_vertices_wp, upper_faces_wp
    )

    vertex_bound = min(
        float(
            od.reduce.min(
                od.proximity.closest_point_on_mesh(
                    upper_vertices_wp, upper_faces_wp, lower_vertices_wp
                )[1]
            )
        ),
        float(
            od.reduce.min(
                od.proximity.closest_point_on_mesh(
                    lower_vertices_wp, lower_faces_wp, upper_vertices_wp
                )[1]
            )
        ),
    )
    assert vertex_bound > distance + 1e-4  # non-vacuity: no vertex achieves the minimum

    distance_ml = _meshlib_distance(lower_tm, upper_tm)
    assert np.isclose(distance, distance_ml, rtol=1e-5, atol=1e-6)
    assert 0 <= face_a < len(lower_tm.faces)
    assert 0 <= face_b < len(upper_tm.faces)


@pytest.mark.parity("mesh_to_mesh_distance", "meshlib")
def test_mesh_to_mesh_distance_gap_overlap_and_contact(device: str) -> None:
    """
    Class A on three regimes, one of which is the exact zero.

    Separated, overlapping and corner-to-corner, because they take different paths: a positive
    distance comes from the fifteen sub-distances, a zero comes from the intersection test
    short-circuiting them, and a corner contact is the case where the two disagree if either is
    wrong about closed versus open triangles. ``findDistance`` matches on all three -- measured
    1.00000000 for two spheres three apart, 0.00000000 for overlapping ones and 0.28284270 for two
    unit boxes offset by ``(1.2, 1.2)``, which is ``0.2 * sqrt(2)`` exactly.

    That last number is why the corner case is here: it is a value only a correct closest-feature
    search produces, where a bounding-box answer would give 0.2 and a centroid answer something else
    entirely.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=2)
    cases = []
    far_tm = tm.creation.icosphere(subdivisions=2)
    far_tm.apply_translation([3.0, 0.0, 0.0])
    cases.append((sphere_tm, far_tm, 1.0))
    overlapping_tm = tm.creation.icosphere(subdivisions=2)
    overlapping_tm.apply_translation([1.5, 0.0, 0.0])
    cases.append((sphere_tm, overlapping_tm, 0.0))
    box_tm = tm.creation.box()
    corner_tm = tm.creation.box()
    corner_tm.apply_translation([1.2, 1.2, 0.0])
    cases.append((box_tm, corner_tm, 0.2 * np.sqrt(2.0)))

    for a_tm, b_tm, expected in cases:
        a_vertices_wp, a_faces_wp = _upload(a_tm, device)
        b_vertices_wp, b_faces_wp = _upload(b_tm, device)
        distance, _face_a, _face_b = od.proximity.mesh_to_mesh_distance(
            a_vertices_wp, a_faces_wp, b_vertices_wp, b_faces_wp
        )
        distance_ml = _meshlib_distance(a_tm, b_tm)
        assert np.isclose(distance, expected, rtol=1e-5, atol=1e-6)
        assert np.isclose(distance, distance_ml, rtol=1e-5, atol=1e-6)


@pytest.mark.parity("mesh_to_mesh_distance", "meshlib")
def test_mesh_to_mesh_distance_when_the_vertex_bound_is_the_answer(device: str) -> None:
    """
    Class A on the regime that the kernel's seeded running minimum could silently break.

    Two icospheres separated along an axis touch closest at a *vertex* of each, so the upper bound
    the vertex query derives is exactly the answer rather than a strict over-estimate. That matters
    because the broad phase seeds its global running minimum with that bound: seeded at exactly
    ``upper_bound ** 2`` the ``>=`` prune discards the very pair achieving it and the call returns
    ``inf``, which is why the seed carries a relative bump. Swept across a gap that closes to zero
    and then reverses into an overlap, so the bump is exercised at, above and below the boundary.

    ``findDistance`` is the oracle for the value; the ``isfinite`` and ``face >= 0`` assertions are
    what catch the pruning failure, since that mode returns ``inf`` rather than a wrong number.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    for gap in (0.5, 0.1, 0.01, 0.0, -0.05):
        shifted_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
        shifted_tm.apply_translation([2.0 + gap, 0.0, 0.0])
        a_vertices_wp, a_faces_wp = _upload(sphere_tm, device)
        b_vertices_wp, b_faces_wp = _upload(shifted_tm, device)
        distance, face_a, face_b = od.proximity.mesh_to_mesh_distance(
            a_vertices_wp, a_faces_wp, b_vertices_wp, b_faces_wp
        )
        distance_ml = _meshlib_distance(sphere_tm, shifted_tm)
        assert np.isfinite(distance), f"gap {gap} pruned the pair achieving the bound"
        assert face_a >= 0
        assert face_b >= 0
        assert np.isclose(distance, max(gap, 0.0), rtol=1e-5, atol=1e-6)
        assert np.isclose(distance, distance_ml, rtol=1e-5, atol=1e-6)


def test_mesh_to_mesh_distance_face_a_tie_break_is_the_lowest_index(device: str) -> None:
    """
    Not a library comparison: pins the documented ``face_a`` tie-break rule.

    Checked against the mesh's own topology, not against a single face the implementation
    happens to prefer. Two icospheres separated so they touch at a single vertex on each side put
    every face
    incident to that vertex -- six, at this subdivision -- in a genuine three-way-or-more tie
    for the global minimum: each of the six touches the other mesh through the identical shared
    point, so each has its own candidate pair at exactly the true minimum distance. The
    docstring promises ``face_a`` is the lowest index among a tied set; this computes that set
    from the mesh's own faces (which ones touch the closest vertex) and asserts the returned
    ``face_a`` is its minimum, rather than merely asserting it is *some* valid tied face.

    The broad phase publishes a running minimum across query-face threads via ``wp.atomic_min``
    so sibling threads can prune against it. Publishing the *exact* value would let a thread
    whose own tied candidate has a box gap equal to that value be pruned by the ``>=`` test
    before it reaches the leaf test -- and which of several tied threads loses that race would
    depend on scheduling order, breaking the documented guarantee. ``_GLOBAL_BEST_RELAX``
    (`kernels/proximity.py`) removes the possibility structurally: every publish is strictly
    looser than any tied candidate's exact box gap, so no tied thread can ever be pruned by
    another's publish regardless of interleaving -- which is what makes it sound to assert
    equality against a specific index rather than mere set membership.

    This assertion did not reproduce a wrong ``face_a`` when checked against a deliberately
    unrelaxed build on this hardware, at this scale and at a synthetic scale of 60 simultaneous
    tied clusters (360 tied faces) -- CUDA's block scheduling for these grid sizes appears to
    complete tied threads in an order that already favours the low index, so this is not a proven
    bug-catcher for that specific regression. It still pins a real, falsifiable claim (one of six
    plausible indices), which is why it stays as a positive-value test of the documented contract.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    shifted_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    shifted_tm.apply_translation([2.0, 0.0, 0.0])

    vertices_a = np.asarray(sphere_tm.vertices)
    faces_a = np.asarray(sphere_tm.faces)
    closest_vertex_a = int(np.argmax(vertices_a[:, 0]))
    tied_faces_a = np.flatnonzero((faces_a == closest_vertex_a).any(axis=1))
    assert tied_faces_a.size > 1  # a real multi-face tie, not a vacuous single-face one

    a_vertices_wp, a_faces_wp = numpy_to_warp(vertices_a, faces_a.ravel().astype(np.int32), device)
    b_vertices_wp, b_faces_wp = _upload(shifted_tm, device)
    for _ in range(5):
        distance, face_a, _face_b = od.proximity.mesh_to_mesh_distance(
            a_vertices_wp, a_faces_wp, b_vertices_wp, b_faces_wp
        )
        assert np.isclose(distance, 0.0, atol=1e-6)
        assert face_a == int(tied_faces_a.min())


@pytest.mark.parity(
    "mesh_to_mesh_distance",
    "pymeshlab",
    benchmarked=False,
    reason="Tested but not timed: get_hausdorff_distance samples one surface and "
    "queries the other, so its cost is set by samplenum rather than by the mesh, "
    "and it computes the max and the moments as well. meshlib is the timed oracle.",
)
@pytest.mark.parametrize(
    ("configuration", "expected"),
    [("crossed_bars", 0.8), ("vertex_witness", 2.0), ("rotated", None), ("overlap", 0.0)],
)
def test_mesh_to_mesh_distance_matches_pymeshlab(
    device: str, configuration: str, expected: float | None
) -> None:
    """
    Class B: the same minimum, against a sampled reference whose `min` bounds it from above.

    A second oracle for the group, and one that answers a different question than ``findDistance``
    does. ``get_hausdorff_distance`` samples mesh 0 and takes the *exact* point-to-surface distance
    of each sample to mesh 1, so its ``min`` is the smallest of those -- a real distance between the
    surfaces, hence an **upper bound** on their minimum, tight exactly when some sample lands on the
    witness. The named transform is that bound direction; it is asserted separately from the
    equality so a future sampling change degrades the tight assert and not the sound one.

    The bound is tight on all four configurations here, and for two different reasons worth keeping
    apart. For the convex pairs it is tight because the support point of a *polytope* in any
    direction is a vertex, so the witness is a sampled vertex whatever ``samplenum`` is -- measured
    identical at 1 000 and 100 000 samples, and 1.00361180 against ordito's 1.00361184 on the
    rotated pair, where no vertex is axis-aligned. For the crossed bars it is tight because the
    witness is a two-dimensional patch: the lower bar's top face and the upper bar's bottom face are
    parallel and overlap in projection, so face sampling cannot miss it. That second case is the one
    that makes this claim non-redundant with the vertex-only regime -- and note it also means the
    bars are *not* a fixture that separates a sampled reference from an exact one, which is why
    ``findDistance`` remains the oracle for
    [`test_mesh_to_mesh_distance_matches_meshlib_where_no_vertex_wins`][tests.test_proximity.test_mesh_to_mesh_distance_matches_meshlib_where_no_vertex_wins].

    Hence ``samplenum=1000``: measured on all four arms, the reference's ``min`` at 1 000 samples
    is the same double as at 3 000, 10 000 and 100 000, so the margins asserted below are unchanged
    by the smaller count.

    !!! warning "`maxdist` silently returns `inf` past its default cap"
        The default ``maxdist`` is a percentage of the bounding-box diagonal, and a pair separated
        further than it comes back with ``min`` of ``inf`` -- no exception, no warning. Measured:
        two unit spheres at a 2.0 gap read ``inf`` at every ``samplenum`` until ``maxdist`` is
        passed explicitly, at which point they read 2.00000000. So the ``vertex_witness`` row is
        exactly the configuration that a defaulted call reports as unreachable, and it is in the
        parametrization for that reason.
    """
    if configuration == "crossed_bars":
        a_tm, b_tm = _crossed_bars(45.0)
    else:
        a_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
        b_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
        if configuration == "rotated":
            b_tm.apply_transform(tm.transformations.rotation_matrix(0.37, [0.3, 1.0, 0.2]))
            b_tm.apply_translation([3.0, 0.0, 0.0])
        else:
            b_tm.apply_translation([4.0 if configuration == "vertex_witness" else 1.0, 0.0, 0.0])

    a_vertices_wp, a_faces_wp = _upload(a_tm, device)
    b_vertices_wp, b_faces_wp = _upload(b_tm, device)
    distance, _face_a, _face_b = od.proximity.mesh_to_mesh_distance(
        a_vertices_wp, a_faces_wp, b_vertices_wp, b_faces_wp
    )

    meshset_pml = ml.MeshSet()
    for mesh_tm in (a_tm, b_tm):
        meshset_pml.add_mesh(
            ml.Mesh(
                np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
                np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            )
        )
    minimum_pml = float(
        meshset_pml.get_hausdorff_distance(
            sampledmesh=0,
            targetmesh=1,
            samplevert=True,
            sampleedge=True,
            sampleface=True,
            samplenum=1000,
            maxdist=ml.PureValue(1.0e6),
        )["min"]
    )

    assert np.isfinite(minimum_pml), "maxdist capped the search; see the warning above"
    if expected is not None:
        assert np.isclose(distance, expected, rtol=1e-5, atol=1e-6)
    assert minimum_pml >= distance - 1e-5  # a sampled minimum cannot beat the true one
    assert np.isclose(distance, minimum_pml, rtol=1e-5, atol=1e-6)


def test_mesh_to_mesh_distance_tiled_pass_agrees_with_the_capped_one(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the block-cooperative second pass against the thread pass alone.

    ``mesh_to_mesh_distance`` splits its broad phase in two on CUDA -- a thread per query face,
    capped at ``_QUERY_CANDIDATE_CAP`` candidates, then a whole block per face that exceeded the
    cap -- because the traversal is a long tail on a flat floor (98 % of faces return no candidate,
    0.5 % carry half the work). The oracle for the *value* is
    ``test_mesh_to_mesh_distance_when_the_vertex_bound_is_the_answer`` above, which compares against
    MeshLib; what this pins is that routing a face through the tiled walk instead of the serial one
    does not change the answer.

    The cap is driven to ``1`` so that every face with any candidate at all overflows, which is the
    configuration that exercises the tiled kernel hardest; the reference run raises it past any
    possible candidate count so the tiled pass never launches. On the cpu device the split is
    disabled outright (``wp.launch_tiled`` runs one lane per block there) and both runs take the
    same path, which makes this a no-op rather than a skip.

    The witness ``face_b`` is deliberately **not** compared: the two walks tie-break differently and
    the ties are real, which the function's own note measures.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    shifted_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    shifted_tm.apply_translation([2.05, 0.3, 0.0])
    a_vertices_wp, a_faces_wp = _upload(sphere_tm, device)
    b_vertices_wp, b_faces_wp = _upload(shifted_tm, device)
    tiled_launches = []
    tiled_kernel = kernel_proximity.face_to_mesh_distance_tiled

    def counting(real_launch_tiled: Callable[..., object]):
        def counting_launch_tiled(*args: object, **kwargs: object):
            # Keyed on the kernel, not on the call: ``od.reduce.min`` derives this function's
            # upper bound through a tiled launch of its own, so counting every ``launch_tiled``
            # would make the non-vacuity assertion below pass with the split disabled --
            # measured, it did.
            if kwargs.get("kernel", args[0] if args else None) is tiled_kernel:
                tiled_launches.append(kwargs.get("dim", None))
            return real_launch_tiled(*args, **kwargs)

        return counting_launch_tiled

    # The wrapper launches through ``ordito._launch``, whose CPU path falls back to
    # ``wp.launch_tiled``; wrap both, the cached one only when it is reached directly.
    with monkeypatch.context() as patch:
        patch.setattr(od.proximity, "_QUERY_CANDIDATE_CAP", 1)
        patch.setattr(_launch, "launch_tiled", counting(_launch.launch_tiled))
        patch.setattr(wp, "launch_tiled", counting(wp.launch_tiled))
        split = od.proximity.mesh_to_mesh_distance(
            a_vertices_wp, a_faces_wp, b_vertices_wp, b_faces_wp
        )
    with monkeypatch.context() as patch:
        patch.setattr(od.proximity, "_QUERY_CANDIDATE_CAP", 1 << 30)
        whole = od.proximity.mesh_to_mesh_distance(
            a_vertices_wp, a_faces_wp, b_vertices_wp, b_faces_wp
        )

    if wp.get_device(device).is_cuda:
        # Non-vacuity: without this the test passes when the tiled pass never runs at all.
        assert tiled_launches, "a cap of 1 must reach the tiled pass at all"
        assert tiled_launches[0], "the tiled pass must be handed a non-empty overflow list"
    assert np.isclose(split[0], whole[0], rtol=1e-6, atol=1e-9)
    assert split[1] == whole[1]
    assert split[0] > 0.0


@pytest.mark.parametrize("side", [8, 16, 32])
def test_mesh_to_mesh_distance_when_every_face_overflows_the_cap(device: str, side: int) -> None:
    """
    Not a library comparison: the answer is exact by construction, and it used to be ``inf``.

    Two flat unit sheets, one lifted 0.3 in ``z``, so every closest pair is exactly 0.300000 and no
    reference is needed to know it. What the fixture is *for* is the second pass: a query face's
    box grown by the 0.3 bound covers a large disc of the other sheet, so from ``side = 16``
    (512 faces) **every** face overflows ``_QUERY_CANDIDATE_CAP`` and every face is re-walked --
    the configuration the scan meshes never reach, where only 0.02-0.5 % of faces overflow.

    What it pins is the second pass's **publish rule**: it must keep the better of its own answer
    and the grid pass's, not overwrite. Its prune drops a candidate whose box gap is ``>=`` the
    running global minimum, and that minimum is by then the answer itself -- published by this very
    face moments earlier -- so an overwriting re-walk prunes the winning pair, returns ``inf``, and
    replaces 0.3 with it. The cpu device runs no second pass at all (the cap is disabled there), so
    it answered correctly throughout and is the arm that named the bug.

    Parametrized across the transition on purpose. Mutation-probed: restoring the overwrite fails
    ``side = 16`` and ``32`` and **passes** ``side = 8``, which overflows nothing -- so a regression
    here reads as a size-dependent failure, and a fixture that stopped saturating the pass would
    stop testing anything.

    It does **not** cover the other defect this pair of sheets was found alongside --
    ``wp.tile_bvh_query_aabb`` returning out-of-range primitive indices once a traversal round
    overruns its shared result buffer (CLAUDE.md section 12.2). Checked rather than assumed:
    ``compute-sanitizer`` reports **0 errors** on this fixture with the kernel's index guard removed
    as well as with it, at every ``side`` here, so the overrun is not reachable at this size. That
    defect's evidence is a benchmark-scale mesh and it has no test.
    """
    positions = np.linspace(0.0, 1.0, side + 1)
    grid_x, grid_y = np.meshgrid(positions, positions, indexing="ij")
    vertices_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1).astype(
        np.float64
    )
    corners = np.arange((side + 1) ** 2).reshape(side + 1, side + 1)
    lower_left, lower_right = corners[:-1, :-1], corners[1:, :-1]
    upper_right, upper_left = corners[1:, 1:], corners[:-1, 1:]
    faces_np = np.concatenate(
        [
            np.stack([lower_left, lower_right, upper_right], axis=-1).reshape(-1, 3),
            np.stack([lower_left, upper_right, upper_left], axis=-1).reshape(-1, 3),
        ]
    ).ravel()

    gap = 0.3
    sheet_vertices_wp, sheet_faces_wp = numpy_to_warp(
        vertices_np, faces_np.astype(np.int32), device
    )
    lifted_vertices_wp, lifted_faces_wp = numpy_to_warp(
        vertices_np + np.array([0.0, 0.0, gap]), faces_np.astype(np.int32), device
    )
    distance, face_a, face_b = od.proximity.mesh_to_mesh_distance(
        sheet_vertices_wp, sheet_faces_wp, lifted_vertices_wp, lifted_faces_wp
    )
    assert np.isclose(distance, gap, rtol=1e-6, atol=1e-7)
    assert face_a >= 0
    assert face_b >= 0


def test_mesh_to_mesh_distance_upper_bound_and_edge_cases(device: str) -> None:
    """
    Not a library comparison: what ``upper_bound`` does, including when it is wrong.

    A sound bound must not change the answer, and a **too-small** one is documented to give a wrong
    answer rather than a slow one -- so that is asserted, because a silent contract is worse than a
    loud one and a reader deserves to see the failure mode demonstrated. An empty mesh returns
    ``inf`` with no witness.
    """
    lower_tm, upper_tm = _crossed_bars(45.0)
    lower_vertices_wp, lower_faces_wp = _upload(lower_tm, device)
    upper_vertices_wp, upper_faces_wp = _upload(upper_tm, device)
    exact, _face_a, _face_b = od.proximity.mesh_to_mesh_distance(
        lower_vertices_wp, lower_faces_wp, upper_vertices_wp, upper_faces_wp
    )
    generous, _a, _b = od.proximity.mesh_to_mesh_distance(
        lower_vertices_wp, lower_faces_wp, upper_vertices_wp, upper_faces_wp, upper_bound=10.0
    )
    assert np.isclose(generous, exact, rtol=1e-6)

    # Documented: a bound below the true distance culls the winning pair.
    starved, _a, _b = od.proximity.mesh_to_mesh_distance(
        lower_vertices_wp, lower_faces_wp, upper_vertices_wp, upper_faces_wp, upper_bound=0.0
    )
    assert starved > exact

    empty_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    empty_faces_wp = warp_empty(0, wp.int32, device)
    assert od.proximity.mesh_to_mesh_distance(
        lower_vertices_wp, lower_faces_wp, empty_vertices_wp, empty_faces_wp
    ) == (math.inf, -1, -1)
    with pytest.raises(ValueError, match="upper_bound must be non-negative"):
        od.proximity.mesh_to_mesh_distance(
            lower_vertices_wp, lower_faces_wp, upper_vertices_wp, upper_faces_wp, upper_bound=-1.0
        )


@pytest.mark.parity("closest_point_on_edges", "pyvista")
def test_closest_point_on_edges_matches_pyvista(
    device: str, unit_box: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A: distances and closest points against VTK's wireframe query, plus a brute-force argmin.

    The edge set is ``unit_box``'s 12 creases -- a real edge set rather than random segments,
    because the interesting queries are the ones whose answer is a shared *vertex* of two edges,
    which is where a wireframe query differs from a surface one. VTK's ``find_closest_cell`` on a
    one-cell-per-edge ``PolyData`` is exact here (probed at 0.0 distance error).

    The **edge id is compared only where the minimum is unique.** On any edge set sharing vertices,
    several edges achieve the minimum for a query near a shared vertex: measured on a random
    400-edge set, 144 of 500 queries had 2-5 tied edges and every disagreement was a tie to 4.5e-08.
    So the id claim is that the returned edge *achieves* the minimum, which is tie-robust, plus an
    equality on the queries whose argmin is unique.
    """
    mesh_tm, _ = unit_box
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    edges_wp = od.seams.crease_edges(
        *numpy_to_warp(vertices_np, np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device)
    )
    edges_np = edges_wp.numpy()
    assert edges_wp.shape[0] == 12  # the box's creases, and non-empty before anything is compared

    rng = np.random.default_rng(21)
    queries_np = (rng.random((200, 3)) * 1.6 - 0.3).astype(np.float32)
    queries_wp = points_to_warp(queries_np, device)
    vertices_wp = points_to_warp(vertices_np, device)

    closest_wp, distance_wp, edge_id_wp = od.proximity.closest_point_on_edges(
        vertices_wp, edges_wp, queries_wp
    )
    all_distance_np, nearest_np, closest_np = _closest_on_edges_np(
        vertices_np.astype(np.float32), edges_np, queries_np
    )
    minimum_np = all_distance_np[np.arange(queries_np.shape[0]), nearest_np]

    assert np.allclose(distance_wp.numpy(), minimum_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(closest_wp.numpy(), closest_np, rtol=1e-5, atol=1e-5)

    # The returned edge achieves the minimum, whether or not it is the argmin NumPy picked.
    edge_id_np = edge_id_wp.numpy()
    assert np.all(edge_id_np >= 0)
    achieved_np = all_distance_np[np.arange(queries_np.shape[0]), edge_id_np]
    assert np.allclose(achieved_np, minimum_np, rtol=1e-5, atol=1e-5)
    unique_np = (np.abs(all_distance_np - minimum_np[:, None]) < 1e-6).sum(1) == 1
    assert unique_np.sum() > 100  # most queries have a unique nearest edge, so this is not vacuous
    assert np.array_equal(edge_id_np[unique_np], nearest_np[unique_np])

    # VTK: one line cell per edge, so the cell id is the edge id.
    cells_np = np.hstack([np.full((edges_wp.shape[0], 1), 2), edges_np]).astype(np.int64).ravel()
    wireframe_pv = pv.PolyData(vertices_np, lines=cells_np)
    assert wireframe_pv.n_cells == edges_wp.shape[0]
    cell_pv, point_pv = cast(
        "tuple[NDArray[np.int_], NDArray[np.float64]]",
        wireframe_pv.find_closest_cell(queries_np.astype(np.float64), return_closest_point=True),
    )
    assert np.allclose(
        np.linalg.norm(np.asarray(point_pv) - queries_np, axis=1), minimum_np, rtol=1e-5, atol=1e-5
    )
    assert np.allclose(
        all_distance_np[np.arange(queries_np.shape[0]), np.asarray(cell_pv)],
        minimum_np,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parity(
    "closest_point_on_edges",
    "meshlib",
    benchmarked=False,
    reason="findProjectionOnMeshEdges answers one query per call, so a batched row would time a "
    "Python loop over the query set rather than the traversal; pyvista's find_closest_cell is "
    "batched and carries the timed row for this group.",
)
def test_closest_point_on_edges_matches_meshlib(
    device: str, unit_box: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class B: the same distances as ``findProjectionOnMeshEdges`` after a square root.

    Separate from the VTK comparison because this is the reference that cannot be benchmarked -- one
    query per call -- and because its input is a *mesh plus an edge subset* rather than a standalone
    wireframe: ``AABBTreePolyline3(mesh, UndirectedEdgeBitSet)`` is how a crease set is expressed
    there, so the set has to be translated edge by edge through ``topology.findEdge``. The result's
    ``distSq`` is squared, which is the named transform.

    ``upDistLimitSq`` is passed as float32's max rather than ``inf``: an infinite limit segfaults
    inside MeshLib's projection code (CLAUDE.md section 7.6), which is a crash rather than an
    exception.
    """
    mesh_tm, _ = unit_box
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    edges_wp = od.seams.crease_edges(
        *numpy_to_warp(vertices_np, np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device)
    )
    edges_np = edges_wp.numpy()
    assert edges_wp.shape[0] == 12

    rng = np.random.default_rng(22)
    queries_np = (rng.random((64, 3)) * 1.6 - 0.3).astype(np.float32)
    _all_distance_np, nearest_np, _closest_np = _closest_on_edges_np(
        vertices_np.astype(np.float32), edges_np, queries_np
    )
    minimum_np = _all_distance_np[np.arange(queries_np.shape[0]), nearest_np]

    _closest_wp, distance_wp, _edge_wp = od.proximity.closest_point_on_edges(
        points_to_warp(vertices_np, device), edges_wp, points_to_warp(queries_np, device)
    )

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    crease_ml = mm.UndirectedEdgeBitSet()
    crease_ml.resize(mesh_ml.topology.undirectedEdgeSize())
    for edge_np in edges_np:
        edge_id_ml = mesh_ml.topology.findEdge(
            mm.VertId(int(edge_np[0])), mm.VertId(int(edge_np[1]))
        )
        assert edge_id_ml.valid()
        crease_ml.set(edge_id_ml.undirected())
    tree_ml = mm.AABBTreePolyline3(mesh_ml, crease_ml)
    distance_ml = np.array(
        [
            float(
                np.sqrt(
                    mm.findProjectionOnMeshEdges(
                        mm.Vector3f(*query_np.tolist()),
                        mesh_ml,
                        tree_ml,
                        float(np.finfo(np.float32).max),
                    ).distSq
                )
            )
            for query_np in queries_np
        ]
    )
    assert np.allclose(distance_ml, minimum_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(distance_wp.numpy(), distance_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "closest_point_on_edges",
    "pytorch3d",
    benchmarked=False,
    reason="point_mesh_edge_distance reads Meshes.edges_packed -- the mesh's whole undirected edge "
    "set -- and takes no edge subset, so it cannot be handed the crease set both rows of the "
    "closest_point_on_edges group share. A row would price a 19.6x larger input on "
    "bunny_decimated (24 363 edges against 1 242 creases) and 76.9x on bunny (104 288 against "
    "1 357) under that group's name. The values are still comparable on the full edge set, which "
    "is what this test does.",
)
def test_closest_point_on_edges_matches_pytorch3d(device: str) -> None:
    """
    Class B: ``loss.point_mesh_edge_distance`` minus its edge-to-point half is ordito's answer.

    The same decomposition as ``tests/test_metrics.py``'s ``point_mesh_face_distance`` pair and the
    same measured restriction. pytorch3d returns the sum of two squared means over
    ``Meshes.edges_packed`` -- point-to-segment, which is what ``closest_point_on_edges`` computes,
    and **edge-to-point**, the minimum over the cloud per edge -- and ordito has no counterpart for
    the second, so only the forward half is compared.

    The point-to-segment table is exact rather than a port: clamping the projection parameter to
    ``[0, 1]`` *is* the algorithm, with no region classification to get wrong. It reproduces
    pytorch3d's whole scalar to **9.7e-10** when its two directions are summed, which is what
    licenses the subtraction; ordito's forward mean then lands 4.9e-08 from its forward column.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(5)
    queries_np = (rng.normal(size=(300, 3)) * 0.9).astype(np.float32)
    mesh_p3d = trimesh_to_pytorch3d(mesh_tm, device)
    scalar_p3d = float(
        p3d_loss.point_mesh_edge_distance(mesh_p3d, points_to_pytorch3d(queries_np, device=device))
    )
    edges_t = mesh_p3d.edges_packed()
    assert edges_t is not None
    edges_p3d = edges_t.cpu().numpy()

    starts_np = np.asarray(mesh_tm.vertices[edges_p3d[:, 0]], dtype=np.float64)
    directions_np = np.asarray(mesh_tm.vertices[edges_p3d[:, 1]], dtype=np.float64) - starts_np
    parameters_np = np.clip(
        np.einsum("ijk,jk->ij", queries_np[:, None, :] - starts_np, directions_np)
        / np.einsum("jk,jk->j", directions_np, directions_np),
        0.0,
        1.0,
    )
    squared_np = np.sum(
        (queries_np[:, None, :] - (starts_np + parameters_np[..., None] * directions_np)) ** 2,
        axis=-1,
    )
    forward_np, backward_np = squared_np.min(axis=1).mean(), squared_np.min(axis=0).mean()

    assert np.allclose(scalar_p3d, forward_np + backward_np, rtol=1e-6, atol=0.0)

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    edges_wp, _ = od.edges.edges_unique(faces_wp)
    _, distances_wp, _ = od.proximity.closest_point_on_edges(
        vertices_wp, edges_wp, points_to_warp(queries_np, device)
    )

    assert np.allclose(
        float((distances_wp.numpy().astype(np.float64) ** 2).mean()),
        scalar_p3d - backward_np,
        rtol=1e-5,
        atol=1e-7,
    )


def test_closest_point_on_edges_max_dist_and_degenerate(device: str) -> None:
    """
    Not a library comparison: the ``max_dist`` cutoff, the empty inputs and the column guard.

    A miss reports the query itself with ``max_dist`` and ``-1``, matching
    ``closest_point_on_mesh``'s kernel-level convention -- pinned here because a caller reading only
    the distance would otherwise see a plausible number for a query that found nothing.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    edges_np = np.array([[0, 1], [0, 2]], dtype=np.int32)
    vertices_wp = points_to_warp(vertices_np, device)
    edges_wp = odt.as_array2d(wp.array(edges_np, dtype=wp.int32, device=device), wp.int32)
    # The second query is 0.9 from the y-axis edge and 3.0 from the x-axis one, so a 0.5 cutoff
    # rejects both. (At x = 0.5 it would sit at *exactly* 0.5 and be accepted -- the test is
    # inclusive, like every other distance bound here.)
    queries_np = np.array([[0.5, 0.1, 0.0], [0.9, 3.0, 0.0]], dtype=np.float32)
    queries_wp = points_to_warp(queries_np, device)

    closest_wp, distance_wp, edge_id_wp = od.proximity.closest_point_on_edges(
        vertices_wp, edges_wp, queries_wp, max_dist=0.5
    )
    assert np.array_equal(edge_id_wp.numpy(), np.array([0, -1]))
    assert np.isclose(distance_wp.numpy()[0], 0.1, rtol=1e-5, atol=1e-5)
    assert distance_wp.numpy()[1] == np.float32(0.5)  # the cutoff itself, on a miss
    assert np.allclose(closest_wp.numpy()[1], queries_np[1])  # and the query point itself

    empty_closest_wp, empty_distance_wp, empty_edge_wp = od.proximity.closest_point_on_edges(
        vertices_wp, edges_wp, warp_empty(0, wp.vec3, device)
    )
    assert empty_closest_wp.shape == (0,)
    assert empty_distance_wp.shape == (0,)
    assert empty_edge_wp.shape == (0,)

    no_edges_wp = odt.as_array2d(warp_empty((0, 2), wp.int32, device), wp.int32)
    _closest_wp, distance_wp, edge_id_wp = od.proximity.closest_point_on_edges(
        vertices_wp, no_edges_wp, queries_wp
    )
    assert np.all(np.isinf(distance_wp.numpy()))
    assert np.all(edge_id_wp.numpy() == -1)

    with pytest.raises(ValueError, match="shape \\(k, 2\\)"):
        od.proximity.closest_point_on_edges(
            vertices_wp,
            odt.as_array2d(wp.zeros((2, 3), dtype=wp.int32, device=device), wp.int32),
            queries_wp,
        )


def test_normals_at_closest_faces(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = sphere_irregular
    rng = np.random.default_rng(19)
    query_np = rng.random((32, 3)).astype(np.float64)

    query_wp = points_to_warp(query_np, mesh_wp.device)
    normals_wp = od.proximity.normals_at_closest_faces(mesh_wp, query_wp).numpy()

    _, _, triangle_id_wp = od.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, query_wp
    )
    all_normals_wp, _ = od.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    expected_normals_np = all_normals_wp.numpy()[triangle_id_wp.numpy()]
    assert np.allclose(normals_wp, expected_normals_np, rtol=1e-5, atol=1e-5)


def test_normals_at_closest_faces_surface(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = sphere_irregular
    points_np, face_ids_np = tm.sample.sample_surface(mesh_tm, 24, seed=3)[:2]
    expected_normals_np = mesh_tm.face_normals[face_ids_np].astype(np.float32)

    points_wp = points_to_warp(points_np, mesh_wp.device)
    normals_wp = od.proximity.normals_at_closest_faces(mesh_wp, points_wp).numpy()
    assert np.allclose(normals_wp, expected_normals_np, rtol=1e-5, atol=1e-5)


def test_normals_at_closest_faces_empty(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = sphere_irregular
    points_wp = warp_empty(0, wp.vec3, mesh_wp.device)
    normals_wp = od.proximity.normals_at_closest_faces(mesh_wp, points_wp)
    assert normals_wp.shape == (0,)


def test_normals_at_closest_faces_empty_mesh() -> None:
    """A zero-face mesh reports NaN normals rather than segfaulting."""
    # A zero-face wp.Mesh is unsafe to build on CUDA (§12.1), so this is CPU-only by construction,
    # not by a `device` fixture choice: reproducing the crash this guards against needs the real
    # object, and building it on CUDA would corrupt allocator state rather than reproduce anything.
    device = wp.get_device("cpu")
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = warp_empty(0, wp.int32, device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)
    points_wp = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    normals_wp = od.proximity.normals_at_closest_faces(mesh_wp, points_wp)
    assert normals_wp.shape == (2,)
    assert np.all(np.isnan(normals_wp.numpy()))


@pytest.mark.parametrize("mesh_name", _SIGNED_DISTANCE_MESHES)
def test_signed_distance_on_mesh_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (a sign negation): both sign modes against trimesh's opposite convention.

    trimesh is positive *inside*, ordito negative, so the reference is negated -- the named
    transform. On a watertight mesh the ray-parity default and ``sign_mode="winding"`` must both
    match it, and between themselves only the sign may differ: the unsigned distance is the same
    query (ordito against ordito).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np = _random_queries_np()
    expected_np = -tm_proximity.signed_distance(mesh_tm, points_np)
    points_wp = points_to_warp(points_np, mesh_wp.device)

    parity_wp = od.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)
    winding_wp = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
    )
    assert np.allclose(parity_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(winding_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.abs(parity_wp.numpy()), np.abs(winding_wp.numpy()), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("mesh_name", "compare_pymeshlab"),
    [("sphere_irregular", False), ("sphere_irregular_hollow", False), ("torus_irregular", True)],
)
@pytest.mark.parity("signed_distance_on_mesh", "pymeshlab", "igl", "open3d")
def test_signed_distance_on_mesh_matches_pymeshlab_igl_and_open3d(
    request: pytest.FixtureRequest, mesh_name: str, compare_pymeshlab: bool
) -> None:
    """
    Three more sign rules that nevertheless return the same signed number on watertight input.

    The fixture set is chosen to put the normal-based rules under load rather than to flatter them:
    ``sphere_irregular_hollow`` has a cavity, so points inside it have a nearest face whose normal
    faces the other way from the outer shell's, and ``torus_irregular`` is genus 1, where a point in
    the hole is outside the solid but surrounded by surface. Both are exactly where a normal-based
    sign is supposed to be unreliable -- and where a parity rule is exactly what should *not* fail.
    Every comparison asserts the signs equal, then the values.

    **pymeshlab, Class A**: a **third** sign convention. ordito's default mode casts perturbed
    parity rays and MeshLab signs by the dot product with the *closest point's normal*, so a priori
    this is the pair most at risk of a systematic sign flip. On ``torus_irregular`` it does not
    flip: sign agreement **400 / 400** and the signed values match to **1.1e-07**, so the assert is
    a direct ``allclose`` with no transform on the value at all (the comment below says why the two
    spheres are not compared). Two
    named transforms on the *plumbing*, not the value: the query points go in as a second,
    face-less mesh (``measuremesh=1``, ``refmesh=0``) and the answer is read off that mesh's
    ``vertex_scalar_array()``. ``signeddist=True`` is what makes it signed rather than absolute, and
    is the parameter the benchmark passes.

    **igl, Class B (a named sign-rule choice)**: a direct ``allclose`` against the pseudonormal
    sign type. igl's angle-weighted pseudonormal sign is a *fourth* rule beside ordito's parity
    rays, trimesh's and MeshLab's closest-point normal, and on these three watertight fixtures it
    agrees with ordito on every sign and to 2.5-3.2e-07 in value -- so the comparison needs no
    transform. The named choice is
    *which* ``sign_type`` is the oracle, and it is not free: for both ``WINDING_NUMBER`` and
    ``FAST_WINDING_NUMBER`` igl returns ``(1 - 2 * w) * d`` with ``w`` the **continuous** winding
    number, not ``sign(1 - 2 * w) * d``. Its magnitude is therefore ``|d|`` only where ``w`` is
    exactly 0 or 1 and is scaled down near the surface -- measured on ``bunny_decimated``, ``|S|``
    departs from the pseudonormal type's by 2.7e-2 of the bbox diagonal where the pseudonormal type
    agrees with ordito to 8e-8. So the winding types cannot be compared on value at all, and the
    benchmark's ``sign_mode="winding"`` igl row is a cost comparison only.

    **open3d, Class A**: Embree's ``compute_signed_distance`` signs by counting ray crossings --
    the same rule as ordito's default ``"parity"`` mode -- and uses the same Warp-SDF orientation
    (negative inside), so no transform is needed on either the sign or the value: every sign equal
    and the values within 3.0-4.5e-07 on the three fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np = _queries_in_box_np(mesh_tm, 400, seed=4)
    points_wp = points_to_warp(points_np, mesh_wp.device)
    signed_np = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    ).numpy()

    # pymeshlab is the inaccurate side on both irregular spheres, where igl, open3d and ordito
    # agree to 3e-7: 7.0e-4 off igl's exact pseudonormal distance on ``sphere_irregular``'s
    # needles, and on ``sphere_irregular_hollow`` one query of 400 signed wrongly (+0.249 against
    # -0.249): its sign is the closest face's normal, which a concave edge or vertex as the closest
    # feature defeats. On ``torus_irregular`` it agrees to 1.1e-7, so that is where it is compared.
    if compare_pymeshlab:
        meshset_pml = trimesh_to_pymeshlab(mesh_tm)
        meshset_pml.add_mesh(ml.Mesh(vertex_matrix=np.ascontiguousarray(points_np)))
        meshset_pml.compute_scalar_by_distance_from_another_mesh_per_vertex(
            measuremesh=1, refmesh=0, signeddist=True
        )
        signed_pml = np.asarray(meshset_pml.current_mesh().vertex_scalar_array())
        assert np.array_equal(np.sign(signed_np), np.sign(signed_pml))
        assert np.allclose(signed_np, signed_pml, rtol=1e-5, atol=1e-5)

    signed_igl, _, _, _ = igl.signed_distance(
        np.ascontiguousarray(points_np), *mesh_igl(mesh_tm), igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    assert np.array_equal(np.sign(signed_np), np.sign(signed_igl))
    assert np.allclose(signed_np, signed_igl, rtol=1e-5, atol=1e-5)

    scene_o3d = o3d.t.geometry.RaycastingScene()
    scene_o3d.add_triangles(trimesh_to_open3d_t(mesh_tm))
    signed_o3d = scene_o3d.compute_signed_distance(
        o3d.core.Tensor(np.ascontiguousarray(points_np, dtype=np.float32))  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
    ).numpy()
    assert np.array_equal(np.sign(signed_np), np.sign(signed_o3d))
    assert np.allclose(signed_np, signed_o3d, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _SIGNED_DISTANCE_MESHES)
@pytest.mark.parity("signed_distance_on_mesh", "meshlib")
def test_signed_distance_on_mesh_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A, sign included: MeshLib shares ordito's convention exactly, and trimesh's does not.

    ``findSignedDistances(refMesh, testPoints)`` is the batched form and the one used here; the
    per-query ``signedDistanceToMesh`` gives the identical numbers. Both are **negative inside**,
    which is Warp's SDF convention and open3d's, against trimesh's opposite one -- so this pair
    needs no negation and is asserted without one, which is what makes it a check on the sign rather
    than a restatement of it.

    Two options that had to be set rather than accepted. ``maxDistSq`` defaults to ``FLT_MAX`` but
    ``nullOutsideMinMax`` is ``True``, so a query outside the band comes back as a null rather than
    a distance; and ``signMode`` defaults to ``ProjectionNormal``, which is neither of ordito's two
    modes by construction. Measured on ``cave_cube``, all three of ``ProjectionNormal``,
    ``WindingRule`` and ``HoleWindingRule`` agreed with ordito's parity mode to **0.0** on 300
    queries spanning the cavity, so the default is used; on the irregular fixtures it agrees on
    every sign and to 3.1e-07 / 3.6e-07 (``sphere_irregular`` / ``_hollow``). A cavity is what
    makes that non-trivial, since a convex mesh cannot separate a projection-normal sign from a
    parity one.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np = _queries_in_bounds_np(mesh_tm, 200, seed=42)
    points_wp = points_to_warp(points_np, mesh_wp.device)

    cloud_ml = mn.pointCloudFromPoints(cast("Buffer", np.ascontiguousarray(points_np)))
    distances_ml = meshlib_scalars_to_numpy(
        mm.findSignedDistances(trimesh_to_meshlib(mesh_tm), cloud_ml.points)
    )
    distances_wp = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    ).numpy()

    assert (distances_ml < 0).any()  # non-vacuity: some query is inside, so the sign is exercised
    assert (distances_ml > 0).any()
    assert np.allclose(distances_wp, distances_ml, rtol=1e-5, atol=1e-5)
    # The convention, stated as an assert: trimesh's sign is the other one.
    assert np.allclose(distances_wp, -tm_proximity.signed_distance(mesh_tm, points_np), atol=1e-4)


@pytest.mark.parametrize("mesh_name", _SIGNED_DISTANCE_MESHES_TORUS)
@pytest.mark.parity("signed_distance_on_mesh", "pyvista")
def test_signed_distance_on_mesh_matches_pyvista(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: ``vtkImplicitPolyDataDistance`` is an exact SDF and shares ordito's sign convention.

    This is the strongest pyvista row in the suite and the reason the registration was worth having:
    negative inside on both sides with no negation, correlation 1.0000000, **max absolute difference
    4.1e-07 to 4.6e-07** and identical signs on 400 queries on each fixture. It comes back float64,
    so the residual is ordito's ``float32`` vertex buffer.

    Not to be confused with vedo's ``Mesh.signed_distance``, which is ``vtkSignedDistance`` -- a
    tangent-plane point-cloud estimator that correlates 0.956 with a max absolute difference of
    0.375 on the same input, and is a class-D row rather than this one.

    The three fixtures are the ones a parity sign rule must not fail on, as in the open3d
    comparison above; ``sphere_irregular_hollow``'s cavity is signed *outside* by both libraries.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(5)
    centre_np = mesh_tm.bounds.mean(axis=0)
    extent_np = 1.4 * (mesh_tm.bounds[1] - mesh_tm.bounds[0])
    points_np = centre_np - 0.5 * extent_np + rng.random((400, 3)) * extent_np

    signed_pv = np.asarray(
        points_to_pyvista(points_np)
        .compute_implicit_distance(trimesh_to_pyvista(mesh_tm))
        .point_data["implicit_distance"]
    )
    # Both signs present, so the sign comparison is not riding on a constant.
    assert (signed_pv < 0).any()
    assert (signed_pv > 0).any()

    points_wp = points_to_warp(points_np, mesh_wp.device)
    signed_wp = od.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)

    assert np.array_equal(np.sign(signed_wp.numpy()), np.sign(signed_pv))
    assert np.allclose(signed_wp.numpy(), signed_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "sphere_irregular_hollow", "saddle_graded"]
)
def test_signed_distance_on_mesh_winding_sign_matches_exact_winding_number(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The builtin's Barnes-Hut sign must match thresholding the exact solid-angle sum.

    This is the property that makes ``sign_mode="winding"`` worth having: it holds on the open
    fixture (``saddle_graded``) too, where ray parity has no principled answer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(7)
    lower_np, upper_np = mesh_tm.bounds
    points_np = rng.uniform(lower_np, upper_np, size=(500, 3))
    points_wp = points_to_warp(points_np, mesh_wp.device)

    inside_wp = (
        od.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
        ).numpy()
        < 0.0
    )
    inside_exact = (
        od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, points_wp).numpy() > 0.5
    )
    assert np.array_equal(inside_wp, inside_exact)


def test_signed_distance_on_mesh_rejects_unknown_sign_mode(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = sphere_irregular
    points_wp = warp_empty(4, wp.vec3, mesh_wp.device)
    with pytest.raises(ValueError, match="sign_mode"):
        od.proximity.signed_distance_on_mesh(
            mesh_wp.points,
            mesh_wp.indices,
            points_wp,
            sign_mode="nearest",  # pyright: ignore[reportArgumentType]  # deliberately off-menu
        )


@pytest.mark.parametrize("mesh_name", _SIGNED_DISTANCE_MESHES)
def test_supplied_mesh_gives_the_same_answer(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    ``mesh=`` is an optimization, so its only correctness obligation is to change nothing.

    Asserted **exactly**, not approximately: both paths query the same BVH over the same buffers,
    so the results must be bit-identical, and a tolerance here would hide a mesh built over the
    wrong geometry. Covers both queries and both of ``signed_distance_on_mesh``'s parity settings
    that accept a mesh at all.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(21)
    points_wp = points_to_warp(rng.normal(scale=1.5, size=(256, 3)), mesh_wp.device)
    prebuilt_wp = wp.Mesh(points=wp.clone(mesh_wp.points), indices=wp.clone(mesh_wp.indices))

    rebuilt = od.proximity.closest_point_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)
    supplied = od.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, mesh=prebuilt_wp
    )
    for from_rebuild, from_supplied in zip(rebuilt, supplied, strict=True):
        assert np.array_equal(from_rebuild.numpy(), from_supplied.numpy())

    assert np.array_equal(
        od.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp).numpy(),
        od.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, mesh=prebuilt_wp
        ).numpy(),
    )


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
def test_signed_distance_on_mesh_winding_accepts_only_a_flagged_mesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: a supplied mesh in winding mode is accepted iff it carries the flag.

    Without ``support_winding_number=True`` the winding builtin silently falls back to ray
    parity -- a wrong answer that looks like a right one -- so an unflagged mesh is refused. A
    flagged one must give exactly the answer of the self-built mesh, which carries the oracle
    (``test_signed_distance_on_mesh_winding_sign_matches_exact_winding_number``). On the open
    ``hemisphere`` the two signs disagree somewhere, so a mesh whose flag were ignored would fail
    the equality rather than pass it by coincidence.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(5)
    lower_np, upper_np = mesh_tm.bounds
    points_wp = points_to_warp(rng.uniform(lower_np, upper_np, size=(500, 3)), mesh_wp.device)
    unflagged_wp = wp.Mesh(points=wp.clone(mesh_wp.points), indices=wp.clone(mesh_wp.indices))
    flagged_wp = wp.Mesh(
        points=wp.clone(mesh_wp.points),
        indices=wp.clone(mesh_wp.indices),
        support_winding_number=True,
    )

    with pytest.raises(ValueError, match="support_winding_number"):
        od.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding", mesh=unflagged_wp
        )
    self_built = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
    ).numpy()
    supplied = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding", mesh=flagged_wp
    ).numpy()
    assert np.array_equal(self_built, supplied)
    if not mesh_tm.is_watertight:  # only an open surface signs differently by parity
        parity = od.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, mesh=flagged_wp
        ).numpy()
        assert not np.array_equal(np.sign(parity), np.sign(supplied))


@pytest.mark.parametrize("mesh_name", _SIGNED_DISTANCE_MESHES)
def test_signed_distance_on_mesh_coplanar(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    outside_np = np.asarray([mesh_tm.bounds[0] + [100.0, 0.0, 0.0]], dtype=np.float32)
    outside_wp = points_to_warp(outside_np, mesh_wp.device)
    outside_signed_wp = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, outside_wp
    )
    assert (outside_signed_wp.numpy() > 0.0).all()


def test_signed_distance_on_mesh_on_surface(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = sphere_irregular
    surface_np, _face_idx = tm.sample.sample_surface(mesh_tm, 50)[:2]
    surface_wp = points_to_warp(surface_np, mesh_wp.device)
    signed_wp = od.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, surface_wp)
    signed_np = signed_wp.numpy()
    assert (np.abs(signed_np) <= max(TOLERANCE_MERGE, 1e-4)).all()


def test_signed_distance_contains_points_consistency(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = sphere_irregular
    rng = np.random.default_rng(9)
    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    points_wp = points_to_warp(inside_np, mesh_wp.device)
    signed_np = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    ).numpy()
    contains_np = od.ray.contains_points(mesh_wp, points_wp).numpy()
    off_surface = np.abs(signed_np) > TOLERANCE_MERGE
    assert np.array_equal(contains_np[off_surface], signed_np[off_surface] < 0.0)


@pytest.mark.parity("signed_distance_grid", "open3d")
def test_signed_distance_grid_matches_open3d(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A: the sampled lattice equals open3d's ``compute_signed_distance`` at the same points.

    The reference shares ordito's sign convention exactly -- negative inside, no negation needed --
    so this is an equality rather than a mapping, and it is the whole field rather than a statistic:
    measured max deviation **1.27e-07** over a 26 x 26 x 26 lattice, which is float32 rounding.

    The lattice's own geometry is asserted alongside it, because a field is only usable if the box
    it claims to span is the box it does span: sample ``[0, 0, 0]`` sits on the returned ``lower``,
    the spacing is exactly ``voxel_size`` on every axis, and the padding is honoured. Those are what
    ``marching_cubes`` reads through ``bounds``, so getting them wrong shifts every offset surface
    by a fraction of a cell without failing any value comparison.
    """
    voxel_size = 0.1
    pad = 3
    mesh_tm, _ = sphere_irregular
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    field_wp, (lower, upper) = od.proximity.signed_distance_grid(
        vertices_wp, faces_wp, voxel_size, pad=pad
    )
    field_np = field_wp.numpy()

    # The box: the mesh's own, grown by exactly ``pad`` cells, and spaced by exactly ``voxel_size``.
    mesh_lower_np = np.asarray(mesh_tm.vertices).min(axis=0)
    assert np.allclose(odt.vec3_floats(lower), mesh_lower_np - pad * voxel_size, atol=1e-6)
    for axis in range(3):
        span = float(upper[axis]) - float(lower[axis])
        assert np.isclose(span / (field_np.shape[axis] - 1), voxel_size, rtol=1e-6)

    samples_np = od.voxels.grid_points(field_wp.shape, bounds=(lower, upper), device=device).numpy()
    scene_o3d = o3d.t.geometry.RaycastingScene()
    scene_o3d.add_triangles(trimesh_to_open3d_t(mesh_tm))
    distance_o3d = scene_o3d.compute_signed_distance(
        o3d.core.Tensor(samples_np.astype(np.float32))  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
    ).numpy()
    assert np.abs(field_np.ravel() - distance_o3d).max() < 1e-5

    # Non-vacuity in both signs: the lattice has to straddle the surface to be worth comparing.
    assert field_np.min() < -0.5 * voxel_size
    assert field_np.max() > 0.5 * voxel_size


def test_signed_distance_grid_guards_and_conventions(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the two guards, and that ``bounds`` overrides the mesh's own box.

    The ``mesh=`` argument is forwarded rather than interpreted, so
    ``signed_distance_on_mesh``'s rule about ``sign_mode="winding"`` (the mesh must carry
    ``support_winding_number``) applies here unchanged -- that is asserted, because a wrapper that
    quietly built its own mesh would hide an unflagged one.
    """
    mesh_tm, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )

    with pytest.raises(ValueError, match="pad"):
        od.proximity.signed_distance_grid(vertices_wp, faces_wp, 0.1, pad=-1)
    with pytest.raises(ValueError, match="at least one face"):
        od.proximity.signed_distance_grid(vertices_wp, warp_empty(0, wp.int32, device), 0.1)
    with pytest.raises(ValueError, match="support_winding_number"):
        od.proximity.signed_distance_grid(
            vertices_wp, faces_wp, 0.1, sign_mode="winding", mesh=mesh_wp
        )

    # An explicit box is used as given (before padding), whatever the mesh's own extent: a cube of
    # half-width 0.25 about the centre of mass, inside a body whose radius is at least 0.85.
    centre_np = mesh_tm.center_mass
    tight_field_wp, (tight_lower, _tight_upper) = od.proximity.signed_distance_grid(
        vertices_wp,
        faces_wp,
        0.1,
        bounds=(wp.vec3(*(centre_np - 0.25).tolist()), wp.vec3(*(centre_np + 0.25).tolist())),
        pad=0,
    )
    assert np.allclose(odt.vec3_floats(tight_lower), centre_np - 0.25, atol=1e-6)
    # Entirely inside the body, so every sample of that box is inside it.
    assert tight_field_wp.numpy().max() < 0.0


def _force_winding_tree(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Move ``winding_number``'s size gate so every call takes (or skips) the patch hierarchy."""
    threshold = 0 if enabled else 2**62
    for name in (
        "WINDING_TREE_FROM_PAIRS_CUDA",
        "WINDING_TREE_FROM_QUERIES_CUDA",
        "WINDING_TREE_FROM_PAIRS_CPU",
        "WINDING_TREE_FROM_QUERIES_CPU",
    ):
        monkeypatch.setattr(kernel_proximity, name, threshold)


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "sphere_irregular_hollow", "saddle_graded"]
)
@pytest.mark.parametrize("path", ["serial", "tiled", "tree"])
@pytest.mark.parity("winding_number", "igl")
@pytest.mark.parity(
    "winding_number_serial",
    "igl",
    benchmarked=False,
    reason="the tiled parametrization above is what covers the serial group: igl.winding_number is "
    "one implementation, already timed under winding_number, so a second row here would time the "
    "identical call twice under two names. This group exists to pin ordito's own tiled=False "
    "path, which the reference has no counterpart for.",
)
def test_winding_number_random(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, mesh_name: str, path: str
) -> None:
    """
    Class A: the generalized winding number against ``igl.winding_number``, on all three paths.

    Parametrized over the path so the serial sum, the tiled sum and the patch hierarchy are each
    held to the same reference rather than to each other -- 200 queries spanning inside, outside
    and near-surface. The size gate is moved so each arm takes its own path at this size.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    query_np = _random_queries_np()
    _force_winding_tree(monkeypatch, enabled=path == "tree")

    winding_igl = igl.winding_number(*mesh_igl(mesh_tm), query_np)
    query_wp = points_to_warp(query_np, mesh_wp.device)
    winding_wp = od.proximity.winding_number(
        mesh_wp.points, mesh_wp.indices, query_wp, tiled=path != "serial"
    )
    assert np.allclose(winding_wp.numpy(), np.asarray(winding_igl).ravel(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _SIGNED_DISTANCE_MESHES)
@pytest.mark.parity("winding_number", "meshlib")
def test_winding_number_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: ``calcFastWindingNumber`` is a Barnes-Hut approximation, so ``beta`` is the transform.

    ordito sums the solid angle of every triangle exactly; MeshLib traverses the mesh's AABB tree
    and replaces a distant subtree by its dipole approximation, with ``beta`` the accuracy parameter
    that decides how distant is distant enough. So the two agree only in the limit, and the
    comparison has to say where that limit is rather than pick a tolerance and hope.

    Measured on 200 queries against ``icosphere(3)``, max absolute difference by ``beta``:

    | ``beta`` | 2 (its own default) | 4 | 8 | 20 | 100 |
    |---|---|---|---|---|---|
    | max diff | 2.4e-02 | 5.0e-03 | 1.2e-03 | 1.1e-05 | 1.4e-06 |

    ``beta=20`` is used here: two orders of magnitude inside the 1e-3 tolerance, and the default of
    2 would fail it by 24x -- which is the point, since a comparison that passed at ``beta=2`` would
    be tolerating the approximation rather than measuring the quantity.

    ``calcDipoles`` must run first and takes the tree, not the mesh alone; its two-argument overload
    is the one that *returns* the dipoles; the three-argument form fills a caller-owned buffer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    query_np = _queries_in_bounds_np(mesh_tm, 200, seed=42)
    query_wp = points_to_warp(query_np, mesh_wp.device)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    tree_ml = mesh_ml.getAABBTree()
    dipoles_ml = mm.calcDipoles(tree_ml, mesh_ml)
    winding_ml = np.array(
        [
            mm.calcFastWindingNumber(
                dipoles_ml, tree_ml, mesh_ml, mm.Vector3f(*point_np.tolist()), 20.0, mm.FaceId()
            )
            for point_np in query_np
        ]
    )
    winding_wp = od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp).numpy()

    assert (winding_ml > 0.5).any()  # non-vacuity: inside and outside are both represented
    assert (winding_ml < 0.5).any()
    assert np.allclose(winding_wp, winding_ml, rtol=1e-3, atol=1e-3)


def test_winding_number_tiled_matches_exact(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = sphere_irregular
    rng = np.random.default_rng(17)
    query_np = rng.random((100, 3), dtype=np.float32) * 2.0 - 1.0
    query_wp = points_to_warp(query_np, mesh_wp.device)
    exact_wp = od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=False)
    tiled_wp = od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=True)
    assert np.allclose(tiled_wp.numpy(), exact_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_winding_number_tiled_matches_igl_on_the_cpu() -> None:
    """
    Class A: the tiled winding sum on the CPU device the ``device`` fixture never reaches.

    ``wp.launch_tiled`` executes exactly one lane per block on Warp 1.18's CPU backend -- the lane
    index from ``wp.tid()`` is always 0 -- so the block-wide ``wp.tile_sum`` this reduction used to
    perform summed one face per 64-face tile and returned a winding number off by up to 0.99 there,
    i.e. a whole turn. The reduction is lane-free now; this is the test that fails if it regresses.
    The CUDA tiled path is held to igl by ``test_winding_number_random[True-*]``.
    """
    kernel_device = "cpu"
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    rng = np.random.default_rng(43)
    query_np = rng.random((200, 3), dtype=np.float64) * 3.0 - 1.5

    winding_igl = igl.winding_number(*mesh_igl(mesh_tm), query_np)
    query_wp = points_to_warp(query_np, kernel_device)
    winding_wp = od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=True)
    assert np.allclose(winding_wp.numpy(), np.asarray(winding_igl).ravel(), rtol=1e-5, atol=1e-5)


def _near_surface_queries_np(mesh_tm: tm.Trimesh, n: int, seed: int) -> np.ndarray:
    """``n`` queries jittered off the surface at three scales, plus a quarter drawn from the box."""
    rng = np.random.default_rng(seed)
    diagonal = float(np.linalg.norm(mesh_tm.bounds[1] - mesh_tm.bounds[0]))
    on_surface = mesh_tm.vertices[rng.integers(0, len(mesh_tm.vertices), n)]
    scales = np.repeat([1e-2, 1e-3, 1e-4], -(-n // 3))[:n, None] * diagonal
    near = on_surface + rng.normal(size=(n, 3)) * scales
    return np.vstack([near, _queries_in_bounds_np(mesh_tm, n // 4, seed)])


def _winding_soup(mesh_tm: tm.Trimesh, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``mesh_tm`` as a soup: some faces flipped, duplicated, degenerate, and non-manifold fins."""
    rng = np.random.default_rng(seed)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    flipped = rng.random(len(faces_np)) < 0.2
    faces_np[flipped] = faces_np[flipped][:, ::-1]
    duplicated = faces_np[rng.random(len(faces_np)) < 0.05]
    degenerate = np.repeat(rng.integers(0, len(mesh_tm.vertices), 20)[:, None], 3, axis=1)
    # A third face on existing edges: non-manifold fins.
    fins = np.column_stack([faces_np[:10, 0], faces_np[:10, 1], faces_np[10:20, 2]])
    soup_np = np.vstack([faces_np, duplicated, degenerate, fins])
    return np.asarray(mesh_tm.vertices, dtype=np.float64), soup_np[rng.permutation(len(soup_np))]


@pytest.mark.parametrize(
    "mesh_name",
    ["sphere_irregular", "sphere_irregular_hollow", "saddle_graded", "mobius", "boy_surface"],
)
@pytest.mark.parametrize("soup", [False, True], ids=["mesh", "soup"])
def test_winding_number_tree_matches_the_direct_sum(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, mesh_name: str, soup: bool
) -> None:
    """
    Ordito against ordito: the patch hierarchy against the serial exact sum, near the surface.

    The serial sum carries the oracle (``test_winding_number_random``); this pins the hierarchy to
    it where the hierarchy has the most to get wrong -- queries a hundredth to a ten-thousandth of
    the diagonal off the surface, which sit just outside many patches' regions -- on closed, open
    and non-orientable fixtures, and on a soup of each with flipped, duplicated, degenerate and
    non-manifold fin faces, whose boundary chains do not cancel. A fan identity applied where it
    does not hold is off by a whole turn, not by rounding, so the 1e-5 bar separates the two. The
    result is also bit-identical between two calls: the hierarchy sums in fixed point.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = mesh_wp.device
    if soup:
        vertices_wp, faces_wp = numpy_to_warp(*_winding_soup(mesh_tm, seed=7), device)
    else:
        vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    query_wp = points_to_warp(_near_surface_queries_np(mesh_tm, 600, seed=11), device)

    serial_wp = od.proximity.winding_number(vertices_wp, faces_wp, query_wp, tiled=False)
    _force_winding_tree(monkeypatch, enabled=True)
    tree_wp = od.proximity.winding_number(vertices_wp, faces_wp, query_wp)
    again_wp = od.proximity.winding_number(vertices_wp, faces_wp, query_wp)

    serial_np = serial_wp.numpy()
    assert np.ptp(serial_np) > 0.5  # non-vacuity: both sides of the surface are queried
    assert np.allclose(tree_wp.numpy(), serial_np, rtol=0.0, atol=1e-5)
    assert np.array_equal(tree_wp.numpy(), again_wp.numpy())


def test_winding_number_tree_fans_from_inside_the_patch_region(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the closing fan's apex must lie in the region the query is outside of.

    One planar triangle ``x + y + z = 1`` spanning the unit box, finely subdivided, so the root
    patch keeps a fan over its 192-edge boundary. Its box is the unit box, but its 14-DOP has the
    diagonal slab pinned to ``x + y + z = 1``, so a query on the box diagonal at ``x + y + z =
    1.35`` is inside the box and outside the region. An apex at the box centre (``x + y + z =
    1.5``) lies outside that region: patch plus fan is then a closed surface enclosing the query,
    and the answer is off by exactly one turn. The apex the hierarchy uses -- the mean of the
    patch's corners -- lies on the triangle.
    """
    subdivisions = 64
    rows = [(i, j) for i in range(subdivisions + 1) for j in range(subdivisions + 1 - i)]
    index = {ij: k for k, ij in enumerate(rows)}
    barycentric_np = (
        np.array([(i, j, subdivisions - i - j) for i, j in rows], dtype=np.float64) / subdivisions
    )
    faces_np = []
    for i, j in rows:
        if i + j < subdivisions:
            faces_np.append((index[i, j], index[i + 1, j], index[i, j + 1]))
        if i + j < subdivisions - 1:
            faces_np.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    vertices_wp, faces_wp = numpy_to_warp(barycentric_np, np.array(faces_np), device)
    diagonal_np = np.linspace(0.05, 0.95, 37)[:, None] * np.ones(3)
    rng = np.random.default_rng(5)
    query_np = np.vstack([diagonal_np, rng.random((200, 3))])
    query_wp = points_to_warp(query_np, device)

    serial_np = od.proximity.winding_number(vertices_wp, faces_wp, query_wp, tiled=False).numpy()
    _force_winding_tree(monkeypatch, enabled=True)
    tree_np = od.proximity.winding_number(vertices_wp, faces_wp, query_wp).numpy()

    assert len(faces_np) == subdivisions**2
    assert np.abs(serial_np).max() > 0.1  # non-vacuity: the triangle subtends a real angle
    assert np.allclose(tree_np, serial_np, rtol=0.0, atol=1e-5)


def test_winding_number_tree_matches_igl_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Class A: the patch hierarchy on the CPU device the ``device`` fixture never reaches.

    Its walk is a ``launch_tiled`` kernel whose lanes split each patch's work by
    ``wp.block_dim()``: one lane on the CPU device, which must then cover every fan edge and face
    itself. A lane partition that ignored the block width would sum a fraction of them there.
    """
    kernel_device = "cpu"
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    query_np = _near_surface_queries_np(mesh_tm, 200, seed=43)
    _force_winding_tree(monkeypatch, enabled=True)

    winding_igl = igl.winding_number(*mesh_igl(mesh_tm), query_np)
    query_wp = points_to_warp(query_np, kernel_device)
    winding_wp = od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp)
    assert np.ptp(np.asarray(winding_igl)) > 0.5
    assert np.allclose(winding_wp.numpy(), np.asarray(winding_igl).ravel(), rtol=1e-5, atol=1e-5)


def test_winding_number_in_the_cavity_and_the_shell(
    sphere_irregular_hollow: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: winding 0 in the hollow, 1 in the solid between the shells.

    The cavity's centre is the inner shell's vertex mean (the inward-wound inner shell cancels the
    outer one there); 0.9 of the way to the outer shell's farthest vertex the point is inside the
    outer shell only.
    """
    mesh_tm, mesh_wp = sphere_irregular_hollow
    shells = mesh_tm.split(only_watertight=False)
    inner_tm = min(shells, key=lambda shell: len(shell.vertices))
    outer_tm = max(shells, key=lambda shell: len(shell.vertices))
    centre_np = inner_tm.vertices.mean(axis=0)
    farthest_np = outer_tm.vertices[
        np.argmax(np.linalg.norm(outer_tm.vertices - centre_np, axis=1))
    ]
    points_np = np.array([centre_np, centre_np + 0.9 * (farthest_np - centre_np)], dtype=np.float32)
    points_wp = points_to_warp(points_np, mesh_wp.device)
    winding_wp = od.proximity.winding_number(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(winding_wp.numpy(), [0.0, 1.0], atol=1e-3)


# (name, callable) pairs; each callable takes (vertices, faces, points) and returns the tuple of
# arrays the wrapper produces -- closest_point_on_mesh returns three, the other two return one.
_PROXIMITY_EMPTY_POINTS_CASES = [
    ("closest_point_on_mesh", lambda v, f, p: od.proximity.closest_point_on_mesh(v, f, p)),
    ("signed_distance_on_mesh", lambda v, f, p: (od.proximity.signed_distance_on_mesh(v, f, p),)),
    ("winding_number", lambda v, f, p: (od.proximity.winding_number(v, f, p),)),
]


@pytest.mark.parametrize(
    "proximity_fn",
    [case[1] for case in _PROXIMITY_EMPTY_POINTS_CASES],
    ids=[case[0] for case in _PROXIMITY_EMPTY_POINTS_CASES],
)
def test_proximity_empty_points_is_a_noop(
    device: str,
    proximity_fn: Callable[
        [wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.vec3]], tuple[wp.array[object], ...]
    ],
) -> None:
    """Not a library comparison: zero query points give an all-empty result for every operator."""
    vertices = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    points = warp_empty(0, wp.vec3, device)
    for result_wp in proximity_fn(vertices, faces, points):
        assert result_wp.shape == (0,)


# Each wrapper's answer on a mesh with no faces, one fill value per returned array: no closest
# point (NaN), at infinite distance, on no face (-1); and nothing to wind around (0).
_PROXIMITY_EMPTY_FACES_CASES = [
    (
        "closest_point_on_mesh",
        lambda v, f, p: od.proximity.closest_point_on_mesh(v, f, p),
        (math.nan, math.inf, -1),
    ),
    (
        "signed_distance_on_mesh",
        lambda v, f, p: (od.proximity.signed_distance_on_mesh(v, f, p),),
        (math.inf,),
    ),
    ("winding_number", lambda v, f, p: (od.proximity.winding_number(v, f, p),), (0.0,)),
]


@pytest.mark.parametrize(
    ("proximity_fn", "fills"),
    [(case[1], case[2]) for case in _PROXIMITY_EMPTY_FACES_CASES],
    ids=[case[0] for case in _PROXIMITY_EMPTY_FACES_CASES],
)
def test_proximity_empty_faces(
    device: str,
    proximity_fn: Callable[
        [wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.vec3]], tuple[wp.array[object], ...]
    ],
    fills: tuple[float, ...],
) -> None:
    """Not a library comparison: a face-less mesh answers every query with the empty-mesh fill."""
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = warp_empty(0, wp.int32, device)
    points = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    results = proximity_fn(vertices, faces, points)
    assert len(results) == len(fills)
    for result_wp, fill in zip(results, fills, strict=True):
        assert result_wp.shape == (2,)
        values_np = np.asarray(result_wp.numpy(), dtype=np.float64)
        assert np.all(np.isnan(values_np)) if math.isnan(fill) else np.all(values_np == fill)


# --------------------------------------------------------------------------------------
# containing_faces_2d
# --------------------------------------------------------------------------------------


def _triangular_lattice_np(rows: int = 26, cols: int = 30) -> np.ndarray:
    """
    Points of a triangular lattice, whose Delaunay triangulation is equilateral throughout.

    The fixture choice is the whole reason the comparison below can be exact. Any Delaunay
    triangulation of a *bounded* point set has needle triangles along its convex hull, and this
    function's single-candidate query cannot resolve a query lying within ``float32`` noise of the
    shared edge of two needles -- measured at 3 in 20 000 on random points, aspect ratios 2 000 to
    12 000. A triangular lattice has no needles anywhere (aspect ratio 6.93 at worst, 2.31 median),
    including along its hull, so it isolates the algorithm from that resolution limit.
    """
    height = np.sqrt(3.0) / 2.0
    return np.array(
        [(col + 0.5 * (row % 2), row * height) for row in range(rows) for col in range(cols)],
        dtype=np.float64,
    )


@pytest.mark.parity("containing_faces_2d", "scipy", "pyvista")
def test_containing_faces_2d_matches_scipy_and_pyvista(device: str) -> None:
    """
    Class A: the same triangle index as ``Delaunay.find_simplex`` and ``find_containing_cell``.

    Both references are given the *same* triangulation -- scipy's own ``simplices`` are what ordito
    locates against -- so the face numbering is shared and each comparison is an integer array
    equality, ``-1`` for outside included, with no transform at all. Measured agreement is
    ``1.0000000`` over the 40 000 queries against scipy; pyvista is compared on the first 10 000,
    where it measured **1.0000**, with the batched call and a per-point loop giving byte-identical
    answers and no false ``-1``. VTK's only transform is the embedding: it locates in 3-D, so the
    lattice and the queries get a zero ``z``.

    Worth stating explicitly because the neighbouring reference is the opposite: section 6 records
    ``igl.in_element`` as *unusable* for this question -- it never reports element 0 for a query
    inside it, returns different answers for the same query depending on the batch size, and aborts
    with ``malloc(): invalid size`` on a 200-point Delaunay. VTK's locator has none of those
    defects, so a reader generalizing from libigl would skip a reference that works.

    Non-vacuous in both directions by construction: the query box overhangs the lattice, so roughly
    72% of queries land inside and 28% outside, and both counts are asserted before comparing. An
    implementation returning ``-1`` everywhere, or a face for everything, fails one of them.
    """
    points_np = _triangular_lattice_np()
    triangulation_sp = Delaunay(points_np)
    faces_np = np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32)
    rng = np.random.default_rng(11)
    queries_np = rng.random((40_000, 2)) * np.array([34.0, 26.0]) - 2.0

    vertices_wp = points_to_warp_uv(points_np, device)
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=device)
    queries_wp = points_to_warp_uv(queries_np, device)
    located_wp = od.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp).numpy()

    faces_sp = triangulation_sp.find_simplex(queries_np)
    assert (faces_sp >= 0).sum() > 20_000, "the reference places most queries inside"
    assert (faces_sp < 0).sum() > 5_000, "and a substantial minority outside"
    assert np.array_equal(located_wp, faces_sp)

    n_pv = 10_000
    mesh_pv = pv.PolyData.from_regular_faces(
        np.column_stack([points_np, np.zeros(points_np.shape[0])]), faces_np
    )
    queries_3d_np = np.column_stack([queries_np[:n_pv], np.zeros(n_pv)])
    located_pv = np.asarray(mesh_pv.find_containing_cell(queries_3d_np))
    assert (located_pv >= 0).sum() > 5_000, "the reference places most queries inside"
    assert (located_pv < 0).sum() > 1_000, "and a substantial minority outside"
    assert np.array_equal(located_wp[:n_pv], located_pv)


def test_containing_faces_2d_locates_every_triangle_from_its_centroid(device: str) -> None:
    """
    Every triangle contains its own centroid, so locating the centroids must be the identity.

    A different failure mode from the random-query test: it visits *all* faces exactly once, so a
    face never reachable by the query -- one dropped from the BVH, or one whose barycentric test is
    mis-signed -- shows up here even if random queries happen to miss it.
    """
    points_np = _triangular_lattice_np(rows=12, cols=14)
    faces_np = np.ascontiguousarray(Delaunay(points_np).simplices, dtype=np.int32)
    centroids_np = points_np[faces_np].mean(axis=1)

    vertices_wp = points_to_warp_uv(points_np, device)
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=device)
    centroids_wp = points_to_warp_uv(centroids_np, device)

    located_np = od.proximity.containing_faces_2d(vertices_wp, faces_wp, centroids_wp).numpy()

    assert np.array_equal(located_np, np.arange(faces_np.shape[0], dtype=np.int32))


def test_containing_faces_2d_degrades_only_on_needle_triangles(device: str) -> None:
    """
    The documented resolution limit, pinned: on extreme slivers a query can come back ``-1``.

    A Delaunay triangulation of random points always has needles along its convex hull, and a query
    within ``float32`` noise of the shared edge of two of them may be attributed to neither: the
    nearest triangle by distance is the neighbour, and the barycentric test then rejects it.

    Asserted as a *bounded* failure of a *specific* shape, not as a tolerance: at least 99.9% of
    queries agree with scipy, and every disagreement is ordito reporting ``-1`` -- never a
    different face, which would be a wrong answer rather than a declined one. Measured: 3 in
    20 000, on triangles of aspect ratio 1 900 to 12 400.
    """
    rng = np.random.default_rng(0)
    points_np = rng.random((2_000, 2)) * 4.0 - 2.0
    triangulation_sp = Delaunay(points_np)
    faces_np = np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32)
    queries_np = rng.random((20_000, 2)) * 5.0 - 2.5

    vertices_wp = points_to_warp_uv(points_np, device)
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=device)
    queries_wp = points_to_warp_uv(queries_np, device)

    faces_wp_np = od.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp).numpy()

    faces_sp = triangulation_sp.find_simplex(queries_np)
    assert (faces_wp_np == faces_sp).mean() >= 0.999
    disagreement = faces_wp_np != faces_sp
    assert np.all(faces_wp_np[disagreement] == -1), "declines, never a different face"


def test_containing_faces_2d_single_triangle(device: str) -> None:
    """One triangle: inside, outside, and a vertex, with the tolerance carrying no weight."""
    vertices_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    queries_wp = wp.array(
        np.array([[0.25, 0.25], [0.9, 0.9], [0.0, 0.0], [-5.0, 3.0]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )

    located_np = od.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp).numpy()

    assert np.array_equal(located_np, [0, -1, 0, -1])


def test_containing_faces_2d_empty(device: str) -> None:
    """No queries gives no answers; no faces gives ``-1`` for every query."""
    vertices_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    no_queries_wp = wp.zeros(0, dtype=wp.vec2, device=device)
    assert od.proximity.containing_faces_2d(vertices_wp, faces_wp, no_queries_wp).size == 0

    queries_wp = wp.array(np.array([[0.25, 0.25]], dtype=np.float32), dtype=wp.vec2, device=device)
    no_faces_wp = wp.zeros(0, dtype=wp.int32, device=device)
    assert np.array_equal(
        od.proximity.containing_faces_2d(vertices_wp, no_faces_wp, queries_wp).numpy(), [-1]
    )
