"""Regression tests for ``ordito.ray`` against trimesh."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from tests.conversions import (
    points_to_pyvista,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pyvista,
    warp_empty,
)


def _assert_longest_ray_allclose(distances_wp_np: np.ndarray, distances_tm_np: np.ndarray) -> None:
    assert np.array_equal(np.isinf(distances_wp_np), np.isinf(distances_tm_np))
    finite_tm = ~np.isinf(distances_tm_np)
    if finite_tm.any():
        assert np.allclose(
            distances_wp_np[finite_tm], distances_tm_np[finite_tm], rtol=1e-5, atol=1e-5
        )


def _upward_rays(mesh_tm: tm.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    ``n`` rays starting below the mesh and firing ``+z``, with origins inside its xy footprint.

    Drawing the origins from the unit square instead put every one of them outside the x-extent of
    both the ``icosahedron`` and ``hemisphere`` fixtures -- each translated by ``(-1, 0, 2)`` -- so
    all three ``intersects_*`` comparisons below ran on 256 misses, asserted that two all-miss
    answers agree, and duplicated the all-miss test. Measured: 140 of 256 rays hit the
    icosahedron, 134 the hemisphere and 174 ``saddle_graded``. ``test_ray_queries_miss_together``
    keeps its own construction, which misses for a robust reason -- firing ``+y`` from below never
    leaves the plane ``z = min_z - 5`` -- rather than by an accident of where the fixture sits.
    """
    rng = np.random.default_rng(seed)
    # The footprint is widened by a tenth of its extent on each side so that some rays miss even a
    # height field: ``saddle_graded`` covers its whole xy box, and a draw inside the box hit it 256
    # times out of 256.
    lower, upper = mesh_tm.bounds[0, :2], mesh_tm.bounds[1, :2]
    margin = 0.1 * (upper - lower)
    origins_np = np.column_stack(
        (
            rng.uniform(lower - margin, upper + margin, size=(n, 2)),
            np.full(n, mesh_tm.bounds[0, 2] - 5.0),
        )
    ).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    return origins_np, directions_np


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "saddle_graded"])
def test_intersects_match_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on ``intersects_first`` and ``intersects_any``, Class B on ``intersects_location``.

    The first-hit face index and the per-ray hit mask agree with trimesh's ray engine, ray for ray.
    The location table's ``(ray, face)`` hit pairs equal trimesh's after a shared lexsort (row
    order): ordito emits one row per hit in BVH order and trimesh in its own, so both sides are
    ordered by ``(ray, face)`` first -- exact on integer rows. The hit *positions* are checked by
    [`test_intersects_location_cave_cube`], which has an analytic answer to compare against.
    Each reference answer is asserted to carry hits; without that, two all-miss answers agree (see
    ``_upward_rays``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)
    origins_wp = points_to_warp(origins_np, mesh_wp.device)
    directions_wp = points_to_warp(directions_np, mesh_wp.device)

    triangle_wp = od.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert (triangle_tm != -1).any()
    assert np.array_equal(triangle_wp, triangle_tm)

    hit_wp = od.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy()
    hit_tm = mesh_tm.ray.intersects_any(origins_np, directions_np)
    assert hit_tm.any()
    assert np.array_equal(hit_wp, hit_tm)

    _loc_wp, ray_wp, tri_wp = od.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    tri_tm, ray_tm = mesh_tm.ray.intersects_id(origins_np, directions_np, multiple_hits=False)
    tri_wp_np = tri_wp.numpy()
    ray_wp_np = ray_wp.numpy()
    order_wp = np.lexsort((tri_wp_np, ray_wp_np))
    order_tm = np.lexsort((tri_tm, ray_tm))
    assert tri_tm.size > 0
    assert np.array_equal(tri_wp_np[order_wp], tri_tm[order_tm])
    assert np.array_equal(ray_wp_np[order_wp], ray_tm[order_tm])


def test_intersects_location_cave_cube(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class B (segment canonicalization): a 20x20 ray grid fired ``+z`` through the hollow fixture.

    Each ray reports its *first* hit, on both sides (``multiple_hits=False``), so every hit lies on
    the outer wall's bottom face: its ``z`` is the mesh's lowest and its ``xy`` the ray's own, an
    analytic answer beside trimesh's. The grid overhangs the footprint, so its edge rays miss on
    both sides. Row order is not defined by either library, so the hits are compared sorted by ray.
    """
    mesh_tm, mesh_wp = cave_cube
    origins_np = tm.util.grid_linspace(
        mesh_tm.bounds[:, :2] + np.reshape([-0.02, 0.02], (-1, 1)), 20
    )
    origins_np = np.column_stack((origins_np, np.ones(len(origins_np)) * -100.0)).astype(np.float32)
    directions_np = np.ones((len(origins_np), 3), dtype=np.float32) * [0.0, 0.0, 1.0]

    origins_wp = points_to_warp(origins_np, mesh_wp.device)
    directions_wp = points_to_warp(directions_np, mesh_wp.device)
    loc_wp, ray_wp, _tri_wp = od.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    loc_tm, ray_tm, _tri_tm = mesh_tm.ray.intersects_location(
        origins_np, directions_np, multiple_hits=False
    )

    ray_wp_np = ray_wp.numpy()
    loc_wp_np = loc_wp.numpy()
    order_wp = np.argsort(ray_wp_np)
    order_tm = np.argsort(ray_tm)
    assert 0 < ray_tm.size < len(origins_np)  # hits, and misses at the overhang
    assert np.array_equal(ray_wp_np[order_wp], ray_tm[order_tm])
    assert np.allclose(loc_wp_np[order_wp], loc_tm[order_tm], rtol=1e-5, atol=1e-4)
    for p, r in zip(loc_wp_np[order_wp], ray_wp_np[order_wp], strict=False):
        assert np.allclose(p[:2], origins_np[r][:2], rtol=1e-5, atol=1e-5)
        assert np.isclose(p[2], mesh_tm.bounds[0, 2], atol=1e-4)


def test_ray_queries_miss_together(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on an all-miss answer, which is a claim here rather than a vacuous comparison.

    Firing ``+y`` from below the mesh never leaves the plane ``z = min_z - 5``, so every ray
    misses for a robust reason, and each query states the emptiness explicitly (see
    ``_upward_rays``): an empty location table, every first-hit face ``-1`` and trimesh's, an
    all-``False`` hit mask and trimesh's, and an infinite longest ray as ``trimesh.proximity``
    reports it.
    """
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0
    origins_wp = points_to_warp(origins_np, mesh_wp.device)
    directions_wp = points_to_warp(directions_np, mesh_wp.device)

    loc_wp, ray_wp, tri_wp = od.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    assert loc_wp.size == 0
    assert tri_wp.size == 0
    assert ray_wp.size == 0

    triangle_wp = od.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    assert np.array_equal(triangle_wp, mesh_tm.ray.intersects_first(origins_np, directions_np))
    assert (triangle_wp == -1).all()

    hit_wp = od.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy()
    assert np.array_equal(hit_wp, mesh_tm.ray.intersects_any(origins_np, directions_np))
    assert not hit_wp.any()

    distances_wp_np = od.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, origins_np, directions_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)
    assert np.isinf(distances_wp_np).all()


def _multi_ray_intersect_ml(
    mesh_part_ml: mm.MeshPart, origins_np: np.ndarray, directions_np: np.ndarray
) -> mm.MultiRayMeshIntersectResult:
    """
    Run MeshLib's batched ray query, requesting every output field.

    The fields are **opt-in and silently so**: ``MultiRayMeshIntersectResult`` starts with every
    one of ``isectFaces`` / ``isectPts`` / ``rayDistances`` / ``intersectingRays`` set to ``None``,
    and ``multiRayMeshIntersect`` fills only the ones already holding a container. A result read
    without this step is all ``None`` with no error raised anywhere.
    """
    origins_ml = mm.std_vector_Vector3_float()
    directions_ml = mm.std_vector_Vector3_float()
    for origin_np, direction_np in zip(origins_np, directions_np, strict=True):
        origins_ml.append(mm.Vector3f(*np.asarray(origin_np, dtype=float).tolist()))
        directions_ml.append(mm.Vector3f(*np.asarray(direction_np, dtype=float).tolist()))

    result_ml = mm.MultiRayMeshIntersectResult()
    result_ml.isectFaces = mm.std_vector_Id_FaceTag()
    result_ml.isectPts = mm.std_vector_Vector3_float()
    result_ml.rayDistances = mm.std_vector_float()
    result_ml.intersectingRays = mm.BitSet()
    mm.multiRayMeshIntersect(mesh_part_ml, origins_ml, directions_ml, result_ml)
    return result_ml


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "saddle_graded"])
@pytest.mark.parity("intersects_first", "meshlib")
@pytest.mark.parity("intersects_any", "meshlib")
@pytest.mark.parity("intersects_location", "meshlib")
def test_intersects_match_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on all three: ``multiRayMeshIntersect`` answers every one of them from one traversal.

    The batched call is the pairing rather than the per-ray ``rayMeshIntersect`` (which agrees too,
    on every one of 100 rays) because a Python loop over rays would be measuring the loop -- the
    same reason the curvature comparisons use ``mrmeshnumpy``'s batched forms. Its four outputs map
    one-to-one onto ordito's three entry points: ``isectFaces`` is ``intersects_first``'s dense
    face array, ``intersectingRays`` is ``intersects_any``'s mask, and ``isectPts`` paired with
    that bitset is ``intersects_location``'s compacted ``(points, rays, faces)`` triple, once that
    triple is sorted by ray -- its compaction emits rows in BVH order, not in ray order, which the
    trimesh comparison above handles with a sort too.

    Two conventions, both named. MeshLib returns the hits **densely** with an invalid ``FaceId``
    where ordito writes ``-1``, so the transform on the first form is ``FaceId -> int`` with
    invalid mapping to ``-1``; and its location output is dense too, so the class-B half is
    selecting the rows ``intersectingRays`` marks, in ray order, which is the order ordito's
    compaction already produces.

    Non-vacuous: 140 of 256 rays hit the icosahedron, 134 the hemisphere and 174 the saddle, so
    neither the hit mask nor the location table is empty or full, and the miss rays check the
    ``-1`` convention.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)
    origins_wp = points_to_warp(origins_np, mesh_wp.device)
    directions_wp = points_to_warp(directions_np, mesh_wp.device)

    result_ml = _multi_ray_intersect_ml(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), origins_np, directions_np
    )
    hit_ml = np.zeros(origins_np.shape[0], dtype=bool)
    hit_ml[list(result_ml.intersectingRays)] = True
    faces_ml = np.array(
        [
            int(face_ml) if valid else -1
            for face_ml, valid in zip(result_ml.isectFaces, hit_ml, strict=True)
        ],
        dtype=np.int32,
    )
    points_ml = np.array([[point.x, point.y, point.z] for point in result_ml.isectPts])

    assert 0 < hit_ml.sum() < origins_np.shape[0]  # both branches present

    faces_wp = od.ray.intersects_first(mesh_wp, origins_wp, directions_wp)
    assert np.array_equal(faces_wp.numpy(), faces_ml)

    hit_wp = od.ray.intersects_any(mesh_wp, origins_wp, directions_wp)
    assert np.array_equal(hit_wp.numpy(), hit_ml)

    locations_wp, rays_wp, triangles_wp = od.ray.intersects_location(
        mesh_wp, origins_wp, directions_wp
    )
    order_wp = np.argsort(rays_wp.numpy(), kind="stable")
    assert np.array_equal(rays_wp.numpy()[order_wp], np.flatnonzero(hit_ml))
    assert np.array_equal(triangles_wp.numpy()[order_wp], faces_ml[hit_ml])
    assert np.allclose(locations_wp.numpy()[order_wp], points_ml[hit_ml], rtol=1e-5, atol=1e-5)


@pytest.mark.parity("intersects_first", "open3d")
@pytest.mark.parity("intersects_any", "open3d")
@pytest.mark.parity("intersects_location", "open3d")
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "saddle_graded"])
def test_intersects_match_open3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on the first two, Class B on the third: Embree answers each group with its own method.

    One method per group, not one method for all three -- which is the correction this test carries.
    ``test_occlusions`` is a genuine **any-hit** traversal and returns nothing else, so it is
    ``intersects_any``'s exact counterpart and the only reference in the suite that is allowed to
    stop early (MeshLib's ``closestIntersect`` stays on). ``cast_rays`` returns ``primitive_ids``
    and ``t_hit`` densely, which is ``intersects_first`` directly and ``intersects_location`` after
    one transform.

    Two conventions, both named. A miss is ``RaycastingScene.INVALID_ID`` where ordito writes
    ``-1``, and ``t_hit`` is ``inf`` there. And the hit *position* is not returned at all: it is
    ``origin + t_hit * direction``, which is the class-B half -- the same
    ``float32`` recomputation ordito's kernel does, so the two agree to the ray parameter's own
    precision rather than exactly.

    ordito's ``intersects_location`` compacts in BVH order rather than ray order, so its rows are
    sorted by ray first, exactly as the meshlib and trimesh comparisons above do.

    Non-vacuous: the assert requires both branches present in the hit mask, so a scene that hit
    everything or nothing would fail rather than agree.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)
    origins_wp = points_to_warp(origins_np, mesh_wp.device)
    directions_wp = points_to_warp(directions_np, mesh_wp.device)

    scene_o3d = o3d.t.geometry.RaycastingScene()
    scene_o3d.add_triangles(
        o3d.core.Tensor(  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=o3d.core.Dtype.Float32
        ),
        o3d.core.Tensor(  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
            np.ascontiguousarray(mesh_tm.faces, dtype=np.uint32), dtype=o3d.core.Dtype.UInt32
        ),
    )
    rays_o3d = o3d.core.Tensor(  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
        np.ascontiguousarray(np.hstack([origins_np, directions_np]), dtype=np.float32),
        dtype=o3d.core.Dtype.Float32,
    )
    hit_o3d = scene_o3d.test_occlusions(rays_o3d).numpy()
    cast_o3d = scene_o3d.cast_rays(rays_o3d)
    faces_o3d = cast_o3d["primitive_ids"].numpy().astype(np.int64)
    distance_o3d = cast_o3d["t_hit"].numpy().astype(np.float64)
    faces_o3d = np.where(faces_o3d == o3d.t.geometry.RaycastingScene.INVALID_ID, -1, faces_o3d)

    assert 0 < hit_o3d.sum() < origins_np.shape[0]  # both branches present
    # The any-hit answer and the nearest-hit answer are the same mask, which pins the two methods.
    assert np.array_equal(hit_o3d, faces_o3d >= 0)

    assert np.array_equal(
        od.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy(),
        faces_o3d.astype(np.int32),
    )
    assert np.array_equal(
        od.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy(), hit_o3d
    )

    locations_wp, rays_wp, triangles_wp = od.ray.intersects_location(
        mesh_wp, origins_wp, directions_wp
    )
    order_wp = np.argsort(rays_wp.numpy(), kind="stable")
    assert np.array_equal(rays_wp.numpy()[order_wp], np.flatnonzero(hit_o3d))
    assert np.array_equal(triangles_wp.numpy()[order_wp], faces_o3d[hit_o3d].astype(np.int32))
    # The hit position is a transform of t_hit, not an output: origin + t * direction.
    points_o3d = origins_np[hit_o3d] + distance_o3d[hit_o3d, None] * directions_np[hit_o3d]
    assert np.allclose(locations_wp.numpy()[order_wp], points_o3d, rtol=1e-5, atol=1e-5)


def test_intersects_empty_rays(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    origins_wp = warp_empty(0, wp.vec3, mesh_wp.device)
    directions_wp = warp_empty(0, wp.vec3, mesh_wp.device)
    assert od.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
    assert od.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
    loc_wp, ray_wp, tri_wp = od.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    assert loc_wp.size == 0
    assert tri_wp.size == 0
    assert ray_wp.size == 0
    assert od.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)


@pytest.mark.parametrize("n_origins", [0, 4])
def test_intersects_rejects_mismatched_shapes(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], n_origins: int
):
    """
    Not a library comparison: an argument-validation guard, which no reference library exposes.

    All four ray queries document a ``ValueError`` for origins and directions of unequal length,
    and all four raise it -- including on an *empty* origin buffer, which is the arm that pins
    where the check sits. It used to run after the ``n == 0`` early return, so a mismatched empty
    call returned an empty answer for a question that has none; with ``n_origins=4`` alone the
    guard passes wherever it is placed, so the ``0`` arm is the one doing the work.
    """
    _, mesh_wp = icosahedron
    origins_wp = warp_empty(n_origins, wp.vec3, mesh_wp.device)
    directions_wp = warp_empty(n_origins + 3, wp.vec3, mesh_wp.device)
    for query in (
        od.ray.intersects_first,
        od.ray.intersects_any,
        od.ray.intersects_location,
        od.ray.longest_ray,
    ):
        with pytest.raises(ValueError, match="same shape"):
            query(mesh_wp, origins_wp, directions_wp)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_longest_ray(request: pytest.FixtureRequest, mesh_name: str):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(3)
    n = 128
    origins_np = (rng.random((n, 3)).astype(np.float32) * 2.0 + mesh_tm.bounds[0]).astype(
        np.float32
    )
    directions_np = rng.normal(size=(n, 3)).astype(np.float32)

    origins_wp = points_to_warp(origins_np, mesh_wp.device)
    directions_wp = points_to_warp(directions_np, mesh_wp.device)
    distances_wp_np = od.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, origins_np, directions_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)


def test_longest_ray_surface_normals(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(11)
    n = 64
    query_np = rng.random((n, 3)).astype(np.float64)
    closest_np, _distance_np, triangle_id_np = tm.proximity.closest_point(mesh_tm, query_np)
    normals_np = mesh_tm.face_normals[triangle_id_np].astype(np.float32)

    origins_wp = points_to_warp(closest_np, mesh_wp.device)
    directions_wp = points_to_warp(normals_np, mesh_wp.device)
    distances_wp_np = od.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, closest_np, normals_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)


def test_contains_points(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: inside/outside against ``Trimesh.contains``, over three deliberately-placed groups.

    Points near the centre, just outside the bounds, and far away -- the third group is what
    catches a ray that runs out of range rather than reporting a miss.
    """
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(7)

    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    outside_np = rng.random((50, 3)) * 4.0 + mesh_tm.bounds[1]
    far_np = rng.random((30, 3)) * 100.0 + 1.0 + mesh_tm.bounds[1]
    center_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)

    points_wp = points_to_warp(inside_np, mesh_wp.device)
    contains_wp = od.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(inside_np))

    points_wp = points_to_warp(outside_np, mesh_wp.device)
    contains_wp = od.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(outside_np))

    points_wp = points_to_warp(far_np, mesh_wp.device)
    contains_wp = od.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(far_np))

    points_wp = points_to_warp(center_np, mesh_wp.device)
    contains_wp = od.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(center_np))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("winding_number", "pyvista")
def test_contains_points_matches_pyvista(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: ``select_interior_points`` is the same inside/outside predicate, point for point.

    VTK's answer is a per-point bool in the ``selected_points`` array of the returned cloud (the
    older ``SelectedPoints`` spelling is deprecated in pyvista 0.48), so the only transform is
    reading it out of the right array and casting to ``bool``. Measured **1.000** agreement on
    2 000 uniform queries against ``icosphere(3)``, 604 of them inside.

    The three fixtures are the ones a ray-parity rule should not fail on: a convex solid, a hollow
    shell whose interior is *outside*, and a genus-1 hole. The counts are asserted non-trivial in
    both directions, so a predicate that answered a constant would fail rather than agree -- which
    is why the queries are drawn from a box **grown 40%** about the centre rather than from
    ``mesh_tm.bounds``: on ``cave_cube`` the cavity is 0.1 of a unit box, so a bbox-uniform sample
    lands inside the solid 600 times out of 600 and the comparison would be a constant.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(11)
    centre_np = mesh_tm.bounds.mean(axis=0)
    extent_np = 1.4 * (mesh_tm.bounds[1] - mesh_tm.bounds[0])
    points_np = centre_np - 0.5 * extent_np + rng.random((600, 3)) * extent_np

    selected_pv = points_to_pyvista(points_np).select_interior_points(trimesh_to_pyvista(mesh_tm))
    contains_pv = np.asarray(selected_pv.point_data["selected_points"]).astype(bool)
    assert 0 < int(contains_pv.sum()) < len(points_np)  # both answers present

    points_wp = points_to_warp(points_np, mesh_wp.device)
    assert np.array_equal(od.ray.contains_points(mesh_wp, points_wp).numpy(), contains_pv)


def test_contains_cavity(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the origin sits in ``cave_cube``'s hollow, so it must read *outside*.

    The one case a bounding-box or convex test gets wrong, and the reason this fixture exists.
    trimesh agrees, so the comparison is direct rather than an asserted constant.
    """
    mesh_tm, mesh_wp = cave_cube
    origin_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    points_wp = points_to_warp(origin_np, mesh_wp.device)
    contains_wp = od.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(origin_np))


def test_contains_empty_points(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points_wp = warp_empty(0, wp.vec3, mesh_wp.device)
    assert od.ray.contains_points(mesh_wp, points_wp).numpy().shape == (0,)
