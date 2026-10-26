"""
Tests for point-cloud surface reconstruction, in ``ordito.reconstruction``'s own order.

Each of the six entry points has its own section below, and each names its reference there --
``scipy.spatial.Delaunay``, MeshLib's ``triangulatePointCloud``, open3d and pymeshlab for Poisson,
an analytic field for marching cubes, pymeshlab and igl for resampling, and open3d as a loose
face-count check for ball pivoting. Where no library computes the same vertex set, which is most of
them, the comparison is metric or topological -- watertightness, edge-manifoldness, Euler
characteristic, surface closeness -- rather than vertex-for-vertex.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Literal, cast

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.spatial import ConvexHull, cKDTree

import ordito as od
from ordito import _launch
from ordito.kernels import reconstruction as kernel_reconstruction
from ordito.kernels.algorithms import ball_pivoting as kernel_bpa
from tests.comparisons import (
    assert_unordered_rows_equal,
    canonical_winding,
    edge_multiplicity,
    hausdorff_surface_two_sided,
    hausdorff_two_sided,
    lexsort_rows,
    symmetric_surface_distance,
    undirected_edges,
)
from tests.conftest import CLOSED_MESHES, ROUND_FRAME_SCALE, ROUND_FRAME_SHIFT, saddle_graded_arrays
from tests.conversions import (
    meshlib_to_trimesh,
    numpy_to_meshlib,
    numpy_to_warp,
    open3d_to_trimesh,
    points_to_meshlib,
    points_to_open3d,
    points_to_pymeshlab,
    points_to_pyvista,
    points_to_warp,
    points_to_warp_uv,
    warp_empty,
    warp_to_trimesh,
)


def _meshlib_triangulate(
    points_np: np.ndarray, normals_np: np.ndarray | None, num_neighbours: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a reference mesh via MeshLib, returning ``(vertices, faces)`` numpy arrays.

    With ``normals_np=None`` MeshLib estimates and orients the normals itself.
    """
    cloud_ml = points_to_meshlib(points_np, normals_np)
    params_ml = mm.TriangulationParameters()
    params_ml.numNeighbours = num_neighbours
    mesh_ml = mm.triangulatePointCloud(cloud_ml, params_ml)
    return np.asarray(mn.getNumpyVerts(mesh_ml)), np.asarray(mn.getNumpyFaces(mesh_ml.topology))


def _sphere_cloud(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    sphere_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return (sphere_tm.vertices.astype(np.float64), sphere_tm.vertex_normals.astype(np.float64))


def _torus_grid_cloud() -> tuple[np.ndarray, np.ndarray, float]:
    """Return a quad-grid torus's vertices and normals, and its mean edge length."""
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=48, minor_sections=24
    )
    mean_edge = float(
        np.linalg.norm(
            torus_tm.vertices[torus_tm.edges_unique[:, 0]]
            - torus_tm.vertices[torus_tm.edges_unique[:, 1]],
            axis=1,
        ).mean()
    )
    return (
        torus_tm.vertices.astype(np.float64),
        torus_tm.vertex_normals.astype(np.float64),
        mean_edge,
    )


def _to_warp(points_np: np.ndarray, normals_np: np.ndarray, device: str):
    """Upload an oriented cloud as one pair, since every Poisson test needs both buffers."""
    return points_to_warp(points_np, device), points_to_warp(normals_np, device)


def _bpa_radius(points_wp: wp.array[wp.vec3]) -> float:
    """
    Return the ball-pivoting wrapper's own auto-radius: 1.5x the mean distance to six neighbours.

    Recomputed on the host and passed explicitly, so every call of a test pivots the same ball.
    """
    spacing = od.neighbors.query_nearest(points_wp, points_wp, k=7, backend="bvh")[1].numpy()[:, 1:]
    return 1.5 * float(spacing[np.isfinite(spacing) & (spacing > 0.0)].mean())


# ---------------------------------------------------------------------------
# 2D Delaunay triangulation (reference: scipy.spatial.Delaunay)
# ---------------------------------------------------------------------------


def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _edge_set(faces_flat: np.ndarray) -> np.ndarray:
    return np.unique(undirected_edges(np.asarray(faces_flat).reshape(-1, 3)), axis=0)


def _incircle_violations(points_np: np.ndarray, faces_flat: np.ndarray) -> int:
    """Count interior edges whose opposite apex lies inside the adjacent triangle circumcircle."""
    faces = faces_flat.reshape(-1, 3)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(faces):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault((int(min(a, b)), int(max(a, b))), []).append(fi)

    def in_circle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
        m = np.array(
            [
                [a[0] - d[0], a[1] - d[1], (a[0] - d[0]) ** 2 + (a[1] - d[1]) ** 2],
                [b[0] - d[0], b[1] - d[1], (b[0] - d[0]) ** 2 + (b[1] - d[1]) ** 2],
                [c[0] - d[0], c[1] - d[1], (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2],
            ]
        )
        return np.linalg.det(m)

    violations = 0
    for (u, v), fs in edge_faces.items():
        if len(fs) != 2:
            continue
        apex = []
        for fi in fs:
            apex.extend([int(x) for x in faces[fi] if int(x) not in (u, v)])
        if len(apex) != 2:
            continue
        d0, d1 = apex
        a, b, c = points_np[u], points_np[v], points_np[d0]
        if _cross2(b - a, c - a) < 0:
            a, b = b, a
        if in_circle(a, b, c, points_np[d1]) > 1e-9:
            violations += 1
    return violations


def _random_points_2d() -> np.ndarray:
    return np.random.default_rng(42).random((200, 2)).astype(np.float32)


def _perturbed_grid_2d() -> np.ndarray:
    rng = np.random.default_rng(11)
    grid = np.stack(np.meshgrid(np.arange(8.0), np.arange(8.0)), axis=-1).reshape(-1, 2)
    return (grid + rng.normal(0.0, 0.05, grid.shape)).astype(np.float32)


@pytest.mark.parametrize(
    "make_points",
    [
        pytest.param(_random_points_2d, id="random"),
        pytest.param(_perturbed_grid_2d, id="perturbed_grid"),
    ],
)
@pytest.mark.parity("delaunay_triangulation", "scipy")
def test_delaunay_matches_scipy(device: str, make_points: Callable[[], np.ndarray]):
    """
    Class B: the same triangulation as Qhull's, compared as an undirected edge set.

    Equality with Qhull's edge set implies the empty-circumcircle property and the convex-hull
    coverage; what an undirected edge set cannot see is orientation, so every triangle is also
    asserted counter-clockwise. On a uniform random cloud and on a jittered grid, whose
    near-cocircular quads are where a sloppy in-circle predicate would pick the other diagonal.
    """
    from scipy.spatial import Delaunay

    points_np = make_points()
    points_wp = points_to_warp_uv(points_np, device)

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy()
    faces_sp = Delaunay(points_np.astype(np.float64)).simplices

    assert np.array_equal(_edge_set(faces_wp), _edge_set(faces_sp.reshape(-1)))
    tris = points_np.astype(np.float64)[faces_wp.reshape(-1, 3)]
    assert np.all(_cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]) > 0.0)


@pytest.mark.parity("delaunay_triangulation", "pyvista")
def test_delaunay_contains_the_pyvista_triangulation(device: str):
    """
    Class B: ``delaunay_2d``'s triangles are a strict **subset** of ordito's; the gap is hull.

    Both are Delaunay triangulations of the same cloud, so on the interior they agree face for face;
    they differ on how much of the convex hull they keep. Measured on 200 uniform points: pyvista
    returns **374** triangles, every one of them present in ordito's **384**, so the 10 extra are
    hull slivers ``vtkDelaunay2D`` drops. The comparison is therefore containment plus a bound on
    the excess, not equality -- and stating it that way is the point, since an equality assert would
    fail on a correct implementation.

    The cloud is passed to VTK as ``(x, y, 0)``: ``delaunay_2d`` projects onto the best-fit plane,
    so a genuinely planar input is what keeps its answer comparable to a 2-D triangulator's.
    """
    rng = np.random.default_rng(0)
    points_np = rng.random((200, 2))
    points_wp = points_to_warp_uv(points_np, device)

    triangulated_pv = points_to_pyvista(np.column_stack([points_np, np.zeros(len(points_np))]))
    triangulated_pv = triangulated_pv.delaunay_2d()
    assert triangulated_pv.is_all_triangles
    faces_pv = {
        tuple(row) for row in np.sort(np.asarray(triangulated_pv.regular_faces), axis=1).tolist()
    }
    assert len(faces_pv) > 300, "the reference triangulated the cloud before it is compared to"

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    faces_od = {tuple(row) for row in np.sort(faces_wp, axis=1).tolist()}

    assert faces_pv <= faces_od
    # The excess is hull slivers only: a few triangles, not a different triangulation, and each
    # one has a corner on the convex hull (10 of 10 on this seed; VTK can peel a second layer on
    # other clouds, so that is a measured fact of this one, not a law).
    excess = faces_od - faces_pv
    assert len(excess) < 0.05 * len(faces_od)
    hull = set(ConvexHull(points_np).vertices.tolist())
    assert all(hull.intersection(face) for face in excess)


def test_delaunay_matches_warp_edge_flip_from_the_same_seed(device: str):
    """
    Class A: the identical face set as ``warp.geometry.delaunay_edge_flip`` run on the same seed.

    Both sides flip the lexicographic seed triangulation to Delaunay, so this pins ordito's flip
    loop (its claims, its in-circle predicate, its incremental adjacency) against an independent
    implementation of the same phase, face for face, at a size Qhull's edge-set comparison above
    does not reach. The seed itself is ordito's and is covered by the scipy comparison; random
    points in general position have one Delaunay triangulation, so the shared seed cannot make
    the comparison agree by construction.
    """
    import warp.geometry

    rng = np.random.default_rng(20_000)
    points_np = rng.random((20_000, 2)).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_od = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    seed_np = od.reconstruction._lexicographic_triangulation(points_wp)  # pyright: ignore[reportPrivateUsage]
    faces_warp_wp = wp.array(np.asarray(seed_np).reshape(-1, 3), dtype=wp.int32, device=device)
    # Warp annotates the faces ``wp.array2d``, a static helper no runtime array is typed as.
    flips_wp = warp.geometry.delaunay_edge_flip(
        points_wp, cast("wp.array2d[wp.int32]", faces_warp_wp)
    )

    assert int(flips_wp.numpy()[0]) > 0, "the reference had a non-Delaunay seed to repair"
    assert np.array_equal(
        lexsort_rows(np.sort(faces_od, axis=1)),
        lexsort_rows(np.sort(faces_warp_wp.numpy(), axis=1)),
    )


@pytest.mark.parametrize("layout", ["random", "perturbed_grid"])
def test_delaunay_native_flip_start_changes_nothing(
    device: str, monkeypatch: pytest.MonkeyPatch, layout: str
):
    """
    Ordito against ordito: starting the flips from Warp's own pass returns the same faces.

    Past ``_NATIVE_DELAUNAY_FLIP_FROM`` points on CUDA, ``warp.geometry.delaunay_edge_flip`` runs
    before the exact flip loop, which then repairs whatever that faster pass left; the scipy
    comparison above carries the oracle for the loop. The threshold is forced to zero here so a
    small input takes that path, on a random cloud and on a jittered grid whose near-cocircular
    quads are where two in-circle predicates disagree.
    """
    rng = np.random.default_rng(3)
    if layout == "random":
        points_np = rng.random((20_000, 2)).astype(np.float32)
    else:
        grid_np = np.stack(np.meshgrid(np.arange(60.0), np.arange(60.0)), axis=-1).reshape(-1, 2)
        points_np = (grid_np + rng.normal(0.0, 1e-3, grid_np.shape)).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_loop = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    monkeypatch.setattr(od.reconstruction, "_NATIVE_DELAUNAY_FLIP_FROM", 0)
    faces_native = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    assert faces_loop.shape[0] > 1000
    assert np.array_equal(
        lexsort_rows(np.sort(faces_loop, axis=1)), lexsort_rows(np.sort(faces_native, axis=1))
    )


def test_delaunay_cocircular(device: str):
    # Regular 12-gon plus centre: many cocircular quadruples; compare on invariants, not triangles.
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    points_np = np.vstack([ring, [[0.0, 0.0]]]).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    assert _incircle_violations(points_np.astype(np.float64), faces_wp) == 0

    from scipy.spatial import ConvexHull

    tris = points_np.astype(np.float64)[faces_wp]
    area = float(np.abs(_cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])).sum() * 0.5)
    assert np.isclose(area, float(ConvexHull(points_np.astype(np.float64)).volume), rtol=1e-5)


def test_delaunay_collinear_and_too_few(device: str):
    # Collinear points triangulate to nothing; fewer than three raise.
    points_np = np.array([[float(i), 0.0] for i in range(5)], dtype=np.float32)
    assert od.reconstruction.delaunay_triangulation(points_to_warp_uv(points_np, device)).size == 0
    points_wp = wp.array(np.zeros((2, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    with pytest.raises(ValueError, match="at least 3 points"):
        od.reconstruction.delaunay_triangulation(points_wp)


# ---------------------------------------------------------------------------
# triangulate_point_cloud (reference: MeshLib ``triangulatePointCloud``)
# ---------------------------------------------------------------------------


@pytest.mark.parity("triangulate_point_cloud", "meshlib")
def test_sphere_matches_meshlib_reference(device: str):
    """
    Class B (unordered rows): on a clean uniform cloud the two triangulations are *identical*.

    Not a count bound, which is what this test used to assert. Measured on a 642-point icosphere
    cloud at ``num_neighbours=18``: both sides return 1 280 faces and the face **sets** are equal
    row for row after canonicalizing winding and lexsorting -- the greedy fan optimization
    reproduces MeshLib's local triangulation exactly, not merely to a face count. Both keep the
    input points as their vertices, so the indices are directly comparable with no remap.

    The count bound is kept as the headline (it is what fails first and most legibly) and the set
    equality is what carries the claim; ``num_neighbours=8`` gives the same answer, so the agreement
    is not a property of one parameter value. The tie-breaking divergence that does exist shows up
    on a quad-grid cloud instead, where the diagonal choice is genuinely free --
    [`test_torus_matches_meshlib_reference`] pins that one as a surface distance.

    Not a library comparison, on the same reconstruction: a closed genus-0 triangulation of ``n``
    points has exactly ``2 n - 4`` faces (Euler), is watertight and edge-manifold, references every
    input point, and its vertices lie on the unit sphere.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_ml, faces_ml = _meshlib_triangulate(points_np, normals_np, 18)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18
    )
    faces_wp_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)
    faces_ml_np = np.asarray(faces_ml, dtype=np.int64)

    assert faces_ml_np.shape[0] > 0  # non-vacuity: the reference reconstructed something
    assert abs(faces_wp_np.shape[0] - faces_ml_np.shape[0]) <= max(2, faces_ml_np.shape[0] // 100)
    # Both index the input cloud, so the face sets are comparable without a vertex remap.
    assert np.allclose(vertices_ml, points_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertices_wp.numpy(), points_np, rtol=1e-5, atol=1e-5)
    assert np.array_equal(
        lexsort_rows(canonical_winding(faces_wp_np)), lexsort_rows(canonical_winding(faces_ml_np))
    )

    n_points = points_np.shape[0]
    assert faces_wp_np.shape[0] == 2 * n_points - 4
    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert od.validation.is_edge_manifold(faces_wp)
    assert od.measures.euler_characteristic(faces_wp) == 2
    assert np.unique(faces_wp_np).size == n_points
    radii = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.allclose(radii, 1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("triangulate_point_cloud", "meshlib")
def test_torus_matches_meshlib_reference(device: str):
    """
    Class C (a surface distance): the input class where the *diagonal* choice is genuinely free.

    A quad-grid cloud has no canonical triangulation -- each quad can be split either way at equal
    cost -- so unlike the icosphere above the two libraries do not agree row for row: measured
    **268 of 2 304** faces differ on CUDA, 166 on the CPU. What does agree is the face count
    (2 304 exactly, both devices) and the surface: each grid quad of a torus is a planar trapezoid,
    so either diagonal spans the same plane and the two-sided Hausdorff distance is round-off,
    measured **3.6e-07** (CUDA) and **4.0e-07** (CPU) of the 0.128 mean edge. The bound is 3x the
    worse, ``1.2e-06`` of the mean edge -- a quad split off its plane would not pass it.

    Mutation probe and margin: dropping ``num_neighbours`` to 4 leaves ordito with 240 faces of
    1 280 on the sphere cloud and a Hausdorff distance of **0.348**, 2.3x the mean edge.

    Not a library comparison, on the same reconstruction: the result is a watertight,
    edge-manifold genus-1 surface (``V - E + F = 0``).
    """
    points_np, normals_np, mean_edge = _torus_grid_cloud()
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_ml, faces_ml = _meshlib_triangulate(points_np, normals_np, 16)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=16
    )
    faces_wp_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)
    faces_ml_np = np.asarray(faces_ml, dtype=np.int64)

    assert faces_ml_np.shape[0] > 0  # non-vacuity
    assert faces_wp_np.shape[0] == faces_ml_np.shape[0]
    assert (
        hausdorff_surface_two_sided(
            vertices_wp.numpy().astype(np.float64),
            faces_wp_np,
            np.asarray(vertices_ml, dtype=np.float64),
            faces_ml_np,
        )
        < 1.2e-6 * mean_edge
    )
    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert od.validation.is_edge_manifold(faces_wp)
    assert od.measures.euler_characteristic(faces_wp) == 0


@pytest.mark.parity("triangulate_point_cloud", "meshlib")
def test_graded_open_cloud_matches_meshlib_reference(device: str):
    """
    Class C (a statistic of the face sets): the share of MeshLib's faces ordito reproduces.

    On a graded open cloud -- ``saddle_graded``'s vertices with its vertex normals, needles of
    aspect ~4 800 and a rim -- a fan over the 16 nearest neighbours often misses a neighbour its
    Delaunay fan needs, so MeshLib rebuilds such fans from a wider search
    (``automaticRadiusIncrease``, on by default). Excludes the bug class "a fan is never rebuilt,
    or rebuilt from the wrong radius". Measured on the CPU: ordito reproduces **0.9932** of
    MeshLib's 2 046 faces and returns 2 046 (28 faces differ); the bound 0.979 is 3.1x the
    measured disagreement. Mutation probe: without the rebuild (the previous ordito) it reproduced
    0.8534 with 1 754 faces, failing both asserts. Both index the input points, matched by
    position since ordito drops an unreferenced point.
    """
    vertices_np, faces_np = saddle_graded_arrays()
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    used = np.unique(faces_np)
    points_np = np.asarray(mesh_tm.vertices[used], dtype=np.float64)
    normals_np = np.asarray(mesh_tm.vertex_normals[used], dtype=np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _vertices_ml, faces_ml = _meshlib_triangulate(points_np, normals_np, 16)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=16
    )
    distance, input_index = cKDTree(points_np).query(vertices_wp.numpy().astype(np.float64))
    assert np.max(distance) < 1e-5  # every vertex is an input point

    def row_set(rows: np.ndarray) -> set[tuple[int, ...]]:
        return {tuple(row) for row in np.sort(rows, axis=1).tolist()}

    faces_ml_set = row_set(np.asarray(faces_ml, dtype=np.int64))
    faces_wp_set = row_set(np.asarray(input_index)[faces_wp.numpy().reshape(-1, 3)])
    assert len(faces_ml_set) > 0  # non-vacuity
    assert len(faces_wp_set & faces_ml_set) >= 0.979 * len(faces_ml_set)
    assert abs(len(faces_wp_set) - len(faces_ml_set)) <= 0.01 * len(faces_ml_set)


@pytest.mark.parity("triangulate_point_cloud_unoriented", "meshlib")
def test_torus_without_normals_matches_meshlib_reference(device: str):
    """
    Class C (a surface distance and a volume): neither side is given normals.

    The torus is not star-shaped about its centroid, so normals pointed away from the centroid
    disagree in sign across each side of the tube; MeshLib orients its estimated normals over the
    whole cloud instead. The two surfaces agree as in
    [`test_torus_matches_meshlib_reference`] (diagonal choices differ), and the signed volume,
    which only a consistently and outward wound surface carries, agrees too: MeshLib's is the
    torus's own 2.3837. Measured, the same on both devices but for the volume: 2 304 faces against
    MeshLib's 2 308 (four extra on its side, held exactly), a Hausdorff distance of 3.4e-07 of the
    mean edge (bound 1.2e-06), and the volume to 3.4e-08 relative on CUDA and 1.6e-06 on the CPU
    (bound 5e-06).

    Mutation probes and margins: normals oriented away from the centroid, as this path once used,
    returned 2 016 faces with holes and a volume of 3.38 (off by 42 %, against the 1 % bound);
    reading each neighbour's normal on the center's side alone still left eight five-edge holes
    (2 280 faces), because a flipped center normal reverses its fan's ring and the Delone profit,
    rounded in ring order, then picked the other diagonal of the grid's exactly tied quads.

    Not a library comparison, on the same reconstruction: consistently wound and closed.
    """
    points_np, _, mean_edge = _torus_grid_cloud()

    vertices_ml, faces_ml = _meshlib_triangulate(points_np, None, 16)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_to_warp(points_np, device), num_neighbours=16
    )
    faces_wp_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)
    faces_ml_np = np.asarray(faces_ml, dtype=np.int64)

    assert faces_ml_np.shape[0] > 0  # non-vacuity
    assert faces_ml_np.shape[0] - faces_wp_np.shape[0] == 4
    assert (
        hausdorff_surface_two_sided(
            vertices_wp.numpy().astype(np.float64),
            faces_wp_np,
            np.asarray(vertices_ml, dtype=np.float64),
            faces_ml_np,
        )
        < 1.2e-6 * mean_edge
    )
    volume_ml = tm.Trimesh(vertices_ml, faces_ml_np, process=False).volume
    assert volume_ml > 0.0  # non-vacuity: MeshLib's surface faces outward
    assert math.isclose(od.measures.volume(vertices_wp, faces_wp), volume_ml, rel_tol=5e-6)
    assert od.validation.is_winding_consistent(faces_wp)
    assert od.validation.is_watertight(vertices_wp, faces_wp)


@pytest.mark.parity("triangulate_point_cloud_unoriented", "meshlib")
@pytest.mark.parametrize("seed", [0, 1])
def test_open_cloud_without_normals_faces_like_meshlib(device: str, seed: int):
    """
    Class B (faces matched by vertex set, windings compared): an open surface faces MeshLib's way.

    A closed surface has an outside; an open one does not, so without normals the side it faces
    is a convention. MeshLib's (``autoOrientLocalTriangulations``) turns each connected component
    away from the cloud's bounding-box centre at the component's point farthest from that centre.
    The cloud is a sine-wave sheet, not convex, longer on one side so that the farthest point is
    unique, and shuffled by ``seed`` so that the point order is not the answer. Every face the two
    triangulations share by vertex set must carry the same winding: measured 1 445 and 1 411 shared
    faces of MeshLib's 1 919 on CUDA, 1 385 and 1 343 on the CPU (the grid's diagonals differ
    elsewhere), and all six shuffles 0-5 agree on both devices.

    Mutation probe: keeping only the consistent winding, each component following its
    lowest-indexed face, reversed every shared face for shuffles 0 and 5 of 0-5 and passed the
    rest; shuffle 0 fails and shuffle 1 passes, so a constant answer cannot pass both.
    """
    x, y = np.meshgrid(np.arange(-2.0, 2.61, 0.1), np.arange(-1.0, 1.01, 0.1))
    sheet_np = np.stack([x.ravel(), y.ravel(), 0.4 * np.sin(np.pi * x.ravel())], axis=1)
    points_np = sheet_np[np.random.default_rng(seed).permutation(sheet_np.shape[0])]

    _vertices_ml, faces_ml = _meshlib_triangulate(points_np, None, 18)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_to_warp(points_np, device), num_neighbours=18
    )
    # Both index the input cloud, so the face sets are comparable without a vertex remap.
    assert np.allclose(vertices_wp.numpy(), points_np, rtol=1e-5, atol=1e-5)
    faces_wp_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)
    faces_ml_np = np.asarray(faces_ml, dtype=np.int64)
    assert faces_ml_np.shape[0] > 0  # non-vacuity

    def row_set(rows: np.ndarray) -> set[tuple[int, ...]]:
        return {tuple(row) for row in rows.tolist()}

    shared = row_set(np.sort(faces_wp_np, axis=1)) & row_set(np.sort(faces_ml_np, axis=1))
    agreeing = row_set(canonical_winding(faces_wp_np)) & row_set(canonical_winding(faces_ml_np))
    assert len(shared) >= 0.6 * faces_ml_np.shape[0]
    assert len(agreeing) == len(shared)
    assert od.validation.is_winding_consistent(faces_wp)


def test_open_hemisphere_keeps_single_boundary(device: str):
    sphere_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    upper = sphere_tm.vertices[:, 2] >= -1e-9
    points_np = sphere_tm.vertices[upper].astype(np.float64)
    normals_np = sphere_tm.vertex_normals[upper].astype(np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18
    )
    assert od.validation.is_edge_manifold(faces_wp)
    # The intended equator rim is one large boundary loop, not filled and not fragmented. The
    # rim's fans search twice as far (MeshLib's automatic radius increase), which can leave small
    # gaps beside the rim: slivers between it and a chord across three of its points, or a notch
    # touching it at a vertex, which is its own loop. MeshLib leaves one sliver here (rim
    # perimeter 6.27, sliver 0.69); ordito two slivers and a 4-vertex notch.
    loops = sorted(od.boundary.boundary_loops(vertices_wp, faces_wp), key=lambda loop: -loop.size)
    heights = np.abs(vertices_wp.numpy()[:, 2])
    n_equator = int(np.count_nonzero(np.abs(points_np[:, 2]) < 1e-6))
    rim_np = loops[0].numpy()
    assert rim_np.size >= 0.85 * n_equator
    assert (heights[rim_np] < 1e-6).all()
    for gap in loops[1:]:
        assert gap.size <= 4
        assert heights[gap.numpy()].max() < 0.08  # within one edge of the rim


def test_build_local_triangulations_two_neighbour_ring_no_duplicate(device: str):
    """
    A fan with exactly two live neighbours and no boundary gap emits one triangle, not two.

    Direct ``kernel_reconstruction.build_local_triangulations`` launch (bypassing the neighbour
    query and normal estimation) so the degenerate ring -- ``cycle_prev(i) == cycle_next(i)`` for
    both of the two live slots -- is reached deterministically rather than by engineering the
    greedy edge-removal loop's convergence. ``boundary_angle`` near ``2*pi`` keeps this specific
    two-point ring from being read as a boundary (its two angular gaps sum to ``2*pi``, so neither
    can exceed the default ``0.9*pi`` threshold, which is what makes the ordinary wrapper unable to
    reach this branch with only two neighbours -- see the regression-fixture note in
    ``plans/review.md``).
    """
    angle = np.deg2rad(170.0)
    points_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.cos(angle), np.sin(angle), 0.0]], dtype=np.float64
    )
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.tile([0.0, 0.0, 1.0], (3, 1)), dtype=wp.vec3, device=device)
    neighbor_idx_wp = wp.array(
        np.array([[1, 2, -1]], dtype=np.int32), dtype=wp.int32, device=device
    )
    neighbor_dist_wp = wp.array(
        np.array([[1.0, 1.0, 0.0]], dtype=np.float32), dtype=wp.float32, device=device
    )
    out_tris_wp = wp.zeros((1, 3, 3), dtype=wp.int32, device=device)
    # Seeded off zero, so the assert below also shows the kernel wrote the count itself.
    out_counts_wp = wp.full(1, -1, dtype=wp.int32, device=device)
    out_grown_radius_wp = wp.full(1, -1.0, dtype=wp.float32, device=device)
    wp.launch(
        kernel_reconstruction.build_local_triangulations,
        dim=1,
        inputs=[
            points_wp,
            normals_wp,
            neighbor_idx_wp,
            neighbor_dist_wp,
            wp.float32(0.0),
            wp.float32(math.pi / 2.0),
            wp.float32(2.0 * math.pi - 0.01),
            kernel_reconstruction.NORMALS_ORIENTED,
            out_tris_wp,
            out_counts_wp,
            out_grown_radius_wp,
        ],
        device=device,
    )
    assert out_counts_wp.numpy()[0] == 1
    triangle = out_tris_wp.numpy()[0, 0]
    assert set(triangle.tolist()) == {0, 1, 2}
    # The 170-degree wedge's circumcircle reaches far past the unit search radius, so the fan grows
    # its search to MeshLib's cap, twice that radius.
    assert out_grown_radius_wp.numpy()[0] == 2.0


def test_build_local_triangulations_two_gaps_emits_nothing(device: str):
    """
    A fan with two disjoint angular gaps emits no triangle rather than bridging them.

    Direct kernel launch (bypassing the neighbour query): six neighbours in two tight clusters
    (0-2 degrees and 180-182 degrees) leave two gaps of ~178 degrees each, both past the default
    ``boundary_angle`` (0.9*pi = 162 degrees). ``border`` can mark only one gap, so before this
    guard a spurious triangle could connect the two clusters across the other, unregistered one;
    now the point contributes nothing instead.
    """
    angles_deg = [0.0, 1.0, 2.0, 180.0, 181.0, 182.0]
    points_np = np.array(
        [[0.0, 0.0, 0.0]]
        + [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0] for a in angles_deg],
        dtype=np.float64,
    )
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    normals_wp = wp.array(
        np.tile([0.0, 0.0, 1.0], (points_np.shape[0], 1)), dtype=wp.vec3, device=device
    )
    neighbor_idx_wp = wp.array(
        np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32), dtype=wp.int32, device=device
    )
    neighbor_dist_wp = wp.array(np.ones((1, 6), dtype=np.float32), dtype=wp.float32, device=device)
    out_tris_wp = wp.zeros((1, 6, 3), dtype=wp.int32, device=device)
    # Seeded off zero: the early return this fan takes must still report its (empty) count.
    out_counts_wp = wp.full(1, -1, dtype=wp.int32, device=device)
    out_grown_radius_wp = wp.full(1, -1.0, dtype=wp.float32, device=device)
    wp.launch(
        kernel_reconstruction.build_local_triangulations,
        dim=1,
        inputs=[
            points_wp,
            normals_wp,
            neighbor_idx_wp,
            neighbor_dist_wp,
            wp.float32(0.0),
            wp.float32(math.pi / 2.0),
            wp.float32(0.9 * math.pi),
            kernel_reconstruction.NORMALS_ORIENTED,
            out_tris_wp,
            out_counts_wp,
            out_grown_radius_wp,
        ],
        device=device,
    )
    assert out_counts_wp.numpy()[0] == 0
    # Its gaps are borders, so it searches again in twice its unit radius.
    assert out_grown_radius_wp.numpy()[0] == 2.0


def test_holes_seal_small_hole(device: str):
    rng = np.random.default_rng(0)
    points_np, normals_np = _sphere_cloud(4)
    # Remove a small cluster of points to open a genuine boundary hole. Wider than the doubled
    # search of the rim's fans, which triangulate straight across a 0.2 puncture (MeshLib's do
    # too); at 0.35 both leave the same 7-vertex hole.
    seed = points_np[rng.integers(points_np.shape[0])]
    keep = np.linalg.norm(points_np - seed, axis=1) > 0.35
    points_np, normals_np = points_np[keep], normals_np[keep]
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    # No filling: the punctured region stays an open boundary.
    vertices_open, faces_open = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18, crit_hole_length=0.0
    )
    assert not od.validation.is_watertight(vertices_open, faces_open)

    # Large threshold: the hole is sealed into a watertight, manifold mesh.
    vertices_filled, faces_filled = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18, crit_hole_length=10.0
    )
    assert od.validation.is_edge_manifold(faces_filled)
    assert od.validation.is_watertight(vertices_filled, faces_filled)


def test_too_few_points(device: str):
    """
    Not a library comparison: two points admit no triangle, and nothing else computes this.

    Also pins that the returned vertices are the function's **own** buffer on this path. Two points
    reach the "no candidate survived" early return, which used to hand the caller's array straight
    back while every other exit -- including the populated one, through ``_assemble_faces`` --
    clones. Writing into the result must not reach the input.
    """
    points_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=4)
    assert faces_wp.size == 0

    assert np.allclose(vertices_wp.numpy(), points_np, rtol=1e-5, atol=1e-5)
    vertices_wp.fill_(wp.vec3(9.0, 9.0, 9.0))
    assert np.allclose(points_wp.numpy(), points_np, rtol=1e-5, atol=1e-5)


def test_empty_cloud_and_invalid_parameters(device: str):
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        wp.zeros(0, dtype=wp.vec3, device=device)
    )
    assert vertices_wp.size == 0
    assert faces_wp.size == 0
    points_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="pass at most one of num_neighbours and radius"):
        od.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=8, radius=1.0)
    with pytest.raises(ValueError, match="max_neighbours must be <="):
        od.reconstruction.triangulate_point_cloud(
            points_wp, max_neighbours=kernel_reconstruction.MAX_NEIGHBOURS + 1
        )


# ======================================================================================
# Screened-Poisson reconstruction (screened_poisson)
#
# References: open3d ``create_from_point_cloud_poisson`` (``_o3d``) and PyMeshLab
# ``generate_surface_reconstruction_screened_poisson`` (``_pml``). Both reconstruct a different
# vertex set than ordito, so every comparison is metric/topological, never vertex-for-vertex.
# ======================================================================================


def _poisson_depth(device: str) -> int:
    """
    Octree depth for the solving Poisson tests: 6 on CUDA, 5 on the CPU device.

    Not a correctness difference -- CPU and CUDA reconstruct the same surface, which
    [`test_poisson_cpu_matches_cuda`][] pins vertex for vertex -- but a cost one. The ``dense``
    solve is over the ``2 ** depth`` cubed node grid **whatever the cloud size**, so it octuples per
    level, and the CPU backend is not close to CUDA on it.

    Measured with the two depths **interleaved in one process** and read as the minimum of
    three, dropping a level is an **8.6x** saving per CPU solve, matching the 8x the grid size
    predicts. Both are negligible on CUDA.

    !!! warning "Quote the ratio, not the wall clock"
        This box's CPU timings swing ~40 % with background load, and the same depth-6 solve has been
        seen to spread more than fourfold across full-suite runs. The *ratio* is stable because both
        sides move together, which is exactly why one wants an interleaved A/B rather than
        two numbers taken apart.

    Depth 4 is not an option: 2 144 faces is too coarse to resolve the torus hole.

    **Three tests opt out and keep a depth-6 literal**, each because its claim stops holding at 5,
    which is why the level is a helper and not a blanket edit:

    - ``test_poisson_watertight_manifold[sphere-dense]`` -- the depth-5 surface self-intersects, so
      it is not watertight on *either* device, and the radius tolerances are sized to a depth-6
      cell;
    - ``test_poisson_matches_open3d_metric`` and ``..._pymeshlab_metric`` -- the class-C margin
      falls
      from 3.4x / 3.7x to **2.64x / 2.44x**, under the 3x floor.

    Keep the *reference* libraries at whatever depth their own comment specifies -- open3d and
    pymeshlab return identical output at 5 and 6 on this cloud and are pinned to 5 on both devices,
    so the reference side does not move with this and no comparison is weakened on CPU only.
    """
    return 5 if wp.get_device(device).is_cpu else 6


def _torus_cloud(n_major: int = 40, n_minor: int = 20, r_major: float = 1.0, r_minor: float = 0.35):
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    uu = uu.ravel()
    vv = vv.ravel()
    cx = np.cos(uu)
    cy = np.sin(uu)
    px = (r_major + r_minor * np.cos(vv)) * cx
    py = (r_major + r_minor * np.cos(vv)) * cy
    pz = r_minor * np.sin(vv)
    points = np.stack([px, py, pz], axis=1)
    nx = np.cos(vv) * cx
    ny = np.cos(vv) * cy
    nz = np.sin(vv)
    normals = np.stack([nx, ny, nz], axis=1)
    return points.astype(np.float64), normals.astype(np.float64)


def _points_to_surface(points_np: np.ndarray, mesh: tm.Trimesh) -> float:
    """Mean distance from a point set to the nearest point on a mesh surface."""
    return float(np.abs(tm.proximity.signed_distance(mesh, points_np)).mean())


def _open3d_poisson(points_np: np.ndarray, normals_np: np.ndarray, depth: int) -> tm.Trimesh:
    """
    Open3D's screened Poisson, pinned to **one thread** -- which is what makes it affordable.

    ``n_threads`` defaults to ``-1``, meaning one thread per core, and on this 642-point cloud
    Kazhdan's solver is far below its parallel break-even: the barriers dominate and the wall clock
    grows monotonically with the thread count from the first doubling, reaching well over a
    hundredfold at the default. Measured one setting per process, and the answer does not move --
    **7 976 faces at every setting**.

    At one thread the CPU time is within a factor of two of the wall clock, so one thread is
    essentially the floor and no thread count can beat it by much; pinning costs nothing even on an
    idle box. Unpinned, this test also spread by a fifth between two runs twenty minutes apart --
    the spread is the reason for the pin, not the mean. The pymeshlab twin has the same defect and a
    much worse constant; see ``test_poisson_matches_pymeshlab_metric``.
    """
    pcd = points_to_open3d(points_np, normals_np)
    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1
    )
    return open3d_to_trimesh(mesh_o3d)


def test_poisson_dense_solve_converges_in_few_iterations(device: str) -> None:
    """
    Not a library comparison: no reference exposes its Poisson solver's iteration count.

    The dense grid's conjugate gradient is preconditioned by a geometric V-cycle, whose point is an
    iteration count that does not grow with the resolution: every level of the cascade must reach
    the default tolerance inside 30 iterations, which Jacobi -- 100 to 800 per level on this kind of
    input -- does not. A level that runs out warns, so promoting the warning to an error is the
    assertion. The second call is the converse: a budget of two iterations cannot converge, and
    has to say so rather than return a surface from an unconverged field silently. The converged
    surface must also be outward oriented, which a positive enclosed volume shows.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    depth = _poisson_depth(device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vertices_wp, faces_wp = od.reconstruction.screened_poisson(
            points_wp, normals_wp, depth=depth, full_depth=4, solver_iterations=30
        )
    assert warp_to_trimesh(vertices_wp, faces_wp).volume > 0.0
    with pytest.warns(UserWarning, match="iteration cap"):
        od.reconstruction.screened_poisson(
            points_wp, normals_wp, depth=depth, full_depth=4, solver_iterations=2
        )


@pytest.mark.parity(
    "screened_poisson",
    "open3d",
    benchmarked=False,
    reason="open3d wraps Kazhdan's CPU solver, which was 6 322 s across the open3d and pymeshlab "
    "screened_poisson rows -- 73 % of the whole benchmark suite -- and [dragon-open3d-*] ran 93 "
    "minutes without completing a round. Those rows were removed rather than capped; the "
    "comparison lives here, at a size a correctness test can afford.",
)
@pytest.mark.slow_cpu(114.9)
def test_poisson_matches_open3d_metric(device: str):
    """
    Class C: mean sample-to-surface distance, there being no vertex correspondence to compare.

    The two solvers march different octrees and return different meshes -- 32 552 faces against
    open3d's 7 976 on this cloud -- so no vertex, face or count agrees and only a surface metric
    can state the claim. Excludes the bug class "the iso-surface is in the wrong place": a global
    radius error, a sign flip on the indicator, a mislocated level set.

    Measured agreement is **0.0044** and the threshold is 0.015, a **3.4x** margin. Mutation
    probe, run against this assert: scaling ordito's answer by 0.98 makes it read 0.0190 and the
    assert fails, so a 2% radius error on a unit sphere is caught. A mesh against itself scores
    exactly 0.0, which is the whole point of the metric -- see the warning below.

    **Pinned to depth 6 on both devices**, unlike the rest of this section, which drops to 5 on CPU
    (see [`_poisson_depth`][]). Re-measured at depth 5: agreement widens to **0.0057**, a 2.64x
    margin, under the 3x floor for a class-C threshold. The mutation probe still
    fires there (0.0194), so it is the margin that fails the bar and not the sensitivity -- but a
    threshold at 2.6x its measured value is a latent flake, and saving 155 s of CPU time is not
    worth buying one.

    !!! warning "This assert used to be vacuous, and the reason generalises"
        It was ``symmetric_chamfer(...) < 0.03``, which is sample-to-*sample* and therefore has a
        noise floor of ``0.5 * sqrt(area / n_samples)`` = 0.028 on this fixture. The real
        disagreement, 0.0043, was two hundredths below that floor: the pass margin was 1.08x and
        a mesh compared with *itself* scored 0.0279. Any threshold within a few percent of a
        sampling floor is testing the sampler.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_od = warp_to_trimesh(vertices_wp, faces_wp)
    # depth=5 on the reference side, not 6: on this 642-point cloud open3d returns the *same*
    # 7 976 faces at both depths (measured) at a third of the cost, so the extra octree level is
    # cost without an answer. Only the reference drops -- ordito stays at depth 6, pinned on both
    # devices per the docstring, where it resolves 32 552 faces either way.
    mesh_o3d = _open3d_poisson(points_np, normals_np, depth=5)
    mean_distance, _ = symmetric_surface_distance(mesh_od, mesh_o3d)
    assert mean_distance < 0.015


@pytest.mark.parametrize(
    "method", [pytest.param("dense", marks=pytest.mark.slow_cpu(18.5), id="dense"), "adaptive"]
)
def test_poisson_screening_improves_fit(device: str, method: Literal["dense", "adaptive"]):
    """
    Ordito against ordito: screening ties the surface to the samples, so the fit cannot worsen.

    On both backends, each on the cloud it was tuned against (``adaptive`` resolves a finer one).
    These are also the only reconstructions at ``point_weight=0``, which is why the
    no-degenerate-face invariant is asserted here as well as in
    ``test_poisson_small_scale_cloud_is_watertight_without_degenerate_faces``. That config is
    ill-conditioned -- the operator is held SPD by a ``1e-4`` floor alone -- and the dense
    backend's raw marching-cubes output carried 27-64 zero-area triangles, run to run, before the
    extraction kept its vertices off the lattice nodes. They are not
    cosmetic: a zero-area face has no normal to orient, and trimesh's ``closest_point`` divides by
    its zero-length edge, so ``_points_to_surface`` below emitted an intermittent
    ``RuntimeWarning: invalid value encountered in divide``. The screened side is included so the
    assertion is not one-sided.
    """
    points_np, normals_np = _sphere_cloud(3 if method == "dense" else 4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_screened, faces_screened = od.reconstruction.screened_poisson(
        points_wp,
        normals_wp,
        depth=_poisson_depth(device),
        full_depth=4,
        point_weight=4.0,
        method=method,
    )
    vertices_unscreened, faces_unscreened = od.reconstruction.screened_poisson(
        points_wp,
        normals_wp,
        depth=_poisson_depth(device),
        full_depth=4,
        point_weight=0.0,
        method=method,
    )
    fit_screened = _points_to_surface(points_np, warp_to_trimesh(vertices_screened, faces_screened))
    fit_unscreened = _points_to_surface(
        points_np, warp_to_trimesh(vertices_unscreened, faces_unscreened)
    )
    # Screening ties the surface to the samples: the fit is at least as good.
    assert fit_screened <= fit_unscreened + 1e-4
    # Neither output may carry a zero-area triangle -- see the docstring. Asserted on the faces
    # actually returned, so this fails if the cleanup tail is dropped from either code path.
    for vertices_wp, faces_wp in (
        (vertices_screened, faces_screened),
        (vertices_unscreened, faces_unscreened),
    ):
        assert np.all(od.triangles.face_nondegenerate_mask(vertices_wp, faces_wp).numpy())


@pytest.mark.parity(
    "screened_poisson",
    "pymeshlab",
    benchmarked=False,
    reason="pymeshlab wraps the same Kazhdan CPU solver open3d does, and the two rows together "
    "were 6 322 s -- 73 % of the whole benchmark suite -- for a reference ordito already beats "
    "15-25x. Removed from benchmarks/test_reconstruction.py rather than capped; the comparison "
    "lives here instead.",
)
@pytest.mark.slow_cpu(72.4)
def test_poisson_matches_pymeshlab_metric(device: str):
    """
    Class C: the [`test_poisson_matches_open3d_metric`][] comparison against the other reference.

    Measured agreement is **0.0041** against the same 0.015 threshold, a **3.7x** margin, and the
    same mutation probe reads 0.0189 here and fails; the noise-floor warning on that test applies
    unchanged. **Pinned to depth 6 on both devices** for the reason given there: at depth 5 the
    agreement widens to **0.0062**, a 2.44x margin, below the 3x floor. Worth having
    both: open3d and pymeshlab agree with each other to 0.0015, a quarter of either one's distance
    to ordito, so they are not independent enough for one to stand in for the other -- but that
    also means a ordito regression would have to move past *both* to stay unnoticed.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_od = warp_to_trimesh(vertices_wp, faces_wp)

    # An oriented point cloud, so this is one of the few places a MeshSet is built from vertices
    # alone rather than through ``conversions.trimesh_to_pymeshlab``.
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            vertex_matrix=np.ascontiguousarray(points_np),
            v_normals_matrix=np.ascontiguousarray(normals_np),
        )
    )
    # depth=5, not 6, for the reason given on ``test_poisson_matches_open3d_metric``: MeshLab
    # returns the same 7 976 faces at both depths on this cloud. Do not go below 5 without
    # re-measuring -- the axis is *not* monotonic in either time or resolution, and depth=4 drops
    # to 2 024 faces.
    #
    # ``threads=1`` is load-bearing, and it is the single largest cost in the whole test suite.
    # ``print_filter_parameter_list`` reports ``threads : int = 48`` on this box -- the default is
    # ``hardware_concurrency`` -- and on a 642-point cloud the solver is far below its parallel
    # break-even, so the wall clock grows monotonically with the thread count while the answer does
    # not move (7 976 faces at every setting) -- roughly fiftyfold from one thread to the 48
    # default. Unpinned, this test also doubled between two runs twenty minutes apart, against a
    # ordito solve four orders of magnitude cheaper, so its cost was a function of how busy the
    # machine was rather than of anything either implementation does. One thread's CPU time bounds
    # the worst case; a global ``OMP_NUM_THREADS`` cap does *not* help, because MeshLab's filter
    # sets its own thread count from this parameter and overrides the environment (measured).
    meshset_pml.generate_surface_reconstruction_screened_poisson(depth=5, threads=1)
    mesh_current = meshset_pml.current_mesh()
    mesh_pml = tm.Trimesh(
        vertices=mesh_current.vertex_matrix(), faces=mesh_current.face_matrix(), process=False
    )
    mean_distance, _ = symmetric_surface_distance(mesh_od, mesh_pml)
    assert mean_distance < 0.015


@pytest.mark.parity(
    "screened_poisson",
    "open3d",
    benchmarked=False,
    reason="open3d wraps Kazhdan's CPU solver, whose screened_poisson rows were removed from the "
    "benchmarks for dominating the suite's wall clock (see test_poisson_matches_open3d_metric); "
    "the octree backend's comparison lives here, at a size a correctness test can afford.",
)
@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
def test_poisson_octree_matches_open3d(
    request: pytest.FixtureRequest, device: str, mesh_name: str
) -> None:
    """
    Class C: the fraction of open3d's vertices lying on ordito's ``method="octree"`` surface.

    The backend solves open3d's own system with its own non-converged schedule, so its surface is
    open3d's; only the triangulation differs (marching cubes over each depth's leaf cells against
    PoissonRecon's per-face polygons), so the vertices themselves cannot be matched one to one but
    every open3d vertex must sit on ordito's surface. Excludes the bug classes "wrong system" (a
    screening weight, a constraint, a density or a splat depth off), "wrong solver schedule" and
    "wrong unknowns" (the active nodes above ``full_depth``): each moves the field, and the
    non-converged far field moves with it.

    At depth 6 (``full_depth`` 5, so the finest depth is sparse) 99.74 % (``sphere_irregular``)
    and 99.41 % (``saddle_graded``, open) of open3d's vertices are within a thousandth of a cell
    of ordito's surface; the rest are at changes of depth, where open3d stitches a coarse leaf to
    its finer neighbours' vertices. The threshold, 98 %, allows 3.4x the measured misses. Mutation
    probes, each failing the assert: 9 sweeps instead of 8 (28 % / 33 %), ``point_weight`` 2.2
    instead of 2 (10 % / 31 %), and the splats past a high face dropped instead of aliased to the
    midplane nodes (29 % / 80 %).
    """
    mesh_tm, _ = request.getfixturevalue(mesh_name)
    points_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    normals_np = np.asarray(mesh_tm.vertex_normals, dtype=np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, method="octree"
    )
    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        points_to_open3d(points_np, normals_np), depth=6, n_threads=1
    )
    vertices_o3d = np.asarray(mesh_o3d.vertices)
    assert vertices_o3d.shape[0] > 1000
    mesh_od = warp_to_trimesh(vertices_wp, faces_wp)
    cell = 1.1 * float(np.ptp(points_np, axis=0).max()) / 2**6
    distance = np.abs(tm.proximity.signed_distance(mesh_od, vertices_o3d)) / cell
    assert np.mean(distance < 1e-3) > 0.98


@pytest.mark.parametrize(
    ("n_points", "arguments", "match"),
    [
        pytest.param(162, {"depth": 6, "full_depth": 8}, "full_depth", id="full_depth_above_depth"),
        pytest.param(162, {"depth": 11}, "full_depth", id="depth_too_large"),
        pytest.param(162, {"scale": 0.0}, "scale", id="zero_scale"),
        pytest.param(162, {"method": "bogus"}, "method", id="off_menu_method"),
        pytest.param(162, {"depth": 10, "method": "octree"}, "depth", id="octree_depth_10"),
        pytest.param(
            162, {"confidence": True, "method": "octree"}, "confidence", id="octree_confidence"
        ),
        pytest.param(2, {"depth": 4, "full_depth": 3}, "at least 3 points", id="two_points"),
    ],
)
def test_poisson_rejects_invalid_arguments(
    device: str, n_points: int, arguments: dict[str, object], match: str
):
    """Each invalid parameter, and a cloud too small to reconstruct, raises ``ValueError``."""
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np[:n_points], normals_np[:n_points], device)
    with pytest.raises(ValueError, match=match):
        od.reconstruction.screened_poisson(points_wp, normals_wp, **arguments)  # pyright: ignore[reportArgumentType]


def test_poisson_rejects_one_normal_per_point_mismatch(device: str) -> None:
    """
    Not a library comparison: a ``normals`` array not one per point raises ``ValueError``.

    Every splat and sampling kernel reads ``normals[i]`` for each point, so a shorter array would
    read past its end; the boundary values one short and one long of ``n`` both raise.
    """
    points_np, normals_np = _sphere_cloud(2)
    for n_normals in (10, 161, 163):
        points_wp, normals_wp = _to_warp(points_np, np.resize(normals_np, (n_normals, 3)), device)
        assert points_wp.size == 162
        with pytest.raises(ValueError, match="one normal per point"):
            od.reconstruction.screened_poisson(points_wp, normals_wp, depth=5, full_depth=4)


def test_poisson_cpu_matches_cuda():
    """Class A: the CPU cascadic solve reconstructs the CUDA surface, vertex for vertex."""
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    points_np, normals_np = _sphere_cloud(2)

    surfaces = {}
    for device in ("cpu", "cuda:0"):
        points_wp, normals_wp = _to_warp(points_np, normals_np, device)
        vertices_wp, faces_wp = od.reconstruction.screened_poisson(
            points_wp, normals_wp, depth=4, full_depth=3
        )
        surfaces[device] = (vertices_wp.numpy(), faces_wp.numpy())

    assert surfaces["cpu"][1].size > 0
    assert np.array_equal(surfaces["cpu"][1], surfaces["cuda:0"][1])
    assert np.allclose(surfaces["cpu"][0], surfaces["cuda:0"][0], rtol=1e-5, atol=1e-5)


# ======================================================================================
# Screened-Poisson, warp.fem adaptive backend (method="adaptive")
#
# The adaptive Nanogrid + variational assembly reconstructs a different vertex set than the dense
# backend, so comparisons stay metric/topological (dense-vs-fem cross-check is discretization
# agreement, never equality). CUDA-only, like the dense backend.
# ======================================================================================


_POISSON_WATERTIGHT_CASES = [
    pytest.param("sphere", "dense", marks=pytest.mark.slow_cpu(71.9), id="sphere-dense"),
    pytest.param("sphere", "adaptive", id="sphere-adaptive"),
    pytest.param("torus_irregular", "dense", id="torus-dense"),
    pytest.param("torus_irregular", "adaptive", id="torus-adaptive"),
]


@pytest.mark.parametrize(("shape", "method"), _POISSON_WATERTIGHT_CASES)
def test_poisson_watertight_manifold(
    device: str, shape: str, method: Literal["dense", "adaptive"]
) -> None:
    """
    Not a library comparison: reconstructing a closed surface from a clean, oriented cloud closes.

    ``sphere`` asserts the reconstructed radius hugs the unit sphere in addition to the topology;
    ``torus`` only checks watertightness and genus, since it has no analogous single-number shape
    to pin. ``adaptive`` additionally asserts outward orientation directly (the dense backend's is
    asserted by [`test_poisson_dense_solve_converges_in_few_iterations`]).

    **The sphere/dense cell is pinned to depth 6 on both devices**, unlike every other cell here,
    which uses [`_poisson_depth`] and so drops to 5 on CPU. Measured on this 642-point cloud,
    one reconstruction per process on **both** devices: depth 5 gives 7 976 faces that are
    edge-manifold with zero boundary edges but **self-intersecting**, so ``is_watertight`` --
    which follows Open3D and includes that clause -- is ``False``. Depth 6 closes it, and the
    sphere/dense radius tolerances are calibrated to a depth-6 cell (~0.034) as well. Not a device
    difference: CPU and CUDA agree at every depth, and the depth-5 answer is equally
    non-watertight on both -- the pin, and the ~68 s it costs on CPU, is unique to this one cell.
    """
    if shape == "sphere":
        # adaptive resolves a finer cloud than dense; each was tuned against its own resolution.
        points_np, normals_np = _sphere_cloud(3 if method == "dense" else 4)
        depth = 6 if method == "dense" else _poisson_depth(device)
    else:
        points_np, normals_np = _torus_cloud()
        depth = _poisson_depth(device)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=depth, full_depth=4, method=method
    )
    mesh_od = warp_to_trimesh(vertices_wp, faces_wp)

    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert mesh_od.euler_number == (2 if shape == "sphere" else 0)

    if shape != "sphere":
        return
    if method == "adaptive":
        assert mesh_od.volume > 0.0  # outward orientation
    radius_od = np.linalg.norm(mesh_od.vertices, axis=1)
    mean_tol, max_tol = (0.02, 0.06) if method == "dense" else (0.03, 0.08)
    assert abs(radius_od.mean() - 1.0) < mean_tol
    assert np.abs(radius_od - 1.0).max() < max_tol


def test_poisson_small_scale_cloud_is_watertight_without_degenerate_faces(device: str) -> None:
    """
    Not a library comparison: a cloud at millimetre scale reconstructs closed and sliver-free.

    ``face_nondegenerate_mask`` rejects an altitude under an *absolute* ``1e-8``, so how many
    marching-cubes slivers fall under it grows as the lattice spacing shrinks -- at depth 9 on the
    scan meshes, dozens a call. Without ``screened_poisson``'s edge margin they were dropped by its
    degenerate-face cleanup and each opened the surface, since their neighbours do not share their
    vertices: this cloud, the 642-point sphere scaled to radius ``1e-3``, came back with 16-18
    boundary edges at depth 5 on both devices, the same thing the 0.2-m scans do at depth 9.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(1e-3 * points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=5, full_depth=4
    )

    assert np.all(od.triangles.face_nondegenerate_mask(vertices_wp, faces_wp).numpy())
    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert warp_to_trimesh(vertices_wp, faces_wp).euler_number == 2


def _surface_cloud(mesh_tm: tm.Trimesh, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixture's surface uniformly (seeded) with its face normals, as an oriented cloud."""
    sample = tm.sample.sample_surface(mesh_tm, n_points, seed=np.random.default_rng(11))
    return np.asarray(sample[0], np.float64), np.asarray(mesh_tm.face_normals[sample[1]])


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_poisson_adaptive_matches_dense(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Ordito against ordito: the narrow-band backend reconstructs the dense backend's surface.

    The band backend solves the dense lattice's own system, stored only near the surface, with the
    lattice away from it held at the coarser levels' prolonged solution, tau-corrected by an FAS
    cycle so those levels see the finest one's solution; the dense backend carries the open3d and
    pymeshlab comparisons above. Measured in finest-cell units on a seeded surface sample of the
    closed fixture, the two sit a few thousandths of a cell apart, against a cell of discretization
    error in either; the bound is in cells so it means the same at either device's depth.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = str(mesh_wp.points.device)
    points_np, normals_np = _surface_cloud(mesh_tm, 20_000)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    depth = _poisson_depth(device) + 1
    cell = 1.1 * float(np.ptp(points_np, axis=0).max()) / 2**depth

    vertices_dense, faces_dense = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=depth, full_depth=4, method="dense"
    )
    vertices_band, faces_band = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=depth, full_depth=4, method="adaptive"
    )

    assert od.validation.is_watertight(vertices_band, faces_band)
    mean_distance, max_distance = symmetric_surface_distance(
        warp_to_trimesh(vertices_dense, faces_dense), warp_to_trimesh(vertices_band, faces_band)
    )
    # Measured 0.0021-0.0026 cells mean and 0.05-0.07 cells worst on both devices; the bounds are
    # 3x and 3.5x that. The cascade without its FAS cycle (16 finest rounds: the coarse levels'
    # rediscretized operators fixing the ghosts) puts the mean at
    # 0.010-0.011 cells, and not solving the band levels at all (``_BAND_ROUNDS = 0``) at 0.09-0.11
    # on CUDA, so the mean bound bites on both.
    assert mean_distance < 0.008 * cell
    assert max_distance < 0.25 * cell


def test_poisson_band_fas_cycle_tracks_the_dense_solution(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the band backend's FAS cycle puts its surface on the dense backend's.

    With every brick in every band there are no ghosts, so the FAS cycle (inject the fine solution,
    restrict the composite residual, solve the tau-corrected coarse level, prolong its change,
    re-solve) is a multigrid cycle for the finest lattice's own system -- the one the dense backend
    solves -- and each such cycle divides the field's error by about 15. Without the tau
    correction the coarse levels solve their rediscretized operators instead, and the bands' fixed
    handful of rounds leaves the surface where that put it. The dense backend carries the library
    comparisons.
    """
    points_np, _normals_np = _sphere_cloud(4)
    radius_np = 1.0 + 0.15 * np.sin(5.0 * points_np[:, 0]) * np.cos(4.0 * points_np[:, 1])
    mesh_tm = tm.Trimesh(points_np * radius_np[:, None], tm.creation.icosphere(4).faces)
    points_wp, normals_wp = _to_warp(mesh_tm.vertices, mesh_tm.vertex_normals, device)
    depth = _poisson_depth(device) + 1
    cell = 1.1 * float(np.ptp(mesh_tm.vertices, axis=0).max()) / 2**depth
    vertices_dense, faces_dense = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=depth, full_depth=4, solver_tolerance=1e-6
    )
    dense_tm = warp_to_trimesh(vertices_dense, faces_dense)

    band_hierarchy_flags = od.reconstruction._band_hierarchy_flags  # pyright: ignore[reportPrivateUsage]

    def whole_lattice(
        flags: wp.array[wp.int32], finest: int, base: int, flags_device: wp.DeviceLike
    ) -> dict[int, tuple[wp.array[wp.int32], int]]:
        flags.fill_(1)
        return band_hierarchy_flags(flags, finest, base, flags_device)

    monkeypatch.setattr(od.reconstruction, "_band_hierarchy_flags", whole_lattice)
    vertices_band, faces_band = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=depth, full_depth=4, method="adaptive"
    )
    distance = symmetric_surface_distance(dense_tm, warp_to_trimesh(vertices_band, faces_band))[0]
    # Measured 0.0017 cells mean on CUDA (depth 7) and 0.00066 on the CPU (depth 6); the cascade
    # without the FAS cycle measured 0.021 and 0.008, so the bound, 3x the CUDA value, bites on
    # both.
    assert distance < 5e-3 * cell


def test_poisson_band_nests_and_extracts_closed(device: str) -> None:
    """
    Not a library comparison: the band hierarchy loses nothing, and its extraction is closed.

    ``_screened_poisson_band`` counts, on the device, every coarse node a restriction or a
    Dirichlet ghost found missing from the coarser band, and every crossing cell its extraction
    band could not own, and warns when either is nonzero; on a sphere cloud both are impossible by
    the bands' nesting, so it must not warn. The raw extraction, before ``screened_poisson``'s
    cleanup, must already be watertight and free of degenerate faces -- the property the band's
    composite field and edge margin are for.
    """
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    lower, _upper, size = od.reconstruction._poisson_cube(points_wp, 1.1)  # pyright: ignore[reportPrivateUsage]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vertices_wp, faces_wp = od.reconstruction._screened_poisson_band(  # pyright: ignore[reportPrivateUsage]
            points_wp,
            normals_wp,
            lower,
            size,
            depth=_poisson_depth(device) + 1,
            full_depth=4,
            screen=4.0,
            solver_iterations=100,
            solver_tolerance=1e-5,
            confidence=False,
        )
    assert faces_wp.size > 0
    assert np.all(od.triangles.face_nondegenerate_mask(vertices_wp, faces_wp).numpy())
    assert od.validation.is_watertight(vertices_wp, faces_wp)


def test_poisson_adaptive_confidence_runs(device: str):
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp,
        normals_wp,
        depth=_poisson_depth(device),
        full_depth=4,
        confidence=True,
        method="adaptive",
    )
    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert warp_to_trimesh(vertices_wp, faces_wp).euler_number == 2


@pytest.mark.parametrize(
    ("method", "confidence"),
    [("dense", False), ("dense", True), ("adaptive", False), ("adaptive", True), ("octree", False)],
)
def test_poisson_is_reproducible(
    device: str, method: Literal["dense", "adaptive", "octree"], confidence: bool
) -> None:
    """
    Ordito against ordito: the identical call, four times, must return identical buffers.

    No reference library can carry this -- it is a claim about this implementation's summation
    order; the surfaces themselves are compared with open3d and pymeshlab above. The splat of the
    normals onto the lattice and the adaptive backend's restriction down its levels were float
    atomics, whose CUDA commit order moved the right-hand side's last bits and through the solve
    the output vertices, so these four calls failed it on CUDA for both methods and both
    weightings. With the deterministic setting on (``tests/conftest.py``) the splat sums each
    node's terms in one fixed order and the restriction gathers in ascending fine-node order, so
    the outputs repeat byte for byte. The CPU device repeats either way. A jittered cloud with
    normals of varying length, so ``confidence=True`` weights unequally. The octree backend's
    density and normal splats are group sums under the same setting (it takes no ``confidence``).
    """
    points_np, normals_np = _sphere_cloud(4)
    rng = np.random.default_rng(0)
    points_np = points_np + rng.normal(scale=0.01, size=points_np.shape)
    normals_np = normals_np * rng.uniform(0.5, 1.5, (points_np.shape[0], 1))
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    runs = [
        od.reconstruction.screened_poisson(
            points_wp,
            normals_wp,
            depth=_poisson_depth(device),
            full_depth=4,
            confidence=confidence,
            method=method,
        )
        for _ in range(4)
    ]
    (vertices_wp, faces_wp), *repeats = runs
    vertices_np, faces_np = vertices_wp.numpy(), faces_wp.numpy()
    assert faces_np.size > 0
    for vertices_wp, faces_wp in repeats:
        assert np.array_equal(vertices_wp.numpy(), vertices_np)
        assert np.array_equal(faces_wp.numpy(), faces_np)


# ---------------------------------------------------------------------------
# Marching cubes and uniform resampling (analytic + pymeshlab / skimage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.2, -0.2])
def test_resample_uniform_offsets_a_sphere(
    offset: float, sphere_round: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class C (a radius bound): a sphere is the one shape whose offset surface is known exactly.

    Both signs are covered because they are different code paths in spirit -- a positive offset
    needs the lattice padded beyond the bounding box (or it clips) and a negative one does not.

    The input is a polyhedron inscribed in a sphere of radius ``R = 1.7`` (``sphere_round``), so its
    own surface sits between ``R - sag`` and ``R``, ``sag`` its deepest face-plane distance below
    the sphere (0.028 here); the offset shifts that band without widening it. Measured: every
    output vertex inside ``[R + offset - 0.0284, R + offset]`` at both offsets. The bound
    allows a fifth of a voxel either side of that band; a 5 % scale error (0.085) fails it.
    """
    sphere_tm, sphere_wp = sphere_round
    vertices_wp, faces_wp = sphere_wp.points, sphere_wp.indices
    voxel_size = 0.05
    radius = ROUND_FRAME_SCALE
    corners_np = np.asarray(sphere_tm.triangles, dtype=np.float64) - ROUND_FRAME_SHIFT
    normals_np = np.cross(corners_np[:, 1] - corners_np[:, 0], corners_np[:, 2] - corners_np[:, 0])
    normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
    sag = radius - float(np.abs(np.einsum("ij,ij->i", normals_np, corners_np[:, 0])).min())

    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size, offset=offset
    )
    radii_np = np.linalg.norm(out_vertices_wp.numpy() - ROUND_FRAME_SHIFT, axis=1)
    assert radii_np.min() > radius + offset - sag - 0.2 * voxel_size
    assert radii_np.max() < radius + offset + 0.2 * voxel_size

    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)
    assert out_tm.is_watertight
    assert out_tm.volume > 0.0


def test_resample_uniform_repairs_a_broken_mesh(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the topology comes from the grid, so the input's cannot leak.

    The input here has duplicated faces, an inverted one and a non-manifold edge — three defects
    that each need their own function in [`ordito.repair`][ordito.repair] — and the resampled
    result is a clean watertight sphere regardless, enclosing the input's volume (measured 0.09 %
    off it at this voxel).
    """
    sphere_tm, _sphere_tm_wp = sphere_irregular
    faces_np = np.asarray(sphere_tm.faces)
    broken_np = np.vstack([faces_np, faces_np[:20], faces_np[30:40][:, ::-1]])
    vertices_wp = points_to_warp(sphere_tm.vertices, device)
    faces_wp = wp.array(
        np.ascontiguousarray(broken_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    assert not od.validation.is_edge_manifold(faces_wp)

    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=0.06
    )
    assert od.validation.is_edge_manifold(out_faces_wp, allow_boundary_edges=False)
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), sphere_tm.volume, rtol=0.01)


@pytest.mark.parity("resample_uniform", "pymeshlab")
def test_resample_uniform_matches_pymeshlab(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (a surface distance): the same algorithm at the same absolute cell size.

    One parameter hazard, found by probing: its ``offset`` as a ``PercentageValue`` runs from full
    erosion at ``0%`` to full dilation at ``100%``, so **``PercentageValue(50)`` -- its default --
    is the *zero* offset** and ``PercentageValue(0)`` erodes a unit sphere down to radius 0.30.
    Passing ``PureValue(0.0)`` instead means an absolute offset of zero, which is what
    ``offset=0.0`` is here.

    **The fixture sets the bars.** Every resampler rounds ``sphere_irregular``'s cone vertices
    (angle defects up to 130 degrees) by most of a voxel, so the *worst* sample of each sits far
    from the input -- at 0.06: ordito 0.48 voxels, MeshLib 0.55, igl 0.73, pymeshlab 1.69 -- and
    two resamplers anchored a fraction of a cell apart differ there by as much. (On ``icosphere(3)``
    everything agreed to 0.06 voxels; that was the smooth input, not a tighter agreement.) What a
    lattice anchoring, isolevel or sign error moves is the *mean* distance, so that is the claim:
    ordito against MeshLab **0.017** voxels (bar 0.05), ordito against the input 0.021 (MeshLab
    0.021; bar 0.065). Volumes agree to 1.0e-4 relative (bar 3e-4). Identical on both devices.

    **Mutation probes.** Asking ordito for ``offset`` of a quarter voxel moves its mean distance to
    the input to 0.247 voxels; resampling a 1.05x scaled input to 1.26. The worst case is still
    bounded, at a voxel (measured 0.48), against a local defect.
    """
    sphere_tm, sphere_wp = sphere_irregular
    voxel_size = 0.06
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(sphere_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.generate_resampled_uniform_mesh(
        cellsize=ml.PureValue(voxel_size), offset=ml.PureValue(0.0)
    )
    mesh_pml = meshset_pml.current_mesh()
    pml_tm = tm.Trimesh(mesh_pml.vertex_matrix(), mesh_pml.face_matrix(), process=False)

    vertices_wp, faces_wp = sphere_wp.points, sphere_wp.indices
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)
    assert out_tm.is_watertight
    assert pml_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(pml_tm.volume), rtol=3e-4)
    assert symmetric_surface_distance(out_tm, pml_tm)[0] < 0.05 * voxel_size
    mean_to_input, max_to_input = symmetric_surface_distance(out_tm, sphere_tm)
    assert mean_to_input < 0.065 * voxel_size
    assert max_to_input < voxel_size


@pytest.mark.parity("resample_uniform", "igl")
def test_resample_uniform_matches_igl(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (no vertex correspondence): the two isosurfaces coincide to **0.014 of a voxel** (mean).

    ``igl.offset_surface(V, F, isolevel, s, sign_type)`` samples the same signed distance field on a
    grid and marches it. Two named parameter transforms put the two on one lattice: ``isolevel=0``
    is ordito's zero offset, and ``s`` is a *cell count along the longest axis* rather than a
    length, so it gets ``round(longest_extent / voxel_size)``. The sign mode is ``PSEUDONORMAL``,
    which ``tests/test_proximity.py::test_signed_distance_on_mesh_matches_pymeshlab_igl_and_open3d``
    establishes agrees with ordito's default -- the winding modes would scale the field by
    ``1 - 2w`` and move the isosurface.

    **The fixture sets the bars.** Every resampler rounds ``sphere_irregular``'s cone vertices
    (angle defects up to 130 degrees) by most of a voxel, so the *worst* sample of each sits far
    from the input -- at 0.06: ordito 0.48 voxels, MeshLib 0.55, igl 0.73, pymeshlab 1.69 -- and
    two resamplers anchored a fraction of a cell apart differ there by as much. (On ``icosphere(3)``
    everything agreed to 0.06 voxels; that was the smooth input, not a tighter agreement.) What a
    lattice anchoring, isolevel or sign error moves is the *mean* distance, so that is the claim:
    ordito against igl **0.014** voxels (bar 0.045), ordito against the input 0.021 (igl 0.023;
    bar 0.065); volumes within 1.15e-4 relative (bar 3.5e-4). Identical on both devices.

    **Bug class excluded:** a grid anchored differently, or an isolevel or sign convention that
    shifts the surface. **Mutation probe, measured:** re-running igl at ``isolevel`` a quarter
    voxel moves the mean distance to **0.253 voxels**, 18x the agreement and 5.6x the bar.
    """
    sphere_tm, sphere_wp = sphere_irregular
    voxel_size = 0.06
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(sphere_tm.faces, dtype=np.int64)

    extent = float((vertices_np.max(axis=0) - vertices_np.min(axis=0)).max())
    vertices_igl, faces_igl = igl.offset_surface(
        vertices_np,
        faces_np,
        0.0,
        max(2, round(extent / voxel_size)),
        igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
    )[:2]
    mesh_igl = tm.Trimesh(vertices_igl, faces_igl, process=False)

    vertices_wp, faces_wp = sphere_wp.points, sphere_wp.indices
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)

    assert mesh_igl.is_watertight
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(mesh_igl.volume), rtol=3.5e-4)
    assert symmetric_surface_distance(out_tm, mesh_igl)[0] < 0.045 * voxel_size
    assert symmetric_surface_distance(out_tm, sphere_tm)[0] < 0.065 * voxel_size


@pytest.mark.parity("resample_uniform", "meshlib")
def test_resample_uniform_matches_meshlib(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (no correspondence, and a different amount of work): both land on the same surface.

    ``rebuildMesh`` samples the same signed distance field on a grid of the same ``voxelSize`` and
    marches it, then -- unlike ordito, igl and MeshLab -- **decimates** the result at its own
    defaults (``decimate`` and ``preSubdivide`` are both on). So the face counts are not comparable
    at all. What is comparable is the surface.

    **The fixture sets the bars.** Every resampler rounds ``sphere_irregular``'s cone vertices
    (angle defects up to 130 degrees) by most of a voxel, so the *worst* sample of each sits far
    from the input -- at 0.06: ordito 0.48 voxels, MeshLib 0.55, igl 0.73, pymeshlab 1.69 -- and
    two resamplers anchored a fraction of a cell apart differ there by as much. (On ``icosphere(3)``
    everything agreed to 0.06 voxels; that was the smooth input, not a tighter agreement.) What a
    lattice anchoring, isolevel or sign error moves is the *mean* distance, so that is the claim:
    at a 2 % voxel, ordito against the rebuilt mesh **0.032** voxels, and each against the input
    0.038 (ordito) and 0.043 (rebuilt), all under a 0.1-voxel bar (about 3x the worst already);
    volumes within 9.1e-4 relative (bar 3e-3). Identical on both devices.

    **Bug class excluded:** a grid anchored differently, or a sign or isolevel convention that moves
    the isosurface -- checked here against an independent implementation of the whole pipeline
    rather than of the field alone. **Mutation probe, measured:** running MeshLib at 4x the voxel
    size moves the ordito-to-MeshLib mean to **0.39 voxels**; asking ordito for an offset of a
    quarter voxel moves its mean to the input to 0.24.
    """
    sphere_tm, sphere_wp = sphere_irregular
    diagonal = float(
        np.linalg.norm(sphere_tm.vertices.max(axis=0) - sphere_tm.vertices.min(axis=0))
    )
    voxel_size = 0.02 * diagonal

    vertices_wp, faces_wp = sphere_wp.points, sphere_wp.indices
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)

    settings_ml = mm.RebuildMeshSettings()
    settings_ml.voxelSize = voxel_size
    mesh_ml = numpy_to_meshlib(sphere_tm.vertices, sphere_tm.faces)
    rebuilt_tm = meshlib_to_trimesh(mm.rebuildMesh(mm.MeshPart(mesh_ml), settings_ml))

    assert rebuilt_tm.faces.shape[0] > 0  # non-vacuity: the reference rebuilt something
    assert rebuilt_tm.is_watertight
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(rebuilt_tm.volume), rtol=3e-3)
    assert symmetric_surface_distance(out_tm, rebuilt_tm)[0] < 0.1 * voxel_size
    # Both are a resampling *of the input*, not merely of each other.
    for resampled_tm in (out_tm, rebuilt_tm):
        assert symmetric_surface_distance(resampled_tm, sphere_tm)[0] < 0.1 * voxel_size


def test_resample_uniform_coarser_is_smaller(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: a wider voxel gives fewer triangles, and still a closed surface."""
    _sphere_tm, sphere_wp = sphere_irregular
    vertices_wp, faces_wp = sphere_wp.points, sphere_wp.indices
    counts = []
    for voxel_size in (0.05, 0.1, 0.2):
        _out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
            vertices_wp, faces_wp, voxel_size=voxel_size
        )
        counts.append(out_faces_wp.size // 3)
        assert od.validation.is_edge_manifold(out_faces_wp, allow_boundary_edges=False)
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize(("primitive", "voxel_size"), [("box", 0.05), ("icosphere", None)])
def test_resample_uniform_is_watertight_when_nodes_land_on_the_surface(
    device: str, primitive: str, voxel_size: float | None
) -> None:
    """
    Not a library comparison: the docstring's "always closed", on lattices that test it.

    A unit box on a ``0.05`` grid puts whole planes of lattice nodes exactly on its faces, where
    the distance is zero. Without an edge margin the extracted vertices land on those nodes, the
    slivers around them are deleted as degenerate, and the surface opens (non-manifold on CUDA,
    five components on CPU). The icosphere at the default voxel size is the case that read as
    self-intersecting on CUDA alone.
    """
    if primitive == "box":
        vertices_wp, faces_wp = od.creation.box(device=device)
    else:
        vertices_wp, faces_wp = od.creation.icosphere(subdivisions=2, device=device)
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    assert od.validation.is_watertight(out_vertices_wp, out_faces_wp)
    labels_np = od.adjacency.face_connected_component_labels(out_faces_wp).numpy()
    assert np.unique(labels_np).shape[0] == 1
    assert not (~od.triangles.face_nondegenerate_mask(out_vertices_wp, out_faces_wp).numpy()).any()


def test_resample_uniform_invalid_and_empty(device: str) -> None:
    sphere_tm = tm.creation.icosphere(subdivisions=1, radius=1.0)
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    with pytest.raises(ValueError, match="voxel_size > 0"):
        od.reconstruction.resample_uniform(vertices_wp, faces_wp, voxel_size=0.0)
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(vertices_wp, faces_wp)
    assert out_vertices_wp.size == 0
    assert out_faces_wp.size == 0


# ======================================================================================
# Ball pivoting (ball_pivoting)
#
# The wave-parallel front is interpolating (output vertices are input points) and edge-manifold
# after cleanup, but is not guaranteed watertight on densely sampled closed surfaces (v1
# limitation), so the tests assert those robust invariants rather than watertightness / Euler.
# open3d BPA (``_o3d``) is a loose face-count reference on the icosphere and the wound-triangle-set
# oracle on a Fibonacci sphere, where both pivot the identical triangulation.
# ======================================================================================


@pytest.mark.parity("ball_pivoting", "open3d")
@pytest.mark.parametrize("normal_sign", [1.0, -1.0], ids=["outward", "inward"])
def test_ball_pivoting_winds_faces_with_the_normals(device: str, normal_sign: float):
    """
    Class A against open3d: the same wound triangle set, every face agreeing with the normals.

    Open3D winds each triangle so its normal agrees with the point normals (``CreateTriangle``), and
    on this Fibonacci sphere both implementations pivot the identical triangulation, so the
    comparison is the whole wound face set, not only the unoriented one. Parametrized over outward
    and inward normals: the winding must follow the normals, not be re-derived from the volume.

    Without the fix the seed triangle took whatever corner order its neighbourhood scan produced and
    every pivot repeated its front edge in the parent's direction, so neighbouring faces alternated
    winding (half the raw faces opposed the normals). On this fixture the result is not watertight,
    so the cleanup's volume flip did not run and the output kept whichever winding its
    per-component seed face had (volume -4.18 with outward normals, one face of 7 992 agreeing with
    them). Both arms fail the per-face agreement assert without the fix.
    """
    from scipy.spatial import cKDTree

    points_wp = od.sample.sample_fibonacci_sphere(4000, device=device)
    points_np = points_wp.numpy().astype(np.float64)
    normals_np = normal_sign * points_np
    normals_wp = points_to_warp(normals_np, device)
    radius = 0.08

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    distance, input_index = cKDTree(points_np).query(vertices_wp.numpy().astype(np.float64))
    assert np.max(distance) < 1e-6
    faces_np = np.asarray(input_index)[faces_wp.numpy().reshape(-1, 3)]

    face_normals_np = np.cross(
        points_np[faces_np[:, 1]] - points_np[faces_np[:, 0]],
        points_np[faces_np[:, 2]] - points_np[faces_np[:, 0]],
    )
    agreement = np.einsum("ij,ikj->ik", face_normals_np, normals_np[faces_np])
    assert np.all(agreement > 0.0)
    assert np.sign(od.measures.volume(vertices_wp, faces_wp)) == normal_sign

    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        points_to_open3d(points_np, normals_np), o3d.utility.DoubleVector([radius])
    )
    faces_o3d = np.asarray(mesh_o3d.triangles)
    assert faces_o3d.shape[0] > 0
    assert_unordered_rows_equal(canonical_winding(faces_np), canonical_winding(faces_o3d))


def test_ball_pivoting_is_reproducible(device: str):
    """
    Two runs of one build on one cloud must reconstruct the same triangles.

    A wave resolves competing proposals with ``wp.atomic_min`` over ``proposal_key`` — packed from
    the proposal's own source edge — rather than over the ``wp.atomic_add`` slot it was handed, so
    nothing about the answer depends on the order threads reach an atomic. A losing proposal is not
    merely reordered: its front edge is retried a wave later against mutated state, so the
    divergence used to compound into different geometry.

    **The fixture is the load-bearing choice.** Every icosphere is reproducible even *without* the
    fix — a uniformly sampled closed sphere reconstructs to its exact Euler triangulation, so wave
    order has nothing left to decide — and asserting on one would be vacuous. So, since co-spherical
    ties are broken by index, is the regular torus (its exact 4 096-face triangulation), and so is
    this torus at a jitter of 0.01; at 0.02, four runs with the claim keyed on the arrival slot (the
    pre-fix priority) give 3 791 / 3 792 / 3 790 / 3 792 faces on cuda, against a constant 3 790
    and one mesh with ``proposal_key``.

    Two assertions, covering different halves, both on the **wound** triangle set -- same
    triangles and same winding. Not compared buffer-to-buffer: ``CNT_FACE`` is a ``wp.atomic_add``,
    so the row order is deliberately still arrival-ordered:

    * the wave loop's own output, which is the property the key buys;
    * the public function, byte for byte: its cleanup tail emits the faces in ascending
      unoriented-key order (``repair.resolve_duplicated_faces``), so the returned buffers do not
      inherit the wave loop's row order, and it keeps the winding the wave loop wrote from the
      normals rather than re-deriving it -- at an explicit radius and at its own guess, which is a
      fixed-order sum with the deterministic setting on (an atomic one moved the guess a few ULP
      between calls, enough to tip a borderline pivot).

    The claim is the setting-on contract, which every test runs under (``tests/conftest.py``);
    with it off the guess folds by atomics and is not promised to repeat.
    """
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=64, minor_sections=32
    )
    # Jittered: the regular torus's cells are co-circular, and with those ties broken by index it
    # reconstructs to its exact triangulation, which leaves wave order nothing to decide.
    jitter_np = np.random.default_rng(0).uniform(-0.02, 0.02, torus_tm.vertices.shape)
    points_wp, normals_wp = _to_warp(torus_tm.vertices + jitter_np, torus_tm.vertex_normals, device)
    n_points = points_wp.size
    # The wrapper's own auto-radius, recomputed on the host, for the raw wave-loop runs, which take
    # a radius; the last pair of public calls guesses it on the device instead.
    radius = _bpa_radius(points_wp)

    grid = od.neighbors.hashgrid_from_points(points_wp, radius)
    bvh = od.neighbors.bvh_from_points(points_wp)
    raw_runs = []
    for _ in range(2):
        state = od.reconstruction._BpaState(  # pyright: ignore[reportPrivateUsage]
            points_wp, normals_wp, grid, bvh, radius, 0.2, -1.0, 4 * n_points + 16
        )
        od.reconstruction._bpa_run(state, 16 * n_points)  # pyright: ignore[reportPrivateUsage]
        n_faces = int(state.counters.numpy()[kernel_bpa.CNT_FACE])
        raw_runs.append(state.all_faces.numpy()[: n_faces * 3].reshape(-1, 3))

    assert raw_runs[0].shape[0] > 0
    assert_unordered_rows_equal(canonical_winding(raw_runs[0]), canonical_winding(raw_runs[1]))

    # ``radius=0`` is the default: the call guesses its own radius.
    for radius_arg in (radius, 0.0):
        first = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius_arg)
        second = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius_arg)
        assert first[1].size > 0
        for first_wp, second_wp in zip(first, second, strict=True):
            assert np.array_equal(first_wp.numpy(), second_wp.numpy())


def test_ball_pivoting_seeds_no_triangle_twice(device: str):
    """
    Two points seeding with each other as partner must not both commit, on a dense neighbourhood.

    Not a library comparison: the property is internal to the wave loop's claim, which no
    reference has. ``seed_triangles`` truncates its neighbourhood scan at ``MAX_SEED_NEIGHBORS``,
    so with a ball several spacings wide (about 200 points within the ``2 * radius`` scan here)
    two points can each seed with the other as partner. Keyed by the unordered pair, the two
    proposals tied, both won every vertex claim and both committed, often as the same triangle
    with the same winding, which ``repair.resolve_duplicated_faces`` then rejects as
    non-orientable (happy_buddha raised so). With the claim keyed by the ordered pair, seeder
    first, at most one of them commits.

    Asserted on the wave loop's own soup, before the cleanup's duplicate resolution could hide a
    pair (a reversed duplicate would cancel silently there), and on the public call, which raised.
    Every one of ten seeds of this cloud committed a duplicate under the unordered key.
    """
    n_points = 400
    rng = np.random.default_rng(0)
    points_np = np.column_stack([rng.uniform(0.0, 1.0, (n_points, 2)), np.zeros(n_points)])
    normals_np = np.tile([0.0, 0.0, 1.0], (n_points, 1))
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    radius = 4.0 / math.sqrt(n_points)

    grid = od.neighbors.hashgrid_from_points(points_wp, radius)
    bvh = od.neighbors.bvh_from_points(points_wp)
    state = od.reconstruction._BpaState(  # pyright: ignore[reportPrivateUsage]
        points_wp, normals_wp, grid, bvh, radius, 0.2, -1.0, 4 * n_points + 16
    )
    od.reconstruction._bpa_run(state, 16 * n_points)  # pyright: ignore[reportPrivateUsage]
    n_faces = int(state.counters.numpy()[kernel_bpa.CNT_FACE])
    sorted_np = np.sort(state.all_faces.numpy()[: n_faces * 3].reshape(-1, 3), axis=1)
    assert n_faces > 0
    assert np.unique(sorted_np, axis=0).shape[0] == n_faces

    _vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    assert faces_wp.size > 0


def test_ball_pivoting_grows_the_triangle_budget(device: str):
    """
    A budget far below what the mesh needs must grow, not raise or truncate.

    Growing rehashes the edge table (slot indices move) and rebuilds the front list from it, so
    this also covers that path. The result has to match a run that never had to grow.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    grid = od.neighbors.hashgrid_from_points(points_wp, 2.0 * 0.2)
    bvh = od.neighbors.bvh_from_points(points_wp)

    faces_per_budget = []
    for start_budget in (64, 4 * points_np.shape[0] + 16):
        state = od.reconstruction._BpaState(  # pyright: ignore[reportPrivateUsage]
            points_wp, normals_wp, grid, bvh, 0.2, 0.2, math.cos(math.pi / 2.0), start_budget
        )
        od.reconstruction._bpa_run(state, 16 * points_np.shape[0])  # pyright: ignore[reportPrivateUsage]
        counters_np = state.counters.numpy()
        assert counters_np[kernel_bpa.CNT_DONE] == 1
        faces_per_budget.append(int(counters_np[kernel_bpa.CNT_FACE]))
    grown, direct = faces_per_budget
    assert grown > 64  # it really did outgrow the initial allocation
    assert abs(grown - direct) <= 0.01 * direct


@pytest.mark.parametrize("max_waves", [1, 7, 8])
def test_bpa_front_swap_tracks_the_waves_that_ran(
    device: str, max_waves: int, monkeypatch: pytest.MonkeyPatch
):
    """
    Ordito against ordito: ``front_in`` must hold the buffer the last wave that *ran* wrote.

    ``_bpa_run`` queues ``_BPA_WAVES_PER_BATCH`` waves before it looks at the counters, and swaps
    ``front_in`` / ``front_out`` after each queued one -- but every wave kernel opens with
    ``if counters[CNT_CONTINUE] == 0: return``, so the waves after the device stops write nothing.
    An odd number of those used to leave the pair exchanged, and the next ``compact()`` then read
    the wrong buffer; in the first batch that buffer is the one ``wp.empty`` allocated and nothing
    ever wrote, so it was an unbounded out-of-bounds read of the edge table, not merely a stale
    front. Measured before the fix: on a 40 962-point cloud, four of five ``compact()`` calls over a
    seven-cloud sweep read the wrong buffer, and two of three such processes died with
    ``CUDA error 700`` at an unrelated readback later on.

    ``max_waves`` is the parametrization because it *is* the trigger and it is exact: ``end_wave``
    clears ``CNT_CONTINUE`` once ``CNT_WAVE >= max_waves``, so an odd value ends the batch with an
    odd number of no-op waves on any cloud at all, and an even value is a control that passes
    either way: 1 and 7 are the odd ends of the batch, 8 fills it. A fixture large enough to reach
    ``compact()`` on its own is ~25 000 points, far too slow for a unit test -- the parameter
    reaches the same code path on 162.

    Buffer *identity* is the assertion rather than any count: ``id(state.front_out)`` before a wave
    that runs is the buffer that wave writes, and ``grow()`` / ``compact()`` each re-establish the
    invariant themselves (one rebuilds ``front_in`` from the edge table, the other swaps after
    writing), so recording their result covers a run that takes either.

    The mutation probe bites on the final assert, and only for the odd values: reverting the parity
    correction fails every odd ``max_waves`` (1, 3 and 7 were probed) with *"front_in is not the
    buffer the last wave that ran wrote"* and leaves the even ones (2, 4 and 8) passing. The entry
    assert inside ``compact()`` is a second guard on the one place a violation is actually *read*
    rather than merely present; neither it nor the ``grow()`` arm fires at these wave caps, which is
    the point of keeping the caps small.
    """
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    n_points = points_wp.size
    radius = 0.2
    grid = od.neighbors.hashgrid_from_points(points_wp, radius)
    bvh = od.neighbors.bvh_from_points(points_wp)
    state = od.reconstruction._BpaState(  # pyright: ignore[reportPrivateUsage]
        points_wp, normals_wp, grid, bvh, radius, 0.2, -1.0, 4 * n_points + 16
    )

    # The buffer that should be live, appended to by whatever last established it.
    live: list[int] = []
    resyncs = 0
    unpatched_wave = od.reconstruction._bpa_wave  # pyright: ignore[reportPrivateUsage]
    unpatched = {name: getattr(od.reconstruction._BpaState, name) for name in ("compact", "grow")}  # pyright: ignore[reportPrivateUsage]

    def recording_wave(tracked: od.reconstruction._BpaState, waves: int, *, begin: bool) -> None:  # pyright: ignore[reportPrivateUsage]
        if int(tracked.counters.numpy()[kernel_bpa.CNT_CONTINUE]):
            live.append(id(tracked.front_out))
        unpatched_wave(tracked, waves, begin=begin)

    def checked(name: str):
        def wrapper(tracked: od.reconstruction._BpaState) -> None:  # pyright: ignore[reportPrivateUsage]
            nonlocal resyncs
            resyncs += 1
            if name == "compact":
                assert id(tracked.front_in) == live[-1], (
                    "compact() read a buffer no wave wrote: the host swapped for a wave that "
                    "did not run"
                )
            unpatched[name](tracked)
            live.append(id(tracked.front_in))

        return wrapper

    monkeypatch.setattr(od.reconstruction, "_bpa_wave", recording_wave)
    for name in unpatched:
        monkeypatch.setattr(od.reconstruction._BpaState, name, checked(name))  # pyright: ignore[reportPrivateUsage]
    od.reconstruction._bpa_run(state, max_waves)  # pyright: ignore[reportPrivateUsage]

    counters_np = state.counters.numpy()
    waves_run = int(counters_np[kernel_bpa.CNT_WAVE])
    # Non-vacuity: the run must have reached the parametrized cap, so the batch really did end with
    # `_BPA_WAVES_PER_BATCH - max_waves` no-op waves rather than converging before the cap.
    assert waves_run == max_waves
    assert len(live) == waves_run + resyncs
    assert id(state.front_in) == live[-1], "front_in is not the buffer the last wave that ran wrote"


def test_ball_pivoting_on_a_bvh_leaf_past_the_tile_walk_buffer(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: an overfull point-BVH leaf leaves the reconstruction unchanged.

    The patch is a jittered 24 x 24 grid of spacing 0.005 inside one Morton cell, followed by 31
    far points that pin the Morton normalisation and form a 30-step chain -- one point in each cell
    whose index has a single bit set, plus the far corner -- so on CUDA the LBVH builder stops at
    depth 32 with the whole patch (575 points) in one leaf. At radius 0.02 every pivot search box
    meets about 270 of them, past the 160-entry shared buffer of ``wp.tile_bvh_query_aabb``
    (``kernels.proximity.TILE_WALK_LEAF_MAX``): before the slab walk in ``pivot_front_edges`` this
    faulted with CUDA error 700. The far points sit after the patch and are never triangulated, so
    the patch alone -- whose tree has no such leaf and takes the cooperative walk -- is the oracle,
    on the vertices and on the wound triangles. The jitter breaks the regular grid's co-circular
    quadruples, whose tie-broken reconstruction is not what this test is about.

    On CUDA the overfull bitset is asserted non-empty for the chained cloud and empty for the patch,
    so the test cannot pass by both runs taking the builtin's walk. The cpu device builds a
    different tree (no overfull leaf here), walks serially and marks nothing, so there it only pins
    agreement; under ``--cpu-blocks`` it runs the bitset pass over that tree, which must flag
    nothing.
    """
    side, spacing = 24, 0.005
    positions = 0.4 + spacing * np.arange(side)
    grid_x, grid_y = np.meshgrid(positions, positions, indexing="ij")
    patch_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 0.25)], axis=1)
    patch_np[:, :2] += np.random.default_rng(0).uniform(-0.2, 0.2, (grid_x.size, 2)) * spacing
    chain_np = np.array(
        [np.eye(3)[axis] * 2**bit + 0.5 for bit in range(10) for axis in range(3)]
        + [np.full(3, 1023.5)]
    )

    flags: list[wp.array[wp.int32]] = []

    def capturing(real_launch: Callable[..., object]):
        def capturing_launch(*args: object, **kwargs: object):
            if kwargs.get("kernel", args[0] if args else None) is (
                kernel_bpa.mark_overfull_point_leaves
            ):
                flags.append(cast("list[wp.array[wp.int32]]", kwargs["inputs"])[1])
            return real_launch(*args, **kwargs)

        return capturing_launch

    monkeypatch.setattr(_launch, "launch", capturing(_launch.launch))
    monkeypatch.setattr(wp, "launch", capturing(wp.launch))

    results = []
    for points_np in (np.concatenate([patch_np, chain_np]), patch_np):
        normals_np = np.tile([0.0, 0.0, 1.0], (points_np.shape[0], 1))
        points_wp, normals_wp = _to_warp(points_np, normals_np, device)
        vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=0.02)
        results.append((vertices_wp.numpy(), faces_wp.numpy()))

    if wp.get_device(device).is_cuda:
        # ``_launch.launch`` may forward to the patched ``wp.launch``: one launch can record twice.
        flags = list({id(bitset): bitset for bitset in flags}.values())
        assert len(flags) == 2
        assert flags[0].numpy().any(), "the chained cloud's tree must hold an overfull leaf"
        assert not flags[1].numpy().any()
    elif wp.config.enable_cpu_blocks:
        flags = list({id(bitset): bitset for bitset in flags}.values())
        assert len(flags) == 2
        # The bitset pass runs wherever blocks have several lanes, but the cpu device's tree has
        # no overfull leaf here, so both bitsets stay empty.
        assert not any(bitset.numpy().any() for bitset in flags)
    else:
        assert not flags
    (chained_vertices_np, chained_faces_np), (patch_vertices_np, patch_faces_np) = results
    assert patch_faces_np.shape[0] > 3 * patch_np.shape[0]
    assert np.array_equal(chained_vertices_np, patch_vertices_np)
    assert_unordered_rows_equal(
        canonical_winding(chained_faces_np), canonical_winding(patch_faces_np)
    )


@pytest.mark.parametrize("frame", ["axis_aligned", "rotated_offset"])
@pytest.mark.parametrize("chained", [False, True], ids=["grid", "grid_and_chain"])
@pytest.mark.parametrize(("side", "spacing"), [(40, 0.0125), (4, 0.03)], ids=["40x40", "4x4"])
def test_ball_pivoting_triangulates_an_unjittered_grid(
    device: str,
    side: int,
    spacing: float,
    chained: bool,
    frame: Literal["axis_aligned", "rotated_offset"],
) -> None:
    """
    Not a library comparison: an unjittered grid must come out as its own exact triangulation.

    Every cell of a regular grid has four co-circular corners, so both of its triangulations have an
    empty ball and the rotation from one triangle's ball to the other's is zero: no reference is an
    oracle here, since each settles the tie by rounding (Open3D tests ``radius - 1e-16``). Before
    the tie rule (``kernel_bpa.band_point_inside``) and the rotation clamp (``pivot_candidate``),
    rounding settled it here too, differently per triangle and per device: pivots took the other
    diagonal's triangle on top of their own, wound against the normals, and on the axis-aligned
    grid the cleanup raised ``non-orientable duplicate face group`` (cpu, and cuda with the chain);
    elsewhere the result was not winding-consistent and missed 100-450 faces. Every 40 x 40 case
    failed before the fix. The 4 x 4 grid is where the clamp mattered: a cell's far corner rotated
    by a rounded-negative zero, folded onto the ``TWO_PI`` sentinel, and the cell was left half open
    (17 faces of 18 on cuda, in one build of the kernel). The sign of that zero follows how nvcc
    contracts the angle, which moves with unrelated code, so this case guards the clamp rather
    than probing it.

    The invariants exclude each of those: the exact face count ``2 (side - 1)^2`` and the grid's
    area rule out both a missing triangle and an overlapping pair (with every face wound with the
    normals, an overlap adds area); ordito's validators and an independent count of edge
    incidences and directed half-edges rule out a non-manifold or inconsistently wound result. The
    chain is the 31-point Morton chain of
    ``test_ball_pivoting_on_a_bvh_leaf_past_the_tile_walk_buffer``, which the reported reproducer
    appended; it is far from the grid and never triangulated. The rotated, translated frame moves
    the ball centres away from the origin, where the float32 round-off the band must absorb
    grows with the coordinates.
    """
    radius = 0.05
    positions = spacing * np.arange(side)
    grid_x, grid_y = np.meshgrid(positions, positions, indexing="ij")
    points_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1)
    normal_np = np.array([0.0, 0.0, 1.0])
    if frame == "rotated_offset":
        from scipy.spatial.transform import Rotation

        rotation_np = Rotation.from_euler("xyz", [0.4, -0.7, 1.1]).as_matrix()
        points_np = points_np @ rotation_np.T + np.array([3.0, -2.0, 1.5])
        normal_np = rotation_np @ normal_np
    if chained:
        chain_np = np.array(
            [np.eye(3)[axis] * 2**bit + 0.5 for bit in range(10) for axis in range(3)]
            + [np.full(3, 1023.5)]
        )
        points_np = np.concatenate([points_np, chain_np])
    normals_np = np.tile(normal_np, (points_np.shape[0], 1))
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    _assert_grid_triangulation(vertices_wp, faces_wp, side, spacing, normal_np, face_rtol=1e-3)


@pytest.mark.parametrize("frame", ["axis_aligned", "rotated_offset"])
@pytest.mark.parametrize("deviation", [5e-6, 3e-5, 9e-5])
def test_ball_pivoting_triangulates_a_near_co_circular_grid(
    device: str, deviation: float, frame: Literal["axis_aligned", "rotated_offset"]
) -> None:
    """
    Not a library comparison: cells co-circular to within ``deviation * radius`` triangulate once.

    One corner of every cell (the interior points with both indices odd) is shifted along ``x`` so
    that it lies ``deviation * radius`` off the ball through the cell's other three corners, inside
    or outside by the cell. Every such cell is a near-tie: it has exactly one empty-ball
    triangulation, but only by that margin. The empty-ball test used to count a point up to
    ``1e-4 * radius`` inside the sphere as outside, so every cell passed both triangulations and
    pivots wound faces against the normals; every case here failed then except the rotated 5e-6,
    which fell inside the float32 band where an index rule already decided. Now a point within
    float32 round-off of the sphere is placed in float64, and anything further by its float32
    distance alone, so each cell gets its one triangulation. In the axis-aligned frame 5e-6 lies
    inside that band (about 2e-5 there) and 3e-5 and 9e-5 outside it; in the rotated, translated
    frame the band is wider than all three. Both paths are reached. Same invariants as
    ``test_ball_pivoting_triangulates_an_unjittered_grid``; the axis-aligned arm also checks the
    construction, in float64 on the float32 coordinates the reconstruction is given.
    """
    side, spacing, radius = 40, 0.0125, 0.05
    positions = spacing * np.arange(side)
    grid_x, grid_y = np.meshgrid(positions, positions, indexing="ij")
    points_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1)
    index = np.arange(side)
    interior_odd = (index % 2 == 1) & (index < side - 1)
    shifted = (interior_odd[:, None] & interior_odd[None, :]).ravel()
    # A corner moved by ``shift`` along a cell diagonal's x component moves ``shift / sqrt(2)``
    # off the circumcircle (radius ``spacing / sqrt(2)``), which moves it ``radius`` times
    # ``circumradius / radius**2`` as far off the ball's sphere, relative to ``radius``.
    points_np[shifted, 0] += 2.0 * deviation * radius**2 / spacing

    if frame == "axis_aligned":
        corners_np = points_np.astype(np.float32).astype(np.float64)
        for i, j in [(0, 0), (1, 1), (2, 5), (17, 22)]:
            cell = corners_np[[i * side + j, (i + 1) * side + j, (i + 1) * side + j + 1]]
            fourth = corners_np[i * side + j + 1]
            center = _ball_center_float64(cell, radius)
            assert abs(np.linalg.norm(fourth - center) / radius - 1.0) == pytest.approx(
                deviation, rel=0.05
            )

    normal_np = np.array([0.0, 0.0, 1.0])
    if frame == "rotated_offset":
        from scipy.spatial.transform import Rotation

        rotation_np = Rotation.from_euler("xyz", [0.4, -0.7, 1.1]).as_matrix()
        points_np = points_np @ rotation_np.T + np.array([3.0, -2.0, 1.5])
        normal_np = rotation_np @ normal_np
    normals_np = np.tile(normal_np, (points_np.shape[0], 1))
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    _assert_grid_triangulation(vertices_wp, faces_wp, side, spacing, normal_np, face_rtol=5e-3)


def _ball_center_float64(corners_np: np.ndarray, radius: float) -> np.ndarray:
    """Return the centre of the ``radius`` ball on three ``z = 0`` corners, on the ``+z`` side."""
    a, b, c = corners_np
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    circumcenter = a + (np.cross(normal, ab) * ac.dot(ac) + np.cross(ac, normal) * ab.dot(ab)) / (
        2.0 * normal.dot(normal)
    )
    height = math.sqrt(radius**2 - float(np.sum((circumcenter - a) ** 2)))
    return circumcenter + height * np.array([0.0, 0.0, 1.0])


def _assert_grid_triangulation(
    vertices_wp: wp.array[wp.vec3],
    faces_wp: wp.array[wp.int32],
    side: int,
    spacing: float,
    normal_np: np.ndarray,
    face_rtol: float,
) -> None:
    """
    Assert that a reconstructed ``side x side`` planar grid is its exact triangulation.

    The exact face count and the outline's area rule out both a missing triangle and an
    overlapping pair (with every face wound with the normal, an overlap adds area); ordito's
    validators and an independent count of edge incidences and directed half-edges rule out a
    non-manifold or inconsistently wound result.
    """
    assert od.validation.is_edge_manifold(faces_wp)
    assert od.validation.is_winding_consistent(faces_wp)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    assert vertices_np.shape == (side * side, 3)
    assert faces_np.shape[0] == 2 * (side - 1) ** 2
    multiplicity = edge_multiplicity(faces_np)
    assert int(multiplicity.max()) == 2
    assert int((multiplicity == 1).sum()) == 4 * (side - 1)  # one rim, around the square
    half_edges_np = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]])
    assert np.unique(half_edges_np, axis=0).shape[0] == half_edges_np.shape[0]
    assert warp_to_trimesh(vertices_wp, faces_wp).is_winding_consistent

    corners_np = vertices_np[faces_np]
    face_area_np = 0.5 * np.cross(
        corners_np[:, 1] - corners_np[:, 0], corners_np[:, 2] - corners_np[:, 0]
    )
    assert np.all(face_area_np @ normal_np > 0.0)  # wound with the normals
    assert np.allclose(face_area_np @ normal_np, 0.5 * spacing**2, rtol=face_rtol)
    assert math.isclose(
        float(np.sum(face_area_np @ normal_np)), ((side - 1) * spacing) ** 2, rel_tol=1e-4
    )


@pytest.mark.parity("ball_pivoting", "pymeshlab", "open3d")
def test_ball_pivoting_matches_pymeshlab_and_open3d(device: str):
    """
    Class B (faces matched by input-point index, windings compared): the same triangles.

    BPA adds no vertices, so every output vertex of all three is an input sample (Hausdorff
    3.8e-08) and each face maps to a triple of input indices through a nearest-sample lookup. On
    the triple sets, measured identically on both devices:

    - **open3d** at radii ``(r, 2r)`` returns the *identical* 1 280 faces with the identical
      winding -- the two fronts advance in different orders and still close the same triangles.
    - **MeshLab** (VCGlib's original BPA) returns 1 277, of which 1 274 are ordito's, every one
      wound the same way. The remainder is 3 faces on its side against 6 on ordito's: ordito's
      result is watertight and MeshLab's is not, those three missing faces being unclosed holes.

    Previously a symmetric chamfer at 1.3x its own sampling floor (measured 0.99x, so 1.13x below
    a 2 % scale probe) and a factor-of-two face-count band against open3d.

    **The reference's ``clustering`` parameter is load-bearing and its zero is not "off".** At
    ``clustering=0`` the filter returns **0 faces**; 20% is MeshLab's default and the seed-triangle
    spacing floor the algorithm needs. ``benchmarks/test_reconstruction.py`` passed 0 and so timed a
    filter that reconstructed nothing, and took *longer* doing it than the real thing;
    that row now passes the default.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    radius = _bpa_radius(points_wp)

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)

    meshset_pml = points_to_pymeshlab(points_np, normals_np)
    meshset_pml.generate_surface_reconstruction_ball_pivoting(
        ballradius=ml.PureValue(radius), clustering=20.0
    )
    vertices_pml = np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)
    faces_pml = np.asarray(meshset_pml.current_mesh().face_matrix())
    assert faces_pml.shape[0] > 0, "clustering=0 reconstructs nothing; this must not regress"

    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        points_to_open3d(points_np, normals_np), o3d.utility.DoubleVector([radius, 2.0 * radius])
    )

    # BPA adds no vertices, so all three must be exactly the input samples.
    assert hausdorff_two_sided(mesh_wp.vertices, points_np) < 1e-5
    assert hausdorff_two_sided(vertices_pml, points_np) < 1e-5
    assert hausdorff_two_sided(np.asarray(mesh_o3d.vertices), points_np) < 1e-5
    # Every input point is used, and the closed genus-0 surface has the Euler face count.
    assert np.unique(mesh_wp.faces).size == points_np.shape[0]
    assert mesh_wp.faces.shape[0] == 2 * points_np.shape[0] - 4
    assert mesh_wp.is_watertight

    tree = cKDTree(points_np)

    def wound_triples(vertices_np: np.ndarray, faces_np: np.ndarray) -> set[tuple[int, ...]]:
        """Each face as its input-sample indices, rotated to start at the smallest."""
        rows_np = np.asarray(tree.query(vertices_np)[1], dtype=np.int64)[faces_np]
        return {tuple(row) for row in canonical_winding(rows_np).tolist()}

    def unwound(triples: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
        return {tuple(sorted(row)) for row in triples}

    triples_wp = wound_triples(mesh_wp.vertices, mesh_wp.faces)
    triples_o3d = wound_triples(np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles))
    triples_pml = wound_triples(vertices_pml, faces_pml)
    assert len(triples_wp) == mesh_wp.faces.shape[0]

    assert triples_wp == triples_o3d

    shared = unwound(triples_wp) & unwound(triples_pml)
    assert len(triples_wp & triples_pml) == len(shared), "every shared face is wound alike"
    assert len(unwound(triples_pml) - shared) <= 3
    assert len(unwound(triples_wp) - shared) <= 6
    # MeshLab leaves its three missing faces as holes; ordito closes the surface.
    assert not tm.Trimesh(vertices_pml, faces_pml, process=False).is_watertight


def test_ball_pivoting_small_radius_leaves_holes(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _idx, dist = od.neighbors.query_nearest(points_wp, points_wp, k=2, backend="bvh")
    spacing = float(np.mean(dist.numpy()[:, 1][np.isfinite(dist.numpy()[:, 1])]))

    _v_small, faces_small = od.reconstruction.ball_pivoting(
        points_wp, normals_wp, radius=0.2 * spacing
    )
    _v_ok, faces_ok = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=1.5 * spacing)
    # A ball far smaller than the sampling never rests on three points: far fewer (or no) faces.
    assert faces_small.size < faces_ok.size


def test_ball_pivoting_too_few_points(device: str):
    points_wp = wp.array(np.zeros((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ones((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    _vertices, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp)
    assert faces_wp.size == 0


# --- od.config.deterministic(False): the arrival-order block commits ---


def _reconstruct(device: str, function: str) -> tuple[np.ndarray, np.ndarray]:
    """Run one reconstruction on a sphere cloud (or mesh), returning its vertices and faces."""
    points_wp, normals_wp = _to_warp(*_sphere_cloud(4), device)
    if function == "triangulate_point_cloud":
        result = od.reconstruction.triangulate_point_cloud(points_wp)
    elif function == "ball_pivoting":
        result = od.reconstruction.ball_pivoting(points_wp)
    elif function == "screened_poisson":
        result = od.reconstruction.screened_poisson(points_wp, normals_wp, depth=5, full_depth=4)
    else:
        vertices_wp, faces_wp = od.creation.icosphere(subdivisions=2, device=device)
        result = od.reconstruction.resample_uniform(vertices_wp, faces_wp, voxel_size=0.1)
    return result[0].numpy(), result[1].numpy()


@pytest.mark.parametrize(
    "function", ["triangulate_point_cloud", "ball_pivoting", "screened_poisson", "resample_uniform"]
)
def test_reconstruction_arrival_order_agrees_with_the_default(device: str, function: str) -> None:
    """
    Ordito against ordito: determinism off reconstructs the default's surface.

    The mode reaches only the reductions that set a scale or an orientation (the centroid that
    orients estimated normals, the automatic ball radius, the Poisson iso-value and the signed
    volume that orients the result), whose last bits move no lattice crossing or pivot of these
    inputs: the triangulations must be equal, and the vertices equal to float32 rounding.
    """
    fixed_vertices, fixed_faces = _reconstruct(device, function)
    with od.config.deterministic(False):
        arrival_vertices, arrival_faces = _reconstruct(device, function)
    assert fixed_faces.size > 0
    assert np.array_equal(fixed_faces, arrival_faces)
    assert np.allclose(fixed_vertices, arrival_vertices, rtol=1e-5, atol=1e-5)
