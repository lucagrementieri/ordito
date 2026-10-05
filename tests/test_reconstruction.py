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

import ordito as od
from ordito.kernels import reconstruction as kernel_reconstruction
from ordito.kernels.algorithms import ball_pivoting as kernel_bpa
from tests.comparisons import (
    assert_unordered_rows_equal,
    canonical_winding,
    edge_multiplicity,
    hausdorff_surface_two_sided,
    hausdorff_two_sided,
    lexsort_rows,
    symmetric_chamfer,
    symmetric_surface_distance,
    undirected_edges,
)
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
    points_np: np.ndarray, normals_np: np.ndarray, num_neighbours: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a reference mesh via MeshLib, returning ``(vertices, faces)`` numpy arrays."""
    cloud_ml = points_to_meshlib(points_np, normals_np)
    params_ml = mm.TriangulationParameters()
    params_ml.numNeighbours = num_neighbours
    mesh_ml = mm.triangulatePointCloud(cloud_ml, params_ml)
    return np.asarray(mn.getNumpyVerts(mesh_ml)), np.asarray(mn.getNumpyFaces(mesh_ml.topology))


def _sphere_cloud(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    sphere_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return (sphere_tm.vertices.astype(np.float64), sphere_tm.vertex_normals.astype(np.float64))


def _to_warp(points_np: np.ndarray, normals_np: np.ndarray, device: str):
    """Upload an oriented cloud as one pair, since every Poisson test needs both buffers."""
    return points_to_warp(points_np, device), points_to_warp(normals_np, device)


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


@pytest.mark.parity("delaunay_triangulation", "scipy")
def test_delaunay_matches_scipy_random(device: str):
    """Class B: the same triangulation as Qhull's, compared as an undirected edge set."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(42)
    points_np = rng.random((200, 2)).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy()
    faces_sp = Delaunay(points_np.astype(np.float64)).simplices

    assert np.array_equal(_edge_set(faces_wp), _edge_set(faces_sp.reshape(-1)))


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
    # The excess is hull slivers only: a few triangles, not a different triangulation.
    assert len(faces_od - faces_pv) < 0.05 * len(faces_od)


@pytest.mark.parametrize("n_points", [200, 20_000])
def test_delaunay_matches_warp_edge_flip_from_the_same_seed(device: str, n_points: int):
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

    rng = np.random.default_rng(n_points)
    points_np = rng.random((n_points, 2)).astype(np.float32)
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


def test_delaunay_no_violations(device: str):
    rng = np.random.default_rng(7)
    points_np = rng.random((150, 2)).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)

    # All triangles counter-clockwise.
    tris = points_np.astype(np.float64)[faces_wp]
    cross = _cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert np.all(cross > 0.0)
    assert _incircle_violations(points_np.astype(np.float64), faces_wp) == 0


def test_delaunay_covers_hull(device: str):
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(3)
    points_np = rng.random((120, 2)).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    tris = points_np.astype(np.float64)[faces_wp]
    area = float(np.abs(_cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])).sum() * 0.5)
    hull_area = float(ConvexHull(points_np.astype(np.float64)).volume)
    assert np.isclose(area, hull_area, rtol=1e-5, atol=1e-6)


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


def test_delaunay_grid_perturbed(device: str):
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(11)
    grid = np.stack(np.meshgrid(np.arange(8.0), np.arange(8.0)), axis=-1).reshape(-1, 2)
    points_np = (grid + rng.normal(0.0, 0.05, grid.shape)).astype(np.float32)
    points_wp = points_to_warp_uv(points_np, device)

    faces_wp = od.reconstruction.delaunay_triangulation(points_wp).numpy()
    faces_sp = Delaunay(points_np.astype(np.float64)).simplices
    assert np.array_equal(_edge_set(faces_wp), _edge_set(faces_sp.reshape(-1)))


def test_delaunay_collinear(device: str):
    points_np = np.array([[float(i), 0.0] for i in range(5)], dtype=np.float32)
    points_wp = points_to_warp_uv(points_np, device)
    faces_wp = od.reconstruction.delaunay_triangulation(points_wp)
    assert faces_wp.size == 0


def test_delaunay_too_few(device: str):
    points_wp = wp.array(np.zeros((2, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    with pytest.raises(ValueError, match="at least 3 points"):
        od.reconstruction.delaunay_triangulation(points_wp)


# ---------------------------------------------------------------------------
# triangulate_point_cloud (reference: MeshLib ``triangulatePointCloud``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subdivisions", [3])
def test_sphere_is_closed_manifold(device: str, subdivisions: int):
    points_np, normals_np = _sphere_cloud(subdivisions)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18
    )
    faces_np = faces_wp.numpy()
    n_points = points_np.shape[0]

    # A closed genus-0 triangulation of n points has exactly 2n - 4 faces (Euler).
    assert faces_np.size // 3 == 2 * n_points - 4
    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert od.validation.is_edge_manifold(faces_wp)
    assert od.measures.euler_characteristic(faces_wp) == 2
    # every input point is referenced
    assert np.unique(faces_np).size == n_points

    # reconstructed vertices lie on the unit sphere
    radii = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.allclose(radii, 1.0, rtol=1e-5, atol=1e-5)


def test_torus_is_genus_one(device: str):
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=48, minor_sections=24
    )
    points_np = torus_tm.vertices.astype(np.float64)
    normals_np = torus_tm.vertex_normals.astype(np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=16
    )
    assert od.validation.is_watertight(vertices_wp, faces_wp)
    assert od.validation.is_edge_manifold(faces_wp)
    # genus-1 closed surface: V - E + F = 0
    assert od.measures.euler_characteristic(faces_wp) == 0


@pytest.mark.parametrize("subdivisions", [3])
@pytest.mark.parity("triangulate_point_cloud", "meshlib")
def test_matches_meshlib_reference(device: str, subdivisions: int):
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
    """
    points_np, normals_np = _sphere_cloud(subdivisions)
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


@pytest.mark.parity("triangulate_point_cloud", "meshlib")
def test_torus_matches_meshlib_reference(device: str):
    """
    Class C (a surface distance): the input class where the *diagonal* choice is genuinely free.

    A quad-grid cloud has no canonical triangulation -- each quad can be split either way at equal
    cost -- so unlike the icosphere above the two libraries do not agree row for row: measured
    **575 of 2 304** faces differ. What does agree is the face count (2 304 exactly) and the
    surface, to a two-sided Hausdorff distance of **5.1e-08** against a mean edge length of 0.128,
    i.e. seven orders of magnitude below the triangle scale.

    Mutation probe and margin: dropping ``num_neighbours`` to 4 leaves ordito with 240 faces of
    1 280 on the sphere cloud and a Hausdorff distance of **0.348**, 2.3x the mean edge -- so the
    threshold used here (1 % of the mean edge) is seven orders of magnitude clear of a
    reconstruction that has genuinely gone wrong, and the assert is not tolerating the diagonal
    disagreement by being loose.
    """
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=48, minor_sections=24
    )
    points_np = torus_tm.vertices.astype(np.float64)
    normals_np = torus_tm.vertex_normals.astype(np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    mean_edge = float(
        np.linalg.norm(
            torus_tm.vertices[torus_tm.edges_unique[:, 0]]
            - torus_tm.vertices[torus_tm.edges_unique[:, 1]],
            axis=1,
        ).mean()
    )

    vertices_ml, faces_ml = _meshlib_triangulate(points_np, normals_np, 16)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=16
    )
    faces_wp_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)
    faces_ml_np = np.asarray(faces_ml, dtype=np.int64)

    assert faces_ml_np.shape[0] > 0  # non-vacuity
    assert abs(faces_wp_np.shape[0] - faces_ml_np.shape[0]) <= max(2, faces_ml_np.shape[0] // 100)
    assert (
        hausdorff_surface_two_sided(
            vertices_wp.numpy().astype(np.float64),
            faces_wp_np,
            np.asarray(vertices_ml, dtype=np.float64),
            faces_ml_np,
        )
        < 0.01 * mean_edge
    )


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
    # The intended equator rim is a single large boundary loop, not filled and not fragmented.
    loops = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops) == 1


def test_estimated_normals_path_runs(device: str):
    # Exercise the normal-estimation code path (normals=None). Global orientation of PCA normals is
    # a documented best-effort step, so we check the pipeline runs and yields an edge-manifold mesh
    # whose vertices still lie on the sphere -- not full watertightness.
    points_np, _ = _sphere_cloud(3)
    points_wp = points_to_warp(points_np, device)

    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=18)
    assert faces_wp.numpy().size > 0
    assert od.validation.is_edge_manifold(faces_wp)
    radii = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.allclose(radii, 1.0, rtol=1e-5, atol=1e-5)


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
            out_tris_wp,
            out_counts_wp,
        ],
        device=device,
    )
    assert out_counts_wp.numpy()[0] == 1
    triangle = out_tris_wp.numpy()[0, 0]
    assert set(triangle.tolist()) == {0, 1, 2}


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
            out_tris_wp,
            out_counts_wp,
        ],
        device=device,
    )
    assert out_counts_wp.numpy()[0] == 0


def test_holes_seal_small_hole(device: str):
    rng = np.random.default_rng(0)
    points_np, normals_np = _sphere_cloud(4)
    # Remove a small cluster of points to open a genuine boundary hole.
    seed = points_np[rng.integers(points_np.shape[0])]
    keep = np.linalg.norm(points_np - seed, axis=1) > 0.2
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


def test_empty_cloud(device: str):
    points_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    vertices_wp, faces_wp = od.reconstruction.triangulate_point_cloud(points_wp)
    assert vertices_wp.size == 0
    assert faces_wp.size == 0


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


def test_invalid_parameters(device: str):
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

    Measured with the two depths **interleaved in one process** and read as the minimum of three
    (CLAUDE.md section 15.7), dropping a level is an **8.6x** saving per CPU solve, matching the 8x
    the grid size predicts. Both are negligible on CUDA.

    !!! warning "Quote the ratio, not the wall clock"
        This box's CPU timings swing ~40 % with background load, and the same depth-6 solve has been
        seen to spread more than fourfold across full-suite runs. The *ratio* is stable because both
        sides move together, which is exactly why section 13 asks for an interleaved A/B rather than
        two numbers taken apart.

    Depth 4 is not an option: 2 144 faces is too coarse to resolve the torus hole.

    **Three tests opt out and keep a depth-6 literal**, each because its claim stops holding at 5,
    which is why the level is a helper and not a blanket edit:

    - ``test_poisson_sphere_watertight_manifold`` -- the depth-5 surface self-intersects, so it is
      not
      watertight on *either* device, and the radius tolerances are sized to a depth-6 cell;
    - ``test_poisson_matches_open3d_metric`` and ``..._pymeshlab_metric`` -- the class-C margin
      falls
      from 3.4x / 3.7x to **2.64x / 2.44x**, under section 6's 3x floor.

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


def test_poisson_outward_orientation(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=_poisson_depth(device), full_depth=4
    )
    # Outward normals => positive enclosed volume.
    assert warp_to_trimesh(vertices_wp, faces_wp).volume > 0.0


def test_poisson_dense_solve_converges_in_few_iterations(device: str) -> None:
    """
    Not a library comparison: no reference exposes its Poisson solver's iteration count.

    The dense grid's conjugate gradient is preconditioned by a geometric V-cycle, whose point is an
    iteration count that does not grow with the resolution: every level of the cascade must reach
    the default tolerance inside 30 iterations, which Jacobi -- 100 to 800 per level on this kind of
    input -- does not. A level that runs out warns, so promoting the warning to an error is the
    assertion. The second call is the converse: a budget of two iterations cannot converge, and
    has to say so rather than return a surface from an unconverged field silently.
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
    margin, under the 3x floor section 6 sets for a class-C threshold. The mutation probe still
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


@pytest.mark.slow_cpu(18.5)
def test_poisson_screening_improves_fit(device: str):
    """
    Ordito against ordito: screening ties the surface to the samples, so the fit cannot worsen.

    Also the only test that reconstructs at ``point_weight=0``, which is why the no-degenerate-face
    invariant is asserted here rather than in a test of its own (CLAUDE.md section 7.4). That config
    is ill-conditioned -- the operator is held SPD by a ``1e-4`` floor alone -- and its raw
    marching-cubes output carried 27-64 zero-area triangles, run to run, until
    ``screened_poisson`` grew its ``remove_degenerate_faces`` tail. They are not cosmetic: a
    zero-area face has no normal to orient, and trimesh's ``closest_point`` divides by its
    zero-length edge, so ``_points_to_surface`` below emitted an intermittent
    ``RuntimeWarning: invalid value encountered in divide`` from this test alone. The screened side
    emits none at any depth and is included so the assertion is not one-sided.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_screened, faces_screened = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=_poisson_depth(device), full_depth=4, point_weight=4.0
    )
    vertices_unscreened, faces_unscreened = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=_poisson_depth(device), full_depth=4, point_weight=0.0
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


def test_poisson_requires_normals_and_valid_params(device: str):
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    with pytest.raises(ValueError, match="full_depth"):
        od.reconstruction.screened_poisson(points_wp, normals_wp, depth=6, full_depth=8)
    with pytest.raises(ValueError, match="full_depth"):
        od.reconstruction.screened_poisson(points_wp, normals_wp, depth=11)
    with pytest.raises(ValueError, match="scale"):
        od.reconstruction.screened_poisson(points_wp, normals_wp, scale=0.0)


def test_poisson_too_few_points(device: str):
    points_wp = wp.array(np.zeros((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ones((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="at least 3 points"):
        od.reconstruction.screened_poisson(points_wp, normals_wp, depth=4, full_depth=3)


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
    pytest.param("torus", "dense", id="torus-dense"),
    pytest.param("torus", "adaptive", id="torus-adaptive"),
]


@pytest.mark.parametrize(("shape", "method"), _POISSON_WATERTIGHT_CASES)
def test_poisson_watertight_manifold(
    device: str, shape: str, method: Literal["dense", "adaptive"]
) -> None:
    """
    Not a library comparison: reconstructing a closed surface from a clean, oriented cloud closes.

    ``sphere`` asserts the reconstructed radius hugs the unit sphere in addition to the topology;
    ``torus`` only checks watertightness and genus, since it has no analogous single-number shape
    to pin. ``adaptive`` additionally asserts outward orientation directly (the dense backend has
    its own [`test_poisson_outward_orientation`]).

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


def test_poisson_adaptive_matches_dense(device: str):
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_dense, faces_dense = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=_poisson_depth(device), full_depth=4, method="dense"
    )
    vertices_adaptive, faces_adaptive = od.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=_poisson_depth(device), full_depth=4, method="adaptive"
    )
    # Same iso-surface on two different grids, so a surface metric rather than a correspondence.
    # Its old ``symmetric_chamfer(...) < 0.05`` sat only 1.8x above that helper's 0.028 sampling
    # floor, which is most of what it was measuring; sample-to-surface has no floor.
    mean_distance, _ = symmetric_surface_distance(
        warp_to_trimesh(vertices_dense, faces_dense),
        warp_to_trimesh(vertices_adaptive, faces_adaptive),
    )
    assert mean_distance < 0.02


def test_poisson_adaptive_screening_improves_fit(device: str):
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_screened, faces_screened = od.reconstruction.screened_poisson(
        points_wp,
        normals_wp,
        depth=_poisson_depth(device),
        full_depth=4,
        point_weight=4.0,
        method="adaptive",
    )
    vertices_unscreened, faces_unscreened = od.reconstruction.screened_poisson(
        points_wp,
        normals_wp,
        depth=_poisson_depth(device),
        full_depth=4,
        point_weight=0.0,
        method="adaptive",
    )
    fit_screened = _points_to_surface(points_np, warp_to_trimesh(vertices_screened, faces_screened))
    fit_unscreened = _points_to_surface(
        points_np, warp_to_trimesh(vertices_unscreened, faces_unscreened)
    )
    assert fit_screened <= fit_unscreened + 1e-4


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


def test_poisson_invalid_method(device: str):
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    with pytest.raises(ValueError, match="method"):
        od.reconstruction.screened_poisson(points_wp, normals_wp, method="bogus")  # pyright: ignore[reportArgumentType]  # deliberately off-menu


# ---------------------------------------------------------------------------
# Marching cubes and uniform resampling (analytic + pymeshlab / skimage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 0.2, -0.2])
def test_resample_uniform_offsets_a_sphere(
    device: str, offset: float, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class C (a radius bound): a sphere is the one shape whose offset surface is known exactly.

    Both signs are covered because they are different code paths in spirit — a positive offset needs
    the lattice padded beyond the bounding box (or it clips) and a negative one does not.
    """
    sphere_tm, _sphere_tm_wp = icosphere
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    voxel_size = 0.05

    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size, offset=offset
    )
    radii_np = np.linalg.norm(out_vertices_wp.numpy(), axis=1)
    # The icosphere is *inscribed*, so its own surface sits between ``cos`` of half the face angle
    # and 1; the offset shifts that band without widening it much.
    assert np.abs(radii_np - (1.0 + offset)).max() < 2.0 * voxel_size

    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)
    assert out_tm.is_watertight
    assert out_tm.volume > 0.0


def test_resample_uniform_repairs_a_broken_mesh(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the topology comes from the grid, so the input's cannot leak.

    The input here has duplicated faces, an inverted one and a non-manifold edge — three defects
    that each need their own function in [`ordito.repair`][ordito.repair] — and the resampled
    result is a clean watertight sphere regardless.
    """
    sphere_tm, _sphere_tm_wp = icosphere
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
    assert np.isclose(np.abs(out_tm.volume), 4.0 / 3.0 * np.pi, rtol=0.1)


@pytest.mark.parity("resample_uniform", "pymeshlab")
def test_resample_uniform_matches_pymeshlab(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class C (a surface distance): the same algorithm at the same absolute cell size.

    One parameter hazard, found by probing: its ``offset`` as a ``PercentageValue`` runs from full
    erosion at ``0%`` to full dilation at ``100%``, so **``PercentageValue(50)`` — its default — is
    the *zero* offset** and ``PercentageValue(0)`` erodes a unit sphere down to radius 0.30. Passing
    ``PureValue(0.0)`` instead means an absolute offset of zero, which is what ``offset=0.0`` is
    here.

    What must agree is the surface: both watertight, both enclosing the sphere's volume, and a
    two-sided Hausdorff distance between them of well under a cell.

    **Mutation probe**, and it retightened the threshold. The measured agreement is **0.00313**,
    identical on both devices; the bug class is resampling a *differently scaled* surface, so the
    probe re-runs the reference on a scaled sphere and reads the distance: 1.05x gives 0.0514, 1.10x
    gives 0.1020, 1.15x gives 0.1521. The original ``2.0 * voxel_size`` bar (0.12, a 38x headroom)
    admitted every scale error up to 10 %, so it is now ``0.5 * voxel_size``: 0.03, still **9.6x**
    the measured agreement, and it now separates a 5 % scale error.
    """
    sphere_tm, _sphere_tm_wp = icosphere
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

    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(pml_tm.volume), rtol=0.05)

    # Two-sided Hausdorff between the surfaces, within a cell.
    sample_wp, _face = tm.sample.sample_surface(out_tm, 4000, seed=0)[:2]
    sample_pml, _face_pml = tm.sample.sample_surface(pml_tm, 4000, seed=1)[:2]
    assert np.abs(tm.proximity.signed_distance(pml_tm, sample_wp)).max() < 0.5 * voxel_size
    assert np.abs(tm.proximity.signed_distance(out_tm, sample_pml)).max() < 0.5 * voxel_size


@pytest.mark.parity("resample_uniform", "igl")
def test_resample_uniform_matches_igl(device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (no vertex correspondence): the two isosurfaces coincide to **0.06 of a voxel**.

    ``igl.offset_surface(V, F, isolevel, s, sign_type)`` samples the same signed distance field on a
    grid and marches it. Two named parameter transforms put the two on one lattice: ``isolevel=0``
    is ordito's zero offset, and ``s`` is a *cell count along the longest axis* rather than a
    length, so it gets ``round(longest_extent / voxel_size)``. The sign mode is ``PSEUDONORMAL``,
    which ``tests/test_proximity.py::test_signed_distance_on_mesh_matches_igl`` establishes agrees
    with ordito's default to 8e-8 -- the winding modes would scale the field by ``1 - 2w`` and move
    the isosurface.

    No correspondence exists between the outputs (4 186 vertices against ordito's 5 310 on this
    fixture, since the two march the lattice into different triangle sets), so the comparison is the
    surface: both watertight, enclosed volumes within 5%, and a two-sided Hausdorff distance under a
    quarter of a voxel.

    **Bug class excluded:** a grid anchored differently, or an isolevel or sign convention that
    shifts the surface -- exactly what the plan flagged as the risk for this pair. **Mutation probe,
    measured:** re-running igl at ``isolevel=0.02`` and ``0.05`` moves the one-sided Hausdorff to
    **0.36 and 0.86 voxels** against 0.06 at zero, so the ``0.25``-voxel bound sits 4.2x above the
    measured agreement and fails on a shift of a third of a voxel. That is what makes it a test of
    the anchoring rather than of "both are roughly a sphere".
    """
    sphere_tm, _sphere_tm_wp = icosphere
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

    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)

    assert mesh_igl.is_watertight
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(mesh_igl.volume), rtol=0.05)

    sample_wp, _face_wp = tm.sample.sample_surface(out_tm, 4000, seed=0)[:2]
    sample_igl, _face_igl = tm.sample.sample_surface(mesh_igl, 4000, seed=1)[:2]
    assert np.abs(tm.proximity.signed_distance(mesh_igl, sample_wp)).max() < 0.25 * voxel_size
    assert np.abs(tm.proximity.signed_distance(out_tm, sample_igl)).max() < 0.25 * voxel_size


@pytest.mark.parity("resample_uniform", "meshlib")
def test_resample_uniform_matches_meshlib(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class C (no correspondence, and a different amount of work): both land on the same surface.

    ``rebuildMesh`` samples the same signed distance field on a grid of the same ``voxelSize`` and
    marches it, then -- unlike ordito, igl and MeshLab -- **decimates** the result at its own
    defaults (``decimate`` and ``preSubdivide`` are both on). So the face counts are not comparable
    at all: measured 1 332 faces against ordito's 7 772 at a 2 % voxel on ``icosphere(3)``. What is
    comparable is the surface, and the two agree to a two-sided Hausdorff distance of **0.084 of a
    voxel**, each sitting within 0.11 voxels of the input.

    **Bug class excluded:** a grid anchored differently, or a sign or isolevel convention that moves
    the isosurface -- the same risk the igl pair above is written against, checked here against an
    independent implementation of the whole pipeline rather than of the field alone. **Mutation
    probe, measured:** running MeshLib at 4x the voxel size moves the distance to **0.43 voxels**
    and asking ordito for ``offset=0.1`` moves it to **1.53 voxels**, against 0.084 when the two
    agree -- so the 0.25-voxel bound is 3x above the measured agreement and fails on either
    mismatch.
    """
    sphere_tm, _sphere_wp = icosphere
    diagonal = float(
        np.linalg.norm(sphere_tm.vertices.max(axis=0) - sphere_tm.vertices.min(axis=0))
    )
    voxel_size = 0.02 * diagonal

    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
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
    assert np.isclose(np.abs(out_tm.volume), np.abs(rebuilt_tm.volume), rtol=0.05)
    assert (
        hausdorff_surface_two_sided(
            out_tm.vertices, out_tm.faces, rebuilt_tm.vertices, rebuilt_tm.faces
        )
        < 0.25 * voxel_size
    )
    # Both are a resampling *of the input*, not merely of each other.
    for resampled_tm in (out_tm, rebuilt_tm):
        assert (
            hausdorff_surface_two_sided(
                resampled_tm.vertices, resampled_tm.faces, sphere_tm.vertices, sphere_tm.faces
            )
            < 0.25 * voxel_size
        )


def test_resample_uniform_coarser_is_smaller(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Not a library comparison: a wider voxel gives fewer triangles, and still a closed surface."""
    sphere_tm, _sphere_tm_wp = icosphere
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    counts = []
    for voxel_size in (0.05, 0.1, 0.2):
        _out_vertices_wp, out_faces_wp = od.reconstruction.resample_uniform(
            vertices_wp, faces_wp, voxel_size=voxel_size
        )
        counts.append(out_faces_wp.size // 3)
        assert od.validation.is_edge_manifold(out_faces_wp, allow_boundary_edges=False)
    assert counts == sorted(counts, reverse=True)


def test_resample_uniform_invalid(device: str) -> None:
    sphere_tm = tm.creation.icosphere(subdivisions=1, radius=1.0)
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    with pytest.raises(ValueError, match="voxel_size > 0"):
        od.reconstruction.resample_uniform(vertices_wp, faces_wp, voxel_size=0.0)


def test_resample_uniform_empty(device: str) -> None:
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
# open3d BPA (``_o3d``) is used only as a loose face-count sanity reference.
# ======================================================================================


def test_ball_pivoting_interpolates_input(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    assert faces_wp.size > 0

    # Interpolating: every output vertex coincides with an input point.
    from scipy.spatial import cKDTree

    distances = cKDTree(points_np).query(vertices_np)[0]
    assert np.max(distances) < 1e-6
    # Most input points are incorporated on a well-sampled sphere.
    assert len(vertices_np) >= 0.8 * points_np.shape[0]


def test_ball_pivoting_edge_manifold(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _vertices, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp)
    # The cleanup tail removes non-manifold faces, so no edge is shared by more than two faces.
    assert edge_multiplicity(faces_wp.numpy().reshape(-1, 3)).max() <= 2


def test_ball_pivoting_closes_a_dense_sphere(device: str):
    """
    The strongest end-to-end guard available: a uniformly sampled closed surface must close.

    With a persistent front and Border-edge retirement, a subdivided icosphere reconstructs to
    exactly the Euler face count with no boundary edge at all. That single assertion catches both
    directions of failure at once — retiring an edge that could still have succeeded would leave
    holes, and letting colliding fronts triangulate a neighbourhood twice (the artefact the
    front-rebuilt-per-wave design produced, at 24% boundary edges and 3.1 faces per vertex) would
    push the face count well past ``2 v - 4``.
    """
    from scipy.spatial import cKDTree

    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    multiplicity = edge_multiplicity(faces_np)

    assert int((multiplicity == 1).sum()) == 0  # watertight
    assert int(multiplicity.max()) == 2  # edge-manifold
    n_referenced = len(np.unique(faces_np))
    assert n_referenced == points_np.shape[0]  # every input point used
    assert faces_np.shape[0] == 2 * n_referenced - 4  # Euler, for a closed genus-0 surface

    # And it interpolates: every input point is a vertex of the result.
    assert np.max(cKDTree(vertices_wp.numpy().astype(np.float64)).query(points_np)[0]) < 1e-6


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
    order has nothing left to decide — and asserting on one would be vacuous. Measured on this
    torus, four runs of the pre-fix implementation give 2 980 / 3 036 / 3 042 / 3 089 faces and
    four different meshes, against a constant 3 269 and one mesh after it.

    Two assertions, covering different halves:

    * the wave loop's own output, compared as a **wound** triangle set — same triangles and same
      winding, which is the property the key buys. Not compared buffer-to-buffer: ``CNT_FACE`` is a
      ``wp.atomic_add``, so the row order is deliberately still arrival-ordered;
    * the public function, compared as an **unoriented** triangle set, because its cleanup tail
      runs [`repair.make_winding_consistent`][ordito.repair.make_winding_consistent], whose
      arbitrary per-component seed face is still chosen nondeterministically (measured on this same
      fixture). That is a separate defect in that module, and it is why the second assertion sorts
      within the row rather than using ``canonical_winding``.
    """
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=64, minor_sections=32
    )
    points_wp, normals_wp = _to_warp(torus_tm.vertices, torus_tm.vertex_normals, device)
    n_points = points_wp.size
    # The wrapper's own auto-radius, recomputed on the host and then passed explicitly to every run
    # below -- including the two public calls. That is load-bearing, not tidiness: the wrapper's own
    # `radius <= 0` guess averages the spacings with a device reduction whose summation order is not
    # fixed, so it lands a few ULP apart between calls and tips a borderline pivot. Dropping
    # `radius=` from either public call makes this test flaky rather than stricter.
    spacing = od.neighbors.query_nearest(points_wp, points_wp, k=7, backend="bvh")[1].numpy()[:, 1:]
    radius = 1.5 * float(spacing[np.isfinite(spacing) & (spacing > 0.0)].mean())

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

    _vertices, first_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    _vertices, second_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    assert_unordered_rows_equal(
        np.sort(first_wp.numpy().reshape(-1, 3), axis=1),
        np.sort(second_wp.numpy().reshape(-1, 3), axis=1),
    )


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


@pytest.mark.parametrize("max_waves", [1, 2, 3, 4, 7, 8])
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
    odd number of no-op waves on any cloud at all, and the even values are the controls that pass
    either way. A fixture large enough to reach ``compact()`` on its own is ~25 000 points, far too
    slow for a unit test -- the parameter reaches the same code path on 162.

    Buffer *identity* is the assertion rather than any count: ``id(state.front_out)`` before a wave
    that runs is the buffer that wave writes, and ``grow()`` / ``compact()`` each re-establish the
    invariant themselves (one rebuilds ``front_in`` from the edge table, the other swaps after
    writing), so recording their result covers a run that takes either.

    The mutation probe bites on the final assert, and only for the odd values: reverting the parity
    correction fails ``max_waves`` 1, 3 and 7 with *"front_in is not the buffer the last wave that
    ran wrote"* and leaves 2, 4 and 8 passing. The entry assert inside ``compact()`` is a second
    guard on the one place a violation is actually *read* rather than merely present; neither it
    nor the ``grow()`` arm fires at these wave caps, which is the point of keeping the caps small.
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


@pytest.mark.parity("ball_pivoting", "open3d")
def test_ball_pivoting_face_count_near_open3d(device: str):
    """
    Class C (a face count within a band): the two BPA implementations pick different triangles.

    Ball pivoting's output depends on its seed order and its pivot tie-breaks, so no
    correspondence exists -- what must agree is roughly how much surface got covered, at the
    *same* radius, which is derived from the cloud's own spacing and handed to both.
    ``benchmarks/README`` records open3d as a loose face-count reference only, and this is that
    comparison.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    # Auto-guessed radius roughly matches the mean spacing; use it for open3d too.
    _idx, dist = od.neighbors.query_nearest(points_wp, points_wp, k=7, backend="bvh")
    spacing = float(np.mean(dist.numpy()[:, 1:][np.isfinite(dist.numpy()[:, 1:])]))
    radius = 1.5 * spacing

    _vertices, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    n_faces_od = faces_wp.size // 3

    pcd = points_to_open3d(points_np, normals_np)
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, 2.0 * radius])
    )
    n_faces_o3d = np.asarray(mesh_o3d.triangles).shape[0]
    # Same order of magnitude as open3d (both reconstruct ~2n triangles on a closed sphere).
    assert 0.5 * n_faces_o3d <= n_faces_od <= 2.0 * n_faces_o3d


@pytest.mark.parity("ball_pivoting", "pymeshlab")
def test_ball_pivoting_matches_pymeshlab(device: str):
    """
    Class C against VCGlib's original BPA -- but a much tighter one than the open3d row above.

    Two ball-pivoting fronts advance in different orders and produce different triangles, so there
    is no face correspondence to recover. What is comparable is sharp: BPA adds no vertices, so the
    **vertex sets are identical** (Hausdorff **3.8e-08**, Class A), and at the same radius the two
    close the same surface -- **1 280 faces against 1 277**, 0.23% apart.

    **The reference's ``clustering`` parameter is load-bearing and its zero is not "off".** At
    ``clustering=0`` the filter returns **0 faces**; 20% is MeshLab's default and the seed-triangle
    spacing floor the algorithm needs. ``benchmarks/test_reconstruction.py`` passed 0 and so timed a
    filter that reconstructed nothing, and took *longer* doing it than the real thing;
    that row now passes the default.

    **Bug class excluded:** a front that closes the surface at the wrong scale or drifts off the
    samples. **Mutation probe** on the chamfer, normalized by the reference's own sampling floor
    (0.0196 at 8 000 samples, essentially the whole signal, so the raw number is not usable):
    measured ratio **0.99**, against **1.47** for a 2% scale of ordito's output, **2.76** for 5%
    and
    **3.57** for a half-spacing translation. The 1.3 bound therefore sits 1.3x above the measured
    agreement and 1.13x below the smallest probe -- deliberately recorded as the weakest link here,
    because two BPA meshes over one point set cannot differ by more than the sampling density. The
    face-count and vertex-set asserts are the ones with real margin (21x and exact).

    One genuine difference, asserted rather than smoothed over: ordito's result is watertight and
    MeshLab's is not, because those 3 missing faces are unclosed holes.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _index_wp, distance_wp = od.neighbors.query_nearest(points_wp, points_wp, k=7, backend="bvh")
    neighbor_distance_np = distance_wp.numpy()[:, 1:]
    spacing = float(np.mean(neighbor_distance_np[np.isfinite(neighbor_distance_np)]))
    radius = 1.5 * spacing

    vertices_wp, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    mesh_wp = warp_to_trimesh(vertices_wp, faces_wp)

    meshset_pml = points_to_pymeshlab(points_np, normals_np)
    meshset_pml.generate_surface_reconstruction_ball_pivoting(
        ballradius=ml.PureValue(radius), clustering=20.0
    )
    mesh_pml = tm.Trimesh(
        np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64),
        np.asarray(meshset_pml.current_mesh().face_matrix()),
        process=False,
    )
    assert mesh_pml.faces.shape[0] > 0, "clustering=0 reconstructs nothing; this must not regress"

    # BPA adds no vertices, so both must be exactly the input samples.
    assert hausdorff_two_sided(mesh_wp.vertices, points_np) < 1e-5
    assert hausdorff_two_sided(mesh_wp.vertices, mesh_pml.vertices) < 1e-5
    assert 0.95 <= mesh_wp.faces.shape[0] / mesh_pml.faces.shape[0] <= 1.05

    floor = symmetric_chamfer(mesh_pml, mesh_pml, n_samples=8000)
    assert symmetric_chamfer(mesh_wp, mesh_pml, n_samples=8000) < 1.3 * floor

    # ordito closes the surface; MeshLab leaves those three faces as holes.
    assert mesh_wp.is_watertight
    assert not mesh_pml.is_watertight


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


def test_ball_pivoting_estimated_normals(device: str):
    points_np, _normals = _sphere_cloud(3)
    points_wp = points_to_warp(points_np, device)
    # normals=None triggers PCA normal estimation (valid for this star-shaped cloud).
    _vertices, faces_wp = od.reconstruction.ball_pivoting(points_wp, None)
    assert faces_wp.size > 0


def test_ball_pivoting_too_few_points(device: str):
    points_wp = wp.array(np.zeros((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ones((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    _vertices, faces_wp = od.reconstruction.ball_pivoting(points_wp, normals_wp)
    assert faces_wp.size == 0
