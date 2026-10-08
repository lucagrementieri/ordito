"""Regression tests for ``ordito.sample`` vs ``trimesh.sample`` (CPU reference)."""

from __future__ import annotations

import heapq
import math
from typing import Any, cast

import igl
import numpy as np
import numpy.typing as npt
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import torch
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

import ordito as od
from ordito.kernels import sample as kernel_sample
from tests.comparisons import assert_unordered_rows_equal, chamfer_two_sided
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_warp,
    points_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    warp_empty,
)


@pytest.mark.parametrize("count", [4096, 100_000])
def test_sample_fibonacci_sphere_phase_stays_low_discrepancy(device: str, count: int):
    """
    Class A against a float64 NumPy evaluation of the lattice's own closed form.

    The spiral phase is ``count`` multiples of the golden angle, so it grows without bound in
    ``count``; accumulating it in float32 lost the low digits the low-discrepancy property lives
    in. Measured before the fix: azimuth error 1.7e-04 rad at 1 024 rising to 0.21 rad at 1e6, and
    a minimum neighbour spacing 11 % below the reference's at 100 000. ``count = 100_000`` is
    asserted because that is where the loss is unambiguous and the test is still cheap; the same
    hazard and the same float64 fix are documented at
    ``kernels/bounds.oriented_box_candidate_axes``. At both counts every direction is also unit and
    the set's centroid sits at the origin, as a near-uniform covering of the sphere requires.
    """
    directions_np = od.sample.sample_fibonacci_sphere(count, device=device).numpy()
    assert directions_np.shape == (count, 3)
    assert np.allclose(np.linalg.norm(directions_np, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(directions_np.mean(axis=0), 0.0, atol=1e-2)

    index_np = np.arange(count, dtype=np.float64)
    z_np = 1.0 - 2.0 * (index_np + 0.5) / count
    radius_np = np.sqrt(np.maximum(0.0, 1.0 - z_np * z_np))
    theta_np = math.pi * (3.0 - math.sqrt(5.0)) * index_np
    expected_np = np.stack(
        [radius_np * np.cos(theta_np), radius_np * np.sin(theta_np), z_np], axis=1
    )
    assert np.allclose(directions_np, expected_np, rtol=1e-5, atol=1e-5)

    # The property the lattice exists for, and the one the float32 phase actually destroyed: the
    # closest pair must be no tighter than the reference construction's.
    def min_spacing(points_np: np.ndarray) -> float:
        distances, _ = cKDTree(points_np).query(points_np, k=2)
        return float(np.asarray(distances)[:, 1].min())

    assert min_spacing(directions_np) >= 0.99 * min_spacing(expected_np)


def test_sample_fibonacci_hemisphere_positive_z(device: str):
    directions_np = od.sample.sample_fibonacci_hemisphere(1000, device=device).numpy()
    assert directions_np.shape == (1000, 3)
    assert np.all(directions_np[:, 2] > 0.0)
    norms = np.linalg.norm(directions_np, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5)


def test_sample_fibonacci_cone(device: str) -> None:
    """Every direction inside the cone, and both ends of the range match sphere/hemisphere."""
    half_angle = np.deg2rad(25.0)
    directions_np = od.sample.sample_fibonacci_cone(512, half_angle, device=device).numpy()
    assert np.allclose(np.linalg.norm(directions_np, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    polar_np = np.arccos(np.clip(directions_np[:, 2], -1.0, 1.0))
    assert polar_np.max() <= half_angle + 1e-6
    # Uniform in solid angle means uniform in z, so the mean z is the midpoint of the band.
    assert np.isclose(directions_np[:, 2].mean(), 0.5 * (1.0 + np.cos(half_angle)), atol=1e-3)

    assert np.allclose(
        od.sample.sample_fibonacci_cone(64, np.pi / 2.0, device=device).numpy(),
        od.sample.sample_fibonacci_hemisphere(64, device=device).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        od.sample.sample_fibonacci_cone(64, np.pi, device=device).numpy(),
        od.sample.sample_fibonacci_sphere(64, device=device).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sample_fibonacci_cone_invalid(device: str) -> None:
    with pytest.raises(ValueError, match=r"half_angle must be in \(0, pi\]"):
        od.sample.sample_fibonacci_cone(8, 0.0, device=device)
    with pytest.raises(ValueError, match=r"half_angle must be in \(0, pi\]"):
        od.sample.sample_fibonacci_cone(8, 4.0, device=device)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("sample_surface", "trimesh", "igl")
def test_sample_surface(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B, three samplers against the area law they all claim: per-face frequency / area fraction.

    Two seeded RNGs cannot be aligned, so the comparable quantity is the *distribution* rather than
    the samples: each library's per-face sample frequency must match that face's share of the total
    area. All three return the face index directly, so the only transform is the ``bincount``.

    The bug class this excludes is a mis-weighted CDF -- sampling by face *index* or uniformly per
    face instead of by area, which is the classic error in this routine and is invisible to a
    count-and-on-surface check. ``half_torus`` is the fixture because its faces vary in area by
    construction, so a uniform-per-face sampler fails the assert; on an icosahedron, where every
    face has the same area, it would pass.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    count = 10_000
    n_faces = int(mesh_tm.faces.shape[0])
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, seed=0)[1]
    freq_tm = np.bincount(face_idx_tm, minlength=n_faces) / count

    face_idx_wp = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, count, seed=0)[1]
    freq_wp = np.bincount(face_idx_wp.numpy(), minlength=n_faces) / count

    face_idx_igl = igl.random_points_on_mesh(
        count,
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
        0,
    )[1]
    freq_igl = np.bincount(face_idx_igl, minlength=n_faces) / count

    freq_expected = mesh_tm.area_faces / mesh_tm.area_faces.sum()
    assert np.allclose(freq_wp, freq_expected, rtol=0.08, atol=0.008)
    assert np.allclose(freq_tm, freq_expected, rtol=0.08, atol=0.008)
    assert np.allclose(freq_igl, freq_expected, rtol=0.08, atol=0.008)


@pytest.mark.parity("sample_surface", "pytorch3d")
def test_sample_surface_matches_pytorch3d(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C: a distributional bound, because two independent RNGs give no correspondence at all.

    The statistic is the two-sided Chamfer distance between the two 1 000-point sets, and it is
    bounded rather than matched: it reads **0.0299**, the sampling floor (ordito against itself at
    another seed reads 0.0316). **Mutation probe and margin:** sampling the reference from a mesh
    scaled by 1.15 takes it to **0.200**, a factor of 6.7; the 0.09 bar sits 3x above the floor and
    2.2x below the mutation (a 1.05x scale, 0.051, is under it).

    **Bug class excluded:** a sampler that lands off the surface, concentrates on the wrong faces,
    or ignores its area weights -- all three move the point cloud far enough to blow the bound. Not
    excluded: any defect that preserves the surface distribution, e.g. a correlated RNG. The
    per-face proportionality assert covers the weighting half directly: ``sphere_irregular``'s face
    areas span 6 129x and the per-face counts of 20 000 samples correlate **0.985** with them (on a
    regular icosphere the areas spanned 1.05x and the correlation was noise).
    """
    mesh_tm, mesh_wp = sphere_irregular
    torch.manual_seed(4)
    samples_p3d = p3d_ops.sample_points_from_meshes(trimesh_to_pytorch3d(mesh_tm), 1000)[0].numpy()
    samples_wp, _ = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, 1000, seed=4)

    assert samples_p3d.shape == (1000, 3)
    # Both on the surface (measured 4.1e-7 and 2.6e-7 off it), then the distributions.
    assert tm.proximity.closest_point(mesh_tm, samples_p3d)[1].max() < 1e-5
    assert tm.proximity.closest_point(mesh_tm, samples_wp.numpy())[1].max() < 1e-5
    assert chamfer_two_sided(samples_wp.numpy(), samples_p3d) < 0.09

    # The mutation probe, run rather than merely recorded: a 1.15x mesh must fail that bound.
    scaled_tm = mesh_tm.copy()
    scaled_tm.apply_scale(1.15)
    torch.manual_seed(4)
    scaled_p3d = p3d_ops.sample_points_from_meshes(trimesh_to_pytorch3d(scaled_tm), 1000)[0].numpy()
    assert chamfer_two_sided(samples_wp.numpy(), scaled_p3d) > 0.09

    # Area weighting, on a mesh whose face areas actually differ.
    _, face_indices_wp = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, 20000, seed=1)
    areas_np = od.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)[1].numpy()
    counts_np = np.bincount(face_indices_wp.numpy(), minlength=len(mesh_tm.faces))
    assert areas_np.max() / areas_np.min() > 100.0
    assert float(np.corrcoef(counts_np, areas_np)[0, 1]) > 0.9


def test_sample_surface_with_face_weights(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (a frequency comparison): two RNGs cannot produce the same samples.

    What is comparable is the *distribution* of chosen faces, so the per-face frequency is
    compared against trimesh's under the same weights -- a linear ramp, so an implementation
    ignoring weights gives a flat histogram and fails clearly rather than marginally.
    """
    mesh_tm, mesh_wp = sphere_irregular
    count = 10_000

    weights_np = np.arange(mesh_tm.faces.shape[0], dtype=np.float32)
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, face_weight=weights_np, seed=0)[1]

    weights_wp = wp.array(weights_np, dtype=wp.float32, device=mesh_wp.points.device)
    face_idx_wp = od.sample.sample_surface(
        mesh_wp.points, mesh_wp.indices, count, face_weight=weights_wp, seed=0
    )[1]

    n_faces = int(mesh_tm.faces.shape[0])
    freq_wp = np.bincount(face_idx_wp.numpy(), minlength=n_faces) / count
    freq_tm = np.bincount(face_idx_tm, minlength=n_faces) / count

    freq_expected = weights_np / weights_np.sum()
    assert np.allclose(freq_wp, freq_expected, rtol=0.07, atol=0.01)
    assert np.allclose(freq_tm, freq_expected, rtol=0.07, atol=0.01)


def test_sample_surface_poisson_disk_count_and_on_surface(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
):
    """Exactly ``count`` samples, each with its face index, every one on the surface."""
    mesh_tm, mesh_wp = sphere_irregular
    count = 100
    pts, fids = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, seed=1
    )
    assert pts.shape == (count,)
    assert fids.shape == (count,)
    _, dists, _ = tm.proximity.closest_point(mesh_tm, pts.numpy())
    assert np.all(dists < 1e-4)


def test_sample_surface_poisson_disk_min_distance(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = sphere_irregular
    count = 100
    init_factor = 5.0
    surface_area = float(mesh_tm.area)
    ratio = 1.0 / init_factor
    r_max = 2.0 * math.sqrt((surface_area / count) / (2.0 * math.sqrt(3.0)))
    r_min = r_max * 0.65 * (1.0 - ratio**1.5)

    points, _ = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, init_factor=init_factor, seed=2
    )
    min_dist = float(pdist(points.numpy()).min())
    assert min_dist >= r_min * 0.9


def _sequential_sample_elimination(
    points_np: np.ndarray, count: int, r_max: float, r_min: float
) -> np.ndarray:
    """
    Weighted sample elimination by its definition: delete the heaviest point, one at a time.

    The weight of a point is ``sum (1 - max(d, r_min) / r_max) ** 8`` over its neighbours inside
    ``r_max`` (Yuksel 2015; open3d's ``sample_points_poisson_disk`` uses the same constants), and
    each deletion takes its terms off its neighbours' weights. Ties go to the smaller index. Kept as
    a heap in float32, the precision ordito's weights carry. Returns the kept mask.
    """
    # One list of neighbour indices per point (scipy's stubs type the batched call as one row).
    neighbours = cast(
        "list[list[int]]",
        np.asarray(cKDTree(points_np).query_ball_point(points_np, r_max), dtype=object).tolist(),
    )

    def term(i: int, j: int) -> np.float32:
        distance = max(float(np.linalg.norm(points_np[i] - points_np[j])), r_min)
        return np.float32((1.0 - distance / r_max) ** 8)

    weights = np.array(
        [
            sum((term(i, j) for j in row if j != i), np.float32(0.0))
            for i, row in enumerate(neighbours)
        ],
        dtype=np.float32,
    )
    alive = np.ones(points_np.shape[0], dtype=bool)
    heap = [(-weights[i], i) for i in range(points_np.shape[0])]
    heapq.heapify(heap)
    remaining = points_np.shape[0]
    while remaining > count:
        negated, i = heapq.heappop(heap)
        if not alive[i] or -negated != weights[i]:
            continue
        alive[i] = False
        remaining -= 1
        for j in neighbours[i]:
            if j != i and alive[j]:
                weights[j] -= term(i, j)
                heapq.heappush(heap, (-weights[j], j))
    return alive


def test_sample_surface_poisson_disk_keeps_the_sequential_elimination_set(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the parallel rounds keep exactly the sequential algorithm's samples.

    The oracle is the algorithm's own definition (``_sequential_sample_elimination``) run on the
    same initial pool -- ``sample_surface`` at the same seed and ``init_factor * count`` points. The
    rounds used to delete every local weight maximum at once, which spends the deletion budget on
    sparse regions that sequential elimination never reaches: on this mesh 18 of 20 seeds kept a
    pair under 0.15 ``r_max`` apart where the sequential set keeps 0.66 or more. With the round's
    weight floor the kept set is the sequential one.
    """
    mesh_tm, mesh_wp = sphere_irregular
    count, init_factor, seed = 100, 5.0, 4
    pool_wp, _ = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, 500, seed=seed)
    pool_np = np.asarray(pool_wp.numpy(), dtype=np.float64)
    r_max = 2.0 * math.sqrt((float(mesh_tm.area) / count) / (2.0 * math.sqrt(3.0)))
    r_min = r_max * 0.65 * (1.0 - (count / 500.0) ** 1.5)
    kept_np = pool_np[_sequential_sample_elimination(pool_np, count, r_max, r_min)]

    points_wp, _ = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, init_factor=init_factor, seed=seed
    )
    assert_unordered_rows_equal(
        np.asarray(points_wp.numpy(), dtype=np.float64),
        kept_np.astype(np.float32).astype(np.float64),
    )
    assert pdist(kept_np).min() > 0.6 * r_max  # non-vacuity: the oracle's set is well spaced


def test_sample_surface_poisson_disk_spacing_matches_open3d(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C: the closest pair, relative to ``r_max``, against open3d's same algorithm.

    open3d's ``sample_points_poisson_disk`` is weighted sample elimination with the same
    constants, so the two sets are statistically alike though their pools differ. Over five seeds
    open3d's closest pair sits at 0.66-0.71 of ``r_max``; ordito's must reach 0.6 on every seed and
    sit within 0.05 of open3d's median. The defect this excludes (local-maxima rounds without a
    weight floor) read 0.02-0.43.
    """
    mesh_tm, mesh_wp = sphere_irregular
    count = 100
    r_max = 2.0 * math.sqrt((float(mesh_tm.area) / count) / (2.0 * math.sqrt(3.0)))
    closest_wp, closest_o3d = [], []
    for seed in range(5):
        points_wp, _ = od.sample.sample_surface_poisson_disk(
            mesh_wp.points, mesh_wp.indices, count, seed=seed
        )
        closest_wp.append(pdist(points_wp.numpy()).min() / r_max)
        o3d.utility.random.seed(seed)
        cloud_o3d = trimesh_to_open3d(mesh_tm).sample_points_poisson_disk(number_of_points=count)
        closest_o3d.append(pdist(np.asarray(cloud_o3d.points)).min() / r_max)
    assert min(closest_o3d) > 0.6  # the reference's own spacing, so the comparison means something
    assert min(closest_wp) > 0.6
    assert abs(np.median(closest_wp) - np.median(closest_o3d)) < 0.05


def test_sample_surface_poisson_disk_high_init_factor(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """Exercise the GPU top-k branch when local maxima exceed excess."""
    _, mesh_wp = sphere_irregular
    count = 20
    pts, fids = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, init_factor=20.0, seed=3
    )
    assert pts.shape == (count,)
    assert fids.shape == (count,)


def _two_patch_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Build a 10x10 patch and a 1x1 patch 60 apart: one component 100x the area of the other."""

    def patch(x0: float, s: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.array(
            [[x0 + s * i / n, s * j / n, 0.0] for i in range(n + 1) for j in range(n + 1)],
            dtype=np.float32,
        )
        faces = np.array(
            [
                index
                for i in range(n)
                for j in range(n)
                for index in (
                    i * (n + 1) + j,
                    i * (n + 1) + j + 1,
                    (i + 1) * (n + 1) + j,
                    i * (n + 1) + j + 1,
                    (i + 1) * (n + 1) + j + 1,
                    (i + 1) * (n + 1) + j,
                )
            ],
            dtype=np.int32,
        )
        return vertices, faces

    near_vertices, near_faces = patch(0.0, 10.0, 4)
    far_vertices, far_faces = patch(60.0, 1.0, 2)
    return (
        np.vstack([near_vertices, far_vertices]),
        np.concatenate([near_faces, far_faces + len(near_vertices)]),
    )


def test_sample_surface_poisson_disk_keeps_an_isolated_component(device: str):
    """
    Class C (per-component share): ordito against ``pymeshlab``'s Poisson-disk sampler.

    Excludes the bug class "elimination drops a whole component". The share is the only
    comparable statistic -- MeshLab's filter is parametrized by radius and returns whatever count
    that yields (132 here for a requested 200), so the counts do not correspond -- but it is the
    statistic the defect moved, and it moved it to exactly zero.

    Weighted sample elimination deletes each round's local weight maxima. A point with no alive
    neighbour inside ``r_max`` has weight 0, the *least* crowded state there is, and was flagged a
    maximum vacuously; and because ``_poisson_edge_weight`` clamps distances below ``r_min``, a
    cluster of equally crowded points tied exactly and was flagged entire. Together those wiped the
    small patch on every seed measured, with the returned count still exactly ``count`` -- so
    nothing about the result's shape or spacing could show it.

    Margin: the far component's area share is 1/101. MeshLab measures 0.0115-0.0152 and ordito
    0.0050-0.0150 per seed, mean 0.0100; the band below admits all of those with ~2.5x headroom
    either way and excludes 0 outright.
    """
    vertices_np, faces_np = _two_patch_mesh()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    count = 200
    area_share = 1.0 / 101.0

    mesh_set = ml.MeshSet()
    mesh_set.add_mesh(ml.Mesh(vertices_np.astype(np.float64), faces_np.reshape(-1, 3)))
    radius = 2.0 * math.sqrt((101.0 / count) / (2.0 * math.sqrt(3.0)))
    mesh_set.generate_sampling_poisson_disk(radius=ml.PureValue(radius), samplenum=count)
    points_ml = mesh_set.current_mesh().vertex_matrix()
    far_ml = int((points_ml[:, 0] > 30.0).sum())
    # The oracle has to find the component before its share can mean anything.
    assert far_ml > 0
    share_ml = far_ml / len(points_ml)

    far_counts = []
    for seed in range(5):
        points_wp, _ = od.sample.sample_surface_poisson_disk(
            vertices_wp, faces_wp, count, seed=seed
        )
        far_counts.append(int((points_wp.numpy()[:, 0] > 30.0).sum()))
    # Never dropped -- this is the assertion the defect failed, at every seed.
    assert min(far_counts) >= 1

    share_od = float(np.mean(far_counts)) / count
    assert 0.4 * area_share <= share_od <= 2.5 * area_share
    assert 0.4 * share_ml <= share_od <= 2.5 * share_ml


def test_find_local_maxima_flags_an_independent_set(device: str):
    """
    Not a library comparison: no reference exposes one elimination round, only its end result.

    The round deletes every flagged point at once, which is only sound if no two flagged points
    are neighbours. Maximality on the weight alone does not give that when weights tie, and ties
    are systematic rather than rare -- ``_poisson_edge_weight`` clamps any distance below
    ``r_min`` up to it, so a point's weight is exactly its close-neighbour count times one
    constant. Ordering on ``(weight, -index)`` restores it. Asserted on a clique of four mutually
    adjacent points at one weight, plus a neighbourless point, which also pins that a point with
    no alive neighbour is never flagged.

    Excludes: a whole tied cluster deleted in one round, and the least crowded point deleted
    first. It does not check *which* member of a tie survives, which the algorithm does not define
    beyond being reproducible. A second launch pins the round's weight floor: with it above the
    clique's weight nothing is flagged, maximal or not.
    """
    # Points 0-3 form a clique at one weight; point 4 is alive with no alive neighbour.
    neighbour_rows = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [4]]
    weights_np = np.array([2.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float32)
    indices_np = np.array([i for row in neighbour_rows for i in row], dtype=np.int32)
    offsets_np = np.cumsum([0] + [len(row) for row in neighbour_rows]).astype(np.int32)

    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    offsets_wp = wp.array(offsets_np, dtype=wp.int32, device=device)
    alive_wp = wp.ones(5, dtype=wp.int32, device=device)
    is_max_wp = wp.zeros(5, dtype=wp.int32, device=device)
    count_wp = wp.zeros(1, dtype=wp.int32, device=device)
    no_floor_wp = wp.full(1, np.inf, dtype=wp.float32, device=device)  # minus the floor
    wp.launch(
        kernel_sample.find_local_maxima,
        dim=5,
        inputs=[weights_wp, alive_wp, indices_wp, offsets_wp, no_floor_wp, is_max_wp, count_wp],
        device=device,
    )
    is_max_np = is_max_wp.numpy()

    # Exactly one of the tied clique, and never the neighbourless point.
    assert int(is_max_np[:4].sum()) == 1
    assert int(is_max_np[4]) == 0
    # The count the elimination loop reads back instead of reducing the mask agrees with it.
    assert int(count_wp.numpy()[0]) == int(is_max_np.sum())
    # The general property, stated over the graph rather than over this fixture's shape.
    flagged = set(np.flatnonzero(is_max_np).tolist())
    for i in flagged:
        assert flagged.isdisjoint(set(neighbour_rows[i]) - {i})

    floor_wp = wp.full(1, -3.0, dtype=wp.float32, device=device)  # a floor of 3, above every weight
    floored_wp = wp.zeros(5, dtype=wp.int32, device=device)
    wp.launch(
        kernel_sample.find_local_maxima,
        dim=5,
        inputs=[weights_wp, alive_wp, indices_wp, offsets_wp, floor_wp, floored_wp, count_wp],
        device=device,
    )
    assert not floored_wp.numpy().any()


def _blue_noise_radius_for_count(surface_area: float, n: int) -> float:
    return math.sqrt((surface_area * 0.5 / (n * 0.6162910373)) / math.pi)


def test_sample_surface_blue_noise_min_distance(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = sphere_irregular
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 80)
    points, _ = od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=2)
    points_np = points.numpy().reshape(-1, 3)
    if points_np.shape[0] >= 2:
        min_dist = float(pdist(points_np).min())
        assert min_dist >= radius * 0.99


def test_sample_surface_blue_noise_count_and_on_surface(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
):
    """The count is within 1.5x of igl's estimate for the radius, every sample on the surface."""
    mesh_tm, mesh_wp = sphere_irregular
    surface_area = float(mesh_tm.area)
    expected = 50
    radius = _blue_noise_radius_for_count(surface_area, expected)
    points, _ = od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=0)
    n = points.size
    igl_expected = (
        surface_area * (math.pi * math.sqrt(3.0) / 6.0) / (math.pi * radius * radius / 4.0)
    )
    assert 0.5 * igl_expected <= n <= 1.5 * igl_expected
    _, dists, _ = tm.proximity.closest_point(mesh_tm, points.numpy())
    assert np.all(dists < 1e-4)


def _blue_noise_statistics(
    points_np: npt.ArrayLike, radius: float, dense_np: np.ndarray
) -> tuple[float, float]:
    """
    Return the two radius-relative quantities comparable across blue-noise samplers.

    Returns the closest pair as a multiple of ``radius`` (the Poisson-disk property itself) and the
    worst uncovered gap as a multiple of ``radius`` (how space-filling the set is, measured against
    a dense uniform sample of the same surface).
    """
    points_np = np.asarray(points_np, dtype=np.float64).reshape(-1, 3)
    return (
        float(pdist(points_np).min()) / radius,
        float(np.max(cKDTree(points_np).query(dense_np)[0])) / radius,
    )


@pytest.mark.parity("blue_noise", "open3d", "pymeshlab", "igl", "meshlib")
def test_sample_surface_blue_noise_matches_open3d_pymeshlab_and_igl(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C: five blue-noise samplers, five algorithms, no correspondence between the point sets.

    ordito reduces a dense pool by randomized priority (flat background grid), Open3D's
    ``sample_points_poisson_disk`` runs Yuksel's sample *elimination* from a dense uniform cloud,
    MeshLab's ``generate_sampling_poisson_disk`` is Corsini's hierarchical dart throwing, and
    ``igl.blue_noise`` is Bridson active-list dart throwing -- which is what ordito itself ran
    until the algorithm was replaced, and whose ``30x`` pool oversampling ordito still uses; and
    **meshlib's ``pointUniformSampling`` subsamples a point cloud** rather than a surface, so it is
    the one reference here that has to be given ordito's own dense pool as its input -- which makes
    its row the cleanest algorithm-against-algorithm reading of the five, since the pool is
    identical.
    Nothing about the individual samples is shared -- not their count, not their positions, not even
    their number given the same parameter -- so the comparison is on the properties all four claim.
    MeshLab and igl both accept a *radius* (``radius=PureValue(r)`` overrides ``samplenum``; igl's
    third argument is ``r``), so they get ordito's own parameter; Open3D takes a count and gets
    ordito's output count, exactly as the benchmark parametrizes them.

    **Bug class excluded:** a sampler that is not blue noise at all (assert 1) and one that is blue
    noise over only part of the surface (assert 2). Both are live failure modes for a
    grid-based dart thrower -- a mis-sized background cell rejects too little, a mis-mapped cell-to-
    face seeding covers too little -- and neither is visible to
    ``test_sample_surface_blue_noise_min_distance``, which tests ordito against itself.

    **Measured, with both mutation probes**, on ``sphere_irregular`` at a radius sized for 300
    samples (ordito draws 704). Two degenerate stand-ins are scored alongside: a uniform Monte-Carlo
    cloud of the *same size* (blue noise's null hypothesis) and one confined to the fifth of the
    faces nearest one face.

    | | closest pair / r | worst gap / r | samples |
    |---|---|---|---|
    | ordito | 1.001 | 1.094 | 704 |
    | igl, same radius | 1.000 | 1.045 | 725 |
    | MeshLab, same radius | 1.000 | 1.147 | 749 |
    | meshlib, same radius and pool | 1.000 | **1.000** | 897 |
    | Open3D, same count | 0.933 | 1.304 | 704 |
    | uniform Monte Carlo | **0.029** | 1.943 | 704 |
    | one patch | **0.008** | **14.615** | 704 |

    So assert 1 (``>= 0.85``) clears the worst reference by 9 % and both probes by 29x -- it is the
    assert carrying the bug class. Assert 2 (worst gap ``<= 1.4``) holds for all five and separates
    the clumped probe by 10x but the Monte-Carlo one by only 1.39x. (A "every face hit" count is no
    coverage measure here: 704 samples cannot touch all 996 faces of a mesh whose face areas span
    6 129x; on a 20-face icosahedron it was the third assert.)

    igl is the **closest of the four in output**: 725 samples against ordito's 704 (3 % apart,
    against MeshLab's 749) and the tightest coverage of the radius-driven samplers at 1.045 r.

    ordito's ``1.000`` in the first column is **exact rather than tolerant**, and is a property of
    the algorithm rather than of this fixture: an accepted point is never within ``r`` of another
    accepted one, because the later of any such pair would already have been discarded by the
    earlier one's ball. The two ``0.005`` probe rows are what the column looks like without that.
    """
    mesh_tm, mesh_wp = sphere_irregular
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 300)
    dense_np, _face_index = tm.sample.sample_surface(mesh_tm, 20_000, seed=3)[:2]

    points_wp, _face_index_wp = od.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=11
    )
    n_samples = points_wp.size

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.generate_sampling_poisson_disk(radius=ml.PureValue(radius))
    points_pml = np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)

    points_o3d = np.asarray(
        trimesh_to_open3d(mesh_tm).sample_points_poisson_disk(number_of_points=n_samples).points
    )

    points_igl = igl.blue_noise(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
        radius,
    )[2]

    # meshlib thins a *cloud*, so it gets the same dense pool the statistics are measured against.
    cloud_ml = points_to_meshlib(np.ascontiguousarray(dense_np, dtype=np.float64))
    settings_ml = mm.UniformSamplingSettings()
    settings_ml.distance = radius
    points_ml = dense_np[
        meshlib_bitset_to_numpy(mm.pointUniformSampling(cloud_ml, settings_ml), dense_np.shape[0])
    ]

    closest_wp, gap_wp = _blue_noise_statistics(points_wp.numpy(), radius, dense_np)
    closest_pml, gap_pml = _blue_noise_statistics(points_pml, radius, dense_np)
    closest_o3d, gap_o3d = _blue_noise_statistics(points_o3d, radius, dense_np)
    closest_igl, gap_igl = _blue_noise_statistics(points_igl, radius, dense_np)
    closest_ml, gap_ml = _blue_noise_statistics(points_ml, radius, dense_np)

    # 1. The Poisson-disk property, on all five.
    assert min(closest_wp, closest_pml, closest_o3d, closest_igl, closest_ml) >= 0.85
    # 2. Space-filling, on all five.
    assert max(gap_wp, gap_pml, gap_o3d, gap_igl, gap_ml) <= 1.4
    # 3. And the radius parametrization agrees: the same radius yields the same order of samples.
    assert 0.8 <= n_samples / points_pml.shape[0] <= 1.25
    # meshlib keeps more of the pool at the same radius (measured 897 against 704, a ratio of
    # 0.78), which is a maximal-set tie-breaking difference rather than a different radius.
    assert 0.7 <= n_samples / points_ml.shape[0] <= 1.3


def test_sample_surface_blue_noise_radius_invalid(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = sphere_irregular
    with pytest.raises(ValueError, match="radius"):
        od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, 0.0)


@pytest.mark.parity(
    "sample_volume",
    "trimesh",
    benchmarked=False,
    reason="trimesh.sample.volume_mesh is rejection sampling against a ray-parity containment "
    "test, so it returns a variable number of points for a requested count and its cost is the "
    "mesh's fill ratio rather than the count -- timing it against an exact fan decomposition "
    "would compare a stochastic method with a deterministic one. What is comparable is its "
    "containment predicate, which is what this test uses it for.",
)
def test_sample_volume_containment(sphere_well_shaped: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (containment): every sample must be inside, with trimesh as the inside/outside oracle.

    Stochastic output with no correspondence, so containment is the strongest exact statement
    available; the distribution is [`test_sample_volume_uniform`].

    Premise: a solid star-shaped about its centroid, the input ``sample_volume`` accepts.
    ``sphere_well_shaped`` is bumpy and irregular but every centroid tetrahedron is positive (the
    smallest +0.003); ``sphere_irregular`` has two inverted ones and is refused
    (``test_sample_volume_rejects``).
    """
    mesh_tm, mesh_wp = sphere_well_shaped
    count = 5_000
    points_np = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, count, seed=42).numpy()
    assert points_np.shape == (count, 3)
    assert mesh_tm.contains(points_np).all()


def test_sample_volume_uniform(sphere_well_shaped: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (a moment): the sample mean must approach the centre of mass.

    The threshold is derived rather than chosen: at 20 000 samples the per-axis standard error
    is ~0.006, so ``atol=0.05`` is ~8 sigma (measured offset 0.011) -- clear of noise and still
    tight enough to catch a distribution biased toward one side of the solid. Fixture premise as in
    ``test_sample_volume_containment``.
    """
    mesh_tm, mesh_wp = sphere_well_shaped
    points_np = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 20_000, seed=0).numpy()
    assert np.allclose(points_np.mean(axis=0), mesh_tm.center_mass, atol=0.05)


@pytest.mark.parametrize(
    ("fixture_name", "match"),
    [
        pytest.param("saddle_graded", "watertight", id="not_watertight"),
        pytest.param(None, "zero volume", id="zero_volume"),
        pytest.param("torus_irregular", "star-shaped", id="not_star_shaped"),
        pytest.param("sphere_irregular", "star-shaped", id="not_star_shaped_bumpy"),
    ],
)
def test_sample_volume_rejects(
    request: pytest.FixtureRequest, device: str, fixture_name: str | None, match: str
) -> None:
    """
    The three inputs a centroid fan cannot sample, each refused by name.

    * ``saddle_graded`` is open, so it fails the watertight gate.
    * A doubled triangle (the same face with both windings) is edge-manifold with no boundary
      edges, so it passes the watertight gate, yet it encloses nothing: the surface centroid is
      coplanar with both faces, so every fanned tetrahedron has a signed volume of exactly 0.0.
    * ``torus_irregular`` is watertight with positive total volume, but fanning tetrahedra from the
      centroid (the hole of the torus) makes the inner half of the tube contribute negative signed
      volumes (336 of its 1 000).
    * ``sphere_irregular`` is a genus-0 near-sphere and still not star-shaped about its centroid:
      two of its 996 fanned tetrahedra are inverted (the smallest -0.0075), where its bumps fold
      over the line of sight. Refused, not sampled with a hole in the distribution.
    """
    if fixture_name is None:
        vertices_wp = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            dtype=wp.vec3,
            device=device,
        )
        faces_wp = wp.array([0, 1, 2, 0, 2, 1], dtype=wp.int32, device=device)
    else:
        _, mesh_wp = request.getfixturevalue(fixture_name)
        vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    with pytest.raises(ValueError, match=match):
        od.sample.sample_volume(vertices_wp, faces_wp, 100)


@pytest.mark.parametrize("sampler", ["fibonacci_sphere", "poisson_disk", "blue_noise", "volume"])
def test_samplers_are_deterministic(request: pytest.FixtureRequest, sampler: str) -> None:
    """
    Ordito against ordito: two calls with the same arguments and seed return identical buffers.

    The lattice takes no seed, so its claim is plain reproducibility; the three seeded samplers
    are called at ``seed=7`` twice, and every returned buffer (points and, where returned, face
    indices) is compared bit for bit. ``sample_volume`` takes ``sphere_well_shaped``, the solid it
    accepts (``test_sample_volume_containment``).
    """
    _, mesh_wp = request.getfixturevalue(
        "sphere_well_shaped" if sampler == "volume" else "sphere_irregular"
    )

    def draw() -> tuple[wp.array[Any], ...]:
        if sampler == "fibonacci_sphere":
            return (od.sample.sample_fibonacci_sphere(500, device=mesh_wp.points.device),)
        if sampler == "poisson_disk":
            return od.sample.sample_surface_poisson_disk(
                mesh_wp.points, mesh_wp.indices, 80, seed=7
            )
        if sampler == "blue_noise":
            return od.sample.sample_surface_blue_noise(
                mesh_wp.points, mesh_wp.indices, _blue_noise_radius_for_count(1.0, 30), seed=7
            )
        return (od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 200, seed=7),)

    first, second = draw(), draw()
    for array_a, array_b in zip(first, second, strict=True):
        assert np.array_equal(array_a.numpy(), array_b.numpy())


@pytest.mark.parametrize(
    "sampler",
    [
        "fibonacci_sphere",
        "fibonacci_hemisphere",
        "fibonacci_cone",
        "poisson_disk",
        "blue_noise_no_faces",
        "volume",
    ],
)
def test_samplers_return_empty_buffers_for_an_empty_request(
    request: pytest.FixtureRequest, sampler: str
) -> None:
    """A zero count (or, for blue noise, a face-less mesh) returns every buffer at length zero."""
    _, mesh_wp = request.getfixturevalue(
        "sphere_well_shaped" if sampler == "volume" else "sphere_irregular"
    )
    device = mesh_wp.points.device
    if sampler == "fibonacci_sphere":
        buffers = (od.sample.sample_fibonacci_sphere(0, device=device),)
    elif sampler == "fibonacci_hemisphere":
        buffers = (od.sample.sample_fibonacci_hemisphere(0, device=device),)
    elif sampler == "fibonacci_cone":
        buffers = (od.sample.sample_fibonacci_cone(0, 1.0, device=device),)
    elif sampler == "poisson_disk":
        buffers = od.sample.sample_surface_poisson_disk(mesh_wp.points, mesh_wp.indices, 0)
    elif sampler == "blue_noise_no_faces":
        empty_faces = warp_empty(0, wp.int32, device)
        buffers = od.sample.sample_surface_blue_noise(mesh_wp.points, empty_faces, 0.1, seed=0)
    else:
        buffers = (od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 0),)
    for buffer in buffers:
        assert buffer.shape == (0,)


def test_resolve_seed_passes_a_seed_through_and_draws_one_otherwise() -> None:
    """
    The one seed convention every generator in the package shares, including across modules.

    ``None`` has to mean *draw one*, not *use zero* -- a silent zero would make every unseeded call
    in the package return the same sample set. Two consecutive draws are asserted distinct, which is
    what separates the two readings.
    """
    assert od.sample.resolve_seed(1234) == 1234
    assert od.sample.resolve_seed(0) == 0

    drawn = [od.sample.resolve_seed(None) for _ in range(8)]
    assert all(0 <= seed < 2**31 for seed in drawn)
    assert all(isinstance(seed, int) for seed in drawn)
    # Eight identical draws would be a fixed default wearing a random one's signature.
    assert len(set(drawn)) > 1
