"""Regression tests for ``ordito.sample`` vs ``trimesh.sample`` (CPU reference)."""

from __future__ import annotations

import math

import igl
import numpy as np
import numpy.typing as npt
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
from tests.comparisons import chamfer_two_sided
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_warp,
    points_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    warp_empty,
)


def test_sample_fibonacci_sphere_unit(device: str):
    directions_wp = od.sample.sample_fibonacci_sphere(1000, device=device)
    directions_np = directions_wp.numpy()
    assert directions_np.shape == (1000, 3)
    norms = np.linalg.norm(directions_np, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5)


def test_sample_fibonacci_sphere_uniform(device: str):
    # A near-uniform covering of the sphere has its centroid essentially at the origin.
    directions_np = od.sample.sample_fibonacci_sphere(4096, device=device).numpy()
    assert np.allclose(directions_np.mean(axis=0), 0.0, atol=1e-2)


def test_sample_fibonacci_sphere_deterministic(device: str):
    directions_a = od.sample.sample_fibonacci_sphere(500, device=device).numpy()
    directions_b = od.sample.sample_fibonacci_sphere(500, device=device).numpy()
    assert np.array_equal(directions_a, directions_b)


def test_sample_fibonacci_sphere_empty(device: str):
    directions_wp = od.sample.sample_fibonacci_sphere(0, device=device)
    assert directions_wp.shape == (0,)


def test_sample_fibonacci_hemisphere_positive_z(device: str):
    directions_np = od.sample.sample_fibonacci_hemisphere(1000, device=device).numpy()
    assert directions_np.shape == (1000, 3)
    assert np.all(directions_np[:, 2] > 0.0)
    norms = np.linalg.norm(directions_np, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5)


def test_sample_fibonacci_hemisphere_empty(device: str):
    directions_wp = od.sample.sample_fibonacci_hemisphere(0, device=device)
    assert directions_wp.shape == (0,)


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
    assert od.sample.sample_fibonacci_cone(0, 1.0, device=device).shape == (0,)


@pytest.mark.parity("sample_surface", "trimesh", "igl")
def test_sample_surface(half_torus: tuple[tm.Trimesh, wp.Mesh]):
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
    mesh_tm, mesh_wp = half_torus
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
def test_sample_surface_matches_pytorch3d(icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C: a distributional bound, because two independent RNGs give no correspondence at all.

    The statistic is the two-sided Chamfer distance between the two 1 000-point sets, and it is
    bounded rather than matched. **Mutation probe and margin:** sampling the reference from a mesh
    scaled by 1.15 takes the same statistic from **0.00768** to **0.05227**, a factor of **6.8** --
    clear of section 6's 3x bar, and the threshold below sits between the two.

    **Bug class excluded:** a sampler that lands off the surface, concentrates on the wrong faces,
    or ignores its area weights -- all three move the point cloud far enough to blow the bound. Not
    excluded: any defect that preserves the surface distribution, e.g. a correlated RNG. The
    per-face proportionality assert covers the weighting half directly, on a deliberately
    anisotropic mesh where the face areas span 4.2x (measured correlation 0.90 at 20 000 samples;
    on a regular icosphere the areas span 1.05x and the correlation is noise, which is why the
    invariant needs its own fixture rather than riding on this one).
    """
    mesh_tm, mesh_wp = icosphere_coarse
    torch.manual_seed(4)
    samples_p3d = p3d_ops.sample_points_from_meshes(trimesh_to_pytorch3d(mesh_tm), 1000)[0].numpy()
    samples_wp, _ = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, 1000, seed=4)

    assert samples_p3d.shape == (1000, 3)
    assert float(np.abs(np.linalg.norm(samples_p3d, axis=1) - 1.0).max()) < 0.05
    assert chamfer_two_sided(samples_wp.numpy(), samples_p3d) < 0.02

    # The mutation probe, run rather than merely recorded: a 1.15x mesh must fail that bound.
    scaled_tm = mesh_tm.copy()
    scaled_tm.apply_scale(1.15)
    torch.manual_seed(4)
    scaled_p3d = p3d_ops.sample_points_from_meshes(trimesh_to_pytorch3d(scaled_tm), 1000)[0].numpy()
    assert chamfer_two_sided(samples_wp.numpy(), scaled_p3d) > 0.02

    # Area weighting, on a mesh whose face areas actually differ.
    stretched_tm = mesh_tm.copy()
    stretched_tm.apply_scale([1.0, 1.0, 4.0])
    vertices_wp, faces_wp = numpy_to_warp(
        stretched_tm.vertices, stretched_tm.faces, str(mesh_wp.points.device)
    )
    _, face_indices_wp = od.sample.sample_surface(vertices_wp, faces_wp, 20000, seed=1)
    areas_np = od.triangles.face_normals_and_areas(vertices_wp, faces_wp)[1].numpy()
    counts_np = np.bincount(face_indices_wp.numpy(), minlength=len(stretched_tm.faces))
    assert areas_np.max() / areas_np.min() > 4.0
    assert float(np.corrcoef(counts_np, areas_np)[0, 1]) > 0.8


def test_sample_surface_with_face_weights(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (a frequency comparison): two RNGs cannot produce the same samples.

    What is comparable is the *distribution* of chosen faces, so the per-face frequency is
    compared against trimesh's under the same weights -- a linear ramp, so an implementation
    ignoring weights gives a flat histogram and fails clearly rather than marginally.
    """
    mesh_tm, mesh_wp = icosahedron
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


def test_sample_surface_poisson_disk_count(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    count = 100
    pts, fids = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, seed=0
    )
    assert pts.shape == (count,)
    assert fids.shape == (count,)


def test_sample_surface_poisson_disk_on_surface(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    count = 100
    pts, _ = od.sample.sample_surface_poisson_disk(mesh_wp.points, mesh_wp.indices, count, seed=1)
    _, dists, _ = tm.proximity.closest_point(mesh_tm, pts.numpy())
    assert np.all(dists < 1e-4)


def test_sample_surface_poisson_disk_min_distance(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
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


def test_sample_surface_poisson_disk_deterministic(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points_a, face_indices_a = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, 80, seed=7
    )
    points_b, face_indices_b = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, 80, seed=7
    )
    assert np.array_equal(points_a.numpy(), points_b.numpy())
    assert np.array_equal(face_indices_a.numpy(), face_indices_b.numpy())


def test_sample_surface_poisson_disk_count_zero(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points, face_indices = od.sample.sample_surface_poisson_disk(mesh_wp.points, mesh_wp.indices, 0)
    assert points.shape == (0,)
    assert face_indices.shape == (0,)


def test_sample_surface_poisson_disk_high_init_factor(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """Exercise the GPU top-k branch when local maxima exceed excess."""
    _, mesh_wp = icosahedron
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
    beyond being reproducible.
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
    wp.launch(
        kernel_sample.find_local_maxima,
        dim=5,
        inputs=[weights_wp, alive_wp, indices_wp, offsets_wp, is_max_wp, count_wp],
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


def test_sample_fibonacci_sphere_phase_stays_low_discrepancy(device: str):
    """
    Class A against a float64 NumPy evaluation of the lattice's own closed form.

    The spiral phase is ``count`` multiples of the golden angle, so it grows without bound in
    ``count``; accumulating it in float32 lost the low digits the low-discrepancy property lives
    in. Measured before the fix: azimuth error 1.7e-04 rad at 1 024 rising to 0.21 rad at 1e6, and
    a minimum neighbour spacing 11 % below the reference's at 100 000. ``count = 100_000`` is
    asserted because that is where the loss is unambiguous and the test is still cheap; the same
    hazard and the same float64 fix are documented at
    ``kernels/bounds.oriented_box_candidate_axes``.
    """
    count = 100_000
    directions_np = od.sample.sample_fibonacci_sphere(count, device=device).numpy()

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


def _blue_noise_radius_for_count(surface_area: float, n: int) -> float:
    return math.sqrt((surface_area * 0.5 / (n * 0.6162910373)) / math.pi)


def test_sample_surface_blue_noise_min_distance(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 80)
    points, _ = od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=2)
    points_np = points.numpy().reshape(-1, 3)
    if points_np.shape[0] >= 2:
        min_dist = float(pdist(points_np).min())
        assert min_dist >= radius * 0.99


def test_sample_surface_blue_noise_on_surface(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 50)
    pts, _ = od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=1)
    _, dists, _ = tm.proximity.closest_point(mesh_tm, pts.numpy())
    assert np.all(dists < 1e-4)


def test_sample_surface_blue_noise_deterministic(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(1.0, 30)
    points_a, face_indices_a = od.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=7
    )
    points_b, face_indices_b = od.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=7
    )
    assert np.array_equal(points_a.numpy(), points_b.numpy())
    assert np.array_equal(face_indices_a.numpy(), face_indices_b.numpy())


def test_sample_surface_blue_noise_count_order_of_magnitude(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
):
    mesh_tm, mesh_wp = icosahedron
    surface_area = float(mesh_tm.area)
    expected = 50
    radius = _blue_noise_radius_for_count(surface_area, expected)
    points, _ = od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=0)
    n = points.size
    igl_expected = (
        surface_area * (math.pi * math.sqrt(3.0) / 6.0) / (math.pi * radius * radius / 4.0)
    )
    assert 0.5 * igl_expected <= n <= 1.5 * igl_expected


def _blue_noise_statistics(
    mesh_tm: tm.Trimesh, points_np: npt.ArrayLike, radius: float, dense_np: np.ndarray
) -> tuple[float, float, int]:
    """
    Return the three radius-relative quantities comparable across blue-noise samplers.

    Returns the closest pair as a multiple of ``radius`` (the Poisson-disk property itself), the
    worst uncovered gap as a multiple of ``radius`` (how space-filling the set is, measured against
    a dense uniform sample of the same surface), and the number of faces carrying at least one
    sample.
    """
    points_np = np.asarray(points_np, dtype=np.float64).reshape(-1, 3)
    _closest, _distance, face_index_np = tm.proximity.closest_point(mesh_tm, points_np)
    return (
        float(pdist(points_np).min()) / radius,
        float(np.max(cKDTree(points_np).query(dense_np)[0])) / radius,
        np.unique(face_index_np).size,
    )


@pytest.mark.parity("blue_noise", "open3d", "pymeshlab", "igl", "meshlib")
def test_sample_surface_blue_noise_matches_open3d_pymeshlab_and_igl(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
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
    noise over only part of the surface (asserts 2 and 3). Both are live failure modes for a
    grid-based dart thrower -- a mis-sized background cell rejects too little, a mis-mapped cell-to-
    face seeding covers too little -- and neither is visible to
    ``test_sample_surface_blue_noise_min_distance``, which tests ordito against itself.

    **Measured, with both mutation probes.** Two degenerate stand-ins are scored alongside: a
    uniform Monte-Carlo cloud of the *same size* (blue noise's null hypothesis) and one confined to
    4 of the 20 faces.

    | | closest pair / r | worst gap / r | faces hit |
    |---|---|---|---|
    | ordito | 1.000 | 1.064 | 20 / 20 |
    | igl, same radius | 1.000 | 1.073 | 20 / 20 |
    | MeshLab, same radius | 1.000 | 1.112 | 20 / 20 |
    | meshlib, same radius and pool | 1.000 | **0.982** | 20 / 20 |
    | Open3D, same count | 0.926 | 1.197 | 20 / 20 |
    | uniform Monte Carlo | **0.005** | 1.878 | 20 / 20 |
    | one patch | **0.005** | **17.781** | **4 / 20** |

    So assert 1 (``>= 0.85``) clears the worst reference by 8% and both probes by **200x** -- it is
    the assert carrying the bug class. Assert 3 (every face hit) separates the clumped probe by a
    factor of 5. Assert 2 (worst gap ``<= 1.4``) is the weakest of the three at a 1.34x margin
    over the Monte-Carlo probe, and it is applied to ordito, igl and MeshLab -- the three that
    received the identical radius. Open3D is exempt from it: its elimination sampler genuinely
    leaves larger gaps on a 20-face mesh, a property of its algorithm rather than a disagreement.

    igl is the **closest of the three in output** and the only one that beats ordito on a column:
    747 samples against ordito's 749 (0.3% apart, against MeshLab's 769) and the tightest coverage
    of the four at 1.073 r. That is worth stating next to the benchmark, where ordito is 4-24x
    faster than it -- the port is not buying its speed with quality.

    ordito's ``1.000`` in the first column is **exact rather than tolerant**, and is a property of
    the algorithm rather than of this fixture: an accepted point is never within ``r`` of another
    accepted one, because the later of any such pair would already have been discarded by the
    earlier one's ball. The two ``0.005`` probe rows are what the column looks like without that.
    """
    mesh_tm, mesh_wp = icosahedron
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

    closest_wp, gap_wp, faces_wp = _blue_noise_statistics(
        mesh_tm, points_wp.numpy(), radius, dense_np
    )
    closest_pml, gap_pml, faces_pml = _blue_noise_statistics(mesh_tm, points_pml, radius, dense_np)
    closest_o3d, _gap_o3d, faces_o3d = _blue_noise_statistics(mesh_tm, points_o3d, radius, dense_np)
    closest_igl, gap_igl, faces_igl = _blue_noise_statistics(mesh_tm, points_igl, radius, dense_np)
    closest_ml, gap_ml, faces_ml = _blue_noise_statistics(mesh_tm, points_ml, radius, dense_np)

    # 1. The Poisson-disk property, on all five.
    assert min(closest_wp, closest_pml, closest_o3d, closest_igl, closest_ml) >= 0.85
    # 2. Space-filling, on the four that received the identical radius.
    assert max(gap_wp, gap_pml, gap_igl, gap_ml) <= 1.4
    # 3. Every face reached, on all five.
    assert faces_wp == faces_pml == faces_o3d == faces_igl == faces_ml == mesh_tm.faces.shape[0]
    # 4. And the radius parametrization agrees: the same radius yields the same order of samples.
    assert 0.8 <= n_samples / points_pml.shape[0] <= 1.25
    # meshlib keeps more of the pool at the same radius (measured 930 against 740, a ratio of
    # 0.80), which is a maximal-set tie-breaking difference rather than a different radius.
    assert 0.7 <= n_samples / points_ml.shape[0] <= 1.3


def test_sample_surface_blue_noise_radius_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="radius"):
        od.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, 0.0)


def test_sample_surface_blue_noise_empty_faces(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_wp = icosahedron[1]
    empty_faces = warp_empty(0, wp.int32, mesh_wp.points.device)
    points, face_indices = od.sample.sample_surface_blue_noise(
        mesh_wp.points, empty_faces, 0.1, seed=0
    )
    assert points.shape == (0,)
    assert face_indices.shape == (0,)


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
def test_sample_volume_containment(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (containment): every sample must be inside, with trimesh as the inside/outside oracle.

    Stochastic output with no correspondence, so containment is the strongest exact statement
    available; the distribution is [`test_sample_volume_uniform`].
    """
    mesh_tm, mesh_wp = icosahedron
    count = 5_000
    points_np = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, count, seed=42).numpy()
    assert points_np.shape == (count, 3)
    assert mesh_tm.contains(points_np).all()


def test_sample_volume_uniform(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (a moment): the sample mean must approach the centre of mass.

    The threshold is derived rather than chosen: at 20 000 samples the per-axis standard error
    is ~0.003, so ``atol=0.05`` is ~16 sigma -- clear of noise and still tight enough to catch
    a distribution biased toward one side of the solid.
    """
    # With 20 000 samples the per-axis std-of-mean is ~0.003, so atol=0.05 is safe.
    mesh_tm, mesh_wp = icosahedron
    points_np = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 20_000, seed=0).numpy()
    assert np.allclose(points_np.mean(axis=0), mesh_tm.center_mass, atol=0.05)


def test_sample_volume_deterministic(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    pts_a = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 200, seed=7).numpy()
    pts_b = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 200, seed=7).numpy()
    assert np.array_equal(pts_a, pts_b)


def test_sample_volume_count_zero(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    pts = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 0)
    assert pts.shape == (0,)


def test_sample_volume_not_watertight(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = half_torus
    with pytest.raises(ValueError, match="watertight"):
        od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 100)


def test_sample_volume_zero_volume(device: str):
    # A doubled triangle (the same face with both windings) is edge-manifold with no boundary
    # edges, so it passes the watertight gate, yet it encloses nothing: the surface centroid is
    # coplanar with both faces, so every fanned tetrahedron has a signed volume of exactly 0.0.
    vertices_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array([0, 1, 2, 0, 2, 1], dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="zero volume"):
        od.sample.sample_volume(vertices_wp, faces_wp, 100)


def test_sample_volume_not_star_shaped(torus: tuple[tm.Trimesh, wp.Mesh]):
    # Watertight with positive total volume, but fanning tetrahedra from the centroid (the hole
    # of the torus) makes the inner half of the tube contribute negative signed volumes.
    _, mesh_wp = torus
    with pytest.raises(ValueError, match="star-shaped"):
        od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 100)


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
