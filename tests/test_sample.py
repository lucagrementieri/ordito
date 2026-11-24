"""Regression tests for ``ordito.sample`` vs ``trimesh.sample`` (CPU reference)."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

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
from scipy.stats import ks_2samp

import ordito as od
from ordito.kernels import sample as kernel_sample
from tests.comparisons import assert_unordered_rows_equal, chamfer_two_sided
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_warp,
    points_to_meshlib,
    points_to_warp,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    warp_empty,
)


def test_sample_fibonacci_sphere_phase_stays_low_discrepancy(device: str):
    """
    Class A against a float64 NumPy evaluation of the lattice's own closed form.

    The spiral phase is ``count`` multiples of the golden angle, so it grows without bound in
    ``count``; accumulating it in float32 lost the low digits the low-discrepancy property lives
    in. Measured before the fix: azimuth error 1.7e-04 rad at 1 024 rising to 0.21 rad at 1e6, and
    a minimum neighbour spacing 11 % below the reference's at 100 000. ``count = 100_000`` is
    asserted because that is where the loss is unambiguous and the test is still cheap; the same
    hazard and the same float64 fix are documented at
    ``kernels/bounds.oriented_box_candidate_axes``. Every direction is also unit and the set's
    centroid sits at the origin, as a near-uniform covering of the sphere requires.
    """
    count = 100_000
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


@pytest.mark.parametrize("count", [1024, 100_000])
@pytest.mark.parity("sample_fibonacci_sphere", "pymeshlab")
def test_sample_fibonacci_sphere_matches_pymeshlab(device: str, count: int) -> None:
    """
    Class B (``y`` mirrored): MeshLab's ``create_sphere_points(Fibonacci)`` is the same lattice.

    MeshLab winds its spiral the other way about ``z``, the golden angle's sign, so its points are
    ordito's with ``y`` negated, row for row: measured within **9.1e-7** at 1 024 points and
    **4.9e-6** at 100 000 (its phase is evaluated in lower precision; ordito's sits within ``1e-5``
    of the ``float64`` closed form, asserted above). Unmirrored the rows are up to 2.0 apart.
    """
    directions_np = od.sample.sample_fibonacci_sphere(count, device=device).numpy()
    meshset_pml = ml.MeshSet()
    meshset_pml.create_sphere_points(pointnum=count, spheregentech="Fibonacci")
    points_pml = np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)

    assert points_pml.shape == (count, 3)
    assert np.abs(directions_np - points_pml).max() > 1.0  # non-vacuity: the mirror is needed
    mirrored_np = directions_np * np.array([1.0, -1.0, 1.0])
    assert np.allclose(mirrored_np, points_pml, rtol=1e-5, atol=1e-5)


def test_sample_fibonacci_cone(device: str) -> None:
    """
    Every direction inside the cone, and both ends of the range match sphere/hemisphere.

    The hemisphere is itself checked as one: every direction unit and strictly above ``z = 0``.
    """
    half_angle = np.deg2rad(25.0)
    directions_np = od.sample.sample_fibonacci_cone(512, half_angle, device=device).numpy()
    assert np.allclose(np.linalg.norm(directions_np, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    polar_np = np.arccos(np.clip(directions_np[:, 2], -1.0, 1.0))
    assert polar_np.max() <= half_angle + 1e-6
    # Uniform in solid angle means uniform in z, so the mean z is the midpoint of the band.
    assert np.isclose(directions_np[:, 2].mean(), 0.5 * (1.0 + np.cos(half_angle)), atol=1e-3)

    hemisphere_np = od.sample.sample_fibonacci_hemisphere(1000, device=device).numpy()
    assert hemisphere_np.shape == (1000, 3)
    assert np.all(hemisphere_np[:, 2] > 0.0)
    assert np.allclose(np.linalg.norm(hemisphere_np, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(
        od.sample.sample_fibonacci_cone(1000, np.pi / 2.0, device=device).numpy(),
        hemisphere_np,
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        od.sample.sample_fibonacci_cone(64, np.pi, device=device).numpy(),
        od.sample.sample_fibonacci_sphere(64, device=device).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("sample_surface", "trimesh", "igl")
def test_sample_surface(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class C, three samplers against the area law they all claim: sample share per area decile.

    Two seeded RNGs cannot be aligned, so the comparable quantity is the *distribution* rather than
    the samples. All three return the face index directly; the faces are split into ten bins by
    area rank ([`_quantile_bin_shares`]) and each library's share of samples per bin must match the
    bin's share of the area. Per-face frequencies were compared before and had no power:
    ``saddle_graded`` has 2 178 faces, so 10 000 samples put ~4.6 on a face, and a uniform-per-face
    sampler sat within the per-face noise (1.19e-03 against the samplers' own 1.0-1.3e-03).

    **Bug class excluded:** a mis-weighted CDF -- sampling uniformly per face or by face *index*
    instead of by area, the classic error in this routine and invisible to a count-and-on-surface
    check. ``saddle_graded``'s areas grade over three orders of magnitude, so the area deciles hold
    0.05 % to 32 % of the area. **Measured** worst bin error, identical on both devices: ordito
    **0.0020**, igl 0.0052, trimesh **0.0105**; bound ``0.03`` (3x trimesh). **Mutation probe:** a
    uniform-per-face sampler reads 0.221, 7x the bound.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    count = 10_000
    n_faces = int(mesh_tm.faces.shape[0])
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, seed=0)[1]

    face_idx_wp = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, count, seed=0)[1]

    face_idx_igl = igl.random_points_on_mesh(
        count,
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
        0,
    )[1]

    area_np = mesh_tm.area_faces
    expected_np = _quantile_bin_shares(np.arange(n_faces), area_np, area_np)
    uniform_np = _quantile_bin_shares(np.arange(n_faces), area_np)
    assert np.abs(uniform_np - expected_np).max() > 0.2  # non-vacuity: the deciles differ in area
    for face_idx in (face_idx_wp.numpy(), np.asarray(face_idx_tm), np.asarray(face_idx_igl)):
        assert np.abs(_quantile_bin_shares(face_idx, area_np) - expected_np).max() < 0.03


@pytest.mark.parity("sample_surface", "open3d")
def test_sample_surface_matches_open3d(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C, the area-decile statistic of [`test_sample_surface`] applied to Open3D's sampler.

    ``sample_points_uniformly`` returns bare points, so each sample's face is recovered by a
    closest-point query (the named transform; every sample lies on the surface, measured
    2.2e-15 off it). With the faces split into ten bins by area rank, each sampler's share of
    samples per bin must match the bin's share of the area. **Measured** worst bin error at 10 000
    samples: Open3D 0.0029 / 0.0054 / 0.0058 over seeds 0-2, ordito 0.0020; bound ``0.03``.
    **Bug class excluded:** a CDF weighted by anything but area. **Mutation probe**, run below: a
    sampler uniform per face reads 0.22 on ``saddle_graded``, whose deciles hold 0.05 % to 32 %
    of the area.
    """
    mesh_tm, mesh_wp = saddle_graded
    count = 10_000
    n_faces = int(mesh_tm.faces.shape[0])
    o3d.utility.random.seed(0)
    samples_o3d = np.asarray(trimesh_to_open3d(mesh_tm).sample_points_uniformly(count).points)
    _closest_np, distance_o3d, face_idx_o3d = tm.proximity.closest_point(mesh_tm, samples_o3d)
    assert samples_o3d.shape == (count, 3)
    assert float(np.max(distance_o3d)) < 1e-6

    face_idx_wp = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, count, seed=0)[1]

    area_np = mesh_tm.area_faces
    expected_np = _quantile_bin_shares(np.arange(n_faces), area_np, area_np)
    for face_idx in (face_idx_wp.numpy(), np.asarray(face_idx_o3d)):
        assert np.abs(_quantile_bin_shares(face_idx, area_np) - expected_np).max() < 0.03

    uniform_np = np.random.default_rng(0).integers(0, n_faces, size=count)
    assert np.abs(_quantile_bin_shares(uniform_np, area_np) - expected_np).max() > 0.03


def _quantile_bin_shares(
    face_index_np: np.ndarray, key_np: np.ndarray, weight_np: np.ndarray | None = None
) -> np.ndarray:
    """
    Return each of ten bins' share of ``face_index_np``, faces binned by the rank of ``key_np``.

    With ``weight_np`` the share is weighted per entry by the face's weight instead of counted, so
    ``_quantile_bin_shares(np.arange(n), key, weight)`` is each bin's share of the total weight.
    """
    n_bins = 10
    order_np = np.argsort(key_np, kind="stable")
    bin_np = np.empty(key_np.shape[0], dtype=np.int64)
    bin_np[order_np] = np.arange(key_np.shape[0]) * n_bins // key_np.shape[0]
    weights_np = None if weight_np is None else weight_np[face_index_np]
    shares_np = np.bincount(bin_np[face_index_np], weights=weights_np, minlength=n_bins)
    return shares_np / shares_np.sum()


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
    regular icosphere the areas spanned 1.05x and the correlation was noise), bounded at ``0.955``,
    3x the measured shortfall from 1.
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
    assert float(np.corrcoef(counts_np, areas_np)[0, 1]) > 0.955


def test_sample_surface_with_face_weights(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (a frequency comparison): two RNGs cannot produce the same samples.

    What is comparable is the *distribution* of chosen faces, so ordito's and trimesh's samples
    under the same weights -- a linear ramp over the face index -- are binned into weight deciles
    ([`_quantile_bin_shares`]) and each bin's share must match its share of the total weight. A
    per-face comparison has no power at ~10 samples a face.

    **Measured** worst bin error, identical on both devices: ordito **0.0035**, trimesh **0.0050**;
    bound ``0.015`` (3x trimesh). **Mutation probes:** ignoring the weights (sampling by area) reads
    0.080 and a flat histogram 0.090, 5.4x and 6x the bound.
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
    expected_np = _quantile_bin_shares(np.arange(n_faces), weights_np, weights_np)
    for face_idx in (face_idx_wp.numpy(), face_idx_tm):
        assert np.abs(_quantile_bin_shares(face_idx, weights_np) - expected_np).max() < 0.015


def test_sample_surface_barycentric_reproduces_the_samples(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class A against trimesh: the coordinates evaluate to the samples on their own faces.

    [`trimesh.triangles.barycentric_to_points`][] on each sample's face and coordinates gives the
    sample back; the coordinates are non-negative and sum to one, and asking for them changes no
    draw.
    """
    _, mesh_wp = sphere_irregular
    samples, face_index, barycentric = od.sample.sample_surface(
        mesh_wp.points, mesh_wp.indices, 2000, seed=4, return_barycentric=True
    )
    plain_samples, plain_face_index = od.sample.sample_surface(
        mesh_wp.points, mesh_wp.indices, 2000, seed=4
    )
    assert np.array_equal(samples.numpy(), plain_samples.numpy())
    assert np.array_equal(face_index.numpy(), plain_face_index.numpy())
    barycentric_np = barycentric.numpy()
    assert barycentric_np.min() >= 0.0
    assert np.allclose(barycentric_np.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
    triangles_np = mesh_wp.points.numpy()[
        mesh_wp.indices.numpy().reshape(-1, 3)[face_index.numpy()]
    ]
    points_tm = tm.triangles.barycentric_to_points(triangles_np, barycentric_np)
    assert np.allclose(samples.numpy(), points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        pytest.param("short_weights", "face_weight length must match", id="short_weights"),
        pytest.param("zero_weights", "total face weight must be positive", id="zero_weights"),
        pytest.param("zero_area", "mesh has zero surface area", id="zero_area"),
        pytest.param("no_faces", "mesh has no faces", id="no_faces"),
    ],
)
def test_sample_surface_rejects(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], case: str, match: str
) -> None:
    """
    Not a library comparison: the four inputs ``sample_surface`` documents it refuses.

    A ``face_weight`` one entry short, an all-zero ``face_weight`` (no CDF to invert), a mesh whose
    faces have no area (every vertex at one point) and a mesh with no faces, each raised by name
    rather than drawing from a garbage CDF. The short weights raise even at ``count == 0``, before
    the empty short-circuit.
    """
    _, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    device = vertices_wp.device
    n_faces = faces_wp.size // 3
    count = 10
    face_weight = None
    if case == "short_weights":
        face_weight = wp.ones(n_faces - 1, dtype=wp.float32, device=device)
        count = 0
    elif case == "zero_weights":
        face_weight = wp.zeros(n_faces, dtype=wp.float32, device=device)
    elif case == "zero_area":
        vertices_wp = wp.zeros_like(vertices_wp)
    else:
        faces_wp = warp_empty(0, wp.int32, device)
    with pytest.raises(ValueError, match=match):
        od.sample.sample_surface(vertices_wp, faces_wp, count, face_weight=face_weight, seed=0)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        pytest.param("zero_weights", "total face weight must be positive", id="zero_weights"),
        pytest.param("zero_area", "mesh has zero surface area", id="zero_area"),
        pytest.param("no_faces", "mesh has no faces", id="no_faces"),
    ],
)
def test_sample_surface_poisson_disk_rejects(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], case: str, match: str
) -> None:
    """
    Not a library comparison: the surfaces ``sample_surface_poisson_disk`` documents it refuses.

    An all-zero ``face_weight``, a mesh whose faces have no area and a mesh with no faces leave its
    uniform pool no distribution to draw from, so each is raised by name.
    """
    _, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    face_weight = None
    if case == "zero_weights":
        face_weight = wp.zeros(faces_wp.size // 3, dtype=wp.float32, device=vertices_wp.device)
    elif case == "zero_area":
        vertices_wp = wp.zeros_like(vertices_wp)
    else:
        faces_wp = warp_empty(0, wp.int32, vertices_wp.device)
    with pytest.raises(ValueError, match=match):
        od.sample.sample_surface_poisson_disk(
            vertices_wp, faces_wp, 10, face_weight=face_weight, seed=0
        )


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
    open3d's closest pair sits at 0.664-0.695 of ``r_max`` and ordito's at 0.668-0.693 (both
    devices alike); ordito's must reach 0.65 on every seed and its median sit within 0.02 of
    open3d's (measured 0.002 apart -- the bound is the seed-to-seed spread of a five-seed median,
    not 3x a difference that is itself noise). The defect this excludes (local-maxima rounds
    without a weight floor) read 0.02-0.43.
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
    assert min(closest_o3d) > 0.65  # the reference's own spacing, so the comparison means something
    assert min(closest_wp) > 0.65
    assert abs(np.median(closest_wp) - np.median(closest_o3d)) < 0.02


def test_sample_surface_poisson_disk_draws_only_from_weighted_faces(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: no reference Poisson sampler takes per-face weights.

    The pool is drawn by ``face_weight``, so with zero weight on every face below the equator no
    survivor may sit on one, and exactly ``count`` survive. Excludes ignoring the weights (half the
    samples would land on the zero-weight half) and drawing the pool by area instead.
    """
    mesh_tm, mesh_wp = sphere_irregular
    upper_np = mesh_tm.triangles_center[:, 2] > mesh_tm.center_mass[2]
    assert 0 < upper_np.sum() < upper_np.size
    weights_wp = wp.array(
        upper_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.points.device
    )
    count = 60
    points_wp, faces_wp = od.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, face_weight=weights_wp, seed=2
    )
    faces_np = faces_wp.numpy()
    assert faces_np.shape == (count,)
    assert upper_np[faces_np].all()
    _, dists, _ = tm.proximity.closest_point(mesh_tm, points_wp.numpy())
    assert np.all(dists < 1e-4)


def _scattered_triangles(device: str) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Build 100 right triangles of leg 0.01, spaced 1.0 apart along ``x``."""
    leg = 0.01
    triangle_np = np.array([[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [0.0, leg, 0.0]])
    vertices_np = np.concatenate([triangle_np + np.array([float(i), 0.0, 0.0]) for i in range(100)])
    faces_np = np.arange(300, dtype=np.int32).reshape(-1, 3)
    return numpy_to_warp(vertices_np, faces_np, device)


@pytest.mark.parametrize(
    ("count", "init_factor"),
    [pytest.param(2, 1.5, id="isolated_pool"), pytest.param(10, 30.0, id="crowded_pool")],
)
def test_sample_surface_poisson_disk_spreads_over_scattered_triangles(
    device: str, count: int, init_factor: float
) -> None:
    """
    Not a library comparison: no reference exposes the elimination's tie handling.

    100 tiny triangles 1.0 apart, against an ``r_max`` of a few leg lengths: two pool points share
    a neighbourhood only when they share a triangle, and every in-triangle distance is below
    ``r_min``, so the clamped weights tie exactly. ``isolated_pool`` draws three points on three
    triangles, so no point has a neighbour and no point is a local maximum: the round must rank the
    whole alive set. ``crowded_pool`` draws ~3 points a triangle; the early rounds delete the
    crowded ones and the last round again meets an alive set with no neighbours. Either way the
    answer is exactly ``count`` samples, no two on one triangle (two there are within ``r_max``,
    while a triangle with one survivor was always available).
    """
    vertices_wp, faces_wp = _scattered_triangles(device)
    _, face_index_wp = od.sample.sample_surface_poisson_disk(
        vertices_wp, faces_wp, count, init_factor=init_factor, seed=0
    )
    face_index_np = face_index_wp.numpy()
    assert face_index_np.shape == (count,)
    assert np.unique(face_index_np).size == count


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

    Margin: the far component's area share is 1/101 = 0.0099. MeshLab measures **0.0152** and
    ordito 0.010-0.020 per seed, mean **0.0150** over five seeds (identical on both devices) -- both
    above the area share by the same factor, since the small patch is mostly rim. ordito's mean
    must sit within ``[0.6, 1.6]`` of MeshLab's (measured 0.99; a far count of ~3 a seed carries
    ~30 % noise in a five-seed mean) and within ``[0.6, 2.5]`` of the area share (measured 1.52),
    and the defect's 0 fails both outright.
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
    assert 0.6 * area_share <= share_od <= 2.5 * area_share
    assert 0.6 * share_ml <= share_od <= 1.6 * share_ml


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
    # The loop state: its sixth slot is the floor as an ordered key of minus the floor weight,
    # first with no floor (minus infinity), then with a floor of 3, above every weight.
    state_wp = wp.array([0, 1, 5, 0, 0, _ordered_key(np.inf), 0], dtype=wp.int32, device=device)
    wp.launch(
        kernel_sample.find_local_maxima,
        dim=5,
        inputs=[weights_wp, alive_wp, indices_wp, offsets_wp, state_wp, is_max_wp],
        device=device,
    )
    is_max_np = is_max_wp.numpy()

    # Exactly one of the tied clique, and never the neighbourless point.
    assert int(is_max_np[:4].sum()) == 1
    assert int(is_max_np[4]) == 0
    # The count the elimination loop branches on instead of reducing the mask agrees with it.
    assert int(state_wp.numpy()[3]) == int(is_max_np.sum())
    # The general property, stated over the graph rather than over this fixture's shape.
    flagged = set(np.flatnonzero(is_max_np).tolist())
    for i in flagged:
        assert flagged.isdisjoint(set(neighbour_rows[i]) - {i})

    floor_wp = wp.array([0, 1, 5, 0, 0, _ordered_key(-3.0), 0], dtype=wp.int32, device=device)
    floored_wp = wp.zeros(5, dtype=wp.int32, device=device)
    wp.launch(
        kernel_sample.find_local_maxima,
        dim=5,
        inputs=[weights_wp, alive_wp, indices_wp, offsets_wp, floor_wp, floored_wp],
        device=device,
    )
    assert not floored_wp.numpy().any()


def _ordered_key(value: float) -> int:
    """``kernels/array.ordered_float_bits`` of ``value`` on the host, as ``int32`` bits."""
    bits = int(np.array([value], dtype=np.float32).view(np.uint32)[0])
    key = (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000
    return int(np.array([key], dtype=np.uint32).view(np.int32)[0])


def _blue_noise_radius_for_count(surface_area: float, n: int) -> float:
    return math.sqrt((surface_area * 0.5 / (n * 0.6162910373)) / math.pi)


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
    face seeding covers too little.

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
    | Open3D, same count, seeded | 0.933 | 1.180 | 704 |
    | uniform Monte Carlo | **0.029** | 1.943 | 704 |
    | one patch | **0.008** | **14.615** | 704 |

    Every number is identical on both devices. Assert 1 is per sampler: ordito's own closest pair
    is the exact property below (``>= 1 - 1e-5``), the three radius-driven references must reach
    ``0.999`` and Open3D, which is handed a count rather than a radius, ``0.85`` -- 29x above both
    probes, the assert carrying the bug class. Assert 2 bounds ordito's worst gap at ``1.2`` (the
    Monte-Carlo probe reads 1.62x that, the clumped one 12x) and every reference's at ``1.3``.
    Open3D is seeded: unseeded, its worst gap moved between 1.16 and 1.32 from run to run. (A
    "every face hit" count is no coverage measure here: 704 samples cannot touch all 996 faces of a
    mesh whose face areas span 6 129x; on a 20-face icosahedron it was the third assert.)

    Assert 3 is the radius parametrization: igl is the **closest of the four in output**, 725
    samples against ordito's 704 (ratio 0.971, bound ``[0.9, 1.1]``), MeshLab 749 (0.940, bound
    ``[0.82, 1.18]``, 3x its deviation), and the tightest coverage of the radius-driven samplers at
    1.045 r.

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

    o3d.utility.random.seed(11)
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
    assert closest_wp >= 1.0 - 1e-5
    assert min(closest_pml, closest_igl, closest_ml) >= 0.999
    assert closest_o3d >= 0.85
    # 2. Space-filling, on all five.
    assert gap_wp <= 1.2
    assert max(gap_pml, gap_o3d, gap_igl, gap_ml) <= 1.3
    # 3. And the radius parametrization agrees: the same radius yields the same order of samples.
    assert 0.9 <= n_samples / np.asarray(points_igl).shape[0] <= 1.1
    assert 0.82 <= n_samples / points_pml.shape[0] <= 1.18
    # meshlib keeps more of the pool at the same radius (measured 897 against 704, a ratio of
    # 0.78), which is a maximal-set tie-breaking difference rather than a different radius.
    assert 0.7 <= n_samples / points_ml.shape[0] <= 1.3


def _in_ball_case(mesh_tm: tm.Trimesh, ball: str) -> tuple[np.ndarray, float]:
    """
    Return the center and radius of a test ball on ``mesh_tm``.

    ``"wide"`` holds about three hundred whole faces; ``"small"`` sits on the largest face's
    centroid at 1.2 mean edges, so several faces cross it with every corner outside.
    """
    if ball == "wide":
        extent = float(mesh_tm.extents.max())
        return np.asarray(mesh_tm.vertices[0]) + 0.01 * extent, 0.25 * extent
    largest = int(np.argmax(mesh_tm.area_faces))
    return (
        np.asarray(mesh_tm.triangles_center[largest]),
        1.2 * float(mesh_tm.edges_unique_length.mean()),
    )


def _in_ball_dense(
    mesh_tm: tm.Trimesh, center_np: np.ndarray, radius: float, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return trimesh's uniform draw over the whole surface pruned to the ball, and its faces."""
    samples_tm, face_index_tm = tm.sample.sample_surface(mesh_tm, draws, seed=0)[:2]
    inside = np.linalg.norm(samples_tm - center_np, axis=1) <= radius
    return samples_tm[inside], np.asarray(face_index_tm)[inside]


def _assert_on_reported_faces_in_ball(
    mesh_tm: tm.Trimesh,
    points_np: np.ndarray,
    face_index_np: np.ndarray,
    center_np: np.ndarray,
    radius: float,
) -> None:
    """Every sample lies in the closed ball and on the face it reports."""
    assert np.linalg.norm(points_np - center_np, axis=1).max() <= radius * (1.0 + 1e-6)
    closest_tm = tm.triangles.closest_point(mesh_tm.triangles[face_index_np], points_np)
    assert np.linalg.norm(closest_tm - points_np, axis=1).max() < 1e-5


@pytest.mark.parametrize("ball", ["wide", "small"])
@pytest.mark.parity("sample_surface_in_ball", "trimesh")
def test_sample_surface_in_ball_matches_pruned_trimesh(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], ball: str
) -> None:
    """
    Class C, against trimesh's whole-surface uniform draw pruned to the ball: two distributions.

    No library samples a ball's part of a surface, but the answer is defined by the pruned
    whole-surface draw (2 M trimesh samples), so the comparison is on distribution: the
    Kolmogorov-Smirnov distance between the two samples' distances from the center, and the share
    of samples on faces crossing the ball with no corner in it.

    **Bug classes excluded:** selecting the faces by their corners (the ``"small"`` ball's three
    crossing faces hold 24 % of its area) and a mis-weighted draw over the selected faces. Both
    pass every per-sample check below. **Measured**, identical on both devices, ``count = 50 000``:
    KS 0.0059 (``"wide"``) and 0.0047 (``"small"``), bound 0.03; crossing share 0 against 0 and
    0.2439 against trimesh's 0.2355, bound 0.03 on the difference (a lattice integration of the
    faces gives 0.2419: trimesh's draw carries most of the 0.0084). **Mutation probes:** faces
    selected by a corner in the ball read crossing share 0 on the ``"small"`` ball (0.236 off, 8x
    the bound) and KS 0.067; area ignored over the selected faces reads KS 0.15 / 0.20.
    """
    mesh_tm, mesh_wp = sphere_irregular
    center_np, radius = _in_ball_case(mesh_tm, ball)
    count = 50_000
    points_wp, face_index_wp = od.sample.sample_surface_in_ball(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*center_np.tolist()), radius, count, seed=0
    )
    samples_tm, face_index_tm = _in_ball_dense(mesh_tm, center_np, radius, 2_000_000)

    corner_inside = np.linalg.norm(mesh_tm.vertices - center_np, axis=1) <= radius
    crossing_np = ~corner_inside[mesh_tm.faces].any(axis=1)
    # Non-vacuity: the wide ball spans hundreds of faces, the small one has crossing faces.
    assert np.unique(face_index_tm).size > (100 if ball == "wide" else 8)
    if ball == "small":
        assert crossing_np[face_index_tm].mean() > 0.1

    assert points_wp.shape == (count,)
    points_np = points_wp.numpy()
    _assert_on_reported_faces_in_ball(mesh_tm, points_np, face_index_wp.numpy(), center_np, radius)
    distance_np = np.linalg.norm(points_np - center_np, axis=1)
    distance_tm = np.linalg.norm(samples_tm - center_np, axis=1)
    assert ks_2samp(distance_np, distance_tm).statistic < 0.03
    share_np = crossing_np[face_index_wp.numpy()].mean()
    assert abs(share_np - crossing_np[face_index_tm].mean()) < 0.03


def test_sample_surface_in_ball_covers_a_ball_inside_one_face(device: str) -> None:
    """
    Not a library comparison: a disk inside a face of the cube, where the answer is analytic.

    The ball sits inside one triangle of the cube's ``x = 0.5`` face, away from its corners, so
    no vertex and no centroid lies in it and only the faces crossing the sphere carry samples. Its
    disk covers ``2.5e-3`` of the triangle, so the draw takes several rejection rounds. The
    samples must fill the disk uniformly: their mean is the center (the disk's centroid), and half
    of them lie within ``radius / sqrt(2)`` (the disk's area law). A sampler confined to a ring or
    an off-center patch fails one of the two; one drawing nothing returns no samples.
    """
    vertices, faces = od.creation.box(device=device)
    center_np = np.array([0.5, 0.1, 0.2])
    radius = 0.02
    count = 4_000
    points_wp, face_index_wp = od.sample.sample_surface_in_ball(
        vertices, faces, wp.vec3(*center_np.tolist()), radius, count, seed=4
    )
    assert points_wp.shape == (count,)
    points_np = points_wp.numpy()
    offsets_np = points_np - center_np
    assert np.abs(offsets_np[:, 0]).max() < 1e-6
    distance_np = np.linalg.norm(offsets_np, axis=1)
    assert distance_np.max() <= radius * (1.0 + 1e-6)
    assert np.unique(face_index_wp.numpy()).size == 1
    # The mean's standard error is ``radius / (2 sqrt(count))`` a coordinate: 1.6e-4.
    assert np.abs(offsets_np.mean(axis=0)).max() < 8e-4
    # A binomial share of 4 000: standard error 0.008.
    assert abs((distance_np <= radius / math.sqrt(2.0)).mean() - 0.5) < 0.04


# Balls cutting the unit cube (``od.creation.box``) that hold no vertex and no face centroid, each
# meeting the faces a different way: across the edge between two faces from outside the solid;
# over three faces near a corner they share, from inside, missing the corner (the ``x = 0.5``
# disk touches neither edge nor diagonal of its triangle); across the diagonal splitting the top
# face; and strictly inside one top triangle, touching none of its edges.
_HARD_CUBE_BALLS = {
    "edge": ((0.6, 0.6, 0.0), 0.35),
    "corner": ((0.3, 0.3, 0.3), 0.25),
    "diagonal": ((0.0, 0.0, 0.3), 0.25),
    "interior": ((0.35, -0.35, 0.55), math.hypot(0.1, 0.05)),
}


def _lattice_in_ball(
    triangles_np: np.ndarray, center_np: np.ndarray, radius: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return each triangle's share of the area inside the ball, and distances to points covering it.

    Every triangle carries the same ``k * (k + 1) / 2`` points of a uniform barycentric lattice,
    so on triangles of equal area (the cube's) the lattice points in the ball are an evenly spread
    sample of the surface inside it, and each triangle's count of them its share of that area.
    """
    rows, cols = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    below = rows + cols < k
    u = (rows[below] + 1.0 / 3.0) / k
    w = (cols[below] + 1.0 / 3.0) / k
    counts = np.zeros(triangles_np.shape[0])
    distances = []
    for t, (a, b, c) in enumerate(triangles_np):
        lattice = a + np.outer(u, b - a) + np.outer(w, c - a)
        distance = np.linalg.norm(lattice - center_np, axis=1)
        distance = distance[distance <= radius]
        counts[t] = distance.size
        distances.append(distance)
    return counts / counts.sum(), np.concatenate(distances)


@pytest.mark.parametrize("ball", list(_HARD_CUBE_BALLS))
def test_sample_surface_in_ball_selects_faces_by_their_closest_point(
    device: str, ball: str
) -> None:
    """
    Not a library comparison: on the cube the answer is a lattice integration, exact to 1e-3.

    Each ball (``_HARD_CUBE_BALLS``) meets one to five of the cube's twelve triangles and holds
    none of their vertices or centroids, so a selection by corner or centroid draws from no face,
    and an edge test misses a triangle whose interior alone reaches into the ball. The draw must
    put each triangle's share of the samples at its share of the area in the ball (a 1 200-row
    lattice on every triangle) and none on any other, and the samples' distances from the center
    must follow the lattice's. **Measured**, identical on both devices over three seeds of 50 000
    samples: worst share error 0.0031 (bound 0.01), KS 0.0068 (bound 0.025). **Mutation
    probes:** selecting by a corner or centroid in the ball returns no samples on every ball;
    selecting by an edge reaching the ball returns none on ``"interior"`` and leaves the
    ``"corner"`` ball's ``x = 0.5`` triangle, a third of its area, unsampled (share error 0.333;
    its KS stays at 0.005, so the share is the assert that carries it).
    """
    center_tuple, radius = _HARD_CUBE_BALLS[ball]
    center_np = np.array(center_tuple)
    vertices, faces = od.creation.box(device=device)
    vertices_np = vertices.numpy().astype(np.float64)
    triangles_np = vertices_np[faces.numpy().reshape(-1, 3)]
    # The premise: no vertex and no centroid in the ball, yet it meets the surface.
    assert np.linalg.norm(vertices_np - center_np, axis=1).min() > radius
    assert np.linalg.norm(triangles_np.mean(axis=1) - center_np, axis=1).min() > radius
    shares_np, distances_np = _lattice_in_ball(triangles_np, center_np, radius, 1_200)
    assert (
        np.count_nonzero(shares_np) == {"edge": 2, "corner": 5, "diagonal": 2, "interior": 1}[ball]
    )

    count = 50_000
    points_wp, face_index_wp = od.sample.sample_surface_in_ball(
        vertices, faces, wp.vec3(*center_tuple), radius, count, seed=0
    )
    assert points_wp.shape == (count,)
    points_np = points_wp.numpy()
    face_index_np = face_index_wp.numpy()
    assert np.linalg.norm(points_np - center_np, axis=1).max() <= radius * (1.0 + 1e-6)
    closest_np = tm.triangles.closest_point(triangles_np[face_index_np], points_np)
    assert np.linalg.norm(closest_np - points_np, axis=1).max() < 1e-6
    sample_shares_np = np.bincount(face_index_np, minlength=12) / count
    assert np.all(sample_shares_np[shares_np == 0.0] == 0.0)
    assert np.abs(sample_shares_np - shares_np).max() < 0.01
    distance_np = np.linalg.norm(points_np - center_np, axis=1)
    assert ks_2samp(distance_np, distances_np).statistic < 0.025


def test_sample_surface_in_ball_poisson_disk_spreads_the_count(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: no library eliminates a pool confined to a ball.

    The ``"poisson_disk"`` method runs ``sample_surface_poisson_disk``'s elimination over a
    uniform pool inside the ball, its radii sized off the area inside it. It returns exactly
    ``count`` samples in the ball and on the surface, spread as Poisson-disk samples: the closest
    pair at least half the elimination radius ``r_max`` (measured 0.73 ``r_max`` on the
    ``"small"`` ball, identical on both devices; the uniform method's closest pair is 0.10), and
    every point of a dense uniform draw in the ball within 1.2 ``r_max`` of a sample (measured
    0.87).
    """
    mesh_tm, mesh_wp = sphere_irregular
    center_np, radius = _in_ball_case(mesh_tm, "small")
    count = 100
    center = wp.vec3(*center_np.tolist())
    points_wp, face_index_wp = od.sample.sample_surface_in_ball(
        mesh_wp.points, mesh_wp.indices, center, radius, count, method="poisson_disk", seed=2
    )
    assert points_wp.shape == (count,)
    points_np = points_wp.numpy()
    _assert_on_reported_faces_in_ball(mesh_tm, points_np, face_index_wp.numpy(), center_np, radius)
    dense_np, _ = _in_ball_dense(mesh_tm, center_np, radius, 400_000)
    area_in_ball = mesh_tm.area * dense_np.shape[0] / 400_000
    r_max = 2.0 * math.sqrt(area_in_ball / count / (2.0 * math.sqrt(3.0)))
    assert pdist(points_np).min() >= 0.5 * r_max
    assert np.max(cKDTree(points_np).query(dense_np)[0]) <= 1.2 * r_max
    uniform_np = od.sample.sample_surface_in_ball(
        mesh_wp.points, mesh_wp.indices, center, radius, count, seed=2
    )[0].numpy()
    assert pdist(uniform_np).min() < 0.25 * r_max  # Non-vacuity: a uniform draw clumps.


@pytest.mark.parametrize("sampler", ["surface", "in_ball"])
def test_poisson_disk_without_oversampling_returns_the_uniform_draw(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], sampler: str
) -> None:
    """
    Ordito against ordito: at ``init_factor=1`` the pool is the answer, and nothing is eliminated.

    The pool is ``count`` uniform samples drawn with the call's seed, so the Poisson-disk sampler
    must return the uniform sampler's draw for the same seed, point for point and face for face,
    over the whole surface and inside a ball alike. Non-vacuity: the default oversampled pool
    eliminates down to a different set.
    """
    mesh_tm, mesh_wp = sphere_irregular
    count = 60
    if sampler == "surface":
        draw = partial(
            od.sample.sample_surface_poisson_disk, mesh_wp.points, mesh_wp.indices, count
        )
        uniform_wp, uniform_faces_wp = od.sample.sample_surface(
            mesh_wp.points, mesh_wp.indices, count, seed=4
        )
    else:
        center_np, radius = _in_ball_case(mesh_tm, "small")
        center = wp.vec3(*center_np.tolist())
        draw = partial(
            od.sample.sample_surface_in_ball,
            mesh_wp.points,
            mesh_wp.indices,
            center,
            radius,
            count,
            method="poisson_disk",
        )
        uniform_wp, uniform_faces_wp = od.sample.sample_surface_in_ball(
            mesh_wp.points, mesh_wp.indices, center, radius, count, seed=4
        )
    pool_wp, pool_faces_wp = draw(init_factor=1.0, seed=4)
    assert uniform_wp.shape == (count,)
    assert np.array_equal(pool_wp.numpy(), uniform_wp.numpy())
    assert np.array_equal(pool_faces_wp.numpy(), uniform_faces_wp.numpy())
    eliminated_wp, _ = draw(seed=4)
    assert not np.array_equal(eliminated_wp.numpy(), uniform_wp.numpy())


@pytest.mark.parametrize("ball", ["wide", "small"])
def test_sample_surface_in_ball_blue_noise_packs_the_ball(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], ball: str
) -> None:
    """
    Not a library comparison: no library throws darts over the part of a surface in a ball.

    The ``"blue_noise"`` method prunes ``sample_surface_blue_noise``'s pool to the ball before the
    dart throw, so the samples are a maximal ``spacing``-packing of the surface inside the ball,
    its border included. Closest pair at least ``spacing`` (exact, as for the whole surface), and
    the worst gap from a dense uniform draw in the ball at most 1.15 ``spacing`` (measured 1.054
    ``"wide"`` / 1.049 ``"small"``, identical on both devices). **Mutation probe:** packing the
    faces meeting the ball and pruning the packing to it leaves the rim's samples without their
    outside neighbours' replacements, gaps 1.29 / 1.19.
    """
    mesh_tm, mesh_wp = sphere_irregular
    center_np, radius = _in_ball_case(mesh_tm, ball)
    spacing = radius / (6.0 if ball == "wide" else 4.0)
    points_wp, face_index_wp = od.sample.sample_surface_in_ball(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*center_np.tolist()),
        radius,
        method="blue_noise",
        spacing=spacing,
        seed=5,
    )
    points_np = points_wp.numpy()
    assert points_wp.size > 10
    _assert_on_reported_faces_in_ball(mesh_tm, points_np, face_index_wp.numpy(), center_np, radius)
    dense_np, _ = _in_ball_dense(mesh_tm, center_np, radius, 400_000)
    closest, gap = _blue_noise_statistics(points_np, spacing, dense_np)
    assert closest >= 1.0 - 1e-5
    assert gap <= 1.15


def test_sample_surface_in_ball_returns_nothing_for_a_ball_off_the_surface(device: str) -> None:
    """
    Not a library comparison: a ball meeting no face returns empty buffers for every method.

    The ball is off the cube, so no face reaches into it and the draw has nothing to weight; the
    call returns rather than raising, as for a zero ``count``.
    """
    vertices, faces = od.creation.box(device=device)
    center = wp.vec3(3.0, 0.0, 0.0)
    cases: tuple[tuple[int, Literal["uniform", "poisson_disk"]], ...] = (
        (10, "uniform"),
        (10, "poisson_disk"),
        (0, "uniform"),
    )
    for count, method in cases:
        buffers = od.sample.sample_surface_in_ball(
            vertices, faces, center, 1.0, count, method=method
        )
        assert all(buffer.shape == (0,) for buffer in buffers)
    buffers = od.sample.sample_surface_in_ball(
        vertices, faces, center, 1.0, method="blue_noise", spacing=0.1
    )
    assert all(buffer.shape == (0,) for buffer in buffers)


def test_sample_surface_in_ball_blue_noise_of_a_ball_no_pool_point_reaches(device: str) -> None:
    """
    Not a library comparison: a ball on the surface that the spacing's pool misses keeps nothing.

    The pool is drawn at the spacing's density over the faces meeting the ball, about a hundred
    points on a cube face for a ``0.1`` spacing, so a ball of radius ``1e-3`` on that face holds
    none of them (seeded): the dart throw gets an empty pool and returns empty buffers rather than
    gridding nothing. Non-vacuity: a ball of ``0.05`` on the same face keeps samples, all inside.
    """
    vertices, faces = od.creation.box(device=device)
    center_np = np.array([0.5, 0.1, 0.2])
    center = wp.vec3(*center_np.tolist())
    buffers = od.sample.sample_surface_in_ball(
        vertices, faces, center, 1e-3, method="blue_noise", spacing=0.1, seed=0
    )
    assert all(buffer.shape == (0,) for buffer in buffers)
    samples_wp, _ = od.sample.sample_surface_in_ball(
        vertices, faces, center, 0.05, method="blue_noise", spacing=0.1, seed=0
    )
    samples_np = samples_wp.numpy()
    assert samples_np.size > 0
    assert np.linalg.norm(samples_np - center_np, axis=1).max() <= 0.05


def test_sample_surface_in_ball_returns_what_its_rounds_kept(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: the draw's round limit, which a vanishing ball reaches.

    Held to one round of at most 8 000 draws (twice the count), a ball covering ``2.5e-3`` of its
    face keeps about twenty: the call returns those, each in the ball and on its face, rather than
    ``count`` samples.
    """
    monkeypatch.setattr(od.sample, "_BALL_ROUNDS", 1)
    monkeypatch.setattr(od.sample, "_BALL_ROUND_DRAW_CAP", 0)
    vertices, faces = od.creation.box(device=device)
    center_np = np.array([0.5, 0.1, 0.2])
    points_wp, face_index_wp = od.sample.sample_surface_in_ball(
        vertices, faces, wp.vec3(*center_np.tolist()), 0.02, 4_000, seed=4
    )
    assert 0 < points_wp.shape[0] < 100
    assert face_index_wp.shape == points_wp.shape
    assert np.linalg.norm(points_wp.numpy() - center_np, axis=1).max() <= 0.02 * (1.0 + 1e-6)


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

    At 20 000 samples the per-axis standard error is ~0.006 and the measured offset is **0.011**
    (on ``x``, both devices), so ``atol=0.035`` is 3x the measurement and ~6 sigma -- clear of noise
    and tight enough to catch a distribution biased toward one side of the solid. Fixture premise
    as in ``test_sample_volume_containment``.
    """
    mesh_tm, mesh_wp = sphere_well_shaped
    points_np = od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 20_000, seed=0).numpy()
    assert np.allclose(points_np.mean(axis=0), mesh_tm.center_mass, rtol=0.0, atol=0.035)


@pytest.mark.parametrize(
    ("fixture_name", "match"),
    [
        pytest.param("saddle_graded", "watertight", id="not_watertight"),
        pytest.param(None, "zero volume", id="zero_volume"),
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


def test_sample_volume_rejects_a_faceless_mesh_only_when_sampling(device: str) -> None:
    """
    Not a library comparison: a mesh with no faces encloses nothing to sample.

    It passes the watertight gate vacuously, so ``count == 0`` returns an empty buffer, and any
    positive count raises by name rather than reading an empty CDF.
    """
    vertices_wp = points_to_warp(np.eye(3), device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.sample.sample_volume(vertices_wp, faces_wp, 0).shape == (0,)
    with pytest.raises(ValueError, match="mesh has no faces; cannot sample its volume"):
        od.sample.sample_volume(vertices_wp, faces_wp, 5)


@pytest.mark.parametrize(
    "sampler",
    [
        "fibonacci_sphere",
        "poisson_disk",
        "blue_noise",
        "volume",
        "in_ball_uniform",
        "in_ball_poisson_disk",
        "in_ball_blue_noise",
    ],
)
def test_samplers_are_deterministic(request: pytest.FixtureRequest, sampler: str) -> None:
    """
    Ordito against ordito: two calls with the same arguments and seed return identical buffers.

    The lattice takes no seed, so its claim is plain reproducibility; the three seeded samplers
    are called at ``seed=7`` twice, and every returned buffer (points and, where returned, face
    indices) is compared bit for bit. ``sample_volume`` takes ``sphere_well_shaped``, the solid it
    accepts (``test_sample_volume_containment``). The in-ball draws take the ``"small"`` ball,
    whose uniform pool takes several rounds.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(
        "sphere_well_shaped" if sampler == "volume" else "sphere_irregular"
    )
    center_np, radius = _in_ball_case(mesh_tm, "small")
    center = wp.vec3(*center_np.tolist())

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
        if sampler.startswith("in_ball_"):
            method = "uniform" if sampler == "in_ball_uniform" else "poisson_disk"
            if sampler == "in_ball_blue_noise":
                return od.sample.sample_surface_in_ball(
                    mesh_wp.points,
                    mesh_wp.indices,
                    center,
                    radius,
                    method="blue_noise",
                    spacing=radius / 4.0,
                    seed=7,
                )
            return od.sample.sample_surface_in_ball(
                mesh_wp.points, mesh_wp.indices, center, radius, 300, method=method, seed=7
            )
        return (od.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 200, seed=7),)

    first, second = draw(), draw()
    for array_a, array_b in zip(first, second, strict=True):
        assert np.array_equal(array_a.numpy(), array_b.numpy())


@pytest.mark.parametrize("sampler", ["surface", "volume"])
def test_cdf_samplers_are_deterministic_on_a_large_mesh(device: str, sampler: str) -> None:
    """
    Ordito against ordito: seeded CDF draws on a 1.3M-face sphere repeat bit for bit.

    ``test_samplers_are_deterministic`` runs on fixtures too small to see the scan: both samplers
    invert a float32 face CDF built by an inclusive scan, and CUB's arrival-order scan on CUDA
    moved that CDF's last bits from call to call at this size, so every pair of seeded draws of
    100k points differed in some sample. A second draw must equal the first.
    """
    vertices, faces = od.creation.icosphere(8, device=device)

    def draw() -> wp.array[wp.vec3]:
        if sampler == "surface":
            return od.sample.sample_surface(vertices, faces, 100_000, seed=7)[0]
        return od.sample.sample_volume(vertices, faces, 100_000, seed=7)

    assert np.array_equal(draw().numpy(), draw().numpy())


def test_sample_surface_arrival_order_scan_draws_the_same_points(device: str) -> None:
    """
    Ordito against ordito: determinism off draws the default's points, to rounding.

    The two scans of a 1.3M-face CDF sum in different orders, and their float32 rounding there is
    about one face's share of the total: a seeded draw keeps its random numbers, so a sample can
    only move to a neighbouring entry of the CDF -- measured within two face indices, a fifth of
    the samples moving -- never elsewhere on the mesh. The draws themselves are pinned to the
    reference sampler by ``test_sample_surface``; this pins the flag to the default. On the CPU
    device both arms are one sequential scan and agree exactly.
    """
    vertices, faces = od.creation.icosphere(8, device=device)
    fixed, fixed_faces = od.sample.sample_surface(vertices, faces, 100_000, seed=7)
    with od.config.deterministic(False):
        fast, fast_faces = od.sample.sample_surface(vertices, faces, 100_000, seed=7)
    assert np.abs(fixed_faces.numpy() - fast_faces.numpy()).max() <= 2
    assert np.allclose(np.linalg.norm(fast.numpy(), axis=1), 1.0, atol=1e-5)
    if wp.get_device(device).is_cpu:
        assert np.array_equal(fixed.numpy(), fast.numpy())


@pytest.mark.parametrize(
    "sampler",
    [
        "fibonacci_sphere",
        "fibonacci_hemisphere",
        "fibonacci_cone",
        "surface",
        "surface_barycentric",
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
    elif sampler == "surface":
        buffers = od.sample.sample_surface(mesh_wp.points, mesh_wp.indices, 0, seed=0)
    elif sampler == "surface_barycentric":
        buffers = od.sample.sample_surface(
            mesh_wp.points, mesh_wp.indices, 0, seed=0, return_barycentric=True
        )
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


@pytest.mark.parametrize(
    "sampler", ["sample_volume", "sample_surface_poisson_disk", "sample_surface_blue_noise"]
)
def test_arrival_order_scan_samplers_draw_valid_samples(device: str, sampler: str) -> None:
    """
    Ordito against ordito: determinism off draws a valid sample of the default's size.

    The mode only swaps the fixed-order scan of the CDF for CUB's, so a seeded draw keeps its
    random numbers and a sample can move only to a neighbouring CDF entry: the volume samples stay
    inside the unit sphere, and the surface samplers return the same count on the surface. On the
    CPU device both arms are one sequential scan and the draws are equal.
    """
    vertices, faces = od.creation.icosphere(4, device=device)
    if sampler == "sample_volume":
        fixed = od.sample.sample_volume(vertices, faces, 2000, seed=3)
        with od.config.deterministic(False):
            arrival = od.sample.sample_volume(vertices, faces, 2000, seed=3)
        assert np.linalg.norm(arrival.numpy(), axis=1).max() <= 1.0 + 1e-5
    else:
        draw = getattr(od.sample, sampler)
        size = 200 if sampler == "sample_surface_poisson_disk" else 0.15
        fixed = draw(vertices, faces, size, seed=3)[0]
        with od.config.deterministic(False):
            arrival = draw(vertices, faces, size, seed=3)[0]
        assert np.allclose(np.linalg.norm(arrival.numpy(), axis=1), 1.0, atol=1e-2)
        if sampler == "sample_surface_poisson_disk":
            assert arrival.shape == fixed.shape
    assert fixed.shape[0] > 0
    if wp.get_device(device).is_cpu:
        assert np.array_equal(fixed.numpy(), arrival.numpy())


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_blue_noise(
                wp.zeros(3, dtype=wp.vec3, device=mesh_wp.points.device),
                wp.array([0, 1, 2], dtype=wp.int32, device=mesh_wp.points.device),
                0.1,
                seed=0,
            ),
            "mesh has zero surface area",
            id="blue_noise_zero_area",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_fibonacci_cone(8, 0.0, device=mesh_wp.points.device),
            r"half_angle must be in \(0, pi\]",
            id="cone_half_angle_zero",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_fibonacci_cone(8, 4.0, device=mesh_wp.points.device),
            r"half_angle must be in \(0, pi\]",
            id="cone_half_angle_past_pi",
        ),
        # A pool below ``count`` is refused, even at ``count == 0``.
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_poisson_disk(
                mesh_wp.points, mesh_wp.indices, 0, init_factor=0.5
            ),
            r"init_factor must be >= 1, got 0\.5",
            id="poisson_init_factor_below_one_empty",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_poisson_disk(
                mesh_wp.points, mesh_wp.indices, 10, init_factor=0.5
            ),
            r"init_factor must be >= 1, got 0\.5",
            id="poisson_init_factor_below_one",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_blue_noise(
                mesh_wp.points, mesh_wp.indices, 0.0
            ),
            "radius",
            id="blue_noise_radius_zero",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points,
                mesh_wp.indices,
                wp.vec3(0.0, 0.0, 0.0),
                1.0,
                5,
                method="grid",  # pyright: ignore[reportArgumentType]
            ),
            "method must be one of",
            id="in_ball_method",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 0.0), 0.0, 5
            ),
            r"radius must be > 0, got 0\.0",
            id="in_ball_radius_zero",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 0.0), 1.0
            ),
            "method='uniform' takes a count >= 0, got None",
            id="in_ball_uniform_no_count",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 0.0), 1.0, -1
            ),
            "takes a count >= 0, got -1",
            id="in_ball_negative_count",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 0.0), 1.0, 5, spacing=0.1
            ),
            "spacing must be None",
            id="in_ball_uniform_spacing",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 0.0), 1.0, method="blue_noise"
            ),
            "takes a spacing > 0, got None",
            id="in_ball_blue_noise_no_spacing",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points,
                mesh_wp.indices,
                wp.vec3(0.0, 0.0, 0.0),
                1.0,
                5,
                method="blue_noise",
                spacing=0.1,
            ),
            "count must be None",
            id="in_ball_blue_noise_count",
        ),
        pytest.param(
            lambda mesh_wp: od.sample.sample_surface_in_ball(
                mesh_wp.points,
                mesh_wp.indices,
                wp.vec3(0.0, 0.0, 0.0),
                1.0,
                0,
                method="poisson_disk",
                init_factor=0.5,
            ),
            r"init_factor must be >= 1, got 0\.5",
            id="in_ball_poisson_init_factor",
        ),
    ],
)
def test_invalid_arguments_raise(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], call: Callable[[wp.Mesh], object], message: str
) -> None:
    """Not a library comparison: each documented ``ValueError``, with its message."""
    _, mesh_wp = sphere_irregular
    with pytest.raises(ValueError, match=message):
        call(mesh_wp)
