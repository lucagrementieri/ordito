"""
Benchmarks for ``ordito.metrics``: the differentiable Chamfer loss and the plain metrics.

Cloud/mesh B is the same mesh translated by 5% of its bbox diagonal (untimed setup), so both
directions of every symmetric metric do real work.

Differentiable case
-------------------
``chamfer_mesh_to_mesh_loss`` carries ``requires_grad=True`` on both vertex buffers; each timed
round records a fresh tape, runs backward and zeroes the gradients, which is the real
optimization-loop cost. It is ordito-only: open3d has no autodiff, so timing forward-only against
forward-plus-backward would be a misleading ratio rather than a useful baseline.

Non-differentiable cases
------------------------
``chamfer_points_to_points`` and ``hausdorff_points_to_points`` are each **two**
[`query_nearest`][ordito.neighbors.query_nearest] calls at ``k=1`` plus a
reduction, so they are the direct measurement for a change to the k-NN kernel — the same kernel
that sits under all twelve of ``distance.py``'s nearest-neighbour call sites.

**open3d** is the reference: ``PointCloud.compute_point_cloud_distance`` returns exactly the forward
nearest-neighbour Euclidean distances (a serial ``KDTreeFlann`` search), from which both metrics
follow — Chamfer as the mean of the squares in each direction (the pytorch3d convention ordito
uses), Hausdorff as the overall maximum. Both baselines therefore run the same two searches ordito
does, and the host-side ``numpy`` reduction over the returned vector is inside the timed region
because open3d has no device-side equivalent to hide it behind.

**pymeshlab**'s ``get_hausdorff_distance`` is the second reference for the Hausdorff group, and it
returns ``{min, max, mean, RMS, n_samples, ...}`` in one call. Two parameters decide whether the row
means anything: it samples **one** layer and searches the other, so it is one-directional and the
symmetric answer needs both calls (both are timed); and ``samplenum`` defaults to **8**, so
``samplevert=True`` with ``samplenum`` set to the full cloud size is what makes it sample every
point instead of measuring almost nothing. It has no Chamfer entry point -- the mean of the squares
is not among the statistics it returns -- so it appears in the Hausdorff group alone.

trimesh and libigl have no point-cloud Chamfer/Hausdorff entry point (``igl.hausdorff`` is
mesh-to-mesh only and is already the documented reference for
[`hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh] in ``tests/``), so neither
appears here.
"""

from __future__ import annotations

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.loss as p3d_loss
import pytorch3d.structures as p3d_structures
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, points_torch_from_numpy, skip_larger_than

_TRANSLATION_FRACTION = 0.05

_grad_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}
_cloud_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_cloud_wp_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}
_cloud_o3d_cache: dict[str, tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]] = {}
_cloud_pml_cache: dict[str, ml.MeshSet] = {}


# MeshLib's own float upper bound: ``findProjections`` segfaults on ``math.inf`` rather than
# raising, and a ``0.0`` in that slot silently returns all-zero distances.
_MESHLIB_FLT_MAX = 3.4028234663852886e38


# pytorch3d's ``_C`` carries no spatial structure on either device, so both cloud rows here are
# Theta(N^2) and the CPU one is unaffordable past a feature mesh: it quadruples per doubling, so a
# scan mesh extrapolates to minutes *per round*, and ``--benchmark-json`` is written at session end,
# which would cost the whole file's rows.


def _clouds_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(a, b)`` float64 clouds: the mesh vertices and a rigidly translated copy."""
    name = bench_case.mesh_name
    if name not in _cloud_np_cache:
        vertices = bench_case.vertices_np
        offset = _TRANSLATION_FRACTION * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        _cloud_np_cache[name] = (vertices, np.ascontiguousarray(vertices + offset))
    return _cloud_np_cache[name]


def _clouds_wp(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _cloud_wp_cache:
        cloud_a, cloud_b = _clouds_np(bench_case)
        cloud_a_wp, cloud_b_wp = (
            wp.array(
                np.ascontiguousarray(c, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
            )
            for c in (cloud_a, cloud_b)
        )
        _cloud_wp_cache[key] = (cloud_a_wp, cloud_b_wp)
    return _cloud_wp_cache[key]


def _clouds_o3d(bench_case: BenchCase) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    name = bench_case.mesh_name
    if name not in _cloud_o3d_cache:
        cloud_a_o3d, cloud_b_o3d = (
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)) for c in _clouds_np(bench_case)
        )
        _cloud_o3d_cache[name] = (cloud_a_o3d, cloud_b_o3d)
    return _cloud_o3d_cache[name]


def _grad_inputs(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _grad_cache:
        vertices = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        offset = 0.05 * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        vertices_a = wp.array(vertices, dtype=wp.vec3, device=bench_case.device, requires_grad=True)
        vertices_b = wp.array(
            vertices + np.float32(offset),
            dtype=wp.vec3,
            device=bench_case.device,
            requires_grad=True,
        )
        _grad_cache[key] = (vertices_a, vertices_b)
    return _grad_cache[key]


@pytest.mark.benchmark(group="chamfer_mesh_to_mesh_loss")
@pytest.mark.benchlibs("ordito")
def test_chamfer_mesh_to_mesh_loss(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "dragon")
    vertices_a, vertices_b = _grad_inputs(bench_case)
    faces = bench_case.faces_wp

    def run() -> wp.array[wp.float32]:
        tape = wp.Tape()
        loss = od.metrics.chamfer_mesh_to_mesh_loss(vertices_a, faces, vertices_b, faces, tape=tape)
        tape.backward(loss=loss)
        tape.zero()
        return loss

    loss = bench_case.run(run)
    assert loss.shape == (1,)


@pytest.mark.benchmark(group="chamfer_points_to_points")
@pytest.mark.benchlibs("ordito", "open3d", "pytorch3d")
@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
def test_chamfer_points_to_points(bench_case: BenchCase, single_directional: bool) -> None:
    """
    Point-cloud Chamfer: ``k=1`` searches plus a mean-of-squares reduction.

    ``single_directional=False`` does not merely double the work -- it builds a *second*
    acceleration structure over the other cloud. So the symmetric row should be more than 2x the
    one-way row, and how much more is the build cost, which is otherwise invisible.

    **pytorch3d** is the only reference here with GPU kernels, and its ``chamfer_distance`` is the
    convention this module documents -- same squared distances, same mean-of-two-directions, pinned
    at 7.02e-08 in ``tests/test_metrics.py::test_chamfer_points_to_points_matches_pytorch3d``. It
    has no spatial structure on either device, which makes this the group where its crossover is
    sharpest: it wins at 20 000 points and loses by nearly two orders of magnitude at 200 000. Read
    the two rows as one curve; see the
    ``LIBRARIES`` block in [`conftest.py`](conftest.py) for why half of that swing is ordito's own
    search-radius heuristic. ``single_directional=True`` maps onto its ``single_directional=True``
    exactly, so the parametrize axis is shared rather than emulated.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        cloud_a_p3d, cloud_b_p3d = (
            points_torch_from_numpy(cloud_np, bench_case.torch_device)
            for cloud_np in _clouds_np(bench_case)
        )
        chamfer_p3d, _ = bench_case.run(
            lambda: p3d_loss.chamfer_distance(
                cloud_a_p3d, cloud_b_p3d, single_directional=single_directional
            )
        )
        assert float(chamfer_p3d) > 0.0
        return
    if bench_case.kind == "ordito":
        cloud_a, cloud_b = _clouds_wp(bench_case)
        chamfer = bench_case.run(
            lambda: od.metrics.chamfer_points_to_points(
                cloud_a, cloud_b, single_directional=single_directional
            )
        )
    elif single_directional:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        chamfer = bench_case.run(
            lambda: float(
                np.square(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))).mean()
            )
        )
    else:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        chamfer = bench_case.run(
            lambda: float(
                np.square(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))).mean()
                + np.square(np.asarray(cloud_b.compute_point_cloud_distance(cloud_a))).mean()
            )
        )
    assert chamfer > 0.0


_jittered_wp_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}


def _jittered_clouds_wp(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    """Build a *coincident* pair: the vertices and a copy jittered by ``1e-4`` of the diagonal."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _jittered_wp_cache:
        vertices = bench_case.vertices_np
        diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        jitter = np.random.default_rng(0).normal(scale=1e-4 * diagonal, size=vertices.shape)
        points_wp, jittered_wp = (
            wp.array(
                np.ascontiguousarray(c, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
            )
            for c in (vertices, vertices + jitter)
        )
        _jittered_wp_cache[key] = (points_wp, jittered_wp)
    return _jittered_wp_cache[key]


@pytest.mark.benchmark(group="chamfer_points_to_points_coincident")
@pytest.mark.benchlibs("ordito")
def test_chamfer_points_to_points_coincident(bench_case: BenchCase) -> None:
    """
    Symmetric point-cloud Chamfer on a coincident pair, the other end of the displacement axis.

    The shifted ``chamfer_points_to_points`` rows put every answer several spacings away; here
    every answer is a fraction of one, which is where a seeded hash grid answers the backward half
    faster than the closest-point descent on a large cloud, and where ``metrics`` chooses it. The
    choice reads the forward answer back, so the two rows together price what it wins and what it
    costs.
    """
    skip_larger_than(bench_case, "dragon")
    cloud_a, cloud_b = _jittered_clouds_wp(bench_case)
    chamfer = bench_case.run(lambda: od.metrics.chamfer_points_to_points(cloud_a, cloud_b))
    assert chamfer > 0.0


def _clouds_meshset_pml(bench_case: BenchCase) -> tuple[ml.MeshSet, int]:
    """
    Return the two clouds as face-less meshes 0 and 1 of one MeshSet, plus their point count.

    ``get_hausdorff_distance`` writes the per-sample distance into the sampled mesh's vertex scalar
    attribute and touches nothing else, so this is cached rather than rebuilt per round.
    """
    if bench_case.mesh_name not in _cloud_pml_cache:
        meshset_pml = ml.MeshSet()
        for cloud_np in _clouds_np(bench_case):
            meshset_pml.add_mesh(
                ml.Mesh(vertex_matrix=np.ascontiguousarray(cloud_np, dtype=np.float64))
            )
        _cloud_pml_cache[bench_case.mesh_name] = meshset_pml
    count = int(_clouds_np(bench_case)[0].shape[0])
    return _cloud_pml_cache[bench_case.mesh_name], count


@pytest.mark.benchmark(group="hausdorff_points_to_points")
@pytest.mark.benchlibs("ordito", "open3d", "pymeshlab")
def test_hausdorff_points_to_points(bench_case: BenchCase) -> None:
    """Symmetric point-cloud Hausdorff: the same two searches, reduced with ``max`` instead."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pymeshlab":
        # ``get_hausdorff_distance`` is *one-directional* by construction -- it samples one layer
        # and searches the other -- so the symmetric answer is both directions, and both are timed.
        # ``samplevert=True`` with ``samplenum`` at the full cloud size makes it sample every point
        # rather than its default 8, which would otherwise measure almost nothing.
        meshset_pml, count = _clouds_meshset_pml(bench_case)

        def hausdorff_pml() -> float:
            forward = meshset_pml.get_hausdorff_distance(
                sampledmesh=0, targetmesh=1, samplevert=True, samplenum=count
            )
            backward = meshset_pml.get_hausdorff_distance(
                sampledmesh=1, targetmesh=0, samplevert=True, samplenum=count
            )
            return max(float(forward["max"]), float(backward["max"]))

        assert bench_case.run(hausdorff_pml) > 0.0
        return
    if bench_case.kind == "ordito":
        cloud_a, cloud_b = _clouds_wp(bench_case)
        hausdorff = bench_case.run(lambda: od.metrics.hausdorff_points_to_points(cloud_a, cloud_b))
    else:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        hausdorff = bench_case.run(
            lambda: max(
                float(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b)).max()),
                float(np.asarray(cloud_b.compute_point_cloud_distance(cloud_a)).max()),
            )
        )
    assert hausdorff > 0.0


@pytest.mark.benchmark(group="chamfer_points_to_mesh")
@pytest.mark.benchlibs("ordito", "igl", "meshlib", "pytorch3d")
def test_chamfer_points_to_mesh(bench_case: BenchCase) -> None:
    """
    Cloud-to-surface Chamfer: an exact mesh query forward, a ``k=1`` cloud search backward.

    open3d's closest-point-on-surface equivalent lives in the *tensor* API
    (``o3d.t.geometry.RaycastingScene.compute_distance``), which is a different implementation from
    the legacy ``o3d.geometry`` baselines the rest of this suite uses; mixing the two in one table
    would compare implementations, not libraries.

    **libigl's row is the forward half only**, and is a *lower* bound rather than a race:
    ``igl.point_mesh_squared_distance`` is exactly the cloud-to-surface query -- it is already the
    oracle for this group in ``tests/test_metrics.py`` -- but it has no cloud-to-cloud counterpart,
    so the backward ``k=1`` search ordito also performs has no igl equivalent to pair it with. Read
    the row as "what the expensive half costs on one core"; the same partial-reference convention as
    ``igl.doublearea`` in [`test_triangles.py`](test_triangles.py).

    **MeshLib's row is the forward half too**, and is the multi-threaded one -- so it is the fair
    fight of the two references. Its AABB tree is built and pre-warmed outside the timed callable,
    because the tree is cached on the ``Mesh`` and a cold first query is one to two orders of
    magnitude above a warm one; that matches the ordito branch, which is handed a ``wp.Mesh`` it
    does not rebuild. Its
    ``distSq`` output is exactly ordito's ``point_reduction=None`` array, compared element-wise in
    ``tests/test_metrics.py``.

    **pytorch3d's row is the whole scalar and the only GPU one**, but it is a *different* pair of
    directions: ``point_mesh_face_distance`` sums point-to-triangle with **face**-to-point where
    ordito's backward half is mesh-vertex-to-point. So it is not an upper or lower bound on
    ordito's answer, it is a neighbouring quantity of the same cost shape -- two exact
    point-to-triangle sweeps -- which is why it is timed here and why
    ``tests/test_metrics.py::test_chamfer_points_to_mesh_matches_pytorch3d`` compares only the
    forward half, through a hand-ported per-pair table. Its ``Meshes`` and ``Pointclouds`` are
    immutable and pure, so both are built outside the timed callable, and the ``*_packed()``
    accessors the loss reads are warmed there too.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        mesh_p3d = bench_case.mesh_p3d
        _ = mesh_p3d.verts_packed(), mesh_p3d.faces_packed()  # warm the cached derivations
        cloud_p3d = p3d_structures.Pointclouds(
            points=[points_torch_from_numpy(_clouds_np(bench_case)[1], bench_case.torch_device)[0]]
        )
        distance_p3d = bench_case.run(
            lambda: p3d_loss.point_mesh_face_distance(mesh_p3d, cloud_p3d)
        )
        assert float(distance_p3d) > 0.0
        return
    if bench_case.kind == "meshlib":
        cloud_np = _clouds_np(bench_case)[1]
        # The tree is built lazily on the first query and cached on the Mesh, so it is built and
        # pre-warmed *outside* the timed callable -- the row then prices the query, matching what
        # the ordito branch does with a wp.Mesh already in hand. Timing the build instead is one to
        # two orders of magnitude different (CLAUDE.md section 7.6).
        mesh_ml = bench_case.new_mesh_ml()
        projector_ml = mm.PointsToMeshProjector()
        projector_ml.updateMeshData(mesh_ml)
        points_ml = mm.std_vector_Vector3_float()
        for point in cloud_np:
            points_ml.append(mm.Vector3f(float(point[0]), float(point[1]), float(point[2])))
        warmup_ml = mm.std_vector_MeshProjectionResult()
        projector_ml.findProjections(
            warmup_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _MESHLIB_FLT_MAX, 0.0
        )

        def project() -> mm.std_vector_MeshProjectionResult:
            results_ml = mm.std_vector_MeshProjectionResult()
            projector_ml.findProjections(
                results_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _MESHLIB_FLT_MAX, 0.0
            )
            return results_ml

        assert len(bench_case.run(project, rounds=3)) == cloud_np.shape[0]
        return
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference builds its AABB tree per call")
        cloud_np = _clouds_np(bench_case)[1]
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        squared_igl, _face_igl, _closest_igl = bench_case.run(
            lambda: igl.point_mesh_squared_distance(cloud_np, vertices_np, faces_np), rounds=3
        )
        assert squared_igl.shape[0] == cloud_np.shape[0]
        return
    cloud = _clouds_wp(bench_case)[1]
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    chamfer = bench_case.run(lambda: od.metrics.chamfer_points_to_mesh(cloud, vertices, faces))
    assert chamfer > 0.0
