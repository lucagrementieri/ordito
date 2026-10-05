"""
Regression tests for ``ordito.metrics`` Chamfer and Hausdorff metrics.

Point-cloud metrics compare against SciPy (``KDTree`` for Chamfer,
``scipy.spatial.distance.directed_hausdorff`` for Hausdorff). Mesh-surface metrics
compare against libigl's ``igl.point_mesh_squared_distance`` (the primitive behind
``igl::hausdorff``); ``igl.hausdorff`` itself is not exposed in the Python bindings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.loss as p3d_loss
import pytorch3d.structures as p3d_structures
import torch
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from meshlib import mrmeshpy as mm
from pytorch3d.loss import point_mesh_distance as p3d_point_mesh_distance
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

import ordito as od
from ordito.kernels import metrics as kernel_metrics
from tests.comparisons import assert_nonconstant
from tests.conversions import (
    numpy_to_warp,
    points_to_open3d,
    points_to_pymeshlab,
    points_to_pytorch3d,
    points_to_torch,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pytorch3d,
    trimesh_to_warp,
    warp_empty,
)

_DiffReduction = Literal["mean", "sum"]

# igl requires float64 vertices / int64 faces; Warp uses float32 / int32, so mesh-surface
# references diverge from Warp at roughly float32 precision.
_MESH_RTOL = 1e-4
_MESH_ATOL = 1e-5


def _igl_point_mesh_sqr_dist(
    query_np: np.ndarray, vertices_np: np.ndarray, faces_np: np.ndarray
) -> np.ndarray:
    sqr_distances_igl, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(query_np, dtype=np.float64),
        np.ascontiguousarray(vertices_np, dtype=np.float64),
        np.ascontiguousarray(faces_np, dtype=np.int64),
    )
    return np.asarray(sqr_distances_igl, dtype=np.float64)


# ---------------------------------------------------------------------------
# Chamfer: point cloud to point cloud
# ---------------------------------------------------------------------------


def test_chamfer_points_to_points_mean(device: str) -> None:
    rng = np.random.default_rng(0)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    chamfer_np = np.mean(distance_xy_np**2) + np.mean(distance_yx_np**2)

    chamfer_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-5)


def test_chamfer_points_to_points_sum(device: str) -> None:
    rng = np.random.default_rng(1)
    x_np = (rng.random((50, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((70, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    chamfer_np = np.sum(distance_xy_np**2) + np.sum(distance_yx_np**2)

    chamfer_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction="sum"
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-4)


def test_chamfer_points_to_points_max(device: str) -> None:
    rng = np.random.default_rng(2)
    x_np = (rng.random((50, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((70, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    chamfer_np = max(np.max(distance_xy_np**2), np.max(distance_yx_np**2))

    chamfer_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction="max"
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-5)


def test_chamfer_points_to_points_single_directional(device: str) -> None:
    rng = np.random.default_rng(3)
    x_np = (rng.random((40, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((90, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    chamfer_np = np.mean(KDTree(y_np).query(x_np)[0] ** 2)

    chamfer_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), single_directional=True
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-5)


def test_chamfer_points_to_points_unreduced(device: str) -> None:
    rng = np.random.default_rng(4)
    x_np = (rng.random((40, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((90, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    squared_xy_np = KDTree(y_np).query(x_np)[0] ** 2
    squared_yx_np = KDTree(x_np).query(y_np)[0] ** 2

    forward_wp, backward_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction=None
    )
    assert np.allclose(forward_wp.numpy(), squared_xy_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(backward_wp.numpy(), squared_yx_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
@pytest.mark.parity("chamfer_points_to_points", "open3d")
def test_chamfer_points_to_points_matches_open3d(device: str, single_directional: bool) -> None:
    """
    Class B: Open3D returns the per-point nearest distances, not the Chamfer scalar.

    ``PointCloud.compute_point_cloud_distance`` is the *unsquared, unreduced* one-way answer, so the
    named transform is ordito's own reduction applied to it -- square, then mean, and add the
    reverse direction when ``single_directional=False``. That is exactly what the benchmark's open3d
    branch does, so this asserts the two sides of that row compute the same number.
    """
    rng = np.random.default_rng(21)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    cloud_x_o3d, cloud_y_o3d = points_to_open3d(x_np), points_to_open3d(y_np)
    chamfer_o3d = np.square(
        np.asarray(cloud_x_o3d.compute_point_cloud_distance(cloud_y_o3d))
    ).mean()
    if not single_directional:
        chamfer_o3d += np.square(
            np.asarray(cloud_y_o3d.compute_point_cloud_distance(cloud_x_o3d))
        ).mean()

    chamfer_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device),
        points_to_warp(y_np, device),
        single_directional=single_directional,
    )
    assert np.allclose(chamfer_wp, chamfer_o3d, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("chamfer_points_to_points", "pytorch3d")
def test_chamfer_points_to_points_matches_pytorch3d(device: str) -> None:
    """
    Class A: ``loss.chamfer_distance`` is the convention ``ordito.metrics`` documents.

    This is the pair the module's own prose has been asserting since it was written -- *"Chamfer
    distances follow the ``pytorch3d`` convention and are built on squared distances"* -- with
    nothing running pytorch3d. Measured 7.02e-08 relative on two seeded clouds of 300 and 400
    points, which is the number those docstrings now carry.

    Two **different** clouds, deliberately: ``chamfer_distance(x, x)`` is 0.0 and so is ordito's,
    so a self-comparison passes while testing nothing.
    """
    rng = np.random.default_rng(7)
    a_np = rng.normal(size=(300, 3)).astype(np.float32)
    b_np = (rng.normal(size=(400, 3)) * 1.1).astype(np.float32)
    chamfer_p3d, _ = cast(
        "tuple[torch.Tensor, object]",
        p3d_loss.chamfer_distance(points_to_torch(a_np, device), points_to_torch(b_np, device)),
    )
    chamfer_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(a_np, device), points_to_warp(b_np, device)
    )

    assert float(chamfer_p3d) > 1e-3
    assert np.allclose(chamfer_wp, float(chamfer_p3d), rtol=1e-6, atol=0.0)


def _pairing(icosphere: tuple[tm.Trimesh, wp.Mesh], pairing: str) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` for the three pairings that decide the backward cloud search."""
    vertices_np = np.asarray(icosphere[0].vertices, dtype=np.float32)
    rng = np.random.default_rng(31)
    jittered_np = (vertices_np + rng.normal(scale=1e-4, size=vertices_np.shape)).astype(np.float32)
    if pairing == "coincident":
        return vertices_np, jittered_np
    if pairing == "displaced":
        return vertices_np, (jittered_np + 0.3).astype(np.float32)
    # ``x`` covers half of ``y``: the forward answer is small and the backward one is not.
    return vertices_np[vertices_np[:, 0] < 0.0], jittered_np


# Which search the backward half takes on each pairing: the seeded hash grid (``backend`` left at
# its default) when the forward answer is small against the cloud's spacing and covers nearly every
# query, the closest-point descent (``"bvh"``) otherwise -- ``partial`` is the pairing whose small
# forward answer the coverage test rejects. The forward half is the descent every time.
_BACKWARD_BACKEND = {"coincident": None, "displaced": "bvh", "partial": "bvh"}


@pytest.mark.parametrize("pairing", ["coincident", "displaced", "partial"])
@pytest.mark.parametrize("size_gate", [False, True])
def test_chamfer_points_to_points_matches_kdtree_on_either_backward_search(
    device: str,
    icosphere: tuple[tm.Trimesh, wp.Mesh],
    pairing: str,
    size_gate: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Class A: every per-point distance, both directions, against ``scipy.spatial.KDTree``.

    The backward search is chosen from the forward answer, so each pairing is one arm of that
    choice and the spy asserts the arm was taken -- a pairing that silently took the other search
    would leave one of the two unexercised. The fixture is far below the cloud size the choice is
    considered at, so ``size_gate=False`` lowers that threshold to reach it; with the threshold in
    place every pairing takes the descent, and so does every pairing on the CPU, where the choice
    is never made.
    """
    if not size_gate:
        monkeypatch.setattr(od.metrics, "_GRID_BACKWARD_MIN_POINTS", 0)
    x_np, y_np = _pairing(icosphere, pairing)
    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    assert np.ptp(distance_yx_np) > 1e-5  # not a constant answer (7.4)

    backends: list[object] = []
    query_nearest = cast("Callable[..., object]", od.neighbors.query_nearest)

    def spy(*args: object, **kwargs: object):
        backends.append(kwargs.get("backend"))
        return query_nearest(*args, **kwargs)

    monkeypatch.setattr(od.neighbors, "query_nearest", spy)
    forward_wp, backward_wp = od.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction=None
    )
    chooses = not size_gate and wp.get_device(device).is_cuda
    assert backends == ["bvh", _BACKWARD_BACKEND[pairing] if chooses else "bvh"]
    assert np.allclose(np.sqrt(forward_wp.numpy()), distance_xy_np, rtol=1e-5, atol=1e-6)
    assert np.allclose(np.sqrt(backward_wp.numpy()), distance_yx_np, rtol=1e-5, atol=1e-6)

    hausdorff_wp = od.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    expected = max(np.max(distance_xy_np), np.max(distance_yx_np))
    assert np.isclose(hausdorff_wp, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("pairing", ["coincident", "displaced", "partial"])
def test_chamfer_points_to_points_loss_grad_on_either_backward_search(
    device: str,
    icosphere: tuple[tm.Trimesh, wp.Mesh],
    pairing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Class A: the loss and both gradients against the closed form over ``KDTree``'s assignment.

    The differentiable form reads the backward search's *index*, so this is the check that the
    grid arm hands the tape the same assignment the descent would -- on a jittered pair, where no
    two candidates tie. The size threshold is lowered so the fixture reaches the choice.
    """
    monkeypatch.setattr(od.metrics, "_GRID_BACKWARD_MIN_POINTS", 0)
    x_np, y_np = (a.astype(np.float64) for a in _pairing(icosphere, pairing))
    n, m = len(x_np), len(y_np)
    nn_xy = KDTree(y_np).query(x_np)[1]
    nn_yx = KDTree(x_np).query(y_np)[1]
    loss_np = ((x_np - y_np[nn_xy]) ** 2).sum(-1).mean() + ((y_np - x_np[nn_yx]) ** 2).sum(
        -1
    ).mean()
    grad_x_np = 2.0 / n * (x_np - y_np[nn_xy])
    grad_y_np = 2.0 / m * (y_np - x_np[nn_yx])
    np.add.at(grad_y_np, nn_xy, -2.0 / n * (x_np - y_np[nn_xy]))
    np.add.at(grad_x_np, nn_yx, -2.0 / m * (y_np - x_np[nn_yx]))

    x_wp = wp.array(x_np, dtype=wp.vec3, device=device, requires_grad=True)
    y_wp = wp.array(y_np, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    loss_wp = od.metrics.chamfer_points_to_points_loss(x_wp, y_wp, tape=tape)
    tape.backward(loss=loss_wp)

    assert x_wp.grad is not None
    assert y_wp.grad is not None
    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)
    assert np.allclose(x_wp.grad.numpy(), grad_x_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)
    assert np.allclose(y_wp.grad.numpy(), grad_y_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)


# ---------------------------------------------------------------------------
# Chamfer: mesh to mesh (vertex-to-surface)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere"])
def test_chamfer_mesh_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_a_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    vertices_b_np = vertices_a_np + np.array([0.3, -0.15, 0.2])

    sqr_a_to_b_np = _igl_point_mesh_sqr_dist(vertices_a_np, vertices_b_np, faces_np)
    sqr_b_to_a_np = _igl_point_mesh_sqr_dist(vertices_b_np, vertices_a_np, faces_np)
    chamfer_np = np.mean(sqr_a_to_b_np) + np.mean(sqr_b_to_a_np)

    vertices_b_wp = points_to_warp(vertices_b_np, mesh_wp.device)
    chamfer_wp = od.metrics.chamfer_mesh_to_mesh(
        mesh_wp.points, mesh_wp.indices, vertices_b_wp, mesh_wp.indices
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


def test_chamfer_mesh_to_mesh_identical_is_zero(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    chamfer_wp = od.metrics.chamfer_mesh_to_mesh(
        mesh_wp.points, mesh_wp.indices, mesh_wp.points, mesh_wp.indices
    )
    assert np.allclose(chamfer_wp, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Chamfer: point cloud to mesh (mixed)
# ---------------------------------------------------------------------------

# MeshLib's own float upper bound. ``math.inf`` segfaults in ``findProjections`` rather than
# raising, which is why this is spelled out rather than computed.
_MESHLIB_FLT_MAX = 3.4028234663852886e38


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("chamfer_points_to_mesh", "meshlib")
def test_chamfer_points_to_mesh_forward_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the forward half: ``PointsToMeshProjector`` returns the same squared distances.

    The complement to [`test_chamfer_points_to_mesh`], which composes igl and scipy to reach the
    whole two-sided Chamfer. This one isolates the expensive half -- the cloud-to-surface query --
    and compares it **element-wise** rather than through a mean, which is strictly stronger: a mean
    hides a permutation and a pair of compensating errors, and MeshLib reports one
    ``MeshProjectionResult`` per query point so the correspondence is available. `distSq` is already
    squared, matching ordito's ``point_reduction=None`` output with no transform.

    Measured 2.4e-07 max absolute difference over 500 points (1.1e-07 relative), which is ordito's
    float32 vertex storage against MeshLib's float32 -- see the `getNumpyVerts` note in CLAUDE.md
    section 7.6.

    Three MeshLib call conventions this depends on, each a documented hazard:
    ``updateMeshData`` stores a raw pointer, so ``mesh_ml`` is bound to a name that outlives every
    query; ``upDistLimitSq`` precedes ``loDistLimitSq`` and must be MeshLib's own ``FLT_MAX``
    (``math.inf`` segfaults, and passing ``0.0`` in that slot silently returns all-zero distances --
    which is what a first attempt at this test did); and the AABB tree is built lazily on first
    query, so it is pre-warmed here to keep the comparison independent of call order.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(7)
    center = np.asarray(mesh_tm.vertices, dtype=np.float64).mean(axis=0)
    points_np = (center + rng.normal(scale=0.6, size=(500, 3))).astype(np.float32)

    forward_wp = od.metrics.chamfer_points_to_mesh(
        points_to_warp(points_np, mesh_wp.device),
        mesh_wp.points,
        mesh_wp.indices,
        single_directional=True,
        point_reduction=None,
    ).numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)  # must outlive the projector: it stores a raw pointer
    projector_ml = mm.PointsToMeshProjector()
    projector_ml.updateMeshData(mesh_ml)
    points_ml = mm.std_vector_Vector3_float()
    for point in points_np:
        points_ml.append(mm.Vector3f(float(point[0]), float(point[1]), float(point[2])))
    results_ml = mm.std_vector_MeshProjectionResult()
    projector_ml.findProjections(
        results_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _MESHLIB_FLT_MAX, 0.0
    )
    forward_ml = np.array([result.distSq for result in results_ml], dtype=np.float64)

    # Non-vacuity: an all-zero pair of answers would satisfy the comparison below.
    assert forward_ml.shape == forward_wp.shape
    assert forward_ml.min() > 0.0
    assert_nonconstant(forward_ml, tol=1e-3)
    assert np.allclose(forward_wp, forward_ml, rtol=_MESH_RTOL, atol=_MESH_ATOL)


def _point_triangle_squared(points_np: np.ndarray, triangles_np: np.ndarray) -> np.ndarray:
    """
    ``(n_points, n_triangles)`` squared point-to-triangle distances, the Ericson region test.

    A hand port of the standard closest-point-on-triangle classification (Ericson, *Real-Time
    Collision Detection*, section 5.1.5), which is the algorithm ``point_mesh_face_distance``'s
    CUDA/C++ kernel implements. Section 6 sanctions a hand port as a *test* oracle, and it is
    needed here rather than optional: pytorch3d returns only the **sum of two directions**, so
    without an independent per-pair table there is no way to isolate the forward half that ordito
    computes. Validated in ``test_chamfer_points_to_mesh_matches_pytorch3d`` both ways -- its
    forward column reproduces ``trimesh.proximity.closest_point`` to 1e-16 and its two columns
    summed reproduce pytorch3d's whole scalar to 1.3e-07.
    """
    a_np, b_np, c_np = triangles_np[:, 0], triangles_np[:, 1], triangles_np[:, 2]
    ab_np, ac_np = b_np - a_np, c_np - a_np
    queries_np = points_np[:, None, :]
    to_a_np, to_b_np, to_c_np = queries_np - a_np, queries_np - b_np, queries_np - c_np
    d1_np = np.einsum("ijk,jk->ij", to_a_np, ab_np)
    d2_np = np.einsum("ijk,jk->ij", to_a_np, ac_np)
    d3_np = np.einsum("ijk,jk->ij", to_b_np, ab_np)
    d4_np = np.einsum("ijk,jk->ij", to_b_np, ac_np)
    d5_np = np.einsum("ijk,jk->ij", to_c_np, ab_np)
    d6_np = np.einsum("ijk,jk->ij", to_c_np, ac_np)
    va_np = d3_np * d6_np - d5_np * d4_np
    vb_np = d5_np * d2_np - d1_np * d6_np
    vc_np = d1_np * d4_np - d3_np * d2_np
    denominator_np = va_np + vb_np + vc_np
    with np.errstate(divide="ignore", invalid="ignore"):
        weights_np = np.stack([va_np, vb_np, vc_np], axis=-1) / denominator_np[..., None]
    closest_np = np.einsum("ijk,jkl->ijl", weights_np, triangles_np)

    def on_segment(
        start_np: np.ndarray,
        end_np: np.ndarray,
        numerator_np: np.ndarray,
        denominator_np: np.ndarray,
    ):
        """Take the closest point on one edge, with the parameter clamped to ``[0, 1]``."""
        safe_np = np.where(denominator_np == 0.0, 1.0, denominator_np)
        parameter_np = np.clip(
            np.where(denominator_np == 0.0, 0.0, numerator_np / safe_np), 0.0, 1.0
        )
        return start_np + parameter_np[..., None] * (end_np - start_np)

    # The six degenerate regions, applied outermost-last so a vertex region wins over its edges.
    for region_np, value_np in (
        (
            (va_np <= 0) & ((d4_np - d3_np) >= 0) & ((d5_np - d6_np) >= 0),
            on_segment(b_np, c_np, d4_np - d3_np, (d4_np - d3_np) + (d5_np - d6_np)),
        ),
        ((vb_np <= 0) & (d2_np >= 0) & (d6_np <= 0), on_segment(a_np, c_np, d2_np, d2_np - d6_np)),
        ((vc_np <= 0) & (d1_np >= 0) & (d3_np <= 0), on_segment(a_np, b_np, d1_np, d1_np - d3_np)),
        ((d6_np >= 0) & (d5_np <= d6_np), np.broadcast_to(c_np, closest_np.shape)),
        ((d3_np >= 0) & (d4_np <= d3_np), np.broadcast_to(b_np, closest_np.shape)),
        ((d1_np <= 0) & (d2_np <= 0), np.broadcast_to(a_np, closest_np.shape)),
    ):
        closest_np = np.where(region_np[..., None], value_np, closest_np)
    return np.sum((queries_np - closest_np) ** 2, axis=-1)


@pytest.mark.parity("chamfer_points_to_mesh", "pytorch3d")
def test_chamfer_points_to_mesh_matches_pytorch3d(device: str) -> None:
    """
    Class B: pytorch3d's scalar **minus** its face-to-point half is ordito's forward mean.

    A measured restriction rather than a caution. ``point_mesh_face_distance`` is the sum of two
    squared means: point-to-*triangle*, which is ordito's forward direction, and
    **face-to-point** -- the minimum over the cloud of the point-to-triangle distance, one per
    face. ordito's backward direction is mesh-**vertex** to nearest query, a different quantity:
    measured 0.606 against pytorch3d's 0.586 here, so the two whole scalars are not comparable and
    the backward direction is deliberately not compared.

    The decomposition is what licenses the subtraction, and it is checked rather than assumed: the
    hand-ported per-pair table's two columns summed reproduce pytorch3d's scalar to 1.3e-07, and
    its forward column reproduces ``trimesh.proximity.closest_point`` to 1e-16. Only then is
    ``scalar - backward`` a quantity ordito can be held to; measured 1.02e-07 against it.

    ``min_triangle_area`` is left at its default and ``icosphere`` clears it, so no face is
    silently dropped from pytorch3d's side.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(5)
    queries_np = (rng.normal(size=(300, 3)) * 0.9).astype(np.float32)
    scalar_p3d = float(
        p3d_loss.point_mesh_face_distance(
            trimesh_to_pytorch3d(mesh_tm, device), points_to_pytorch3d(queries_np, device=device)
        )
    )
    squared_np = _point_triangle_squared(
        np.asarray(queries_np, np.float64), np.ascontiguousarray(mesh_tm.vertices[mesh_tm.faces])
    )
    forward_np, backward_np = squared_np.min(axis=1).mean(), squared_np.min(axis=0).mean()
    _, reference_np, _ = tm_proximity.closest_point(mesh_tm, np.asarray(queries_np, np.float64))

    # The oracle is only usable if it reproduces both sides it stands between.
    assert np.allclose(scalar_p3d, forward_np + backward_np, rtol=1e-6, atol=0.0)
    assert np.allclose(forward_np, (reference_np**2).mean(), rtol=1e-12, atol=0.0)

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    forward_wp = od.metrics.chamfer_points_to_mesh(
        points_to_warp(queries_np, device), vertices_wp, faces_wp, single_directional=True
    )
    both_wp = od.metrics.chamfer_points_to_mesh(
        points_to_warp(queries_np, device), vertices_wp, faces_wp
    )

    assert np.allclose(forward_wp, scalar_p3d - backward_np, rtol=1e-6, atol=1e-7)
    # And the backward halves really do differ, so the restriction above is not decoration.
    assert not np.allclose(both_wp, scalar_p3d, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("chamfer_points_to_mesh", "igl")
def test_chamfer_points_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (a named composition): igl supplies the forward half, scipy the backward one.

    ``igl.point_mesh_squared_distance`` *is* the cloud-to-surface query and is exact, but there is
    no igl cloud-to-cloud counterpart, so the reference Chamfer is assembled from it plus a
    ``KDTree`` ``k=1`` search back from the mesh vertices -- the same two halves ordito computes,
    in the same order, each from an independent implementation. That composition is why the
    benchmark's igl row is the forward half alone and reads as a lower bound.

    The tolerance is ``_MESH_RTOL`` rather than ``1e-5`` for a stated reason: igl consumes float64
    vertices where Warp carries float32, so the two agree only to float32 precision here.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)

    rng = np.random.default_rng(5)
    center = vertices_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.6, size=(120, 3))).astype(np.float32)

    forward_sqr_igl = _igl_point_mesh_sqr_dist(points_np, vertices_np, faces_np)
    backward_sqr_np = KDTree(points_np.astype(np.float64)).query(vertices_np)[0] ** 2
    chamfer_np = np.mean(forward_sqr_igl) + np.mean(backward_sqr_np)

    chamfer_wp = od.metrics.chamfer_points_to_mesh(
        points_to_warp(points_np, mesh_wp.device), mesh_wp.points, mesh_wp.indices
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


# ---------------------------------------------------------------------------
# Hausdorff: point cloud to point cloud
# ---------------------------------------------------------------------------


def test_hausdorff_points_to_points(device: str) -> None:
    rng = np.random.default_rng(6)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    hausdorff_np = max(directed_hausdorff(x_np, y_np)[0], directed_hausdorff(y_np, x_np)[0])

    hausdorff_wp = od.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=1e-5, atol=1e-5)


def test_hausdorff_points_to_points_single_directional(device: str) -> None:
    rng = np.random.default_rng(7)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    hausdorff_np = directed_hausdorff(x_np, y_np)[0]

    hausdorff_wp = od.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), single_directional=True
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("hausdorff_points_to_points", "open3d", "pymeshlab")
def test_hausdorff_points_to_points_matches_open3d_and_pymeshlab(device: str) -> None:
    """
    Class B against both references the benchmark times, each needing a different named transform.

    **Open3D** returns the per-point one-way distances from ``compute_point_cloud_distance``, so the
    transform is the ``max`` reduction plus the second direction -- the symmetric Hausdorff is
    ``max`` over both.

    **MeshLab** returns a *dict of statistics* from ``get_hausdorff_distance``, so the transform is
    the ``"max"`` key, again over both directions. Two further parameters are load-bearing and are
    the same ones the benchmark passes: the filter is one-directional by construction (it samples
    ``sampledmesh`` and searches ``targetmesh``), and its default ``samplenum=8`` would compare
    eight random points rather than the cloud, so ``samplevert=True`` with ``samplenum`` at the full
    count is what makes it sample every point. The clouds go in as **face-less** meshes, since
    MeshLab's sampler would otherwise sample the surface instead of the vertices.
    """
    rng = np.random.default_rng(22)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    cloud_x_o3d, cloud_y_o3d = points_to_open3d(x_np), points_to_open3d(y_np)
    hausdorff_o3d = max(
        np.asarray(cloud_x_o3d.compute_point_cloud_distance(cloud_y_o3d)).max(),
        np.asarray(cloud_y_o3d.compute_point_cloud_distance(cloud_x_o3d)).max(),
    )

    meshset_pml = points_to_pymeshlab(x_np)
    meshset_pml.add_mesh(ml.Mesh(np.ascontiguousarray(y_np, dtype=np.float64)))
    hausdorff_pml = max(
        float(
            meshset_pml.get_hausdorff_distance(
                sampledmesh=0, targetmesh=1, samplevert=True, samplenum=x_np.shape[0]
            )["max"]
        ),
        float(
            meshset_pml.get_hausdorff_distance(
                sampledmesh=1, targetmesh=0, samplevert=True, samplenum=y_np.shape[0]
            )["max"]
        ),
    )

    hausdorff_wp = od.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    assert np.allclose(hausdorff_wp, hausdorff_o3d, rtol=1e-5, atol=1e-5)
    assert np.allclose(hausdorff_wp, hausdorff_pml, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Hausdorff: mesh to mesh (direct port of igl::hausdorff)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere"])
def test_hausdorff_mesh_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_a_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    vertices_b_np = vertices_a_np + np.array([0.3, -0.15, 0.2])

    sqr_a_to_b_np = _igl_point_mesh_sqr_dist(vertices_a_np, vertices_b_np, faces_np)
    sqr_b_to_a_np = _igl_point_mesh_sqr_dist(vertices_b_np, vertices_a_np, faces_np)
    hausdorff_np = np.sqrt(max(sqr_a_to_b_np.max(), sqr_b_to_a_np.max()))

    vertices_b_wp = points_to_warp(vertices_b_np, mesh_wp.device)
    hausdorff_wp = od.metrics.hausdorff_mesh_to_mesh(
        mesh_wp.points, mesh_wp.indices, vertices_b_wp, mesh_wp.indices
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


# ---------------------------------------------------------------------------
# Hausdorff: point cloud to mesh (mixed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_hausdorff_points_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)

    rng = np.random.default_rng(8)
    center = vertices_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.6, size=(120, 3))).astype(np.float32)

    forward_np = np.sqrt(_igl_point_mesh_sqr_dist(points_np, vertices_np, faces_np).max())
    backward_np = np.max(KDTree(points_np.astype(np.float64)).query(vertices_np)[0])
    hausdorff_np = max(forward_np, backward_np)

    hausdorff_wp = od.metrics.hausdorff_points_to_mesh(
        points_to_warp(points_np, mesh_wp.device), mesh_wp.points, mesh_wp.indices
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_chamfer_points_to_points_empty(device: str) -> None:
    x_wp = warp_empty(0, wp.vec3, device)
    y_wp = points_to_warp(np.zeros((5, 3), dtype=np.float32), device)
    assert od.metrics.chamfer_points_to_points(x_wp, y_wp) == 0.0


def test_chamfer_points_to_points_empty_unreduced(device: str) -> None:
    x_wp = warp_empty(0, wp.vec3, device)
    y_wp = points_to_warp(np.zeros((5, 3), dtype=np.float32), device)
    forward_wp, backward_wp = od.metrics.chamfer_points_to_points(x_wp, y_wp, point_reduction=None)
    assert forward_wp.shape == (0,)
    assert backward_wp.shape == (0,)


def test_hausdorff_points_to_mesh_empty_faces(device: str) -> None:
    points_wp = points_to_warp(np.zeros((5, 3), dtype=np.float32), device)
    vertices_wp = points_to_warp(np.zeros((3, 3), dtype=np.float32), device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.metrics.hausdorff_points_to_mesh(points_wp, vertices_wp, faces_wp) == 0.0


def test_hausdorff_points_to_points_identical_is_zero(device: str) -> None:
    rng = np.random.default_rng(9)
    x_np = (rng.random((30, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    x_wp = points_to_warp(x_np, device)
    assert od.metrics.hausdorff_points_to_points(x_wp, x_wp) == 0.0


# ---------------------------------------------------------------------------
# Differentiable Chamfer losses (gradient checks)
# ---------------------------------------------------------------------------
#
# References reimplement the pytorch3d gradient convention in numpy: the
# nearest-neighbor / closest-face assignment is held constant, so the loss is a smooth
# function of the coordinates. Point-cloud gradients use the closed-form derivative of
# the squared-distance chamfer; mesh-surface gradients use central finite differences of
# the numpy point-triangle distance (there is no simple closed form across the
# face/edge/vertex regions). The assignment fed to each reference is the exact one
# ordito used, isolating the gradient computation from float32 argmin tie-breaking.

_GRAD_RTOL = 1e-4
_GRAD_ATOL = 1e-4
_FD_RTOL = 1e-2
_FD_ATOL = 2e-3


def _nn_indices_np(a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
    """Index in ``b`` of the nearest point to each row of ``a`` (brute force)."""
    squared = ((a_np[:, None, :] - b_np[None, :, :]) ** 2).sum(-1)
    return squared.argmin(1).astype(np.int64)


def _pt_tri_sq_np(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Squared distance from ``p`` to triangle ``(a, b, c)`` (numpy mirror of the kernel)."""
    q, e1, e2 = p - a, b - a, c - a
    e1e1, e1e2, e2e2 = e1 @ e1, e1 @ e2, e2 @ e2
    det = e1e1 * e2e2 - e1e2 * e1e2
    if det > e1e1 * e2e2 * 1e-6:
        e1p, e2p = e1 @ q, e2 @ q
        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det
        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            r = q - s * e1 - t * e2
            return float(r @ r)

    def seg(qq: np.ndarray, ss: np.ndarray, length_sq: float | np.ndarray) -> float:
        u = np.clip((qq @ ss) / length_sq, 0.0, 1.0)
        r = qq - u * ss
        return float(r @ r)

    d1 = seg(q, e1, e1e1)
    d2 = seg(q, e2, e2e2)
    d12 = seg(q - e1, e2 - e1, (e2 - e1) @ (e2 - e1))
    return min(d1, d2, d12)


def _surface_sq_np(
    points_np: np.ndarray, verts_np: np.ndarray, faces_np: np.ndarray, face_id_np: np.ndarray
) -> np.ndarray:
    """Per-point squared distance to the assigned triangle ``face_id[i]``."""
    out = np.empty(len(points_np), dtype=np.float64)
    for i in range(len(points_np)):
        f = int(face_id_np[i])
        a = verts_np[faces_np[3 * f + 0]]
        b = verts_np[faces_np[3 * f + 1]]
        c = verts_np[faces_np[3 * f + 2]]
        out[i] = _pt_tri_sq_np(points_np[i], a, b, c)
    return out


def _reduce_np(per_point: np.ndarray, reduction: str) -> float:
    return float(per_point.mean() if reduction == "mean" else per_point.sum())


def _fd_grad(loss_fn: Callable[[], float], arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central finite-difference gradient of ``loss_fn`` w.r.t. in-place array ``arr``."""
    grad = np.zeros_like(arr)
    flat = arr.reshape(-1)
    grad_flat = grad.reshape(-1)
    for idx in range(flat.size):
        original = flat[idx]
        flat[idx] = original + eps
        loss_plus = loss_fn()
        flat[idx] = original - eps
        loss_minus = loss_fn()
        flat[idx] = original
        grad_flat[idx] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_points_to_points_loss_grad(
    device: str, reduction: _DiffReduction, single_directional: bool
) -> None:
    rng = np.random.default_rng(20)
    x_np = (rng.random((25, 3)) * 4.0 - 2.0).astype(np.float32).astype(np.float64)
    y_np = (rng.random((18, 3)) * 4.0 - 2.0).astype(np.float32).astype(np.float64)
    n, m = len(x_np), len(y_np)

    nn_xy = _nn_indices_np(x_np, y_np)
    nn_yx = _nn_indices_np(y_np, x_np)

    # Reference loss and closed-form gradient (pytorch3d convention, fixed assignment).
    scale_f = (1.0 / n) if reduction == "mean" else 1.0
    loss_np = _reduce_np(((x_np - y_np[nn_xy]) ** 2).sum(-1), reduction)
    grad_x_np = 2.0 * scale_f * (x_np - y_np[nn_xy])
    grad_y_np = np.zeros_like(y_np)
    np.add.at(grad_y_np, nn_xy, -2.0 * scale_f * (x_np - y_np[nn_xy]))
    if not single_directional:
        scale_b = (1.0 / m) if reduction == "mean" else 1.0
        loss_np += _reduce_np(((y_np - x_np[nn_yx]) ** 2).sum(-1), reduction)
        grad_y_np += 2.0 * scale_b * (y_np - x_np[nn_yx])
        np.add.at(grad_x_np, nn_yx, -2.0 * scale_b * (y_np - x_np[nn_yx]))

    x_wp = wp.array(x_np, dtype=wp.vec3, device=device, requires_grad=True)
    y_wp = wp.array(y_np, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    loss_wp = od.metrics.chamfer_points_to_points_loss(
        x_wp, y_wp, tape=tape, point_reduction=reduction, single_directional=single_directional
    )
    tape.backward(loss=loss_wp)

    assert x_wp.grad is not None
    assert y_wp.grad is not None
    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)
    assert np.allclose(x_wp.grad.numpy(), grad_x_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)
    assert np.allclose(y_wp.grad.numpy(), grad_y_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)


@pytest.mark.parity(
    "chamfer_points_to_points_loss",
    "pytorch3d",
    benchmarked=False,
    reason="chamfer_points_to_points_loss carries no benchmark group of its own (only its "
    "non-differentiable sibling chamfer_points_to_points does); this is a correctness-only "
    "comparison against pytorch3d's autograd, not a timed one.",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_points_to_points_loss_grad_matches_pytorch3d(
    device: str, reduction: _DiffReduction, single_directional: bool
) -> None:
    """
    Class A: pytorch3d's own autograd through ``loss.chamfer_distance`` is the gradient oracle.

    Unlike the closed-form NumPy check above, this backpropagates through pytorch3d's own
    nearest-neighbor assignment and reduction rather than a hand-derived formula. No restriction to
    a single direction is needed here (contrast the mesh comparison below): both directions are
    plain point-to-point nearest neighbor on both sides, which is exactly what pytorch3d's
    ``single_directional`` keyword already means, so the two definitions agree completely rather
    than needing a subtraction to isolate a comparable half. Measured close to float32 machine
    epsilon on the loss and both gradients (max absolute gradient difference 4.77e-07 here).
    """
    rng = np.random.default_rng(25)
    x_np = (rng.random((25, 3)) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((18, 3)) * 4.0 - 2.0).astype(np.float32)

    x_t = torch.tensor(x_np, device=device, requires_grad=True)
    y_t = torch.tensor(y_np, device=device, requires_grad=True)
    loss_p3d, _ = cast(
        "tuple[torch.Tensor, object]",
        p3d_loss.chamfer_distance(
            x_t[None], y_t[None], point_reduction=reduction, single_directional=single_directional
        ),
    )
    loss_p3d.backward()

    x_wp = wp.array(x_np, dtype=wp.vec3, device=device, requires_grad=True)
    y_wp = wp.array(y_np, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    loss_wp = od.metrics.chamfer_points_to_points_loss(
        x_wp, y_wp, tape=tape, point_reduction=reduction, single_directional=single_directional
    )
    tape.backward(loss=loss_wp)

    assert x_wp.grad is not None
    assert y_wp.grad is not None
    assert x_t.grad is not None
    assert y_t.grad is not None
    assert np.allclose(loss_wp.numpy()[0], loss_p3d.item(), rtol=1e-5, atol=1e-6)
    assert np.allclose(x_wp.grad.numpy(), x_t.grad.cpu().numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(y_wp.grad.numpy(), y_t.grad.cpu().numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_points_to_mesh_loss_grad(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], reduction: _DiffReduction, single_directional: bool
) -> None:
    mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.device
    verts_np = np.asarray(mesh_tm.vertices, dtype=np.float32).astype(np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1)

    rng = np.random.default_rng(21)
    center = verts_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.5, size=(10, 3))).astype(np.float32).astype(np.float64)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=True)
    verts_wp = wp.array(verts_np, dtype=wp.vec3, device=device, requires_grad=True)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    # Fixed assignments exactly as the loss function computes them.
    face_id = od.proximity.closest_point_on_mesh(verts_wp, faces_wp, points_wp)[2].numpy()
    nn_vp = od.neighbors.query_nearest(points_wp, verts_wp, k=1)[0].numpy()

    def evaluate_loss_np() -> float:
        total = _reduce_np(_surface_sq_np(points_np, verts_np, faces_np, face_id), reduction)
        if not single_directional:
            total += _reduce_np(((verts_np - points_np[nn_vp]) ** 2).sum(-1), reduction)
        return total

    loss_np = evaluate_loss_np()
    grad_points_np = _fd_grad(evaluate_loss_np, points_np)
    grad_verts_np = _fd_grad(evaluate_loss_np, verts_np)

    tape = wp.Tape()
    loss_wp = od.metrics.chamfer_points_to_mesh_loss(
        points_wp,
        verts_wp,
        faces_wp,
        tape=tape,
        point_reduction=reduction,
        single_directional=single_directional,
    )
    tape.backward(loss=loss_wp)

    assert points_wp.grad is not None
    assert verts_wp.grad is not None
    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.allclose(points_wp.grad.numpy(), grad_points_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.allclose(verts_wp.grad.numpy(), grad_verts_np, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parity(
    "chamfer_points_to_mesh_loss",
    "pytorch3d",
    benchmarked=False,
    reason="chamfer_points_to_mesh_loss carries no benchmark group of its own (only its "
    "non-differentiable sibling chamfer_points_to_mesh does), so this is correctness-only, and "
    "single_directional=True only: pytorch3d's point_mesh_face_distance sums point-to-face with "
    "face-to-point (see test_chamfer_points_to_mesh_matches_pytorch3d), where ordito's backward "
    "direction is vertex-to-nearest-point, so the two-sided loss and its gradient are not "
    "comparable. The forward-only case needs no numpy subtraction trick to isolate, unlike the "
    "value-only comparison above: pytorch3d's own point_face_distance primitive already returns "
    "just the point-to-triangle term, with correct gradients w.r.t. both the points and the mesh "
    "vertices, so it *is* the forward-only oracle directly.",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_chamfer_points_to_mesh_loss_grad_matches_pytorch3d(
    device: str, reduction: _DiffReduction
) -> None:
    """
    Class B, forward-only: pytorch3d's ``point_mesh_distance.point_face_distance`` autograd.

    ``point_mesh_face_distance`` (used by the value-only comparison above) always sums this term
    with its face-to-point half, which has no ordito counterpart -- see that test's own docstring.
    Its lower-level primitive, ``point_face_distance``, returns only the per-point squared
    point-to-triangle distance the sum is built from, still differentiable w.r.t. both the query
    points and the mesh vertices (through ``verts_packed()[faces_packed()]``), which is exactly
    what ``chamfer_points_to_mesh_loss(single_directional=True)`` computes -- so this needs no
    subtraction, unlike the two-sided value comparison. Measured close to float32 machine epsilon
    on the loss and both gradients (max absolute gradient difference 1.48e-06 here).
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    verts_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    rng = np.random.default_rng(24)
    center = verts_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.5, size=(20, 3))).astype(np.float32)

    verts_t = torch.tensor(verts_np, device=device, requires_grad=True)
    faces_t = torch.tensor(faces_np, device=device)
    points_t = torch.tensor(points_np, device=device, requires_grad=True)
    meshes_p3d = p3d_structures.Meshes(verts=[verts_t], faces=[faces_t])
    pcls_p3d = p3d_structures.Pointclouds(points=[points_t])

    # The packed representation `point_mesh_face_distance` builds internally (see its source),
    # taken apart here to reach its forward-only primitive instead of the point+face sum.
    points_packed = pcls_p3d.points_packed()
    points_first_idx = pcls_p3d.cloud_to_packed_first_idx()
    max_points = pcls_p3d.num_points_per_cloud().max().item()
    verts_packed = meshes_p3d.verts_packed()
    assert verts_packed is not None
    tris = verts_packed[meshes_p3d.faces_packed()]
    tris_first_idx = meshes_p3d.mesh_to_faces_packed_first_idx()
    point_to_face = p3d_point_mesh_distance.point_face_distance(
        points_packed, points_first_idx, tris, tris_first_idx, max_points
    )
    loss_p3d = point_to_face.mean() if reduction == "mean" else point_to_face.sum()
    loss_p3d.backward()

    verts_wp = wp.array(verts_np, dtype=wp.vec3, device=device, requires_grad=True)
    faces_wp = wp.array(faces_np.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    loss_wp = od.metrics.chamfer_points_to_mesh_loss(
        points_wp, verts_wp, faces_wp, tape=tape, point_reduction=reduction, single_directional=True
    )
    tape.backward(loss=loss_wp)

    assert points_wp.grad is not None
    assert verts_wp.grad is not None
    assert points_t.grad is not None
    assert verts_t.grad is not None
    assert np.allclose(loss_wp.numpy()[0], loss_p3d.item(), rtol=1e-5, atol=1e-6)
    assert np.allclose(points_wp.grad.numpy(), points_t.grad.cpu().numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(verts_wp.grad.numpy(), verts_t.grad.cpu().numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parity(
    "chamfer_mesh_to_mesh_loss",
    "numpy",
    benchmarked=False,
    reason="the reference is a central finite difference of the same loss, which costs two forward "
    "evaluations per coordinate -- 6n for an n-vertex pair -- so timing it would price the "
    "oracle's own cost rather than a competing autodiff. No installed library differentiates a "
    "mesh-to-mesh Chamfer at all; the gradient values are what is comparable.",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_mesh_to_mesh_loss_grad(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], reduction: _DiffReduction, single_directional: bool
) -> None:
    mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.device
    verts_a_np = np.asarray(mesh_tm.vertices, dtype=np.float32).astype(np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1)
    verts_b_np = (verts_a_np + np.array([0.3, -0.15, 0.2])).astype(np.float32).astype(np.float64)

    verts_a_wp = wp.array(verts_a_np, dtype=wp.vec3, device=device, requires_grad=True)
    verts_b_wp = wp.array(verts_b_np, dtype=wp.vec3, device=device, requires_grad=True)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    face_id_ab = od.proximity.closest_point_on_mesh(verts_b_wp, faces_wp, verts_a_wp)[2].numpy()
    face_id_ba = od.proximity.closest_point_on_mesh(verts_a_wp, faces_wp, verts_b_wp)[2].numpy()

    def evaluate_loss_np() -> float:
        total = _reduce_np(_surface_sq_np(verts_a_np, verts_b_np, faces_np, face_id_ab), reduction)
        if not single_directional:
            total += _reduce_np(
                _surface_sq_np(verts_b_np, verts_a_np, faces_np, face_id_ba), reduction
            )
        return total

    loss_np = evaluate_loss_np()
    grad_a_np = _fd_grad(evaluate_loss_np, verts_a_np)
    grad_b_np = _fd_grad(evaluate_loss_np, verts_b_np)

    tape = wp.Tape()
    loss_wp = od.metrics.chamfer_mesh_to_mesh_loss(
        verts_a_wp,
        faces_wp,
        verts_b_wp,
        faces_wp,
        tape=tape,
        point_reduction=reduction,
        single_directional=single_directional,
    )
    tape.backward(loss=loss_wp)

    assert verts_a_wp.grad is not None
    assert verts_b_wp.grad is not None
    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.allclose(verts_a_wp.grad.numpy(), grad_a_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    if not single_directional:
        assert np.allclose(verts_b_wp.grad.numpy(), grad_b_np, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_chamfer_losses_match_numpy_on_both_devices(kernel_device: str) -> None:
    """
    Pin both chamfer loss reductions on **both** devices against a closed-form NumPy sum.

    ``wp.launch_tiled`` runs exactly one lane per block on Warp 1.18's CPU backend, so the
    block-wide ``wp.tile_sum`` these losses used to perform accumulated one point per 64-point tile
    there and returned a loss roughly 64x too small -- measured 0.43 absolute on this size of cloud.
    Each term now has a lane-free ``*_sliced`` kernel for CPU and keeps the ``*_tiled`` one on CUDA,
    so the parametrization is what covers both: dropping either device leaves a kernel untested.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(91)
    x_np = rng.standard_normal((500, 3))
    y_np = rng.standard_normal((300, 3)) * 0.7
    x_wp = points_to_warp(x_np, kernel_device)
    y_wp = points_to_warp(y_np, kernel_device)

    # Bidirectional mean-reduced squared chamfer, exactly what the default arguments compute.
    squared_np = ((x_np[:, None, :] - y_np[None, :, :]) ** 2).sum(-1)
    loss_np = squared_np.min(1).mean() + squared_np.min(0).mean()
    loss_wp = od.metrics.chamfer_points_to_points_loss(x_wp, y_wp)
    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=1e-4, atol=1e-4)

    # Single-directional points-to-mesh: the reference is the distance to the nearest triangle,
    # which for a convex mesh sampled outside it is the distance to its surface.
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    surface_loss_wp = od.metrics.chamfer_points_to_mesh_loss(
        x_wp, mesh_wp.points, mesh_wp.indices, single_directional=True
    )
    _closest_np, distance_np, _face_np = tm_proximity.closest_point(mesh_tm, x_np)
    assert np.allclose(surface_loss_wp.numpy()[0], (distance_np**2).mean(), rtol=1e-3, atol=1e-3)


def test_chamfer_loss_no_tape_has_value_but_no_grad(device: str) -> None:
    rng = np.random.default_rng(22)
    x_np = (rng.random((12, 3)) * 2.0 - 1.0).astype(np.float32)
    y_np = (rng.random((9, 3)) * 2.0 - 1.0).astype(np.float32)
    x_wp = points_to_warp(x_np, device)
    y_wp = points_to_warp(y_np, device)

    loss_wp = od.metrics.chamfer_points_to_points_loss(x_wp, y_wp)
    reduced = od.metrics.chamfer_points_to_points(x_wp, y_wp)
    assert np.allclose(loss_wp.numpy()[0], reduced, rtol=1e-5, atol=1e-5)


def test_chamfer_loss_rejects_max_reduction(device: str) -> None:
    x_wp = points_to_warp(np.zeros((3, 3), dtype=np.float32), device)
    y_wp = points_to_warp(np.ones((3, 3), dtype=np.float32), device)
    with pytest.raises(ValueError, match="mean"):
        od.metrics.chamfer_points_to_points_loss(x_wp, y_wp, point_reduction="max")  # pyright: ignore[reportArgumentType]  # deliberately off-menu


def test_chamfer_points_to_points_loss_empty(device: str) -> None:
    x_wp = warp_empty(0, wp.vec3, device)
    y_wp = points_to_warp(np.zeros((4, 3), dtype=np.float32), device)
    loss_wp = od.metrics.chamfer_points_to_points_loss(x_wp, y_wp)
    assert loss_wp.shape == (1,)
    assert float(loss_wp.numpy()[0]) == 0.0


@wp.kernel  # pyright: ignore[reportUntypedFunctionDecorator]  # wp.kernel has no return annotation
def _probe_point_triangle_sq_dist(
    query: wp.array[wp.vec3], triangle: wp.array[wp.vec3], out_dist_sq: wp.array[wp.float32]
) -> None:
    # Kernel scope: Warp's stubs type an element read as an array.
    out_dist_sq[0] = kernel_metrics.point_triangle_sq_dist(  # pyright: ignore[reportIndexIssue]
        query[0],  # pyright: ignore[reportArgumentType]
        triangle[0],  # pyright: ignore[reportArgumentType]
        triangle[1],  # pyright: ignore[reportArgumentType]
        triangle[2],  # pyright: ignore[reportArgumentType]
    )


@pytest.mark.parametrize(
    ("name", "triangle"),
    [
        ("well_shaped", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ("two_coincident", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ("all_coincident", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ("collinear", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ],
)
def test_point_triangle_sq_dist_grad_is_finite_on_degenerate_faces(
    device: str, name: str, triangle: list[list[float]]
) -> None:
    """
    Not a library comparison: no reference exposes the point-triangle primitive's adjoint.

    ``project_segment_sq_dist`` divides by the segment's squared length, which is exactly zero
    when a triangle has two coincident vertices. The *value* survives that -- ``wp.min`` over a
    ``wp.vec3`` reduces with ``<``, which a NaN always loses, so the two sound edges win -- and
    only the adjoint saw the ``0 / 0``. Since the chamfer loss accumulates with an atomic add, one
    such face in query range returned NaN for the caller's entire gradient, not just its own term.

    The guard is exact rather than a tolerance: a zero-length segment is the single point at the
    origin, so ``q - s * segment`` is ``q`` for every ``s`` and the answer is ``length_sq(q)``
    however the parameter is chosen. Values are asserted unchanged for that reason, alongside
    finiteness.

    A degenerate triangle is a *kink* of the function, so no two-sided derivative exists there and
    central differences do not apply; what is asserted instead is the strongest statement available
    at a kink -- the adjoint lies within the bracket spanned by the two one-sided derivatives. On
    the well-shaped arm, which is differentiable, that bracket collapses and the check is exact.
    """
    triangle_np = np.array(triangle, dtype=np.float64)
    query_np = np.array([[0.3, 0.3, 1.0]], dtype=np.float64)

    def value(vertices_np: np.ndarray) -> float:
        triangle_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device)
        query_wp = wp.array(query_np.astype(np.float32), dtype=wp.vec3, device=device)
        out_wp = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(
            _probe_point_triangle_sq_dist,
            dim=1,
            inputs=[query_wp, triangle_wp, out_wp],
            device=device,
        )
        return float(out_wp.numpy()[0])

    triangle_wp = wp.array(
        triangle_np.astype(np.float32), dtype=wp.vec3, device=device, requires_grad=True
    )
    query_wp = wp.array(
        query_np.astype(np.float32), dtype=wp.vec3, device=device, requires_grad=True
    )
    out_wp = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            _probe_point_triangle_sq_dist,
            dim=1,
            inputs=[query_wp, triangle_wp, out_wp],
            device=device,
        )
    tape.backward(loss=out_wp)
    assert triangle_wp.grad is not None
    assert query_wp.grad is not None
    gradient_np = triangle_wp.grad.numpy().astype(np.float64)

    assert np.isfinite(gradient_np).all()
    assert np.isfinite(query_wp.grad.numpy()).all()

    # The guard changes no value: the closest point is unmoved in every arm.
    expected = {
        "well_shaped": 1.0,
        "two_coincident": 1.09,
        "all_coincident": 1.18,
        "collinear": 1.09,
    }[name]
    assert value(triangle_np) == pytest.approx(expected, abs=1e-6)

    step = 1e-4
    base = value(triangle_np)
    for vertex in range(3):
        for axis in range(3):
            plus_np = triangle_np.copy()
            plus_np[vertex, axis] += step
            minus_np = triangle_np.copy()
            minus_np[vertex, axis] -= step
            forward = (value(plus_np) - base) / step
            backward = (base - value(minus_np)) / step
            low, high = min(forward, backward), max(forward, backward)
            assert low - 2e-3 <= gradient_np[vertex, axis] <= high + 2e-3


def _sliced_term_gradients(device: str, term: str, n_slices: int) -> tuple[float, list[np.ndarray]]:
    """Loss and input gradients of one sliced chamfer term, launched at ``n_slices`` threads."""
    rng = np.random.default_rng(31)
    points_np = rng.normal(size=(5, 3)).astype(np.float32)
    targets_np = rng.normal(size=(12, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=True)
    targets_wp = wp.array(targets_np, dtype=wp.vec3, device=device, requires_grad=True)
    loss_wp = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        if term == "nn":
            nearest_wp = wp.array(
                _nn_indices_np(points_np, targets_np).astype(np.int32), device=device
            )
            wp.launch(
                kernel_metrics.chamfer_nn_term_sliced,
                dim=n_slices,
                inputs=[points_wp, targets_wp, nearest_wp, wp.float32(0.5), n_slices],
                outputs=[loss_wp],
                device=device,
            )
        else:
            faces_wp = wp.array(np.arange(12, dtype=np.int32), device=device)
            face_id_wp = wp.array(np.array([0, 1, 2, 3, -1], dtype=np.int32), device=device)
            wp.launch(
                kernel_metrics.chamfer_surface_term_sliced,
                dim=n_slices,
                inputs=[points_wp, targets_wp, faces_wp, face_id_wp, wp.float32(0.5), n_slices],
                outputs=[loss_wp],
                device=device,
            )
    tape.backward(loss=loss_wp)
    assert points_wp.grad is not None
    assert targets_wp.grad is not None
    return float(loss_wp.numpy()[0]), [points_wp.grad.numpy(), targets_wp.grad.numpy()]


@pytest.mark.parametrize("term", ["nn", "surface"])
def test_sliced_chamfer_terms_grad_with_more_slices_than_points(device: str, term: str) -> None:
    """
    Ordito against ordito: a sliced term launched with empty slices against one serial slice.

    Five points over 45 slices leaves 40 threads with no element. Warp's backward pass reverses a
    dynamic ``range`` through ``iter_reverse``, which turns such an empty range into one iteration
    at its start, past the end of the points -- so without the kernels' empty-slice guard those
    threads read out of bounds and added a gradient (11x the true one on the nn term). The single
    slice has no empty thread and is the reference; the nn term's is also checked against its
    closed form, which is what shows the reference itself is right. ``slice_count`` never launches
    more slices than points, so no public call reaches this -- the guard is for any other caller.
    """
    loss_serial, grads_serial = _sliced_term_gradients(device, term, 1)
    loss_wide, grads_wide = _sliced_term_gradients(device, term, 45)
    assert loss_wide == pytest.approx(loss_serial, rel=1e-6)
    for wide, serial in zip(grads_wide, grads_serial, strict=True):
        assert np.abs(serial).max() > 0.0
        np.testing.assert_allclose(wide, serial, rtol=1e-5, atol=1e-6)
    if term == "nn":
        rng = np.random.default_rng(31)
        points_np = rng.normal(size=(5, 3)).astype(np.float64)
        targets_np = rng.normal(size=(12, 3)).astype(np.float64)
        nearest = _nn_indices_np(points_np, targets_np)
        grad_points_np = points_np - targets_np[nearest]
        np.testing.assert_allclose(grads_serial[0], grad_points_np, rtol=1e-5, atol=1e-5)
