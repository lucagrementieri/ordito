"""
Regression tests for ``ordito.neighbors`` ball / k-nearest query APIs.

Against SciPy ``KDTree`` (BVH and HashGrid backends), and ``igl.knn`` as a second exact k-NN.
"""

from __future__ import annotations

import heapq
import math
import warnings
from collections import deque
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import igl
import numpy as np
import open3d as o3d
import pytest
import pytorch3d.ops as p3d_ops
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree

import ordito as od
import ordito.typing as odt
from ordito.kernels import neighbors as kernel_neighbors
from tests.conversions import (
    meshlib_indices_to_numpy,
    meshlib_scalars_to_numpy,
    numpy_to_meshlib_bitset,
    numpy_to_warp,
    points_to_meshlib,
    points_to_open3d,
    points_to_torch,
    points_to_warp,
    trimesh_to_meshlib,
    warp_empty,
)


def test_mesh_from_points_answers_the_nearest_point(device: str) -> None:
    """
    Class A against ``KDTree``: the collapsed-triangle mesh's closest face is the nearest point.

    Reached through ``query_nearest(k=1, backend="bvh")``, whose search the mesh is. Queries sit
    up to a cloud diameter off the cloud, which is the regime the mesh descent exists for; the face
    index is the point index, so indices are compared as well as distances. The empty cloud
    raises rather than building a triangle-free ``wp.Mesh`` (``.claude/CLAUDE.md`` section 12.1).
    """
    rng = np.random.default_rng(3)
    points = rng.random((500, 3), dtype=np.float32)
    queries = np.ascontiguousarray(rng.random((80, 3), dtype=np.float32) * 3.0 - 1.0)
    points_wp = points_to_warp(points, device)
    mesh = od.neighbors.mesh_from_points(points_wp)
    assert np.array_equal(mesh.indices.numpy(), np.arange(3 * points.shape[0]) // 3)
    distances_kd, indices_kd = KDTree(points).query(queries)
    indices_wp, distances_wp = od.neighbors.query_nearest(
        points_wp, points_to_warp(queries, device), k=1, backend="bvh"
    )
    assert np.array_equal(indices_wp.numpy(), indices_kd)
    assert np.allclose(distances_wp.numpy(), distances_kd, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="at least one point"):
        od.neighbors.mesh_from_points(points_to_warp(np.zeros((0, 3), np.float32), device))


def test_hashgrid_from_points_sizes_its_default_grid_from_the_point_count(device: str) -> None:
    """
    Not a library comparison for the resolution, which no reference has; Class A for the answer.

    ``grid_bins=None`` keeps 128 bins up to ``128 ** 3 / 2`` points and grows past it (the next
    multiple of 32 whose cube holds two bins a point), while an explicit resolution is honoured
    as given. The resolution is read off Warp's own ``HashGrid`` dimensions. The answer must not
    depend on it: a three-bin grid folds every cell of the cloud onto 27 buckets, and its ball
    counts still equal ``KDTree``'s and the default grid's.
    """
    at_threshold = od.neighbors.hashgrid_from_points(
        points_to_warp(np.zeros((128**3 // 2, 3), np.float32), device), 1.0
    )
    past_threshold = od.neighbors.hashgrid_from_points(
        points_to_warp(np.zeros((128**3 // 2 + 1, 3), np.float32), device), 1.0
    )
    assert (at_threshold._dim_x, at_threshold._dim_y, at_threshold._dim_z) == (128, 128, 128)  # pyright: ignore[reportPrivateUsage]
    assert (past_threshold._dim_x, past_threshold._dim_y, past_threshold._dim_z) == (160,) * 3  # pyright: ignore[reportPrivateUsage]

    rng = np.random.default_rng(11)
    points = rng.random((2_000, 3), dtype=np.float32) * 4.0
    radius = 0.3
    counts_kd = KDTree(points).query_ball_point(points, radius, return_length=True)
    assert np.ptp(counts_kd) > 0
    points_wp = points_to_warp(points, device)
    for grid_bins in (None, 3):
        grid = od.neighbors.hashgrid_from_points(points_wp, radius, grid_bins)
        assert grid._dim_x == (128 if grid_bins is None else grid_bins)  # pyright: ignore[reportPrivateUsage]
        counts_wp = od.neighbors.query_ball_count(points_wp, points_wp, radius, grid_bins=grid_bins)
        assert np.array_equal(counts_wp.numpy(), counts_kd)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_single(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    query = points[0].copy()
    radius = 0.5

    query_indices_np = np.asarray(kdtree.query_ball_point(query, radius, return_sorted=False))
    query_distances_np = np.linalg.norm(points[query_indices_np] - query, axis=1)
    order = np.argsort(query_distances_np)
    query_indices_np = query_indices_np[order]
    query_distances_np = query_distances_np[order]

    points_wp = points_to_warp(points, device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = partial(od.neighbors.query_ball, backend=backend)

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    assert np.array_equal(
        np.sort(query_indices_unsorted_wp.numpy()), np.sort(query_indices_wp.numpy())
    )
    assert np.allclose(
        np.sort(query_distances_unsorted_wp.numpy()),
        np.sort(query_distances_wp.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty_ball(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    query = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    radius = 0.5

    points_wp = points_to_warp(points, device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = partial(od.neighbors.query_ball, backend=backend)

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )
    assert query_indices_wp.shape == (0,)
    assert query_distances_wp.shape == (0,)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity("query_ball_bvh", "scipy")
@pytest.mark.parity("query_ball_hashgrid", "scipy")
def test_query_ball_batch(device: str, backend: Literal["bvh", "hashgrid"]):
    """
    Class B: per-query neighbour lists against ``KDTree.query_ball_point``, sorted by distance.

    scipy returns each list unordered, so the named transform sorts it by distance to match
    ``return_sorted=True``; the ``return_sorted=False`` call is then compared against ordito's
    own sorted answer, since only the *set* is defined there. Both backends, three queries.
    """
    rng = np.random.default_rng(1)
    points = rng.random((50, 3), dtype=np.float32) * 3.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]].copy()
    radius = 0.5

    query_indices_np = [
        np.asarray(indices)
        for indices in kdtree.query_ball_point(queries, radius, return_sorted=False)
    ]
    query_distances_np = [
        np.linalg.norm(points[indices] - query, axis=1)
        for indices, query in zip(query_indices_np, queries, strict=False)
    ]
    orders = [np.argsort(distances) for distances in query_distances_np]
    query_indices_np = [
        indices[order] for indices, order in zip(query_indices_np, orders, strict=False)
    ]
    query_distances_np = [
        distances[order] for distances, order in zip(query_distances_np, orders, strict=False)
    ]

    points_wp = points_to_warp(points, device)
    query_wp = points_to_warp(queries, device)
    query_ball = partial(od.neighbors.query_ball, backend=backend)

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    for single_query_indices_wp, single_query_indices_np in zip(
        query_indices_wp, query_indices_np, strict=False
    ):
        assert np.array_equal(single_query_indices_wp.numpy(), single_query_indices_np)
    for single_query_distances_wp, single_query_distances_np in zip(
        query_distances_wp, query_distances_np, strict=False
    ):
        assert np.allclose(
            single_query_distances_wp.numpy(), single_query_distances_np, rtol=1e-5, atol=1e-5
        )

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    for single_query_indices_unsorted_wp, single_query_indices_wp in zip(
        query_indices_unsorted_wp, query_indices_wp, strict=False
    ):
        assert np.array_equal(
            np.sort(single_query_indices_unsorted_wp.numpy()),
            np.sort(single_query_indices_wp.numpy()),
        )
    for single_query_distances_unsorted_wp, single_query_distances_wp in zip(
        query_distances_unsorted_wp, query_distances_wp, strict=False
    ):
        assert np.allclose(
            np.sort(single_query_distances_unsorted_wp.numpy()),
            np.sort(single_query_distances_wp.numpy()),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_count_matches_scipy_and_the_list_form(
    device: str, backend: Literal["bvh", "hashgrid"]
) -> None:
    """
    Class A: the per-query counts equal ``KDTree.query_ball_point`` lengths, on both backends.

    Also cross-checked against the ``with_offsets`` form of the same query, since the count is
    exactly what that function's offsets differ by -- a count kernel that disagreed with the
    gather it sizes is the failure worth catching. Measured counts on this cloud: 2, 3, 3, 10, 7,
    so no query is empty and none holds the whole cloud.
    """
    rng = np.random.default_rng(1)
    points_np = rng.random((200, 3), dtype=np.float32) * 3.0
    queries_np = points_np[[10, 20, 30, 55, 120]].copy()
    radius = 0.5

    points_wp = points_to_warp(points_np, device)
    queries_wp = points_to_warp(queries_np, device)
    query_ball_count = partial(od.neighbors.query_ball_count, backend=backend)
    query_ball_with_offsets = partial(od.neighbors.query_ball_with_offsets, backend=backend)

    counts_wp = query_ball_count(points_wp, queries_wp, radius)
    counts_np = np.array(
        [
            len(indices)
            for indices in cast(
                "list[list[int]]", KDTree(points_np).query_ball_point(queries_np, radius)
            )
        ],
        dtype=np.int32,
    )

    assert counts_np.min() > 0
    assert counts_np.max() < points_np.shape[0]
    assert np.array_equal(counts_wp.numpy(), counts_np)

    indices_wp, _distances_wp, offsets_wp = query_ball_with_offsets(points_wp, queries_wp, radius)
    assert np.array_equal(np.diff(offsets_wp.numpy()), counts_np)
    assert indices_wp.size == int(counts_np.sum())


@pytest.mark.parametrize("copy", [False, True])
def test_query_ball_is_its_packed_form_split(device: str, copy: bool) -> None:
    """
    Ordito against ordito: the list form equals ``query_ball_with_offsets`` split per query.

    The packed form carries the scipy oracle (``test_query_ball_*``); this pins the list to it, and
    ``copy`` to views of the packed buffers or to independent storage.
    """
    rng = np.random.default_rng(12)
    points_wp = points_to_warp(rng.random((300, 3)).astype(np.float32), device)
    queries_wp = points_to_warp(rng.random((25, 3)).astype(np.float32), device)
    radius = 0.2

    indices_wp, distances_wp = od.neighbors.query_ball(points_wp, queries_wp, radius, copy=copy)
    flat_wp, flat_distances_wp, offsets_wp = od.neighbors.query_ball_with_offsets(
        points_wp, queries_wp, radius
    )
    bounds_np = offsets_wp.numpy()
    assert len(indices_wp) == len(distances_wp) == bounds_np.size - 1 == 25
    assert int(bounds_np[-1]) > 25  # non-vacuous: most queries find neighbours
    for query, (index_wp, distance_wp) in enumerate(zip(indices_wp, distances_wp, strict=True)):
        begin, end = int(bounds_np[query]), int(bounds_np[query + 1])
        assert np.array_equal(index_wp.numpy(), flat_wp.numpy()[begin:end])
        assert np.array_equal(distance_wp.numpy(), flat_distances_wp.numpy()[begin:end])
        assert (index_wp._ref is None) == copy  # pyright: ignore[reportPrivateUsage]


def test_query_bvh_ball_matches_a_brute_force_ball_overlap(device: str) -> None:
    """
    Class A: the broad-phase hits are exactly the boxes the ball of ``radius`` reaches.

    Not a library comparison for the *shape*: no reference binds a broad-phase ball query over
    arbitrary bounds, so the oracle is the exact point-to-AABB squared distance
    ``sum(max(lower - q, q - upper, 0)**2) <= radius**2``, which is what
    ``wp.bvh_query_sphere``'s node test computes.

    The assert that matters is the *strict* containment against the cube query on the same bounds
    and the same radius: the ball's hits must be a proper subset, or this function is returning the
    cube's answer under a ball's name -- which is precisely how a query that silently ignored the
    radius would pass an exactness check against itself.
    """
    rng = np.random.default_rng(4)
    lower_np = (rng.random((40, 3)) * 2.0).astype(np.float32)
    upper_np = (lower_np + rng.random((40, 3)) * 0.3).astype(np.float32)
    queries_np = (rng.random((6, 3)) * 2.0).astype(np.float32)
    radius = 0.25
    n_queries = queries_np.shape[0]

    bvh = od.neighbors.bvh_from_bounds(
        points_to_warp(lower_np, device), points_to_warp(upper_np, device)
    )
    queries_wp = points_to_warp(queries_np, device)
    indices_wp, offsets_wp = od.neighbors.query_bvh_ball(bvh, queries_wp, radius)
    indices_np = indices_wp.numpy()
    bounds_np = offsets_wp.numpy()

    assert bounds_np.shape == (n_queries + 1,)
    assert indices_np.size > 0
    assert int(bounds_np[-1]) == indices_np.size

    n_matched = 0
    for query_index, query_np in enumerate(queries_np):
        gap_np = np.maximum(np.maximum(lower_np - query_np, query_np - upper_np), 0.0)
        within_np = np.flatnonzero((gap_np * gap_np).sum(axis=1) <= radius * radius)
        hits_np = indices_np[bounds_np[query_index] : bounds_np[query_index + 1]]
        assert np.array_equal(np.sort(hits_np), within_np)
        n_matched += within_np.size > 0
    assert n_matched >= 3

    # The ball is strictly inside the cube of the same half extent, so its hits must be a proper
    # subset of what ``query_bvh_box`` returns for that cube -- a query ignoring the radius would
    # return the cube's answer and still pass the exactness check above, since it would then be
    # compared against a cube oracle it happens to match.
    cube_lower_wp = points_to_warp(queries_np - radius, device)
    cube_upper_wp = points_to_warp(queries_np + radius, device)
    cube_indices_wp, _cube_offsets_wp = od.neighbors.query_bvh_box(
        bvh, cube_lower_wp, cube_upper_wp
    )
    assert int(indices_np.size) < cube_indices_wp.size


def test_query_bvh_ball_degenerate_inputs(device: str) -> None:
    """
    Not a library comparison: an empty query set and a query that hits nothing.

    The two early returns, mirroring ``test_query_bvh_box_degenerate_inputs``: a zero-hit query
    must still produce the total-terminated offsets shape the general path does, or a caller
    reading the trailing total breaks.
    """
    lower_np = np.zeros((4, 3), dtype=np.float32)
    upper_np = np.full((4, 3), 0.1, dtype=np.float32)
    bvh = od.neighbors.bvh_from_bounds(
        points_to_warp(lower_np, device), points_to_warp(upper_np, device)
    )

    empty_indices_wp, empty_offsets_wp = od.neighbors.query_bvh_ball(
        bvh, warp_empty(0, wp.vec3, device), 0.25
    )
    assert empty_indices_wp.shape == (0,)
    assert np.array_equal(empty_offsets_wp.numpy(), np.zeros(1, np.int32))

    far_wp = points_to_warp(np.array([[99.0, 99.0, 99.0]], dtype=np.float32), device)
    miss_indices_wp, miss_offsets_wp = od.neighbors.query_bvh_ball(bvh, far_wp, 0.25)
    assert miss_indices_wp.shape == (0,)
    assert np.array_equal(miss_offsets_wp.numpy(), np.zeros(2, np.int32))


@pytest.mark.parity("query_bvh_box", "open3d")
def test_query_bvh_box_matches_exact_containment(device: str) -> None:
    """
    Class A: on a point BVH the hits are exactly the points inside each query box.

    A degenerate leaf bound intersects the query box iff the point is in it, so there is no
    broad-phase superset here and the comparison is an equality against two independent exact
    answers: NumPy's ``lower <= p <= upper`` and open3d's
    ``AxisAlignedBoundingBox.get_point_indices_within_bounding_box``. Both share Warp's
    **inclusive** convention, which is the one thing a caller can get wrong here -- measured on Warp
    1.18 and open3d 0.19, a point exactly on a face is inside the box for both -- so the last case
    below constructs one point on the lower face and one on the upper face rather than trusting
    random data to land there.

    Also pins the total-terminated offsets shape the ball sibling's test pins.
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((400, 3)).astype(np.float32)
    centers_np = rng.random((12, 3)).astype(np.float32)
    lower_np = np.ascontiguousarray(centers_np - 0.12, dtype=np.float32)
    upper_np = np.ascontiguousarray(centers_np + 0.18, dtype=np.float32)
    n_queries = centers_np.shape[0]

    points_wp = points_to_warp(points_np, device)
    bvh = od.neighbors.bvh_from_points(points_wp)
    indices_wp, offsets_wp = od.neighbors.query_bvh_box(
        bvh, points_to_warp(lower_np, device), points_to_warp(upper_np, device)
    )
    indices_np = indices_wp.numpy()
    offsets_np = offsets_wp.numpy()
    assert offsets_np.shape == (n_queries + 1,)
    assert int(offsets_np[-1]) == indices_np.size

    cloud_o3d = points_to_open3d(points_np.astype(np.float64))
    n_nonempty = 0
    for query_index in range(n_queries):
        box_lower_np, box_upper_np = lower_np[query_index], upper_np[query_index]
        inside_np = np.flatnonzero(
            np.all((points_np >= box_lower_np) & (points_np <= box_upper_np), axis=1)
        )
        hits_np = np.sort(indices_np[offsets_np[query_index] : offsets_np[query_index + 1]])
        assert np.array_equal(hits_np, inside_np)

        box_o3d = o3d.geometry.AxisAlignedBoundingBox(
            box_lower_np.astype(np.float64), box_upper_np.astype(np.float64)
        )
        assert np.array_equal(
            np.sort(np.asarray(box_o3d.get_point_indices_within_bounding_box(cloud_o3d.points))),
            inside_np,
        )

        n_nonempty += inside_np.size > 0
    # 9 of the 12 boxes hold at least one point here; a run where none did would pass vacuously.
    assert n_nonempty >= 9

    # Inclusive on both faces, and an inverted box selects nothing.
    face_np = np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    face_bvh = od.neighbors.bvh_from_points(points_to_warp(face_np, device))
    face_indices_wp, face_offsets_wp = od.neighbors.query_bvh_box(
        face_bvh,
        wp.array(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], np.float32), wp.vec3, device=device),
        wp.array(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], np.float32), wp.vec3, device=device),
    )
    assert np.array_equal(np.sort(face_indices_wp.numpy()), np.array([0, 1, 2]))
    assert np.array_equal(face_offsets_wp.numpy(), np.array([0, 3, 3]))


@pytest.mark.parity(
    "query_bvh_box",
    "meshlib",
    benchmarked=False,
    reason="findPointsInBox reports every hit through a Python callback, so a timed row would "
    "price the interpreter rather than the search; it is exact as a correctness oracle, and open3d "
    "carries the timed row for this group instead.",
)
def test_query_bvh_box_matches_meshlib(device: str) -> None:
    """
    Class A: the same hit set as ``findPointsInBox``, which is the third exact box query.

    Separate from the open3d comparison because meshlib's is the one that cannot be benchmarked --
    it reports each hit through a ``func_void_from_Id_VertTag_Vector3_float`` callback, so a timed
    row would measure the interpreter. As a correctness oracle it is exact, and it agrees on the
    inclusive-face convention: the second box below has a point on each of its faces.
    """
    rng = np.random.default_rng(17)
    points_np = rng.random((200, 3)).astype(np.float32)
    lower_np = np.array([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]], dtype=np.float32)
    upper_np = np.array([[0.6, 0.7, 0.55], [1.0, 1.0, 1.0]], dtype=np.float32)

    bvh = od.neighbors.bvh_from_points(points_to_warp(points_np, device))
    indices_wp, offsets_wp = od.neighbors.query_bvh_box(
        bvh, points_to_warp(lower_np, device), points_to_warp(upper_np, device)
    )
    indices_np, offsets_np = indices_wp.numpy(), offsets_wp.numpy()

    def collect_into(sink: list[int]) -> Callable[[int, object], None]:
        """Each hit arrives as ``(VertId, Vector3f)``, so the sink cannot be ``list.append``."""
        return lambda vertex_id, _position: sink.append(int(vertex_id))

    cloud_ml = points_to_meshlib(points_np)
    for query_index in range(lower_np.shape[0]):
        hits_ml: list[int] = []
        mm.findPointsInBox(
            cloud_ml,
            mm.Box3f(
                mm.Vector3f(*lower_np[query_index].tolist()),
                mm.Vector3f(*upper_np[query_index].tolist()),
            ),
            cast("mm.func_void_from_Id_VertTag_Vector3_float", collect_into(hits_ml)),
        )
        hits_wp = indices_np[offsets_np[query_index] : offsets_np[query_index + 1]]
        assert np.array_equal(np.sort(np.asarray(hits_ml, dtype=np.int32)), np.sort(hits_wp))
    # The second box holds the whole cloud, so neither side's answer is empty.
    assert int(offsets_np[-1]) - int(offsets_np[-2]) == points_np.shape[0]


def test_query_bvh_box_degenerate_inputs(device: str) -> None:
    """
    An empty query set, a query that hits nothing, and mismatched corner lengths.

    Not a library comparison: these are this wrapper's own early returns and its one guard. The
    offsets shape has to match the general path in both early returns, since a caller reading the
    trailing total would otherwise index past the end.
    """
    points_wp = wp.array(
        np.zeros((4, 3), dtype=np.float32) + np.array([0.0, 0.0, 0.0]), dtype=wp.vec3, device=device
    )
    bvh = od.neighbors.bvh_from_points(points_wp)
    empty_wp = warp_empty(0, wp.vec3, device)

    indices_wp, offsets_wp = od.neighbors.query_bvh_box(bvh, empty_wp, empty_wp)
    assert indices_wp.shape == (0,)
    assert np.array_equal(offsets_wp.numpy(), np.zeros(1, np.int32))

    far_lower_wp = wp.array(np.full((1, 3), 99.0, np.float32), dtype=wp.vec3, device=device)
    far_upper_wp = wp.array(np.full((1, 3), 100.0, np.float32), dtype=wp.vec3, device=device)
    miss_indices_wp, miss_offsets_wp = od.neighbors.query_bvh_box(bvh, far_lower_wp, far_upper_wp)
    assert miss_indices_wp.shape == (0,)
    assert np.array_equal(miss_offsets_wp.numpy(), np.zeros(2, np.int32))

    with pytest.raises(ValueError, match="same length"):
        od.neighbors.query_bvh_box(
            bvh, wp.array(np.zeros((2, 3), np.float32), wp.vec3, device=device), empty_wp
        )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)

    points_wp = points_to_warp(points, device)
    empty_points_wp = warp_empty(0, wp.vec3, device)
    empty_queries_wp = warp_empty(0, wp.vec3, device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = points_to_warp(points[-3:], device)
    radius = 0.5
    query_ball = partial(od.neighbors.query_ball, backend=backend)

    indices, distances = query_ball(empty_points_wp, query_wp, radius)
    assert indices.shape == (0,)
    assert distances.shape == (0,)

    indices, distances = query_ball(empty_points_wp, queries_wp, radius)
    assert len(indices) == queries_wp.size
    assert len(distances) == queries_wp.size
    for single_indices, single_distances in zip(indices, distances, strict=False):
        assert single_indices.shape == (0,)
        assert single_distances.shape == (0,)

    indices, distances = query_ball(points_wp, empty_queries_wp, radius)
    assert len(indices) == 0
    assert len(distances) == 0


def test_knn_initial_radius_matches_uniform_density(device: str):
    rng = np.random.default_rng(3)
    points = rng.random((4000, 3), dtype=np.float32)
    points_wp = points_to_warp(points, device)

    for k in (1, 8):
        radius_wp = od.neighbors.knn_initial_radius(points_wp, k)
        # Expected count in a ball of that radius under the uniform model that defines it.
        volume_np = float(np.prod(points.max(axis=0) - points.min(axis=0)))
        expected_np = (3.0 * (k / 4000) * volume_np / (4.0 * math.pi)) ** (1.0 / 3.0)
        assert np.isclose(radius_wp, expected_np, rtol=1e-6)
        # It really does find about k neighbours: the median count should be within a factor of a
        # few of k, which is what makes a single scan the common case.
        counts_np = np.asarray(
            KDTree(points).query_ball_point(points[:200], radius_wp, return_length=True)
        )
        assert k <= np.median(counts_np) <= 12 * k + 12


def test_knn_initial_radius_degenerate_clouds(device: str):
    """Flat, collinear, coincident and ``k >= n`` clouds all get a usable (or infinite) radius."""
    rng = np.random.default_rng(4)
    planar_np = rng.random((500, 3), dtype=np.float32)
    planar_np[:, 2] = 0.0
    collinear_np = np.zeros((500, 3), dtype=np.float32)
    collinear_np[:, 0] = np.linspace(0.0, 1.0, 500)
    coincident_np = np.zeros((10, 3), dtype=np.float32)

    def radius_of(points_np: np.ndarray, k: int = 4) -> float:
        return od.neighbors.knn_initial_radius(points_to_warp(points_np, device), k)

    # d = 2: pi r^2 = (k / n) * area
    area_np = float(np.prod(planar_np[:, :2].max(axis=0) - planar_np[:, :2].min(axis=0)))
    assert np.isclose(radius_of(planar_np), math.sqrt((4 / 500) * area_np / math.pi), rtol=1e-6)
    # d = 1: 2 r = (k / n) * length
    assert np.isclose(radius_of(collinear_np), 0.5 * (4 / 500) * 1.0, rtol=1e-5)
    # Degenerate box and k >= n both mean "one complete scan".
    assert radius_of(coincident_np) == math.inf
    assert radius_of(planar_np, k=500) == math.inf
    assert od.neighbors.knn_initial_radius(warp_empty(0, wp.vec3, device), 1) == math.inf


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 40])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
@pytest.mark.parity("query_nearest_bvh_k1", "scipy")
@pytest.mark.parity("query_nearest_hashgrid_k1", "scipy")
@pytest.mark.parity("bvh_from_points", "scipy")
@pytest.mark.parity("hashgrid_from_points", "scipy")
def test_query_nearest_single(
    device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float
):
    """
    Class A: k-nearest indices and distances against ``KDTree.query``, over k and ``max_radius``.

    Exact rather than set-compared because the clouds are random and tie-free, so the k-th
    neighbour is unambiguous; [`test_query_nearest_ties`] handles the constructed case where it
    is not. The ``max_radius`` axis is what pins the sentinel written for a query with fewer
    than k neighbours in range.
    """
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 4.0
    kdtree = KDTree(points)
    query = points[0].copy()

    points_wp = points_to_warp(points, device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_distances_np, query_indices_np = kdtree.query(query, k=k, distance_upper_bound=max_radius)
    query_indices_np = np.atleast_1d(np.asarray(query_indices_np))
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.atleast_1d(np.asarray(query_distances_np))

    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    query_indices_wp, query_distances_wp = query_nearest(
        points_wp, query_wp, k=k, max_radius=max_radius
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 10])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
@pytest.mark.parity("query_nearest_bvh_k7", "scipy")
@pytest.mark.parity("query_nearest_hashgrid_k7", "scipy")
def test_query_nearest_batch(
    device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float
):
    """
    Class A: the batched form, with queries offset off the cloud so some fall short of k.

    The ``+ 0.5`` displacement is deliberate -- querying *at* a data point makes the first
    neighbour trivially itself at distance zero, which hides an off-by-one in the heap.
    """
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]] + 0.5

    points_wp = points_to_warp(points, device)
    query_wp = points_to_warp(queries, device)

    query_distances_np, query_indices_np = kdtree.query(
        queries, k=k, distance_upper_bound=max_radius
    )
    query_indices_np = np.asarray(query_indices_np)
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.asarray(query_distances_np)

    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    query_indices_wp, query_distances_wp = query_nearest(
        points_wp, query_wp, k=k, max_radius=max_radius
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 4, 8, 9, 16, 17, 32, 33, 64, 65])
@pytest.mark.parity("query_nearest_bvh_k7", "scipy")
@pytest.mark.parity("query_nearest_hashgrid_k7", "scipy")
@pytest.mark.parity("query_nearest_bvh_k64", "scipy")
def test_query_nearest_row_buckets(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class A. Every candidate-row bucket size and one ``k`` past each, against ``KDTree``.

    The row lives in a ``wp.types.vector(length=K)`` register value whose ``K`` is one of
    ``kernels.neighbors.KNN_ROW_BUCKETS``, so a ``k`` that does not equal its bucket keeps ``K``
    neighbours and returns the first ``k``. An off-by-one in that tail is invisible at ``k == K``
    and shows only just above and just below a boundary, which is what this sweep pins. ``k=65``
    is past the largest bucket and exercises the global-memory fallback kernel.

    The cloud is random in a box, so no two points tie in ``float32`` distance from a query and the
    index comparison is exact; ties are covered by
    [`test_query_nearest_ties`][tests.test_neighbors.test_query_nearest_ties].
    """
    rng = np.random.default_rng(11)
    points = rng.random((300, 3), dtype=np.float32) * 5.0
    queries = rng.random((40, 3), dtype=np.float32) * 5.0

    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    query_indices_wp, query_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    query_distances_np, query_indices_np = KDTree(points).query(queries, k=k)

    assert np.array_equal(query_indices_wp.numpy(), np.asarray(query_indices_np))
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 7, 64])
@pytest.mark.parity("query_nearest_bvh_k1", "igl")
@pytest.mark.parity("query_nearest_bvh_k7", "igl")
@pytest.mark.parity("query_nearest_bvh_k64", "igl")
@pytest.mark.parity("query_nearest_hashgrid_k1", "igl")
@pytest.mark.parity("query_nearest_hashgrid_k7", "igl")
@pytest.mark.parity("bvh_from_points", "igl")
@pytest.mark.parity("hashgrid_from_points", "igl")
def test_query_nearest_matches_igl(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class A, against the second exact k-NN. Indices, element-wise, at the three benchmarked ``k``.

    ``igl.knn`` returns ``(n_queries, k)`` ``int64`` neighbour indices sorted by distance -- the
    same layout and the same order as ``KDTree``'s -- so this is a direct comparison and not a set
    one. It is a genuinely independent implementation: an octree walk against ordito's BVH / hash
    grid and scipy's k-d tree, three different structures for one answer.

    The ``backend`` axis is what carries the four hash-grid markers alongside the four BVH ones:
    igl's answer does not depend on which structure ordito asks, so one comparison covers both
    groups and the benchmark pair reads as ordito's index being the only difference.

    The two ``*_from_points`` markers ride here for the reason the scipy ones do: a structure build
    has no output to compare, so it is validated through the query that consumes it -- and on the
    igl side the octree is quite literally an argument to ``igl.knn``, passed as
    ``*igl.octree(points)[:4]``.

    The cloud is random in a box, so no two points tie in ``float32`` distance from a query and the
    index comparison is exact.
    """
    rng = np.random.default_rng(11)
    points = rng.random((300, 3)) * 5.0
    queries = rng.random((40, 3)) * 5.0

    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    query_indices_wp, _distances_wp = query_nearest(points_wp, queries_wp, k=k)
    query_indices_igl = igl.knn(queries, points, k, *igl.octree(points)[:4])

    assert np.array_equal(query_indices_wp.numpy().reshape(queries.shape[0], k), query_indices_igl)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parity("query_nearest_bvh_k1", "pytorch3d")
@pytest.mark.parity("query_nearest_bvh_k7", "pytorch3d")
@pytest.mark.parity("query_nearest_hashgrid_k1", "pytorch3d")
@pytest.mark.parity("query_nearest_hashgrid_k7", "pytorch3d")
def test_query_nearest_matches_pytorch3d(
    device: str, backend: Literal["bvh", "hashgrid"], k: int
) -> None:
    """
    Class B, against the fourth exact k-NN: ``ops.knn_points``, whose distances are **squared**.

    That square root is the whole transform -- measured **0.0** afterwards on the CPU device and
    1.19e-07 on CUDA, where pytorch3d's own kernel is a different reduction order -- and the
    indices need none, measured **exactly** equal on both devices and at both ``k``. pytorch3d is a
    genuinely independent implementation and the most different of the four: no spatial structure
    at all, just the pairwise loop, against ordito's BVH / hash grid, scipy's k-d tree and igl's
    octree. That is also what makes it the crossover row the benchmark exists for -- brute force
    with perfect coalescing beats a BVH descent at 20 000 points and loses by 98x at 200 000.

    ``k=64`` is left out because there is no ``query_nearest_hashgrid_k64`` group to pair with, and
    the two BVH-only markers would then read as a different claim than the four here.

    The batched wrap is the trap this asserts against first: ``knn_points`` handed a bare
    ``(P, 3)`` reads it as ``(N=P, P1=3, D)`` and compares three points without complaint, so the
    reference's output shape is checked before its values.
    """
    rng = np.random.default_rng(3)
    points_np = rng.normal(size=(500, 3)).astype(np.float32)
    queries_np = rng.normal(size=(300, 3)).astype(np.float32)
    nearest_p3d = p3d_ops.knn_points(
        points_to_torch(queries_np, device), points_to_torch(points_np, device), K=k
    )
    indices_wp, distances_wp = od.neighbors.query_nearest(
        points_to_warp(points_np, device), points_to_warp(queries_np, device), k=k, backend=backend
    )

    assert nearest_p3d.idx.shape == (1, queries_np.shape[0], k)
    assert np.array_equal(
        indices_wp.numpy().reshape(queries_np.shape[0], k), nearest_p3d.idx[0].cpu().numpy()
    )
    assert np.allclose(
        distances_wp.numpy().reshape(queries_np.shape[0], k),
        np.sqrt(nearest_p3d.dists[0].cpu().numpy()),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity("query_ball_bvh", "pytorch3d")
@pytest.mark.parity("query_ball_hashgrid", "pytorch3d")
def test_query_ball_matches_pytorch3d(device: str, backend: Literal["bvh", "hashgrid"]) -> None:
    """
    Class B: ``ops.ball_query`` after removing its ``K`` cap, compared as **sets** per query.

    Two named transforms, both forced by pytorch3d's fixed-width output. It returns a dense
    ``(N, P, K)`` block padded with ``-1``, so ``K`` has to be passed at or above the largest true
    neighbour count -- **26** on this cloud, and a smaller ``K`` is a silent truncation rather than
    an error. And the fill order is ascending **index**, not ascending distance (verified), so only
    the set is comparable; ``return_sorted`` has no counterpart to pin against.

    The cap is asserted not to bite, which is the anti-vacuity check that matters here: a ``K``
    swallowing every row would make the set comparison trivially true on the truncated prefix.
    Some rows are legitimately **empty** at this radius (the cloud is Gaussian, so the tails are
    sparse), which is why the second assert is on the total rather than on the per-row minimum.
    """
    rng = np.random.default_rng(3)
    points_np = rng.normal(size=(500, 3)).astype(np.float32)
    queries_np = rng.normal(size=(50, 3)).astype(np.float32)
    radius = 0.6
    ball_p3d = p3d_ops.ball_query(
        points_to_torch(queries_np, device), points_to_torch(points_np, device), K=40, radius=radius
    )
    indices_p3d = ball_p3d.idx[0].cpu().numpy()
    flat_wp, _, offsets_wp = od.neighbors.query_ball_with_offsets(
        points_to_warp(points_np, device),
        points_to_warp(queries_np, device),
        radius,
        backend=backend,
    )

    counts_p3d = (indices_p3d >= 0).sum(axis=1)
    assert counts_p3d.max() < 40, "the K cap truncated a row; raise it"
    assert int(counts_p3d.sum()) > queries_np.shape[0]
    flat_np, offsets_np = flat_wp.numpy(), offsets_wp.numpy()
    for query in range(queries_np.shape[0]):
        assert set(flat_np[offsets_np[query] : offsets_np[query + 1]].tolist()) == set(
            indices_p3d[query][indices_p3d[query] >= 0].tolist()
        )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 7, 64])
@pytest.mark.parity("query_nearest_bvh_k1", "open3d")
@pytest.mark.parity("query_nearest_bvh_k7", "open3d")
@pytest.mark.parity("query_nearest_bvh_k64", "open3d")
@pytest.mark.parity("query_nearest_hashgrid_k1", "open3d")
@pytest.mark.parity("query_nearest_hashgrid_k7", "open3d")
def test_query_nearest_matches_open3d(
    device: str, backend: Literal["bvh", "hashgrid"], k: int
) -> None:
    """
    Class A, against the third exact k-NN: ``o3d.core.nns.NearestNeighborSearch.knn_search``.

    Open3D's batched tensor search (not the legacy ``KDTreeFlann`` per-query loop) returns
    ``(n_queries, k)`` indices sorted by distance in ``KDTree``'s layout, plus **squared**
    distances -- the square root is the named transform that makes the distance half Class B on
    its own; the index half needs none. The cloud is random in a box, so no two points tie in
    ``float32`` distance from a query and the index comparison is exact.

    The ``backend`` axis covers the hash-grid groups as well as the BVH ones: open3d's answer is
    the same whichever structure ordito asks, which is why the two benchmark groups carry the
    identical reference row.
    """
    rng = np.random.default_rng(11)
    points = rng.random((300, 3)) * 5.0
    queries = rng.random((40, 3)) * 5.0

    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    query_indices_wp, query_distances_wp = query_nearest(points_wp, queries_wp, k=k)

    nns_o3d = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
    assert nns_o3d.knn_index()
    indices_o3d, squared_o3d = nns_o3d.knn_search(o3d.core.Tensor(queries), k)  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults

    assert np.array_equal(
        query_indices_wp.numpy().reshape(queries.shape[0], k), indices_o3d.numpy()
    )
    assert np.allclose(
        query_distances_wp.numpy().reshape(queries.shape[0], k),
        np.sqrt(squared_o3d.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity("query_ball_bvh", "open3d")
@pytest.mark.parity("query_ball_hashgrid", "open3d")
def test_query_ball_matches_open3d(device: str, backend: Literal["bvh", "hashgrid"]) -> None:
    """
    Class A on the neighbour sets: ``fixed_radius_search`` against the ``*_with_offsets`` form.

    Open3D returns the same CSR-like ``(indices, distances, offsets)`` triple ordito's offsets
    form does, with squared distances. Per-query neighbour *sets* are compared (order within a
    radius query is not part of either contract), and the counts vector is compared exactly. The
    cloud is random, so no point sits at exactly the radius and the two libraries' boundary rules
    (Open3D's radius searches are exclusive at exactly ``r``; ordito's inclusive) cannot differ.
    """
    rng = np.random.default_rng(3)
    points = rng.random((400, 3)) * 3.0
    queries = rng.random((60, 3)) * 3.0
    radius = 0.4

    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)
    query_ball_with_offsets = partial(od.neighbors.query_ball_with_offsets, backend=backend)
    neighbors_wp, _distances_wp, offsets_wp = query_ball_with_offsets(points_wp, queries_wp, radius)

    nns_o3d = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
    assert nns_o3d.fixed_radius_index(radius)
    indices_o3d, _squared_o3d, offsets_o3d = nns_o3d.fixed_radius_search(
        o3d.core.Tensor(queries),  # pyright: ignore[reportCallIssue]  # the stub drops dtype/device defaults
        radius,
    )
    indices_o3d = indices_o3d.numpy()
    offsets_o3d = offsets_o3d.numpy()

    neighbors_np = neighbors_wp.numpy()
    bounds_np = offsets_wp.numpy()
    assert int(bounds_np[-1]) == neighbors_np.size
    starts_np, ends_np = bounds_np[:-1], bounds_np[1:]
    assert np.array_equal(ends_np - starts_np, np.diff(offsets_o3d))
    for query_index in range(queries.shape[0]):
        set_wp = set(neighbors_np[starts_np[query_index] : ends_np[query_index]])
        set_o3d = set(indices_o3d[offsets_o3d[query_index] : offsets_o3d[query_index + 1]])
        assert set_wp == set_o3d
    assert neighbors_np.size > 0  # non-vacuous: the radius actually finds neighbours


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", sorted(kernel_neighbors.KNN_ROW_BUCKETS))
def test_query_nearest_ties(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class B (reduction: which of two equidistant points wins a slot is not specified).

    On an integer lattice a query has dozens of *exactly* tied neighbours, so the row's insert
    order is exercised rather than assumed — a shift chain that mishandles a run of equal distances
    keeps a farther point and reports a distance ``KDTree`` does not, which no random cloud reveals.
    Half the queries sit on lattice points (maximal ties), half at cell centres. ``k`` runs over
    every ``KNN_ROW_BUCKETS`` size because the register-row carry is written out inline once per
    bucket (see the comment above ``_bvh_nearest_row_kernel``), so each copy meets the lattice here.

    The ``k`` distances must match ``KDTree`` exactly and every returned index must actually sit at
    the distance reported for it; the *identity* of a tied neighbour is not compared, because both
    choices are correct answers — the same thing
    [`query_nearest`][ordito.neighbors.query_nearest] tells callers.
    """
    axis = np.arange(12, dtype=np.float32)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(12)
    on_lattice = lattice[rng.choice(lattice.shape[0], size=30, replace=False)]
    at_centers = lattice[rng.choice(lattice.shape[0], size=30, replace=False)] + 0.5
    queries = np.ascontiguousarray(np.vstack([on_lattice, at_centers]), dtype=np.float32)

    points_wp = points_to_warp(lattice, device)
    queries_wp = points_to_warp(queries, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    query_indices_wp, query_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    # ``k == 1`` returns length-m 1D arrays on both sides; reshape so one comparison covers all k.
    rows = (queries.shape[0], k)
    query_distances_np = np.reshape(KDTree(lattice).query(queries, k=k)[0], rows)

    if k > 1:
        # Non-vacuity: the fixture must actually put runs of equal float32 distances in the rows,
        # or the insert's tie handling is never exercised and this is a plain smoke test.
        assert np.any(query_distances_np[:, 1:] == query_distances_np[:, :-1])
    assert np.allclose(
        query_distances_wp.numpy().reshape(rows), query_distances_np, rtol=1e-5, atol=1e-5
    )
    gathered = np.linalg.norm(
        lattice[query_indices_wp.numpy().reshape(rows)] - queries[:, None, :], axis=-1
    )
    assert np.allclose(gathered, query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_exact_duplicates_answer_a_nearest(
    device: str, backend: Literal["bvh", "hashgrid"]
) -> None:
    """
    Class B against ``KDTree`` (reduction: which copy of a duplicated point wins is unspecified).

    Every point is stored twice, so every ``k = 1`` query ties exactly between two indices. The
    three ``k = 1`` searches break that tie differently -- the BVH-backend mesh query takes
    whichever collapsed triangle its descent reaches first -- and all of them are correct, so the
    claim pinned is that the index returned is *a* nearest: one of the two copies of ``KDTree``'s
    answer, at exactly the distance reported. Queries on the cloud and a cloud diameter off it.
    """
    rng = np.random.default_rng(21)
    base = rng.random((300, 3), dtype=np.float32)
    points = np.ascontiguousarray(np.vstack([base, base]))
    queries = np.ascontiguousarray(
        np.vstack([rng.random((50, 3)), rng.random((50, 3)) + 1.0]), dtype=np.float32
    )
    indices_wp, distances_wp = od.neighbors.query_nearest(
        points_to_warp(points, device), points_to_warp(queries, device), k=1, backend=backend
    )
    distances_kd, indices_kd = KDTree(base).query(queries)
    indices_np = indices_wp.numpy()
    assert np.array_equal(indices_np % base.shape[0], indices_kd)
    assert np.allclose(distances_wp.numpy(), distances_kd, rtol=1e-5, atol=1e-5)
    gathered = np.linalg.norm(points[indices_np] - queries, axis=-1)
    assert np.allclose(gathered, distances_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_knn_sorted_insert_rejects_nan_instead_of_writing_past_its_row(device: str):
    """
    Not a library comparison: no reference can observe an out-of-bounds kernel write.

    ``knn_sorted_insert`` admits a candidate on two comparisons and then places it with
    ``binary_search_index`` (``searchsorted(side="right")``) plus ``array_shift_insert``. Every
    comparison against a NaN is false, so a NaN distance passes *both* admissions (``NaN > radius``
    and ``NaN >= inf`` are each false), the search's ``values[mid] > value`` probe never fires, and
    it returns ``n`` -- whereupon ``array_shift_insert`` writes ``row[n]``, one element past the
    caller's ``k``-wide row.

    A NaN query reaches it: ``wp.hash_grid_query`` clamps a NaN centre into the guard region and
    enumerates a real cell, and ``knn_linear_scan`` walks the whole cloud unconditionally, so every
    candidate arrives with ``d`` NaN. Only the global-memory row path is exposed -- the register-row
    kernels bound their carry by ``row_size`` -- so ``k`` must exceed the largest
    ``KNN_ROW_BUCKETS`` entry, which is what makes this a kernel-level test rather than a
    ``query_nearest`` one: the public call allocates exactly ``(m, k)``, so the overrun of the last
    query's row lands past the whole allocation and is invisible from Python (host-heap corruption
    on the CPU device, section 12 of ``.claude/CLAUDE.md``).

    The canary is the second row of a two-row output with **one** thread launched, so nothing is
    entitled to touch it. Mutation probe: deleting the ``wp.isnan`` guard from
    ``knn_sorted_insert`` fails the two canary asserts on both devices, with row 1 holding the NaN
    and the offending point index; the ``row0`` assert passes either way, so it is the canary that
    carries this test and not the answer.
    """
    k = max(kernel_neighbors.KNN_ROW_BUCKETS) + 6
    rng = np.random.default_rng(0)
    points = rng.random((200, 3), dtype=np.float32) * 0.1
    points_wp = points_to_warp(points, device)
    grid = wp.HashGrid(16, 16, 16, device=device)
    grid.build(points_wp, 0.05)

    indices_wp = odt.empty_2d((2, k), wp.int32, device=device)
    distances_wp = odt.empty_2d((2, k), wp.float32, device=device)
    indices_wp.fill_(-7)
    distances_wp.fill_(-7.0)
    queries_wp = points_to_warp(np.full((1, 3), np.nan, dtype=np.float32), device)
    wp.launch(
        kernel_neighbors.query_hashgrid_nearest_neighbors,
        dim=1,
        inputs=[
            points_wp,
            queries_wp,
            grid.id,
            wp.int32(k),
            wp.float32(1e30),
            wp.float32(0.05),
            wp.float32(1.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.1, 0.1, 0.1),
            wp.int32(0),  # no deferral: this global-row kernel serves k above every bucket
            wp.zeros(1, dtype=wp.int32, device=device),
            indices_wp,
            distances_wp,
        ],
        device=device,
    )
    # A NaN query matches nothing, which is the answer the row itself should carry.
    assert np.array_equal(distances_wp.numpy()[0], np.full(k, np.inf, dtype=np.float32))
    # The canary: no thread was launched over row 1, so it must still hold the fill.
    assert np.array_equal(distances_wp.numpy()[1], np.full(k, -7.0, dtype=np.float32))
    assert np.array_equal(indices_wp.numpy()[1], np.full(k, -7, dtype=np.int32))


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 2])
def test_query_nearest_empty(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class A on the shape, against ``KDTree.query``: a degenerate input keeps its rank.

    ``query_nearest`` collapses the trailing axis at ``k == 1``, exactly as ``KDTree.query`` does,
    so the rank is a function of ``k`` and the form of ``queries`` alone -- never of whether an
    answer was found. Both degenerate inputs used to bypass that collapse and come back rank-2 at
    ``k == 1`` while every populated call returned rank-1, so ``indices[i]`` was a scalar on an
    ordinary cloud and a length-1 row on an empty one.

    **``k`` is the parametrization because ``k = 2`` was this test's hardcoded value and is the one
    value that cannot see the defect.** The mutation probe is the fix itself: routing the two early
    returns back through ``odt.empty_2d`` / an unshaped ``_empty_nearest`` fails all four ``k == 1``
    cells on the rank asserts below, and leaves every ``k == 2`` cell passing.

    Class B on the values, one named difference: a slot with no neighbour is ``-1`` here and the
    tree size (so ``0`` for an empty cloud) in SciPy, which is why only the distances are compared
    element-wise.
    """
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)
    empty = np.empty((0, 3), dtype=np.float32)

    points_wp = points_to_warp(points, device)
    empty_points_wp = warp_empty(0, wp.vec3, device)
    empty_queries_wp = warp_empty(0, wp.vec3, device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = points_to_warp(points[-3:], device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)

    # A single `wp.vec3` query is rank-1 of length k whatever k is -- SciPy returns a 0-d scalar at
    # k == 1 instead, the one place the two conventions differ, since Warp has no 0-d array.
    indices, distances = query_nearest(empty_points_wp, query_wp, k=k)
    assert np.array_equal(indices.numpy(), -np.ones(k))
    assert np.array_equal(distances.numpy(), np.full(k, np.inf))

    # The anchor: whatever rank a populated call returns is the rank both degenerate calls owe.
    populated_distances_np = np.asarray(KDTree(points).query(points[-3:], k=k)[0])
    indices, distances = query_nearest(points_wp, queries_wp, k=k)
    assert indices.shape == populated_distances_np.shape
    assert distances.shape == populated_distances_np.shape

    for points_arg, queries_arg, points_np, queries_np in (
        (empty_points_wp, queries_wp, empty, points[-3:]),
        (points_wp, empty_queries_wp, points, empty),
    ):
        distances_np = np.asarray(KDTree(points_np).query(queries_np, k=k)[0])
        indices, distances = query_nearest(points_arg, queries_arg, k=k)
        assert indices.shape == distances_np.shape
        assert distances.shape == distances_np.shape
        assert np.array_equal(distances.numpy(), distances_np.astype(np.float32))
        assert np.array_equal(indices.numpy(), np.full(distances_np.shape, -1, dtype=np.int32))


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3])
def test_query_nearest_initial_radius_invariance(
    device: str, backend: Literal["bvh", "hashgrid"], k: int
):
    """
    ``initial_radius`` is a speed knob: every value must give byte-identical output.

    The three values exercise all three loop paths — the default estimate (usually one certified
    scan), ``inf`` (one forced complete scan), and a radius far too small (repeated geometric
    growth, which is also the case that would double-insert if a scan forgot to reset its row).
    """
    rng = np.random.default_rng(5)
    points = rng.random((400, 3), dtype=np.float32) * 3.0
    queries = rng.random((60, 3), dtype=np.float32) * 3.0
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)

    reference_indices_wp, reference_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    for initial_radius in (math.inf, 1e-6 * diagonal, 0.0):
        indices_wp, distances_wp = query_nearest(
            points_wp, queries_wp, k=k, initial_radius=initial_radius
        )
        assert np.array_equal(indices_wp.numpy(), reference_indices_wp.numpy())
        assert np.array_equal(distances_wp.numpy(), reference_distances_wp.numpy())

    query_distances_np, query_indices_np = KDTree(points).query(queries, k=k)
    assert np.array_equal(reference_indices_wp.numpy(), np.asarray(query_indices_np))
    assert np.allclose(reference_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_clustered(device: str, backend: Literal["bvh", "hashgrid"]):
    """
    Tight clusters in a mostly empty box: the density estimate under-shoots badly on purpose.

    Queries sit in the void between clusters, so the first scans come back empty and the search
    has to grow geometrically. On the hash-grid backend the radius outruns the cell width, which
    is the path that hands off to the exact linear scan.
    """
    rng = np.random.default_rng(6)
    centers = rng.random((6, 3)) * 100.0
    points = np.concatenate([c + rng.normal(scale=0.01, size=(150, 3)) for c in centers])
    points = np.ascontiguousarray(points, dtype=np.float32)
    queries = np.ascontiguousarray(rng.random((80, 3)) * 100.0, dtype=np.float32)

    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)

    for k in (1, 5):
        indices_wp, distances_wp = query_nearest(points_wp, queries_wp, k=k)
        query_distances_np, query_indices_np = KDTree(points).query(queries, k=k)
        assert np.array_equal(indices_wp.numpy(), np.asarray(query_indices_np))
        assert np.allclose(distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-4)


def test_query_nearest_defers_far_rows_to_a_closest_point_query(device: str) -> None:
    """
    Class A, against ``scipy.spatial.KDTree``: a ``k = 1`` row the grid gives up on is still exact.

    The hash-grid backend hands a row it cannot certify within its widest walk to a closest-point
    query over the cloud, which builds a ``wp.Mesh``. Counting those builds is what makes the
    comparison non-vacuous: the displaced queries must reach the second pass and the on-cloud ones
    must not, or the test is checking the grid twice. ``max_radius`` is honoured on the deferred
    rows too -- a row whose nearest point is out of range comes back unfilled.
    """
    rng = np.random.default_rng(12)
    points_np = rng.random((4000, 3), dtype=np.float32)
    far_np = np.ascontiguousarray(points_np[:500] + 5.0, dtype=np.float32)
    near_np = np.ascontiguousarray(points_np[:500] + 1e-4, dtype=np.float32)
    points_wp = points_to_warp(points_np, device)
    builds = []
    original = cast("Callable[..., None]", wp.Mesh.__init__)

    def counting_init(self: wp.Mesh, *args: object, **kwargs: object) -> None:
        builds.append(1)
        original(self, *args, **kwargs)

    wp.Mesh.__init__ = counting_init
    try:
        for queries_np, expect_deferral in ((far_np, True), (near_np, False)):
            builds.clear()
            indices_wp, distances_wp = od.neighbors.query_nearest(
                points_wp, points_to_warp(queries_np, device), k=1, backend="hashgrid"
            )
            assert bool(builds) == expect_deferral
            distances_np, indices_np = KDTree(points_np).query(queries_np, k=1)
            assert np.array_equal(indices_wp.numpy(), indices_np)
            assert np.allclose(distances_wp.numpy(), distances_np, rtol=1e-5, atol=1e-5)
        builds.clear()
        cap = float(np.median(KDTree(points_np).query(far_np, k=1)[0]))
        indices_wp, distances_wp = od.neighbors.query_nearest(
            points_wp, points_to_warp(far_np, device), k=1, backend="hashgrid", max_radius=cap
        )
        assert builds
    finally:
        wp.Mesh.__init__ = original
    distances_np, indices_np = KDTree(points_np).query(far_np, k=1, distance_upper_bound=cap)
    in_range = np.isfinite(distances_np)
    assert 0 < in_range.sum() < len(far_np)  # non-vacuity: both kinds of row occur
    assert np.array_equal(indices_wp.numpy()[in_range], np.asarray(indices_np)[in_range])
    assert (indices_wp.numpy()[~in_range] == -1).all()
    assert np.isinf(distances_wp.numpy()[~in_range]).all()


def test_query_nearest_defers_sparse_self_rows_to_the_bvh(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against ``scipy.spatial.KDTree``: a ``k > 1`` row the grid defers is still exact.

    A self-query on a cloud whose density falls by two decades along one axis: the grid's cell is
    sized for the mean spacing, so rows in the sparse end outgrow its widest walk and go to the
    BVH's radius search rather than scanning the whole cloud. Counting BVH builds keeps it
    non-vacuous -- the graded self-query must reach the second pass, while a query set displaced
    off the cloud must not (it keeps the scan). A skipped second pass would leave its rows marked
    ``DEFERRED_ROW`` and fail the comparison. ``max_radius`` is honoured on the deferred rows too.
    """
    rng = np.random.default_rng(23)
    n = 20_000
    decades = 2.0 * np.log(10.0)
    x = np.log1p(rng.random(n) * np.expm1(decades)) / decades
    graded_np = np.ascontiguousarray(
        np.stack([x, 0.2 * rng.random(n), 0.2 * rng.random(n)], axis=1), dtype=np.float32
    )
    uniform_np = rng.random((n, 3), dtype=np.float32)
    far_np = np.ascontiguousarray(uniform_np[:500] * 1.5 + 2.0, dtype=np.float32)
    builds = []
    original = od.neighbors.bvh_from_points

    def counting_build(points: wp.array[wp.vec3], leaf_size: int = 4) -> wp.Bvh:
        builds.append(1)
        return original(points, leaf_size)

    monkeypatch.setattr(od.neighbors, "bvh_from_points", counting_build)
    for points_np, queries_np, expect_deferral in (
        (graded_np, graded_np, True),
        (uniform_np, far_np, False),
    ):
        points_wp = points_to_warp(points_np, device)
        for k in (8, 30):
            builds.clear()
            indices_wp, distances_wp = od.neighbors.query_nearest(
                points_wp, points_to_warp(queries_np, device), k=k, backend="hashgrid"
            )
            assert bool(builds) == expect_deferral
            distances_np = KDTree(points_np).query(queries_np, k=k)[0]
            assert np.allclose(distances_wp.numpy(), distances_np, rtol=1e-5, atol=1e-6)
            # Indices through the distance they name: a float32 near-tie may order two
            # neighbours differently from the float64 tree, which the distance row cannot see.
            named = np.linalg.norm(points_np[indices_wp.numpy()] - queries_np[:, None], axis=2)
            assert np.allclose(named, distances_np, rtol=1e-5, atol=1e-6)

    points_wp = points_to_warp(graded_np, device)
    distances_np = np.asarray(KDTree(graded_np).query(graded_np, k=8)[0])
    cap = float(np.quantile(distances_np[:, -1], 0.99))
    builds.clear()
    indices_wp, distances_wp = od.neighbors.query_nearest(
        points_wp, points_wp, k=8, backend="hashgrid", max_radius=cap
    )
    assert builds
    in_range = distances_np <= cap
    assert 0 < (~in_range).sum()  # non-vacuity: some slots are out of range
    assert np.allclose(distances_wp.numpy()[in_range], distances_np[in_range], rtol=1e-5)
    assert (indices_wp.numpy()[~in_range] == -1).all()


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_degenerate_clouds(device: str, backend: Literal["bvh", "hashgrid"]):
    """Coincident, collinear and planar clouds: zero-extent axes must not break the search."""
    rng = np.random.default_rng(7)
    coincident_np = np.full((20, 3), 2.5, dtype=np.float32)
    collinear_np = np.zeros((60, 3), dtype=np.float32)
    collinear_np[:, 0] = np.linspace(-1.0, 1.0, 60)
    planar_np = rng.random((60, 3), dtype=np.float32)
    planar_np[:, 2] = 0.75

    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    for points in (coincident_np, collinear_np, planar_np):
        # Queries on the cloud and far outside it (the latter has an unbounded complete radius).
        queries = np.ascontiguousarray(np.vstack((points[:5], points[:5] + 40.0)), dtype=np.float32)
        points_wp = points_to_warp(points, device)
        queries_wp = points_to_warp(queries, device)

        indices_wp, distances_wp = query_nearest(points_wp, queries_wp, k=3)
        query_distances_np, _query_indices_np = KDTree(points).query(queries, k=3)
        # Coincident points tie at every slot, so compare the distances (and that the indices are
        # in range and distinct), not the arbitrary index order.
        assert np.allclose(distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)
        assert indices_wp.numpy().min() >= 0
        assert all(len(set(row.tolist())) == 3 for row in indices_wp.numpy())


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_prebuilt_index(device: str, backend: Literal["bvh", "hashgrid"]):
    """A hoisted BVH / hash grid gives the same answer as letting the query build its own."""
    rng = np.random.default_rng(8)
    points = rng.random((300, 3), dtype=np.float32) * 2.0
    queries = rng.random((40, 3), dtype=np.float32) * 2.0
    points_wp = points_to_warp(points, device)
    queries_wp = points_to_warp(queries, device)

    k = 4
    initial_radius = od.neighbors.knn_initial_radius(points_wp, k)
    accelerator = (
        od.neighbors.bvh_from_points(points_wp)
        if backend == "bvh"
        else od.neighbors.hashgrid_from_points(points_wp, initial_radius)
    )

    # No ``backend=`` on this call: the prebuilt structure selects it by its own type, which is the
    # merged API's dispatch rule.
    indices_wp, distances_wp = od.neighbors.query_nearest(
        points_wp, queries_wp, k=k, initial_radius=initial_radius, accelerator=accelerator
    )
    built_indices_wp, built_distances_wp = od.neighbors.query_nearest(
        points_wp, queries_wp, k=k, backend=backend
    )
    assert np.array_equal(indices_wp.numpy(), built_indices_wp.numpy())
    assert np.array_equal(distances_wp.numpy(), built_distances_wp.numpy())


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_rejects_negative_initial_radius(
    device: str, backend: Literal["bvh", "hashgrid"]
):
    points_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)
    with pytest.raises(ValueError, match="initial_radius"):
        query_nearest(points_wp, points_wp, k=1, initial_radius=-1.0)


def test_backend_and_accelerator_must_agree(device: str) -> None:
    """
    Not a library comparison: the one new failure mode the merged query API introduced.

    ``backend`` and ``accelerator`` can disagree, and the answer is to be loud rather than to
    silently prefer one -- a caller who passes a ``wp.Bvh`` and writes ``backend="hashgrid"`` has a
    bug, not a preference. Checked on all four entry points, since each resolves the pair itself.

    The two positive branches matter as much as the raise: a prebuilt structure alone must be
    accepted (it selects the backend by its type), and a matching pair must be accepted too, or the
    guard would be rejecting correct calls.
    """
    points_wp = points_to_warp(np.random.default_rng(0).random((32, 3)), device)
    bvh = od.neighbors.bvh_from_points(points_wp)
    grid = od.neighbors.hashgrid_from_points(points_wp, 0.25)

    # Four signatures behind one loop: typed as the common call shape, not their union.
    cases = cast(
        "tuple[tuple[Callable[..., object], dict[str, object]], ...]",
        (
            (od.neighbors.query_ball, {"r": 0.25}),
            (od.neighbors.query_ball_count, {"r": 0.25}),
            (od.neighbors.query_ball_with_offsets, {"r": 0.25}),
            (od.neighbors.query_nearest, {"k": 2}),
        ),
    )
    for query, extra in cases:
        with pytest.raises(ValueError, match="contradicts the accelerator"):
            query(points_wp, points_wp, accelerator=bvh, backend="hashgrid", **extra)
        with pytest.raises(ValueError, match="contradicts the accelerator"):
            query(points_wp, points_wp, accelerator=grid, backend="bvh", **extra)
        with pytest.raises(ValueError, match='backend must be "hashgrid" or "bvh"'):
            query(points_wp, points_wp, backend="kdtree", **extra)

        # An accelerator on its own, and a matching pair, are both fine.
        query(points_wp, points_wp, accelerator=bvh, **extra)
        query(points_wp, points_wp, accelerator=grid, backend="hashgrid", **extra)


def test_the_two_backends_agree(device: str) -> None:
    """
    Ordito against ordito: the two backends are one function, so they must answer identically.

    Not a parity assert -- scipy carries the oracle for both, in the class-A tests above. This pins
    the claim the merge rests on: ``backend`` is a *cost* choice, so a divergence here would mean
    the keyword changed the answer and the shared ``query_ball_count`` group name is a lie.
    """
    rng = np.random.default_rng(7)
    points_wp = points_to_warp(rng.random((400, 3)), device)
    queries_wp = points_to_warp(rng.random((60, 3)), device)

    counts_bvh = od.neighbors.query_ball_count(points_wp, queries_wp, 0.3, backend="bvh")
    counts_grid = od.neighbors.query_ball_count(points_wp, queries_wp, 0.3, backend="hashgrid")
    assert int(counts_bvh.numpy().sum()) > 0  # non-vacuity: the radius finds neighbours
    assert np.array_equal(counts_bvh.numpy(), counts_grid.numpy())

    idx_bvh, dist_bvh = od.neighbors.query_nearest(points_wp, queries_wp, k=5, backend="bvh")
    idx_grid, dist_grid = od.neighbors.query_nearest(points_wp, queries_wp, k=5, backend="hashgrid")
    assert np.array_equal(idx_bvh.numpy(), idx_grid.numpy())
    assert np.allclose(dist_bvh.numpy(), dist_grid.numpy(), rtol=1e-5, atol=1e-5)

    # ``k = 1`` has three searches -- the grid (deferring far rows to the collapsed-triangle mesh),
    # the BVH backend's mesh query, and a caller's ``wp.Bvh`` walked by radius deepening -- and
    # every one computes the distance as ``wp.length`` of the same difference, so they agree to the
    # bit. The shifted queries sit off the cloud, where the grid defers and the walk deepens.
    shifted_wp = points_to_warp(rng.random((60, 3)) + 1.5, device)
    for queries in (queries_wp, shifted_wp):
        answers = [
            od.neighbors.query_nearest(points_wp, queries, k=1, backend="bvh"),
            od.neighbors.query_nearest(points_wp, queries, k=1, backend="hashgrid"),
            od.neighbors.query_nearest(
                points_wp, queries, k=1, accelerator=od.neighbors.bvh_from_points(points_wp)
            ),
        ]
        for indices, distances in answers[1:]:
            assert np.array_equal(indices.numpy(), answers[0][0].numpy())
            assert np.array_equal(distances.numpy(), answers[0][1].numpy())


@pytest.mark.parity(
    "query_weighted_nearest",
    "meshlib",
    benchmarked=False,
    reason="findClosestWeightedPoint reads every site's weight through a Python callback and "
    "answers one query per call, so a timed row would price the interpreter twice over; no other "
    "installed reference has an additively weighted query at all.",
)
def test_query_weighted_nearest_matches_meshlib(device: str) -> None:
    """
    Class A: the winner and its weighted distance, against MeshLib and brute force.

    Three answers again, one of them exhaustive: the ``O(n*m)`` score matrix settles the truth and
    MeshLib confirms the convention, which is the part worth confirming. Probed on the wheel, its
    ``dist`` is exactly ``min(|p - q| - w(p))`` and its ``vId`` the argmin, so ordito's return
    needs no transform. Its ``pointWeight`` is a per-vertex Python callback and ``maxWeight``
    defaults to **0.0**, which would silently prune the true winner on any positive weight set; both
    are set explicitly here.

    Non-vacuity is the real risk in this test, and it is measured rather than asserted loosely: with
    weights drawn over 0.4 on a unit cube, **81%** of queries have a different winner than the
    unweighted query, so the comparison could not pass by accident on a weight-blind implementation.
    That fraction is asserted.
    """
    rng = np.random.default_rng(7)
    points_np = rng.random((2000, 3)).astype(np.float32)
    weights_np = (rng.random(2000) * 0.4).astype(np.float32)
    queries_np = (rng.random((200, 3)) * 1.2 - 0.1).astype(np.float32)

    score_np = np.linalg.norm(queries_np[:, None, :] - points_np[None], axis=-1) - weights_np[None]
    nearest_np, weighted_np = score_np.argmin(1), score_np.min(1)
    plain_np = np.linalg.norm(queries_np[:, None, :] - points_np[None], axis=-1).argmin(1)
    assert (nearest_np != plain_np).mean() > 0.5  # the weights decide most queries, so not vacuous

    index_wp, distance_wp = od.neighbors.query_weighted_nearest(
        points_to_warp(points_np, device),
        wp.array(weights_np, dtype=wp.float32, device=device),
        points_to_warp(queries_np, device),
    )
    assert np.array_equal(index_wp.numpy(), nearest_np)
    assert np.allclose(distance_wp.numpy(), weighted_np, rtol=1e-5, atol=1e-5)

    tree_ml = mm.AABBTreePoints(points_to_meshlib(points_np))
    params_ml = mm.DistanceFromWeightedPointsComputeParams()
    params_ml.pointWeight = cast(
        "mm.func_float_from_Id_VertTag", lambda vertex_id: float(weights_np[int(vertex_id)])
    )
    params_ml.maxWeight = float(weights_np.max())
    params_ml.minWeight = float(weights_np.min())
    for query_index in range(0, queries_np.shape[0], 8):  # every 8th: one Python call per query
        result_ml = mm.findClosestWeightedPoint(
            mm.Vector3f(*queries_np[query_index].tolist()), tree_ml, params_ml
        )
        assert int(result_ml.vId) == int(nearest_np[query_index])
        assert np.isclose(result_ml.dist, weighted_np[query_index], rtol=1e-5, atol=1e-5)


def test_query_weighted_nearest_conventions(device: str) -> None:
    """
    Not a library comparison: the ``w = 0`` identity, negative distances, and the two empty inputs.

    Two claims that make the function's contract legible. At zero weights it *is*
    [`query_nearest`][ordito.neighbors.query_nearest] -- pinned as a
    ordito-against-ordito check, where the oracle lives on the unweighted side (scipy, in the k-NN
    tests above). And a query inside a site's radius reports a **negative** weighted distance, which
    is the sign a caller reads to mean "covered".
    """
    rng = np.random.default_rng(8)
    points_np = rng.random((300, 3)).astype(np.float32)
    queries_np = rng.random((50, 3)).astype(np.float32)
    points_wp = points_to_warp(points_np, device)
    queries_wp = points_to_warp(queries_np, device)

    zero_wp = wp.zeros(300, dtype=wp.float32, device=device)
    index_wp, distance_wp = od.neighbors.query_weighted_nearest(points_wp, zero_wp, queries_wp)
    plain_index_wp, plain_distance_wp = od.neighbors.query_nearest(
        points_wp, queries_wp, k=1, backend="bvh"
    )
    assert np.array_equal(index_wp.numpy(), plain_index_wp.numpy())
    assert np.allclose(distance_wp.numpy(), plain_distance_wp.numpy(), rtol=1e-6, atol=1e-6)

    covered_wp = wp.full(300, wp.float32(2.0), dtype=wp.float32, device=device)
    _covered_index_wp, covered_distance_wp = od.neighbors.query_weighted_nearest(
        points_wp, covered_wp, queries_wp
    )
    assert np.all(covered_distance_wp.numpy() < 0.0)  # every query is inside every site's radius

    empty_index_wp, empty_distance_wp = od.neighbors.query_weighted_nearest(
        points_wp, zero_wp, warp_empty(0, wp.vec3, device)
    )
    assert empty_index_wp.shape == (0,)
    assert empty_distance_wp.shape == (0,)

    no_sites_index_wp, no_sites_distance_wp = od.neighbors.query_weighted_nearest(
        warp_empty(0, wp.vec3, device), warp_empty(0, wp.float32, device), queries_wp
    )
    assert np.all(no_sites_index_wp.numpy() == -1)
    assert np.all(np.isinf(no_sites_distance_wp.numpy()))

    with pytest.raises(ValueError, match="one entry per point"):
        od.neighbors.query_weighted_nearest(
            points_wp, wp.zeros(2, dtype=wp.float32, device=device), queries_wp
        )


@pytest.mark.parity("nearest_neighbor_distance", "open3d")
def test_nearest_neighbor_distance_matches_open3d(device: str) -> None:
    """
    Class A: element for element against ``PointCloud.compute_nearest_neighbor_distance``.

    The distance to the nearest *other* point, so the self-match in slot 0 of the k-NN table must be
    skipped -- an implementation returning column 0 would report all zeros, which is what the
    strictly-positive assert below excludes. It also pins the strided-column read: column 1 of an
    ``(n, 2)`` table is a non-contiguous view, and the value comparison here is the check that the
    copy out of it is the column and not the buffer's leading entries.

    Also asserted, since no reference has an opinion on it: the mean of this array is a *scale*, so
    scaling the cloud must scale it by the same factor (both ``reconstruction`` call sites divide a
    length by it).
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((600, 3))

    distance_o3d = np.asarray(points_to_open3d(points_np).compute_nearest_neighbor_distance())

    points_wp = points_to_warp(points_np, device)
    distance_wp = od.neighbors.nearest_neighbor_distance(points_wp)

    assert distance_o3d.min() > 0.0  # non-vacuity: a random cloud has no coincident points
    assert np.allclose(distance_wp.numpy(), distance_o3d, rtol=1e-5, atol=1e-5)

    scaled_wp = points_to_warp(3.0 * points_np, device)
    scaled_distance_wp = od.neighbors.nearest_neighbor_distance(scaled_wp)
    assert np.allclose(scaled_distance_wp.numpy(), 3.0 * distance_wp.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parity("nearest_neighbor_distance", "meshlib")
def test_nearest_neighbor_distance_matches_meshlib(device: str) -> None:
    """
    Class B (an index where ordito returns a length): the same neighbour, exactly.

    ``findNClosestPointsPerPoint(cloud, 1)`` returns the *index* of each point's closest **other**
    point -- self is excluded, so the ``k=2``-and-drop-column-0 transform ordito's own
    implementation needs has no counterpart here -- and the distance is then taken from the index.
    Against a ``float64`` scipy oracle the reference's own indices reproduce its distances to
    **0.0**, so the only error in the comparison is ordito's ``float32`` arithmetic.

    **``numNei`` above 1 is a heap, not a sorted list**, which is why the pair is written at 1.
    Measured on this cloud: the ``k`` returned ids are exactly scipy's ``k`` nearest (set equality
    1.0000 at ``k=3``) but only **90.8 %** of the rows come back in decreasing-distance order, and
    the *nearest* is the last entry rather than the first. At ``numNei=2`` the last entry is the
    nearest in 100 % of rows -- a two-element heap is ordered by construction -- so a pair written
    at 2 would pass while resting on an accident of the heap size.
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((600, 3))

    cloud_ml = points_to_meshlib(points_np)
    nearest_ml = meshlib_indices_to_numpy(
        # MeshLib's stub omits ``VertId.__index__``, which it implements at runtime.
        mm.findNClosestPointsPerPoint(cloud_ml, 1)  # pyright: ignore[reportArgumentType]
    )
    assert nearest_ml.shape == (points_np.shape[0],)
    assert np.all(nearest_ml != np.arange(points_np.shape[0]))  # the closest *other* point
    distance_ml = np.linalg.norm(points_np[nearest_ml] - points_np, axis=1)

    points_wp = points_to_warp(points_np, device)
    distance_wp = od.neighbors.nearest_neighbor_distance(points_wp)

    assert distance_ml.min() > 0.0  # non-vacuity: a random cloud has no coincident points
    assert np.allclose(distance_wp.numpy(), distance_ml, rtol=1e-5, atol=1e-5)


def test_nearest_neighbor_distance_coincident_and_degenerate(device: str) -> None:
    """
    Not a library comparison: the two cases Open3D answers differently, plus the exact-zero one.

    Two coincident points are at distance zero from each other and both libraries say so, but a
    cloud of fewer than two points has no answer at all -- Open3D reports ``0.0`` there and this
    reports ``inf``, the value its own k-NN uses for a slot it could not fill. The divergence is
    documented rather than papered over, so it is pinned here rather than in the parity test.
    """
    coincident_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    coincident_wp = points_to_warp(coincident_np, device)
    distance_wp = od.neighbors.nearest_neighbor_distance(coincident_wp).numpy()
    assert np.array_equal(distance_wp, np.array([0.0, 0.0, 5.0], dtype=np.float32))

    for n_points in (0, 1):
        sparse_wp = wp.array(
            np.zeros((n_points, 3), dtype=np.float32), dtype=wp.vec3, device=device
        )
        answer_wp = od.neighbors.nearest_neighbor_distance(sparse_wp).numpy()
        assert answer_wp.shape == (n_points,)
        assert np.all(np.isinf(answer_wp))


@pytest.mark.parity("closest_pair", "meshlib", "scipy")
def test_closest_pair_matches_meshlib(device: str) -> None:
    """
    Class A against ``findTwoClosestPoints``, Class B against scipy, plus an exhaustive oracle.

    scipy's transform is the one its benchmark row runs: ``KDTree.query(points, k=2)`` gives every
    point's nearest *other* point, and an ``argmin`` over the second column reduces that to this
    function's answer -- the same reduction the sibling ``nearest_neighbor_distance`` group leaves
    out, which is what makes the two scipy rows differ by exactly it.

    Four answers compared, three of them independent implementations and one exhaustive: the
    ``O(n^2)`` NumPy distance matrix settles what the answer *is*, so the meshlib row proves the
    reference agrees rather than defining the truth. The cloud is random and therefore tie-free,
    which is what makes the *pair* comparable at all -- ties are a real possibility on a lattice and
    the two libraries need not break them the same way.

    Also asserts the pair is the argmin of
    [`nearest_neighbor_distance`][ordito.neighbors.nearest_neighbor_distance], the per-point form
    of the same query: that is the invariant tying the two entry points together, and it is what
    would break if the int64 key packing lost a bit.
    """
    rng = np.random.default_rng(5)
    points_np = rng.random((500, 3)).astype(np.float32)

    distance_np = np.linalg.norm(points_np[:, None, :] - points_np[None, :, :], axis=2)
    np.fill_diagonal(distance_np, np.inf)
    index_a_np, index_b_np = np.unravel_index(np.argmin(distance_np), distance_np.shape)

    points_wp = points_to_warp(points_np, device)
    index_a_wp, index_b_wp, distance_wp = od.neighbors.closest_pair(points_wp)
    assert {index_a_wp, index_b_wp} == {int(index_a_np), int(index_b_np)}
    assert np.isclose(distance_wp, distance_np[index_a_np, index_b_np], rtol=1e-5, atol=1e-5)

    pair_ml = mm.findTwoClosestPoints(points_to_meshlib(points_np))
    assert {int(pair_ml[0]), int(pair_ml[1])} == {index_a_wp, index_b_wp}

    per_point_wp = od.neighbors.nearest_neighbor_distance(points_wp).numpy()
    assert int(np.argmin(per_point_wp)) == index_a_wp
    assert np.isclose(float(per_point_wp.min()), distance_wp, rtol=1e-6, atol=1e-6)

    # scipy, through the k=2 self-query its benchmark row times, reduced by an argmin.
    distances_np, indices_np = KDTree(points_np).query(points_np, k=2)
    distances_np, indices_np = np.asarray(distances_np), np.asarray(indices_np)
    nearest_np = int(np.argmin(distances_np[:, 1]))
    assert nearest_np == index_a_np
    assert int(indices_np[nearest_np, 1]) == index_b_np
    assert np.isclose(float(distances_np[nearest_np, 1]), distance_wp, rtol=1e-6, atol=1e-6)


def test_closest_pair_ties_and_degenerate(device: str) -> None:
    """
    Not a library comparison: the tie rule and the ``n < 2`` guard.

    Exact duplicates make the distance zero at *two* indices, so the answer is a convention rather
    than a measurement: the ``int64`` key stores the index in its low half, so a ``min`` reduction
    defers to the smaller index. Pinned because a caller deduplicating a cloud reads ``index_a`` as
    "the first offender" and a change here would silently move it.
    """
    duplicated_np = np.array(
        [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32
    )
    duplicated_wp = points_to_warp(duplicated_np, device)
    index_a_wp, index_b_wp, distance_wp = od.neighbors.closest_pair(duplicated_wp)
    assert (index_a_wp, index_b_wp) == (0, 2)
    assert distance_wp == 0.0

    for n_points in (0, 1):
        with pytest.raises(ValueError, match="at least two points"):
            od.neighbors.closest_pair(
                wp.array(np.zeros((n_points, 3), dtype=np.float32), dtype=wp.vec3, device=device)
            )


def _geodesic_ball_neighborhoods_oracle(
    vertices_np: np.ndarray, faces_np: np.ndarray, radius: float, min_count: int = 6
) -> tuple[list[list[int]], np.ndarray]:
    """
    Host NumPy reference (the original ``_geodesic_ball_neighborhoods``) used as the oracle.

    Returns the per-vertex collected lists and the reference-neighbor array.
    """
    n = vertices_np.shape[0]
    adjacency: list[list[int]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for tri in faces_np:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            adjacency[a].append(b)
            adjacency[b].append(a)
    reference = np.array(
        [min(adjacency[i]) if adjacency[i] else i for i in range(n)], dtype=np.int32
    )
    per_vertex: list[list[int]] = []
    for i in range(n):
        center = vertices_np[i]
        visited = {i}
        queue = deque([i])
        collected: list[int] = []
        extras: list[tuple[float, int]] = []
        while queue:
            current = queue.popleft()
            collected.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                distance = float(np.linalg.norm(center - vertices_np[neighbor]))
                if distance < radius:
                    queue.append(neighbor)
                elif len(collected) < min_count:
                    heapq.heappush(extras, (distance, neighbor))
                visited.add(neighbor)
        while extras and len(collected) < min_count:
            _, cand = heapq.heappop(extras)
            collected.append(cand)
            for neighbor in adjacency[cand]:
                if neighbor in visited:
                    continue
                distance = float(np.linalg.norm(center - vertices_np[neighbor]))
                heapq.heappush(extras, (distance, neighbor))
                visited.add(neighbor)
        per_vertex.append(collected)
    return per_vertex, reference


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_geodesic_ball_neighborhoods(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """On-device geodesic balls match the NumPy/libigl oracle (per-row set equality)."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)

    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)

    radius = 3.0 * od.edges.mean_edge_length(vertices_wp, faces_wp)
    neighbor_indices_wp, offsets_wp, reference_wp = od.neighbors.geodesic_ball(
        vertices_wp, faces_wp, radius
    )
    per_vertex_oracle, reference_oracle = _geodesic_ball_neighborhoods_oracle(
        vertices_np, faces_np, radius
    )

    assert np.array_equal(reference_wp.numpy(), reference_oracle)

    neighbor_indices = neighbor_indices_wp.numpy()
    offsets = offsets_wp.numpy()
    n = vertices_np.shape[0]
    assert offsets.shape == (n + 1,)
    assert int(offsets[n]) == neighbor_indices.size
    for i in range(n):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        neighbors_wp = {int(x) for x in neighbor_indices[start:end]}
        assert neighbors_wp == set(per_vertex_oracle[i]), f"vertex {i} neighborhood differs"


@pytest.mark.parity(
    "query_geodesic_ball",
    "meshlib",
    benchmarked=False,
    reason="computeSurfaceDistances answers one source per call, so a row would time a Python loop "
    "over the vertex buffer rather than MeshLib -- the per-element rule. It is nonetheless the "
    "only reference that answers this question at all, and it agrees exactly, which is what "
    "this test "
    "records; its own timed row is in the heat_geodesic group as a fast-marching front.",
)
def test_geodesic_ball_matches_meshlib(device: str) -> None:
    """
    Class A on the ball's membership set: the same vertices, on every source sampled.

    MeshLib is the only reference that answers this at all -- ``computeSurfaceDistances`` runs a
    fast-marching front from a ``VertBitSet`` of sources and truncates it at ``maxDist``, so
    thresholding its field at the radius *is* the geodesic ball. Measured on ``icosphere(3)`` at
    three mean edges, over seven sampled sources: **31, 33, 33, 34, 33, 33, 34** on both sides,
    exactly.

    Two things make that agreement meaningful rather than lucky. The front is **not** Dijkstra over
    the edge graph -- it crosses triangle interiors -- so the two implementations reach the same set
    by different routes and a shared off-by-one in the traversal is excluded. And the radius is
    chosen so ``min_count`` cannot bind: that argument *expands* a ball to reach its floor, so at a
    radius where a ball holds six or fewer vertices ordito would legitimately return more than the
    distance field does. Every ball here holds 31 or more.

    It answers one source per call, which is why this samples rather than sweeping every vertex --
    and why there is no benchmark row.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3)
    vertices_np = np.asarray(sphere_tm.vertices)
    n_vertices = vertices_np.shape[0]

    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(sphere_tm.faces).ravel().astype(np.int32),
        dtype=wp.int32,
        device=device,
    )
    radius = 3.0 * float(od.edges.mean_edge_length(vertices_wp, faces_wp))
    neighbor_indices_wp, offsets_wp, _reference_wp = od.neighbors.geodesic_ball(
        vertices_wp, faces_wp, radius
    )
    neighbor_indices_np = neighbor_indices_wp.numpy()
    offsets_np = offsets_wp.numpy()

    mesh_ml = trimesh_to_meshlib(sphere_tm)
    for source in range(0, n_vertices, 97):
        ball_wp = set(
            neighbor_indices_np[int(offsets_np[source]) : int(offsets_np[source + 1])].tolist()
        )

        seeds_np = np.arange(n_vertices) == source
        seeds_ml = mm.VertBitSet(numpy_to_meshlib_bitset(seeds_np))
        distance_ml = meshlib_scalars_to_numpy(
            mm.computeSurfaceDistances(mesh_ml, seeds_ml, radius)
        )
        ball_ml = set(np.flatnonzero(distance_ml <= radius).tolist())

        assert len(ball_ml) > 6, (
            "min_count cannot be allowed to bind, or the two disagree by design"
        )
        assert ball_wp == ball_ml, f"source {source}"


def test_geodesic_ball_neighborhoods_overflow_warns(device: str) -> None:
    """Not a library comparison: a ball past the 512 cap clamps and counts its source once."""
    # A subdivided icosphere has > 512 vertices; a radius covering the whole mesh makes every
    # vertex's geodesic ball the entire connected component, exceeding the fixed scratch capacity.
    mesh_tm = tm.creation.icosphere(subdivisions=4)
    assert mesh_tm.vertices.shape[0] > 512
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    radius = 100.0 * float(mesh_tm.scale)

    # Every vertex's ball is the whole mesh, so every vertex is clipped exactly once.
    n_vertices = mesh_tm.vertices.shape[0]
    with pytest.warns(
        UserWarning, match=f"{n_vertices} of {n_vertices} neighborhoods exceeded the fixed capacity"
    ):
        _, offsets_wp, _ = od.neighbors.geodesic_ball(vertices_wp, faces_wp, radius)
    # Clamped, not crashed: every per-vertex count fits within the fixed capacity.
    assert np.diff(offsets_wp.numpy()).max() <= 512


@pytest.mark.parametrize("radius_scale", [3.0, 100.0], ids=["ball", "clipped"])
def test_geodesic_ball_chunks_agree(
    device: str, monkeypatch: pytest.MonkeyPatch, radius_scale: float
) -> None:
    """
    Ordito against ordito: sources split over several launches give the one-launch answer.

    A mesh larger than one chunk of sources is walked a chunk at a time, re-filling the shared
    scratch pools between launches and stitching the per-chunk rows together; no fixture in this
    suite is that large, so the chunk is forced down to an uneven split. The one-launch call carries
    the oracle (the libigl and meshlib comparisons above). Rows, offsets, reference neighbours and
    the clipped-source warning must be identical, on a ball and on a radius that clips every source.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=4)
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    radius = radius_scale * float(od.edges.mean_edge_length(vertices_wp, faces_wp))
    with warnings.catch_warnings(record=True) as whole_warnings:
        warnings.simplefilter("always")
        whole = od.neighbors.geodesic_ball(vertices_wp, faces_wp, radius)
    monkeypatch.setattr(od.neighbors, "_GEODESIC_BALL_CHUNK", 700)
    assert vertices_wp.size > 3 * 700
    with warnings.catch_warnings(record=True) as chunked_warnings:
        warnings.simplefilter("always")
        chunked = od.neighbors.geodesic_ball(vertices_wp, faces_wp, radius)
    for whole_wp, chunked_wp in zip(whole, chunked, strict=True):
        assert np.array_equal(whole_wp.numpy(), chunked_wp.numpy())
    assert [str(w.message) for w in whole_warnings] == [str(w.message) for w in chunked_warnings]
    assert len(whole_warnings) == (1 if radius_scale > 10.0 else 0)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity(
    "query_ball_bvh",
    "meshlib",
    benchmarked=False,
    reason="findPointsInBall reports each neighbour through a Python callback, so a benchmark row "
    "would time 20 000 queries' worth of callback dispatch rather than MeshLib's traversal -- the "
    "same reason section 6 bars a per-vertex Python loop from a benchmark row. There is no batched "
    "radius form: PointsProjector answers k=1 only (and does carry the query_nearest_bvh_k1 row), "
    "and findNClosestPointsPerPoint takes a neighbour count rather than a radius. scipy and open3d "
    "carry the timed rows for this group.",
)
def test_query_ball_matches_meshlib(device: str, backend: Literal["bvh", "hashgrid"]) -> None:
    """
    Class B: ``findPointsInBall`` reports its neighbours through a *callback*, one at a time.

    Two transforms, both forced by the interface. The results arrive by callback rather than as an
    array, so they are accumulated in Python and the answer is a *set* per query -- which is all
    either contract promises for a radius query. And the callback's ``distSq`` is squared, so the
    square root is the second transform, the same one the open3d comparison above makes.

    ``Ball3f`` is likewise built from a centre and a **squared** radius (``radiusSq``), which is the
    field a call written against a plain radius would silently get wrong by a square -- so the
    counts are asserted to be non-trivial rather than merely equal.

    Unlike open3d's ``KDTreeFlann``, MeshLib's ball search is **inclusive at exactly ``r``**, which
    is ordito's rule too: measured on three points placed at distance exactly 1.0 from the query,
    both return all three. The random cloud here cannot show that, so it is pinned separately in
    the second half of this test.
    """
    rng = np.random.default_rng(3)
    points_np = rng.random((400, 3)) * 3.0
    queries_np = rng.random((60, 3)) * 3.0
    radius = 0.4

    points_wp = points_to_warp(points_np, device)
    queries_wp = points_to_warp(queries_np, device)
    query_ball = partial(od.neighbors.query_ball, backend=backend)
    neighbours_wp, distances_wp = query_ball(points_wp, queries_wp, radius)

    cloud_ml = points_to_meshlib(points_np)
    total_found = 0
    for query_index, query_np in enumerate(queries_np):
        found_ml: list[tuple[int, float]] = []

        def collect(
            result_ml: mm.PointsProjectionResult,
            *_args: object,
            found: list[tuple[int, float]] = found_ml,
        ) -> mm.Processing:
            found.append((int(result_ml.vId), float(result_ml.distSq)))
            return mm.Processing.Continue

        ball_ml = mm.Ball3f()
        ball_ml.center = mm.Vector3f(*query_np.tolist())
        ball_ml.radiusSq = radius * radius
        mm.findPointsInBall(
            cloud_ml,
            ball_ml,
            cast(
                "mm.func_Processing_from_PointsProjectionResult_Vector3_float_Ball_Vector3_float",
                collect,
            ),
        )

        indices_ml = np.array([index for index, _distance in found_ml], dtype=np.int32)
        squared_ml = np.array([distance for _index, distance in found_ml])
        order_ml = np.argsort(indices_ml)
        order_wp = np.argsort(neighbours_wp[query_index].numpy())

        assert set(neighbours_wp[query_index].list()) == set(indices_ml.tolist())
        assert np.allclose(
            distances_wp[query_index].numpy()[order_wp],
            np.sqrt(squared_ml[order_ml]),
            rtol=1e-5,
            atol=1e-5,
        )
        total_found += indices_ml.size
    assert total_found > queries_np.shape[0]  # non-vacuity: the balls are not empty

    # The boundary rule, which the random cloud cannot reach: both are inclusive at exactly r.
    tie_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]])
    tie_wp = points_to_warp(tie_np, device)
    tie_indices_wp, _tie_distances_wp = query_ball(tie_wp, wp.vec3(0.0, 0.0, 0.0), 1.0)
    tie_found: list[int] = []

    def collect_tie(result_ml: mm.PointsProjectionResult, *_args: object) -> mm.Processing:
        tie_found.append(int(result_ml.vId))
        return mm.Processing.Continue

    tie_ball_ml = mm.Ball3f()
    tie_ball_ml.center = mm.Vector3f(0.0, 0.0, 0.0)
    tie_ball_ml.radiusSq = 1.0
    mm.findPointsInBall(
        points_to_meshlib(tie_np),
        tie_ball_ml,
        cast(
            "mm.func_Processing_from_PointsProjectionResult_Vector3_float_Ball_Vector3_float",
            collect_tie,
        ),
    )
    assert sorted(tie_found) == [0, 1, 2, 3]
    assert sorted(tie_indices_wp.list()) == [0, 1, 2, 3]


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parity("query_nearest_bvh_k1", "meshlib")
@pytest.mark.parity("query_nearest_hashgrid_k1", "meshlib")
@pytest.mark.parity(
    "query_nearest_bvh_k7",
    "meshlib",
    benchmarked=False,
    reason="findNClosestPointsPerPoint takes no query set -- it answers the cloud against itself, "
    "so at the benchmark's 20 000 displaced queries against 35 947 vertices it would be timing a "
    "different amount of work than every other row in the group. MeshLib's only batched form that "
    "accepts a query cloud is PointsProjector, which is k=1 and carries the "
    "query_nearest_bvh_k1 row; findFewClosestPoints is per query and would time a Python loop. "
    "scipy, igl and open3d carry the timed rows at k=7 and k=64.",
)
@pytest.mark.parity(
    "query_nearest_hashgrid_k7",
    "meshlib",
    benchmarked=False,
    reason="the same k=1-only limitation the query_nearest_bvh_k7 declaration above records, and "
    "for the same reason: PointsProjector is meshlib's only batched query-cloud form and it "
    "answers one neighbour. The hash-grid group therefore times scipy, igl and open3d at k=7, "
    "exactly as its BVH sibling does, while this test still compares both backends here.",
)
def test_query_nearest_matches_meshlib(
    device: str, backend: Literal["bvh", "hashgrid"], k: int
) -> None:
    """
    Class B: ``findNClosestPointsPerPoint`` is self-excluding, unordered, and cloud-against-itself.

    Three transforms, and each is a property of the reference rather than a formatting choice. It
    takes only a ``PointCloud`` and a neighbour count -- there is no separate query set -- so the
    comparison is the cloud against itself; it **excludes** each point from its own neighbour list,
    so ordito is asked for ``k + 1`` and its first column (the point itself, at distance 0) is
    dropped; and its rows are *not* distance-ordered, so the two are compared as sets per row.

    The flat ``Buffer_VertId`` is reshaped to ``(n_points, k)`` -- there is no shape on the returned
    buffer, only ``n * k`` ids in row-major order, so a wrong ``k`` reshapes silently into a
    plausible-looking answer. The self-column assert is what pins that: ordito's own first column
    must be the identity before anything is dropped.

    Both backends, as every other k-NN comparison in this file: meshlib's answer does not depend on
    which structure ordito asks, so the ``k=1`` markers cover the hash-grid group as well as the
    BVH one.

    This is the batched form deliberately. ``findFewClosestPoints`` is per query and a Python loop
    over it would time the loop, which is the same reason section 6 prefers ``mrmeshnumpy``'s
    batched curvature calls.
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((300, 3)) * 5.0
    points_wp = points_to_warp(points_np, device)
    query_nearest = partial(od.neighbors.query_nearest, backend=backend)

    neighbours_ml = np.array(
        [
            int(vertex_ml)
            for vertex_ml in mm.findNClosestPointsPerPoint(points_to_meshlib(points_np), k)
        ]
    ).reshape(points_np.shape[0], k)

    indices_wp, _distances_wp = query_nearest(points_wp, points_wp, k=k + 1)
    indices_np = indices_wp.numpy().reshape(points_np.shape[0], k + 1)

    assert np.array_equal(indices_np[:, 0], np.arange(points_np.shape[0]))  # the self column
    assert neighbours_ml.shape == (points_np.shape[0], k)
    for row_wp, row_ml in zip(indices_np[:, 1:], neighbours_ml, strict=True):
        assert set(row_wp.tolist()) == set(row_ml.tolist())

    if k != 1:
        return
    # The other batched form, and the only one that accepts a *query* cloud: exact at k=1, indices
    # and distances alike, which is what the query_nearest_bvh_k1 benchmark row times. The cloud
    # must be held in a name -- ``setPointCloud`` stores a raw pointer, so a temporary segfaults
    # rather than raising, the same trap ``PointsToMeshProjector`` carries in test_proximity.py.
    queries_np = rng.random((40, 3)) * 5.0
    queries_wp = points_to_warp(queries_np, device)
    queries_ml = mm.std_vector_Vector3_float()
    for query_np in queries_np:
        queries_ml.append(mm.Vector3f(*query_np.tolist()))
    cloud_ml = points_to_meshlib(points_np)  # must outlive the projector: it holds a raw pointer
    projector_ml = mm.PointsProjector()
    projector_ml.setPointCloud(cloud_ml)
    projections_ml = mm.std_vector_PointsProjectionResult()
    projector_ml.findProjections(projections_ml, queries_ml, mm.FindProjectionOnPointsSettings())

    nearest_wp, distances_wp = query_nearest(points_wp, queries_wp, k=1)
    assert np.array_equal(
        nearest_wp.numpy(), np.array([int(result.vId) for result in projections_ml])
    )
    assert np.allclose(
        distances_wp.numpy(),
        np.sqrt([result.distSq for result in projections_ml]),
        rtol=1e-5,
        atol=1e-5,
    )
