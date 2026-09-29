"""
Benchmarks for ``triwarp.neighbors``: the k-nearest-neighbour queries.

This is the direct gate for the k-NN search radius. Twelve call sites in ``triwarp.metrics``, the
point-cloud ICP path in ``triwarp.registration`` and three sites in ``triwarp.reconstruction`` all
bottom out here, and until this module existed none of them had a measurement isolating the query
kernel from the surrounding algorithm.

Setup
-----
``points`` is the mesh's own vertices; ``queries`` is a fixed 20 000-point subsample pushed off the
surface by 1 % of the bbox diagonal along a fixed direction, so no query coincides with a data
point (which would make ``k=1`` trivially certifiable at radius 0) while the spacing stays
realistic.

Axes
----
Nothing here is driven by a mesh property — the meshes are used only as point clouds — so the axes
are the structure's own parameters:

* **k**, as separate groups (``_k1`` / ``_k7``) rather than a sweep, because the two are different
  call sites and different *code*: the candidate row is register-resident with one kernel per
  row-size bucket (``kernels.neighbors.KNN_ROW_BUCKETS``). ``k=1`` is what ``distance.py`` and ICP
  use, ``k=7`` is ``ball_pivoting``'s seed table. The larger in-repo values (30, 64) are timed by
  [`test_points.py`](test_points.py), where the outlier statistics ask for them.
* **build against query**. The k-NN groups include the build, which is what a caller passing raw
  buffers pays; the ``*_from_points`` groups time the build alone, so subtracting them answers
  whether ICP's per-iteration index rebuild is the cost or a red herring.
* **structure resolution** — ``leaf_size``, ``grid_bins``, and ``radius`` for the ball queries. A
  leaf too large builds cheaply and traverses slowly; too few bins degenerates into a linear scan
  and too many pays for empty cells. Ball cost is the expected neighbour count, roughly
  ``density * radius ** 3``, and the two-phase count-then-fill pays it twice.

The ball groups use the ``*_with_offsets`` form: plain ``query_*_ball`` returns a Python ``list`` of
per-query arrays, adding ``O(n_queries)`` of host slicing that would swamp the kernel. That cost is
real but belongs to a different question.

For the k-NN groups the timed callable is everything the public function does, index build and
bounds reduction included — hiding the build would misreport the API. The build and ball groups
pass a prebuilt structure where the signature allows one, so they isolate what they name.

Reference
---------
**Every reference here is index-agnostic, so the BVH and hash-grid groups carry the same rows.**
None of the four exposes a structure choice, and the cloud, the queries and ``k`` are identical
between the two group families, so a ``query_*_bvh`` / ``query_*_hashgrid`` pair reads as **one**
comparison with triwarp's own index as the axis. The backend is a keyword, not a second entry
point, so the suffix is that keyword's value and the two rows are the same function timed twice;
``tests/test_neighbors.py`` parametrizes every comparison over ``backend`` and
``test_the_two_backends_agree`` pins that the pair returns identical answers, which is what makes
it a cost comparison rather than two measurements of two things.

**scipy** ``spatial.KDTree`` — the same reference the tests validate against, and the only exact
k-NN in the group with a matching signature. Its construction is inside the timed region for the
same reason triwarp's index build is.

**libigl** ``igl.knn`` is the second exact k-NN, byte-identical to ``KDTree``'s indices on a
tie-free cloud. Three things shape its rows:

* **Seven positional arguments, and the octree is built over the *data* cloud**:
  ``igl.knn(queries, points, k, *igl.octree(points)[:4])``. The build is timed inside, and also
  alone in ``bvh_from_points``.
* **``igl.octree``'s cost is mostly Python objects, and its variance is entirely the garbage
  collector** — it returns per-cell point lists as nested Python lists, several per input point, so
  enabling ``gc`` turns a tight distribution into a several-fold spread. **Read the igl rows'
  medians, not their minima**, and read the build-included numbers as pricing the binding as much
  as the search. Same hazard as the ``*_lists`` variants elsewhere, except here it lands on the only
  structure igl exposes for k-NN, so there is no array form to switch to.
* **Capped at ``bunny``**, one step below scipy's ``dragon``, because that churn grows with the cell
  count. ``k > n_points`` silently returns ``n_points`` columns rather than raising.

**open3d** ``o3d.core.nns.NearestNeighborSearch`` is the third exact k-NN and is **batched** — the
legacy ``KDTreeFlann``'s only query is a Python loop over ``search_knn_vector_3d``, several times
dearer because the loop times the interpreter. Its distances come back **squared** (as FLANN's do)
and ``fixed_radius_search`` returns a CSR-like ``(indices, distances, offsets)`` triple. The index
build sits inside the timed callable for the k-NN groups and outside for the ball group, matching
scipy; note ``fixed_radius_index(radius)`` bakes the radius in, so each radius pays its own build.
**trimesh** has no k-NN entry point.

``nearest_neighbor_distance`` is the one group whose queries are the cloud *itself* rather than the
subsample, because that is how its callers use it — once over the whole input, before
``reconstruction`` can pick a ball radius or an octree depth. open3d's counterpart searches point by
point in C++, serial but not a Python loop; triwarp goes from level with it to more than an order of
magnitude ahead across the ``scale`` axis, the ratio growing because triwarp's cost is nearly flat.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import pytorch3d.ops as p3d_ops
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree

import triwarp as tw
from conftest import BenchCase, points_torch_from_numpy, skip_larger_than

_SEED = 42
_N_QUERIES = 20_000
_OFFSET_FRACTION = 0.01

# BVH leaf sizes and hash-grid bin counts: the structure-resolution knobs.
_LEAF_SIZES = [4, 32]
_GRID_BINS = [32, 256]

# Ball radii as multiples of the mean edge length. Expected neighbours grow ~cubically, so these
# two points are roughly 8x apart in work.
_RADIUS_SCALES = [2.0, 4.0]

# pytorch3d's ``ball_query`` returns a fixed-width ``(N, P, K)`` block padded with ``-1`` rather
# than a ragged CSR, so ``K`` has to sit above the largest true neighbour count at the widest
# radius this module sweeps, measured on ``bunny`` at the widest scale. An
# undersized ``K`` does not raise -- it stops filling the row, which makes the call *faster* and
# the answer wrong -- so ``_run_pytorch3d_ball`` asserts the cap did not bite.
_PYTORCH3D_BALL_K = 256

# Weight spreads for ``query_weighted_nearest``, in mean edge lengths. The weighted query has to
# search out to ``answer + max_weight`` before it can certify, so this -- not the point count -- is
# what drives its deepening rounds.
_WEIGHT_SPREADS = [0.5, 4.0]

# Query boxes for ``query_bvh_box``, far fewer than ``_N_QUERIES``. open3d's
# ``AxisAlignedBoundingBox`` carries no index and no batch form, so its row is one Python call and
# one full linear scan *per box*: at 20 000 boxes over bunny that is 7e8 point tests a round. 256
# keeps the reference inside a second while still being a realistic "many query regions" shape.
_N_BOXES = 256

_queries_np_cache: dict[str, np.ndarray] = {}
_queries_wp_cache: dict[tuple[str, str], wp.array] = {}
_bvh_cache: dict[tuple[str, str], wp.Bvh] = {}
_kdtree_cache: dict[str, KDTree] = {}
_pcd_o3d_cache: dict[str, object] = {}
_cloud_ml_cache: dict[str, mm.PointCloud] = {}
_queries_ml_cache: dict[str, mm.std_vector_Vector3_float] = {}


def _queries_np(bench_case: BenchCase) -> np.ndarray:
    """Subsample the mesh vertices at a fixed seed and displace them off the surface."""
    name = bench_case.mesh_name
    if name not in _queries_np_cache:
        rng = np.random.default_rng(_SEED)
        vertices = bench_case.vertices_np
        count = min(_N_QUERIES, vertices.shape[0])
        indices = rng.choice(vertices.shape[0], size=count, replace=False)
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        offset = _OFFSET_FRACTION * diagonal * np.array([1.0, -1.0, 0.5]) / np.sqrt(2.25)
        _queries_np_cache[name] = np.ascontiguousarray(vertices[indices] + offset)
    return _queries_np_cache[name]


def _queries_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _queries_wp_cache:
        _queries_wp_cache[key] = wp.array(
            np.ascontiguousarray(_queries_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _queries_wp_cache[key]


def _cloud_ml(bench_case: BenchCase) -> mm.PointCloud:
    """
    Wrap the mesh vertices in a ``PointCloud``, cached per mesh so the reference outlives its users.

    ``PointsProjector.setPointCloud`` stores a **raw pointer**: handing it a temporary cloud leaves
    it reading freed memory and segfaults on a cloud this size rather than raising, which is the
    same rule ``PointsToMeshProjector`` follows in ``test_proximity.py``. The cache is that
    reference, and it keeps the lazily built tree warm as well.
    """
    if bench_case.mesh_name not in _cloud_ml_cache:
        from meshlib import mrmeshnumpy as mn

        _cloud_ml_cache[bench_case.mesh_name] = mn.pointCloudFromPoints(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        )
    return _cloud_ml_cache[bench_case.mesh_name]


def _queries_ml(bench_case: BenchCase) -> mm.std_vector_Vector3_float:
    """Build the query cloud as a MeshLib vector, cached: the fill is a per-point Python loop."""
    if bench_case.mesh_name not in _queries_ml_cache:
        queries_ml = mm.std_vector_Vector3_float()
        for query_np in _queries_np(bench_case):
            queries_ml.append(mm.Vector3f(*query_np.tolist()))
        _queries_ml_cache[bench_case.mesh_name] = queries_ml
    return _queries_ml_cache[bench_case.mesh_name]


def _bvh(bench_case: BenchCase) -> wp.Bvh:
    """Prebuilt BVH over the cloud -- an *input* for the ball group, timed on its own elsewhere."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _bvh_cache:
        _bvh_cache[key] = tw.neighbors.bvh_from_points(bench_case.vertices_wp)
    return _bvh_cache[key]


def _pcd_o3d(bench_case: BenchCase):
    """Open3D cloud over the same vertices, once per mesh: the pure query calls do not mutate it."""
    import open3d as o3d

    if bench_case.mesh_name not in _pcd_o3d_cache:
        _pcd_o3d_cache[bench_case.mesh_name] = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(bench_case.vertices_np)
        )
    return _pcd_o3d_cache[bench_case.mesh_name]


def _kdtree(bench_case: BenchCase) -> KDTree:
    """Prebuilt scipy KDTree over the same cloud, for the ball-query comparison."""
    if bench_case.mesh_name not in _kdtree_cache:
        _kdtree_cache[bench_case.mesh_name] = KDTree(bench_case.vertices_np)
    return _kdtree_cache[bench_case.mesh_name]


def _run_scipy(bench_case: BenchCase, k: int) -> None:
    queries_np = _queries_np(bench_case)
    points_np = bench_case.vertices_np
    distances_np, indices_np = bench_case.run(lambda: KDTree(points_np).query(queries_np, k=k))
    assert np.asarray(indices_np).shape[0] == queries_np.shape[0]
    assert np.asarray(distances_np).shape[0] == queries_np.shape[0]


def _run_igl_knn(bench_case: BenchCase, k: int) -> None:
    """``igl.knn`` with the octree build inside the timed region, as scipy's ``KDTree`` is."""
    skip_larger_than(bench_case, "bunny", "igl.octree allocates one Python list per octree cell")
    queries_np, points_np = _queries_np(bench_case), bench_case.vertices_np
    indices_igl = bench_case.run(
        lambda: igl.knn(queries_np, points_np, k, *igl.octree(points_np)[:4])
    )
    assert indices_igl.shape == (queries_np.shape[0], k)


def _run_o3d_nns_knn(bench_case: BenchCase, k: int) -> None:
    """Batched ``o3d.core.nns.knn_search`` with the index build inside the timed region."""
    import open3d as o3d

    queries_np, points_np = _queries_np(bench_case), bench_case.vertices_np
    queries_t = o3d.core.Tensor(queries_np)
    points_t = o3d.core.Tensor(points_np)

    def knn_o3d() -> tuple:
        nns = o3d.core.nns.NearestNeighborSearch(points_t)
        nns.knn_index()
        return nns.knn_search(queries_t, k)

    indices_o3d, _squared_o3d = bench_case.run(knn_o3d)
    assert indices_o3d.shape == (queries_np.shape[0], k)


def _run_pytorch3d_knn(bench_case: BenchCase, k: int) -> None:
    """
    Batched ``knn_points`` -- brute force, so there is no index build to place inside or outside.

    That is the whole reason the row is comparable at all: every other reference in this module
    times a structure build plus a traversal, and this one times ``n_points x n_queries`` distance
    evaluations. The upload is hoisted out, a small share of the query at these sizes, which matches
    what the triwarp branches do with their ``wp.array`` buffers.

    There is no CPU row to cap: pytorch3d is registered for its CUDA kernels alone (see the
    ``LIBRARIES`` block in [`conftest.py`](conftest.py)). Its host path is Theta(N x Q) with no
    pruning -- a clean quadratic, which is minutes per round at ``dragon`` and is why the row is
    absent rather than capped.
    """
    queries_p3d = points_torch_from_numpy(_queries_np(bench_case), bench_case.torch_device)
    points_p3d = points_torch_from_numpy(bench_case.vertices_np, bench_case.torch_device)
    nearest_p3d = bench_case.run(lambda: p3d_ops.knn_points(queries_p3d, points_p3d, K=k))
    assert nearest_p3d.idx.shape == (1, _queries_np(bench_case).shape[0], k)


def _run_pytorch3d_ball(bench_case: BenchCase, radius: float) -> None:
    """
    Batched ``ball_query`` at the same radius, with ``K`` above the largest true neighbour count.

    ``K`` is a real parameter of the answer and not just of the buffer: pytorch3d stops filling a
    row once it has ``K`` hits, so an undersized ``K`` makes the row *faster* and the answer wrong.
    It is asserted here rather than trusted, which is the section 13 rule about verifying values
    and not only timing.
    """
    queries_p3d = points_torch_from_numpy(_queries_np(bench_case), bench_case.torch_device)
    points_p3d = points_torch_from_numpy(bench_case.vertices_np, bench_case.torch_device)
    ball_p3d = bench_case.run(
        lambda: p3d_ops.ball_query(queries_p3d, points_p3d, K=_PYTORCH3D_BALL_K, radius=radius)
    )
    counts_p3d = (ball_p3d.idx[0] >= 0).sum(dim=1)
    assert int(counts_p3d.max()) < _PYTORCH3D_BALL_K, "the K cap truncated a row; raise it"


def _run_meshlib_projector(bench_case: BenchCase) -> None:
    """
    ``PointsProjector`` over the query cloud: meshlib's only batched ``k=1``, tree pre-warmed.

    Shared by both ``k=1`` groups because meshlib has no spatial-index choice to make -- the
    reference call is identical whichever structure triwarp uses on its side, which is the whole
    reason the BVH and hash-grid groups are comparable at all.

    The point cloud is held in a name because ``setPointCloud`` stores a raw pointer and a temporary
    segfaults; the projector and its lazily built tree are outside the timed callable, so the row
    prices the queries rather than the build.
    """
    cloud_ml = _cloud_ml(bench_case)  # cached: the projector does not own it
    queries_ml = _queries_ml(bench_case)
    projector_ml = mm.PointsProjector()
    projector_ml.setPointCloud(cloud_ml)
    settings_ml = mm.FindProjectionOnPointsSettings()

    def nearest_ml() -> mm.std_vector_PointsProjectionResult:
        results_ml = mm.std_vector_PointsProjectionResult()
        projector_ml.findProjections(results_ml, queries_ml, settings_ml)
        return results_ml

    nearest_ml()  # pre-warm the lazily built tree
    assert len(bench_case.run(nearest_ml)) == _queries_np(bench_case).shape[0]


@pytest.mark.benchmark(group="query_nearest_bvh_k1")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d", "meshlib", "pytorch3d")
def test_query_nearest_bvh_k1(bench_case: BenchCase) -> None:
    """
    ``k=1`` BVH k-NN — the exact call ICP and the Chamfer family make.

    At ``k = 1`` with no prebuilt ``accelerator`` this is not a radius search but one
    closest-point descent over [`mesh_from_points`][triwarp.neighbors.mesh_from_points], so the
    row times that tree's build plus the query, and needs neither the bounding box nor the density
    estimate the deepening search reads back.

    meshlib's ``PointsProjector`` is the only batched form it has that takes a *query* cloud, and
    it answers ``k=1`` only -- which is why meshlib appears in this group and not in the ``k7`` or
    ``k64`` ones (``tests/test_neighbors.py`` records that as a ``benchmarked=False`` claim). It is
    exact against triwarp on both indices and distances.

    **pytorch3d** is the fifth exact k-NN and the only reference with GPU kernels, so its
    ``-cuda`` row is the one GPU-against-GPU comparison in the module. It has no spatial structure
    on either device -- just the pairwise loop -- which makes it a *crossover* rather than a bar:
    pytorch3d wins at a small point count and loses by nearly two orders of magnitude at a large
    one. The ``LIBRARIES`` block in [`conftest.py`](conftest.py) carries a sweep of the radius
    walk, half of whose swing is its search-radius heuristic; this row answers through a pruned
    closest-point descent over [`mesh_from_points`][triwarp.neighbors.mesh_from_points] instead,
    which has no radius, so that split does not describe it. Its ``dists`` are **squared**, which
    is the named transform ``tests/test_neighbors.py::test_query_nearest_matches_pytorch3d``
    applies; its indices are exactly triwarp's. Absent from ``k64`` for the same reason meshlib is
    absent from ``k7``: there is no ``query_nearest_hashgrid_k64`` group to pair with, so two
    BVH-only markers would read as a different claim than the four here.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "meshlib":
        _run_meshlib_projector(bench_case)
        return
    if bench_case.kind == "pytorch3d":
        _run_pytorch3d_knn(bench_case, 1)
        return
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_nearest(points, queries, k=1, backend="bvh")
        )
        assert indices.shape == (queries.shape[0],)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 1)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 1)
    else:
        _run_scipy(bench_case, 1)


@pytest.mark.benchmark(group="query_nearest_hashgrid_k1")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d", "meshlib", "pytorch3d")
def test_query_nearest_hashgrid_k1(bench_case: BenchCase) -> None:
    """
    ``k=1`` hash-grid k-NN — the backend ``distance.py`` picks.

    The reference rows are the *same four calls* ``query_nearest_bvh_k1`` times, because none of the
    four references has a spatial-index choice to expose: the cloud, the 20 000 displaced queries
    and ``k`` are identical between the two groups and only triwarp's structure differs. So the pair
    of groups reads as one comparison with triwarp's index as the axis, which is what
    ``tests/test_neighbors.py`` asserts by parametrizing every k-NN comparison over both backends.

    **pytorch3d** is the fifth exact k-NN and the only reference with GPU kernels, so its
    ``-cuda`` row is the one GPU-against-GPU comparison in the module. It has no spatial structure
    on either device -- just the pairwise loop -- which makes it a *crossover* rather than a bar:
    pytorch3d wins at a small point count and loses by nearly two orders of magnitude at a large
    one. Half of that swing is triwarp's own search-radius heuristic and not brute force scaling;
    the ``LIBRARIES`` block in [`conftest.py`](conftest.py)
    carries the sweep and the reading. Its ``dists`` are **squared**, which is the named transform
    ``tests/test_neighbors.py::test_query_nearest_matches_pytorch3d`` applies; its indices are
    exactly triwarp's. Absent from ``k64`` for the same reason meshlib is absent from ``k7``: there
    is no ``query_nearest_hashgrid_k64`` group to pair with, so two BVH-only markers would read as
    a different claim than the four here.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "meshlib":
        _run_meshlib_projector(bench_case)
        return
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_nearest(points, queries, k=1)
        )
        assert indices.shape == (queries.shape[0],)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 1)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 1)
    else:
        _run_scipy(bench_case, 1)


@pytest.mark.benchmark(group="query_nearest_bvh_k7")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d", "pytorch3d")
def test_query_nearest_bvh_k7(bench_case: BenchCase) -> None:
    """
    ``k=7`` BVH k-NN: the tree side of the backend pair ``query_nearest_hashgrid_k7`` completes.

    ``ball_pivoting``'s ``k=7`` seed-candidate table is a self-query on the default backend, which
    this displaced query set does not reproduce; the pair times the two indexes on one input.

    **pytorch3d** is the fifth exact k-NN and the only reference with GPU kernels, so its
    ``-cuda`` row is the one GPU-against-GPU comparison in the module. It has no spatial structure
    on either device -- just the pairwise loop -- which makes it a *crossover* rather than a bar:
    pytorch3d wins at a small point count and loses by nearly two orders of magnitude at a large
    one. Half of that swing is triwarp's own search-radius heuristic and not brute force scaling;
    the ``LIBRARIES`` block in [`conftest.py`](conftest.py)
    carries the sweep and the reading. Its ``dists`` are **squared**, which is the named transform
    ``tests/test_neighbors.py::test_query_nearest_matches_pytorch3d`` applies; its indices are
    exactly triwarp's. Absent from ``k64`` for the same reason meshlib is absent from ``k7``: there
    is no ``query_nearest_hashgrid_k64`` group to pair with, so two BVH-only markers would read as
    a different claim than the four here.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        _run_pytorch3d_knn(bench_case, 7)
        return
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_nearest(points, queries, k=7, backend="bvh")
        )
        assert indices.shape == (queries.shape[0], 7)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 7)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 7)
    else:
        _run_scipy(bench_case, 7)


@pytest.mark.benchmark(group="query_nearest_bvh_k64")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d")
def test_query_nearest_bvh_k64(bench_case: BenchCase) -> None:
    """
    ``k=64`` BVH k-NN — the largest register-row bucket, and the k axis's far end.

    The candidate row costs ``2 * k`` registers, so this is the last ``k`` the row fits in them; at
    65 the search falls back to the global-memory-row kernel and the cost per neighbour jumps
    severalfold. The group exists to keep that boundary visible:
    without it the suite's k axis stops at 7 and the regime the row storage governs is unmeasured.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_nearest(points, queries, k=64, backend="bvh")
        )
        assert indices.shape == (queries.shape[0], 64)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 64)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 64)
    else:
        _run_scipy(bench_case, 64)


@pytest.mark.benchmark(group="query_nearest_hashgrid_k7")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d", "pytorch3d")
def test_query_nearest_hashgrid_k7(bench_case: BenchCase) -> None:
    """
    ``k=7`` hash-grid k-NN.

    Same three references as ``query_nearest_bvh_k7``, and for the reason given on the ``k1`` group:
    the reference call does not change with triwarp's index. meshlib is absent here rather than
    exempt -- its only batched query-cloud form is ``k=1`` (see ``query_nearest_bvh_k1``).

    **pytorch3d** is the fifth exact k-NN and the only reference with GPU kernels, so its
    ``-cuda`` row is the one GPU-against-GPU comparison in the module. It has no spatial structure
    on either device -- just the pairwise loop -- which makes it a *crossover* rather than a bar:
    pytorch3d wins at a small point count and loses by nearly two orders of magnitude at a large
    one. Half of that swing is triwarp's own search-radius heuristic and not brute force scaling;
    the ``LIBRARIES`` block in [`conftest.py`](conftest.py)
    carries the sweep and the reading. Its ``dists`` are **squared**, which is the named transform
    ``tests/test_neighbors.py::test_query_nearest_matches_pytorch3d`` applies; its indices are
    exactly triwarp's. Absent from ``k64`` for the same reason meshlib is absent from ``k7``: there
    is no ``query_nearest_hashgrid_k64`` group to pair with, so two BVH-only markers would read as
    a different claim than the four here.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_nearest(points, queries, k=7)
        )
        assert indices.shape == (queries.shape[0], 7)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 7)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 7)
    else:
        _run_scipy(bench_case, 7)


@pytest.mark.benchmark(group="query_weighted_nearest")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("weight_spread", _WEIGHT_SPREADS)
def test_query_weighted_nearest(bench_case: BenchCase, weight_spread: float) -> None:
    """
    Additively weighted nearest site, and the axis is the **weight spread** rather than the mesh.

    That axis is the whole cost model. Certification needs the search radius to exceed the answer's
    distance by ``max_weight``, so a wide weight distribution forces more deepening rounds than a
    narrow one over the identical geometry: the two points here are 0.5 and 4 mean edges of spread,
    and everything else is held fixed. Read the ratio between them, not the absolute numbers.

    **No reference row.** MeshLib's ``findClosestWeightedPoint`` is the only additively weighted
    query in any installed library, and it reads each site's weight through a Python callback *and*
    answers one query per call, so a row would price the interpreter twice over; the correctness
    comparison lives in ``tests/test_neighbors.py`` as a ``benchmarked=False`` claim. Read this
    group against ``query_nearest_bvh_k1`` instead, which is the same traversal with the weight
    term dropped.

    A wider weight spread costs several times as much, so the axis is doing what it was chosen for.
    Read each mesh's row in isolation: a row that follows another case in the same process can
    report an order of magnitude above what it costs alone.

    The large-mesh row was two orders of magnitude worse before a float32 stall in the deepening
    loop was fixed (see ``kernels/neighbors.query_weighted_nearest_neighbors``): a fraction of a
    percent of queries never certified, burned the whole attempt budget and then scanned the entire
    cloud. This group is what found it, which
    is the argument for the benchmark landing with the function rather than after it.
    """
    skip_larger_than(bench_case, "dragon")
    points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
    rng = np.random.default_rng(_SEED)
    weights = wp.array(
        (rng.random(bench_case.n_vertices) * weight_spread * bench_case.mean_edge).astype(
            np.float32
        ),
        dtype=wp.float32,
        device=bench_case.device,
    )
    bvh = _bvh(bench_case)
    max_weight = float(weight_spread * bench_case.mean_edge)
    indices, distances = bench_case.run(
        lambda: tw.neighbors.query_weighted_nearest(
            points, weights, queries, max_weight=max_weight, accelerator=bvh
        )
    )
    assert indices.shape == (queries.shape[0],)
    assert distances.shape == (queries.shape[0],)


@pytest.mark.benchmark(group="bvh_from_points")
@pytest.mark.benchlibs("triwarp", "scipy", "igl")
@pytest.mark.parametrize("leaf_size", _LEAF_SIZES)
def test_bvh_from_points(bench_case: BenchCase, leaf_size: int) -> None:
    """
    Structure build alone: the cost a caller amortizes, or fails to.

    Subtract this from ``query_nearest_bvh_k1`` to get the query in isolation. Neither reference
    takes a leaf-size parameter, so each one's two rows are identical by construction and are there
    as fixed bars; triwarp's own slope is the other half of the ``leaf_size`` trade-off.

    ``igl.octree`` is the structure ``igl.knn`` consumes, so this row is also what the k-NN groups'
    build-included numbers carry -- nearly all of them -- and most of *that* is the Python lists it
    returns rather than the tree; see the module docstring.
    """
    skip_larger_than(bench_case, "dragon", "the scipy reference builds single-threaded")
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        bvh = bench_case.run(lambda: tw.neighbors.bvh_from_points(points, leaf_size=leaf_size))
        assert bvh is not None
    elif bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl.octree is superlinear in the point count")
        points_np = bench_case.vertices_np
        point_indices_igl, _, _, _ = bench_case.run(lambda: igl.octree(points_np))[:4]
        assert len(point_indices_igl) > 0
    else:
        points_np = bench_case.vertices_np
        assert bench_case.run(lambda: KDTree(points_np)) is not None


@pytest.mark.benchmark(group="hashgrid_from_points")
@pytest.mark.benchlibs("triwarp", "scipy", "igl")
@pytest.mark.parametrize("grid_bins", _GRID_BINS)
def test_hashgrid_from_points(bench_case: BenchCase, grid_bins: int) -> None:
    """
    The hash-grid build, swept over its bin count: the other structure's amortization floor.

    The same two reference structures ``bvh_from_points`` times, and for the same reason: each is
    the index its library's k-NN query consumes, so subtracting this group from
    ``query_nearest_hashgrid_k1`` isolates the query on both sides of the ratio. Neither reference
    has a bin-count parameter, so each one's two rows are identical by construction and stand as
    fixed bars against triwarp's slope -- the convention ``bvh_from_points`` already uses for
    ``leaf_size``.
    """
    skip_larger_than(bench_case, "dragon", "the scipy reference builds single-threaded")
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl.octree is superlinear in the point count")
        points_np = bench_case.vertices_np
        point_indices_igl, _, _, _ = bench_case.run(lambda: igl.octree(points_np))[:4]
        assert len(point_indices_igl) > 0
        return
    if bench_case.kind == "scipy":
        points_np = bench_case.vertices_np
        assert bench_case.run(lambda: KDTree(points_np)) is not None
        return
    points = bench_case.vertices_wp
    radius = _RADIUS_SCALES[0] * bench_case.mean_edge
    grid = bench_case.run(
        lambda: tw.neighbors.hashgrid_from_points(points, radius, grid_bins=grid_bins)
    )
    assert grid is not None


@pytest.mark.benchmark(group="query_ball_bvh")
@pytest.mark.benchlibs("triwarp", "scipy", "open3d", "pytorch3d")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES)
def test_query_ball_bvh(bench_case: BenchCase, radius_scale: float) -> None:
    """
    Radius query over a prebuilt BVH: cost is the neighbour count, so ~8x between the radii.

    All three structures are prebuilt: triwarp's BVH, scipy's cached ``KDTree``, and open3d's
    ``fixed_radius_index`` -- the last per radius, because that index bakes the radius in.
    ``fixed_radius_search`` returns a CSR-like triple, which is exactly the ``*_with_offsets``
    layout triwarp's row times, so neither side pays per-query host slicing.

    **pytorch3d**'s ``ball_query`` is the fourth structure-free row and the only GPU one, and it is
    the group where the absence of an index costs most: one to two orders of magnitude behind
    triwarp with no crossover at either end -- unlike its k-NN, which triwarp loses at the small
    end. Two conventions shape the row and both
    are asserted rather than assumed: it takes a fixed ``K`` and pads with ``-1`` rather than
    returning a ragged CSR, so ``K`` is passed above the largest true neighbour count and a smaller
    one would silently truncate; and it fills in ascending **index** order, not by distance, which
    is why ``tests/test_neighbors.py::test_query_ball_matches_pytorch3d`` compares sets.
    """
    skip_larger_than(bench_case, "bunny", "the neighbour count grows cubically with the radius")
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        bvh = _bvh(bench_case)
        neighbors, _distances, offsets = bench_case.run(
            lambda: tw.neighbors.query_ball_with_offsets(points, queries, radius, accelerator=bvh)
        )
        assert offsets.shape[0] == int(queries.shape[0]) + 1
        assert neighbors.shape[0] >= 0
    elif bench_case.kind == "open3d":
        import open3d as o3d

        queries_t = o3d.core.Tensor(_queries_np(bench_case))
        nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(bench_case.vertices_np))
        assert nns.fixed_radius_index(radius)
        _indices_o3d, _squared_o3d, offsets_o3d = bench_case.run(
            lambda: nns.fixed_radius_search(queries_t, radius)
        )
        assert offsets_o3d.shape[0] == queries_t.shape[0] + 1
    else:
        tree, queries_np = _kdtree(bench_case), _queries_np(bench_case)
        found = bench_case.run(lambda: tree.query_ball_point(queries_np, radius))
        assert len(found) == queries_np.shape[0]


@pytest.mark.benchmark(group="query_bvh_box")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_query_bvh_box(bench_case: BenchCase) -> None:
    """
    Per-query axis-aligned box over a prebuilt BVH -- an indexed batch against an unindexed scan.

    The box analogue of ``query_ball_bvh``, and unlike that group the reference is not another tree:
    open3d's ``AxisAlignedBoundingBox.get_point_indices_within_bounding_box`` walks every point, so
    the row prices *index versus no index* as much as it prices the traversal, and the loop over
    ``_N_BOXES`` is on open3d's side of the line. That is the honest comparison available -- neither
    ``o3d.core.nns`` nor scipy's ``KDTree`` has a box query at all -- and it is why the box count is
    256 rather than the module's usual 20 000.

    The boxes are asymmetric about their centres (``-2`` to ``+3`` mean edges), so a caller cannot
    read this as the cube query with a different name: the corners are per query and the two sides
    of each axis differ.

    triwarp's two rows being *identical* while open3d's move with the point count is the whole
    content of the row -- the BVH descent does not see the cloud size at this box count, and the
    scan does.
    """
    skip_larger_than(bench_case, "bunny", "the open3d row is O(boxes x points) with no index")
    centers_np = _queries_np(bench_case)[:_N_BOXES]
    edge = bench_case.mean_edge
    lower_np = np.ascontiguousarray(centers_np - 2.0 * edge)
    upper_np = np.ascontiguousarray(centers_np + 3.0 * edge)

    if bench_case.kind == "open3d":
        import open3d as o3d

        cloud_points = _pcd_o3d(bench_case).points
        boxes_o3d = [
            o3d.geometry.AxisAlignedBoundingBox(lower_np[i], upper_np[i])
            for i in range(centers_np.shape[0])
        ]

        def box_query_o3d() -> int:
            return sum(
                len(box.get_point_indices_within_bounding_box(cloud_points)) for box in boxes_o3d
            )

        assert bench_case.run(box_query_o3d) >= 0
        return

    bvh = _bvh(bench_case)
    lower_wp = wp.array(lower_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device)
    upper_wp = wp.array(upper_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device)
    indices, offsets = bench_case.run(lambda: tw.neighbors.query_bvh_box(bvh, lower_wp, upper_wp))
    assert offsets.shape == (centers_np.shape[0] + 1,)
    assert indices.shape[0] >= 0


@pytest.mark.benchmark(group="query_ball_hashgrid")
@pytest.mark.benchlibs("triwarp", "scipy", "open3d", "pytorch3d")
@pytest.mark.parametrize("grid_bins", _GRID_BINS)
def test_query_ball_hashgrid(bench_case: BenchCase, grid_bins: int) -> None:
    """
    The hash-grid ball query, swept over the bin count at a fixed radius.

    Too few bins and each cell holds enough points that the query degenerates into a linear scan;
    too many and the build pays for cells nothing lands in. Where the optimum sits depends on the
    cloud's density, so this pair is the cheapest way to see which side of it the default is on.

    Both references are the ones ``query_ball_bvh`` times, at the same radius over the same cloud --
    scipy's cached ``KDTree`` and open3d's ``fixed_radius_index``, each prebuilt so no row pays for
    a structure. Neither has a bin count, so their two rows per mesh are identical bars and only
    triwarp's move; that is the axis this group exists for.

    **pytorch3d**'s ``ball_query`` is the fourth structure-free row and the only GPU one, and it is
    the group where the absence of an index costs most: one to two orders of magnitude behind
    triwarp with no crossover at either end -- unlike its k-NN, which triwarp loses at the small
    end. Two conventions shape the row and both
    are asserted rather than assumed: it takes a fixed ``K`` and pads with ``-1`` rather than
    returning a ragged CSR, so ``K`` is passed above the largest true neighbour count and a smaller
    one would silently truncate; and it fills in ascending **index** order, not by distance, which
    is why ``tests/test_neighbors.py::test_query_ball_matches_pytorch3d`` compares sets.
    """
    skip_larger_than(bench_case, "bunny", "the neighbour count grows cubically with the radius")
    radius = _RADIUS_SCALES[0] * bench_case.mean_edge
    if bench_case.kind == "pytorch3d":
        _run_pytorch3d_ball(bench_case, radius)
        return
    if bench_case.kind == "open3d":
        import open3d as o3d

        queries_t = o3d.core.Tensor(_queries_np(bench_case))
        nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(bench_case.vertices_np))
        assert nns.fixed_radius_index(radius)
        _indices_o3d, _squared_o3d, offsets_o3d = bench_case.run(
            lambda: nns.fixed_radius_search(queries_t, radius)
        )
        assert offsets_o3d.shape[0] == queries_t.shape[0] + 1
        return
    if bench_case.kind == "scipy":
        tree, queries_np = _kdtree(bench_case), _queries_np(bench_case)
        found = bench_case.run(lambda: tree.query_ball_point(queries_np, radius))
        assert len(found) == queries_np.shape[0]
        return
    points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
    grid = tw.neighbors.hashgrid_from_points(points, radius, grid_bins=grid_bins)
    neighbors, _distances, offsets = bench_case.run(
        lambda: tw.neighbors.query_ball_with_offsets(points, queries, radius, accelerator=grid)
    )
    assert offsets.shape[0] == int(queries.shape[0]) + 1
    assert neighbors.shape[0] >= 0


@pytest.mark.benchmark(group="query_geodesic_ball")
@pytest.mark.benchlibs("triwarp")
def test_query_geodesic_ball(bench_case: BenchCase) -> None:
    """
    The mesh-graph ball: a BFS that enqueues a neighbour only when it is inside the radius.

    The one query here that is not a spatial one, and the reason it is not: a Euclidean
    hash-grid query at the same radius would also return vertices across a fold of the surface
    -- the opposite wall of a torus tube -- which is what corrupts the quadric fit in
    ``curvature.principal_curvature``, its only caller. So this row prices the surface-aware
    alternative to the ball groups above rather than a variant of them.

    There is no reference row, and it is a cost objection rather than an absence: meshlib's
    ``computeSurfaceDistances`` truncated at ``maxDist`` *is* this ball and agrees with it exactly
    (``tests/test_neighbors.py``), but it answers **one source per call**, so a row would
    time a Python loop over the vertex buffer rather than MeshLib -- the per-element rule. Its own
    timed row is in the ``heat_geodesic`` group, where the same function is priced as a
    fast-marching front over the whole mesh.
    """
    skip_larger_than(bench_case, "happy_buddha")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    radius = 5.0 * float(tw.edges.mean_edge_length(vertices, faces))
    _, offsets, _ = bench_case.run(lambda: tw.neighbors.geodesic_ball(vertices, faces, radius))
    assert offsets.shape == (vertices.shape[0] + 1,)


@pytest.mark.benchmark(group="closest_pair")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "meshlib", "scipy")
def test_closest_pair(bench_case: BenchCase) -> None:
    """
    The cloud's *minimum* spacing: the same ``k=2`` self-query, reduced to one pair.

    Read next to ``nearest_neighbor_distance``, which is the same query returning the whole column:
    the difference between the two rows is one ``int64`` key pass, one device reduction and two
    single-element readbacks, so the pair prices what reducing on the device costs against handing
    the array back.

    meshlib's ``findTwoClosestPoints`` is the same answer and is batched and callback-free. Its
    point tree is cached on the cloud, so the row drops it per round exactly as the
    ``nearest_neighbor_distance`` row does -- otherwise the second round onward would time a query
    against a warm tree while triwarp rebuilds its BVH inside every call.

    scipy reaches the same answer through the ``k=2`` self-query its sibling
    ``nearest_neighbor_distance`` row already times, plus an ``argmin`` over the second column. So
    the two groups share a reference *and* an input, and the difference between their scipy rows is
    exactly that reduction -- which is the same thing the triwarp pair measures, on the other side
    of the host/device line. Its ``KDTree`` build is inside the timed callable, like triwarp's BVH.

    Unlike most reductions in this suite the GPU row wins at every size, and the reason is that the
    query dominates rather than the reduce. Read meshlib's *medians* rather than its minima:
    dropping the cached tree per round leaves it with an order-of-magnitude spread where triwarp's
    is a few percent.
    """
    if bench_case.kind == "scipy":
        points_np = bench_case.vertices_np

        def closest_pair_np() -> tuple[int, int, float]:
            distances, indices = KDTree(points_np).query(points_np, k=2)
            nearest = int(np.argmin(distances[:, 1]))
            return nearest, int(indices[nearest, 1]), float(distances[nearest, 1])

        _first, _second, spacing = bench_case.run(closest_pair_np)
        assert spacing >= 0.0
        return
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # held in a name: the tree is a raw pointer into it

        def uncached_cloud_ml() -> mm.PointCloud:
            cloud_ml.invalidateCaches()
            return cloud_ml

        pair_ml = bench_case.run(mm.findTwoClosestPoints, setup=uncached_cloud_ml)
        assert len(pair_ml) == 2
        return
    points = bench_case.vertices_wp
    index_a, index_b, distance = bench_case.run(lambda: tw.neighbors.closest_pair(points))
    assert 0 <= index_a < bench_case.n_vertices
    assert 0 <= index_b < bench_case.n_vertices
    assert distance >= 0.0


@pytest.mark.benchmark(group="nearest_neighbor_distance")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d", "meshlib")
def test_nearest_neighbor_distance(bench_case: BenchCase) -> None:
    """
    The cloud's own spacing: a ``k=2`` self-query, keeping the second column.

    A *self*-query, so unlike every group above the query count is the cloud size rather than
    ``_N_QUERIES`` -- which is the point, since this is what ``reconstruction`` calls once on the
    whole input before it can pick a radius or an octree depth. Everything past the k-NN search is a
    strided column copy, so the ratio against ``query_nearest_bvh_k1`` prices that tail.

    open3d's ``compute_nearest_neighbor_distance`` builds a ``KDTreeFlann`` and then searches one
    point at a time in C++ -- so it is serial, but not the Python-per-query loop the legacy tree
    would be from this side, and it is the same quantity to 9e-09.

    meshlib's ``findNClosestPointsPerPoint(cloud, 1)`` is the batched, multi-threaded form of the
    same self-query and returns the neighbour *index*; the distance is one subtraction away and is
    not timed on either side. **Its point tree is cached on the cloud and the row drops it per
    round**, which is worth several-fold and is what makes the three rows comparable: triwarp builds
    a BVH and open3d a ``KDTreeFlann`` inside their own calls, where meshlib would otherwise reuse a
    cached one. ``invalidateCaches`` is the lever rather than a fresh ``PointCloud`` per round,
    because rebuilding the cloud itself adds the whole point allocation and with it a spread that
    swamps what is being measured.
    At the large end triwarp leads the multi-threaded reference by several times and the serial one
    by an order of magnitude. At the small end all three land within noise of each other -- the
    cloud is too small to fill the GPU or to pay for a thread pool, which is where a ratio here
    stops meaning anything.
    """
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # held in a name: the tree is a raw pointer into it

        def uncached_cloud_ml() -> mm.PointCloud:
            """Drop the lazily built tree so the timed call pays for building one."""
            cloud_ml.invalidateCaches()
            return cloud_ml

        neighbours_ml = bench_case.run(
            lambda cloud: mm.findNClosestPointsPerPoint(cloud, 1), setup=uncached_cloud_ml
        )
        assert neighbours_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "open3d":
        cloud = _pcd_o3d(bench_case)
        distance_o3d = bench_case.run(cloud.compute_nearest_neighbor_distance)
        assert len(distance_o3d) == bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    distance = bench_case.run(lambda: tw.neighbors.nearest_neighbor_distance(points))
    assert distance.shape == (bench_case.n_vertices,)
