"""
Benchmarks for ``triwarp.proximity`` hot paths.

Covers winding number, signed distance, tangent spheres and geodesic-ball queries. The AABB
reduction moved to [`test_bounds.py`](test_bounds.py), where ``triwarp.bounds`` lives.
``winding_number`` is O(n_queries x n_faces) even in the tiled variant, so ``lucy`` is skipped; the
pinned serial (``tiled=False``) path is additionally capped at ``bunny`` because one thread per
query walking every face takes minutes beyond that.

Read ``winding_number`` and ``signed_distance_on_mesh[winding]`` together: both answer an
inside/outside question from solid angle, but the first accumulates it exactly over every face while
the second lets Warp's BVH traversal approximate it and keeps only the sign. The gap between them is
the cost of needing the winding *value* rather than just its sign.

Open3D has no equivalent for anything in this module: no generalized winding number -- its
inside/outside test is raycasting-based, a different algorithm answering a coarser question -- and
no tangent-sphere, local-thickness or geodesic-ball query at all.

``signed_distance_on_mesh`` has **two** references, and they are the only two that exist: libigl's
``igl.signed_distance``, whose ``sign_type`` axis maps onto triwarp's ``sign_mode`` one-for-one, and
pymeshlab's, whose sign rule is a third algorithm and therefore appears once.

**One thing libigl's signed distance does that no assert may ignore:** for both winding-based sign
types it returns ``(1 - 2w) * d`` rather than ``sign(1 - 2w) * d``, with ``w`` the *continuous*
winding number. So its magnitude is only ``|d|`` where ``w`` is exactly 0 or 1, and near the surface
it is scaled down -- measured on ``bunny_decimated``, ``|S|`` deviates from the pseudonormal type's
by **2.7e-2 of the bbox diagonal** for both winding types, while the pseudonormal type agrees with
triwarp to **8e-8**. That is why the parity oracle is the pseudonormal type and why the winding row
here is a *cost* comparison only.

**And the Barnes-Hut approximation does pay**, which is worth recording because a smaller probe had
suggested otherwise. At this module's query count ``igl.fast_winding_number`` is 2.6-3.6x
``igl.winding_number`` for a maximum winding deviation of 0.004. It is not a row of its own because
triwarp exposes no approximate-winding entry point to put opposite it (``winding_number`` is the
exact sum; the Barnes-Hut walk exists only inside ``signed_distance_on_mesh(sign_mode="winding")``),
but it is the number to weigh a fast-winding port against.

``compute_scalar_by_distance_from_another_mesh_per_vertex(signeddist=True)`` measures every vertex
of one mesh against another, so the query points go in as a second, face-less mesh and the answer
comes back on their vertex scalar attribute. Three things to read its row against:

- **Its sign is a third algorithm.** MeshLab takes the dot product with the reference normal at the
  closest point -- neither triwarp's 5-ray parity test nor its Barnes-Hut winding accumulation. So
  it appears once rather than twice, in the ``parity`` row.
- **Its per-query cost grows with the reference mesh**, an order of magnitude across the scan sweep
  at a fixed query count, while being cleanly linear in the query count at a fixed mesh. A
  closest-point query that is *not* sublinear in the face count is the opposite of what triwarp's
  BVH does -- two orders of magnitude against it, widening with the mesh -- which is why it is
  capped at ``bunny`` with ``rounds=3``.
- It writes only the vertex scalar, so the two-mesh MeshSet is built once and shared.

``shape_diameter`` is the one group whose reference is nowhere near a millisecond:
``compute_scalar_by_shape_diameter_function_per_vertex`` is the most expensive per-vertex filter
MeshLab ships, because it traces 64 rays from every vertex on one core. That is exactly the shape of
work a GPU should win outright, which is why the port exists. Two caveats: its ``cone_amplitude``
parameter is a **no-op** in the 2025.07 build (byte-identical output at 90 and 120 degrees), so its
cone is whatever it is, and it writes only the vertex scalar, so the MeshSet is shared. The ``rays``
sweep is the axis, not the mesh: the cost is exactly linear in it on both sides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.spatial import Delaunay

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase, BenchLibrary, mesh_ml_from_numpy, skip_larger_than

if TYPE_CHECKING:
    from typing_extensions import Buffer

_QUERY_SEED = 42
_N_QUERIES = 10_000

# Query counts for ``winding_number``, the one genuinely O(queries x faces) function here: the
# other half of its product, swept independently of the mesh.
_N_QUERIES_SWEEP = [10_000, 100_000]

_query_cache: dict[tuple[str, str, int], wp.array[wp.vec3]] = {}
_pml_distance_cache: dict[tuple[str, str], ml.MeshSet] = {}
_ml_query_cache: dict[tuple[str, str, int], mm.std_vector_Vector3_float] = {}

# MeshLib's own default distance limit. ``inf`` here is a hard crash, not an exception.
_FLT_MAX = 3.4028234663852886e38


def _query_points_np(bench_case: BenchCase, count: int = _N_QUERIES) -> np.ndarray:
    """Query points: subsampled vertices jittered by 10% of the bbox diagonal."""
    rng = np.random.default_rng(_QUERY_SEED)
    vertices = bench_case.vertices_np
    idx = rng.integers(0, vertices.shape[0], size=count)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return vertices[idx] + rng.normal(scale=0.1 * diagonal, size=(count, 3))


def _query_points_wp(bench_case: BenchCase, count: int = _N_QUERIES) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device), count)
    if key not in _query_cache:
        _query_cache[key] = wp.array(
            np.ascontiguousarray(_query_points_np(bench_case, count), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _query_cache[key]


def _query_points_ml(bench_case: BenchCase, count: int = _N_QUERIES) -> mm.std_vector_Vector3_float:
    """
    Build the query cloud as a MeshLib vector, cached per ``(mesh, count)``.

    Every batched MeshLib query here takes ``std_vector_Vector3_float``, and filling it is a Python
    loop over ``count`` ``Vector3f`` constructions -- 10 000 of them, which is comparable to the
    query itself. It is the query's *input*, so it is cached outside the timed callable exactly as
    the ``wp.array`` and ``o3d.core.Tensor`` clouds are.
    """
    key = (bench_case.mesh_name, "meshlib", count)
    if key not in _ml_query_cache:
        points_ml = mm.std_vector_Vector3_float()
        for point_np in _query_points_np(bench_case, count):
            points_ml.append(mm.Vector3f(*point_np.tolist()))
        _ml_query_cache[key] = points_ml
    return _ml_query_cache[key]


@pytest.mark.benchmark(group="winding_number")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista", "meshlib")
@pytest.mark.parametrize("n_queries", _N_QUERIES_SWEEP)
def test_winding_number(bench_case: BenchCase, n_queries: int) -> None:
    """
    Exact winding number: no BVH, every query sums over every face.

    The one genuinely ``O(queries x faces)`` function in the module, so both sizes are swept -- the
    mesh by the registry and the query count here. A 10x step in queries that is not a 10x step in
    time would mean the launch is not saturating the device.

    **pyvista answers the reduced question**: ``select_interior_points`` returns the inside/outside
    *bool* rather than the winding number itself, which is what ``ray.contains_points`` returns and
    what it is compared against (agreement 1.000 on 2 000 queries, ``tests/test_ray.py``). Read its
    row as the cost of the predicate, not of the number -- and note it uses a BVH where this group's
    two other rows deliberately do not.

    Its ``check_surface=False`` is **required, not a shortcut**: the filter validates first and
    raises ``RuntimeError: Surface is not closed`` on every scan mesh in the registry, so with the
    check on there is no mesh here the row can run at all. Disabling it times the ray casting
    itself, which is the comparable work; the *answer* on an open surface is undefined by VTK's own
    documentation, which is why the value comparison lives in ``tests/test_ray.py`` on a closed
    fixture and this row asserts only the shape.

    **meshlib answers a Barnes-Hut approximation of it**, and that is the whole reason its row is
    interesting here: ``FastWindingNumber(mesh).calcFromVector`` walks the mesh's AABB tree and
    replaces a distant subtree by a dipole, so unlike triwarp's and igl's rows it is *not*
    ``O(queries x faces)`` and should not follow the product. ``beta=20`` is the accuracy at which
    it agrees with the exact sum to 1e-05; its own default of 2 is 24x looser, so a row at the
    default would be timing a coarser answer. The tree build is inside the timed callable because
    ``FastWindingNumber`` is constructed per call, the same no-hoisting situation igl's AABB tree
    and triwarp's ``wp.Mesh`` are in.
    """
    skip_larger_than(bench_case, "happy_buddha", "O(queries x faces): lucy is untenable")
    if n_queries > _N_QUERIES:
        # 100k queries against dragon is 8.7e10 pair evaluations, and against happy_buddha 1.1e11.
        # The 10x query step is measurable on the medium meshes and the product is what it says.
        skip_larger_than(bench_case, "bunny", "the wide query sweep is only tenable up to bunny")
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "vtkSelectEnclosedPoints casts rays serially")
        mesh_pv = bench_case.mesh_pv
        cloud_pv = pv.PolyData(
            np.ascontiguousarray(_query_points_np(bench_case, n_queries), dtype=np.float64)
        )
        selected_pv = bench_case.run(
            lambda: cloud_pv.select_interior_points(mesh_pv, check_surface=False), rounds=3
        )
        assert np.asarray(selected_pv.point_data["selected_points"]).shape == (n_queries,)
        return
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        points_ml = _query_points_ml(bench_case, n_queries)

        def winding_ml() -> mm.std_vector_float:
            result_ml = mm.std_vector_float()
            mm.FastWindingNumber(mesh_ml).calcFromVector(
                result_ml,
                points_ml,
                20.0,
                mm.FaceId(),
                # pybind11 wraps the Python callable in the functor the stub names.
                lambda _progress: True,  # pyright: ignore[reportArgumentType]
            )
            return result_ml

        assert len(bench_case.run(winding_ml)) == n_queries
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points = _query_points_wp(bench_case, n_queries)
        result = bench_case.run(lambda: tw.proximity.winding_number(vertices, faces, points))
        assert result.shape == (n_queries,)
    else:  # igl exact generalized winding number
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        points = _query_points_np(bench_case, n_queries)
        result = bench_case.run(lambda: igl.winding_number(vertices, faces, points))
        assert result.shape == (n_queries,)


@pytest.mark.benchmark(group="winding_number_serial")
@pytest.mark.benchlibs("triwarp")
def test_winding_number_serial(bench_case: BenchCase) -> None:
    """Pinned ``tiled=False`` reference path: one thread per query loops over every face."""
    skip_larger_than(bench_case, "bunny", "serial winding takes minutes beyond bunny")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    result = bench_case.run(
        lambda: tw.proximity.winding_number(vertices, faces, points, tiled=False)
    )
    assert result.shape == (_N_QUERIES,)


def _distance_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Build the reference mesh at id 0 and the query points as a face-less mesh at id 1, once.

    The filter writes only mesh 1's vertex scalar attribute and leaves both geometries alone, so
    sharing is sound (verified: repeated calls return bit-identical scalars at the same cost).
    """
    key = (bench_case.mesh_name, "pml")
    if key not in _pml_distance_cache:
        meshset_pml = bench_case.new_meshset_pml()
        meshset_pml.add_mesh(
            ml.Mesh(vertex_matrix=np.ascontiguousarray(_query_points_np(bench_case)))
        )
        _pml_distance_cache[key] = meshset_pml
    return _pml_distance_cache[key]


@pytest.mark.benchmark(group="closest_point_on_mesh")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
def test_closest_point_on_mesh(bench_case: BenchCase) -> None:
    """
    The unsigned closest-point query, without the sign work the group below pays for.

    Read against ``signed_distance_on_mesh``: both walk a BVH to the nearest triangle, and the
    difference between the groups is what signing costs -- five perturbed parity rays or a winding
    traversal on triwarp's side, a projection-normal test on MeshLib's. That comparison is why this
    group exists separately rather than being folded into the signed one.

    meshlib's batched form is ``PointsToMeshProjector``; the per-query ``findProjection`` free
    function gives identical distances but would time a Python loop. The mesh is built and the AABB
    tree pre-warmed outside the timed callable, so the row prices the traversal -- the
    ``new_mesh_ml`` rule for a query row. Two ways to crash this call rather than get an exception:
    ``upDistLimitSq`` must be ``FLT_MAX``, not ``inf``, and **the projector keeps a raw pointer to
    the mesh it was given**, so a temporary leaves it reading freed memory -- Open3D's
    ``from_legacy`` hazard in a second library.

    pyvista's ``find_closest_cell`` is the third batched form, through a ``vtkStaticCellLocator``.
    It is the most *accurate* reference in the group -- against ``igl.point_mesh_squared_distance``
    it agrees to float64 rounding on both distance and point -- but it returns the closest **point**
    and the cell, never the distance, so its row is that much wider than what it is timed against
    and the subtraction stays outside the timed callable. Its locator is built lazily and cached, so
    it is pre-warmed rather than timed, like MeshLib's AABB tree.

    **The per-query cost is not flat in the mesh, and the knee is between the million- and
    ten-million-face meshes.** This group holds its query count fixed, so the effect shows up in a
    *caller* instead. With the ``wp.Mesh`` build separated out, the per-query cost falls from the
    small meshes (launch overhead at a small dim) to a minimum in the middle and then rises by an
    order of magnitude on the largest: a cache cliff, the BVH having stopped fitting. The build
    stays linear throughout. Worth knowing before reading any large-mesh row that ends in a
    closest-point query as an algorithm result.

    That cliff is why ``mesh_to_mesh_distance`` does not derive its bound by querying at every
    vertex: on the largest mesh that was nearly all of its call, to prune a traversal worth a
    fraction of a percent of it, and a subsample bounds the answer just as soundly
    (``proximity._BOUND_SAMPLE_TARGET`` carries the sweep). This group still prices the unsampled
    query, which is what keeps that cliff visible.
    """
    if bench_case.kind == "pyvista":
        # Superlinear across the scan sweep: VTK's locator is single-threaded, so this is capped
        # where the suite's other host rows are.
        skip_larger_than(bench_case, "bunny", "VTK's locator is single-threaded")
        mesh_pv = bench_case.mesh_pv
        queries_np = _query_points_np(bench_case)
        mesh_pv.find_closest_cell(queries_np[:1], return_closest_point=True)  # pre-warm
        _cells_pv, closest_pv = bench_case.run(
            lambda: mesh_pv.find_closest_cell(queries_np, return_closest_point=True)
        )
        assert np.asarray(closest_pv).shape == (_N_QUERIES, 3)
        return
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()  # must outlive the projector: it holds a raw pointer
        points_ml = _query_points_ml(bench_case)
        projector_ml = mm.PointsToMeshProjector()
        projector_ml.updateMeshData(mesh_ml)

        def project_ml() -> mm.std_vector_MeshProjectionResult:
            result_ml = mm.std_vector_MeshProjectionResult()
            projector_ml.findProjections(
                result_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _FLT_MAX, 0.0
            )
            return result_ml

        project_ml()  # pre-warm: the mesh's AABB tree is built lazily on first use
        assert len(bench_case.run(project_ml)) == _N_QUERIES
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    closest, distances, faces_hit = bench_case.run(
        lambda: tw.proximity.closest_point_on_mesh(vertices, faces, points)
    )
    assert closest.shape == (_N_QUERIES,)
    assert distances.shape == (_N_QUERIES,)
    assert faces_hit.shape == (_N_QUERIES,)


_crease_np_cache: dict[str, np.ndarray] = {}
_crease_wp_cache: dict[tuple[str, str], twt.Array2dInt32] = {}


def _crease_edges_np(bench_case: BenchCase) -> np.ndarray:
    """
    Build the mesh's sharp edges at 30 degrees on the host, cached once per mesh.

    Built with trimesh rather than with ``tw.seams.crease_edges`` so that **both** rows of the edge
    query get the identical edge set: ``vertices_wp`` is triwarp-only, so a triwarp-built set could
    not be handed to the pyvista row, and timing each side against its own crease set would fold a
    different input into a query comparison.
    """
    if bench_case.mesh_name not in _crease_np_cache:
        mesh_tm = tm.Trimesh(
            vertices=bench_case.vertices_np, faces=bench_case.faces_np, process=False
        )
        sharp_tm = np.degrees(mesh_tm.face_adjacency_angles) >= 30.0
        _crease_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            mesh_tm.face_adjacency_edges[sharp_tm], dtype=np.int32
        )
    return _crease_np_cache[bench_case.mesh_name]


def _crease_edges_wp(bench_case: BenchCase) -> twt.Array2dInt32:
    """Upload that edge set to the case's device, cached: an input, not part of the measure."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _crease_wp_cache:
        _crease_wp_cache[key] = twt.as_array2d(
            wp.array(_crease_edges_np(bench_case), dtype=wp.int32, device=bench_case.device),
            wp.int32,
        )
    return _crease_wp_cache[key]


@pytest.mark.benchmark(group="closest_point_on_edges")
@pytest.mark.benchlibs("triwarp", "pyvista")
def test_closest_point_on_edges(bench_case: BenchCase) -> None:
    """
    The **wireframe** closest-point query: the same queries against the mesh's crease edges.

    Read against ``closest_point_on_mesh``, which answers the same queries against the surface. The
    two are different structures over the same geometry -- a BVH of per-edge boxes against Warp's
    triangle mesh BVH -- and the edge set is far smaller than the face set, so the ratio prices
    triwarp's hand-written deepening traversal against Warp's built-in one on an easier input.

    The crease set is built outside the timed callable and on the host, so both rows get the
    identical edge set (see ``_crease_edges_np``); it is the *input* here, and triwarp's own
    ``crease_edges`` is timed in ``test_seams.py``. Its size is a mesh property rather than a knob,
    which is why this group takes no axis of its own.

    pyvista is the only batched reference. VTK's ``find_closest_cell`` on a one-line-cell-per-edge
    ``PolyData`` is exact for this query (probed: 0.0 distance error, every cell id matching a
    brute-force argmin), and its locator is pre-warmed rather than timed, as in the surface group.
    MeshLib's ``findProjectionOnMeshEdges`` is the other exact reference and is deliberately absent:
    it answers one query per call, so a row would time a Python loop rather than the traversal
    (``tests/test_proximity.py`` carries it as a ``benchmarked=False`` claim).

    Two readings. triwarp leads pyvista by well over an order of magnitude, and the slope across the
    scan sweep is the crease *count* rather than the face count. And the number to read it against
    is ``closest_point_on_mesh`` on the same queries and mesh: **this query is slower over an edge
    set some thirty times smaller**, so the hand-written deepening loop is losing to Warp's built-in
    mesh traversal rather than to the geometry. A query far from every crease pays several empty
    scans before the radius reaches anything, which is where that gap lives and what a future
    ``initial_radius`` estimate (the k-NN path already has one) would close.
    """
    edges_np = _crease_edges_np(bench_case)
    n_edges = int(edges_np.shape[0])
    if n_edges == 0:
        pytest.skip("no crease edges on this mesh at 30 degrees")

    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "VTK's locator is single-threaded")
        cells_np = np.hstack([np.full((n_edges, 1), 2), edges_np]).astype(np.int64).ravel()
        wireframe_pv = pv.PolyData(bench_case.vertices_np, lines=cells_np)
        queries_np = np.ascontiguousarray(_query_points_np(bench_case), dtype=np.float64)
        wireframe_pv.find_closest_cell(queries_np[:1], return_closest_point=True)  # pre-warm
        _cells_pv, closest_pv = bench_case.run(
            lambda: wireframe_pv.find_closest_cell(queries_np, return_closest_point=True)
        )
        assert np.asarray(closest_pv).shape == (_N_QUERIES, 3)
        return

    vertices = bench_case.vertices_wp
    edges = _crease_edges_wp(bench_case)
    queries = _query_points_wp(bench_case)
    closest, distances, edge_ids = bench_case.run(
        lambda: tw.proximity.closest_point_on_edges(vertices, edges, queries)
    )
    assert closest.shape == (_N_QUERIES,)
    assert distances.shape == (_N_QUERIES,)
    assert edge_ids.shape == (_N_QUERIES,)


@pytest.mark.benchmark(group="signed_distance_on_mesh")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pymeshlab", "pyvista", "meshlib")
@pytest.mark.parametrize("sign_mode", ["parity", "winding"])
def test_signed_distance_on_mesh(
    bench_case: BenchCase, sign_mode: Literal["parity", "winding"]
) -> None:
    """
    The two sign modes, on the same closest-point query.

    ``"winding"`` walks the BVH accumulating solid angle (Barnes-Hut, ``accuracy=2.0``) instead of
    casting 5 perturbed parity rays, and needs a ``wp.Mesh`` carrying the per-node solid-angle
    expansion -- so the ``wp.Mesh`` build inside the timed region differs between the two, which is
    intentional: it is part of what the mode costs. Both include that build because
    ``signed_distance_on_mesh`` constructs its own mesh, so there is no way for a caller to hoist
    it.

    **libigl is the only reference whose sign axis maps onto both of triwarp's modes**, which is why
    it appears twice where pymeshlab appears once: ``SIGNED_DISTANCE_TYPE_PSEUDONORMAL`` against
    ``"parity"`` and ``SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER`` against ``"winding"`` -- the
    second is the same Barnes-Hut family triwarp's mode is. Its AABB tree is built per call, as
    triwarp's ``wp.Mesh`` is, and it gets ``rounds=3`` like the pymeshlab row.

    **open3d's row is Embree**: ``RaycastingScene.compute_signed_distance`` signs by ray parity, so
    it pairs with ``"parity"`` only, and it shares triwarp's sign convention exactly. The scene
    build sits inside the timed callable for the same no-hoisting reason.

    **pyvista's row is an exact SDF and the closest match in the set**:
    ``compute_implicit_distance`` (``vtkImplicitPolyDataDistance``) shares triwarp's sign convention
    and agrees to 1.5e-07 with identical signs on 2 000 queries, which is why it is the parity
    oracle for this group. One row only: it has a single sign rule.

    Two things to read off the rows. The references are one to two orders of magnitude behind, and
    **the mode ratio disagrees between the two sides** -- igl's winding sign costs over twice its
    pseudonormal one where triwarp's two modes are within a third of each other, because the
    solid-angle walk rides the BVH traversal triwarp is already doing. Read igl's *medians* here,
    not its minima: its pseudonormal row spreads nearly twofold on the larger mesh.
    """
    if bench_case.kind == "pyvista":
        if sign_mode != "parity":
            pytest.skip("vtkImplicitPolyDataDistance has one sign rule, so it takes one row")
        skip_larger_than(
            bench_case, "bunny", "vtkImplicitPolyDataDistance is a serial per-query walk"
        )
        mesh_pv = bench_case.mesh_pv
        cloud_pv = pv.PolyData(np.ascontiguousarray(_query_points_np(bench_case), dtype=np.float64))
        distance_pv = bench_case.run(lambda: cloud_pv.compute_implicit_distance(mesh_pv), rounds=3)
        assert np.asarray(distance_pv.point_data["implicit_distance"]).shape == (_N_QUERIES,)
        return

    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl rebuilds a single-threaded AABB tree per call")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        points_np = _query_points_np(bench_case)
        sign_type = (
            igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            if sign_mode == "parity"
            else igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        )
        distance_igl, _, _, _ = bench_case.run(
            lambda: igl.signed_distance(points_np, vertices_np, faces_np, sign_type), rounds=3
        )
        assert distance_igl.shape == (_N_QUERIES,)
        return

    if bench_case.kind == "open3d":
        if sign_mode != "parity":
            pytest.skip("Embree signs by ray parity, so it pairs with triwarp's parity mode only")
        import open3d as o3d

        vertices_f32 = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        faces_u32 = np.ascontiguousarray(bench_case.faces_np, dtype=np.uint32)
        # The stub omits Tensor's dtype/device defaults.
        queries_t = o3d.core.Tensor(np.ascontiguousarray(_query_points_np(bench_case), np.float32))  # pyright: ignore[reportCallIssue]

        def signed_distance_o3d() -> o3d.core.Tensor:
            # The Embree BVH build goes inside, mirroring triwarp's own in-call wp.Mesh build.
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.core.Tensor(vertices_f32), o3d.core.Tensor(faces_u32))  # pyright: ignore[reportCallIssue]
            return scene.compute_signed_distance(queries_t)

        assert bench_case.run(signed_distance_o3d).shape == (_N_QUERIES,)
        return

    if bench_case.kind == "meshlib":
        if sign_mode != "parity":
            pytest.skip(
                "MeshLib's default signMode is the projection normal: one row, like MeshLab"
            )
        # ``findSignedDistances`` is the batched form and takes a ``VertCoords``, so the cloud is
        # uploaded as a PointCloud outside the timed callable, as every other row's cloud is. It
        # builds the reference mesh's AABB tree on first use, and that build is inside -- the same
        # no-hoisting position triwarp's per-call ``wp.Mesh`` is in.
        mesh_np = (bench_case.vertices_np, bench_case.faces_np)
        cloud_ml = mn.pointCloudFromPoints(
            cast("Buffer", np.ascontiguousarray(_query_points_np(bench_case), dtype=np.float64))
        )

        def signed_distance_ml() -> mm.VertScalars:
            return mm.findSignedDistances(mesh_ml_from_numpy(*mesh_np), cloud_ml.points)

        assert bench_case.run(signed_distance_ml).size() == _N_QUERIES
        return

    if bench_case.kind == "pymeshlab":
        if sign_mode != "parity":
            pytest.skip("MeshLab signs by the closest-point normal: a third mode, so one row only")
        skip_larger_than(bench_case, "bunny", "768 us per query on dragon: 85 s for one row")
        meshset_pml = _distance_meshset_pml(bench_case)
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_distance_from_another_mesh_per_vertex(
                measuremesh=1, refmesh=0, signeddist=True
            ),
            rounds=3,
        )
        assert meshset_pml.mesh(1).vertex_scalar_array().shape == (_N_QUERIES,)
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    distance = bench_case.run(
        lambda: tw.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode=sign_mode)
    )
    assert distance.shape == (_N_QUERIES,)


_LATTICE_ROWS = [26, 80, 240]
_lattice_cache: dict[int, tuple[np.ndarray, Delaunay, np.ndarray]] = {}


def _triangular_lattice_2d(rows: int) -> tuple[np.ndarray, Delaunay, np.ndarray]:
    """
    Build lattice points, their Delaunay triangulation and a query cloud overhanging it by 10%.

    Cached per size: the triangulation is the benchmark's *input*, so building it inside the timed
    region would price scipy's Delaunay rather than either library's point location.
    """
    if rows not in _lattice_cache:
        cols = rows + 4
        height = np.sqrt(3.0) / 2.0
        points_np = np.array(
            [(col + 0.5 * (row % 2), row * height) for row in range(rows) for col in range(cols)],
            dtype=np.float64,
        )
        triangulation_sp = Delaunay(points_np)
        extent_np = points_np.max(axis=0) - points_np.min(axis=0)
        rng = np.random.default_rng(4)
        queries_np = (
            points_np.min(axis=0) - 0.05 * extent_np + rng.random((_N_QUERIES, 2)) * 1.1 * extent_np
        )
        _lattice_cache[rows] = (points_np, triangulation_sp, queries_np)
    return _lattice_cache[rows]


@pytest.mark.benchmark(group="containing_faces_2d")
@pytest.mark.benchlibs("triwarp", "scipy", "pyvista")
@pytest.mark.parametrize("rows", _LATTICE_ROWS, ids=["lattice26", "lattice80", "lattice240"])
def test_containing_faces_2d(bench_lib: BenchLibrary, rows: int) -> None:
    """
    Point location in a planar triangulation: BVH candidate, then a barycentric sign test.

    The only group in this module with **no input mesh** -- a 2D triangulation is not one of the
    registry's surfaces -- so it takes ``bench_lib`` and builds its own: a triangular lattice
    whose Delaunay triangulation is equilateral throughout, swept over three sizes. The lattice
    is not arbitrary. Any Delaunay of a bounded point set has needle triangles on its hull, and
    a query within float32 noise of two needles' shared edge resolves to neither, so a random-
    point triangulation would make this row's own correctness assert flaky for a reason that has
    nothing to do with cost.

    **The comparison is deliberately unfavourable to triwarp.**
    ``scipy.spatial.Delaunay.find_simplex`` is timed on a triangulation built *outside* the
    timed region, while triwarp's row includes building its BVH on every call -- there is no
    prebuilt-index entry point on this side. Read the row as a floor on triwarp's margin, not as
    a like-for-like split; scipy is doing strictly less work per call.

    pyvista's ``find_containing_cell`` is in scipy's position rather than triwarp's: its locator is
    built lazily on the ``PolyData`` and pre-warmed here, so it too is timed with its index in hand.
    It locates in **3-D** -- the lattice and the queries get a zero ``z`` -- and it is the only
    reference in the suite that answers this question correctly: ``igl.in_element`` returns
    batch-size-dependent answers and aborts on a 200-point Delaunay (section 6), which is why the
    row exists at all.
    """
    points_np, triangulation_sp, queries_np = _triangular_lattice_2d(rows)
    if bench_lib.kind == "pyvista":
        mesh_pv = pv.PolyData.from_regular_faces(
            np.column_stack([points_np, np.zeros(points_np.shape[0])]),
            np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32),
        )
        queries_3d_np = np.column_stack([queries_np, np.zeros(queries_np.shape[0])])
        mesh_pv.find_containing_cell(queries_3d_np[:1])  # pre-warm the cell locator
        located_pv = np.asarray(bench_lib.run(lambda: mesh_pv.find_containing_cell(queries_3d_np)))
        assert located_pv.shape == (_N_QUERIES,)
        assert (located_pv >= 0).any()
        return
    if bench_lib.kind == "scipy":
        located_sp = bench_lib.run(lambda: triangulation_sp.find_simplex(queries_np))
        assert located_sp.shape == (_N_QUERIES,)
        assert (located_sp >= 0).any()
        return

    device = bench_lib.device
    vertices_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32).ravel(),
        dtype=wp.int32,
        device=device,
    )
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    located_wp = bench_lib.run(
        lambda: tw.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp)
    )
    # Exact on this lattice, which is why the fixture is a lattice; see the docstring.
    assert np.array_equal(located_wp.numpy(), triangulation_sp.find_simplex(queries_np))


_CLEARANCE_OFFSETS = [1.2, 2.0]
_clearance_cache: dict[tuple[str, str, float], wp.array[wp.vec3]] = {}


def _separated_vertices_wp(bench_case: BenchCase, offset: float) -> wp.array[wp.vec3]:
    """Translate a self-copy far enough away to be disjoint: the second mesh of the pair."""
    key = (bench_case.mesh_name, str(bench_case.device), offset)
    if key not in _clearance_cache:
        vertices_np = bench_case.vertices_np
        extent = vertices_np.max(axis=0) - vertices_np.min(axis=0)
        shift_np = np.array([offset * float(extent[0]), 0.0, 0.0])
        _clearance_cache[key] = wp.array(
            np.ascontiguousarray(vertices_np + shift_np, dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _clearance_cache[key]


@pytest.mark.benchmark(group="mesh_to_mesh_distance")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("offset", _CLEARANCE_OFFSETS, ids=["near", "far"])
def test_mesh_to_mesh_distance(bench_case: BenchCase, offset: float) -> None:
    """
    Clearance between a mesh and a translated copy of itself, at two separations.

    The **separation is the axis**, and it runs the opposite way to the obvious prediction. The
    bound derived from the vertex query grows with the gap, yet a distant pair is the **cheap** one
    on every mesh: once the running minimum prunes by box-to-box gap, a large true clearance rejects
    almost every candidate on that lower bound immediately, while a tight one leaves many pairs
    genuinely close and each has to be measured.

    meshlib's ``findDistance`` is a BVH-versus-BVH descent with a running bound -- pruning with
    information this two-phase form only has once the second phase starts -- and it is
    multi-threaded. So this is the row where a sequential-pruning algorithm is expected to compete
    well against a wavefront. Both rows build their own acceleration structure inside the callable.

    Two things to read off the rows. They are **nearly flat in the face count up to about a
    million**, so over that range the cost is the candidate count and not the mesh;
    ``bunny_decimated`` is the slowest per face because its duplicated faces manufacture
    near-zero-gap candidates no bound can prune. And the two prunes inside the kernel are what make
    the numbers reportable at all: without them the same rows are an order of magnitude dearer.

    **The largest mesh is not flat and the reason is a different function.** Per stage it is almost
    entirely the *bound*; inside that bound the ``wp.Mesh`` build is linear and the **queries** are
    the cliff, once the BVH stops fitting in cache. So the superlinearity belongs to
    ``closest_point_on_mesh``, is a memory-hierarchy effect rather than an algorithm defect, and is
    *not* what a BVH-versus-BVH rewrite of this function would fix.

    **The stage split inverts with size, which is where any future work has to be aimed.** The query
    dominates exactly where the gap is -- the small-mesh rows, the only ones meshlib is not capped
    out of -- and on the large ones the bound is most of the call. So a cheaper bound (a subsampled
    vertex query is still sound: the minimum over any *subset* of A's vertices is still an upper
    bound) only moves rows that contribute no gap, and a faster query only moves the small ones.

    **The query half is a load-balancing problem rather than the pruning problem it reads as.**
    Counted on a scan mesh: a tiny fraction of the broad phase's candidates survive the box prune,
    so the leaf test is not the cost -- but the overwhelming majority of query faces return no
    candidate at all, a fraction of a percent carry half the traversal, and the busiest face walks
    thousands alone. The walk runs as a capped thread pass plus a warp per straggler
    (``proximity._QUERY_CANDIDATE_CAP``), worth several-fold end to end with the distance and
    ``face_a`` identical. Two levers declined on the way: ``block_dim`` (256, the default, wins at
    every value from 32) and tightening the query margin -- the vertex bound *is* the answer to full
    precision on all four rows, so the query only confirms a distance the bound already found.

    A BVH-pair wavefront remains the one change that would also delete the bound phase, and it is
    **larger than it looks**: Warp exposes ``bvh_query_aabb`` / ``bvh_query_ray`` and no
    node-by-node traversal, so a pair descent means building our own hierarchy rather than reusing
    ``wp.Bvh``.
    """
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny", "findDistance is a serial descent per pair")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        extent = vertices_np.max(axis=0) - vertices_np.min(axis=0)
        shifted_np = vertices_np + np.array([offset * float(extent[0]), 0.0, 0.0])
        first_ml = mesh_ml_from_numpy(vertices_np, faces_np)
        second_ml = mesh_ml_from_numpy(shifted_np, faces_np)

        def distance_ml() -> float:
            return float(
                mm.findDistance(
                    mm.MeshPart(first_ml),
                    mm.MeshPart(second_ml),
                    # None is the identity transform; the stub omits that it is nullable.
                    None,  # pyright: ignore[reportArgumentType]
                    float(np.finfo(np.float32).max),
                ).distSq
            )

        assert bench_case.run(distance_ml, rounds=3) > 0.0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    shifted = _separated_vertices_wp(bench_case, offset)
    distance, face_a, face_b = bench_case.run(
        lambda: tw.proximity.mesh_to_mesh_distance(vertices, faces, shifted, faces), rounds=3
    )
    assert distance > 0.0
    assert face_a >= 0
    assert face_b >= 0
