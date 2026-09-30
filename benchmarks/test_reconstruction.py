"""
Benchmarks for ``triwarp.reconstruction``.

The point cloud is a registry mesh's own vertices with area-weighted vertex normals — deterministic
(no sampling RNG), consistently oriented, and it scales with the mesh. Both are precomputed and
cached: they are the *input*, not part of the operation being timed.

**open3d** is the reference for ball pivoting
(``create_from_point_cloud_ball_pivoting``, Bernardini's BPA at the identical radius); it gets the
same points and computes its own area-weighted normals, the same quantity
[`vertex_normals`][triwarp.vertices.vertex_normals] produces.

**pymeshlab** is the third BPA implementation, and worth a row because
``generate_surface_reconstruction_ball_pivoting`` is VCGlib's original BPA — so its row and open3d's
price two wrappers over one lineage and only triwarp's is a different program. It gets the
identical absolute radius via ``PureValue`` (the ``0%`` default autoguesses one, which would
compare two different parameters) with ``clustering=0`` to disable a merge step triwarp does not do.
The filter
pushes its output as a new layer without touching the cloud, so the cloud MeshSet is cached;
``set_current_mesh(0)`` inside the callable restores the layer the previous round's push moved away
from.

**Screened Poisson is timed for triwarp alone.** Both CPU references wrap Kazhdan's own solver and
between them cost the **majority of the whole suite's wall clock** to re-measure a reference triwarp
beats by more than an order of magnitude, with the largest cell running well over an hour without
completing a round. The rows are **removed**, not capped, and the agreement they were the parity
evidence for is checked in ``tests/test_reconstruction.py`` at a size a correctness test can afford.
What remains is a triwarp-only regression row over ``method`` x ``depth``.

**The pymeshlab cloud is not quite the same cloud**, and that is forced rather than chosen: it drops
every point whose area-weighted normal is exactly zero — the unreferenced vertices every scan mesh
carries — because the screened-Poisson filter rejects a cloud carrying any null normal outright. The
cleaning is kept for the BPA row so the reference's input stays the documented one; the alternative,
``preclean=True``, moves the same cleaning *inside* the timed filter, which is worse.

``generate_surface_reconstruction_vcg`` was tried as a *fourth* algorithm and **rejected**: it
returns **zero faces** on this input at every voxel size probed. A row that silently measures a
no-op is worse than no row.

``triangulate_point_cloud`` has **meshlib** and nothing else. The nearest open3d and pymeshlab
analogues are alpha shapes — a different algorithm, so timing them would compare algorithm choices
rather than implementations — where ``triangulatePointCloud`` is the same local fan optimization at
the same ``numNeighbours`` and returns the identical face set on a clean uniform cloud.

``ball_pivoting`` is built around a **persistent front** — an edge hash table plus a compacted
boundary-edge list, mutated in place and never re-derived from the triangle soup. Three things pay
for it: retiring provably-dead front edges (under a per-wave rebuild most pivots are re-searches of
edges already known to be impossible), caching each front edge's best candidate so the majority that
lose a vertex claim re-validate in O(1), and dropping the per-wave ``edges_unique`` + sort + three
readbacks a rebuild needs. It also reconstructs a **much better surface** — far fewer faces per
referenced vertex, an order of magnitude fewer boundary edges — because a per-wave rebuild lets
colliding fronts triangulate a neighbourhood in overlapping layers, which cleanup then tears back
into open patches.

What is left is ``pivot_front_edges``, nearly all of the kernel time, and it is *occupancy*-bound:
a wave has a few hundred live front edges doing serial dependent work on a device with hundreds of
thousands of thread slots, which is why the smaller mesh is behind.

A warp per edge is the answer, and the *index* is what constrains it: Warp's hash grid exposes only
a sequential per-thread iterator with no per-cell entry point, so its walk — most of a query's cost
— cannot be split across lanes. Warp's **BVH** can be, through ``tile_bvh_query_aabb``, and that is
several-fold over the hash grid at the front sizes a wave has. (A *serial* BVH walk is slower than
the hash grid, so the win is the cooperation and not the tree; a hand-rolled hashed cell grid would
be a further several-fold this kernel's own ceiling cannot spend.) So ``pivot_front_edges`` runs one
block per front edge, one warp per block, walking the BVH cooperatively, with the empty-ball test
left as a per-lane serial hash-grid query — at that point each lane tests a different ball, so there
is nothing to cooperate on and the speedup comes from running 32-way concurrently. Worth nearly 5x
on both meshes, and it takes the group from several times behind pymeshlab to several times ahead.

Sizing notes:

- ``ball_pivoting`` uses ``1.5 * mean_edge``. The ``4 * n + 16`` triangle budget doubles on demand,
  so a larger multiple completes; it is not benchmarked, because a ball that wide searches a much
  larger neighbourhood per pivot and measures a different thing.
- ``screened_poisson`` in ``dense`` mode is dominated by the ``2 ** depth`` cubed node grid rather
  than the point count, so it costs the same on ``bunny_decimated`` and ``bunny``; ``adaptive`` does
  scale with the cloud. Both run at the default ``depth=8``.

Everything is capped at ``bunny``, and the cap is load-bearing: point-cloud triangulation and ball
pivoting are superlinear in the cloud size. The CPU screened-Poisson references made it
non-negotiable — one cell ran over an hour without completing a round — and those rows are now
removed outright rather than capped. The cap stays for the algorithms still timed against a
reference. This is a deliberate coverage gap: the optimizations these benchmarks gate are host-sync
and launch-overhead fixes, which show up at these sizes.
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
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, BenchLibrary, skip_larger_than

if TYPE_CHECKING:
    import open3d as o3d
    from typing_extensions import Buffer

# MeshLib's default neighbour count for local fan triangulation.
_NUM_NEIGHBOURS = 18

# Ball radius as a multiple of the mean edge length (a proxy for point spacing).
_BPA_RADIUS_FRACTION = 1.5

# Octree depths of ``triwarp.reconstruction.screened_poisson``. This is the module's dominant knob
# by a wide margin: ``dense`` mode is a ``2^depth`` cubed node grid, so each step is ~8x the nodes
# and the point count is almost secondary.
_POISSON_DEPTHS = [7, 9]

# A depth-9 dense solve and a ball-pivoting front are both seconds a call.
_HEAVY_ROUNDS = 3
_POISSON_DEPTH = 8

_normals_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_cloud_o3d_cache: dict[str, o3d.geometry.PointCloud] = {}
_cloud_pml_cache: dict[str, ml.MeshSet] = {}
_cloud_ml_cache: dict[str, mm.PointCloud] = {}


def _normals(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Area-weighted vertex normals for the point cloud, cached per ``(mesh, device)``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _normals_cache[key] = tw.vertices.vertex_normals(vertices, faces)
    return _normals_cache[key]


def _cloud_o3d(bench_case: BenchCase) -> o3d.geometry.PointCloud:
    """
    Oriented open3d point cloud: the mesh vertices with area-weighted vertex normals.

    The same input triwarp gets, with the normals computed by open3d's own
    ``compute_vertex_normals`` (also area-weighted). Cached — it is the input, not the operation.
    """
    if bench_case.mesh_name not in _cloud_o3d_cache:
        import open3d as o3d

        mesh_o3d = bench_case.mesh_o3d
        mesh_o3d.compute_vertex_normals()
        cloud = o3d.geometry.PointCloud(mesh_o3d.vertices)
        cloud.normals = mesh_o3d.vertex_normals
        _cloud_o3d_cache[bench_case.mesh_name] = cloud
    return _cloud_o3d_cache[bench_case.mesh_name]


def _cloud_ml(bench_case: BenchCase) -> mm.PointCloud:
    """
    Build the oriented cloud as a ``meshlib.PointCloud``, cached per mesh.

    ``_normals`` goes through ``bench_case.vertices_wp``, which needs a Warp device a CPU-bound
    case does not have -- the same reason the pymeshlab cloud takes its normals from trimesh, whose
    ``vertex_normals`` are area-weighted like triwarp's. Unlike that cloud this one keeps *every*
    point, null normals included: ``triangulatePointCloud`` accepts them where screened Poisson
    rejects the cloud outright, so meshlib reconstructs from exactly the points triwarp does.
    """
    if bench_case.mesh_name not in _cloud_ml_cache:
        from meshlib import mrmeshnumpy as mn

        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        # MeshLib's stubs type these as ``Buffer``, which numpy's stubs implement from 3.12, and its
        # ``normals`` setter as ``VertCoords``, which pybind11 converts the vector into.
        cloud_ml = mn.pointCloudFromPoints(
            cast("Buffer", np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        )
        cloud_ml.normals = mn.fromNumpyArray(  # pyright: ignore[reportAttributeAccessIssue]
            cast(
                "Buffer", np.ascontiguousarray(np.asarray(mesh_tm.vertex_normals), dtype=np.float64)
            )
        )
        _cloud_ml_cache[bench_case.mesh_name] = cloud_ml
    return _cloud_ml_cache[bench_case.mesh_name]


def _cloud_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Build the oriented point cloud as a face-less pymeshlab mesh, cached per mesh.

    Both reconstruction filters *push* their output onto the MeshSet rather than editing mesh 0, so
    the cloud itself is never touched -- but ``current_mesh()`` moves to the new layer, so the
    filter is always called on the cached set with mesh 0 still current from the previous round's
    push. ``set_current_mesh(0)`` restores that, and is part of the timed callable because it is the
    cost of driving the MeshSet API rather than of the reconstruction.
    """
    if bench_case.mesh_name not in _cloud_pml_cache:
        # ``_normals`` goes through ``bench_case.vertices_wp``, which needs a Warp device -- and a
        # pymeshlab case has none. trimesh's ``vertex_normals`` is area-weighted too, which is the
        # same argument the open3d branch makes for using open3d's own.
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        normals_np = np.asarray(mesh_tm.vertex_normals, dtype=np.float64)
        # Screened Poisson *rejects* a cloud carrying any null normal outright ("Filter requires
        # correct per vertex normals"), and every scan mesh has unreferenced vertices, whose
        # area-weighted normal is exactly zero -- 47 of bunny_decimated's 8 171 and 1 113 of bunny's
        # 35 947. Dropping them here rather than passing ``preclean=True`` keeps the cleaning out of
        # the timed region; the cost is that the reference reconstructs from 0.6% / 3.1% fewer
        # points than triwarp and open3d do, which is recorded in the module docstring.
        keep_np = np.linalg.norm(normals_np, axis=1) > 0.0
        meshset_pml = ml.MeshSet()
        meshset_pml.add_mesh(
            ml.Mesh(
                vertex_matrix=np.ascontiguousarray(
                    bench_case.vertices_np[keep_np], dtype=np.float64
                ),
                v_normals_matrix=np.ascontiguousarray(normals_np[keep_np], dtype=np.float64),
            )
        )
        _cloud_pml_cache[bench_case.mesh_name] = meshset_pml
    meshset_pml = _cloud_pml_cache[bench_case.mesh_name]
    meshset_pml.set_current_mesh(0)
    return meshset_pml


# Point counts for the planar Delaunay sweep. It is a *sequential* incremental insertion followed
# by a parallel flip loop, so the axis worth showing is the point count rather than any mesh size.
_DELAUNAY_POINTS = [2_000, 20_000]

_delaunay_np_cache: dict[int, np.ndarray] = {}


def _delaunay_points_np(n_points: int) -> np.ndarray:
    """``(n, 2)`` uniform planar cloud, cached per size -- the *input*, not part of the work."""
    if n_points not in _delaunay_np_cache:
        _delaunay_np_cache[n_points] = np.ascontiguousarray(
            np.random.default_rng(0).random((n_points, 2)), dtype=np.float64
        )
    return _delaunay_np_cache[n_points]


@pytest.mark.benchmark(group="delaunay_triangulation")
@pytest.mark.benchlibs("triwarp", "scipy", "pyvista")
@pytest.mark.parametrize("n_points", _DELAUNAY_POINTS)
def test_delaunay_triangulation(bench_lib: BenchLibrary, n_points: int) -> None:
    """
    Planar Delaunay triangulation, on the point-count axis -- the module's one mesh-free group.

    It takes ``bench_lib`` rather than ``bench_case`` for the reason ``test_creation.py``'s groups
    do: there is no input mesh, so the work is sized by a plain ``parametrize`` (the same 2-D cloud
    both references see, built once per size outside the timed region).

    !!! note "This row was the suite's largest loss, and both halves of it are now compiled"
        **The seed.** The overwhelming majority of the row was a host-side pure-Python sweep doing
        hundreds of thousands of ``_orient2d`` calls while the device did almost nothing. Porting
        that sweep to a single-thread Warp **CPU** kernel took an order of magnitude off it. The CPU
        device is not a concession: the identical sweep is two orders of magnitude cheaper on one
        CPU thread than on one **CUDA** thread, so a single GPU thread is the wrong tool.

        **The flip loop**, which then dominated the small row. Its cost was not the launch count
        the first reading blamed: most of the host time per pass went to
        [`face_adjacency`][triwarp.adjacency.face_adjacency] and to a *second* radix sort of the
        same edge keys ``face_adjacency`` had already sorted internally, against a fraction of that
        in device work for the whole loop. Since a flip leaves the vertex, face and interior-edge
        counts alone, that whole working set is invariant and is built once into fixed buffers by
        one sort and five launches -- worth about 2x at the large end and more at the small one.
        See [`delaunay_triangulation`][triwarp.reconstruction.delaunay_triangulation].

    The three implementations answer the same question by different means: triwarp seeds a
    sequential lexicographic incremental triangulation and then drives its **parallel** edge-flip
    loop to the empty-circumcircle fixed point, scipy calls Qhull, and VTK runs
    ``vtkDelaunay2D``'s serial insertion. The answers agree on the interior and differ only in hull
    slivers -- pyvista's 374 triangles are a strict *subset* of triwarp's 384 on a 200-point cloud
    (``tests/test_reconstruction.py``), so read the counts as well as the clock.
    """
    points_np = _delaunay_points_np(n_points)
    if bench_lib.kind == "pyvista":
        cloud_pv = pv.PolyData(np.column_stack([points_np, np.zeros(n_points)]).astype(np.float64))
        triangulated_pv = bench_lib.run(cloud_pv.delaunay_2d)
        assert triangulated_pv.n_faces > 0
        return
    if bench_lib.kind == "scipy":
        from scipy.spatial import Delaunay

        triangulated_sp = bench_lib.run(lambda: Delaunay(points_np))
        assert triangulated_sp.simplices.shape[1] == 3
        return
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=bench_lib.device
    )
    faces = bench_lib.run(lambda: tw.reconstruction.delaunay_triangulation(points_wp))
    assert faces.size % 3 == 0
    assert faces.size > 0


@pytest.mark.benchmark(group="triangulate_point_cloud")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_triangulate_point_cloud(bench_case: BenchCase) -> None:
    """
    Local fan triangulation of an oriented cloud, against the only library that has the same one.

    ``numNeighbours`` is triwarp's ``num_neighbours``, passed the same value on both sides -- it is
    the k-NN size the fan is optimized over and so the parameter that sets the work. meshlib builds
    its own k-NN structure inside the call, as triwarp does, so nothing is hoisted out on either
    side; the cloud and its normals are the input and are cached.
    """
    skip_larger_than(bench_case, "bunny", "local triangulation above bunny dominates the suite")
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)
        parameters_ml = mm.TriangulationParameters()
        parameters_ml.numNeighbours = _NUM_NEIGHBOURS
        mesh_ml = bench_case.run(lambda: mm.triangulatePointCloud(cloud_ml, parameters_ml))
        assert mesh_ml.topology.numValidFaces() > 0
        return
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.triangulate_point_cloud(
            points, normals, num_neighbours=_NUM_NEIGHBOURS
        )
    )
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="ball_pivoting")
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
def test_ball_pivoting(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny", "ball pivoting above bunny dominates the suite")
    radius = _BPA_RADIUS_FRACTION * bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        # VCGlib's original BPA, given the identical absolute radius via ``PureValue`` (its default
        # ``0%`` autoguesses one, which would compare two different algorithms' parameters).
        #
        # ``clustering`` stays at MeshLab's default 20%: at ``0`` the filter reconstructs
        # **nothing** on the same cloud and the same radius, and takes *longer* doing it, so this
        # a row at ``0`` therefore times a failure and reads slower for it. The clustering fraction
        # is a seed-triangle spacing floor, not an optional post-pass. At the default the two agree:
        # triwarp's 1 280, asserted in
        # tests/test_reconstruction.py::test_ball_pivoting_matches_pymeshlab.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(
            lambda: cloud_pml.generate_surface_reconstruction_ball_pivoting(
                ballradius=ml.PureValue(radius), clustering=20.0
            ),
            rounds=_HEAVY_ROUNDS,
        )
        return
    if bench_case.kind == "triwarp":
        points, normals = bench_case.vertices_wp, _normals(bench_case)
        _vertices, faces = bench_case.run(
            lambda: tw.reconstruction.ball_pivoting(points, normals, radius=radius),
            rounds=_HEAVY_ROUNDS,
        )
        assert int(faces.shape[0]) > 0
    else:  # open3d's BPA takes a radius list; give it the single identical radius
        import open3d as o3d

        cloud = _cloud_o3d(bench_case)
        radii = o3d.utility.DoubleVector([radius])
        mesh_bpa = bench_case.run(
            lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
        )
        assert len(mesh_bpa.triangles) > 0


@pytest.mark.noparity(
    "meshlib",
    reason="D2 a different algorithm with a measured disagreement: pointsToDistanceVolume is a "
    "Gaussian-weighted distance field over a fixed lattice, marched by gridToMesh -- not Kazhdan's "
    "screened Poisson, which solves for an indicator function. On an icosphere(4) cloud at a 2% "
    "voxel it returns *two* components: an outer one within 0.095 of the cloud (1.4 voxels, so the "
    "surface itself is recovered) and an inner shell of 2 652 faces at radius ~0.55 on a unit "
    "sphere, up to 0.518 from the cloud, which the Poisson solvers do not produce. Its "
    "parameterization is a lattice (origin, dimensions, voxelSize, sigma) rather than an octree "
    "depth, so triwarp's depth cannot be mapped onto it either; the row is a cost comparison "
    "against a *fast* implicit reconstructor at a matched cell size. open3d and pymeshlab wrap the "
    "same Kazhdan solver triwarp implements and carry the correctness comparison in "
    "tests/test_reconstruction.py, at a size a correctness test can afford.",
)
@pytest.mark.benchmark(group="screened_poisson")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("depth", _POISSON_DEPTHS)
@pytest.mark.parametrize("method", ["dense", "adaptive"])
def test_screened_poisson(
    bench_case: BenchCase, method: Literal["dense", "adaptive"], depth: int
) -> None:
    # **The CPU Poisson rows were removed, deliberately.** open3d and pymeshlab both wrap Kazhdan's
    # CPU solver, and between them they were the majority of the whole benchmark suite's wall clock
    # while measuring a reference triwarp had already beaten by more than an order of magnitude:
    # seconds per call at the default depth, and well over an hour without completing a round on the
    # largest cloud. The comparison itself is not lost: it lives in
    # ``tests/test_reconstruction.py``, which still checks both references for agreement at a size a
    # correctness test can afford. What is left here is triwarp against meshlib, which is a
    # *different* implicit reconstructor (see the exemption above) and, unlike the two Kazhdan
    # wrappers, cheap enough to keep.
    skip_larger_than(
        bench_case, "bunny", "screened Poisson above bunny dominates the suite (93 min at dragon)"
    )
    if bench_case.kind == "meshlib":
        if method == "adaptive":
            pytest.skip("its lattice is uniform: there is no adaptive variant to match")
        # The lattice is matched to triwarp's octree depth by cell *count* along the longest axis,
        # ``2 ** depth``, which is the only parameter the two share. ``sigma`` is set from the
        # cloud's own spacing rather than left at its default of 1.0, an absolute length that would
        # mean something different on every mesh. The volume and the marching are timed together:
        # neither half alone is a reconstruction.
        cloud_ml = _cloud_ml(bench_case)
        vertices_np = bench_case.vertices_np
        lower_np = vertices_np.min(axis=0)
        extent_np = vertices_np.max(axis=0) - lower_np
        voxel = float(extent_np.max()) / (2**depth)
        origin_np = lower_np - 3.0 * voxel
        dimensions = np.ceil((extent_np + 6.0 * voxel) / voxel).astype(int) + 1

        volume_params_ml = mm.PointsToDistanceVolumeParams()
        volume_params_ml.voxelSize = mm.Vector3f(voxel, voxel, voxel)
        volume_params_ml.sigma = 2.0 * bench_case.mean_edge
        volume_params_ml.minWeight = 0.5
        volume_params_ml.origin = mm.Vector3f(*origin_np.tolist())
        volume_params_ml.dimensions = mm.Vector3i(*(int(n) for n in dimensions))
        mesh_settings_ml = mm.GridToMeshSettings()
        mesh_settings_ml.voxelSize = mm.Vector3f(voxel, voxel, voxel)

        def reconstruct_ml() -> int:
            volume_ml = mm.pointsToDistanceVolume(cloud_ml, volume_params_ml)
            grid_ml = mm.simpleVolumeToDenseGrid(volume_ml)
            return mm.gridToMesh(grid_ml, mesh_settings_ml).topology.numValidFaces()

        assert bench_case.run(reconstruct_ml, rounds=_HEAVY_ROUNDS) > 0
        return
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.screened_poisson(points, normals, depth=depth, method=method),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(faces.shape[0]) > 0


# Cell sizes for the resampling group, as a fraction of the bbox diagonal. 2% is MeshLab's own
# default; 1% is eight times the field evaluation, which is what makes the pair worth having.
_RESAMPLE_CELL_FRACTIONS = [0.02, 0.01]


@pytest.mark.benchmark(group="resample_uniform")
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab", "meshlib")
@pytest.mark.parametrize("cell_fraction", _RESAMPLE_CELL_FRACTIONS)
def test_resample_uniform(bench_case: BenchCase, cell_fraction: float) -> None:
    """
    Sample the signed distance field on a grid and march it: the one **cubic** group in the suite.

    Both rows get the identical absolute cell size, derived from the mesh's own bounding-box
    diagonal on the host so neither side computes its own. The pair is a slope check: halving the
    cell is 8x the lattice on both sides, and triwarp rises only slightly for it, so the field
    evaluation is *not* what dominates and the fixed marching and cleanup passes are. MeshLab is
    sublinear over the same step too. Read a regression here as the slope steepening rather than the
    absolute number moving.

    MeshLab's ``offset`` parameter is passed as ``PureValue(0.0)``: as a ``PercentageValue`` it runs
    from full erosion at 0% to full dilation at 100%, so its own 50% default is the *zero* offset
    and ``PercentageValue(0)`` would erode the mesh away.

    **libigl's ``offset_surface`` is the same operation at ``isolevel=0``** -- sample the signed
    distance field on a grid, march it -- and its ``signed_distance_type`` is given
    ``PSEUDONORMAL``, the mode ``tests/test_proximity.py`` establishes agrees with triwarp's default
    sign to 8e-8. Its resolution parameter ``s`` is a *cell count along the longest axis*, not a
    length, so it receives ``round(longest_extent / voxel_size)`` -- the named transform that puts
    all three rows on the identical lattice. In-harness it is sublinear in the lattice like the
    other two and needs no cap of its own beyond the group's.
    """
    diagonal = float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    voxel_size = cell_fraction * diagonal
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the field is evaluated on one core")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        extent = float((vertices_np.max(axis=0) - vertices_np.min(axis=0)).max())
        resolution = max(2, round(extent / voxel_size))
        vertices_igl, faces_igl = bench_case.run(
            lambda: igl.offset_surface(
                vertices_np, faces_np, 0.0, resolution, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            )[:2],
            rounds=_HEAVY_ROUNDS,
        )
        assert faces_igl.shape[0] > 0
        assert vertices_igl.shape[1] == 3
        return
    if bench_case.kind == "meshlib":
        # ``rebuildMesh`` returns a *new* mesh but takes a ``MeshPart`` over the input and builds
        # its own voxel grid inside, so the input mesh is built once outside the timed callable.
        # Its ``voxelSize`` is the same absolute length the other three rows get. It is doing
        # strictly more than the other three at its defaults -- ``preSubdivide`` and ``decimate``
        # are both on -- which is a real difference in what the operation *is*, so read the row as
        # "resample and clean up" rather than as the marching alone.
        skip_larger_than(bench_case, "bunny", "the lattice is marched on the CPU")
        mesh_ml = bench_case.new_mesh_ml()
        settings_ml = mm.RebuildMeshSettings()
        settings_ml.voxelSize = voxel_size

        def resample_ml() -> int:
            return mm.rebuildMesh(mm.MeshPart(mesh_ml), settings_ml).topology.numValidFaces()

        assert bench_case.run(resample_ml, rounds=_HEAVY_ROUNDS) > 0
        return
    if bench_case.kind == "pymeshlab":
        # ``generate_*`` pushes a new mesh onto the set, so the MeshSet is rebuilt per round.
        skip_larger_than(bench_case, "bunny", "MeshLab marches the whole lattice on one core")
        new_meshset_pml = bench_case.new_meshset_pml

        def resample_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.generate_resampled_uniform_mesh(
                cellsize=ml.PureValue(voxel_size), offset=ml.PureValue(0.0)
            )
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(resample_pml, rounds=_HEAVY_ROUNDS) > 0
        return
    skip_larger_than(bench_case, "bunny", "a 1% lattice over dragon is 8 GB of field")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _out_vertices, out_faces = bench_case.run(
        lambda: tw.reconstruction.resample_uniform(vertices, faces, voxel_size=voxel_size),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(out_faces.shape[0]) > 0
