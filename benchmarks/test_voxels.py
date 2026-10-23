"""
Benchmarks for ``ordito.voxels``.

Every group here is really a race between **one hash table and one NanoVDB grid**. open3d keeps a
``std::unordered_map<Eigen::Vector3i>`` and walks it on one core; ordito keeps a NanoVDB index grid
whose build deduplicates as it goes and whose membership query is an ``O(1)`` probe from a kernel.
So the interesting reading is not any single row but the *slope*: how each side responds to the
cubic axis when the cell shrinks.

References
----------
**open3d** is the direct counterpart for four of the groups and computes the same answer in each:
``create_from_triangle_mesh_within_bounds`` runs Moller's tri-box test per candidate cell (in
``float64``, over a slightly wider candidate window that cannot change the accept set),
``create_from_point_cloud`` bins with the same ``min_bound - voxel_size / 2`` anchor,
``voxel_down_sample`` averages the same cells, and ``check_if_included`` answers the same membership
question. All four are asserted element-wise in ``tests/test_voxels.py``.

**trimesh** covers the morphology and the box meshing. ``voxel.morphology.binary_dilation`` and
``fill_cavities`` are ``scipy.ndimage`` on a dense array, so those two rows are really "sparse grid
against dense ndimage" and their axis is the one that separates them: the dense side pays for the
whole bounding box, the sparse side only for the occupied cells. ``voxel.ops.multibox`` builds
``12 n`` triangles on the host with no corner sharing.

**meshlib** adds the second dense implementation of the dilation, and the only multi-threaded one:
``expandVoxelsMask`` walks the same bounding box on every core, over a ``VoxelBitSet`` rather than a
NumPy array. It shares trimesh's input verbatim, so the ``dilate`` group reads as one sparse GPU
grid against two dense CPU passes over the identical occupancy. Its bitset **mutates**, so the row
rebuilds it every round -- through ``pedantic``'s untimed ``setup`` rather than inside the timed
callable, the one row in the suite that needs the distinction -- the load costs more than the
dilation it precedes, so folding it in would report the load (``BenchLibrary.run`` says why).

**libigl** covers the two lattice functions, ``grid`` and ``unique_sparse_voxel_corners``. Both are
single-core C++ over the same integer arithmetic, so they read as a bandwidth comparison.

``trimesh.voxel.creation.voxelize_subdivide`` is timed on the ``voxelize_mesh`` axis but is
**not** a parity reference — it is a different algorithm (subdivide every triangle until its edges
are under half a pitch, then round each vertex to a cell), and its result neither contains nor is
contained in the exact accept set. open3d is the oracle for that group.

The small rows
--------------
``from_cells``, ``grid_transform``, ``resolve_voxel_grid``, ``cell_indices``, ``cell_centers``,
``occupancy_at_cells``, ``pool_by_voxel``, ``surface_voxels``, ``from_dense`` and ``to_field`` are
each one or two launches over the voxel or point set, so at the small end their rows time the
wrapper floor rather than the operation; they are on the scan sweep anyway, so the large end shows
whether the work ever rises above that floor. ``grid_transform`` reads two values off the volume on
the host and takes one small mesh rather than the sweep.

``cells`` is the reverse case, because for ``order="sorted"`` the wrapper floor *is* the operation
at every size -- flat over three orders of magnitude of voxel counts -- so "the row would time the
floor" is a reason to keep the row rather than to drop it. That is what caught the two host
readbacks the bound stage used to make.

``erode``, ``closing``, ``opening`` and ``surface_voxels`` run the same candidate and probe kernels
as ``dilate``, composed differently, so their rows should read as one or two ``dilate`` rows. Each
carries the dense reference its test compares against: MeshLib's ``shrinkVoxelsMask`` for
``erode`` (the twin of the ``dilate`` row), trimesh's ``morphology.binary_closing`` for ``closing``
and ``scipy.ndimage.binary_opening`` for ``opening``. The dense inputs are padded so the dense side
sees the same unbounded lattice the sparse side does, as ``tests/test_voxels.py`` pads them.
``surface_voxels`` has no reference row: ``trimesh.voxel.morphology.surface`` is the ``erode``
complement on a dense array, which the ``erode`` row already prices.

``union`` / ``intersection`` / ``difference`` and ``revoxelize`` are ordito-only for the reference
half: MeshLib answers the set algebra as a bitwise fold over a dense ``VoxelBitSet``, so a ratio
against a sparse rebuild reports the fixture's density, and trimesh's ``ops.boolean_sparse`` needs
the optional ``sparse`` package, which is not a dependency here. trimesh's ``revoxelized`` samples a
different lattice from the one ordito resamples onto, so it is not the same operation. Both
omissions are declared with ``pytest.mark.parity(..., benchmarked=False)`` in
``tests/test_voxels.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import igl
import numpy as np
import pytest
import pytorch3d.ops as p3d_ops
import pyvista as pv
import scipy.ndimage as ndi
import trimesh as tm
import trimesh.voxel.morphology as tm_morphology
import trimesh.voxel.ops as tm_ops
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, BenchLibrary, points_torch_from_numpy, skip_larger_than

if TYPE_CHECKING:
    from typing_extensions import Buffer

# Cell widths as a fraction of the bounding-box diagonal. The pair is a slope check: 1/256 is 64x
# the cells of 1/64, and the whole point of the flattened (triangle, cell) work-item design is that
# the cost tracks the *accepted* cell count rather than the triangle count. Mirrors
# ``benchmarks/test_reconstruction.py``'s ``_RESAMPLE_CELL_FRACTIONS`` pair.
_CELL_DIVISORS = [64, 256]

# One cell width for the groups whose axis is the mesh rather than the pitch.
_MORPHOLOGY_DIVISOR = 96

# A voxelization at 1/256 of the diagonal and a dense morphology pass over its bounding box are
# both seconds a call on the CPU references.
_HEAVY_ROUNDS = 3


def _voxel_size(bench_case: BenchCase, divisor: int) -> float:
    """Absolute cell width from the mesh's own bbox diagonal, identical on every row."""
    vertices_np = bench_case.vertices_np
    return float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0))) / divisor


@pytest.mark.benchmark(group="voxelize_mesh")
@pytest.mark.benchlibs("ordito", "open3d", "trimesh", "pyvista", "meshlib")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_voxelize_mesh(bench_case: BenchCase, divisor: int) -> None:
    """
    The **cubic** group: one thread per (triangle, candidate cell) pair against two serial loops.

    This is the group that would catch a regression to a thread-per-triangle launch. Per-triangle
    candidate windows span orders of magnitude on any real mesh, so a thread-per-triangle kernel is
    load-imbalanced by that same factor while its total work is unchanged — the symptom would be
    the 1/256 point rising far faster than the accepted cell count does.

    open3d gets the identical absolute cell width and the identical ``min_bound``, so both sides
    lay down the same lattice; ``create_from_triangle_mesh_within_bounds`` is used rather than
    ``create_from_triangle_mesh`` precisely so the bounds are the caller's on both sides.

    **pyvista's row is the solid mask**, ``voxelize_binary_mask``, which is why it takes the same
    ``dimensions`` the divisor implies rather than a cell width: it is sized by grid extent, not by
    pitch. It fills the interior, so read it against ordito's ``mode="solid"`` cost rather than the
    surface row timed here -- the containment relation between the two answers is pinned in
    ``tests/test_voxels.py``. It runs to hundreds of milliseconds at the coarse lattice, so it is
    capped at ``bunny``.
    """
    voxel_size = _voxel_size(bench_case, divisor)
    origin = bench_case.vertices_np.min(axis=0) - 0.5 * voxel_size

    if bench_case.kind == "meshlib":
        # ``meshToVolume`` builds an OpenVDB **distance** band rather than an occupancy set, so its
        # row does strictly more than the surface rows: it evaluates a signed distance within
        # ``surfaceOffset`` voxels of the surface instead of accepting or rejecting each cell. The
        # offset is left at its default of 3 voxels, which is the band width the geometry
        # comparison in ``tests/test_voxels.py`` is checked at. It takes a ``MeshPart`` over a mesh
        # it does not own, so the mesh is held in a name for the row's lifetime.
        skip_larger_than(bench_case, "bunny", "the band is rasterized on the CPU")
        mesh_ml = bench_case.new_mesh_ml()
        part_ml = mm.MeshPart(mesh_ml)
        params_ml = mm.MeshToVolumeParams()
        params_ml.voxelSize = mm.Vector3f(voxel_size, voxel_size, voxel_size)
        volume_ml = bench_case.run(
            lambda: mm.meshToVolume(part_ml, params_ml), rounds=_HEAVY_ROUNDS
        )
        assert volume_ml.dims.x > 0
        return
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "voxelize_binary_mask is 502 ms at 64 cubed")
        mesh_pv = bench_case.mesh_pv
        extent_np = bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0)
        dimensions = tuple(int(max(2, round(float(extent) / voxel_size))) for extent in extent_np)
        mask_pv = bench_case.run(
            lambda: mesh_pv.voxelize_binary_mask(dimensions=dimensions), rounds=_HEAVY_ROUNDS
        )
        assert np.asarray(mask_pv.point_data["mask"]).any()
        return
    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "open3d walks the candidate cube on one core")
        mesh_o3d = bench_case.mesh_o3d
        upper = origin + np.ceil((bench_case.vertices_np.max(axis=0) - origin) / voxel_size) * (
            voxel_size
        )
        grid_o3d = bench_case.run(
            lambda: o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh_o3d, voxel_size=voxel_size, min_bound=origin, max_bound=upper
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert len(grid_o3d.get_voxels()) > 0
        return
    if bench_case.kind == "trimesh":
        skip_larger_than(
            bench_case, "bunny_decimated", "voxelize_subdivide subdivides the whole mesh in Python"
        )
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = bench_case.run(
            lambda: tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size),
            rounds=_HEAVY_ROUNDS,
        )
        assert grid_tm.filled_count > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    origin_wp = wp.vec3(*origin.tolist())
    grid = bench_case.run(
        lambda: od.voxels.voxelize_mesh(vertices, faces, voxel_size, origin=origin_wp),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(grid.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="voxelize_points")
@pytest.mark.benchlibs("ordito", "open3d")
def test_voxelize_points(bench_case: BenchCase) -> None:
    """
    Pure ``N``: bin a cloud and deduplicate, with the pitch pinned so only the point count moves.

    This is where the grid build itself is regression-guarded: it is markedly cheaper than the
    equivalent ``grouping.unique_rows`` dedup, which is what the module's data structure choice
    rests on.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "open3d bins into a hash map on one core")
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
        grid_o3d = bench_case.run(
            lambda: o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
        )
        assert len(grid_o3d.get_voxels()) > 0
        return

    points = bench_case.vertices_wp
    grid = bench_case.run(lambda: od.voxels.voxelize_points(points, voxel_size))
    assert int(grid.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="voxel_down_sample")
@pytest.mark.benchlibs("ordito", "open3d", "meshlib")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_voxel_down_sample(bench_case: BenchCase, divisor: int) -> None:
    """
    The inverse map plus a deterministic segment reduce, against open3d's hash-map accumulation.

    ordito pays a radix sort over the point count that open3d does not, in exchange for a bitwise
    reproducible mean; the pitch axis is what shows whether that sort or the grid build dominates.
    A finer pitch means more voxels and shorter segments, so the sort's share rises while the sort
    itself — one pass over the point count — does not move at all.
    """
    voxel_size = _voxel_size(bench_case, divisor)
    if bench_case.kind == "meshlib":
        # ``pointGridSampling`` stops one step short of the other two rows: it returns the *bitset*
        # of one surviving point per occupied cell and never pools a position, so read it as a lower
        # bound. The cloud is the input and is built outside; it must also outlive the
        # ``PointCloudPart``, which does not own it.
        from meshlib import mrmeshnumpy as mn

        # MeshLib's stubs type this as ``Buffer``, which numpy's stubs implement only from 3.12.
        cloud_ml = mn.pointCloudFromPoints(
            cast("Buffer", np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        )
        part_ml = mm.PointCloudPart(cloud_ml)
        sampled_ml = bench_case.run(lambda: mm.pointGridSampling(part_ml, voxel_size))
        assert 0 < sampled_ml.count() <= bench_case.n_vertices
        return
    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "open3d accumulates into a hash map on one core")
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
        pooled_o3d = bench_case.run(lambda: cloud_o3d.voxel_down_sample(voxel_size))
        assert len(pooled_o3d.points) > 0
        return

    points = bench_case.vertices_wp
    pooled = bench_case.run(lambda: od.voxels.voxel_down_sample(points, voxel_size))
    assert int(pooled.shape[0]) > 0


@pytest.mark.benchmark(group="pool_by_voxel")
@pytest.mark.benchlibs("ordito")
def test_pool_by_voxel(bench_case: BenchCase) -> None:
    """
    ``voxel_down_sample``'s pooling onto a grid the caller already holds: one probe per point.

    The grid is the cloud's own at the morphology pitch, built outside the timed callable, so the
    row is the per-point probe, the segment sort and the mean, with no grid build. ordito-only:
    ``tests/test_voxels.py`` checks it against NumPy, and ``voxel_down_sample`` carries the
    open3d comparison.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    points = bench_case.vertices_wp
    grid = od.voxels.voxelize_points(points, voxel_size)
    pooled = bench_case.run(lambda: od.voxels.pool_by_voxel(grid, points, points))
    assert int(pooled.shape[0]) == int(grid.get_active_stats().voxel_count)


@pytest.mark.benchmark(group="cells")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("order", ["grid", "sorted"])
def test_cells(bench_case: BenchCase, order: Literal["grid", "sorted"]) -> None:
    """
    The two row orders of the same readout, so the pair prices the ordering and nothing else.

    ``"grid"`` is the volume's own leaf-major order and is a slice of a buffer the volume already
    holds -- no launch, no allocation. ``"sorted"`` adds a per-column ``minmax``, a key-packing
    launch, a radix sort over the voxel count and a gather. The difference between the two rows is
    therefore the whole cost of the ordering.

    No reference row: open3d's ``get_voxels`` returns a Python list of ``Voxel`` objects, so a row
    would time the object churn rather than the readout. This is the one group here whose axis is
    not really the mesh -- both orders stay **host-bound at every voxel count** -- which is why the
    group exists at all: it is where a wrapper-floor change shows up.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_points(bench_case.vertices_wp, voxel_size)
    rows = bench_case.run(lambda: od.voxels.cells(grid, order=order))
    assert int(rows.shape[1]) == 3


@pytest.mark.benchmark(group="from_cells")
@pytest.mark.benchlibs("ordito")
def test_from_cells(bench_case: BenchCase) -> None:
    """
    ``cells``' inverse: a NanoVDB index-grid build from integer cell rows, deduplicating as it goes.

    The rows are the grid's own (``cells``, outside the timed callable), so the row is the build at
    the voxel count the morphology groups use. ordito-only.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_points(bench_case.vertices_wp, voxel_size)
    rows = od.voxels.cells(grid)
    _size, origin = od.voxels.grid_transform(grid)
    rebuilt = bench_case.run(lambda: od.voxels.from_cells(rows, voxel_size, origin))
    assert int(rebuilt.get_active_stats().voxel_count) == int(rows.shape[0])


@pytest.mark.benchmark(group="grid_transform")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("ordito")
def test_grid_transform(bench_case: BenchCase) -> None:
    """
    The cell width and origin read off the volume's transform: host work, flat in the voxel count.

    One small mesh is the whole axis, since nothing about the grid's size reaches this call.
    ordito-only.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_points(bench_case.vertices_wp, voxel_size)
    size, _origin = bench_case.run(lambda: od.voxels.grid_transform(grid))
    assert np.isclose(size, voxel_size, rtol=1e-5)


@pytest.mark.benchmark(group="resolve_voxel_grid")
@pytest.mark.benchlibs("ordito")
def test_resolve_voxel_grid(bench_case: BenchCase) -> None:
    """
    The default lattice for a cloud: one bounding-box reduction, then host arithmetic.

    Both defaults are left unset, so the row prices the ``aabb`` it takes; that reduction is what
    scales with the point count. ordito-only.
    """
    points = bench_case.vertices_wp
    voxel_size, _origin = bench_case.run(lambda: od.voxels.resolve_voxel_grid(points))
    assert voxel_size > 0.0


@pytest.mark.benchmark(group="cell_indices")
@pytest.mark.benchlibs("ordito")
def test_cell_indices(bench_case: BenchCase) -> None:
    """The integer cell of every point: one floor per coordinate, a pure per-point map."""
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    points = bench_case.vertices_wp
    origin = wp.vec3(*bench_case.vertices_np.min(axis=0).tolist())
    rows = bench_case.run(lambda: od.voxels.cell_indices(points, voxel_size, origin=origin))
    assert tuple(rows.shape) == (bench_case.n_vertices, 3)


@pytest.mark.benchmark(group="cell_centers")
@pytest.mark.benchlibs("ordito")
def test_cell_centers(bench_case: BenchCase) -> None:
    """The world-space centre of every voxel: one index-to-world map per voxel. ordito-only."""
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_points(bench_case.vertices_wp, voxel_size)
    centers = bench_case.run(lambda: od.voxels.cell_centers(grid))
    assert centers.size == int(grid.get_active_stats().voxel_count)


# Query counts for the membership group. The grid is fixed, so this axis is purely "how does the
# cost of a lookup scale" -- one serial hash probe per query on the reference, one kernel thread
# per query here.
_QUERY_COUNTS = [10_000, 1_000_000]


@pytest.mark.benchmark(group="occupancy_at_points")
@pytest.mark.benchlibs("ordito", "open3d")
@pytest.mark.parametrize("n_queries", _QUERY_COUNTS)
def test_occupancy_at_points(bench_case: BenchCase, n_queries: int) -> None:
    """
    Membership against a pinned grid: ``check_if_included`` is one hash lookup per query, serially.

    The queries are the mesh's own vertices, tiled with a deterministic jitter so that the answer
    is a genuine mix of hits and misses rather than all-true — a query set entirely inside the grid
    would let a constant answer keep pace.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    vertices_np = bench_case.vertices_np
    rng = np.random.default_rng(4)
    repeats = int(np.ceil(n_queries / vertices_np.shape[0]))
    queries_np = np.ascontiguousarray(
        (
            np.tile(vertices_np, (repeats, 1))
            + rng.normal(0.0, 2.0 * voxel_size, (repeats * vertices_np.shape[0], 3))
        )[:n_queries]
    )

    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "one serial hash lookup per query")
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_np))
        grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
        queries_o3d = o3d.utility.Vector3dVector(queries_np)
        inside_o3d = bench_case.run(lambda: grid_o3d.check_if_included(queries_o3d))
        assert len(inside_o3d) == n_queries
        return

    points = bench_case.vertices_wp
    grid = od.voxels.voxelize_points(points, voxel_size)
    queries = wp.array(queries_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device)
    mask = bench_case.run(lambda: od.voxels.occupancy_at_points(grid, queries))
    assert int(mask.shape[0]) == n_queries


@pytest.mark.benchmark(group="occupancy_at_cells")
@pytest.mark.benchlibs("ordito")
def test_occupancy_at_cells(bench_case: BenchCase) -> None:
    """
    Membership of integer cells: ``occupancy_at_points`` without the world-to-index map.

    The queries are the cells of the grid's one-shell dilation (built outside the timed callable),
    so the answer is a genuine mix: every original cell is a hit and the added shell is all misses.
    ordito-only: ``occupancy_at_points`` carries the open3d comparison.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    queries = od.voxels.cells(od.voxels.dilate(grid))
    mask = bench_case.run(lambda: od.voxels.occupancy_at_cells(grid, queries))
    assert int(mask.shape[0]) == int(queries.shape[0])


def _shifted_grid(grid: wp.Volume, device: str | None) -> wp.Volume:
    """
    Return the same voxel set moved three cells along ``x``, on the same lattice.

    Smaller than any fixture's extent at the morphology pitch, so the two sets overlap heavily and
    still differ: all three set operations do real work, as in ``tests/test_voxels.py``.
    """
    voxel_size, origin = od.voxels.grid_transform(grid)
    rows_np = od.voxels.cells(grid).numpy() + np.array([3, 0, 0], dtype=np.int32)
    rows = odt.as_array2d(wp.array(rows_np, dtype=wp.int32, device=device), wp.int32)
    return od.voxels.from_cells(rows, voxel_size, origin)


@pytest.mark.benchmark(group="union")
@pytest.mark.benchlibs("ordito")
def test_union(bench_case: BenchCase) -> None:
    """
    Cells in either grid: the two cell arrays concatenated and handed to the builder.

    The cheapest of the three set operations, with no membership probe; read the other two
    against it. ordito-only (see the module docstring).
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    shifted = _shifted_grid(grid, bench_case.device)
    joined = bench_case.run(lambda: od.voxels.union(grid, shifted))
    assert int(joined.get_active_stats().voxel_count) > int(grid.get_active_stats().voxel_count)


@pytest.mark.benchmark(group="intersection")
@pytest.mark.benchlibs("ordito")
def test_intersection(bench_case: BenchCase) -> None:
    """Cells in both grids: one probe of the second grid per cell of the first. ordito-only."""
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    shifted = _shifted_grid(grid, bench_case.device)
    common = bench_case.run(lambda: od.voxels.intersection(grid, shifted))
    assert int(common.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="difference")
@pytest.mark.benchlibs("ordito")
def test_difference(bench_case: BenchCase) -> None:
    """Cells in the first grid only: ``intersection``'s probe with the test inverted."""
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    shifted = _shifted_grid(grid, bench_case.device)
    remainder = bench_case.run(lambda: od.voxels.difference(grid, shifted))
    assert int(remainder.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="revoxelize")
@pytest.mark.benchlibs("ordito")
def test_revoxelize(bench_case: BenchCase) -> None:
    """
    Resample onto a lattice of half the cell width: one occupancy probe per new cell centre.

    Refinement rather than coarsening, so the dense lattice the probe runs over is eight times the
    source box and the row prices the expensive direction. ordito-only (see the module docstring).
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    finer = bench_case.run(
        lambda: od.voxels.revoxelize(grid, 0.5 * voxel_size), rounds=_HEAVY_ROUNDS
    )
    assert int(finer.get_active_stats().voxel_count) > int(grid.get_active_stats().voxel_count)


def _voxel_mask_ml(occupancy_np: np.ndarray) -> mm.VoxelBitSet:
    """
    Load a dense occupancy array into a MeshLib ``VoxelBitSet``, in ``VoxelId`` order.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib_bitset`` (the two suites do not
    import each other). ``BitSet.fromBlocks`` takes the packed ``uint64`` blocks as a Python
    list, which is why this is one ``np.packbits`` and not a per-cell
    ``set()`` loop -- two orders of magnitude cheaper. ``VolumeIndexer`` decodes
    ``x + dims.x * y + dims.x * dims.y * z``, so ``x`` runs fastest and the array flattens with
    ``order="F"``; ``fromBlocks`` rounds up to whole blocks, so the size is trimmed back after.

    This is ``setup`` work, not timed work: ``expandVoxelsMask`` **mutates** the bitset, so every
    round needs a fresh one, and since the load costs more than the dilation, building it inside the
    timed callable would report the load instead of the morphology.
    """
    packed_np = np.packbits(occupancy_np.ravel(order="F"), bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(packed_np.tolist())
    bitset_ml.resize(occupancy_np.size)
    return mm.VoxelBitSet(bitset_ml)


def _expand_ml(mask_ml: mm.VoxelBitSet, indexer_ml: mm.VolumeIndexer) -> mm.VoxelBitSet:
    """Dilate in place by one 6-neighbour shell; ``expandVoxelsMask`` itself returns ``None``."""
    mm.expandVoxelsMask(mask_ml, indexer_ml, 1)
    return mask_ml


def _shrink_ml(mask_ml: mm.VoxelBitSet, indexer_ml: mm.VolumeIndexer) -> mm.VoxelBitSet:
    """Erode in place by one 6-neighbour shell; ``shrinkVoxelsMask`` itself returns ``None``."""
    mm.shrinkVoxelsMask(mask_ml, indexer_ml, 1)
    return mask_ml


def _thick_dense_tm(bench_case: BenchCase, voxel_size: float, pad: int) -> np.ndarray:
    """
    Return trimesh's surface voxelization, padded by ``pad`` cells and dilated once.

    A two-sided shell, the dense counterpart of ``_thick_grid``, for the rows whose operation
    removes cells: a one-cell shell erodes or opens to almost nothing, which would time an empty
    answer. The padding gives the dense side room for every cell the sparse side may reach.
    """
    mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
    occupancy_np = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).encoding.dense
    return ndi.binary_dilation(np.pad(occupancy_np, pad))


def _thick_grid(bench_case: BenchCase, voxel_size: float) -> wp.Volume:
    """Return the mesh's surface voxelization dilated once: the cell-removing rows' input."""
    return od.voxels.dilate(
        od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    )


@pytest.mark.benchmark(group="dilate")
@pytest.mark.benchlibs("ordito", "trimesh", "meshlib")
def test_dilate(bench_case: BenchCase) -> None:
    """
    Grow the set by one shell: a ``(k + 1) * n_voxels`` candidate buffer against dense ndimage.

    The candidate buffer is what to watch here -- it is 27 ``vec3`` per voxel at
    ``connectivity=26``, hundreds of megabytes at a million voxels, and the reason 6 is the default.
    trimesh's
    reference is ``scipy.ndimage.binary_dilation`` over the *dense* bounding box, so the two sides
    scale with different quantities and the gap widens with sparsity.

    The two CPU rows share one input — trimesh's dense voxelization at the same pitch — so they are
    two dense implementations of the identical 6-neighbour dilation over the identical occupancy,
    asserted equal cell-for-cell in ``tests/test_voxels.py``. MeshLib's is the multi-threaded one,
    which is the whole reason it is here: it walks the same dense box on every core.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind in ("trimesh", "meshlib"):
        skip_larger_than(bench_case, "bunny", "both dilate the whole dense bounding box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size)
        encoding_tm = grid_tm.encoding
        if bench_case.kind == "trimesh":
            dilated_tm = bench_case.run(
                lambda: tm_morphology.binary_dilation(encoding_tm), rounds=_HEAVY_ROUNDS
            )
            assert dilated_tm.sum > 0
            return

        occupancy_np = encoding_tm.dense
        indexer_ml = mm.VolumeIndexer(mm.Vector3i(*(int(n) for n in occupancy_np.shape)))
        dilated_ml = bench_case.run(
            lambda mask_ml: _expand_ml(mask_ml, indexer_ml),
            setup=lambda: _voxel_mask_ml(occupancy_np),
        )
        assert dilated_ml.count() > int(occupancy_np.sum())
        return

    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    dilated = bench_case.run(lambda: od.voxels.dilate(grid))
    assert int(dilated.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="erode")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_erode(bench_case: BenchCase) -> None:
    """
    Shrink the set by one shell: ``dilate``'s probe kernel with the test inverted.

    The input is a dilated surface voxelization (``_thick_grid``), so the erosion keeps a real
    core. MeshLib's ``shrinkVoxelsMask`` is the dense multi-threaded twin of the ``dilate`` row's
    ``expandVoxelsMask``, asserted equal to ``erode`` in ``tests/test_voxels.py``; its input is the
    same kind of two-sided shell from trimesh's voxelization, and it mutates, so the bitset is
    rebuilt in the untimed ``setup`` as in ``dilate``.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny", "MeshLib erodes the whole dense bounding box")
        occupancy_np = _thick_dense_tm(bench_case, voxel_size, 1)
        indexer_ml = mm.VolumeIndexer(mm.Vector3i(*(int(n) for n in occupancy_np.shape)))
        eroded_ml = bench_case.run(
            lambda mask_ml: _shrink_ml(mask_ml, indexer_ml),
            setup=lambda: _voxel_mask_ml(occupancy_np),
        )
        assert 0 < eroded_ml.count() < int(occupancy_np.sum())
        return

    grid = _thick_grid(bench_case, voxel_size)
    eroded = bench_case.run(lambda: od.voxels.erode(grid))
    assert 0 < int(eroded.get_active_stats().voxel_count) < int(grid.get_active_stats().voxel_count)


@pytest.mark.benchmark(group="closing")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_closing(bench_case: BenchCase) -> None:
    """
    ``dilate`` then ``erode`` on the surface voxelization: two morphology passes in one call.

    trimesh's ``morphology.binary_closing`` is ``scipy.ndimage.binary_closing`` over the dense box,
    which ``tests/test_voxels.py`` compares against; the box is padded by two cells, as the test
    pads it, so the dense intermediate dilation is not clipped at the array border.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "two dense ndimage passes over the bounding box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        occupancy_np = np.pad(
            tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).encoding.dense, 2
        )
        closed_tm = bench_case.run(
            lambda: tm_morphology.binary_closing(occupancy_np), rounds=_HEAVY_ROUNDS
        )
        assert closed_tm.sum >= int(occupancy_np.sum())
        return

    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    closed = bench_case.run(lambda: od.voxels.closing(grid))
    assert int(closed.get_active_stats().voxel_count) >= int(grid.get_active_stats().voxel_count)


@pytest.mark.benchmark(group="opening")
@pytest.mark.benchlibs("ordito", "scipy")
def test_opening(bench_case: BenchCase) -> None:
    """
    ``erode`` then ``dilate`` on a two-sided shell, so the opening keeps a real core.

    scipy's ``ndimage.binary_opening`` is the reference ``tests/test_voxels.py`` compares against
    (trimesh does not wrap it), run over the dense two-sided shell from trimesh's voxelization.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "scipy":
        skip_larger_than(bench_case, "bunny", "two dense ndimage passes over the bounding box")
        occupancy_np = _thick_dense_tm(bench_case, voxel_size, 2)
        opened_np = bench_case.run(lambda: ndi.binary_opening(occupancy_np), rounds=_HEAVY_ROUNDS)
        assert 0 < int(opened_np.sum()) <= int(occupancy_np.sum())
        return

    grid = _thick_grid(bench_case, voxel_size)
    opened = bench_case.run(lambda: od.voxels.opening(grid))
    assert int(opened.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="surface_voxels")
@pytest.mark.benchlibs("ordito")
def test_surface_voxels(bench_case: BenchCase) -> None:
    """
    The cells with an empty neighbour: ``erode``'s probe, keeping the complement.

    On the same two-sided shell as ``erode``, so the two rows read side by side. ordito-only (see
    the module docstring).
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = _thick_grid(bench_case, voxel_size)
    shell = bench_case.run(lambda: od.voxels.surface_voxels(grid))
    assert 0 < int(shell.get_active_stats().voxel_count) <= int(grid.get_active_stats().voxel_count)


@pytest.mark.benchmark(group="fill_cavities")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_fill_cavities(bench_case: BenchCase) -> None:
    """
    The one group whose cost is *not* the grid: it is the empty complement, which grows cubically.

    ordito labels the empty cells' 6-connected components with an ECL-CC union-find driven from an
    implicit stencil — three backward probes per cell, one launch, zero edge memory — where
    ``scipy.ndimage.binary_fill_holes`` runs a serial binary propagation over the same dense box.
    A regression to an explicit edge list would show up here first, as memory before time.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "binary_fill_holes iterates over the dense box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size)
        encoding_tm = grid_tm.encoding
        filled_tm = bench_case.run(
            lambda: tm_morphology.fill_holes(encoding_tm), rounds=_HEAVY_ROUNDS
        )
        assert filled_tm.sum > 0
        return

    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    filled = bench_case.run(lambda: od.voxels.fill_cavities(grid), rounds=_HEAVY_ROUNDS)
    assert int(filled.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="fill_orthographic")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_fill_orthographic(bench_case: BenchCase) -> None:
    """
    The cheap fill: three axis sweeps and an intersection, against three dense NumPy reductions.

    Both sides work on the dense occupancy box, so unlike ``fill_cavities`` this pair really is the
    same shape of work on the same data and reads as a straight parallel-versus-serial comparison.
    trimesh's version builds three ``cumsum``-style masks with NumPy, so it is the fastest of its
    morphology functions and the hardest of them to beat.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "three dense NumPy passes over the bounding box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        occupancy_np = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).matrix
        filled_tm = bench_case.run(
            lambda: tm_ops.fill_orthographic(occupancy_np), rounds=_HEAVY_ROUNDS
        )
        assert filled_tm.sum() > 0
        return

    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    filled = bench_case.run(lambda: od.voxels.fill_orthographic(grid), rounds=_HEAVY_ROUNDS)
    assert int(filled.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="to_boxes")
@pytest.mark.benchlibs("ordito", "trimesh", "pyvista")
def test_to_boxes(bench_case: BenchCase) -> None:
    """
    Mesh the voxel set as cubes: shared nanogrid corners against ``12 n`` unshared triangles.

    ``multibox`` tiles a template cube per centre and never welds, so it always emits ``8 n``
    vertices; ordito's corners come deduplicated out of ``warp.fem``'s vertex grid, which is a
    smaller output *and* skips a welding pass. Both rows emit every face (``cull_internal=False``)
    so the comparison is like for like.

    pyvista reaches the same answer through ``glyph(geom=pv.Cube(), scale=False, orient=False)``,
    which is VTK's template-instancing filter and so belongs with trimesh on the unwelded side:
    **exactly 12 triangles per centre** after ``.triangulate()``, the same ``12 n`` ``multibox``
    emits. ``scale=False`` and ``orient=False`` are both
    load-bearing: left at their defaults the filter reads a scalar and a vector array off the cloud
    and would size and rotate each cube. The ``.triangulate()`` is inside the row because without it
    the output is quads and the counts are not comparable with either other side.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "VTK instances a template cube per voxel")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        centers_np = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).points
        cloud_pv = pv.PolyData(np.ascontiguousarray(centers_np))
        cube_pv = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
        boxes_pv = bench_case.run(
            lambda: cloud_pv.glyph(geom=cube_pv, scale=False, orient=False).triangulate(),
            rounds=_HEAVY_ROUNDS,
        )
        assert boxes_pv.n_cells == 12 * centers_np.shape[0]
        return
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "multibox tiles a template cube per voxel in Python")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        centers_np = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).points
        boxes_tm = bench_case.run(
            lambda: tm_ops.multibox(centers_np, pitch=voxel_size), rounds=_HEAVY_ROUNDS
        )
        assert boxes_tm.faces.shape[0] > 0
        return

    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    _vertices, faces = bench_case.run(lambda: od.voxels.to_boxes(grid, cull_internal=False))
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="voxel_corners")
@pytest.mark.benchlibs("ordito", "igl")
def test_voxel_corners(bench_case: BenchCase) -> None:
    """
    The deduplicated corner lattice, from ``warp.fem``'s vertex grid against igl's own hash.

    igl's ``unique_sparse_voxel_corners`` builds all ``8 n`` candidate subscripts, packs each into
    an ``int64`` code and runs ``igl::unique`` over them. ordito builds none of that: the nanogrid
    geometry derives its vertex grid from the cell grid, and the per-cell indices are eight ``O(1)``
    probes. The geometry construction is the fixed cost on this row.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl hashes 8 n candidate subscripts on one core")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size)
        cells_np = np.ascontiguousarray(grid_tm.sparse_indices.astype(np.int64))
        depth = int(np.ceil(np.log2(cells_np.max() + 2)))
        origin_np = np.zeros(3)
        corners_igl = bench_case.run(
            lambda: igl.unique_sparse_voxel_corners(
                origin_np, voxel_size * (1 << depth), depth, cells_np
            )[0],
            rounds=_HEAVY_ROUNDS,
        )
        assert corners_igl.shape[0] > 0
        return

    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    corners, _cell_corners = bench_case.run(lambda: od.voxels.voxel_corners(grid))
    assert int(corners.shape[0]) > 0


# Lattice resolutions for the mesh-free group. 256 cubed is 16.7 M samples, the size at which an
# SDF round trip is actually run.
_LATTICE_RESOLUTIONS = [64, 256]


_normalized_np_cache: dict[str, np.ndarray] = {}
_splat_field_cache: dict[int, np.ndarray] = {}


def _splat_bounds(bench_case: BenchCase) -> tuple[wp.vec3, wp.vec3]:
    """Return the mesh's bounding box as a lattice ``bounds`` pair, so no vertex is outside."""
    lower_np = bench_case.vertices_np.min(axis=0)
    upper_np = bench_case.vertices_np.max(axis=0)
    return wp.vec3(*lower_np.tolist()), wp.vec3(*upper_np.tolist())


def _normalized_points_np(bench_case: BenchCase) -> np.ndarray:
    """
    Map the mesh's vertices into the ``[-1, 1]`` cube pytorch3d's local space wants.

    An *input* rather than part of the work, so it is cached and built on the host: the ordito row
    reads the raw positions and a ``bounds`` pair instead, which is the same affine map expressed
    where ordito expresses it. Handing pytorch3d unnormalized coordinates would put every point
    outside its volume and time a clamp.
    """
    if bench_case.mesh_name not in _normalized_np_cache:
        vertices_np = bench_case.vertices_np
        lower_np, upper_np = vertices_np.min(axis=0), vertices_np.max(axis=0)
        extent_np = np.where(upper_np > lower_np, upper_np - lower_np, 1.0)
        _normalized_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            2.0 * (vertices_np - lower_np) / extent_np - 1.0, dtype=np.float32
        )
    return _normalized_np_cache[bench_case.mesh_name]


def _splat_field_np(resolution: int) -> np.ndarray:
    """Build a fixed lattice to sample, cached per resolution: the input, not the operation."""
    if resolution not in _splat_field_cache:
        _splat_field_cache[resolution] = np.ascontiguousarray(
            np.random.default_rng(0).normal(size=(resolution,) * 3), dtype=np.float32
        )
    return _splat_field_cache[resolution]


@pytest.mark.benchmark(group="splat_onto_grid")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "pytorch3d")
def test_splat_onto_grid(bench_case: BenchCase) -> None:
    """
    Scatter a per-point vector onto a dense lattice: eight atomic adds per point, then a divide.

    The only group in this module whose input is a *cloud* rather than a grid, and the only one
    whose cost is linear in the point count rather than cubic in the resolution -- the lattice is
    held at 64^3 so the axis is the mesh's vertex count alone.

    **pytorch3d** is the only reference and it is the same algorithm at the same weights, pinned
    bit-for-bit on the host in ``tests/test_voxels.py::test_splat_onto_grid_matches_pytorch3d``. It
    has CUDA kernels of its own, so both its rows are real: read the ``-cuda`` one against
    ``ordito-cuda``. Its lattice is transposed relative to ordito's and its coordinates are the
    ``[-1, 1]`` cube, so the row hands it the equivalent box rather than the same numbers; neither
    difference is work. Its ``volume_densities`` / ``volume_features`` are *inputs* it accumulates
    into, so they are zeroed outside the timed callable -- allocating them inside would price two
    ``64^3`` allocations, which is what ordito's row pays and states below.
    """
    resolution = 64
    if bench_case.kind == "pytorch3d":
        import torch

        points_p3d = points_torch_from_numpy(
            _normalized_points_np(bench_case), bench_case.torch_device
        )
        values_p3d = points_torch_from_numpy(
            _normalized_points_np(bench_case), bench_case.torch_device
        )
        densities_p3d = torch.zeros(
            (1, 1, resolution, resolution, resolution), device=bench_case.torch_device
        )
        features_p3d = torch.zeros(
            (1, 3, resolution, resolution, resolution), device=bench_case.torch_device
        )
        splatted_p3d = bench_case.run(
            lambda: p3d_ops.add_points_features_to_volume_densities_features(
                points_p3d,
                values_p3d,
                densities_p3d.clone(),
                features_p3d.clone(),
                mode="trilinear",
                align_corners=True,
            )
        )
        assert splatted_p3d[0].shape[1] == 3
        return
    points = bench_case.vertices_wp
    bounds = _splat_bounds(bench_case)
    field, density = bench_case.run(
        lambda: od.voxels.splat_onto_grid(
            points, points, (resolution, resolution, resolution), bounds=bounds
        )
    )
    assert field.shape == (resolution,) * 3
    assert int(density.shape[0]) == resolution


@pytest.mark.benchmark(group="sample_grid_trilinear")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "pytorch3d")
def test_sample_grid_trilinear(bench_case: BenchCase) -> None:
    """
    The gather half: eight coalesced lattice reads per query against the same eight atomics.

    Read against ``splat_onto_grid`` above -- same stencil, same lattice, same query set, and the
    only difference is scatter against gather. The gap is what atomic contention costs, which is
    otherwise invisible.

    **pytorch3d** routes its volume sampling through ``torch.nn.functional.grid_sample``, which is
    what this row times: that is torch's own trilinear sampler and the independent implementation
    ``tests/test_voxels.py::test_sample_grid_trilinear_matches_pytorch3d`` compares against
    (4.77e-07). ``align_corners=True`` and ``padding_mode="border"`` are the settings that match
    ordito's ``bounds`` and its clamped stencil; both are non-default and both change the answer
    rather than the cost.
    """
    resolution = 64
    field_np = _splat_field_np(resolution)
    if bench_case.kind == "pytorch3d":
        import torch

        volume_p3d = torch.as_tensor(field_np.transpose(2, 1, 0), device=bench_case.torch_device)[
            None, None
        ]
        queries_p3d = points_torch_from_numpy(
            _normalized_points_np(bench_case), bench_case.torch_device
        ).reshape(1, 1, 1, -1, 3)
        sampled_p3d = bench_case.run(
            lambda: torch.nn.functional.grid_sample(
                volume_p3d, queries_p3d, mode="bilinear", padding_mode="border", align_corners=True
            )
        )
        assert sampled_p3d.numel() == bench_case.n_vertices
        return
    field = odt.as_array3d(
        wp.array(field_np, dtype=wp.float32, device=bench_case.device), wp.float32
    )
    points = bench_case.vertices_wp
    bounds = _splat_bounds(bench_case)
    sampled = bench_case.run(lambda: od.voxels.sample_grid_trilinear(field, points, bounds=bounds))
    assert sampled.size == bench_case.n_vertices


@pytest.mark.benchmark(group="grid_points")
@pytest.mark.benchlibs("ordito", "igl", "pyvista")
@pytest.mark.parametrize("resolution", _LATTICE_RESOLUTIONS)
def test_grid_points(bench_lib: BenchLibrary, resolution: int) -> None:
    """
    Lattice generation and nothing else, so this row reads as a memory-bandwidth floor.

    One of the few mesh-free groups in the suite (``bench_lib`` rather than ``bench_case``): the
    axis is resolution, and there is no input to speak of. ``igl.grid`` fills the same
    ``resolution ** 3`` by 3 array with a triple loop.

    pyvista's ``ImageData(...).points`` is VTK's lattice generation and nothing else, which is the
    honest caveat and also the point: it is a *constructor*, so the row measures the same allocate-
    and-fill this group exists to price. Its ordering is **x fastest** where ordito's is z fastest,
    so the two differ by a permutation and not by a value (``tests/test_voxels.py`` names it).
    """
    shape = (resolution, resolution, resolution)
    if bench_lib.kind == "pyvista":
        if resolution > 128:
            pytest.skip("VTK materializes 16.7 M points through a Python property")
        spacing = 1.0 / max(resolution - 1, 1)
        lattice_pv = bench_lib.run(
            lambda: np.asarray(
                pv.ImageData(
                    dimensions=shape, origin=(0.0, 0.0, 0.0), spacing=(spacing,) * 3
                ).points
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert lattice_pv.shape[0] == resolution**3
        return
    if bench_lib.kind == "igl":
        if resolution > 128:
            pytest.skip("igl fills 16.7 M rows with a serial triple loop")
        res_np = np.array(shape)
        lattice_igl = bench_lib.run(lambda: igl.grid(res_np), rounds=_HEAVY_ROUNDS)
        assert lattice_igl.shape[0] == resolution**3
        return

    device = bench_lib.device
    bounds = (wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0))
    lattice = bench_lib.run(lambda: od.voxels.grid_points(shape, bounds=bounds, device=device))
    assert lattice.size == resolution**3


@pytest.mark.benchmark(group="to_dense")
@pytest.mark.benchlibs("ordito")
def test_to_dense(bench_case: BenchCase) -> None:
    """
    The tight occupied box as a dense lattice: one bounds launch and readback, then one scatter.

    ordito-only: trimesh's ``sparse_to_matrix`` takes non-negative cells and returns no origin, so
    its lattice is not this one. The grid is the mesh voxelized at the morphology groups' size,
    built outside the timed callable.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    occupancy, _origin = bench_case.run(lambda: od.voxels.to_dense(grid))
    assert occupancy.ndim == 3


@pytest.mark.benchmark(group="from_dense")
@pytest.mark.benchlibs("ordito")
def test_from_dense(bench_case: BenchCase) -> None:
    """
    ``to_dense``'s inverse: compact the occupied lattice cells and build the grid from them.

    The lattice is ``to_dense``'s answer on the same grid, built outside the timed callable, so the
    pair is a round trip at one voxel set. ordito-only.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    occupancy, origin_cell = od.voxels.to_dense(grid)
    _size, origin = od.voxels.grid_transform(grid)
    rebuilt = bench_case.run(
        lambda: od.voxels.from_dense(occupancy, voxel_size, origin, origin_cell=origin_cell)
    )
    assert int(rebuilt.get_active_stats().voxel_count) == int(grid.get_active_stats().voxel_count)


@pytest.mark.benchmark(group="to_field")
@pytest.mark.benchlibs("ordito")
def test_to_field(bench_case: BenchCase) -> None:
    """
    ``to_dense`` padded by one cell and written as ``float32``: the field marching cubes takes.

    ordito-only: trimesh's ``matrix_to_marching_cubes`` builds the same padded field internally and
    never returns it.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    grid = od.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    field, _bounds = bench_case.run(lambda: od.voxels.to_field(grid))
    assert field.ndim == 3
