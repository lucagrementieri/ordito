"""
Benchmarks for ``triwarp.voxels``.

Every group here is really a race between **one hash table and one NanoVDB grid**. open3d keeps a
``std::unordered_map<Eigen::Vector3i>`` and walks it on one core; triwarp keeps a NanoVDB index grid
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

What has no group, and why
--------------------------
``from_cells``, ``grid_transform``, ``cell_indices``, ``cell_centers``,
``occupancy_at_cells``, ``erode``, ``surface_voxels``, ``to_dense``, ``from_dense`` and
``to_field`` are each one launch over the voxel set with no allocation of their
own, so a row would time the wrapper floor rather than the operation. Most are covered anyway
through a group that calls them: ``fill_cavities`` and ``fill_orthographic`` both go through
``to_dense`` (and ``fill_orthographic`` through ``from_dense``, whose candidate pass
``fill_cavities`` fuses into its own last kernel), and ``surface_voxels`` runs the same probe kernel
as ``erode``.

``cells`` is the exception, because for ``order="sorted"`` the wrapper floor *is* the operation at
every size -- flat over three orders of magnitude of voxel counts -- so "the row would time the
floor" is a reason to keep the row rather than to drop it. That is what caught the two host
readbacks the bound stage used to make.

``surface_voxels`` has no reference row either: ``trimesh.voxel.morphology.surface`` exists, but it
is the ``erode`` complement on a dense array and would measure the same thing the ``dilate`` row
already does.

``closing`` and ``opening`` are ``dilate`` and ``erode`` composed in the two orders, so their rows
would be the sum of two rows that already exist. ``union`` / ``intersection`` / ``difference`` and
``revoxelize`` have no row for the reference half rather than the triwarp half: MeshLib answers the
set algebra as a bitwise fold over a dense ``VoxelBitSet``, so a ratio against a sparse rebuild
reports the fixture's density, and trimesh's ``ops.boolean_sparse`` needs the optional ``sparse``
package, which is not a dependency here. ``revoxelize`` is one occupancy probe per new cell over a
dense lattice that ``to_dense`` / ``from_dense`` already carry the cost of. All four declare the
omission with ``pytest.mark.parity(..., benchmarked=False)`` in ``tests/test_voxels.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import igl
import numpy as np
import pytest
import pytorch3d.ops as p3d_ops
import pyvista as pv
import trimesh as tm
import trimesh.voxel.morphology as tm_morphology
import trimesh.voxel.ops as tm_ops
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
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


@pytest.mark.noparity(
    "trimesh",
    oracle="open3d",
    reason="D2 a different algorithm with a measured disagreement: voxelize_subdivide splits every "
    "triangle until its edges are under pitch/2 and then rounds each resulting vertex to a cell, "
    "so it accepts a cell iff some *sample* lands in it. That set neither contains nor is "
    "contained in the exact tri-box accept set -- it misses cells a triangle merely clips and adds "
    "cells no part of the triangle reaches. open3d runs the identical 13-axis test and is asserted "
    "cell-for-cell in tests/test_voxels.py::test_voxelize_mesh_matches_open3d.",
)
@pytest.mark.benchmark(group="voxelize_mesh")
@pytest.mark.benchlibs("triwarp", "open3d", "trimesh", "pyvista", "meshlib")
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
    pitch. It fills the interior, so read it against triwarp's ``mode="solid"`` cost rather than the
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
        lambda: tw.voxels.voxelize_mesh(vertices, faces, voxel_size, origin=origin_wp),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(grid.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="voxelize_points")
@pytest.mark.benchlibs("triwarp", "open3d")
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
    grid = bench_case.run(lambda: tw.voxels.voxelize_points(points, voxel_size))
    assert int(grid.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="voxel_down_sample")
@pytest.mark.benchlibs("triwarp", "open3d", "meshlib")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_voxel_down_sample(bench_case: BenchCase, divisor: int) -> None:
    """
    The inverse map plus a deterministic segment reduce, against open3d's hash-map accumulation.

    triwarp pays a radix sort over the point count that open3d does not, in exchange for a bitwise
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
    pooled = bench_case.run(lambda: tw.voxels.voxel_down_sample(points, voxel_size))
    assert int(pooled.shape[0]) > 0


@pytest.mark.benchmark(group="cells")
@pytest.mark.benchlibs("triwarp")
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
    grid = tw.voxels.voxelize_points(bench_case.vertices_wp, voxel_size)
    rows = bench_case.run(lambda: tw.voxels.cells(grid, order=order))
    assert int(rows.shape[1]) == 3


# Query counts for the membership group. The grid is fixed, so this axis is purely "how does the
# cost of a lookup scale" -- one serial hash probe per query on the reference, one kernel thread
# per query here.
_QUERY_COUNTS = [10_000, 1_000_000]


@pytest.mark.benchmark(group="occupancy_at_points")
@pytest.mark.benchlibs("triwarp", "open3d")
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
    grid = tw.voxels.voxelize_points(points, voxel_size)
    queries = wp.array(queries_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device)
    mask = bench_case.run(lambda: tw.voxels.occupancy_at_points(grid, queries))
    assert int(mask.shape[0]) == n_queries


def _voxel_mask_ml(occupancy_np: np.ndarray) -> mm.VoxelBitSet:
    """
    Load a dense occupancy array into a MeshLib ``VoxelBitSet``, in ``VoxelId`` order.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib_bitset`` (the two suites do not
    import each other). ``BitSet.fromBlocks`` takes the packed ``uint64`` blocks through a
    ``std_vector_unsigned_long``, which is why this is one ``np.packbits`` and not a per-cell
    ``set()`` loop -- two orders of magnitude cheaper. ``VolumeIndexer`` decodes
    ``x + dims.x * y + dims.x * dims.y * z``, so ``x`` runs fastest and the array flattens with
    ``order="F"``; ``fromBlocks`` rounds up to whole blocks, so the size is trimmed back after.

    This is ``setup`` work, not timed work: ``expandVoxelsMask`` **mutates** the bitset, so every
    round needs a fresh one, and since the load costs more than the dilation, building it inside the
    timed callable would report the load instead of the morphology.
    """
    packed_np = np.packbits(occupancy_np.ravel(order="F"), bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(mm.std_vector_unsigned_long(packed_np.tolist()))
    bitset_ml.resize(occupancy_np.size)
    return mm.VoxelBitSet(bitset_ml)


def _expand_ml(mask_ml: mm.VoxelBitSet, indexer_ml: mm.VolumeIndexer) -> mm.VoxelBitSet:
    """Dilate in place by one 6-neighbour shell; ``expandVoxelsMask`` itself returns ``None``."""
    mm.expandVoxelsMask(mask_ml, indexer_ml, 1)
    return mask_ml


@pytest.mark.benchmark(group="dilate")
@pytest.mark.benchlibs("triwarp", "trimesh", "meshlib")
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

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    dilated = bench_case.run(lambda: tw.voxels.dilate(grid))
    assert int(dilated.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="fill_cavities")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_fill_cavities(bench_case: BenchCase) -> None:
    """
    The one group whose cost is *not* the grid: it is the empty complement, which grows cubically.

    triwarp labels the empty cells' 6-connected components with an ECL-CC union-find driven from an
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

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    filled = bench_case.run(lambda: tw.voxels.fill_cavities(grid), rounds=_HEAVY_ROUNDS)
    assert int(filled.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="fill_orthographic")
@pytest.mark.benchlibs("triwarp", "trimesh")
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

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    filled = bench_case.run(lambda: tw.voxels.fill_orthographic(grid), rounds=_HEAVY_ROUNDS)
    assert int(filled.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="to_boxes")
@pytest.mark.benchlibs("triwarp", "trimesh", "pyvista")
def test_to_boxes(bench_case: BenchCase) -> None:
    """
    Mesh the voxel set as cubes: shared nanogrid corners against ``12 n`` unshared triangles.

    ``multibox`` tiles a template cube per centre and never welds, so it always emits ``8 n``
    vertices; triwarp's corners come deduplicated out of ``warp.fem``'s vertex grid, which is a
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

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    _vertices, faces = bench_case.run(lambda: tw.voxels.to_boxes(grid, cull_internal=False))
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="voxel_corners")
@pytest.mark.benchlibs("triwarp", "igl")
def test_voxel_corners(bench_case: BenchCase) -> None:
    """
    The deduplicated corner lattice, from ``warp.fem``'s vertex grid against igl's own hash.

    igl's ``unique_sparse_voxel_corners`` builds all ``8 n`` candidate subscripts, packs each into
    an ``int64`` code and runs ``igl::unique`` over them. triwarp builds none of that: the nanogrid
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

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    corners, _cell_corners = bench_case.run(lambda: tw.voxels.voxel_corners(grid))
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

    An *input* rather than part of the work, so it is cached and built on the host: the triwarp row
    reads the raw positions and a ``bounds`` pair instead, which is the same affine map expressed
    where triwarp expresses it. Handing pytorch3d unnormalized coordinates would put every point
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
@pytest.mark.benchlibs("triwarp", "pytorch3d")
def test_splat_onto_grid(bench_case: BenchCase) -> None:
    """
    Scatter a per-point vector onto a dense lattice: eight atomic adds per point, then a divide.

    The only group in this module whose input is a *cloud* rather than a grid, and the only one
    whose cost is linear in the point count rather than cubic in the resolution -- the lattice is
    held at 64^3 so the axis is the mesh's vertex count alone.

    **pytorch3d** is the only reference and it is the same algorithm at the same weights, pinned
    bit-for-bit on the host in ``tests/test_voxels.py::test_splat_onto_grid_matches_pytorch3d``. It
    has CUDA kernels of its own, so both its rows are real: read the ``-cuda`` one against
    ``triwarp-cuda``. Its lattice is transposed relative to triwarp's and its coordinates are the
    ``[-1, 1]`` cube, so the row hands it the equivalent box rather than the same numbers; neither
    difference is work. Its ``volume_densities`` / ``volume_features`` are *inputs* it accumulates
    into, so they are zeroed outside the timed callable -- allocating them inside would price two
    ``64^3`` allocations, which is what triwarp's row pays and states below.
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
        lambda: tw.voxels.splat_onto_grid(
            points, points, (resolution, resolution, resolution), bounds=bounds
        )
    )
    assert field.shape == (resolution,) * 3
    assert int(density.shape[0]) == resolution


@pytest.mark.benchmark(group="sample_grid_trilinear")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "pytorch3d")
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
    triwarp's ``bounds`` and its clamped stencil; both are non-default and both change the answer
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
    field = twt.as_array3d(
        wp.array(field_np, dtype=wp.float32, device=bench_case.device), wp.float32
    )
    points = bench_case.vertices_wp
    bounds = _splat_bounds(bench_case)
    sampled = bench_case.run(lambda: tw.voxels.sample_grid_trilinear(field, points, bounds=bounds))
    assert sampled.size == bench_case.n_vertices


@pytest.mark.benchmark(group="grid_points")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista")
@pytest.mark.parametrize("resolution", _LATTICE_RESOLUTIONS)
def test_grid_points(bench_lib: BenchLibrary, resolution: int) -> None:
    """
    Lattice generation and nothing else, so this row reads as a memory-bandwidth floor.

    One of the few mesh-free groups in the suite (``bench_lib`` rather than ``bench_case``): the
    axis is resolution, and there is no input to speak of. ``igl.grid`` fills the same
    ``resolution ** 3`` by 3 array with a triple loop.

    pyvista's ``ImageData(...).points`` is VTK's lattice generation and nothing else, which is the
    honest caveat and also the point: it is a *constructor*, so the row measures the same allocate-
    and-fill this group exists to price. Its ordering is **x fastest** where triwarp's is z fastest,
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
    lattice = bench_lib.run(lambda: tw.voxels.grid_points(shape, bounds=bounds, device=device))
    assert lattice.size == resolution**3
