"""
Benchmarks for ``triwarp.bounds``: the axis-aligned box, its diagonal, and the oriented box.

Axis: the **scan sweep**, the only thing the axis-aligned groups can be driven by — a component-wise
min/max over ``V`` has no topology and no parameters. ``oriented_bounding_box`` has a second axis of
its own, the candidate count, held *fixed* at ``_ROTATIONS`` here so the sweep still reads as a
sweep.

This module exists for one reason beyond coverage: ``aabb`` is on the hot path of *every* k-NN query
(``neighbors.query_*`` calls it to size the hash grid), and it is the clearest example in the suite
of a row that is **host-latency-bound at the small end and bandwidth-bound at the large end**. Below
roughly ``10 ** 3`` vertices it reports the wrapper floor rather than the reduction, so a NumPy
``min``/``max`` pair wins by orders of magnitude there and loses at ``lucy``. The row is here for
the crossover; neither endpoint means anything alone. That is also why ``aabb`` does not use the
generic [`reduce.minmax`][triwarp.reduce.minmax] path — it writes both corners into one six-element
buffer so the host pays a single readback.

References
----------
**trimesh** has no ``bounds`` function, only the cached ``Trimesh.bounds`` property, so the row runs
the uncached formula behind it rather than timing a cache lookup. **open3d**'s
``get_axis_aligned_bounding_box`` is its own reduction over the same vertices.

**libigl** answers both groups and in both returns *more* than was asked for: ``igl.bounding_box``
builds the box as geometry (8 corners and 12 triangles), so its row includes constructing a mesh
triwarp never materialises, and ``igl.bounding_box_diagonal`` computes the box internally. Read both
as upper bounds.

For the **oriented** box the references answer it a different way. ``igl.oriented_bounding_box``
searches the *same* global candidate set triwarp's first phase does (Super-Fibonacci over ``SO(3)``,
identity appended) but has no refinement phase, so triwarp's row carries trust-region rounds igl's
does not — rounds that buy the quality the ``tests/test_bounds.py`` bands pin, and that are several
times the sampled-only phase reachable with ``refine_iterations=0``.
``trimesh.bounds.oriented_bounds`` and open3d's ``get_minimal_oriented_bounding_box`` are the
*other* algorithm family — convex hull, then the minimal box flush with each hull face — so their
rows price a hull triwarp never builds and their cost does not scale with the candidate count at
all. Read them as the cost of that approach rather than the same work done slower: trimesh is dearer
than igl on a scan mesh even though the hull collapses the cloud by an order of magnitude, because
the caliper search then runs over every hull face serially.

One warning about igl's row, in the spirit of the ``igl.octree`` finding in ``test_neighbors``:
``igl.oriented_bounding_box`` is threaded and its cost depends on the state of that thread pool —
the same call at the same count spreads several-fold between a standalone probe and the harness, so
quote the in-harness median and nothing else.

``aabb_union`` gets no row: it is four ``wp.vec3`` component-wise minima at Python scope with no
device work, so a row would time the interpreter.

The module's other half runs the opposite way — points against a box that already exists — and its
three groups (``points_in_aabb``, ``crop_points``, ``crop_mesh``) share one shape: triwarp is
**flat** in the input and open3d **linear**, so each row is a *crossover* and not a ratio. The flat
readings are the host launch floor (three to five launches plus the one readback that sizes the
output), so nothing on triwarp's side is bandwidth-bound anywhere in this sweep and a change will
show up as a slope long before it shows up as a level. The crossover sits near 40 000 vertices for
the two point groups and below 2 562 for the mesh crop, which is the one row worth reading as a win.
Every row here is sub-10 ms and therefore clock-state bimodal (see ``benchmarks/README.md``), so
treat the *crossover* as the finding and not any single ratio.

``crop_points`` is the ``points_in_aabb`` row plus a gather, kept rather than declined because
open3d answers it through a *different* entry point (``PointCloud.crop`` rather than
``get_point_indices_within_bounding_box``) whose extra work — materializing the survivors — is the
same extra work triwarp does, so the two rows bracket it on both sides. Check a change against the
gather's own cost, timed **interleaved** in one process, rather than against the difference of two
medians from separate rows: it is one ``wp.copy`` launch and flat over the whole size range.

The two ``*_mask`` entry points get **no** rows: each is its group's row minus the ``flatnonzero``
that sizes the index buffer, so a row would re-time the same ``wp.map`` under a fourth and fifth
name.

**No reference other than open3d answers the query half**, which was checked rather than assumed:
trimesh has no box crop for either a cloud or a mesh, and pyvista's ``clip_box`` is a *clip* — it
splits the straddling cells and introduces vertices the input never had, so it answers a different
question (its ``crinkle=True`` mode keeps whole cells, which is ``face_mode="any"`` and not the
all-corners rule these groups time).
"""

from __future__ import annotations

import igl
import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase

# Candidate orientations scored by every ``oriented_bounding_box`` row, triwarp's and igl's alike.
# Fixed and shared: the two libraries search the identical candidate set, so a row that let them
# score different counts would compare quality against cost. It is also triwarp's own default.
_ROTATIONS = 4096

# Query points for the ``enclosing_diagonal`` row. Fixed count and seed: the cost is two box
# reductions and is flat in the second cloud's size at any realistic count, so sweeping it would
# add a second axis measuring nothing.
_N_QUERIES = 10_000
_QUERY_SEED = 7

_query_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}


def _queries_np(bench_case: BenchCase) -> np.ndarray:
    """Query points outside the mesh's own box, so the union is strictly larger than either side."""
    rng = np.random.default_rng(_QUERY_SEED)
    vertices = bench_case.vertices_np
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return rng.random((_N_QUERIES, 3)) * extent + vertices.max(axis=0)


def _queries_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _query_cache:
        _query_cache[key] = wp.array(
            np.ascontiguousarray(_queries_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _query_cache[key]


@pytest.mark.benchmark(group="aabb")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "igl", "pyvista", "meshlib")
def test_aabb(bench_case: BenchCase) -> None:
    """
    A min/max reduce over the vertices, and the suite's clearest host-latency floor.

    VTK caches nothing but returns the box interleaved per axis (``xmin, xmax, ymin, ...``), which
    is a layout the test decodes and this row does not care about.

    meshlib's ``computeBoundingBox`` returns the two corners directly and is the only row here that
    is **multi-threaded**, which on a reduce this cheap mostly prices its own fork-join -- read it
    against ``triwarp-cuda``, not ``triwarp-cpu``.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        box_ml = bench_case.run(
            # ``None`` is the whole mesh; the stub omits that the region is nullable.
            lambda: mm.computeBoundingBox(mesh_ml.topology, mesh_ml.points, None)  # pyright: ignore[reportArgumentType]
        )
        assert box_ml.min.x <= box_ml.max.x
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        bounds_pv = bench_case.run(lambda: np.asarray(mesh_pv.bounds))
        assert bounds_pv.shape == (6,)
        return
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        lower, upper = bench_case.run(lambda: tw.bounds.aabb(vertices))
        assert lower[0] <= upper[0]
    elif bench_case.kind == "trimesh":
        # what an uncached ``trimesh.Trimesh.bounds`` computes: numpy min/max per axis
        vertices = bench_case.vertices_np
        result = bench_case.run(lambda: np.vstack((vertices.min(axis=0), vertices.max(axis=0))))
        assert result.shape == (2, 3)
    elif bench_case.kind == "igl":
        # Returns the box as geometry: 8 corner vertices plus the 12 triangles of its hull.
        vertices_np = bench_case.vertices_np
        corners_igl, faces_igl = bench_case.run(lambda: igl.bounding_box(vertices_np))
        assert corners_igl.shape == (8, 3)
        assert faces_igl.shape == (12, 3)
    else:  # open3d's own bound reduction over the same vertices
        mesh_o3d = bench_case.mesh_o3d
        box_o3d = bench_case.run(mesh_o3d.get_axis_aligned_bounding_box)
        assert box_o3d.get_min_bound()[0] <= box_o3d.get_max_bound()[0]


@pytest.mark.benchmark(group="enclosing_diagonal")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista")
def test_enclosing_diagonal(bench_case: BenchCase) -> None:
    """
    The default search radius every mesh query derives, over the mesh *and* the query points.

    Two clouds rather than one, so against the ``aabb`` group above this is a second launch into
    the same six-slot buffer -- not a second box and not a second readback. The row answers what
    that launch costs on top of the single-cloud reduction. Every ``max_dist=None`` call in
    ``proximity``, ``ray``, ``visibility`` and ``registration`` pays exactly this.

    Reducing a concatenation of the two clouds instead is declined -- a loss at every size --
    because the union is an allocation and a copy of both clouds against one launch that is flat in
    its ``dim``.

    igl's ``bounding_box_diagonal`` takes one point set, so its side is fed the stacked cloud --
    which makes its row also price the ``vstack`` a caller would need, and that copy is the point:
    triwarp never materializes the union.
    """
    if bench_case.kind == "pyvista":
        # ``DataSet.length`` is the *single*-cloud diagonal -- VTK has no two-cloud form -- so this
        # row is the one-box half of what the other two compute. Read it as a floor.
        mesh_pv = bench_case.mesh_pv
        assert bench_case.run(lambda: float(mesh_pv.length)) > 0.0
        return
    if bench_case.kind == "triwarp":
        vertices, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        diagonal = bench_case.run(lambda: tw.bounds.enclosing_diagonal(vertices, queries))
        assert diagonal > 0.0
        return
    vertices_np, queries_np = bench_case.vertices_np, _queries_np(bench_case)
    diagonal_igl = bench_case.run(
        lambda: igl.bounding_box_diagonal(np.vstack([vertices_np, queries_np]))
    )
    assert diagonal_igl > 0.0


# Box for every containment row: the low corner of the mesh's own bounds, 60% of each side and
# padded 1% outward, which is the construction ``tests/test_bounds.py`` documents. Fixed rather than
# swept: the predicate is six comparisons per point whatever the box holds, so the *selectivity* is
# not a cost axis (verified -- the full box and an empty one measure within noise of each other),
# and only the compaction downstream sees how many points survived.
_BOX_KEEP = 0.6
_BOX_PAD = 0.01

_cloud_o3d_cache: dict[str, o3d.geometry.PointCloud] = {}
_vector_o3d_cache: dict[str, o3d.utility.Vector3dVector] = {}


def _box_np(bench_case: BenchCase) -> np.ndarray:
    """Build the ``(2, 3)`` query box on the host, so both sides get identical corners."""
    vertices_np = bench_case.vertices_np
    lower_np = vertices_np.min(axis=0)
    extent_np = vertices_np.max(axis=0) - lower_np
    return np.stack([lower_np - _BOX_PAD * extent_np, lower_np + _BOX_KEEP * extent_np])


def _corners_wp(box_np: np.ndarray) -> tuple[wp.vec3, wp.vec3]:
    """Return the same box as the two ``wp.vec3`` corners triwarp takes."""
    return wp.vec3(*box_np[0].tolist()), wp.vec3(*box_np[1].tolist())


def _vector_o3d(bench_case: BenchCase) -> o3d.utility.Vector3dVector:
    """Wrap the vertices as an Open3D vector once per mesh: an input conversion, not the op."""
    name = bench_case.mesh_name
    if name not in _vector_o3d_cache:
        _vector_o3d_cache[name] = o3d.utility.Vector3dVector(bench_case.vertices_np)
    return _vector_o3d_cache[name]


def _cloud_o3d(bench_case: BenchCase) -> o3d.geometry.PointCloud:
    """
    Open3D point cloud over the same vertices, built once per mesh.

    One cloud serves every round here, unlike the ``remove_*`` filters in ``test_points.py``:
    ``crop`` returns a fresh geometry and mutates nothing (checked -- the input's point count is
    unchanged after the call).
    """
    name = bench_case.mesh_name
    if name not in _cloud_o3d_cache:
        _cloud_o3d_cache[name] = o3d.geometry.PointCloud(_vector_o3d(bench_case))
    return _cloud_o3d_cache[name]


@pytest.mark.benchmark(group="points_in_aabb")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_points_in_aabb(bench_case: BenchCase) -> None:
    """
    Six comparisons per point, then the scan that compacts them: the module's cheapest group.

    Read this row as a **crossover, not a ratio**. triwarp is flat -- one ``wp.map``, one
    ``array_cast``, one scan, one 4-byte readback and one scatter, none of which the vertex count
    moves at these sizes -- while open3d walks the cloud once on one core and grows linearly, so
    open3d wins by an order of magnitude at the small end and loses by several times at the large
    one. Neither endpoint means anything alone; the sweep is the measurement.

    open3d's ``get_point_indices_within_bounding_box`` returns exactly this index list and is
    compared against it element-wise in ``tests/test_bounds.py``. The ``Vector3dVector`` its box
    query takes is built in setup, so the row prices the query and not the conversion.
    """
    box_np = _box_np(bench_case)
    if bench_case.kind == "triwarp":
        vertices_wp = bench_case.vertices_wp
        lower_wp, upper_wp = _corners_wp(box_np)
        indices_wp = bench_case.run(
            lambda: tw.bounds.points_in_aabb(vertices_wp, lower_wp, upper_wp)
        )
        assert 0 < indices_wp.size < bench_case.n_vertices
        return
    box_o3d = o3d.geometry.AxisAlignedBoundingBox(box_np[0], box_np[1])
    vector_o3d = _vector_o3d(bench_case)
    indices_o3d = bench_case.run(lambda: box_o3d.get_point_indices_within_bounding_box(vector_o3d))
    assert 0 < len(indices_o3d) < bench_case.n_vertices


@pytest.mark.benchmark(group="crop_points")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_crop_points(bench_case: BenchCase) -> None:
    """
    The selection plus the gather that materializes it -- ``points_in_aabb`` with the copy included.

    The gather is flat over the whole size range (one ``wp.copy`` launch), so this row
    should track the ``points_in_aabb`` row at a constant offset and any *slope* appearing here is
    the finding. Both sides materialize: open3d's ``PointCloud.crop`` runs the same index query as
    the group above and then copies the survivors through ``SelectByIndex``, and triwarp returns
    the indices alongside the points so a caller never pays the query twice.
    """
    box_np = _box_np(bench_case)
    if bench_case.kind == "triwarp":
        vertices_wp = bench_case.vertices_wp
        lower_wp, upper_wp = _corners_wp(box_np)
        kept_wp, indices_wp = bench_case.run(
            lambda: tw.bounds.crop_points(vertices_wp, lower_wp, upper_wp)
        )
        assert kept_wp.size == indices_wp.size
        assert 0 < kept_wp.size < bench_case.n_vertices
        return
    box_o3d = o3d.geometry.AxisAlignedBoundingBox(box_np[0], box_np[1])
    cloud_o3d = _cloud_o3d(bench_case)
    kept_o3d = bench_case.run(lambda: cloud_o3d.crop(box_o3d))
    assert 0 < len(kept_o3d.points) < bench_case.n_vertices


@pytest.mark.benchmark(group="crop_mesh")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_crop_mesh(bench_case: BenchCase) -> None:
    """
    The vertex query, the all-corners face reduction, and the compaction that reindexes from zero.

    The module's widest gap and the one worth watching: an order of magnitude at the top of the
    sweep, where the two point groups manage a few times, because the mesh crop is where the
    work stops being one comparison per point -- the face reduction and the vertex compaction both
    scale, and open3d does them serially. It is also flat on triwarp's side across the whole sweep,
    which says the four-launch chain is still host-bound at 327 680 faces and that a change here
    will show up as a *slope* long before it shows up as a level.

    This is the row that pins the mask-input route through
    [`triwarp.selection.submesh_from_vertex_mask`][triwarp.selection.submesh_from_vertex_mask]:
    the crop hands it a mask, which reduces onto faces directly rather than going back through
    ``isin``, and that route is about 2x on CUDA in isolation. Reading this row against the
    ``submesh`` groups in ``benchmarks/test_selection.py`` is how a regression in either is
    attributed.

    open3d's ``TriangleMesh.crop`` applies the identical all-corners rule (compared element-wise,
    modulo vertex renumbering, in ``tests/test_bounds.py``) and returns a fresh mesh, so the shared
    input mesh stays valid across rounds.
    """
    box_np = _box_np(bench_case)
    if bench_case.kind == "triwarp":
        vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
        lower_wp, upper_wp = _corners_wp(box_np)
        sub_vertices_wp, sub_faces_wp = bench_case.run(
            lambda: tw.bounds.crop_mesh(vertices_wp, faces_wp, lower_wp, upper_wp)
        )
        assert 0 < sub_faces_wp.size // 3 < bench_case.n_faces
        assert 0 < sub_vertices_wp.size < bench_case.n_vertices
        return
    box_o3d = o3d.geometry.AxisAlignedBoundingBox(box_np[0], box_np[1])
    mesh_o3d = bench_case.mesh_o3d
    cropped_o3d = bench_case.run(lambda: mesh_o3d.crop(box_o3d))
    assert 0 < len(cropped_o3d.triangles) < bench_case.n_faces


@pytest.mark.benchmark(group="oriented_bounding_box")
@pytest.mark.benchlibs("triwarp", "igl", "trimesh", "open3d", "pyvista")
def test_oriented_bounding_box(bench_case: BenchCase) -> None:
    """
    The sampled-plus-refined minimum-volume box: ``_ROTATIONS`` global frames, then eight rounds.

    Cost is ``rotations * n_vertices`` point transforms for the global phase plus eight 512-frame
    refinement rounds of device-side frame generation, extent scoring and selection. The whole
    search runs on the device and reads back one row at the end, so what this row times is device
    work plus a fixed launch chain, of which the refinement is the larger part. The row times the
    *default*, refinement included, because that is what a caller gets -- and what the quality
    bands in ``tests/test_bounds.py`` are checked against; igl walks the same global candidates over
    a CPU ``parallel_for`` with no refinement phase.

    The CPU references run into hundreds of milliseconds on ``bunny`` and take ``rounds=3`` for it,
    the same allowance the other second-scale rows in the suite use.

    **pyvista's row is PCA of the points**, a single fixed orientation rather than a search -- so it
    is the floor of this group by construction and the only reference here that triwarp should beat
    on *quality* as well.

    open3d times ``get_minimal_oriented_bounding_box`` -- its hull-based approximate minimizer,
    the same algorithm family as trimesh's row -- not ``get_oriented_bounding_box``, whose PCA box
    does not minimize anything and sits measurably above triwarp's volume on a tilted fixture where
    the minimal box is close to it. Like both other references it is insensitive to
    ``_ROTATIONS`` by construction.
    """
    if bench_case.kind == "pyvista":
        # PCA of the points: one fixed orientation, so it is a floor and not a minimizer (the
        # volume gap is pinned in tests/test_bounds.py).
        mesh_pv = bench_case.mesh_pv
        box_pv = bench_case.run(lambda: mesh_pv.oriented_bounding_box(as_composite=False), rounds=3)
        assert box_pv.volume > 0.0
        return
    if bench_case.kind == "open3d":
        vertices_np = bench_case.vertices_np

        def minimal_box_o3d() -> object:
            cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_np))
            return cloud_o3d.get_minimal_oriented_bounding_box()

        assert bench_case.run(minimal_box_o3d, rounds=3).volume() > 0.0
        return
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        rotation, lower, upper = bench_case.run(
            lambda: tw.bounds.oriented_bounding_box(vertices, _ROTATIONS)
        )
        assert lower[0] <= upper[0]
        assert np.isfinite(rotation[0, 0])
    elif bench_case.kind == "igl":
        # The same candidate set at the same count; igl returns the frame alone, transposed relative
        # to triwarp's because it multiplies row vectors on the right.
        vertices_np = bench_case.vertices_np
        frame_igl = bench_case.run(
            lambda: igl.oriented_bounding_box(vertices_np, _ROTATIONS), rounds=3
        )
        assert frame_igl.shape == (3, 3)
    else:
        # A different algorithm, not a slower one: convex hull, then the best hull-face-flush box.
        # Insensitive to ``_ROTATIONS`` by construction, which is why the row does not take it.
        vertices_np = bench_case.vertices_np
        _, extents_tm = bench_case.run(
            lambda: tm.bounds.oriented_bounds(tm.PointCloud(vertices_np)), rounds=3
        )
        assert extents_tm.shape == (3,)
