"""
Benchmarks for ``ordito.polyline``.

The module has 27 public functions over one data shape — an ordered ``(n,)`` array of ``wp.vec3`` —
falling into four cost classes. One representative of each is timed rather than all 27, because
within a class the kernels differ only in the per-segment expression:

* **Whole-polyline reductions** (``polyline_length``, ``polyline_centroid``, ``polyline_normal``,
  ``polyline_radius``) — one pass into a scalar, launch-latency bound at these sizes.
  ``polyline_radius`` is the dearest: it projects every segment onto a plane and finds each
  segment's closest point before reducing.
* **Per-vertex maps** (``polyline_angles``, ``cumulative_arc_length``) — one purely local value per
  vertex.
* **Resampling** (``polyline_upsample`` / ``resample`` / ``downsample``) — a data-dependent output
  length, so a scan plus a host readback of the size before the write pass.
* **Simplification** (``polyline_simplify``) — Ramer-Douglas-Peucker evaluated
  **level-synchronously**: one round of four ``dim=n`` launches per level of the split tree, driven
  by ``wp.capture_while`` so no round costs a readback. The cost is the tree's *depth*, about
  ``log2(n)`` on a boundary loop. Warp cannot express the recursion but can express its **levels**,
  and the two accept the same points, because breadth-first and depth-first evaluation of one split
  tree differ only in order — worth nearly two orders of magnitude on the longest rim against a
  single-thread stack kernel. Its Notes carry the rows that lose and why the depth is bounded by
  the accepted count.

``polyline_point_distance`` is timed separately as the only function whose cost is a product of two
sizes (query points x segments).

``triangulate_polygon`` is the module's worked example of a cost that is not where it looks. It
delegates to a **parallel** multi-round ear clipper: a round clips a whole *independent set* of
ears, so what is serial is the **round count**. Up to ``EAR_ONE_BLOCK_MAX`` corners the whole round
loop is **one block** that runs every round itself, which trades the recording and replay of a
conditional graph -- flat in ``n``, and most of a small ring's call -- for three block barriers a
round; past it a round is four ``dim=n`` launches in a ``wp.capture_while`` loop. The 2D ring is
clipped as given: there is no plane fit, and the ring length, orientation and reflex count arrive
in one readback, so a small ring's residual is that readback, the face count and one launch.

The round *count* is what this group caught first: ranking competing ear candidates by raw ring
index lets ear ``i - 2`` suppress ear ``i`` on an alternating star, so exactly **one** ear is
clipped per round and the loop runs to its ``n`` cap. ``select_independent`` ranks by a bijective
hash instead — the textbook maximal-independent-set rule — and retires a constant fraction per
round.

Axis: **polyline**, a loop-length sweep rather than a face-count one, since nothing here reads a
face. Polylines are mesh *boundary loops*; the scan meshes are excluded because they are near-closed
surfaces whose holes are a handful of vertices each, so they would measure launch latency alone. The
longest loop of each mesh is gathered into a dense buffer once per (mesh, device) and reused, so the
timed region holds only the polyline function.

Two groups carry a second sweep on the parameter that drives them: ``polyline_simplify`` on its
tolerance (which sets the tree depth and so the round count) and ``polyline_point_distance`` on the
query count.

References
----------
**meshlib and pyvista are the baselines** and between them cover every group except the two
per-vertex maps. MeshLib's ``Polyline3`` is a complete polyline library; VTK reaches the same
operations through a single-cell ``PolyData``. Two hazards decide every row:

* **The single line cell.** ``pv.lines_from_points`` gives one two-point cell *per segment*, and
  every polyline filter then restarts at each — ``compute_arc_length`` reports orders of magnitude
  short and ``decimate_polyline`` is a **no-op at every reduction**. ``_polyline_pv`` builds one
  cell for that reason; a row built the other way times the right filter on the wrong input and
  reads as a suspiciously fast reference.
* **``pack()`` is mandatory after a decimation, and skipping it is silent.** ``vertsDeleted``
  reports the deletions while ``points.size()`` is unchanged and only ``topology.numValidVerts()``
  reflects them. ``totalLength()`` is already correct before packing, so a *length* comparison
  passes unpacked while a *point-count* one silently reads the input's count and reads as a no-op.
  The ``getNumpyFaces``-without-``pack()`` rule, in a class it does not name.

Three groups stay ordito-only, per function rather than blanket: **``polyline_radius``** (no
reference computes it — ``findCenterFromPoints`` is a centroid and ``findMaxProjectionOnPolyline``
is ``polyline_point_distance``'s question, which already carries its rows); **``polyline_angles``**
(MeshLib has ``edgeVector`` only, so a reference row would time a Python loop); and
``polyline_triangulate``, whose sibling ``triangulate_polygon`` carries trimesh while the clipper
itself carries meshlib and pyvista.

trimesh, libigl and open3d are unregistered, each for its own reason: trimesh models polylines as
``Path3D`` entities and its only simplification is a colinear-run merge; libigl's ``igl.upsample``
is *mesh* subdivision and its C++ ``ramer_douglas_peucker`` is not bound; and open3d's ``LineSet``
stores unordered segments with no ordering, length, resampling or simplification at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
import shapely.geometry as sg
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, BenchLibrary, skip_larger_than

# MeshLib expresses several gates as *absolute* lengths whose defaults assume a unit-scale input
# (``DecimatePolylineSettings.maxError`` is 1e-3). Passing this instead disables the gate outright,
# so a row stops for the reason it is given rather than for the fixture's units.
_UNBOUNDED = 1e30

# Resampling step, as a fraction of the mean segment length: < 1 upsamples, > 1 downsamples.
_UPSAMPLE_FRACTION = 0.5
_DOWNSAMPLE_FRACTION = 4.0

# Ramer-Douglas-Peucker tolerances, as a fraction of the polyline's bounding-box diagonal. A
# tighter tolerance keeps more points and so recurses deeper, which on a single-thread kernel is
# the whole cost; the pair is two orders of magnitude apart so the slope is unambiguous.
_SIMPLIFY_FRACTIONS = [1e-3, 1e-1]

# Query-point counts for polyline_point_distance: the second size in its points x segments product,
# swept independently of the polyline length the axis provides.
_N_QUERIES = [1 << 12, 1 << 16]

_polyline_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_query_cache: dict[tuple[str, str, int], wp.array[wp.vec3]] = {}


def _polyline_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Longest boundary loop of the mesh as a dense ``wp.vec3`` polyline, cached per case."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _polyline_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loops = od.boundary.boundary_loops(vertices, faces)
        if not loops:
            pytest.skip(f"{bench_case.mesh_name} has no boundary loop to use as a polyline")
        longest = max(loops, key=lambda loop: loop.size)
        # Warp's stubs type ``wp.empty``'s dtype as ``float`` and ``wp.copy``'s source as a dense
        # array, though it copies from the ``indexedarray`` a gather returns.
        dense = wp.empty(longest.size, dtype=wp.vec3, device=bench_case.device)  # pyright: ignore[reportArgumentType]
        wp.copy(dense, vertices[longest])  # pyright: ignore[reportArgumentType]
        _polyline_cache[key] = dense
    return _polyline_cache[key]


def _segment_scale(bench_case: BenchCase) -> tuple[float, float]:
    """
    ``(mean_segment_length, bbox_diagonal)`` of the polyline, computed on the host once.

    Through ``_polyline_np`` rather than ``_polyline_wp``, for the reason that helper exists: a
    ``cpu_bound`` reference case has no Warp device, and every resampling row now reads this scale
    on both sides of its branch to give both libraries the same target.
    """
    polyline = _polyline_np(bench_case)
    steps = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    diagonal = float(np.linalg.norm(polyline.max(axis=0) - polyline.min(axis=0)))
    return float(steps.mean()), diagonal


def _query_points_wp(bench_case: BenchCase, count: int) -> wp.array[wp.vec3]:
    """Random query points inside the polyline's bounding box, cached per (case, count)."""
    key = (bench_case.mesh_name, str(bench_case.device), count)
    if key not in _query_cache:
        polyline = _polyline_wp(bench_case).numpy()
        rng = np.random.default_rng(20260726)
        points = rng.uniform(polyline.min(axis=0), polyline.max(axis=0), size=(count, 3))
        _query_cache[key] = wp.array(
            np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
        )
    return _query_cache[key]


_polyline_np_cache: dict[str, np.ndarray] = {}
_query_np_cache: dict[tuple[str, int], np.ndarray] = {}


def _polyline_np(bench_case: BenchCase) -> np.ndarray:
    """
    Build the same polyline as a host array, for the CPU-bound rows.

    ``_polyline_wp`` goes through ``bench_case.vertices_wp``, which needs a Warp device a
    ``cpu_bound`` case does not have -- the situation ``test_points.py``'s pymeshlab cloud is in.
    The extraction runs on the ``cpu`` device instead: it is the benchmark's *input*, so which
    device derives it is immaterial, and the result is cached per mesh.
    """
    if bench_case.mesh_name not in _polyline_np_cache:
        vertices = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.vec3,
            device="cpu",
        )
        faces = wp.array(
            np.ascontiguousarray(bench_case.faces_np.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device="cpu",
        )
        loops = od.boundary.boundary_loops(vertices, faces)
        if not loops:
            pytest.skip(f"{bench_case.mesh_name} has no boundary loop to use as a polyline")
        longest = max(loops, key=lambda loop: loop.size)
        _polyline_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            bench_case.vertices_np[longest.numpy()], dtype=np.float64
        )
    return _polyline_np_cache[bench_case.mesh_name]


def _query_points_np(bench_case: BenchCase, count: int) -> np.ndarray:
    """Draw the same query cloud as [`_query_points_wp`], on the host and at the same seed."""
    key = (bench_case.mesh_name, count)
    if key not in _query_np_cache:
        polyline_np = _polyline_np(bench_case)
        rng = np.random.default_rng(20260726)
        _query_np_cache[key] = rng.uniform(
            polyline_np.min(axis=0), polyline_np.max(axis=0), size=(count, 3)
        )
    return _query_np_cache[key]


_contour_ml_cache: dict[tuple[str, str], mm.std_vector_Vector3_float] = {}
_polyline_ml_cache: dict[tuple[str, str], mm.Polyline3] = {}


def _contour_ml(bench_case: BenchCase) -> mm.std_vector_Vector3_float:
    """
    Build the benchmark's polyline as a MeshLib contour, cached.

    Filling it is a per-point Python loop over ``Vector3f`` constructions -- comparable to the
    reduction it feeds -- and it is the *input*, so it is built once per polyline like every other
    library's copy in this file.
    """
    key = (bench_case.mesh_name, "contour")
    if key not in _contour_ml_cache:
        contour_ml = mm.std_vector_Vector3_float()
        for point_np in _polyline_np(bench_case):
            contour_ml.append(mm.Vector3f(*point_np.tolist()))
        _contour_ml_cache[key] = contour_ml
    return _contour_ml_cache[key]


def _polyline_ml(bench_case: BenchCase) -> mm.Polyline3:
    """
    Build the same points as a ``Polyline3``, through the **constructor**.

    ``addFromPoints`` binds a raw ``Vector3f*`` plus a count rather than a vector, so the
    single-contour constructor is the usable route. Cached: it builds an AABB tree lazily on first
    query, which the row below pre-warms rather than times.
    """
    key = (bench_case.mesh_name, "polyline")
    if key not in _polyline_ml_cache:
        _polyline_ml_cache[key] = mm.Polyline3(_contour_ml(bench_case))
    return _polyline_ml_cache[key]


_polyline_pv_cache: dict[str, pv.PolyData] = {}


def _polyline_pv(bench_case: BenchCase) -> pv.PolyData:
    """
    Build the benchmark's polyline as a ``PolyData`` holding **one** line cell, cached.

    The single cell is not a detail: ``pv.lines_from_points`` gives one two-point cell per segment
    and every polyline filter then restarts at each of them -- ``compute_arc_length`` reports 0.0638
    for a 200-point helix whose length is 12.7049, so a row built that way would time the right
    filter on the wrong input. Cached like the MeshLib ``Polyline3`` beside it, and for the same
    second reason: ``find_closest_cell`` builds a cell locator lazily on first use, which the row
    below pre-warms rather than times.
    """
    if bench_case.mesh_name not in _polyline_pv_cache:
        points_np = _polyline_np(bench_case)
        _polyline_pv_cache[bench_case.mesh_name] = pv.PolyData(
            points_np, lines=np.hstack([[points_np.shape[0]], np.arange(points_np.shape[0])])
        )
    return _polyline_pv_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="polyline_length")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
def test_polyline_length(bench_case: BenchCase) -> None:
    """
    Summed segment length: the cheapest whole-polyline reduction, launch-latency bound.

    meshlib's ``calcLength`` sums the same segments and returns a **bit-identical** float32
    (``tests/test_polyline.py``), so this pair is a pure host-against-device reading of the same
    arithmetic -- and on a reduction this cheap ordito's row is its launch latency, which is what
    makes the comparison worth having. The contour is the input and is cached.

    pyvista's ``compute_arc_length`` **does more**: it writes the *cumulative* length at every
    point and the total is its last entry, where both other rows return the scalar directly. So read
    its row as a per-point pass rather than as a reduction -- and it is still cheap across the axis,
    because VTK walks one line cell.
    """
    if bench_case.kind == "pyvista":
        line_pv = _polyline_pv(bench_case)
        arc_pv = bench_case.run(line_pv.compute_arc_length)
        assert float(np.asarray(arc_pv["arc_length"]).max()) > 0.0
        return
    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        assert bench_case.run(lambda: mm.calcLength(contour_ml)) > 0.0
        return
    polyline = _polyline_wp(bench_case)
    length = bench_case.run(lambda: od.polyline.polyline_length(polyline))
    assert length > 0.0


@pytest.mark.benchmark(group="polyline_normal")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_polyline_normal(bench_case: BenchCase) -> None:
    """
    Newell's loop normal: a ``wp.cross`` accumulated over the closing segments.

    ordito-only, and it is an absence rather than a cost objection -- no registered library
    computes a polyline's Newell normal. meshlib's ``Polyline3`` offers a centroid
    (``findCenterFromPoints``) and a projection but no loop normal, and pyvista's line filters read
    an arc length rather than an orientation.

    **A second row in the whole-polyline-reduction class**, which the module docstring above
    otherwise times through one representative. The rule held while the class really did differ
    "only in the per-segment expression", and it stopped holding: ``polyline_length`` maps
    ``segment_length`` and reduces through ``ordito.reduce``, so it was always a proper block
    reduction, while this one accumulated one ``wp.atomic_add`` per thread into a single
    ``wp.vec3`` slot -- every thread in the launch contending for one address, and the reduction
    serialized. Converting it to the lane-strided form
    (``kernels/polyline.py::accumulate_radius_frame``, which ``polyline_normal`` launches) is worth
    one to two orders of magnitude at the point counts this axis reaches, with the answer three
    orders of magnitude more accurate against a float64 reference. So the representative had the
    *good* shape and the class member it stood in for did not, which is what a one-row class cannot
    show.
    ``polyline_centroid`` stays unrepresented: it reaches ``ordito.reduce`` the way
    ``polyline_length`` does.

    ``rim_long`` is the row that matters here -- 65 536 vertices, the axis's asymptotic point.
    """
    polyline = _polyline_wp(bench_case)
    normal = bench_case.run(lambda: od.polyline.polyline_normal(polyline))
    assert float(np.linalg.norm(np.asarray(list(normal), dtype=np.float64))) > 0.0


@pytest.mark.benchmark(group="polyline_radius")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_polyline_radius(bench_case: BenchCase) -> None:
    """
    Per-segment plane projection and closest-point search, then a reduction.

    ordito-only, and it is an absence rather than a cost objection: no registered library computes
    a polyline's radius about a centre and normal. ``Polyline3.findCenterFromPoints`` is a centroid
    and ``findMaxProjectionOnPolyline`` projects points *onto* a polyline, which is
    ``polyline_point_distance``'s question and already carries its rows. See the module docstring.
    """
    polyline = _polyline_wp(bench_case)
    radius = bench_case.run(lambda: od.polyline.polyline_radius(polyline, reduction="min"))
    assert radius >= 0.0


@pytest.mark.benchmark(group="polyline_angles")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_polyline_angles(bench_case: BenchCase) -> None:
    """
    Per-vertex turning angle: the ``wp.acos`` path, one angle per point.

    ordito-only. MeshLib has ``edgeVector`` and nothing above it, so a reference row would time a
    Python loop over the segments rather than MeshLib -- the per-element rule. See the module
    docstring.
    """
    polyline = _polyline_wp(bench_case)
    angles = bench_case.run(lambda: od.polyline.polyline_angles(polyline))
    assert angles.size == polyline.size


def _polyline_cpu(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """
    Build the benchmark's polyline as a ``wp.vec3`` array on the **cpu** device, cached per mesh.

    The reference rows for ``polyline_downsample`` and ``polyline_simplify`` are driven by the
    *count* ordito reaches rather than by their own error parameter (each row says why), so they
    have to call ordito once to learn it. That call is outside the timed callable and is not the
    measurement, so it runs on the cpu -- a ``cpu_bound`` case has no Warp device of its own.
    """
    key = (bench_case.mesh_name, "cpu-polyline")
    if key not in _polyline_cache:
        _polyline_cache[key] = wp.array(
            np.ascontiguousarray(_polyline_np(bench_case)), dtype=wp.vec3, device="cpu"
        )
    return _polyline_cache[key]


@pytest.mark.benchmark(group="polyline_upsample")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_upsample_polyline(bench_case: BenchCase) -> None:
    """
    Arc-length upsampling at half the mean segment length: scan, readback, then a lerp pass.

    meshlib's ``subdividePolyline`` takes the same ``maxEdgeLen`` and makes the same guarantee, but
    it **bisects** where ordito splits each segment into equal pieces, so at a target that is not
    a power-of-two fraction of the input spacing it overshoots -- roughly doubling the point count,
    because one halving leaves it a hair over the cap and forces a second. Both satisfy the cap;
    read the row as a cost at a *shared
    post-condition* rather than at a shared output size (``tests/test_polyline.py``).

    It mutates, so the ``Polyline3`` is rebuilt inside the timed callable, and ``maxEdgeSplits`` is
    raised from its default of **1 000** -- the same trap ``SubdivideSettings`` has, and on these
    axes it would stop the reference in the first few percent of the work.
    """
    step = _UPSAMPLE_FRACTION * _segment_scale(bench_case)[0]
    n_points = _polyline_np(bench_case).shape[0]

    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)

        def upsample_ml() -> int:
            polyline_ml = mm.Polyline3(contour_ml)
            settings_ml = mm.PolylineSubdivideSettings()
            settings_ml.maxEdgeLen = step
            settings_ml.maxEdgeSplits = 10_000_000
            mm.subdividePolyline(polyline_ml, settings_ml)
            return polyline_ml.topology.numValidVerts()

        assert bench_case.run(upsample_ml) >= n_points
        return

    polyline = _polyline_wp(bench_case)
    dense = bench_case.run(lambda: od.polyline.polyline_upsample(polyline, step))
    assert dense.size >= polyline.size


@pytest.mark.benchmark(group="polyline_downsample")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_downsample_polyline(bench_case: BenchCase) -> None:
    """
    Arc-length downsampling at four times the mean segment length.

    meshlib's ``decimatePolyline`` reaches a *count* rather than a step, so its row is given the
    count ordito's step produces (``maxDeletedVertices``) and its ``maxError`` is opened up so the
    count is what binds -- otherwise the two rows would stop for different reasons and the ratio
    would mean nothing. That makes this the module's clearest structural contrast: an arc-length
    resample is one scan and one gather where a decimator is a priority queue of collapses.

    ``optimizeVertexPos`` is turned **off**. It defaults on and moves each surviving vertex to a
    fitted position, which is work ordito does not do and which would additionally leave the output
    off the input point set.
    """
    step = _DOWNSAMPLE_FRACTION * _segment_scale(bench_case)[0]

    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        cpu_polyline = _polyline_cpu(bench_case)
        n_points = cpu_polyline.size
        n_kept = od.polyline.polyline_downsample(cpu_polyline, step).size

        def downsample_ml() -> int:
            polyline_ml = mm.Polyline3(contour_ml)
            settings_ml = mm.DecimatePolylineSettings_Vector3f()
            settings_ml.maxDeletedVertices = max(n_points - n_kept, 0)
            settings_ml.maxError = _UNBOUNDED
            settings_ml.optimizeVertexPos = False
            return int(mm.decimatePolyline(polyline_ml, settings_ml).vertsDeleted)

        assert bench_case.run(downsample_ml) >= 0
        return

    polyline = _polyline_wp(bench_case)
    sparse = bench_case.run(lambda: od.polyline.polyline_downsample(polyline, step))
    assert sparse.size >= 2


@pytest.mark.benchmark(group="polyline_simplify")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
@pytest.mark.parametrize("tolerance_fraction", _SIMPLIFY_FRACTIONS)
def test_simplify_polyline(bench_case: BenchCase, tolerance_fraction: float) -> None:
    """
    Ramer-Douglas-Peucker, one round of four ``dim=n`` launches per level of the split tree.

    Warp forbidding recursion does *not* make the split serial: a level-synchronous evaluation
    accepts the same points and is nearly two orders of magnitude ahead of a stack kernel on the
    longest rim. Read the two ``saddle`` rows as floor rows, the graph capture being a large share
    of a sub-millisecond call, rather than as outliers.

    Neither reference is Ramer-Douglas-Peucker, and **neither is driven by ordito's tolerance**,
    which is the thing to know before reading the ratio: both are given the *reduction* ordito's
    tolerance produces, so the rows price three ways of removing the same number of points.

    Driving them by their own error parameter was rejected on a measurement.
    ``decimatePolyline``'s ``maxError`` is a collapse cost, **not** a deviation bound: on a random
    walk its output sits several times further from the input than the tolerance it was given, where
    ordito's and pyvista's stay well inside it. So a tolerance-matched pair would be two different
    amounts of work under one parameter name.

    * **meshlib** ``decimatePolyline`` at ``maxDeletedVertices`` = ordito's deletion count, with
      ``maxError`` opened up so the count is what binds. ``optimizeVertexPos`` is turned off for the
      reason the ``downsample`` row records; it mutates, so the ``Polyline3`` is rebuilt per round.
    * **pyvista** ``decimate_polyline`` takes a reduction *fraction*, handed the same count. It
      needs the **single-cell** ``PolyData`` ``_polyline_pv`` builds: on a ``lines_from_points``
      polyline (one two-point cell per segment) it is a no-op at every reduction, which is the
      silent-wrong-input hazard that builder exists for.
    """
    tol = tolerance_fraction * _segment_scale(bench_case)[1]

    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        cpu_polyline = _polyline_cpu(bench_case)
        n_points = cpu_polyline.size
        n_kept = od.polyline.polyline_simplify(cpu_polyline, tol)[0].size

        def simplify_ml() -> int:
            polyline_ml = mm.Polyline3(contour_ml)
            settings_ml = mm.DecimatePolylineSettings_Vector3f()
            settings_ml.maxDeletedVertices = max(n_points - n_kept, 0)
            settings_ml.maxError = _UNBOUNDED
            settings_ml.optimizeVertexPos = False
            return int(mm.decimatePolyline(polyline_ml, settings_ml).vertsDeleted)

        assert bench_case.run(simplify_ml) >= 0
        return
    if bench_case.kind == "pyvista":
        line_pv = _polyline_pv(bench_case)
        cpu_polyline = _polyline_cpu(bench_case)
        n_points = cpu_polyline.size
        n_kept = od.polyline.polyline_simplify(cpu_polyline, tol)[0].size
        reduction = min(max(1.0 - n_kept / n_points, 0.0), 0.999)
        assert bench_case.run(lambda: line_pv.decimate_polyline(reduction)).n_points >= 2
        return

    polyline = _polyline_wp(bench_case)
    simplified, kept = bench_case.run(lambda: od.polyline.polyline_simplify(polyline, tol))
    assert simplified.size == kept.size


@pytest.mark.benchmark(group="polyline_point_distance")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
@pytest.mark.parametrize("n_queries", _N_QUERIES)
def test_distance_to_polyline(bench_case: BenchCase, n_queries: int) -> None:
    """
    Brute-force point-to-segment distance: the one case whose cost is points x segments.

    That is the contrast meshlib's row is here for: ``findProjectionOnPolyline`` walks an **AABB
    tree** over the segments, so its cost is ``points x log(segments)`` where ordito's is the full
    product -- read the gap across the ``polyline`` axis rather than at one point. It has no
    batched form, so the row loops in Python and prices that loop along with the queries; the tree
    is built lazily and is pre-warmed outside the timed callable.

    pyvista's ``find_closest_cell`` is the same query through a ``vtkStaticCellLocator`` and,
    unlike MeshLib's, it is **batched** -- so its row is the honest tree-walk comparison and the
    MeshLib one is a Python loop next to it. It reports the closest *point*, not the distance, so
    the row's output is one array wider than what it is timed against; the distance is a host
    subtraction and is left out deliberately.

    **rim_long is skipped for pyvista**: its locator degrades catastrophically on a
    65 536-segment single cell, by orders of magnitude against a short loop. That is the shape of
    the axis this group exists to show, and one row of it would cost more than
    the rest of the module put together.
    """
    if bench_case.kind == "pyvista":
        if bench_case.mesh_name == "rim_long":
            pytest.skip("VTK's line locator is 104 s at this loop length; capped at saddle")
        line_pv = _polyline_pv(bench_case)
        queries_np = _query_points_np(bench_case, n_queries)
        line_pv.find_closest_cell(queries_np[:1], return_closest_point=True)  # pre-warm
        _cells_pv, closest_pv = bench_case.run(
            lambda: line_pv.find_closest_cell(queries_np, return_closest_point=True)
        )
        assert np.asarray(closest_pv).shape == (n_queries, 3)
        return
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny", "the query is a per-point Python loop")
        polyline_ml = _polyline_ml(bench_case)
        queries_np = _query_points_np(bench_case, n_queries)
        mm.findProjectionOnPolyline(mm.Vector3f(*queries_np[0].tolist()), polyline_ml)  # pre-warm

        def distances_ml() -> float:
            return sum(
                mm.findProjectionOnPolyline(mm.Vector3f(*point_np.tolist()), polyline_ml).distSq
                for point_np in queries_np
            )

        assert bench_case.run(distances_ml) >= 0.0
        return
    polyline = _polyline_wp(bench_case)
    points = _query_points_wp(bench_case, n_queries)
    distance = bench_case.run(lambda: od.polyline.polyline_point_distance(points, polyline))
    assert distance.size == n_queries


# The ``closed=True`` rows. A boundary loop does not repeat its first vertex, so every one of these
# takes the closing segment the flag adds -- the path that used to go through ``polyline_close``'s
# closure readback and full-buffer copy. ordito-only: what they price is the closure handling
# around an algorithm the open rows above already compare against the references, so a reference
# row here would re-time the same comparison with one more segment.


@pytest.mark.benchmark(group="polyline_upsample_closed")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_upsample_closed_polyline(bench_case: BenchCase) -> None:
    """``polyline_upsample(closed=True)`` at the open row's step: one more segment, as a ring."""
    step = _UPSAMPLE_FRACTION * _segment_scale(bench_case)[0]
    polyline = _polyline_wp(bench_case)
    dense = bench_case.run(lambda: od.polyline.polyline_upsample(polyline, step, closed=True))
    assert dense.size >= polyline.size


@pytest.mark.benchmark(group="polyline_downsample_closed")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_downsample_closed_polyline(bench_case: BenchCase) -> None:
    """``polyline_downsample(closed=True)`` at the open row's step, the closing edge counted."""
    step = _DOWNSAMPLE_FRACTION * _segment_scale(bench_case)[0]
    polyline = _polyline_wp(bench_case)
    sparse = bench_case.run(lambda: od.polyline.polyline_downsample(polyline, step, closed=True))
    assert sparse.size >= 2


@pytest.mark.benchmark(group="polyline_resample_closed")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_resample_closed_polyline(bench_case: BenchCase) -> None:
    """
    ``polyline_resample(closed=True)`` to the input's own point count.

    A cumulative arc length and one interpolation pass, with no data-dependent output size and so
    no readback of its own.
    """
    polyline = _polyline_wp(bench_case)
    n_points = polyline.size
    ring = bench_case.run(lambda: od.polyline.polyline_resample(polyline, n_points, closed=True))
    assert ring.size == n_points


@pytest.mark.benchmark(group="polyline_point_distance_closed")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("ordito")
def test_distance_to_closed_polyline(bench_case: BenchCase) -> None:
    """
    ``polyline_point_distance(closed=True)`` at the smaller query count.

    That is where the closure handling is the largest share of the call; the larger count is the
    open row's points x segments product again.
    """
    polyline = _polyline_wp(bench_case)
    points = _query_points_wp(bench_case, _N_QUERIES[0])
    distance = bench_case.run(
        lambda: od.polyline.polyline_point_distance(points, polyline, closed=True)
    )
    assert distance.size == _N_QUERIES[0]


# Vertex counts for the triangulation group. A simple polygon of ``n`` vertices always yields
# ``n - 2`` triangles, so the count is the only axis and it is swept directly rather than through a
# mesh: neither library's ear clipping is driven by anything else.
_POLYGON_SIZES = [64, 1024]

_polygon_np_cache: dict[int, np.ndarray] = {}


def _polygon_np(n_vertices: int) -> np.ndarray:
    """Build a non-convex star of ``n_vertices``, cached: a fan would trivialize a convex one."""
    if n_vertices not in _polygon_np_cache:
        angles_np = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
        radii_np = np.where(np.arange(n_vertices) % 2 == 0, 1.0, 0.45)
        _polygon_np_cache[n_vertices] = np.column_stack(
            [radii_np * np.cos(angles_np), radii_np * np.sin(angles_np)]
        )
    return _polygon_np_cache[n_vertices]


@pytest.mark.benchmark(group="polyline_triangulate")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
@pytest.mark.parametrize("n_vertices", _POLYGON_SIZES)
def test_triangulate_polyline(bench_case: BenchCase, n_vertices: int) -> None:
    """
    Ear clipping a simple polygon: the one group here whose input is not the mesh's boundary.

    ``benchmeshes("sphere_small")`` pins it to a single case because the mesh is irrelevant -- the
    polygon is generated from ``n_vertices`` alone -- and the sweep is that count, which is the only
    thing either implementation's cost depends on. A **star** rather than a convex ring: a fan over
    vertex 0 triangulates any convex polygon in linear time and would make both rows measure
    nothing.

    meshlib's ``triangulateContours`` takes 2-D **closed** contours (the first point repeated) and
    returns a whole ``Mesh``, so its row carries that construction where ordito's returns an index
    buffer -- it is doing more, and the two agree on the triangle count and total area
    (``tests/test_polyline.py``). Both build their input contour outside the timed callable.

    pyvista's ``triangulate_contours`` is VTK's ear clipper over the same closed loop, and it too
    returns a whole ``PolyData`` -- but of the loop's **own** points: it introduces zero Steiner
    points, so the count is ``n - 2`` on all three sides. Its line cell must carry the repeated
    first index, the same closing convention MeshLib's contour needs.
    """
    polygon_np = _polygon_np(n_vertices)
    if bench_case.kind == "pyvista":
        indices_np = np.append(np.arange(n_vertices), 0)  # closed: the repeat is required
        loop_pv = pv.PolyData(
            np.column_stack([polygon_np, np.zeros(n_vertices)]),
            lines=np.hstack([[indices_np.size], indices_np]),
        )
        filled_pv = bench_case.run(loop_pv.triangulate_contours)
        assert filled_pv.n_cells == n_vertices - 2
        return
    if bench_case.kind == "meshlib":
        contour_ml = mm.std_vector_Vector2_float()
        for point_np in np.vstack([polygon_np, polygon_np[:1]]):
            contour_ml.append(mm.Vector2f(float(point_np[0]), float(point_np[1])))
        contours_ml = mm.std_vector_std_vector_Vector2_float()
        contours_ml.append(contour_ml)
        mesh_ml = bench_case.run(lambda: mm.triangulateContours(contours_ml))
        assert mesh_ml.topology.numValidFaces() == n_vertices - 2
        return
    points_wp = wp.array(
        np.ascontiguousarray(np.column_stack([polygon_np, np.zeros(n_vertices)]), dtype=np.float32),
        dtype=wp.vec3,
        device=bench_case.device,
    )
    faces = bench_case.run(lambda: od.polyline.polyline_triangulate(points_wp))
    assert int(faces.shape[0]) == n_vertices - 2


_STAR_RINGS = [64, 1024]


def _star_np(n: int) -> np.ndarray:
    """Non-convex star ring: alternating radii, so ear clipping cannot take the single-fan path."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    radius_np = np.where(np.arange(n) % 2 == 0, 1.0, 0.45)
    return np.column_stack((radius_np * np.cos(angle_np), radius_np * np.sin(angle_np)))


def _star_wp(n: int, device: str) -> wp.array[wp.vec2]:
    return wp.array(
        np.ascontiguousarray(_star_np(n), dtype=np.float32), dtype=wp.vec2, device=device
    )


@pytest.mark.benchmark(group="triangulate_polygon")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("ring_size", _STAR_RINGS)
def test_triangulate_polygon(bench_lib: BenchLibrary, ring_size: int) -> None:
    """
    Ear clipping on a *non-convex* ring: the only group here that reaches the clipper at all.

    ``creation.extrude_polygon`` hands the triangulator a convex ring, which takes the single-fan
    fast path and never reaches the ear loop (its row is in [`test_creation.py`](test_creation.py)).
    A star ring forces it. Both points take the single-block ear loop, whose cost is **how many
    rounds the independent-set rule needs** times three block barriers plus each round's ear tests:
    a few dozen rounds either way, against a count proportional to the ring size before the ranking
    hash. The small point is close to the floor of one launch and two readbacks. No open3d
    counterpart.
    """
    if bench_lib.kind == "ordito":
        ring_wp = _star_wp(ring_size, str(bench_lib.device))
        _vertices, faces_wp = bench_lib.run(lambda: od.polyline.triangulate_polygon(ring_wp))
        assert faces_wp.size // 3 == ring_size - 2
    else:
        polygon = sg.Polygon(_star_np(ring_size))
        _vertices, faces_tm = bench_lib.run(lambda: tm.creation.triangulate_polygon(polygon))
        assert faces_tm.shape[0] > 0
