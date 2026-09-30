"""
Benchmarks for ``triwarp.holes``.

Axis: **loops_dp** -- ``rim_short`` (two loops of 512) against ``holes_many`` (512 loops of 3).
The module has the highest non-``N`` sensitivity in the package and this pair separates its two
independent drivers:

* **Loop length**, cubed. ``fill_min_weight`` runs a minimum-weight triangulation DP over a
  ``B x B`` table per loop, filled by ``B - 2`` *sequential* steps, then traced back one thread per
  loop without the table ever leaving the device. Total work is ``sum(B_i^3)`` and no batching can
  remove it: this is the real cost the module exists to pay.

  The *steps* are a dependency chain and their count is a floor, but they no longer have to be
  issued launches. At ``rim_short`` the sweep's host launch cost had come to within ~20% of its
  device time, overlapping it, which bounded what removing launches could ever buy -- and the
  sweep now spends that bound: one group of spans is recorded and replayed, taking this row 1.36x
  and leaving ``holes_many`` flat, with the reference columns unmoved across the two sessions as
  the control. Read the remaining "launch-bound" phrasing in this module as a claim about the
  *many-rim* end, where the sweep is one launch and everything else is fixed cost.

  Most of what *can* be removed already was. Widening the per-span launch from one thread per
  interval to one *block* per interval, its lanes striding the apex loop, is worth several-fold on
  the long rims: the grid is ``n_loops * (max_B - span)`` wide, so two long rims put ~1 024 threads
  on the machine and the whole ``B^3`` term ran at a fraction of a percent of it. Same launch
  count, same DP, byte-identical triangulation (see ``fill_dp_span_tiled``).
* **Loop count**, which should cost nothing and does so only because the DP is batched across
  loops. Per loop it would be a ``.numpy()`` readback, a forbidden-chord pass over the whole mesh
  and its own span launches, so many three-vertex holes -- where the DP itself is one triangle per
  hole -- cost *more* than the genuinely expensive long-rim point. Batching also runs
  ``rim_short``'s two rims concurrently. The last per-loop host term was the traceback, a Python
  stack walk per rim over a read-back predecessor table; on ``holes_many`` it alone was 22% of the
  call, and moving it onto the device took the whole group 1.21x.

That is why the axis meshes are *small*: face count is not the variable, and sizing these meshes up
would only add DP table entries that the ``B^3`` term already dominates. The two points differ by
several-fold in the right direction, which is what the axis is for -- an *inversion* there is the
tell that a per-loop sequence has crept back in. ``fill_fan`` runs on the wider **loops** axis
instead, because it
has no DP and so can afford ``rim_long``'s much longer rims -- it is the floor this module's cost is
measured against.

This group's median moves substantially between runs of the *same* code depending on which
reference rows share the process, so read ratios rather than absolutes here.

One cost the ``loops_dp`` pair cannot see is the default subdivision-target derivation inside
``fill_smooth``: its axis is the raw loop *count*, and 512 loops is too few for it to register
against the DP. ``fill_smooth_target_edge`` runs that one path on the **loops_dense** axis
(512 -> 8 192 loops on the same vertex buffer) instead.

References
----------
**trimesh**'s ``repair.fill_holes`` is a much weaker algorithm (it fans triangles across small
holes and gives up on large ones), so it is timing context rather than an equivalent -- the
assertion only checks that faces were added or preserved.

**open3d**'s ``fill_holes`` lives on the newer tensor API (``open3d.t.geometry.TriangleMesh``) and
wraps a hole-filling pass over the boundary loops, which is the closest analogue to triwarp's
minimum-weight triangulation. It returns a new tensor mesh, but ``from_legacy`` is a full
conversion, so that conversion is hoisted out of the timed callable and only ``fill_holes`` is
measured.

**pymeshlab**'s ``meshing_close_holes`` is a third algorithm again -- an ear-clipping fill with an
optional self-intersection check (``selfintersection=True`` by default, left on) rather than a
minimum-weight DP. One parameter matters and it is a trap: **``maxholesize`` is an edge count with a
default of 30**, so on the long-rim axis point the filter closes *nothing* and returns immediately
with ``{'closed_holes': 0}``. Lifting it is what makes the axis points comparable at all; the dict
it returns is what makes that checkable, and the assertion below reads it.

Note what the axis then says: the reference's two axis points are close together where triwarp's
differ by more than an order of magnitude. That is the ``B^3`` term -- MeshLab's ear clipping is
quadratic at worst, so it does not pay it, and its rows are the honest price of *not* computing a
minimum-weight triangulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, face_bitset_ml, mesh_ml_from_numpy

if TYPE_CHECKING:
    import open3d as o3d

# The DP runs to hundreds of milliseconds a call on both axis points.
_ROUNDS = 3

# Open shells per ``open_components`` mesh. Read from the registry name rather than counted per
# round: the count is what the joins are asserted against, so deriving it from the mesh under test
# would let a driver that joined nothing still pass.
_OPEN_COMPONENT_COUNTS = {"open_parts_4": 4, "open_parts_16": 16, "open_parts_64": 64}

_tensor_mesh_cache: dict[str, o3d.t.geometry.TriangleMesh] = {}


def _tensor_mesh_o3d(bench_case: BenchCase) -> o3d.t.geometry.TriangleMesh:
    """Convert to the tensor API once per mesh for ``fill_holes`` (conversion is not the op)."""
    if bench_case.mesh_name not in _tensor_mesh_cache:
        import open3d as o3d

        _tensor_mesh_cache[bench_case.mesh_name] = o3d.t.geometry.TriangleMesh.from_legacy(
            bench_case.mesh_o3d
        )
    return _tensor_mesh_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="fill_fan")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp")
def test_fill_fan(bench_case: BenchCase) -> None:
    """The cheapest filler -- one fan per loop, no DP -- so it can take the long-rim axis."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(lambda: tw.holes.fill_fan(vertices, faces))
    assert result.shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="fill_cone")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_fill_cone(bench_case: BenchCase) -> None:
    """
    One centroid vertex and one triangle per boundary edge -- ``fill_fan`` plus an apex.

    Shares ``fill_fan``'s wide ``loops`` axis for the same reason: no DP, so it can afford
    ``rim_long``'s 65 536-vertex rims. meshlib's ``fillHoleTrivially`` is the same operation
    (pinned in tests/test_holes.py::test_fill_cone_matches_meshlib) but fills **one hole per
    call**, so its row includes the Python loop over ``findHoleRepresentiveEdges`` -- which is the
    honest cost of asking MeshLib for the same answer, and is why the ``holes_many`` point is the
    one to read for per-loop overhead rather than the long-rim point. Rebuilt per round because it
    mutates.
    """
    if bench_case.kind == "meshlib":

        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
                mm.fillHoleTrivially(mesh_ml, edge_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) >= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    new_vertices, new_faces = bench_case.run(
        lambda: tw.holes.fill_cone(vertices, faces), rounds=_ROUNDS
    )
    assert new_faces.shape[0] >= faces.shape[0]
    assert new_vertices.shape[0] >= vertices.shape[0]


@pytest.mark.noparity(
    "trimesh",
    reason="D2 a weaker algorithm for the same task: tm.repair.fill_holes fans triangles across "
    "small holes and gives up on large ones, where fill_min_weight runs the minimum-weight "
    "interval DP, so the two produce different triangulations by design and trimesh has no "
    "minimum-weight answer to compare against. meshlib is the oracle for the DP itself, in "
    "tests/test_holes.py::test_fill_min_weight_matches_meshlib.",
)
@pytest.mark.benchmark(group="fill_min_weight")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib")
def test_fill_min_weight(bench_case: BenchCase) -> None:
    """The ``B^3`` DP: few long loops against many short ones, at a comparable total boundary."""
    if bench_case.kind == "meshlib":
        # The only row here running the *same* algorithm -- a minimum-weight Liepa/Klincsek DP over
        # the same ``plane_normalized`` metric, which is why it is the oracle in
        # tests/test_holes.py::test_fill_min_weight_matches_meshlib and the other three references
        # are timing context. Two settings are load-bearing: ``maxPolygonSubdivisions`` is raised
        # so MeshLib runs the exhaustive search rather than sub-sampling a long rim (the same
        # 1 000 the parity test passes), and the mesh is rebuilt inside the timed callable because
        # ``fillHoles`` mutates it and rounds 2..n would find nothing left to fill.
        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            edges_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            params_ml = mm.FillHoleParams()
            params_ml.maxPolygonSubdivisions = 1000
            params_ml.metric = mm.getPlaneNormalizedFillMetric(mesh_ml, edges_ml[0])
            mm.fillHoles(mesh_ml, edges_ml, params_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > bench_case.n_faces
        return
    if bench_case.kind == "pymeshlab":
        # ``maxholesize`` is an *edge count* cap, and its default of 30 would silently close nothing
        # on ``rim_short``'s two 512-edge rims, returning immediately. Lifting it is what makes the
        # two axis points comparable at all, and the returned dict is asserted on so a future
        # default change cannot quietly turn this row back into a no-op.
        statistics_pml = bench_case.run(
            lambda: bench_case.new_meshset_pml().meshing_close_holes(maxholesize=1_000_000),
            rounds=_ROUNDS,
        )
        assert statistics_pml["closed_holes"] > 0
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.holes.fill_min_weight(vertices, faces), rounds=_ROUNDS)
        assert result.shape[0] >= faces.shape[0]
    elif bench_case.kind == "trimesh":  # mutates in place: rebuild inside the timed callable
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.repair.fill_holes(mesh)
            return mesh

        result = bench_case.run(run, rounds=_ROUNDS)
        assert result.faces.shape[0] >= faces.shape[0]
    else:  # open3d tensor API: fill_holes returns a new mesh, so the converted input is reusable
        mesh_t = _tensor_mesh_o3d(bench_case)
        n_faces = bench_case.n_faces
        filled = bench_case.run(mesh_t.fill_holes, rounds=_ROUNDS)
        assert int(filled.triangle["indices"].shape[0]) >= n_faces


@pytest.mark.benchmark(group="fill_min_weight_chords")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("resolve_multiple_edges", [False, True], ids=["plain", "chords"])
def test_fill_min_weight_chords(bench_case: BenchCase, resolve_multiple_edges: bool) -> None:
    """
    What the forbidden-chord pass costs on top of the DP.

    ``resolve_multiple_edges=True`` builds a ``B x B`` table of chords that already exist as mesh
    edges and bans them from the triangulation, which is a second quadratic pass per loop. Whether
    that is a rounding error next to the cubic DP or a real fraction of it is only visible against
    the ``plain`` row.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.holes.fill_min_weight(
            vertices, faces, resolve_multiple_edges=resolve_multiple_edges
        ),
        rounds=_ROUNDS,
    )
    assert result.shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="fill_smooth")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("triangulate_only", [True, False], ids=["dp_only", "refined"])
def test_fill_smooth(bench_case: BenchCase, triangulate_only: bool) -> None:
    """
    What refinement and smoothing cost on top of the DP that produced the patch.

    The ``dp_only`` row is ``fill_min_weight`` plus the patch mask, so the gap to ``refined`` is the
    whole refine-and-smooth stage. Parametrizing rather than timing only the full call is what makes
    that stage attributable: on the ``loops_dp`` axis the cubic DP dominates ``rim_short``, so a
    single ``refined`` number cannot say whether a change moved the DP or the smoothing.

    ``max_edge`` is deliberately left at ``None`` so the default target-edge derivation is inside
    the timed callable -- it is the one part of this path whose cost scales with the *mesh* rather
    than with the rims, and leaving it out would hide a regression there.
    """
    if bench_case.kind == "meshlib":
        if triangulate_only:
            pytest.skip("fillHoleNicely has no triangulate-only mode; the refined row is the pair")

        # ``fillHoleNicely`` is the same three stages triwarp's ``fill_smooth`` runs -- DP fill,
        # subdivide the patch, smooth it -- and it is the oracle in
        # tests/test_holes.py::test_fill_smooth_statistics_vs_meshlib, where the comparison is the
        # enclosed volume because the two patches share no vertices. ``maxEdgeLen`` is left at its
        # own default rather than fed from triwarp's derived target, so the two rows refine to
        # different densities: read this as the cost of the *stage*, not as a like-for-like
        # subdivision. Rebuilt per round because it mutates.
        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            settings_ml = mm.FillHoleNicelySettings()
            for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
                mm.fillHoleNicely(mesh_ml, edge_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.holes.fill_smooth(vertices, faces, triangulate_only=triangulate_only),
        rounds=_ROUNDS,
    )
    assert result[1].shape[0] >= faces.shape[0]


_region_cache: dict[tuple[str, str], tuple] = {}


def _cap_region(bench_case: BenchCase) -> tuple:
    """
    Build a contiguous face region: the cap above the mesh's 80th height percentile.

    Contiguity is what makes this a region edit rather than a hole-filling benchmark: a scattered
    mask opens one rim per face, and the DP is cubic in the rim length, so the two shapes are not
    the same measurement at all.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _region_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        height_np = vertices_np[faces_np].mean(axis=1)[:, 2]
        mask_np = height_np > np.quantile(height_np, 0.8)
        _region_cache[key] = (wp.array(mask_np, dtype=wp.bool, device=bench_case.device), mask_np)
    return _region_cache[key]


@pytest.mark.benchmark(group="refill_region")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("triangulate_only", [True, False], ids=["dp_only", "refined"])
def test_refill_region(bench_case: BenchCase, triangulate_only: bool) -> None:
    """
    Delete a contiguous region and rebuild it: the deletion, the DP, and optionally the refinement.

    Read against ``fill_smooth`` above, whose two rows are the same two stages over the mesh's
    *existing* holes: the difference is the region extraction and the rim classification, which is
    ``delete_region_keep_boundary``'s own group in ``test_selection.py``. Parametrized the same way
    for the same reason -- a single refined number cannot say whether a change moved the DP or the
    smoothing.

    meshlib's ``patchMesh`` is exactly this call and ``tests/test_holes.py`` pins the two to the
    same vertex count, face count and volume in ``triangulateOnly`` mode. It mutates, so it gets a
    fresh mesh per round, and its region bitset is built in ``setup`` -- it is the input.

    Both sides are dominated by the rim DP and the refinement here, which is why this row is close
    to meshlib where ``delete_region_keep_boundary``'s is an order of magnitude behind -- that group
    isolates the extraction, and the extraction is the part triwarp does slowly.

    **The launch count grows with the longest rim, not per rim and not per component.** The DP's
    count is ``2 * (max_rim - 2)`` -- one launch per triangulation span across *all* loops
    (``holes._run_hole_dp`` batches them) and a second sweep for the ``min_area`` retry. The two
    smoothing rows are flat in the launch count because their systems are the patch's, not the
    mesh's.

    **A capture is refuted for all three stages that could take one, each for its own reason.** The
    DP sweep is device-bound at a large rim; ``smooth_region``'s solve is *already* one captured CG
    graph (nearly all of that stage, and ``"auto"`` correctly declines the V-cycle for it); and the
    flip and subdivision loops have data-dependent trip counts with a host readback each round and
    run **once** per call, which is the record-and-replay-once *loss* rather than the replay win.
    """
    if bench_case.kind == "meshlib":
        _mask_wp, mask_np = _cap_region(bench_case)
        settings_ml = mm.FillHoleNicelySettings()
        settings_ml.triangulateOnly = triangulate_only

        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            region_ml = face_bitset_ml(mask_np)
            mm.patchMesh(mesh_ml, region_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > 0
        return

    mask_wp, _mask_np = _cap_region(bench_case)
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _out_vertices, out_faces = bench_case.run(
        lambda: tw.holes.refill_region(vertices, faces, mask_wp, triangulate_only=triangulate_only),
        rounds=_ROUNDS,
    )
    assert int(out_faces.shape[0]) > 0


@pytest.mark.benchmark(group="fill_smooth_target_edge")
@pytest.mark.benchaxis("loops_dense")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("derive_target", [True, False], ids=["derived", "explicit"])
def test_fill_smooth_target_edge(bench_case: BenchCase, derive_target: bool) -> None:
    """
    What deriving the default subdivision target costs as the loop count grows.

    ``fill_smooth(max_edge=None)`` measures every rim to pick its target edge, and that
    measurement's axis is the loop *count* -- not the rim length, and not the mesh size -- which
    the ``loops_dp`` meshes cap at 512, too few to register against the DP. The ``loops_dense``
    axis holds the vertex buffer fixed and multiplies the loop count by 16, and the gap between
    the ``derived`` and ``explicit`` rows is exactly the derivation. Interleaved A/B at the top of
    the axis: within noise, against hundreds of milliseconds of per-loop ``.numpy()`` readbacks in
    the form it replaced -- the regression this group exists to catch,
    and one that lands in the *min* (deterministic host cost), where this box's occasional 3x
    clock-state excursions do not. ``explicit`` passes the mesh's mean edge length, which on a
    punched sphere is the value the derivation returns anyway, so the refine stage does identical
    work in both rows.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    max_edge = None if derive_target else bench_case.mean_edge
    result = bench_case.run(
        lambda: tw.holes.fill_smooth(vertices, faces, max_edge=max_edge), rounds=_ROUNDS
    )
    assert result[1].shape[0] >= faces.shape[0]


# --- Stitching two rims: the same DP over a band rather than a cap -----------------------------

_stitch_cache: dict[tuple[str, str], tuple] = {}
_stitch_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _face_slice(bench_case: BenchCase, lo: int, hi: int) -> wp.array[wp.int32]:
    """Contiguous face-index range as a device buffer (untimed setup, so numpy is fine)."""
    return wp.array(np.arange(lo, hi, dtype=np.int32), dtype=wp.int32, device=bench_case.device)


def _submesh(bench_case: BenchCase, lo: int, hi: int) -> tuple:
    """Faces ``[lo, hi)`` of the case mesh as a compact standalone ``(vertices, faces)`` pair."""
    return tw.selection.submesh_from_face_indices(
        bench_case.vertices_wp, bench_case.faces_wp, _face_slice(bench_case, lo, hi)
    )


def _stitch_halves(bench_case: BenchCase) -> tuple:
    """
    Cut the open tube into two rings, so each half has exactly one boundary loop to stitch.

    ``_open_cylinder`` emits its lower band of triangles before its upper one, so the first and
    second halves of the face buffer are exactly the two rings -- no adjacency query needed.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _stitch_cache:
        half = bench_case.n_faces // 2
        _stitch_cache[key] = (
            _submesh(bench_case, 0, half),
            _submesh(bench_case, half, bench_case.n_faces),
        )
    return _stitch_cache[key]


@pytest.mark.benchmark(group="stitch")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp")
def test_stitch(bench_case: BenchCase) -> None:
    """Greedy band between two rims: O(La + Lb), the cheap counterpart of the DP below."""
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    _vertices, faces = bench_case.run(lambda: tw.holes.stitch(va, fa, vb, fb))
    assert int(faces.shape[0]) > 0


def _stitch_pair_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the same two rings as [`_stitch_halves`] as one NumPy mesh with two disjoint rims.

    MeshLib's ``stitchHoles`` takes a *single* ``Mesh`` holding both holes, so the two halves are
    compacted independently and index-offset into one buffer -- the same construction
    ``tests/test_holes.py::_meshlib_stitch_band`` uses, which is what makes the benchmark and the
    parity test measure the same thing.
    """
    if bench_case.mesh_name not in _stitch_np_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        half = bench_case.n_faces // 2
        parts = []
        for lo, hi in ((0, half), (half, bench_case.n_faces)):
            used, inverse = np.unique(faces_np[lo:hi], return_inverse=True)
            parts.append((vertices_np[used], inverse.reshape(-1, 3)))
        (va_np, fa_np), (vb_np, fb_np) = parts
        _stitch_np_cache[bench_case.mesh_name] = (
            np.vstack([va_np, vb_np]),
            np.vstack([fa_np, fb_np + len(va_np)]),
        )
    return _stitch_np_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="stitch_min_weight")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_stitch_min_weight(bench_case: BenchCase) -> None:
    """
    Grid DP over the two rims: an La x Lb table whose cells are La + Lb sequential steps deep.

    meshlib is the only reference in the package that has this operation at all -- ``stitchHoles``
    is a real two-loop minimum-weight stitch where trimesh and pymeshlab have nothing, which is why
    it is also the oracle in tests/test_holes.py::test_stitch_min_weight_matches_meshlib. The
    four-argument overload is used deliberately (see section 6): the two-argument one finds the
    rims itself, and timing that would fold hole detection into the DP.

    **This row used to be launch-bound and is not any more.** The grid's ``La + Lb`` sequential
    steps were one kernel launch each -- 2 049 of them at 1 024-vertex rims, 24.6 ms of a 29 ms
    call against 7.0 ms of device time. They are now block barriers inside a tiled schedule, one
    launch per tile-diagonal, with a byte-identical band: **this row moved 2.9x** and the gain
    grows with the rim, reaching 6.9x on the DP-only call at twice these rims. What is left is the
    DP's own cells plus the fixed preamble the rims share with ``stitch``, so read this against
    that group rather than against its own history.
    """
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = _stitch_pair_np(bench_case)

        def run_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            edges_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            params_ml = mm.StitchHolesParams()
            params_ml.metric = mm.getComplexStitchMetric(mesh_ml)
            mm.stitchHoles(mesh_ml, edges_ml[0], edges_ml[1], params_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > len(faces_np)
        return
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    up = wp.vec3(0.0, 0.0, 1.0)
    _vertices, faces = bench_case.run(
        lambda: tw.holes.stitch_min_weight(va, fa, vb, fb, up_dir=up), rounds=_ROUNDS
    )
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="fillable_loop_mask")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_fillable_loop_mask(bench_case: BenchCase) -> None:
    """
    Which rims a min-weight fill can close: one face pass, plus one readback of the rims.

    The chord test is a sweep over every face's edges, so the cost tracks the *mesh* rather than the
    boundary; the pinch test rides on one readback of the loops, which tracks the boundary. Read
    against ``fill_min_weight`` -- the point of the mask is that it is a small fraction of the fill
    it predicts, so a caller can afford to ask first.

    meshlib's half of the answer is ``findRepeatedVertsOnHoleBd``, the pinch condition, which is the
    only one of its three fillability predicates this ports (the other two are recorded as not
    matching in ``tests/test_holes.py``). So the row is a **partial** comparison on the reference
    side and does strictly less work -- read it as a floor, not as a like-for-like.

    The first version read each rim back separately; the rims are concatenated on the device now and
    read in one transfer, which was worth about 2x. What is
    left is the chord sweep, which is a pass over every unique edge and so tracks the mesh -- there
    is no smaller correct version of that test, and it is the half meshlib does not do at all.

    **Attributed per stage, every benchmarked row is now a floor row.**
    The floor is **``edges_unique``** -- the shared unique/group stack, which has its own group and
    its own floor, and which is flat here across the face range, so it is that stack's fixed cost
    rather than this function's. Nothing local can move it: even deleting the chord test outright
    leaves a floor an order of magnitude above meshlib's cached-topology arithmetic. Two things were
    taken because they were free rather than because they showed up: the rim concatenation now
    re-uses the buffer ``boundary_loops`` already packed (``copy=False``, see
    ``benchmarks/test_boundary.py::test_loop_perimeters``), and an ``index_bound`` readback is gone
    because ``vertices`` already states the bound.

    ``dragon``'s remainder used to be a Python loop over its 407 rims building two vertex-indexed
    tables, and that was the one stage here with an algorithm left in it. It is gone: the pinch
    tests are one per-vertex slot count on the device and the tables are scattered by a kernel, so
    the whole predicate is readback-free and the many-rim row is 2.4x what it was while the
    few-rim rows are unchanged. What that removes is a *loop-count* term, which is why it does not
    show on ``bunny``.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        bits_ml = bench_case.run(lambda: mm.findRepeatedVertsOnHoleBd(mesh_ml.topology))
        assert bits_ml.size() >= 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = tw.boundary.boundary_loops(vertices, faces)
    if not loops:
        pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to judge")
    fillable = bench_case.run(lambda: tw.holes.fillable_loop_mask(vertices, faces, loops))
    assert int(fillable.shape[0]) == len(loops)


@pytest.mark.benchmark(group="extend_hole")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_extend_hole(bench_case: BenchCase) -> None:
    """
    Project every rim to a plane and bridge to it: two launches over the rim, plus two copies.

    The work is proportional to the *rim*, not the mesh -- but the two ``wp.copy`` calls that carry
    the input through are proportional to the mesh, and on a scan mesh they dominate. So read this
    against ``boundary_loops``, which produces the rims: the gap between them is the bridge itself
    and it is small.

    meshlib's ``extendAllHoles`` takes the plane directly and mutates, so its mesh is rebuilt per
    round. The two agree exactly on the face and vertex counts and on the plane the new rim lands in
    (``tests/test_holes.py``).

    The largest margin in this module -- well over an order of magnitude ahead of meshlib at every
    size -- and it is structural rather than clever: the extension is two launches over the rim
    while meshlib inserts the faces into a halfedge structure one at a time. On the largest mesh the
    two whole-mesh copies dominate everything the rim does.

    !!! note "This row hoists the rims out, so it misses one axis the default path has"
        Passing ``loops`` is what keeps the row measuring the extension rather than
        ``boundary_loops``, and it is the right call -- but it also hands the function a *list*, so
        the row cannot see the cost of deriving that list. On the ``loops=None`` default the rims
        come from ``boundary_loops_with_offsets`` and are never split into one array per rim; a
        407-rim scan mesh measured 1.65x from that alone, and the gap grows with the rim count.
    """
    height = float(bench_case.vertices_np[:, 2].max()) + 1.0
    if bench_case.kind == "meshlib":

        def extend_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            mm.extendAllHoles(mesh_ml, mm.Plane3f(mm.Vector3f(0.0, 0.0, 1.0), height))
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(extend_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = tw.boundary.boundary_loops(vertices, faces)
    if not loops:
        pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to extend")
    origin = wp.vec3(0.0, 0.0, height)
    normal = wp.vec3(0.0, 0.0, 1.0)
    extended_vertices, extended_faces = bench_case.run(
        lambda: tw.holes.extend_hole(vertices, faces, normal, origin, loops), rounds=3
    )
    assert int(extended_faces.shape[0]) > int(faces.shape[0])
    assert int(extended_vertices.shape[0]) > bench_case.n_vertices


@pytest.mark.benchmark(group="build_bottom")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_build_bottom(bench_case: BenchCase) -> None:
    """
    The same band as ``extend_hole``, with the plane fitted to each rim instead of given.

    Read against ``extend_hole``: the difference between the two rows is exactly the plane fit --
    one atomic-min pass over the rim and one map over the loops, both of which track the *rim* and
    not the mesh. Everything else is shared code, so a gap here that is not small would mean the
    fit had become the cost rather than the band.

    meshlib's ``buildBottom`` takes one hole at a time and mutates, so its mesh is rebuilt per round
    and every rim is bottomed in a loop -- which is what its row measures against triwarp's single
    batched launch set. The two agree on the counts and on where each base plane lands
    (``tests/test_holes.py``).

    Read against ``extend_hole`` on the same meshes: the plane fit adds a fraction of a millisecond
    on the small ones and reads *negative* on the largest, which is session drift rather than a
    saving -- the two share every launch but the atomic-min pass and one map over the loops, both of
    which track the rim. The fit is not the cost, which is what this row
    was written to establish.
    """
    if bench_case.kind == "meshlib":

        def bottom_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
                mm.buildBottom(mesh_ml, edge_ml, mm.Vector3f(0.0, 0.0, 1.0), 0.0)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(bottom_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = tw.boundary.boundary_loops(vertices, faces)
    if not loops:
        pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to bottom")
    direction = wp.vec3(0.0, 0.0, 1.0)
    bottomed_vertices, bottomed_faces = bench_case.run(
        lambda: tw.holes.build_bottom(vertices, faces, direction, 0.0, loops), rounds=3
    )
    assert int(bottomed_faces.shape[0]) > int(faces.shape[0])
    assert int(bottomed_vertices.shape[0]) > bench_case.n_vertices


@pytest.mark.benchmark(group="bridge_edges")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_bridge_edges(bench_case: BenchCase) -> None:
    """
    Two triangles, so this times the *validation* and the buffer copy, not the patch.

    Deliberately so: the patch is a fixed six integers whatever the mesh, and everything that
    scales is the check that both edges are on the rim and that the patch would not duplicate an
    edge -- one face-parallel census of the queried pairs' undirected and directed matches, and one
    readback. The row is therefore the price of ``validate=True``, which is the only decision a
    caller of this function has to make.

    That check was host-side Python when it landed -- a ``set`` comprehension over every mesh edge
    -- which made this row **minutes** on ``lucy`` and is what this benchmark existed to find. It
    became a boundary-edge build plus a membership scan, and is now a single scan of the faces with
    no edge table, roughly an order of magnitude cheaper again on ``lucy``.

    meshlib's ``makeBridge`` works on a halfedge structure that already knows which edges are on the
    boundary, so its row is the patch alone and is expected to win by a wide margin at every size;
    the comparable statement is that both produce the same two triangles
    (``tests/test_holes.py``), and that triwarp's cost is a *choice* the ``validate`` switch turns
    off.
    """
    # The rim is derived on the host so both branches see the same two edges; the meshlib branch
    # has no device to build a triwarp buffer on.
    corners = bench_case.faces_np.reshape(-1, 3)
    directed = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]], axis=0)
    undirected = np.sort(directed, axis=1)
    _, first, counts = np.unique(undirected, axis=0, return_index=True, return_counts=True)
    rim_np = directed[first[counts == 1]]
    if len(rim_np) < 16:
        pytest.skip(f"{bench_case.mesh_name} has no rim long enough to bridge across")
    edge_a = (int(rim_np[0][0]), int(rim_np[0][1]))
    edge_b = (int(rim_np[len(rim_np) // 2][0]), int(rim_np[len(rim_np) // 2][1]))

    if bench_case.kind == "meshlib":

        def bridge_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            a_ml = mesh_ml.topology.findEdge(mm.VertId(edge_a[1]), mm.VertId(edge_a[0]))
            b_ml = mesh_ml.topology.findEdge(mm.VertId(edge_b[1]), mm.VertId(edge_b[0]))
            return mm.makeBridge(mesh_ml.topology, a_ml, b_ml).newFaces

        assert bench_case.run(bridge_ml, rounds=3) >= 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    bridged_faces = bench_case.run(
        lambda: tw.holes.bridge_edges(vertices, faces, edge_a, edge_b), rounds=3
    )
    assert int(bridged_faces.shape[0]) > int(faces.shape[0])


@pytest.mark.benchmark(group="bridge_edges_smooth")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_bridge_edges_smooth(bench_case: BenchCase) -> None:
    """
    The curved strip: the same validation as ``bridge_edges``, plus a fixed-size spline.

    Read against ``bridge_edges``, whose edge pair this shares. The difference between the two rows
    is the strip -- one launch to find the two incident faces' third corners, two small readbacks
    and a host-side sample loop whose length is the span over ``sampling_step`` and does **not**
    grow with the mesh. So the gap should be flat across the axis; a gap that widens with the mesh
    means something in the strip path started reading a whole buffer.

    ``sampling_step`` is 5 % of the bounding-box diagonal, which puts both sides at a comparable
    number of segments -- meshlib derives its own count from an arc-length estimate, so the two
    never match exactly and the comparison is on the strips as surfaces
    (``tests/test_holes.py``).

    Against ``bridge_edges`` on the same pairs, the strip costs well under a millisecond and does
    not track the mesh, which is the flatness this row is here to check. The largest mesh is the one
    to read: the strip is *cheaper* there than on a small one in absolute terms while the shared
    validation dominates, so the two halves are cleanly separated.
    """
    corners = bench_case.faces_np.reshape(-1, 3)
    directed = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]], axis=0)
    _, first, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_index=True, return_counts=True
    )
    rim_np = directed[first[counts == 1]]
    if len(rim_np) < 16:
        pytest.skip(f"{bench_case.mesh_name} has no rim long enough to bridge across")
    edge_a = (int(rim_np[0][0]), int(rim_np[0][1]))
    edge_b = (int(rim_np[len(rim_np) // 2][0]), int(rim_np[len(rim_np) // 2][1]))
    extent = bench_case.vertices_np.max(0) - bench_case.vertices_np.min(0)
    sampling_step = 0.05 * float(np.linalg.norm(extent))

    if bench_case.kind == "meshlib":

        def bridge_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            a_ml = mesh_ml.topology.findEdge(mm.VertId(edge_a[1]), mm.VertId(edge_a[0]))
            b_ml = mesh_ml.topology.findEdge(mm.VertId(edge_b[1]), mm.VertId(edge_b[0]))
            return mm.makeSmoothBridge(mesh_ml, a_ml, b_ml, sampling_step).newFaces

        assert bench_case.run(bridge_ml, rounds=3) >= 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    strip_vertices, strip_faces = bench_case.run(
        lambda: tw.holes.bridge_edges_smooth(vertices, faces, edge_a, edge_b, sampling_step),
        rounds=3,
    )
    assert int(strip_faces.shape[0]) > int(faces.shape[0])
    assert int(strip_vertices.shape[0]) >= bench_case.n_vertices


@pytest.mark.benchmark(group="join_closest_components")
@pytest.mark.benchaxis("open_components")
@pytest.mark.benchlibs("triwarp", "pymeshfix")
def test_join_closest_components(bench_case: BenchCase) -> None:
    """
    Greedy nearest-link agglomeration over *open* shells: the driver, not the bridge.

    **The axis is the join count, and the face count is very nearly free.** At a fixed face count
    triwarp is linear in the ``k - 1`` joins; holding the components fixed and cutting the mesh
    instead barely moves it, a 4x face reduction costing a percent. That is the shape the function's
    own Notes predict -- each round rebuilds the component labelling and the rim table from the
    updated face buffer, and each is a fixed chain of wrapper calls whose launch overhead dominates
    the per-face work at these sizes.

    So this group watches the **round count times the per-round chain**, and anything that moves it
    will be a launch removed from that chain rather than a faster kernel. One such change is already
    in: the pair selection and ``bridge_edges``' rim check used to build the same oriented boundary
    table twice per round, which was a seventh or so of the call.

    A closed lattice cannot serve this. ``parts_64`` returns **0 bridges** -- a shell with no
    boundary has nothing to bridge to and is left alone by design -- so the ``components`` axis
    would time a no-op, which is why ``open_components`` exists and drops one face per shell.

    The ratio *narrows* with the join count because triwarp's per-round chain is the launch-bound
    part while MeshFix's is sequential C++ over a halfedge structure it already holds -- the gap is
    widest where triwarp's fixed overhead is amortized least.

    pymeshfix's row is timed rather than declared because the operation clears its load by a wide
    margin, which is the ~30 % rule in ``conftest``'s LIBRARIES block -- from half the round at the
    fewest components to nearly all of it at the most. That split is an isolated measurement, since
    the harness can only time the whole callable: the load cannot leave it, a ``PyTMesh`` taking
    exactly one ``load_array``. So read the row as join-plus-load and subtract accordingly, most of
    all at the few-component point, where the load is over half of it and the reported ratio
    correspondingly overstates the algorithmic gap.

    Both sides add exactly ``2 * (k - 1)`` faces and no vertices, which is what the assertions
    check: a driver that fanned a rim or joined the wrong number of shells fails on the count. The
    parity claim compares the same counts, because the two pick different incident edges to bridge
    and so need not produce the same two triangles.
    """
    n_faces = bench_case.n_faces
    n_components = _OPEN_COMPONENT_COUNTS[bench_case.mesh_name]
    expected = 2 * (n_components - 1)

    if bench_case.kind == "pymeshfix":

        def join_pmf() -> int:
            tin_pmf = bench_case.new_tmesh_pmf()
            tin_pmf.join_closest_components()
            return tin_pmf.n_faces

        assert bench_case.run(join_pmf, rounds=3) == n_faces + expected
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    joined = bench_case.run(lambda: tw.holes.join_closest_components(vertices, faces), rounds=3)
    assert int(joined.shape[0]) // 3 == n_faces + expected
