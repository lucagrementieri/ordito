"""
Benchmarks for ``ordito.remesh``.

Three axes, because the four entry points fail to scale for three different reasons:

* ``subdivide`` on the **scan sweep** — exactly 4x the faces, one pass, no data dependence. The
  module's throughput baseline and the only group a face count fully explains.
* ``subdivide_to_size`` on **scale**, sweeping ``max_edge`` — output grows as
  ``(L_max / max_edge) ** 2`` while the *pass count* grows only as ``log2(L_max / max_edge)``, so
  halving the target is roughly four times the output for one extra pass. Edge-length *variance*
  matters too: one long edge forces another pass over the whole mesh.
* ``flip_to_delaunay`` and ``isotropic_remesh`` on **quality** — the iterative paths, where the
  round count is set by how far the input is from the fixed point rather than by its size.
  ``saddle_graded`` exists for exactly this: identical connectivity to ``saddle``, worst aspect
  ratio 4 719 against 1.6, so it is badly non-Delaunay and full of slivers while ``saddle`` is
  nearly converged. A near-Delaunay mesh flips in one or two rounds; a bad one runs all
  ``max_iter``.

Sizing comes from the mesh's own mean edge length (computed once from the NumPy source so every
library gets the *same* target), which keeps the work proportional to the mesh rather than to an
absolute length that would explode on one and no-op on another. ``flip_to_delaunay`` mutates its
face buffer in place, so the timed callable clones it — one device copy, negligible against the
passes it feeds.

References
----------
``subdivide`` has an exact open3d equivalent in ``subdivide_midpoint(1)`` (same 1:4 split, and it
returns a new mesh so the shared one is reusable) and a trimesh one; ``subdivide_to_size`` has a
trimesh counterpart. Open3D has no edge-flip pass and no isotropic remesher, and its
``subdivide_midpoint`` takes an iteration count rather than a length target so it cannot split
adaptively. trimesh has neither iterative path.

**pymeshlab** is the *only* reference ``isotropic_remesh`` has, and an unusually close one:
``meshing_isotropic_explicit_remeshing`` runs the same five stages in the same order (refine,
collapse, edge-swap, Laplacian relax, reproject), each individually switchable, so the comparison is
stage-for-stage. Two differences are left in place — it preserves crease edges above
``featuredeg=30`` and its ``checksurfdist`` default rejects any local operation deviating more than
1 % of the bbox diagonal — because turning them off would measure a filter no MeshLab user runs.
Ordito's counterpart to the second (``max_deviation``) is deliberately **not** passed: at
MeshLab's own 1 % default it is byte-identically a no-op on both fixtures, so passing it would buy
a per-iteration closest-point pass and no change in output.

It also covers ``subdivide_to_size`` through
``meshing_surface_subdivision_midpoint(threshold=...)``, which lands on the identical output face
count. It cannot cover the uniform ``subdivide`` group, and the reason is a hard failure rather than
a slow row: **midpoint subdivision raises** ``Mesh has some not 2 manifold faces`` on every scan
mesh — the same boundary the libigl and potpourri3d references hit — so the midpoint reference lives
on the ``scale`` axis with ``subdivide_to_size``.

``flip_by_objective`` joins ``flip_to_delaunay`` on **quality** because it is the same engine with a
different predicate, so the pair isolates what the predicate costs from what the flip machinery
costs. The two objectives behave differently there by construction: the planarity one refuses any
quad that is not flat and so converges in one pass on a curved input, where the curvature one has
work everywhere. MeshLab's ``meshing_edge_flip_by_planar_optimization`` is the reference for the
first, at the same ``pthreshold`` and ``planartype``; it rewrites the topology, so that row rebuilds
the MeshSet.

``quadric_decimate`` pairs with ``cluster_decimate`` because they solve one problem with opposite
structures: clustering is three data-independent passes, the quadric method an iterated greedy loop
whose pass count depends on how contested the rings are — exactly what triangle shape changes. All
three serial references are here (``igl.decimate``, open3d, pymeshlab), the best-referenced port in
the package. Note MeshLab's filter defaults to ``autoclean=True`` and deletes unreferenced vertices,
so its MeshSet is rebuilt per round; and the *quality* comparison lives in ``tests/test_remesh.py``,
where ordito measures a lower Hausdorff error than igl and open3d on ``icosphere(4)``. On
``bunny`` and ``dragon`` at 0.1 and 0.01 its mean deviation is below MeshLib's, pymeshlab's and
open3d's, and within 1.1x of the best at 0.5; its maximum ranges from below the best of them to 5x
MeshLib's (``dragon`` at 0.5, where MeshLib's is unusually low). **That held only once it stopped
freezing features by default** (every reference keeps none; at 30 degrees a scan could not go below
about a tenth of its faces and its mean deviation was 3-60x the references') **and gained
Garland-Heckbert's boundary planes**, without which rim vertices slid off the surface.

**Ordito was long the slower one on that group, and the reason is the pass count rather than the
per-pass work**: an independent set commits a fraction of the candidates, so reaching a tenth of the
faces takes many passes, each paying a full edge/adjacency/quadric rebuild plus two radix sorts that
a serial queue pays none of. Committing **several** independent sets against one rebuild closed part
of it (see ``_QUADRIC_RATIOS`` for that and for what the other candidate lever was worth),
capturing the pass closed more, and relaxing the independence rule to the exact conflict (no
endpoint of either collapse in the other's closed 1-rings, rather than disjoint rings) halved the
pass count.

``cluster_decimate`` is the other **scan sweep** group and the interesting one to read against
``subdivide``: the same shape of work in reverse (bin, remap, dedup — no data dependence, no
iteration), so if the two do not scale alike something is wrong with one. Both open3d and pymeshlab
are references, and the open3d one is *exact* — same grid anchor, same cell means, identical face
count — which is rare enough here to be worth stating. Cell width is a fraction of the mean edge
length so the ratio is comparable across meshes.

Caps: ``isotropic_remesh`` stops at ``bunny`` on the scan sweep (it is an order of magnitude dearer
on ``dragon`` and would dominate the suite); the split paths stop at ``dragon`` because a 1:4
subdivision of ``happy_buddha`` / ``lucy`` does not fit a sane memory budget; the ``trimesh``
reference for ``subdivide_to_size`` stops at ``bunny`` for a ratio the smaller meshes already
establish, and the pymeshlab one skips ``sphere_large`` for the same reason.
"""

from __future__ import annotations

from typing import Literal

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, face_bitset_ml, mesh_ml_from_numpy, skip_larger_than

# MeshLib expresses several gates as *absolute* lengths whose defaults assume a unit-scale mesh
# (``SubdivideSettings.maxDeviationAfterFlip`` is 1.0). Passing this instead disables the gate
# rather than letting the fixture's units decide whether it binds.
_UNBOUNDED = 1e30

_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """
    Cache one ``meshlib.Mesh`` per mesh, for the *non*-mutating rows only.

    Every other meshlib row in this file rebuilds inside its timed callable, because the operation
    rewrites the mesh. ``verticesGridSampling`` is the exception -- it returns a bitset and leaves
    the mesh alone -- so its row may share one, and sharing also keeps the lazily built AABB tree
    warm across rounds.
    """
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


# Iterations for the full remeshing pipeline. The default is 10; 3 keeps the case under a couple
# of seconds while still exercising the split/collapse/flip/smooth/reproject loop several times.
_REMESH_ITERATIONS = 3

# Split targets as a fraction of the mean edge length. Both are below 1.0 so edges actually split;
# halving the fraction quadruples the output for one extra pass, which is the point of the pair.
_SPLIT_FRACTIONS = [0.7, 0.35]

# The iterative paths run to seconds a call.
_ROUNDS = 3


@pytest.mark.benchmark(group="subdivide")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pytorch3d")
def test_subdivide(bench_case: BenchCase) -> None:
    """
    Exactly 4x the faces in one pass: the module's clean throughput baseline.

    Both references differ from ordito only in the output *ordering*, which is what makes the
    parity comparison a centroid match rather than an array compare (``tests/test_remesh.py``).

    !!! warning "``igl.upsample`` is memory-unsafe on the scan meshes and is deliberately absent"
        It is midpoint 1:4 subdivision with ordito's semantics exactly, so it *should* be the
        widest reference agreement in the module -- but on a scan mesh it corrupts the process heap
        and takes the whole pytest session down with a SIGSEGV, which loses every other row in this
        module because ``--benchmark-json`` is only written at session end.

        One selection per process: the igl rows crash some of the time with no other library in the
        selection at all and *every* time with the other libraries present, and per mesh almost
        always on the two small meshes and never on the largest. So it is not an interaction with
        ordito, and it is not a size limit -- the *smallest* mesh fails most and the largest never
        does. Compacting the unreferenced vertices away makes it worse, so there is nothing to pass
        it that makes it safe. This is the third memory-unsafe binding in this wheel, alongside
        ``igl.loop`` and ``igl.in_element``.
        It stays a *tested* reference on the small clean ``icosahedron`` fixture, where it is
        reliable.

    **pytorch3d** is the third reference and the only GPU one. ``SubdivideMeshes()`` is constructed
    *inside* the timed callable on purpose: passing a mesh to its constructor caches the subdivision
    topology and every later call reuses it, which would time a gather rather than a subdivision --
    and ordito's row rebuilds everything each call. It agrees with ordito on sorted coordinates
    exactly, so like the other two it differs only in output ordering. The row carries its
    ``Meshes`` build, which is cheap next to a 4x face expansion.
    """
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    if bench_case.kind == "pytorch3d":
        mesh_p3d = bench_case.mesh_p3d
        n_faces = bench_case.n_faces
        subdivided_p3d = bench_case.run(lambda: p3d_ops.SubdivideMeshes()(mesh_p3d))
        assert subdivided_p3d.faces_packed().shape[0] == 4 * n_faces
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(lambda: od.remesh.subdivide(vertices, faces))
        assert int(new_faces.shape[0]) == 4 * faces.size
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        _new_vertices, new_faces = bench_case.run(lambda: tm.remesh.subdivide(vertices, faces))
        assert new_faces.shape[0] == 4 * faces.shape[0]
    else:  # open3d midpoint subdivision returns a new mesh, so the shared one is reusable
        mesh_o3d = bench_case.mesh_o3d
        n_faces = bench_case.n_faces
        subdivided = bench_case.run(lambda: mesh_o3d.subdivide_midpoint(number_of_iterations=1))
        assert len(subdivided.triangles) == 4 * n_faces


@pytest.mark.benchmark(group="subdivide_loop")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl", "open3d")
def test_subdivide_loop(bench_case: BenchCase) -> None:
    """
    Loop subdivision: the same 1:4 split as ``subdivide``, with smooth stencils not midpoints.

    Read against ``subdivide`` above, which does identical topology work and returns the same output
    size: the gap between them is the cost of the stencils alone -- two extra atomic accumulation
    passes (per-edge opposite vertices, per-vertex ring sums) and two position kernels over grids
    the midpoint split never visits. The two groups sit on different axes for the reason below, so
    compare per-triangle throughput rather than raw medians.

    ``igl.loop`` and Open3D's ``subdivide_loop`` are the same variant as ordito -- the two agree
    with each other to 2e-16 on the relocated originals, all three using **Warren's** ``beta``
    (``3/(8k)``, ``3/16`` at ``k = 3``). ``igl.loop``'s ``number_of_subdivs`` stays at 1, which is
    one ordito call. Both references take ``rounds=3``.

    **This group is on the ``scale`` axis, not the scan sweep, because ``igl.loop`` cannot survive
    the scan meshes.** With no Warp in the process it aborts with ``free(): invalid pointer`` on a
    five-vertex mesh with three faces on one edge, and SIGSEGVs on ``bunny_decimated``, whose
    duplicated faces leave it not edge-manifold. On ``bunny`` it does return, with silent ``NaN``
    rows -- one per unreferenced vertex, since ``igl::adjacency_list`` is sized ``F.max() + 1``
    where ``igl::loop`` indexes it to ``n_verts``. That is exactly the hazard the ``scale`` axis
    exists for.

    ``trimesh.remesh.subdivide_loop`` gets no row either, for two independent reasons: it uses
    Loop's *original* trigonometric ``beta``, which agrees with Warren's exactly at valence 6 and
    differs elsewhere, so it could not be a parity oracle; and it divides by the neighbour count, so
    its own ``assert np.isfinite`` fails on any mesh with an unreferenced vertex. ordito keeps an
    unreferenced vertex where it is and has no stencil that can divide by zero, which is why its row
    is the one that asserts finiteness.
    """
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        new_vertices, new_faces = bench_case.run(lambda: od.remesh.subdivide_loop(vertices, faces))
        assert new_faces.size == 4 * faces.size
        assert np.isfinite(new_vertices.numpy()).all()
    elif bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        vertices_loop_igl, faces_loop_igl = bench_case.run(
            lambda: igl.loop(vertices_np, faces_np), rounds=3
        )
        assert faces_loop_igl.shape[0] == 4 * bench_case.n_faces
        assert np.isfinite(vertices_loop_igl).all()
    else:  # open3d returns a new mesh, so the shared one is reusable
        mesh_o3d = bench_case.mesh_o3d
        n_faces = bench_case.n_faces
        subdivided = bench_case.run(
            lambda: mesh_o3d.subdivide_loop(number_of_iterations=1), rounds=3
        )
        assert len(subdivided.triangles) == 4 * n_faces


@pytest.mark.benchmark(group="subdivide_to_size")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "trimesh", "pymeshlab", "meshlib")
@pytest.mark.parametrize("split_fraction", _SPLIT_FRACTIONS)
def test_subdivide_to_size(bench_case: BenchCase, split_fraction: float) -> None:
    """
    Adaptive splitting to an edge-length target: quadratic in output, logarithmic in passes.

    meshlib's ``subdivideMesh`` takes the same absolute ``maxEdgeLen`` and makes the same guarantee
    -- every edge under the cap -- but it also *flips* as it splits, so it is doing more than the
    other three rows and lands on a slightly different mesh (3 320 faces against ordito's 3 200 on
    a test-size sphere; ``tests/test_remesh.py``). ``maxEdgeSplits`` is raised from its default of
    1 000, which would stop the reference long before the cap on any registry mesh, and
    ``maxDeviationAfterFlip`` from its default of 1.0 -- an *absolute* length, so leaving it would
    make the flip gate depend on the fixture's units. It mutates in place, so its mesh is rebuilt
    inside the timed callable.
    """
    max_edge = split_fraction * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def subdivide_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.SubdivideSettings()
            settings_ml.maxEdgeLen = max_edge
            settings_ml.maxEdgeSplits = 10_000_000
            settings_ml.maxDeviationAfterFlip = _UNBOUNDED
            mm.subdivideMesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(subdivide_ml, rounds=_ROUNDS) >= bench_case.n_faces
        return
    if bench_case.kind == "pymeshlab":
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("MeshLab's midpoint refinement takes ~6 s at this size and target")
        # ``iterations`` is a pass cap rather than a convergence criterion, so it is set well above
        # the log2 depth the target needs; the surplus passes find nothing to refine.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().meshing_surface_subdivision_midpoint(
                iterations=10, threshold=ml.PureValue(max_edge)
            ),
            rounds=_ROUNDS,
        )
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(
            lambda: od.remesh.subdivide_to_size(vertices, faces, max_edge), rounds=_ROUNDS
        )
        assert new_faces.size >= faces.size
    else:
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("trimesh subdivide_to_size takes tens of seconds at this size")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(
            lambda: tm.remesh.subdivide_to_size(vertices, faces, max_edge), rounds=_ROUNDS
        )
        assert result[1].shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="split_edges")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("split_fraction", [0.25, 1.0], ids=["quarter", "all"])
def test_split_edges(bench_case: BenchCase, split_fraction: float) -> None:
    """
    One crack-free split pass over a given edge mask: the primitive ``subdivide_to_size`` iterates.

    Timed separately from that group because the two answer different questions.
    ``subdivide_to_size`` measures the whole *loop* — how many passes a length target needs — and
    hides the per-pass cost inside it; this measures one pass at a known mask density, so a
    regression in the emission templates or in the ``edges_unique`` build shows up here undiluted.
    The ``all`` case is the regular 1-to-4 subdivision and should track the ``subdivide`` group
    closely (same output, one extra edge-table build); the gap between ``quarter`` and ``all`` is
    how much of the pass is fixed edge-table cost rather than emission, and it should be *sublinear*
    in the mask density because the sort runs over every edge either way.

    The mask is built on the host once per case and excluded from the timing, since choosing which
    edges to split is the caller's job and varies per use.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    unique_edges, inverse = od.edges.edges_unique(faces)
    n_edges = int(unique_edges.shape[0])
    if split_fraction >= 1.0:
        mask = wp.full(n_edges, True, dtype=wp.bool, device=bench_case.device)
    else:
        rng = np.random.default_rng(20260811)
        mask = wp.array(
            np.ascontiguousarray(rng.random(n_edges) < split_fraction),
            dtype=wp.bool,
            device=bench_case.device,
        )
    new_vertices, new_faces = bench_case.run(
        lambda: od.remesh.split_edges(
            vertices, faces, mask, unique_edges=unique_edges, inverse=inverse
        ),
        rounds=_ROUNDS,
    )
    assert new_faces.size >= faces.size
    assert new_vertices.size >= vertices.size


# Split budgets for the region refiner, as a fraction of the input face count. ``None`` runs to
# convergence; the finite one makes ``max_splits`` bind, which is the only way the budget-truncation
# branch is reached at all.
_REGION_SPLIT_BUDGETS = [None, 0.25]


def _region_half_np(bench_case: BenchCase) -> np.ndarray:
    """Dense face mask covering the half of the mesh below the median face-centroid ``x``."""
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    centroid_x = vertices_np[faces_np, 0].mean(axis=1)
    return np.ascontiguousarray(centroid_x < np.median(centroid_x))


def _region_half(bench_case: BenchCase) -> wp.array[wp.bool]:
    """Face mask covering the half of the mesh below the median face-centroid ``x``."""
    return wp.array(_region_half_np(bench_case), dtype=wp.bool, device=bench_case.device)


@pytest.mark.benchmark(group="subdivide_region_to_size")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "meshlib")
@pytest.mark.parametrize("split_budget", _REGION_SPLIT_BUDGETS)
def test_subdivide_region_to_size(bench_case: BenchCase, split_budget: float | None) -> None:
    """
    The same adaptive split restricted to a face region, with and without a split budget.

    The pair is the point rather than either row alone. ``max_splits=None`` refines the region to
    convergence; a finite budget stops early *and* takes a branch the unbudgeted path never
    reaches, ``_keep_longest_edges``, which ranks the pass's eligible edges and keeps the longest
    ones that fit. Only the budgeted row prices that ranking, and it runs **once per call** rather
    than once per pass: truncating a pass spends the whole remaining budget, so the next pass exits
    at ``remaining <= 0``.

    meshlib is the only reference with the region restriction, and it has the *whole* parameter set:
    ``SubdivideSettings`` carries ``region``, ``maxEdgeLen``, ``maxEdgeSplits``,
    ``maxAngleChangeAfterFlip`` and ``maxDeviationAfterFlip``, which is this function's signature
    argument for argument -- so both rows are given the same target, the same split budget and the
    same unbounded flip gate. trimesh's and MeshLab's refiners take an edge-length target over the
    *whole* mesh with no region restriction and no split cap, which is why the sibling
    ``subdivide_to_size`` group carries them and this one does not.

    It is doing slightly more than ordito for the same reason its sibling row is: ``subdivideMesh``
    flips as it splits, so it lands on a nearby rather than identical mesh -- slightly more faces
    and slightly less area, because a flip moves the surface where a split does not
    (``tests/test_remesh.py``). Two traps in the settings
    object: ``maxEdgeSplits`` defaults to 1 000, which stops it long before the cap on any registry
    mesh, and ``maintainRegion`` is a ``FaceBitSet`` rather than the bool its name reads as.
    """
    max_edge = 0.35 * bench_case.mean_edge
    max_splits = None if split_budget is None else int(split_budget * bench_case.n_faces)

    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        region_np = _region_half_np(bench_case)
        budget_ml = 10_000_000 if max_splits is None else max_splits

        def subdivide_region_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.SubdivideSettings()
            settings_ml.maxEdgeLen = max_edge
            settings_ml.maxEdgeSplits = budget_ml
            settings_ml.maxDeviationAfterFlip = _UNBOUNDED
            settings_ml.region = face_bitset_ml(region_np)
            mm.subdivideMesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(subdivide_region_ml, rounds=_ROUNDS) >= bench_case.n_faces
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    region = _region_half(bench_case)
    _new_vertices, new_faces, new_region = bench_case.run(
        lambda: od.remesh.subdivide_region_to_size(
            vertices, faces, region, max_edge, max_splits=max_splits
        ),
        rounds=_ROUNDS,
    )
    assert new_faces.size >= faces.size
    assert new_region.size == new_faces.size // 3


@pytest.mark.benchmark(group="refine_region_to_density")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito")
def test_refine_region_to_density(bench_case: BenchCase) -> None:
    """
    Liepa's density criterion instead of a target edge length, on the axis it exists for.

    The ``quality`` axis is the one that separates the two refiners: ``saddle`` and
    ``saddle_graded`` have identical vertex counts, face counts and connectivity, and differ only in
    how unevenly the triangles are sized. A target-length refiner does the same work on both -- it
    compares every edge against one number -- where this one compares each triangle against its
    *own* neighbourhood, so the graded row is where its per-vertex scale attribute earns its keep
    and the uniform row is the control.

    The cost is one scan pair plus one kernel per pass, and the split is 1 -> 3 at the centroid, so
    unlike edge bisection there is no crack-free template and no agreement with the neighbours --
    which is what makes the criterion cheap in parallel rather than merely correct.

    There is no reference row. pymeshfix performs this refinement, but only as a stage inside
    ``fill_small_boundaries``, behind a load that is most of the round and cannot be hoisted out of
    it, so a row would price the load;
    ``tests/test_holes.py`` compares the answers instead. pymeshlab's ``refineholeedgelen`` is a
    target *length*, i.e. the other criterion.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    region = _region_half(bench_case)

    _new_vertices, new_faces, new_region = bench_case.run(
        lambda: od.remesh.refine_region_to_density(vertices, faces, region), rounds=_ROUNDS
    )
    assert new_faces.size >= faces.size
    assert new_region.size == new_faces.size // 3


@pytest.mark.benchmark(group="flip_to_delaunay")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_flip_to_delaunay(bench_case: BenchCase) -> None:
    """
    Flip rounds until convergence: driven by distance from Delaunay, not by size.

    meshlib's ``makeDeloneEdgeFlips`` is the same empty-circumcircle criterion reached by a serial
    queue where ordito commits a conflict-free independent set per round, so this pair is the
    clearest parallel-against-serial contrast in the module -- and the two reach the *same*
    fixpoint: after ordito's pass MeshLib finds zero further flips to make
    (``tests/test_remesh.py``). It mutates, so its mesh is rebuilt inside the timed callable, which
    is the same thing the ordito row's ``wp.clone`` is doing and for the same reason: without it,
    rounds 2..n would start from an already-Delaunay mesh.
    """
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def flip_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            return mm.makeDeloneEdgeFlips(mesh_ml, mm.DeloneSettings(), 100)

        assert bench_case.run(flip_ml, rounds=_ROUNDS) >= 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp

    # ``flip_to_delaunay`` rewrites the winding in place, so each round needs a fresh buffer;
    # without the clone every round after the first would start from an already-Delaunay mesh.
    def run() -> wp.array[wp.int32]:
        return od.remesh.flip_to_delaunay(vertices, wp.clone(faces), max_iter=100)

    flipped = bench_case.run(run, rounds=_ROUNDS)
    assert flipped.size == faces.size


@pytest.mark.noparity(
    "pymeshlab",
    reason="D2 the same five stages under a different stopping rule, with the disagreement "
    "measured and tabulated in this function's own docstring: ordito runs a fixed iterations x "
    "five parallel launches while MeshLab works a serial local-operation queue until the "
    "operations stop paying off, so at iterations=3 and the identical target length the two "
    "return different meshes with no vertex correspondence -- 35 568 faces against 34 946 on "
    "saddle, and on saddle_graded 41 604 against 31 414 with ordito missing the target edge "
    "length by 15% (0.854 of target against 0.983) and leaving a 99th-percentile aspect ratio of "
    "8.60 against 1.87. That gap is the finding this row exists to report, not a tolerance to "
    "widen; the quality statistics themselves are asserted against MeshLib and against the *input* "
    "in tests/test_remesh.py, in test_remesh_edge_concentration and "
    "test_remesh_emits_no_degenerate_faces, rather than against MeshLab. The aspect-ratio half of "
    "that gap used to read 352 against 1.87 and has closed 41x since, in two steps and for one "
    "reason each: the collapse stage gained the fold veto its quadric sibling already had "
    "(352 -> 9.13), and the smooth stage became the area-equalizing relaxation its Notes promise "
    "rather than the plain one-ring centroid (9.13 -> 8.60). The stopping rule is what is left. "
    "Re-measured once before that, after isotropic_remesh gained max_deviation, since an absent "
    "surface-distance gate used to be a listed source of the gap: passing MeshLab's own "
    "checksurfdist default (1% of the bbox diagonal, 0.0288 here) changed nothing at all -- "
    "byte-identical face counts on both fixtures -- because on these patches the remesh never "
    "moves a vertex that far, so the bound does not bind. The parameter is therefore deliberately "
    "NOT passed in this row: it would add a closest-point pass per iteration to the timing while "
    "provably changing no output.",
)
@pytest.mark.benchmark(group="isotropic_remesh")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab", "meshlib")
def test_isotropic_remesh(bench_case: BenchCase) -> None:
    """
    Five stages x ``iterations``, on a nearly-converged mesh and a badly conditioned one.

    MeshLab runs the identical five stages, and the pair reads in opposite directions: pymeshlab
    spreads several-fold across the axis where ordito is flat, so a modest win on ``saddle``
    becomes a large one on ``saddle_graded``. MeshLib's own queue is level with ordito, and its
    unchanged column is what certifies the ordito row.

    **This is not a like-for-like win, and the timing alone is misleading.** ordito is flat because
    its loop is a fixed ``iterations`` x five launches whatever the input looks like; MeshLab's
    serial local-operation queue keeps working until the operations stop paying off. Comparing the
    *outputs* at the same iteration count and target length says what the difference buys: on
    ``saddle`` ordito is past parity on aspect ratio and within a couple of percent of the
    requested length, while on the *graded* patch it misses the target by more and leaves a far
    worse 99th-percentile aspect ratio than MeshLab.

    **The two ordito rows moved together and for one reason, which is worth stating because the
    graded row got worse.** The collapse stage's parallel independent set locked by the *raw* edge
    index, which ``edges_unique`` orders lexicographically and which is therefore spatially monotone
    on any structured mesh -- so the lock had one local minimum and committed **one** collapse per
    pass out of tens of thousands of candidates, at the same wall clock as the hashed key that
    commits thousands. ``kernels/remesh.py``'s ``scramble_index`` fixes it. It also unmasked the
    smoother defect below, since closed: with collapse finally committing, the graded patch's worst
    triangles got worse rather than better. That is not the key's doing -- at the tests' own target
    the raw key leaves several times more float32-degenerate faces on this mesh than the hashed one
    -- it is what a collapse stage does on anisotropic input when the smoother downstream of it
    cannot see the anisotropy.

    Two bugs were found by reading this pair against the reference, both pre-existing and invisible
    to ``tests/test_remesh.py``, which only ever ran the remesher on clean closed icospheres:

    * The swap stage flipped on the valence objective with only a convexity guard, so on a graded
      mesh
      it turned slivers into worse slivers and in float32 hit exactly-zero area on a few percent of
      them. It now also rejects any flip that would create a degenerate triangle or worsen the
      pair's aspect ratio.
    * ``_smooth_pass`` is now the area-equalizing relaxation this module's Notes promise, weighting
      each
      one-ring neighbour by its barycentric area. It had to be: a regular graded grid is
      valence-perfect, already Delaunay, *and* a fixed point of the unweighted Laplacian, so with
      the plain centroid three of the five stages were blind to its anisotropy by construction and
      only split/collapse acted. The ``cave_cube`` self-intersection that had blocked this was never
      the smoother's -- it was the missing collapse fold veto -- and with that in place the
      weighting lands with no veto of its own needed (one is kept anyway, because it measures free).
      What is left is the stopping rule, which is this group's standing D2 exemption. The whole call
      is also faster, since a better-shaped mesh gives split and collapse less to do.
    """
    target = bench_case.mean_edge
    if bench_case.kind == "meshlib":
        # A third stopping rule: MeshLib runs its own local-operation queue to convergence, so like
        # MeshLab's row it is not a fixed-iteration count and the comparison is what the outputs
        # look like rather than what one round costs. ``projectOnOriginalMesh`` is left off, which
        # is its default and matches ordito: turning it on adds a closest-point pass per vertex.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def remesh_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.RemeshSettings()
            settings_ml.targetEdgeLen = target
            mm.remesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(remesh_ml, rounds=_ROUNDS) > 0
        return
    if bench_case.kind == "pymeshlab":
        # ``PureValue`` so both sides get the identical absolute target rather than a percentage of
        # a bounding box the two meshes do not share.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().meshing_isotropic_explicit_remeshing(
                iterations=_REMESH_ITERATIONS, targetlen=ml.PureValue(target)
            ),
            rounds=_ROUNDS,
        )
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _new_vertices, new_faces = bench_case.run(
        lambda: od.remesh.isotropic_remesh(
            vertices, faces, target_length=target, iterations=_REMESH_ITERATIONS
        ),
        rounds=_ROUNDS,
    )
    assert new_faces.size > 0


@pytest.mark.benchmark(group="intrinsic_delaunay")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_intrinsic_delaunay(bench_case: BenchCase) -> None:
    """
    The flip loop: rounds of predicate, claim, length update and commit until nothing is left.

    On the ``scale`` axis this measures the *no-work* path -- an icosphere is already intrinsically
    Delaunay, so the loop pays one round to discover that and stops -- which is the honest baseline
    for the flag being on by default. The meshes that actually flip are the quad grids, and none of
    them is on this axis; ``tests/test_intrinsic.py`` covers those for correctness.
    """
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        intrinsic_faces, lengths, _ = bench_case.run(
            lambda: od.remesh.intrinsic_delaunay(vertices, faces)
        )
        assert intrinsic_faces.shape == faces.shape
        assert lengths.shape == (bench_case.n_faces, 3)
    else:  # igl does the flips and the assembly in one call, so its row includes both
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        matrix_igl = bench_case.run(
            lambda: igl.intrinsic_delaunay_cotmatrix(vertices_np, faces_np)[0]
        )
        assert matrix_igl.shape == (bench_case.n_vertices, bench_case.n_vertices)


# Cluster cell widths as a multiple of the mean edge length. 2x welds most 1-rings into one vertex
# (roughly a 4x face reduction) and 6x is aggressive decimation; below 1x almost nothing merges.
_CLUSTER_FACTORS = [2.0, 6.0]


@pytest.mark.noparity(
    "pymeshlab",
    oracle="open3d",
    reason="D2 a differently anchored grid with a different cell representative, and open3d is "
    "the exact oracle right beside it: meshing_decimation_clustering keeps a per-cell "
    "representative on a grid whose origin is not open3d's min_bound - voxel_size / 2, so at the "
    "same threshold it returns measurably different meshes -- 2 792 faces against ordito's 2 768 "
    "at a 0.1 cell on icosphere(4), then 656 against 768 at 0.2 and 156 against 252 at 0.4, i.e. "
    "1% to 62% apart and diverging as the cell grows. open3d assigns cells identically and is "
    "asserted to the exact face and vertex count in "
    "tests/test_remesh.py::test_cluster_decimate_matches_open3d, which also pins the grid anchor; "
    "MeshLab cannot be a second oracle for a quantity open3d already fixes exactly.",
)
@pytest.mark.benchmark(group="cluster_decimate")
@pytest.mark.benchlibs("ordito", "open3d", "pymeshlab", "meshlib")
@pytest.mark.parametrize("cell_factor", _CLUSTER_FACTORS)
def test_cluster_decimate(bench_case: BenchCase, cell_factor: float) -> None:
    """
    Bin, remap, dedup: decimation with no priority queue, against two exact-ish references.

    Its axis spans the wrapper floor and the win: the smallest mesh sits at the floor (see
    ``test_creation::test_box``) and loses, while the largest wins by nearly two orders of
    magnitude. A single point from this group is uninterpretable in either direction.
    """
    voxel_size = cell_factor * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        # ``verticesGridSampling`` stops one step earlier than the other three rows: it returns the
        # *bitset* of one surviving vertex per occupied cell and never rebuilds the faces, so read
        # it as a lower bound on the group. It leaves the mesh alone, so one mesh serves the rounds.
        mesh_part_ml = mm.MeshPart(_mesh_ml(bench_case))
        sampled_ml = bench_case.run(lambda: mm.verticesGridSampling(mesh_part_ml, voxel_size))
        assert 0 < sampled_ml.count() <= bench_case.n_vertices
        return
    if bench_case.kind == "pymeshlab":
        # ``meshing_decimation_clustering`` rewrites the topology, so the MeshSet is rebuilt inside
        # the timed callable. Its ``threshold`` is a length, hence ``PureValue`` fed from the same
        # ``voxel_size`` both other rows get rather than a percentage of its own bbox diagonal.
        skip_larger_than(bench_case, "bunny", "MeshLab's clustering is a serial pass over the grid")
        new_meshset_pml = bench_case.new_meshset_pml

        def cluster_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_decimation_clustering(threshold=ml.PureValue(voxel_size))
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(cluster_pml) >= 0
        return
    if bench_case.kind == "open3d":
        # Returns a new mesh and never touches its input, so the shared mesh is safe across rounds.
        mesh_o3d = bench_case.mesh_o3d
        simplified_o3d = bench_case.run(
            lambda: mesh_o3d.simplify_vertex_clustering(voxel_size=voxel_size)
        )
        assert len(simplified_o3d.triangles) <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _decimated_vertices, decimated_faces = bench_case.run(
        lambda: od.remesh.cluster_decimate(vertices, faces, voxel_size=voxel_size)
    )
    assert int(decimated_faces.shape[0]) // 3 <= bench_case.n_faces


@pytest.mark.benchmark(group="flip_by_objective")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab")
@pytest.mark.parametrize("objective", ["planarity", "curvature"])
def test_flip_by_objective(
    bench_case: BenchCase, objective: Literal["planarity", "curvature"]
) -> None:
    """The Delone engine with another predicate: the same passes, a different candidate set."""
    if bench_case.kind == "pymeshlab":
        if objective == "curvature":
            pytest.skip("MeshLab's curvature flip has no comparable parameterization")
        new_meshset_pml = bench_case.new_meshset_pml

        def flip_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_edge_flip_by_planar_optimization(
                pthreshold=1.0, planartype="area/max side", iterations=1
            )
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(flip_pml, rounds=_ROUNDS) == bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    flipped = bench_case.run(
        lambda: od.remesh.flip_by_objective(vertices, faces, objective=objective), rounds=_ROUNDS
    )
    assert flipped.size == faces.size


# Reduction ratios for the quadric decimator. 0.5 is a mild pass and 0.1 is the ratio MeshLab's own
# dialogue defaults near; the loop count grows as the target falls, which is the shape of this
# group.
#
# That shape is the whole cost: at an aggressive target the call is **over 90 % host**, across
# dozens of geometry rebuilds of ~40 wrapper calls each, and a rebuild costs the same whatever the
# face count -- so the count of wrapper calls per rebuild is the lever, not the kernels and not the
# mesh. Three things priced against that: hoisting the per-pass allocations is worth a few percent
# at most and was not done (a memset that initializes a kernel input cannot be removed by moving the
# allocation); committing several independent sets per rebuild cuts the rebuild count by roughly the
# round count; and grouping the pass's edges **once**, rather than letting the candidate scoring and
# the feature/boundary classification group the same ``3 * n_faces`` rows three times between them,
# is worth nearly as much again, almost all of it ``_classify``.
#
# What moved it most was **capturing the pass**: the pass body's host readbacks were replaced by the
# scans they were reading, every buffer was fixed at a bound the mesh cannot exceed, and one graph
# is replayed for every pass. Several-fold on top, and it makes ordito the fastest of the five on
# every cell. See ``remesh._DecimationBuffers``.
#
# **What is left is the graph itself, not the loop around it** (``saddle_graded`` at 0.1, 44 passes,
# measured on a shared box, min of 9 interleaved reps). The replays are ~84 % of the call; replaying
# them back to back with no readback between is within 2 % of the real loop, and CUDA events around
# each replay agree with that to 0.3 %, so the per-pass 4-byte readback bubble is ~2 % of the call
# and moving the pass loop under ``wp.capture_while`` (which needs every allocation in
# ``_issue_pass`` hoisted, since a conditional body may not allocate) is not worth building. A
# replay takes ~1.5x the summed duration of its ~110 kernel, memset and memcpy nodes measured
# uncaptured, so a third of it is inter-node gaps inside the graph: the lever now is fewer nodes
# per pass (fusion inside ``_issue_pass`` and ``_run_collapse_rounds``). The remaining ~14 % is the
# issued pass 0 and the recorded pass 1, which cost about the same as each other and ~3x a replay.
_QUADRIC_RATIOS = [0.5, 0.1]


@pytest.mark.benchmark(group="quadric_decimate")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "igl", "open3d", "pymeshlab", "pyvista", "meshlib")
@pytest.mark.parametrize("target_ratio", _QUADRIC_RATIOS)
def test_quadric_decimate(bench_case: BenchCase, target_ratio: float) -> None:
    """
    Greedy quadric collapses to a face budget: batched independent sets against four queues.

    ``PolyData.decimate`` is ``vtkDecimatePro``, the reference with the least surface deviation on
    the test sphere (``tests/test_remesh.py`` carries the comparison). It takes a
    *reduction fraction* where the other four take a face count, so the ratio is
    converted rather than the count passed.
    """
    target_faces = max(4, int(target_ratio * bench_case.n_faces))
    if bench_case.kind == "meshlib":
        # ``decimateMesh`` takes a *deleted*-face budget, not a target, so the count is converted;
        # it lands on the requested face count exactly (``tests/test_remesh.py``). ``packMesh=True``
        # is inside the timed region on purpose -- without it the result cannot be read at all, so
        # it is part of what the operation costs here. It mutates, hence the rebuild per round.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        n_faces = bench_case.n_faces

        def decimate_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.DecimateSettings()
            settings_ml.maxDeletedFaces = n_faces - target_faces
            settings_ml.packMesh = True
            mm.decimateMesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(decimate_ml, rounds=_ROUNDS) > 0
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        decimated_pv = bench_case.run(lambda: mesh_pv.decimate(1.0 - target_ratio), rounds=_ROUNDS)
        assert decimated_pv.n_faces > 0
        return
    if bench_case.kind == "pymeshlab":
        # ``autoclean=True`` (the default) deletes unreferenced vertices, so this filter is not
        # idempotent even in geometry and the MeshSet has to be rebuilt inside the timed callable.
        new_meshset_pml = bench_case.new_meshset_pml

        def decimate_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(decimate_pml, rounds=_ROUNDS) > 0
        return
    if bench_case.kind == "igl":
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        result_igl = bench_case.run(
            lambda: igl.decimate(vertices_np, faces_np, target_faces), rounds=_ROUNDS
        )
        assert np.asarray(result_igl[1]).shape[0] > 0
        return
    if bench_case.kind == "open3d":
        # Returns a new mesh and never touches its input, so the shared mesh is safe across rounds.
        mesh_o3d = bench_case.mesh_o3d
        simplified_o3d = bench_case.run(
            lambda: mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_faces),
            rounds=_ROUNDS,
        )
        assert len(simplified_o3d.triangles) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _decimated_vertices, decimated_faces = bench_case.run(
        lambda: od.remesh.quadric_decimate(vertices, faces, target_faces=target_faces),
        rounds=_ROUNDS,
    )
    assert decimated_faces.size // 3 <= bench_case.n_faces
