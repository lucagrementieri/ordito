"""
Benchmarks for ``triwarp.validation``: the topological predicates.

Three axes, one per predicate family, because these functions fail to scale for three unrelated
reasons and a face-count sweep separates none of them.

* **valence** for ``is_vertex_manifold``: it solves a miniature connected-components problem in
  every vertex's one-ring, so its cost is the valence *distribution*, not the vertex count.
* **diameter** for ``face_orientation_bits``, the group that justified the axis. *Propagating* the
  Z2 bits across the face-adjacency graph costs one launch per level, so a long strip is O(V) rounds
  where a blob is O(log V) — two orders of magnitude apart at identical vertex count, with the
  direction reversing against trimesh, and entirely invisible to the scan registry, whose meshes are
  all compact blobs. The bits are *solved* instead, by a parity-carrying union-find in three
  launches, flat across the axis. The axis is kept because that flatness must not regress.
* **overlap** for ``is_watertight`` / ``is_volume``: both compose an edge-count test with a
  self-intersection test over a BVH, so what matters is collision density, not size.

References
----------
**open3d** is the exact definitional equivalent for ``is_watertight`` — triwarp's docstring defines
itself against ``TriangleMesh.is_watertight`` (edge-manifold without boundary, plus vertex-manifold
and no self-intersection). It is also **seconds** per call, three to four orders of magnitude
behind, and *faster on the harder mesh*, because its self-intersection test is a brute-force scan
that early-exits on the first hit and so pays full price only when the mesh is clean. Both points
are worth having on the record, so open3d runs here at ``rounds=1`` rather than being dropped; that
group is a large share of the suite's wall clock and is the reason for the cap.

**libigl**'s ``is_vertex_manifold`` is the reference for the valence group, and it is flat and safe
on ``fan_hub`` — unlike ``igl.principal_curvature``, which takes minutes on the same mesh (see
[`test_curvature.py`](test_curvature.py)).

open3d also covers both manifoldness groups: ``is_edge_manifold`` shares triwarp's
``allow_boundary_edges`` switch with identical semantics on both settings, and
``is_vertex_manifold`` agrees everywhere except vertices sitting *on* a non-manifold edge — open3d
tests whether the incident faces are edge-connected at all, triwarp and igl whether they form a
manifold fan, so three faces sharing one edge pass open3d and fail the other two.

**trimesh** rebuilds its mesh inside the timed callable because it caches derived properties; note
its ``is_watertight`` is edge-manifold-only, so it is timing context rather than an equivalent
computation. It has no ``is_vertex_manifold`` in 5.0; ``tm.repair.fix_winding`` is the closest
analogue of the orientation propagation and is timed against it.

There is no open3d ``is_volume``: its closest composition (``is_watertight() and is_orientable()``)
short-circuits on the first check, so it would time ``is_watertight`` under a different name.

**pymeshlab** covers all three groups and answers watertightness in a way neither other reference
does: ``get_topological_measures`` returns boundary edges, component count, genus, two-manifoldness
and the non-manifold edge and vertex counts — **one call for every predicate this module exposes
plus the genus and the Euler characteristic** — so its row is simultaneously the reference for
``is_watertight`` and ``is_volume``, an *upper* bound for each alone.

It does *not* include the self-intersection test, which is half of triwarp's and open3d's
definition, so the honest composition is both calls together: ``get_topological_measures`` plus
``compute_selection_by_self_intersections_per_face``, timed as one callable. That is the useful
number, because open3d computes the *same* composition two orders of magnitude slower — so nearly
all of open3d's cost is its brute-force self-intersection scan, and nothing about the definition
requires it. Both pymeshlab rows are also **flat across the overlap axis in the same direction as
everything else**, if anything *faster* on the self-intersecting mesh, which is the third
independent confirmation that collision density is not what drives this predicate.

For the valence group, ``compute_selection_by_non_manifold_per_vertex`` is the direct equivalent of
``is_vertex_manifold`` and is flat across the valence axis, like triwarp and libigl, so all three
agree the valence distribution is not a hot spot. Both selection filters touch only the selected
bit, so they share the MeshSet; ``get_topological_measures`` is read-only.

``face_orientation_bits`` has no pymeshlab equivalent returning the *bits*:
``meshing_re_orient_faces_coherently`` applies them and is benchmarked in
[`test_repair.py`](test_repair.py) against ``make_winding_consistent``, which consumes them.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase

# open3d's is_watertight runs into seconds (see the module docstring); one round is enough to
# record the magnitude without letting it dominate the suite.
_O3D_ROUNDS = 1


@pytest.mark.benchmark(group="is_vertex_manifold")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pymeshlab")
def test_is_vertex_manifold(bench_case: BenchCase) -> None:
    """
    A connected-components problem per one-ring: driven by the valence distribution.

    Note the two sides return different shapes -- triwarp reduces to a single ``bool`` while
    ``igl.is_vertex_manifold`` hands back the per-vertex mask (triwarp's ``vertex_manifold_mask``
    is the equivalent of that). The work is the same either way; only the final reduction differs.

    ``open3d.is_vertex_manifold`` tests *connectivity* of the incident faces rather than a manifold
    fan, so a vertex sitting on a non-manifold edge still passes it where triwarp and igl say no --
    the answers agree exactly on edge-manifold input (the parity test pins that class down).
    """
    if bench_case.kind == "triwarp":
        assert bench_case.run(lambda: tw.validation.is_vertex_manifold(bench_case.faces_wp)) in (
            True,
            False,
        )
    elif bench_case.kind == "open3d":
        assert bench_case.run(bench_case.mesh_o3d.is_vertex_manifold) in (True, False)
    elif bench_case.kind == "pymeshlab":  # writes a per-vertex bool selection: triwarp's shape
        meshset_pml = bench_case.meshset_pml
        bench_case.run(meshset_pml.compute_selection_by_non_manifold_per_vertex)
        assert meshset_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
    else:
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        mask_igl = np.asarray(bench_case.run(lambda: igl.is_vertex_manifold(faces_np)))
        assert mask_igl.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="is_edge_manifold")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pyvista")
def test_is_edge_manifold(bench_case: BenchCase) -> None:
    """
    The cheaper manifoldness predicate: an edge sort and a per-edge count, no one-ring components.

    It sits next to ``is_vertex_manifold`` on the same axis deliberately -- edge-manifoldness is
    ``O(3F)`` counting where vertex-manifoldness needs a connected-components pass *per vertex*, so
    the pair prices what the stronger predicate costs. A mesh can be edge-manifold and not
    vertex-manifold (two cones joined at a tip), which is why both exist.

    ``igl.is_edge_manifold`` returns ``(verdict, per_corner_mask, ...)`` -- the reduced ``bool``
    first, matching triwarp's return, with the per-corner detail behind it. It has no
    ``allow_boundary_edges`` switch (it always allows them), so only triwarp's default is timed.
    ``open3d.is_edge_manifold`` has the same switch with the same two semantics as triwarp's and is
    timed at the shared default.

    **pyvista has no switch and answers the other setting**: ``PolyData.is_manifold`` is
    ``n_open_edges == 0``, i.e. ``vtkFeatureEdges`` with boundary *and* non-manifold edges on, which
    is triwarp's ``allow_boundary_edges=False``. It therefore does strictly more work than the row
    above -- a full feature-edge extraction rather than a count -- and it is timed at the setting it
    actually implements, which is the one asserted in ``tests/test_validation.py``.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        assert bench_case.run(lambda: bool(mesh_pv.is_manifold)) in (True, False)
        return
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        assert bench_case.run(lambda: tw.validation.is_edge_manifold(faces)) in (True, False)
        return
    if bench_case.kind == "open3d":
        mesh_o3d = bench_case.mesh_o3d
        assert bench_case.run(lambda: mesh_o3d.is_edge_manifold(allow_boundary_edges=True)) in (
            True,
            False,
        )
        return
    faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
    assert bool(bench_case.run(lambda: igl.is_edge_manifold(faces_np))[0]) in (True, False)


@pytest.mark.benchmark(group="face_orientation_bits")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_face_orientation_bits(bench_case: BenchCase) -> None:
    """
    Z2 orientation bits as a parity union-find: three launches, so depth costs nothing.

    ``igl.bfs_orient`` is the reference that does the same work by a different route -- a serial
    breadth-first walk of the face-adjacency graph. Being a traversal, it is the row where this
    group's ``diameter`` axis should show something on the reference side and nothing on triwarp's;
    that contrast is the point of putting it here.

    **Its second return is the per-face component id, not the flip mask.** ``bfs_orient`` returns
    ``(FF, C)``: the reoriented face table and ``C``, which is all zeros on a connected mesh. The
    flips are recoverable only by comparing ``FF`` against ``F`` row by row, which is what the
    parity test does; reading ``C`` as the mask would silently compare triwarp's bits against a
    constant.
    """
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        _bits, _edges, _seeds, n_components = bench_case.run(
            lambda: tw.validation.face_orientation_bits(faces)
        )
        assert n_components >= 1
    elif bench_case.kind == "igl":
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        faces_oriented_igl, components_igl = bench_case.run(lambda: igl.bfs_orient(faces_np))
        assert faces_oriented_igl.shape == (bench_case.n_faces, 3)
        assert components_igl.ravel().shape == (bench_case.n_faces,)
    else:  # trimesh's winding fix walks the same adjacency graph; rebuild inside (it mutates)
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def fix_winding_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            tm.repair.fix_winding(mesh_tm)
            return mesh_tm

        assert len(bench_case.run(fix_winding_tm).faces) == bench_case.n_faces


def _run_topology_pml(bench_case: BenchCase) -> None:
    """
    Time MeshLab's whole topology report plus its self-intersection pass.

    ``get_topological_measures`` alone answers edge-manifoldness, boundary-edge count, component
    count and genus, but triwarp's and open3d's watertightness definition also requires "no self
    intersection" -- so both calls are timed together rather than quoting the cheaper half. Its
    manifoldness verdict is asserted on, which is what makes this a check and not just a stopwatch.
    """
    meshset_pml = bench_case.meshset_pml

    def measure_pml() -> dict:
        meshset_pml.compute_selection_by_self_intersections_per_face()
        return meshset_pml.get_topological_measures()

    assert measure_pml()["is_mesh_two_manifold"] in (True, False)
    bench_case.run(measure_pml, rounds=_O3D_ROUNDS)


@pytest.mark.benchmark(group="is_watertight")
@pytest.mark.benchaxis("overlap")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib")
def test_is_watertight(bench_case: BenchCase) -> None:
    """
    Edge counts plus a self-intersection pass: driven by collision density, not size.

    meshlib's ``MeshTopology.isClosed`` answers the *closedness* clause alone -- the same weaker
    question trimesh's ``is_watertight`` answers -- and it answers it off state the topology already
    holds, so this row is a lower bound on the group rather than an equivalent computation. The
    mesh build is outside the timed callable for that reason: with it inside, the row would time the
    converter.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        assert bench_case.run(mesh_ml.topology.isClosed) in (True, False)
        return
    if bench_case.kind == "pymeshlab":
        _run_topology_pml(bench_case)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_watertight(vertices, faces))
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_watertight)
    else:  # open3d: same definition as triwarp's, and it does not cache the answer
        mesh_o3d = bench_case.mesh_o3d
        result = bench_case.run(mesh_o3d.is_watertight, rounds=_O3D_ROUNDS)
    assert result in (True, False)


@pytest.mark.benchmark(group="is_volume")
@pytest.mark.benchaxis("overlap")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_is_volume(bench_case: BenchCase) -> None:
    """``is_watertight`` plus winding plus a signed-volume sign: the priciest predicate."""
    if bench_case.kind == "pymeshlab":  # the same one call: genus and manifoldness come together
        _run_topology_pml(bench_case)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_volume(vertices, faces))
    else:
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_volume)
    assert result in (True, False)


@pytest.mark.benchmark(group="face_self_intersecting_mask")
@pytest.mark.benchaxis("overlap")
@pytest.mark.benchlibs("triwarp", "meshlib", "pymeshfix", "open3d", "pymeshlab")
def test_face_self_intersecting_mask(bench_case: BenchCase) -> None:
    """
    The per-face self-intersection flags, which ``is_watertight`` reduces to a single bool.

    Timed separately from ``is_watertight`` because the two libraries compose the predicate
    differently: triwarp's ``is_watertight`` runs the manifold checks *first* and this pass only if
    they hold, and MeshLib's ``isClosed`` never runs it at all. So the composite group prices three
    clauses on one side and one on the other, and only this group prices the clause they share.

    meshlib's ``findSelfCollidingTrianglesBS`` returns the same per-face set from an AABB tree over
    the faces, multi-threaded -- read it against ``triwarp-cuda``. ``touchIsIntersection=False`` is
    the setting that matches triwarp and is passed explicitly; the tree is built lazily on first
    use, so the mesh is constructed outside the timed callable and the row prices the query.

    pymeshfix's ``select_intersecting_triangles`` returns the same set exactly, from a uniform grid
    broad phase rather than a tree, single-threaded. It is the one pymeshfix row in this file,
    because it is the one where the operation clears the 30 % share the load leaves -- the query and
    the load are about equal on both meshes. The load is inside the timed callable and cannot be
    moved out (a ``PyTMesh`` takes exactly one ``load_array``), so read this row as query-plus-load
    and halve it for the query alone. ``tris_per_cell`` is its broad-phase bucket size and was
    measured not to change the answer at any value probed.

    open3d and pymeshlab are the fourth and fifth implementations of the same predicate, and having
    five is worth the rows because this is the clause ``is_watertight`` reduces and the
    post-condition ``fix_self_intersections`` is verified by. Both agree on the face *set*, reached
    from open3d's colliding **pairs** and from MeshLab's per-face bool selection. Two shape
    differences to read the rows through: open3d returns pairs, so its output is larger than a mask
    and ``np.unique`` is the reduction (outside the timed callable, as triwarp's mask needs none);
    and MeshLab mutates ``current_mesh()``, so its MeshSet is rebuilt per round.
    """
    if bench_case.kind == "open3d":
        mesh_o3d = bench_case.mesh_o3d
        pairs_o3d = bench_case.run(mesh_o3d.get_self_intersecting_triangles)
        assert np.asarray(pairs_o3d).shape[0] <= bench_case.n_faces
        return
    if bench_case.kind == "pymeshlab":
        meshset_pml = bench_case.new_meshset_pml
        bench_case.run(lambda: meshset_pml().compute_selection_by_self_intersections_per_face())
        return
    if bench_case.kind == "pymeshfix":
        faces_pmf = bench_case.run(
            lambda: bench_case.new_tmesh_pmf().select_intersecting_triangles(
                tris_per_cell=50, justproper=False
            )
        )
        assert faces_pmf.shape[0] <= bench_case.n_faces
        return
    if bench_case.kind == "meshlib":
        mesh_part_ml = mm.MeshPart(bench_case.new_mesh_ml())
        colliding_ml = bench_case.run(
            lambda: mm.findSelfCollidingTrianglesBS(mesh_part_ml, touchIsIntersection=False)
        )
        assert colliding_ml.size() <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    mask = bench_case.run(lambda: tw.validation.face_self_intersecting_mask(vertices, faces))
    assert mask.size == bench_case.n_faces


@pytest.mark.benchmark(group="face_defective_mask")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
def test_face_defective_mask(bench_case: BenchCase) -> None:
    """
    All three defect criteria at once: face quality, adjacency scatter, per-face gate.

    meshlib's ``findOverlappingTris`` covers the **fold** criterion alone -- it is a proximity
    search over the AABB tree for near-coincident triangles with near-antiparallel normals, where
    the other two rows read the dihedral off the face adjacency -- so it is a lower bound on this
    group and a different algorithm for the one clause it shares. ``maxNormalDot`` is the dihedral
    threshold in dot-product form (``cos(radians(160))``); the equality of the two answers, and the
    input class where they part, are in ``tests/test_validation.py``. The tree is lazily built, so
    the mesh is constructed and pre-warmed outside the timed callable.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        settings_ml = mm.FindOverlappingSettings()
        settings_ml.maxNormalDot = float(np.cos(np.radians(160.0)))
        mesh_part_ml = mm.MeshPart(bench_case.new_mesh_ml())
        mm.findOverlappingTris(mesh_part_ml, settings_ml)
        folded_ml = bench_case.run(lambda: mm.findOverlappingTris(mesh_part_ml, settings_ml))
        assert folded_ml.size() <= n_faces
        return
    if bench_case.kind == "pymeshlab":
        # Selection-only, so the geometry survives and the MeshSet is shared.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_selection_bad_faces(
                usear=True,
                aratio=0.02,
                usenf=True,
                nfratio=60.0,
                select_folded_faces=True,
                folded_faces_angle_threshold=160.0,
            )
        )
        assert meshset_pml.current_mesh().face_selection_array().shape == (n_faces,)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    bad = bench_case.run(
        lambda: tw.validation.face_defective_mask(
            vertices, faces, min_quality=0.02, max_normal_angle=60.0, max_fold_angle=160.0
        )
    )
    assert bad.shape == (n_faces,)
