"""
Benchmarks for ``ordito.boundary.boundary_loops``.

Axis: **loops**. Boundary extraction is not driven by mesh size -- ``sphere_med`` has 81 920 faces
and no boundary at all, and costs less than the 1 024-face ``rim_short``. What it is driven by is
the *shape* of the boundary, along two independent directions that the axis separates:

- **loop length** decides the ranking work. ``rim_long``'s two rims of 65 536 vertices are the
  asymptotic case: a per-vertex successor walk is O(L^2) on a loop of length L, while the Wyllie
  pointer jumping the implementation uses is O(L log L), one kernel launch per round.
- **loop count** decides the host work. Per loop, a slice of a read-back offset table plus a
  ``wp.clone`` makes ``holes_many``'s 512 three-vertex loops *more* expensive than ``rim_long``'s
  two enormous ones despite a quarter of the boundary vertices. ``boundary_loops`` is a slicing
  wrapper over the packed ``boundary_loops_with_offsets`` for that reason.

The spread over an unchanged face count is the point, and it is what identified the per-loop host
sequence rather than the ranking as the thing to batch: with the wrong end on top this group runs an
order of magnitude across the axis.

References
----------
**trimesh**'s ``Trimesh.outline()`` is the equivalent and is rebuilt inside the timed callable,
since it caches its internals on the mesh. **libigl**'s ``boundary_loop`` returns only the longest
loop, so it is doing strictly less work on ``holes_many`` -- noted rather than corrected, because
the alternative is not comparing against it at all.

**open3d** has no boundary-loop extraction: it can report *which* edges are boundary edges
(``get_non_manifold_edges(allow_boundary_edges=False)``) but never orders them into loops, which is
the whole cost of this function.

**pymeshlab** is in the same position and appears in the ``boundary_edges`` group only:
``compute_selection_from_mesh_border`` marks the boundary *vertices* as a bool selection array,
which is the same find-the-boundary pass stopped one step short of an edge list -- and it has
nothing that orders the result into loops either. It is read-only apart from the selection bit, so
it shares the MeshSet.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase


@pytest.mark.benchmark(group="boundary_loops")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("ordito", "trimesh", "igl", "meshlib")
def test_boundary_loops(bench_case: BenchCase) -> None:
    """
    Ranking plus batched extraction, across no boundary / two long rims / many short loops.

    The four references return four different things and only ordito returns every loop as an
    ordered vertex list: ``igl.boundary_loop`` gives the longest loop alone, ``Trimesh.outline()``
    gives a ``Path3D``, and meshlib gives one ``EdgeId`` per hole that the caller must walk. The
    meshlib row therefore includes the ``getLeftRing`` walk, without which it would be timing
    hole *detection* only and would not be the same quantity -- but the walk is a Python loop over
    the ring, so read the row as an upper bound on MeshLib's own cost rather than as its floor.
    The mesh is built inside the timed callable because the ring walk is the first thing to touch
    the topology's edge structure, which is lazily built and cached like the AABB tree.
    """
    if bench_case.kind == "meshlib":

        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            total = 0
            for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
                total += len(mesh_ml.topology.getLeftRing(edge_ml))
            return total

        bench_case.run(run_ml)
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loops = bench_case.run(lambda: od.boundary.boundary_loops(vertices, faces))
        assert isinstance(loops, list)
    elif bench_case.kind == "trimesh":  # rebuild inside: trimesh caches outline internals
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).outline())
        assert result is not None
    else:  # igl returns only the longest loop
        faces = bench_case.faces_np
        bench_case.run(lambda: igl.boundary_loop(faces))


@pytest.mark.benchmark(group="boundary_edges")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("ordito", "trimesh", "igl", "pymeshlab", "pyvista", "meshlib")
def test_boundary_edges(bench_case: BenchCase) -> None:
    """
    The unordered predecessor of ``boundary_loops``: the edge sort without the ranking.

    Subtracting this group from ``boundary_loops`` separates the two costs, which is what says
    whether a regression is in the sort (scales with faces) or in the loop extraction (scales with
    loop count). It reads flat across the whole axis, so everything above it in
    ``boundary_loops`` is ranking and extraction.

    ``igl.boundary_facets`` is the one reference here that returns the same *thing* ordito does --
    an ``(n_boundary, 2)`` edge list, plus the incident face and corner indices as second and third
    returns, which is strictly more than ordito's two columns. It is the right row for this group
    and not for ``boundary_loops``, where ``igl.boundary_loop`` returns only the longest loop.
    """
    if bench_case.kind == "meshlib":
        # ``getBoundaryVerts`` stops one step earlier than ordito too, marking the boundary
        # *vertices* rather than returning the pairs -- the same relationship pymeshlab's row has,
        # and pinned in tests/test_boundary.py::test_boundary_vertex_indices. Pure (it reads the
        # topology and allocates a bitset), so the mesh is built once outside the timed callable;
        # the build would otherwise dominate a sub-millisecond pass.
        mesh_ml = bench_case.new_mesh_ml()
        n_vertices = bench_case.n_vertices
        verts_ml = bench_case.run(lambda: mm.getBoundaryVerts(mesh_ml.topology))
        assert verts_ml.size() <= n_vertices
        return
    if bench_case.kind == "pyvista":
        # ``extract_feature_edges`` with only the boundary class on -- the same find-the-boundary
        # pass, returning a line-cell PolyData. Its ``n_open_edges`` shortcut is *not* this
        # quantity: it counts non-manifold edges too (see tests/test_boundary.py).
        mesh_pv = bench_case.mesh_pv
        edges_pv = bench_case.run(
            lambda: mesh_pv.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )
        )
        assert edges_pv.n_cells >= 0
        return
    if bench_case.kind == "igl":
        faces_np = bench_case.faces_np
        edges_igl, _face_igl, _corner_igl = bench_case.run(lambda: igl.boundary_facets(faces_np))
        assert edges_igl.ndim == 2
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        edges = bench_case.run(lambda: od.boundary.boundary_edges(vertices, faces))
        assert edges.ndim == 2
    elif bench_case.kind == "pymeshlab":
        # ``compute_selection_from_mesh_border`` marks the boundary *vertices* rather than returning
        # the edge pairs, so it does the same find-the-boundary pass and stops one step earlier.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(meshset_pml.compute_selection_from_mesh_border)
        assert meshset_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
    else:  # the pure trimesh.grouping path, not a cached Trimesh property
        faces_np = bench_case.faces_np

        def boundary_tm() -> np.ndarray:
            edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)
            return edges_np[tm.grouping.group_rows(edges_np, require_count=1)]

        assert bench_case.run(boundary_tm).ndim == 2


@pytest.mark.benchmark(group="ears")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("ordito", "igl")
def test_ears(bench_case: BenchCase) -> None:
    """
    Ear triangles -- two boundary edges each -- as ``(face, opposite corner)`` pairs.

    ``holes`` uses these to recognise a rim that closes with a single triangle, so the axis
    is the boundary shape rather than the mesh size, like the rest of the module. ``igl.ears``
    returns the identical pair of arrays, which is unusual enough to note: this is one of the few
    groups where ordito and igl agree on the *output convention* and not merely the quantity, so
    the parity assert in ``tests/test_boundary.py`` needs only a row sort.
    """
    if bench_case.kind == "ordito":
        faces = bench_case.faces_wp
        ears, opposite = bench_case.run(lambda: od.boundary.ears(faces))
        assert ears.shape == opposite.shape
        return
    faces_np = bench_case.faces_np
    ears_igl, opposite_igl = bench_case.run(lambda: igl.ears(faces_np))
    assert ears_igl.shape == opposite_igl.shape


@pytest.mark.benchmark(group="loop_perimeters")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_loop_perimeters(bench_case: BenchCase) -> None:
    """
    Both loop measures at once: one segmented launch over every rim of the mesh.

    The cost is the *packing* plus one launch, not the loops themselves -- which is the claim, since
    meshlib answers one hole per call and the scan meshes carry many. Its ``holePerimeter`` takes a
    representative edge, so its row is a Python loop over ``findHoleRepresentiveEdges`` and grows
    with the hole count where ordito's does not; that is the shape of the comparison rather than a
    handicap, since there is no batched entry point to call instead.

    The loops are produced **outside** the timed callable on both sides: ``boundary_loops`` has its
    own group and would otherwise dominate this one.

    **And the packing was nearly all of it, re-packing a buffer that was already packed.** On a
    many-rim mesh ``pack_1d_arrays`` -- one ``warp.copy`` per rim -- dwarfed the single launch the
    function exists for, at hundreds of launches and about half of them host. But
    ``boundary_loops`` produced those rims by *slicing* one
    packed buffer, so the pack rebuilt what the caller already held. ``pack_1d_arrays(copy=False)``
    The two small rows are the honest limit of it: at five rims the pack was never the cost, and
    what is left is launch marshalling and allocation against meshlib's cached-topology arithmetic.
    They are floor rows and should not be optimized further.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
        if not len(holes_ml):
            pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to measure")

        def measure_ml() -> float:
            return sum(
                mm.holePerimeter(mesh_ml.topology, mesh_ml.points, edge) for edge in holes_ml
            )

        assert bench_case.run(measure_ml) > 0.0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = od.boundary.boundary_loops(vertices, faces)
    if not loops:
        pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to measure")
    perimeters = bench_case.run(lambda: od.boundary.loop_perimeters(vertices, loops))
    assert int(perimeters.shape[0]) == len(loops)


@pytest.mark.benchmark(group="loop_directed_areas")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_loop_directed_areas(bench_case: BenchCase) -> None:
    """
    The vector measure of the same loops, read against the scalar one above.

    Same packing, same launch shape, one cross product per segment instead of one length -- so the
    gap between the two groups is what the extra arithmetic costs and nothing else. meshlib's
    ``holeDirArea`` is again per hole.

    That shared packing is why this row moved with its sibling and by the same factor when
    ``pack_1d_arrays`` learned to reuse an already-packed buffer. Read the measurement in
    ``test_loop_perimeters`` above; there
    is nothing separate to say about this one, which is the point of the pair.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
        if not len(holes_ml):
            pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to measure")

        def areas_ml() -> int:
            return len(
                [mm.holeDirArea(mesh_ml.topology, mesh_ml.points, edge) for edge in holes_ml]
            )

        assert bench_case.run(areas_ml) == len(holes_ml)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = od.boundary.boundary_loops(vertices, faces)
    if not loops:
        pytest.skip(f"{bench_case.mesh_name} is closed: there is no rim to measure")
    areas = bench_case.run(lambda: od.boundary.loop_directed_areas(vertices, loops))
    assert int(areas.shape[0]) == len(loops)
