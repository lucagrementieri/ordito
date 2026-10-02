"""
Benchmarks for ``ordito.adjacency``: the face-pair table and the two quantities derived from it.

``face_adjacency`` is the package's most reused topological query and the widest composition in it:
an edge build, a row hash, a radix sort over ``3F`` keys, a scan-and-compact, and a gather. Two of
those steps read back to the host -- the compaction's output size, which is data-dependent and
unavoidable, and the hash radix, which is *not* when the caller knows the vertex count. That second
one is the ``n_vertices`` axis below: same answer, one less serialisation point, and the gap
between the two ids is what that keyword buys.

``face_adjacency_unshared`` and ``face_adjacency_angles`` are each one launch over an existing
adjacency table, so they are timed with the table precomputed. Timing them from raw faces would
just re-measure ``face_adjacency`` three times.

trimesh is the host reference for all three. Its ``Trimesh`` properties are cached, so each round
builds a fresh mesh; ``benchmarks/test_mesh.py`` covers the warm-cache side under
``mesh_face_adjacency`` and is deliberately a separate group.

**libigl** covers the first two groups with a single call: ``igl.triangle_triangle_adjacency``
returns ``(TT, TTi)``, the neighbour across each corner's edge and that edge's index in the
neighbour, which is ordito's ``face_adjacency`` and ``face_adjacency_unshared`` in one pass and one
layout. Both rows therefore time the *same* igl call -- that is not double-counting, it is the
honest statement that igl does not separate them, and it makes ``face_adjacency_unshared``'s igl row
an upper bound rather than a like-for-like. ``face_adjacency_angles`` has no igl equivalent
(``igl.dihedral_angles`` is per-tet, on a tetrahedral mesh, not a surface).

Local convexity
---------------
``face_adjacency_projections`` and ``face_adjacency_convex`` (moved here with the ``convex``
module's adjacency half) are one thread per adjacent face pair projecting the neighbour's unshared
vertex onto the first face's plane. The arithmetic is trivial; the cost is almost entirely the
*adjacency
construction* that precedes it when the optional ``face_adjacency`` arguments are not supplied. Both
are timed the way a caller who has nothing precomputed would call them, so these numbers are
"adjacency + projection" and it is the adjacency that dominates -- which is also why they sit in
this file rather than with the precomputed-table groups above. ``face_adjacency_convex`` is
``face_adjacency_projections`` plus a threshold, so the delta between the two groups is the
comparison pass alone.

**trimesh** is a genuine baseline for *both*: ``Trimesh.face_adjacency_projections`` is the same
scalar per adjacent face pair in the same layout, and ``Trimesh.face_adjacency_convex`` is that plus
the threshold. Both are *cached properties*, so the ``Trimesh`` is rebuilt inside the timed callable
-- otherwise rounds 2..n would return a memoized array and measure nothing. That rebuild also pays
trimesh's own ``face_adjacency`` construction, which is the honest comparison since the ordito side
builds its adjacency inside the timed region too. Both rows are capped at ``bunny``; the projection
reference is two orders of magnitude behind ordito at the top of the scale axis. **libigl** has no
local-convexity binding, so igl is absent from these two.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, skip_larger_than

_adjacency_cache: dict[tuple[str, str], tuple[odt.Array2dInt32, odt.Array2dInt32]] = {}


def _adjacency(bench_case: BenchCase) -> tuple[odt.Array2dInt32, odt.Array2dInt32]:
    """Precomputed ``(face_adjacency, face_adjacency_edges)`` for the derived-quantity groups."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _adjacency_cache:
        _adjacency_cache[key] = od.adjacency.face_adjacency(
            bench_case.faces_wp, return_edges=True, n_vertices=bench_case.n_vertices
        )
    return _adjacency_cache[key]


@pytest.mark.benchmark(group="face_adjacency")
@pytest.mark.benchlibs("ordito", "trimesh", "igl")
@pytest.mark.parametrize("known_radix", [False, True], ids=["inferred", "known_nv"])
def test_face_adjacency(bench_case: BenchCase, known_radix: bool) -> None:
    """
    Manifold face pairs, with the hash radix inferred against supplied.

    ``inferred`` pays a ``reduce.minmax`` over the edge rows and the host readback that ends it;
    ``known_nv`` passes the vertex count instead and skips both. The answer is identical either
    way, so the whole gap is that one serialisation point -- if it does not show, the keyword is
    not worth threading through callers.

    ``igl.triangle_triangle_adjacency`` answers the same question in a **per-corner** layout: a
    ``(n_faces, 3)`` table whose entry ``[f, i]`` is the face across edge ``i`` of face ``f``, or
    ``-1``. That is a superset of ordito's pair list -- every pair appears twice, once from each
    side -- so the row is a fair cost comparison and the parity assert carries the pair-extraction
    transform (``tests/test_adjacency.py``). Note the array form is the one to use: the
    ``triangle_triangle_adjacency_lists`` variant is the same computation returning
    ``list[list[int]]`` and costs two orders of magnitude more, i.e. it would price nanobind rather
    than the algorithm.
    """
    if bench_case.kind == "ordito":
        faces_wp = bench_case.faces_wp
        n_vertices = bench_case.n_vertices if known_radix else None
        adjacency = bench_case.run(
            lambda: od.adjacency.face_adjacency(faces_wp, n_vertices=n_vertices)
        )
        assert adjacency.shape[1] == 2
        assert adjacency.shape[0] > 0
        return
    if known_radix:
        pytest.skip("neither reference has a radix-hint equivalent")
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if bench_case.kind == "igl":
        adjacency_igl, _corner_igl = bench_case.run(
            lambda: igl.triangle_triangle_adjacency(faces_np)
        )
        assert adjacency_igl.shape == (bench_case.n_faces, 3)
        return
    adjacency_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency
    )
    assert len(adjacency_tm) > 0


@pytest.mark.benchmark(group="face_adjacency_unshared")
@pytest.mark.benchlibs("ordito", "trimesh", "igl")
@pytest.mark.parametrize("tabled", [True, False], ids=["tabled", "from_faces"])
def test_face_adjacency_unshared(bench_case: BenchCase, tabled: bool) -> None:
    """
    The off-edge corner of each adjacent face, with and without the adjacency tables.

    ``tabled`` is one launch over a precomputed ``(face_adjacency, face_adjacency_edges)`` pair and
    is the floor for this operation. ``from_faces`` starts from the face buffer alone, which is what
    a caller who does not already hold those tables pays: the grouping is unavoidable, but the
    tables themselves are not -- the shared edge and both owning faces are recoverable from the
    grouped edge indices. The gap between this id and ``face_adjacency`` + ``tabled`` is what that
    saves.
    """
    if bench_case.kind == "ordito":
        faces_wp = bench_case.faces_wp
        if tabled:
            adjacency, adjacency_edges = _adjacency(bench_case)
            unshared = bench_case.run(
                lambda: od.adjacency.face_adjacency_unshared(
                    faces_wp, face_adjacency=adjacency, face_adjacency_edges=adjacency_edges
                )
            )
        else:
            n_vertices = bench_case.n_vertices
            unshared = bench_case.run(
                lambda: od.adjacency.face_adjacency_unshared(faces_wp, n_vertices=n_vertices)
            )
        assert unshared.shape[1] == 2
        return
    if not tabled:
        pytest.skip("neither reference exposes a table-free path")
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if bench_case.kind == "igl":
        # TTi is the *corner* index of the shared edge in the neighbour, from which the unshared
        # vertex follows as F[g, (j + 2) % 3] -- so igl computes the same information as part of the
        # adjacency pass rather than as a second step, and this row is the whole pass.
        _adjacency_igl, corner_igl = bench_case.run(
            lambda: igl.triangle_triangle_adjacency(faces_np)
        )
        assert corner_igl.shape == (bench_case.n_faces, 3)
        return
    unshared_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_unshared
    )
    assert len(unshared_tm) > 0


@pytest.mark.benchmark(group="face_connected_component_labels")
@pytest.mark.benchlibs("ordito", "igl", "pyvista", "meshlib")
def test_face_connected_component_labels(bench_case: BenchCase) -> None:
    """
    Per-face component ids over the face-adjacency graph: the edge build plus a label propagation.

    ``validation.face_orientation_bits`` calls this pair, and ``benchmarks/test_graph.py`` times it
    on the ``diameter`` axis where the question is how the propagation scales with graph depth.
    Here it is on the scan sweep, which is the axis igl can be compared on:
    ``igl.facet_components`` walks edge-edge adjacency serially, so its cost is the face count.

    ``igl.facet_components`` returns ``(n_components, labels)`` -- the count first, which is easy to
    unpack wrongly -- and numbers components ``0..k-1`` in its own discovery order where ordito
    labels each component by a representative face. The partition is identical; the names are not
    (``tests/test_adjacency.py``).

    VTK's ``connectivity('all')`` is the same partition under a third numbering, and it returns a
    whole new ``PolyData`` carrying the ``RegionId`` cell array rather than the labels alone -- so
    its row prices the copy as well as the traversal.
    """
    if bench_case.kind == "meshlib":
        # ``FaceIncidence.PerEdge`` is ordito's rule and is passed explicitly: the ``PerVertex``
        # setting is a different operation, not a tuning (2 components against 1 on a bowtie). It
        # returns the components as bitsets rather than a label array, so its cost includes
        # materializing k of them.
        mesh_part_ml = mm.MeshPart(bench_case.new_mesh_ml())
        components_ml = bench_case.run(
            lambda: mm.getAllComponents(mesh_part_ml, mm.MeshComponents.FaceIncidence.PerEdge)
        )
        assert 0 < len(components_ml) <= bench_case.n_faces
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        labelled_pv = bench_case.run(lambda: mesh_pv.connectivity("all"))
        assert np.asarray(labelled_pv.cell_data["RegionId"]).shape == (bench_case.n_faces,)
        return
    if bench_case.kind == "ordito":
        faces_wp = bench_case.faces_wp
        labels = bench_case.run(lambda: od.adjacency.face_connected_component_labels(faces_wp))
        assert labels.shape == (bench_case.n_faces,)
        return
    faces_np = bench_case.faces_np
    n_components_igl, labels_igl = bench_case.run(lambda: igl.facet_components(faces_np))
    assert n_components_igl >= 1
    assert labels_igl.ravel().shape == (bench_case.n_faces,)


@pytest.mark.benchmark(group="face_adjacency_angles")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_face_adjacency_angles(bench_case: BenchCase) -> None:
    """Dihedral angle per adjacent pair, from a precomputed adjacency table and fresh normals."""
    if bench_case.kind == "ordito":
        vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
        adjacency, _ = _adjacency(bench_case)
        angles = bench_case.run(
            lambda: od.adjacency.face_adjacency_angles(
                vertices_wp, faces_wp, face_adjacency=adjacency
            )
        )
        assert angles.shape[0] > 0
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    angles_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_angles
    )
    assert len(angles_tm) > 0


@pytest.mark.benchmark(group="vertex_face_adjacency")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("ordito", "igl")
@pytest.mark.parametrize("known_nv", [False, True], ids=["inferred", "known_nv"])
def test_vertex_face_adjacency(bench_case: BenchCase, known_nv: bool) -> None:
    """
    The vertex-to-face incidence CSR: a count, a scan and a scatter over ``3F``.

    On the **valence** axis rather than the scan sweep, because the scatter is ``3F`` atomic
    increments of a per-vertex cursor -- ``fan_hub``'s two valence-40 960 hubs take 40 960 atomics
    each on the same address, which is the one input shape that could serialise it. (The
    ``test_vertices.py`` scatters say it does not; this asks it for the widest-row case.)

    ``inferred`` pays a host readback to learn the row count, ``known_nv`` is handed it -- the same
    escape hatch ``face_adjacency`` has, and the same reason: a caller who knows ``n_vertices``
    should not pay a sync for it. Note the two answers differ in *shape* on a mesh with unreferenced
    trailing vertices, since inference returns ``faces.max() + 1`` rows.

    ``igl.vertex_triangle_adjacency(F, n)`` returns the identical CSR with the pair reversed
    (``(VF, NI)``), and always takes ``n`` explicitly, so it has no ``inferred`` row.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "igl":
        if not known_nv:
            pytest.skip("igl always takes n explicitly")
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        payload_igl, offsets_igl = bench_case.run(
            lambda: igl.vertex_triangle_adjacency(faces_np, n_vertices)
        )
        assert offsets_igl.ravel().shape == (n_vertices + 1,)
        assert payload_igl.ravel().shape == (3 * bench_case.n_faces,)
        return
    faces_wp = bench_case.faces_wp
    supplied = n_vertices if known_nv else None
    vertex_faces, offsets = bench_case.run(
        lambda: od.adjacency.vertex_face_adjacency(faces_wp, n_vertices=supplied)
    )
    assert offsets.size == n_vertices + 1
    assert vertex_faces.size == 3 * bench_case.n_faces


@pytest.mark.benchmark(group="face_adjacency_projections")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_face_adjacency_projections(bench_case: BenchCase) -> None:
    """
    Unshared-vertex plane projections per adjacent face pair, adjacency built inside.

    ``Trimesh.face_adjacency_projections`` is the same quantity in the same layout -- one scalar per
    adjacent face pair, keyed by that pair -- and is what
    ``tests/test_adjacency.py::test_face_adjacency_projections`` compares against, so this row is a
    like-for-like. It is a cached property, so the mesh is rebuilt inside the timed callable exactly
    as ``face_adjacency_convex``'s row does, which means both sides pay for the adjacency the
    projection needs; that is the honest comparison here, since the adjacency is what dominates.
    Capped at ``bunny``: the reference is single-core NumPy and is two orders of magnitude behind
    ordito at the top of the scale axis.
    """
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        projections = bench_case.run(
            lambda: od.adjacency.face_adjacency_projections(vertices, faces)
        )
        assert projections.shape[0] >= 0
    else:  # a cached Trimesh property: rebuild inside, as face_adjacency_convex's row does
        skip_larger_than(bench_case, "bunny", "the reference projects on one core")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        projections_tm = bench_case.run(
            lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_projections
        )
        assert projections_tm.shape[0] >= 0


@pytest.mark.benchmark(group="face_adjacency_convex")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_face_adjacency_convex(bench_case: BenchCase) -> None:
    """Locally-convex adjacent face pairs: the projection plus a tolerance threshold."""
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        convex = bench_case.run(lambda: od.adjacency.face_adjacency_convex(vertices, faces))
        assert convex.shape[0] >= 0
    else:  # rebuild inside: face_adjacency_convex is a cached Trimesh property
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        convex_tm = bench_case.run(
            lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_convex
        )
        assert convex_tm.dtype == bool
