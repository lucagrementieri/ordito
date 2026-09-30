"""
Benchmarks for ``triwarp.edges`` vs trimesh and libigl (igl), on real scan meshes.

Mirrors the correctness tests in ``tests/test_edges.py`` but times each function instead of
asserting equality. Every test is parametrised over ``(mesh_name, library)`` by the harness in
``conftest.py`` and receives a ``bench_case`` bundling the inputs and a GPU-safe ``run``; the
``benchlibs`` marker declares which reference libraries implement an equivalent of the function.

The triwarp ``edges_unique*`` calls pass ``n_vertices=`` (known from the mesh) so the API's
internal ``.numpy().max()`` host sync does not dominate the GPU measurement.

**open3d** has no equivalent for anything in this module. Its ``TriangleMesh`` exposes edges only as
diagnostics over specific predicates (``get_non_manifold_edges``,
``get_self_intersecting_triangles``) and never as a general edge list, so there is no
``faces_to_edges`` / ``edges_unique`` / ``edge_lengths`` to time against.

**potpourri3d** does have one: ``pp3d.edges`` returns geometry-central's internal undirected edge
list. It cannot run on the scan meshes -- geometry-central rejects a mesh with an unreferenced
vertex, and every scan mesh has some -- so it is timed in the separate ``edges_unique_manifold``
group on the synthetic ``scale`` axis instead. Its ordering is geometry-central's own, so the row
includes building the halfedge mesh those indices refer to. That ordering does *not* make it
unusable as an oracle -- sorting dissolves it, and
``tests/test_edges.py::test_edges_unique_matches_potpourri3d`` asserts the two edge sets are equal.
It has no counterpart for the directed, per-corner or length variants.

**libigl does have a length variant**: ``igl.edge_lengths(V, F)`` is the same ``(n_faces, 3)``
per-corner table
``face_edge_lengths`` returns, in the same column order, so that group is a direct comparison rather
than an unreferenced one. (``igl.squared_edge_lengths`` is bound as well and would pair with a
squared entry point; triwarp has none, so there is no row for it.) igl still has no *unique*-edge
length list, which is why ``edges_unique_length`` and ``edges_length`` stay triwarp-only.

**pymeshlab** appears in ``mean_unique_edge_length`` alone. ``get_geometric_measures`` returns
``avg_edge_length`` in a dict alongside the area, volume, barycentre and inertia tensor, so it is an
*upper* bound on the mean edge length taken by itself; the same call is the ``centroid`` reference
in [`test_triangles.py`](test_triangles.py), so those two rows are literally the same measurement
read twice. It has no general edge list either: ``Mesh.edge_matrix()`` exists but is populated only
by filters that build the edge topology, not on demand, so there is nothing to time for
``faces_to_edges`` / ``edges_unique`` / the length variants.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, skip_larger_than


@pytest.mark.benchmark(group="faces_to_edges")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_faces_to_edges(bench_case: BenchCase) -> None:
    """
    The directed ``(3F, 2)`` all-edges table: the cheapest thing in the module on every library.

    ``igl.oriented_facets`` is the same expansion and the fastest single igl call in the whole
    reference suite relative to its output size -- it emits the same 3F directed pairs, in a
    different row order (class B, see ``tests/test_edges.py``).
    """
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.faces_to_edges(faces))
        assert result.shape == (faces.shape[0], 2)
    elif bench_case.kind == "igl":
        faces_np = bench_case.faces_np
        result_igl = bench_case.run(lambda: igl.oriented_facets(faces_np))
        assert result_igl.shape == (bench_case.n_faces * 3, 2)
    else:  # trimesh
        faces = bench_case.faces_np
        result = bench_case.run(lambda: tm.geometry.faces_to_edges(faces))
        assert result.shape == (faces.shape[0] * 3, 2)


@pytest.mark.benchmark(group="faces_to_edges_sorted")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_faces_to_edges_sorted(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.faces_to_edges(faces, sorted=True))
        assert result.shape == (faces.shape[0], 2)
    else:  # trimesh
        faces = bench_case.faces_np
        result = bench_case.run(lambda: np.sort(tm.geometry.faces_to_edges(faces), axis=1))
        assert result.shape == (faces.shape[0] * 3, 2)


@pytest.mark.benchmark(group="edges_face")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_edges_face(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.edges_face(faces))
        assert result.shape == (faces.shape[0],)
    else:  # trimesh: Trimesh.edges_face == repeat(arange(n_faces), 3)
        n_faces = bench_case.faces_np.shape[0]
        result = bench_case.run(lambda: np.repeat(np.arange(n_faces, dtype=np.int64), 3))
        assert result.shape == (n_faces * 3,)


def _run_edges_pytorch3d(bench_case: BenchCase, *, inverse: bool = False) -> None:
    """
    Time ``edges_packed`` (or its face map) with the ``Meshes`` built inside the timed callable.

    Not an optimization to hoist: both are **memoized accessors on an immutable container**, so a
    shared ``Meshes`` would have rounds 2..n read a cached tensor and the row would report ~0. That
    makes it the only reference here whose row carries a container construction; the build is one
    ``torch.as_tensor`` pair over arrays already in the right dtype, an order of magnitude under
    the normals derivation beside it, so it is a small share of a grouping
    pass rather than the row.
    """
    import pytorch3d.structures as p3d_structures
    import torch

    vertices_p3d = torch.as_tensor(
        np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
        device=bench_case.torch_device,
    )
    faces_p3d = torch.as_tensor(
        np.ascontiguousarray(bench_case.faces_np, dtype=np.int64), device=bench_case.torch_device
    )

    def run_p3d():
        mesh_p3d = p3d_structures.Meshes(verts=[vertices_p3d], faces=[faces_p3d])
        return mesh_p3d.faces_packed_to_edges_packed() if inverse else mesh_p3d.edges_packed()

    assert bench_case.run(run_p3d).shape[1] == (3 if inverse else 2)


def _run_edges_unique_reference(bench_case: BenchCase) -> None:
    """
    Time whichever unique-undirected-edge reference this case names.

    Shared by the two ``edges_unique`` groups because the ``n_vertices=`` hint is triwarp's own
    parameter: no reference takes a vertex-count hint, so each one runs the identical call in both
    groups and its two rows are a fixed bar against triwarp's with/without pair.
    """
    if bench_case.kind == "pytorch3d":
        _run_edges_pytorch3d(bench_case)
        return
    if bench_case.kind == "pyvista":
        # ``extract_all_edges`` returns the same unique undirected set, wrapped in a line-cell
        # PolyData -- so its row carries the container build as well as the grouping.
        mesh_pv = bench_case.mesh_pv
        edges_pv = bench_case.run(mesh_pv.extract_all_edges)
        assert edges_pv.n_cells > 0
    elif bench_case.kind == "trimesh":
        faces = bench_case.faces_np

        def run():
            edges_sorted = np.sort(tm.geometry.faces_to_edges(faces), axis=1)
            unique_idx, _ = tm.grouping.unique_rows(edges_sorted)
            return edges_sorted[unique_idx]

        assert run().shape[1] == 2
        bench_case.run(run)
    else:  # igl.unique_edge_map -> (E, uE, EMAP, uEC, uEE); uE is the unique undirected edges
        faces = bench_case.faces_np
        result = bench_case.run(lambda: igl.unique_edge_map(faces)[1])
        assert result.shape[1] == 2


@pytest.mark.benchmark(group="edges_unique")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pyvista", "pytorch3d")
def test_edges_unique(bench_case: BenchCase) -> None:

    if bench_case.kind != "triwarp":
        _run_edges_unique_reference(bench_case)
        return
    faces, nv = bench_case.faces_wp, bench_case.n_vertices
    unique_edges, _ = bench_case.run(lambda: tw.edges.edges_unique(faces, n_vertices=nv))
    assert unique_edges.shape[1] == 2


@pytest.mark.benchmark(group="edges_unique_auto_nv")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pyvista", "pytorch3d")
def test_edges_unique_auto_n_vertices(bench_case: BenchCase) -> None:
    """
    Time ``edges_unique`` without the ``n_vertices=`` shortcut.

    Exercises the internal vertex-count inference (a full-array host max before the
    ``n_vertices`` device-reduce fix).

    The same three references ``edges_unique`` times, because the hint is triwarp's parameter and
    none of them has one: they compute the identical unique edge list either way, so their rows are
    the fixed bar and the difference between the two groups is entirely triwarp's inference. Read
    the pair, not this row alone -- on its own it prices a reference that never changes.

    Time the unique undirected edge list with the ``n_vertices=`` radix hint supplied.

    **pytorch3d** joins the reference set as ``Meshes.edges_packed``, which is the same
    deduplication keyed on the hash ``V * v0 + v1`` -- so its rows come back already lexsorted,
    which is why ``tests/test_edges.py::test_edges_unique_and_inverse_match_pytorch3d`` only has to
    move triwarp's side. It is a memoized accessor on an immutable container, so its ``Meshes``
    is built **inside** the timed callable; reading it on a shared object would price nothing.

    **pytorch3d** has no hint either, so its row is the same fixed bar the other three are.
    """
    if bench_case.kind != "triwarp":
        _run_edges_unique_reference(bench_case)
        return
    faces = bench_case.faces_wp
    unique_edges, _ = bench_case.run(lambda: tw.edges.edges_unique(faces))
    assert unique_edges.shape[1] == 2


@pytest.mark.benchmark(group="edges_unique_manifold")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl", "potpourri3d")
def test_edges_unique_manifold(bench_case: BenchCase) -> None:
    """
    The same unique-edge list on the clean synthetic meshes, where potpourri3d can run.

    ``pp3d.edges`` raises ``GC_SAFETY_ASSERT FAILURE ... unreferenced vertex`` on every scan mesh --
    geometry-central refuses to build a mesh with a vertex no face references, and the scans all
    have some. Rather than skip the row, this group draws the comparison on the ``scale`` axis, the
    same move that unblocked several libigl comparisons (see the README's measured-hazards section).
    igl is kept alongside so the group is not a two-row table.
    """
    if bench_case.kind == "triwarp":
        faces, nv = bench_case.faces_wp, bench_case.n_vertices
        unique_edges, _ = bench_case.run(lambda: tw.edges.edges_unique(faces, n_vertices=nv))
        assert unique_edges.shape[1] == 2
    elif bench_case.kind == "potpourri3d":
        # geometry-central's internal edge list, in its own ordering: a timing comparison, not a
        # parity one. Building the halfedge mesh those indices refer to is the cost being shown.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: pp3d.edges(vertices_np, faces_np))
        assert result.shape[1] == 2
    else:  # igl.unique_edge_map -> (E, uE, EMAP, uEC, uEE); uE is the unique undirected edges
        faces = bench_case.faces_np
        result = bench_case.run(lambda: igl.unique_edge_map(faces)[1])
        assert result.shape[1] == 2


@pytest.mark.benchmark(group="edges_unique_inverse")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pytorch3d")
def test_edges_unique_inverse(bench_case: BenchCase) -> None:
    if bench_case.kind == "pytorch3d":
        _run_edges_pytorch3d(bench_case, inverse=True)
        return
    if bench_case.kind == "triwarp":
        faces, nv = bench_case.faces_wp, bench_case.n_vertices
        result = bench_case.run(lambda: tw.edges.edges_unique_inverse(faces, n_vertices=nv))
        assert result.shape == (faces.shape[0],)
    elif bench_case.kind == "trimesh":
        faces = bench_case.faces_np

        def run():
            edges_sorted = np.sort(tm.geometry.faces_to_edges(faces), axis=1)
            _, inverse = tm.grouping.unique_rows(edges_sorted)
            return inverse

        bench_case.run(run)
    else:  # igl.unique_edge_map -> EMAP (index 2) maps each directed edge to its unique edge
        faces = bench_case.faces_np
        bench_case.run(lambda: igl.unique_edge_map(faces)[2])


@pytest.mark.benchmark(group="edges_unique_length")
@pytest.mark.benchlibs("triwarp", "trimesh", "meshlib")
def test_edges_unique_length(bench_case: BenchCase) -> None:
    """
    One length per undirected edge, which on both sides is really a deduplication.

    meshlib's ``edgeLengths`` reads its half-edge structure rather than sorting, so its row prices a
    different route to the same answer -- and the structure is lazily built and cached on the
    topology like the AABB tree, so the mesh is built inside the timed callable to keep that build
    inside the measurement, matching the dedup the other two rows pay for.

    Time the directed-edge to unique-edge map, which on every side is the grouping's inverse.

    **pytorch3d**'s ``faces_packed_to_edges_packed`` is the same map in its own edge numbering and
    in the column order ``(e12, e20, e01)``; both are named transforms in
    ``tests/test_edges.py::test_edges_unique_and_inverse_match_pytorch3d`` and neither costs
    anything. Its ``Meshes`` is built inside the timed callable, for the memoization reason the
    ``edges_unique`` group gives.
    """
    if bench_case.kind == "meshlib":

        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            return mm.edgeLengths(mesh_ml.topology, mesh_ml.points).size()

        assert bench_case.run(run_ml) > 0
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        nv = bench_case.n_vertices
        result = bench_case.run(
            lambda: tw.edges.edges_unique_length(vertices, faces, n_vertices=nv)
        )
        assert result.ndim == 1
    else:  # trimesh: unique undirected edges then Euclidean norm
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            edges_sorted = np.sort(tm.geometry.faces_to_edges(faces), axis=1)
            unique_idx, _ = tm.grouping.unique_rows(edges_sorted)
            unique = edges_sorted[unique_idx]
            return np.linalg.norm(vertices[unique[:, 1]] - vertices[unique[:, 0]], axis=1)

        bench_case.run(run)


@pytest.mark.benchmark(group="edges_length")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_edges_length(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.edges_length(vertices, faces))
        assert result.shape == (faces.shape[0],)
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            edges = np.asarray(tm.geometry.faces_to_edges(faces))
            return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)

        bench_case.run(run)
    else:  # igl.edge_lengths -> (#F, 3) per-face directed edge lengths
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: igl.edge_lengths(vertices, faces))
        assert result.shape == (faces.shape[0], 3)


@pytest.mark.benchmark(group="mean_edge_length")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_mean_edge_length(bench_case: BenchCase) -> None:
    """
    The **per-face** edge average -- ``3 * n_faces`` lengths, every interior edge counted twice.

    Paired with ``igl.edge_lengths(...).mean()``, which is the same quantity: it is what
    ``CurvatureCalculator::getAverageEdge`` computes and therefore what ``igl::principal_curvature``
    scales its sphere radius by. **Not** ``igl.avg_edge_length`` -- that averages the unique edge
    list and is a different number on any mesh with a boundary, which every scan mesh here has. It
    is timed in the ``mean_unique_edge_length`` group below against the triwarp function that
    matches it.

    So this group is the cheap one: no deduplication, one pass and a reduction.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        assert bench_case.run(lambda: tw.edges.mean_edge_length(vertices, faces)) >= 0.0
    elif bench_case.kind == "trimesh":  # mean of all per-face edge norms
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            tri = vertices[faces]
            return float(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2).mean())

        bench_case.run(run)
    else:  # igl.edge_lengths -> (#F, 3), the per-face table; its mean is getAverageEdge
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        bench_case.run(lambda: float(np.asarray(igl.edge_lengths(vertices, faces)).mean()))


@pytest.mark.benchmark(group="mean_unique_edge_length")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pymeshlab", "meshlib")
def test_mean_unique_edge_length(bench_case: BenchCase) -> None:
    """
    The **unique** edge average, where the deduplication is most of the cost.

    Every row here computes the identical number: ``igl::avg_edge_length`` builds the unique edge
    list and averages it, MeshLab reports the same value as ``avg_edge_length``, and the trimesh row
    is that formula in numpy. Contrast the ``mean_edge_length`` group above, which is the per-face
    average -- the two differ by 0.55% on an open half-torus and this sweep's meshes all have
    boundaries.

    The interesting comparison is the dedup: triwarp reaches the unique edges through a sort where
    the numpy row goes through ``np.unique(axis=0)``, which is why this group is several times the
    cost of its per-face twin on both sides. meshlib is a fifth row on the same number and a sixth
    route to it -- its half-edge structure, built inside the timed callable for the reason the
    ``edges_unique_length`` group above gives.
    """
    if bench_case.kind == "meshlib":

        def run_ml() -> float:
            mesh_ml = bench_case.new_mesh_ml()
            return mm.averageEdgeLength(mesh_ml.topology, mesh_ml.points)

        assert bench_case.run(run_ml) > 0.0
        return
    if bench_case.kind == "pymeshlab":
        # ``get_geometric_measures`` is read-only and returns ``avg_edge_length`` alongside the
        # area, volume, barycentre and inertia tensor -- one call for all of them, so this row is an
        # upper bound on the mean edge length taken alone. Capped at ``bunny`` for the same reason
        # as its twin row in ``test_triangles``: 1.12 s a call on ``dragon``.
        skip_larger_than(bench_case, "bunny", "get_geometric_measures is 1.12 s a call on dragon")
        meshset_pml = bench_case.meshset_pml
        assert bench_case.run(meshset_pml.get_geometric_measures)["avg_edge_length"] >= 0.0
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        assert bench_case.run(lambda: tw.edges.mean_unique_edge_length(vertices, faces)) >= 0.0
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            edges = np.unique(np.sort(tm.geometry.faces_to_edges(faces), axis=1), axis=0)
            return float(
                np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1).mean()
            )

        bench_case.run(run)
    else:
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        bench_case.run(lambda: float(igl.avg_edge_length(vertices, faces)))


@pytest.mark.benchmark(group="face_edge_lengths")
@pytest.mark.benchlibs("triwarp", "igl")
def test_face_edge_lengths(bench_case: BenchCase) -> None:
    """
    The table alone: one pass, three lengths per face, no reduction.

    ``igl.edge_lengths(V, F)`` returns the identical ``(n_faces, 3)`` per-corner table in the same
    column order, agreeing to triwarp's float32 vertex buffer where every column permutation differs
    by orders of magnitude more, so the match is the convention and not a coincidence. This is the
    one row in this module where igl has a *length* entry point.
    """
    if bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        lengths_igl = bench_case.run(lambda: igl.edge_lengths(vertices_np, faces_np))
        assert lengths_igl.shape == (bench_case.n_faces, 3)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    lengths = bench_case.run(lambda: tw.edges.face_edge_lengths(vertices, faces))
    assert lengths.shape == (bench_case.n_faces, 3)
