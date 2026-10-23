"""
Benchmarks for ``ordito.graph``: connected components and the weighted envelope.

Two axes, matching the two ways a graph algorithm gets slow:

* **components** for ``connected_component_labels``. ECL-CC is three launches whatever the graph
  looks like, but the pointer-jumping hook/flatten converges at a rate set by component structure,
  so 1 / 64 / 1024 components at a fixed 81 920 faces is the shape that would expose a regression.
  The same two labellings are *also* run on the **diameter** axis, where they must read flat --
  that is a separate claim from the component sweep and it is load-bearing elsewhere in the
  package, so it is regression-covered rather than measured once (see
  ``test_connected_component_labels_depth``).
* **diameter**, for ``shortest_path_envelope`` -- one launch per relaxation pass, and the pass
  count is the diameter of the region that violates the bound. The work per pass sweeps the whole
  CSR rather than a frontier, so there is no narrow-frontier handover to make. Seeded from a single
  spike, the worst case on purpose: the cap has to cross the whole mesh, so read the row as an upper
  bound rather than a typical one. Capped at ``bunny_decimated`` for that reason.

The vertex-adjacency CSR matrix is prebuilt (untimed, cached per mesh/device) so the timings
isolate the graph algorithms from the edge sort that produces them -- including
``shortest_path_envelope``'s length-weighted one, whose weights are ``edges_unique_length``.

``combine.split`` is the other component-count-driven function in the package and sits in
[`test_combine.py`](test_combine.py) with the rest of ``ordito.combine``, on the same
``components`` axis.

References
----------
The scipy reference sits in the ``scipy`` library slot: ``scipy.sparse.csgraph
.connected_components`` is the labelling oracle for both component groups (and the backend
trimesh itself uses for graph traversals).

Neither **trimesh** nor **open3d** appears: both functions take an abstract CSR adjacency matrix,
and open3d exposes no graph-traversal API over one -- its connectivity work is mesh-bound
(``cluster_connected_triangles``), which is what ``split`` uses over in ``test_combine``.

**libigl does take that argument**, despite the paragraph above.
``igl.connected_components`` accepts a ``scipy.sparse`` adjacency matrix directly -- the same
argument ``connected_component_labels`` takes -- and ``igl.facet_components(F)`` is the dual-graph
labelling ``face_connected_component_labels`` computes. So both labellings have a second
implementation and the face one, which had none at all, now has two.

**pymeshlab** answers two groups. ``shortest_path_envelope``'s reference is
``apply_scalar_saturation_per_vertex``, which is the same relaxation read as a Lipschitz cap on a
per-vertex scalar (VCG ``UpdateQuality::VertexSaturate``); its ``gradientthr`` divides the edge
length, so both sides receive the identical weights and the row is apples-to-apples. It needs the
scalar attribute to exist on the MeshSet and mutates it, so that row rebuilds the set inside the
timed callable.

Its other appearance, ``connected_component_labels``, comes with a caveat: it has no filter
that returns a label array. The closest thing that runs the component pass *without* also splitting
or deleting anything is ``compute_selection_by_small_disconnected_components_per_face`` at
``nbfaceratio=0.0`` -- it labels every component and then thresholds against a fraction of the
largest one, selecting nothing. So the row is "label everything, then one threshold pass", against
ordito's "label everything". The labelling is the only graph work MeshLab exposes.
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import scipy.sparse as sp
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, skip_larger_than

_adjacency_cache: dict[tuple[str, str], odt.BsrMatrix[wp.float32]] = {}
_scipy_cache: dict[str, sp.csr_matrix] = {}


def _adjacency(bench_case: BenchCase) -> odt.BsrMatrix[wp.float32]:
    """Vertex adjacency as a warp BSR matrix -- an *input*, built once per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _adjacency_cache:
        unique_edges, _ = od.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
        _adjacency_cache[key] = od.graph.edges_to_csr(bench_case.n_vertices, unique_edges)
    return _adjacency_cache[key]


def _scipy_graph(bench_case: BenchCase) -> sp.csr_matrix:
    """Build the same adjacency as a scipy CSR matrix, for the component-labelling oracle."""
    if bench_case.mesh_name not in _scipy_cache:
        faces = bench_case.faces_np
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        n = bench_case.n_vertices
        data = np.ones(edges.shape[0], dtype=np.float32)
        graph = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        _scipy_cache[bench_case.mesh_name] = graph
    return _scipy_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="edges_to_csr")
@pytest.mark.benchlibs("ordito")
def test_edges_to_csr(bench_case: BenchCase) -> None:
    """
    The symmetric vertex adjacency from the unique edges: one key launch and the CSR key sort.

    Timed alone: it is the input every other group here takes prebuilt.
    """
    unique_edges, _ = od.edges.edges_unique(bench_case.faces_wp, n_vertices=bench_case.n_vertices)
    n_vertices = bench_case.n_vertices
    adjacency = bench_case.run(lambda: od.graph.edges_to_csr(n_vertices, unique_edges))
    assert adjacency.nrow == n_vertices


@pytest.mark.benchmark(group="edges_to_neighbor_lists")
@pytest.mark.benchlibs("ordito", "igl")
def test_edges_to_neighbor_lists(bench_case: BenchCase) -> None:
    """
    Packed per-vertex neighbour lists from the unique edges: a degree count, a scan and a scatter.

    The unsorted, arrival-order form that ``edges_to_csr`` above trades for its key sort, so the two
    rows read side by side as what the sort costs. ``igl.adjacency_list`` is the same relation as a
    Python list of lists (``tests/test_graph.py::test_edges_to_neighbor_lists_matches_igl``); it
    reads the faces directly, so its row includes the edge discovery ordito's input already holds.
    Capped at ``bunny``, past which building one Python list per vertex dominates.
    """
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference builds one Python list per vertex")
        faces_igl = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        adjacency_igl = bench_case.run(lambda: igl.adjacency_list(faces_igl))
        assert len(adjacency_igl) == bench_case.n_vertices
        return
    n_vertices = bench_case.n_vertices
    unique_edges, _ = od.edges.edges_unique(bench_case.faces_wp, n_vertices=n_vertices)
    neighbors, offsets = bench_case.run(
        lambda: od.graph.edges_to_neighbor_lists(n_vertices, unique_edges)
    )
    assert offsets.shape == (n_vertices + 1,)
    assert neighbors.shape == (2 * unique_edges.shape[0],)


@pytest.mark.benchmark(group="edges_to_neighbor_lists_from_counts")
@pytest.mark.benchlibs("ordito")
def test_edges_to_neighbor_lists_from_counts(bench_case: BenchCase) -> None:
    """
    The same neighbour lists from degrees the caller already holds: the scan and the scatter only.

    Read against ``edges_to_neighbor_lists`` for the price of the range check and degree count it
    skips. ``counts`` is overwritten by the call, so each round gets a fresh copy outside the timed
    region. Ordito only: the test pins it against ``edges_to_neighbor_lists``.
    """
    n_vertices = bench_case.n_vertices
    unique_edges, _ = od.edges.edges_unique(bench_case.faces_wp, n_vertices=n_vertices)
    counts_np = np.bincount(unique_edges.numpy().reshape(-1), minlength=n_vertices)
    counts = wp.array(counts_np.astype(np.int32), dtype=wp.int32, device=bench_case.device)
    neighbors, offsets = bench_case.run(
        lambda fresh: od.graph.edges_to_neighbor_lists_from_counts(unique_edges, fresh),
        setup=lambda: wp.clone(counts),
    )
    assert offsets.shape == (n_vertices + 1,)
    assert neighbors.shape == (2 * unique_edges.shape[0],)


@pytest.mark.benchmark(group="connected_component_labels")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("ordito", "scipy", "igl", "pymeshlab", "meshlib")
def test_connected_component_labels(bench_case: BenchCase) -> None:
    """
    ECL-CC hook and flatten, over 1 / 64 / 1024 components at a fixed face count.

    meshlib's ``getAllComponentsVerts`` is the only reference here that returns the components
    themselves -- one ``VertBitSet`` each, in its own traversal order -- rather than a label array
    or a derived selection, so its cost includes materializing ``k`` bitsets and should be the row
    that moves most across this axis. It reads the topology rather than an assembled adjacency, so
    the mesh is built outside the timed callable exactly as ordito's CSR is.

    ``igl.connected_components`` is the other label-array reference and the closest match to
    ordito's signature -- a ``scipy.sparse`` adjacency in, labels out. It builds that adjacency
    inside the timed callable (``igl.adjacency_matrix`` is the only form it takes), so on this axis
    its row carries the build where scipy's does not; the two are read against ordito separately
    rather than against each other. Note it counts every *isolated vertex* as its own component, so
    its component count differs from the others on a mesh with unreferenced vertices while the
    partition it induces on the referenced ones does not.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "igl":
        faces_np = bench_case.faces_np
        n_labelled_igl, labels_igl, _sizes_igl = bench_case.run(
            lambda: igl.connected_components(igl.adjacency_matrix(faces_np))
        )
        assert int(n_labelled_igl) >= 1
        assert np.asarray(labels_igl).shape[0] <= n_vertices
        return
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        # ``None`` is the whole mesh; the stub omits that the region is nullable.
        components_ml = bench_case.run(lambda: mm.getAllComponentsVerts(mesh_ml, None))  # pyright: ignore[reportArgumentType]
        assert 0 < len(components_ml) <= n_vertices
        return
    if bench_case.kind == "pymeshlab":
        # MeshLab has no filter that hands back a label array; the closest thing that *runs* the
        # component pass without also splitting the mesh is the small-component face selection,
        # which labels every component and then thresholds against a fraction of the largest one.
        # ``nbfaceratio=0.0`` selects nothing, so the timing is the labelling pass and the
        # threshold, not a deletion.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_selection_by_small_disconnected_components_per_face(
                nbfaceratio=0.0
            )
        )
        assert meshset_pml.current_mesh().face_selection_array().shape == (bench_case.n_faces,)
        return
    if bench_case.kind == "ordito":
        adjacency = _adjacency(bench_case)
        labels = bench_case.run(lambda: od.graph.connected_component_labels(adjacency))
        assert labels.shape == (n_vertices,)
    else:
        graph = _scipy_graph(bench_case)
        n_labelled, labels_np = bench_case.run(lambda: sp.csgraph.connected_components(graph))
        assert n_labelled >= 1
        assert labels_np.shape == (n_vertices,)


_edges_cache: dict[tuple[str, str], odt.Array2dInt32] = {}


def _unique_edges(bench_case: BenchCase) -> odt.Array2dInt32:
    """Unique-edge table -- an *input*, built once per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edges_cache:
        _edges_cache[key] = od.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )[0]
    return _edges_cache[key]


@pytest.mark.benchmark(group="connected_component_labels_from_edges")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("ordito", "scipy", "igl")
def test_connected_component_labels_from_edges(bench_case: BenchCase) -> None:
    """
    ECL-CC over an edge list rather than an assembled CSR, on the same components axis.

    The default ``validate=True`` with the node count passed, so the row carries the range check and
    its readback against ``connected_component_labels``'s prebuilt adjacency. Both references take a
    prebuilt ``scipy.sparse`` adjacency here (``tests/test_graph.py::
    test_connected_component_labels_matches_igl`` compares the partition with both), so this row,
    unlike the CSR group's, does not charge igl for ``adjacency_matrix``.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "igl":
        adjacency_igl = igl.adjacency_matrix(np.ascontiguousarray(bench_case.faces_np, np.int64))
        n_labelled_igl, _labels_igl, _sizes_igl = bench_case.run(
            lambda: igl.connected_components(adjacency_igl)
        )
        assert int(n_labelled_igl) >= 1
        return
    if bench_case.kind == "scipy":
        graph = _scipy_graph(bench_case)
        n_labelled, labels_np = bench_case.run(lambda: sp.csgraph.connected_components(graph))
        assert n_labelled >= 1
        assert labels_np.shape == (n_vertices,)
        return
    unique_edges = _unique_edges(bench_case)
    labels = bench_case.run(
        lambda: od.graph.connected_component_labels_from_edges(unique_edges, n_vertices)
    )
    assert labels.shape == (n_vertices,)


@pytest.mark.benchmark(group="connected_component_parity_from_edges")
@pytest.mark.benchlibs("ordito")
def test_connected_component_parity_from_edges(bench_case: BenchCase) -> None:
    """
    The checked parity union-find over the mesh's unique edges, every sign ``1``.

    Ordito only: no reference library solves a Z2 potential over an edge list. The default
    ``validate=True`` is timed, so the row includes the endpoint and sign range checks and their
    readback; the edge table and signs are inputs, built outside the timed callable.
    """
    n_vertices = bench_case.n_vertices
    unique_edges, _ = od.edges.edges_unique(bench_case.faces_wp, n_vertices=n_vertices)
    signs = wp.ones(unique_edges.shape[0], dtype=wp.int32, device=unique_edges.device)
    labels, parity = bench_case.run(
        lambda: od.graph.connected_component_parity_from_edges(unique_edges, signs, n_vertices)
    )
    assert labels.shape == (n_vertices,)
    assert parity.shape == (n_vertices,)


def _successor_edges_np(bench_case: BenchCase, shape: str) -> tuple[np.ndarray, int]:
    """
    Build a successor graph sized by the mesh, its rows shuffled with a fixed seed.

    ``"faces"`` is the halfedge ``next`` graph -- ``n_faces`` cycles of three over ``3 * n_faces``
    halfedge ids, the many-small-cycles shape of a mesh with many holes. ``"one_cycle"`` is one
    cycle through every vertex in a random order, which takes the most pointer-jumping rounds.
    """
    rng = np.random.default_rng(0)
    if shape == "faces":
        n = 3 * bench_case.n_faces
        tails = np.arange(n, dtype=np.int32)
        heads = (tails // 3) * 3 + (tails % 3 + 1) % 3
    else:
        n = bench_case.n_vertices
        tails = rng.permutation(n).astype(np.int32)
        heads = np.roll(tails, -1)
    edges_np = np.stack((tails, heads), axis=1).astype(np.int32)
    return edges_np[rng.permutation(n)], n


@pytest.mark.benchmark(group="successor_cycles")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("shape", ["faces", "one_cycle"])
def test_successor_cycles(bench_case: BenchCase, shape: str) -> None:
    """
    Every cycle of a successor graph, ranked and packed, with the default range check.

    Ordito only: no reference library ranks the cycles of a successor graph. The edge table is an
    input, uploaded outside the timed callable. On the ``scale`` axis, so the census gate sees it.
    """
    edges_np, node_count = _successor_edges_np(bench_case, shape)
    edges = odt.as_array2d(wp.array(edges_np, dtype=wp.int32, device=bench_case.device), wp.int32)
    flat_cycles, offsets = bench_case.run(lambda: od.graph.successor_cycles(edges, node_count))
    assert flat_cycles.shape == (node_count,)
    assert offsets.shape[0] == (node_count // 3 if shape == "faces" else 1) + 1


@pytest.mark.benchmark(group="successor_cycles_from_table")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("shape", ["faces", "one_cycle"])
def test_successor_cycles_from_table(bench_case: BenchCase, shape: str) -> None:
    """
    The cycles of ``successor_cycles``'s graphs, from a successor table the caller already holds.

    No range check and no table scatter, so read against ``successor_cycles`` for what those cost.
    Ordito only: no reference library ranks the cycles of a successor graph.
    """
    edges_np, node_count = _successor_edges_np(bench_case, shape)
    next_np = np.full(node_count, -1, dtype=np.int32)
    next_np[edges_np[:, 0]] = edges_np[:, 1]
    tails = wp.array(np.ascontiguousarray(edges_np[:, 0]), dtype=wp.int32, device=bench_case.device)
    next_node = wp.array(next_np, dtype=wp.int32, device=bench_case.device)
    flat_cycles, offsets = bench_case.run(
        lambda: od.graph.successor_cycles_from_table(tails, next_node)
    )
    assert flat_cycles.shape == (node_count,)
    assert offsets.shape[0] == (node_count // 3 if shape == "faces" else 1) + 1


@pytest.mark.benchmark(group="connected_component_labels_depth")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("ordito", "scipy", "igl")
def test_connected_component_labels_depth(bench_case: BenchCase) -> None:
    """
    The depth-robustness gate: ECL-CC on a graph of diameter 130 against one of diameter 20 481.

    Pointer jumping has no level loop, so this pair reads **flat** across the two diameters. That
    is not a self-evident property -- every level-synchronous traversal in the package
    fails it, ``shortest_path_envelope`` below included -- and it is the premise the parity
    union-find in ``validation.face_orientation_bits`` rests on. A slope appearing here is the
    regression that would invalidate it.

    ``igl.connected_components`` takes a ``scipy.sparse`` adjacency matrix -- the same argument
    ordito's function takes -- so it is a genuine third labelling here rather than a mesh-bound
    stand-in. Its adjacency comes from ``igl.adjacency_matrix(F)`` and is built inside the timed
    callable, because that is the only form it accepts; the scipy row reuses this module's cached
    CSR, so read the two reference rows as build-included and build-excluded rather than against
    each other.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        adjacency = _adjacency(bench_case)
        labels = bench_case.run(lambda: od.graph.connected_component_labels(adjacency))
        assert labels.shape == (n_vertices,)
    elif bench_case.kind == "igl":
        faces_np = bench_case.faces_np
        n_labelled_igl, labels_igl, _sizes_igl = bench_case.run(
            lambda: igl.connected_components(igl.adjacency_matrix(faces_np))
        )
        assert int(n_labelled_igl) == 1
        assert np.asarray(labels_igl).shape[0] == n_vertices
    else:
        graph = _scipy_graph(bench_case)
        n_labelled, labels_np = bench_case.run(lambda: sp.csgraph.connected_components(graph))
        assert n_labelled == 1
        assert labels_np.shape == (n_vertices,)


@pytest.mark.benchmark(group="face_connected_component_labels_depth")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("ordito", "scipy", "igl")
def test_face_connected_component_labels_depth(bench_case: BenchCase) -> None:
    """
    Same gate one level up, over the face-adjacency graph.

    This is the build-plus-ECL-CC pair ``validation.face_orientation_bits`` actually calls.

    Flat across the two diameters as well -- the edge sort dominates and neither half of it is
    depth-sensitive.

    Both references are **build-included**, which is what makes them fair here: ordito's row times
    the dual-graph construction *and* the labelling, so a reference handed a prebuilt matrix would
    be pricing half the work. "No scipy counterpart" is an argument about the timed region rather
    than about the comparison -- the answer is to put the reference's build inside its own callable,
    not to leave the group unreferenced.

    - **igl** ``facet_components(F)`` is the direct counterpart: faces in, per-face labels out, dual
      graph built internally.
    - **scipy** builds the dual explicitly -- each edge shared by two faces contributes one entry --
      and then runs ``connected_components``, so its row is the same two phases ordito's is.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "igl":
        faces_np = bench_case.faces_np
        n_labelled_igl, labels_igl = bench_case.run(lambda: igl.facet_components(faces_np))
        assert int(n_labelled_igl) >= 1
        assert np.asarray(labels_igl).shape[0] == n_faces
        return
    if bench_case.kind == "scipy":
        faces_np = bench_case.faces_np

        def face_components_np() -> tuple[int, np.ndarray]:
            """Build the dual graph from shared edges, then label it -- both phases timed."""
            edges = np.sort(
                np.concatenate((faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]])),
                axis=1,
            )
            owner = np.tile(np.arange(faces_np.shape[0]), 3)
            order = np.lexsort((edges[:, 1], edges[:, 0]))
            edges, owner = edges[order], owner[order]
            shared = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
            dual = sp.coo_matrix(
                (np.ones(shared.size, dtype=np.int8), (owner[shared], owner[shared + 1])),
                shape=(faces_np.shape[0], faces_np.shape[0]),
            ).tocsr()
            return sp.csgraph.connected_components(dual)

        n_labelled_np, labels_np = bench_case.run(face_components_np)
        assert n_labelled_np >= 1
        assert labels_np.shape == (n_faces,)
        return
    labels = bench_case.run(
        lambda: od.adjacency.face_connected_component_labels(bench_case.faces_wp)
    )
    assert labels.shape == (n_faces,)


_weighted_cache: dict[tuple[str, str], odt.BsrMatrix[wp.float32]] = {}


def _length_weighted_adjacency(bench_case: BenchCase) -> odt.BsrMatrix[wp.float32]:
    """Vertex adjacency weighted by Euclidean edge length -- an *input*, built once per case."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _weighted_cache:
        unique_edges, _ = od.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
        lengths = od.edges.edges_unique_length(
            bench_case.vertices_wp, bench_case.faces_wp, unique_edges
        )
        _weighted_cache[key] = od.graph.edges_to_csr(bench_case.n_vertices, unique_edges, lengths)
    return _weighted_cache[key]


def _spike_field_np(bench_case: BenchCase) -> np.ndarray:
    """Build a delta at vertex 0, so every pass has work and the relaxation is diameter-deep."""
    values_np = np.zeros(bench_case.n_vertices, dtype=np.float64)
    values_np[0] = 10.0
    return values_np


@pytest.mark.benchmark(group="shortest_path_envelope")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_shortest_path_envelope(bench_case: BenchCase) -> None:
    """
    Weighted relaxation to the shortest-path envelope: pass count is the graph diameter.

    Capped at ``bunny_decimated`` on both sides. The spike seed makes every pass matter, so the
    ordito row is ``diameter`` launches deep and the MeshLab row is a serial flood over the same
    region -- neither says anything new at larger scale that the two smallest meshes do not. The
    weighted adjacency is an input and is built outside the timed callable, like the unweighted
    one the labelling groups take.
    """
    skip_larger_than(bench_case, "bunny_decimated", "a spike-seeded relaxation is diameter-deep")
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":

        def saturate_pml() -> int:
            meshset_pml = ml.MeshSet()
            meshset_pml.add_mesh(
                ml.Mesh(
                    bench_case.vertices_np,
                    np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
                    v_scalar_array=_spike_field_np(bench_case),
                )
            )
            meshset_pml.apply_scalar_saturation_per_vertex(gradientthr=1.0)
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(saturate_pml) == n_vertices
        return
    adjacency = _length_weighted_adjacency(bench_case)
    values = wp.array(
        _spike_field_np(bench_case).astype(np.float32), dtype=wp.float32, device=bench_case.device
    )
    envelope = bench_case.run(lambda: od.graph.shortest_path_envelope(adjacency, values), rounds=3)
    assert envelope.shape == (n_vertices,)
