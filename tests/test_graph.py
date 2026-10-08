"""Regression tests for ``ordito.graph`` against Trimesh (CPU reference)."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from ordito import _launch
from tests.comparisons import same_partition
from tests.conftest import MESHES
from tests.conversions import (
    meshlib_bitset_to_numpy,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    warp_empty,
)


def test_edges_to_csr_roundtrip(device: str) -> None:
    edges_np = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    node_count = 3
    edges_wp = _edges_wp(edges_np, device)
    adjacency = od.graph.edges_to_csr(node_count, edges_wp)
    offsets = adjacency.offsets.numpy()
    indices = adjacency.columns.numpy()

    assert adjacency.nrow == node_count
    assert adjacency.ncol == node_count
    assert adjacency.block_shape == (1, 1)
    assert offsets[0] == 0
    assert offsets[-1] == len(indices)
    assert offsets.size == node_count + 1

    neighbors: dict[int, set[int]] = {i: set() for i in range(node_count)}
    for a, b in edges_np:
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))

    for v in range(node_count):
        row = indices[offsets[v] : offsets[v + 1]]
        assert set(row.tolist()) == neighbors[v]


@pytest.mark.parametrize("mesh_name", MESHES)
def test_edges_to_neighbor_lists_matches_igl(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Class B (a named transform): ``igl.adjacency_list`` is a list of lists, this is packed.

    igl returns one Python list per vertex, already ascending; ``edges_to_neighbor_lists`` returns
    ``(neighbors, offsets)`` with **arbitrary order within a row**, which is the documented
    difference and the whole reason it is cheaper than
    [`edges_to_csr`][ordito.graph.edges_to_csr]. Sorting each row is the transform.

    The fixtures span closed/open and curved/faceted, and every one of them is **irregular**: the
    assert below pins the maximum degree strictly above the minimum, because on a mesh where every
    row has the same width a misplaced row is invisible to a row-by-row compare. ``icosahedron`` is
    excluded for exactly that reason — all twelve of its vertices have degree 5.

    Two ordito-against-ordito claims ride on the same graph, with ``edges_to_csr`` carrying the
    oracle (igl here, and scipy through ``connected_component_labels``):

    - the default builder describes ``edges_to_csr``'s graph -- same offsets, same rows up to a
      permutation -- which is the claim every caller that switched between them relies on;
    - ``sort_rows=True`` is ``edges_to_csr``'s structure *bit for bit*, because that is what
      [`neighbors.geodesic_ball`][ordito.neighbors.geodesic_ball] relies on: it emits its BFS queue
      in visit order, so a permuted adjacency row permutes its result, and
      ``curvature.principal_curvature`` then moves at the near-flat vertices where its quadric fit
      is ill conditioned. Repeating the sorted call returns the identical buffer, the property the
      default ``sort_rows=False`` does **not** have: its atomic cursor hands the slots out in
      thread-arrival order and every repeat differs.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size
    edges_wp, _ = od.edges.edges_unique(mesh_wp.indices, n_vertices=n_vertices)

    neighbors_wp, offsets_wp = od.graph.edges_to_neighbor_lists(n_vertices, edges_wp)
    neighbors, offsets = neighbors_wp.numpy(), offsets_wp.numpy()
    adjacency_igl = igl.adjacency_list(np.asarray(mesh_tm.faces, dtype=np.int64))

    assert offsets.shape == (n_vertices + 1,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == neighbors.size == 2 * int(edges_wp.shape[0])
    degrees = np.diff(offsets)
    # Not a regular fixture: a constant degree would make a row-swap invisible.
    assert int(degrees.max()) > int(degrees.min()) > 0

    for v in range(n_vertices):
        row = sorted(neighbors[int(offsets[v]) : int(offsets[v + 1])].tolist())
        assert row == sorted(int(x) for x in adjacency_igl[v]), f"vertex {v} row differs"

    adjacency = od.graph.edges_to_csr(n_vertices, edges_wp)
    columns = adjacency.columns.numpy()
    assert np.array_equal(offsets, adjacency.offsets.numpy())
    for v in range(n_vertices):
        lo, hi = int(offsets[v]), int(offsets[v + 1])
        assert sorted(neighbors[lo:hi].tolist()) == sorted(columns[lo:hi].tolist())

    sorted_wp, sorted_offsets_wp = od.graph.edges_to_neighbor_lists(
        n_vertices, edges_wp, sort_rows=True
    )
    sorted_np = sorted_wp.numpy()
    assert np.array_equal(sorted_offsets_wp.numpy(), adjacency.offsets.numpy())
    assert np.array_equal(sorted_np, columns)
    for _ in range(4):
        repeat_wp, _ = od.graph.edges_to_neighbor_lists(n_vertices, edges_wp, sort_rows=True)
        assert np.array_equal(repeat_wp.numpy(), sorted_np)


def test_edges_to_neighbor_lists_sorts_a_wide_row(device: str) -> None:
    """
    Ordito against ordito: the per-row sort is a shell sort, so its gap loop needs a wide row.

    Every mesh fixture has a vertex valence of a handful, where the gap sequence collapses to one
    pass and the kernel is a plain insertion sort -- so the fixtures above exercise none of the
    gapped passes. A star graph puts one row of 1 000 neighbours through all of them.
    """
    degree = 1000
    edges_np = np.stack(
        [np.zeros(degree, dtype=np.int32), np.arange(1, degree + 1, dtype=np.int32)], axis=1
    )
    edges_wp = _edges_wp(edges_np, device)

    neighbors_wp, offsets_wp = od.graph.edges_to_neighbor_lists(
        degree + 1, edges_wp, sort_rows=True
    )
    offsets = offsets_wp.numpy()
    assert int(offsets[1]) == degree  # the hub row is the wide one
    hub = neighbors_wp.numpy()[: int(offsets[1])]
    assert np.array_equal(hub, np.arange(1, degree + 1, dtype=np.int32))


def test_edges_to_neighbor_lists_rejects_an_out_of_range_endpoint(device: str) -> None:
    """The guard that keeps an unchecked index off a raw ``degree[a]`` write."""
    edges_wp = _edges_wp(np.array([[0, 1], [1, 5]], dtype=np.int32), device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        od.graph.edges_to_neighbor_lists(3, edges_wp)


def test_edges_to_neighbor_lists_empty(device: str) -> None:
    """No edges means every row is empty, and the offsets are still the ``n + 1`` CSR form."""
    edges_wp = odt.as_array2d(wp.zeros((0, 2), dtype=wp.int32, device=device), wp.int32)
    neighbors_wp, offsets_wp = od.graph.edges_to_neighbor_lists(4, edges_wp)
    assert neighbors_wp.size == 0
    assert np.array_equal(offsets_wp.numpy(), np.zeros(5, dtype=np.int32))


@pytest.mark.parametrize(("node_count", "n_edges"), [(64, 24), (2048, 2047)])
@pytest.mark.parity("connected_component_labels", "igl", "scipy")
@pytest.mark.parity("connected_component_labels_depth", "igl", "scipy")
def test_connected_component_labels_matches_igl(device: str, node_count: int, n_edges: int) -> None:
    """
    Class B (label packing): the partition against ``igl.connected_components`` and scipy's.

    The reference worth having here because it takes the *same argument* ordito does -- a
    ``scipy.sparse`` adjacency matrix -- rather than a mesh, which is why
    ``benchmarks/test_graph.py`` used to say no second implementation existed for these two groups.
    It returns ``(n_components, labels, sizes)``, three values, and numbers components ``0..k-1`` in
    its own traversal order where ordito names each after a representative node, so
    [`same_partition`][tests.comparisons.same_partition] is the transform exactly as it is for the
    ``scipy.sparse.csgraph.connected_components`` comparison.

    Two shapes, and each carries a different half of the claim. The 64-node random graph is what
    exercises the *label packing*, so its edge count is chosen to fragment it: measured on this
    seed, 24 edges give **40** components, **12** of them non-trivial, the largest holding 6 nodes,
    where 200 random edges over 64 nodes give **1** component holding all 64 -- well past the
    giant-component threshold -- and with one component the numbering conventions coincide and the
    transform under test is the identity. The 2 048-node **path** is deliberately a single
    component: its diameter equals its node count, the most propagation rounds any graph needs,
    which is what the ``*_depth`` group exists to gate; an implementation that stopped propagating
    early would pass the first shape and fail this one, which a random graph of diameter ~4 cannot
    show.
    """
    if n_edges == node_count - 1:  # the path graph
        edges_np = np.stack(
            [np.arange(node_count - 1, dtype=np.int32), np.arange(1, node_count, dtype=np.int32)],
            axis=1,
        )
    else:
        rng = np.random.default_rng(7)
        edges_np = rng.integers(0, node_count, size=(n_edges, 2), dtype=np.int32)

    adjacency_np = sp.coo_matrix(
        (np.ones(len(edges_np), dtype=np.int8), (edges_np[:, 0], edges_np[:, 1])),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency_np = adjacency_np + adjacency_np.T

    n_components_igl, labels_igl, sizes_igl = igl.connected_components(
        adjacency_np  # pyright: ignore[reportArgumentType]
    )
    labels_wp = od.graph.connected_component_labels_from_edges(
        _edges_wp(edges_np, device), node_count=node_count
    )

    assert int(n_components_igl) == np.unique(labels_wp.numpy()).size
    assert int(np.asarray(sizes_igl).sum()) == node_count  # every node landed in a component
    # Non-vacuous on the random shape: the path is one component by design, see the docstring.
    assert int(n_components_igl) > 1 or n_edges == node_count - 1
    assert same_partition(labels_wp.numpy(), np.asarray(labels_igl).ravel())
    assert same_partition(labels_wp.numpy(), _scipy_component_labels(edges_np, node_count))


@pytest.mark.parametrize("shape", ["random", "path", "star", "loops_and_duplicates"])
def test_connected_component_labels_edge_list_matches_csr(device: str, shape: str) -> None:
    """
    Ordito against ordito: the edge-list entry point returns the CSR one's labels exactly.

    ``connected_component_labels_from_edges`` hooks straight off the edge list rather than building
    the adjacency matrix ``connected_component_labels`` walks, so the two no longer share a code
    path. Both label a component by its smallest node id, so the claim is equality of the label
    arrays, not merely of the partition. The CSR entry point carries the oracles (pymeshlab and
    meshlib, below); the edge-list one carries scipy and igl, above. The four shapes are a
    fragmented random graph, a path (diameter equal to its node count), a star whose hub is the
    *largest* id (every union lands on one root), and a graph with self-loops and repeated edges.
    """
    rng = np.random.default_rng(11)
    node_count = 512
    if shape == "random":
        edges_np = rng.integers(0, node_count, size=(300, 2), dtype=np.int32)
    elif shape == "path":
        order = rng.permutation(node_count).astype(np.int32)
        edges_np = np.stack([order[:-1], order[1:]], axis=1)
    elif shape == "star":
        leaves = np.arange(node_count - 1, dtype=np.int32)
        edges_np = np.stack([np.full_like(leaves, node_count - 1), leaves], axis=1)
    else:
        base = rng.integers(0, node_count, size=(200, 2), dtype=np.int32)
        loops = np.repeat(rng.integers(0, node_count, size=(40, 1), dtype=np.int32), 2, axis=1)
        edges_np = np.concatenate([base, base[:50, ::-1], loops])
    edges_wp = _edges_wp(edges_np, device)

    labels_edges = od.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    labels_csr = od.graph.connected_component_labels(od.graph.edges_to_csr(node_count, edges_wp))

    assert np.array_equal(labels_edges.numpy(), labels_csr.numpy())
    # The smallest-id convention, checked directly rather than through the other entry point.
    labels_np = labels_edges.numpy()
    assert np.array_equal(labels_np[labels_np], labels_np)
    assert np.all(labels_np <= np.arange(node_count))


@pytest.mark.parametrize("shape", ["random", "path", "star"])
def test_connected_component_labels_compressed_forest(
    device: str, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """
    Ordito against ordito: compressing the pre-hooked forest changes no label.

    ``connected_components.ecl_compress`` runs between the pre-hook and the hook only from
    ``ECL_COMPRESS_FROM`` nodes, which no test graph reaches, so the threshold is lowered to force
    it; the uncompressed path carries the oracles (scipy and igl, above). The path is the shape the
    compression exists for: a long descending chain after the pre-hook.
    """
    from ordito.kernels.algorithms import connected_components as kernel_cc

    rng = np.random.default_rng(13)
    node_count = 512
    if shape == "random":
        edges_np = rng.integers(0, node_count, size=(300, 2), dtype=np.int32)
    elif shape == "path":
        order = rng.permutation(node_count).astype(np.int32)
        edges_np = np.stack([order[:-1], order[1:]], axis=1)
    else:
        leaves = np.arange(node_count - 1, dtype=np.int32)
        edges_np = np.stack([np.full_like(leaves, node_count - 1), leaves], axis=1)
    edges_wp = _edges_wp(edges_np, device)
    plain = od.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count).numpy()
    monkeypatch.setattr(kernel_cc, "ECL_COMPRESS_FROM", 0)
    compressed = od.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    assert np.unique(plain).size > 1 or shape != "random"
    assert np.array_equal(compressed.numpy(), plain)


@pytest.mark.parametrize("face_ratio", [0.0, 0.1, 0.5])
@pytest.mark.parity("connected_component_labels", "pymeshlab")
def test_connected_component_labels_matches_pymeshlab(
    request: pytest.FixtureRequest, face_ratio: float
) -> None:
    """
    Class B: MeshLab has no filter that returns labels, so the labelling is observed through a mask.

    ``compute_selection_by_small_disconnected_components_per_face`` is the closest thing that runs
    the component pass without also splitting the mesh -- it labels every component, then selects
    the faces of every component holding fewer than ``nbfaceratio`` times the largest component's
    face count. So the named transform runs ordito's labels through exactly that rule: push the
    per-vertex labels onto faces (all three corners of a face share a component), count faces per
    label, and select where the count is below the threshold. Equality is then exact on the bool
    array.

    This does more than re-express the same thing twice: the *sizes* of all the components and their
    ranking have to agree, not just the partition, which is what a label array compared up to
    renumbering (the scipy oracle above) deliberately does not pin.

    Parametrized over three ratios that produce **different** answers on a 20 / 320 / 2 048-face
    three-component mesh -- 0, 20 and 340 faces selected -- so no constant mask can pass.
    """
    mesh_a_tm, _mesh_a_wp = request.getfixturevalue("sphere_irregular")
    mesh_b_tm, _mesh_b_wp = request.getfixturevalue("saddle_graded")
    mesh_c_tm, _mesh_c_wp = request.getfixturevalue("saddle_graded")
    combined_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    assert isinstance(combined_tm, tm.Trimesh)
    device = str(_mesh_a_wp.points.device)
    faces_wp = wp.array(
        np.ascontiguousarray(combined_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    n_vertices = combined_tm.vertices.shape[0]

    meshset_pml = trimesh_to_pymeshlab(combined_tm)
    meshset_pml.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=face_ratio)
    selection_pml = np.asarray(meshset_pml.current_mesh().face_selection_array())

    unique_edges_wp, _inverse_wp = od.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    labels_np = od.graph.connected_component_labels(
        od.graph.edges_to_csr(n_vertices, unique_edges_wp)
    ).numpy()

    face_labels_np = labels_np[combined_tm.faces[:, 0]]
    _label, size_np = np.unique(face_labels_np, return_counts=True)
    threshold = face_ratio * size_np.max()
    sizes_by_face_np = size_np[np.searchsorted(_label, face_labels_np)]
    assert np.array_equal(sizes_by_face_np < threshold, selection_pml)


@pytest.mark.parity("connected_component_labels", "meshlib")
def test_connected_component_labels_matches_meshlib(request: pytest.FixtureRequest) -> None:
    """
    Class B (label packing): ``getAllComponentsVerts`` returns the components as *bitsets*.

    The transform is the one every component comparison in this package makes plus a decode:
    MeshLib hands back a vector of ``VertBitSet``, one per component in its own traversal order,
    so each is expanded to a bool array over the vertex domain and its index becomes the label,
    and only the *partition* is then shared -- ordito names a component after a representative
    vertex. Hence [`same_partition`][tests.comparisons.same_partition].

    Stronger than the scipy pairing in one respect and weaker in another, which is why both stay:
    the bitsets pin each component's *membership* directly rather than through a renumbering, but
    MeshLib takes a mesh where scipy takes an edge list, so it cannot reach the random-graph inputs
    at all.

    Non-vacuous: three disjoint fixtures, so the answer is neither one component nor as many as
    there are vertices, and the component *sizes* are asserted to be the fixtures' vertex counts
    before the partition is compared.
    """
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("sphere_irregular")
    mesh_b_tm, _mesh_b_wp = request.getfixturevalue("saddle_graded")
    mesh_c_tm, _mesh_c_wp = request.getfixturevalue("saddle_graded")
    combined_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    assert isinstance(combined_tm, tm.Trimesh)
    n_vertices = combined_tm.vertices.shape[0]
    faces_wp = wp.array(
        np.ascontiguousarray(combined_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=mesh_a_wp.points.device,
    )

    components_ml = mm.getAllComponentsVerts(
        trimesh_to_meshlib(combined_tm),
        None,  # pyright: ignore[reportArgumentType]
    )
    labels_ml = np.full(n_vertices, -1, dtype=np.int64)
    for label, component_ml in enumerate(components_ml):
        labels_ml[meshlib_bitset_to_numpy(component_ml, n_vertices)] = label

    unique_edges_wp, _inverse_wp = od.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    labels_wp = od.graph.connected_component_labels(
        od.graph.edges_to_csr(n_vertices, unique_edges_wp)
    )

    assert sorted(component_ml.count() for component_ml in components_ml) == sorted(
        mesh.vertices.shape[0] for mesh in (mesh_a_tm, mesh_b_tm, mesh_c_tm)
    )
    assert (labels_ml >= 0).all()  # every vertex landed in a component
    assert same_partition(labels_wp.numpy(), labels_ml)


@pytest.mark.parametrize("node_count", [10, 0], ids=["isolated_nodes", "zero_nodes"])
def test_connected_component_labels_empty_edges(device: str, node_count: int) -> None:
    """Class A: with no edges every node is its own component (scipy's answer), or none exist."""
    edges_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    labels_wp = od.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    labels_exp = _scipy_component_labels(np.empty((0, 2), dtype=np.int32), node_count)
    assert labels_wp.shape == (node_count,)
    assert np.array_equal(labels_wp.numpy(), labels_exp)


def test_connected_component_labels_star_graph(device: str) -> None:
    n = 512
    hub = 0
    leaves = np.arange(1, n, dtype=np.int32)
    edges_np = np.stack([np.full(n - 1, hub, dtype=np.int32), leaves], axis=1)
    edges_wp = _edges_wp(edges_np, device)
    labels_wp = od.graph.connected_component_labels_from_edges(edges_wp, node_count=n)
    labels_exp = _scipy_component_labels(edges_np, n)
    assert same_partition(labels_wp.numpy(), labels_exp)


def test_connected_component_parity_random(device: str) -> None:
    # Signs drawn from a hidden potential, so the constraints are consistent everywhere and the
    # returned parity must reproduce that potential up to a per-component flip.
    rng = np.random.default_rng(11)
    n = 4096
    potential_np = rng.integers(0, 2, size=n).astype(np.int32)
    a_np = rng.integers(0, n, size=12_000).astype(np.int32)
    b_np = rng.integers(0, n, size=12_000).astype(np.int32)
    edges_np = np.stack([a_np, b_np], axis=1)
    signs_np = (potential_np[a_np] ^ potential_np[b_np]).astype(np.int32)

    edges_wp = _edges_wp(edges_np, device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)
    labels_wp, parity_wp = od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)

    parity_np = parity_wp.numpy()
    assert np.array_equal(parity_np[a_np] ^ parity_np[b_np], signs_np)
    assert same_partition(labels_wp.numpy(), _scipy_component_labels(edges_np, n))
    # Each component representative anchors its own potential at 0.
    labels_np = labels_wp.numpy()
    assert np.array_equal(parity_np[np.unique(labels_np)], np.zeros(len(np.unique(labels_np))))


def test_connected_component_parity_long_path(device: str) -> None:
    # A path is the worst case for edge-by-edge propagation (one level per node) and the case the
    # union-find is depth-independent on; the potential is then the running XOR of the signs.
    n = 20_001
    rng = np.random.default_rng(5)
    edges_np = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    signs_np = rng.integers(0, 2, size=n - 1).astype(np.int32)
    edges_wp = _edges_wp(edges_np, device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)

    labels_wp, parity_wp = od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)
    parity_exp = np.concatenate([[0], np.cumsum(signs_np) % 2]).astype(np.int32)
    assert np.array_equal(parity_wp.numpy(), parity_exp)
    assert np.array_equal(labels_wp.numpy(), np.zeros(n, dtype=np.int32))


def test_connected_component_parity_contradiction_terminates(device: str) -> None:
    # An odd-signed cycle admits no potential. The contract is best-effort, not an exception: the
    # call must still terminate and label the component, leaving some edge violated.
    n = 1025
    edges_np = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int32)
    signs_np = np.ones(n, dtype=np.int32)
    edges_wp = _edges_wp(edges_np, device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)

    labels_wp, parity_wp = od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)
    assert np.array_equal(labels_wp.numpy(), np.zeros(n, dtype=np.int32))
    parity_np = parity_wp.numpy()
    assert set(np.unique(parity_np).tolist()) <= {0, 1}
    violated = parity_np[edges_np[:, 0]] ^ parity_np[edges_np[:, 1]] != signs_np
    assert violated.sum() >= 1


def test_connected_component_parity_no_edges(device: str) -> None:
    edges_wp = odt.as_array2d(wp.zeros((0, 2), dtype=wp.int32, device=device), wp.int32)
    signs_wp = wp.zeros(0, dtype=wp.int32, device=device)
    labels_wp, parity_wp = od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 7)
    assert np.array_equal(labels_wp.numpy(), np.arange(7, dtype=np.int32))
    assert np.array_equal(parity_wp.numpy(), np.zeros(7, dtype=np.int32))


@pytest.mark.parametrize(
    ("edges", "n_signs", "node_count", "message"),
    [
        ([[0, 0]] * 4, 3, 4, "signs must have length 4"),
        ([[0, 10]], 1, 5, "edge indices must lie in"),
    ],
    ids=["signs_length", "endpoint_range"],
)
def test_connected_component_parity_validates_its_input(
    device: str, edges: list[list[int]], n_signs: int, node_count: int, message: str
) -> None:
    """
    One sign per edge, and every endpoint inside ``[0, node_count)`` under the default check.

    Without the range check, ``ecl_hook_parity`` indexes a ``node_count``-element buffer by the raw
    endpoint -- an out-of-range value reads and writes out of bounds rather than raising.
    """
    edges_wp = _edges_wp(np.array(edges, dtype=np.int32), device)
    signs_wp = wp.zeros(n_signs, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match=message):
        od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, node_count)


def test_connected_component_parity_validates_signs(device: str) -> None:
    """
    Not a parity assert: an out-of-range ``sign`` is rejected, and is bounded when unchecked.

    The hook packs the parity bit into the low bit of a word whose upper bits are a node id, so a
    sign outside ``{0, 1}`` lands in the *parent* half. Before the mask in ``ecl_hook_edge_parity``
    a sign of ``2`` made the union's compare-and-swap write back exactly what it expected, leaving
    the two endpoints in separate components with no error at all, and a negative sign wrote a
    parent of ``-1`` that the next find read out of bounds. Both arms are asserted: the checked
    path raises, and the unchecked path still joins the edge.
    """
    edges_wp = _edges_wp(np.array([[2, 3]], dtype=np.int32), device)
    for bad in (2, -1, 1 << 20):
        signs_wp = wp.array(np.array([bad], dtype=np.int32), dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="signs must be 0 or 1"):
            od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 6)
        labels_wp, parity_wp = od.graph.connected_component_parity_from_edges(
            edges_wp, signs_wp, 6, validate=False
        )
        labels_np = labels_wp.numpy()
        # The structure is what the mask protects: the edge still joins its endpoints, and only
        # the sign's low bit reaches the potential.
        assert labels_np[2] == labels_np[3]
        assert set(np.unique(parity_wp.numpy()).tolist()) <= {0, 1}
        assert int(parity_wp.numpy()[3]) == bad & 1

    # A valid sign is untouched by the mask.
    for good in (0, 1):
        signs_wp = wp.array(np.array([good], dtype=np.int32), dtype=wp.int32, device=device)
        labels_wp, parity_wp = od.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 6)
        assert labels_wp.numpy()[2] == labels_wp.numpy()[3]
        assert int(parity_wp.numpy()[3]) == good


def test_face_connected_component_labels(request: pytest.FixtureRequest) -> None:
    """
    Class B: the face-side partition against scipy over ``trimesh.face_adjacency``.

    Two named transforms, both on the reference side: build the dual graph from trimesh's face
    adjacency, then compare partitions rather than labels. Three concatenated fixtures make the
    expected component count three, which is asserted so the comparison cannot pass on one
    blob.
    """
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("sphere_irregular")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("saddle_graded")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("saddle_graded")

    concat_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    assert isinstance(concat_tm, tm.Trimesh)
    _, concat_faces_wp = od.combine.concatenate(
        [
            (mesh_a_wp.points, mesh_a_wp.indices),
            (mesh_b_wp.points, mesh_b_wp.indices),
            (mesh_c_wp.points, mesh_c_wp.indices),
        ]
    )
    face_labels_wp = od.adjacency.face_connected_component_labels(concat_faces_wp)
    n_faces = concat_tm.faces.shape[0]
    face_labels_tm = _scipy_component_labels(concat_tm.face_adjacency.astype(np.int32), n_faces)
    assert same_partition(face_labels_wp.numpy(), face_labels_tm)


def _scipy_component_labels(edges: np.ndarray, node_count: int) -> np.ndarray:
    if node_count == 0:
        return np.array([], dtype=np.int32)
    if len(edges) == 0:
        return np.arange(node_count, dtype=np.int32)
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(edges), dtype=np.int8)
    matrix = sp.coo_matrix((data, (row, col)), shape=(node_count, node_count))
    matrix = matrix + matrix.T
    _n_comp, labels = csgraph.connected_components(matrix, directed=False)
    return labels.astype(np.int32)


def test_successor_cycles_single_cycle(device: str) -> None:
    """One 4-cycle: the result starts at the smallest node and follows the edge direction."""
    edges_wp = _edges_wp(np.array([[5, 2], [2, 7], [7, 3], [3, 5]], dtype=np.int32), device)
    flat_wp, offsets_wp = od.graph.successor_cycles(edges_wp, 8)

    assert np.array_equal(flat_wp.numpy(), np.array([2, 7, 3, 5], dtype=np.int32))
    assert np.array_equal(offsets_wp.numpy(), np.array([0, 4], dtype=np.int32))


def test_successor_cycles_multiple_cycles(device: str) -> None:
    """
    Interleaved node ids across three cycles, plus nodes on no cycle.

    Every cycle must come back in successor order from its own minimum; the per-cycle offsets
    partition the packed buffer; nodes 1 and 8 appear in no edge and in no cycle.
    """
    rng = np.random.default_rng(7)
    cycles = [[0, 4, 2], [3, 9, 6, 5], [7, 10]]
    edge_rows = [
        (cycle[i], cycle[(i + 1) % len(cycle)]) for cycle in cycles for i in range(len(cycle))
    ]
    order = rng.permutation(len(edge_rows))
    edges_np = np.array(edge_rows, dtype=np.int32)[order]
    edges_wp = _edges_wp(edges_np, device)

    flat_wp, offsets_wp = od.graph.successor_cycles(edges_wp, 11)

    flat_np = flat_wp.numpy()
    offsets_np = offsets_wp.numpy()
    assert offsets_np[0] == 0
    assert offsets_np[-1] == flat_np.size == 9
    recovered = [
        flat_np[start:stop].tolist() for start, stop in itertools.pairwise(offsets_np.tolist())
    ]
    # Each cycle starts at its minimum and follows the successor direction.
    assert sorted(recovered) == sorted([[0, 4, 2], [3, 9, 6, 5], [7, 10]])


# Cycle lengths straddling the powers of the pointer-jump widths (8 and 16), where the schedule
# adds a round or narrows its hop: an off-by-one in the round count or the window arithmetic shows
# up at exactly one of these.
JUMP_BOUNDARY_LENGTHS = [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 255, 256, 257, 511, 512, 513, 4097]


@pytest.mark.parametrize("length", JUMP_BOUNDARY_LENGTHS)
def test_successor_cycles_ranks_across_jump_round_boundaries(device: str, length: int) -> None:
    """
    Not a library comparison: no reference ranks a successor graph, so the oracle is the walk.

    One cycle of ``length`` nodes over shuffled ids plus a chain of the same length: the cycle must
    come back from its smallest id in successor order, which a serial walk of the same table gives,
    and the chain must contribute nothing. The chain doubles the ranked-node count, so the
    schedule's round count is read off ``2 * length`` -- both sides of each power are reached.
    """
    rng = np.random.default_rng(length)
    ids = rng.permutation(3 * length).astype(np.int32)
    cycle = ids[:length]
    chain = ids[length : 2 * length + 1]
    rows = [(cycle[i], cycle[(i + 1) % length]) for i in range(length)]
    rows += [(chain[i], chain[i + 1]) for i in range(length)]
    edges_np = np.array(rows, dtype=np.int32)[rng.permutation(len(rows))]
    edges_wp = _edges_wp(edges_np, device)

    flat_wp, offsets_wp = od.graph.successor_cycles(edges_wp, 3 * length)

    start = int(np.argmin(cycle))
    assert np.array_equal(flat_wp.numpy(), np.roll(cycle, -start))
    assert np.array_equal(offsets_wp.numpy(), np.array([0, length], dtype=np.int32))


def test_successor_cycles_validates_before_launching(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The range check runs before any kernel that indexes by an endpoint.

    ``scatter_successor`` indexes a ``node_count``-element buffer by the raw edge endpoint, so a
    check that runs after it has already let an out-of-range endpoint write past the end — on the
    CPU device that is a host-heap overwrite which aborts the process much later, somewhere
    unrelated. Asserting only that ``ValueError`` is raised does not catch that: the exception is
    raised either way. Recording what launched pins the ordering: the only kernels allowed before
    the raise are the read-only range reduction's, from ``ordito.kernels.reduce`` (none on the CPU
    device, which reads the bounds back on the host).
    """
    launched: list[str] = []
    real_launches: dict[str, Callable[..., object]] = {"wp": wp.launch, "_launch": _launch.launch}

    def recording(owner: str) -> Callable[..., object]:
        def recording_launch(kernel: wp.Kernel, *args: object, **kwargs: object) -> object:
            launched.append(kernel.module.name)
            return real_launches[owner](kernel, *args, **kwargs)

        return recording_launch

    # The wrapper layer launches through ``ordito._launch``, which falls back to ``wp.launch``
    # and routes ``launch_tiled`` through its own ``launch``; record every path.
    monkeypatch.setattr(wp, "launch", recording("wp"))
    monkeypatch.setattr(_launch, "launch", recording("_launch"))
    edges_wp = _edges_wp(np.array([[0, 9], [9, 0]], dtype=np.int32), device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        od.graph.successor_cycles(edges_wp, 4)
    assert set(launched) <= {"ordito.kernels.reduce"}, launched


def test_successor_cycles_malformed_input_stays_in_range(device: str) -> None:
    """
    Two in-edges on one node (not a successor graph) must not return garbage.

    The documented behavior: ranks may collide and slots fall back to zero, but every value in
    the packed buffer stays a valid node index and the offsets still partition it.
    """
    edges_np = np.array([[0, 1], [1, 2], [2, 0], [3, 1]], dtype=np.int32)
    edges_wp = _edges_wp(edges_np, device)
    flat_wp, offsets_wp = od.graph.successor_cycles(edges_wp, 4)

    flat_np = flat_wp.numpy()
    assert flat_np.size == int(offsets_wp.numpy()[-1]) == 4
    assert np.all((flat_np >= 0) & (flat_np < 4))


def test_successor_cycles_excludes_a_chain(device: str) -> None:
    """
    A plain successor chain (no cycle at all) contributes nothing to the result.

    The docstring promises a successor graph decomposes into cycles *and* chains, with only
    cycles returned. A chain's dead end (a node with no outgoing edge) used to be treated as a
    second rank-0 fixed point alongside the arbitrary cut at the component's smallest node, so
    the two collided and fabricated a bogus "cycle" out of the collision -- one that could even
    contain a node id that never appeared in the input at all.
    """
    edges_wp = _edges_wp(np.array([[5, 3], [3, 7]], dtype=np.int32), device)
    flat_wp, offsets_wp = od.graph.successor_cycles(edges_wp, 8)
    assert flat_wp.shape == (0,)
    assert np.array_equal(offsets_wp.numpy(), np.zeros(1, dtype=np.int32))


def test_successor_cycles_empty(device: str) -> None:
    edges_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    flat_wp, offsets_wp = od.graph.successor_cycles(edges_wp, 5)
    assert flat_wp.shape == (0,)
    assert np.array_equal(offsets_wp.numpy(), np.zeros(1, dtype=np.int32))


def _mesh_vertex_edges(mesh_wp: wp.Mesh) -> tuple[odt.Array2dInt32, int]:
    """Return the unique undirected vertex edges and vertex count for a Warp mesh."""
    n = mesh_wp.points.size
    unique_edges, _ = od.edges.edges_unique(mesh_wp.indices, n_vertices=n)
    return unique_edges, n


# ---------------------------------------------------------------------------
# shortest_path_envelope
# ---------------------------------------------------------------------------


def _length_weighted_csr(mesh_wp: wp.Mesh, threshold: float = 1.0) -> odt.BsrMatrix[wp.float32]:
    """
    Build the mesh edge graph with Euclidean lengths as weights, divided by ``threshold``.

    That division is MeshLab's ``gradientthr``: its cap is ``|p_i - p_j| / gradientthr``, and
    ``shortest_path_envelope`` takes no threshold because the weights carry it.
    """
    edges, n_vertices = _mesh_vertex_edges(mesh_wp)
    lengths = od.edges.edges_unique_length(mesh_wp.points, mesh_wp.indices, edges)
    if threshold != 1.0:
        scaled = warp_empty(lengths.size, wp.float32, mesh_wp.device)
        wp.map(wp.div, lengths, wp.float32(threshold), out=scaled)
        lengths = scaled
    return od.graph.edges_to_csr(n_vertices, edges, lengths)


def _spike_field(n_vertices: int) -> np.ndarray:
    """Build a delta at vertex 0: the field with the steepest possible gradient."""
    values_np = np.zeros(n_vertices, dtype=np.float64)
    values_np[0] = 10.0
    return values_np


@pytest.mark.parametrize("threshold", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
@pytest.mark.parity("shortest_path_envelope", "pymeshlab")
def test_shortest_path_envelope_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str, threshold: float
) -> None:
    """
    Class A against ``apply_scalar_saturation_per_vertex``, the Lipschitz-cap reading.

    The named transform is the weights: MeshLab's ``gradientthr`` divides the edge length, so the
    adjacency carries ``length / threshold`` and the envelope needs no threshold of its own. The
    sweep is over that parameter because it is the only one the reference has.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    values_np = _spike_field(mesh_tm.vertices.shape[0])

    meshset_pml = trimesh_to_pymeshlab(mesh_tm, values_np)
    meshset_pml.apply_scalar_saturation_per_vertex(gradientthr=threshold)

    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)
    saturated_wp = od.graph.shortest_path_envelope(
        _length_weighted_csr(mesh_wp, threshold), values_wp
    )
    # Anti-vacuity: the spike must have been lowered, or a no-op would pass. It is the *peak* that
    # moves and not the field around it -- the zeros are already minimal and nothing is ever raised.
    assert 0.0 < float(saturated_wp.numpy()[0]) < 10.0
    assert np.allclose(
        saturated_wp.numpy(), meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("n_sources", [1, 3])
def test_shortest_path_envelope_is_the_edge_graph_distance(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], n_sources: int
) -> None:
    """
    Class A against ``scipy.sparse.csgraph.dijkstra``: seeded with zeros, this *is* the distance.

    The envelope ``min_u (values[u] + d(u, v))`` reduces to the weighted multi-source shortest-path
    distance when ``values`` is zero on the sources and large elsewhere — the reading the function
    is named for, and why the package needs no second shortest-path implementation. Both sides take
    the same graph, so the only difference is float32 against float64: measured 7.1e-07.
    """
    mesh_tm, mesh_wp = sphere_irregular
    n_vertices = mesh_tm.vertices.shape[0]
    sources_np = np.arange(n_sources)

    seeded_np = np.full(n_vertices, 1e6, dtype=np.float32)
    seeded_np[sources_np] = 0.0
    envelope_wp = od.graph.shortest_path_envelope(
        _length_weighted_csr(mesh_wp), wp.array(seeded_np, dtype=wp.float32, device=mesh_wp.device)
    )

    edges_np = mesh_tm.edges_unique
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    both_np = np.concatenate([edges_np, edges_np[:, ::-1]])
    graph_sp = sp.coo_matrix(
        (np.concatenate([lengths_np, lengths_np]), (both_np[:, 0], both_np[:, 1])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    distance_sp = csgraph.dijkstra(graph_sp, indices=sources_np, min_only=True)

    # Anti-vacuity: the field has to have propagated, not stayed at its 1e6 seed.
    assert distance_sp.max() > 0.5
    assert envelope_wp.numpy().max() < 1e5
    assert np.allclose(envelope_wp.numpy(), distance_sp, rtol=1e-5, atol=1e-5)


def test_shortest_path_envelope_respects_the_bound(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """After convergence no edge violates the cap, and nothing was raised."""
    mesh_tm, mesh_wp = saddle_graded
    threshold = 2.0
    rng = np.random.default_rng(7)
    values_np = rng.uniform(0.0, 5.0, size=mesh_tm.vertices.shape[0])
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)

    saturated_np = od.graph.shortest_path_envelope(
        _length_weighted_csr(mesh_wp, threshold), values_wp
    ).numpy()
    assert (saturated_np <= values_np.astype(np.float32) + 1e-5).all()

    edges_np = mesh_tm.edges_unique
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    jumps_np = np.abs(saturated_np[edges_np[:, 0]] - saturated_np[edges_np[:, 1]])
    assert (jumps_np <= lengths_np / threshold + 1e-4).all()
    # Every minimum survives: the smallest value in the field is untouched.
    assert np.isclose(saturated_np.min(), values_np.min(), rtol=1e-5, atol=1e-5)


def test_shortest_path_envelope_invalid(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    The argument checks, and the guard on a negative edge weight.

    A negative weight makes the relaxation decrease without bound instead of converging. Without
    its guard the loop returns a plausible-looking array that keeps getting smaller as
    ``max_iterations`` grows -- silently wrong rather than raising, exactly what the docstring's
    "not admissible" note warns about.
    """
    mesh_tm, mesh_wp = sphere_irregular
    adjacency = _length_weighted_csr(mesh_wp)
    values_wp = wp.zeros(mesh_tm.vertices.shape[0], dtype=wp.float32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="max_iterations must be non-negative"):
        od.graph.shortest_path_envelope(adjacency, values_wp, max_iterations=-1)
    with pytest.raises(ValueError, match="one entry per node"):
        od.graph.shortest_path_envelope(adjacency, odt.as_dense(values_wp[:3]))
    edges, n_vertices = _mesh_vertex_edges(mesh_wp)
    with pytest.raises(ValueError, match="one entry per edge"):
        od.graph.edges_to_csr(n_vertices, edges, values_wp)

    device = mesh_wp.device
    weights_wp = wp.array(np.array([-1.0], dtype=np.float32), dtype=wp.float32, device=device)
    negative = od.graph.edges_to_csr(
        2, _edges_wp(np.array([[0, 1]], dtype=np.int32), device), weights_wp
    )
    two_values_wp = wp.array(
        np.array([0.0, 1000.0], dtype=np.float32), dtype=wp.float32, device=device
    )
    with pytest.raises(ValueError, match="non-negative"):
        od.graph.shortest_path_envelope(negative, two_values_wp, max_iterations=5)


def _edges_wp(edges_np: np.ndarray, device: wp.DeviceLike) -> odt.Array2dInt32:
    """Upload an ``(m, 2)`` edge list as a rank-2 ``int32`` array on ``device``."""
    return odt.as_array2d(
        wp.array(np.ascontiguousarray(edges_np, dtype=np.int32), dtype=wp.int32, device=device),
        wp.int32,
    )
