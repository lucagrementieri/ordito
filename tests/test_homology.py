"""
Structural tests for ``ordito.homology``.

**meshlib is the one reference that computes a basis**, through ``detectBasisTunnels``, and it is
compared here for what two bases share: the group they span, read off each loop's exact Z/2
homology class. potpourri3d does not bind geometry-central's homology code and neither trimesh nor
libigl computes one, so for everything else the invariants stand in for an oracle -- each loop has
to be a simple closed walk along real mesh edges, and non-contractible. A basis is not unique, so
nothing checks *which* loops come back: measured on a torus, ordito returns loops of 32 and 18
edges where MeshLib returns 72 and 32, both valid.
"""

from __future__ import annotations

import itertools
from collections.abc import Container

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from tests.conversions import numpy_to_warp, trimesh_to_meshlib


def _is_simple_edge_cycle(loop: np.ndarray, edges_tm: Container[tuple[int, ...]]) -> bool:
    """Whether a vertex sequence closes up along mesh edges without repeating a vertex."""
    if len(loop) < 3 or len(set(loop.tolist())) != len(loop):
        return False
    closed = np.append(loop, loop[0])
    return all(tuple(sorted((int(a), int(b)))) in edges_tm for a, b in itertools.pairwise(closed))


@pytest.mark.parametrize(("mesh_name", "genus"), [("sphere_irregular", 0), ("genus_two", 2)])
def test_homology_generator_count_is_twice_the_genus(
    request: pytest.FixtureRequest, mesh_name: str, genus: int
) -> None:
    """
    Not a library comparison: no reference computes a homology basis, so the claim is structural.

    The count is pinned by the Euler characteristic, which also fixes the tree-cotree identity
    ``E = (V - 1) + (F - 1) + 2g``: an implementation letting the primal tree take an edge the
    cotree already has, or leaving one in neither, returns the wrong count. trimesh supplies only
    the mesh's own edge set, used to check each loop *is* a walk along real edges -- closed, and
    visiting no vertex twice -- which excludes a "loop" jumping between unconnected vertices. The
    count assert makes the walk check non-vacuous on every genus above zero.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_tm = {tuple(sorted(edge)) for edge in mesh_tm.edges_unique.tolist()}
    loops = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    # The genus the fixture is built for, and the genus the mesh actually has, must agree first.
    assert od.measures.euler_characteristic(mesh_wp.indices) == 2 - 2 * genus
    assert len(loops) == 2 * genus
    for loop in loops:
        assert _is_simple_edge_cycle(loop.numpy(), edges_tm)


@pytest.mark.parametrize(("mesh_name", "genus"), [("torus_irregular", 1), ("genus_two", 2)])
@pytest.mark.parity("homology_generators", "meshlib")
def test_homology_generator_count_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, genus: int
) -> None:
    """
    Class B: equal after a named transform -- each loop's Z/2 homology class, one bit per cocycle.

    A homology basis is not unique, so the loops themselves are not comparable: on the torus ordito
    returns cycles of 32 and 18 edges where MeshLib's ``detectBasisTunnels`` returns 72 and 32. What
    two bases of the same group share is the group they span. Each loop is mapped to its class in
    ``H1(M; Z/2)`` by pairing it with ``2 * genus`` independent cocycles built from SciPy's own
    tree-cotree decomposition (``_z2_homology_classes``), an exact integer computation. Both
    libraries' ``(2g, 2g)`` class matrices must then have full rank over Z/2: every loop is
    non-contractible *and* no loop is a sum of the others. Both sides span all of ``H1``, so their
    spans are equal -- which is the comparison, and it is exact rather than a count.

    Mutation probes: replacing one of ordito's loops by another loop of the same basis (a duplicate
    generator, still simple, closed, non-contractible and correctly counted) drops ordito's rank to
    ``2g - 1`` and fails; so does a contractible loop (a face boundary), which pairs to zero with
    every cocycle. The old count-only comparison passed both.

    MeshLib's ``EdgeId`` walk is decoded as everywhere in this suite: ``org`` gives each step's
    tail and the last step's ``dest`` closes the loop, which must be simple.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.asarray(mesh_tm.faces)
    loops_wp = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    tunnels_ml = mm.detectBasisTunnels(mm.MeshPart(mesh_ml))

    assert len(tunnels_ml) == 2 * genus
    assert len(loops_wp) == len(tunnels_ml)
    walks_ml: list[list[int]] = []
    for tunnel_ml in tunnels_ml:
        walk_np = [int(mesh_ml.topology.org(edge_ml)) for edge_ml in tunnel_ml]
        assert len(walk_np) >= 3
        assert len(set(walk_np)) == len(walk_np)  # simple: no vertex repeats
        assert int(mesh_ml.topology.dest(tunnel_ml[-1])) == walk_np[0]  # and closed
        walks_ml.append(walk_np)

    classes_wp = _z2_homology_classes(faces_np, [loop.numpy().tolist() for loop in loops_wp])
    classes_ml = _z2_homology_classes(faces_np, walks_ml)
    assert _z2_rank(classes_wp) == 2 * genus
    assert _z2_rank(classes_ml) == 2 * genus


def _z2_homology_classes(faces_np: np.ndarray, walks: list[list[int]]) -> np.ndarray:
    """
    ``(len(walks), 2g)`` boolean Z/2 homology class of each closed vertex walk, built with SciPy.

    A breadth-first primal tree, then a spanning tree of the dual graph over the edges it left; the
    ``2g`` edges in neither are the generators. Cocycle ``k`` is one on generator ``k``, zero on
    every other generator and on the tree, and its cotree values are solved leaf to root over the
    dual tree so every face sums to zero. A cocycle vanishes on boundaries, so its pairing with a
    cycle depends only on the cycle's class, and the ``2g`` cocycles are independent.
    """
    corners = np.sort(faces_np[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, corner_edge = np.unique(corners, axis=0, return_inverse=True)
    corner_edge = corner_edge.reshape(-1, 3)
    n_vertices, n_faces, n_edges = int(faces_np.max()) + 1, len(faces_np), len(edges)
    graph = sp.coo_matrix(
        (np.ones(n_edges), (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)
    )
    _, parents = csgraph.breadth_first_order(graph, 0, directed=False, return_predecessors=True)
    children = np.flatnonzero(parents >= 0)
    edge_index = {(int(a), int(b)): e for e, (a, b) in enumerate(edges.tolist())}
    in_tree = np.zeros(n_edges, dtype=bool)
    for child in children.tolist():
        parent = int(parents[child])
        in_tree[edge_index[min(child, parent), max(child, parent)]] = True

    order = np.argsort(corner_edge.ravel(), kind="stable")
    edge_faces = np.repeat(np.arange(n_faces), 3)[order].reshape(-1, 2)
    free = np.flatnonzero(~in_tree)
    dual = sp.coo_matrix(
        (np.ones(len(free)), (edge_faces[free, 0], edge_faces[free, 1])), shape=(n_faces, n_faces)
    )
    face_order, face_parents = csgraph.breadth_first_order(
        csgraph.minimum_spanning_tree(dual.tocsr()), 0, directed=False, return_predecessors=True
    )
    dual_edge = {(int(edge_faces[e].min()), int(edge_faces[e].max())): int(e) for e in free}
    in_cotree = np.zeros(n_edges, dtype=bool)
    parent_edge = np.full(n_faces, -1)
    for face in face_order[1:].tolist():
        parent = int(face_parents[face])
        parent_edge[face] = dual_edge[min(face, parent), max(face, parent)]
        in_cotree[parent_edge[face]] = True

    generators = np.flatnonzero(~in_tree & ~in_cotree)
    cocycles = np.zeros((n_edges, len(generators)), dtype=bool)
    cocycles[generators, np.arange(len(generators))] = True
    # Leaves first: every other edge of the face is already set when its parent edge is solved.
    for face in face_order[:0:-1].tolist():
        face_edges = corner_edge[face]
        others = face_edges[face_edges != parent_edge[face]]
        cocycles[parent_edge[face]] = np.logical_xor.reduce(cocycles[others], axis=0)

    classes = np.zeros((len(walks), len(generators)), dtype=bool)
    for row, walk in enumerate(walks):
        for a, b in itertools.pairwise([*walk, walk[0]]):
            classes[row] ^= cocycles[edge_index[min(a, b), max(a, b)]]
    return classes


def _z2_rank(matrix: np.ndarray) -> int:
    """Rank over Z/2 of a boolean matrix, by Gaussian elimination."""
    rows = matrix.copy()
    rank = 0
    for column in range(rows.shape[1]):
        pivots = np.flatnonzero(rows[rank:, column])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        rows[[rank, pivot]] = rows[[pivot, rank]]
        below = np.flatnonzero(rows[:, column])
        rows[below[below != rank]] ^= rows[rank]
        rank += 1
        if rank == rows.shape[0]:
            break
    return rank


def test_homology_generators_are_not_contractible(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = torus_irregular
    loops = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    # A contractible loop bounds a disk, so cutting the *faces* along it would split the mesh in
    # two. Removing a genuine generator's vertices instead leaves the surface in one piece: the
    # standard cheap witness that a cycle wraps a handle rather than a disk.
    faces_np = np.asarray(mesh_tm.faces)
    for loop in loops:
        on_loop = np.zeros(len(mesh_tm.vertices), dtype=bool)
        on_loop[loop.numpy()] = True
        kept = faces_np[~on_loop[faces_np].any(axis=1)]
        assert len(kept) > 0
        components = tm.graph.connected_components(
            tm.geometry.faces_to_edges(kept), nodes=np.unique(kept)
        )
        assert len(components) == 1


def _greedy_loop_lengths(faces_np: np.ndarray, n_vertices: int, root: int) -> list[int]:
    """
    Hop lengths through ``root`` of the greedy system of loops, built with SciPy alone.

    A breadth-first tree from ``root``, then a maximum spanning tree of the dual graph over the
    edges the tree left, weighted by each edge's loop length through the root ``depth(a) + depth(b)
    + 1``; the edges in neither tree are the generators. The trees are SciPy's and break ties their
    own way; the weights depend only on the depths, which every breadth-first tree shares.
    """
    corners = np.sort(faces_np[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, corner_edge = np.unique(corners, axis=0, return_inverse=True)
    graph = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)
    )
    _, parents = csgraph.breadth_first_order(graph, root, directed=False, return_predecessors=True)
    depth = csgraph.shortest_path(graph, unweighted=True, directed=False, indices=root)
    children = np.flatnonzero(parents >= 0)
    tree = set(
        zip(
            np.minimum(children, parents[children]).tolist(),
            np.maximum(children, parents[children]).tolist(),
            strict=True,
        )
    )
    in_tree = np.array([(a, b) in tree for a, b in edges.tolist()])
    order = np.argsort(corner_edge.ravel(), kind="stable")
    edge_faces = np.repeat(np.arange(len(faces_np)), 3)[order].reshape(-1, 2)
    weight = depth[edges[:, 0]] + depth[edges[:, 1]] + 1
    candidates = np.flatnonzero(~in_tree)
    # A minimum spanning tree of ``max + 1 - weight`` is a maximum one of ``weight``.
    cost = weight[candidates].max() + 1 - weight[candidates]
    dual = sp.coo_matrix(
        (cost, (edge_faces[candidates, 0], edge_faces[candidates, 1])),
        shape=(len(faces_np), len(faces_np)),
    )
    forest = csgraph.minimum_spanning_tree(dual.tocsr()).tocoo()
    cotree = set(
        zip(
            np.minimum(forest.row, forest.col).tolist(),
            np.maximum(forest.row, forest.col).tolist(),
            strict=True,
        )
    )
    return sorted(
        int(weight[e])
        for e in candidates.tolist()
        if (int(edge_faces[e].min()), int(edge_faces[e].max())) not in cotree
    )


@pytest.mark.parametrize("mesh_name", ["torus", "torus_irregular", "genus_two"])
@pytest.mark.parametrize("root", [0, 5])
def test_homology_generators_are_the_greedy_system(
    request: pytest.FixtureRequest, mesh_name: str, root: int
) -> None:
    """
    Class B: equal after a named transform -- each loop's hop length through the root, sorted.

    The basis is the greedy system of loops through ``root`` (Erickson and Whittlesey): every
    generator is the shortest loop through the root independent of the ones before it, which is
    what makes a handle near the root come out as its own short loop. SciPy builds the same system
    from its own breadth-first tree and its own minimum spanning tree, and its trees break ties
    differently, so the loops are not comparable -- their weights are. A generator traced through
    the apex ``a`` of its two tree paths has hop length ``len(loop)``, and the based loop the
    greedy order ranks is that plus the stem to the root twice, ``len(loop) + 2 * depth(a)``.

    Mutation probe: ranking the dual forest by lowest edge index, the construction this replaced,
    fails every case (``torus`` at root 0 reads ``[33, 42]`` against ``[32,
    32]``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.asarray(mesh_tm.faces)
    n_vertices = len(mesh_tm.vertices)
    expected_np = _greedy_loop_lengths(faces_np, n_vertices, root)
    assert len(expected_np) >= 2  # non-vacuity: every fixture has a handle

    graph = sp.coo_matrix(
        (np.ones(faces_np.size), (faces_np.ravel(), np.roll(faces_np, -1, axis=1).ravel())),
        shape=(n_vertices, n_vertices),
    )
    depth = csgraph.shortest_path(graph, unweighted=True, directed=False, indices=root)
    loops = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices, root=root)
    based_lengths = sorted(int(loop.size + 2 * depth[loop.numpy()].min()) for loop in loops)
    assert based_lengths == expected_np


def test_homology_generators_reject_a_bad_root(torus_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the two ways a root can fail to be a vertex the faces reach."""
    _, mesh_wp = torus_irregular
    vertices_np = mesh_wp.points.numpy()
    with pytest.raises(ValueError, match="root must index a vertex"):
        od.homology.homology_generators(mesh_wp.points, mesh_wp.indices, root=len(vertices_np))
    # One extra vertex no face uses: in range, but the traversal from it reaches nothing.
    padded_wp = wp.array(
        np.vstack([vertices_np, vertices_np[:1]]), dtype=wp.vec3, device=mesh_wp.points.device
    )
    with pytest.raises(ValueError, match="is not referenced by any face"):
        od.homology.homology_generators(padded_wp, mesh_wp.indices, root=len(vertices_np))


def test_homology_generators_are_reproducible(genus_two: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: the docstring promises a reproducible basis, so pin it.

    Both spanning trees are built by parallel claims, and a basis that depended on which thread got
    there first would still pass every invariant in this file -- the loops would be simple, closed
    and non-contractible, just different ones. What makes it deterministic is that each claim is
    resolved by an extremum rather than by arrival: the primal tree takes the lowest-indexed
    frontier vertex (``wp.atomic_min`` on the parent) and the dual forest the incident edge with the
    longest loop through the root, the lowest index on a tie. Four runs, because a race that loses
    once in two is not one worth shipping.


    ``genus_two`` rather than the torus so there are four generators to disagree about, and the
    loops are compared *elementwise*: a count-only check passes on any valid basis, which is
    exactly what this is written to exclude.

    Mutation probe: replacing the ``wp.atomic_min`` claim with a plain racy write fails **this
    test's elementwise comparison and nothing else** -- the other tests in this file stay
    green, because the loops stay simple, closed, non-contractible and correctly counted. That is
    the whole argument for the test existing.
    """
    _, mesh_wp = genus_two
    runs = [
        [
            loop.numpy().tolist()
            for loop in od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)
        ]
        for _ in range(4)
    ]

    # Non-vacuity: an empty basis would compare equal to itself four times over.
    assert len(runs[0]) == 4
    assert all(len(loop) >= 3 for loop in runs[0])
    for run in runs[1:]:
        assert run == runs[0]


def test_homology_generators_reject_a_boundary(saddle_graded: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = saddle_graded
    with pytest.raises(ValueError, match="closed surface"):
        od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)


def test_homology_generators_ignore_an_unreferenced_vertex(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: an unreferenced vertex must not change the basis it plays no part in.

    The generator count is ``n_edges`` minus the two spanning trees' edge counts, so a primal tree
    rooted at a vertex carrying no edges spans one isolated node and hands every primal edge over
    as a generator. Vertex 0 is the one the root would default to, so prepending an unreferenced
    vertex is exactly the input that reaches it; measured before the fix, this returned 129
    generators instead of 2.
    """
    mesh_tm, _ = torus_irregular
    vertices_np = np.vstack([[[10.0, 10.0, 10.0]], np.asarray(mesh_tm.vertices)])
    faces_np = np.asarray(mesh_tm.faces) + 1
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    loops = od.homology.homology_generators(vertices_wp, faces_wp)

    # Non-vacuity: the unreferenced vertex really is unreferenced, and the surface really is genus
    # 1 -- ``euler_characteristic`` counts referenced vertices, so it is unmoved by the extra row.
    assert 0 not in set(faces_np.ravel().tolist())
    assert od.measures.euler_characteristic(faces_wp) == 0
    assert len(loops) == 2
    edges_tm = {tuple(sorted((int(a) + 1, int(b) + 1))) for a, b in mesh_tm.edges_unique.tolist()}
    for loop in loops:
        assert _is_simple_edge_cycle(loop.numpy(), edges_tm)


def test_homology_generators_reject_a_disconnected_surface(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: the connectivity precondition, which the counting argument needs.

    Two disjoint spheres are each genus 0, so an implementation that honoured the precondition
    would answer ``[]``; a single primal tree spans only the first component and hands the second's
    ``V - 1`` edges over as generators (measured: 41 of them). Both components are closed, so the
    boundary guard cannot see this.
    """
    mesh_tm, _ = sphere_irregular
    other = mesh_tm.copy()
    other.apply_translation([5.0, 0.0, 0.0])
    both = tm.util.concatenate([mesh_tm, other])
    assert isinstance(both, tm.Trimesh)
    vertices_wp, faces_wp = numpy_to_warp(np.asarray(both.vertices), np.asarray(both.faces), device)

    # Non-vacuity: the closed-surface guard passes, so connectivity is the only thing left to fail.
    assert int(od.boundary.boundary_edges(vertices_wp, faces_wp).shape[0]) == 0
    with pytest.raises(ValueError, match="connected surface"):
        od.homology.homology_generators(vertices_wp, faces_wp)


@pytest.mark.parametrize("n_vertices", [0, 4], ids=["no_vertices", "unreferenced_vertices"])
def test_homology_generators_without_edges(device: str, n_vertices: int) -> None:
    """
    Not a library comparison: the degenerate input the closed-surface guard must answer, not raise.

    A mesh with no faces has no boundary edges either, so it is vacuously closed and the honest
    answer is an empty basis. The boundary-count reduction is what used to raise here, with a
    ``ValueError`` whose message said the mesh had a boundary. The populated-vertex arm reaches the
    edge-count early return with vertices present; the empty arm has none at all.
    """
    vertices_wp = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)

    assert od.homology.homology_generators(vertices_wp, faces_wp) == []


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "genus_two"])
def test_homology_generators_is_its_packed_form_split(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the list form is the packed form, loop by loop.

    ``homology_generators`` carries the MeshLib count comparison and the invariants above; this
    pins ``homology_generators_with_offsets`` to it -- the same basis in the same order, with
    ``[0]`` offsets for the sphere -- and pins the precomputed ``unique_edges`` / ``edge_map`` to
    the derived numbering; half a pair raises.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    loops_wp = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)
    flat_wp, offsets_wp = od.homology.homology_generators_with_offsets(
        mesh_wp.points, mesh_wp.indices
    )

    offsets_np = offsets_wp.numpy()
    assert offsets_np.size == len(loops_wp) + 1
    assert offsets_np[-1] == flat_wp.size
    flat_np = flat_wp.numpy()
    for i, loop_wp in enumerate(loops_wp):
        assert np.array_equal(loop_wp.numpy(), flat_np[offsets_np[i] : offsets_np[i + 1]])

    unique_edges, edge_map = od.edges.edges_unique(mesh_wp.indices, n_vertices=mesh_wp.points.size)
    flat_supplied_wp, offsets_supplied_wp = od.homology.homology_generators_with_offsets(
        mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges, edge_map=edge_map
    )
    assert np.array_equal(flat_supplied_wp.numpy(), flat_np)
    assert np.array_equal(offsets_supplied_wp.numpy(), offsets_np)
    with pytest.raises(ValueError, match="together"):
        od.homology.homology_generators(mesh_wp.points, mesh_wp.indices, edge_map=edge_map)
