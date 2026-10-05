"""
Structural tests for ``ordito.homology``.

**meshlib is the one reference that computes a basis**, through ``detectBasisTunnels``, and it is
compared here for the only thing two bases can share: their *count*, forced by the Euler
characteristic. potpourri3d does not bind geometry-central's homology code and neither trimesh nor
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


@pytest.mark.parametrize(
    ("mesh_name", "genus"),
    # ``bohemian_dome`` is genus 1 like the torus but *self-intersecting*, so the generators are
    # found on a surface whose embedding gives no hint of where they run.
    [("icosahedron", 0), ("torus", 1), ("bohemian_dome", 1), ("genus_two", 2)],
)
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


@pytest.mark.parametrize(("mesh_name", "genus"), [("torus", 1), ("genus_two", 2)])
@pytest.mark.parity("homology_generators", "meshlib")
def test_homology_generator_count_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, genus: int
) -> None:
    """
    Class C (a count): a homology basis is not unique, so the loop *count* is all two share.

    ``detectBasisTunnels`` returns a vector of ``EdgeId`` paths -- MeshLib's own basis of
    non-contractible cycles -- and both libraries must find ``2 * genus`` of them. Which loops they
    are is free: on the torus ordito returns cycles of 32 and 18 edges where MeshLib returns 72 and
    32, and both are correct bases of the same first homology group.

    So the count is the comparison and the *structure* is what makes it non-vacuous: each of
    MeshLib's paths is checked to be a genuine closed edge walk with no repeated vertex, which is
    the same property [`test_homology_generator_count_is_twice_the_genus`] asserts on
    ordito's side. A reference returning ``2 * genus`` arbitrary edge lists would pass a bare count
    and fail this.

    The decode is the ``EdgeId`` one this suite uses throughout: ``org`` gives each step's tail and
    the last step's ``dest`` closes the loop.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    loops_wp = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    tunnels_ml = mm.detectBasisTunnels(mm.MeshPart(mesh_ml))

    assert len(tunnels_ml) == 2 * genus
    assert len(loops_wp) == len(tunnels_ml)
    for tunnel_ml in tunnels_ml:
        walk_np = [int(mesh_ml.topology.org(edge_ml)) for edge_ml in tunnel_ml]
        assert len(walk_np) >= 3
        assert len(set(walk_np)) == len(walk_np)  # simple: no vertex repeats
        assert int(mesh_ml.topology.dest(tunnel_ml[-1])) == walk_np[0]  # and closed


def test_homology_generators_are_not_contractible(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = torus
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


def test_homology_generators_are_reproducible(genus_two: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: the docstring promises a reproducible basis, so pin it.

    Both spanning trees are built by parallel claims, and a basis that depended on which thread got
    there first would still pass every invariant in this file -- the loops would be simple, closed
    and non-contractible, just different ones. What makes it deterministic is that each claim is
    resolved by an extremum rather than by arrival: the primal tree takes the lowest-indexed
    frontier vertex (``wp.atomic_min`` on the parent) and the dual forest the lowest-indexed
    incident edge. Four runs, because a race that loses once in two is not one worth shipping.

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


def test_homology_generators_reject_a_boundary(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = hemisphere
    with pytest.raises(ValueError, match="closed surface"):
        od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)


def test_homology_generators_ignore_an_unreferenced_vertex(
    torus: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: an unreferenced vertex must not change the basis it plays no part in.

    The generator count is ``n_edges`` minus the two spanning trees' edge counts, so a primal tree
    rooted at a vertex carrying no edges spans one isolated node and hands every primal edge over
    as a generator. Vertex 0 is the one the root would default to, so prepending an unreferenced
    vertex is exactly the input that reaches it; measured before the fix, this returned 129
    generators instead of 2.
    """
    mesh_tm, _ = torus
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
    icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: the connectivity precondition, which the counting argument needs.

    Two disjoint spheres are each genus 0, so an implementation that honoured the precondition
    would answer ``[]``; a single primal tree spans only the first component and hands the second's
    ``V - 1`` edges over as generators (measured: 41 of them). Both components are closed, so the
    boundary guard cannot see this.
    """
    mesh_tm, _ = icosahedron
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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "torus", "genus_two"])
def test_homology_generators_is_its_packed_form_split(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the list form is the packed form, loop by loop.

    ``homology_generators`` carries the MeshLib count comparison and the invariants above; this
    pins ``homology_generators_with_offsets`` to it -- the same basis in the same order, with
    ``[0]`` offsets for the sphere.
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
