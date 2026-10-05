"""Regression tests for ``ordito.halfedge`` against Trimesh (CPU reference)."""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from tests.conftest import MESHES

# ---------------------------------------------------------------------------
# halfedge_twins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_halfedge_twins_are_a_symmetric_pairing(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is the algebra.

    Three properties that together pin the pairing without a reference implementation -- it is an
    involution, no halfedge is its own twin, and twins run over the same undirected edge (that last
    is where ``trimesh.geometry.faces_to_edges`` comes in, as a *definition* of the edge a halfedge
    spans rather than as a second answer).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    twins_wp = od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))

    twins = twins_wp.numpy()
    assert twins.shape == (3 * len(mesh_tm.faces),)
    interior = np.flatnonzero(twins >= 0)
    # Involution: crossing an edge twice returns to the same halfedge.
    assert np.array_equal(twins[twins[interior]], interior)
    # A halfedge is never its own twin, and twins run over the same undirected edge.
    halfedge_endpoints_tm = tm.geometry.faces_to_edges(mesh_tm.faces)
    assert np.array_equal(
        np.sort(halfedge_endpoints_tm[interior], axis=1),
        np.sort(halfedge_endpoints_tm[twins[interior]], axis=1),
    )


@pytest.mark.parametrize("mesh_name", MESHES)
def test_halfedge_twins_matches_warp_tri_tri_adjacency(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the twins are ``warp.geometry.tri_tri_adjacency``'s neighbours after a named transform.

    Warp reports, per face and local vertex ``j``, the face across the edge *opposite* ``j``;
    halfedge ``3 * f + k`` runs from corner ``k`` to ``k + 1``, so it is the edge opposite corner
    ``(k + 2) % 3`` and its twin's face is ``twin // 3``. The two agree on every edge-manifold,
    consistently wound input, which every fixture here is. They part where ordito's precondition
    bites: Warp needs no consistent winding and pairs an edge whose two faces are wound against
    each other, which ``halfedge_twins`` rejects.
    """
    import warp.geometry

    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    # Warp annotates the faces ``wp.array2d``, a static helper no runtime array is typed as.
    faces_2d_wp = cast(
        "wp.array2d[wp.int32]",
        wp.array(np.asarray(mesh_tm.faces, dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
    )
    neighbors_warp, _edges_warp = warp.geometry.tri_tri_adjacency(
        faces_2d_wp, vertex_count=n_vertices
    )

    twins = od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=n_vertices).numpy()
    neighbors_od = np.empty((len(mesh_tm.faces), 3), dtype=np.int32)
    for k in range(3):
        twin = twins[k::3]
        neighbors_od[:, (k + 2) % 3] = np.where(twin >= 0, twin // 3, -1)

    assert np.array_equal(neighbors_od, neighbors_warp.numpy())


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_halfedge_twins_has_no_boundary_on_closed_mesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    twins_wp = od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))
    assert (twins_wp.numpy() >= 0).all()


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_halfedge_twins_boundary_matches_oriented_boundary_edges(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    twins = od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices)).numpy()

    halfedge_endpoints_tm = tm.geometry.faces_to_edges(mesh_tm.faces)
    boundary_from_twins = np.asarray(halfedge_endpoints_tm)[twins < 0]
    boundary_wp = od.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)

    assert len(boundary_from_twins) > 0
    assert {tuple(edge) for edge in boundary_from_twins} == {
        tuple(edge) for edge in boundary_wp.numpy()
    }


def test_halfedge_twins_rejects_non_manifold_edge(device: str) -> None:
    # Three triangles hinged on the edge (0, 1): "the" opposite halfedge is not defined.
    faces_wp = wp.array(
        np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=np.int32), dtype=wp.int32, device=device
    )
    with pytest.raises(ValueError, match="edge-manifold"):
        od.halfedge.halfedge_twins(faces_wp, n_vertices=5)


@pytest.mark.parametrize("consistent", [True, False])
def test_halfedge_twins_rejects_a_face_pair_wound_against_each_other(
    consistent: bool, device: str
) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is the algebra.

    Parametrized over both windings because the rejecting arm alone would pass against a function
    that rejects *everything*. Both arms are the same two triangles sharing edge ``{1, 2}``; only
    the second face's winding differs. Beside ``(0, 1, 2)``, which crosses that edge as ``1 -> 2``,
    the *consistent* neighbour is ``(2, 1, 3)`` -- it crosses back as ``2 -> 1`` -- while
    ``(1, 2, 3)`` is the inconsistent one, crossing ``1 -> 2`` a second time. The key
    ``halfedge_twins`` sorts on is the *undirected* edge, so nothing in the pairing itself
    distinguishes the two cases, which is exactly the defect this guards.
    """
    second = [2, 1, 3] if consistent else [1, 2, 3]
    faces_np = np.array([0, 1, 2, *second], dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    if not consistent:
        with pytest.raises(ValueError, match="consistently wound"):
            od.halfedge.halfedge_twins(faces_wp, n_vertices=4)
        return
    twins = od.halfedge.halfedge_twins(faces_wp, n_vertices=4).numpy()
    # Exactly the shared edge is paired, and it is the involution the docstring promises.
    assert np.count_nonzero(twins >= 0) == 2
    paired = np.flatnonzero(twins >= 0)
    assert np.array_equal(twins[twins[paired]], paired)


@pytest.mark.parametrize("mesh_name", ["boy_surface", "mobius"])
def test_halfedge_twins_rejects_a_non_orientable_surface(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is orientability.

    A non-orientable surface admits no consistent winding, so somewhere on it two faces must meet
    an edge the same way round and no halfedge twin table exists at all. Both fixtures are closed
    or bounded, edge-manifold and vertex-manifold, so *every other* precondition this function has
    is satisfied -- which is why the pairing used to succeed and hand back a table that satisfies
    ``twins[twins[h]] == h`` while breaking the "opposite directions" half of the same sentence.

    What that cost downstream, measured on ``boy_surface`` before the fix: 39 edges paired the
    wrong way, after which ``vertex_one_rings`` *succeeded* and returned 227 ring entries whose
    halfedge does not originate at the owning vertex, plus 74 halfedges appearing in two rings at
    once. On ``mobius`` the same corruption instead made the rotation close early, and the wrapper
    reported a "pinch point" on a mesh that has none.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    assert not mesh_tm.is_winding_consistent
    with pytest.raises(ValueError, match="consistently wound"):
        od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))
    # The same rejection reaches every consumer that derives the table for itself.
    with pytest.raises(ValueError, match="consistently wound"):
        od.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))


def test_halfedge_twins_rejects_a_duplicated_face(device: str) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is the algebra.

    Two arms of one distinction the winding guard has to draw. An *exactly* duplicated face makes
    all three of its edges carry two halfedges pointing the same way, which is the same defect a
    non-orientable surface produces and is rejected; a *reversed* duplicate is a consistently wound
    (if degenerate) closed surface, every edge is crossed once each way, and it must still pair.
    """
    with pytest.raises(ValueError, match="consistently wound"):
        od.halfedge.halfedge_twins(
            wp.array(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device),
            n_vertices=3,
        )
    reversed_wp = wp.array(
        np.array([0, 1, 2, 0, 2, 1], dtype=np.int32), dtype=wp.int32, device=device
    )
    assert (od.halfedge.halfedge_twins(reversed_wp, n_vertices=3).numpy() >= 0).all()


def test_vertex_one_rings_rejects_a_pinched_vertex(device: str) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is the algebra.

    Two closed degenerate sheets meeting at vertex 0 alone. Every edge is crossed once each way, so
    the twin table is sound and the winding guard has nothing to say; what fails is the *rotation*,
    which closes after one fan and leaves the other unvisited. Here so the winding guard added
    beside it cannot quietly take over this diagnosis -- the two failures are different, and the
    message the caller gets has to stay different too.
    """
    faces_wp = wp.array(
        np.array([0, 1, 2, 0, 2, 1, 0, 3, 4, 0, 4, 3], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    twins = od.halfedge.halfedge_twins(faces_wp, n_vertices=5)
    assert (twins.numpy() >= 0).all()
    with pytest.raises(ValueError, match="vertex-manifold"):
        od.halfedge.vertex_one_rings(faces_wp, twins=twins, n_vertices=5)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_validate_false_skips_only_the_check(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: ``validate=False`` changes whether the mesh is checked, not the answer.

    The validated path carries the oracle -- the involution test above and the trimesh ring tests
    below. On a valid mesh both builders must return byte-identical tables either way, and the
    unvalidated calls must take no host readback, since skipping one is their whole purpose.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp, n_vertices = mesh_wp.indices, len(mesh_tm.vertices)
    twins_wp = od.halfedge.halfedge_twins(faces_wp, n_vertices=n_vertices)
    rings_wp = od.halfedge.vertex_one_rings(faces_wp, n_vertices=n_vertices)
    readbacks: list[tuple[int, ...]] = []
    original = wp.array.numpy

    def counting_numpy(
        self: wp.array[object, Any], *, _suppress_bfloat16_warning: bool = False
    ) -> np.ndarray:
        readbacks.append(self.shape)
        return original(self, _suppress_bfloat16_warning=_suppress_bfloat16_warning)

    wp.array.numpy = counting_numpy
    try:
        unchecked_twins_wp = od.halfedge.halfedge_twins(
            faces_wp, n_vertices=n_vertices, validate=False
        )
        unchecked_rings_wp = od.halfedge.vertex_one_rings(
            faces_wp, n_vertices=n_vertices, validate=False
        )
    finally:
        wp.array.numpy = original
    assert readbacks == []
    assert np.array_equal(unchecked_twins_wp.numpy(), twins_wp.numpy())
    for unchecked_wp, checked_wp in zip(unchecked_rings_wp, rings_wp, strict=True):
        assert np.array_equal(unchecked_wp.numpy(), checked_wp.numpy())


def _sorted_and_bucketed_pairings(
    faces_wp: wp.array[wp.int32], n_vertices: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Twins and defect counts from the key sort (no vertex count) and the buckets (forced on)."""
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    pairings = []
    for bound in (None, n_vertices):
        defect_counts = wp.zeros(2, dtype=wp.int32, device=faces_wp.device)
        twins = od.halfedge._pair_halfedges(faces_wp, bound, defect_counts)  # pyright: ignore[reportPrivateUsage]
        pairings.append((twins.numpy(), defect_counts.numpy()))
    (sorted_twins, sorted_defects), (bucketed_twins, bucketed_defects) = pairings
    return sorted_twins, sorted_defects, bucketed_twins, bucketed_defects


@pytest.mark.parametrize("mesh_name", [*MESHES, "boy_surface", "mobius"])
def test_bucketed_twins_match_the_sorted_twins_on_fixtures(
    request: pytest.FixtureRequest, mesh_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the per-vertex bucket pairing reproduces the key-sort pairing.

    The sort carries the oracle (the tests above); CUDA takes the buckets whenever the vertex count
    is known and the CPU only when forced, so this pins the two to each other on both devices,
    defect counts included. ``boy_surface`` and ``mobius`` are the arms where the counts are not
    zero (edges wound the same way), so the comparison is not of two empty answers.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    sorted_twins, sorted_defects, bucketed_twins, bucketed_defects = _sorted_and_bucketed_pairings(
        mesh_wp.indices, len(mesh_tm.vertices), monkeypatch
    )
    assert np.array_equal(bucketed_twins, sorted_twins)
    assert np.array_equal(bucketed_defects, sorted_defects)
    if mesh_name in ("boy_surface", "mobius"):
        assert sorted_defects[1] > 0


@pytest.mark.parametrize(
    ("faces", "n_vertices", "expected_defects"),
    [
        pytest.param([0, 1, 2, 0, 1, 3, 0, 1, 4], 5, [1, 0], id="three-faces-on-an-edge"),
        pytest.param([0, 1, 2, 0, 1, 3, 1, 0, 4, 1, 0, 5], 6, [1, 0], id="four-faces-on-an-edge"),
        pytest.param([0, 1, 2, 1, 2, 3], 4, [0, 1], id="pair-wound-the-same-way"),
        pytest.param([0, 1, 2, 0, 1, 2], 3, [0, 3], id="duplicated-face"),
        pytest.param([0, 0, 1], 2, [0, 0], id="repeated-vertex"),
    ],
)
def test_bucketed_twins_match_the_sorted_twins_on_defects(
    faces: list[int],
    n_vertices: int,
    expected_defects: list[int],
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ordito against ordito: the buckets classify each defect as the sort's runs do, counted once.

    A run of three or more halfedges is one non-manifold edge, however many faces; a same-way pair
    is one misoriented edge, and a duplicated face is three. A face repeating a vertex has a
    self-edge that matches nothing, as in the sort.
    """
    faces_wp = wp.array(np.array(faces, dtype=np.int32), dtype=wp.int32, device=device)
    sorted_twins, sorted_defects, bucketed_twins, bucketed_defects = _sorted_and_bucketed_pairings(
        faces_wp, n_vertices, monkeypatch
    )
    assert np.array_equal(sorted_defects, expected_defects)
    assert np.array_equal(bucketed_defects, sorted_defects)
    assert np.array_equal(bucketed_twins, sorted_twins)


@pytest.mark.parametrize("hub_first", [True, False])
def test_bucketed_twins_on_a_hub(
    hub_first: bool, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: a high-valence vertex pairs as the sort pairs it, numbered either way.

    The bucket owner is the lower-*degree* endpoint, so the hub's spokes hold its edges whether the
    hub is numbered below every spoke (``hub_first``, the case a lower-*index* owner would pile
    into one bucket) or above.
    """
    cone_tm = tm.creation.cone(radius=1.0, height=1.0, sections=512)
    faces_np = np.asarray(cone_tm.faces, dtype=np.int32)
    valence = np.bincount(faces_np.ravel(), minlength=len(cone_tm.vertices))
    order = np.argsort(-valence if hub_first else valence, kind="stable")
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    faces_wp = wp.array(renumber[faces_np].ravel().astype(np.int32), dtype=wp.int32, device=device)
    sorted_twins, sorted_defects, bucketed_twins, bucketed_defects = _sorted_and_bucketed_pairings(
        faces_wp, len(cone_tm.vertices), monkeypatch
    )
    assert valence.max() == 512
    assert np.array_equal(bucketed_twins, sorted_twins)
    assert np.array_equal(bucketed_defects, sorted_defects)


def test_halfedge_twins_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert od.halfedge.halfedge_twins(faces_wp, n_vertices=0).shape == (0,)


# ---------------------------------------------------------------------------
# vertex_one_rings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_vertex_one_ring_sizes_match_incident_face_counts(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A after a named transform (Class B): ring sizes are the incident-face-corner counts.

    One outgoing halfedge per incident corner, so ``np.bincount`` over the flat face buffer is the
    reference -- exact, no tolerance. The second assert is what makes it a *partition*: every
    halfedge appears in exactly one ring, which a size check alone would not catch.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    ring_wp, offsets_wp, _ = od.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=n_vertices)

    # One outgoing halfedge per incident face-corner.
    incident_faces_tm = np.bincount(mesh_tm.faces.reshape(-1), minlength=n_vertices)
    assert np.array_equal(np.diff(offsets_wp.numpy()), incident_faces_tm)
    assert np.array_equal(np.sort(ring_wp.numpy()), np.arange(3 * len(mesh_tm.faces)))


@pytest.mark.parity(
    "vertex_one_rings",
    "trimesh",
    benchmarked=False,
    reason="Trimesh.vertex_neighbors does strictly less: it groups each vertex's "
    "neighbours and stops, where a ring is the rotationally *ordered* outgoing-halfedge "
    "fan plus the boundary flag. Timing them against each other would compare grouping "
    "with ordering -- measured 133 ms against 0.68 ms on sphere_med, most of which is "
    "that gap. The neighbour sets still agree, which is what this checks.",
)
@pytest.mark.parity(
    "vertex_one_rings_scale",
    "trimesh",
    benchmarked=False,
    reason="Trimesh.vertex_neighbors does strictly less: it groups each vertex's "
    "neighbours and stops, where a ring is the rotationally *ordered* outgoing-halfedge "
    "fan plus the boundary flag. Timing them against each other would compare grouping "
    "with ordering -- measured 133 ms against 0.68 ms on sphere_med, most of which is "
    "that gap. The neighbour sets still agree, which is what this checks.",
)
@pytest.mark.parametrize("mesh_name", MESHES)
def test_vertex_one_ring_neighbor_counts_match_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``Trimesh.vertex_neighbors`` counts, after adding one per boundary vertex.

    The transform is the definitional difference between the two structures, not a fudge: a ring
    holds one halfedge per incident *face*, which is one fewer than the neighbour count exactly when
    the fan is open. Folding ``is_boundary`` into the comparison means it also tests that flag,
    which is why it is asserted here as an addend rather than masked out.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    _, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )

    # A ring holds one halfedge per incident face, so it is one short of the neighbor count at a
    # boundary vertex (whose fan is open) and equal to it in the interior.
    neighbors_tm = np.array([len(neighbors) for neighbors in mesh_tm.vertex_neighbors])
    ring_sizes = np.diff(offsets_wp.numpy())
    assert np.array_equal(ring_sizes + is_boundary_wp.numpy().astype(np.int32), neighbors_tm)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_vertex_one_rings_are_rotationally_ordered(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: the *ordering* claim, which no reference exposes.

    trimesh's ``vertex_neighbors`` is a set, so it can check the ring's membership but never that
    consecutive entries rotate around the vertex. This asserts that directly -- successive faces
    share an edge *through this vertex* -- and then that one more rotation closes an interior fan
    and falls off an open one, which is the same predicate ``is_boundary`` reports.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )
    twins = od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=n_vertices).numpy()
    offsets, ring, is_boundary = offsets_wp.numpy(), ring_wp.numpy(), is_boundary_wp.numpy()

    for vertex in range(n_vertices):
        ring_halfedges = ring[offsets[vertex] : offsets[vertex + 1]]
        # Every entry leaves this vertex.
        assert np.array_equal(
            mesh_tm.faces.reshape(-1)[ring_halfedges], np.full(len(ring_halfedges), vertex)
        )
        # Consecutive entries lie in faces sharing an edge *at this vertex*, i.e. the walk rotates
        # around the vertex rather than wandering over the surface.
        for current, following in itertools.pairwise(ring_halfedges):
            shared = set(mesh_tm.faces[current // 3]) & set(mesh_tm.faces[following // 3])
            assert len(shared) == 2
            assert vertex in shared
        # One more rotation from the last entry closes an interior fan and falls off an open one.
        last = ring_halfedges[-1]
        previous = last + 2 if last % 3 == 0 else last - 1
        assert (twins[previous] == ring_halfedges[0]) != bool(is_boundary[vertex])


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_vertex_one_rings_boundary_flags_match_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: trimesh's multiplicity-1 edge grouping, reduced to a per-vertex boolean.

    ``group_rows(..., require_count=1)`` gives the boundary *edges*; the named transform is taking
    the unique vertices they touch. Only the open fixtures, because the flag is uniformly ``False``
    on a closed mesh and the comparison would hold for a constant.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    _, _, is_boundary_wp = od.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=n_vertices)

    boundary_vertices_tm = np.zeros(n_vertices, dtype=bool)
    boundary_vertices_tm[
        np.unique(mesh_tm.edges[tm.grouping.group_rows(mesh_tm.edges_sorted, require_count=1)])
    ] = True
    assert np.array_equal(is_boundary_wp.numpy(), boundary_vertices_tm)


def test_vertex_one_rings_isolated_vertex_is_empty(device: str) -> None:
    # Vertex 3 is unreferenced: it gets an empty ring rather than a bogus one.
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(faces_wp, n_vertices=4)

    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1, 2, 3, 3]))
    assert np.array_equal(np.sort(ring_wp.numpy()), np.array([0, 1, 2]))
    assert np.array_equal(is_boundary_wp.numpy(), np.array([True, True, True, False]))


def test_vertex_one_rings_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(faces_wp, n_vertices=0)
    assert offsets_wp.shape == (1,)
    assert ring_wp.shape == (0,)
    assert is_boundary_wp.shape == (0,)


def test_require_matching_twins_rejects_a_table_from_another_mesh(
    request: pytest.FixtureRequest,
) -> None:
    """
    Ordito against ordito: a ``twins=`` table built for a different mesh must be rejected.

    Not a library comparison: no reference library exposes a caller-supplied halfedge twin table.
    The table is indexed *by halfedge*, so one cached from a smaller mesh is short rather than
    merely stale, and the kernels that walk it (``ring_degrees_and_starts``, ``write_one_rings``)
    index past its end -- which on the CPU device reads the host heap silently rather than raising,
    the hazard CLAUDE.md section 12.1 records. Parametrized over all three public ``twins=`` entry
    points because the check is one shared validator and a site that skips it is invisible
    otherwise.

    The accepting arm is asserted too: the matching table must go through, or the guard would pass
    by rejecting everything.
    """
    _, coarse_wp = request.getfixturevalue("icosahedron")
    _, fine_wp = request.getfixturevalue("icosphere_coarse")
    coarse_twins_wp = od.halfedge.halfedge_twins(coarse_wp.indices)
    fine_twins_wp = od.halfedge.halfedge_twins(fine_wp.indices)
    assert coarse_twins_wp.size < fine_twins_wp.size

    contour_wp = od.selection.region_boundary_edges(
        fine_wp.indices,
        wp.array(np.arange(fine_wp.indices.size // 3) < 4, dtype=wp.bool, device=fine_wp.device),
        oriented=True,
    )
    calls = (
        lambda twins: od.halfedge.vertex_one_rings(fine_wp.indices, twins=twins),
        lambda twins: od.tangent_space.halfedge_transport_angles(
            fine_wp.points, fine_wp.indices, twins=twins
        ),
        lambda twins: od.selection.faces_left_of_contour(fine_wp.indices, contour_wp, twins=twins),
    )
    for call in calls:
        with pytest.raises(ValueError, match="one entry per halfedge"):
            call(coarse_twins_wp)
        call(fine_twins_wp)  # the matching table is accepted


@pytest.mark.parametrize(
    "corruption", ["swapped_pair", "out_of_range", "broken_involution", "all_boundary"]
)
def test_require_matching_twins_rejects_a_table_that_is_not_the_opposite_halfedge(
    corruption: str, request: pytest.FixtureRequest
) -> None:
    """
    Ordito against ordito: a right-length ``twins=`` table whose entries are wrong is rejected.

    Not a library comparison: no reference library exposes a caller-supplied halfedge twin table.
    The sibling above covers the *length* half of the contract; this covers the structural half,
    over the same three public ``twins=`` entry points, because only one of the three reaches the
    ring rotation and a check placed there would leave the other two unguarded.

    Three of the four corruptions are rejected by the validator and one deliberately is not.
    ``all_boundary`` -- every entry ``-1`` -- claims each halfedge has no twin, and refuting that
    means finding out whether another halfedge spans the same edge, which is the radix sort
    ``halfedge_twins`` runs and precisely the work ``twins=`` exists to skip. It is not silent: a
    fabricated boundary shortens the fan, so ``vertex_one_rings`` raises on the ring it cannot
    complete. This arm asserts *that* message rather than the validator's, so a future change that
    quietly drops the downstream backstop fails here.
    """
    _, mesh_wp = request.getfixturevalue("icosphere_coarse")
    faces_wp = mesh_wp.indices
    twins_np = od.halfedge.halfedge_twins(faces_wp).numpy()
    # Non-vacuity: an all-boundary table has to differ from the real one, i.e. the mesh is closed.
    assert (twins_np >= 0).all()

    corrupted_np = twins_np.copy()
    if corruption == "swapped_pair":
        corrupted_np[0], corrupted_np[3] = twins_np[3], twins_np[0]
    elif corruption == "out_of_range":
        corrupted_np[0] = 10**6
    elif corruption == "broken_involution":
        corrupted_np[0] = twins_np[1]
    else:
        corrupted_np[:] = -1
    corrupted_wp = wp.array(corrupted_np, dtype=wp.int32, device=faces_wp.device)

    if corruption == "all_boundary":
        with pytest.raises(ValueError, match="vertex-manifold"):
            od.halfedge.vertex_one_rings(faces_wp, twins=corrupted_wp)
        return

    contour_wp = od.selection.region_boundary_edges(
        faces_wp,
        wp.array(np.arange(faces_wp.size // 3) < 4, dtype=wp.bool, device=faces_wp.device),
        oriented=True,
    )
    calls = (
        lambda twins: od.halfedge.vertex_one_rings(faces_wp, twins=twins),
        lambda twins: od.tangent_space.halfedge_transport_angles(
            mesh_wp.points, faces_wp, twins=twins
        ),
        lambda twins: od.selection.faces_left_of_contour(faces_wp, contour_wp, twins=twins),
    )
    genuine_wp = wp.array(twins_np, dtype=wp.int32, device=faces_wp.device)
    for call in calls:
        with pytest.raises(ValueError, match="opposite halfedge"):
            call(corrupted_wp)
        call(genuine_wp)  # the genuine table is accepted


def test_require_matching_twins_bounds_a_twin_against_the_halfedge_count(device: str) -> None:
    """
    Ordito against ordito: a twin index is bounded by the halfedge count, not the face length.

    Not a library comparison: no reference library exposes a caller-supplied halfedge twin table.
    The two lengths differ whenever the face buffer is ragged -- the package defines the halfedge
    count as ``faces.shape[0] // 3 * 3``, so up to two trailing entries are not halfedges. A twin
    index landing in that gap is inside ``faces`` and outside the halfedges, so a range test
    against the wrong one of the two lets ``halfedge_destination`` read one past the end, which on
    the CPU device is a host-heap read rather than a fault (CLAUDE.md section 12.1).
    """
    faces_wp = wp.array(
        np.array([0, 1, 2, 0, 2, 3, 1, 2, 3, 7], dtype=np.int32), dtype=wp.int32, device=device
    )
    n_halfedges = faces_wp.size // 3 * 3
    # Non-vacuity: the trailing entry is what makes the two lengths disagree.
    assert n_halfedges < faces_wp.size

    twins_np = np.full(n_halfedges, -1, dtype=np.int32)
    twins_np[0] = n_halfedges  # in range for faces, out of range for halfedges
    with pytest.raises(ValueError, match="opposite halfedge"):
        od.halfedge.require_matching_twins(
            faces_wp, wp.array(twins_np, dtype=wp.int32, device=device)
        )


# ---------------------------------------------------------------------------
# halfedge_mates
# ---------------------------------------------------------------------------


def _expected_mates(faces_np: np.ndarray) -> np.ndarray:
    """``halfedge_mates``' codes from NumPy grouping of trimesh's per-halfedge edge rows."""
    edges = np.sort(tm.geometry.faces_to_edges(faces_np.reshape(-1, 3)), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    expected = np.full(len(edges), -1, dtype=np.int64)
    for edge in np.flatnonzero(counts >= 2):
        members = np.flatnonzero(inverse == edge)
        if len(members) == 2:
            expected[members] = members[::-1]
        else:
            expected[members] = -2 - members.min()
    return expected


@pytest.mark.parametrize("bounded", [True, False], ids=["buckets", "sort"])
@pytest.mark.parametrize("mesh_name", [*MESHES, "boy_surface", "mobius"])
def test_halfedge_mates_match_numpy_edge_grouping(
    request: pytest.FixtureRequest, mesh_name: str, bounded: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class B: the mates are NumPy's grouping of trimesh's halfedge rows by undirected edge.

    ``trimesh.geometry.faces_to_edges`` lists halfedge ``3 * f + k`` in ordito's order; grouping
    its sorted rows with ``np.unique`` gives each edge's halfedges, which name the codes (the
    partner of a pair, ``-1`` alone, ``-2 - lowest`` for three or more). Run through both
    builders on both devices: the per-vertex buckets (a vertex count given, size gate and CPU
    forced open) and the key sort (no vertex count). ``boy_surface`` and ``mobius`` carry pairs
    wound the same way, which a mate keeps and a twin does not.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_MATES_FROM_HALFEDGES", 0)
    n_vertices = len(mesh_tm.vertices) if bounded else None
    mates = od.halfedge.halfedge_mates(mesh_wp.indices, n_vertices).numpy()
    expected = _expected_mates(np.asarray(mesh_tm.faces))
    assert np.count_nonzero(expected >= 0) > 0
    assert np.array_equal(mates, expected)


@pytest.mark.parametrize("bounded", [True, False], ids=["buckets", "sort"])
@pytest.mark.parametrize(
    ("faces", "n_vertices"),
    [
        pytest.param([5, 6, 7, 0, 1, 2, 0, 1, 3, 0, 1, 4], 8, id="three-faces-on-an-edge"),
        pytest.param([6, 7, 8, 0, 1, 2, 0, 1, 3, 1, 0, 4, 1, 0, 5], 9, id="four-faces-on-an-edge"),
        pytest.param([0, 1, 2, 1, 2, 3], 4, id="pair-wound-the-same-way"),
        pytest.param([0, 1, 2, 0, 1, 2], 3, id="duplicated-face"),
        pytest.param([0, 0, 1], 2, id="repeated-vertex"),
    ],
)
def test_halfedge_mates_on_defects(
    faces: list[int], n_vertices: int, bounded: bool, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class B: on each defect the mates are NumPy's grouping, by both builders.

    The edges of three and four faces carry the ``-2 - lowest`` code on every member, which is
    what lets a consumer act on such an edge once (a leading unrelated face keeps ``lowest`` off
    halfedge 0, where a bare ``-2`` would read the same); the same-way pair and the duplicated face
    are pairs, where ``halfedge_twins`` would reject them.
    """
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_MATES_FROM_HALFEDGES", 0)
    faces_np = np.array(faces, dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    mates = od.halfedge.halfedge_mates(faces_wp, n_vertices if bounded else None).numpy()
    assert np.array_equal(mates, _expected_mates(faces_np))


@pytest.mark.parametrize("hub_first", [True, False])
def test_halfedge_mates_on_a_hub(
    hub_first: bool, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: a 512-valence hub pairs as the sort pairs it, numbered either way.

    The sort carries the oracle (``test_halfedge_mates_match_numpy_edge_grouping``); this pins the
    buckets to it where a lower-*index* owner would pile every spoke into one bucket.
    """
    cone_tm = tm.creation.cone(radius=1.0, height=1.0, sections=512)
    faces_np = np.asarray(cone_tm.faces, dtype=np.int32)
    valence = np.bincount(faces_np.ravel(), minlength=len(cone_tm.vertices))
    order = np.argsort(-valence if hub_first else valence, kind="stable")
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    faces_wp = wp.array(renumber[faces_np].ravel().astype(np.int32), dtype=wp.int32, device=device)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_MATES_FROM_HALFEDGES", 0)
    bucketed = od.halfedge.halfedge_mates(faces_wp, len(cone_tm.vertices)).numpy()
    sorted_ = od.halfedge.halfedge_mates(faces_wp).numpy()
    assert valence.max() == 512
    assert np.array_equal(bucketed, sorted_)


def test_halfedge_mates_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert od.halfedge.halfedge_mates(faces_wp, 0).shape == (0,)
