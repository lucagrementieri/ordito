"""Regression tests for ``ordito.halfedge`` against Trimesh (CPU reference)."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from tests.conftest import MESHES
from tests.conversions import warp_empty

# ---------------------------------------------------------------------------
# halfedge_twins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_halfedge_twins_matches_warp_tri_tri_adjacency(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the twins are ``warp.geometry.tri_tri_adjacency``'s neighbours after a named transform.

    Not a library comparison, for the pairing's own algebra: it is an involution, no halfedge is its
    own twin, and twins run over the same undirected edge (``trimesh.geometry.faces_to_edges`` is
    there as a *definition* of the edge a halfedge spans, not as a second answer).

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

    neighbors_od = np.empty((len(mesh_tm.faces), 3), dtype=np.int32)
    for k in range(3):
        twin = twins[k::3]
        neighbors_od[:, (k + 2) % 3] = np.where(twin >= 0, twin // 3, -1)

    assert np.array_equal(neighbors_od, neighbors_warp.numpy())


def _faces(values: list[int], device: str) -> wp.array[wp.int32]:
    return wp.array(np.array(values, dtype=np.int32), dtype=wp.int32, device=device)


def _ragged_twins_call(device: str) -> None:
    faces_wp = _faces([0, 1, 2, 0, 2, 3, 1, 2, 3, 7], device)
    n_halfedges = faces_wp.size // 3 * 3
    # Non-vacuity: the trailing entry is what makes the two lengths disagree.
    assert n_halfedges < faces_wp.size
    twins_np = np.full(n_halfedges, -1, dtype=np.int32)
    twins_np[0] = n_halfedges  # in range for faces, out of range for halfedges
    od.halfedge.require_matching_twins(faces_wp, wp.array(twins_np, dtype=wp.int32, device=device))


def _pinched_rings_call(device: str) -> None:
    faces_wp = _faces([0, 1, 2, 0, 2, 1, 0, 3, 4, 0, 4, 3], device)
    twins = od.halfedge.halfedge_twins(faces_wp, n_vertices=5)
    # The twin table is sound: the winding guard has nothing to say, the rotation fails.
    assert (twins.numpy() >= 0).all()
    od.halfedge.vertex_one_rings(faces_wp, twins=twins, n_vertices=5)


_INVALID_HALFEDGE_CASES: list[tuple[str, Callable[[str], object], str]] = [
    (
        "non-manifold-edge",
        lambda device: od.halfedge.halfedge_twins(
            _faces([0, 1, 2, 0, 1, 3, 0, 1, 4], device), n_vertices=5
        ),
        "edge-manifold",
    ),
    (
        "duplicated-face",
        lambda device: od.halfedge.halfedge_twins(_faces([0, 1, 2, 0, 1, 2], device), n_vertices=3),
        "consistently wound",
    ),
    ("pinched-vertex", _pinched_rings_call, "vertex-manifold"),
    ("twin-past-the-halfedge-count", _ragged_twins_call, "opposite halfedge"),
]


@pytest.mark.parametrize(
    ("call", "match"),
    [pytest.param(call, match, id=name) for name, call, match in _INVALID_HALFEDGE_CASES],
)
def test_halfedge_rejects_invalid_small_inputs(
    call: Callable[[str], object], match: str, device: str
) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is the algebra.

    Three faces hinged on one edge have no "the" opposite halfedge. An *exactly* duplicated face
    carries all three edges twice the same way, the defect a non-orientable surface produces; a
    *reversed* duplicate is consistently wound (if degenerate) and must still pair, asserted below.
    Two closed sheets meeting at vertex 0 alone pair soundly but their rotation closes after one
    fan, and that diagnosis must stay distinct from the winding guard's. A twin index is bounded
    by the halfedge count (``faces.shape[0] // 3 * 3``), not the face length: on a ragged buffer a
    range test against the wrong one lets ``halfedge_destination`` read past the end, a host-heap
    read on CPU.
    """
    with pytest.raises(ValueError, match=match):
        call(device)
    assert (
        od.halfedge.halfedge_twins(_faces([0, 1, 2, 0, 2, 1], device), n_vertices=3).numpy() >= 0
    ).all()


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


def test_validate_false_skips_only_the_check(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: ``validate=False`` changes whether the mesh is checked, not the answer.

    The validated path carries the oracle -- the involution test above and the trimesh ring tests
    below. On a valid mesh both builders must return byte-identical tables either way, and the
    unvalidated calls must take no host readback, since skipping one is their whole purpose.
    """
    mesh_tm, mesh_wp = sphere_irregular
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

    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", counting_numpy)
        unchecked_twins_wp = od.halfedge.halfedge_twins(
            faces_wp, n_vertices=n_vertices, validate=False
        )
        unchecked_rings_wp = od.halfedge.vertex_one_rings(
            faces_wp, n_vertices=n_vertices, validate=False
        )
    assert readbacks == []
    assert np.array_equal(unchecked_twins_wp.numpy(), twins_wp.numpy())
    for unchecked_wp, checked_wp in zip(unchecked_rings_wp, rings_wp, strict=True):
        assert np.array_equal(unchecked_wp.numpy(), checked_wp.numpy())


def _sorted_and_bucketed_pairings(
    faces_wp: wp.array[wp.int32], n_vertices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Twins and defect counts from the key sort (no vertex count) and the buckets.

    The buckets are reached on the CPU device only under ``force_halfedge_buckets``.
    """
    pairings = []
    for bound in (None, n_vertices):
        defect_counts = wp.zeros(2, dtype=wp.int32, device=faces_wp.device)
        twins = od.halfedge._pair_halfedges(faces_wp, bound, defect_counts)  # pyright: ignore[reportPrivateUsage]
        pairings.append((twins.numpy(), defect_counts.numpy()))
    (sorted_twins, sorted_defects), (bucketed_twins, bucketed_defects) = pairings
    return sorted_twins, sorted_defects, bucketed_twins, bucketed_defects


@pytest.mark.usefixtures("force_halfedge_buckets")
@pytest.mark.parametrize("mesh_name", [*MESHES, "mobius"])
def test_bucketed_twins_match_the_sorted_twins_on_fixtures(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the per-vertex bucket pairing reproduces the key-sort pairing.

    The sort carries the oracle (the tests above); CUDA takes the buckets whenever the vertex count
    is known and the CPU only when forced, so this pins the two to each other on both devices,
    defect counts included. ``mobius`` is the arm where the counts are not zero (edges wound the
    same way), so the comparison is not of two empty answers.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    sorted_twins, sorted_defects, bucketed_twins, bucketed_defects = _sorted_and_bucketed_pairings(
        mesh_wp.indices, len(mesh_tm.vertices)
    )
    assert np.array_equal(bucketed_twins, sorted_twins)
    assert np.array_equal(bucketed_defects, sorted_defects)
    # A non-orientable surface pairs some halfedges running the same way: a direction defect.
    if not mesh_tm.is_winding_consistent:
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
@pytest.mark.usefixtures("force_halfedge_buckets")
def test_bucketed_twins_match_the_sorted_twins_on_defects(
    faces: list[int], n_vertices: int, expected_defects: list[int], device: str
) -> None:
    """
    Ordito against ordito: the buckets classify each defect as the sort's runs do, counted once.

    A run of three or more halfedges is one non-manifold edge, however many faces; a same-way pair
    is one misoriented edge, and a duplicated face is three. A face repeating a vertex has a
    self-edge that matches nothing, as in the sort.
    """
    faces_wp = wp.array(np.array(faces, dtype=np.int32), dtype=wp.int32, device=device)
    sorted_twins, sorted_defects, bucketed_twins, bucketed_defects = _sorted_and_bucketed_pairings(
        faces_wp, n_vertices
    )
    assert np.array_equal(sorted_defects, expected_defects)
    assert np.array_equal(bucketed_defects, sorted_defects)
    assert np.array_equal(bucketed_twins, sorted_twins)


@pytest.mark.usefixtures("force_halfedge_buckets")
@pytest.mark.parametrize("hub_first", [True, False], ids=["hub-first", "hub-last"])
def test_bucketed_pairings_on_a_hub(hub_first: bool, device: str) -> None:
    """
    Ordito against ordito: a 512-valence hub pairs as the sort pairs it, numbered either way.

    The sort carries the oracle (the twin tests above, ``test_halfedge_mates_match_numpy_edge_
    grouping`` for the mates); this pins the buckets to it, twins with their defect counts and
    mates both. The bucket owner is the lower-*degree* endpoint, so the hub's spokes hold its edges
    whether the hub is numbered below every spoke (``hub_first``, the case a lower-*index* owner
    would pile into one bucket) or above.
    """
    cone_tm = tm.creation.cone(radius=1.0, height=1.0, sections=512)
    faces_np = np.asarray(cone_tm.faces, dtype=np.int32)
    n_vertices = len(cone_tm.vertices)
    valence = np.bincount(faces_np.ravel(), minlength=n_vertices)
    order = np.argsort(-valence if hub_first else valence, kind="stable")
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    faces_wp = wp.array(renumber[faces_np].ravel().astype(np.int32), dtype=wp.int32, device=device)
    sorted_twins, sorted_defects, bucketed_twins, bucketed_defects = _sorted_and_bucketed_pairings(
        faces_wp, n_vertices
    )
    assert valence.max() == 512
    assert np.array_equal(bucketed_twins, sorted_twins)
    assert np.array_equal(bucketed_defects, sorted_defects)
    bucketed_mates = od.halfedge.halfedge_mates(faces_wp, n_vertices).numpy()
    sorted_mates = od.halfedge.halfedge_mates(faces_wp).numpy()
    assert np.array_equal(bucketed_mates, sorted_mates)


def test_halfedge_builders_empty(device: str) -> None:
    """Twins, one-rings and mates of an empty face buffer: empty, rings with their one offset."""
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert od.halfedge.halfedge_twins(faces_wp, n_vertices=0).shape == (0,)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(faces_wp, n_vertices=0)
    assert offsets_wp.shape == (1,)
    assert ring_wp.shape == (0,)
    assert is_boundary_wp.shape == (0,)
    assert od.halfedge.halfedge_mates(faces_wp, 0).shape == (0,)


# ---------------------------------------------------------------------------
# vertex_one_rings
# ---------------------------------------------------------------------------


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
    which is why it is asserted here as an addend rather than masked out; with the ring sizes pinned
    by the first assert, the flag is the neighbour count minus the incident-face count exactly.

    Class B too, the ring sizes: one outgoing halfedge per incident corner, so ``np.bincount`` over
    the flat face buffer is the reference -- exact, no tolerance. The sort assert is what makes the
    rings a *partition*: every halfedge appears in exactly one ring, which a size check alone would
    not catch.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )

    # One outgoing halfedge per incident face-corner.
    incident_faces_tm = np.bincount(mesh_tm.faces.reshape(-1), minlength=n_vertices)
    assert np.array_equal(np.diff(offsets_wp.numpy()), incident_faces_tm)
    assert np.array_equal(np.sort(ring_wp.numpy()), np.arange(3 * len(mesh_tm.faces)))

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


def test_vertex_one_rings_return_twins_matches_the_two_calls(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: ``return_twins=True`` hands back the table ``halfedge_twins`` builds.

    Not a library comparison: the claim is that the one-readback form returns exactly what the two
    separate calls do, rings included, with no reference involved. A supplied table comes back as
    the same array object, since nothing was built.
    """
    mesh_tm, mesh_wp = sphere_irregular
    n_vertices = len(mesh_tm.vertices)
    twins_wp = od.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=n_vertices)
    rings_wp = od.halfedge.vertex_one_rings(mesh_wp.indices, twins=twins_wp, n_vertices=n_vertices)
    *joint_rings_wp, joint_twins_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices, return_twins=True
    )
    assert np.array_equal(joint_twins_wp.numpy(), twins_wp.numpy())
    for joint_wp, separate_wp in zip(joint_rings_wp, rings_wp, strict=True):
        assert np.array_equal(joint_wp.numpy(), separate_wp.numpy())
    *_, passed_through_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, twins=twins_wp, n_vertices=n_vertices, return_twins=True
    )
    assert passed_through_wp is twins_wp


@pytest.mark.parametrize("mesh_name", MESHES)
def test_vertex_one_rings_bucketed_walk_matches_the_sorted_walk(
    request: pytest.FixtureRequest, mesh_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the bucketed pairing's ring walk equals the key-sorted pairing's.

    Not a library comparison: the ring order is ordito's own convention, and its oracle is
    ``test_vertex_one_ring_neighbor_counts_match_trimesh`` on the sorted path. The bucketed path
    (CUDA by default, opened on the CPU here) hands the walk its degrees and starts from the
    pairing instead of counting them over the twins, so every ring, offset, boundary flag and
    twin must come out identical, on the closed and open fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    sorted_rings = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices, return_twins=True
    )
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    bucketed_rings = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices, return_twins=True
    )
    assert sorted_rings[0].size == mesh_wp.indices.size
    for bucketed_wp, sorted_wp in zip(bucketed_rings, sorted_rings, strict=True):
        assert np.array_equal(bucketed_wp.numpy(), sorted_wp.numpy())


def test_vertex_one_rings_isolated_vertex_is_empty(device: str) -> None:
    # Vertex 3 is unreferenced: it gets an empty ring rather than a bogus one.
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(faces_wp, n_vertices=4)

    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1, 2, 3, 3]))
    assert np.array_equal(np.sort(ring_wp.numpy()), np.array([0, 1, 2]))
    assert np.array_equal(is_boundary_wp.numpy(), np.array([True, True, True, False]))


@pytest.mark.parametrize(
    "corruption",
    ["another_mesh", "swapped_pair", "out_of_range", "broken_involution", "all_boundary"],
)
def test_require_matching_twins_rejects_a_table_that_is_not_the_opposite_halfedge(
    corruption: str, request: pytest.FixtureRequest
) -> None:
    """
    Ordito against ordito: a ``twins=`` table of the wrong length or with wrong entries is rejected.

    Not a library comparison: no reference library exposes a caller-supplied halfedge twin table.
    ``another_mesh`` is the *length* half of the contract: the table is indexed by halfedge, so one
    cached from a smaller mesh is short rather than merely stale, and the kernels that walk it
    (``ring_degrees_and_starts``, ``write_one_rings``) would index past its end -- on the CPU
    device a silent host-heap read. The rest is the structural half. Every arm goes through all
    three public ``twins=`` entry points, because the check is one shared validator and a site that
    skips it is invisible otherwise. The accepting call is asserted too, or the guard would pass by
    rejecting everything.

    Of the four structural corruptions, three are rejected by the validator and one deliberately
    is not. ``all_boundary`` -- every entry ``-1`` -- claims each halfedge has no twin, and
    refuting that means finding out whether another halfedge spans the same edge, which is the
    radix sort ``halfedge_twins`` runs and precisely the work ``twins=`` exists to skip. It is not
    silent: a fabricated boundary shortens the fan, so ``vertex_one_rings`` raises on the ring it
    cannot complete. This arm asserts *that* message rather than the validator's, so a future
    change that quietly drops the downstream backstop fails here.
    """
    _, mesh_wp = request.getfixturevalue("sphere_irregular")
    faces_wp = mesh_wp.indices
    twins_np = od.halfedge.halfedge_twins(faces_wp).numpy()
    # Non-vacuity: an all-boundary table has to differ from the real one, i.e. the mesh is closed.
    assert (twins_np >= 0).all()

    corrupted_np = twins_np.copy()
    match = "opposite halfedge"
    if corruption == "another_mesh":
        _, coarse_wp = request.getfixturevalue("sphere_well_shaped")
        corrupted_np = od.halfedge.halfedge_twins(coarse_wp.indices).numpy()
        assert corrupted_np.size < twins_np.size
        match = "one entry per halfedge"
    elif corruption == "swapped_pair":
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
        with pytest.raises(ValueError, match=match):
            call(corrupted_wp)
        call(genuine_wp)  # the genuine table is accepted


def test_require_matching_twins_called_directly(device: str) -> None:
    """
    Not a library comparison: the validator's own entry point, on its trivially valid inputs.

    No reference exposes a caller-supplied twin table. ``None`` (nothing supplied) and an empty
    mesh with an empty table are accepted without a launch; the same call rejects a one-triangle
    table claiming a twin, so the accepting arms are not vacuous.
    """
    faces_wp = _faces([0, 1, 2], device)
    od.halfedge.require_matching_twins(faces_wp, None)
    od.halfedge.require_matching_twins(
        warp_empty(0, wp.int32, device), warp_empty(0, wp.int32, device)
    )
    od.halfedge.require_matching_twins(faces_wp, wp.full(3, -1, dtype=wp.int32, device=device))
    with pytest.raises(ValueError, match="opposite halfedge"):
        od.halfedge.require_matching_twins(
            faces_wp, wp.array([1, 0, -1], dtype=wp.int32, device=device)
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


@pytest.mark.usefixtures("force_halfedge_buckets")
@pytest.mark.parametrize("bounded", [True, False], ids=["buckets", "sort"])
@pytest.mark.parametrize("mesh_name", [*MESHES, "mobius"])
def test_halfedge_mates_match_numpy_edge_grouping(
    request: pytest.FixtureRequest, mesh_name: str, bounded: bool
) -> None:
    """
    Class B: the mates are NumPy's grouping of trimesh's halfedge rows by undirected edge.

    ``trimesh.geometry.faces_to_edges`` lists halfedge ``3 * f + k`` in ordito's order; grouping
    its sorted rows with ``np.unique`` gives each edge's halfedges, which name the codes (the
    partner of a pair, ``-1`` alone, ``-2 - lowest`` for three or more). Run through both
    builders on both devices: the per-vertex buckets (a vertex count given, size gate and CPU
    forced open) and the key sort (no vertex count). ``mobius`` carries pairs wound the same
    way, which a mate keeps and a twin does not.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
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
@pytest.mark.usefixtures("force_halfedge_buckets")
def test_halfedge_mates_on_defects(
    faces: list[int], n_vertices: int, bounded: bool, device: str
) -> None:
    """
    Class B: on each defect the mates are NumPy's grouping, by both builders.

    The edges of three and four faces carry the ``-2 - lowest`` code on every member, which is
    what lets a consumer act on such an edge once (a leading unrelated face keeps ``lowest`` off
    halfedge 0, where a bare ``-2`` would read the same); the same-way pair and the duplicated face
    are pairs, where ``halfedge_twins`` would reject them.
    """
    faces_np = np.array(faces, dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    mates = od.halfedge.halfedge_mates(faces_wp, n_vertices if bounded else None).numpy()
    assert np.array_equal(mates, _expected_mates(faces_np))


def test_halfedge_mates_key_order_is_the_edge_key_sort(
    request: pytest.FixtureRequest, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the key order is ordito's own sort, recomputed by NumPy.

    On the sort path it is the stable argsort of each halfedge's ``min + max * n_vertices`` key
    (``adjacency.sorted_face_edge_keys``' order); on the bucket path there is none. The mates are
    the ones the plain call returns either way.
    """
    mesh_tm, mesh_wp = sphere_irregular
    faces_np = np.asarray(mesh_tm.faces)
    n_vertices = len(mesh_tm.vertices)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)
    keys_np = edges_np[:, 0] + edges_np[:, 1] * n_vertices
    mates, key_order = od.halfedge.halfedge_mates(
        mesh_wp.indices, n_vertices, return_key_order=True
    )
    assert key_order is not None
    assert np.array_equal(key_order.numpy(), np.argsort(keys_np, kind="stable"))
    assert np.array_equal(mates.numpy(), _expected_mates(faces_np))
    request.getfixturevalue("force_halfedge_buckets")
    mates, key_order = od.halfedge.halfedge_mates(
        mesh_wp.indices, n_vertices, return_key_order=True
    )
    assert key_order is None
    assert np.array_equal(mates.numpy(), _expected_mates(faces_np))


@pytest.mark.parametrize("n_classes", [1, 2])
@pytest.mark.parametrize("given_order", [True, False], ids=["key-order", "sorted"])
def test_key_ordered_halfedges_match_numpy(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], n_classes: int, given_order: bool
) -> None:
    """
    Not a library comparison: the order is ordito's edge-key convention, recomputed by NumPy.

    Random flags over every halfedge, so both halfedges of an edge are often flagged together and
    the key ties have to break by halfedge index. Each class comes out in stable ascending key
    order, class 0 first -- from the sort's permutation when it is given (nothing sorted), and by
    sorting only the flagged halfedges when not.
    """
    mesh_tm, mesh_wp = sphere_irregular
    device = mesh_wp.indices.device
    faces_np = np.asarray(mesh_tm.faces)
    n_vertices = len(mesh_tm.vertices)
    n = faces_np.size
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)
    keys_np = edges_np[:, 0] + edges_np[:, 1] * n_vertices
    flags_np = (np.random.default_rng(7).random((n_classes, n)) < 0.4).astype(np.int32)
    order_np = np.argsort(keys_np, kind="stable")
    expected = np.concatenate([order_np[flags_np[c][order_np] == 1] for c in range(n_classes)])
    _, key_order = od.halfedge.halfedge_mates(mesh_wp.indices, n_vertices, return_key_order=True)
    assert key_order is not None
    table_np = flags_np[:, key_order.numpy()] if given_order else flags_np
    inclusive = wp.array(
        np.cumsum(table_np.ravel()).astype(np.int32), dtype=wp.int32, device=device
    )
    halfedges = od.halfedge.key_ordered_halfedges(
        mesh_wp.indices,
        inclusive,
        int(flags_np.sum()),
        key_order=key_order if given_order else None,
        n_vertices=n_vertices,
    ).numpy()
    assert np.any(np.diff(keys_np[expected]) == 0)
    assert np.array_equal(halfedges, expected)


def test_key_ordered_halfedges_validates(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = sphere_irregular
    device = mesh_wp.indices.device
    n = mesh_wp.indices.size
    with pytest.raises(ValueError, match="whole number"):
        od.halfedge.key_ordered_halfedges(
            mesh_wp.indices, wp.zeros(n + 1, dtype=wp.int32, device=device), 0
        )
    with pytest.raises(ValueError, match="n_vertices is required"):
        od.halfedge.key_ordered_halfedges(
            mesh_wp.indices, wp.zeros(2 * n, dtype=wp.int32, device=device), 0
        )


def test_key_ordered_halfedges_single_class_and_empty_selection(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the key order is ordito's convention, recomputed by NumPy.

    A single class needs no ``n_vertices``: the sort then packs at the pair radix, which must give
    the same stable ascending key order as the vertex-count radix. An empty selection returns an
    empty order with no sort.
    """
    mesh_tm, mesh_wp = sphere_irregular
    device = mesh_wp.indices.device
    faces_np = np.asarray(mesh_tm.faces)
    n_vertices = len(mesh_tm.vertices)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)
    keys_np = edges_np[:, 0] + edges_np[:, 1] * n_vertices
    flags_np = (np.random.default_rng(3).random(faces_np.size) < 0.4).astype(np.int32)
    order_np = np.argsort(keys_np, kind="stable")
    expected = order_np[flags_np[order_np] == 1]
    inclusive = wp.array(np.cumsum(flags_np).astype(np.int32), dtype=wp.int32, device=device)
    halfedges = od.halfedge.key_ordered_halfedges(
        mesh_wp.indices, inclusive, int(flags_np.sum())
    ).numpy()
    assert expected.size > 0
    assert np.array_equal(halfedges, expected)

    none_flagged = wp.zeros(faces_np.size, dtype=wp.int32, device=device)
    assert od.halfedge.key_ordered_halfedges(mesh_wp.indices, none_flagged, 0).shape == (0,)
    _, key_order = od.halfedge.halfedge_mates(mesh_wp.indices, n_vertices, return_key_order=True)
    assert key_order is not None
    from_order = od.halfedge.key_ordered_halfedges(
        mesh_wp.indices, none_flagged, 0, key_order=key_order
    )
    assert from_order.shape == (0,)
