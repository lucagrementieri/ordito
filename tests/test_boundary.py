"""Regression tests for ``ordito.boundary`` against Trimesh (CPU reference)."""

from __future__ import annotations

from collections import Counter

import igl
import numpy as np
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito as od
from tests.comparisons import (
    assert_same_loop_set,
    boundary_loop_sizes,
    lexsort_rows,
    trimesh_outline_loops,
)
from tests.conftest import OPEN_MESHES
from tests.conversions import (
    numpy_to_warp,
    points_to_warp,
    pyvista_edges_to_indices,
    trimesh_to_meshlib,
    trimesh_to_pymeshfix,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    warp_empty,
)


# Open-surface fixtures that actually have a boundary (watertight solids do not).
@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_edges", "trimesh", "pyvista", "igl")
def test_boundary_edges_match_trimesh_pyvista_and_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B against all three references: the same edge set after a canonical row order.

    Neither side defines the order in which boundary edges come back, so every comparison is a
    lexsort of the rows.

    - **trimesh**: its multiplicity-1 rows of ``edges_sorted``. The rows are min-first on both
      sides, which is what makes the lexsort sufficient.
    - **pyvista**: ``extract_feature_edges(boundary_edges=True)`` with the other three classes off,
      after the index remap of VTK's renumbered output (as in ``tests/test_seams.py``). The flags
      matter more here than anywhere else in the suite, because VTK's default turns on the
      *feature* edges too and the count would then include every crease. **Do not map
      ``PolyData.n_open_edges`` to this quantity**: it is ``vtkFeatureEdges`` with boundary **and
      non-manifold** edges on, so on three faces sharing one edge it reads 7 where ordito counts 6
      boundary edges. Only ``is_manifold`` (``n_open_edges == 0``) maps cleanly, and that is
      ``tests/test_validation.py``'s row.
    - **igl**: ``igl.boundary_facets`` returns the same edge list plus the incident face of each
      edge and that edge's corner index -- strictly more than ordito's two columns, and the
      benchmark reads its row that way. Only the first return is compared. igl's edges come out
      **oriented** (they carry the incident face's winding), so they are sorted within each row
      for the undirected set and compared unsorted against
      [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges], which agrees with igl's
      orientation vertex for vertex. That directed compare, where only the row order is
      canonicalized, is what catches a reversed half-edge.

    Both fixtures are open, so the references are non-empty by construction -- asserted anyway,
    since running this on a closed mesh would compare two empty sets and pass.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    edges_pv = pyvista_edges_to_indices(
        trimesh_to_pyvista(mesh_tm).extract_feature_edges(
            boundary_edges=True, feature_edges=False, non_manifold_edges=False, manifold_edges=False
        ),
        mesh_tm.vertices,
    )
    edges_igl, _face_igl, _corner_igl = igl.boundary_facets(mesh_tm.faces.astype(np.int64))
    assert len(edges_pv) > 0

    boundary_edges_wp = od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    oriented_wp = od.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)

    canonical_wp = lexsort_rows(boundary_edges_wp.numpy())
    assert np.array_equal(canonical_wp, lexsort_rows(boundary_edges_tm))
    assert np.array_equal(canonical_wp, lexsort_rows(edges_pv))
    assert np.array_equal(canonical_wp, lexsort_rows(np.sort(edges_igl, axis=1)))
    assert np.array_equal(lexsort_rows(oriented_wp.numpy()), lexsort_rows(np.asarray(edges_igl)))


@pytest.mark.parametrize("mesh_name", [*OPEN_MESHES, "mobius"])
def test_boundary_queries_bucketed_match_the_key_sort(
    request: pytest.FixtureRequest, mesh_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the bucketed mates give the key sort's boundary, row for row.

    The sort path carries the oracles (``test_boundary_edges_match_trimesh_pyvista_and_igl``,
    ``test_boundary_loops`` and the
    rest); on the bucket path only the boundary halfedges are sorted, so the edge rows are pinned
    to strictly ascending ``(max, min)`` keys and every query to the sort path's answer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices, faces = mesh_wp.points, mesh_wp.indices

    def queries() -> list[np.ndarray]:
        values, offsets = od.boundary.boundary_loops_with_offsets(vertices, faces)
        return [
            od.boundary.boundary_edges(vertices, faces).numpy(),
            od.boundary.oriented_boundary_edges(vertices, faces).numpy(),
            values.numpy(),
            offsets.numpy(),
            od.boundary.boundary_vertex_indices(vertices, faces).numpy(),
        ]

    sorted_ = queries()
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_MATES_FROM_HALFEDGES", 0)
    bucketed = queries()
    assert len(mesh_tm.outline().entities) > 0
    _assert_edge_key_order(bucketed[0])
    _assert_edge_key_order(bucketed[1])
    for sorted_np, bucketed_np in zip(sorted_, bucketed, strict=True):
        assert np.array_equal(sorted_np, bucketed_np)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_edges", "pymeshlab", "meshlib")
def test_boundary_vertex_indices(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: two references mark the boundary *vertices* where ordito returns the edge pairs.

    MeshLab's ``compute_selection_from_mesh_border`` and MeshLib's ``getBoundaryVerts`` both do the
    same find-the-boundary pass and stop one step earlier, giving a per-vertex bool where ordito
    gives edges. Two named transforms make them comparable: each reference is read as a mask (off
    ``vertex_selection_array()``, which the MeshLab filter returns nothing from, and off
    ``mn.getNumpyBitSet``, which is already domain-sized), and ordito's edge pairs are projected
    down with ``np.unique`` -- which is exactly what
    [`boundary_vertex_indices`][ordito.boundary.boundary_vertex_indices] computes, so the
    projection is a function under test rather than test-side glue.

    MeshLib is *not* also the oracle for the edges themselves. Its
    ``findRegionBoundaryUndirectedEdgesInsideMesh`` looks like the counterpart and is not: the
    "InsideMesh" is load-bearing, and handed an all-``True`` region it returns **zero** edges
    because it excludes the mesh's own boundary by construction. It is the oracle for
    [`region_boundary_edges`][ordito.selection.region_boundary_edges] instead, where
    tests/test_selection.py pins it.

    [`boundary_vertices`][ordito.boundary.boundary_vertices] is checked on the same fixtures as the
    *positions* gathered on the trimesh side, ``vertices[unique(edges)]``, which also fixes the
    order since ordito returns ascending indices too.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    vertex_indices_tm = np.unique(boundary_edges_tm)
    vertex_indices_wp = od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_from_mesh_border()
    selection_pml = np.asarray(meshset_pml.current_mesh().vertex_selection_array())

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    selection_ml = mn.getNumpyBitSet(mm.getBoundaryVerts(mesh_ml.topology))

    assert vertex_indices_wp.size > 0  # non-vacuity: an empty rim would pass everything below
    assert np.array_equal(vertex_indices_wp.numpy(), vertex_indices_tm)
    assert np.array_equal(np.flatnonzero(selection_pml), vertex_indices_wp.numpy())
    assert np.array_equal(np.flatnonzero(selection_ml), vertex_indices_wp.numpy())
    # And the edges themselves project onto the same vertex set.
    edges_wp = od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(np.unique(edges_wp.numpy()), np.flatnonzero(selection_pml))
    vertices_wp = od.boundary.boundary_vertices(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        vertices_wp.numpy(), mesh_tm.vertices[vertex_indices_tm], rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_loops", "igl")
def test_boundary_loops(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A, and unusually strong for a loop comparison: same count, order and start vertex.

    ``igl.boundary_loop_all`` happens to agree with ordito on all three -- loops ranked by
    length, each starting at its lowest vertex index and walked the same way round -- so no
    canonicalization is needed at all. [`test_boundary_loops_matches_trimesh_outline`] is the
    class-B version of the same claim, against a reference that fixes none of those
    conventions.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loops_igl = igl.boundary_loop_all(mesh_tm.faces.astype(np.int64))
    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)

    assert len(loops_wp) == len(loops_igl)
    for loop_wp, loop_igl in zip(loops_wp, loops_igl, strict=True):
        assert np.array_equal(loop_wp.numpy(), np.asarray(loop_igl))


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_loops", "trimesh")
def test_boundary_loops_matches_trimesh_outline(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``Trimesh.outline()`` returns the same loops as ``Path3D`` entities.

    Three named transforms, all conventions rather than results, and all three live in
    [`tests.comparisons.trimesh_outline_loops`][] and
    [`tests.comparisons.assert_same_loop_set`][] because
    ``tests/test_mesh.py`` needs the identical pair for the ``Trimesh`` container property: the
    entities index the mesh's own vertex array, a closed entity repeats its first point as its last,
    and neither the order between loops nor the starting point within one is defined by either
    library.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loops_tm = trimesh_outline_loops(mesh_tm)
    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)

    assert len(loops_tm) > 0  # non-vacuous: these fixtures have rims
    assert_same_loop_set([loop.numpy() for loop in loops_wp], loops_tm)


@pytest.mark.parity(
    "boundary_loops",
    "pymeshfix",
    benchmarked=False,
    reason="n_boundaries is a property computed by load_array itself, so there is no separable "
    "operation to time -- a row would price the 67.9 ms load on bunny_decimated and report it as "
    "a loop count. The count is the whole answer, so it is asserted here instead.",
)
@pytest.mark.parametrize(
    ("mesh_name", "n_loops"),
    [("icosahedron", 0), ("hemisphere", 1), ("half_torus", 2), ("saddle_graded", 1)],
)
def test_boundary_loops_count_matches_pymeshfix(
    request: pytest.FixtureRequest, mesh_name: str, n_loops: int
) -> None:
    """
    Class A on the count: integer equality against ``PyTMesh.n_boundaries``.

    pymeshfix has no vertex-loop entry point -- it reports only how many rims there are -- so this
    is the whole of what it can say about this group, and it says it exactly: 0 / 1 / 2 across the
    three fixtures.

    Two things make the assert meaningful rather than incidental. The expected count is
    parametrized *in* rather than read off either library, so a pair of implementations that agreed
    on a wrong answer would still fail; and it spans a closed mesh, so one of the three cases is a
    genuine zero rather than the vacuous ``[] == []`` that comparing two open meshes would give.

    The load is asserted to have changed nothing first, which is not a formality here: the same
    call cuts connectivity before counting, and a hemisphere sliced without ``merge_vertices()``
    loads as 137 vertices from 121 and reports **17** rims where the surface has one. The
    ``tests/conftest.py`` fixtures merge, so they come back untouched.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    tin_pmf = trimesh_to_pymeshfix(mesh_tm)
    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)

    assert tin_pmf.n_points == mesh_tm.vertices.shape[0]  # the loader left the mesh alone
    assert tin_pmf.n_faces == mesh_tm.faces.shape[0]
    assert tin_pmf.n_boundaries == n_loops
    assert len(loops_wp) == n_loops


def _meshlib_hole_rings(mesh_ml: mm.Mesh) -> list[list[tuple[int, int]]]:
    """
    Every MeshLib hole as an ordered list of ``(org, dest)`` vertex pairs.

    This is the ``EdgeId`` -> ``(v0, v1)`` decoding the whole MeshLib boundary family runs on:
    ``findHoleRepresentiveEdges`` names one ``EdgeId`` per hole, ``getLeftRing`` walks that hole
    into an ordered ring of ``EdgeId``, and ``org`` / ``dest`` turn each one into the vertex pair.
    The chaining property ``dest(e_i) == org(e_{i+1})`` is asserted by the caller rather than
    assumed, because it is what makes the ring a *loop* rather than an unordered edge set.
    """
    rings_ml = []
    for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
        ring_ml = mesh_ml.topology.getLeftRing(edge_ml)
        rings_ml.append(
            [(mesh_ml.topology.org(e).get(), mesh_ml.topology.dest(e).get()) for e in ring_ml]
        )
    return rings_ml


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_loops", "meshlib")
def test_boundary_loops_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: the same loops after one named transform -- MeshLib walks each rim the *other* way.

    MeshLib has no vertex-loop entry point at all; it names one ``EdgeId`` per hole and the caller
    walks it. So the transform is the decoding in [`_meshlib_hole_rings`]: ring of ``EdgeId`` ->
    ``org()`` per edge -> vertex loop. That decoding is what every other MeshLib boundary and
    hole-filling comparison depends on, which is why this test asserts it in three separate pieces
    instead of trusting it -- the ring chains (``dest(e_i) == org(e_{i+1})``), the undirected edge
    sets agree, and the *directed* pairs are exactly ordito's reversed.

    That reversal is the transform, and it is pinned rather than canonicalised away:
    ``getLeftRing`` walks the **hole**, whose left face is the missing one, where
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges] follows the surface's own
    face winding. The two therefore run opposite by construction on every rim, and asserting that
    -- rather than comparing direction-agnostically the way
    [`test_boundary_loops_matches_trimesh_outline`] must -- is what would catch MeshLib changing
    the convention under us.

    Not run on ``mobius``, though it is the suite's other open fixture, and not because ordito
    cannot answer there -- [`test_boundary_loops_mobius_is_one_cycle`] shows it returns the correct
    single 78-cycle. It is that **no reference agrees with the truth**: MeshLib's hole ring reads
    156 and igl cuts the one cycle into three open chains. There is nothing to compare against.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    rings_ml = _meshlib_hole_rings(trimesh_to_meshlib(mesh_tm))

    # Non-vacuity, and the fixture check section 6 asks for: a raw ``slice_plane`` surface reports
    # 17 phantom rims to MeshLib where it has one, so the hole *count* is asserted before anything
    # per-hole is compared. These fixtures merge their vertices, which is what makes them sound.
    assert len(rings_ml) == len(loops_wp) > 0

    for ring_ml in rings_ml:
        # The ring is a loop: each edge's destination is the next edge's origin.
        assert all(ring_ml[i][1] == ring_ml[(i + 1) % len(ring_ml)][0] for i in range(len(ring_ml)))

    # Directed pairs: MeshLib's hole ring runs against the surface winding, edge for edge.
    edges_wp = od.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices).numpy()
    pairs_ml = np.array([pair for ring_ml in rings_ml for pair in ring_ml], dtype=np.int32)
    assert np.array_equal(lexsort_rows(edges_wp), lexsort_rows(pairs_ml[:, ::-1]))

    for loop_wp, ring_ml in zip(
        sorted((loop.numpy() for loop in loops_wp), key=lambda loop: int(loop.min())),
        sorted(rings_ml, key=lambda ring: min(org for org, _ in ring)),
        strict=True,
    ):
        loop_ml = np.array([org for org, _ in ring_ml], dtype=np.int32)
        # Reversed, then rotated onto ordito's start vertex -- an exact cyclic match, not the
        # direction-agnostic one, so the convention itself stays under test.
        reversed_ml = loop_ml[::-1]
        rotated_ml = np.roll(reversed_ml, -int(np.flatnonzero(reversed_ml == loop_wp[0])[0]))
        assert np.array_equal(loop_wp, rotated_ml)


def test_boundary_loops_mobius_is_one_cycle(mobius: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: on a non-orientable surface no reference computes the right answer.

    The ground truth is checked here rather than borrowed, and it is cheap to state: every one of
    the Moebius band's 78 boundary vertices lies on exactly two boundary edges, so the boundary is
    a disjoint union of cycles; walking it from any vertex covers all 78 and closes. One loop of
    78, which is also what topology says -- a Moebius band has a single boundary circle, and this
    fixture is a 39-column strip whose boundary wraps it twice.

    Both references are wrong here, differently, which is why this is an invariant test:

    - ``igl.boundary_loop_all`` returns ``1 + 39 + 38``. Each of those three has exactly one
      consecutive pair that is **not** a boundary edge -- they are open chains closed artificially,
      and one of them is a single vertex. ``igl.boundary_loop`` then reports the longest, 39.
    - MeshLib's ``findHoleRepresentiveEdges`` + ``getLeftRing`` gives a 156-edge ring.

    ordito was wrong too until the undirected fallback landed: the directed boundary edges are not
    a successor graph here (one seam vertex has out-degree 2), so ``succ[tail] = head`` dropped an
    edge and the walk returned 78 entries over 40 distinct vertices. That is the regression this
    pins -- the distinctness assert is the one that failed before, not the length.

    The loop *direction* is deliberately not asserted: with no consistent winding there is no
    direction to be right about, only a reproducible one.
    """
    mesh_tm, mesh_wp = mobius
    assert not od.validation.is_orientable(mesh_wp.indices)  # the fixture's whole point here

    boundary_pairs = {
        tuple(sorted(pair))
        for pair in od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).numpy().tolist()
    }
    degree = Counter(vertex for pair in boundary_pairs for vertex in pair)
    assert len(boundary_pairs) == 78
    assert set(degree.values()) == {2}, "2-regular is what makes the single-cycle claim meaningful"

    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    assert len(loops_wp) == 1
    loop_np = loops_wp[0].numpy()

    assert loop_np.size == 78
    assert len(set(loop_np.tolist())) == 78  # the assert that failed before the fallback existed
    assert set(loop_np.tolist()) == set(degree)
    assert all(
        tuple(sorted((int(loop_np[i]), int(loop_np[(i + 1) % 78])))) in boundary_pairs
        for i in range(78)
    ), "consecutive entries must be real boundary edges, and the last must close onto the first"

    # igl is not merely ordered differently -- it reports three loops where there is one.
    assert [len(loop) for loop in igl.boundary_loop_all(mesh_tm.faces.astype(np.int64))] == [
        1,
        39,
        38,
    ]


def test_boundary_loops_two_mobius_bands_keep_one_direction_each(
    mobius: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the undirected walk keeps exactly one of each loop's two mirrors.

    Two disjoint Moebius bands, the second's ids offset past the first's: the walk runs over darts,
    so each boundary circle comes back once per direction, and only the even-starting mirror is
    kept. Two loops of 78, in ascending order of their smallest vertex, each the first band's loop
    shifted by the offset -- which pins the filter per cycle rather than on the only cycle there
    is, and the ascending order of the kept cycles.
    """
    mesh_tm, mesh_wp = mobius
    n = mesh_tm.vertices.shape[0]
    vertices_np = np.concatenate([mesh_tm.vertices, mesh_tm.vertices + 5.0])
    faces_np = np.concatenate([mesh_tm.faces, mesh_tm.faces + n])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, mesh_wp.device)

    single_np = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)[0].numpy()
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)

    assert [loop_wp.size for loop_wp in loops_wp] == [78, 78]
    assert np.array_equal(loops_wp[0].numpy(), single_np)
    assert np.array_equal(loops_wp[1].numpy(), single_np + n)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus", "icosahedron"])
def test_boundary_loop_sizes_helper_agrees_with_boundary_loops(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the numpy loop tracer in ``tests.comparisons`` reproduces ``boundary_loops``' sizes.

    [`boundary_loop_sizes`][tests.comparisons.boundary_loop_sizes] is used as an *oracle* by the
    hole-filling tests, so it needs one of its own -- a wrong tracer would silently weaken every
    assert built on it. ``boundary_loops`` is the right thing to check it against here because it is
    itself pinned element-wise against ``igl.boundary_loop_all`` and ``Trimesh.outline()`` two tests
    up. Measured: 24 on ``hemisphere``, 32 and 32 on ``half_torus``, none on ``icosahedron``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    sizes_np = boundary_loop_sizes(np.asarray(mesh_tm.faces))

    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    expected = sorted((loop_wp.size for loop_wp in loops_wp if loop_wp.size >= 3), reverse=True)
    assert sizes_np == expected
    assert (len(expected) == 0) == bool(mesh_tm.is_watertight)


def test_boundary_loop_sizes_refuses_a_pinched_rim() -> None:
    """
    Two rims meeting at one vertex have no well-defined loop through it, and the helper says so.

    Tracing on regardless would return a plausible wrong count rather than an error, which is the
    failure mode an oracle can least afford. Built by opening two holes in an icosphere that share a
    vertex -- reachable from ordinary face deletion, not a contrived mesh.
    """
    _, faces_np = _pinched_icosphere()

    with pytest.raises(ValueError, match="two incident boundary edges"):
        boundary_loop_sizes(faces_np)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_loops_with_offsets_matches_boundary_loops(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the list form is the packed form split, loop for loop.

    The default list is views into one shared buffer; ``copy=True`` must give the same loops in
    independent storage.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    copies_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices, copy=True)
    flat_wp, offsets_wp = od.boundary.boundary_loops_with_offsets(mesh_wp.points, mesh_wp.indices)

    offsets_np = offsets_wp.numpy()
    assert len(loops_wp) == offsets_np.size - 1
    assert int(offsets_np[0]) == 0
    assert flat_wp.size == int(offsets_np[-1])
    for i, loop_wp in enumerate(loops_wp):
        begin, end = int(offsets_np[i]), int(offsets_np[i + 1])
        assert np.array_equal(loop_wp.numpy(), flat_wp.numpy()[begin:end])

    assert len(copies_wp) == len(loops_wp)
    for view_wp, copy_wp in zip(loops_wp, copies_wp, strict=True):
        assert np.array_equal(view_wp.numpy(), copy_wp.numpy())
    if len(loops_wp) > 1:
        assert loops_wp[0].ptr is not None
        assert loops_wp[1].ptr is not None
        assert loops_wp[0].ptr != loops_wp[1].ptr
        # Adjacent views share one allocation; the copies do not.
        assert loops_wp[1].ptr - loops_wp[0].ptr == 4 * loops_wp[0].size


@pytest.mark.parametrize("rim", [7, 8, 9, 15, 16, 17, 63, 64, 65, 255, 256, 257, 4095, 4097])
def test_boundary_loops_rank_across_jump_round_boundaries(device: str, rim: int) -> None:
    """
    Not a library comparison: the loop of a fan is known, so the oracle is its construction.

    A disk fanned from one centre over a rim of ``rim`` vertices, its ids shuffled: the one loop
    is the rim in winding order, from its smallest id. The rim length is the ranked-node count, so
    lengths straddling the powers of the pointer-jump widths (8 and 16) put the closed-cycle
    ranking one side and the other of each added round.
    """
    rng = np.random.default_rng(rim)
    ids = rng.permutation(rim + 1).astype(np.int32)
    centre, ring = ids[0], ids[1:]
    faces_np = np.stack([np.full(rim, centre), ring, np.roll(ring, -1)], axis=1)
    angle = 2.0 * np.pi * np.arange(rim) / rim
    positions = np.zeros((rim + 1, 3))
    positions[ring] = np.stack([np.cos(angle), np.sin(angle), np.zeros(rim)], axis=1)
    vertices_wp, faces_wp = numpy_to_warp(positions, faces_np, device)

    flat_wp, offsets_wp = od.boundary.boundary_loops_with_offsets(vertices_wp, faces_wp)

    assert np.array_equal(flat_wp.numpy(), np.roll(ring, -int(np.argmin(ring))))
    assert np.array_equal(offsets_wp.numpy(), np.array([0, rim], dtype=np.int32))


def test_boundary_loops_non_manifold_terminates(device: str) -> None:
    """
    Not a library comparison: a bowtie's two rims, each its own loop, from face winding alone.

    Two triangles sharing only vertex 2 give that vertex two outgoing boundary edges, so a walk
    over *vertices* has two successors for it and no answer -- it used to keep one (last write
    wins) and return a loop missing edges. Walked by halfedge sector, each triangle's rim closes on
    its own, in the triangle's own winding. Also the original regression: it must terminate.
    """
    vertices_wp = wp.array(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2, 2, 3, 4], dtype=np.int32), dtype=wp.int32, device=device)

    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)

    def rotated_to_minimum(loop_np: np.ndarray) -> tuple[int, ...]:
        return tuple(int(v) for v in np.roll(loop_np, -int(np.argmin(loop_np))))

    assert sorted(rotated_to_minimum(loop_wp.numpy()) for loop_wp in loops_wp) == [
        (0, 1, 2),
        (2, 3, 4),
    ]


def test_boundary_loops_walk_a_pinched_rim_edge_by_edge(device: str) -> None:
    """
    Class B, against trimesh's boundary edges: each appears in exactly one loop, once, in winding.

    Two holes opened in an icosphere that share a vertex, the construction
    ``test_boundary_loop_sizes_refuses_a_pinched_rim`` uses -- reachable from ordinary face
    deletion. At the shared vertex the rim has two outgoing edges, and walking it by vertex left
    successor slots at ``0``: loops with fake ``(0, 0)`` edges, on both devices. The named
    transform is from consecutive loop pairs to directed edges; the claim is that they are exactly
    the oriented boundary edges, which ``trimesh`` reads off the faces with no loop walk at all.
    """
    vertices_np, faces_np = _pinched_icosphere()
    holed_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    # Non-vacuity: the rim really is pinched -- more boundary edges touch some vertex than two.
    boundary_np = holed_tm.edges[tm_grouping.group_rows(holed_tm.edges_sorted, require_count=1)]
    assert np.bincount(boundary_np.ravel()).max() > 2
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)

    walked_np = np.concatenate(
        [
            np.stack([loop_np, np.roll(loop_np, -1)], axis=1)
            for loop_np in (loop_wp.numpy() for loop_wp in loops_wp)
        ]
    )
    assert (walked_np[:, 0] != walked_np[:, 1]).all()  # no fake self-edge
    assert Counter(map(tuple, walked_np.tolist())) == Counter(map(tuple, boundary_np.tolist()))


@pytest.mark.parametrize(
    "extra_faces",
    [[[1, 0, 5], [0, 1, 6]], [[0, 1, 6], [0, 1, 7]], [[1, 0, 7], [1, 0, 5]]],
    ids=["fin_opposed", "fin_aligned", "fin_reversed"],
)
def test_boundary_loops_never_invent_an_edge(device: str, extra_faces: list[list[int]]) -> None:
    """
    Class B, against trimesh's boundary edges: on any input, every loop pair is a boundary edge.

    The guarantee ``selection.delete_region_keep_boundary`` rests on, pinned where it can fail: a
    bowtie pinched at vertex 0 whose pinch fan carries a *three-faced* edge ``(0, 1)``, so the
    pinch walk's rotation meets an edge with no twin. What such a mesh may do is lose a loop -- the
    rotation dead-ends there and that chain never closes -- so the assert is one-sided: no pair
    that is not a boundary edge, no boundary edge twice. Three fin orientations cover a twin table
    with the three-faced edge wound each way. The positive half is
    ``test_boundary_loops_walk_a_pinched_rim_edge_by_edge``.
    """
    vertices_np = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [-1, 0, 0],
            [-1, -1, 0],
            [0.5, -1, 0],
            [0.5, 0, 1],
            [0.5, 0, -1],
        ],
        dtype=np.float64,
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 4], *extra_faces])
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    # Non-vacuity: vertex 0 is a pinch, and edge (0, 1) is shared by three faces.
    assert Counter(map(tuple, mesh_tm.edges_sorted.tolist()))[(0, 1)] == 3
    boundary_np = mesh_tm.edges_sorted[
        tm_grouping.group_rows(mesh_tm.edges_sorted, require_count=1)
    ]
    boundary = set(map(tuple, boundary_np.tolist()))
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)

    pairs = [
        tuple(sorted((int(a), int(b))))
        for loop_np in (loop_wp.numpy() for loop_wp in loops_wp)
        for a, b in zip(loop_np, np.roll(loop_np, -1), strict=True)
    ]
    assert pairs  # the untouched triangle's rim is still traced
    assert set(pairs) <= boundary
    assert len(pairs) == len(set(pairs))


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_loop(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the singular form against ``igl.boundary_loop``, which is igl's *longest* loop.

    That projection is a real difference from ``boundary_loop_all`` and is why this is its own
    test: on ``half_torus``, whose two rims are the same length, it also pins the tie-break.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loop_igl = igl.boundary_loop(mesh_tm.faces.astype(np.int64))
    loop_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)

    assert np.array_equal(loop_wp.numpy(), loop_igl)


@pytest.mark.parametrize(
    ("faces_np", "expected_ears"),
    [
        # An open fan: centre 0 with a 4-vertex boundary chain, so the two end triangles are ears.
        (np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int32), 2),
        # Two triangles sharing one edge: both are ears.
        (np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32), 2),
        # A four-triangle strip: only the two ends are ears, the middle pair has one boundary edge.
        (np.array([[0, 1, 2], [1, 3, 2], [3, 4, 2], [4, 5, 2]], dtype=np.int32), 2),
    ],
    ids=["fan3", "strip2", "strip4"],
)
@pytest.mark.parity("ears", "igl")
def test_ears_match_igl(device: str, faces_np: np.ndarray, expected_ears: int) -> None:
    """
    Class B (an edge-numbering shift): ``ear_opp`` is offset by one between the two libraries.

    Both report an ear as ``(face, index of the non-boundary edge)`` and both find the same faces,
    but the *edge numbering* differs and the difference is exactly a cyclic shift:

    - ordito numbers local edge ``i`` as ``(faces[f, i], faces[f, (i + 1) % 3])``;
    - libigl's ``ears`` reads its mask from ``on_boundary``, whose column ``i`` is documented as
      "whether **opposite** facet is on boundary" -- edge ``i`` is the one *opposite vertex* ``i``,
      i.e. ``(faces[f, (i + 1) % 3], faces[f, (i + 2) % 3])``.

    So ``ordito_opp == (igl_opp + 1) % 3``, and that is the named transform. Neither convention is
    wrong; ``boundary.ears``'s docstring states ordito's.

    **This test replaces a vacuous one.** The previous version compared the two libraries on
    ``hemisphere`` and ``half_torus``, where *neither* returns any ear at all -- the assert was
    ``[] == []`` on both fixtures, so the numbering difference went unnoticed and any regression
    would have too. The inputs here are the smallest meshes that produce ears, and each case asserts
    the expected count first, so an implementation returning nothing fails rather than passes.

    The last assert is the one that does not lean on igl: for every reported ear it checks against
    ``oriented_boundary_edges`` that the two edges *other* than ``ear_opp`` really are boundary
    edges, under ordito's own numbering.
    """
    faces_wp = wp.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device)
    # Positions are irrelevant to ears (pure connectivity) but boundary_edges wants a vertex buffer.
    n_vertices = int(faces_np.max()) + 1
    vertices_wp = wp.array(
        np.ascontiguousarray(
            np.stack([np.arange(n_vertices), np.zeros(n_vertices), np.zeros(n_vertices)], axis=1),
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )

    ear_igl, ear_opp_igl = igl.ears(np.ascontiguousarray(faces_np, dtype=np.int64))
    ear_wp, ear_opp_wp = od.boundary.ears(faces_wp)

    assert ear_wp.size == expected_ears
    assert np.asarray(ear_igl).size == expected_ears

    pairs_igl = np.stack([ear_igl, (np.asarray(ear_opp_igl) + 1) % 3], axis=1)
    pairs_wp = np.stack([ear_wp.numpy(), ear_opp_wp.numpy()], axis=1)
    assert np.array_equal(lexsort_rows(pairs_wp), lexsort_rows(pairs_igl))

    oriented_boundary = od.boundary.oriented_boundary_edges(vertices_wp, faces_wp)
    boundary_set = {tuple(row) for row in oriented_boundary.numpy()}
    directed_edges = od.edges.faces_to_edges(faces_wp).numpy()
    for face_idx, opp in zip(ear_wp.numpy(), ear_opp_wp.numpy(), strict=True):
        f = int(face_idx)
        for local_edge in ((int(opp) + 1) % 3, (int(opp) + 2) % 3):
            assert tuple(directed_edges[3 * f + local_edge]) in boundary_set


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_ears_none_on_smooth_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class D exemption in test form: neither library finds an ear on either subdivided rim.

    A rim built by subdivision never leaves a triangle with two boundary edges, so this is the
    negative half of ``test_ears_match_igl`` and is kept separate from it rather than standing in
    for a comparison. Not ``conftest.OPEN_MESHES``: ``saddle_graded`` is a split quad grid, two of
    whose corners are single triangles -- ears, which both libraries find -- and ears are pure
    connectivity, so its grading adds nothing ``test_ears_match_igl`` does not already cover.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces.astype(np.int64)

    ear_igl, _ear_opp_igl = igl.ears(faces_np)
    ear_wp, ear_opp_wp = od.boundary.ears(mesh_wp.indices)

    assert np.asarray(ear_igl).size == 0
    assert ear_wp.shape == (0,)
    assert ear_opp_wp.shape == (0,)


def test_boundary_queries_watertight(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: a closed mesh has no boundary edges, vertices, loops or ears."""
    _, mesh_wp = icosahedron
    _assert_no_boundary(mesh_wp.points, mesh_wp.indices)


def test_boundary_queries_empty(device: str) -> None:
    """Not a library comparison: a mesh with no vertices or faces answers every query empty."""
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    _assert_no_boundary(vertices_wp, faces_wp)


@pytest.mark.parity("loop_perimeters", "meshlib")
@pytest.mark.parity("loop_directed_areas", "meshlib")
@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_loop_perimeters_and_directed_areas_match_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the perimeter and Class B on the area vector, whose **sign** is the transform.

    MeshLib answers per hole through a representative edge: ``holePerimeter`` is a scalar and
    ``holeDirArea`` a ``Vector3d`` whose norm is the spanned area and whose direction is the loop's
    normal. The perimeter agrees to 1e-6 with no transform at all.

    The directed area comes back **negated**, and that is a convention rather than an error: the two
    libraries walk a rim in opposite directions, so the same loop's winding -- and so the sign of
    every cross product summed around it -- is opposite. Measured on the sliced hemisphere,
    ``[0, 0, -2.9461]`` against ``[0, 0, +2.9461]``. The **norms** are compared without any
    transform, which is the part that carries the magnitude, and the negation is asserted separately
    so a genuine direction disagreement could not hide inside it.

    The two libraries enumerate holes in their own orders, so the scalar comparisons go through
    **sorted** lists -- the second named transform, and the reason ``half_torus`` (two rims) is
    usable here at all. The signed vector needs an actual pairing, so it is asserted only where
    there is a single rim, and the branch is asserted to be reached.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops_wp) > 0  # non-vacuity: an open fixture, so there is a rim to measure

    perimeters_np = od.boundary.loop_perimeters(vertices_wp, loops_wp).numpy()
    areas_np = od.boundary.loop_directed_areas(vertices_wp, loops_wp).numpy()
    assert perimeters_np.min() > 0.0
    assert np.linalg.norm(areas_np, axis=1).min() > 0.0

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
    assert len(holes_ml) == len(loops_wp)
    perimeters_ml = np.array(
        [mm.holePerimeter(mesh_ml.topology, mesh_ml.points, e) for e in holes_ml]
    )
    areas_ml = np.array(
        [
            [
                mm.holeDirArea(mesh_ml.topology, mesh_ml.points, e).x,
                mm.holeDirArea(mesh_ml.topology, mesh_ml.points, e).y,
                mm.holeDirArea(mesh_ml.topology, mesh_ml.points, e).z,
            ]
            for e in holes_ml
        ]
    )

    assert np.allclose(np.sort(perimeters_np), np.sort(perimeters_ml), rtol=1e-5)
    assert np.allclose(
        np.sort(np.linalg.norm(areas_np, axis=1)),
        np.sort(np.linalg.norm(areas_ml, axis=1)),
        rtol=1e-5,
    )
    if len(loops_wp) == 1:
        assert np.allclose(areas_np[0], -areas_ml[0], rtol=1e-5, atol=1e-5)
    else:
        assert mesh_name == "half_torus"  # the only multi-rim fixture here, and it stays that way


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_loop_measures_agree_with_the_single_loop_forms(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a parity assert: it pins the batched measures to ``ordito.polyline``, which has an oracle.

    ``loop_perimeters`` is a segmented ``polyline_length(closed=True)`` and must equal it loop for
    loop; ``loop_directed_areas`` must point along ``polyline_normal``, which is the same quantity
    normalized. Both single-loop functions are compared against references elsewhere, so a
    divergence here is the batching's.

    Also asserted: the directed area is **origin-independent** -- translating the mesh cannot change
    it, because the cross products of a closed ring cancel the shift. That is the property that lets
    the kernel skip a centroid pass, and it is invisible in any comparison against a reference that
    also happens to be centred.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    perimeters_np = od.boundary.loop_perimeters(vertices_wp, loops_wp).numpy()
    areas_np = od.boundary.loop_directed_areas(vertices_wp, loops_wp).numpy()

    for index, loop_wp in enumerate(loops_wp):
        points_wp = od.array.gather(vertices_wp, loop_wp)
        assert np.isclose(
            perimeters_np[index], od.polyline.polyline_length(points_wp, closed=True), rtol=1e-5
        )
        normal_wp = od.polyline.polyline_normal(points_wp)
        direction_np = areas_np[index] / np.linalg.norm(areas_np[index])
        assert np.allclose(direction_np, np.array(list(normal_wp)), rtol=1e-4, atol=1e-4)

    shifted_wp = points_to_warp(
        vertices_wp.numpy() + np.array([3.0, -7.0, 11.0], dtype=np.float32), vertices_wp.device
    )
    assert np.allclose(
        od.boundary.loop_directed_areas(shifted_wp, loops_wp).numpy(),
        areas_np,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_batched_loop_measures_agree_with_the_list_forms(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the packed entry points against the list ones, which carry the oracle.

    ``loop_perimeters_from_offsets`` and ``loop_directed_areas_from_offsets`` exist so that a caller
    holding ``boundary_loops_with_offsets``' output can measure it without splitting it back into a
    Python list
    and repacking; this asserts the two forms are the same measure. The list forms are the ones
    compared against a reference, so a divergence here is the packed path's.

    Asserted at ``1e-5`` rather than exactly, and that is not slack: both kernels accumulate with
    ``wp.atomic_add``, so on CUDA the summation order differs between two launches over the same
    data and the last bits of a ``float32`` differ with it. Measured while these were written --
    bit-identical on the CPU, agreeing to 9.8e-08 against a host recomputation on CUDA. An
    ``array_equal`` here would fail on CUDA for a correct implementation.

    Also asserted: passing the precomputed ``loop_id`` gives the same answer as letting the function
    derive it, which is the keyword ``ordito.holes`` uses to keep its own cost.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops_wp) > 0  # non-vacuity: an empty comparison would pass and test nothing
    flat_wp, offsets_wp = od.boundary.boundary_loops_with_offsets(vertices_wp, faces_wp)

    assert np.allclose(
        od.boundary.loop_perimeters_from_offsets(vertices_wp, flat_wp, offsets_wp).numpy(),
        od.boundary.loop_perimeters(vertices_wp, loops_wp).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    areas_np = od.boundary.loop_directed_areas(vertices_wp, loops_wp).numpy()
    assert np.allclose(
        od.boundary.loop_directed_areas_from_offsets(vertices_wp, flat_wp, offsets_wp).numpy(),
        areas_np,
        rtol=1e-5,
        atol=1e-5,
    )

    sizes_np = np.diff(offsets_wp.numpy())
    owner_np = np.repeat(np.arange(sizes_np.size, dtype=np.int32), sizes_np)
    owner_wp = wp.array(owner_np, dtype=wp.int32, device=vertices_wp.device)
    assert np.allclose(
        od.boundary.loop_directed_areas_from_offsets(
            vertices_wp, flat_wp, offsets_wp, loop_id=owner_wp
        ).numpy(),
        areas_np,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "offsets",
    [[0, 3], [0, 3, 2, 6], [1, 6], [0, 7]],
    ids=["unterminated", "decreasing", "start", "end"],
)
def test_packed_loop_measures_reject_malformed_offsets(device: str, offsets: list[int]) -> None:
    """
    Not a library comparison: the ``validate`` guard is what keeps a malformed pair off the kernel.

    Six packed positions. An unterminated, length-``n`` offsets array fails it like any other
    malformed one, and ``validate=False`` is the documented way past it.
    """
    vertices_wp = wp.zeros(6, dtype=wp.vec3, device=device)
    flat_wp = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array(offsets, dtype=np.int32), dtype=wp.int32, device=device)
    for measure in (
        od.boundary.loop_perimeters_from_offsets,
        od.boundary.loop_directed_areas_from_offsets,
    ):
        with pytest.raises(ValueError, match="offsets must run non-decreasing"):
            measure(vertices_wp, flat_wp, offsets_wp)
    good_wp = wp.array(np.array([0, 2, 6], dtype=np.int32), dtype=wp.int32, device=device)
    assert od.boundary.loop_perimeters_from_offsets(vertices_wp, flat_wp, good_wp).shape == (2,)


def test_loop_measures_empty(device: str) -> None:
    """Not a library comparison: no loops, and a loop of zero length, both measure to nothing."""
    vertices_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    assert od.boundary.loop_perimeters(vertices_wp, []).shape == (0,)
    assert od.boundary.loop_directed_areas(vertices_wp, []).shape == (0,)
    empty_loop_wp = warp_empty(0, wp.int32, device)
    assert od.boundary.loop_perimeters(vertices_wp, [empty_loop_wp]).shape == (0,)
    # Both halves of the loop guard, which is ``odt.ensure_ndim`` at one call rather than the
    # hand-written rank-and-dtype test it replaced. Only the dtype half was ever reached before, so
    # the rank half is here to pin that the single call still covers what the two-clause ``if`` did.
    with pytest.raises(TypeError, match=r"expected dtype"):
        od.boundary.loop_perimeters(vertices_wp, [wp.zeros(3, dtype=wp.float32, device=device)])
    with pytest.raises(TypeError, match=r"expected 1D array"):
        od.boundary.loop_perimeters(vertices_wp, [wp.zeros((3, 2), dtype=wp.int32, device=device)])


def _boundary_indices_tm(mesh_tm: tm.Trimesh) -> np.ndarray:
    return np.asarray(tm_grouping.group_rows(mesh_tm.edges_sorted, require_count=1))


def _assert_edge_key_order(rows_np: np.ndarray) -> None:
    """Assert undirected rows strictly ascend by ``(max, min)``, ``edges_unique``'s key order."""
    high = rows_np.max(axis=1).astype(np.int64)
    low = rows_np.min(axis=1).astype(np.int64)
    assert np.all(np.diff(high * (int(high.max(initial=0)) + 1) + low) > 0)


def _pinched_icosphere() -> tuple[np.ndarray, np.ndarray]:
    """``icosphere(2)`` with two holes opened that share one vertex: a pinched rim."""
    sphere_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    centers_np = sphere_tm.triangles_center
    keep_np = np.ones(sphere_tm.faces.shape[0], dtype=bool)
    keep_np[np.argsort(-centers_np[:, 2])[:6]] = False
    keep_np[np.argsort(centers_np[:, 2])[:2]] = False
    return np.asarray(sphere_tm.vertices), np.asarray(sphere_tm.faces[keep_np])


def _assert_no_boundary(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> None:
    """Assert every boundary query answers empty on a mesh with no boundary."""
    assert od.boundary.boundary_edges(vertices_wp, faces_wp).shape == (0, 2)
    assert od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).shape == (0, 2)
    assert od.boundary.boundary_vertex_indices(vertices_wp, faces_wp).shape == (0,)
    assert od.boundary.boundary_vertices(vertices_wp, faces_wp).shape == (0,)
    assert od.boundary.boundary_loops(vertices_wp, faces_wp) == []
    assert od.boundary.longest_boundary_loop(vertices_wp, faces_wp).shape == (0,)
    ear_wp, ear_opp_wp = od.boundary.ears(faces_wp)
    assert ear_wp.shape == (0,)
    assert ear_opp_wp.shape == (0,)
