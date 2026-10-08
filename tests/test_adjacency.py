"""Regression tests for ``ordito.adjacency`` against Trimesh (CPU reference)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from ordito.constants import TOLERANCE_MERGE
from tests.comparisons import lexsort_rows, same_partition
from tests.conftest import CLOSED_MESHES, MESHES
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_meshlib,
    trimesh_to_meshlib,
    trimesh_to_pyvista,
    warp_empty,
)


def _adjacency_order(adjacency_np: np.ndarray) -> np.ndarray:
    """Row order that sorts ``(f0, f1)`` adjacency pairs canonically."""
    return np.lexsort((adjacency_np[:, 1], adjacency_np[:, 0]))


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_adjacency", "trimesh")
def test_face_adjacency(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """Class A: face pairs and their shared edges, elementwise after a canonical row sort."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    adjacency_edges_tm = mesh_tm.face_adjacency_edges
    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )

    order_tm = _adjacency_order(adjacency_tm)
    order_wp = _adjacency_order(adjacency_wp.numpy())
    assert np.array_equal(adjacency_wp.numpy()[order_wp], adjacency_tm[order_tm])
    assert np.array_equal(adjacency_edges_wp.numpy()[order_wp], adjacency_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", MESHES)
def test_face_adjacency_radix_is_invariant_to_an_oversized_base(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the edge grouping is unchanged by any base above ``max(faces)``.

    ``_sorted_pair_scan`` asserts the partition is invariant to a sufficiently large radix,
    which is what lets a caller holding a vertex buffer with *unreferenced* vertices pass
    ``vertices.shape[0]`` rather than pay the ``reduce.minmax`` that infers ``max(faces) + 1``.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    tight = od.array.index_bound(mesh_wp.indices)

    baseline_wp = od.adjacency.face_adjacency(mesh_wp.indices, n_vertices=tight)
    for base in (tight + 1, tight + 1000):
        assert np.array_equal(
            od.adjacency.face_adjacency(mesh_wp.indices, n_vertices=base).numpy(),
            baseline_wp.numpy(),
        )


@pytest.mark.parametrize("mesh_name", [*CLOSED_MESHES, "sphere_irregular", "boy_surface"])
@pytest.mark.parametrize("n_vertices_given", [False, True])
def test_face_adjacency_edges_paired_matches_the_grouped_path(
    request: pytest.FixtureRequest, mesh_name: str, n_vertices_given: bool
) -> None:
    """
    Ordito against ordito: ``edges_paired=True`` is byte-identical to the run-detecting path.

    The default path carries the oracle (``test_face_adjacency``); this pins the shortcut to it on
    every closed fixture, including the non-orientable ``boy_surface``, over both radix sources.
    The row *order* is compared, not only the set, because callers mix the two paths and rely on
    row alignment.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    n_vertices = mesh_wp.points.size if n_vertices_given else None
    grouped_wp, grouped_edges_wp = od.adjacency.face_adjacency(
        faces_wp, return_edges=True, n_vertices=n_vertices
    )
    paired_wp, paired_edges_wp = od.adjacency.face_adjacency(
        faces_wp, return_edges=True, n_vertices=n_vertices, edges_paired=True
    )
    assert paired_wp.shape == (3 * (faces_wp.size // 3) // 2, 2)
    assert np.array_equal(paired_wp.numpy(), grouped_wp.numpy())
    assert np.array_equal(paired_edges_wp.numpy(), grouped_edges_wp.numpy())
    assert np.array_equal(
        od.adjacency.face_adjacency(faces_wp, edges_paired=True).numpy(), grouped_wp.numpy()
    )


def test_face_adjacency_edges_paired_rejects_an_odd_face_count(device: str) -> None:
    """A closed triangle mesh has an even face count, so an odd one cannot keep the promise."""
    faces_wp = wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="even face count"):
        od.adjacency.face_adjacency(faces_wp, edges_paired=True)


@pytest.mark.parametrize(
    ("function", "shapes"),
    [
        pytest.param(
            lambda faces_wp, _: od.adjacency.face_adjacency(faces_wp, return_edges=True),
            [(0, 2), (0, 2)],
            id="face_adjacency",
        ),
        pytest.param(
            lambda faces_wp, _: od.adjacency.vertex_face_adjacency(faces_wp, n_vertices=0),
            [(0,), (1,)],
            id="vertex_face_adjacency",
        ),
        pytest.param(
            lambda faces_wp, _: (od.adjacency.face_adjacency_unshared(faces_wp),),
            [(0, 2)],
            id="face_adjacency_unshared",
        ),
        pytest.param(
            lambda faces_wp, vertices_wp: (
                od.adjacency.face_adjacency_angles(vertices_wp, faces_wp),
            ),
            [(0,)],
            id="face_adjacency_angles",
        ),
        pytest.param(
            lambda faces_wp, vertices_wp: (
                od.adjacency.face_adjacency_projections(vertices_wp, faces_wp),
            ),
            [(0,)],
            id="face_adjacency_projections",
        ),
        pytest.param(
            lambda faces_wp, vertices_wp: (
                od.adjacency.face_adjacency_convex(vertices_wp, faces_wp),
            ),
            [(0,)],
            id="face_adjacency_convex",
        ),
    ],
)
def test_empty_mesh_gives_empty_tables(
    device: str,
    function: Callable[[wp.array[wp.int32], wp.array[wp.vec3]], tuple[Any, ...]],
    shapes: list[tuple[int, ...]],
) -> None:
    """
    Each table on an empty face buffer has its documented shape with no rows.

    ``vertex_face_adjacency`` at ``n_vertices=0`` still returns its one ``[0]`` offset.
    """
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = warp_empty(0, wp.vec3, device)
    results = function(faces_wp, vertices_wp)
    assert [result.shape for result in results] == shapes


@pytest.mark.parametrize(
    "function",
    [
        od.adjacency.face_adjacency_unshared,
        od.adjacency.face_adjacency_projections,
        od.adjacency.face_adjacency_convex,
    ],
)
def test_half_a_precomputed_pair_raises_even_on_an_empty_mesh(
    device: str, function: Callable[..., object]
) -> None:
    """
    Not a library comparison: no reference takes a precomputed face-adjacency pair at all.

    The four wrappers that accept ``(face_adjacency, face_adjacency_edges)`` used to disagree about
    when a half-supplied pair is rejected -- two checked before their empty-mesh guard and two
    returned an empty answer first, so the same wrong call raised or did not depending on the mesh.
    The empty mesh is the whole point of the test: a non-empty one has always raised, so a
    regression here is invisible without it.

    ``face_adjacency_unshared`` takes ``faces`` first and the other two take ``vertices, faces``,
    which is why the call goes through ``*args`` rather than a shared signature.
    """
    faces_wp = warp_empty(0, wp.int32, device)
    vertices_wp = warp_empty(0, wp.vec3, device)
    adjacency_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    args = (
        (faces_wp,) if function is od.adjacency.face_adjacency_unshared else (vertices_wp, faces_wp)
    )
    with pytest.raises(ValueError, match="both be provided or both omitted"):
        function(*args, adjacency_wp)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_the_precomputed_pair_reaches_the_same_answer_as_deriving_it(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: passing the pair in agrees with letting each wrapper derive it.

    The wrappers taking ``(face_adjacency, face_adjacency_edges)`` each derive it inline from
    [`face_adjacency`][ordito.adjacency.face_adjacency] when it is omitted, so nothing external
    can be the oracle -- the claim is that the two paths are the same computation, and the oracle
    for the derived path is the reference comparison each wrapper carries in its own test.

    ``n_vertices`` is exercised on the derive path because it is documented as changing only the
    row-hashing radix and not the answer; a wrong radix collides edge keys and silently drops
    adjacency rows, which is what the row-count assert below would catch.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size
    adjacency_wp, edges_wp = od.adjacency.face_adjacency(mesh_wp.indices, return_edges=True)
    assert int(adjacency_wp.shape[0]) > 0
    tight_wp, tight_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True, n_vertices=n_vertices
    )
    assert np.array_equal(tight_wp.numpy(), adjacency_wp.numpy())
    assert np.array_equal(tight_edges_wp.numpy(), edges_wp.numpy())

    for supplied_np, derived_np in (
        (
            od.adjacency.face_adjacency_unshared(mesh_wp.indices, adjacency_wp, edges_wp).numpy(),
            od.adjacency.face_adjacency_unshared(mesh_wp.indices).numpy(),
        ),
        (
            od.adjacency.face_adjacency_projections(
                mesh_wp.points, mesh_wp.indices, adjacency_wp, edges_wp
            ).numpy(),
            od.adjacency.face_adjacency_projections(mesh_wp.points, mesh_wp.indices).numpy(),
        ),
        (
            od.adjacency.face_adjacency_convex(
                mesh_wp.points, mesh_wp.indices, adjacency_wp, edges_wp
            ).numpy(),
            od.adjacency.face_adjacency_convex(mesh_wp.points, mesh_wp.indices).numpy(),
        ),
    ):
        assert len(supplied_np) == int(adjacency_wp.shape[0])
        assert np.array_equal(supplied_np, derived_np)


def test_require_paired_adjacency_accepts_both_and_neither(device: str) -> None:
    """
    Not a library comparison: no reference takes a precomputed face-adjacency pair at all.

    The two accepting cases as well as the raise, because a validator that rejects everything
    passes a test written around the raise alone.
    """
    pair_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    od.adjacency.require_paired_adjacency(None, None)
    od.adjacency.require_paired_adjacency(pair_wp, pair_wp)
    for half in ((pair_wp, None), (None, pair_wp)):
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            od.adjacency.require_paired_adjacency(*half)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("vertex_face_adjacency", "igl")
def test_vertex_face_adjacency_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row order): the same ``(vertex_faces, offsets)`` CSR, arbitrary within a row.

    ``igl.vertex_triangle_adjacency(F, n)`` returns ``(VF, NI)`` -- the payload and the offsets, in
    that order, exactly ordito's pair reversed -- so the only transform is the unpacking plus
    sorting each row. Both give ``n_vertices + 1`` offsets, so no sentinel has to be appended.

    Row order is genuinely undefined in ordito's version (a counting-sort scatter, so it is thread
    order) and the docstring says so, which is why the rows are compared as **sets**: the scatter
    picks each slot with a ``wp.atomic_add`` on a per-vertex cursor, so two runs on CUDA order a row
    differently. The offsets are compared exactly: those are not order-dependent, and an off-by-one
    there is the failure mode this function's consumers -- the decimator's normal-flip guard -- see
    as silent corruption.

    Both the supplied-``n_vertices`` call and the one that infers it (at the cost of a readback)
    are compared against igl.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    n_vertices = mesh_wp.points.size
    faces_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)

    payload_igl, offsets_igl = igl.vertex_triangle_adjacency(faces_np, n_vertices)
    for payload_wp, offsets_wp in (
        od.adjacency.vertex_face_adjacency(faces_wp, n_vertices=n_vertices),
        od.adjacency.vertex_face_adjacency(faces_wp),
    ):
        assert np.array_equal(offsets_wp.numpy(), np.asarray(offsets_igl).ravel())
        bounds_np, payload_np = offsets_wp.numpy(), payload_wp.numpy()
        for vertex in range(n_vertices):
            row_wp = payload_np[bounds_np[vertex] : bounds_np[vertex + 1]]
            row_igl = np.asarray(payload_igl).ravel()[bounds_np[vertex] : bounds_np[vertex + 1]]
            assert np.array_equal(np.sort(row_wp), np.sort(row_igl))


def test_vertex_face_adjacency_unreferenced_vertex(device: str) -> None:
    """
    A vertex no face touches gets an **empty row**, not a missing one.

    That is the property the offsets encode and the reason ``n_vertices`` is a parameter rather than
    inferred unconditionally: with two trailing unreferenced vertices the payload is unchanged and
    only the offsets grow, repeating the final value.
    """
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    payload_wp, offsets_wp = od.adjacency.vertex_face_adjacency(faces_wp, n_vertices=5)

    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1, 2, 3, 3, 3], dtype=np.int32))
    assert np.array_equal(np.sort(payload_wp.numpy()), np.zeros(3, dtype=np.int32))


def test_vertex_face_adjacency_zero_rows_with_faces(device: str) -> None:
    """``n_vertices=0`` on a non-empty mesh returns zeros, not an unwritten buffer."""
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    payload_wp, offsets_wp = od.adjacency.vertex_face_adjacency(faces_wp, n_vertices=0)
    assert offsets_wp.shape == (1,)
    assert np.array_equal(payload_wp.numpy(), np.zeros(3, dtype=np.int32))


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_adjacency_unshared", "trimesh")
def test_face_adjacency_unshared(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """Class A: the off-edge corner of each adjacent face, elementwise after a row sort."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    unshared_tm = mesh_tm.face_adjacency_unshared.astype(np.int32)

    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_precomputed_wp = od.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    order_tm = _adjacency_order(adjacency_tm)
    order_wp = _adjacency_order(adjacency_wp.numpy())
    assert np.array_equal(unshared_precomputed_wp.numpy()[order_wp], unshared_tm[order_tm])

    # The table-free path must agree **row for row**, not merely as a set: callers pair its output
    # with a separately-computed face_adjacency, so a permutation between the two would silently
    # mis-associate every row.
    unshared_wp = od.adjacency.face_adjacency_unshared(mesh_wp.indices)
    assert np.array_equal(unshared_wp.numpy(), unshared_precomputed_wp.numpy())


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_adjacency", "igl")
@pytest.mark.parity("face_adjacency_unshared", "igl")
def test_face_adjacency_and_unshared_match_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B on both: igl's per-corner table decoded into ordito's pair list and off-edge corners.

    ``igl.triangle_triangle_adjacency`` returns ``(TT, TTi)`` in a ``(n_faces, 3)`` **corner**
    layout: ``TT[f, i]`` is the face across edge ``i`` of face ``f`` (``-1`` on a boundary) and
    ``TTi[f, i]`` is that edge's index within the neighbour. Two named transforms turn it into what
    ordito returns, and both are exact:

    1. **pairs** -- collect ``(f, TT[f, i])`` over every corner with a neighbour, sort each pair and
       deduplicate. Every interior pair appears exactly twice in igl's table (once per side), so the
       deduplicated count must equal ordito's row count, which the assert checks by shape before
       comparing values.
    2. **unshared corners** -- igl's edge ``i`` of face ``f`` runs ``(F[f, i], F[f, (i + 1) % 3])``,
       so the vertex *off* that edge is ``F[f, (i + 2) % 3]``. Reading that for both sides of a pair
       gives ordito's ``face_adjacency_unshared`` row.

    The second transform is the one worth pinning: the ``(i + 2) % 3`` offset depends on igl's edge
    numbering convention, and getting it wrong yields a table that is a *cyclic shift* of the right
    answer -- still a valid-looking set of vertex indices, and still one vertex per face.

    The three fixtures cover closed (``icosahedron``) and bounded (``half_torus``, ``hemisphere``)
    meshes, so the ``-1`` boundary entries are exercised rather than assumed away.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_wp.indices.numpy().reshape(-1, 3).astype(np.int64)

    neighbours_igl, corners_igl = igl.triangle_triangle_adjacency(faces_np)
    has_neighbour = neighbours_igl >= 0
    face_of_corner = np.broadcast_to(np.arange(faces_np.shape[0])[:, None], neighbours_igl.shape)[
        has_neighbour
    ]
    pairs_igl = np.unique(
        np.sort(np.stack([face_of_corner, neighbours_igl[has_neighbour]], axis=1), axis=1), axis=0
    )
    # The off-edge corner of face f across its edge i, keyed by (f, neighbour).
    off_edge_igl = {
        (int(f), int(neighbours_igl[f, i])): int(faces_np[f, (i + 2) % 3])
        for f in range(faces_np.shape[0])
        for i in range(3)
        if neighbours_igl[f, i] >= 0
    }

    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_wp = od.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    adjacency_np = adjacency_wp.numpy()

    assert np.array_equal(lexsort_rows(np.sort(adjacency_np, axis=1)), pairs_igl)
    assert np.array_equal(
        unshared_wp.numpy(),
        np.array(
            [
                [off_edge_igl[(int(a), int(b))], off_edge_igl[(int(b), int(a))]]
                for a, b in adjacency_np
            ],
            dtype=np.int32,
        ),
    )
    # TTi is what makes transform 2 possible; assert it is the corner index it claims to be.
    assert np.array_equal(corners_igl[has_neighbour] >= 0, np.ones(has_neighbour.sum(), dtype=bool))


def test_face_adjacency_unshared_duplicate_faces(device: str) -> None:
    """
    Two coincident triangles: the answer follows the *recorded shared edge*, not a set difference.

    The pair meets across all three of its edges, so three adjacency rows are reported and each
    one's unshared vertex is the corner off *that* edge -- ``[[2, 2], [1, 1], [0, 0]]``, which is
    what ``trimesh.graph.face_adjacency_unshared`` returns for this mesh. A "vertex of one face
    absent from the other" rule would give ``-1`` three times, since no vertex of either face is
    absent from the other; this is the only input class where the two rules diverge, and it is why
    the table-free kernel derives the shared edge from the *edge* index rather than the face pair.

    Both the tabled and table-free paths are checked, since only the former existed when this
    behaviour was first pinned.
    """
    faces_np = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(faces_wp, return_edges=True)
    unshared_tabled_wp = od.adjacency.face_adjacency_unshared(
        faces_wp, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    unshared_wp = od.adjacency.face_adjacency_unshared(faces_wp)

    assert adjacency_wp.shape == (3, 2)
    assert np.array_equal(adjacency_wp.numpy(), np.tile(np.array([0, 1], dtype=np.int32), (3, 1)))
    # The off-edge corner of {0, 1, 2}, computed independently in NumPy, for both faces of the pair.
    off_edge_np = np.array(
        [[int(3 - edge[0] - edge[1])] * 2 for edge in adjacency_edges_wp.numpy()], dtype=np.int32
    )
    assert np.array_equal(unshared_tabled_wp.numpy(), off_edge_np)
    assert np.array_equal(unshared_wp.numpy(), off_edge_np)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_adjacency_angles", "trimesh")
def test_face_adjacency_angles(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: equal after indexing both sides by their ``(f0, f1)`` pair.

    The two implementations emit adjacency rows in different orders (sort-key order here, edge-list
    order in trimesh), so the angle arrays are matched through the face pair they belong to rather
    than positionally.

    Ordito against ordito on the precomputed path as well: supplying ``face_normals`` takes the same
    route as deriving them, and that result is compared against trimesh the same way.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    angles_tm = mesh_tm.face_adjacency_angles

    adjacency_wp = od.adjacency.face_adjacency(mesh_wp.indices)
    face_normals_wp, _ = od.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    angles_wp = od.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    )
    angles_precomputed_wp = od.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp, face_normals=face_normals_wp
    )
    assert np.allclose(angles_wp.numpy(), angles_precomputed_wp.numpy(), rtol=1e-5, atol=1e-5)

    angles_tm_lookup = {
        (int(row[0]), int(row[1])): float(angles_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    adjacency_np = adjacency_wp.numpy()
    for result_wp in (angles_wp, angles_precomputed_wp):
        angles_wp_lookup = {
            (int(row[0]), int(row[1])): float(angle)
            for row, angle in zip(adjacency_np, result_wp.numpy(), strict=True)
        }
        assert angles_wp_lookup.keys() == angles_tm_lookup.keys()
        keys = list(angles_tm_lookup)
        assert np.allclose(
            [angles_wp_lookup[key] for key in keys],
            [angles_tm_lookup[key] for key in keys],
            rtol=1e-4,
            atol=5e-4,
        )


@pytest.mark.parametrize("mesh_name", ["sphere_irregular_hollow", "saddle_graded"])
@pytest.mark.parity(
    "face_adjacency_angles",
    "meshlib",
    benchmarked=False,
    reason="dihedralAngle answers one undirected edge per call, so a batched row "
    "would be a Python loop over the edge buffer and would price the loop rather "
    "than MeshLib -- the per-element rule. trimesh carries the timed "
    "row for this group. What MeshLib adds here is the sign, which no other "
    "reference for this group reports.",
)
def test_face_adjacency_angles_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (an absolute value and an edge-to-pair mapping), and the sign is a second claim.

    ``dihedralAngle`` is **signed** -- negative where the two faces form a concave surface -- where
    ordito splits the quantity in two: ``face_adjacency_angles`` is the unsigned magnitude and
    [`face_adjacency_convex`][ordito.adjacency.face_adjacency_convex] carries the side. So the
    named transform is ``abs``, and the test then spends MeshLib's extra information on the *other*
    half
    of the pair, which trimesh cannot check: positive must mean convex, edge for edge.

    Measured on ``cave_cube``, whose 48 adjacency rows split 20 convex / 4 concave / 24 flat: the
    magnitudes agree to **0.0** and the sign agrees with ``face_adjacency_convex`` on every row,
    with the four concave rows at exactly -pi/2. The fixtures are chosen for that split --
    ``icosahedron`` is convex, so its every row is positive and the sign claim would test one
    branch.

    The mapping is by face *pair* rather than by index: MeshLib keys the angle by undirected edge,
    so the loop reads ``left(e)`` and ``right(e)`` and asserts every ordito row was found, which is
    what makes a missed pair a failure rather than a silently smaller comparison.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True, n_vertices=mesh_wp.points.size
    )
    angles_wp = od.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    ).numpy()
    convex_wp = od.adjacency.face_adjacency_convex(
        mesh_wp.points, mesh_wp.indices, adjacency_wp, adjacency_edges_wp
    ).numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    topology_ml, points_ml = mesh_ml.topology, mesh_ml.points
    signed_ml: dict[tuple[int, int], float] = {}
    for undirected in range(topology_ml.undirectedEdgeSize()):
        edge_ml = mm.EdgeId(2 * undirected)
        left_ml, right_ml = topology_ml.left(edge_ml), topology_ml.right(edge_ml)
        if not (left_ml.valid() and right_ml.valid()):
            continue  # a boundary edge has one face and MeshLib reports 0 for it
        pair = (int(left_ml), int(right_ml))
        signed_ml[min(pair), max(pair)] = mm.dihedralAngle(
            topology_ml, points_ml, mm.UndirectedEdgeId(undirected)
        )

    pairs_wp = [(min(map(int, row)), max(map(int, row))) for row in adjacency_wp.numpy()]
    assert len(signed_ml) == len(pairs_wp) > 0  # non-vacuity, and the mapping is a bijection
    dihedral_ml = np.array([signed_ml[pair] for pair in pairs_wp])

    assert np.allclose(angles_wp, np.abs(dihedral_ml), rtol=1e-5, atol=1e-5)
    # The sign, which is ordito's other function: positive dihedral <-> a locally convex pair --
    # where both definitions decide by a sign. ``face_adjacency_convex`` is trimesh's: convex when
    # the unshared vertex's *projection* is below ``TOLERANCE_MERGE``, an absolute length, so a
    # pair whose projection sits inside that band reads convex whatever its dihedral. The
    # projection is the dihedral times the unshared vertex's distance from the edge, which on
    # ``saddle_graded``'s needles is ~6.6e-6, so concave pairs of dihedral up to 9.4e-5 fall in the
    # band there (208 of 13 333 rows); outside it the two signs agree on every row of every fixture.
    projections_wp = od.adjacency.face_adjacency_projections(
        mesh_wp.points, mesh_wp.indices, adjacency_wp, adjacency_edges_wp
    ).numpy()
    decided = (np.abs(dihedral_ml) > 1e-6) & (np.abs(projections_wp) >= TOLERANCE_MERGE)
    assert np.array_equal((dihedral_ml > 0.0)[decided], convex_wp[decided])
    assert int((dihedral_ml < -1e-6).sum()) > 0  # both branches present, or the sign claim is one
    assert int((dihedral_ml > 1e-6).sum()) > 0


@pytest.mark.parity("face_adjacency_projections", "trimesh")
@pytest.mark.parametrize("mesh_name", MESHES)
def test_face_adjacency_projections(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (dict index): the projection is keyed by its adjacency *pair*, not by row position.

    ordito and trimesh both return one projection per adjacent face pair, but in different row
    orders, and the value only means anything paired with its own row -- so both sides are
    indexed into a dict by ``(face_a, face_b)`` before comparing. The key-set assert is what
    makes that sound: it fails if the two disagree about *which* pairs are adjacent, which a
    value comparison over a shared key subset would hide.

    Ordito against ordito on the precomputed path as well: supplying ``face_adjacency_unshared`` and
    ``face_normals`` takes the same route as deriving them, and that result is compared against
    trimesh the same way.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    projections_tm = mesh_tm.face_adjacency_projections

    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    projections_wp = od.adjacency.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    projections_precomputed_wp = od.adjacency.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=od.adjacency.face_adjacency_unshared(
            mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
        ),
        face_normals=od.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)[0],
    )
    assert np.allclose(
        projections_wp.numpy(), projections_precomputed_wp.numpy(), rtol=1e-5, atol=1e-5
    )

    adjacency_wp_np = adjacency_wp.numpy()
    projections_tm_lookup = {
        (int(row[0]), int(row[1])): float(projections_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    for result_wp in (projections_wp, projections_precomputed_wp):
        projections_wp_np = result_wp.numpy()
        projections_wp_lookup = {
            (int(row[0]), int(row[1])): float(projections_wp_np[i])
            for i, row in enumerate(adjacency_wp_np)
        }
        assert projections_wp_lookup.keys() == projections_tm_lookup.keys()
        for key, projection_tm in projections_tm_lookup.items():
            projection_wp = projections_wp_lookup[key]
            assert np.isclose(projection_wp, projection_tm, rtol=1e-4, atol=5e-4)


def test_face_adjacency_projections_degenerate_second_face(device: str) -> None:
    """
    A degenerate second face's ``-1`` ``face_adjacency_unshared`` entry must not read out of bounds.

    ``vertices[-1]`` would otherwise be read; the row reports as ``+inf``
    instead, so it never registers as locally convex.
    """
    vertices_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2, 1, 2, 0], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency_wp = odt.as_array2d(
        wp.array(np.array([[0, 1]], dtype=np.int32), device=device), wp.int32
    )
    adjacency_edges_wp = odt.as_array2d(
        wp.array(np.array([[1, 2]], dtype=np.int32), device=device), wp.int32
    )
    unshared_wp = odt.as_array2d(
        wp.array(np.array([[0, -1]], dtype=np.int32), device=device), wp.int32
    )
    projections_wp = od.adjacency.face_adjacency_projections(
        vertices_wp,
        faces_wp,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=unshared_wp,
    )
    assert np.isposinf(projections_wp.numpy()[0])

    convex_wp = od.adjacency.face_adjacency_convex(
        vertices_wp,
        faces_wp,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=unshared_wp,
    )
    assert not bool(convex_wp.numpy()[0])


def test_face_adjacency_projections_unshared_length_mismatch_raises(device: str) -> None:
    """A caller-supplied ``face_adjacency_unshared`` shorter than ``face_adjacency`` must raise."""
    vertices_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2, 1, 2, 0], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency_wp = odt.as_array2d(
        wp.array(np.array([[0, 1]], dtype=np.int32), device=device), wp.int32
    )
    adjacency_edges_wp = odt.as_array2d(
        wp.array(np.array([[1, 2]], dtype=np.int32), device=device), wp.int32
    )
    empty_unshared_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    with pytest.raises(ValueError, match="row count must match"):
        od.adjacency.face_adjacency_projections(
            vertices_wp,
            faces_wp,
            face_adjacency=adjacency_wp,
            face_adjacency_edges=adjacency_edges_wp,
            face_adjacency_unshared=empty_unshared_wp,
        )


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_adjacency_convex", "trimesh")
def test_face_adjacency_convex(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (dict index): the per-pair convexity flag, keyed like the projections above.

    Same transform and the same reason as [`test_face_adjacency_projections`]. Non-vacuous by
    fixture choice rather than by an assert: ``icosahedron`` is convex at every edge and
    ``half_torus`` is not, so the boolean is exercised both ways across the parametrisation.

    Ordito against ordito on the precomputed path as well: supplying ``face_adjacency_unshared`` and
    ``face_normals`` gives the identical mask, which is compared against trimesh the same way.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    convex_tm = mesh_tm.face_adjacency_convex

    adjacency_wp, adjacency_edges_wp = od.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    convex_wp = od.adjacency.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    convex_precomputed_wp = od.adjacency.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=od.adjacency.face_adjacency_unshared(
            mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
        ),
        face_normals=od.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)[0],
    )
    assert np.array_equal(convex_wp.numpy(), convex_precomputed_wp.numpy())

    adjacency_wp_np = adjacency_wp.numpy()
    convex_tm_lookup = {
        (int(row[0]), int(row[1])): bool(convex_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    for result_wp in (convex_wp, convex_precomputed_wp):
        convex_wp_np = result_wp.numpy()
        convex_wp_lookup = {
            (int(row[0]), int(row[1])): bool(convex_wp_np[i])
            for i, row in enumerate(adjacency_wp_np)
        }
        assert convex_wp_lookup.keys() == convex_tm_lookup.keys()
        for key, is_convex_tm in convex_tm_lookup.items():
            assert convex_wp_lookup[key] == is_convex_tm


def _face_labels_np(faces_np: np.ndarray) -> np.ndarray:
    """
    Label the face dual graph with scipy: shared edges become entries, then a component pass.

    Written out rather than taken from a library because no reference builds the dual *and* labels
    it in one call except igl's ``facet_components`` -- which is another half of the comparison
    below, so reusing it would be comparing igl with itself. This is the same two-phase shape
    ordito's function has and the same one ``benchmarks/test_graph.py`` times on the scipy row.
    """
    edges_np = np.sort(
        np.concatenate((faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]])), axis=1
    )
    owner_np = np.tile(np.arange(faces_np.shape[0]), 3)
    order_np = np.lexsort((edges_np[:, 1], edges_np[:, 0]))
    edges_np, owner_np = edges_np[order_np], owner_np[order_np]
    shared_np = np.flatnonzero(np.all(edges_np[1:] == edges_np[:-1], axis=1))
    dual_np = sp.coo_matrix(
        (np.ones(shared_np.size, dtype=np.int8), (owner_np[shared_np], owner_np[shared_np + 1])),
        shape=(faces_np.shape[0], faces_np.shape[0]),
    ).tocsr()
    return sp.csgraph.connected_components(dual_np)[1]


def _face_labels_ml(components_ml: Iterable[mm.BitSet], n_faces: int) -> np.ndarray:
    """Decode MeshLib's vector of ``FaceBitSet`` components into a per-face label array."""
    labels_np = np.full(n_faces, -1, dtype=np.int64)
    for label, component_ml in enumerate(components_ml):
        labels_np[meshlib_bitset_to_numpy(component_ml, n_faces)] = label
    return labels_np


def _two_copies(mesh_tm: tm.Trimesh) -> tm.Trimesh:
    """``mesh_tm`` and a disjoint translated copy of it, as one mesh."""
    doubled_tm = tm.util.concatenate([mesh_tm, mesh_tm.copy().apply_translation([10.0, 0.0, 0.0])])
    assert isinstance(doubled_tm, tm.Trimesh)
    return doubled_tm


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_connected_component_labels", "igl", "meshlib", "pyvista")
@pytest.mark.parity("face_connected_component_labels_depth", "igl", "scipy")
def test_face_connected_component_labels_matches_igl_scipy_meshlib_and_pyvista(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, mesh_name: str
) -> None:
    """
    Class B against all four: the same partition under different label *names*.

    Every reference numbers components its own way, so each comparison goes through
    [`same_partition`][tests.comparisons.same_partition] (relabel by first appearance, i.e.
    [`canonical_labels`][tests.comparisons.canonical_labels]). ordito's label propagation names
    each component after a representative face, so on a two-component mesh it returns e.g.
    ``{0, 12}``.

    - **igl**: ``igl.facet_components`` numbers ``0..k-1`` in its own traversal order and returns
      ``(n_components, labels)`` -- the count **first**, which is the unpacking trap here.
    - **scipy** is a genuinely different decomposition of the work: it builds the dual graph
      explicitly (see [`_face_labels_np`]) and then labels it, where igl does both internally and
      ordito does both on the device. That is why the ``*_depth`` group -- whose benchmark rows are
      all build-included -- claims both libraries here.
    - **MeshLib**: ``getAllComponents`` with ``FaceIncidence.PerEdge``, ordito's rule (see
      [`test_face_connected_component_labels_is_per_edge`] for the input that separates it from
      ``PerVertex``). The result is a vector of ``FaceBitSet`` in MeshLib's own order, each padded
      to the face domain, its index taken as the label. Note the overload set -- a second form takes
      ``maxComponentCount`` and returns a ``(components, count)`` **tuple**, so the result's type is
      asserted by unpacking it as a plain sequence here.
    - **pyvista**: ``connectivity('all')`` writes a ``RegionId`` **cell** array numbered ``0..k-1``
      whose numbering is neither ordito's nor igl's -- measured on two disjoint spheres it labels
      the *first* component ``1``, so even a pack-by-first-appearance comparison fails and only the
      partition is shared (pyvista ships its own ``pack_labels`` for the same reason).

    Each fixture and its two-copy union are checked, because a labelling that collapsed everything
    into one component would pass on a single-component fixture alone (``cave_cube`` is two
    components already, its union four). On the union, ordito
    against ordito: compressing the pre-hooked forest changes no label.
    ``connected_components.ecl_compress`` runs between the pre-hook and the hook only from
    ``ECL_COMPRESS_FROM`` faces, which no fixture reaches, so the threshold is lowered to force it.
    """
    from ordito.kernels.algorithms import connected_components as kernel_cc

    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    faces_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)
    n_faces = faces_np.shape[0]

    n_components_igl, labels_igl = igl.facet_components(faces_np)
    components_ml = mm.getAllComponents(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), mm.MeshComponents.FaceIncidence.PerEdge
    )
    labels_pv = np.asarray(trimesh_to_pyvista(mesh_tm).connectivity("all").cell_data["RegionId"])
    labels_wp = od.adjacency.face_connected_component_labels(faces_wp).numpy()

    assert n_components_igl == len(components_ml) == np.unique(labels_wp).size
    assert same_partition(labels_wp, np.asarray(labels_igl).ravel())
    assert same_partition(labels_wp, _face_labels_np(faces_np))
    assert same_partition(labels_wp, _face_labels_ml(components_ml, n_faces))
    assert same_partition(labels_wp, labels_pv)

    # Two disjoint copies: the labelling must split them, which a constant output would not.
    doubled_tm = _two_copies(mesh_tm)
    doubled_np = doubled_tm.faces.astype(np.int64)
    doubled_wp = wp.array(
        np.ascontiguousarray(doubled_np.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=faces_wp.device,
    )
    n_doubled_igl, labels_doubled_igl = igl.facet_components(doubled_np)
    doubled_ml = mm.getAllComponents(
        mm.MeshPart(trimesh_to_meshlib(doubled_tm)), mm.MeshComponents.FaceIncidence.PerEdge
    )
    labels_doubled_pv = np.asarray(
        trimesh_to_pyvista(doubled_tm).connectivity("all").cell_data["RegionId"]
    )
    labels_doubled_wp = od.adjacency.face_connected_component_labels(doubled_wp).numpy()

    assert n_doubled_igl == 2 * n_components_igl
    assert len(doubled_ml) == 2 * len(components_ml)
    assert np.unique(labels_doubled_pv).size == 2 * n_components_igl
    assert same_partition(labels_doubled_wp, np.asarray(labels_doubled_igl).ravel())
    assert same_partition(labels_doubled_wp, _face_labels_np(doubled_np))
    assert same_partition(labels_doubled_wp, _face_labels_ml(doubled_ml, 2 * n_faces))
    assert same_partition(labels_doubled_wp, labels_doubled_pv)

    monkeypatch.setattr(kernel_cc, "ECL_COMPRESS_FROM", 0)
    compressed_wp = od.adjacency.face_connected_component_labels(doubled_wp).numpy()
    assert np.unique(labels_doubled_wp).size >= 2
    assert np.array_equal(compressed_wp, labels_doubled_wp)


def test_face_connected_component_labels_is_per_edge(device: str) -> None:
    """
    Class B against MeshLib: the pair that pins *which* incidence rule ordito implements.

    ``getAllComponents`` takes a ``FaceIncidence`` and the two settings are different operations,
    not two tunings: ``PerEdge`` connects faces sharing an edge, which is ordito's rule, and
    ``PerVertex`` connects faces sharing a single vertex. On a bowtie -- two triangles meeting at
    one vertex -- they read **2** components and **1**, and ordito reads 2. No other reference in
    this module exposes that choice, so this is the only test that can fail if the convention ever
    drifts.
    """
    bowtie_vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    bowtie_faces_np = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    bowtie_ml = mm.MeshPart(numpy_to_meshlib(bowtie_vertices_np, bowtie_faces_np))
    bowtie_wp = wp.array(
        np.ascontiguousarray(bowtie_faces_np.reshape(-1)), dtype=wp.int32, device=device
    )
    per_edge_ml = mm.getAllComponents(bowtie_ml, mm.MeshComponents.FaceIncidence.PerEdge)
    per_vertex_ml = mm.getAllComponents(bowtie_ml, mm.MeshComponents.FaceIncidence.PerVertex)
    assert (len(per_edge_ml), len(per_vertex_ml)) == (2, 1)
    labels_wp = od.adjacency.face_connected_component_labels(bowtie_wp).numpy()
    assert same_partition(labels_wp, _face_labels_ml(per_edge_ml, 2))
    assert np.unique(labels_wp).size == 2


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "sphere_irregular_hollow", "saddle_graded"]
)
@pytest.mark.parametrize("bounded", [False, True])
def test_sorted_face_edge_keys(
    request: pytest.FixtureRequest, mesh_name: str, bounded: bool
) -> None:
    """
    Class A against ``numpy.argsort(kind="stable")`` of trimesh's sorted halfedge rows, packed.

    The bounded arm packs against the vertex count and sorts only the low bits that radix needs;
    the unbounded arm packs against the pair radix and sorts all 64. Both must give the stable
    order of the same rows, so the permutation is compared exactly, and the keys against the
    packing the radix implies.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_np = np.sort(mesh_tm.edges, axis=1).astype(np.uint64)
    n_vertices = len(mesh_tm.vertices)
    radix = np.uint64(n_vertices) if bounded else np.uint64(1 << 32)
    keys_np = edges_np[:, 0] + edges_np[:, 1] * radix
    order_np = np.argsort(keys_np, kind="stable")
    assert np.unique(keys_np).size < keys_np.size

    sorted_wp, order_wp = od.adjacency.sorted_face_edge_keys(
        mesh_wp.indices, n_vertices=n_vertices if bounded else None
    )
    assert np.array_equal(order_wp.numpy(), order_np)
    assert np.array_equal(sorted_wp.numpy(), keys_np[order_np])
