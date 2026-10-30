"""Regression tests for ``ordito.edges`` against Trimesh (CPU reference)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.comparisons import assert_unordered_rows_equal
from tests.conftest import MESHES
from tests.conversions import (
    meshlib_scalars_to_numpy,
    numpy_to_warp,
    points_to_warp,
    pyvista_edges_to_indices,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
    warp_empty,
)

if TYPE_CHECKING:
    import pyvista as pv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _faces_np_to_wp(faces_np: np.ndarray, device: str) -> wp.array[wp.int32]:
    return wp.array(faces_np.flatten().astype(np.int32), dtype=wp.int32, device=device)


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_rows", [False, True], ids=["directed", "sorted"])
@pytest.mark.parity("faces_to_edges", "trimesh")
@pytest.mark.parity("faces_to_edges_sorted", "trimesh")
def test_edges(device: str, sort_rows: bool) -> None:
    """
    Class A: the directed ``3F`` edge table against ``trimesh.geometry.faces_to_edges``, in order.

    Row order is part of the claim -- face-major, three per face -- because ``edges_face`` and
    ``edges_unique_inverse`` are indexed by the same position. [`test_edges_match_igl`] is the
    class-B version against a reference that groups by corner instead.

    The ``sorted=True`` arm compares against trimesh's table with each row sorted. The sort on the
    reference side *is* the definition of the keyword, not an accommodation, which is why that arm
    stays Class A rather than B.
    """
    rng = np.random.default_rng(int(sort_rows))
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = tm.geometry.faces_to_edges(faces_np)
    if sort_rows:
        edges_np = np.sort(edges_np, axis=1)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = od.edges.faces_to_edges(faces_wp, sorted=sort_rows)
    assert np.array_equal(edges_wp.numpy(), edges_np)


@pytest.mark.parity("faces_to_edges", "igl")
def test_edges_match_igl(device: str) -> None:
    """
    Class B (an explicit row permutation): ``igl.oriented_facets`` emits the same ``3F`` pairs.

    trimesh lists all three edges of face 0, then all three of face 1, and ordito matches that
    exactly (the test above). igl groups by *corner* instead, and numbers each edge by the corner
    opposite it: block ``k`` holds every face's edge ``(f[k+1], f[k+2])``, which is ordito's edge
    ``(k + 1) % 3``. That permutation is applied to ordito's rows and the tables compared with
    ``array_equal``, so every row is matched to its own face and slot; a ``lexsort`` of both, as
    this test used, only matched the multiset and passed for edges assigned to the wrong face.

    The rows are compared **directed**, without sorting the pair itself: ``oriented_facets`` keeps
    the winding, so a comparison that sorted within each row would stop testing the orientation and
    pass for a table with any edge reversed.
    """
    rng = np.random.default_rng(0)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)

    edges_igl = np.asarray(igl.oriented_facets(np.ascontiguousarray(faces_np, dtype=np.int64)))
    edges_wp = od.edges.faces_to_edges(_faces_np_to_wp(faces_np, device))

    assert edges_igl.shape == (faces_np.shape[0] * 3, 2)
    corner_major_wp = edges_wp.numpy().reshape(-1, 3, 2)[:, [1, 2, 0]].transpose(1, 0, 2)
    assert np.array_equal(corner_major_wp.reshape(-1, 2), edges_igl)


# ---------------------------------------------------------------------------
# edges_face
# ---------------------------------------------------------------------------


@pytest.mark.parity("edges_face", "trimesh")
def test_edges_face(device: str) -> None:
    """
    Not a library comparison: the face index of each edge row is arithmetic, not another answer.

    ``edges_face`` is ``repeat(arange(n_faces), 3)`` by construction, and that identity is what
    lets a caller map an edge row back to its face. trimesh holds the same array, but deriving
    it there would be this same expression.
    """
    rng = np.random.default_rng(3)
    n_faces = 24
    faces_np = rng.integers(0, 50, size=(n_faces, 3), dtype=np.int32)

    faces_wp = _faces_np_to_wp(faces_np, device)
    face_idx_wp = od.edges.edges_face(faces_wp)

    # each face f contributes edges at positions 3*f, 3*f+1, 3*f+2
    expected = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    assert np.array_equal(face_idx_wp.numpy(), expected)


# ---------------------------------------------------------------------------
# edges_unique
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("edges_unique", "trimesh", "pyvista")
@pytest.mark.parity("edges_unique_auto_nv", "trimesh", "pyvista")
@pytest.mark.parity("edges_unique_manifold", "potpourri3d")
def test_edges_unique(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A against trimesh as returned, against pyvista and potpourri3d up to row order.

    trimesh's ``unique_rows`` keys each min-first pair with column 0 in the low bits, so its rows
    ascend by the same ``(max, min)`` key ordito's sort produces: its table is compared with
    ``array_equal`` against ordito's rows *as returned*, untransformed, which pins the row order
    that ``quadric_decimate``, the lock keys and the new-vertex numbering read. pyvista and
    potpourri3d order their rows by their own constructions, so their tables are compared with
    ordito's as row sets (``od.equality.rows_equal``), each undirected pair written min-first. The
    inverse that pairs with this set is checked by [`test_edges_unique_inverse`].

    - **trimesh** (``unique_rows`` over the sorted edge table) and **pyvista**
      (``extract_all_edges``, a line-cell ``PolyData``) are called against the form that omits
      ``n_vertices=``, so this is the *inferred* radix base -- the path the
      ``edges_unique_auto_nv`` benchmark group times, which is why that group's markers ride here.
      The hint is ordito's own parameter and no reference has one, so both groups compare against
      the identical answer. VTK's set is genuinely the *unique* undirected one, not the
      ``3 * n_faces`` directed one: measured 30 = 30 on the icosahedron, 264 = 264 on the
      hemisphere, 18 = 18 on a box, element for element.
    - **potpourri3d** (``pp3d.edges``, geometry-central's internal halfedge ordering) is compared
      against the hinted call. ``benchmarks/test_edges.py`` calls that row "a timing comparison,
      not a parity one" because of the ordering; sorting dissolves it, and the sets must match
      exactly. The fixtures are manifold for the reason the benchmark draws this row on the
      synthetic ``scale`` axis -- ``pp3d.edges`` raises ``GC_SAFETY_ASSERT FAILURE ...
      unreferenced vertex`` on any mesh carrying an unreferenced vertex, which every scan mesh
      does.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]
    edges_pv = pyvista_edges_to_indices(
        cast("pv.PolyData", trimesh_to_pyvista(mesh_tm).extract_all_edges()), mesh_tm.vertices
    )
    edges_pp = np.asarray(
        pp3d.edges(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
        )
    )

    unique_edges_wp, _ = od.edges.edges_unique(mesh_wp.indices)
    unique_edges_wp_np = unique_edges_wp.numpy()
    hinted_edges_wp, _ = od.edges.edges_unique(mesh_wp.indices, n_vertices=mesh_wp.points.size)

    assert len(edges_pv) > 0  # non-vacuous: two empty sets would compare equal
    assert np.array_equal(unique_edges_wp_np, unique_edges_tm)
    assert od.equality.rows_equal(unique_edges_wp, _min_first_rows(edges_pv, mesh_wp.device))
    assert od.equality.rows_equal(hinted_edges_wp, _min_first_rows(edges_pp, mesh_wp.device))


def _min_first_rows(edges_np: np.ndarray, device: wp.DeviceLike) -> odt.Array2dInt32:
    """Upload a reference's undirected edges as an ``(n, 2)`` table, each pair min-first."""
    pairs_np = np.ascontiguousarray(np.sort(np.asarray(edges_np), axis=1), dtype=np.int32)
    return odt.as_array2d(wp.array(pairs_np, dtype=wp.int32, device=device), wp.int32)


def _in_ordito_edge_order(edges_np: np.ndarray) -> np.ndarray:
    """Put a reference's unique edges in ordito's order: pairs min-first, rows by ``(max, min)``."""
    pairs_np = np.sort(np.asarray(edges_np).astype(np.int32), axis=1)
    return pairs_np[np.lexsort((pairs_np[:, 0], pairs_np[:, 1]))]


@pytest.mark.parity("unique_edge_count", "trimesh")
@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parametrize("bounded", [False, True], ids=["inferred", "n_vertices"])
def test_unique_edge_count(request: pytest.FixtureRequest, mesh_name: str, bounded: bool) -> None:
    """Class A on an integer: the count equals ``len(Trimesh.edges_unique)``."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size if bounded else None
    count_wp = od.edges.unique_edge_count(mesh_wp.indices, n_vertices)
    assert len(mesh_tm.edges_unique) > 0
    assert count_wp == len(mesh_tm.edges_unique)
    assert count_wp == od.edges.edges_unique(mesh_wp.indices, n_vertices=n_vertices)[0].shape[0]


def test_unique_edge_count_validates(device: str) -> None:
    """Not a parity assert: the range check and the empty mesh."""
    faces_wp = wp.array([0, 1, 2, 2, 1, 3], dtype=wp.int32, device=device)
    assert od.edges.unique_edge_count(faces_wp) == 5
    with pytest.raises(ValueError, match="less than n_vertices"):
        od.edges.unique_edge_count(faces_wp, 3)
    assert od.edges.unique_edge_count(wp.zeros(0, dtype=wp.int32, device=device)) == 0


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("edges_unique_inverse", "trimesh")
def test_edges_unique_inverse(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Ordito against ordito: ``unique_edges[inverse]`` must rebuild the sorted edge table.

    The inverse indexes into ordito's *own* row order, which no reference shares and which
    [`test_edges_unique`] deliberately sorts away -- so the reconstruction is the only sound
    oracle. ``faces_to_edges`` carries its own trimesh comparison above, so this is not
    circular.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_edges_wp, inverse_wp = od.edges.edges_unique(mesh_wp.indices)
    edges_sorted_wp = od.edges.faces_to_edges(mesh_wp.indices, sorted=True)

    # unique_edges[inverse] must reconstruct edges_sorted
    reconstructed = unique_edges_wp.numpy()[inverse_wp.numpy()]
    assert np.array_equal(reconstructed, edges_sorted_wp.numpy())


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parametrize("from_rows", [False, True], ids=["faces", "edges_sorted"])
def test_edges_unique_counts_match_trimesh(
    request: pytest.FixtureRequest, mesh_name: str, from_rows: bool
) -> None:
    """
    Class A against trimesh: ``return_counts`` is the occurrence count of trimesh's unique rows.

    trimesh's ``unique_rows`` orders its rows as ordito does (see [`test_edges_unique`]), so the
    counts compare as returned, over both the face path and the ``edges_sorted`` path. The open
    fixture has rim edges (count 1) and interior ones (count 2), so a constant count fails; the
    rows and inverse must equal the call without counts.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_sorted = od.edges.faces_to_edges(mesh_wp.indices, sorted=True) if from_rows else None

    _, inverse_tm = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    counts_tm = np.bincount(inverse_tm)
    assert set(np.unique(counts_tm).tolist()) == {1, 2}

    unique_wp, inverse_wp, counts_wp = od.edges.edges_unique(
        mesh_wp.indices, edges_sorted, return_counts=True
    )
    plain_unique_wp, plain_inverse_wp = od.edges.edges_unique(mesh_wp.indices, edges_sorted)
    assert np.array_equal(counts_wp.numpy(), counts_tm)
    assert np.array_equal(unique_wp.numpy(), plain_unique_wp.numpy())
    assert np.array_equal(inverse_wp.numpy(), plain_inverse_wp.numpy())


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("edges_unique", "igl")
@pytest.mark.parity("edges_unique_auto_nv", "igl")
@pytest.mark.parity("edges_unique_manifold", "igl")
@pytest.mark.parity("edges_unique_inverse", "igl")
def test_edges_unique_and_inverse_match_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The unique undirected edge list and its inverse map, against ``igl.unique_edge_map``.

    igl returns ``(E, uE, EMAP, uEC, uEE)``; ``uE`` is the unique undirected list and ``EMAP`` sends
    each of the ``3 * n_faces`` directed edges to its row in ``uE``. Both libraries are free to
    order
    ``uE`` however they like, so this is Class B twice over rather than a weakened comparison:

    - the edge *sets*: igl's table is put into ordito's ``(max, min)`` row order and compared
      with ordito's rows as returned (``_in_ordito_edge_order``);
    - the inverse maps index into two differently ordered tables *and* are indexed by two different
      directed-edge orderings, so they need both ends aligned. Composing each map with its own table
      removes the first difference; the second is a fixed permutation, because igl stacks its
      directed edges by column (``[F[:,1:3]; F[:,[2,0]]; F[:,0:2]]``, so entry ``f + k * n_faces``)
      while ordito interleaves them per face (``3f+0 = (v0,v1)``, ``3f+1 = (v1,v2)``,
      ``3f+2 = (v2,v0)``). Verified exact on icosahedron, icosphere(2) and box.

    Composing and then permuting is what makes this a real check of the map rather than of the edge
    set: it would fail on an off-by-one, on a permuted table, or on two directed edges assigned to
    the wrong unique row -- none of which a bare count of distinct labels would catch.

    ``edges_unique_manifold`` is the same call on the clean synthetic meshes, which is where the
    benchmark draws it so potpourri3d can run alongside; the fixtures here are all manifold.

    ``edges_unique_auto_nv`` rides here too, and unlike the trimesh and pyvista comparisons this one
    passes ``n_vertices=`` explicitly -- so the hinted and inferred calls are asserted equal below
    rather than the marker resting on the hint being irrelevant. That equality is the whole content
    of the inferred-base group: the hint only chooses the radix width.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size
    faces_np = mesh_tm.faces.astype(np.int64)

    _e_igl, unique_edges_igl, inverse_igl = igl.unique_edge_map(faces_np)[:3]
    unique_edges_igl = np.asarray(unique_edges_igl)
    inverse_igl = np.asarray(inverse_igl).ravel()

    unique_edges_wp, inverse_wp = od.edges.edges_unique(mesh_wp.indices, n_vertices=n_vertices)
    unique_edges_np = unique_edges_wp.numpy()

    # The standalone entry point is what the ``edges_unique_inverse`` benchmark group times; it must
    # return the same map the combined call does.
    standalone_wp = od.edges.edges_unique_inverse(mesh_wp.indices, n_vertices=n_vertices)
    assert np.array_equal(standalone_wp.numpy(), inverse_wp.numpy())

    assert np.array_equal(unique_edges_np, _in_ordito_edge_order(unique_edges_igl))

    # The inferred-base call (the ``edges_unique_auto_nv`` group) must return the same set, so
    # igl's answer covers both groups: the hint only picks the radix width.
    auto_edges_wp, _auto_inverse_wp = od.edges.edges_unique(mesh_wp.indices)
    assert np.array_equal(auto_edges_wp.numpy(), _in_ordito_edge_order(unique_edges_igl))

    # Compose each inverse map with its own table, then reorder igl's directed edges into ordito's
    # per-face interleaving (see the docstring) so the two are indexed the same way.
    n_faces = len(faces_np)
    order_igl = np.empty(3 * n_faces, dtype=np.int64)
    order_igl[0::3] = np.arange(n_faces) + 2 * n_faces  # (v0, v1)
    order_igl[1::3] = np.arange(n_faces) + 0 * n_faces  # (v1, v2)
    order_igl[2::3] = np.arange(n_faces) + 1 * n_faces  # (v2, v0)

    resolved_wp = np.sort(unique_edges_np[inverse_wp.numpy()], axis=1)
    resolved_igl = np.sort(unique_edges_igl[inverse_igl][order_igl], axis=1)
    assert np.array_equal(resolved_wp, resolved_igl)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("edges_unique", "pytorch3d")
@pytest.mark.parity("edges_unique_auto_nv", "pytorch3d")
@pytest.mark.parity("edges_unique_inverse", "pytorch3d")
def test_edges_unique_and_inverse_match_pytorch3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B, twice: the edge set after a sort, and the inverse map after a remap and a permutation.

    ``Meshes.edges_packed`` orders its rows by the hash ``V * v0 + v1``, i.e. lexsorted by
    ``(min, max)``; it is put into ordito's ``(max, min)`` order and compared with ordito's rows
    as returned, so only the reference side moves for the set half. The inverse half needs two
    steps and both were recovered by measurement: pytorch3d numbers its unique edges differently,
    so ordito's labels are remapped into pytorch3d's numbering through the sorted pair, and only
    then does the column order line up -- ``faces_packed_to_edges_packed[:, (2, 0, 1)]``, because
    pytorch3d builds it by concatenating ``[e12, e20, e01]`` and reshaping, against ordito's
    ``(e01, e12, e20)``.

    Doing the remap *and* the permutation is what makes this a check of the map rather than of the
    edge set: it fails on an off-by-one, on a permuted table, or on two directed edges assigned to
    the wrong unique row.

    ``edges_unique_auto_nv`` rides here because pytorch3d has no vertex-count hint either -- it
    computes the same list whichever way ordito is called, so one comparison covers both groups.
    The hinted and inferred calls are asserted equal below rather than the marker resting on the
    hint being irrelevant; that equality *is* the whole content of the inferred-base group, since
    the hint only chooses the radix width.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_p3d = trimesh_to_pytorch3d(mesh_tm)
    edges_t = mesh_p3d.edges_packed()
    face_edges_t = mesh_p3d.faces_packed_to_edges_packed()
    assert edges_t is not None
    assert face_edges_t is not None
    edges_p3d = edges_t.numpy()
    face_edges_p3d = face_edges_t.numpy()

    n_vertices = mesh_wp.points.size
    unique_edges_wp, inverse_wp = od.edges.edges_unique(mesh_wp.indices)
    unique_edges_np = unique_edges_wp.numpy()
    hinted_edges_wp, hinted_inverse_wp = od.edges.edges_unique(
        mesh_wp.indices, n_vertices=n_vertices
    )
    assert np.array_equal(hinted_edges_wp.numpy(), unique_edges_np)
    assert np.array_equal(hinted_inverse_wp.numpy(), inverse_wp.numpy())
    face_edges_np = od.edges.edges_unique_inverse(mesh_wp.indices).numpy().reshape(-1, 3)

    assert edges_p3d.shape[0] > 0
    assert np.array_equal(inverse_wp.numpy().reshape(-1, 3), face_edges_np)
    assert np.array_equal(unique_edges_np, _in_ordito_edge_order(edges_p3d))

    slot_p3d = {tuple(row): index for index, row in enumerate(np.sort(edges_p3d, axis=1).tolist())}
    remap_np = np.array([slot_p3d[tuple(row)] for row in np.sort(unique_edges_np, axis=1).tolist()])
    assert np.array_equal(remap_np[face_edges_np], face_edges_p3d[:, [2, 0, 1]])


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_edges_unique_radix_is_invariant_to_an_oversized_base(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the unique-edge table and its inverse are unchanged by any base above ``max(faces)``.

    A caller holding ``vertices`` passes ``vertices.shape[0]`` to skip the inference readback, and
    that count exceeds ``max(faces) + 1`` whenever the mesh carries unreferenced vertices.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    tight = od.array.index_bound(mesh_wp.indices)
    edges_tight_wp, inverse_tight_wp = od.edges.edges_unique(mesh_wp.indices, n_vertices=tight)

    for base in (tight + 1, tight + 1000):
        edges_wp, inverse_wp = od.edges.edges_unique(mesh_wp.indices, n_vertices=base)
        assert np.array_equal(edges_wp.numpy(), edges_tight_wp.numpy())
        assert np.array_equal(inverse_wp.numpy(), inverse_tight_wp.numpy())


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_edges_unique_from_precomputed_sorted_edges(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the ``edges_sorted`` path returns the face path's table and inverse.

    The face path carries the library comparisons above; this pins the precomputed-rows entry
    point, which packs its own keys and unpacks each unique row from its key, at every radix it
    can pack against -- inferred (validated), supplied (validated and not), and the unbounded pair
    radix -- and its range check.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    faces = mesh_wp.indices
    edges_sorted_wp = od.edges.faces_to_edges(faces, sorted=True)
    n_vertices = mesh_wp.points.shape[0]
    edges_ref_wp, inverse_ref_wp = od.edges.edges_unique(faces, n_vertices=n_vertices)
    for n_vertices_arg, validate in (
        (None, True),
        (n_vertices, True),
        (n_vertices, False),
        (None, False),
    ):
        edges_wp, inverse_wp = od.edges.edges_unique(
            faces, edges_sorted=edges_sorted_wp, n_vertices=n_vertices_arg, validate=validate
        )
        assert np.array_equal(edges_wp.numpy(), edges_ref_wp.numpy())
        assert np.array_equal(inverse_wp.numpy(), inverse_ref_wp.numpy())

    tight = od.array.index_bound(faces)
    with pytest.raises(ValueError, match="must be less than n_vertices"):
        od.edges.edges_unique(faces, edges_sorted=edges_sorted_wp, n_vertices=tight - 1)
    with pytest.raises(ValueError, match="must be positive"):
        od.edges.edges_unique(faces, edges_sorted=edges_sorted_wp, n_vertices=0)


# ---------------------------------------------------------------------------
# edges_unique_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("edges_unique_length", "trimesh", "meshlib")
def test_edges_unique_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A against trimesh, Class B (the reference reordered) against MeshLib.

    trimesh's unique edges already come in ordito's ``(max, min)`` row order (see
    [`test_edges_unique`]), so its lengths are compared elementwise as computed. MeshLib's lengths
    are carried with their own edges into that order (``_in_ordito_edge_order``'s key) first. A
    length written at the wrong row fails either way; comparing sorted multisets, as this test once
    did, passed for any permutation. The trimesh-side lengths come from a ``float32`` copy of the
    vertices, so this is not measuring ordito's precision against numpy's.

    MeshLib's ``edgeLengths`` is the second reference and indexes by ``UndirectedEdgeId``, its own
    numbering, decoded to a vertex pair through ``org`` / ``dest`` of the edge. Its answer comes
    back as an ``UndirectedEdgeScalars`` container, which ``np.asarray`` turns into a 0-d
    ``object`` array rather than raising, so it goes through
    [`tests.conversions.meshlib_scalars_to_numpy`][].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(
        verts_np[unique_edges_tm[:, 1]] - verts_np[unique_edges_tm[:, 0]], axis=1
    )

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = od.edges.edges_unique_length(vertices_wp, mesh_wp.indices)
    lengths_wp_np = lengths_wp.numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    lengths_ml = meshlib_scalars_to_numpy(mm.edgeLengths(mesh_ml.topology, mesh_ml.points))

    topology_ml = mesh_ml.topology
    edges_ml = np.array(
        [
            [topology_ml.org(edge_ml).get(), topology_ml.dest(edge_ml).get()]
            for edge_ml in (mm.EdgeId(mm.UndirectedEdgeId(ue)) for ue in range(lengths_ml.shape[0]))
        ]
    )
    assert lengths_wp_np.shape == lengths_ml.shape  # non-vacuity, and MeshLib's own edge count
    assert np.allclose(lengths_wp_np, lengths_tm, rtol=1e-5, atol=1e-5)
    pairs_ml = np.sort(edges_ml, axis=1)
    order_ml = np.lexsort((pairs_ml[:, 0], pairs_ml[:, 1]))
    assert np.allclose(lengths_wp_np, lengths_ml[order_ml], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# edges_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("edges_length", "trimesh")
def test_edges_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """Class A: per-edge lengths elementwise, the face-major row order being trimesh's too."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    edges_np = mesh_tm.edges
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(verts_np[edges_np[:, 1]] - verts_np[edges_np[:, 0]], axis=1)

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = od.edges.edges_length(vertices_wp, mesh_wp.indices)

    assert np.allclose(lengths_wp.numpy(), lengths_tm, rtol=1e-5, atol=1e-5)


def test_precomputed_edge_tables_must_be_pairs(device: str) -> None:
    """
    Not a library comparison: the shape guard on every precomputed edge-table keyword here.

    ``_edge_lengths``' kernel reads columns 0 and 1 out of whatever rank-2 table it is handed, so a
    wider one is accepted and its remaining columns silently ignored -- an ``(m, 3)`` *face* buffer
    passed where an edge list belongs returned plausible lengths rather than raising. A rank-1
    buffer already failed at launch, so only the wide case ever needed catching, and it needed
    catching on all three keywords.

    The happy path is asserted alongside so the guard cannot be satisfied by rejecting everything,
    and on it a precomputed table must give exactly the lengths the function derives itself.
    """
    rng = np.random.default_rng(11)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = points_to_warp(verts_np, device)
    triples_wp = odt.as_array2d(wp.array(faces_np, dtype=wp.int32, ndim=2, device=device), wp.int32)

    with pytest.raises(ValueError, match=r"unique_edges must have shape \(k, 2\)"):
        od.edges.edges_unique_length(vertices_wp, faces_wp, unique_edges=triples_wp)
    with pytest.raises(ValueError, match=r"edges_in must have shape \(k, 2\)"):
        od.edges.edges_length(vertices_wp, faces_wp, edges_in=triples_wp)
    with pytest.raises(ValueError, match=r"edges_sorted must have shape \(k, 2\)"):
        od.edges.edges_unique(faces_wp, edges_sorted=triples_wp)

    # Non-vacuity: the same three keywords still accept the tables they are meant to take.
    pairs_wp = od.edges.faces_to_edges(faces_wp, sorted=True)
    unique_wp, _inverse = od.edges.edges_unique(faces_wp, edges_sorted=pairs_wp)
    assert od.edges.edges_length(vertices_wp, faces_wp, edges_in=pairs_wp).size == 30
    assert (
        od.edges.edges_unique_length(vertices_wp, faces_wp, unique_edges=unique_wp).size
        == unique_wp.shape[0]
    )

    # A precomputed table gives the lengths the function derives from ``faces`` itself, bit for bit
    # and row for row: the derived lengths are written at the unique edges' ranks off the sort.
    unique_fresh_wp, _ = od.edges.edges_unique(faces_wp)
    assert np.array_equal(
        od.edges.edges_unique_length(vertices_wp, faces_wp, unique_edges=unique_fresh_wp).numpy(),
        od.edges.edges_unique_length(vertices_wp, faces_wp).numpy(),
    )
    edges_in_wp = od.edges.faces_to_edges(faces_wp)
    assert np.allclose(
        od.edges.edges_length(vertices_wp, faces_wp, edges_in=edges_in_wp).numpy(),
        od.edges.edges_length(vertices_wp, faces_wp).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("mean_unique_edge_length", "igl", "pymeshlab", "trimesh", "meshlib")
@pytest.mark.parity("mean_edge_length", "igl", "trimesh")
@pytest.mark.parity("edges_length", "igl")
def test_edge_length_averages_match_their_references(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: both edge averages, each against the libigl function it is meant to reproduce.

    libigl carries two, and ordito now exposes one for each:
    [`mean_edge_length`][ordito.edges.mean_edge_length] is
    ``CurvatureCalculator::getAverageEdge`` (each edge once per incident face), which
    ``igl::principal_curvature`` uses to set its sphere radius;
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length] is ``igl::avg_edge_length``
    (each edge once), which ``igl::heat_geodesics`` uses to set its timestep.

    They agree exactly on a closed mesh -- every edge has two incident faces there -- and diverge
    otherwise: measured **0.452405 against 0.449910** on ``half_torus`` (64 boundary edges) and
    **0.293087 against 0.291590** on ``hemisphere`` (24), while ``icosahedron`` reads 1.051462
    either way. The final block asserts that divergence, which is what stops the two being quietly
    swapped for one another.

    The per-face length *table* is Class B: igl's ``(n_faces, 3)`` column ``k`` is the edge
    opposite corner ``k``, ``(f[k+1], f[k+2])``, which is ordito's edge ``(k + 1) % 3``; that
    column permutation is applied and the tables compared elementwise (sorting each row, as this
    test did, passed for a face's lengths assigned to the wrong edges).

    MeshLib's ``averageEdgeLength`` is a fourth reference for the unique-edge average specifically,
    and it settles *which* mean it computes -- undirected edges, once each -- which is the whole
    distinction this test exists to hold. It has no counterpart for the per-face average.

    The per-face average is also checked against its numpy definition (not a library comparison:
    trimesh exposes no such average, and its benchmark row computes this expression), every face
    counting three edges, so an interior edge counts twice and a boundary edge once.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = mesh_tm.faces.astype(np.int64)

    lengths_igl = np.asarray(igl.edge_lengths(vertices_np, faces_np))
    lengths_wp = od.edges.edges_length(mesh_wp.points, mesh_wp.indices).numpy().reshape(-1, 3)
    assert np.allclose(lengths_wp[:, [1, 2, 0]], lengths_igl, rtol=1e-5, atol=1e-5)

    # Per-face average == igl's curvature-side getAverageEdge, i.e. edge_lengths().mean().
    per_face_wp = float(od.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices))
    assert np.isclose(per_face_wp, float(lengths_igl.mean()), rtol=1e-4)
    triangles_np = vertices_np[mesh_tm.faces]
    per_face_np = float(np.linalg.norm(triangles_np - triangles_np[:, [1, 2, 0]], axis=2).mean())
    assert np.allclose(per_face_wp, per_face_np, rtol=1e-4, atol=1e-4)

    # Unique-edge average == igl::avg_edge_length == MeshLab's avg_edge_length.
    unique_wp = float(od.edges.mean_unique_edge_length(mesh_wp.points, mesh_wp.indices))
    assert np.isclose(unique_wp, float(igl.avg_edge_length(vertices_np, faces_np)), rtol=1e-4)
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_geometric_measures()
    assert np.isclose(unique_wp, float(measures_pml["avg_edge_length"]), rtol=1e-4)
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    assert np.isclose(unique_wp, mm.averageEdgeLength(mesh_ml.topology, mesh_ml.points), rtol=1e-4)

    # And the numpy dedup, which is the formula the trimesh benchmark row uses.
    edges_np = np.unique(np.sort(tm.geometry.faces_to_edges(mesh_tm.faces), axis=1), axis=0)
    unique_np = float(
        np.linalg.norm(vertices_np[edges_np[:, 1]] - vertices_np[edges_np[:, 0]], axis=1).mean()
    )
    assert np.isclose(unique_wp, unique_np, rtol=1e-4)

    # The two averages coincide only when there is no boundary.
    n_boundary = int(od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape[0])
    if n_boundary == 0:
        assert np.isclose(per_face_wp, unique_wp, rtol=1e-4)
        assert np.isclose(unique_wp, per_face_np, rtol=1e-6)
    else:
        assert not np.isclose(per_face_wp, unique_wp, rtol=1e-4)
        assert not np.isclose(unique_wp, per_face_np, rtol=1e-4)


# --- face_edge_lengths ----------------------------------------------------------------
@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("face_edge_lengths", "igl")
def test_face_edge_lengths_are_the_opposite_edges(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A against ``igl.edge_lengths``, plus the explicit opposite-edge construction.

    Both are kept because they answer different questions. igl is the library reference and returns
    the identical ``(n_faces, 3)`` table in the same column order (measured 3.7e-08 apart on
    icosphere(2), ordito's float32 vertex buffer being the floor); the hand-rolled stack *names*
    the convention -- corner ``k`` holds the length of the edge opposite vertex ``k`` -- which is
    the part a reader needs and which a second library agreeing cannot state.

    The column order is load-bearing and this checks it: every non-identity permutation of igl's
    columns differs from ordito's answer by 0.045 on icosphere(2), so an order mistake fails rather
    than passing on symmetry.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths = od.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()

    triangles = np.asarray(mesh_tm.vertices)[np.asarray(mesh_tm.faces)]
    expected = np.stack(
        [
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        ],
        axis=1,
    )
    assert np.allclose(lengths, expected, rtol=1e-5, atol=1e-5)

    lengths_igl = igl.edge_lengths(
        np.ascontiguousarray(mesh_tm.vertices), np.ascontiguousarray(mesh_tm.faces.astype(np.int64))
    )
    assert np.allclose(lengths, lengths_igl, rtol=1e-5, atol=1e-5)


def test_edges_unique_unvalidated_matches_the_validated_answer(device: str) -> None:
    """
    ``validate=False`` skips a range check, not any of the work that produces the answer.

    Ordito against ordito: the oracle for the row set itself is
    [`test_edges_unique`], and what this pins is that the keyword every internal
    caller now passes cannot change what those callers see. It also pins the guard the default
    still provides, on an out-of-range index that would otherwise pack into a colliding key and
    silently merge two distinct edges.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).ravel()
    _vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_vertices = int(vertices_np.shape[0])

    validated, validated_inverse = od.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    unvalidated, unvalidated_inverse = od.edges.edges_unique(
        faces_wp, n_vertices=n_vertices, validate=False
    )
    assert validated.shape[0] > 0
    assert np.array_equal(unvalidated.numpy(), validated.numpy())
    assert np.array_equal(unvalidated_inverse.numpy(), validated_inverse.numpy())

    # The default still catches a face index outside ``[0, n_vertices)``.
    broken_np = faces_np.copy()
    broken_np[0] = n_vertices + 5
    broken_wp = wp.array(broken_np, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="must be less than n_vertices"):
        _ = od.edges.edges_unique(broken_wp, n_vertices=n_vertices)

    # And the inferred-bound path checks the half it can: a negative index.
    negative_np = faces_np.copy()
    negative_np[0] = -3
    negative_wp = wp.array(negative_np, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="non-negative"):
        _ = od.edges.edges_unique(negative_wp)

    # The unique-edge mean reads its range flag in the same copy as its total.
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="must be less than n_vertices"):
        _ = od.edges.mean_unique_edge_length(vertices_wp, broken_wp)
    with pytest.raises(ValueError, match="non-negative"):
        _ = od.edges.mean_unique_edge_length(vertices_wp, negative_wp)
    with pytest.raises(ValueError, match="non-negative"):
        _ = od.edges.edges_unique(negative_wp, n_vertices=n_vertices)


_EMPTY_CASES: dict[
    str, tuple[Callable[[wp.array[wp.int32], wp.array[wp.vec3]], object], object]
] = {
    "faces_to_edges": (lambda faces, _v: od.edges.faces_to_edges(faces).shape, (0, 2)),
    "faces_to_edges_sorted": (
        lambda faces, _v: od.edges.faces_to_edges(faces, sorted=True).shape,
        (0, 2),
    ),
    "edges_face": (lambda faces, _v: od.edges.edges_face(faces).shape, (0,)),
    "edges_unique": (lambda faces, _v: od.edges.edges_unique(faces)[0].shape, (0, 2)),
    "edges_unique_inverse": (lambda faces, _v: od.edges.edges_unique(faces)[1].shape, (0,)),
    "edges_length": (lambda faces, v: od.edges.edges_length(v, faces).shape, (0,)),
    "mean_edge_length": (lambda faces, v: od.edges.mean_edge_length(v, faces), 0.0),
}


@pytest.mark.parametrize("case", list(_EMPTY_CASES))
def test_empty_faces(device: str, case: str) -> None:
    """Not a library comparison: every entry point answers an empty face buffer without raising."""
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    compute, expected = _EMPTY_CASES[case]
    assert compute(faces_wp, verts_wp) == expected


def test_edge_entry_points_on_their_less_travelled_paths(device: str) -> None:
    """
    Class A against trimesh, plus the guards: the paths the fixture tests do not take.

    Two triangles sharing an edge (five unique edges, all lengths distinct). The unvalidated
    ``edges_unique`` with no ``n_vertices`` packs at the pair radix rather than reducing the
    faces, and must return the validated rows exactly; ``edges_unique_length`` with an explicit
    count and ``mean_unique_edge_length(validate=False)`` must equal trimesh's unique lengths. The
    guards raise the documented messages, and every empty table (no faces, no supplied rows)
    answers empty or ``0.0`` rather than launching.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.5, 1.7, 0.4]], dtype=np.float32
    )
    faces_np = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    lengths_tm = np.sort(mesh_tm.edges_unique_length)
    assert lengths_tm.shape == (5,)
    assert np.unique(lengths_tm).size == 5

    unique_wp, inverse_wp = od.edges.edges_unique(faces_wp)
    unchecked_wp, unchecked_inverse_wp = od.edges.edges_unique(faces_wp, validate=False)
    assert np.array_equal(unchecked_wp.numpy(), unique_wp.numpy())
    assert np.array_equal(unchecked_inverse_wp.numpy(), inverse_wp.numpy())
    assert_unordered_rows_equal(unique_wp.numpy(), mesh_tm.edges_unique)

    counted_wp = od.edges.edges_unique_length(vertices_wp, faces_wp, n_vertices=4)
    assert np.allclose(np.sort(counted_wp.numpy()), lengths_tm, rtol=1e-5, atol=1e-5)
    mean_wp = od.edges.mean_unique_edge_length(vertices_wp, faces_wp, validate=False)
    assert np.isclose(mean_wp, lengths_tm.mean(), rtol=1e-5, atol=1e-5)

    negative_rows_wp = odt.as_array2d(
        wp.array([[-1, 2], [0, 1]], dtype=wp.int32, device=device), wp.int32
    )
    with pytest.raises(ValueError, match="non-negative, got a minimum of -1"):
        od.edges.edges_unique(faces_wp, edges_sorted=negative_rows_wp, n_vertices=4)
    with pytest.raises(ValueError, match="n_vertices must be positive, got 0"):
        od.edges.edges_unique(faces_wp, n_vertices=0)

    no_rows_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    no_edges_wp, no_inverse_wp = od.edges.edges_unique(
        faces_wp, edges_sorted=no_rows_wp, n_vertices=4, validate=False
    )
    assert no_edges_wp.shape == (0, 2)
    assert no_inverse_wp.shape == (0,)
    assert od.edges.edges_length(vertices_wp, faces_wp, edges_in=no_rows_wp).shape == (0,)

    no_faces_wp = warp_empty(0, wp.int32, device)
    assert od.edges.edges_unique_length(vertices_wp, no_faces_wp).shape == (0,)
    assert od.edges.face_edge_lengths(vertices_wp, no_faces_wp).shape == (0, 3)
    assert od.edges.mean_unique_edge_length(vertices_wp, no_faces_wp) == 0.0


def test_mean_edge_length_arrival_order_agrees_with_the_default(device: str) -> None:
    """
    Ordito against ordito: determinism off returns the default's mean to rounding.

    A 5 120-face sphere spans fifteen blocks, whose partial sums the flag adds by atomics instead
    of in a fixed order; the reference comparison above pins the default.
    """
    vertices_wp, faces_wp = od.creation.icosphere(subdivisions=4, device=device)
    fixed = od.edges.mean_edge_length(vertices_wp, faces_wp)
    with od.config.deterministic(False):
        arrival = od.edges.mean_edge_length(vertices_wp, faces_wp)
    assert fixed > 0.0
    assert np.isclose(fixed, arrival, rtol=1e-6, atol=0.0)


@pytest.mark.parity("edge_length_sum", "numpy")
@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_edge_length_sum_is_the_mean_times_the_edge_count(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Class A: the device total equals NumPy's sum of the per-face edge lengths, and is the mean's.

    ``mean_edge_length`` reads this very total back, so its value over ``3 * n_faces`` must be the
    mean bit for bit; the NumPy sum of the ``(3 * n_faces,)`` lengths pins the total itself.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces)
    lengths_np = np.linalg.norm(
        vertices_np[np.roll(faces_np, -1, axis=1)] - vertices_np[faces_np], axis=2
    )
    total_wp = od.edges.edge_length_sum(mesh_wp.points, mesh_wp.indices)
    assert total_wp.shape == (1,)
    assert np.allclose(total_wp.numpy()[0], lengths_np.sum(), rtol=1e-5, atol=1e-5)
    assert float(total_wp.numpy()[0]) / mesh_wp.indices.size == od.edges.mean_edge_length(
        mesh_wp.points, mesh_wp.indices
    )
    # No faces: the total is the zero the mean's ``0.0`` stands for.
    no_faces_wp = warp_empty(0, wp.int32, mesh_wp.device)
    assert od.edges.edge_length_sum(mesh_wp.points, no_faces_wp).numpy().tolist() == [0.0]
