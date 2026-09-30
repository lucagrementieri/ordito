"""Regression tests for ``triwarp.edges`` against Trimesh (CPU reference)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from tests.comparisons import assert_unordered_rows_equal, lexsort_rows
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
)

if TYPE_CHECKING:
    import pyvista as pv

# Not ``conftest.MESHES``: this predates that constant and has never carried ``cave_cube``.
_EDGE_MESHES = ["icosahedron", "half_torus", "hemisphere"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _faces_np_to_wp(faces_np: np.ndarray, device: str) -> wp.array[wp.int32]:
    return wp.array(faces_np.flatten().astype(np.int32), dtype=wp.int32, device=device)


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


@pytest.mark.parity("faces_to_edges", "trimesh")
def test_edges(device: str) -> None:
    """
    Class A: the directed ``3F`` edge table against ``trimesh.geometry.faces_to_edges``, in order.

    Row order is part of the claim -- face-major, three per face -- because ``edges_face`` and
    ``edges_unique_inverse`` are indexed by the same position. [`test_edges_match_igl`] is the
    class-B version against a reference that groups by corner instead.
    """
    rng = np.random.default_rng(0)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = tm.geometry.faces_to_edges(faces_np)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = tw.edges.faces_to_edges(faces_wp)
    assert np.array_equal(edges_wp.numpy(), edges_np)


@pytest.mark.parity("faces_to_edges", "igl")
def test_edges_match_igl(device: str) -> None:
    """
    Class B (row order): ``igl.oriented_facets`` emits the same ``3F`` directed pairs, permuted.

    trimesh lists all three edges of face 0, then all three of face 1, and triwarp matches that
    exactly (the test above). igl groups by *corner* instead -- every face's edge 0, then every
    face's edge 1 -- so the row order differs while the multiset does not, which is what
    [`tests.comparisons.lexsort_rows`][] canonicalises.

    The rows are compared **directed**, without sorting the pair itself: ``oriented_facets`` keeps
    the winding, so a comparison that sorted within each row would stop testing the orientation and
    pass for a table with any edge reversed.
    """
    rng = np.random.default_rng(0)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)

    edges_igl = np.asarray(igl.oriented_facets(np.ascontiguousarray(faces_np, dtype=np.int64)))
    edges_wp = tw.edges.faces_to_edges(_faces_np_to_wp(faces_np, device))

    assert edges_igl.shape == (faces_np.shape[0] * 3, 2)
    assert np.array_equal(lexsort_rows(edges_wp.numpy()), lexsort_rows(edges_igl))


@pytest.mark.parity("faces_to_edges_sorted", "trimesh")
def test_edges_sorted(device: str) -> None:
    """
    Class A: the ``sorted=True`` form against trimesh's table with each row sorted.

    The sort on the reference side *is* the definition of the keyword, not an accommodation,
    which is why this stays Class A rather than B.
    """
    rng = np.random.default_rng(1)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = tw.edges.faces_to_edges(faces_wp, sorted=True)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_edges_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.edges.faces_to_edges(faces_wp).shape == (0, 2)
    assert tw.edges.faces_to_edges(faces_wp, sorted=True).shape == (0, 2)


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
    face_idx_wp = tw.edges.edges_face(faces_wp)

    # each face f contributes edges at positions 3*f, 3*f+1, 3*f+2
    expected = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    assert np.array_equal(face_idx_wp.numpy(), expected)


def test_edges_face_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.edges.edges_face(faces_wp).shape == (0,)


# ---------------------------------------------------------------------------
# edges_unique
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_unique", "trimesh")
@pytest.mark.parity("edges_unique_auto_nv", "trimesh")
def test_edges_unique(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row-set canonicalization): the unique undirected edge set, both sides lexsorted.

    triwarp's order comes from a parallel hash and trimesh's from ``unique_rows``, so neither
    is defined; the rows are already min-first on both sides. The inverse that pairs with this
    set is checked by [`test_edges_unique_inverse`], which this sort would otherwise
    invalidate.

    The call omits ``n_vertices=``, so this is the *inferred* radix base -- the path the
    ``edges_unique_auto_nv`` benchmark group times, which is why that group's marker rides here.
    The hint is triwarp's own parameter and no reference has one, so both groups compare against
    the identical trimesh answer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]

    unique_edges_wp, _ = tw.edges.edges_unique(mesh_wp.indices)
    unique_edges_wp_np = unique_edges_wp.numpy()

    assert np.array_equal(lexsort_rows(unique_edges_wp_np), lexsort_rows(unique_edges_tm))


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_unique_inverse", "trimesh")
def test_edges_unique_inverse(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Triwarp against triwarp: ``unique_edges[inverse]`` must rebuild the sorted edge table.

    The inverse indexes into triwarp's *own* row order, which no reference shares and which
    [`test_edges_unique`] deliberately sorts away -- so the reconstruction is the only sound
    oracle. ``faces_to_edges`` carries its own trimesh comparison above, so this is not
    circular.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices)
    edges_sorted_wp = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)

    # unique_edges[inverse] must reconstruct edges_sorted
    reconstructed = unique_edges_wp.numpy()[inverse_wp.numpy()]
    assert np.array_equal(reconstructed, edges_sorted_wp.numpy())


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_unique_manifold", "potpourri3d")
def test_edges_unique_matches_potpourri3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The unique undirected edge set against geometry-central's, which lists it in its own order.

    ``benchmarks/test_edges.py`` calls this row "a timing comparison, not a parity one" because
    ``pp3d.edges`` returns geometry-central's internal halfedge ordering. That is a statement about
    *order*, and sorting dissolves it -- the sets themselves must match exactly, so this is Class B
    with a lexsort, at full tolerance rather than a weakened one.

    Both sides are canonicalised twice over: ``np.sort(..., axis=1)`` because the pair is undirected
    and the two libraries need not agree on which endpoint comes first, then a row lexsort. Uses the
    manifold fixtures for the reason the benchmark draws this row on the synthetic ``scale`` axis --
    ``pp3d.edges`` raises ``GC_SAFETY_ASSERT FAILURE ... unreferenced vertex`` on any mesh carrying
    an unreferenced vertex, which every scan mesh does.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_pp = np.asarray(
        pp3d.edges(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
        )
    )

    unique_edges_wp, _ = tw.edges.edges_unique(mesh_wp.indices, n_vertices=mesh_wp.points.size)
    assert_unordered_rows_equal(np.sort(unique_edges_wp.numpy(), axis=1), np.sort(edges_pp, axis=1))


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
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

    - the edge *sets* are compared after sorting each pair and then lexsorting the rows;
    - the inverse maps index into two differently ordered tables *and* are indexed by two different
      directed-edge orderings, so they need both ends aligned. Composing each map with its own table
      removes the first difference; the second is a fixed permutation, because igl stacks its
      directed edges by column (``[F[:,1:3]; F[:,[2,0]]; F[:,0:2]]``, so entry ``f + k * n_faces``)
      while triwarp interleaves them per face (``3f+0 = (v0,v1)``, ``3f+1 = (v1,v2)``,
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

    unique_edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices, n_vertices=n_vertices)
    unique_edges_np = unique_edges_wp.numpy()

    # The standalone entry point is what the ``edges_unique_inverse`` benchmark group times; it must
    # return the same map the combined call does.
    standalone_wp = tw.edges.edges_unique_inverse(mesh_wp.indices, n_vertices=n_vertices)
    assert np.array_equal(standalone_wp.numpy(), inverse_wp.numpy())

    assert_unordered_rows_equal(np.sort(unique_edges_np, axis=1), np.sort(unique_edges_igl, axis=1))

    # The inferred-base call (the ``edges_unique_auto_nv`` group) must return the same set, so
    # igl's answer covers both groups: the hint only picks the radix width.
    auto_edges_wp, _auto_inverse_wp = tw.edges.edges_unique(mesh_wp.indices)
    assert_unordered_rows_equal(
        np.sort(auto_edges_wp.numpy(), axis=1), np.sort(unique_edges_igl, axis=1)
    )

    # Compose each inverse map with its own table, then reorder igl's directed edges into triwarp's
    # per-face interleaving (see the docstring) so the two are indexed the same way.
    n_faces = len(faces_np)
    order_igl = np.empty(3 * n_faces, dtype=np.int64)
    order_igl[0::3] = np.arange(n_faces) + 2 * n_faces  # (v0, v1)
    order_igl[1::3] = np.arange(n_faces) + 0 * n_faces  # (v1, v2)
    order_igl[2::3] = np.arange(n_faces) + 1 * n_faces  # (v2, v0)

    resolved_wp = np.sort(unique_edges_np[inverse_wp.numpy()], axis=1)
    resolved_igl = np.sort(unique_edges_igl[inverse_igl][order_igl], axis=1)
    assert np.array_equal(resolved_wp, resolved_igl)


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_unique", "pytorch3d")
@pytest.mark.parity("edges_unique_auto_nv", "pytorch3d")
@pytest.mark.parity("edges_unique_inverse", "pytorch3d")
def test_edges_unique_and_inverse_match_pytorch3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B, twice: the edge set after a sort, and the inverse map after a remap and a permutation.

    ``Meshes.edges_packed`` orders its rows by the hash ``V * v0 + v1``, i.e. already lexsorted, so
    only triwarp's side moves for the set half. The inverse half needs two steps and both were
    recovered by measurement: pytorch3d numbers its unique edges differently, so triwarp's labels
    are remapped into pytorch3d's numbering through the sorted pair, and only then does the column
    order line up -- ``faces_packed_to_edges_packed[:, (2, 0, 1)]``, because pytorch3d builds it by
    concatenating ``[e12, e20, e01]`` and reshaping, against triwarp's ``(e01, e12, e20)``.

    Doing the remap *and* the permutation is what makes this a check of the map rather than of the
    edge set: it fails on an off-by-one, on a permuted table, or on two directed edges assigned to
    the wrong unique row.

    ``edges_unique_auto_nv`` rides here because pytorch3d has no vertex-count hint either -- it
    computes the same list whichever way triwarp is called, so one comparison covers both groups.
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
    unique_edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices)
    unique_edges_np = unique_edges_wp.numpy()
    hinted_edges_wp, hinted_inverse_wp = tw.edges.edges_unique(
        mesh_wp.indices, n_vertices=n_vertices
    )
    assert np.array_equal(hinted_edges_wp.numpy(), unique_edges_np)
    assert np.array_equal(hinted_inverse_wp.numpy(), inverse_wp.numpy())
    face_edges_np = tw.edges.edges_unique_inverse(mesh_wp.indices).numpy().reshape(-1, 3)

    assert edges_p3d.shape[0] > 0
    assert np.array_equal(inverse_wp.numpy().reshape(-1, 3), face_edges_np)
    assert_unordered_rows_equal(np.sort(unique_edges_np, axis=1), np.sort(edges_p3d, axis=1))

    slot_p3d = {tuple(row): index for index, row in enumerate(np.sort(edges_p3d, axis=1).tolist())}
    remap_np = np.array([slot_p3d[tuple(row)] for row in np.sort(unique_edges_np, axis=1).tolist()])
    assert np.array_equal(remap_np[face_edges_np], face_edges_p3d[:, [2, 0, 1]])


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_unique", "pyvista")
@pytest.mark.parity("edges_unique_auto_nv", "pyvista")
def test_edges_unique_matches_pyvista(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: ``extract_all_edges`` returns the same set as a line-cell ``PolyData``.

    The named transform is only the ordering -- VTK is free to emit its line cells in any order, so
    each row is sorted and the rows lexsorted, the same treatment igl's ``uE`` gets above. It is
    genuinely the *unique* undirected set on VTK's side too, not the ``3 * n_faces`` directed one:
    measured 30 = 30 on the icosahedron, 264 = 264 on the hemisphere, 18 = 18 on a box, with the
    sets equal element for element in each case.

    Like the trimesh comparison above, this omits ``n_vertices=`` and so covers the inferred-base
    group as well.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_pv = pyvista_edges_to_indices(
        cast("pv.PolyData", trimesh_to_pyvista(mesh_tm).extract_all_edges()), mesh_tm.vertices
    )

    unique_edges_wp, _inverse_wp = tw.edges.edges_unique(mesh_wp.indices)

    assert len(edges_pv) > 0  # non-vacuous: two empty sets would compare equal
    assert_unordered_rows_equal(np.sort(unique_edges_wp.numpy(), axis=1), edges_pv)


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
def test_edges_unique_radix_is_invariant_to_an_oversized_base(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the unique-edge table and its inverse are unchanged by any base above ``max(faces)``.

    A caller holding ``vertices`` passes ``vertices.shape[0]`` to skip the inference readback, and
    that count exceeds ``max(faces) + 1`` whenever the mesh carries unreferenced vertices.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    tight = tw.array.index_bound(mesh_wp.indices)
    edges_tight_wp, inverse_tight_wp = tw.edges.edges_unique(mesh_wp.indices, n_vertices=tight)

    for base in (tight + 1, tight + 1000):
        edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices, n_vertices=base)
        assert np.array_equal(edges_wp.numpy(), edges_tight_wp.numpy())
        assert np.array_equal(inverse_wp.numpy(), inverse_tight_wp.numpy())


def test_edges_unique_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unique_wp, inverse_wp = tw.edges.edges_unique(faces_wp)
    assert unique_wp.shape == (0, 2)
    assert inverse_wp.shape == (0,)


# ---------------------------------------------------------------------------
# edges_unique_inverse (standalone)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
def test_edges_unique_inverse_standalone(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)

    _unique_edges_wp, inverse_from_unique = tw.edges.edges_unique(mesh_wp.indices)
    inverse_standalone = tw.edges.edges_unique_inverse(mesh_wp.indices)
    assert np.array_equal(inverse_from_unique.numpy(), inverse_standalone.numpy())


# ---------------------------------------------------------------------------
# edges_unique_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_unique_length", "trimesh", "meshlib")
def test_edges_unique_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: the unique edge lengths as a sorted multiset, since the row orders differ.

    Sorting the lengths is weaker than pairing each with its edge, and that is the honest limit
    here: the pairing is pinned separately by [`test_edges_unique`] plus
    [`test_edges_unique_inverse`]. The reference lengths come from a ``float32`` copy of the
    vertices, so this is not measuring triwarp's precision against numpy's.

    MeshLib's ``edgeLengths`` is the second reference and needs the same sort -- it indexes by
    ``UndirectedEdgeId``, which is its own numbering. Its answer comes back as an
    ``UndirectedEdgeScalars`` container, which ``np.asarray`` turns into a 0-d ``object`` array
    rather than raising, so it goes through
    [`tests.conversions.meshlib_scalars_to_numpy`][] (section 6).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(
        verts_np[unique_edges_tm[:, 1]] - verts_np[unique_edges_tm[:, 0]], axis=1
    )

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = tw.edges.edges_unique_length(vertices_wp, mesh_wp.indices)
    lengths_wp_np = lengths_wp.numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    lengths_ml = meshlib_scalars_to_numpy(mm.edgeLengths(mesh_ml.topology, mesh_ml.points))

    # lengths are unordered — sort both for comparison
    assert lengths_wp_np.shape == lengths_ml.shape  # non-vacuity, and MeshLib's own edge count
    assert np.allclose(np.sort(lengths_wp_np), np.sort(lengths_tm), rtol=1e-4, atol=1e-4)
    assert np.allclose(np.sort(lengths_wp_np), np.sort(lengths_ml), rtol=1e-4, atol=1e-4)


def test_edges_unique_length_precomputed(device: str) -> None:
    rng = np.random.default_rng(7)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = points_to_warp(verts_np, device)

    unique_edges_wp, _ = tw.edges.edges_unique(faces_wp)
    lengths_via_precomputed = tw.edges.edges_unique_length(
        vertices_wp, faces_wp, unique_edges=unique_edges_wp
    )
    lengths_fresh = tw.edges.edges_unique_length(vertices_wp, faces_wp)
    assert np.allclose(lengths_via_precomputed.numpy(), lengths_fresh.numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# edges_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("edges_length", "trimesh")
def test_edges_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: per-edge lengths as a sorted multiset, matching the unique form's convention.

    Row order is in fact shared with trimesh here -- both face-major -- so this could compare
    elementwise; it sorts for consistency with the test above, and [`test_edges`] is what pins
    the order itself.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    edges_np = mesh_tm.edges
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(verts_np[edges_np[:, 1]] - verts_np[edges_np[:, 0]], axis=1)

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = tw.edges.edges_length(vertices_wp, mesh_wp.indices)

    assert np.allclose(np.sort(lengths_wp.numpy()), np.sort(lengths_tm), rtol=1e-4, atol=1e-4)


def test_edges_length_precomputed(device: str) -> None:
    rng = np.random.default_rng(8)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = points_to_warp(verts_np, device)

    edges_in_wp = tw.edges.faces_to_edges(faces_wp)
    lengths_via_precomputed = tw.edges.edges_length(vertices_wp, faces_wp, edges_in=edges_in_wp)
    lengths_fresh = tw.edges.edges_length(vertices_wp, faces_wp)
    assert np.allclose(lengths_via_precomputed.numpy(), lengths_fresh.numpy(), rtol=1e-5, atol=1e-5)


def test_precomputed_edge_tables_must_be_pairs(device: str) -> None:
    """
    Not a library comparison: the shape guard on every precomputed edge-table keyword here.

    ``_edge_lengths``' kernel reads columns 0 and 1 out of whatever rank-2 table it is handed, so a
    wider one is accepted and its remaining columns silently ignored -- an ``(m, 3)`` *face* buffer
    passed where an edge list belongs returned plausible lengths rather than raising. A rank-1
    buffer already failed at launch, so only the wide case ever needed catching, and it needed
    catching on all three keywords.

    The happy path is asserted alongside so the guard cannot be satisfied by rejecting everything.
    """
    rng = np.random.default_rng(11)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = points_to_warp(verts_np, device)
    triples_wp = twt.as_array2d(wp.array(faces_np, dtype=wp.int32, ndim=2, device=device), wp.int32)

    with pytest.raises(ValueError, match=r"unique_edges must have shape \(k, 2\)"):
        tw.edges.edges_unique_length(vertices_wp, faces_wp, unique_edges=triples_wp)
    with pytest.raises(ValueError, match=r"edges_in must have shape \(k, 2\)"):
        tw.edges.edges_length(vertices_wp, faces_wp, edges_in=triples_wp)
    with pytest.raises(ValueError, match=r"edges_sorted must have shape \(k, 2\)"):
        tw.edges.edges_unique(faces_wp, edges_sorted=triples_wp)

    # Non-vacuity: the same three keywords still accept the tables they are meant to take.
    pairs_wp = tw.edges.faces_to_edges(faces_wp, sorted=True)
    unique_wp, _inverse = tw.edges.edges_unique(faces_wp, edges_sorted=pairs_wp)
    assert tw.edges.edges_length(vertices_wp, faces_wp, edges_in=pairs_wp).size == 30
    assert (
        tw.edges.edges_unique_length(vertices_wp, faces_wp, unique_edges=unique_wp).size
        == unique_wp.shape[0]
    )


def test_edges_length_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert tw.edges.edges_length(verts_wp, faces_wp).shape == (0,)


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("mean_edge_length", "trimesh")
def test_mean_edge_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: the **per-face** average, where every face counts three edges.

    An interior edge is therefore counted twice and a boundary edge once. The closing block asserts
    this really is a different number from
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] on the open fixtures -- the
    two functions exist because libigl needs both, so a change collapsing them into one must fail.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    verts_np = mesh_tm.vertices.astype(np.float64)
    triangles_np = verts_np[mesh_tm.faces]
    per_face_np = float(np.linalg.norm(triangles_np - triangles_np[:, [1, 2, 0]], axis=2).mean())

    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    assert np.allclose(
        tw.edges.mean_edge_length(vertices_wp, mesh_wp.indices), per_face_np, rtol=1e-4, atol=1e-4
    )

    # The unique-edge average is the same number only when there is no boundary.
    unique_wp = tw.edges.mean_unique_edge_length(vertices_wp, mesh_wp.indices)
    n_boundary = int(tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape[0])
    if n_boundary == 0:
        assert np.isclose(unique_wp, per_face_np, rtol=1e-6)
    else:
        assert not np.isclose(unique_wp, per_face_np, rtol=1e-4)


@pytest.mark.parametrize("mesh_name", _EDGE_MESHES)
@pytest.mark.parity("mean_unique_edge_length", "igl", "pymeshlab", "trimesh", "meshlib")
@pytest.mark.parity("mean_edge_length", "igl")
@pytest.mark.parity("edges_length", "igl")
def test_edge_length_averages_match_their_references(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Both edge averages, each against the libigl function it is meant to reproduce.

    libigl carries two, and triwarp now exposes one for each:
    [`mean_edge_length`][triwarp.edges.mean_edge_length] is
    ``CurvatureCalculator::getAverageEdge`` (each edge once per incident face), which
    ``igl::principal_curvature`` uses to set its sphere radius;
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] is ``igl::avg_edge_length``
    (each edge once), which ``igl::heat_geodesics`` uses to set its timestep.

    They agree exactly on a closed mesh -- every edge has two incident faces there -- and diverge
    otherwise: measured **0.452405 against 0.449910** on ``half_torus`` (64 boundary edges) and
    **0.293087 against 0.291590** on ``hemisphere`` (24), while ``icosahedron`` reads 1.051462
    either way. The final block asserts that divergence, which is what stops the two being quietly
    swapped for one another.

    The per-face length *table* is Class B: igl's ``(n_faces, 3)`` uses its opposite-edge corner
    convention against triwarp's flat face-order buffer, so rows are sorted before comparing.

    MeshLib's ``averageEdgeLength`` is a fourth reference for the unique-edge average specifically,
    and it settles *which* mean it computes -- undirected edges, once each -- which is the whole
    distinction this test exists to hold. It has no counterpart for the per-face average.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = mesh_tm.faces.astype(np.int64)

    lengths_igl = np.asarray(igl.edge_lengths(vertices_np, faces_np))
    lengths_wp = tw.edges.edges_length(mesh_wp.points, mesh_wp.indices).numpy().reshape(-1, 3)
    assert np.allclose(
        np.sort(lengths_wp, axis=1), np.sort(lengths_igl, axis=1), rtol=1e-4, atol=1e-5
    )

    # Per-face average == igl's curvature-side getAverageEdge, i.e. edge_lengths().mean().
    per_face_wp = float(tw.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices))
    assert np.isclose(per_face_wp, float(lengths_igl.mean()), rtol=1e-4)

    # Unique-edge average == igl::avg_edge_length == MeshLab's avg_edge_length.
    unique_wp = float(tw.edges.mean_unique_edge_length(mesh_wp.points, mesh_wp.indices))
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
    n_boundary = int(tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape[0])
    if n_boundary == 0:
        assert np.isclose(per_face_wp, unique_wp, rtol=1e-4)
    else:
        assert not np.isclose(per_face_wp, unique_wp, rtol=1e-4)


def test_mean_edge_length_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert tw.edges.mean_edge_length(verts_wp, faces_wp) == 0.0


# --- face_edge_lengths ----------------------------------------------------------------
@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("face_edge_lengths", "igl")
def test_face_edge_lengths_are_the_opposite_edges(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A against ``igl.edge_lengths``, plus the explicit opposite-edge construction.

    Both are kept because they answer different questions. igl is the library reference and returns
    the identical ``(n_faces, 3)`` table in the same column order (measured 3.7e-08 apart on
    icosphere(2), triwarp's float32 vertex buffer being the floor); the hand-rolled stack *names*
    the convention -- corner ``k`` holds the length of the edge opposite vertex ``k`` -- which is
    the part a reader needs and which a second library agreeing cannot state.

    The column order is load-bearing and this checks it: every non-identity permutation of igl's
    columns differs from triwarp's answer by 0.045 on icosphere(2), so an order mistake fails rather
    than passing on symmetry.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths = tw.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()

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

    Triwarp against triwarp: the oracle for the row set itself is
    ``test_edges_unique_matches_trimesh``, and what this pins is that the keyword every internal
    caller now passes cannot change what those callers see. It also pins the guard the default
    still provides, on an out-of-range index that would otherwise pack into a colliding key and
    silently merge two distinct edges.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).ravel()
    _vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_vertices = int(vertices_np.shape[0])

    validated, validated_inverse = tw.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    unvalidated, unvalidated_inverse = tw.edges.edges_unique(
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
        _ = tw.edges.edges_unique(broken_wp, n_vertices=n_vertices)

    # And the inferred-bound path checks the half it can: a negative index.
    negative_np = faces_np.copy()
    negative_np[0] = -3
    negative_wp = wp.array(negative_np, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="non-negative"):
        _ = tw.edges.edges_unique(negative_wp)
