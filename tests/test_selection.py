"""Regression tests for ``ordito.selection`` against Trimesh (CPU reference)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import open3d as o3d
import pytest
import pyvista as pv
import scipy.sparse as sp
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.sparse import csgraph
from scipy.spatial import KDTree

import ordito as od
import ordito.typing as odt
from tests.comparisons import lexsort_rows, undirected_edges
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_meshlib,
    numpy_to_meshlib_bitset,
    points_to_warp,
    pyvista_edges_to_indices,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    warp_empty,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer


def _grid_mesh(n: int = 5):
    xs, ys = np.meshgrid(np.arange(float(n)), np.arange(float(n)))
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces += [a, a + 1, a + n + 1, a, a + n + 1, a + n]
    return vertices, np.array(faces, dtype=np.int32)


def _graph_distance(faces_np: np.ndarray, n: int, seed: np.ndarray) -> np.ndarray:
    """BFS graph distance from the seed set over the undirected mesh edges (NumPy oracle)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    edges = undirected_edges(faces_np.reshape(-1, 3))
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return dijkstra(graph, indices=np.flatnonzero(seed), unweighted=True).min(axis=0)


@pytest.mark.parity("region_boundary_edges", "meshlib")
def test_region_boundary_edges(device: str):
    """
    Class A against the function this one ports, plus the hand-written rule it is meant to encode.

    ``findRegionBoundaryUndirectedEdgesInsideMesh`` is the operation
    [`region_boundary_edges`][ordito.selection.region_boundary_edges] is named after, and the
    "InsideMesh" half of that name is the whole content: it returns the edges separating the region
    from the rest of the *interior*, excluding the mesh's own boundary. Handed an all-``True``
    region it therefore returns **zero** edges, which is why it is not an oracle for
    [`boundary_edges`][ordito.boundary.boundary_edges] however much the name suggests otherwise.

    The named transform is only the decoding: MeshLib answers with an ``UndirectedEdgeBitSet``, so
    each set bit becomes an ``EdgeId`` and then an ``(org, dest)`` pair. The set-based oracle below
    is kept alongside rather than replaced -- it states the rule in one line where the reference
    only agrees with it, which is what catches the two of them sharing a misreading.
    """
    vertices_np, faces_np = _grid_mesh(5)
    n_faces = len(faces_np) // 3
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    region = np.zeros(n_faces, dtype=bool)
    region[:6] = True  # a contiguous block of faces
    region_wp = wp.array(region, dtype=wp.bool, device=device)

    edges_wp_np = od.selection.region_boundary_edges(faces_wp, region_wp).numpy()
    edges_wp_set = {tuple(sorted(int(x) for x in e)) for e in edges_wp_np}

    # Oracle: undirected edges with exactly two incident faces, exactly one in the region.
    faces = faces_np.reshape(-1, 3)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(faces):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault((int(min(a, b)), int(max(a, b))), []).append(fi)
    expected = {
        e for e, fs in edge_faces.items() if len(fs) == 2 and (region[fs[0]] ^ region[fs[1]])
    }
    assert len(expected) > 0  # non-vacuity: an empty seam would pass every assert below
    assert edges_wp_set == expected

    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    bits_ml = mn.getNumpyBitSet(
        mm.findRegionBoundaryUndirectedEdgesInsideMesh(
            mesh_ml.topology, mn.faceBitSetFromBools(cast("Buffer", region))
        )
    )
    edges_ml = {
        tuple(sorted((mesh_ml.topology.org(edge).get(), mesh_ml.topology.dest(edge).get())))
        for edge in (mm.EdgeId(mm.UndirectedEdgeId(int(i))) for i in np.flatnonzero(bits_ml))
    }
    assert edges_ml == expected

    # The name's "InsideMesh" is the content: over the whole mesh it excludes the rim entirely.
    all_faces_ml = mn.faceBitSetFromBools(cast("Buffer", np.ones(n_faces, dtype=bool)))
    assert (
        mm.findRegionBoundaryUndirectedEdgesInsideMesh(mesh_ml.topology, all_faces_ml).count() == 0
    )


@pytest.mark.parity("region_boundary_edges", "pyvista")
def test_region_boundary_edges_matches_pyvista(device: str) -> None:
    """
    Class B: VTK reaches the same seam by construction, minus the mesh's own boundary.

    pyvista has no seam filter. The route is ``extract_cells(region).extract_surface()`` followed by
    that surface's ``extract_feature_edges(boundary_edges=True)`` -- and the sub-surface's boundary
    is the seam **plus** whatever part of the mesh rim the region contains, where
    ``region_boundary_edges`` keeps only the interior half ("InsideMesh", as the MeshLib pairing
    above spells out). So the named transform is subtracting
    [`boundary_edges`][ordito.boundary.boundary_edges], and at that it is exact. On this 6x6 grid:
    an interior region gives **12 = 12** edges with nothing on the rim, and the corner region gives
    4 against pyvista's 8, where the 4 extra are exactly the rim edges it contains. The same pair on
    a curved patch cut out of ``icosphere(3)`` reads 48 = 48 and 26 against 54 (28 on the rim), and
    on the closed sphere 44 = 44.

    Both regions are asserted here because only the second exercises the transform and only the
    first shows the two agree without it -- and the pair is what rules out the subtraction hiding a
    real disagreement.

    The remap is not optional: ``extract_feature_edges`` returns a new ``PolyData`` carrying only
    the points its lines touch, renumbered, so
    [`pyvista_edges_to_indices`][tests.conversions.pyvista_edges_to_indices] keys them back by
    position. And the fixture must not come from ``trimesh.slice_plane``: on a hemisphere built that
    way the same comparison leaves 21 edges unexplained by the transform, all of it the fragmented
    rim that converter is known for.
    """
    vertices_np, faces_np = _grid_mesh(6)
    n_faces = len(faces_np) // 3
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    mesh_pv = pv.PolyData.from_regular_faces(
        vertices_np, np.ascontiguousarray(faces_np.reshape(-1, 3), dtype=np.int32)
    )
    rim = {
        tuple(sorted(int(x) for x in edge))
        for edge in od.boundary.boundary_edges(vertices_wp, faces_wp).numpy()
    }
    assert len(rim) == 20  # the 6x6 grid's own boundary, which pyvista's route picks up

    centroids_np = vertices_np[faces_np.reshape(-1, 3)].mean(axis=1)
    interior_np = (np.abs(centroids_np[:, 0] - 2.5) < 1.2) & (
        np.abs(centroids_np[:, 1] - 2.5) < 1.2
    )
    corner_np = np.zeros(n_faces, dtype=bool)
    corner_np[:6] = True  # a contiguous block against the grid's rim

    for name, region_np, touches_rim in (
        ("interior", interior_np, False),
        ("corner", corner_np, True),
    ):
        region_wp = wp.array(region_np, dtype=wp.bool, device=device)
        edges_wp = {
            tuple(sorted(int(x) for x in edge))
            for edge in od.selection.region_boundary_edges(faces_wp, region_wp).numpy()
        }

        surface_pv = mesh_pv.extract_cells(np.flatnonzero(region_np)).extract_surface(
            algorithm="dataset_surface"
        )
        edges_pv = {
            tuple(sorted(int(x) for x in edge))
            for edge in pyvista_edges_to_indices(
                surface_pv.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    non_manifold_edges=False,
                    manifold_edges=False,
                ),
                vertices_np,
            )
        }

        assert len(edges_wp) > 0, name  # non-vacuity on both sides
        assert len(edges_pv) > 0, name
        assert (edges_pv & rim != set()) is touches_rim, name  # the transform is exercised once
        assert edges_pv - rim == edges_wp, name


def _upper_half_contour(mesh_wp: wp.Mesh) -> tuple[np.ndarray, odt.Array2dInt32]:
    """Return the faces whose centroid lies above the mean height, and their oriented seam."""
    faces_wp = mesh_wp.indices
    centroids_np = od.triangles.face_centroids(mesh_wp.points, faces_wp).numpy()
    region_np = centroids_np[:, 2] > centroids_np[:, 2].mean()
    region_wp = wp.array(region_np, dtype=wp.bool, device=faces_wp.device)
    return region_np, od.selection.region_boundary_edges(faces_wp, region_wp, oriented=True)


@pytest.mark.parametrize("mesh_name", ["icosphere_coarse", "unit_box", "hemisphere"])
def test_region_boundary_edges_oriented_round_trips_through_the_fill(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: this pins two ordito entry points to each other as an inverse pair.

    ``region_boundary_edges(oriented=True)`` and ``faces_left_of_contour`` are dual, so the round
    trip must be the identity on the mask -- **exactly**, not approximately, since both sides
    are combinatorial. The oracle for the pair lives on the fill side
    (``test_faces_left_of_contour_matches_meshlib``); this test exists because the *orientation* is
    the part no reference pins, and because getting it wrong is invisible.

    That last point is the reason for the second half. With the default unoriented rows, each row's
    direction is whichever way makes it ascending, so seeds land on **both** sides and the fill
    returns the whole mesh -- a plausible-looking answer that no assertion on the fill alone would
    catch. It is asserted here so the documented ``oriented=`` requirement has a test behind it.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    n_faces = faces_wp.size // 3

    region_np, oriented_wp = _upper_half_contour(mesh_wp)
    region_wp = wp.array(region_np, dtype=wp.bool, device=faces_wp.device)
    assert np.array_equal(
        od.selection.faces_left_of_contour(faces_wp, oriented_wp).numpy(), region_np
    )
    # The oriented rows are the same undirected set as the default ones, only directed.
    unoriented_wp = od.selection.region_boundary_edges(faces_wp, region_wp)
    assert np.array_equal(
        np.sort(oriented_wp.numpy(), axis=1), np.sort(unoriented_wp.numpy(), axis=1)
    )
    assert int(od.selection.faces_left_of_contour(faces_wp, unoriented_wp).numpy().sum()) == n_faces


@pytest.mark.parametrize("oriented", [False, True])
def test_region_boundary_edges_bucketed_match_the_key_sort(
    icosphere: tuple[tm.Trimesh, wp.Mesh], oriented: bool, request: pytest.FixtureRequest
) -> None:
    """
    Ordito against ordito: the bucketed mates give the key sort's seam, row for row.

    The sort path carries the oracles (``test_region_boundary_edges`` and the pyvista
    comparison); on the bucket path only the seam's halfedges are sorted, so the rows are pinned to
    strictly ascending ``(max, min)`` keys and to the sort path's rows.
    """
    mesh_tm, mesh_wp = icosphere
    mask_np = mesh_tm.triangles_center[:, 0] < 0.2
    face_mask = wp.array(mask_np, dtype=wp.bool, device=mesh_wp.indices.device)
    n_vertices = len(mesh_tm.vertices)
    sorted_np = od.selection.region_boundary_edges(
        mesh_wp.indices, face_mask, n_vertices, oriented=oriented
    ).numpy()
    request.getfixturevalue("force_halfedge_buckets")
    bucketed_np = od.selection.region_boundary_edges(
        mesh_wp.indices, face_mask, n_vertices, oriented=oriented
    ).numpy()
    assert len(sorted_np) > 0
    _assert_edge_key_order(bucketed_np)
    assert np.array_equal(sorted_np, bucketed_np)


@pytest.mark.parametrize(
    ("entry_point", "message"),
    [
        pytest.param("region_boundary_edges", "one entry per face", id="region_boundary_edges"),
        pytest.param(
            "exclude_fully_selected_components",
            "one entry per vertex",
            id="exclude_fully_selected_components",
        ),
        pytest.param("submesh_from_face_mask", "one entry per face", id="submesh_from_face_mask"),
    ],
)
def test_masks_of_the_wrong_length_are_rejected(
    entry_point: str, message: str, device: str
) -> None:
    """Not a parity assert: pins each mask's ``ValueError`` guard against an out-of-bounds read."""
    _vertices_np, faces_np = _grid_mesh(3)
    n_faces = len(faces_np) // 3
    n_vertices = int(faces_np.max()) + 1
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    short_face_mask = wp.zeros(n_faces - 1, dtype=wp.bool, device=device)
    calls = {
        "region_boundary_edges": lambda: od.selection.region_boundary_edges(
            faces_wp, short_face_mask
        ),
        "exclude_fully_selected_components": lambda: od.selection.exclude_fully_selected_components(
            faces_wp, wp.zeros(n_vertices - 1, dtype=wp.bool, device=device), n_vertices
        ),
        "submesh_from_face_mask": lambda: od.selection.submesh_from_face_mask(
            wp.zeros(n_vertices, dtype=wp.vec3, device=device), faces_wp, short_face_mask
        ),
    }
    with pytest.raises(ValueError, match=message):
        calls[entry_point]()


def _meshlib_contour(
    topology_ml: mm.MeshTopology, contour_np: np.ndarray
) -> mm.std_vector_Id_EdgeTag:
    """Directed vertex pairs as MeshLib's ``EdgeId`` vector, as ``fillContourLeft`` takes it."""
    contour_ml = mm.std_vector_Id_EdgeTag()
    for start, end in contour_np.tolist():
        contour_ml.append(topology_ml.findEdge(mm.VertId(int(start)), mm.VertId(int(end))))
    return contour_ml


@pytest.mark.parity("faces_left_of_contour", "meshlib")
@pytest.mark.parametrize("mesh_name", ["icosphere_coarse", "torus", "unit_box"])
def test_faces_left_of_contour_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: identical masks, and identical *left* conventions, with no transform at all.

    ``fillContourLeft`` takes a vector of directed ``EdgeId`` and returns the ``FaceBitSet`` on the
    left of that walk. The convention agreement is the part worth pinning: a winding disagreement
    would show up as the exact complement, which is why the reversed contour is checked in the same
    test -- MeshLib's answer for the reversed rows is ordito's complement, not its own answer, so
    the two libraries agree about which side "left" is rather than merely partitioning the mesh the
    same way.

    Also asserted: the two sides are disjoint and cover the mesh. A contour built by
    [`region_boundary_edges`][ordito.selection.region_boundary_edges] with ``oriented=True``
    separates by construction, so anything else would be a fill leaking across the cut.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    device = faces_wp.device
    n_faces = faces_wp.size // 3

    region_np, contour_wp = _upper_half_contour(mesh_wp)
    assert 0 < int(region_np.sum()) < n_faces  # non-vacuity: both sides have faces
    assert int(contour_wp.shape[0]) > 0  # and the seam between them is not empty

    topology_ml = trimesh_to_meshlib(mesh_tm).topology
    contour_np = contour_wp.numpy()
    left_ml = meshlib_bitset_to_numpy(
        mm.fillContourLeft(topology_ml, _meshlib_contour(topology_ml, contour_np)), n_faces
    )
    left_wp = od.selection.faces_left_of_contour(faces_wp, contour_wp)
    assert np.array_equal(left_wp.numpy(), left_ml)

    reversed_wp = odt.as_array2d(
        wp.array(np.ascontiguousarray(contour_np[:, ::-1]), dtype=wp.int32, device=device), wp.int32
    )
    right_ml = meshlib_bitset_to_numpy(
        mm.fillContourLeft(topology_ml, _meshlib_contour(topology_ml, contour_np[:, ::-1])), n_faces
    )
    right_wp = od.selection.faces_left_of_contour(faces_wp, reversed_wp)
    assert np.array_equal(right_wp.numpy(), right_ml)
    assert np.array_equal(right_ml, ~left_ml)
    assert not np.any(left_wp.numpy() & right_wp.numpy())
    assert np.all(left_wp.numpy() | right_wp.numpy())


def test_faces_left_of_contour_edge_cases(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the three degenerate contours, each with a different right answer.

    An **empty** contour blocks nothing and seeds nothing, so the answer is all-``False`` rather
    than all-``True`` -- the fill is seeded by the contour, and no contour means no seed. A contour
    of rows that are **not mesh edges** is the same case reached differently. And a
    **non-separating** contour -- a homology generator on a torus, which by definition does not
    bound -- returns the *whole* mesh, because the flood fill genuinely reaches everywhere. That
    is the honest answer, and the docstring says to check the count when a contour means to close.
    """
    _, mesh_wp = torus
    faces_wp = mesh_wp.indices
    device = faces_wp.device
    n_faces = faces_wp.size // 3

    empty_wp = odt.empty_2d((0, 2), wp.int32, device=device)
    assert not np.any(od.selection.faces_left_of_contour(faces_wp, empty_wp).numpy())

    absent_wp = odt.as_array2d(
        wp.array(np.array([[0, 0], [1, 1]], dtype=np.int32), dtype=wp.int32, device=device),
        wp.int32,
    )
    assert not np.any(od.selection.faces_left_of_contour(faces_wp, absent_wp).numpy())

    loops_wp = od.homology.homology_generators(mesh_wp.points, faces_wp)
    assert len(loops_wp) == 2  # non-vacuity: genus 1, so a non-bounding cycle exists
    loop_np = loops_wp[0].numpy()
    cycle_wp = odt.as_array2d(
        wp.array(
            np.stack([loop_np, np.roll(loop_np, -1)], axis=1).astype(np.int32),
            dtype=wp.int32,
            device=device,
        ),
        wp.int32,
    )
    assert int(od.selection.faces_left_of_contour(faces_wp, cycle_wp).numpy().sum()) == n_faces

    with pytest.raises(ValueError, match=r"shape \(k, 2\)"):
        od.selection.faces_left_of_contour(
            faces_wp, odt.as_array2d(wp.zeros((2, 3), dtype=wp.int32, device=device), wp.int32)
        )


def test_faces_left_of_contour_compressed_forest(
    icosphere: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: compressing the dual-graph forest changes no face.

    The fill compresses its pre-hooked forest only from ``connected_components.ECL_COMPRESS_FROM``
    faces, which no fixture reaches, so the threshold is lowered to force it; meshlib is the oracle
    above. The contour cuts the sphere in two, so the answer is neither empty nor the whole mesh.
    """
    from ordito.kernels.algorithms import connected_components as kernel_cc

    _, mesh_wp = icosphere
    faces_wp = mesh_wp.indices
    _region_np, contour_wp = _upper_half_contour(mesh_wp)
    plain = od.selection.faces_left_of_contour(faces_wp, contour_wp).numpy()
    monkeypatch.setattr(kernel_cc, "ECL_COMPRESS_FROM", 0)
    compressed = od.selection.faces_left_of_contour(faces_wp, contour_wp).numpy()
    assert 0 < int(plain.sum()) < plain.size
    assert np.array_equal(compressed, plain)


@pytest.mark.parity(
    "exclude_fully_selected_components",
    "scipy",
    benchmarked=False,
    reason="the reference is scipy.sparse.csgraph.connected_components plus a per-component all() "
    "on the host, which is a composition rather than a bound equivalent -- no library exposes this "
    "predicate -- so a row would time a host labelling against a device pass over an edge set that "
    "ordito builds inside the call. The classification is what is comparable.",
)
@pytest.mark.parametrize("n_sub", [1, 2])
def test_exclude_fully_selected_components_matches_scipy(device: str, n_sub: int) -> None:
    """
    Class A: the same classification as ``csgraph.connected_components`` plus a per-component all().

    No library binds this predicate, so the reference is composed from one that does the hard half.
    ``csgraph.connected_components`` over the vertex adjacency labels the components independently
    of ordito's own connectivity pass, and the rule on top -- drop a component iff *every* one of
    its
    vertices is selected -- is one line, which is what makes this an oracle rather than a
    reimplementation: the part that could plausibly be wrong is the labelling, and that comes from
    scipy.

    The input is built to exercise all three branches at once, which is what keeps it non-vacuous: a
    **fully** selected component (dropped), a **partially** selected one (kept intact) and an
    **unselected** one (unchanged). Measured on the ``n_sub=1`` arm: 3 components, 16 selected
    vertices in, 4 out -- so a function that dropped nothing, dropped everything, or ignored
    component boundaries would each fail a different assert below.
    """
    meshes = (
        tm.creation.icosahedron(),
        tm.creation.icosphere(subdivisions=n_sub),
        tm.creation.box(),
    )
    parts = []
    for index, mesh_tm in enumerate(meshes):
        vertices_np = np.ascontiguousarray(mesh_tm.vertices) + np.array([5.0 * index, 0.0, 0.0])
        parts.append(
            (
                points_to_warp(vertices_np, device),
                wp.array(
                    np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
                    dtype=wp.int32,
                    device=device,
                ),
            )
        )
    vertices_wp, faces_wp = od.combine.concatenate(parts)
    n_vertices = vertices_wp.size
    offsets = np.cumsum([0, *(len(mesh_tm.vertices) for mesh_tm in meshes)])

    mask_np = np.zeros(n_vertices, dtype=bool)
    mask_np[offsets[0] : offsets[1]] = True  # component 0: fully selected
    mask_np[offsets[1] : offsets[1] + 4] = True  # component 1: partially selected
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)

    kept_wp = od.selection.exclude_fully_selected_components(faces_wp, mask_wp, n_vertices).numpy()

    faces_2d_np = faces_wp.numpy().reshape(-1, 3)
    rows_np = np.concatenate([faces_2d_np[:, 0], faces_2d_np[:, 1], faces_2d_np[:, 2]])
    columns_np = np.concatenate([faces_2d_np[:, 1], faces_2d_np[:, 2], faces_2d_np[:, 0]])
    adjacency_np = sp.coo_matrix(
        (np.ones(rows_np.size), (rows_np, columns_np)), shape=(n_vertices, n_vertices)
    )
    n_components, labels_np = csgraph.connected_components(adjacency_np, directed=False)
    kept_np = mask_np.copy()
    for component in range(n_components):
        members_np = labels_np == component
        if mask_np[members_np].all():
            kept_np[members_np] = False

    # Non-vacuity: three distinct components, and the answer is neither the input nor empty.
    assert n_components == 3
    assert int(mask_np.sum()) > int(kept_np.sum()) > 0
    assert np.array_equal(kept_wp, kept_np)


def test_exclude_fully_selected_components_compressed_forest(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: compressing the vertex forest changes no component.

    Both union-finds (from the faces, and from a supplied edge table) compress their pre-hooked
    forest only from ``connected_components.ECL_COMPRESS_FROM`` vertices, which no test mesh
    reaches, so the threshold is lowered to force it; scipy is the oracle above.
    """
    from ordito.kernels.algorithms import connected_components as kernel_cc

    ico = tm.creation.icosahedron()
    sphere = tm.creation.icosphere(subdivisions=1)
    verts, faces = od.combine.concatenate(
        [
            (
                points_to_warp(ico.vertices, device),
                wp.array(ico.faces.reshape(-1), dtype=wp.int32, device=device),
            ),
            (
                points_to_warp(sphere.vertices + np.array([5.0, 0.0, 0.0]), device),
                wp.array(sphere.faces.reshape(-1), dtype=wp.int32, device=device),
            ),
        ]
    )
    n = verts.size
    mask_np = np.zeros(n, dtype=bool)
    mask_np[: len(ico.vertices)] = True
    mask_np[len(ico.vertices) : len(ico.vertices) + 3] = True
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    edges, _inverse = od.edges.edges_unique(faces)

    def answers() -> tuple[np.ndarray, np.ndarray]:
        from_faces = od.selection.exclude_fully_selected_components(faces, mask_wp, n).numpy()
        from_edges = od.selection.exclude_fully_selected_components(
            faces, mask_wp, n, unique_edges=edges
        ).numpy()
        return from_faces, from_edges

    plain = answers()
    monkeypatch.setattr(kernel_cc, "ECL_COMPRESS_FROM", 0)
    compressed = answers()
    assert plain[0].any()
    assert np.array_equal(compressed[0], plain[0])
    assert np.array_equal(compressed[1], plain[1])


def test_face_selections_of_nothing_are_empty(device: str) -> None:
    """No face index, no face group or no vertex index: an empty submesh, packing or face list."""
    vertices_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32), dtype=wp.int32, device=device)
    empty_wp = warp_empty(0, wp.int32, device)

    submesh_vertices_wp, submesh_faces_wp = od.selection.submesh_from_face_indices(
        vertices_wp, faces_wp, empty_wp
    )
    assert submesh_vertices_wp.shape == (0,)
    assert submesh_faces_wp.shape == (0,)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = od.selection.submeshes_from_face_groups(
        vertices_wp, faces_wp, empty_wp, wp.zeros(1, dtype=wp.int32, device=device)
    )
    assert vertices_all_wp.shape == (0,)
    assert np.array_equal(vertex_offsets_wp.numpy(), [0])
    assert faces_all_wp.shape == (0,)

    face_indices_wp = od.selection.face_indices_from_vertex_indices(faces_wp, empty_wp)
    assert face_indices_wp.shape == (0,)


@pytest.mark.parity("submesh_from_face_indices", "open3d", "pyvista")
def test_submesh_from_face_indices_matches_open3d_and_pyvista(
    request: pytest.FixtureRequest,
) -> None:
    """
    Class B: the same triangles, once all three answers are lifted into the input's numbering.

    All three libraries compact the vertex buffer -- measured on ``icosphere(2)``'s upper half,
    152 of 320 faces referencing **89** of 162 vertices, and all three return 89. What differs is
    how each one tells you the mapping back:

    | | vertices returned | how the input's numbering is recovered |
    |---|---|---|
    | ordito | 89 | ``return_index=True`` returns the map |
    | pyvista ``extract_cells`` | 89 | ``vtkOriginalPointIds`` on the result's point data |
    | open3d ``select_faces_by_mask`` | 89 | **no map at all** -- matched by position |

    open3d's row is the one that needs care. With no map returned, its positions are matched against
    the input's vertex table by nearest neighbour, which is sound only because every kept position
    is a *copy* of an input position rather than a recomputation -- asserted as a residual below
    1e-06
    plus a bijection check. An exact key lookup would raise, since its tensor API stores ``float32``
    where the input is ``float64``.

    **The plan this came from recorded that open3d and pyvista "keep every vertex", and both
    halves were wrong** -- measured on a selection that happened to reference all of them. An
    interleaved face set does exactly that, which is why the fixture here is a **spatial** half:
    with every vertex still referenced, all three compactions are no-ops and the whole transform
    goes untested.

    **pymeshlab cannot be compared here at all, and the reason is its interface.** It has no
    array-valued face-selection setter -- a selection comes from a
    ``compute_selection_by_condition_per_face`` *expression* over face attributes, which can
    express a contiguous range (``fi<80``) and not an arbitrary index set. Asserted on the range
    form, so the limitation is pinned and the filter is exercised on the one shape it accepts.

    Non-vacuous: a strict face subset that leaves 73 vertices unreferenced, so neither an empty nor
    a whole-mesh answer would pass and every compaction is real.
    """
    mesh_tm, mesh_wp = request.getfixturevalue("icosphere_coarse")
    n_faces = mesh_tm.faces.shape[0]
    # A spatial half rather than every other face: an interleaved set still references every
    # vertex, so ordito's compaction would be a no-op and the transform would go untested.
    upper_np = np.asarray(mesh_tm.vertices)[np.asarray(mesh_tm.faces)].mean(axis=1)[:, 2] > 0.0
    indices_np = np.flatnonzero(upper_np).astype(np.int32)
    assert 0 < indices_np.size < n_faces  # non-vacuity: a strict subset
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    sub_vertices_wp, sub_faces_wp, vertex_map_wp = od.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, indices_wp, return_index=True
    )
    # ordito's faces, lifted back into the input's vertex numbering.
    faces_wp = vertex_map_wp.numpy()[sub_faces_wp.numpy().reshape(-1, 3)]
    assert sub_vertices_wp.size < mesh_tm.vertices.shape[0]  # it really compacted

    mask_np = np.zeros(n_faces, dtype=bool)
    mask_np[indices_np] = True
    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(trimesh_to_open3d(mesh_tm))
    selected_o3d = mesh_o3d.select_faces_by_mask(
        o3d.core.Tensor(mask_np, dtype=o3d.core.Dtype.Bool)  # pyright: ignore[reportCallIssue]  # the stub drops the device default
    )
    # open3d compacts and returns no vertex map, so its positions are matched to the input's by
    # nearest neighbour -- its tensor API stores float32, so an exact key lookup raises KeyError.
    positions_o3d = selected_o3d.vertex.positions.numpy().astype(np.float64)
    residual_o3d, original_o3d = map(
        np.asarray, KDTree(np.asarray(mesh_tm.vertices)).query(positions_o3d)
    )
    assert residual_o3d.max() < 1e-6  # every kept position is a copy, not a recomputation
    assert np.unique(original_o3d).size == original_o3d.size  # and the match is a bijection
    faces_o3d = original_o3d[selected_o3d.triangle.indices.numpy().astype(np.int64)]

    extracted_pv = trimesh_to_pyvista(mesh_tm).extract_cells(indices_np)
    original_pv = np.asarray(extracted_pv.point_data["vtkOriginalPointIds"])
    faces_pv = original_pv[np.asarray(extracted_pv.cells_dict[5])]

    # All three compact to the same vertex count, which is the referenced set.
    assert positions_o3d.shape[0] == sub_vertices_wp.size
    assert extracted_pv.n_points == sub_vertices_wp.size
    for reference_faces in (faces_o3d, faces_pv):
        assert reference_faces.shape[0] == indices_np.size
        assert np.array_equal(
            lexsort_rows(np.sort(reference_faces, axis=1)),
            lexsort_rows(np.sort(faces_wp.astype(np.int64), axis=1)),
        )

    # pymeshlab's selection is an expression, so only a contiguous range can be handed to it.
    half = n_faces // 2
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_by_condition_per_face(condselect=f"fi<{half}")
    meshset_pml.generate_from_selected_faces()
    assert int(meshset_pml.current_mesh().face_number()) == half


@pytest.mark.parity("submesh_from_face_indices", "trimesh")
@pytest.mark.parametrize(
    "face_indices_list",
    [pytest.param([0], id="single-face"), pytest.param([0, 0, 0, 5, 5, 12, 12], id="duplicated")],
)
def test_submesh_from_face_indices_matches_trimesh_on_short_lists(
    request: pytest.FixtureRequest, face_indices_list: list[int]
) -> None:
    """
    Class A: a short index list against ``trimesh.util.submesh``, positions and remapped indices.

    ``repair=False, append=False`` on the reference side is not a transform but a *disabling* of
    one: trimesh would otherwise weld and reorder, a different operation from this one. The
    duplicated arm is where the *face* count is the interesting half: a face named three times must
    appear three times -- the output length is asserted exactly -- while its vertices are shared, so
    the vertex count is bounded by the distinct faces' corners rather than fixed.
    """
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    face_indices_np = np.array(face_indices_list, dtype=np.int32)
    face_indices = wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device)
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    submesh_vertices_wp, submesh_faces_wp = od.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, face_indices
    )
    assert submesh_vertices_wp.shape == (len(submesh_tm.vertices),)
    assert submesh_vertices_wp.size <= len(np.unique(face_indices_np)) * 3
    assert submesh_faces_wp.shape == (len(face_indices_np) * 3,)
    assert np.allclose(submesh_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(submesh_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("selection", ["random_third", "all_faces"])
@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_submesh_from_face_indices_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str, selection: str
) -> None:
    """
    Class A over a random third of the faces and over every face, on three fixtures, no transform.

    Both the vertex *order* and the index remapping are compared elementwise, so this pins the
    first-occurrence compaction rule and not merely the resulting set. Selecting every face must
    reproduce the input mesh, not merely an equivalent one: the identity case is the one that pins
    the compaction's *order*, since any renumbering that is not the identity there would still give
    a valid mesh with the same geometry.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_faces = mesh_tm.faces.shape[0]
    if selection == "random_third":
        rng = np.random.default_rng(42)
        n_select = max(1, n_faces // 3)
        face_indices_np = rng.choice(n_faces, size=n_select, replace=False).astype(np.int32)
    else:
        face_indices_np = np.arange(n_faces, dtype=np.int32)
    face_indices = wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device)
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    submesh_vertices_wp, submesh_faces_wp = od.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, face_indices
    )
    assert submesh_vertices_wp.size <= mesh_tm.vertices.shape[0]
    assert submesh_faces_wp.size == face_indices_np.size * 3
    assert np.allclose(submesh_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(submesh_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "half_torus"])
def test_submeshes_from_face_groups_matches_single(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """Ordito against ordito: each group's slice equals the single-group call on that group."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = mesh_wp.points.device
    rng = np.random.default_rng(11)
    n_faces = mesh_tm.faces.shape[0]

    # Four disjoint, non-empty groups of ascending face indices (what ``split`` produces).
    order_np = rng.permutation(n_faces).astype(np.int32)
    cuts_np = np.sort(rng.choice(np.arange(1, n_faces), size=3, replace=False))
    groups_np = [np.sort(part) for part in np.split(order_np, cuts_np)]
    offsets_np = np.cumsum([0, *(len(group) for group in groups_np)]).astype(np.int32)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = od.selection.submeshes_from_face_groups(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.concatenate(groups_np).astype(np.int32), dtype=wp.int32, device=device),
        wp.array(offsets_np, dtype=wp.int32, device=device),
    )
    vertex_bounds_np = vertex_offsets_wp.list()
    face_bounds_np = offsets_np.tolist()
    assert vertex_bounds_np[-1] == vertices_all_wp.size
    assert face_bounds_np[-1] == n_faces

    for group, v_begin, v_end, f_begin, f_end in zip(
        groups_np,
        vertex_bounds_np[:-1],
        vertex_bounds_np[1:],
        face_bounds_np[:-1],
        face_bounds_np[1:],
        strict=True,
    ):
        single_vertices_wp, single_faces_wp = od.selection.submesh_from_face_indices(
            mesh_wp.points,
            mesh_wp.indices,
            wp.array(group.astype(np.int32), dtype=wp.int32, device=device),
        )
        assert np.array_equal(vertices_all_wp.numpy()[v_begin:v_end], single_vertices_wp.numpy())
        assert np.array_equal(
            faces_all_wp.numpy()[3 * f_begin : 3 * f_end], single_faces_wp.numpy()
        )

    # ...and the first group also matches the trimesh reference directly.
    submesh_tm = tm.util.submesh(mesh_tm, [groups_np[0]], repair=False, append=False)[0]
    assert np.allclose(vertices_all_wp.numpy()[: vertex_bounds_np[1]], submesh_tm.vertices)
    assert np.array_equal(
        faces_all_wp.numpy()[: 3 * face_bounds_np[1]], submesh_tm.faces.reshape(-1)
    )


def test_submeshes_from_face_groups_shared_vertex(device: str) -> None:
    """Two bodies touching at one vertex: the shared vertex is duplicated into both groups."""
    # Two triangles meeting only at vertex 2, so face adjacency keeps them separate.
    vertices_np = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [-1, 1, 0]], dtype=np.float32
    )
    faces_np = np.array([0, 1, 2, 2, 3, 4], dtype=np.int32)
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = od.selection.submeshes_from_face_groups(
        vertices_wp,
        faces_wp,
        wp.array([0, 1], dtype=wp.int32, device=device),
        wp.array([0, 1, 2], dtype=wp.int32, device=device),
    )
    # 3 + 3 vertices, not 5: vertex 2 belongs to both groups and is emitted in each.
    assert np.array_equal(vertex_offsets_wp.numpy(), [0, 3, 6])
    assert vertices_all_wp.size == 6
    assert np.array_equal(faces_all_wp.numpy(), [0, 1, 2, 0, 1, 2])
    assert np.array_equal(vertices_all_wp.numpy()[:3], vertices_np[[0, 1, 2]])
    assert np.array_equal(vertices_all_wp.numpy()[3:], vertices_np[[2, 3, 4]])


def test_submeshes_from_face_groups_unreferenced_vertices(device: str) -> None:
    """Vertices no face uses are dropped, exactly as the single-group path drops them."""
    vertices_np = np.array(
        [[0, 0, 0], [9, 9, 9], [1, 0, 0], [0, 1, 0], [8, 8, 8]], dtype=np.float32
    )
    faces_np = np.array([0, 2, 3], dtype=np.int32)
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = od.selection.submeshes_from_face_groups(
        vertices_wp,
        faces_wp,
        wp.array([0], dtype=wp.int32, device=device),
        wp.array([0, 1], dtype=wp.int32, device=device),
    )
    assert np.array_equal(vertex_offsets_wp.numpy(), [0, 3])
    assert np.array_equal(vertices_all_wp.numpy(), vertices_np[[0, 2, 3]])
    assert np.array_equal(faces_all_wp.numpy(), [0, 1, 2])


@pytest.mark.parametrize("mesh_name", ["icosphere", "half_torus"])
def test_submesh_return_index_carries_an_attribute(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: the vertex map is what it says, and it makes an attribute portable.

    The map's whole purpose is that a per-vertex field survives the extraction, so the test is that
    round trip rather than a property of the indices: gathering the *positions* through it must
    reproduce the submesh's own vertex buffer, which is the strongest available check because the
    extraction computes those positions by a different route.

    Two structural claims beside it: the map is strictly ascending (it is the sorted unique
    referenced set, which is what lets a caller treat it as a sorted lookup), and both entry points
    agree -- ``submesh_from_face_mask`` is a thin wrapper and its map must be the index form's.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(11)
    n_faces = mesh_tm.faces.shape[0]
    face_mask_np = rng.random(n_faces) > 0.4
    face_mask_np[rng.integers(0, n_faces)] = True
    device = mesh_wp.points.device
    face_mask_wp = wp.array(face_mask_np, dtype=wp.bool, device=device)

    sub_vertices_wp, sub_faces_wp, vertex_index_wp = od.selection.submesh_from_face_mask(
        mesh_wp.points, mesh_wp.indices, face_mask_wp, return_index=True
    )
    vertex_index_np = vertex_index_wp.numpy()
    assert vertex_index_np.shape == (sub_vertices_wp.size,)
    assert np.all(np.diff(vertex_index_np) > 0)  # ascending, so it is a sorted lookup

    # The round trip: an attribute gathered through the map is the submesh's own answer.
    carried_wp = od.array.gather(mesh_wp.points, vertex_index_wp)
    assert np.allclose(carried_wp.numpy(), sub_vertices_wp.numpy())

    _index_vertices_wp, _index_faces_wp, index_map_wp = od.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, od.array.flatnonzero(face_mask_wp), return_index=True
    )
    assert np.array_equal(index_map_wp.numpy(), vertex_index_np)
    assert sub_faces_wp.size > 0


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_submesh_from_face_mask(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A against trimesh *and* against the index form, which are different claims.

    The trimesh comparison says the submesh is right; the index-form comparison says the mask entry
    point agrees with the one that already has an oracle. A mask built from a coin flip is forced to
    select at least one face, so neither comparison can be satisfied by an empty answer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(7)
    n_faces = mesh_tm.faces.shape[0]
    face_mask_np = rng.choice([False, True], size=n_faces, replace=True)
    face_mask_np[rng.integers(0, n_faces)] = True
    face_indices_np = np.flatnonzero(face_mask_np).astype(np.int32)

    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    face_mask = wp.array(face_mask_np, dtype=wp.bool, device=mesh_wp.points.device)
    got_vertices_wp, got_faces_wp = od.selection.submesh_from_face_mask(
        mesh_wp.points, mesh_wp.indices, face_mask
    )
    exp_vertices_wp, exp_faces_wp = od.selection.submesh_from_face_indices(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device),
    )
    assert np.allclose(got_vertices_wp.numpy(), exp_vertices_wp.numpy())
    assert np.array_equal(got_faces_wp.numpy(), exp_faces_wp.numpy())
    assert np.allclose(got_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(got_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parity("delete_region_keep_boundary", "meshlib")
def test_delete_region_keep_boundary_matches_meshlib(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: the same survivors and the same opened rim as ``delRegionKeepBd``.

    The region is a cap -- every face whose centroid is above ``z = 0.8`` -- because a *contiguous*
    region opens exactly one rim, which is what makes the loop comparison a statement rather than a
    coincidence. Measured: both sides keep **1 148** of 1 280 faces and report **one** loop of
    **36** vertices.

    MeshLib returns the rims as directed-edge lists and ordito as vertex cycles, so the transform
    is reading a length off each; the loop *count* and its length are the shared quantity, and the
    survivors are compared as a face count. ``delRegionKeepBd`` mutates its mesh, so it gets a fresh
    one, and ``keepLoneHoles=False`` is passed explicitly since it is the parameter that decides
    whether a rim bounding nothing is reported.
    """
    mesh_tm, mesh_wp = icosphere
    n_faces = mesh_tm.faces.shape[0]
    region_np = np.asarray(mesh_tm.triangles_center)[:, 2] > 0.8
    assert 0 < int(region_np.sum()) < n_faces  # the region is neither empty nor everything

    region_wp = wp.array(region_np, dtype=wp.bool, device=mesh_wp.points.device)
    kept_vertices_wp, kept_faces_wp, new_loops = od.selection.delete_region_keep_boundary(
        mesh_wp.points, mesh_wp.indices, region_wp
    )

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    region_ml = mm.FaceBitSet(numpy_to_meshlib_bitset(region_np))
    region_ml.resize(n_faces)
    loops_ml = mm.delRegionKeepBd(mesh_ml, region_ml, False)

    assert kept_faces_wp.size // 3 == mesh_ml.topology.numValidFaces()
    assert len(new_loops) == len(loops_ml)
    assert sorted(loop.size for loop in new_loops) == sorted(len(loop_ml) for loop_ml in loops_ml)
    # The rim is a real cycle in the kept mesh, which the loop lengths alone would not say.
    kept_boundary_np = od.boundary.boundary_edges(kept_vertices_wp, kept_faces_wp).numpy()
    assert len(kept_boundary_np) == sum(loop.size for loop in new_loops)


def test_delete_region_keep_boundary_with_nothing_deleted_reports_no_rim(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: an empty deletion opens no rim, whatever rims the input has.

    The hemisphere arrives with one, which is what makes the empty answer a claim rather than a
    tautology -- its rim is a loop of the survivor, and it must not be reported as new.
    """
    mesh_tm, mesh_wp = hemisphere
    device = mesh_wp.points.device
    assert od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).size > 0
    nothing = wp.zeros(mesh_tm.faces.shape[0], dtype=wp.bool, device=device)
    kept_vertices_wp, kept_faces_wp, loops = od.selection.delete_region_keep_boundary(
        mesh_wp.points, mesh_wp.indices, nothing
    )
    assert loops == []
    assert np.array_equal(kept_faces_wp.numpy(), mesh_wp.indices.numpy())
    assert kept_vertices_wp.size == mesh_wp.points.size


def test_delete_region_keep_boundary_reports_only_new_rims(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: an input that already has a rim, which is where "new" earns its name.

    On a closed mesh every loop of the survivor is new and the distinction is invisible. Here the
    hemisphere arrives with one rim, and two regions are deleted: one **away** from that rim, which
    must report exactly one new loop, and one **touching** it, which must report the *extended* loop
    rather than nothing -- deleting a face on an existing rim grows that rim, and a caller filling
    only "new" loops would otherwise leave the extension open. That is the "every edge was already a
    boundary edge" rule the docstring states, and the case that rules out the naive "any edge".

    Both regions are grown face by face over shared edges, so each opens one *simple* rim; the
    loops are asserted simple to keep it that way. An earlier version took the first faces in index
    order, which were not contiguous: the interior region's rim was pinched, and the test passed on
    a loop the vertex walk had left full of ``(0, 0)`` edges. The pinched case is the next test.
    """
    mesh_tm, mesh_wp = hemisphere
    n_faces = mesh_tm.faces.shape[0]
    device = mesh_wp.points.device
    rim_vertices_np = od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    assert rim_vertices_np.size > 0  # the fixture has a rim to be confused by

    faces_np = mesh_tm.faces
    touches_rim_np = np.asarray(np.isin(faces_np, rim_vertices_np).any(axis=1))
    rim_distance_np = KDTree(mesh_tm.vertices[rim_vertices_np]).query(mesh_tm.triangles_center)[0]

    # A region away from the rim: one new loop, and the original rim is not reported.
    interior_np = _grown_region(mesh_tm, int(np.argmax(rim_distance_np)), 6, ~touches_rim_np)
    _kept_vertices_wp, kept_faces_wp, interior_loops = od.selection.delete_region_keep_boundary(
        mesh_wp.points, mesh_wp.indices, wp.array(interior_np, dtype=wp.bool, device=device)
    )
    assert kept_faces_wp.size // 3 == n_faces - int(interior_np.sum())
    assert len(interior_loops) == 1
    interior_loop_np = interior_loops[0].numpy()
    assert np.unique(interior_loop_np).size == interior_loop_np.size  # a simple rim
    assert np.isin(interior_loop_np, faces_np[interior_np]).all()  # and the region's own

    # A region on the rim: the loop it grows is reported, not skipped.
    on_rim_np = np.flatnonzero(np.isin(faces_np, rim_vertices_np).sum(axis=1) >= 2)
    edge_np = _grown_region(mesh_tm, int(on_rim_np[0]), 4, np.ones(n_faces, dtype=bool))
    _edge_vertices_wp, _edge_faces_wp, edge_loops = od.selection.delete_region_keep_boundary(
        mesh_wp.points, mesh_wp.indices, wp.array(edge_np, dtype=wp.bool, device=device)
    )
    assert len(edge_loops) == 1
    edge_loop_np = edge_loops[0].numpy()
    assert np.unique(edge_loop_np).size == edge_loop_np.size
    assert edge_loop_np.size > rim_vertices_np.size  # the rim grew rather than vanished


def test_delete_region_keep_boundary_reports_a_pinched_rim(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: two deleted faces that share only a corner open a rim that is new.

    The deletion's rim meets itself at the shared vertex, so it is not a set of simple loops through
    the vertices. The loops come back walked by halfedge sector, every rim edge once, and all of
    them are new: nothing reported may be an input rim edge, and every edge of both deleted faces
    must be reported. This is the case the vertex walk used to answer with fake ``(0, 0)`` edges.
    """
    mesh_tm, mesh_wp = hemisphere
    n_faces = mesh_tm.faces.shape[0]
    device = mesh_wp.points.device
    rim_vertices_np = od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    faces_np = mesh_tm.faces
    interior_faces_np = np.flatnonzero(~np.isin(faces_np, rim_vertices_np).any(axis=1))
    # Two faces meeting at exactly one vertex and no edge: their rims meet at that vertex.
    edge_neighbours = {frozenset(pair) for pair in mesh_tm.face_adjacency.tolist()}
    first = int(interior_faces_np[0])
    second = next(
        int(f)
        for f in interior_faces_np
        if np.intersect1d(faces_np[f], faces_np[first]).size == 1
        and frozenset((first, int(f))) not in edge_neighbours
    )
    pinched_np = np.zeros(n_faces, dtype=bool)
    pinched_np[[first, second]] = True

    _kept_vertices_wp, _kept_faces_wp, loops = od.selection.delete_region_keep_boundary(
        mesh_wp.points, mesh_wp.indices, wp.array(pinched_np, dtype=wp.bool, device=device)
    )
    reported_np = np.concatenate(
        [
            np.sort(np.stack([loop_np, np.roll(loop_np, -1)], axis=1), axis=1)
            for loop_np in (loop_wp.numpy() for loop_wp in loops)
        ]
    )
    deleted_edges_np = np.sort(
        np.concatenate([faces_np[pinched_np][:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 0))]),
        axis=1,
    )
    assert np.array_equal(lexsort_rows(reported_np), lexsort_rows(deleted_edges_np))


@pytest.mark.parametrize("region", ["nothing", "interior", "two_interior", "on_rim", "pinched"])
def test_delete_region_keep_boundary_is_its_packed_form_split(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], region: str
) -> None:
    """
    Ordito against ordito: the list form is the packed form, loop by loop.

    ``delete_region_keep_boundary`` carries the MeshLib comparison above; this pins
    ``delete_region_keep_boundary_with_offsets`` to it over every shape of answer: no rim, one,
    two, the input's rim grown (so a loop is classified *and* kept), and a pinched rim.
    """
    mesh_tm, mesh_wp = hemisphere
    n_faces = mesh_tm.faces.shape[0]
    device = mesh_wp.points.device
    rim_vertices_np = od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    touches_rim_np = np.asarray(np.isin(mesh_tm.faces, rim_vertices_np).any(axis=1))
    interior_faces_np = np.flatnonzero(~touches_rim_np)
    mask_np = np.zeros(n_faces, dtype=bool)
    if region in ("interior", "two_interior"):
        mask_np |= _grown_region(mesh_tm, int(interior_faces_np[0]), 5, ~touches_rim_np)
    if region == "two_interior":
        far = int(interior_faces_np[-1])
        mask_np |= _grown_region(mesh_tm, far, 5, ~touches_rim_np)
    if region == "on_rim":
        on_rim_np = np.flatnonzero(np.isin(mesh_tm.faces, rim_vertices_np).sum(axis=1) >= 2)
        mask_np |= _grown_region(mesh_tm, int(on_rim_np[0]), 4, np.ones(n_faces, dtype=bool))
    if region == "pinched":
        first = int(interior_faces_np[0])
        adjacent = {frozenset(pair) for pair in mesh_tm.face_adjacency.tolist()}
        second = next(
            int(f)
            for f in interior_faces_np
            if np.intersect1d(mesh_tm.faces[f], mesh_tm.faces[first]).size == 1
            and frozenset((first, int(f))) not in adjacent
        )
        mask_np[[first, second]] = True
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)

    vertices_wp, faces_wp, loops_wp = od.selection.delete_region_keep_boundary(
        mesh_wp.points, mesh_wp.indices, mask_wp
    )
    packed = od.selection.delete_region_keep_boundary_with_offsets(
        mesh_wp.points, mesh_wp.indices, mask_wp
    )
    packed_vertices_wp, packed_faces_wp, flat_wp, offsets_wp = packed

    expected_loops = {"nothing": 0, "interior": 1, "two_interior": 2, "on_rim": 1}
    if region in expected_loops:
        assert len(loops_wp) == expected_loops[region]
    else:
        assert len(loops_wp) >= 1
    assert np.array_equal(packed_vertices_wp.numpy(), vertices_wp.numpy())
    assert np.array_equal(packed_faces_wp.numpy(), faces_wp.numpy())
    offsets_np = offsets_wp.numpy()
    assert offsets_np.size == len(loops_wp) + 1
    assert offsets_np[0] == 0
    assert offsets_np[-1] == flat_wp.size
    flat_np = flat_wp.numpy()
    for i, loop_wp in enumerate(loops_wp):
        assert np.array_equal(loop_wp.numpy(), flat_np[offsets_np[i] : offsets_np[i + 1]])


def _grown_region(
    mesh_tm: tm.Trimesh, seed_face: int, size: int, allowed_np: np.ndarray
) -> np.ndarray:
    """Face mask of ``size`` faces grown breadth-first from ``seed_face`` over shared edges."""
    neighbours: dict[int, list[int]] = {}
    for a, b in mesh_tm.face_adjacency.tolist():
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    region, frontier = [seed_face], [seed_face]
    while frontier and len(region) < size:
        face = frontier.pop(0)
        for other in neighbours.get(face, []):
            if allowed_np[other] and other not in region and len(region) < size:
                region.append(other)
                frontier.append(other)
    assert len(region) == size  # non-vacuity: the region really has the size the test assumes
    mask_np = np.zeros(mesh_tm.faces.shape[0], dtype=bool)
    mask_np[region] = True
    return mask_np


def _vertex_selection(mesh_tm: tm.Trimesh, seed: int, fraction: int) -> np.ndarray:
    """
    Draw a random vertex selection that is guaranteed to contain at least one *whole* face.

    ``face_mode="all"`` keeps a face only when all three of its corners are selected, which a
    sparse random draw essentially never produces: at ``n // 4`` of the icosahedron's 12 vertices
    and ``n // 5`` of the hemisphere's 97, the measured answer was **0 faces**, so both sides of
    the comparison were empty and the ``all`` half of the parametrisation asserted nothing. Seeding
    the draw with the corners of every fourth face fixes that without giving up the random part.
    """
    rng = np.random.default_rng(seed)
    n_vertices = mesh_tm.vertices.shape[0]
    random_np = rng.choice(n_vertices, size=max(3, n_vertices // fraction), replace=False)
    whole_faces_np = mesh_tm.faces[:: max(1, mesh_tm.faces.shape[0] // 4)].reshape(-1)
    return np.union1d(random_np, whole_faces_np).astype(np.int32)


def _face_indices_from_vertex_indices_np(
    faces_np: np.ndarray, vertex_indices_np: np.ndarray, *, face_mode: str
) -> np.ndarray:
    vertex_hit = np.isin(faces_np, vertex_indices_np)
    if face_mode == "all":
        face_hit = np.all(vertex_hit, axis=1)
    else:
        face_hit = np.any(vertex_hit, axis=1)
    return np.flatnonzero(face_hit).astype(np.int32)


# ---------------------------------------------------------------------------
# Vertex-selection morphology + region utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("face_mode", ["all", "any"])
def test_submesh_from_vertex_indices(
    request: pytest.FixtureRequest, face_mode: Literal["all", "any"]
) -> None:
    """
    Class B: trimesh's submesh of the faces a numpy predicate picks, over both ``face_mode`` values.

    The named transform is on the reference side: trimesh has no vertex-driven submesh, so the face
    set is derived here by ``_face_indices_from_vertex_indices_np`` and handed to it. That helper
    is itself the oracle in [`test_face_indices_from_vertex_indices`], so it is not assumed correct.
    """
    mesh_tm, mesh_wp = request.getfixturevalue("half_torus")
    rng = np.random.default_rng(13)
    n_vertices = mesh_tm.vertices.shape[0]
    vertex_indices_np = rng.choice(n_vertices, size=max(3, n_vertices // 5), replace=False).astype(
        np.int32
    )
    vertex_indices = wp.array(vertex_indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    face_indices_np = _face_indices_from_vertex_indices_np(
        mesh_tm.faces, vertex_indices_np, face_mode=face_mode
    )
    # half_torus is dense enough that "all" selects 8 faces here; keep it that way.
    assert face_indices_np.size > 0
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    got_vertices_wp, got_faces_wp = od.selection.submesh_from_vertex_indices(
        mesh_wp.points, mesh_wp.indices, vertex_indices, face_mode=face_mode
    )
    assert np.allclose(got_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(got_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("face_mode", ["all", "any"])
def test_submesh_from_vertex_mask(
    request: pytest.FixtureRequest, face_mode: Literal["all", "any"]
) -> None:
    """
    The mask form and the index form of the same selection agree, in both ``face_mode`` branches.

    Not a reference comparison -- it pins the two entry points to each other, so the oracle for the
    selection rule itself is [`test_face_indices_from_vertex_indices`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue("hemisphere")
    selected = _vertex_selection(mesh_tm, seed=17, fraction=5)
    vertex_mask_np = np.zeros(mesh_tm.vertices.shape[0], dtype=bool)
    vertex_mask_np[selected] = True
    vertex_mask = wp.array(vertex_mask_np, dtype=wp.bool, device=mesh_wp.points.device)

    got_vertices_wp, got_faces_wp = od.selection.submesh_from_vertex_mask(
        mesh_wp.points, mesh_wp.indices, vertex_mask, face_mode=face_mode
    )
    exp_vertices_wp, exp_faces_wp = od.selection.submesh_from_vertex_indices(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(selected, dtype=wp.int32, device=mesh_wp.points.device),
        face_mode=face_mode,
    )
    # Two empty submeshes compare equal, which is what the "all" branch used to do.
    assert exp_faces_wp.size > 0
    assert np.allclose(got_vertices_wp.numpy(), exp_vertices_wp.numpy())
    assert np.array_equal(got_faces_wp.numpy(), exp_faces_wp.numpy())


@pytest.mark.parity("expand_vertex_mask", "pymeshlab")
def test_expand_vertex_mask_matches_pymeshlab_dilatation(device: str):
    """
    Class B (face-based morphology): MeshLab dilates a *face* selection, not a vertex one.

    Neither trimesh nor open3d nor libigl has selection morphology. MeshLab does, but it dilates the
    *face* set, so the composition that lines up with a vertex-mask hop is:

        seed one vertex -> transfer to faces (``inclusive=False``, any selected vertex)
        -> k x Dilate Selection -> read the *vertex* selection back

    which is exactly ``expand_vertex_mask(seed, k)``. Verified to the element on a 9x9 grid for
    ``k = 1..4``. Note ``inclusive=False`` is load-bearing: the default ``True`` selects only faces
    whose *every* vertex is selected, which on a single-vertex seed is no faces at all and clears
    the selection.

    **Erosion does not map the same way** and is deliberately not checked here: MeshLab's Erode
    Selection removes a face when any of its vertices is on the boundary of the selection, so
    reading the vertex selection back gives the vertices of the surviving *faces* -- measured at
    51 / 39 / 25 vertices after 1 / 2 / 3 erosions where ``shrink_vertex_mask`` gives 19 / 7 / 1.
    Different operation, not a discrepancy. ``test_vertex_morphology_on_an_irregular_mesh`` keeps
    the graph-distance oracle, ``test_expand_and_shrink_vertex_mask_match_meshlib`` the library.
    """
    vertices_np, faces_np = _grid_mesh(9)
    n_vertices = len(vertices_np)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    seed_np = np.zeros(n_vertices, dtype=bool)
    centre = n_vertices // 2
    seed_np[centre] = True
    seed_wp = wp.array(seed_np, dtype=wp.bool, device=device)

    mesh_tm = tm.Trimesh(vertices_np, faces_np.reshape(-1, 3), process=False)
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_by_condition_per_vertex(condselect=f"(vi == {centre})")
    meshset_pml.compute_selection_transfer_vertex_to_face(inclusive=False)

    for hops in (1, 2, 3, 4):
        meshset_pml.apply_selection_dilatation()
        assert np.array_equal(
            od.selection.expand_vertex_mask(faces_wp, seed_wp, hops).numpy(),
            meshset_pml.current_mesh().vertex_selection_array(),
        )


@pytest.mark.parametrize("hops", [1, 2, 3])
@pytest.mark.parity("expand_vertex_mask", "meshlib")
@pytest.mark.parity("shrink_vertex_mask", "meshlib")
def test_expand_and_shrink_vertex_mask_match_meshlib(device: str, hops: int) -> None:
    """
    Class A on both, and the reference that makes ``shrink_vertex_mask`` comparable at all.

    MeshLab's Erode Selection is a *face* operation and is a documented class-D exemption for the
    erosion half (measured 51 / 39 / 25 surviving vertices where ordito gives 19 / 7 / 1), which
    left ``shrink_vertex_mask`` with a numpy oracle and no library to check it against. MeshLib's
    ``shrink`` takes a ``VertBitSet`` and erodes it by one-ring layers, which is ordito's operation
    exactly: element-for-element agreement at every hop count here, in both directions.

    Two things about the call, both of which fail silently if got wrong. ``expand`` and ``shrink``
    are **overload sets** whose region form returns ``None`` and *mutates* the bitset in place --
    the sibling overload taking a single ``VertId`` returns a new one instead, so a call written
    against the wrong overload reads an unchanged mask rather than raising. And the region is
    built with ``mn.vertBitSetFromBools`` and read back through
    [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy], which pads it to the
    vertex domain.

    Non-vacuous in both directions: the seed grows 7 -> 19 -> 35 -> 55 vertices of 162 and the
    eroded set falls 92 -> 73 -> 51 -> 31, so neither answer is the whole mesh or nothing.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    n_vertices = mesh_tm.vertices.shape[0]
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    seed_np = mesh_tm.vertices[:, 2] > 0.9
    region_np = mesh_tm.vertices[:, 2] > -0.2
    assert 0 < seed_np.sum() < region_np.sum() < n_vertices

    for mask_np, grow in ((seed_np, True), (region_np, False)):
        mask_wp = wp.array(np.ascontiguousarray(mask_np), dtype=wp.bool, device=device)
        morphed_wp = (
            od.selection.expand_vertex_mask(faces_wp, mask_wp, hops)
            if grow
            else od.selection.shrink_vertex_mask(faces_wp, mask_wp, hops)
        )

        mesh_ml = numpy_to_meshlib(mesh_tm.vertices, mesh_tm.faces)
        region_ml = mn.vertBitSetFromBools(cast("Buffer", np.ascontiguousarray(mask_np)))
        # The in-place overload: it returns None and rewrites `region_ml`.
        assert (mm.expand if grow else mm.shrink)(mesh_ml.topology, region_ml, hops) is None
        morphed_ml = meshlib_bitset_to_numpy(region_ml, n_vertices)

        assert 0 < morphed_ml.sum() < n_vertices
        assert morphed_ml.sum() != mask_np.sum()  # the reference actually moved the boundary
        assert np.array_equal(morphed_wp.numpy(), morphed_ml)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_vertex_morphology_on_an_irregular_mesh(device: str, hops: int) -> None:
    """
    Not a library comparison: a NumPy edge-list oracle, on the inputs a face-based round could miss.

    A round marks the corners of every face with a selected corner, which is the edge-neighbour
    dilation only because two vertices of a triangle mesh are one-ring neighbours exactly when they
    share a face. This pins that on a grid and what it lacks: a boundary, a non-manifold fin (three
    faces on one edge), a degenerate face with a repeated corner, and two unreferenced vertices --
    one selected, which must stay selected and spread nowhere, and one not, which nothing reaches.
    The seed is random and several vertices wide, and both outcomes of every vertex occur.
    """
    vertices_np, faces_np = _grid_mesh(6)
    n_grid = len(vertices_np)
    fin, loose_on, loose_off = n_grid, n_grid + 1, n_grid + 2
    extra = [7, 8, fin, 14, 14, 20]  # a third face on the interior edge (7, 8); a degenerate face
    faces_np = np.concatenate([faces_np, np.array(extra, dtype=np.int32)])
    n = n_grid + 3
    rng = np.random.default_rng(11)
    seed = np.zeros(n, dtype=bool)
    seed[rng.choice(n_grid, size=4, replace=False)] = True
    seed[loose_on] = True
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    seed_wp = wp.array(seed, dtype=wp.bool, device=device)

    expanded_np = od.selection.expand_vertex_mask(faces_wp, seed_wp, hops).numpy()
    expected_expand = _graph_distance(faces_np, n, seed) <= hops
    assert expected_expand[loose_on]
    assert not expected_expand[loose_off]
    assert 0 < expected_expand.sum() < n
    assert np.array_equal(expanded_np, expected_expand)

    grown = ~seed
    grown[[loose_on, loose_off]] = [True, False]
    shrunk_np = od.selection.shrink_vertex_mask(
        faces_wp, wp.array(grown, dtype=wp.bool, device=device), hops
    ).numpy()
    expected_shrink = _graph_distance(faces_np, n, ~grown) > hops
    assert 0 < expected_shrink.sum() < n
    assert np.array_equal(shrunk_np, expected_shrink)


@pytest.mark.parametrize("face_mode", ["all", "any"])
def test_face_indices_from_vertex_indices(
    request: pytest.FixtureRequest, face_mode: Literal["all", "any"]
) -> None:
    """Class A: both ``face_mode`` branches equal the numpy predicate, face index for face index."""
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    vertex_indices_np = _vertex_selection(mesh_tm, seed=11, fraction=4)
    vertex_indices = wp.array(vertex_indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    face_indices_wp = od.selection.face_indices_from_vertex_indices(
        mesh_wp.indices, vertex_indices, face_mode=face_mode
    )
    face_indices_ref_np = _face_indices_from_vertex_indices_np(
        mesh_tm.faces, vertex_indices_np, face_mode=face_mode
    )
    # Two empty face lists compare equal, which is what the "all" branch used to do.
    assert face_indices_ref_np.size > 0
    assert np.array_equal(face_indices_wp.numpy(), face_indices_ref_np)


def _assert_edge_key_order(rows_np: np.ndarray) -> None:
    """Assert undirected rows strictly ascend by ``(max, min)``, ``edges_unique``'s key order."""
    high = rows_np.max(axis=1).astype(np.int64)
    low = rows_np.min(axis=1).astype(np.int64)
    assert np.all(np.diff(high * (int(high.max(initial=0)) + 1) + low) > 0)
