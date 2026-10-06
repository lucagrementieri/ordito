"""Regression tests for ``ordito.mesh.Trimesh`` against Trimesh (CPU reference)."""

# This file tests Trimesh's private property cache and key tables directly.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from ordito.mesh import _ORIENTATION_DEPENDENT_KEYS, _TOPOLOGY_KEYS
from tests.comparisons import (
    SET_VALUED_CACHE_KEYS,
    assert_nonconstant,
    assert_same_loop_set,
    bsr_arrays,
    comparable_arrays,
    csr_row_sets,
    lexsort_rows,
    trimesh_outline_loops,
)
from tests.conftest import CLOSED_MESHES, MESHES, OPEN_MESHES, populate_cache
from tests.conversions import points_to_warp, points_to_warp_uv, trimesh_to_warp, warp_empty

# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_construction_flat_and_2d_faces_agree(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh_flat = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    mesh_2d = od.Trimesh(mesh_wp.points, mesh_wp.indices.reshape((-1, 3)))
    assert np.array_equal(mesh_flat.faces.numpy(), mesh_2d.faces.numpy())


@pytest.mark.parametrize(
    ("faces_shape", "error", "match"),
    [
        pytest.param((4, 3, 1), TypeError, None, id="rank-3"),
        pytest.param((3, 4), ValueError, "shape", id="four-columns"),
        pytest.param((4,), ValueError, "multiple of 3", id="flat-not-a-multiple-of-three"),
    ],
)
def test_construction_bad_faces_raises(
    device: str, faces_shape: tuple[int, ...], error: type[Exception], match: str | None
) -> None:
    """Not a parity assert: a face buffer that is not ``(3 * n_faces,)`` or ``(n_faces, 3)``."""
    vertices_wp = wp.zeros(12, dtype=wp.vec3, device=device)
    faces_wp = wp.zeros(faces_shape, dtype=wp.int32, device=device)
    with pytest.raises(error, match=match):
        od.Trimesh(vertices_wp, faces_wp)


def test_from_warp_mesh_seeds_warp_mesh_cache(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.warp_mesh is mesh_wp
    assert mesh.vertices is mesh_wp.points
    assert mesh.faces is mesh_wp.indices


def test_warp_mesh_raises_for_empty_mesh(device: str) -> None:
    # A warp.Mesh with zero triangles silently corrupts CUDA state (Warp 1.18); warp_mesh must
    # raise instead of building one.
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    mesh = od.Trimesh(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="zero triangles"):
        _ = mesh.warp_mesh


def test_traced_mesh_traces_the_rays_warp_mesh_traces(
    cave_cube: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: two BVHs over one mesh, which must change the cost and not the hits.

    The values' oracle is ``tests/test_visibility.py`` (trimesh / libigl); this pins that the cuBQL
    tree returns them bit for bit, for an any-hit bundle and a closest-hit cone alike, on a
    non-convex solid whose occlusion is not a constant.
    """
    _mesh_tm, mesh_wp = cave_cube
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    normals = od.vertices.vertex_normals(mesh.vertices, mesh.faces)
    occlusion_wp = od.visibility.ambient_occlusion(mesh.warp_mesh, mesh.vertices, normals=normals)
    traced_occlusion_wp = od.visibility.ambient_occlusion(
        mesh.traced_mesh, mesh.vertices, normals=normals
    )
    assert np.ptp(occlusion_wp.numpy()) > 0.1
    assert np.array_equal(traced_occlusion_wp.numpy(), occlusion_wp.numpy())
    diameter_wp = od.visibility.shape_diameter(mesh.warp_mesh, mesh.vertices, normals=normals)
    traced_diameter_wp = od.visibility.shape_diameter(
        mesh.traced_mesh, mesh.vertices, normals=normals
    )
    assert np.array_equal(traced_diameter_wp.numpy(), diameter_wp.numpy(), equal_nan=True)


def test_traced_mesh_is_warp_mesh_where_cubql_cannot_help(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a library comparison: on the CPU the default tree is kept, so nothing builds twice."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    if mesh.device.is_cpu or not wp.is_cubql_available():
        assert mesh.traced_mesh is mesh.warp_mesh
    else:
        assert mesh.traced_mesh is not mesh.warp_mesh


def test_mesh_for_rays_rents_then_buys(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: the online choice switches once the counted rays reach the budget.

    The budget is patched to 100 rays so the arithmetic is visible: two calls of 40 stay on
    `warp_mesh`, the third crosses the budget and builds `traced_mesh`, every later call keeps it,
    `invalidate` clears the count, and a single call over the budget switches at once.
    """
    monkeypatch.setattr("ordito.mesh._TRACED_MESH_FIXED_RAYS", 100)
    monkeypatch.setattr("ordito.mesh._TRACED_MESH_RAYS_PER_FACE", 0)
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    if mesh.device.is_cpu or not wp.is_cubql_available():
        assert mesh.mesh_for_rays(10**9) is mesh.warp_mesh
        return
    assert mesh.mesh_for_rays(40) is mesh.warp_mesh
    assert mesh.mesh_for_rays(40) is mesh.warp_mesh
    assert mesh.mesh_for_rays(30) is mesh.traced_mesh
    assert mesh.traced_mesh is not mesh.warp_mesh
    assert mesh.mesh_for_rays(1) is mesh.traced_mesh
    mesh.invalidate()
    assert mesh.mesh_for_rays(99) is mesh.warp_mesh
    mesh.invalidate()
    assert mesh.mesh_for_rays(100) is mesh.traced_mesh


def test_mesh_for_rays_rejects_a_negative_count(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="non-negative"):
        od.Trimesh.from_warp_mesh(mesh_wp).mesh_for_rays(-1)


def test_mesh_from_numpy_round_trip(icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    """Class A: the numpy arrays survive the upload unchanged, positions and indices alike."""
    mesh_tm, _mesh_wp = icosahedron
    mesh = od.io.mesh_from_numpy(mesh_tm.vertices, mesh_tm.faces, device=device)
    assert np.allclose(mesh.vertices.numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
    assert np.array_equal(mesh.faces.numpy().reshape(-1, 3), mesh_tm.faces)


# ---------------------------------------------------------------------------
# geometry vs trimesh
# ---------------------------------------------------------------------------


@pytest.fixture
def three_spheres(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Build three disjoint ``icosphere(1)`` bodies, ten units apart along ``x``."""
    parts = [tm.creation.icosphere(subdivisions=1) for _ in range(3)]
    for index, part in enumerate(parts):
        part.apply_translation([index * 10.0, 0.0, 0.0])
    combined_tm = tm.util.concatenate(parts)
    assert isinstance(combined_tm, tm.Trimesh)
    return combined_tm, trimesh_to_warp(combined_tm, device)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("mesh_vertex_normals", "trimesh")
def test_geometry_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on the cached geometry: same quantities, same order, no transform.

    ``mesh.py`` mirrors ``trimesh.Trimesh``'s property names deliberately (the one allowlisted
    exception to the summary-line rule in section 10), so this is the test that the names mean the
    same thing and not merely that they exist. Run over every fixture, closed and open, convex and
    not.

    ``bounds`` is trimesh's ``bounds``, ``extents`` its side lengths, and ``enclosing_diagonal`` its
    ``scale``: ``Trimesh.scale`` is documented as an order-of-magnitude figure but is exactly the
    diagonal of the axis-aligned box (measured equal to ``norm(bounds[1] - bounds[0])`` to 16
    digits), which is what makes this an equality and not a bound. The pairing is worth pinning
    because this property derives the diagonal from the cached box on the host instead of reducing
    again, so a wrong corner convention would show up here and nowhere else.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    assert np.allclose(mesh.face_normals.numpy(), mesh_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.face_areas.numpy(), mesh_tm.area_faces, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.area, mesh_tm.area, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.face_angles.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.centroid, mesh_tm.centroid, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.vertex_normals.numpy(), mesh_tm.vertex_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.vertex_defects.numpy(), mesh_tm.vertex_defects, rtol=1e-5, atol=1e-5)
    assert np.allclose(
        mesh.triangles_center.numpy(), mesh_tm.triangles_center, rtol=1e-5, atol=1e-5
    )

    lower, upper = mesh.bounds
    assert np.allclose(np.array([list(lower), list(upper)]), mesh_tm.bounds, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.extents, mesh_tm.extents, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.enclosing_diagonal, mesh_tm.scale, rtol=1e-5, atol=1e-5)
    assert min(mesh.extents) > 0.0  # non-vacuity: no fixture is flat


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_mass_properties_match_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: ``volume``, ``center_mass`` and ``moment_inertia`` against trimesh's.

    Closed fixtures only, and that is the definition rather than convenience: all three integrate
    tetrahedra from the origin to every face, which telescopes to an origin-independent answer only
    for a closed surface. On an open mesh both libraries return an origin-dependent number, so
    there would be nothing to compare.

    Non-vacuity: the inertia tensor is asserted to have spread across its entries, so a
    permutation or a wrong reference point could not pass on an isotropic answer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    assert mesh.volume == pytest.approx(mesh_tm.volume, rel=1e-5)
    assert np.allclose(mesh.center_mass, mesh_tm.center_mass, rtol=1e-4, atol=1e-4)
    inertia_wp = np.array(mesh.moment_inertia, dtype=np.float64).reshape(3, 3)
    assert np.allclose(inertia_wp, mesh_tm.moment_inertia, rtol=1e-4, atol=1e-4)
    assert_nonconstant(inertia_wp, tol=1e-3)


@pytest.mark.parametrize("mesh_name", [*MESHES, "three_spheres"])
def test_body_count_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: ``body_count`` against ``trimesh.Trimesh.body_count``.

    Non-vacuity: the expected count is asserted too, and it is not constant -- ``cave_cube`` is two
    bodies (the box and the cavity's inner shell), ``three_spheres`` three disjoint spheres, and
    the other fixtures one each.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    assert od.Trimesh.from_warp_mesh(mesh_wp).body_count == mesh_tm.body_count
    assert mesh_tm.body_count == {"cave_cube": 2, "three_spheres": 3}.get(mesh_name, 1)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_faces_unique_edges_indexes_edges_unique(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: ``faces_unique_edges`` against ``trimesh.Trimesh.faces_unique_edges``.

    Both sides index their *own* ``edges_unique``, whose row order is each library's, so the
    indices are not comparable directly -- the named transform is to resolve each side's indices
    through its own table and compare the resulting edges, which is what the property means.
    That makes this Class B on the indices and Class A on what they denote; the assert below is
    the latter.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    unique_wp = mesh.edges_unique.numpy()
    resolved_wp = unique_wp[mesh.faces_unique_edges.numpy()]
    resolved_tm = mesh_tm.edges_unique[mesh_tm.faces_unique_edges]
    assert np.array_equal(np.sort(resolved_wp, axis=2), np.sort(resolved_tm, axis=2))
    # ...and it is a view of the inverse, not a copy.
    assert mesh.faces_unique_edges.ptr == mesh.edges_unique_inverse.ptr


@pytest.mark.parametrize("mesh_name", MESHES)
def test_face_adjacency_projections_and_convex_match_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the two remaining ``face_adjacency_*`` quantities against trimesh's.

    Non-vacuity: ``face_adjacency_convex`` is asserted to contain both answers on ``cave_cube``,
    the non-convex fixture, since a property returning all-``True`` would otherwise pass on a
    convex mesh.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    assert np.allclose(
        mesh.face_adjacency_projections.numpy(),
        mesh_tm.face_adjacency_projections,
        rtol=1e-4,
        atol=1e-4,
    )
    convex_wp = mesh.face_adjacency_convex.numpy()
    assert np.array_equal(convex_wp, mesh_tm.face_adjacency_convex)
    if mesh_name == "cave_cube":
        assert convex_wp.any()
        assert not convex_wp.all()


# ---------------------------------------------------------------------------
# queries and mesh-producing methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_contains_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: ``contains`` against ``trimesh.Trimesh.contains`` on a grid spanning the mesh.

    Non-vacuity: both answers are asserted to contain inside *and* outside points, so a method
    returning a constant could not pass. Queries are drawn on a lattice over the bounding box
    rather than at random, so the two libraries are never asked about a point on the surface,
    where a parity ray is genuinely ambiguous and the answers may differ.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    lower, upper = mesh_tm.bounds
    axes = [np.linspace(lo, hi, 7)[1:-1] + 1e-3 for lo, hi in zip(lower, upper, strict=True)]
    grid_np = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    inside_wp = mesh.contains(points_to_warp(grid_np, mesh.device)).numpy()
    inside_tm = mesh_tm.contains(grid_np)
    assert inside_wp.any()
    assert not inside_wp.all()
    assert np.array_equal(inside_wp, inside_tm)


def test_contains_reuses_the_cached_bvh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a parity assert: the query goes through the cached BVH rather than building one."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    _ = mesh.contains(points_to_warp(np.zeros((1, 3)), mesh.device))
    assert mesh._cache["warp_mesh"] is mesh_wp


@pytest.mark.parametrize("mesh_name", MESHES)
def test_sample_lies_on_the_surface(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: sampling is stochastic, so there is no answer to compare against.

    Asserts what the sampler must satisfy instead -- every point lies on the face it reports, and
    the reported faces span more than one triangle. The barycentric residual excludes the bug
    class of a point/face index mismatch, which a distance-to-surface check alone would not: a
    point sampled from face ``j`` and labelled ``i`` still lies on the surface.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    points_wp, face_index_wp = mesh.sample(512, seed=3)

    points_np, faces_np = points_wp.numpy(), face_index_wp.numpy()
    assert np.unique(faces_np).size > 1
    assert faces_np.min() >= 0
    assert faces_np.max() < mesh.n_faces
    corners_np = mesh_tm.vertices[mesh_tm.faces[faces_np]]
    # Distance from each sample to the plane of the face it claims: zero if the pairing is right.
    normals_np = np.cross(corners_np[:, 1] - corners_np[:, 0], corners_np[:, 2] - corners_np[:, 0])
    normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
    offsets_np = np.einsum("ij,ij->i", points_np - corners_np[:, 0], normals_np)
    assert np.abs(offsets_np).max() < 1e-4


def test_sample_is_reproducible_under_a_seed(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: one seed gives one answer.

    The oracle for the sampler's correctness is ``test_sample_lies_on_the_surface``; this pins
    only that the seed is honoured, which that test cannot see.
    """
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    first, _ = mesh.sample(64, seed=11)
    second, _ = mesh.sample(64, seed=11)
    assert np.array_equal(first.numpy(), second.numpy())


@pytest.mark.parametrize("mesh_name", MESHES)
def test_submesh_matches_the_free_function(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Ordito against ordito: the mask and index forms of ``submesh`` agree with each other.

    ``ordito.selection`` carries the oracle for the extraction itself (``tests/test_selection.py``
    compares it against trimesh); this pins the class's dtype dispatch, which is the only thing the
    method adds and the only place a wrong branch would show.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    keep_np = np.zeros(mesh.n_faces, dtype=bool)
    keep_np[::2] = True

    by_mask = mesh.submesh(wp.array(keep_np, dtype=wp.bool, device=mesh.device))
    by_index = mesh.submesh(
        wp.array(np.flatnonzero(keep_np).astype(np.int32), dtype=wp.int32, device=mesh.device)
    )
    assert by_mask.n_faces == keep_np.sum()
    assert np.array_equal(by_mask.vertices.numpy(), by_index.vertices.numpy())
    assert np.array_equal(by_mask.faces.numpy(), by_index.faces.numpy())


def test_submesh_bad_dtype_raises(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a parity assert: the guard on a selector that is neither indices nor a mask."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    with pytest.raises(TypeError, match=r"wp\.int32 or wp\.bool"):
        mesh.submesh(wp.zeros(4, dtype=wp.float32, device=mesh.device))


def test_split_and_add_round_trip(three_spheres: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: three disjoint spheres split into three bodies, matching ``trimesh.Trimesh.split``.

    Also pins the pair: ``__add__`` recombines them into a mesh with the original counts, which
    is the round trip the two methods claim.
    """
    combined_tm, mesh_wp = three_spheres
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)

    bodies = mesh.split()
    assert len(bodies) == len(combined_tm.split(only_watertight=False)) == 3
    assert sorted(body.n_faces for body in bodies) == sorted(
        len(part.faces) for part in combined_tm.split(only_watertight=False)
    )

    rejoined = bodies[0] + bodies[1] + bodies[2]
    assert rejoined.n_vertices == mesh.n_vertices
    assert rejoined.n_faces == mesh.n_faces
    assert rejoined.body_count == 3


def test_copy_shares_nothing(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a parity assert: ``copy`` breaks the aliasing every other method preserves."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    _ = mesh.face_normals
    duplicate = mesh.copy()

    assert duplicate.vertices.ptr != mesh.vertices.ptr
    assert duplicate.faces.ptr != mesh.faces.ptr
    assert np.array_equal(duplicate.vertices.numpy(), mesh.vertices.numpy())
    assert not duplicate._cache
    # ...where `with_vertices` deliberately aliases the face buffer instead.
    assert mesh.with_vertices(mesh.vertices).faces.ptr == mesh.faces.ptr


# ---------------------------------------------------------------------------
# edges / adjacency vs trimesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_edges_match_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A, and the row *order* is part of the claim -- ``array_equal``, not a lexsort.

    ``edges`` / ``edges_sorted`` / ``edges_face`` are all indexed by the same halfedge position, so
    a reordering would silently break the correspondence between them even while each stayed a
    correct set. That is why this one is not canonicalized where
    [`test_edges_unique_matches_trimesh`] is.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    assert np.array_equal(mesh.edges.numpy(), mesh_tm.edges)
    assert np.array_equal(mesh.edges_sorted.numpy(), mesh_tm.edges_sorted)
    assert np.array_equal(mesh.edges_face.numpy(), mesh_tm.edges_face)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_edges_unique_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row-set canonicalization): the unique edge *set* matches after a shared lexsort.

    Here the order genuinely is not defined by either library -- ordito's comes from a parallel
    hash and trimesh's from a serial scan -- so both sides are sorted. The `inverse` is then checked
    against ordito's own ``edges_sorted`` rather than trimesh's, because it indexes into ordito's
    row order and that is the invariant the sort throws away.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    unique_wp = mesh.edges_unique.numpy()
    unique_tm = mesh_tm.edges_unique
    assert np.array_equal(lexsort_rows(unique_wp), lexsort_rows(unique_tm))

    # unique_edges[inverse] must reconstruct edges_sorted, independent of row order.
    reconstructed = unique_wp[mesh.edges_unique_inverse.numpy()]
    assert np.array_equal(reconstructed, mesh.edges_sorted.numpy())


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("mesh_face_adjacency", "trimesh")
def test_face_adjacency_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B, two named transforms: sort within each pair, then lexsort the rows.

    An adjacency pair is unordered and the list of pairs is unordered, and neither library defines
    either -- so both have to be canonicalized before the sets can be compared elementwise.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    adjacency_wp = mesh.face_adjacency.numpy()
    adjacency_tm = tm.graph.face_adjacency(mesh=mesh_tm)
    assert np.array_equal(lexsort_rows(adjacency_wp), lexsort_rows(np.sort(adjacency_tm, axis=1)))


# ---------------------------------------------------------------------------
# halfedge connectivity, incidence and the discrete operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_vertex_face_adjacency_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B on ``vertex_face_adjacency``: row *sets*, after two named transforms.

    trimesh returns a dense ``(n_vertices, max_degree)`` array right-padded with ``-1`` where
    ordito returns a CSR pair, and neither orders a row -- ordito's counting sort fills each row
    through an atomic cursor, so the order differs between two calls on the same mesh. The set is
    the whole claim, and it is the one the free function's docstring makes ("each row is a set, not
    a rotation").
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    faces_np, offsets_np = (array.numpy() for array in mesh.vertex_face_adjacency)

    padded_tm = mesh_tm.vertex_faces
    assert padded_tm.shape[0] == mesh.n_vertices
    for vertex in range(mesh.n_vertices):
        row = faces_np[offsets_np[vertex] : offsets_np[vertex + 1]]
        row_tm = padded_tm[vertex][padded_tm[vertex] >= 0]
        assert np.array_equal(np.sort(row), np.sort(row_tm))


def test_halfedge_properties_match_the_free_functions(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: the facade against ``ordito.halfedge``, which carries the oracle.

    No reference library exposes a halfedge structure (``tests/test_halfedge.py`` is invariant-only
    for that reason), so what is testable here is that the properties are the free functions'
    answers and that the class's ``n_vertices`` shortcut -- which replaces the inferred vertex
    count both functions would otherwise read back -- does not change them.
    """
    _mesh_tm, mesh_wp = icosphere_coarse
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    faces_wp = mesh.faces

    assert np.array_equal(mesh.halfedge_twins.numpy(), od.halfedge.halfedge_twins(faces_wp).numpy())
    rings_free = od.halfedge.vertex_one_rings(faces_wp)
    for cached, free in zip(mesh.vertex_one_rings, rings_free, strict=True):
        assert np.array_equal(cached.numpy(), free.numpy())


def test_operator_properties_match_the_free_functions(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: the operator group against the functions that assemble it.

    Each of these is compared with a reference elsewhere -- ``cotmatrix`` and the mass diagonal
    against igl in ``tests/test_laplacian.py``, ``laplacian_operator`` against trimesh in
    ``tests/test_smoothing.py``, the frames and both heat bundles against potpourri3d in
    ``tests/test_tangent.py`` and ``tests/test_heat_*.py``. So the free functions carry the oracle
    and what is left to pin here is that the properties feed them the same mesh, including the
    cached by-products they are built from (``cotmatrix`` from ``cotmatrix_entries``, the mass
    diagonal from ``face_areas``).
    """
    _mesh_tm, mesh_wp = icosphere_coarse
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    vertices_wp, faces_wp = mesh.vertices, mesh.faces

    assert np.allclose(
        mesh.cotmatrix_entries.numpy(),
        od.laplacian.cotmatrix_entries(vertices_wp, faces_wp).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        mesh.mass_matrix_entries.numpy(),
        od.laplacian.mass_matrix_entries(vertices_wp, faces_wp).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    for cached_matrix, free_matrix in (
        (mesh.cotmatrix, od.laplacian.cotmatrix(vertices_wp, faces_wp)),
        (mesh.laplacian_operator, od.laplacian.laplacian(vertices_wp, faces_wp)),
    ):
        for cached_np, free_np in zip(
            bsr_arrays(cached_matrix), bsr_arrays(free_matrix), strict=True
        ):
            assert np.allclose(cached_np, free_np, rtol=1e-5, atol=1e-5)

    frames_free = od.tangent_space.vertex_tangent_frames(vertices_wp, faces_wp)
    for cached, free in zip(mesh.vertex_tangent_frames, frames_free, strict=True):
        assert np.allclose(cached.numpy(), free.numpy(), rtol=1e-5, atol=1e-5)
    # The gauge is the normal's, so its third field is the class's own vertex normals verbatim.
    assert mesh.vertex_tangent_frames[2] is mesh.vertex_normals


# ---------------------------------------------------------------------------
# boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_matches_free_functions(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    boundary_edges_ref = od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(mesh.boundary_edges.numpy(), boundary_edges_ref.numpy())

    boundary_vertex_indices_ref = od.boundary.boundary_vertex_indices(
        mesh_wp.points, mesh_wp.indices
    )
    assert np.array_equal(mesh.boundary_vertex_indices.numpy(), boundary_vertex_indices_ref.numpy())

    assert len(mesh.boundary_loops) > 0


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("mesh_boundary_loops", "trimesh")
def test_boundary_loops_matches_trimesh_outline(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the cached property against ``Trimesh.outline()``, the property trimesh caches.

    The container-level twin of
    [`tests.test_boundary.test_boundary_loops_matches_trimesh_outline`][], and it exists separately
    because ``benchmarks/test_mesh.py`` times the two as separate groups -- the free function on the
    ``loops`` axis, this one warm against cold. Both named transforms live in
    [`tests.comparisons.trimesh_outline_loops`][]: the entities index the mesh's own vertex array,
    and a closed entity repeats its first point as its last.

    The loop *set* is the claim, not the order: ordito ranks loops by length and trimesh by
    traversal, and neither fixes a starting point within a loop.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    loops_tm = trimesh_outline_loops(mesh_tm)
    assert len(loops_tm) > 0  # non-vacuous: these fixtures have rims
    assert_same_loop_set([loop.numpy() for loop in mesh.boundary_loops], loops_tm)


def test_boundary_loops_empty_for_closed_mesh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.boundary_loops == []
    assert mesh.boundary_vertex_indices.shape == (0,)


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------


@pytest.fixture
def half_flipped_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> tuple[tm.Trimesh, wp.Mesh]:
    """Reverse every other ``icosahedron`` face: orientable, but not consistently wound."""
    mesh_tm, mesh_wp = icosahedron
    faces_np = np.array(mesh_tm.faces)
    faces_np[::2] = faces_np[::2, ::-1]
    flipped_tm = tm.Trimesh(mesh_tm.vertices, faces_np, process=False)
    return flipped_tm, trimesh_to_warp(flipped_tm, mesh_wp.device)


_PREDICATE_MESHES = [*MESHES, "boy_surface", "mobius", "half_flipped_icosahedron"]


@pytest.mark.parametrize("mesh_name", _PREDICATE_MESHES)
def test_predicates_match_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on an integer and two booleans, parametrized over inputs giving *both* answers.

    ``euler_characteristic`` is ``V - E + F`` against ``Trimesh.euler_number``, exactly: 2 on the
    closed orientable meshes, 0 on ``half_torus`` and ``mobius``, 1 on ``hemisphere`` and
    ``boy_surface``. ``is_winding_consistent`` and ``is_volume`` (the conjunction trimesh calls
    ``is_volume``) answer ``True`` on the closed orientable fixtures; the open fixtures, the
    non-orientable ones and ``half_flipped_icosahedron`` supply the ``False`` branches. Section 7.4
    rules out asserting a predicate on one branch only, so the expected answers are asserted too.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.euler_characteristic == int(mesh_tm.euler_number)
    assert mesh.is_winding_consistent == bool(mesh_tm.is_winding_consistent)
    assert mesh.is_volume == bool(mesh_tm.is_volume)

    consistent = mesh_name not in {"boy_surface", "mobius", "half_flipped_icosahedron"}
    assert bool(mesh_tm.is_winding_consistent) == consistent
    assert bool(mesh_tm.is_volume) == (consistent and mesh_name in CLOSED_MESHES)


@pytest.mark.parametrize("mesh_name", [*MESHES, "bohemian_dome"])
@pytest.mark.parity("mesh_is_watertight", "trimesh")
def test_is_watertight_decomposes_into_trimesh_clause(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``Trimesh.is_watertight`` is exactly one of the two conjuncts this property tests.

    The named transform is the conjunction itself. trimesh answers "every edge is shared by exactly
    two faces"; ordito follows Open3D and requires that **and** no self-intersection, so the two
    disagree by construction on a closed surface that passes through itself and comparing them
    directly would look like a bug in ordito. Supplying the missing conjunct makes it an equality:
    ``mesh.is_watertight == mesh_tm.is_watertight and not is_self_intersecting(...)``, measured
    exact on all eight surfaces probed (the four fixtures here plus ``roman``, ``klein``,
    ``cross_cap`` and ``mobius``).

    That is a stronger claim than an implication, and it is the one worth making: an implication
    (ordito ⟹ trimesh) would also pass for a property that always answered ``False``.

    Non-vacuous in both directions on this parametrisation -- ``icosahedron`` and ``cave_cube`` are
    watertight both ways, the two open fixtures fail trimesh's clause, and ``bohemian_dome`` is the
    input that separates the two definitions: trimesh ``True``, ordito ``False``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)

    edge_manifold_closed_tm = bool(mesh_tm.is_watertight)
    self_intersecting = od.validation.is_self_intersecting(mesh_wp)
    assert mesh.is_watertight == (edge_manifold_closed_tm and not self_intersecting)


@pytest.mark.parametrize("mesh_name", [*MESHES, "boy_surface", "mobius", "bohemian_dome"])
def test_is_watertight_builds_the_bvh_only_when_it_needs_one(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the cold property against the free function, and what it caches.

    The free function carries the oracle (``test_is_watertight_decomposes_into_trimesh_clause``).
    What this pins is the cold path the ``from_warp_mesh`` tests never reach: a ``Trimesh`` built
    from buffers has no BVH, and builds -- and caches -- `warp_mesh` exactly when the two manifold
    tests pass and the broad phase needs it. ``mobius`` and the open fixtures stop at the edge
    test; ``boy_surface`` and ``bohemian_dome`` pass both manifold tests and reach the broad phase,
    so both outcomes of the caching are asserted.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    reaches_broad_phase = od.validation.is_edge_manifold(
        mesh_wp.indices, allow_boundary_edges=False
    ) and od.validation.is_vertex_manifold(mesh_wp.indices)
    assert mesh.is_watertight == od.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert ("warp_mesh" in mesh._cache) == reaches_broad_phase
    if reaches_broad_phase:
        assert mesh.warp_mesh is mesh._cache["warp_mesh"]


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_is_edge_and_vertex_manifold_closed_meshes(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.is_edge_manifold is True
    assert mesh.is_vertex_manifold is True
    assert mesh.is_self_intersecting is False


# ---------------------------------------------------------------------------
# cache mechanics
# ---------------------------------------------------------------------------


def test_cached_properties_are_frozen(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    with pytest.raises(AttributeError):
        mesh.face_normals = mesh.face_normals


def test_invalidate_clears_cache_and_recomputes(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    normals_before = mesh.face_normals
    mesh.invalidate()
    assert mesh._cache == {}
    normals_after = mesh.face_normals
    assert normals_after is not normals_before
    assert np.array_equal(normals_after.numpy(), normals_before.numpy())


def _translated(mesh: od.Trimesh) -> od.Trimesh:
    """Return ``mesh.with_vertices`` over its own vertices moved one unit along ``z``."""
    translated_np = mesh.vertices.numpy() + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return mesh.with_vertices(points_to_warp(translated_np, mesh.device))


def test_with_vertices_keeps_topology_drops_geometry(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    edges_before = mesh.edges
    _ = mesh.face_normals
    _ = mesh.warp_mesh

    moved = _translated(mesh)
    assert moved.edges is edges_before
    assert "face_normals" not in moved._cache
    assert "warp_mesh" not in moved._cache
    assert moved.faces is mesh.faces


@pytest.mark.parametrize("key", sorted(_TOPOLOGY_KEYS))
def test_with_vertices_carries_every_topology_key(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], key: str
) -> None:
    """Each faces-only cached property survives ``with_vertices`` (the ``_TOPOLOGY_KEYS`` rule)."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    before = getattr(mesh, key)
    assert key in mesh._cache

    moved = _translated(mesh)

    assert key in moved._cache, f"{key} was recomputed instead of carried forward"
    assert moved._cache[key] is before


def test_with_vertices_wrong_count_raises(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    too_few = points_to_warp(mesh.vertices.numpy()[:-1], mesh.device)
    with pytest.raises(ValueError, match="vertex count"):
        mesh.with_vertices(too_few)


def test_with_faces_starts_with_empty_cache(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    _ = mesh.edges

    new_mesh = mesh.with_faces(wp.clone(mesh.faces))
    assert new_mesh._cache == {}
    assert np.array_equal(new_mesh.edges.numpy(), mesh.edges.numpy())


def test_warp_mesh_supports_ray_queries(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: rays against the BVH ``warp_mesh`` builds, matching ``trimesh``'s ray queries.

    The mesh is built from buffers, so ``warp_mesh`` is a BVH of its own rather than the fixture's.
    Non-vacuity: the rays are chosen to give both answers (down through the centre hits, up from
    above and down beside the mesh miss).
    """
    mesh_tm, mesh_wp = icosahedron
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    center_np = mesh_tm.bounds.mean(axis=0)
    origins_np = center_np + np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0], [5.0, 5.0, 3.0]])
    directions_np = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    hits_wp = od.ray.intersects_any(
        mesh.warp_mesh,
        points_to_warp(origins_np, mesh.device),
        points_to_warp(directions_np, mesh.device),
    ).numpy()
    hits_tm = mesh_tm.ray.intersects_any(origins_np, directions_np)
    assert mesh.warp_mesh is not mesh_wp
    assert np.array_equal(hits_wp, hits_tm)
    assert np.array_equal(hits_tm, [True, False, False])


@pytest.mark.parity("reverse_winding", "trimesh")
@pytest.mark.parametrize("mesh_name", MESHES)
def test_invert_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: ``Trimesh.invert`` against ``trimesh.Trimesh.invert``, face buffer for face buffer.

    Comparable elementwise rather than up to a rotation of each row because both reverse with
    ``np.fliplr``'s convention -- ``(a, b, c) -> (c, b, a)``. Vertices must be untouched on both
    sides, which is what separates this from a mirroring transform. Not a library comparison, the
    last assert: inverting twice returns the original face buffer, the property that defines the
    operation.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    inverted = mesh.invert()

    flipped_tm = mesh_tm.copy()
    flipped_tm.invert()
    assert np.array_equal(inverted.faces.numpy().reshape(-1, 3), flipped_tm.faces)
    assert np.array_equal(inverted.vertices.numpy(), mesh.vertices.numpy())
    # non-vacuity: a no-op `invert` would pass the vertex assert and fail this one
    assert not np.array_equal(inverted.faces.numpy(), mesh.faces.numpy())
    # The involution excludes a cyclic rotation standing in for a reversal, which would agree on
    # normals and volume and never return to the original face buffer.
    assert np.array_equal(inverted.invert().faces.numpy(), mesh.faces.numpy())


@pytest.mark.parametrize("mesh_name", MESHES)
def test_invert_carried_cache_matches_recomputation(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: every entry ``invert`` carries equals recomputing it from scratch.

    The same gate as ``test_carried_cache_matches_recomputation``, for the winding flip. The
    oracle is the uncached ``od.Trimesh(vertices, reversed_faces)``.

    Non-vacuity: the carried set is asserted non-empty and to include the assembled ``cotmatrix``,
    which is the one worth carrying; and the run spans a closed and an open mesh, because
    ``laplacian_operator``, ``oriented_boundary_edges`` and ``boundary_loops`` survive a flip on a
    closed mesh and fail on an open one.

    Not a parity assert, the other half: a flip drops the same orientation-dependent caches a
    mirroring transform does, asserted after checking the fixture populated them.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    inverted = mesh.invert()

    assert _ORIENTATION_DEPENDENT_KEYS & set(mesh._cache), "fixture did not populate them"
    assert not (_ORIENTATION_DEPENDENT_KEYS & set(inverted._cache))
    assert inverted._cache
    assert "cotmatrix" in inverted._cache
    reference = od.Trimesh(inverted.vertices, inverted.faces)
    for key, carried in inverted._cache.items():
        if key in SET_VALUED_CACHE_KEYS:
            assert csr_row_sets(
                cast("tuple[wp.array[wp.int32], wp.array[wp.int32]]", carried)
            ) == csr_row_sets(getattr(reference, key)), (
                f"invert carried a stale {key} on {mesh_name}"
            )
            continue
        for got, expected in zip(
            comparable_arrays(carried), comparable_arrays(getattr(reference, key)), strict=True
        ):
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-4), (
                f"invert carried a stale {key} on {mesh_name}"
            )


def test_invert_negates_normals_and_volume(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: the flip negates normals and volume, matching trimesh on both.

    The normals are the carried-and-negated path rather than a recomputation, so this checks the
    shortcut `invert` takes; ``volume`` is dropped and recomputed, and must come back negated.
    """
    mesh_tm, mesh_wp = icosphere
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    inverted = mesh.invert()

    assert np.allclose(
        inverted.face_normals.numpy(), -mesh.face_normals.numpy(), rtol=1e-5, atol=1e-5
    )
    assert np.allclose(
        inverted.vertex_normals.numpy(), -mesh.vertex_normals.numpy(), rtol=1e-5, atol=1e-5
    )
    assert inverted.volume == pytest.approx(-mesh.volume, rel=1e-5)

    flipped_tm = mesh_tm.copy()
    flipped_tm.invert()
    assert np.allclose(inverted.face_normals.numpy(), flipped_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert inverted.volume == pytest.approx(flipped_tm.volume, rel=1e-5)


# ---------------------------------------------------------------------------
# precomputed arguments: the cache feeding the free functions
# ---------------------------------------------------------------------------


# Every cached property that reads vertex *positions* and so must not survive ``with_vertices``.
# The topology-only half is ``_TOPOLOGY_KEYS`` and is covered by the parametrized test above; this
# is the other side of the same rule, and it exists because the operator group below is the easiest
# place to get it wrong -- an operator is assembled from the connectivity *and* the geometry.
_GEOMETRY_KEYS = (
    "bounds",
    "enclosing_diagonal",
    "cotmatrix_entries",
    "cotmatrix",
    "mass_matrix_entries",
    "volume",
    "center_mass",
    "moment_inertia",
    "extents",
    "triangles_center",
    "face_adjacency_projections",
    "face_adjacency_convex",
    # `face_adjacency_angles` reads `face_normals`, so it is position-dependent too.
    "face_adjacency_angles",
    # `laplacian_operator` is deliberately absent: it weights every 1-ring neighbour equally, so it
    # reads `faces` and never the positions -- measured bit-identical across a *scrambled* vertex
    # buffer on a closed and an open mesh. It lives in `_TOPOLOGY_KEYS`, and the parametrization
    # over that set below is what covers it.
    "vertex_tangent_frames",
    "heat_operators",
    "vector_heat_operators",
)


@pytest.mark.parametrize("key", _GEOMETRY_KEYS)
def test_with_vertices_drops_every_position_dependent_cache(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh], key: str
) -> None:
    """Each position-dependent cached property is dropped by ``with_vertices``, not carried."""
    _mesh_tm, mesh_wp = icosphere_coarse
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    assert getattr(mesh, key) is not None
    assert key in mesh._cache

    moved = _translated(mesh)
    assert key not in moved._cache


@pytest.mark.parametrize(
    ("name", "dependencies"),
    [
        ("face_normals", ("face_areas",)),
        ("center_mass", ("moment_inertia",)),
        ("moment_inertia", ("center_mass",)),
        ("edges_unique", ("edges_unique_inverse",)),
        ("face_adjacency", ("face_adjacency_edges",)),
        ("enclosing_diagonal", ("bounds",)),
        ("vertex_one_rings", ("halfedge_twins",)),
        ("cotmatrix", ("cotmatrix_entries",)),
        ("mass_matrix_entries", ("face_areas",)),
        ("vertex_tangent_frames", ("vertex_normals", "vertex_one_rings", "halfedge_twins")),
        ("heat_operators", ("cotmatrix_entries",)),
        ("vector_heat_operators", ("heat_operators", "vertex_tangent_frames")),
    ],
)
def test_cached_property_reuses_its_dependencies(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh], name: str, dependencies: tuple[str, ...]
) -> None:
    """
    A composed property is assembled *through* the cache, so its parts land in it too.

    Not a parity assert: it is the class's own contract. It is worth a test rather than a comment
    because each of these properties would compute the identical answer while rebuilding its parts
    privately, and nothing about the returned value would show the difference -- only the cache
    does. The first five rows are by-products: one call computes both, so reading either caches
    the other. A second read returns the cached object itself, for the property and its parts.
    """
    _mesh_tm, mesh_wp = icosphere_coarse
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    value = getattr(mesh, name)
    assert value is not None
    for dependency in dependencies:
        assert dependency in mesh._cache, f"{name} rebuilt {dependency} instead of caching it"
        assert getattr(mesh, dependency) is mesh._cache[dependency]
    assert getattr(mesh, name) is value


def test_vector_heat_operators_shares_its_two_sub_bundles(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    The vector bundle's second and third fields *are* the sibling properties, not equal copies.

    Identity rather than equality is the claim: the three properties must be one assembly however
    they are reached, since ``log_map``'s radius is asserted to be the ``heat_geodesic`` distance
    and two bundles at two diffusion times would split that into two numbers.
    """
    _mesh_tm, mesh_wp = icosphere_coarse
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    vector_system, scalar, frames, _preconditioner = mesh.vector_heat_operators

    assert scalar is mesh.heat_operators
    assert frames is mesh.vertex_tangent_frames
    assert int(vector_system.nrow) == mesh.n_vertices


# One id per entry of the table below, module-level so the parametrization is visible without
# building a mesh; the test asserts the two stay in step.
_PRECOMPUTED_ARGUMENT_IDS = (
    "seams.cut_along_edges(twins=)",
    "seams.uv_seam_edges(twins=)",
    "repair.flatten_degree3_vertices(rings=)",
    "energies.hessian_energy(vertex_faces=)",
    "smoothing.equalize_triangle_areas(vertex_faces=)",
    "smoothing.smooth_region_boundary(vertex_faces=)",
    "smoothing.filter_normals(face_normals=)",
    "smoothing.filter_mut_dif_laplacian(face_normals=)",
    "laplacian.mass_matrix_entries(face_areas=)",
    "laplacian.mass_matrix(face_areas=)",
    "triangles.corner_normals(face_normals=)",
    "curvature.principal_curvature(face_normals=)",
    "validation.face_defective_mask(face_normals=)",
    "proximity.normals_at_closest_faces(face_normals=)",
    "heat.distance.heat_operators(cot_entries=)",
    "heat.vector.vector_heat_operators(scalar_operators=)",
)


def _precomputed_argument_cases(
    mesh: od.Trimesh,
) -> dict[str, tuple[Callable[[], object], Callable[[], object]]]:
    """
    ``id -> (rebuild-it-yourself call, fed-from-the-cache call)`` per precomputed argument.

    One table here rather than a pin in each wrapper's own test file, because the claim is about the
    *pairing* -- that a given cached property is the right thing to hand a given keyword -- and not
    about any wrapper's behaviour, which its own module's tests cover. A wrong pairing is otherwise
    silent: every one of these arguments is a buffer of the right shape and dtype whichever mesh
    quantity it came from.
    """
    vertices_wp, faces_wp = mesh.vertices, mesh.faces
    device = mesh.device
    creases_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=0.0)
    corner_uv_np = mesh.vertices.numpy()[faces_wp.numpy()][:, :2]
    texcoords_wp = points_to_warp_uv(corner_uv_np, str(device))
    centroids_np = od.triangles.face_centroids(vertices_wp, faces_wp).numpy()
    region_wp = wp.array(centroids_np[:, 2] > 0.0, dtype=wp.bool, device=device)
    queries_wp = points_to_warp(
        np.array([[0.4, 0.4, 0.4], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32), str(device)
    )

    return {
        "seams.cut_along_edges(twins=)": (
            lambda: od.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp),
            lambda: od.seams.cut_along_edges(
                vertices_wp, faces_wp, creases_wp, twins=mesh.halfedge_twins
            ),
        ),
        "seams.uv_seam_edges(twins=)": (
            lambda: od.seams.uv_seam_edges(faces_wp, texcoords_wp),
            lambda: od.seams.uv_seam_edges(faces_wp, texcoords_wp, twins=mesh.halfedge_twins),
        ),
        "repair.flatten_degree3_vertices(rings=)": (
            lambda: od.repair.flatten_degree3_vertices(vertices_wp, faces_wp),
            lambda: od.repair.flatten_degree3_vertices(
                vertices_wp, faces_wp, rings=mesh.vertex_one_rings
            ),
        ),
        "energies.hessian_energy(vertex_faces=)": (
            lambda: od.energies.hessian_energy(vertices_wp, faces_wp),
            lambda: od.energies.hessian_energy(
                vertices_wp, faces_wp, vertex_faces=mesh.vertex_face_adjacency
            ),
        ),
        "smoothing.equalize_triangle_areas(vertex_faces=)": (
            lambda: od.smoothing.equalize_triangle_areas(vertices_wp, faces_wp, iterations=2),
            lambda: od.smoothing.equalize_triangle_areas(
                vertices_wp, faces_wp, iterations=2, vertex_faces=mesh.vertex_face_adjacency
            ),
        ),
        "smoothing.smooth_region_boundary(vertex_faces=)": (
            lambda: od.smoothing.smooth_region_boundary(vertices_wp, faces_wp, region_wp),
            lambda: od.smoothing.smooth_region_boundary(
                vertices_wp, faces_wp, region_wp, vertex_faces=mesh.vertex_face_adjacency
            ),
        ),
        "smoothing.filter_normals(face_normals=)": (
            lambda: od.smoothing.filter_normals(vertices_wp, faces_wp, iterations=3),
            lambda: od.smoothing.filter_normals(
                vertices_wp,
                faces_wp,
                iterations=3,
                face_normals=mesh.face_normals,
                face_areas=mesh.face_areas,
            ),
        ),
        "smoothing.filter_mut_dif_laplacian(face_normals=)": (
            lambda: od.smoothing.filter_mut_dif_laplacian(vertices_wp, faces_wp, iterations=2),
            lambda: od.smoothing.filter_mut_dif_laplacian(
                vertices_wp,
                faces_wp,
                iterations=2,
                face_normals=mesh.face_normals,
                face_areas=mesh.face_areas,
            ),
        ),
        "laplacian.mass_matrix_entries(face_areas=)": (
            lambda: od.laplacian.mass_matrix_entries(vertices_wp, faces_wp),
            lambda: od.laplacian.mass_matrix_entries(
                vertices_wp, faces_wp, face_areas=mesh.face_areas
            ),
        ),
        "laplacian.mass_matrix(face_areas=)": (
            lambda: od.laplacian.mass_matrix(vertices_wp, faces_wp),
            lambda: od.laplacian.mass_matrix(vertices_wp, faces_wp, face_areas=mesh.face_areas),
        ),
        "triangles.corner_normals(face_normals=)": (
            lambda: od.triangles.corner_normals(
                vertices_wp, faces_wp, creases_wp, weighting="area"
            ),
            lambda: od.triangles.corner_normals(
                vertices_wp,
                faces_wp,
                creases_wp,
                weighting="area",
                twins=mesh.halfedge_twins,
                face_normals=mesh.face_normals,
                face_areas=mesh.face_areas,
            ),
        ),
        # ``[2:]`` keeps the two curvature *magnitudes* and drops the two directions, which are a
        # gauge on this fixture rather than an answer: every vertex of a sphere is umbilic, so any
        # tangent direction is principal and two identical calls disagree by up to 1.618 (measured)
        # while the magnitudes agree to 3.6e-07.
        "curvature.principal_curvature(face_normals=)": (
            lambda: od.curvature.principal_curvature(vertices_wp, faces_wp, radius=2)[2:],
            lambda: od.curvature.principal_curvature(
                vertices_wp,
                faces_wp,
                radius=2,
                face_normals=mesh.face_normals,
                face_areas=mesh.face_areas,
            )[2:],
        ),
        "validation.face_defective_mask(face_normals=)": (
            lambda: od.validation.face_defective_mask(
                vertices_wp, faces_wp, max_normal_angle=60.0, max_fold_angle=160.0
            ),
            lambda: od.validation.face_defective_mask(
                vertices_wp,
                faces_wp,
                max_normal_angle=60.0,
                max_fold_angle=160.0,
                face_normals=mesh.face_normals,
            ),
        ),
        "proximity.normals_at_closest_faces(face_normals=)": (
            lambda: od.proximity.normals_at_closest_faces(mesh.warp_mesh, queries_wp),
            lambda: od.proximity.normals_at_closest_faces(
                mesh.warp_mesh, queries_wp, face_normals=mesh.face_normals
            ),
        ),
        "heat.distance.heat_operators(cot_entries=)": (
            lambda: od.heat.heat_operators(vertices_wp, faces_wp),
            lambda: od.heat.heat_operators(
                vertices_wp, faces_wp, cot_entries=mesh.cotmatrix_entries
            ),
        ),
        "heat.vector.vector_heat_operators(scalar_operators=)": (
            lambda: od.heat.vector_heat_operators(vertices_wp, faces_wp),
            lambda: od.heat.vector_heat_operators(
                vertices_wp,
                faces_wp,
                scalar_operators=mesh.heat_operators,
                frames=mesh.vertex_tangent_frames,
            ),
        ),
    }


@pytest.mark.parametrize("name", _PRECOMPUTED_ARGUMENT_IDS)
def test_a_precomputed_argument_does_not_change_the_answer(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh], name: str
) -> None:
    """
    Every keyword added for the cache returns what the wrapper would have computed itself.

    Not a parity assert: ordito against ordito, and the *recomputing* call is the one every
    reference comparison in the suite already runs, so it carries the oracle. What this closes is
    the failure mode a precomputed argument introduces and nothing else can see -- the wrapper
    trusting a buffer that is the right shape and the wrong quantity.

    Tolerant rather than exact on the float rows because two of the quantities involved are built
    by atomic scatters (the lumped mass, the vertex normals) and one runs a CG solve, so bit
    equality is not a property of the *unchanged* code either.
    """
    _mesh_tm, mesh_wp = icosphere_coarse
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    cases = _precomputed_argument_cases(mesh)
    # The ids are a module-level tuple so a case added to the table without one (or the reverse)
    # fails here rather than silently going untested.
    assert set(cases) == set(_PRECOMPUTED_ARGUMENT_IDS)
    rebuild, from_cache = cases[name]

    rebuilt = comparable_arrays(rebuild())
    cached = comparable_arrays(from_cache())
    assert rebuilt, f"{name}: nothing comparable came back"
    assert len(rebuilt) == len(cached), name
    for left, right in zip(rebuilt, cached, strict=True):
        if left.dtype.kind == "f":
            assert np.allclose(left, right, rtol=1e-5, atol=1e-5), name
        else:
            assert np.array_equal(left, right), name
