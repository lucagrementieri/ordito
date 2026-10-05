"""
Regression tests for ``ordito.tangent_space`` against potpourri3d (CPU reference).

Tangent frames are only defined up to a rotation within the tangent plane — each library picks its
own reference halfedge — so the comparisons here are the gauge-invariant ones: normals directly,
frames through the rotation that relates them, and transport angles through the holonomy around each
face, which is what the connection Laplacian's phases encode independently of any frame choice.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from tests.conftest import MESHES


def _vector_heat_solver_pp(mesh_tm: tm.Trimesh) -> pp3d.MeshVectorHeatSolver:
    # ``use_intrinsic_delaunay=False`` so both sides discretize the same triangulation; potpourri3d
    # defaults to flipping to an intrinsic Delaunay triangulation first.
    return pp3d.MeshVectorHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
        use_intrinsic_delaunay=False,
    )


# ---------------------------------------------------------------------------
# vertex_tangent_frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("vertex_tangent_frames", "potpourri3d")
def test_vertex_tangent_frames_match_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (gauge fix): only the normal is comparable directly; the tangents share a rotation.

    Both libraries pick ``basis_x`` by their own convention, so the comparison is that the two
    frames differ by a rotation *about the shared normal* -- one angle for the whole frame,
    checked by reconstructing each side's ``basis_y`` from the other's. Comparing ``basis_x``
    elementwise is what section 6 rules out. The frame is also checked orthonormal and
    right-handed against its own normal at ``1e-5``, tighter than the cross-library ``1e-4``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    basis_x_wp, basis_y_wp, normal_wp = od.tangent_space.vertex_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )
    basis_x_pp, basis_y_pp, normal_pp = (
        np.asarray(basis) for basis in _vector_heat_solver_pp(mesh_tm).get_tangent_frames()
    )

    basis_x, basis_y, normal = basis_x_wp.numpy(), basis_y_wp.numpy(), normal_wp.numpy()
    assert np.allclose(np.linalg.norm(basis_x, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.linalg.norm(basis_y, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose((basis_x * basis_y).sum(axis=1), 0.0, rtol=1e-5, atol=1e-5)
    assert np.allclose((basis_x * normal).sum(axis=1), 0.0, rtol=1e-5, atol=1e-5)
    # Right-handed: basis_x x basis_y points along the normal.
    assert np.allclose((np.cross(basis_x, basis_y) * normal).sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)

    # Normals are frame-independent, so they must agree outright.
    assert np.allclose(normal, normal_pp, rtol=1e-4, atol=1e-4)
    # The two tangent bases span the same plane and differ by a rotation about the normal, so the
    # components of basis_x in potpourri3d's basis are a unit (cos, sin) pair.
    cosine = (basis_x * basis_x_pp).sum(axis=1)
    sine = (basis_x * basis_y_pp).sum(axis=1)
    assert np.allclose(np.hypot(cosine, sine), 1.0, rtol=1e-4, atol=1e-4)
    # ... and the same rotation carries basis_y onto potpourri3d's, i.e. the frames are right-handed
    # in the same orientation rather than mirrored.
    assert np.allclose((basis_y * basis_x_pp).sum(axis=1), -sine, rtol=1e-4, atol=1e-4)
    assert np.allclose((basis_y * basis_y_pp).sum(axis=1), cosine, rtol=1e-4, atol=1e-4)


def test_vertex_tangent_frames_isolated_vertex(device: str) -> None:
    # An unreferenced vertex has no halfedge to take a reference direction from; the frame must
    # still be finite and orthogonal to its (zero-sum, then normalized) normal.
    vertices_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]]),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    basis_x_wp, basis_y_wp, _ = od.tangent_space.vertex_tangent_frames(vertices_wp, faces_wp)

    assert np.isfinite(basis_x_wp.numpy()).all()
    assert np.allclose(np.linalg.norm(basis_y_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# halfedge_tangent_angles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_halfedge_tangent_angles_span_the_rescaled_disk(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    ring_wp, offsets_wp, is_boundary_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )
    angles = od.tangent_space.halfedge_tangent_angles(
        mesh_wp.points, mesh_wp.indices, rings=(ring_wp, offsets_wp, is_boundary_wp)
    ).numpy()

    offsets, ring, is_boundary = offsets_wp.numpy(), ring_wp.numpy(), is_boundary_wp.numpy()
    corner_angles = od.triangles.face_angles(mesh_wp.points, mesh_wp.indices).numpy()
    for vertex in range(n_vertices):
        ring_halfedges = ring[offsets[vertex] : offsets[vertex + 1]]
        if len(ring_halfedges) == 0:
            continue
        ring_angles = angles[ring_halfedges]
        corners = np.array([corner_angles[h // 3, h % 3] for h in ring_halfedges])
        full_turn = np.pi if is_boundary[vertex] else 2 * np.pi
        scale = full_turn / corners.sum()

        # The ring starts at the reference direction, and each step is the rescaled corner angle.
        assert ring_angles[0] == 0.0
        expected = scale * np.concatenate([[0.0], np.cumsum(corners)[:-1]])
        assert np.allclose(ring_angles, expected, rtol=1e-5, atol=1e-5)
        # The last corner closes the disk exactly: a half-turn at a boundary vertex, a full turn in
        # the interior, whatever the vertex's angle defect was.
        assert np.isclose(ring_angles[-1] + scale * corners[-1], full_turn, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# halfedge_transport_angles
# ---------------------------------------------------------------------------


# Only meshes without right-angle triangles: an edge whose two opposite angles are both 90 degrees
# has cotangent weight exactly zero, which erases its phase from the connection Laplacian entirely.
# In ``cave_cube`` and ``half_torus`` every face carries such an edge (both are quad grids split by
# a diagonal), so no entry is left to read a transport angle out of — a limit of the oracle, not of
# the computation. Those meshes are covered by the round-trip test below.
@pytest.mark.parity(
    "halfedge_transport_angles",
    "potpourri3d",
    benchmarked=False,
    reason="the transport angles are only reachable through get_connection_laplacian, which "
    "builds a whole MeshVectorHeatSolver -- a halfedge mesh, a cotangent Laplacian and a "
    "factorization -- and the phases then have to be divided out of it weight by weight. A row "
    "would price that solver build under this group's name; the holonomy is still comparable.",
)
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_halfedge_transport_angle_holonomy_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (holonomy): a single transport angle is gauge-dependent; the loop product is not.

    The reference rotations cancel around a closed loop, so the face holonomy is the only
    comparable quantity -- extracted from potpourri3d's connection Laplacian by dividing out
    its real cotangent weights. The comment above records why ``cave_cube`` and ``half_torus``
    cannot be used at all.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rho = od.tangent_space.halfedge_transport_angles(mesh_wp.points, mesh_wp.indices).numpy()

    # A single transport angle depends on both endpoints' reference directions, but the holonomy
    # around a face does not — the reference rotations cancel around a closed loop — so that is what
    # can be compared across libraries. potpourri3d's connection Laplacian holds
    # ``w_ij * exp(i * rho_ji)``; dividing by its real cotangent Laplacian, which carries the same
    # weights, leaves the phase (up to one global sign convention, which a triple product cubes).
    faces = np.asarray(mesh_tm.faces)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    connection_pp = _vector_heat_solver_pp(mesh_tm).get_connection_laplacian().tocsr()
    cotangent_pp = pp3d.cotan_laplacian(vertices_np, faces_np).tocsr()

    corners = ((faces[:, 0], faces[:, 1]), (faces[:, 1], faces[:, 2]), (faces[:, 2], faces[:, 0]))
    weights = [np.asarray(cotangent_pp[row, column]).ravel() for row, column in corners]
    usable = np.logical_and.reduce([np.abs(weight) > 1e-9 for weight in weights])
    assert usable.any()

    phase_pp = np.ones(int(usable.sum()), dtype=complex)
    for (row, column), weight in zip(corners, weights, strict=True):
        phase_pp *= np.asarray(connection_pp[row, column]).ravel()[usable] / weight[usable]
    holonomy_pp = np.angle(phase_pp)
    holonomy_wp = (rho[0::3] + rho[1::3] + rho[2::3])[usable]

    assert np.allclose(
        np.angle(np.exp(1j * (holonomy_wp + holonomy_pp))), 0.0, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("mesh_name", MESHES)
def test_halfedge_transport_angles_are_antisymmetric(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    twins = od.halfedge.halfedge_twins(mesh_wp.indices).numpy()
    rho = od.tangent_space.halfedge_transport_angles(mesh_wp.points, mesh_wp.indices).numpy()

    # Transporting a vector across an edge and back is the identity: rho_ij = -rho_ji (mod 2*pi).
    interior = np.flatnonzero(twins >= 0)
    round_trip = np.angle(np.exp(1j * (rho[interior] + rho[twins[interior]])))
    assert np.allclose(round_trip, 0.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("face_tangent_frames", "igl")
def test_face_tangent_frames_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on all three vectors -- the one tangent-space comparison in the package that can be.

    Every other frame in this module is **gauge-dependent**: a vertex frame is pinned to "the first
    halfedge of the ring", which depends on face ordering, so it agrees with another library's only
    up to a rotation about the normal -- which is why the potpourri3d comparison above uses
    invariants. A *face* frame is not: "the first edge of the face" is a property of the face table
    itself, and ``igl.local_basis`` uses the same rule, so ``basis_x``, ``basis_y`` and the normal
    are each compared element-wise with no transform.

    That makes this the test that pins the **convention** rather than a property: were ordito to
    switch to, say, the longest edge or a projected global axis, the frame would still be
    orthonormal and still span the face, and only this comparison would notice. Since igl's frame
    is unit, right-handed, in the face and along the first edge, element-wise agreement pins all
    four properties on ordito's side too.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    basis_x_igl, basis_y_igl, normal_igl = igl.local_basis(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )

    basis_x_wp, basis_y_wp, normal_wp = od.tangent_space.face_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )

    assert np.allclose(basis_x_wp.numpy(), basis_x_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(basis_y_wp.numpy(), basis_y_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(normal_wp.numpy(), normal_igl, rtol=1e-5, atol=1e-5)


def test_face_tangent_frames_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    basis_x, basis_y, normals = od.tangent_space.face_tangent_frames(vertices_wp, faces_wp)
    assert basis_x.shape == (0,)
    assert basis_y.shape == (0,)
    assert normals.shape == (0,)
