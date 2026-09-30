"""Regression tests for ``triwarp.interpolation`` against igl (CPU reference)."""

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import points_to_pyvista, points_to_warp, trimesh_to_pyvista


@pytest.mark.parity("average_onto_faces", "igl", "pyvista")
def test_average_onto_faces(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on both: the vertex-to-face mean, element-wise, no transform.

    VTK spells it ``point_data_to_cell_data``, which for a triangle *is* the corner mean (measured
    1.3e-07 here); the array has to be seeded on the mesh and read back by name rather than passed,
    which is where the answer lives rather than a transform of it.
    """
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(0)

    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    vertex_values_np = rng.uniform(size=mesh_tm.vertices.shape[0])
    face_values_igl = igl.average_onto_faces(faces_np, vertex_values_np)

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    mesh_pv.point_data["field"] = np.ascontiguousarray(vertex_values_np)
    face_values_pv = np.asarray(mesh_pv.point_data_to_cell_data().cell_data["field"])

    vertex_values_wp = wp.array(vertex_values_np, dtype=wp.float32, device=mesh_wp.device)
    face_values_wp = tw.interpolation.average_onto_faces(mesh_wp.indices, vertex_values_wp)
    assert np.allclose(face_values_wp.numpy(), face_values_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(face_values_wp.numpy(), face_values_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("average_onto_vertices", "pymeshlab", "igl", "pyvista")
def test_average_onto_vertices(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A against libigl, Class B against MeshLab's face-to-vertex scalar transfer.

    ``compute_scalar_transfer_face_to_vertex`` reads the *face* scalar attribute and writes the
    *vertex* one rather than taking and returning arrays, so the transform is seeding
    ``f_scalar_array`` on the way in and reading ``vertex_scalar_array()`` on the way out.
    ``areaweight=False`` is load-bearing and is what the benchmark passes: its default weights each
    incident face by area, where this function takes the plain corner mean.

    VTK's ``cell_data_to_point_data`` is Class A as well (measured 8.7e-08) and needs no weighting
    flag: it is the unweighted incident-cell mean.
    """
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(1)

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    face_values_np = rng.uniform(size=mesh_tm.faces.shape[0])
    vertex_values_igl = igl.average_onto_vertices(vertices_np, faces_np, face_values_np)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_np),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            f_scalar_array=np.ascontiguousarray(face_values_np),
        )
    )
    meshset_pml.compute_scalar_transfer_face_to_vertex(areaweight=False)
    vertex_values_pml = np.asarray(meshset_pml.current_mesh().vertex_scalar_array())

    face_values_wp = wp.array(face_values_np, dtype=wp.float32, device=mesh_wp.device)
    vertex_values_wp = tw.interpolation.average_onto_vertices(
        n_vertices, mesh_wp.indices, face_values_wp
    )
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    mesh_pv.cell_data["field"] = np.ascontiguousarray(face_values_np)
    vertex_values_pv = np.asarray(mesh_pv.cell_data_to_point_data().point_data["field"])

    assert np.allclose(vertex_values_wp.numpy(), vertex_values_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_pml, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("average_from_edges_onto_vertices", "igl")
def test_average_from_edges_onto_vertices(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the edge-to-vertex mean, over igl's own halfedge numbering.

    ``igl.orient_halfedges(F)`` supplies the ``(E, oE)`` tables *both* sides consume, so this does
    not compare two edge numberings -- it compares the averaging over one. That is the honest split:
    triwarp has no ``orient_halfedges`` of its own to pair against igl's, and giving each side its
    own numbering would make a mismatch of index conventions look like a mismatch of averages.
    """
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(2)

    n_vertices = mesh_tm.vertices.shape[0]
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    edges_igl, orientation_igl = igl.orient_halfedges(faces_np)
    edges_igl = np.asarray(edges_igl)
    orientation_igl = np.asarray(orientation_igl)
    n_unique_edges = int(edges_igl.max()) + 1
    edge_values_np = rng.uniform(size=n_unique_edges)
    vertex_values_igl = igl.average_from_edges_onto_vertices(
        faces_np, edges_igl, orientation_igl, edge_values_np
    )

    edges_wp = wp.array(edges_igl.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)
    orientation_wp = wp.array(
        orientation_igl.astype(np.int32), dtype=wp.int32, device=mesh_wp.device
    )
    edge_values_wp = wp.array(edge_values_np, dtype=wp.float32, device=mesh_wp.device)
    vertex_values_wp = tw.interpolation.average_from_edges_onto_vertices(
        n_vertices, mesh_wp.indices, edges_wp, orientation_wp, edge_values_wp
    )
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_igl, rtol=1e-5, atol=1e-5)


def _transfer_meshes(device: str) -> tuple[tm.Trimesh, tm.Trimesh]:
    """Build a fine source and a coarse target of one shape: the remesh/decimate situation."""
    del device
    return tm.creation.icosphere(subdivisions=3), tm.creation.icosphere(subdivisions=2)


@pytest.mark.parity("transfer_onto_vertices", "pymeshlab", "pyvista")
def test_transfer_onto_vertices_matches_pymeshlab(device: str):
    """
    Class A: ``transfer_attributes_per_vertex`` is the same barycentric pull, same values.

    pyvista's ``sample`` is the third implementation, through a VTK cell locator, and it agrees to
    **5.96e-08** on a coincident target -- which is the input class where all three do the same work
    and is why the comparison runs there. Two of its conventions decide how it is read: it
    interpolates only where the query lands *inside* a source cell and marks the rest in
    ``vtkValidPointMask`` rather than extrapolating, so the comparison is on the valid set and the
    mask's own count is asserted; and it must not be given
    ``snap_to_closest_point``, which snaps to the nearest source *vertex* rather than the nearest
    point on the surface and is measurably worse.
    """
    source_tm, target_tm = _transfer_meshes(device)
    values_np = np.ascontiguousarray(source_tm.vertices[:, 0] + 2.0, dtype=np.float64)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(source_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(source_tm.faces, dtype=np.int32),
            v_scalar_array=values_np,
        )
    )
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(target_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(target_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.transfer_attributes_per_vertex(
        sourcemesh=0,
        targetmesh=1,
        qualitytransfer=True,
        colortransfer=False,
        upperbound=ml.PercentageValue(50),
    )
    meshset_pml.set_current_mesh(1)
    transferred_pml = meshset_pml.current_mesh().vertex_scalar_array()

    source_vertices_wp = points_to_warp(source_tm.vertices, device)
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    target_vertices_wp = points_to_warp(target_tm.vertices, device)
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=device)
    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, target_vertices_wp
    )
    assert np.allclose(transferred_wp.numpy(), transferred_pml, rtol=1e-4, atol=1e-4)
    assert np.isfinite(distance_wp.numpy()).all()

    # pyvista, on the coincident target where all three transfer the same field.
    source_pv = pv.PolyData(
        np.ascontiguousarray(source_tm.vertices),
        faces=np.hstack(
            [np.full((source_tm.faces.shape[0], 1), 3), np.ascontiguousarray(source_tm.faces)]
        ).ravel(),
    )
    source_pv.point_data["field"] = values_np
    sampled_pv = pv.PolyData(np.ascontiguousarray(source_tm.vertices)).sample(source_pv)
    valid_pv = np.asarray(sampled_pv.point_data["vtkValidPointMask"]).astype(bool)
    self_wp, _self_distance = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, source_vertices_wp
    )
    assert valid_pv.all()  # a coincident target is inside a source cell everywhere
    assert np.allclose(
        self_wp.numpy()[valid_pv],
        np.asarray(sampled_pv.point_data["field"])[valid_pv],
        rtol=1e-4,
        atol=1e-4,
    )


def test_transfer_onto_vertices_reproduces_a_linear_field(device: str):
    """A field that is linear on the source triangles must come back exactly, not blurred."""
    source_tm, target_tm = _transfer_meshes(device)
    direction_np = np.array([0.3, -0.6, 0.74])
    values_np = source_tm.vertices @ direction_np

    source_vertices_wp = points_to_warp(source_tm.vertices, device)
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    # Target vertices projected onto the source surface first, so "linear on the source" holds
    # exactly rather than up to the two spheres' radial gap.
    projected_wp, _distance, _face = tw.proximity.closest_point_on_mesh(
        source_vertices_wp, source_faces_wp, points_to_warp(target_tm.vertices, device)
    )
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=device)
    transferred_wp, _distance = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, projected_wp
    )
    assert np.allclose(
        transferred_wp.numpy(), projected_wp.numpy() @ direction_np, rtol=1e-4, atol=1e-4
    )


def test_transfer_onto_vertices_vec3_field(device: str):
    """The transfer is dtype-generic: a ``wp.vec3`` field (a normal, a colour) works unchanged."""
    source_tm, target_tm = _transfer_meshes(device)
    source_vertices_wp = points_to_warp(source_tm.vertices, device)
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    target_vertices_wp = points_to_warp(target_tm.vertices, device)
    # Transferring the source *positions* must reproduce each target vertex's closest point.
    transferred_wp, _distance = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, source_vertices_wp, target_vertices_wp
    )
    closest_wp, _distance, _face = tw.proximity.closest_point_on_mesh(
        source_vertices_wp, source_faces_wp, target_vertices_wp
    )
    assert np.allclose(transferred_wp.numpy(), closest_wp.numpy(), rtol=1e-4, atol=1e-4)


def test_transfer_onto_vertices_vec2_field(device: str):
    """
    Triwarp against triwarp: a ``wp.vec2`` UV field transfers, and componentwise like two scalars.

    Not a library comparison: pymeshlab's ``transfer_attributes_per_vertex`` moves its own named
    attributes, not an arbitrary buffer, so there is no reference to hand a bare UV pair to. The
    oracle is instead the ``wp.float32`` path this module already compares against pymeshlab --
    the transfer is linear in the field, so a ``vec2`` result must equal its two scalar transfers
    stacked, exactly.

    The dtype is the point: ``transfer_through_operator`` accepts ``wp.vec2`` and is documented as
    the one to prefer *when an operator exists*, which makes this the fallback for the same field.
    It used to raise a bare ``KeyError`` from the overload table instead, while the module's own
    docstring advertised "any Warp dtype closed under scaling and addition".
    """
    source_tm, target_tm = _transfer_meshes(device)
    source_vertices_wp = points_to_warp(source_tm.vertices, device)
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    target_vertices_wp = points_to_warp(target_tm.vertices, device)

    uv_np = np.ascontiguousarray(source_tm.vertices[:, :2], dtype=np.float32)
    uv_wp = wp.array(uv_np, dtype=wp.vec2, device=device)
    transferred_wp, _distance = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, uv_wp, target_vertices_wp
    )
    assert transferred_wp.dtype == wp.vec2
    assert float(np.ptp(transferred_wp.numpy())) > 1e-3, "a constant field compares vacuously"

    for column in range(2):
        scalar_wp = wp.array(
            np.ascontiguousarray(uv_np[:, column]), dtype=wp.float32, device=device
        )
        scalar_transfer_wp, _distance = tw.interpolation.transfer_onto_vertices(
            source_vertices_wp, source_faces_wp, scalar_wp, target_vertices_wp
        )
        assert np.array_equal(transferred_wp.numpy()[:, column], scalar_transfer_wp.numpy())


def test_transfer_onto_vertices_survives_a_sliver_source_face(device: str):
    """
    Class B: against the source field's own linear form, which a sliver must not destroy.

    A decimated or reconstructed source mesh -- the input this transfer exists for -- carries
    slivers, and a face thin enough for Cramer's Gram determinant to cancel to zero in float32 made
    the transferred value ``nan`` while the returned ``distance`` stayed finite, so the confidence
    measure a caller checks reported a healthy hit on a poisoned value. The transform naming the
    class is that the field is linear on the source, so its exact value at the closest point is
    ``direction . closest`` -- computable without any barycentric solve at all, which is what makes
    it an independent oracle rather than a restatement of the kernel.

    The source is a single sliver rather than a mesh so the query has no well-shaped face to fall
    back on; ``height`` is the one that reproduced the ``nan``.
    """
    height = 1e-5
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]], dtype=np.float32)
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    direction_np = np.array([0.3, -0.6, 0.74])
    values_np = np.ascontiguousarray(vertices_np @ direction_np, dtype=np.float32)
    targets_np = np.array([[0.4, 0.2, 0.1], [0.7, -0.3, 0.0]], dtype=np.float32)

    source_vertices_wp = points_to_warp(vertices_np, device)
    source_faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    targets_wp = points_to_warp(targets_np, device)

    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, targets_wp
    )
    closest_wp, _distance, face_wp = tw.proximity.closest_point_on_mesh(
        source_vertices_wp, source_faces_wp, targets_wp
    )
    assert np.all(face_wp.numpy() >= 0), "the query must land on the sliver, not miss it"
    assert np.all(np.isfinite(distance_wp.numpy()))
    assert np.all(np.isfinite(transferred_wp.numpy()))
    expected_np = closest_wp.numpy().astype(np.float64) @ direction_np
    assert np.allclose(transferred_wp.numpy(), expected_np, rtol=1e-4, atol=1e-4)


def test_transfer_onto_vertices_misses_stay_zero(device: str):
    """A target beyond ``max_dist`` keeps the zero fill and reports ``inf``."""
    source_tm, _target_tm = _transfer_meshes(device)
    source_vertices_wp = points_to_warp(source_tm.vertices, device)
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    values_wp = wp.array(
        np.ones(source_tm.vertices.shape[0], dtype=np.float32), dtype=wp.float32, device=device
    )
    targets_wp = wp.array(
        np.array([[0.0, 0.0, 1.0], [100.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, targets_wp, max_dist=0.1
    )
    assert np.isclose(transferred_wp.numpy()[0], 1.0, rtol=1e-5)
    assert transferred_wp.numpy()[1] == 0.0
    assert np.isinf(distance_wp.numpy()[1])


def test_transfer_onto_vertices_length_mismatch(device: str):
    source_vertices_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    source_faces_wp = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    values_wp = wp.zeros(3, dtype=wp.float32, device=device)
    targets_wp = wp.zeros(2, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="one entry per source vertex"):
        tw.interpolation.transfer_onto_vertices(
            source_vertices_wp, source_faces_wp, values_wp, targets_wp
        )


def test_transfer_onto_vertices_empty(device: str):
    source_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    source_faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    values_wp = wp.zeros(0, dtype=wp.float32, device=device)
    targets_wp = wp.zeros(3, dtype=wp.vec3, device=device)
    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, targets_wp
    )
    assert np.array_equal(transferred_wp.numpy(), np.zeros(3, dtype=np.float32))
    assert np.isinf(distance_wp.numpy()).all()


def _scattered_cloud(
    n_source: int = 500, n_query: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a random source cloud carrying ``x ** 2`` and random queries, as the reference does."""
    rng = np.random.default_rng(0)
    source_np = rng.uniform(-1.0, 1.0, size=(n_source, 3))
    return source_np, source_np[:, 0] ** 2, rng.uniform(-1.0, 1.0, size=(n_query, 3))


def _interpolate_pv(
    source_np: np.ndarray,
    values_np: np.ndarray,
    query_np: np.ndarray,
    radius: float,
    sharpness: float,
    n_points: int | None = None,
) -> np.ndarray:
    """``DataSet.interpolate``: ``vtkPointInterpolator`` with a ``vtkGaussianKernel``."""
    source_pv = points_to_pyvista(source_np)
    source_pv.point_data["v"] = np.ascontiguousarray(values_np)
    interpolated_pv = points_to_pyvista(query_np).interpolate(
        source_pv, radius=radius, sharpness=sharpness, n_points=n_points, null_value=0.0
    )
    return np.asarray(interpolated_pv.point_data["v"])


def _interpolate_wp(
    source_np: np.ndarray,
    values_np: np.ndarray,
    query_np: np.ndarray,
    radius: float,
    sharpness: float,
    device: str,
    k: int | None = None,
) -> np.ndarray:
    return tw.interpolation.interpolate_from_points(
        points_to_warp(source_np, device),
        wp.array(
            np.ascontiguousarray(values_np, dtype=np.float32), dtype=wp.float32, device=device
        ),
        points_to_warp(query_np, device),
        radius,
        k=k,
        sharpness=sharpness,
    ).numpy()


@pytest.mark.parametrize(("radius", "sharpness"), [(0.2, 1.0), (0.2, 2.0), (0.2, 8.0), (1.0, 2.0)])
@pytest.mark.parity("interpolate_from_points", "pyvista")
def test_interpolate_from_points_matches_pyvista(device: str, radius: float, sharpness: float):
    """
    Class A, element-wise, against ``DataSet.interpolate`` — the same Gaussian kernel.

    The weight was **recovered** from the reference rather than read from its docs: on a two-source
    probe VTK's answer matches ``exp(-(sharpness * d / radius) ** 2)`` to eight digits and nothing
    else (a ``sharpness * (d / radius) ** 2`` form and a Shepard ``1 / d ** sharpness`` form are
    both ruled out, at 0.30153478 against 0.39651675 and 0.13793103). ``sharpness`` is swept as the
    live axis, and the sweep starts at 1.0 because VTK **clamps it up** to that — its own ``0.5``
    behaves as ``1.0``.
    """
    source_np, values_np, query_np = _scattered_cloud()
    interpolated_pv = _interpolate_pv(source_np, values_np, query_np, radius, sharpness)
    interpolated_wp = _interpolate_wp(
        source_np, values_np, query_np, radius, sharpness, device=device
    )
    # Anti-vacuity: a radius that reached nothing would make both sides the null value everywhere.
    assert (interpolated_pv != 0.0).sum() > 0.8 * query_np.shape[0]
    assert np.allclose(interpolated_wp, interpolated_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("k", [4, 16])
@pytest.mark.parity("interpolate_from_points", "pyvista")
def test_interpolate_from_points_k_nearest_matches_pyvista(device: str, k: int):
    """
    Class A on the k-nearest footprint, against ``interpolate(n_points=k)``.

    Measured: the reference still scales the weights by ``radius`` in this mode — the footprint is
    the only thing ``n_points`` changes, so a row of the same ``k`` neighbours interpolates
    differently at a different radius (0.30153478 / 0.44769209 / 0.48687801 at radius 1 / 2 / 4).
    That is why ``radius`` stays required here.
    """
    source_np, values_np, query_np = _scattered_cloud()
    interpolated_pv = _interpolate_pv(source_np, values_np, query_np, 1.0, 2.0, n_points=k)
    interpolated_wp = _interpolate_wp(source_np, values_np, query_np, 1.0, 2.0, device=device, k=k)
    assert (interpolated_pv != 0.0).all()
    assert np.allclose(interpolated_wp, interpolated_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("interpolate_from_points", "pyvista")
def test_interpolate_from_points_is_exact_at_the_sources(device: str):
    """
    A query on a data point returns that point's value, on both sides — the interpolation property.

    Class A, and it pins a convention rather than a formula: the Gaussian blend of the coincident
    point's *neighbours* would not reproduce the datum (measured 5.39 against 7.0 on a two-source
    probe), so VTK short-circuits a zero distance and triwarp copies that.
    """
    source_np, values_np, _ = _scattered_cloud()
    coincident_np = source_np[:5]
    interpolated_pv = _interpolate_pv(source_np, values_np, coincident_np, 0.2, 2.0)
    interpolated_wp = _interpolate_wp(source_np, values_np, coincident_np, 0.2, 2.0, device=device)
    assert np.allclose(interpolated_pv, values_np[:5], rtol=1e-6, atol=1e-6)
    assert np.allclose(interpolated_wp, values_np[:5], rtol=1e-5, atol=1e-5)


@pytest.mark.parity("interpolate_from_points", "pyvista")
def test_interpolate_from_points_unreached_queries_get_the_null_value(device: str):
    """Class A on the miss case: a radius reaching almost nothing leaves 47 of 50 queries null."""
    source_np, values_np, query_np = _scattered_cloud()
    interpolated_pv = _interpolate_pv(source_np, values_np, query_np, 0.05, 2.0)
    interpolated_wp = _interpolate_wp(source_np, values_np, query_np, 0.05, 2.0, device=device)
    assert (interpolated_pv == 0.0).sum() == 47
    assert np.array_equal(interpolated_wp == 0.0, interpolated_pv == 0.0)
    assert np.allclose(interpolated_wp, interpolated_pv, rtol=1e-5, atol=1e-5)

    # The null value is the caller's, and it is what an unreached query gets.
    far_np = np.array([[100.0, 100.0, 100.0]])
    filled_wp = tw.interpolation.interpolate_from_points(
        points_to_warp(source_np, device),
        wp.array(
            np.ascontiguousarray(values_np, dtype=np.float32), dtype=wp.float32, device=device
        ),
        points_to_warp(far_np, device),
        0.2,
        null_value=-7.0,
    )
    assert filled_wp.numpy()[0] == -7.0


def test_interpolate_from_points_vec3_field(device: str):
    """A ``wp.vec3`` field interpolates componentwise, which is the second registered overload."""
    source_np, values_np, query_np = _scattered_cloud()
    vectors_np = np.column_stack((values_np, 2.0 * values_np, -values_np))
    interpolated_wp = tw.interpolation.interpolate_from_points(
        points_to_warp(source_np, device),
        points_to_warp(vectors_np, device),
        points_to_warp(query_np, device),
        0.2,
    ).numpy()
    scalar_wp = _interpolate_wp(source_np, values_np, query_np, 0.2, 2.0, device=device)
    assert np.allclose(interpolated_wp[:, 0], scalar_wp, rtol=1e-5, atol=1e-5)
    assert np.allclose(interpolated_wp[:, 1], 2.0 * scalar_wp, rtol=1e-5, atol=1e-5)
    assert np.allclose(interpolated_wp[:, 2], -scalar_wp, rtol=1e-5, atol=1e-5)


def test_interpolate_from_points_invalid(device: str):
    source_np, values_np, query_np = _scattered_cloud()
    source_wp = points_to_warp(source_np, device)
    values_wp = wp.array(
        np.ascontiguousarray(values_np, dtype=np.float32), dtype=wp.float32, device=device
    )
    query_wp = points_to_warp(query_np, device)
    with pytest.raises(ValueError, match="one entry per source point"):
        tw.interpolation.interpolate_from_points(source_wp, values_wp[:10], query_wp, 0.2)
    with pytest.raises(ValueError, match="radius must be positive"):
        tw.interpolation.interpolate_from_points(source_wp, values_wp, query_wp, 0.0)
    with pytest.raises(ValueError, match="k must be positive"):
        tw.interpolation.interpolate_from_points(source_wp, values_wp, query_wp, 0.2, k=0)


def test_interpolate_from_points_empty(device: str):
    empty_points = wp.empty(0, dtype=wp.vec3, device=device)
    empty_values = wp.empty(0, dtype=wp.float32, device=device)
    query_wp = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    interpolated_wp = tw.interpolation.interpolate_from_points(
        empty_points, empty_values, query_wp, 0.5, null_value=3.0
    )
    assert interpolated_wp.list() == [3.0]
    assert (
        tw.interpolation.interpolate_from_points(empty_points, empty_values, empty_points, 0.5).size
        == 0
    )
