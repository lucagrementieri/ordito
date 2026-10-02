from typing import cast

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import pyvista as pv
import scipy.spatial
import trimesh.geometry as tm_geometry
import trimesh.points as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito.grouping as od_grouping
import ordito.neighbors as od_neighbors
import ordito.points as od
import ordito.typing as odt
from tests.comparisons import assert_same_up_to_sign
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_indices_to_numpy,
    points_to_meshlib,
    points_to_open3d,
    points_to_pymeshlab,
    points_to_torch,
    points_to_warp,
    warp_empty,
)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic near-uniform points on the unit sphere (unique pairwise distances)."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _random_points(n: int, seed: int) -> np.ndarray:
    """Fixed-seed float32 point cloud."""
    return np.random.default_rng(seed).standard_normal((n, 3)).astype(np.float32)


@pytest.mark.parity("point_plane_distance", "trimesh", "pyvista")
@pytest.mark.parity("half_space_mask", "pyvista")
def test_point_plane_distance(device: str) -> None:
    """
    Class A: signed distances against trimesh and against VTK's plane implicit function.

    The normal is not unit length here, which is what makes the normalization part of the claim
    rather than an assumption both sides happen to share.

    pyvista's ``compute_implicit_distance`` evaluates the same signed dot product, and the
    comparison is made against the **exact** float64 answer as well as against ordito's so the
    residuals are attributable: measured 1.27e-07 for pyvista and 2.11e-07 for ordito on a
    1 000-point cloud, i.e. each at its own storage precision rather than either being wrong.

    ``half_space_mask`` is claimed here too, because pyvista reaches it the only way it can -- one
    host threshold on that distance field -- and asserting the threshold agrees is the whole content
    of the claim. It is the *strict* boundary that makes this worth pinning; the meshlib comparison
    below covers the same convention from the other side.

    Two things about the ``pv.Plane``: its implicit function is unbounded but the object carries an
    extent, so it is built large enough to span the cloud; and it is a *parameter* rather than an
    input, which is why the benchmark row builds it outside its timed callable.
    """
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((50, 3))
    plane_normal_np = rng.standard_normal(3)
    plane_origin_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np, plane_origin_np)

    points_wp = points_to_warp(points_np, device)
    distances_wp = od.point_plane_distance(
        points_wp, wp.vec3(*plane_normal_np.tolist()), wp.vec3(*plane_origin_np.tolist())
    )

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)

    # pyvista, against the exact answer as well as against ordito's.
    unit_np = plane_normal_np / np.linalg.norm(plane_normal_np)
    exact_np = (points_np - plane_origin_np) @ unit_np
    extent = 20.0 * float(np.abs(points_np).max())
    plane_pv = pv.Plane(
        center=plane_origin_np.tolist(), direction=unit_np.tolist(), i_size=extent, j_size=extent
    )
    distances_pv = np.asarray(
        pv.PolyData(points_np).compute_implicit_distance(plane_pv)["implicit_distance"]
    )
    assert np.abs(distances_pv - exact_np).max() < 1e-6
    assert np.allclose(distances_wp.numpy(), distances_pv, rtol=1e-5, atol=1e-5)

    # half_space_mask is that field thresholded, which is pyvista's only route to it.
    mask_wp = od.half_space_mask(
        points_wp, wp.vec3(*plane_normal_np.tolist()), wp.vec3(*plane_origin_np.tolist())
    )
    assert 0 < int(mask_wp.numpy().sum()) < points_np.shape[0]  # both branches present
    assert np.array_equal(mask_wp.numpy(), distances_pv > 0.0)


@pytest.mark.parity("half_space_mask", "meshlib")
def test_half_space_mask_matches_meshlib(device: str) -> None:
    """
    Class A: the same half-space selection as ``findHalfSpacePoints``, boundary convention included.

    meshlib's plane is ``dot(n, x) = d`` and ordito's is a normal plus a point on it, so the
    transform is ``d = dot(n, origin)`` -- an argument mapping, not a value one, which is why this
    is Class A rather than B. The normal is deliberately **not** unit length: only the sign of the
    projection can matter, and a comparison at unit scale would not show that.

    The **strict** boundary is the part worth pinning, since it is the one convention two
    implementations can silently disagree on. Measured against the wheel: with the plane ``z = 1``,
    a point at exactly ``z = 1`` is in neither half for either library. The invariant that follows
    is asserted directly -- the masks for opposite normals are disjoint and together cover every
    point off the plane.
    """
    points_np = _random_points(300, seed=3)
    plane_normal_np = np.array([0.4, -1.7, 0.9], dtype=np.float64)
    plane_origin_np = np.array([0.1, 0.2, -0.3], dtype=np.float64)

    plane_ml = mm.Plane3f(
        mm.Vector3f(*plane_normal_np.tolist()), float(plane_normal_np @ plane_origin_np)
    )
    mask_ml = meshlib_bitset_to_numpy(
        mm.findHalfSpacePoints(points_to_meshlib(points_np), plane_ml), points_np.shape[0]
    )

    points_wp = points_to_warp(points_np, device)
    normal_wp = wp.vec3(*plane_normal_np.tolist())
    origin_wp = wp.vec3(*plane_origin_np.tolist())
    mask_wp = od.half_space_mask(points_wp, normal_wp, origin_wp)

    assert np.array_equal(mask_wp.numpy(), mask_ml)
    assert 0 < int(mask_ml.sum()) < points_np.shape[0]  # both answers present, so not vacuous

    # Opposite normals partition the points off the plane, which the strict test is what makes true.
    opposite_wp = od.half_space_mask(points_wp, wp.vec3(*(-plane_normal_np).tolist()), origin_wp)
    assert not np.any(mask_wp.numpy() & opposite_wp.numpy())
    assert np.all(mask_wp.numpy() | opposite_wp.numpy())

    # A point exactly on the plane is in neither half, for both libraries and both normals.
    on_plane_np = np.ascontiguousarray(plane_origin_np[None, :], dtype=np.float32)
    on_plane_wp = points_to_warp(on_plane_np, device)
    assert not bool(od.half_space_mask(on_plane_wp, normal_wp, origin_wp).numpy()[0])
    assert not bool(
        od.half_space_mask(on_plane_wp, wp.vec3(*(-plane_normal_np).tolist()), origin_wp).numpy()[0]
    )
    assert not bool(
        meshlib_bitset_to_numpy(
            mm.findHalfSpacePoints(points_to_meshlib(on_plane_np), plane_ml), 1
        )[0]
    )


def test_half_space_mask_defaults_to_the_origin(device: str) -> None:
    """
    Not a library comparison: the ``plane_origin=None`` default and the empty input.

    ``None`` means the world origin, matching
    [`point_plane_distance`][ordito.points.point_plane_distance], so the mask is then the sign of
    ``dot(n, p)`` alone.
    """
    points_np = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    points_wp = points_to_warp(points_np, device)
    assert np.array_equal(
        od.half_space_mask(points_wp, wp.vec3(0.0, 0.0, 1.0)).numpy(),
        np.array([True, False, False]),
    )

    empty_wp = od.half_space_mask(warp_empty(0, wp.vec3, device), wp.vec3(1.0, 0.0, 0.0))
    assert empty_wp.shape == (0,)
    assert empty_wp.dtype == wp.bool


def test_centroid(device: str) -> None:
    points_np = _random_points(128, seed=14)
    points_wp = points_to_warp(points_np, device)
    centroid_wp = od.centroid(points_wp)
    assert np.allclose(centroid_wp.numpy()[0], points_np.mean(axis=0), rtol=1e-5, atol=1e-5)


def test_gram_matrix(device: str) -> None:
    # 200 = 3 * 64 + 8 exercises the multi-tile reduction and remainder path.
    points_np = _random_points(200, seed=10)
    points_wp = points_to_warp(points_np, device)
    gram_np = points_np.T @ points_np
    assert np.allclose(od.gram_matrix(points_wp).numpy()[0], gram_np, rtol=1e-4, atol=1e-4)


def test_gram_matrix_empty(device: str) -> None:
    points_wp = warp_empty(0, wp.vec3, device)
    assert np.allclose(od.gram_matrix(points_wp).numpy()[0], np.zeros((3, 3)))


@pytest.mark.parity("fit_line", "trimesh")
def test_fit_line(device: str) -> None:
    """
    Class B (sign fix): the major axis against ``trimesh.points.major_axis``, up to direction.

    An eigenvector is defined only up to sign, so the comparison goes through
    [`tests.comparisons.assert_same_up_to_sign`][]. The cloud is elongated 1000:1 so the axis
    itself is well determined -- on an isotropic cloud there would be nothing to compare.
    """
    rng = np.random.default_rng(2)
    # points strongly elongated along a known direction so the major axis
    # is well-defined and robust to the SVD sign convention.
    direction_np = rng.standard_normal(3)
    direction_np /= np.linalg.norm(direction_np)
    t_np = rng.uniform(-10.0, 10.0, size=200)
    points_np = t_np[:, None] * direction_np[None, :] + 0.01 * rng.standard_normal((200, 3))

    axis_tm = tm.major_axis(points_np)

    points_wp = points_to_warp(points_np, device)
    axis_wp = od.fit_line(points_wp)

    # axis is direction-only: compare up to sign against trimesh and the
    # ground-truth direction.
    assert np.isclose(np.abs(np.dot(np.array(axis_wp), axis_tm)), 1.0, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(axis_wp), direction_np)), 1.0, atol=1e-3)


def test_centered_covariance(device: str) -> None:
    points_np = _random_points(200, seed=11)
    points_wp = points_to_warp(points_np, device)
    centered_np = points_np - points_np.mean(axis=0)
    scatter_np = centered_np.T @ centered_np
    cov_wp = od.centered_covariance(points_wp)
    assert np.allclose(cov_wp.numpy()[0], scatter_np, rtol=1e-4, atol=1e-4)


def test_centered_covariance_precomputed_center(device: str) -> None:
    points_np = _random_points(150, seed=12)
    points_wp = points_to_warp(points_np, device)
    mean_np = points_np.mean(axis=0)
    center_wp = points_to_warp(mean_np.reshape(1, 3), device)
    centered_np = points_np - mean_np
    scatter_np = centered_np.T @ centered_np
    cov_wp = od.centered_covariance(points_wp, center=center_wp)
    assert np.allclose(cov_wp.numpy()[0], scatter_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("principal_axes", "pyvista")
def test_principal_axes(device: str) -> None:
    """Class A up to the free sign of the first two axes: rows, eigenvalues and centroid."""
    rng = np.random.default_rng(7)
    # A well-separated spectrum, so every axis is individually determined. Offset from the origin
    # too, which is what separates a centred fit from fit_line's uncentred one.
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])) + 5.0
    points_wp = points_to_warp(points_np, device)

    rotation_wp, eigenvalues_wp, centroid_wp = od.principal_axes(points_wp)
    rotation_np = np.array(rotation_wp).reshape(3, 3)

    axes_pv = pv.principal_axes(points_np)
    for row in range(3):
        assert np.isclose(abs(float(np.dot(rotation_np[row], axes_pv[row]))), 1.0, atol=1e-5)

    # A proper, orthonormal rotation -- the determinant is what pins the third row's sign.
    assert np.allclose(rotation_np @ rotation_np.T, np.eye(3), atol=1e-5)
    assert np.isclose(float(np.linalg.det(rotation_np)), 1.0, atol=1e-5)

    # Eigenvalues are of the scatter matrix, so they carry no 1/n.
    scatter_np = np.cov(points_np.T) * (points_np.shape[0] - 1)
    assert np.allclose(
        np.array(eigenvalues_wp), np.linalg.eigvalsh(scatter_np)[::-1], rtol=1e-5, atol=1e-5
    )
    assert np.allclose(np.array(centroid_wp), points_np.mean(axis=0), rtol=1e-5, atol=1e-5)


def _point_accumulator_ml(points_np: np.ndarray) -> mm.PointAccumulator:
    """
    Accumulate a NumPy cloud into a ``PointAccumulator``, the object behind both fitting oracles.

    ``accumulatePoints`` takes a ``std_vector_Vector3_float``, so the fill is a per-point Python
    loop -- fine at test size, and the reason neither of these pairs carries a benchmark row of its
    own beyond the ones already there.
    """
    accumulator_ml = mm.PointAccumulator()
    points_ml = mm.std_vector_Vector3_float()
    for point_np in np.asarray(points_np, dtype=np.float64):
        points_ml.append(mm.Vector3f(*point_np.tolist()))
    mm.accumulatePoints(accumulator_ml, points_ml)
    return accumulator_ml


@pytest.mark.parity("fit_plane", "meshlib")
def test_fit_plane_matches_meshlib(device: str) -> None:
    """
    Class B (sign gauge only): ``PointAccumulator.getBestPlanef`` is the same least-squares plane.

    The transform is the *representation*: MeshLib returns a ``Plane3f`` -- a unit normal and an
    offset ``d`` -- where ordito returns a centroid and a normal, so the comparison is the normal
    up to sign (both are the smallest-eigenvalue covariance eigenvector, whose direction neither
    library fixes) plus the plane equation ``n . c == d`` evaluated at ordito's centroid.

    Measured on an anisotropic 500-point cloud: ``|dot| = 1.0000000`` to seven digits, which is
    tighter than the trimesh and MeshLab pairings above and is why this is the one asserted at
    ``1e-6``. The offset check is what makes it a plane comparison rather than a direction one --
    a fit that found the right orientation through the wrong point would pass the dot alone.
    """
    rng = np.random.default_rng(3)
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])) + np.array(
        [2.0, -1.0, 0.5]
    )
    points_wp = points_to_warp(points_np, device)
    normal_wp, centroid_wp = od.fit_plane(points_wp)

    plane_ml = _point_accumulator_ml(points_np).getBestPlanef()
    normal_ml = np.array([plane_ml.n.x, plane_ml.n.y, plane_ml.n.z], dtype=np.float64)

    assert np.isclose(np.linalg.norm(normal_ml), 1.0, atol=1e-6)  # non-vacuity: a real plane
    assert np.isclose(
        abs(float(np.dot(np.array(centroid_wp), normal_ml)) - plane_ml.d), 0.0, atol=1e-4
    )
    assert np.isclose(abs(float(np.dot(np.array(normal_wp), normal_ml))), 1.0, atol=1e-6)


@pytest.mark.parity("principal_axes", "meshlib")
def test_principal_axes_matches_meshlib(device: str) -> None:
    """
    Class B: the same eigen-decomposition, reported in the **opposite** order.

    ``getCenteredCovarianceEigen`` is an out-parameter call -- it takes a centroid, a ``Matrix3``
    and an eigenvalue vector to fill and returns a ``bool`` -- and it orders its eigenvalues
    **ascending** where ordito orders them descending, so the named transform is a reversal plus
    the per-axis sign every eigenvector comparison needs. Measured on a 500-point anisotropic cloud
    the eigenvalues agree to 4 digits (4542.72 / 538.20 / 19.62 on both sides, reversed) and each
    axis matches to ``|dot| = 1``.

    Its eigenvalues are of the *scatter* matrix, carrying no ``1/n``, which is ordito's convention
    too -- so this pins that convention against a second library, where the pyvista pairing above
    pins the axes alone.
    """
    rng = np.random.default_rng(4)
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])) + np.array(
        [2.0, -1.0, 0.5]
    )
    points_wp = points_to_warp(points_np, device)
    rotation_wp, eigenvalues_wp, centroid_wp = od.principal_axes(points_wp)

    centroid_ml = mm.Vector3d()
    eigenvectors_ml = mm.Matrix3d()
    eigenvalues_ml = mm.Vector3d()
    assert _point_accumulator_ml(points_np).getCenteredCovarianceEigen(
        centroid_ml, eigenvectors_ml, eigenvalues_ml
    )

    axes_ml = np.array(
        [
            [eigenvectors_ml.x.x, eigenvectors_ml.x.y, eigenvectors_ml.x.z],
            [eigenvectors_ml.y.x, eigenvectors_ml.y.y, eigenvectors_ml.y.z],
            [eigenvectors_ml.z.x, eigenvectors_ml.z.y, eigenvectors_ml.z.z],
        ]
    )[::-1]  # ascending -> descending
    values_ml = np.array([eigenvalues_ml.x, eigenvalues_ml.y, eigenvalues_ml.z])[::-1]

    assert values_ml[0] > values_ml[1] > values_ml[2] > 0.0  # non-vacuity: a separated spectrum
    assert np.allclose(
        np.array(centroid_wp), [centroid_ml.x, centroid_ml.y, centroid_ml.z], atol=1e-4
    )
    assert np.allclose(np.array(eigenvalues_wp), values_ml, rtol=1e-4, atol=1e-4)
    rotation_np = np.array(rotation_wp).reshape(3, 3)
    for row in range(3):
        assert np.isclose(abs(float(np.dot(rotation_np[row], axes_ml[row]))), 1.0, atol=1e-4)


def test_principal_axes_is_not_fit_line(device: str) -> None:
    """
    Pin the distinction the two functions exist to keep apart.

    ``fit_line`` is ``trimesh.points.major_axis``: a *singular-value-weighted sum of all three*
    right singular vectors of the uncentered matrix. ``principal_axes`` is the leading eigenvector
    of the centred covariance. On an ordinary cloud these are different directions, which is why the
    second was added rather than the first being changed.

    Note what the weighted sum costs: because it mixes all three vectors, its value depends on each
    one's *sign*, and no SVD fixes those. So ``fit_line`` reproduces its own oracle only where one
    singular value dominates -- measured ``|dot|`` 1.000 on a 1000:1 needle against **0.992** on the
    moderate cloud below. That is a property of the quantity, not a defect in either port.
    """
    rng = np.random.default_rng(3)
    moderate_np = rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])
    moderate_wp = points_to_warp(moderate_np, device)

    leading_np = np.linalg.eigh(np.cov(moderate_np.T))[1][:, -1]
    first_axis_np = np.array(od.principal_axes(moderate_wp)[0]).reshape(3, 3)[0]

    # principal_axes is the leading eigenvector ...
    assert np.isclose(abs(float(np.dot(first_axis_np, leading_np))), 1.0, atol=1e-4)
    # ... and the weighted major axis is a measurably different direction, on both sides.
    assert abs(float(np.dot(tm.major_axis(moderate_np), leading_np))) < 0.99
    assert abs(float(np.dot(np.array(od.fit_line(moderate_wp)), leading_np))) < 0.99

    # On a needle every definition coincides, which is why the difference went unnoticed.
    needle_np = (rng.uniform(-10.0, 10.0, 200)[:, None] * np.array([0.3, 0.5, 0.8])) + (
        0.01 * rng.standard_normal((200, 3))
    )
    needle_wp = points_to_warp(needle_np, device)
    needle_first_np = np.array(od.principal_axes(needle_wp)[0]).reshape(3, 3)[0]
    assert np.isclose(abs(float(np.dot(needle_first_np, tm.major_axis(needle_np)))), 1.0, atol=1e-3)
    assert np.isclose(
        abs(float(np.dot(np.array(od.fit_line(needle_wp)), tm.major_axis(needle_np)))),
        1.0,
        atol=1e-3,
    )


def test_principal_axes_degenerate_spectrum(device: str) -> None:
    """Class C (one axis only): a near-equal eigenpair leaves its own plane arbitrary."""
    rng = np.random.default_rng(5)
    # 1000:1 needle: axes 2 and 3 both sit in the noise, so their split is not determined.
    points_np = (rng.uniform(-10.0, 10.0, 200)[:, None] * np.array([0.3, 0.5, 0.8])) + (
        0.01 * rng.standard_normal((200, 3))
    )
    points_wp = points_to_warp(points_np, device)
    rotation_np = np.array(od.principal_axes(points_wp)[0]).reshape(3, 3)
    axes_pv = pv.principal_axes(points_np)

    assert np.isclose(abs(float(np.dot(rotation_np[0], axes_pv[0]))), 1.0, atol=1e-5)
    # The degenerate pair still spans the same plane, even though the axes within it differ.
    assert np.isclose(
        abs(float(np.dot(rotation_np[0], np.cross(axes_pv[1], axes_pv[2])))), 1.0, atol=1e-4
    )
    assert np.allclose(rotation_np @ rotation_np.T, np.eye(3), atol=1e-5)


def test_principal_axes_empty_and_single(device: str) -> None:
    """An empty cloud gives the identity frame; a single point gives a frame at that point."""
    empty_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    rotation_wp, eigenvalues_wp, centroid_wp = od.principal_axes(empty_wp)
    assert np.array_equal(np.array(rotation_wp).reshape(3, 3), np.eye(3))
    assert np.array_equal(np.array(eigenvalues_wp), np.zeros(3))
    assert np.array_equal(np.array(centroid_wp), np.zeros(3))

    single_wp = wp.array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    rotation_wp, _eigenvalues_wp, centroid_wp = od.principal_axes(single_wp)
    assert np.isclose(float(np.linalg.det(np.array(rotation_wp).reshape(3, 3))), 1.0, atol=1e-6)
    assert np.allclose(np.array(centroid_wp), np.array([1.0, 2.0, 3.0]))


@pytest.mark.parity("fit_plane", "trimesh", "pymeshlab")
def test_fit_plane(device: str) -> None:
    """
    Class B against both references, each needing one named transform.

    trimesh's ``plane_fit`` returns the centroid and normal directly. **MeshLab** returns a *dict*,
    so the transform is the ``"fitting_plane_normal"`` key -- the rotation matrix and average error
    it also builds are work ordito does not do. Two of its quirks are load-bearing and match what
    the benchmark passes: it raises ``Cannot compute rotation: there is no selection`` unless
    something is selected, so ``set_selection_all`` is how it is told to fit *all* the points; and
    it takes a face-less MeshSet, which is what these bare points are. Both references leave the
    normal
    **sign** free (it is a covariance eigenvector), so all three are compared up to sign.
    """
    rng = np.random.default_rng(3)
    points_np = rng.standard_normal((80, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    meshset_pml = points_to_pymeshlab(points_np)
    meshset_pml.set_selection_all()
    normal_pml = np.asarray(
        meshset_pml.compute_matrix_by_fitting_to_plane(targetplane="XY plane")[
            "fitting_plane_normal"
        ],
        dtype=np.float64,
    )
    normal_pml /= np.linalg.norm(normal_pml)

    points_wp = points_to_warp(points_np, device)
    normal_wp, centroid_wp = od.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    # normal is sign-ambiguous: compare up to sign.
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_pml)), 1.0, atol=1e-4)


@pytest.mark.parity("fit_plane", "pyvista")
def test_fit_plane_normal_matches_pyvista(device: str) -> None:
    """
    Class B, and the transform is a **projection**: only the normal is comparable.

    ``pv.fit_plane_to_points(..., return_meta=True)`` returns ``(plane, centre, normal)`` and its
    ``centre`` is *not* the centroid -- VTK re-centres the returned plane on the middle of the
    fitted patch, measured ``[5.018, 4.734, 5.008]`` against the cloud's own ``[4.784, 4.949,
    4.998]`` on the fixture below. So comparing the origin would fail for a correct implementation;
    the normal is the shared quantity, up to sign, and it agrees to ``|dot| = 1.0000004``.

    Both the centre and the normal come back **float32** even though pyvista stores points in
    float64, which is why the tolerance here is ``1e-4`` rather than tighter.
    """
    rng = np.random.default_rng(7)
    # Nearly planar and off the origin: the normal is well determined, and the centre pyvista
    # returns is not the centroid.
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.05])) + 5.0

    _plane_pv, centre_pv, normal_pv = cast(
        "tuple[pv.PolyData, np.ndarray, np.ndarray]",
        pv.fit_plane_to_points(points_np, return_meta=True),
    )

    points_wp = points_to_warp(points_np, device)
    normal_wp, centroid_wp = od.fit_plane(points_wp)

    assert np.isclose(
        abs(float(np.dot(np.array(normal_wp), np.asarray(normal_pv)))), 1.0, atol=1e-4
    )
    # The origins are different quantities; asserting that keeps the projection above honest.
    assert np.allclose(np.array(centroid_wp), points_np.mean(axis=0), rtol=1e-4, atol=1e-4)
    assert not np.allclose(np.asarray(centre_pv), points_np.mean(axis=0), atol=1e-2)


@pytest.mark.parametrize(
    "normal",
    [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.95, 0.1, 0.2), (0.3, 0.9, 0.1), (0.0, -2.0, 0.0)],
)
def test_plane_basis_is_right_handed_and_orthonormal(normal: tuple[float, float, float]) -> None:
    """
    All four properties its docstring promises, over both branches of the axis choice.

    The implementation picks its seed axis by whether ``|n_x| > 0.9``, so the parametrisation spans
    both: two normals take the ``x`` seed and three the ``y`` seed. A non-unit normal is included
    because the signature says it need not be normalized. Measured residuals at most 3.7e-08.
    """
    u_wp, v_wp = od.plane_basis(wp.vec3(*normal))

    u_np = np.array(list(u_wp))
    v_np = np.array(list(v_wp))
    unit_np = np.asarray(normal) / np.linalg.norm(normal)

    assert np.allclose(np.linalg.norm(u_np), 1.0, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.linalg.norm(v_np), 1.0, rtol=1e-6, atol=1e-6)
    assert np.allclose([u_np @ v_np, u_np @ unit_np, v_np @ unit_np], 0.0, rtol=1e-6, atol=1e-6)
    # Right-handed: (u, v, n) in that order, so u x v is the normal rather than its negation.
    assert np.allclose(np.cross(u_np, v_np), unit_np, rtol=1e-6, atol=1e-6)


def test_covariance(device: str) -> None:
    points_np = _random_points(200, seed=13)
    points_wp = points_to_warp(points_np, device)
    cov_np = np.cov(points_np.T, ddof=1)
    assert np.allclose(od.covariance(points_wp).numpy()[0], cov_np, rtol=1e-4, atol=1e-4)


def test_covariance_too_few_points_raises(device: str) -> None:
    points_wp = wp.zeros(1, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="ddof"):
        od.covariance(points_wp)


def test_fit_line_large(device: str) -> None:
    """
    Class B (sign fix): the same comparison at 5 000 points, past the tiled reduction's tile size.

    ``5000 = 78 * 64 + 8`` exercises both the multi-tile path and its remainder branch, which
    the 200-point test above does not reach at all.
    """
    # n far larger than TILE_1D (64) to exercise the multi-tile reduction path
    # and the remainder branch (5000 = 78 * 64 + 8).
    rng = np.random.default_rng(5)
    direction_np = rng.standard_normal(3)
    direction_np /= np.linalg.norm(direction_np)
    t_np = rng.uniform(-10.0, 10.0, size=5000)
    points_np = t_np[:, None] * direction_np[None, :] + 0.01 * rng.standard_normal((5000, 3))

    axis_tm = tm.major_axis(points_np)

    points_wp = points_to_warp(points_np, device)
    axis_wp = od.fit_line(points_wp)

    assert np.isclose(np.abs(np.dot(np.array(axis_wp), axis_tm)), 1.0, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(axis_wp), direction_np)), 1.0, atol=1e-3)


def test_fit_plane_large(device: str) -> None:
    """
    Class B (sign fix on the normal): ``trimesh.points.plane_fit`` at 5 000 points.

    Same multi-tile motivation as the line fit above. The centroid needs no sign fix and is
    compared directly, which is what separates a reduction bug from an eigenvector one.
    """
    # n far larger than TILE_1D (64) to exercise the multi-tile reduction path.
    rng = np.random.default_rng(6)
    points_np = rng.standard_normal((5000, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    points_wp = points_to_warp(points_np, device)
    normal_wp, centroid_wp = od.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)


def test_point_plane_distance_no_origin(device: str) -> None:
    """
    Class A: the default-origin overload, where the plane passes through the world origin.

    A separate test because the default is a different code path, not a value the caller could
    pass -- and an implementation defaulting to the *centroid* instead would pass the test
    above and fail this one.
    """
    rng = np.random.default_rng(1)
    points_np = rng.standard_normal((30, 3))
    plane_normal_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np)

    points_wp = points_to_warp(points_np, device)
    distances_wp = od.point_plane_distance(points_wp, wp.vec3(*plane_normal_np.tolist()))

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("radial_sort", "trimesh")
def test_radial_sort(device: str) -> None:
    """
    Class A: the angular order against a numpy ``arctan2`` argsort, index for index.

    The angles are evenly spaced by construction, which is what makes an exact index comparison
    sound: with random angles two neighbours can differ by less than ``float32`` resolves and
    the orders diverge legitimately (section 6's note on ``lexsort`` and float ties).
    """
    rng = np.random.default_rng(7)
    n = 256
    # evenly spaced angles so the radial order is unambiguous and float32 cannot
    # flip the order of neighboring points relative to the float64 reference.
    theta_np = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius_np = rng.uniform(0.1, 2.0, n)
    points_np = np.column_stack(
        (np.cos(theta_np) * radius_np, np.sin(theta_np) * radius_np, np.zeros(n))
    )
    points_np = points_np[rng.permutation(n)]
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([0.0, 0.0, 1.0])

    ordered_tm = tm.radial_sort(points_np, origin=origin_np, normal=normal_np)

    points_wp = points_to_warp(points_np, device)
    ordered_wp = od.radial_sort(
        points_wp, wp.vec3(*origin_np.tolist()), wp.vec3(*normal_np.tolist())
    )

    assert np.allclose(ordered_wp.numpy(), ordered_tm, rtol=1e-5, atol=1e-5)


def test_radial_sort_perpendicular_to_a_tilted_normal(device: str) -> None:
    """
    Not a library comparison: trimesh shares this defect, so it cannot serve as the oracle here.

    trimesh's own axis0 formula, ``[normal[0], normal[2], -normal[1]]``, is perpendicular to
    ``normal`` only when ``normal[0] == 0`` -- for any other normal, ``dot(normal, axis0) ==
    normal.x**2 != 0``, so axis0 carries a leftover component along ``normal`` into every point's
    angle. ``test_radial_sort`` alone cannot catch this: it fixes ``normal = (0, 0, 1)``, the one
    case where the defect is exactly zero. This builds points around the maximally-degenerate case,
    ``normal = (1, 0, 0)`` -- where the old formula's axis0 becomes ``normal`` itself and axis1
    collapses to the zero vector, so every point's key reads ``atan2(x, 0)`` and the "sort" ties
    every key to the same value -- and checks the descending order against an independently-built
    orthonormal frame that has nothing to do with whichever axis pair ``radial_sort`` constructs
    internally.
    """
    rng = np.random.default_rng(9)
    n = 200
    theta_np = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius_np = rng.uniform(0.1, 2.0, n)
    # Points confined to the y-z plane, which is perpendicular to normal = (1, 0, 0).
    points_np = np.column_stack(
        (np.zeros(n), np.cos(theta_np) * radius_np, np.sin(theta_np) * radius_np)
    )
    points_np = points_np[rng.permutation(n)]
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([1.0, 0.0, 0.0])

    points_wp = points_to_warp(points_np, device)
    ordered_wp = od.radial_sort(
        points_wp, wp.vec3(*origin_np.tolist()), wp.vec3(*normal_np.tolist())
    ).numpy()

    # Any right-handed orthonormal pair spanning the plane perpendicular to `normal` gives a
    # monotonic (if not identical) reparametrization of the true angle around it, so the
    # *descending cyclic order* is basis-independent -- checked here against a (y, z) pair that
    # has no relationship to whichever axis0/axis1 `radial_sort` happens to build internally.
    angles_ref = np.arctan2(ordered_wp[:, 1], ordered_wp[:, 2])
    deltas = np.diff(np.concatenate([angles_ref, angles_ref[:1]]))
    deltas = (deltas + np.pi) % (2.0 * np.pi) - np.pi  # wrap into (-pi, pi]
    # A genuine radial order takes exactly one lap, so every step (however parametrized) has the
    # same sign; the old, collapsed-key order does not, since every key tied to the same value
    # leaves the points in their permuted input order instead.
    assert np.all(deltas > 0.0) or np.all(deltas < 0.0)


def test_radial_sort_with_start(device: str) -> None:
    """
    Class A: the same order rotated to begin at a supplied start direction.

    The input is permuted first, so a function ignoring ``start`` and returning the input order
    cannot pass. The reference rotation is computed in numpy from the same start vector.
    """
    rng = np.random.default_rng(8)
    n = 256
    theta_np = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius_np = rng.uniform(0.1, 2.0, n)
    points_np = np.column_stack(
        (np.cos(theta_np) * radius_np, np.sin(theta_np) * radius_np, np.zeros(n))
    )
    points_np = points_np[rng.permutation(n)]
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([0.0, 0.0, 1.0])
    start_np = np.array([1.0, 0.0, 0.0])

    ordered_tm = tm.radial_sort(points_np, origin=origin_np, normal=normal_np, start=start_np)

    points_wp = points_to_warp(points_np, device)
    ordered_wp = od.radial_sort(
        points_wp,
        wp.vec3(*origin_np.tolist()),
        wp.vec3(*normal_np.tolist()),
        start=wp.vec3(*start_np.tolist()),
    )

    assert np.allclose(ordered_wp.numpy(), ordered_tm, rtol=1e-5, atol=1e-5)


def test_radial_sort_parallel_start_raises(device: str) -> None:
    points_np = np.zeros((4, 3))
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([0.0, 0.0, 1.0])
    # start parallel to normal is invalid.
    start_np = np.array([0.0, 0.0, 2.0])

    with pytest.raises(ValueError, match=r"must not.*parallel"):
        tm.radial_sort(points_np, origin=origin_np, normal=normal_np, start=start_np)

    points_wp = points_to_warp(points_np, device)
    with pytest.raises(ValueError, match=r"must not.*parallel"):
        od.radial_sort(
            points_wp,
            wp.vec3(*origin_np.tolist()),
            wp.vec3(*normal_np.tolist()),
            start=wp.vec3(*start_np.tolist()),
        )


@pytest.mark.parity(
    "estimate_normals",
    "open3d",
    "pymeshlab",
    benchmarked=False,
    reason="the marked test below calls the *table-taking* form, "
    "estimate_normals(points, neighbours), which is exactly what this group times -- "
    "the neighbour table is its input. A row would put the search back inside the "
    "callable and re-time estimate_normals_knn under a second name, since every "
    "reference builds its own.",
)
@pytest.mark.parity("estimate_normals_knn", "open3d", "pymeshlab")
def test_estimate_normals_matches_open3d(device: str) -> None:
    """
    Class B against both references: the same search-plus-PCA, with the normal's sign left free.

    Both fix the smallest-eigenvalue covariance eigenvector and neither fixes its direction, so the
    named transform on all three sides is ``|dot| == 1``. **MeshLab** additionally needs
    ``smoothiter=0`` -- its default runs a normal-smoothing pass afterwards, which ordito does not
    do -- and a face-less MeshSet, since ``compute_normal_for_point_clouds`` is for datasets with no
    faces. Both parameters are the ones the benchmark passes.

    MeshLab agrees essentially exactly (worst measured ``|dot|`` 0.9999999), so it gets a hard
    bound; Open3D disagrees on a handful of points because the two break k-nearest *ties*
    differently, which is why its assert is a fraction rather than a per-point bound.
    """
    knn = 30
    points_np = _fibonacci_sphere(2000)

    # Open3D reference: PCA normal from the k-nearest neighbourhood (KNN includes self).
    pcd = points_to_open3d(points_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals_o3d = np.asarray(pcd.normals)

    meshset_pml = points_to_pymeshlab(points_np)
    meshset_pml.compute_normal_for_point_clouds(k=knn, smoothiter=0)
    normals_pml = np.asarray(meshset_pml.current_mesh().vertex_normal_matrix())

    # ordito: build the same k-neighbourhood (self + knn-1 = knn points total), then PCA.
    points_wp = points_to_warp(points_np, device)
    neighbor_idx_wp, _ = od_neighbors.query_nearest(points_wp, points_wp, k=knn, backend="bvh")
    normals_wp = od.estimate_normals(points_wp, neighbor_idx_wp)

    # Both estimators fix the smallest-eigenvalue covariance eigenvector but leave the sign
    # free, so compare up to sign. A few points may disagree on KNN ties; require the vast
    # majority to align.
    abs_dots = np.abs(np.einsum("ij,ij->i", normals_wp.numpy(), normals_o3d))
    assert np.mean(abs_dots > 0.99) > 0.98
    assert_same_up_to_sign(normals_wp.numpy(), normals_pml, atol=1e-5)


@pytest.mark.parity("estimate_normals_knn", "pytorch3d")
def test_estimate_normals_matches_pytorch3d(device: str) -> None:
    """
    Class B: the smallest-eigenvector normal, up to sign (``|dot| == 1``).

    ``disambiguate_directions=False`` is not a convenience: with it ``True`` pytorch3d applies the
    SHOT sign rule, which ordito has no counterpart for, so the two would disagree on a
    fixture-dependent subset of points for a reason that is not about the eigenvector. Left off,
    both sides leave the sign free and the comparison is the *subspace* -- measured min ``|dot|``
    0.9999979 and mean 1.0 over 600 points at a 16-neighbour window.

    Both sides get the identical neighbourhood: ordito's k-NN is what
    ``tests/test_neighbors.py::test_query_nearest_matches_pytorch3d`` pins against ``knn_points``,
    so a disagreement here is in the eigen-decomposition rather than in the gather.
    """
    rng = np.random.default_rng(13)
    points_np = rng.normal(size=(600, 3)).astype(np.float32)
    points_np /= np.linalg.norm(points_np, axis=1, keepdims=True)
    normals_p3d = p3d_ops.estimate_pointcloud_normals(
        points_to_torch(points_np, device), neighborhood_size=16, disambiguate_directions=False
    )[0]
    points_wp = points_to_warp(points_np, device)
    neighbor_idx_wp, _ = od_neighbors.query_nearest(points_wp, points_wp, k=16)
    normals_wp = od.estimate_normals(points_wp, neighbor_idx_wp)

    assert normals_p3d.shape == (600, 3)
    alignment_np = np.abs(
        np.einsum("ij,ij->i", normals_wp.numpy(), normals_p3d.cpu().numpy().astype(np.float32))
    )
    assert float(alignment_np.min()) > 0.999


@pytest.mark.parity(
    "estimate_normals",
    "meshlib",
    benchmarked=False,
    reason="the marked test below calls the *table-taking* form, "
    "estimate_normals(points, neighbours), which is exactly what this group times -- "
    "the neighbour table is its input. A row would put the search back inside the "
    "callable and re-time estimate_normals_knn under a second name, since every "
    "reference builds its own.",
)
@pytest.mark.parity("estimate_normals_knn", "meshlib")
def test_estimate_normals_matches_meshlib(device: str) -> None:
    """
    Class B (sign gauge): ``makeUnorientedNormals`` is the same PCA under a *radius* search.

    The named transform is the neighbourhood: MeshLib searches by **radius** where ordito is given
    a k-nearest table, so the two see the same points only where the cloud is uniform -- which is
    what the Fibonacci sphere is for. Measured there: minimum ``|dot|`` **0.99966** at a radius of
    1.5 mean spacings and **0.99945** at 3.0, i.e. every one of 2 000 normals agrees, and the result
    is insensitive to the radius over a factor of two.

    ``makeOrientedNormals`` is the oriented sibling and fixes the sign by propagating over a
    spanning structure; on a closed cloud it agrees with the outward radial direction on **100 %**
    of points, which is the assert below. That is the half ordito's ``orient_reference`` does with
    a single reference vector instead, and the two are compared here for the first time.
    """
    knn = 30
    points_np = _fibonacci_sphere(2000)
    spacing = float(np.sqrt(4.0 * np.pi / points_np.shape[0]))

    points_wp = points_to_warp(points_np, device)
    neighbor_idx_wp, _distances_wp = od_neighbors.query_nearest(
        points_wp, points_wp, k=knn, backend="bvh"
    )
    normals_wp = od.estimate_normals(points_wp, neighbor_idx_wp).numpy()

    cloud_ml = points_to_meshlib(points_np)
    for radius_scale in (1.5, 3.0):
        unoriented_ml = mm.makeUnorientedNormals(cloud_ml, radius_scale * spacing)
        normals_ml = mn.toNumpyArray(unoriented_ml)
        assert normals_ml.shape == points_np.shape  # non-vacuity: one normal per point
        assert np.abs(np.einsum("ij,ij->i", normals_wp, normals_ml)).min() > 0.999

    # The oriented form, against ordito's outward reference: both point away from the centre.
    oriented_ml = mm.makeOrientedNormals(cloud_ml, 2.0 * spacing)
    outward_np = points_np / np.linalg.norm(points_np, axis=1, keepdims=True)
    assert (np.einsum("ij,ij->i", mn.toNumpyArray(oriented_ml), outward_np) > 0.0).all()


def test_estimate_normals_orientation(device: str) -> None:
    # Small negative slack: the kernel enforces the sign in float32, so a float64
    # recomputation can dip just below zero at a zero-crossing.
    tol = 1e-5
    points_np = _fibonacci_sphere(1000)
    centroid_np = points_np.mean(axis=0)
    points_wp = points_to_warp(points_np, device)
    neighbor_idx_wp, _ = od_neighbors.query_nearest(points_wp, points_wp, k=20, backend="bvh")

    # Default: outward from the cloud centroid (the reference vector the kernel uses).
    normals_default = od.estimate_normals(points_wp, neighbor_idx_wp).numpy()
    assert np.all(np.einsum("ij,ij->i", normals_default, points_np - centroid_np) >= -tol)

    # Align with a fixed direction (Open3D orient_normals_to_align_with_direction).
    reference_np = np.array([0.0, 0.0, 1.0])
    normals_dir = od.estimate_normals(
        points_wp, neighbor_idx_wp, orient_reference=wp.vec3(*reference_np.tolist())
    ).numpy()
    assert np.all(normals_dir @ reference_np >= -tol)

    # Toward a camera at the sphere centre (Open3D orient_normals_towards_camera_location):
    # every normal points inward, i.e. opposite the outward position vector.
    normals_cam = od.estimate_normals(
        points_wp, neighbor_idx_wp, camera_location=wp.vec3(0.0, 0.0, 0.0)
    ).numpy()
    assert np.all(np.einsum("ij,ij->i", normals_cam, points_np) <= tol)


def test_estimate_normals_rejects_a_table_that_is_not_one_row_per_point(device: str) -> None:
    """
    Not a library comparison: no reference validates a caller-supplied neighbour table.

    The launch is one thread per point and each thread reads its own row of ``neighbor_idx``, so a
    table with fewer rows than the cloud is an out-of-bounds read, not a short answer -- and on the
    CPU device a Warp array is host heap, so it is heap corruption with no exception in release
    mode (CLAUDE.md section 12.1). Only the rank was checked.

    Both directions are asserted, because only the short one is unsafe and a guard written against
    inequality is the honest contract: a table with *more* rows than points is a caller error too,
    it just happens to be a survivable one, and accepting it silently would leave the argument's
    meaning ambiguous.
    """
    points_wp = points_to_warp(_fibonacci_sphere(10), device)
    for rows in (6, 14):
        neighbor_idx_wp = odt.as_array2d(
            wp.zeros((rows, 4), dtype=wp.int32, device=device), wp.int32
        )
        with pytest.raises(ValueError, match=r"one row per point"):
            od.estimate_normals(points_wp, neighbor_idx_wp)

    # The accepting case, so the guard is a bound and not a blanket refusal.
    matching_wp, _ = od_neighbors.query_nearest(points_wp, points_wp, k=4, backend="bvh")
    assert od.estimate_normals(points_wp, matching_wp).shape == (10,)


def test_estimate_normals_mutually_exclusive_orientation(device: str) -> None:
    points_wp = points_to_warp(_fibonacci_sphere(16), device)
    neighbor_idx_wp, _ = od_neighbors.query_nearest(points_wp, points_wp, k=8, backend="bvh")
    with pytest.raises(ValueError, match=r"at most one"):
        od.estimate_normals(
            points_wp,
            neighbor_idx_wp,
            orient_reference=wp.vec3(0.0, 0.0, 1.0),
            camera_location=wp.vec3(0.0, 0.0, 0.0),
        )


def _cloud_with_outliers(seed: int = 3, n_inliers: int = 400, n_outliers: int = 15) -> np.ndarray:
    """Build a tight Gaussian blob plus far stragglers, which land last in the array."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(size=(n_inliers, 3)), rng.normal(scale=6.0, size=(n_outliers, 3))])


def _loop_reference(points_np: np.ndarray, k: int, scale: float = 3.0) -> np.ndarray:
    """LoOP scores (Kriegel et al.) via scipy, the float64 oracle for the Warp implementation."""
    from scipy.spatial import cKDTree
    from scipy.special import erf

    distance_np, idx_np = cKDTree(points_np).query(points_np, k=k)
    sigma_np = np.sqrt(np.mean(distance_np**2, axis=1))
    plof_np = sigma_np / sigma_np[idx_np].mean(axis=1) - 1.0
    normalizer = scale * np.sqrt((plof_np**2).mean())
    return np.maximum(0.0, erf(plof_np / (normalizer * np.sqrt(2.0))))


def test_outlier_probability_matches_scipy(device: str) -> None:
    k = 32
    points_np = _cloud_with_outliers()
    probability_np = _loop_reference(points_np, k)

    points_wp = points_to_warp(points_np, device)
    neighbor_idx_wp, neighbor_distance_wp = od_neighbors.query_nearest(
        points_wp, points_wp, k=k, backend="bvh"
    )
    probability_wp = od.outlier_probability(neighbor_idx_wp, neighbor_distance_wp)
    assert np.allclose(probability_wp.numpy(), probability_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("outlier_probability", "pymeshlab")
def test_outlier_probability_ranks_the_planted_outliers(device: str) -> None:
    """Class C (a ranking plus a subset): the 15 planted outliers must score highest."""
    k = 32
    n_inliers, n_outliers = 400, 15
    points_np = _cloud_with_outliers(n_inliers=n_inliers, n_outliers=n_outliers)

    points_wp = points_to_warp(points_np, device)
    neighbor_idx_wp, neighbor_distance_wp = od_neighbors.query_nearest(
        points_wp, points_wp, k=k, backend="bvh"
    )
    probability_wp = od.outlier_probability(neighbor_idx_wp, neighbor_distance_wp)
    ranked = np.argsort(-probability_wp.numpy())
    assert set(ranked[:n_outliers].tolist()) == set(range(n_inliers, n_inliers + n_outliers))

    # pymeshlab's ``compute_selection_point_cloud_outliers`` is the same LoOP score under a
    # ``propthreshold``; it differs from this port in how its k-d tree counts the query point, so
    # compare the *sets* rather than the scores. Everything it flags must be a planted outlier, and
    # every point this port flags at MeshLab's own default threshold must be flagged there too.
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(np.ascontiguousarray(points_np, dtype=np.float64)))
    meshset_pml.compute_selection_point_cloud_outliers(propthreshold=0.8, knearest=k)
    selected_pml = np.flatnonzero(meshset_pml.current_mesh().vertex_selection_array())
    assert set(selected_pml.tolist()) <= set(range(n_inliers, n_inliers + n_outliers))
    assert set(np.flatnonzero(probability_wp.numpy() > 0.8).tolist()) <= set(selected_pml.tolist())


def test_outlier_probability_is_scale_invariant(device: str) -> None:
    """``sigma`` scales with the cloud but ``plof`` is a ratio, so the score must not move."""
    points_np = _cloud_with_outliers()
    scores = []
    for factor in (1.0, 100.0):
        points_wp = points_to_warp(points_np * factor, device)
        neighbor_idx_wp, neighbor_distance_wp = od_neighbors.query_nearest(
            points_wp, points_wp, k=32, backend="bvh"
        )
        scores.append(od.outlier_probability(neighbor_idx_wp, neighbor_distance_wp).numpy())
    assert np.allclose(scores[0], scores[1], rtol=1e-4, atol=1e-4)


def test_outlier_probability_invalid_scale(device: str) -> None:
    points_wp = points_to_warp(_fibonacci_sphere(16), device)
    neighbor_idx_wp, neighbor_distance_wp = od_neighbors.query_nearest(
        points_wp, points_wp, k=4, backend="bvh"
    )
    with pytest.raises(ValueError, match="scale must be positive"):
        od.outlier_probability(neighbor_idx_wp, neighbor_distance_wp, scale=0.0)


def test_outlier_probability_shape_mismatch(device: str) -> None:
    points_wp = points_to_warp(_fibonacci_sphere(16), device)
    neighbor_idx_wp, _ = od_neighbors.query_nearest(points_wp, points_wp, k=4, backend="bvh")
    _, neighbor_distance_wp = od_neighbors.query_nearest(points_wp, points_wp, k=5, backend="bvh")
    with pytest.raises(ValueError, match="same shape"):
        od.outlier_probability(neighbor_idx_wp, neighbor_distance_wp)


@pytest.mark.parity("statistical_outlier_mask", "open3d")
def test_statistical_outlier_mask_matches_open3d(device: str) -> None:
    """
    Class B (mask inversion): Open3D returns the indices it *keeps*, so they are inverted.

    Both sides use the same ``k`` and ``std_ratio`` and the same planted cloud, so the
    comparison is exact on the mask -- the transform is only that Open3D reports the
    complement.
    """
    k, std_ratio = 20, 2.0
    points_np = _cloud_with_outliers()

    pcd = points_to_open3d(points_np)
    _kept, keep_indices = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
    outlier_o3d = np.ones(points_np.shape[0], dtype=bool)
    outlier_o3d[np.asarray(keep_indices)] = False

    points_wp = points_to_warp(points_np, device)
    _idx, neighbor_distance_wp = od_neighbors.query_nearest(
        points_wp, points_wp, k=k, backend="bvh"
    )
    outlier_wp = od.statistical_outlier_mask(neighbor_distance_wp, std_ratio=std_ratio)
    assert np.array_equal(outlier_wp.numpy().astype(bool), outlier_o3d)


@pytest.mark.parity(
    "statistical_outlier_mask",
    "meshlib",
    benchmarked=False,
    reason="findOutliers is a different criterion, not a different tuning: its four modes are "
    "connectivity- and normal-based (SmallComponents, WeaklyConnected, FarSurface, AwayNormal) "
    "where statistical_outlier_mask is a z-score on the k-NN mean distance, and it takes a radius "
    "where ordito takes a neighbour count. Timing them against each other would price two "
    "different questions -- measured on a planted cloud, SmallComponents flags 26 points to "
    "ordito's 13 while both contain all 15 planted outliers. open3d's remove_statistical_outlier "
    "is the same z-score and carries the timed row; the comparison here is what MeshLib can still "
    "say about the answer.",
)
def test_statistical_outlier_mask_matches_meshlib(device: str) -> None:
    """
    Class C (recall and containment): MeshLib's criterion is stricter, and strictly wider.

    ``findOutliers`` with ``OutlierTypeMask.SmallComponents`` labels a point an outlier when its
    connected component under a radius graph is small, which is a different question from a z-score
    on the k-NN mean distance -- so there is no correspondence to compare and the statistic is what
    each finds. On the planted cloud both find **all 15** seeded outliers, and ordito's 13 flagged
    points are a **subset** of MeshLib's 26: it is the more conservative of the two, with zero false
    positives against MeshLib's 11.

    **Bug class excluded:** a detector that flags the wrong points, or flags on the wrong scale --
    containment is what a looser threshold cannot fake, since a mask that grew arbitrarily would
    break the subset relation in the other direction. **Mutation probe, measured:** shuffling
    ordito's mask drops the element-wise agreement from 0.969 to 0.906 -- too thin a margin to
    assert on, which is exactly why the assert is recall plus containment rather than agreement.

    **The default mask is unusable and crashes**: ``FindOutliersParams.mask`` defaults to ``All``,
    which includes ``AwayNormal``, and that criterion **segfaults** on a cloud carrying no normals
    -- no exception, no traceback. The mode is always set explicitly here.
    """
    k, std_ratio = 20, 2.0
    points_np = _cloud_with_outliers()
    n_outliers = 15
    planted_np = np.zeros(points_np.shape[0], dtype=bool)
    planted_np[-n_outliers:] = True

    points_wp = points_to_warp(points_np, device)
    _idx_wp, neighbor_distance_wp = od_neighbors.query_nearest(
        points_wp, points_wp, k=k, backend="bvh"
    )
    outlier_wp = od.statistical_outlier_mask(neighbor_distance_wp, std_ratio=std_ratio).numpy()

    cloud_ml = points_to_meshlib(points_np)
    params_ml = mm.FindOutliersParams()
    params_ml.radius = 1.0
    params_ml.mask = mm.OutlierTypeMask.SmallComponents  # ``All`` segfaults without normals
    outlier_ml = meshlib_bitset_to_numpy(mm.findOutliers(cloud_ml, params_ml), points_np.shape[0])

    assert (
        outlier_ml.sum() < 0.1 * points_np.shape[0]
    )  # non-vacuity: not "everything is an outlier"
    assert (outlier_ml & planted_np).sum() == n_outliers  # the reference found every planted one
    assert (outlier_wp & planted_np).sum() >= n_outliers - 2
    assert (outlier_wp & ~outlier_ml).sum() == 0  # ordito's set is contained in MeshLib's


def test_statistical_outlier_mask_empty(device: str) -> None:
    neighbor_distance_wp = odt.empty_2d((0, 8), wp.float32, device=device)
    assert od.statistical_outlier_mask(neighbor_distance_wp).shape == (0,)


def test_statistical_outlier_mask_flags_coincident_and_empty_rows_below_two_counted(
    device: str,
) -> None:
    """
    Not a library comparison: the docstring's own guarantee, at the input it used to skip.

    The docstring promises "a point with an empty or fully coincident neighbourhood is marked as
    an outlier" unconditionally, but the ``counted < 2`` early return (guarding the cloud
    deviation's ``ddof=1`` division) used to return an all-``False`` mask whenever at most one row
    in the whole cloud had any finite neighbour distance -- silently dropping that guarantee
    instead of applying the two-thirds of ``is_statistical_outlier``'s predicate
    (``count == 0 or mean_distance <= 0.0``) that needs no cloud statistic at all. Three rows here:
    one fully coincident (every neighbour at distance 0), two fully empty (every slot ``inf``) --
    every one of the three must read ``True``.
    """
    neighbor_distance_wp = odt.as_array2d(
        wp.array(np.array([[0.0], [np.inf], [np.inf]], dtype=np.float32), device=device), wp.float32
    )
    outlier_wp = od.statistical_outlier_mask(neighbor_distance_wp).numpy()
    assert np.array_equal(outlier_wp, np.array([True, True, True]))


@pytest.mark.parity("radius_outlier_mask", "open3d")
def test_radius_outlier_mask_matches_open3d(device: str) -> None:
    """
    Class B (reference entry point): Open3D's own rule, evaluated through its tree *serially*.

    ``remove_radius_outlier`` is the obvious oracle and it is **nondeterministic**: it shares one
    ``KDTreeFlann`` across an OpenMP loop whose radius search is not thread-safe under that sharing,
    and eight repetitions of a 500-point cloud returned three distinct keep sets (43 / 44 / 45
    points), differing by one or two points each. So the comparison goes through the same tree one
    query at a time and applies the filter's own published rule -- ``count > nb_points``, self
    counted -- which reproduces this function's mask **exactly** here. That is the named transform.

    Parametrized over three thresholds so the mask is neither all-``True`` nor all-``False``, which
    a constant answer would otherwise pass, and the counts themselves are cross-checked against the
    same tree so a wrong count cannot cancel against a wrong comparison.
    """
    radius = 0.15
    rng = np.random.default_rng(0)
    points_np = rng.random((500, 3)).astype(np.float32).astype(np.float64)

    cloud_o3d = points_to_open3d(points_np)
    tree_o3d = o3d.geometry.KDTreeFlann(cloud_o3d)
    count_o3d = np.array(
        [tree_o3d.search_radius_vector_3d(cloud_o3d.points[i], radius)[0] for i in range(500)]
    )

    points_wp = points_to_warp(points_np, device)
    count_wp = od_neighbors.query_ball_count(points_wp, points_wp, radius)
    assert np.array_equal(count_wp.numpy(), count_o3d)

    for min_neighbors in (3, 6, 12):
        outlier_o3d = count_o3d <= min_neighbors
        # non-vacuity: this threshold splits the cloud rather than condemning or sparing all of it
        assert 0 < outlier_o3d.sum() < 500
        outlier_wp = od.radius_outlier_mask(points_wp, radius, min_neighbors)
        assert np.array_equal(outlier_wp.numpy().astype(bool), outlier_o3d)


def test_radius_outlier_mask_invalid_arguments(device: str) -> None:
    points_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="radius"):
        od.radius_outlier_mask(points_wp, 0.0, 2)
    with pytest.raises(ValueError, match="min_neighbors"):
        od.radius_outlier_mask(points_wp, 1.0, 0)
    empty_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert od.radius_outlier_mask(empty_wp, 1.0, 2).shape == (0,)


@pytest.mark.parity(
    "point_finite_mask",
    "open3d",
    benchmarked=False,
    reason="one wp.map over a three-component isfinite predicate, so a group would measure the "
    "launch floor and nothing else -- ~11 us of host time whatever the cloud, which is the same "
    "number every other trivial map in the package would report. open3d's remove_non_finite_points "
    "additionally copies the surviving cloud out, so its side would be timing the copy. The "
    "comparison is what open3d can still say about the answer, and it is exact.",
)
def test_point_finite_mask_matches_open3d(device: str) -> None:
    """
    Class B (mask against a kept subset): ``remove_non_finite_points`` returns the surviving cloud.

    The transform is the only one available -- Open3D hands back points, not a mask -- so the kept
    positions are compared against the rows this mask selects, in order. One ``NaN`` and both
    infinities are planted, in each of the three coordinate slots, since a predicate testing only
    ``point[0]`` would pass a single-column probe.
    """
    rng = np.random.default_rng(4)
    points_np = rng.random((40, 3))
    points_np[3, 1] = np.nan
    points_np[7, 0] = np.inf
    points_np[11, 2] = -np.inf
    points_np[19, 2] = np.nan

    kept_o3d = np.asarray(points_to_open3d(points_np).remove_non_finite_points().points)

    points_wp = points_to_warp(points_np, device)
    finite_wp = od.point_finite_mask(points_wp).numpy().astype(bool)

    assert kept_o3d.shape[0] == 36  # non-vacuity: the reference dropped exactly the four planted
    assert np.array_equal(np.flatnonzero(~finite_wp), np.array([3, 7, 11, 19]))
    assert np.allclose(points_np[finite_wp], kept_o3d, rtol=1e-5, atol=1e-5)

    empty_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert od.point_finite_mask(empty_wp).shape == (0,)


@pytest.mark.parity("point_duplicate_mask", "open3d")
def test_point_duplicate_mask_matches_open3d(device: str) -> None:
    """
    Class B (mask against a kept subset): ``remove_duplicated_points``'s survivors, in order.

    Open3D returns the deduplicated cloud, so the comparison is its rows against the rows the
    complement of this mask selects — which also pins the *first-occurrence* rule, since keeping the
    last occurrence instead would reorder the survivors. ``-0.0`` against ``+0.0`` is planted
    deliberately: IEEE-754 equality holds between them, so the reference merges them and a raw
    bit-pattern key would not.
    """
    rng = np.random.default_rng(0)
    base_np = rng.random((20, 3)).astype(np.float32)
    points_np = np.concatenate(
        [
            base_np,
            base_np[:5],
            base_np[10:15],
            np.array([[-0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    )

    kept_o3d = np.asarray(
        points_to_open3d(points_np).remove_duplicated_points().points, dtype=np.float32
    )

    points_wp = points_to_warp(points_np, device)
    duplicate_wp = od.point_duplicate_mask(points_wp).numpy().astype(bool)

    # non-vacuity: 10 repeats plus the second zero row, so both the mask and its complement matter
    assert duplicate_wp.sum() == 11
    assert kept_o3d.shape[0] == 21
    assert np.array_equal(points_np[~duplicate_wp], kept_o3d)

    empty_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert od.point_duplicate_mask(empty_wp).shape == (0,)


@pytest.mark.parity("point_duplicate_mask", "meshlib")
def test_point_duplicate_mask_matches_meshlib(device: str) -> None:
    """
    Class B (a representative map against a mask): ``map != index`` is exactly this mask.

    ``findSmallestCloseVertices(cloud, 0.0)`` sends every point to the **smallest-indexed** point
    within ``closeDist``, itself when it is the first of its class -- so the two conventions line
    up without a choice being made: the entries it moves are precisely the repeats this flags, and
    the first-occurrence rule is the same one. Element for element on the fixture the open3d pair
    above uses, ``-0.0`` row included, which the reference merges with ``+0.0`` because their
    distance is zero.

    ``closeDist=0.0`` is an exact-equality request rather than a degenerate tolerance -- MeshLib's
    test is inclusive at the radius, so a zero radius matches coincident points and nothing else,
    which is what makes it the *same* question ``remove_duplicated_points`` answers rather than
    ``grouping.unique_rows``' bucketed one. ``findCloseVertices`` is the same search returning a
    bitset, but that one flags **both** members of a coincident pair (40 against 20 on a planted
    cloud) and so is not this mask.
    """
    rng = np.random.default_rng(0)
    base_np = rng.random((20, 3)).astype(np.float32)
    points_np = np.concatenate(
        [
            base_np,
            base_np[:5],
            base_np[10:15],
            np.array([[-0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    )

    representative_ml = meshlib_indices_to_numpy(
        # MeshLib's stub omits ``VertId.__index__``, which it implements at runtime.
        mm.findSmallestCloseVertices(points_to_meshlib(points_np), 0.0)  # pyright: ignore[reportArgumentType]
    )
    duplicate_ml = representative_ml != np.arange(points_np.shape[0])

    points_wp = points_to_warp(points_np, device)
    duplicate_wp = od.point_duplicate_mask(points_wp).numpy().astype(bool)

    assert duplicate_ml.sum() == 11  # non-vacuity: the mask and its complement both matter
    assert np.array_equal(duplicate_wp, duplicate_ml)


def test_point_duplicate_mask_separates_one_ulp(device: str) -> None:
    """
    Not a library comparison: the guard on the *load-bearing* trick, which no reference can see.

    Open3D works in ``float64``, so a ``float32``-adjacent pair is two distinct positions to it
    whatever this function does — the comparison above cannot tell an exact key from a bucketed one.
    Two coordinates one ULP apart must land in different classes, which is what distinguishes the
    bit-reinterpretation key from ``grouping.unique_rows``' relative bucket: that one merges
    anything within ~2.4e-4 relative and would call all three rows here duplicates.
    """
    one = np.float32(1.0)
    next_one = np.nextafter(one, np.float32(2.0))
    assert one != next_one
    points_np = np.array([[one, 0.0, 0.0], [next_one, 0.0, 0.0], [one, 0.0, 0.0]], dtype=np.float32)

    points_wp = points_to_warp(points_np, device)
    duplicate_wp = od.point_duplicate_mask(points_wp).numpy().astype(bool)

    # row 1 is one ULP away and is its own position; row 2 repeats row 0 exactly
    assert np.array_equal(duplicate_wp, np.array([False, False, True]))

    _unique_bucketed = od_grouping.unique_rows(points_wp)
    assert _unique_bucketed.size == 1  # the bucketed key merges all three


def _position_hash_slot_np(points_np: np.ndarray, mask: int) -> np.ndarray:
    """``kernels.points.position_hash_slot`` in NumPy, to construct colliding positions."""
    bits = points_np.view(np.uint32).astype(np.uint64)
    bits[points_np == 0.0] = 0  # the two zeros fold together
    mult = np.uint64(11400714819323198485)
    with np.errstate(over="ignore"):
        key = ((bits[:, 0] << np.uint64(32)) | bits[:, 1]) ^ (bits[:, 2] * mult)
        h = key * mult
    h ^= h >> np.uint64(32)
    return (h & np.uint64(mask)).astype(np.int64)


def test_point_duplicate_mask_distinct_points_sharing_a_slot(device: str) -> None:
    """
    Not a library comparison: a hash collision between *distinct* positions is not a duplicate.

    The table stores point indices and compares a candidate's bits against the prober's own, so
    two positions landing on one home slot must stay two classes. Four points make an eight-slot
    table; the first two are distinct positions chosen to share a home slot (asserted through a
    NumPy transcription of the hash, so the fixture cannot silently stop colliding), the last two
    repeat them. A mask that trusted the slot would flag rows 1 and 3 alike.
    """
    rng = np.random.default_rng(7)
    candidates_np = rng.random((64, 3)).astype(np.float32)
    slots_np = _position_hash_slot_np(candidates_np, 7)
    first = 0
    second = int(np.flatnonzero(slots_np[1:] == slots_np[first])[0]) + 1
    points_np = candidates_np[[first, second, first, second]]
    assert _position_hash_slot_np(points_np[:2], 7).tolist() == [slots_np[first]] * 2
    assert not np.array_equal(points_np[0], points_np[1])

    duplicate_wp = od.point_duplicate_mask(points_to_warp(points_np, device)).numpy()

    assert np.array_equal(duplicate_wp, np.array([False, False, True, True]))


@pytest.mark.parity("farthest_point_sample", "open3d")
def test_farthest_point_sample_matches_open3d(device: str) -> None:
    """
    Class B (order discarded): the same *set* as ``farthest_point_down_sample``, at four counts.

    Open3D returns the selected points through ``SelectByIndex``, which emits them in ascending
    index order, so its sequence is not recoverable and only the set can be compared -- the
    transform. The sequence is pinned separately below, against a transcription of its own loop.
    """
    rng = np.random.default_rng(0)
    points_np = rng.random((500, 3)).astype(np.float32).astype(np.float64)
    cloud_o3d = points_to_open3d(points_np)

    points_wp = points_to_warp(points_np, device)
    for count in (1, 4, 32, 64):
        selected_o3d = np.asarray(cloud_o3d.farthest_point_down_sample(count).points)
        index_o3d = {
            int(np.argmin(np.linalg.norm(points_np - point, axis=1))) for point in selected_o3d
        }
        assert len(index_o3d) == count  # non-vacuity: the reference really returned `count` points

        index_wp = od.farthest_point_sample(points_wp, count).numpy()
        assert set(index_wp.tolist()) == index_o3d


@pytest.mark.parity("farthest_point_sample", "pytorch3d")
def test_farthest_point_sample_matches_pytorch3d(device: str) -> None:
    """
    Class A: the greedy **index sequence**, byte-identical, not merely the selected set.

    A strictly stronger oracle than the open3d one above, and worth having for exactly that reason:
    open3d routes every selection through ``SelectByIndex``, which emits survivors in ascending
    index order and destroys the greedy order, so that comparison can only assert set equality.
    pytorch3d returns the indices in the order it picked them. Both libraries and ordito resolve
    an arg-max tie to the lowest index, which is what makes an exact sequence comparison legitimate
    on a cloud that happens to tie.

    ``random_start_point=False`` pins pytorch3d's start to index 0, which is ordito's default.
    """
    rng = np.random.default_rng(7)
    points_np = rng.normal(size=(500, 3)).astype(np.float32)
    _, indices_p3d = p3d_ops.sample_farthest_points(
        points_to_torch(points_np, device), K=8, random_start_point=False
    )
    indices_wp = od.farthest_point_sample(points_to_warp(points_np, device), 8)

    assert indices_p3d.shape == (1, 8)
    assert np.unique(indices_p3d.cpu().numpy()).size == 8
    assert np.array_equal(indices_wp.numpy(), indices_p3d[0].cpu().numpy())


def _farthest_point_sequence(points_np: np.ndarray, count: int, start: int = 0) -> np.ndarray:
    """
    Open3D's ``FarthestPointDownSample`` loop transcribed, which is the only oracle for the order.

    Strict ``>`` on the running maximum, so the lowest index wins a tie — the convention the packed
    ``atomic_max`` key reproduces.
    """
    selected = []
    distances = np.full(points_np.shape[0], np.inf)
    farthest = start
    for _ in range(count):
        selected.append(farthest)
        squared = ((points_np - points_np[farthest]) ** 2).sum(axis=1)
        distances = np.minimum(distances, squared)
        farthest = int(np.argmax(distances))  # numpy argmax already breaks ties towards low indices
    return np.array(selected, dtype=np.int32)


@pytest.mark.parametrize("start", [0, 137])
def test_farthest_point_sample_sequence_and_coverage(device: str, start: int) -> None:
    """
    Class A against a transcription of the reference loop, plus the property no reference asserts.

    Two claims the set comparison above cannot make. First the **order**: entry 0 is ``start`` and
    each later entry is the arg-max, tie broken low — compared index for index against the NumPy
    port. Second **coverage monotonicity**: the distance from the cloud to the selected set can only
    shrink as the count grows, which is the defining property of the greedy choice and would break
    under an arg-*min* or a stale distance buffer.
    """
    rng = np.random.default_rng(2)
    points_np = rng.random((400, 3)).astype(np.float32)
    points_wp = points_to_warp(points_np, device)

    index_wp = od.farthest_point_sample(points_wp, 40, start=start).numpy()
    assert np.array_equal(
        index_wp, _farthest_point_sequence(points_np.astype(np.float64), 40, start)
    )

    radii = []
    for count in (4, 10, 40):
        chosen_np = points_np[od.farthest_point_sample(points_wp, count, start=start).numpy()]
        radii.append(
            float(
                np.linalg.norm(points_np[:, None, :] - chosen_np[None, :, :], axis=2)
                .min(axis=1)
                .max()
            )
        )
    assert radii[0] >= radii[1] >= radii[2]


def test_farthest_point_sample_invalid_arguments(device: str) -> None:
    points_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert od.farthest_point_sample(points_wp, 0).shape == (0,)
    with pytest.raises(ValueError, match="count"):
        od.farthest_point_sample(points_wp, 5)
    with pytest.raises(ValueError, match="count"):
        od.farthest_point_sample(points_wp, -1)
    with pytest.raises(ValueError, match="start"):
        od.farthest_point_sample(points_wp, 2, start=4)


@pytest.mark.parity("vector_angle", "trimesh")
def test_vector_angle(device: str) -> None:
    """
    Class B (input packing): ``trimesh.geometry.vector_angle`` wants the pairs as ``(n, 2, 3)``.

    The named transform is the reshape on the reference side; the values are compared directly.
    Unit vectors on both sides, so this is the angle formula alone rather than a normalization
    test.
    """
    rng = np.random.default_rng(42)
    n = 64
    vecs_a_np = rng.standard_normal((n, 3))
    vecs_a_np /= np.linalg.norm(vecs_a_np, axis=1, keepdims=True)
    vecs_b_np = rng.standard_normal((n, 3))
    vecs_b_np /= np.linalg.norm(vecs_b_np, axis=1, keepdims=True)

    pairs_np = np.stack([vecs_a_np, vecs_b_np], axis=1)
    angles_tm = tm_geometry.vector_angle(pairs_np)

    vecs_a_wp = points_to_warp(vecs_a_np, device)
    vecs_b_wp = points_to_warp(vecs_b_np, device)
    angles_wp = od.vector_angle(vecs_a_wp, vecs_b_wp)
    assert np.allclose(angles_wp.numpy(), angles_tm, rtol=1e-5, atol=1e-5)


def test_vector_angle_empty(device: str) -> None:
    vecs_a_wp = warp_empty(0, wp.vec3, device)
    vecs_b_wp = warp_empty(0, wp.vec3, device)
    angles_wp = od.vector_angle(vecs_a_wp, vecs_b_wp)
    assert angles_wp.shape == (0,)


@pytest.mark.parity("convex_subset_mask", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parity("convex_subset", "trimesh", "open3d", "pymeshlab")
def test_convex_subset_mask_against_the_three_qhull_backends(device: str) -> None:
    """
    Class C (soundness plus a recall bound), against exact qhull.

    ``benchmarks/test_convex.py`` says of these three rows that "this is not a parity comparison":
    trimesh, Open3D and pymeshlab all run **qhull** and return the exact hull as a *mesh*, while
    ``convex_subset`` returns an approximate *vertex subset* from a direction sweep. That rules
    out equality -- it does not rule out a test. Two properties are checkable and are exactly
    what an approximate hull filter has to guarantee:

    - **soundness**, asserted exactly: every point ordito selects must be a true hull vertex. This
      is the half that catches a real bug -- an implementation that returned interior points, or the
      whole cloud, fails immediately, and no tolerance is involved. Note this is the *fixture's*
      guarantee, not the function's: it holds because 500 standard-normal points are in general
      position, so no support direction ties. The invariant that holds for every input is the weaker
      "every selected point lies on the hull boundary" -- on a cloud with coplanar ties (a grid over
      each face of a cube) the mask also selects face-edge midpoints, which are boundary points but
      not hull vertices, and this assertion would fail there by design.
    - **recall**, asserted with a bound: measured **27 of 31** hull vertices at
      ``n_directions=256`` (0.871) against all three references, which agree with each other on the
      hull exactly. The 0.70 floor leaves room for a different direction set without admitting a
      filter that has stopped finding most of the hull. The gap is not slack in the test -- the four
      missed vertices have normal cones spanning 3e-5 to 2e-3 of the sphere, so no direction sample
      this size is expected to find them; recall reaches 1.00 on this cloud at 16384 directions.

    Marked for all three libraries deliberately: they compute the identical answer here, so one
    assertion covers all three rows, and confirming they agree is itself worth a line -- it says the
    benchmark's three qhull rows are pricing wrappers around one algorithm, not three algorithms.

    Both hull entry points are checked against the references here, which is why the marker names
    ``convex_subset`` as well as ``convex_subset_mask``: the two benchmark groups time the same
    approximation against the same qhull bar, so one comparison is the honest place for both claims.
    The *mask against subset* equality below is ordito-against-ordito -- the mask is the entry
    point carrying the oracle, and ``test_convex_subset_points`` pins the same pair on positions.
    """
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    points_wp = points_to_warp(points_np, device)

    selected = set(
        np.flatnonzero(od.convex_subset_mask(points_wp, n_directions=256).numpy()).tolist()
    )

    def hull_indices(hull_vertices: np.ndarray, tolerance: float = 1e-9) -> set[int]:
        """Map a hull's vertex positions back onto indices into the input cloud."""
        distance_np, index_np = scipy.spatial.cKDTree(points_np).query(np.asarray(hull_vertices))
        assert np.max(distance_np) < tolerance  # every returned position is one of the inputs
        return set(np.asarray(index_np)[distance_np < tolerance].tolist())

    # The compacted entry point resolves back to the identical index set, so the assertions below
    # speak for both benchmark groups rather than only the mask one. Its positions come back
    # ``float32`` where the three references hand back the ``float64`` inputs verbatim, so the
    # lookup needs a float32-scale tolerance: measured 1.2e-07 of round-trip error against a
    # minimum inter-point spacing of 0.046 in this cloud, so 1e-5 is unambiguous by ~4 600x.
    assert hull_indices(od.convex_subset(points_wp, n_directions=256).numpy(), 1e-5) == selected

    hull_tm = hull_indices(tm.PointCloud(points_np).convex_hull.vertices)
    mesh_o3d, _kept = points_to_open3d(points_np).compute_convex_hull()
    hull_o3d = hull_indices(np.asarray(mesh_o3d.vertices))
    meshset_pml = points_to_pymeshlab(points_np)
    meshset_pml.generate_convex_hull()
    hull_pml = hull_indices(meshset_pml.current_mesh().vertex_matrix())

    # The three qhull wrappers agree, so any one of them is "the" exact hull.
    assert hull_tm == hull_o3d == hull_pml
    assert len(hull_tm) > 0

    for name, hull in (("trimesh", hull_tm), ("open3d", hull_o3d), ("pymeshlab", hull_pml)):
        assert selected <= hull, f"{name}: selected a point that is not a hull vertex"
        assert len(selected & hull) / len(hull) > 0.70, f"{name}: recall too low"


def test_convex_subset_mask_sound(device: str) -> None:
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    points_wp = points_to_warp(points_np, device)

    mask_wp = od.convex_subset_mask(points_wp, n_directions=256)
    selected = np.flatnonzero(mask_wp.numpy())

    hull_scipy = scipy.spatial.ConvexHull(points_np)
    assert set(selected.tolist()) <= set(hull_scipy.vertices.tolist())


def test_convex_subset_mask_scale_invariant(device: str) -> None:
    rng = np.random.default_rng(7)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    scaled_np = points_np * 1e4

    points_wp = points_to_warp(points_np, device)
    scaled_wp = points_to_warp(scaled_np, device)

    mask_wp = od.convex_subset_mask(points_wp, n_directions=256)
    mask_scaled_wp = od.convex_subset_mask(scaled_wp, n_directions=256)
    assert np.array_equal(mask_wp.numpy(), mask_scaled_wp.numpy())

    hull_scipy = scipy.spatial.ConvexHull(scaled_np)
    selected = np.flatnonzero(mask_scaled_wp.numpy())
    assert set(selected.tolist()) <= set(hull_scipy.vertices.tolist())


def test_convex_subset_recall(device: str) -> None:
    rng = np.random.default_rng(2)
    points_np = rng.standard_normal((200, 3)).astype(np.float64)
    points_wp = points_to_warp(points_np, device)

    mask_wp = od.convex_subset_mask(points_wp, n_directions=4096)
    selected = set(np.flatnonzero(mask_wp.numpy()).tolist())

    hull_scipy = scipy.spatial.ConvexHull(points_np)
    assert selected == set(hull_scipy.vertices.tolist())


def test_convex_subset_points(device: str) -> None:
    rng = np.random.default_rng(4)
    points_np = rng.standard_normal((300, 3)).astype(np.float64)
    points_wp = points_to_warp(points_np, device)

    mask_wp = od.convex_subset_mask(points_wp, n_directions=256)
    subset_wp = od.convex_subset(points_wp, n_directions=256)

    selected = np.flatnonzero(mask_wp.numpy())
    expected_points = points_np[np.sort(selected)]
    assert np.allclose(subset_wp.numpy(), expected_points, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["gaussian", "ball", "cube"])
def test_convex_masks_slice_filter_matches_exhaustive(
    device: str, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """
    Ordito against ordito: the slice-filtered sweeps mark exactly what the exhaustive ones do.

    From ``SUPPORT_SLICE_FILTER_FROM`` points the support sweeps walk only the slices whose
    extremes reach a threshold; no fixture here is that large, so the threshold is moved both ways
    on one cloud. The exhaustive path carries the oracles (the qhull backends and scipy, around
    this test). The cloud is offset far from the origin so the dot products carry real rounding --
    the case the slice bounds' margin exists for.
    """
    points_np = _cloud(kind, 20_000, seed=21) + np.array([300.0, -200.0, 100.0])
    points_wp = points_to_warp(points_np, device)

    def masks() -> list[np.ndarray]:
        return [
            od.convex_subset_mask(points_wp, n_directions=256).numpy(),
            od.convex_superset_mask(points_wp, subdivisions=2).numpy(),
        ]

    monkeypatch.setattr(od, "SUPPORT_SLICE_FILTER_FROM", 1 << 30)
    exhaustive = masks()
    monkeypatch.setattr(od, "SUPPORT_SLICE_FILTER_FROM", 0)
    filtered = masks()
    assert all(0 < int(mask.sum()) < mask.size for mask in exhaustive)
    for exhaustive_mask, filtered_mask in zip(exhaustive, filtered, strict=True):
        assert np.array_equal(filtered_mask, exhaustive_mask)


def test_convex_subset_mask_empty(device: str) -> None:
    points_wp = warp_empty(0, wp.vec3, device)
    mask_wp = od.convex_subset_mask(points_wp)
    assert mask_wp.shape == (0,)


def _cloud(kind: str, n: int, seed: int) -> np.ndarray:
    """Build one of the point distributions the superset filter behaves differently on."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((n, 3))
    if kind == "cube":
        return rng.random((n, 3))
    direction_np = rng.standard_normal((n, 3))
    direction_np /= np.linalg.norm(direction_np, axis=1, keepdims=True)
    return direction_np * rng.random((n, 1)) ** (1.0 / 3.0)


# Fraction of the cloud the filter is allowed to keep, per distribution, at ``subdivisions=3`` on
# 20k points. Measured 0.40% / 3.60% / 1.46% for gaussian / ball / cube; these bounds sit ~3x above
# that, which is what makes the containment assertion below non-vacuous -- an all-``True`` mask
# (the trivially correct superset) keeps 100% and fails every one of them.
_MAX_KEPT_FRACTION = {"gaussian": 0.012, "ball": 0.11, "cube": 0.05}


@pytest.mark.parametrize("kind", ["gaussian", "ball", "cube"])
@pytest.mark.parity("convex_superset_mask", "scipy")
def test_convex_superset_mask_contains_the_exact_hull(device: str, kind: str) -> None:
    """
    Class B: exact containment of the reference hull's vertex set.

    The one named transform is reading [`scipy.spatial.ConvexHull`][]'s hull vertex *indices* as a
    boolean mask.

    Containment rather than equality is the point, not a weakening: ``convex_superset_mask`` is
    defined as a conservative filter, and "no hull vertex is ever discarded" is the whole contract.
    The assertion is exact -- no tolerance -- and it is the assertion that fails if the tetrahedron
    interior test is ever wrong in the unsafe direction.

    Containment alone would be vacuous (an all-``True`` mask satisfies it), so the second assertion
    bounds how much the filter keeps. Both must hold: the first catches a filter that discards too
    much, the second a filter that discards too little. Parametrized over three distributions
    because selectivity varies by two orders of magnitude between them -- near-spherical clouds are
    the easy case and flat-faced ones the hard case -- so a single fixture would hide a regression
    on the others.
    """
    points_np = _cloud(kind, 20_000, seed=11)
    points_wp = points_to_warp(points_np, device)

    mask_np = od.convex_superset_mask(points_wp, subdivisions=3).numpy()
    kept = set(np.flatnonzero(mask_np).tolist())
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    assert hull_scipy <= kept, f"{kind}: discarded {len(hull_scipy - kept)} true hull vertices"
    assert mask_np.mean() < _MAX_KEPT_FRACTION[kind], f"{kind}: filter kept {mask_np.mean():.3%}"


@pytest.mark.parametrize("kind", ["gaussian", "cube"])
def test_convex_superset_mask_tightens_with_subdivisions(device: str, kind: str) -> None:
    """More directions wrap the hull more closely, and the guarantee holds at every level."""
    points_np = _cloud(kind, 5_000, seed=12)
    points_wp = points_to_warp(points_np, device)
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    counts = []
    for subdivisions in (0, 1, 2, 3):
        mask_np = od.convex_superset_mask(points_wp, subdivisions=subdivisions).numpy()
        assert hull_scipy <= set(np.flatnonzero(mask_np).tolist())
        counts.append(int(mask_np.sum()))

    assert counts == sorted(counts, reverse=True), f"not monotone in subdivisions: {counts}"
    assert counts[-1] < counts[0]
    assert counts[-1] >= len(hull_scipy)


def test_convex_superset_mask_contains_the_subset_mask(device: str) -> None:
    """The two one-sided filters bracket the hull: subset ``<=`` hull vertices ``<=`` superset."""
    points_np = _cloud("gaussian", 5_000, seed=13)
    points_wp = points_to_warp(points_np, device)

    subset_np = od.convex_subset_mask(points_wp, n_directions=256).numpy()
    superset_np = od.convex_superset_mask(points_wp, subdivisions=3).numpy()
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    assert set(np.flatnonzero(subset_np).tolist()) <= hull_scipy
    assert hull_scipy <= set(np.flatnonzero(superset_np).tolist())
    assert np.array_equal(subset_np & superset_np, subset_np)


def test_convex_superset_mask_scale_invariant(device: str) -> None:
    """The flatness and margin tests are relative, so scaling the cloud cannot change the mask."""
    points_np = _cloud("gaussian", 5_000, seed=14)
    points_wp = points_to_warp(points_np, device)
    scaled_wp = points_to_warp(points_np * 10000.0, device)

    assert np.array_equal(
        od.convex_superset_mask(points_wp).numpy(), od.convex_superset_mask(scaled_wp).numpy()
    )


@pytest.mark.parametrize("kind", ["coplanar", "collinear", "identical", "three_points"])
def test_convex_superset_mask_degenerate_keeps_everything(device: str, kind: str) -> None:
    """
    Degenerate clouds have no non-flat tetrahedron, so nothing is certified interior.

    Keeping every point is the conservative answer and a valid (if useless) superset -- the failure
    mode this guards against is the opposite one, where a flat tetrahedron's ill-conditioned inverse
    reports arbitrary points as interior and discards hull vertices.
    """
    rng = np.random.default_rng(15)
    if kind == "coplanar":
        points_np = np.column_stack([rng.standard_normal((500, 2)), np.zeros(500)])
    elif kind == "collinear":
        points_np = np.outer(np.linspace(0.0, 1.0, 100), np.array([1.0, 2.0, 3.0]))
    elif kind == "identical":
        points_np = np.ones((50, 3))
    else:
        points_np = rng.standard_normal((3, 3))

    points_wp = points_to_warp(points_np, device)
    assert od.convex_superset_mask(points_wp).numpy().all()


def test_convex_superset_mask_empty(device: str) -> None:
    points_wp = warp_empty(0, wp.vec3, device)
    mask_wp = od.convex_superset_mask(points_wp)
    assert mask_wp.shape == (0,)


@pytest.mark.parametrize("mask_device", ["cpu", "cuda:0"])
def test_support_sweep_agrees_across_devices(mask_device: str) -> None:
    """
    Both hull filters must agree on CPU and CUDA, which is not automatic.

    ``wp.launch_tiled`` runs exactly **one** lane per block on Warp 1.17's CPU backend -- the lane
    index from ``wp.tid()`` is always 0 -- so the block-wide ``wp.tile_max`` reduction the support
    sweep originally used silently reduced over one point per 64-point tile there. That returned an
    under-estimated support maximum, which made ``convex_subset_mask`` mark interior points (its
    soundness assertion above fails outright on CPU) and made ``convex_superset_mask`` build a
    shrunken shell. Both kernels are now lane-free; this pins that, on the device where every
    ``device``-fixture test is silent because the fixture prefers ``cuda:0``.
    """
    if mask_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    points_np = _cloud("gaussian", 5_000, seed=16)
    points_wp = points_to_warp(points_np, mask_device)
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    subset_np = od.convex_subset_mask(points_wp, n_directions=128).numpy()
    superset_np = od.convex_superset_mask(points_wp, subdivisions=2).numpy()

    assert set(np.flatnonzero(subset_np).tolist()) <= hull_scipy
    assert hull_scipy <= set(np.flatnonzero(superset_np).tolist())
    # The tile bug's signature was a wildly less selective filter, not a wrong-shaped one.
    assert superset_np.mean() < 0.05
