"""Regression tests for ``ordito.transform`` against Trimesh (CPU reference)."""

# This file tests ``ordito.Trimesh``'s private transform cache directly.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from ordito.mesh import _ORIENTATION_DEPENDENT_KEYS
from ordito.transform import TransformKind
from tests.comparisons import SET_VALUED_CACHE_KEYS, comparable_arrays, csr_row_sets
from tests.conftest import MESHES, OPEN_MESHES, populate_cache
from tests.conversions import (
    numpy_to_warp,
    points_to_open3d,
    points_to_warp,
    trimesh_to_pyvista,
    warp_empty,
    warp_to_trimesh,
)

if TYPE_CHECKING:
    import open3d as o3d

# One representative matrix per class, reused across the classification and cache tests. The
# similarity is a scale *composed with a rotation* on purpose: a pure scale leaves every direction
# fixed, which makes normals and tangent frames look carryable when they are not.
_ROTATION = od.transform.rotation_matrix((0.3, 0.5, 0.81), 1.1, (0.2, 0.1, 0.0))


def _compose(a: wp.mat44, b: wp.mat44) -> wp.mat44:
    """Host matrix product ``a @ b``, as a ``wp.mat44``."""
    product = np.array(a, dtype=np.float64).reshape(4, 4) @ np.array(b, dtype=np.float64).reshape(
        4, 4
    )
    return wp.mat44(*product.flatten().tolist())


TRANSFORMS: list[tuple[TransformKind, wp.mat44]] = [
    (TransformKind.IDENTITY, od.transform.translation_matrix((0.0, 0.0, 0.0))),
    (TransformKind.TRANSLATION, od.transform.translation_matrix((1.5, -2.0, 0.5))),
    (TransformKind.RIGID, _ROTATION),
    (TransformKind.REFLECTION, od.transform.reflection_matrix((0.3, 0.5, 0.81), (0.2, 0.1, 0.0))),
    (
        TransformKind.SIMILARITY,
        _compose(_ROTATION, od.transform.scale_matrix(2.0, (0.2, 0.1, 0.0))),
    ),
    (TransformKind.AFFINE, od.transform.scale_matrix((2.0, 1.0, 0.5))),
    (TransformKind.SINGULAR, od.transform.scale_matrix((1.0, 1.0, 0.0))),
]


# ---------------------------------------------------------------------------
# matrix builders
# ---------------------------------------------------------------------------


@pytest.mark.parity("transform_points", "trimesh")
def test_rotation_matrix_matches_trimesh() -> None:
    """Class A: ``rotation_matrix`` against ``trimesh.transformations.rotation_matrix``."""
    axis, angle, center = (0.3, 0.5, 0.81), 1.234, (1.0, -2.0, 0.5)
    matrix_wp = np.array(od.transform.rotation_matrix(axis, angle, center)).reshape(4, 4)
    matrix_tm = tm.transformations.rotation_matrix(angle, list(axis), list(center))
    assert np.allclose(matrix_wp, matrix_tm, rtol=1e-5, atol=1e-5)


def test_reflection_matrix_matches_trimesh() -> None:
    """Class A: ``reflection_matrix`` against ``trimesh.transformations.reflection_matrix``."""
    normal, center = (0.3, 0.5, 0.81), (1.0, -2.0, 0.5)
    matrix_wp = np.array(od.transform.reflection_matrix(normal, center)).reshape(4, 4)
    matrix_tm = tm.transformations.reflection_matrix(list(center), list(normal))
    assert np.allclose(matrix_wp, matrix_tm, rtol=1e-5, atol=1e-5)


def test_translation_and_scale_matrices_match_trimesh() -> None:
    """Class A: the two remaining builders against ``trimesh.transformations``."""
    offset = (1.0, -2.0, 0.5)
    translation_wp = np.array(od.transform.translation_matrix(offset)).reshape(4, 4)
    assert np.allclose(translation_wp, tm.transformations.translation_matrix(list(offset)))

    scale_wp = np.array(od.transform.scale_matrix(3.0, offset)).reshape(4, 4)
    scale_tm = tm.transformations.scale_matrix(3.0, list(offset))
    assert np.allclose(scale_wp, scale_tm, rtol=1e-5, atol=1e-5)


def test_rotation_matrix_zero_axis_raises() -> None:
    """Not a parity assert: the guard on a degenerate axis."""
    with pytest.raises(ValueError, match="non-zero axis"):
        od.transform.rotation_matrix((0.0, 0.0, 0.0), 1.0)


def test_rotation_matrix_nan_axis_raises() -> None:
    """
    Not a parity assert: a NaN-valued axis must not silently pass the zero-length guard.

    Every comparison against NaN is False, so a guard spelled ``norm == 0.0`` lets a NaN axis
    through and fills the whole matrix with NaN instead of raising.
    """
    with pytest.raises(ValueError, match="finite, non-zero axis"):
        od.transform.rotation_matrix((float("nan"), 0.0, 0.0), 0.5)


def test_reflection_matrix_zero_normal_raises() -> None:
    """Not a parity assert: the guard on a degenerate plane normal."""
    with pytest.raises(ValueError, match="non-zero normal"):
        od.transform.reflection_matrix((0.0, 0.0, 0.0))


def test_reflection_matrix_nan_normal_raises() -> None:
    """Not a parity assert: see ``test_rotation_matrix_nan_axis_raises``, same guard shape."""
    with pytest.raises(ValueError, match="finite, non-zero normal"):
        od.transform.reflection_matrix((float("nan"), 0.0, 0.0))


def test_scale_matrix_bad_length_raises() -> None:
    """Not a parity assert: the guard on a factor that is neither scalar nor length-3."""
    with pytest.raises(ValueError, match="scalar or length-3"):
        od.transform.scale_matrix((1.0, 2.0))


# ---------------------------------------------------------------------------
# buffer operations
# ---------------------------------------------------------------------------


@pytest.mark.parity("transform_points", "trimesh")
@pytest.mark.parametrize("matrix", [m for _, m in TRANSFORMS], ids=[k for k, _ in TRANSFORMS])
def test_transform_points_matches_trimesh(device: str, matrix: wp.mat44) -> None:
    """Class A: ``transform_points`` against ``trimesh.transformations.transform_points``."""
    points_np = np.random.default_rng(0).normal(size=(256, 3)).astype(np.float32)
    points_wp = points_to_warp(points_np, device)
    points_tm = tm.transformations.transform_points(
        points_np.astype(np.float64), np.array(matrix, dtype=np.float64).reshape(4, 4)
    )
    assert np.allclose(
        od.transform.transform_points(points_wp, matrix).numpy(), points_tm, rtol=1e-5, atol=1e-5
    )


def test_transform_points_in_place(device: str) -> None:
    """Not a parity assert: ``out=points`` writes the same answer the allocating form does."""
    points_np = np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32)
    matrix = _ROTATION
    allocated = od.transform.transform_points(points_to_warp(points_np, device), matrix)
    in_place = points_to_warp(points_np, device)
    returned = od.transform.transform_points(in_place, matrix, out=in_place)
    assert returned is in_place
    assert np.array_equal(in_place.numpy(), allocated.numpy())


def test_transform_points_accepts_a_device_matrix(device: str) -> None:
    """
    Ordito against ordito: the device-array matrix form agrees with the scalar one.

    The scalar form carries the oracle (``test_transform_points_matches_trimesh``); this pins the
    path ``registration.icp`` uses, which never brings its fitted matrix back to the host.
    """
    points_wp = points_to_warp(np.random.default_rng(2).normal(size=(64, 3)), device)
    scalar = od.transform.transform_points(points_wp, _ROTATION)
    on_device = od.transform.transform_points(
        points_wp, wp.array([_ROTATION], dtype=wp.mat44, device=device)
    )
    assert np.allclose(on_device.numpy(), scalar.numpy(), rtol=1e-5, atol=1e-5)


def test_transform_points_mismatched_out_length_raises(device: str) -> None:
    """
    Not a parity assert: an undersized ``out`` must raise, not silently overrun it.

    ``transform_points``'s device-array-matrix path launches straight into ``out`` with no shape
    check of its own (unlike the scalar-matrix ``wp.map`` path, which validates internally) --
    this is the guard that closes that gap for both paths alike.
    """
    points_wp = points_to_warp(np.random.default_rng(3).normal(size=(5, 3)), device)
    with pytest.raises(ValueError, match="out must have length"):
        od.transform.transform_points(points_wp, _ROTATION, out=warp_empty(2, wp.vec3, device))
    with pytest.raises(ValueError, match="out must have length"):
        od.transform.transform_points(
            points_wp,
            wp.array([_ROTATION], dtype=wp.mat44, device=device),
            out=warp_empty(2, wp.vec3, device),
        )


def test_transform_points_mismatched_matrix_array_length_raises(device: str) -> None:
    """Not a parity assert: the ``(1,)`` device-array contract, checked rather than assumed."""
    points_wp = points_to_warp(np.random.default_rng(4).normal(size=(5, 3)), device)
    with pytest.raises(ValueError, match="length-1"):
        od.transform.transform_points(
            points_wp, wp.array([_ROTATION, _ROTATION], dtype=wp.mat44, device=device)
        )


def test_as_mat44_mismatched_length_raises(device: str) -> None:
    """Not a parity assert: the ``(1,)`` device-array contract, at ``as_mat44`` itself."""
    with pytest.raises(ValueError, match="length-1"):
        od.transform.as_mat44(wp.array([_ROTATION, _ROTATION], dtype=wp.mat44, device=device))
    with pytest.raises(ValueError, match="length-1"):
        od.transform.as_mat44(warp_empty(0, wp.mat44, device))


def test_matrix_to_numpy_round_trips_a_scalar_and_a_device_matrix(device: str) -> None:
    """Not a parity assert: the host readback both forms resolve through, pinned directly."""
    expected = np.array(_ROTATION, dtype=np.float64).reshape(4, 4)
    assert np.array_equal(od.transform.matrix_to_numpy(_ROTATION), expected)
    on_device = wp.array([_ROTATION], dtype=wp.mat44, device=device)
    assert np.array_equal(od.transform.matrix_to_numpy(on_device), expected)


def test_transform_vectors_ignores_translation(device: str) -> None:
    """
    Not a library comparison: no reference exposes the vector map separately from the point map.

    Pins the defining difference -- a pure translation moves points and fixes directions -- which
    excludes the bug of routing directions through the point map.
    """
    vectors_np = np.random.default_rng(3).normal(size=(64, 3)).astype(np.float32)
    vectors_wp = points_to_warp(vectors_np, device)
    translation = od.transform.translation_matrix((5.0, -3.0, 2.0))
    assert np.allclose(
        od.transform.transform_vectors(vectors_wp, translation).numpy(),
        vectors_np,
        rtol=1e-5,
        atol=1e-5,
    )
    # ...and under a rotation it agrees with the point map about the origin.
    assert np.allclose(
        od.transform.transform_vectors(vectors_wp, _ROTATION).numpy(),
        tm.transformations.transform_points(
            vectors_np.astype(np.float64),
            np.array(od.transform.rotation_matrix((0.3, 0.5, 0.81), 1.1)).reshape(4, 4),
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_transform_normals_stays_perpendicular(device: str) -> None:
    """
    Not a library comparison: no reference binds the normal map on its own.

    Asserts the property that defines it -- the image normal is perpendicular to the images of two
    tangents -- under a *non-uniform* scale, where the naive linear map fails. The margin is what
    makes the test bite: routing these through ``transform_vectors`` instead measures a maximum
    tangent dot product of 0.53 against this assert's 1e-6.
    """
    rng = np.random.default_rng(4)
    normals_np = rng.normal(size=(64, 3))
    normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
    # Two tangents spanning each normal's plane.
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (64, 1))
    tangent_a = np.cross(normals_np, helper)
    tangent_a /= np.linalg.norm(tangent_a, axis=1, keepdims=True)
    tangent_b = np.cross(normals_np, tangent_a)

    matrix = od.transform.scale_matrix((3.0, 1.0, 0.4))
    linear = np.array(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
    mapped = od.transform.transform_normals(points_to_warp(normals_np, device), matrix).numpy()

    for tangent in (tangent_a, tangent_b):
        image = tangent @ linear.T
        image /= np.linalg.norm(image, axis=1, keepdims=True)
        assert np.abs(np.einsum("ij,ij->i", mapped.astype(np.float64), image)).max() < 1e-6
    assert np.allclose(np.linalg.norm(mapped, axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_transform_normals_singular_matrix_raises(device: str) -> None:
    """Not a parity assert: a flattening transform has no normal map."""
    normals_wp = points_to_warp(np.eye(3), device)
    with pytest.raises(ValueError, match="invertible"):
        od.transform.transform_normals(normals_wp, od.transform.scale_matrix((1.0, 1.0, 0.0)))


def test_transform_normals_singular_matrix_raises_on_empty_input(device: str) -> None:
    """
    Not a parity assert: the singularity check is a property of ``matrix``, not of ``normals``.

    An empty ``normals`` array used to make the early return skip ``normal_matrix`` entirely, so
    the same singular matrix that raises above was silently accepted whenever there was nothing to
    transform.
    """
    normals_wp = warp_empty(0, wp.vec3, device)
    with pytest.raises(ValueError, match="invertible"):
        od.transform.transform_normals(normals_wp, od.transform.scale_matrix((1.0, 1.0, 0.0)))


def test_transform_vectors_and_normals_accept_a_device_matrix(device: str) -> None:
    """
    Ordito against ordito: the ``(1,)`` device-array matrix form agrees with the scalar one.

    ``as_mat44``'s docstring has always claimed every entry point accepts the array form;
    ``transform_vectors``/``transform_normals`` were typed as a scalar ``wp.mat44`` only until this
    pinned the claim true for them too.
    """
    vectors_wp = points_to_warp(np.random.default_rng(5).normal(size=(32, 3)), device)
    on_device_matrix = wp.array([_ROTATION], dtype=wp.mat44, device=device)
    assert np.allclose(
        od.transform.transform_vectors(vectors_wp, on_device_matrix).numpy(),
        od.transform.transform_vectors(vectors_wp, _ROTATION).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    normals_np = vectors_wp.numpy()
    normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
    normals_wp = points_to_warp(normals_np, device)
    assert np.allclose(
        od.transform.transform_normals(normals_wp, on_device_matrix).numpy(),
        od.transform.transform_normals(normals_wp, _ROTATION).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parity("transform_mesh", "trimesh")
@pytest.mark.parity("transform_points", "pyvista")
def test_transform_points_matches_pyvista(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class A: ``transform_points`` against VTK's ``vtkTransformFilter`` through pyvista."""
    mesh_tm, mesh_wp = icosphere
    matrix_np = np.array(_ROTATION, dtype=np.float64).reshape(4, 4)
    moved_pv = trimesh_to_pyvista(mesh_tm).transform(matrix_np, inplace=False)
    assert np.allclose(
        od.transform.transform_points(mesh_wp.points, _ROTATION).numpy(),
        np.asarray(moved_pv.points),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parity("transform_points", "open3d")
def test_transform_points_matches_open3d(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class A: ``transform_points`` against ``open3d.geometry.PointCloud.transform``."""
    mesh_tm, mesh_wp = icosphere
    matrix_np = np.array(_ROTATION, dtype=np.float64).reshape(4, 4)
    cloud_o3d = cast(
        "o3d.geometry.PointCloud", points_to_open3d(mesh_tm.vertices).transform(matrix_np)
    )
    assert np.allclose(
        od.transform.transform_points(mesh_wp.points, _ROTATION).numpy(),
        np.asarray(cloud_o3d.points),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parity("transform_normals", "pyvista")
def test_transform_normals_matches_pyvista(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: the covector map against VTK's ``transform_all_input_vectors=True``.

    Under a **non-uniform** scale, which is what makes this non-vacuous: VTK normalizes its
    output like ordito does, and for any isometry the inverse transpose and the plain linear map
    coincide, so a rotation would pass even for an implementation using the wrong one. Measured on
    this fixture, the naive map sits 0.896 from VTK's answer against this assert's 1e-5.
    """
    mesh_tm, mesh_wp = icosphere
    matrix = od.transform.scale_matrix((3.0, 1.0, 0.4))
    matrix_np = np.array(matrix, dtype=np.float64).reshape(4, 4)

    mesh_pv = trimesh_to_pyvista(mesh_tm).compute_normals(point_normals=True, cell_normals=False)
    normals_pv = np.asarray(mesh_pv.point_data["Normals"], dtype=np.float64)
    moved_pv = mesh_pv.transform(matrix_np, transform_all_input_vectors=True, inplace=False)
    expected_pv = np.asarray(moved_pv.point_data["Normals"], dtype=np.float64)

    normals_wp = points_to_warp(normals_pv, mesh_wp.device)
    mapped_wp = od.transform.transform_normals(normals_wp, matrix).numpy()
    # Both sides are unit and share a sign convention (VTK's own normals are the input), so this
    # is elementwise rather than up-to-sign.
    assert np.allclose(mapped_wp, expected_pv, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_transform_mesh_matches_trimesh_apply_transform(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: ``transform_mesh`` against ``trimesh.Trimesh.apply_transform``, mirror included.

    trimesh reverses winding on a negative determinant too, so the face buffers are directly
    comparable rather than needing a canonical winding.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    matrix = TRANSFORMS[3][1]  # the reflection: the case where winding must flip
    assert od.transform.reverses_orientation(matrix)

    vertices_wp, faces_wp = od.transform.transform_mesh(mesh_wp.points, mesh_wp.indices, matrix)
    moved_tm = mesh_tm.copy()
    moved_tm.apply_transform(np.array(matrix, dtype=np.float64).reshape(4, 4))

    assert np.allclose(vertices_wp.numpy(), moved_tm.vertices, rtol=1e-5, atol=1e-5)
    assert np.array_equal(faces_wp.numpy().reshape(-1, 3), moved_tm.faces)


def test_transform_mesh_keeps_volume_positive(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the reason the winding flip exists, stated as an invariant.

    A mirrored mesh whose winding was *not* reversed has inward normals and a negative volume;
    measured -4.1888 against +4.1888 here, so the assert has three orders of margin on its sign.
    """
    _mesh_tm, mesh_wp = icosphere
    before = od.measures.volume(mesh_wp.points, mesh_wp.indices)
    assert before > 0.0
    vertices_wp, faces_wp = od.transform.transform_mesh(
        mesh_wp.points, mesh_wp.indices, TRANSFORMS[3][1]
    )
    assert od.measures.volume(vertices_wp, faces_wp) == pytest.approx(before, rel=1e-5)


def test_transform_mesh_mismatched_out_faces_length_raises(
    icosphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a parity assert: an undersized ``out_faces`` must raise, not silently overrun it."""
    _mesh_tm, mesh_wp = icosphere
    with pytest.raises(ValueError, match="out_faces must have length"):
        od.transform.transform_mesh(
            mesh_wp.points,
            mesh_wp.indices,
            _ROTATION,
            out_faces=warp_empty(3, wp.int32, mesh_wp.device),
        )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "matrix"), TRANSFORMS, ids=[k for k, _ in TRANSFORMS])
def test_classify_transform(kind: TransformKind, matrix: wp.mat44) -> None:
    """
    Not a library comparison: no reference classifies a matrix this way.

    ``trimesh.transformations.is_rigid`` answers one bit of the seven-way question. Parametrized
    over every member so the test cannot pass by always returning one class.
    """
    assert od.transform.classify_transform(matrix) == kind


def test_classify_transform_mirroring_scale_is_a_similarity() -> None:
    """
    Not a library comparison: the one case where orientation and metric class come apart.

    A uniform scale of -2 reverses orientation *and* changes lengths, so it is a similarity and not
    a reflection -- reporting `REFLECTION` would promise the length preservation that the cache
    strata rely on.
    """
    mirroring = od.transform.scale_matrix(-2.0)
    assert od.transform.classify_transform(mirroring) == TransformKind.SIMILARITY
    assert od.transform.reverses_orientation(mirroring)
    assert od.transform.transform_scale(mirroring) == pytest.approx(2.0, rel=1e-5)


def test_classify_transform_survives_composed_float32_rotations() -> None:
    """
    Not a library comparison: the drift the ``rtol`` default exists to absorb.

    50 rounded ``float32`` rotations composed in sequence still classify as `RIGID`; the point is
    that the tolerance is not so tight that ordinary composition demotes a rigid motion to
    `AFFINE` and silently costs a caller its cache.
    """
    composed = np.eye(4)
    step = np.array(od.transform.rotation_matrix((0.3, 0.5, 0.81), 0.17)).reshape(4, 4)
    for _ in range(50):
        composed = step.astype(np.float32) @ composed
    assert (
        od.transform.classify_transform(wp.mat44(*composed.flatten().tolist()))
        == TransformKind.RIGID
    )


def test_classify_transform_anisotropic_scale_is_affine_not_singular() -> None:
    """
    Not a library comparison: the singularity test must be scale-invariant under anisotropy too.

    A determinant-vs-trace-derived-scale singularity test flags a sufficiently anisotropic (but
    perfectly invertible) scale as `SINGULAR`, because a non-uniform scale's determinant is always
    below its isotropic-scale expectation by an amount that grows with the aspect ratio (AM-GM).
    ``diag(1000, 1, 1)`` is exactly the module's own `AFFINE` example and has determinant 1000 --
    not remotely singular -- but was misclassified `SINGULAR` before the singular-value-ratio test
    replaced the determinant-vs-scale one.
    """
    anisotropic = od.transform.scale_matrix((1000.0, 1.0, 1.0))
    assert od.transform.classify_transform(anisotropic) == TransformKind.AFFINE
    # And the normal map, which used the same flawed test independently, must not raise either.
    od.transform.normal_matrix(anisotropic)


def test_classify_transform_projective_matrix_is_singular() -> None:
    """
    Not a library comparison: a non-affine bottom row is not applied with a perspective divide.

    ``wp.transform_point`` drops the ``w`` divide, so nothing downstream of such a matrix is
    meaningful and the bottom rung is the honest answer.
    """
    projective = wp.mat44(1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0.5, 0, 0, 1.0)
    assert od.transform.classify_transform(projective) == TransformKind.SINGULAR


# ---------------------------------------------------------------------------
# Trimesh.transform and its cache strata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize(("kind", "matrix"), TRANSFORMS[1:], ids=[k for k, _ in TRANSFORMS[1:]])
def test_carried_cache_matches_recomputation(
    request: pytest.FixtureRequest, mesh_name: str, kind: TransformKind, matrix: wp.mat44
) -> None:
    """
    Ordito against ordito: every carried cache entry equals recomputing it from scratch.

    This is what makes the strata in ``ordito/mesh.py`` a measurement rather than an argument.
    The oracle is the *uncached* mesh: ``od.Trimesh(transformed_vertices, transformed_faces)``
    recomputes every property from the transformed buffers, and each key the transform carried (or
    rotated) must agree with it. Adding a key to a carry set that does not survive fails here.

    Non-vacuity: the carried set is asserted non-empty, and the run covers a closed and an open
    mesh -- four keys (``oriented_boundary_edges``, ``boundary_loops``, ``laplacian_operator`` and
    the tangent frames' gauge) survive a mirror on a *closed* mesh and fail on an open one, so a
    closed-only fixture would pass while carrying wrong values.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    moved = mesh.transform(matrix)

    assert moved is not mesh
    assert moved._cache, f"{kind} carried nothing at all"

    reference = od.Trimesh(moved.vertices, moved.faces)
    for key, carried in moved._cache.items():
        if key == "warp_mesh":
            continue
        if key in SET_VALUED_CACHE_KEYS:
            assert csr_row_sets(
                cast("tuple[wp.array[wp.int32], wp.array[wp.int32]]", carried)
            ) == csr_row_sets(getattr(reference, key)), (
                f"{kind} carried a stale {key} on {mesh_name}"
            )
            continue
        fresh = comparable_arrays(getattr(reference, key))
        for got, expected in zip(comparable_arrays(carried), fresh, strict=True):
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-4), (
                f"{kind} carried a stale {key} on {mesh_name}"
            )


@pytest.mark.parametrize("mesh_name", MESHES)
def test_transform_carries_the_expensive_operators_through_a_rigid_motion(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: a rigid motion aliases the heavy assemblies rather than rebuilding.

    The oracle for the *values* is ``test_carried_cache_matches_recomputation``; this pins the
    thing that makes the method worth having, which a value comparison cannot see -- that the
    carried entries are the same objects, so no work was done.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    cotmatrix, entries, angles = mesh.cotmatrix, mesh.cotmatrix_entries, mesh.face_angles
    moved = mesh.transform(_ROTATION)
    assert moved.cotmatrix is cotmatrix
    assert moved.cotmatrix_entries is entries
    assert moved.face_angles is angles


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize(("kind", "matrix"), TRANSFORMS[1:], ids=[k for k, _ in TRANSFORMS[1:]])
def test_transform_rotates_the_tangent_frames_where_it_can(
    request: pytest.FixtureRequest, mesh_name: str, kind: TransformKind, matrix: wp.mat44
) -> None:
    """
    Ordito against ordito: which transforms keep ``vertex_tangent_frames``, and how.

    ``test_carried_cache_matches_recomputation`` is the oracle for the frames' *values*, and it
    cannot see this: a key that is silently **dropped** is simply absent from the carried cache, so
    a value comparison over the carried keys passes vacuously. That is exactly how the frames came
    to be rebuilt on every rigid pose change while `Trimesh.transform` documented that directions
    are rotated -- the branch that rotates them was unreachable, because the key was in no carry
    set the branch's own guard would let through. This asserts the membership, which is the half
    the value test structurally cannot check.

    Three regimes, and each pins a different mechanism:

    - **translation** carries the frame *verbatim*, the same objects, because `_carry_directions`
      returns before rotating anything -- a translation moves no direction.
    - **rigid** and **similarity** carry a *rotated* frame: new arrays, since a rotation genuinely
      moves the basis. The gauge commutes with both because it is built from unit directions.
    - **reflection**, **affine** and **singular** drop it. The mirror is the interesting one: it is
      dropped by the `_ORIENTATION_DEPENDENT_KEYS` subtraction rather than by the stratum, because
      ``basis_y = normal x basis_x`` flips with the winding while the normal itself does not.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    frames = mesh.vertex_tangent_frames
    moved = mesh.transform(matrix)
    carried = cast(
        "tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]] | None",
        moved._cache.get("vertex_tangent_frames"),
    )

    if kind in (TransformKind.REFLECTION, TransformKind.AFFINE, TransformKind.SINGULAR):
        assert carried is None, f"{kind} carried a frame it does not preserve"
        return

    assert carried is not None, f"{kind} dropped the tangent frames instead of carrying them"
    if kind is TransformKind.TRANSLATION:
        assert all(got is original for got, original in zip(carried, frames, strict=True)), (
            "a translation moves no direction, so the frame should be the same arrays"
        )
        return

    # Rotated, not aliased -- and the third element is `vertex_normals` by that property's
    # documented invariant, so `transform` must publish the same array into both slots rather than
    # leaving a stale normal beside a fresh frame.
    assert all(got is not original for got, original in zip(carried, frames, strict=True))
    assert moved._cache["vertex_normals"] is carried[2]

    basis_x, basis_y, normal = (component.numpy() for component in carried)
    assert np.allclose(np.linalg.norm(basis_x, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.einsum("ij,ij->i", basis_x, normal), 0.0, atol=1e-5)
    assert np.allclose(np.cross(normal, basis_x), basis_y, atol=1e-5)
    # Non-vacuity: the rotation must actually have moved the basis, or "rotated" is untested.
    assert float(np.abs(basis_x - frames[0].numpy()).max()) > 1e-3


def test_adjacency_projections_and_convex_stay_consistent(
    cave_cube: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a parity assert: the two adjacency quantities never contradict each other.

    ``face_adjacency_convex`` is exactly ``face_adjacency_projections <= TOLERANCE_MERGE``, and
    neither is
    carried -- the mask because a rotation's ~1e-7 flips it on a coplanar pair, the projection so
    it cannot be left holding a value the recomputed mask disagrees with. On ``cave_cube``, whose
    box faces are coplanar, that pairing is what a stratum holding only one of them would break.
    """
    _mesh_tm, mesh_wp = cave_cube
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    moved = mesh.transform(_ROTATION)
    assert "face_adjacency_projections" not in moved._cache
    assert "face_adjacency_convex" not in moved._cache

    projections = moved.face_adjacency_projections.numpy()
    convex = moved.face_adjacency_convex.numpy()
    assert np.array_equal(convex, projections <= od.constants.TOLERANCE_MERGE)
    assert convex.any()  # non-vacuity: this fixture is not convex everywhere
    assert not convex.all()
    # ...and the pairs that make the mask unsafe to carry are inside that band: this rotation
    # puts two coplanar pairs at 1.9e-09 and 6.5e-09, where a different one would put them past
    # 1e-8 and flip the answer.
    assert (np.abs(projections) < od.constants.TOLERANCE_MERGE).any()


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parametrize("key", ["volume", "center_mass", "moment_inertia"])
def test_mass_properties_are_never_carried(
    request: pytest.FixtureRequest, mesh_name: str, key: str
) -> None:
    """
    Not a parity assert: the mass properties are dropped even by a translation, and must be.

    All three integrate tetrahedra from the origin, which telescopes to an origin-independent
    answer only on a **closed** surface. On an open mesh a translation changes them -- measured
    2.3e-01 on ``volume`` and 8.7e-01 on ``moment_inertia`` for a hemisphere moved by
    (1.5, -2, 0.5) -- so no stratum may hold them. Asserting the drop *and* the disagreement keeps
    this from reading as arbitrary conservatism.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    assert key in mesh._cache

    moved = mesh.transform(od.transform.translation_matrix((1.5, -2.0, 0.5)))
    assert key not in moved._cache

    before = np.array(getattr(mesh, key), dtype=np.float64).ravel()
    after = np.array(getattr(moved, key), dtype=np.float64).ravel()
    assert not np.allclose(before, after, rtol=1e-3, atol=1e-3)


def test_transform_identity_returns_self(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a parity assert: the identity short-circuit does no work at all."""
    _mesh_tm, mesh_wp = icosphere
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.transform(od.transform.translation_matrix((0.0, 0.0, 0.0))) is mesh


def test_transform_mirror_drops_the_orientation_dependent_caches(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a parity assert: the mirror case drops the caches that reverse with the winding.

    On an *open* mesh, which is where three of these actually differ -- a closed fixture carries
    them wrongly and no value comparison notices.
    """
    _mesh_tm, mesh_wp = hemisphere
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    carried_before = set(mesh._cache)
    assert _ORIENTATION_DEPENDENT_KEYS & carried_before, "fixture did not populate the keys tested"

    moved = mesh.transform(TRANSFORMS[3][1])
    assert not (_ORIENTATION_DEPENDENT_KEYS & set(moved._cache))


def test_transform_singular_carries_topology_only(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a parity assert: a flattening transform keeps connectivity and drops the predicates.

    ``nondegenerate_faces`` is the one that matters: a bijection preserves it, and a singular map
    makes every face degenerate, so carrying it would report a flattened mesh as sound.
    """
    _mesh_tm, mesh_wp = icosphere
    mesh = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp))
    assert "nondegenerate_faces" in mesh._cache
    assert mesh.nondegenerate_faces.numpy().all()
    # Rank *one*, not rank two: projecting a sphere onto a plane leaves most faces with area, so
    # a rank-2 map would not make the carried mask wrong and the test would not bite.
    moved = mesh.transform(od.transform.scale_matrix((1.0, 0.0, 0.0)))
    assert "nondegenerate_faces" not in moved._cache
    assert not moved.nondegenerate_faces.numpy().any()


def test_transform_assume_skips_classification(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a parity assert: ``assume=`` selects the stratum the classifier would have picked.

    Passing the *correct* promise must reach the same cache as inferring it, which is the only
    property that can be checked -- an incorrect promise is documented as unchecked.
    """
    _mesh_tm, mesh_wp = icosphere
    inferred = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp)).transform(_ROTATION)
    promised = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp)).transform(
        _ROTATION, assume="rigid"
    )
    assert set(inferred._cache) == set(promised._cache)


def test_transform_bad_assume_raises(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a parity assert: the guard on an unknown promise."""
    _mesh_tm, mesh_wp = icosphere
    with pytest.raises(ValueError, match="isometry"):
        od.Trimesh.from_warp_mesh(mesh_wp).transform(_ROTATION, assume="isometry")


@pytest.mark.parametrize("mesh_name", MESHES)
def test_transform_agrees_with_the_free_function(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the cached path and the buffer path produce the same mesh.

    ``transform_mesh`` carries the oracle (``test_transform_mesh_matches_trimesh_apply_transform``);
    this pins that the cache-carrying wrapper does not change the geometry it wraps.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    matrix = TRANSFORMS[4][1]
    moved = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp)).transform(matrix)
    vertices_wp, faces_wp = od.transform.transform_mesh(mesh_wp.points, mesh_wp.indices, matrix)
    assert np.allclose(moved.vertices.numpy(), vertices_wp.numpy(), rtol=1e-5, atol=1e-5)
    assert np.array_equal(moved.faces.numpy(), faces_wp.numpy())


def test_transform_round_trip_recovers_the_mesh(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: a transform composed with its inverse returns the original mesh, via trimesh.

    Uses ``trimesh.transformations.inverse_matrix`` to build the inverse, so the round trip is a
    genuine composition rather than ordito checking its own arithmetic twice.
    """
    _mesh_tm, mesh_wp = icosphere
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    inverse = wp.mat44(
        *tm.transformations.inverse_matrix(np.array(_ROTATION, dtype=np.float64).reshape(4, 4))
        .flatten()
        .tolist()
    )
    restored = mesh.transform(_ROTATION).transform(inverse)
    assert np.allclose(restored.vertices.numpy(), mesh.vertices.numpy(), rtol=1e-5, atol=1e-5)
    assert np.array_equal(restored.faces.numpy(), mesh.faces.numpy())


def test_transform_matches_trimesh_end_to_end(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: ``Trimesh.transform`` against ``trimesh.Trimesh.apply_transform``.

    The end-to-end claim -- that the cached facade and trimesh's mutating method agree on the mesh,
    its area and its volume after a similarity.
    """
    mesh_tm, mesh_wp = icosphere
    matrix = TRANSFORMS[4][1]
    moved = populate_cache(od.Trimesh.from_warp_mesh(mesh_wp)).transform(matrix)

    moved_tm = mesh_tm.copy()
    moved_tm.apply_transform(np.array(matrix, dtype=np.float64).reshape(4, 4))

    assert np.allclose(moved.vertices.numpy(), moved_tm.vertices, rtol=1e-5, atol=1e-5)
    assert moved.area == pytest.approx(moved_tm.area, rel=1e-4)
    assert od.measures.volume(moved.vertices, moved.faces) == pytest.approx(
        moved_tm.volume, rel=1e-4
    )
    assert np.allclose(
        np.asarray(warp_to_trimesh(moved.vertices, moved.faces).face_normals),
        moved_tm.face_normals,
        rtol=1e-4,
        atol=1e-4,
    )


def test_transform_updates_the_bounding_box_under_a_translation(
    icosphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: the host-updated box equals the recomputed one.

    ``bounds`` carries the oracle through ``od.bounds.aabb``; this pins the shortcut that avoids
    re-reducing the vertex buffer, which is the translation stratum's whole point.
    """
    _mesh_tm, mesh_wp = icosphere
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    _ = mesh.bounds, mesh.centroid
    moved = mesh.transform(od.transform.translation_matrix((1.5, -2.0, 0.5)))
    assert "bounds" in moved._cache
    reference = od.Trimesh(moved.vertices, moved.faces)
    assert np.allclose(np.array(moved.bounds), np.array(reference.bounds), rtol=1e-5, atol=1e-5)
    assert np.allclose(np.array(moved.centroid), np.array(reference.centroid), rtol=1e-5, atol=1e-5)


def test_transform_empty_mesh(device: str) -> None:
    """Not a parity assert: an empty mesh transforms to an empty mesh rather than raising."""
    vertices_wp, faces_wp = numpy_to_warp(np.zeros((0, 3)), np.zeros(0, dtype=np.int32), device)
    moved = od.Trimesh(vertices_wp, faces_wp).transform(_ROTATION)
    assert moved.n_vertices == 0
    assert moved.n_faces == 0
