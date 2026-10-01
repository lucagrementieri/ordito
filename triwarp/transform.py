"""
Affine transforms: build a ``4x4`` matrix, apply it to a buffer, and classify what it preserves.

Three layers, and which one a caller wants depends on what they hold. The **buffer ops**
([`transform_points`][triwarp.transform.transform_points] and its vector / normal / mesh siblings)
take a matrix and an array and support in-place operation through ``out=``. The **builders**
([`translation_matrix`][triwarp.transform.translation_matrix],
[`rotation_matrix`][triwarp.transform.rotation_matrix],
[`scale_matrix`][triwarp.transform.scale_matrix],
[`reflection_matrix`][triwarp.transform.reflection_matrix]) compose the matrix on the host, each
about an optional ``center``. And [`classify_transform`][triwarp.transform.classify_transform]
answers what a matrix *preserves* -- lengths, angles, orientation -- which is what
[`Trimesh.transform`][triwarp.mesh.Trimesh.transform] reads to decide how much of its cache
survives.

Points, vectors and normals are three different maps and picking the wrong one is a silent error
rather than a loud one:

| quantity | maps by | translation |
|---|---|---|
| position | ``M`` | applied |
| displacement, tangent | linear block of ``M`` | ignored |
| **normal** | **inverse transpose** of the linear block | ignored |

The first two agree on everything but the translation, so a normal pushed through
[`transform_vectors`][triwarp.transform.transform_vectors] is *correct under any isometry* and
tilts off the surface under a shear or a non-uniform scale -- tens of degrees off under an
ordinary axis scale. Only unit scale hides it, which is why the distinction gets its own entry
point.

See Also
--------
[`triwarp.registration`][triwarp.registration]
    Fits a transform ([`procrustes`][triwarp.registration.procrustes],
    [`icp`][triwarp.registration.icp]); this module applies one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import cast

import numpy as np
import warp as wp

import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import require_same_device
from triwarp.kernels import repair as kernel_repair
from triwarp.kernels import transform as kernel_transform

# Relative tolerance on the Gram matrix ``R Rt`` when deciding whether a linear block is a
# rotation, a reflection or a uniform scale. A matrix composed from several rotations drifts off
# orthogonality in ``float32``, so an exact test would demote a rigid motion to
# ``TransformKind.AFFINE`` -- correct, but it throws away the cache carry-forward that
# classification exists to enable. 1e-5 is the package's usual comparison tolerance and holds for a
# few dozen composed float32 rotations; past that a caller should pass ``assume=`` rather than
# loosen this.
ORTHOGONALITY_RTOL = 1e-5


class TransformKind(StrEnum):
    """
    What an affine transform preserves, ordered from most structure to least.

    Each member's guarantees *include* every later member's: a `TRANSLATION` is a `RIGID` motion
    whose rotation is the identity, and a `RIGID` motion is a `SIMILARITY` of scale 1. The
    ordering is what makes this useful as a cache key -- see
    [`Trimesh.transform`][triwarp.mesh.Trimesh.transform].

    Attributes
    ----------
    IDENTITY
        Changes nothing.
    TRANSLATION
        Identity linear block. Preserves lengths, angles, orientation **and every direction**, so
        normals and tangent frames are unchanged rather than merely rotated.
    RIGID
        Orthogonal linear block with positive determinant: a rotation, optionally with a
        translation. Preserves lengths, angles and orientation; rotates directions.
    REFLECTION
        Orthogonal linear block with negative determinant *and unit scale*. Preserves lengths and
        unsigned angles but **reverses orientation**, so face winding must be flipped to keep
        normals outward. A mirroring transform that also scales is `SIMILARITY`, not this.
    SIMILARITY
        Orthogonal up to a single factor: a uniform scale, optionally rotated, mirrored and
        translated. Preserves angles and therefore cotangent weights; scales lengths by
        [`transform_scale`][triwarp.transform.transform_scale] and areas by its square. May or
        may not reverse orientation -- unlike the other members this says nothing about that, so
        read the determinant sign separately.
    AFFINE
        An invertible map that is none of the above -- a non-uniform scale or a shear. Being a
        bijection it still preserves connectivity, incidence and self-intersection; nothing
        metric.
    SINGULAR
        Not an invertible affine map: a rank-deficient linear block, or a bottom row that makes
        the matrix projective rather than affine. Flattens the mesh, so even the predicates a
        bijection would preserve are gone and only connectivity survives.
    """

    IDENTITY = "identity"
    TRANSLATION = "translation"
    RIGID = "rigid"
    REFLECTION = "reflection"
    SIMILARITY = "similarity"
    AFFINE = "affine"
    SINGULAR = "singular"


def transform_points(
    points: wp.array[wp.vec3],
    matrix: wp.mat44 | wp.array[wp.mat44],
    *,
    out: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.vec3]:
    """
    Apply an affine transform to a buffer of positions.

    Parameters
    ----------
    points
        ``(n,)`` positions.
    matrix
        ``4x4`` transform, as a scalar ``wp.mat44`` or a ``(1,)`` ``wp.array[wp.mat44]``. The
        array form keeps a fitted transform on the device, so an
        [`icp`][triwarp.registration.icp] result can be applied without a host readback.
    out
        ``(n,)`` destination, allocated when ``None``. Pass ``out=points`` to transform in place.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` transformed positions: ``out``, or a freshly allocated buffer on ``points.device``.

    Raises
    ------
    ValueError
        If ``out`` is given and its length differs from ``points``, or ``matrix`` is a device
        array without exactly one entry.
    RuntimeError
        If ``points``, ``matrix`` and ``out`` are not all on one device.

    Notes
    -----
    Safe in place: each thread reads exactly the element it writes.

    See Also
    --------
    [`transform_vectors`][triwarp.transform.transform_vectors]
    [`transform_normals`][triwarp.transform.transform_normals]
    [`transform_mesh`][triwarp.transform.transform_mesh]
    """
    require_same_device(points=points, matrix=matrix, out=out)
    result = _alloc_or_out(out, points.size, wp.vec3, points.device)
    if points.size == 0:
        return result
    if result.size != points.size:
        raise ValueError(f"transform_points: out must have length {points.size}, got {result.size}")
    if isinstance(matrix, wp.array):
        # The one path that never crosses to host: the kernel reads the device pointer directly,
        # so a fitted (e.g. icp) transform never has to leave the device. `apply_transform_mat44`
        # reads `matrix[0]` unconditionally, which is why the (1,) shape is checked above rather
        # than left to a silent partial read.
        if matrix.size != 1:
            raise ValueError(
                "transform_points: matrix must be a length-1 wp.array[wp.mat44], got shape "
                f"{tuple(matrix.shape)}"
            )
        _launch.launch(
            kernel_transform.apply_transform_mat44,
            dim=points.size,
            inputs=[points, matrix, result],
            device=points.device,
        )
        return result
    _launch.map(kernel_transform.transform_point_mat44, points, matrix, out=result)
    return result


def transform_vectors(
    vectors: wp.array[wp.vec3],
    matrix: wp.mat44 | wp.array[wp.mat44],
    *,
    out: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.vec3]:
    """
    Apply the linear block of an affine transform to a buffer of directions, ignoring translation.

    For a displacement, a velocity or a tangent vector. **Not** for a normal unless the transform
    is an isometry -- see [`transform_normals`][triwarp.transform.transform_normals] and this
    module's docstring. The result is not renormalized: a scale in ``matrix`` scales the output.

    Parameters
    ----------
    vectors
        ``(n,)`` directions.
    matrix
        ``4x4`` transform; its translation column is ignored. Scalar ``wp.mat44`` or a ``(1,)``
        ``wp.array[wp.mat44]`` -- unlike [`transform_points`][triwarp.transform.transform_points]
        there is no on-device kernel for this map, so the array form still costs one host readback
        (see [`as_mat44`][triwarp.transform.as_mat44]).
    out
        ``(n,)`` destination, allocated when ``None``. Pass ``out=vectors`` to transform in place.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` transformed directions: ``out``, or a freshly allocated buffer on
        ``vectors.device``.

    Raises
    ------
    ValueError
        If ``matrix`` is a device array without exactly one entry.
    RuntimeError
        If ``vectors``, ``matrix`` and ``out`` are not all on one device.

    See Also
    --------
    [`transform_points`][triwarp.transform.transform_points]
    [`transform_normals`][triwarp.transform.transform_normals]
    """
    require_same_device(vectors=vectors, matrix=matrix, out=out)
    result = _alloc_or_out(out, vectors.size, wp.vec3, vectors.device)
    if vectors.size == 0:
        return result
    _launch.map(kernel_transform.transform_vector_mat44, vectors, as_mat44(matrix), out=result)
    return result


def transform_normals(
    normals: wp.array[wp.vec3],
    matrix: wp.mat44 | wp.array[wp.mat44],
    *,
    out: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.vec3]:
    """
    Map unit normals through an affine transform by its inverse transpose, and renormalize.

    A normal is a covector: it maps by the inverse transpose of the linear block, which is the
    only map keeping it perpendicular to the transformed surface under a shear or a non-uniform
    scale. Zero normals stay zero, the degenerate-face convention
    [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas] writes.

    Parameters
    ----------
    normals
        ``(n,)`` normals, unit or zero.
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array. Its linear block must be invertible;
        the host-side inverse this needs (see [`normal_matrix`][triwarp.transform.normal_matrix])
        is computed once and reused, regardless of ``normals``' length.
    out
        ``(n,)`` destination, allocated when ``None``. Pass ``out=normals`` to transform in place.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` unit (or zero) normals: ``out``, or a freshly allocated buffer on
        ``normals.device``.

    Raises
    ------
    ValueError
        If the linear block of ``matrix`` is singular, so no normal map exists -- checked even
        when ``normals`` is empty, since the matrix itself is what's invalid.
    RuntimeError
        If ``normals``, ``matrix`` and ``out`` are not all on one device.

    Notes
    -----
    Orientation is *not* corrected: a mirroring ``matrix`` leaves these pointing inward until the
    face winding is reversed too. [`transform_mesh`][triwarp.transform.transform_mesh] is the
    entry point that handles both.

    See Also
    --------
    [`normal_matrix`][triwarp.transform.normal_matrix]
    [`transform_vectors`][triwarp.transform.transform_vectors]
    """
    require_same_device(normals=normals, matrix=matrix, out=out)
    # Computed before the empty-input early return: the matrix's own invertibility is a property
    # of `matrix` alone, not of how many normals there are, so a zero-length `normals` must not
    # silently skip it.
    linear_normal_matrix = normal_matrix(matrix)
    result = _alloc_or_out(out, normals.size, wp.vec3, normals.device)
    if normals.size == 0:
        return result
    _launch.map(kernel_transform.transform_normal_mat33, normals, linear_normal_matrix, out=result)
    return result


def transform_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    matrix: wp.mat44 | wp.array[wp.mat44],
    *,
    out_vertices: wp.array[wp.vec3] | None = None,
    out_faces: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Transform a mesh's vertices, reversing face winding when the transform mirrors.

    The winding flip is what keeps normals outward: a transform with negative determinant maps the
    face's corner cross product to ``-M n``, so reversing the corner order restores ``+M n``.
    Without it a mirrored mesh has inward normals and a negative
    [`volume`][triwarp.measures.volume].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array.
    out_vertices, out_faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` destinations, allocated when ``None``. Pass the
        inputs to transform in place.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` transformed positions.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` face buffer, corner-reversed if ``matrix`` mirrors and otherwise a copy
        of the input.

    Raises
    ------
    ValueError
        If ``out_vertices`` or ``out_faces`` is given and its length differs from ``vertices`` or
        ``faces``, or ``matrix`` is a device array without exactly one entry.
    RuntimeError
        If ``vertices``, ``faces``, ``matrix``, ``out_vertices`` and ``out_faces`` are not all on
        one device.

    See Also
    --------
    [`transform_points`][triwarp.transform.transform_points]
    [`Trimesh.transform`][triwarp.mesh.Trimesh.transform]
        The cached-mesh entry point, which carries derived quantities forward.
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        matrix=matrix,
        out_vertices=out_vertices,
        out_faces=out_faces,
    )
    new_vertices = transform_points(vertices, matrix, out=out_vertices)
    new_faces = _alloc_or_out(out_faces, faces.size, wp.int32, faces.device)
    if new_faces.size != faces.size:
        raise ValueError(
            f"transform_mesh: out_faces must have length {faces.size}, got {new_faces.size}"
        )
    n_faces = faces.size // 3
    if reverses_orientation(matrix):
        if n_faces > 0:
            _launch.launch(
                kernel_repair.reverse_face_winding,
                dim=n_faces,
                inputs=[faces, new_faces],
                device=faces.device,
            )
    elif new_faces.ptr != faces.ptr:
        _launch.copy(new_faces, faces)
    return new_vertices, new_faces


def translation_matrix(offset: wp.vec3 | Sequence[float]) -> wp.mat44:
    """
    Homogeneous matrix translating by ``offset``.

    Parameters
    ----------
    offset
        Length-3 translation.

    Returns
    -------
    wp.mat44
        A `TransformKind.TRANSLATION` matrix.

    Examples
    --------
    ```python
    up = tw.transform.translation_matrix((0.0, 0.0, 1.0))
    moved = tw.transform.transform_points(v, up)
    ```

    See Also
    --------
    [`rotation_matrix`][triwarp.transform.rotation_matrix]
    [`scale_matrix`][triwarp.transform.scale_matrix]
    [`reflection_matrix`][triwarp.transform.reflection_matrix]
    """
    return _compose(np.eye(3), _vec3_host(offset, "offset"))


def rotation_matrix(
    axis: wp.vec3 | Sequence[float], angle: float, center: wp.vec3 | Sequence[float] | None = None
) -> wp.mat44:
    """
    Homogeneous matrix rotating by ``angle`` radians about ``axis``, through ``center``.

    Parameters
    ----------
    axis
        Length-3 rotation axis; normalized internally, so its magnitude is ignored.
    angle
        Rotation angle in radians, counter-clockwise about ``axis`` by the right-hand rule.
    center
        Fixed point of the rotation, the origin when ``None``.

    Returns
    -------
    wp.mat44
        A `TransformKind.RIGID` matrix (`TransformKind.IDENTITY` at ``angle = 0``).

    Raises
    ------
    ValueError
        If ``axis`` is zero-length or not finite, so no rotation is defined.

    See Also
    --------
    [`translation_matrix`][triwarp.transform.translation_matrix]
    [`scale_matrix`][triwarp.transform.scale_matrix]
    """
    direction = _vec3_host(axis, "axis")
    norm = float(np.linalg.norm(direction))
    # `not np.isfinite(norm)` rather than only `norm == 0.0`: a NaN- or Inf-valued axis (e.g. one
    # built from a degenerate cross product upstream) gives a non-finite `norm`, and every
    # comparison against NaN is False, so `== 0.0` alone would let it through and silently fill the
    # whole matrix with NaN.
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation_matrix requires a finite, non-zero axis")
    direction = direction / norm
    # Rodrigues' rotation formula. Built on the host in float64 and rounded once on the way into
    # the float32 ``wp.mat44``, so composing a handful of these stays inside
    # ``ORTHOGONALITY_RTOL``.
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # `np.cross(np.eye(3), direction)`'s row i is `e_i x direction`, which expands to exactly the
    # skew-symmetric cross-product matrix Rodrigues' formula needs -- no need to spell out the six
    # signed entries by hand.
    cross = np.cross(np.eye(3), direction)
    rotation = cos_a * np.eye(3) + sin_a * cross + (1.0 - cos_a) * np.outer(direction, direction)
    return _compose_about(rotation, center)


def scale_matrix(
    factor: float | Sequence[float], center: wp.vec3 | Sequence[float] | None = None
) -> wp.mat44:
    """
    Homogeneous matrix scaling about ``center``, uniformly or per axis.

    Parameters
    ----------
    factor
        A single scale factor, or a length-3 per-axis factor. A per-axis factor whose components
        differ is a `TransformKind.AFFINE` transform, not a `TransformKind.SIMILARITY` one: it
        does not preserve angles, so the cotangent weights of a mesh scaled that way have to be
        rebuilt.
    center
        Fixed point of the scaling, the origin when ``None``.

    Returns
    -------
    wp.mat44
        A `TransformKind.SIMILARITY` matrix for a uniform positive ``factor``,
        `TransformKind.AFFINE` otherwise.

    Raises
    ------
    ValueError
        If ``factor`` is neither a scalar nor length-3.

    See Also
    --------
    [`translation_matrix`][triwarp.transform.translation_matrix]
    [`rotation_matrix`][triwarp.transform.rotation_matrix]
    """
    factors = np.asanyarray(factor, dtype=np.float64).reshape(-1)
    if factors.size == 1:
        factors = np.repeat(factors, 3)
    elif factors.size != 3:
        raise ValueError(f"scale_matrix factor must be a scalar or length-3, got {factors.size}")
    return _compose_about(np.diag(factors), center)


def reflection_matrix(
    normal: wp.vec3 | Sequence[float], center: wp.vec3 | Sequence[float] | None = None
) -> wp.mat44:
    """
    Homogeneous matrix reflecting through the plane with the given ``normal`` and point.

    Parameters
    ----------
    normal
        Length-3 plane normal; normalized internally.
    center
        A point on the mirror plane, the origin when ``None``.

    Returns
    -------
    wp.mat44
        A `TransformKind.REFLECTION` matrix. Applying it to a mesh reverses face winding --
        [`transform_mesh`][triwarp.transform.transform_mesh] and
        [`Trimesh.transform`][triwarp.mesh.Trimesh.transform] do that for you,
        [`transform_points`][triwarp.transform.transform_points] does not.

    Raises
    ------
    ValueError
        If ``normal`` is zero-length or not finite, so no plane is defined.

    See Also
    --------
    [`rotation_matrix`][triwarp.transform.rotation_matrix]
    [`triwarp.points.half_space_mask`][]
    """
    direction = _vec3_host(normal, "normal")
    norm = float(np.linalg.norm(direction))
    # See rotation_matrix's identical guard for why this is `not np.isfinite` and not `== 0.0`.
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("reflection_matrix requires a finite, non-zero normal")
    direction = direction / norm
    return _compose_about(np.eye(3) - 2.0 * np.outer(direction, direction), center)


def normal_matrix(matrix: wp.mat44 | wp.array[wp.mat44]) -> wp.mat33:
    """
    Inverse transpose of a transform's linear block: the map that carries normals.

    Parameters
    ----------
    matrix
        ``4x4`` transform with an invertible linear block, scalar or ``(1,)`` device array.

    Returns
    -------
    wp.mat33
        The ``3x3`` matrix [`transform_normals`][triwarp.transform.transform_normals] applies.
        For an isometry this equals the linear block itself.

    Raises
    ------
    ValueError
        If the linear block is singular to within
        [`ORTHOGONALITY_RTOL`][triwarp.transform.ORTHOGONALITY_RTOL] -- the same tolerance-based
        test [`classify_transform`][triwarp.transform.classify_transform] uses, not an exact
        ``det == 0`` check, so a near-singular block (e.g. one axis scaled by ``1e-40``) raises
        here instead of silently inverting into an overflowing ``wp.mat33``.

    See Also
    --------
    [`transform_normals`][triwarp.transform.transform_normals]
    [`classify_transform`][triwarp.transform.classify_transform]
    """
    linear = matrix_to_numpy(matrix)[:3, :3]
    if _is_singular(linear, ORTHOGONALITY_RTOL):
        raise ValueError("normal_matrix requires an invertible linear block, got a singular one")
    return wp.mat33(*np.linalg.inv(linear).T.flatten())


def classify_transform(
    matrix: wp.mat44 | wp.array[wp.mat44], *, rtol: float = ORTHOGONALITY_RTOL
) -> TransformKind:
    """
    Decide what an affine transform preserves: lengths, angles, orientation, or only connectivity.

    Reads the ``3x3`` linear block's Gram matrix ``R Rt``. When it is a positive multiple of the
    identity the transform is a similarity, and the multiple is the squared scale; the sign of
    ``det R`` then separates rotations from reflections, and an identity block separates a pure
    translation. Everything else is `TransformKind.AFFINE`.

    Parameters
    ----------
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array. A device array costs one host readback.
    rtol
        Relative tolerance on the Gram matrix, defaulting to
        [`ORTHOGONALITY_RTOL`][triwarp.transform.ORTHOGONALITY_RTOL]. A matrix composed from many
        ``float32`` rotations eventually drifts past any fixed tolerance and is classified
        `TransformKind.AFFINE` -- conservative, never wrong, but it costs the caller a cache. Pass
        ``assume=`` to [`Trimesh.transform`][triwarp.mesh.Trimesh.transform] rather than widening
        this.

    Returns
    -------
    TransformKind
        The strongest class the matrix satisfies.

    Notes
    -----
    Singularity is decided from the linear block's singular values, not from its determinant: a
    block is rank-deficient to within ``rtol`` when its smallest singular value is at most an
    ``rtol`` fraction of its largest -- the reciprocal condition number, which stays correct
    however anisotropic the block is. A determinant-vs-scale test would not: a non-uniform scale
    can have an arbitrarily small determinant relative to its trace while remaining perfectly
    invertible (``diag(1000, 1, 1)`` is `TransformKind.AFFINE`, not singular, despite a determinant
    500x below its isotropic-scale expectation).

    Orientation is **not** part of the answer except at `TransformKind.REFLECTION`, which is the
    unit-scale mirror. A mirroring scale is `TransformKind.SIMILARITY`, so a caller that needs to
    know whether winding flips must read ``det`` of the linear block rather than switching on this
    result.

    Examples
    --------
    ```python
    kind = tw.transform.classify_transform(tw.transform.rotation_matrix((0.0, 0.0, 1.0), 0.5))
    ```

    See Also
    --------
    [`TransformKind`][triwarp.transform.TransformKind]
    [`Trimesh.transform`][triwarp.mesh.Trimesh.transform]
    """
    host = matrix_to_numpy(matrix)
    linear, offset = host[:3, :3], host[:3, 3]
    # The bottom row has to be [0, 0, 0, 1] for the map to be affine at all. A projective matrix
    # is not merely "some other affine map": ``wp.transform_point`` applies it without the
    # perspective divide, so nothing downstream is meaningful -- report the bottom rung.
    if not np.allclose(host[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=rtol, atol=rtol):
        return TransformKind.SINGULAR

    gram = linear @ linear.T
    scale_sq = _scale_sq_from_gram(gram)
    determinant = float(np.linalg.det(linear))
    if _is_singular(linear, rtol):
        return TransformKind.SINGULAR
    if not np.allclose(gram, scale_sq * np.eye(3), rtol=rtol, atol=rtol * scale_sq):
        return TransformKind.AFFINE

    # Scale is tested before the determinant sign, so a *mirroring* similarity (a uniform scale
    # of -2, say) is reported as SIMILARITY and not as REFLECTION: it does not preserve lengths,
    # and REFLECTION promises that it does. Orientation reversal is a separate axis from the
    # metric class -- read it off ``det`` directly, the way
    # [`transform_mesh`][triwarp.transform.transform_mesh] does.
    if abs(scale_sq - 1.0) > rtol:
        return TransformKind.SIMILARITY
    if determinant < 0.0:
        return TransformKind.REFLECTION
    if np.allclose(linear, np.eye(3), rtol=rtol, atol=rtol):
        translates = not np.allclose(offset, 0.0, rtol=rtol, atol=rtol)
        return TransformKind.TRANSLATION if translates else TransformKind.IDENTITY
    return TransformKind.RIGID


def transform_scale(matrix: wp.mat44 | wp.array[wp.mat44]) -> float:
    """
    Uniform scale factor of a similarity transform.

    Parameters
    ----------
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array.

    Returns
    -------
    float
        The positive factor by which lengths scale. Meaningful only when
        [`classify_transform`][triwarp.transform.classify_transform] returns
        `TransformKind.SIMILARITY` or a stronger class; for a general
        `TransformKind.AFFINE` matrix it is the root-mean-square of the linear block's singular
        values and no single factor exists.

    See Also
    --------
    [`classify_transform`][triwarp.transform.classify_transform]
    """
    linear = matrix_to_numpy(matrix)[:3, :3]
    return float(math.sqrt(_scale_sq_from_gram(linear @ linear.T)))


def as_mat44(matrix: wp.mat44 | wp.array[wp.mat44]) -> wp.mat44:
    """
    Read a transform parameter as a scalar ``wp.mat44``, whichever form it arrives in.

    Every entry point here accepts a ``(1,)`` device array. For
    [`transform_points`][triwarp.transform.transform_points] and
    [`transform_mesh`][triwarp.transform.transform_mesh] the array's pointer reaches the kernel
    directly, so a fitted transform never leaves the device; every other entry point -- the
    orientation flip, the classification, the normal map, `transform_vectors` -- needs the value on
    the host regardless of which form it arrives in, and this is the one place it crosses.

    Parameters
    ----------
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array.

    Returns
    -------
    wp.mat44
        The matrix as a host value, returned unchanged when it already is one.

    Raises
    ------
    ValueError
        If ``matrix`` is a device array without exactly one entry.

    Notes
    -----
    A device array costs one host readback. Every caller here is making a host branch over a whole
    launch, so the matrix has to cross either way. Calling this more than once on the same
    unchanged ``matrix`` within one operation (e.g. once for classification, once for the normal
    map) pays the readback again each time -- resolve it to a host value once and pass that value
    down instead, the way [`Trimesh.transform`][triwarp.mesh.Trimesh.transform] does.

    See Also
    --------
    [`classify_transform`][triwarp.transform.classify_transform]
    """
    if not isinstance(matrix, wp.array):
        return matrix
    if matrix.size != 1:
        raise ValueError(
            f"as_mat44 requires a length-1 wp.array[wp.mat44], got shape {tuple(matrix.shape)}"
        )
    return cast(wp.mat44, matrix.list()[0])


def matrix_to_numpy(matrix: wp.mat44 | wp.array[wp.mat44]) -> np.ndarray:
    """
    Read a transform parameter back as a host ``(4, 4)`` ``float64`` array.

    For host matrix arithmetic that has no ``wp.mat44`` equivalent -- composing with a NumPy
    matrix built elsewhere, or reading a row/column out directly. Every decision this module makes
    on a transform ([`classify_transform`][triwarp.transform.classify_transform],
    [`reverses_orientation`][triwarp.transform.reverses_orientation],
    [`normal_matrix`][triwarp.transform.normal_matrix]) goes through this.

    Parameters
    ----------
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` ``float64`` copy.

    Raises
    ------
    ValueError
        If ``matrix`` is a device array without exactly one entry.

    See Also
    --------
    [`as_mat44`][triwarp.transform.as_mat44]
    """
    return np.array(as_mat44(matrix), dtype=np.float64).reshape(4, 4)


def reverses_orientation(matrix: wp.mat44 | wp.array[wp.mat44]) -> bool:
    """
    Whether a transform mirrors, so that face winding must be reversed to keep normals outward.

    Parameters
    ----------
    matrix
        ``4x4`` transform, scalar or ``(1,)`` device array.

    Returns
    -------
    bool
        ``True`` when the linear block has a negative determinant.

    Notes
    -----
    Deliberately separate from [`classify_transform`][triwarp.transform.classify_transform]:
    orientation is an axis of its own, and only `TransformKind.REFLECTION` implies it. A mirroring
    *scale* classifies as `TransformKind.SIMILARITY` and still reverses winding, so a caller that
    switches on the kind alone gets it wrong.

    See Also
    --------
    [`transform_mesh`][triwarp.transform.transform_mesh]
    [`classify_transform`][triwarp.transform.classify_transform]
    """
    return float(np.linalg.det(matrix_to_numpy(matrix)[:3, :3])) < 0.0


# --- private helpers -------------------------------------------------------


def _compose_about(linear: np.ndarray, center: wp.vec3 | Sequence[float] | None) -> wp.mat44:
    """Homogeneous matrix applying ``linear`` about ``center`` (the origin when ``None``)."""
    if center is None:
        return _compose(linear, np.zeros(3))
    fixed = _vec3_host(center, "center")
    return _compose(linear, fixed - linear @ fixed)


def _compose(linear: np.ndarray, offset: np.ndarray) -> wp.mat44:
    """Pack a ``3x3`` block and a length-3 translation into a ``wp.mat44``."""
    host = np.eye(4)
    host[:3, :3] = linear
    host[:3, 3] = offset
    return wp.mat44(*host.flatten())


def _vec3_host(value: wp.vec3 | Sequence[float], name: str) -> np.ndarray:
    """Read a length-3 host vector out of a ``wp.vec3`` or any sequence."""
    array = np.asanyarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{name} must be length-3, got {array.size}")
    return array


def _scale_sq_from_gram(gram: np.ndarray) -> float:
    """
    Mean squared singular value of a linear block, given its Gram matrix ``linear @ linear.T``.

    The squared uniform scale for an isotropic block. Shared by
    [`classify_transform`][triwarp.transform.classify_transform] (which already has ``gram`` in
    hand) and [`transform_scale`][triwarp.transform.transform_scale] (which builds it fresh) so the
    two can't silently define "scale" two different ways.
    """
    return float(np.trace(gram)) / 3.0


def _is_singular(linear: np.ndarray, rtol: float) -> bool:
    """
    Whether a ``3x3`` linear block is rank-deficient to within ``rtol``, regardless of anisotropy.

    Compares the smallest singular value against the largest -- the reciprocal condition number --
    rather than the determinant against a trace-derived scale. A non-uniform (anisotropic) scale
    can have an arbitrarily small determinant relative to its trace while remaining perfectly
    invertible, so a determinant-vs-scale test flags it as singular once its axes differ enough
    (``diag(1000, 1, 1)`` at the module's default tolerance); the singular-value ratio stays
    correct at any anisotropy, and degenerates to the old test's intent for an isotropic block.
    Shared by [`classify_transform`][triwarp.transform.classify_transform] and
    [`normal_matrix`][triwarp.transform.normal_matrix], which tested this with an exact ``det == 0``
    before and so missed every near-singular (rather than exactly singular) block.
    """
    singular_values = np.linalg.svd(linear, compute_uv=False)
    sigma_max = float(singular_values[0])
    if sigma_max <= 0.0:
        return True
    return float(singular_values[-1]) <= rtol * sigma_max


def _alloc_or_out(
    out: twt.ArrayNd | None, n: int, dtype: type, device: wp.DeviceLike
) -> twt.ArrayNd:
    """
    Allocate a fresh ``(n,)`` buffer of ``dtype`` when ``out`` is ``None``, else reuse ``out``.

    Every buffer-op in this module (`transform_points`, `transform_vectors`, `transform_normals`,
    `transform_mesh`) repeats this exact "allocate-or-reuse, then let the caller check for the
    empty case" idiom; sharing it here keeps the four from silently drifting on the allocation
    itself (e.g. a `device=` mistake) while each keeps its own empty-input and length checks, which
    differ enough between them (a `ValueError` on some, none on others) that folding those in too
    would obscure rather than simplify.
    """
    return _launch.empty(n, dtype=dtype, device=device) if out is None else out
