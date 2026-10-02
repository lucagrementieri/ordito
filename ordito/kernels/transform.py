"""
Affine transform primitives: the three point/vector/normal maps and the device-matrix launcher.

The three `wp.func`s here are `wp.map` targets rather than kernels (CLAUDE.md section 3.5), so the
public wrappers in [`ordito.transform`][ordito.transform] carry the allocation and the in-place
``out=`` contract. `apply_transform_mat44` stays a kernel because its matrix lives in a *device*
array: `wp.map` broadcasts a uniform, and reading a fitted transform back to the host just to
apply it would cost the readback [`ordito.registration`][ordito.registration] exists to avoid.
"""

import warp as wp

from ordito.kernels.predicates import normalize_or_zero


@wp.func
def transform_point_mat44(point: wp.vec3, matrix: wp.mat44) -> wp.vec3:
    # wp.transform_point(mat44, vec3) is exactly ``(M * vec4(p, 1)).xyz``.
    return wp.transform_point(matrix, point)


@wp.func
def transform_vector_mat44(vector: wp.vec3, matrix: wp.mat44) -> wp.vec3:
    # ``(M * vec4(v, 0)).xyz``: the linear block alone, so a translation is ignored. Correct for a
    # displacement or a tangent, and *not* for a normal -- see ``transform_normal_mat33``.
    return wp.transform_vector(matrix, vector)


@wp.func
def transform_normal_mat33(normal: wp.vec3, normal_matrix: wp.mat33) -> wp.vec3:
    # A normal is a covector: it maps by the inverse transpose of the linear block, not by the
    # block itself, or a non-uniform scale tilts it off the transformed surface. The caller builds
    # ``normal_matrix`` on the host ([`normal_matrix`][ordito.transform.normal_matrix]) because it
    # is one 3x3 inverse per launch, not one per thread.
    #
    # Length is not preserved even for a rotation once float32 rounding is in play, so renormalize
    # unconditionally. A zero input stays zero: the degenerate-face convention
    # ``face_normals_and_areas`` writes, which every consumer already reads.
    #
    # ``predicates.normalize_or_zero`` takes a ``<=`` tolerance where this guard was ``> 0.0``
    # (strictly zero); passing an explicit ``wp.float32(0.0)`` preserves that exactly rather than
    # silently moving this normal-transform path onto the package's ``TOLERANCE_ZERO_CONSTANT``.
    mapped = normal_matrix * normal
    return normalize_or_zero(mapped, wp.float32(0.0))


@wp.kernel
def apply_transform_mat44(
    points: wp.array[wp.vec3], matrix: wp.array[wp.mat44], out_points: wp.array[wp.vec3]
) -> None:
    i = wp.int32(wp.tid())
    out_points[i] = transform_point_mat44(points[i], matrix[0])
