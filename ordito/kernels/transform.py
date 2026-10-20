"""
Affine transform primitives: the three point/vector/normal maps and the device-matrix launcher.

The three `wp.func`s here are `wp.map` targets rather than kernels, so the
public wrappers in [`ordito.transform`][ordito.transform] carry the allocation and the in-place
``out=`` contract. `apply_transform_mat44` stays a kernel because its matrix lives in a *device*
array: `wp.map` broadcasts a uniform, and reading a fitted transform back to the host just to
apply it would cost the readback [`ordito.registration`][ordito.registration] exists to avoid.
"""

import warp as wp

from ordito.kernels.predicates import normalize_or_zero
from ordito.kernels.triangles import corner_triple, write_corner_triple_reversible

wp.set_module_options({"enable_backward": False})


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


# What ``transform_mesh_buffers`` does with the faces: leave them (an in-place call that keeps the
# winding), copy them, or copy them corner-reversed (a mirroring matrix).
FACES_KEEP = wp.int32(0)
FACES_COPY = wp.int32(1)
FACES_REVERSE = wp.int32(2)


@wp.func
def transform_mesh_row(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    matrix: wp.mat44,
    face_mode: wp.int32,
    i: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # Vertex ``i`` placed by ``matrix`` and face ``i`` written under ``face_mode``, each guarded by
    # its own array's length; shared by ``transform_mesh_buffers`` and
    # ``kernels/mesh.transform_mesh_and_directions``, which also maps a mesh's cached directions.
    if i < vertices.shape[0]:
        out_vertices[i] = transform_point_mat44(vertices[i], matrix)
    if face_mode != FACES_KEEP and i < faces.shape[0] // 3:
        a, b, c = corner_triple(faces, i)
        write_corner_triple_reversible(out_faces, i, a, b, c, face_mode == FACES_REVERSE)


@wp.kernel
def transform_mesh_buffers(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    matrix: wp.mat44,
    face_mode: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``transform.transform_mesh`` in one launch over ``max(n_vertices, n_faces)``: thread ``i``
    # places vertex ``i`` (``transform_point_mat44``, the ``wp.map`` of ``transform_points``) and
    # writes face ``i`` under ``face_mode`` (``repair.reverse_face_winding``'s reversal). Each
    # thread reads its own rows before writing them, so it runs in place.
    i = wp.int32(wp.tid())
    transform_mesh_row(vertices, faces, matrix, face_mode, i, out_vertices, out_faces)
