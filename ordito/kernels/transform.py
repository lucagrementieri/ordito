"""
Affine transform primitives: the three point/vector/normal maps and the device-matrix launcher.

The three `wp.func`s here are `wp.map` targets rather than kernels, so the
public wrappers in [`ordito.transform`][ordito.transform] carry the allocation and the in-place
``out=`` contract. `apply_transform_mat44` stays a kernel because its matrix lives in a *device*
array: `wp.map` broadcasts a uniform, and reading a fitted transform back to the host just to
apply it would cost the readback [`ordito.registration`][ordito.registration] exists to avoid.
"""

import warp as wp

from ordito.kernels.array import declare_lazy_kernels, declare_map_signatures, map_probe
from ordito.kernels.predicates import normalize_or_zero, orient3d_sign_f32
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

# From this many faces the CPU runs ``transform.transform_mesh`` under a host matrix that keeps the
# winding as two passes, the point map and then a ``wp.copy`` of the faces (none in place), instead
# of one ``transform_mesh_buffers`` launch. The CPU runs a launch's threads in turn, and the fused
# launch spends a thread on every face copying three indices where ``wp.copy`` is a memcpy, plus
# idle threads past the last vertex. CPU calls, a rotation, new/old: 0.86-0.90x at 16 k faces,
# 0.70-0.76x on ``bunny``, 0.56-0.63x on ``dragon``, par on ``sphere_small`` (5 k) (2026-11-12).
# A mirror keeps the fused launch: the map plus ``repair.reverse_face_winding`` read 0.77x on
# ``dragon`` when the output pages were resident and 0.93x when every call faulted them in, but
# 1.6x in a heap state where both forms took ~2 000 minor faults a call and only the two passes
# paid for them, and 1.08-1.12x at 16 k faces. CUDA keeps the one launch at every size.
# ``Trimesh.transform`` takes the same bound on its longest buffer before mapping carried
# directions a launch a buffer on the CPU (``kernels/mesh.transform_mesh_and_directions``).
CPU_TWO_PASS_FROM_FACES = 2**13


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


@wp.func
def mirrors_mat44(matrix: wp.mat44) -> wp.bool:
    # ``transform.reverses_orientation`` on the device: the linear block's determinant is negative,
    # as the exact sign of ``orient3d(0, r0, r1, r2) = det(r0, r1, r2)`` (``float32`` filter first,
    # so nearly every thread decides in a few ``float32`` operations). The host's ``np.linalg.det``
    # is rounded, so the two can part only for a block singular to within rounding.
    r0 = wp.vec3(matrix[0, 0], matrix[0, 1], matrix[0, 2])
    r1 = wp.vec3(matrix[1, 0], matrix[1, 1], matrix[1, 2])
    r2 = wp.vec3(matrix[2, 0], matrix[2, 1], matrix[2, 2])
    return orient3d_sign_f32(wp.vec3(0.0, 0.0, 0.0), r0, r1, r2) < 0


@wp.kernel
def transform_mesh_buffers_device(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    matrix: wp.array[wp.mat44],
    face_mode: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``transform_mesh_buffers`` for a matrix in a ``(1,)`` device array (a fitted transform),
    # which differs only in deciding the winding here: ``face_mode`` (``FACES_KEEP`` or
    # ``FACES_COPY``) becomes ``FACES_REVERSE`` when the matrix mirrors, so the call reads nothing
    # back. CUDA only (0.22x on sphere_small and dragon, 0.96x on lucy). On the CPU the
    # per-thread sign test cost 1.1-4.6x the old two passes (sphere_small to dragon) and reading
    # the matrix there instead still lost 1.26-1.54x on dragon: one thread writing a vertex and a
    # face runs slower on the CPU than the map and the face pass apart
    # (``CPU_TWO_PASS_FROM_FACES``), so the CPU keeps them.
    i = wp.int32(wp.tid())
    m = matrix[0]
    mode = wp.where(mirrors_mat44(m), FACES_REVERSE, face_mode)
    transform_mesh_row(vertices, faces, m, mode, i, out_vertices, out_faces)


@declare_lazy_kernels
def _declare_lazy_map_kernels() -> None:
    """Declare the ``wp.map`` signatures this module's ops are reached at, each its own module."""
    dense = map_probe
    declare_map_signatures(
        [
            (transform_normal_mat33, (dense(wp.vec3), wp.mat33()), wp.vec3),
            (transform_point_mat44, (dense(wp.vec3), wp.mat44()), wp.vec3),
            (transform_vector_mat44, (dense(wp.vec3), wp.mat44()), wp.vec3),
        ]
    )
