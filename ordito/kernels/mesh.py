import warp as wp

from ordito.kernels.transform import transform_mesh_row, transform_normal_mat33

wp.set_module_options({"enable_backward": False})


@wp.func
def transform_normal_at(
    normals: wp.array[wp.vec3], normal_matrix: wp.mat33, i: wp.int32, out_normals: wp.array[wp.vec3]
) -> None:
    # ``transform.transform_normals``' per-element map, guarded by this array's own length so one
    # launch over the longest array can serve several of different lengths.
    if i < normals.shape[0]:
        out_normals[i] = transform_normal_mat33(normals[i], normal_matrix)


@wp.kernel
def transform_mesh_and_directions(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    matrix: wp.mat44,
    face_mode: wp.int32,
    normal_matrix: wp.mat33,
    face_normals: wp.array[wp.vec3],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
    out_face_normals: wp.array[wp.vec3],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
    out_vertex_normals: wp.array[wp.vec3],
) -> None:
    # ``Trimesh.transform`` in one launch over the longest buffer: the positions and faces
    # (``transform.transform_mesh``'s row) and up to four cached directions
    # (``transform.transform_normals``' map), so a transformed mesh carrying its normals costs one
    # launch.
    i = wp.int32(wp.tid())
    transform_mesh_row(vertices, faces, matrix, face_mode, i, out_vertices, out_faces)
    transform_normal_at(face_normals, normal_matrix, i, out_face_normals)
    transform_normal_at(basis_x, normal_matrix, i, out_basis_x)
    transform_normal_at(basis_y, normal_matrix, i, out_basis_y)
    transform_normal_at(vertex_normals, normal_matrix, i, out_vertex_normals)


@wp.kernel
def negate_cached_normals(
    face_normals: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    out_face_normals: wp.array[wp.vec3],
    out_vertex_normals: wp.array[wp.vec3],
) -> None:
    # ``Trimesh.invert``'s two negations (``wp.neg``, exact) in one launch over the longer array;
    # an absent buffer is zero-length and skipped by its guard.
    i = wp.int32(wp.tid())
    if i < face_normals.shape[0]:
        out_face_normals[i] = -face_normals[i]
    if i < vertex_normals.shape[0]:
        out_vertex_normals[i] = -vertex_normals[i]
