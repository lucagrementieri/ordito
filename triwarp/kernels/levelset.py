import warp as wp


@wp.func
def shell_vertex(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    outside: wp.float32,
    inside: wp.float32,
    v: wp.int32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    # Both layers of a thickened shell at vertex ``v``: the outward-displaced copy in the first
    # ``n_vertices`` slots and the inward-displaced one after it, so the second layer's vertex ``v``
    # is at ``v + n_vertices`` and the face emitters below can shift by a constant.
    n_vertices = vertices.shape[0]
    position = vertices[v]
    normal = normals[v]
    out_vertices[v] = position + outside * normal
    out_vertices[n_vertices + v] = position - inside * normal


@wp.func
def shell_face(
    faces: wp.array[wp.int32], n_vertices: wp.int32, f: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # The two layers' triangles of face ``f``: the outer copy verbatim, the inner copy shifted by
    # ``n_vertices`` and **wound backwards**, because it faces into the shell rather than out of it.
    # Corners 1 and 2 are swapped, which is ``repair.flip_faces_masked``'s reversal without the
    # mask.
    n_faces = faces.shape[0] // 3
    corner0 = faces[3 * f]
    corner1 = faces[3 * f + 1]
    corner2 = faces[3 * f + 2]
    out_faces[3 * f] = corner0
    out_faces[3 * f + 1] = corner1
    out_faces[3 * f + 2] = corner2
    inner = 3 * (n_faces + f)
    out_faces[inner] = n_vertices + corner0
    out_faces[inner + 1] = n_vertices + corner2
    out_faces[inner + 2] = n_vertices + corner1


@wp.func
def shell_band(
    boundary_edges: wp.array2d[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    e: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # The band closing the shell along boundary edge ``e``: two triangles spanning the outer edge
    # ``(a, b)`` and its inner copy. The winding follows the *directed* boundary edge, which
    # ``boundary.oriented_boundary_edges`` returns in the outer layer's own face winding -- so the
    # band inherits that orientation instead of guessing one, and the whole shell comes out
    # consistently wound. Verified on ``hemisphere`` and ``half_torus``: watertight, consistent, and
    # positive volume.
    outer_a = boundary_edges[e, 0]
    outer_b = boundary_edges[e, 1]
    inner_a = n_vertices + outer_a
    inner_b = n_vertices + outer_b
    slot = base + 6 * e
    out_faces[slot] = outer_a
    out_faces[slot + 1] = inner_b
    out_faces[slot + 2] = outer_b
    out_faces[slot + 3] = outer_a
    out_faces[slot + 4] = inner_a
    out_faces[slot + 5] = inner_b


@wp.kernel
def shell_mesh(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    outside: wp.float32,
    inside: wp.float32,
    faces: wp.array[wp.int32],
    boundary_edges: wp.array2d[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The whole thickened shell in one launch over ``n_vertices + n_faces + n_rim`` threads: the
    # three emitters write disjoint slots and read only the input, so each thread takes one vertex,
    # one face or one boundary edge by its range (``creation.revolve_mesh``'s layout).
    t = wp.int32(wp.tid())
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] // 3
    if t < n_vertices:
        shell_vertex(vertices, normals, outside, inside, t, out_vertices)
    elif t < n_vertices + n_faces:
        shell_face(faces, n_vertices, t - n_vertices, out_faces)
    else:
        shell_band(boundary_edges, n_vertices, 6 * n_faces, t - n_vertices - n_faces, out_faces)
