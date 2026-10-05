import warp as wp

from ordito.constants import PI, TOLERANCE_ZERO_CONSTANT, TWO_PI
from ordito.kernels.halfedge import halfedge_destination
from ordito.kernels.predicates import unit_tangent
from ordito.kernels.triangles import face_normals_and_area

wp.set_module_options({"enable_backward": False})


@wp.func
def corner_angle(face_angles: wp.array2d[wp.float32], h: wp.int32) -> wp.float32:
    # The corner at the *origin* of halfedge ``3*f + k`` is corner ``k`` of face ``f``.
    return face_angles[h // wp.int32(3), h % wp.int32(3)]


@wp.kernel
def halfedge_tangent_angles(
    face_angles: wp.array2d[wp.float32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_angles: wp.array[wp.float32],
) -> None:
    # One thread per vertex, two passes over its counter-clockwise ring: the first sums the corner
    # angles into the total angle Theta_v, the second lays the halfedges out as polar coordinates in
    # the flattened tangent plane. Rescaling by 2*pi/Theta_v (pi/Theta_v at a boundary vertex, whose
    # fan spans a half-disk) is what makes the flattening consistent on a cone point.
    v = wp.int32(wp.tid())
    begin = ring_offsets[v]
    end = ring_offsets[v + 1]
    if end <= begin:
        return

    total = wp.float32(0.0)
    for j in range(begin, end):
        total += corner_angle(face_angles, ring_halfedges[j])

    # Initialized before the branch: a variable assigned only inside an ``if`` is readable
    # afterwards in Warp but uninitialized when the branch was not taken.
    scale = wp.float32(0.0)
    if total > TOLERANCE_ZERO_CONSTANT:
        if is_boundary[v]:
            scale = PI / total
        else:
            scale = TWO_PI / total

    accumulated = wp.float32(0.0)
    for j in range(begin, end):
        h = ring_halfedges[j]
        out_angles[h] = scale * accumulated
        accumulated += corner_angle(face_angles, h)


@wp.func
def any_perpendicular(normal: wp.vec3) -> wp.vec3:
    # Tangent direction for a vertex with no ring to take one from. Crosses the normal with x or
    # y -- whichever of those two the normal is *less* aligned with -- so the cross product is never
    # degenerate: at most one of x and y can be nearly parallel to a unit vector.
    #
    # This never returns z, and that is deliberate rather than an oversight in the comparison. The
    # contract is "some unit vector perpendicular to ``normal``", not "the most numerically
    # comfortable one", and a three-way argmin would satisfy the contract while returning a
    # *different* tangent for every z-dominant normal -- rotating the frames that
    # ``vertex_tangent_frames`` builds and the hemisphere bundles ``visibility`` builds on them.
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > wp.abs(normal[1]):
        axis = wp.vec3(0.0, 1.0, 0.0)
    return wp.normalize(wp.cross(normal, axis))


@wp.kernel
def vertex_tangent_frames(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
) -> None:
    # The reference direction is the vertex's first ring halfedge projected into the tangent plane,
    # which is exactly the halfedge that ``halfedge_tangent_angles`` assigns polar angle 0 -- so the
    # two functions describe the same coordinate system.
    v = wp.int32(wp.tid())
    normal = normals[v]
    if wp.length(normal) <= TOLERANCE_ZERO_CONSTANT:
        # An unreferenced vertex has no tangent plane at all. Emit a fixed unit frame rather than
        # zero vectors, which would surface downstream as NaN out of a normalization.
        out_basis_x[v] = wp.vec3(1.0, 0.0, 0.0)
        out_basis_y[v] = wp.vec3(0.0, 1.0, 0.0)
        return
    basis_x = any_perpendicular(normal)
    begin = ring_offsets[v]
    if ring_offsets[v + 1] > begin:
        h = ring_halfedges[begin]
        direction = vertices[halfedge_destination(faces, h)] - vertices[v]
        tangential, length = unit_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)
        if length > TOLERANCE_ZERO_CONSTANT:
            basis_x = tangential
    out_basis_x[v] = basis_x
    out_basis_y[v] = wp.cross(normal, basis_x)


@wp.func
def face_tangent_basis(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32, normal: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """
    Return the in-plane frame ``(basis_x, basis_y)`` of face ``f`` given its unit ``normal``.

    The face's own plane needs no projection: the first edge already lies in it, so ``basis_x`` is
    just that edge normalized and ``basis_y`` closes the right-handed frame. A degenerate face has
    no first edge to speak of; ``normalize`` returns zero there (Warp's ``kEps`` is 0) and the
    cross product follows, so the frame degrades to zeros rather than to NaN.
    """
    edge = vertices[faces[f * 3 + 1]] - vertices[faces[f * 3 + 0]]
    basis_x = wp.normalize(edge)
    return basis_x, wp.cross(normal, basis_x)


@wp.kernel
def face_tangent_frames(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
) -> None:
    f = wp.int32(wp.tid())
    basis_x, basis_y = face_tangent_basis(vertices, faces, f, normals[f])
    out_basis_x[f] = basis_x
    out_basis_y[f] = basis_y


@wp.kernel
def face_tangent_frames_and_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # Same frame as ``face_tangent_frames``, for the caller that has no normals to hand: computing
    # the normal here costs one cross product and one normalize, against a launch, an
    # ``(n_faces,)`` round trip through global memory, and the areas buffer that producer writes
    # and this consumer never reads.
    f = wp.int32(wp.tid())
    normal, _area = face_normals_and_area(vertices, faces, f)
    basis_x, basis_y = face_tangent_basis(vertices, faces, f, normal)
    out_basis_x[f] = basis_x
    out_basis_y[f] = basis_y
    out_normals[f] = normal


@wp.func
def wrap_angle(angle: wp.float32) -> wp.float32:
    # Fold into (-pi, pi] without a modulo: the round trip through the unit circle is branch-free
    # and immune to the sign convention of ``%`` inside kernels.
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.kernel
def halfedge_transport_angles(
    twins: wp.array[wp.int32], tangent_angles: wp.array[wp.float32], out_rho: wp.array[wp.float32]
) -> None:
    h = wp.int32(wp.tid())
    twin = twins[h]
    if twin >= wp.int32(0):
        opposite = tangent_angles[twin]
    else:
        # A boundary edge exists as a halfedge in one direction only. At the destination vertex it
        # is the fan-closing edge, whose polar angle is the rescaled total angle: pi.
        opposite = PI
    out_rho[h] = wrap_angle(opposite + PI - tangent_angles[h])
