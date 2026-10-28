import warp as wp

from ordito.constants import PI, TOLERANCE_ZERO_CONSTANT, TWO_PI
from ordito.kernels.halfedge import halfedge_destination
from ordito.kernels.predicates import unit_tangent, world_to_tangent
from ordito.kernels.triangles import (
    face_corner_angles,
    face_normals_and_area,
    halfedge_corner_angle,
)
from ordito.kernels.vertices import normalized_accumulated_row

wp.set_module_options({"enable_backward": False})


@wp.func
def corner_angle(face_angles: wp.array2d[wp.float32], h: wp.int32) -> wp.float32:
    # The corner at the *origin* of halfedge ``3*f + k`` is corner ``k`` of face ``f``.
    return face_angles[h // wp.int32(3), h % wp.int32(3)]


@wp.func
def ring_tangent_scale(total: wp.float32, boundary: wp.bool) -> wp.float32:
    # Rescaling by 2*pi/Theta_v (pi/Theta_v at a boundary vertex, whose fan spans a half-disk) is
    # what makes the flattening consistent on a cone point; zero for a fan with no angle at all.
    # Initialized before the branch: a variable assigned only inside an ``if`` is readable
    # afterwards in Warp but uninitialized when the branch was not taken.
    scale = wp.float32(0.0)
    if total > TOLERANCE_ZERO_CONSTANT:
        if boundary:
            scale = PI / total
        else:
            scale = TWO_PI / total
    return scale


@wp.func
def ring_reference_slot(
    ring_halfedges: wp.array[wp.int32], begin: wp.int32, end: wp.int32, boundary: wp.bool
) -> wp.int32:
    # The ring slot polar angle 0 is measured from: geometry-central's ``vertex.halfedge()``, so
    # every tangent-space quantity is in its gauge. An open fan starts at its outgoing boundary
    # halfedge, which is the ring's first; a closed fan at the outgoing halfedge of the vertex's
    # highest-index face, which is the ring's largest halfedge since halfedges ascend by face.
    best = begin
    if not boundary:
        for j in range(begin + 1, end):
            if ring_halfedges[j] > ring_halfedges[best]:
                best = j
    return best - begin


@wp.func
def ring_slot(begin: wp.int32, end: wp.int32, start: wp.int32, t: wp.int32) -> wp.int32:
    # The ``t``-th ring entry counted from slot ``start``, wrapping around the (closed) ring.
    return begin + (start + t) % (end - begin)


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
    # the flattened tangent plane (``ring_tangent_scale``), from the reference slot
    # (``ring_reference_slot``).
    # ``halfedge_tangent_angles_from_positions`` is the same walk forming the corner angles itself.
    v = wp.int32(wp.tid())
    begin = ring_offsets[v]
    end = ring_offsets[v + 1]
    if end <= begin:
        return

    total = wp.float32(0.0)
    for j in range(begin, end):
        total += corner_angle(face_angles, ring_halfedges[j])

    boundary = is_boundary[v]
    scale = ring_tangent_scale(total, boundary)
    start = ring_reference_slot(ring_halfedges, begin, end, boundary)
    accumulated = wp.float32(0.0)
    for t in range(end - begin):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        out_angles[h] = scale * accumulated
        accumulated += corner_angle(face_angles, h)


@wp.kernel
def halfedge_tangent_angles_from_positions(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_angles: wp.array[wp.float32],
) -> None:
    # ``halfedge_tangent_angles`` for a caller with no corner-angle table: each halfedge's corner
    # angle is ``triangles.face_corner_angles``' (the ``angles`` kernel's own body, so the same
    # bits), formed once in the first pass and parked in the halfedge's own output slot until the
    # second pass reads it back and overwrites it -- a halfedge lies on exactly one ring, its
    # origin's, so the slot is this thread's alone. Each halfedge forms its face's three angles to
    # keep the one it needs (the degeneracy guard reads all three), three times the table's
    # arithmetic, in place of the table's launch and its ``(n_faces, 3)`` round trip.
    v = wp.int32(wp.tid())
    begin = ring_offsets[v]
    end = ring_offsets[v + 1]
    if end <= begin:
        return

    total = wp.float32(0.0)
    for j in range(begin, end):
        h = ring_halfedges[j]
        a0, a1, a2 = face_corner_angles(vertices, faces, h // wp.int32(3))
        angle = wp.vec3(a0, a1, a2)[h % wp.int32(3)]
        out_angles[h] = angle
        total += angle

    boundary = is_boundary[v]
    scale = ring_tangent_scale(total, boundary)
    start = ring_reference_slot(ring_halfedges, begin, end, boundary)
    accumulated = wp.float32(0.0)
    for t in range(end - begin):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        angle = out_angles[h]
        out_angles[h] = scale * accumulated
        accumulated += angle


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


@wp.func
def fan_anchor(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    boundary: wp.bool,
    vertex: wp.int32,
    scale: wp.float32,
    start: wp.int32,
    frame_x: wp.vec3,
    frame_y: wp.vec3,
) -> wp.vec2:
    # Where normalized polar angle 0 lies in the plane of ``(frame_x, frame_y)``, fitted to the
    # whole fan: each incident edge in the frame, rotated back by its normalized angle (``scale``
    # times the corner angles accumulated from ring slot ``start``), summed. An open fan has one
    # edge more than wedges: the last wedge's far edge, at half a turn. This is geometry-central's
    # vertex tangent basis, and it is defined for every fan, including a saddle's, whose edges fold
    # over one another in projection.
    begin = ring_offsets[vertex]
    end = ring_offsets[vertex + 1]
    origin = vertices[vertex]
    anchor_x = wp.float32(0.0)
    anchor_y = wp.float32(0.0)
    accumulated = wp.float32(0.0)
    for t in range(end - begin):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        edge = world_to_tangent(vertices[halfedge_destination(faces, h)] - origin, frame_x, frame_y)
        theta = scale * accumulated
        cos_theta = wp.cos(theta)
        sin_theta = wp.sin(theta)
        anchor_x += edge[0] * cos_theta + edge[1] * sin_theta
        anchor_y += edge[1] * cos_theta - edge[0] * sin_theta
        accumulated += halfedge_corner_angle(vertices, faces, h)
    if boundary:
        h = ring_halfedges[end - 1]
        f = h // wp.int32(3)
        k = h % wp.int32(3)
        edge = world_to_tangent(vertices[faces[3 * f + (k + 2) % 3]] - origin, frame_x, frame_y)
        anchor_x -= edge[0]
        anchor_y -= edge[1]
    return wp.vec2(anchor_x, anchor_y)


@wp.func
def vertex_tangent_basis(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normal: wp.vec3,
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    boundary: wp.bool,
    v: wp.int32,
) -> tuple[wp.vec3, wp.vec3]:
    # ``basis_x`` is where ``halfedge_tangent_angles`` puts polar angle 0, fitted to the whole fan
    # (``fan_anchor``) rather than read off the reference halfedge, which it equals only where the
    # fan is flat -- so the two functions describe one coordinate system, geometry-central's. The
    # fit is taken in a provisional frame around the reference edge's projection.
    if wp.length(normal) <= TOLERANCE_ZERO_CONSTANT:
        # An unreferenced vertex has no tangent plane at all. Emit a fixed unit frame rather than
        # zero vectors, which would surface downstream as NaN out of a normalization.
        return wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)
    basis_x = any_perpendicular(normal)
    begin = ring_offsets[v]
    end = ring_offsets[v + 1]
    if end > begin:
        start = ring_reference_slot(ring_halfedges, begin, end, boundary)
        h = ring_halfedges[begin + start]
        direction = vertices[halfedge_destination(faces, h)] - vertices[v]
        tangential, length = unit_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)
        if length > TOLERANCE_ZERO_CONSTANT:
            basis_x = tangential
        total = wp.float32(0.0)
        for j in range(begin, end):
            total += halfedge_corner_angle(vertices, faces, ring_halfedges[j])
        scale = ring_tangent_scale(total, boundary)
        if scale > wp.float32(0.0):
            frame_y = wp.cross(normal, basis_x)
            anchor = fan_anchor(
                vertices,
                faces,
                ring_offsets,
                ring_halfedges,
                boundary,
                v,
                scale,
                start,
                basis_x,
                frame_y,
            )
            norm = wp.length(anchor)
            if norm > TOLERANCE_ZERO_CONSTANT * length:
                basis_x = wp.normalize(anchor[0] * basis_x + anchor[1] * frame_y)
    return basis_x, wp.cross(normal, basis_x)


@wp.kernel
def vertex_tangent_frames(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
) -> None:
    # The frame around a caller's normals; ``vertex_tangent_frames_from_sums`` forms the normal
    # from the angle-weighted accumulator first.
    v = wp.int32(wp.tid())
    basis_x, basis_y = vertex_tangent_basis(
        vertices, faces, normals[v], ring_offsets, ring_halfedges, is_boundary[v], v
    )
    out_basis_x[v] = basis_x
    out_basis_y[v] = basis_y


@wp.kernel
def vertex_tangent_frames_from_sums(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normal_sums: wp.array2d[wp.float64],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # ``vertex_tangent_frames`` with the normal's own last step folded in: the ``float64``
    # accumulator ``kernels/vertices.scatter_angle_weighted_normals`` filled, narrowed and
    # normalized here exactly as ``vertices.normalize_accumulated_rows`` does
    # (``normalized_accumulated_row``), so the normal written is ``vertex_normals``' bit for bit.
    v = wp.int32(wp.tid())
    normal = normalized_accumulated_row(normal_sums, v)
    out_normals[v] = normal
    basis_x, basis_y = vertex_tangent_basis(
        vertices, faces, normal, ring_offsets, ring_halfedges, is_boundary[v], v
    )
    out_basis_x[v] = basis_x
    out_basis_y[v] = basis_y


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
