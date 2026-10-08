import warp as wp

from ordito.constants import PI, TOLERANCE_ZERO_CONSTANT, TWO_PI
from ordito.kernels.array import binary_search_index, loop_point, to_vec3, wrap_index
from ordito.kernels.halfedge import halfedge_destination
from ordito.kernels.predicates import project_out_normal, unit_tangent, world_to_tangent
from ordito.kernels.tangent_space import corner_angle
from ordito.kernels.triangles import face_normal, local_corner

wp.set_module_options({"enable_backward": False})

# How far past either end of an edge, as a fraction of the edge, a crossing still counts as on it.
CORNER_SLACK = wp.float32(1e-5)

# Bound on the faces a walk circles while passing through one vertex.
MAX_FAN_FACES = wp.constant(256)


@wp.func
def exit_edge(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: wp.int32,
    normal: wp.vec3,
    point: wp.vec3,
    direction: wp.vec3,
    entry_edge: wp.int32,
    length_epsilon: wp.float32,
) -> tuple[wp.int32, wp.float32]:
    # Nearest edge the ray ``point + t * direction`` leaves this triangle through, solving
    # ``point + t d = A + s (B - A)`` in the face's plane for each edge. Two things stop the walk
    # exiting through an edge it already stands on: the edge it entered through is skipped outright,
    # and a crossing nearer than ``length_epsilon`` is rejected -- a walk starting at a *vertex*
    # stands on two edges at once, and without this it spins around the fan making no progress.
    best_edge = wp.int32(-1)
    best_t = wp.float32(0.0)
    for k in range(3):
        if k == entry_edge:
            continue
        a = vertices[faces[f * 3 + k]]
        b = vertices[faces[f * 3 + (k + 1) % 3]]
        edge = b - a
        # ``direction`` and ``normal`` are unit vectors (the caller always hands in a normalized
        # tangent), so ``denom`` is ``|edge| * sin(angle between direction and edge)`` -- it scales
        # with the mesh's own edge length, not with a fixed absolute unit. Comparing it against
        # ``length_epsilon`` (already ``mean_edge_length``-scaled, same as the crossing-distance
        # test below) makes this a scale-invariant "is the edge parallel to within this angle"
        # test; a fixed absolute constant here rejected a genuinely non-parallel edge on any mesh
        # small enough that ``|edge|`` itself approached that constant.
        denom = wp.dot(normal, wp.cross(direction, edge))
        if wp.abs(denom) <= length_epsilon:
            continue
        t = -wp.dot(normal, wp.cross(point - a, edge)) / denom
        s = wp.dot(normal, wp.cross(point - a, direction)) / -denom
        # A ray through a corner -- one aimed along an edge reaches the edge's far vertex -- meets
        # the opposite edge exactly at an end, where rounding puts ``s`` a hair outside [0, 1]; an
        # exact test then finds no exit and the walk stopped dead. Within ``CORNER_SLACK`` of an
        # end the crossing is accepted, so the ray passes through the vertex.
        if t <= length_epsilon or s < -CORNER_SLACK or s > wp.float32(1.0) + CORNER_SLACK:
            continue
        if best_edge == wp.int32(-1) or t < best_t:
            best_edge = k
            best_t = t
    return best_edge, best_t


@wp.func
def unfold_direction(
    direction: wp.vec3, axis: wp.vec3, normal_from: wp.vec3, normal_to: wp.vec3
) -> wp.vec3:
    # Rotate the direction about the shared edge by the dihedral angle: unfold the two triangles
    # into a common plane and keep walking straight. That is what makes the path a *straightest*
    # geodesic rather than merely a shortest one.
    unit_axis = wp.normalize(axis)
    angle = wp.atan2(
        wp.dot(wp.cross(normal_from, normal_to), unit_axis), wp.dot(normal_from, normal_to)
    )
    rotated = wp.quat_rotate(wp.quat_from_axis_angle(unit_axis, angle), direction)
    # Re-project: the rotation is exact in theory but drifts, and a direction with a component along
    # the new normal would walk off the surface.
    tangential, length = unit_tangent(rotated, normal_to, TOLERANCE_ZERO_CONSTANT)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return rotated
    return tangential


@wp.func
def emit_walk_point(
    out_points: wp.array[wp.vec3], write_begin: wp.int32, count: wp.int32, point: wp.vec3
) -> wp.int32:
    """
    Conditionally write ``point`` at the walk's next output slot; return the incremented count.

    ``trace_walk`` / ``descent_walk`` are each a two-pass walk sharing one implementation: a
    counting pass (``write_begin < 0``, nothing written) sizes the polyline, and a writing pass
    (``write_begin >= 0``) fills it at ``out_points[write_begin : write_begin + count]``. Every step
    of both walks conditionally writes one point and advances ``count`` by exactly one, so this is
    the whole idiom factored once; a plain return (not ``wp.ref``) is enough since nothing else is
    mutated between the write and the increment.
    """
    if write_begin >= wp.int32(0):
        out_points[write_begin + count] = point
    return count + wp.int32(1)


@wp.func
def leave_through_wedge(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex: wp.int32,
    chosen: wp.int32,
    offset_in_wedge: wp.float32,
) -> tuple[wp.int32, wp.vec3]:
    # The face of halfedge ``chosen`` (which leaves ``vertex``) and the 3D direction that turns
    # ``offset_in_wedge`` radians from that halfedge's edge toward the face's other edge at the
    # vertex, in the face's plane.
    f = chosen // wp.int32(3)
    normal = face_normal(vertices, faces, f)
    edge = vertices[halfedge_destination(faces, chosen)] - vertices[vertex]
    tangential, length = unit_tangent(edge, normal, TOLERANCE_ZERO_CONSTANT)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)
    rotation = wp.quat_from_axis_angle(normal, offset_in_wedge)
    return f, wp.quat_rotate(rotation, tangential)


@wp.func
def corner_at(faces: wp.array[wp.int32], f: wp.int32, vertex: wp.int32) -> wp.int32:
    # Which corner (0, 1, 2) of face ``f`` is ``vertex``.
    corner = wp.int32(0)
    if faces[f * 3 + 1] == vertex:
        corner = wp.int32(1)
    if faces[f * 3 + 2] == vertex:
        corner = wp.int32(2)
    return corner


@wp.func
def corner_angle_at(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32, c: wp.int32
):
    # The interior angle of face ``f`` at its corner ``c``, from the positions.
    apex = vertices[faces[f * 3 + c]]
    near = vertices[faces[f * 3 + (c + 1) % 3]] - apex
    far = vertices[faces[f * 3 + (c + 2) % 3]] - apex
    return wp.atan2(wp.length(wp.cross(near, far)), wp.dot(near, far))


@wp.func
def next_face_around(
    faces: wp.array[wp.int32], twins: wp.array[wp.int32], f: wp.int32, c: wp.int32
) -> wp.int32:
    # The face after ``f`` in the fan of its corner ``c``, turning from the corner's near edge
    # toward its far one: across the edge from the far vertex back to the corner, ``-1`` at the
    # boundary.
    twin = twins[f * 3 + (c + 2) % 3]
    if twin == wp.int32(-1):
        return wp.int32(-1)
    return twin // wp.int32(3)


@wp.func
def continue_through_vertex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    f: wp.int32,
    vertex: wp.int32,
    direction: wp.vec3,
) -> tuple[wp.int32, wp.vec3]:
    # A straightest geodesic that runs into a vertex leaves it with half the vertex's total angle on
    # either side (Polthier and Schmies): the arriving direction, reversed, is placed in the fan of
    # face ``f`` (the face the walk is in), and the walk leaves half a turn of the fan further on.
    # The fan is walked through the twins, so no ring table is needed. ``-1`` when the fan reaches
    # the boundary, where the walk stops as it does at the rim.
    c = corner_at(faces, f, vertex)
    apex = vertices[vertex]
    normal = face_normal(vertices, faces, f)
    near = vertices[faces[f * 3 + (c + 1) % 3]] - apex
    back = -direction
    arrival = wp.atan2(wp.dot(normal, wp.cross(near, back)), wp.dot(near, back))
    arrival = wp.clamp(arrival, wp.float32(0.0), corner_angle_at(vertices, faces, f, c))

    total = wp.float32(0.0)
    g = f
    for _step in range(MAX_FAN_FACES):
        total += corner_angle_at(vertices, faces, g, corner_at(faces, g, vertex))
        g = next_face_around(faces, twins, g, corner_at(faces, g, vertex))
        if g == wp.int32(-1):
            return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)
        if g == f:
            break
    if g != f:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)

    target = arrival + wp.float32(0.5) * total
    if target >= total:
        target -= total
    accumulated = wp.float32(0.0)
    g = f
    for _step in range(MAX_FAN_FACES):
        gc = corner_at(faces, g, vertex)
        wedge = corner_angle_at(vertices, faces, g, gc)
        if target <= accumulated + wedge:
            return leave_through_wedge(vertices, faces, vertex, g * 3 + gc, target - accumulated)
        accumulated += wedge
        g = next_face_around(faces, twins, g, gc)
    return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)


@wp.func
def corner_hit(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: wp.int32,
    edge: wp.int32,
    point: wp.vec3,
) -> wp.int32:
    # The vertex a crossing of edge ``edge`` of face ``f`` at ``point`` lands on, or ``-1``: within
    # ``CORNER_SLACK`` of the edge's length of either end.
    a = vertices[faces[f * 3 + edge]]
    b = vertices[faces[f * 3 + (edge + 1) % 3]]
    reach = CORNER_SLACK * wp.length(b - a)
    if wp.length(point - a) <= reach:
        return faces[f * 3 + edge]
    if wp.length(point - b) <= reach:
        return faces[f * 3 + (edge + 1) % 3]
    return wp.int32(-1)


@wp.func
def trace_walk(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    start_face: wp.int32,
    start_point: wp.vec3,
    start_direction: wp.vec3,
    arc_length: wp.float32,
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    write_begin: wp.int32,
    out_points: wp.array[wp.vec3],
) -> wp.int32:
    # Straightest-geodesic walk from one point, for the arc length ``|start_direction|``. Returns
    # the number of polyline points; writes them from ``write_begin`` when that is non-negative, so
    # the counting and writing passes share this one implementation.
    face = start_face
    normal = face_normal(vertices, faces, face)
    point = start_point
    tangential, tangential_length = unit_tangent(start_direction, normal, TOLERANCE_ZERO_CONSTANT)
    remaining = arc_length

    count = wp.int32(0)
    count = emit_walk_point(out_points, write_begin, count, point)
    if remaining <= TOLERANCE_ZERO_CONSTANT or tangential_length <= TOLERANCE_ZERO_CONSTANT:
        return count

    # ``unit_tangent`` already normalized it (the length guard above proved it could).
    direction = tangential
    entry_edge = wp.int32(-1)
    for _step in range(max_steps):
        edge, distance = exit_edge(
            vertices, faces, face, normal, point, direction, entry_edge, length_epsilon
        )
        if edge == wp.int32(-1):
            # No exit found: a degenerate triangle, or a direction grazing a corner. Stop here
            # rather than leave the surface.
            break
        if distance >= remaining:
            point = point + remaining * direction
            count = emit_walk_point(out_points, write_begin, count, point)
            remaining = wp.float32(0.0)
            break

        point = point + distance * direction
        remaining -= distance
        count = emit_walk_point(out_points, write_begin, count, point)

        hit = corner_hit(vertices, faces, face, edge, point)
        if hit != wp.int32(-1):
            # Through a vertex: unfolding across one edge has no meaning there.
            next_face, next_direction = continue_through_vertex(
                vertices, faces, twins, face, hit, direction
            )
            if next_face == wp.int32(-1):
                break  # a boundary vertex
            point = vertices[hit]
            face = next_face
            normal = face_normal(vertices, faces, face)
            direction = next_direction
            entry_edge = wp.int32(-1)
            continue

        twin = twins[face * 3 + edge]
        if twin == wp.int32(-1):
            break  # the path ran into the mesh boundary
        a = vertices[faces[face * 3 + edge]]
        b = vertices[faces[face * 3 + (edge + 1) % 3]]
        next_face = twin // wp.int32(3)
        next_normal = face_normal(vertices, faces, next_face)
        direction = unfold_direction(direction, b - a, normal, next_normal)
        face = next_face
        normal = next_normal
        entry_edge = twin % wp.int32(3)
    return count


@wp.func
def start_direction_at_vertex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: wp.array2d[wp.float32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    vertex: wp.int32,
    direction: wp.vec3,
) -> tuple[wp.int32, wp.vec3]:
    # Which incident face a direction leaves the vertex through, and the 3D direction to use.
    #
    # The vertex's tangent space is its fan with every corner angle rescaled so the fan spans a full
    # turn (half a turn at a boundary vertex): the normalized polar angle of Polthier and Schmies,
    # the flattening of ``tangent.halfedge_tangent_angles``. A 3D direction is read as such an angle
    # through a frame fitted to the fan: each incident edge, projected into the tangent plane, is
    # rotated back by its normalized angle, and the sum of those vectors is where normalized angle 0
    # lies in the plane. The fit uses every edge, so the result depends on neither the ring's first
    # halfedge nor ``basis_x`` (any frame of the plane gives the same answer), and it is defined for
    # every fan, including a saddle's, whose edges fold over one another in projection. It is
    # geometry-central's vertex tangent basis, so ``potpourri3d.GeodesicTracer`` starts the same
    # walk. A direction along an edge leaves along it only where the fan is flat; placing directions
    # between the projected edges that bracket them instead gives every edge exactly, but has no
    # answer for a folded fan and depended on the frame there.
    begin = ring_offsets[vertex]
    end = ring_offsets[vertex + 1]
    if end <= begin:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)
    total = wp.float32(0.0)
    for j in range(begin, end):
        total += corner_angle(face_angles, ring_halfedges[j])
    if total <= TOLERANCE_ZERO_CONSTANT:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)
    full_turn = TWO_PI
    if is_boundary[vertex]:
        full_turn = PI
    scale = full_turn / total

    origin = vertices[vertex]
    anchor_x = wp.float32(0.0)
    anchor_y = wp.float32(0.0)
    accumulated = wp.float32(0.0)
    for j in range(begin, end):
        h = ring_halfedges[j]
        f_j = h // wp.int32(3)
        k = h % wp.int32(3)
        edge = world_to_tangent(
            vertices[faces[3 * f_j + (k + 1) % 3]] - origin, basis_x[vertex], basis_y[vertex]
        )
        theta = scale * accumulated
        cos_theta = wp.cos(theta)
        sin_theta = wp.sin(theta)
        anchor_x += edge[0] * cos_theta + edge[1] * sin_theta
        anchor_y += edge[1] * cos_theta - edge[0] * sin_theta
        accumulated += corner_angle(face_angles, h)
    if is_boundary[vertex]:
        # An open fan has one edge more than wedges: the last wedge's far edge, at half a turn.
        h = ring_halfedges[end - 1]
        f_j = h // wp.int32(3)
        k = h % wp.int32(3)
        edge = world_to_tangent(
            vertices[faces[3 * f_j + (k + 2) % 3]] - origin, basis_x[vertex], basis_y[vertex]
        )
        anchor_x -= edge[0]
        anchor_y -= edge[1]

    tangent = world_to_tangent(direction, basis_x[vertex], basis_y[vertex])
    angle = wp.atan2(tangent[1], tangent[0]) - wp.atan2(anchor_y, anchor_x)
    if angle < wp.float32(0.0):
        angle += TWO_PI
    if angle < wp.float32(0.0):
        angle += TWO_PI
    if angle >= TWO_PI:
        angle -= TWO_PI
    if angle > full_turn:
        # A boundary vertex's fan spans half a turn; a direction outside it points off the surface.
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)

    # Walk the ring until the accumulated normalized angle passes the target.
    accumulated = wp.float32(0.0)
    chosen = ring_halfedges[end - 1]
    offset_in_wedge = wp.float32(0.0)
    for j in range(begin, end):
        h = ring_halfedges[j]
        wedge = scale * corner_angle(face_angles, h)
        if angle <= accumulated + wedge or j == end - 1:
            chosen = h
            offset_in_wedge = (angle - accumulated) / scale
            break
        accumulated += wedge

    # Undo the rescale: rotate the chosen halfedge's direction by the true in-face angle.
    return leave_through_wedge(vertices, faces, vertex, chosen, offset_in_wedge)


@wp.kernel
def trace_from_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_angles: wp.array2d[wp.float32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    start_vertices: wp.array[wp.int32],
    directions: wp.array[wp.vec3],
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    r = wp.int32(wp.tid())
    v = start_vertices[r]
    direction = directions[r]
    f, in_face = start_direction_at_vertex(
        vertices,
        faces,
        face_angles,
        ring_offsets,
        ring_halfedges,
        is_boundary,
        basis_x,
        basis_y,
        v,
        direction,
    )
    write_begin = wp.int32(-1)
    if offsets.shape[0] > 0:
        write_begin = offsets[r]
    if f == wp.int32(-1):
        # An isolated vertex, or a direction pointing out of a boundary vertex's fan: nowhere to go.
        if write_begin >= wp.int32(0):
            out_points[write_begin] = vertices[v]
        out_counts[r] = wp.int32(1)
        return
    # The trace length is measured in the *vertex's* tangent plane, not in the plane of whichever
    # incident face the walk starts in -- the vertex has one tangent space and the fan's faces each
    # tilt differently out of it.
    normal = vertex_normals[v]
    arc_length = wp.length(project_out_normal(direction, normal))
    out_counts[r] = trace_walk(
        vertices,
        faces,
        twins,
        f,
        vertices[v],
        in_face,
        arc_length,
        max_steps,
        length_epsilon,
        write_begin,
        out_points,
    )


@wp.kernel
def trace_from_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    start_faces: wp.array[wp.int32],
    start_bary: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # One ray per thread. ``offsets`` is empty on the counting pass, which is how the two passes
    # share ``trace_walk``.
    r = wp.int32(wp.tid())
    f = start_faces[r]
    bary = start_bary[r]
    point = (
        bary[0] * vertices[faces[f * 3 + 0]]
        + bary[1] * vertices[faces[f * 3 + 1]]
        + bary[2] * vertices[faces[f * 3 + 2]]
    )
    write_begin = wp.int32(-1)
    if offsets.shape[0] > 0:
        write_begin = offsets[r]
    # The trace length is the direction's component in the *face's* plane: a direction leaving the
    # surface traces only what is tangential to it.
    direction = directions[r]
    normal = face_normal(vertices, faces, f)
    arc_length = wp.length(project_out_normal(direction, normal))
    out_counts[r] = trace_walk(
        vertices,
        faces,
        twins,
        f,
        point,
        direction,
        arc_length,
        max_steps,
        length_epsilon,
        write_begin,
        out_points,
    )


@wp.func
def descend_at_vertex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    gradients: wp.array[wp.vec3d],
    v: wp.int32,
) -> wp.int32:
    # Which of ``v``'s incident faces the descent continues into, or -1 if none does.
    #
    # This is the case a pure in-face walk cannot handle and the reason a descent path is a state
    # machine rather than a loop: at a vertex the field has no single gradient, so the walk has to
    # ask each face of the fan whether *its* constant descent direction points inward from ``v``.
    # The test is that direction against both edges of the fan wedge at ``v``; the steepest
    # admissible face wins, which is what makes the choice deterministic rather than fan-order
    # dependent.
    best_face = wp.int32(-1)
    best_slope = wp.float64(0.0)
    for slot in range(face_offsets[v], face_offsets[v + 1]):
        f = vertex_faces[slot]
        corner = local_corner(faces, f, v)
        if corner < 0:
            continue
        gradient = gradients[f]
        slope = wp.length(gradient)
        if slope <= wp.float64(0.0):
            continue
        direction = -to_vec3(gradient) / wp.float32(slope)
        normal_of = face_normal(vertices, faces, f)
        # Is the direction inside the fan wedge at ``v``? The wedge is spanned by the two incident
        # edges, and the test is the orientation-agnostic "same side of each": ``d`` is inside when
        # it turns the same way from the first edge as the second does, and the same way from the
        # second as the first does. A weaker test -- rejecting only a direction negative against
        # *both* edges -- lets through a face whose descent leaves through the vertex itself, and
        # then the walk finds no exit edge and stops after one point, which truncated nearly every
        # path to a single point.
        first = vertices[faces[f * 3 + (corner + 1) % 3]] - vertices[v]
        second = vertices[faces[f * 3 + (corner + 2) % 3]] - vertices[v]
        wedge = wp.dot(wp.cross(first, second), normal_of)
        if wedge == 0.0:
            continue  # a degenerate corner spans no wedge
        if wp.dot(wp.cross(first, direction), normal_of) * wedge < 0.0:
            continue
        if wp.dot(wp.cross(direction, second), normal_of) * wedge < 0.0:
            continue
        if best_face == wp.int32(-1) or slope > best_slope:
            best_face = f
            best_slope = slope
    return best_face


@wp.func
def descend_to_neighbour(
    faces: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    v: wp.int32,
) -> wp.int32:
    # The lowest-valued vertex of ``v``'s 1-ring, or -1 when ``v`` is already the lowest.
    #
    # The fallback for a vertex no face's descent leads out of, which is the discrete form of
    # "descend along an edge": the ring is read off the incident faces rather than from an ordered
    # one-ring, because the order is irrelevant here and the face CSR exists on meshes where a
    # rotational order does not.
    best = wp.int32(-1)
    best_value = values[v]
    for slot in range(face_offsets[v], face_offsets[v + 1]):
        f = vertex_faces[slot]
        for k in range(3):
            other = faces[f * 3 + k]
            if other != v and values[other] < best_value:
                best = other
                best_value = values[other]
    return best


@wp.func
def graph_predecessor(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    graph_distance: wp.array[wp.float32],
    v: wp.int32,
) -> wp.int32:
    # ``v``'s parent in the shortest-edge-path tree of ``graph_distance`` (a converged
    # ``graph.shortest_path_envelope`` from the sources): the 1-ring neighbour minimizing
    # ``d(u) + |p_u - p_v|``, or -1 when none lies strictly lower than ``v`` (a source, or a
    # component no source reaches). Every edge length is positive, so the minimizer's ``d`` is
    # strictly below ``d(v)`` and a chain of these steps ends at a source.
    best = wp.int32(-1)
    best_value = graph_distance[v]
    best_through = wp.float32(3.4e38)
    for slot in range(face_offsets[v], face_offsets[v + 1]):
        f = vertex_faces[slot]
        for k in range(3):
            other = faces[f * 3 + k]
            if other != v and graph_distance[other] < best_value:
                through = graph_distance[other] + wp.length(vertices[other] - vertices[v])
                if through < best_through:
                    best = other
                    best_through = through
    return best


@wp.func
def descent_walk(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    gradients: wp.array[wp.vec3d],
    start_vertex: wp.int32,
    stop_value: wp.float64,
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    graph_distance: wp.array[wp.float32],
    write_begin: wp.int32,
    out_points: wp.array[wp.vec3],
) -> tuple[wp.int32, wp.int32]:
    # Follow the steepest descent of a per-vertex field from ``start_vertex`` until the field drops
    # to ``stop_value``. With a geodesic distance field to a source, that traces the geodesic *back*
    # to the source -- the path a caller reads in either direction.
    #
    # A two-state machine: **at a vertex** (``face < 0``) or **inside a face**. Every branch either
    # writes a point whose field value is strictly lower than the last, or stops -- which is what
    # makes the walk terminate rather than orbit, and it is the reason the vertex state exists at
    # all. The field has no single gradient at a vertex, and the two cases a purely in-face walk
    # cannot express are exactly the ones that arise there: a descent that runs *along* an edge, and
    # one that leaves through a corner.
    count = wp.int32(0)
    vertex = start_vertex
    point = vertices[vertex]
    count = emit_walk_point(out_points, write_begin, count, point)

    # The field value at the last written point, tracked exactly rather than read off ``vertex`` --
    # ``vertex`` goes stale the moment the walk crosses into a second (or third, ...) face without
    # passing through a vertex in between, and the flat-face fallback below needs the value at
    # *this* point, not at whichever vertex the walk last stood on.
    last_value = values[vertex]

    face = wp.int32(-1)
    entry_edge = wp.int32(-1)
    # Where the walk stopped short of ``stop_value``: a vertex (a local minimum of the field) or a
    # face (a flat one, or a descent into the mesh boundary). Both -1 when it arrived, or ran out
    # of ``max_steps`` -- a cap, not a stop.
    short_vertex = wp.int32(-1)
    short_face = wp.int32(-1)
    for _step in range(max_steps):
        if face < wp.int32(0):
            # --- at a vertex -------------------------------------------------------------------
            if values[vertex] <= stop_value:
                break
            chosen = descend_at_vertex(
                vertices, faces, face_offsets, vertex_faces, gradients, vertex
            )
            if chosen >= wp.int32(0):
                face = chosen
                point = vertices[vertex]
                entry_edge = wp.int32(-1)
                continue
            # No face's descent leads out of this vertex: step along an edge instead.
            neighbour = descend_to_neighbour(faces, face_offsets, vertex_faces, values, vertex)
            if neighbour < wp.int32(0):
                short_vertex = vertex  # a local minimum of the field
                break
            vertex = neighbour
            point = vertices[vertex]
            last_value = values[vertex]
            count = emit_walk_point(out_points, write_begin, count, point)
            continue

        # --- inside a face -------------------------------------------------------------------
        gradient = gradients[face]
        slope = wp.length(gradient)
        normal = face_normal(vertices, faces, face)
        edge = wp.int32(-1)
        distance = wp.float32(0.0)
        direction = wp.vec3(0.0, 0.0, 0.0)
        if slope > wp.float64(0.0):
            descent = -to_vec3(gradient) / wp.float32(slope)
            direction, tangential_length = unit_tangent(descent, normal, TOLERANCE_ZERO_CONSTANT)
            if tangential_length > TOLERANCE_ZERO_CONSTANT:
                edge, distance = exit_edge(
                    vertices, faces, face, normal, point, direction, entry_edge, length_epsilon
                )
        if edge < wp.int32(0):
            # A flat face, or a descent grazing a corner: fall back to this face's lowest corner.
            lowest = faces[face * 3]
            for k in range(1, 3):
                if values[faces[face * 3 + k]] < values[lowest]:
                    lowest = faces[face * 3 + k]
            if values[lowest] >= last_value and face >= wp.int32(0):
                short_face = face  # no progress available here
                break
            vertex = lowest
            point = vertices[vertex]
            last_value = values[vertex]
            count = emit_walk_point(out_points, write_begin, count, point)
            face = wp.int32(-1)
            continue

        point = point + distance * direction
        # Exact, not interpolated: ``direction`` is ``-gradient / slope``, so the field's
        # directional derivative along it is ``-slope`` and the step is a straight line inside one
        # face's affine field.
        last_value -= wp.float64(slope) * wp.float64(distance)
        count = emit_walk_point(out_points, write_begin, count, point)

        start = faces[face * 3 + edge]
        end = faces[face * 3 + (edge + 1) % 3]
        if values[start] <= stop_value or values[end] <= stop_value:
            # The stop value sits on a corner of the edge just reached: finish *at* that vertex
            # rather than on the edge, so a distance field's path closes exactly on its source.
            reached = start
            if values[end] < values[start]:
                reached = end
            count = emit_walk_point(out_points, write_begin, count, vertices[reached])
            break

        # The exit landed *on* a corner of that edge, not across it. Hand the walk to the vertex
        # state, which is the state that can express what happens at a vertex -- and, just as
        # importantly, re-reads ``last_value`` from ``values`` instead of carrying the accumulated
        # one across into the next face.
        #
        # Both halves matter, and it is the second that this exists for. Carrying on into the twin
        # face leaves the walk standing on a vertex with no exit edge, so the flat-face fallback
        # below fires and compares that vertex's own value against a ``last_value`` accumulated
        # over the preceding steps. The two are the same number up to float drift, so the walk
        # continued on one device and stopped as a "local minimum" on the other, truncating a
        # fraction of the paths partway along. An icosphere routes descents exactly through
        # vertices often enough for this to be systematic rather than a coincidence.
        landed = wp.int32(-1)
        if wp.length(point - vertices[start]) <= length_epsilon:
            landed = start
        elif wp.length(point - vertices[end]) <= length_epsilon:
            landed = end
        if landed >= wp.int32(0):
            vertex = landed
            point = vertices[vertex]
            last_value = values[vertex]
            face = wp.int32(-1)
            continue

        twin = twins[face * 3 + edge]
        if twin == wp.int32(-1):
            short_face = face  # the descent ran into the mesh boundary
            break
        next_face = twin // wp.int32(3)
        next_gradient = gradients[next_face]
        enters = wp.bool(False)
        if wp.length(next_gradient) > wp.float64(0.0):
            # Does the next face's descent point *into* it? Tested against the edge's true inward
            # normal in that face's plane -- a cheaper test against "the direction of the opposite
            # corner" is wrong on an obtuse triangle, which is what left paths stopping early.
            next_normal = face_normal(vertices, faces, next_face)
            inward = wp.cross(next_normal, vertices[end] - vertices[start])
            opposite = faces[next_face * 3 + (twin % wp.int32(3) + 2) % 3]
            if wp.dot(inward, vertices[opposite] - vertices[start]) < 0.0:
                inward = -inward
            next_descent = -to_vec3(next_gradient)
            enters = wp.dot(next_descent, inward) > 0.0
        if enters:
            face = next_face
            entry_edge = twin % wp.int32(3)
            continue

        # The descent runs along this edge: slide to its lower-valued endpoint.
        vertex = start
        if values[end] < values[start]:
            vertex = end
        point = vertices[vertex]
        last_value = values[vertex]
        count = emit_walk_point(out_points, write_begin, count, point)
        face = wp.int32(-1)

    stopped_short = wp.int32(0)
    if short_vertex >= wp.int32(0) or short_face >= wp.int32(0):
        stopped_short = wp.int32(1)
        if graph_distance.shape[0] > 0:
            # Finish along the shortest-edge-path tree. From a face, first to its corner nearest
            # the source by that distance (a straight segment inside the face).
            if short_vertex < wp.int32(0):
                short_vertex = faces[short_face * 3]
                for k in range(1, 3):
                    corner = faces[short_face * 3 + k]
                    if graph_distance[corner] < graph_distance[short_vertex]:
                        short_vertex = corner
                count = emit_walk_point(out_points, write_begin, count, vertices[short_vertex])
            vertex = short_vertex
            for _step in range(max_steps):
                parent = graph_predecessor(
                    vertices, faces, face_offsets, vertex_faces, graph_distance, vertex
                )
                if parent < wp.int32(0):
                    break
                vertex = parent
                count = emit_walk_point(out_points, write_begin, count, vertices[vertex])
    return count, stopped_short


@wp.kernel
def descent_paths(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    gradients: wp.array[wp.vec3d],
    starts: wp.array[wp.int32],
    stop_value: wp.float64,
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    graph_distance: wp.array[wp.float32],
    short_paths: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # One path per thread, in three passes sharing ``descent_walk`` -- the convention
    # ``trace_from_faces`` uses, plus one. ``offsets`` is empty on the counting passes.
    #
    # * Counting, ``graph_distance`` empty: a non-empty ``short_paths`` (``n_paths + 1`` zeros)
    #   records which paths stopped short of ``stop_value`` and, in its last slot, how many.
    # * Counting again with ``graph_distance`` (the completion): only the flagged paths walk, as
    #   only their lengths change; the rest keep their counts.
    # * Writing, ``offsets`` given: every path walks and writes its points.
    r = wp.int32(wp.tid())
    write_begin = wp.int32(-1)
    if offsets.shape[0] > 0:
        write_begin = offsets[r]
    elif graph_distance.shape[0] > 0 and short_paths.shape[0] > 0:
        if short_paths[r] == wp.int32(0):
            return
    count, stopped_short = descent_walk(
        vertices,
        faces,
        twins,
        face_offsets,
        vertex_faces,
        values,
        gradients,
        starts[r],
        stop_value,
        max_steps,
        length_epsilon,
        graph_distance,
        write_begin,
        out_points,
    )
    out_counts[r] = count
    if offsets.shape[0] == 0 and graph_distance.shape[0] == 0 and short_paths.shape[0] > 0:
        if stopped_short != wp.int32(0):
            short_paths[r] = wp.int32(1)
            wp.atomic_add(short_paths, starts.shape[0], wp.int32(1))


@wp.func
def ring_slot_of(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    v: wp.int32,
    target: wp.int32,
) -> wp.int32:
    # Which slot of ``v``'s one-ring points at ``target``, or -1 when the two are not adjacent.
    for s in range(ring_offsets[v], ring_offsets[v + 1]):
        if halfedge_destination(faces, ring_halfedges[s]) == target:
            return s
    return wp.int32(-1)


@wp.func
def ring_arc_length(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    v: wp.int32,
    slot_from: wp.int32,
    slot_to: wp.int32,
    step: wp.int32,
) -> tuple[wp.float32, wp.int32]:
    # Length of the walk around ``v``'s *link* from one ring slot to another, plus how many link
    # vertices it passes strictly between them. The link of an interior manifold vertex is a closed
    # cycle, so the two directions give the two ways round; ``step`` picks one.
    begin = ring_offsets[v]
    n = ring_offsets[v + 1] - begin
    total = wp.float32(0.0)
    interior = wp.int32(0)
    previous = halfedge_destination(faces, ring_halfedges[slot_from])
    s = slot_from
    for _ in range(n):
        s = begin + wrap_index(s - begin + step, n)
        current = halfedge_destination(faces, ring_halfedges[s])
        total += wp.length(vertices[current] - vertices[previous])
        previous = current
        if s == slot_to:
            break
        interior += 1
    return total, interior


# ``shorten_loop_with_offsets``' device-side loop state, one ``wp.int32`` word each. The sweep
# kernels read the parity and the stop flag from it and the closing ``shorten_loop_advance`` writes
# it, so a group of sweeps replays as one recorded graph with one read of the state per group.
SHORTEN_SWEEPS = wp.constant(0)  # sweeps run so far; its parity is the next sweep's
SHORTEN_DONE = wp.constant(1)  # set once the loop has stopped: every kernel then returns at once
SHORTEN_UNCHANGED = wp.constant(2)  # consecutive sweeps that accepted nothing
SHORTEN_CHANGED = wp.constant(3)  # replacements accepted by the sweep in flight
SHORTEN_OVERFLOW = wp.constant(4)  # the sweep in flight did not fit the buffers and was undone
SHORTEN_LENGTH = wp.constant(5)  # positions in the current loops
SHORTEN_ANY = wp.constant(6)  # whether any sweep has changed the loops
SHORTEN_MAX_SWEEPS = wp.constant(7)
SHORTEN_STATE_SLOTS = 8


@wp.func
def shorten_loop_owner(loop_offsets: wp.array[wp.int32], t: wp.int32) -> wp.int32:
    # The loop holding position ``t``: the last offset at or below it, so an empty loop (equal
    # neighbouring offsets) is never the answer.
    return binary_search_index(loop_offsets, t) - 1


@wp.func
def shorten_loop_past_end(loop_offsets: wp.array[wp.int32], t: wp.int32) -> wp.bool:
    # Whether buffer slot ``t`` lies past the current loops (their total length is the offsets'
    # last entry): every per-slot sweep kernel launches over the whole buffer and skips these.
    return t >= loop_offsets[loop_offsets.shape[0] - 1]


@wp.func
def remap_loop_offset(
    loop_offsets: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    t: wp.int32,
    out_loop_offsets: wp.array[wp.int32],
):
    # Threads up to ``n_loops`` map loop ``t``'s offset through ``positions``, the scan a sweep
    # stage rewrote the slots with, so a stage's per-loop offsets ride its per-slot launch.
    if t < loop_offsets.shape[0]:
        out_loop_offsets[t] = positions[loop_offsets[t]]


@wp.func
def shorten_loop_rewriting(state: wp.array[wp.int32], positions: wp.array[wp.int32]) -> wp.bool:
    # Whether the sweep in flight rewrites the loops: it accepted something and its rewritten
    # length (the counts' scanned total, ``positions``' entry ``capacity``) fits the buffers.
    capacity = positions.shape[0] - 1
    return (
        state[SHORTEN_DONE] == 0 and state[SHORTEN_CHANGED] != 0 and positions[capacity] <= capacity
    )


@wp.kernel
def shorten_loop_counts(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    loop_vertices: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    tolerance: wp.float32,
    state: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_arc_slot: wp.array[wp.int32],
    out_arc_step: wp.array[wp.int32],
) -> None:
    # One thread per buffer slot; slots past the current loops count nothing. A position is
    # *active* when its parity matches this sweep's, so no two neighbours are ever rewritten at
    # once and each replacement sees an unmodified triple. Each accepted replacement is counted in
    # ``state[SHORTEN_CHANGED]``.
    t = wp.int32(wp.tid())
    if state[SHORTEN_DONE] != 0:
        return
    out_counts[t] = wp.int32(0)
    if shorten_loop_past_end(loop_offsets, t):
        return
    loop = shorten_loop_owner(loop_offsets, t)
    begin = loop_offsets[loop]
    n = loop_offsets[loop + 1] - begin
    p = t - begin
    out_counts[t] = wp.int32(1)
    out_arc_slot[t] = wp.int32(-1)
    out_arc_step[t] = wp.int32(0)
    if n < 3 or p % 2 != state[SHORTEN_SWEEPS] % 2:
        return
    # An odd-length cycle makes positions 0 and n - 1 neighbours *and* both even, so the even sweep
    # gives up the last one rather than letting two adjacent threads rewrite one triple.
    if n % 2 == 1 and p == n - 1:
        return
    b = loop_vertices[t]
    if is_boundary[b]:
        return  # the link of a boundary vertex is a path, not a cycle: there is no way round

    a = loop_vertices[begin + wrap_index(p - 1, n)]
    c = loop_vertices[begin + loop_point(p + 1, n)]
    if a == c:
        # The loop doubles back through b. Dropping b leaves the duplicate that the compaction pass
        # removes, and both together contract the spur.
        out_counts[t] = wp.int32(0)
        wp.atomic_add(state, SHORTEN_CHANGED, 1)
        return
    slot_a = ring_slot_of(faces, ring_offsets, ring_halfedges, b, a)
    slot_c = ring_slot_of(faces, ring_offsets, ring_halfedges, b, c)
    if slot_a < 0 or slot_c < 0:
        return

    through = wp.length(vertices[b] - vertices[a]) + wp.length(vertices[c] - vertices[b])
    forward, forward_interior = ring_arc_length(
        vertices, faces, ring_offsets, ring_halfedges, b, slot_a, slot_c, 1
    )
    backward, backward_interior = ring_arc_length(
        vertices, faces, ring_offsets, ring_halfedges, b, slot_a, slot_c, -1
    )
    best = forward
    step = wp.int32(1)
    interior = forward_interior
    if backward < best:
        best = backward
        step = wp.int32(-1)
        interior = backward_interior
    if best < through - tolerance:
        out_counts[t] = interior
        out_arc_slot[t] = slot_a
        out_arc_step[t] = step
        wp.atomic_add(state, SHORTEN_CHANGED, 1)


@wp.kernel
def shorten_loop_write(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    loop_vertices: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    arc_slot: wp.array[wp.int32],
    arc_step: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_loop_vertices: wp.array[wp.int32],
    out_loop_offsets: wp.array[wp.int32],
) -> None:
    # ``positions`` is ``shorten_loop_counts``' counts scanned in place behind a leading zero, so a
    # position's count is the step between its offset and the next. Threads up to ``n_loops`` also
    # map each loop's offset through that remap.
    t = wp.int32(wp.tid())
    if not shorten_loop_rewriting(state, positions):
        return
    remap_loop_offset(loop_offsets, positions, t, out_loop_offsets)
    if shorten_loop_past_end(loop_offsets, t):
        return
    count = positions[t + 1] - positions[t]
    if count == 0:
        return  # b dropped: either the loop doubled back through it, or a -- c is itself an edge
    if arc_slot[t] < 0:
        out_loop_vertices[positions[t]] = loop_vertices[t]
        return
    begin = ring_offsets[loop_vertices[t]]
    n = ring_offsets[loop_vertices[t] + 1] - begin
    s = arc_slot[t]
    for k in range(count):
        s = begin + wrap_index(s - begin + arc_step[t], n)
        out_loop_vertices[positions[t] + k] = halfedge_destination(faces, ring_halfedges[s])


@wp.kernel
def distinct_from_predecessor(
    loop_vertices: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Marks the survivors of a run of repeats within each loop: a position is kept unless it
    # repeats its predecessor. The first position of a loop is always kept, so a run that wraps the
    # seam keeps its head. Slots past the rewritten loops keep nothing.
    t = wp.int32(wp.tid())
    if not shorten_loop_rewriting(state, positions):
        return
    out_counts[t] = wp.int32(0)
    if shorten_loop_past_end(loop_offsets, t):
        return
    if t == loop_offsets[shorten_loop_owner(loop_offsets, t)]:
        out_counts[t] = wp.int32(1)
    elif loop_vertices[t] != loop_vertices[t - 1]:
        out_counts[t] = wp.int32(1)


@wp.kernel
def compact_kept(
    values: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    kept: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_values: wp.array[wp.int32],
    out_loop_offsets: wp.array[wp.int32],
) -> None:
    # Stream compaction against 0/1 counts scanned in place behind a leading zero, so a kept
    # position is a step of the scan: a scatter, so it stays a kernel. Threads up to ``n_loops``
    # map each loop's offset through the same scan.
    t = wp.int32(wp.tid())
    if not shorten_loop_rewriting(state, positions):
        return
    remap_loop_offset(loop_offsets, kept, t, out_loop_offsets)
    slot = kept[t]
    if kept[t + 1] != slot:
        out_values[slot] = values[t]


@wp.kernel
def shorten_loop_advance(
    positions: wp.array[wp.int32], kept: wp.array[wp.int32], out_state: wp.array[wp.int32]
) -> None:
    # dim=1, closing a sweep. A sweep whose rewrite would not fit flags the overflow and stops the
    # loop *without* counting itself -- its rewrite and compaction were skipped, so the loops are
    # the ones it started from and the host can grow the buffers and run it again. Otherwise the
    # sweep counts, and the loop stops once two consecutive sweeps (one of each parity) accepted
    # nothing, the cap is reached, or no position is left. Two, not one: a rewrite can shift which
    # positions land on which parity, so one unchanged sweep says nothing about the other parity.
    if out_state[SHORTEN_DONE] != 0:
        return
    capacity = positions.shape[0] - 1
    changed = out_state[SHORTEN_CHANGED]
    if changed != 0 and positions[capacity] > capacity:
        out_state[SHORTEN_OVERFLOW] = 1
        out_state[SHORTEN_DONE] = 1
        return
    out_state[SHORTEN_CHANGED] = 0
    sweeps = out_state[SHORTEN_SWEEPS] + 1
    out_state[SHORTEN_SWEEPS] = sweeps
    unchanged = wp.int32(0)
    if changed == 0:
        unchanged = out_state[SHORTEN_UNCHANGED] + 1
    else:
        out_state[SHORTEN_ANY] = 1
        out_state[SHORTEN_LENGTH] = kept[capacity]
    out_state[SHORTEN_UNCHANGED] = unchanged
    if unchanged >= 2 or sweeps >= out_state[SHORTEN_MAX_SWEEPS] or out_state[SHORTEN_LENGTH] == 0:
        out_state[SHORTEN_DONE] = 1


@wp.kernel
def seed_graph_sources(sources: wp.array[wp.int32], out_distance: wp.array[wp.float32]) -> None:
    # ``geodesic_path``'s edge-graph distance seed: zero at every source vertex (the rest of
    # ``out_distance`` holds the "unreached" value it was allocated with).
    out_distance[sources[wp.int32(wp.tid())]] = wp.float32(0.0)
