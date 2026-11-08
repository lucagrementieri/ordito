from typing import Any

import warp as wp

from ordito.constants import INT32_MAX_CONSTANT, PI, TOLERANCE_ZERO_CONSTANT, TWO_PI
from ordito.kernels.array import device_is_cpu
from ordito.kernels.halfedge import halfedge_destination, halfedge_prev
from ordito.kernels.normals import gathered_angle_weighted_sums, normalized_narrowed
from ordito.kernels.predicates import any_perpendicular, reference_tangent, world_to_tangent
from ordito.kernels.reduce import block_chunk, block_max, block_sum
from ordito.kernels.triangles import (
    face_corner_angles_d,
    face_normals_and_area,
    halfedge_corner_angle,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def corner_angle(face_angles: wp.array2d[wp.float32], h: wp.int32) -> wp.float32:
    # The corner at the *origin* of halfedge ``3*f + k`` is corner ``k`` of face ``f``.
    return face_angles[h // wp.int32(3), h % wp.int32(3)]


@wp.func
def ring_tangent_scale(total: wp.Float, boundary: wp.bool) -> wp.Float:
    # Rescaling by 2*pi/Theta_v (pi/Theta_v at a boundary vertex, whose fan spans a half-disk) is
    # what makes the flattening consistent on a cone point; zero for a fan with no angle at all.
    # Initialized before the branch: a variable assigned only inside an ``if`` is readable
    # afterwards in Warp but uninitialized when the branch was not taken.
    scale = type(total)(0.0)
    if total > type(total)(TOLERANCE_ZERO_CONSTANT):
        if boundary:
            scale = type(total)(PI) / total
        else:
            scale = type(total)(TWO_PI) / total
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


@wp.func
def table_corner_angle(
    face_angles: wp.array2d[wp.float32], _same_table: wp.array2d[wp.float32], h: wp.int32
) -> wp.float64:
    # ``tangent_angle_rows``' corner angle at the origin of ``h`` from a ``float32`` angle table,
    # passed as both angle sources.
    return wp.float64(corner_angle(face_angles, h))


@wp.func
def position_corner_angle(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], h: wp.int32
) -> wp.float64:
    # ``tangent_angle_rows``' corner angle at the origin of ``h`` formed from the positions:
    # ``triangles.face_corner_angles_d``' (the ``angles`` kernel's rule in ``float64``, as
    # geometry-central forms its angles: the vector heat method's transport reads these, and
    # ``float32`` angles alone moved its log map by 4e-5 on needle meshes). Each halfedge forms its
    # face's three angles to keep the one it needs (the degeneracy guard reads all three), three
    # times the table's arithmetic, in place of the table's launch and its ``(n_faces, 3)`` round
    # trip.
    a0, a1, a2 = face_corner_angles_d(vertices, faces, h // wp.int32(3))
    return wp.vec3d(a0, a1, a2)[h % wp.int32(3)]


# Vertices a block of the tangent-angle kernels holds, a vertex a lane on CUDA, and the ring size
# from which a CUDA block lays a vertex's ring out together rather than one lane walking it
# (``tangent_angle_ring_block``).
TANGENT_ROWS_PER_BLOCK = 256
TANGENT_LONG_RING = 128


@wp.func
def tangent_angle_ring(
    angle_data: Any,
    angle_faces: Any,
    ring_halfedges: wp.array[wp.int32],
    begin: wp.int32,
    end: wp.int32,
    boundary: wp.bool,
    angle_of: wp.Function,
    park: wp.bool,
    out_angles: wp.array[wp.float64],
):
    # One lane's two passes over a counter-clockwise ring: the first sums the corner angles
    # (``angle_of``) into the total angle Theta_v, the second lays the halfedges out as polar
    # coordinates in the flattened tangent plane (``ring_tangent_scale``), from the reference slot
    # (``ring_reference_slot``). With ``park`` each angle formed in the first pass waits in the
    # halfedge's own output slot until the second reads it back and overwrites it -- a halfedge
    # lies on exactly one ring, its origin's, so the slot is this ring's alone; without it (a table
    # lookup, cheaper to repeat than a scattered write and read: parked, the table variant cost
    # 1.1-1.7x on CUDA, 2026-11-08) the second pass forms it again.
    total = wp.float64(0.0)
    for j in range(begin, end):
        h = ring_halfedges[j]
        angle = angle_of(angle_data, angle_faces, h)
        if park:
            out_angles[h] = angle
        total += angle
    scale = ring_tangent_scale(total, boundary)
    start = ring_reference_slot(ring_halfedges, begin, end, boundary)
    accumulated = wp.float64(0.0)
    for t in range(end - begin):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        angle = wp.float64(0.0)
        if park:
            angle = out_angles[h]
        else:
            angle = angle_of(angle_data, angle_faces, h)
        out_angles[h] = scale * accumulated
        accumulated += angle


@wp.func
def tangent_angle_ring_block(
    angle_data: Any,
    angle_faces: Any,
    ring_halfedges: wp.array[wp.int32],
    begin: wp.int32,
    end: wp.int32,
    boundary: wp.bool,
    lane: wp.int32,
    angle_of: wp.Function,
    out_angles: wp.array[wp.float64],
):
    # ``tangent_angle_ring`` by the whole block (every lane must call it), for a hub whose ring one
    # lane would walk through ``2 d`` ``float64`` corner angles in turn. The ring, read from the
    # reference slot, is cut into one contiguous chunk a lane: each lane forms and parks its
    # chunk's angles and sums them in order, every lane sums the chunk totals before its own (and
    # all of them, the total) in lane order, and each lays its chunk out from there. The sums are
    # a fixed chunked order, so the ring repeats bit for bit; it is not the one-lane walk's order,
    # which ``float64`` rounding separates by ulps (``tile_scan_*`` takes no ``float64``).
    n = end - begin
    lanes = wp.block_dim()
    start = wp.int32(0)
    if not boundary:
        # ``ring_reference_slot`` by the block: the slot of the ring's largest halfedge.
        largest = wp.int32(-1)
        for j in range(begin + lane, end, lanes):
            largest = wp.max(largest, ring_halfedges[j])
        largest = block_max(largest)
        slot = wp.int32(-1)
        for j in range(begin + lane, end, lanes):
            if ring_halfedges[j] == largest:
                slot = j - begin
        start = block_max(slot)
    chunk = (n + lanes - 1) // lanes
    first = wp.min(lane * chunk, n)
    last = wp.min(first + chunk, n)
    partial = wp.float64(0.0)
    for t in range(first, last):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        angle = angle_of(angle_data, angle_faces, h)
        out_angles[h] = angle
        partial += angle
    partials = wp.tile(partial)
    before = wp.float64(0.0)
    total = wp.float64(0.0)
    for i in range(lanes):
        if i == lane:
            before = total
        total += wp.tile_extract(partials, i)
    scale = ring_tangent_scale(total, boundary)
    accumulated = before
    for t in range(first, last):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        angle = out_angles[h]
        out_angles[h] = scale * accumulated
        accumulated += angle


@wp.func
def tangent_angle_rows(
    angle_data: Any,
    angle_faces: Any,
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    blk: wp.int32,
    lane: wp.int32,
    angle_of: wp.Function,
    park: wp.bool,
    out_angles: wp.array[wp.float64],
):
    # Block ``blk``'s ``TANGENT_ROWS_PER_BLOCK`` vertices of a tangent-angle launch: a lane walks
    # each ring (``tangent_angle_ring``), and on CUDA a ring of ``TANGENT_LONG_RING`` halfedges or
    # more is left to the whole block after (``tangent_angle_ring_block``), so a block without one
    # pays one reduction. The CPU device, whose single lane walks every ring in turn anyway,
    # defers none, so its angles are the one-lane walk's.
    first, rows = block_chunk(is_boundary.shape[0], blk, TANGENT_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    long_from = TANGENT_LONG_RING
    if device_is_cpu() != 0:
        long_from = INT32_MAX_CONSTANT
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        v = first + r
        begin = ring_offsets[v]
        end = ring_offsets[v + 1]
        if end - begin >= long_from:
            found = 1
        elif end > begin:
            tangent_angle_ring(
                angle_data, angle_faces, ring_halfedges, begin, end, is_boundary[v],
                angle_of, park, out_angles,
            )  # fmt: skip
    if block_sum(found) == 0:
        return
    for r in range(rows):
        v = first + r
        begin = ring_offsets[v]
        end = ring_offsets[v + 1]
        if end - begin >= long_from:
            tangent_angle_ring_block(
                angle_data, angle_faces, ring_halfedges, begin, end, is_boundary[v], lane,
                angle_of, out_angles,
            )  # fmt: skip


@wp.kernel
def halfedge_tangent_angles(
    face_angles: wp.array2d[wp.float32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_angles: wp.array[wp.float64],
) -> None:
    # ``tangent_angle_rows`` over a ``float32`` corner-angle table; launched tiled.
    # ``halfedge_tangent_angles_from_positions`` is the same walk forming the corner angles itself.
    blk, lane = wp.tid()
    tangent_angle_rows(
        face_angles, face_angles, ring_offsets, ring_halfedges, is_boundary, blk, lane,
        table_corner_angle, False, out_angles,
    )  # fmt: skip


@wp.kernel
def halfedge_tangent_angles_from_positions(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_angles: wp.array[wp.float64],
) -> None:
    # ``halfedge_tangent_angles`` for a caller with no corner-angle table: ``tangent_angle_rows``
    # forming each angle from the positions (``position_corner_angle``); launched tiled.
    blk, lane = wp.tid()
    tangent_angle_rows(
        vertices, faces, ring_offsets, ring_halfedges, is_boundary, blk, lane,
        position_corner_angle, True, out_angles,
    )  # fmt: skip


@wp.func
def ring_angle_total(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    vertex: wp.int32,
) -> wp.float32:
    # Total corner angle Theta_v of ``vertex``'s ring, summed in ring order from angles formed on
    # demand (``triangles.halfedge_corner_angle``). ``normals.gathered_angle_weighted_sums`` forms
    # the same sum, bit for bit, on its normal walk.
    total = wp.float32(0.0)
    for j in range(ring_offsets[vertex], ring_offsets[vertex + 1]):
        total += halfedge_corner_angle(vertices, faces, ring_halfedges[j])
    return total


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
        edge = world_to_tangent(vertices[faces[halfedge_prev(h)]] - origin, frame_x, frame_y)
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
    total: wp.float32,
) -> tuple[wp.vec3, wp.vec3]:
    # ``basis_x`` is where ``halfedge_tangent_angles`` puts polar angle 0, fitted to the whole fan
    # (``fan_anchor``) rather than read off the reference halfedge, which it equals only where the
    # fan is flat -- so the two functions describe one coordinate system, geometry-central's. The
    # fit is taken in a provisional frame around the reference edge's projection. ``total`` is the
    # vertex's corner angle sum in ring order, which the caller forms on a walk it already makes.
    if wp.length(normal) <= TOLERANCE_ZERO_CONSTANT:
        # An unreferenced vertex has no tangent plane at all. Emit a fixed unit frame rather than
        # zero vectors, which would surface downstream as NaN out of a normalization.
        return wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)
    begin = ring_offsets[v]
    end = ring_offsets[v + 1]
    if end <= begin:
        basis_x = any_perpendicular(normal)
        return basis_x, wp.cross(normal, basis_x)
    start = ring_reference_slot(ring_halfedges, begin, end, boundary)
    h = ring_halfedges[begin + start]
    direction = vertices[halfedge_destination(faces, h)] - vertices[v]
    basis_x, length = reference_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)
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
    # The frame around a caller's normals; ``vertex_tangent_frames_gathered`` forms the normal
    # over the ring first.
    v = wp.int32(wp.tid())
    total = ring_angle_total(vertices, faces, ring_offsets, ring_halfedges, v)
    basis_x, basis_y = vertex_tangent_basis(
        vertices, faces, normals[v], ring_offsets, ring_halfedges, is_boundary[v], v, total
    )
    out_basis_x[v] = basis_x
    out_basis_y[v] = basis_y


@wp.kernel
def vertex_tangent_frames_gathered(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # ``vertex_tangent_frames`` with the angle-weighted normal gathered over the ring the frame
    # walks (``normals.gathered_angle_weighted_sums``), in place of a ``float64`` scatter, its
    # zeroed accumulator and a launch; the same walk totals the corner angles the frame's scale
    # reads, which the caller-normal kernel walks for alone. The normal written is
    # ``vertex_normals``' up to the order its ``float64`` sum is taken in.
    #
    # A hub's ring stays one thread's walk here (``fan_hub`` ~2 850x ``sphere_med``'s kernel).
    # ``tangent_angle_rows``' block deferral was built for both frame kernels (2026-11-08):
    # byte-identical elsewhere, the hub 0.016-0.018x and 450x nearer a ``float64`` frame, but the
    # inlined block path alone cost ordinary meshes 1.11-1.23x, wherever the block's decision was
    # taken (a reduction after or before the walks, a barrier-free warp vote) and under a
    # 56-register cap; packing its sums into vectors cost 1.2-1.7x. Only removing the block code
    # restored par.
    v = wp.int32(wp.tid())
    sums, total = gathered_angle_weighted_sums(vertices, faces, ring_offsets, ring_halfedges, v)
    normal = normalized_narrowed(sums[0], sums[1], sums[2])
    out_normals[v] = normal
    basis_x, basis_y = vertex_tangent_basis(
        vertices, faces, normal, ring_offsets, ring_halfedges, is_boundary[v], v, total
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
def wrap_angle(angle: wp.Float) -> wp.Float:
    # Fold into (-pi, pi] without a modulo: the round trip through the unit circle is branch-free
    # and immune to the sign convention of ``%`` inside kernels.
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.kernel
def halfedge_transport_angles(
    twins: wp.array[wp.int32], tangent_angles: wp.array[wp.float64], out_rho: wp.array[wp.float64]
) -> None:
    h = wp.int32(wp.tid())
    twin = twins[h]
    if twin >= wp.int32(0):
        opposite = tangent_angles[twin]
    else:
        # A boundary edge exists as a halfedge in one direction only. At the destination vertex it
        # is the fan-closing edge, whose polar angle is the rescaled total angle: pi.
        opposite = wp.float64(PI)
    out_rho[h] = wrap_angle(opposite + wp.float64(PI) - tangent_angles[h])
