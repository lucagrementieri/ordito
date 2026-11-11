from typing import Any

import warp as wp

from ordito.constants import INT32_MAX_CONSTANT, PI, TOLERANCE_ZERO_CONSTANT, TWO_PI
from ordito.kernels.array import device_is_cpu
from ordito.kernels.halfedge import halfedge_destination, halfedge_prev, ring_long_from
from ordito.kernels.normals import (
    gathered_angle_weighted_sums,
    halfedge_angle_weighted_term,
    normalized_narrowed,
)
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
def parked_corner_angle(
    parked: wp.array[wp.float64], _same_table: wp.array[wp.float64], h: wp.int32
) -> wp.float64:
    # ``tangent_angle_rows``' corner angle at the origin of ``h`` parked in its own output slot by
    # ``park_corner_angles``, passed as both angle sources and as the output: a halfedge lies on
    # exactly one ring, its origin's, so the walk reads its slot before overwriting it.
    return parked[h]


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
    out_angles: wp.array[wp.float64],
):
    # One lane's two passes over a counter-clockwise ring: the first sums the corner angles
    # (``angle_of``) into the total angle Theta_v, the second lays the halfedges out as polar
    # coordinates in the flattened tangent plane (``ring_tangent_scale``), from the reference slot
    # (``ring_reference_slot``), reading each angle again before overwriting its slot (a table
    # lookup, or the angle ``park_corner_angles`` parked in that slot: parking a table variant's
    # angles cost it 1.1-1.7x on CUDA, a scattered write and read, 2026-11-08).
    total = wp.float64(0.0)
    for j in range(begin, end):
        total += angle_of(angle_data, angle_faces, ring_halfedges[j])
    scale = ring_tangent_scale(total, boundary)
    start = ring_reference_slot(ring_halfedges, begin, end, boundary)
    accumulated = wp.float64(0.0)
    for t in range(end - begin):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        angle = angle_of(angle_data, angle_faces, h)
        out_angles[h] = scale * accumulated
        accumulated += angle


@wp.func
def ring_reference_slot_block(
    ring_halfedges: wp.array[wp.int32],
    begin: wp.int32,
    end: wp.int32,
    boundary: wp.bool,
    lane: wp.int32,
) -> wp.int32:
    # ``ring_reference_slot`` by the whole block (every lane must call it, and every lane gets it):
    # the slot of the ring's largest halfedge, or ``0`` on an open fan.
    start = wp.int32(0)
    if not boundary:
        lanes = wp.block_dim()
        largest = wp.int32(-1)
        for j in range(begin + lane, end, lanes):
            largest = wp.max(largest, ring_halfedges[j])
        largest = block_max(largest)
        slot = wp.int32(-1)
        for j in range(begin + lane, end, lanes):
            if ring_halfedges[j] == largest:
                slot = j - begin
        start = block_max(slot)
    return start


@wp.func
def ring_lane_chunk(begin: wp.int32, end: wp.int32, lane: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Lane ``lane``'s contiguous chunk ``[first, last)`` of a ring's ``end - begin`` slots counted
    # from its reference slot, the ring cut into one chunk a lane of the block.
    n = end - begin
    chunk = (n + wp.block_dim() - 1) // wp.block_dim()
    first = wp.min(lane * chunk, n)
    return first, wp.min(first + chunk, n)


@wp.func
def lanes_before_and_total(partial: wp.float64, lane: wp.int32) -> tuple[wp.float64, wp.float64]:
    # The sum of the lanes' ``partial`` before lane ``lane`` and over every lane, both in lane
    # order, so a ring cut by ``ring_lane_chunk`` resumes each chunk where the one before ended
    # (``tile_scan_*`` takes no ``float64``). Every lane must call it.
    partials = wp.tile(partial)
    before = wp.float64(0.0)
    total = wp.float64(0.0)
    for i in range(wp.block_dim()):
        if i == lane:
            before = total
        total += wp.tile_extract(partials, i)
    return before, total


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
    start = ring_reference_slot_block(ring_halfedges, begin, end, boundary, lane)
    first, last = ring_lane_chunk(begin, end, lane)
    partial = wp.float64(0.0)
    for t in range(first, last):
        h = ring_halfedges[ring_slot(begin, end, start, t)]
        angle = angle_of(angle_data, angle_faces, h)
        out_angles[h] = angle
        partial += angle
    before, total = lanes_before_and_total(partial, lane)
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
                angle_of, out_angles,
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
    # ``halfedge_tangent_angles_parked`` is the same walk over the angles ``park_corner_angles``
    # formed from the positions.
    blk, lane = wp.tid()
    tangent_angle_rows(
        face_angles, face_angles, ring_offsets, ring_halfedges, is_boundary, blk, lane,
        table_corner_angle, out_angles,
    )  # fmt: skip


# Lanes of a ``park_corner_angles`` block: a thread a face of ``float64`` arithmetic, so a small
# mesh's faces spread over more SMs' ``float64`` units (2026-11-11, device time against 256:
# ``sphere_small`` 0.47x, 11.1 -> 5.2 us; ``sphere_med`` 1.00x; dragon 0.97x; 32 the same as 64).
# Its own width, so the kernel is in a module of its own (``module="unique"``).
PARK_LANES = 64


@wp.kernel(enable_backward=False, module="unique")
def park_corner_angles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_angles: wp.array[wp.float64]
) -> None:
    # A face's three corner angles (``triangles.face_corner_angles_d``: the ``angles`` kernel's
    # rule in ``float64``, as geometry-central forms its angles -- the vector heat method's
    # transport reads these, and ``float32`` angles alone moved its log map by 4e-5 on needle
    # meshes) parked in its three halfedges' output slots ``3 f + k`` for
    # ``halfedge_tangent_angles_parked``. Formed in the ring walk instead, each halfedge formed its
    # face's three (the degeneracy guard reads all three) to keep one, three times the ``float64``
    # arithmetic on the lanes of 256-vertex blocks, few enough SMs that a small mesh's walk was
    # bound by their ``float64`` issue (``sphere_small`` and ``sphere_med`` alike ~60 us, dragon
    # 0.88 ms; a hub's on one SM). A thread a face.
    f = wp.int32(wp.tid())
    a0, a1, a2 = face_corner_angles_d(vertices, faces, f)
    out_angles[3 * f] = a0
    out_angles[3 * f + 1] = a1
    out_angles[3 * f + 2] = a2


@wp.kernel
def halfedge_tangent_angles_parked(
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    out_angles: wp.array[wp.float64],
) -> None:
    # ``halfedge_tangent_angles`` once ``park_corner_angles`` has parked every halfedge's corner
    # angle in its output slot: ``tangent_angle_rows`` reading them back (``parked_corner_angle``);
    # launched tiled.
    blk, lane = wp.tid()
    tangent_angle_rows(
        out_angles, out_angles, ring_offsets, ring_halfedges, is_boundary, blk, lane,
        parked_corner_angle, out_angles,
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
def fan_anchor_term(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3,
    h: wp.int32,
    theta: wp.float32,
    frame_x: wp.vec3,
    frame_y: wp.vec3,
) -> wp.vec2:
    # ``fan_anchor``'s term of ring halfedge ``h``: its edge in the frame, rotated back by its
    # normalized polar angle ``theta``. Shared with ``long_ring_frame``'s block walk.
    edge = world_to_tangent(vertices[halfedge_destination(faces, h)] - origin, frame_x, frame_y)
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    return wp.vec2(
        edge[0] * cos_theta + edge[1] * sin_theta, edge[1] * cos_theta - edge[0] * sin_theta
    )


@wp.func
def fan_rim_edge(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3,
    last: wp.int32,
    frame_x: wp.vec3,
    frame_y: wp.vec3,
) -> wp.vec2:
    # An open fan's closing rim edge in the frame, the far edge of its last wedge (ring halfedge
    # ``last``), at half a turn: ``fan_anchor`` subtracts it.
    return world_to_tangent(vertices[faces[halfedge_prev(last)]] - origin, frame_x, frame_y)


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
        term = fan_anchor_term(vertices, faces, origin, h, scale * accumulated, frame_x, frame_y)
        anchor_x += term[0]
        anchor_y += term[1]
        accumulated += halfedge_corner_angle(vertices, faces, h)
    if boundary:
        edge = fan_rim_edge(vertices, faces, origin, ring_halfedges[end - 1], frame_x, frame_y)
        anchor_x -= edge[0]
        anchor_y -= edge[1]
    return wp.vec2(anchor_x, anchor_y)


@wp.func
def anchored_basis_x(
    basis_x: wp.vec3, frame_y: wp.vec3, anchor: wp.vec2, length: wp.float32
) -> wp.vec3:
    # ``basis_x`` turned to the fan's ``anchor`` (``fan_anchor``) in the provisional frame
    # ``(basis_x, frame_y)``, unless the anchor vanishes against the reference edge's ``length``.
    if wp.length(anchor) > TOLERANCE_ZERO_CONSTANT * length:
        return wp.normalize(anchor[0] * basis_x + anchor[1] * frame_y)
    return basis_x


@wp.func
def ring_reference_tangent(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    h: wp.int32,
    v: wp.int32,
    normal: wp.vec3,
) -> tuple[wp.vec3, wp.float32]:
    # The provisional ``basis_x`` the fan's anchor is fitted in: reference halfedge ``h``'s edge
    # projected into the tangent plane of unit ``normal`` at ``v``, and that projection's length.
    direction = vertices[halfedge_destination(faces, h)] - vertices[v]
    return reference_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)


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
    basis_x, length = ring_reference_tangent(
        vertices, faces, ring_halfedges[begin + start], v, normal
    )
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
        basis_x = anchored_basis_x(basis_x, frame_y, anchor, length)
    return basis_x, wp.cross(normal, basis_x)


def _vertex_tangent_frames(name: str, short_only: bool) -> wp.Kernel:
    # The frame around a caller's normals; ``_vertex_tangent_frames_gathered`` forms the normal
    # over the ring first. ``short_only`` (the ``_short`` kernels) skips a ring of
    # ``halfedge.ring_long_from`` halfedges or more, which ``vertex_one_rings`` listed and
    # ``vertex_tangent_frames_long`` lays out a block a ring; the wrapper takes it only when that
    # list is known to be nonempty, so a mesh without a long ring runs the plain kernel and pays no
    # argument or test for it.
    def _kernel(
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        normals: wp.array[wp.vec3],
        ring_offsets: wp.array[wp.int32],
        ring_halfedges: wp.array[wp.int32],
        is_boundary: wp.array[wp.bool],
        out_basis_x: wp.array[wp.vec3],
        out_basis_y: wp.array[wp.vec3],
    ) -> None:
        v = wp.int32(wp.tid())
        if wp.static(short_only):
            if ring_offsets[v + 1] - ring_offsets[v] >= ring_long_from(ring_halfedges.shape[0]):
                return
        total = ring_angle_total(vertices, faces, ring_offsets, ring_halfedges, v)
        basis_x, basis_y = vertex_tangent_basis(
            vertices, faces, normals[v], ring_offsets, ring_halfedges, is_boundary[v], v, total
        )
        out_basis_x[v] = basis_x
        out_basis_y[v] = basis_y

    return wp.kernel(_kernel, name=name)


vertex_tangent_frames = _vertex_tangent_frames("vertex_tangent_frames", False)
vertex_tangent_frames_short = _vertex_tangent_frames("vertex_tangent_frames_short", True)


def _vertex_tangent_frames_gathered(name: str, short_only: bool) -> wp.Kernel:
    # ``_vertex_tangent_frames`` with the angle-weighted normal gathered over the ring the frame
    # walks (``normals.gathered_angle_weighted_sums``), in place of a ``float64`` scatter, its
    # zeroed accumulator and a launch; the same walk totals the corner angles the frame's scale
    # reads, which the caller-normal kernel walks for alone. The normal written is
    # ``vertex_normals``' up to the order its ``float64`` sum is taken in. ``short_only`` as there.
    #
    # The long rings' block is a kernel of its own because block code inlined here costs every
    # ordinary walk: ``tangent_angle_rows``' block deferral built into both frame kernels
    # (2026-11-08) was byte-identical elsewhere and took the hub 0.016-0.018x, but cost ordinary
    # meshes 1.11-1.23x, wherever the block's decision was taken (a reduction after or before the
    # walks, a barrier-free warp vote) and under a 56-register cap; packing its sums into vectors
    # cost 1.2-1.7x.
    def _kernel(
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        ring_offsets: wp.array[wp.int32],
        ring_halfedges: wp.array[wp.int32],
        is_boundary: wp.array[wp.bool],
        out_basis_x: wp.array[wp.vec3],
        out_basis_y: wp.array[wp.vec3],
        out_normals: wp.array[wp.vec3],
    ) -> None:
        v = wp.int32(wp.tid())
        if wp.static(short_only):
            if ring_offsets[v + 1] - ring_offsets[v] >= ring_long_from(ring_halfedges.shape[0]):
                return
        sums, total = gathered_angle_weighted_sums(vertices, faces, ring_offsets, ring_halfedges, v)
        normal = normalized_narrowed(sums[0], sums[1], sums[2])
        out_normals[v] = normal
        basis_x, basis_y = vertex_tangent_basis(
            vertices, faces, normal, ring_offsets, ring_halfedges, is_boundary[v], v, total
        )
        out_basis_x[v] = basis_x
        out_basis_y[v] = basis_y

    return wp.kernel(_kernel, name=name)


vertex_tangent_frames_gathered = _vertex_tangent_frames_gathered(
    "vertex_tangent_frames_gathered", False
)
vertex_tangent_frames_gathered_short = _vertex_tangent_frames_gathered(
    "vertex_tangent_frames_gathered_short", True
)

# Lanes of a ``vertex_tangent_frames_long`` block, which lays out one long ring: ``fan_hub``'s two
# 40 960-spoke hubs took the launch 1.3x as long at 256 (2026-11-10), and 1 024 asks more registers
# than a block holds (CUDA error 701). Its own width, so the kernel is in a module of its own
# (``module="unique"``) rather than rebuilding the per-vertex kernels' at it.
FRAME_LONG_LANES = 512


@wp.func
def long_ring_frame(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    boundary: wp.bool,
    v: wp.int32,
    given_normal: wp.vec3,
    gather: wp.bool,
    lane: wp.int32,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    # ``vertex_tangent_basis`` (after ``gathered_angle_weighted_sums`` when ``gather``, else around
    # ``given_normal``) by the whole block, for a ring one thread would walk twice through ``d``
    # corners in turn; every lane must call it, and every lane gets ``(basis_x, basis_y, normal)``.
    # The sums are the walk's terms in a fixed chunked order, in ``float64`` throughout, so a hub's
    # frame repeats bit for bit but is not the one-thread walk's (whose ``float32`` sums it is
    # nearer a ``float64`` frame than). The ring is never empty here.
    begin = ring_offsets[v]
    end = ring_offsets[v + 1]
    sums = wp.vec4d()
    for j in range(begin + lane, end, wp.block_dim()):
        term, angle = halfedge_angle_weighted_term(vertices, faces, ring_halfedges[j])
        sums += wp.vec4d(term[0], term[1], term[2], wp.float64(angle))
    sums = block_sum(sums)
    normal = given_normal
    if gather:
        normal = normalized_narrowed(sums[0], sums[1], sums[2])
    if wp.length(normal) <= TOLERANCE_ZERO_CONSTANT:
        return wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), normal
    start = ring_reference_slot_block(ring_halfedges, begin, end, boundary, lane)
    basis_x, length = ring_reference_tangent(
        vertices, faces, ring_halfedges[begin + start], v, normal
    )
    scale = ring_tangent_scale(wp.float32(sums[3]), boundary)
    if scale > wp.float32(0.0):
        # ``fan_anchor`` a chunk a lane, each resuming the polar angle where the chunk before it
        # ended (``lanes_before_and_total``).
        frame_y = wp.cross(normal, basis_x)
        origin = vertices[v]
        first, last = ring_lane_chunk(begin, end, lane)
        partial = wp.float64(0.0)
        for t in range(first, last):
            h = ring_halfedges[ring_slot(begin, end, start, t)]
            partial += wp.float64(halfedge_corner_angle(vertices, faces, h))
        accumulated, _total = lanes_before_and_total(partial, lane)
        anchor = wp.vec2d()
        for t in range(first, last):
            h = ring_halfedges[ring_slot(begin, end, start, t)]
            theta = scale * wp.float32(accumulated)
            anchor += wp.vec2d(fan_anchor_term(vertices, faces, origin, h, theta, basis_x, frame_y))
            accumulated += wp.float64(halfedge_corner_angle(vertices, faces, h))
        anchor = block_sum(anchor)
        if boundary:
            anchor -= wp.vec2d(
                fan_rim_edge(vertices, faces, origin, ring_halfedges[end - 1], basis_x, frame_y)
            )
        basis_x = anchored_basis_x(basis_x, frame_y, wp.vec2(anchor), length)
    return basis_x, wp.cross(normal, basis_x), normal


@wp.kernel(enable_backward=False, module="unique")
def vertex_tangent_frames_long(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    long_rings: wp.array[wp.int32],
    gather: wp.bool,
    out_basis_x: wp.array[wp.vec3],
    out_basis_y: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # The frames the ``_short`` frame kernels skipped, a block
    # a listed vertex (``long_ring_frame``); launched tiled at ``FRAME_LONG_LANES``, and only when
    # the caller knows a long ring exists, so a mesh without one never launches it. ``gather``
    # forms the normal into ``out_normals``; otherwise ``normals`` is read and ``out_normals`` is
    # not written (the caller passes ``normals`` for both).
    blk, lane = wp.tid()
    v = long_rings[blk]
    basis_x, basis_y, normal = long_ring_frame(
        vertices, faces, ring_offsets, ring_halfedges, is_boundary[v], v, normals[v], gather, lane
    )
    if lane == 0:
        out_basis_x[v] = basis_x
        out_basis_y[v] = basis_y
        if gather:
            out_normals[v] = normal


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
