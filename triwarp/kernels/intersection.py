from typing import Any

import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels import triangles as kernel_triangles
from triwarp.kernels.array import (
    OverloadTable,
    csr_run_start,
    declare_map_signatures,
    map_probe,
    map_probe_single,
)
from triwarp.kernels.predicates import point_plane_dot, triangles_intersect
from triwarp.kernels.proximity import mesh_aabb_collect

SLICE_SIGN_INSIDE = wp.constant(wp.int32(-1))
SLICE_SIGN_OUTSIDE = wp.constant(wp.int32(1))
SLICE_SIGN_ON_PLANE = wp.constant(wp.int32(0))
EDGE_DENOM_EPSILON = wp.constant(wp.float32(1e-12))

CASE_NONE = wp.constant(wp.int32(0))
CASE_BASIC = wp.constant(wp.int32(1))
CASE_ONE_VERTEX = wp.constant(wp.int32(2))
CASE_ONE_EDGE = wp.constant(wp.int32(3))


@wp.func
def triangle_case_code(s0: wp.int32, s1: wp.int32, s2: wp.int32) -> wp.int32:
    # ``CASE_ONE_EDGE`` fires only for the sorted pattern ``(0, 0, 1)`` -- an edge lying in the
    # plane with its free vertex strictly positive -- and *not* for its sign-flipped mirror
    # ``(-1, 0, 0)``. That asymmetry looks like a bug (an edge-in-plane face with the free vertex
    # negative silently reports no segment) and was changed to fire on both for a session, which
    # regressed test_mesh_with_plane_matches_trimesh: an in-plane mesh edge is normally shared by
    # two faces whose own free vertices sit on *opposite* sides of the plane, so making the test
    # symmetric makes both faces emit the same segment (a duplicate), where the asymmetric form
    # emits it from exactly one side. Confirmed on the icosahedron's axis-plane cut: the shared
    # edge (vtx0, vtx1) lies exactly in the z=0 plane, face [0, 5, 1]'s free vertex is at +z (kept)
    # and face [0, 1, 7]'s is at -z (dropped) -- trimesh reports that edge once, matching only the
    # positive side. A genuine residual gap remains for a *boundary* edge (one incident face, no
    # partner to report from) whose lone free vertex is negative, which still reports nothing; that
    # is unmeasured and is a narrower, real question than the symmetric fix this comment replaces.
    sa, sb, sc = kernel_array.sort3(s0, s1, s2)
    coded = wp.int32(14) + (sa << 3) + (sb << 2) + (sc << 1)
    if coded == wp.int32(4) or coded == wp.int32(12):
        return CASE_BASIC
    if coded == wp.int32(8):
        return CASE_ONE_VERTEX
    if coded == wp.int32(16):
        return CASE_ONE_EDGE
    return CASE_NONE


@wp.func
def plane_with_line(
    plane_normal: wp.vec3, plane_origin: wp.vec3, p0: wp.vec3, p1: wp.vec3, line_segments: wp.bool
) -> tuple[wp.vec3, wp.bool]:
    line_dir = wp.normalize(p1 - p0)
    n = wp.normalize(plane_normal)
    t = wp.dot(n, plane_origin - p0)
    b = wp.dot(n, line_dir)
    valid = wp.abs(b) > TOLERANCE_ZERO_CONSTANT
    if line_segments:
        test = wp.dot(n, plane_origin - p1)
        different_sides = wp.sign(t) != wp.sign(test)
        nonzero = (wp.abs(t) > TOLERANCE_ZERO_CONSTANT) or (wp.abs(test) > TOLERANCE_ZERO_CONSTANT)
        valid = valid and different_sides and nonzero
    if valid:
        d = t / b
        return p0 + line_dir * d, wp.bool(True)
    return wp.vec3(0.0, 0.0, 0.0), wp.bool(False)


@wp.func
def find_unique_sign_vertex(s0: wp.int32, s1: wp.int32, s2: wp.int32) -> wp.int32:
    # The corner whose sign differs from the other two -- shared by ``CASE_BASIC`` (this file's
    # own sign convention), the split path's edge-to-edge cut (``face_level_set_signs``'
    # convention) and ``marching_triangles_segments`` (a ``0`` / ``1`` side flag), which is what
    # makes the predicate itself convention-independent: it only compares the three inputs to each
    # other, never to a named sign value. Every caller has already established that exactly one of
    # the three differs, which is what makes corner 2 a safe fallthrough rather than a third test;
    # the all-equal input the two conditions above would otherwise disagree on never reaches a
    # caller.
    if s0 != s1 and s0 != s2:
        return wp.int32(0)
    if s1 != s0 and s1 != s2:
        return wp.int32(1)
    return wp.int32(2)


@wp.func
def vertex_at(local_index: wp.int32, v0: Any, v1: Any, v2: Any) -> Any:
    # Generic over the vector's precision: called on ``wp.vec3`` throughout
    # ``mesh_with_plane_segment_for_face`` and on ``wp.vec3d`` inside
    # ``triangle_intersection_segment``, which does its own arithmetic in float64 (see there).
    if local_index == wp.int32(0):
        return v0
    if local_index == wp.int32(1):
        return v1
    return v2


@wp.func
def corner_and_successors(corner: wp.int32, v0: Any, v1: Any, v2: Any) -> tuple[Any, Any, Any]:
    # Corner ``corner``'s vertex followed by the next two around the face, in winding order.
    return (
        vertex_at(corner, v0, v1, v2),
        vertex_at((corner + wp.int32(1)) % wp.int32(3), v0, v1, v2),
        vertex_at((corner + wp.int32(2)) % wp.int32(3), v0, v1, v2),
    )


@wp.func
def find_corner_with_sign(s0: wp.int32, s1: wp.int32, s2: wp.int32, sign: wp.int32) -> wp.int32:
    # The corner carrying ``sign``, for the cut cases where exactly one does. Corner 2 is the
    # fallthrough rather than a third test: the caller has already established that one of the
    # three matches.
    if s0 == sign:
        return wp.int32(0)
    if s1 == sign:
        return wp.int32(1)
    return wp.int32(2)


@wp.func
def mesh_with_plane_segment_for_face(
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    s0: wp.int32,
    s1: wp.int32,
    s2: wp.int32,
) -> tuple[wp.bool, wp.vec3, wp.vec3]:
    case_code = triangle_case_code(s0, s1, s2)

    if case_code == CASE_BASIC:
        unique_v, va, vb = corner_and_successors(find_unique_sign_vertex(s0, s1, s2), v0, v1, v2)
        p_a, valid_a = plane_with_line(plane_normal, plane_origin, unique_v, va, False)
        p_b, valid_b = plane_with_line(plane_normal, plane_origin, unique_v, vb, False)
        if valid_a and valid_b:
            return True, p_a, p_b
        return False, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    if case_code == CASE_ONE_VERTEX:
        on_plane_i = find_corner_with_sign(s0, s1, s2, SLICE_SIGN_ON_PLANE)
        on_plane_v, va, vb = corner_and_successors(on_plane_i, v0, v1, v2)
        hit, valid = plane_with_line(plane_normal, plane_origin, va, vb, False)
        if valid:
            return True, on_plane_v, hit
        return False, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    if case_code == CASE_ONE_EDGE:
        if s0 == wp.int32(0) and s1 == wp.int32(0):
            i0 = wp.int32(0)
            i1 = wp.int32(1)
        elif s0 == wp.int32(0) and s2 == wp.int32(0):
            i0 = wp.int32(0)
            i1 = wp.int32(2)
        else:
            i0 = wp.int32(1)
            i1 = wp.int32(2)
        return True, vertex_at(i0, v0, v1, v2), vertex_at(i1, v0, v1, v2)

    return False, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def mesh_with_plane_segments(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    out_cut: wp.array[wp.int32],
    out_segments: wp.array2d[wp.vec3],
) -> None:
    # ``out_cut`` is a ``0`` / ``1`` flag, the ``int32`` that ``compact_cut_segments``' caller scans
    # in place, so no mask has to be converted before the scan. Each corner's plane dot is taken
    # here rather than read from a per-vertex buffer, which would be a map and an allocation for
    # the same value.
    f = wp.int32(wp.tid())
    i0 = faces[f * 3]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    s0 = kernel_array.sign_with_tolerance(
        point_plane_dot(v0, plane_normal, plane_origin), TOLERANCE_MERGE_CONSTANT
    )
    s1 = kernel_array.sign_with_tolerance(
        point_plane_dot(v1, plane_normal, plane_origin), TOLERANCE_MERGE_CONSTANT
    )
    s2 = kernel_array.sign_with_tolerance(
        point_plane_dot(v2, plane_normal, plane_origin), TOLERANCE_MERGE_CONSTANT
    )
    valid, p0, p1 = mesh_with_plane_segment_for_face(
        plane_normal, plane_origin, v0, v1, v2, s0, s1, s2
    )
    out_cut[f] = wp.where(valid, 1, 0)
    out_segments[f, 0] = p0
    out_segments[f, 1] = p1


@wp.func
def vec3_equal(a: wp.vec3, b: wp.vec3) -> wp.bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def vec3_argsort(a: wp.vec3d) -> tuple[wp.int32, wp.int32, wp.int32]:
    # float64 rather than generic: its only caller is ``intersection_line_coordinate``, itself only
    # called from ``triangle_intersection_segment``, which does its whole computation in float64 --
    # see the comment there for why.
    xy = a[0] <= a[1]
    yz = a[1] <= a[2]
    xz = a[0] <= a[2]
    if xy:
        if yz:
            return wp.int32(0), wp.int32(1), wp.int32(2)
        if xz:
            return wp.int32(0), wp.int32(2), wp.int32(1)
        return wp.int32(2), wp.int32(0), wp.int32(1)
    if xz:
        return wp.int32(1), wp.int32(0), wp.int32(2)
    if yz:
        return wp.int32(1), wp.int32(2), wp.int32(0)
    return wp.int32(2), wp.int32(1), wp.int32(0)


@wp.func
def interval_intersect(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    return wp.vec2d(wp.max(a[0], b[0]), wp.min(a[1], b[1]))


@wp.func
def intersection_line_coordinate(
    start1: wp.vec3d, direction1: wp.vec3d, start2: wp.vec3d, direction2: wp.vec3d
) -> wp.float64:
    minors = wp.cross(direction1, direction2)
    order_x, _order_y, order_z = vec3_argsort(minors)
    minor_x = minors[order_x]
    minor_z = minors[order_z]
    i = order_z
    if minor_z >= -minor_x:
        i = order_z
    else:
        i = order_x
    offset_minors = wp.cross(start2 - start1, direction2)
    numerator = offset_minors[i]
    denominator = minors[i]
    return numerator / denominator


@wp.func
def triangles_share_vertex(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> wp.bool:
    return (
        vec3_equal(a0, b0)
        or vec3_equal(a0, b1)
        or vec3_equal(a0, b2)
        or vec3_equal(a1, b0)
        or vec3_equal(a1, b1)
        or vec3_equal(a1, b2)
        or vec3_equal(a2, b0)
        or vec3_equal(a2, b1)
        or vec3_equal(a2, b2)
    )


@wp.func
def triangle_intersection_segment(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> tuple[wp.bool, wp.vec3, wp.vec3]:
    # Done in float64 on float32 inputs, the same precedent ``predicates.triangles_intersect``
    # already sets one level up (the broad phase that accepts a pair before this narrow phase runs)
    # -- widening is lossless, so this is the same geometry, and what the extra precision buys is
    # the *decisions*. ``line_direction = cross(normal, other_normal)`` is the one that needed it:
    # for two triangles whose planes meet at a small dihedral angle, this cross product subtracts
    # two float32 products that can agree to more digits than float32 carries, underflowing to the
    # exact zero vector even though the true magnitude -- and the float64 broad-phase test that
    # already accepted the pair -- is nonzero, which drops a genuinely crossing pair's segment.
    da0 = kernel_array.to_vec3d(a0)
    da1 = kernel_array.to_vec3d(a1)
    da2 = kernel_array.to_vec3d(a2)
    db0 = kernel_array.to_vec3d(b0)
    db1 = kernel_array.to_vec3d(b1)
    db2 = kernel_array.to_vec3d(b2)

    edge0 = da1 - da0
    edge1 = da2 - da0
    normal = wp.normalize(wp.cross(edge0, edge1))

    other_edge0 = db1 - db0
    other_edge1 = db2 - db0
    other_normal = wp.normalize(wp.cross(other_edge0, other_edge1))

    proj1 = wp.vec3d(
        wp.dot(other_normal, da0 - db0),
        wp.dot(other_normal, da1 - db0),
        wp.dot(other_normal, da2 - db0),
    )
    proj2 = wp.vec3d(
        wp.dot(normal, db0 - da0), wp.dot(normal, db1 - da0), wp.dot(normal, db2 - da0)
    )

    order1_x, order1_y, order1_z = vec3_argsort(proj1)
    proj1a = proj1[order1_x]
    proj1c = proj1[order1_z]
    proj1b = proj1[order1_y]

    va_min = vertex_at(order1_x, da0, da1, da2)
    va_max = vertex_at(order1_z, da0, da1, da2)
    line_origin = (va_min * proj1c - va_max * proj1a) / (proj1c - proj1a)
    line_direction = wp.cross(normal, other_normal)

    edge_direction = vertex_at(
        wp.where(proj1b >= wp.float64(0.0), order1_x, order1_z), da0, da1, da2
    ) - vertex_at(order1_y, da0, da1, da2)
    t2 = intersection_line_coordinate(
        line_origin, line_direction, vertex_at(order1_y, da0, da1, da2), edge_direction
    )

    if t2 > wp.float64(0.0):
        interval = wp.vec2d(wp.float64(0.0), t2)
    else:
        interval = wp.vec2d(t2, wp.float64(0.0))

    order2_x, order2_y, order2_z = vec3_argsort(proj2)
    edge_direction = vertex_at(order2_z, db0, db1, db2) - vertex_at(order2_x, db0, db1, db2)
    s1 = intersection_line_coordinate(
        line_origin, line_direction, vertex_at(order2_x, db0, db1, db2), edge_direction
    )
    edge_direction = vertex_at(
        wp.where(proj2[order2_y] >= wp.float64(0.0), order2_x, order2_z), db0, db1, db2
    ) - vertex_at(order2_y, db0, db1, db2)
    s2 = intersection_line_coordinate(
        line_origin, line_direction, vertex_at(order2_y, db0, db1, db2), edge_direction
    )

    if s1 <= s2:
        other_interval = wp.vec2d(s1, s2)
    else:
        other_interval = wp.vec2d(s2, s1)

    interval = interval_intersect(interval, other_interval)

    p0 = line_origin + interval[0] * line_direction
    p1 = line_origin + interval[1] * line_direction
    return True, kernel_array.to_vec3(p0), kernel_array.to_vec3(p1)


@wp.func
def corners_intersect(
    qa: wp.vec3, qb: wp.vec3, qc: wp.vec3, ta: wp.vec3, tb: wp.vec3, tc: wp.vec3
) -> wp.bool:
    # The narrow phase on two faces' corners: two faces that share a vertex are adjacent, not
    # intersecting, and every other pair is decided by the triangle-triangle test. The one spelling
    # of that rule; ``crossing_slot_flags`` calls it on corners it goes on to reuse.
    if triangles_share_vertex(qa, qb, qc, ta, tb, tc):
        return False
    return triangles_intersect(qa, qb, qc, ta, tb, tc)


@wp.func
def candidate_pair_intersects(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    query_face: wp.int32,
    target_face: wp.int32,
) -> wp.bool:
    # The narrow phase of one broad-phase candidate, by face index. Shared by
    # ``mark_crossing_slot_masks`` and ``validation.self_intersection_marks``.
    qa, qb, qc = kernel_triangles.face_vertices(query_vertices, query_faces, query_face)
    ta, tb, tc = kernel_triangles.face_vertices(target_vertices, target_faces, target_face)
    return corners_intersect(qa, qb, qc, ta, tb, tc)


@wp.kernel
def collect_face_box_candidates(
    mesh_id: wp.uint64,
    face_lower: wp.array[wp.vec3],
    face_upper: wp.array[wp.vec3],
    max_hits: wp.int32,
    out_targets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    """
    Face-box broad phase against a mesh: query face ``f``'s candidates at ``f * max_hits``.

    Shared by ``validation``'s self-intersection test (the query faces are the mesh's own) and the
    two-mesh tests here. One traversal per query face over its bounding box, taking at most
    ``max_hits`` candidates in traversal order (``proximity.mesh_aabb_collect``, the same walk and
    the same cap as ``proximity.query_mesh_aabb_count`` / ``query_mesh_aabb_neighbors``), written
    into a fixed-stride slot block. The fixed stride is what spares the count pass, the scan and the
    host read of the total that a packed candidate list needs, and the second traversal that
    fills it; the unused tail of each block is simply never read.

    The boxes are ``triangles.face_aabb_bounds``' stored buffers, not formed here from the
    corners: the identical walk over the identical candidates measured several times slower with
    the box formed in the kernel.
    """
    f = wp.int32(wp.tid())
    out_counts[f] = mesh_aabb_collect(
        mesh_id, face_lower[f], face_upper[f], max_hits, wp.bool(True), f * max_hits, out_targets
    )


@wp.func
def candidate_slot_query(counts: wp.array[wp.int32], max_hits: wp.int32, s: wp.int32) -> wp.int32:
    # The query face owning slot ``s`` of a fixed-stride candidate table -- ``max_hits`` slots per
    # query face, the first ``counts[q]`` of them live, as ``collect_face_box_candidates`` writes
    # it -- or ``-1`` for a slot past its block's
    # count. A sentinel rather than a boolean, so each caller keeps its own ``< 0`` test.
    q = s // max_hits
    return wp.where(s % max_hits < counts[q], q, wp.int32(-1))


@wp.kernel
def mark_crossing_slot_masks(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    max_hits: wp.int32,
    out_mask_query: wp.array[wp.bool],
    out_mask_target: wp.array[wp.bool],
) -> None:
    # The narrow phase and the marking in one pass over the fixed-stride broad-phase slots: flag
    # both faces of each live candidate that intersects (idempotent ``True`` writes), so a mask
    # needs no per-pair verdict buffer, no compaction and no readback.
    s = wp.int32(wp.tid())
    q = candidate_slot_query(counts, max_hits, s)
    if q < 0:
        return
    t = targets[s]
    if candidate_pair_intersects(query_vertices, query_faces, target_vertices, target_faces, q, t):
        out_mask_query[q] = True
        out_mask_target[t] = True


@wp.kernel
def crossing_slot_flags(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    max_hits: wp.int32,
    with_segment: wp.bool,
    out_flags: wp.array[wp.int32],
) -> None:
    # The narrow-phase verdict of every fixed-stride broad-phase slot, as the ``0`` / ``1``
    # ``int32`` flag ``compact_crossing_slots`` scans in place, so the verdict buffer is also the
    # scan buffer. ``with_segment`` (warp-uniform) additionally rejects a pair whose intersection
    # segment is degenerate, which is ``mesh_with_mesh``'s question; ``mesh_collision_pairs`` asks
    # only whether the faces cross.
    #
    # ``triangle_intersection_segment`` recomputes each pair's normals, edge vectors and
    # plane-distance projections that ``triangles_intersect`` already derived; sharing them would
    # need the segment extraction's ordering and division threaded through the predicate, which
    # the verdict-only callers never want.
    s = wp.int32(wp.tid())
    q = candidate_slot_query(counts, max_hits, s)
    if q < 0:
        out_flags[s] = 0
        return
    qa, qb, qc = kernel_triangles.face_vertices(query_vertices, query_faces, q)
    ta, tb, tc = kernel_triangles.face_vertices(target_vertices, target_faces, targets[s])
    hit = corners_intersect(qa, qb, qc, ta, tb, tc)
    if hit and with_segment:
        valid, p0, p1 = triangle_intersection_segment(qa, qb, qc, ta, tb, tc)
        hit = valid and wp.length(p1 - p0) > TOLERANCE_MERGE_CONSTANT
    out_flags[s] = wp.where(hit, 1, 0)


@wp.kernel
def compact_crossing_slots(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    max_hits: wp.int32,
    inclusive: wp.array[wp.int32],
    swapped: wp.bool,
    out_pairs: wp.array2d[wp.int32],
    out_segments: wp.array2d[wp.vec3],
) -> None:
    # One thread per kept slot, ``k``-th in slot order -- which is candidate order, query face by
    # query face -- found in the in-place inclusive scan of ``crossing_slot_flags``' flags as the
    # first slot whose running total reaches ``k + 1``. Launched over the kept count rather than
    # over every slot, since the kept slots are a small, scattered fraction of the table. Exactly
    # one of the two outputs is live; the other is ``None``, whose shape reads 0.
    #
    # ``out_pairs`` rows are in the caller's ``(a, b)`` column order: the broad phase queries the
    # *larger* mesh's faces against the smaller one's BVH, so ``swapped`` says the query is mesh b
    # and puts the columns back. ``out_segments`` recomputes each kept slot's segment, the one
    # ``crossing_slot_flags`` tested, rather than reading it from a per-slot buffer: that buffer is
    # ``max_hits`` segments per query face, and allocating it measured dearer than the recompute.
    k = wp.int32(wp.tid())
    i = wp.lower_bound(inclusive, k + 1)
    q = i // max_hits
    t = targets[i]
    if out_pairs.shape[0] > 0:
        out_pairs[k, 0] = wp.where(swapped, t, q)
        out_pairs[k, 1] = wp.where(swapped, q, t)
    if out_segments.shape[0] > 0:
        qa, qb, qc = kernel_triangles.face_vertices(query_vertices, query_faces, q)
        ta, tb, tc = kernel_triangles.face_vertices(target_vertices, target_faces, t)
        _valid, p0, p1 = triangle_intersection_segment(qa, qb, qc, ta, tb, tc)
        out_segments[k, 0] = p0
        out_segments[k, 1] = p1


@wp.func
def edge_level_crossing(
    origin: wp.vec3, dest: wp.vec3, value_origin: wp.float32, value_dest: wp.float32
) -> wp.vec3:
    # Where the field crosses zero along one edge, from the two endpoint values. For a plane's
    # signed distance this is ``edge_plane_intersection``'s algebra with the dot products already
    # taken: ``dot(o - a, n) / dot(b - a, n)`` is ``va / (va - vb)``.
    denominator = value_origin - value_dest
    if denominator == wp.float32(0.0):
        denominator = EDGE_DENOM_EPSILON
    return origin + (dest - origin) * (value_origin / denominator)


# How a face meets the level set. The three kept classes are numbered in the order the fused
# compaction below lays them out, so a class is also its block index plus one. ``ON_PLANE`` is a
# provisional answer that ``resolve_on_plane_faces`` turns into ``INSIDE`` or ``DROP``.
SLICE_CLASS_DROP = wp.constant(wp.int32(0))
SLICE_CLASS_INSIDE = wp.constant(wp.int32(1))
SLICE_CLASS_CUT_QUAD = wp.constant(wp.int32(2))
SLICE_CLASS_CUT_TRI = wp.constant(wp.int32(3))
SLICE_CLASS_ON_PLANE = wp.constant(wp.int32(4))

# Kept classes, i.e. the number of blocks the flag buffer carries.
SLICE_CLASSES = 3


@wp.func
def face_level_set_signs(
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    f: wp.int32,
    out_signs: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32]:
    # One face's three level-set signs, stored, plus the two sums every classifier decides on.
    #
    # ``SLICE_SIGN_INSIDE`` (-1) is the ``>= isovalue`` side, so the tolerance sign is negated and a
    # value on the level set counts as positive. ``signs_sum`` separates "all one side" from "cut"
    # and ``signs_asum`` counts the corners strictly off the plane; between them they name every
    # case both classifiers below distinguish.
    #
    # Shared by ``classify_faces_for_slice`` and ``classify_faces_for_split``, which differ only in
    # the 3-way vs 4-way decision they map these two sums onto -- that is the genuine variation, and
    # the nine statements above it were a copy. Kept here rather than in ``kernels/predicates.py``
    # because it writes an output array: that module holds pure predicates, and both callers of this
    # one are in this file.
    i0, i1, i2 = kernel_triangles.corner_triple(faces, f)
    s0 = -kernel_array.sign_with_tolerance(vertex_dots[i0], TOLERANCE_MERGE_CONSTANT)
    s1 = -kernel_array.sign_with_tolerance(vertex_dots[i1], TOLERANCE_MERGE_CONSTANT)
    s2 = -kernel_array.sign_with_tolerance(vertex_dots[i2], TOLERANCE_MERGE_CONSTANT)
    out_signs[f, 0] = s0
    out_signs[f, 1] = s1
    out_signs[f, 2] = s2
    return s0 + s1 + s2, wp.abs(s0) + wp.abs(s1) + wp.abs(s2)


@wp.func
def write_class_flags(
    face_class: wp.int32,
    f: wp.int32,
    n_faces: wp.int32,
    n_classes: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # Selection flags for every kept class at once, as ``n_classes`` blocks of ``n_faces`` in one
    # buffer. Scanning that buffer once compacts all the classes into one index array with the
    # blocks contiguous, so the whole partition costs one scan and one host readback rather than one
    # of each per class. Class ``c`` maps to block ``c - 1``; class 0 is "not selected". Every face
    # writes all of its slots, which is what keeps the buffer from needing a memset first.
    #
    # Called from the classifiers themselves rather than from a pass of its own, so a class and
    # its flags come out of one launch -- the flags read nothing but the class the same thread just
    # decided. ``n_classes`` is an argument so the clip's three kept classes and the split's four
    # share it.
    for block in range(n_classes):
        out_flags[block * n_faces + f] = wp.where(face_class == block + 1, wp.int32(1), wp.int32(0))


@wp.kernel
def classify_faces_for_slice(
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_classes: wp.array[wp.int32],
    out_signs: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    signs_sum, signs_asum = face_level_set_signs(faces, vertex_dots, f, out_signs)

    # A face lying in the plane also satisfies the "wholly inside" test (both sums are zero), so
    # it has to be tested first -- its side is decided from its normal, not from its vertices.
    face_class = SLICE_CLASS_DROP
    if signs_asum == wp.int32(0):
        face_class = SLICE_CLASS_ON_PLANE
    elif signs_sum == -signs_asum:
        face_class = SLICE_CLASS_INSIDE
    elif signs_asum >= wp.int32(2) and wp.abs(signs_sum) <= wp.int32(1):
        if signs_sum < wp.int32(0):
            face_class = SLICE_CLASS_CUT_QUAD
        else:
            face_class = SLICE_CLASS_CUT_TRI
    out_classes[f] = face_class
    write_class_flags(face_class, f, faces.shape[0] // 3, SLICE_CLASSES, out_flags)


@wp.func
def on_plane_face_side(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], plane_normal: wp.vec3, f: wp.int32
) -> tuple[wp.bool, wp.bool]:
    # A face lying in the plane has no vertex-based tie-break, so its own normal decides which side
    # it belongs to -- opposing the plane's normal (``dot < 0``) is the side both
    # ``resolve_on_plane_faces`` and ``label_faces_by_plane_side`` keep. The second return flags a
    # degenerate (zero-area) face, which has no usable normal at all; each caller maps that onto its
    # own answer rather than sharing one here, because the two operations disagree on it for a good
    # reason -- a clip has a side to drop the face into, a split does not.
    normal, area = kernel_triangles.face_normals_and_area(vertices, faces, f)
    is_below = wp.dot(normal, plane_normal) < wp.float32(0.0)
    return is_below, area <= TOLERANCE_ZERO_CONSTANT


@wp.kernel
def resolve_on_plane_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    out_classes: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Runs after ``classify_faces_for_slice`` has written the classes *and* their flags, so a face
    # it moves has to move its flag too. ``ON_PLANE`` is not a kept class and carries no flag, and
    # ``DROP`` carries none either, so only a face resolved to ``INSIDE`` gains one -- in block 0,
    # at ``f``, which is what ``write_class_flags`` would have written for it.
    f = wp.int32(wp.tid())
    if out_classes[f] != SLICE_CLASS_ON_PLANE:
        return
    is_below, degenerate = on_plane_face_side(vertices, faces, plane_normal, f)
    if not degenerate and is_below:
        out_classes[f] = SLICE_CLASS_INSIDE
        out_flags[f] = wp.int32(1)
    else:
        out_classes[f] = SLICE_CLASS_DROP


@wp.kernel
def scatter_slice_class(
    inclusive: wp.array[wp.int32],
    n_faces: wp.int32,
    out_indices: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # A flag-driven compaction over the blocked flag buffer, scanned in place, that writes the
    # *face* index rather than the flag index, so each block comes out addressing faces. The last
    # thread of each block also writes that block's count -- its running total minus the previous
    # block's -- so the host reads the counts once, after this launch, and no pass of its own
    # derives them.
    t = wp.int32(wp.tid())
    slot, flag = kernel_array.scanned_count(inclusive, t)
    if flag != 0:
        out_indices[slot] = t % n_faces
    if (t + 1) % n_faces == 0:
        previous = wp.int32(0)
        if t >= n_faces:
            previous = inclusive[t - n_faces]
        out_counts[t // n_faces] = inclusive[t] - previous


@wp.func
def canonical_edge_crossing(
    vertices: wp.array[wp.vec3], values: wp.array[wp.float32], i: wp.int32, j: wp.int32
) -> wp.vec3:
    # Always interpolate from the lower-numbered endpoint, so the two faces sharing a cut edge
    # evaluate the identical expression and land on **bitwise equal** points. That is what lets
    # ``clip_mesh_with_field(cap=True)`` weld the section rim exactly rather than by tolerance.
    a = wp.min(i, j)
    b = wp.max(i, j)
    return edge_level_crossing(vertices[a], vertices[b], values[a], values[b])


@wp.func
def face_edge_crossing(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    face_index: wp.int32,
    edge: wp.int32,
) -> wp.vec3:
    # Where the level set crosses edge ``edge`` of a face, from the per-vertex field the classifier
    # already signed -- so the plane case reuses those dot products rather than recomputing them.
    # Edge ``e`` joins corner ``e`` to corner ``e + 1``, which is the numbering the two emit
    # kernels' ``edge_a`` / ``edge_b`` are expressed in.
    base = face_index * wp.int32(3)
    return canonical_edge_crossing(
        vertices, vertex_values, faces[base + edge], faces[base + (edge + wp.int32(1)) % 3]
    )


@wp.func
def cut_face_context(
    face_indices: wp.array[wp.int32], face_signs: wp.array2d[wp.int32], tid: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32, wp.int32]:
    # The cut face work item ``tid`` owns: its face index, the base of its corner triple in the
    # flat face buffer, and its three corner signs.
    #
    # The prologue all four emit kernels below open with -- they differ in *which* corner the signs
    # single out and in what they write, never in how they find the face. Shared rather than
    # repeated so ``face_signs``' row layout and the ``3 * f`` corner convention have one
    # statement each. Three of the four compile to byte-identical SASS and
    # ``emit_split_cut_corner`` to 8 instructions fewer, so it is free or better.
    face_index = face_indices[tid]
    s0, s1, s2 = kernel_triangles.row_triple(face_signs, face_index)
    return face_index, face_index * wp.int32(3), s0, s1, s2


@wp.func
def emit_cut_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    face_index: wp.int32,
    cut: wp.int32,
    edge_0: wp.int32,
    edge_1: wp.int32,
    vertex_base: wp.int32,
    out_new_verts: wp.array[wp.vec3],
) -> tuple[wp.int32, wp.int32]:
    # Write the two new vertices cut face ``cut`` contributes, and return their indices in the
    # *concatenated* buffer, where the new block starts at ``vertex_base``.
    #
    # Every cut face emits exactly two, whichever side is alone in sign: a quad cut splits into two
    # triangles and a corner cut into one, but both are bounded by the same two edge crossings. So
    # the slot is a function of the thread alone and neither kernel needs a counter -- which is the
    # invariant this function exists to state, and the reason ``2 * cut`` is computed once rather
    # than seventeen times across the two callers.
    #
    # The crossings are computed here rather than read from a table a previous launch filled. That
    # table cost a launch, an ``(n_cut, 3)`` ``vec3`` allocation and a full round trip of it
    # through global memory -- and it computed **three** crossings per cut face where every caller
    # of this function uses exactly two, so the fused form does less arithmetic as well as less
    # traffic. ``canonical_edge_crossing`` interpolates from the lower-numbered endpoint, so a
    # crossing is bitwise identical however many times and from whichever face it is evaluated;
    # that is what keeps the rim weldable and what makes recomputing it here free of consequence.
    #
    # Measured, output byte-identical (including ``cap=True``, which welds the rim and so depends
    # on that bitwise equality): 1.09x on ``clip_mesh_with_field``, 1.11x on
    # ``slice_mesh_with_plane`` and 1.03x on ``split_mesh_with_plane``, two launches and two
    # allocations fewer. Predicted from the launch count alone it looked like 2 %; the extra came
    # from the arithmetic, since the tabulating pass computed three crossings per cut face to be
    # read for two.
    slot = wp.int32(2) * cut
    out_new_verts[slot] = face_edge_crossing(vertices, faces, vertex_values, face_index, edge_0)
    out_new_verts[slot + wp.int32(1)] = face_edge_crossing(
        vertices, faces, vertex_values, face_index, edge_1
    )
    return vertex_base + slot, vertex_base + slot + wp.int32(1)


@wp.kernel
def emit_quad_cut(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    vertex_values: wp.array[wp.float32],
    vertex_base: wp.int32,
    out_new_verts: wp.array[wp.vec3],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    face_index, base, s0, s1, s2 = cut_face_context(face_indices, face_signs, tid)
    outside = find_corner_with_sign(s0, s1, s2, SLICE_SIGN_OUTSIDE)
    inside_a = (outside + wp.int32(1)) % wp.int32(3)
    inside_b = (outside + wp.int32(2)) % wp.int32(3)
    v_a = faces[base + inside_a]
    v_b = faces[base + inside_b]
    edge_a = (outside + wp.int32(2)) % wp.int32(3)
    edge_b = outside
    new_i0, new_i1 = emit_cut_vertices(
        vertices, faces, vertex_values, face_index, tid, edge_a, edge_b, vertex_base, out_new_verts
    )
    # The quad becomes two triangles, in the same pair of rows the two vertices went into.
    row = wp.int32(2) * tid
    out_new_faces[row, 0] = v_a
    out_new_faces[row, 1] = v_b
    out_new_faces[row, 2] = new_i0
    out_new_faces[row + wp.int32(1), 0] = new_i0
    out_new_faces[row + wp.int32(1), 1] = new_i1
    out_new_faces[row + wp.int32(1), 2] = v_a


@wp.kernel
def emit_tri_cut(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    vertex_values: wp.array[wp.float32],
    vertex_base: wp.int32,
    out_new_verts: wp.array[wp.vec3],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    face_index, base, s0, s1, s2 = cut_face_context(face_indices, face_signs, tid)
    inside = find_corner_with_sign(s0, s1, s2, SLICE_SIGN_INSIDE)
    v_inside = faces[base + inside]
    corner_a = (inside + wp.int32(1)) % wp.int32(3)
    corner_b = (inside + wp.int32(2)) % wp.int32(3)
    edge_0 = inside
    edge_1 = corner_b
    new_i0, new_i1 = emit_cut_vertices(
        vertices, faces, vertex_values, face_index, tid, edge_0, edge_1, vertex_base, out_new_verts
    )
    # The corner stays one triangle, so this kernel writes one face row per cut, not two.
    out_new_faces[tid, 0] = v_inside
    # ``classify_faces_for_slice`` routes both a genuine two-crossing cut (both neighbours
    # strictly outside) and "one neighbour sits exactly on the level set" into this same class --
    # they share ``signs_asum == 2`` and cannot be told apart there. An on-plane neighbour is not a
    # real crossing: reuse its existing vertex directly rather than interpolating a near-duplicate a
    # hair's breadth away from it. Whichever of ``new_i0`` / ``new_i1`` is unused in that case is
    # simply left unreferenced -- the caller's ``remove_unreferenced_vertices`` sweeps it up, so no
    # buffer accounting changes.
    if face_signs[face_index, corner_a] == SLICE_SIGN_ON_PLANE:
        out_new_faces[tid, 1] = faces[base + corner_a]
        out_new_faces[tid, 2] = new_i1
    elif face_signs[face_index, corner_b] == SLICE_SIGN_ON_PLANE:
        out_new_faces[tid, 1] = new_i0
        out_new_faces[tid, 2] = faces[base + corner_b]
    else:
        out_new_faces[tid, 1] = new_i0
        out_new_faces[tid, 2] = new_i1


# Face classes for the both-sides split. Numbered from 1 contiguously because
# ``write_class_flags`` maps class ``c`` to block ``c - 1``; class 0 is "not selected", which this
# taxonomy never needs since a split keeps every face.
SPLIT_CLASS_POSITIVE = wp.constant(wp.int32(1))
SPLIT_CLASS_NEGATIVE = wp.constant(wp.int32(2))
SPLIT_CLASS_CUT_EDGES = wp.constant(wp.int32(3))
SPLIT_CLASS_CUT_CORNER = wp.constant(wp.int32(4))
SPLIT_CLASSES = 4


@wp.kernel
def classify_faces_for_split(
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_classes: wp.array[wp.int32],
    out_signs: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Four classes rather than the clip's three, because a split keeps both sides and so has to
    # tell the two *uncut* sides apart -- and because a face with one corner exactly on the level
    # set splits into two triangles, not three. The sign convention is ``face_level_set_signs``'.
    f = wp.int32(wp.tid())
    signs_sum, signs_asum = face_level_set_signs(faces, vertex_dots, f, out_signs)
    face_class = SPLIT_CLASS_CUT_CORNER
    if signs_sum == -signs_asum:
        # Every corner on the positive side, or all three exactly on the level set -- which counts
        # as positive, so the face is kept whole rather than cut along itself.
        face_class = SPLIT_CLASS_POSITIVE
    elif signs_sum == signs_asum:
        face_class = SPLIT_CLASS_NEGATIVE
    elif signs_asum == wp.int32(3):
        face_class = SPLIT_CLASS_CUT_EDGES
    # Otherwise one corner is exactly on the level set and the other two on opposite sides: a
    # single edge crossing, joined to that corner -- the ``CUT_CORNER`` default above.
    out_classes[f] = face_class
    write_class_flags(face_class, f, faces.shape[0] // 3, SPLIT_CLASSES, out_flags)


@wp.func
def emit_split_uncut_face(
    faces: wp.array[wp.int32],
    uncut_indices: wp.array[wp.int32],
    n_positive: wp.int32,
    k: wp.int32,
    out_new_faces: wp.array2d[wp.int32],
    out_positive: wp.array[wp.bool],
) -> None:
    # A face no cut touches, copied through with its side label. ``uncut_indices`` is the
    # partition's index buffer, whose first two blocks are the positive and then the negative
    # faces, so row ``k`` belongs to the positive side exactly when ``k < n_positive``.
    i0, i1, i2 = kernel_triangles.corner_triple(faces, uncut_indices[k])
    out_new_faces[k, 0] = i0
    out_new_faces[k, 1] = i1
    out_new_faces[k, 2] = i2
    out_positive[k] = k < n_positive


@wp.func
def unique_edge_order_key(a: wp.int64, b: wp.int64, base: wp.int64) -> wp.int64:
    # The undirected edge key in ``edges.edges_unique``' row order: that packs a sorted row with
    # its first column as the low digit, so the larger endpoint is the high one.
    return wp.max(a, b) * base + wp.min(a, b)


@wp.kernel
def split_crossed_halfedge_keys(
    faces: wp.array[wp.int32],
    cut_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    key_base: wp.int64,
    out_keys: wp.array[wp.int64],
    out_halfedges: wp.array[wp.int32],
) -> None:
    # One thread per halfedge of a cut face: the undirected key of the edge when the level set
    # crosses its interior (strictly opposite corner signs, the classifier's own tolerance), else
    # ``key_base ** 2``, above every edge key. Written straight into a radix sort's double-width
    # buffer with the cut-local halfedge ``3 j + k`` as payload, so equal keys -- the two faces
    # sharing a crossed edge -- land next to each other, in the order a mesh-wide unique-edge
    # table lists the edges.
    j, k = wp.tid()
    face_index = cut_indices[j]
    a = faces[3 * face_index + k]
    b = faces[3 * face_index + (k + 1) % 3]
    key = key_base * key_base
    if face_signs[face_index, k] * face_signs[face_index, (k + 1) % 3] < 0:
        key = unique_edge_order_key(wp.int64(a), wp.int64(b), key_base)
    out_keys[3 * j + k] = key
    out_halfedges[3 * j + k] = 3 * j + k


@wp.kernel
def split_crossed_edge_flags(
    keys: wp.array[wp.int64], key_base: wp.int64, out_flags: wp.array[wp.int32]
) -> None:
    # ``1`` at the first of each run of equal crossed-edge keys in the sorted halfedges: one new
    # vertex per crossed edge, numbered in ascending edge order by the inclusive scan of these.
    q = wp.int32(wp.tid())
    first = csr_run_start(keys, q, key_base * key_base)
    out_flags[q] = wp.where(first, wp.int32(1), wp.int32(0))


@wp.kernel
def split_crossed_edge_vertices(
    vertices: wp.array[wp.vec3],
    vertex_dots: wp.array[wp.float32],
    keys: wp.array[wp.int64],
    halfedges: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    key_base: wp.int64,
    vertex_base: wp.int32,
    out_cut_vertices: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # Give every crossed cut-face halfedge its edge's new vertex, and have the first halfedge of
    # each edge write the crossing point. ``canonical_edge_crossing`` interpolates from the lower
    # endpoint, so both faces' views of the edge agree. ``out_points`` is the whole output vertex
    # buffer: the input's ``vertex_base`` positions, copied by the first threads of this launch (it
    # runs over ``max(n_halfedges, vertex_base)``), then the crossings.
    q = wp.int32(wp.tid())
    if q < vertex_base:
        out_points[q] = vertices[q]
    if q >= rank.shape[0]:
        return
    key = keys[q]
    if key >= key_base * key_base:
        return
    slot, first = kernel_array.scanned_count(rank, q)
    slot = slot + first - 1
    out_cut_vertices[halfedges[q]] = vertex_base + slot
    if first != 0:
        high = wp.int32(key // key_base)
        low = wp.int32(key % key_base)
        out_points[vertex_base + slot] = canonical_edge_crossing(vertices, vertex_dots, low, high)


@wp.func
def emit_split_cut_edges(
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    cut_vertices: wp.array[wp.int32],
    tid: wp.int32,
    cut: wp.int32,
    row: wp.int32,
    out_new_faces: wp.array2d[wp.int32],
    out_positive: wp.array[wp.bool],
) -> None:
    # The generic cut: the level set enters through one edge and leaves through another, so the face
    # becomes a corner triangle plus a quad -- three triangles sharing the two crossing vertices.
    # Windings match ``emit_tri_cut`` and ``emit_quad_cut``, which emit these same triangles one
    # side at a time; the difference is that both sides are kept here. ``tid`` is the face's slot
    # in the partition, ``cut`` its rank among the cut faces and ``row`` its first output row.
    face_index, base, s0, s1, s2 = cut_face_context(face_indices, face_signs, tid)
    lone = find_unique_sign_vertex(s0, s1, s2)
    next_corner = (lone + wp.int32(1)) % wp.int32(3)
    last_corner = (lone + wp.int32(2)) % wp.int32(3)
    # ``p0`` on the edge leaving the lone corner, ``p1`` on the edge arriving at it.
    # ``cut_vertices`` holds the new vertex on each of this cut face's edges, edge ``k`` joining
    # corners ``k`` and ``k + 1``.
    p0 = cut_vertices[3 * cut + lone]
    p1 = cut_vertices[3 * cut + last_corner]
    v_next = faces[base + next_corner]
    v_last = faces[base + last_corner]

    out_new_faces[row, 0] = faces[base + lone]
    out_new_faces[row, 1] = p0
    out_new_faces[row, 2] = p1
    out_new_faces[row + wp.int32(1), 0] = v_next
    out_new_faces[row + wp.int32(1), 1] = v_last
    out_new_faces[row + wp.int32(1), 2] = p1
    out_new_faces[row + wp.int32(2), 0] = p1
    out_new_faces[row + wp.int32(2), 1] = p0
    out_new_faces[row + wp.int32(2), 2] = v_next

    lone_is_positive = face_signs[face_index, lone] == SLICE_SIGN_INSIDE
    out_positive[row] = lone_is_positive
    out_positive[row + wp.int32(1)] = not lone_is_positive
    out_positive[row + wp.int32(2)] = not lone_is_positive


@wp.func
def emit_split_cut_corner(
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    cut_vertices: wp.array[wp.int32],
    tid: wp.int32,
    cut: wp.int32,
    row: wp.int32,
    out_new_faces: wp.array2d[wp.int32],
    out_positive: wp.array[wp.bool],
) -> None:
    # One corner sits exactly on the level set, so the cut runs from it to the single crossing on
    # the opposite edge: two triangles, no quad. Emitting three the other kernel's way would put a
    # zero-area sliver in the output, which is the only reason this class exists separately.
    face_index, base, s0, s1, s2 = cut_face_context(face_indices, face_signs, tid)
    # Exactly one of the three corners is on the level set here, which is the precondition
    # ``find_corner_with_sign`` states: it tests corners 0 and 1 and falls through to 2, which
    # agrees with the by-elimination chain this replaces on every input that satisfies it. One
    # spelling of "which corner carries this sign" in the file, not two that can drift apart.
    on_level = find_corner_with_sign(s0, s1, s2, SLICE_SIGN_ON_PLANE)
    next_corner = (on_level + wp.int32(1)) % wp.int32(3)
    last_corner = (on_level + wp.int32(2)) % wp.int32(3)
    crossing = cut_vertices[3 * cut + next_corner]

    out_new_faces[row, 0] = faces[base + on_level]
    out_new_faces[row, 1] = faces[base + next_corner]
    out_new_faces[row, 2] = crossing
    out_new_faces[row + wp.int32(1), 0] = faces[base + on_level]
    out_new_faces[row + wp.int32(1), 1] = crossing
    out_new_faces[row + wp.int32(1), 2] = faces[base + last_corner]
    out_positive[row] = face_signs[face_index, next_corner] == SLICE_SIGN_INSIDE
    out_positive[row + wp.int32(1)] = face_signs[face_index, last_corner] == SLICE_SIGN_INSIDE


@wp.kernel
def emit_split_faces(
    faces: wp.array[wp.int32],
    class_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    cut_vertices: wp.array[wp.int32],
    n_positive: wp.int32,
    n_uncut: wp.int32,
    n_edges: wp.int32,
    out_new_faces: wp.array2d[wp.int32],
    out_positive: wp.array[wp.bool],
) -> None:
    # Every face's output rows in one launch, one thread per partition slot. The partition lists
    # the uncut faces (positive, then negative), then the edge cuts, then the corner cuts, and the
    # output rows follow the same order -- one row per uncut face, three per edge cut, two per
    # corner cut -- so each class's rows start where the previous class's end, and cut face ``j``'s
    # crossing vertices sit at ``3 j``. Replaces one launch per class and the views they took.
    k = wp.int32(wp.tid())
    if k < n_uncut:
        emit_split_uncut_face(faces, class_indices, n_positive, k, out_new_faces, out_positive)
        return
    cut = k - n_uncut
    if cut < n_edges:
        emit_split_cut_edges(
            faces,
            class_indices,
            face_signs,
            cut_vertices,
            k,
            cut,
            n_uncut + 3 * cut,
            out_new_faces,
            out_positive,
        )
        return
    emit_split_cut_corner(
        faces,
        class_indices,
        face_signs,
        cut_vertices,
        k,
        cut,
        n_uncut + 3 * n_edges + 2 * (cut - n_edges),
        out_new_faces,
        out_positive,
    )


@wp.kernel
def plane_crossed_edge_mask(
    unique_edges: wp.array2d[wp.int32],
    vertex_dots: wp.array[wp.float32],
    tolerance: wp.float32,
    out_crossed: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # An edge needs a new vertex only when the plane passes through its *interior*: an endpoint
    # already in the plane (within ``tolerance``) serves as the crossing itself, so splitting there
    # would emit a duplicate. Strict opposite signs is therefore the condition, and it is also why
    # at most two of a triangle's three edges can ever be flagged, for one of two reasons depending
    # on the sign triple: a zero-valued corner caps its two incident edges at one crossing between
    # them by itself (whichever of its two neighbours differs from it takes the one crossing; the
    # other neighbour either agrees with it, or the zero blocks that edge's test outright), and
    # with no zero present there are only two sign values among three corners, so a third crossing
    # would need all three edges to alternate sign, which three values pulled from a two-value set
    # cannot do. Either way ``emit_size_faces``' 3-split branch is unreachable from here.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    sign_a = kernel_array.sign_with_tolerance(vertex_dots[a], tolerance)
    sign_b = kernel_array.sign_with_tolerance(vertex_dots[b], tolerance)
    crossed = sign_a * sign_b < wp.int32(0)
    # ``out_crossed`` is for a caller that needs the mask itself (``split_edges`` takes one); one
    # that does not passes ``None``, whose shape reads 0. ``out_flags`` is the same verdict as the
    # ``0`` / ``1`` count the caller scans for each crossing's slot, so it needs no conversion pass
    # before the scan, and it is what ``plane_edge_crossing_points`` reads.
    if out_crossed.shape[0] > 0:
        out_crossed[e] = crossed
    out_flags[e] = wp.where(crossed, wp.int32(1), wp.int32(0))


@wp.kernel
def plane_edge_crossing_points(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    crossed_flags: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_points: wp.array[wp.vec3],
) -> None:
    # ``remesh.split_child_counts_and_midpoints``' midpoint with the level-set crossing in its
    # place. One point per *unique edge* rather than per cut face, which is what makes the split
    # crack-free where ``_clip_with_vertex_field`` is cracked: the two faces sharing the edge
    # address the same new vertex, where a per-face crossing would leave two coincident copies and
    # a seam of loose edges (which is exactly why ``clip_mesh_with_field(cap=True)`` has to weld
    # before it can fill).
    #
    # Launched over every unique edge and gated on the mask, writing each crossing at the slot the
    # exclusive scan of that mask gave it -- so neither split needs the crossed edges' own index
    # list, which would be a second compaction of the mask the scan already compacted.
    e = wp.int32(wp.tid())
    if crossed_flags[e] != 0:
        out_points[offsets[e]] = canonical_edge_crossing(
            vertices, vertex_dots, unique_edges[e, 0], unique_edges[e, 1]
        )


@wp.kernel
def label_faces_by_plane_side(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    tolerance: wp.float32,
    out_above: wp.array[wp.bool],
) -> None:
    # After the split no face straddles the plane, so the *largest-magnitude* vertex dot decides the
    # side for the whole face. Reading the extremum rather than a sum or a centroid keeps a sliver
    # face -- two crossing vertices at dot 0 and one real vertex just off the plane -- on the side
    # its real vertex is on, where a centroid would divide the offset by three and a sum would let
    # two rounding-level zeros outvote it. The dots are taken here, per corner, rather than read
    # from a per-vertex buffer: the appended crossing points sit on the plane by construction, so a
    # buffer over the grown vertex array would be one more map and allocation for the same values.
    f = wp.int32(wp.tid())
    extreme = wp.float32(0.0)
    for corner in range(3):
        value = point_plane_dot(vertices[faces[f * 3 + corner]], plane_normal, plane_origin)
        if wp.abs(value) > wp.abs(extreme):
            extreme = value

    side = kernel_array.sign_with_tolerance(extreme, tolerance)
    if side != wp.int32(0):
        out_above[f] = side > wp.int32(0)
        return

    # The face lies *in* the plane, so its vertices give no answer. Decide from its own normal, the
    # same tie-break ``resolve_on_plane_faces`` applies (``on_plane_face_side``), so that the
    # ``above`` block of this split holds exactly the faces ``slice_mesh_with_plane`` keeps. Unlike
    # the clip, a split has nowhere to drop a degenerate face, so it goes to ``above`` by
    # convention rather than being excluded.
    is_below, degenerate = on_plane_face_side(vertices, faces, plane_normal, f)
    out_above[f] = degenerate or is_below


@wp.func
def crossing_point(
    value_from: wp.Float, value_to: wp.Float, point_from: wp.vec3, point_to: wp.vec3
) -> wp.vec3:
    # Linear crossing of the zero level set along one edge. The caller only passes edges whose
    # endpoints straddle zero with the ``>= 0`` convention below, so the denominator is never zero.
    t = wp.float32(value_from / (value_from - value_to))
    return point_from + t * (point_to - point_from)


@wp.func
def edge_key(a: wp.int64, b: wp.int64, base: wp.int64) -> wp.int64:
    # Order-independent key of the undirected edge ``(a, b)``: injective for vertex indices below
    # ``base``, which is what lets two faces sharing a crossing agree on it without a unique-edge
    # table. See ``marching_triangles_segments`` for what it replaced.
    return wp.min(a, b) * base + wp.max(a, b)


@wp.kernel
def marching_triangles_segments(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.Float],
    isovalue: wp.Float,
    key_base: wp.int64,
    out_cut: wp.array[wp.int32],
    out_segments: wp.array2d[wp.vec3],
    out_edges: wp.array2d[wp.int64],
) -> None:
    # One thread per face. Each value is re-zeroed at ``isovalue`` in the field's own precision as
    # it is read, and a value of exactly zero counts as positive, so every cut face has exactly one
    # vertex alone in sign and yields exactly one segment: the two edges incident to that vertex
    # are the crossed ones.
    f = wp.int32(wp.tid())
    i0, i1, i2 = kernel_triangles.corner_triple(faces, f)
    d0 = values[i0] - isovalue
    d1 = values[i1] - isovalue
    d2 = values[i2] - isovalue

    # A NaN field value (an unreachable vertex in a heat-distance field, say) must be rejected
    # before the sign test below, not after: ``NaN >= 0.0`` is ``False`` under IEEE-754, so it would
    # otherwise land in the same bucket as a genuine negative value, pass as a "lone corner" against
    # two real opposite-signed neighbours, and feed ``crossing_point`` a ``NaN`` that reaches the
    # returned curve with no filter anywhere downstream (unlike ``mesh_with_mesh``, whose
    # ``crossing_slot_flags`` rejects a degenerate segment before it is kept).
    if wp.isnan(d0) or wp.isnan(d1) or wp.isnan(d2):
        out_cut[f] = 0
        return

    zero = values.dtype(0.0)
    p0 = wp.where(d0 >= zero, wp.int32(1), wp.int32(0))
    p1 = wp.where(d1 >= zero, wp.int32(1), wp.int32(0))
    p2 = wp.where(d2 >= zero, wp.int32(1), wp.int32(0))

    if p0 == p1 and p1 == p2:
        out_cut[f] = 0
        return

    # Local index of the vertex whose sign differs from the other two.
    lone = find_unique_sign_vertex(p0, p1, p2)
    next_index = (lone + wp.int32(1)) % wp.int32(3)
    prev_index = (lone + wp.int32(2)) % wp.int32(3)

    # The corners and their re-zeroed values, indexed by local corner: already in registers.
    corners = wp.vec3i(i0, i1, i2)
    shifted = wp.vector(d0, d1, d2)
    vertex_lone = corners[lone]
    vertex_next = corners[next_index]
    vertex_prev = corners[prev_index]
    value_lone = shifted[lone]

    point_next = crossing_point(
        value_lone, shifted[next_index], vertices[vertex_lone], vertices[vertex_next]
    )
    point_prev = crossing_point(
        value_lone, shifted[prev_index], vertices[vertex_lone], vertices[vertex_prev]
    )
    # Halfedge ``3f + k`` spans local corners ``k`` and ``k + 1``, so the edge from ``lone`` to the
    # next corner joins ``vertex_lone`` to ``vertex_next``, and the edge from the previous corner to
    # ``lone`` joins ``vertex_prev`` to ``vertex_lone``. A key that stitches segments into curves
    # only has to be *equal for the two faces sharing a crossing and distinct otherwise*, and the
    # sorted vertex pair already is -- so it is computed here rather than looked up in a dense
    # unique-edge table.
    # A dense unique-edge table would densify *every* mesh edge where a level set crosses a small
    # fraction of them; the densification ``_link_segments`` genuinely needs happens there instead,
    # over the crossing endpoints alone.
    #
    # ``key_base`` is the vertex count, so the pair packs without collision; the caller resolves it
    # from its own ``n_vertices`` argument.
    edge_next = edge_key(wp.int64(vertex_lone), wp.int64(vertex_next), key_base)
    edge_prev = edge_key(wp.int64(vertex_prev), wp.int64(vertex_lone), key_base)

    # Orient the segment so the region where the field exceeds the isovalue lies to its left, with
    # the face normal as up. That makes the crossing shared by two faces an outgoing endpoint of one
    # and an incoming endpoint of the other, which is what lets the curves be linked at all.
    out_cut[f] = 1
    if value_lone >= zero:
        out_segments[f, 0] = point_next
        out_segments[f, 1] = point_prev
        out_edges[f, 0] = edge_next
        out_edges[f, 1] = edge_prev
    else:
        out_segments[f, 0] = point_prev
        out_segments[f, 1] = point_next
        out_edges[f, 0] = edge_prev
        out_edges[f, 1] = edge_next


@wp.func
def copy_cut_segment(
    inclusive: wp.array[wp.int32],
    segments: wp.array2d[wp.vec3],
    i: wp.int32,
    out_segments: wp.array2d[wp.vec3],
) -> wp.int32:
    # Copy row ``i`` of a per-face segment buffer to its compacted slot and return that slot, or
    # ``-1`` when the row is not cut. Shared by the two compactions below, which differ only in how
    # they hand on ``marching_triangles_segments``' crossing keys.
    slot, cut = kernel_array.scanned_count(inclusive, i)
    if cut == 0:
        return wp.int32(-1)
    out_segments[slot, 0] = segments[i, 0]
    out_segments[slot, 1] = segments[i, 1]
    return slot


@wp.kernel
def compact_cut_segments(
    inclusive: wp.array[wp.int32],
    segments: wp.array2d[wp.vec3],
    out_segments: wp.array2d[wp.vec3],
    out_rows: wp.array[wp.int32],
) -> None:
    # The cut rows of the two per-face segment producers above, in row order, from the in-place
    # inclusive scan of their ``0`` / ``1`` flags: ``flatnonzero`` and one gather per buffer in one
    # pass. ``out_rows`` (the source row of each kept segment) is optional -- a caller that has no
    # use for it passes ``None``, whose shape reads 0.
    i = wp.int32(wp.tid())
    slot = copy_cut_segment(inclusive, segments, i, out_segments)
    if slot < 0:
        return
    if out_rows.shape[0] > 0:
        out_rows[slot] = i


@wp.kernel
def compact_cut_segment_keys(
    inclusive: wp.array[wp.int32],
    segments: wp.array2d[wp.vec3],
    edges: wp.array2d[wp.int64],
    out_segments: wp.array2d[wp.vec3],
    out_keys: wp.array[wp.int64],
    out_endpoints: wp.array[wp.int32],
) -> None:
    # ``compact_cut_segments`` with the crossing keys: the key of endpoint ``e`` of kept segment
    # ``s`` goes to slot ``2 s + e`` of ``out_keys``. The device link hands it a radix sort's
    # double-width key buffer and the endpoint ids ``2 s + e`` as the sort's payload, so the sort
    # needs no staging pass; the host link passes ``None`` for the payload and reads keys and
    # segments back together, ``out_segments`` aliasing the tail of ``out_keys``' buffer.
    i = wp.int32(wp.tid())
    slot = copy_cut_segment(inclusive, segments, i, out_segments)
    if slot < 0:
        return
    out_keys[2 * slot] = edges[i, 0]
    out_keys[2 * slot + 1] = edges[i, 1]
    if out_endpoints.shape[0] > 0:
        out_endpoints[2 * slot] = 2 * slot
        out_endpoints[2 * slot + 1] = 2 * slot + 1


# The device link of ``marching_triangles``: pair endpoints through the sorted crossing keys, then
# rank every segment along its curve by multi-hop pointer jumping. It computes exactly what
# ``intersection._link_segments`` computes on the host, and the two are pinned to each other.
#
# A segment's ranking state is one ``wp.vec4i``: ``(ahead, lowest, offset, steps)``. ``ahead`` is
# the segment ``width`` hops downstream (an open curve's last segment is a fixed point),
# ``lowest`` the smallest segment index in the window of ``width`` segments starting here,
# ``offset`` how many hops into that window it first occurs, and ``steps`` how many real hops the
# window spans. Once the window is at least as long as the curve, an open curve's segments read
# their hop count to its last segment off ``steps``, and a closed curve's read the index it is
# entered at off ``lowest`` and their hop count to it off ``offset``.
LINK_MALFORMED = wp.constant(wp.int32(1))


@wp.kernel
def link_sorted_endpoints(
    keys: wp.array[wp.int64],
    endpoints: wp.array[wp.int32],
    n_endpoints: wp.int32,
    error_slot: wp.int32,
    out_link: wp.array[wp.int32],
    out_state: wp.array[wp.vec4i],
    out_error: wp.array[wp.int32],
) -> None:
    # One thread per sorted endpoint. An interior crossing is a run of two equal keys, one
    # segment's outgoing endpoint (even id) and another's incoming one (odd id); a boundary
    # crossing is a run of one. The thread holding a segment's incoming endpoint writes its
    # successor (``out_link[s]``) and its initial ranking state, the one holding its outgoing
    # endpoint whether it has a predecessor (``out_link[n + s]``), so every slot has exactly one
    # writer. Anything else -- a longer run, or two endpoints of one kind on one crossing -- is a
    # level set the host link must judge, flagged in ``out_error[error_slot]``.
    q = wp.int32(wp.tid())
    endpoint = endpoints[q]
    key = keys[q]
    partner = wp.int32(-1)
    same_prev = wp.bool(False)
    if q > 0:
        same_prev = keys[q - 1] == key
    if same_prev:
        partner = endpoints[q - 1]
    if q + 1 < n_endpoints:
        if keys[q + 1] == key:
            if same_prev:
                out_error[error_slot] = LINK_MALFORMED
            partner = endpoints[q + 1]
    if partner >= 0 and partner % 2 == endpoint % 2:
        out_error[error_slot] = LINK_MALFORMED

    segment = endpoint // 2
    n_segments = n_endpoints // 2
    if endpoint % 2 == 0:
        out_link[n_segments + segment] = wp.where(partner >= 0, wp.int32(1), wp.int32(0))
        return
    successor = wp.int32(-1)
    if partner >= 0:
        successor = partner // 2
    out_link[segment] = successor
    if successor >= 0:
        out_state[segment] = wp.vec4i(successor, segment, 0, 1)
    else:
        out_state[segment] = wp.vec4i(segment, segment, 0, 0)


@wp.kernel
def link_rank_round(
    state: wp.array[wp.vec4i], hops: wp.int32, width: wp.int32, out_state: wp.array[wp.vec4i]
) -> None:
    # One multi-hop pointer-jumping round: merge the ``hops`` consecutive windows of ``width``
    # segments starting here into one of ``hops * width``, under ``array.merge_window_minimum``'s
    # first-occurrence rule.
    i = wp.int32(wp.tid())
    own = state[i]
    ahead = own[0]
    lowest = own[1]
    offset = own[2]
    steps = own[3]
    for k in range(1, hops):
        window = state[ahead]
        lowest, offset = kernel_array.merge_window_minimum(
            lowest, offset, window[1], window[2], k * width
        )
        steps = steps + window[3]
        ahead = window[0]
    out_state[i] = wp.vec4i(ahead, lowest, offset, steps)


@wp.func
def linked_curve_is_closed(
    link: wp.array[wp.int32], state: wp.array[wp.vec4i], i: wp.int32
) -> wp.bool:
    # Once ranked, a segment on an open curve points at that curve's last segment, which has no
    # successor; one on a closed curve never reaches a fixed point.
    return link[state[i][0]] >= 0


@wp.kernel
def link_curve_weights(
    link: wp.array[wp.int32],
    state: wp.array[wp.vec4i],
    n_segments: wp.int32,
    out_weights: wp.array[wp.int32],
    out_head_of_tail: wp.array[wp.int32],
) -> None:
    # Name each curve by one segment and weigh it by the points it emits. Open curves are named by
    # their predecessor-free first segment at ``i`` and weigh their segment count plus one (the
    # last segment emits both endpoints); closed curves by the segment they are entered at, their
    # lowest index, at ``n + i``, and weigh their segment count. Every other slot weighs zero, so
    # the inclusive scan of the weights lists the curves -- open ones first, each kind in index
    # order -- with each named slot holding its curve's end bound. An open curve's segments only
    # know its last segment, so its first writes its own index there.
    i = wp.int32(wp.tid())
    own = state[i]
    open_weight = wp.int32(0)
    closed_weight = wp.int32(0)
    if linked_curve_is_closed(link, state, i):
        if own[1] == i:
            closed_weight = state[link[i]][2] + 1
    elif link[n_segments + i] == 0:
        open_weight = own[3] + 2
        out_head_of_tail[own[0]] = i
    out_weights[i] = open_weight
    out_weights[n_segments + i] = closed_weight


@wp.kernel
def emit_linked_curves(
    segments: wp.array2d[wp.vec3],
    link: wp.array[wp.int32],
    state: wp.array[wp.vec4i],
    bounds: wp.array[wp.int32],
    head_of_tail: wp.array[wp.int32],
    n_segments: wp.int32,
    out_points: wp.array[wp.vec3],
) -> None:
    # Write each segment's outgoing endpoint at its place along its curve in the packed output,
    # and an open curve's last segment its incoming one after it.
    i = wp.int32(wp.tid())
    own = state[i]
    closed = linked_curve_is_closed(link, state, i)
    name = own[1] + n_segments
    if not closed:
        name = head_of_tail[own[0]]
    start, count = kernel_array.scanned_count(bounds, name)
    position = count - wp.int32(2) - own[3]
    if closed:
        position = (count - own[2]) % count
    out_points[start + position] = segments[i, 0]
    if not closed and link[i] < 0:
        out_points[start + position + 1] = segments[i, 1]


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5.
#
# Only the scalar field being contoured is generic: ``marching_triangles`` accepts a ``wp.float32``
# or ``wp.float64`` per-vertex field (the heat solvers produce the latter), while the geometry it
# writes stays ``wp.vec3``.
# The concrete handle keyed by the field dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable].
MARCHING_TRIANGLES_SEGMENTS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global MARCHING_TRIANGLES_SEGMENTS
    MARCHING_TRIANGLES_SEGMENTS = OverloadTable(
        marching_triangles_segments,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[d],
                d,
                wp.int64,
                wp.array[wp.int32],
                wp.array2d[wp.vec3],
                wp.array2d[wp.int64],
            ]
            for d in (wp.float32, wp.float64)
        },
    )


_register_overloads()


@wp.func
def shift_to_float32(value: wp.float64, isovalue: wp.float64) -> wp.float32:
    """Re-zero a ``float64`` field value at ``isovalue`` in its own precision, then narrow it."""
    return wp.float32(value - isovalue)


@wp.func
def is_positive_split_class(face_class: wp.int32) -> wp.bool:
    """Side label for an *uncut* face, so the no-crossing path needs no second classifier."""
    return face_class == SPLIT_CLASS_POSITIVE


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (
                plane_with_line,
                (wp.vec3(), wp.vec3(), dense(wp.vec3), dense(wp.vec3), wp.bool(True)),
                [wp.vec3, wp.bool],
            ),
            (
                plane_with_line,
                (wp.vec3(), wp.vec3(), single(wp.vec3), single(wp.vec3), wp.bool(True)),
                [wp.vec3, wp.bool],
            ),
        ]
    )


_declare_map_kernels()
