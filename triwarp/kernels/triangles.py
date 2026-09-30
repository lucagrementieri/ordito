from typing import Any

import warp as wp

from triwarp.constants import PI, TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import (
    OverloadTable,
    binary_search_sorted_contains,
    pack_edge_key,
    to_vec3d,
)
from triwarp.kernels.halfedge import halfedge_next, halfedge_prev
from triwarp.kernels.predicates import (
    segment_coordinate,
    side_lengths,
    stable_normalize,
    triangle_aabb,
    triangle_aspect_ratio,
    triangle_double_area,
    triangle_normal,
    vector_angle,
)

# ``face_quality`` metric selectors. Passed as a warp-uniform kernel argument so all four share one
# compiled module (a ``wp.Function`` cannot be a kernel argument -- see AGENTS.md section 2.7).
QUALITY_ASPECT_RATIO = wp.constant(wp.int32(0))  # circumradius / (2 * inradius), 1 .. +inf
QUALITY_RADIUS_RATIO = wp.constant(wp.int32(1))  # VCG QualityRadii, 0 .. 1
QUALITY_AREA_MAX_SIDE = wp.constant(wp.int32(2))  # 2 * area / longest_side^2, 0 .. sqrt(3)/2
QUALITY_MEAN_RATIO = wp.constant(wp.int32(3))  # 4 * sqrt(3) * area / (a^2 + b^2 + c^2), 0 .. 1
QUALITY_AREA = wp.constant(wp.int32(4))  # plain triangle area


@wp.func
def corner_triple(buffer: wp.array[Any], row: wp.int32) -> tuple[Any, Any, Any]:
    """
    Load the three entries of row ``row`` of a flat 3-stride buffer.

    Generic over the element type, because what this names is the *row layout* and not the payload:
    the corner indices of a face out of ``faces``, the unique-edge ids of its three corners out of
    an ``inverse`` map, a per-corner scalar out of a ``(3F,)`` field. Where the three loaded values
    are then used to gather from a per-vertex array,
    [`face_vertices`][triwarp.kernels.triangles.face_vertices] does both steps in one call; this
    is the half for callers that need the indices themselves as well.

    [`row_triple`][triwarp.kernels.triangles.row_triple] is the rank-2 form, for the tables this
    package stores as ``(n_faces, 3)`` rather than flat.
    """
    base = row * wp.int32(3)
    return buffer[base], buffer[base + wp.int32(1)], buffer[base + wp.int32(2)]


@wp.func
def row_triple(buffer: wp.array2d[Any], row: wp.int32) -> tuple[Any, Any, Any]:
    """
    Load the three entries of row ``row`` of a rank-2 buffer.

    The rank-2 form of [`corner_triple`][triwarp.kernels.triangles.corner_triple], and generic for
    the same reason: what both name is a *row layout* and not a payload -- a face's corner indices
    out of an ``(n_faces, 3)`` table, its three edge lengths, its three half-cotangents, its three
    plane signs. The two live together because a reader meeting one of the tree's two face layouts
    should be shown the other.
    """
    return buffer[row, 0], buffer[row, 1], buffer[row, 2]


@wp.func
def write_row_triple(out: wp.array2d[Any], row: wp.int32, a: Any, b: Any, c: Any) -> None:
    """
    Write ``(a, b, c)`` into row ``row`` of a rank-2 buffer.

    The write-side counterpart of [`row_triple`][triwarp.kernels.triangles.row_triple], for the
    rank-2 tables this package stores as ``(n, 3)`` -- the sibling gap
    [`write_corner_triple`][triwarp.kernels.triangles.write_corner_triple] leaves for the flat
    3-stride layout. Reached from ``kernels/voxels.py``, which had the identical three-assignment
    write spelled out at four sites before this existed.
    """
    out[row, 0] = a
    out[row, 1] = b
    out[row, 2] = c


@wp.func
def local_corner(faces: wp.array[wp.int32], f: wp.int32, vertex: wp.int32) -> wp.int32:
    """
    Which corner (0, 1, 2) of face ``f`` holds ``vertex``, or ``-1`` if none does.

    The inverse of [`corner_triple`][triwarp.kernels.triangles.corner_triple]: that reads a face's
    three vertex indices by corner, this looks up a vertex's corner within a named face. First-match
    wins on a degenerate face with a repeated vertex index, matching every other by-index lookup in
    this module (e.g. [`corner_triple`][triwarp.kernels.triangles.corner_triple]'s own row order).
    """
    for k in range(3):
        if faces[f * 3 + k] == vertex:
            return k
    return wp.int32(-1)


@wp.func
def write_corner_triple(
    out: wp.array[wp.int32], row: wp.int32, a: wp.int32, b: wp.int32, c: wp.int32
) -> None:
    """
    Write ``(a, b, c)`` into row ``row`` of a flat 3-stride buffer.

    The write-side counterpart of [`corner_triple`][triwarp.kernels.triangles.corner_triple], and
    only for the sites that permute or relocate an existing triple -- an edge flip's two rewritten
    faces (``remesh.write_flipped_quad``), a compaction moving a survivor to a new row
    (``remesh.compact_decimation_pass``, which is exactly where naming the row pays: it reads one
    row base and writes another). Most triangle-emitting kernels in the tree instead synthesize a
    face from local context (a fan, a cut, a bridge) rather than reorder one, and those stay as they
    are -- there is no shared decision to extract from a one-off construction. No
    separate "reversed" sibling: a full ``np.fliplr``-style reversal is this same function called
    ``write_corner_triple(out, row, c, b, a)`` -- callers pass their own arguments in the order they
    want written, the way ``repair.reverse_face_winding`` does, rather than naming a second
    ``@wp.func`` whose body would be this one's with its parameters permuted (identical code once
    inlined, so the only thing a "reversed" sibling would add is a second name for the same three
    atomics).
    ``repair.flip_faces_masked`` and ``levelset.shell_faces`` keep corner 0 and swap only corners 1
    and 2 -- a *different* reversal, already cross-referenced to each other, and not an adopter of
    this function: conflating the two conventions behind one flag would be the correctness hazard
    CLAUDE.md section 2.4 warns against, not a simplification.
    """
    out[row * wp.int32(3) + wp.int32(0)] = a
    out[row * wp.int32(3) + wp.int32(1)] = b
    out[row * wp.int32(3) + wp.int32(2)] = c


@wp.func
def write_corner_triple_reversible(
    out: wp.array[wp.int32], row: wp.int32, a: wp.int32, b: wp.int32, c: wp.int32, reverse: wp.bool
) -> None:
    """
    [`write_corner_triple`][triwarp.kernels.triangles.write_corner_triple], reversed by ``reverse``.

    The one decision several face-emitting kernels share -- write a triple forward, or with its
    first and last corners swapped (trimesh's ``np.fliplr``) -- factored once rather than repeated
    at every call site that computes a per-face orientation flag
    (``creation.revolve_cap_faces``/``offset_cap_faces_both``/``write_prism_face``). Still just
    ``write_corner_triple`` under either branch, so this is not the permuted "reversed" sibling that
    function's own docstring declines to add: the swap happens once here, at the one place a
    *runtime* flag decides which of the two a caller wanted, not as a second three-atomic body.
    """
    if reverse:
        write_corner_triple(out, row, c, b, a)
    else:
        write_corner_triple(out, row, a, b, c)


@wp.func
def face_vertices(
    vertices: wp.array[Any], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[Any, Any, Any]:
    """
    Load the three per-corner values of face ``face_index`` from a flat index buffer.

    Generic over the value dtype: works for positions (``wp.vec3``/``wp.vec3d``/``wp.vec2``)
    as well as per-vertex scalar fields.
    """
    i0, i1, i2 = corner_triple(faces, face_index)
    return vertices[i0], vertices[i1], vertices[i2]


@wp.func
def face_normal(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32):
    """
    Compute the unit normal of face ``face_index``; the zero vector when the face is degenerate.

    The by-index form of [`triangle_normal`][triwarp.kernels.predicates.triangle_normal], which is
    how most callers want it. Identical to the normal
    [`face_normals_and_area`][triwarp.kernels.triangles.face_normals_and_area] returns -- both
    normalize whenever the cross product is non-zero and hand back exactly the zero vector when it
    is not -- so reach for that one when the area is wanted too, and for this one when it is not.
    """
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return triangle_normal(v0, v1, v2)


@wp.func
def face_vertices_vec3d(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[wp.vec3d, wp.vec3d, wp.vec3d]:
    """Load the three corners of face ``face_index`` promoted to ``wp.vec3d``."""
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return to_vec3d(v0), to_vec3d(v1), to_vec3d(v2)


@wp.kernel
def face_signed_volumes(
    vertices: wp.array[Any], faces: wp.array[wp.int32], center: Any, out_volumes: wp.array[wp.Float]
) -> None:
    # Signed volume of the tetrahedron (center, v0, v1, v2); the sum over faces is the mesh volume.
    fi = wp.int32(wp.tid())
    p0, p1, p2 = face_vertices(vertices, faces, fi)
    d = wp.dot(p0 - center, wp.cross(p1 - center, p2 - center))
    out_volumes[fi] = d / type(d)(6.0)


@wp.func
def triangle_cross(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> wp.vec3:
    """Unnormalized normal of face ``face_index``: the cross product of its two first edges."""
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return wp.cross(v1 - v0, v2 - v0)


@wp.func
def triangle_edges(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Return the three edge vectors of face ``face_index``: ``(v1 - v0, v2 - v0, v2 - v1)``."""
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return v1 - v0, v2 - v0, v2 - v1


@wp.func
def face_normals_and_area(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[wp.vec3, wp.float32]:
    """
    Return the unit normal and the area of face ``face_index``, in one pass over its corners.

    The normal is [`face_normal`][triwarp.kernels.triangles.face_normal]'s exactly -- unit, or the
    zero vector for an exactly degenerate face -- and this one returns the area alongside it.

    **The zero test is against zero and not against a tolerance, and that is load-bearing at small
    mesh scale.** ``|cross|`` scales as ``h^2``, so an absolute floor of
    ``TOLERANCE_ZERO_CONSTANT`` (which this used to carry) puts *every* face of a small enough mesh
    below it and hands back the raw cross product as a "unit" normal of magnitude ``2 * area``.
    Nothing downstream reads that as an error: ``vertices.vertex_normals`` then area-weights those,
    squaring the smallness, and ``wp.normalize`` sees a vector whose ``length_sq`` has underflowed
    ``float32`` to exactly zero -- so every vertex normal on the mesh comes back zero, and with them
    every curvature, every smoothing normal and every sign test built on one. Dividing by ``norm``
    is well conditioned for *any* positive ``norm``; at ``norm == 0`` the cross product is already
    the zero vector, so the two branches agree there and the tolerance was never protecting the
    division.
    """
    normal = triangle_cross(vertices, faces, face_index)
    norm = wp.length(normal)
    if norm > wp.float32(0.0):
        normal = normal / norm
    area = 0.5 * norm
    return normal, area


@wp.kernel
def face_normals_and_areas(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
    out_areas: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    normal, area = face_normals_and_area(vertices, faces, f)
    out_normals[f] = normal
    out_areas[f] = area


@wp.func
def face_corner_angles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    """
    Interior angles at the three corners of face ``face_index``; all zero for a degenerate face.

    The body of the ``angles`` kernel, named so a kernel that weights by corner angle can take them
    in registers rather than through an ``(n_faces, 3)`` table -- ``vertices.vertex_normals``'
    ``"angle"`` path is the one that does.
    """
    edges = triangle_edges(vertices, faces, face_index)

    # ``vector_angle`` is atan2(|a x b|, a . b) and is scale-free, so the edges go in unnormalized
    # (three ``wp.normalize`` calls fewer) -- and a sliver, whose angles sit near 0 and pi, is
    # precisely where the ``acos(a . b)`` this replaces amplified the round-off already in the dot
    # product. Against a float64 reference on the same float32 vertex buffer the gain grows with
    # sliverness, and on a finely sectioned cylinder the old form was closing on the 1e-5 the parity
    # tests compare at.
    #
    # It also removes the spurious pi/2 that ``acos`` of a zeroed ``normalize`` returned for a
    # zero-length edge; the degeneracy guard below now fires on the angle itself rather than on
    # the third angle's residue. Both spellings end at (0, 0, 0) there, by different routes.
    #
    # The other corner-angle formulation in the tree is ``energies.internal_angles_and_sums``, which
    # is the law of cosines on squared edge lengths in float64 and additionally accumulates the
    # per-vertex angle sums its curvature correction needs. It is not this kernel at a wider dtype:
    # there each angle is derived independently (so the three sum to pi only up to round-off) and a
    # sliver reads 0 or pi through the acos clamp, where this one takes the third angle as
    # ``PI - a0 - a1`` and zeroes all three of a degenerate face.
    a0 = vector_angle(edges[0], edges[1])
    a1 = vector_angle(-edges[0], edges[2])
    a2 = PI - a0 - a1
    if (
        (a0 < TOLERANCE_MERGE_CONSTANT)
        or (a1 < TOLERANCE_MERGE_CONSTANT)
        or (a2 < TOLERANCE_MERGE_CONSTANT)
    ):
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    return a0, a1, a2


@wp.kernel
def angles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_angles: wp.array2d[wp.float32]
) -> None:
    f = wp.int32(wp.tid())
    a0, a1, a2 = face_corner_angles(vertices, faces, f)
    write_row_triple(out_angles, f, a0, a1, a2)


@wp.func
def triangle_radius_ratio(a: Any, b: Any, c: Any) -> wp.Float:
    # VCG ``QualityRadii`` ("inradius/circumradius"): the ratio of the two radii, rescaled so an
    # equilateral triangle reads 1 (the bare geometric ratio is 1/2 there). Symmetric in the three
    # side lengths; zero for a degenerate triangle.
    bc, ca, ab = side_lengths(a, b, c)
    product = ab * ca * bc
    if product <= type(product)(0.0):
        return type(product)(0.0)
    return (ab + ca - bc) * (bc + ab - ca) * (ca + bc - ab) / product


@wp.func
def triangle_area_max_side(a: Any, b: Any, c: Any) -> wp.Float:
    # VCG ``Quality`` ("area/max side"): twice the area over the longest side squared, so it is
    # scale-invariant despite the name. ``sqrt(3)/2`` for an equilateral triangle, 0 for a
    # degenerate one.
    ab = b - a
    ac = c - a
    longest_sq = wp.max(wp.max(wp.length_sq(ab), wp.length_sq(ac)), wp.length_sq(wp.sub(c, b)))
    if longest_sq <= type(longest_sq)(0.0):
        return type(longest_sq)(0.0)
    return wp.length(wp.cross(ab, ac)) / longest_sq


@wp.func
def triangle_mean_ratio(a: Any, b: Any, c: Any) -> wp.Float:
    # VCG ``QualityMeanRatio``: ``4 * sqrt(3) * area / (a^2 + b^2 + c^2)`` -- 1 for an equilateral
    # triangle, 0 for a degenerate one.
    ab = b - a
    ac = c - a
    sum_sq = wp.length_sq(ab) + wp.length_sq(ac) + wp.length_sq(wp.sub(c, b))
    if sum_sq <= type(sum_sq)(0.0):
        return type(sum_sq)(0.0)
    return type(sum_sq)(2.0) * wp.sqrt(type(sum_sq)(3.0)) * wp.length(wp.cross(ab, ac)) / sum_sq


@wp.func
def triangle_quality(a: Any, b: Any, c: Any, metric: wp.int32) -> wp.Float:
    # Warp-uniform dispatch over the ``QUALITY_*`` selectors: one compiled module for all five
    # metrics, since a ``wp.Function`` cannot cross the ``wp.launch`` boundary.
    if metric == QUALITY_ASPECT_RATIO:
        return triangle_aspect_ratio(a, b, c)
    if metric == QUALITY_RADIUS_RATIO:
        return triangle_radius_ratio(a, b, c)
    if metric == QUALITY_AREA_MAX_SIDE:
        return triangle_area_max_side(a, b, c)
    if metric == QUALITY_MEAN_RATIO:
        return triangle_mean_ratio(a, b, c)
    return type(a[0])(0.5) * triangle_double_area(a, b, c)


@wp.kernel
def face_quality(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: wp.int32,
    out_quality: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    out_quality[f] = triangle_quality(v0, v1, v2, metric)


@wp.kernel
def face_nondegenerate_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_nondegenerate: wp.array[wp.bool]
) -> None:
    f = wp.int32(wp.tid())
    e0, e1, _ = triangle_edges(vertices, faces, f)
    _, area = face_normals_and_area(vertices, faces, f)
    length_e0 = wp.length(e0)
    length_e1 = wp.length(e1)
    height_e0 = 2.0 * area / length_e0
    height_e1 = 2.0 * area / length_e1
    out_nondegenerate[f] = (
        (height_e0 > TOLERANCE_MERGE_CONSTANT)
        and (height_e1 > TOLERANCE_MERGE_CONSTANT)
        and (length_e0 > TOLERANCE_MERGE_CONSTANT)
        and (length_e1 > TOLERANCE_MERGE_CONSTANT)
    )


@wp.kernel
def barycentric_to_points(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    barycentric: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
) -> None:
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    face_barycentric = barycentric[f]
    s = face_barycentric[0] + face_barycentric[1] + face_barycentric[2]
    face_barycentric = face_barycentric / s
    out_points[f] = v0 * face_barycentric[0] + v1 * face_barycentric[1] + v2 * face_barycentric[2]


@wp.func
def point_barycentric(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, point: wp.vec3) -> wp.vec3:
    """
    Barycentric coordinates of ``point`` projected into the plane of triangle ``(v0, v1, v2)``.

    The denominator is written ``|e0 x e1|^2``, from products that never cancel. **The obvious
    alternative -- Cramer's rule on the Gram system, whose denominator is the algebraically equal
    ``|e0|^2 |e1|^2 - (e0 . e1)^2`` -- was a second public ``method`` here and was removed, because
    it is worse at every triangle shape and scale and better at none.** Its two terms agree to more
    digits as the corner angle closes, so the subtraction loses them.

    Measured against a ``float128`` oracle over a wide sweep of base scales and triangle heights,
    this form sits at ``float32`` eps in every cell, flat in the scale, where Cramer degrades
    monotonically with sliverness: at a height around ``1e-4`` of the base it returns values that
    are *finite and wholly wrong*, which is worse than the ``nan`` it gives below that because
    nothing downstream can detect it. The breakdown is not a ``float32`` artifact -- the same sweep
    shows ``float128`` Cramer departing from the exact answer too -- so no widening rescues it. The
    two cost the same, both at the launch floor.

    Zero area -- a triangle that really is a segment or a point, not merely a thin one -- is
    answered rather than forwarded as an infinity: the coordinates are then taken along the longest
    edge, which is not an approximation but the triangle itself.
    """
    e0 = v1 - v0
    e1 = v2 - v0
    w = point - v0
    n = wp.cross(e0, e1)
    denom = wp.length_sq(n)
    if denom > 0.0:
        inverse_denominator = 1.0 / denom
        b1 = wp.dot(wp.cross(w, e1), n) * inverse_denominator
        b2 = wp.dot(wp.cross(e0, w), n) * inverse_denominator
        return wp.vec3(1.0 - b1 - b2, b1, b2)

    d01 = wp.length_sq(e0)
    d12 = wp.length_sq(v2 - v1)
    d20 = wp.length_sq(e1)
    if d01 >= d12 and d01 >= d20:
        t = segment_coordinate(v0, v1, point)
        return wp.vec3(1.0 - t, t, 0.0)
    if d12 >= d20:
        t = segment_coordinate(v1, v2, point)
        return wp.vec3(0.0, 1.0 - t, t)
    t = segment_coordinate(v2, v0, point)
    return wp.vec3(t, 0.0, 1.0 - t)


@wp.kernel
def points_to_barycentric(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_barycentric: wp.array[wp.vec3],
) -> None:
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    out_barycentric[f] = point_barycentric(v0, v1, v2, points[f])


@wp.kernel
def closest_point(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_closest: wp.array[wp.vec3],
) -> None:
    f = wp.int32(wp.tid())
    corner_a, corner_b, corner_c = face_vertices(vertices, faces, f)
    ab, ac, bc = triangle_edges(vertices, faces, f)

    # check if P is in vertex region outside A
    ap = points[f] - corner_a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    is_a = d1 < 0.0 and d2 < 0.0
    if is_a:
        out_closest[f] = corner_a
        return

    # check if P in vertex region outside B
    bp = points[f] - corner_b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    is_b = d3 > -TOLERANCE_ZERO_CONSTANT and d4 <= d3
    if is_b:
        out_closest[f] = corner_b
        return

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = (
        vc < TOLERANCE_ZERO_CONSTANT
        and d1 > -TOLERANCE_ZERO_CONSTANT
        and d3 < TOLERANCE_ZERO_CONSTANT
    )
    if is_ab:
        v = d1 / (d1 - d3)
        out_closest[f] = corner_a + v * ab
        return

    # check if P in vertex region outside C
    cp = points[f] - corner_c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    is_c = d6 > -TOLERANCE_ZERO_CONSTANT and d5 <= d6
    if is_c:
        out_closest[f] = corner_c
        return

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (
        vb < TOLERANCE_ZERO_CONSTANT
        and d2 > -TOLERANCE_ZERO_CONSTANT
        and d6 < TOLERANCE_ZERO_CONSTANT
    )
    if is_ac:
        w = d2 / (d2 - d6)
        out_closest[f] = corner_a + w * ac
        return

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = (
        va < TOLERANCE_ZERO_CONSTANT
        and (d4 - d3) > -TOLERANCE_ZERO_CONSTANT
        and (d5 - d6) > -TOLERANCE_ZERO_CONSTANT
    )
    if is_bc:
        d43 = d4 - d3
        w = d43 / (d43 + (d5 - d6))
        out_closest[f] = corner_b + w * bc
        return

    # any remaining points must be inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    out_closest[f] = corner_a + ab * v + ac * w


@wp.func
def face_centroid(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32) -> wp.vec3:
    # Barycentre of face ``f``: the mean of its three corners.
    v0, v1, v2 = face_vertices(vertices, faces, f)
    return (v0 + v1 + v2) / 3.0


@wp.func
def face_area_weighted_centroid(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.vec3, wp.float32]:
    # One face's contribution to a *surface* centroid: its barycentre scaled by its area, and that
    # area. Summing the pair over the mesh and dividing gives the area-weighted mean, which is what
    # distinguishes a surface centroid from the plain mean of ``face_centroid`` above.
    #
    # Both are returned because every caller needs both and the area is already in registers.
    # ``triangle_double_area`` over the corners already loaded rather than
    # ``face_normals_and_area``, which re-enters ``face_vertices`` through ``triangle_cross`` and
    # normalizes a normal a centroid reduction then discards; the expression is the same one either
    # way (``0.5 * |cross(v1 - v0, v2 - v0)|``), so the areas are bit-identical.
    v0, v1, v2 = face_vertices(vertices, faces, f)
    area = 0.5 * triangle_double_area(v0, v1, v2)
    return (v0 + v1 + v2) * (area / 3.0), area


@wp.kernel
def face_centroids(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_centroids: wp.array[wp.vec3]
) -> None:
    f = wp.int32(wp.tid())
    out_centroids[f] = face_centroid(vertices, faces, f)


@wp.func
def face_gradient(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    f: wp.int32,
) -> wp.vec3d:
    # Gradient of a per-vertex scalar field inside face ``f``, in the face's plane:
    #   grad = 1/(2A) * sum_k values_k * (n x e_k^opp),   e_k^opp the CCW edge opposite corner k.
    #
    # Accumulated in float64. The fields this serves (diffused heat, geodesic distance) decay
    # exponentially and a float32 sum of the cross products loses the far field, so the geometry is
    # promoted rather than the result being widened after the fact.
    #
    # A degenerate face contributes nothing and returns the zero vector.
    i0, i1, i2 = corner_triple(faces, f)
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    n = to_vec3d(normals[f])
    area = wp.float64(areas[f])

    grad = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if area > wp.float64(0.0):
        grad = (
            values[i0] * wp.cross(n, v2 - v1)
            + values[i1] * wp.cross(n, v0 - v2)
            + values[i2] * wp.cross(n, v1 - v0)
        ) / (wp.float64(2.0) * area)
    return grad


@wp.func
def face_unit_gradient(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    f: wp.int32,
) -> wp.vec3d:
    # The direction of ``face_gradient``, zero for a zero gradient (a degenerate face, a constant
    # field). Scaled by its largest component before ``normalize``: far from a heat source the
    # gradient is as small as ``1e-300``, whose squared length underflows ``float64`` and would
    # read as zero -- ``igl::heat_geodesics_solve`` normalizes the same way, for the same reason.
    return stable_normalize(face_gradient(vertices, faces, normals, areas, values, f))


@wp.kernel
def face_gradients(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    out_gradients: wp.array[wp.vec3d],
) -> None:
    f = wp.int32(wp.tid())
    out_gradients[f] = face_gradient(vertices, faces, normals, areas, values, f)


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5. This module backs 15 kernel modules and 2 wrappers, so each rebuild is
# widely felt.
#
# The vertex precision and the volume precision move together: ``face_signed_volumes`` reads a
# ``wp.vec3`` cloud into ``wp.float32`` volumes or a ``wp.vec3d`` one into ``wp.float64``, never a
# mixture, because the wrapper derives the output dtype from the vertex dtype.
# The concrete handle keyed by the vertex dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable]. This kernel is the tree's clearest case:
# it is generic in *three* parameters (the vertex array, the apex vector and the output scalar), and
# resolution cost scales with that count, so launching through the handle roughly halves its host
# cost with the output bit-identical.
FACE_SIGNED_VOLUMES: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global FACE_SIGNED_VOLUMES
    FACE_SIGNED_VOLUMES = OverloadTable(
        face_signed_volumes,
        {
            vector: [wp.array[vector], wp.array[wp.int32], vector, wp.array[scalar]]
            for vector, scalar in ((wp.vec3, wp.float32), (wp.vec3d, wp.float64))
        },
    )


_register_overloads()


@wp.func
def is_crease_edge(
    faces: wp.array[wp.int32],
    crease_keys_sorted: wp.array[wp.uint64],
    base: wp.uint64,
    halfedge: wp.int32,
) -> wp.bool:
    # Whether the undirected edge a halfedge lies on is in the crease set. An empty set answers
    # ``False`` for every edge without a special case: ``binary_search_sorted_contains`` short-
    # circuits before reading, so the no-crease path costs one compare per rotation step.
    return binary_search_sorted_contains(
        crease_keys_sorted, pack_edge_key(faces[halfedge], faces[halfedge_next(halfedge)], base)
    )


@wp.func
def corner_weight(
    weights: wp.array[wp.float32], per_corner: wp.bool, halfedge: wp.int32
) -> wp.float32:
    """
    Weight of corner ``halfedge``, from a per-corner table or a per-face one.

    ``weighting="area"`` has one weight per *face*, and broadcasting it to the face's three corners
    on the host cost an ``arange_repeat`` launch and a gather -- two launches and two buffers to
    restate ``halfedge // 3``, which this kernel already computes. The branch is warp-uniform.
    """
    if per_corner:
        return weights[halfedge]
    return weights[halfedge // 3]


@wp.kernel
def corner_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    corner_weights: wp.array[wp.float32],
    weights_per_corner: wp.bool,
    crease_keys_sorted: wp.array[wp.uint64],
    base: wp.uint64,
    out_corner_normals: wp.array2d[wp.vec3],
) -> None:
    # One thread per corner, which is one thread per halfedge: corner ``k`` of face ``f`` sits at
    # ``faces[3f + k]``, and halfedge ``3f + k`` is the one leaving that vertex inside that face.
    #
    # The corner's normal averages the *smooth group* around its vertex -- the faces reachable by
    # rotating about it without crossing a crease -- weighted by corner angle, then normalized. So
    # with no creases every corner of a vertex gets that vertex's angle-weighted normal, and with
    # every edge a crease each corner gets its own face's normal. Those two are exact identities and
    # are what the test pins.
    corner = wp.int32(wp.tid())
    face = corner // 3
    total = face_normals[face] * corner_weight(corner_weights, weights_per_corner, corner)

    # Counter-clockwise about the vertex: ``h -> twins[prev(h)]``, the rotation
    # ``halfedge.vertex_one_rings`` walks. The edge crossed is the one ``prev(h)`` lies on.
    #
    # ``closed_loop`` is tracked separately from ``halfedge == corner`` rather than read off it:
    # the very first step can break on a crease *before* ``halfedge`` is ever reassigned, which
    # leaves it sitting at its initial value of ``corner`` -- indistinguishable from a genuine full
    # rotation back to the start if the two were conflated, and that collision silently skipped the
    # clockwise half whenever a corner's own ``prev`` edge happened to be a crease.
    halfedge = corner
    closed_loop = wp.bool(False)
    for _ in range(twins.shape[0]):
        crossing = halfedge_prev(halfedge)
        if is_crease_edge(faces, crease_keys_sorted, base, crossing):
            break
        halfedge = twins[crossing]
        if halfedge < 0:
            break  # a boundary edge ends the fan
        if halfedge == corner:
            closed_loop = wp.bool(True)
            break  # back at the start: the fan closed and every face is already counted
        total += face_normals[halfedge // 3] * corner_weight(
            corner_weights, weights_per_corner, halfedge
        )

    # Clockwise, the inverse rotation ``h -> next(twins[h])``, crossing ``h``'s own edge. Skipped
    # entirely when the fan already closed, since every face is then already counted.
    if not closed_loop:
        halfedge = corner
        for _ in range(twins.shape[0]):
            if is_crease_edge(faces, crease_keys_sorted, base, halfedge):
                break
            twin = twins[halfedge]
            if twin < 0:
                break
            halfedge = halfedge_next(twin)
            if halfedge == corner:
                break
            total += face_normals[halfedge // 3] * corner_weight(
                corner_weights, weights_per_corner, halfedge
            )

    length = wp.length(total)
    if length > TOLERANCE_ZERO_CONSTANT:
        total = total / length
    out_corner_normals[face, corner - face * 3] = total


@wp.kernel
def face_aabb_bounds(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    lower, upper = triangle_aabb(v0, v1, v2)
    out_lower[f] = lower
    out_upper[f] = upper
