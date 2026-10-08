"""
Precision-generic geometric predicates and the small triangle quantities they share.

These are the standard triangle quantities -- double area, circumcircle and minimum-enclosing-
circle diameters, aspect ratio, dihedral angle. Each `@wp.func` here is generic over the scalar
type, so one definition instantiates at whatever precision the calling kernel uses; before this
module existed the same quantities were duplicated per precision across three kernel modules.

The rule that keeps arrivals coming here is: a general geometric quantity
must not live in a module that owns an *algorithm*, or unrelated modules import the algorithm to
reach the geometry. ``triangle_aabb``, ``triangle_double_area``, ``circumcircle_diameter``,
``point_plane_dot`` and ``triangle_aabb_overlap`` all arrived that way.

Degenerate inputs return [`float_inf`][ordito.kernels.predicates.float_inf] -- an actual infinity,
so callers detect the case with ``wp.isinf`` rather than by comparing against a magic large value.

!!! note
    Vector parameters are annotated ``Any`` rather than a concrete ``wp.vec3`` / ``wp.vec3d``:
    Warp has no generic vector annotation, and ``Any`` lets the kernel's own vector type flow
    through. Scalars use ``wp.Float``. Literals inside these functions must be built with
    ``type(x)(...)`` so they instantiate at the caller's precision.
"""

import math
from typing import Any

import warp as wp

from ordito.constants import (
    FLOAT64_INF_CONSTANT,
    TOLERANCE_MERGE_CONSTANT,
    TOLERANCE_ZERO_CONSTANT,
    TOLERANCE_ZERO_F64,
)
from ordito.kernels import array as kernel_array
from ordito.kernels.array import cross2, declare_map_signatures, map_probe, sort3

wp.set_module_options({"enable_backward": False})

# Full turn in ``float64``; ``type(x)(TWO_PI_F64)`` narrows it to the caller's precision, and at
# ``float32`` that is bit-identical to ``2 * wp.PI`` (verified on both devices, Warp 1.18).
TWO_PI_F64 = wp.constant(wp.float64(2.0 * math.pi))


@wp.func
def float_inf(sample: wp.Float) -> wp.Float:
    # Positive infinity at ``sample``'s precision: the degenerate-case sentinel for every predicate
    # below. Callers test it with ``wp.isinf``.
    return type(sample)(wp.INF)


@wp.func
def orient2d(a: Any, b: Any, c: Any):
    # Twice the signed area of triangle ABC in 2D: > 0 when C is left of the directed line A->B.
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@wp.func
def newell_term(a: Any, b: Any, origin: Any):
    # One edge's term of Newell's sum, ``(a - o) x (b - o)``, taken about a point ``o`` of its own
    # ring. Around a closed ring the sum is the same for any ``o``, but only in exact arithmetic:
    # about the coordinate origin it subtracts cross products of the size ``|p|^2`` to leave a
    # ring area that can be orders smaller, so a 3-edge needle hole at coordinates near 4 came out
    # 6.3e-5 off in ``float32``. About a ring point every term is of the ring's own size.
    return wp.cross(a - origin, b - origin)


@wp.func
def triangle_area_vector(a: Any, b: Any, c: Any):
    # ``cross(b - a, c - a)`` -- twice the vector area of triangle ABC -- formed at the corner
    # opposite the longest edge, which is the same vector in exact arithmetic and the
    # well-conditioned one in floating point. At any corner the two edge vectors are near-parallel
    # exactly when that corner's angle is small, and their cross product then cancels: the error
    # is ``eps / sin(angle)`` relative, i.e. ``eps`` times the aspect ratio on a needle. The
    # largest angle is at least 60 degrees, so there the error is a few ``eps`` whatever the
    # shape. Measured on ``saddle_graded``'s aspect-4 900 needles: unit normals off ``float64`` by
    # 6.7e-5 at the first corner on CUDA (FMA contraction of the cancelling products makes it 10x
    # the CPU's 7.0e-6), 1.2e-7 here on both devices.
    #
    # The tie-break also keeps a face with a repeated vertex exactly zero on CUDA: one of the two
    # edges at the chosen corner is then the zero vector, where the first-corner form left FMA's
    # ~1e-8 residue.
    ab = b - a
    bc = c - b
    ca = a - c
    l_ab = wp.length_sq(ab)
    l_bc = wp.length_sq(bc)
    l_ca = wp.length_sq(ca)
    if l_bc >= l_ca and l_bc >= l_ab:
        return wp.cross(ab, -ca)  # corner a
    if l_ca >= l_ab:
        return wp.cross(bc, -ab)  # corner b
    return wp.cross(ca, -bc)  # corner c


@wp.func
def triangle_normal(a: Any, b: Any, c: Any):
    # Unit normal of triangle ABC, or the zero vector when the triangle is degenerate. Warp's
    # ``kEps`` is 0, so ``normalize`` already returns the zero vector for a zero-length input.
    return wp.normalize(triangle_area_vector(a, b, c))


@wp.func
def plane_basis(normal: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    # An arbitrary orthonormal in-plane basis for the plane perpendicular to ``normal``. Kernel-
    # scope mirror of ``ordito.points.plane_basis``, for callers that hold the normal in device
    # memory and must not read it back to build the frame -- a general geometric helper with no
    # point-cloud dependency, which is why it lives here rather than in ``kernels/points.py``: it
    # is reached from two unrelated kernel modules (``kernels/smoothing.py``,
    # ``kernels/polyline.py``), the same "reached from a second module" trigger already applied to
    # ``triangle_aabb``/``triangle_double_area``/``circumcircle_diameter``.
    unit_normal = wp.normalize(normal)
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(unit_normal[0]) > 0.9:
        axis = wp.vec3(0.0, 1.0, 0.0)
    u = wp.normalize(wp.cross(axis, unit_normal))
    v = wp.cross(unit_normal, u)
    return u, v


@wp.func
def triangle_double_area(a: Any, b: Any, c: Any) -> wp.Float:
    # Twice the area of triangle ABC: the norm of the edge cross product.
    # Kept undivided because most callers either compare it against zero or fold the half into a
    # constant of their own.
    return wp.length(triangle_area_vector(a, b, c))


@wp.func
def doublearea_from_lengths(l0: wp.Float, l1: wp.Float, l2: wp.Float) -> wp.Float:
    # Twice the area of a triangle from its three side lengths -- the *intrinsic* counterpart of
    # ``triangle_double_area``, for callers holding an edge-length table rather than positions.
    #
    # Kahan's rearrangement of Heron's formula, ``(a + (b + c))(c - (a - b))(c + (a - b))
    # (a + (b - c))`` with ``a >= b >= c``: the *largest* side first. ``sort3`` sorts ascending, so
    # ``a`` is ``l2``. With the order reversed (the smallest side as ``a``, as this read until
    # 2026-10-08) the parentheses no longer pair sides whose difference is exact, and a needle's
    # area came out 1.2e-4 off in ``float32`` on ``saddle_graded`` (1.6e-7 in this order). The
    # ``wp.max`` clamps a slightly negative product from round-off, and the ``isnan`` catches what
    # the clamp does not.
    c, b, a = sort3(l0, l1, l2)
    arg = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    dbl_area = type(l0)(0.5) * wp.sqrt(wp.max(arg, type(l0)(0.0)))
    if wp.isnan(dbl_area):
        return type(l0)(0.0)
    return dbl_area


@wp.func
def squared_edge_lengths(a: Any, b: Any, c: Any) -> tuple[wp.Float, wp.Float, wp.Float]:
    # Squared side lengths of triangle ABC, each *opposite* the corner of the same index -- the
    # ``igl`` intrinsic convention, and the input every law-of-cosines form below expects.
    return wp.length_sq(b - c), wp.length_sq(c - a), wp.length_sq(a - b)


@wp.func
def side_lengths(a: Any, b: Any, c: Any) -> tuple[wp.Float, wp.Float, wp.Float]:
    # Side lengths of triangle ABC, same opposite-corner convention as ``squared_edge_lengths``.
    #
    # Three ``wp.length`` calls rather than three square roots of that one: ``length`` is a single
    # ``sqrt`` of the same sum, so the composed spelling would add nothing but a name. The two
    # exist as a pair because the callers genuinely split -- an intrinsic law-of-cosines form wants
    # the squares, a radius ratio or an aspect ratio wants the lengths.
    return wp.length(b - c), wp.length(c - a), wp.length(a - b)


@wp.func
def law_of_cosines_angle(
    adjacent_a: wp.Float, adjacent_b: wp.Float, opposite: wp.Float
) -> wp.Float:
    # Angle between two adjacent sides of a triangle, from its three side lengths alone -- the
    # single-corner form of ``corner_cosines_from_l2``, for the intrinsic algorithms that hold an
    # edge-length table and never a vertex position.
    #
    # Not a call to ``corner_cosines_from_l2``: that one takes *squared* lengths, and it has no
    # degenerate guard because its callers disagree about the policy. This one does have one, and
    # returning 0 for a vanishing ``2ab`` is what ``remesh``'s intrinsic flip is written against.
    denominator = type(adjacent_a)(2.0) * adjacent_a * adjacent_b
    if denominator <= type(adjacent_a)(TOLERANCE_ZERO_CONSTANT):
        return type(adjacent_a)(0.0)
    cosine = (adjacent_a * adjacent_a + adjacent_b * adjacent_b - opposite * opposite) / denominator
    return wp.acos(cosine)  # wp.acos auto-clamps to [-1, 1]


@wp.func
def corner_cosines_from_l2(
    l2_0: wp.Float, l2_1: wp.Float, l2_2: wp.Float
) -> tuple[wp.Float, wp.Float, wp.Float]:
    # The three corner cosines of a triangle from its squared side lengths alone: the law of
    # cosines, ``cos(C) = (a^2 + b^2 - c^2) / 2ab``, with ``l2_e`` opposite corner ``e``.
    #
    # **No degenerate guard, deliberately.** The callers' policies differ and both are considered:
    # ``energies.internal_angles_and_sums`` wants the ``wp.acos`` of these and lets its clamp
    # report 0 or pi for a sliver, where ``remesh.law_of_cosines_angle`` returns 0 for a vanishing
    # denominator. Folding either policy in here would silently change the other caller's answer,
    # so the division by zero stays visible at the call site that has to decide about it.
    two = type(l2_0)(2.0)
    l0 = wp.sqrt(l2_0)
    l1 = wp.sqrt(l2_1)
    l2 = wp.sqrt(l2_2)
    return (
        (l2_1 + l2_2 - l2_0) / (two * l1 * l2),
        (l2_2 + l2_0 - l2_1) / (two * l2 * l0),
        (l2_0 + l2_1 - l2_2) / (two * l0 * l1),
    )


@wp.func
def triangle_aabb(a: Any, b: Any, c: Any) -> tuple[Any, Any]:
    # Lower and upper corners of triangle ABC's axis-aligned bounding box. ``wp.min`` / ``wp.max``
    # on vectors are element-wise.
    return wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))


@wp.func
def segment_aabb(a: Any, b: Any) -> tuple[Any, Any]:
    # Lower and upper corners of segment AB's axis-aligned bounding box -- the input a *segment*
    # BVH is built from, as ``triangle_aabb`` is for a face BVH. Element-wise for the same reason.
    return wp.min(a, b), wp.max(a, b)


@wp.func
def is_in_aabb(point: Any, min_bound: Any, max_bound: Any) -> wp.bool:
    # Is the point inside an axis-aligned box, boundary included? The containment counterpart of
    # ``triangle_aabb`` above: that one builds a box, this one tests against it.
    #
    # Six explicit comparisons rather than the shorter ``wp.min(point, min_bound) == min_bound``,
    # which is the same predicate on finite input and the *wrong* one on a ``nan`` coordinate:
    # ``wp.min`` returns the other operand for a ``nan`` (measured), so the vector form reports a
    # ``nan`` point as inside every box. Each ``>=`` / ``<=`` here is false for a ``nan`` instead,
    # which is both the honest answer and the reference convention -- open3d's
    # ``GetPointIndicesWithinBoundingBox`` compares component-wise for exactly this reason.
    #
    # Generic over the scalar type but fixed at three components: a vector has no readable
    # component count in kernel scope (the same restriction a matrix's ``.shape``
    # has), so a rank-free spelling would have to go back through
    # ``wp.min`` and give up the ``nan`` answer.
    return (
        point[0] >= min_bound[0]
        and point[0] <= max_bound[0]
        and point[1] >= min_bound[1]
        and point[1] <= max_bound[1]
        and point[2] >= min_bound[2]
        and point[2] <= max_bound[2]
    )


@wp.func
def is_strictly_inside_aabb(point: Any, min_bound: Any, max_bound: Any) -> wp.bool:
    # The open-box twin of ``is_in_aabb`` above: every comparison is strict, so a point exactly on
    # the boundary reports outside rather than inside. Same six-explicit-comparisons shape and the
    # same reason -- a ``nan`` coordinate reports false either way, where a ``wp.min``-based
    # shortcut would not.
    return (
        point[0] > min_bound[0]
        and point[0] < max_bound[0]
        and point[1] > min_bound[1]
        and point[1] < max_bound[1]
        and point[2] > min_bound[2]
        and point[2] < max_bound[2]
    )


@wp.func
def is_in_obb(point: Any, rotation: Any, min_bound: Any, max_bound: Any) -> wp.bool:
    # Is the point inside an oriented box, boundary included? Exactly ``is_in_aabb`` in box
    # coordinates -- ``rotation`` is the world-to-box frame whose *rows* are the box axes, the
    # convention ``bounds.oriented_bounding_box`` returns -- so the two share one predicate and one
    # boundary rule rather than spelling the six comparisons twice.
    return is_in_aabb(rotation * point, min_bound, max_bound)


@wp.func
def point_plane_dot(point: Any, plane_normal: Any, plane_origin: Any) -> wp.Float:
    # Signed, *unnormalized* distance from a point to the plane through ``plane_origin`` with
    # normal ``plane_normal``: positive on the normal's side, zero on the plane. Unnormalized
    # because every caller either only reads the sign (``points.half_space_mask``, the slice
    # classifiers) or divides by ``wp.length(plane_normal)`` itself
    # (``points.point_plane_distance``), so normalizing here would be a square root two of the
    # three call paths throw away.
    return wp.dot(point - plane_origin, plane_normal)


# Moller's tribox3, the 13-axis separating-axis test between a triangle and an axis-aligned box,
# and the three helpers it is built from. Concrete ``float32`` rather than scalar-generic like most
# of this module: the box-axis sweep needs a unit vector per axis and a vector literal cannot be
# built at the caller's precision without threading a type witness through all four signatures,
# which "no speculative generality" rules out while ``voxels.mark_surface_voxels``
# remains the only caller and is ``float32``. Generalize when a ``float64`` caller appears.
@wp.func
def axis_interval_projection(axis: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec2:
    p = wp.vec3(wp.dot(axis, v0), wp.dot(axis, v1), wp.dot(axis, v2))
    # Single-argument wp.min / wp.max reduce a vector to its extreme element.
    return wp.vec2(wp.min(p), wp.max(p))


@wp.func
def unit_axis(axis: wp.int32) -> wp.vec3:
    if axis == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if axis == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def plane_box_overlap(normal: wp.vec3, offset: wp.float32, half: wp.vec3) -> wp.bool:
    # Moller's ``planeBoxOverlap``: the plane ``dot(normal, x) == offset`` meets the box
    # ``[-half, half]`` iff ``|offset|`` is within the box's support along ``normal``.
    support = (
        wp.abs(normal[0]) * half[0] + wp.abs(normal[1]) * half[1] + wp.abs(normal[2]) * half[2]
    )
    return wp.abs(offset) <= support


@wp.func
def edge_axes_separate(
    edge: wp.vec3, half: wp.vec3, a0: wp.vec3, a1: wp.vec3, a2: wp.vec3
) -> wp.bool:
    # The three cross-product axes ``e_i x edge`` of Moller's tribox3, written out rather than
    # crossed with a unit vector: ``e_x x (x, y, z) == (0, -z, y)`` and cyclically. A degenerate
    # edge gives a zero axis, whose intervals are both ``[0, 0]`` and therefore never separate.
    axis_x = wp.vec3(0.0, -edge[2], edge[1])
    axis_y = wp.vec3(edge[2], 0.0, -edge[0])
    axis_z = wp.vec3(-edge[1], edge[0], 0.0)
    for a in range(3):
        axis = axis_x
        if a == 1:
            axis = axis_y
        elif a == 2:
            axis = axis_z
        # The two endpoints of ``edge`` project to the same value on ``e_i x edge``, so projecting
        # all three vertices gives the identical interval the AXISTEST_* macros compute from two.
        interval = axis_interval_projection(axis, a0, a1, a2)
        radius = wp.abs(axis[0]) * half[0] + wp.abs(axis[1]) * half[1] + wp.abs(axis[2]) * half[2]
        if interval[0] > radius or interval[1] < -radius:
            return True
    return False


@wp.func
def triangle_aabb_overlap(
    center: wp.vec3, half: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3
) -> wp.bool:
    # Moller's tribox3, the 13-axis separating-axis test between a triangle and an axis-aligned
    # box: the three box face normals, the triangle's own plane, and the nine edge-cross axes. No
    # epsilon, matching Open3D's ``IntersectionTest::TriangleAABB`` (which runs it in ``float64``,
    # so tangency within ``float32`` rounding is where the two can disagree).
    a0 = v0 - center
    a1 = v1 - center
    a2 = v2 - center

    for axis in range(3):
        interval = axis_interval_projection(unit_axis(axis), a0, a1, a2)
        if interval[0] > half[axis] or interval[1] < -half[axis]:
            return False

    edge0 = a1 - a0
    edge1 = a2 - a1
    edge2 = a0 - a2
    if edge_axes_separate(edge0, half, a0, a1, a2):
        return False
    if edge_axes_separate(edge1, half, a0, a1, a2):
        return False
    if edge_axes_separate(edge2, half, a0, a1, a2):
        return False

    normal = wp.cross(edge0, edge1)
    return plane_box_overlap(normal, wp.dot(normal, a0), half)


@wp.func
def vector_angle(a: Any, b: Any) -> wp.Float:
    # Unsigned angle in [0, pi] between two vectors, as ``atan2(|a x b|, a . b)`` rather than
    # ``acos(a . b)``.
    #
    # The atan2 form is the accurate one and the reason this is the single spelling in the tree.
    # ``acos`` has an infinite derivative at +-1, so for nearly-parallel vectors -- the *common*
    # case here: two coplanar faces across an edge, a straight run of a polyline, two near-collinear
    # endpoint normals -- it amplifies the round-off already in the dot product, while the cross
    # product carries the small angle directly. Against a float64 reference on float32 face normals
    # the acos form's worst error over a coplanar cap fan lands *past* the 1e-5 the parity tests
    # compare at, so the old spelling was one fixture away from failing rather than merely less
    # tidy. It is also scale-free, so callers may pass unnormalized vectors (``tris_angle_profit``
    # passes raw cross products) and a zero-length input gives ``atan2(0, 0) == 0`` rather than the
    # spurious pi/2 that ``acos`` of a zeroed ``normalize`` returns.
    return wp.atan2(wp.length(wp.cross(a, b)), wp.dot(a, b))


@wp.func
def dihedral_angle(left_normal: Any, right_normal: Any, edge_vector: Any) -> wp.Float:
    # Signed angle between the two face normals about the shared edge.
    edge_direction = wp.normalize(edge_vector)
    sine = wp.dot(edge_direction, wp.cross(left_normal, right_normal))
    cosine = wp.dot(left_normal, right_normal)
    return wp.atan2(sine, cosine)


@wp.func
def _circumdiameter_sq_from_sides(
    a: Any, b: Any, c: Any, bc: wp.Float, ca: wp.Float, ab: wp.Float
) -> wp.Float:
    # Shared tail of circumcircle_diameter_sq and mincircle_diameter_sq once the caller's own
    # degenerate-side dispatch has ruled out a repeated vertex: the squared circumdiameter from the
    # three squared side lengths and the doubled-area cross product. Zero area (collinear, distinct
    # vertices) means no circumcircle at all.
    f = wp.length_sq(triangle_area_vector(a, b, c))
    if f <= type(ab)(0.0):
        return float_inf(ab)
    return ab * ca * bc / f


@wp.func
def circumcircle_diameter_sq(a: Any, b: Any, c: Any) -> wp.Float:
    # Squared diameter of triangle ABC's circumcircle.
    # A zero-length side collapses to the opposite side -- this is the exact limit of the
    # circumdiameter as that side shrinks to zero (the two endpoints coincide and the circumcircle
    # degenerates to the segment to the third vertex), not a stand-in for "undefined"; only a
    # zero-*area* triangle with distinct vertices has no circumcircle and returns float_inf.
    bc, ca, ab = squared_edge_lengths(a, b, c)
    zero = type(ab)(0.0)
    if ab <= zero:
        return ca
    if ca <= zero:
        return bc
    if bc <= zero:
        return ab
    return _circumdiameter_sq_from_sides(a, b, c, bc, ca, ab)


@wp.func
def circumcircle_diameter(a: Any, b: Any, c: Any) -> wp.Float:
    # Diameter (not squared) of triangle ABC's circumcircle; +inf when degenerate, which the
    # square root preserves.
    return wp.sqrt(circumcircle_diameter_sq(a, b, c))


@wp.func
def delone_metrics(a: Any, b: Any, c: Any, d: Any) -> tuple[wp.Float, wp.Float]:
    # For quadrangle ABCD, the pair of Delone metrics compared when deciding whether to flip the
    # diagonal AC to BD: each is the larger circumcircle of the two triangles that diagonal makes.
    # Either may be infinite (a degenerate triangle), so callers test with ``wp.isinf`` rather than
    # subtracting blindly.
    metric_ac = wp.max(circumcircle_diameter_sq(a, c, d), circumcircle_diameter_sq(c, a, b))
    metric_bd = wp.max(circumcircle_diameter_sq(b, d, a), circumcircle_diameter_sq(d, b, c))
    return metric_ac, metric_bd


@wp.func
def mincircle_diameter_sq(a: Any, b: Any, c: Any) -> wp.Float:
    # Squared diameter of the smallest circle enclosing triangle ABC: for an obtuse triangle that
    # is the circle on the longest side, otherwise it is the circumcircle.
    bc, ca, ab = squared_edge_lengths(a, b, c)
    if ca >= bc + ab:
        return ca
    if bc >= ab + ca:
        return bc
    if ab >= ca + bc:
        return ab
    return _circumdiameter_sq_from_sides(a, b, c, bc, ca, ab)


@wp.func
def triangle_aspect_ratio(a: Any, b: Any, c: Any) -> wp.Float:
    # Circum-radius over twice the in-radius, ``abc / (8 (s - a)(s - b)(s - c))``. Grows without
    # bound for slivers, so a degenerate triangle returns +inf.
    #
    # The Heron product is formed as ``8 A^2 / s`` from the area vector rather than from the
    # differences ``s - a``: on a needle those cancel, and in float32 that read
    # ``saddle_graded``'s aspect-2 400 faces up to 0.22 % off float64 (3.7e-7 now; the reciprocal
    # ``triangles.triangle_radius_ratio`` 3.4e-5 before). ``8 A^2`` is ``2 |area vector|^2``.
    bc, ca, ab = side_lengths(a, b, c)
    double_area_sq = wp.length_sq(triangle_area_vector(a, b, c))
    if double_area_sq <= type(bc)(0.0):
        return float_inf(bc)
    half_perimeter = (bc + ca + ab) / type(bc)(2.0)
    return bc * ca * ab * half_perimeter / (type(bc)(2.0) * double_area_sq)


@wp.func
def unfold_on_plane(b: Any, c: Any, d: Any, to_left: wp.bool):
    # Place ``c`` in the plane relative to the already-placed 2D point ``d``, preserving the angle
    # and length of the 3D pair (b, c).
    dot_bc = wp.dot(b, c)
    cross_bc = wp.length(wp.cross(b, c))
    dd = wp.length_sq(d)
    zero = type(dd)(0.0)
    if dd <= zero:
        return d * zero
    # Initialised before the branch: a variable assigned only inside an ``if`` is uninitialised in
    # Warp when the branch is not taken.
    orthogonal = type(d)(-d[1], d[0])
    if not to_left:
        orthogonal = type(d)(d[1], -d[0])
    return (dot_bc * d + cross_bc * orthogonal) / dd


@wp.func
def line_isect(b: Any, c: Any, d: Any) -> wp.Float:
    # Parameter along segment 0->B where it meets line C-D.
    c1 = cross2(d, c)
    c2 = cross2(c - b, d - b)
    zero = type(c1)(0.0)
    if c1 == zero and c2 == zero:
        bb = wp.length_sq(b)
        if bb == zero:
            return zero
        return (wp.dot(c, b) + wp.dot(d, b)) / (type(c1)(2.0) * bb)
    cc = c1 + c2
    if cc == zero:
        return zero
    return c1 / cc


@wp.func
def is_unfold_quadrangle_convex(a: Any, b: Any, c: Any, d: Any) -> wp.bool:
    # Unfold triangles ABC and ACD into a common plane; the quadrangle is convex exactly when the
    # shortest B->D path crosses diagonal AC strictly between A and C. ``wp.vector`` seeds B's 2D
    # image at the caller's precision, which is how this stays generic despite mapping
    # vec3 -> vec2 / vec3d -> vec2d.
    unfold_b = wp.vector(wp.length(b - a), type(a[0])(0.0))
    unfold_c = unfold_on_plane(b - a, c - a, unfold_b, wp.bool(True))
    unfold_d = unfold_on_plane(c - a, d - a, unfold_c, wp.bool(True))
    x = line_isect(unfold_c, unfold_b, unfold_d)
    # Clamping ``x`` to [0, 1] first is a common formulation and cannot change this strict test,
    # so it is dropped.
    return x > type(x)(0.0) and x < type(x)(1.0)


@wp.func
def project_out_normal(vector: Any, normal: Any):
    # Component of ``vector`` in the plane orthogonal to unit ``normal``.
    return vector - wp.dot(vector, normal) * normal


@wp.func
def unit_tangent(vector: Any, normal: Any, tolerance: Any) -> tuple[Any, wp.Float]:
    # Project ``vector`` into the plane orthogonal to unit ``normal`` and normalize it, returning
    # the projection's length alongside.
    #
    # The length is returned rather than folded into a fallback because every caller wants a
    # different one when it is degenerate: keep a precomputed perpendicular, return the unprojected
    # input, skip the contribution entirely, or report failure. Callers compare the length against
    # their own tolerance and branch; when it is above tolerance the first result is the unit
    # tangent, and when it is not the first result is the *unnormalized* projection.
    tangential = project_out_normal(vector, normal)
    length = wp.length(tangential)
    if length > tolerance:
        return tangential / length, length
    return tangential, length


@wp.func
def world_to_tangent(value: Any, basis_x: Any, basis_y: Any):
    # A world-space vector's components in a vertex's (orthonormal) tangent basis.
    #
    # Sits beside ``project_out_normal`` / ``unit_tangent`` because it is the third member of the
    # same family: those two take a vector *to* the tangent plane, this one names it *in* the
    # plane's own coordinates. Three callers had the two dot products spelled out --
    # ``heat.splat_curve_normals`` (widened to float64 for the vector-diffusion source),
    # ``heat.world_to_tangent_unit`` (normalized) and ``geodesic_walk.start_direction_at_vertex``
    # (read as a polar angle) -- and the component *order* is the decision they share: it fixes
    # the sign of every angle taken against the frame, and a transposed copy still returns a
    # plausible tangent vector.
    #
    # No normalization and no tolerance: each caller floors against a different thing (a relative
    # fraction of its own field maximum, or nothing at all), so the guard is theirs. Generic over
    # the precision for the usual reason -- the callers hold float32 frames and want a float32
    # result, and widening is ``array.to_vec2d`` afterwards rather than a second body.
    #
    # The inverse is [`heat.tangent_to_world`][ordito.kernels.heat.tangent_to_world], which stays
    # in the module that publishes it as a ``wp.map`` op.
    return wp.vector(wp.dot(value, basis_x), wp.dot(value, basis_y))


@wp.func
def normalize_or_zero(v: Any, tolerance: Any):
    length = wp.length(v)
    if length <= tolerance:
        return type(v)()
    return v / length


@wp.func
def stable_length(v: Any):
    """
    ``length(v)`` without underflow: the vector is scaled by its largest component first.

    ``wp.length`` squares the components, so any vector shorter than about ``1e-154`` in
    ``float64`` reads as zero. The heat method's far field is far smaller than that -- its
    diffused values fall by a near-constant factor per ring of vertices -- and
    ``igl::heat_geodesics_solve`` takes the same max-scaled norm for the same reason.
    """
    largest = wp.max(wp.abs(v))
    if largest == type(largest)(0.0):
        return largest
    return largest * wp.length(v / largest)


@wp.func
def stable_normalize(v: Any):
    """``normalize(v)`` without underflow (``stable_length``); zero for an exactly zero ``v``."""
    largest = wp.max(wp.abs(v))
    if largest == type(largest)(0.0):
        return type(v)()
    return wp.normalize(v / largest)


@wp.func
def angle_defect(angle_sum: wp.Float) -> wp.Float:
    """Angle defect at a vertex: a full turn minus the incident corner angles."""
    return type(angle_sum)(TWO_PI_F64) - angle_sum


@wp.func
def barycentric_gram(a: Any, b: Any, c: Any, p: Any) -> tuple[wp.Float, wp.Float, wp.Float]:
    """
    Cramer's rule on the Gram system of triangle ABC's two edge vectors, undivided.

    Returns ``(numerator_1, numerator_2, determinant)``: the barycentric coordinates of ``p`` are
    ``(det - n1 - n2, n1, n2) / det``, and when ``p`` is out of plane they are those of its
    orthogonal projection into the triangle's plane. **Dimension-generic** -- ``wp.length_sq`` and
    ``wp.dot`` say nothing about the ambient dimension, so this serves ``wp.vec2`` and ``wp.vec3``
    from one body, which is why it exists.

    Undivided, and therefore guard-free, so that a caller can apply its own degenerate policy:
    [`barycentric_2d`][ordito.kernels.predicates.barycentric_2d], its one caller today, wants a
    definite answer for a degenerate triangle so that ``min(b) >= -eps`` rejects it without a
    separate area check.

    **Do not reach for this to build a 3-D barycentric solve.** The Gram determinant is the
    ill-conditioned way to write a triangle's squared area: its two terms agree to more digits as
    the corner angle closes, so the subtraction loses them and the coordinates come back ``nan`` on
    a sliver that still has positive area.
    [`point_barycentric`][ordito.kernels.triangles.point_barycentric] forms the same quantity as
    ``|e0 x e1|^2`` instead. The 2-D containment test is a different use, where ``p`` and the
    triangle are coplanar by construction and the degenerate case is one the caller wants reported
    rather than conditioned away.
    """
    e0 = b - a
    e1 = c - a
    w = p - a
    d00 = wp.length_sq(e0)
    d01 = wp.dot(e0, e1)
    d11 = wp.length_sq(e1)
    d20 = wp.dot(w, e0)
    d21 = wp.dot(w, e1)
    return d11 * d20 - d01 * d21, d00 * d21 - d01 * d20, d00 * d11 - d01 * d01


@wp.func
def barycentric_2d(q0: wp.vec2, q1: wp.vec2, q2: wp.vec2, p: wp.vec2) -> wp.vec3:
    """
    Barycentric coordinates ``(b0, b1, b2)`` of ``p`` in the 2D triangle ``(q0, q1, q2)``.

    Returns a vector with a negative component for degenerate triangles, so a caller testing
    ``min(b) >= -eps`` for containment rejects them without a separate area check.
    """
    n1, n2, denom = barycentric_gram(q0, q1, q2, p)
    if wp.abs(denom) < wp.float32(1e-20):
        return wp.vec3(-1.0, -1.0, -1.0)
    inverse_denominator = 1.0 / denom
    b1 = n1 * inverse_denominator
    b2 = n2 * inverse_denominator
    b0 = 1.0 - b1 - b2
    return wp.vec3(b0, b1, b2)


@wp.func
def plane_crossing_span(distance: wp.vec3d, coordinate: wp.vec3d) -> tuple[wp.bool, wp.vec2d]:
    # Where a triangle meets the other triangle's plane, as an interval along the two planes'
    # intersection line. ``distance`` holds its three vertices' signed distances to that plane and
    # ``coordinate`` their positions along the line's direction; each edge whose endpoints straddle
    # the plane contributes one crossing, interpolated at the same ratio.
    #
    # A vertex exactly *on* the plane contributes itself, which only matters for a configuration the
    # caller has already excluded -- it early-outs unless the distances genuinely straddle -- so it
    # is here for the one live case: a vertex on the plane with the other two on opposite sides,
    # which is a real crossing.
    lo = FLOAT64_INF_CONSTANT
    hi = -FLOAT64_INF_CONSTANT
    zero = wp.float64(0.0)
    for i in range(3):
        j = (i + 1) % 3
        first = distance[i]
        second = distance[j]
        if first == zero:
            lo = wp.min(lo, coordinate[i])
            hi = wp.max(hi, coordinate[i])
        if first * second < zero:
            weight = first / (first - second)
            crossing = coordinate[i] + weight * (coordinate[j] - coordinate[i])
            lo = wp.min(lo, crossing)
            hi = wp.max(hi, crossing)
    return lo <= hi, wp.vec2d(lo, hi)


@wp.func
def _triangles_intersect_d(
    da0: wp.vec3d, da1: wp.vec3d, da2: wp.vec3d, db0: wp.vec3d, db1: wp.vec3d, db2: wp.vec3d
) -> wp.bool:
    # Moller's interval test itself, taking the already-widened float64 corners. Split out of
    # triangles_intersect so a caller that needs those corners afterward regardless of the verdict
    # (triangle_triangle_distance_sq, which falls through to a float64 distance solve on a `False`)
    # converts its six vertices once rather than once per call.
    zero = wp.float64(0.0)

    normal_a = wp.cross(da1 - da0, da2 - da0)
    distance_b = wp.vec3d(
        wp.dot(normal_a, db0 - da0), wp.dot(normal_a, db1 - da0), wp.dot(normal_a, db2 - da0)
    )
    # Entirely in one closed half-space of A's plane: separated, coplanar, or touching at most.
    if wp.min(distance_b) >= zero or wp.max(distance_b) <= zero:
        return False

    normal_b = wp.cross(db1 - db0, db2 - db0)
    distance_a = wp.vec3d(
        wp.dot(normal_b, da0 - db0), wp.dot(normal_b, da1 - db0), wp.dot(normal_b, da2 - db0)
    )
    if wp.min(distance_a) >= zero or wp.max(distance_a) <= zero:
        return False

    # Both straddle, so the planes are neither parallel nor coincident and this cannot vanish.
    direction = wp.cross(normal_a, normal_b)
    valid_a, span_a = plane_crossing_span(
        distance_a, wp.vec3d(wp.dot(direction, da0), wp.dot(direction, da1), wp.dot(direction, da2))
    )
    valid_b, span_b = plane_crossing_span(
        distance_b, wp.vec3d(wp.dot(direction, db0), wp.dot(direction, db1), wp.dot(direction, db2))
    )
    if not valid_a or not valid_b:
        return False
    return span_a[0] <= span_b[1] and span_b[0] <= span_a[1]


@wp.func
def triangles_intersect(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> wp.bool:
    # Do two triangles cross transversally? Moller's interval test: each triangle is cut by the
    # other's plane into an interval along the planes' intersection line, and they intersect exactly
    # when those two intervals overlap. Coplanar and merely touching configurations are **not**
    # intersections here, which is the "contact is not intersection" convention
    # ``validation.face_self_intersecting_mask`` documents.
    #
    # **Not a separating-axis test, and the reason is exactness rather than speed.** SAT over two
    # triangles is only exact while every axis is non-degenerate, and an edge-edge cross product
    # **vanishes for parallel edges** -- which a regular grid is full of. Projecting onto the zero
    # axis collapses every interval to ``[0, 0]``, which reads as "overlapping", so a pair separated
    # only along such an axis is reported as intersecting: against an exact float64 arbiter that was
    # half the flagged faces on a self-intersecting torus. This is slightly *slower* than the SAT
    # (a data-dependent edge loop in ``plane_crossing_span`` against straight-line arithmetic), and
    # that is the right trade: both sit on the launch floor at real candidate counts, and the faster
    # one was answering a different question.
    #
    # **The arithmetic is float64 on float32 inputs**, and that is load-bearing. Widening a float32
    # is lossless, so this is the same geometry; what the extra precision buys is the *decisions* --
    # the sign of a plane distance, and the comparison of two intervals. Audited pair by pair over a
    # self-intersecting surface's broad-phase candidates, the float32 kernel disagreed with this
    # same algorithm in float64 several times more often than two float32 implementations with
    # different association orders disagreed with each other, which is what says the residual was
    # arithmetic and not logic. It roughly doubles this kernel and is a few percent end to end, the
    # narrow phase being a small share of ``face_self_intersecting_mask``.
    return _triangles_intersect_d(
        kernel_array.to_vec3d(a0),
        kernel_array.to_vec3d(a1),
        kernel_array.to_vec3d(a2),
        kernel_array.to_vec3d(b0),
        kernel_array.to_vec3d(b1),
        kernel_array.to_vec3d(b2),
    )


@wp.func
def segment_coordinate(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.float32:
    """Clamped projection parameter of ``p`` onto segment ``a -> b`` in ``[0, 1]``."""
    ab = b - a
    length_sq = wp.max(wp.length_sq(ab), TOLERANCE_MERGE_CONSTANT)
    return wp.clamp(wp.dot(p - a, ab) / length_sq, 0.0, 1.0)


@wp.func
def closest_point_on_segment(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Point on segment ``a -> b`` closest to ``p``."""
    return wp.lerp(a, b, segment_coordinate(a, b, p))


@wp.func
def point_to_segment_distance(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.float32:
    """Euclidean distance from ``p`` to the closest point on segment ``a -> b``."""
    return wp.length(p - closest_point_on_segment(a, b, p))


@wp.func
def segment_segment_distance_sq(
    p0: wp.vec3d, p1: wp.vec3d, q0: wp.vec3d, q1: wp.vec3d
) -> wp.float64:
    # Squared distance between two closed segments (Ericson, *Real-Time Collision Detection*). The
    # clamped two-parameter solve rather than the closest-point pair, because every caller here
    # wants the magnitude and the pair costs two more lerps.
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    d0 = p1 - p0
    d1 = q1 - q0
    r = p0 - q0
    a = wp.dot(d0, d0)
    e = wp.dot(d1, d1)
    f = wp.dot(d1, r)
    s = zero
    t = zero
    if a <= TOLERANCE_ZERO_F64 and e <= TOLERANCE_ZERO_F64:
        return wp.length_sq(r)  # both degenerate to points
    if a <= TOLERANCE_ZERO_F64:
        t = wp.clamp(f / e, zero, one)
    else:
        c = wp.dot(d0, r)
        if e <= TOLERANCE_ZERO_F64:
            s = wp.clamp(-c / a, zero, one)
        else:
            b = wp.dot(d0, d1)
            denominator = a * e - b * b
            if denominator > TOLERANCE_ZERO_F64:
                s = wp.clamp((b * f - c * e) / denominator, zero, one)
            t = (b * s + f) / e
            # Clamping ``t`` moves the optimum, so ``s`` is re-solved against the clamped value --
            # skipping this is the classic parallel-segment error.
            if t < zero:
                t = zero
                s = wp.clamp(-c / a, zero, one)
            elif t > one:
                t = one
                s = wp.clamp((b - c) / a, zero, one)
    return wp.length_sq((p0 + d0 * s) - (q0 + d1 * t))


@wp.func
def point_triangle_offset(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.vec3d:
    # ``p`` minus its closest point on a closed triangle, by the seven-region barycentric test. A
    # degenerate triangle falls through to its edges, which is why the vertex and edge regions are
    # tested before the interior one rather than after. The offset rather than the point, so the
    # distance below is formed from the same differences it always was.
    zero = wp.float64(0.0)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= zero and d2 <= zero:
        return ap

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= zero and d4 <= d3:
        return bp

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= zero and d5 <= d6:
        return cp

    vc = d1 * d4 - d3 * d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        return ap - ab * (d1 / (d1 - d3))
    vb = d5 * d2 - d1 * d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        return ap - ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= zero and (d4 - d3) >= zero and (d5 - d6) >= zero:
        return bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))

    denominator = va + vb + vc
    if denominator <= TOLERANCE_ZERO_F64:
        return ap  # degenerate: the edge cases above already covered it
    return ap - ab * (vb / denominator) - ac * (vc / denominator)


@wp.func
def point_triangle_distance_sq(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    """Squared distance from ``p`` to the closed triangle ``a, b, c``."""
    return wp.length_sq(point_triangle_offset(p, a, b, c))


@wp.func
def triangle_triangle_distance_sq(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> wp.float32:
    # Squared distance between two closed triangles: zero when they cross, otherwise the smallest of
    # nine edge-edge distances and six point-triangle distances. Those fifteen are exhaustive for
    # *disjoint* triangles and all strictly positive for *crossing* ones, which is why the
    # intersection test is not an optimization -- without it a pair that genuinely meets reports the
    # distance between their boundaries instead of zero.
    #
    # ``float64`` inside on ``float32`` input, the same choice ``triangles_intersect`` documents and
    # for the same reason: widening is lossless, and what the precision buys is the *decisions* --
    # here the region tests and the clamped parameter solves, each of which is a comparison of
    # differences of products.
    #
    # Converted once: calling ``triangles_intersect`` here would widen all six corners a second
    # time on the common disjoint-triangle path (the intersection test itself already widens and
    # discards its own copies), so the intersection test runs on these corners via
    # ``_triangles_intersect_d`` instead of going through the ``wp.vec3``-taking wrapper. This is a
    # duplication fix and not a measured speedup -- the kernel's cost is the broad-phase BVH walk
    # and the fifteen edge/point distance solves, not six scalar casts.
    p0 = kernel_array.to_vec3d(a0)
    p1 = kernel_array.to_vec3d(a1)
    p2 = kernel_array.to_vec3d(a2)
    q0 = kernel_array.to_vec3d(b0)
    q1 = kernel_array.to_vec3d(b1)
    q2 = kernel_array.to_vec3d(b2)
    if _triangles_intersect_d(p0, p1, p2, q0, q1, q2):
        return wp.float32(0.0)

    best = segment_segment_distance_sq(p0, p1, q0, q1)
    best = wp.min(best, segment_segment_distance_sq(p0, p1, q1, q2))
    best = wp.min(best, segment_segment_distance_sq(p0, p1, q2, q0))
    best = wp.min(best, segment_segment_distance_sq(p1, p2, q0, q1))
    best = wp.min(best, segment_segment_distance_sq(p1, p2, q1, q2))
    best = wp.min(best, segment_segment_distance_sq(p1, p2, q2, q0))
    best = wp.min(best, segment_segment_distance_sq(p2, p0, q0, q1))
    best = wp.min(best, segment_segment_distance_sq(p2, p0, q1, q2))
    best = wp.min(best, segment_segment_distance_sq(p2, p0, q2, q0))

    best = wp.min(best, point_triangle_distance_sq(p0, q0, q1, q2))
    best = wp.min(best, point_triangle_distance_sq(p1, q0, q1, q2))
    best = wp.min(best, point_triangle_distance_sq(p2, q0, q1, q2))
    best = wp.min(best, point_triangle_distance_sq(q0, p0, p1, p2))
    best = wp.min(best, point_triangle_distance_sq(q1, p0, p1, p2))
    best = wp.min(best, point_triangle_distance_sq(q2, p0, p1, p2))
    return wp.float32(best)


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense = map_probe
    declare_map_signatures([(angle_defect, (dense(wp.float32),), wp.float32)])


_declare_map_kernels()
