"""Kernels for closing boundary holes and stitching two rims (``ordito.holes``)."""

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT, INT32_MAX_CONSTANT
from ordito.kernels import array as kernel_array
from ordito.kernels.array import (
    loop_next_slot,
    loop_point,
    loop_rim_edge_vertices,
    pack_nearest_key,
    update_argmin,
)
from ordito.kernels.array import wrap_index as _wrap
from ordito.kernels.halfedge import halfedge_endpoints, halfedge_prev
from ordito.kernels.predicates import (
    circumcircle_diameter,
    dihedral_angle,
    project_out_normal,
    side_lengths,
    triangle_aspect_ratio,
    triangle_double_area,
)
from ordito.kernels.reduce import block_argmin, block_barrier, block_min
from ordito.kernels.triangles import corner_triple

wp.set_module_options({"enable_backward": False})

# Big-but-finite penalty for a triangulation the metric rejects: lets the DP keep a bad
# triangulation rather than break entirely, while staying below ``float`` precision limits
# when summed.
BAD_METRIC = wp.constant(wp.float32(1e10))

# Fill-metric selectors for ``triangle_fill_metric`` / ``fill_edge_term`` / ``fill_dp_span``.
METRIC_PLANE_NORMALIZED = wp.constant(wp.int32(0))
METRIC_MIN_AREA = wp.constant(wp.int32(1))
METRIC_CIRCUMSCRIBED = wp.constant(wp.int32(2))
METRIC_PLANE = wp.constant(wp.int32(3))
METRIC_MIN_TRI_ANGLE = wp.constant(wp.int32(4))
METRIC_EDGE_LENGTH = wp.constant(wp.int32(5))
METRIC_UNIVERSAL = wp.constant(wp.int32(6))
METRIC_MAX_DIHEDRAL = wp.constant(wp.int32(7))
METRIC_COMPLEX_FILL = wp.constant(wp.int32(8))

# How per-triangle / per-edge terms accumulate across the triangulation.
COMBINE_SUM = wp.constant(wp.int32(0))
COMBINE_MAX = wp.constant(wp.int32(1))

# sin of 60 degrees = sqrt(3)/2, the maximal possible minimal-angle sine (equilateral triangle).
MAX_MIN_ANGLE_SIN = wp.constant(wp.float32(0.86602540378443864676))


# One convention for a loop's extent, and this is it: the package's total-terminated
# ``loop_offsets``, uploaded once by ``holes._PackedLoops`` -- loop ``ell`` is
# ``[loop_offsets[ell], loop_offsets[ell + 1])``. Kernels here must not re-derive the size from a
# ``total`` and an ``n_loops``, or the file answers one question two ways and a new kernel picks a
# third.


@wp.kernel
def fan_faces(
    flat_loops: wp.array[wp.int32], loop_offsets: wp.array[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    ell = wp.int32(wp.tid())
    o = loop_offsets[ell]
    s = loop_offsets[ell + 1] - o
    # This loop contributes s - 2 fan triangles; earlier loops occupy o - 2 * ell of them.
    base = o - 2 * ell
    for k in range(1, s - 1):
        t = base + (k - 1)
        out_faces[3 * t + 0] = flat_loops[o]
        out_faces[3 * t + 1] = flat_loops[o + k + 1]
        out_faces[3 * t + 2] = flat_loops[o + k]


@wp.kernel
def cone_fill(
    vertices: wp.array[wp.vec3],
    flat_loops: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_centroids: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # One thread per hole: place the loop's apex at its centroid and emit the cone fan around it.
    # The two were separate launches over the same loops; the fan does not read the centroid (the
    # apex is an *index*, ``n_vertices + ell``), so nothing is shared but the loop's own bounds --
    # which is exactly what makes one launch enough.
    ell = wp.int32(wp.tid())
    o = loop_offsets[ell]
    s = loop_offsets[ell + 1] - o
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(s):
        acc = acc + vertices[flat_loops[o + j]]
    out_centroids[ell] = acc * (1.0 / wp.float32(s))

    apex = n_vertices + ell
    # This loop contributes s cone triangles; the cone base equals o (scan of the loop sizes).
    for j in range(s):
        t = o + j
        nxt = o + (j + 1) % s
        out_faces[3 * t + 0] = apex
        out_faces[3 * t + 1] = flat_loops[nxt]
        out_faces[3 * t + 2] = flat_loops[o + j]


@wp.func
def min_triangle_angle_sin(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # sin of the smallest angle = shortest edge / circumcircle diameter.
    bc, ca, ab = side_lengths(a, b, c)
    if ab <= 0.0 or ca <= 0.0 or bc <= 0.0:
        return 0.0
    f = triangle_double_area(a, b, c)
    return f * wp.min(wp.vec3(ab, ca, bc)) / (ab * ca * bc)


@wp.func
def triangle_fill_metric(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    plane_normal: wp.vec3,
    char_area: wp.float32,
    metric_id: wp.int32,
) -> wp.float32:
    # Per-triangle term of each fill metric (0 for the purely edge-based metrics).
    if metric_id == METRIC_MIN_AREA:
        return triangle_double_area(a, b, c)
    if metric_id == METRIC_CIRCUMSCRIBED:
        return circumcircle_diameter(a, b, c)
    if metric_id == METRIC_MIN_TRI_ANGLE:
        return wp.exp(25.0 * (MAX_MIN_ANGLE_SIN - min_triangle_angle_sin(a, b, c)))
    if metric_id == METRIC_EDGE_LENGTH or metric_id == METRIC_MAX_DIHEDRAL:
        return 0.0
    if metric_id == METRIC_UNIVERSAL:
        return circumcircle_diameter(a, b, c)
    if metric_id == METRIC_COMPLEX_FILL:
        aspect_ratio = triangle_aspect_ratio(a, b, c)
        if aspect_ratio > BAD_METRIC:
            return BAD_METRIC
        # 1e2 is the empirical area weight; char_area == 1 / maxEdgeLengthSq.
        return aspect_ratio + 100.0 * triangle_double_area(a, b, c) * char_area
    if metric_id == METRIC_PLANE:
        if wp.dot(plane_normal, wp.cross(b - a, c - a)) < 0.0:
            return BAD_METRIC
        return circumcircle_diameter(a, b, c)
    # METRIC_PLANE_NORMALIZED (the default): penalize triangles
    # flipped or tilted > 60 degrees off the hole plane, and thin slivers.
    #
    # At source level the three calls below form the same three edge differences twice, take the
    # same three dot products twice and the same cross product twice -- roughly 38 of ~90 flops in
    # the innermost function of an ``O(B^3)`` loop. **nvcc already removes all but three of them**,
    # which is worth knowing because the duplicates sit in *different basic blocks*
    # (``circumcircle_diameter_sq`` and ``triangle_aspect_ratio`` each carry early returns), so
    # their elimination needs partial-redundancy elimination rather than local CSE and there was no
    # reason to assume it. Confirmed by hand-fusing this branch and diffing the regenerated PTX:
    # three subtractions out of ~1 100 arithmetic instructions, which is not a speed change, and
    # the fused form is a twenty-line block where this is three named calls. Do not re-propose the
    # fusion without a PTX diff that says something different.
    face_norm = wp.cross(b - a, c - a)
    face_dbl_area_sq = wp.length_sq(face_norm)
    if face_dbl_area_sq == 0.0:
        return BAD_METRIC
    dot_res = wp.dot(plane_normal, face_norm)
    if dot_res < 0.0 or dot_res * dot_res * 4.0 < face_dbl_area_sq:
        return BAD_METRIC
    aspect_ratio = triangle_aspect_ratio(a, b, c)
    if aspect_ratio > BAD_METRIC:
        return BAD_METRIC
    return circumcircle_diameter(a, b, c) * aspect_ratio


@wp.func
def fill_edge_term(
    a: wp.vec3, b: wp.vec3, lft: wp.vec3, rgt: wp.vec3, metric_id: wp.int32
) -> wp.float32:
    # Per-edge term for edge a->b, with opposite apexes lft (left) and rgt (right); 0 for the
    # triangle-only metrics.
    if metric_id == METRIC_EDGE_LENGTH:
        return wp.length(b - a)
    ab = b - a
    if metric_id == METRIC_UNIVERSAL:
        norm_l = wp.cross(lft - a, ab)
        norm_r = wp.cross(ab, rgt - a)
        dbl_area = wp.length(norm_l) + wp.length(norm_r)
        return wp.sqrt(dbl_area) * wp.exp(5.0 * wp.abs(dihedral_angle(norm_l, norm_r, ab)))
    if metric_id == METRIC_MAX_DIHEDRAL:
        norm_l = wp.cross(lft - a, ab)
        norm_r = wp.cross(ab, rgt - a)
        return wp.abs(dihedral_angle(norm_l, norm_r, ab))
    if metric_id == METRIC_COMPLEX_FILL:
        norm_a = wp.cross(rgt - b, -ab)
        norm_c = wp.cross(lft - a, ab)
        denom = wp.length(norm_a) * wp.length(norm_c)
        if denom == 0.0:
            return BAD_METRIC
        cos_ac = wp.dot(norm_a, norm_c) / denom
        if cos_ac <= -1.0:
            return BAD_METRIC
        t = (1.0 - cos_ac) / (1.0 + cos_ac)
        return t * t * t * t
    return 0.0


@wp.func
def combine_metric(accumulated: wp.float32, term: wp.float32, combine_id: wp.int32) -> wp.float32:
    if combine_id == COMBINE_MAX:
        return wp.max(accumulated, term)
    return accumulated + term


@wp.func
def char_area_from_max(max_edge_sq: wp.float32) -> wp.float32:
    # Characteristic-area scale for the complex-fill metric: 1 / maxEdgeLengthSq, or 1 for a
    # fully degenerate rim.
    if max_edge_sq > 0.0:
        return 1.0 / max_edge_sq
    return 1.0


@wp.kernel
def loop_rim_metrics(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    base: wp.uint64,
    out_max_edge_sq: wp.array[wp.float32],
    out_normal: wp.array[wp.vec3],
    out_loop_pos: wp.array[wp.vec3],
    out_keys: wp.array[wp.uint64],
) -> None:
    # One thread per rim vertex of every loop at once. Each thread owns the rim edge leaving its
    # vertex and folds it into its loop's two per-loop scalars: the longest edge (segmented max,
    # the ``char_area`` scale) and the Newell normal sum (segmented sum, the hole plane). The
    # per-loop equivalents are ``od.reduce.max`` over a gathered rim and
    # ``od.polyline.polyline_normal``, each of which costs a host synchronization per loop.
    #
    # The same pass writes the two per-slot tables the fill reads next, since it has already
    # loaded both: the rim vertex's position (the gather ``loop_pos`` would otherwise be) and the
    # edge's undirected key, which is ``rim_edge_keys``' output for the rim-opposite probe.
    t = wp.int32(wp.tid())
    ell = loop_id[t]
    u, v = loop_rim_edge_vertices(flat_loops, loop_id, loop_offsets, t)
    a = vertices[u]
    c = vertices[v]
    out_loop_pos[t] = a
    out_keys[t] = kernel_array.pack_edge_key(u, v, base)
    wp.atomic_max(out_max_edge_sq, ell, wp.length_sq(c - a))
    wp.atomic_add(out_normal, ell, wp.cross(a, c))


@wp.kernel
def finalize_rim_metrics(
    max_edge_sq: wp.array[wp.float32],
    newell_sums: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
    out_char_areas: wp.array[wp.float32],
) -> None:
    # The two per-loop results ``loop_rim_metrics`` accumulated, finished in one pass over the
    # loops: the Newell sum normalized into the hole plane's unit normal, and the longest edge
    # turned into the ``char_area`` scale.
    ell = wp.int32(wp.tid())
    out_normals[ell] = wp.normalize(newell_sums[ell])
    out_char_areas[ell] = char_area_from_max(max_edge_sq[ell])


@wp.kernel
def init_dp_base(
    loop_offsets: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    active: wp.array[wp.int32],
    out_dp: wp.array[wp.float32],
    out_prev: wp.array[wp.int32],
) -> None:
    # dp[i, j] = min metric to triangulate the sub-polygon on chord (i, j) and the arc i..j;
    # adjacent spans (rim edges) cost 0, everything else starts unfilled. The ragged table packs
    # every loop's ``B x B`` block end to end, so loop ``ell`` owns
    # ``out_dp[dp_offsets[ell] + i * B + j]``. Launched over ``(n_loops, max_B)`` with one thread
    # per row: rows past a loop's own ``B`` exit immediately.
    ell, i = wp.tid()
    if active[ell] == 0:
        return
    b = loop_offsets[ell + 1] - loop_offsets[ell]
    if i >= b:
        return
    base = dp_offsets[ell]
    row = base + i * b
    for j in range(b):
        out_prev[row + j] = -1
        # ``j == i - 1`` is the mirror of rim edge ``(i - 1, i)``: see ``store_dp``.
        if j == i + 1 or j == i - 1:
            out_dp[row + j] = 0.0
        else:
            out_dp[row + j] = BAD_METRIC


# Slots of the min-weight fill's one ``wp.int32`` state buffer, which the host reads twice: whether
# any loop needs the ``min_area`` retry (``flag_bad_triangulations``), and how many triangles the
# traceback fell short of the padded ``B - 2`` per loop (``traceback_fill_triangles``). One
# allocation serves both, since the second read is after the first.
FILL_STATE_RETRY = wp.constant(0)
FILL_STATE_SHORTFALL = wp.constant(1)
FILL_STATE_SLOTS = 2


# Lanes per block of ``fill_dp_span_tiled``: one block owns one interval and its lanes stride the
# apex loop. Short spans leave lanes idle, which costs nothing the one-thread-per-interval kernel
# was not already wasting on its ~1 024-wide grid.
#
# **Two values, and the choice is an occupancy question rather than a rim-length one.** The grid is
# ``(n_loops, max_size - span)``, so a *wide* block only pays when that grid on its own would
# starve the device -- which is CLAUDE.md section 2.3's block-per-item rule in a second place, and
# getting it backwards costs over 10 %. Measured with the emitted face buffer identical at every
# block size: the wide block wins at one or two long rims and **loses from a few dozen rims up,
# whatever the rim length** -- the case with the longest spans and a hundred of them is the worst
# loss, which is what rules out keying on rim length alone. Hence both conditions in
# ``hole_dp_block``: a long rim for the lanes to have work, and few enough rims that the grid needs
# them. Within the winning corner the size of the win is mesh-dependent, so treat the thresholds as
# a floor rather than a formula.
#
# Two notes on method, each of which reversed a conclusion here. The previous reading of this knob,
# "flat between 32 and 128", was taken on Warp 1.16 and no longer holds on 1.17 -- re-probe a tuning
# constant after an upgrade rather than trusting the comment. And a first version of this cut keyed
# on the rim length alone; it was caught by an A/B on a *scattered* deleted region, not by the
# benchmark rows, so a many-rim case belongs in any future sweep of it. The old tie-break reason
# still stands on its own terms -- a single-warp block takes the ``warp_count == 1`` fast path in
# Warp's ``tile_reduce_impl`` -- it is simply outweighed when the grid is narrow and the apex loop
# long.
HOLE_DP_BLOCK = 32
HOLE_DP_BLOCK_LONG = 128
# Longest rim at or above which the wide block can pay (measured flat for both at exactly 256)...
HOLE_DP_LONG_RIM = 256
# ...and the rim count above which the grid no longer needs it. Between 8 and 32 in the table above;
# placed at the low end because the losses past it are consistent and the wins below it are not.
HOLE_DP_WIDE_GRID_LOOPS = 4


# Spans per recorded CUDA graph in ``holes._run_hole_dp``, and the two bounds on when recording
# one pays at all.
#
# The span sweep is a chain of ``max_B - 2`` launches differing only in which span they compute,
# and through the middle of its range the *host* is the critical path: at a 512-vertex rim its 510
# launches cost more wall than the kernels they issue, which overlap underneath them. Moving the
# span onto the device (``HoleFillTables.span_base``) makes the launches identical, so one group of
# ``HOLE_DP_GRAPH_SPANS`` can be recorded once and replayed over the whole sweep -- and a replayed
# launch costs a fraction of an issued one.
#
# **The group size is a shallow optimum and the two bounds are not.** Swept over the real sweep
# at four rim lengths, 8 won at every one and 16 came within a few percent; 64 and up lose,
# because recording costs one ordinary launch per span in the group. Both bounds below were
# measured the same way, interleaved in one clock state, with the emitted face buffer
# byte-identical throughout.
#
# The lower bound is the sweep length, because recording is paid once against a sequence it has to
# be short beside -- this is the "record and replay once" case, which loses outright:
#
#     spans   6     10    14    22    30    46    62
#     ratio   0.57x 0.73x 0.84x 0.99x 1.20x 1.32x 1.44x
#
# The upper bound is the *device work a span launch carries*, which is what decides whether the
# host was ever the critical path: a span level evaluates ``n_loops * (B - span) * (span - 1)``
# apex candidates, so ``n_loops * B^2`` is that count up to a constant. Past the crossover the
# kernels already cover their own launches and replaying them only adds the per-group advance. The
# quantity separates rim *length* from rim *count* correctly, which is why it is not a bound on B:
#
#     n_loops*B^2   0.52M 0.59M 0.82M 0.88M 0.99M 1.05M 1.18M 1.57M 1.61M
#     ratio         1.44x 1.34x 1.25x 1.07x 1.07x 1.03x 0.99x 0.91x 0.96x
#
# Two readings that are *not* the cause of the far end and were each measured away rather than
# assumed: the replayed group's over-wide grid is free (0.91-1.02x for the same sweep issued at
# maximal width, device-bound and host-bound alike), and the group size does not rescue it (every
# size from 8 to 128 loses at 2 rims of 1 024).
HOLE_DP_GRAPH_SPANS = 8
HOLE_DP_CAPTURE_FROM = 32
HOLE_DP_CAPTURE_MAX_WORK = 1_000_000


def hole_dp_captures(max_size: int, n_loops: int) -> bool:
    """
    Whether the span sweep over ``n_loops`` rims whose longest is ``max_size`` should be recorded.

    Long enough that recording it is short beside the sequence it replaces, and light enough per
    span that the host was the critical path to begin with. See the tables above; the caller adds
    the device test, since only CUDA has a graph to record.
    """
    return (
        max_size - 2 >= HOLE_DP_CAPTURE_FROM
        and n_loops * max_size * max_size <= HOLE_DP_CAPTURE_MAX_WORK
    )


def hole_dp_block(max_size: int, n_loops: int) -> int:
    """
    Lanes per block for a DP sweep over ``n_loops`` rims whose longest is ``max_size``.

    Wide only when the rim is long enough to give the lanes work *and* there are few enough rims
    that the ``(n_loops, max_size - span)`` grid would otherwise starve the device. See the table
    above for why both conditions are needed.
    """
    if max_size >= HOLE_DP_LONG_RIM and n_loops <= HOLE_DP_WIDE_GRID_LOOPS:
        return HOLE_DP_BLOCK_LONG
    return HOLE_DP_BLOCK


@wp.struct
class HoleFillTables:
    """
    The hole-filling DP's invariant inputs, bundled so the per-span launches carry one argument.

    ``holes._run_hole_dp`` runs ``fill_dp_span`` once per span -- ``max_B - 2`` times for the whole
    mesh -- and every one of these values is the same on every launch. A ``wp.launch`` argument
    costs about a microsecond of host time, linearly and on both devices, so a 16-argument kernel
    launched hundreds of times spent milliseconds marshalling constants. Only the span offset
    stays an argument.

    **``dp`` and ``prev`` are in here too, and that is a measured decision rather than a tidy
    one.** They are the launch's in-place *output*, so leaving them as arguments reads better and
    that is how this struct originally drew the line. But their pointers do not change across the
    sweep either, and wherever the host is the sweep's critical path that legibility costs a real
    fraction of the call. Moving the two in is a clear win on a long rim and flat on a short one,
    whose sweep is a few dozen launches rather than hundreds. It still pays where the sweep is
    *recorded* rather than issued, because recording marshals each launch exactly once as an
    ordinary one, and on the CPU device, which never records at all.
    ``holes._run_hole_dp`` binds them once, right where it binds everything else.

    **``span_base`` is here for a different reason, and it is what lets the sweep be captured.**
    The per-span launches differ in exactly one thing -- which span they are -- so moving that one
    value onto the device makes consecutive launches *identical*, and an identical sequence is the
    one thing CUDA graph capture pays for. The launch argument is then the offset within the
    recorded group and ``span_base[0]`` is the group's first span, stepped on the device by
    [`advance_span_base`][ordito.kernels.holes.advance_span_base]. See ``holes._run_hole_dp``.

    Build it ONCE in the wrapper and reuse it: construction is not free, and doing it per launch
    would give most of the saving back.

    A cross-*session* before/after read this as a win on the long rims and a loss on the short ones;
    the loss was drift in the surrounding work, which is what interleaving in one clock state is
    for.
    """

    loop_pos: wp.array[wp.vec3]
    loop_offsets: wp.array[wp.int32]
    dp_offsets: wp.array[wp.int32]
    active: wp.array[wp.int32]
    plane_normals: wp.array[wp.vec3]
    forbidden: wp.array[wp.int32]
    rim_opp_pos: wp.array[wp.vec3]
    rim_opp_valid: wp.array[wp.int32]
    char_areas: wp.array[wp.float32]
    dp: wp.array[wp.float32]
    prev: wp.array[wp.int32]
    span_base: wp.array[wp.int32]
    metric_id: wp.int32
    combine_id: wp.int32
    smooth_bd: wp.int32


@wp.func
def store_dp(
    tables: HoleFillTables, base: wp.int32, b: wp.int32, i: wp.int32, j: wp.int32, value: wp.float32
):
    # Write interval (i, j)'s cost, ``i < j``, at ``(i, j)`` and mirrored at ``(j, i)``. A loop's
    # ``B x B`` block uses only its upper triangle, so the mirror costs no memory, and it is what
    # lets ``apex_children`` read child ``(k, j)`` -- the *column* of the apex loop -- contiguously
    # in ``k``. ``init_dp_base`` seeds the mirrored rim edges.
    tables.dp[base + i * b + j] = value
    tables.dp[base + j * b + i] = value


@wp.func
def apex_children(
    tables: HoleFillTables, b: wp.int32, base: wp.int32, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.float32:
    # The two sub-intervals' combined cost for apex ``k`` of the interval (i, j): the first term
    # of ``apex_cost``, and a lower bound on all of it. Every term ``apex_cost`` combines in after
    # it is non-negative (areas, lengths, diameters, aspect ratios, ``exp``, ``abs`` and a fourth
    # power; ``BAD_METRIC`` and infinity are positive) and both combines are monotone -- a float
    # sum with a non-negative addend never rounds below its first operand, and a max never falls
    # -- so the bound holds exactly in ``float32``, not just in exact arithmetic.
    return combine_metric(
        tables.dp[base + i * b + k], tables.dp[base + j * b + k], tables.combine_id
    )


@wp.func
def apex_cost(
    tables: HoleFillTables,
    o: wp.int32,
    b: wp.int32,
    base: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    is_top: wp.bool,
    a_pos: wp.vec3,
    c_pos: wp.vec3,
    plane_normal: wp.vec3,
    char_area: wp.float32,
    children: wp.float32,
) -> wp.float32:
    # Metric of triangulating the interval (i, j) with apex ``k``: the two sub-intervals' costs
    # (``children``, from ``apex_children``), this triangle's term, and the per-edge (dihedral)
    # terms for the interior chords (i, k) / (k, j) taken at the neighbouring sub-interval's apex --
    # or, for a rim edge, at the existing face's opposite vertex when ``tables.smooth_bd`` is set.
    k_pos = tables.loop_pos[o + k]
    tri = triangle_fill_metric(a_pos, k_pos, c_pos, plane_normal, char_area, tables.metric_id)
    # The children were read by ``apex_children``, child ``(k, j)`` from its mirror in the free
    # lower triangle (``store_dp``) so that, with the lanes of ``fill_dp_span_tiled`` striding
    # ``k``, neither read is strided by the table's row length. A *separate* transposed table was
    # once a net loss -- its extra stores were not hidden while the triangle metric dominated the
    # apex -- but the branch and bound in ``apex_children`` removes most metric evaluations, after
    # which the strided column read was the bulk of a long rim's sweep: on a 2 765-vertex rim the
    # in-table mirror took the fill from 1.22x to 1.57x the exhaustive loop, at no extra memory.
    left = base + i * b + k
    right = base + k * b + j
    val = combine_metric(children, tri, tables.combine_id)

    # Each ``prev`` entry is bound once: a global load repeated across a branch is not something
    # the compiler is obliged to common up.
    if k > i + 1:
        left_prev = tables.prev[left]
        if left_prev >= 0:
            e = fill_edge_term(
                a_pos, k_pos, tables.loop_pos[o + left_prev], c_pos, tables.metric_id
            )
            val = combine_metric(val, e, tables.combine_id)
    elif tables.smooth_bd != 0 and tables.rim_opp_valid[o + i] != 0:
        e = fill_edge_term(a_pos, k_pos, tables.rim_opp_pos[o + i], c_pos, tables.metric_id)
        val = combine_metric(val, e, tables.combine_id)

    if j > k + 1:
        right_prev = tables.prev[right]
        if right_prev >= 0:
            e = fill_edge_term(
                k_pos, c_pos, tables.loop_pos[o + right_prev], a_pos, tables.metric_id
            )
            val = combine_metric(val, e, tables.combine_id)
    elif tables.smooth_bd != 0 and tables.rim_opp_valid[o + k] != 0:
        e = fill_edge_term(k_pos, c_pos, tables.rim_opp_pos[o + k], a_pos, tables.metric_id)
        val = combine_metric(val, e, tables.combine_id)

    # Closing rim edge (loop[b-1] -> loop[0]) is the base of the whole loop and has no parent, so
    # its boundary term is added here for the top interval only.
    if is_top and tables.smooth_bd != 0 and tables.rim_opp_valid[o + b - 1] != 0:
        e = fill_edge_term(a_pos, c_pos, tables.rim_opp_pos[o + b - 1], k_pos, tables.metric_id)
        val = combine_metric(val, e, tables.combine_id)
    return val


@wp.func
def relax_apex(
    tables: HoleFillTables,
    o: wp.int32,
    b: wp.int32,
    base: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    is_top: wp.bool,
    a_pos: wp.vec3,
    c_pos: wp.vec3,
    plane_normal: wp.vec3,
    char_area: wp.float32,
    bound: wp.float32,
    best_val: wp.ref[wp.float32],
    best_k: wp.ref[wp.int32],
):
    # One apex of the interval (i, j) under the branch and bound both span kernels run: an apex
    # whose children alone reach the running best cannot win the strict-``<`` argmin
    # (``apex_children``'s bound is exact, and an equal cost keeps the earlier ``k``), nor can one
    # whose children exceed ``bound`` (the tiled kernel's block-shared best; infinity elsewhere), so
    # its triangle and edge terms are never evaluated. The answer, ties included, is the exhaustive
    # loop's. One rule for the serial loop and both rounds of the tiled one.
    children = apex_children(tables, b, base, i, j, k)
    if children < best_val and children <= bound:
        val = apex_cost(
            tables, o, b, base, i, j, k, is_top, a_pos, c_pos, plane_normal, char_area, children
        )
        update_argmin(best_val, best_k, val, k)


@wp.func
def interval_apex(value: wp.float32, apex: wp.int32) -> wp.int32:
    # The apex both span kernels store beside interval cost ``value``. An infinite cost means every
    # apex left available required at least one forbidden sub-chord (a genuinely infinite child
    # cost propagates through the sum/max in ``apex_cost``, never a finite BAD_METRIC), so there is
    # no legal triangulation of this span at all -- not merely a bad-looking one -- and the apex is
    # ``-1``, which the traceback skips.
    return wp.where(value >= FLOAT32_INF_CONSTANT, wp.int32(-1), apex)


@wp.kernel(enable_backward=False)
def fill_dp_span(tables: HoleFillTables, span_offset: wp.int32) -> None:
    # One thread per span-``span`` interval (i, j = i + span) of every loop at once; reads only
    # strictly smaller spans, so successive launches (span = 2, 3, ...) are the DP barriers.
    #
    # Launched over ``(n_loops, max_B - span)``: threads whose loop is shorter than the current
    # span exit at once, so a mesh whose loops differ wildly in length wastes some of the grid, but
    # the launch *count* is ``max_B - 1`` for the whole mesh instead of ``B - 1`` per loop.
    #
    # This is the **CPU** engine and the tie-break reference; CUDA runs
    # ``fill_dp_span_tiled``, which must agree with it apex for apex (see
    # ``tests/test_holes.py::test_fill_dp_span_tiled_matches_serial``).
    #
    # The prologue it shares with the tiled kernel is deliberately not extracted. Three of its
    # lines are early ``return``s and a ``@wp.func`` cannot return from its caller, so hoisting it
    # would mean a validity flag plus restructured guards in both kernels -- more code at the call
    # sites than it removes. What is left after the guards is five independent loads that read
    # fine where they are; grouping them would name a bag, not a quantity. The genuinely shared
    # arithmetic already moved: ``apex_cost`` went from 19 parameters to 14 when the tables became
    # ``HoleFillTables``.
    ell, i = wp.tid()
    # The span is the group's base plus this launch's offset within it, so that every launch of a
    # recorded group is byte-identical bar one integer -- see ``HoleFillTables.span_base``.
    span = tables.span_base[0] + span_offset
    if tables.active[ell] == 0:
        return
    b = tables.loop_offsets[ell + 1] - tables.loop_offsets[ell]
    if span >= b or i >= b - span:
        return
    j = i + span
    base = tables.dp_offsets[ell]
    if tables.forbidden[base + i * b + j] != 0:
        # Interior chord would duplicate an existing mesh edge (non-manifold) — leave it unfilled.
        # This is a *hard* rejection and must use the true-infinite sentinel, not ``BAD_METRIC``:
        # ``BAD_METRIC`` is also what a legal-but-ugly triangle scores (see
        # ``triangle_fill_metric``), and that value is deliberately still selectable so the
        # min-area fallback has something to work with. Using it here too would let a parent
        # interval, forced to choose between two forbidden children, numerically prefer whichever
        # BAD_METRIC total happened to be smaller and emit its own triangle anyway — reusing the
        # very chord that made the child infeasible. Only genuine infinity propagates through
        # ``combine_metric``'s sum/max without being mistaken for "bad but legal".
        store_dp(tables, base, b, i, j, FLOAT32_INF_CONSTANT)
        tables.prev[base + i * b + j] = -1
        return
    o = tables.loop_offsets[ell]
    plane_normal = tables.plane_normals[ell]
    char_area = tables.char_areas[ell]
    a_pos = tables.loop_pos[o + i]
    c_pos = tables.loop_pos[o + j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = wp.int32(-1)
    for k in range(i + 1, j):
        relax_apex(
            tables,
            o,
            b,
            base,
            i,
            j,
            k,
            is_top,
            a_pos,
            c_pos,
            plane_normal,
            char_area,
            FLOAT32_INF_CONSTANT,
            best_val,
            best_k,
        )
    store_dp(tables, base, b, i, j, best_val)
    tables.prev[base + i * b + j] = interval_apex(best_val, best_k)


@wp.kernel(enable_backward=False)
def fill_dp_span_tiled(tables: HoleFillTables, span_offset: wp.int32) -> None:
    # One *block* per span-``span`` interval, its lanes striding the apex loop. Same DP, same launch
    # count, ``block_dim`` times the parallelism: the serial kernel above puts at most
    # ``n_loops * (max_B - span)`` threads on the machine, which for a single long boundary is a few
    # hundred out of a few hundred thousand.
    #
    # **The stride is ``wp.block_dim()``, not ``HOLE_DP_BLOCK``, and that is what makes this kernel
    # portable.** On CUDA they agree, because the launch passes whichever of the two constants
    # ``hole_dp_block`` picked -- which is a second reason the stride must be the runtime value and
    # not the constant, since the constant is now only one of the two it could have been launched
    # with. They differ on the CPU device, where ``wp.launch_tiled`` runs exactly one lane
    # per block through Warp 1.18 (unless the experimental ``wp.config.enable_cpu_blocks`` is set)
    # and ``wp.block_dim()`` reads 1. With the constant, lane 0 was the
    # only lane running and it stepped by 32, so the DP minimized over every 32nd apex and returned
    # a valid-looking, equal-count, *wrong* triangulation. With the runtime value the single CPU
    # lane strides by 1, covers every apex, and the two tile reductions below degenerate to
    # one-element tiles that return that lane's own answer. Byte-identical to ``fill_dp_span`` on
    # both devices, and the runtime stride costs nothing on CUDA -- the loop body is an
    # ``apex_cost`` call, so there was never much for a compile-time step to unroll.
    #
    # **The tie-break is the contract, not the cost.** ``update_argmin`` takes the *smallest* apex
    # ``k`` at equal cost, and that choice decides the emitted triangles, so a differently-tied
    # reduction is a valid, equal-cost, *different* filling -- which every metric/count test in the
    # suite passes. The two-stage reduction below reproduces it exactly and without any float
    # bit-packing: the block minimum of the cost, then the block minimum of ``k`` over just the
    # lanes that attained it. A lane's own ``update_argmin`` already holds the smallest ``k`` at its
    # own minimum, so the pair is (min cost, min k attaining it) -- which is what an ascending
    # strict ``<`` scan returns. Lanes with no apex, and an all-non-finite interval, both leave
    # ``(inf, -1)`` and agree with the serial kernel there too.
    ell, i, t = wp.tid()
    # As in ``fill_dp_span``: the group's base span plus this launch's offset within the group.
    span = tables.span_base[0] + span_offset
    # Every guard below is warp-uniform (it reads only ``ell``, ``i`` and ``span``), so the whole
    # block returns together and the tile reductions never run in divergent control flow.
    if tables.active[ell] == 0:
        return
    b = tables.loop_offsets[ell + 1] - tables.loop_offsets[ell]
    if span >= b or i >= b - span:
        return
    j = i + span
    base = tables.dp_offsets[ell]
    if tables.forbidden[base + i * b + j] != 0:
        # True infinity, not ``BAD_METRIC`` — see the identical branch in ``fill_dp_span``.
        if t == 0:
            store_dp(tables, base, b, i, j, FLOAT32_INF_CONSTANT)
            tables.prev[base + i * b + j] = -1
        return
    o = tables.loop_offsets[ell]
    plane_normal = tables.plane_normals[ell]
    char_area = tables.char_areas[ell]
    a_pos = tables.loop_pos[o + i]
    c_pos = tables.loop_pos[o + j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = wp.int32(-1)
    # The serial kernel's branch and bound, in two rounds. Each lane's first apex is evaluated
    # outright; once the apexes outnumber the lanes, one block minimum then hands every lane the
    # best cost any lane has seen, and every later apex is skipped when its children alone reach
    # the lane's own best *or exceed* the block's. The second test is strict because an apex tied
    # with the block's best could still win on its smaller ``k``; an apex strictly above it can
    # never be the block's minimum, and a lane whose recorded best it would have been then cannot
    # hold the minimum either -- so the two-stage reduction below reads the exhaustive loop's pair.
    stride = wp.block_dim()
    k = i + 1 + t
    if k < j:
        relax_apex(
            tables,
            o,
            b,
            base,
            i,
            j,
            k,
            is_top,
            a_pos,
            c_pos,
            plane_normal,
            char_area,
            FLOAT32_INF_CONSTANT,
            best_val,
            best_k,
        )
        k += stride
    shared = FLOAT32_INF_CONSTANT
    if j - i - 1 > stride:
        # Block-uniform: ``i``, ``j`` and the stride are the block's, so every lane reaches it.
        shared = block_min(best_val)
    while k < j:
        relax_apex(
            tables,
            o,
            b,
            base,
            i,
            j,
            k,
            is_top,
            a_pos,
            c_pos,
            plane_normal,
            char_area,
            shared,
            best_val,
            best_k,
        )
        k += stride
    block_val, block_k = block_argmin(best_val, best_k)
    if t == 0:
        store_dp(tables, base, b, i, j, block_val)
        tables.prev[base + i * b + j] = interval_apex(block_val, block_k)


@wp.kernel(enable_backward=False)
def advance_span_base(step: wp.int32, out_span_base: wp.array[wp.int32]) -> None:
    # dim=1. Close a recorded group of span launches by moving the base on to the next group's
    # first span. It is the last node of the graph ``holes._run_hole_dp`` records, so every span
    # kernel of the group has already read the old base by the time it runs, and consecutive
    # replays serialize on the stream -- which is the whole reason the step cannot instead be
    # folded into the last span kernel, whose blocks read the base at their own start and would
    # race it.
    #
    # Deliberately not ``array.loop_advance``: that closes a ``wp.capture_while`` round and writes
    # a condition and a progress flag this sweep has neither of -- its own docstring asks a hot
    # round loop to price a bespoke advance first, and this is one.
    out_span_base[0] = out_span_base[0] + step


@wp.kernel(enable_backward=False)
def traceback_fill_triangles(
    flat_loops: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    triangle_offsets: wp.array[wp.int32],
    prev: wp.array[wp.int32],
    stack: wp.array[wp.vec2i],
    out_counts: wp.array[wp.int32],
    out_triangles: wp.array2d[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # One thread per loop, walking its own predecessor table: interval ``(i, j)`` splits at apex
    # ``k = prev[i, j]`` into triangle ``(i, j, k)`` -- reversed rim winding, matching ``fan_faces``
    # -- and sub-intervals ``(i, k)``, ``(k, j)``. An apex of ``-1`` marks an interval no legal
    # triangulation covers (a forbidden chord, or a pinched rim), and is skipped, so a loop emits
    # *at most* ``B - 2`` triangles and the caller compacts on ``out_counts``. The triangles a loop
    # falls short by are added to ``out_state[FILL_STATE_SHORTFALL]`` (arriving zeroed), so the
    # caller's "is the padded buffer already the answer" test is one 4-byte read, and the scan
    # that places the compacted blocks runs only when some loop did fall short.
    #
    # The walk is depth-first with ``(k, j)`` taken before ``(i, k)``, which is what fixes the
    # emitted order -- the same order for the same table on either device.
    #
    # ``stack`` is caller-owned scratch, one slot per packed loop vertex, so loop ``ell`` owns
    # ``stack[loop_offsets[ell] : + B]``. That is exactly enough: the walk starts one deep and each
    # emitted triangle nets one entry, so the depth never passes ``B - 1``.
    ell = wp.int32(wp.tid())
    o = loop_offsets[ell]
    b = loop_offsets[ell + 1] - o
    if b < 3:
        out_counts[ell] = 0
        return
    base = dp_offsets[ell]
    tri = triangle_offsets[ell]
    stack[o] = wp.vec2i(0, b - 1)
    top = wp.int32(1)
    n = wp.int32(0)
    while top > 0:
        top -= 1
        interval = stack[o + top]
        i = interval[0]
        j = interval[1]
        if j - i >= 2:
            k = prev[base + i * b + j]
            if k >= 0:
                out_triangles[tri + n, 0] = flat_loops[o + i]
                out_triangles[tri + n, 1] = flat_loops[o + j]
                out_triangles[tri + n, 2] = flat_loops[o + k]
                n += 1
                stack[o + top] = wp.vec2i(i, k)
                top += 1
                stack[o + top] = wp.vec2i(k, j)
                top += 1
    out_counts[ell] = n
    if n < b - 2:
        wp.atomic_add(out_state, FILL_STATE_SHORTFALL, b - 2 - n)


@wp.kernel
def compact_fill_triangles(
    counts: wp.array[wp.int32],
    padded_offsets: wp.array[wp.int32],
    packed_ends: wp.array[wp.int32],
    padded: wp.array2d[wp.int32],
    out_triangles: wp.array2d[wp.int32],
) -> None:
    # Close the gaps a skipped apex left in the padded per-loop triangle blocks. ``packed_ends`` is
    # the *inclusive* scan of ``counts``, so a loop's packed block ends there and begins
    # ``counts[ell]`` earlier -- the same buffer the caller read its total from, rather than a
    # second exclusive scan of it.
    ell, t = wp.tid()
    count = counts[ell]
    if t >= count:
        return
    dst = packed_ends[ell] - count + t
    src = padded_offsets[ell] + t
    out_triangles[dst, 0] = padded[src, 0]
    out_triangles[dst, 1] = padded[src, 1]
    out_triangles[dst, 2] = padded[src, 2]


@wp.kernel
def flag_bad_triangulations(
    loop_offsets: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    dp: wp.array[wp.float32],
    out_retry: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Per-loop min-area retry mask: the whole-loop interval is dp[0, B - 1]. Testing it on device
    # replaces copying every loop's ``B x B`` table to the host to read one scalar out of it.
    # ``out_state[FILL_STATE_RETRY]`` (arriving zeroed) is set by any loop that fails, so the
    # caller's "does any loop need the fallback" test is one 4-byte read rather than a reduction
    # over the mask. Every writer stores the same value, so the race is benign.
    ell = wp.int32(wp.tid())
    top = dp[dp_offsets[ell] + loop_offsets[ell + 1] - loop_offsets[ell] - 1]
    bad = top >= BAD_METRIC
    out_retry[ell] = wp.where(bad, wp.int32(1), wp.int32(0))
    if bad:
        out_state[FILL_STATE_RETRY] = 1


@wp.kernel
def rim_edge_keys(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
) -> None:
    # The undirected key of rim edge ``(loop[t], loop[t + 1])`` of every loop at once, keyed on
    # slot ``t``: the table ``probe_rim_edges`` searches. The fill computes the same keys inside
    # ``loop_rim_metrics``; this is the stand-alone form for a caller with no rim metrics to take.
    t = wp.int32(wp.tid())
    u, v = loop_rim_edge_vertices(flat_loops, loop_id, loop_offsets, t)
    out_keys[t] = kernel_array.pack_edge_key(u, v, base)


# ``probe_rim_edges``' two non-vertex answers: a rim edge no face contains, and one several do.
RIM_NO_FACE = wp.constant(-1)
RIM_MANY_FACES = wp.constant(-2)


@wp.kernel
def probe_rim_edges(
    faces: wp.array[wp.int32],
    sorted_rim_keys: wp.array[wp.uint64],
    rim_slots: wp.array[wp.int32],
    base: wp.uint64,
    out_third: wp.array[wp.int32],
) -> None:
    # One thread per face: find, for every rim edge, the vertex opposite it in the faces containing
    # it. The rim is the small side of the question, so the rim's keys are the sorted table and
    # each face probes it with its own three edges -- where probing a sorted table of every mesh
    # edge with the rim's cost a radix sort over the whole mesh.
    #
    # ``out_third`` arrives as ``RIM_NO_FACE`` and ends as the third vertex of the *single*
    # adjacent face, or ``RIM_NO_FACE`` / ``RIM_MANY_FACES`` when there is none or more than one --
    # a face count folded into the answer, so one buffer carries what a count and a vertex did.
    # The first face claims the slot with a compare-and-swap, and any later one finds it taken and
    # marks it; every such writer stores the same value, so the race is benign.
    #
    # A key can occur at more than one rim slot (a loop that walks one edge twice), so every
    # matching slot is visited rather than the first.
    f = wp.int32(wp.tid())
    n = sorted_rim_keys.shape[0]
    for k in range(3):
        # Halfedge ``h = 3f + k`` runs ``v_k -> v_{k+1}``; the third corner is the origin of the
        # previous halfedge, which is ``halfedge_prev`` rather than a second spelling of the cycle.
        h = 3 * f + k
        a = faces[h]
        b = faces[3 * f + (k + 1) % 3]
        third = faces[halfedge_prev(h)]
        key = kernel_array.pack_edge_key(a, b, base)
        index = kernel_array.binary_search_index_left(sorted_rim_keys, key)
        while index < n and sorted_rim_keys[index] == key:
            slot = rim_slots[index]
            if wp.atomic_cas(out_third, slot, RIM_NO_FACE, third) != RIM_NO_FACE:
                out_third[slot] = RIM_MANY_FACES
            index += 1


@wp.kernel
def rim_opposite_positions(
    vertices: wp.array[wp.vec3],
    third: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_valid: wp.array[wp.int32],
) -> None:
    # Per rim edge: a single adjacent face means its third vertex blends the fill dihedral metrics
    # into the surface; none, or several, and the edge has no one opposite vertex.
    t = wp.int32(wp.tid())
    out_positions[t] = wp.vec3(0.0, 0.0, 0.0)
    out_valid[t] = wp.int32(0)
    if third[t] >= 0:
        out_positions[t] = vertices[third[t]]
        out_valid[t] = wp.int32(1)


@wp.kernel
def scatter_loop_positions(
    flat_loops: wp.array[wp.int32], out_flat_slot: wp.array[wp.int32]
) -> None:
    # Where each mesh vertex sits in the packed loop buffer, for the chord test below. The value
    # stored is the *flat* index, which is globally unique and increases with position inside a
    # loop — so the highest position still wins on a duplicated loop vertex (matching the
    # last-write-wins dict the host implementation built), and a pinch vertex shared by two loops
    # resolves to the later loop rather than needing the scratch cleared between loops.
    t = wp.int32(wp.tid())
    wp.atomic_max(out_flat_slot, flat_loops[t], t)


@wp.kernel
def mark_forbidden_chords(
    faces: wp.array[wp.int32],
    flat_slot: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # Chords (non-adjacent loop positions) that already exist as mesh edges are forbidden
    # outright, rather than re-routed. Idempotent writes: no atomics needed. One pass
    # over the whole mesh marks the masks of *all* loops, because ``flat_slot`` distinguishes them:
    # an edge whose endpoints land in two different loops is not a chord of either. One thread per
    # halfedge, its endpoints read off ``faces``: an interior edge is visited from both sides, and
    # the second visit writes what the first did.
    u, v = halfedge_endpoints(faces, wp.int32(wp.tid()))
    tu = flat_slot[u]
    tv = flat_slot[v]
    if tu < 0 or tv < 0:
        return
    ell = loop_id[tu]
    if ell != loop_id[tv]:
        return
    o = loop_offsets[ell]
    b = loop_offsets[ell + 1] - o
    lo = wp.min(tu, tv) - o
    hi = wp.max(tu, tv) - o
    if hi - lo >= 2 and hi - lo <= b - 2:
        base = dp_offsets[ell]
        out_mask[base + lo * b + hi] = wp.int32(1)
        out_mask[base + hi * b + lo] = wp.int32(1)


# --- stitching two boundary loops -------------------------------------------------------------


@wp.func
def rim_vertex(
    loop: wp.array[wp.int32], n: wp.int32, i: wp.int32, shift: wp.int32, flip: wp.bool
) -> wp.int32:
    # Entry ``i`` of a rim loop rolled by ``shift`` -- of the *reversed* loop when ``flip`` -- read
    # straight from the loop: ``loop[n - 1 - wrap(i + shift)]`` or ``loop[wrap(i + shift)]``. One
    # rule for the reversal, the rolls and their composition, so ``stitch_loops`` never
    # materializes the reversed or rolled loops it reasons about.
    k = _wrap(i + shift, n)
    if flip:
        k = n - 1 - k
    return loop[k]


@wp.kernel
def stitch_rim_positions(
    vertices_a: wp.array[wp.vec3],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    loop_b: wp.array[wp.int32],
    out_a_pos: wp.array[wp.vec3],
    out_b_pos: wp.array[wp.vec3],
) -> None:
    # Both rims' positions in one launch over ``n + m`` threads: rim A reversed, so both rims wind
    # the same way, and rim B as given.
    t = wp.int32(wp.tid())
    n = loop_a.shape[0]
    if t < n:
        out_a_pos[t] = vertices_a[rim_vertex(loop_a, n, t, 0, True)]
    else:
        out_b_pos[t - n] = vertices_b[loop_b[t - n]]


# The two costs ``row_argmin`` minimises, selected per launch (warp-uniform).
LOOP_PAIR_PERIMETER = wp.constant(0)
LOOP_PAIR_SQ_DISTANCE = wp.constant(1)


@wp.func
def loop_pair_cost(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    n_a: wp.int32,
    i: wp.int32,
    j: wp.int32,
    metric: wp.int32,
) -> wp.float32:
    # The cost of pairing rim-A position ``i`` with rim-B position ``j``: with
    # ``LOOP_PAIR_PERIMETER`` the triangle perimeter less its A edge,
    # ``|a_i - b_j| + |a_{i+1} - b_j|`` (the zippering's objective), otherwise the squared distance
    # ``|a_i - b_j|^2`` (the stitch band's start pair). Evaluated where it is minimised rather than
    # tabulated first: every reader takes a row's minimum, so a whole ``(n_a, n_b)`` table was
    # written to be read once.
    b = b_pos[j]
    if metric == LOOP_PAIR_SQ_DISTANCE:
        return wp.length_sq(a_pos[i] - b)
    return wp.length(a_pos[i] - b) + wp.length(a_pos[loop_point(i + 1, n_a)] - b)


# Lanes per row of ``row_argmin``.
ROW_ARGMIN_BLOCK = 64


@wp.kernel(enable_backward=False)
def row_argmin(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    n_a: wp.int32,
    m_b: wp.int32,
    metric: wp.int32,
    out_col: wp.array[wp.int32],
    out_val: wp.array[wp.float32],
) -> None:
    # Each A row's lowest-cost B column (``loop_pair_cost``), lowest column winning a tie -- the
    # answer of one thread walking the row with ``update_argmin``, bit for bit: each lane walks the
    # columns congruent to it in ascending order, and ``reduce.block_argmin`` takes the smallest
    # value and the lowest column among the lanes that attained it.
    #
    # One block per row (``wp.launch_tiled(dim=n_a)``) because the costs are no longer tabulated by
    # a 2-D launch first: one thread per row walking every column measured 0.79x at 4 096-vertex
    # rims, where the tabulation's parallelism was worth more than the table's traffic. Lanes stride
    # by ``wp.block_dim()``, so the one CPU lane walks the whole row (CLAUDE.md section 2.2).
    i, lane = wp.tid()
    best_col = INT32_MAX_CONSTANT
    best_val = FLOAT32_INF_CONSTANT
    for j in range(lane, m_b, wp.block_dim()):
        update_argmin(best_val, best_col, loop_pair_cost(a_pos, b_pos, n_a, i, j, metric), j)
    best_val, best_col = block_argmin(best_val, best_col)
    if lane == 0:
        out_col[i] = best_col
        out_val[i] = best_val


@wp.kernel(enable_backward=False)
def global_argmin(
    col_min: wp.array[wp.int32],
    val_min: wp.array[wp.float32],
    n_a: wp.int32,
    out_shift: wp.array[wp.int32],
) -> None:
    # Single-thread reduction: out_shift = (shift_a, shift_b).
    #
    # One lane walking ``n_a`` elements looks like the antipattern it usually is, and folding it
    # into ``row_argmin`` -- which would reduce each row's winner into a packed
    # ``pack_nearest_key`` atomic and drop this launch entirely -- was measured and **declined**. It
    # is a fraction of a percent of the whole ``stitch_loops`` call at every rim size, it is
    # launch-dominated rather than loop-dominated, and its share *falls* with rim size, so the
    # saving would be largest exactly where the call is already cheap.
    best_row = wp.int32(0)
    best_val = val_min[0]
    for i in range(1, n_a):
        update_argmin(best_val, best_row, val_min[i], i)
    out_shift[0] = best_row
    out_shift[1] = col_min[best_row]


@wp.kernel
def rolled_edge_map(
    col_min: wp.array[wp.int32],
    shift: wp.array[wp.int32],
    n_a: wp.int32,
    m_b: wp.int32,
    out_edge: wp.array[wp.int32],
) -> None:
    # Per-edge B vertex after rolling both loops so the global-min pair is first
    # (``argmin(roll(roll(perimeters, -shift_a, 0), -shift_b, 1), axis=1)``). ``shift`` is
    # ``global_argmin``'s ``(shift_a, shift_b)``, read here rather than passed through the host.
    i = wp.int32(wp.tid())
    out_edge[i] = _wrap(col_min[_wrap(i + shift[0], n_a)] - shift[1], m_b)


@wp.kernel(enable_backward=False)
def resolve_corrections(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    unsorted_indices: wp.array[wp.int32],
    next_indices: wp.array[wp.int32],
    n_corrections: wp.int32,
    row_roll: wp.int32,
    col_roll: wp.int32,
    n_a: wp.int32,
    m_b: wp.int32,
    out_edge: wp.array[wp.int32],
) -> None:
    # Single-thread sequential correction: force ``out_edge`` non-decreasing by re-picking, for
    # each unsorted edge, the B vertex minimizing the perimeter within the bracket of its stable
    # neighbours. ``out_edge`` has length ``n_a + 1`` with the sentinel ``out_edge[n_a] == m_b``.
    # The perimeter is ``loop_pair_cost`` over the unrolled loops, indexed through the running
    # ``row_roll``/``col_roll``.
    for k in range(n_corrections):
        idx = unsorted_indices[k]
        lo = out_edge[idx - 1]
        hi = out_edge[next_indices[k]]
        if hi > m_b - 1:
            hi = m_b - 1
        row = _wrap(idx + row_roll, n_a)
        best_col = lo
        best_val = loop_pair_cost(
            a_pos, b_pos, n_a, row, _wrap(lo + col_roll, m_b), LOOP_PAIR_PERIMETER
        )
        for c in range(lo + 1, hi + 1):
            cost = loop_pair_cost(
                a_pos, b_pos, n_a, row, _wrap(c + col_roll, m_b), LOOP_PAIR_PERIMETER
            )
            update_argmin(best_val, best_col, cost, c)
        out_edge[idx] = best_col


@wp.kernel
def bridge_faces(
    loop_a: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
    edge: wp.array[wp.int32],
    row_roll: wp.int32,
    col_roll: wp.int32,
    b_offset: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # The band's triangles in one launch over ``n + m`` threads, into one buffer: A-edge ``i``'s
    # triangle at row ``i``, B-edge ``j``'s at row ``n + j``. Both rims are read through
    # ``rim_vertex`` -- A reversed and rolled by ``row_roll``, B rolled by ``col_roll`` and offset
    # by ``b_offset`` into the concatenated vertex buffer -- so the rolled loops are never written.
    t = wp.int32(wp.tid())
    n = loop_a.shape[0]
    m = loop_b.shape[0]
    if t < n:
        i = t
        out_faces[3 * t + 0] = rim_vertex(loop_a, n, i, row_roll, True)
        out_faces[3 * t + 1] = rim_vertex(loop_a, n, i + 1, row_roll, True)
        out_faces[3 * t + 2] = rim_vertex(loop_b, m, edge[i], col_roll, False) + b_offset
        return
    j = t - n
    # ``edge`` is non-decreasing, so the upper bound over its first ``n`` entries is the count of
    # entries at or below ``j`` -- the A-loop vertex this B-loop vertex fans to. A result of ``n``
    # wraps to the first.
    apex = kernel_array.binary_search_index(edge[:n], j) % n
    # The B edge is reversed (``fliplr``) so the bridge winding matches mesh B's faces.
    out_faces[3 * t + 0] = rim_vertex(loop_b, m, j + 1, col_roll, False) + b_offset
    out_faces[3 * t + 1] = rim_vertex(loop_b, m, j, col_roll, False) + b_offset
    out_faces[3 * t + 2] = rim_vertex(loop_a, n, apex, row_roll, True)


# --- Minimum-weight hole triangulation (the Liepa/Klincsek interval DP) -----------------------


# --- Metric-based two-hole stitching (the same DP over a grid rather than an interval) ---------

# Stitch-metric selectors for ``stitch_triangle_metric`` / ``stitch_edge_metric`` / stitch DP.
METRIC_COMPLEX_STITCH = wp.constant(wp.int32(0))
METRIC_EDGE_LENGTH_STITCH = wp.constant(wp.int32(1))
METRIC_VERTICAL_STITCH = wp.constant(wp.int32(2))

# came-from directions stored per grid cell.
CAME_NONE = wp.constant(wp.int32(-1))
CAME_A = wp.constant(wp.int32(0))  # reached (i, j) by advancing loop A: from (i-1, j)
CAME_B = wp.constant(wp.int32(1))  # reached (i, j) by advancing loop B: from (i, j-1)


@wp.func
def stitch_triangle_metric(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, metric_id: wp.int32
) -> wp.float32:
    # Per-band-triangle term of each stitch metric. ``a`` and ``c`` MUST be the two ends of the
    # *new connection edge* this band triangle introduces (one vertex of each rim) and ``b`` the
    # remaining, rim-side corner -- ``edge_length_stitch`` is documented as "summed connection-edge
    # length" and reads exactly that pair. The other two metrics are invariant under any
    # permutation of the three corners (``triangle_aspect_ratio`` is symmetric, and the vertical
    # term uses only ``|cross|``, ``|dot(up, cross)|`` and the sum of squared side lengths), so
    # this ordering rule constrains nothing but the metric that reads it.
    if metric_id == METRIC_EDGE_LENGTH_STITCH:
        return wp.length(c - a)
    if metric_id == METRIC_VERTICAL_STITCH:
        ab = b - a
        ac = c - a
        bc = c - b
        norm = wp.cross(ab, ac)
        pp = wp.abs(wp.dot(up, norm))
        sides = wp.length_sq(ab) + wp.length_sq(ac) + wp.length_sq(bc)
        return wp.length_sq(norm) + 100.0 * pp * pp + 0.5 * sides * sides
    # METRIC_COMPLEX_STITCH: proportional to triangle aspect ratio (1e-2 as aspect is unbounded).
    return (triangle_aspect_ratio(a, b, c) - 1.0) * 0.01


@wp.func
def stitch_edge_metric(a: wp.vec3, b: wp.vec3, lft: wp.vec3, rgt: wp.vec3) -> wp.float32:
    # complex_stitch edge term: (1 - cos dihedral) * 1e4 between the two triangles sharing edge a-b
    # Normals are unit here, unlike the fill dihedral.
    ab = b - a
    norm_l = wp.normalize(wp.cross(lft - a, ab))
    norm_r = wp.normalize(wp.cross(ab, rgt - a))
    return (1.0 - wp.dot(norm_l, norm_r)) * 1.0e4


@wp.func
def stitch_prev_apex(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    came: wp.array2d[wp.int32],
    n_a: wp.int32,
    n_b: wp.int32,
    i: wp.int32,
    j: wp.int32,
) -> wp.vec3:
    # Apex of the band triangle already placed at cell (i, j) -- the vertex advanced to reach it.
    # Only called when came[i, j] is valid.
    if came[i, j] == CAME_A:
        return a_pos[(i - 1) % n_a]
    return b_pos[(j - 1) % n_b]


@wp.kernel
def set_dp_origin(out_dp: wp.array2d[wp.float32]) -> None:
    # Seed the stitch grid DP: the empty band consuming 0 edges of either loop costs nothing.
    out_dp[0, 0] = 0.0


@wp.struct
class StitchTables:
    """
    The stitch DP's invariant inputs, bundled so the sweep's launches carry one argument.

    ``holes._run_stitch_dp`` launches one of the two kernels below repeatedly -- once per
    tile-diagonal, or once per anti-diagonal for the reference schedule -- and every one of these
    ten values is the same on every launch. Only the diagonal index and the two in-place DP tables
    vary. This is ``HoleFillTables``' argument exactly, in the same file and on the same shape of
    loop; see that struct's docstring for the per-argument cost model and for why the struct must
    be built **once**, outside the loop.

    Not graph capture: recording a graph costs at least what issuing the launches costs, so capture
    pays only on a sequence that is *replayed*, and this loop runs once per call. Measured at the
    diagonal schedule's launch count, capture-and-replay-once is a loss where a bundle is a
    substantial win. The tiled schedule then removed most of those launches outright, which is the
    one thing capture could not do here.
    """

    a_pos: wp.array[wp.vec3]
    b_pos: wp.array[wp.vec3]
    a_opp: wp.array[wp.vec3]
    a_opp_valid: wp.array[wp.int32]
    b_opp: wp.array[wp.vec3]
    b_opp_valid: wp.array[wp.int32]
    up: wp.vec3
    metric_id: wp.int32
    n_a: wp.int32
    n_b: wp.int32


@wp.func
def stitch_dp_cell(
    tables: StitchTables,
    i: wp.int32,
    j: wp.int32,
    out_dp: wp.array2d[wp.float32],
    out_came: wp.array2d[wp.int32],
) -> tuple[wp.float32, wp.int32]:
    # One cell of the stitch grid DP: ``dp[i, j]`` is the min cost of the band consuming ``i``
    # edges of A and ``j`` edges of B from the aligned start at cell (0, 0), and ``came[i, j]``
    # says which rim the last step advanced. It reads *only* ``(i - 1, j)`` and ``(i, j - 1)``,
    # which is what lets the two kernels below schedule it differently and still agree cell for
    # cell -- one thread per anti-diagonal cell, or one block per tile of the grid.
    #
    # The caller owns cell (0, 0) and must not call here for it. The cost is returned rather than
    # stored so that neither schedule has to decide where the answer goes.
    a_pos = tables.a_pos
    b_pos = tables.b_pos
    up = tables.up
    metric_id = tables.metric_id
    n_a = tables.n_a
    n_b = tables.n_b
    # Never let a full ring come from one loop before touching the other.
    if (i == n_a and j == 0) or (j == n_b and i == 0):
        return BAD_METRIC, CAME_NONE

    complex_edge = metric_id == METRIC_COMPLEX_STITCH
    best = FLOAT32_INF_CONSTANT
    best_came = CAME_NONE

    # Advance loop A: new triangle (a[i-1], b[j], a[i]). Whichever rim advances, the connection
    # edge the step introduces is (a[i], b[j]) -- the other two corners were already joined at the
    # cell this came from -- so both branches pass it as ``stitch_triangle_metric``'s first and
    # third arguments and the rim-side corner as the second.
    if i >= 1 and out_dp[i - 1, j] < BAD_METRIC:
        a_prev = a_pos[(i - 1) % n_a]
        a_cur = a_pos[i % n_a]
        b_cur = b_pos[j % n_b]
        w = out_dp[i - 1, j] + stitch_triangle_metric(a_cur, a_prev, b_cur, up, metric_id)
        if complex_edge:
            if out_came[i - 1, j] != CAME_NONE:
                c_op = stitch_prev_apex(a_pos, b_pos, out_came, n_a, n_b, i - 1, j)
                w = w + stitch_edge_metric(a_prev, b_cur, c_op, a_cur)
            if tables.a_opp_valid[(i - 1) % n_a] != 0:
                w = w + stitch_edge_metric(a_cur, a_prev, tables.a_opp[(i - 1) % n_a], b_cur)
        update_argmin(best, best_came, w, CAME_A)

    # Advance loop B: new triangle (a[i], b[j-1], b[j]).
    if j >= 1 and out_dp[i, j - 1] < BAD_METRIC:
        a_cur = a_pos[i % n_a]
        b_prev = b_pos[(j - 1) % n_b]
        b_cur = b_pos[j % n_b]
        w = out_dp[i, j - 1] + stitch_triangle_metric(a_cur, b_prev, b_cur, up, metric_id)
        if complex_edge:
            if out_came[i, j - 1] != CAME_NONE:
                c_op = stitch_prev_apex(a_pos, b_pos, out_came, n_a, n_b, i, j - 1)
                w = w + stitch_edge_metric(a_cur, b_prev, c_op, b_cur)
            # ``rim_edge_keys`` keys slot k on the rim edge (loop[k], loop[k + 1]), so
            # the edge (b[j - 1], b[j]) introduced by this step is slot j - 1, not j -- the same
            # convention the A branch uses above.
            if tables.b_opp_valid[(j - 1) % n_b] != 0:
                w = w + stitch_edge_metric(b_prev, b_cur, tables.b_opp[(j - 1) % n_b], a_cur)
        update_argmin(best, best_came, w, CAME_B)

    return best, best_came


@wp.kernel(enable_backward=False)
def stitch_dp_diag(
    tables: StitchTables,
    diag: wp.int32,
    out_dp: wp.array2d[wp.float32],
    out_came: wp.array2d[wp.int32],
) -> None:
    # One thread per cell (i, j) on anti-diagonal ``diag = i + j``; each reads only the previous
    # diagonal, so launching diag = 1, 2, ... in order are the DP barriers.
    #
    # This is the reference schedule and the one the CPU device takes. ``stitch_dp_tile`` below
    # computes the identical cells in a different order -- the difference is *only* where the
    # barriers come from, a launch boundary here against a block barrier there -- and
    # ``tests/test_holes.py::test_stitch_dp_tile_matches_diagonal`` pins the two to each other.
    i_lo = wp.max(0, diag - tables.n_b)
    i = i_lo + wp.int32(wp.tid())
    j = diag - i
    if i > tables.n_a or j < 0 or j > tables.n_b or (i == 0 and j == 0):
        return
    best, best_came = stitch_dp_cell(tables, i, j, out_dp, out_came)
    out_dp[i, j] = best
    out_came[i, j] = best_came


# Side of the square grid tile ``stitch_dp_tile`` gives one block, and the block's lane count with
# it. The tiled schedule trades launches for block barriers: a ``(n_a + 1) x (n_b + 1)`` grid takes
# ``2 * ceil(n / TILE)`` launches instead of ``n_a + n_b``, and each block pays ``2 * TILE - 1``
# barriers for the ``TILE^2`` cells it covers.
#
# **32 because that is one warp, and the barrier is the cost.** Swept 16/32/64/128/256 against the
# diagonal schedule on both devices at rims of 128 to 2 048: CUDA takes 11.0-13.3x and peaks at 32
# at every size, falling to ~10x at 64 and ~8.5x at 256 -- which is Warp's ``tile_reduce_impl``
# taking its ``warp_count == 1`` fast path (a ballot and a shuffle, no shared round trip: 126 ns
# against 325 at 64 lanes). The CPU device also prefers the tiled schedule, 2.4-7.8x, but peaks at
# 64; 32 costs it ~4% and one constant is worth that, so this is deliberately *not* device-split.
#
# One consequence of 32 being a single warp: the block needs no barrier at all at this width, so
# removing the ``wp.tile_sum`` below still passes here. Probe that guard at 64 lanes or more, where
# it fails every run.
STITCH_DP_TILE = 32


@wp.kernel(enable_backward=False)
def stitch_dp_tile(
    tables: StitchTables,
    block_diag: wp.int32,
    bi_lo: wp.int32,
    tile: wp.int32,
    out_dp: wp.array2d[wp.float32],
    out_came: wp.array2d[wp.int32],
) -> None:
    # One *block* per ``tile x tile`` square of the grid, all the squares on one tile-diagonal
    # ``block_diag = bi + bj`` per launch. A square depends only on the squares above and to the
    # left of it, which sat on the previous tile-diagonal and so finished in the previous launch;
    # inside the square the same dependency makes its own anti-diagonals sequential, and *those*
    # barriers are block-local rather than launches. That is the whole point: the grid needs
    # ``n_a + n_b`` sequential steps either way, and this pays for all but ``2 * ceil(n / tile)``
    # of them with a block barrier at a few hundred nanoseconds instead of a launch at ~12 us.
    #
    # ``wp.tile_sum`` over a per-lane value is the barrier. Warp exposes no bare ``__syncthreads``
    # (§10 has no barrier builtin), and a block-collective reduction both synchronises and orders
    # the global writes the next diagonal reads -- verified by removing it, which fails every run
    # at 64 lanes and above and, at 32, passes only because one warp needs no barrier at all.
    #
    # **The lane stride is ``wp.block_dim()``, not ``tile``.** They agree on CUDA, where the launch
    # passes ``tile`` as its ``block_dim``. On the CPU device ``wp.launch_tiled`` runs one lane per
    # block through Warp 1.18, so the runtime value reads 1 and that single lane walks every row of
    # each anti-diagonal in turn -- correct, because the cells of one anti-diagonal are independent
    # of each other. With the constant it would compute one cell in ``tile`` and leave the rest of
    # the square at its fill value.
    blk, t = wp.tid()
    bi = bi_lo + blk
    i0 = bi * tile
    j0 = (block_diag - bi) * tile
    n_a = tables.n_a
    n_b = tables.n_b
    for d in range(2 * tile - 1):
        di = t
        while di < tile:
            dj = d - di
            if dj >= 0 and dj < tile:
                i = i0 + di
                j = j0 + dj
                # The grid is ``(n_a + 1) x (n_b + 1)``, so the last tile of each axis is partial;
                # and cell (0, 0) is the seed the driver wrote, not a cell to compute.
                if i <= n_a and j <= n_b and (i != 0 or j != 0):
                    best, best_came = stitch_dp_cell(tables, i, j, out_dp, out_came)
                    out_dp[i, j] = best
                    out_came[i, j] = best_came
            di += wp.block_dim()
        # Every lane must see this anti-diagonal's writes before reading them as the next one's
        # predecessors.
        block_barrier()


@wp.kernel
def count_loop_vertices(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_counts: wp.array[wp.int32],
    out_fillable: wp.array[wp.bool],
) -> None:
    # How many packed loop slots each mesh vertex occupies, across *every* loop at once. The
    # range test is fused in rather than left to the caller because ``out_counts`` is indexed by
    # the value read here: a loop naming a vertex the mesh does not have would otherwise scatter
    # outside the histogram, which on the CPU device is host-heap corruption rather than a wrong
    # answer. Such a loop cannot be triangulated over mesh vertices either, so it is cleared.
    t = wp.int32(wp.tid())
    v = flat_loops[t]
    if v < 0 or v >= n_vertices:
        out_fillable[loop_id[t]] = False
        return
    wp.atomic_add(out_counts, v, 1)


@wp.kernel
def scatter_fillable_loop_slots(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    vertex_counts: wp.array[wp.int32],
    out_fillable: wp.array[wp.bool],
    out_loop_slots: wp.array[wp.vec2i],
) -> None:
    # The pinch test, then which loop owns each mesh vertex and where along it, for the chord test
    # below. One occurrence count answers both pinch tests at once: a vertex holding two packed
    # slots is visited twice by one loop (that loop is pinched at it) or once by each of two loops
    # (both are pinched there), and in either case every loop touching it is unfillable -- so
    # "occupies more than one slot" is exactly the union of the two conditions. Such a vertex is
    # not written into the table, and every other one is held by exactly one slot, which is what
    # makes it single-valued with no collision to resolve -- and, since a vertex holds a slot
    # exactly when its count is 1, lets the chord test read the count as the table's validity and
    # the table start uninitialized. A loop the pinch test clears may still write its other
    # vertices, and the chord test may then clear it again: every write to ``out_fillable`` is an
    # idempotent ``False``, and a loop still fillable at the end had every vertex written. The
    # range test repeats ``count_loop_vertices``', which has already cleared such a loop.
    t = wp.int32(wp.tid())
    v = flat_loops[t]
    if v < 0 or v >= vertex_counts.shape[0]:
        return
    ell = loop_id[t]
    if vertex_counts[v] > 1:
        out_fillable[ell] = False
        return
    out_loop_slots[v] = wp.vec2i(ell, t - loop_offsets[ell])


@wp.func
def clear_loop_if_chord(
    a: wp.int32,
    b: wp.int32,
    vertex_counts: wp.array[wp.int32],
    loop_slots: wp.array[wp.vec2i],
    loop_offsets: wp.array[wp.int32],
    out_fillable: wp.array[wp.bool],
) -> None:
    # A *chord* is a mesh edge joining two vertices of one boundary loop that are not neighbours
    # along it. A min-weight fill triangulates over the loop's own vertices, so it can propose that
    # chord as a fill edge -- and the mesh already has one, which makes the result non-manifold.
    # That is the hazard ``fill_min_weight(resolve_multiple_edges=True)`` works around after the
    # fact; naming it per loop lets a caller decide before filling. Symmetric in ``a`` and ``b``.
    # ``loop_slots[v]`` is written exactly where ``vertex_counts[v] == 1``
    # (``scatter_fillable_loop_slots``), so the count is tested before the table is read.
    if vertex_counts[a] != 1 or vertex_counts[b] != 1:
        return
    slot_a = loop_slots[a]
    slot_b = loop_slots[b]
    loop = slot_a[0]
    if slot_b[0] != loop:
        return
    size = loop_offsets[loop + 1] - loop_offsets[loop]
    gap = slot_a[1] - slot_b[1]
    if gap < 0:
        gap = -gap
    if gap != 1 and gap != size - 1:
        out_fillable[loop] = False


@wp.kernel
def clear_loops_with_chords(
    faces: wp.array[wp.int32],
    vertex_counts: wp.array[wp.int32],
    loop_slots: wp.array[wp.vec2i],
    loop_offsets: wp.array[wp.int32],
    out_fillable: wp.array[wp.bool],
) -> None:
    # The chord test over every face's three edges rather than over the unique edges: the only
    # write is an idempotent ``False`` and the test is symmetric in its two ends, so an edge seen
    # from both of its faces, in either direction, gives the answer it gives once -- and no sort is
    # needed to deduplicate them.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    clear_loop_if_chord(a, b, vertex_counts, loop_slots, loop_offsets, out_fillable)
    clear_loop_if_chord(b, c, vertex_counts, loop_slots, loop_offsets, out_fillable)
    clear_loop_if_chord(c, a, vertex_counts, loop_slots, loop_offsets, out_fillable)


@wp.kernel
def loop_extreme_projection(
    vertices: wp.array[wp.vec3],
    loop_vertices: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    direction: wp.vec3,
    out_extremes: wp.array[wp.float32],
) -> None:
    # The smallest ``dot(position, direction)`` over each rim, which is where that rim's bottom
    # plane sits. ``out_extremes`` arrives pre-filled with +infinity.
    i = wp.int32(wp.tid())
    wp.atomic_min(out_extremes, loop_id[i], wp.dot(vertices[loop_vertices[i]], direction))


@wp.func
def plane_origin_from_extreme(
    extreme: wp.float32, direction: wp.vec3, extension: wp.float32
) -> wp.vec3:
    # A point on the plane through the rim's extreme vertex, pushed ``extension`` further along
    # ``-direction``. Only its component along ``direction`` matters to the projection. Formed by
    # ``extend_rim_to_ring`` as it reads it.
    return direction * (extreme - extension)


@wp.kernel
def extend_rim_to_ring(
    vertices: wp.array[wp.vec3],
    loop_vertices: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origins: wp.array[wp.vec3],
    extremes: wp.array[wp.float32],
    extension: wp.float32,
    ring_base: wp.int32,
    out_positions: wp.array[wp.vec3],
    out_faces: wp.array2d[wp.int32],
) -> None:
    # One thread per rim vertex: project it onto its loop's plane and emit the two triangles
    # bridging its rim edge to the projected ring. The bridge addresses the ring by *index*
    # (``ring_base + slot``) and never reads a projected position, so the two are independent over
    # the same rim and one launch covers both.
    t = wp.int32(wp.tid())

    # The orthogonal projection onto the loop's plane. The *ring* of these is what the rim is
    # bridged to, so the extension is exactly the ruled surface between the two. The origin is per
    # loop rather than global because a bottom plane is fitted to each rim separately.
    #
    # The shared predicate takes the offset from the origin and returns it, so the origin is
    # subtracted and added back -- one more subtract than writing the projection out, and a
    # cancellation the direct form does not have. Not bit-identical to it, and deliberately so: the
    # disagreement sits at float32 epsilon and stays there whatever the model's scale or distance
    # from the origin, rather than growing. Face buffers are unchanged, so nothing topological turns
    # on it, and the off-plane residual is a wash between the two forms.
    #
    # The origin is the caller's (``plane_origins``) or, when ``extremes`` is non-empty, formed here
    # from the loop's extreme projection (``plane_origin_from_extreme``, ``build_bottom``'s plane),
    # so no per-loop origin table is mapped first.
    a = loop_vertices[t]
    point = vertices[a]
    loop = loop_id[t]
    origin = wp.vec3()
    if extremes.shape[0] > 0:
        origin = plane_origin_from_extreme(extremes[loop], plane_normal, extension)
    else:
        origin = plane_origins[loop]
    out_positions[t] = origin + project_out_normal(point - origin, plane_normal)

    # Two triangles per rim edge, joining it to the corresponding edge of the projected ring. The
    # rim runs with the surface on its left, so the quad ``(a, b, b', a')`` is wound the other way
    # round to keep the extension's outward side the same as the mesh's.
    next_slot = loop_next_slot(loop_id, loop_offsets, t)
    b = loop_vertices[next_slot]
    projected_a = ring_base + t
    projected_b = ring_base + next_slot

    row = wp.int32(2) * t
    out_faces[row, 0] = a
    out_faces[row, 1] = projected_b
    out_faces[row, 2] = b
    out_faces[row + wp.int32(1), 0] = a
    out_faces[row + wp.int32(1), 1] = projected_a
    out_faces[row + wp.int32(1), 2] = projected_b


@wp.kernel
def bridge_edge_census(
    faces: wp.array[wp.int32], queries: wp.array2d[wp.int32], out_census: wp.array2d[wp.int32]
) -> None:
    # Everything a bridge asks of the mesh about a handful of vertex pairs ``(u, v)``, from one
    # scan of the faces' edges: ``out_census[q, 0]`` counts the face edges joining ``u`` and ``v``
    # in either direction, ``[q, 1]`` those wound ``u -> v``, and ``[q, 2]`` is one plus the third
    # corner of a face winding ``u -> v`` (``0`` when none does). ``out_census`` arrives zeroed.
    #
    # The first two answer both halves of ``holes.bridge_edges``' validation with no edge table.
    # A row of ``boundary.oriented_boundary_edges`` is a face edge whose undirected key occurs
    # exactly once, so ``(u, v)`` is on the rim wound as its face winds it exactly when both counts
    # are 1; and the patch would duplicate an edge exactly when the first is nonzero -- a row of
    # ``edges.faces_to_edges(sorted=True)``. The third is the corner ``bridge_edges_smooth``'s
    # spline needs, for a boundary edge the one face's, so at most one thread stores it.
    #
    # The corners go into a ``wp.vec3i`` rather than staying the tuple ``corner_triple`` returns,
    # because ``k`` is a *runtime* index and a tuple cannot be subscripted by one in kernel scope.
    f, q = wp.tid()
    u = queries[q, 0]
    v = queries[q, 1]
    low = wp.min(u, v)
    high = wp.max(u, v)
    c0, c1, c2 = corner_triple(faces, f)
    corner = wp.vec3i(c0, c1, c2)
    for k in range(3):
        a = corner[k]
        b = corner[_wrap(k + 1, 3)]
        if wp.min(a, b) == low and wp.max(a, b) == high:
            wp.atomic_add(out_census, q, 0, 1)
            if a == u and b == v:
                wp.atomic_add(out_census, q, 1, 1)
                out_census[q, 2] = corner[_wrap(k + 2, 3)] + 1


# Lanes per member of ``reduce_closest_cross_label_pair``.
CLOSEST_PAIR_BLOCK = 64


@wp.kernel(enable_backward=False)
def reduce_closest_cross_label_pair(
    vertices: wp.array[wp.vec3],
    members: wp.array2d[wp.int32],
    labels: wp.array[wp.int32],
    max_distance_sq: wp.float32,
    out_best: wp.array[wp.int64],
    out_partner: wp.array[wp.int32],
) -> None:
    # The globally closest pair of ``members`` carrying **different** labels, a member being the
    # first column of each row of ``members`` (the tail of an oriented boundary edge, read in place
    # rather than copied out of the strided column first), reduced into one
    # ``int64`` by ``pack_nearest_key`` -- "smallest distance, lowest index on a tie", so the answer
    # is deterministic whatever the thread order. Block ``i`` finds member ``i``'s nearest
    # cross-label partner and records it in ``out_partner[i]``, so the winning *pair* is
    # recoverable from the key (whose low half is the query index) plus one lookup.
    #
    # The scan is exhaustive: every member is tested against every other. That is ``O(B^2)`` for
    # ``B`` members and it is the deliberate choice, because the members here are *boundary*
    # vertices of a mesh with more than one component -- a single-component mesh has no cross-label
    # pair and the caller never launches this -- so ``B`` is split across the components that
    # exist. Accelerating it means a structure per round and a label predicate inside the query;
    # nothing has measured a need for that yet.
    #
    # One block per query: ``B`` is a few hundred on the benchmarked inputs, so one thread per
    # query walking every member was a short grid of long dependent chains. The lanes stride the
    # members by ``wp.block_dim()`` with the serial walk's strict-``<`` ``update_argmin``, so each
    # keeps the lowest index among its own ties, and ``block_argmin`` keeps the lowest across lanes
    # -- the serial walk's answer exactly, on the one CPU lane too.
    i, lane = wp.tid()
    n = members.shape[0]
    position = vertices[members[i, 0]]
    label = labels[i]
    best_sq = FLOAT32_INF_CONSTANT
    best = wp.int32(-1)
    for j in range(lane, n, wp.block_dim()):
        if labels[j] == label:
            continue
        update_argmin(best_sq, best, wp.length_sq(vertices[members[j, 0]] - position), j)
    best_sq, best = block_argmin(best_sq, best)
    if lane == 0:
        out_partner[i] = best
        if best >= 0 and best_sq <= max_distance_sq:
            wp.atomic_min(out_best, 0, pack_nearest_key(wp.sqrt(best_sq), i))


@wp.kernel
def edge_tail_labels(
    edges: wp.array2d[wp.int32], vertex_labels: wp.array[wp.int32], out_labels: wp.array[wp.int32]
) -> None:
    # The label of each edge's tail vertex, read through the column in place -- a Python-scope
    # gather would need the strided column cloned dense first, since it ignores an index's stride.
    i = wp.int32(wp.tid())
    out_labels[i] = vertex_labels[edges[i, 0]]


@wp.kernel
def closest_pair_rows(
    best: wp.array[wp.int64],
    partner: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    seed: wp.int64,
    out_rows: wp.array[wp.int32],
) -> None:
    # Decode ``reduce_closest_cross_label_pair``'s answer into the five integers the host needs:
    # whether any pair was found, then the two edge rows it names -- the winning slot (the key's
    # low half) and that thread's partner. One small buffer, so the host reads the whole answer in
    # one transfer instead of the key, then the partner, then the rows. ``best`` is re-armed to
    # ``seed`` once read, so a caller reusing it for the next round needs no fill.
    key = best[0]
    best[0] = seed
    if key == seed:
        out_rows[0] = 0
        return
    slot_a = wp.int32(wp.uint32(wp.uint64(key) & wp.uint64(4294967295)))
    slot_b = partner[slot_a]
    out_rows[0] = 1
    out_rows[1] = edges[slot_a, 0]
    out_rows[2] = edges[slot_a, 1]
    out_rows[3] = edges[slot_b, 0]
    out_rows[4] = edges[slot_b, 1]
