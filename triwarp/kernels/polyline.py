import warp as wp

from triwarp.constants import ALLCLOSE_ATOL_CONSTANT, ALLCLOSE_RTOL_CONSTANT, FLOAT32_INF_CONSTANT
from triwarp.kernels.array import (
    LOOP_CONDITION,
    LOOP_ROUND,
    binary_search_index,
    cross2,
    is_close_vec3,
    lift_vec2,
    loop_point,
    lowbias32,
    scanned_count,
)
from triwarp.kernels.predicates import (
    closest_point_on_segment,
    orient2d,
    plane_basis,
    point_to_segment_distance,
    project_out_normal,
    vector_angle,
)
from triwarp.kernels.reduce import (
    block_barrier,
    block_chunk_1d,
    block_max,
    block_min,
    block_sum,
    commit_block_sum,
)


@wp.func
def segment_displacement(polyline: wp.array[wp.vec3], i: wp.int32) -> wp.vec3:
    """Displacement vector ``polyline[i + 1] - polyline[i]`` of segment ``i``."""
    return polyline[i + 1] - polyline[i]


@wp.func
def project_point_to_plane(p: wp.vec3, origin: wp.vec3, unit_normal: wp.vec3) -> wp.vec3:
    """Orthogonal projection of ``p`` onto the plane through ``origin`` with ``unit_normal``."""
    return p - unit_normal * wp.dot(p - origin, unit_normal)


@wp.func
def line_squared_distance(p: wp.vec3, s: wp.vec3, d: wp.vec3, seg_sq_len: wp.float32) -> wp.float32:
    """
    Squared perpendicular distance from ``p`` to the infinite line ``s -> d``.

    Mirrors ``igl::project_to_line`` with an **unclamped** parameter ``t`` (distance to the
    line, not the segment). ``seg_sq_len`` is the precomputed ``dot(d - s, d - s)``.
    """
    dms = d - s
    smp = s - p
    t = -wp.dot(dms, smp) / seg_sq_len
    return wp.length_sq(p - wp.lerp(s, d, t))


@wp.func
def segment_length(start: wp.vec3, end: wp.vec3) -> wp.float32:
    return wp.length(end - start)


@wp.func
def segment_midpoint_and_length(start: wp.vec3, end: wp.vec3) -> tuple[wp.vec3, wp.float32]:
    return wp.lerp(start, end, 0.5), wp.length(end - start)


@wp.func
def ring_closing_flag(first: wp.vec3, last: wp.vec3) -> wp.int32:
    # ``polyline.is_closed``'s predicate (``endpoints_coincide``'s, below), on the device: ``1``
    # when the last point repeats the first within ``allclose``'s tolerance. Every thread of a
    # kernel that needs the closure evaluates it on the same two points, so all of them agree, and
    # the decision costs neither a launch of its own nor a readback.
    return wp.where(
        is_close_vec3(first, last, ALLCLOSE_RTOL_CONSTANT, ALLCLOSE_ATOL_CONSTANT),
        wp.int32(1),
        wp.int32(0),
    )


@wp.func
def closing_segment_flag(polyline: wp.array[wp.vec3], wrap_open: wp.int32) -> wp.int32:
    # ``1`` when ``closed=True`` adds a segment: the caller asked for the loop (``wrap_open``) and
    # the last point does not already repeat the first. That segment runs from ``polyline[n - 1]``
    # back to ``polyline[0]`` and is reached by wrapping the index, which is what
    # ``polyline_close``'s copy with the first point appended held at its slot ``n``. Evaluated per
    # thread on the same two points (``ring_closing_flag``), so the closure costs no readback; and
    # not at all on an open call, whose threads never load the endpoints for it.
    if wrap_open == 0:
        return wp.int32(0)
    n = polyline.shape[0]
    return wp.int32(1) - ring_closing_flag(polyline[0], polyline[n - 1])


@wp.kernel
def vertex_turning_angles(
    polyline: wp.array[wp.vec3], wrap_open: wp.int32, out_angles: wp.array[wp.float32]
) -> None:
    # One angle per vertex, written in its final slot: the angle between the segment arriving at
    # vertex ``j`` and the one leaving it, with segment ``k`` running from ``polyline[k]`` to
    # ``polyline[(k + 1) % n]``.
    #
    # Whether the ends coincide is ``ring_closing_flag`` -- ``is_closed``'s predicate -- evaluated
    # by every thread on the same two points, so neither a closure readback nor a launch of its own
    # sits in front of this one. A loop has ``n - 1`` segments and its
    # two ends are one vertex, so both take the angle between the last segment and the first; an
    # open polyline's ends have no angle and get 0. ``wrap_open`` is the caller's ``closed=True``
    # on a polyline whose ends do *not* coincide: the closing segment is reached by wrapping the
    # index -- ``n`` segments, and only vertex 0 closes -- which reproduces bit for bit what the
    # same kernel would compute over a ``polyline_close`` copy, without making the copy.
    j = wp.int32(wp.tid())
    n = polyline.shape[0]
    loop = ring_closing_flag(polyline[0], polyline[n - 1]) != 0
    n_segments = n - 1 + closing_segment_flag(polyline, wrap_open)
    before = j - 1
    after = j
    angle = wp.float32(0.0)
    if j == 0 or (j == n - 1 and n_segments == n - 1):
        before = n_segments - 1
        after = 0
    if (j != 0 and j != n - 1) or n_segments == n or loop:
        # ``vector_angle`` is scale-free, so the segments go in unnormalized.
        angle = vector_angle(
            polyline[loop_point(before + 1, n)] - polyline[before],
            polyline[loop_point(after + 1, n)] - polyline[after],
        )
    out_angles[j] = angle


@wp.func
def segment_range_distance(
    polyline: wp.array[wp.vec3], p: wp.vec3, begin: wp.int32, end: wp.int32, best: wp.float32
) -> wp.float32:
    # ``best`` lowered by the distance from ``p`` to each open segment ``begin .. end - 1``, in
    # order. The one segment loop of ``distance_to_segments`` and its sliced sibling: a minimum is
    # order-free, so however the segments are split over threads the answer is the same float.
    for i in range(begin, end):
        best = wp.min(best, point_to_segment_distance(polyline[i], polyline[i + 1], p))
    return best


@wp.func
def closing_segment_distance(
    polyline: wp.array[wp.vec3], wrap_open: wp.int32, p: wp.vec3, best: wp.float32
) -> wp.float32:
    # ``best`` lowered by the distance from ``p`` to the closing segment ``closed=True`` adds, when
    # it adds one (``closing_segment_flag``); tested after the open segments, in the order the
    # ``polyline_close`` copy put it in -- last.
    n = polyline.shape[0]
    if closing_segment_flag(polyline, wrap_open) != 0:
        best = wp.min(best, point_to_segment_distance(polyline[n - 1], polyline[0], p))
    return best


@wp.kernel
def distance_to_segments(
    points: wp.array[wp.vec3],
    polyline: wp.array[wp.vec3],
    wrap_open: wp.int32,
    out_distances: wp.array[wp.float32],
) -> None:
    # One thread per query over every segment. The closing segment ``closed=True`` adds is tested
    # *after* the open ones rather than by wrapping the loop index, so the ``points x segments``
    # loop carries no modulo, and in the same order the ``polyline_close`` copy put it in -- last.
    # ``distance_to_segment_slices`` is the same test with the segments split over a second grid
    # dimension, for a query count too small to fill the device.
    tid = wp.int32(wp.tid())
    p = points[tid]
    n = polyline.shape[0]
    best = segment_range_distance(polyline, p, 0, n - 1, wp.float32(FLOAT32_INF_CONSTANT))
    out_distances[tid] = closing_segment_distance(polyline, wrap_open, p, best)


# ``polyline_point_distance``'s slicing on CUDA: enough ``(query, slice)`` threads to fill the
# device, and slices no shorter than this many segments. Swept on this box over 272 to 65 536
# segments and 1 to 65 536 queries, answers byte-identical: 24x at 4 096 queries against 65 536
# segments, 1.6x at 65 536 against 65 536, 2.4x on a 272-segment loop at 4 096 queries, never
# slower.
# The lanes of a warp are consecutive *queries* in one slice, so they read the same segment and
# its loads broadcast; ordered the other way, lanes on different slices, it was 2-6x slower.
POINT_DISTANCE_THREADS = 1 << 22
POINT_DISTANCE_MIN_SLICE = 32


@wp.kernel
def distance_to_segment_slices(
    points: wp.array[wp.vec3],
    polyline: wp.array[wp.vec3],
    wrap_open: wp.int32,
    slice_length: wp.int32,
    out_distances: wp.array[wp.float32],
) -> None:
    # ``distance_to_segments`` with the open segments split into slices of ``slice_length``, one
    # thread per ``(query, slice)`` and an ``atomic_min`` into the query's slot, which the caller
    # seeds with ``inf``; the last slice also tests the closing segment. A query count below the
    # device's width leaves one thread walking every segment serially -- a few thousand threads on
    # a 170-SM part -- and the slices are what fill it. The minimum is order-free and every
    # distance is the same expression on the same operands, so the answer is the unsliced one bit
    # for bit.
    slice_index, tid = wp.tid()
    p = points[tid]
    n = polyline.shape[0]
    begin = slice_index * slice_length
    end = wp.min(begin + slice_length, n - 1)
    best = segment_range_distance(polyline, p, begin, end, wp.float32(FLOAT32_INF_CONSTANT))
    if end == n - 1:
        best = closing_segment_distance(polyline, wrap_open, p, best)
    wp.atomic_min(out_distances, tid, best)


@wp.kernel
def distance_to_first_point(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3], out_distances: wp.array[wp.float32]
) -> None:
    tid = wp.int32(wp.tid())
    out_distances[tid] = wp.length(points[tid] - polyline[0])


@wp.kernel
def segment_step_counts(
    polyline: wp.array[wp.vec3],
    step_size: wp.float32,
    wrap_open: wp.int32,
    out_steps: wp.array[wp.int32],
) -> None:
    # Launched over ``n - 1`` segments, or ``n`` for ``closed=True``. Slot ``n - 1`` is then the
    # closing segment when ``closing_segment_flag`` adds one, and a zero count when the input
    # already repeats its first point: a segment with no samples, which the offsets search in
    # ``segment_parameter`` never selects, so the output is the ``n - 1``-segment one. That is
    # what lets the host size the launch without asking whether the ends coincide.
    i = wp.int32(wp.tid())
    n = polyline.shape[0]
    if i == n - 1 and closing_segment_flag(polyline, wrap_open) == 0:
        out_steps[i] = 0
        return
    length = wp.length(polyline[loop_point(i + 1, n)] - polyline[i])
    out_steps[i] = wp.max(wp.int32(wp.floor(length / step_size)), wp.int32(1))


@wp.func
def segment_parameter(
    offsets: wp.array[wp.int32], steps: wp.array[wp.int32], j: wp.int32
) -> tuple[wp.int32, wp.float32]:
    # Which segment output sample ``j`` belongs to, and its parameter in ``[0, 1)`` along that
    # segment. ``offsets`` is the exclusive scan of ``steps``, so the containing segment is the last
    # offset not past ``j`` -- ``binary_search_index`` returns the first strictly greater, hence the
    # ``- 1``.
    segment = binary_search_index(offsets, j) - 1
    k = j - offsets[segment]
    return segment, wp.float32(k) / wp.float32(steps[segment])


@wp.kernel
def upsample_gather(
    polyline: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # ``loop_point`` reaches the closing segment's end, the first point (``segment_step_counts``).
    j = wp.int32(wp.tid())
    segment, weight = segment_parameter(offsets, steps, j)
    n = polyline.shape[0]
    out_points[j] = wp.lerp(polyline[segment], polyline[loop_point(segment + 1, n)], weight)


CURVATURE_EPS = wp.constant(wp.float32(1.0e-6))


@wp.func
def plane_normal(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, wp.bool]:
    """
    Return the plane normal of segment vectors ``a``, ``b``, ``c``, plus whether they are collinear.

    Returns whichever of ``b x (a + c)`` and ``b x (a - c)`` has the larger magnitude, which stays
    well-defined when ``a`` and ``c`` are nearly parallel or anti-parallel.

    The degeneracy test is scale-free: it reads ``sin^2`` of the angle between ``b`` and the chosen
    sum/difference (``length_sq(normal) / (length_sq(b) * length_sq(reference))``), not an absolute
    cross-product magnitude. ``length_sq(normal)`` scales as the *4th power* of the coordinate
    scale, so comparing it to a fixed absolute epsilon silently reclassifies an ordinary
    non-collinear bend as degenerate once the polyline's coordinates drop a couple of orders of
    magnitude below 1, collapsing a fitted arc to its straight chord.

    ``scale`` still needs a floor against a genuinely zero-length ``b`` or reference vector, which
    makes the ratio vacuously small regardless of angle -- but the floor has to be an *exact*-zero
    test and not another fixed epsilon, since ``scale`` is the same 4th-power quantity the ratio
    exists to stop comparing against a constant. A duplicated (bit-identical) point makes ``b`` or
    ``reference`` the exact zero vector at any coordinate scale, which is what the floor tests for.
    """
    n1 = wp.cross(b, a + c)
    n2 = wp.cross(b, a - c)
    if wp.length_sq(n1) >= wp.length_sq(n2):
        normal = n1
        reference = a + c
    else:
        normal = n2
        reference = a - c
    scale = wp.length_sq(b) * wp.length_sq(reference)
    degenerate = scale <= wp.float32(0.0) or wp.length_sq(normal) < CURVATURE_EPS * scale
    return normal, degenerate


@wp.func
def endpoint_normals(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """
    In-plane unit normals at the two ends of segment ``b`` bracketed by neighbours ``a``, ``c``.

    Rotate each segment 90 degrees within the fitted plane (``plane_normal``) and average the edge
    normal with each neighbour's normal. Returns two zero vectors when the segments are (nearly)
    collinear, signalling the caller to fall back to a
    straight chord.
    """
    normal, degenerate = plane_normal(a, b, c)
    if degenerate:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
    nod = wp.normalize(wp.cross(normal, b))
    no = wp.normalize(nod + wp.normalize(wp.cross(normal, a)))
    nd = wp.normalize(nod + wp.normalize(wp.cross(normal, c)))
    return no, nd


@wp.func
def arc_point(po: wp.vec3, pd: wp.vec3, no: wp.vec3, nd: wp.vec3, t: wp.float32) -> wp.vec3:
    """
    Point at parameter ``t`` in ``[0, 1]`` along the circular arc from ``po`` to ``pd``.

    The arc is the one whose unit end-normals are ``no`` and ``nd``; its midpoint offset from the
    chord is the ``(|chord| / 2) * tan(theta / 4)`` sagitta, generalised here to every ``t`` for
    multi-point subdivision. Degenerate inputs (zero-length chord, collinear neighbours
    signalled by zero normals, straight/near-straight arc, or a cusp) collapse to the straight
    chord ``po + t * (pd - po)``, so ``t == 0`` always returns ``po`` exactly.
    """
    b = pd - po
    chord = wp.length(b)
    linear = wp.lerp(po, pd, t)
    if chord < CURVATURE_EPS:
        # ``linear``, not a bare ``po``, to match the docstring's straight-chord contract and its
        # sibling degenerate branches below -- numerically indistinguishable here (``t * chord`` is
        # already under ``CURVATURE_EPS``), but ``linear`` is exactly ``po`` at ``t == 0`` too, so
        # the "t == 0 returns po exactly" guarantee is unaffected.
        return linear
    # Zero end-normals are the collinear sentinel from endpoint_normals; unit normals have norm 1.
    if wp.length_sq(no) < 0.5 or wp.length_sq(nd) < 0.5:
        return linear
    theta = vector_angle(no, nd)
    if theta < CURVATURE_EPS:
        return linear
    tangent = wp.normalize(b)
    # wp.sign is -1 below zero and +1 otherwise, matching the guard this replaces.
    bulge = wp.sign(wp.dot(b, nd - no)) * (no + nd)
    m = project_out_normal(bulge, tangent)  # bulge direction, orthogonalised against the chord
    if wp.length_sq(m) < CURVATURE_EPS:
        return linear
    m = wp.normalize(m)
    alpha = 0.5 * theta
    radius = chord / (2.0 * wp.sin(alpha))
    center = 0.5 * (po + pd) - radius * wp.cos(alpha) * m
    phi = (2.0 * t - 1.0) * alpha
    return center + radius * (wp.cos(phi) * m + wp.sin(phi) * tangent)


@wp.kernel
def smooth_upsample_gather(
    polyline: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    closed: wp.int32,
    out_points: wp.array[wp.vec3],
) -> None:
    # A loop is ``closed=True`` *or* an input whose last point already repeats its first -- the
    # module's convention -- and which of the two decides the distinct-vertex count ``m``: the
    # repeated point is one vertex, the appended closing segment (``segment_step_counts``) is not
    # a new one. Both are evaluated per thread, so ``polyline_smooth_upsample`` asks the host
    # nothing about the closure.
    j = wp.int32(wp.tid())
    segment, t = segment_parameter(offsets, steps, j)
    n = polyline.shape[0]
    po = polyline[segment]
    pd = polyline[loop_point(segment + 1, n)]
    repeated = ring_closing_flag(polyline[0], polyline[n - 1])
    # Locate the vertices bracketing this segment; interior segments fit a curvature arc, boundary
    # segments of an open polyline (missing a neighbour) stay linear.
    has_neighbours = 0
    prev_index = 0
    next_index = 0
    if closed == 1 or repeated == 1:
        m = n - repeated  # distinct vertices: a repeated last point is the first one
        prev_index = (segment - 1 + m) % m
        next_index = (segment + 2) % m
        has_neighbours = 1
    elif segment >= 1 and segment + 2 <= n - 1:
        prev_index = segment - 1
        next_index = segment + 2
        has_neighbours = 1
    if has_neighbours == 0:
        out_points[j] = wp.lerp(po, pd, t)
        return
    no, nd = endpoint_normals(po - polyline[prev_index], pd - po, polyline[next_index] - pd)
    out_points[j] = arc_point(po, pd, no, nd, t)


@wp.func
def seam_repeats_first(polyline: wp.array[wp.vec3], wrap_open: wp.int32) -> wp.int32:
    # ``1`` when a ``closed=True`` arc-length table (``arc_segment_lengths``) ends in the
    # zero-length stand-in for a closing segment the input already had -- its last entry is then
    # no point of the loop, and a walk or a search over the table stops one entry short.
    return wrap_open - closing_segment_flag(polyline, wrap_open)


@wp.kernel
def arc_segment_lengths(
    polyline: wp.array[wp.vec3], wrap_open: wp.int32, out_cumulative: wp.array[wp.float32]
) -> None:
    # Segment lengths for an arc-length table, launched over ``n - 1`` segments or ``n`` for
    # ``closed=True``. The closing slot ``n - 1`` holds the segment back to the first point when
    # ``closing_segment_flag`` adds one and an exact ``0`` when the input already repeats it, so
    # the scan's prefix is the ``n - 1``-segment table and its last entry duplicates the one
    # before (``seam_repeats_first`` tells the consumers to ignore it). One kernel rather than a
    # ``wp.map`` of ``segment_length`` over ``polyline[:-1]`` / ``polyline[1:]``: a map cannot wrap
    # the index, and the same arithmetic without the map's host-side resolution.
    #
    # Written one slot along, into the table itself, with thread 0 writing its leading zero: the
    # caller then scans ``out_cumulative[1:]`` in place, the same values at the same positions a
    # separate lengths buffer held, so neither that buffer nor a zero-fill of the table is needed.
    i = wp.int32(wp.tid())
    n = polyline.shape[0]
    if i == 0:
        out_cumulative[0] = wp.float32(0.0)
    length = wp.float32(0.0)
    if i < n - 1:
        length = segment_length(polyline[i], polyline[i + 1])
    elif closing_segment_flag(polyline, wrap_open) != 0:
        length = segment_length(polyline[n - 1], polyline[0])
    out_cumulative[i + 1] = length


@wp.kernel
def greedy_downsample_mask(
    cumulative_lengths: wp.array[wp.float32],
    step_size: wp.float32,
    polyline: wp.array[wp.vec3],
    wrap_open: wp.int32,
    out_keep: wp.array[wp.int32],
) -> None:
    # Single-thread greedy walk (dim == 1): the selection is sequential because each kept point
    # moves the reference the next one is measured from.
    #
    # **Kept for short polylines only**, and it is `polyline_downsample`'s
    # ``_DOWNSAMPLE_DOUBLING_FROM`` that decides. The walk is a few tens of nanoseconds a point, so
    # it is the cheapest thing available until the point count pays for the handful of launches
    # the parallel form below costs (``graph.pointer_jump_schedule``); the crossover is on that
    # constant, measured at 0.89x serial-over-jumped at 1 024 points and 1.15x at 2 048.
    #
    # ``out_keep`` is ``0`` / ``1`` flags rather than a mask, so the caller scans it in place for
    # the kept count and the compaction (``gather_kept_points``). ``polyline`` is read only for the
    # closure (``seam_repeats_first``); an open call passes ``wrap_open = 0`` and may pass ``None``.
    n = cumulative_lengths.shape[0] - seam_repeats_first(polyline, wrap_open)
    out_keep[0] = 1
    last = cumulative_lengths[0]
    for i in range(1, n):
        if cumulative_lengths[i] - last >= step_size:
            out_keep[i] = 1
            last = cumulative_lengths[i]


@wp.kernel
def greedy_successors(
    cumulative_lengths: wp.array[wp.float32],
    step_size: wp.float32,
    polyline: wp.array[wp.vec3],
    wrap_open: wp.int32,
    out_successor: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
) -> None:
    # The greedy walk's step function, for every point at once: ``out_successor[i]`` is the point
    # the walk would keep next *if* it had just kept ``i``, or ``n`` when the polyline ends first.
    # The walk is then the orbit of 0 under this map, which ``double_greedy_orbit`` below
    # enumerates in a logarithmic number of rounds instead of ``n`` steps. Thread 0 also seeds that
    # orbit -- the walk always keeps the first point -- so the caller's zeroed mask needs no
    # separate write.
    #
    # A hand-written lower bound rather than ``array.binary_search_index_left`` because the
    # predicate has to be **the serial kernel's, character for character**: ``cum[mid] - base`` and
    # ``cum[mid] - step`` are not the same test in float32, so searching on a shifted key would
    # move the accepted set at the boundary. It is monotone in ``mid`` because ``cum`` is
    # non-decreasing, which is what makes the search valid at all.
    #
    # The search stops at ``n_walk``, one short of the table when its last entry is a closed
    # input's repeated first point (``seam_repeats_first``), and "no successor" is still the
    # table length ``n`` -- the absorbing state ``double_greedy_orbit`` reads off the buffer size.
    # So that entry is never reached, and every other successor is the one the table without it
    # gives: the same search over the same prefix.
    i = wp.int32(wp.tid())
    n = cumulative_lengths.shape[0]
    n_walk = n - seam_repeats_first(polyline, wrap_open)
    if i == 0:
        out_keep[0] = 1
    base = cumulative_lengths[i]
    lo = i + 1
    hi = n_walk
    while lo < hi:
        mid = (lo + hi) // 2
        if cumulative_lengths[mid] - base >= step_size:
            hi = mid
        else:
            lo = mid + 1
    out_successor[i] = wp.where(lo >= n_walk, n, lo)


@wp.kernel
def double_greedy_orbit(
    successor: wp.array[wp.int32],
    reached: wp.array[wp.int32],
    hops: wp.int32,
    out_squared: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
) -> None:
    # One pointer-jumping round, both halves in one launch. Given ``successor`` holding
    # ``succ^W`` and ``reached`` holding ``{succ^t(0) : t < W}``:
    #
    # **Spread.** Mark ``succ^(t + k W)(0)`` for each reached point and ``k < hops``, so the set
    # covers ``t < hops W``. Rounds with ``hops^rounds >= n + 1`` therefore cover the whole orbit,
    # whatever its length -- the walk advances by at least ``step_size`` each time, so the orbit is
    # at most ``n`` long. ``reached`` and ``out_keep`` are **the same buffer**, updated in place,
    # and that is safe *and* deliberate. Every write is ``1``, so a lost update is impossible; a
    # thread that happens to see a mark written this round propagates extra hops, which can only
    # mark other points of the same orbit (``succ`` of an orbit point is one). So intermediate
    # rounds are nondeterministic in *which* extra points they mark and the final answer is not,
    # because the round count alone guarantees completeness.
    #
    # **Power.** ``succ^(hops W)`` from ``succ^W``, for the next round: the same chase through the
    # old table. ``n`` is the absorbing state (the walk has run off the end) and stays absorbing,
    # and the chase stops there before it would index past the table. Ping-ponged rather than
    # written in place, and that is load-bearing: in place a thread could read a slot another
    # thread had already advanced, giving ``succ^(a + b)`` for uncontrolled ``a``, ``b`` -- which
    # breaks the round count's guarantee above.
    #
    # The two halves share a launch because neither reads what the other writes this round: the
    # spread reads ``successor``, which the power only reads too, and the power never touches the
    # mask.
    i = wp.int32(wp.tid())
    n = successor.shape[0]
    marking = reached[i] != 0
    j = successor[i]
    for _ in range(hops - 1):
        if j < n:
            if marking:
                out_keep[j] = 1
            j = successor[j]
    out_squared[i] = j


@wp.kernel
def gather_kept_points(
    inclusive: wp.array[wp.int32],
    polyline: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
    out_indices: wp.array[wp.int32],
) -> None:
    # Compact the kept points of a downsample or a simplification from the in-place inclusive scan
    # of their 0/1 flags, in one launch: ``array.flatnonzero`` then ``array.gather`` is a scatter
    # of indices into their own buffer and a second pass through them. Entry ``n`` of a
    # ``closed=True`` table is the closing segment's end, the first point, hence the wrap.
    # ``out_indices`` -- the kept entries' own indices, which only the simplification returns --
    # may be ``None``, a null descriptor of length zero that no thread then writes.
    i = wp.int32(wp.tid())
    start, count = scanned_count(inclusive, i)
    if count != 0:
        out_points[start] = polyline[loop_point(i, polyline.shape[0])]
        if out_indices.shape[0] != 0:
            out_indices[start] = i


RDP_LINE_EPS = wp.constant(wp.float32(1.0e-7))  # libigl FLOAT_EPS: degenerate-segment threshold
RDP_SETTLED = wp.constant(wp.int32(-1))  # ``span_lo`` sentinel: this point's fate is decided


@wp.func
def rdp_chord_squared_distance(
    polyline: wp.array[wp.vec3], i: wp.int32, lo: wp.int32, hi: wp.int32
) -> wp.float32:
    """Squared distance from ``polyline[i]`` to the chord ``polyline[lo] -> polyline[hi]``."""
    # ``hi`` may be entry ``n`` of a ``closed=True`` loop, the first point again (``loop_point``);
    # ``lo`` and ``i`` lie strictly before it.
    start = polyline[lo]
    end = polyline[loop_point(hi, polyline.shape[0])]
    seg_sq_len = wp.length_sq(end - start)
    if seg_sq_len <= RDP_LINE_EPS:
        return wp.length_sq(polyline[i] - start)  # degenerate chord: distance to the shared point
    return line_squared_distance(polyline[i], start, end, seg_sq_len)


@wp.func
def rdp_loop_length(polyline: wp.array[wp.vec3], wrap_open: wp.int32) -> wp.int32:
    # Entries of the polyline the recursion runs over: ``n``, plus the closing point ``closed=True``
    # adds when the input does not already repeat its first (``closing_segment_flag``).
    # The caller's buffers are sized ``n + 1`` for ``closed=True`` whatever the answer, so entry
    # ``n`` of an input that already repeats its first point is a dead slot, settled and dropped.
    return polyline.shape[0] + closing_segment_flag(polyline, wrap_open)


@wp.func
def rdp_seed_point(
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    i: wp.int32,
    n: wp.int32,
) -> None:
    # Ramer-Douglas-Peucker, level-synchronous: one round per level of the recursion tree instead
    # of one thread walking the whole tree. Round 0 puts every interior point in the single span
    # ``(0, n - 1)``; the two endpoints are kept unconditionally and never belong to a span.
    #
    # A span is identified by its **left endpoint**, and that is the whole reason there is no span
    # list to build or compact: the open spans at any level partition the polyline, so their left
    # endpoints are distinct and index a plain ``(n,)`` accumulator directly.
    #
    # Each per-point step of a round is one ``@wp.func`` here, shared by the four-launch round
    # (``rdp_seed_spans`` ... ``rdp_split_spans``) and the one-block loop (``rdp_simplify_block``).
    #
    # ``n`` is ``rdp_loop_length``; an index at or past it is the dead slot, settled and not kept.
    # ``keep`` holds ``0`` / ``1`` flags rather than a mask, so the caller scans it in place.
    if i == 0 or i >= n - 1:
        span_lo[i] = RDP_SETTLED
        span_hi[i] = RDP_SETTLED
        keep[i] = wp.where(i == n - 1 or i == 0, wp.int32(1), wp.int32(0))
    else:
        span_lo[i] = 0
        span_hi[i] = n - 1
        keep[i] = wp.int32(0)


@wp.kernel
def rdp_seed_spans(
    polyline: wp.array[wp.vec3],
    wrap_open: wp.int32,
    out_span_lo: wp.array[wp.int32],
    out_span_hi: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Round 0 (``rdp_seed_point``). Thread 0 also seeds the round loop's ``state`` -- ``[levels
    # run, loop condition]`` -- so the loop needs no host upload of its own.
    i = wp.int32(wp.tid())
    if i == 0:
        out_state[LOOP_ROUND] = wp.int32(0)
        out_state[LOOP_CONDITION] = wp.int32(1)
    rdp_seed_point(out_span_lo, out_span_hi, out_keep, i, rdp_loop_length(polyline, wrap_open))


@wp.func
def rdp_arm_span(
    span_max: wp.array[wp.float32], span_argmax: wp.array[wp.int32], i: wp.int32
) -> None:
    # Arm span slot ``i``'s accumulators: no maximum yet, and an argmax past every valid index, so
    # ``atomic_min`` always wins.
    span_max[i] = -1.0
    span_argmax[i] = span_max.shape[0]


@wp.kernel
def rdp_begin_round(
    state: wp.array[wp.int32],
    out_span_max: wp.array[wp.float32],
    out_span_argmax: wp.array[wp.int32],
) -> None:
    # Round, pass 1 of 4: arm the per-span accumulators and clear the loop condition. Kept as its
    # own launch rather than folded into the split kernel (which knows each child span's slot)
    # because a ping-ponged pair of accumulators cannot be swapped inside a captured graph -- the
    # buffers are baked in at capture time. One extra ``dim=n`` launch per round buys the readback.
    #
    # ``state`` is read-and-incremented round-index/loop-condition scratch carried across launches
    # by the caller (the same buffer ``rdp_split_spans`` takes under this name), not a fresh
    # per-call answer -- it does not wear the ``out_`` prefix for that reason.
    i = wp.int32(wp.tid())
    if i == 0:
        state[LOOP_ROUND] = state[LOOP_ROUND] + 1
        state[LOOP_CONDITION] = 0
    rdp_arm_span(out_span_max, out_span_argmax, i)


@wp.func
def rdp_measure_point(
    polyline: wp.array[wp.vec3],
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    squared_distances: wp.array[wp.float32],
    span_max: wp.array[wp.float32],
    i: wp.int32,
) -> None:
    # Every unsettled point measures itself against its span's chord and max-reduces into the
    # span's slot. One thread per *point* rather than per span, so a round costs the same whatever
    # shape the level has -- which is what makes the depth, and not the span sizes, the cost model.
    lo = span_lo[i]
    if lo >= 0:
        squared_distance = rdp_chord_squared_distance(polyline, i, lo, span_hi[i])
        squared_distances[i] = squared_distance
        wp.atomic_max(span_max, lo, squared_distance)


@wp.kernel
def rdp_span_max(
    polyline: wp.array[wp.vec3],
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    out_squared_distances: wp.array[wp.float32],
    out_span_max: wp.array[wp.float32],
) -> None:
    # Round, pass 2 of 4 (``rdp_measure_point``).
    i = wp.int32(wp.tid())
    rdp_measure_point(polyline, span_lo, span_hi, out_squared_distances, out_span_max, i)


@wp.func
def rdp_claim_argmax(
    span_lo: wp.array[wp.int32],
    squared_distances: wp.array[wp.float32],
    span_max: wp.array[wp.float32],
    span_argmax: wp.array[wp.int32],
    i: wp.int32,
) -> None:
    # Recover *which* point won. Reducing the index with ``atomic_min`` over every point holding
    # the span's maximum keeps the lowest such index, which is exactly what the strict '>' argmax
    # of the recursive form kept (Eigen maxCoeff, and libigl's tie convention).
    #
    # A packed ``(bits, ~index)`` int64 key would fold this into the measuring step --
    # ``wp.atomic_max`` on ``wp.int64`` works on both devices -- and is not used, because the float
    # comparison here is against a value this same expression produced, so it is exact without
    # reinterpreting bits.
    lo = span_lo[i]
    if lo >= 0 and squared_distances[i] >= span_max[lo]:
        wp.atomic_min(span_argmax, lo, i)


@wp.kernel
def rdp_span_argmax(
    span_lo: wp.array[wp.int32],
    squared_distances: wp.array[wp.float32],
    span_max: wp.array[wp.float32],
    out_span_argmax: wp.array[wp.int32],
) -> None:
    # Round, pass 3 of 4 (``rdp_claim_argmax``).
    i = wp.int32(wp.tid())
    rdp_claim_argmax(span_lo, squared_distances, span_max, out_span_argmax, i)


@wp.func
def rdp_split_point(
    squared_tolerance: wp.float32,
    span_max: wp.array[wp.float32],
    span_argmax: wp.array[wp.int32],
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    i: wp.int32,
) -> wp.int32:
    # Split or settle point ``i``, returning ``1`` when it survives into the next level.
    # ``span_lo`` / ``span_hi`` are the point's span and are rewritten in place to its child span.
    #
    # The keep set is identical to the recursive form's by construction: there, everything starts
    # kept and a span within tolerance drops its interior; here, nothing starts kept and every
    # split point is kept. Both leave exactly the endpoints and the split points, because the
    # terminal spans partition the polyline. The loop terminates because a child span is strictly
    # narrower than its parent and a span two wide holds a single point, which settles either way.
    lo = span_lo[i]
    if lo < 0:
        return wp.int32(0)
    hi = span_hi[i]
    split = span_argmax[lo]
    # ``split <= lo or split >= hi`` catches an unresolved argmax -- ``split`` still at the sentinel
    # ``rdp_arm_span`` armed -- and settling the span turns what would be a read past the end of
    # the polyline on the next round into a dropped interior.
    #
    # It is **defensive and measured to be unreachable**, which is worth saying because the obvious
    # reason to expect otherwise is wrong: a non-finite coordinate does *not* produce it, because
    # ``wp.atomic_max`` does not propagate ``NaN`` (verified on both devices). So either some point
    # wrote a real maximum, and that same point then satisfies the ``>=`` in ``rdp_claim_argmax``
    # and resolves the index; or every interior distance was ``NaN``, the accumulator keeps the
    # ``-1.0`` it was armed with, and the tolerance test above settles the span first. Every
    # non-finite shape probed gives byte-identical answers with the two comparisons deleted.
    if span_max[lo] <= squared_tolerance or split <= lo or split >= hi:
        span_lo[i] = RDP_SETTLED  # the whole span is within tolerance, so its interior drops
        return wp.int32(0)
    if i == split:
        keep[i] = wp.int32(1)
        span_lo[i] = RDP_SETTLED
        return wp.int32(0)
    if i < split:
        span_hi[i] = split  # ``lo < i < split``, so the child span is never degenerate
    else:
        span_lo[i] = split
    return wp.int32(1)


@wp.kernel
def rdp_split_spans(
    squared_tolerance: wp.float32,
    span_max: wp.array[wp.float32],
    span_argmax: wp.array[wp.int32],
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
) -> None:
    # Round, pass 4 of 4 (``rdp_split_point``). ``span_lo`` / ``span_hi`` are neither an input nor
    # the answer; ``state`` is the round loop's own condition, raised whenever a point survives
    # into the next level.
    i = wp.int32(wp.tid())
    if rdp_split_point(squared_tolerance, span_max, span_argmax, span_lo, span_hi, out_keep, i):
        # A plain store, not an atomic: one address, one value, nothing to serialize (the rule is
        # on ``array.LOOP_CONDITION``, which also says why the array must be zero-initialized).
        state[LOOP_CONDITION] = 1


# Polylines up to this many points are simplified by ``rdp_simplify_block`` -- the whole round loop
# as one block -- rather than by the captured four-launch round, on CUDA; the CPU device takes the
# block form at every size, where a launch grid is a serial loop anyway (1.5-5.4x from 64 to 8 192
# points, keep masks byte-identical). Measured on CUDA over a spiral and a random walk: 2.0-2.3x to
# 1 024 points, 1.6-1.7x at 2 048, 1.28-1.30x at 4 096, and 0.86-0.96x at 8 192, where each lane
# walks eight points a step and the device-wide form wins again. Past ``ear_clip_block``'s 1 024
# because a round here is four streaming passes, not an O(ring) ear test per corner.
RDP_ONE_BLOCK_MAX = 4096
RDP_BLOCK_DIM = 1024


@wp.kernel(enable_backward=False)
def rdp_simplify_block(
    polyline: wp.array[wp.vec3],
    wrap_open: wp.int32,
    squared_tolerance: wp.float32,
    out_spans: wp.array2d[wp.int32],
    out_span_values: wp.array2d[wp.float32],
) -> None:
    # ``rdp_seed_spans`` plus every round of ``rdp_begin_round`` -> ``rdp_span_max`` ->
    # ``rdp_span_argmax`` -> ``rdp_split_spans``, as one block whose lanes stride the polyline by
    # ``wp.block_dim()`` -- the same per-point steps, so on the CPU device, where a block is one
    # lane, it is the same serial walk the launches make and the keep mask is byte-identical. The
    # block barriers are ``block_sum`` calls (Warp exposes no other): one after each step, the
    # last of which also counts the survivors, so the loop condition is block-uniform. What it
    # removes is the conditional graph the launch form records on every call, which at a thousand
    # points is most of the call; ``ear_clip_block`` is the same trade.
    #
    # ``out_spans`` is ``(4, n_entries)``: rows ``span_lo``, ``span_hi``, ``span_argmax`` and the
    # kept flags, which the caller scans in place; ``out_span_values`` is ``(2, n_entries)``:
    # ``span_max`` and the squared distances. Two allocations rather than six.
    _block, lane = wp.tid()
    span_lo = out_spans[0]
    span_hi = out_spans[1]
    span_argmax = out_spans[2]
    keep = out_spans[3]
    span_max = out_span_values[0]
    squared_distances = out_span_values[1]
    n_entries = out_spans.shape[1]
    n = rdp_loop_length(polyline, wrap_open)
    for i in range(lane, n_entries, wp.block_dim()):
        rdp_seed_point(span_lo, span_hi, keep, i, n)
    open_points = block_sum(wp.int32(1))
    while open_points > 0:
        for i in range(lane, n_entries, wp.block_dim()):
            rdp_arm_span(span_max, span_argmax, i)
        block_barrier()
        for i in range(lane, n_entries, wp.block_dim()):
            rdp_measure_point(polyline, span_lo, span_hi, squared_distances, span_max, i)
        block_barrier()
        for i in range(lane, n_entries, wp.block_dim()):
            rdp_claim_argmax(span_lo, squared_distances, span_max, span_argmax, i)
        block_barrier()
        survivors = wp.int32(0)
        for i in range(lane, n_entries, wp.block_dim()):
            survivors += rdp_split_point(
                squared_tolerance, span_max, span_argmax, span_lo, span_hi, keep, i
            )
        open_points = block_sum(survivors)


@wp.kernel
def broadcast_first_point(polyline: wp.array[wp.vec3], out_points: wp.array[wp.vec3]) -> None:
    j = wp.int32(wp.tid())
    out_points[j] = polyline[0]


@wp.kernel
def resample_interp(
    polyline: wp.array[wp.vec3],
    cumulative_lengths: wp.array[wp.float32],
    num_points: wp.int32,
    wrap_open: wp.int32,
    out_points: wp.array[wp.vec3],
) -> None:
    # Linear interpolation at evenly spaced arc lengths, mimicking numpy.interp:
    # constant (clamped) extrapolation at the endpoints.
    #
    # ``num_points`` is the *spacing* count: ``closed=True`` launches ``num_points`` threads at a
    # spacing of ``num_points + 1`` samples, so its seam sample is never computed rather than
    # computed and sliced off. Table entry ``k`` is point ``loop_point(k, n)`` -- entry ``n`` of a
    # closed table is the closing segment's end, the first point -- and a table ending in a
    # repeated first point (``seam_repeats_first``) is read ``n_table`` long. The search runs over
    # the whole table: its last entry then equals the one before, so every in-range answer is the
    # shorter table's, and an out-of-range one lands on ``n_table``'s last point either way.
    j = wp.int32(wp.tid())
    n = polyline.shape[0]
    n_table = cumulative_lengths.shape[0] - seam_repeats_first(polyline, wrap_open)
    total = cumulative_lengths[n_table - 1]
    x = wp.float32(0.0)
    if num_points > 1:
        x = wp.float32(j) / wp.float32(num_points - 1) * total
    hi = binary_search_index(cumulative_lengths, x)
    if hi == 0:
        out_points[j] = polyline[0]
    elif hi >= n_table:
        out_points[j] = polyline[loop_point(n_table - 1, n)]
    else:
        denominator = cumulative_lengths[hi] - cumulative_lengths[hi - 1]
        t = wp.float32(0.0)
        if denominator > 0.0:
            t = (x - cumulative_lengths[hi - 1]) / denominator
        out_points[j] = wp.lerp(polyline[hi - 1], polyline[loop_point(hi, n)], t)


@wp.func
def radius_segment_distances(
    start: wp.vec3, end: wp.vec3, center: wp.vec3, normal: wp.vec3
) -> wp.float32:
    # In-plane distance from ``center`` to one projected segment. ``normal`` need not be unit.
    unit_normal = wp.normalize(normal)
    a = project_point_to_plane(start, center, unit_normal)
    b = project_point_to_plane(end, center, unit_normal)
    return wp.length(closest_point_on_segment(a, b, center) - center)


# Slot layout of ``accumulate_radius_frame``'s buffer: the length-weighted midpoint sums
# ``polyline_centroid`` divides -- ``sum(midpoint * length)`` in slots 0..2, ``sum(length)`` in slot
# 3 -- then Newell's normal.
RADIUS_FRAME_LENGTH = wp.constant(wp.int32(3))
RADIUS_FRAME_NORMAL = wp.constant(wp.int32(4))
RADIUS_FRAME_SIZE = 7


@wp.kernel
def accumulate_radius_frame(
    polyline: wp.array[wp.vec3],
    n_segments: wp.int32,
    wrap_open: wp.int32,
    with_normal: wp.int32,
    out_frame: wp.array[wp.float32],
) -> None:
    # A polyline's plane frame in one pass, for ``polyline_centroid``, ``polyline_normal`` and
    # ``polyline_radius``'s defaults: the length-weighted midpoint sums over ``n_segments`` segments
    # plus the closing one ``closing_segment_flag`` adds for ``wrap_open`` -- ``n - 1``, or ``n``
    # for a ``closed=True`` loop whose closing segment is reached by wrapping the index -- and, with
    # ``with_normal``, Newell's sum over the loop -- ``n - 1`` pairs when the last point repeats the
    # first (``ring_closing_flag``, decided here rather than by a launch of its own), ``n`` with the
    # index wrapped otherwise. The two ranges share their chunks and their lane stride, so each
    # lane accumulates every term in the order a kernel of either alone would, and one
    # ``block_sum`` over the packed seven is componentwise the two block sums: the sums are the
    # ones two separate kernels produce, bit for bit. ``polyline_centroid`` asks for the midpoint
    # sums alone and ``polyline_normal`` (``n_segments = 0``) for the Newell sum alone.
    #
    # Launch it over ``blocks_1d(n)``, which covers both ranges: neither exceeds ``n``.
    #
    # This kernel, ``accumulate_loop_frame`` and ``accumulate_turning_angle`` are the lane-strided
    # single-slot reduction of CLAUDE.md section 13.2: ``wp.launch_tiled(dim=blocks_1d(n),
    # block_dim=TILE_1D)``, lanes striding their own block's chunk by ``wp.block_dim()``, one
    # atomic commit per block. Striding by ``wp.block_dim()`` is what makes it correct on both
    # devices, so there is no ``prefers_tiled_reduction`` branch. Only an *unconditional* atomic
    # belongs in this shape.
    #
    # A ``closed=True`` loop is the one ``polyline_close`` would build, without the copy: its
    # Newell pairs are this input's ``n - ring_closing_flag`` either way, and its segments' ends
    # are ``loop_point``'s, so the sums are the copy's bit for bit.
    chunk, lane = wp.tid()
    n = polyline.shape[0]
    segments = n_segments + closing_segment_flag(polyline, wrap_open)
    n_pairs = wp.where(
        with_normal != 0, n - ring_closing_flag(polyline[0], polyline[n - 1]), wp.int32(0)
    )
    offset, count = block_chunk_1d(wp.max(segments, n_pairs), chunk)
    if count <= 0:
        return
    weighted = wp.vec3(0.0, 0.0, 0.0)
    total = wp.float32(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        start = polyline[i]
        if i < segments:
            midpoint, length = segment_midpoint_and_length(start, polyline[loop_point(i + 1, n)])
            weighted += midpoint * length
            total += length
        if i < n_pairs:
            normal += wp.cross(start, polyline[loop_point(i + 1, n)])
    commit_block_sum(
        lane,
        wp.vector(weighted[0], weighted[1], weighted[2], total, normal[0], normal[1], normal[2]),
        out_frame,
        0,
    )


@wp.func
def radius_plane(
    frame: wp.array[wp.float32],
    center: wp.vec3,
    normal: wp.vec3,
    frame_center: wp.int32,
    frame_normal: wp.int32,
) -> tuple[wp.vec3, wp.vec3]:
    # ``polyline_radius``'s plane: the centre and the normal each taken either from the caller or,
    # where it left them to default, from ``accumulate_radius_frame``'s sums -- the centroid is the
    # sums' quotient and the normal the normalized Newell vector, the values ``polyline_centroid``
    # / ``polyline_normal`` return.
    plane_center = center
    if frame_center != 0:
        plane_center = wp.vec3(frame[0], frame[1], frame[2]) / frame[RADIUS_FRAME_LENGTH]
    plane_normal = normal
    if frame_normal != 0:
        plane_normal = wp.normalize(
            wp.vec3(
                frame[RADIUS_FRAME_NORMAL],
                frame[RADIUS_FRAME_NORMAL + 1],
                frame[RADIUS_FRAME_NORMAL + 2],
            )
        )
    return plane_center, plane_normal


@wp.kernel
def radius_distances(
    polyline: wp.array[wp.vec3],
    frame: wp.array[wp.float32],
    center: wp.vec3,
    normal: wp.vec3,
    frame_center: wp.int32,
    frame_normal: wp.int32,
    out_distances: wp.array[wp.float32],
) -> None:
    # dim == n_segments (``accumulate_radius_frame``'s, the closing one wrapping its index):
    # ``radius_segment_distances`` per segment about ``radius_plane``. ``polyline_radius``'s median,
    # which needs every distance; ``radius_reduce`` below folds the other reductions instead.
    i = wp.int32(wp.tid())
    plane_center, plane_normal = radius_plane(frame, center, normal, frame_center, frame_normal)
    out_distances[i] = radius_segment_distances(
        polyline[i], polyline[loop_point(i + 1, polyline.shape[0])], plane_center, plane_normal
    )


# ``radius_reduce``'s reductions, and its two-slot result: the reduced value, then the segment
# count ``closing_segment_flag`` decided -- the mean's divisor, which the host does not know.
RADIUS_MIN = wp.constant(wp.int32(0))
RADIUS_MAX = wp.constant(wp.int32(1))
RADIUS_SUM = wp.constant(wp.int32(2))
RADIUS_RESULT_COUNT = wp.constant(wp.int32(1))
RADIUS_RESULT_SIZE = 2


@wp.kernel
def radius_reduce(
    polyline: wp.array[wp.vec3],
    frame: wp.array[wp.float32],
    center: wp.vec3,
    normal: wp.vec3,
    frame_center: wp.int32,
    frame_normal: wp.int32,
    wrap_open: wp.int32,
    kind: wp.int32,
    out_result: wp.array[wp.float32],
) -> None:
    # ``radius_distances`` folded where it is computed, for ``polyline_radius``'s ``min`` / ``max``
    # / ``mean``: no ``(n_segments,)`` buffer, no reduction launch over it, and the segment count
    # decided per thread (``closing_segment_flag``, the closure ``is_closed`` would read back), so
    # a ``closed=True`` call asks the host nothing. ``kind`` is warp-uniform, which keeps each block
    # reduction below it block-uniform. Launched ``wp.launch_tiled(dim=blocks_1d(n),
    # block_dim=TILE_1D)``, the lane-strided single-slot shape of ``accumulate_radius_frame``;
    # ``out_result[0]`` is seeded with the reduction's identity by the caller. A minimum or a
    # maximum is order-free, so those answers are the reduction over the buffer's exactly; the sum
    # is the tree ``reduce.sum`` forms from 64 elements up.
    chunk, lane = wp.tid()
    n = polyline.shape[0]
    n_segments = n - 1 + closing_segment_flag(polyline, wrap_open)
    if chunk == 0 and lane == 0:
        out_result[RADIUS_RESULT_COUNT] = wp.float32(n_segments)
    offset, count = block_chunk_1d(n_segments, chunk)
    if count <= 0:
        return
    plane_center, plane_normal = radius_plane(frame, center, normal, frame_center, frame_normal)
    low = wp.float32(FLOAT32_INF_CONSTANT)
    high = -low
    total = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        d = radius_segment_distances(
            polyline[i], polyline[loop_point(i + 1, n)], plane_center, plane_normal
        )
        low = wp.min(low, d)
        high = wp.max(high, d)
        total += d
    if kind == RADIUS_MIN:
        block_low = block_min(low)
        if lane == 0:
            wp.atomic_min(out_result, 0, block_low)
    elif kind == RADIUS_MAX:
        block_high = block_max(high)
        if lane == 0:
            wp.atomic_max(out_result, 0, block_high)
    else:
        block_total = block_sum(total)
        if lane == 0:
            wp.atomic_add(out_result, 0, block_total)


# --- polygon triangulation (parallel ear clipping); port of libigl ear_clipping.cpp ---


@wp.func
def point_in_triangle(a: wp.vec2, b: wp.vec2, c: wp.vec2, p: wp.vec2) -> wp.bool:
    """
    Whether ``p`` lies inside or on the boundary of the CCW triangle ``(a, b, c)``.

    Boundary inclusion matters for the ear test: a (reflex) vertex lying exactly on a candidate
    ear's cutting diagonal must block that ear, otherwise a degenerate/overlapping triangle is
    emitted.
    """
    return orient2d(a, b, p) >= 0 and orient2d(b, c, p) >= 0 and orient2d(c, a, p) >= 0


@wp.func
def mirror_y(p: wp.vec2) -> wp.vec2:
    # The reflection that turns a clockwise loop counter-clockwise, shared so the reflex count taken
    # before the ear loop and the points the ear tests read (``ring_point``) are the same values.
    return wp.vec2(p[0], -p[1])


@wp.func
def ring_point(points2d: wp.array[wp.vec2], k: wp.int32, mirror: wp.int32) -> wp.vec2:
    # Ring point ``k`` as the ear tests read it: mirrored (``mirror_y``) when the ring runs
    # clockwise, so a clockwise ring is tested as the counter-clockwise loop the convex and ear
    # tests assume without a pass writing the mirrored loop first. The reflection is exact, so the
    # tests see the very values a mirrored copy would hold. ``mirror`` is warp-uniform.
    p = points2d[k]
    if mirror != 0:
        p = mirror_y(p)
    return p


@wp.func
def is_ear_at(
    points2d: wp.array[wp.vec2],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    i: wp.int32,
    n: wp.int32,
    mirror: wp.int32,
) -> wp.bool:
    # Corner (a, i, b) is an ear iff it is strictly convex and no other active vertex lies
    # strictly inside triangle (a, i, b). Equivalent to libigl's edge-intersection walk for a
    # simple polygon, but simpler to evaluate in parallel per corner.
    #
    # This walk is O(active ring size) per convex candidate corner (a reflex one returns above,
    # in O(1)), so the round with the most active convex candidates -- always round 0 on a ring
    # that is not fully convex, since ``compute_ears`` is launched at ``dim=n`` -- costs O(n) per
    # thread across up to n threads: **O(n^2) total device work**. Measured on a ring built to
    # maximize it, ``polyline_triangulate`` stays close to linear in wall clock while the device
    # still has spare resident threads to hide the O(n) longest thread behind, and turns quadratic
    # past that point. Every polyline this package's own benchmark axis exercises sits inside the
    # near-linear regime; a fix would need a spatially accelerated ear test (a real data structure
    # over the active ring, not a topological linked-list walk) rather than a tuning constant, so
    # this is left as a documented limitation rather than rewritten.
    a = left[i]
    b = right[i]
    if a == b or a == i or b == i:
        return False
    pa = ring_point(points2d, a, mirror)
    pi = ring_point(points2d, i, mirror)
    pb = ring_point(points2d, b, mirror)
    if orient2d(pa, pi, pb) <= 0:
        return False
    # Walk the remaining ring from R[b] up to a, skipping the ear's own vertices.
    j = right[b]
    while j != a:
        if (
            active[j] == 1
            and j != i
            and point_in_triangle(pa, pi, pb, ring_point(points2d, j, mirror))
        ):
            return False
        j = right[j]
    return True


@wp.func
def project_to_plane_2d(point: wp.vec3, center: wp.vec3, u: wp.vec3, v: wp.vec3) -> wp.vec2:
    d = point - center
    return wp.vec2(wp.dot(d, u), wp.dot(d, v))


# Slot layout of the one float32 buffer ``polyline_triangulate`` / ``triangulate_polygon`` read
# back once: the plane frame's seven sums (``accumulate_loop_frame``), then the ring's turning
# angle and two reflex counts (``accumulate_turning_angle``), then the closing flag -- ``1.0`` when
# the input's last point repeats its first and the ring is one point shorter. The flag rides in a
# float slot so the whole prologue is one readback; a 0/1 value is exact in float32.
FRAME_NORMAL = wp.constant(wp.int32(0))
FRAME_WEIGHTED_MIDPOINT = wp.constant(wp.int32(3))
FRAME_LENGTH = wp.constant(wp.int32(6))
RING_TURNING = wp.constant(wp.int32(7))
RING_CLOSING = wp.constant(wp.int32(10))
RING_SUMS_SIZE = 11


@wp.kernel
def accumulate_loop_frame(polyline: wp.array[wp.vec3], out_sums: wp.array[wp.float32]) -> None:
    # Over the loop of ``n_ring`` distinct vertices: the input minus a repeated closing point, which
    # this kernel detects itself (``ring_closing_flag``) and publishes in ``RING_CLOSING`` for the
    # caller's one readback, so the ring length never costs a readback of its own.
    #
    # Newell's normal is cyclic -- element i takes the edge (i, (i + 1) % n_ring), so the
    # wrap-around edge is element n_ring - 1 and no closing vertex has to be appended first. The
    # length-weighted centroid deliberately is *not* cyclic: it runs over the n_ring - 1 open
    # segments, which is what ``polyline_centroid`` (``closed=False``) computes and what this
    # function has always used.
    #
    # Lane-strided single-slot reduction -- see ``accumulate_radius_frame`` for the shape and why
    # no device branch is needed. The two are not one kernel: this one wraps its Newell pairs over
    # the ``n_ring`` distinct vertices, so a repeated closing point's pair ends on ``polyline[0]``,
    # where that one ends on the repeated point itself -- equal only within ``allclose``'s
    # tolerance, so the sums differ -- and it also publishes the closing flag.
    chunk, lane = wp.tid()
    n = polyline.shape[0]
    closing = ring_closing_flag(polyline[0], polyline[n - 1])
    n_ring = n - closing
    if chunk == 0 and lane == 0:
        out_sums[RING_CLOSING] = wp.float32(closing)
    offset, count = block_chunk_1d(n_ring, chunk)
    if count <= 0:
        return
    normal = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    weighted = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    length_total = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        start = polyline[i]
        normal += wp.cross(start, polyline[loop_point(i + 1, n_ring)])
        if i + 1 < n_ring:
            midpoint, length = segment_midpoint_and_length(start, polyline[i + 1])
            weighted += midpoint * length
            length_total += length
    # All seven sums in one block reduction, into ``FRAME_NORMAL`` .. ``FRAME_LENGTH``.
    commit_block_sum(
        lane,
        wp.vector(
            normal[0], normal[1], normal[2], weighted[0], weighted[1], weighted[2], length_total
        ),
        out_sums,
        FRAME_NORMAL,
    )


@wp.kernel
def project_polyline_to_plane(
    polyline: wp.array[wp.vec3], sums: wp.array[wp.float32], out_points2d: wp.array[wp.vec2]
) -> None:
    # dim == n (the ring plus, when ``RING_CLOSING`` is set, the repeated closing point, whose
    # projection nothing reads). Each thread turns ``accumulate_loop_frame``'s sums into the plane
    # frame itself -- the centre and ``plane_basis``' ``(u, v)`` -- so the frame never crosses to
    # the host and no single-thread launch has to build it first. Every thread evaluates the same
    # expressions on the same values, so all of them hold the one frame, bit for bit.
    i = wp.int32(wp.tid())
    normal = wp.vec3(sums[FRAME_NORMAL], sums[FRAME_NORMAL + 1], sums[FRAME_NORMAL + 2])
    weighted_midpoint = wp.vec3(
        sums[FRAME_WEIGHTED_MIDPOINT],
        sums[FRAME_WEIGHTED_MIDPOINT + 1],
        sums[FRAME_WEIGHTED_MIDPOINT + 2],
    )
    u, v = plane_basis(normal)
    center = weighted_midpoint / sums[FRAME_LENGTH]
    out_points2d[i] = project_to_plane_2d(polyline[i], center, u, v)


@wp.kernel
def accumulate_turning_angle(
    points2d: wp.array[wp.vec2], detect_closing: wp.int32, out_sums: wp.array[wp.float32]
) -> None:
    # Cyclic signed exterior angle at each vertex into ``out_sums[RING_TURNING]``; the sum's sign
    # gives the loop orientation. Lane-strided single-slot reduction -- see
    # ``accumulate_radius_frame`` for the shape and why no device branch is needed.
    #
    # Over the ring of ``n_ring`` distinct vertices. Where the ring is the caller's own 2D input
    # (``detect_closing != 0``, ``triangulate_polygon``) this kernel is the first to touch it, so it
    # detects a repeated closing point itself and publishes the flag in ``RING_CLOSING``; the
    # predicate is ``polyline_open``'s on the ``z = 0`` lift, which is what that function used to
    # evaluate. Otherwise (``polyline_triangulate``) ``accumulate_loop_frame`` already decided it on
    # the 3D input, and this kernel reads the flag it wrote.
    #
    # The same pass counts the reflex vertices, so the convex test needs no launch of its own --
    # and counts them twice, because which loop gets tested is not known until the total is:
    # ``orient_ccw`` mirrors a clockwise loop in ``y`` afterwards. Slot ``RING_TURNING + 1`` counts
    # the clockwise turns of the loop as it stands and ``RING_TURNING + 2`` those of its mirror
    # image, each evaluated on exactly the operands ``orient_ccw`` would leave behind (a mirror is
    # ``(x, -y)``, exact in float32), so the caller's pick -- the first when the total is ``>= 0``,
    # the test that decides whether ``orient_ccw`` runs -- is the count of the oriented loop, bit
    # for bit. ``-orient2d`` of the unmirrored loop is *not* a substitute: with FMA contraction on
    # CUDA the two roundings differ, and near-collinear vertices of a fine convex ring then read as
    # reflex. The counts are ``float32`` so every slot shares one buffer and one readback, which is
    # exact for the only question asked of them: a sum of non-negative whole numbers is zero only
    # when every term is.
    chunk, lane = wp.tid()
    n = points2d.shape[0]
    closing = wp.int32(0)
    if detect_closing != 0:
        zero = wp.float32(0.0)
        closing = ring_closing_flag(lift_vec2(points2d[0], zero), lift_vec2(points2d[n - 1], zero))
        if chunk == 0 and lane == 0:
            out_sums[RING_CLOSING] = wp.float32(closing)
    else:
        closing = wp.int32(out_sums[RING_CLOSING])
    n_ring = n - closing
    offset, count = block_chunk_1d(n_ring, chunk)
    if count <= 0:
        return
    local = wp.float32(0.0)
    reflex = wp.float32(0.0)
    reflex_mirrored = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        current = points2d[i]
        i_next = loop_point(i + 1, n_ring)
        nxt = points2d[i_next]
        after = points2d[loop_point(i_next + 1, n_ring)]
        d1 = nxt - current
        d2 = after - nxt
        local += wp.atan2(cross2(d1, d2), wp.dot(d1, d2))
        # The turn at vertex ``i + 1``, whose ring neighbours are ``current`` and ``after``.
        if orient2d(current, nxt, after) < 0:
            reflex += wp.float32(1.0)
        if orient2d(mirror_y(current), mirror_y(nxt), mirror_y(after)) < 0:
            reflex_mirrored += wp.float32(1.0)
    commit_block_sum(lane, wp.vec3(local, reflex, reflex_mirrored), out_sums, RING_TURNING)


@wp.kernel
def fan_triangulate(out_faces: wp.array2d[wp.int32]) -> None:
    # Convex fast-path: fan from vertex 0. dim == n - 2.
    k = wp.int32(wp.tid())
    out_faces[k, 0] = wp.int32(0)
    out_faces[k, 1] = k + 1
    out_faces[k, 2] = k + 2


# The ear loop's state buffer: ``array.LOOP_ROUND`` / ``LOOP_CONDITION``, then the running face
# count ``clip_selected`` appends through. One buffer, seeded by ``init_ring``, so the loop needs no
# host upload and no zero-fill of its own and the final count is a read of one slot.
EAR_COUNT = wp.constant(wp.int32(2))
EAR_STATE_SIZE = 3


@wp.func
def init_ring_slot(
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    i: wp.int32,
    n: wp.int32,
) -> None:
    # Corner ``i`` of an ``n``-corner ring before any clip: linked to both neighbours and active.
    # Shared by ``init_ring`` (one thread per corner) and ``ear_clip_block`` (one block).
    left[i] = wp.where(i == 0, n - 1, i - 1)
    right[i] = loop_point(i + 1, n)
    active[i] = wp.int32(1)


@wp.kernel
def init_ring(
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    if i == 0:
        out_state[LOOP_ROUND] = wp.int32(0)
        out_state[LOOP_CONDITION] = wp.int32(1)
        out_state[EAR_COUNT] = wp.int32(0)
    init_ring_slot(left, right, active, i, left.shape[0])


@wp.func
def ear_flag(
    points2d: wp.array[wp.vec2],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    i: wp.int32,
    n: wp.int32,
    mirror: wp.int32,
) -> wp.int32:
    # ``1`` when corner ``i`` is an active ear. One round's first step, shared by ``compute_ears``
    # (one thread per corner) and ``ear_clip_block`` (one block walks every corner).
    # Nested rather than ``and``-joined: an inactive corner's ``left`` / ``right`` are stale, so
    # its ring walk must never run.
    flag = wp.int32(0)
    if active[i] != 0:
        if is_ear_at(points2d, left, right, active, i, n, mirror):
            flag = wp.int32(1)
    return flag


@wp.kernel
def compute_ears(
    points2d: wp.array[wp.vec2],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    mirror: wp.int32,
    out_is_ear: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    out_is_ear[i] = ear_flag(points2d, left, right, active, i, points2d.shape[0], mirror)


@wp.func
def ear_priority(i: wp.int32) -> wp.uint32:
    # An effectively random but perfectly deterministic order on the ring indices -- see
    # ``array.lowbias32`` for why a hash and not the index itself, and for the bijectivity
    # ``ear_outranks`` leans on.
    return lowbias32(wp.uint32(i))


@wp.func
def ear_outranks(a: wp.int32, b: wp.int32) -> wp.bool:
    # Strict total order on ring indices. The hash is injective, so the index tiebreak below never
    # fires; it is there so the order stays total if the mixer is ever changed.
    key_a = ear_priority(a)
    key_b = ear_priority(b)
    if key_a != key_b:
        return key_a < key_b
    return a < b


@wp.func
def ear_selected(
    is_ear: wp.array[wp.int32], left: wp.array[wp.int32], right: wp.array[wp.int32], i: wp.int32
) -> wp.int32:
    # Select ear i iff it outranks every ear within ring-distance 2. This keeps chosen ears >= 3
    # apart, so their clip footprints {L[i], i, R[i]} are disjoint and can be clipped concurrently.
    # The globally top-ranked ear is always selected, guaranteeing progress.
    #
    # Rank is a *hash* of the ring index rather than the index itself, which is what makes the
    # round count logarithmic. Under the raw index, a ring whose ears alternate (any star polygon)
    # lets the ear at i - 2 suppress the ear at i for every i, so exactly one ear is clipped per
    # round and the clipper runs its full ``n``-round cap. Comparing by an effectively random key
    # instead makes this the textbook maximal-independent-set rule, which retires a constant
    # fraction of the ears per round. Shared by ``select_independent`` and ``ear_clip_block``.
    if is_ear[i] == 0:
        return wp.int32(0)
    ll = left[left[i]]
    left_i = left[i]
    r = right[i]
    rr = right[right[i]]
    if is_ear[ll] == 1 and ear_outranks(ll, i):
        return wp.int32(0)
    if is_ear[left_i] == 1 and ear_outranks(left_i, i):
        return wp.int32(0)
    if is_ear[r] == 1 and ear_outranks(r, i):
        return wp.int32(0)
    if is_ear[rr] == 1 and ear_outranks(rr, i):
        return wp.int32(0)
    return wp.int32(1)


@wp.kernel
def select_independent(
    is_ear: wp.array[wp.int32],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    out_selected: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    out_selected[i] = ear_selected(is_ear, left, right, i)


@wp.func
def clip_ear(
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    i: wp.int32,
    count: wp.array[wp.int32],
    count_slot: wp.int32,
    faces: wp.array2d[wp.int32],
) -> None:
    # Clip the selected ear at corner ``i``: append its triangle at the slot ``count[count_slot]``
    # hands out, retire the corner and link its neighbours past it. Selected ears are >= 3 apart
    # (``ear_selected``), so concurrent clips touch disjoint slots. Shared by ``clip_selected`` and
    # ``ear_clip_block``.
    a = left[i]
    b = right[i]
    slot = wp.atomic_add(count, count_slot, 1)
    faces[slot, 0] = a
    faces[slot, 1] = i
    faces[slot, 2] = b
    active[i] = wp.int32(0)
    right[a] = b
    left[b] = a


@wp.kernel
def clip_selected(
    selected: wp.array[wp.int32],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    out_faces: wp.array2d[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    if selected[i] != 0:
        clip_ear(left, right, active, i, out_state, EAR_COUNT, out_faces)


@wp.kernel
def ear_loop_continue(target: wp.int32, max_rounds: wp.int32, state: wp.array[wp.int32]) -> None:
    # Ear-clipping loop control, kept on device so ``wp.capture_while`` can drive the rounds without
    # a readback each time; the slot table is ``array.LOOP_ROUND`` / ``LOOP_CONDITION``.
    # The round cap is what stops a degenerate or self-intersecting loop that never retires an ear
    # -- the same bound the host-driven form got from iterating ``range(n)``.
    #
    # ``state`` is read-and-incremented round-index/loop-condition scratch carried across launches,
    # not a fresh per-call answer -- see ``rdp_begin_round``'s identical naming.
    state[LOOP_ROUND] = state[LOOP_ROUND] + 1
    if state[EAR_COUNT] < target and state[LOOP_ROUND] < max_rounds:
        state[LOOP_CONDITION] = wp.int32(1)
    else:
        state[LOOP_CONDITION] = wp.int32(0)


# Rings up to this length are clipped by ``ear_clip_block`` -- the whole round loop as one block --
# rather than by the captured four-launch round loop, on CUDA (the CPU device takes the block form
# at every size: its launch grid is a serial loop anyway, and the block form is the same walk
# without a launch and a readback per round, 1.15-1.97x from 512 to 8 192 corners, faces
# byte-identical). At this size a round is too little work to fill the device, so what the
# multi-launch form pays is recording and replaying its conditional graph, flat in ``n``; one block
# of ``EAR_BLOCK_DIM`` lanes, one corner per lane at the cap, trades that for three block barriers
# a round -- 3.0x at 64 corners, 1.46-1.73x at 512-1 024. Past the cap round 0's O(ring) ear test
# runs several corners per lane serially and the device-wide form wins again: 1.06x at 1 500,
# 0.88x at 2 048, 0.41x at 4 096.
EAR_ONE_BLOCK_MAX = 1024
EAR_BLOCK_DIM = 1024


@wp.kernel(enable_backward=False)
def ear_clip_block(
    points2d: wp.array[wp.vec2],
    mirror: wp.int32,
    out_faces: wp.array2d[wp.int32],
    out_ring: wp.array2d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # ``init_ring`` plus every round of ``compute_ears`` -> ``select_independent`` ->
    # ``clip_selected`` -> ``ear_loop_continue``, as one block whose lanes stride the ring by
    # ``wp.block_dim()`` -- so on the CPU device, where a block is one lane, it is the same serial
    # walk the four launches make, corner by corner in index order, and the faces come out
    # byte-identical. The block barriers are ``block_sum`` calls (Warp exposes no other): one after
    # each step, the last of which also totals the round's clips, so the loop condition is
    # block-uniform. The ear rule, the selection rule, the ring's initial links and the clip are
    # the ``ear_flag`` / ``ear_selected`` / ``init_ring_slot`` / ``clip_ear`` the four launches
    # call too.
    #
    # ``out_ring`` is ``(5, n_ring)`` scratch: rows ``left``, ``right``, ``active``, ``is_ear``,
    # ``selected``.
    # ``out_count`` is zeroed here and ends holding the face count.
    _block, lane = wp.tid()
    left = out_ring[0]
    right = out_ring[1]
    active = out_ring[2]
    is_ear = out_ring[3]
    selected = out_ring[4]
    # The ring length, which ``points2d`` may exceed by a repeated closing point.
    n = out_ring.shape[1]
    if lane == 0:
        out_count[0] = wp.int32(0)
    for i in range(lane, n, wp.block_dim()):
        init_ring_slot(left, right, active, i, n)
    count = block_sum(wp.int32(0))
    rounds = wp.int32(0)
    while count < n - 2 and rounds < n:
        for i in range(lane, n, wp.block_dim()):
            is_ear[i] = ear_flag(points2d, left, right, active, i, n, mirror)
        block_barrier()
        for i in range(lane, n, wp.block_dim()):
            selected[i] = ear_selected(is_ear, left, right, i)
        block_barrier()
        clipped = wp.int32(0)
        for i in range(lane, n, wp.block_dim()):
            if selected[i] != 0:
                clip_ear(left, right, active, i, out_count, 0, out_faces)
                clipped += 1
        count += block_sum(clipped)
        rounds += 1


@wp.kernel
def polyline_total_length(
    points: wp.array[wp.vec3], n_segments: wp.int32, out_total: wp.array[wp.float32]
) -> None:
    # Arc length of an open polyline in one launch: the per-segment lengths are summed where they
    # are computed instead of being written to a buffer a separate reduction then reads back in.
    # That is CLAUDE.md section 14.10's producer-consumer fusion applied to a reduction's producer
    # -- ``polyline_length`` ran a ``wp.map`` into an ``(n - 1,)`` scratch array and then
    # ``reduce.sum`` over it: two launches, two allocations and the map's own host-side resolution
    # for an answer that is one number.
    #
    # **This changes the summation order and therefore the last bits of the answer.** A lane-strided
    # fold plus a ``wp.tile_sum`` tree is not the left-to-right sum ``reduce.sum`` happens to
    # perform below ``TILE_1D`` elements, so the result no longer matches a sequential ``float32``
    # accumulation exactly. That is a deliberate trade, and it is why
    # ``tests/test_polyline.py::test_polyline_length_matches_meshlib`` compares with a tolerance
    # rather than with ``==``; a tree sum is in fact the *more* accurate of the two, since it
    # halves the depth over which rounding accumulates.
    #
    # Launched ``wp.launch_tiled(dim=kernel_reduce.blocks_1d(n_segments), block_dim=TILE_1D)``:
    # one block per ``ITEMS_PER_BLOCK_1D`` segments, lanes striding that block's own chunk by
    # ``wp.block_dim()`` (section 2.2, which is what keeps the single-lane CPU device correct), a
    # ``wp.tile_sum`` fold and one atomic per block.
    #
    # ``n_segments`` rather than ``points.shape[0] - 1``, because the closing segment of a *closed*
    # polyline is expressed by wrapping the index here rather than by handing this kernel a copy of
    # the buffer with its first point appended. That copy was an allocation and a full-buffer
    # ``wp.copy`` for one extra segment, and it dominated the closed form once the reduction itself
    # was fused.
    i, lane = wp.tid()
    offset, remaining = block_chunk_1d(n_segments, i)
    if remaining <= 0:
        return
    n_points = points.shape[0]
    total = wp.float32(0.0)
    for k in range(lane, remaining, wp.block_dim()):
        start = offset + k
        total += segment_length(points[start], points[loop_point(start + 1, n_points)])
    block_total = block_sum(total)
    if lane == 0:
        wp.atomic_add(out_total, 0, block_total)


@wp.kernel
def packed_closed_loop_lengths(
    vertices: wp.array[wp.vec3],
    loops: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_lengths: wp.array[wp.float32],
) -> None:
    # ``polyline_total_length``'s closed form for many vertex-index loops at once, one block per
    # loop: the same lane stride over the same segments in the same order and the same
    # ``wp.tile_sum`` fold, so a loop of at most ``ITEMS_PER_BLOCK_1D`` segments -- one block there
    # too, committed onto a zero -- gets the bit-identical length on either device. The two
    # differ only in where a segment's points come from: gathered through the loop's indices
    # here, a dense buffer there. Launched ``wp.launch_tiled(dim=n_loops, block_dim=TILE_1D)``.
    loop, lane = wp.tid()
    base = offsets[loop]
    n = offsets[loop + 1] - base
    total = wp.float32(0.0)
    for k in range(lane, n, wp.block_dim()):
        total += segment_length(
            vertices[loops[base + k]], vertices[loops[base + loop_point(k + 1, n)]]
        )
    length = block_sum(total)
    if lane == 0:
        out_lengths[loop] = length


@wp.kernel
def endpoints_coincide(polyline: wp.array[wp.vec3], out_flag: wp.array[wp.int32]) -> None:
    # Whether a polyline's first and last points coincide, as one thread and one flag.
    #
    # ``polyline.is_closed`` asked this through ``array.allclose`` over two one-element slices,
    # which is a ``wp.map`` into a mask plus a whole reduction over it plus the readback -- more
    # expensive than cloning the entire buffer it is asked about, to compare six floats. The
    # readback stays (the answer decides a host-side branch, and the shape of what
    # ``polyline_close`` returns); everything around it does not.
    #
    # The same ``rtol``/``atol`` predicate ``array.allclose`` applies -- ``ring_closing_flag``, the
    # one every kernel deciding the closure on the device evaluates -- so none of them can disagree
    # about what "coincide" means.
    out_flag[0] = ring_closing_flag(polyline[0], polyline[polyline.shape[0] - 1])
