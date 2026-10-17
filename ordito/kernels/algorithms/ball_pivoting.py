"""
Kernels for wave-parallel ball-pivoting surface reconstruction.

Ports the geometry of Open3D's ``SurfaceReconstructionBallPivoting.cpp`` (``ComputeBallCenter``,
``IsCompatible``, ``FindCandidateVertex``, ``TryTriangleSeed``) to Warp. Instead of the serial
advancing front, every wave pivots a conflict-free independent set of front edges at once and
commits them through a two-phase vertex claim (``wp.atomic_min`` priority on all three vertices),
so at least the globally-lowest-priority triangle always commits and the loop cannot livelock.

The priority is ``proposal_key``, a packing of the proposal's own source edge with a salt drawn from
the wave counter, which is what makes a run **reproducible**: *which* triangles get committed no
longer depends on the order threads reach an atomic, only the order they are written down in does.
The salt is what keeps that from also being *slow* -- a key fixed for the whole run starves the same
front edges wave after wave and roughly doubles the wave count. See that function for why the key is
unique within a wave, and ``commit_triangles`` for the one other place arrival order used to leak in
(the triangle-budget check).

One thing this deliberately does *not* pin, so compare a run as a **set** of wound triangles
rather than buffer to buffer: ``CNT_FACE`` is still a ``wp.atomic_add``, so the face buffer's row
order remains arrival-ordered; determinizing it would mean sorting the whole buffer for a property
no caller has asked for.

Winding
-------
Every committed triangle is wound so its normal agrees with the point normals, and this is
load-bearing for the geometry, not only for the output. A seed is wound by its vertices' normals
(``seed_triangles``); a pivot across front edge ``(src, tgt)`` -- stored in the direction its one
face traverses it -- proposes ``(tgt, src, c)``, the consistent winding. ``pivot_front_edges`` reads
the ball's rotation sense off that stored direction: about ``tgt - src`` of a correctly wound face a
positive rotation carries the ball *away* from the face. On a wrongly wound face the same rule rolls
the ball back over the face it came from, and the crease guard (which compares ``|cos|``) passes a
triangle folded flat onto its parent -- the thin fins an inconsistently wound front grows.

Persistent state
----------------
Everything the algorithm needs lives in device buffers that survive the whole run, so a wave costs
launches only -- no allocations, no host readbacks, and nothing is re-derived from scratch:

* **Edge table.** An open-addressed hash keyed by the undirected edge (``edge_key``), holding the
  incident-face count, the single incident face's directed orientation (``edge_src`` / ``edge_tgt``
  / ``edge_opp``) while the edge is on the boundary, a retirement flag and a cached best candidate.
  ``commit_triangles`` is its only mutator. This replaces re-deriving the front from the whole
  triangle soup each wave with ``edges_unique`` + a sorted interior-key array, and turns the
  manifold guard from a binary search into one hash probe.
* **Front list.** Edge-table slots of the boundary edges, compacted by ``pivot_front_edges`` into a
  second buffer each wave (the caller ping-pongs them) and extended by ``commit_triangles`` with
  the boundary edges the new triangles create.
* **Per-point state.** ``point_used`` and ``boundary_degree`` are maintained incrementally by
  ``commit_triangles``; a point is available as a pivot candidate iff it is unused or still has an
  incident boundary edge.
* **Counters.** One ``int32`` array (see the ``CNT_*`` slots) carries the face count, the front and
  proposal counts, the seeding/continue/done flags and the wave number, so the wave loop's control
  flow is device-resident and can run inside a captured graph.

Border edges
------------
``pivot_front_edges`` retires a front edge whose search found no candidate (Open3D's
``BallPivotingEdgeType::Border``) and never looks at it again. Without that, the overwhelming
majority of pivots in a run are re-tests of edges already known to be dead.

Retiring them is **output-preserving, not an approximation**, because every rejection in the
candidate loop is monotone in the wave index:

* the geometric tests (``compute_ball_center``, ``is_compatible``, ``ball_is_empty``, the crease
  and clustering guards) depend only on the fixed points, normals and radius;
* availability is sticky-false -- a vertex that is used and has no incident boundary edge can never
  gain another triangle, since seeding needs an unused point and pivoting needs an available
  candidate, so it can never become available again;
* the interior-edge guard only ever tightens: faces are appended and never removed, so an edge's
  face count only grows.

So the set of valid candidates for a given front edge shrinks monotonically. Two consequences are
used here: an edge with no candidate can be retired, and -- the other half of the same theorem -- a
**cached** best candidate is still the argmin as long as it is still valid, so an edge that found
one but lost the vertex claim (most of them, every wave) is re-checked in O(1) instead of
re-searched over its whole neighbourhood. An edge is retired only when the *search* failed, never
when it merely lost the claim.
"""

from typing import Any

import warp as wp

from ordito.constants import TWO_PI, UINT64_MAX_CONSTANT
from ordito.kernels.array import pack_edge_key
from ordito.kernels.grouping import hash_find, hash_find_or_insert
from ordito.kernels.predicates import dihedral_angle, float_inf, triangle_normal
from ordito.kernels.proximity import flag_overfull_leaf, meets_overfull_leaf, straggler_slab
from ordito.kernels.reduce import block_argmin

wp.set_module_options({"enable_backward": False})

# Per-thread neighbour scratch for the seed search (Open3D re-scans the KNN result twice).
MAX_SEED_NEIGHBORS = 64
# ``seed_triangles``' per-point cache: a neighbour index is the orphan the point last deferred to,
# ``SEED_UNKNOWN`` nothing yet, and ``SEED_EXHAUSTED`` a point that can never seed (see there).
SEED_UNKNOWN = wp.constant(wp.int32(-1))
SEED_EXHAUSTED = wp.constant(wp.int32(-2))
# Half-width of the band around the ball's sphere, in float32 rounding units of the ball centre's
# largest coordinate (``sphere_band_width``), inside which float32 cannot be trusted to say which
# side of the sphere a point is on, so ``band_point_inside`` decides it in float64. A co-circular
# grid corner's float32 distance from the computed centre is off by at most 1.25 such units
# (measured on translated and rotated planar grids, on both devices); 16 is a wide margin, and a
# wider band only refines more points, never changes an answer.
BALL_BAND_ULPS = wp.float32(16.0)
FLOAT32_EPS = wp.float32(1.1920929e-07)  # numpy.finfo(numpy.float32).eps
# Half-width of the float64 tie, in float64 rounding units of the squared scale of the centre
# (``band_point_inside``): a point whose float64 power against the ball is smaller than this is on
# the sphere, and the symbolic index rule decides it. Only an exactly co-spherical input (a grid
# whose coordinates are exact in binary) lands there; float32 rounding of the input coordinates
# alone already moves a nominal tie a million times further.
BALL_TIE_ULPS64 = wp.float64(64.0)
FLOAT64_EPS = wp.float64(2.220446049250313e-16)  # numpy.finfo(numpy.float64).eps
# The largest float32 below ``TWO_PI``: the ceiling ``pivot_candidate`` clamps a rotation to, so
# that no candidate ties the ``TWO_PI`` "nothing found" sentinel it is compared against.
PIVOT_ANGLE_MAX = wp.float32(6.283185)

# ``counters`` slots. Keeping the wave loop's whole control state on device is what lets the loop
# body run without a host synchronisation, and ultimately inside a captured CUDA graph.
CNT_FACE = wp.constant(0)  # committed triangles so far
CNT_PREV_FACE = wp.constant(1)  # ... as of the start of this wave (the progress test)
CNT_FRONT = wp.constant(2)  # entries in the incoming front list
CNT_NEXT_FRONT = wp.constant(3)  # entries written to the outgoing front list
CNT_LIVE = wp.constant(4)  # of those, how many were live (drives host-side compaction)
CNT_PROPOSAL = wp.constant(5)  # triangles proposed this wave
CNT_SEEDING = wp.constant(6)  # this wave seeds orphans instead of pivoting
CNT_CONTINUE = wp.constant(7)  # the wave loop's condition
CNT_DONE = wp.constant(8)  # the loop finished for good (as opposed to pausing to grow)
CNT_WAVE = wp.constant(9)  # waves executed
CNT_GROW = wp.constant(10)  # the triangle budget is exhausted; hand back to the host
CNT_SEED_FAILED = wp.constant(11)  # orphans marked "cannot seed" so far (see seed_triangles)
CNT_PREV_SEED_FAILED = wp.constant(12)  # ... as of the start of this wave (the other progress test)
BPA_COUNTERS = 13

# Lanes per front edge in the cooperative pivot search. One warp: the block-cooperative BVH walk
# hands one candidate per lane per step, and 32 covers the ~88 candidates an edge enumerates in
# three steps while keeping the tile reductions on Warp's single-warp fast path.
BPA_PIVOT_BLOCK = 32

EDGE_LIVE = wp.constant(0)
EDGE_RETIRED = wp.constant(1)


@wp.struct
class BpaEdgeTable:
    """
    The open-addressed edge table, as one kernel argument instead of nine.

    Every wave launches ``pivot_front_edges`` (27 arguments before this, 19 after) and
    ``commit_triangles`` (20, now 13), and both carried the whole table; seven functions here
    thread it. A ``wp.launch`` argument costs roughly a microsecond of host time, and a run issues
    over a thousand launches, so the bundle is worth a measurable fraction of the wave loop's
    marshalling -- though only a small fraction of the whole call, which is dominated by the pivot
    search.

    Bound once by ``_BpaState`` and rebound only when ``grow`` reallocates (``_bind_edge_table``
    is called from ``_allocate_budget``, which is the only place the arrays are replaced).
    ``rehash_edges`` still takes the old and new arrays loose, because it is the one kernel that
    sees two tables at once.
    """

    key: wp.array[wp.uint64]
    count: wp.array[wp.int32]
    src: wp.array[wp.int32]
    tgt: wp.array[wp.int32]
    opp: wp.array[wp.int32]
    state: wp.array[wp.int32]
    cand: wp.array[wp.int32]
    mask: wp.int32
    key_base: wp.uint64


@wp.func
def compute_ball_center(v1: Any, v2: Any, v3: Any, normal_sum: Any, radius: wp.Float):
    # Center of the radius-``radius`` ball touching all three points, on the ``normal_sum`` side.
    # Returns a sentinel of (inf, inf, inf) when the ball does not exist (points too far apart or
    # nearly collinear), which the caller treats as failure. Generic over the precision: the
    # pivot works in float32, and ``refined_ball_center`` re-forms a centre in float64 to decide
    # a point lying within float32 round-off of the sphere.
    inf = float_inf(radius)
    fail = type(v1)(inf, inf, inf)
    c = wp.length_sq(v2 - v1)
    b = wp.length_sq(v1 - v3)
    a = wp.length_sq(v3 - v2)
    alpha = a * (b + c - a)
    beta = b * (a + c - b)
    gamma = c * (a + b - c)
    abg = alpha + beta + gamma
    if abg < type(radius)(1e-16):
        return fail
    circ_center = (alpha * v1 + beta * v2 + gamma * v3) / abg
    circ_radius2 = a * b * c
    sa = wp.sqrt(a)
    sb = wp.sqrt(b)
    sc = wp.sqrt(c)
    denom = (sa + sb + sc) * (sb + sc - sa) * (sc + sa - sb) * (sa + sb - sc)
    if denom <= type(radius)(0.0):
        return fail
    circ_radius2 = circ_radius2 / denom
    height = radius * radius - circ_radius2
    if height < type(radius)(0.0):
        return fail
    tr_norm = triangle_normal(v1, v2, v3)
    pt_norm = wp.normalize(normal_sum)
    if wp.dot(tr_norm, pt_norm) < type(radius)(0.0):
        tr_norm = -tr_norm
    return circ_center + wp.sqrt(height) * tr_norm


@wp.func
def is_compatible(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, na: wp.vec3, nb: wp.vec3, nc: wp.vec3
) -> wp.bool:
    # The triangle normal must agree (within tolerance) with all three oriented point normals.
    normal = triangle_normal(a, b, c)
    if wp.dot(normal, na) < -1e-16:
        normal = -normal
    return (
        wp.dot(normal, na) > -1e-16 and wp.dot(normal, nb) > -1e-16 and wp.dot(normal, nc) > -1e-16
    )


@wp.func
def sphere_band_width(center: wp.vec3, radius: wp.float32) -> wp.float32:
    # Half-width of the band around the sphere of the ball centred at ``center`` inside which
    # float32 cannot place a point: ``BALL_BAND_ULPS`` rounding units of the centre's largest
    # coordinate (plus the radius, for a centre near the origin), since that is what the centre is
    # computed to.
    scale = wp.max(wp.max(wp.abs(center[0]), wp.abs(center[1])), wp.abs(center[2])) + radius
    return BALL_BAND_ULPS * FLOAT32_EPS * scale


@wp.func
def refined_ball_center(
    points: wp.array[wp.vec3],
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    center: wp.vec3,
    radius: wp.float32,
) -> wp.vec3d:
    # The float32 ball centre ``center`` of ``a``, ``b`` and ``c``, re-formed in float64 from the
    # same (exact) float32 coordinates. The side of the triangle is the float32 centre's, read off
    # the direction from the triangle's centroid to it; the two balls through three points are
    # mirror images, so that direction cannot be mistaken unless the centre lies in the plane,
    # where the two coincide. Infinite when the float64 ball does not exist (a triangle whose
    # circumradius is ``radius`` to within float32 round-off).
    pa = wp.vec3d(points[a])
    pb = wp.vec3d(points[b])
    pc = wp.vec3d(points[c])
    centroid = (pa + pb + pc) / wp.float64(3.0)
    return compute_ball_center(pa, pb, pc, wp.vec3d(center) - centroid, wp.float64(radius))


@wp.func
def band_point_inside(
    points: wp.array[wp.vec3],
    center64: wp.vec3d,
    radius: wp.float32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    j: wp.int32,
) -> wp.bool:
    # Whether point ``j``, within float32 round-off of the sphere of the ball through ``a``, ``b``
    # and ``c`` (``sphere_band_width``), lies inside it: decided in float64 against the refined
    # centre ``center64`` (``refined_ball_center``), and, for a point on the sphere even there, by a
    # symbolic index rule.
    #
    # Deciding the band by "outside" instead -- or by any tolerance wider than round-off -- made
    # every triangle of four (near-)co-spherical points pass, so a grid cell, or any quadrilateral
    # sampled co-circular to within the tolerance, got both triangulations, wound against each
    # other. A geometric answer is consistent across the triangles of one such set, so exactly one
    # triangulation passes; the float64 answer is that answer for every input float32 can hold
    # except an exact tie, which the index rule settles.
    #
    # The index rule is a symbolic perturbation (simulation of simplicity): the lowest-indexed of
    # the four points is pushed radially outward by an infinitesimal amount. If that is ``j``, it
    # leaves the sphere and ``j`` is outside. If it is a vertex ``m`` of the triangle, the ball
    # follows it: the centre moves along the circle of centres equidistant from the other two
    # vertices ``u`` and ``v``, towards ``m``'s side of the plane through the centre, ``u`` and
    # ``v``, so ``j`` falls inside exactly when it is on ``m``'s side of that plane. Every triangle
    # of a co-spherical set is judged by the same perturbation, so exactly one triangulation of the
    # set has every ball empty -- for a planar convex cell, the one whose diagonal avoids the
    # lowest-indexed corner -- whatever order the tests run in. Pure geometry and indices, like the
    # rest of the empty-ball test, so the monotonicity that retires a front edge and caches its
    # candidate (module docstring) is untouched.
    pj = wp.vec3d(points[j])
    if center64[0] == wp.float64(wp.inf):
        # No float64 ball: the triangle's circumradius is the radius to within float32 round-off.
        # The point counts as outside, as it would on the sphere of a ball that barely exists.
        return False
    radius64 = wp.float64(radius)
    power = wp.length_sq(pj - center64) - radius64 * radius64
    scale = wp.max(wp.max(wp.abs(center64[0]), wp.abs(center64[1])), wp.abs(center64[2]))
    scale = scale + radius64
    tolerance = BALL_TIE_ULPS64 * FLOAT64_EPS * scale * scale
    if power < -tolerance:
        return True
    if power > tolerance:
        return False
    m = wp.min(wp.min(a, b), c)
    if j < m:
        return False
    u = b
    v = c
    if m == b:
        u = c
        v = a
    elif m == c:
        u = a
        v = b
    plane = wp.cross(wp.vec3d(points[u]) - center64, wp.vec3d(points[v]) - center64)
    side_j = wp.dot(plane, pj - center64)
    side_m = wp.dot(plane, wp.vec3d(points[m]) - center64)
    return side_j * side_m > wp.float64(0.0)


@wp.func
def ball_holds_point(
    grid_id: wp.uint64,
    points: wp.array[wp.vec3],
    center: wp.vec3,
    radius: wp.float32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
) -> wp.bool:
    # Whether a point other than ``a``, ``b`` and ``c`` lies inside the ball beyond float32
    # round-off: ``ball_is_empty`` less the band, for ``seed_triangles``, which decides the band
    # (``ball_band_blocks``) outside its pair loop. The two together are ``ball_is_empty``'s
    # predicate exactly.
    inner = radius - sphere_band_width(center, radius)
    inner_sq = inner * inner
    query = wp.hash_grid_query(grid_id, center, radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if j != a and j != b and j != c:
            if wp.length_sq(center - points[j]) < inner_sq:
                return True
    return False


@wp.func
def ball_band_blocks(
    grid_id: wp.uint64,
    points: wp.array[wp.vec3],
    center: wp.vec3,
    radius: wp.float32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
) -> wp.bool:
    # Whether a point in the band around the ball's sphere lies inside it (``band_point_inside``):
    # the other half of ``ball_is_empty``'s predicate, as a walk of its own. The float64 centre is
    # formed only once a band point turns up.
    band = sphere_band_width(center, radius)
    inner = radius - band
    inner_sq = inner * inner
    outer = radius + band
    outer_sq = outer * outer
    refined = wp.bool(False)
    center64 = wp.vec3d(0.0)
    query = wp.hash_grid_query(grid_id, center, outer)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if j != a and j != b and j != c:
            distance_sq = wp.length_sq(center - points[j])
            if distance_sq >= inner_sq:
                if distance_sq < outer_sq:
                    if not refined:
                        center64 = refined_ball_center(points, a, b, c, center, radius)
                        refined = wp.bool(True)
                    if band_point_inside(points, center64, radius, a, b, c, j):
                        return True
    return False


@wp.func
def ball_is_empty(
    grid_id: wp.uint64,
    points: wp.array[wp.vec3],
    center: wp.vec3,
    radius: wp.float32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
) -> wp.bool:
    # True when no point other than the three defining ones lies strictly inside the ball.
    #
    # Compared squared, matching ``candidate_is_viable``'s clustering test. The two spellings are
    # not the same predicate in float32, so a module that used both could accept a point as a
    # candidate and reject the same distance as an occupancy -- and the ball radius is one of the
    # two distances BPA is entirely built out of. The argument is consistency, not speed: the swap
    # measures flat.
    #
    # **``wp.bvh_query_sphere`` (Warp 1.17) was built here and reverted: it is a large loss.** It
    # looks like the ideal fit -- the point BVH's bounds are degenerate, so an exact sphere-AABB
    # test at the shrunk radius *is* this predicate, with no narrow phase and no second radius,
    # where the grid must query at ``radius`` and filter at the band's inner edge because a cell
    # walk cannot express either radius exactly. It was also *correct*, returning an equal face
    # buffer. It is simply slower, because this is a *small-radius, well-centred* query, which is
    # the hash grid's best case and not the BVH's: the grid was built with cell width ``radius``, so
    # a probe reaches its cells by address arithmetic and stops, while the BVH pays a root-to-leaf
    # descent per call and there are millions of calls.
    #
    # **Do not read the win ``pivot_front_edges`` gets from ``wp.tile_bvh_query_aabb`` as
    # transferring to here**: that one is a *block-cooperative* walk over a ``2 * radius`` reach
    # with enough candidates to split across 32 lanes, and every part of that description is
    # load-bearing.
    #
    # A point nearer the sphere than float32 can resolve (``sphere_band_width``) is decided by
    # ``band_point_inside``, in float64 and, at an exact tie, by index; every other point by its
    # float32 distance, with no tolerance beyond round-off (Open3D's test is ``radius - 1e-16`` in
    # double). The walk reaches the band's outer edge, so a point just past ``radius`` is not
    # missed by the cell walk.
    #
    # The float64 decision runs in a second walk (``ball_band_blocks``), and only for a ball this
    # one found a band point in, which is rare off exact ties: inline in this walk it cost
    # ``pivot_front_edges`` about an eighth of its time, and a second walk for every ball that
    # passes (every cached candidate's re-validation does) a fifth. ``seed_triangles`` goes further
    # and tests only the inside in its pair loop (``ball_holds_point``): there even this walk's
    # band flag slowed the kernel by about half.
    band = sphere_band_width(center, radius)
    inner = radius - band
    inner_sq = inner * inner
    outer = radius + band
    outer_sq = outer * outer
    in_band = wp.bool(False)
    query = wp.hash_grid_query(grid_id, center, outer)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if j != a and j != b and j != c:
            distance_sq = wp.length_sq(center - points[j])
            if distance_sq < inner_sq:
                return False
            if distance_sq < outer_sq:
                in_band = wp.bool(True)
    if in_band:
        return not ball_band_blocks(grid_id, points, center, radius, a, b, c)
    return True


@wp.func
def reset_wave_counters(counters: wp.array[wp.int32]) -> None:
    # Reset the per-wave counters and snapshot the face count and the seed-failure count the two
    # progress tests compare against. The vertex claim is *not* cleared here: ``propose_triangle``
    # clears the three vertices it is about to contend for, which is the only part of an n-sized
    # array a wave ever reads. Idempotent: running it twice before a wave changes nothing.
    counters[CNT_NEXT_FRONT] = 0
    counters[CNT_LIVE] = 0
    counters[CNT_PROPOSAL] = 0
    counters[CNT_PREV_FACE] = counters[CNT_FACE]
    counters[CNT_PREV_SEED_FAILED] = counters[CNT_SEED_FAILED]


@wp.kernel(enable_backward=False)
def begin_wave(counters: wp.array[wp.int32]) -> None:
    # The first wave of a batch's reset. Every later wave's is the tail of the ``end_wave`` before
    # it, which resets whenever the loop keeps going; this launch covers the start of a run and a
    # restart after the host grew the budget or compacted the front, where the previous
    # ``end_wave`` stopped the loop and so reset nothing.
    if counters[CNT_CONTINUE] == 0:
        return
    reset_wave_counters(counters)


@wp.func
def push_front_edge(
    slot: wp.int32, capacity: wp.int32, counters: wp.array[wp.int32], out_front: wp.array[wp.int32]
) -> None:
    # Append an edge to the outgoing front list, bounded by the list's allocated capacity.
    #
    # ``front_out`` is sized ``3 * max_faces + n``, exactly the worst case the commit budget
    # admits, so there is no slack: any accounting slip would scatter past the end, and because
    # the buffer comes from the memory pool such a write lands in *another live allocation*
    # and corrupts it silently rather than faulting.
    #
    # Overflow is *recoverable*, which is why this drops rather than clamps: the front list is a
    # cache of the edge hash table, not the source of truth, and ``_BpaState.grow`` rebuilds it from
    # that table with ``collect_front_from_table``. So a dropped entry is restored by the same host
    # round-trip that doubles the budget, and raising ``CNT_GROW`` is what asks for it. In practice
    # the front peaks at a fraction of a percent of capacity and this never fires; it is hardening.
    position = wp.atomic_add(counters, CNT_NEXT_FRONT, 1)
    if position < capacity:
        out_front[position] = slot
        return
    counters[CNT_GROW] = 1


@wp.func
def propose_triangle(
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.uint64],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    # Append a candidate triangle to this wave's proposal list. Both the seed and the pivot path
    # write here, so the claim and commit kernels have a single dense list to walk.
    #
    # Releasing the three vertices for this wave's claim happens here rather than in a sweep over
    # all n: only a proposed vertex is ever read back, and two proposals sharing a vertex write the
    # same sentinel. That removes an n-wide launch from every wave, which on a small cloud was most
    # of the wave. The sentinel is the largest ``uint64`` because the claim is a ``wp.atomic_min``
    # over ``proposal_key``, which is unsigned and bounded by 2^63.
    out_owner[a] = UINT64_MAX_CONSTANT
    out_owner[b] = UINT64_MAX_CONSTANT
    out_owner[c] = UINT64_MAX_CONSTANT
    slot = wp.atomic_add(counters, CNT_PROPOSAL, 1)
    if slot >= capacity:
        # Dropping a proposal is safe and self-healing: the front edge that raised it stays live and
        # is searched again next wave. Only the *write* would be unsafe -- and unlike a front-list
        # overflow this needs no host round-trip, so it raises no flag.
        return
    out_a[slot] = a
    out_b[slot] = b
    out_c[slot] = c


@wp.kernel(enable_backward=False)
def seed_triangles(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    point_used: wp.array[wp.bool],
    seed_failed: wp.array[wp.bool],
    seed_blocker: wp.array[wp.int32],
    grid_id: wp.uint64,
    radius: wp.float32,
    clustering: wp.float32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.uint64],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    if counters[CNT_CONTINUE] == 0 or counters[CNT_SEEDING] == 0:
        return
    p = wp.int32(wp.tid())
    if point_used[p]:
        return
    # The deferral below, remembered: ``seed_blocker[p]`` is the neighbour this point last yielded
    # to, and while it is still an orphan that has not given up, the walk would yield to it again
    # and the answer is unchanged, so two loads replace a hash-grid walk. Exact, not approximate:
    # the walk yields at the *first* qualifying neighbour in the grid's order, the grid is built
    # once per run so that order is fixed, and both flags only ever move a neighbour *out* of
    # qualifying -- so no neighbour ahead of the blocker can start to qualify, and the ones ahead of
    # it that stop being orphans only bring it forward, never past the ``MAX_SEED_NEIGHBORS`` break.
    #
    # ``SEED_EXHAUSTED`` is the other thing worth remembering: a point whose failed attempt *saw its
    # whole neighbourhood*. Whether an attempt succeeds depends on the stored neighbours alone --
    # ``ball_is_empty`` and ``is_compatible`` are geometry, and never read ``point_used`` -- and a
    # later walk stores the same neighbours minus the ones used since, so every later attempt
    # tries a subset of the pairs that already failed and fails again. Such a point returns here
    # for the rest of the run. A walk cut short by the ``MAX_SEED_NEIGHBORS`` break is not
    # remembered: once stored neighbours are used, the next walk reaches past the break and can
    # store a neighbour the failed one never tried.
    blocker = seed_blocker[p]
    if blocker == SEED_EXHAUSTED:
        return
    if blocker >= 0:
        if not point_used[blocker] and not seed_failed[blocker]:
            return

    # Gather the local orphan neighbourhood into scratch so it can be scanned as a double loop.
    # A point yields to any lower-index orphan it *sees* that has not already given up
    # (``seed_failed``, below), so seed fronts start well separated and do not collide into
    # overlapping sheets before they can glue.
    #
    # ``wp.zeros``, not the ``wp.types.vector(length=K)`` register row ``neighbors.py`` uses:
    # measured, the register row gives nothing here. It wins only when every access is an unrolled
    # compile-time slot, and every access here is a runtime index, which spills the vector to local
    # memory — where ``wp.zeros`` already puts it. ``wp.fixedarray`` is not a third option: its own
    # docstring says it is "only used during codegen, and for type hints", and it *is* the codegen
    # type of a kernel-scope ``wp.zeros``.
    #
    # **The ``break`` bounds the walk and not just the candidate list, so the "lowest-index orphan
    # seeds" rule holds only while the neighbourhood fits in ``MAX_SEED_NEIGHBORS``. That is
    # deliberate and tightening it is a regression.** Past that many stored neighbours the scan
    # stops before it can see a lower-indexed orphan, so adjacent points both seed — and real
    # neighbourhoods are past it routinely at the radii a run uses. The exact form (bound the store,
    # let the walk run to completion) was built and measured: it collapses the wave-0 seed count to
    # one or zero, and zero seeds makes ``end_wave`` raise ``CNT_DONE`` on an empty mesh.
    #
    # ``seed_failed`` is what keeps the rule from deadlocking: it elects **one** seeder per
    # neighbourhood, so a neighbourhood whose lowest-indexed orphan cannot form a valid seed
    # triangle would otherwise be dead for good — the next wave sees identical state and fails
    # identically. Marking a failed elected point releases its neighbourhood to the next-lowest
    # deferring orphan (``end_wave`` keeps seeding while new failures are still being discovered).
    # Sound because a point's candidate set only ever *shrinks* across waves (``point_used`` only
    # goes false -> true), so a point that exhausted its candidates cannot succeed with a subset.
    nbr = wp.zeros(shape=MAX_SEED_NEIGHBORS, dtype=wp.int32)
    count = wp.int32(0)
    truncated = wp.bool(False)
    query = wp.hash_grid_query(grid_id, points[p], 2.0 * radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if count >= MAX_SEED_NEIGHBORS:
            truncated = wp.bool(True)
            break
        if j != p and not point_used[j]:
            if j < p and not seed_failed[j]:
                seed_blocker[p] = j
                return  # a lower-index, still-viable orphan neighbour will seed this instead
            nbr[count] = j
            count += 1

    # Squared, matching the same vcglib clustering rule as tested in ``candidate_is_viable``; see
    # ``ball_is_empty`` for why one module must not spell one predicate two ways.
    min_cluster_sq = clustering * radius * clustering * radius
    # The pair loop tests occupancy by distance alone (``ball_holds_point``) and the tie rule
    # (``ball_band_blocks``) runs outside it, once per pair that passes, resuming the loop after a
    # pair a tie rejects: the same predicate as ``ball_is_empty``, but with the tie walk out of the
    # loop nest, which keeps it as fast as the plain test was. ``resume`` is the flat index
    # ``i0 * MAX_SEED_NEIGHBORS + i1`` of the first pair still to try.
    resume = wp.int32(0)
    while resume >= 0:
        found_a = wp.int32(-1)
        found_b = wp.int32(-1)
        found_pair = wp.int32(-1)
        found_center = wp.vec3(0.0)
        for i0 in range(count):
            if found_a >= 0:
                break
            a = nbr[i0]
            for i1 in range(i0 + 1, count):
                if i0 * MAX_SEED_NEIGHBORS + i1 < resume:
                    continue
                b = nbr[i1]
                if wp.length_sq(points[a] - points[b]) < min_cluster_sq:
                    continue
                center = compute_ball_center(
                    points[p], points[a], points[b], normals[p] + normals[a] + normals[b], radius
                )
                if center[0] == wp.inf:
                    continue
                if not is_compatible(
                    points[p], points[a], points[b], normals[p], normals[a], normals[b]
                ):
                    continue
                if not ball_holds_point(grid_id, points, center, radius, p, a, b):
                    found_a = a
                    found_b = b
                    found_pair = i0 * MAX_SEED_NEIGHBORS + i1
                    found_center = center
                    break
        resume = wp.int32(-1)
        if found_a >= 0:
            a = found_a
            b = found_b
            if ball_band_blocks(grid_id, points, found_center, radius, p, a, b):
                resume = found_pair + 1
            else:
                # Wound so its normal agrees with the point normals (Open3D's ``CreateTriangle``
                # rule). Every later triangle inherits this winding across the edge it pivots over
                # (``pivot_front_edges``), whose rotation sense is read off it. Reversed as
                # ``(a, p, b)`` rather than ``(p, b, a)`` so the ``proposal_key`` pair stays
                # ``{p, a}``.
                outward = wp.dot(
                    triangle_normal(points[p], points[a], points[b]),
                    normals[p] + normals[a] + normals[b],
                )
                if outward < 0.0:
                    propose_triangle(
                        a, p, b, front_capacity, counters, out_owner, out_a, out_b, out_c
                    )
                else:
                    propose_triangle(
                        p, a, b, front_capacity, counters, out_owner, out_a, out_b, out_c
                    )
                return

    # No candidate pair among those seen produced a valid seed: release this neighbourhood so a
    # deferring orphan can try on a later wave, and count it so ``end_wave`` knows the seed-failure
    # state is still changing and keeps seeding rather than declaring the run done.
    if not seed_failed[p]:
        seed_failed[p] = True
        wp.atomic_add(counters, CNT_SEED_FAILED, 1)
    if not truncated:
        seed_blocker[p] = SEED_EXHAUSTED


@wp.func
def point_is_available(
    p: wp.int32, point_used: wp.array[wp.bool], boundary_degree: wp.array[wp.int32]
) -> wp.bool:
    # Orphan, or still on the advancing front. A used point with no incident boundary edge is
    # fully interior and can never be pivoted onto again — which is what makes retirement sound.
    return (not point_used[p]) or boundary_degree[p] > 0


@wp.func
def edge_is_interior(u: wp.int32, v: wp.int32, edges: BpaEdgeTable) -> wp.bool:
    # Manifold guard: an edge already shared by two triangles may not gain a third. One hash probe,
    # where the previous design needed a binary search into a freshly sorted key array.
    slot = hash_find(pack_edge_key(u, v, edges.key_base), edges.key, edges.mask)
    return slot >= 0 and edges.count[slot] >= 2


@wp.func
def candidate_prefilter(
    points: wp.array[wp.vec3],
    p_src: wp.vec3,
    p_tgt: wp.vec3,
    src: wp.int32,
    tgt: wp.int32,
    opp: wp.int32,
    c: wp.int32,
    min_cluster_sq: wp.float32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
) -> wp.bool:
    # The cheap half of the candidate test: identity, availability and vcglib clustering. Compared
    # squared, since this runs once per point the grid hands back.
    if c == src or c == tgt or c == opp:
        return False
    if not point_is_available(c, point_used, boundary_degree):
        return False
    p_c = points[c]
    if wp.length_sq(p_c - p_src) < min_cluster_sq:
        return False
    return wp.length_sq(p_c - p_tgt) >= min_cluster_sq


@wp.func
def candidate_accepted(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    p_src: wp.vec3,
    p_tgt: wp.vec3,
    tri_norm: wp.vec3,
    src: wp.int32,
    tgt: wp.int32,
    c: wp.int32,
    center: wp.vec3,
    grid_id: wp.uint64,
    radius: wp.float32,
    crease_cos: wp.float32,
    edges: BpaEdgeTable,
) -> wp.bool:
    # The expensive half: crease, manifoldness, normal compatibility and the empty-ball test, in
    # increasing order of cost. Shared verbatim between the full search and the O(1) re-validation
    # of a cached candidate, so the two can never disagree about what a valid candidate is.
    #
    # ``tri_norm`` (the pivoting triangle's normal) is loop-invariant and passed in.
    if (
        crease_cos > -1.0
        and wp.abs(wp.dot(tri_norm, triangle_normal(p_src, p_tgt, points[c]))) < crease_cos
    ):
        return False
    if edge_is_interior(src, c, edges):
        return False
    if edge_is_interior(tgt, c, edges):
        return False
    if not is_compatible(p_src, p_tgt, points[c], normals[src], normals[tgt], normals[c]):
        return False
    return ball_is_empty(grid_id, points, center, radius, src, tgt, c)


@wp.kernel(enable_backward=False)
def mark_overfull_point_leaves(bvh: wp.uint64, out_flags: wp.array[wp.int32]) -> None:
    # ``kernels.proximity.flag_overfull_leaf`` over the point BVH into the zeroed node bitset
    # ``out_flags``, so ``meets_overfull_leaf`` routes a pivot search away from
    # ``wp.tile_bvh_query_aabb`` only when it may meet such a leaf. Warp's BVH builders make a node
    # at depth 32 a leaf whatever its size, which a cluster of points sharing one Morton key
    # reaches. ``dim`` is at least the node count; nodes past it return.
    flag_overfull_leaf(bvh, wp.int32(wp.tid()), out_flags)


@wp.func
def pivot_candidate(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    p_src: wp.vec3,
    p_tgt: wp.vec3,
    tri_norm: wp.vec3,
    src: wp.int32,
    tgt: wp.int32,
    opp: wp.int32,
    c: wp.int32,
    mp: wp.vec3,
    axis: wp.vec3,
    a_dir: wp.vec3,
    min_cluster_sq: wp.float32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    grid_id: wp.uint64,
    radius: wp.float32,
    crease_cos: wp.float32,
    edges: BpaEdgeTable,
    best_angle: wp.float32,
    best: wp.int32,
) -> tuple[wp.float32, wp.int32]:
    # One candidate ``c`` of a lane's pivot search folded into its running ``(best_angle, best)``.
    # Shared by ``pivot_front_edges``' two walks, which differ only in how they enumerate the box.
    #
    # ``0 <= c < n_points`` is load-bearing, not defensive: a lane with no candidate in a step of
    # the cooperative walk gets -1, and Warp's BVH walks are bound-checked wherever they are used
    # (``.claude/CLAUDE.md`` on ``wp.tile_bvh_query_aabb``). An index outside the range is never a
    # primitive of this BVH, so the test rejects nothing the walk was entitled to return.
    if c < 0 or c >= points.shape[0]:
        return best_angle, best
    if not candidate_prefilter(
        points, p_src, p_tgt, src, tgt, opp, c, min_cluster_sq, point_used, boundary_degree
    ):
        return best_angle, best
    new_center = compute_ball_center(
        p_src, p_tgt, points[c], normals[src] + normals[tgt] + normals[c], radius
    )
    if new_center[0] == wp.inf:
        return best_angle, best
    b_dir = wp.normalize(new_center - mp)
    # Rotation about the edge from the current ball centre to this candidate's, signed by ``axis``
    # and folded into [0, 2pi).
    #
    # ``dihedral_angle`` is atan2(|a x b| along the axis, a . b), not ``acos(a . b)``, and here that
    # is a correctness matter rather than a tidy-up: the candidate that *wins* is the one nearest
    # coplanar with the current triangle, which is exactly where acos has an infinite derivative.
    # Measured against a float64 reference on float32 directions, acos's absolute error *equals*
    # the angle below ~1e-5 rad -- it returns zero, so every candidate closer than that ranked
    # identically and the winner fell out of float32 rounding, where atan2 stays accurate to many
    # more digits. The two forms agree elsewhere, and this one is also cheaper: the cross product is
    # the same one the sign test needed.
    #
    # The clamp keeps a candidate from tying the ``TWO_PI`` sentinel ``best_angle`` starts at. A
    # candidate on the current ball's sphere -- the far corner of a co-circular grid cell, the
    # only one whose ball is empty there -- rotates the ball by exactly zero, which rounding can
    # make a tiny negative angle; folded into [0, 2pi) that rounds onto ``TWO_PI`` itself, the
    # strict test below rejected the one valid candidate, and the edge retired, leaving a hole
    # (seen on CUDA, whose contraction gives the negative sign). Clamped, it ranks last but is kept;
    # every other candidate there has a nonempty ball, since rolling the ball forward at all takes
    # in the corner.
    angle = dihedral_angle(a_dir, b_dir, axis)
    if angle < 0.0:
        angle = wp.min(angle + TWO_PI, PIVOT_ANGLE_MAX)
    if angle < best_angle and candidate_accepted(
        points, normals, p_src, p_tgt, tri_norm, src, tgt, c, new_center,
        grid_id, radius, crease_cos, edges,
    ):  # fmt: skip
        return angle, c
    return best_angle, best


@wp.kernel(enable_backward=False)
def pivot_front_edges(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    grid_id: wp.uint64,
    bvh_id: wp.uint64,
    overfull: wp.array[wp.int32],
    radius: wp.float32,
    clustering: wp.float32,
    crease_cos: wp.float32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    front_in: wp.array[wp.int32],
    grid_stride: wp.int32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    out_owner: wp.array[wp.uint64],
    front_out: wp.array[wp.int32],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    # **One block per front edge, one warp per block.** The pivot search is a neighbourhood walk per
    # edge, and a wave has only a few hundred live edges, so thread-per-edge left the device idle:
    # the wave cost was nearly flat in the front size. A warp an edge is what fills it.
    #
    # The candidate walk therefore runs on the **BVH**, not the hash grid: Warp's hash grid exposes
    # only a sequential per-thread iterator with no per-cell entry point, so its walk cannot be
    # split across lanes -- and the walk is most of a query's cost. ``tile_bvh_query_aabb`` hands
    # one candidate per lane per step instead, which is several-fold faster at the front sizes a
    # wave actually has. (A *serial* BVH walk is slower than the hash grid, so the win is the
    # cooperation, not the tree.)
    #
    # The box the BVH returns is a different superset of the true candidate set than the grid's
    # cells were -- both are supersets, and ``candidate_prefilter`` plus ``compute_ball_center``'s
    # ``inf`` do the actual rejecting, so the accepted set is unchanged.
    #
    # ``ball_is_empty`` inside ``candidate_accepted`` stays a per-lane serial hash-grid query: at
    # that point every lane is testing a *different* candidate ball, so there is nothing for the
    # block to cooperate on. It runs 32-way concurrently instead, which is where its speedup comes
    # from.
    if counters[CNT_CONTINUE] == 0:
        return
    block, lane = wp.tid()
    seeding = counters[CNT_SEEDING] != 0
    min_cluster_sq = (clustering * radius) * (clustering * radius)
    # Grid-stride over the front so the launch dimension is a fixed constant — a hard requirement
    # for capturing the wave loop as a CUDA graph, and it also keeps the cost proportional to the
    # front rather than to its high-water mark.
    for i in range(block, counters[CNT_FRONT], grid_stride):
        slot = front_in[i]
        # An edge leaves the front for good when a second face closes it or its search failed.
        # Every lane reads the same state, so the whole block takes this branch together and the
        # tile reductions below are always reached uniformly.
        if edges.count[slot] != 1 or edges.state[slot] != EDGE_LIVE:
            continue
        if lane == 0:
            push_front_edge(slot, front_capacity, counters, front_out)
            wp.atomic_add(counters, CNT_LIVE, 1)
        if seeding:
            continue  # a seeding wave only carries the front forward

        src = edges.src[slot]
        tgt = edges.tgt[slot]
        opp = edges.opp[slot]
        p_src = points[src]
        p_tgt = points[tgt]
        tri_norm = triangle_normal(p_src, p_tgt, points[opp])

        # Re-validate the cached argmin first. It stays the argmin while it stays valid (the
        # candidate set only shrinks), and most front edges lose the vertex claim each wave and come
        # back here unchanged, so this is the difference from a full neighbourhood search. Every
        # lane evaluates it on identical data -- the loads broadcast, so the redundancy is free --
        # and only lane 0 acts on the result.
        cached = edges.cand[slot]
        if cached >= 0 and candidate_prefilter(
            points, p_src, p_tgt, src, tgt, opp, cached, min_cluster_sq, point_used, boundary_degree
        ):
            cached_center = compute_ball_center(
                p_src, p_tgt, points[cached], normals[src] + normals[tgt] + normals[cached], radius
            )
            if cached_center[0] != wp.inf and candidate_accepted(
                points, normals, p_src, p_tgt, tri_norm, src, tgt, cached, cached_center,
                grid_id, radius, crease_cos, edges,
            ):  # fmt: skip
                if lane == 0:
                    propose_triangle(
                        tgt, src, cached, front_capacity, counters, out_owner, out_a, out_b, out_c
                    )
                continue

        center = compute_ball_center(
            p_src, p_tgt, points[opp], normals[src] + normals[tgt] + normals[opp], radius
        )
        if center[0] == wp.inf:
            edges.state[slot] = EDGE_RETIRED  # every lane stores the same value
            continue

        mp = wp.lerp(p_src, p_tgt, 0.5)
        # Signed by the stored edge direction, so the rotation rolls away from ``opp`` only because
        # the face is wound with the normals (module docstring, "Winding").
        axis = wp.normalize(p_tgt - p_src)
        a_dir = wp.normalize(center - mp)

        # Each lane keeps its own running best over the candidates it is handed. That weakens the
        # "reject a non-improving candidate before the expensive tests" pruning — a lane cannot see
        # the other lanes' minima — so more candidates reach ``candidate_accepted``. It is still the
        # same answer: the global minimum over accepted candidates is the minimum of the per-lane
        # minima, and the extra acceptance tests are paid for by their own 32-way concurrency.
        best_angle = TWO_PI
        best = wp.int32(-1)
        reach = wp.vec3(2.0 * radius)
        # **The box is walked by ``wp.tile_bvh_query_aabb`` only when it meets no overfull leaf**
        # (``mark_overfull_point_leaves``): that builtin overruns its shared result buffer on a
        # round meeting a leaf of more than ``TILE_WALK_LEAF_MAX`` points, dropping candidates and
        # reading past the buffer inside the builtin, where no check on the index it returns can
        # reach (``kernels.proximity``, above ``TILE_WALK_LEAF_MAX``). Every lane computes the same
        # answer from the same box, so the branch is block-uniform and the builtin's collective
        # walk is reached by every lane or by none. Any other box is cut into one slab per lane
        # (``straggler_slab``), each walked by the serial ``wp.bvh_query_aabb``, which drops
        # nothing; a point on a cut is tested twice, which only repeats a test. On the cpu device
        # (one lane a block) both walks are the same serial query over the whole box.
        lower = mp - reach
        upper = mp + reach
        if meets_overfull_leaf(bvh_id, overfull, lower, upper):
            slab_lower, slab_upper = straggler_slab(lower, upper, lane, wp.block_dim())
            serial = wp.bvh_query_aabb(bvh_id, slab_lower, slab_upper)
            c = wp.int32(0)
            while wp.bvh_query_next(serial, c):
                best_angle, best = pivot_candidate(
                    points, normals, p_src, p_tgt, tri_norm, src, tgt, opp, c, mp, axis, a_dir,
                    min_cluster_sq, point_used, boundary_degree, grid_id, radius, crease_cos,
                    edges, best_angle, best,
                )  # fmt: skip
        else:
            query = wp.tile_bvh_query_aabb(bvh_id, lower, upper)
            while wp.tile_query_valid(query):
                # A lane with no candidate this step gets -1; the tile is block-wide, so it cannot
                # simply leave the loop.
                c = wp.untile(wp.tile_bvh_query_next(query))
                best_angle, best = pivot_candidate(
                    points, normals, p_src, p_tgt, tri_norm, src, tgt, opp, c, mp, axis, a_dir,
                    min_cluster_sq, point_used, boundary_degree, grid_id, radius, crease_cos,
                    edges, best_angle, best,
                )  # fmt: skip

        # The winner must not depend on which lane happened to see it, which is what
        # ``block_argmin``'s second stage is for. When no lane found anything every lane still holds
        # ``(TWO_PI, -1)``, so it returns -1.
        _block_angle, block_best = block_argmin(best_angle, best)

        edges.cand[slot] = block_best
        if block_best < 0:
            # The search found nothing, so this edge is Open3D's ``Border``: retiring it is
            # output-preserving because the candidate set only ever shrinks (module docstring),
            # not an approximation. An edge that merely *lost* its vertex claim keeps its cached
            # candidate and comes back next wave, which is the branch above.
            edges.state[slot] = EDGE_RETIRED
        elif lane == 0:
            # ``(tgt, src, c)``: the new triangle crosses the edge in the direction opposite to the
            # face that owns it, which is what makes the two consistently wound.
            propose_triangle(
                tgt, src, block_best, front_capacity, counters, out_owner, out_a, out_b, out_c
            )


@wp.func
def wave_salt(wave: wp.int32) -> wp.int32:
    # Per-wave permutation seed for ``proposal_key``. Knuth's multiplicative hash, so consecutive
    # wave numbers give unrelated salts rather than salts differing in one bit, masked to 31 bits so
    # xoring it into a non-negative point index cannot set the sign bit -- which is what keeps the
    # packed key under 2^63 and clear of the unclaimed sentinel.
    return wp.int32((wp.uint32(wave) * wp.uint32(2654435761)) & wp.uint32(0x7FFFFFFF))


@wp.func
def proposal_key(a: wp.int32, b: wp.int32, salt: wp.int32) -> wp.uint64:
    # Priority of a proposed triangle, and the whole reason a run is reproducible: it is derived
    # from the proposal's own vertices, not from the order it reached ``propose_triangle``. That
    # call's slot comes from a ``wp.atomic_add``, i.e. the arrival order of a thousand-block
    # launch, so using it as the claim priority makes *which* triangle wins a contested vertex a
    # function of GPU scheduling — and the loser's front edge is retried a wave later against
    # mutated state, so the difference compounds into a different mesh rather than a permuted one.
    #
    # ``a`` and ``b`` are the proposal's *source edge* — the pivoting front edge ``(src, tgt)``, or
    # a seeding point and its first partner — and that pair is unique among one wave's proposals,
    # which is what makes this a total order on them:
    #
    # * a pivot wave proposes at most once per live front edge, and a front edge *is* one slot of a
    #   table keyed by the undirected pair, so no two share ``{src, tgt}``;
    # * a seed wave proposes at most once per seeding point ``p``, and ``seed_triangles`` seeds only
    #   a point that is the lowest-indexed unused point of the neighbourhood it *scanned*, so ``p``
    #   is the smaller of its pair and distinct seeds give distinct pairs;
    # * the two never share a wave — ``pivot_front_edges`` proposes nothing while ``CNT_SEEDING``.
    #
    # The second bullet is exact only while a neighbourhood fits in ``MAX_SEED_NEIGHBORS``; past
    # that the scan is truncated deliberately (see ``seed_triangles``) and two points can each seed
    # with the other as partner, giving one key twice. **That degrades rather than corrupting**,
    # which is what makes the truncation affordable: both proposals win and commit, they share
    # exactly the edge ``(p1, p2)``, and ``register_face_edge`` folds them correctly —
    # ``hash_find_or_insert`` publishes through a single ``atomic_cas`` so both threads converge on
    # one slot, and the ``atomic_add`` hands ``previous == 0`` to one and ``1`` to the other, so the
    # edge is inserted once, pushed once, and its boundary degree nets to zero. What is *not*
    # preserved is the per-wave vertex-disjointness ``register_face_edge`` states above; the state
    # stays consistent because every transition it makes is an atomic read-modify-write.
    #
    # ``salt`` rotates that order **per wave**, which is what keeps the ordering from being global.
    # An order fixed for the whole run starves a high-key front edge — it loses every contested
    # vertex it enters, is retried, and loses again, so a wave commits a smaller independent set, at
    # roughly double the waves for an unchanged face count. It gives nothing back to the scheduler:
    # the salt is a function of ``CNT_WAVE``, device state the wave loop advances deterministically.
    #
    # Xoring both halves preserves the injectivity the total order needs, and the salt is masked to
    # 31 bits so it cannot set a sign bit. A bit pack rather than ``pack_edge_key``'s
    # ``lo + hi * base``, so neither kernel computing it carries the point count as an argument and
    # the key stays under 2^63; spelled as a multiply because that is the same operation on
    # ``uint64`` as a 32-bit shift and reads as the pack it is.
    lo = wp.uint64(wp.uint32(wp.min(a, b) ^ salt))
    hi = wp.uint64(wp.uint32(wp.max(a, b) ^ salt))
    return hi * wp.uint64(4294967296) + lo


@wp.kernel(enable_backward=False)
def claim_triangle_vertices(
    tri_a: wp.array[wp.int32],
    tri_b: wp.array[wp.int32],
    tri_c: wp.array[wp.int32],
    grid_stride: wp.int32,
    proposal_capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.uint64],
) -> None:
    # Priority claim (lowest ``proposal_key`` wins) on all three vertices of each proposed triangle.
    if counters[CNT_CONTINUE] == 0:
        return
    # ``propose_triangle`` increments the counter past the capacity before dropping, so the count
    # is not a safe bound on the list it indexes.
    proposals = wp.min(counters[CNT_PROPOSAL], proposal_capacity)
    # Same wave index, and so the same salt, as ``commit_triangles`` reads: ``end_wave`` is what
    # advances ``CNT_WAVE`` and it runs after both.
    salt = wave_salt(counters[CNT_WAVE])
    for t in range(wp.int32(wp.tid()), proposals, grid_stride):
        key = proposal_key(tri_a[t], tri_b[t], salt)
        wp.atomic_min(out_owner, tri_a[t], key)
        wp.atomic_min(out_owner, tri_b[t], key)
        wp.atomic_min(out_owner, tri_c[t], key)


@wp.func
def register_face_edge(
    u: wp.int32,
    v: wp.int32,
    opp: wp.int32,
    boundary_degree: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    front_out: wp.array[wp.int32],
) -> None:
    # Fold one edge of a just-committed triangle into the persistent state.
    #
    # A wave's committed triangles are vertex-disjoint (that is exactly what the ``owner`` claim
    # buys), so they share no edge and no vertex: within a wave each edge slot is touched by one
    # thread. The atomics still make it safe, but the count transitions are unambiguous — 0 -> 1
    # means a new boundary edge, 1 -> 2 means one just closed.
    slot = hash_find_or_insert(pack_edge_key(u, v, edges.key_base), edges.key, edges.mask)
    previous = wp.atomic_add(edges.count, slot, 1)
    if previous == 0:
        edges.src[slot] = u
        edges.tgt[slot] = v
        edges.opp[slot] = opp
        edges.cand[slot] = -1
        push_front_edge(slot, front_capacity, counters, front_out)
        wp.atomic_add(boundary_degree, u, 1)
        wp.atomic_add(boundary_degree, v, 1)
    elif previous == 1:
        wp.atomic_add(boundary_degree, u, -1)
        wp.atomic_add(boundary_degree, v, -1)


@wp.kernel(enable_backward=False)
def commit_triangles(
    tri_a: wp.array[wp.int32],
    tri_b: wp.array[wp.int32],
    tri_c: wp.array[wp.int32],
    owner: wp.array[wp.uint64],
    max_faces: wp.int32,
    grid_stride: wp.int32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    front_out: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The single mutator of the persistent state: a proposal that owns all three of its vertices
    # this wave becomes a triangle, and the same thread folds it into the face buffer, the used
    # mask, the edge table, the boundary degrees and the outgoing front.
    if counters[CNT_CONTINUE] == 0:
        return
    # See ``claim_triangle_vertices``: the proposal counter overshoots its list on overflow.
    proposals = wp.min(counters[CNT_PROPOSAL], front_capacity)
    # The triangle budget is tested for the *whole* wave rather than per triangle, because a
    # partial commit would hand the last slots out in ``wp.atomic_add`` arrival order -- the one
    # thing this design does not let the answer depend on. Declining every proposal is also the
    # cheaper recovery: nothing is mutated, so each front edge keeps its cached candidate and the
    # same wave is re-proposed unchanged once the host has doubled the budget.
    #
    # ``CNT_PREV_FACE``, not ``CNT_FACE``: ``begin_wave`` snapshots it and no thread writes it, so
    # every thread reads the same value. Reading ``CNT_FACE`` here would race the ``atomic_add``
    # below and let a late thread decline a wave the early ones already committed to.
    #
    # It is conservative -- most proposals lose the claim and never needed a slot -- but the budget
    # starts at ``4 n + 16`` against a run that commits about ``2 n``, so this is the growth tail
    # and not the common path. Bounding the wave this way is also what lets the scatter below drop
    # its own range check: committed triangles are at most ``proposals``.
    if counters[CNT_PREV_FACE] + proposals > max_faces:
        counters[CNT_GROW] = 1
        return
    salt = wave_salt(counters[CNT_WAVE])
    for t in range(wp.int32(wp.tid()), proposals, grid_stride):
        a = tri_a[t]
        b = tri_b[t]
        c = tri_c[t]
        key = proposal_key(a, b, salt)
        if owner[a] != key or owner[b] != key or owner[c] != key:
            continue
        slot = wp.atomic_add(counters, CNT_FACE, 1)

        out_faces[slot * 3 + 0] = a
        out_faces[slot * 3 + 1] = b
        out_faces[slot * 3 + 2] = c
        point_used[a] = True
        point_used[b] = True
        point_used[c] = True
        register_face_edge(a, b, c, boundary_degree, front_capacity, counters, edges, front_out)
        register_face_edge(b, c, a, boundary_degree, front_capacity, counters, edges, front_out)
        register_face_edge(c, a, b, boundary_degree, front_capacity, counters, edges, front_out)


@wp.kernel(enable_backward=False)
def end_wave(max_waves: wp.int32, counters: wp.array[wp.int32]) -> None:
    # Advance the device-resident wave state and decide whether the loop keeps going.
    #
    # A wave either pivots the current front or seeds orphans; seeding runs whenever the previous
    # wave committed nothing. The termination test used to be exactly that stall (a stalled pivot
    # followed by one stalled seed wave means there is nothing left to do) -- but a single seeding
    # wave marking some elected orphans ``seed_failed`` (see ``seed_triangles``) only *releases*
    # their neighbourhoods for the *next* wave to try, so declaring done the first time a seeding
    # wave commits nothing would end the run before any of those releases had a chance to matter.
    # The second progress test (``CNT_SEED_FAILED`` against its snapshot) is what keeps seeding
    # going for as long as the failure state is still changing, and only declares done once a
    # seeding wave commits no faces *and* discovers no new failures either -- genuinely nothing
    # left to try, not just nothing committed this particular wave.
    if counters[CNT_CONTINUE] == 0:
        return
    counters[CNT_FRONT] = counters[CNT_NEXT_FRONT]
    counters[CNT_WAVE] = counters[CNT_WAVE] + 1
    if counters[CNT_GROW] != 0:
        # A wave that asked the host for room says nothing about progress, and must not be read as
        # a stall: ``commit_triangles`` declines the whole wave when the triangle budget is short,
        # and a front-list overflow is recovered by rebuilding the front from the edge table. Both
        # re-run this wave's work. Testing progress here would flip a pivot wave to seeding, or —
        # if this one was already seeding — raise ``CNT_DONE`` and end the run with a live front,
        # since ``_bpa_run`` checks done before it checks grow.
        counters[CNT_CONTINUE] = 0
        return
    if counters[CNT_FACE] != counters[CNT_PREV_FACE]:
        counters[CNT_SEEDING] = 0
    elif counters[CNT_SEEDING] != 0:
        if counters[CNT_SEED_FAILED] == counters[CNT_PREV_SEED_FAILED]:
            counters[CNT_DONE] = 1
            counters[CNT_CONTINUE] = 0
        # else: still seeding, and the failure set is still changing -- keep going.
    else:
        counters[CNT_SEEDING] = 1
    # Hand control back to the host to compact a front that has accumulated too many retired
    # entries, or because the wave cap was hit. (Growing the budget is the early return above.)
    if counters[CNT_WAVE] >= max_waves:
        counters[CNT_CONTINUE] = 0
    elif counters[CNT_NEXT_FRONT] > 4 * counters[CNT_LIVE] + 1024:
        counters[CNT_CONTINUE] = 0
    # The next wave's reset, done here rather than by a ``begin_wave`` launch of its own whenever
    # the loop keeps going. A loop handed back to the host keeps this wave's counters, which the
    # host's growth and compaction read; the batch that resumes it opens with ``begin_wave``.
    if counters[CNT_CONTINUE] != 0:
        reset_wave_counters(counters)


@wp.kernel(enable_backward=False)
def compact_front(
    front_in: wp.array[wp.int32],
    grid_stride: wp.int32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    front_out: wp.array[wp.int32],
) -> None:
    # Drop closed and retired edges from the front. ``pivot_front_edges`` already does this as a
    # side effect, but a long run of seeding waves (which carry the front forward untouched) or a
    # burst of commits can still leave it sparse; the caller runs this when it does.
    for i in range(wp.int32(wp.tid()), counters[CNT_FRONT], grid_stride):
        slot = front_in[i]
        if edges.count[slot] == 1 and edges.state[slot] == EDGE_LIVE:
            push_front_edge(slot, front_capacity, counters, front_out)


@wp.kernel(enable_backward=False)
def rehash_edges(
    old_key: wp.array[wp.uint64],
    old_count: wp.array[wp.int32],
    old_src: wp.array[wp.int32],
    old_tgt: wp.array[wp.int32],
    old_opp: wp.array[wp.int32],
    old_state: wp.array[wp.int32],
    old_cand: wp.array[wp.int32],
    new_mask: wp.int32,
    new_key: wp.array[wp.uint64],
    new_count: wp.array[wp.int32],
    new_src: wp.array[wp.int32],
    new_tgt: wp.array[wp.int32],
    new_opp: wp.array[wp.int32],
    new_state: wp.array[wp.int32],
    new_cand: wp.array[wp.int32],
) -> None:
    # Re-insert every occupied slot into a larger table when the triangle budget grows. Slot
    # indices change, so the caller rebuilds the front list from the new table afterwards.
    h = wp.int32(wp.tid())
    stored = old_key[h]
    if stored == wp.uint64(0):
        return
    key = stored - wp.uint64(1)  # undo ``encode_key``: 0 is the empty sentinel
    slot = hash_find_or_insert(key, new_key, new_mask)
    new_count[slot] = old_count[h]
    new_src[slot] = old_src[h]
    new_tgt[slot] = old_tgt[h]
    new_opp[slot] = old_opp[h]
    new_state[slot] = old_state[h]
    new_cand[slot] = old_cand[h]


@wp.kernel(enable_backward=False)
def collect_front_from_table(
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    out_front: wp.array[wp.int32],
) -> None:
    # Rebuild the front list by scanning the edge table, after a rehash has moved every slot.
    h = wp.int32(wp.tid())
    if edges.key[h] == wp.uint64(0):
        return
    if edges.count[h] == 1 and edges.state[h] == EDGE_LIVE:
        push_front_edge(h, front_capacity, counters, out_front)
