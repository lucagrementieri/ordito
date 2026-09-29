from typing import NamedTuple

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, INT64_MAX
from triwarp.kernels import array as kernel_array
from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.kernels.array import declare_map_signatures, map_probe, map_probe_single
from triwarp.kernels.reduce import block_chunk_1d, block_min

# Iterative-deepening k-nearest search. A scan at radius ``r`` enumerates every point within
# Euclidean distance ``r``, so a row whose k-th distance is at most ``r`` is provably the exact
# k-NN and the loop can stop. Both accelerators enumerate the ball directly -- the BVH through
# ``wp.bvh_query_sphere`` and the grid through ``wp.hash_grid_query`` -- so the certificate is the
# same statement on both, and neither pays for the bounding cube's extra 6/pi ~ 1.91x of volume.
#
# The loop is a bounded ``for``, never a ``while``: with a NaN query every comparison is false and
# ``r * RADIUS_GROWTH`` stays NaN, which would hang the device. The last attempt is forced to the
# complete radius, which makes termination unconditional and the result exact regardless.
#
# The attempt budget is generous because growth is geometric: the doublings before the successful
# scan sum to less than that scan costs, so a spare attempt is nearly free — whereas running *out*
# of attempts forces the complete scan, which is the whole-cloud brute force this change exists to
# avoid. 16 doublings cover a query 3 x 10^4 spacings away from the cloud.
MAX_SEARCH_ATTEMPTS = wp.constant(wp.int32(16))
RADIUS_GROWTH = wp.constant(wp.float32(2.0))

# Index a grid k-NN row carries when its search was handed to a tree -- ``nearest_point_via_mesh``
# at ``k = 1``, the BVH row kernel's ``only_deferred`` pass above it -- rather than finished by a
# linear scan. Distinct from ``-1``, which is an answer: no point in range.
DEFERRED_ROW = wp.constant(wp.int32(-2))

# Above ``k = 1`` a grid k-NN row is deferred to the BVH only when its nearest point lies within
# this fraction of the initial radius, i.e. when the query is (essentially) a point of the cloud.
# Measured on a 1.5x-scaled, jittered bunny as the query set: the rows the grid gives up on there
# answer within ~2 initial radii yet cost the BVH's radius search more than a scan of the whole
# cloud, because a small ball in the empty space a surface cloud encloses still overlaps many of
# the tree's large interior nodes. A self-query in the sparse part of a density-graded cloud is the
# opposite case, and the one the deferral exists for.
DEFER_NEAR = wp.constant(wp.float32(0.5))

# Candidate-row sizes the register-resident k-NN kernels are generated for. The row is held in a
# ``wp.types.vector(length=K)`` value type, i.e. in registers, so ``K`` must be a compile-time
# constant and one kernel exists per bucket. A query takes the smallest bucket that fits its ``k``,
# keeps the ``K`` nearest (a superset of the ``k`` nearest) and writes out only the first ``k``
# slots; a ``k`` past the largest bucket falls back to the global-memory row kernels.
#
# The register row beats the global-memory one at every bucket, by more the wider the row.
# **64 is the last bucket because that is where the curve turns, not because the gain runs out**: a
# row costs ``2 * K`` registers, and ``wp.get_cuda_kernel_properties`` reports the six shipped
# buckets at 42-222 registers with ``local_memory_size`` **0** at every one, so nothing shipped
# spills. A 96-wide row extrapolates past the 255-register-per-thread hardware limit and must
# spill, which drops it back below the k=64 gain -- the bucket list ends where the register file
# does, not at an arbitrary cut, and no in-repo caller asks for more.
#
# **``cuda_max_registers`` does not move that wall, and it was measured rather than assumed.**
# Capping below a kernel's natural register count does not make it leaner, it makes it *spill*:
# rebuilt under every cap, the shipped buckets report non-zero ``local_memory_size`` and run
# several times slower, with results identical. That is also the answer to the 96-bucket question
# above. The buckets stay uncapped.
#
# The buckets are not free: they cost ``2 x len(KNN_ROW_BUCKETS)`` generated kernels, which roughly
# triples this module's cold-cache compile (once per Warp version and arch) and doubles its warm
# per-process load.
KNN_ROW_BUCKETS = (1, 4, 8, 16, 32, 64)

# Which accelerator ``ball_count_in_radius`` / ``ball_collect`` traverse. The ball query is one
# algorithm -- same acceptance rule, same emit protocol -- over two broad phases whose query
# objects are different types with different ``_next`` builtins, so the enumeration cannot be
# abstracted behind a ``wp.Function`` parameter (CLAUDE.md section 2.7: ``wp.launch`` cannot pass
# one as a kernel argument). An int selector can: the branch is warp-uniform, both traversals
# compile into this one module, and the merged kernels measure within noise of the two they replace
# on the BVH side and slightly faster on the hash-grid side, where the counting pass no longer
# allocates a throwaway per-thread distance slot.
ACCEL_HASHGRID = wp.constant(wp.int32(0))
ACCEL_BVH = wp.constant(wp.int32(1))


# The two BVH walk protocols, over a *query* rather than over a bvh id and bounds.
#
# Taking the query is what Warp 1.17 made possible and it is the whole point of these two: a
# ``wp.BvhQuery`` is the common type over the aabb / sphere / capsule query kinds, and a
# ``@wp.func`` may take one as a parameter and another may return one. Before that, a walk was
# pinned to the constructor that opened it, so every query kind carried its own copy of these four
# lines.
#
# ``@wp.func`` calls inline at codegen, so this is free -- confirmed rather than assumed, with
# ``wp.get_cuda_kernel_properties`` reporting identical register counts and zero local memory
# across the extraction. And the move is *provably* behaviour-neutral where a float32 extraction
# would not be (CLAUDE.md section 2.4 warns that a green suite is not evidence): neither shared run
# contains a floating-point expression, so there is no evaluation order for it to disturb.


@wp.func
def bvh_walk_count(query: wp.BvhQuery) -> wp.int32:
    # Every primitive the query returns, counted, with no narrow phase.
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        c = c + 1
    return c


@wp.func
def bvh_walk_emit(query: wp.BvhQuery, base: wp.int32, out_indices: wp.array[wp.int32]) -> wp.int32:
    # The same primitives ``bvh_walk_count`` counts, written contiguously from ``base``. Counting
    # and emitting are separate walks rather than one ``write``-flagged function: the counting pass
    # then needs no output array at all, and the emit loop carries no per-candidate branch.
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        out_indices[base + c] = j
        c = c + 1
    return c


# The ball-shaped broad phase over bounds, and it is *not* a tighter spelling of the box one below.
# ``wp.bvh_query_sphere`` prunes on an exact sphere-AABB squared distance where
# ``wp.bvh_query_aabb`` prunes on a box overlap, and on Warp 1.17 the box traversal is far the more
# expensive of the two *per candidate returned* -- isolated with the box's own *inscribed* cube
# (half extent ``r / sqrt(3)``, so strictly fewer candidates than the ball), it still costs several
# times the ball query. So the ball trims candidates *and* walks more cheaply, and a caller whose
# predicate is a ball wants this pair rather than the box pair plus a narrow phase. Both counts are
# exact against a brute-force oracle; CLAUDE.md section 12.8 records the verdict.
@wp.func
def ball_count_in_bounds(bvh_id: wp.uint64, q: wp.vec3, radius: wp.float32) -> wp.int32:
    # Broad-phase hits of the ball: every bound whose AABB comes within ``radius`` of ``q``.
    return bvh_walk_count(wp.bvh_query_sphere(bvh_id, q, radius))


@wp.kernel
def query_bvh_ball_neighbors(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    bvh_walk_emit(wp.bvh_query_sphere(bvh_id, queries[tid], radius), offsets[tid], out_indices)


# The per-query-box pair, in the order its wrapper appears (section 5). Same walk skeleton as the
# ball above and a different node test; the count half is a ``wp.map`` over
# ``aabb_count_in_bounds`` rather than a kernel shim (CLAUDE.md section 3.5).
#
# There was a third pair here, for one warp-uniform half extent, and it is gone: it was exactly
# this one at ``q -+ h``, returned an identical set, and the corner buffers it saved measured flat
# against a query several milliseconds long. Do not reintroduce it without a number.
@wp.func
def aabb_count_in_bounds(bvh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3) -> wp.int32:
    # Broad-phase hits of the box.
    return bvh_walk_count(wp.bvh_query_aabb(bvh_id, lower, upper, root=-1))


@wp.func
def aabb_collect_in_bounds(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
) -> None:
    # Emit the same hits ``aabb_count_in_bounds`` counted; the emit protocol is ``bvh_walk_emit``'s.
    bvh_walk_emit(wp.bvh_query_aabb(bvh_id, lower, upper, root=-1), base, out_indices)


@wp.kernel
def query_bvh_box_neighbors(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    aabb_collect_in_bounds(bvh_id, query_lower[tid], query_upper[tid], offsets[tid], out_indices)


@wp.func
def in_ball(offset: wp.vec3, radius: wp.float32) -> wp.bool:
    # The ball predicate every hash-grid ball walk applies to a candidate ``points[j] - q``: the
    # squared form, for the reasons ``ball_count_in_radius`` gives. The count, the collect and
    # ``kernels.interpolation.interpolate_from_points_in_ball`` all call it, since a count that
    # accepts what the collect rejects leaves the CSR it sized with unwritten slots.
    return wp.length_sq(offset) <= radius * radius


@wp.func
def ball_count_in_radius(
    points: wp.array[wp.vec3], accel: wp.int32, accel_id: wp.uint64, q: wp.vec3, radius: wp.float32
) -> wp.int32:
    # Points within Euclidean ``radius`` of ``q``, over either accelerator (see ACCEL_* above).
    #
    # The two query objects are deliberately *differently named*: a Warp variable's type is fixed by
    # its first assignment, so binding one ``query`` name to a hash-grid query in one branch and a
    # BVH query in the other does not compile. Do not "tidy" them into a single name. Warp 1.17's
    # ``wp.BvhQuery`` common type lifted this *between BVH kinds* -- a box query and a sphere query
    # can share one variable -- but a hash-grid query is still not a ``BvhQuery``.
    #
    # **The predicate is the squared one, on both branches, and that is a decision rather than a
    # style.** ``wp.bvh_query_sphere`` prunes on an exact sphere-AABB squared-distance test, and on
    # this BVH -- built by ``neighbors.bvh_from_points`` as ``wp.Bvh(points, points)``, so every
    # leaf bound is a degenerate point -- that test *is* the point-in-ball test. There is no narrow
    # phase left to write, and it beats the cube-plus-``wp.length`` form it replaced by more the
    # denser the neighbourhood, on both devices.
    #
    # The hash-grid branch is spelled ``wp.length_sq`` to *match* it. ``wp.length(d) <= radius`` and
    # ``wp.length_sq(d) <= radius * radius`` are not the same predicate in float32 -- ``sqrt`` and
    # ``radius * radius`` round independently -- and the sphere query agrees with the squared form
    # on every row and with the sqrt form on all but a handful, identically on CPU and CUDA. So
    # leaving the grid on ``wp.length`` would make ``backend="hashgrid"`` and ``backend="bvh"``
    # answer differently at the boundary, which is the one-predicate-one-spelling defect section 3
    # of CLAUDE.md names; ``test_the_two_backends_agree`` is the gate.
    #
    # The two spellings are speed-flat as a *narrow* phase, which is why an earlier pass kept the
    # sqrt form; what changed is that the squared form is now what the tighter broad phase speaks.
    c = wp.int32(0)
    j = wp.int32(0)
    if accel == ACCEL_HASHGRID:
        query = wp.hash_grid_query(accel_id, q, radius)
        while wp.hash_grid_query_next(query, j):
            if in_ball(points[j] - q, radius):
                c = c + 1
    else:
        # No narrow phase: on degenerate point bounds the sphere-AABB test is the ball test, so
        # this is the bare walk ``bvh_walk_count`` also serves the box query with.
        c = bvh_walk_count(wp.bvh_query_sphere(accel_id, q, radius))
    return c


@wp.kernel
def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    accel: wp.int32,
    accel_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    out_neighbor_counts[tid] = ball_count_in_radius(points, accel, accel_id, queries[tid], radius)


@wp.func
def ball_collect(
    points: wp.array[wp.vec3],
    accel: wp.int32,
    accel_id: wp.uint64,
    q: wp.vec3,
    radius: wp.float32,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    # Emit the same neighbours ``ball_count_in_radius`` counted, contiguously from ``base``, with
    # the distance the caller wants alongside. Traversal order fixes the within-query order, which
    # is why the wrapper's ``return_sorted`` is a separate segmented sort.
    #
    # Both branches must accept exactly what ``ball_count_in_radius`` counts or the offsets it
    # produced would not fit -- so the predicate is the squared one here too, for the reasons
    # written there. The BVH branch needs no acceptance test at all (the sphere query already made
    # it), and the hash-grid branch tests squared and takes the square root only for the candidates
    # it keeps, where the sqrt form paid for one on every candidate it walked.
    c = wp.int32(0)
    j = wp.int32(0)
    if accel == ACCEL_HASHGRID:
        query = wp.hash_grid_query(accel_id, q, radius)
        while wp.hash_grid_query_next(query, j):
            offset = points[j] - q
            if in_ball(offset, radius):
                out_indices[base + c] = j
                out_distances[base + c] = wp.length(offset)
                c = c + 1
    else:
        query_sphere = wp.bvh_query_sphere(accel_id, q, radius)
        while wp.bvh_query_next(query_sphere, j):
            out_indices[base + c] = j
            out_distances[base + c] = wp.length(points[j] - q)
            c = c + 1


@wp.kernel
def query_ball_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    accel: wp.int32,
    accel_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    ball_collect(
        points, accel, accel_id, queries[tid], radius, offsets[tid], out_indices, out_distances
    )


@wp.func
def knn_sorted_insert(
    point_index: wp.int32,
    d: wp.float32,
    k: wp.int32,
    max_radius: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> None:
    # Insert ``(point_index, d)`` into the ascending k-nearest rows, dropping the current worst.
    #
    # **The NaN rejection is a memory-safety guard, not tidiness.** Every comparison against a NaN
    # is false, so a NaN ``d`` passes both admission tests below (``NaN > max_radius`` and
    # ``NaN >= inf`` are each false) and then steers ``binary_search_index`` -- which is
    # ``searchsorted(side="right")``, so its ``values[mid] > value`` probe never fires -- all the
    # way to ``left = n``. It returns ``n``, and ``array_shift_insert`` writes ``row[n]``: one
    # element past the caller's ``k``-wide row, i.e. into the next query's row, or past the whole
    # ``(m, k)`` allocation for the last query. Confirmed on both devices with a canary row that no
    # thread was launched over: it comes back holding the NaN and the offending point index.
    #
    # It is reachable from the public surface at ``k > 64`` (below that the register-row kernels
    # bound their own loops): a NaN query point makes ``wp.hash_grid_query`` clamp into the guard
    # region and enumerate cell 0 (``warp/native/hashgrid.h`` asserts on it only in debug mode),
    # and ``knn_linear_scan`` enumerates the whole cloud unconditionally, so *every* candidate then
    # arrives here with ``d`` NaN. A NaN in ``points`` reaches it the same way. On the CPU device
    # an out-of-bounds kernel write is host-heap corruption rather than a fault.
    #
    # Spelled ``wp.isnan`` rather than folding the test into the two comparisons below (``if not
    # (d <= max_radius): return``): the negated form generates the same code but reads as a typo,
    # and this is the one line here a future edit must not "simplify".
    if wp.isnan(d):
        return
    if d > max_radius:
        return
    if d >= out_distances_row[k - 1]:
        return
    if k == 1:
        out_indices_row[0] = point_index
        out_distances_row[0] = d
        return
    slot = kernel_array.binary_search_index(out_distances_row, d)
    kernel_array.array_shift_insert(out_distances_row, d, slot)
    kernel_array.array_shift_insert(out_indices_row, point_index, slot)


@wp.func
def knn_reset_row(
    k: wp.int32, out_indices_row: wp.array[wp.int32], out_distances_row: wp.array[wp.float32]
) -> None:
    # Every scan starts from an empty row. ``knn_sorted_insert`` does not deduplicate, so a
    # re-scan over a wider radius would otherwise insert each already-found point a second time.
    for i in range(k):
        out_indices_row[i] = wp.int32(-1)
        out_distances_row[i] = FLOAT32_INF_CONSTANT


@wp.func
def complete_radius(q: wp.vec3, min_bound: wp.vec3, max_bound: wp.vec3) -> wp.float32:
    # Smallest **ball** radius about ``q`` that contains the whole point bounding box, i.e. the
    # radius at which a scan is provably complete: the distance from ``q`` to the box's farthest
    # corner. Per-query, so it is tighter than a global diagonal, and unbounded for a query far
    # outside the box (which is what keeps that case exact).
    #
    # This is a *ball* radius and not the cube half-extent it used to be, because every enumeration
    # it bounds is now a ``wp.bvh_query_sphere`` or a ``wp.hash_grid_query``, both of which take a
    # Euclidean radius. The two differ by up to sqrt(3), and the cube form was the smaller of the
    # two -- so as a ball radius it did *not* cover the box, which would have made a
    # forced-complete final attempt incomplete. Every caller therefore gets a radius at least as
    # large as before: exact where it was exact, and no longer under-covering on the hash-grid path,
    # whose ``wp.hash_grid_query`` has always read this as a Euclidean radius.
    reach = wp.max(q - min_bound, max_bound - q)  # per-component farthest face distance
    return wp.length(reach)


@wp.func
def knn_bvh_scan(
    points: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    q: wp.vec3,
    k: wp.int32,
    max_radius: wp.float32,
    r: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> wp.float32:
    # Refill the row from the **ball** of radius ``r`` about ``q`` and return the k-th best distance
    # (``inf`` when fewer than ``k`` points were accepted). Acceptance stays ``d <= max_radius``;
    # ``r`` bounds only the enumeration.
    #
    # The enumeration used to be the cube ``[q +/- r]``, whose certificate rested on Chebyshev
    # distance never exceeding Euclidean. ``wp.bvh_query_sphere`` enumerates the ball itself, so the
    # certificate is now direct -- every point outside the ball is farther than ``r``, hence a
    # ``worst <= r`` row is exact -- over a strictly smaller candidate set: the cube holds
    # 6/pi ~ 1.91x the ball's volume, and it is that ratio the caller's speedup comes out of.
    knn_reset_row(k, out_indices_row, out_distances_row)
    query = wp.bvh_query_sphere(bvh_id, q, r)
    point_index = wp.int32(0)
    while wp.bvh_query_next(query, point_index):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)
    return out_distances_row[k - 1]


@wp.func
def next_search_radius(worst: wp.float32, r: wp.float32, r_hard: wp.float32) -> wp.float32:
    # The iterative-deepening decision every exact search in this package makes after a scan at
    # radius ``r``, one definition because it is a *rule*, not arithmetic: the row is final when
    # everything outside the ball is farther than ``worst`` (certified) or when ``r`` already
    # reached ``r_hard`` (the scan was complete); otherwise deepen. ``-1`` says final -- a real
    # radius is never negative -- and each caller maps it onto its own ``break`` / ``return``.
    if worst <= r:
        return wp.float32(-1.0)
    if r >= r_hard:
        return wp.float32(-1.0)
    # Deepen. A full row (finite ``worst``) reaching past the ball certifies at exactly ``worst``,
    # so jump there; an unfilled row has no bound to jump to and grows geometrically instead.
    if worst < FLOAT32_INF_CONSTANT:
        return wp.min(worst, r_hard)
    return wp.min(r * RADIUS_GROWTH, r_hard)


@wp.func
def search_radius_bounds(
    q: wp.vec3,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    max_radius: wp.float32,
    initial_radius: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    # The iterative deepening's two starting radii about query ``q``: the hard cap ``r_hard`` --
    # the caller's ``max_radius`` or the radius at which a scan is provably complete, whichever is
    # smaller -- and the first attempt's radius, the caller's seed held under that cap. A search
    # with no caller cap passes ``FLOAT32_INF_CONSTANT``.
    r_hard = wp.min(max_radius, complete_radius(q, min_bound, max_bound))
    return r_hard, wp.min(initial_radius, r_hard)


@wp.func
def attempt_radius(attempt: wp.int32, r: wp.float32, r_hard: wp.float32) -> wp.float32:
    # The radius attempt ``attempt`` of a BVH deepening scans at: ``r``, except that the last of
    # ``MAX_SEARCH_ATTEMPTS`` is forced complete, exact whatever the growth did. The hash-grid
    # searches have no such attempt -- they fall back to the linear scan instead.
    return wp.where(attempt == MAX_SEARCH_ATTEMPTS - 1, r_hard, r)


@wp.kernel
def query_bvh_nearest_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    k: wp.int32,
    max_radius: wp.float32,
    initial_radius: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    only_deferred: wp.int32,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    # ``only_deferred`` is the register-row twins' switch for finishing the rows a grid search
    # deferred; this global-row form serves only ``k`` above the largest bucket, where the grid
    # never defers, so it takes the argument and ignores it.
    tid = wp.int32(wp.tid())
    q = queries[tid]
    out_indices_row = out_indices[tid]
    out_distances_row = out_distances[tid]

    r_hard, r = search_radius_bounds(q, min_bound, max_bound, max_radius, initial_radius)
    # This kernel used to carry a note saying it may hold exactly one ``wp.bvh_query_*`` call site,
    # because ``bvh_query`` declares a large ``__shared__`` stack and a second textual call site
    # would double it and fail to compile. **That was never true of the plain query.** In
    # ``warp/native/bvh.h`` the ``__shared__`` declaration sits inside the *tiled* constructor;
    # ``wp.bvh_query_aabb`` / ``_sphere`` use a per-thread ``int stack[BVH_QUERY_STACK_SIZE]`` in
    # local storage. Verified: a kernel with four plain call sites compiles and runs at
    # block_dim=256 on Warp 1.17 with zero local memory. The single call site here is just what the
    # loop needs, not a constraint -- so a future edit needing a second one may add it.
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        r = attempt_radius(attempt, r, r_hard)
        worst = knn_bvh_scan(
            points, bvh_id, q, k, max_radius, r, out_indices_row, out_distances_row
        )
        r = next_search_radius(worst, r, r_hard)
        if r < 0.0:
            break  # certified exact, or the scan was already complete


# ---------------------------------------------------------------------------------------------
# Register-row k-NN kernel factories.
#
# ``query_bvh_nearest_neighbors`` above keeps its candidate row in the *output* arrays, so every
# accepted candidate pays a binary search plus two shift passes over global memory and the row is
# re-zeroed on every deepening attempt: at a wide ``k`` the row, not the geometry, is the call.
# Holding it in a ``wp.types.vector(length=K)`` puts it in registers, which is why these kernels
# exist and why ``K`` must be a compile-time constant. A ``wp.zeros(shape=K)`` stack array is a
# **2x loss** instead, because nothing promotes it to registers (CLAUDE.md section 2.9).
#
# Everything the row touches **once per query or per attempt** is factored into the generated
# ``@wp.func`` set below. The **per-candidate insert alone stays inline**, three times, and the
# split is measured: passing the row to a ``wp.ref`` helper is a large loss on **CPU** at the wide
# buckets, passing it by value a far larger loss on **CUDA**, and the once-per-attempt helpers are
# flat on both. Do not "finish" this dedup by moving the insert too.
#
# The contract the copies must hold is the *distance* row, not the tie-break. ``placed`` is **not**
# load-bearing: without it the carry skips a slot holding an equal distance and displaces further
# down, which permutes *which* of several equidistant points fills a slot and leaves every distance
# bit-identical — and ``query_nearest``'s docstring declares a tied neighbour's identity
# unspecified.
#
# So the guard on an edit here is a *tied* fixture, not a second implementation to diff against:
# ``tests/test_neighbors.py::test_query_nearest_ties`` runs every ``KNN_ROW_BUCKETS`` size against
# ``scipy.spatial.KDTree`` on an integer lattice, where a query has dozens of exactly tied
# neighbours. Verified live: shortening the shift chain by one slot fails every case.
# ---------------------------------------------------------------------------------------------


class _RowHelpers(NamedTuple):
    """One bucket's register-row vector types and the ``@wp.func``s that operate on a whole row."""

    distances: type
    indices: type
    reset: wp.Function
    kth: wp.Function
    write: wp.Function


def _row_helpers(row_size: int) -> _RowHelpers:
    """Generate the ``K = row_size`` row helper set; ``K`` must be a compile-time constant."""
    vec_distances = wp.types.vector(length=row_size, dtype=wp.float32)
    vec_indices = wp.types.vector(length=row_size, dtype=wp.int32)

    def _reset(row_distances: wp.ref[vec_distances], row_indices: wp.ref[vec_indices]) -> None:
        # Every scan starts from an empty row: the insert does not deduplicate, so a re-scan over a
        # wider radius would otherwise insert each already-found point a second time.
        for slot in range(row_size):
            row_indices[slot] = wp.int32(-1)
            row_distances[slot] = FLOAT32_INF_CONSTANT

    def _kth(row_distances: wp.ref[vec_distances], k: wp.int32) -> wp.float32:
        # The k-th distance certifies the scan (``inf`` while the row is unfilled). Read through an
        # unrolled compare rather than ``row_distances[k - 1]``: ``k`` is a runtime value and a
        # runtime index into a vector spills it to local memory.
        worst = FLOAT32_INF_CONSTANT
        for slot in range(row_size):
            if slot == k - 1:
                worst = row_distances[slot]
        return worst

    def _write(
        row_distances: wp.ref[vec_distances],
        row_indices: wp.ref[vec_indices],
        tid: wp.int32,
        k: wp.int32,
        out_indices: wp.array2d[wp.int32],
        out_distances: wp.array2d[wp.float32],
    ) -> None:
        # Only the first ``k`` slots are the caller's answer; the bucket's tail is padding.
        for slot in range(row_size):
            if slot < k:
                out_indices[tid, slot] = row_indices[slot]
                out_distances[tid, slot] = row_distances[slot]

    return _RowHelpers(
        vec_distances,
        vec_indices,
        wp.func(_reset, name=f"knn_row_reset{row_size}"),
        wp.func(_kth, name=f"knn_row_kth{row_size}"),
        wp.func(_write, name=f"knn_row_write{row_size}"),
    )


# One set per bucket: ``wp.ref`` needs a concrete dtype, so the helpers cannot be generic in ``K``.
_ROW_HELPERS = {row_size: _row_helpers(row_size) for row_size in KNN_ROW_BUCKETS}


def _bvh_nearest_row_kernel(row_size: int, name: str):
    """``K = row_size`` register-row variant of ``query_bvh_nearest_neighbors``."""
    vec_distances, vec_indices, row_reset, row_kth, row_write = _ROW_HELPERS[row_size]

    def _kernel(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        bvh_id: wp.uint64,
        k: wp.int32,
        max_radius: wp.float32,
        initial_radius: wp.float32,
        min_bound: wp.vec3,
        max_bound: wp.vec3,
        only_deferred: wp.int32,
        out_indices: wp.array2d[wp.int32],
        out_distances: wp.array2d[wp.float32],
    ) -> None:
        tid = wp.int32(wp.tid())
        if only_deferred != 0 and out_indices[tid, 0] != DEFERRED_ROW:
            return  # the grid already certified this row
        q = queries[tid]
        row_indices = vec_indices()
        row_distances = vec_distances()

        r_hard, r = search_radius_bounds(q, min_bound, max_bound, max_radius, initial_radius)
        # The ball enumeration and its certificate are ``knn_bvh_scan``'s, which this factory
        # cannot call because the row lives in registers rather than in the output arrays; the
        # traversal is the only part duplicated, and the reason it is duplicated is the row type.
        #
        # **The sphere query is a win at most buckets and a small loss at one, and which one moves
        # with the cloud** -- eight of ten measured cells win, every one byte-identical in the
        # index rows. The loss is *not* any of the three things it looks like, each checked: the
        # deepening sequence is unchanged (the certificate compares the k-th *distance*, which no
        # enumeration shape can move), the ball ``complete_radius`` is not implicated, and nothing
        # spills (``local_memory_size`` 0 at every bucket, and *fewer* registers for the sphere form
        # at five of six). What is left is the per-node arithmetic: the exact sphere-AABB test costs
        # more per node than a slab test, and at the bucket where row-insertion traffic and
        # traversal cost balance, the ~1.91x candidate saving stops covering it.
        for attempt in range(MAX_SEARCH_ATTEMPTS):
            r = attempt_radius(attempt, r, r_hard)
            row_reset(row_distances, row_indices)
            query = wp.bvh_query_sphere(bvh_id, q, r)
            point_index = wp.int32(0)
            while wp.bvh_query_next(query, point_index):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = wp.int32(0)
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index
            worst = row_kth(row_distances, k)
            r = next_search_radius(worst, r, r_hard)
            if r < 0.0:
                break  # certified exact, or the scan was already complete

        row_write(row_distances, row_indices, tid, k, out_indices, out_distances)

    # ``enable_backward=False`` is mandatory, not a choice: the row helpers take ``wp.ref``
    # parameters and a ``wp.ref`` helper has no adjoint, so the module fails to compile without it.
    return wp.kernel(_kernel, name=name, enable_backward=False)


_BVH_NEAREST_ROW_KERNELS = {
    row_size: _bvh_nearest_row_kernel(row_size, f"query_bvh_nearest_neighbors_row{row_size}")
    for row_size in KNN_ROW_BUCKETS
}


def bvh_nearest_kernel(k: int) -> wp.Kernel:
    """Register-row BVH k-NN kernel for the smallest bucket fitting ``k``, else global-row."""
    for row_size in KNN_ROW_BUCKETS:
        if k <= row_size:
            return _BVH_NEAREST_ROW_KERNELS[row_size]
    return query_bvh_nearest_neighbors


@wp.func
def knn_hashgrid_scan(
    points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    q: wp.vec3,
    k: wp.int32,
    max_radius: wp.float32,
    r: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> wp.float32:
    # Hash-grid twin of ``knn_bvh_scan``; ``wp.hash_grid_query`` enumerates every cell overlapping
    # ``[q +/- r]``, so the same "k-th distance <= r certifies" argument applies.
    #
    # The two are NOT merged behind the ``ACCEL_*`` selector the ball queries use, and the reason
    # is the one that separates section 2.1's case from section 2.5's: there the two accelerators
    # ran the *same* algorithm and only the enumeration differed, so a warp-uniform int made them
    # one kernel. Here the enclosing searches have genuinely diverged -- the grid path takes a
    # ``widest`` bound and falls back to ``knn_linear_scan`` once the radius outgrows the cell
    # width (a cell walk is cubic in the radius, so past that it costs more than touching every
    # point), while the BVH path forces a final complete attempt instead. A merged kernel would
    # carry a parameter that is ignored on one path and a fallback branch that belongs to the
    # other. What the two genuinely share is already shared: ``knn_reset_row``,
    # ``knn_sorted_insert``, ``complete_radius`` and ``next_search_radius``; what is left is three
    # lines of traversal each, plus each one's own certification policy.
    knn_reset_row(k, out_indices_row, out_distances_row)
    query = wp.hash_grid_query(grid_id, q, r)
    point_index = wp.int32(-1)
    while wp.hash_grid_query_next(query, point_index):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)
    return out_distances_row[k - 1]


@wp.func
def knn_linear_scan(
    points: wp.array[wp.vec3],
    q: wp.vec3,
    k: wp.int32,
    max_radius: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> None:
    # Exact fallback for the grid path once the radius outgrows the cell width. This is the same
    # per-row cost the diagonal-radius query used to pay for *every* row, so it is never a
    # regression against the previous behaviour.
    knn_reset_row(k, out_indices_row, out_distances_row)
    for point_index in range(points.shape[0]):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)


@wp.kernel
def query_hashgrid_nearest_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    k: wp.int32,
    max_radius: wp.float32,
    initial_radius: wp.float32,
    widest: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    defer: wp.int32,
    out_deferred: wp.array[wp.int32],
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    # ``defer`` is the register-row twin's switch (see there); this global-row form serves only
    # ``k`` above the largest bucket, where no caller defers, so it takes the arguments and ignores
    # them.
    tid = wp.int32(wp.tid())
    q = queries[tid]
    out_indices_row = out_indices[tid]
    out_distances_row = out_distances[tid]

    r_hard, r = search_radius_bounds(q, min_bound, max_bound, max_radius, initial_radius)
    for _attempt in range(MAX_SEARCH_ATTEMPTS):
        if not r <= widest:
            # Past ``widest`` a cell walk costs more than touching every point (and a NaN query
            # lands here too, which is what bounds this loop). Finish exactly instead.
            break
        worst = knn_hashgrid_scan(
            points, grid_id, q, k, max_radius, r, out_indices_row, out_distances_row
        )
        r = next_search_radius(worst, r, r_hard)
        if r < 0.0:
            return  # certified exact, or the scan was already complete
    knn_linear_scan(points, q, k, max_radius, out_indices_row, out_distances_row)


def _hashgrid_nearest_row_kernel(row_size: int, name: str):
    """``K = row_size`` register-row variant of ``query_hashgrid_nearest_neighbors``."""
    vec_distances, vec_indices, row_reset, row_kth, row_write = _ROW_HELPERS[row_size]

    def _kernel(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        grid_id: wp.uint64,
        k: wp.int32,
        max_radius: wp.float32,
        initial_radius: wp.float32,
        widest: wp.float32,
        min_bound: wp.vec3,
        max_bound: wp.vec3,
        defer: wp.int32,
        out_deferred: wp.array[wp.int32],
        out_indices: wp.array2d[wp.int32],
        out_distances: wp.array2d[wp.float32],
    ) -> None:
        tid = wp.int32(wp.tid())
        q = queries[tid]
        row_indices = vec_indices()
        row_distances = vec_distances()

        r_hard, r = search_radius_bounds(q, min_bound, max_bound, max_radius, initial_radius)
        certified = wp.int32(0)
        nearest = FLOAT32_INF_CONSTANT  # the last cell walk's nearest distance
        for _attempt in range(MAX_SEARCH_ATTEMPTS):
            if not r <= widest:
                # Past ``widest`` a cell walk costs more than touching every point (and a NaN
                # query lands here too, which is what bounds this loop). Finish exactly instead.
                break
            row_reset(row_distances, row_indices)
            query = wp.hash_grid_query(grid_id, q, r)
            point_index = wp.int32(-1)
            while wp.hash_grid_query_next(query, point_index):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = wp.int32(0)
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index
            nearest = row_distances[0]
            worst = row_kth(row_distances, k)
            r = next_search_radius(worst, r, r_hard)
            if r < 0.0:
                certified = 1  # certified exact, or the scan was already complete
                break

        if certified == 0 and defer != 0 and (k == 1 or nearest <= DEFER_NEAR * initial_radius):
            # Hand the row to a tree instead of scanning. At ``k = 1`` that is
            # ``nearest_point_via_mesh``, whose descent costs the same however far the query is
            # from the cloud. Above it, the BVH row kernel's ``only_deferred`` pass, and only for a
            # query sitting *on* a point of the cloud (``DEFER_NEAR``): that is a self-query in a
            # part of the cloud sparser than the cell width was sized for, where the ball search
            # stays small and a scan would touch every point. A query off the cloud keeps the scan,
            # because there the tree is the slower finish (see ``DEFER_NEAR``).
            # Marked ``DEFERRED_ROW`` rather than ``-1``, which already means "nothing within
            # ``max_radius``", and counted so the caller can skip the second pass when none was.
            row_reset(row_distances, row_indices)
            row_write(row_distances, row_indices, tid, k, out_indices, out_distances)
            out_indices[tid, 0] = DEFERRED_ROW
            wp.atomic_add(out_deferred, 0, 1)
            return
        if certified == 0:
            # Exact fallback once the radius outgrows the cell width or the attempt budget runs
            # out — the register twin of ``knn_linear_scan``.
            row_reset(row_distances, row_indices)
            for point_index in range(points.shape[0]):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = wp.int32(0)
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index

        row_write(row_distances, row_indices, tid, k, out_indices, out_distances)

    _kernel.__name__ = name
    _kernel.__qualname__ = name
    return wp.kernel(_kernel, enable_backward=False)  # ``wp.ref`` helpers, as in the BVH twin


_HASHGRID_NEAREST_ROW_KERNELS = {
    row_size: _hashgrid_nearest_row_kernel(
        row_size, f"query_hashgrid_nearest_neighbors_row{row_size}"
    )
    for row_size in KNN_ROW_BUCKETS
}


def hashgrid_nearest_kernel(k: int) -> wp.Kernel:
    """Register-row grid k-NN kernel for the smallest bucket fitting ``k``, else global-row."""
    for row_size in KNN_ROW_BUCKETS:
        if k <= row_size:
            return _HASHGRID_NEAREST_ROW_KERNELS[row_size]
    return query_hashgrid_nearest_neighbors


@wp.kernel
def point_triangle_indices(out_indices: wp.array[wp.int32]) -> None:
    # Corner ``c`` of triangle ``i`` is point ``i``: every triangle collapsed onto one point, so a
    # ``wp.Mesh`` over them is a point cloud whose closest-face query is a nearest-point query.
    c = wp.int32(wp.tid())
    out_indices[c] = c // 3


@wp.func
def mesh_nearest_point(
    mesh_id: wp.uint64, points: wp.array[wp.vec3], q: wp.vec3, max_radius: wp.float32
) -> tuple[wp.int32, wp.float32]:
    # The nearest point to ``q`` within ``max_radius``, by the mesh BVH's best-first closest-point
    # descent over the collapsed triangles ``point_triangle_indices`` builds: ``(-1, inf)`` when
    # none is in range. Its cost follows the tree's depth rather than the distance to the answer,
    # which is what neither the grid's cell walk nor the BVH's radius deepening does. The closest
    # point of a triangle whose three corners coincide is that corner, exactly, and the distance is
    # recomputed with the ball searches' own ``wp.length`` so a row answers the same whichever
    # search decided it.
    #
    # **This rests on a Warp implementation detail, read from ``native/mesh.h`` (Warp 1.17)**: the
    # leaf test skips a sliver when ``|n| / sum(|e|^2) < 1e-6``, which for a collapsed triangle is
    # ``0 / 0`` -- NaN, which compares false, so the triangle is *tested* rather than culled.
    # ``tests/test_neighbors.py::test_mesh_from_points_answers_the_nearest_point`` fails if a
    # release ever culls it.
    index = wp.int32(-1)
    distance = wp.float32(FLOAT32_INF_CONSTANT)
    hit = wp.mesh_query_point_no_sign(mesh_id, q, max_radius)
    if hit.result:
        d = wp.length(points[hit.face] - q)
        if d <= max_radius:
            index = hit.face
            distance = d
    return index, distance


@wp.kernel
def nearest_point_via_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    max_radius: wp.float32,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    # The ``k = 1`` rows the grid deferred; ``query_nearest_via_mesh`` answers every row instead.
    tid = wp.int32(wp.tid())
    if out_indices[tid, 0] != DEFERRED_ROW:
        return
    index, distance = mesh_nearest_point(mesh_id, points, queries[tid], max_radius)
    out_indices[tid, 0] = index
    out_distances[tid, 0] = distance


@wp.kernel
def query_nearest_via_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    max_radius: wp.float32,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    # ``query_nearest``'s ``k = 1`` BVH-backend search, every row through the collapsed-triangle
    # mesh. Differs from ``nearest_point_via_mesh`` only in answering every row rather than the
    # grid's deferred ones, which saves a ``DEFERRED_ROW`` pre-fill of the output.
    tid = wp.int32(wp.tid())
    index, distance = mesh_nearest_point(mesh_id, points, queries[tid], max_radius)
    out_indices[tid, 0] = index
    out_distances[tid, 0] = distance


@wp.kernel
def query_weighted_nearest_neighbors(
    points: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    max_weight: wp.float32,
    initial_radius: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    # Nearest site under the *weighted* distance ``|p - q| - w(p)``, i.e. the Apollonius / power
    # nearest neighbour. One site, so no candidate row and no register bucket: the whole k-NN row
    # machinery above collapses to two scalars here.
    #
    # ``max_weight`` is what makes the search prunable, and it is the only thing that does. A site
    # outside the ball of radius ``r`` has ``|p - q| > r``, so its score exceeds
    # ``r - max_weight``; a best score at or below that bound is therefore certified, and the
    # deepening target is ``best + max_weight`` -- which is always past the current ``r``, since
    # failing the test means ``best + max_weight > r``. That is exactly ``deepen_radius``'s contract
    # with the bound shifted, so the helper is shared with the k-NN kernels rather than re-derived.
    #
    # The test is spelled ``best + max_weight <= r`` and **not** the algebraically identical
    # ``best <= r - max_weight``, because the deepening step sets ``r`` to ``best + max_weight``: in
    # float32 the subtracted form can then fail against the very radius it just asked for, when
    # ``best`` is large next to ``max_weight`` and the addition rounds. Measured with the queries
    # several spacings off the surface, the subtracted form stalled at a fixed radius for the whole
    # attempt budget on a fraction of a percent of queries, each of which then paid the
    # forced-complete scan over the entire cloud. Computing the same expression on both sides makes
    # the loop exact, and it is what keeps the worst case bounded.
    tid = wp.int32(wp.tid())
    q = queries[tid]

    r_hard, r = search_radius_bounds(q, min_bound, max_bound, FLOAT32_INF_CONSTANT, initial_radius)
    best = FLOAT32_INF_CONSTANT
    best_index = wp.int32(-1)
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        r = attempt_radius(attempt, r, r_hard)
        query = wp.bvh_query_sphere(bvh_id, q, r)
        point_index = wp.int32(0)
        while wp.bvh_query_next(query, point_index):
            score = wp.length(points[point_index] - q) - weights[point_index]
            if score < best:
                best = score
                best_index = point_index
        r = next_search_radius(best + max_weight, r, r_hard)
        if r < 0.0:
            break  # certified exact, or the scan was already complete

    out_indices[tid] = best_index
    out_distances[tid] = best


# The seed a block's running minimum starts from in ``nearest_key_argmin``: above every real key,
# since ``pack_nearest_key`` is non-negative for any non-negative distance.
_NO_KEY = wp.constant(wp.int64(INT64_MAX))


@wp.kernel
def nearest_key_argmin(distances: wp.array[wp.float32], out_result: wp.array[wp.int64]) -> None:
    # The smallest ``pack_nearest_key(distances[i], i)`` over ``i``: which element carries the
    # smallest distance, lowest index on a tie, with that distance's own float32 bits in the high
    # half. ``out_result[0]`` must arrive seeded at ``_NO_KEY``; slot 1 belongs to
    # ``nearest_key_partner``.
    #
    # The one kernel two "global argmin plus its partner" answers reduce through --
    # ``neighbors.closest_pair`` over column 1 of a ``k=2`` self-query, and
    # ``proximity.mesh_to_mesh_distance`` over its per-face squared distances. Each used to write a
    # whole ``int64`` key array for ``reduce.min`` to read back; here the key is packed where it is
    # compared and never stored, and the result stays on the device for ``nearest_key_partner``, so
    # the pair costs one readback between them rather than two or three.
    #
    # ``distances`` may be a strided column view; it is only ever indexed. Launched
    # ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: lanes stride the block's own
    # ``ITEMS_PER_BLOCK_1D`` chunk by ``wp.block_dim()``, so on the CPU device, where that reads 1,
    # the single lane covers the chunk and the one-element tile holds its true minimum.
    chunk, lane = wp.tid()
    offset, count = block_chunk_1d(distances.shape[0], chunk)
    if count <= 0:
        return
    best = _NO_KEY
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        best = wp.min(best, kernel_array.pack_nearest_key(distances[i], i))
    block_best = block_min(best)
    if lane == 0:
        wp.atomic_min(out_result, 0, block_best)


@wp.kernel
def nearest_key_partner(
    result: wp.array[wp.int64], partners: wp.array[wp.int32], out_result: wp.array[wp.int64]
) -> None:
    # ``out_result[1] = partners[winner]``, the winner being the low half of ``result[0]`` --
    # ``nearest_key_argmin``'s answer. ``dim=1``. ``result`` and ``out_result`` are the *same*
    # two-slot buffer at every call site, passed twice so the kernel reads slot 0 and writes slot 1
    # of it: that is what lets the key and its partner come back in one readback. The single thread
    # reads before it writes, and the two touch different slots.
    winner = wp.int32(result[0] & wp.int64(0xFFFFFFFF))
    out_result[1] = wp.int64(partners[winner])


@wp.kernel
def geodesic_ball_reference_neighbors(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_reference: wp.array[wp.int32],
) -> None:
    """Lowest-indexed edge neighbor per vertex (libigl ``adjacency_list[i][0]``); self if alone."""
    i = wp.int32(wp.tid())
    start = adj_offsets[i]
    end = adj_offsets[i + 1]
    if start == end:
        out_reference[i] = i
        return
    minimum = adj_columns[start]
    for k in range(start + 1, end):
        if adj_columns[k] < minimum:
            minimum = adj_columns[k]
    out_reference[i] = minimum


@wp.kernel(enable_backward=False)
def query_geodesic_ball_collect(
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    chunk_start: wp.int32,
    queue_pool: wp.array2d[wp.int32],
    visited_pool: wp.array2d[wp.int32],
    ext_dist_pool: wp.array2d[wp.float32],
    ext_idx_pool: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    # Scratch lives in wrapper-allocated global-memory pools (one row per thread of the current
    # chunk) instead of kilobytes of per-thread local arrays; the wrapper pre-fills the visited pool
    # with -1 before each launch. Single pass: after this kernel the thread's queue row holds
    # the collected set (``queue_pool[t][:out_counts[chunk_start + t]]``) ready to gather.
    t = wp.int32(wp.tid())
    i = chunk_start + t
    out_counts[i] = kernel_bfs.per_source_bfs_collect(
        i,
        vertices,
        adj_offsets,
        adj_columns,
        radius,
        min_count,
        queue_pool[t],
        visited_pool[t],
        ext_dist_pool[t],
        ext_idx_pool[t],
        out_overflow,
    )


@wp.kernel
def gather_queue_rows(
    queue_pool: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],
    local_offsets: wp.array[wp.int32],
    chunk_start: wp.int32,
    out_flat: wp.array[wp.int32],
) -> None:
    # Compact the chunk's queue rows into its flat CSR buffer. Adjacent j threads read one
    # queue row and write one out_flat segment contiguously (coalesced on both sides).
    # ``counts`` is the global per-source array (indexed at chunk_start + t); ``local_offsets``
    # is the chunk-local exclusive scan of this chunk's counts.
    t, j = wp.tid()
    if j >= counts[chunk_start + t]:
        return
    out_flat[local_offsets[t] + j] = queue_pool[t, j]


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
            (aabb_count_in_bounds, (wp.uint64(1), dense(wp.vec3), dense(wp.vec3)), wp.int32),
            (aabb_count_in_bounds, (wp.uint64(1), single(wp.vec3), single(wp.vec3)), wp.int32),
            (ball_count_in_bounds, (wp.uint64(1), dense(wp.vec3), wp.float32(1)), wp.int32),
            (ball_count_in_bounds, (wp.uint64(1), single(wp.vec3), wp.float32(1)), wp.int32),
        ]
    )


_declare_map_kernels()
