import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT, TOLERANCE_MERGE_CONSTANT, TWO_PI
from ordito.kernels import triangles as kernel_triangles
from ordito.kernels.array import RegisterBlockedTable, lift_vec2, morton_code_30, to_vec3, to_vec3d
from ordito.kernels.neighbors import (
    MAX_SEARCH_ATTEMPTS,
    attempt_radius,
    next_search_radius,
    search_radius_bounds,
)
from ordito.kernels.predicates import (
    barycentric_2d,
    closest_point_on_segment,
    is_in_aabb,
    is_strictly_inside_aabb,
    triangle_aabb,
    triangle_triangle_distance_sq,
)
from ordito.kernels.reduce import block_argmin, block_sum

wp.set_module_options({"enable_backward": False})

# ``face_to_mesh_distance`` / ``_tiled`` publish each thread's own best distance into
# ``global_best_sq`` so other threads can prune against it (see the wrapper's seeding comment for
# why this needs a bound rather than a tight one). Publishing the *exact* value has the same
# bound-is-the-answer failure the wrapper's seed epsilon already guards against, one level down and
# across threads rather than across the two passes: whenever two distinct query faces genuinely tie
# for the global minimum -- the documented common case, e.g. every face around the vertex realising
# the closest approach -- and both have a tight (corner-touching) AABB against the target, the first
# thread to publish the exact minimum prunes the *other* tied thread's own winning candidate before
# it reaches the leaf test, leaving that thread's ``out_distance_sq`` at ``inf``. The reported
# distance is still correct (the surviving thread found it too), but the documented "``face_a`` is
# the lowest index on a tie" guarantee silently depends on scheduling order. The same relative
# margin the wrapper's seed uses keeps every publication just loose enough that a tied thread's own
# tight bound is never pruned by another thread's, at a prune strength cost too small to measure.
_GLOBAL_BEST_RELAX = wp.float32(1.0 + 1e-4)

# ...and a floor under that publication, for the one distance the relaxation cannot loosen: ``0``.
# The prune skips a candidate whose box gap is ``>=`` the limit, so once some thread publishes an
# exact zero every other thread's zero-gap candidates -- the crossings of every other face that
# touches the target -- are skipped, and which face comes back as ``face_a`` is whichever thread
# published first rather than the lowest index. The wrapper floors its seed at the same value for
# the same reason. Measured before the floor: CUDA returned a different ``face_a`` than the cpu
# device on overlapping pairs, and not always the same one twice.
_MIN_POSITIVE_FLOAT32 = wp.float32(1.1754943508222875e-38)

# ``mesh_to_mesh_distance``'s three running scalars, one ``float32`` buffer: the published global
# best squared distance every walker prunes against (slot 0, which ``update_nearest_face_pair``
# reads), the brute-force corner seed that caps the sampled bound queries, and the sampled upper
# bound itself, which grows every query face's broad-phase box. The bound stays on the device, so
# the walk is issued behind the sampling with no readback in between.
BEST_SQ_SLOT = wp.constant(wp.int32(0))
SEED_SQ_SLOT = wp.constant(wp.int32(1))
UPPER_BOUND_SLOT = wp.constant(wp.int32(2))


@wp.func
def publish_best_sq(global_best_sq: wp.array[wp.float32], distance_sq: wp.float32) -> None:
    # Publish a real squared distance between the surfaces into the running prune limit, relaxed by
    # ``_GLOBAL_BEST_RELAX`` and floored at ``_MIN_POSITIVE_FLOAT32`` for the reasons above. The
    # map is monotone in ``distance_sq``, so the minimum of the publications is the publication of
    # the minimum: the sampled bound phase seeds the limit with this rule too, one point at a time.
    wp.atomic_min(
        global_best_sq,
        BEST_SQ_SLOT,
        wp.max(distance_sq * _GLOBAL_BEST_RELAX, _MIN_POSITIVE_FLOAT32),
    )


@wp.func
def closest_point_query(
    mesh_id: wp.uint64, p: wp.vec3, max_dist: wp.float32
) -> tuple[wp.vec3, wp.float32, wp.int32]:
    # One point's closest point on the mesh, its distance and the face carrying it; a miss returns
    # the query point, ``max_dist`` and ``-1``, so the sentinel convention lives in one place.
    #
    # Named rather than left inline in the kernel below because ``registration``'s ICP loop fuses
    # this query with the passes that consume its answer, and a cross-reference in prose is a
    # claim only a shared function can keep true.
    query = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        return closest, wp.length(p - closest), query.face
    return p, max_dist, wp.int32(-1)


@wp.kernel
def closest_point_on_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_face: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    closest, distance, face = closest_point_query(mesh_id, points[tid], max_dist)
    out_closest[tid] = closest
    out_distance[tid] = distance
    out_face[tid] = face


@wp.kernel
def closest_point_on_edges(
    vertices: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    max_dist: wp.float32,
    initial_radius: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_edge: wp.array[wp.int32],
) -> None:
    # The wireframe counterpart of ``closest_point_on_mesh``, and the reason it is a hand-written
    # traversal rather than a ``wp.mesh_query_point_no_sign`` over degenerate triangles: that query
    # **rejects** a zero-area triangle outright, on both devices, so that shortcut answers nothing
    # at all rather than answering approximately.
    #
    # 1.17's ``wp.mesh_query_sphere`` *does* handle them -- it falls back to a closest-point-on-
    # longest-edge test. It is still not the shortcut, for two reasons: it answers "which faces meet
    # this ball", not "which point is nearest", so the deepening loop and the
    # ``closest_point_on_segment`` narrow phase below both stay; and reaching it would mean carrying
    # a ``wp.Mesh`` of degenerate triangles in place of the ``wp.Bvh`` over edge bounds, which is
    # the same broad phase through a heavier object. What 1.17 did buy this kernel is the sphere
    # query on the BVH it already has, below.
    #
    # Iterative deepening, sharing ``search_radius_bounds`` / ``attempt_radius`` /
    # ``next_search_radius`` with the k-NN kernels next door: a scan of the **ball** of radius
    # ``r`` about ``q`` enumerates every edge whose *closest point* is within ``r`` -- that point is
    # then inside the ball, so the edge's AABB contains it and therefore overlaps the ball -- which
    # is what makes ``best <= r`` a proof
    # of exactness rather than a heuristic. The enumeration was the bounding cube until Warp 1.17
    # supplied ``wp.bvh_query_sphere``; the proof above is the same either way, and the ball is 6/pi
    # ~ 1.91x less volume to walk. The cube was re-timed after Warp 1.18 slowed the sphere walk: it
    # wins only at the largest meshes (1.04-1.09x at ``happy_buddha`` / ``lucy``), ties at
    # ``dragon`` and loses 0.76x at ``bunny_decimated``, too small and too mixed for a size gate.
    # Unlike the point BVH next door this still needs its narrow phase,
    # because an edge's bounds are not degenerate -- a sphere may overlap the AABB of an edge whose
    # closest point lies outside it.
    tid = wp.int32(wp.tid())
    q = queries[tid]

    r_hard, r = search_radius_bounds(q, min_bound, max_bound, max_dist, initial_radius)
    best_distance = FLOAT32_INF_CONSTANT
    best_edge = wp.int32(-1)
    best_point = q
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        r = attempt_radius(attempt, r, r_hard)
        query = wp.bvh_query_sphere(bvh_id, q, r)
        edge_index = wp.int32(0)
        while wp.bvh_query_next(query, edge_index):
            candidate = closest_point_on_segment(
                vertices[edges[edge_index, 0]], vertices[edges[edge_index, 1]], q
            )
            d = wp.length(candidate - q)
            # Acceptance is ``d <= max_dist``; ``r`` bounds only the enumeration.
            if d < best_distance and d <= max_dist:
                best_distance = d
                best_edge = edge_index
                best_point = candidate
        r = next_search_radius(best_distance, r, r_hard)
        if r < 0.0:
            break  # certified exact, or the scan was already complete

    if best_edge < 0:
        # Miss convention copied from ``closest_point_on_mesh`` above, so the two agree.
        out_closest[tid] = q
        out_distance[tid] = max_dist
        out_edge[tid] = wp.int32(-1)
    else:
        out_closest[tid] = best_point
        out_distance[tid] = best_distance
        out_edge[tid] = best_edge


@wp.func
def aabb_distance_sq(
    a_lower: wp.vec3, a_upper: wp.vec3, b_lower: wp.vec3, b_upper: wp.vec3
) -> wp.float32:
    # Squared distance between two axis-aligned boxes: per axis, the gap between them or zero when
    # they overlap. A lower bound on the distance between anything inside them, which is what makes
    # it a sound prune.
    gap = wp.max(wp.max(a_lower - b_upper, b_lower - a_upper), wp.vec3(0.0, 0.0, 0.0))
    return wp.length_sq(gap)


@wp.func
def face_pair_distance_sq(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    lower: wp.vec3,
    upper: wp.vec3,
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    target_lower: wp.array[wp.vec3],
    target_upper: wp.array[wp.vec3],
    candidate: wp.int32,
    limit: wp.float32,
) -> wp.float32:
    # One broad-phase candidate, tested: the exact triangle-triangle distance, or ``inf`` when the
    # box gap alone already rules the pair out. A box-to-box gap is a lower bound on the triangle
    # distance, so a candidate whose boxes are farther apart than ``limit`` cannot win and never
    # reaches the fifteen-case leaf test.
    #
    # Shared by the two kernels below, which differ only in *who walks the candidates*: a thread
    # each in ``face_to_mesh_distance``, a whole block in ``face_to_mesh_distance_tiled``.
    if aabb_distance_sq(lower, upper, target_lower[candidate], target_upper[candidate]) >= limit:
        return FLOAT32_INF_CONSTANT
    b0, b1, b2 = kernel_triangles.face_vertices(target_vertices, target_faces, candidate)
    return triangle_triangle_distance_sq(a0, a1, a2, b0, b1, b2)


@wp.func
def query_face_broad_phase(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32, upper_bound: wp.float32
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    # One query face's three corners, its own AABB, and that box grown by ``upper_bound`` -- which
    # is the box handed to the traversal, and the reason the ungrown one is returned beside it:
    # ``face_pair_distance_sq`` prunes on the *ungrown* gap, so a caller needs both.
    #
    # Shared by the two kernels below, which differ only in who walks the candidates.
    a0, a1, a2 = kernel_triangles.face_vertices(vertices, faces, f)
    lower, upper = triangle_aabb(a0, a1, a2)
    margin = wp.vec3(upper_bound, upper_bound, upper_bound)
    return a0, a1, a2, lower, upper, lower - margin, upper + margin


@wp.func
def update_nearest_face_pair(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    lower: wp.vec3,
    upper: wp.vec3,
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    target_lower: wp.array[wp.vec3],
    target_upper: wp.array[wp.vec3],
    candidate: wp.int32,
    best: wp.float32,
    witness: wp.int32,
    global_best_sq: wp.array[wp.float32],
) -> tuple[wp.float32, wp.int32]:
    # One candidate, tested and folded into the walker's running best: the new ``(best, witness)``,
    # and the publication into ``global_best_sq`` that lets every other walker prune against it.
    #
    # This is the *decision rule*, not just the arithmetic, and that is why it is named. The two
    # kernels below wrote it out identically -- the prune limit is ``wp.min(local, global)``, the
    # update is strictly ``<`` so the lowest-index candidate wins a tie, and the publication is
    # relaxed by ``_GLOBAL_BEST_RELAX`` for the reason this module's header documents. A copy of a
    # three-part rule like that is a copy that drifts, and the module header's argument for the
    # relaxation is a claim only a shared function can keep true for both walkers.
    distance_sq = face_pair_distance_sq(
        a0,
        a1,
        a2,
        lower,
        upper,
        target_vertices,
        target_faces,
        target_lower,
        target_upper,
        candidate,
        wp.min(best, global_best_sq[BEST_SQ_SLOT]),
    )
    if distance_sq < best:
        publish_best_sq(global_best_sq, distance_sq)
        return distance_sq, candidate
    return best, witness


# Relative slack on the seeded ``max_dist`` of the sampled closest-point queries below. The seed is
# a corner-to-corner distance, so the sample point it was measured from has its own closest point
# *within* it -- but at the boundary, and ``mesh_query_point_no_sign`` reports distances a few
# times 1e-5 off an exact oracle. The slack keeps that point a hit.
_SEED_RELAX = wp.float32(1.0 + 1e-4)

# Slices the target sample is split into by ``sampled_corner_gap_sq``'s second grid dimension.
SEED_SLICES = 32


@wp.func
def sampled_corner(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], stride: wp.int32, k: wp.int32
) -> wp.vec3:
    # The ``k``-th point of a strided face sample: the first corner of face ``k * stride``. A face
    # corner rather than a vertex, because a vertex no face references is not a point of the
    # surface, and the upper bound these points feed has to be a distance between surfaces.
    return vertices[faces[3 * k * stride]]


@wp.kernel
def sampled_corner_gap_sq(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    query_stride: wp.int32,
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    target_stride: wp.int32,
    n_target_samples: wp.int32,
    out_bounds: wp.array[wp.float32],
) -> None:
    # Brute force over two small corner samples: the smallest squared distance between a sample
    # point of the query surface and one of the target surface, into ``out_bounds[SEED_SQ_SLOT]``.
    # Both are surface points, so it is an upper bound on the surfaces' distance -- a loose one,
    # but it is only the ``max_dist`` that lets ``sampled_corner_distance_min`` stop a far query
    # early instead of walking the BVH.
    #
    # ``dim=(slices, query samples)``: the query sample is the lane index, so every lane of a warp
    # reads the same target sample at each step and the load broadcasts; the slices split the
    # target samples so the launch is not a thousand threads each walking a thousand points. The
    # minimum is order-free, so the result does not depend on the split.
    s, i = wp.tid()
    p = sampled_corner(query_vertices, query_faces, query_stride, i)
    best = FLOAT32_INF_CONSTANT
    for j in range(s, n_target_samples, SEED_SLICES):
        q = sampled_corner(target_vertices, target_faces, target_stride, j)
        best = wp.min(best, wp.length_sq(p - q))
    if best < out_bounds[SEED_SQ_SLOT]:
        wp.atomic_min(out_bounds, SEED_SQ_SLOT, best)


@wp.kernel
def sampled_corner_distance_min(
    target_mesh: wp.uint64,
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    query_stride: wp.int32,
    out_bounds: wp.array[wp.float32],
) -> None:
    # The smallest distance from a sample point of the query surface to the target -- the upper
    # bound the broad phase is grown by, into ``out_bounds[UPPER_BOUND_SLOT]`` -- and the same
    # distance published into the walk's prune limit (``publish_best_sq``), so the walk starts
    # from it with no host round trip. ``max_dist`` is ``sampled_corner_gap_sq``'s seed, so a point
    # farther than it is a miss that stops near the top of the BVH and returns ``max_dist``, which
    # is still a valid upper bound; and the seed's own query point is a sample here too, so the
    # minimum never rests on a miss.
    #
    # The ``<`` read before the atomics keeps the single-slot contention to the points that improve
    # it; ``atomic_min`` is exact, so the minimum is the one a reduction would return.
    p = sampled_corner(query_vertices, query_faces, query_stride, wp.int32(wp.tid()))
    _closest, distance, _face = closest_point_query(
        target_mesh, p, wp.sqrt(out_bounds[SEED_SQ_SLOT]) * _SEED_RELAX
    )
    if distance < out_bounds[UPPER_BOUND_SLOT]:
        wp.atomic_min(out_bounds, UPPER_BOUND_SLOT, distance)
        publish_best_sq(out_bounds, distance * distance)


@wp.kernel
def face_to_mesh_distance(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    target_lower: wp.array[wp.vec3],
    target_upper: wp.array[wp.vec3],
    target_mesh: wp.uint64,
    candidate_cap: wp.int32,
    global_best_sq: wp.array[wp.float32],
    out_distance_sq: wp.array[wp.float32],
    out_witness: wp.array[wp.int32],
    counter: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
) -> None:
    # One thread per face of the query mesh: expand its own AABB by the upper bound
    # (``global_best_sq[UPPER_BOUND_SLOT]``, the same buffer's third slot) and test every
    # target face whose AABB it then meets. That bound is what makes the broad phase sound -- the
    # true minimum is at most the bound, so the pair achieving it has AABBs within that
    # distance and cannot be missed.
    #
    # Two prunes stand between a candidate and the fifteen-case leaf test, both inside
    # ``face_pair_distance_sq``'s ``limit``, and both matter because for *well-separated* meshes the
    # bound is roughly the answer, so every face's grown box meets a large part of the other mesh.
    # The first is local and exact: this thread's own best. The second reads a **global** running
    # minimum other threads have published -- which makes the amount of work nondeterministic but
    # not the answer, since it only ever skips pairs that cannot beat a distance already achieved.
    #
    # **``candidate_cap`` is what makes this the first of two passes.** The traversal is wildly
    # unbalanced: the overwhelming majority of query faces have no candidate at all, and a fraction
    # of a percent carry half of the candidate tests. So a thread that is still going after
    # ``candidate_cap`` candidates stops, appends its face to ``overflow``, and lets
    # ``face_to_mesh_distance_tiled`` re-walk it with a whole block. Pass a cap of ``INT32_MAX`` to
    # disable the split and settle every face here, which is what the CPU device does --
    # ``wp.launch_tiled`` runs one lane per block there. ``wp.mesh_get_bvh`` (Warp 1.17) hands back
    # the ``wp.Mesh``'s *own* BVH over its faces, so the caller builds no second structure.
    target_bvh = wp.mesh_get_bvh(target_mesh)
    f = wp.int32(wp.tid())
    a0, a1, a2, lower, upper, grown_lower, grown_upper = query_face_broad_phase(
        query_vertices, query_faces, f, global_best_sq[UPPER_BOUND_SLOT]
    )

    best = FLOAT32_INF_CONSTANT
    witness = wp.int32(-1)
    seen = wp.int32(0)
    overflowed = wp.bool(False)
    query = wp.bvh_query_aabb(target_bvh, grown_lower, grown_upper)
    candidate = wp.int32(0)
    while wp.bvh_query_next(query, candidate):
        seen += 1
        if seen > candidate_cap:
            overflowed = wp.bool(True)
            break
        best, witness = update_nearest_face_pair(
            a0,
            a1,
            a2,
            lower,
            upper,
            target_vertices,
            target_faces,
            target_lower,
            target_upper,
            candidate,
            best,
            witness,
            global_best_sq,
        )
    out_distance_sq[f] = best
    out_witness[f] = witness
    if overflowed:
        # The face is *not* settled: whatever it wrote above is a partial answer over the first
        # ``candidate_cap`` candidates, and the tiled pass overwrites both entries. Publishing it
        # anyway is what keeps the global minimum tight while that pass runs.
        overflow[wp.atomic_add(counter, 0, 1)] = f


@wp.kernel(enable_backward=False)
def face_to_mesh_distance_tiled(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    target_lower: wp.array[wp.vec3],
    target_upper: wp.array[wp.vec3],
    target_mesh: wp.uint64,
    overflow: wp.array[wp.int32],
    global_best_sq: wp.array[wp.float32],
    out_distance_sq: wp.array[wp.float32],
    out_witness: wp.array[wp.int32],
) -> None:
    # **One block per straggler face**, re-walking the query the thread pass gave up on with
    # ``wp.tile_bvh_query_aabb``, which hands one candidate per lane per step. Same candidate set,
    # same ``face_pair_distance_sq`` test; only the walk's *depth* changes, from one thread's
    # thousands of sequential steps to that over the block width. This is the identical trick
    # ``kernels/algorithms/ball_pivoting.py`` uses on its pivot search, and for the identical
    # reason -- see its comment for why a *serial* BVH walk is not the win.
    #
    # Each lane keeps its own running best, so a lane cannot prune against its siblings' minima and
    # slightly more candidates reach the leaf test. The answer is unchanged: the minimum over the
    # block is the minimum of the per-lane minima, and the global atomic is still read every step.
    # ``wp.mesh_get_bvh`` (Warp 1.17) hands back the ``wp.Mesh``'s *own* BVH over its faces, so the
    # caller builds no second structure -- see the wrapper for the measured share.
    target_bvh = wp.mesh_get_bvh(target_mesh)
    slot = wp.int32(wp.tid())
    f = overflow[slot]
    n_target_faces = target_faces.shape[0] // 3
    a0, a1, a2, lower, upper, grown_lower, grown_upper = query_face_broad_phase(
        query_vertices, query_faces, f, global_best_sq[UPPER_BOUND_SLOT]
    )

    best = FLOAT32_INF_CONSTANT
    witness = wp.int32(-1)
    query = wp.tile_bvh_query_aabb(target_bvh, grown_lower, grown_upper)
    while wp.tile_query_valid(query):
        candidate = wp.untile(wp.tile_bvh_query_next(query))
        # A lane with no candidate this step gets -1; the tile is block-wide, so it cannot simply
        # leave the loop.
        #
        # The **upper** half of that test is not defensive: ``wp.tile_bvh_query_aabb`` hands back
        # out-of-range indices on a query whose traversal round finds more primitives than its
        # internal buffer holds, and without this line they are dereferenced. See
        # ``kernels/algorithms/ball_pivoting.py::pivot_front_edges`` for the diagnosis
        # and the read of Warp's own source; the short version is that
        # ``tile_bvh.h`` counts results with an unconditional ``atomicAdd`` and guards only the
        # *write* against a ``block_dim * 5`` capacity, so once a round overruns it the consumer
        # reads uninitialised shared memory as a primitive index. ``compute-sanitizer`` names this
        # kernel and this load, reading wildly out of range, on a large straggler set.
        if candidate >= 0 and candidate < n_target_faces:
            best, witness = update_nearest_face_pair(
                a0,
                a1,
                a2,
                lower,
                upper,
                target_vertices,
                target_faces,
                target_lower,
                target_upper,
                candidate,
                best,
                witness,
                global_best_sq,
            )
    # The witness must not depend on which lane happened to see it, which is what
    # ``block_argmin``'s second stage is for. When no lane found a candidate every lane still holds
    # ``(inf, -1)``, so it returns -1 and no fixup is needed here.
    block_best, block_witness = block_argmin(best, witness)
    # **``<``, not an overwrite, and that is a correctness fix rather than a tidy-up.** What the
    # grid pass left in ``out_distance_sq[f]`` is a *partial* answer -- the best over its first
    # ``candidate_cap`` candidates -- but it is a real distance between two real triangles, so the
    # smaller of the two is always the better answer and never a wrong one.
    #
    # Overwriting loses it, and loses it in exactly the case that matters.
    # ``face_pair_distance_sq`` skips a candidate whose box gap is ``>=`` its limit, and the limit
    # here is the *running global minimum* -- which, by the time this pass runs, is frequently the
    # answer itself, published by this very face in the grid pass. Its re-walk then prunes every
    # candidate including the pair that achieved it, comes back ``inf``, and overwrites the right
    # answer with it: two close parallel sheets returned ``inf`` on CUDA at every size where every
    # face overflows the cap, while the cpu device, which runs no second pass at all, returned the
    # right distance. This is the identical bound-is-the-answer trap the ``global_best_sq`` seeding
    # in the wrapper documents, one level down: there the fix is a relative bump on the seed, here
    # it is keeping what the first pass already found.
    #
    # It is also what makes this pass **safe against the ``wp.tile_bvh_query_aabb`` result-buffer
    # overrun** the guard above can only half-fix: a round that silently
    # dropped primitives now leaves the grid pass's answer standing instead of replacing it with a
    # worse one, so an overrun can cost accuracy but can no longer cost correctness outright.
    if block_best < out_distance_sq[f]:
        out_distance_sq[f] = block_best
        out_witness[f] = block_witness


@wp.func
def closest_face_or_first(mesh_id: wp.uint64, p: wp.vec3, max_dist: wp.float32) -> wp.int32:
    # The face closest to ``p``, or face 0 on a miss -- the wrapper's documented convention, which
    # also keeps a table read in range. Shared by the two ``normals_at_closest_faces*`` kernels.
    #
    # Only the face index is wanted, so this stops at the query -- no ``mesh_eval_position`` and no
    # distance.
    query = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
    return wp.where(query.result, query.face, wp.int32(0))


@wp.kernel
def normals_at_closest_faces(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    face_normals: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # The normal of the face closest to each point, gathered from a caller's table in the same
    # thread that found the face. ``normals_at_closest_faces_computed`` is the no-table variant.
    tid = wp.int32(wp.tid())
    out_normals[tid] = face_normals[closest_face_or_first(mesh_id, points[tid], max_dist)]


@wp.kernel
def normals_at_closest_faces_computed(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
) -> None:
    # ``normals_at_closest_faces`` with the hit face's normal formed from its corners in the thread,
    # by the same ``face_normals_and_area`` the per-face table is built with, so the value is the
    # table's entry. The two kernels differ only in where the normal comes from: a whole-mesh table
    # costs a launch and two face-sized buffers the queries read a few entries of.
    tid = wp.int32(wp.tid())
    face = closest_face_or_first(mesh_id, points[tid], max_dist)
    normal, _area = kernel_triangles.face_normals_and_area(vertices, faces, face)
    out_normals[tid] = normal


@wp.func
def signed_distance_from_query(
    mesh_id: wp.uint64,
    p: wp.vec3,
    result: wp.bool,
    face: wp.int32,
    u: wp.float32,
    v: wp.float32,
    sign: wp.float32,
    max_dist: wp.float32,
) -> wp.float32:
    # Closest-point evaluation and signing shared by both signed-distance kernels: a miss reports
    # the cutoff, a point inside the merge tolerance of the surface stays positive (the sign is not
    # meaningful there), and everything else takes the sign the query returned.
    #
    # The query struct itself cannot be the parameter -- ``mesh_query_point_sign_parity`` and
    # ``mesh_query_point_sign_winding_number`` return different types -- so the fields the tail
    # reads are passed individually. The two *heads* stay separate kernels on purpose: their
    # builtins take different parameters and the winding one carries a precondition its caller
    # chose, which a runtime selector would hide.
    if not result:
        return max_dist
    closest = wp.mesh_eval_position(mesh_id, face, u, v)
    dist = wp.length(p - closest)
    if dist <= TOLERANCE_MERGE_CONSTANT:
        return dist
    return sign * dist


@wp.kernel
def signed_distance_on_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
    out_distance: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    p = points[tid]
    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    out_distance[tid] = signed_distance_from_query(
        mesh_id, p, query.result, query.face, query.u, query.v, query.sign, max_dist
    )


@wp.kernel
def signed_distance_on_mesh_winding(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    accuracy: wp.float32,
    winding_threshold: wp.float32,
    out_distance: wp.array[wp.float32],
) -> None:
    # Only the sign differs from ``signed_distance_on_mesh``; the closest-point and tolerance-band
    # handling is the shared ``signed_distance_from_query``. ``mesh_id`` MUST come from a
    # ``wp.Mesh`` built with ``support_winding_number=True`` -- otherwise this builtin silently
    # falls back to ray parity (warp/native/mesh.h:1348).
    tid = wp.int32(wp.tid())
    p = points[tid]
    query = wp.mesh_query_point_sign_winding_number(
        mesh_id, p, max_dist, accuracy, winding_threshold
    )
    out_distance[tid] = signed_distance_from_query(
        mesh_id, p, query.result, query.face, query.u, query.v, query.sign, max_dist
    )


@wp.func
def solid_angle_terms(
    v0: wp.vec3, vl0: wp.float32, b: wp.vec3, c: wp.vec3, p: wp.vec3
) -> tuple[wp.float32, wp.float32]:
    """
    ``(y, x)`` with ``atan2(y, x)`` half the solid angle of triangle ``(p + v0, b, c)`` at ``p``.

    The first corner arrives already relative to ``p``, with its length, so a fan of triangles
    sharing that corner (``winding_number_tree``'s caps) computes it once per fan.
    """
    v1 = b - p
    v2 = c - p
    vl1 = wp.length(v1)
    vl2 = wp.length(v2)
    # det([v0; v1; v2]) as the scalar triple product — cheaper than materializing the matrix.
    detf = wp.dot(v0, wp.cross(v1, v2))
    dp0 = wp.dot(v1, v2)
    dp1 = wp.dot(v2, v0)
    dp2 = wp.dot(v0, v1)
    denom = vl0 * vl1 * vl2 + dp0 * vl0 + dp1 * vl1 + dp2 * vl2
    return detf, denom


@wp.func
def solid_angle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.float32:
    """Signed solid angle subtended by triangle (a, b, c) at point p (``igl::solid_angle``)."""
    v0 = a - p
    detf, denom = solid_angle_terms(v0, wp.length(v0), b, c, p)
    return wp.atan2(detf, denom) / TWO_PI


@wp.func
def solid_angle_at_face(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32, p: wp.vec3
) -> wp.float32:
    v0, v1, v2 = kernel_triangles.face_vertices(vertices, faces, f)
    return solid_angle(v0, v1, v2, p)


@wp.kernel
def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    q = wp.int32(wp.tid())
    p = query_points[q]
    w = wp.float32(0.0)
    for f in range(n_faces):
        w = w + solid_angle_at_face(vertices, faces, f, p)
    out_winding[q] = w


# Queries one ``winding_number_tiled`` thread may sum together; a launch takes the widest
# ``RegisterBlockedTable.launch_shape`` allows, as ``kernels/points.hull_support_extremes`` does.
WINDING_WIDTHS = (4, 2, 1)


def _winding_number_tiled_kernel(width: int) -> wp.Kernel:
    """Build ``winding_number_tiled`` over ``width`` queries per thread."""
    points_t = wp.types.matrix(shape=(width, 3), dtype=wp.float32)
    totals_t = wp.types.vector(length=width, dtype=wp.float32)

    def winding_number_tiled(
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        n_faces: wp.int32,
        n_slices: wp.int32,
        query_points: wp.array[wp.vec3],
        out_winding: wp.array[wp.float32],
    ) -> None:
        # One thread per (query block, face slice): each walks a strided slice of the face list for
        # ``width`` queries at once and commits one atomic per query. A face's three vertices are
        # gathered once for the whole block, which is what the walk is bound by; each query keeps
        # its own running sum in the same face order, so the per-thread totals are the one-query
        # form's bit for bit.
        #
        # Lane-free because the threads partition the **outer** work -- the face list -- rather than
        # a sequence one block owns, so there is no `wp.block_dim()` to stride by; on the CPU
        # device, where `wp.launch_tiled` runs one lane per block through Warp 1.18, that lane would
        # cover `1/block_dim` of the slice. See
        # `face_to_mesh_distance_tiled` above for the other side of the rule.
        #
        # **The block-per-query rewrite was measured here and declined.** It looked like the
        # strongest candidate in the tree -- the query dimension is already the outer one and the
        # walk covers every face -- and the gain evaporates as the grid fills: a real win on a small
        # mesh with few queries, and nothing at all once the query count is large. A gain that
        # shrinks with the input is a decline, and the reason is that this
        # grid is `n_queries x n_face_slices` and already wide; see
        # `kernels/points.py::hull_support_extremes` for the same trade measured to an outright
        # loss.
        #
        # Note this is *not* why `winding_number` above exists -- that is the public `tiled=False`
        # exact-sum reference, with its own benchmark group, and no conversion here would retire it.
        b, j = wp.tid()
        n_queries = query_points.shape[0]
        q0 = b * width
        # A slot past the last query repeats it; only the commit below skips it.
        p = points_t()
        for d in range(width):
            p[d] = query_points[wp.min(q0 + d, n_queries - 1)]
        total = totals_t()
        for face_idx in range(j, n_faces, n_slices):
            v0, v1, v2 = kernel_triangles.face_vertices(vertices, faces, face_idx)
            for d in range(width):
                total[d] = total[d] + solid_angle(v0, v1, v2, p[d])
        for d in range(width):
            if q0 + d < n_queries:
                wp.atomic_add(out_winding, q0 + d, total[d])

    return wp.kernel(winding_number_tiled, name=f"winding_number_tiled_{width}")


WINDING_NUMBER_TILED = RegisterBlockedTable(
    "winding_number_tiled", _winding_number_tiled_kernel, WINDING_WIDTHS
)


# ---------------------------------------------------------------------------------------------
# Exact hierarchical winding number (``proximity.winding_number`` above the size gate)
#
# Jacobson, Kavan and Sorkine-Hornung (2013): a patch ``S`` of the mesh and a fan ``C`` closing its
# boundary (one triangle ``(apex, a, b)`` per boundary halfedge ``a -> b``) form a closed 2-chain,
# whose winding number is an integer and is 0 at any point outside a convex region holding both.
# So outside that region ``w_S(q) = sum Omega(apex, a, b) / 4 pi`` over the boundary halfedges --
# exactly, with no expansion and no accuracy parameter -- and it costs the patch's *perimeter*
# rather than its area. The chain algebra holds for any soup: a halfedge whose partner is missing,
# runs the same way, or shares a non-manifold edge simply stays on the boundary (its fan triangle
# is still exact; a cancelling pair only saves work).
#
# The patches are the nodes of an implicit complete binary tree over the faces sorted by the Morton
# code of their centroid: heap numbering (root 1, children ``2k`` and ``2k + 1``), ``n_leaves`` a
# power of two, leaf ``j`` at node ``n_leaves + j`` holding sorted faces ``[j * L, (j + 1) * L)``,
# every node a contiguous face range. A node's region is its box intersected with four diagonal
# slabs (a 14-DOP), and its apex is the mean of its face corners, which lies inside both.
# A node keeps a cap only when its perimeter is below its face count; otherwise it is summed face
# by face, which is cheaper and equally exact. Leaves never keep one.
#
# A query descends only into nodes whose region contains it, so per query the work is the caps of
# the siblings along its path plus the leaves it lands in -- about the square root of the face
# count (2.1 k solid angles per query on a 16 k-face mesh, 17.5 k on 0.87 M faces, 53 k on 28 M,
# where the direct sum pays the face count). Measured:
#
# - **Per query it is compute-bound at the direct sum's own rate** (~2e11 solid angles a second),
#   so the gain is the work ratio. Reading caps as vertex-index pairs instead of stored positions
#   (a third of the memory) was 0.30x; accumulating the angles as a complex product to drop the
#   ``atan2`` was 0.81x.
# - **The 14-DOP test is 1.24-1.25x over the box alone**; leaf size is flat from 4 to 32 faces (32
#   stores the fewest cap edges); a 128-lane block is 1.23x over 32 lanes; Morton-sorting the
#   queries is 1.04-1.08x and not done. A Hilbert face order measured 1.08-1.23x on the query and is
#   left open.
# - **A Barnes-Hut far field on top of the caps buys nothing** (Barill et al. 2018, order 1 and 2,
#   beta 2-8): the siblings a query is outside of touch its own node, so they are never well
#   separated, and forcing more expansions makes a block walk thousands of nodes serially (3-30x
#   slower). Warp's own order-2 walk (``wp::mesh_query_winding_number``, reachable only through
#   ``wp.func_native``) needs its accuracy at 8 for a 1.8e-5 error and is slower than this at 6.
# - **One block per query cell of a lattice (lanes over queries, the walk testing the cell's box)
#   was 16x slower** than one block per query on ``offset_mesh``'s dragon lattice.


# Faces per leaf. The cost is flat from 4 to 32; 32 keeps the fewest caps (a leaf is summed face by
# face anyway, and a node of a few leaves rarely has a perimeter below its face count).
WINDING_TREE_LEAF_FACES = 32
# The size gate in ``proximity.winding_number``: the hierarchy once both ``n_queries`` and
# ``n_queries * n_faces`` reach these, the tiled direct sum below. The build costs about a face's
# worth of work per face plus a fixed few tenths of a millisecond on CUDA, the direct sum a fixed
# cost per pair, so the crossover is a pair count with a query floor for meshes large enough that
# the build alone outweighs a few queries. Swept build + walk against the tiled sum (2026-10-06):
# CUDA level at 2.7e8 pairs on 16 k faces (0.98x), 2.6x / 1.3x / 1.5x at 2.2-2.8e8 pairs on 69 k /
# 0.87 M / 1.1 M faces, losing below 7e7 (0.3-0.9x); on 28 M faces 0.68x at 64 queries and 2.6x at
# 256. The CPU device's direct sum is two orders slower per pair and its build no slower per face,
# so its gate is far lower: 1.3-3.0x from 6.5e4 pairs at 64 queries on 320-5 k faces, 0.94x at 16
# queries.
WINDING_TREE_FROM_PAIRS_CUDA = 1 << 28
WINDING_TREE_FROM_QUERIES_CUDA = 128
WINDING_TREE_FROM_PAIRS_CPU = 1 << 16
WINDING_TREE_FROM_QUERIES_CPU = 32
# Lanes per query block: they split each node's cap edges and faces, striding by
# ``wp.block_dim()``. Swept build + walk at 16 / 32 / 64 / 128 lanes: 32 is best on 69 k faces
# (1.13x over 64 at 10 k queries, 1.50x at 100 k) and within 1.07x of the best (64) on 0.87-1.1 M
# faces; 128 lost to both everywhere but 28 M faces (1.01x). It is also the block width this module
# already launches with, so it adds no compiled variant of the module.
WINDING_TREE_BLOCK_DIM = 32
# Every ``atan2`` term is added as an integer multiple of ``2**-bits`` radians (the walk's
# ``fixed_point_scale`` is ``2**bits``), so the sum is the same whichever lane adds which term in
# whichever order: the cap edges are placed by atomic cursors, and the result is still
# bit-reproducible. A float32 term of magnitude at least ``2**(23 - bits)`` scales to an exact
# integer, so only smaller terms round, by at most half a unit. A query sums at most ``n_faces``
# terms -- a node contributes its cap only when that is shorter than its face list -- each at most
# ``pi < 4`` in magnitude, so ``bits = 61 - ceil(log2(n_faces))`` keeps the 64-bit total from
# overflowing; ``WINDING_FIXED_POINT_MAX_BITS`` caps it where no float32 term needs more.
WINDING_FIXED_POINT_MAX_BITS = 40
# Relative growth of a node's region before a query counts as outside it: the apex is rounded to
# float32, so it may sit an ulp outside a region it lies on the face of, and the identity needs it
# inside. Growing the region only sends a query down one more level.
WINDING_REGION_GROWTH = wp.float32(1.0e-6)


@wp.func
def diagonal_slabs(p: wp.vec3) -> wp.vec4:
    # Coordinates along the four body diagonals: with the box's three axes, a 14-DOP.
    return wp.vec4(p[0] + p[1] + p[2], p[0] + p[1] - p[2], p[0] - p[1] + p[2], -p[0] + p[1] + p[2])


@wp.kernel
def winding_face_keys(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    corners: wp.array[wp.float32],
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # dim == n_faces: the Morton code of each face centroid over the vertices' box, and the face's
    # own index, into the radix sort's double-width buffers. ``corners`` is
    # ``kernel_reduce.minmax_vec3_chunked``'s ``[lower, -upper]``, read here so no readback sits
    # between the reduction and the sort.
    f = wp.int32(wp.tid())
    lower = wp.vec3(corners[0], corners[1], corners[2])
    extent = wp.vec3(-corners[3], -corners[4], -corners[5]) - lower
    inv_extent = wp.vec3()
    for axis in range(3):
        if extent[axis] > 0.0:
            inv_extent[axis] = 1023.0 / extent[axis]
    a, b, c = kernel_triangles.face_vertices(vertices, faces, f)
    out_keys[f] = morton_code_30((a + b + c) / 3.0, lower, inv_extent)
    out_order[f] = f


@wp.kernel
def winding_sorted_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_corners: wp.array[wp.vec3],
    out_rank: wp.array[wp.int32],
) -> None:
    # dim == n_faces: the corners of the ``i``-th face in Morton order, three to a face, so a node's
    # faces are one contiguous run of positions, and each face's position in that order.
    i = wp.int32(wp.tid())
    f = order[i]
    a, b, c = kernel_triangles.face_vertices(vertices, faces, f)
    out_corners[3 * i] = a
    out_corners[3 * i + 1] = b
    out_corners[3 * i + 2] = c
    out_rank[f] = i


@wp.kernel
def winding_leaves(
    corners: wp.array[wp.vec3],
    leaf_faces: wp.int32,
    n_leaves: wp.int32,
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
    out_slab_lower: wp.array[wp.vec4],
    out_slab_upper: wp.array[wp.vec4],
    out_corner_sum: wp.array[wp.vec3d],
    out_count: wp.array[wp.int32],
    out_first: wp.array[wp.int32],
) -> None:
    # dim == n_leaves: leaf ``j`` (node ``n_leaves + j``) over its sorted faces. A padding leaf past
    # the last face gets count 0 and an empty region; the walk never enters one.
    j = wp.int32(wp.tid())
    n_faces = corners.shape[0] // 3
    start = wp.min(j * leaf_faces, n_faces)
    end = wp.min(start + leaf_faces, n_faces)
    lower = wp.vec3(FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT)
    upper = -lower
    slab_lower = wp.vec4(
        FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT
    )
    slab_upper = -slab_lower
    corner_sum = wp.vec3d()
    for t in range(3 * start, 3 * end):
        p = corners[t]
        lower = wp.min(lower, p)
        upper = wp.max(upper, p)
        d = diagonal_slabs(p)
        slab_lower = wp.min(slab_lower, d)
        slab_upper = wp.max(slab_upper, d)
        corner_sum = corner_sum + to_vec3d(p)
    k = n_leaves + j
    out_lower[k] = lower
    out_upper[k] = upper
    out_slab_lower[k] = slab_lower
    out_slab_upper[k] = slab_upper
    out_corner_sum[k] = corner_sum
    out_count[k] = end - start
    out_first[k] = start


@wp.kernel
def winding_merge_level(
    level_start: wp.int32,
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
    out_slab_lower: wp.array[wp.vec4],
    out_slab_upper: wp.array[wp.vec4],
    out_corner_sum: wp.array[wp.vec3d],
    out_count: wp.array[wp.int32],
    out_first: wp.array[wp.int32],
) -> None:
    # dim == level width: node ``level_start + tid`` from its two children, which the previous
    # launch (one level deeper) wrote into the same arrays.
    k = level_start + wp.int32(wp.tid())
    left = 2 * k
    right = left + 1
    out_lower[k] = wp.min(out_lower[left], out_lower[right])
    out_upper[k] = wp.max(out_upper[left], out_upper[right])
    out_slab_lower[k] = wp.min(out_slab_lower[left], out_slab_lower[right])
    out_slab_upper[k] = wp.max(out_slab_upper[left], out_slab_upper[right])
    out_corner_sum[k] = out_corner_sum[left] + out_corner_sum[right]
    out_count[k] = out_count[left] + out_count[right]
    out_first[k] = out_first[left]


@wp.func
def halfedge_cap_levels(
    h: wp.int32,
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    leaf_faces: wp.int32,
    depth: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # Halfedge ``h``'s leaf, and the shallowest tree level at which it lies on a node's boundary:
    # it does at every level from there down to the leaves. It cancels against its mate inside
    # every node holding both faces, i.e. down to their leaves' lowest common ancestor, whose
    # level is ``depth`` minus the bit length of the two leaf indices' xor -- but only when the
    # mate runs the other way. A boundary edge, a same-way pair and a non-manifold edge
    # (``mates[h] < 0`` or a mate that does not reverse ``h``) never cancel: always correct, a
    # cancelled pair only saves work. ``count_cap_edges`` and ``fill_cap_edges`` must agree
    # halfedge by halfedge, hence one function.
    f = h // 3
    leaf = rank[f] // leaf_faces
    first_level = wp.int32(0)
    mate = mates[h]
    if mate >= 0:
        if faces[mate] == faces[3 * f + (h - 3 * f + 1) % 3]:
            x = leaf ^ (rank[mate // 3] // leaf_faces)
            bits = wp.int32(0)
            while x > 0:
                x = x >> 1
                bits += 1
            first_level = depth - bits + 1
    return leaf, first_level


@wp.kernel
def count_cap_edges(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    leaf_faces: wp.int32,
    depth: wp.int32,
    out_perimeter: wp.array[wp.int32],
) -> None:
    # dim == 3 * n_faces: each halfedge adds itself to the perimeter of every internal node (levels
    # below ``depth``) on whose boundary it lies. ``out_perimeter`` is zeroed by the caller.
    h = wp.int32(wp.tid())
    leaf, first_level = halfedge_cap_levels(h, faces, mates, rank, leaf_faces, depth)
    for level in range(first_level, depth):
        wp.atomic_add(out_perimeter, ((1 << depth) + leaf) >> (depth - level), 1)


@wp.func
def keeps_cap(
    k: wp.int32, n_leaves: wp.int32, perimeter: wp.array[wp.int32], count: wp.array[wp.int32]
) -> wp.bool:
    # Does node ``k`` keep a cap? Internal nodes only, and only when the cap is cheaper than the
    # node's faces. The offsets, the fill and the walk all ask this, so it is one function.
    return k >= 1 and k < n_leaves and perimeter[k] < count[k]


@wp.kernel
def cap_edge_counts(
    n_leaves: wp.int32,
    perimeter: wp.array[wp.int32],
    count: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
) -> None:
    # dim == 2 * n_leaves: node ``k``'s cap size into ``out_offsets[k + 1]`` (0 for a node without a
    # cap) and the leading 0, for an in-place inclusive scan into the ``n + 1`` offsets.
    k = wp.int32(wp.tid())
    size = wp.int32(0)
    if keeps_cap(k, n_leaves, perimeter, count):
        size = perimeter[k]
    out_offsets[k + 1] = size
    if k == 0:
        out_offsets[0] = 0


@wp.kernel
def fill_cap_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    leaf_faces: wp.int32,
    depth: wp.int32,
    perimeter: wp.array[wp.int32],
    count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_cap_start: wp.array[wp.vec3],
    out_cap_end: wp.array[wp.vec3],
    out_fill: wp.array[wp.int32],
) -> None:
    # dim == 3 * n_faces: ``count_cap_edges`` again, writing each halfedge's endpoints into the cap
    # of every node that keeps one. ``out_fill`` is a zeroed per-node cursor, so the order inside a
    # cap varies run to run; the walk's fixed-point sum does not see it.
    h = wp.int32(wp.tid())
    leaf, first_level = halfedge_cap_levels(h, faces, mates, rank, leaf_faces, depth)
    n_leaves = wp.int32(1) << depth
    if first_level < depth:
        f = h // 3
        a = vertices[faces[h]]
        b = vertices[faces[3 * f + (h - 3 * f + 1) % 3]]
        for level in range(first_level, depth):
            k = (n_leaves + leaf) >> (depth - level)
            if keeps_cap(k, n_leaves, perimeter, count):
                slot = offsets[k] + wp.atomic_add(out_fill, k, 1)
                out_cap_start[slot] = a
                out_cap_end[slot] = b


@wp.func
def outside_node_region(
    q: wp.vec3,
    slabs: wp.vec4,
    lower: wp.vec3,
    upper: wp.vec3,
    slab_lower: wp.vec4,
    slab_upper: wp.vec4,
) -> wp.bool:
    # Is ``q`` outside the node's 14-DOP, grown by ``WINDING_REGION_GROWTH`` of its coordinates'
    # magnitude? A ``nan`` query reports outside (every comparison is false), and its fan sum is
    # ``nan`` like the direct sum's.
    scale = wp.max(
        wp.max(wp.abs(lower[0]), wp.abs(upper[0])),
        wp.max(
            wp.max(wp.abs(lower[1]), wp.abs(upper[1])), wp.max(wp.abs(lower[2]), wp.abs(upper[2]))
        ),
    )
    grow = WINDING_REGION_GROWTH * scale
    box_grow = wp.vec3(grow, grow, grow)
    slab_grow = 3.0 * wp.vec4(grow, grow, grow, grow)
    if not is_in_aabb(q, lower - box_grow, upper + box_grow):
        return True
    lo = slab_lower - slab_grow
    hi = slab_upper + slab_grow
    return not (
        slabs[0] >= lo[0]
        and slabs[0] <= hi[0]
        and slabs[1] >= lo[1]
        and slabs[1] <= hi[1]
        and slabs[2] >= lo[2]
        and slabs[2] <= hi[2]
        and slabs[3] >= lo[3]
        and slabs[3] <= hi[3]
    )


@wp.func
def fixed_point_angle(detf: wp.float32, denom: wp.float32, scale: wp.float32) -> wp.int64:
    return wp.int64(wp.round(wp.atan2(detf, denom) * scale))


@wp.kernel
def winding_number_tree(
    corners: wp.array[wp.vec3],
    n_leaves: wp.int32,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    slab_lower: wp.array[wp.vec4],
    slab_upper: wp.array[wp.vec4],
    corner_sum: wp.array[wp.vec3d],
    count: wp.array[wp.int32],
    first: wp.array[wp.int32],
    perimeter: wp.array[wp.int32],
    cap_offsets: wp.array[wp.int32],
    cap_start: wp.array[wp.vec3],
    cap_end: wp.array[wp.vec3],
    fixed_point_scale: wp.float32,
    fixed_point_to_turns: wp.float64,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    # One block per query (``launch_tiled``, dim == n_queries). Every lane walks the whole tree --
    # the same branches, since they share the query -- and the lanes split each node's cap edges or
    # faces, striding by ``wp.block_dim()`` (one lane on the CPU device, which
    # then covers them all). The walk is stackless: descend to ``2k``; past a finished node climb
    # while it is a right child, then step to its sibling; the root finishing ends it.
    q_index, lane = wp.tid()
    block = wp.block_dim()
    q = query_points[q_index]
    slabs = diagonal_slabs(q)
    total = wp.int64(0)
    k = wp.int32(1)
    while True:
        c = count[k]
        descend = False
        if c > 0:
            outside = outside_node_region(
                q, slabs, lower[k], upper[k], slab_lower[k], slab_upper[k]
            )
            if outside and keeps_cap(k, n_leaves, perimeter, count):
                apex = to_vec3(corner_sum[k] / wp.float64(3 * c))
                v0 = apex - q
                vl0 = wp.length(v0)
                for t in range(cap_offsets[k] + lane, cap_offsets[k + 1], block):
                    detf, denom = solid_angle_terms(v0, vl0, cap_start[t], cap_end[t], q)
                    total += fixed_point_angle(detf, denom, fixed_point_scale)
            elif outside or k >= n_leaves:
                start = first[k]
                for t in range(start + lane, start + c, block):
                    v0 = corners[3 * t] - q
                    detf, denom = solid_angle_terms(
                        v0, wp.length(v0), corners[3 * t + 1], corners[3 * t + 2], q
                    )
                    total += fixed_point_angle(detf, denom, fixed_point_scale)
            else:
                descend = True
        if descend:
            k = 2 * k
        else:
            while (k & 1) == 1 and k > 1:
                k = k >> 1
            if k == 1:
                break
            k = k + 1
    block_total = block_sum(total)
    if lane == 0:
        out_winding[q_index] = wp.float32(wp.float64(block_total) * fixed_point_to_turns)


@wp.func
def mesh_aabb_collect(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    max_hits: wp.int32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) up to ``max_hits`` face hits.
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_idx = wp.int32(0)
    c = wp.int32(0)
    # ``wp.mesh_query_next`` is the canonical iterator from Warp 1.17 -- it advances an AABB query
    # and a sphere query alike, and ``wp.mesh_query_aabb_next`` survives only as its alias.
    #
    # The cap is checked *first*: this is a genuine short-circuit (the codegen'd C++ ``&&``), so
    # once ``c`` reaches ``max_hits`` the loop stops asking the BVH for another candidate instead of
    # advancing the traversal one more step only to discard what it finds.
    while c < max_hits and wp.mesh_query_next(query, face_idx):
        if write:
            out_indices[base + c] = face_idx
        c = c + 1
    return c


@wp.kernel
def query_mesh_aabb_count(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    max_hits: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    out_counts[tid] = mesh_aabb_collect(
        mesh_id,
        query_lower[tid],
        query_upper[tid],
        max_hits,
        wp.bool(False),
        wp.int32(0),
        out_counts,
    )


@wp.kernel
def query_mesh_aabb_neighbors(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    max_hits: wp.int32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    mesh_aabb_collect(
        mesh_id,
        query_lower[tid],
        query_upper[tid],
        max_hits,
        wp.bool(True),
        offsets[tid],
        out_indices,
    )


@wp.kernel
def face_containing_point_2d(
    mesh_id: wp.uint64,
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec2],
    search_radius: wp.float32,
    barycentric_epsilon: wp.float32,
    out_face: wp.array[wp.int32],
) -> None:
    # Point location in a 2D triangulation: a closest-point query against the same triangulation
    # lifted to ``z = 0`` picks the *candidate* face, and a barycentric sign test decides.
    #
    # The candidate is sufficient rather than merely plausible: a point inside some triangle is at
    # distance zero from it, so the closest triangle is a containing one whenever any exists.
    #
    # The two-stage form is not redundant. Accepting on the query radius alone misclassifies a
    # fraction of a percent of random queries -- the closest-point distance for an in-plane point is
    # not exactly zero in float32, so a radius tight enough to reject points just outside the
    # triangulation also rejects points just inside it, and **no radius separates the two**. The
    # barycentric test is a sign test on the query's own coordinates, orders of magnitude sharper,
    # so the radius only has to be loose enough to find the candidate.
    #
    # An unbounded query returns the identical answer (a query farther than the radius from every
    # face has a barycentric coordinate far below ``-barycentric_epsilon``), and it was measured and
    # declined: the radius prunes the descent of every query landing outside, so dropping it, and
    # with it the reduction and readback that size it, is 2.4x at 40 000 queries on CUDA but
    # 1.1-1.2x slower at a million, and 1.2-2.3x slower at every size on the CPU.
    tid = wp.int32(wp.tid())
    p = points[tid]
    out_face[tid] = wp.int32(-1)
    query = wp.mesh_query_point_no_sign(mesh_id, lift_vec2(p, wp.float32(0.0)), search_radius)
    if not query.result:
        return

    face = query.face
    c0, c1, c2 = kernel_triangles.face_vertices(vertices, faces, face)
    barycentric = barycentric_2d(c0, c1, c2, p)
    if wp.min(barycentric[0], wp.min(barycentric[1], barycentric[2])) >= -barycentric_epsilon:
        out_face[tid] = face


@wp.kernel
def contains_points_sign_parity(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
    mesh_min: wp.vec3,
    mesh_max: wp.vec3,
    out_contains: wp.array[wp.bool],
) -> None:
    tid = wp.int32(wp.tid())
    p = points[tid]

    if not is_strictly_inside_aabb(p, mesh_min, mesh_max):
        out_contains[tid] = False
        return

    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    out_contains[tid] = query.result and query.sign < wp.float32(0.0)
