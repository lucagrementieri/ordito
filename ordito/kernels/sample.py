import math

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT
from ordito.kernels.array import scanned_count
from ordito.kernels.reduce import (
    block_chunk_1d,
    block_sum,
    fold_block_totals,
    last_ticket_vector,
    minmax_vec3_chunk,
    stage_block_total,
    write_block_value,
)
from ordito.kernels.triangles import (
    face_area,
    face_area_weighted_centroid,
    face_signed_volume,
    face_vertices,
)

wp.set_module_options({"enable_backward": False})

# Golden angle in radians: pi * (3 - sqrt(5)) ~ 2.399963. Successive multiples of this
# angle place points on the Fibonacci lattice, the most uniform simple spiral on a sphere.
#
# Held in float64 because it is multiplied by the sample index: the phase reaches
# ``2.4 * count`` radians, so at 100 000 directions the argument is ~2.4e5 and a float32
# argument reduction has already thrown away the low digits that make the spiral
# low-discrepancy. Same hazard, and the same fix, as
# ``kernels/bounds.oriented_box_candidate_axes``.
GOLDEN_ANGLE = wp.constant(wp.float64(math.pi * (3.0 - math.sqrt(5.0))))


@wp.func
def fibonacci_direction(i: wp.int32, count: wp.int32, z_span: wp.float32) -> wp.vec3:
    # Direction ``i`` of the ``count``-point Fibonacci lattice over a band of height ``z_span``.
    # z descends uniformly through (1 - z_span, 1); the offset 0.5 centers the samples.
    # z_span = 2 covers the full sphere, z_span = 1 the positive-z hemisphere.
    #
    # Only the phase runs in float64, and only the sine and cosine of it are narrowed: ``z`` is a
    # bounded interpolation that float32 resolves exactly well enough, while ``theta`` grows
    # without bound in ``count``. Measured against a float64 evaluation, the float32 phase costs
    # nothing to ~4 096 directions, 11 % of the minimum neighbour spacing at 100 000, and 0.21 rad
    # of azimuth at 1e6 -- the uniformity this lattice exists for. It costs nothing: the whole
    # call is launch- and allocation-bound, flat from 64 to 200 000 directions on both devices.
    count_f = wp.float32(count)
    z = 1.0 - z_span * (wp.float32(i) + 0.5) / count_f
    radius = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    theta = GOLDEN_ANGLE * wp.float64(i)
    return wp.vec3(radius * wp.float32(wp.cos(theta)), radius * wp.float32(wp.sin(theta)), z)


@wp.kernel
def fibonacci_lattice(
    count: wp.int32, z_span: wp.float32, out_directions: wp.array[wp.vec3]
) -> None:
    # The whole lattice, one direction a thread (``fibonacci_direction``).
    i = wp.int32(wp.tid())
    out_directions[i] = fibonacci_direction(i, count, z_span)


@wp.kernel
def face_areas_and_vertex_box(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    box: wp.int32,
    out_cdf: wp.array[wp.float32],
) -> None:
    # ``triangles.face_areas`` into the leading ``n_faces`` slots of ``out_cdf`` and, from the
    # trailing ``chunks_1d(n_vertices)`` threads, the vertices' box folded into packed box ``box``
    # of the same buffer (``reduce.minmax_vec3_chunked``'s layout, seeded ``inf``), so the CDF's
    # total and the box come back in one readback.
    t = wp.int32(wp.tid())
    n_faces = faces.shape[0] // 3
    if t < n_faces:
        out_cdf[t] = face_area(vertices, faces, t)
    else:
        minmax_vec3_chunk(vertices, t - n_faces, out_cdf, box)


@wp.kernel
def sample_surface(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cdf: wp.array[wp.float32],
    total: wp.float32,
    seed: wp.int32,
    out_points: wp.array[wp.vec3],
    out_face_indices: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    state = wp.rand_init(seed, tid)
    # ``wp.sample_cdf`` over the *unnormalized* prefix sum: the same single ``randf`` draw, so the
    # random stream continues exactly as before, scaled to ``total`` rather than the cdf divided by
    # it -- no normalizing pass. The two place a bin edge differently only by float32 rounding,
    # below the prefix scan's own; inline because ``randf`` advances ``state`` in place.
    fi = wp.lower_bound(cdf, wp.randf(state) * total)

    v0, v1, v2 = face_vertices(vertices, faces, fi)

    uv = wp.sample_triangle(state)
    w = 1.0 - uv.x - uv.y

    out_face_indices[tid] = fi
    out_points[tid] = v0 * w + v1 * uv.x + v2 * uv.y


@wp.kernel(module="unique")
def surface_centroid_apex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_apex: wp.array[wp.vec4],
    out_volumes: wp.array[wp.float32],
) -> None:
    # The area-weighted surface centroid ``sample.sample_volume`` fans its tetrahedra from, in one
    # launch: each block folds the ``triangles.face_area_weighted_centroid`` sums (components 0-2)
    # and areas (component 3) of the faces ``block_chunk_1d`` gives it, lanes striding them by
    # ``wp.block_dim()`` (so the one CPU lane walks them all), into its row of a
    # ``_launch.block_partials``-layout buffer of width 1; the block drawing the last ticket adds
    # the rows in a fixed order and writes the centroid (all-NaN for zero area, as
    # ``measures.surface_centroid`` returns) and the area into slot 0. Always the fixed-order
    # commit, so a seeded draw repeats on CUDA: the apex moves every sample. Its own module
    # (``module="unique"``): the only tiled launch here, at ``TILE_1D``, so the module launched at
    # the default width is not built a second time. Block 0 also clears the negative-volume flag
    # in the last slot of ``tetrahedron_volumes_flag_negative``'s buffer, which runs next and
    # writes every other slot, so that buffer needs no zeroing of its own.
    block, lane = wp.tid()
    if block == 0 and lane == 0:
        out_volumes[out_volumes.shape[0] - 1] = wp.float32(0.0)
    n_blocks = out_apex.shape[0] - 2
    offset, count = block_chunk_1d(faces.shape[0] // 3, block)
    total = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for k in range(lane, count, wp.block_dim()):
        contrib, area = face_area_weighted_centroid(vertices, faces, offset + k)
        total += wp.vec4(contrib[0], contrib[1], contrib[2], area)
    folded = block_sum(total)
    if stage_block_total(block, lane, folded, out_apex, 0, 1, n_blocks, last_ticket_vector) != 0:
        summed = fold_block_totals(lane, out_apex, 0, 1, n_blocks, wp.vec4(0.0, 0.0, 0.0, 0.0))
        area = summed[3]
        apex = wp.vec4(wp.nan, wp.nan, wp.nan, area)
        if area != wp.float32(0.0):
            apex = wp.vec4(summed[0] / area, summed[1] / area, summed[2] / area, area)
        write_block_value(lane, out_apex, 0, 1, apex)


@wp.func
def apex_point(center: wp.array[wp.vec4]) -> wp.vec3:
    # The fan apex ``surface_centroid_apex`` leaves in slot 0.
    apex = center[0]
    return wp.vec3(apex[0], apex[1], apex[2])


@wp.kernel
def tetrahedron_volumes_flag_negative(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.array[wp.vec4],
    out_volumes: wp.array[wp.float32],
) -> None:
    # ``triangles.face_signed_volumes`` (the same ``face_signed_volume`` helper) with the
    # star-shaped test folded in: the buffer carries one slot past the faces, zero-seeded (by
    # ``surface_centroid_apex``), that any
    # negative tetrahedron sets to ``1``. The wrapper scans the face slots in place into the CDF
    # and reads its total and this flag in one readback, rather than reducing a ``min``
    # separately. Every writer stores the same value, so the unordered stores need no atomic.
    # ``center`` is the buffer ``surface_centroid_apex`` writes, so the centroid never crosses to
    # the host.
    fi = wp.int32(wp.tid())
    volume = face_signed_volume(vertices, faces, fi, apex_point(center))
    out_volumes[fi] = volume
    if volume < wp.float32(0.0):
        out_volumes[out_volumes.shape[0] - 1] = wp.float32(1.0)


@wp.kernel
def sample_volume_tetrahedra(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.array[wp.vec4],
    cdf: wp.array[wp.float32],
    total: wp.float32,
    seed: wp.int32,
    out_points: wp.array[wp.vec3],
) -> None:
    tid = wp.int32(wp.tid())
    state = wp.rand_init(seed, tid)
    # Drawn as ``sample_surface`` draws (see there).
    fi = wp.lower_bound(cdf, wp.randf(state) * total)

    v0, v1, v2 = face_vertices(vertices, faces, fi)

    # Uniform sampling in the tetrahedron (center, v0, v1, v2) via order statistics of
    # 3 U(0,1) samples. Sorted values s1 ≤ s2 ≤ s3 give spacings
    # (s1, s2-s1, s3-s2, 1-s3) as barycentric coords for (center, v0, v1, v2):
    # P = center*(1-s3) + v0*s1 + v1*(s2-s1) + v2*(s3-s2)
    a = wp.randf(state)
    b = wp.randf(state)
    c = wp.randf(state)

    # Single-argument wp.min / wp.max reduce a vector to its extreme element.
    abc = wp.vec3(a, b, c)
    s1 = wp.min(abc)
    s3 = wp.max(abc)
    s2 = a + b + c - s1 - s3

    apex = apex_point(center)
    out_points[tid] = apex * (1.0 - s3) + v0 * s1 + v1 * (s2 - s1) + v2 * (s3 - s2)


@wp.func
def _poisson_edge_weight(
    d: wp.float32, r_max: wp.float32, r_min: wp.float32, alpha: wp.float32
) -> wp.float32:
    d_eff = wp.max(d, r_min)
    return wp.pow(wp.float32(1.0) - d_eff / r_max, alpha)


@wp.kernel
def compute_poisson_weights(
    nbr_indices: wp.array[wp.int32],
    nbr_dists: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    r_max: wp.float32,
    r_min: wp.float32,
    alpha: wp.float32,
    out_weights: wp.array[wp.float32],
) -> None:
    i = wp.int32(wp.tid())
    if alive[i] == 0:
        out_weights[i] = wp.float32(0.0)
        return
    start = offsets[i]
    end = offsets[i + 1]
    w = wp.float32(0.0)
    for k in range(start, end):
        j = nbr_indices[k]
        if j == i or alive[j] == 0:
            continue
        w += _poisson_edge_weight(nbr_dists[k], r_max, r_min, alpha)
    out_weights[i] = w


@wp.func
def alive_negated_weight(weight: wp.float32, alive: wp.int32) -> wp.float32:
    # The sort key of the round's weight floor: ascending order of ``-weight`` over the alive
    # points, the deleted ones pushed past the end.
    return wp.where(alive != 0, -weight, FLOAT32_INF_CONSTANT)


@wp.kernel
def alive_negated_weight_keys(
    weights: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    out_keys: wp.array[wp.float32],
    out_order: wp.array[wp.int32],
) -> None:
    # The round's sort keys and the identity payload, written straight into the leading halves of
    # the radix sort's double-width buffers (the upper halves are its scratch).
    i = wp.int32(wp.tid())
    out_keys[i] = alive_negated_weight(weights[i], alive[i])
    out_order[i] = i


@wp.kernel
def find_local_maxima(
    weights: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    weight_floor: wp.array[wp.float32],
    out_is_max: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # ``out_count[0]`` accumulates the number of flagged points -- the round's one host-visible
    # quantity -- as a conditional atomic on the flag this thread already holds, rather than a
    # device reduction re-reading the whole ``out_is_max`` mask afterwards. The caller zeroes it
    # before each round.
    #
    # ``weight_floor[0]`` is minus the weight of the round's ``excess``-th heaviest alive point,
    # and a point lighter than that is not flagged however it compares with its neighbours.
    # Sequential elimination deletes the *globally* heaviest point each time, so its budget of
    # deletions goes to the crowded regions and a locally heaviest point of a sparse region is
    # never reached; flagging every local maximum spent the budget thinning sparse regions and
    # left crowded pairs standing, 0.10 of ``r_max`` apart where sequential elimination keeps
    # 0.70. With the floor the rounds keep exactly the sequential set (100 of 100 on an
    # icosphere(2)).
    i = wp.int32(wp.tid())
    if alive[i] == 0 or weights[i] < -weight_floor[0]:
        out_is_max[i] = 0
        return
    wi = wp.max(weights[i], wp.float32(0.0))
    start = offsets[i]
    end = offsets[i + 1]
    # Starts at 0, not 1, and only rises once an alive neighbour has been seen: a point with no
    # alive neighbour inside ``r_max`` carries weight 0, the least crowded state there is, so it is
    # the *last* point elimination should reach -- not a vacuous maximum. Flagging it deleted the
    # sole survivor of every isolated patch, wiping small far components off the result entirely.
    is_max = wp.int32(0)
    for k in range(start, end):
        j = nbr_indices[k]
        if j == i or alive[j] == 0:
            continue
        wj = wp.max(weights[j], wp.float32(0.0))
        # Ordered on ``(weight, -index)``, so a tie is broken by the smaller index. The tie-break
        # is what makes the flagged set an *independent* set in the neighbour graph -- two flagged
        # points can never be neighbours -- which is the property the round-based elimination
        # assumes when it deletes every flagged point at once. A strict ``>`` alone does not give
        # it, and the ties are systematic rather than rare: ``_poisson_edge_weight`` clamps any
        # distance below ``r_min`` up to it, so every neighbour inside ``r_min`` contributes the
        # identical term and a point's weight is exactly its close-neighbour count times one
        # constant. A clique of equally crowded points was therefore flagged entire and deleted in
        # a single round, taking a whole dense cluster out at once. It costs at most one extra
        # round -- 26 against 26 and 27 on an icosphere(4) at 200 and 2 000 samples -- because
        # only a tie is decided differently.
        if wj > wi or (wj == wi and j < i):
            is_max = 0
            break
        is_max = 1
    out_is_max[i] = is_max
    if is_max != 0:
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def mark_ranked_flagged(
    flagged: wp.array[wp.int32], order: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # One thread per kept rank ``k < excess``: flag the pool point the weight sort put at rank
    # ``k`` among the ``flagged`` ones. The gather of the chosen indices, their bool mask and its
    # ``int32`` cast in one launch over a zeroed ``out_flags``.
    k = wp.int32(wp.tid())
    out_flags[flagged[order[k]]] = wp.int32(1)


@wp.kernel
def apply_deletions(
    deleted_mask: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    nbr_dists: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    r_max: wp.float32,
    r_min: wp.float32,
    alpha: wp.float32,
    alive: wp.array[wp.int32],
    weights: wp.array[wp.float32],
) -> None:
    # Delete this round's flagged points and take their contributions off the survivors' weights,
    # in one pass. ``alive`` and ``weights`` are in-place state carried across rounds.
    #
    # A deleted thread clears its own ``alive`` slot while its neighbours may be reading it. That
    # is benign because the skip below tests ``deleted_mask[j]`` as well: a neighbour this round
    # deletes is skipped whether or not its ``alive`` write has landed, and a survivor's slot is
    # not written this round at all. So the predicate -- and the set of contributions subtracted --
    # is exactly that of clearing every deleted slot first and subtracting after.
    i = wp.int32(wp.tid())
    if deleted_mask[i] == 0:
        return
    alive[i] = 0
    start = offsets[i]
    end = offsets[i + 1]
    for k in range(start, end):
        j = nbr_indices[k]
        if j == i or alive[j] == 0 or deleted_mask[j] != 0:
            continue
        contribution = _poisson_edge_weight(nbr_dists[k], r_max, r_min, alpha)
        wp.atomic_add(weights, j, -contribution)


@wp.kernel
def emit_kept_samples(
    inclusive: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    face_indices: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
    out_face_indices: wp.array[wp.int32],
) -> None:
    # The kept samples and their faces, in input order, from the in-place inclusive scan of their
    # 0/1 keep flags: sample ``i`` is kept exactly where the scan steps (``array.scanned_count``,
    # ``scatter.scatter_index_where_scanned``'s rule) and lands at the exclusive value -- the index
    # compaction and both gathers in one launch. Shared by the Poisson-disk elimination (its
    # ``alive`` flags) and the blue-noise dart throw (``blue_noise.dart_accepted_pool_flags``).
    i = wp.int32(wp.tid())
    slot, count = scanned_count(inclusive, i)
    if count != 0:
        out_points[slot] = points[i]
        out_face_indices[slot] = face_indices[i]
