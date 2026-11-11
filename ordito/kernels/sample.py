import math

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT
from ordito.kernels.array import (
    LOOP_CONDITION,
    append_slot,
    device_is_cpu,
    float_from_ordered_bits,
    ordered_float_bits,
    scanned_count,
)
from ordito.kernels.reduce import (
    block_chunk,
    fold_block_totals,
    last_atomic_block,
    last_ticket_vector,
    minmax_vec3_chunk,
    peer_counted_add,
    radix_select_bin,
    radix_select_digit,
    stage_block_total,
    write_block_value,
)
from ordito.kernels.triangles import (
    block_area_weighted_centroid_sum,
    face_area,
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
GOLDEN_ANGLE = wp.float64(math.pi * (3.0 - math.sqrt(5.0)))


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
    out_barycentric: wp.array[wp.vec3],
) -> None:
    # A non-empty ``out_barycentric`` receives each point's coordinates in its face, corner order.
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
    if out_barycentric.shape[0] > 0:
        out_barycentric[tid] = wp.vec3(w, uv.x, uv.y)


@wp.kernel(module="unique")
def surface_centroid_apex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_apex: wp.array[wp.vec4],
    out_volumes: wp.array[wp.float32],
) -> None:
    # The area-weighted surface centroid ``sample.sample_volume`` fans its tetrahedra from, in one
    # launch: each block folds the ``triangles.face_area_weighted_centroid`` sums (components 0-2)
    # and areas (component 3) of its faces (``triangles.block_area_weighted_centroid_sum``) into
    # its row of a ``_launch.block_partials``-layout buffer of width 1; the block drawing the last
    # ticket adds the rows in a fixed order and writes the centroid (all-NaN for zero area, as
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
    folded = block_area_weighted_centroid_sum(vertices, faces, block, lane)
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


@wp.func
def neighbour_chunk(nbr_indices: wp.array[wp.int32], k: wp.int32, end: wp.int32) -> wp.vec4i:
    # Entries ``k`` to ``k + 3`` of a neighbour row ending at ``end``, every address clamped to the
    # row's last entry, so the four loads (and the four the caller issues through them) are in
    # flight together; the caller masks the slots past ``end``. Every pass of the elimination walks
    # a row of ~19 neighbours in one thread, a chain of dependent loads that set its CUDA device
    # time (``find_local_maxima`` 18 us a round on a 10 k pool, an entry at a time); taken four at
    # a time the walks ran 0.29-0.94x (maxima) and 0.50-0.73x (deletions) from 10 k to 1 M pool
    # points, and eight at a time no faster. The CPU's deletions took 0.83x too. Each sum still
    # adds in row order.
    last = end - 1
    return wp.vec4i(
        nbr_indices[wp.min(k, last)],
        nbr_indices[wp.min(k + 1, last)],
        nbr_indices[wp.min(k + 2, last)],
        nbr_indices[wp.min(k + 3, last)],
    )


@wp.func
def neighbour_chunk_dists(nbr_dists: wp.array[wp.float32], k: wp.int32, end: wp.int32) -> wp.vec4:
    # The distances of the four entries ``neighbour_chunk`` names, at the same clamped addresses.
    last = end - 1
    return wp.vec4(
        nbr_dists[wp.min(k, last)],
        nbr_dists[wp.min(k + 1, last)],
        nbr_dists[wp.min(k + 2, last)],
        nbr_dists[wp.min(k + 3, last)],
    )


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
    for k in range(start, end, 4):
        js = neighbour_chunk(nbr_indices, k, end)
        dists = neighbour_chunk_dists(nbr_dists, k, end)
        alive_j = wp.vec4i(alive[js[0]], alive[js[1]], alive[js[2]], alive[js[3]])
        for s in range(4):
            j = js[s]
            if k + s < end and j != i and alive_j[s] != 0:
                w += _poisson_edge_weight(dists[s], r_max, r_min, alpha)
    out_weights[i] = w


# The elimination loop's device state, one ``int32`` table a call: the ``wp.capture_while``
# condition at ``array.LOOP_CONDITION`` (slot 0 unused), the alive count, the round's maxima count,
# then the weight floor's radix select: the ticket of its passes' last blocks, the floor key's
# digits selected so far (after the last pass the whole key: ``array.ordered_float_bits`` of minus
# the floor weight, as ``int32`` bits) and the rank still to find below them. The alive count is
# stepped by the first launch of the *next* round, so the last round's pair stays readable after
# the loop: whether it deleted is ``poisson_round_deletes`` of it.
POISSON_ALIVE = wp.int32(2)
POISSON_MAXIMA = wp.int32(3)
POISSON_TICKET = 4
POISSON_PREFIX = wp.int32(5)
POISSON_RANK = wp.int32(6)
POISSON_STATE_SIZE = 7

# The floor is the ``excess``-th smallest of the alive points' ordered keys, selected a digit a
# pass (``floor_digit``) by ``reduce``'s radix select (``RADIX_SELECT_PASSES_32``, see there for
# its price against the sort). A block owns ``FLOOR_ITEMS_PER_BLOCK`` points and launches at the
# default width, so the module is built once.
FLOOR_ITEMS_PER_BLOCK = 256


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
    # A round's sort keys and the identity payload (the CPU device's floor, and the terminal
    # round's order), written straight into the leading halves of the radix sort's double-width
    # buffers (the upper halves are its scratch).
    i = wp.int32(wp.tid())
    out_keys[i] = alive_negated_weight(weights[i], alive[i])
    out_order[i] = i


@wp.func
def close_poisson_books(state: wp.array[wp.int32]) -> None:
    # The previous round's maxima, all deleted if it ran this far, leave the alive count, and the
    # maxima count is re-armed for this round's ``find_local_maxima``.
    state[POISSON_ALIVE] = state[POISSON_ALIVE] - state[POISSON_MAXIMA]
    state[POISSON_MAXIMA] = 0


@wp.func
def floor_digit_bin(
    weight: wp.float32, alive: wp.int32, prefix: wp.uint32, shift: wp.int32, width: wp.int32
) -> wp.int32:
    # The bin a point counts into in the floor pass taking ``width`` bits at ``shift`` of its
    # ordered key (``reduce.radix_select_bin``), or ``-1``: a dead point (keyed past every alive
    # one, and the rank sought is below the alive count), or one whose key lacks the digits
    # selected so far. The first pass (``shift + width == 32``) matches no prefix: ``state`` still
    # holds the previous round's floor there.
    if alive == 0:
        return -1
    selected = wp.where(shift + width == 32, wp.uint64(0), wp.uint64(prefix))
    return radix_select_bin(wp.uint64(ordered_float_bits(-weight)), selected, shift, width)


@wp.func
def append_floor_digit(
    lane: wp.int32,
    count: wp.int32,
    shift: wp.int32,
    width: wp.int32,
    hist: wp.array[wp.int32],
    state: wp.array[wp.int32],
) -> None:
    # A pass's last word, every lane of the block that counted last: the digit holding the rank
    # still sought (the ``excess``-th smallest key's in the first pass, read off the stepped alive
    # count), appended to the selected prefix, and the rank left within it.
    first_pass = shift + width == 32
    rank = wp.where(first_pass, state[POISSON_ALIVE] - count - 1, state[POISSON_RANK])
    digit, within, _run = radix_select_digit(lane, rank, hist)
    if lane == 0:
        prefix = wp.uint32(state[POISSON_PREFIX])
        above = wp.where(first_pass, wp.uint32(0), prefix << wp.uint32(width))
        state[POISSON_PREFIX] = wp.cast(above | wp.uint32(digit), wp.int32)
        state[POISSON_RANK] = within


@wp.kernel
def floor_digit(
    weights: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    count: wp.int32,
    shift: wp.int32,
    width: wp.int32,
    hist: wp.array[wp.int32],
    ticket: wp.array[wp.int32],
    state: wp.array[wp.int32],
) -> None:
    # One digit of the round's weight floor on CUDA, ``width`` bits at ``shift`` of the ordered
    # key: every point ``floor_digit_bin`` counts goes into its bin (peers pooled: the keys crowd a
    # few bins), and the last block, whose reads see every block's adds, appends the digit. Lane 0
    # of block 0 of the first pass first closes the previous round's books, which the last block's
    # rank reads.
    block, lane = wp.tid()
    if shift + width == 32 and block == 0 and lane == 0:
        close_poisson_books(state)
    prefix = wp.uint32(state[POISSON_PREFIX])
    n = weights.shape[0]
    offset, items = block_chunk(n, block, FLOOR_ITEMS_PER_BLOCK)
    for t in range(offset + lane, offset + items, wp.block_dim()):
        b = floor_digit_bin(weights[t], alive[t], prefix, shift, width)
        if b >= 0:
            peer_counted_add(hist, b)
    n_blocks = (n + FLOOR_ITEMS_PER_BLOCK - 1) // FLOOR_ITEMS_PER_BLOCK
    if last_atomic_block(lane, ticket, n_blocks) != 0:
        append_floor_digit(lane, count, shift, width, hist, state)


@wp.kernel
def floor_from_sorted(
    sorted_negated: wp.array[wp.float32], count: wp.int32, state: wp.array[wp.int32]
) -> None:
    # ``dim=1``: the round's weight floor on the CPU device, after ``alive_negated_weight_keys``
    # and a radix sort: the previous round's books closed, then the ``excess``-th smallest key
    # stored where the ``floor_digit`` passes leave it. Three histogram passes, even in one
    # thread, cost the CPU's call 1.05-1.08x the sort they replace on CUDA.
    close_poisson_books(state)
    floor = sorted_negated[state[POISSON_ALIVE] - count - 1]
    state[POISSON_PREFIX] = wp.cast(ordered_float_bits(floor), wp.int32)


@wp.func
def outweighs(weight_j: wp.float32, j: wp.int32, wi: wp.float32, i: wp.int32) -> wp.int32:
    # 1 when neighbour ``j`` of weight ``weight_j`` (clamped at 0) is heavier than point ``i`` of
    # clamped weight ``wi`` on ``(weight, -index)``, so a tie is broken by the smaller index. The
    # tie-break is what makes the flagged set an *independent* set in the neighbour graph -- two
    # flagged points can never be neighbours -- which is the property the round-based elimination
    # assumes when it deletes every flagged point at once. A strict ``>`` alone does not give it,
    # and the ties are systematic rather than rare: ``_poisson_edge_weight`` clamps any distance
    # below ``r_min`` up to it, so every neighbour inside ``r_min`` contributes the identical term
    # and a point's weight is exactly its close-neighbour count times one constant. A clique of
    # equally crowded points was therefore flagged entire and deleted in a single round, taking a
    # whole dense cluster out at once. It costs at most one extra round -- 26 against 26 and 27 on
    # an icosphere(4) at 200 and 2 000 samples -- because only a tie is decided differently.
    wj = wp.max(weight_j, wp.float32(0.0))
    return wp.where(wj > wi or (wj == wi and j < i), wp.int32(1), wp.int32(0))


@wp.kernel
def find_local_maxima(
    weights: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_is_max: wp.array[wp.int32],
) -> None:
    # ``state[POISSON_MAXIMA]`` accumulates the number of flagged points -- the round's one
    # quantity the loop branches on -- as a conditional append (``append_slot``, one atomic a warp)
    # on the flag this thread already holds, rather than a device reduction re-reading the whole
    # ``out_is_max`` mask afterwards. the first ``floor_digit`` pass zeroes it before each round.
    #
    # The floor is the weight of the round's ``excess``-th heaviest alive point (``excess`` = alive
    # count - ``count``), selected into ``state[POISSON_PREFIX]`` as an ordered key of minus the
    # weight, and a point lighter than that is not flagged however it compares with its
    # neighbours. It is compared as a float, so a zero's sign cannot change the test.
    # Sequential elimination deletes the *globally* heaviest point each time, so its budget of
    # deletions goes to the crowded regions and a locally heaviest point of a sparse region is
    # never reached; flagging every local maximum spent the budget thinning sparse regions and
    # left crowded pairs standing, 0.10 of ``r_max`` apart where sequential elimination keeps
    # 0.70. With the floor the rounds keep exactly the sequential set (100 of 100 on an
    # icosphere(2)).
    i = wp.int32(wp.tid())
    floor = -float_from_ordered_bits(wp.cast(state[POISSON_PREFIX], wp.uint32))
    if alive[i] == 0 or weights[i] < floor:
        out_is_max[i] = 0
        return
    wi = wp.max(weights[i], wp.float32(0.0))
    start = offsets[i]
    end = offsets[i + 1]
    # ``seen`` starts at 0 and only rises once an alive neighbour has been seen: a point with no
    # alive neighbour inside ``r_max`` carries weight 0, the least crowded state there is, so it is
    # the *last* point elimination should reach -- not a vacuous maximum. Flagging it deleted the
    # sole survivor of every isolated patch, wiping small far components off the result entirely.
    #
    # On CUDA the row is walked a ``neighbour_chunk`` at a time, so the decision cannot depend on
    # the order within one: ``seen`` rises at any alive neighbour and ``beaten`` at any heavier
    # one, and the walk stops after the chunk that beats it. The CPU device walks an entry at a
    # time: it has no load latency to hide, and a chunk's later entries cost it 1.18x the kernel
    # (a 100 k pool) where the first already beats the point.
    seen = wp.int32(0)
    beaten = wp.int32(0)
    if device_is_cpu() != 0:
        for k in range(start, end):
            j = nbr_indices[k]
            if j != i and alive[j] != 0:
                seen = 1
                if outweighs(weights[j], j, wi, i) != 0:
                    beaten = 1
                    break
    else:
        for k in range(start, end, 4):
            js = neighbour_chunk(nbr_indices, k, end)
            weights_j = wp.vec4(weights[js[0]], weights[js[1]], weights[js[2]], weights[js[3]])
            alive_j = wp.vec4i(alive[js[0]], alive[js[1]], alive[js[2]], alive[js[3]])
            for s in range(4):
                j = js[s]
                if k + s < end and j != i and alive_j[s] != 0:
                    seen = 1
                    beaten = wp.max(beaten, outweighs(weights_j[s], j, wi, i))
            if beaten != 0:
                break
    is_max = wp.where(seen != 0 and beaten == 0, wp.int32(1), wp.int32(0))
    out_is_max[i] = is_max
    if is_max != 0:
        append_slot(state, POISSON_MAXIMA)


@wp.kernel
def mark_sorted_prefix(order: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    # One thread per rank ``k < excess`` of the round's weight sort: flag the point at rank ``k``.
    # The terminal round in which no point is a local maximum deletes the ``excess`` heaviest alive
    # points, which lead the sort (the deleted ones are keyed past the end).
    k = wp.int32(wp.tid())
    out_flags[order[k]] = wp.int32(1)


@wp.kernel
def sorted_flags(
    flags: wp.array[wp.int32], order: wp.array[wp.int32], out_sorted: wp.array[wp.int32]
) -> None:
    # The round's maxima flags in the order of its weight sort, for the scan that ranks them.
    k = wp.int32(wp.tid())
    out_sorted[k] = flags[order[k]]


@wp.kernel
def unflag_past_rank(
    ranks: wp.array[wp.int32],
    order: wp.array[wp.int32],
    excess: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # Clear every flagged point the inclusive scan of ``sorted_flags`` ranks past ``excess``, so
    # the terminal round whose maxima outnumber what is left deletes exactly the ``excess``
    # heaviest of them, ties to the smaller index (the stable sort's identity payload).
    k = wp.int32(wp.tid())
    rank, flagged = scanned_count(ranks, k)
    if flagged != 0 and rank >= excess:
        out_flags[order[k]] = 0


@wp.func
def poisson_round_deletes(state: wp.array[wp.int32], count: wp.int32) -> wp.bool:
    # Whether the round deletes all of its maxima: some were flagged and they fit in what is left
    # to delete. The other two outcomes end the loop and the host finishes the last round.
    n_max = state[POISSON_MAXIMA]
    return n_max > 0 and n_max <= state[POISSON_ALIVE] - count


@wp.func
def close_poisson_round(
    i: wp.int32, deletes: wp.bool, count: wp.int32, state: wp.array[wp.int32]
) -> None:
    # Thread 0 of a deletion pass publishes whether another round runs: this one deleted and left
    # more than ``count`` alive. No thread of the pass reads the condition slot.
    if i == 0:
        more = deletes and state[POISSON_ALIVE] - state[POISSON_MAXIMA] > count
        state[LOOP_CONDITION] = wp.where(more, wp.int32(1), wp.int32(0))


@wp.kernel
def apply_deletions(
    deleted_mask: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    nbr_dists: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    r_max: wp.float32,
    r_min: wp.float32,
    alpha: wp.float32,
    count: wp.int32,
    guarded: wp.int32,
    alive: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    state: wp.array[wp.int32],
) -> None:
    # Delete this round's flagged points and take their contributions off the survivors' weights,
    # in one pass. ``alive`` and ``weights`` are in-place state carried across rounds. Inside the
    # loop (``guarded``) the pass runs only when ``poisson_round_deletes`` says the round deletes
    # its maxima, and thread 0 publishes the loop's condition (``close_poisson_round``); the host's
    # terminal round passes ``guarded = 0`` with a mask cut to ``excess``.
    #
    # A deleted thread clears its own ``alive`` slot while its neighbours may be reading it. That
    # is benign because the skip below tests ``deleted_mask[j]`` as well: a neighbour this round
    # deletes is skipped whether or not its ``alive`` write has landed, and a survivor's slot is
    # not written this round at all. So the predicate -- and the set of contributions subtracted --
    # is exactly that of clearing every deleted slot first and subtracting after.
    i = wp.int32(wp.tid())
    deletes = guarded == 0 or poisson_round_deletes(state, count)
    close_poisson_round(i, deletes, count, state)
    if not deletes or deleted_mask[i] == 0:
        return
    alive[i] = 0
    start = offsets[i]
    end = offsets[i + 1]
    for k in range(start, end, 4):
        js = neighbour_chunk(nbr_indices, k, end)
        dists = neighbour_chunk_dists(nbr_dists, k, end)
        alive_j = wp.vec4i(alive[js[0]], alive[js[1]], alive[js[2]], alive[js[3]])
        deleted_j = wp.vec4i(
            deleted_mask[js[0]], deleted_mask[js[1]], deleted_mask[js[2]], deleted_mask[js[3]]
        )
        for s in range(4):
            j = js[s]
            if k + s < end and j != i and alive_j[s] != 0 and deleted_j[s] == 0:
                wp.atomic_add(weights, j, -_poisson_edge_weight(dists[s], r_max, r_min, alpha))


@wp.kernel
def gather_deletions(
    deleted_mask: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    nbr_dists: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    r_max: wp.float32,
    r_min: wp.float32,
    alpha: wp.float32,
    count: wp.int32,
    guarded: wp.int32,
    alive: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    state: wp.array[wp.int32],
) -> None:
    # ``apply_deletions`` under the deterministic setting: the same deletions and contributions,
    # gathered by each survivor over its own row rather than scattered by the deleted points. Two
    # maxima deleted in one round can share a neighbour (each is within ``r_max`` of it but not of
    # the other), and the scatter's float atomics then subtract in arrival order, which on a large
    # pool (lucy at 2 M samples) changed the samples run to run. The rows are symmetric (one ball
    # query of the pool against itself, ``|p - q| == |q - p|``), so a survivor's row holds exactly
    # the deleted points that would have scattered to it, and it subtracts them in row order. Only
    # a survivor writes its own weight and only a deleted point its own ``alive`` slot, and the
    # skip reads ``deleted_mask`` alone, so nothing races. The kernels differ only in who loops.
    # Every survivor walks its row each round, which the scatter's deleted points alone did: the
    # call takes 1.10-1.27x on CUDA (bunny 20 k, dragon 200 k, lucy 2 M samples); the CPU device
    # keeps the scatter, whose threads run in a fixed order there.
    i = wp.int32(wp.tid())
    deletes = guarded == 0 or poisson_round_deletes(state, count)
    close_poisson_round(i, deletes, count, state)
    if not deletes:
        return
    if deleted_mask[i] != 0:
        alive[i] = 0
        return
    if alive[i] == 0:
        return
    w = weights[i]
    end = offsets[i + 1]
    for k in range(offsets[i], end, 4):
        js = neighbour_chunk(nbr_indices, k, end)
        dists = neighbour_chunk_dists(nbr_dists, k, end)
        deleted_j = wp.vec4i(
            deleted_mask[js[0]], deleted_mask[js[1]], deleted_mask[js[2]], deleted_mask[js[3]]
        )
        for s in range(4):
            if k + s < end and js[s] != i and deleted_j[s] != 0:
                w -= _poisson_edge_weight(dists[s], r_max, r_min, alpha)
    weights[i] = w


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
