import math

import warp as wp

from ordito.kernels.array import scanned_count
from ordito.kernels.triangles import face_vertices

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


@wp.kernel
def fibonacci_lattice(
    count: wp.int32, z_span: wp.float32, out_directions: wp.array[wp.vec3]
) -> None:
    # z descends uniformly through (1 - z_span, 1); the offset 0.5 centers the samples.
    # z_span = 2 covers the full sphere, z_span = 1 the positive-z hemisphere.
    #
    # Only the phase runs in float64, and only the sine and cosine of it are narrowed: ``z`` is a
    # bounded interpolation that float32 resolves exactly well enough, while ``theta`` grows
    # without bound in ``count``. Measured against a float64 evaluation, the float32 phase costs
    # nothing to ~4 096 directions, 11 % of the minimum neighbour spacing at 100 000, and 0.21 rad
    # of azimuth at 1e6 -- the uniformity this lattice exists for. It costs nothing: the whole
    # call is launch- and allocation-bound, flat from 64 to 200 000 directions on both devices.
    i = wp.int32(wp.tid())
    count_f = wp.float32(count)
    z = 1.0 - z_span * (wp.float32(i) + 0.5) / count_f
    radius = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    theta = GOLDEN_ANGLE * wp.float64(i)
    out_directions[i] = wp.vec3(
        radius * wp.float32(wp.cos(theta)), radius * wp.float32(wp.sin(theta)), z
    )


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


@wp.kernel
def sample_volume_tetrahedra(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.vec3,
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

    out_points[tid] = center * (1.0 - s3) + v0 * s1 + v1 * (s2 - s1) + v2 * (s3 - s2)


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


@wp.kernel
def find_local_maxima(
    weights: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_is_max: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # ``out_count[0]`` accumulates the number of flagged points -- the round's one host-visible
    # quantity -- as a conditional atomic on the flag this thread already holds, rather than a
    # device reduction re-reading the whole ``out_is_max`` mask afterwards. The caller zeroes it
    # before each round.
    i = wp.int32(wp.tid())
    if alive[i] == 0:
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
