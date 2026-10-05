import warp as wp

from ordito.constants import TOLERANCE_PLANAR_CONSTANT
from ordito.kernels.array import RegisterBlockedTable, morton_code_30
from ordito.kernels.proximity import closest_point_query
from ordito.kernels.reduce import block_sum
from ordito.kernels.tangent_space import any_perpendicular

wp.set_module_options({"enable_backward": False})

# Weighting of a ray inside the bundle. Passed as a warp-uniform kernel argument so both schemes
# share one compiled module (see AGENTS.md section 2.7 on runtime selection).
WEIGHT_COSINE = wp.constant(wp.int32(0))  # Lambert's cosine law: the physical ambient integral
WEIGHT_UNIFORM = wp.constant(wp.int32(1))  # every direction counts once (libigl's convention)

# Lanes per point for the bundle kernels below, which are launched with ``wp.launch_tiled`` -- one
# *block* per point, its lanes striding the ray bundle. One *thread* per point was the natural
# spelling and it starves the device: ``dim = n_points`` is a few thousand threads on a feature
# mesh, a small fraction of what an RTX-class GPU can hold, each walking 64-256 BVH queries in
# sequence. The block form measures several-fold faster with results bit-identical, by more the
# longer the ray bundle.
#
# 32 lanes measures the same as 64 to within noise; 256 loses on a short bundle, where most lanes
# then sit idle. The stride below is ``wp.block_dim()``, not this constant, so the same kernel is
# correct on the CPU device, where ``wp.launch_tiled`` runs one lane per block and
# ``wp.block_dim()`` reads 1: that lane covers every ray and the tile reductions return its own sum
# (the ``kernels/holes.py::fill_dp_span_tiled`` convention).
BUNDLE_BLOCK = 64


@wp.func
def bundle_direction(
    local: wp.vec3, normal: wp.vec3, basis_x: wp.vec3, basis_y: wp.vec3
) -> wp.vec3:
    # Lift a direction given in the ``+z``-axis frame of a Fibonacci lattice into the frame whose
    # ``z`` is ``normal``. Keeping the lattice in one canonical frame and rotating it per point is
    # what lets every point share a single direction array while staying low-discrepancy -- folding
    # a whole-sphere lattice onto the normal's side (libigl's approach) would not.
    return basis_x * local[0] + basis_y * local[1] + normal * local[2]


@wp.func
def hemisphere_frame(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    i: wp.int32,
    offset: wp.float32,
    sign: wp.float32,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    # The ray bundle's frame at point ``i``: ``(axis, basis_x, basis_y, origin)``, ready for
    # ``bundle_direction``. ``sign`` is ``+1`` for an outward hemisphere (occlusion, which asks what
    # the sky sees) and ``-1`` for an inward one (shape diameter, which asks how thick the solid
    # is); it flips the axis and steps the origin to the matching side of the surface.
    #
    # The *tangent* pair is built from the outward normal at both signs, deliberately. Crossing with
    # ``axis`` instead would flip ``basis_y``, mirroring the lattice about the ``x`` axis and
    # sending every ray somewhere else -- a change the parity tests would see and no reader would
    # have intended. Only the axis and the origin depend on ``sign``.
    normal = wp.normalize(normals[i])
    basis_x = any_perpendicular(normal)
    basis_y = wp.cross(normal, basis_x)
    axis = sign * normal
    return axis, basis_x, basis_y, points[i] + axis * offset


@wp.func
def bundle_ray(
    directions: wp.array[wp.vec3],
    r: wp.int32,
    weight_mode: wp.int32,
    axis: wp.vec3,
    basis_x: wp.vec3,
    basis_y: wp.vec3,
) -> tuple[wp.vec3, wp.float32]:
    # Ray ``r`` of a point's hemisphere bundle: its direction and weight. A hemisphere lattice has
    # ``local[2] == dot(direction, normal)`` by construction, so the cosine weight is already there
    # and needs no dot product. Shared by ``occlusion`` and ``obscurance``.
    local = directions[r]
    weight = wp.where(weight_mode == WEIGHT_COSINE, local[2], wp.float32(1.0))
    return bundle_direction(local, axis, basis_x, basis_y), weight


@wp.func
def commit_blocked_fraction(
    lane: wp.int32,
    i: wp.int32,
    total_weight: wp.float32,
    total_blocked: wp.float32,
    out_occlusion: wp.array[wp.float32],
) -> None:
    # The block's weighted blocked fraction for point ``i``: every lane's two sums folded by one
    # ``block_sum`` (a barrier, so every lane runs it), stored by lane 0. Shared by ``occlusion``
    # and ``obscurance``.
    block = block_sum(wp.vec2(total_weight, total_blocked))
    if lane == 0:
        out_occlusion[i] = wp.where(block[0] <= 0.0, wp.float32(0.0), block[1] / block[0])


@wp.kernel
def occlusion(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    weight_mode: wp.int32,
    max_t: wp.float32,
    offset: wp.float32,
    out_occlusion: wp.array[wp.float32],
) -> None:
    # Weighted fraction of an outward hemisphere bundle that is blocked: binary ambient occlusion,
    # where a hit at any distance blocks fully. A ray needs only whether it is blocked, so this is
    # ``obscurance``'s loop with ``mesh_query_ray_anyhit``, which stops at the first hit rather
    # than searching for the nearest: 1.10-1.17x on the scan meshes, values identical. Kept as its
    # own kernel because compiling both queries behind a ``tau`` branch in one body lost 0.90x.
    #
    # One block per point, lanes striding the bundle (see ``BUNDLE_BLOCK``); each lane accumulates
    # its rays and ``commit_blocked_fraction`` combines them.
    i, t = wp.tid()
    axis, basis_x, basis_y, origin = hemisphere_frame(points, normals, i, offset, wp.float32(1.0))
    total_weight = wp.float32(0.0)
    total_blocked = wp.float32(0.0)
    for r in range(t, directions.shape[0], wp.block_dim()):
        direction, weight = bundle_ray(directions, r, weight_mode, axis, basis_x, basis_y)
        total_weight += weight
        if wp.mesh_query_ray_anyhit(mesh_id, origin, direction, max_t):
            total_blocked += weight
    commit_blocked_fraction(t, i, total_weight, total_blocked, out_occlusion)


@wp.kernel
def obscurance(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    tau: wp.float32,
    weight_mode: wp.int32,
    max_t: wp.float32,
    offset: wp.float32,
    out_occlusion: wp.array[wp.float32],
) -> None:
    # Iones et al.'s volumetric obscurance (``tau > 0``): an occluder at distance ``t`` contributes
    # ``exp(-tau t)`` of its ray's weight, so a distant wall barely darkens the point. Binary
    # occlusion is the ``tau -> 0`` limit, which ``occlusion`` traces with an any-hit query; this
    # kernel needs the nearest hit's distance. Same bundle and block layout as ``occlusion``.
    i, t = wp.tid()
    axis, basis_x, basis_y, origin = hemisphere_frame(points, normals, i, offset, wp.float32(1.0))
    total_weight = wp.float32(0.0)
    total_blocked = wp.float32(0.0)
    for r in range(t, directions.shape[0], wp.block_dim()):
        direction, weight = bundle_ray(directions, r, weight_mode, axis, basis_x, basis_y)
        total_weight += weight
        query = wp.mesh_query_ray(mesh_id, origin, direction, max_t)
        if query.result:
            total_blocked += weight * wp.exp(-tau * query.t)
    commit_blocked_fraction(t, i, total_weight, total_blocked, out_occlusion)


# The point-major layout of the bundle kernels for large clouds: a block of ``POINT_MAJOR_BLOCK``
# lanes takes ``POINT_MAJOR_POINTS`` points that are consecutive in Morton order
# (``morton_point_order``), lane ``t`` tracing point ``t % POINT_MAJOR_POINTS`` along ray group ``t
# // POINT_MAJOR_POINTS``, so a warp traces 32 neighbouring points along one lattice direction and
# their BVH walks share nodes. Per-point sums over the groups are one ``block_sum`` of a vector
# holding each lane's sums in its own point's slot (reading them off a lane-sized ``(groups,
# points)`` tile instead measured 0.89x / 0.87x at 64 / 256 rays on a 14 M-point scan). It pays only
# once the cloud is dense enough for 32 Morton neighbours to be close
# (``_OCCLUSION_POINT_MAJOR_FROM`` in ``ordito/visibility.py``); below that one block per point
# wins. Binary occlusion only: the same layout for ``shape_diameter``'s inward closest-hit bundle
# measured 0.65x (a row of scratch per point, scattered stores) and 0.98x with the scratch
# ray-major, at 64 rays on a 14 M-point scan, though an outward closest-hit bundle gains 1.25x
# there; inward rays all cross the volume. CUDA only: the CPU device runs one lane per block, which
# this partition cannot cover.
POINT_MAJOR_POINTS = 32
POINT_MAJOR_BLOCK = 128
PointMajorSums = wp.types.vector(length=POINT_MAJOR_POINTS, dtype=wp.float32)


@wp.kernel
def morton_point_order(
    points: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_extent: wp.vec3,
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # dim == n_points: each point's Morton code and its own index, written into the radix sort's
    # double-width buffers.
    i = wp.int32(wp.tid())
    out_keys[i] = morton_code_30(points[i], lower, inv_extent)
    out_order[i] = i


@wp.kernel
def occlusion_point_major(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    order: wp.array[wp.int32],
    directions: wp.array[wp.vec3],
    weight_mode: wp.int32,
    max_t: wp.float32,
    offset: wp.float32,
    out_occlusion: wp.array[wp.float32],
) -> None:
    # ``occlusion`` in the point-major layout (see ``POINT_MAJOR_POINTS``): the same rays and the
    # same any-hit test, summed over a different lane partition, so a value can differ from
    # ``occlusion``'s in its last bits. Launched ``dim=(ceil(n / POINT_MAJOR_POINTS),)`` at
    # ``POINT_MAJOR_BLOCK`` lanes; ``order`` is the Morton permutation.
    block, t = wp.tid()
    p = t % POINT_MAJOR_POINTS
    group = t // POINT_MAJOR_POINTS
    slot = block * POINT_MAJOR_POINTS + p
    total_weight = wp.float32(0.0)
    total_blocked = wp.float32(0.0)
    i = wp.int32(0)
    if slot < points.shape[0]:
        i = order[slot]
        axis, basis_x, basis_y, origin = hemisphere_frame(
            points, normals, i, offset, wp.float32(1.0)
        )
        groups = wp.block_dim() // POINT_MAJOR_POINTS
        for r in range(group, directions.shape[0], groups):
            direction, weight = bundle_ray(directions, r, weight_mode, axis, basis_x, basis_y)
            total_weight += weight
            if wp.mesh_query_ray_anyhit(mesh_id, origin, direction, max_t):
                total_blocked += weight
    weights = PointMajorSums()
    blocked = PointMajorSums()
    for k in range(POINT_MAJOR_POINTS):
        if k == p:
            weights[k] = total_weight
            blocked[k] = total_blocked
    weights = block_sum(weights)
    blocked = block_sum(blocked)
    if group == 0 and slot < points.shape[0]:
        out_occlusion[i] = wp.where(weights[p] <= 0.0, wp.float32(0.0), blocked[p] / weights[p])


@wp.func
def inward_distance(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32, offset: wp.float32
) -> wp.float32:
    # How far one inward ray travels before leaving the volume, ``inf`` when it never does; the ray
    # started ``offset`` inside the surface, which the distance adds back. Shared by
    # ``shape_diameter`` and ``shape_diameter_point_major``.
    query = wp.mesh_query_ray(mesh_id, origin, direction, max_t)
    distance = wp.float32(wp.inf)
    if query.result:
        distance = offset + query.t
    return distance


@wp.func
def diameter_window(
    total: wp.float32, total_sq: wp.float32, hits: wp.float32
) -> tuple[wp.float32, wp.float32]:
    # The mean distance of a point's hits and their deviation, from the first pass's sums (at least
    # one hit).
    mean = total / hits
    return mean, wp.sqrt(wp.max(0.0, total_sq / hits - mean * mean))


@wp.func
def keeps_distance(
    distance: wp.float32, mean: wp.float32, deviation: wp.float32, trim: wp.float32
) -> wp.bool:
    # The trimming rule: a hit within ``trim`` deviations of the mean is kept.
    return not wp.isinf(distance) and wp.abs(distance - mean) <= trim * deviation


@wp.kernel
def shape_diameter(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    max_t: wp.float32,
    offset: wp.float32,
    trim: wp.float32,
    scratch: wp.array2d[wp.float32],
    out_diameter: wp.array[wp.float32],
) -> None:
    # Shapira et al.'s shape diameter function: a cone of rays *into* the volume, and the
    # outlier-trimmed cosine-weighted mean of the distances they travel before leaving it.
    #
    # The trimming is what makes this robust rather than just a mean: near a concavity a handful of
    # rays escape through the opening or cross the whole model, and those few dominate an untrimmed
    # average. The pass structure is dictated by that -- distances go into ``scratch`` first,
    # because the second pass must revisit them against a mean and deviation the first pass had not
    # finished computing yet, and re-casting the rays instead would double the only expensive part.
    #
    # One block per point, lanes striding the bundle in both passes (see ``BUNDLE_BLOCK``); the
    # per-lane sums are combined block-wide, so every lane holds the same mean and deviation.
    #
    # The ``(n_points, n_rays)`` scratch is not the cost, though it is gigabytes on a large mesh
    # (15 GB at 14 M points and 256 rays): holding each lane's few distances in a register vector
    # instead (one unrolled kernel per rays-per-lane bucket) measured 0.97-0.98x on bunny, dragon
    # and lucy at 64 and 256 rays. Each lane writes and rereads its own contiguous slots, which
    # the cache absorbs; the traced rays are the cost.
    i, t = wp.tid()
    # Inward, so the bundle's axis is ``-normal`` and the origin steps *below* the surface.
    axis, basis_x, basis_y, origin = hemisphere_frame(points, normals, i, offset, wp.float32(-1.0))

    n_rays = directions.shape[0]
    total = wp.float32(0.0)
    total_sq = wp.float32(0.0)
    hits = wp.float32(0.0)
    for r in range(t, n_rays, wp.block_dim()):
        distance = inward_distance(
            mesh_id, origin, bundle_direction(directions[r], axis, basis_x, basis_y), max_t, offset
        )
        if not wp.isinf(distance):
            total += distance
            total_sq += distance * distance
            hits += 1.0
        scratch[i, r] = distance
    # One block reduction of the three sums, so every lane holds the same mean and deviation.
    moments = block_sum(wp.vec3(total, total_sq, hits))
    total = moments[0]
    total_sq = moments[1]
    hits = moments[2]

    if hits == 0.0:
        if t == 0:
            out_diameter[i] = wp.inf  # an open surface with nothing on the other side
        return
    mean, deviation = diameter_window(total, total_sq, hits)

    kept = wp.float32(0.0)
    weighted = wp.float32(0.0)
    for r in range(t, n_rays, wp.block_dim()):
        distance = scratch[i, r]
        if keeps_distance(distance, mean, deviation, trim):
            weight = directions[r][2]  # cosine of the angle from the cone axis
            kept += weight
            weighted += weight * distance
    trimmed = block_sum(wp.vec2(kept, weighted))
    kept = trimmed[0]
    weighted = trimmed[1]
    if t == 0:
        # ``kept <= 0``: every ray trimmed away (only possible at trim = 0), fall back to the mean.
        out_diameter[i] = wp.where(kept <= 0.0, mean, weighted / kept)


@wp.func
def ray_direction(normal: wp.vec3, sign: wp.float32) -> wp.vec3:
    # The unit ray direction ``sign * normalize(normal)``: one map where a normalize and a
    # negation were two, and scaling by ``+-1`` is exact, so the directions are unchanged.
    return sign * wp.normalize(normal)


@wp.func
def sphere_center(point: wp.vec3, normal: wp.vec3, radius: wp.float32) -> wp.vec3:
    if wp.isinf(radius) or wp.isnan(radius):
        return wp.vec3(wp.nan, wp.nan, wp.nan)
    return point + normal * radius


@wp.func
def tangent_sphere_radius(
    point: wp.vec3, normal: wp.vec3, touch: wp.vec3
) -> tuple[wp.float32, wp.bool]:
    # Radius of the sphere tangent at ``point`` (centre along ``normal``) that passes through
    # ``touch``, and whether it exists: the shared rule of the support seed and the shrink step,
    # which reject a vanishing denominator identically and differ only in what they keep instead.
    diff = touch - point
    denom = wp.float32(2.0) * wp.dot(diff, normal)
    if wp.abs(denom) < TOLERANCE_PLANAR_CONSTANT:
        return wp.float32(0.0), False
    return wp.length_sq(diff) / denom, True


@wp.func
def init_sphere_radii_finite(
    distance: wp.float32, point: wp.vec3, direction: wp.vec3
) -> tuple[wp.float32, wp.bool, wp.bool, wp.vec3]:
    # Finite longest-ray hits initialise directly; escaped rays (inf distance) are deferred to
    # the tiled support-point passes below. Their slots default to the "no valid support"
    # outcome so an empty support subset needs no fix-up. The initial centre is written here, at
    # the radius just chosen, and the support pass rewrites it for the slots it resolves -- so no
    # separate ``sphere_center`` map over the whole array is needed.
    radius = wp.float32(wp.inf)
    finite = wp.bool(False)
    if not wp.isinf(distance):
        radius = distance * wp.float32(0.5)
        finite = True
    return radius, finite, not finite, sphere_center(point, direction, radius)


@wp.func
def pack_support_candidate(projection: wp.float32, index: wp.int32) -> wp.uint64:
    # Order-preserving float32 -> uint32 mapping (sign bit set for non-negatives, all bits
    # inverted for negatives) packed above the bit-inverted index, so a single atomic_max
    # selects the greatest projection with the LOWEST index as the tie-break. Zero never
    # occurs as a real packed value, so it doubles as the "no candidate" sentinel.
    bits = wp.cast(projection, wp.uint32)
    if bits & wp.uint32(0x80000000) != wp.uint32(0):
        key = ~bits
    else:
        key = bits | wp.uint32(0x80000000)
    return (wp.uint64(key) << wp.uint64(32)) | wp.uint64(~wp.uint32(index))


# Deferred queries one ``support_argmax_sliced`` thread may reduce together; a launch takes the
# widest ``RegisterBlockedTable.launch_shape`` allows, as ``kernels/points.hull_support_extremes``
# does.
SUPPORT_ARGMAX_WIDTHS = (4, 2, 1)


def _support_argmax_sliced_kernel(width: int) -> wp.Kernel:
    """Build ``support_argmax_sliced`` over ``width`` deferred queries per thread."""
    normals_t = wp.types.matrix(shape=(width, 3), dtype=wp.float32)
    best_t = wp.types.vector(length=width, dtype=wp.float32)
    index_t = wp.types.vector(length=width, dtype=wp.int32)

    def support_argmax_sliced(
        mesh_vertices: wp.array[wp.vec3],
        n_vertices: wp.int32,
        n_slices: wp.int32,
        normals: wp.array[wp.vec3],
        support_indices: wp.array[wp.int32],
        out_packed: wp.array[wp.uint64],
    ) -> None:
        # Support point of the vertex cloud per deferred query: argmax of dot(v, n). One thread per
        # (query, vertex slice) strides over the vertices, reduces its own running best into a
        # packed (projection, index) key, and commits one atomic; the packed key's ordering makes
        # atomic_max the global argmax with the lowest index as tie-break.
        #
        # `_sliced`, not `_tiled`, and the name is the contract: this is launched with a plain
        # `wp.launch` and must stay lane-free, because the threads partition the **outer** work --
        # the vertex cloud -- rather than a sequence one block owns, so there is no `wp.block_dim()`
        # to stride by. On the CPU device, where `wp.launch_tiled` runs one lane per block through
        # Warp 1.18, that lane would cover `1/block_dim` of the slice. See `.claude/CLAUDE.md`
        # section 2.2, and `obscurance` above for the other side of the rule -- one block per point,
        # striding by `wp.block_dim()`, `wp.tile_sum` on both devices.
        #
        # Converting this to one block per deferred query is the same trade
        # `kernels/points.py::hull_support_extremes` records and it is **declined for the same
        # measured reason**: the slice dimension is what fills the device here, so the block form
        # leaves one block per query and loses badly once the cloud is large. `obscurance` above
        # qualified because it had no slice dimension at all.
        #
        # Thread ``(b, j)`` takes queries ``b * width ..`` over vertex slice ``j``: each vertex it
        # loads is projected onto every one of their normals from registers, so the cloud is
        # streamed once per ``width`` queries. Each query keeps its own running best in the same
        # vertex order, so the keys it commits are the one-query form's.
        b, j = wp.tid()
        n_queries = support_indices.shape[0]
        q0 = b * width
        # A slot past the last query repeats it; only the commit below skips it.
        query_normals = normals_t()
        for d in range(width):
            query_normals[d] = normals[support_indices[wp.min(q0 + d, n_queries - 1)]]
        best = best_t(-wp.inf)
        best_index = index_t()
        # A strict `>` already resolves a tie to the lowest index, because `idx` ascends: the first
        # occurrence of a repeated projection is the one that takes `best`, and every later equal
        # one fails the test. An explicit `idx < best_index` arm would be unreachable.
        for idx in range(j, n_vertices, n_slices):
            vertex = mesh_vertices[idx]
            for d in range(width):
                projection = wp.dot(vertex, query_normals[d])
                if projection > best[d]:
                    best[d] = projection
                    best_index[d] = idx
        for d in range(width):
            if q0 + d < n_queries and not wp.isinf(best[d]):
                wp.atomic_max(out_packed, q0 + d, pack_support_candidate(best[d], best_index[d]))

    return wp.kernel(support_argmax_sliced, name=f"support_argmax_sliced_{width}")


SUPPORT_ARGMAX_SLICED = RegisterBlockedTable(
    "support_argmax_sliced", _support_argmax_sliced_kernel, SUPPORT_ARGMAX_WIDTHS
)


@wp.kernel
def init_sphere_radii_support(
    mesh_vertices: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    support_indices: wp.array[wp.int32],
    packed_support: wp.array[wp.uint64],
    out_radii: wp.array[wp.float32],
    out_not_converged: wp.array[wp.bool],
    out_centers: wp.array[wp.vec3],
) -> None:
    # Tail pass over the deferred subset: decode the support point and derive the
    # tangent-sphere radius, scattering it back into the full arrays.
    q = wp.int32(wp.tid())
    tid = support_indices[q]
    packed = packed_support[q]
    p = points[tid]
    n = normals[tid]
    # No support point, or one not above the tangent plane: no finite sphere, and nothing to
    # shrink.
    radius = wp.float32(wp.inf)
    found = wp.bool(False)
    if packed != wp.uint64(0):
        best = wp.int32(~wp.uint32(packed & wp.uint64(0xFFFFFFFF)))
        max_proj = wp.dot(mesh_vertices[best], n) - wp.dot(p, n)
        # `denom` is algebraically `2 * max_proj` (both are `2 * dot(mesh_vertices[best] - p, n)`),
        # so ``tangent_sphere_radius``'s own guard looks redundant against this one -- it is not,
        # on float32: the two are computed by differently-associated expressions
        # (`dot(a, n) - dot(b, n)` here, `dot(a - b, n)` there), so they can disagree by a rounding
        # error this check already cleared. Keep both; do not "simplify" by reusing `max_proj` in
        # place of `denom`.
        if not (max_proj < TOLERANCE_PLANAR_CONSTANT):
            radius, found = tangent_sphere_radius(p, n, mesh_vertices[best])
            radius = wp.where(found, radius, wp.float32(wp.inf))
    out_radii[tid] = radius
    out_not_converged[tid] = found
    out_centers[tid] = sphere_center(p, n, radius)


@wp.kernel
def step_sphere_shrink(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    centers: wp.array[wp.vec3],
    old_radii: wp.array[wp.float32],
    max_t: wp.float32,
    convergence_threshold: wp.float32,
    not_converged: wp.array[wp.bool],
    round_slot: wp.int32,
    out_radii: wp.array[wp.float32],
    out_centers: wp.array[wp.vec3],
    out_not_converged: wp.array[wp.bool],
    out_n_not_converged: wp.array[wp.int32],
) -> None:
    # Every lane writes all three outputs (converged lanes pass their state through), so the
    # wrapper can ping-pong two preallocated buffer sets instead of cloning per iteration, and
    # extra launches on a fully converged state are harmless no-ops.
    #
    # The closest-point query of the current centre runs *here*, after the convergence test, rather
    # than as a ``closest_point_on_mesh`` launch ahead of this one: the step reads that query's
    # answer only at its own index, so the fusion removes a launch per iteration and three
    # ``m``-sized buffers the answer made a round trip through -- and a lane that has already
    # converged no longer pays a BVH query whose answer it would ignore. The query and its miss
    # convention are ``proximity.closest_point_query``'s, so the values are the ones the separate
    # pass wrote.
    #
    # A lane that stops -- already converged, touching the surface, or with no tangent sphere --
    # passes its radius and centre through unchanged; one publication serves every outcome.
    tid = wp.int32(wp.tid())
    p = points[tid]
    radius = old_radii[tid]
    center = centers[tid]
    still_shrinking = wp.bool(False)
    if not_converged[tid]:
        nearest, nearest_distance, _face = closest_point_query(mesh_id, center, max_t)
        dist_to_start = wp.length(center - p)
        if not (wp.abs(nearest_distance - dist_to_start) < TOLERANCE_PLANAR_CONSTANT):
            new_r, found = tangent_sphere_radius(p, normals[tid], nearest)
            if found:
                still_shrinking = radius - new_r >= convergence_threshold
                radius = new_r
                center = sphere_center(p, normals[tid], new_r)
    out_radii[tid] = radius
    out_centers[tid] = center
    out_not_converged[tid] = still_shrinking
    # The next round's convergence count, taken by the kernel that decides it rather than by a
    # reduction launch over ``out_not_converged`` in the wrapper's loop.
    if still_shrinking:
        wp.atomic_add(out_n_not_converged, round_slot, wp.int32(1))
