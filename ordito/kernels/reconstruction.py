"""
Kernels for point-cloud surface reconstruction.

The heart is
[`build_local_triangulations`][ordito.kernels.reconstruction.build_local_triangulations], a
per-point fan triangulation: each point orders its neighbours by angle in the tangent plane and
then improves the fan toward a Delaunay-like one. The geometry helpers below are the standard
triangle predicates, shared with ``kernels/predicates.py``.
"""

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT, PI, TWO_PI
from ordito.kernels.algorithms.conjugate_gradient import (
    cg_publish_round_dots,
    cg_round_terms,
    cg_widen,
)
from ordito.kernels.array import (
    append_slot,
    is_positive_finite,
    ordered_float_bits,
    pack_triangle_key,
    ravel_index,
    scanned_count,
    sort3,
    sorted_run_start,
    to_vec2d,
    trilinear_cell,
    trilinear_corner,
    trilinear_weight,
    update_argmax,
)
from ordito.kernels.predicates import (
    circumcircle_center,
    circumcircle_diameter_sq,
    is_unfold_quadrangle_convex,
    orient2d,
    orient3d_sign_f32,
    project_out_normal,
    triangle_aspect_ratio,
    vector_angle,
)
from ordito.kernels.reduce import (
    block_chunk_1d,
    block_sum,
    device_is_cpu,
    store_sum_and_count,
    term_sum_1d,
    warp_min_at_lead,
)

wp.set_module_options({"enable_backward": False})

# Compile-time upper bound on the per-point fan size (neighbours kept for one center).
# Per-thread scratch arrays are sized to this; the runtime ``max_neighbours`` must not exceed it.
MAX_NEIGHBOURS = 64
# A fan whose search radius grows (``grown_radius``) is rebuilt from up to
# ``max(2 * k, GROWN_NEIGHBOURS_MIN)`` nearest neighbours inside its grown radius.
# ``GROWN_NEIGHBOURS`` sizes that rebuild's scratch rows and holds the largest such count
# ``MAX_NEIGHBOURS`` allows.
GROWN_NEIGHBOURS_MIN = 100
GROWN_NEIGHBOURS = 2 * MAX_NEIGHBOURS

# Faces with aspect ratio above this are treated as degenerate and removed first.
CRITICAL_ASPECT_RATIO = wp.constant(wp.float32(1e3))
# Neighbour filter: drop a neighbour whose oriented normal opposes the center normal.
NORMAL_FILTER_DOT = wp.constant(wp.float32(-0.3))

# ``local_fan``'s cached-weight markers: a slot to recompute, and a stable one. A removal weight is
# a finite sum of bounded terms or ``+inf``, so it is never ``-inf`` nor the lowest finite float.
FAN_WEIGHT_DIRTY = wp.constant(wp.float32(-FLOAT32_INF_CONSTANT))
FAN_WEIGHT_STABLE = wp.constant(wp.float32(-3.4028234663852886e38))

# ``build_local_triangulations``' selector: the normals are trusted (oriented) or carry an
# arbitrary sign per point (unoriented).
NORMALS_UNORIENTED = wp.int32(0)
NORMALS_ORIENTED = wp.int32(1)


# --------------------------------------------------------------------------------------
# Lexicographic incremental triangulation (the Delaunay seed)
# --------------------------------------------------------------------------------------


@wp.kernel
def positive_finite_sum_and_count(
    values: wp.array[wp.float32], stride: wp.int32, out_sum_and_count: wp.array[wp.float64]
) -> None:
    # Numerator and denominator of "the mean over the strictly positive finite entries" in one
    # pass, as one row per block (``store_sum_and_count``) added in a fixed order into the first
    # two slots, which the caller reads once. A neighbour-distance table carries a
    # zero per self-match and an ``inf`` per unfilled slot, and neither belongs in a spacing.
    #
    # ``reduce``'s mask-shaped block fold (``_reduce_bool_1d_tiled``): the lanes stride the block's
    # chunk by ``wp.block_dim()``, which is what keeps it right on the CPU device, and fold their
    # registers with one tile sum per quantity. Both accumulate in ``float64``, where a count is
    # exact at any cloud size.
    i, t = wp.tid()
    n = values.shape[0]
    base, remaining = block_chunk_1d(n, i)
    if remaining <= 0:
        return
    total = wp.float64(0.0)
    count = wp.float64(0.0)
    for k in range(t, remaining, wp.block_dim()):
        value = values[base + k]
        if is_positive_finite(value):
            total += wp.float64(value)
            count += wp.float64(1.0)
    store_sum_and_count(i, t, total, count, out_sum_and_count, stride)


@wp.kernel
def lexicographic_keys(
    points: wp.array[wp.vec2], out_keys: wp.array[wp.uint64], out_order: wp.array[wp.int32]
) -> None:
    # The ``(x, y)`` lexicographic order as one sortable ``uint64`` key per point, written with the
    # identity payload into a radix sort's double-width buffers: the stable sort is then
    # ``numpy.lexsort((y, x))`` exactly, ties included. ``array.ordered_float_bits`` folds ``-0.0``
    # onto ``+0.0`` and every ``NaN`` onto one key above ``+inf``, which is how ``lexsort`` compares
    # them.
    i = wp.int32(wp.tid())
    p = points[i]
    high = wp.uint64(ordered_float_bits(p[0]))
    out_keys[i] = (high << wp.uint64(32)) | wp.uint64(ordered_float_bits(p[1]))
    out_order[i] = i


@wp.func
def hull_slot(start: wp.int32, j: wp.int32, capacity: wp.int32) -> wp.int32:
    # Physical slot of logical hull position ``j`` in ``lexicographic_triangulation``'s ring.
    return (start + j) % capacity


@wp.kernel(enable_backward=False)
def lexicographic_triangulation(
    points: wp.array[wp.vec2],
    order: wp.array[wp.int32],
    max_faces: wp.int32,
    hull: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # One thread, deliberately: this is a hull sweep whose every step depends on the previous
    # boundary, so there is nothing to parallelise. It is launched on the **CPU** device for the
    # same reason -- the identical sweep is two orders of magnitude cheaper on one CPU thread than
    # on one CUDA thread, and cheaper again than the Python loop it replaces. A single GPU thread is
    # the wrong tool and the numbers say so. Positions are widened to ``float64`` as they are read.
    #
    # The boundary is a ring in ``hull`` (scratch, ``2 * (n + 1)`` slots): logical position ``j``
    # lives at ``hull_slot(start, j, capacity)``. Each insertion evaluates every boundary edge's
    # orientation once -- every visibility decision depends on all of them, which is what keeps a
    # degenerate (non-contiguous) visible set handled exactly as an array sweep would -- and in
    # the same pass emits the visible edges' triangles in logical order and finds the visible arc.
    # The kept arc ``right .. left`` then becomes the new boundary by moving ``start`` to
    # ``right``; only when it wraps past logical 0 is its head (``0 .. left``) copied after the
    # tail. In a lex sweep the visible arc surrounds the last inserted vertex, at the logical end,
    # so that copy is rare and an insertion costs one orientation pass.
    #
    # ``out_counts`` carries [faces written, faces wanted]. They differ only when a degenerate input
    # (duplicate or collinear points) drives the visible arc past the 2n bound a real triangulation
    # obeys; the wrapper raises on that rather than letting the writes wrap.
    if wp.tid() != 0:
        return
    n = order.shape[0]
    capacity = hull.shape[0]
    zero = wp.float64(0.0)
    n_faces = wp.int32(0)
    n_boundary = wp.int32(0)
    start = wp.int32(0)

    for i in range(2, n):
        ci = order[i]
        curr = to_vec2d(points[ci])

        if n_boundary == 0:
            # Every point so far is collinear; the first off-line point fans the whole prefix.
            side = orient2d(to_vec2d(points[order[0]]), to_vec2d(points[order[1]]), curr)
            if side != zero:
                for j in range(i - 1):
                    if n_faces < max_faces:
                        first = order[j]
                        second = order[j + 1]
                        if side < zero:
                            first, second = second, first
                        out_faces[3 * n_faces + 0] = first
                        out_faces[3 * n_faces + 1] = second
                        out_faces[3 * n_faces + 2] = ci
                    n_faces += 1
                for j in range(i + 1):
                    # The prefix in lex order, plus ``curr``; reversed when ``curr`` is right of it,
                    # so the boundary comes out counter-clockwise either way.
                    if side > zero:
                        hull[j] = order[j]
                    else:
                        hull[j] = order[i - j]
                n_boundary = i + 1
            continue

        nb = n_boundary
        # Every edge ``curr`` can see becomes a triangle, wound so the new face agrees with the
        # boundary's orientation. The visible edges form one contiguous arc: ``left`` starts it,
        # ``right`` ends it (the first kept vertex). Edge ``j`` runs from logical ``j`` to ``j + 1``
        # (the last one closes the ring), and its predecessor's orientation decides both ends, so
        # the closing edge is evaluated first as edge 0's predecessor.
        first_vertex = hull[hull_slot(start, 0, capacity)]
        last_vertex = hull[hull_slot(start, nb - 1, capacity)]
        closing = orient2d(to_vec2d(points[last_vertex]), to_vec2d(points[first_vertex]), curr)
        previous = closing
        left = wp.int32(-1)
        right = wp.int32(-1)
        a = first_vertex
        for j in range(nb):
            b = first_vertex
            orientation = closing
            if j + 1 < nb:
                b = hull[hull_slot(start, j + 1, capacity)]
                orientation = orient2d(to_vec2d(points[a]), to_vec2d(points[b]), curr)
            if orientation < zero:
                if n_faces < max_faces:
                    out_faces[3 * n_faces + 0] = b
                    out_faces[3 * n_faces + 1] = a
                    out_faces[3 * n_faces + 2] = ci
                n_faces += 1
            if orientation >= zero and previous < zero:
                right = j
            elif orientation < zero and previous >= zero:
                left = j
            previous = orientation
            a = b
        # No visible edge at all means a degenerate insertion (a duplicate point, since a lex sweep
        # always sees the hull from the new rightmost point otherwise). Both indices stay -1, which
        # in the Python original wrapped to the last boundary entry and broke the walk immediately;
        # mapping them to ``nb - 1`` reproduces that exactly rather than reading out of bounds.
        #
        # That collapse also fires -- and drops every other boundary vertex from the hull,
        # unreferenced, rather than only the duplicate -- when ``curr`` is exactly collinear with
        # the whole current boundary. A hand-traced general-position case exists in principle at
        # ``n_boundary == 2``, but a large randomized search over properly lex-sorted point sets
        # never reduced ``n_boundary`` to 2 at all: reaching it seems to require an earlier
        # degenerate step, which is the same input class ``test_delaunay_collinear`` already accepts
        # producing zero faces for. So the dropped vertex ends up in the "unreferenced by any face"
        # outcome that class already produces deliberately. Left as is; revisit only with a concrete
        # reachable repro, not a hand-traced hypothetical one.
        if right < 0:
            right = nb - 1
        if left < 0:
            left = nb - 1

        # Keep the non-visible arc right..left going forward, then insert ``curr`` after it.
        if left < right:
            # The kept arc wraps past logical 0: move its head ``0 .. left`` after the tail.
            for j in range(left + 1):
                hull[hull_slot(start, nb + j, capacity)] = hull[hull_slot(start, j, capacity)]
            left += nb
        hull[hull_slot(start, left + 1, capacity)] = ci
        n_boundary = left - right + 2
        start = hull_slot(start, right, capacity)

    out_counts[0] = wp.min(n_faces, max_faces)
    out_counts[1] = n_faces


# --------------------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------------------


@wp.func
def circumcircle_diameter_sq_of(
    points: wp.array[wp.vec3], i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.float32:
    # A triangle's squared circumdiameter, rounded the same way whichever fan asks: the corners go
    # in index order, not in the asking fan's ring order.
    lo, mid, hi = sort3(i, j, k)
    return circumcircle_diameter_sq(points[lo], points[mid], points[hi])


@wp.func
def delone_flip_profit_sq(
    points: wp.array[wp.vec3], a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32
) -> wp.float32:
    # Empty-circumcircle (Delone) profit of flipping diagonal AC to BD: the metrics of
    # ``predicates.delone_metrics``, over vertex indices instead of positions, so that each triangle
    # is rounded by its vertex set. A quad grid's two diagonals tie exactly, and the rounded
    # profit's sign then picks one: in ring order that sign follows the center normal's (the ring
    # runs the other way round under a flipped one), so two fans differing in it chose different
    # diagonals and agreed on no triangle.
    metric_ac = wp.max(
        circumcircle_diameter_sq_of(points, a, c, d), circumcircle_diameter_sq_of(points, c, a, b)
    )
    metric_bd = wp.max(
        circumcircle_diameter_sq_of(points, b, d, a), circumcircle_diameter_sq_of(points, d, b, c)
    )
    return metric_ac - metric_bd


@wp.func
def tris_angle_profit(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, crit_ang: wp.float32
) -> wp.float32:
    # Dihedral angle across edge AC minus the critical angle.
    ac = c - a
    ab = b - a
    ad = d - a
    dir_abc = wp.cross(ab, ac)
    dir_acd = wp.cross(ac, ad)
    return vector_angle(dir_abc, dir_acd) - crit_ang


@wp.func
def cycle_step(nbr: wp.array[wp.int32], m: wp.int32, i: wp.int32, step: wp.int32) -> wp.int32:
    # Walk the ring of ``m`` fan slots from ``i`` in direction ``step`` until a live neighbour
    # (``nbr[j] >= 0``) is found; return ``i`` if the whole ring is dead. One function with an
    # explicit direction rather than a ``+1`` and a ``-1`` copy that differed only in their wrap
    # guard -- the fan optimizer removes slots as it goes, so both directions have to skip holes
    # and the skipping is the whole body.
    #
    # This is called from `edge_removal_weight` inside `build_local_triangulations`'s O(m) x O(m)
    # greedy removal loop, so an adversarial removal order (long runs of dead slots) makes one call
    # O(m) and the whole loop O(m^3) against the O(m^2) shape it otherwise implies -- an
    # incrementally-maintained doubly-linked `prev`/`next` pair would cap it at O(1)/O(m^2).
    # Measured instead of assumed, sweeping `num_neighbours` at a fixed point count: the kernel's
    # own cost scales with the O(m^2) shape the loop nominally has, not the O(m^3) worst case, so on
    # an ordinary point cloud the walk is not hitting long dead-slot runs and the doubly-linked
    # rewrite would buy little. `m` is capped at `MAX_NEIGHBOURS` regardless, bounding the absolute
    # worst case. Declined; re-measure before reopening this if a future caller's removal pattern
    # looks different (e.g. a very anisotropic or highly clustered cloud). Re-measured once the
    # removal weights were cached (2026-10-27, crown clouds' grown fans of ~55 neighbours): a
    # linked ring took ``rebuild_grown_local_triangulations`` 6.7 -> 6.1 ms (CUDA 13.4) and
    # 20.0 -> 19.2 ms (CUDA 12.9), identical output; still declined for the extra scratch row.
    j = i
    for _ in range(m):
        j = j + step
        if j >= m:
            j = 0
        elif j < 0:
            j = m - 1
        if nbr[j] >= 0:
            return j
    return i


@wp.func
def cycle_next(nbr: wp.array[wp.int32], m: wp.int32, i: wp.int32) -> wp.int32:
    return cycle_step(nbr, m, i, wp.int32(1))


@wp.func
def cycle_prev(nbr: wp.array[wp.int32], m: wp.int32, i: wp.int32) -> wp.int32:
    return cycle_step(nbr, m, i, wp.int32(-1))


# --------------------------------------------------------------------------------------
# Fan edge-removal weight: the priority with which a fan edge is dropped
# --------------------------------------------------------------------------------------
@wp.func
def neighbour_normal(n_center: wp.vec3, n: wp.vec3, oriented: wp.int32) -> wp.vec3:
    # An unoriented normal's sign is arbitrary, so a fan reads it on its center's side: within
    # one fan the normals then agree as trusted ones would, and the fans' windings, which may
    # disagree between centers, are made consistent on the assembled mesh.
    if oriented == NORMALS_UNORIENTED and wp.dot(n_center, n) < 0.0:
        return -n
    return n


@wp.func
def edge_removal_weight(
    center: wp.int32,
    i: wp.int32,
    m: wp.int32,
    border: wp.int32,
    crit_angle: wp.float32,
    normalizer_sq: wp.float32,
    nbr: wp.array[wp.int32],
    ang: wp.array[wp.float32],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    oriented: wp.int32,
) -> wp.vec2:
    # Returns (weight, stable) with stable encoded as result[1] > 0.5.
    stable = wp.vec2(0.0, 1.0)
    prev = cycle_prev(nbr, m, i)
    nxt = cycle_next(nbr, m, i)

    a = points[center]
    n_center = normals[center]

    # --- boundary-edge handling ---
    if border >= 0 and (nbr[i] == border or nbr[prev] == border):
        next_el = nbr[prev] == border
        if next_el:
            prev_ind = i
            next_ind = nxt
            other_id = nxt
        else:
            prev_ind = prev
            next_ind = i
            other_id = prev
        length_sq = wp.length_sq(a - points[nbr[i]])
        other_length_sq = wp.length_sq(a - points[nbr[other_id]])
        if length_sq < other_length_sq:
            return stable
        bb = points[nbr[prev_ind]]
        cc = points[nbr[next_ind]]
        if triangle_aspect_ratio(a, bb, cc) <= CRITICAL_ASPECT_RATIO:
            return stable
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)

    # --- interior-edge handling ---
    dif_angle = ang[nxt] - ang[prev]
    if dif_angle < 0.0:
        dif_angle += TWO_PI
    if dif_angle > PI:
        return stable  # removing this edge would leave an angle > pi

    av = center
    bv = nbr[nxt]
    cv = nbr[i]
    dv = nbr[prev]
    b = points[bv]
    c = points[cv]
    d = points[dv]
    ab = b - a
    ac = c - a
    ad = d - a

    ac_length_sq = wp.length_sq(ac)
    if (
        ac_length_sq > wp.length_sq(ab) and triangle_aspect_ratio(a, b, c) > CRITICAL_ASPECT_RATIO
    ) or (
        ac_length_sq > wp.length_sq(ad) and triangle_aspect_ratio(a, c, d) > CRITICAL_ASPECT_RATIO
    ):
        # degenerate triangle, longest edge -> remove as fast as possible
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)

    # flip possibility: trusted normals allow a flip across a fold, else require convexity
    c_norm = neighbour_normal(n_center, normals[cv], oriented)
    flip = False
    if wp.dot(n_center, c_norm) < 0.0:
        flip = True
    else:
        flip = is_unfold_quadrangle_convex(a, b, c, d)
    if not flip:
        return stable

    delone_prof = delone_flip_profit_sq(points, av, bv, cv, dv)
    if delone_prof == 0.0 and wp.min(av, cv) > wp.min(bv, dv):
        delone_prof = -1.0
    angle_prof = tris_angle_profit(a, b, c, d, crit_angle)
    if delone_prof < 0.0 and angle_prof <= 0.0:
        return stable

    weight = 0.0
    if delone_prof > 0.0:
        weight += delone_prof / normalizer_sq
    if angle_prof > 0.0:
        weight += angle_prof

    # ``ac_length_sq`` is already the squared norm of ``ac`` (computed above for the aspect-ratio
    # guard), so its square root is ``norm_val`` -- no need for a second length computation.
    norm_val = wp.sqrt(ac_length_sq)
    if norm_val == 0.0:
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)
    plane_dist = wp.abs(wp.dot(n_center, ac))
    weight += plane_dist / norm_val

    # trusted-normal agreement bonuses
    weight += 5.0 * (1.0 - wp.dot(n_center, c_norm))
    tri_norm = wp.normalize(wp.cross(ab, ac) + wp.cross(ac, ad))
    tri_norm_weight = wp.dot(tri_norm, c_norm)
    if tri_norm_weight < 0.0:
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)
    weight += 5.0 * (1.0 - tri_norm_weight)

    return wp.vec2(weight, 0.0)


# --------------------------------------------------------------------------------------
# Local fan triangulation: a Boissonnat-style fan about each point, greedily optimized
# --------------------------------------------------------------------------------------
@wp.func
def search_radius(
    neighbor_dist: wp.array2d[wp.float32], row: wp.int32, radius: wp.float32
) -> wp.float32:
    # The radius a fan's neighbours were searched in: the given radius,
    # or the distance to the farthest neighbour found (the ``k``-th nearest).
    if radius > 0.0:
        return radius
    farthest = wp.float32(0.0)
    for i in range(neighbor_dist.shape[1]):
        d = neighbor_dist[row, i]
        if is_positive_finite(d):
            farthest = wp.max(farthest, d)
    return farthest


@wp.func
def grown_radius(
    a: wp.vec3,
    points: wp.array[wp.vec3],
    nbr: wp.array[wp.int32],
    m: wp.int32,
    live: wp.int32,
    border: wp.int32,
    base_radius: wp.float32,
) -> wp.float32:
    # The automatic radius increase of the local triangulation: the radius an optimized
    # fan centred at ``a`` searches its neighbours again in, zero when it keeps ``base_radius``. A
    # fan with a border, or fewer than two live neighbours, doubles it; any other grows to the
    # farthest reach of its triangles' circumcircles, since a point
    # inside one of them would make the fan not Delaunay, capped at the double.
    doubled_radius = base_radius + base_radius
    reach = doubled_radius
    if live >= 2 and border < 0:
        reach = wp.float32(0.0)
        for i in range(m):
            if nbr[i] < 0:
                continue
            nxt = cycle_next(nbr, m, i)
            b = points[nbr[i]]
            c = points[nbr[nxt]]
            diameter_sq = circumcircle_diameter_sq(a, b, c)
            if reach * reach >= diameter_sq:
                continue
            center = circumcircle_center(b - a, c - a)
            reach = wp.max(reach, wp.length(center) + 0.5 * wp.sqrt(diameter_sq))
        reach = wp.min(reach, doubled_radius)
    return wp.where(reach > base_radius, reach, wp.float32(0.0))


@wp.func
def local_fan(
    v: wp.int32,
    row: wp.int32,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    neighbor_idx: wp.array2d[wp.int32],
    neighbor_dist: wp.array2d[wp.float32],
    radius: wp.float32,
    crit_angle: wp.float32,
    boundary_angle: wp.float32,
    oriented: wp.int32,
    nbr: wp.array[wp.int32],
    ang: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    capacity: wp.int32,
    out_tris: wp.array3d[wp.int32],
) -> tuple[wp.int32, wp.float32]:
    # Point ``v``'s fan over the neighbours in row ``row`` of the neighbour table that lie within
    # ``radius`` (any distance when it is zero), into ``out_tris[v, 0 : count]``, with ``nbr``,
    # ``ang`` and ``weights`` the caller's ``capacity``-slot scratch rows. Returns the triangle
    # count and the radius the automatic radius increase grows this fan's search to
    # (``grown_radius``), or zero when it would not grow.
    k = neighbor_idx.shape[1]
    cap = out_tris.shape[1]
    base_radius = search_radius(neighbor_dist, row, radius)

    a = points[v]
    n_center = normals[v]

    # --- gather + filter neighbours ---
    # A constructor call -- wp.int32(...) / wp.float32(...) -- declares a mutable Warp dynamic
    # variable; a bare literal is a compile-time constant that freezes the enclosing loop
    # (the fan count stays zero -> no faces). The constructor is what matters, not which spelling
    # of it: the older int()/float() forms are the same builtins under a different name.
    m = wp.int32(0)
    for i in range(k):
        if m >= capacity:
            # Silent truncation, not a check: this file cannot itself raise
            # if the caller's ``max_neighbours`` violated the module-level invariant. The wrapper's
            # ``max_neighbours > MAX_NEIGHBOURS`` guard is what makes this branch unreachable at
            # the public entry point; a direct launch of the kernels bypasses it.
            break
        nb = neighbor_idx[row, i]
        if nb < 0 or nb == v:
            continue
        if radius > 0.0 and neighbor_dist[row, i] > radius:
            continue
        pn = points[nb]
        if pn[0] == a[0] and pn[1] == a[1] and pn[2] == a[2]:
            continue  # coincident with center
        # Only an oriented normal can oppose the center's: an unoriented one is read on its side.
        if oriented == NORMALS_ORIENTED and wp.dot(n_center, normals[nb]) < NORMAL_FILTER_DOT:
            continue
        nbr[m] = nb
        m += 1

    # A fan with a border, or with fewer than two neighbours, searches again in twice its radius.
    doubled_radius = base_radius + base_radius
    if m < 2:
        return wp.int32(0), doubled_radius

    # --- tangent-plane basis (project neighbours onto plane through center) ---
    base = wp.vec3(0.0, 0.0, 0.0)
    normalizer_sq = wp.float32(0.0)
    for i in range(m):
        d = points[nbr[i]] - a
        pv = project_out_normal(d, n_center)
        if normalizer_sq <= 0.0:
            base = pv
            normalizer_sq = wp.length_sq(pv)
    if normalizer_sq <= 0.0:
        normalizer_sq = 1.0
    base = wp.normalize(base)  # zero-length base normalizes to the zero vector (Warp kEps == 0)

    # --- polar angle of each neighbour around the center in the tangent plane ---
    for i in range(m):
        d = points[nbr[i]] - a
        pv = project_out_normal(d, n_center)
        if wp.length_sq(pv) > 0.0:
            vec = wp.normalize(pv)
        else:
            vec = base
        cp = wp.cross(vec, base)
        # wp.sign is -1 below zero and +1 otherwise, matching the guard this replaces.
        s = wp.sign(wp.dot(cp, n_center))
        ang[i] = wp.atan2(s * wp.length(cp), wp.dot(vec, base))

    # --- sort neighbours by angle (selection sort; m <= MAX_NEIGHBOURS) ---
    for i in range(m):
        mn = i
        for j in range(i + 1, m):
            if ang[j] < ang[mn]:
                mn = j
        if mn != i:
            ta = ang[i]
            ang[i] = ang[mn]
            ang[mn] = ta
            tn = nbr[i]
            nbr[i] = nbr[mn]
            nbr[mn] = tn

    # --- boundary detection: angular gaps wider than boundary_angle ---
    border = wp.int32(-1)
    gap_count = wp.int32(0)
    for i in range(m):
        if i + 1 < m:
            diff = ang[i + 1] - ang[i]
        else:
            diff = ang[0] + TWO_PI - ang[i]
        if diff > boundary_angle:
            gap_count += 1
            if border < 0:
                border = nbr[i]
    if gap_count > 1:
        # Two or more disjoint gaps: this point sits between separate regions of the cloud (a thin
        # bridge/peninsula, or a point equidistant from two disjoint holes). `border` can only ever
        # mark one gap, so the others would read as ordinary interior edges below, and a spurious
        # triangle could bridge the two disconnected regions across an unregistered gap. Rather than
        # guess which one to keep, this point's fan contributes no triangle this round -- a real fix
        # needs the fan to carry more than one boundary marker (or split into multiple arcs), which
        # is a redesign of the boundary representation this function does not attempt. Leaving a
        # point untriangulated is an accepted outcome elsewhere in the reconstruction pipeline (a
        # confirmed triangle needs two of its three vertices' fans to agree, so no single point's
        # fan is required to succeed), and `crit_hole_length` fills the small holes this leaves.
        # Its gaps are borders, so it searches again in twice its radius, as a one-gap fan does.
        return wp.int32(0), doubled_radius

    # --- greedy fan optimisation (linear-scan replacement of the priority queue) ---
    # A slot's weight reads only the slot, its live neighbours on either side and ``border``, so a
    # removal changes exactly the weights of the two slots beside it (``border`` moves only onto
    # the one before it): the weights are cached and those two marked for recomputation, which
    # takes the loop from ``m`` weights a step to two. One call site computes every weight, so
    # the cached values are the ones the full rescan computed, and the argmax ties fall the same.
    for i in range(m):
        weights[i] = FAN_WEIGHT_DIRTY
    current = m  # inherits m's dynamic-variable type (m is already mutable)
    for _step in range(m):
        best_w = wp.float32(-FLOAT32_INF_CONSTANT)  # the constructor keeps this mutable
        best_pos = wp.int32(-1)
        for i in range(m):
            if nbr[i] < 0:
                continue
            if weights[i] == FAN_WEIGHT_DIRTY:
                res = edge_removal_weight(
                    v, i, m, border, crit_angle, normalizer_sq, nbr, ang, points, normals, oriented
                )
                weights[i] = wp.where(res[1] > 0.5, FAN_WEIGHT_STABLE, res[0])
            if weights[i] == FAN_WEIGHT_STABLE:
                continue  # stable, cannot remove
            update_argmax(best_w, best_pos, weights[i], i)
        if best_pos < 0:
            break
        old = nbr[best_pos]
        prv = cycle_prev(nbr, m, best_pos)
        nbr[best_pos] = -1
        current -= 1
        if old == border:
            border = nbr[prv]
        weights[prv] = FAN_WEIGHT_DIRTY
        weights[cycle_next(nbr, m, prv)] = FAN_WEIGHT_DIRTY
        if current < 2:
            for i in range(m):
                nbr[i] = -1
            break

    # --- emit fan triangles between consecutive surviving neighbours ---
    slot = wp.int32(0)
    for i in range(m):
        if nbr[i] < 0:
            continue
        if border >= 0 and nbr[i] == border:
            continue  # boundary gap: no triangle here
        nxt = cycle_next(nbr, m, i)
        if nxt == i:
            continue
        if border < 0 and nbr[i] > nbr[nxt] and cycle_prev(nbr, m, i) == nxt:
            # No boundary gap and exactly two live neighbours remain: the ring is a 2-cycle, so
            # ``prev(i) == next(i)`` and this wedge and (nxt, i)'s wedge describe the identical
            # triangle (there is no third live neighbour to make them distinct). Emit it once, from
            # the lower-indexed neighbour, rather than writing it twice.
            continue
        bidx = nbr[i]
        cidx = nbr[nxt]
        pb = points[bidx]
        pc = points[cidx]
        # orient so the face normal agrees with the center normal
        if wp.dot(wp.cross(pb - a, pc - a), n_center) < 0.0:
            bidx, cidx = cidx, bidx
        if slot < cap:
            out_tris[v, slot, 0] = v
            out_tris[v, slot, 1] = bidx
            out_tris[v, slot, 2] = cidx
            slot += 1
    return slot, grown_radius(a, points, nbr, m, current, border, base_radius)


@wp.kernel(enable_backward=False)
def build_local_triangulations(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    neighbor_idx: wp.array2d[wp.int32],
    neighbor_dist: wp.array2d[wp.float32],
    radius: wp.float32,
    crit_angle: wp.float32,
    boundary_angle: wp.float32,
    oriented: wp.int32,
    out_tris: wp.array3d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_grown_radius: wp.array[wp.float32],
    out_grown_flags: wp.array[wp.int32],
) -> None:
    # Point ``v``'s fan fills ``out_tris[v, 0 : out_counts[v]]`` contiguously, so the emitted
    # candidates are addressed by a scan of ``n`` counts rather than a compaction of an ``(n, k)``
    # mask. Every point reports a count, and the radius its fan grows to (zero when it does not),
    # which [`rebuild_grown_local_triangulations`] rebuilds it from, with its ``0`` / ``1`` flag
    # for the compaction of the grown points. The two kernels run one fan body, ``local_fan``, and
    # differ in their neighbour rows, their scratch size and their radius.
    v = wp.int32(wp.tid())

    # Both rows stay ``wp.zeros`` rather than becoming ``wp.types.vector(length=MAX_NEIGHBOURS)``
    # register rows: every access below is a *runtime* index (the gather cursor, the selection
    # sort's ``mn``, the fan scan's ``i``), which spills a vector to local memory anyway, and
    # ``edge_removal_weight`` / ``cycle_prev`` / ``cycle_next`` take them as ``wp.array``, so the
    # vector form would additionally need ``wp.ref`` variants of all three. Measured on the clean
    # case of the same class -- ``algorithms/ball_pivoting.seed_triangles``, one 64-wide row with no
    # helper passing -- the register row gives nothing end to end, so there is no gain to trade that
    # complexity for.
    #
    # And ``wp.fixedarray`` is not a third option: its own docstring says it is "only used during
    # codegen, and for type hints" -- it *is* the codegen type of a kernel-scope ``wp.zeros``.
    nbr = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.int32)
    ang = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.float32)
    weights = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.float32)
    count, grown = local_fan(
        v,
        v,
        points,
        normals,
        neighbor_idx,
        neighbor_dist,
        radius,
        crit_angle,
        boundary_angle,
        oriented,
        nbr,
        ang,
        weights,
        wp.int32(MAX_NEIGHBOURS),
        out_tris,
    )
    out_counts[v] = count
    out_grown_radius[v] = grown
    out_grown_flags[v] = wp.where(grown != 0.0, wp.int32(1), wp.int32(0))


@wp.kernel(enable_backward=False)
def rebuild_grown_local_triangulations(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    grown_points: wp.array[wp.int32],
    grown_radii: wp.array[wp.float32],
    neighbor_idx: wp.array2d[wp.int32],
    neighbor_dist: wp.array2d[wp.float32],
    crit_angle: wp.float32,
    boundary_angle: wp.float32,
    oriented: wp.int32,
    lanes: wp.int32,
    out_tris: wp.array3d[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The second fan of a point whose first one grew its radius: the same ``local_fan`` over row
    # ``t`` of the grown neighbour table, cut at the grown radius, into the point's own slots.
    # Launched with ``lanes`` threads a fan, of which the first does the work: when the fans are
    # few their serial optimizations, which take different paths, would run one after the other
    # in a warp of fans.
    tid = wp.int32(wp.tid())
    if tid % lanes != 0:
        return
    t = tid // lanes
    v = grown_points[t]
    nbr = wp.zeros(shape=GROWN_NEIGHBOURS, dtype=wp.int32)
    ang = wp.zeros(shape=GROWN_NEIGHBOURS, dtype=wp.float32)
    weights = wp.zeros(shape=GROWN_NEIGHBOURS, dtype=wp.float32)
    count, _grown = local_fan(
        v,
        t,
        points,
        normals,
        neighbor_idx,
        neighbor_dist,
        grown_radii[v],
        crit_angle,
        boundary_angle,
        oriented,
        nbr,
        ang,
        weights,
        wp.int32(GROWN_NEIGHBOURS),
        out_tris,
    )
    out_counts[v] = count


@wp.kernel
def candidate_triangle_keys(
    tris: wp.array3d[wp.int32],
    inclusive_counts: wp.array[wp.int32],
    radix: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_slots: wp.array[wp.int32],
) -> None:
    # Launched over ``(n, k)``: fan slot ``(v, j)`` below point ``v``'s count writes, at its packed
    # position, the unoriented key of its triangle (``array.pack_triangle_key``, the mixed radix
    # ``grouping.hash_indices_rows`` uses, so keys order and collide exactly as that function's
    # would) -- and its flat slot ``v * k + j`` as the sort payload. Slots ascend within
    # the packed order, so a stable sort keeps the lowest slot first in every run of equal keys.
    v, j = wp.tid()
    start, count = scanned_count(inclusive_counts, v)
    if j >= count:
        return
    out_keys[start + j] = pack_triangle_key(tris[v, j, 0], tris[v, j, 1], tris[v, j, 2], radix)
    out_slots[start + j] = v * tris.shape[1] + j


# --------------------------------------------------------------------------------------
# Oriented candidate classes and the contested regions of the serial builder
# --------------------------------------------------------------------------------------
# A triangle three fans agree on in one winding, or two; ``classify_oriented_repeats``' codes.
REPEAT_NONE = wp.constant(0)
REPEAT_THREE = wp.constant(1)
REPEAT_TWO = wp.constant(2)
# A vertex no seed reaches within the rings grown so far.
FAR_FROM_SEEDS = wp.constant(2147483647)


@wp.func
def winding_flipped(a: wp.int32, b: wp.int32, c: wp.int32) -> wp.bool:
    # Whether sorting ``(a, b, c)`` takes an odd number of swaps, the
    # test of a fan triangle's winding against its sorted triple.
    flipped = wp.bool(False)
    if a > b:
        flipped = not flipped
        a, b = b, a
    if a > c:
        flipped = not flipped
        a, c = c, a
    if b > c:
        flipped = not flipped
    return flipped


# --------------------------------------------------------------------------------------
# Orienting unoriented fans: a flood from the farthest points, in rounds
#
# The reference visits the points one at a time, the next always the fan its visited neighbours
# decide by the widest margin. Where no two fans that share a triangle conflict, the result does
# not depend on that order: every fan of a component ends consistent with the component's first,
# and the first follows the same rule here (no positive margin: the largest touched fan, else the
# farthest point). Where fans conflict the order places the seams, and the rounds approach it by
# deciding only the fans the visited ones support by a margin of at least ``min(top, 3)``. A round
# is six launches: the votes, one thread a fan row (``count_flood_votes``); each fan's majority and
# margin (``settle_flood_fans``); the seeds (``pick_touched_seed``, ``pick_farthest_seed``); the
# visits (``choose_flood_visits``) and their neighbours, one thread a row
# (``spread_flood_visits``). Inert fans, which share no triangle, are seeded together: the
# reference visits them one at a time ahead of the next fan that shares one, and their windings
# decide nothing (their touches can, in principle, hand that fan's component a different first
# fan; no measured cloud showed it).
# --------------------------------------------------------------------------------------
@wp.func
def packed_candidate(
    sorted_slots: wp.array[wp.int32],
    inclusive_counts: wp.array[wp.int32],
    fan_width: wp.int32,
    i: wp.int32,
) -> wp.int32:
    # The packed row of sorted candidate ``i``: its fan's first row plus its slot within the fan.
    slot = sorted_slots[i]
    v = slot // fan_width
    start, _count = scanned_count(inclusive_counts, v)
    return start + slot - v * fan_width


@wp.kernel
def pack_flood_candidates(
    sorted_keys: wp.array[wp.uint64],
    sorted_slots: wp.array[wp.int32],
    tris: wp.array2d[wp.int32],
    inclusive_counts: wp.array[wp.int32],
    fan_width: wp.int32,
    out_rows: wp.array2d[wp.int32],
    out_partners: wp.array[wp.vec2i],
    out_codes: wp.array[wp.int32],
) -> None:
    # The flood's candidate table, over sorted position ``i``: each fan slot's triangle at its
    # packed position (a fan's triangles contiguous, its center in column 0), its code (twice the
    # fan plus the row's winding against its sorted triple) and its partners, the packed rows of
    # the other fans carrying the same triangle (at most its two other corners', ``-1`` past them),
    # read off the run of equal unoriented keys. A vote then reads two codes, not the run.
    i = wp.int32(wp.tid())
    slot = sorted_slots[i]
    packed = packed_candidate(sorted_slots, inclusive_counts, fan_width, i)
    for c in range(3):
        out_rows[packed, c] = tris[slot, c]
    wound = winding_flipped(tris[slot, 0], tris[slot, 1], tris[slot, 2])
    out_codes[packed] = 2 * (slot // fan_width) + wp.where(wound, 1, 0)
    first = i
    while not sorted_run_start(sorted_keys, first):
        first -= 1
    partners = wp.vec2i(-1, -1)
    found = wp.int32(0)
    j = first
    while j < sorted_slots.shape[0] and found < 2:
        if j != first and sorted_run_start(sorted_keys, j):
            break
        if j != i:
            partners[found] = packed_candidate(sorted_slots, inclusive_counts, fan_width, j)
            found += 1
        j += 1
    out_partners[packed] = partners


# The flood decides in bulk every fan whose margin reaches ``min(top, FLOOD_BULK_MARGIN)``, ``top``
# the largest margin of the round: a fan decided on one neighbour's two votes copies that
# neighbour's errors, and waiting for a third vote keeps the seams near the serial flood's.
FLOOD_BULK_MARGIN = wp.int32(3)
# The flood's state words: the cursor into the farthest-first order, the going flag (``1`` until
# no fan is left, the condition of the round loop), the first unvisited position past the cursor
# (CUDA, arriving the order's length), then per round parity a slot of the round's largest margin,
# its largest touched fan that shares a triangle (the seed), its frontier's count and its largest
# touched inert fan (one that shares no triangle with any other).
FLOOD_CURSOR = 0
FLOOD_GOING = 1
FLOOD_FARTHEST = 2
FLOOD_SLOTS = 3
FLOOD_SLOT_WORDS = 4
FLOOD_TOP = 0
FLOOD_SEED = 1
FLOOD_COUNT = 2
FLOOD_INERT = 3

# The seed priority of a point: its squared distance from ``center`` less ``limit``, each product
# and sum rounded on its own (no FMA contraction on CUDA), summed ``x``, ``y``, ``z``, so both
# devices order the points alike.
_UNFUSED_FARTHEST_PRIORITY = """
#if defined(__CUDA_ARCH__)
float x = __fsub_rn(p[0], center[0]);
float y = __fsub_rn(p[1], center[1]);
float z = __fsub_rn(p[2], center[2]);
return __fsub_rn(__fadd_rn(__fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y)), __fmul_rn(z, z)), limit);
#else
float x = p[0] - center[0];
float y = p[1] - center[1];
float z = p[2] - center[2];
float xx = x * x;
float yy = y * y;
float zz = z * z;
float xy = xx + yy;
float sum = xy + zz;
return sum - limit;
#endif
"""


@wp.func_native(_UNFUSED_FARTHEST_PRIORITY)
def unfused_farthest_priority(p: wp.vec3, center: wp.vec3, limit: wp.float32) -> wp.float32: ...


@wp.kernel
def start_fan_flood(
    points: wp.array[wp.vec3],
    rows: wp.array2d[wp.int32],
    inclusive_counts: wp.array[wp.int32],
    partners: wp.array[wp.vec2i],
    center: wp.vec3,
    max_distance_sq: wp.float32,
    out_flipped: wp.array[wp.int32],
    out_visited: wp.array[wp.int32],
    out_touched: wp.array[wp.int32],
    out_inert: wp.array[wp.int32],
    out_far_keys: wp.array[wp.uint64],
    out_far_order: wp.array[wp.int32],
) -> None:
    # Fan ``v`` faces away from ``center`` by a majority of its triangles (the exact sign of each
    # triangle's normal against the direction from ``center``), is flagged inert when no other fan
    # carries any of its triangles (its winding then decides nothing), and gets its key in the
    # farthest-first seed order: the priority's ordered bits above the index, complemented so that
    # an ascending sort puts the largest first, an empty fan last. An empty fan starts visited.
    v = wp.int32(wp.tid())
    start, count = scanned_count(inclusive_counts, v)
    pv = points[v]
    balance = wp.int32(0)
    inert = wp.int32(1)
    for j in range(count):
        row = start + j
        # ``dot(pv - center, (pb - pv) x (pc - pv))`` is ``-det(pb - pv, pc - pv, center - pv)``.
        balance -= orient3d_sign_f32(pv, points[rows[row, 1]], points[rows[row, 2]], center)
        if partners[row][0] >= 0:
            inert = 0
    out_inert[v] = inert
    out_flipped[v] = wp.where(balance < 0, 1, 0)
    out_visited[v] = wp.where(count == 0, 1, 0)
    out_touched[v] = 0
    key = ~wp.uint64(0)
    if count > 0:
        priority = unfused_farthest_priority(pv, center, max_distance_sq)
        key = ~((wp.uint64(ordered_float_bits(priority)) << wp.uint64(32)) | wp.uint64(v))
    out_far_keys[v] = key
    out_far_order[v] = v


@wp.func
def row_vote(
    row: wp.int32,
    partners: wp.array[wp.vec2i],
    codes: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    flipped: wp.array[wp.int32],
) -> wp.int32:
    # The visited fans' vote on row ``row``'s fan from its triangle: ``0`` to keep the fan's
    # winding, ``1`` to flip it, ``-1`` when no visited fan carries the triangle or they disagree.
    pair = partners[row]
    same = wp.int32(0)
    opposite = wp.int32(0)
    for t in range(2):
        other = pair[t]
        if other >= 0:
            code = codes[other]
            w = code // 2
            if visited[w] != 0:
                if (code % 2) != flipped[w]:
                    opposite += 1
                else:
                    same += 1
    vote = wp.int32(-1)
    if (same > 0) != (opposite > 0):
        agreed = wp.where(same > 0, 0, 1)
        vote = wp.where(codes[row] % 2 == agreed, 0, 1)
    return vote


@wp.kernel
def count_flood_votes(
    inclusive_counts: wp.array[wp.int32],
    partners: wp.array[wp.vec2i],
    codes: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    flipped: wp.array[wp.int32],
    stale: wp.array[wp.int32],
    frontier: wp.array[wp.int32],
    fan_width: wp.int32,
    slot: wp.int32,
    lanes: wp.int32,
    state: wp.array[wp.int32],
    votes: wp.array[wp.int32],
) -> None:
    # The votes on every stale frontier fan (one a visit has touched since it was last weighed),
    # one thread a row: the visited fans carrying a row's triangle, when they agree, vote to keep
    # the fan's winding (``votes[2 u]``) or to flip it (``votes[2 u + 1]``), arriving zero. The
    # ``lanes`` threads stride over the frontier's ``(fan, row)`` pairs.
    base = FLOOD_SLOTS + FLOOD_SLOT_WORDS * slot
    items = state[base + FLOOD_COUNT] * fan_width
    e = wp.int32(wp.tid())
    while e < items:
        u = frontier[slot * visited.shape[0] + e // fan_width]
        j = e % fan_width
        e += lanes
        if visited[u] != 0 or stale[u] == 0:
            continue
        start, count = scanned_count(inclusive_counts, u)
        if j >= count:
            continue
        vote = row_vote(start + j, partners, codes, visited, flipped)
        if vote >= 0:
            wp.atomic_add(votes, 2 * u + vote, 1)


@wp.kernel
def settle_flood_fans(
    visited: wp.array[wp.int32],
    frontier: wp.array[wp.int32],
    slot: wp.int32,
    lanes: wp.int32,
    flipped: wp.array[wp.int32],
    stale: wp.array[wp.int32],
    votes: wp.array[wp.int32],
    margin: wp.array[wp.int32],
    state: wp.array[wp.int32],
) -> None:
    # Every stale frontier fan turns to its votes' majority (a tie keeps its winding) and takes the
    # margin, ``-1`` without votes, its counts cleared; the others keep theirs, since only a visit
    # changes a fan's votes. The round's largest margin gathers in its state slot (arriving
    # ``0``). The ``lanes`` threads stride over the frontier.
    base = FLOOD_SLOTS + FLOOD_SLOT_WORDS * slot
    count = state[base + FLOOD_COUNT]
    i = wp.int32(wp.tid())
    while i < count:
        u = frontier[slot * visited.shape[0] + i]
        i += lanes
        if visited[u] != 0:
            continue
        if stale[u] != 0:
            stale[u] = 0
            keep = votes[2 * u]
            flip = votes[2 * u + 1]
            votes[2 * u] = 0
            votes[2 * u + 1] = 0
            weight = wp.int32(-1)
            if keep + flip > 0:
                if flip > keep:
                    flipped[u] = 1
                elif keep > flip:
                    flipped[u] = 0
                weight = wp.abs(keep - flip)
            margin[u] = weight
        if margin[u] > 0:
            wp.atomic_max(state, base + FLOOD_TOP, margin[u])


@wp.kernel
def pick_touched_seed(
    visited: wp.array[wp.int32],
    inert: wp.array[wp.int32],
    frontier: wp.array[wp.int32],
    far_order: wp.array[wp.int32],
    slot: wp.int32,
    lanes: wp.int32,
    state: wp.array[wp.int32],
) -> None:
    # A round with no positive margin seeds instead: the largest unvisited frontier fan (its votes,
    # if any, tie) into the slot's seed word, an inert one into its inert word (both arriving
    # ``-1``). On CUDA it also finds the first unvisited position of the farthest-first order past
    # the cursor (one ``atomic_min`` a warp), which ``pick_farthest_seed`` takes when no fan is
    # touched; the CPU device walks the cursor there instead. The ``lanes`` threads stride over
    # the frontier, then over the order.
    base = FLOOD_SLOTS + FLOOD_SLOT_WORDS * slot
    if state[base + FLOOD_TOP] > 0:
        return
    count = state[base + FLOOD_COUNT]
    i = wp.int32(wp.tid())
    while i < count:
        u = frontier[slot * visited.shape[0] + i]
        if visited[u] == 0:
            word = wp.where(inert[u] != 0, FLOOD_INERT, FLOOD_SEED)
            wp.atomic_max(state, base + word, u)
        i += lanes
    if device_is_cpu() != 0:
        return
    cursor = state[FLOOD_CURSOR]
    position = cursor + wp.int32(wp.tid())
    while position < far_order.shape[0]:
        candidate = wp.int32(2147483647)
        if visited[far_order[position]] == 0:
            candidate = position
        first = warp_min_at_lead(candidate)
        if first < far_order.shape[0]:
            wp.atomic_min(state, FLOOD_FARTHEST, first)
        position += lanes
        # The found position only falls, so one below this thread's next ends its search.
        if position > state[FLOOD_FARTHEST]:
            break


@wp.kernel
def pick_farthest_seed(
    far_order: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    slot: wp.int32,
    state: wp.array[wp.int32],
) -> None:
    # One thread. Resets the other parity's slot for the next round, which alone reads it (its
    # frontier count then gathers this round's appends). A seeding round no fan touches takes the
    # farthest unvisited fan: the first unvisited position of the farthest-first order past the
    # cursor, which ``pick_touched_seed`` found on CUDA and the CPU device walks to; none left ends
    # the flood. Visits only grow, so the cursor moves there.
    base = FLOOD_SLOTS + FLOOD_SLOT_WORDS * slot
    other = FLOOD_SLOTS + FLOOD_SLOT_WORDS * (1 - slot)
    state[other + FLOOD_TOP] = 0
    state[other + FLOOD_SEED] = -1
    state[other + FLOOD_COUNT] = 0
    state[other + FLOOD_INERT] = -1
    n = far_order.shape[0]
    first = state[FLOOD_FARTHEST]
    state[FLOOD_FARTHEST] = n
    if state[base + FLOOD_TOP] > 0 or state[base + FLOOD_SEED] >= 0:
        return
    if state[base + FLOOD_INERT] >= 0:
        return
    if device_is_cpu() != 0:
        first = state[FLOOD_CURSOR]
        while first < n:
            if visited[far_order[first]] == 0:
                break
            first += 1
    state[FLOOD_CURSOR] = first
    if first == n:
        state[FLOOD_GOING] = 0
    else:
        state[base + FLOOD_SEED] = far_order[first]


@wp.func
def append_frontier(
    state: wp.array[wp.int32],
    next_count: wp.int32,
    frontier: wp.array[wp.int32],
    at: wp.int32,
    u: wp.int32,
) -> None:
    frontier[at + append_slot(state, next_count)] = u


@wp.func
def touch_flood_neighbour(
    x: wp.int32,
    visited: wp.array[wp.int32],
    touched: wp.array[wp.int32],
    stale: wp.array[wp.int32],
    state: wp.array[wp.int32],
    next_count: wp.int32,
    frontier: wp.array[wp.int32],
    at: wp.int32,
) -> None:
    # A visited fan's neighbour ``x``: unvisited, it turns stale, and joins the next frontier the
    # first time any visit touches it.
    if visited[x] == 0:
        stale[x] = 1
        if wp.atomic_max(touched, x, 1) == 0:
            append_frontier(state, next_count, frontier, at, x)


@wp.kernel
def choose_flood_visits(
    rows: wp.array2d[wp.int32],
    inclusive_counts: wp.array[wp.int32],
    margin: wp.array[wp.int32],
    inert: wp.array[wp.int32],
    slot: wp.int32,
    lanes: wp.int32,
    state: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    touched: wp.array[wp.int32],
    stale: wp.array[wp.int32],
    chosen: wp.array[wp.int32],
    frontier: wp.array[wp.int32],
) -> None:
    # The round's visits: every frontier fan whose margin reaches ``min(top, FLOOD_BULK_MARGIN)``;
    # with no positive margin, every inert frontier fan above the seed when there is one (the
    # reference visits them one by one before the seed, and their windings decide nothing), else
    # the seed. A visited fan's winding is final, and ``chosen[i]`` marks frontier entry ``i``'s
    # visit for ``spread_flood_visits``; an undecided frontier fan carries over to the next
    # frontier (``append_slot``, one atomic a warp). Thread 0 visits a seed that is not on the
    # frontier (the farthest point) and touches its neighbours itself. The appends' order decides
    # nothing: every decision is a fan's own, or a maximum. The ``lanes`` threads stride over the
    # frontier.
    n = visited.shape[0]
    base = FLOOD_SLOTS + FLOOD_SLOT_WORDS * slot
    next_count = FLOOD_SLOTS + FLOOD_SLOT_WORDS * (1 - slot) + FLOOD_COUNT
    at = (1 - slot) * n
    count = state[base + FLOOD_COUNT]
    top = state[base + FLOOD_TOP]
    seed = state[base + FLOOD_SEED]
    # The inert fans the reference would visit before the seed: every one above it at once.
    inert_batch = state[base + FLOOD_INERT] > seed
    i = wp.int32(wp.tid())
    if i == 0 and top <= 0 and not inert_batch and seed >= 0:
        if touched[seed] == 0:
            visited[seed] = 1
            start, fan_count = scanned_count(inclusive_counts, seed)
            for j in range(fan_count):
                for c in range(1, 3):
                    touch_flood_neighbour(
                        rows[start + j, c], visited, touched, stale, state, next_count, frontier, at
                    )
    while i < count:
        u = frontier[slot * n + i]
        visit = wp.bool(False)
        if visited[u] == 0:
            if top > 0:
                visit = margin[u] >= wp.min(top, FLOOD_BULK_MARGIN)
            elif inert_batch:
                visit = inert[u] != 0 and u > seed
            else:
                visit = u == seed
            if visit:
                visited[u] = 1
            else:
                append_frontier(state, next_count, frontier, at, u)
        chosen[i] = wp.where(visit, 1, 0)
        i += lanes


@wp.kernel
def spread_flood_visits(
    rows: wp.array2d[wp.int32],
    inclusive_counts: wp.array[wp.int32],
    chosen: wp.array[wp.int32],
    fan_width: wp.int32,
    slot: wp.int32,
    lanes: wp.int32,
    state: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    touched: wp.array[wp.int32],
    stale: wp.array[wp.int32],
    frontier: wp.array[wp.int32],
) -> None:
    # The neighbours of the round's visited frontier fans, one thread a row: each touched as
    # ``touch_flood_neighbour`` does. The ``lanes`` threads stride over the frontier's
    # ``(fan, row)`` pairs.
    n = visited.shape[0]
    base = FLOOD_SLOTS + FLOOD_SLOT_WORDS * slot
    next_count = FLOOD_SLOTS + FLOOD_SLOT_WORDS * (1 - slot) + FLOOD_COUNT
    at = (1 - slot) * n
    items = state[base + FLOOD_COUNT] * fan_width
    e = wp.int32(wp.tid())
    while e < items:
        i = e // fan_width
        j = e % fan_width
        e += lanes
        if chosen[i] == 0:
            continue
        u = frontier[slot * n + i]
        start, count = scanned_count(inclusive_counts, u)
        if j >= count:
            continue
        for c in range(1, 3):
            touch_flood_neighbour(
                rows[start + j, c], visited, touched, stale, state, next_count, frontier, at
            )


@wp.kernel
def flip_fan_windings(
    flipped: wp.array[wp.int32], inclusive_counts: wp.array[wp.int32], tris: wp.array3d[wp.int32]
) -> None:
    # Reverse every triangle of a fan the flood flipped, in place.
    v, j = wp.tid()
    _start, count = scanned_count(inclusive_counts, v)
    if j >= count or flipped[v] == 0:
        return
    b = tris[v, j, 1]
    tris[v, j, 1] = tris[v, j, 2]
    tris[v, j, 2] = b


@wp.kernel
def classify_oriented_repeats(
    sorted_keys: wp.array[wp.uint64],
    sorted_slots: wp.array[wp.int32],
    tris: wp.array2d[wp.int32],
    n: wp.int32,
    out_flags: wp.array[wp.int32],
    out_codes: wp.array[wp.int32],
) -> None:
    # The classes at the start of each run of equal unoriented keys:
    # three fans in the sorted triple's winding, or three in the other, is a ``t3`` triangle in
    # that winding; else two in one winding a ``t2`` one. One fan, or two disagreeing, is neither.
    # ``out_codes`` holds the class times two plus the flip bit, ``out_flags`` the scan's ones.
    i = wp.int32(wp.tid())
    code = wp.int32(0)
    if sorted_run_start(sorted_keys, i):
        key = sorted_keys[i]
        same = wp.int32(0)
        opposite = wp.int32(0)
        for j in range(3):
            if i + j < n:
                if sorted_keys[i + j] == key:
                    slot = sorted_slots[i + j]
                    if winding_flipped(tris[slot, 0], tris[slot, 1], tris[slot, 2]):
                        opposite += 1
                    else:
                        same += 1
        if same == 3:
            code = 2 * REPEAT_THREE
        elif opposite == 3:
            code = 2 * REPEAT_THREE + 1
        elif same == 2:
            code = 2 * REPEAT_TWO
        elif opposite == 2:
            code = 2 * REPEAT_TWO + 1
    out_codes[i] = code
    out_flags[i] = wp.where(code != 0, 1, 0)


@wp.kernel
def emit_oriented_repeats(
    inclusive_flags: wp.array[wp.int32],
    codes: wp.array[wp.int32],
    sorted_slots: wp.array[wp.int32],
    tris: wp.array2d[wp.int32],
    radix: wp.uint64,
    out_rows: wp.array2d[wp.int32],
    out_low_keys: wp.array[wp.uint64],
    out_high_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_rep3: wp.array[wp.int32],
) -> None:
    # Each classified triangle stored lowest vertex first: ``(a, b, c)`` sorted,
    # or ``(a, c, b)`` in the other winding. Its keys order the rows as the builder adds them,
    # the ``t3`` triangles before the ``t2`` ones and each class lexicographically: the second and
    # third vertex in the low key, the class and the first vertex in the high one, for two stable
    # sorts. ``out_rep3`` (arriving zeroed) counts the ``t3`` rows.
    i = wp.int32(wp.tid())
    row, flag = scanned_count(inclusive_flags, i)
    if flag == 0:
        return
    code = codes[i]
    slot = sorted_slots[i]
    a, b, c = sort3(tris[slot, 0], tris[slot, 1], tris[slot, 2])
    if code % 2 == 1:
        b, c = c, b
    out_rows[row, 0] = a
    out_rows[row, 1] = b
    out_rows[row, 2] = c
    group = code // 2 - REPEAT_THREE
    out_low_keys[row] = wp.uint64(b) * radix + wp.uint64(c)
    out_high_keys[row] = wp.uint64(group) * radix + wp.uint64(a)
    out_order[row] = row
    if group == 0:
        wp.atomic_add(out_rep3, 0, 1)


@wp.kernel
def directed_edge_keys(
    rows: wp.array2d[wp.int32],
    radix: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_rows: wp.array[wp.int32],
) -> None:
    # Launched over ``(n_rows, 3)``: corner ``k``'s directed edge to the next corner, keyed
    # ``tail * radix + head``, with its row as the payload.
    f, k = wp.tid()
    m = rows.shape[0]
    out_keys[k * m + f] = wp.uint64(rows[f, k]) * radix + wp.uint64(rows[f, (k + 1) % 3])
    out_rows[k * m + f] = f


@wp.kernel
def mark_shared_directed_edges(
    sorted_keys: wp.array[wp.uint64],
    sorted_rows: wp.array[wp.int32],
    n: wp.int32,
    out_conflict: wp.array[wp.int32],
) -> None:
    # A row with a directed edge another row has too: the builder keeps at most one of them.
    # ``out_conflict`` arrives zeroed; every writer writes 1.
    i = wp.int32(wp.tid())
    key = sorted_keys[i]
    shared = wp.bool(False)
    if i > 0:
        if sorted_keys[i - 1] == key:
            shared = True
    if i + 1 < n:
        if sorted_keys[i + 1] == key:
            shared = True
    if shared:
        out_conflict[sorted_rows[i]] = 1


@wp.func
def star_link(rows: wp.array2d[wp.int32], f: wp.int32, v: wp.int32) -> wp.vec2i:
    # The edge of row ``f`` opposite ``v``, in the row's winding seen from ``v``.
    k = wp.int32(0)
    if rows[f, 1] == v:
        k = 1
    elif rows[f, 2] == v:
        k = 2
    return wp.vec2i(rows[f, (k + 1) % 3], rows[f, (k + 2) % 3])


@wp.kernel
def classify_candidate_stars(
    rows: wp.array2d[wp.int32],
    vertex_rows: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    conflict: wp.array[wp.int32],
    out_distance: wp.array[wp.int32],
    out_hole_next: wp.array[wp.int32],
) -> None:
    # One thread per vertex. Its candidate rows are *clean* when none shares a directed edge and
    # their opposite edges chain into one path or one cycle: one fan, which the builder adds
    # whole whatever the order. Any other vertex seeds a contested region (distance 0). A clean
    # path's last vertex is where the hole leaves the vertex (its next hole vertex).
    v = wp.int32(wp.tid())
    start = offsets[v]
    degree = offsets[v + 1] - start
    out_hole_next[v] = -1
    out_distance[v] = FAR_FROM_SEEDS
    if degree == 0:
        return
    seed = wp.bool(False)
    for j in range(degree):
        if conflict[vertex_rows[start + j]] != 0:
            seed = True
    first = wp.int32(0)
    starts = wp.int32(0)
    tail = wp.int32(-1)
    if not seed:
        for j in range(degree):
            link = star_link(rows, vertex_rows[start + j], v)
            entered = wp.bool(False)
            left = wp.bool(False)
            for m in range(degree):
                other = star_link(rows, vertex_rows[start + m], v)
                if other[1] == link[0]:
                    entered = True
                if other[0] == link[1]:
                    left = True
            if not entered:
                starts += 1
                first = j
            if not left:
                tail = link[1]
        if starts > 1:
            seed = True
    if not seed:
        # Walk the chain from its start (or from any row of a cycle) and count the rows it meets.
        current = star_link(rows, vertex_rows[start + first], v)
        steps = wp.int32(1)
        for _ in range(degree):
            found = wp.int32(-1)
            for m in range(degree):
                other = star_link(rows, vertex_rows[start + m], v)
                if other[0] == current[1]:
                    found = m
            if found < 0 or found == first:
                break
            current = star_link(rows, vertex_rows[start + found], v)
            steps += 1
        if steps != degree:
            seed = True
    if seed:
        out_distance[v] = 0
    elif starts == 1:
        out_hole_next[v] = tail


@wp.kernel
def grow_seed_rings(
    rows: wp.array2d[wp.int32], ring: wp.int32, distance: wp.array[wp.int32]
) -> None:
    # One ring further from the seeds: a row touching ring ``ring - 1`` puts its other corners at
    # ``ring`` at most. In place; a corner is lowered by ``atomic_min`` only.
    f = wp.int32(wp.tid())
    nearest = wp.min(wp.min(distance[rows[f, 0]], distance[rows[f, 1]]), distance[rows[f, 2]])
    if nearest == ring - 1:
        for k in range(3):
            wp.atomic_min(distance, rows[f, k], ring)


@wp.kernel
def mark_contested_rows(
    rows: wp.array2d[wp.int32],
    distance: wp.array[wp.int32],
    rings: wp.int32,
    out_row_flags: wp.array[wp.int32],
    out_vertex_flags: wp.array[wp.int32],
) -> None:
    # The rows within ``rings`` of a seed, and every vertex they touch. ``out_vertex_flags``
    # arrives zeroed.
    f = wp.int32(wp.tid())
    nearest = wp.min(wp.min(distance[rows[f, 0]], distance[rows[f, 1]]), distance[rows[f, 2]])
    contested = nearest <= rings
    out_row_flags[f] = wp.where(contested, 1, 0)
    if contested:
        for k in range(3):
            out_vertex_flags[rows[f, k]] = 1


@wp.kernel
def localize_contested_vertices(
    used: wp.array[wp.int32],
    distance: wp.array[wp.int32],
    rings: wp.int32,
    out_global_to_local: wp.array[wp.int32],
    out_complete: wp.array[wp.int32],
) -> None:
    # Renumber the contested rows' vertices from 0 in ascending order, and flag those whose whole
    # star is among the rows (within ``rings`` of a seed). ``out_global_to_local`` arrives ``-1``.
    i = wp.int32(wp.tid())
    v = used[i]
    out_global_to_local[v] = i
    out_complete[i] = wp.where(distance[v] <= rings, 1, 0)


@wp.kernel
def localize_rows(
    contested: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    global_to_local: wp.array[wp.int32],
    out_local_rows: wp.array2d[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    f = contested[i]
    for k in range(3):
        out_local_rows[i, k] = global_to_local[rows[f, k]]


@wp.kernel
def apply_builder_state(
    contested: wp.array[wp.int32],
    state: wp.array[wp.int32],
    added: wp.int32,
    out_keep: wp.array[wp.int32],
) -> None:
    # ``out_keep`` arrives all ones: a contested row stays only where the builder added it.
    i = wp.int32(wp.tid())
    out_keep[contested[i]] = wp.where(state[i] == added, 1, 0)


# ======================================================================================
# Screened-Poisson dense-grid kernels (Kazhdan PoissonRecon-style, index-space h = 1)
#
# A node-centered dense grid of ``res**3`` scalar nodes is stored flat (row-major
# ``(i * res + j) * res + k``). The pipeline: (1) trilinear-splat the oriented normals into
# a vector field ``V`` with a density weight ``W``; (2) normalize ``V /= max(W, eps)``;
# (3) build the RHS ``b = -div V`` by central differences; (4) solve
# ``(L_N + point_weight * W) x = b`` matrix-free (7-point homogeneous-Neumann Laplacian plus a
# lumped screening diagonal); (5) sample ``x`` at the input points for the iso-value. The screening
# diagonal is strictly positive at occupied nodes, so the operator is SPD (no constant null space).
# Everything is index-space (grid spacing 1); the world scale is reapplied by marching cubes.
# ======================================================================================
POISSON_WEIGHT_EPS = wp.constant(wp.float32(1e-8))


@wp.func
def poisson_grid_index(i: wp.int32, j: wp.int32, k: wp.int32, res: wp.int32) -> wp.int32:
    # Row-major flat index into a res**3 node grid (cube: ny = nz = res).
    return ravel_index(i, j, k, res, res)


@wp.func
def poisson_sample_grid(
    field: wp.array[wp.float32], res: wp.int32, gx: wp.float32, gy: wp.float32, gz: wp.float32
) -> wp.float32:
    # Trilinear interpolation of ``field`` at grid coordinate (gx, gy, gz) in [0, res - 1].
    #
    # The cell decomposition is shared with ``splat_normals`` below and with
    # ``kernels/interpolation.sample_grid_trilinear`` / ``kernels/scatter.splat_grid_trilinear``.
    # What is *not* shared is the evaluation: this reads a **flat** ``res**3`` float32 buffer and
    # nests ``wp.lerp``, where the public pair reads a rank-3 array of any dtype and sums the eight
    # corner weights. The two are the same function of the same inputs and differ only in float32
    # summation order, so they are deliberately not merged: this one's arithmetic is what the
    # Poisson iso-value was measured against, and the storage differs anyway.
    base, next_corner, fractions = trilinear_cell(wp.vec3(gx, gy, gz), wp.vec3i(res, res, res))
    i0, j0, k0 = base[0], base[1], base[2]
    # The far corner comes from ``trilinear_cell`` rather than being written ``i0 + 1`` here, so a
    # ``res == 1`` grid cannot index past this flat ``res**3`` buffer. Identical to ``i0 + 1`` at
    # every ``res >= 2``, which is every resolution the Poisson solver builds.
    i1, j1, k1 = next_corner[0], next_corner[1], next_corner[2]
    fx, fy, fz = fractions[0], fractions[1], fractions[2]
    c000 = field[poisson_grid_index(i0, j0, k0, res)]
    c100 = field[poisson_grid_index(i1, j0, k0, res)]
    c010 = field[poisson_grid_index(i0, j1, k0, res)]
    c110 = field[poisson_grid_index(i1, j1, k0, res)]
    c001 = field[poisson_grid_index(i0, j0, k1, res)]
    c101 = field[poisson_grid_index(i1, j0, k1, res)]
    c011 = field[poisson_grid_index(i0, j1, k1, res)]
    c111 = field[poisson_grid_index(i1, j1, k1, res)]
    c00 = wp.lerp(c000, c100, fx)
    c10 = wp.lerp(c010, c110, fx)
    c01 = wp.lerp(c001, c101, fx)
    c11 = wp.lerp(c011, c111, fx)
    c0 = wp.lerp(c00, c10, fy)
    c1 = wp.lerp(c01, c11, fy)
    return wp.lerp(c0, c1, fz)


@wp.func
def splat_cell(
    points: wp.array[wp.vec3], lower: wp.vec3, inv_cell: wp.float32, res: wp.int32, s: wp.int32
) -> tuple[wp.vec3i, wp.vec3i, wp.vec3]:
    # Sample ``s``'s lattice cell (``trilinear_cell``): its two corners and the fractions.
    g = (points[s] - lower) * inv_cell
    return trilinear_cell(g, wp.vec3i(res, res, res))


@wp.func
def splat_sample(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    confidence: wp.int32,
    s: wp.int32,
) -> tuple[wp.vec3i, wp.vec3i, wp.vec3, wp.vec3, wp.float32]:
    # Everything sample ``s`` splats: its cell (``splat_cell``), its unit direction and its weight.
    # Shared by the arrival-order splats (``splat_normals``, ``poisson_band.band_splat``) and the
    # fixed-order one (``splat_term``), so the two forms add the same terms.
    n = normals[s]
    length = wp.length(n)
    # Confidence weighting scales the splat by the normal magnitude; otherwise unit weight.
    weight = wp.float32(1.0)
    if confidence != 0:
        weight = length
    # Unit direction, reusing ``length`` instead of ``wp.normalize``'s own redundant sqrt; a zero
    # vector's length is already zero, so it is left unchanged -- the same zero-vector result
    # ``wp.normalize`` gives that input (magnitude is carried separately by ``weight``).
    if length > 0.0:
        n = n / length
    base, next_corner, fractions = splat_cell(points, lower, inv_cell, res, s)
    return base, next_corner, fractions, n, weight


@wp.func
def splat_corner_offset(corner: wp.int32) -> wp.vec3i:
    # The 0/1 offsets of a cell's ``corner``-th corner, ``corner = 4 di + 2 dj + dk``: the order the
    # arrival-order splats' three nested loops visit them in.
    return wp.vec3i(corner >> 2, (corner >> 1) & 1, corner & 1)


@wp.kernel(enable_backward=False)
def splat_normals(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    confidence: wp.int32,
    out_vx: wp.array[wp.float32],
    out_vy: wp.array[wp.float32],
    out_vz: wp.array[wp.float32],
    out_w: wp.array[wp.float32],
) -> None:
    # The oriented normals into the ``res**3`` node grid, trilinearly, in arrival order: the
    # deterministic setting's form on CUDA is ``splat_corner_keys`` and ``sum_splat_chunks``.
    s = wp.int32(wp.tid())
    base, next_corner, fractions, n, weight = splat_sample(
        points, normals, cube_lower, inv_cell, res, confidence, s
    )
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = trilinear_weight(fractions, di, dj, dk) * weight
                corner = trilinear_corner(base, next_corner, wp.vec3i(di, dj, dk))
                idx = poisson_grid_index(corner[0], corner[1], corner[2], res)
                wp.atomic_add(out_vx, idx, w * n[0])
                wp.atomic_add(out_vy, idx, w * n[1])
                wp.atomic_add(out_vz, idx, w * n[2])
                wp.atomic_add(out_w, idx, w)


@wp.kernel(enable_backward=False)
def splat_corner_keys(
    points: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    out_keys: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # ``splat_normals``' eight targets per sample as sort keys, the payload ``8 s + corner`` naming
    # the term, for ``sum_splat_chunks`` and ``sum_splat_runs`` to add each node's terms in the one
    # order a stable sort lists them in.
    # ``poisson_band.band_splat_keys`` is the same for a band's storage.
    s = wp.int32(wp.tid())
    base, next_corner, _fractions = splat_cell(points, cube_lower, inv_cell, res, s)
    for c in range(8):
        corner = trilinear_corner(base, next_corner, splat_corner_offset(c))
        out_keys[8 * s + c] = poisson_grid_index(corner[0], corner[1], corner[2], res)
        out_order[8 * s + c] = 8 * s + c


@wp.func
def splat_term(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    confidence: wp.int32,
    term: wp.int32,
) -> wp.vec4:
    # Term ``8 s + corner`` of the splat (``splat_corner_keys``): the ``V`` and ``W`` sample ``s``
    # adds to its cell's ``corner``-th node, the values ``splat_normals`` adds.
    _base, _next, fractions, n, weight = splat_sample(
        points, normals, lower, inv_cell, res, confidence, term >> 3
    )
    offset = splat_corner_offset(term & 7)
    w = trilinear_weight(fractions, offset[0], offset[1], offset[2]) * weight
    return wp.vec4(w * n[0], w * n[1], w * n[2], w)


@wp.func
def store_splat(
    key: wp.int32,
    total: wp.vec4,
    out_vx: wp.array[wp.float32],
    out_vy: wp.array[wp.float32],
    out_vz: wp.array[wp.float32],
    out_w: wp.array[wp.float32],
) -> None:
    # A node's finished sums; a key at or past the outputs' length is a sentinel and stores nothing.
    if key < out_w.shape[0]:
        out_vx[key] = total[0]
        out_vy[key] = total[1]
        out_vz[key] = total[2]
        out_w[key] = total[3]


@wp.kernel(enable_backward=False)
def sum_splat_chunks(
    sorted_keys: wp.array[wp.int32],
    order: wp.array[wp.int32],
    count: wp.int32,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    confidence: wp.int32,
    chunk: wp.int32,
    out_heads: wp.array[wp.vec4],
    out_tails: wp.array[wp.vec4],
    out_vx: wp.array[wp.float32],
    out_vy: wp.array[wp.float32],
    out_vz: wp.array[wp.float32],
    out_w: wp.array[wp.float32],
) -> None:
    # The fixed-order splat's first pass (dense grid or band), one thread per ``chunk``
    # consecutive stably sorted terms: it adds each node's piece of the chunk in sort order. A run
    # starting and ending inside the chunk is a node's whole sum and is stored; the chunk's first
    # piece is kept in ``out_heads`` and a run starting in it that continues past its end in
    # ``out_tails``, for ``sum_splat_runs`` to add up. Every sum has one fixed order, so the nodes
    # repeat bit for bit where ``splat_normals``' atomics commit in arrival order. A node no term
    # reaches keeps its zero. The chunks split a long run of one node's terms (a coarse level's
    # node sees thousands of samples) across threads: one thread a run serialised it, at 37x the
    # atomic splat's time on ``dragon`` at ``res = 9``. The exact group sum (``kernels/reduce.py``,
    # no sort) was priced against this: 0.5x the wall time at 100 k points (``res = 64``) and
    # 0.6x on a coarse ``res = 16`` lattice, but 1.5-1.8x at 2 M points on ``res = 256``, whose
    # ``int64`` words are twice the bytes to clear, scatter into and read back, so this form stays.
    k = wp.int32(wp.tid())
    start = k * chunk
    end = wp.min(start + chunk, count)
    key = sorted_keys[start]
    piece = start
    total = wp.vec4(0.0)
    for j in range(start, end):
        if sorted_keys[j] != key:
            if piece == start:
                out_heads[k] = total
            if sorted_run_start(sorted_keys, piece):
                store_splat(key, total, out_vx, out_vy, out_vz, out_w)
            key = sorted_keys[j]
            piece = j
            total = wp.vec4(0.0)
        total += splat_term(points, normals, lower, inv_cell, res, confidence, order[j])
    if piece == start:
        out_heads[k] = total
    if sorted_run_start(sorted_keys, piece):
        if end < count and sorted_keys[end] == key:
            out_tails[k] = total
        else:
            store_splat(key, total, out_vx, out_vy, out_vz, out_w)


@wp.kernel(enable_backward=False)
def sum_splat_runs(
    sorted_keys: wp.array[wp.int32],
    count: wp.int32,
    chunk: wp.int32,
    heads: wp.array[wp.vec4],
    tails: wp.array[wp.vec4],
    out_vx: wp.array[wp.float32],
    out_vy: wp.array[wp.float32],
    out_vz: wp.array[wp.float32],
    out_w: wp.array[wp.float32],
) -> None:
    # The second pass: the chunk where a run crossing a chunk boundary starts adds its tail and the
    # head piece of every later chunk the run reaches, in chunk order, and stores the node's sums.
    k = wp.int32(wp.tid())
    start = k * chunk
    end = start + chunk
    if end >= count:
        return
    key = sorted_keys[end - 1]
    if sorted_keys[end] != key or (start > 0 and sorted_keys[start - 1] == key):
        return
    total = tails[k]
    m = k + 1
    while m < heads.shape[0]:
        total += heads[m]
        m += 1
        if m * chunk >= count:
            break
        if sorted_keys[m * chunk] != key:
            break
    store_splat(key, total, out_vx, out_vy, out_vz, out_w)


@wp.func
def normalized_field_component(
    field: wp.array[wp.float32], weights: wp.array[wp.float32], idx: wp.int32
) -> wp.float32:
    # One splatted vector-field component divided by its own density weight -- the per-element
    # operation `normalize_vector_field` used to do as a separate pass. Kept as a `@wp.func` (not a
    # standalone kernel) so `poisson_level_setup` can read a normalized value at each of its six
    # neighbour offsets directly, rather than reading back a value a prior launch wrote: see the
    # measurement in ``poisson_level_setup`` below.
    inv = 1.0 / wp.max(weights[idx], POISSON_WEIGHT_EPS)
    return field[idx] * inv


@wp.func
def poisson_neighbor_degree(i: wp.int32, j: wp.int32, k: wp.int32, res: wp.int32) -> wp.float32:
    # Number of (i, j, k)'s up-to-6 axis-aligned grid neighbours that lie inside [0, res) -- the
    # diagonal degree of the homogeneous-Neumann 7-point Poisson stencil. Shared so
    # ``screened_laplacian_row``'s operator and ``poisson_level_setup``'s Jacobi
    # preconditioner can never disagree about which grid nodes are boundary nodes.
    deg = wp.float32(0.0)
    if i + 1 < res:
        deg += 1.0
    if i - 1 >= 0:
        deg += 1.0
    if j + 1 < res:
        deg += 1.0
    if j - 1 >= 0:
        deg += 1.0
    if k + 1 < res:
        deg += 1.0
    if k - 1 >= 0:
        deg += 1.0
    return deg


@wp.func
def screened_laplacian_scaled_inverse_diagonal(
    lap: wp.float32,
    weight: wp.float32,
    screen: wp.float32,
    omega: wp.float32,
    res: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.float32:
    # ``omega / A_ii`` for ``screened_laplacian_row``'s operator at a node of density ``weight``,
    # ``A_ii`` taken as 1 where it is not positive: the damped-Jacobi scaling of every V-cycle level
    # -- the solve's own grid (``poisson_level_setup``, ``lap = 1``) and each coarser one
    # (``poisson_mg_coarsen``). Stored with the damping folded in because every reader multiplies by
    # ``omega * (1 / A_ii)`` anyway, so the stored product is the value it formed.
    d = lap * poisson_neighbor_degree(i, j, k, res) + screen * weight
    if d <= 0.0:
        d = 1.0
    return omega * (1.0 / d)


@wp.func
def screened_laplacian_row(
    lap: wp.float32,
    x: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    screen: wp.float32,
    res: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.float32:
    # Row ``(i, j, k)`` of ``A x`` for ``A = L_N + screen * diag(W)``, the homogeneous-Neumann
    # 7-point stencil plus the screening term, with the stencil scaled by ``lap``: 1 on the grid the
    # solve is posed on, ``2 ** level`` on the multigrid preconditioner's coarser grids (see
    # ``poisson_mg_coarsen``). Shared by the solve's initial residual
    # (``poisson_cg_initial``), its rounds' mat-vec (``poisson_cg_matvec_dots``) and the V-cycle's
    # smoother and residual, so none of them can apply a different operator.
    idx = poisson_grid_index(i, j, k, res)
    xc = x[idx]
    deg = poisson_neighbor_degree(i, j, k, res)
    # Same boundary test as ``poisson_neighbor_degree``, expanded to also gather the live
    # neighbours' values -- the six bounds checks are free next to the six global loads they guard.
    acc = wp.float32(0.0)
    if i + 1 < res:
        acc += x[poisson_grid_index(i + 1, j, k, res)]
    if i - 1 >= 0:
        acc += x[poisson_grid_index(i - 1, j, k, res)]
    if j + 1 < res:
        acc += x[poisson_grid_index(i, j + 1, k, res)]
    if j - 1 >= 0:
        acc += x[poisson_grid_index(i, j - 1, k, res)]
    if k + 1 < res:
        acc += x[poisson_grid_index(i, j, k + 1, res)]
    if k - 1 >= 0:
        acc += x[poisson_grid_index(i, j, k - 1, res)]
    return lap * (deg * xc - acc) + screen * weights[idx] * xc


@wp.func
def poisson_grid_coordinates(index: wp.int32, res: wp.int32) -> wp.vec3i:
    # The inverse of ``poisson_grid_index``: row-major, ``k`` fastest.
    return wp.vec3i(index // (res * res), (index // res) % res, index % res)


@wp.kernel(enable_backward=False)
def poisson_cg_initial(
    n: wp.int32,
    span: wp.int32,
    res: wp.int32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    rhs: wp.array[wp.float32],
    x: wp.array[wp.float32],
    smoother: wp.array[wp.float32],
    out_r: wp.array[wp.float32],
    out_u: wp.array[wp.float32],
    out_p: wp.array[wp.float32],
    out_s: wp.array[wp.float32],
    out_start: wp.array[wp.float32],
    out_partials: wp.array3d[wp.float64],
) -> None:
    # ``kernels/algorithms/conjugate_gradient.cg_initial`` with the screened Laplacian applied
    # matrix-free: ``r = b - A x`` from the warm start, ``p = s = 0`` and the first stage of
    # ``||b||^2`` in ``float64``, over one column padded to whole blocks of ``span`` entries. A CSR
    # of this stencil would be seven entries a node, and at ``2 ** 8 + 1`` nodes a side that is
    # more memory than the whole solve.
    #
    # Where ``cg_initial`` writes ``u = D^-1 r``, this writes the preconditioner's first step
    # instead: the V-cycle's zero-start pre-smoothing sweep ``smoother * r`` (``smoother`` being
    # ``screened_laplacian_scaled_inverse_diagonal``), which is the sweep's own arithmetic on the
    # stored ``float32`` residual. The V-cycle then writes ``u``'s live rows; only its pad is
    # written here, as the zero every reduction over it assumes.
    blk, t = wp.tid()
    acc = wp.float64(0.0)
    for kk in range(t, span, wp.block_dim()):
        local = blk * span + kk
        b = wp.float64(0.0)
        r = wp.float32(0.0)
        if local < n:
            g = poisson_grid_coordinates(local, res)
            b = wp.float64(rhs[local])
            r = wp.float32(
                b
                - wp.float64(
                    screened_laplacian_row(
                        wp.float32(1.0), x, weights, screen, res, g[0], g[1], g[2]
                    )
                )
            )
            out_start[local] = smoother[local] * r
        else:
            out_u[local] = wp.float32(0.0)
        out_r[local] = r
        out_p[local] = wp.float32(0.0)
        out_s[local] = wp.float32(0.0)
        acc += b * b
    total = block_sum(acc)
    if t == 0:
        out_partials[0, 0, blk] = total


@wp.kernel(enable_backward=False)
def poisson_cg_matvec_dots(
    n: wp.int32,
    span: wp.int32,
    res: wp.int32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    r: wp.array[wp.float32],
    u: wp.array[wp.float32],
    gamma_new: wp.array[wp.float64],
    alpha_new: wp.array[wp.float64],
    out_w: wp.array[wp.float32],
    out_partials: wp.array3d[wp.float64],
    out_gamma_old: wp.array[wp.float64],
    out_alpha_old: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # ``conjugate_gradient.cg_matvec_dots`` with the screened Laplacian applied matrix-free, for
    # the dense Poisson grid's one column: ``w = A u`` and the round's three dots
    # (``cg_round_terms``), then the shared tail (``cg_publish_round_dots``). ``cg_update`` is
    # shared outright, at its ``float32`` overload.
    blk, t = wp.tid()
    acc = wp.vec3(0.0, 0.0, 0.0)
    for kk in range(t, span, wp.block_dim()):
        local = blk * span + kk
        w = wp.float32(0.0)
        if local < n:
            g = poisson_grid_coordinates(local, res)
            w = screened_laplacian_row(wp.float32(1.0), u, weights, screen, res, g[0], g[1], g[2])
        out_w[local] = w
        acc += cg_round_terms(r[local], u[local], w)
    total = cg_widen(block_sum(acc))
    cg_publish_round_dots(
        total,
        0,
        blk,
        t,
        gamma_new,
        alpha_new,
        out_partials,
        out_gamma_old,
        out_alpha_old,
        out_state,
    )


@wp.kernel(enable_backward=False)
def poisson_level_setup(
    vx: wp.array[wp.float32],
    vy: wp.array[wp.float32],
    vz: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    screen: wp.float32,
    res: wp.int32,
    omega: wp.float32,
    out_b: wp.array[wp.float32],
    out_smoother: wp.array[wp.float32],
) -> None:
    # The two per-node passes a solve level needs, in one launch: the right-hand side and the
    # V-cycle's damped-Jacobi scaling on this grid (``screened_laplacian_scaled_inverse_diagonal``).
    # They are independent -- neither reads the other -- and both already read ``weights`` at this
    # node, so fusing removes a launch over the whole ``res**3`` grid and shares the node index. A
    # level is solved once per octree depth, so this launch pair ran several times per call.
    i, j, k = wp.tid()
    # Central differences in index space; one-sided at the grid boundary (denominator 1 there).
    # That denominator is 0, not 1, at res == 1 (every axis's clamp collapses to ip == im), but
    # reconstruction.screened_poisson validates 3 <= full_depth <= depth <= 10, so the smallest
    # reachable grid is res = 2**3 + 1 = 9 and no caller in the tree reaches res == 1. Not guarded
    # here (no speculative generality) -- add a guard only if a future
    # caller can legitimately reach res == 1.
    ip = wp.min(i + 1, res - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, res - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, res - 1)
    km = wp.max(k - 1, 0)
    # Normalized in place here rather than by a separate `normalize_vector_field` pass over the
    # whole grid: that pass was a few percent of ``screened_poisson``'s device time, almost all of
    # it the extra launch plus a full res**3-element write-back of vx/vy/vz that this function
    # immediately reads back at six neighbour offsets anyway. Fusing removes one launch and that
    # write-back, at the cost of reading `weights` at the same six offsets instead of once per node.
    # `normalized_field_component` reproduces the deleted pass's exact
    # `field[idx] * (1.0 / max(weight, eps))` operation (a multiply by the reciprocal, not a divide)
    # so the result is bit-identical to computing it as a separate pass first.
    dx = (
        normalized_field_component(vx, weights, poisson_grid_index(ip, j, k, res))
        - normalized_field_component(vx, weights, poisson_grid_index(im, j, k, res))
    ) / wp.float32(ip - im)
    dy = (
        normalized_field_component(vy, weights, poisson_grid_index(i, jp, k, res))
        - normalized_field_component(vy, weights, poisson_grid_index(i, jm, k, res))
    ) / wp.float32(jp - jm)
    dz = (
        normalized_field_component(vz, weights, poisson_grid_index(i, j, kp, res))
        - normalized_field_component(vz, weights, poisson_grid_index(i, j, km, res))
    ) / wp.float32(kp - km)
    centre = poisson_grid_index(i, j, k, res)
    out_b[centre] = -(dx + dy + dz)

    out_smoother[centre] = screened_laplacian_scaled_inverse_diagonal(
        wp.float32(1.0), weights[centre], screen, omega, res, i, j, k
    )


@wp.func
def poisson_prolong_node(
    coarse: wp.array[wp.float32], res_c: wp.int32, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.float32:
    # ``(P coarse)`` at fine node ``(i, j, k)``, trilinear factor-2 prolongation:
    # ``res_f - 1 == 2 * (res_c - 1)``, so fine node ``i`` sits at coarse coordinate ``i / 2``.
    # Shared by the cascade's warm start (``prolong_grid``) and the V-cycle's coarse-grid
    # correction (``poisson_mg_prolong_add``), whose restriction is its exact adjoint.
    return poisson_sample_grid(
        coarse, res_c, wp.float32(i) * 0.5, wp.float32(j) * 0.5, wp.float32(k) * 0.5
    )


@wp.kernel(enable_backward=False)
def prolong_grid(
    coarse: wp.array[wp.float32], res_c: wp.int32, res_f: wp.int32, out_fine: wp.array[wp.float32]
) -> None:
    # The cascade's warm start: the coarser level's solution prolonged onto this level's grid.
    i, j, k = wp.tid()
    out_fine[poisson_grid_index(i, j, k, res_f)] = poisson_prolong_node(coarse, res_c, i, j, k)


# ---------------------------------------------------------------------------
# Geometric multigrid V-cycle for the dense grid's screened Poisson solve
# ---------------------------------------------------------------------------
#
# The preconditioner ``reconstruction._PoissonMultigrid`` applies inside the conjugate gradient, on
# the nested node grids ``res_{l+1} = (res_l - 1) / 2 + 1``. Level ``l``'s operator is
# ``2 ** l * L_l + screen * W_l`` with ``W_{l+1} = P^T W_l``: the Galerkin product ``P^T A P`` of
# the 7-point index-space Laplacian under trilinear ``P`` is a 27-point stencil whose action on
# smooth modes tends to twice the coarse 7-point one (and ``P^T diag(W) P`` has row sums ``P^T W``),
# so this is that product rediscretized -- spectrally within a factor of four of it, which is all a
# preconditioner needs, and still matrix-free. Restriction is ``P^T`` and the smoother damped
# Jacobi, the same number of sweeps before and after, which keeps the cycle symmetric.


@wp.func
def poisson_restrict_node(
    fine: wp.array[wp.float32], res_f: wp.int32, ci: wp.int32, cj: wp.int32, ck: wp.int32
) -> wp.float32:
    # ``(P^T fine)`` at coarse node ``(ci, cj, ck)``: the exact adjoint of
    # ``poisson_prolong_node``. The coarse node sits on fine node ``2 (ci, cj, ck)`` and gathers the
    # ``3^3`` fine nodes around it at weight ``1`` per axis on the node itself and ``1/2`` either
    # side, skipping those outside the grid -- exactly the fine nodes whose interpolation reads it.
    acc = wp.float32(0.0)
    for di in range(-1, 2):
        fi = 2 * ci + di
        if fi >= 0 and fi < res_f:
            wi = wp.where(di == 0, wp.float32(1.0), wp.float32(0.5))
            for dj in range(-1, 2):
                fj = 2 * cj + dj
                if fj >= 0 and fj < res_f:
                    wij = wi * wp.where(dj == 0, wp.float32(1.0), wp.float32(0.5))
                    for dk in range(-1, 2):
                        fk = 2 * ck + dk
                        if fk >= 0 and fk < res_f:
                            wk = wp.where(dk == 0, wp.float32(1.0), wp.float32(0.5))
                            acc += wij * wk * fine[poisson_grid_index(fi, fj, fk, res_f)]
    return acc


@wp.kernel(enable_backward=False)
def poisson_mg_coarsen(
    weights: wp.array[wp.float32],
    res_f: wp.int32,
    res_c: wp.int32,
    lap: wp.float32,
    screen: wp.float32,
    omega: wp.float32,
    out_weights: wp.array[wp.float32],
    out_smoother: wp.array[wp.float32],
) -> None:
    # One V-cycle level's operator from the level above: ``W_c = P^T W_f`` and the level's
    # damped-Jacobi scaling, which reads only the node's own coarse weight -- so one launch.
    ci, cj, ck = wp.tid()
    idx = poisson_grid_index(ci, cj, ck, res_c)
    weight = poisson_restrict_node(weights, res_f, ci, cj, ck)
    out_weights[idx] = weight
    out_smoother[idx] = screened_laplacian_scaled_inverse_diagonal(
        lap, weight, screen, omega, res_c, ci, cj, ck
    )


@wp.kernel(enable_backward=False)
def poisson_mg_restrict(
    fine: wp.array[wp.float32],
    res_f: wp.int32,
    res_c: wp.int32,
    smoother: wp.array[wp.float32],
    out_coarse: wp.array[wp.float32],
    out_start: wp.array[wp.float32],
) -> None:
    # ``P^T fine`` into the coarse level's right-hand side, and that level's zero-start
    # pre-smoothing sweep ``smoother * b`` from the same register: every level's cycle starts from
    # a zero guess, where a sweep reads no neighbour, so the sweep is pointwise and needs no launch
    # of its own.
    ci, cj, ck = wp.tid()
    idx = poisson_grid_index(ci, cj, ck, res_c)
    value = poisson_restrict_node(fine, res_f, ci, cj, ck)
    out_coarse[idx] = value
    out_start[idx] = smoother[idx] * value


@wp.kernel(enable_backward=False)
def poisson_mg_prolong_add(
    coarse: wp.array[wp.float32],
    res_c: wp.int32,
    res_f: wp.int32,
    fine: wp.array[wp.float32],
    out_fine: wp.array[wp.float32],
) -> None:
    # ``out = fine + P coarse``, the coarse-grid correction. ``out_fine`` may be ``fine`` itself:
    # each thread reads and writes only its own node.
    i, j, k = wp.tid()
    idx = poisson_grid_index(i, j, k, res_f)
    out_fine[idx] = fine[idx] + poisson_prolong_node(coarse, res_c, i, j, k)


@wp.kernel(enable_backward=False)
def poisson_mg_smooth(
    res: wp.int32,
    lap: wp.float32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    smoother: wp.array[wp.float32],
    b: wp.array[wp.float32],
    x: wp.array[wp.float32],
    out_x: wp.array[wp.float32],
) -> None:
    # One damped-Jacobi sweep ``out = x + omega D^-1 (b - A x)`` at one V-cycle level, out of place
    # because a sweep reads every neighbour's previous value. ``smoother`` is ``omega D^-1``. The
    # zero-start sweep every level opens with is not this kernel: it reads no neighbour, so it rides
    # in whichever launch produced the level's right-hand side (``poisson_mg_restrict``,
    # ``poisson_cg_initial`` and the conjugate gradient's ``cg_update``).
    i, j, k = wp.tid()
    idx = poisson_grid_index(i, j, k, res)
    ax = screened_laplacian_row(lap, x, weights, screen, res, i, j, k)
    out_x[idx] = x[idx] + smoother[idx] * (b[idx] - ax)


@wp.kernel(enable_backward=False)
def poisson_mg_residual(
    res: wp.int32,
    lap: wp.float32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    b: wp.array[wp.float32],
    x: wp.array[wp.float32],
    out_r: wp.array[wp.float32],
) -> None:
    # ``b - A x`` at one V-cycle level, the residual the next level corrects.
    i, j, k = wp.tid()
    idx = poisson_grid_index(i, j, k, res)
    out_r[idx] = b[idx] - screened_laplacian_row(lap, x, weights, screen, res, i, j, k)


@wp.func
def sample_field_at_point(
    field: wp.array[wp.float32],
    res: wp.int32,
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    point: wp.vec3,
) -> wp.float32:
    # The ``res**3`` node field at a world-space point: its grid coordinate, trilinearly.
    g = (point - cube_lower) * inv_cell
    return poisson_sample_grid(field, res, g[0], g[1], g[2])


@wp.kernel(enable_backward=False)
def sample_field_trilinear(
    field: wp.array[wp.float32],
    res: wp.int32,
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    points: wp.array[wp.vec3],
    out_values: wp.array[wp.float32],
) -> None:
    # The field at every sample, for the confidence-weighted iso-value; the uniform one sums
    # ``sample_field_at_point`` without storing it (``FIELD_SAMPLE_SUM``).
    s = wp.int32(wp.tid())
    out_values[s] = sample_field_at_point(field, res, cube_lower, inv_cell, points[s])


@wp.struct
class FieldSampleTerms:
    """The samples and the node field ``FIELD_SAMPLE_SUM`` sums ``sample_field_at_point`` over."""

    n: wp.int32
    field: wp.array[wp.float32]
    res: wp.int32
    cube_lower: wp.vec3
    inv_cell: wp.float32
    points: wp.array[wp.vec3]


@wp.func
def field_sample_term(local: wp.int32, offset: wp.int32, terms: FieldSampleTerms) -> wp.float32:
    # The field at sample ``offset + local``, the term of the Poisson iso-value's mean.
    return sample_field_at_point(
        terms.field, terms.res, terms.cube_lower, terms.inv_cell, terms.points[offset + local]
    )


# The uniform iso-value's sum, bit for bit ``reduce.sum`` of what ``sample_field_trilinear`` writes.
FIELD_SAMPLE_SUM = term_sum_1d(field_sample_term, FieldSampleTerms, "poisson_field_sample_sum")
