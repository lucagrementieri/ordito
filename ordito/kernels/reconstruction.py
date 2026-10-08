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
    is_positive_finite,
    ordered_float_bits,
    pack_triangle_key,
    ravel_index,
    scanned_count,
    sorted_run_start,
    to_vec2d,
    trilinear_cell,
    trilinear_corner,
    trilinear_weight,
    update_argmax,
)
from ordito.kernels.predicates import (
    delone_metrics,
    is_unfold_quadrangle_convex,
    orient2d,
    project_out_normal,
    triangle_aspect_ratio,
    vector_angle,
)
from ordito.kernels.reduce import block_chunk_1d, block_sum, commit_sum_and_count

wp.set_module_options({"enable_backward": False})

# Compile-time upper bound on the per-point fan size (neighbours kept for one center).
# Per-thread scratch arrays are sized to this; the runtime ``max_neighbours`` must not exceed it.
MAX_NEIGHBOURS = 64

# Faces with aspect ratio above this are treated as degenerate and removed first.
CRITICAL_ASPECT_RATIO = wp.constant(wp.float32(1e3))
# Neighbour filter: drop a neighbour whose oriented normal opposes the center normal.
NORMAL_FILTER_DOT = wp.constant(wp.float32(-0.3))


# --------------------------------------------------------------------------------------
# Lexicographic incremental triangulation (the Delaunay seed)
# --------------------------------------------------------------------------------------


@wp.kernel
def positive_finite_sum_and_count(
    values: wp.array[wp.float32], out_sum_and_count: wp.array[wp.float64]
) -> None:
    # Numerator and denominator of "the mean over the strictly positive finite entries" in one
    # pass, into one two-slot buffer the caller reads once. A neighbour-distance table carries a
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
    commit_sum_and_count(t, total, count, out_sum_and_count)


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
def delone_flip_profit_sq(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> wp.float32:
    # Empty-circumcircle (Delone) profit of flipping diagonal AC to BD.
    metric_ac, metric_bd = delone_metrics(a, b, c, d)
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
    # looks different (e.g. a very anisotropic or highly clustered cloud).
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
    flip = False
    if wp.dot(n_center, normals[cv]) < 0.0:
        flip = True
    else:
        flip = is_unfold_quadrangle_convex(a, b, c, d)
    if not flip:
        return stable

    delone_prof = delone_flip_profit_sq(a, b, c, d)
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
    c_norm = normals[cv]
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
@wp.kernel(enable_backward=False)
def build_local_triangulations(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    neighbor_idx: wp.array2d[wp.int32],
    neighbor_dist: wp.array2d[wp.float32],
    radius: wp.float32,
    crit_angle: wp.float32,
    boundary_angle: wp.float32,
    out_tris: wp.array3d[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Point ``v``'s fan fills ``out_tris[v, 0 : out_counts[v]]`` contiguously, so the emitted
    # candidates are addressed by a scan of ``n`` counts rather than a compaction of an ``(n, k)``
    # mask. The count is written before any early return, so every point reports one.
    v = wp.int32(wp.tid())
    k = neighbor_idx.shape[1]
    cap = out_tris.shape[1]
    out_counts[v] = wp.int32(0)

    a = points[v]
    n_center = normals[v]

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

    # --- gather + filter neighbours ---
    # A constructor call -- wp.int32(...) / wp.float32(...) -- declares a mutable Warp dynamic
    # variable; a bare literal is a compile-time constant that freezes the enclosing loop
    # (the fan count stays zero -> no faces). The constructor is what matters, not which spelling
    # of it: the older int()/float() forms are the same builtins under a different name.
    m = wp.int32(0)
    for i in range(k):
        if m >= MAX_NEIGHBOURS:
            # Silent truncation, not a check: this file cannot itself raise
            # if the caller's ``max_neighbours`` violated the module-level invariant. The wrapper's
            # ``max_neighbours > MAX_NEIGHBOURS`` guard is what makes this branch unreachable at
            # the public entry point; a direct launch of this kernel bypasses it.
            break
        nb = neighbor_idx[v, i]
        if nb < 0 or nb == v:
            continue
        if radius > 0.0 and neighbor_dist[v, i] > radius:
            continue
        pn = points[nb]
        if pn[0] == a[0] and pn[1] == a[1] and pn[2] == a[2]:
            continue  # coincident with center
        if wp.dot(n_center, normals[nb]) < NORMAL_FILTER_DOT:
            continue
        nbr[m] = nb
        m += 1

    if m < 2:
        return

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
        return

    # --- greedy fan optimisation (linear-scan replacement of the priority queue) ---
    current = m  # inherits m's dynamic-variable type (m is already mutable)
    for _step in range(m):
        best_w = wp.float32(-FLOAT32_INF_CONSTANT)  # the constructor keeps this mutable
        best_pos = wp.int32(-1)
        for i in range(m):
            if nbr[i] < 0:
                continue
            res = edge_removal_weight(
                v, i, m, border, crit_angle, normalizer_sq, nbr, ang, points, normals
            )
            if res[1] > 0.5:
                continue  # stable, cannot remove
            update_argmax(best_w, best_pos, res[0], i)
        if best_pos < 0:
            break
        old = nbr[best_pos]
        prv = cycle_prev(nbr, m, best_pos)
        nbr[best_pos] = -1
        current -= 1
        if old == border:
            border = nbr[prv]
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
        # orient so the face normal agrees with the (trusted) center normal
        if wp.dot(wp.cross(pb - a, pc - a), n_center) < 0.0:
            bidx, cidx = cidx, bidx
        if slot < cap:
            out_tris[v, slot, 0] = v
            out_tris[v, slot, 1] = bidx
            out_tris[v, slot, 2] = cidx
            slot += 1
    out_counts[v] = slot


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


@wp.kernel
def repeated_triangle_flags(
    sorted_keys: wp.array[wp.uint64], n: wp.int32, out_flags: wp.array[wp.int32]
) -> None:
    # ``1`` at the start of a run of two or three equal keys: a triangle two or three fans agree
    # on. ``grouping.mark_group_starts`` is the same test for one exact run length; this one
    # accepts either length in one pass, because keeping both keeps them merged in key order.
    # Nested rather than one ``and`` chain: each test reads an index only its guard makes valid.
    i = wp.int32(wp.tid())
    key = sorted_keys[i]
    flag = wp.int32(0)
    if sorted_run_start(sorted_keys, i) and i + 1 < n:
        if sorted_keys[i + 1] == key:
            flag = 1
            if i + 3 < n:
                if sorted_keys[i + 3] == key:
                    flag = 0  # a run longer than three
    out_flags[i] = flag


@wp.kernel
def emit_repeated_triangles(
    inclusive_flags: wp.array[wp.int32],
    sorted_slots: wp.array[wp.int32],
    tris: wp.array2d[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # One face per flagged run, in key order, taken from the run's first -- lowest -- fan slot so
    # its winding is that fan's. The flag is the step of the in-place inclusive scan.
    i = wp.int32(wp.tid())
    row, flag = scanned_count(inclusive_flags, i)
    if flag == 0:
        return
    slot = sorted_slots[i]
    for c in range(3):
        out_faces[3 * row + c] = tris[slot, c]


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
    s = wp.int32(wp.tid())
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

    g = (points[s] - cube_lower) * inv_cell
    base, next_corner, fractions = trilinear_cell(g, wp.vec3i(res, res, res))

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


@wp.kernel(enable_backward=False)
def sample_field_trilinear(
    field: wp.array[wp.float32],
    res: wp.int32,
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    points: wp.array[wp.vec3],
    out_values: wp.array[wp.float32],
) -> None:
    s = wp.int32(wp.tid())
    g = (points[s] - cube_lower) * inv_cell
    out_values[s] = poisson_sample_grid(field, res, g[0], g[1], g[2])
