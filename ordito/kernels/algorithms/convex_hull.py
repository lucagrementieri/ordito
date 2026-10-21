"""
Kernels for the exact 3D convex hull of a point cloud: ffHull's growth and Flip-Flop stages.

The algorithm is Gao, Cao, Tan and Huang's *Flip-Flop: Convex Hull Construction via Star-Shaped
Polyhedron in 3D* (I3D 2013). Every intermediate surface is a closed triangulated polyhedron that is
*star-shaped* from one kernel point ``s`` -- each face sees ``s`` strictly beneath it, so the
radial projection from ``s`` tiles the sphere with the faces' cones -- and that invariant is what
lets thousands of local edits run at once without a global consistency check:

- **Growth.** Every point is located in the cone of exactly one face. Each round, every face with
  points beyond it splits 1-3 at the farthest of them; a point beneath or on its face lies in the
  tetrahedron ``(s, face)`` and is dropped for good. A split cone is three disjoint child cones, so
  a point relocates with two orientation tests and the surface stays star-shaped. Every round also
  runs one flip round of the V-criterion below (``propose_flips`` with ``labels`` zero), and the
  pass follows a point through the flip too, since a flip's new faces tile its old faces' cones.
  Flipping while growing keeps the surface near the hull of its vertices, so a pass drops the
  points under its dents rather than inserting them as vertices Flip-Flop must later remove:
  Flip-Flop's rounds fell 52 -> 17 on bunny, 99 -> 42 on dragon, 34 -> 8 on a 1 M sphere and
  7 143 -> 517 on a two-ring drum (the paper flips only once growth is done). Growth stops when no
  point is beyond its face.
- **Flip-Flop.** The grown polyhedron contains every point but is not convex: it has *reflex*
  edges and vertices that are not extreme. A reflex edge whose flip keeps the surface star-shaped
  is flipped (the V-criterion, which only grows the volume). A reflex edge that cannot be flipped
  has an endpoint inside the tetrahedron the other three vertices span with ``s``, which is
  therefore not extreme; it is labelled, and edges incident to the smallest-index labelled vertex
  are flipped even when convex, so its degree falls to 3 and a 3-1 flip deletes it (the
  D-criterion). The paper proves the two criteria together always reach the convex hull of points
  in general position. Each round every face proposes one flip, locks the two or three triangles it
  rewrites with ``wp.atomic_min`` of its index, and flips only if it holds every lock, so the
  lowest proposer always proceeds and conflicting flips never meet.

Robustness
----------
Every decision is the exact sign of a ``predicates.orient3d_sign``, which is exact for ``float32``
input plus one kernel point rounded to 28 bits. Ties are resolved by rules that keep the surface
valid rather than by perturbing the input:

- The kernel point is perturbed symbolically to ``s + eps (1, eps, eps^2)`` (``orient_kernel``), so
  every cone test against a non-collinear triple of points is strict. Without it a kernel point
  sharing a plane with many input points (a symmetric cloud's centroid on a mirror plane) splits a
  face at a point exactly on its cone's boundary, and the child face sees ``s`` edge-on.
- A point exactly on its face's plane is dropped: it lies in the closed tetrahedron ``(s, face)``.
- A flat (coplanar) edge is convex: it is never flipped by volume.
- A flip whose new face would see ``s`` edge-on (a collinear corner triple) is not taken.

Two outcomes of exactly degenerate input are left to the wrapper, which rebuilds from a subset of
the surface's vertices without changing the hull:

- **A stall.** Removing a labelled vertex that lies on a hull edge between collinear neighbours can
  need a flip through a collinear triangle, which the tie rules refuse, and the reflex edges around
  it then wait on its label for good. Every label is certified (``certified_inside``: the vertex
  lies in a closed tetrahedron of ``s`` and three vertices, so it is a convex combination of the
  others), and a reflex edge either proposes, labels or waits on a label, so a stalled surface
  always has labelled vertices to drop.
- **Vertices on a facet or an edge.** The flips converge, but points lying exactly on a hull facet
  or edge stay vertices. ``count_strict_edges`` and ``flag_extreme_vertices`` find them (a vertex
  of a convex triangulated surface is an extreme point exactly when at least three strictly convex
  edges meet there), and the rebuild keeps the extreme vertices alone.

Storage
-------
Faces are ``tri`` rows wound outward (``s`` beneath) and ``adj`` rows packing, for local edge
``k = (t[k], t[k + 1])``, the face across it and that face's own local edge as ``3 * face + edge``.
A split keeps the parent's slot for the child on its edge 0 and appends the two others, so a
neighbour reads where its edge went without a search. A 3-1 flip leaves two dead slots
(``tri[f][0] < 0``), compacted only when the faces are written out. Every counter, tie-break and
slot allocation is order-independent (integer scans, ``atomic_max`` / ``atomic_min`` of unique
keys), so the hull is the same face list on every run.
"""

import warp as wp

from ordito.kernels import array as kernel_array
from ordito.kernels import predicates as kernel_predicates

wp.set_module_options({"enable_backward": False})

# Slots of the growth loop's ``counters`` word: points appended by this round's pass, faces split
# this round (the scan's tail, which the split kernel copies here for the host), the surviving
# point count the split kernel moves out of the append slot so the next pass starts from zero, and
# the start kernel's status.
COUNT_APPENDED = wp.int32(0)
COUNT_SPLITS = wp.int32(1)
COUNT_LIVE = wp.int32(2)
COUNT_STATUS = wp.int32(3)
COUNTERS_SIZE = 4

# Start statuses: a usable tetrahedron, a cloud with no four non-coplanar points, and a tetrahedron
# so thin that no 28-bit kernel point lies strictly inside it.
STATUS_OK = wp.int32(0)
STATUS_COPLANAR = wp.int32(1)
STATUS_FLAT = wp.int32(2)

# Slots of the closing check word: strictly reflex edges left after Flip-Flop (zero for a converged
# hull), whether any vertex is not an extreme point, and the live face count.
CHECK_REFLEX = wp.int32(0)
CHECK_NON_EXTREME = wp.int32(1)
CHECK_LIVE_FACES = wp.int32(2)
CHECK_SIZE = 3

LOCK_FREE = wp.int32(2147483647)
DEAD_FACE = wp.vec3i(-1, -1, -1)

# Flip proposals: no flip, a 2-2 flip of the proposed edge, and a 3-1 flip removing its first or
# second endpoint.
FLIP_NONE = wp.int32(0)
FLIP_22 = wp.int32(1)
FLIP_31_FIRST = wp.int32(2)
FLIP_31_SECOND = wp.int32(3)

# Barycentric weights of the kernel point in the start tetrahedron. Deliberately unequal: the
# centroid of a symmetric tetrahedron lies on the cloud's mirror planes, which makes many cone
# tests exact ties and sends them down the exact path. The second set is the fallback for a
# tetrahedron so thin that rounding the first to 28 bits leaves it outside.
KERNEL_WEIGHTS = wp.vec4d(0.2617, 0.2389, 0.2551, 0.2443)
KERNEL_WEIGHTS_FALLBACK = wp.vec4d(0.25, 0.25, 0.25, 0.25)


@wp.func
def point_d(points: wp.array[wp.vec3], index: wp.int32) -> wp.vec3d:
    return kernel_array.to_vec3d(points[index])


@wp.func
def next3(k: wp.int32) -> wp.int32:
    return wp.where(k == 2, 0, k + 1)


@wp.func
def prev3(k: wp.int32) -> wp.int32:
    return wp.where(k == 0, 2, k - 1)


@wp.func
def unpack_edge(ref: wp.int32) -> wp.vec2i:
    # ``(face, local edge)`` of an ``adj`` entry.
    face = ref // 3
    return wp.vec2i(face, ref - 3 * face)


@wp.struct
class KernelPoint:
    """The kernel point as the predicates take it: 28-bit, rounded to ``float32``, the rounding."""

    exact: wp.vec3d
    rounded: wp.vec3
    error: wp.float32


@wp.func
def kernel_point_of(s: wp.vec3d) -> KernelPoint:
    out = KernelPoint()
    out.exact = s
    rounded = wp.vec3(wp.float32(s[0]), wp.float32(s[1]), wp.float32(s[2]))
    out.rounded = rounded
    shift = s - kernel_array.to_vec3d(rounded)
    largest = wp.max(wp.abs(shift[0]), wp.max(wp.abs(shift[1]), wp.abs(shift[2])))
    out.error = wp.float32(largest) * wp.float32(1.0001)
    return out


@wp.func
def orient_kernel(s: KernelPoint, x: wp.vec3, y: wp.vec3, p: wp.vec3) -> wp.int32:
    # ``orient3d_sign(s, x, y, p)`` with the kernel point perturbed to ``s + eps (1, eps, eps^2)``.
    # The determinant is ``-N . (s - x)`` with ``N = (y - x) x (p - x)``, so an exact zero takes
    # the sign of ``-N_x``, then ``-N_y``, then ``-N_z``: each an exact 2D orientation of the raw
    # coordinates, and only a collinear ``(x, y, p)`` still reads zero.
    sign = kernel_predicates.orient3d_sign_from(s.exact, s.rounded, s.error, x, y, p)
    axis = wp.int32(0)
    while sign == 0 and axis < 3:
        # Projections onto (y, z), (z, x), (x, y): ``N_x``, ``N_y``, ``N_z``.
        i = next3(axis)
        j = prev3(axis)
        sign = -kernel_predicates.orient2d_sign(
            wp.float64(x[i]),
            wp.float64(x[j]),
            wp.float64(y[i]),
            wp.float64(y[j]),
            wp.float64(p[i]),
            wp.float64(p[j]),
        )
        axis += 1
    return sign


@wp.func
def in_cone(s: KernelPoint, a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.bool:
    # ``p`` in the closed cone from ``s`` over the outward face ``(a, b, c)``.
    corners = wp.matrix_from_rows(a, b, c)
    inside = wp.bool(True)
    k = wp.int32(0)
    while inside and k < 3:
        inside = orient_kernel(s, corners[k], corners[next3(k)], p) >= 0
        k += 1
    return inside


@wp.func
def beyond_face(
    points: wp.array[wp.vec3], face: wp.vec3i, p: wp.vec3
) -> tuple[wp.bool, wp.float32]:
    # Whether ``p`` is strictly beyond the face's plane, and the filtered determinant as its
    # distance proxy (the face's area times the distance, so it ranks points of one face).
    a = points[face[0]]
    b = points[face[1]]
    c = points[face[2]]
    filtered = kernel_predicates.orient3d_filtered_f32(a, b, c, p)
    sign = wp.int32(filtered[1])
    if sign == 0:
        sign = kernel_predicates.orient3d_sign(
            kernel_array.to_vec3d(a),
            kernel_array.to_vec3d(b),
            kernel_array.to_vec3d(c),
            kernel_array.to_vec3d(p),
        )
    return sign > 0, filtered[0]


@wp.func
def farthest_key(distance: wp.float32, index: wp.int32) -> wp.int64:
    # A point's bid for its face: largest distance proxy, lowest index on a tie. A point decided
    # beyond only by a later stage can carry a non-positive proxy; it bids as zero, still above
    # the ``-1`` an unclaimed face holds.
    return kernel_array.pack_farthest_key(wp.max(distance, wp.float32(0.0)), index)


@wp.func
def commit_beyond(
    i: wp.int32,
    face: wp.int32,
    proxy: wp.float32,
    counters: wp.array[wp.int32],
    best: wp.array[wp.int64],
    flags: wp.array[wp.int32],
    out_live: wp.array[wp.vec2i],
):
    # Keep a point that is beyond its face: append it to the next live list, bid for the face and
    # flag the face for this round's split. The list's order is arrival order and decides nothing:
    # the bids are an ``atomic_max`` of unique keys.
    out_live[kernel_array.append_slot(counters, COUNT_APPENDED)] = wp.vec2i(i, face)
    # A bid below the face's current best changes nothing, so it skips the atomic: the first pass
    # has four faces for the whole cloud, and every bid serialized on them. A stale read only costs
    # an atomic that loses.
    key = farthest_key(proxy, i)
    if key > best[face]:
        wp.atomic_max(best, face, key)
    flags[face] = 1


# --- Start: the tetrahedron, the kernel point and the first location -------------------------


@wp.kernel
def pick_tetrahedron_corner(
    points: wp.array[wp.vec3], stage: wp.int32, stride: wp.int32, keys: wp.array[wp.int64]
) -> None:
    # One of four dependent arg-max sweeps, one thread per strided slice: the point farthest from
    # point 0, then the one farthest from that, then the one farthest from their line, then the one
    # farthest from the plane of the three. ``keys[stage]`` collects the slice maxima.
    slice_index = wp.int32(wp.tid())
    n = points.shape[0]
    anchor = points[0]
    b = anchor
    c = anchor
    if stage > 0:
        anchor = points[kernel_array.unpack_ranked_index(keys[0])]
    if stage > 1:
        b = points[kernel_array.unpack_ranked_index(keys[1])]
    if stage > 2:
        c = points[kernel_array.unpack_ranked_index(keys[2])]
    best = wp.int64(-1)
    i = slice_index
    while i < n:
        p = points[i]
        score = wp.float32(0.0)
        if stage < 2:
            score = wp.length_sq(p - anchor)
        elif stage == 2:
            score = wp.length_sq(wp.cross(p - anchor, b - anchor))
        else:
            score = wp.abs(kernel_predicates.orient3d_filtered_f32(anchor, b, c, p)[0])
        # Only a good spread is wanted here, not an exact maximum; a non-finite score (an overflow
        # on huge coordinates) ranks as zero.
        score = wp.where(wp.isfinite(score), score, wp.float32(0.0))
        best = wp.max(best, kernel_array.pack_farthest_key(score, i))
        i += stride
    wp.atomic_max(keys, stage, best)


@wp.func
def kernel_point_inside(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, weights: wp.vec4d
) -> tuple[wp.bool, wp.vec3d]:
    # The 28-bit kernel point at ``weights`` and whether it lies strictly inside the tetrahedron
    # whose outward faces ``start_tetrahedron`` writes.
    s = kernel_predicates.round_to_28_bits(
        weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d
    )
    inside = (
        kernel_predicates.orient3d_sign(a, b, c, s) < 0
        and kernel_predicates.orient3d_sign(a, d, b, s) < 0
        and kernel_predicates.orient3d_sign(b, d, c, s) < 0
        and kernel_predicates.orient3d_sign(c, d, a, s) < 0
    )
    return inside, s


@wp.kernel
def start_tetrahedron(
    points: wp.array[wp.vec3],
    keys: wp.array[wp.int64],
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    best: wp.array[wp.int64],
    counters: wp.array[wp.int32],
    out_corners: wp.array[wp.int32],
    out_kernel_point: wp.array[wp.vec3d],
) -> None:
    # dim=1. Wind the four picked corners outward, write the tetrahedron's faces and adjacency,
    # and place the kernel point; or record why the cloud has no tetrahedron.
    ia = kernel_array.unpack_ranked_index(keys[0])
    ib = kernel_array.unpack_ranked_index(keys[1])
    ic = kernel_array.unpack_ranked_index(keys[2])
    i_d = kernel_array.unpack_ranked_index(keys[3])
    sign = kernel_predicates.orient3d_sign(
        point_d(points, ia), point_d(points, ib), point_d(points, ic), point_d(points, i_d)
    )
    if sign == 0:
        counters[COUNT_STATUS] = STATUS_COPLANAR
        return
    if sign > 0:
        # Swap ``b`` and ``c`` so ``d`` lies beneath ``(a, b, c)``.
        swap = ib
        ib = ic
        ic = swap
    a = point_d(points, ia)
    b = point_d(points, ib)
    c = point_d(points, ic)
    d = point_d(points, i_d)
    inside, s = kernel_point_inside(a, b, c, d, KERNEL_WEIGHTS)
    if not inside:
        inside, s = kernel_point_inside(a, b, c, d, KERNEL_WEIGHTS_FALLBACK)
    if not inside:
        counters[COUNT_STATUS] = STATUS_FLAT
        return
    # Faces (a, b, c), (a, d, b), (b, d, c), (c, d, a): each leaves the fourth corner beneath.
    tri[0] = wp.vec3i(ia, ib, ic)
    tri[1] = wp.vec3i(ia, i_d, ib)
    tri[2] = wp.vec3i(ib, i_d, ic)
    tri[3] = wp.vec3i(ic, i_d, ia)
    adj[0] = wp.vec3i(5, 8, 11)
    adj[1] = wp.vec3i(10, 6, 0)
    adj[2] = wp.vec3i(4, 9, 1)
    adj[3] = wp.vec3i(7, 3, 2)
    for f in range(4):
        best[f] = wp.int64(-1)
    out_corners[0] = ia
    out_corners[1] = ib
    out_corners[2] = ic
    out_corners[3] = i_d
    out_kernel_point[0] = s


@wp.kernel
def seed_points(
    points: wp.array[wp.vec3],
    corners: wp.array[wp.int32],
    kernel_point: wp.array[wp.vec3d],
    tri: wp.array[wp.vec3i],
    counters: wp.array[wp.int32],
    best: wp.array[wp.int64],
    flags: wp.array[wp.int32],
    out_live: wp.array[wp.vec2i],
) -> None:
    # Locate every point but the corners in the cone of one tetrahedron face and keep those beyond
    # it: the first round's pass.
    i = wp.int32(wp.tid())
    if counters[COUNT_STATUS] != STATUS_OK:
        return
    if i == corners[0] or i == corners[1] or i == corners[2] or i == corners[3]:
        return
    s = kernel_point_of(kernel_point[0])
    p = points[i]
    # A dynamic loop keeps one inlined ``in_cone`` (see ``propose_flips``).
    face = wp.int32(0)
    while face < 3 and not in_cone(
        s, points[tri[face][0]], points[tri[face][1]], points[tri[face][2]], p
    ):
        face += 1
    beyond, proxy = beyond_face(points, tri[face], p)
    if beyond:
        commit_beyond(i, face, proxy, counters, best, flags, out_live)


# --- Growth rounds ---------------------------------------------------------------------------


@wp.func
def split_neighbor_ref(
    ref: wp.int32, flags: wp.array[wp.int32], offsets: wp.array[wp.int32], n_faces: wp.int32
) -> wp.int32:
    # Where the far side of an edge is after this round's splits: a split neighbour hands each of
    # its edges to the child it made for it, as that child's edge 0.
    far = unpack_edge(ref)
    out = ref
    if flags[far[0]] != 0:
        slot = far[0]
        if far[1] > 0:
            slot = n_faces + 2 * (offsets[far[0]] - 1) + far[1] - 1
        out = 3 * slot
    return out


@wp.kernel
def split_faces(
    n_faces: wp.int32,
    flags: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    best: wp.array[wp.int64],
    child: wp.array[wp.int32],
    counters: wp.array[wp.int32],
    dirty: wp.array[wp.int32],
    out_next_flags: wp.array[wp.int32],
    out_split_corners: wp.array[wp.vec4i],
) -> None:
    # Split every flagged face 1-3 at its winning point ``v``: ``(a, b, v)`` keeps the slot,
    # ``(b, c, v)`` and ``(c, a, v)`` take the two slots its inclusive scan entry reserves past
    # ``n_faces``. ``child`` tells the next pass where a face's points went (``-1``: nowhere), and
    # the next round's flags are cleared here because this round's are still being read.
    f = wp.int32(wp.tid())
    if f == 0:
        counters[COUNT_SPLITS] = offsets[n_faces - 1]
        counters[COUNT_LIVE] = counters[COUNT_APPENDED]
        counters[COUNT_APPENDED] = 0
    out_next_flags[f] = 0
    if flags[f] == 0:
        child[f] = -1
        return
    v = kernel_array.unpack_ranked_index(best[f])
    base = n_faces + 2 * (offsets[f] - 1)
    t = tri[f]
    old = adj[f]
    edge0 = split_neighbor_ref(old[0], flags, offsets, n_faces)
    edge1 = split_neighbor_ref(old[1], flags, offsets, n_faces)
    edge2 = split_neighbor_ref(old[2], flags, offsets, n_faces)
    tri[f] = wp.vec3i(t[0], t[1], v)
    tri[base] = wp.vec3i(t[1], t[2], v)
    tri[base + 1] = wp.vec3i(t[2], t[0], v)
    adj[f] = wp.vec3i(edge0, 3 * base + 2, 3 * (base + 1) + 1)
    adj[base] = wp.vec3i(edge1, 3 * (base + 1) + 2, 3 * f + 1)
    adj[base + 1] = wp.vec3i(edge2, 3 * f + 2, 3 * base + 1)
    child[f] = base
    # The split's corners, for the next pass: the flip round between them may rewrite these slots.
    out_split_corners[f] = wp.vec4i(t[0], t[1], t[2], v)
    dirty[f] = 1
    dirty[base] = 1
    dirty[base + 1] = 1
    best[f] = wp.int64(-1)
    best[base] = wp.int64(-1)
    best[base + 1] = wp.int64(-1)
    out_next_flags[base] = 0
    out_next_flags[base + 1] = 0
    # An unsplit neighbour learns its new neighbour from this thread, the only writer of that edge.
    children = wp.vec3i(f, base, base + 1)
    for k in range(3):
        far = unpack_edge(old[k])
        dirty[far[0]] = 1
        if flags[far[0]] == 0:
            adj[far[0]][far[1]] = 3 * children[k]


@wp.kernel
def assign_points(
    points: wp.array[wp.vec3],
    kernel_point: wp.array[wp.vec3d],
    tri: wp.array[wp.vec3i],
    child: wp.array[wp.int32],
    split_corners: wp.array[wp.vec4i],
    changed: wp.array[wp.int32],
    group: wp.array[wp.vec3i],
    live: wp.array[wp.vec2i],
    counters: wp.array[wp.int32],
    best: wp.array[wp.int64],
    flags: wp.array[wp.int32],
    out_live: wp.array[wp.vec2i],
) -> None:
    # One growth pass: follow a split face's point into the child cone holding it, then, if the
    # round's flips rewrote that child, into the rewritten face holding it; drop the point if it
    # became the split vertex or lies on or beneath its face, and otherwise bid for the face.
    # The three children of ``(a, b, c)`` split at ``v`` are cut by the planes through ``s``, ``v``
    # and each corner, so two or three signs place the point. A flip rewrites two or three faces
    # into ones whose cones tile the same region, so one of the flip's live faces holds it.
    entry = live[wp.int32(wp.tid())]
    i = entry[0]
    f = entry[1]
    base = child[f]
    p = points[i]
    s = kernel_point_of(kernel_point[0])
    if base >= 0:
        split = split_corners[f]
        v = split[3]
        if i == v:
            return
        pv = points[v]
        # ``(a, b, v)`` holds the point when ``side_a >= 0 >= side_b``; otherwise ``(b, c, v)`` when
        # ``side_b > 0 >= side_c`` and ``(c, a, v)`` else. One call site, in a dynamic loop.
        sides = wp.vec3i(0, 0, 0)
        needed = wp.int32(2)
        k = wp.int32(0)
        while k < needed:
            sides[k] = orient_kernel(s, pv, points[split[k]], p)
            if k == 1 and sides[1] > 0:
                needed = 3
            k += 1
        if sides[0] < 0 or sides[1] > 0:
            f = wp.where(sides[1] > 0 and sides[2] <= 0, base, base + 1)
    if changed[f] != 0:
        members = group[f]
        chosen = wp.int32(-1)
        fallback = wp.int32(-1)
        m = wp.int32(0)
        while chosen < 0 and m < 3:
            u = members[m]
            if u >= 0 and tri[u][0] >= 0:
                fallback = u
                t = tri[u]
                if in_cone(s, points[t[0]], points[t[1]], points[t[2]], p):
                    chosen = u
            m += 1
        f = wp.where(chosen >= 0, chosen, fallback)
    beyond, proxy = beyond_face(points, tri[f], p)
    if beyond:
        commit_beyond(i, f, proxy, counters, best, flags, out_live)


# --- Flip-Flop -------------------------------------------------------------------------------
#
# A face re-decides its edges only while it is *dirty*: it was rewritten, a neighbour was, one of
# its corners was labelled, or its last proposal lost a lock. An edge's decision reads nothing
# beyond the two faces on it, the faces across the far edges of the first (the degree-3 tests) and
# the four corners' labels, and each of those changes marks the deciding face dirty, so skipping a
# clean face never skips a flip. Without it every round re-tested every edge, and on a cloud with
# exactly coplanar facets each flat edge re-ran the exact path every round (the prism of cocircular
# rings, 6.4 k faces: 0.7 ms a round).
#
# Labels are round stamps: ``label[v]`` is the round that labelled ``v`` (``0``: none). A round
# reads a label as set only when an *earlier* round wrote it, so every proposal of one round sees
# the same labels whatever the order its threads run in, and the closing kernel recognizes this
# round's labels by their stamp.


@wp.func
def is_labelled(label: wp.array[wp.int32], v: wp.int32, stamp: wp.int32) -> wp.bool:
    return label[v] != 0 and label[v] < stamp


@wp.func
def certified_inside(s: KernelPoint, x: wp.vec3, q1: wp.vec3, q2: wp.vec3, q3: wp.vec3) -> wp.bool:
    # Whether ``x`` lies in the closed tetrahedron ``(s, q1, q2, q3)``: then it is a convex
    # combination of three vertices and of ``s``, which lies strictly inside the hull, so it is not
    # an extreme point and dropping it leaves the hull unchanged. Every label carries this
    # certificate; that is what lets the wrapper drop the labelled vertices when Flip-Flop stalls.
    # A degenerate triangle (collinear corners, or ``s`` in its plane) certifies nothing.
    turn = orient_kernel(s, q1, q2, q3)
    corners = wp.matrix_from_rows(q1, q2, q3)
    if turn < 0:
        corners = wp.matrix_from_rows(q1, q3, q2)
    inside = turn != 0 and in_cone(s, corners[0], corners[1], corners[2], x)
    # ``corners`` sees ``s`` beneath it, so ``x`` must be beneath or on its plane.
    return (
        inside and kernel_predicates.orient3d_sign_f32(corners[0], corners[1], corners[2], x) <= 0
    )


@wp.func
def label_vertex(
    label: wp.array[wp.int32], state: wp.array[wp.int32], v: wp.int32, stamp: wp.int32
):
    if label[v] == 0:
        label[v] = stamp
        state[kernel_array.LOOP_PROGRESS] = 1


@wp.func
def flip_proposal(
    points: wp.array[wp.vec3],
    s: KernelPoint,
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    label: wp.array[wp.int32],
    state: wp.array[wp.int32],
    stamp: wp.int32,
    labels_on: wp.bool,
    f: wp.int32,
    k: wp.int32,
) -> wp.vec2i:
    # Decide the edge ``k = (a, b)`` of ``f = (a, b, c)``, whose far face is ``(b, a, d)``: the flip
    # to propose and, for a 3-1 flip, the third face it rewrites. A reflex edge proposes a flip,
    # labels a certified non-extreme endpoint, or waits on a label already set -- so a round that
    # changes nothing while a reflex edge is left always has a label for the wrapper to drop.
    t1 = tri[f]
    a = t1[k]
    b = t1[next3(k)]
    c = t1[prev3(k)]
    far = unpack_edge(adj[f][k])
    d = tri[far[0]][prev3(far[1])]
    pa = points[a]
    pb = points[b]
    pc = points[c]
    pd = points[d]
    reflex = kernel_predicates.orient3d_sign_f32(pa, pb, pc, pd) > 0
    # ``a`` has degree 3 when the face across ``(c, a)`` is ``(a, c, d)``; ``b`` when the face
    # across ``(b, c)`` is ``(b, d, c)``. Both at once is the tetrahedron, which is convex.
    across_ca = unpack_edge(adj[f][prev3(k)])
    across_bc = unpack_edge(adj[f][next3(k)])
    a_degree3 = tri[across_ca[0]][prev3(across_ca[1])] == d
    b_degree3 = tri[across_bc[0]][prev3(across_bc[1])] == d
    out = wp.vec2i(FLIP_NONE, 0)
    if a_degree3 and b_degree3:
        return out
    if a_degree3 or b_degree3:
        # Removing the degree-3 endpoint leaves its link as one face, ``(b, c, d)`` for ``a`` and
        # ``(a, d, c)`` for ``b``, which must see ``s`` beneath it; where it cannot (a collinear
        # link), a reflex edge labels the endpoint if it can certify it.
        x31 = wp.where(a_degree3, a, b)
        px = wp.where(a_degree3, pa, pb)
        link0 = wp.where(a_degree3, pb, pa)
        link1 = wp.where(a_degree3, pc, pd)
        link2 = wp.where(a_degree3, pd, pc)
        if is_labelled(label, x31, stamp) or reflex:
            if orient_kernel(s, link0, link1, link2) > 0:
                out = wp.vec2i(wp.where(a_degree3, FLIP_31_FIRST, FLIP_31_SECOND), 0)
                out[1] = wp.where(a_degree3, across_ca[0], across_bc[0])
            elif reflex and labels_on and certified_inside(s, px, link0, link1, link2):
                label_vertex(label, state, x31, stamp)
        return out
    # The labelled vertex of smallest index among the four, if any.
    x = wp.int32(-1)
    if is_labelled(label, a, stamp):
        x = a
    if is_labelled(label, b, stamp) and (x < 0 or b < x):
        x = b
    if is_labelled(label, c, stamp) and (x < 0 or c < x):
        x = c
    if is_labelled(label, d, stamp) and (x < 0 or d < x):
        x = d
    # A convex 2-2 edge flips only for the D-criterion; with ``x`` absent or off the edge nothing
    # can happen, and the two cone tests -- the exact path on every flat edge of a coplanar
    # facet -- are skipped.
    if not reflex and x != a and x != b:
        return out
    # A 2-2 edge flips to ``(a, d, c)`` and ``(d, b, c)``, which tile the old faces' cones exactly
    # when ``a`` and ``b`` lie strictly on the two sides of the plane ``(s, c, d)`` that make both
    # new faces see ``s`` beneath them: ``a`` on the negative side, ``b`` on the positive.
    flippable = wp.bool(True)
    side = wp.int32(0)
    while flippable and side < 2:
        q = wp.where(side == 0, pa, pb)
        flippable = orient_kernel(s, pc, pd, q) * wp.where(side == 0, -1, 1) > 0
        side += 1
    if flippable:
        # V-criterion: a reflex edge with no labelled vertex. D-criterion: any flippable edge
        # whose flip lowers the degree of the smallest labelled vertex.
        if (x < 0 and reflex) or x == a or x == b:
            out = wp.vec2i(FLIP_22, 0)
    elif reflex and labels_on:
        # Unflippable and reflex: one endpoint lies inside the tetrahedron the other three span
        # with ``s``, so it is not extreme.
        candidate = wp.int32(0)
        while candidate < 2:
            px = wp.where(candidate == 0, pa, pb)
            other = wp.where(candidate == 0, pb, pa)
            if certified_inside(s, px, other, pc, pd):
                label_vertex(label, state, wp.where(candidate == 0, a, b), stamp)
                candidate = 2
            candidate += 1
    return out


@wp.kernel
def propose_flips(
    points: wp.array[wp.vec3],
    kernel_point: wp.array[wp.vec3d],
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    label: wp.array[wp.int32],
    state: wp.array[wp.int32],
    dirty: wp.array[wp.int32],
    changed: wp.array[wp.int32],
    lock: wp.array[wp.int32],
    labels: wp.int32,
    out_proposal: wp.array[wp.vec3i],
) -> None:
    # Each dirty live face proposes at most one flip, over the edges it owns (the lower face index
    # of the two), and bids for every triangle that flip rewrites. A face that proposes nothing is
    # clean until something it reads changes. With ``labels`` zero (the flip round interleaved with
    # growth) nothing is labelled, so only the V-criterion flips: volume only grows, and the points
    # it encloses can be dropped as soon as the next pass sees them.
    f = wp.int32(wp.tid())
    changed[f] = 0
    proposal = wp.vec3i(FLIP_NONE, 0, 0)
    if dirty[f] != 0 and tri[f][0] >= 0:
        s = kernel_point_of(kernel_point[0])
        stamp = state[kernel_array.LOOP_ROUND] + 1
        # A dynamic bound, because Warp unrolls a constant ``range(3)`` into three inlined copies of
        # ``flip_proposal``. Not a speed fix: the round time did not move (the float32 filter was
        # the lever, see ``predicates.orient3d_filtered_f32``), but the kernel stays one copy.
        edges = wp.int32(3)
        for k in range(edges):
            if proposal[0] == FLIP_NONE:
                g = adj[f][k] // 3
                if f < g:
                    decided = flip_proposal(
                        points, s, tri, adj, label, state, stamp, labels != 0, f, k
                    )
                    if decided[0] != FLIP_NONE:
                        proposal = wp.vec3i(decided[0], decided[1], k)
                        wp.atomic_min(lock, f, f)
                        wp.atomic_min(lock, g, f)
                        if decided[0] != FLIP_22:
                            wp.atomic_min(lock, decided[1], f)
        if proposal[0] == FLIP_NONE:
            dirty[f] = 0
    out_proposal[f] = proposal


@wp.func
def find_edge(t: wp.vec3i, x: wp.int32, y: wp.int32) -> wp.int32:
    # Local index of the directed edge ``(x, y)`` in ``t``, or ``-1``.
    out = wp.int32(-1)
    for k in range(3):
        if t[k] == x and t[next3(k)] == y:
            out = k
    return out


@wp.kernel
def apply_flips(
    proposal: wp.array[wp.vec3i],
    lock: wp.array[wp.int32],
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    state: wp.array[wp.int32],
    changed: wp.array[wp.int32],
    group: wp.array[wp.vec3i],
) -> None:
    # Flip where the proposer holds every lock. The new triangles keep their *old* outer
    # neighbours in ``adj``; ``repair_flip_adjacency`` resolves those that flipped too, through
    # ``group``, the slots each flip rewrote.
    f = wp.int32(wp.tid())
    p = proposal[f]
    code = p[0]
    if code == FLIP_NONE:
        return
    k = p[2]
    far = unpack_edge(adj[f][k])
    g = far[0]
    h = p[1]
    if lock[f] != f or lock[g] != f:
        return
    if code != FLIP_22 and lock[h] != f:
        return
    t1 = tri[f]
    a = t1[k]
    b = t1[next3(k)]
    c = t1[prev3(k)]
    d = tri[g][prev3(far[1])]
    across_bc = adj[f][next3(k)]
    across_ca = adj[f][prev3(k)]
    across_ad = adj[g][next3(far[1])]
    across_db = adj[g][prev3(far[1])]
    state[kernel_array.LOOP_PROGRESS] = 1
    changed[f] = 1
    changed[g] = 1
    if code == FLIP_22:
        tri[f] = wp.vec3i(a, d, c)
        tri[g] = wp.vec3i(d, b, c)
        adj[f] = wp.vec3i(across_ad, 3 * g + 2, across_ca)
        adj[g] = wp.vec3i(across_db, across_bc, 3 * f + 1)
        group[f] = wp.vec3i(f, g, -1)
        group[g] = wp.vec3i(f, g, -1)
        return
    th = tri[h]
    changed[h] = 1
    group[f] = wp.vec3i(f, g, h)
    group[g] = wp.vec3i(f, g, h)
    group[h] = wp.vec3i(f, g, h)
    tri[g] = DEAD_FACE
    tri[h] = DEAD_FACE
    if code == FLIP_31_FIRST:
        # ``h = (a, c, d)``: its ``(c, d)`` edge survives into ``(b, c, d)``.
        tri[f] = wp.vec3i(b, c, d)
        adj[f] = wp.vec3i(across_bc, adj[h][find_edge(th, c, d)], across_db)
    else:
        # ``h = (b, d, c)``: its ``(d, c)`` edge survives into ``(a, d, c)``.
        tri[f] = wp.vec3i(a, d, c)
        adj[f] = wp.vec3i(across_ad, adj[h][find_edge(th, d, c)], across_ca)


@wp.kernel
def repair_flip_adjacency(
    changed: wp.array[wp.int32],
    group: wp.array[wp.vec3i],
    label: wp.array[wp.int32],
    state: wp.array[wp.int32],
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    lock: wp.array[wp.int32],
    dirty: wp.array[wp.int32],
) -> None:
    # Close a flip round. A rewritten triangle's outer neighbour either flipped too -- then the
    # edge now lives in one of that flip's slots, found by searching its group -- or did not, and
    # then this thread, the edge's only writer, points it back. A rewritten face, or one with a
    # corner labelled this round, marks itself and its neighbours dirty. Locks are freed.
    t = wp.int32(wp.tid())
    lock[t] = LOCK_FREE
    corners = tri[t]
    if corners[0] < 0:
        return
    stamp = state[kernel_array.LOOP_ROUND] + 1
    touched = (
        changed[t] != 0
        or label[corners[0]] == stamp
        or label[corners[1]] == stamp
        or label[corners[2]] == stamp
    )
    if changed[t] != 0:
        for k in range(3):
            x = corners[k]
            y = corners[next3(k)]
            far = unpack_edge(adj[t][k])
            if changed[far[0]] != 0:
                members = group[far[0]]
                for m in range(3):
                    u = members[m]
                    if u >= 0:
                        ku = find_edge(tri[u], y, x)
                        if ku >= 0:
                            adj[t][k] = 3 * u + ku
            else:
                adj[far[0]][far[1]] = 3 * t + k
    if touched:
        dirty[t] = 1
        for k in range(3):
            dirty[adj[t][k] // 3] = 1


# --- Extreme vertices and output -------------------------------------------------------------


@wp.kernel
def count_strict_edges(
    points: wp.array[wp.vec3],
    tri: wp.array[wp.vec3i],
    adj: wp.array[wp.vec3i],
    out_convex: wp.array[wp.int32],
    out_check: wp.array[wp.int32],
) -> None:
    # Per vertex, the strictly convex edges meeting it; and the strictly reflex edges left, which a
    # converged hull has none of. Each edge is counted by its lower face.
    f = wp.int32(wp.tid())
    t = tri[f]
    if t[0] < 0:
        return
    edges = wp.int32(3)
    for k in range(edges):
        far = unpack_edge(adj[f][k])
        if f < far[0]:
            a = t[k]
            b = t[next3(k)]
            sign = kernel_predicates.orient3d_sign_f32(
                points[a], points[b], points[t[prev3(k)]], points[tri[far[0]][prev3(far[1])]]
            )
            if sign < 0:
                wp.atomic_add(out_convex, a, 1)
                wp.atomic_add(out_convex, b, 1)
            elif sign > 0:
                wp.atomic_add(out_check, CHECK_REFLEX, 1)


@wp.kernel
def flag_extreme_vertices(
    tri: wp.array[wp.vec3i],
    convex: wp.array[wp.int32],
    out_check: wp.array[wp.int32],
    out_vertex_flags: wp.array[wp.int32],
    out_face_flags: wp.array[wp.int32],
) -> None:
    # A vertex is an extreme point when at least three strictly convex edges meet it; with two it
    # lies on a hull edge, with none inside a facet. Every face writes its corners' flags (the
    # same value from each), its own live flag for the output scan, and counts itself live.
    f = wp.int32(wp.tid())
    t = tri[f]
    live = t[0] >= 0
    out_face_flags[f] = wp.where(live, 1, 0)
    if not live:
        return
    wp.atomic_add(out_check, CHECK_LIVE_FACES, 1)
    for k in range(3):
        v = t[k]
        extreme = convex[v] >= 3
        out_vertex_flags[v] = wp.where(extreme, 1, 0)
        if not extreme:
            out_check[CHECK_NON_EXTREME] = 1


@wp.kernel
def flag_unlabelled_vertices(
    tri: wp.array[wp.vec3i], label: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # The vertices a stalled surface is rebuilt from: every live vertex but the labelled, which are
    # certified non-extreme.
    f = wp.int32(wp.tid())
    t = tri[f]
    if t[0] < 0:
        return
    for k in range(3):
        if label[t[k]] == 0:
            out_flags[t[k]] = 1


@wp.kernel
def write_live_faces(
    tri: wp.array[wp.vec3i],
    offsets: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The live faces in slot order, as a flat buffer; ``remap`` (when non-empty) maps the indices of
    # a rebuilt subset back to the caller's points.
    f = wp.int32(wp.tid())
    t = tri[f]
    if t[0] < 0:
        return
    row = 3 * (offsets[f] - 1)
    for k in range(3):
        v = t[k]
        if remap.shape[0] > 0:
            v = remap[v]
        out_faces[row + k] = v
