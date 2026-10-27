"""
A serial half-edge mesh builder that adds candidate triangles in a fixed order, keeping the first.

Backs the contested regions of
[`triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud]. The triangulation
assembles its candidate triangles one at a time in a sorted order, through the splices of a
half-edge structure, refusing a triangle that would repeat a directed edge or pinch a vertex and
deferring one whose vertex has no single gap to attach it to. It then deletes the faces that make
a hole pass a vertex twice and adds the remaining triangles again, up to three rounds. Every
decision reads the rings the earlier ones left, so the order is the algorithm: this module runs it
in one thread, splice for splice.

Only the candidates near a conflict reach it. Where every candidate around a vertex forms one fan
with no repeated directed edge the builder adds them all, whatever the order, so
``reconstruction`` keeps those directly and hands this module the faces within a few rings of a
conflict, renumbered locally, in their global order. Its rings are exact where a vertex's whole
star is present (``complete``); the hole walk of the complicating-face test continues past them
through ``hole_next``, the single hole edge leaving each clean vertex.

The records: half-edge ``2 e`` and ``2 e + 1`` are the two sides of edge ``e``,
``next`` / ``prev`` rotate around the origin, ``-1`` is an invalid vertex, face or edge.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})

# ``add_face``'s answer: added, retry after other faces, or never.
ADD_SUCCESS = wp.constant(0)
ADD_TRY_LATER = wp.constant(1)
ADD_FAIL = wp.constant(2)

# A candidate's state: outside the current region, waiting to be added, added, deleted.
FACE_OUTSIDE = wp.constant(0)
FACE_PENDING = wp.constant(1)
FACE_ADDED = wp.constant(2)
FACE_DELETED = wp.constant(3)

# The triangulation runs at most this many delete-and-add rounds.
COMPLICATING_ROUNDS = 3
# ``vacant``'s codes in ``delete_marked_faces``: kept, vacated now, lone before.
EDGE_KEPT = wp.constant(0)
EDGE_VACATED = wp.constant(1)
EDGE_LONE = wp.constant(2)


@wp.struct
class HalfEdges:
    """The half-edge records and the per-vertex and per-face entry edges, plus the record count."""

    next: wp.array[wp.int32]
    prev: wp.array[wp.int32]
    org: wp.array[wp.int32]
    left: wp.array[wp.int32]
    edge_per_vertex: wp.array[wp.int32]
    edge_per_face: wp.array[wp.int32]
    count: wp.array[wp.int32]


@wp.func
def sym(e: wp.int32) -> wp.int32:
    return e + 1 - 2 * (e % 2)


@wp.func
def set_org_ring(t: HalfEdges, a: wp.int32, v: wp.int32) -> None:
    # Every edge of ``a``'s origin ring.
    t.org[a] = v
    i = t.next[a]
    while i != a:
        t.org[i] = v
        i = t.next[i]


@wp.func
def set_left_ring(t: HalfEdges, a: wp.int32, f: wp.int32) -> None:
    # Every edge of ``a``'s left ring.
    t.left[a] = f
    i = t.prev[sym(a)]
    while i != a:
        t.left[i] = f
        i = t.prev[sym(i)]


@wp.func
def from_same_origin_ring(t: HalfEdges, a: wp.int32, b: wp.int32) -> wp.bool:
    ia = a
    ib = b
    while True:
        if ia == ib:
            return True
        ia = t.next[ia]
        if ia == a:
            return False
        if ia == ib:
            return True
        ib = t.prev[ib]
        if ib == b:
            return False
    return False


@wp.func
def from_same_left_ring(t: HalfEdges, a: wp.int32, b: wp.int32) -> wp.bool:
    ia = a
    ib = b
    while True:
        if ia == ib:
            return True
        ia = t.prev[sym(ia)]
        if ia == a:
            return False
        if ia == ib:
            return True
        ib = sym(t.next[ib])
        if ib == b:
            return False
    return False


@wp.func
def splice(t: HalfEdges, a: wp.int32, b: wp.int32) -> None:
    # Merge the origin rings of ``a`` and ``b``, or split them if they
    # are one, carrying the origin and left ids across.
    if a == b:
        return
    an = t.next[a]
    bn = t.next[b]
    ao = t.org[a]
    bo = t.org[b]
    al = t.left[a]
    bl = t.left[b]
    same_org = ao == bo
    same_left = al == bl
    if not same_org:
        if ao >= 0:
            set_org_ring(t, b, ao)
        elif bo >= 0:
            set_org_ring(t, a, bo)
    if not same_left:
        if al >= 0:
            set_left_ring(t, b, al)
        elif bl >= 0:
            set_left_ring(t, a, bl)
    t.next[a] = bn
    t.next[b] = an
    an_prev = t.prev[an]
    t.prev[an] = t.prev[bn]
    t.prev[bn] = an_prev
    if same_org and t.org[b] >= 0:
        set_org_ring(t, b, -1)
        v = t.org[a]
        if not from_same_origin_ring(t, t.edge_per_vertex[v], a):
            t.edge_per_vertex[v] = a
    if same_left and t.left[b] >= 0:
        set_left_ring(t, b, -1)
        f = t.left[a]
        if not from_same_left_ring(t, t.edge_per_face[f], a):
            t.edge_per_face[f] = a


@wp.func
def set_org(t: HalfEdges, a: wp.int32, v: wp.int32) -> None:
    old = t.org[a]
    if v == old:
        return
    set_org_ring(t, a, v)
    if old >= 0:
        t.edge_per_vertex[old] = -1
    if v >= 0:
        t.edge_per_vertex[v] = a


@wp.func
def set_left(t: HalfEdges, a: wp.int32, f: wp.int32) -> None:
    old = t.left[a]
    if f == old:
        return
    set_left_ring(t, a, f)
    if old >= 0:
        t.edge_per_face[old] = -1
    if f >= 0:
        t.edge_per_face[f] = a


@wp.func
def make_edge(t: HalfEdges) -> wp.int32:
    h = t.count[0]
    t.count[0] = h + 2
    for side in range(2):
        t.next[h + side] = h + side
        t.prev[h + side] = h + side
        t.org[h + side] = -1
        t.left[h + side] = -1
    return h


@wp.func
def find_edge(t: HalfEdges, o: wp.int32, d: wp.int32) -> wp.int32:
    e0 = t.edge_per_vertex[o]
    if e0 < 0:
        return -1
    e = e0
    while True:
        if t.org[sym(e)] == d:
            return e
        e = t.next[e]
        if e == e0:
            return -1
    return -1


@wp.func
def only_left_hole(t: HalfEdges, a: wp.int32) -> wp.int32:
    # The edge from ``a`` with no left face when it is the only such edge (``a`` has one gap),
    # else -1.
    e0 = t.edge_per_vertex[a]
    if e0 < 0:
        return -1
    eh = e0
    while t.left[eh] >= 0:
        eh = t.next[eh]
        if eh == e0:
            return -1
    e = t.next[eh]
    while e != e0:
        if t.left[e] < 0:
            return -1
        e = t.next[e]
    return eh


@wp.func
def add_face(t: HalfEdges, f: wp.int32, vs: wp.vec3i) -> wp.int32:
    # Add one triangle, its checks in this order: a repeated
    # vertex or a directed edge that already has a left face fails, two existing edges at a vertex
    # that are not consecutive in its ring fail, a vertex that shares no edge with the face and has
    # not exactly one gap defers.
    if vs[0] == vs[1] or vs[0] == vs[2] or vs[1] == vs[2]:
        return ADD_FAIL
    simple = wp.vec3i(0, 0, 0)
    e = wp.vec3i(-1, -1, -1)
    hole = wp.vec3i(-1, -1, -1)
    for i in range(3):
        i1 = (i + 1) % 3
        if t.edge_per_vertex[vs[i]] < 0:
            simple[i] = 1
        else:
            ei = find_edge(t, vs[i], vs[i1])
            e[i] = ei
            if ei >= 0:
                if t.left[ei] >= 0:
                    return ADD_FAIL
                simple[i] = 1
                simple[i1] = 1
    for i in range(3):
        i2 = (i + 2) % 3
        if e[i] >= 0 and e[i2] >= 0:
            if t.next[e[i]] != sym(e[i2]):
                return ADD_FAIL
        if simple[i] == 0:
            h = only_left_hole(t, vs[i])
            hole[i] = h
            if h < 0:
                return ADD_TRY_LATER
    for i in range(3):
        if e[i] < 0:
            e[i] = make_edge(t)
    for i in range(3):
        i2 = (i + 2) % 3
        ei = e[i]
        if t.org[ei] == vs[i] and t.org[sym(e[i2])] == vs[i]:
            continue
        if hole[i] >= 0 and hole[i] != ei:
            splice(t, hole[i], ei)
        splice(t, ei, t.prev[sym(e[i2])])
        set_org(t, ei, vs[i])
    set_left(t, e[0], f)
    return ADD_SUCCESS


@wp.func
def add_pending_faces(
    t: HalfEdges, tris: wp.array2d[wp.int32], state: wp.array[wp.int32], active: wp.array[wp.int32]
) -> wp.int32:
    # Add the pending faces: ascending, pass after pass until a pass
    # adds nothing. A deferred face stays in the pass list; one that fails leaves it but stays
    # pending, so the next call retries both.
    n_active = wp.int32(0)
    for f in range(tris.shape[0]):
        if state[f] == FACE_PENDING:
            active[n_active] = f
            n_active += 1
    total = wp.int32(0)
    while True:
        added = wp.int32(0)
        kept = wp.int32(0)
        for k in range(n_active):
            face = active[k]
            result = add_face(t, face, wp.vec3i(tris[face, 0], tris[face, 1], tris[face, 2]))
            if result == ADD_TRY_LATER:
                active[kept] = face
                kept += 1
            elif result == ADD_SUCCESS:
                state[face] = FACE_ADDED
                added += 1
        n_active = kept
        if added == 0:
            break
        total += added
    return total


@wp.func
def same_hole_returns(
    t: HalfEdges,
    g: wp.int32,
    complete: wp.array[wp.int32],
    local_to_global: wp.array[wp.int32],
    global_to_local: wp.array[wp.int32],
    hole_next: wp.array[wp.int32],
) -> wp.bool:
    # Whether the hole behind hole edge ``g`` (no left face, from ``v``) leaves ``v`` again by
    # another edge before it closes: ``v`` is then repeated on that hole. The walk is the hole's
    # left ring, ``prev(sym(h))``, read from the records where the vertex's star is complete and
    # from the clean vertex's single hole edge elsewhere, in global vertex ids.
    v = local_to_global[t.org[g]]
    o = v
    d = local_to_global[t.org[sym(g)]]
    d0 = d
    limit = 2 * global_to_local.shape[0] + 8
    for _ in range(limit):
        dl = global_to_local[d]
        nd = wp.int32(-1)
        if dl >= 0:
            if complete[dl] != 0:
                nd = local_to_global[t.org[sym(t.prev[find_edge(t, dl, global_to_local[o])])]]
            else:
                nd = hole_next[d]
        else:
            nd = hole_next[d]
        o = d
        d = nd
        if o == v:
            return d != d0
        if d < 0:
            return False
    return False


@wp.func
def wedge_angle(t: HalfEdges, points: wp.array[wp.vec3], e0: wp.int32) -> wp.float32:
    # The angle the faces after hole edge ``e0`` span around its origin, rotating clockwise to
    # the next gap, each angle as ``atan2(|a x b|, a . b)``.
    total = wp.float32(0.0)
    e = e0
    while t.left[sym(e)] >= 0:
        d1 = points[t.org[sym(e)]] - points[t.org[e]]
        e = t.prev[e]
        d0 = points[t.org[sym(e)]] - points[t.org[e]]
        total += wp.atan2(wp.length(wp.cross(d0, d1)), wp.dot(d0, d1))
    return total


@wp.func
def mark_wedge(t: HalfEdges, e0: wp.int32, marked: wp.array[wp.int32]) -> wp.int32:
    count = wp.int32(0)
    e = e0
    while t.left[sym(e)] >= 0:
        marked[t.left[sym(e)]] = 1
        count += 1
        e = t.prev[e]
    return count


@wp.func
def mark_complicating_faces(
    t: HalfEdges,
    points: wp.array[wp.vec3],
    complete: wp.array[wp.int32],
    local_to_global: wp.array[wp.int32],
    global_to_local: wp.array[wp.int32],
    hole_next: wp.array[wp.int32],
    marked: wp.array[wp.int32],
) -> wp.int32:
    # At every vertex a hole passes twice, keep the widest fan
    # between two gaps (the first in ring order on a tie) and mark the faces of every other one.
    n_marked = wp.int32(0)
    for v in range(t.edge_per_vertex.shape[0]):
        e0 = t.edge_per_vertex[v]
        if e0 < 0 or complete[v] == 0:
            continue
        repeated = wp.bool(False)
        a = e0
        while True:
            if t.left[a] < 0 and not repeated:
                repeated = same_hole_returns(
                    t, a, complete, local_to_global, global_to_local, hole_next
                )
            a = t.next[a]
            if a == e0:
                break
        if not repeated:
            continue
        widest = wp.int32(-1)
        widest_angle = wp.float32(-1.0)
        e = e0
        while True:
            if t.left[e] < 0:
                angle = wedge_angle(t, points, e)
                if angle <= widest_angle:
                    n_marked += mark_wedge(t, e, marked)
                else:
                    if widest >= 0:
                        n_marked += mark_wedge(t, widest, marked)
                    widest = e
                    widest_angle = angle
            e = t.next[e]
            if e == e0:
                break
    return n_marked


@wp.func
def delete_marked_faces(
    t: HalfEdges, state: wp.array[wp.int32], marked: wp.array[wp.int32], vacant: wp.array[wp.int32]
) -> None:
    # Delete the marked faces, in these passes: an edge is vacated when no face
    # beside it survives, a vertex keeps an edge that survives or is dropped, the surviving
    # records skip the vacated edges in their rings, and the vacated edges become lone.
    n_half = t.count[0]
    for ue in range(n_half // 2):
        a = 2 * ue
        b = a + 1
        code = EDGE_VACATED
        if (
            t.left[a] < 0
            and t.org[a] < 0
            and t.next[a] == a
            and t.prev[a] == a
            and t.left[b] < 0
            and t.org[b] < 0
            and t.next[b] == b
            and t.prev[b] == b
        ):
            code = EDGE_LONE
        else:
            if t.left[a] >= 0:
                if marked[t.left[a]] == 0:
                    code = EDGE_KEPT
            if t.left[b] >= 0:
                if marked[t.left[b]] == 0:
                    code = EDGE_KEPT
        vacant[ue] = code
    for v in range(t.edge_per_vertex.shape[0]):
        e0 = t.edge_per_vertex[v]
        if e0 < 0:
            continue
        found = wp.int32(-1)
        e = e0
        while True:
            if vacant[e // 2] == EDGE_KEPT:
                found = e
                break
            e = t.next[e]
            if e == e0:
                break
        if found < 0:
            t.edge_per_vertex[v] = -1
        elif vacant[e0 // 2] != EDGE_KEPT:
            t.edge_per_vertex[v] = found
    for f in range(state.shape[0]):
        if marked[f] != 0:
            t.edge_per_face[f] = -1
            state[f] = FACE_DELETED
    for h in range(n_half):
        if vacant[h // 2] != EDGE_KEPT:
            continue
        n = t.next[h]
        while vacant[n // 2] != EDGE_KEPT:
            n = t.next[n]
        p = t.prev[h]
        while vacant[p // 2] != EDGE_KEPT:
            p = t.prev[p]
        t.next[h] = n
        t.prev[h] = p
        if t.left[h] >= 0:
            if marked[t.left[h]] != 0:
                t.left[h] = -1
    for h in range(n_half):
        if vacant[h // 2] == EDGE_VACATED:
            t.next[h] = h
            t.prev[h] = h
            t.org[h] = -1
            t.left[h] = -1
    for f in range(state.shape[0]):
        marked[f] = 0


@wp.kernel(enable_backward=False)
def build_candidate_mesh(
    points: wp.array[wp.vec3],
    tris: wp.array2d[wp.int32],
    n_rep3: wp.int32,
    complete: wp.array[wp.int32],
    local_to_global: wp.array[wp.int32],
    global_to_local: wp.array[wp.int32],
    hole_next: wp.array[wp.int32],
    t: HalfEdges,
    active: wp.array[wp.int32],
    marked: wp.array[wp.int32],
    vacant: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # dim=1. The assembly: the triangles three fans agree on (the first ``n_rep3``
    # rows), then those left with the ones two fans agree on, then the delete-and-add rounds.
    n_faces = tris.shape[0]
    t.count[0] = 0
    for v in range(t.edge_per_vertex.shape[0]):
        t.edge_per_vertex[v] = -1
    for f in range(n_faces):
        t.edge_per_face[f] = -1
        marked[f] = 0
        out_state[f] = wp.where(f < n_rep3, FACE_PENDING, FACE_OUTSIDE)
    add_pending_faces(t, tris, out_state, active)
    for f in range(n_rep3, n_faces):
        out_state[f] = FACE_PENDING
    add_pending_faces(t, tris, out_state, active)
    for _ in range(COMPLICATING_ROUNDS):
        if (
            mark_complicating_faces(
                t, points, complete, local_to_global, global_to_local, hole_next, marked
            )
            == 0
        ):
            break
        delete_marked_faces(t, out_state, marked, vacant)
        if add_pending_faces(t, tris, out_state, active) == 0:
            break
