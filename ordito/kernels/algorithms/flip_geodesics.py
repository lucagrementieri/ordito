"""
geometry-central's edge-flip geodesics, as potpourri3d's ``EdgeFlipGeodesicSolver`` runs them.

``FlipEdgeNetwork`` behind ``find_geodesic_path``; backs
``geodesic_walk.geodesic_path(method="potpourri3d")``. A path starts as the shortest edge path
from a source and is shortened by flipping the edges of a *signpost intrinsic triangulation* inside
its sharpest wedge, one wedge at a time from a priority queue keyed on the wedge angle, until every
wedge is straight to within ``EPS_ANGLE``; the result is a locally shortest path whose intrinsic
edges are then traced across the input mesh to give the polyline.

Everything that does not depend on the flip order runs one thread an element, on the mesh's
device:

- The halfedge numbering of ``ManifoldSurfaceMesh`` (edge ``e`` is halfedges ``2 e`` and
  ``2 e + 1``, numbered in the order a scan of the faces' corners meets them, so a scan of
  first-encounter flags; the exterior halfedges of the boundary are the odd ones of their edge),
  the embedded and intrinsic geometry and the signposts, one thread a corner, face or vertex.
- The shortest edge paths. geometry-central's lazy Dijkstra settles each vertex at the least
  left-to-right floating sum over the paths reaching it (addition is monotone), which is the
  fixpoint of a ``float64`` Bellman-Ford relaxation; the halfedge it settles a vertex by is the
  least ``(distance, halfedge)`` among the in-halfedges that reach it at that sum. The CPU device,
  which runs a launch's threads one after another, takes the Dijkstra itself.
- The trace of the shortened paths' intrinsic edges, one thread an edge.

The shortening is not: within a path every decision reads the triangulation the earlier flips
left, so the order is the algorithm, and one thread on the CPU device runs it flip for flip, path
after path: the wedge queue keyed on ``(angle, type, segment id)`` with its stale-entry tests and
its purge every 1 000 shortenings, and the flip test of ``flipEdgeIfPossible``. Each path starts
from the input triangulation, as potpourri3d's ``rewind`` restores it; here the halfedges and
edges a path's flips touched are restored from the input tables. One GPU thread a path, each with
its own copy-on-write table over the input, reproduced the answer but ran 3-14x slower end to end
(``sphere_med``, 1 to 4 096 paths): a flip is a chain of dependent loads, which a GPU thread waits
on 30-60x longer than a CPU core, and a batch waits on its slowest path.

Paths are open and simple to start with (a Dijkstra path), so the closed-loop cases of
``FlipEdgeNetwork`` (a single-edge loop, a length-2 loop) never arise and are not ported.
"""

import warp as wp

from ordito.kernels.array import LOOP_PROGRESS, copy_staged, first_staged_chunk, stage_point

wp.set_module_options({"enable_backward": False})

PI_D = wp.float64(3.141592653589793)
TWO_PI_D = wp.float64(6.283185307179586)
# geometry-central's ``FlipEdgeNetwork::EPS_ANGLE``: a wedge within this of pi is straight.
EPS_ANGLE = wp.float64(1e-5)
# ``IntrinsicTriangulation::triangleTestEPS``: the flip test's relative signed-area margin.
TRIANGLE_TEST_EPS = wp.float64(1e-6)
# ``trace_geodesic.cpp``'s two projection tolerances.
TRACE_EPS_TIGHT = wp.float64(1e-12)
TRACE_EPS_LOOSE = wp.float64(1e-9)
# ``iterativeShorten`` rebuilds its queue from the live wedges after this many shortenings.
PURGE_INTERVAL = wp.int32(1000)

# A wedge's turn: straight, or the side its smaller angle opens to (``SegmentAngleType``).
TURN_STRAIGHT = wp.int32(0)
TURN_LEFT = wp.int32(1)
TURN_RIGHT = wp.int32(2)

# What an output entry is: an intrinsic edge to trace, an input edge (its two vertices), or a
# path of one point (a target that is a source, or that no source reaches).
ENTRY_TRACED = wp.int32(0)
ENTRY_ORIGINAL = wp.int32(1)
ENTRY_POINT = wp.int32(2)

# A trace point's element: a vertex, a point on an edge (``t`` along ``e.halfedge()``), a point in
# a face (barycentric from ``f.halfedge()``'s tail).
POINT_VERTEX = wp.int32(0)
POINT_EDGE = wp.int32(1)
POINT_FACE = wp.int32(2)

# ``incoming``'s two marks besides a halfedge: unreached, and a source.
UNDISCOVERED = wp.int32(-1)
SOURCE = wp.int32(-2)

# Slots of ``Network.state``.
STATE_NEXT_ID = 0
STATE_HEAP_SIZE = 1
STATE_FREE_SIZE = 2
STATE_SLOT_TOP = 3
STATE_TOUCHED_SIZE = 4
STATE_HEAD = 5
STATE_SIZE = 6

# Slots of the shortening's ``out_status``: the output entries counted (written or not), the pools
# that ran out (the call reruns with them grown), the paths cut off by the step bound.
STATUS_ENTRIES = 0
STATUS_HEAP_OVERFLOW = 1
STATUS_SEGMENT_OVERFLOW = 2
STATUS_UNCONVERGED = 3
STATUS_SIZE = 4

# Slots of the call's flag word: a source or target out of range, a vertex whose faces do not form
# one fan.
FLAG_INVALID_INDEX = 0
FLAG_NONMANIFOLD_VERTEX = 1
FLAG_SIZE = 2


@wp.struct
class InputMesh:
    """The input mesh in ``ManifoldSurfaceMesh`` numbering, with its embedded and intrinsic data."""

    position: wp.array[wp.vec3d]
    next: wp.array[wp.int32]
    vertex: wp.array[wp.int32]
    face: wp.array[wp.int32]
    face_halfedge: wp.array[wp.int32]
    vertex_halfedge: wp.array[wp.int32]
    boundary_out: wp.array[wp.int32]
    length: wp.array[wp.float64]
    angle_sum: wp.array[wp.float64]
    vector_in_face: wp.array[wp.vec2d]
    vector_in_vertex: wp.array[wp.vec2d]
    intrinsic_angle_sum: wp.array[wp.float64]
    signpost: wp.array[wp.float64]


@wp.struct
class Intrinsic:
    """The signpost intrinsic triangulation's mutable tables; the input's are its starting state."""

    next: wp.array[wp.int32]
    vertex: wp.array[wp.int32]
    length: wp.array[wp.float64]
    signpost: wp.array[wp.float64]
    original: wp.array[wp.int32]


@wp.struct
class Network:
    """One path's segments, the per-edge segment stacks, the wedge queue and the flip log."""

    segment_halfedge: wp.array[wp.int32]
    segment_prev: wp.array[wp.int32]
    segment_next: wp.array[wp.int32]
    segment_id: wp.array[wp.int32]
    stack_prev: wp.array[wp.int32]
    stack_next: wp.array[wp.int32]
    free: wp.array[wp.int32]
    edge_front: wp.array[wp.int32]
    edge_back: wp.array[wp.int32]
    heap_angle: wp.array[wp.float64]
    heap_turn: wp.array[wp.int32]
    heap_id: wp.array[wp.int32]
    heap_slot: wp.array[wp.int32]
    touched: wp.array[wp.int32]
    scratch: wp.array[wp.int32]
    state: wp.array[wp.int32]


@wp.func
def twin(h: wp.int32) -> wp.int32:
    return h + 1 - 2 * (h % 2)


@wp.func
def complex_mul(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def complex_div(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    denom = b[0] * b[0] + b[1] * b[1]
    return wp.vec2d((a[0] * b[0] + a[1] * b[1]) / denom, (a[1] * b[0] - a[0] * b[1]) / denom)


@wp.func
def complex_pow(c: wp.vec2d, p: wp.float64) -> wp.vec2d:
    # libstdc++'s ``std::pow(complex<double>, double)``: a positive real stays real, everything
    # else goes through the polar form of ``exp(p log c)``.
    result = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    if c[1] == wp.float64(0.0) and c[0] > wp.float64(0.0):
        result = wp.vec2d(wp.pow(c[0], p), wp.float64(0.0))
    else:
        rho = wp.exp(p * wp.log(wp.sqrt(c[0] * c[0] + c[1] * c[1])))
        theta = p * wp.atan2(c[1], c[0])
        result = wp.vec2d(rho * wp.cos(theta), rho * wp.sin(theta))
    return result


@wp.func
def normalize2(v: wp.vec2d) -> wp.vec2d:
    r = wp.float64(1.0) / wp.sqrt(v[0] * v[0] + v[1] * v[1])
    return wp.vec2d(v[0] * r, v[1] * r)


@wp.func
def from_angle(theta: wp.float64) -> wp.vec2d:
    return wp.vec2d(wp.cos(theta), wp.sin(theta))


@wp.func
def cross2(a: wp.vec2d, b: wp.vec2d) -> wp.float64:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def clamp_d(x: wp.float64, low: wp.float64, high: wp.float64) -> wp.float64:
    return wp.min(wp.max(x, low), high)


@wp.func
def unit3(v: wp.vec3d) -> wp.vec3d:
    return v / wp.length(v)


@wp.func
def corner_angle_from_lengths(
    l_a: wp.float64, l_b: wp.float64, l_opposite: wp.float64
) -> wp.float64:
    # ``EdgeLengthGeometry::cornerAngle``: the angle between sides ``l_a`` and ``l_b``.
    q = (l_a * l_a + l_b * l_b - l_opposite * l_opposite) / (wp.float64(2.0) * l_a * l_b)
    return wp.acos(clamp_d(q, wp.float64(-1.0), wp.float64(1.0)))


# ---------------------------------------------------------------------------------------------
# The input mesh, one thread a corner, face or vertex
# ---------------------------------------------------------------------------------------------


@wp.kernel
def flag_first_corners(twins: wp.array[wp.int32], out_first: wp.array[wp.int32]) -> None:
    # A corner creates its undirected edge when it is the first of the two to traverse it.
    c = wp.int32(wp.tid())
    t = twins[c]
    out_first[c] = wp.where(t == -1 or t > c, 1, 0)


@wp.func
def corner_halfedge(
    twins: wp.array[wp.int32], first_offsets: wp.array[wp.int32], c: wp.int32
) -> wp.int32:
    # ``ManifoldSurfaceMesh``'s halfedge of corner ``c``: ``2 e`` for the corner that created edge
    # ``e``, ``2 e + 1`` for its twin.
    t = twins[c]
    result = 2 * first_offsets[c]
    if t != -1 and t < c:
        result = 2 * first_offsets[t] + 1
    return result


@wp.kernel
def number_corners(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    first_offsets: wp.array[wp.int32],
    out_mesh: InputMesh,
    out_last_corner: wp.array[wp.int32],
    out_degree: wp.array[wp.int32],
    out_boundary_corner: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # The interior halfedge of each corner, and the exterior one across a boundary corner. A vertex
    # takes the last corner leaving it as ``v.halfedge()`` (or, on the boundary, its boundary
    # corner) and counts its outgoing halfedges for the manifold test.
    c = wp.int32(wp.tid())
    f = c // 3
    k = c - 3 * f
    h = corner_halfedge(twins, first_offsets, c)
    tip_corner = 3 * f + (k + 1) % 3
    v = faces[c]
    out_mesh.vertex[h] = v
    out_mesh.face[h] = f
    out_mesh.next[h] = corner_halfedge(twins, first_offsets, tip_corner)
    if k == 0:
        out_mesh.face_halfedge[f] = h
    wp.atomic_max(out_last_corner, v, c)
    wp.atomic_add(out_degree, v, 1)
    if twins[c] == -1:
        x = h + 1
        w = faces[tip_corner]
        out_mesh.vertex[x] = w
        out_mesh.face[x] = -1
        if wp.atomic_add(out_degree, w, 1 << 16) >= 1 << 16:
            # A second exterior halfedge leaving ``w``: two boundary loops pass through it.
            out_flags[FLAG_NONMANIFOLD_VERTEX] = 1
        out_mesh.boundary_out[w] = x
        out_boundary_corner[v] = c


@wp.kernel
def face_geometry(
    out_corner_angle: wp.array[wp.float64],
    out_intrinsic_corner_angle: wp.array[wp.float64],
    out_mesh: InputMesh,
) -> None:
    # ``EmbeddedGeometryInterface`` per face: edge lengths (written by ``e.halfedge()``'s face),
    # corner angles from unit vectors, the intrinsic corner angles from the lengths, the face
    # layout from ``f.halfedge()`` along the x axis; and the exterior ``next`` across each
    # boundary edge, to the exterior halfedge leaving its tip.
    f = wp.int32(wp.tid())
    h_ab = out_mesh.face_halfedge[f]
    h_bc = out_mesh.next[h_ab]
    h_ca = out_mesh.next[h_bc]
    p_a = out_mesh.position[out_mesh.vertex[h_ab]]
    p_b = out_mesh.position[out_mesh.vertex[h_bc]]
    p_c = out_mesh.position[out_mesh.vertex[h_ca]]
    # ``norm(p[e.halfedge().vertex()] - p[e.halfedge().next().vertex()])``, read from whichever
    # side of the edge this face holds.
    l_ab = wp.where(h_ab % 2 == 0, wp.length(p_a - p_b), wp.length(p_b - p_a))
    l_bc = wp.where(h_bc % 2 == 0, wp.length(p_b - p_c), wp.length(p_c - p_b))
    l_ca = wp.where(h_ca % 2 == 0, wp.length(p_c - p_a), wp.length(p_a - p_c))
    if h_ab % 2 == 0:
        out_mesh.length[h_ab // 2] = l_ab
    if h_bc % 2 == 0:
        out_mesh.length[h_bc // 2] = l_bc
    if h_ca % 2 == 0:
        out_mesh.length[h_ca // 2] = l_ca

    out_corner_angle[h_bc] = wp.acos(
        clamp_d(wp.dot(unit3(p_c - p_b), unit3(p_a - p_b)), wp.float64(-1.0), wp.float64(1.0))
    )
    out_corner_angle[h_ca] = wp.acos(
        clamp_d(wp.dot(unit3(p_a - p_c), unit3(p_b - p_c)), wp.float64(-1.0), wp.float64(1.0))
    )
    out_corner_angle[h_ab] = wp.acos(
        clamp_d(wp.dot(unit3(p_b - p_a), unit3(p_c - p_a)), wp.float64(-1.0), wp.float64(1.0))
    )
    out_intrinsic_corner_angle[h_ab] = corner_angle_from_lengths(l_ab, l_ca, l_bc)
    out_intrinsic_corner_angle[h_bc] = corner_angle_from_lengths(l_bc, l_ab, l_ca)
    out_intrinsic_corner_angle[h_ca] = corner_angle_from_lengths(l_ca, l_bc, l_ab)

    normal = wp.cross(p_a, p_b) + wp.cross(p_b, p_c) + wp.cross(p_c, p_a)
    area = wp.float64(0.5) * wp.length(normal)
    height = wp.float64(2.0) * area / l_ab
    width = wp.sqrt(wp.max(wp.float64(0.0), l_ca * l_ca - height * height))
    if l_bc * l_bc > l_ab * l_ab + l_ca * l_ca:
        width = -width
    out_mesh.vector_in_face[h_ab] = wp.vec2d(l_ab, wp.float64(0.0))
    out_mesh.vector_in_face[h_bc] = wp.vec2d(width - l_ab, height)
    out_mesh.vector_in_face[h_ca] = wp.vec2d(-width, -height)

    h = h_ab
    for _k in range(3):
        if h % 2 == 0:
            if out_mesh.face[h + 1] == -1:
                out_mesh.next[h + 1] = out_mesh.boundary_out[out_mesh.vertex[h]]
        h = out_mesh.next[h]


@wp.kernel
def vertex_geometry(
    twins: wp.array[wp.int32],
    first_offsets: wp.array[wp.int32],
    last_corner: wp.array[wp.int32],
    degree: wp.array[wp.int32],
    boundary_corner: wp.array[wp.int32],
    corner_angle: wp.array[wp.float64],
    intrinsic_corner_angle: wp.array[wp.float64],
    out_mesh: InputMesh,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``v.halfedge()``; ``ManifoldSurfaceMesh``'s manifold test (the orbit covers every outgoing
    # halfedge, which fails at a vertex whose faces form two fans); the two angle sums; and,
    # orbiting counter-clockwise from ``v.halfedge()``, the input's ``halfedgeVectorsInVertex``
    # (corner angles rescaled to a full turn, half a turn at the boundary) and the signposts (the
    # intrinsic corner angles unscaled).
    v = wp.int32(wp.tid())
    corner = wp.where(boundary_corner[v] != -1, boundary_corner[v], last_corner[v])
    if corner == -1:
        out_mesh.vertex_halfedge[v] = -1
        return
    first = corner_halfedge(twins, first_offsets, corner)
    out_mesh.vertex_halfedge[v] = first
    n_out = degree[v] % (1 << 16) + degree[v] // (1 << 16)
    steps = wp.int32(0)
    h = first
    while True:
        steps += 1
        if steps > n_out:
            break
        h = out_mesh.next[twin(h)]
        if h == first:
            break
    if steps != n_out:
        out_flags[FLAG_NONMANIFOLD_VERTEX] = 1
        return

    # Both angle sums add the corners in halfedge order, as ``mesh.corners()`` visits them: the
    # last bit of a sum decides a wedge whose two sides tie exactly (``unit_box``).
    angle_sum = wp.float64(0.0)
    intrinsic_angle_sum = wp.float64(0.0)
    previous = wp.int32(-1)
    for _i in range(n_out):
        smallest = wp.int32(-1)
        h = first
        while True:
            if out_mesh.face[h] != -1 and h > previous and (smallest == -1 or h < smallest):
                smallest = h
            h = out_mesh.next[twin(h)]
            if h == first:
                break
        if smallest == -1:
            break
        angle_sum += corner_angle[smallest]
        intrinsic_angle_sum += intrinsic_corner_angle[smallest]
        previous = smallest
    out_mesh.angle_sum[v] = angle_sum
    out_mesh.intrinsic_angle_sum[v] = intrinsic_angle_sum

    turn = wp.where(out_mesh.boundary_out[v] != -1, PI_D, TWO_PI_D)
    scale = turn / angle_sum
    coordinate = wp.float64(0.0)
    running = wp.float64(0.0)
    h = first
    while True:
        out_mesh.vector_in_vertex[h] = from_angle(coordinate) * out_mesh.length[h // 2]
        out_mesh.signpost[h] = running
        if out_mesh.face[h] == -1:
            break
        coordinate += scale * corner_angle[h]
        running += intrinsic_corner_angle[h]
        h = twin(out_mesh.next[out_mesh.next[h]])
        if h == first:
            break


# ---------------------------------------------------------------------------------------------
# Shortest edge paths
# ---------------------------------------------------------------------------------------------


@wp.kernel
def seed_sources(
    sources: wp.array[wp.int32],
    out_distance: wp.array[wp.float64],
    out_source_mask: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    s = sources[i]
    if s < 0 or s >= out_distance.shape[0]:
        out_flags[FLAG_INVALID_INDEX] = 1
        return
    out_distance[s] = wp.float64(0.0)
    out_source_mask[s] = 1


@wp.kernel
def check_targets(
    targets: wp.array[wp.int32], n_vertices: wp.int32, out_flags: wp.array[wp.int32]
) -> None:
    p = wp.int32(wp.tid())
    if targets[p] < 0 or targets[p] >= n_vertices:
        out_flags[FLAG_INVALID_INDEX] = 1


@wp.func
def relax_vertex(mesh: InputMesh, distance: wp.array[wp.float64], v: wp.int32) -> wp.bool:
    # Lower ``v``'s distance to the least ``d(u) + length`` over its neighbours ``u``: the sum
    # ``shortestEdgePath`` pushes for ``v`` when it settles ``u``. ``True`` when it moved.
    first = mesh.vertex_halfedge[v]
    if first == -1:
        return False
    best = distance[v]
    h = first
    while True:
        best = wp.min(best, distance[mesh.vertex[twin(h)]] + mesh.length[h // 2])
        h = mesh.next[twin(h)]
        if h == first:
            break
    moved = best < distance[v]
    if moved:
        distance[v] = best
    return moved


# Rounds a block of ``relax_distances`` runs over its own vertices in one pass, at most. Measured
# (2026-11-08, one source, CUDA): 4 best on ``sphere_med`` and bunny, 16 on dragon, 8 within 10 %
# of both and best on ``sphere_large``; past it a block's rounds chase values its neighbours are
# still lowering.
BLOCK_RELAX_ROUNDS = wp.int32(8)


@wp.kernel
def relax_distances(
    mesh: InputMesh,
    distance: wp.array[wp.float64],
    order: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_state: wp.array[wp.int32],
) -> None:
    # One relaxation pass, in place: a neighbour's value read before or after its own update this
    # pass is a path's sum either way, so any schedule of passes reaches the same least sums. The
    # vertices go in Hilbert order (``order``), tiled, so a block's vertices are a patch of the
    # mesh, and the block relaxes them round after round (``wp.tile_sum`` the barrier) until none
    # moves: a pass carries a distance across the patch rather than an edge or two.
    block, lane = wp.tid()
    i = block * wp.block_dim() + lane
    v = wp.int32(-1)
    if i < n_vertices:
        v = order[i]
    moved_any = wp.int32(0)
    for _round in range(BLOCK_RELAX_ROUNDS):
        moved = wp.int32(0)
        if v >= 0 and relax_vertex(mesh, distance, v):
            moved = wp.int32(1)
        if wp.tile_sum(wp.tile(moved))[0] == 0:
            break
        moved_any = wp.int32(1)
    if moved_any != 0 and lane == 0:
        out_state[LOOP_PROGRESS] = 1


@wp.kernel
def settle_incoming(
    mesh: InputMesh,
    distance: wp.array[wp.float64],
    source_mask: wp.array[wp.int32],
    out_incoming: wp.array[wp.int32],
) -> None:
    # The halfedge ``shortestEdgePath`` settles ``v`` by: its queue pops the least
    # ``(distance, halfedge)``, and every entry for ``v`` at ``v``'s distance comes from a nearer
    # neighbour, settled first.
    v = wp.int32(wp.tid())
    result = UNDISCOVERED
    if source_mask[v] != 0:
        result = SOURCE
    else:
        first = mesh.vertex_halfedge[v]
        d = distance[v]
        if first != -1 and d < wp.float64(wp.inf):
            h = first
            while True:
                u = mesh.vertex[twin(h)]
                g = twin(h)
                if distance[u] < d and distance[u] + mesh.length[h // 2] == d:
                    if result == UNDISCOVERED or g < result:
                        result = g
                h = mesh.next[twin(h)]
                if h == first:
                    break
    out_incoming[v] = result


# ---------------------------------------------------------------------------------------------
# The CPU device's shortest-edge tree
# ---------------------------------------------------------------------------------------------


@wp.func
def dijkstra_push(
    distance: wp.array[wp.float64],
    halfedge: wp.array[wp.int32],
    state: wp.array[wp.int32],
    d: wp.float64,
    h: wp.int32,
) -> None:
    i = state[0]
    state[0] = i + 1
    while i > 0:
        parent = (i - 1) // 2
        if distance[parent] < d or (distance[parent] == d and halfedge[parent] < h):
            break
        distance[i] = distance[parent]
        halfedge[i] = halfedge[parent]
        i = parent
    distance[i] = d
    halfedge[i] = h


@wp.func
def dijkstra_pop(
    distance: wp.array[wp.float64], halfedge: wp.array[wp.int32], state: wp.array[wp.int32]
) -> tuple[wp.float64, wp.int32]:
    top_distance = distance[0]
    top_halfedge = halfedge[0]
    size = state[0] - 1
    state[0] = size
    d = distance[size]
    h = halfedge[size]
    i = wp.int32(0)
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        right = child + 1
        if right < size and (
            distance[right] < distance[child]
            or (distance[right] == distance[child] and halfedge[right] < halfedge[child])
        ):
            child = right
        if d < distance[child] or (d == distance[child] and h < halfedge[child]):
            break
        distance[i] = distance[child]
        halfedge[i] = halfedge[child]
        i = child
    if size > 0:
        distance[i] = d
        halfedge[i] = h
    return top_distance, top_halfedge


@wp.func
def enqueue_neighbors(
    mesh: InputMesh,
    incoming: wp.array[wp.int32],
    distance: wp.array[wp.float64],
    halfedge: wp.array[wp.int32],
    state: wp.array[wp.int32],
    v: wp.int32,
    d: wp.float64,
) -> None:
    first = mesh.vertex_halfedge[v]
    if first == -1:
        return
    h = first
    while True:
        tip = mesh.vertex[twin(h)]
        if incoming[tip] == UNDISCOVERED:
            dijkstra_push(distance, halfedge, state, d + mesh.length[h // 2], h)
        h = mesh.next[twin(h)]
        if h == first:
            break


@wp.kernel
def shortest_edge_tree_serial(
    mesh: InputMesh,
    source_mask: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    distance: wp.array[wp.float64],
    halfedge: wp.array[wp.int32],
    wanted: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_incoming: wp.array[wp.int32],
) -> None:
    # The CPU device's tree: ``shortestEdgePath``'s lazy Dijkstra itself, from every source at
    # once, until every target is settled. ``distance`` and ``halfedge`` are its queue.
    n_vertices = out_incoming.shape[0]
    state[0] = 0
    for v in range(n_vertices):
        out_incoming[v] = UNDISCOVERED
        wanted[v] = 0
    for v in range(n_vertices):
        if source_mask[v] != 0:
            out_incoming[v] = SOURCE
    for v in range(n_vertices):
        if source_mask[v] != 0:
            enqueue_neighbors(mesh, out_incoming, distance, halfedge, state, v, wp.float64(0.0))
    remaining = wp.int32(0)
    for i in range(targets.shape[0]):
        t = targets[i]
        if t >= 0 and t < n_vertices:
            if out_incoming[t] == UNDISCOVERED and wanted[t] == 0:
                wanted[t] = 1
                remaining += 1
    while state[0] > 0 and remaining > 0:
        d, h = dijkstra_pop(distance, halfedge, state)
        v = mesh.vertex[twin(h)]
        if out_incoming[v] != UNDISCOVERED:
            continue
        out_incoming[v] = h
        remaining -= wanted[v]
        enqueue_neighbors(mesh, out_incoming, distance, halfedge, state, v, d)


# ---------------------------------------------------------------------------------------------
# The signpost intrinsic triangulation, one thread on the CPU device
# ---------------------------------------------------------------------------------------------


@wp.func
def layout_triangle_vertex(
    p_a: wp.vec2d, p_b: wp.vec2d, l_bc: wp.float64, l_ca: wp.float64
) -> wp.vec2d:
    # ``layoutTriangleVertex``: the third corner from Heron's area and the projected width.
    ab = p_b - p_a
    l_ab = wp.sqrt(ab[0] * ab[0] + ab[1] * ab[1])
    s = (l_ab + l_bc + l_ca) / wp.float64(2.0)
    area = wp.sqrt(wp.max(wp.float64(0.0), s * (s - l_ab) * (s - l_bc) * (s - l_ca)))
    h = wp.float64(2.0) * area / l_ab
    w = (l_ab * l_ab - l_bc * l_bc + l_ca * l_ca) / (wp.float64(2.0) * l_ab)
    unit = wp.vec2d(ab[0] / l_ab, ab[1] / l_ab)
    perp = wp.vec2d(-unit[1], unit[0])
    return p_a + w * unit + h * perp


@wp.func
def is_interior(mesh: InputMesh, h: wp.int32) -> wp.bool:
    # Flips never touch the boundary, so a halfedge is exterior on the intrinsic triangulation
    # exactly when it is on the input.
    return mesh.face[h] != -1


@wp.func
def turn_of(mesh: InputMesh, v: wp.int32) -> wp.float64:
    return wp.where(mesh.boundary_out[v] != -1, PI_D, TWO_PI_D)


@wp.func
def update_signpost(mesh: InputMesh, tri: Intrinsic, h: wp.int32) -> None:
    # ``updateAngleFromCWNeighor`` for an interior halfedge with an interior twin, the only kind a
    # flip creates: the clockwise neighbour's angle plus the corner between them, wrapped at an
    # interior vertex.
    cw = tri.next[twin(h)]
    opposite = tri.next[cw]
    other = tri.next[opposite]
    corner = corner_angle_from_lengths(
        tri.length[cw // 2], tri.length[other // 2], tri.length[opposite // 2]
    )
    angle = tri.signpost[cw] + corner
    v = tri.vertex[h]
    if mesh.boundary_out[v] == -1:
        angle = wp.mod(angle, mesh.intrinsic_angle_sum[v])
    tri.signpost[h] = angle


@wp.func
def flip_connectivity(tri: Intrinsic, e: wp.int32) -> wp.bool:
    # ``SurfaceMesh::flip``: rotate edge ``e`` counter-clockwise inside its two triangles.
    ha1 = 2 * e
    ha2 = tri.next[ha1]
    ha3 = tri.next[ha2]
    hb1 = ha1 + 1
    hb2 = tri.next[hb1]
    hb3 = tri.next[hb2]
    if ha2 == hb1 or hb2 == ha1:
        return False
    vc = tri.vertex[ha3]
    vd = tri.vertex[hb3]
    tri.next[ha1] = hb3
    tri.next[hb3] = ha2
    tri.next[ha2] = ha1
    tri.next[hb1] = ha3
    tri.next[ha3] = hb2
    tri.next[hb2] = hb1
    tri.vertex[ha1] = vc
    tri.vertex[hb1] = vd
    return True


@wp.func
def touch(net: Network, tri: Intrinsic, e: wp.int32) -> None:
    # Log the edge and the four halfedges around it a flip is about to rewire. Past the log's
    # capacity the size keeps counting, which tells the rewind to restore every table instead.
    size = net.state[STATE_TOUCHED_SIZE]
    net.state[STATE_TOUCHED_SIZE] = size + 5
    if size + 5 <= net.touched.shape[0]:
        h = 2 * e
        net.touched[size] = e
        net.touched[size + 1] = tri.next[h]
        net.touched[size + 2] = tri.next[tri.next[h]]
        net.touched[size + 3] = tri.next[h + 1]
        net.touched[size + 4] = tri.next[tri.next[h + 1]]


@wp.func
def flip_edge_if_possible(mesh: InputMesh, tri: Intrinsic, net: Network, e: wp.int32) -> wp.bool:
    # ``SignpostIntrinsicTriangulation::flipEdgeIfPossible``.
    h = 2 * e
    if not is_interior(mesh, h) or not is_interior(mesh, h + 1):
        return False
    ha1 = tri.next[h]
    ha2 = tri.next[ha1]
    hb1 = tri.next[h + 1]
    hb2 = tri.next[hb1]
    l01 = tri.length[ha1 // 2]
    l12 = tri.length[ha2 // 2]
    l23 = tri.length[hb1 // 2]
    l30 = tri.length[hb2 // 2]
    l02 = tri.length[e]
    p3 = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    p0 = wp.vec2d(l30, wp.float64(0.0))
    p2 = layout_triangle_vertex(p3, p0, l02, l23)
    p1 = layout_triangle_vertex(p2, p0, l01, l12)
    a1 = cross2(p1 - p0, p3 - p0)
    a2 = cross2(p3 - p2, p1 - p2)
    area_eps = TRIANGLE_TEST_EPS * (a1 + a2)
    if a1 < area_eps or a2 < area_eps:
        return False
    d = p1 - p3
    new_length = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if not wp.isfinite(new_length):
        return False
    if ha1 == h + 1 or hb1 == h:
        return False
    touch(net, tri, e)
    flip_connectivity(tri, e)
    tri.length[e] = new_length
    update_signpost(mesh, tri, h)
    update_signpost(mesh, tri, h + 1)
    tri.original[e] = 0
    return True


@wp.func
def halfedge_vector(mesh: InputMesh, tri: Intrinsic, h: wp.int32) -> wp.vec2d:
    # ``SignpostIntrinsicTriangulation::halfedgeVector``: the signpost rescaled to the input
    # vertex's tangent plane, at the edge's length.
    v = tri.vertex[h]
    scale = wp.float64(1.0) / (mesh.intrinsic_angle_sum[v] / turn_of(mesh, v))
    return from_angle(tri.signpost[h] * scale) * tri.length[h // 2]


# ---------------------------------------------------------------------------------------------
# The path network
# ---------------------------------------------------------------------------------------------


@wp.func
def outside_segment(net: Network, h: wp.int32) -> wp.int32:
    # ``getOutsideSegment``: the stack's front faces ``e.halfedge()``, its back the twin.
    e = h // 2
    return wp.where(h == 2 * e, net.edge_front[e], net.edge_back[e])


@wp.func
def push_outside(net: Network, h: wp.int32, s: wp.int32) -> None:
    e = h // 2
    if h == 2 * e:
        front = net.edge_front[e]
        net.stack_prev[s] = -1
        net.stack_next[s] = front
        if front != -1:
            net.stack_prev[front] = s
        else:
            net.edge_back[e] = s
        net.edge_front[e] = s
    else:
        back = net.edge_back[e]
        net.stack_next[s] = -1
        net.stack_prev[s] = back
        if back != -1:
            net.stack_next[back] = s
        else:
            net.edge_front[e] = s
        net.edge_back[e] = s


@wp.func
def pop_outside(net: Network, h: wp.int32) -> None:
    e = h // 2
    if h == 2 * e:
        s = net.edge_front[e]
        front = net.stack_next[s]
        net.edge_front[e] = front
        if front != -1:
            net.stack_prev[front] = -1
        else:
            net.edge_back[e] = -1
    else:
        s = net.edge_back[e]
        back = net.stack_prev[s]
        net.edge_back[e] = back
        if back != -1:
            net.stack_next[back] = -1
        else:
            net.edge_front[e] = -1


@wp.func
def new_segment(net: Network, h: wp.int32, prev: wp.int32) -> wp.int32:
    # A fresh slot (a freed one first) under the next unique id; ``-1`` when the pool is out.
    free_size = net.state[STATE_FREE_SIZE]
    s = wp.int32(-1)
    if free_size > 0:
        s = net.free[free_size - 1]
        net.state[STATE_FREE_SIZE] = free_size - 1
    else:
        top = net.state[STATE_SLOT_TOP]
        if top < net.segment_halfedge.shape[0]:
            s = top
            net.state[STATE_SLOT_TOP] = top + 1
    if s != -1:
        net.segment_halfedge[s] = h
        net.segment_prev[s] = prev
        net.segment_next[s] = -1
        net.segment_id[s] = net.state[STATE_NEXT_ID]
        net.state[STATE_NEXT_ID] = net.state[STATE_NEXT_ID] + 1
    return s


@wp.func
def free_segment(net: Network, s: wp.int32) -> None:
    net.segment_id[s] = -1
    size = net.state[STATE_FREE_SIZE]
    net.free[size] = s
    net.state[STATE_FREE_SIZE] = size + 1


@wp.func
def side_angles(
    mesh: InputMesh, tri: Intrinsic, h_prev: wp.int32, h_next: wp.int32
) -> tuple[wp.float64, wp.float64]:
    # ``measureSideAngles``: the wedge angles left and right of the path at ``h_next``'s tail,
    # from the signposts; across the boundary a side is unbounded.
    v = tri.vertex[h_next]
    s = mesh.intrinsic_angle_sum[v]
    angle_in = tri.signpost[twin(h_prev)]
    angle_out = tri.signpost[h_next]
    boundary = mesh.boundary_out[v] != -1
    infinity = wp.float64(wp.inf)
    right = wp.float64(0.0)
    if angle_in < angle_out:
        right = angle_out - angle_in
    elif boundary:
        right = infinity
    else:
        right = (s - angle_in) + angle_out
    left = wp.float64(0.0)
    if angle_out < angle_in:
        left = angle_in - angle_out
    elif boundary:
        left = infinity
    else:
        left = (s - angle_out) + angle_in
    return left, right


@wp.func
def min_wedge_angle(mesh: InputMesh, tri: Intrinsic, net: Network, s: wp.int32) -> wp.float64:
    prev = net.segment_prev[s]
    result = PI_D
    if prev != -1:
        left, right = side_angles(mesh, tri, net.segment_halfedge[prev], net.segment_halfedge[s])
        result = wp.min(left, right)
    return result


@wp.func
def heap_less(
    net: Network, i: wp.int32, angle: wp.float64, turn: wp.int32, ident: wp.int32
) -> wp.bool:
    # The queue's ``std::greater`` order on ``(angle, type, segment)``: is entry ``i`` first.
    a = net.heap_angle[i]
    if a != angle:
        return a < angle
    t = net.heap_turn[i]
    if t != turn:
        return t < turn
    return net.heap_id[i] < ident


@wp.func
def heap_push(
    net: Network, angle: wp.float64, turn: wp.int32, s: wp.int32, out_status: wp.array[wp.int32]
) -> None:
    i = net.state[STATE_HEAP_SIZE]
    if i >= net.heap_angle.shape[0]:
        out_status[STATUS_HEAP_OVERFLOW] = 1
        return
    net.state[STATE_HEAP_SIZE] = i + 1
    ident = net.segment_id[s]
    while i > 0:
        parent = (i - 1) // 2
        if heap_less(net, parent, angle, turn, ident):
            break
        net.heap_angle[i] = net.heap_angle[parent]
        net.heap_turn[i] = net.heap_turn[parent]
        net.heap_id[i] = net.heap_id[parent]
        net.heap_slot[i] = net.heap_slot[parent]
        i = parent
    net.heap_angle[i] = angle
    net.heap_turn[i] = turn
    net.heap_id[i] = ident
    net.heap_slot[i] = s


@wp.func
def heap_pop(net: Network) -> tuple[wp.float64, wp.int32, wp.int32, wp.int32]:
    top_angle = net.heap_angle[0]
    top_turn = net.heap_turn[0]
    top_id = net.heap_id[0]
    top_slot = net.heap_slot[0]
    size = net.state[STATE_HEAP_SIZE] - 1
    net.state[STATE_HEAP_SIZE] = size
    angle = net.heap_angle[size]
    turn = net.heap_turn[size]
    ident = net.heap_id[size]
    slot = net.heap_slot[size]
    i = wp.int32(0)
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        right = child + 1
        if right < size and heap_less(
            net, right, net.heap_angle[child], net.heap_turn[child], net.heap_id[child]
        ):
            child = right
        if not heap_less(net, child, angle, turn, ident):
            break
        net.heap_angle[i] = net.heap_angle[child]
        net.heap_turn[i] = net.heap_turn[child]
        net.heap_id[i] = net.heap_id[child]
        net.heap_slot[i] = net.heap_slot[child]
        i = child
    if size > 0:
        net.heap_angle[i] = angle
        net.heap_turn[i] = turn
        net.heap_id[i] = ident
        net.heap_slot[i] = slot
    return top_angle, top_turn, top_id, top_slot


@wp.func
def queue_wedge(
    mesh: InputMesh, tri: Intrinsic, net: Network, s: wp.int32, out_status: wp.array[wp.int32]
) -> None:
    # ``addToWedgeAngleQueue``: both sides of the wedge ending at ``s``, smaller first, each only
    # when it is not straight. No clearance test here: a wedge blocked now may clear later.
    if s == -1:
        return
    prev = net.segment_prev[s]
    if prev == -1:
        return
    left, right = side_angles(mesh, tri, net.segment_halfedge[prev], net.segment_halfedge[s])
    limit = PI_D - EPS_ANGLE
    min_angle = right
    max_angle = left
    min_turn = TURN_RIGHT
    max_turn = TURN_LEFT
    if left < right:
        min_angle = left
        max_angle = right
        min_turn = TURN_LEFT
        max_turn = TURN_RIGHT
    if min_angle > limit:
        return
    heap_push(net, min_angle, min_turn, s, out_status)
    if max_angle > limit:
        return
    heap_push(net, max_angle, max_turn, s, out_status)


@wp.func
def wedge_is_clear(tri: Intrinsic, net: Network, s: wp.int32, turn: wp.int32) -> wp.bool:
    # ``wedgeIsClear``: both bounding segments are outermost on their edges and no path edge
    # enters the wedge.
    prev = net.segment_prev[s]
    h_prev = net.segment_halfedge[prev]
    h_next = net.segment_halfedge[s]
    clear = wp.bool(True)
    if turn == TURN_LEFT:
        if outside_segment(net, h_prev) != prev or outside_segment(net, h_next) != s:
            clear = False
        else:
            h = tri.next[h_prev]
            while h != h_next:
                if net.edge_front[h // 2] != -1:
                    clear = False
                    break
                h = tri.next[twin(h)]
    else:
        if outside_segment(net, twin(h_prev)) != prev or outside_segment(net, twin(h_next)) != s:
            clear = False
        else:
            h = twin(tri.next[tri.next[twin(h_prev)]])
            while h != h_next:
                if net.edge_front[h // 2] != -1:
                    clear = False
                    break
                h = twin(tri.next[tri.next[h]])
    return clear


@wp.func
def replace_segments(
    mesh: InputMesh,
    tri: Intrinsic,
    net: Network,
    s: wp.int32,
    turn: wp.int32,
    n_new: wp.int32,
    backwards: wp.bool,
    out_status: wp.array[wp.int32],
) -> wp.bool:
    # ``FlipEdgePath::replacePathSegment``: swap the wedge's two segments for the ``n_new``
    # halfedges in ``scratch`` (read back to front and twinned when ``backwards``).
    h_next = net.segment_halfedge[s]
    prev = net.segment_prev[s]
    next_next = net.segment_next[s]
    h_prev = net.segment_halfedge[prev]
    prev_prev = net.segment_prev[prev]
    if turn == TURN_LEFT:
        pop_outside(net, h_prev)
        pop_outside(net, h_next)
    else:
        pop_outside(net, twin(h_prev))
        pop_outside(net, twin(h_next))
    free_segment(net, prev)
    free_segment(net, s)

    loop_prev = prev_prev
    for i in range(n_new):
        h = net.scratch[i]
        if backwards:
            h = twin(net.scratch[n_new - 1 - i])
        added = new_segment(net, h, loop_prev)
        if added == -1:
            out_status[STATUS_SEGMENT_OVERFLOW] = 1
            return False
        if turn == TURN_LEFT:
            push_outside(net, twin(h), added)
        else:
            push_outside(net, h, added)
        if loop_prev != -1:
            net.segment_next[loop_prev] = added
        else:
            net.state[STATE_HEAD] = added
        queue_wedge(mesh, tri, net, added, out_status)
        loop_prev = added

    net.segment_next[loop_prev] = next_next
    if next_next != -1:
        net.segment_prev[next_next] = loop_prev
        queue_wedge(mesh, tri, net, next_next, out_status)

    # Outside segments sharing the wedge's two edges may have been unblocked.
    if turn == TURN_LEFT:
        queue_wedge(mesh, tri, net, outside_segment(net, h_next), out_status)
        queue_wedge(mesh, tri, net, outside_segment(net, h_prev), out_status)
    else:
        queue_wedge(mesh, tri, net, outside_segment(net, twin(h_next)), out_status)
        queue_wedge(mesh, tri, net, outside_segment(net, twin(h_prev)), out_status)
    return True


@wp.func
def shorten_at(
    mesh: InputMesh,
    tri: Intrinsic,
    net: Network,
    s: wp.int32,
    turn: wp.int32,
    out_status: wp.array[wp.int32],
) -> wp.bool:
    # ``locallyShortenAt``: flip every flippable edge inside the wedge (walking it clockwise,
    # stepping back after each flip), then route the path along the wedge's far side if that is
    # not longer. ``False`` when the step bound or a pool ran out.
    prev = net.segment_prev[s]
    h_prev = net.segment_halfedge[prev]
    h_next = net.segment_halfedge[s]
    initial_length = tri.length[h_prev // 2] + tri.length[h_next // 2]
    s_prev = h_prev
    s_next = h_next
    backwards = wp.bool(False)
    if turn == TURN_RIGHT:
        s_prev = twin(h_next)
        s_next = twin(h_prev)
        backwards = True

    limit = net.scratch.shape[0] * 4 + 64
    h = tri.next[s_prev]
    s_prev_twin = twin(s_prev)
    steps = wp.int32(0)
    while h != s_next:
        steps += 1
        if steps > limit:
            return False
        if h == s_prev_twin:
            h = tri.next[twin(h)]
            continue
        if flip_edge_if_possible(mesh, tri, net, h // 2):
            h = twin(tri.next[twin(h)])
        else:
            h = tri.next[twin(h)]

    n_new = wp.int32(0)
    new_length = wp.float64(0.0)
    h = tri.next[s_prev]
    while True:
        if n_new >= net.scratch.shape[0]:
            return False
        side = tri.next[h]
        net.scratch[n_new] = twin(side)
        n_new += 1
        new_length += tri.length[side // 2]
        if h == s_next:
            break
        h = tri.next[twin(h)]

    if new_length > initial_length:
        return True
    return replace_segments(mesh, tri, net, s, turn, n_new, backwards, out_status)


@wp.func
def queue_all_wedges(
    mesh: InputMesh, tri: Intrinsic, net: Network, out_status: wp.array[wp.int32]
) -> None:
    net.state[STATE_HEAP_SIZE] = 0
    s = net.state[STATE_HEAD]
    while s != -1:
        queue_wedge(mesh, tri, net, s, out_status)
        s = net.segment_next[s]


@wp.func
def shorten_path(
    mesh: InputMesh, tri: Intrinsic, net: Network, out_status: wp.array[wp.int32]
) -> wp.bool:
    # ``iterativeShorten`` with no iteration or length limit, as potpourri3d calls it. ``False``
    # when the step bound or a pool ran out.
    n_iterations = wp.int32(0)
    limit = net.scratch.shape[0] * 64 + 4096
    while net.state[STATE_HEAP_SIZE] > 0:
        angle, turn, ident, s = heap_pop(net)
        if net.segment_id[s] != ident:
            continue
        if min_wedge_angle(mesh, tri, net, s) != angle:
            continue
        if not wedge_is_clear(tri, net, s, turn):
            continue
        if not shorten_at(mesh, tri, net, s, turn, out_status):
            return False
        if out_status[STATUS_HEAP_OVERFLOW] != 0 or out_status[STATUS_SEGMENT_OVERFLOW] != 0:
            return False
        n_iterations += 1
        if n_iterations > limit:
            return False
        if n_iterations % PURGE_INTERVAL == 0:
            queue_all_wedges(mesh, tri, net, out_status)
    return True


@wp.func
def rewind(mesh: InputMesh, tri: Intrinsic, net: Network) -> None:
    # Empty the segment stacks and restore what the flips touched from the input tables (all of
    # them when the log ran out).
    s = net.state[STATE_HEAD]
    while s != -1:
        e = net.segment_halfedge[s] // 2
        net.edge_front[e] = -1
        net.edge_back[e] = -1
        s = net.segment_next[s]
    size = net.state[STATE_TOUCHED_SIZE]
    if size <= net.touched.shape[0]:
        for i in range(size // 5):
            e = net.touched[5 * i]
            for j in range(5):
                h = net.touched[5 * i + j]
                if j == 0:
                    h = 2 * e
                tri.next[h] = mesh.next[h]
                tri.vertex[h] = mesh.vertex[h]
                tri.signpost[h] = mesh.signpost[h]
            h = 2 * e + 1
            tri.next[h] = mesh.next[h]
            tri.vertex[h] = mesh.vertex[h]
            tri.signpost[h] = mesh.signpost[h]
            tri.length[e] = mesh.length[e]
            tri.original[e] = 1
    else:
        for h in range(tri.next.shape[0]):
            tri.next[h] = mesh.next[h]
            tri.vertex[h] = mesh.vertex[h]
            tri.signpost[h] = mesh.signpost[h]
        for e in range(tri.length.shape[0]):
            tri.length[e] = mesh.length[e]
            tri.original[e] = 1


@wp.func
def emit_entry(
    i: wp.int32,
    path: wp.int32,
    kind: wp.int32,
    tail: wp.int32,
    tip: wp.int32,
    vector: wp.vec2d,
    out_path: wp.array[wp.int32],
    out_kind: wp.array[wp.int32],
    out_tail: wp.array[wp.int32],
    out_tip: wp.array[wp.int32],
    out_vector: wp.array[wp.vec2d],
) -> None:
    if i < out_kind.shape[0]:
        out_path[i] = path
        out_kind[i] = kind
        out_tail[i] = tail
        out_tip[i] = tip
        out_vector[i] = vector


@wp.kernel
def shorten_paths(
    mesh: InputMesh,
    tri: Intrinsic,
    net: Network,
    incoming: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_path: wp.array[wp.int32],
    out_kind: wp.array[wp.int32],
    out_tail: wp.array[wp.int32],
    out_tip: wp.array[wp.int32],
    out_vector: wp.array[wp.vec2d],
    out_status: wp.array[wp.int32],
) -> None:
    # Every path, one after another from the same triangulation: ``find_geodesic_path`` from the
    # nearest source to each target, written as the intrinsic halfedges of the shortened path.
    for e in range(net.edge_front.shape[0]):
        net.edge_front[e] = -1
        net.edge_back[e] = -1

    n_entries = wp.int32(0)
    out_offsets[0] = 0
    zero = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    for p in range(targets.shape[0]):
        target = targets[p]
        if incoming[target] < 0:
            emit_entry(
                n_entries, p, ENTRY_POINT, target, target, zero,
                out_path, out_kind, out_tail, out_tip, out_vector,
            )  # fmt: skip
            n_entries += 1
            out_offsets[p + 1] = n_entries
            continue

        # The Dijkstra path, source to target, as potpourri3d hands it to ``reinitializePath``.
        n_path = wp.int32(0)
        v = target
        while incoming[v] >= 0:
            h = incoming[v]
            net.scratch[n_path] = h
            n_path += 1
            v = mesh.vertex[h]
        for k in range(STATE_SIZE):
            net.state[k] = 0
        net.state[STATE_HEAD] = -1
        loop_prev = wp.int32(-1)
        for k in range(n_path):
            h = net.scratch[n_path - 1 - k]
            s = new_segment(net, h, loop_prev)
            if s == -1:
                out_status[STATUS_SEGMENT_OVERFLOW] = 1
                return
            push_outside(net, h, s)
            if loop_prev != -1:
                net.segment_next[loop_prev] = s
            else:
                net.state[STATE_HEAD] = s
            queue_wedge(mesh, tri, net, s, out_status)
            loop_prev = s

        if not shorten_path(mesh, tri, net, out_status):
            if out_status[STATUS_HEAP_OVERFLOW] != 0 or out_status[STATUS_SEGMENT_OVERFLOW] != 0:
                return
            out_status[STATUS_UNCONVERGED] = out_status[STATUS_UNCONVERGED] + 1

        s = net.state[STATE_HEAD]
        while s != -1:
            h = net.segment_halfedge[s]
            kind = wp.where(tri.original[h // 2] != 0, ENTRY_ORIGINAL, ENTRY_TRACED)
            emit_entry(
                n_entries, p, kind, tri.vertex[h], tri.vertex[twin(h)],
                halfedge_vector(mesh, tri, h),
                out_path, out_kind, out_tail, out_tip, out_vector,
            )  # fmt: skip
            n_entries += 1
            s = net.segment_next[s]
        out_offsets[p + 1] = n_entries
        rewind(mesh, tri, net)
    out_status[STATUS_ENTRIES] = n_entries


# ---------------------------------------------------------------------------------------------
# Tracing intrinsic edges across the input mesh
# ---------------------------------------------------------------------------------------------


@wp.struct
class Step:
    """``TraceSubResult``: where one step of a trace ended, or the halfedge it crossed."""

    terminated: wp.bool
    cross_halfedge: wp.int32
    t_cross: wp.float64
    direction: wp.vec2d
    length: wp.float64
    end_kind: wp.int32
    end_element: wp.int32
    end_coords: wp.vec3d


@wp.func
def face_coordinates(mesh: InputMesh, f: wp.int32, k: wp.int32) -> wp.vec2d:
    # ``vertexCoordinatesInTriangle``: corner ``k`` of face ``f`` laid out from ``f.halfedge()``.
    h = mesh.face_halfedge[f]
    result = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    if k == 1:
        result = mesh.vector_in_face[h]
    elif k == 2:
        result = -mesh.vector_in_face[mesh.next[mesh.next[h]]]
    return result


@wp.func
def index_in_face(mesh: InputMesh, h: wp.int32) -> wp.int32:
    first = mesh.face_halfedge[mesh.face[h]]
    result = wp.int32(2)
    if h == first:
        result = 0
    elif h == mesh.next[first]:
        result = 1
    return result


@wp.func
def permute_from_canonical(b: wp.vec3d, k: wp.int32) -> wp.vec3d:
    return wp.vec3d(b[k % 3], b[(1 + k) % 3], b[(2 + k) % 3])


@wp.func
def permute_to_canonical(b: wp.vec3d, k: wp.int32) -> wp.vec3d:
    result = wp.vec3d()
    for i in range(3):
        result[(i + k) % 3] = b[i]
    return result


@wp.func
def cartesian_to_barycentric(mesh: InputMesh, f: wp.int32, v: wp.vec2d) -> wp.vec3d:
    # ``cartesianVectorToBarycentric``: the displacement with zero sum whose corners' combination
    # is ``v``; corner 0 sits at the origin.
    c1 = face_coordinates(mesh, f, 1)
    c2 = face_coordinates(mesh, f, 2)
    det = c1[0] * c2[1] - c2[0] * c1[1]
    b1 = (v[0] * c2[1] - c2[0] * v[1]) / det
    b2 = (c1[0] * v[1] - v[0] * c1[1]) / det
    b = wp.vec3d(-b1 - b2, b1, b2)
    mean = (b[0] + b[1] + b[2]) / wp.float64(3.0)
    return wp.vec3d(b[0] - mean, b[1] - mean, b[2] - mean)


@wp.func
def edge_t(h: wp.int32, t: wp.float64) -> wp.float64:
    # ``convertTToEdge``.
    return wp.where(h % 2 == 0, t, wp.float64(1.0) - t)


@wp.func
def trace_in_face(
    mesh: InputMesh,
    f: wp.int32,
    start: wp.vec3d,
    vector: wp.vec3d,
    direction: wp.vec2d,
    length: wp.float64,
    hittable: wp.vec3i,
) -> Step:
    # ``traceInFaceBarycentric``.
    result = Step()
    end = start + vector
    if end[0] >= wp.float64(0.0) and end[1] >= wp.float64(0.0) and end[2] >= wp.float64(0.0):
        result.terminated = True
        result.end_kind = POINT_FACE
        result.end_element = f
        result.end_coords = end
        result.length = wp.float64(0.0)
        return result

    t_ray = wp.float64(wp.inf)
    cross_halfedge = wp.int32(-1)
    opposite = wp.int32(-1)
    h = mesh.face_halfedge[f]
    for i in range(3):
        h = mesh.next[h]
        if hittable[(i + 1) % 3] == 0 or vector[i] >= wp.float64(0.0):
            continue
        t_raw = -start[i] / vector[i]
        if t_raw < t_ray:
            t_ray = t_raw
            cross_halfedge = h
            opposite = i
    t_ray = clamp_d(t_ray, wp.float64(0.0), wp.float64(1.0) - TRACE_EPS_LOOSE)
    if cross_halfedge == -1:
        result.terminated = True
        result.end_kind = POINT_FACE
        result.end_element = f
        result.end_coords = start
        result.length = wp.float64(0.0)
        return result

    on_edge = start + t_ray * vector
    a = on_edge[(opposite + 1) % 3]
    b = on_edge[(opposite + 2) % 3]
    t_cross = clamp_d(b / (a + b), wp.float64(0.0), wp.float64(1.0))
    remaining = (wp.float64(1.0) - t_ray) * length
    crossing = face_coordinates(mesh, f, (opposite + 2) % 3) - face_coordinates(
        mesh, f, (opposite + 1) % 3
    )
    result.direction = complex_div(direction, normalize2(crossing))
    result.length = remaining
    if mesh.face[twin(cross_halfedge)] == -1:
        result.terminated = True
        result.end_kind = POINT_EDGE
        result.end_element = cross_halfedge // 2
        result.end_coords = wp.vec3d(
            edge_t(cross_halfedge, t_cross), wp.float64(0.0), wp.float64(0.0)
        )
        return result
    result.terminated = False
    result.cross_halfedge = cross_halfedge
    result.t_cross = t_cross
    return result


@wp.func
def trace_from_edge_into_face(
    mesh: InputMesh,
    from_halfedge: wp.int32,
    t_from: wp.float64,
    direction: wp.vec2d,
    length: wp.float64,
) -> Step:
    # ``traceInFaceFromEdge``: enter the face across ``from_halfedge``.
    face_halfedge = twin(from_halfedge)
    f = mesh.face[face_halfedge]
    d = -direction
    d = wp.vec2d(d[0], wp.max(d[1], TRACE_EPS_LOOSE))
    in_face = complex_mul(normalize2(mesh.vector_in_face[face_halfedge]), d)
    k = index_in_face(mesh, face_halfedge)
    b = permute_from_canonical(cartesian_to_barycentric(mesh, f, in_face), k)
    b = wp.vec3d(b[0], b[1], wp.max(b[2], TRACE_EPS_TIGHT))
    diff = -(b[0] + b[1] + b[2])
    if diff > wp.float64(0.0):
        b = wp.vec3d(b[0], b[1], b[2] + diff)
    else:
        third = diff / wp.float64(3.0)
        b = wp.vec3d(b[0] + third, b[1] + third, b[2] + third)
    t = clamp_d(t_from, wp.float64(0.0), wp.float64(1.0))
    start = wp.vec3d()
    start[k] = t
    start[(k + 1) % 3] = wp.float64(1.0) - t
    hittable = wp.vec3i(1, 1, 1)
    hittable[k] = 0
    return trace_in_face(
        mesh, f, start, permute_to_canonical(b, k) * length, in_face, length, hittable
    )


@wp.func
def trace_from_edge(
    mesh: InputMesh, e: wp.int32, t_edge: wp.float64, direction: wp.vec2d, length: wp.float64
) -> Step:
    # ``traceGeodesic_fromEdge``.
    t = clamp_d(t_edge, wp.float64(0.0), wp.float64(1.0))
    result = Step()
    if direction[1] >= wp.float64(0.0):
        result = trace_from_edge_into_face(mesh, 2 * e + 1, wp.float64(1.0) - t, -direction, length)
    elif mesh.face[2 * e + 1] == -1:
        result.terminated = True
        result.end_kind = POINT_EDGE
        result.end_element = e
        result.end_coords = wp.vec3d(t, wp.float64(0.0), wp.float64(0.0))
    else:
        result = trace_from_edge_into_face(mesh, 2 * e, t, direction, length)
    return result


@wp.func
def trace_from_vertex(
    mesh: InputMesh, v: wp.int32, direction: wp.vec2d, length: wp.float64
) -> Step:
    # ``traceGeodesic_fromVertex``: find the wedge the direction falls in (or the nearest
    # halfedge, when rounding puts it in none), undo the angle rescaling, and step into the face.
    first = mesh.vertex_halfedge[v]
    wedge = wp.int32(-1)
    relative = wp.vec2d()
    min_cross = wp.float64(wp.inf)
    min_halfedge = wp.int32(-1)
    min_direction = wp.vec2d()
    h = first
    while True:
        if mesh.face[h] == -1:
            break
        h_next = twin(mesh.next[mesh.next[h]])
        interval_start = normalize2(mesh.vector_in_vertex[h])
        interval_end = normalize2(mesh.vector_in_vertex[h_next])
        cross_start = cross2(interval_start, direction)
        cross_end = cross2(interval_end, direction)
        if cross_start > wp.float64(0.0) and cross_end <= wp.float64(0.0):
            wedge = h
            relative = complex_div(direction, interval_start)
            break
        if wp.abs(cross_start) < min_cross:
            min_cross = wp.abs(cross_start)
            min_halfedge = h
            min_direction = wp.vec2d(wp.float64(1.0), TRACE_EPS_TIGHT)
        if wp.abs(cross_end) < min_cross:
            min_cross = wp.abs(cross_end)
            min_halfedge = h_next
            min_direction = wp.vec2d(wp.float64(1.0), -TRACE_EPS_TIGHT)
        h = h_next
        if h == first:
            break
    if wedge == -1:
        edge_direction = wp.where(min_halfedge % 2 == 0, min_direction, -min_direction)
        return trace_from_edge(
            mesh, min_halfedge // 2, edge_t(min_halfedge, wp.float64(0.0)), edge_direction, length
        )
    relative = normalize2(complex_pow(relative, mesh.angle_sum[v] / turn_of(mesh, v)))
    in_face = complex_mul(relative, normalize2(mesh.vector_in_face[wedge]))

    # ``traceInFaceTowardsEdge`` from ``wedge``'s tail towards the opposite edge.
    towards = mesh.next[wedge]
    f = mesh.face[towards]
    k = index_in_face(mesh, wedge)
    b = permute_from_canonical(cartesian_to_barycentric(mesh, f, in_face) * length, k)
    b = wp.vec3d(
        wp.min(b[0], TRACE_EPS_TIGHT), wp.max(b[1], wp.float64(0.0)), wp.max(b[2], wp.float64(0.0))
    )
    diff = -(b[0] + b[1] + b[2])
    if diff > wp.float64(0.0):
        half = diff / wp.float64(2.0)
        b = wp.vec3d(b[0], b[1] + half, b[2] + half)
    else:
        b = wp.vec3d(b[0] + diff, b[1], b[2])
    start = wp.vec3d()
    start[k] = wp.float64(1.0)
    hittable = wp.vec3i(0, 0, 0)
    hittable[(k + 1) % 3] = 1
    return trace_in_face(mesh, f, start, permute_to_canonical(b, k), in_face, length, hittable)


@wp.func
def point_position(mesh: InputMesh, kind: wp.int32, element: wp.int32, coords: wp.vec3d) -> wp.vec3:
    # ``SurfacePoint::interpolate`` of the vertex positions.
    position = wp.vec3d()
    if kind == POINT_VERTEX:
        position = mesh.position[element]
    elif kind == POINT_EDGE:
        h = 2 * element
        t = coords[0]
        position = (wp.float64(1.0) - t) * mesh.position[mesh.vertex[h]] + t * mesh.position[
            mesh.vertex[h + 1]
        ]
    else:
        h = mesh.face_halfedge[element]
        h1 = mesh.next[h]
        position = (
            coords[0] * mesh.position[mesh.vertex[h]]
            + coords[1] * mesh.position[mesh.vertex[h1]]
            + coords[2] * mesh.position[mesh.vertex[mesh.next[h1]]]
        )
    return wp.vec3(position)


@wp.func
def write_point(
    mesh: InputMesh,
    index: wp.int32,
    limit: wp.int32,
    top: wp.int32,
    kind: wp.int32,
    element: wp.int32,
    coords: wp.vec3d,
    staged_points: wp.array[wp.vec3],
    staged_next: wp.array[wp.int32],
    chunk: wp.ref[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # Point ``index`` of a trace lands at ``top - index``: the paths come out backwards. Past
    # ``limit`` (all of it on the counting pass) it is staged from ``chunk`` (``array.stage_point``)
    # for ``write_traced_points`` to copy.
    if index < limit:
        out_points[top - index] = point_position(mesh, kind, element, coords)
    elif chunk >= 0:
        stage_point(
            staged_points, staged_next, chunk, index, point_position(mesh, kind, element, coords)
        )


@wp.func
def edge_touches(mesh: InputMesh, e: wp.int32, v: wp.int32) -> wp.bool:
    return mesh.vertex[2 * e] == v or mesh.vertex[2 * e + 1] == v


@wp.func
def trace_points(
    mesh: InputMesh,
    tail: wp.int32,
    target: wp.int32,
    vector: wp.vec2d,
    max_iterations: wp.int32,
    limit: wp.int32,
    top: wp.int32,
    staged_points: wp.array[wp.vec3],
    staged_next: wp.array[wp.int32],
    first_chunk: wp.int32,
    out_points: wp.array[wp.vec3],
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # ``traceGeodesic`` from ``tail`` with ``includePath``, writing its first ``limit`` points and
    # staging the rest from ``first_chunk`` (``write_point``).
    # Returns the point count, the last point ``trimTraceResult`` keeps (the last edge crossing not
    # on an edge of ``target``, else the start) and that crossing's edge, ``-1`` for the start.
    chunk = wp.int32(first_chunk)
    length = wp.length(vector)
    direction = vector / length
    write_point(
        mesh, 0, limit, top, POINT_VERTEX, tail, wp.vec3d(), staged_points, staged_next, chunk,
        out_points,
    )  # fmt: skip
    n = wp.int32(1)
    kept = wp.int32(0)
    kept_edge = wp.int32(-1)
    step = trace_from_vertex(mesh, tail, direction, length)
    iteration = wp.int32(0)
    while not step.terminated:
        e = step.cross_halfedge // 2
        t = edge_t(step.cross_halfedge, step.t_cross)
        write_point(
            mesh, n, limit, top, POINT_EDGE, e, wp.vec3d(t, wp.float64(0.0), wp.float64(0.0)),
            staged_points, staged_next, chunk, out_points,
        )  # fmt: skip
        if not edge_touches(mesh, e, target):
            kept = n
            kept_edge = e
        n += 1
        if iteration >= max_iterations:
            return n, kept, kept_edge
        step = trace_from_edge_into_face(
            mesh, step.cross_halfedge, step.t_cross, step.direction, step.length
        )
        iteration += 1
    write_point(
        mesh, n, limit, top, step.end_kind, step.end_element, step.end_coords, staged_points,
        staged_next, chunk, out_points,
    )  # fmt: skip
    if step.end_kind == POINT_EDGE and not edge_touches(mesh, step.end_element, target):
        kept = n
        kept_edge = step.end_element
    n += 1
    return n, kept, kept_edge


@wp.func
def trim_succeeds(
    mesh: InputMesh, tail: wp.int32, kept_edge: wp.int32, target: wp.int32
) -> wp.bool:
    # ``trimTraceResult``'s success test on the last kept point: the start vertex, or an edge
    # crossing, must touch ``target``'s one-ring.
    found = wp.bool(False)
    if kept_edge == -1:
        if tail == target:
            found = True
        else:
            first = mesh.vertex_halfedge[tail]
            h = first
            while True:
                if mesh.vertex[twin(h)] == target:
                    found = True
                    break
                h = mesh.next[twin(h)]
                if h == first:
                    break
    else:
        h = 2 * kept_edge
        found = (
            mesh.vertex[h] == target
            or mesh.vertex[h + 1] == target
            or mesh.vertex[mesh.next[mesh.next[h]]] == target
            or mesh.vertex[mesh.next[mesh.next[h + 1]]] == target
        )
    return found


@wp.kernel(enable_backward=False)
def count_traced_points(
    mesh: InputMesh,
    offsets: wp.array[wp.int32],
    path: wp.array[wp.int32],
    kind: wp.array[wp.int32],
    tail: wp.array[wp.int32],
    tip: wp.array[wp.int32],
    vector: wp.array[wp.vec2d],
    max_iterations: wp.int32,
    staged_points: wp.array[wp.vec3],
    staged_next: wp.array[wp.int32],
    no_points: wp.array[wp.vec3],
    out_limits: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Each entry's points: a traced edge trimmed to its far vertex when the trim succeeds, else
    # the whole trace. An entry that ends on a vertex drops it unless it ends the path, since the
    # next entry starts there (``getPathPolyline``'s repeated-point test).
    i = wp.int32(wp.tid())
    last = i + 1 == offsets[path[i] + 1]
    end = wp.where(last, 1, 0)
    k = kind[i]
    if k == ENTRY_POINT:
        out_limits[i] = 1
        out_counts[i] = 1
    elif k == ENTRY_ORIGINAL:
        out_limits[i] = 1
        out_counts[i] = 1 + end
    else:
        n, kept, kept_edge = trace_points(
            mesh,
            tail[i],
            tip[i],
            vector[i],
            max_iterations,
            0,
            0,
            staged_points,
            staged_next,
            first_staged_chunk(staged_next, i),
            no_points,
        )
        if trim_succeeds(mesh, tail[i], kept_edge, tip[i]):
            out_limits[i] = kept + 1
            out_counts[i] = kept + 1 + end
        else:
            out_limits[i] = -n
            out_counts[i] = n


@wp.kernel(enable_backward=False)
def write_traced_points(
    mesh: InputMesh,
    offsets: wp.array[wp.int32],
    path: wp.array[wp.int32],
    kind: wp.array[wp.int32],
    tail: wp.array[wp.int32],
    tip: wp.array[wp.int32],
    vector: wp.array[wp.vec2d],
    max_iterations: wp.int32,
    staged_points: wp.array[wp.vec3],
    staged_next: wp.array[wp.int32],
    limits: wp.array[wp.int32],
    point_offsets: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # The points ``count_traced_points`` counted, each path's written back to front so that it
    # runs from its target to the source: copied from what that pass staged, or traced again where
    # the pool did not hold them.
    i = wp.int32(wp.tid())
    p = path[i]
    first = point_offsets[offsets[p]]
    last = point_offsets[offsets[p + 1]] - 1
    top = first + last - point_offsets[i]
    k = kind[i]
    limit = limits[i]
    if k == ENTRY_TRACED:
        n_write = wp.abs(limit)
        if not copy_staged(staged_points, staged_next, i, out_points, top, -1, n_write):
            trace_points(
                mesh,
                tail[i],
                tip[i],
                vector[i],
                max_iterations,
                n_write,
                top,
                staged_points,
                staged_next,
                wp.int32(-1),
                out_points,
            )
    else:
        out_points[top] = point_position(mesh, POINT_VERTEX, tail[i], wp.vec3d())
    if limit > 0 and point_offsets[i + 1] - point_offsets[i] > limit:
        out_points[top - limit] = point_position(mesh, POINT_VERTEX, tip[i], wp.vec3d())
