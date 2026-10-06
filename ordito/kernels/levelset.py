import warp as wp
from warp.geometry import IsoSurfaceMarchingCubes

from ordito.constants import FLOAT32_INF_CONSTANT, TOLERANCE_MERGE_CONSTANT
from ordito.kernels.array import ravel_index

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------------------------
# Signed-distance level sets on a corner lattice, signed by the generalized winding number.
#
# ``signed_distance_level_set``'s winding path reproduces, bit for bit, the surface the dense
# lattice of ``proximity.signed_distance_on_mesh(sign_mode="winding")`` values extracts, without
# evaluating that field everywhere:
#
# * **Distance in a band.** Marching cubes reads a node's value only where an edge from it crosses
#   ``iso``, and both ends of such an edge lie within one cell diagonal of the level; everywhere
#   else only the node's side of ``iso`` matters. So the closest-point search is capped at
#   ``|iso|`` plus a cell diagonal and a miss is stored as that cap, on the node's own side.
#   Under the cap the search visits a subset of the uncapped search's nodes in the same order, and
#   its strict ``<`` keeps the first of equal distances, so a hit is the uncapped answer bit for
#   bit. A field signed by the winding number of an open surface can change sign away from the
#   surface, so a capped node can still end a crossing edge; ``resolve_crossing_endpoints``
#   re-queries exactly those uncapped.
# * **Sign from the lattice.** For a closed 2-chain the winding number is an integer: the signed
#   count of crossings of a ray. An open mesh M is closed by the cone K over its boundary 1-chain
#   (from one boundary vertex per boundary component), and w(M) = w(M - K) + w(K): the first term
#   counts crossings along each lattice column, rasterising every triangle of M and of K onto the
#   columns, the second sums the solid angles of the few cone triangles. That is the exact winding
#   number, where Warp's own (``mesh_query_point_sign_winding_number``) is a Barnes-Hut
#   approximation of it, so the two agree except where the exact value is close to 1/2 or the
#   lattice's own evaluation is in doubt (a crossing within rounding of a node, a node on a cone
#   triangle, the two column directions disagreeing, a node within rounding of the surface). Those
#   nodes are *undecided*: they are summed exactly with Warp's per-triangle solid angle, and a node
#   still within ``delta`` of 1/2 takes Warp's own sign.
#
# Measured (CUDA, every node of lattices at 64-256 cells across the scan meshes and the test
# fixtures): Warp's approximation stays within 1e-3 of the exact winding number on an open
# hemisphere and within 0.02 on the scan meshes, so the 0.1 margin is not what the agreement rests
# on. Without the on-cone test a hemisphere opening along y (its cone edge-on to both column
# directions) diverges; a flag for triangles whose projection is a sliver was built and never
# changed a surface over 324 fixture and tilted-box configurations once the two column directions
# are compared, so it was dropped.
# ---------------------------------------------------------------------------------------------

# A rasterised crossing within this many cells (along its column) of a node leaves that node's
# integer winding in doubt.
AMBIGUOUS_CROSSING = wp.constant(wp.float32(1.0e-3))
# Cone components seen from farther than this many bounding radii are bounded rather than summed.
CONE_NEAR_RADII = wp.constant(wp.float32(4.0))
INV_FOUR_PI = wp.constant(wp.float32(0.07957747154594767))


@wp.func
def halfedge_endpoints(faces: wp.array[wp.int32], h: wp.int32) -> wp.vec2i:
    f = h // 3
    corner = h - 3 * f
    return wp.vec2i(faces[h], faces[3 * f + (corner + 1) % 3])


@wp.func
def halfedge_orientation(faces: wp.array[wp.int32], h: wp.int32) -> wp.int32:
    # +1 for a halfedge running from its lower vertex index to its higher, -1 the other way, 0 for
    # a degenerate one: its cone triangle would be degenerate too and contributes nothing.
    ends = halfedge_endpoints(faces, h)
    orientation = wp.int32(0)
    if ends[0] < ends[1]:
        orientation = 1
    elif ends[0] > ends[1]:
        orientation = -1
    return orientation


@wp.kernel
def boundary_chain_multiplicity(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], out_net: wp.array[wp.int32]
) -> None:
    # The boundary 1-chain of the faces: per undirected edge, the net number of times its halfedges
    # run from its lower vertex to its higher, stored at one representative halfedge (the edge's own
    # halfedge when alone, the lower of a pair, the lowest of a run of three or more, which
    # ``halfedge.halfedge_mates`` encodes as ``-2 - lowest``). ``out_net`` arrives zeroed.
    h = wp.int32(wp.tid())
    mate = mates[h]
    if mate == -1:
        out_net[h] = halfedge_orientation(faces, h)
    elif mate >= 0:
        if h < mate:
            out_net[h] = halfedge_orientation(faces, h) + halfedge_orientation(faces, mate)
    else:
        wp.atomic_add(out_net, -2 - mate, halfedge_orientation(faces, h))


@wp.kernel
def boundary_chain_counts(net: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    h = wp.int32(wp.tid())
    out_counts[h] = wp.abs(net[h])


@wp.kernel
def emit_boundary_chain(
    faces: wp.array[wp.int32],
    net: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_chain: wp.array2d[wp.int32],
) -> None:
    # Each boundary edge as many times as its multiplicity, directed the way its net runs.
    h = wp.int32(wp.tid())
    multiplicity = net[h]
    if multiplicity == 0:
        return
    ends = halfedge_endpoints(faces, h)
    low = wp.min(ends[0], ends[1])
    high = wp.max(ends[0], ends[1])
    start = wp.where(multiplicity > 0, low, high)
    end = wp.where(multiplicity > 0, high, low)
    base = offsets[h]
    for t in range(wp.abs(multiplicity)):
        out_chain[base + t, 0] = start
        out_chain[base + t, 1] = end


@wp.kernel
def cone_component_keys(
    chain: wp.array2d[wp.int32], labels: wp.array[wp.int32], out_keys: wp.array[wp.int32]
) -> None:
    # A chain edge's boundary component, named by its smallest vertex: the cone's apex.
    e = wp.int32(wp.tid())
    out_keys[e] = labels[chain[e, 0]]


@wp.kernel
def mark_run_starts(sorted_keys: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    t = wp.int32(wp.tid())
    out_flags[t] = wp.where(t == 0 or sorted_keys[t] != sorted_keys[t - 1], 1, 0)


@wp.kernel
def cone_bounds(
    vertices: wp.array[wp.vec3],
    chain: wp.array2d[wp.int32],
    order: wp.array[wp.int32],
    sorted_keys: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    out_radius: wp.array[wp.float32],
) -> None:
    # A ball around each component's apex holding its whole cone (the convex hull of the apex and
    # the component's boundary vertices), for the far-field bound in ``cone_winding``. Inflated by
    # a rounding margin so the bound stays one.
    c = wp.int32(wp.tid())
    apex = vertices[sorted_keys[starts[c]]]
    radius = wp.float32(0.0)
    for t in range(starts[c], starts[c + 1]):
        e = order[t]
        radius = wp.max(radius, wp.length(vertices[chain[e, 0]] - apex))
        radius = wp.max(radius, wp.length(vertices[chain[e, 1]] - apex))
    out_radius[c] = radius * 1.001 + 1.0e-30


@wp.func
def lattice_coordinates(p: wp.vec3, lower: wp.vec3, inv_step: wp.vec3, axis: wp.int32) -> wp.vec3:
    # A world point in lattice index units, permuted so the column direction is the last
    # component: axis 0 runs columns along z, axis 1 along x.
    q = wp.cw_mul(p - lower, inv_step)
    return wp.where(axis == 0, q, wp.vec3(q[1], q[2], q[0]))


@wp.func
def lattice_node(u: wp.int32, v: wp.int32, w: wp.int32, axis: wp.int32) -> wp.vec3i:
    # The natural ``(i, j, k)`` of the node at column ``(u, v)``, height ``w``.
    return wp.where(axis == 0, wp.vec3i(u, v, w), wp.vec3i(w, u, v))


@wp.func
def precedes(a: wp.vec3, b: wp.vec3, ia: wp.int32, ib: wp.int32) -> wp.bool:
    # A canonical endpoint order by position, the index only breaking an exact tie, so the two faces
    # sharing an edge -- and coincident duplicate vertices -- evaluate it identically.
    result = ia < ib
    if a[2] != b[2]:
        result = a[2] < b[2]
    if a[1] != b[1]:
        result = a[1] < b[1]
    if a[0] != b[0]:
        result = a[0] < b[0]
    return result


@wp.func
def edge_function(low: wp.vec2, high: wp.vec2, x: wp.float32, y: wp.float32) -> wp.float32:
    # Side of the column ``(x, y)`` relative to the canonical edge ``low -> high``. A column exactly
    # on the edge is pushed along one fixed generic direction (simulation of simplicity), so every
    # face sharing the edge decides it the same way and a column through an edge or a vertex
    # crosses the closed chain exactly once per sheet.
    dx = high[0] - low[0]
    dy = high[1] - low[1]
    side = dx * (y - low[1]) - dy * (x - low[0])
    if side == 0.0:
        side = dx * 0.7548776662466927 - dy * 0.5698402909980532
        if side == 0.0:
            side = wp.where(dx > 0.0 or (dx == 0.0 and dy > 0.0), 1.0, -1.0)
    return side


@wp.func
def directed_edge_function(
    ia: wp.int32,
    ib: wp.int32,
    pa: wp.vec3,
    pb: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    x: wp.float32,
    y: wp.float32,
) -> wp.float32:
    a2 = wp.vec2(a[0], a[1])
    b2 = wp.vec2(b[0], b[1])
    return wp.where(
        precedes(pa, pb, ia, ib), edge_function(a2, b2, x, y), -edge_function(b2, a2, x, y)
    )


@wp.func
def rasterize_crossings(
    ia: wp.int32,
    ib: wp.int32,
    ic: wp.int32,
    pa: wp.vec3,
    pb: wp.vec3,
    pc: wp.vec3,
    weight: wp.int32,
    lower: wp.vec3,
    inv_step: wp.vec3,
    axis: wp.int32,
    crossings: wp.array3d[wp.int32],
    ambiguous: wp.array3d[wp.int32],
) -> None:
    # Every column through the triangle's projection gets its signed crossing at the highest node
    # below it, so a suffix sum along the column (``column_suffix_sums``) is, at each node, the
    # signed count of crossings above it. The sign is the projected orientation, so a closed,
    # outward-wound surface counts 1 inside and 0 outside.
    n_u = wp.where(axis == 0, crossings.shape[0], crossings.shape[1])
    n_v = wp.where(axis == 0, crossings.shape[1], crossings.shape[2])
    n_w = wp.where(axis == 0, crossings.shape[2], crossings.shape[0])
    a = lattice_coordinates(pa, lower, inv_step, axis)
    b = lattice_coordinates(pb, lower, inv_step, axis)
    c = lattice_coordinates(pc, lower, inv_step, axis)
    u0 = wp.max(wp.int32(wp.ceil(wp.min(a[0], wp.min(b[0], c[0])))), 0)
    u1 = wp.min(wp.int32(wp.floor(wp.max(a[0], wp.max(b[0], c[0])))), n_u - 1)
    v0 = wp.max(wp.int32(wp.ceil(wp.min(a[1], wp.min(b[1], c[1])))), 0)
    v1 = wp.min(wp.int32(wp.floor(wp.max(a[1], wp.max(b[1], c[1])))), n_v - 1)
    if u0 > u1 or v0 > v1:
        return
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if area == 0.0:
        return
    orientation = wp.where(area > 0.0, 1, -1)
    w_low = wp.min(a[2], wp.min(b[2], c[2]))
    w_high = wp.max(a[2], wp.max(b[2], c[2]))
    for u in range(u0, u1 + 1):
        for v in range(v0, v1 + 1):
            x = wp.float32(u)
            y = wp.float32(v)
            side_ab = directed_edge_function(ia, ib, pa, pb, a, b, x, y)
            side_bc = directed_edge_function(ib, ic, pb, pc, b, c, x, y)
            side_ca = directed_edge_function(ic, ia, pc, pa, c, a, x, y)
            inside = wp.where(
                orientation > 0,
                side_ab > 0.0 and side_bc > 0.0 and side_ca > 0.0,
                side_ab < 0.0 and side_bc < 0.0 and side_ca < 0.0,
            )
            if inside:
                # Clamped to the triangle's own height range: a projection that is a sliver
                # interpolates badly, and the column it then miscounts disagrees with the other
                # direction's count, which leaves its nodes undecided.
                total = side_ab + side_bc + side_ca
                height = wp.clamp(
                    (side_bc * a[2] + side_ca * b[2] + side_ab * c[2]) / total, w_low, w_high
                )
                nearest = wp.int32(wp.round(height))
                if (
                    wp.abs(height - wp.float32(nearest)) < AMBIGUOUS_CROSSING
                    and nearest >= 0
                    and nearest < n_w
                ):
                    node = lattice_node(u, v, nearest, axis)
                    ambiguous[node[0], node[1], node[2]] = 1
                below = wp.min(wp.int32(wp.ceil(height)), n_w) - 1
                if below >= 0:
                    node = lattice_node(u, v, below, axis)
                    wp.atomic_add(crossings, node[0], node[1], node[2], orientation * weight)


@wp.kernel
def rasterize_face_crossings(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lower: wp.vec3,
    inv_step: wp.vec3,
    axis: wp.int32,
    out_crossings: wp.array3d[wp.int32],
    out_ambiguous: wp.array3d[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    ia = faces[3 * f]
    ib = faces[3 * f + 1]
    ic = faces[3 * f + 2]
    rasterize_crossings(
        ia,
        ib,
        ic,
        vertices[ia],
        vertices[ib],
        vertices[ic],
        1,
        lower,
        inv_step,
        axis,
        out_crossings,
        out_ambiguous,
    )


@wp.kernel
def rasterize_cone_crossings(
    vertices: wp.array[wp.vec3],
    chain: wp.array2d[wp.int32],
    labels: wp.array[wp.int32],
    lower: wp.vec3,
    inv_step: wp.vec3,
    axis: wp.int32,
    out_crossings: wp.array3d[wp.int32],
    out_ambiguous: wp.array3d[wp.int32],
) -> None:
    # The cone K, subtracted: triangle ``(apex, a, b)`` for each boundary edge ``a -> b``, the apex
    # being the component's smallest vertex (a real vertex, so its index orders it canonically).
    e = wp.int32(wp.tid())
    a = chain[e, 0]
    b = chain[e, 1]
    apex = labels[a]
    rasterize_crossings(
        apex,
        a,
        b,
        vertices[apex],
        vertices[a],
        vertices[b],
        -1,
        lower,
        inv_step,
        axis,
        out_crossings,
        out_ambiguous,
    )


@wp.kernel
def column_suffix_sums(axis: wp.int32, crossings: wp.array3d[wp.int32]) -> None:
    # In place: each node's signed count of the crossings above it along its column.
    u, v = wp.tid()
    n_w = wp.where(axis == 0, crossings.shape[2], crossings.shape[0])
    total = wp.int32(0)
    for t in range(n_w):
        w = n_w - 1 - t
        node = lattice_node(u, v, w, axis)
        total = total + crossings[node[0], node[1], node[2]]
        crossings[node[0], node[1], node[2]] = total


@wp.func
def warp_solid_angle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.float32:
    # Warp's ``robust_solid_angle`` (``native/solid_angle.h``), operation for operation, so that a
    # query lying in a face's plane gets the zero Warp's own winding evaluation gives it -- which is
    # what decides the sign of a node within rounding of the surface. Not
    # ``kernels/proximity.solid_angle``: that is the closed-form Van Oosterom-Strackee formula,
    # equal to rounding but not at that tie.
    qa = a - p
    qb = b - p
    qc = c - p
    length_a = wp.length(qa)
    length_b = wp.length(qb)
    length_c = wp.length(qc)
    angle = wp.float32(0.0)
    if length_a != 0.0 and length_b != 0.0 and length_c != 0.0:
        qa = qa / length_a
        qb = qb / length_b
        qc = qc / length_c
        numerator = wp.dot(qa, wp.cross(qb - qa, qc - qa))
        if numerator != 0.0:
            denominator = 1.0 + wp.dot(qa, qb) + wp.dot(qa, qc) + wp.dot(qb, qc)
            angle = 2.0 * wp.atan2(numerator, denominator)
    return angle


@wp.func
def on_cone_triangle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.bool:
    # ``p`` within rounding of the cone triangle itself, where its solid angle jumps by 4 pi and the
    # rasterised crossing (absent altogether when the triangle is edge-on to the columns) cannot
    # be trusted to agree with it.
    qa = a - p
    qb = b - p
    qc = c - p
    length_a = wp.length(qa)
    length_b = wp.length(qb)
    length_c = wp.length(qc)
    numerator = wp.dot(qa, wp.cross(qb, qc))
    denominator = (
        length_a * length_b * length_c
        + wp.dot(qa, qb) * length_c
        + wp.dot(qb, qc) * length_a
        + wp.dot(qc, qa) * length_b
    )
    return wp.abs(numerator) <= 1.0e-4 * length_a * length_b * length_c and denominator <= 0.0


@wp.func
def ball_winding_bound(distance: wp.float32, radius: wp.float32) -> wp.float32:
    # The largest |winding number| any surface inside a ball can have, seen from ``distance`` away
    # from its centre: the ball's solid angle over 4 pi.
    ratio = wp.min(radius / distance, 1.0)
    return 0.5 * (1.0 - wp.sqrt(1.0 - ratio * ratio))


@wp.func
def cone_component_winding(
    vertices: wp.array[wp.vec3],
    chain: wp.array2d[wp.int32],
    order: wp.array[wp.int32],
    apex: wp.vec3,
    first: wp.int32,
    last: wp.int32,
    p: wp.vec3,
) -> wp.vec2:
    # One component's cone: (its winding number at ``p``, 1 if ``p`` lies on it).
    angle = wp.float32(0.0)
    on_cone = wp.float32(0.0)
    for t in range(first, last):
        e = order[t]
        a = vertices[chain[e, 0]]
        b = vertices[chain[e, 1]]
        angle = angle + warp_solid_angle(apex, a, b, p)
        if on_cone_triangle(apex, a, b, p):
            on_cone = 1.0
    return wp.vec2(angle * INV_FOUR_PI, on_cone)


@wp.kernel
def lattice_winding_numbers(
    points: wp.array3d[wp.vec3],
    crossings_z: wp.array3d[wp.int32],
    crossings_x: wp.array3d[wp.int32],
    ambiguous: wp.array3d[wp.int32],
    distance: wp.array3d[wp.float32],
    vertices: wp.array[wp.vec3],
    chain: wp.array2d[wp.int32],
    order: wp.array[wp.int32],
    sorted_keys: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    radius: wp.array[wp.float32],
    delta: wp.float32,
    sign_irrelevant_below: wp.float32,
    surface_tie: wp.float32,
    out_winding: wp.array3d[wp.float32],
    out_slots: wp.array3d[wp.int32],
    out_undecided: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # The node's winding number: the column count plus the cones' solid angle, every cone farther
    # than ``CONE_NEAR_RADII`` of its radius bounded instead of summed unless the bound leaves the
    # sign (with its ``delta`` margin) open. A node whose sign matters -- it may end a crossing edge
    # or sit on the other side of ``iso`` under either sign (``distance >= sign_irrelevant_below``)
    # -- and whose lattice value is in doubt gets a slot in ``out_undecided`` (``slot + 1`` in
    # ``out_slots``; ``out_count[0]`` counts them all, past the buffer too).
    i, j, k = wp.tid()
    p = points[i, j, k]
    base = wp.float32(crossings_z[i, j, k])
    near = wp.float32(0.0)
    far_bound = wp.float32(0.0)
    on_cone = wp.float32(0.0)
    n_components = starts.shape[0] - 1
    for component in range(n_components):
        apex = vertices[sorted_keys[starts[component]]]
        reach = wp.length(p - apex)
        if reach > CONE_NEAR_RADII * radius[component]:
            far_bound = far_bound + ball_winding_bound(reach, radius[component])
        else:
            term = cone_component_winding(
                vertices, chain, order, apex, starts[component], starts[component + 1], p
            )
            near = near + term[0]
            on_cone = wp.max(on_cone, term[1])
    winding = base + near
    if wp.abs(winding - 0.5) <= far_bound + delta:
        for component in range(n_components):
            apex = vertices[sorted_keys[starts[component]]]
            if wp.length(p - apex) > CONE_NEAR_RADII * radius[component]:
                term = cone_component_winding(
                    vertices, chain, order, apex, starts[component], starts[component + 1], p
                )
                winding = winding + term[0]
                on_cone = wp.max(on_cone, term[1])
    out_winding[i, j, k] = winding
    d = distance[i, j, k]
    doubtful = (
        wp.abs(winding - 0.5) < delta
        or ambiguous[i, j, k] != 0
        or on_cone != 0.0
        or crossings_z[i, j, k] != crossings_x[i, j, k]
        or d <= surface_tie
    )
    slot = wp.int32(0)
    if doubtful and d >= sign_irrelevant_below and d > TOLERANCE_MERGE_CONSTANT:
        slot = wp.atomic_add(out_count, 0, 1) + 1
        if slot <= out_undecided.shape[0]:
            out_undecided[slot - 1] = p
    out_slots[i, j, k] = slot


@wp.kernel
def exact_winding_slices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    n_slices: wp.int32,
    out_winding: wp.array[wp.float32],
) -> None:
    # The exact winding number of the undecided points, every face summed with Warp's own solid
    # angle; one thread per (point, face slice), committed into the zeroed ``out_winding``.
    q, s = wp.tid()
    p = points[q]
    n_faces = faces.shape[0] // 3
    angle = wp.float32(0.0)
    for f in range(s, n_faces, n_slices):
        angle = angle + warp_solid_angle(
            vertices[faces[3 * f]], vertices[faces[3 * f + 1]], vertices[faces[3 * f + 2]], p
        )
    wp.atomic_add(out_winding, q, angle * INV_FOUR_PI)


@wp.kernel
def take_exact_winding(
    exact: wp.array[wp.float32],
    delta: wp.float32,
    winding: wp.array3d[wp.float32],
    slots: wp.array3d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # In place: an undecided node takes its exact winding number and is settled, unless that too is
    # within ``delta`` of 1/2, where only Warp's own approximation reproduces Warp's sign; those
    # keep their slot and are counted.
    i, j, k = wp.tid()
    slot = slots[i, j, k]
    if slot == 0:
        return
    value = exact[slot - 1]
    if wp.abs(value - 0.5) < delta:
        wp.atomic_add(out_count, 0, 1)
    else:
        winding[i, j, k] = value
        slots[i, j, k] = 0


@wp.kernel
def warp_winding_sign(
    mesh: wp.uint64,
    points: wp.array3d[wp.vec3],
    slots: wp.array3d[wp.int32],
    winding: wp.array3d[wp.float32],
) -> None:
    # In place: Warp's own sign at the nodes still in doubt, from the very query
    # ``proximity.signed_distance_on_mesh(sign_mode="winding")`` makes (accuracy 2, threshold 1/2,
    # unbounded), written as winding 1 inside and 0 outside.
    i, j, k = wp.tid()
    if slots[i, j, k] == 0:
        return
    query = wp.mesh_query_point_sign_winding_number(
        mesh, points[i, j, k], FLOAT32_INF_CONSTANT, 2.0, 0.5
    )
    winding[i, j, k] = wp.where(query.sign < 0.0, 1.0, 0.0)


@wp.kernel
def capped_distance(
    mesh: wp.uint64,
    points: wp.array3d[wp.vec3],
    cap: wp.float32,
    out_distance: wp.array3d[wp.float32],
) -> None:
    # Unsigned distance to the closest point, or ``cap`` when none is closer: under the cap the
    # search returns the uncapped search's face and coordinates bit for bit.
    i, j, k = wp.tid()
    p = points[i, j, k]
    query = wp.mesh_query_point_no_sign(mesh, p, cap)
    d = cap
    if query.result:
        d = wp.length(p - wp.mesh_eval_position(mesh, query.face, query.u, query.v))
    out_distance[i, j, k] = d


@wp.func
def signed_value(d: wp.float32, winding: wp.float32) -> wp.float32:
    # ``proximity.signed_distance_from_query``'s tail: positive within the merge tolerance of the
    # surface, else negative inside (winding above 1/2).
    return wp.where(d <= TOLERANCE_MERGE_CONSTANT, d, wp.where(winding > 0.5, -d, d))


@wp.kernel
def signed_band_values(
    distance: wp.array3d[wp.float32],
    winding: wp.array3d[wp.float32],
    out_values: wp.array3d[wp.float32],
) -> None:
    i, j, k = wp.tid()
    out_values[i, j, k] = signed_value(distance[i, j, k], winding[i, j, k])


@wp.func
def crosses(
    values: wp.array3d[wp.float32],
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    above: wp.bool,
    iso: wp.float32,
) -> wp.bool:
    return (values[i, j, k] >= iso) != above


@wp.kernel
def resolve_crossing_endpoints(
    mesh: wp.uint64,
    points: wp.array3d[wp.vec3],
    cap: wp.float32,
    iso: wp.float32,
    winding: wp.array3d[wp.float32],
    values: wp.array3d[wp.float32],
    out_field: wp.array3d[wp.float32],
) -> None:
    # The field marching cubes reads: every node as signed, except a capped node with an axis
    # neighbour on the other side of ``iso`` -- an edge marching cubes interpolates across -- which
    # gets its exact distance from the uncapped search. Only a field that is not 1-Lipschitz (signed
    # by the winding number of an open surface) has one.
    i, j, k = wp.tid()
    value = values[i, j, k]
    result = value
    if wp.abs(value) >= cap:
        above = value >= iso
        crossing = False
        if i > 0:
            crossing = crossing or crosses(values, i - 1, j, k, above, iso)
        if i + 1 < values.shape[0]:
            crossing = crossing or crosses(values, i + 1, j, k, above, iso)
        if j > 0:
            crossing = crossing or crosses(values, i, j - 1, k, above, iso)
        if j + 1 < values.shape[1]:
            crossing = crossing or crosses(values, i, j + 1, k, above, iso)
        if k > 0:
            crossing = crossing or crosses(values, i, j, k - 1, above, iso)
        if k + 1 < values.shape[2]:
            crossing = crossing or crosses(values, i, j, k + 1, above, iso)
        if crossing:
            p = points[i, j, k]
            query = wp.mesh_query_point_no_sign(mesh, p, FLOAT32_INF_CONSTANT)
            d = wp.length(p - wp.mesh_eval_position(mesh, query.face, query.u, query.v))
            result = signed_value(d, winding[i, j, k])
    out_field[i, j, k] = result


# ---------------------------------------------------------------------------------------------
# thicken_mesh
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# marching_cubes: the dense-lattice extraction, in two passes over the nodes
# ---------------------------------------------------------------------------------------------
#
# A port of ``warp.geometry.IsoSurfaceMarchingCubes.extract``'s algorithm (Apache-2.0): the same
# case tables, the same vertex numbering -- one vertex per crossing lattice edge, edges ordered by
# their lower node row-major and then by axis -- the same per-cell triangle order and the same
# interpolation arithmetic, so its output is Warp's bit for bit, vertex and face order included.
# What changed is the bookkeeping. Warp counts and emits vertices over a ``(nx, ny, nz, 3)`` edge
# grid and faces over the cell grid, with two zeroed ``3 * n_nodes`` int32 buffers, a scan of
# each count buffer and a ``(nx, ny, nz, 3)`` edge-to-vertex table the face pass reads back -- 44
# bytes of scratch a node, eleven times the field, and two scans and two host reads. Here one
# thread per node counts both its crossing edges and its cell's triangles into one ``vec2i``, one
# scan turns the counts into both offsets, and the emit pass finds an edge's vertex from its
# owner node's scanned count, recomputing which of that node's edges cross: 8 bytes a node, one
# scan, one read. The tables are Warp's public ``IsoSurfaceMarchingCubes`` attributes, packed into
# one array by ``levelset._marching_cubes_table``; the edge-owner layout is derived from them.

# Offsets into that packed table: the case-to-triangle ranges first (257 entries), then the
# triangles' local edge indices, then one packed entry per cube edge -- its owner corner's offset
# in bits 0-2 and its axis from bit 3 (``MARCHING_CUBES_EDGES``).
MC_TRI_BASE = len(IsoSurfaceMarchingCubes.CASE_TO_TRI_RANGE)
MC_EDGE_BASE = MC_TRI_BASE + len(IsoSurfaceMarchingCubes.TRI_LOCAL_INDICES)

MC_CORNERS = IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS


def _packed_edge(first: int, second: int) -> int:
    """One cube edge packed as ``owner offset | axis << 3``: its lower corner and its axis."""
    a, b = MC_CORNERS[first], MC_CORNERS[second]
    owner = [min(a[c], b[c]) for c in range(3)]
    axis = next(c for c in range(3) if a[c] != b[c])
    return owner[0] | (owner[1] << 1) | (owner[2] << 2) | (axis << 3)


MARCHING_CUBES_EDGES = tuple(
    _packed_edge(*pair) for pair in IsoSurfaceMarchingCubes.EDGE_TO_CORNERS
)

MARCHING_CUBES_TABLE = (
    tuple(IsoSurfaceMarchingCubes.CASE_TO_TRI_RANGE)
    + tuple(IsoSurfaceMarchingCubes.TRI_LOCAL_INDICES)
    + MARCHING_CUBES_EDGES
)


@wp.func
def mc_values_cross(here: wp.float32, there: wp.float32, iso: wp.float32) -> wp.int32:
    # 1 when an edge whose ends hold ``here`` and ``there`` straddles ``iso`` (``>=`` on one end,
    # ``<`` on the other: Warp's test, so a NaN end never crosses). Shared with the narrow-band
    # extraction in ``kernels/algorithms/poisson_band``, which reads a composite field.
    return wp.where((here >= iso and there < iso) or (here < iso and there >= iso), 1, 0)


@wp.func
def mc_edge_vertex(
    lower: wp.vec3,
    delta: wp.vec3,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    axis: wp.int32,
    here: wp.float32,
    there: wp.float32,
    iso: wp.float32,
    margin: wp.float32,
) -> wp.vec3:
    # The vertex on the crossing edge from node ``(i, j, k)`` along ``axis``: Warp's
    # ``extract_vertices_kernel`` arithmetic statement for statement, which is what keeps the dense
    # extraction bit-identical to it. ``margin`` 0 is Warp's ``clamp(t, 0, 1)`` exactly; above 0
    # it keeps the vertex that fraction of the edge off both nodes (``marching_cubes``'
    # ``edge_margin``).
    io = i + wp.where(axis == 0, 1, 0)
    jo = j + wp.where(axis == 1, 1, 0)
    ko = k + wp.where(axis == 2, 1, 0)
    t = (iso - here) / (there - here)
    t = wp.clamp(t, margin, 1.0 - margin)
    here_pos = lower + wp.vec3(
        wp.float32(i) * delta.x, wp.float32(j) * delta.y, wp.float32(k) * delta.z
    )
    there_pos = lower + wp.vec3(
        wp.float32(io) * delta.x, wp.float32(jo) * delta.y, wp.float32(ko) * delta.z
    )
    return wp.lerp(here_pos, there_pos, t)


@wp.func
def mc_triangle_edge(table: wp.array[wp.int32], slot: wp.int32) -> wp.vec4i:
    # The cube edge a triangle corner sits on, from its ``TRI_LOCAL_INDICES`` slot: the owner
    # corner's offset from the cell's lower node and the edge's axis.
    packed = table[MC_EDGE_BASE + table[MC_TRI_BASE + slot]]
    return wp.vec4i(packed & 1, (packed >> 1) & 1, (packed >> 2) & 1, packed >> 3)


@wp.func
def mc_edge_crosses(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    axis: wp.int32,
) -> wp.int32:
    # 1 when the lattice edge from node ``(i, j, k)`` along ``axis`` exists and its ends straddle
    # ``iso`` (``>=`` on one end, ``<`` on the other: Warp's test, so a NaN end never crosses).
    io = i + wp.where(axis == 0, 1, 0)
    jo = j + wp.where(axis == 1, 1, 0)
    ko = k + wp.where(axis == 2, 1, 0)
    if io >= field.shape[0] or jo >= field.shape[1] or ko >= field.shape[2]:
        return 0
    return mc_values_cross(field[i, j, k], field[io, jo, ko], iso)


@wp.func
def mc_case_code(
    field: wp.array3d[wp.float32], iso: wp.float32, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.int32:
    # The cell at node ``(i, j, k)``'s 8-bit case: bit ``c`` set when corner ``c`` is at or above
    # ``iso``, corners in ``CUBE_CORNER_OFFSETS`` order.
    code = wp.int32(0)
    for c in range(8):
        value = field[
            i + wp.static(MC_CORNERS[c][0]),
            j + wp.static(MC_CORNERS[c][1]),
            k + wp.static(MC_CORNERS[c][2]),
        ]
        if value >= iso:
            code += wp.static(1 << c)
    return code


@wp.func
def mc_cell_triangles(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    table: wp.array[wp.int32],
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.vec2i:
    # ``(first table slot, triangle count)`` of the cell whose lower corner is node ``(i, j, k)``;
    # a node on the lattice's upper face along any axis owns no cell and returns no triangles.
    if i + 1 >= field.shape[0] or j + 1 >= field.shape[1] or k + 1 >= field.shape[2]:
        return wp.vec2i(0, 0)
    code = mc_case_code(field, iso, i, j, k)
    start = table[code]
    return wp.vec2i(start, (table[code + 1] - start) // 3)


@wp.kernel
def marching_cubes_counts(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    table: wp.array[wp.int32],
    out_counts: wp.array[wp.vec2i],
) -> None:
    # Per node: how many of its three positive-axis edges cross ``iso`` (its vertices) and how many
    # triangles its cell emits. Scanned inclusively in place, the pair is both output offsets.
    i, j, k = wp.tid()
    vertices = (
        mc_edge_crosses(field, iso, i, j, k, 0)
        + mc_edge_crosses(field, iso, i, j, k, 1)
        + mc_edge_crosses(field, iso, i, j, k, 2)
    )
    cell = mc_cell_triangles(field, iso, table, i, j, k)
    out_counts[ravel_index(i, j, k, field.shape[1], field.shape[2])] = wp.vec2i(vertices, cell[1])


@wp.kernel
def marching_cubes_emit(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    lower: wp.vec3,
    delta: wp.vec3,
    margin: wp.float32,
    table: wp.array[wp.int32],
    offsets: wp.array[wp.vec2i],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # Writes node ``(i, j, k)``'s crossing-edge vertices and its cell's triangles at the slots the
    # inclusive scan of ``marching_cubes_counts`` gives (each count is subtracted back off its
    # inclusive total). The vertex arithmetic is ``mc_edge_vertex``'s.
    i, j, k = wp.tid()
    ny = field.shape[1]
    nz = field.shape[2]
    ends = offsets[ravel_index(i, j, k, ny, nz)]
    crossing = wp.vec3i(
        mc_edge_crosses(field, iso, i, j, k, 0),
        mc_edge_crosses(field, iso, i, j, k, 1),
        mc_edge_crosses(field, iso, i, j, k, 2),
    )
    slot = ends[0] - (crossing[0] + crossing[1] + crossing[2])
    for axis in range(3):
        if crossing[axis] != 0:
            there = field[
                i + wp.where(axis == 0, 1, 0),
                j + wp.where(axis == 1, 1, 0),
                k + wp.where(axis == 2, 1, 0),
            ]
            out_vertices[slot] = mc_edge_vertex(
                lower, delta, i, j, k, axis, field[i, j, k], there, iso, margin
            )
            slot += 1

    cell = mc_cell_triangles(field, iso, table, i, j, k)
    first_face = ends[1] - cell[1]
    for tri in range(cell[1]):
        for s in range(3):
            edge = mc_triangle_edge(table, cell[0] + 3 * tri + s)
            oi = i + edge[0]
            oj = j + edge[1]
            ok = k + edge[2]
            axis = edge[3]
            # The owner's vertices sit at the end of its scanned count, in axis order, so this
            # edge's is its total less one less each crossing edge on a higher axis. An edge the
            # case table names but the crossing test rejects -- only possible with a NaN corner,
            # which the case code reads as below ``iso`` -- has no vertex, and gets Warp's ``-1``.
            vertex = wp.int32(-1)
            if mc_edge_crosses(field, iso, oi, oj, ok, axis) != 0:
                vertex = offsets[ravel_index(oi, oj, ok, ny, nz)][0] - 1
                for higher in range(axis + 1, 3):
                    vertex -= mc_edge_crosses(field, iso, oi, oj, ok, higher)
            out_faces[3 * (first_face + tri) + s] = vertex


@wp.func
def shell_vertex(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    outside: wp.float32,
    inside: wp.float32,
    v: wp.int32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    # Both layers of a thickened shell at vertex ``v``: the outward-displaced copy in the first
    # ``n_vertices`` slots and the inward-displaced one after it, so the second layer's vertex ``v``
    # is at ``v + n_vertices`` and the face emitters below can shift by a constant.
    n_vertices = vertices.shape[0]
    position = vertices[v]
    normal = normals[v]
    out_vertices[v] = position + outside * normal
    out_vertices[n_vertices + v] = position - inside * normal


@wp.func
def shell_face(
    faces: wp.array[wp.int32], n_vertices: wp.int32, f: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # The two layers' triangles of face ``f``: the outer copy verbatim, the inner copy shifted by
    # ``n_vertices`` and **wound backwards**, because it faces into the shell rather than out of it.
    # Corners 1 and 2 are swapped, which is ``repair.flip_faces_masked``'s reversal without the
    # mask.
    n_faces = faces.shape[0] // 3
    corner0 = faces[3 * f]
    corner1 = faces[3 * f + 1]
    corner2 = faces[3 * f + 2]
    out_faces[3 * f] = corner0
    out_faces[3 * f + 1] = corner1
    out_faces[3 * f + 2] = corner2
    inner = 3 * (n_faces + f)
    out_faces[inner] = n_vertices + corner0
    out_faces[inner + 1] = n_vertices + corner2
    out_faces[inner + 2] = n_vertices + corner1


@wp.func
def shell_band(
    boundary_edges: wp.array2d[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    e: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # The band closing the shell along boundary edge ``e``: two triangles spanning the outer edge
    # ``(a, b)`` and its inner copy. The winding follows the *directed* boundary edge, which
    # ``boundary.oriented_boundary_edges`` returns in the outer layer's own face winding -- so the
    # band inherits that orientation instead of guessing one, and the whole shell comes out
    # consistently wound. Verified on ``hemisphere`` and ``half_torus``: watertight, consistent, and
    # positive volume.
    outer_a = boundary_edges[e, 0]
    outer_b = boundary_edges[e, 1]
    inner_a = n_vertices + outer_a
    inner_b = n_vertices + outer_b
    slot = base + 6 * e
    out_faces[slot] = outer_a
    out_faces[slot + 1] = inner_b
    out_faces[slot + 2] = outer_b
    out_faces[slot + 3] = outer_a
    out_faces[slot + 4] = inner_a
    out_faces[slot + 5] = inner_b


@wp.kernel
def shell_mesh(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    outside: wp.float32,
    inside: wp.float32,
    faces: wp.array[wp.int32],
    boundary_edges: wp.array2d[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The whole thickened shell in one launch over ``n_vertices + n_faces + n_rim`` threads: the
    # three emitters write disjoint slots and read only the input, so each thread takes one vertex,
    # one face or one boundary edge by its range (``creation.revolve_mesh``'s layout).
    t = wp.int32(wp.tid())
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] // 3
    if t < n_vertices:
        shell_vertex(vertices, normals, outside, inside, t, out_vertices)
    elif t < n_vertices + n_faces:
        shell_face(faces, n_vertices, t - n_vertices, out_faces)
    else:
        shell_band(boundary_edges, n_vertices, 6 * n_faces, t - n_vertices - n_faces, out_faces)
