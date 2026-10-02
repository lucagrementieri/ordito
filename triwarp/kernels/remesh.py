from typing import Any

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT, TOLERANCE_ZERO_CONSTANT, UINT64_MAX_CONSTANT
from triwarp.kernels.adjacency import edge_pair_topology, write_edge_row, write_face_edge_keys
from triwarp.kernels.array import (
    LOOP_CONDITION,
    LOOP_ROUND,
    binary_search_sorted_contains,
    lowbias32,
    pack_edge_key,
    pack_index_triple,
    scanned_slot,
    sort3,
    sorted_run_start,
    to_vec2d,
    to_vec3,
    to_vec3d,
    unpack_edge_key,
)
from triwarp.kernels.grouping import (
    hash_find,
    hash_find_or_insert,
    hash_slot,
    hash_slot_words3,
    key_set_remove,
    next_slot,
    sorted_run_of_length,
)
from triwarp.kernels.predicates import (
    delone_metrics,
    dihedral_angle,
    is_unfold_quadrangle_convex,
    law_of_cosines_angle,
    mincircle_diameter_sq,
    orient2d,
    project_out_normal,
    triangle_aspect_ratio,
    triangle_normal,
    vector_angle,
)
from triwarp.kernels.scatter import (
    accumulate_endpoint_value,
    add_corner_triple,
    add_edge_valence,
    lock_two_rings,
    mark_corners,
    record_edge_incidence,
    stamp_two_rings,
    two_rings_hold,
)
from triwarp.kernels.triangles import (
    corner_triple,
    face_normal,
    face_normals_and_area,
    face_vertices_vec3d,
    local_corner,
    triangle_quality,
    write_corner_triple,
    write_row_triple,
)
from triwarp.kernels.voxels import squared_distance_to_own_cell_center, voxel_cell

# The collapse round loop's third state slot, **appended** after ``array.LOOP_ROUND`` and
# ``LOOP_CONDITION`` so the shared two keep their numbers: the total commits as of the end of
# the previous round, which is how ``end_collapse_round`` decides whether a round progressed.
COLLAPSE_COMMITS = wp.constant(wp.int32(2))
COLLAPSE_STATE_SIZE = 3

# Delaunay / Delone edge-flip constants. The flip predicate runs in float64 deliberately:
# circumcircle diameters of near-degenerate triangles round too coarsely in float32, which sends
# the flip loop non-terminating.
DELONE_CRITICAL_DOT = wp.constant(wp.float64(-0.9))
DELONE_EPS = wp.constant(wp.float64(1e-7))
NO_ANGLE_CHANGE_LIMIT = wp.constant(wp.float64(6.283185307179586))  # 2*pi (NoAngleChangeLimit)
F32_LARGE = wp.constant(wp.float32(3.0e38))  # "disabled gate" sentinel (~FLT_MAX)

# Loop subdivision stencil weights. The even-vertex relaxation uses Warren's beta rather than Loop's
# original trigonometric weight, which is the choice ``igl::loop`` makes; see `loop_even_positions`.
LOOP_ODD_ENDPOINT = wp.constant(wp.float32(3.0 / 8.0))
LOOP_ODD_OPPOSITE = wp.constant(wp.float32(1.0 / 8.0))
LOOP_BOUNDARY_SELF = wp.constant(wp.float32(3.0 / 4.0))
LOOP_BOUNDARY_NEIGHBOR = wp.constant(wp.float32(1.0 / 8.0))
LOOP_BETA_VALENCE_3 = wp.constant(wp.float32(3.0 / 16.0))
LOOP_BETA_NUMERATOR = wp.constant(wp.float32(3.0 / 8.0))

# Which of Loop's three even-vertex rules applies, as returned by ``loop_even_weights``. The mode is
# what a caller needs beyond the two weights, because the neighbour weight lands on a *different*
# neighbour set in each case: none, the boundary neighbours only, or the whole 1-ring.
LOOP_EVEN_KEEP = wp.constant(wp.int32(0))
LOOP_EVEN_BOUNDARY = wp.constant(wp.int32(1))
LOOP_EVEN_INTERIOR = wp.constant(wp.int32(2))


@wp.func
def edge_midpoint(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.vec3:
    v0 = vertices[unique_edges[e, 0]]
    v1 = vertices[unique_edges[e, 1]]
    return wp.lerp(v0, v1, wp.float32(0.5))


@wp.kernel
def compute_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    k = wp.int32(wp.tid())
    out_midpoints[k] = edge_midpoint(vertices, unique_edges, k)


@wp.func
def split_face_four(fv: wp.vec3i, mv: wp.vec3i) -> tuple[wp.vec3i, wp.vec3i, wp.vec3i, wp.vec3i]:
    # 1 -> 4 loop-subdivision template: three corner triangles, then the central triangle.
    t0 = wp.vec3i(fv[0], mv[0], mv[2])
    t1 = wp.vec3i(mv[0], fv[1], mv[1])
    t2 = wp.vec3i(mv[2], mv[1], fv[2])
    t3 = wp.vec3i(mv[0], mv[1], mv[2])
    return t0, t1, t2, t3


@wp.kernel
def subdivide_faces(
    faces: wp.array[wp.int32],
    corner_edge: wp.array[wp.int32],
    vertex_offset: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # Corner ``k``'s edge is ``corner_edge[3f + k]`` and every edge is split, so its new vertex is
    # ``vertex_offset`` plus the edge id -- the shift is applied here rather than materialised as a
    # ``3 * n_faces`` index buffer by a separate pass that this kernel would read straight back.
    f = wp.int32(wp.tid())
    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = wp.vec3i(
        vertex_offset + corner_edge[f * 3 + 0],
        vertex_offset + corner_edge[f * 3 + 1],
        vertex_offset + corner_edge[f * 3 + 2],
    )
    t0, t1, t2, t3 = split_face_four(fv, mv)
    base = f * 12
    out_faces[base + 0] = t0[0]
    out_faces[base + 1] = t0[1]
    out_faces[base + 2] = t0[2]
    out_faces[base + 3] = t1[0]
    out_faces[base + 4] = t1[1]
    out_faces[base + 5] = t1[2]
    out_faces[base + 6] = t2[0]
    out_faces[base + 7] = t2[1]
    out_faces[base + 8] = t2[2]
    out_faces[base + 9] = t3[0]
    out_faces[base + 10] = t3[1]
    out_faces[base + 11] = t3[2]


@wp.kernel
def loop_edge_opposites(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    out_opposite_sum: wp.array[wp.vec3],
    out_face_count: wp.array[wp.int32],
) -> None:
    # Per unique edge: how many faces use it, and the sum of the vertices opposite it in each.
    # Corner ``j`` of face ``f`` spans ``(fv[j], fv[j + 1])`` and its opposite vertex is
    # ``fv[j + 2]``, so one pass over the faces gathers both halves of the Loop odd-vertex stencil.
    f = wp.int32(wp.tid())
    for j in range(3):
        e = edge_of_corner[f * 3 + j]
        wp.atomic_add(out_opposite_sum, e, vertices[faces[f * 3 + (j + 2) % 3]])
        wp.atomic_add(out_face_count, e, 1)


@wp.func
def loop_odd_weights(edge_face_count: wp.int32) -> tuple[wp.float32, wp.float32]:
    # Loop's odd (edge) stencil as weights: 3/8 on each endpoint and 1/8 on each opposite vertex for
    # an interior edge, the midpoint rule otherwise. A boundary edge (one face) or a non-manifold
    # one (three or more) has no well-defined pair of opposite vertices, so both fall back together.
    # Shared by the position kernel and the interpolation-operator triplet kernels, so the operator
    # and the positions ``subdivide_loop`` returns cannot describe different surfaces.
    if edge_face_count == 2:
        return LOOP_ODD_ENDPOINT, LOOP_ODD_OPPOSITE
    return wp.float32(0.5), wp.float32(0.0)


@wp.kernel
def loop_vertex_rings(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_opposite_sum: wp.array[wp.vec3],
    out_valence: wp.array[wp.int32],
    out_ring_sum: wp.array[wp.vec3],
    out_boundary_count: wp.array[wp.int32],
    out_boundary_sum: wp.array[wp.vec3],
    out_odd_positions: wp.array[wp.vec3],
) -> None:
    # Per vertex: its valence and 1-ring position sum, plus the same two restricted to boundary
    # edges. Driven by the *unique* edge list rather than by the faces, so the valence is the number
    # of distinct neighbours on any input -- the count a per-face pass would have to deduplicate
    # (each neighbour appears twice around an interior vertex but once at a boundary).
    #
    # The edge's own odd (Loop) vertex rides the same thread, from the endpoints it has just loaded:
    # 3/8 on each endpoint and 1/8 on each of the two opposite vertices (``loop_odd_weights``).
    e = wp.int32(wp.tid())
    v0 = unique_edges[e, 0]
    v1 = unique_edges[e, 1]
    p0 = vertices[v0]
    p1 = vertices[v1]
    endpoint_weight, opposite_weight = loop_odd_weights(edge_face_count[e])
    out_odd_positions[e] = endpoint_weight * (p0 + p1) + opposite_weight * edge_opposite_sum[e]
    wp.atomic_add(out_valence, v0, 1)
    wp.atomic_add(out_valence, v1, 1)
    wp.atomic_add(out_ring_sum, v0, p1)
    wp.atomic_add(out_ring_sum, v1, p0)
    if edge_face_count[e] == 1:
        wp.atomic_add(out_boundary_count, v0, 1)
        wp.atomic_add(out_boundary_count, v1, 1)
        wp.atomic_add(out_boundary_sum, v0, p1)
        wp.atomic_add(out_boundary_sum, v1, p0)


@wp.func
def loop_even_weights(
    valence: wp.int32, boundary_count: wp.int32
) -> tuple[wp.float32, wp.float32, wp.int32]:
    # Loop's even (original) stencil as (self weight, neighbour weight, mode); see the LOOP_EVEN_*
    # constants for the mode. Warren's beta -- 3/16 at valence 3 and 3/(8n) above it -- the variant
    # ``igl::loop`` uses, not Loop's original weight. Shared as ``loop_odd_weights`` is.
    if boundary_count == 2:
        # Boundary vertex: 3/4 of itself, 1/8 of each neighbour *along the boundary*. Its interior
        # neighbours do not enter, which is what keeps a shared boundary curve identical on both
        # sides of a seam.
        return LOOP_BOUNDARY_SELF, LOOP_BOUNDARY_NEIGHBOR, LOOP_EVEN_BOUNDARY
    if boundary_count == 0 and valence > 0:
        beta = LOOP_BETA_VALENCE_3
        if valence != 3:
            beta = LOOP_BETA_NUMERATOR / wp.float32(valence)
        return wp.float32(1.0) - wp.float32(valence) * beta, beta, LOOP_EVEN_INTERIOR
    # Anything else keeps its position: an isolated vertex with no edges, or a non-manifold boundary
    # vertex where one or three-plus boundary edges meet and neither stencil is defined.
    return wp.float32(1.0), wp.float32(0.0), LOOP_EVEN_KEEP


@wp.kernel
def loop_even_positions(
    vertices: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    ring_sum: wp.array[wp.vec3],
    boundary_count: wp.array[wp.int32],
    boundary_sum: wp.array[wp.vec3],
    out_positions: wp.array[wp.vec3],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # Loop's even (original) vertices, relaxed towards their 1-ring; the rule is
    # ``loop_even_weights``, and the mode says which neighbour sum the weight multiplies.
    #
    # With non-empty triplet buffers (``subdivide_loop(return_operator=True)``) the vertex's
    # self-weight triplet of the interpolation operator rides the same thread: row ``v`` is the
    # relocated original vertex ``v`` (see the operator's triplet kernels below).
    v = wp.int32(wp.tid())
    self_weight, neighbor_weight, mode = loop_even_weights(valence[v], boundary_count[v])
    neighbor = ring_sum[v]
    if mode == LOOP_EVEN_BOUNDARY:
        neighbor = boundary_sum[v]
    out_positions[v] = self_weight * vertices[v] + neighbor_weight * neighbor
    if out_rows.shape[0] > 0:
        out_rows[v] = v
        out_cols[v] = v
        out_values[v] = self_weight


# The interpolation operator ``subdivide_loop(return_operator=True)`` assembles, emitted as triplets
# from the same three grids the positions come from and through the same two weight functions. Row
# ``v`` is the relocated original vertex ``v`` (its self-weight is written by
# ``loop_even_positions``); row ``n_vertices + e`` is the odd vertex on unique edge ``e``. Every
# slot is written -- a zero weight where a rule does not apply -- because
# ``triplet_buffers`` hands back uninitialized memory (CLAUDE.md section 3.7).
@wp.kernel
def loop_edge_triplets(
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    valence: wp.array[wp.int32],
    boundary_count: wp.array[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # Four triplets per unique edge: each endpoint's contribution to the odd row, and each
    # endpoint's contribution to the *other* endpoint's even row. Whether that second pair carries
    # any weight depends on the receiving vertex's mode -- the whole 1-ring for an interior vertex,
    # only the boundary neighbours for a boundary one -- which is the one place the operator has to
    # know what the accumulating ``loop_vertex_rings`` pass knows.
    e = wp.int32(wp.tid())
    v0 = unique_edges[e, 0]
    v1 = unique_edges[e, 1]
    is_boundary_edge = edge_face_count[e] == 1
    slot = base + 4 * e

    endpoint_weight, _opposite_weight = loop_odd_weights(edge_face_count[e])
    odd_row = n_vertices + e
    out_rows[slot] = odd_row
    out_cols[slot] = v0
    out_values[slot] = endpoint_weight
    out_rows[slot + 1] = odd_row
    out_cols[slot + 1] = v1
    out_values[slot + 1] = endpoint_weight

    for side in range(2):
        receiver = v0
        donor = v1
        if side == 1:
            receiver = v1
            donor = v0
        _self_weight, neighbor_weight, mode = loop_even_weights(
            valence[receiver], boundary_count[receiver]
        )
        weight = wp.float32(0.0)
        if mode == LOOP_EVEN_INTERIOR or (mode == LOOP_EVEN_BOUNDARY and is_boundary_edge):
            weight = neighbor_weight
        out_rows[slot + 2 + side] = receiver
        out_cols[slot + 2 + side] = donor
        out_values[slot + 2 + side] = weight


@wp.kernel
def loop_opposite_triplets(
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # The 1/8 wings of the odd stencil, over the same (face, corner) grid ``loop_edge_opposites``
    # sums them on: corner ``j`` spans ``(fv[j], fv[j+1])`` and its opposite vertex is ``fv[j+2]``.
    f = wp.int32(wp.tid())
    for j in range(3):
        e = edge_of_corner[f * 3 + j]
        _endpoint_weight, opposite_weight = loop_odd_weights(edge_face_count[e])
        slot = base + 3 * f + j
        out_rows[slot] = n_vertices + e
        out_cols[slot] = faces[f * 3 + (j + 2) % 3]
        out_values[slot] = opposite_weight


@wp.func
def hysteresis_bands(sizing: wp.float32) -> tuple[wp.float32, wp.float32]:
    # The Botsch-Kobbelt collapse-below / split-above pair for one vertex's target length. Both
    # bands from one read: asked for separately they are two passes over the same field for two
    # scalings of the same number.
    return 4.0 / 5.0 * sizing, 4.0 / 3.0 * sizing


@wp.func
def split_corner_midpoints(
    corner_edge: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    vertex_offset: wp.int32,
    f: wp.int32,
) -> wp.vec3i:
    # The new-vertex index on each of face ``f``'s three edges, ``-1`` where the edge is not split.
    # Corner ``k`` spans the edge ``corner_edge[3f + k]``, and a split edge's new vertex is appended
    # at ``vertex_offset`` plus its rank among the split edges, which is ``offsets[e]``.
    #
    # Read per corner rather than tabulated per edge and gathered: the table would be one launch
    # over the edges plus a ``3 * n_faces`` gather, both only to hand ``emit_size_faces`` three
    # values it can look up itself.
    mv = wp.vec3i(-1, -1, -1)
    for k in range(3):
        e = corner_edge[f * 3 + k]
        if split_mask[e]:
            mv[k] = vertex_offset + offsets[e]
    return mv


@wp.kernel
def split_child_counts_and_midpoints(
    corner_edge: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    write_midpoints: wp.int32,
    out_counts: wp.array[wp.int32],
    out_mid: wp.array[wp.vec3],
) -> None:
    # Two independent writes of one refine pass, over ``max(n_faces, n_edges)`` threads:
    #
    # - how many triangles face ``t`` becomes under ``emit_size_faces``' templates: one more than
    #   the number of its edges being split. Scanned, this is each face's first output row, so the
    #   emit pass writes the compact buffer directly rather than four fixed slots and a compaction;
    # - with ``write_midpoints`` (warp-uniform; unset when the caller supplies the positions), each
    #   split edge ``t``'s midpoint at its rank -- the rule ``split_corner_midpoints`` reads from
    #   the face side, naming the index the rank gives it.
    t = wp.int32(wp.tid())
    if t < out_counts.shape[0]:
        count = wp.int32(1)
        for k in range(3):
            if split_mask[corner_edge[t * 3 + k]]:
                count += 1
        out_counts[t] = count
    if write_midpoints != 0 and t < split_mask.shape[0] and split_mask[t]:
        out_mid[offsets[t]] = edge_midpoint(vertices, unique_edges, t)


@wp.kernel
def fill_edge_mean_sizing(
    sizing: wp.array[wp.float32],
    unique_edges: wp.array2d[wp.int32],
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    out_sizing: wp.array[wp.float32],
) -> None:
    # ``split_child_counts_and_midpoints``' midpoint for the sizing field rather than the position:
    # the value carried to a
    # new midpoint is the endpoint mean ``mark_long_edges`` tested the edge with.
    e = wp.int32(wp.tid())
    if split_mask[e]:
        out_sizing[offsets[e]] = wp.float32(0.5) * (
            sizing[unique_edges[e, 0]] + sizing[unique_edges[e, 1]]
        )


@wp.func
def unique_edge_length(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.float32:
    # Length of unique edge ``e``, in the subtraction order ``edges.edges_unique_length`` uses, so
    # a threshold test on it sees the same ``float32`` value that function would have returned.
    return wp.length(vertices[unique_edges[e, 1]] - vertices[unique_edges[e, 0]])


@wp.kernel
def mark_long_edges(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    max_edge: wp.float32,
    sizing: wp.array[wp.float32],
    use_sizing: wp.bool,
    out_long: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``length > target`` per unique edge, with the length computed here rather than read from a
    # separate length pass whose only consumer this is. The target is ``max_edge`` or, against a
    # per-vertex sizing field, the mean of the endpoints' -- the standard reading of a
    # vertex-sampled sizing function, and symmetric in the edge's orientation. One warp-uniform
    # branch rather than two kernels, since the two differ by a parameter; ``sizing`` is not read
    # (and may be a null array) when ``use_sizing`` is false. The verdict is written twice: as the
    # mask the split reads, and as the ``int32`` flag the caller scans in place into its ranks.
    e = wp.int32(wp.tid())
    target = max_edge
    if use_sizing:
        target = wp.float32(0.5) * (sizing[unique_edges[e, 0]] + sizing[unique_edges[e, 1]])
    is_long = unique_edge_length(vertices, unique_edges, e) > target
    out_long[e] = is_long
    out_flags[e] = wp.where(is_long, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_size_faces(
    faces: wp.array[wp.int32],
    corner_edge: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    vertex_offset: wp.int32,
    vertices: wp.array[wp.vec3],
    index_in: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    out_faces: wp.array2d[wp.int32],
    out_index: wp.array[wp.int32],
) -> None:
    # Re-triangulate face ``f`` by how many of its edges are split, writing its children straight
    # into rows ``face_offsets[f] ..`` of the compact output (the scanned counts of
    # ``split_child_counts_and_midpoints``).
    # The children come out in template order ``t0, t1, ..``, which is the order the fixed-slot
    # form's compaction kept them in, so the output is the same buffer either way.
    f = wp.int32(wp.tid())
    src = index_in[f]

    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = split_corner_midpoints(corner_edge, split_mask, offsets, vertex_offset, f)

    s0 = wp.where(mv[0] >= 0, wp.int32(1), wp.int32(0))
    s1 = wp.where(mv[1] >= 0, wp.int32(1), wp.int32(0))
    s2 = wp.where(mv[2] >= 0, wp.int32(1), wp.int32(0))
    count = s0 + s1 + s2

    # Up to four children; only the first ``count + 1`` are written.
    t0 = wp.vec3i(0, 0, 0)
    t1 = wp.vec3i(0, 0, 0)
    t2 = wp.vec3i(0, 0, 0)
    t3 = wp.vec3i(0, 0, 0)

    if count == 0:
        # No split edges: the face passes through unchanged.
        t0 = fv
    elif count == 1:
        # Rotate so the split edge is (a, b); fan its midpoint p to the
        # opposite corner c as [a, p, c], [p, b, c].
        j = wp.int32(0)
        if s1 == 1:
            j = 1
        if s2 == 1:
            j = 2
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        t0 = wp.vec3i(a, p, c)
        t1 = wp.vec3i(p, b, c)
    elif count == 2:
        # Rotate so the unsplit edge is (c, a); emit corner triangle [p, b, q]
        # plus the quad (a, p, q, c) cut along its shorter diagonal.
        u = wp.int32(0)
        if s1 == 0:
            u = 1
        if s2 == 0:
            u = 2
        j = (u + 1) % 3
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        q = mv[(j + 1) % 3]
        t0 = wp.vec3i(p, b, q)
        d_aq = wp.length_sq(vertices[a] - vertices[q])
        d_pc = wp.length_sq(vertices[p] - vertices[c])
        if d_aq <= d_pc:
            t1 = wp.vec3i(a, p, q)
            t2 = wp.vec3i(a, q, c)
        else:
            t1 = wp.vec3i(a, p, c)
            t2 = wp.vec3i(p, q, c)
    else:
        # Three split edges: the regular 1 -> 4 split (matches subdivide).
        t0, t1, t2, t3 = split_face_four(fv, mv)

    base = face_offsets[f]
    write_row_triple(out_faces, base, t0[0], t0[1], t0[2])
    out_index[base] = src
    if count >= 1:
        write_row_triple(out_faces, base + 1, t1[0], t1[1], t1[2])
        out_index[base + 1] = src
    if count >= 2:
        write_row_triple(out_faces, base + 2, t2[0], t2[1], t2[2])
        out_index[base + 2] = src
    if count >= 3:
        write_row_triple(out_faces, base + 3, t3[0], t3[1], t3[2])
        out_index[base + 3] = src


# ---------------------------------------------------------------------------
# Region-restricted subdivision helpers
# ---------------------------------------------------------------------------


@wp.kernel
def mark_long_region_edges(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    max_edge: wp.float32,
    out_long: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``mark_long_edges``' uniform test restricted to the edges with at least one incident face in
    # the region -- region-border edges included. One thread per face corner, over a zeroed mask:
    # the region test and the length test in one launch, where a per-edge pass would first need the
    # region membership scattered into an edge mask of its own. An edge is tested once per region
    # face holding it, at most twice on a manifold, and every write is the same ``True``, so the
    # race is benign. ``out_flags`` (zeroed too) carries the same marks as ``int32``, for the
    # caller to scan in place into the split's ranks.
    i = wp.int32(wp.tid())
    if region_flags[i // 3] != 0:
        e = inverse[i]
        if unique_edge_length(vertices, unique_edges, e) > max_edge:
            out_long[e] = wp.bool(True)
            out_flags[e] = wp.int32(1)


# ---------------------------------------------------------------------------
# Region-restricted density refinement (Liepa 2003, section 3)
# ---------------------------------------------------------------------------


@wp.kernel
def scale_attribute_edge_keys(
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
) -> None:
    # ``adjacency.face_edge_keys`` with every ``region`` face's keys lifted by
    # ``base ** 2``, above any key an outside face packs: one sort then puts the surrounding
    # mesh's edges first, in ``edges_unique``'s ascending-key order, and the region's after them,
    # so ``scatter_scale_attribute`` reads either class off the sort with no compaction.
    f = wp.int32(wp.tid())
    write_face_edge_keys(faces, f, 3 * f, base, out_keys)
    lift = wp.where(region[f], base * base, wp.uint64(0))
    for k in range(3):
        out_keys[3 * f + k] = out_keys[3 * f + k] + lift


@wp.kernel
def scatter_scale_attribute(
    vertices: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.uint64],
    base: wp.uint64,
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    # One unique edge per run of ``scale_attribute_edge_keys``' sorted keys: its length added to
    # both endpoints and counted at each, in ``edges_unique``'s orientation and order. Only the
    # outside class is counted, unless it is empty (the smallest key is already lifted), when the
    # whole mesh is.
    i = wp.int32(wp.tid())
    lift = base * base
    key = sorted_keys[i]
    everything = sorted_keys[0] >= lift
    if not sorted_run_start(sorted_keys, i):
        return
    if key >= lift:
        if not everything:
            return
        key = key - lift
    lo, hi = unpack_edge_key(key, base)
    accumulate_endpoint_value(lo, hi, wp.length(vertices[hi] - vertices[lo]), out_sum, out_valence)


@wp.func
def density_split_wanted(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    scale_a: wp.float32,
    scale_b: wp.float32,
    scale_c: wp.float32,
    alpha: wp.float32,
) -> wp.bool:
    # Liepa's density criterion for splitting a patch triangle at its centroid. Each vertex carries
    # a *scale attribute* -- the average length of the edges incident to it in the surrounding mesh
    # -- and the centroid inherits the mean of its three. The triangle is split when, for **every**
    # corner ``m``, the centroid is far from ``m`` relative to the centroid's own scale *and* the
    # centroid's scale is coarse relative to ``m``'s:
    #
    #     alpha * |centroid - v_m| > scale(centroid)   and   alpha * scale(centroid) > scale(v_m)
    #
    # The first clause is what refines; the second is what stops the recursion at the surrounding
    # sampling instead of running to the tolerance. Both must hold at all three corners, so a
    # triangle already matching its neighbourhood's density is left alone and the pass converges.
    #
    # ``alpha`` is the paper's ``sqrt(2)``, exposed because it is the one real knob: raising it
    # refines further, lowering it stops sooner.
    centroid = (a + b + c) / wp.float32(3.0)
    scale = (scale_a + scale_b + scale_c) / wp.float32(3.0)
    if alpha * scale <= wp.max(scale_a, wp.max(scale_b, scale_c)):
        return False
    return (
        alpha * wp.length(centroid - a) > scale
        and alpha * wp.length(centroid - b) > scale
        and alpha * wp.length(centroid - c) > scale
    )


@wp.kernel
def mark_density_splits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    alpha: wp.float32,
    out_split: wp.array[wp.int32],
) -> None:
    # Which region faces want a centroid split this pass, as 0/1 so the result scans directly into
    # the new-vertex slots ``emit_density_splits`` needs. A face outside the region never splits,
    # which is what keeps the refinement inside the patch.
    f = wp.int32(wp.tid())
    if not region[f]:
        out_split[f] = wp.int32(0)
        return
    i, j, k = corner_triple(faces, f)
    wanted = density_split_wanted(
        vertices[i], vertices[j], vertices[k], scale[i], scale[j], scale[k], alpha
    )
    out_split[f] = wp.where(wanted, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_density_splits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    split_offsets: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_positions: wp.array[wp.vec3],
    out_scale: wp.array[wp.float32],
    out_faces: wp.array[wp.int32],
    out_region: wp.array[wp.bool],
    out_region_flags: wp.array[wp.int32],
) -> None:
    # One pass of the 1 -> 3 centroid split, faces and new vertices in the same launch.
    #
    # A centroid split is **per-triangle independent** -- the new vertex is interior to the triangle
    # and no edge is divided -- so unlike edge bisection it needs none of ``subdivide_to_size``'s
    # crack-free 1/2/3 templates and no agreement with the neighbours. That is the whole reason this
    # criterion suits a parallel refinement: one scan for the new-vertex slots, and one kernel.
    #
    # The face slots need no scan of their own: every face before ``f`` becomes one face plus two
    # more if it split, so ``f``'s first output face is ``f + 2 * split_offsets[f]``. The offsets
    # are the ``n_faces + 1`` exclusive scan of ``mark_density_splits``' flags, so a face split
    # exactly where they step.
    #
    # Child faces inherit their parent's region membership, matching
    # ``subdivide_region_to_size``, so a caller's patch mask survives the pass. The membership is
    # also written as the ``int32`` flags the flip pass reads, unless ``out_region_flags`` is a null
    # descriptor -- the ``array.astype`` launch that would otherwise follow every pass.
    f = wp.int32(wp.tid())
    i, j, k = corner_triple(faces, f)
    slot = split_offsets[f]
    first = f + wp.int32(2) * slot
    base = first * 3
    inside = region[f]
    flag = wp.where(inside, 1, 0)
    write_flags = out_region_flags.shape[0] > 0
    if split_offsets[f + 1] == slot:
        out_faces[base + 0] = i
        out_faces[base + 1] = j
        out_faces[base + 2] = k
        out_region[first] = inside
        if write_flags:
            out_region_flags[first] = flag
        return

    center = n_vertices + slot
    out_positions[slot] = (vertices[i] + vertices[j] + vertices[k]) / wp.float32(3.0)
    out_scale[slot] = (scale[i] + scale[j] + scale[k]) / wp.float32(3.0)
    out_faces[base + 0] = i
    out_faces[base + 1] = j
    out_faces[base + 2] = center
    out_faces[base + 3] = j
    out_faces[base + 4] = k
    out_faces[base + 5] = center
    out_faces[base + 6] = k
    out_faces[base + 7] = i
    out_faces[base + 8] = center
    for child in range(3):
        out_region[first + child] = inside
        if write_flags:
            out_region_flags[first + child] = flag


# ---------------------------------------------------------------------------
# Float64 Delone edge-flip predicate. Computed in double precision for the reason the constants
# above give.
# ---------------------------------------------------------------------------


@wp.func
def _segments_dist_sq_d(p1: wp.vec3d, q1: wp.vec3d, p2: wp.vec3d, q2: wp.vec3d) -> wp.float64:
    # Squared distance between segments [p1,q1] and [p2,q2] (Ericson, clamped closest points).
    eps = wp.float64(1e-30)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    aa = wp.length_sq(d1)
    ee = wp.length_sq(d2)
    f = wp.dot(d2, r)
    s = wp.float64(0.0)
    t = wp.float64(0.0)
    if aa <= eps and ee <= eps:
        return wp.length_sq(p1 - p2)
    if aa <= eps:
        t = wp.clamp(f / ee, wp.float64(0.0), wp.float64(1.0))
    else:
        cc = wp.dot(d1, r)
        if ee <= eps:
            s = wp.clamp(-cc / aa, wp.float64(0.0), wp.float64(1.0))
        else:
            bb = wp.dot(d1, d2)
            denom = aa * ee - bb * bb
            if denom != wp.float64(0.0):
                s = wp.clamp((bb * f - cc * ee) / denom, wp.float64(0.0), wp.float64(1.0))
            t = (bb * s + f) / ee
            if t < wp.float64(0.0):
                t = wp.float64(0.0)
                s = wp.clamp(-cc / aa, wp.float64(0.0), wp.float64(1.0))
            elif t > wp.float64(1.0):
                t = wp.float64(1.0)
                s = wp.clamp((bb - cc) / aa, wp.float64(0.0), wp.float64(1.0))
    cp1 = p1 + s * d1
    cp2 = p2 + t * d2
    return wp.length_sq(cp1 - cp2)


@wp.func
def _check_delone_quadrangle_d(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, max_angle_change: wp.float64
) -> wp.bool:
    # Returns True to KEEP the current diagonal (a-c), False to flip to (b-d). Exact port of
    # checkDeloneQuadrangle(Vector3d, ...).
    n_abc = triangle_normal(a, b, c)
    n_acd = triangle_normal(a, c, d)
    old_pocket = wp.dot(n_abc, n_acd) < DELONE_CRITICAL_DOT

    n_abd = triangle_normal(a, b, d)
    n_dbc = triangle_normal(d, b, c)
    new_pocket = wp.dot(n_abd, n_dbc) < DELONE_CRITICAL_DOT

    if old_pocket != new_pocket:
        return new_pocket

    if old_pocket:
        metric_ac = wp.max(mincircle_diameter_sq(a, c, d), mincircle_diameter_sq(c, a, b))
        metric_bd = wp.max(mincircle_diameter_sq(b, d, a), mincircle_diameter_sq(d, b, c))
        return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)

    if max_angle_change < NO_ANGLE_CHANGE_LIMIT:
        old_angle = dihedral_angle(n_abd, n_dbc, d - b)
        new_angle = dihedral_angle(n_abc, n_acd, a - c)
        if wp.abs(old_angle - new_angle) > max_angle_change:
            return True

    metric_ac, metric_bd = delone_metrics(a, b, c, d)

    if wp.isinf(metric_ac):
        if wp.isinf(metric_bd):
            return wp.length_sq(a - c) <= wp.length_sq(b - d)
        return False
    return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)


# ---------------------------------------------------------------------------
# 2D orientation / incircle predicate (for delaunay_triangulation)
# ---------------------------------------------------------------------------


@wp.func
def _incircle_d(a: wp.vec2d, b: wp.vec2d, c: wp.vec2d, d: wp.vec2d) -> wp.float64:
    # Positive iff d is inside the circumcircle of CCW triangle (a, b, c). The 3x3 determinant is
    # kept expanded in components (as in ``predicates.orient2d``) because the term order is what
    # makes the sign reliable near cocircularity; only the squared radii go through ``length_sq``.
    ad = a - d
    bd = b - d
    cd = c - d
    a2 = wp.length_sq(ad)
    b2 = wp.length_sq(bd)
    c2 = wp.length_sq(cd)
    return (
        ad[0] * (bd[1] * c2 - b2 * cd[1])
        - ad[1] * (bd[0] * c2 - b2 * cd[0])
        + a2 * (bd[0] * cd[1] - bd[1] * cd[0])
    )


# ---------------------------------------------------------------------------
# Shared parallel edge-flip core
# ---------------------------------------------------------------------------


@wp.kernel
def mark_edge_pair_starts(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    edge_set_mask: wp.int32,
    base: wp.uint64,
    edge_set: wp.array[wp.uint64],
    out_starts: wp.array[wp.int32],
    out_halfedge_row: wp.array[wp.int32],
    out_valence: wp.array[wp.int32],
) -> None:
    # ``grouping.mark_group_starts`` at ``length=2`` (the same ``sorted_run_of_length`` rule). Both
    # emit ``int32`` for the same reason: the flag feeds ``warp.utils.array_scan``, which has no
    # bool overload. Flags the position that starts a run of *exactly* two equal keys, i.e. an edge
    # shared by exactly two face corners.
    #
    # Differs from ``mark_unique_edge_starts`` below only in requiring the run to be exactly two:
    # that one takes every run whatever its length, because the decimation pass wants all unique
    # edges where a flip pass wants only the manifold-interior ones.
    #
    # An incremental build (``out_halfedge_row`` given, ``edge_set_mask >= 0``) seeds the flip
    # loop's other two structures in the same pass: every halfedge starts with no row
    # (``emit_flip_topology`` then writes the interior ones), and every run -- every undirected
    # edge, whatever its multiplicity -- goes into the zeroed duplicate-edge set. A non-empty
    # ``out_valence`` (zeroed by the caller) gets every vertex's degree from the same runs
    # (``scatter.add_edge_valence``), which the valence flip objective reads.
    i = wp.int32(wp.tid())
    if out_halfedge_row.shape[0] > 0:
        out_halfedge_row[order[i]] = -1
    if sorted_run_start(sorted_keys, i):
        if edge_set_mask >= 0:
            hash_find_or_insert(sorted_keys[i], edge_set, edge_set_mask)
        if out_valence.shape[0] > 0:
            add_edge_valence(sorted_keys, i, base, out_valence)
    out_starts[i] = wp.where(sorted_run_of_length(sorted_keys, n, i, 2), wp.int32(1), wp.int32(0))


@wp.func
def write_flip_row(
    faces: wp.array[wp.int32],
    edge_0: wp.int32,
    edge_1: wp.int32,
    slot: wp.int32,
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
) -> None:
    # One interior-edge row of the flip tables from its two halfedges: the shared edge's sorted
    # endpoints, the face pair ascending and each face's opposite apex aligned with it. Only the
    # halfedges' *faces* and the edge are read, so the row is the same whichever of the two comes
    # first -- which is what lets ``refresh_flip_rows`` rebuild a row from halfedges that moved.
    shared_a, shared_b, face_0, face_1, unshared_0, unshared_1 = edge_pair_topology(
        faces, edge_0, edge_1
    )
    out_adjacency_edges[slot, 0] = shared_a
    out_adjacency_edges[slot, 1] = shared_b
    if face_0 <= face_1:
        out_adjacency[slot, 0] = face_0
        out_adjacency[slot, 1] = face_1
        out_unshared[slot, 0] = unshared_0
        out_unshared[slot, 1] = unshared_1
    else:
        out_adjacency[slot, 0] = face_1
        out_adjacency[slot, 1] = face_0
        out_unshared[slot, 0] = unshared_1
        out_unshared[slot, 1] = unshared_0


@wp.kernel
def emit_flip_topology(
    faces: wp.array[wp.int32],
    order: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
    out_row_halfedges: wp.array2d[wp.int32],
    out_halfedge_row: wp.array[wp.int32],
) -> None:
    # The whole face-adjacency table a flip pass needs, in one launch over the ``3 * n_faces``
    # sorted edge slots: ``adjacency.face_adjacency``'s pair table, its shared-edge endpoints and
    # ``face_adjacency_unshared``'s opposite apexes. Everything comes out of the two grouped *edge*
    # indices, so no edge table is materialized and nothing is gathered through one.
    #
    # ``ranks`` is the inclusive scan of ``starts``, so ``ranks[i] - 1`` is the row a start writes
    # -- the same ascending-key row order the ``flatnonzero`` compaction inside
    # ``grouping.group`` produces, which is what keeps this byte-identical to the composed path.
    #
    # An incremental build keeps the two halfedges too, in both directions (row -> halfedges,
    # halfedge -> row; ``mark_edge_pair_starts`` resets the second first), which is what lets
    # ``commit_flips`` keep the rows current across a flip round instead of regrouping the whole
    # mesh. A plain build passes ``None`` for both.
    i = wp.int32(wp.tid())
    if starts[i] == 0:
        return
    slot = ranks[i] - 1
    edge_0 = order[i]
    edge_1 = order[i + 1]
    write_flip_row(faces, edge_0, edge_1, slot, out_adjacency, out_adjacency_edges, out_unshared)
    if out_halfedge_row.shape[0] == 0:
        return
    out_row_halfedges[slot, 0] = edge_0
    out_row_halfedges[slot, 1] = edge_1
    out_halfedge_row[edge_0] = slot
    out_halfedge_row[edge_1] = slot


@wp.kernel
def refresh_flip_rows(
    faces: wp.array[wp.int32],
    row_halfedges: wp.array2d[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
    out_face_claim: wp.array[wp.uint64],
    out_edge_claim: wp.array[wp.uint64],
) -> None:
    # Every row of the flip tables again, from the halfedges ``commit_flips`` kept current: the
    # rows ``emit_flip_topology`` would write after a full regroup, row for row, except that each
    # edge keeps the row it had instead of moving to its new key's rank -- which the flip loop does
    # not depend on, since ``claim_flips`` ranks candidates by key rather than by row.
    #
    # Also re-arms the next round's two claim tables, which this round's commit was the last to
    # read and the next round's candidates never read: two memsets fewer per round, the row threads
    # striding over tables longer than the row count.
    k = wp.int32(wp.tid())
    rows = row_halfedges.shape[0]
    for f in range(k, out_face_claim.shape[0], rows):
        out_face_claim[f] = UINT64_MAX_CONSTANT
    for s in range(k, out_edge_claim.shape[0], rows):
        out_edge_claim[s] = UINT64_MAX_CONSTANT
    write_flip_row(
        faces,
        row_halfedges[k, 0],
        row_halfedges[k, 1],
        k,
        out_adjacency,
        out_adjacency_edges,
        out_unshared,
    )


@wp.kernel
def mark_sorted_run_starts(
    sorted_keys: wp.array[wp.uint64], out_starts: wp.array[wp.int32]
) -> None:
    # The first position of every run of equal keys: ``mark_unique_edge_starts`` below over a
    # buffer with no padded tail, for ``_FlipTopology.edges_unique``. ``int32`` for the scan.
    i = wp.int32(wp.tid())
    start = wp.int32(0)
    if sorted_run_start(sorted_keys, i):
        start = wp.int32(1)
    out_starts[i] = start


@wp.kernel
def emit_sorted_unique_edges(
    faces: wp.array[wp.int32],
    order: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
) -> None:
    # ``edges.edges_unique``'s two returns from a sort of every corner's edge key: ``ranks`` is the
    # inclusive scan of ``mark_sorted_run_starts``, so ``ranks[i] - 1`` is the ascending-key unique
    # index ``grouping.unique_1d`` assigns the key at sorted position ``i``. ``emit_pass_edges``
    # below is the decimation pass's form: live corners and a capacity bound, a chunked exclusive
    # scan in place of the inclusive one, and the incidence and adjacency slots in the same launch.
    i = wp.int32(wp.tid())
    corner = order[i]
    e = ranks[i] - 1
    out_inverse[corner] = e
    if starts[i] != 0:
        write_edge_row(faces, corner, e, out_unique_edges)


@wp.func
def _resolve_flip_quad(
    faces: wp.array[wp.int32], f0: wp.int32, u: wp.int32, v: wp.int32, d0: wp.int32, d1: wp.int32
) -> wp.vec4i:
    # Orient the flip quad so f0 traverses a->c (its apex d0 is the left apex "d"); the other
    # face's apex d1 is the right apex "b". Returns (a, b, c, d); a<0 marks inconsistent winding.
    a = wp.int32(-1)
    c = wp.int32(-1)
    for k in range(3):
        va = faces[f0 * 3 + k]
        vb = faces[f0 * 3 + (k + 1) % 3]
        if va == u and vb == v:
            a = u
            c = v
        if va == v and vb == u:
            a = v
            c = u
    return wp.vec4i(a, d1, c, d0)


@wp.func
def edge_key_exists(
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key: wp.uint64,
) -> wp.bool:
    # Is ``key`` an edge of the current triangulation? Two spellings of one set, chosen by a
    # warp-uniform selector: the flip loop's sorted keys while a build's sort is still current
    # (``edge_set_mask < 0``), its hashed key set once a flip has committed and made the sort stale.
    if edge_set_mask < 0:
        return binary_search_sorted_contains(sorted_edge_keys, key)
    return hash_find(key, edge_set, edge_set_mask) >= 0


@wp.func
def _resolve_flip_quad_guarded(
    faces: wp.array[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key_base: wp.uint64,
    k: wp.int32,
    f0: wp.int32,
    out_quad: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # Shared flip-candidate preamble: reject missing apexes, inconsistent winding, b == d, and
    # flips that would duplicate an existing edge. Returns the unpacked ``(a, b, c, d)`` with
    # a < 0 when not flippable, so every candidate kernel opens with the same two lines; out_quad[k]
    # is written only for valid quads (claim/commit read quad[k] only when the caller has set
    # out_flip[k], which stays False for rejected/non-flipped edges).
    a = wp.int32(-1)
    b = wp.int32(-1)
    c = wp.int32(-1)
    d = wp.int32(-1)
    d0 = unshared[k, 0]
    d1 = unshared[k, 1]
    if d0 >= 0 and d1 >= 0:
        quad = _resolve_flip_quad(faces, f0, adjacency_edges[k, 0], adjacency_edges[k, 1], d0, d1)
        if (
            quad[0] >= 0
            and quad[1] != quad[3]
            and not edge_key_exists(
                sorted_edge_keys, edge_set, edge_set_mask, pack_edge_key(quad[1], quad[3], key_base)
            )
        ):
            a = quad[0]
            b = quad[1]
            c = quad[2]
            d = quad[3]
            out_quad[k, 0] = a
            out_quad[k, 1] = b
            out_quad[k, 2] = c
            out_quad[k, 3] = d
    return a, b, c, d


@wp.func
def _resolve_flip_quad_in_region(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key_base: wp.uint64,
    k: wp.int32,
    out_quad: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # ``_resolve_flip_quad_guarded`` plus the region test, folded into its ``a < 0`` contract: an
    # edge with either incident face outside the region is not flippable, for the same reason a
    # missing apex is not.
    #
    # Region-restricted rather than universal because two of the four candidate kernels have no
    # region to restrict to: ``valence_flip_candidates`` is whole-mesh by construction (its only
    # caller takes no ``region``) and ``incircle_flip_candidates`` triangulates a planar point set.
    f0 = adjacency[k, 0]
    if region_flags[f0] == 0 or region_flags[adjacency[k, 1]] == 0:
        return wp.int32(-1), wp.int32(-1), wp.int32(-1), wp.int32(-1)
    return _resolve_flip_quad_guarded(
        faces,
        adjacency_edges,
        unshared,
        sorted_edge_keys,
        edge_set,
        edge_set_mask,
        key_base,
        k,
        f0,
        out_quad,
    )


@wp.func
def flip_quad_positions_d(
    vertices: wp.array[Any], a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32
) -> tuple[wp.vec3d, wp.vec3d, wp.vec3d, wp.vec3d]:
    # The four corners of a flip quad, promoted to ``float64``. Every flip predicate in this module
    # -- convexity, the Delone empty-circumcircle test, the segment distance -- runs in float64 on a
    # float32 vertex buffer, because they are *branches*: a lost digit changes a flip decision
    # rather than a printed number.
    return (
        to_vec3d(vertices[a]),
        to_vec3d(vertices[b]),
        to_vec3d(vertices[c]),
        to_vec3d(vertices[d]),
    )


@wp.kernel
def delone_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key_base: wp.uint64,
    max_angle_change: wp.float32,
    max_deviation_sq: wp.float32,
    critical_aspect: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    a, b, c, d = _resolve_flip_quad_in_region(
        faces,
        adjacency,
        adjacency_edges,
        unshared,
        region_flags,
        sorted_edge_keys,
        edge_set,
        edge_set_mask,
        key_base,
        k,
        out_quad,
    )
    if a < 0:
        return
    ap, bp, cp, dp = flip_quad_positions_d(vertices, a, b, c, d)
    if max_deviation_sq < F32_LARGE:
        if _segments_dist_sq_d(ap, cp, bp, dp) > wp.float64(max_deviation_sq):
            return
    if not is_unfold_quadrangle_convex(ap, bp, cp, dp):
        return
    angle = wp.float64(max_angle_change)
    if critical_aspect < F32_LARGE and angle < NO_ANGLE_CHANGE_LIMIT:
        max_aspect = wp.max(triangle_aspect_ratio(ap, cp, dp), triangle_aspect_ratio(cp, ap, bp))
        if max_aspect > wp.float64(critical_aspect):
            angle = NO_ANGLE_CHANGE_LIMIT
    out_flip[k] = not _check_delone_quadrangle_d(ap, bp, cp, dp, angle)


@wp.kernel
def incircle_flip_candidates(
    points: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key_base: wp.uint64,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    a, b, c, d = _resolve_flip_quad_guarded(
        faces,
        adjacency_edges,
        unshared,
        sorted_edge_keys,
        edge_set,
        edge_set_mask,
        key_base,
        k,
        f0,
        out_quad,
    )
    if a < 0:
        return
    ap = to_vec2d(points[a])
    bp = to_vec2d(points[b])
    cp = to_vec2d(points[c])
    dp = to_vec2d(points[d])
    # Post-flip triangles (a, b, d) and (d, b, c) must both be positively oriented (convex quad).
    if orient2d(ap, bp, dp) <= wp.float64(0.0) or orient2d(dp, bp, cp) <= wp.float64(0.0):
        return
    # f0 = (a, c, d) is CCW; flip iff the opposite apex b lies inside its circumcircle.
    out_flip[k] = _incircle_d(ap, cp, dp, bp) > wp.float64(0.0)


@wp.func
def flip_priority(
    adjacency_edges: wp.array2d[wp.int32], key_base: wp.uint64, k: wp.int32
) -> wp.uint64:
    # A candidate's rank in the independent-set claims: its edge's key. Interior-edge keys are
    # distinct, and a regroup emits the rows in ascending key order, so the smallest key is the
    # smallest row index there -- the claims pick the same winners whether or not the rows have
    # been regrouped since, which is what lets the flip loop keep its rows across rounds.
    return pack_edge_key(adjacency_edges[k, 0], adjacency_edges[k, 1], key_base)


@wp.func
def new_diagonal_claim_slot(
    quad: wp.array2d[wp.int32], key_base: wp.uint64, edge_claim_mask: wp.int32, k: wp.int32
) -> wp.int32:
    # The edge-claim slot of candidate ``k``'s new diagonal ``b-d``: the one address ``claim_flips``
    # writes and ``flip_claim_won`` reads it back from, so the two cannot disagree on it.
    return hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)


@wp.func
def flip_claim_won(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    face_claim: wp.array[wp.uint64],
    edge_claim: wp.array[wp.uint64],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    k: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.bool]:
    # Did candidate ``k`` win every claim ``claim_flips`` below wrote -- both its faces and the
    # hashed slot of the new diagonal it would create? Returns the two face indices alongside the
    # verdict because the caller needs them straight afterwards, and both are already loaded here.
    if not flip[k]:
        return wp.int32(-1), wp.int32(-1), False
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    priority = flip_priority(adjacency_edges, key_base, k)
    if face_claim[f0] != priority or face_claim[f1] != priority:
        return f0, f1, False
    if edge_claim[new_diagonal_claim_slot(quad, key_base, edge_claim_mask, k)] != priority:
        return f0, f1, False
    return f0, f1, True


@wp.kernel
def claim_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_face_claim: wp.array[wp.uint64],
    out_edge_claim: wp.array[wp.uint64],
) -> None:
    k = wp.int32(wp.tid())
    if not flip[k]:
        return
    priority = flip_priority(adjacency_edges, key_base, k)
    wp.atomic_min(out_face_claim, adjacency[k, 0], priority)
    wp.atomic_min(out_face_claim, adjacency[k, 1], priority)
    wp.atomic_min(
        out_edge_claim, new_diagonal_claim_slot(quad, key_base, edge_claim_mask, k), priority
    )


@wp.func
def write_flipped_quad(
    out_faces: wp.array[wp.int32],
    f0: wp.int32,
    f1: wp.int32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    d: wp.int32,
) -> None:
    # Rewrite the two faces of a flipped quad ``(a, b, c, d)``: the new diagonal is b-d, so the
    # faces become ``(a, b, d)`` and ``(c, d, b)``, which is the pair that preserves the original
    # winding.
    #
    # One winding convention, named once rather than spelled out by each of the two kernels that
    # commits a flip (the extrinsic ``commit_flips`` and the intrinsic ``commit_intrinsic_flips``).
    # A duplicated *decision* rather than duplicated arithmetic: a permuted copy still writes two
    # well-formed triangles covering the same quad, so the mesh stays manifold and only its
    # orientation quietly inverts along the flipped edges.
    write_corner_triple(out_faces, f0, a, b, d)
    write_corner_triple(out_faces, f1, c, d, b)


@wp.func
def local_directed_edge(
    faces: wp.array[wp.int32], f: wp.int32, u: wp.int32, v: wp.int32
) -> wp.int32:
    # The corner ``j`` of face ``f`` whose halfedge runs ``u -> v`` (``faces[3f + j] == u`` and the
    # next corner is ``v``), or ``-1``.
    j = wp.int32(-1)
    for k in range(3):
        if faces[f * 3 + k] == u and faces[f * 3 + (k + 1) % 3] == v:
            j = k
    return j


@wp.func
def move_row_halfedge(
    row_halfedges: wp.array2d[wp.int32], row: wp.int32, old: wp.int32, new: wp.int32
) -> None:
    # Repoint interior-edge row ``row``'s halfedge ``old`` at ``new``. Race-free across a commit
    # launch: a row's two halfedges lie in two different faces, each face belongs to at most one
    # committing flip, and a flip only ever writes the column holding a halfedge of its own face --
    # the other column never equals ``old``, whatever the other flip is writing into it.
    if row < 0:
        return
    if row_halfedges[row, 0] == old:
        row_halfedges[row, 0] = new
    else:
        row_halfedges[row, 1] = new


@wp.kernel
def commit_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    face_claim: wp.array[wp.uint64],
    edge_claim: wp.array[wp.uint64],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    edge_set_mask: wp.int32,
    row_halfedges: wp.array2d[wp.int32],
    halfedge_row: wp.array[wp.int32],
    edge_set: wp.array[wp.uint64],
    valence: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # Commit the winning flips, and keep the loop's incremental state describing the new faces:
    # the halfedge <-> row maps (so ``refresh_flip_rows`` can rebuild every row without a regroup),
    # the duplicate-edge set once it exists (``edge_set_mask >= 0``; the old diagonal out, the new
    # one in -- it cannot already be there,
    # or the candidate guard would have rejected the flip, and the edge claim stops two flips
    # creating it) and, when given, the vertex valences the diagonal moves between.
    #
    # The quad is ``(a, b, c, d)`` with ``f0`` traversing ``a -> c`` and apex ``d``, ``f1``
    # traversing ``c -> a`` and apex ``b`` (``_resolve_flip_quad``); ``write_flipped_quad`` turns
    # them into ``(a, b, d)`` and ``(c, d, b)``. Each side edge keeps its row but changes halfedge:
    # ``a-b`` and ``b-c`` were ``f1``'s, ``c-d`` and ``d-a`` ``f0``'s, and afterwards ``a-b`` and
    # ``d-a`` are ``f0``'s, ``c-d`` and ``b-c`` ``f1``'s.
    k = wp.int32(wp.tid())
    f0, f1, won = flip_claim_won(
        flip, quad, adjacency, adjacency_edges, face_claim, edge_claim, edge_claim_mask, key_base, k
    )
    if not won:
        return
    a = quad[k, 0]
    b = quad[k, 1]
    c = quad[k, 2]
    d = quad[k, 3]
    if halfedge_row.shape[0] == 0:
        # A plain build's round: no incremental state to keep; the caller regroups.
        write_flipped_quad(out_faces, f0, f1, a, b, c, d)
        wp.atomic_add(out_count, 0, 1)
        return
    j = local_directed_edge(out_faces, f0, a, c)
    i = local_directed_edge(out_faces, f1, c, a)
    old_cd = f0 * 3 + (j + 1) % 3
    old_da = f0 * 3 + (j + 2) % 3
    old_ab = f1 * 3 + (i + 1) % 3
    old_bc = f1 * 3 + (i + 2) % 3
    row_cd = halfedge_row[old_cd]
    row_da = halfedge_row[old_da]
    row_ab = halfedge_row[old_ab]
    row_bc = halfedge_row[old_bc]

    write_flipped_quad(out_faces, f0, f1, a, b, c, d)
    # (a, b, d): a->b, b->d, d->a;  (c, d, b): c->d, d->b, b->c.
    halfedge_row[f0 * 3 + 0] = row_ab
    halfedge_row[f0 * 3 + 1] = k
    halfedge_row[f0 * 3 + 2] = row_da
    halfedge_row[f1 * 3 + 0] = row_cd
    halfedge_row[f1 * 3 + 1] = k
    halfedge_row[f1 * 3 + 2] = row_bc
    row_halfedges[k, 0] = f0 * 3 + 1
    row_halfedges[k, 1] = f1 * 3 + 1
    move_row_halfedge(row_halfedges, row_ab, old_ab, f0 * 3 + 0)
    move_row_halfedge(row_halfedges, row_da, old_da, f0 * 3 + 2)
    move_row_halfedge(row_halfedges, row_cd, old_cd, f1 * 3 + 0)
    move_row_halfedge(row_halfedges, row_bc, old_bc, f1 * 3 + 2)

    if edge_set_mask >= 0:
        key_set_remove(pack_edge_key(a, c, key_base), edge_set, edge_set_mask)
        hash_find_or_insert(pack_edge_key(b, d, key_base), edge_set, edge_set_mask)
    # ``valence`` is passed as ``None`` (a null descriptor, shape 0) when the loop tracks none.
    if valence.shape[0] > 0:
        wp.atomic_sub(valence, a, 1)
        wp.atomic_sub(valence, c, 1)
        wp.atomic_add(valence, b, 1)
        wp.atomic_add(valence, d, 1)
    wp.atomic_add(out_count, 0, 1)


# ===========================================================================
# Isotropic explicit remeshing (Botsch-Kobbelt split/collapse/flip/smooth/reproject)
#
# Split reuses subdivide_to_size. The kernels below add: feature/boundary classification
# (per-vertex FREE/CREASE/CORNER codes), a parallel edge-collapse primitive with full 1-ring
# locking + link-condition guard, valence-driven edge flips, tangential Laplacian smoothing, and
# reprojection of free vertices onto the original surface.
# ===========================================================================
FREE_VERTEX = wp.constant(wp.int32(0))
CREASE_VERTEX = wp.constant(wp.int32(1))
CORNER_VERTEX = wp.constant(wp.int32(2))

# What ``collapse_survivor`` decided about where the merged vertex may go.
COLLAPSE_REJECTED = wp.constant(wp.int32(0))  # the edge must not collapse at all
COLLAPSE_PINNED = wp.constant(wp.int32(1))  # the survivor keeps its own position
COLLAPSE_FREE = wp.constant(wp.int32(2))  # the caller places it -- midpoint, or a quadric optimum


@wp.func
def collapse_survivor_of_codes(
    cu: wp.int32, cv: wp.int32, u: wp.int32, v: wp.int32, is_boundary: wp.bool
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # ``collapse_survivor`` with the two endpoint codes already in hand. The quadric pass derives
    # them from the feature counts in the kernel that scores the edge, so it keeps no code table.
    if cu == CORNER_VERTEX and cv == CORNER_VERTEX:
        # Two corners: the edge between two fixed points cannot shorten.
        return u, v, COLLAPSE_REJECTED
    if cu >= CREASE_VERTEX and cv >= CREASE_VERTEX:
        # Two feature vertices: collapse only along a boundary edge, and only when both are plain
        # creases. When that holds neither endpoint is preferred, so the placement stays free.
        if not (is_boundary and cu == CREASE_VERTEX and cv == CREASE_VERTEX):
            return u, v, COLLAPSE_REJECTED
        return u, v, COLLAPSE_FREE
    if cu >= CREASE_VERTEX:
        return u, v, COLLAPSE_PINNED
    if cv >= CREASE_VERTEX:
        return v, u, COLLAPSE_PINNED
    return u, v, COLLAPSE_FREE


@wp.func
def collapse_survivor(
    codes: wp.array[wp.int32], u: wp.int32, v: wp.int32, is_boundary: wp.bool
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # Which endpoint of edge ``(u, v)`` survives the collapse, which one is removed, and whether
    # the survivor's position is pinned or free -- the feature rule alone, with no geometry in it.
    #
    # One rule, two decimators: ``collapse_candidates`` and ``quadric_collapse_candidates`` differ
    # only in what they do with ``COLLAPSE_FREE`` -- one takes the midpoint, the other minimizes the
    # summed quadric.
    return collapse_survivor_of_codes(codes[u], codes[v], u, v, is_boundary)


@wp.func
def is_feature_edge(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    e: wp.int32,
    feature_angle: wp.float32,
) -> wp.bool:
    # Is unique edge ``e`` a feature edge: a boundary edge, or an interior one whose two faces meet
    # at more than ``feature_angle``? The decision rule behind every FREE / CREASE / CORNER code,
    # shared by ``scatter_feature_edge_counts`` and the quadric pass's ``count_pass_edges``.
    count = edge_face_count[e]
    feature = count == 1
    if count == 2:
        normal_a, _area_a = face_normals_and_area(vertices, faces, edge_faces[e, 0])
        normal_b, _area_b = face_normals_and_area(vertices, faces, edge_faces[e, 1])
        feature = vector_angle(normal_a, normal_b) > feature_angle
    return feature


@wp.kernel
def scatter_feature_edge_counts(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    feature_angle: wp.float32,
    out_feature_count: wp.array[wp.int32],
    out_boundary_vertex: wp.array[wp.bool],
) -> None:
    # Add 1 to both endpoints of every feature edge: a boundary edge, or an interior edge whose two
    # faces meet at more than ``feature_angle``. One launch over the unique edges answers both
    # questions from the incidence table, where ``_classify`` used to re-group the same 3 * n_faces
    # rows twice (once as boundary edges, once as face adjacency) to ask them separately.
    e = wp.int32(wp.tid())
    boundary = edge_face_count[e] == 1
    if is_feature_edge(vertices, faces, edge_face_count, edge_faces, e, feature_angle):
        v0 = unique_edges[e, 0]
        v1 = unique_edges[e, 1]
        wp.atomic_add(out_feature_count, v0, 1)
        wp.atomic_add(out_feature_count, v1, 1)
        if boundary:
            # Every writer stores the same value, so the mask needs no atomic.
            out_boundary_vertex[v0] = True
            out_boundary_vertex[v1] = True


@wp.func
def finalize_vertex_codes(feature_count: wp.int32) -> wp.int32:
    # 0 feature edges -> FREE; exactly 2 -> CREASE (on a smooth feature/boundary line);
    # anything else (1 = feature endpoint, >=3 = junction) -> CORNER (frozen).
    code = CORNER_VERTEX
    if feature_count == 0:
        code = FREE_VERTEX
    elif feature_count == 2:
        code = CREASE_VERTEX
    return code


@wp.func
def csr_common_neighbor_count(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.int32:
    # Number of vertices adjacent to both a and b (two nested scans; degrees are tiny).
    count = wp.int32(0)
    for i in range(offsets[a], offsets[a + 1]):
        w = columns[i]
        for j in range(offsets[b], offsets[b + 1]):
            if columns[j] == w:
                count += 1
    return count


@wp.func
def satisfies_link_condition(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    u: wp.int32,
    v: wp.int32,
    is_boundary: wp.bool,
) -> wp.bool:
    # May edge (u, v) collapse without changing the surface's topology? Exactly two vertices
    # adjacent to both endpoints for an interior edge -- the two apexes of its own faces -- and one
    # for a boundary edge. A third shared neighbour means the edge closes a tetrahedral loop the
    # collapse would pinch shut.
    #
    # Shared by both collapse-candidate kernels because it is a *decision rule*: two copies could
    # diverge into accepting an edge in one decimator and rejecting it in the other.
    required = 2
    if is_boundary:
        required = 1
    return csr_common_neighbor_count(offsets, columns, u, v) == required


@wp.func
def move_flips_normal(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    moved: wp.int32,
    partner: wp.int32,
    target: wp.vec3,
) -> wp.bool:
    # Would moving ``moved`` to ``target`` invert any face it still belongs to? Every incident face
    # keeps its other two corners and must keep its orientation.
    #
    # ``partner`` names a vertex whose incident faces are exempt, and it is what makes one helper
    # serve both callers of this rule. A *collapse* welds ``moved`` onto ``partner``, so the two
    # faces holding both endpoints vanish and must be skipped; a *smoothing* step moves one vertex
    # and welds nothing, so it passes ``-1``, which no corner index equals and which therefore
    # exempts nothing.
    #
    # This is the guard that separates a usable decimator from one that produces self-intersecting
    # geometry, and it is why the vertex-face CSR is built at all.
    for slot in range(vertex_face_offsets[moved], vertex_face_offsets[moved + 1]):
        f = vertex_faces[slot]
        i0, i1, i2 = corner_triple(faces, f)
        if i0 == partner or i1 == partner or i2 == partner:
            continue
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        before = wp.cross(p1 - p0, p2 - p0)
        if i0 == moved:
            p0 = target
        elif i1 == moved:
            p1 = target
        else:
            p2 = target
        after = wp.cross(p1 - p0, p2 - p0)
        before_length = wp.length(before)
        after_length = wp.length(after)
        if before_length <= 0.0:
            continue  # already degenerate: nothing to invert
        if after_length <= 0.0:
            return True  # the collapse would flatten it outright
        if wp.dot(before / before_length, after / after_length) < COLLAPSE_MIN_NORMAL_DOT:
            return True
    return False


@wp.func
def collapse_folds_a_face(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    s: wp.int32,
    r: wp.int32,
    p: wp.vec3,
) -> wp.bool:
    # The fold veto of an edge collapse welding ``r`` onto ``s`` at ``p``: both endpoints move to
    # ``p``, so both directions of ``move_flips_normal`` are tested, unconditionally -- even under a
    # pinned placement, where ``s`` stays put and its half is a no-op. One decision rule for both
    # decimators, ``collapse_candidates`` and ``quadric_collapse_candidates``.
    return move_flips_normal(
        vertices, faces, vertex_face_offsets, vertex_faces, r, s, p
    ) or move_flips_normal(vertices, faces, vertex_face_offsets, vertex_faces, s, r, p)


# ---------------------------------------------------------------------------
# Readback-free decimation pass (see ``remesh._DecimationBuffers``)
#
# Every kernel below works on **fixed-capacity** buffers whose live prefix length lives in a device
# array, so a whole pass can be issued once and replayed as a CUDA graph. Padding is carried by a
# **dummy vertex** at index ``n_vertices_capacity``, which every padded face corner points at: the
# padded faces are the dummy triangle, so any per-face kernel may run over them and find a
# degenerate face that contributes nothing. Per-edge and per-corner kernels instead stop at the live
# counts in ``state``, because a padded corner or edge names that one dummy and counting them would
# pile every one of their atomics onto a single address. The edge buffers keep one **dummy edge
# slot** at index ``n_edges_capacity``, reached only if the edge capacity were ever exceeded.
# ---------------------------------------------------------------------------

DECIMATION_FACES = wp.constant(wp.int32(0))
DECIMATION_VERTICES = wp.constant(wp.int32(1))
DECIMATION_EDGES = wp.constant(wp.int32(2))
DECIMATION_COMMITS = wp.constant(wp.int32(3))

# The claim table's value at a vertex no candidate has claimed yet: above every ``scramble_index``.
UNCLAIMED_KEY = wp.constant(wp.int64(2**63 - 1))

EDGE_KEY_PAD = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))


@wp.kernel(enable_backward=False)
def collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    codes: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    out_survivor: wp.array[wp.int32],
    out_removed: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
    out_claim: wp.array[wp.int64],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
) -> None:
    # ``low`` and ``high`` are per *vertex* rather than scalars so that one code path serves both
    # the uniform target and an adaptive sizing field; the uniform case fills them with a constant.
    # An edge's own band is the mean of its endpoints', matching ``mark_long_edges``. The length is
    # computed here: this is its only reader, and it reads it at its own edge.
    #
    # The edge threads also seed the per-vertex state the claim and the commit start from -- the
    # unclaimed keys, the identity collapse map and the working positions -- striding over the
    # vertices: this kernel reads none of it, and the next two launches are its only readers. That
    # is a fill, an ``arange`` and a clone fewer per pass.
    k = wp.int32(wp.tid())
    for t in range(k, out_remap.shape[0], unique_edges.shape[0]):
        out_claim[t] = UNCLAIMED_KEY
        out_remap[t] = t
        out_positions[t] = vertices[t]
    out_survivor[k] = -1
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    if unique_edge_length(vertices, unique_edges, k) >= wp.float32(0.5) * (low[u] + low[v]):
        return
    is_boundary = edge_face_count[k] == 1

    # Choose the surviving vertex and its target position; this decimator places a free collapse at
    # the edge midpoint, where ``quadric_collapse_candidates`` minimizes the summed quadric.
    s, r, placement = collapse_survivor(codes, u, v, is_boundary)
    if placement == COLLAPSE_REJECTED:
        return
    p = wp.lerp(vertices[u], vertices[v], 0.5)
    if placement == COLLAPSE_PINNED:
        p = vertices[s]

    if not satisfies_link_condition(offsets, columns, u, v, is_boundary):
        return

    # Anti-oscillation: reject if the collapse would create an edge longer than the high band. The
    # band is read at the far endpoint ``w``, so a collapse reaching into a finely-sized region is
    # judged by that region's target rather than by the survivor's.
    #
    # **Both** rings under a free placement, not just the removed vertex's. The reattached edges
    # from ``r``'s neighbours are the obvious new ones, but a ``COLLAPSE_FREE`` placement moves the
    # *survivor* to the midpoint as well, so every edge from ``s``'s own neighbours to ``p`` is
    # equally new and equally able to overshoot the band. Under ``COLLAPSE_PINNED`` the survivor
    # keeps its position and its ring is unchanged, which is the case one walk covers.
    for i in range(offsets[r], offsets[r + 1]):
        w = columns[i]
        if w != s and wp.length(p - vertices[w]) > high[w]:
            return
    if placement == COLLAPSE_FREE:
        for i in range(offsets[s], offsets[s + 1]):
            w = columns[i]
            if w != r and wp.length(p - vertices[w]) > high[w]:
                return

    # The fold veto (``collapse_folds_a_face``), last because every test above rejects more
    # cheaply.
    #
    # Nothing else here notices a collapse that inverts an incident face: the link condition is
    # topological and the band walks above bound *lengths*. It is effectively free, because
    # rejecting a collapse early removes more downstream work than the vertex-face CSR costs.
    if collapse_folds_a_face(vertices, faces, vertex_face_offsets, vertex_faces, s, r, p):
        return

    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = p


@wp.func
def scramble_index(index: wp.int32) -> wp.int64:
    # Spatially incoherent *and injective* lock key for the independent-set pass, from the
    # candidate's own index. Two separate properties, and the selection needs both.
    #
    # **Incoherent**: ``edges_unique`` orders edges lexicographically by endpoint index, which on
    # any structured mesh is spatially monotone -- and a monotone key field has one local
    # minimum, so a min-key lock commits a single collapse per pass however many candidates there
    # are. The high half is ``array.lowbias32`` with the top bit cleared, so the key stays
    # non-negative and ``INT64_MAX`` remains usable as the unclaimed sentinel.
    #
    # **Injective**, which is why the key is 64 bits and not the natural 32: the win test
    # (``scatter.two_rings_hold``) is equality against a neighbourhood minimum, so two candidates
    # sharing a key both win and both commit -- overlapping 1-rings, a corrupted mesh rather than a
    # worse one. A masked ``lowbias32`` is exactly 2-to-1, so the index goes in the low half, which
    # disturbs nothing but a tie: that now goes to the lower index instead of to both. It is also
    # what lets both collapse paths run **one** lock pass rather than following it with a second,
    # raw-index one.
    hashed = wp.int64(lowbias32(wp.uint32(index)) & wp.uint32(0x7FFFFFFF))
    return (hashed << wp.int64(32)) | wp.int64(index)


@wp.kernel(enable_backward=False)
def claim_collapse_key(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
) -> None:
    # The winning (smallest scrambled) key over the closed 1-rings of both endpoints. The quadric
    # pass makes the same claim inside ``drop_locked_and_claim``, and both read the answer the same
    # way (``scatter.two_rings_hold``).
    #
    # The key is ``scramble_index(k)`` and not ``k`` for the reason that function records: a min-key
    # lock over a *spatially monotone* key field has essentially one local minimum, so it commits a
    # single collapse per pass however many candidates there are. Hashing it is what makes the
    # collapse stage track its target size on a structured mesh, which
    # ``test_collapse_pass_commits_a_useful_fraction_on_a_structured_patch`` asserts against; the
    # icosphere fixtures cannot see it, because there the split stage does all the work.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    lock_two_rings(offsets, columns, s, removed[k], scramble_index(k), out_min_key)


@wp.kernel(enable_backward=False)
def commit_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    claim: wp.array[wp.int64],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    if not two_rings_hold(offsets, columns, claim, s, r, scramble_index(k)):
        return
    out_remap[r] = s
    out_positions[s] = pos[k]
    wp.atomic_add(out_count, 0, 1)


@wp.func
def remapped_corner_triple(
    faces: wp.array[wp.int32], remap: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.bool]:
    # Face ``f``'s corners through a vertex map, and whether they are still three distinct
    # vertices -- a face survives a vertex remap only if they are. Every decimation here ends in
    # one: an edge collapse merges two of them, vertex clustering sends two into the same cell.
    a, b, c = corner_triple(faces, f)
    i0 = remap[a]
    i1 = remap[b]
    i2 = remap[c]
    return i0, i1, i2, i0 != i1 and i1 != i2 and i0 != i2


@wp.kernel
def valence_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key_base: wp.uint64,
    valence: wp.array[wp.int32],
    boundary_vertex: wp.array[wp.bool],
    feature_angle: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    # Never flip a feature edge (sharp dihedral between the two incident faces).
    n0 = face_normal(vertices, faces, f0)
    n1 = face_normal(vertices, faces, f1)
    # ``vector_angle`` is the atan2 form; two coplanar faces across an edge is the common
    # case here and is exactly where ``acos(dot)`` loses its digits, and this is a *branch*,
    # so the lost digits change a flip decision rather than a printed number. The sibling
    # feature test in ``classify_vertices`` above already reads this way.
    if vector_angle(n0, n1) > feature_angle:
        return
    a, b, c, d = _resolve_flip_quad_guarded(
        faces,
        adjacency_edges,
        unshared,
        sorted_edge_keys,
        edge_set,
        edge_set_mask,
        key_base,
        k,
        f0,
        out_quad,
    )
    if a < 0:
        return
    ap, bp, cp, dp = flip_quad_positions_d(vertices, a, b, c, d)
    if not is_unfold_quadrangle_convex(ap, bp, cp, dp):
        return
    # Shape guard. Convexity makes the flip *legal* but says nothing about the shape of what it
    # produces, and the valence objective below is blind to geometry: on a graded mesh it will
    # happily turn two slivers into two worse ones, which in float32 lands on exactly-zero area.
    # ``delone_flip_candidates`` has its own deviation and aspect gates; this is the equivalent for
    # the valence objective.
    #
    # ``triangle_aspect_ratio`` is circumradius / 2 * inradius and returns +inf for a degenerate
    # triangle, so the two tests below read as "never create a degenerate triangle" and "never make
    # the worse of the pair worse". Post-flip faces are (a, b, d) and (c, d, b) -- see commit_flips.
    aspect_after = wp.max(triangle_aspect_ratio(ap, bp, dp), triangle_aspect_ratio(cp, dp, bp))
    if not wp.isfinite(aspect_after):
        return
    if aspect_after > wp.max(triangle_aspect_ratio(ap, cp, dp), triangle_aspect_ratio(cp, ap, bp)):
        return
    # Target valence: 4 on the boundary, 6 in the interior.
    ta = wp.where(boundary_vertex[a], 4, 6)
    tb = wp.where(boundary_vertex[b], 4, 6)
    tc = wp.where(boundary_vertex[c], 4, 6)
    td = wp.where(boundary_vertex[d], 4, 6)
    va = valence[a]
    vb = valence[b]
    vc = valence[c]
    vd = valence[d]
    before = (
        (va - ta) * (va - ta)
        + (vb - tb) * (vb - tb)
        + (vc - tc) * (vc - tc)
        + (vd - td) * (vd - td)
    )
    after = (
        (va - 1 - ta) * (va - 1 - ta)
        + (vb + 1 - tb) * (vb + 1 - tb)
        + (vc - 1 - tc) * (vc - 1 - tc)
        + (vd + 1 - td) * (vd + 1 - td)
    )
    out_flip[k] = after < before


@wp.kernel
def accumulate_one_ring(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    vertex_areas: wp.array[wp.float32],
    out_sum: wp.array[wp.vec3],
    out_weight: wp.array[wp.float32],
) -> None:
    # The area-equalizing one ring of Botsch-Kobbelt: each neighbour is weighted by its own
    # barycentric area, so the centroid leans toward the sparsely sampled side of the ring and the
    # relaxation redistributes sampling density rather than only straightening the surface.
    #
    # The *unweighted* centroid this replaced could not: it is a fixed point of a regular graded
    # grid, which is exactly the input the stage exists for. Area weighting improves the curved
    # fixtures, perturbs an already-uniform structured grid slightly, and speeds the whole call up
    # either way, because a better-shaped mesh gives the split and collapse stages less to do.
    e = wp.int32(wp.tid())
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    area_u = vertex_areas[u]
    area_v = vertex_areas[v]
    wp.atomic_add(out_sum, u, area_v * vertices[v])
    wp.atomic_add(out_weight, u, area_v)
    wp.atomic_add(out_sum, v, area_u * vertices[u])
    wp.atomic_add(out_weight, v, area_u)


@wp.func
def tangential_smooth_step(
    vertex: wp.vec3,
    code: wp.int32,
    normal: wp.vec3,
    ring_sum: wp.vec3,
    ring_weight: wp.float32,
    lam: wp.float32,
) -> wp.vec3:
    # Move a free vertex toward its area-weighted one-ring centroid, but only within the tangent
    # plane, so the surface is smoothed without being shrunk. Pinned vertices, isolated ones and
    # any vertex whose whole ring is degenerate (zero total area) stay put.
    p = vertex
    if code != FREE_VERTEX or ring_weight <= 0.0:
        return p
    centroid = ring_sum / ring_weight
    delta = centroid - p
    tangential = project_out_normal(delta, normal)
    return p + lam * tangential


@wp.kernel(enable_backward=False)
def smooth_free_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    codes: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    ring_sum: wp.array[wp.vec3],
    ring_weight: wp.array[wp.float32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    lam: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # One tangential relaxation step, vetoed per vertex by the same fold rule the two collapse
    # candidates run -- AGENTS.md section 2.4's one decision rule, one spelling.
    #
    # **It is insurance, and the reason to keep it is that it is free**, not that a fixture needs
    # it: veto against no-veto is within noise and the whole remesh suite passes either way. It
    # fires on a graded saddle patch and on no other fixture.
    #
    # **A convex vertex link cannot fold under this step at all**, which is why the well-shaped
    # fixtures cannot reach the veto: the target is a convex combination of the one ring, so it
    # lies inside the link polygon, and a point inside the link cannot invert a fan triangle. It
    # takes a strongly non-convex link -- a notch of near neighbours opposite far, heavy ones --
    # which is what ``test_smooth_pass_vetoes_a_move_that_would_fold_a_face`` builds.
    #
    # The veto is read against the *current* positions of the whole ring, so a pass that moves
    # several neighbours at once can in principle still fold; it bounds each move against the
    # geometry it was computed from, which is what an explicit relaxation can promise. A rejected
    # vertex simply keeps its position, so the pass can never do worse than not running.
    v = wp.int32(wp.tid())
    proposed = tangential_smooth_step(
        vertices[v], codes[v], normals[v], ring_sum[v], ring_weight[v], lam
    )
    if move_flips_normal(vertices, faces, vertex_face_offsets, vertex_faces, v, -1, proposed):
        out_positions[v] = vertices[v]
        return
    out_positions[v] = proposed


@wp.func
def reproject_vertices(
    vertex: wp.vec3, code: wp.int32, mesh_id: wp.uint64, max_dist: wp.float32
) -> wp.vec3:
    # Snap a free vertex back onto the closest point of the original surface, undoing the drift the
    # smoothing pass introduces. Pinned vertices and failed queries keep their position.
    if code != FREE_VERTEX:
        return vertex
    query = wp.mesh_query_point_no_sign(mesh_id, vertex, max_dist)
    if query.result:
        return wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    return vertex


@wp.func
def clamp_to_surface_band(
    vertex: wp.vec3, mesh_id: wp.uint64, max_deviation: wp.float32, max_dist: wp.float32
) -> wp.vec3:
    # Pull a vertex back until it is within ``max_deviation`` of the original surface, along the
    # line to its own closest point. Unlike ``reproject_vertices`` this applies to *every* vertex --
    # a crease or corner is exactly the kind that drifts and that reprojection refuses to touch --
    # and it moves the vertex only as far as the bound requires, so a vertex already inside the band
    # is untouched and detail is not flattened onto the input surface.
    query = wp.mesh_query_point_no_sign(mesh_id, vertex, max_dist)
    if not query.result:
        return vertex
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    offset = vertex - closest
    distance = wp.length(offset)
    if distance <= max_deviation:
        return vertex
    return closest + offset * (max_deviation / distance)


@wp.kernel
def build_intrinsic_twins(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    out_twin: wp.array[wp.int32],
) -> None:
    # The *one* point in ``intrinsic_delaunay``'s loop where a vertex-pair-keyed adjacency table
    # (``_FlipTopology``'s, built once from the caller's still-simplicial input) is trustworthy: a
    # halfedge ``h = 3 * f + e`` opposite corner ``e`` (``edge_lengths``' own indexing) paired with
    # its twin across each of ``adjacency``'s rows. Every flip after this one maintains ``out_twin``
    # incrementally instead of re-deriving it, because a second flip can create a second edge
    # between two already-adjacent vertices, and at that point no vertex-pair key can tell which of
    # several same-key corners is which edge's true twin -- see ``intrinsic_delaunay_candidates``.
    k = wp.int32(wp.tid())
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    corner0 = local_corner(faces, f0, unshared[k, 0])
    corner1 = local_corner(faces, f1, unshared[k, 1])
    if corner0 < 0 or corner1 < 0:
        return
    h0 = f0 * 3 + corner0
    h1 = f1 * 3 + corner1
    out_twin[h0] = h1
    out_twin[h1] = h0


@wp.kernel
def intrinsic_delaunay_candidates(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    twin: wp.array[wp.int32],
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
    out_new_length: wp.array[wp.float32],
    out_neighbors: wp.array2d[wp.int32],
    out_face_claim: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_no_remap: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
) -> None:
    # Mark the interior edges that violate the local Delaunay condition, measure what the flipped
    # edge would be -- both from edge lengths only, which is what makes the retriangulation
    # intrinsic: no vertex moves, so the *surface* is unchanged and only its triangulation improves
    # -- and claim the flip's two faces for the independent set ``commit_intrinsic_flips`` takes.
    #
    # Indexed per *halfedge* (``h = 3 * f + e``, the edge opposite corner ``e`` -- ``edge_lengths``'
    # own convention) and read through ``twin`` rather than a duplicate-edge-keyed adjacency table.
    # Nothing here looks an edge up by its endpoint labels, so a flip this kernel proposes may
    # legitimately create a second edge between two vertices some other edge already connects: the
    # two are simply two different halfedges and never collide. That is what lifts the limitation
    # ``remesh.intrinsic_delaunay``'s Notes used to describe -- see ``commit_intrinsic_flips`` for
    # the other half, maintaining ``twin`` across the flip this decides.
    h = wp.int32(wp.tid())
    out_flip[h] = False
    out_new_length[h] = 0.0
    # The three buffers ``commit_intrinsic_flips`` expects cleared are cleared here, before any
    # exit: they are exactly this grid's length (``out_count`` aside), nothing in this kernel reads
    # them, and the kernel that does is the next launch. ``out_face_claim`` cannot be: the claim
    # below ``atomic_min``s into it from other threads of this same launch, so it arrives reset --
    # by ``fixup_twin_remap`` of the previous iteration, and by the caller before the first.
    out_remap[h] = INT32_MAX_CONSTANT
    out_no_remap[h] = False
    if h == 0:
        out_count[0] = 0
    h1 = twin[h]
    if h1 < 0 or h1 <= h:
        return  # boundary, or the mirror of a lower-indexed canonical candidate

    f0 = h // 3
    e0 = h % 3
    f1 = h1 // 3
    e1 = h1 % 3
    apex0 = faces[f0 * 3 + e0]
    apex1 = faces[f1 * 3 + e1]
    u = faces[f0 * 3 + (e0 + 1) % 3]
    v = faces[f0 * 3 + (e0 + 2) % 3]
    quad = _resolve_flip_quad(faces, f0, u, v, apex0, apex1)
    first = quad[0]
    if first < wp.int32(0) or quad[1] == quad[3]:
        return
    second = quad[2]

    corner_f0_first = local_corner(faces, f0, first)
    corner_f0_second = local_corner(faces, f0, second)
    corner_f1_first = local_corner(faces, f1, first)
    corner_f1_second = local_corner(faces, f1, second)
    if corner_f0_first < 0 or corner_f0_second < 0:
        return
    if corner_f1_first < 0 or corner_f1_second < 0:
        return

    shared = edge_lengths[f0, e0]
    first_apex0 = edge_lengths[f0, corner_f0_second]
    second_apex0 = edge_lengths[f0, corner_f0_first]
    first_apex1 = edge_lengths[f1, corner_f1_second]
    second_apex1 = edge_lengths[f1, corner_f1_first]

    # The Delaunay test: the two angles facing the shared edge sum past a straight angle exactly
    # when the edge's cotangent weight would go negative.
    angle0 = law_of_cosines_angle(first_apex0, second_apex0, shared)
    angle1 = law_of_cosines_angle(first_apex1, second_apex1, shared)
    if angle0 + angle1 <= wp.PI:
        return

    # Unfold both triangles about the shared edge and measure the other diagonal. The wedge angles
    # at ``first`` add because the two triangles lie on opposite sides of the shared edge.
    wedge0 = law_of_cosines_angle(shared, first_apex0, second_apex0)
    wedge1 = law_of_cosines_angle(shared, first_apex1, second_apex1)
    total = wedge0 + wedge1
    flipped = (
        first_apex0 * first_apex0
        + first_apex1 * first_apex1
        - (2.0 * first_apex0 * first_apex1 * wp.cos(total))
    )
    if flipped <= TOLERANCE_ZERO_CONSTANT:
        return

    out_quad[h, 0] = quad[0]
    out_quad[h, 1] = quad[1]
    out_quad[h, 2] = quad[2]
    out_quad[h, 3] = quad[3]
    # The pre-flip twin of each of the quad's four *other* edges, captured now because
    # ``commit_intrinsic_flips`` must fix up both this face pair's own halfedges and each of these
    # neighbors' twin pointer, and by then it is looking at whichever candidate won its own claim,
    # not necessarily this one -- these are read once, while the mesh still agrees with ``quad``.
    out_neighbors[h, 0] = twin[f1 * 3 + corner_f1_second]  # edge (a, b), opposite c in f1
    out_neighbors[h, 1] = twin[f1 * 3 + corner_f1_first]  # edge (b, c), opposite a in f1
    out_neighbors[h, 2] = twin[f0 * 3 + corner_f0_first]  # edge (c, d), opposite a in f0
    out_neighbors[h, 3] = twin[f0 * 3 + corner_f0_second]  # edge (d, a), opposite c in f0
    out_new_length[h] = wp.sqrt(flipped)
    out_flip[h] = True
    # The claim: only the flip's own two faces -- unlike the vertex-pair-keyed flip loops' shared
    # ``claim_flips``, there is no new-edge hash to also claim, because two flips creating an edge
    # with the same endpoint labels are no longer a conflict at all (see above). The *other* four
    # faces a commit touches (each one's twin pointer, not its connectivity) are handled without a
    # lock, by ``fixup_twin_remap``. It rides this kernel because the decision has this one exit;
    # the vertex-pair-keyed engines decide at several, which is why theirs stays a launch.
    wp.atomic_min(out_face_claim, f0, h)
    wp.atomic_min(out_face_claim, f1, h)


@wp.func
def intrinsic_flip_claim_won(
    flip: wp.array[wp.bool], twin: wp.array[wp.int32], face_claim: wp.array[wp.int32], h: wp.int32
) -> tuple[wp.int32, wp.int32, wp.bool]:
    # ``flip_claim_won``'s counterpart for the halfedge-twin engine, minus the new-edge hash claim
    # that engine also checks -- see the claim at the end of ``intrinsic_delaunay_candidates``.
    if not flip[h]:
        return wp.int32(-1), wp.int32(-1), False
    f0 = h // 3
    f1 = twin[h] // 3
    if face_claim[f0] != h or face_claim[f1] != h:
        return f0, f1, False
    return f0, f1, True


@wp.kernel
def commit_intrinsic_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    face_claim: wp.array[wp.int32],
    new_length: wp.array[wp.float32],
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    twin: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_no_remap: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
) -> None:
    # Rewrites connectivity, edge lengths and this flip's own six halfedge slots in one launch, and
    # records how the *other* four -- the neighbors across the quad's non-diagonal edges -- are to
    # be fixed up, rather than writing into them directly.
    #
    # A neighbor across edge (d, a), say, may itself be winning an unrelated flip in this same
    # round: if it is, that flip's own commit is concurrently overwriting *its* three halfedge
    # slots, including the one this thread would otherwise read to learn "my new slot number" or
    # write to redirect it -- a genuine cross-thread race, not merely a stale value. So this thread
    # writes only into cells it exclusively owns (its own two faces' six slots, all locked by
    # ``intrinsic_flip_claim_won``): ``twin[f0 * 3 + 1] = neighbors[h, 3]`` places the *pre-round*
    # neighbor halfedge (captured back in ``intrinsic_delaunay_candidates``, before any commit in
    # this round ran) as a placeholder, and ``out_remap[old_slot] = new_slot`` -- keyed by this
    # face's own *old* slot number, also exclusively owned -- is what a neighbor reads, in
    # ``fixup_twin_remap``, to learn that old slot became this one. Two such placeholders can point
    # at each other (both across the same unaffected edge) and each is corrected by the *other*
    # side's remap entry, independent of whether either, both, or neither side actually flipped.
    h = wp.int32(wp.tid())
    f0, f1, won = intrinsic_flip_claim_won(flip, twin, face_claim, h)
    if not won:
        return

    a = quad[h, 0]
    b = quad[h, 1]
    c = quad[h, 2]
    d = quad[h, 3]
    diagonal = new_length[h]

    # ``faces`` still reads pre-flip here; the four other edges of the flip quad, read by vertex
    # label (and so by which corner still holds it, not by a fixed offset) so this does not depend
    # on which physical corner a vertex happens to occupy.
    corner_f0_a = local_corner(faces, f0, a)
    corner_f0_c = local_corner(faces, f0, c)
    corner_f1_a = local_corner(faces, f1, a)
    corner_f1_c = local_corner(faces, f1, c)
    first_apex0 = edge_lengths[f0, corner_f0_c]  # edge (d, a)
    second_apex0 = edge_lengths[f0, corner_f0_a]  # edge (c, d)
    first_apex1 = edge_lengths[f1, corner_f1_c]  # edge (a, b)
    second_apex1 = edge_lengths[f1, corner_f1_a]  # edge (b, c)
    old_h_da = f0 * 3 + corner_f0_c
    old_h_cd = f0 * 3 + corner_f0_a
    old_h_ab = f1 * 3 + corner_f1_c
    old_h_bc = f1 * 3 + corner_f1_a

    # The winding is ``write_flipped_quad``'s, shared with ``commit_flips``; what is intrinsic
    # here is the length table below it -- each new corner's opposite edge is either the new
    # diagonal or one of the four edges just measured above.
    write_flipped_quad(faces, f0, f1, a, b, c, d)

    edge_lengths[f0, 0] = diagonal
    edge_lengths[f0, 1] = first_apex0
    edge_lengths[f0, 2] = first_apex1
    edge_lengths[f1, 0] = diagonal
    edge_lengths[f1, 1] = second_apex1
    edge_lengths[f1, 2] = second_apex0

    h_f0_diagonal = f0 * 3 + 0
    h_f1_diagonal = f1 * 3 + 0
    h_f0_da = f0 * 3 + 1
    h_f0_ab = f0 * 3 + 2
    h_f1_bc = f1 * 3 + 1
    h_f1_cd = f1 * 3 + 2

    twin[h_f0_diagonal] = h_f1_diagonal
    twin[h_f1_diagonal] = h_f0_diagonal
    twin[h_f0_da] = neighbors[h, 3]  # placeholder: this edge's pre-round neighbor
    twin[h_f0_ab] = neighbors[h, 0]
    twin[h_f1_bc] = neighbors[h, 1]
    twin[h_f1_cd] = neighbors[h, 2]

    out_remap[old_h_da] = h_f0_da
    out_remap[old_h_cd] = h_f1_cd
    out_remap[old_h_ab] = h_f0_ab
    out_remap[old_h_bc] = h_f1_bc

    # Only the new diagonal's own two halfedges must never go through ``fixup_twin_remap``: they
    # already point at each other, freshly, and each other's index is also a face this same commit
    # owns -- so the *lookup* ``remap[twin[h_f0_diagonal]]`` reads ``remap[h_f1_diagonal]``, a slot
    # this same flip may separately have written for an unrelated carried-over edge (if the old
    # corner arithmetic happened to land there), which would silently misapply to the diagonal. The
    # four carried-over cells just written above (``h_f0_da`` etc.) are not exempted: each of their
    # placeholder values names a genuinely *different* face's row, so the same collision cannot
    # arise, and they need exactly the same fixup an untouched face's cell would if that other face
    # also flipped this round.
    out_no_remap[h_f0_diagonal] = True
    out_no_remap[h_f1_diagonal] = True
    wp.atomic_add(out_count, 0, 1)


@wp.kernel
def fixup_twin_remap(
    remap: wp.array[wp.int32],
    no_remap: wp.array[wp.bool],
    twin: wp.array[wp.int32],
    out_face_claim: wp.array[wp.int32],
) -> None:
    # The other half of ``commit_intrinsic_flips``'s deferred neighbor fixup: every halfedge except
    # this round's two brand-new diagonal cells (``no_remap``, set only there) asks whether its twin
    # pointer's target moved this round. For a halfedge whose own face did not flip, the pointer is
    # exactly what it was before the round started, so this is safe whether or not the *target* face
    # flipped: ``remap`` is keyed by each flipped face's own old slot, which is where the answer
    # lives if it flipped, and stays at the sentinel (leaving ``twin`` unchanged) if it did not.
    #
    # Also re-arms the next round's face-claim table (one entry per face, a third of this grid),
    # which nothing here reads: its reader, ``intrinsic_delaunay_candidates``, claims into it from
    # every thread and so cannot reset it itself.
    h = wp.int32(wp.tid())
    if h < out_face_claim.shape[0]:
        out_face_claim[h] = INT32_MAX_CONSTANT
    if no_remap[h]:
        return
    target = twin[h]
    if target < 0:
        return
    remapped = remap[target]
    if remapped != INT32_MAX_CONSTANT:
        twin[h] = remapped


@wp.func
def cluster_cell_key(
    vertex: wp.vec3, origin: wp.vec3, inverse_size: wp.float32, cell_bound: wp.uint64
) -> wp.uint64:
    # A vertex's voxel cell packed into the ``uint64`` key ``remesh.cluster_decimate`` orders its
    # clusters by: ``voxels.cell_indices`` and ``grouping.hash_indices_rows`` in one thread, the
    # same ``voxel_cell`` and the same packing, so the keys are theirs bit for bit. Pure in its
    # inputs, which is what lets ``cluster_insert_vertices`` recompute a stored vertex's key rather
    # than keep a table of keys.
    cell = voxel_cell(vertex, origin, inverse_size)
    return pack_index_triple(cell[0], cell[1], cell[2], cell_bound)


@wp.kernel
def cluster_insert_vertices(
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    cell_bound: wp.uint64,
    mask: wp.int32,
    out_cell_table: wp.array[wp.int32],
    out_vertex_cell: wp.array[wp.int32],
) -> None:
    # Each vertex's cluster, as the slot its cell key occupies in one open-addressing table of
    # vertex indices (``-1`` empty, caller-filled; the ``points.point_duplicate_first`` table with
    # a cell key in place of a position). A vertex claims an empty slot with ``atomic_cas``; an
    # occupied slot holds *some* vertex of its cell, whose key is recomputed to compare. Which
    # member ends up stored is arrival order and nothing reads it but for its key, so no
    # ``atomic_min`` is needed. The slots number the clusters arbitrarily: the output order is
    # fixed later by sorting the keys of the cells a face keeps (``cluster_mark_faces``).
    v = wp.int32(wp.tid())
    key = cluster_cell_key(vertices[v], origin, inverse_size, cell_bound)
    h = hash_slot(wp.int64(key), mask)
    while True:
        # Read before claiming: a cluster's vertices all probe its slot, and once it is taken a
        # plain load finds the occupant without the atomic (a stale ``-1`` only costs the CAS).
        prev = out_cell_table[h]
        if prev == wp.int32(-1):
            prev = wp.atomic_cas(out_cell_table, h, wp.int32(-1), v)
            if prev == wp.int32(-1):
                break
        if cluster_cell_key(vertices[prev], origin, inverse_size, cell_bound) == key:
            break
        h = next_slot(h, mask)
    out_vertex_cell[v] = h


@wp.func
def sorted_cluster_triple(
    faces: wp.array[wp.int32], vertex_cell: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.bool]:
    # Face ``f``'s three clusters in ascending order and whether they are distinct: the face's
    # orientation-free identity after welding, which ``cluster_mark_faces`` compares exactly.
    i0, i1, i2, distinct = remapped_corner_triple(faces, vertex_cell, f)
    s0, s1, s2 = sort3(i0, i1, i2)
    return s0, s1, s2, distinct


@wp.func
def claim_kept_cluster(
    cell: wp.int32,
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    cell_bound: wp.uint64,
    cell_table: wp.array[wp.int32],
    cell_state: wp.array[wp.int32],
    counters: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_cells: wp.array[wp.int32],
) -> None:
    # The first surviving face to name a cluster moves it off ``-1`` and appends its cell key and
    # slot at a cursor. The cursor order is arrival order; the sort that follows is what orders
    # them, and the keys are distinct, so the result does not depend on it.
    # A plain read first: every surviving face names three clusters and most were claimed long
    # ago, so the atomic is only for the few threads that still see ``-1``.
    if cell_state[cell] != wp.int32(-1):
        return
    if wp.atomic_exch(cell_state, cell, wp.int32(0)) == wp.int32(-1):
        p = wp.atomic_add(counters, 0, wp.int32(1))
        out_keys[p] = cluster_cell_key(vertices[cell_table[cell]], origin, inverse_size, cell_bound)
        out_cells[p] = cell


@wp.kernel
def cluster_mark_faces(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    cell_bound: wp.uint64,
    cell_table: wp.array[wp.int32],
    face_mask: wp.int32,
    out_cell_state: wp.array[wp.int32],
    out_face_table: wp.array[wp.int32],
    out_counters: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_cells: wp.array[wp.int32],
    out_face_slots: wp.array[wp.int32],
) -> None:
    # Everything ``cluster_decimate`` must count before it can size its outputs, in one pass over
    # the faces: which faces survive the weld (three distinct clusters), which clusters a
    # surviving face keeps (``claim_kept_cluster``), and which surviving faces are duplicates of
    # one another. The last is a second ``point_duplicate_first``-style table, of face indices
    # keyed on the sorted cluster triple and compared *exactly* by recomputing the stored face's
    # triple, lowered to the class's smallest face index with ``atomic_min`` -- the first
    # occurrence, whose winding the output keeps. A face that claims an empty slot appends it at
    # the second cursor. ``out_cell_state``, ``out_face_table`` and ``out_counters`` arrive
    # initialised by the caller (``-1``, ``-1``, ``0``) and leave holding the answer.
    f = wp.int32(wp.tid())
    s0, s1, s2, distinct = sorted_cluster_triple(faces, vertex_cell, f)
    if not distinct:
        return
    claim_kept_cluster(
        s0,
        vertices,
        origin,
        inverse_size,
        cell_bound,
        cell_table,
        out_cell_state,
        out_counters,
        out_keys,
        out_cells,
    )
    claim_kept_cluster(
        s1,
        vertices,
        origin,
        inverse_size,
        cell_bound,
        cell_table,
        out_cell_state,
        out_counters,
        out_keys,
        out_cells,
    )
    claim_kept_cluster(
        s2,
        vertices,
        origin,
        inverse_size,
        cell_bound,
        cell_table,
        out_cell_state,
        out_counters,
        out_keys,
        out_cells,
    )
    h = hash_slot_words3(s1, s0, s2, face_mask)
    while True:
        # Read before claiming, as for the clusters: a welded face is usually one of several
        # duplicates, and all of them probe the same slot. A stale read is safe both ways -- a
        # stale ``-1`` falls through to the ``atomic_cas``, which returns the real occupant, and a
        # stale occupant is still a member of its class, only possibly not its smallest.
        prev = out_face_table[h]
        if prev == wp.int32(-1):
            prev = wp.atomic_cas(out_face_table, h, wp.int32(-1), f)
            if prev == wp.int32(-1):
                out_face_slots[wp.atomic_add(out_counters, 1, wp.int32(1))] = h
                break
        t0, t1, t2, _distinct = sorted_cluster_triple(faces, vertex_cell, prev)
        if t0 == s0 and t1 == s1 and t2 == s2:
            if f < prev:
                wp.atomic_min(out_face_table, h, f)
            break
        h = next_slot(h, face_mask)


@wp.kernel
def cluster_rank_cells(sorted_cells: wp.array[wp.int32], out_cell_rank: wp.array[wp.int32]) -> None:
    # A kept cluster's output vertex index is its position in cell-key order, written over the
    # claim flag ``cluster_mark_faces`` left in the same slot (which has no reader left); an
    # unreferenced cluster keeps ``-1``.
    i = wp.int32(wp.tid())
    out_cell_rank[sorted_cells[i]] = i


@wp.func
def vertex_cluster_rank(
    vertex_cell: wp.array[wp.int32], cell_rank: wp.array[wp.int32], v: wp.int32
) -> wp.int32:
    # Vertex ``v``'s output vertex, or ``-1`` when no surviving face keeps its cluster.
    return cell_rank[vertex_cell[v]]


@wp.kernel
def cluster_accumulate(
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_sum: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # Sum and count of every kept cluster's vertices, indexed by output vertex. A cluster no face
    # keeps has no output slot and contributes nothing.
    v = wp.int32(wp.tid())
    r = vertex_cluster_rank(vertex_cell, cell_rank, v)
    if r >= 0:
        wp.atomic_add(out_sum, r, vertices[v])
        wp.atomic_add(out_count, r, 1)


@wp.func
def mean_from_sum(total: wp.vec3, count: wp.int32) -> wp.vec3:
    # A cluster's mean position from ``cluster_accumulate``'s sum and count, dividing by the count
    # converted in place rather than by a separately materialised ``float32`` copy of the counts.
    return total / wp.float32(count)


@wp.kernel
def cluster_means(
    sums: wp.array[wp.vec3], counts: wp.array[wp.int32], out_vertices: wp.array[wp.vec3]
) -> None:
    # Each output vertex's mean. Not in place over ``sums``: that would save one allocation at the
    # price of an in-place allowlist entry for the naming check.
    r = wp.int32(wp.tid())
    out_vertices[r] = mean_from_sum(sums[r], counts[r])


@wp.kernel
def cluster_pick_closest(
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    voxel_size: wp.float32,
    out_closest: wp.array[wp.uint64],
) -> None:
    # The "closest to the cell centre" representative of every kept cluster, in one ``uint64``
    # ``atomic_min``: the squared distance's bits above the vertex index. A non-negative
    # ``float32`` orders as its bit pattern does, so the minimum is the winning distance and, among
    # the vertices tying for it, the lowest index -- a deterministic pick whatever the arrival
    # order. ``out_closest`` arrives holding ``UINT64_MAX``.
    v = wp.int32(wp.tid())
    r = vertex_cluster_rank(vertex_cell, cell_rank, v)
    if r >= 0:
        d = squared_distance_to_own_cell_center(vertices[v], origin, voxel_size)
        key = (wp.uint64(wp.cast(d, wp.uint32)) << wp.uint64(32)) | wp.uint64(wp.uint32(v))
        wp.atomic_min(out_closest, r, key)


@wp.kernel
def cluster_gather_representatives(
    vertices: wp.array[wp.vec3], closest: wp.array[wp.uint64], out_vertices: wp.array[wp.vec3]
) -> None:
    # ``cluster_means``' sibling for the closest-to-centre contraction: the representative
    # vertex's own position, its index the low half of ``cluster_pick_closest``'s key.
    r = wp.int32(wp.tid())
    out_vertices[r] = vertices[wp.int32(closest[r] & wp.uint64(0xFFFFFFFF))]


@wp.func
def ranked_face_corners(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    f: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # Face ``f``'s corners as output vertices, in its own winding.
    a, b, c = corner_triple(faces, f)
    return (
        vertex_cluster_rank(vertex_cell, cell_rank, a),
        vertex_cluster_rank(vertex_cell, cell_rank, b),
        vertex_cluster_rank(vertex_cell, cell_rank, c),
    )


@wp.kernel
def cluster_face_keys(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    face_table: wp.array[wp.int32],
    face_slots: wp.array[wp.int32],
    n_kept: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_faces: wp.array[wp.int32],
) -> None:
    # One unique face per thread: its class representative (the smallest face index,
    # ``cluster_mark_faces``) and the key ``grouping.unique_faces`` orders faces by -- the sorted
    # output-vertex triple packed in radix ``n_kept`` -- so the sort that follows reproduces that
    # function's output order.
    i = wp.int32(wp.tid())
    g = face_table[face_slots[i]]
    r0, r1, r2 = ranked_face_corners(faces, vertex_cell, cell_rank, g)
    s0, s1, s2 = sort3(r0, r1, r2)
    out_keys[i] = pack_index_triple(s0, s1, s2, n_kept)
    out_faces[i] = g


@wp.kernel
def cluster_emit_faces(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    sorted_faces: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The unique faces in key order, each in its representative's winding.
    j = wp.int32(wp.tid())
    r0, r1, r2 = ranked_face_corners(faces, vertex_cell, cell_rank, sorted_faces[j])
    write_corner_triple(out_faces, j, r0, r1, r2)


@wp.func
def mark_surviving_face(
    faces: wp.array[wp.int32], remap: wp.array[wp.int32], f: wp.int32, flags: wp.array[wp.int32]
) -> tuple[wp.int32, wp.int32, wp.int32, wp.bool]:
    # Face ``f``'s corners through ``remap`` and whether it survives, recorded as the ``int32``
    # flags one scan turns into both compactions: face ``f``'s own at ``f``, and a mark at
    # ``n_faces + t`` on every target a surviving face names (plain stores -- every writer stores
    # the same value -- over caller-zeroed marks). The face flag is written whatever its value.
    i0, i1, i2, distinct = remapped_corner_triple(faces, remap, f)
    flags[f] = wp.where(distinct, wp.int32(1), wp.int32(0))
    if distinct:
        mark_corners(flags, faces.shape[0] // 3, i0, i1, i2, wp.int32(1))
    return i0, i1, i2, distinct


@wp.kernel
def mark_surviving_faces(
    faces: wp.array[wp.int32], remap: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # Which faces survive a vertex remap (vertex clustering's cell labels, an edge collapse's
    # survivor map) and which targets they still name, so both compactions are one scan of
    # ``out_flags`` rather than a face gather and a sort-based dedup of the corners. The remapped
    # corners are not stored: ``compact_surviving_faces`` re-reads them through ``remap`` for the
    # faces that survive. ``remap_and_mark_faces`` is the same marking with the corners written.
    mark_surviving_face(faces, remap, wp.int32(wp.tid()), out_flags)


@wp.kernel
def compact_surviving_faces(
    faces: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_totals: wp.array[wp.int32],
) -> None:
    # Launched over every input face. ``face_offsets`` and ``ranks`` are the two halves of one
    # inclusive scan over ``mark_surviving_faces``' face flags followed by its target marks, behind
    # a leading zero, each read as an exclusive scan with its total at the end (``scanned_slot``).
    # A surviving face is written at its dense row, so the rows keep the input's face order -- what
    # a face-mask ``submesh`` of the remapped faces returns -- with every corner renumbered onto
    # the compacted targets, which keep their order too. The last thread publishes both totals
    # (kept targets, surviving faces) side by side, so the host reads them in one copy.
    f = wp.int32(wp.tid())
    n_faces = face_offsets.shape[0] - 1
    if f == n_faces - 1:
        out_totals[0] = ranks[ranks.shape[0] - 1] - ranks[0]
        out_totals[1] = face_offsets[n_faces]
    row, survived = scanned_slot(face_offsets, f)
    if not survived:
        return
    i0, i1, i2, _distinct = remapped_corner_triple(faces, remap, f)
    s0, _k0 = scanned_slot(ranks, i0)
    s1, _k1 = scanned_slot(ranks, i1)
    s2, _k2 = scanned_slot(ranks, i2)
    write_corner_triple(out_faces, row, s0, s1, s2)


@wp.kernel
def compact_collapse_vertices(
    positions: wp.array[wp.vec3],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    ranks: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_low: wp.array[wp.float32],
    out_high: wp.array[wp.float32],
) -> None:
    # A collapse pass's surviving vertices and their two length bands, written into their compacted
    # slots in one pass. ``ranks`` is ``compact_surviving_faces``' scan of the referenced marks, so
    # a vertex is kept exactly where it steps -- the order ``repair.remove_unreferenced_vertices``
    # keeps and gathering the bands by its inverse reproduces.
    v = wp.int32(wp.tid())
    slot, kept = scanned_slot(ranks, v)
    if not kept:
        return
    out_positions[slot] = positions[v]
    out_low[slot] = low[v]
    out_high[slot] = high[v]


# Objective for ``objective_flip_candidates``. A warp-uniform kernel argument rather than a
# ``wp.Function``, so both predicates share one compiled module (AGENTS.md section 2.7).
OBJECTIVE_PLANARITY = wp.constant(wp.int32(0))  # improve triangle shape on a near-planar quad
OBJECTIVE_CURVATURE = wp.constant(wp.int32(1))  # pick whichever diagonal bends the surface less
OBJECTIVE_T_VERTEX = wp.constant(wp.int32(2))  # break up a sliver whose apex sits on the far edge

# Relative margin a flip must beat the current diagonal by. Without it a quad whose two diagonals
# score equally (every quad of a regular grid) flips back and forth forever, one pass each way.
OBJECTIVE_EPS = wp.constant(wp.float32(1e-6))


@wp.kernel
def objective_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    edge_set: wp.array[wp.uint64],
    edge_set_mask: wp.int32,
    key_base: wp.uint64,
    objective: wp.int32,
    metric: wp.int32,
    planar_cos: wp.float32,
    aspect_threshold: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    # Quad convention (shared with ``delone_flip_candidates``): the current diagonal is a-c, with
    # faces (a, b, c) and (a, c, d); the flip replaces it with b-d, giving (a, b, d) and (d, b, c).
    #
    # Fourteen arguments and **deliberately not bundled into a ``@wp.struct``**, unlike
    # ``holes.stitch_dp_diag`` and ``holes.fill_dp_span``, which have the same width. Those launch
    # hundreds of times with nothing else in the loop; this one launches once per *flip round*, and
    # a round rebuilds the whole face adjacency around it, so a bundle would save a fraction of a
    # percent. The width alone is not the criterion; the launch count around it is.
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    a, b, c, d = _resolve_flip_quad_in_region(
        faces,
        adjacency,
        adjacency_edges,
        unshared,
        region_flags,
        sorted_edge_keys,
        edge_set,
        edge_set_mask,
        key_base,
        k,
        out_quad,
    )
    if a < 0:
        return
    ap = vertices[a]
    bp = vertices[b]
    cp = vertices[c]
    dp = vertices[d]

    # A non-convex quad has no valid flip: the new diagonal would fall outside it. The rest of this
    # kernel stays in float32 -- only the convexity branch needs the promoted corners, which is what
    # ``flip_quad_positions_d`` names.
    apd, bpd, cpd, dpd = flip_quad_positions_d(vertices, a, b, c, d)
    if not is_unfold_quadrangle_convex(apd, bpd, cpd, dpd):
        return

    if objective == OBJECTIVE_T_VERTEX:
        # A T-vertex shows up as a sliver: one apex sits (nearly) on the opposite edge, which drives
        # the circumradius-to-inradius ratio through the roof. Flip only when the sliver is *that*
        # bad and the flip actually improves it, so a merely thin triangle is left alone.
        old_worst = wp.max(triangle_aspect_ratio(ap, bp, cp), triangle_aspect_ratio(ap, cp, dp))
        if not (old_worst > aspect_threshold):  # also excludes a NaN ratio
            return
        new_worst = wp.max(triangle_aspect_ratio(ap, bp, dp), triangle_aspect_ratio(dp, bp, cp))
        out_flip[k] = new_worst < old_worst
        return

    normal_abc = triangle_normal(ap, bp, cp)
    normal_acd = triangle_normal(ap, cp, dp)
    normal_abd = triangle_normal(ap, bp, dp)
    normal_dbc = triangle_normal(dp, bp, cp)

    if objective == OBJECTIVE_PLANARITY:
        # Only rewrite a quad that is flat enough for the rewrite not to change the surface. The
        # gate is on the *cosine* of the dihedral so the kernel needs no inverse trigonometry.
        if wp.dot(normal_abc, normal_acd) < planar_cos:
            return
        old_score = wp.min(
            triangle_quality(ap, bp, cp, metric), triangle_quality(ap, cp, dp, metric)
        )
        new_score = wp.min(
            triangle_quality(ap, bp, dp, metric), triangle_quality(dp, bp, cp, metric)
        )
        out_flip[k] = new_score > old_score * (1.0 + OBJECTIVE_EPS)
        return

    # Curvature: keep whichever diagonal leaves the two triangles closer to coplanar. Unlike the
    # planarity objective this deliberately *does* change the surface -- that is the point.
    old_bend = wp.abs(dihedral_angle(normal_abc, normal_acd, cp - ap))
    new_bend = wp.abs(dihedral_angle(normal_abd, normal_dbc, dp - bp))
    out_flip[k] = new_bend < old_bend * (1.0 - OBJECTIVE_EPS)


# ---------------------------------------------------------------------------
# Quadric error metric (Garland-Heckbert) decimation
# ---------------------------------------------------------------------------

# Quadrics are accumulated in float64. That is not caution: the entries are sums of ``area * d^2``
# with ``d`` an absolute plane offset, so on a mesh whose coordinates are far from the origin they
# span many orders of magnitude and a float32 accumulation loses the small ones -- which are exactly
# the terms that distinguish two candidate collapses. libigl and MeshLab both use double here.

# Below this determinant (relative to the quadric's own scale) the 3x3 system is treated as singular
# and the optimum falls back to the edge midpoint: a planar neighbourhood has a whole plane of
# equally good positions and picking one by inversion amplifies noise.
QUADRIC_SINGULAR_EPS = wp.constant(wp.float64(1e-12))

# A collapse is rejected when it would turn an incident face's normal by more than this. Zero would
# allow a face to become exactly degenerate; 0.2 (~78 degrees) still permits real simplification of
# a curved region while refusing an outright fold.
#
# **Not exposed as a keyword.** A veto census over decimations that stopped short of their target
# has the *feature* rule vetoing every remaining edge in most cases and this one vetoing none, so
# the knob a caller short of their target needs is ``feature_angle``, which already exists; a second
# one here would be a keyword for a constraint measured not to bind (section 4.2). Re-run that
# census before widening this value -- it is what keeps the output free of inverted,
# self-intersecting triangles, and nothing in the suite would notice it going soft.
COLLAPSE_MIN_NORMAL_DOT = wp.constant(wp.float32(0.2))


@wp.func
def plane_quadric(normal: wp.vec3d, offset: wp.float64, weight: wp.float64) -> wp.mat44d:
    # Garland-Heckbert fundamental quadric of the plane ``dot(normal, x) + offset = 0``, scaled by
    # ``weight``. Laid out so that ``[p, 1]^T Q [p, 1]`` is the weighted squared distance to the
    # plane: the leading 3x3 block is ``n n^T``, the last row and column are ``offset * n``, and the
    # corner is ``offset^2``.
    a = weight * normal[0]
    b = weight * normal[1]
    c = weight * normal[2]
    d = weight * offset
    return wp.mat44d(
        a * normal[0],
        a * normal[1],
        a * normal[2],
        a * offset,
        b * normal[0],
        b * normal[1],
        b * normal[2],
        b * offset,
        c * normal[0],
        c * normal[1],
        c * normal[2],
        c * offset,
        d * normal[0],
        d * normal[1],
        d * normal[2],
        d * offset,
    )


@wp.func
def quadric_error(quadric: wp.mat44d, p: wp.vec3d) -> wp.float64:
    # ``[p, 1]^T Q [p, 1]``: the accumulated squared distance from ``p`` to every plane folded into
    # ``Q``. Clamped at zero, since a float64 sum of positive-semidefinite terms can still land a
    # hair below it and a negative "error" would sort ahead of every real candidate.
    homogeneous = wp.vec4d(p[0], p[1], p[2], wp.float64(1.0))
    return wp.max(wp.float64(0.0), wp.dot(homogeneous, quadric * homogeneous))


@wp.func
def quadric_optimum(quadric: wp.mat44d, fallback: wp.vec3d) -> wp.vec3d:
    # Position minimizing the quadric: solve ``A p = -b`` for the leading 3x3 block ``A`` and the
    # last column ``b``. ``fallback`` (the edge midpoint) is returned when ``A`` is singular
    # relative to its own scale, which is the planar case -- there the minimum is a whole plane and
    # inverting a near-singular matrix would place the vertex arbitrarily far away.
    a = wp.mat33d(
        quadric[0, 0],
        quadric[0, 1],
        quadric[0, 2],
        quadric[1, 0],
        quadric[1, 1],
        quadric[1, 2],
        quadric[2, 0],
        quadric[2, 1],
        quadric[2, 2],
    )
    scale = wp.abs(quadric[0, 0]) + wp.abs(quadric[1, 1]) + wp.abs(quadric[2, 2])
    if scale <= wp.float64(0.0):
        return fallback
    if wp.abs(wp.determinant(a)) <= QUADRIC_SINGULAR_EPS * scale * scale * scale:
        return fallback
    b = wp.vec3d(quadric[0, 3], quadric[1, 3], quadric[2, 3])
    return -(wp.inverse(a) * b)


@wp.func
def add_face_quadric(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: wp.int32,
    out_quadrics: wp.array[wp.mat44d],
) -> None:
    # Area-weighted plane quadric of face ``f``, scattered onto its three corners. Area weighting is
    # Garland-Heckbert's: a large triangle constrains its vertices more than a sliver does.
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    cross = wp.cross(v1 - v0, v2 - v0)
    double_area = wp.length(cross)
    if double_area <= wp.float64(0.0):
        return
    normal = cross / double_area
    quadric = plane_quadric(normal, -wp.dot(normal, v0), double_area * wp.float64(0.5))
    add_corner_triple(out_quadrics, faces, f, quadric, quadric, quadric)


# One decimation pass is replayed as a CUDA graph, and a replay pays a few microseconds of device
# time per graph *node* on top of the node's own work, so the pass is written as few launches as
# its data dependencies allow: every kernel below does all the work that can share its grid
# barrier. The sequence, with the grid barrier each boundary stands for:
#
#   begin_decimation_pass -> sort corner keys -> mark_unique_edge_starts -> scan ->
#   emit_pass_edges -> count_pass_edges -> scan -> scatter_pass_adjacency ->
#   quadric_collapse_candidates -> sort costs -> drop_past_half ->
#   [drop_locked_and_claim -> mark_collapse_winners -> scan -> commit_budgeted_collapses ->
#    end_collapse_round]* -> remap_and_mark_faces -> scan -> compact_decimation_pass
#
# **The per-pass adjacency is one buffer.** The vertex-vertex CSR and the vertex-face CSR are both
# "count per vertex, scan, scatter", so their counts sit side by side in one array (vertex-vertex
# rows first, then vertex-face rows at ``vertex_capacity``) and one exclusive scan yields both
# offset tables and the two payloads' positions in one shared buffer. Neither CSR's row order is
# sorted -- a row is filled in atomic arrival order -- and nothing reads it in order: the link
# condition counts, the lock and the claim take unions and minima, and the flip veto is an any.


@wp.kernel
def begin_decimation_pass(
    faces: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    state: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_edge_face_count: wp.array[wp.int32],
    out_adjacency_counts: wp.array[wp.int32],
    out_feature_count: wp.array[wp.int32],
    out_quadrics: wp.array[wp.mat44d],
    out_locked: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_vertex_flags: wp.array[wp.int32],
    out_round_state: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Launched over the widest buffer it initializes; each write is guarded by its own length.
    #
    # The corner keys are ``adjacency.face_edge_keys`` over the fixed-capacity face buffer: three
    # keys per live face, and a maximal sentinel for each padded one, which sorts past every real
    # key so bounding the grouping by ``3 * n_faces`` excludes the padding exactly. The identity
    # payload is what makes the radix sort an argsort (``array.sort_pair_indices``).
    #
    # Everything else is the state the rest of the pass accumulates into or starts from, reset here
    # rather than by one memset or copy each: the edge incidence and adjacency counts, the feature
    # counts and quadrics, the round loop's locks, claim keys and collapse map, the working
    # positions, the compaction's vertex marks, and the loop state (``end_collapse_round``'s slot
    # table, seeded with its condition at 1 because ``wp.capture_while`` tests it before the first
    # round) and the pass's commit count.
    t = wp.int32(wp.tid())
    if t < faces.shape[0] // 3:
        c = t * 3
        if t >= state[DECIMATION_FACES]:
            out_keys[c + 0] = EDGE_KEY_PAD
            out_keys[c + 1] = EDGE_KEY_PAD
            out_keys[c + 2] = EDGE_KEY_PAD
        else:
            write_face_edge_keys(faces, t, c, base, out_keys)
        out_order[c + 0] = c + 0
        out_order[c + 1] = c + 1
        out_order[c + 2] = c + 2
    if t < out_edge_face_count.shape[0]:
        out_edge_face_count[t] = 0
    if t < out_adjacency_counts.shape[0]:
        out_adjacency_counts[t] = 0
    if t < out_remap.shape[0]:
        out_feature_count[t] = 0
        out_quadrics[t] = wp.mat44d()
        out_locked[t] = 0
        out_min_key[t] = UNCLAIMED_KEY
        out_remap[t] = t
        out_positions[t] = vertices[t]
    if t < out_vertex_flags.shape[0]:
        out_vertex_flags[t] = 0
    if t == 0:
        out_round_state[LOOP_ROUND] = 0
        out_round_state[LOOP_CONDITION] = 1
        out_round_state[COLLAPSE_COMMITS] = 0
        out_state[DECIMATION_COMMITS] = 0


# The pass's exclusive scans (``remesh._ExclusiveScan``). ``wp.utils.array_scan`` allocates its
# scratch on every call, which a ``wp.capture_while`` body may not do -- and the round loop ranks
# its winners with a scan -- and which puts a memory node into any other captured graph. So the
# pass carries its own two-launch scan over preallocated buffers: each block scans one
# ``SCAN_CHUNK`` of the input through a loaded tile and publishes the chunk's total, and one block
# scans those totals. A reader adds the two (``scanned_prefix``). ``wp.tile_load`` and
# ``wp.tile_store`` are lane-independent and bounds-checked, so the chunk kernels are correct on
# the CPU device, where a tiled launch runs one lane, and on a ragged last chunk.
#
# 512 x 256 lanes is the best of a sweep over chunks of 256 to 4096 and 128 to 512 lanes: 1.05x
# over 2048 on a 35k-face decimation pass, flat on 870k faces, where 512 lanes lost 1.18x.
SCAN_CHUNK = wp.constant(512)
SCAN_BLOCK_DIM = 256


@wp.func
def popcount32(x: wp.uint32) -> wp.int32:
    # Set bits of ``x`` (the SWAR reduction; Warp has no population-count builtin).
    v = x - ((x >> wp.uint32(1)) & wp.uint32(0x55555555))
    v = (v & wp.uint32(0x33333333)) + ((v >> wp.uint32(2)) & wp.uint32(0x33333333))
    v = (v + (v >> wp.uint32(4))) & wp.uint32(0x0F0F0F0F)
    return wp.int32((v * wp.uint32(0x01010101)) >> wp.uint32(24))


@wp.kernel
def scan_chunks_exclusive(
    values: wp.array[wp.int32], out_prefix: wp.array[wp.int32], out_chunk_totals: wp.array[wp.int32]
) -> None:
    chunk, _lane = wp.tid()
    counts = wp.tile_load(values, shape=SCAN_CHUNK, offset=chunk * SCAN_CHUNK)
    wp.tile_store(out_prefix, wp.tile_scan_exclusive(counts), offset=chunk * SCAN_CHUNK)
    wp.tile_store(out_chunk_totals, wp.tile_sum(counts), offset=chunk)


@wp.kernel
def scan_word_counts_exclusive(
    words: wp.array[wp.uint32], out_prefix: wp.array[wp.int32], out_chunk_totals: wp.array[wp.int32]
) -> None:
    # ``scan_chunks_exclusive`` over the set-bit counts of a bitmask's words. The two differ only in
    # the ``popcount32`` applied to the loaded tile.
    chunk, _lane = wp.tid()
    counts = wp.tile_map(
        popcount32, wp.tile_load(words, shape=SCAN_CHUNK, offset=chunk * SCAN_CHUNK)
    )
    wp.tile_store(out_prefix, wp.tile_scan_exclusive(counts), offset=chunk * SCAN_CHUNK)
    wp.tile_store(out_chunk_totals, wp.tile_sum(counts), offset=chunk)


@wp.kernel
def scan_chunk_totals_exclusive(
    chunk_totals: wp.array[wp.int32], out_chunk_offsets: wp.array[wp.int32]
) -> None:
    # One block over every chunk total. Each lane owns a contiguous run sized by
    # ``wp.block_dim()``, so on the CPU device the one lane walks them all and the tile scan over
    # its single element is its own zero.
    _block, lane = wp.tid()
    lanes = wp.block_dim()
    n = chunk_totals.shape[0]
    per = (n + lanes - 1) // lanes
    lo = wp.min(lane * per, n)
    hi = wp.min(lo + per, n)
    total = wp.int32(0)
    for i in range(lo, hi):
        total += chunk_totals[i]
    running = wp.untile(wp.tile_scan_exclusive(wp.tile(total)))
    for i in range(lo, hi):
        out_chunk_offsets[i] = running
        running += chunk_totals[i]


@wp.func
def scanned_prefix(
    prefix: wp.array[wp.int32], chunk_offsets: wp.array[wp.int32], i: wp.int32
) -> wp.int32:
    # Entry ``i`` of the exclusive scan the chunk kernels above computed together.
    return prefix[i] + chunk_offsets[i // SCAN_CHUNK]


@wp.func
def set_bits_before(
    words: wp.array[wp.uint32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    base: wp.int32,
    p: wp.int32,
) -> wp.int32:
    # Set bits ahead of bit ``p`` of the bitmask starting at word ``base``, counting from the start
    # of ``words``: the word scan to that word, plus the word's own bits below ``p``.
    w = base + (p >> 5)
    below = (wp.uint32(1) << wp.uint32(p & 31)) - wp.uint32(1)
    return scanned_prefix(prefix, chunk_offsets, w) + popcount32(words[w] & below)


@wp.kernel
def mark_unique_edge_starts(
    sorted_keys: wp.array[wp.uint64], state: wp.array[wp.int32], out_starts: wp.array[wp.int32]
) -> None:
    # Flag the first position of every run of equal keys among the live corners -- what
    # ``grouping.unique_1d`` answers with a hash table, over sorted keys instead, and emitting
    # ``int32`` so the scan that follows needs no cast. See ``mark_edge_pair_starts`` above for the
    # one condition the two differ by.
    #
    # Not folded into the scan's chunk kernel (``scan_chunks_exclusive``) although the flags are an
    # elementwise map of the keys: that kernel loads its tile with ``wp.tile_load``, which is
    # lane-independent, and forming the flags in it would build the tile from per-lane values,
    # which collapses to one lane per block on the CPU device (CLAUDE.md section 12.2). The saving
    # is one replayed node a pass, about half a percent of a decimation.
    i = wp.int32(wp.tid())
    start = wp.int32(0)
    if i < state[DECIMATION_FACES] * 3 and sorted_run_start(sorted_keys, i):
        start = wp.int32(1)
    out_starts[i] = start


@wp.kernel
def emit_pass_edges(
    faces: wp.array[wp.int32],
    order: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    start_prefix: wp.array[wp.int32],
    start_chunk_offsets: wp.array[wp.int32],
    state: wp.array[wp.int32],
    edge_capacity: wp.int32,
    vertex_capacity: wp.int32,
    out_unique_edges: wp.array2d[wp.int32],
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
    out_adjacency_counts: wp.array[wp.int32],
    out_corner_slots: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # One launch per sorted live corner for everything the corner itself decides: the ascending
    # unique edge rows (the same ascending-key order ``unique_1d`` produces, which is what keeps the
    # edge numbering -- and so ``scramble_index``'s lock keys -- identical to the composed path),
    # the edge's incident faces (``scatter.scatter_edge_incidence``, reached without the corner ->
    # edge map it reads), the corner's slot in its vertex's face list, and, from the last live
    # position, the live edge count. The exclusive scan of ``starts`` at ``i`` is the unique index
    # of the key at sorted position ``i`` once that position has started its run.
    #
    # Padded corners are skipped rather than sent to a dummy slot: every one of them names the same
    # dummy vertex and edge, so counting them would put thousands of atomics on one address.
    i = wp.int32(wp.tid())
    live = state[DECIMATION_FACES] * 3
    if i >= live:
        return
    corner = order[i]
    # The capacity is a bound the *previous* pass measured, and a collapse removes at least three
    # undirected edges and adds none, so an overflow cannot fire. Handled anyway: an overflowing
    # corner goes to the dummy slot one past the capacity, in the incidence count as in the rows.
    run_starts = scanned_prefix(start_prefix, start_chunk_offsets, i) + starts[i]
    e = wp.min(run_starts - 1, edge_capacity)
    if starts[i] != 0 and e < edge_capacity:
        write_edge_row(faces, corner, e, out_unique_edges)
    record_edge_incidence(e, corner, out_edge_face_count, out_edge_faces)
    out_corner_slots[corner] = wp.atomic_add(
        out_adjacency_counts, vertex_capacity + faces[corner], 1
    )
    if i + 1 == live:
        out_state[DECIMATION_EDGES] = wp.min(run_starts, edge_capacity)


@wp.kernel
def count_pass_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    feature_angle: wp.float32,
    out_feature_count: wp.array[wp.int32],
    out_adjacency_counts: wp.array[wp.int32],
    out_edge_slots: wp.array[wp.vec2i],
) -> None:
    # Per live unique edge, once the incidence is complete: its feature vote (``_classify``'s
    # count, ``is_feature_edge``) and its two entries' slots in the vertex-vertex CSR rows. A
    # degenerate edge ``(a, a)`` from a face with a repeated corner is one entry in row ``a``, as
    # the triplet build this replaces accumulated its two identical triplets into one.
    e = wp.int32(wp.tid())
    if e >= state[DECIMATION_EDGES]:
        return
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    if is_feature_edge(vertices, faces, edge_face_count, edge_faces, e, feature_angle):
        wp.atomic_add(out_feature_count, u, 1)
        wp.atomic_add(out_feature_count, v, 1)
    slot_u = wp.atomic_add(out_adjacency_counts, u, 1)
    slot_v = wp.int32(-1)
    if u != v:
        slot_v = wp.atomic_add(out_adjacency_counts, v, 1)
    out_edge_slots[e] = wp.vec2i(slot_u, slot_v)


@wp.kernel
def scatter_pass_adjacency(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_slots: wp.array[wp.vec2i],
    corner_slots: wp.array[wp.int32],
    count_prefix: wp.array[wp.int32],
    count_chunk_offsets: wp.array[wp.int32],
    state: wp.array[wp.int32],
    vertex_capacity: wp.int32,
    out_adjacency: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_quadrics: wp.array[wp.mat44d],
) -> None:
    # Launched over the corners or the offset table, whichever is longer. Thread ``t`` places live
    # edge ``t``'s two vertex-vertex entries and live corner ``t``'s vertex-face entry at the slots
    # the counting kernels drew, writes entry ``t`` of the offset table the scan left split in two,
    # and adds live face ``t``'s quadric -- independent work that only needs the counts and the
    # zeroed quadrics, so it shares this one launch.
    t = wp.int32(wp.tid())
    if t < out_offsets.shape[0]:
        out_offsets[t] = scanned_prefix(count_prefix, count_chunk_offsets, t)
    if t < state[DECIMATION_EDGES]:
        u = unique_edges[t, 0]
        v = unique_edges[t, 1]
        slots = edge_slots[t]
        out_adjacency[scanned_prefix(count_prefix, count_chunk_offsets, u) + slots[0]] = v
        if u != v:
            out_adjacency[scanned_prefix(count_prefix, count_chunk_offsets, v) + slots[1]] = u
    live_faces = state[DECIMATION_FACES]
    if t < live_faces * 3:
        row = scanned_prefix(count_prefix, count_chunk_offsets, vertex_capacity + faces[t])
        out_adjacency[row + corner_slots[t]] = t // 3
    if t < live_faces:
        add_face_quadric(vertices, faces, t, out_quadrics)


@wp.kernel
def quadric_collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    quadrics: wp.array[wp.mat44d],
    feature_count: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_removed: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
    out_cost: wp.array[wp.float32],
    out_sort_keys: wp.array[wp.float32],
    out_sort_order: wp.array[wp.int32],
) -> None:
    # Garland-Heckbert candidate: the cost of collapsing this edge and where its survivor lands.
    # ``out_cost`` is left at +inf for a rejected edge (and for every padded row past the live edge
    # count), so the cost sort puts every rejection past every candidate and the budget cut never
    # picks one up. The cost is written twice: once kept, once into the sort's key buffer beside an
    # identity payload, so nothing is copied between here and that sort.
    k = wp.int32(wp.tid())
    out_survivor[k] = -1
    out_cost[k] = wp.inf
    out_sort_keys[k] = wp.inf
    out_sort_order[k] = k
    if k >= state[DECIMATION_EDGES]:
        return
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    is_boundary = edge_face_count[k] == 1

    # The feature rule is ``collapse_survivor``'s, shared with ``collapse_candidates``. What
    # differs is only the free placement: that one takes the midpoint, this one the quadric's
    # minimizer.
    s, r, placement = collapse_survivor_of_codes(
        finalize_vertex_codes(feature_count[u]),
        finalize_vertex_codes(feature_count[v]),
        u,
        v,
        is_boundary,
    )
    if placement == COLLAPSE_REJECTED:
        return
    free_position = placement == COLLAPSE_FREE

    if not satisfies_link_condition(offsets, columns, u, v, is_boundary):
        return

    quadric = quadrics[u] + quadrics[v]
    midpoint = (to_vec3d(vertices[u]) + to_vec3d(vertices[v])) * wp.float64(0.5)
    optimum = midpoint
    if free_position:
        optimum = quadric_optimum(quadric, midpoint)
    else:
        optimum = to_vec3d(vertices[s])
    target = to_vec3(optimum)

    if collapse_folds_a_face(vertices, faces, vertex_face_offsets, vertex_faces, s, r, target):
        return

    cost = wp.float32(quadric_error(quadric, optimum))
    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = target
    out_cost[k] = cost
    out_sort_keys[k] = cost


@wp.kernel
def drop_past_half(
    order: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_rank: wp.array[wp.int32],
    out_candidates: wp.array[wp.int32],
) -> None:
    # Stage one of the selection: retire every candidate outside the cheapest half of the live
    # edges, ``order`` being the ascending cost ranking. Ranking by cost rather than by edge index
    # is the whole difference between a quadric decimation and a shortest-edge one: the cheapest
    # collapse must win a contested ring.
    #
    # It also records each edge's position in that ranking, which is what lets every round rank its
    # winners with a scan rather than a sort (``commit_budgeted_collapses``).
    i = wp.int32(wp.tid())
    e = order[i]
    out_rank[e] = i
    if i >= wp.max(wp.int32(1), state[DECIMATION_EDGES] // 2):
        out_candidates[e] = -1


@wp.func
def unlocked_candidate(
    candidates: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    k: wp.int32,
) -> wp.int32:
    # Candidate ``k``'s survivor if it may take part in another independent-set round, else -1:
    # retired is every candidate whose two closed 1-rings touch an already-collapsed neighbourhood.
    #
    # That disjointness is exactly what makes reusing the pass's scoring legal. A candidate whose
    # closed 1-rings miss every locked vertex has *no incident face* holding a collapsed endpoint,
    # so its endpoints' quadrics, its cost, its target position, its link condition and its
    # normal-flip veto are all still the ones the scoring pass computed. Fail that test and the
    # candidate must wait for the next geometry rebuild.
    s = candidates[k]
    if s < 0:
        return -1
    if not two_rings_hold(offsets, columns, locked, s, removed[k], wp.int32(0)):
        return -1
    return s


@wp.kernel(enable_backward=False)
def drop_locked_and_claim(
    candidates: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
    out_winner_words: wp.array[wp.uint32],
) -> None:
    # First launch of a round: restore the pass's candidate list minus the locked ones
    # (``unlocked_candidate``), and claim each survivor's closed 2-ring under its hashed key -- the
    # lock half of ``claim_collapse_key``, which reads only this candidate's own survivor and so
    # needs no barrier after the restore. On the first round ``locked`` is all-zero and the restore
    # is the candidate list unchanged. Also clears the winner bitmask ``mark_collapse_winners``
    # sets, whose last reader was the previous round's commit; launched over the edges or the
    # mask's words, whichever is more.
    k = wp.int32(wp.tid())
    if k < out_winner_words.shape[0]:
        out_winner_words[k] = wp.uint32(0)
    if k >= candidates.shape[0]:
        return
    s = unlocked_candidate(candidates, removed, offsets, columns, locked, k)
    out_survivor[k] = s
    if s >= 0:
        lock_two_rings(offsets, columns, s, removed[k], scramble_index(k), out_min_key)


@wp.kernel(enable_backward=False)
def mark_collapse_winners(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    min_key: wp.array[wp.int64],
    rank: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_winner_words: wp.array[wp.uint32],
) -> None:
    # The win test, kept separate from the commit so the round can apply its budget *after* the
    # independent set is known. Trimming members from an independent set keeps it independent;
    # trimming the candidate list beforehand would change which set is found.
    #
    # Each winner sets two bits of one bitmask that one word scan covers: bit ``rank[k]`` of the
    # first half, which is in the pass's cost order, and bit ``k`` of the second, in edge order.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    if not two_rings_hold(offsets, columns, min_key, s, removed[k], scramble_index(k)):
        out_survivor[k] = -1
        return
    p = rank[k]
    half = out_winner_words.shape[0] // 2
    wp.atomic_or(out_winner_words, p >> 5, wp.uint32(1) << wp.uint32(p & 31))
    wp.atomic_or(out_winner_words, half + (k >> 5), wp.uint32(1) << wp.uint32(k & 31))


@wp.func
def winner_budget_rank(
    cost: wp.float32,
    rank: wp.int32,
    k: wp.int32,
    m: wp.int32,
    words: wp.array[wp.uint32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
) -> wp.int32:
    # Winner ``k``'s place in the round's ascending-cost order of winners, the order a stable radix
    # sort of ``winner ? cost : +inf`` over the edge index would give -- from the word scan of
    # ``mark_collapse_winners``' bitmask (``set_bits_before``) instead of from a sort.
    #
    # The pass's cost ranking is that same stable sort over the same key bits with every edge in
    # it, so restricted to the winners it orders them identically, and the bits ahead of ``rank``
    # count the winners ahead of ``k``. That is the whole answer for a cost that sorts before
    # +inf. The round's sort also holds every *non*-winner at +inf, which only an unrepresentable
    # cost can meet: a winner costing exactly +inf ties them and falls behind the ones of lower
    # index, and one whose cost is a positive NaN sorts behind them all. ``wp.cast`` reads the
    # float's bits.
    half = words.shape[0] // 2
    position = set_bits_before(words, prefix, chunk_offsets, 0, rank)
    bits = wp.cast(cost, wp.int32)
    if bits >= 0x7F800000:
        winners = scanned_prefix(prefix, chunk_offsets, half)
        non_winners_before = m - winners
        if bits == 0x7F800000:
            winners_before = set_bits_before(words, prefix, chunk_offsets, half, k) - winners
            non_winners_before = k - winners_before
        position += non_winners_before
    return position


@wp.kernel(enable_backward=False)
def commit_budgeted_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    target_pos: wp.array[wp.vec3],
    cost: wp.array[wp.float32],
    rank: wp.array[wp.int32],
    winner_words: wp.array[wp.uint32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    state: wp.array[wp.int32],
    target_faces: wp.int32,
    round_state: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
    out_locked: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
) -> None:
    # Apply the round's independent set up to its budget, then mark the closed 1-rings of both
    # endpoints of every commit so the next round can tell which candidates it invalidated
    # (``unlocked_candidate``). Launched over the edges or the vertices, whichever is wider: the
    # vertex threads re-arm the claim keys for the next round, which is legal here because
    # ``mark_collapse_winners`` was their last reader.
    #
    # The budget is the rounds' shared one: ``(n_faces - target) // 2`` for the pass -- an interior
    # collapse removes two faces -- less the commits of every round so far, which
    # ``end_collapse_round`` has stored. Its floor of one while nothing has been committed is what
    # lets a pass with a surplus of a single face still finish the job; it cannot manufacture a
    # commit, because the budget only ever *trims* an independent set that is already chosen. A
    # budget of zero commits nothing, which is how a pass that has exhausted its surplus stops.
    #
    # Plain stores for the locks: every write is the same value.
    k = wp.int32(wp.tid())
    if k < out_min_key.shape[0]:
        out_min_key[k] = UNCLAIMED_KEY
    m = survivor.shape[0]
    if k >= m:
        return
    s = survivor[k]
    if s < 0:
        return
    committed = round_state[COLLAPSE_COMMITS]
    budget = (state[DECIMATION_FACES] - target_faces) // 2 - committed
    if committed == 0:
        budget = wp.max(budget, wp.int32(1))
    if winner_budget_rank(cost[k], rank[k], k, m, winner_words, prefix, chunk_offsets) >= budget:
        return
    r = removed[k]
    out_remap[r] = s
    out_positions[s] = target_pos[k]
    wp.atomic_add(out_count, 0, 1)
    stamp_two_rings(offsets, columns, s, r, 1, out_locked)


@wp.kernel
def end_collapse_round(
    max_rounds: wp.int32, count: wp.array[wp.int32], out_state: wp.array[wp.int32]
) -> None:
    # dim=1, last op of a round: decide whether another round against this same scoring is worth
    # running. ``begin_decimation_pass`` seeds the slot table.
    #
    # It stops when the round committed nothing -- a further round cannot, since the state it
    # reads is then unchanged -- or at the round cap. A budget-exhausted pass stops through that
    # same test: its budget is zero, so nothing commits.
    _ = wp.int32(wp.tid())
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + wp.int32(1)
    progressed = count[0] > out_state[COLLAPSE_COMMITS]
    out_state[COLLAPSE_COMMITS] = count[0]
    keep_going = progressed and out_state[LOOP_ROUND] < max_rounds
    out_state[LOOP_CONDITION] = wp.where(keep_going, wp.int32(1), wp.int32(0))


@wp.kernel
def remap_and_mark_faces(
    faces: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``mark_surviving_faces`` for the pass's compaction, also writing the remapped corners, which
    # the pass's own compaction reads (the marks are over ``begin_decimation_pass``' zeroing). A
    # padded face is the dummy triangle, never distinct, so it keeps neither itself nor the dummy
    # vertex.
    f = wp.int32(wp.tid())
    i0, i1, i2, _distinct = mark_surviving_face(faces, remap, f, out_flags)
    write_corner_triple(out_faces, f, i0, i1, i2)


@wp.func
def compacted_vertex(
    flags: wp.array[wp.int32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    n_faces: wp.int32,
    v: wp.int32,
) -> wp.int32:
    # New index of vertex ``v`` after the compaction, or -1 if no surviving face names it. The
    # vertex half of the scan continues the face half's count, so subtract where that ends.
    if flags[n_faces + v] == 0:
        return -1
    return scanned_prefix(prefix, chunk_offsets, n_faces + v) - scanned_prefix(
        prefix, chunk_offsets, n_faces
    )


@wp.kernel
def compact_decimation_pass(
    remapped: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    dummy_vertex: wp.int32,
    collapse_remap: wp.array[wp.int32],
    source: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_source: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Both compactions at once, publishing both counts. The exclusive scan of
    # ``remap_and_mark_faces``' flags runs faces first and vertices after, so a surviving face's
    # row is its scan entry and a kept vertex's slot is ``compacted_vertex``; launched over the
    # faces or the vertices, whichever is wider.
    #
    # A surviving face moves to its row with its corners already renumbered, and every row past the
    # survivors becomes the dummy triangle. Survivors move strictly left and are read from a
    # separate buffer, as the vertices are read from ``positions``, so nothing here can race.
    #
    # The pass's provenance rides the same threads when a caller asked for it (``index`` and
    # ``out_source`` are empty otherwise, which is the whole selector):
    #
    # - a surviving face carries its source-face id to the row the face itself moved to. ``source``
    #   is a copy of the column taken before this launch, because the column cannot be read and
    #   written in one pass.
    # - each *input* vertex's entry of ``index`` is composed with this pass, in place: an input
    #   vertex sits at some live slot, the pass's collapse sends that slot to its survivor, and the
    #   compaction renumbers the survivor (``compacted_vertex``, a pure function of the scan, so no
    #   renumbering table is written first). Composing here rather than returning either map keeps
    #   the map a single array of the *input* length -- fixed width, so the pass stays capturable --
    #   instead of a chain of per-pass maps the caller would have to fold itself. ``index`` is
    #   genuinely in place, which is why it carries no ``out_`` prefix.
    t = wp.int32(wp.tid())
    n_faces = out_faces.shape[0] // 3
    n_vertices = flags.shape[0] - n_faces
    kept_faces = scanned_prefix(prefix, chunk_offsets, n_faces)
    if t == 0:
        last = flags.shape[0] - 1
        kept = scanned_prefix(prefix, chunk_offsets, last) + flags[last]
        out_state[DECIMATION_FACES] = kept_faces
        out_state[DECIMATION_VERTICES] = kept - kept_faces
    if t < n_faces:
        if flags[t] != 0:
            # Reads row ``t`` of ``remapped`` and writes row ``row`` of ``out_faces``: two different
            # row bases, which is the whole hazard here and why both are named rather than spelled
            # as ``* 3 + k``.
            row = scanned_prefix(prefix, chunk_offsets, t)
            a, b, c = corner_triple(remapped, t)
            write_corner_triple(
                out_faces,
                row,
                compacted_vertex(flags, prefix, chunk_offsets, n_faces, a),
                compacted_vertex(flags, prefix, chunk_offsets, n_faces, b),
                compacted_vertex(flags, prefix, chunk_offsets, n_faces, c),
            )
            if t < out_source.shape[0]:
                out_source[row] = source[t]
        if t >= kept_faces:
            write_corner_triple(out_faces, t, dummy_vertex, dummy_vertex, dummy_vertex)
    if t < n_vertices:
        slot = compacted_vertex(flags, prefix, chunk_offsets, n_faces, t)
        if slot >= 0:
            out_vertices[slot] = positions[t]
    if t < index.shape[0]:
        current = index[t]
        if current >= 0:
            survivor = collapse_remap[current]
            index[t] = compacted_vertex(flags, prefix, chunk_offsets, n_faces, survivor)
