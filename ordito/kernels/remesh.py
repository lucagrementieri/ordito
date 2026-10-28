from typing import Any

import warp as wp

from ordito.constants import INT32_MAX_CONSTANT, TOLERANCE_ZERO_CONSTANT, UINT64_MAX_CONSTANT
from ordito.kernels.adjacency import (
    edge_endpoints,
    edge_pair_topology,
    write_edge_row,
    write_face_edge_keys,
    write_face_edge_keys_and_order,
)
from ordito.kernels.array import (
    LOOP_CONDITION,
    LOOP_ROUND,
    binary_search_index_left,
    binary_search_sorted_contains,
    divide_if_positive,
    lowbias32,
    next_row_entry,
    pack_edge_key,
    pack_index_triple,
    popcount32,
    rotate_min_first,
    scanned_slot,
    sort_segment,
    sorted_run_start,
    to_vec2d,
    to_vec3,
    to_vec3d,
    unique_edge_row,
    unpack_edge_key,
)
from ordito.kernels.edges import emit_sorted_unique_edge
from ordito.kernels.grouping import (
    hash_find,
    hash_find_or_insert,
    hash_slot,
    hash_slot_words3,
    key_set_remove,
    next_slot,
    sorted_run_of_length,
)
from ordito.kernels.predicates import (
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
from ordito.kernels.reduce import (
    GROUP_SUM_NONFINITE,
    block_last_ticket,
    block_max,
    block_sum,
    group_sum_field,
    group_sum_from_words,
    group_sum_words,
    last_ticket,
    thread_fence,
)
from ordito.kernels.scatter import (
    accumulate_endpoint_value,
    add_corner_triple,
    add_edge_valence,
    append_to_pair_slot,
    endpoints_hold,
    lock_two_rings,
    mark_corners,
    record_edge_incidence,
    stamp_two_rings,
)
from ordito.kernels.triangles import (
    corner_triple,
    face_normal,
    face_normals_and_area,
    face_vertices_vec3d,
    local_corner,
    triangle_area_max_side,
    triangle_mean_ratio,
    triangle_quality,
    write_corner_triple,
    write_row_triple,
)
from ordito.kernels.vertices import add_to_face_corners, normalized_accumulated_row
from ordito.kernels.voxels import voxel_cell_axes, voxel_cell_center

wp.set_module_options({"enable_backward": False})

# The collapse round loop's state slots **appended** after ``array.LOOP_ROUND`` and
# ``LOOP_CONDITION`` so the shared two keep their numbers: the total commits as of the end of the
# previous round, which is how ``end_collapse_round`` decides whether a round progressed, and the
# faces those commits removed, which is what the next round's face budget is measured against.
COLLAPSE_COMMITS = wp.constant(wp.int32(2))
COLLAPSE_REMOVED_FACES = wp.constant(wp.int32(3))
COLLAPSE_STATE_SIZE = 4

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


@wp.func
def split_face_four(fv: wp.vec3i, mv: wp.vec3i) -> tuple[wp.vec3i, wp.vec3i, wp.vec3i, wp.vec3i]:
    # 1 -> 4 loop-subdivision template: three corner triangles, then the central triangle.
    t0 = wp.vec3i(fv[0], mv[0], mv[2])
    t1 = wp.vec3i(mv[0], fv[1], mv[1])
    t2 = wp.vec3i(mv[2], mv[1], fv[2])
    t3 = wp.vec3i(mv[0], mv[1], mv[2])
    return t0, t1, t2, t3


@wp.func
def write_four_children(
    faces: wp.array[wp.int32],
    corner_edge: wp.array[wp.int32],
    vertex_offset: wp.int32,
    f: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # Face ``f``'s four children under the uniform 1 -> 4 split, into rows ``4f .. 4f + 3``. Corner
    # ``k``'s edge is ``corner_edge[3f + k]`` and every edge is split, so its new vertex is
    # ``vertex_offset`` plus the edge id -- the shift is applied here rather than materialised as a
    # ``3 * n_faces`` index buffer by a separate pass that the split would read straight back.
    # Shared by ``subdivide_mesh`` (``subdivide``) and ``loop_edge_opposites``
    # (``subdivide_loop``), which writes the faces in its own pass over them.
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
def subdivide_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    corner_edge: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The uniform 1 -> 4 split whole, over ``max(n_vertices + n_edges, n_faces)`` threads: the
    # originals followed by one midpoint per unique edge, and each face's four children
    # (``write_four_children``; ``loop_edge_opposites`` writes the same table in its pass over the
    # faces).
    t = wp.int32(wp.tid())
    n_vertices = vertices.shape[0]
    if t < n_vertices:
        out_vertices[t] = vertices[t]
    elif t < out_vertices.shape[0]:
        out_vertices[t] = edge_midpoint(vertices, unique_edges, t - n_vertices)
    if t < faces.shape[0] // 3:
        write_four_children(faces, corner_edge, n_vertices, t, out_faces)


@wp.kernel
def loop_edge_opposites(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    vertex_offset: wp.int32,
    out_opposite_sum: wp.array[wp.vec3],
    out_face_count: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # Per unique edge: how many faces use it, and the sum of the vertices opposite it in each.
    # Corner ``j`` of face ``f`` spans ``(fv[j], fv[j + 1])`` and its opposite vertex is
    # ``fv[j + 2]``, so one pass over the faces gathers both halves of the Loop odd-vertex stencil.
    # The same pass writes the face's four children (``write_four_children``), which read the same
    # corners and edge ids and nothing the stencil produces.
    f = wp.int32(wp.tid())
    for j in range(3):
        e = edge_of_corner[f * 3 + j]
        wp.atomic_add(out_opposite_sum, e, vertices[faces[f * 3 + (j + 2) % 3]])
        wp.atomic_add(out_face_count, e, 1)
    write_four_children(faces, edge_of_corner, vertex_offset, f, out_faces)


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
    out_ring_terms: wp.array2d[wp.vec3],
) -> None:
    # Per vertex: its valence and 1-ring position sum, plus the same two restricted to boundary
    # edges. Driven by the *unique* edge list rather than by the faces, so the valence is the number
    # of distinct neighbours on any input -- the count a per-face pass would have to deduplicate
    # (each neighbour appears twice around an interior vertex but once at a boundary).
    #
    # The ring sum is the one float sum here whose atomic commit order reaches the result: the
    # integer counts are exact, and every other sum that is read has exactly two terms (an interior
    # edge's two opposite vertices, a boundary vertex's two boundary neighbours), which add the same
    # either way round. Under the deterministic setting ``out_ring_terms`` is a non-empty
    # ``(n_unique, 2)`` table and each endpoint's term is stored in it instead, row ``e`` and column
    # ``0`` (``v0``'s) or ``1``, the slot of the endpoint in ``unique_edges``' flat buffer, for
    # ``scatter.sum_sorted_corner_values`` to add in ascending edge order (the branch is
    # warp-uniform).
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
    if out_ring_terms.shape[0] > 0:
        out_ring_terms[e, 0] = p1
        out_ring_terms[e, 1] = p0
    else:
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
# ``triplet_buffers`` hands back uninitialized memory.
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
    split_positions: wp.array[wp.vec3],
    out_counts: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
) -> None:
    # Three independent writes of one refine pass, over ``max(n_faces, n_edges, n_vertices)``
    # threads:
    #
    # - how many triangles face ``t`` becomes under ``emit_size_faces``' templates: one more than
    #   the number of its edges being split. Scanned, this is each face's first output row, so the
    #   emit pass writes the compact buffer directly rather than four fixed slots and a compaction;
    # - vertex ``t`` copied into the prefix of the grown vertex buffer;
    # - the new vertices in its tail: with an empty ``split_positions`` (``None`` at launch), each
    #   split edge ``t``'s midpoint at its rank -- the rule ``split_corner_midpoints`` reads from
    #   the face side, naming the index the rank gives it -- and otherwise the caller's position
    #   ``t``, one per split edge in rank order.
    t = wp.int32(wp.tid())
    n_vertices = vertices.shape[0]
    if t < out_counts.shape[0]:
        count = wp.int32(1)
        for k in range(3):
            if split_mask[corner_edge[t * 3 + k]]:
                count += 1
        out_counts[t] = count
    if t < n_vertices:
        out_vertices[t] = vertices[t]
    if split_positions.shape[0] > 0:
        if t < split_positions.shape[0]:
            out_vertices[n_vertices + t] = split_positions[t]
    elif t < split_mask.shape[0] and split_mask[t]:
        out_vertices[n_vertices + offsets[t]] = edge_midpoint(vertices, unique_edges, t)


@wp.func
def edge_mean_sizing(
    sizing: wp.array[wp.float32], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.float32:
    # The sizing target of unique edge ``e``: the mean of its endpoints' values, the standard
    # reading of a vertex-sampled sizing function and symmetric in the edge's orientation. One
    # definition for the target ``mark_long_edges`` tests the edge against and the value
    # ``extend_edge_mean_sizing`` carries to the midpoint that splits it.
    return wp.float32(0.5) * (sizing[unique_edges[e, 0]] + sizing[unique_edges[e, 1]])


@wp.kernel
def extend_edge_mean_sizing(
    sizing: wp.array[wp.float32],
    unique_edges: wp.array2d[wp.int32],
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    out_sizing: wp.array[wp.float32],
) -> None:
    # ``split_child_counts_and_midpoints``' midpoint for the sizing field rather than the position:
    # the value carried to a new midpoint is the target ``mark_long_edges`` tested the edge with.
    # Over ``max(n_vertices, n_edges)`` threads, thread ``t`` also copies the current field's entry
    # ``t`` into the prefix, so the extended field takes one launch and no copy.
    t = wp.int32(wp.tid())
    n_vertices = sizing.shape[0]
    if t < n_vertices:
        out_sizing[t] = sizing[t]
    if t < split_mask.shape[0]:
        if split_mask[t]:
            out_sizing[n_vertices + offsets[t]] = edge_mean_sizing(sizing, unique_edges, t)


@wp.func
def unique_edge_length(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.float32:
    # Length of unique edge ``e``, in the subtraction order ``edges.edges_unique_length`` uses, so
    # a threshold test on it sees the same ``float32`` value that function would have returned.
    return wp.length(vertices[unique_edges[e, 1]] - vertices[unique_edges[e, 0]])


@wp.func
def mark_long_edge(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    e: wp.int32,
    max_edge: wp.float32,
    sizing: wp.array[wp.float32],
    use_sizing: wp.bool,
    out_long: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # Unique edge ``e``'s verdict, written as ``mark_long_edges`` describes; edge 0 writes the rank
    # buffer's leading zero.
    target = max_edge
    if use_sizing:
        target = edge_mean_sizing(sizing, unique_edges, e)
    is_long = unique_edge_length(vertices, unique_edges, e) > target
    out_long[e] = is_long
    out_flags[e + 1] = wp.where(is_long, wp.int32(1), wp.int32(0))
    if e == 0:
        out_flags[0] = wp.int32(0)


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
    # per-vertex sizing field, ``edge_mean_sizing``. One warp-uniform
    # branch rather than two kernels, since the two differ by a parameter; ``sizing`` is not read
    # (and may be a null array) when ``use_sizing`` is false. The verdict is written twice: as the
    # mask the split reads, and as the ``int32`` flag the caller scans in place into its ranks.
    # ``out_flags`` is that ``n_edges + 1`` rank buffer whole: edge ``e``'s flag goes to entry
    # ``e + 1``, behind the leading zero thread 0 writes, so the buffer needs no zero-fill.
    #
    # **Declined: marking per face instead**, counting each face's children in the same launch
    # (a slot counter beside the flags) so the split and face counts come back in one copy and the
    # child-count launch goes. Measured on a jittered 82 k-face sphere, device time: the per-face
    # mark (three gathered edges, scattered duplicate flag stores) plus an emit that also writes
    # the vertices took the whole pass 1.24x (``subdivide_to_size``) and 1.19x (``split_edges``)
    # of this per-edge form, and the zeroed counter adds a memset to the pass that splits nothing.
    mark_long_edge(
        vertices,
        unique_edges,
        wp.int32(wp.tid()),
        max_edge,
        sizing,
        use_sizing,
        out_long,
        out_flags,
    )


@wp.kernel
def emit_long_edges(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    max_edge: wp.float32,
    sizing: wp.array[wp.float32],
    use_sizing: wp.bool,
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
    out_long: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``edges.emit_sorted_unique_edges`` and ``mark_long_edges`` in one pass over the sorted
    # corners: the position starting an edge's run writes its row and tests it, reading back the
    # row it wrote, so the verdict is ``mark_long_edges``' over the same table bit for bit.
    i = wp.int32(wp.tid())
    _corner, e, start = emit_sorted_unique_edge(
        faces, sorted_keys, order, ranks, i, out_unique_edges, out_inverse
    )
    if start:
        mark_long_edge(
            vertices, out_unique_edges, e, max_edge, sizing, use_sizing, out_long, out_flags
        )


@wp.func
def write_split_children(
    fv: wp.vec3i,
    mv: wp.vec3i,
    vertices: wp.array[wp.vec3],
    base: wp.int32,
    src: wp.int32,
    out_faces: wp.array2d[wp.int32],
    out_index: wp.array[wp.int32],
) -> None:
    # The crack-free re-triangulation of a face ``fv`` by which of its edges carry a new vertex
    # (``mv[k]``, the vertex on edge ``(fv[k], fv[k + 1])``, ``-1`` where unsplit), written into
    # rows ``base ..`` of the compact output with provenance ``src`` (skipped when ``out_index`` is
    # empty). Shared by ``emit_size_faces``, which numbers the new vertices by unique edge, and
    # ``intersection.emit_plane_split_faces``, which numbers only the crossed ones; the templates
    # never assume the new vertex is a midpoint. The children come out in template order ``t0, t1,
    # ..``, which is the order the fixed-slot form's compaction kept them in.
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

    write_row_triple(out_faces, base, t0[0], t0[1], t0[2])
    if out_index.shape[0] > 0:
        out_index[base] = src
    if count >= 1:
        write_row_triple(out_faces, base + 1, t1[0], t1[1], t1[2])
        if out_index.shape[0] > 0:
            out_index[base + 1] = src
    if count >= 2:
        write_row_triple(out_faces, base + 2, t2[0], t2[1], t2[2])
        if out_index.shape[0] > 0:
            out_index[base + 2] = src
    if count >= 3:
        write_row_triple(out_faces, base + 3, t3[0], t3[1], t3[2])
        if out_index.shape[0] > 0:
            out_index[base + 3] = src


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
    # ``split_child_counts_and_midpoints``). An empty ``index_in`` / ``out_index`` (``None`` at
    # launch) carries no provenance.
    #
    # **Declined: writing the children's corner keys for the next pass's sort here**, as a collapse
    # pass's compaction does (``compact_surviving_faces``), to drop that pass's key launch. The
    # variable-length child rows scatter the key stores: this launch went from 0.15 to 0.55 ms
    # against 0.20 ms for the key launch it removed, ``subdivide_to_size`` on a 1.3 M-face sphere
    # 1.06x slower.
    f = wp.int32(wp.tid())
    src = wp.int32(-1)
    if index_in.shape[0] > 0:
        src = index_in[f]

    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = split_corner_midpoints(corner_edge, split_mask, offsets, vertex_offset, f)

    write_split_children(fv, mv, vertices, face_offsets[f], src, out_faces, out_index)


# ---------------------------------------------------------------------------
# Region-restricted subdivision helpers
# ---------------------------------------------------------------------------


@wp.func
def mark_long_region_edge(
    e: wp.int32, out_long: wp.array[wp.bool], out_flags: wp.array[wp.int32]
) -> None:
    # Mark unique edge ``e`` long, in the mask and the zeroed ``int32`` flags; every racing write
    # is the same value.
    out_long[e] = wp.bool(True)
    out_flags[e] = wp.int32(1)


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
            mark_long_region_edge(e, out_long, out_flags)


@wp.kernel
def emit_long_region_edges(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    region_flags: wp.array[wp.int32],
    max_edge: wp.float32,
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
    out_long: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``edges.emit_sorted_unique_edges`` and ``mark_long_region_edges`` in one pass over the sorted
    # corners. A corner measures its edge from its own endpoints, the row the run's first position
    # writes (``adjacency.edge_endpoints``), in ``unique_edge_length``'s subtraction order, so the
    # test is ``mark_long_region_edges``' bit for bit without waiting on that row.
    i = wp.int32(wp.tid())
    corner, e, _start = emit_sorted_unique_edge(
        faces, sorted_keys, order, ranks, i, out_unique_edges, out_inverse
    )
    if region_flags[corner // 3] != 0:
        a, b = edge_endpoints(faces, corner)
        if wp.length(vertices[b] - vertices[a]) > max_edge:
            mark_long_region_edge(e, out_long, out_flags)


@wp.kernel
def longest_edge_sort_keys(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    out_keys: wp.array[wp.float32],
    out_edges: wp.array[wp.int32],
) -> None:
    # A budget-trimmed refinement pass's sort input, one thread per unique edge: each flagged edge
    # (``offsets`` is the ``n_edges + 1`` scan of the flags, so it steps there) writes its negated
    # length -- ascending on it is descending on the length -- and its own index at its rank, the
    # flagged edges in ascending order, straight into the radix sort's double-width buffers.
    e = wp.int32(wp.tid())
    slot, flagged = scanned_slot(offsets, e)
    if flagged:
        out_keys[slot] = -unique_edge_length(vertices, unique_edges, e)
        out_edges[slot] = e


@wp.kernel
def keep_longest_edges(
    sorted_edges: wp.array[wp.int32], remaining: wp.int32, out_mask: wp.array[wp.bool]
) -> None:
    # One thread per flagged edge in descending length order: the first ``remaining`` stay flagged
    # in ``out_mask`` and the rest are cleared, in place (every unflagged edge is already clear).
    i = wp.int32(wp.tid())
    out_mask[sorted_edges[i]] = i < remaining


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


@wp.func
def vertex_scale(
    scale: wp.array[wp.float32], valence: wp.array[wp.float32], v: wp.int32
) -> wp.float32:
    # Vertex ``v``'s scale attribute: ``scale[v]`` itself, or, while ``valence`` is non-empty, the
    # mean ``scatter_scale_attribute``'s sum and count make (``array.divide_if_positive``), formed
    # where it is read so the first pass needs no division launch of its own.
    if valence.shape[0] > 0:
        return divide_if_positive(scale[v], valence[v])
    return scale[v]


@wp.kernel
def mark_density_splits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    valence: wp.array[wp.float32],
    alpha: wp.float32,
    out_split: wp.array[wp.int32],
) -> None:
    # Which region faces want a centroid split this pass, as 0/1 so the result scans directly into
    # the new-vertex slots ``emit_density_splits`` needs. A face outside the region never splits,
    # which is what keeps the refinement inside the patch. ``out_split`` is the caller's
    # ``n_faces + 1`` rank buffer whole: face ``f``'s flag goes to entry ``f + 1``, behind the
    # leading zero thread 0 writes, so the buffer needs no zero-fill.
    f = wp.int32(wp.tid())
    if f == 0:
        out_split[0] = wp.int32(0)
    if not region[f]:
        out_split[f + 1] = wp.int32(0)
        return
    i, j, k = corner_triple(faces, f)
    wanted = density_split_wanted(
        vertices[i],
        vertices[j],
        vertices[k],
        vertex_scale(scale, valence, i),
        vertex_scale(scale, valence, j),
        vertex_scale(scale, valence, k),
        alpha,
    )
    out_split[f + 1] = wp.where(wanted, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_density_splits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    valence: wp.array[wp.float32],
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
    #
    # Over ``max(n_faces, n_vertices)`` threads: thread ``t`` also copies the current position and
    # scale ``t`` into the prefixes of the grown buffers, whose tails the splits write, so the pass
    # takes one launch and no copy.
    f = wp.int32(wp.tid())
    if f < n_vertices:
        out_positions[f] = vertices[f]
        out_scale[f] = vertex_scale(scale, valence, f)
    if f >= region.shape[0]:
        return
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
    out_positions[center] = (vertices[i] + vertices[j] + vertices[k]) / wp.float32(3.0)
    out_scale[center] = (
        vertex_scale(scale, valence, i)
        + vertex_scale(scale, valence, j)
        + vertex_scale(scale, valence, k)
    ) / wp.float32(3.0)
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
    out_boundary_vertex: wp.array[wp.bool],
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
    # The pass also seeds the flip loop's other two structures: every halfedge starts with no row
    # (``out_halfedge_row`` given; ``emit_flip_topology`` then writes the interior ones), and on an
    # incremental build (``edge_set_mask >= 0``) every run -- every undirected edge, whatever its
    # multiplicity -- goes into the zeroed duplicate-edge set. A non-empty
    # ``out_valence`` (zeroed by the caller) gets every vertex's degree from the same runs
    # (``scatter.add_edge_valence``), which the valence flip objective reads, and a non-empty
    # ``out_boundary_vertex`` (zeroed) both endpoints of every run of exactly one -- a boundary
    # edge, the feature counts' rule (one incident face) read off the sort, by plain stores of one
    # value.
    i = wp.int32(wp.tid())
    if out_boundary_vertex.shape[0] > 0 and sorted_run_of_length(sorted_keys, n, i, 1):
        lo, hi = unpack_edge_key(sorted_keys[i], base)
        out_boundary_vertex[lo] = True
        out_boundary_vertex[hi] = True
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
    # The build keeps the two halfedges too, in both directions (row -> halfedges, halfedge -> row;
    # ``mark_edge_pair_starts`` resets the second first), which is what lets ``commit_flips`` keep
    # the rows current across a flip round instead of regrouping the whole mesh. A caller with no
    # flip loop passes ``None`` for both.
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


@wp.func
def refresh_flip_row(
    faces: wp.array[wp.int32],
    row_halfedges: wp.array2d[wp.int32],
    k: wp.int32,
    stride: wp.int32,
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
    out_face_claim: wp.array[wp.uint64],
    out_edge_claim: wp.array[wp.uint64],
) -> None:
    # Row ``k`` of the flip tables again, from the halfedges ``commit_flips`` kept current: the row
    # ``emit_flip_topology`` would write after a full regroup, except that each edge keeps the row
    # it had instead of moving to its new key's rank -- which the flip loop does not depend on,
    # since ``claim_flips`` ranks candidates by key rather than by row.
    #
    # Also re-arms the next round's two claim tables, which this round's commit was the last to
    # read and the next round's candidates never read: two memsets fewer per round, the launch's
    # ``stride`` threads striding over tables longer than their count.
    for f in range(k, out_face_claim.shape[0], stride):
        out_face_claim[f] = UINT64_MAX_CONSTANT
    for s in range(k, out_edge_claim.shape[0], stride):
        out_edge_claim[s] = UINT64_MAX_CONSTANT
    if k < row_halfedges.shape[0]:
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
def refresh_flip_rows(
    faces: wp.array[wp.int32],
    row_halfedges: wp.array2d[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
    out_face_claim: wp.array[wp.uint64],
    out_edge_claim: wp.array[wp.uint64],
) -> None:
    # Every row of an incremental flip round, one thread a row. ``adopt_flip_edge_set`` below is
    # the same refresh launched over the halfedges, for the round that first builds the key set.
    k = wp.int32(wp.tid())
    refresh_flip_row(
        faces,
        row_halfedges,
        k,
        row_halfedges.shape[0],
        out_adjacency,
        out_adjacency_edges,
        out_unshared,
        out_face_claim,
        out_edge_claim,
    )


@wp.kernel
def adopt_flip_edge_set(
    faces: wp.array[wp.int32],
    halfedge_row: wp.array[wp.int32],
    row_halfedges: wp.array2d[wp.int32],
    base: wp.uint64,
    edge_set_mask: wp.int32,
    edge_set: wp.array[wp.uint64],
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
    out_face_claim: wp.array[wp.uint64],
    out_edge_claim: wp.array[wp.uint64],
) -> None:
    # The incremental state after a plain build's first flipping round, without the regroup that
    # would rebuild it (a key launch, a radix sort, a mark, a scan and an emit): the plain build
    # already kept the halfedge <-> row maps and the round's commit moved them, so only the
    # duplicate-edge set is new. One thread per halfedge inserts the current key of every edge
    # once -- an interior row's from its first halfedge, any other halfedge's (a boundary or
    # non-manifold edge, ``halfedge_row == -1``) from itself, duplicates merged by the insert --
    # which is the set ``mark_edge_pair_starts`` seeds from a fresh sort's runs. The same threads
    # then refresh the rows as ``refresh_flip_rows`` does; there are more halfedges than rows.
    h = wp.int32(wp.tid())
    row = halfedge_row[h]
    if row < 0 or row_halfedges[row, 0] == h:
        f = h // 3
        j = h - 3 * f
        hash_find_or_insert(
            pack_edge_key(faces[h], faces[3 * f + (j + 1) % 3], base), edge_set, edge_set_mask
        )
    refresh_flip_row(
        faces,
        row_halfedges,
        h,
        halfedge_row.shape[0],
        out_adjacency,
        out_adjacency_edges,
        out_unshared,
        out_face_claim,
        out_edge_claim,
    )


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
    # An empty ``region_flags`` (``None`` at launch) is the whole mesh.
    f0 = adjacency[k, 0]
    if region_flags.shape[0] > 0 and (region_flags[f0] == 0 or region_flags[adjacency[k, 1]] == 0):
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
    # Commit the winning flips, and keep the loop's state describing the new faces: the halfedge
    # <-> row maps every build keeps (so ``refresh_flip_rows`` can rebuild every row without a
    # regroup),
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
        # No halfedge maps to keep: the caller regroups.
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
def finalize_vertex_codes(feature_count: wp.int32) -> wp.int32:
    # A vertex's code from its incident feature-edge count. ``fill_edge_topology`` stores
    # only the counts, and every reader applies this as it loads one, so no pass rewrites the table.
    # 0 feature edges -> FREE; exactly 2 -> CREASE (on a smooth feature/boundary line);
    # anything else (1 = feature endpoint, >=3 = junction) -> CORNER (frozen).
    code = CORNER_VERTEX
    if feature_count == 0:
        code = FREE_VERTEX
    elif feature_count == 2:
        code = CREASE_VERTEX
    return code


@wp.func
def collapse_survivor_of_codes(
    cu: wp.int32, cv: wp.int32, u: wp.int32, v: wp.int32, is_boundary: wp.bool
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # ``collapse_survivor`` with the two endpoint codes already in hand.
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
    feature_count: wp.array[wp.int32], u: wp.int32, v: wp.int32, is_boundary: wp.bool
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # Which endpoint of edge ``(u, v)`` survives the collapse, which one is removed, and whether
    # the survivor's position is pinned or free -- the feature rule alone, with no geometry in it.
    #
    # One rule, two decimators: ``collapse_candidates`` and ``quadric_collapse_candidates`` differ
    # only in what they do with ``COLLAPSE_FREE`` -- one takes the midpoint, the other minimizes the
    # summed quadric.
    return collapse_survivor_of_codes(
        finalize_vertex_codes(feature_count[u]),
        finalize_vertex_codes(feature_count[v]),
        u,
        v,
        is_boundary,
    )


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
    # shared by ``fill_edge_topology`` and the quadric pass's ``count_pass_edges``.
    count = edge_face_count[e]
    feature = count == 1
    if count == 2:
        normal_a, _area_a = face_normals_and_area(vertices, faces, edge_faces[e, 0])
        normal_b, _area_b = face_normals_and_area(vertices, faces, edge_faces[e, 1])
        feature = vector_angle(normal_a, normal_b) > feature_angle
    return feature


# ``isotropic_remesh``'s per-grouping topology, in one count, one scan and one fill
# (``remesh._edge_incidence``): the edge incidence, the vertex rings, the vertex-face rows and the
# feature counts every collapse pass and every smoothing pass read, which were three separately
# counted, scanned and filled builds (``graph.edges_to_neighbor_lists_from_counts``,
# ``adjacency.vertex_face_adjacency_from_counts`` and a feature-count scatter). The two
# CSRs share one count buffer ``[degree (n_vertices) | corners (n_vertices)]``, one inclusive scan
# behind a leading zero and one column buffer ``[ring entries (2 m) | face entries (3 n_faces)]``,
# so the vertex-face offsets are the scan's upper half read straight into the shared columns --
# the layout ``_DecimationBuffers`` builds for the quadric pass. Rows are in arrival order: every
# reader counts, exits on a hit or takes a minimum over a whole row (the link condition, the band
# walks, the ring locks, both fold vetoes), and the deterministic smoothing ring sorts its rows.
#
# ``remesh._CollapseBuffers`` runs the same launches at fixed widths, the collapse loop's entry
# sizes, and keeps the live sizes in a ``state`` buffer: every kernel that takes ``state`` stops at
# the live prefix it reads there, and one launched with an empty one runs over its whole arrays.
# Slots: the pass's unique edges (``emit_edge_topology``), the live vertices and faces (which
# ``compact_surviving_faces`` overwrites with the next pass's, in its ``out_totals`` order), and
# the pass's commits (zeroed by ``collapse_candidates``).
COLLAPSE_PASS_EDGES = wp.int32(0)
COLLAPSE_PASS_VERTICES = wp.int32(1)
COLLAPSE_PASS_FACES = wp.int32(2)
COLLAPSE_PASS_COMMITS = wp.int32(3)
COLLAPSE_PASS_STATE_SIZE = 4


@wp.func
def count_corner_topology(
    c: wp.int32,
    e: wp.int32,
    faces: wp.array[wp.int32],
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
) -> wp.int32:
    # Corner ``c`` of unique edge ``e``, over zeroed counts: its face onto the edge
    # (``scatter.append_to_pair_slot``, whose arrival slot it returns) and the corner onto its
    # vertex's face count. Corner 0 writes the scan's leading zero.
    n_vertices = out_counts.shape[0] // 2
    slot = append_to_pair_slot(e, c // 3, out_edge_face_count, out_edge_faces)
    wp.atomic_add(out_counts, n_vertices + faces[c], 1)
    if c == 0:
        out_offsets[0] = 0
    return slot


@wp.func
def count_edge_degrees(out_counts: wp.array[wp.int32], u: wp.int32, v: wp.int32) -> None:
    # A unique edge onto both endpoints' ring degrees, once per edge.
    wp.atomic_add(out_counts, u, 1)
    wp.atomic_add(out_counts, v, 1)


@wp.kernel
def count_edge_topology(
    inverse: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    ring: wp.int32,
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
) -> None:
    # One thread per corner of a grouping the caller holds (``count_corner_topology``), and --
    # with ``ring`` (warp-uniform) -- the edge onto both endpoints' degrees, by the one corner that
    # drew the edge's first slot. ``emit_edge_topology`` is the same count fused into the emit of a
    # grouping it builds: the two differ only in where the edge comes from and which corner counts
    # its degrees.
    c = wp.int32(wp.tid())
    e = inverse[c]
    slot = count_corner_topology(
        c, e, faces, out_edge_face_count, out_edge_faces, out_counts, out_offsets
    )
    if ring != 0 and slot == 0:
        count_edge_degrees(out_counts, unique_edges[e, 0], unique_edges[e, 1])


@wp.kernel
def emit_edge_topology(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    ring: wp.int32,
    state: wp.array[wp.int32],
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # ``edges.emit_sorted_unique_edges`` and ``count_edge_topology`` in one pass over the sorted
    # corners: each position emits its corner's inverse entry (and the run start its edge's row),
    # then counts the corner as ``count_edge_topology`` does, the run start counting the degrees.
    # Within a run the stable sort keeps the corners ascending, so the CPU device fills each edge's
    # face slots in the corner order ``count_edge_topology`` does.
    #
    # With a ``state`` (the same buffer as ``out_state``) the sort ran over a fixed width whose
    # dead corners hold ``EDGE_KEY_PAD`` (``compact_surviving_faces``) and so sorted last: only the
    # live positions emit, and the last of them publishes the unique-edge count, which is its rank.
    i = wp.int32(wp.tid())
    if state.shape[0] > 0:
        live = 3 * state[COLLAPSE_PASS_FACES]
        if i >= live:
            if i == 0:
                out_state[COLLAPSE_PASS_EDGES] = 0
            return
        if i + 1 == live:
            out_state[COLLAPSE_PASS_EDGES] = ranks[i]
    corner, e, start = emit_sorted_unique_edge(
        faces, sorted_keys, order, ranks, i, out_unique_edges, out_inverse
    )
    count_corner_topology(
        corner, e, faces, out_edge_face_count, out_edge_faces, out_counts, out_offsets
    )
    if ring != 0 and start:
        u, v = edge_endpoints(faces, corner)
        count_edge_degrees(out_counts, u, v)


@wp.kernel
def fill_edge_topology(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    feature_angle: wp.float32,
    ring: wp.int32,
    state: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_feature_count: wp.array[wp.int32],
) -> None:
    # Over ``max(m, n_faces)`` threads, after the scan of ``count_edge_topology``'s counts: edge
    # ``t`` writes both directed ring entries (with ``ring``) and adds one to both endpoints'
    # feature counts when it is a feature edge (``is_feature_edge``; an empty
    # ``out_feature_count`` skips it), and face
    # ``t`` writes itself into its three
    # vertices' face rows. ``cursor`` is the spent count buffer, counting each row's free slots down
    # as in ``graph.scatter_neighbor_lists`` and ``adjacency.scatter_vertex_faces``. A ``state``
    # holds the live edge and face counts of fixed-width arrays (``emit_edge_topology``).
    t = wp.int32(wp.tid())
    n_vertices = cursor.shape[0] // 2
    n_edges = unique_edges.shape[0]
    n_faces = faces.shape[0] // 3
    if state.shape[0] > 0:
        n_edges = state[COLLAPSE_PASS_EDGES]
        n_faces = state[COLLAPSE_PASS_FACES]
    if t < n_edges:
        u = unique_edges[t, 0]
        v = unique_edges[t, 1]
        if ring != 0:
            out_columns[offsets[u + 1] - wp.atomic_add(cursor, u, -1)] = v
            out_columns[offsets[v + 1] - wp.atomic_add(cursor, v, -1)] = u
        if out_feature_count.shape[0] > 0 and is_feature_edge(
            vertices, faces, edge_face_count, edge_faces, t, feature_angle
        ):
            wp.atomic_add(out_feature_count, u, 1)
            wp.atomic_add(out_feature_count, v, 1)
    if t < n_faces:
        for k in range(3):
            row = n_vertices + faces[t * 3 + k]
            out_columns[offsets[row + 1] - wp.atomic_add(cursor, row, -1)] = t


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
def move_fold_and_sheet(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    moved: wp.int32,
    partner: wp.int32,
    target: wp.vec3,
) -> tuple[wp.bool, wp.vec3, wp.float32]:
    # One walk over the faces of ``moved``, returning ``(flips, sheet, welded_quality)``:
    #
    # - ``flips``: would moving ``moved`` to ``target`` invert any face it still belongs to? Every
    #   incident face keeps its other two corners and must keep its orientation. This is the guard
    #   that separates a usable decimator from one that produces self-intersecting geometry, and it
    #   is why the vertex-face CSR is built at all.
    # - ``sheet``: the summed area vectors of every face of ``moved``, before the move -- which way
    #   the surface around ``moved`` faces, read by ``collapse_joins_opposite_sheets``. The faces
    #   holding ``partner`` count too: summing only the kept ones measured a stronger veto (the
    #   ``happy_buddha`` 0.01 ``i2o`` p99.9 0.36x where this is 0.47x) that cost 1st-percentile
    #   quality (0.97x over the 15-cell set, ``bunny`` 0.01 0.80x).
    # - ``welded_quality``: the worst mean ratio among the faces holding ``partner`` too (1 when
    #   there are none), the faces a collapse removes.
    #
    # ``partner`` names a vertex whose incident faces are exempt, and it is what makes one helper
    # serve both callers of the fold rule. A *collapse* welds ``moved`` onto ``partner``, so the two
    # faces holding both endpoints vanish and must be skipped; a *smoothing* step moves one vertex
    # and welds nothing, so it passes ``-1``, which no corner index equals and which therefore
    # exempts nothing. A caller reading only ``flips`` pays for nothing else: the other two are dead
    # code once the function is inlined.
    sheet = wp.vec3()
    welded_quality = wp.float32(1.0)
    for slot in range(vertex_face_offsets[moved], vertex_face_offsets[moved + 1]):
        f = vertex_faces[slot]
        i0, i1, i2 = corner_triple(faces, f)
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        before = wp.cross(p1 - p0, p2 - p0)
        sheet = sheet + before
        if i0 == partner or i1 == partner or i2 == partner:
            welded_quality = wp.min(welded_quality, triangle_mean_ratio(p0, p1, p2))
            continue
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
            return True, sheet, welded_quality  # the collapse would flatten it outright
        if wp.dot(before / before_length, after / after_length) < COLLAPSE_MIN_NORMAL_DOT:
            return True, sheet, welded_quality
    return False, sheet, welded_quality


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
    # The fold rule alone (``move_fold_and_sheet``), for a caller with no use for the rest.
    flips, _sheet, _welded_quality = move_fold_and_sheet(
        vertices, faces, vertex_face_offsets, vertex_faces, moved, partner, target
    )
    return flips


@wp.func
def collapse_joins_opposite_sheets(
    sheet_r: wp.vec3, sheet_s: wp.vec3, welded_quality: wp.float32
) -> wp.bool:
    # Does a collapse weld two sheets of surface that face away from each other -- the two sides of
    # a thin part, or the two walls of a narrow slit? ``sheet_r`` and ``sheet_s`` are the summed
    # area vectors of each endpoint's faces (``move_fold_and_sheet``); they oppose there and
    # nowhere else on a surface sampled finer than its curvature.
    #
    # Such a collapse is cheap for a memoryless quadric when the part is thin, and it is never one
    # collapse: the welded vertex carries both sheets' planes, its neighbours follow, and the
    # doubly covered sheet left behind then retracts in its own plane at almost no cost. On the
    # ``happy_buddha`` scan at 0.01 the worst ``i2o`` input points sat in slits 1e-4 of the
    # diagonal wide, as deep as 8e-3 of it, all lost that way.
    #
    # **Exempt: a collapse that removes a sliver** (a welded face below
    # ``COLLAPSE_SLIVER_QUALITY``). A scan's thin parts are fringed by the input's own needles,
    # whose short edges cross the rim; vetoing those too kept them in the output (1st-percentile
    # quality 0.21x on ``happy_buddha`` at 0.01, 0.89x over the 15-cell set), while removing one
    # moves the surface by about its own width.
    if welded_quality < COLLAPSE_SLIVER_QUALITY:
        return False
    return wp.dot(sheet_r, sheet_s) < 0.0


@wp.func
def collapse_fold_and_sheets(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    s: wp.int32,
    r: wp.int32,
    p: wp.vec3,
) -> tuple[wp.bool, wp.bool]:
    # ``(folds, joins_opposite_sheets)`` for an edge collapse welding ``r`` onto ``s`` at ``p``:
    # both endpoints move to ``p``, so both directions of ``move_fold_and_sheet`` are walked,
    # unconditionally -- even under a pinned placement, where ``s`` stays put and its half of the
    # fold test is a no-op. A fold short-circuits, since the collapse is rejected either way.
    folds, sheet_r, welded_quality = move_fold_and_sheet(
        vertices, faces, vertex_face_offsets, vertex_faces, r, s, p
    )
    if folds:
        return True, False
    folds, sheet_s, _welded_quality = move_fold_and_sheet(
        vertices, faces, vertex_face_offsets, vertex_faces, s, r, p
    )
    if folds:
        return True, False
    return False, collapse_joins_opposite_sheets(sheet_r, sheet_s, welded_quality)


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
    # The fold veto alone (``collapse_fold_and_sheets``). One decision rule for both decimators:
    # ``collapse_candidates`` reads only this, ``quadric_collapse_candidates`` the sheet test too.
    folds, _joins = collapse_fold_and_sheets(
        vertices, faces, vertex_face_offsets, vertex_faces, s, r, p
    )
    return folds


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
DECIMATION_REMOVED_FACES = wp.constant(wp.int32(4))

# The claim table's value at a vertex no candidate has claimed yet: above every ``scramble_index``.
UNCLAIMED_KEY = wp.constant(wp.int64(2**63 - 1))

EDGE_KEY_PAD = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))


@wp.func
def vertex_band(bands: wp.array[wp.float32], uniform: wp.float32, v: wp.int32) -> wp.float32:
    # Vertex ``v``'s length band: its entry of a per-vertex sizing table, or ``uniform`` when the
    # table is empty (``None`` at launch) because the target is one constant.
    if bands.shape[0] > 0:
        return bands[v]
    return uniform


# What makes an edge a collapse candidate in ``collapse_candidates``: shorter than its length band
# (``isotropic_remesh``), or no longer than the degeneracy repair's bound inside its region
# (``repair.fix_degeneracies``).
COLLAPSE_SELECT_LENGTH = wp.int32(0)
COLLAPSE_SELECT_DEGENERATE = wp.int32(1)

# The worst aspect ratio a degeneracy collapse may leave in a face it reshapes, unless one of the
# faces it touches was already worse.
COLLAPSE_MAX_NEW_ASPECT = wp.float32(20.0)


@wp.func
def edge_faces_in_region(
    faces: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    k: wp.int32,
    region: wp.array[wp.float32],
) -> wp.bool:
    # Do all faces of unique edge ``k`` lie in the region? The region is carried per vertex (a
    # positive entry is inside, which survives the passes' vertex compaction where a face mask
    # would not), and a face is inside when its three corners are. An empty table is everywhere.
    if region.shape[0] == 0:
        return True
    for j in range(wp.min(edge_face_count[k], 2)):
        i0, i1, i2 = corner_triple(faces, edge_faces[k, j])
        if region[i0] <= 0.0 or region[i1] <= 0.0 or region[i2] <= 0.0:
            return False
    return True


@wp.func
def endpoint_collapse_cost(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    removed: wp.int32,
    kept: wp.int32,
) -> tuple[wp.float32, wp.float32, wp.float32]:
    # ``(cost, worst_before, worst_after)`` of collapsing ``removed`` into ``kept``, which keeps
    # its position. ``cost`` is the summed squared distance of ``kept`` from the planes of
    # ``removed``'s faces: a memoryless quadric, which is zero for ``kept``'s own planes and so
    # needs only this one walk. ``worst_before`` is the worst aspect ratio among those faces, the
    # vanishing ones included; ``worst_after`` among the reshaped ones with ``removed`` moved.
    target = vertices[kept]
    cost = wp.float32(0.0)
    worst_before = wp.float32(0.0)
    worst_after = wp.float32(0.0)
    for slot in range(vertex_face_offsets[removed], vertex_face_offsets[removed + 1]):
        i0, i1, i2 = corner_triple(faces, vertex_faces[slot])
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        area = wp.cross(p1 - p0, p2 - p0)
        area_length = wp.length(area)
        if area_length > wp.float32(0.0):
            distance = wp.dot(area, target - vertices[removed]) / area_length
            cost += distance * distance
        worst_before = wp.max(worst_before, triangle_aspect_ratio(p0, p1, p2))
        if i0 == kept or i1 == kept or i2 == kept:
            continue
        if i0 == removed:
            p0 = target
        elif i1 == removed:
            p1 = target
        else:
            p2 = target
        worst_after = wp.max(worst_after, triangle_aspect_ratio(p0, p1, p2))
    return cost, worst_before, worst_after


@wp.kernel(enable_backward=False)
def collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    feature_count: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    low_uniform: wp.float32,
    high_uniform: wp.float32,
    select: wp.int32,
    degeneracy: wp.vec3,
    state: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_removed: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
    out_claim: wp.array[wp.int64],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_state: wp.array[wp.int32],
) -> None:
    # ``select`` picks the candidates: ``COLLAPSE_SELECT_LENGTH`` reads the length bands below,
    # ``COLLAPSE_SELECT_DEGENERATE`` reads ``degeneracy`` = ``(tiny_edge_length, max_length,
    # critical_aspect_ratio)`` instead, with ``low`` as the per-vertex region
    # (``edge_faces_in_region``) and keeps an endpoint. The survivor rule, link condition, fold veto
    # and hashed lock key are shared.
    #
    # ``low`` and ``high`` are per *vertex* against an adaptive sizing field; a uniform target
    # passes them empty and its two bands as ``low_uniform`` / ``high_uniform`` (``vertex_band``),
    # so one code path serves both with no constant field allocated. An edge's own band is the
    # mean of its endpoints', matching ``mark_long_edges``. The length is
    # computed here: this is its only reader, and it reads it at its own edge.
    #
    # The edge threads also seed the per-vertex state the claim and the commit start from -- the
    # unclaimed keys, the identity collapse map and the working positions -- striding over the
    # vertices: this kernel reads none of it, and the next two launches are its only readers. That
    # is a fill, an ``arange`` and a clone fewer per pass.
    #
    # Over fixed-width arrays (a ``state``, ``emit_edge_topology``) every thread seeds and retires
    # its slot, and only the live edges go on: the claim and the commit skip a retired slot. Thread
    # 0 also zeroes the pass's commit count, which ``commit_collapses`` counts into and nothing
    # reads before.
    k = wp.int32(wp.tid())
    for t in range(k, out_remap.shape[0], unique_edges.shape[0]):
        out_claim[t] = UNCLAIMED_KEY
        out_remap[t] = t
        out_positions[t] = vertices[t]
    out_survivor[k] = -1
    if state.shape[0] > 0:
        if k == 0:
            out_state[COLLAPSE_PASS_COMMITS] = 0
        if k >= state[COLLAPSE_PASS_EDGES]:
            return
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    length = unique_edge_length(vertices, unique_edges, k)
    if select == COLLAPSE_SELECT_LENGTH:
        low_u = vertex_band(low, low_uniform, u)
        low_v = vertex_band(low, low_uniform, v)
        if length >= wp.float32(0.5) * (low_u + low_v):
            return
    is_boundary = edge_face_count[k] == 1

    # Choose the surviving vertex and its target position; this decimator places a free collapse at
    # the edge midpoint, where ``quadric_collapse_candidates`` minimizes the summed quadric.
    s, r, placement = collapse_survivor(feature_count, u, v, is_boundary)
    if placement == COLLAPSE_REJECTED:
        return
    p = wp.lerp(vertices[u], vertices[v], 0.5)
    if placement == COLLAPSE_PINNED:
        p = vertices[s]
    # The degeneracy selection: any edge of the region up to the length bound, collapsed into the
    # endpoint whose removal costs less, and only if it leaves no face worse than both
    # ``COLLAPSE_MAX_NEW_ASPECT`` and the worst face it touched, unless the edge is tiny. A critical
    # face also lifts the fold veto: its normal is noise. Ranking the lock key by length, as a
    # serial shortest-first queue would, picked the same collapses on the graded saddle at 1.3x the
    # passes (factor-of-two buckets) and 3.1x (an exact rank), the key field being spatially
    # coherent, so the hashed key stays.
    check_folds = wp.bool(True)
    if select == COLLAPSE_SELECT_DEGENERATE:
        tiny = length <= degeneracy[0]
        if not tiny and length > degeneracy[1]:
            return
        if not edge_faces_in_region(faces, edge_face_count, edge_faces, k, low):
            return
        # Under a pinned placement the survivor is decided; a free one keeps the cheaper endpoint.
        cost_s, worst_before, worst_after = endpoint_collapse_cost(
            vertices, faces, vertex_face_offsets, vertex_faces, r, s
        )
        if placement == COLLAPSE_FREE:
            cost_r, before_r, after_r = endpoint_collapse_cost(
                vertices, faces, vertex_face_offsets, vertex_faces, s, r
            )
            if cost_r < cost_s:
                s, r = r, s
                worst_before = before_r
                worst_after = after_r
        p = vertices[s]
        if not tiny and worst_after > wp.max(COLLAPSE_MAX_NEW_ASPECT, worst_before):
            return
        check_folds = worst_before < degeneracy[2]

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
    if select == COLLAPSE_SELECT_LENGTH:
        for i in range(offsets[r], offsets[r + 1]):
            w = columns[i]
            if w != s and wp.length(p - vertices[w]) > vertex_band(high, high_uniform, w):
                return
        if placement == COLLAPSE_FREE:
            for i in range(offsets[s], offsets[s + 1]):
                w = columns[i]
                if w != r and wp.length(p - vertices[w]) > vertex_band(high, high_uniform, w):
                    return

    # The fold veto (``collapse_folds_a_face``), last because every test above rejects more
    # cheaply.
    #
    # Nothing else here notices a collapse that inverts an incident face: the link condition is
    # topological and the band walks above bound *lengths*. It is effectively free, because
    # rejecting a collapse early removes more downstream work than the vertex-face CSR costs.
    if check_folds and collapse_folds_a_face(
        vertices, faces, vertex_face_offsets, vertex_faces, s, r, p
    ):
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
    # (``scatter.endpoints_hold``) is equality against a neighbourhood minimum, so two candidates
    # sharing a key both win and both commit -- a conflict, a corrupted mesh rather than a
    # worse one. A masked ``lowbias32`` is exactly 2-to-1, so the index goes in the low half, which
    # disturbs nothing but a tie: that now goes to the lower index instead of to both. It is also
    # what lets both collapse paths run **one** lock pass rather than following it with a second,
    # raw-index one.
    hashed = wp.int64(lowbias32(wp.uint32(index)) & wp.uint32(0x7FFFFFFF))
    return (hashed << wp.int64(32)) | wp.int64(index)


# Cost buckets of the quadric pass's lock key (``bucketed_lock_key``): one per halving of the cost
# below the most expensive candidate in the pool, the rest sharing the cheapest bucket.
LOCK_KEY_COST_BUCKETS = wp.constant(wp.int32(24))


@wp.func
def bucketed_lock_key(index: wp.int32, cost: wp.float32, reference: wp.float32) -> wp.int64:
    # ``scramble_index`` with a coarse cost bucket in the top bits: within a neighbourhood the
    # cheaper collapse wins, which is what a serial priority queue does, while candidates of one
    # bucket still compete under the hash, so the key field keeps many local minima. A pure hash
    # spreads the error and lets expensive collapses go first; a pure cost rank is spatially
    # coherent on any smooth surface and commits a handful of collapses a pass.
    #
    # The bucket is the base-2 logarithm of ``cost / reference``, ``reference`` being the most
    # expensive candidate's cost: ties share a bucket, so a flat region, where every candidate
    # costs the same, stays hashed. A rank-based bucket does not, because tied costs rank in index
    # order. Bucket 0 also holds zero costs; the top bits stay clear of the sign.
    # Layout: 5 bits of bucket, 26 of hash, 32 of index -- still injective through the index.
    level = wp.int32(LOCK_KEY_COST_BUCKETS - 1)
    if reference > wp.float32(0.0) and cost < reference:
        level = wp.int32(0)
        if cost > wp.float32(0.0):
            level = wp.clamp(
                LOCK_KEY_COST_BUCKETS - 1 + wp.int32(wp.floor(wp.log2(cost / reference))),
                0,
                LOCK_KEY_COST_BUCKETS - 1,
            )
    hashed = wp.int64(lowbias32(wp.uint32(index)) & wp.uint32(0x3FFFFFF))
    return (wp.int64(level) << wp.int64(58)) | (hashed << wp.int64(32)) | wp.int64(index)


@wp.kernel(enable_backward=False)
def claim_collapse_key(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
) -> None:
    # The winning (smallest scrambled) key over the closed 1-rings of both endpoints, read back at
    # the two endpoints by ``commit_collapses`` (``scatter.endpoints_hold``): no two winners have
    # an endpoint in each other's closed 1-rings, which is exactly the conflict to exclude. The
    # quadric pass claims the same way inside ``drop_locked_and_claim``.
    #
    # The key is ``scramble_index(k)`` and not ``k`` for the reason that function records: a min-key
    # lock over a *spatially monotone* key field has essentially one local minimum, so it commits a
    # single collapse per pass however many candidates there are. Hashing it is what makes the
    # collapse stage track its target size on a structured mesh, which
    # ``test_collapse_pass_commits_a_useful_fraction_on_a_structured_patch`` asserts against; the
    # icosphere fixtures cannot see it, because there the split stage does all the work.
    #
    # ``locked``, when not empty, marks the closed 1-rings of the collapses an earlier round of the
    # same pass committed (``commit_collapses``): a candidate with an endpoint there is retired
    # until the next topology rebuild, the ``unlocked_candidate`` rule, and every other one is
    # still scored against unchanged faces, so it claims again.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    if locked.shape[0] > 0 and not endpoints_hold(locked, s, r, wp.int32(0)):
        survivor[k] = -1
        return
    lock_two_rings(offsets, columns, s, r, scramble_index(k), out_min_key)


@wp.kernel(enable_backward=False)
def commit_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    claim: wp.array[wp.int64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
    out_locked: wp.array[wp.int32],
) -> None:
    # A non-empty ``out_locked`` runs the pass in rounds: each commit retires its candidate and
    # stamps its closed 1-rings for the next round's ``claim_collapse_key``.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    if not endpoints_hold(claim, s, r, scramble_index(k)):
        return
    out_remap[r] = s
    out_positions[s] = pos[k]
    wp.atomic_add(out_count, 0, 1)
    if out_locked.shape[0] > 0:
        survivor[k] = -1
        stamp_two_rings(offsets, columns, s, r, wp.int32(1), out_locked)


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
def scatter_normal_and_area_sums(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_normal_sums: wp.array2d[wp.float64],
    out_area_sums: wp.array[wp.float64],
) -> None:
    # The smoothing pass's two per-vertex face sums from one evaluation of each face, over zeroed
    # ``float64`` sums: ``vertices.scatter_area_weighted_normals``' area-weighted normal and
    # ``scatter.scatter_face_areas_exact``'s area, each added exactly as those kernels add it. Their
    # readers finish them where they read them -- ``normalized_accumulated_row`` for the normal,
    # ``vertex_mass`` for the barycentric mass -- so neither is narrowed by a pass of its own.
    f = wp.int32(wp.tid())
    normal, area = face_normals_and_area(vertices, faces, f)
    value = normal * area
    add_to_face_corners(out_normal_sums, faces, f, value, value, value)
    exact = wp.float64(area)
    add_corner_triple(out_area_sums, faces, f, exact, exact, exact)


@wp.func
def vertex_mass(area_sums: wp.array[wp.float64], v: wp.int32) -> wp.float32:
    # Vertex ``v``'s barycentric mass from its exact incident-area sum: one rounding of a third,
    # ``scatter.scale_face_sums``' arithmetic and so ``laplacian.mass_matrix_entries``' bits.
    return wp.float32(area_sums[v] / wp.float64(3.0))


@wp.kernel
def accumulate_one_ring(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    area_sums: wp.array[wp.float64],
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
    area_u = vertex_mass(area_sums, u)
    area_v = vertex_mass(area_sums, v)
    wp.atomic_add(out_sum, u, area_v * vertices[v])
    wp.atomic_add(out_weight, u, area_v)
    wp.atomic_add(out_sum, v, area_u * vertices[u])
    wp.atomic_add(out_weight, v, area_u)


@wp.kernel
def gather_one_ring(
    ring_offsets: wp.array[wp.int32],
    ring_neighbors: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    area_sums: wp.array[wp.float64],
    out_sum: wp.array[wp.vec3],
    out_weight: wp.array[wp.float32],
) -> None:
    # ``accumulate_one_ring``'s sums as a gather over each vertex's neighbour row, which the caller
    # sorts ascending: one fixed summation order, so the same bits on every run. The scatter's
    # float atomics commit in arrival order, and on CUDA the last bits they move tip later split
    # and collapse decisions, until even ``isotropic_remesh``'s face count differs between runs.
    # The deterministic setting picks this kernel; the scatter is the fast form when it is off.
    # Two algorithms for one quantity, so two kernels: the scatter is one thread per edge.
    v = wp.int32(wp.tid())
    ring_sum = wp.vec3(0.0)
    ring_weight = wp.float32(0.0)
    for slot in range(ring_offsets[v], ring_offsets[v + 1]):
        w = ring_neighbors[slot]
        area = vertex_mass(area_sums, w)
        ring_sum += area * vertices[w]
        ring_weight += area
    out_sum[v] = ring_sum
    out_weight[v] = ring_weight


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


@wp.func
def reproject_vertices(
    vertex: wp.vec3, feature_count: wp.int32, mesh_id: wp.uint64, max_dist: wp.float32
) -> wp.vec3:
    # Snap a free vertex back onto the closest point of the original surface, undoing the drift the
    # smoothing pass introduces. Pinned vertices and failed queries keep their position.
    if finalize_vertex_codes(feature_count) != FREE_VERTEX:
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


@wp.kernel(enable_backward=False)
def smooth_free_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    feature_count: wp.array[wp.int32],
    normal_sums: wp.array2d[wp.float64],
    ring_sum: wp.array[wp.vec3],
    ring_weight: wp.array[wp.float32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    lam: wp.float32,
    reproject: wp.int32,
    mesh_id: wp.uint64,
    max_dist: wp.float32,
    clamp_mesh_id: wp.uint64,
    max_deviation: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # One tangential relaxation step, vetoed per vertex by the same fold rule the two collapse
    # candidates run -- one decision rule, one spelling. With ``reproject`` (warp-uniform) the
    # vertex's new position is then snapped onto the surface ``mesh_id`` by ``reproject_vertices``,
    # the step ``isotropic_remesh`` runs next at the same vertex, so the two take one launch; ``0``
    # leaves the relaxed position as it is and reads neither ``mesh_id`` nor ``max_dist``. A
    # positive ``max_deviation`` then pulls the result into the band about ``clamp_mesh_id``
    # (``clamp_to_surface_band``, the step after that, over every vertex), in the same launch.
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
        vertices[v],
        finalize_vertex_codes(feature_count[v]),
        normalized_accumulated_row(normal_sums, v),
        ring_sum[v],
        ring_weight[v],
        lam,
    )
    if move_flips_normal(vertices, faces, vertex_face_offsets, vertex_faces, v, -1, proposed):
        proposed = vertices[v]
    if reproject != 0:
        proposed = reproject_vertices(proposed, feature_count[v], mesh_id, max_dist)
    if max_deviation > 0.0:
        proposed = clamp_to_surface_band(proposed, clamp_mesh_id, max_deviation, max_dist)
    out_positions[v] = proposed


@wp.kernel
def pair_intrinsic_twins(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    out_twin: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_face_claim: wp.array[wp.int32],
) -> None:
    # The *one* point in ``intrinsic_delaunay``'s loop where a vertex-pair key is trustworthy: the
    # caller's input is still a simplicial complex, so an edge carrying exactly two halfedges (a
    # ``halfedge.halfedge_mates`` partner) is one interior edge, and its two halfedges are twins.
    # Twins are indexed ``h = 3 * f + e`` opposite corner ``e`` (``edge_lengths``' own indexing),
    # the corner found by first match on the face's opposite apex. Every flip after this one
    # maintains ``out_twin`` incrementally instead of re-deriving it, because a second flip can
    # create a second edge between two already-adjacent vertices, and at that point no vertex-pair
    # key can tell which of several same-key corners is which edge's true twin -- see
    # ``intrinsic_delaunay_candidates``. Launched over the halfedges; the pair's lower halfedge
    # writes both directions (the order the edge-key sort's stable runs held them in). Thread ``h``
    # also copies corner ``h`` into ``out_faces``, the face buffer the loop flips in place, and the
    # threads below ``n_faces`` arm the claim table for the loop's first round (``fixup_twin_remap``
    # re-arms it for every later one).
    h = wp.int32(wp.tid())
    out_faces[h] = faces[h]
    if h < out_face_claim.shape[0]:
        out_face_claim[h] = INT32_MAX_CONSTANT
    mate = mates[h]
    if mate <= h:
        return
    _a, _b, f0, f1, apex0, apex1 = edge_pair_topology(faces, h, mate)
    corner0 = local_corner(faces, f0, apex0)
    corner1 = local_corner(faces, f1, apex1)
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
def cluster_cell(
    vertex: wp.vec3, origin: wp.vec3, inverse_size: wp.vec3, cell_max: wp.vec3i
) -> wp.vec3i:
    # The cell ``remesh.cluster_decimate`` bins a vertex into: ``voxel_cell_axes`` clamped to
    # ``[0, cell_max]`` per axis. Only the box-fitted grid relies on the clamp, whose box is the
    # faces' vertices' own bounding box split into whole cells: a vertex on the box's maximum
    # lands in the last cell rather than one past it, and an unreferenced vertex off the box in a
    # border cell. The other two grids enclose every vertex strictly inside ``cell_bound`` and
    # pass ``cell_bound - 1``, so for them the clamp is the identity.
    cell = voxel_cell_axes(vertex, origin, inverse_size)
    return wp.max(wp.min(cell, cell_max), wp.vec3i(0))


@wp.func
def cluster_cell_key(
    vertex: wp.vec3,
    origin: wp.vec3,
    inverse_size: wp.vec3,
    cell_max: wp.vec3i,
    cell_bound: wp.uint64,
) -> wp.uint64:
    # A vertex's voxel cell packed into the ``uint64`` key ``remesh.cluster_decimate`` orders its
    # clusters by: ``voxels.cell_indices`` and ``grouping.hash_indices_rows`` in one thread, the
    # same cell (``voxel_cell_axes`` over three equal widths is ``voxel_cell``) and the same
    # packing, so the keys are theirs bit for bit. ``inverse_size`` is per axis for MeshLab's grid
    # and the box-fitted one, whose cells are stretched to fit a box. Pure in its inputs, which is
    # what lets ``cluster_insert_vertices`` recompute a stored vertex's key rather than keep a
    # table of keys.
    cell = cluster_cell(vertex, origin, inverse_size, cell_max)
    return pack_index_triple(cell[0], cell[1], cell[2], cell_bound)


@wp.kernel
def cluster_insert_vertices(
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.vec3,
    cell_max: wp.vec3i,
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
    key = cluster_cell_key(vertices[v], origin, inverse_size, cell_max, cell_bound)
    h = hash_slot(wp.int64(key), mask)
    while True:
        # Read before claiming: a cluster's vertices all probe its slot, and once it is taken a
        # plain load finds the occupant without the atomic (a stale ``-1`` only costs the CAS).
        prev = out_cell_table[h]
        if prev == wp.int32(-1):
            prev = wp.atomic_cas(out_cell_table, h, wp.int32(-1), v)
            if prev == wp.int32(-1):
                break
        if cluster_cell_key(vertices[prev], origin, inverse_size, cell_max, cell_bound) == key:
            break
        h = next_slot(h, mask)
    out_vertex_cell[v] = h


@wp.func
def oriented_cluster_triple(
    faces: wp.array[wp.int32], vertex_cell: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.bool]:
    # Face ``f``'s three clusters rotated to start at the smallest, and whether they are distinct:
    # the face's identity after welding, which ``cluster_mark_faces`` compares exactly. The
    # rotation keeps the winding, so a face and its reversed twin are two faces, as Open3D and
    # MeshLab both keep them.
    i0, i1, i2, distinct = remapped_corner_triple(faces, vertex_cell, f)
    s0, s1, s2 = rotate_min_first(i0, i1, i2)
    return s0, s1, s2, distinct


@wp.func
def claim_kept_cluster(
    cell: wp.int32,
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.vec3,
    cell_max: wp.vec3i,
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
        out_keys[p] = cluster_cell_key(
            vertices[cell_table[cell]], origin, inverse_size, cell_max, cell_bound
        )
        out_cells[p] = cell


@wp.kernel
def cluster_mark_faces(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.vec3,
    cell_max: wp.vec3i,
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
    # keyed on the oriented cluster triple and compared *exactly* by recomputing the stored face's
    # triple, lowered to the class's smallest face index with ``atomic_min``. A face that claims
    # an empty slot appends it at
    # the second cursor. ``out_cell_state``, ``out_face_table`` and ``out_counters`` arrive
    # initialised by the caller (``-1``, ``-1``, ``0``) and leave holding the answer.
    f = wp.int32(wp.tid())
    s0, s1, s2, distinct = oriented_cluster_triple(faces, vertex_cell, f)
    if not distinct:
        return
    claim_kept_cluster(
        s0,
        vertices,
        origin,
        inverse_size,
        cell_max,
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
        cell_max,
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
        cell_max,
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
        t0, t1, t2, _distinct = oriented_cluster_triple(faces, vertex_cell, prev)
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


# How ``cluster_decimate`` places each output vertex: a warp-uniform selector for the two kernels
# below, which differ by it only in the per-vertex half of their work. ``CLUSTER_AVERAGE_SORTED``
# is the mean under the deterministic setting: the vertices are staged for a stable sort by cluster
# (``stage_cluster_vertex``) and each cluster's run is added in one fixed order
# (``cluster_sorted_means`` and ``cluster_heavy_means``, which write the output positions), where
# ``CLUSTER_AVERAGE``'s atomics add them in arrival order. ``CLUSTER_CLOSEST_UNFUSED`` is
# ``CLUSTER_CLOSEST`` with the distance to the cell centre rounded one operation at a time
# (``unfused_cell_center_distance_sq``), the box-fitted grid's: ties at the centre, which a
# symmetric mesh has many of, then break as a serial single-precision evaluation breaks them, on
# both devices.
CLUSTER_AVERAGE = wp.constant(wp.int32(0))
CLUSTER_CLOSEST = wp.constant(wp.int32(1))
CLUSTER_AVERAGE_SORTED = wp.constant(wp.int32(2))
CLUSTER_CLOSEST_UNFUSED = wp.int32(3)


@wp.func
def accumulate_cluster_vertex(
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    v: wp.int32,
    out_sum: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # Vertex ``v`` into the sum and count of its kept cluster, indexed by output vertex. A cluster
    # no face keeps has no output slot and contributes nothing.
    r = vertex_cluster_rank(vertex_cell, cell_rank, v)
    if r >= 0:
        wp.atomic_add(out_sum, r, vertices[v])
        wp.atomic_add(out_count, r, 1)


@wp.func
def stage_cluster_vertex(
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    item: wp.int32,
    v: wp.int32,
    n_kept: wp.int32,
    out_staged: wp.array[wp.int32],
) -> None:
    # Item ``item`` (vertex ``v`` itself, or a face corner on it) as a key and payload of the stable
    # sort ``cluster_sorted_means`` reads: its kept cluster's output vertex (``n_kept``, past every
    # kept one, when none keeps it) at ``item`` and ``v`` at ``half + item``, the two halves of
    # ``out_staged`` being the sort's double-width keys and payload.
    r = vertex_cluster_rank(vertex_cell, cell_rank, v)
    out_staged[item] = wp.where(r >= 0, r, n_kept)
    out_staged[out_staged.shape[0] // 2 + item] = v


@wp.func
def mean_from_sum(total: wp.vec3, count: wp.int32) -> wp.vec3:
    # A cluster's mean position from ``accumulate_cluster_vertex``'s sum and count, dividing by the
    # count converted in place rather than by a separately materialised ``float32`` copy of the
    # counts.
    return total / wp.float32(count)


# The squared distance from ``position`` to the centre of ``cell`` (``origin + (cell + 0.5) *
# size`` per axis), every product and sum rounded on its own. On CUDA the round-to-nearest
# intrinsics are never contracted into an FMA, which ``wp.length_sq`` over ``voxel_cell_center``
# is (a centre and its distance each one rounding short), and which moved the pick on ties at the
# centre of an icosphere's cells (four of 128 at 10 % of the diagonal). The CPU device takes
# the same operations, one statement each, and its plain ``wp.length_sq`` already picked the
# reference's vertex on every probe. Summed ``x``, then ``y``, then ``z``.
_UNFUSED_CELL_CENTER_DISTANCE_SQ = """
float squares[3];
for (int axis = 0; axis < 3; ++axis) {
#if defined(__CUDA_ARCH__)
    float center = __fadd_rn(
        origin[axis], __fmul_rn(__fadd_rn((float)cell[axis], 0.5f), size[axis]));
    float offset = __fsub_rn(position[axis], center);
    squares[axis] = __fmul_rn(offset, offset);
#else
    float half_cell = (float)cell[axis] + 0.5f;
    float scaled = half_cell * size[axis];
    float center = origin[axis] + scaled;
    float offset = position[axis] - center;
    squares[axis] = offset * offset;
#endif
}
#if defined(__CUDA_ARCH__)
return __fadd_rn(__fadd_rn(squares[0], squares[1]), squares[2]);
#else
float xy = squares[0] + squares[1];
return xy + squares[2];
#endif
"""


@wp.func_native(_UNFUSED_CELL_CENTER_DISTANCE_SQ)
def unfused_cell_center_distance_sq(
    position: wp.vec3, origin: wp.vec3, size: wp.vec3, cell: wp.vec3i
) -> wp.float32: ...


@wp.func
def pick_closest_cluster_vertex(
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.vec3,
    cell_max: wp.vec3i,
    voxel_size: wp.vec3,
    unfused: wp.bool,
    v: wp.int32,
    out_closest: wp.array[wp.uint64],
) -> None:
    # Vertex ``v`` as a candidate for the "closest to the cell centre" representative of its kept
    # cluster, in one ``uint64`` ``atomic_min``: the squared distance's bits above the vertex index.
    # A non-negative ``float32`` orders as its bit pattern does, so the minimum is the winning
    # distance and, among the vertices tying for it, the lowest index -- a deterministic pick
    # whatever the arrival order. ``out_closest`` arrives holding ``UINT64_MAX``.
    r = vertex_cluster_rank(vertex_cell, cell_rank, v)
    if r >= 0:
        # The centre of the cell ``v`` was binned into, clamp included, so a vertex on the
        # box-fitted grid's maximum is measured against the last cell's centre.
        # ``unfused`` is ``CLUSTER_CLOSEST_UNFUSED``'s rounding.
        cell = cluster_cell(vertices[v], origin, inverse_size, cell_max)
        d = wp.where(
            unfused,
            unfused_cell_center_distance_sq(vertices[v], origin, voxel_size, cell),
            wp.length_sq(vertices[v] - voxel_cell_center(cell, origin, voxel_size)),
        )
        key = (wp.uint64(wp.cast(d, wp.uint32)) << wp.uint64(32)) | wp.uint64(wp.uint32(v))
        wp.atomic_min(out_closest, r, key)


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
def cluster_gather_and_face_keys(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.vec3,
    cell_max: wp.vec3i,
    voxel_size: wp.vec3,
    contraction: wp.int32,
    corner_items: wp.int32,
    face_table: wp.array[wp.int32],
    face_slots: wp.array[wp.int32],
    n_unique: wp.int32,
    n_kept: wp.uint64,
    out_sum: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
    out_closest: wp.array[wp.uint64],
    out_keys: wp.array[wp.uint64],
    out_faces: wp.array[wp.int32],
) -> None:
    # Two independent halves of the pass after the clusters are ranked, over ``max(n_items,
    # n_unique)`` threads:
    #
    # - item ``t`` into its cluster's representative: ``accumulate_cluster_vertex`` under
    #   ``CLUSTER_AVERAGE`` (``out_sum`` / ``out_count`` zeroed), staged for the ordered mean's sort
    #   under ``CLUSTER_AVERAGE_SORTED``, or ``pick_closest_cluster_vertex`` under
    #   ``CLUSTER_CLOSEST`` and ``CLUSTER_CLOSEST_UNFUSED``; the other contractions' outputs are
    #   empty. An item is a vertex (Open3D's mean, over every vertex of the cell) or, with
    #   ``corner_items`` set, a face corner (MeshLab's, over every face corner in the cell, so a
    #   vertex counts once per face on it; the box-fitted grid's, whose candidates are on a face);
    # - unique face ``t < n_unique``: its class representative (the smallest face index,
    #   ``cluster_mark_faces``) and its key, the output-vertex triple rotated to start at its
    #   smallest and packed in radix ``n_kept``, which the sort that follows orders faces by.
    t = wp.int32(wp.tid())
    n_items = wp.where(corner_items != 0, faces.shape[0], vertices.shape[0])
    if t < n_items:
        v = wp.where(corner_items != 0, faces[t], t)
        if contraction == CLUSTER_AVERAGE:
            accumulate_cluster_vertex(vertex_cell, cell_rank, vertices, v, out_sum, out_count)
        elif contraction == CLUSTER_AVERAGE_SORTED:
            stage_cluster_vertex(vertex_cell, cell_rank, t, v, wp.int32(n_kept), out_count)
        else:
            pick_closest_cluster_vertex(
                vertex_cell,
                cell_rank,
                vertices,
                origin,
                inverse_size,
                cell_max,
                voxel_size,
                contraction == CLUSTER_CLOSEST_UNFUSED,
                v,
                out_closest,
            )
    if t < n_unique:
        g = face_table[face_slots[t]]
        r0, r1, r2 = ranked_face_corners(faces, vertex_cell, cell_rank, g)
        s0, s1, s2 = rotate_min_first(r0, r1, r2)
        out_keys[t] = pack_index_triple(s0, s1, s2, n_kept)
        out_faces[t] = g


@wp.kernel
def cluster_emit(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    sorted_faces: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    contraction: wp.int32,
    sums: wp.array[wp.vec3],
    counts: wp.array[wp.int32],
    closest: wp.array[wp.uint64],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The output mesh, over ``max(n_kept, n_unique)`` threads: output vertex ``t`` -- its cluster's
    # mean (``mean_from_sum``) or its representative vertex's own position, the index the low half
    # of ``pick_closest_cluster_vertex``'s key -- and unique face ``t`` in key order, in its
    # representative's winding rotated to start at its smallest output vertex (Open3D's rows).
    # Under ``CLUSTER_AVERAGE_SORTED`` the vertices are ``cluster_sorted_means``' and only the faces
    # are written here.
    t = wp.int32(wp.tid())
    if t < out_vertices.shape[0]:
        if contraction == CLUSTER_AVERAGE:
            out_vertices[t] = mean_from_sum(sums[t], counts[t])
        elif contraction == CLUSTER_CLOSEST or contraction == CLUSTER_CLOSEST_UNFUSED:
            out_vertices[t] = vertices[wp.int32(closest[t] & wp.uint64(0xFFFFFFFF))]
    if t < sorted_faces.shape[0]:
        r0, r1, r2 = ranked_face_corners(faces, vertex_cell, cell_rank, sorted_faces[t])
        s0, s1, s2 = rotate_min_first(r0, r1, r2)
        write_corner_triple(out_faces, t, s0, s1, s2)


# How ``cluster_decimate``'s ordered mean splits its runs: a run of at most ``CLUSTER_SERIAL_MAX``
# vertices is added by one thread, a longer one by a block of ``CLUSTER_HEAVY_BLOCK_DIM`` lanes, at
# most ``CLUSTER_HEAVY_BLOCKS`` blocks striding the list of long runs. Swept on CUDA over cells of
# 2 and 6 mean edges and of 10 % and 30 % of the diagonal on ``dragon`` and ``lucy``: a block per
# cluster at every size cost 1.6-3.3x the atomic scatter on fine cells (32-256 lanes, mostly idle
# over a few vertices), a thread per cluster up to 40x on coarse ones (one serial walk over most of
# the mesh); split at 128 the ordered mean costs 1.05-1.17x the scatter on fine cells and 0.1-0.7x
# on coarse ones, where the scatter's atomics pile onto a few addresses. Serial bounds of 16-32
# handed too many short runs to blocks (up to 2.3x at ``lucy`` / 6 edges); 64-256 measured alike.
# The split decides each run's summation order, so it is a fixed constant, not a device query.
CLUSTER_SERIAL_MAX = 128
CLUSTER_HEAVY_BLOCK_DIM = 256
CLUSTER_HEAVY_BLOCKS = 4096


@wp.func
def cluster_run(sorted_clusters: wp.array[wp.int32], r: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Output vertex ``r``'s run ``[start, stop)`` in ``stage_cluster_vertex``'s keys once stably
    # sorted. Every kept cluster holds an item (a face keeps it, so a corner lies in it), so the
    # run is never empty.
    return binary_search_index_left(sorted_clusters, r), binary_search_index_left(
        sorted_clusters, r + 1
    )


@wp.func
def strided_run_sum(
    vertices: wp.array[wp.vec3],
    sorted_vertices: wp.array[wp.int32],
    first: wp.int32,
    stop: wp.int32,
    stride: wp.int32,
) -> wp.vec3:
    # Every ``stride``-th vertex of a sorted run from ``first``, added in ascending vertex order
    # from zero: the whole run at stride one, one lane's share of it at the block's width.
    total = wp.vec3(0.0)
    for i in range(first, stop, stride):
        total += vertices[sorted_vertices[i]]
    return total


@wp.kernel
def cluster_sorted_means(
    vertices: wp.array[wp.vec3],
    sorted_clusters: wp.array[wp.int32],
    sorted_vertices: wp.array[wp.int32],
    serial_max: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_heavy_count: wp.array[wp.int32],
    out_heavy: wp.array[wp.int32],
) -> None:
    # ``CLUSTER_AVERAGE_SORTED``'s output positions, a thread per kept cluster: a run of at most
    # ``serial_max`` vertices is added here, in ascending vertex order, and a longer one is appended
    # to ``out_heavy`` (``out_heavy_count`` caller-zeroed) for ``cluster_heavy_means``. A cluster
    # can hold most of the mesh at a coarse voxel size, which one thread would walk serially. The
    # list's arrival order reaches no sum: each heavy run is added by one block in its own fixed
    # order, whichever block takes it. On the CPU device that order is the ascending one too, the
    # order its atomics commit ``accumulate_cluster_vertex`` in, so the CPU bits are
    # ``CLUSTER_AVERAGE``'s. The exact group sum (``kernels/reduce.py``, no sort) was priced
    # against the sort and these two kernels: 0.5x the wall time at 100 k vertices, but 1.0-1.8x on
    # 512 coarse cells of 2 M vertices (cross-run spread) and 1.2-1.6x on 1.8 M fine cells, so
    # this form stays.
    r = wp.int32(wp.tid())
    start, stop = cluster_run(sorted_clusters, r)
    if stop - start > serial_max:
        out_heavy[wp.atomic_add(out_heavy_count, 0, wp.int32(1))] = r
        return
    total = strided_run_sum(vertices, sorted_vertices, start, stop, wp.int32(1))
    out_vertices[r] = mean_from_sum(total, stop - start)


@wp.kernel
def cluster_heavy_means(
    vertices: wp.array[wp.vec3],
    sorted_clusters: wp.array[wp.int32],
    sorted_vertices: wp.array[wp.int32],
    heavy_count: wp.array[wp.int32],
    heavy: wp.array[wp.int32],
    n_blocks: wp.int32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    # The runs ``cluster_sorted_means`` left, a block per run over ``n_blocks`` blocks striding the
    # list: each lane adds every ``wp.block_dim()``-th vertex of the run in ascending order, and the
    # block folds the lanes' sums in its fixed tree. The CPU device runs one lane per block, which
    # adds the run in ascending vertex order like the serial path. The list's length is uniform
    # over a block, so every lane reaches the fold.
    b, lane = wp.tid()
    for k in range(b, heavy_count[0], n_blocks):
        r = heavy[k]
        start, stop = cluster_run(sorted_clusters, r)
        total = block_sum(
            strided_run_sum(vertices, sorted_vertices, start + lane, stop, wp.block_dim())
        )
        if lane == 0:
            out_vertices[r] = mean_from_sum(total, stop - start)


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
    faces: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Which faces survive a vertex remap (vertex clustering's cell labels, an edge collapse's
    # survivor map) and which targets they still name, so both compactions are one scan of
    # ``out_flags`` rather than a face gather and a sort-based dedup of the corners. The remapped
    # corners are not stored: ``compact_surviving_faces`` re-reads them through ``remap`` for the
    # faces that survive. ``remap_and_mark_faces`` is the same marking with the corners written.
    # A ``state`` holds the live face count of a fixed-width face buffer, whose dead rows keep the
    # zeroed flags they arrive with (``emit_edge_topology``).
    f = wp.int32(wp.tid())
    if state.shape[0] > 0:
        if f >= state[COLLAPSE_PASS_FACES]:
            return
    mark_surviving_face(faces, remap, f, out_flags)


@wp.kernel
def compact_surviving_faces(
    faces: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    key_base: wp.uint64,
    out_faces: wp.array[wp.int32],
    out_totals: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_low: wp.array[wp.float32],
    out_high: wp.array[wp.float32],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # A collapse pass's whole compaction, over ``max(n_faces, n_targets)`` threads. ``face_offsets``
    # and ``ranks`` are the two halves of one inclusive scan over ``mark_surviving_faces``' face
    # flags followed by its target marks, behind a leading zero, each read as an exclusive scan with
    # its total at the end (``scanned_slot``).
    #
    # - Face ``t``, when it survives, is written at its dense row, so the rows keep the input's
    #   face order -- what a face-mask ``submesh`` of the remapped faces returns -- with every
    #   corner renumbered onto the compacted targets, which keep their order too. The last face
    #   thread publishes both totals (kept targets, surviving faces) side by side, so the host
    #   reads them in one copy.
    # - Target ``t``, when a surviving face names it, is written with its two length bands at its
    #   compacted slot -- the order ``repair.remove_unreferenced_vertices`` keeps and gathering the
    #   bands by its inverse reproduces. Empty bands (a uniform target, ``None`` at launch) are not
    #   written. The outputs are sized for every target, so this runs before the host knows the
    #   count; the caller keeps their live prefix.
    # - With ``out_keys`` (empty skips it), a surviving face also writes its edge keys packed
    #   against ``key_base`` and their identity payload into the next pass's sort buffers
    #   (``adjacency.write_face_edge_keys_and_order``), the launch that pass would open with, and
    #   every face row past the survivors writes ``EDGE_KEY_PAD``, which sorts after every real
    #   key: the next sort runs over the whole width, and its live prefix is exactly the sort of
    #   the live corners.
    t = wp.int32(wp.tid())
    n_faces = face_offsets.shape[0] - 1
    if t < ranks.shape[0] - 1:
        slot, kept = scanned_slot(ranks, t)
        if kept:
            out_positions[slot] = positions[t]
            if out_low.shape[0] > 0:
                out_low[slot] = low[t]
            if out_high.shape[0] > 0:
                out_high[slot] = high[t]
    if t >= n_faces:
        return
    if t == n_faces - 1:
        out_totals[0] = ranks[ranks.shape[0] - 1] - ranks[0]
        out_totals[1] = face_offsets[n_faces]
    if out_keys.shape[0] > 0 and t >= face_offsets[n_faces]:
        for k in range(3):
            out_keys[3 * t + k] = EDGE_KEY_PAD
            out_order[3 * t + k] = 3 * t + k
    row, survived = scanned_slot(face_offsets, t)
    if not survived:
        return
    i0, i1, i2, _distinct = remapped_corner_triple(faces, remap, t)
    s0, _k0 = scanned_slot(ranks, i0)
    s1, _k1 = scanned_slot(ranks, i1)
    s2, _k2 = scanned_slot(ranks, i2)
    write_corner_triple(out_faces, row, s0, s1, s2)
    if out_keys.shape[0] > 0:
        write_face_edge_keys_and_order(out_faces, row, key_base, out_keys, out_order)


# Objective for ``objective_flip_candidates``. A warp-uniform kernel argument rather than a
# ``wp.Function``, so both predicates share one compiled module.
OBJECTIVE_PLANARITY = wp.constant(wp.int32(0))  # improve triangle shape on a near-planar quad
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
        # A T-vertex shows up as a sliver whose apex sits (nearly) on its longest edge: the edge
        # flipped is that longest edge, and the sliver is judged by its longest edge over the height
        # onto it, ``1 / triangle_area_max_side`` -- MeshLab's ``meshing_remove_t_vertices`` rule
        # and ratio. A flip is taken when it raises the worse ``height / longest`` of the pair.
        diagonal_sq = wp.length_sq(cp - ap)
        abc_longest = diagonal_sq >= wp.max(wp.length_sq(bp - ap), wp.length_sq(cp - bp))
        acd_longest = diagonal_sq >= wp.max(wp.length_sq(dp - ap), wp.length_sq(dp - cp))
        quality_abc = triangle_area_max_side(ap, bp, cp)
        quality_acd = triangle_area_max_side(ap, cp, dp)
        sliver_quality = wp.float32(1.0) / aspect_threshold
        sliver = (abc_longest and quality_abc < sliver_quality) or (
            acd_longest and quality_acd < sliver_quality
        )
        if not sliver:
            return
        old_worst = wp.min(quality_abc, quality_acd)
        new_worst = wp.min(triangle_area_max_side(ap, bp, dp), triangle_area_max_side(dp, bp, cp))
        out_flip[k] = new_worst > old_worst
        return

    # Planarity: only rewrite a quad that is flat enough for the rewrite not to change the surface.
    # The gate is on the *cosine* of the dihedral so the kernel needs no inverse trigonometry.
    if wp.dot(triangle_normal(ap, bp, cp), triangle_normal(ap, cp, dp)) < planar_cos:
        return
    old_score = wp.min(triangle_quality(ap, bp, cp, metric), triangle_quality(ap, cp, dp, metric))
    new_score = wp.min(triangle_quality(ap, bp, dp, metric), triangle_quality(dp, bp, cp, metric))
    out_flip[k] = new_score > old_score * (1.0 + OBJECTIVE_EPS)


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

# Weight of a boundary edge's constraint plane relative to its squared length
# (``add_boundary_quadric``).
BOUNDARY_QUADRIC_WEIGHT = wp.constant(wp.float64(1.0))

# A collapse -- and a smoothing move -- is rejected when it would turn an incident face's normal by
# more than this. 0.5 is 60 degrees per step.
#
# The guard is *per step* and relative to the face as it stands, so it bounds each collapse's turn,
# not a face's turn since the input: over several passes a face can turn further than any one step
# allowed. At 0.2 (78 degrees) that drift folded a face of the graded height field in
# ``test_collapse_pass_vetoes_a_collapse_that_would_fold_a_face`` once the collapse stage committed
# more per pass; at 0.5 neither decimator stops sooner on any mesh measured, and the quadric one's
# Hausdorff distance fell. **Not exposed as a keyword**: nothing measured binds on it, and loosening
# it is what lets inverted, self-intersecting triangles through.
COLLAPSE_MIN_NORMAL_DOT = wp.constant(wp.float32(0.5))

# ``collapse_joins_opposite_sheets`` lets a collapse weld two opposing sheets when it removes a face
# below this mean ratio. Measured over the 15-cell decimation set (five meshes x 0.5 / 0.1 / 0.01,
# geometric means against no thin-part veto, three tie perturbations each): 0.05 and 0.1 agree
# within the perturbation noise (``i2o`` p99.9 0.93 / 0.94x, 1st-percentile quality 1.00 / 1.00x);
# 0.2 is worse on both (``i2o`` p99.9 0.95x, quality 0.99x).
COLLAPSE_SLIVER_QUALITY = wp.float32(0.05)


# A quadric is symmetric, so it is stored as its ten distinct entries rather than as a
# ``wp.mat44d``: ``[a00, a01, a02, a11, a12, a22, b0, b1, b2, c]`` for ``Q = [[A, b], [b^T, c]]``,
# so that ``[p, 1]^T Q [p, 1] = p^T A p + 2 b.p + c``. 80 bytes where the matrix is 128, and every
# per-vertex quadric is *written* by one thread (``gather_vertex_quadrics``) rather than accumulated
# by float64 atomics.
SymmetricQuadric = wp.types.vector(length=10, dtype=wp.float64)


@wp.func
def plane_quadric(normal: wp.vec3d, offset: wp.float64, weight: wp.float64) -> SymmetricQuadric:
    # Garland-Heckbert fundamental quadric of the plane ``dot(normal, x) + offset = 0``, scaled by
    # ``weight``, so that ``quadric_error`` is the weighted squared distance to the plane: ``A`` is
    # ``n n^T``, ``b`` is ``offset * n`` and ``c`` is ``offset^2``.
    a = weight * normal[0]
    b = weight * normal[1]
    c = weight * normal[2]
    d = weight * offset
    return SymmetricQuadric(
        a * normal[0],
        a * normal[1],
        a * normal[2],
        b * normal[1],
        b * normal[2],
        c * normal[2],
        a * offset,
        b * offset,
        c * offset,
        d * offset,
    )


@wp.func
def quadric_error(quadric: SymmetricQuadric, p: wp.vec3d) -> wp.float64:
    # ``[p, 1]^T Q [p, 1]``: the accumulated squared distance from ``p`` to every plane folded into
    # ``Q``. Clamped at zero, since a float64 sum of positive-semidefinite terms can still land a
    # hair below it and a negative "error" would sort ahead of every real candidate.
    x = p[0]
    y = p[1]
    z = p[2]
    two = wp.float64(2.0)
    quadratic = (
        quadric[0] * x * x
        + quadric[3] * y * y
        + quadric[5] * z * z
        + two * (quadric[1] * x * y + quadric[2] * x * z + quadric[4] * y * z)
    )
    linear = two * (quadric[6] * x + quadric[7] * y + quadric[8] * z)
    return wp.max(wp.float64(0.0), quadratic + linear + quadric[9])


@wp.func
def quadric_optimum(quadric: SymmetricQuadric, fallback: wp.vec3d) -> wp.vec3d:
    # Position minimizing the quadric: solve ``A p = -b``. ``fallback`` (the edge midpoint) is
    # returned when ``A`` is singular relative to its own scale, which is the planar case -- there
    # the minimum is a whole plane and inverting a near-singular matrix would place the vertex
    # arbitrarily far away.
    a = wp.mat33d(
        quadric[0],
        quadric[1],
        quadric[2],
        quadric[1],
        quadric[3],
        quadric[4],
        quadric[2],
        quadric[4],
        quadric[5],
    )
    scale = wp.abs(quadric[0]) + wp.abs(quadric[3]) + wp.abs(quadric[5])
    if scale <= wp.float64(0.0):
        return fallback
    if wp.abs(wp.determinant(a)) <= QUADRIC_SINGULAR_EPS * scale * scale * scale:
        return fallback
    b = wp.vec3d(quadric[6], quadric[7], quadric[8])
    return -(wp.inverse(a) * b)


@wp.func
def face_quadric(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> SymmetricQuadric:
    # Area-weighted plane quadric of face ``f``; zero for a degenerate face. Area weighting is
    # Garland-Heckbert's: a large triangle constrains its vertices more than a sliver does.
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    cross = wp.cross(v1 - v0, v2 - v0)
    double_area = wp.length(cross)
    if double_area <= wp.float64(0.0):
        return SymmetricQuadric()
    normal = cross / double_area
    return plane_quadric(normal, -wp.dot(normal, v0), double_area * wp.float64(0.5))


@wp.func
def boundary_quadric(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face: wp.int32, u: wp.int32, v: wp.int32
) -> SymmetricQuadric:
    # Garland-Heckbert's boundary constraint: the plane through boundary edge ``(u, v)``
    # perpendicular to its one face ``face``, weighted by ``BOUNDARY_QUADRIC_WEIGHT`` times the
    # squared edge length (an area, as the face quadrics' weights are). Both endpoints add it;
    # ``u`` and ``v`` are the edge's row in ``unique_edges``, so the two see the same bits.
    #
    # The face planes alone say nothing about where the boundary runs within the surface, so a
    # collapse that drags a rim inwards -- or cuts a corner of an open patch -- costs nothing, and
    # boundaries were eroded first. Worse, a rim vertex's quadric is nearly singular along the
    # rim's outward direction, and its minimizer slid there: every vertex an open mesh lost far
    # off its surface (a sizeable fraction of the bounding-box diagonal on a scan with holes) was a
    # rim vertex. This plane constrains exactly that direction.
    #
    # A truncated-pseudoinverse placement (minimize only along well-conditioned eigenvectors, keep
    # the midpoint along the rest) was built for the same symptom and measured redundant once this
    # constraint exists: no closed fixture or scan showed a spike without it, and it moved deviation
    # by a few percent either way.
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, face)
    p = to_vec3d(vertices[u])
    edge = to_vec3d(vertices[v]) - p
    normal = wp.cross(edge, wp.cross(v1 - v0, v2 - v0))
    length = wp.length(normal)
    if length <= wp.float64(0.0):
        return SymmetricQuadric()
    normal = normal / length
    return plane_quadric(normal, -wp.dot(normal, p), BOUNDARY_QUADRIC_WEIGHT * wp.dot(edge, edge))


# One decimation pass is replayed as a CUDA graph, and a replay pays a few microseconds of device
# time per graph *node* on top of the node's own work, so the pass is written as few launches as
# its data dependencies allow: every kernel below does all the work that can share its grid
# barrier. The sequence, with the grid barrier each boundary stands for:
#
#   begin_decimation_pass -> sort corner keys -> mark_unique_edge_starts -> scan ->
#   emit_pass_edges -> count_pass_edges -> scan -> scatter_pass_adjacency ->
#   gather_vertex_quadrics -> quadric_collapse_candidates -> sort costs -> drop_past_half ->
#   [drop_locked_and_claim -> mark_collapse_winners -> scan -> commit_budgeted_collapses ->
#    end_collapse_round]* -> remap_and_mark_faces -> scan -> compact_decimation_pass
#
# **The per-pass adjacency is one buffer.** The vertex-vertex CSR and the vertex-face CSR are both
# "count per vertex, scan, scatter", so their counts sit side by side in one array (vertex-vertex
# rows first, then vertex-face rows at ``vertex_capacity``) and one exclusive scan yields both
# offset tables and the two payloads' positions in one shared buffer. Neither CSR's row order is
# sorted -- a row is filled in atomic arrival order -- and nothing reads it in order: the link
# condition counts, the lock and the claim take unions and minima, and the flip veto is an any.
#
# **The vertex quadrics are gathered, not scattered.** Each face writes its plane quadric once
# (``begin_decimation_pass``) and each vertex sums its faces' in one thread
# (``gather_vertex_quadrics``), which costs a graph node over scattering each face's quadric onto
# its corners with float64 atomics, and buys a fixed summation order: the sum is reproducible, so
# the pass is. The scatter's atomics were the pass's largest kernel at scale; on the quadric
# region (``begin`` / ``count`` / ``scatter`` / ``gather``) the gather measured 2.35x at
# ``dragon`` and 2.51x at ``lucy``, 1.07-1.18x on 35-69 k faces, and the whole call 1.13-1.24x
# from ``dragon`` up, level (0.99-1.03x) below. The symmetric 10-double storage is part of it: a
# ``mat44d`` quadric moves 128 bytes where 80 suffice.


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
    out_face_quadrics: wp.array[SymmetricQuadric],
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
    # Each live face's plane quadric is written here too, from the positions the pass starts with:
    # ``gather_vertex_quadrics`` sums them per vertex once the vertex-face rows exist.
    #
    # Everything else is the state the rest of the pass accumulates into or starts from, reset here
    # rather than by one memset or copy each: the edge incidence and adjacency counts, the feature
    # counts, the round loop's locks, claim keys and collapse map, the working
    # positions, the compaction's vertex marks, and the loop state (``end_collapse_round``'s slot
    # table, seeded with its condition at 1 because ``wp.capture_while`` tests it before the first
    # round) and the pass's commit and removed-face counts.
    t = wp.int32(wp.tid())
    if t < faces.shape[0] // 3:
        c = t * 3
        if t >= state[DECIMATION_FACES]:
            out_keys[c + 0] = EDGE_KEY_PAD
            out_keys[c + 1] = EDGE_KEY_PAD
            out_keys[c + 2] = EDGE_KEY_PAD
        else:
            write_face_edge_keys(faces, t, c, base, out_keys)
            out_face_quadrics[t] = face_quadric(vertices, faces, t)
        out_order[c + 0] = c + 0
        out_order[c + 1] = c + 1
        out_order[c + 2] = c + 2
    if t < out_edge_face_count.shape[0]:
        out_edge_face_count[t] = 0
    if t < out_adjacency_counts.shape[0]:
        out_adjacency_counts[t] = 0
    if t < out_remap.shape[0]:
        out_feature_count[t] = 0
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
        out_round_state[COLLAPSE_REMOVED_FACES] = 0
        out_state[DECIMATION_COMMITS] = 0
        out_state[DECIMATION_REMOVED_FACES] = 0


# The pass's exclusive scans (``remesh._ExclusiveScan``). ``wp.utils.array_scan`` allocates its
# scratch on every call, which a ``wp.capture_while`` body may not do -- and the round loop ranks
# its winners with a scan -- and which puts a memory node into any other captured graph. So the
# pass carries its own one-launch scan over preallocated buffers: each block scans one
# ``SCAN_CHUNK`` of the input through a loaded tile and publishes the chunk's total, and the block
# that finishes last scans those totals (``finish_chunk_scan``), where a second one-block launch
# used to. A reader adds the two (``scanned_prefix``). ``wp.tile_load`` and
# ``wp.tile_store`` are lane-independent and bounds-checked, so the chunk kernels are correct on
# the CPU device, where a tiled launch runs one lane, and on a ragged last chunk.
#
# 512 x 256 lanes is the best of a sweep over chunks of 256 to 4096 and 128 to 512 lanes: 1.05x
# over 2048 on a 35k-face decimation pass, flat on 870k faces, where 512 lanes lost 1.18x.
SCAN_CHUNK = wp.constant(512)
SCAN_BLOCK_DIM = 256


@wp.func
def scan_chunk_totals(
    lane: wp.int32, chunk_totals: wp.array[wp.int32], out_chunk_offsets: wp.array[wp.int32]
) -> None:
    # Every chunk total's exclusive scan, by the whole block. Each lane owns a contiguous run sized
    # by ``wp.block_dim()``, so on the CPU device the one lane walks them all and the tile scan over
    # its single element is its own zero.
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
def finish_chunk_scan(
    chunk: wp.int32,
    lane: wp.int32,
    total: wp.int32,
    out_chunk_totals: wp.array[wp.int32],
    out_chunk_offsets: wp.array[wp.int32],
    ticket: wp.array[wp.int32],
) -> None:
    # A chunk block's tail: lane 0 publishes the chunk's ``total`` and draws a ticket, and the block
    # that draws the last one scans every chunk's total (``scan_chunk_totals``) and re-arms the
    # ticket, so one launch is the whole scan (the ``kernels/reduce`` last-block pattern; on the CPU
    # device the blocks run in turn and the fences compile to nothing).
    if lane == 0:
        out_chunk_totals[chunk] = total
    if block_last_ticket(lane, ticket, 0, wp.int32(0), out_chunk_totals.shape[0], last_ticket):
        thread_fence()
        scan_chunk_totals(lane, out_chunk_totals, out_chunk_offsets)
        if lane == 0:
            ticket[0] = 0


@wp.kernel
def scan_chunks_exclusive(
    values: wp.array[wp.int32],
    ticket: wp.array[wp.int32],
    out_prefix: wp.array[wp.int32],
    out_chunk_totals: wp.array[wp.int32],
    out_chunk_offsets: wp.array[wp.int32],
) -> None:
    chunk, lane = wp.tid()
    counts = wp.tile_load(values, shape=SCAN_CHUNK, offset=chunk * SCAN_CHUNK)
    wp.tile_store(out_prefix, wp.tile_scan_exclusive(counts), offset=chunk * SCAN_CHUNK)
    finish_chunk_scan(
        chunk, lane, wp.tile_sum(counts)[0], out_chunk_totals, out_chunk_offsets, ticket
    )


@wp.kernel
def scan_word_counts_exclusive(
    words: wp.array[wp.uint32],
    ticket: wp.array[wp.int32],
    out_prefix: wp.array[wp.int32],
    out_chunk_totals: wp.array[wp.int32],
    out_chunk_offsets: wp.array[wp.int32],
) -> None:
    # ``scan_chunks_exclusive`` over the set-bit counts of a bitmask's words. The two differ only in
    # the ``popcount32`` applied to the loaded tile.
    chunk, lane = wp.tid()
    counts = wp.tile_map(
        popcount32, wp.tile_load(words, shape=SCAN_CHUNK, offset=chunk * SCAN_CHUNK)
    )
    wp.tile_store(out_prefix, wp.tile_scan_exclusive(counts), offset=chunk * SCAN_CHUNK)
    finish_chunk_scan(
        chunk, lane, wp.tile_sum(counts)[0], out_chunk_totals, out_chunk_offsets, ticket
    )


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
    # which collapses to one lane per block on the CPU device. The saving
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
    # edge numbering -- and so the hash in the lock keys -- identical to the composed path),
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


@wp.func
def count_pass_edge(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    feature_angle: wp.float32,
    e: wp.int32,
    tally_boundary: wp.int32,
    out_feature_count: wp.array[wp.int32],
    out_adjacency_counts: wp.array[wp.int32],
    out_edge_slots: wp.array[wp.vec2i],
    out_boundary_quadrics: wp.array[SymmetricQuadric],
    out_boundary_flags: wp.array[wp.int32],
) -> None:
    # Live unique edge ``e``'s share of the pass, once the incidence is complete: its feature vote
    # (the topology fill's count, ``is_feature_edge``), its two entries' slots in the vertex-vertex
    # CSR rows, and a boundary edge's constraint plane (``boundary_quadric``), added to both
    # endpoints' entries of a separate table and flagged there for ``gather_vertex_quadrics``,
    # which folds the table into the vertex quadrics and leaves it zeroed for the next pass. Two
    # terms added to zero give the same sum in either order, so the table is the same run to run
    # wherever no vertex ends three or more boundary edges -- every vertex of a manifold boundary.
    # ``tally_boundary`` counts each endpoint's boundary edges into its flag instead of setting it,
    # which is how ``gather_vertex_quadrics_exact`` finds the vertices whose table it must not
    # read. A degenerate edge ``(a, a)`` from a face with a repeated corner is one entry in row
    # ``a``, as the triplet build this replaces accumulated its two identical triplets into one.
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    if is_feature_edge(vertices, faces, edge_face_count, edge_faces, e, feature_angle):
        wp.atomic_add(out_feature_count, u, 1)
        wp.atomic_add(out_feature_count, v, 1)
    if edge_face_count[e] == 1 and u != v:
        plane = boundary_quadric(vertices, faces, edge_faces[e, 0], u, v)
        wp.atomic_add(out_boundary_quadrics, u, plane)
        wp.atomic_add(out_boundary_quadrics, v, plane)
        if tally_boundary != 0:
            wp.atomic_add(out_boundary_flags, u, 1)
            wp.atomic_add(out_boundary_flags, v, 1)
        else:
            out_boundary_flags[u] = 1
            out_boundary_flags[v] = 1
    slot_u = wp.atomic_add(out_adjacency_counts, u, 1)
    slot_v = wp.int32(-1)
    if u != v:
        slot_v = wp.atomic_add(out_adjacency_counts, v, 1)
    out_edge_slots[e] = wp.vec2i(slot_u, slot_v)


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
    out_boundary_quadrics: wp.array[SymmetricQuadric],
    out_boundary_flags: wp.array[wp.int32],
) -> None:
    # ``count_pass_edge`` per live unique edge, flagging boundary vertices.
    # ``count_pass_edges_exact`` is the deterministic setting's form, counting them.
    e = wp.int32(wp.tid())
    if e >= state[DECIMATION_EDGES]:
        return
    count_pass_edge(
        vertices,
        faces,
        unique_edges,
        edge_face_count,
        edge_faces,
        feature_angle,
        e,
        wp.int32(0),
        out_feature_count,
        out_adjacency_counts,
        out_edge_slots,
        out_boundary_quadrics,
        out_boundary_flags,
    )


@wp.kernel
def count_pass_edges_exact(
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
    out_boundary_quadrics: wp.array[SymmetricQuadric],
    out_boundary_flags: wp.array[wp.int32],
    out_hub_count: wp.array[wp.int32],
) -> None:
    # ``count_pass_edges`` with each boundary vertex's flag holding its boundary-edge count, for
    # ``gather_vertex_quadrics_exact``, which also lists this pass's hub vertices from the count
    # cleared here.
    e = wp.int32(wp.tid())
    if e == 0:
        out_hub_count[0] = 0
    if e >= state[DECIMATION_EDGES]:
        return
    count_pass_edge(
        vertices,
        faces,
        unique_edges,
        edge_face_count,
        edge_faces,
        feature_angle,
        e,
        wp.int32(1),
        out_feature_count,
        out_adjacency_counts,
        out_edge_slots,
        out_boundary_quadrics,
        out_boundary_flags,
    )


@wp.kernel
def scatter_pass_adjacency(
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
) -> None:
    # Launched over the corners or the offset table, whichever is longer. Thread ``t`` places live
    # edge ``t``'s two vertex-vertex entries and live corner ``t``'s vertex-face entry at the slots
    # the counting kernels drew, and writes entry ``t`` of the offset table the scan left split in
    # two -- independent work that only needs the counts, so it shares this one launch.
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


# Rows longer than this are summed in their stored order instead of ascending face order:
# ``gather_vertex_quadrics`` finds each next face by a rescan of the row (``array.next_row_entry``),
# quadratic in its length, which a hub vertex (a cone apex with tens of thousands of faces) cannot
# afford. Under the deterministic setting ``gather_hub_quadrics`` sums such a row exactly instead.
QUADRIC_ORDERED_ROW_MAX = wp.constant(64)


@wp.func
def ordered_face_quadric_sum(
    face_quadrics: wp.array[SymmetricQuadric],
    vertex_faces: wp.array[wp.int32],
    lo: wp.int32,
    hi: wp.int32,
) -> SymmetricQuadric:
    # The faces' planes of the vertex-face row ``vertex_faces[lo:hi]``, summed in **ascending face
    # order** -- the row itself is in atomic arrival order -- so the sum is the same run to run on
    # every device; a row longer than ``QUADRIC_ORDERED_ROW_MAX`` is summed as stored.
    ordered = hi - lo <= QUADRIC_ORDERED_ROW_MAX
    q = SymmetricQuadric()
    last = wp.int32(-1)
    slot = wp.int32(-1)
    for j in range(lo, hi):
        f = vertex_faces[j]
        if ordered:
            last, slot = next_row_entry(vertex_faces, lo, hi, last, slot)
            f = last
        q = q + face_quadrics[f]
    return q


@wp.func
def take_boundary_quadric(
    boundary_quadrics: wp.array[SymmetricQuadric], boundary_flags: wp.array[wp.int32], v: wp.int32
) -> SymmetricQuadric:
    # The boundary planes ``count_pass_edges`` accumulated for flagged vertex ``v``, with its entry
    # and flag reset for the next pass: the table is persistent scratch whose zero state is the
    # invariant, not an output, which is why neither argument wears the output prefix.
    q = boundary_quadrics[v]
    boundary_quadrics[v] = SymmetricQuadric()
    boundary_flags[v] = 0
    return q


@wp.kernel
def gather_vertex_quadrics(
    face_quadrics: wp.array[SymmetricQuadric],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    boundary_quadrics: wp.array[SymmetricQuadric],
    boundary_flags: wp.array[wp.int32],
    out_quadrics: wp.array[SymmetricQuadric],
) -> None:
    # Each vertex's quadric: its faces' planes, summed by one thread over its vertex-face row
    # (``ordered_face_quadric_sum``), so every cost and placement read from it is the same run to
    # run on every device, then the boundary planes ``count_pass_edges`` accumulated for it, whose
    # entry is reset here for the next pass. A degenerate face that names ``v`` twice is visited
    # once per occurrence; its plane is zero, so it adds nothing either way. The dummy vertex's row
    # is empty, so its quadric is zero, as is every unreferenced vertex's.
    #
    # Two sums stay in arrival order on CUDA: a row longer than ``QUADRIC_ORDERED_ROW_MAX``, and
    # the boundary table of a vertex where three or more boundary edges meet. The form for
    # ``ordito.config``'s deterministic setting off; ``gather_vertex_quadrics_exact`` is the form
    # for it on.
    v = wp.int32(wp.tid())
    q = ordered_face_quadric_sum(
        face_quadrics, vertex_faces, vertex_face_offsets[v], vertex_face_offsets[v + 1]
    )
    if boundary_flags[v] != 0:
        q = q + take_boundary_quadric(boundary_quadrics, boundary_flags, v)
    out_quadrics[v] = q


@wp.func
def boundary_edge_at(
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    n_edges: wp.int32,
    base: wp.uint64,
    v: wp.int32,
    w: wp.int32,
) -> wp.int32:
    # The row of edge ``(v, w)`` when it is a boundary edge ``count_pass_edges`` adds a plane for
    # (one face, two distinct ends), else ``-1``.
    if w == v:
        return wp.int32(-1)
    e = unique_edge_row(unique_edges, n_edges, v, w, base)
    if e < 0:
        return wp.int32(-1)
    if edge_face_count[e] != 1:
        return wp.int32(-1)
    return e


@wp.func
def edge_boundary_quadric(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    e: wp.int32,
) -> SymmetricQuadric:
    # Boundary edge ``e``'s plane, bit for bit the one ``count_pass_edges`` adds.
    return boundary_quadric(
        vertices, faces, edge_faces[e, 0], unique_edges[e, 0], unique_edges[e, 1]
    )


@wp.func
def sorted_boundary_quadric_sum(
    neighbor_offsets: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    n_edges: wp.int32,
    base: wp.uint64,
    v: wp.int32,
) -> SymmetricQuadric:
    # The planes of vertex ``v``'s boundary edges, each found from its vertex-vertex row
    # (``boundary_edge_at``) and recomputed bit for bit, summed in ascending neighbour order: the
    # row is sorted in place first, which no later reader of it minds (the link condition counts,
    # the lock and the claim take unions and minima). The table ``count_pass_edges`` filled holds
    # the same planes in arrival order.
    sort_segment(neighbor_offsets, neighbors, v)
    q = SymmetricQuadric()
    for j in range(neighbor_offsets[v], neighbor_offsets[v + 1]):
        e = boundary_edge_at(unique_edges, edge_face_count, n_edges, base, v, neighbors[j])
        if e >= 0:
            q = q + edge_boundary_quadric(vertices, faces, unique_edges, edge_faces, e)
    return q


@wp.kernel
def gather_vertex_quadrics_exact(
    face_quadrics: wp.array[SymmetricQuadric],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    base: wp.uint64,
    boundary_quadrics: wp.array[SymmetricQuadric],
    boundary_flags: wp.array[wp.int32],
    out_hubs: wp.array[wp.int32],
    out_hub_count: wp.array[wp.int32],
    out_quadrics: wp.array[SymmetricQuadric],
) -> None:
    # ``gather_vertex_quadrics`` with its two arrival-order sums put in a fixed order, the
    # deterministic setting's form. A row longer than ``QUADRIC_ORDERED_ROW_MAX`` is listed in
    # ``out_hubs`` and left to ``gather_hub_quadrics``, which adds its faces' planes exactly; and a
    # vertex where three or more boundary edges meet -- the count ``count_pass_edges_exact`` left
    # in its flag -- sums their planes in ascending neighbour order
    # (``sorted_boundary_quadric_sum``) in place of the table filled in arrival order. Every other
    # vertex gets ``gather_vertex_quadrics``' bits: two planes added to zero are the same sum in
    # either order. The table is still taken and reset, for the next pass. The list's order is
    # arrival order, which no sum reads.
    v = wp.int32(wp.tid())
    lo = vertex_face_offsets[v]
    hi = vertex_face_offsets[v + 1]
    q = SymmetricQuadric()
    if hi - lo > QUADRIC_ORDERED_ROW_MAX:
        out_hubs[wp.atomic_add(out_hub_count, 0, 1)] = v
    else:
        q = ordered_face_quadric_sum(face_quadrics, vertex_faces, lo, hi)
    count = boundary_flags[v]
    if count != 0:
        planes = take_boundary_quadric(boundary_quadrics, boundary_flags, v)
        if count >= 3:
            planes = sorted_boundary_quadric_sum(
                neighbor_offsets,
                neighbors,
                vertices,
                faces,
                unique_edges,
                edge_face_count,
                edge_faces,
                state[DECIMATION_EDGES],
                base,
                v,
            )
        q = q + planes
    out_quadrics[v] = q


# A hub's faces' planes summed exactly by one block (``gather_hub_quadrics``): its lanes stride
# the row, a block maximum finds the group's exponent field (``reduce.group_sum_field``, one for the
# quadric's ten components), each lane adds its terms' two fixed-point words at that scale
# (``reduce.group_sum_words``, the shared exact group sum's arithmetic) and a block sum of the
# integer words is exact in any order, to ``2^-(124 - 2h)`` of the largest component for ``2^h``
# terms. A block rather than a thread: summed in one thread -- the words in two passes, or a shell
# sort of the row and an ascending sum -- a hub cost 3.3x the arrival-order gather's time on a mesh
# with two 400-face cones, the whole call 1.4x, and a sort is superlinear in a cone apex of tens
# of thousands of faces. The block form made that call 0.88x the arrival-order one, and costs
# meshes with no hub the one more graph node a pass (at most 1.01x, 4 k to 330 k faces). Not
# ``_launch.group_sums``: the pass is a captured graph that may not allocate, and ten ``float64``
# components are not one of its dtypes.
QuadricWords = wp.types.vector(length=10, dtype=wp.int64)
HUB_BLOCKS = 64
HUB_BLOCK_DIM = 128


@wp.func
def quadric_field(q: SymmetricQuadric) -> wp.int32:
    field = wp.int32(0)
    for c in range(10):
        field = wp.max(field, group_sum_field(q[c]))
    return field


@wp.func
def quadric_words(
    q: SymmetricQuadric, field: wp.int32, headroom: wp.int32
) -> tuple[QuadricWords, QuadricWords]:
    high = QuadricWords()
    low = QuadricWords()
    for c in range(10):
        words = group_sum_words(q[c], field, headroom, 1)
        high[c] = words[0]
        low[c] = words[1]
    return high, low


@wp.func
def quadric_from_words(
    high: QuadricWords, low: QuadricWords, field: wp.int32, headroom: wp.int32
) -> SymmetricQuadric:
    # The words read back as the ``float64`` sum, component by component, as
    # ``reduce.group_sum_value`` reads a buffer's (``reduce.group_sum_from_words``).
    q = SymmetricQuadric()
    for c in range(10):
        q[c] = group_sum_from_words(high[c], low[c], field, headroom, 1)
    return q


@wp.kernel
def gather_hub_quadrics(
    hubs: wp.array[wp.int32],
    hub_count: wp.array[wp.int32],
    face_quadrics: wp.array[SymmetricQuadric],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    quadrics: wp.array[SymmetricQuadric],
) -> None:
    # Each listed hub's faces' planes, summed exactly by one block and added to the boundary
    # planes ``gather_vertex_quadrics_exact`` left in ``quadrics``. ``HUB_BLOCKS`` blocks walk the
    # list, which is empty on most meshes; a block's lanes stride the row by ``wp.block_dim()``, so
    # on the CPU device one lane walks it all and the block reductions are its own values.
    block, lane = wp.tid()
    lanes = wp.block_dim()
    h = block
    while h < hub_count[0]:
        v = hubs[h]
        lo = vertex_face_offsets[v]
        hi = vertex_face_offsets[v + 1]
        field = wp.int32(0)
        for j in range(lo + lane, hi, lanes):
            field = wp.max(field, quadric_field(face_quadrics[vertex_faces[j]]))
        field = block_max(field)
        headroom = wp.int32(0)
        while (wp.int32(1) << headroom) < hi - lo:
            headroom += 1
        high = QuadricWords()
        low = QuadricWords()
        if field != GROUP_SUM_NONFINITE:
            for j in range(lo + lane, hi, lanes):
                term_high, term_low = quadric_words(face_quadrics[vertex_faces[j]], field, headroom)
                high = high + term_high
                low = low + term_low
        high = block_sum(high)
        low = block_sum(low)
        if lane == 0:
            faces_sum = SymmetricQuadric()
            if field == GROUP_SUM_NONFINITE:
                for c in range(10):
                    faces_sum[c] = wp.float64(wp.nan)
            else:
                faces_sum = quadric_from_words(high, low, field, headroom)
            quadrics[v] = faces_sum + quadrics[v]
        h += HUB_BLOCKS


@wp.kernel
def quadric_collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    quadrics: wp.array[SymmetricQuadric],
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
    out_sort_keys: wp.array[wp.float32],
    out_sort_order: wp.array[wp.int32],
) -> None:
    # Garland-Heckbert candidate: the cost of collapsing this edge and where its survivor lands.
    # The cost goes straight into the sort's key buffer beside an identity payload, so nothing is
    # copied between here and that sort; it is left at +inf for a rejected edge (and for every
    # padded row past the live edge count), so the sort puts every rejection past every candidate.
    k = wp.int32(wp.tid())
    out_survivor[k] = -1
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

    # The fold veto and the thin-part veto (``collapse_joins_opposite_sheets``), from one walk.
    folds, joins_sheets = collapse_fold_and_sheets(
        vertices, faces, vertex_face_offsets, vertex_faces, s, r, target
    )
    if folds or joins_sheets:
        return

    cost = wp.float32(quadric_error(quadric, optimum))
    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = target
    out_sort_keys[k] = cost


@wp.kernel
def drop_past_half(
    order: wp.array[wp.int32],
    sorted_cost: wp.array[wp.float32],
    state: wp.array[wp.int32],
    out_rank: wp.array[wp.int32],
    out_candidates: wp.array[wp.int32],
    out_lock_key: wp.array[wp.int64],
) -> None:
    # Stage one of the selection: retire every candidate outside the cheapest half of the live
    # edges, ``order`` being the ascending cost ranking and ``sorted_cost`` the costs in it.
    # Ranking by cost rather than by edge index is the whole difference between a quadric
    # decimation and a shortest-edge one.
    #
    # It also records each edge's position in that ranking, which is what lets every round rank its
    # winners with a scan rather than a sort (``commit_budgeted_collapses``), and its lock key for
    # the pass, bucketed against the most expensive candidate of the half (``bucketed_lock_key``).
    i = wp.int32(wp.tid())
    e = order[i]
    out_rank[e] = i
    pool = wp.max(wp.int32(1), state[DECIMATION_EDGES] // 2)
    if i >= pool:
        out_candidates[e] = -1
        return
    out_lock_key[e] = bucketed_lock_key(e, sorted_cost[i], sorted_cost[pool - 1])


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
    # retired is every candidate with an endpoint in the closed 1-rings of a collapse already
    # committed this pass, which by symmetry is every candidate that conflicts with one
    # (``scatter.endpoints_hold``).
    #
    # Not conflicting is exactly what makes reusing the pass's scoring legal: no face incident to
    # the candidate's endpoints has a vertex a commit moved or deleted, and no commit added or
    # removed an edge at its endpoints, so its quadrics, cost, target position, link condition and
    # normal-flip veto are all still the ones the scoring pass computed. Fail that test and the
    # candidate must wait for the next geometry rebuild.
    s = candidates[k]
    if s < 0:
        return -1
    if not endpoints_hold(locked, s, removed[k], wp.int32(0)):
        return -1
    return s


@wp.kernel(enable_backward=False)
def drop_locked_and_claim(
    candidates: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    lock_key: wp.array[wp.int64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
    out_winner_words: wp.array[wp.uint32],
) -> None:
    # First launch of a round: restore the pass's candidate list minus the ones a commit
    # invalidated (``unlocked_candidate``), and claim each survivor's closed 1-rings under its lock
    # key (``scatter.lock_two_rings``), which reads only this candidate's own survivor and so needs
    # no barrier after the restore. On the first round ``locked`` is all-zero and the restore is the
    # candidate list unchanged. Also clears the winner bitmask ``mark_collapse_winners`` sets, whose
    # last reader was the previous round's commit; launched over the edges or the mask's words,
    # whichever is more.
    k = wp.int32(wp.tid())
    if k < out_winner_words.shape[0]:
        out_winner_words[k] = wp.uint32(0)
    if k >= candidates.shape[0]:
        return
    s = unlocked_candidate(candidates, removed, offsets, columns, locked, k)
    out_survivor[k] = s
    if s >= 0:
        lock_two_rings(offsets, columns, s, removed[k], lock_key[k], out_min_key)


@wp.kernel(enable_backward=False)
def mark_collapse_winners(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    lock_key: wp.array[wp.int64],
    min_key: wp.array[wp.int64],
    edge_face_count: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_winner_words: wp.array[wp.uint32],
) -> None:
    # The win test, kept separate from the commit so the round can apply its budget *after* the
    # independent set is known. Trimming members from an independent set keeps it independent;
    # trimming the candidate list beforehand would change which set is found.
    #
    # Each winner sets bit ``rank[k]`` -- its place in the pass's cost order -- of the first half of
    # one bitmask, and a winner that removes a single face (a boundary edge's collapse) sets the
    # same bit of the second half, so one word scan gives every winner the count of winners, and
    # of single-face winners, cheaper than itself (``winner_faces_before``).
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    if not endpoints_hold(min_key, s, removed[k], lock_key[k]):
        out_survivor[k] = -1
        return
    p = rank[k]
    bit = wp.uint32(1) << wp.uint32(p & 31)
    wp.atomic_or(out_winner_words, p >> 5, bit)
    if edge_face_count[k] == 1:
        wp.atomic_or(out_winner_words, out_winner_words.shape[0] // 2 + (p >> 5), bit)


@wp.func
def winner_faces_before(
    rank: wp.int32,
    words: wp.array[wp.uint32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
) -> wp.int32:
    # Faces removed by the round's winners cheaper than the winner at cost rank ``rank``: two per
    # winner, less one per single-face winner, both read off ``mark_collapse_winners``' bitmask by
    # its word scan (``set_bits_before``). The pass's cost ranking is a stable total order, so the
    # winners ahead of ``rank`` are exactly the ones a serial queue would commit first.
    winners = set_bits_before(words, prefix, chunk_offsets, 0, rank)
    single = set_bits_before(words, prefix, chunk_offsets, words.shape[0] // 2, rank)
    return 2 * winners - (single - scanned_prefix(prefix, chunk_offsets, words.shape[0] // 2))


@wp.kernel(enable_backward=False)
def commit_budgeted_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    target_pos: wp.array[wp.vec3],
    edge_face_count: wp.array[wp.int32],
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
    # Apply the round's independent set up to its face budget, cheapest first, then mark the closed
    # 1-rings of every commit for the next round's conflict test (``unlocked_candidate``). Launched
    # over the edges or the vertices, whichever is wider: the vertex threads re-arm the claim keys
    # for the next round, which is legal here because ``mark_collapse_winners`` was their last
    # reader. Plain stores for the marks: every write is the same value.
    #
    # The budget is in **faces**, shared by the pass's rounds: ``n_faces - target`` less the faces
    # every earlier round removed, which ``end_collapse_round`` has stored. A collapse removes the
    # faces on its edge -- two, or one on a boundary -- and a winner commits when the cheaper
    # winners and itself fit. Counted in collapses instead, a boundary collapse would be charged
    # two faces and every pass but the last would stop short of its share, leaving a tail of passes
    # that commit one collapse each. A winner that would overshoot is still committed when it is the
    # pass's first, so a pass with a surplus of one face on a closed mesh still finishes the job;
    # that cannot manufacture a commit, because the budget only *trims* a chosen independent set.
    # ``out_count`` is ``[commits, removed faces]`` for the pass.
    k = wp.int32(wp.tid())
    if k < out_min_key.shape[0]:
        out_min_key[k] = UNCLAIMED_KEY
    if k >= survivor.shape[0]:
        return
    s = survivor[k]
    if s < 0:
        return
    removed_faces = round_state[COLLAPSE_REMOVED_FACES]
    budget = state[DECIMATION_FACES] - target_faces - removed_faces
    before = winner_faces_before(rank[k], winner_words, prefix, chunk_offsets)
    own = wp.max(wp.int32(1), wp.min(edge_face_count[k], wp.int32(2)))
    if before + own > budget and not (removed_faces == 0 and before == 0 and budget > 0):
        return
    r = removed[k]
    out_remap[r] = s
    out_positions[s] = target_pos[k]
    wp.atomic_add(out_count, 0, 1)
    wp.atomic_add(out_count, 1, own)
    stamp_two_rings(offsets, columns, s, r, 1, out_locked)


@wp.kernel
def end_collapse_round(
    max_rounds: wp.int32, count: wp.array[wp.int32], out_state: wp.array[wp.int32]
) -> None:
    # dim=1, last op of a round: decide whether another round against this same scoring is worth
    # running, and store the round's totals for the next one's budget. ``begin_decimation_pass``
    # seeds the slot table.
    #
    # It stops when the round committed nothing -- a further round cannot, since the state it
    # reads is then unchanged -- or at the round cap. A budget-exhausted pass stops through that
    # same test: its budget is zero, so nothing commits.
    _ = wp.int32(wp.tid())
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + wp.int32(1)
    progressed = count[0] > out_state[COLLAPSE_COMMITS]
    out_state[COLLAPSE_COMMITS] = count[0]
    out_state[COLLAPSE_REMOVED_FACES] = count[1]
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
