from typing import Any

import warp as wp

from ordito.kernels.algorithms.bfs import ball_row_bounds
from ordito.kernels.array import (
    OverloadTable,
    append_slot,
    greater,
    inverse_or_one,
    mask_and_not,
    next_row_entry,
    sort_segment,
    store_coo_key,
    to_vec3,
    to_vec3d,
)
from ordito.kernels.laplacian import cotmatrix_offdiagonal, face_half_cotangents, operator_row
from ordito.kernels.linalg import (
    free_row,
    selected_row,
    shifted_scalar_entry,
    solve_normal_equations,
)
from ordito.kernels.predicates import angle_defect, closest_point_on_segment, plane_basis
from ordito.kernels.reduce import block_chunk, block_chunk_1d, block_max, store_block_sum
from ordito.kernels.scatter import add_corner_triple
from ordito.kernels.triangles import corner_triple, local_corner
from ordito.kernels.vertices import gathered_area_weighted_normal

wp.set_module_options({"enable_backward": False})

# Fixed-size float64 types for the 6-coefficient quadric fit in ``relax_approx``. The rest of the
# kernel runs in float32; the least-squares solve is float64 for conditioning, and it runs through
# ``linalg.solve_normal_equations``, which is rank-generic -- ``kernels/curvature.py``'s 5x5 quadric
# fit is the same call at a different width.
# DBL_EPSILON, the relative accuracy of a float64. The area-equalizing solve compares its system's
# determinant against this times the trace's power, which is the scale-free way to ask whether the
# 1-ring is degenerate enough that the solution cannot be trusted.
DOUBLE_EPSILON = wp.constant(wp.float64(2.220446049250313e-16))

vec6d = wp.types.vector(length=6, dtype=wp.float64)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


# ---------------------------------------------------------------------------
# Region Dirichlet / least-squares smoothing
# ---------------------------------------------------------------------------


@wp.kernel
def edge_cotan_add(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_w: wp.array[wp.float32],
) -> None:
    # Accumulate each face corner's cotangent into its opposite unique edge; the two incident faces
    # sum to the cotangent edge weight cot(alpha) + cot(beta).
    #
    # The per-corner cotangent is ``laplacian.face_half_cotangents`` (its own
    # denominator/degenerate-triangle guard, rather than a second independent one derived from the
    # raw cross product) doubled back to a full cotangent, since this accumulator -- unlike
    # ``laplacian.cotmatrix`` -- wants ``cot(alpha) + cot(beta)`` rather than the half-cotangent
    # convention `laplacian.py`'s own docstring explains. ``inverse[f*3+k]`` is the unique edge id
    # of edge ``(v_k, v_{k+1})`` (``edges.faces_to_edges``'s convention), whose cotangent
    # contribution from this triangle is the angle *opposite* that edge -- i.e. the angle at the
    # corner not on it -- which is why the three half-cotangents land rotated by one slot below.
    f = wp.int32(wp.tid())
    half_cotan0, half_cotan1, half_cotan2 = face_half_cotangents(vertices, faces, f)
    two = wp.float32(2.0)
    add_corner_triple(out_w, inverse, f, two * half_cotan2, two * half_cotan0, two * half_cotan1)


@wp.kernel
def edge_cotan_add_exact(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_w: wp.array[wp.float64],
) -> None:
    # ``edge_cotan_add`` under the deterministic setting, differing only in the accumulator: the
    # same ``float32`` cotangents go into ``float64`` sums unscaled (``scatter_face_areas_exact``'s
    # device). A sum of a few ``float32`` values is exact in ``float64`` while their exponents span
    # under ``29 - log2(count)`` bits, so the commit order changes no bit, and the caller's one
    # narrowing to ``float32`` is the ``float32`` sum a manifold edge's two adds give. The
    # ``float32`` atomics are exact only for those two adds onto zero: on a non-manifold edge
    # (three or more faces, which a refilled self-intersection leaves) they commit in arrival order.
    f = wp.int32(wp.tid())
    half_cotan0, half_cotan1, half_cotan2 = face_half_cotangents(vertices, faces, f)
    two = wp.float32(2.0)
    add_corner_triple(
        out_w,
        inverse,
        f,
        wp.float64(two * half_cotan2),
        wp.float64(two * half_cotan0),
        wp.float64(two * half_cotan1),
    )


@wp.func
def clamp_cotan(w: wp.float32) -> wp.float32:
    # The summed cotangent edge weight is clamped: a degenerate edge gives arbitrarily high cot.
    return wp.clamp(w, wp.float32(-1.0), wp.float32(10.0))


# The region solves never assemble a weight *matrix*. Everything they read of the connectivity is
# the vertex -> unique-edge incidence (``incident_edge_counts`` / ``scatter_incident_edges``, then
# ``array.sort_segments``), built once per region pair, and the per-solve weights stay a per-edge
# array: a row walk reads ``weights[e]`` for each incident edge ``e``. Sorting a row by edge id
# sorts it by neighbour too -- the unique-edge table is sorted with the smaller endpoint first, so a
# vertex's edges to smaller neighbours ``(a, v)`` all precede its edges ``(v, b)`` to larger ones,
# each run in neighbour order -- so every row walk visits the neighbours in ascending order, which
# is the column order the ``bsr_from_triplets`` weight matrix these kernels replace stored them in.
# The sums over a row therefore run in the same order and round the same way.
#
# A self-loop edge ``(v, v)`` (a face repeating a vertex) is left out of the incidence: it is no
# neighbour, and in a row it would duplicate the diagonal's column.


@wp.kernel
def incident_edge_counts(
    unique_edges: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_ranks: wp.array2d[wp.int32],
) -> None:
    # Each vertex's number of incident unique edges, one atomic per endpoint, and each edge's rank
    # in its two endpoints' rows -- the value the atomic returns -- so the fill needs no cursor of
    # its own. ``out_counts`` arrives zeroed and is the tail of an ``n + 1`` offsets buffer, which
    # the caller scans in place.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    if a != b:
        out_ranks[e, 0] = wp.atomic_add(out_counts, a, 1)
        out_ranks[e, 1] = wp.atomic_add(out_counts, b, 1)


@wp.kernel
def scatter_incident_edges(
    unique_edges: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    ranks: wp.array2d[wp.int32],
    out_edges: wp.array[wp.int32],
) -> None:
    # The counting-sort fill of the incidence: ``graph.scatter_neighbor_lists``, writing the edge id
    # rather than the other endpoint -- which is what lets a row walk read the edge's weight -- at
    # the rank ``incident_edge_counts`` handed out. The order within a row is arrival order until
    # ``array.sort_segments`` sorts it.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    if a != b:
        out_edges[offsets[a] + ranks[e, 0]] = e
        out_edges[offsets[b] + ranks[e, 1]] = e


@wp.func
def incident_neighbor(unique_edges: wp.array2d[wp.int32], e: wp.int32, v: wp.int32) -> wp.int32:
    # The endpoint of incident edge ``e`` that is not ``v``.
    return unique_edges[e, 0] + unique_edges[e, 1] - v


@wp.func
def edge_weight(weights: wp.array[wp.float32], e: wp.int32, unit: wp.int32) -> wp.float64:
    # The weight of unique edge ``e``: ``1`` for unit weights (no array is passed), else its
    # clamped cotangent. ``unit`` is warp-uniform.
    if unit != wp.int32(0):
        return wp.float64(1.0)
    return wp.float64(clamp_cotan(weights[e]))


@wp.func
def place_free_entry(
    v: wp.int32, j: wp.int32, slot: wp.int32, diagonal: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # The one layout rule of a free row ``v`` of the free-free pattern: its diagonal sits before
    # its first free neighbour above ``v``, so the row's columns -- free ranks, which are monotone
    # in the vertex index -- are sorted. Called for each free neighbour ``j`` in ascending order
    # with the running ``slot`` and the diagonal's slot (``-1`` until placed); returns ``j``'s slot,
    # the next free slot and the diagonal's. ``close_free_row`` places a diagonal no neighbour
    # came after.
    if diagonal < 0 and j > v:
        diagonal = slot
        slot = slot + 1
    return slot, slot + 1, diagonal


@wp.func
def close_free_row(slot: wp.int32, diagonal: wp.int32) -> wp.int32:
    # The diagonal's slot once the whole row has been walked: see ``place_free_entry``.
    return wp.where(diagonal < 0, slot, diagonal)


@wp.func
def free_degree(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    free_mask: wp.array[wp.bool],
    v: wp.int32,
) -> wp.int32:
    # ``v``'s free neighbours plus ``v`` itself when free: the length of ``v``'s row of the
    # free-free pattern, and of the column set a least-squares row ``v`` carries.
    count = wp.where(free_mask[v], wp.int32(1), wp.int32(0))
    for k in range(offsets[v], offsets[v + 1]):
        if free_mask[incident_neighbor(unique_edges, incident[k], v)]:
            count += 1
    return count


@wp.kernel
def free_pattern_counts(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    sort_incident: wp.int32,
    unique_edges: wp.array2d[wp.int32],
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Row lengths of the free-free pattern -- the diagonal plus the free neighbours -- which is the
    # sparsity of the fixed-rim system ``D - W`` and of the smooth solve's square block ``L_ff``
    # alike. ``out_counts`` is the tail of the ``n_free + 1`` offsets buffer.
    #
    # With ``sort_incident`` set (warp-uniform) the same thread first puts vertex ``v``'s incidence
    # in neighbour order (``array.sort_segment``, the ``sort_segments`` rule), in place: the count
    # reads only ``v``'s own row, so a freshly scattered incidence is sorted and counted in one
    # pass. Every row is sorted, free or not, because the incidence is reused by the region's
    # other solve. Unset, ``incident`` arrives sorted and is only read.
    v = wp.int32(wp.tid())
    if sort_incident != 0:
        sort_segment(offsets, incident, v)
    ri = selected_row(free_mask, free_map, v)
    if ri >= 0:
        out_counts[ri] = free_degree(offsets, incident, unique_edges, free_mask, v)


@wp.kernel
def free_pattern_columns(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    pattern_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
) -> None:
    # The free-free pattern's columns, sorted (``place_free_entry``). Positions only: every solve
    # over the region writes its own values into these slots.
    v = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, v)
    if ri < 0:
        return
    slot = pattern_offsets[ri]
    diagonal = wp.int32(-1)
    for k in range(offsets[v], offsets[v + 1]):
        j = incident_neighbor(unique_edges, incident[k], v)
        cj = selected_row(free_mask, free_map, j)
        if cj >= 0:
            entry, slot, diagonal = place_free_entry(v, j, slot, diagonal)
            out_columns[entry] = cj
    out_columns[close_free_row(slot, diagonal)] = ri


@wp.func
def gather_free_positions(
    points: wp.array[wp.vec3],
    v: wp.int32,
    i: wp.int32,
    out_sol_x: wp.array[wp.float64],
    out_sol_y: wp.array[wp.float64],
    out_sol_z: wp.array[wp.float64],
) -> None:
    # Seed free unknown ``i`` (vertex ``v``) with its current position: the inverse of
    # ``scatter_free_solution``, written by each region solve's assembly kernel for its own rows.
    # Both solves ask CG for the free vertices' *new* positions, whose best available initial guess
    # is their current ones -- and for a vertex no face refers to it is the only one, because such
    # a vertex contributes no row and CG never writes its entry. Seeding from zeros would leave it
    # at the origin. The speed is the smaller half of why this exists.
    p = points[v]
    out_sol_x[i] = wp.float64(p[0])
    out_sol_y[i] = wp.float64(p[1])
    out_sol_z[i] = wp.float64(p[2])


@wp.kernel
def dirichlet_system_values(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    unit: wp.int32,
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    pattern_offsets: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    stabilizer: wp.float64,
    out_values: wp.array[wp.float64],
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
    out_rhs_z: wp.array[wp.float64],
    out_sol_x: wp.array[wp.float64],
    out_sol_y: wp.array[wp.float64],
    out_sol_z: wp.array[wp.float64],
) -> None:
    # SPD umbrella system ``A = D - W`` over the free vertices, sharp boundary, written into the
    # free-free pattern: ``-w`` off the diagonal, ``stabilizer + sum w`` on it, and the fixed
    # one-ring neighbours folded into the right-hand side, plus the optional stabilizer's pull.
    # The solve's initial guess too (``gather_free_positions``).
    v = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, v)
    if ri < 0:
        return
    gather_free_positions(points, v, ri, out_sol_x, out_sol_y, out_sol_z)
    sum_w = stabilizer
    rhs = stabilizer * to_vec3d(points[v])
    slot = pattern_offsets[ri]
    diagonal = wp.int32(-1)
    for k in range(offsets[v], offsets[v + 1]):
        e = incident[k]
        j = incident_neighbor(unique_edges, e, v)
        w = edge_weight(weights, e, unit)
        sum_w += w
        if free_mask[j]:
            entry, slot, diagonal = place_free_entry(v, j, slot, diagonal)
            out_values[entry] = -w
        else:
            rhs += w * to_vec3d(points[j])
    out_values[close_free_row(slot, diagonal)] = sum_w
    out_rhs_x[ri] = rhs[0]
    out_rhs_y[ri] = rhs[1]
    out_rhs_z[ri] = rhs[2]


@wp.kernel
def least_squares_rows(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    unit: wp.int32,
    free_mask: wp.array[wp.bool],
    points: wp.array[wp.vec3],
    out_row_sums: wp.array[wp.float64],
    out_rhs: wp.array[wp.vec3d],
    out_free_degree: wp.array[wp.int32],
) -> None:
    # One least-squares umbrella row per vertex of R = the free vertices plus their first fixed
    # ring: ``p_v = sum_d (w_vd / sumW) p_d``, whose free neighbours are unknowns and whose fixed
    # ones move to the right-hand side ``b_v`` (``-p_v`` too, for a fixed row). A row is *valid*
    # when ``v`` is in R and ``sumW != 0``, and ``out_row_sums`` carries that as its value: ``sumW``
    # for a valid row, ``0`` otherwise, so every later reader asks one question of one number.
    # ``out_rhs`` is written for valid rows only, and ``out_free_degree`` (``free_degree``) for all.
    # M itself is never stored: its entry ``(v, j)`` is ``-w_vj / sumW_v`` (``1`` on a free row's
    # own column), which the normal-equations kernels below recompute from the same two numbers.
    v = wp.int32(wp.tid())
    start = offsets[v]
    end = offsets[v + 1]
    # ``free_degree``'s count, fused into the weight sum's walk; ``v`` is in R exactly when it is
    # non-zero -- free itself, or next to a free vertex.
    count = wp.where(free_mask[v], wp.int32(1), wp.int32(0))
    sum_w = wp.float64(0.0)
    for k in range(start, end):
        e = incident[k]
        sum_w += edge_weight(weights, e, unit)
        if free_mask[incident_neighbor(unique_edges, e, v)]:
            count += 1
    out_free_degree[v] = count
    if count == 0 or sum_w == wp.float64(0.0):
        out_row_sums[v] = wp.float64(0.0)
        return
    out_row_sums[v] = sum_w
    rhs = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if not free_mask[v]:
        rhs = -to_vec3d(points[v])
    for k in range(start, end):
        e = incident[k]
        j = incident_neighbor(unique_edges, e, v)
        if not free_mask[j]:
            coeff = -edge_weight(weights, e, unit) / sum_w
            rhs -= coeff * to_vec3d(points[j])
    out_rhs[v] = rhs


@wp.func
def least_squares_entry(
    row_sums: wp.array[wp.float64], v: wp.int32, j: wp.int32, w: wp.float64
) -> wp.float64:
    # Entry ``(v, j)`` of M on a valid row ``v``: ``1`` on ``v``'s own column, ``-w_vj / sumW_v``
    # on a free neighbour's. ``w`` is ignored on the diagonal.
    if v == j:
        return wp.float64(1.0)
    return -w / row_sums[v]


@wp.func
def gather_valid_row(
    row_sums: wp.array[wp.float64],
    rhs: wp.array[wp.vec3d],
    free_degrees: wp.array[wp.int32],
    v: wp.int32,
    m: wp.float64,
    ax: wp.float64,
    ay: wp.float64,
    az: wp.float64,
    count: wp.int32,
) -> tuple[wp.float64, wp.float64, wp.float64, wp.int32]:
    # One row ``v`` of M's contribution to column ``i`` of the normal equations, ``m = M_vi``:
    # ``m b_v`` into ``M^T b`` and ``v``'s free columns into the product count. An invalid row
    # contributes nothing.
    if row_sums[v] != wp.float64(0.0):
        b = rhs[v]
        ax += m * b[0]
        ay += m * b[1]
        az += m * b[2]
        count += free_degrees[v]
    return ax, ay, az, count


@wp.func
def write_factor_pair(
    valid_i: wp.bool,
    lij: wp.float64,
    scale_i: wp.float64,
    row_sums: wp.array[wp.float64],
    j: wp.int32,
    slot: wp.int32,
    laplacian_form: wp.int32,
    out_factor: wp.array[wp.float64],
    out_factor_t: wp.array[wp.float64],
    out_factor_narrow: wp.array[wp.float32],
    out_factor_t_narrow: wp.array[wp.float32],
) -> wp.float64:
    # Entry ``slot`` = ``(i, j)`` of ``B = D^-1 L_ff`` and of ``B^T`` (``normal_equations_setup``),
    # given ``L_ij = L_ji``; returns ``|L_ij|`` for the Gershgorin sum, ``0`` on an invalid row.
    # Under ``laplacian_form`` (warp-uniform) the entry of ``L_ff`` itself goes into ``out_factor``
    # alone, zero where row ``i`` or row ``j`` is invalid so the matrix stays symmetric, and the
    # three other buffers are not written.
    sum_j = row_sums[j]
    if laplacian_form != wp.int32(0):
        out_factor[slot] = wp.where(valid_i and sum_j != wp.float64(0.0), lij, wp.float64(0.0))
        return wp.float64(0.0)
    factor = wp.where(valid_i, lij * scale_i, wp.float64(0.0))
    factor_t = wp.where(sum_j != wp.float64(0.0), lij * inverse_or_one(sum_j), wp.float64(0.0))
    out_factor[slot] = factor
    out_factor_t[slot] = factor_t
    out_factor_narrow[slot] = wp.float32(factor)
    out_factor_t_narrow[slot] = wp.float32(factor_t)
    return wp.where(valid_i, wp.abs(lij), wp.float64(0.0))


@wp.kernel
def normal_equations_setup(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    unit: wp.int32,
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    row_sums: wp.array[wp.float64],
    rhs: wp.array[wp.vec3d],
    free_degrees: wp.array[wp.int32],
    pattern_offsets: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    laplacian_form: wp.int32,
    laplacian_shift: wp.float64,
    out_term_counts: wp.array[wp.int32],
    out_atb_x: wp.array[wp.float64],
    out_atb_y: wp.array[wp.float64],
    out_atb_z: wp.array[wp.float64],
    out_factor: wp.array[wp.float64],
    out_factor_t: wp.array[wp.float64],
    out_factor_narrow: wp.array[wp.float32],
    out_factor_t_narrow: wp.array[wp.float32],
    out_ratios: wp.array[wp.float64],
    out_sol_x: wp.array[wp.float64],
    out_sol_y: wp.array[wp.float64],
    out_sol_z: wp.array[wp.float64],
) -> None:
    # Everything of the normal equations ``(M^T M) x = M^T b`` one free unknown ``i`` owns, from
    # the rows ``least_squares_rows`` wrote. The rows of M touching column ``i`` are ``i``'s own and
    # its neighbours', ``V_i = {i} + N(i)``, of which the valid ones count, walked in ascending
    # order -- the order ``M^T``'s row ``i`` stores them in, so ``M^T b`` accumulates as a CSR row
    # dot over ``M^T`` would. Three things come out:
    #
    # - ``M^T b``'s row ``i``;
    # - the number of (row, column) products ``M^T M``'s row ``i`` gathers, ``sum |C_v|`` over the
    #   valid ``v`` in ``V_i``, with ``C_v`` row ``v``'s free columns (``free_degrees``) -- the
    #   term segment ``normal_equations_rows`` builds the row in;
    # - row ``i`` of the preconditioner's factor ``B = D^-1 L_ff`` and of its transpose in the
    #   free-free pattern, in ``float64`` and ``float32`` (what the one-block solve applies), and
    #   the row's Gershgorin quantity ``sum_j |L_ij| / D_i``. ``L_ff`` is the free rows'
    #   Laplacian -- ``sumW`` on the diagonal, ``-w`` off it -- and ``D`` its diagonal, ``1`` on
    #   an invalid row, whose factor row is zero. ``B^T``'s entry ``(i, j)`` is ``L_ji / D_j``:
    #   the pattern is symmetric, so the transpose shares it, and where row ``j`` is invalid the
    #   entry is a structural zero. The arithmetic is ``linalg.SquaredLaplacianPreconditioner``'s
    #   own -- ``L * (1/D)`` through ``inverse_or_one``, the absolute values summed in column
    #   order. Under ``laplacian_form`` (the V-cycle form of that preconditioner, which needs
    #   ``L_ff`` and ``D`` rather than the factor) row ``i`` of ``L_ff + laplacian_shift * D``
    #   goes into ``out_factor`` and ``D_i`` (``1`` on an invalid row) into ``out_ratios``; see
    #   ``write_factor_pair``. The shift makes the matrix definite on a free component no fixed
    #   vertex touches, where ``L_ff`` itself is singular.
    #
    # And the solve's initial guess (``gather_free_positions``).
    i = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, i)
    if ri < 0:
        return
    gather_free_positions(points, i, ri, out_sol_x, out_sol_y, out_sol_z)
    start = offsets[i]
    end = offsets[i + 1]
    sum_i = row_sums[i]
    valid_i = sum_i != wp.float64(0.0)
    scale_i = inverse_or_one(wp.where(valid_i, sum_i, wp.float64(1.0)))
    one = wp.float64(1.0)
    # The diagonal entry written: ``L_ii`` for the factor, ``(1 + shift) L_ii`` for ``L_ff``.
    diagonal_value = wp.where(laplacian_form != wp.int32(0), sum_i * (one + laplacian_shift), sum_i)
    ax = wp.float64(0.0)
    ay = wp.float64(0.0)
    az = wp.float64(0.0)
    terms = wp.int32(0)
    total = wp.float64(0.0)
    slot = pattern_offsets[ri]
    diagonal = wp.int32(-1)
    own_pending = wp.bool(True)
    for k in range(start, end):
        e = incident[k]
        v = incident_neighbor(unique_edges, e, i)
        w = edge_weight(weights, e, unit)
        if own_pending and v > i:
            own_pending = False
            ax, ay, az, terms = gather_valid_row(
                row_sums, rhs, free_degrees, i, one, ax, ay, az, terms
            )
        ax, ay, az, terms = gather_valid_row(
            row_sums,
            rhs,
            free_degrees,
            v,
            least_squares_entry(row_sums, v, i, w),
            ax,
            ay,
            az,
            terms,
        )
        if free_mask[v]:
            before = diagonal
            entry, slot, diagonal = place_free_entry(i, v, slot, diagonal)
            if diagonal != before:
                total += write_factor_pair(
                    valid_i,
                    diagonal_value,
                    scale_i,
                    row_sums,
                    i,
                    diagonal,
                    laplacian_form,
                    out_factor,
                    out_factor_t,
                    out_factor_narrow,
                    out_factor_t_narrow,
                )
            total += write_factor_pair(
                valid_i,
                -w,
                scale_i,
                row_sums,
                v,
                entry,
                laplacian_form,
                out_factor,
                out_factor_t,
                out_factor_narrow,
                out_factor_t_narrow,
            )
    if own_pending:
        ax, ay, az, terms = gather_valid_row(row_sums, rhs, free_degrees, i, one, ax, ay, az, terms)
    if diagonal < 0:
        total += write_factor_pair(
            valid_i,
            diagonal_value,
            scale_i,
            row_sums,
            i,
            slot,
            laplacian_form,
            out_factor,
            out_factor_t,
            out_factor_narrow,
            out_factor_t_narrow,
        )
    out_term_counts[ri] = terms
    out_atb_x[ri] = ax
    out_atb_y[ri] = ay
    out_atb_z[ri] = az
    if laplacian_form != wp.int32(0):
        out_ratios[ri] = wp.where(valid_i, sum_i, wp.float64(1.0))
    else:
        out_ratios[ri] = wp.where(valid_i, total / sum_i, wp.float64(0.0))


@wp.func
def accumulate_term(
    column: wp.int32,
    m_vi: wp.float64,
    m_vu: wp.float64,
    start: wp.int32,
    width: wp.int32,
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> wp.int32:
    # Add the term ``M_vi M_vu`` of ``(M^T M)_iu`` into the row's sorted list of distinct columns
    # ``out_columns[start : start + width]`` -- found by bisection, or inserted in order with the
    # tail shifted up one -- and return the list's new width. A column's value starts from its first
    # term's product and takes each later one as a multiply-add, in arrival order.
    lo = wp.int32(0)
    hi = width
    while lo < hi:
        mid = (lo + hi) // 2
        if out_columns[start + mid] < column:
            lo = mid + 1
        else:
            hi = mid
    slot = start + lo
    if lo < width and out_columns[slot] == column:
        out_values[slot] = out_values[slot] + m_vi * m_vu
        return width
    for t in range(start + width, slot, -1):
        out_columns[t] = out_columns[t - 1]
        out_values[t] = out_values[t - 1]
    out_columns[slot] = column
    out_values[slot] = m_vi * m_vu
    return width + 1


@wp.func
def accumulate_row_terms(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    unit: wp.int32,
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    row_sums: wp.array[wp.float64],
    v: wp.int32,
    m_vi: wp.float64,
    start: wp.int32,
    width: wp.int32,
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> wp.int32:
    # Row ``v`` of M's terms of ``M^T M``'s row ``i``, ``m_vi = M_vi``: one per free column ``u``
    # of the row, ``M_vu`` being ``1`` on ``v``'s own column and ``-w_vu / sumW_v`` elsewhere --
    # the value ``least_squares_rows``' row stands for. An invalid row has no terms.
    sum_v = row_sums[v]
    if sum_v == wp.float64(0.0):
        return width
    if free_mask[v]:
        width = accumulate_term(
            free_map[v], m_vi, wp.float64(1.0), start, width, out_columns, out_values
        )
    for q in range(offsets[v], offsets[v + 1]):
        e = incident[q]
        u = incident_neighbor(unique_edges, e, v)
        if free_mask[u]:
            width = accumulate_term(
                free_map[u],
                m_vi,
                -edge_weight(weights, e, unit) / sum_v,
                start,
                width,
                out_columns,
                out_values,
            )
    return width


@wp.kernel
def normal_equations_rows(
    offsets: wp.array[wp.int32],
    incident: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    unit: wp.int32,
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    row_sums: wp.array[wp.float64],
    term_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_widths: wp.array[wp.int32],
) -> None:
    # Row ``i`` of ``M^T M``, sorted, in the head of ``i``'s term segment (sized for every term,
    # which bounds the distinct columns), and its length. The terms ``M_vi M_vu`` -- one per free
    # column ``u`` of each valid row ``v`` of ``V_i`` -- are taken in ascending ``v``, so each
    # entry is summed along ``M^T``'s row from its first product, the two factors unrounded into a
    # multiply-add: ``warp.sparse.bsr_mm``'s arithmetic, which it performs in that order for most
    # entries and in its triplet sort's order for the rest, so the two agree to the last bit on most
    # entries and within one rounding on the others. That is ``bsr_mm``'s expansion of the product,
    # per row, rather than one global sort of every term. ``normal_equations_values`` compacts the
    # rows into the CSR.
    i = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, i)
    if ri < 0:
        return
    start = term_offsets[ri]
    width = wp.int32(0)
    own_pending = wp.bool(True)
    for k in range(offsets[i], offsets[i + 1]):
        e = incident[k]
        v = incident_neighbor(unique_edges, e, i)
        if own_pending and v > i:
            own_pending = False
            width = accumulate_row_terms(
                offsets,
                incident,
                unique_edges,
                weights,
                unit,
                free_mask,
                free_map,
                row_sums,
                i,
                wp.float64(1.0),
                start,
                width,
                out_columns,
                out_values,
            )
        width = accumulate_row_terms(
            offsets,
            incident,
            unique_edges,
            weights,
            unit,
            free_mask,
            free_map,
            row_sums,
            v,
            least_squares_entry(row_sums, v, i, edge_weight(weights, e, unit)),
            start,
            width,
            out_columns,
            out_values,
        )
    if own_pending:
        width = accumulate_row_terms(
            offsets,
            incident,
            unique_edges,
            weights,
            unit,
            free_mask,
            free_map,
            row_sums,
            i,
            wp.float64(1.0),
            start,
            width,
            out_columns,
            out_values,
        )
    out_widths[ri] = width


@wp.kernel
def mark_anchored_components(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    incidence_offsets: wp.array[wp.int32],
    pattern_offsets: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_anchored: wp.array[wp.int32],
) -> None:
    # Flag the free component (``labels``, over the free-free pattern) of every free vertex with a
    # fixed neighbour. Its pattern row holds the diagonal and one entry per free neighbour, so a
    # vertex has a fixed neighbour exactly when it has more incident edges than that. Every writer
    # stores the same ``1``, so the order the stores land in decides nothing.
    v = wp.int32(wp.tid())
    i = selected_row(free_mask, free_map, v)
    if i < 0:
        return
    degree = incidence_offsets[v + 1] - incidence_offsets[v]
    if degree > pattern_offsets[i + 1] - pattern_offsets[i] - 1:
        out_anchored[labels[i]] = wp.int32(1)


@wp.kernel
def normal_equations_values(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    term_offsets: wp.array[wp.int32],
    row_columns: wp.array[wp.int32],
    row_values: wp.array[wp.float64],
    system_offsets: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    anchored: wp.array[wp.int32],
    atb_x: wp.array[wp.float64],
    atb_y: wp.array[wp.float64],
    atb_z: wp.array[wp.float64],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    sol_z: wp.array[wp.float64],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # Row ``i`` of ``M^T M`` moved from the head of its term segment into the compact CSR.
    #
    # A row of a free component no fixed vertex touches (``mark_anchored_components``) is written
    # as the identity instead, with the seed (``sol``) as its right-hand side: the component's
    # umbrella rows fix it only up to a translation, so the system does not place it, and pinning
    # it at its seed keeps its block out of the solve -- a residual of zero there from the start,
    # where its singular block would otherwise stall conjugate gradient under the V-cycle form.
    # Its columns occur only in its own component's rows, which are all pinned, so the matrix stays
    # symmetric. A row with no diagonal entry (a vertex no face refers to) is empty and stays so.
    i = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, i)
    if ri < 0:
        return
    source = term_offsets[ri]
    start = system_offsets[ri]
    pinned = anchored[labels[ri]] == 0
    has_diagonal = wp.bool(False)
    for s in range(system_offsets[ri + 1] - start):
        column = row_columns[source + s]
        out_columns[start + s] = column
        if pinned:
            out_values[start + s] = wp.where(column == ri, wp.float64(1.0), wp.float64(0.0))
            if column == ri:
                has_diagonal = True
        else:
            out_values[start + s] = row_values[source + s]
    if has_diagonal:
        atb_x[ri] = sol_x[ri]
        atb_y[ri] = sol_y[ri]
        atb_z[ri] = sol_z[ri]


@wp.kernel
def scatter_free_solution(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    sol_z: wp.array[wp.float64],
    out_points: wp.array[wp.vec3],
) -> None:
    # The solved positions: the reduced solve's answer over the free vertices, and the position it
    # arrived with for a pinned one. Every vertex is written, so the result needs no copy of
    # ``points`` first. The inverse of ``gather_free_positions``, which seeds the same solve.
    v = wp.int32(wp.tid())
    i = selected_row(free_mask, free_map, v)
    if i >= 0:
        out_points[v] = wp.vec3(wp.float32(sol_x[i]), wp.float32(sol_y[i]), wp.float32(sol_z[i]))
    else:
        out_points[v] = points[v]


@wp.kernel
def fairing_system_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    stiffness: wp.array[wp.float64],
    area_sums: wp.array[wp.float64],
    scale: wp.float64,
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    system_offsets: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_rhs: wp.array2d[wp.float64],
    out_solution: wp.array2d[wp.float64],
) -> None:
    # One implicit-fairing pass's system ``A = M + scale L`` (``scale = -lamb``), its right-hand
    # side and its CG seed (the vertex's current position), one thread per vertex over a mesh
    # operator pattern (``laplacian._mesh_operator_pattern``). The stiffness row is the input's
    # ``stiffness`` values (the conformalized flow) or, with ``stiffness`` empty, formed in the
    # thread from the half-cotangents ``laplacian.cotmatrix_entries_and_area_sums`` writes, as
    # ``laplacian.cotmatrix_rows`` forms it (``cotmatrix_offdiagonal``; the diagonal minus the
    # off-diagonal sum in column order), so the mesh-wide ``L`` is never stored. The mass is
    # ``area_sums`` over three (``scatter.scale_face_sums``) and each entry is
    # ``kernels/linalg.shifted_scalar_entry``'s.
    #
    # With ``fixed_mask`` empty every vertex is free and ``A`` is written over the pattern itself
    # with right-hand side ``M V``. Otherwise this is the Dirichlet pass: the free-free block of
    # ``A`` written into the interior pattern a ``linalg.assemble_interior_system`` call built once
    # for the whole flow (``system_offsets``; its columns are that build's), and the right-hand
    # side ``(M V)_u - A_ub x_b``, the pinned columns subtracted in row order before the mass term
    # is added, as ``interior_row_counts`` / ``interior_system_csr`` extract it. A pinned vertex's
    # value is its ``positions`` entry, which no pass writes.
    #
    # Each vertex's ``area_sums`` entry is cleared once read, pinned or not, so the next pass's
    # face pass adds into zeros with no clear of its own.
    v = wp.int32(wp.tid())
    area_sum = area_sums[v]
    area_sums[v] = wp.float64(0.0)
    dirichlet = fixed_mask.shape[0] > 0
    i = v
    if dirichlet:
        i = free_row(fixed_mask, free_map, v)
        if i < 0:
            return
    m = area_sum / wp.float64(3.0)
    p = positions[v]
    from_cot = stiffness.shape[0] == 0
    slot = system_offsets[i]
    acc0 = wp.float64(0.0)
    acc1 = wp.float64(0.0)
    acc2 = wp.float64(0.0)
    total = wp.float64(0.0)
    diagonal_entry = wp.int32(-1)
    diagonal_slot = wp.int32(-1)
    for e in range(offsets[v], offsets[v + 1]):
        j = columns[e]
        if j == v:
            # Formed after the row: from the cotangents it is minus the off-diagonal sum.
            diagonal_entry = e
            diagonal_slot = slot
            slot += 1
            continue
        raw = wp.float64(0.0)
        if from_cot:
            raw = cotmatrix_offdiagonal(e, run_start, keys, count, order, cot_entries, raw)
            total += raw
        else:
            raw = stiffness[e]
        value = shifted_scalar_entry(raw, scale, j == v, m)
        if dirichlet and fixed_mask[j]:
            pinned = positions[j]
            acc0 -= value * pinned[0]
            acc1 -= value * pinned[1]
            acc2 -= value * pinned[2]
        else:
            out_values[slot] = value
            slot += 1
    if diagonal_entry >= 0:
        raw = -total
        if not from_cot:
            raw = stiffness[diagonal_entry]
        # The flag is read off the pattern rather than written as the constant it is here, as
        # ``kernels/heat.heat_system_rows`` explains: a constant lets the CUDA compiler fuse the
        # mass's addition into the product's rounding.
        out_values[diagonal_slot] = shifted_scalar_entry(
            raw, scale, columns[diagonal_entry] == v, m
        )
    if dirichlet:
        out_rhs[0, i] = acc0 + m * p[0]
        out_rhs[1, i] = acc1 + m * p[1]
        out_rhs[2, i] = acc2 + m * p[2]
    else:
        out_rhs[0, i] = m * p[0]
        out_rhs[1, i] = m * p[1]
        out_rhs[2, i] = m * p[2]
    out_solution[0, i] = p[0]
    out_solution[1, i] = p[1]
    out_solution[2, i] = p[2]


@wp.kernel
def scatter_free_positions(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    sol_z: wp.array[wp.float64],
    out_positions: wp.array[wp.vec3d],
    out_narrowed: wp.array[wp.vec3],
) -> None:
    # Write the reduced solution back to the unpinned vertices only. Pinned ones are left holding
    # whatever they already have, which is their original position -- they never move. Every
    # vertex's position is also written narrowed to ``wp.vec3``, as ``array.astype`` narrows it:
    # the next pass's cotangent and mass build reads that, and after the last pass it is the
    # answer, so no conversion pass follows.
    v = wp.int32(wp.tid())
    i = free_row(fixed_mask, free_map, v)
    position = out_positions[v]
    if i >= 0:
        position = wp.vec3d(sol_x[i], sol_y[i], sol_z[i])
        out_positions[v] = position
    out_narrowed[v] = to_vec3(position)


@wp.func
def rescale_about_center(position: wp.vec3d, center: wp.vec3d, scale: wp.float64) -> wp.vec3d:
    # (position - center) * scale + center: a uniform rescale about an arbitrary fixed point
    # rather than the origin. ``smoothing._VolumeConstraint`` needs this, not a bare
    # multiply, because the mesh being smoothed is rarely centred at the origin and
    # ``trimesh.smoothing.filter_laplacian`` rescales about the mesh's own (fixed, initial) centre
    # of mass -- multiplying by ``scale`` alone silently translates the whole mesh on every pass
    # whenever the two points differ, and the error compounds with the iteration count.
    return (position - center) * scale + center


@wp.kernel
def volume_rescale_parameters(
    volume_current: wp.array[wp.float64],
    anchor_volume: wp.array[wp.float64],
    anchor_moments: wp.array[wp.float64],
    state: wp.array[wp.float64],
) -> None:
    """
    Form the volume constraint's scale and centre once on the device, for a whole pass.

    ``state`` is ``[scale, centre x 3]``, written here from the anchor and the current volume. The
    anchor is read where its block commits left it, in slot 0 of each buffer: the signed volume of
    the input in ``float64`` (``measures.volume``'s kernel) in ``anchor_volume[0]`` and the first
    four of ``measures.moments``' integrals of its ``float32`` copy in ``anchor_moments[0:4]``. The
    centre is formed the way ``measures.moments`` forms it on the host -- first moment over volume
    in ``float64``, rounded to ``float32``, ``NaN`` for a zero volume -- and stays the *initial*
    one on every pass, as ``trimesh.smoothing.filter_laplacian`` rescales about that fixed point.

    The scale is the cube root of the volume ratio, written ``0`` (no positive scale is) where the
    pass is skipped: a zero current volume, or a ratio that is not positive, which is an
    inconsistently wound or non-watertight input whose "volume" no scale factor can restore. **The
    skip has to leave positions untouched, not scale them by 1**: on a mesh with no faces the
    centre is ``NaN``, and rescaling about it by 1 is ``(p - NaN) + NaN``, which propagates rather
    than cancelling. One cube root a pass, not one a vertex: ``float64`` transcendentals run at a
    small fraction of ``float32``'s rate on this class of device.
    """
    current = volume_current[0]
    scale = wp.float64(0.0)
    if current != wp.float64(0.0):
        ratio = anchor_volume[0] / current
        if ratio > wp.float64(0.0):
            scale = wp.pow(ratio, wp.float64(1.0) / wp.float64(3.0))
    state[0] = scale
    total = anchor_moments[0]
    center = wp.vec3d(wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    if total != wp.float64(0.0):
        center = to_vec3d(
            wp.vec3(
                wp.float32(anchor_moments[1] / total),
                wp.float32(anchor_moments[2] / total),
                wp.float32(anchor_moments[3] / total),
            )
        )
    for k in range(3):
        state[1 + k] = center[k]


@wp.func
def rescaled_to_volume(state: wp.array[wp.float64], position: wp.vec3d) -> wp.vec3d:
    # ``position`` rescaled by the scale and about the centre ``volume_rescale_parameters`` wrote;
    # a zero scale is the skip.
    scale = state[0]
    if scale == wp.float64(0.0):
        return position
    return rescale_about_center(position, wp.vec3d(state[1], state[2], state[3]), scale)


@wp.kernel
def rescale_to_volume(state: wp.array[wp.float64], out_positions: wp.array[wp.vec3d]) -> None:
    # In place: the volume constraint applied to a ``float64`` iterate the smoothing continues from.
    v = wp.int32(wp.tid())
    out_positions[v] = rescaled_to_volume(state, out_positions[v])


@wp.kernel
def rescale_to_volume_narrowed(
    state: wp.array[wp.float64], positions: wp.array[wp.vec3d], out_positions: wp.array[wp.vec3]
) -> None:
    # ``rescale_to_volume`` on a pass's last iterate, narrowed to ``float32`` as it is written.
    v = wp.int32(wp.tid())
    out_positions[v] = to_vec3(rescaled_to_volume(state, positions[v]))


@wp.func
def mut_dif_step(
    v_prev: wp.vec3d, lv: wp.vec3d, adil: wp.float64, mean_adil: wp.float64, lamb: wp.float64
) -> wp.vec3d:
    # v' = v + lamber * (L.v - v), lamber = max(0.2 * lamb, min(1.0, lamb * adil / mean_adil)).
    # Not ``wp.clamp``: the two differ once ``0.2 * lamb > 1``, and this nesting order is the one
    # trimesh's ``filter_mut_dif_laplacian`` uses (``np.maximum(..., np.minimum(...))``).
    lamber = wp.max(wp.float64(0.2) * lamb, wp.min(wp.float64(1.0), lamb * adil / mean_adil))
    return wp.lerp(v_prev, lv, lamber)


# Each of the five kernels below fuses ``kernels/laplacian.operator_row`` with the step that
# consumes its result. Every explicit smoothing filter alternates the two, so the pair cost one
# extra launch and one ``(n_vertices,)`` float64x3 round trip through global memory per pass;
# applying the row in the consuming thread removes both. Each step is one expression with exactly
# one caller, so it is written here rather than behind a helper name.
#
# The pass kernels whose filters start from and return ``wp.vec3`` positions take either vector
# type on the read and the write (overloads in ``_register_overloads``): a filter's first pass reads
# the ``float32`` input, widened exactly as each entry is read, and its last writes the ``float32``
# answer, narrowed as it is stored -- the arithmetic of a widening copy before the passes and a
# narrowing one after them, without either.


@wp.func
def diffused_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    coeff: wp.float64,
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of a ``diffuse_vec3_pass``, before the store's narrowing.
    lv = operator_row(offsets, columns, values, positions, i)
    return wp.lerp(wp.vec3d(positions[i]), lv, coeff)


@wp.func
def neighborhood_averaged_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of a ``neighborhood_average_pass``, before the store's narrowing.
    lv = operator_row(offsets, columns, values, positions, i)
    deg = wp.float64(offsets[i + 1] - offsets[i])
    return (wp.vec3d(positions[i]) + deg * lv) / (deg + wp.float64(1.0))


@wp.func
def graph_sharpened_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    weight: wp.float64,
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of a ``graph_sharpen_pass``, before the store's narrowing.
    gv = operator_row(offsets, columns, values, positions, i)
    return wp.vec3d(positions[i]) - weight * gv


# ``vec3_passes_serial``'s pass kinds: the row of ``diffuse_vec3_pass``,
# ``neighborhood_average_pass`` or ``graph_sharpen_pass``.
PASS_DIFFUSE = wp.constant(0)
PASS_NEIGHBORHOOD_AVERAGE = wp.constant(1)
PASS_GRAPH_SHARPEN = wp.constant(2)


@wp.func
def vec3_pass_row(
    kind: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    coeff: wp.float64,
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of the pass ``kind`` names, through the pass kernel's own row function.
    if kind == PASS_NEIGHBORHOOD_AVERAGE:
        return neighborhood_averaged_row(offsets, columns, values, positions, i)
    if kind == PASS_GRAPH_SHARPEN:
        return graph_sharpened_row(offsets, columns, values, positions, coeff, i)
    return diffused_row(offsets, columns, values, positions, coeff, i)


@wp.kernel
def vec3_passes_serial(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    vertices: wp.array[wp.vec3],
    kind: wp.int32,
    coeff_even: wp.float64,
    coeff_odd: wp.float64,
    passes: wp.int32,
    out_scratch_0: wp.array[wp.vec3d],
    out_scratch_1: wp.array[wp.vec3d],
    out_x: wp.array[wp.vec3],
) -> None:
    # Every pass of ``smoothing._vec3_passes`` in one launch of one thread, the CPU device's form
    # (its threads run one after another anyway), for ``passes >= 2`` passes of one kind through
    # one operator, pass ``k`` stepping by ``coeff_even`` or ``coeff_odd`` by its parity. The
    # rotation is that helper's: pass 0 reads ``vertices``, pass ``k`` writes
    # ``out_scratch_{k % 2}`` and the next reads it, and the last writes ``out_x``, narrowed as the
    # pass kernels store it. Each row is the pass kernel's own, so the result is the launches' to
    # the bit. ``out_scratch_1`` may alias ``out_scratch_0`` at two passes, where it is unused.
    n = vertices.shape[0]
    for index in range(passes):
        coeff = wp.where(index % 2 == 0, coeff_even, coeff_odd)
        for i in range(n):
            value = wp.vec3d()
            if index == 0:
                value = vec3_pass_row(kind, offsets, columns, values, vertices, coeff, i)
            elif index % 2 == 1:
                value = vec3_pass_row(kind, offsets, columns, values, out_scratch_0, coeff, i)
            else:
                value = vec3_pass_row(kind, offsets, columns, values, out_scratch_1, coeff, i)
            if index == passes - 1:
                out_x[i] = wp.vec3(value)
            elif index % 2 == 0:
                out_scratch_0[i] = value
            else:
                out_scratch_1[i] = value


@wp.kernel
def diffuse_vec3_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    coeff: wp.float64,
    out_next: wp.array[Any],
) -> None:
    # Explicit diffusion step v' = v + coeff * (L.v - v); coeff = +lambda (shrink) or -nu
    # (inflate). ``wp.lerp`` extrapolates for coeff outside [0, 1], which the inflating step of
    # ``filter_taubin`` relies on.
    i = wp.int32(wp.tid())
    out_next[i] = out_next.dtype(diffused_row(offsets, columns, values, positions, coeff, i))


@wp.kernel
def neighborhood_average_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    out_next: wp.array[Any],
) -> None:
    # Closed 1-ring average: new_v = (v + deg * L.v) / (deg + 1), where L is the neighbors-only
    # averaging operator and deg is the CSR row length (the vertex degree). deg = 0 -> new_v = v.
    i = wp.int32(wp.tid())
    out_next[i] = out_next.dtype(neighborhood_averaged_row(offsets, columns, values, positions, i))


@wp.kernel
def mark_spikes(
    angle_sums: wp.array[wp.float32],
    min_defect: wp.float32,
    slot: wp.int32,
    out_spikes: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
) -> None:
    # A spike is a vertex whose angle *defect* ``2 pi - angle_sum`` exceeds ``min_defect``
    # (``vertices.vertex_defects``' quantity, ``array.greater``'s test), counted into the zeroed
    # ``out_count[slot]`` (a slot per pass, so no pass clears one). In place: the sum is re-zeroed
    # once read, so the next pass's scatter needs no fill.
    v = wp.int32(wp.tid())
    spike = greater(angle_defect(angle_sums[v]), min_defect)
    angle_sums[v] = wp.float32(0.0)
    out_spikes[v] = spike
    if spike:
        wp.atomic_add(out_count, slot, 1)


@wp.kernel
def flatten_spikes_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    spikes: wp.array[wp.bool],
    positions: wp.array[wp.vec3],
    out_next: wp.array[wp.vec3],
) -> None:
    # ``neighborhood_average_pass`` on the spikes alone, from ``float32`` positions widened as they
    # are read and narrowed as the average is written -- the same arithmetic as one
    # ``filter_neighborhood_average`` pass followed by a masked select, without the two
    # whole-mesh conversions. Every other vertex is copied.
    i = wp.int32(wp.tid())
    if not spikes[i]:
        out_next[i] = positions[i]
        return
    lv = operator_row(offsets, columns, values, positions, i)
    deg = wp.float64(offsets[i + 1] - offsets[i])
    out_next[i] = to_vec3((to_vec3d(positions[i]) + deg * lv) / (deg + wp.float64(1.0)))


@wp.kernel
def spike_neighbor_keys(
    faces: wp.array[wp.int32],
    spikes: wp.array[wp.bool],
    base: wp.uint64,
    slot: wp.int32,
    out_count: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The spike rows of ``laplacian.laplacian(symmetric=True)``'s pattern, as
    # ``row * base + column`` keys: each corner on a spike contributes its face's other two
    # corners, a degenerate face's repeated vertex its self-edge. Appended through the zeroed
    # ``out_count[slot]`` cursor into a radix sort's double-width buffers (identity payload) while
    # they have room; past it only the count grows, and the caller re-runs at the size it read.
    f = wp.int32(wp.tid())
    capacity = out_order.shape[0] // 2
    for k in range(3):
        v = faces[3 * f + k]
        if spikes[v]:
            at = wp.atomic_add(out_count, slot, 2)
            if at + 2 <= capacity:
                row = wp.uint64(v) * base
                out_keys[at] = row + wp.uint64(faces[3 * f + (k + 1) % 3])
                out_keys[at + 1] = row + wp.uint64(faces[3 * f + (k + 2) % 3])
                out_order[at] = at
                out_order[at + 1] = at + 1


@wp.func
def spike_row_entry(key: wp.uint64, row: wp.uint64, base: wp.uint64) -> tuple[wp.int32, wp.float32]:
    # A spike row's column and its weight before normalization: ``laplacian_rows``' symmetric rule,
    # ``1`` per unique edge and ``2`` for a degenerate face's self-edge.
    column = key - row * base
    weight = wp.where(column == row, wp.float32(2.0), wp.float32(1.0))
    return wp.int32(column), weight


@wp.kernel
def average_spike_rows(
    sorted_keys: wp.array[wp.uint64],
    base: wp.uint64,
    positions: wp.array[wp.vec3],
    out_rows: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
) -> None:
    # ``flatten_spikes_pass`` on the rows ``spike_neighbor_keys`` collected, from their sorted keys
    # instead of an assembled operator: the first position of each row walks its run twice, once
    # for the unique columns' count and weight total (in column order, as ``laplacian_rows``
    # normalizes) and once for the normalized average (``operator_row``'s arithmetic), so the result
    # is the full operator's bit for bit. It is written beside its row at that position, for
    # ``apply_spike_rows`` to scatter once every row has read the old positions.
    i = wp.int32(wp.tid())
    n = sorted_keys.shape[0]
    key = sorted_keys[i]
    row = key // base
    out_rows[i] = -1
    if i > 0 and sorted_keys[i - 1] // base == row:
        return
    total = wp.float32(0.0)
    degree = wp.int32(0)
    end = i
    previous = key + wp.uint64(1)
    while end < n and sorted_keys[end] // base == row:
        if sorted_keys[end] != previous:
            _column, weight = spike_row_entry(sorted_keys[end], row, base)
            total += weight
            degree += 1
            previous = sorted_keys[end]
        end += 1
    acc = wp.vec3d(0.0, 0.0, 0.0)
    previous = key + wp.uint64(1)
    for j in range(i, end):
        if sorted_keys[j] != previous:
            column, weight = spike_row_entry(sorted_keys[j], row, base)
            acc += wp.float64(weight / total) * to_vec3d(positions[column])
            previous = sorted_keys[j]
    v = wp.int32(row)
    deg = wp.float64(degree)
    out_rows[i] = v
    out_positions[i] = to_vec3((to_vec3d(positions[v]) + deg * acc) / (deg + wp.float64(1.0)))


@wp.kernel
def apply_spike_rows(
    rows: wp.array[wp.int32], averaged: wp.array[wp.vec3], out_positions: wp.array[wp.vec3]
) -> None:
    # In place: ``average_spike_rows``' results written back over the positions they read.
    i = wp.int32(wp.tid())
    v = rows[i]
    if v >= 0:
        out_positions[v] = averaged[i]


@wp.func
def humphrey_residual_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    original: wp.array[wp.vec3],
    alpha: wp.float64,
    lv_out: wp.array[wp.vec3d],
    b_out: wp.array[wp.vec3d],
    i: wp.int32,
) -> None:
    # Row ``i`` of a ``humphrey_residual_pass``: writes its ``L.v`` and ``b``.
    lv = operator_row(offsets, columns, values, positions, i)
    lv_out[i] = lv
    b_out[i] = lv - wp.lerp(wp.vec3d(positions[i]), to_vec3d(original[i]), alpha)


@wp.kernel
def humphrey_residual_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    original: wp.array[wp.vec3],
    alpha: wp.float64,
    out_lv: wp.array[wp.vec3d],
    out_b: wp.array[wp.vec3d],
) -> None:
    # b = L.v - (alpha * original + (1 - alpha) * q), the Humphrey correction term. ``out_lv`` is
    # still written because the update pass below reads it; only ``L.b`` disappears. ``original``
    # is the ``float32`` input, widened as it is read.
    i = wp.int32(wp.tid())
    humphrey_residual_row(offsets, columns, values, positions, original, alpha, out_lv, out_b, i)


@wp.func
def humphrey_update_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lv: wp.array[wp.vec3d],
    b: wp.array[wp.vec3d],
    beta: wp.float64,
    neighbour_sign: wp.float64,
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of a ``humphrey_update_pass``, before the store's narrowing.
    lb = operator_row(offsets, columns, values, b, i)
    return lv[i] - wp.lerp(neighbour_sign * lb, b[i], beta)


@wp.kernel
def humphrey_update_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lv: wp.array[wp.vec3d],
    b: wp.array[wp.vec3d],
    beta: wp.float64,
    neighbour_sign: wp.float64,
    out_next: wp.array[Any],
) -> None:
    # v' = L.v - (beta * b + (1 - beta) * s * L.b), with L.b formed here rather than in a buffer.
    # ``s`` is ``1`` for Vollmer et al.'s update and ``-1`` for MeshLab's, which adds the
    # neighbours' mean push where the paper subtracts it; ``1 * L.b`` is exact, so the paper's
    # update keeps its bits.
    i = wp.int32(wp.tid())
    out_next[i] = out_next.dtype(
        humphrey_update_row(offsets, columns, values, lv, b, beta, neighbour_sign, i)
    )


@wp.kernel
def humphrey_passes_serial(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    original: wp.array[wp.vec3],
    alpha: wp.float64,
    beta: wp.float64,
    neighbour_sign: wp.float64,
    passes: wp.int32,
    out_lv: wp.array[wp.vec3d],
    out_b: wp.array[wp.vec3d],
    out_scratch_0: wp.array[wp.vec3d],
    out_scratch_1: wp.array[wp.vec3d],
    out_x: wp.array[wp.vec3],
) -> None:
    # Every ``filter_humphrey`` pass in one launch of one thread, the CPU device's form: each
    # pass's residual rows, then its update rows, in the rotation of ``vec3_passes_serial``
    # (``original`` is also the first pass's positions). Each row is the pass kernels' own, so the
    # result is the launches' to the bit; ``out_scratch_1`` may alias ``out_scratch_0`` at two
    # passes, where it is unused.
    n = original.shape[0]
    for index in range(passes):
        for i in range(n):
            if index == 0:
                humphrey_residual_row(
                    offsets, columns, values, original, original, alpha, out_lv, out_b, i
                )
            elif index % 2 == 1:
                humphrey_residual_row(
                    offsets, columns, values, out_scratch_0, original, alpha, out_lv, out_b, i
                )
            else:
                humphrey_residual_row(
                    offsets, columns, values, out_scratch_1, original, alpha, out_lv, out_b, i
                )
        for i in range(n):
            value = humphrey_update_row(
                offsets, columns, values, out_lv, out_b, beta, neighbour_sign, i
            )
            if index == passes - 1:
                out_x[i] = wp.vec3(value)
            elif index % 2 == 0:
                out_scratch_0[i] = value
            else:
                out_scratch_1[i] = value


@wp.func
def mut_dif_adil(normal: wp.vec3, position: wp.vec3d, lv: wp.vec3d) -> wp.float64:
    # adil = 1 / max(1e-12, |N . (V - L.V)|), the reciprocal normal-residual magnitude per vertex.
    # One spelling for the pass that sums it and the step that recomputes it from the same three
    # values, so the step's coefficient is bit-identical to the one the mean was taken over.
    residual = wp.abs(wp.dot(to_vec3d(normal), position - lv))
    return wp.float64(1.0) / wp.max(wp.float64(1e-12), residual)


MUT_DIF_ROWS_PER_BLOCK = wp.constant(256)

# ``float32``'s unit roundoff relative to a coordinate: the rounding a ``float32`` input carries.
FLOAT32_ROUNDING = wp.float64(1.1920929e-07)


@wp.func
def mut_dif_adil_rounding(position: wp.vec3d, adil: wp.float64) -> wp.float64:
    # This vertex's term of the mean coefficient's rounding sensitivity: ``adil`` is ``1/r`` with
    # ``r`` the normal residual, so an input rounding ``delta`` of the position moves it by
    # ``adil^2 delta``. Summed over vertices and divided by the sum of ``adil`` it is the relative
    # uncertainty of the mean every step divides by, which a residual near zero (a saddle, an
    # inflection) makes large.
    delta = FLOAT32_ROUNDING * wp.max(
        wp.abs(position[0]), wp.max(wp.abs(position[1]), wp.abs(position[2]))
    )
    return adil * adil * delta


@wp.kernel
def mut_dif_adil_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    normals: wp.array[wp.vec3],
    stride: wp.int32,
    out_lv: wp.array[wp.vec3d],
    out_adil_sum: wp.array[wp.float64],
) -> None:
    # ``out_adil_sum`` holds a ``(coefficients, rounding sensitivity)`` row per block
    # (``mut_dif_adil_rounding``), which the last block adds in a fixed order into slots 0 and 1
    # (``store_block_sum``), so the mean every step divides by repeats bit for bit on CUDA.
    # Applies the operator row and folds each vertex's ``mut_dif_adil`` straight into the mean's
    # sum: ``MUT_DIF_ROWS_PER_BLOCK`` rows per block, lane-strided by ``wp.block_dim()`` (so the
    # CPU device's single lane covers the block's rows), one ``float64`` row per block. The
    # coefficient is not stored: ``mut_dif_step_scaled`` recomputes it at its own vertex from the
    # ``out_lv`` row written here, which costs it a normal load against an ``(n,)`` buffer's write
    # and read and the separate reduction launch. A narrower fold than the reduce module's because
    # the kernel also writes per row. ``positions`` is ``wp.vec3d`` or, on the first pass of a
    # filter with no volume constraint, the ``float32`` input widened as it is read.
    chunk, lane = wp.tid()
    offset, remaining = block_chunk(positions.shape[0], chunk, MUT_DIF_ROWS_PER_BLOCK)
    if remaining <= 0:
        return
    total = wp.float64(0.0)
    rounding = wp.float64(0.0)
    for r in range(lane, remaining, wp.block_dim()):
        i = offset + r
        lv = operator_row(offsets, columns, values, positions, i)
        out_lv[i] = lv
        position = wp.vec3d(positions[i])
        adil = mut_dif_adil(normals[i], position, lv)
        total = total + adil
        rounding = rounding + mut_dif_adil_rounding(position, adil)
    store_block_sum(chunk, lane, wp.vec2d(total, rounding), out_adil_sum, 0, stride)


@wp.func
def add_scaled_normal(v_prev: wp.vec3d, normal: wp.vec3, scale: wp.float64) -> wp.vec3d:
    # v' = v + scale * N; reused for the eps finite-difference probe and the volume correction.
    return v_prev + scale * to_vec3d(normal)


@wp.func
def mut_dif_probe_step(area_max: wp.array[wp.float32]) -> wp.float64:
    # The volume constraint's finite-difference step ``eps = 0.01 * sqrt(max face area)``, formed
    # from the device-resident maximum so that the wrapper reads nothing back to set it. Shared by
    # the probe ``mut_dif_step_scaled`` writes and the slope ``mut_dif_volume_correct`` calibrates.
    return wp.float64(0.01) * wp.sqrt(wp.float64(area_max[0]))


@wp.kernel
def mut_dif_step_scaled(
    positions: wp.array[Any],
    lv: wp.array[wp.vec3d],
    adil_sum: wp.array[wp.float64],
    inv_n: wp.float64,
    lamb: wp.float64,
    normals: wp.array[wp.vec3],
    area_max: wp.array[wp.float32],
    out_next: wp.array[Any],
    out_probe: wp.array[wp.vec3d],
    out_sensitivity: wp.array[wp.float64],
) -> None:
    # ``mut_dif_step`` with the mean coefficient read from a device scalar (adil_sum[0] * inv_n),
    # so the smoothing loop never synchronises with the host. A real kernel rather than wp.map:
    # the length-1 ``adil_sum`` is a uniform argument, which wp.map cannot broadcast.
    #
    # A non-empty ``out_probe`` (the volume constraint's first pass) also receives the stepped
    # vertex offset by ``mut_dif_probe_step`` along its normal: the finite-difference probe
    # ``mut_dif_volume_correct`` calibrates its slope against, formed from the value this thread
    # has just computed rather than read back by a map of its own.
    #
    # Either vector type on the read and the write (``_register_overloads``), as the explicit
    # passes below take them: with no volume constraint the first pass reads the ``float32`` input
    # widened and the last writes the ``float32`` answer narrowed, as ``array.astype`` would.
    i = wp.int32(wp.tid())
    mean_adil = adil_sum[0] * inv_n
    if i == wp.int32(0):
        # The largest relative rounding uncertainty of the mean over the passes, for the warning.
        wp.atomic_max(out_sensitivity, 0, adil_sum[1] / adil_sum[0])
    position = wp.vec3d(positions[i])
    row = lv[i]
    stepped = mut_dif_step(position, row, mut_dif_adil(normals[i], position, row), mean_adil, lamb)
    out_next[i] = out_next.dtype(stepped)
    if out_probe.shape[0] > 0:
        out_probe[i] = add_scaled_normal(stepped, normals[i], mut_dif_probe_step(area_max))


@wp.func
def mut_dif_volume_slope(
    volume_base: wp.array[wp.float64], volume_probe: wp.array[wp.float64], eps: wp.float64
) -> wp.float64:
    # The finite-difference slope d(offset)/d(volume), calibrated once from the first pass's
    # volume and the volume of the same mesh offset by ``eps`` along its normals. Formed here rather
    # than on the host because the only reason to read the two volumes back was to divide them --
    # one pipeline drain per calibration, and the correction below needs them on the device anyway.
    #
    # A zero denominator means the probe displacement did not change the volume at all (a mesh with
    # no faces, or normals orthogonal to every face), and the host version this replaces answered
    # that with a slope of exactly zero rather than an infinity. So does this -- as a branch and
    # not a ``wp.where``, which evaluates both arms and would divide by the zero before discarding
    # the result.
    delta = volume_probe[0] - volume_base[0]
    if delta == wp.float64(0.0):
        return wp.float64(0.0)
    return eps / delta


@wp.kernel
def mut_dif_volume_correct(
    normals: wp.array[wp.vec3],
    volume_initial: wp.array[wp.float64],
    volume_current: wp.array[wp.float64],
    volume_probe: wp.array[wp.float64],
    area_max: wp.array[wp.float32],
    calibrate: wp.int32,
    slope: wp.array[wp.float64],
    out_positions: wp.array[wp.vec3d],
    out_narrowed: wp.array[wp.vec3],
) -> None:
    # Offset every vertex along its normal by ``slope * (volume_initial - volume_current)``, the
    # first-order correction that walks the smoothed mesh's volume back toward the input's. All
    # three scalars are device-resident, so a smoothing pass issues no host synchronisation.
    #
    # Unlike ``rescale_to_volume`` this writes unconditionally, because the host version it
    # replaces did: at a zero slope it applied an offset of exactly zero rather than skipping, and
    # ``v + 0 * N`` is the same value for every finite ``N``. The two therefore agree on a
    # degenerate mesh as well as on an ordinary one, which is the property that matters.
    #
    # The first pass (``calibrate != 0``, warp-uniform) calibrates the slope itself: every thread
    # forms it from the two device volumes (``mut_dif_volume_slope``) rather than reading it, and
    # thread 0 keeps it in ``slope`` for the later passes -- one division per thread against a
    # ``dim=1`` launch, the ``damped_inverse_diagonal`` trade.
    v = wp.int32(wp.tid())
    s = wp.float64(0.0)
    if calibrate != 0:
        s = mut_dif_volume_slope(volume_current, volume_probe, mut_dif_probe_step(area_max))
        if v == 0:
            slope[0] = s
    else:
        s = slope[0]
    offset = s * (volume_initial[0] - volume_current[0])
    corrected = add_scaled_normal(out_positions[v], normals[v], offset)
    # A non-empty ``out_narrowed`` (the last pass) takes the ``float32`` answer, narrowed as
    # ``array.astype`` narrows it, instead of the ``float64`` iterate no later pass reads.
    if out_narrowed.shape[0] > 0:
        out_narrowed[v] = to_vec3(corrected)
    else:
        out_positions[v] = corrected


@wp.func
def extract_components(v: wp.vec3d) -> tuple[wp.float64, wp.float64, wp.float64]:
    return v[0], v[1], v[2]


@wp.func
def combine_components(x: wp.float64, y: wp.float64, z: wp.float64) -> wp.vec3d:
    return wp.vec3d(x, y, z)


@wp.func
def combine_components_narrowed(
    x: wp.float64, y: wp.float64, z: wp.float64
) -> tuple[wp.vec3d, wp.vec3]:
    # ``combine_components`` and the same position narrowed as ``array.astype`` narrows it, for
    # the implicit-fairing pass, whose next pass (or answer) reads the ``wp.vec3`` form.
    position = wp.vec3d(x, y, z)
    return position, to_vec3(position)


@wp.kernel
def mark_free_new_vertices(
    first_new: wp.int32, boundary: wp.array[wp.bool], out_free: wp.array[wp.bool]
) -> None:
    # ``refine_and_smooth_region``'s free set: the vertices from ``first_new`` on (the tail the
    # subdivision appended) that are not on the mesh boundary.
    v = wp.int32(wp.tid())
    out_free[v] = mask_and_not(v >= first_new, boundary[v])


@wp.kernel
def mark_single_use_edge_vertices(
    counts: wp.array[wp.int32], unique_edges: wp.array2d[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # Mark both endpoints of every unique edge used by exactly one face -- a boundary edge, on
    # ``boundary.boundary_edges``' own definition -- given the per-edge use counts. Two edges
    # sharing an endpoint write the same ``True``, so the race is benign. ``out_mask`` starts
    # zeroed.
    e = wp.int32(wp.tid())
    if counts[e] == 1:
        out_mask[unique_edges[e, 0]] = True
        out_mask[unique_edges[e, 1]] = True


@wp.kernel
def operator_max_row_abs_sum(
    offsets: wp.array[wp.int32], values: wp.array[wp.float32], out_max: wp.array[wp.float64]
) -> None:
    # ``max_i sum_j |L_ij|``, the infinity norm the fixed-point iteration below contracts by, in
    # one block fold over rows: lanes stride the block's rows by ``wp.block_dim()`` (right on the
    # CPU device too, where a block is one lane), one ``block_max`` and one commit per block into
    # ``out_max`` (zero-initialized; every row sum is non-negative). An empty row reads 1:
    # ``operator_row`` applies it as the identity.
    i, t = wp.tid()
    n = offsets.shape[0] - 1
    base, remaining = block_chunk_1d(n, i)
    if remaining <= 0:
        return
    local = wp.float64(0.0)
    for k in range(t, remaining, wp.block_dim()):
        row = base + k
        start = offsets[row]
        end = offsets[row + 1]
        total = wp.float64(0.0)
        for e in range(start, end):
            total += wp.abs(wp.float64(values[e]))
        local = wp.max(local, wp.where(end == start, wp.float64(1.0), total))
    block = block_max(local)
    if t == 0:
        wp.atomic_max(out_max, 0, block)


@wp.func
def implicit_laplacian_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lamb: wp.float64,
    rhs: wp.array[Any],
    x: wp.array[Any],
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of one fixed-point step: ``(rhs + lamb L x) / (1 + lamb)``.
    return (wp.vec3d(rhs[i]) + lamb * operator_row(offsets, columns, values, x, i)) / (
        wp.float64(1.0) + lamb
    )


@wp.kernel
def implicit_laplacian_step(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lamb: wp.float64,
    rhs: wp.array[Any],
    x: wp.array[Any],
    out_x: wp.array[wp.vec3d],
) -> None:
    # One step of the fixed-point iteration for the backward-Euler system
    # ``((1 + lamb) I - lamb L) x = rhs``: ``x' = (rhs + lamb L x) / (1 + lamb)``. It contracts by
    # ``lamb ||L||_inf / (1 + lamb)`` whatever ``L``'s symmetry, which conjugate gradient needs.
    # Each thread reads ``rhs`` at its own row only, so ``out_x`` may alias ``rhs`` -- the last
    # step writes the answer over it -- but never ``x``, which the row sum reads at other rows.
    # Shares ``operator_row`` with the explicit step, empty-row convention included. ``rhs`` and
    # ``x`` are ``wp.vec3d``, or on a filter's first pass the ``float32`` input widened as it is
    # read (``_register_overloads``). The row is ``implicit_laplacian_row``, shared with
    # ``implicit_laplacian_steps_serial``.
    i = wp.int32(wp.tid())
    out_x[i] = implicit_laplacian_row(offsets, columns, values, lamb, rhs, x, i)


@wp.kernel
def implicit_laplacian_steps_serial(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lamb: wp.float64,
    steps: wp.int32,
    rhs: wp.array[Any],
    out_scratch_0: wp.array[wp.vec3d],
    out_scratch_1: wp.array[wp.vec3d],
    out_x: wp.array[wp.vec3d],
) -> None:
    # Every ``implicit_laplacian_step`` of one pass in one launch of one thread, walking each
    # step's rows in turn: the CPU device's form, where a launch's threads run one after another
    # anyway. The rotation is ``smoothing._implicit_fixed_point_pass``'s: step ``k`` reads ``rhs``
    # (``k == 0``) or ``out_scratch_{(k - 1) % 2}`` and writes ``out_scratch_{k % 2}``, the last
    # ``out_x``.
    # Each row is ``implicit_laplacian_row``, the step kernel's own, so the result is the launch
    # sequence's to the bit; ``out_x`` may alias ``rhs`` as there (read at its own row only).
    n = out_x.shape[0]
    for step in range(steps):
        for i in range(n):
            value = wp.vec3d()
            if step == 0:
                value = implicit_laplacian_row(offsets, columns, values, lamb, rhs, rhs, i)
            elif step % 2 == 1:
                value = implicit_laplacian_row(
                    offsets, columns, values, lamb, rhs, out_scratch_0, i
                )
            else:
                value = implicit_laplacian_row(
                    offsets, columns, values, lamb, rhs, out_scratch_1, i
                )
            if step == steps - 1:
                out_x[i] = value
            elif step % 2 == 0:
                out_scratch_0[i] = value
            else:
                out_scratch_1[i] = value


@wp.kernel
def implicit_laplacian_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lamb: wp.float64,
    nnz: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # Triplets for AA = (1 + lambda) * I - lambda * L (backward-Euler system, Article 2), where
    # L is the row-stochastic averaging operator. Off-diagonals reuse L's CSR positions; one
    # diagonal triplet per row is appended after the nnz off-diagonals.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    for k in range(start, end):
        out_rows[k] = i
        out_cols[k] = columns[k]
        out_vals[k] = -lamb * wp.float64(values[k])
    diag = nnz + i
    out_rows[diag] = i
    out_cols[diag] = i
    out_vals[diag] = wp.float64(1.0) + lamb


@wp.func
def diffused_scalar(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[wp.float32],
    lamb: wp.float32,
    i: wp.int32,
) -> wp.float32:
    # Row ``i`` of a ``diffuse_scalar_pass``.
    value = field[i]
    start = offsets[i]
    end = offsets[i + 1]
    average = value
    if end > start:
        total = wp.float32(0.0)
        for k in range(start, end):
            total += values[k] * field[columns[k]]
        average = total
    return wp.lerp(value, average, lamb)


@wp.kernel
def diffuse_scalar_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[wp.float32],
    lamb: wp.float32,
    out_next: wp.array[wp.float32],
) -> None:
    # The scalar counterpart of ``diffuse_vec3_pass``: one row of the row-stochastic averaging
    # operator and the diffusion step that consumes it, in the same thread, so the pass issues one
    # launch and never materializes the intermediate average. ``lamb = 1`` replaces the value
    # outright, which is MeshLab's single pass.
    #
    # The row is spelled out here rather than sharing ``kernels/laplacian.operator_row``, and the
    # duplication is deliberate. That helper promotes the float32 weight to float64 because its
    # accumulator is a ``wp.vec3d``; a scalar field is float32 end to end, and ``float64 *
    # float32`` is a hard parse error in Warp ("Input types must be the same"), so no single
    # spelling serves both. Sharing it would mean carrying the field in float64, which was built
    # and measured as a **loss** at every size, because the field is one of four streams the CSR
    # walk reads and doubling its width costs bandwidth the launch saving cannot repay -- before
    # counting the two conversion passes a float64 iterate would add per call.
    #
    # An isolated vertex (empty row) keeps its own value, so it neither drifts to zero nor
    # contaminates its (nonexistent) neighbours. The row is ``diffused_scalar``, shared with
    # ``diffuse_scalar_passes_serial``.
    i = wp.int32(wp.tid())
    out_next[i] = diffused_scalar(offsets, columns, values, field, lamb, i)


@wp.kernel
def diffuse_scalar_passes_serial(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[wp.float32],
    lamb: wp.float32,
    passes: wp.int32,
    out_next_0: wp.array[wp.float32],
    out_next_1: wp.array[wp.float32],
) -> None:
    # Every ``diffuse_scalar_pass`` of a call in one launch of one thread, the CPU device's form:
    # pass 0 reads ``field``, pass ``k`` writes ``out_next_{k % 2}`` and the next reads it, each
    # row ``diffused_scalar``, so the result is the launches' to the bit. ``out_next_1`` may
    # alias ``out_next_0`` at one pass, where it is unused.
    n = field.shape[0]
    for index in range(passes):
        for i in range(n):
            value = wp.float32(0.0)
            if index == 0:
                value = diffused_scalar(offsets, columns, values, field, lamb, i)
            elif index % 2 == 1:
                value = diffused_scalar(offsets, columns, values, out_next_0, lamb, i)
            else:
                value = diffused_scalar(offsets, columns, values, out_next_1, lamb, i)
            if index % 2 == 0:
                out_next_0[i] = value
            else:
                out_next_1[i] = value


@wp.kernel
def scatter_incidence_rows(
    indices: wp.array[wp.int32],
    arity: wp.int32,
    offsets: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_items: wp.array[wp.int32],
) -> None:
    # The incidence CSR of a flat ``arity``-wide index buffer: row ``r`` lists every item
    # ``e // arity`` with an entry ``e`` naming ``r`` -- a vertex's faces from the face buffer
    # (``arity = 3``), a face's adjacency rows from the flattened ``(m, 2)`` pairs (``arity = 2``).
    # One thread per entry. ``cursor`` arrives holding each row's length and counts it down, and
    # the slot is taken from the row's end, so the CPU device, which runs the entries in turn,
    # fills every row in ascending entry order -- the order the item-major scatters these rows
    # replace committed their atomics in -- with no cleared cursor; a CUDA device fills in arrival
    # order unless the caller sorts the rows (the deterministic setting).
    e = wp.int32(wp.tid())
    r = indices[e]
    out_items[offsets[r + 1] - wp.atomic_add(cursor, r, -1)] = e // arity


@wp.func
def agreeing_neighbor(
    face_normals: wp.array[wp.vec3],
    f: wp.int32,
    f0: wp.int32,
    f1: wp.int32,
    threshold_cos: wp.float32,
) -> wp.int32:
    # The neighbour face ``f`` averages with across the adjacent pair ``(f0, f1)`` it belongs to,
    # or ``-1`` when the two normals disagree by more than the crease threshold. That gate is the
    # whole point of ``filter_normals``: across a crease the two normals disagree and simply do not
    # average, so a sharp edge survives an arbitrary number of passes while noise on a flat region
    # diffuses away. The one decision both pass kernels below make, whichever table names the pair.
    if wp.dot(face_normals[f0], face_normals[f1]) <= threshold_cos:
        return -1
    return wp.where(f0 == f, f1, f0)


@wp.func
def smoothed_normal_over_pairs(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    face_adjacency: wp.array2d[wp.int32],
    pair_offsets: wp.array[wp.int32],
    face_pairs: wp.array[wp.int32],
    threshold_cos: wp.float32,
    f: wp.int32,
) -> wp.vec3:
    # Face ``f``'s entry of a ``smooth_face_normals`` pass.
    accumulated = face_areas[f] * face_normals[f]
    for slot in range(pair_offsets[f], pair_offsets[f + 1]):
        k = face_pairs[slot]
        other = agreeing_neighbor(
            face_normals, f, face_adjacency[k, 0], face_adjacency[k, 1], threshold_cos
        )
        if other >= 0:
            accumulated += face_areas[other] * face_normals[other]
    return wp.normalize(accumulated)


@wp.kernel
def smooth_face_normals(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    face_adjacency: wp.array2d[wp.int32],
    pair_offsets: wp.array[wp.int32],
    face_pairs: wp.array[wp.int32],
    threshold_cos: wp.float32,
    out_normals: wp.array[wp.vec3],
) -> None:
    # One pass of ``filter_normals`` over a caller's adjacency pairs, gathered per face: the
    # normalized area-weighted sum of the face's own normal (so a face whose every neighbour is
    # across a crease keeps its own) and its ``agreeing_neighbor``s, read in the order of the
    # face's ``scatter_incidence_rows`` row. One launch a pass, where a pair-major scatter into an
    # accumulator needed a second launch to normalize it; ``face_normals`` and ``out_normals`` are
    # the caller's two ping-pong buffers. ``smooth_face_normals_over_mates`` is the same pass over
    # the halfedge mates, for an adjacency this module derives itself.
    f = wp.int32(wp.tid())
    out_normals[f] = smoothed_normal_over_pairs(
        face_normals, face_areas, face_adjacency, pair_offsets, face_pairs, threshold_cos, f
    )


@wp.kernel
def smooth_face_normals_serial(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    face_adjacency: wp.array2d[wp.int32],
    pair_offsets: wp.array[wp.int32],
    face_pairs: wp.array[wp.int32],
    threshold_cos: wp.float32,
    passes: wp.int32,
    out_normals_0: wp.array[wp.vec3],
    out_normals_1: wp.array[wp.vec3],
) -> None:
    # Every ``smooth_face_normals`` pass of a call in one launch of one thread, the CPU device's
    # form (its threads run one after another anyway). Pass ``k`` reads ``face_normals``
    # (``k == 0``) or the buffer pass ``k - 1`` wrote and writes ``out_normals_{k % 2}``, the
    # rotation of ``smoothing._smoothed_face_normals``; each face is
    # ``smoothed_normal_over_pairs``, the pass kernel's own, so the result is the launches' to the
    # bit. ``out_normals_1`` may alias ``face_normals``, which only pass 0 reads.
    n = face_normals.shape[0]
    for index in range(passes):
        for f in range(n):
            value = wp.vec3()
            if index == 0:
                value = smoothed_normal_over_pairs(
                    face_normals, face_areas, face_adjacency, pair_offsets, face_pairs,
                    threshold_cos, f,
                )  # fmt: skip
            elif index % 2 == 1:
                value = smoothed_normal_over_pairs(
                    out_normals_0, face_areas, face_adjacency, pair_offsets, face_pairs,
                    threshold_cos, f,
                )  # fmt: skip
            else:
                value = smoothed_normal_over_pairs(
                    out_normals_1, face_areas, face_adjacency, pair_offsets, face_pairs,
                    threshold_cos, f,
                )  # fmt: skip
            if index % 2 == 0:
                out_normals_0[f] = value
            else:
                out_normals_1[f] = value


@wp.func
def smoothed_normal_over_mates(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    mates: wp.array[wp.int32],
    threshold_cos: wp.float32,
    f: wp.int32,
) -> wp.vec3:
    # Face ``f``'s entry of a ``smooth_face_normals_over_mates`` pass.
    accumulated = face_areas[f] * face_normals[f]
    for k in range(3):
        mate = mates[3 * f + k]
        if mate >= 0:
            other = agreeing_neighbor(face_normals, f, f, mate // 3, threshold_cos)
            if other >= 0:
                accumulated += face_areas[other] * face_normals[other]
    return wp.normalize(accumulated)


@wp.kernel
def smooth_face_normals_over_mates(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    mates: wp.array[wp.int32],
    threshold_cos: wp.float32,
    out_normals: wp.array[wp.vec3],
) -> None:
    # ``smooth_face_normals`` with the neighbours read off ``halfedge.halfedge_mates``: the face
    # across each of the three edges, where exactly two halfedges span it -- the pairs
    # ``adjacency.face_adjacency`` lists -- in corner order. A fixed three reads per face, with no
    # face -> pair incidence table to build.
    f = wp.int32(wp.tid())
    out_normals[f] = smoothed_normal_over_mates(face_normals, face_areas, mates, threshold_cos, f)


@wp.kernel
def smooth_face_normals_over_mates_serial(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    mates: wp.array[wp.int32],
    threshold_cos: wp.float32,
    passes: wp.int32,
    out_normals_0: wp.array[wp.vec3],
    out_normals_1: wp.array[wp.vec3],
) -> None:
    # ``smooth_face_normals_serial`` over the halfedge mates: every pass in one launch of one
    # thread, each face ``smoothed_normal_over_mates``, the same rotation and aliasing.
    n = face_normals.shape[0]
    for index in range(passes):
        for f in range(n):
            value = wp.vec3()
            if index == 0:
                value = smoothed_normal_over_mates(
                    face_normals, face_areas, mates, threshold_cos, f
                )
            elif index % 2 == 1:
                value = smoothed_normal_over_mates(
                    out_normals_0, face_areas, mates, threshold_cos, f
                )
            else:
                value = smoothed_normal_over_mates(
                    out_normals_1, face_areas, mates, threshold_cos, f
                )
            if index % 2 == 0:
                out_normals_0[f] = value
            else:
                out_normals_1[f] = value


@wp.kernel
def corner_cross_normals(
    vertices: wp.array[Any], faces: wp.array[wp.int32], out_normals: wp.array[Any]
) -> None:
    # A face's unit normal as vcg's ``PerFaceNormalized`` forms it, ``(v1 - v0) x (v2 - v0)``
    # normalized, in the positions' precision: the field MeshLab's two-step normal passes start
    # from. ``triangles.face_normals_and_area`` crosses at the largest angle in ``float32``, the
    # default filter's field; the two differ by rounding only.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    out_normals[f] = wp.normalize(wp.cross(vertices[b] - vertices[a], vertices[c] - vertices[a]))


@wp.func
def smoothed_normal_meshlab(
    face_normals: wp.array[Any],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    threshold_cos: wp.Float,
    f: wp.int32,
) -> Any:
    # Face ``f``'s entry of a ``smooth_face_normals_meshlab`` pass.
    normal = face_normals[f]
    sigma = wp.max(threshold_cos, type(threshold_cos)(0.0))
    c0, c1, c2 = corner_triple(faces, f)
    corners = wp.vec3i(c0, c1, c2)
    accumulated = normal - normal
    for k in range(3):
        for slot in range(vf_offsets[corners[k]], vf_offsets[corners[k] + 1]):
            g = vertex_faces[slot]
            counted = g > f
            for j in range(k):
                if local_corner(faces, g, corners[j]) >= 0:
                    counted = True
            if not counted:
                other = face_normals[g]
                c = wp.clamp(wp.dot(other, normal), type(sigma)(1e-4), type(sigma)(1.0))
                if c >= sigma:
                    w = c - sigma
                    accumulated += other * (w * w)
    return wp.normalize(accumulated)


@wp.kernel
def smooth_face_normals_meshlab(
    face_normals: wp.array[Any],
    face_areas: wp.array[wp.float32],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    threshold_cos: wp.Float,
    out_normals: wp.array[Any],
) -> None:
    # One normal pass of MeshLab's two-step smoothing, as its output measures: the normalized sum
    # of the vertex-adjacent faces' normals, each weighted ``(c - sigma)^2`` where
    # ``c = clamp(dot, 1e-4, 1)`` reaches ``sigma = max(threshold_cos, 0)``; no area weight
    # (``face_areas`` is read by ``smooth_face_normals_over_mates``, the pass this replaces, and
    # takes the slot so both share ``_smoothed_face_normals``). Two algorithms, not a parameter:
    # that one is an area-weighted sum over the three edge neighbours behind a binary gate.
    #
    # The neighbourhood is not the whole vertex ring: MeshLab's answer depends on the face order,
    # and equals averaging face ``f`` with ``f`` itself and the faces of *lower* index around its
    # corners, each counted once. Here: the rows' faces ``g <= f``, skipping a ``g`` that also
    # holds an earlier corner's vertex (already counted there), so the row order does not matter.
    # A Jacobi pass: ``face_normals`` is read-only throughout. Generic over the normals'
    # precision: ``filter_two_step(method="meshlab")`` runs it in ``float64``, MeshLab's. The face
    # is ``smoothed_normal_meshlab``, shared with ``smooth_face_normals_meshlab_serial``.
    f = wp.int32(wp.tid())
    out_normals[f] = smoothed_normal_meshlab(
        face_normals, faces, vertex_faces, vf_offsets, threshold_cos, f
    )


@wp.kernel
def smooth_face_normals_meshlab_serial(
    face_normals: wp.array[Any],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    threshold_cos: wp.Float,
    passes: wp.int32,
    out_normals_0: wp.array[Any],
    out_normals_1: wp.array[Any],
) -> None:
    # ``smooth_face_normals_serial`` for MeshLab's pass: every pass in one launch of one thread,
    # each face ``smoothed_normal_meshlab``, the same rotation and aliasing.
    n = face_normals.shape[0]
    for index in range(passes):
        for f in range(n):
            value = face_normals[f]
            if index == 0:
                value = smoothed_normal_meshlab(
                    face_normals, faces, vertex_faces, vf_offsets, threshold_cos, f
                )
            elif index % 2 == 1:
                value = smoothed_normal_meshlab(
                    out_normals_0, faces, vertex_faces, vf_offsets, threshold_cos, f
                )
            else:
                value = smoothed_normal_meshlab(
                    out_normals_1, faces, vertex_faces, vf_offsets, threshold_cos, f
                )
            if index % 2 == 0:
                out_normals_0[f] = value
            else:
                out_normals_1[f] = value


@wp.func
def fitted_vertex(
    vertices: wp.array[Any],
    faces: wp.array[wp.int32],
    face_normals: wp.array[Any],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    v: wp.int32,
) -> Any:
    # Vertex ``v``'s entry of a ``fit_vertices_to_normals`` step.
    position = vertices[v]
    start = vf_offsets[v]
    end = vf_offsets[v + 1]
    delta = position - position
    for slot in range(start, end):
        f = vertex_faces[slot]
        i0, i1, i2 = corner_triple(faces, f)
        normal = face_normals[f]
        centroid = (vertices[i0] + vertices[i1] + vertices[i2]) / type(position[0])(3.0)
        delta += normal * wp.dot(normal, centroid - position)
    if end > start:
        position = position + delta / type(position[0])(end - start)
    return position


@wp.kernel
def fit_vertices_to_normals(
    vertices: wp.array[Any],
    faces: wp.array[wp.int32],
    face_normals: wp.array[Any],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    out_positions: wp.array[Any],
) -> None:
    # One gradient step of the vertex-fitting half of two-step smoothing (Ohtake et al.): each
    # incident face wants its corner to lie in the plane through the face centroid with the
    # *filtered* normal, and the correction is the component of that offset along the normal.
    #
    # Gathered per vertex over its vertex -> face ``scatter_incidence_rows`` row (one entry per
    # corner reference) and divided by the row's length, which is the step size that makes the
    # iteration a contraction without a tuning constant. ``vertices`` and ``out_positions`` are the
    # caller's two ping-pong buffers, so every correction reads the previous step's positions.
    # Generic over the precision, positions and normals alike (``float64`` for MeshLab's method).
    # The vertex is ``fitted_vertex``, shared with ``fit_vertices_to_normals_serial``.
    v = wp.int32(wp.tid())
    out_positions[v] = fitted_vertex(vertices, faces, face_normals, vertex_faces, vf_offsets, v)


@wp.kernel
def fit_vertices_to_normals_serial(
    vertices: wp.array[Any],
    faces: wp.array[wp.int32],
    face_normals: wp.array[Any],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    steps: wp.int32,
    out_positions_0: wp.array[Any],
    out_positions_1: wp.array[Any],
) -> None:
    # Every ``fit_vertices_to_normals`` step of an outer pass in one launch of one thread, the
    # CPU device's form. Step ``k`` reads ``vertices`` (``k == 0``) or the buffer step ``k - 1``
    # wrote and writes ``out_positions_{k % 2}``; each vertex is ``fitted_vertex``, the step
    # kernel's own, so the result is the launches' to the bit. ``out_positions_1`` may alias
    # ``vertices``, which only step 0 reads.
    n = vertices.shape[0]
    for index in range(steps):
        for v in range(n):
            value = vertices[v]
            if index == 0:
                value = fitted_vertex(vertices, faces, face_normals, vertex_faces, vf_offsets, v)
            elif index % 2 == 1:
                value = fitted_vertex(
                    out_positions_0, faces, face_normals, vertex_faces, vf_offsets, v
                )
            else:
                value = fitted_vertex(
                    out_positions_1, faces, face_normals, vertex_faces, vf_offsets, v
                )
            if index % 2 == 0:
                out_positions_0[v] = value
            else:
                out_positions_1[v] = value


# The Mumford-Shah normal denoiser of Zhang et al. (2015), "Mesh Denoising via a Novel
# Mumford-Shah Framework", eqs. 19-20, and the least-squares vertex update after it. The face
# system's kernels come first (pattern, geometry, values, the normalized solution), then the
# vertex system's (pattern, projection, right-hand side). ``smoothing._MumfordShahNormals`` and
# ``smoothing._NormalsToPoints`` drive them.
#
# Every quantity is formed in ``float32`` up to the system entries, which are ``float64``, so a
# solve reproduces a reference holding ``float32`` geometry in ``float64`` matrices.

# The indicator's ``epsilon`` (eq. 20): an edge's weight ``rho = beta / (2 epsilon)``.
MUMFORD_SHAH_EPSILON = wp.float32(0.001)


@wp.kernel
def mumford_shah_face_keys(
    mates: wp.array[wp.int32],
    n_faces: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The face system's pattern for ``array.csr_from_keys``: slot ``4 f + k`` is face ``f``'s entry
    # with the face across its halfedge ``3 f + k`` (the sentinel on a rim or a non-manifold edge,
    # which have no partner face), slot ``4 f + 3`` its diagonal. The payload is the slot, which
    # ``mumford_shah_face_values`` decodes.
    f = wp.int32(wp.tid())
    for k in range(3):
        mate = mates[3 * f + k]
        partner = wp.where(mate >= 0, mate // 3, wp.int32(-1))
        store_coo_key(4 * f + k, f, partner, n_faces, n_faces, out_keys, out_order)
    store_coo_key(4 * f + 3, f, f, n_faces, n_faces, out_keys, out_order)


@wp.kernel
def mumford_shah_face_geometry(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    out_lengths: wp.array[wp.float32],
    out_perimeters: wp.array[wp.float32],
    out_rhs: wp.array2d[wp.float64],
    out_solution: wp.array2d[wp.float64],
) -> None:
    # What the face system reads of the geometry: every halfedge's length (``3 f + k`` runs from
    # corner ``k`` to ``k + 1``), each face's perimeter summed in corner order -- rim edges
    # included, so a rim face's neighbours weigh as much as an interior face's -- and the
    # original normals as the three right-hand-side columns, which also seed the first solve.
    f = wp.int32(wp.tid())
    perimeter = wp.float32(0.0)
    for k in range(3):
        a = vertices[faces[3 * f + k]]
        b = vertices[faces[3 * f + (k + 1) % 3]]
        length = wp.length(b - a)
        out_lengths[3 * f + k] = length
        perimeter += length
    out_perimeters[f] = perimeter
    normal = normals[f]
    for c in range(3):
        out_rhs[c, f] = wp.float64(normal[c])
        out_solution[c, f] = wp.float64(normal[c])


@wp.func
def solved_normal(solution: wp.array2d[wp.float64], f: wp.int32) -> wp.vec3:
    # Face ``f``'s unit normal from the face system's ``float64`` solution columns.
    return wp.normalize(
        wp.vec3(wp.float32(solution[0, f]), wp.float32(solution[1, f]), wp.float32(solution[2, f]))
    )


@wp.func
def mumford_shah_edge_indicator(
    a: wp.vec3, b: wp.vec3, gamma: wp.float32, beta: wp.float32
) -> wp.float32:
    # The edge indicator's closed form (eq. 20 without its smoothness term): ``rho / (rho + 2 gamma
    # |a - b|^2)``, ``rho = beta / (2 epsilon)``. Near 1 where the two normals agree, near 0 across
    # a crease. Symmetric in ``a`` and ``b`` bit for bit, so both faces of an edge read one value.
    rho = beta / (wp.float32(2.0) * MUMFORD_SHAH_EPSILON)
    return rho / (rho + wp.float32(2.0) * gamma * wp.length_sq(a - b))


@wp.kernel
def mumford_shah_face_values(
    offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    order: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    lengths: wp.array[wp.float32],
    perimeters: wp.array[wp.float32],
    solution: wp.array2d[wp.float64],
    first: wp.int32,
    gamma: wp.float32,
    beta: wp.float32,
    out_values: wp.array[wp.float64],
) -> None:
    # Row ``f`` of the face system (eq. 19): ``n_f + sum_e w_e (n_f - n_r) = n0_f`` over the faces
    # ``r`` across ``f``'s edges, ``w_e = gamma |e| v_e^2 / P``. The indicator ``v_e`` is 1 on the
    # first solve and otherwise ``mumford_shah_edge_indicator`` of the previous solve's normals.
    #
    # ``P`` is a face's perimeter, which makes the stencil row-scaled and so not symmetric: the
    # diagonal takes ``f``'s own, and the coupling of a face pair the *higher-numbered* face's,
    # which symmetrizes the system as a solver reading only its lower triangle does. A face with
    # a zero perimeter couples to nothing through its own row.
    f = wp.int32(wp.tid())
    own = perimeters[f]
    normal = wp.vec3(0.0, 0.0, 0.0)
    if first == 0:
        normal = solved_normal(solution, f)
    weights = wp.vec3(0.0, 0.0, 0.0)
    central = wp.float32(1.0)
    for k in range(3):
        mate = mates[3 * f + k]
        if mate >= 0:
            indicator = wp.float32(1.0)
            if first == 0:
                indicator = mumford_shah_edge_indicator(
                    normal, solved_normal(solution, mate // 3), gamma, beta
                )
            weight = gamma * lengths[3 * f + k] * (indicator * indicator)
            weights[k] = weight
            if own > wp.float32(0.0):
                central += weight / own
    for entry in range(offsets[f], offsets[f + 1]):
        value = wp.float64(0.0)
        for p in range(starts[entry], starts[entry + 1]):
            k = order[p] - 4 * f
            if k == 3:
                value += wp.float64(central)
            else:
                partner = mates[3 * f + k] // 3
                scale = perimeters[wp.max(f, partner)]
                if scale > wp.float32(0.0):
                    value -= wp.float64(weights[k] / scale)
        out_values[entry] = value


@wp.kernel
def normalize_solved_normals(
    solution: wp.array2d[wp.float64], out_normals: wp.array[wp.vec3]
) -> None:
    # The face system's answer as unit normals, once after its last solve.
    f = wp.int32(wp.tid())
    out_normals[f] = solved_normal(solution, f)


@wp.func
def corner_pair_gram(i: wp.int32, j: wp.int32) -> wp.float64:
    # Entry ``(i, j)`` of ``G^T G`` for a face's two shape rows ``G = [[2, -1, -1], [-1, 2, -1]]``
    # (corner 0's and corner 1's offset from the other two): ``[[5, -4, -1], [-4, 5, -1],
    # [-1, -1, 2]]``. Integers, so any summation order is exact.
    diagonal = wp.where(i == 2, wp.float64(2.0), wp.float64(5.0))
    coupling = wp.where(i + j == 1, wp.float64(-4.0), wp.float64(-1.0))
    return wp.where(i == j, diagonal, coupling)


@wp.kernel
def normals_to_points_keys(
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    n_vertices: wp.int32,
    guide_weight_sq: wp.float64,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # The vertex update's normal equations ``(w^2 I + M^T M)`` for ``array.csr_from_coo_keys``:
    # nine corner-pair contributions per face at slots ``9 f + 3 i + j``, then every vertex's
    # guide term ``w^2`` at ``9 n_faces + v``.
    t = wp.int32(wp.tid())
    if t < n_faces:
        for i in range(3):
            for j in range(3):
                slot = 9 * t + 3 * i + j
                store_coo_key(
                    slot,
                    faces[3 * t + i],
                    faces[3 * t + j],
                    n_vertices,
                    n_vertices,
                    out_keys,
                    out_order,
                )
                out_values[slot] = corner_pair_gram(i, j)
    if t < n_vertices:
        slot = 9 * n_faces + t
        store_coo_key(slot, t, t, n_vertices, n_vertices, out_keys, out_order)
        out_values[slot] = guide_weight_sq


@wp.kernel
def vec3_to_columns(values: wp.array[wp.vec3], out_columns: wp.array2d[wp.float64]) -> None:
    # A ``vec3`` field as the three ``float64`` columns a column solver reads and writes.
    i = wp.int32(wp.tid())
    value = values[i]
    for c in range(3):
        out_columns[c, i] = wp.float64(value[c])


@wp.kernel
def columns_to_vec3(columns: wp.array2d[wp.float64], out_values: wp.array[wp.vec3]) -> None:
    # The inverse of ``vec3_to_columns``, rounding each entry to ``float32``.
    i = wp.int32(wp.tid())
    out_values[i] = wp.vec3(
        wp.float32(columns[0, i]), wp.float32(columns[1, i]), wp.float32(columns[2, i])
    )


@wp.func
def collapse_to_line(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    # The degenerate triangle nearest ``(p0, p1, p2)`` with the same centroid: every corner
    # projected onto the line through the centroid toward the corner farthest from it (the first
    # of equally far ones).
    c = (p0 + p1 + p2) / wp.float32(3.0)
    far = p0 - c
    far_sq = wp.length_sq(far)
    if wp.length_sq(p1 - c) > far_sq:
        far = p1 - c
        far_sq = wp.length_sq(far)
    if wp.length_sq(p2 - c) > far_sq:
        far = p2 - c
    d = wp.normalize(far)
    return (c + d * wp.dot(d, p0 - c), c + d * wp.dot(d, p1 - c), c + d * wp.dot(d, p2 - c))


@wp.kernel
def project_to_target_normals(
    solution: wp.array2d[wp.float64],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    out_corner_terms: wp.array[wp.vec3d],
) -> None:
    # Each face's target shape for the vertex update: the current triangle (the ``float32``
    # rounding of the solution columns) projected onto the plane through its centroid with the
    # face's target normal, or collapsed onto a line when the projection turns it over. Its two
    # shape rows ``d0 = 2 q0 - q1 - q2`` and ``d1 = 2 q1 - q0 - q2`` enter the right-hand side as
    # ``G^T (d0, d1)``, written per corner for ``normals_to_points_rhs`` to gather.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    p0 = wp.vec3(
        wp.float32(solution[0, i0]), wp.float32(solution[1, i0]), wp.float32(solution[2, i0])
    )
    p1 = wp.vec3(
        wp.float32(solution[0, i1]), wp.float32(solution[1, i1]), wp.float32(solution[2, i1])
    )
    p2 = wp.vec3(
        wp.float32(solution[0, i2]), wp.float32(solution[1, i2]), wp.float32(solution[2, i2])
    )
    n = normals[f]
    c = (p0 + p1 + p2) / wp.float32(3.0)
    q0 = p0 - n * wp.dot(n, p0 - c)
    q1 = p1 - n * wp.dot(n, p1 - c)
    q2 = p2 - n * wp.dot(n, p2 - c)
    if wp.dot(n, wp.cross(q1 - q0, q2 - q0)) < wp.float32(0.0):
        q0, q1, q2 = collapse_to_line(q0, q1, q2)
    d0 = wp.vec3d(wp.float32(2.0) * q0 - q1 - q2)
    d1 = wp.vec3d(wp.float32(2.0) * q1 - q0 - q2)
    two = wp.float64(2.0)
    out_corner_terms[3 * f] = two * d0 - d1
    out_corner_terms[3 * f + 1] = two * d1 - d0
    out_corner_terms[3 * f + 2] = -d0 - d1


@wp.kernel
def normals_to_points_rhs(
    guide: wp.array[wp.vec3],
    corner_offsets: wp.array[wp.int32],
    corners: wp.array[wp.int32],
    corner_terms: wp.array[wp.vec3d],
    guide_weight_sq: wp.float64,
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Vertex ``v``'s right-hand side ``w^2 g_v + (M^T d)_v``, gathered over its corners (one
    # vertex -> corner ``smoothing._incidence_rows`` row, ascending under the deterministic
    # setting).
    v = wp.int32(wp.tid())
    total = guide_weight_sq * wp.vec3d(guide[v])
    for slot in range(corner_offsets[v], corner_offsets[v + 1]):
        total += corner_terms[corners[slot]]
    for c in range(3):
        out_rhs[c, v] = total[c]


@wp.func
def unsharp_step(
    position: wp.vec3, smoothed: wp.vec3, weight: wp.float32, weight_original: wp.float32
) -> wp.vec3:
    # MeshLab's ``apply_coord_unsharp_mask``: add back a multiple of the high-frequency detail the
    # smoothing pass removed. ``weight_original = 1`` keeps the surface in place and only sharpens.
    return weight_original * position + weight * (position - smoothed)


@wp.kernel
def sharpen_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    coeff: wp.float64,
    original: wp.array[wp.vec3],
    weight: wp.float32,
    weight_original: wp.float32,
    out_sharpened: wp.array[wp.vec3],
) -> None:
    # ``filter_sharpen``'s last smoothing pass and its unsharp mask in one thread: the
    # ``diffuse_vec3_pass`` step narrowed to ``wp.vec3`` as that kernel stores it, then
    # ``unsharp_step`` against the original position, so the smoothed field is never stored.
    i = wp.int32(wp.tid())
    lv = operator_row(offsets, columns, values, positions, i)
    smoothed = wp.vec3(wp.lerp(wp.vec3d(positions[i]), lv, coeff))
    out_sharpened[i] = unsharp_step(original[i], smoothed, weight, weight_original)


@wp.kernel
def graph_sharpen_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[Any],
    weight: wp.float64,
    out_next: wp.array[Any],
) -> None:
    # Open3D's ``filter_sharpen`` step v' = v + weight * sum_j (v - v_j) over the undirected
    # 1-ring, read off the graph Laplacian ``G = A - D`` as v - weight * (G.v).
    # ``diffuse_vec3_pass`` steps toward a row-stochastic average instead; the two share
    # ``operator_row``.
    i = wp.int32(wp.tid())
    out_next[i] = out_next.dtype(
        graph_sharpened_row(offsets, columns, values, positions, weight, i)
    )


@wp.func
def step_along_normal(position: wp.vec3, normal: wp.vec3, distance: wp.float32) -> wp.vec3:
    """Move a vertex along its own normal, which is one half of an inflation step."""
    return position + normal * distance


@wp.kernel
def inflate_displace(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    distance: wp.float32,
    out_displaced: wp.array[wp.vec3],
    out_displaced_wide: wp.array[wp.vec3d],
) -> None:
    # One inflation step: each vertex moved along its area-weighted normal, gathered over its
    # incident faces as it is read (``vertices.vertex_normals``' arithmetic, so the normals are
    # never written and no accumulator is zeroed and scattered per pass), stored at both
    # precisions -- the ``float32`` copy feeds the centre-of-mass integrals, the exact ``float64``
    # widening the diffusion and its volume.
    v = wp.int32(wp.tid())
    normal = gathered_area_weighted_normal(positions, faces, vf_offsets, vf_indices, v)
    moved = step_along_normal(positions[v], normal, distance)
    out_displaced[v] = moved
    out_displaced_wide[v] = to_vec3d(moved)


# ---------------------------------------------------------------------------
# Relaxation family: area equalization, volume-preserving relax, surface-fit relax
# ---------------------------------------------------------------------------


@wp.func
def limit_near_initial(target: wp.vec3, initial: wp.vec3, max_distance: wp.float32) -> wp.vec3:
    # Clamp a proposed position into a ball around where the vertex started. A negative radius means
    # no limit, which is how the wrapper spells ``max_displacement=None`` without a second kernel.
    if max_distance < wp.float32(0.0):
        return target
    offset = target - initial
    distance = wp.length(offset)
    if distance <= max_distance:
        return target
    return initial + offset * (max_distance / distance)


@wp.func
def _rotate_corner_to_front(
    first: wp.int32, second: wp.int32, third: wp.int32, vertex: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # The face's other two corners, in winding order starting after ``vertex``. Winding order is
    # what makes the pair an oriented opposite *edge* rather than an unordered pair.
    if first == vertex:
        return second, third
    if second == vertex:
        return third, first
    return first, second


@wp.func
def equal_area_position_f64(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex: wp.int32,
    normal: wp.vec3,
    no_shrinkage: wp.bool,
) -> wp.vec3:
    """
    Solve for the position minimizing the summed squared areas of the incident triangles.

    Twice the area of the triangle on the opposite edge ``(p, q)`` is ``|(x - p) x (q - p)|``, so
    the objective is a sum of quadratic forms in the free position ``x`` and its minimum is one
    linear solve. Accumulated in ``float64``: the matrix is a sum of rank-deficient terms and a
    near-degenerate 1-ring loses the answer entirely in ``float32``.

    With ``no_shrinkage`` the solve is restricted to the tangent plane through the current position,
    so the vertex slides across the surface instead of sinking into it -- an unconstrained minimum
    of *squared* area pulls the whole 1-ring inward.
    """
    current = positions[vertex]
    matrix = wp.mat33d()
    rhs = wp.vec3d()
    for slot in range(offsets[vertex], offsets[vertex + 1]):
        first, second, third = corner_triple(faces, vertex_faces[slot])
        opposite_start, opposite_end = _rotate_corner_to_front(first, second, third, vertex)
        first_position = to_vec3d(positions[opposite_start])
        edge = to_vec3d(positions[opposite_end]) - first_position
        # ``d d^T - |d|^2 I`` maps x to d x (d x x): the quadratic form whose value at x - p is
        # minus the squared area term.
        term = wp.outer(edge, edge) - wp.identity(n=3, dtype=wp.float64) * wp.dot(edge, edge)
        matrix += term
        rhs += term * first_position

    if no_shrinkage:
        # ``plane_basis`` renormalizes ``normal`` internally to build the tangent frame; the anchor
        # projection below must use that same unit vector, since it is only a projection onto the
        # normal axis when its argument has unit length.
        unit_normal = wp.normalize(normal)
        axis_x, axis_y = plane_basis(normal)
        basis_x = to_vec3d(axis_x)
        basis_y = to_vec3d(axis_y)
        mapped_x = matrix * basis_x
        mapped_y = matrix * basis_y
        off_diagonal = wp.dot(mapped_x, basis_y)
        planar = wp.mat22d(
            wp.dot(mapped_x, basis_x), off_diagonal, off_diagonal, wp.dot(mapped_y, basis_y)
        )
        determinant = wp.determinant(planar)
        trace = planar[0, 0] + planar[1, 1]
        if DOUBLE_EPSILON * wp.abs(trace * trace) >= wp.abs(determinant):
            return current
        anchor = to_vec3d(unit_normal) * wp.dot(to_vec3d(unit_normal), to_vec3d(current))
        reduced = rhs - matrix * anchor
        solution = wp.inverse(planar) * wp.vec2d(wp.dot(reduced, basis_x), wp.dot(reduced, basis_y))
        target = anchor + basis_x * solution[0] + basis_y * solution[1]
        return to_vec3(target)

    determinant = wp.determinant(matrix)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if DOUBLE_EPSILON * wp.abs(trace * trace * trace) >= wp.abs(determinant):
        return current
    target = wp.inverse(matrix) * rhs
    return to_vec3(target)


# ``equal_area_position``'s float32 path is trusted while its system's determinant is at least this
# fraction of the cubed (or, in the tangent plane, squared) trace -- an isotropic 3x3 reads 1/27 --
# and the float64 accumulation is redone below it. Swept at 1e-6 / 1e-4 / 1e-2 on bunny, dragon and
# lucy (one pass, against the float64 accumulation): 1e-6 still let a dragon ring through 3e-6
# off, from 1e-4 the largest difference is the positions' own float32 rounding. Ten passes through
# ``equalize_triangle_areas``: 1.44x bunny, 2.9x dragon, 3.2x lucy (1.8x with ``no_shrinkage``,
# which recomputes normals each pass). The ~500 dragon vertices ``no_shrinkage`` once flung beyond
# 0.01 in ten passes (chaotic, differing run to run and between devices) were boundary vertices,
# whose open fan has no area-preserving motion; the wrapper holds the boundary, and none moves that
# far now.
EQUAL_AREA_F32_CONDITION = wp.constant(wp.float64(1.0e-4))


@wp.func
def equal_area_position(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex: wp.int32,
    normal: wp.vec3,
    no_shrinkage: wp.bool,
) -> wp.vec3:
    """
    ``equal_area_position_f64``'s minimum, from ring sums accumulated in ``float32``.

    The quadratic form depends on the edges alone and the right-hand side is taken relative to the
    free vertex, so the sums hold differences at the ring's own scale rather than products of
    absolute coordinates; the small system is then solved in ``float64`` for the displacement. A
    sum of rank-deficient terms still loses its smallest eigenvalue in ``float32`` when the ring
    is nearly degenerate, so a system whose determinant falls below ``EQUAL_AREA_F32_CONDITION``
    of its trace's power is handed to ``equal_area_position_f64``, the exact ``float64``
    accumulation, which then applies its own degeneracy test. On a well-conditioned ring the two
    agree to the positions' ``float32`` rounding; ``float64`` arithmetic runs at a small fraction
    of ``float32``'s on this hardware, and the ring walk is the whole kernel.

    The row is summed in ascending face order (``next_row_entry``), not slot order: the vertex-face
    rows are sets in arbitrary order, and a ``float32`` sum in that order would move the result's
    last bit from run to run. Finding the next entry rereads the short row and measures free.
    Accumulating each ``float32`` term in ``float64`` instead (order-free in practice) cost 3x.
    """
    current = positions[vertex]
    matrix = wp.mat33()
    rhs = wp.vec3()
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    face = wp.int32(-1)
    face_slot = wp.int32(-1)
    for _step in range(begin, end):
        face, face_slot = next_row_entry(vertex_faces, begin, end, face, face_slot)
        first, second, third = corner_triple(faces, face)
        opposite_start, opposite_end = _rotate_corner_to_front(first, second, third, vertex)
        relative_start = positions[opposite_start] - current
        edge = positions[opposite_end] - positions[opposite_start]
        term = wp.outer(edge, edge) - wp.identity(n=3, dtype=wp.float32) * wp.dot(edge, edge)
        matrix += term
        rhs += term * relative_start
    wide = wp.mat33d(matrix)
    wide_rhs = to_vec3d(rhs)
    if no_shrinkage:
        # The tangent plane through the current position: the displacement has no normal part.
        axis_x, axis_y = plane_basis(normal)
        basis_x = to_vec3d(axis_x)
        basis_y = to_vec3d(axis_y)
        mapped_x = wide * basis_x
        mapped_y = wide * basis_y
        off_diagonal = wp.dot(mapped_x, basis_y)
        planar = wp.mat22d(
            wp.dot(mapped_x, basis_x), off_diagonal, off_diagonal, wp.dot(mapped_y, basis_y)
        )
        trace = planar[0, 0] + planar[1, 1]
        if wp.abs(wp.determinant(planar)) >= EQUAL_AREA_F32_CONDITION * trace * trace:
            solution = wp.inverse(planar) * wp.vec2d(
                wp.dot(wide_rhs, basis_x), wp.dot(wide_rhs, basis_y)
            )
            return to_vec3(to_vec3d(current) + basis_x * solution[0] + basis_y * solution[1])
    else:
        trace = wide[0, 0] + wide[1, 1] + wide[2, 2]
        if wp.abs(wp.determinant(wide)) >= EQUAL_AREA_F32_CONDITION * wp.abs(trace * trace * trace):
            return to_vec3(to_vec3d(current) + wp.inverse(wide) * wide_rhs)
    return equal_area_position_f64(
        positions, faces, offsets, vertex_faces, vertex, normal, no_shrinkage
    )


@wp.func
def in_region(region: wp.array[wp.bool], vertex: wp.int32) -> wp.bool:
    # The relaxation family's region test. An empty ``region`` (the wrappers pass ``None`` when the
    # caller gave no mask) means every vertex, so the default path allocates no all-``True`` mask.
    return region.shape[0] == 0 or region[vertex]


@wp.func
def equalized_vertex(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    no_shrinkage: wp.bool,
    max_displacement: wp.float32,
    vertex: wp.int32,
) -> wp.vec3:
    # Vertex ``vertex``'s entry of an ``equalize_area_step``.
    current = positions[vertex]
    if not in_region(region, vertex) or offsets[vertex] == offsets[vertex + 1]:
        return current
    # The tangent plane is the one through the vertex *now*, so its normal is gathered from the
    # current positions (only read under ``no_shrinkage``).
    normal = wp.vec3()
    if no_shrinkage:
        normal = gathered_area_weighted_normal(positions, faces, offsets, vertex_faces, vertex)
    target = equal_area_position(
        positions, faces, offsets, vertex_faces, vertex, normal, no_shrinkage
    )
    moved = current + (target - current) * force
    return limit_near_initial(moved, initial[vertex], max_displacement)


@wp.kernel
def equalize_area_step(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    no_shrinkage: wp.bool,
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # One pass of area equalization: step each in-region vertex a fraction ``force`` of the way to
    # its own equal-area minimum, then clamp it back near where it started. The vertex is
    # ``equalized_vertex``, shared with ``equalize_area_steps_serial``.
    vertex = wp.int32(wp.tid())
    out_positions[vertex] = equalized_vertex(
        positions, faces, offsets, vertex_faces, region, initial, force, no_shrinkage,
        max_displacement, vertex,
    )  # fmt: skip


@wp.kernel
def equalize_area_steps_serial(
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    no_shrinkage: wp.bool,
    max_displacement: wp.float32,
    passes: wp.int32,
    out_positions_0: wp.array[wp.vec3],
    out_positions_1: wp.array[wp.vec3],
) -> None:
    # Every ``equalize_area_step`` of a call in one launch of one thread, the CPU device's form:
    # the rotation of ``smoothing._float32_passes`` (pass 0 reads ``initial``, pass ``k`` writes
    # ``out_positions_{k % 2}`` and the next reads it), each vertex ``equalized_vertex``, so the
    # result is the launches' to the bit. ``out_positions_1`` may alias ``out_positions_0`` at one
    # pass, where it is unused.
    n = initial.shape[0]
    for index in range(passes):
        for vertex in range(n):
            value = wp.vec3()
            if index == 0:
                value = equalized_vertex(
                    initial, faces, offsets, vertex_faces, region, initial, force, no_shrinkage,
                    max_displacement, vertex,
                )  # fmt: skip
            elif index % 2 == 1:
                value = equalized_vertex(
                    out_positions_0, faces, offsets, vertex_faces, region, initial, force,
                    no_shrinkage, max_displacement, vertex,
                )  # fmt: skip
            else:
                value = equalized_vertex(
                    out_positions_1, faces, offsets, vertex_faces, region, initial, force,
                    no_shrinkage, max_displacement, vertex,
                )  # fmt: skip
            if index % 2 == 0:
                out_positions_0[vertex] = value
            else:
                out_positions_1[vertex] = value


@wp.func
def ring_push(
    positions: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    force: wp.float32,
    vertex: wp.int32,
) -> wp.vec3:
    # Vertex ``vertex``'s entry of ``ring_push_forces``.
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    if not in_region(region, vertex) or begin == end:
        return wp.vec3(0.0, 0.0, 0.0)
    total = wp.vec3d()
    for slot in range(begin, end):
        total += to_vec3d(positions[columns[slot]])
    mean = total / wp.float64(end - begin)
    average = to_vec3(mean)
    return (average - positions[vertex]) * force


@wp.kernel
def ring_push_forces(
    positions: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    force: wp.float32,
    out_push: wp.array[wp.vec3],
) -> None:
    # The plain uniform-relax displacement each in-region vertex would take on its own. Kept as a
    # field rather than applied, because the volume correction below is its ring average.
    vertex = wp.int32(wp.tid())
    out_push[vertex] = ring_push(positions, offsets, columns, region, force, vertex)


@wp.func
def pushed_keeping_volume(
    positions: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    push: wp.array[wp.vec3],
    initial: wp.array[wp.vec3],
    max_displacement: wp.float32,
    vertex: wp.int32,
) -> wp.vec3:
    # Vertex ``vertex``'s entry of ``apply_push_keeping_volume``.
    current = positions[vertex]
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    if not in_region(region, vertex) or begin == end:
        return current
    total = wp.vec3()
    for slot in range(begin, end):
        neighbor = columns[slot]
        if in_region(region, neighbor):
            total += push[neighbor]
    moved = current + push[vertex] - total / wp.float32(end - begin)
    return limit_near_initial(moved, initial[vertex], max_displacement)


@wp.kernel
def apply_push_keeping_volume(
    positions: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    push: wp.array[wp.vec3],
    initial: wp.array[wp.vec3],
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Subtract the ring average of the displacement field from each vertex's own displacement. A
    # translation shared by a whole neighbourhood cancels, so the surface stops drifting inward
    # while the high-frequency part of the relax survives -- which is what preserves the volume.
    # The divisor is the full degree while the sum runs over in-region neighbours only, so a vertex
    # on the region's edge is corrected by less than one that is surrounded.
    vertex = wp.int32(wp.tid())
    out_positions[vertex] = pushed_keeping_volume(
        positions, offsets, columns, region, push, initial, max_displacement, vertex
    )


@wp.kernel
def relax_keeping_volume_serial(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    max_displacement: wp.float32,
    passes: wp.int32,
    out_push: wp.array[wp.vec3],
    out_positions_0: wp.array[wp.vec3],
    out_positions_1: wp.array[wp.vec3],
) -> None:
    # Every ``relax_keep_volume`` pass in one launch of one thread, the CPU device's form: each
    # pass's ``ring_push`` rows, then its ``pushed_keeping_volume`` rows, in the rotation of
    # ``equalize_area_steps_serial``, so the result is the launches' to the bit.
    n = initial.shape[0]
    for index in range(passes):
        for vertex in range(n):
            if index == 0:
                out_push[vertex] = ring_push(initial, offsets, columns, region, force, vertex)
            elif index % 2 == 1:
                out_push[vertex] = ring_push(
                    out_positions_0, offsets, columns, region, force, vertex
                )
            else:
                out_push[vertex] = ring_push(
                    out_positions_1, offsets, columns, region, force, vertex
                )
        for vertex in range(n):
            value = wp.vec3()
            if index == 0:
                value = pushed_keeping_volume(
                    initial, offsets, columns, region, out_push, initial, max_displacement, vertex
                )
            elif index % 2 == 1:
                value = pushed_keeping_volume(
                    out_positions_0, offsets, columns, region, out_push, initial,
                    max_displacement, vertex,
                )  # fmt: skip
            else:
                value = pushed_keeping_volume(
                    out_positions_1, offsets, columns, region, out_push, initial,
                    max_displacement, vertex,
                )  # fmt: skip
            if index % 2 == 0:
                out_positions_0[vertex] = value
            else:
                out_positions_1[vertex] = value


@wp.func
def _neighborhood_frame(
    positions: wp.array[wp.vec3], neighbors: wp.array[wp.int32], begin: wp.int32, end: wp.int32
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.float32]:
    # Principal frame of the neighbourhood point set: its centroid, then the two directions of
    # greatest spread and the one of least. The least-spread direction is the fitted plane's normal,
    # so the same decomposition serves both the planar and the quadric fit.
    #
    # The radius comes back with it because the covariance pass already holds every offset whose
    # length it is, and the quadric fit divides its local coordinates by it -- see the comment at
    # that fit for why a scale-free fit is a correctness requirement and not a nicety.
    count = wp.float32(end - begin)
    centroid = wp.vec3()
    for slot in range(begin, end):
        centroid += positions[neighbors[slot]]
    centroid /= count
    covariance = wp.mat33()
    radius = wp.float32(0.0)
    for slot in range(begin, end):
        offset = positions[neighbors[slot]] - centroid
        covariance += wp.outer(offset, offset)
        radius = wp.max(radius, wp.length(offset))
    _left, _singular, basis = wp.svd3(covariance / count)
    # ``wp.svd3`` orders the singular values descending, so the last column spans the least. Its
    # sign is arbitrary and irrelevant: every use below is a projection along the axis, not a side.
    axis_u = wp.vec3(basis[0, 0], basis[1, 0], basis[2, 0])
    axis_v = wp.vec3(basis[0, 1], basis[1, 1], basis[2, 1])
    axis_w = wp.vec3(basis[0, 2], basis[1, 2], basis[2, 2])
    return centroid, axis_u, axis_v, axis_w, radius


@wp.func
def _quadric_design_row(u: wp.float64, v: wp.float64) -> vec6d:
    # The least-squares row of ``w = a u^2 + b u v + c v^2 + d u + e v + f`` at ``(u, v)``.
    return vec6d(u * u, u * v, v * v, u, v, wp.float64(1.0))


@wp.func
def _quadric_value(coefficients: vec6d, u: wp.float64, v: wp.float64) -> wp.float64:
    # The fitted quadric's height at ``(u, v)``, coefficients in ``_quadric_design_row``'s order.
    return (
        coefficients[0] * u * u
        + coefficients[1] * u * v
        + coefficients[2] * v * v
        + coefficients[3] * u
        + coefficients[4] * v
        + coefficients[5]
    )


@wp.func
def _neighborhood_target(
    positions: wp.array[wp.vec3],
    neighbors: wp.array[wp.int32],
    begin: wp.int32,
    end: wp.int32,
    current: wp.vec3,
    quadric: wp.bool,
) -> wp.vec3:
    # The point above ``current`` of the plane or quadric fitted to the neighbourhood, its frame
    # formed in float32 and the quadric solved in float64.
    centroid, axis_u, axis_v, axis_w, radius = _neighborhood_frame(positions, neighbors, begin, end)
    offset = current - centroid
    # Initialized before the branch, per the kernel-scope scoping rule; the planar fit's answer is
    # exactly this, since the plane passes through the neighbourhood centroid.
    height = wp.float32(0.0)
    # A zero radius means every neighbour sits on the centroid: no quadric to fit, and the plane's
    # answer -- which ``height = 0`` already is -- is the whole of what the neighbourhood says.
    if quadric and radius > wp.float32(0.0):
        # Least squares over ``w = a u^2 + b u v + c v^2 + d u + e v + f`` in the neighbourhood's
        # own frame, **in units of the neighbourhood radius**: the fit is a graph over the plane the
        # neighbourhood already lies closest to.
        #
        # The division by the radius is what makes the fit scale-free, and it is a correctness
        # requirement. The design row spans ``[u^2, u v, v^2, u, v, 1]``, so at mesh scale ``h`` the
        # normal matrix's diagonal spans ``h^8`` down to ``1`` and ``solve_normal_equations``'
        # singularity test -- absolute, and necessarily so, since it cannot see the caller's units
        # -- starts reporting well-conditioned neighbourhoods as singular. That failure is silent:
        # the vertex falls back to the planar answer with nothing said, and on a small enough mesh
        # nearly every vertex does.
        inv_radius = wp.float64(1.0) / wp.float64(radius)
        normal_matrix = mat66d()
        normal_rhs = vec6d()
        for slot in range(begin, end):
            local = positions[neighbors[slot]] - centroid
            u = wp.float64(wp.dot(local, axis_u)) * inv_radius
            v = wp.float64(wp.dot(local, axis_v)) * inv_radius
            row = _quadric_design_row(u, v)
            normal_matrix += wp.outer(row, row)
            normal_rhs += row * (wp.float64(wp.dot(local, axis_w)) * inv_radius)
        coefficients, ok = solve_normal_equations(normal_matrix, normal_rhs)
        if ok:
            # Evaluated in the same units the fit was solved in, then carried back: ``w`` was
            # divided by the radius alongside ``u`` and ``v``, so the height is multiplied by it.
            u = wp.float64(wp.dot(offset, axis_u)) * inv_radius
            v = wp.float64(wp.dot(offset, axis_v)) * inv_radius
            height = wp.float32(wp.float64(radius) * _quadric_value(coefficients, u, v))

    return current + axis_w * (height - wp.dot(offset, axis_w))


@wp.func
def _neighborhood_target_d(
    positions: wp.array[wp.vec3],
    neighbors: wp.array[wp.int32],
    begin: wp.int32,
    end: wp.int32,
    current: wp.vec3,
    quadric: wp.bool,
) -> wp.vec3:
    # ``_neighborhood_target`` with every step in float64: the centroid, the covariance and its
    # ``wp.svd3``, the quadric's local coordinates. The float32 ``svd3`` of a well-conditioned
    # covariance can tilt the normal by ~4e-4, which a vertex ~0.03 off the centroid turns into a
    # 1e-5 move; in float64 the normal is the covariance's to ~1e-8. Same fit, same frame, same
    # scale-free quadric coordinates -- the two differ by precision alone.
    count = wp.float64(end - begin)
    centroid = wp.vec3d()
    for slot in range(begin, end):
        centroid += to_vec3d(positions[neighbors[slot]])
    centroid /= count
    covariance = wp.mat33d()
    radius = wp.float64(0.0)
    for slot in range(begin, end):
        offset = to_vec3d(positions[neighbors[slot]]) - centroid
        covariance += wp.outer(offset, offset)
        radius = wp.max(radius, wp.length(offset))
    _left, _singular, basis = wp.svd3(covariance / count)
    axis_u = wp.vec3d(basis[0, 0], basis[1, 0], basis[2, 0])
    axis_v = wp.vec3d(basis[0, 1], basis[1, 1], basis[2, 1])
    axis_w = wp.vec3d(basis[0, 2], basis[1, 2], basis[2, 2])
    offset = to_vec3d(current) - centroid
    height = wp.float64(0.0)
    if quadric and radius > wp.float64(0.0):
        inv_radius = wp.float64(1.0) / radius
        normal_matrix = mat66d()
        normal_rhs = vec6d()
        for slot in range(begin, end):
            local = (to_vec3d(positions[neighbors[slot]]) - centroid) * inv_radius
            row = _quadric_design_row(wp.dot(local, axis_u), wp.dot(local, axis_v))
            normal_matrix += wp.outer(row, row)
            normal_rhs += row * wp.dot(local, axis_w)
        coefficients, ok = solve_normal_equations(normal_matrix, normal_rhs)
        if ok:
            u = wp.dot(offset, axis_u) * inv_radius
            v = wp.dot(offset, axis_v) * inv_radius
            height = radius * _quadric_value(coefficients, u, v)
    return to_vec3(to_vec3d(current) + axis_w * (height - wp.dot(offset, axis_w)))


@wp.kernel
def relax_approx_step(
    positions: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    row_stride: wp.int32,
    chunk_start: wp.int32,
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    quadric: wp.bool,
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Fit a local surface to the vertex's neighbourhood and step toward the point of that surface
    # above the vertex. The floor below is a uniform 6 for both the planar and the quadric fit --
    # not 3, even though a plane alone needs only that many -- so an under-populated neighbourhood
    # is left alone rather than fitted to whatever it has; ``smoothing.relax_approx``'s docstring
    # states this uniform floor as the intended behavior.
    #
    # The neighbourhoods arrive in either of ``ball_row_bounds``' layouts: the packed list over
    # every vertex at ``chunk_start = 0``, or one chunk of the walk's queue rows, thread ``t``
    # fitting vertex ``chunk_start + t``.
    t = wp.int32(wp.tid())
    vertex = chunk_start + t
    current = positions[vertex]
    begin, end = ball_row_bounds(neighbor_offsets, row_stride, vertex, t)
    if not in_region(region, vertex) or end - begin < 6:
        out_positions[vertex] = current
        return

    target = _neighborhood_target(positions, neighbor_indices, begin, end, current, quadric)
    moved = current + (target - current) * force
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


@wp.kernel
def relax_approx_step_d(
    positions: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    row_stride: wp.int32,
    chunk_start: wp.int32,
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    quadric: wp.bool,
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # ``relax_approx_step`` with the fit formed in float64 (``_neighborhood_target_d``); the guard
    # and the step are the same. Two kernels rather than a precision selector in one: a selector
    # moved the float32 kernel's CUDA rounding at the displacement clamp by an ulp.
    t = wp.int32(wp.tid())
    vertex = chunk_start + t
    current = positions[vertex]
    begin, end = ball_row_bounds(neighbor_offsets, row_stride, vertex, t)
    if not in_region(region, vertex) or end - begin < 6:
        out_positions[vertex] = current
        return

    target = _neighborhood_target_d(positions, neighbor_indices, begin, end, current, quadric)
    moved = current + (target - current) * force
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


@wp.func
def band_field_value(
    field: wp.array[wp.float64],
    free: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array[wp.float64],
    j: wp.int32,
) -> wp.float64:
    # The rim band's harmonic field at vertex ``j``: the reduced solve's answer on a free vertex,
    # the boundary value it was pinned to otherwise -- read in place, so the full-length field is
    # never assembled. A branch rather than ``wp.where``, whose eager arms would index ``solution``
    # at a pinned vertex's meaningless rank.
    if free[j]:
        return solution[free_map[j]]
    return field[j]


@wp.kernel
def project_to_zero_isoline(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    field: wp.array[wp.float64],
    free: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array[wp.float64],
    damping: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Pull each free vertex onto the field's zero level set, which is where the region's rim curve
    # wants to be; the field is ``band_field_value``'s, the pinned values in ``field`` and the free
    # ones in ``solution`` at their ``free_map`` ranks. Inside one triangle the level set is a
    # straight segment between the crossings on the two edges out of the *apex* -- the corner whose
    # sign differs from the other two -- so the nearest point of the whole curve to this vertex is
    # the nearest over its incident triangles' segments. Damped rather than snapped, because the
    # field is recomputed from the moved positions on the next pass and a full step oscillates.
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    if not free[vertex]:
        out_positions[vertex] = current
        return

    best = current
    best_distance = wp.float32(3.4028235e38)
    for slot in range(offsets[vertex], offsets[vertex + 1]):
        first, second, third = corner_triple(faces, vertex_faces[slot])
        value_first = band_field_value(field, free, free_map, solution, first)
        value_second = band_field_value(field, free, free_map, solution, second)
        value_third = band_field_value(field, free, free_map, solution, third)
        # The apex is the corner alone on its side of zero. When every corner shares a sign the
        # level set misses the triangle entirely.
        apex = first
        left = second
        right = third
        value_apex = value_first
        value_left = value_second
        value_right = value_third
        if value_second * value_third > wp.float64(0.0):
            if value_first * value_second > wp.float64(0.0):
                continue
        elif value_first * value_third > wp.float64(0.0):
            apex = second
            left = third
            right = first
            value_apex = value_second
            value_left = value_third
            value_right = value_first
        else:
            apex = third
            left = first
            right = second
            value_apex = value_third
            value_left = value_first
            value_right = value_second

        gap_left = value_apex - value_left
        gap_right = value_apex - value_right
        apex_position = positions[apex]
        # A gap of exactly zero means ``apex`` and that neighbour already share the same (zero)
        # field value, so by linearity the whole edge between them -- not one interior point on
        # it -- lies on the level set; falling through to the crossing formula below would divide
        # by that zero. Using the edge itself as the candidate segment covers the doubly-degenerate
        # case too (every corner on the level set) -- some incident edge is still a valid witness.
        if gap_left == wp.float64(0.0):
            candidate = closest_point_on_segment(apex_position, positions[left], current)
        elif gap_right == wp.float64(0.0):
            candidate = closest_point_on_segment(apex_position, positions[right], current)
        else:
            crossing_left = apex_position + (positions[left] - apex_position) * wp.float32(
                value_apex / gap_left
            )
            crossing_right = apex_position + (positions[right] - apex_position) * wp.float32(
                value_apex / gap_right
            )
            candidate = closest_point_on_segment(crossing_left, crossing_right, current)
        distance = wp.length_sq(candidate - current)
        if distance < best_distance:
            best_distance = distance
            best = candidate

    out_positions[vertex] = current + (best - current) * damping


@wp.func
def row_slot(
    columns: wp.array[wp.int32], start: wp.int32, end: wp.int32, column: wp.int32
) -> wp.int32:
    # The slot of ``column`` in the sorted CSR row ``[start, end)``, or ``-1``. Linear: a mesh
    # Laplacian row is the one-ring plus the diagonal, a handful of entries.
    for e in range(start, end):
        if columns[e] == column:
            return e
    return wp.int32(-1)


@wp.func
def band_corner_seen(
    faces: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    ring_start: wp.int32,
    k: wp.int32,
    corner: wp.int32,
    j: wp.int32,
) -> wp.bool:
    # Whether vertex ``j``, corner ``corner`` of ring face ``vf_indices[k]``, already appeared as a
    # corner of an earlier ring face or of an earlier corner of this one: the first sighting of
    # each neighbour is the one that counts it.
    for q in range(ring_start, k):
        a, b, c = corner_triple(faces, vf_indices[q])
        if a == j or b == j or c == j:
            return True
    f = vf_indices[k]
    for earlier in range(corner):
        if faces[f * 3 + earlier] == j:
            return True
    return False


@wp.kernel
def band_pattern(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    fill: wp.int32,
    pattern_offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
) -> None:
    # The sparsity of the rim band's Dirichlet system, read off the vertex-face rings: free row
    # ``ri`` holds its diagonal and one column per distinct free vertex sharing a face with it --
    # exactly the free-free block ``linalg.assemble_interior_system`` extracts from a mesh-wide
    # cotangent Laplacian, whose triplets pair every two corners of a face, with the columns sorted
    # as that CSR stores them. Launched twice, with a warp-uniform ``fill``: ``0`` writes each
    # row's length into ``out_counts`` (the tail of the ``n_free + 1`` offsets, ``out_columns``
    # unused), ``1`` inserts the columns in order into the rows ``pattern_offsets`` sized
    # (``out_counts`` unused). Band-sized work in place of a whole-mesh matrix build.
    v = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, v)
    if ri < 0:
        return
    start = wp.int32(0)
    if fill != wp.int32(0):
        start = pattern_offsets[ri]
        out_columns[start] = ri
    width = wp.int32(1)
    ring_start = vf_offsets[v]
    for k in range(ring_start, vf_offsets[v + 1]):
        f = vf_indices[k]
        for corner in range(3):
            j = faces[f * 3 + corner]
            if j != v and not fixed_mask[j]:
                if not band_corner_seen(faces, vf_indices, ring_start, k, corner, j):
                    if fill != wp.int32(0):
                        column = free_map[j]
                        slot = start + width
                        while slot > start and out_columns[slot - 1] > column:
                            out_columns[slot] = out_columns[slot - 1]
                            slot -= 1
                        out_columns[slot] = column
                    width += 1
    if fill == wp.int32(0):
        out_counts[ri] = width


@wp.kernel
def band_dirichlet_values(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    field: wp.array2d[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # One thread per vertex, free ones only: row ``ri`` of the Dirichlet system
    # ``(-L)_uu x = (-L)_ub field`` that ``linalg.assemble_interior_system`` would extract from a
    # mesh-wide ``-cotmatrix``, written into ``band_pattern``'s sparsity (the same one) from the
    # current half-cotangents. The connectivity -- and so the pattern and the free set -- is fixed
    # across ``smooth_region_boundary``'s passes while the weights move, so only the band's rows are
    # computed, every pass, and the whole mesh's matrix never is. The weight of the edge
    # opposite corner ``e`` is ``laplacian.face_half_cotangents``' column ``e`` for the current
    # ``positions`` (``kernels/laplacian.mesh_operator_keys``' corner convention), formed here for
    # the band's own faces rather than read from a whole-mesh table, and cast to ``float64`` as
    # ``cotmatrix`` casts it; the off-diagonal is ``-w``, the diagonal ``sum w``, and a pinned
    # neighbour moves ``w * field_j`` to the right-hand side.
    v = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, v)
    if ri < 0:
        return
    start = offsets[ri]
    end = offsets[ri + 1]
    for e in range(start, end):
        out_values[e] = wp.float64(0.0)
    diagonal = wp.float64(0.0)
    rhs = wp.float64(0.0)
    for k in range(vf_offsets[v], vf_offsets[v + 1]):
        f = vf_indices[k]
        c0, c1, c2 = face_half_cotangents(positions, faces, f)
        cot = wp.vec3(c0, c1, c2)
        for e in range(3):
            a = faces[f * 3 + (e + 1) % 3]
            b = faces[f * 3 + (e + 2) % 3]
            if a == v or b == v:
                j = wp.where(a == v, b, a)
                w = wp.float64(cot[e])
                diagonal += w
                if fixed_mask[j]:
                    rhs += w * field[0, j]
                else:
                    slot = row_slot(columns, start, end, free_map[j])
                    if slot >= 0:
                        out_values[slot] -= w
    slot = row_slot(columns, start, end, ri)
    if slot >= 0:
        out_values[slot] += diagonal
    out_rhs[0, ri] = rhs


@wp.func
def band_pins(free: wp.bool, inside: wp.bool) -> tuple[wp.bool, wp.float64]:
    # A vertex's two inputs to the rim band's harmonic solve: whether it is pinned (every vertex
    # off the band), and the field the rim curve is the zero set of -- -1 on the region, +1 outside
    # it. Any two values of opposite sign would do; +-1 keeps the harmonic interpolant's scale
    # comparable to nothing else, which is fine because only its zero set is read. One map writes
    # both.
    side = wp.float64(1.0)
    if inside:
        side = wp.float64(-1.0)
    return not free, side


@wp.kernel
def mark_band_vertices(
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    inside: wp.array[wp.bool],
    out_band: wp.array[wp.bool],
) -> None:
    # The vertices touching both a selected and an unselected face: every corner of an unselected
    # face that some selected face also touches (``inside``). Concurrent writes all store ``True``,
    # so the race is benign; ``out_band`` arrives zeroed.
    f = wp.int32(wp.tid())
    if region[f]:
        return
    a, b, c = corner_triple(faces, f)
    if inside[a]:
        out_band[a] = True
    if inside[b]:
        out_band[b] = True
    if inside[c]:
        out_band[c] = True


# ---------------------------------------------------------------------------
# Windowed-sinc smoothing (Taubin, Zhang and Golub, "Optimal Surface Smoothing as Filter Design")
# ---------------------------------------------------------------------------

# A vertex's stencil class: it does not move under the Laplacian, averages its whole 1-ring, or
# averages its two neighbours along the rim.
STENCIL_FIXED = wp.int32(0)
STENCIL_RING = wp.int32(1)
STENCIL_RIM = wp.int32(2)
# A vertex with this many incident face corners (``2 * 255`` incident directed edges) is held
# fixed: VTK's bound on a stencil's size.
STENCIL_MAX_CORNERS = wp.int32(255)


@wp.kernel
def tally_stencil_edges(
    edges: wp.array2d[wp.int32],
    uses: wp.array[wp.int32],
    crowded_slides: wp.int32,
    out_uses: wp.array[wp.int32],
    out_rim_counts: wp.array[wp.int32],
    out_crowded_counts: wp.array[wp.int32],
    out_rim_pairs: wp.array[wp.vec2i],
) -> None:
    # Per unique edge, into both endpoints: the face corners it carries (``out_uses``, twice a
    # vertex's incident faces), whether it is a rim edge (one face; the first two land in
    # ``out_rim_pairs``, and a third makes the vertex fixed whatever they hold) and whether it is
    # crowded (three or more faces). With ``crowded_slides`` a crowded edge also takes a slot in the
    # pair and in ``out_rim_counts``, which then counts both kinds (``classify_stencil_vertices``).
    # Integer atomics; every output arrives zeroed.
    e = wp.int32(wp.tid())
    m = uses[e]
    for side in range(2):
        v = edges[e, side]
        other = edges[e, 1 - side]
        wp.atomic_add(out_uses, v, m)
        if m > 2:
            wp.atomic_add(out_crowded_counts, v, 1)
        if m == 1 or (m > 2 and crowded_slides != 0):
            slot = wp.atomic_add(out_rim_counts, v, 1)
            if slot < 2:
                out_rim_pairs[v][slot] = other


@wp.func
def rim_pair_stencil(
    vertices: wp.array[wp.vec3], v: wp.int32, pair: wp.vec2i, cos_edge_angle: wp.float64
) -> wp.int32:
    # ``STENCIL_RIM`` where the path through the pair turns at ``v`` by at most the edge angle,
    # else ``STENCIL_FIXED``; symmetric in the pair's order. A zero-length edge normalizes to zero
    # and is fixed, as VTK's ``vtkMath::Normalize`` leaves it.
    p = wp.vec3d(vertices[v])
    incoming = wp.normalize(p - wp.vec3d(vertices[pair[0]]))
    outgoing = wp.normalize(wp.vec3d(vertices[pair[1]]) - p)
    return wp.where(wp.dot(incoming, outgoing) >= cos_edge_angle, STENCIL_RIM, STENCIL_FIXED)


@wp.kernel
def classify_stencil_vertices(
    vertices: wp.array[wp.vec3],
    uses: wp.array[wp.int32],
    rim_counts: wp.array[wp.int32],
    crowded_counts: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    rim_smoothing: wp.int32,
    crowded_slides: wp.int32,
    cos_edge_angle: wp.float64,
    out_classes: wp.array[wp.int32],
) -> None:
    # VTK's stencil analyses without feature edges. They differ by ``crowded_slides`` alone.
    # Off, ``vtkWindowedSincPolyDataFilter``'s (``BuildO1Stencil`` with rim smoothing,
    # ``BuildO2Stencil`` without): an unreferenced vertex, a hub of ``STENCIL_MAX_CORNERS`` faces
    # or more, and any vertex on a crowded edge are fixed; a vertex with no rim edge averages its
    # ring; one with exactly two averages them, unless the rim turns there by more than the edge
    # angle; any other rim count is fixed. On, ``vtkSmoothPolyDataFilter``'s: a crowded edge is a
    # feature edge, counted with the rim edges (``rim_counts`` holds both), so a vertex on two of
    # them slides along them under the same turn test; rim smoothing off fixes only a vertex with a
    # rim edge among them, and there is no hub bound.
    v = wp.int32(wp.tid())
    stencil = STENCIL_FIXED
    special = rim_counts[v]
    crowded = crowded_counts[v]
    if crowded_slides != 0:
        rims = special - crowded
        fits = uses[v] > 0
    else:
        rims = special
        fits = uses[v] > 0 and uses[v] < 2 * STENCIL_MAX_CORNERS and crowded == 0
    if fits:
        if special == 0:
            stencil = STENCIL_RING
        elif special == 2 and (rim_smoothing != 0 or rims == 0):
            stencil = rim_pair_stencil(vertices, v, rim_pairs[v], cos_edge_angle)
    out_classes[v] = stencil


@wp.func
def stencil_laplacian(
    x: wp.array[wp.vec3d],
    v: wp.int32,
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
) -> wp.vec3d:
    # The negative umbrella Laplacian ``x_v - mean(x_stencil)``, each term divided by the stencil
    # size before the sum as VTK forms it; zero at a fixed vertex. A ring stencil reads the
    # vertex's unique neighbours, a rim stencil its two rim neighbours (a two-term sum, so the order
    # the pair was written in does not round differently).
    delta = wp.vec3d()
    stencil = classes[v]
    if stencil == STENCIL_RING:
        start = offsets[v]
        count = offsets[v + 1] - start
        scale = wp.float64(1.0) / wp.float64(count)
        for k in range(count):
            delta += (x[v] - x[neighbors[start + k]]) * scale
    elif stencil == STENCIL_RIM:
        pair = rim_pairs[v]
        delta = (x[v] - x[pair[0]]) * wp.float64(0.5) + (x[v] - x[pair[1]]) * wp.float64(0.5)
    return delta


@wp.kernel
def windowed_sinc_first_pass(
    x0: wp.array[wp.vec3d],
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    c0: wp.float64,
    c1: wp.float64,
    out_x1: wp.array[wp.vec3d],
    out_sum: wp.array[wp.vec3d],
) -> None:
    # The Chebyshev recursion's second term ``x1 = x0 - L x0 / 2`` and the filtered sum
    # ``c0 x0 + c1 x1``; ``windowed_sinc_pass`` is every later term, and the two differ by that
    # recurrence and by setting the sum rather than adding to it.
    v = wp.int32(wp.tid())
    x1 = x0[v] - stencil_laplacian(x0, v, neighbors, offsets, classes, rim_pairs) * wp.float64(0.5)
    out_x1[v] = x1
    out_sum[v] = x0[v] * c0 + x1 * c1


@wp.kernel
def windowed_sinc_pass(
    x0: wp.array[wp.vec3d],
    x1: wp.array[wp.vec3d],
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    c: wp.float64,
    out_x2: wp.array[wp.vec3d],
    out_sum: wp.array[wp.vec3d],
) -> None:
    # One later term of the recursion, ``x2 = 2 x1 - x0 - L x1``, added into the filtered sum with
    # its coefficient.
    v = wp.int32(wp.tid())
    delta = stencil_laplacian(x1, v, neighbors, offsets, classes, rim_pairs)
    x2 = x1[v] - x0[v] + x1[v] - delta
    out_x2[v] = x2
    out_sum[v] = out_sum[v] + x2 * c


# ---------------------------------------------------------------------------
# In-place (Gauss-Seidel) umbrella smoothing: VTK's ``vtkSmoothPolyDataFilter``
# ---------------------------------------------------------------------------
#
# Each sweep moves the vertices in index order, each by ``factor`` toward the mean of its stencil
# as it stands, so a vertex reads its lower-numbered neighbours already moved this sweep and its
# higher-numbered ones as the last sweep left them. ``gauss_seidel_sweeps`` runs that order on one
# thread (the CPU device); ``gauss_seidel_wave`` runs it as a dataflow wavefront, one launch a
# wave, and computes the same values bit for bit, since every update reads the same versions of the
# same positions. The two kernels differ only by the schedule; the update is ``relax_in_place``.


@wp.func
def relax_in_place(
    x: wp.array[wp.vec3d],
    v: wp.int32,
    factor: wp.float64,
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
) -> None:
    # ``x_v += factor (mean(x_stencil) - x_v)``, read from and written to the live buffer.
    x[v] = x[v] - stencil_laplacian(x, v, neighbors, offsets, classes, rim_pairs) * factor


@wp.kernel
def gauss_seidel_sweeps(
    x: wp.array[wp.vec3d],
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    factor: wp.float64,
    sweeps: wp.int32,
) -> None:
    # Every sweep on one thread (``dim=1``), in index order.
    n = x.shape[0]
    for _sweep in range(sweeps):
        for v in range(n):
            if classes[v] != STENCIL_FIXED:
                relax_in_place(x, v, factor, neighbors, offsets, classes, rim_pairs)


@wp.kernel
def gauss_seidel_wave_start(
    classes: wp.array[wp.int32],
    sweeps: wp.int32,
    out_counts: wp.array[wp.int32],
    out_done: wp.array[wp.int32],
) -> None:
    # Each vertex's completed sweep count before the first wave, and no wave run in the trailing
    # slot: a fixed vertex never moves, so it starts done and is counted into ``out_done`` (arrives
    # zeroed), which reaches ``n`` when the last vertex finishes its last sweep.
    v = wp.int32(wp.tid())
    if classes[v] == STENCIL_FIXED:
        out_counts[v] = sweeps
        append_slot(out_done, 0)
    else:
        out_counts[v] = 0
    if v == 0:
        out_counts[classes.shape[0]] = 0


@wp.func
def gauss_seidel_wave_vertex(
    x: wp.array[wp.vec3d],
    counts: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    factor: wp.float64,
    sweeps: wp.int32,
    done: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    v: wp.int32,
) -> wp.int32:
    # One wave at vertex ``v``. A vertex that has completed ``c`` sweeps runs sweep ``c + 1`` once
    # every moving 1-ring neighbour numbered below it has completed ``c + 1`` and every one above
    # it exactly ``c``: the versions the serial order reads, and the only ones a neighbour can hold
    # without waiting on this vertex. Two neighbours are never both ready (each condition
    # contradicts the other's), so a ready vertex reads positions no thread writes in this launch;
    # the counts are read from ``counts`` and written to ``out_counts`` for the same reason. A fixed
    # neighbour is never written and is skipped. A vertex finishing its last sweep is counted into
    # ``done[0]``; once every vertex has, a wave changes nothing, so the caller may overrun the
    # fixpoint. Returns the count slot this vertex took, -1 when it took none.
    slot = wp.int32(-1)
    c = counts[v]
    if c < sweeps:
        ready = wp.int32(1)
        start = offsets[v]
        stop = offsets[v + 1]
        for k in range(start, stop):
            u = neighbors[k]
            if classes[u] != STENCIL_FIXED:
                want = wp.where(u < v, c + 1, c)
                if counts[u] != want:
                    ready = 0
                    break
        if ready != 0:
            relax_in_place(x, v, factor, neighbors, offsets, classes, rim_pairs)
            c += 1
            if c == sweeps:
                slot = append_slot(done, 0)
    out_counts[v] = c
    return slot


@wp.kernel
def gauss_seidel_wave(
    x: wp.array[wp.vec3d],
    counts: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    factor: wp.float64,
    sweeps: wp.int32,
    done: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # One wave over every vertex (``gauss_seidel_wave_vertex``).
    v = wp.int32(wp.tid())
    gauss_seidel_wave_vertex(
        x, counts, neighbors, offsets, classes, rim_pairs, factor, sweeps, done, out_counts, v
    )


@wp.kernel
def gauss_seidel_wave_counted(
    x: wp.array[wp.vec3d],
    counts: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    classes: wp.array[wp.int32],
    rim_pairs: wp.array[wp.vec2i],
    factor: wp.float64,
    sweeps: wp.int32,
    done: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # ``gauss_seidel_wave``, also recording the schedule's length, which depends on the stencil and
    # ``sweeps`` alone, never on the positions. The counts' trailing slot is the waves run before
    # this one, advanced by thread 0 into ``out_counts`` (read from ``counts``, so no thread reads a
    # slot written in this launch); the vertex taking the last count writes it plus one to
    # ``done[1]``. A separate kernel because the bookkeeping cost the plain wave 5-9 % of its
    # device time on dragon even where unused.
    v = wp.int32(wp.tid())
    n = out_counts.shape[0] - 1
    if v == 0:
        out_counts[n] = counts[n] + 1
    slot = gauss_seidel_wave_vertex(
        x, counts, neighbors, offsets, classes, rim_pairs, factor, sweeps, done, out_counts, v
    )
    if slot == n - 1:
        done[1] = counts[n] + 1


# The concrete handles of the generic pass kernels above, keyed by the dtype of the positions read
# and of the positions written (``humphrey_*``: the one generic side); see
# [`OverloadTable`][ordito.kernels.array.OverloadTable]. Filled in by ``_register_overloads``.
DIFFUSE_VEC3_PASS: OverloadTable
SHARPEN_PASS: OverloadTable
GRAPH_SHARPEN_PASS: OverloadTable
NEIGHBORHOOD_AVERAGE_PASS: OverloadTable
CORNER_CROSS_NORMALS: OverloadTable
SMOOTH_FACE_NORMALS_MESHLAB: OverloadTable
FIT_VERTICES_TO_NORMALS: OverloadTable
SMOOTH_FACE_NORMALS_MESHLAB_SERIAL: OverloadTable
FIT_VERTICES_TO_NORMALS_SERIAL: OverloadTable
HUMPHREY_RESIDUAL_PASS: OverloadTable
HUMPHREY_UPDATE_PASS: OverloadTable
MUT_DIF_ADIL_PASS: OverloadTable
MUT_DIF_STEP_SCALED: OverloadTable
IMPLICIT_LAPLACIAN_STEP: OverloadTable
IMPLICIT_LAPLACIAN_STEPS_SERIAL: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global DIFFUSE_VEC3_PASS, NEIGHBORHOOD_AVERAGE_PASS, HUMPHREY_RESIDUAL_PASS
    global HUMPHREY_UPDATE_PASS, SHARPEN_PASS, GRAPH_SHARPEN_PASS, MUT_DIF_ADIL_PASS
    global MUT_DIF_STEP_SCALED
    global IMPLICIT_LAPLACIAN_STEP, IMPLICIT_LAPLACIAN_STEPS_SERIAL
    global CORNER_CROSS_NORMALS, SMOOTH_FACE_NORMALS_MESHLAB, FIT_VERTICES_TO_NORMALS
    global SMOOTH_FACE_NORMALS_MESHLAB_SERIAL, FIT_VERTICES_TO_NORMALS_SERIAL
    vectors = (wp.vec3, wp.vec3d)
    scalars = {wp.vec3: wp.float32, wp.vec3d: wp.float64}
    indices = wp.array[wp.int32]
    CORNER_CROSS_NORMALS = OverloadTable(
        corner_cross_normals, {vec: [wp.array[vec], indices, wp.array[vec]] for vec in (wp.vec3d,)}
    )
    SMOOTH_FACE_NORMALS_MESHLAB = OverloadTable(
        smooth_face_normals_meshlab,
        {
            vec: [
                wp.array[vec],
                wp.array[wp.float32],
                indices,
                indices,
                indices,
                scalars[vec],
                wp.array[vec],
            ]
            for vec in (wp.vec3d,)
        },
    )
    FIT_VERTICES_TO_NORMALS = OverloadTable(
        fit_vertices_to_normals,
        {
            vec: [wp.array[vec], indices, wp.array[vec], indices, indices, wp.array[vec]]
            for vec in vectors
        },
    )
    SMOOTH_FACE_NORMALS_MESHLAB_SERIAL = OverloadTable(
        smooth_face_normals_meshlab_serial,
        {
            vec: [
                wp.array[vec],
                indices,
                indices,
                indices,
                scalars[vec],
                wp.int32,
                wp.array[vec],
                wp.array[vec],
            ]
            for vec in (wp.vec3d,)
        },
    )
    FIT_VERTICES_TO_NORMALS_SERIAL = OverloadTable(
        fit_vertices_to_normals_serial,
        {
            vec: [
                wp.array[vec],
                indices,
                wp.array[vec],
                indices,
                indices,
                wp.int32,
                wp.array[vec],
                wp.array[vec],
            ]
            for vec in vectors
        },
    )
    rows = [wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.float32]]
    IMPLICIT_LAPLACIAN_STEP = OverloadTable(
        implicit_laplacian_step,
        {
            (rhs, x): [*rows, wp.float64, wp.array[rhs], wp.array[x], wp.array[wp.vec3d]]
            for rhs, x in ((wp.vec3, wp.vec3), (wp.vec3, wp.vec3d), (wp.vec3d, wp.vec3d))
        },
    )
    IMPLICIT_LAPLACIAN_STEPS_SERIAL = OverloadTable(
        implicit_laplacian_steps_serial,
        {
            rhs: [*rows, wp.float64, wp.int32, wp.array[rhs], *([wp.array[wp.vec3d]] * 3)]
            for rhs in vectors
        },
    )
    MUT_DIF_ADIL_PASS = OverloadTable(
        mut_dif_adil_pass,
        {
            src: [
                *rows,
                wp.array[src],
                wp.array[wp.vec3],
                wp.int32,
                wp.array[wp.vec3d],
                wp.array[wp.float64],
            ]
            for src in vectors
        },
    )
    MUT_DIF_STEP_SCALED = OverloadTable(
        mut_dif_step_scaled,
        {
            (src, dst): [
                wp.array[src],
                wp.array[wp.vec3d],
                wp.array[wp.float64],
                wp.float64,
                wp.float64,
                wp.array[wp.vec3],
                wp.array[wp.float32],
                wp.array[dst],
                wp.array[wp.vec3d],
                wp.array[wp.float64],
            ]
            for src in vectors
            for dst in vectors
        },
    )
    DIFFUSE_VEC3_PASS = OverloadTable(
        diffuse_vec3_pass,
        {
            (src, dst): [*rows, wp.array[src], wp.float64, wp.array[dst]]
            for src in vectors
            for dst in vectors
        },
    )
    SHARPEN_PASS = OverloadTable(
        sharpen_pass,
        {
            src: [
                *rows,
                wp.array[src],
                wp.float64,
                wp.array[wp.vec3],
                wp.float32,
                wp.float32,
                wp.array[wp.vec3],
            ]
            for src in vectors
        },
    )
    GRAPH_SHARPEN_PASS = OverloadTable(
        graph_sharpen_pass,
        {
            (src, dst): [*rows, wp.array[src], wp.float64, wp.array[dst]]
            for src in vectors
            for dst in vectors
        },
    )
    NEIGHBORHOOD_AVERAGE_PASS = OverloadTable(
        neighborhood_average_pass,
        {(src, dst): [*rows, wp.array[src], wp.array[dst]] for src in vectors for dst in vectors},
    )
    HUMPHREY_RESIDUAL_PASS = OverloadTable(
        humphrey_residual_pass,
        {
            src: [
                *rows,
                wp.array[src],
                wp.array[wp.vec3],
                wp.float64,
                wp.array[wp.vec3d],
                wp.array[wp.vec3d],
            ]
            for src in vectors
        },
    )
    HUMPHREY_UPDATE_PASS = OverloadTable(
        humphrey_update_pass,
        {
            dst: [
                *rows,
                wp.array[wp.vec3d],
                wp.array[wp.vec3d],
                wp.float64,
                wp.float64,
                wp.array[dst],
            ]
            for dst in vectors
        },
    )


_register_overloads()
