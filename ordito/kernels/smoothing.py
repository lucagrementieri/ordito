import warp as wp

from ordito.kernels.array import greater, inverse_or_one, sort_segment, to_vec3, to_vec3d
from ordito.kernels.laplacian import face_half_cotangents, operator_row
from ordito.kernels.linalg import free_row, selected_row, solve_normal_equations
from ordito.kernels.predicates import angle_defect, closest_point_on_segment, plane_basis
from ordito.kernels.reduce import block_chunk, commit_block_total
from ordito.kernels.scatter import add_corner_triple
from ordito.kernels.triangles import corner_triple
from ordito.kernels.vertices import normalized_accumulated_row

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
    out_factor: wp.array[wp.float64],
    out_factor_t: wp.array[wp.float64],
    out_factor_narrow: wp.array[wp.float32],
    out_factor_t_narrow: wp.array[wp.float32],
) -> wp.float64:
    # Entry ``slot`` = ``(i, j)`` of ``B = D^-1 L_ff`` and of ``B^T`` (``normal_equations_setup``),
    # given ``L_ij = L_ji``; returns ``|L_ij|`` for the Gershgorin sum, ``0`` on an invalid row.
    sum_j = row_sums[j]
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
    #   order.
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
                    sum_i,
                    scale_i,
                    row_sums,
                    i,
                    diagonal,
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
            sum_i,
            scale_i,
            row_sums,
            i,
            slot,
            out_factor,
            out_factor_t,
            out_factor_narrow,
            out_factor_t_narrow,
        )
    out_term_counts[ri] = terms
    out_atb_x[ri] = ax
    out_atb_y[ri] = ay
    out_atb_z[ri] = az
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
def normal_equations_values(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    term_offsets: wp.array[wp.int32],
    row_columns: wp.array[wp.int32],
    row_values: wp.array[wp.float64],
    system_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # Row ``i`` of ``M^T M`` moved from the head of its term segment into the compact CSR.
    i = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, i)
    if ri < 0:
        return
    source = term_offsets[ri]
    start = system_offsets[ri]
    for s in range(system_offsets[ri + 1] - start):
        out_columns[start + s] = row_columns[source + s]
        out_values[start + s] = row_values[source + s]


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
def interior_mass_rhs_and_seed(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    positions: wp.array[wp.vec3d],
    rhs: wp.array2d[wp.float64],
    out_solution: wp.array2d[wp.float64],
) -> None:
    # The two per-free-vertex writes the implicit-fairing Dirichlet pass needs before its solve,
    # from one read of the vertex:
    #
    # - Add the linear term ``b_u = (M V)_u`` into the reduced right-hand side, which arrives
    #   holding only ``-A_ub x_b`` from ``linalg.assemble_interior_system`` (that helper eliminates
    #   the pinned columns of a quadratic form, which has no linear term of its own). ``rhs`` is
    #   genuinely in-place -- an accumulator carrying that prior term in, not a fresh answer --
    #   which is why it does not carry the ``out_`` prefix reserved for write-only outputs
    #   (CLAUDE.md section 2.1).
    # - Seed the reduced solve with the free vertices' *current* positions: the exact inverse of
    #   ``scatter_free_positions`` below, and the counterpart of ``gather_free_positions`` above for
    #   the ``fixed_mask`` partition and the float64 storage the implicit-fairing flow carries.
    #   Seeding from the right-hand side instead would leave a vertex no face refers to at the
    #   origin -- it has an all-zero row and so a zero right-hand side, and CG never writes its
    #   entry -- which is the failure ``gather_free_positions`` exists to avoid on the region
    #   solves.
    v = wp.int32(wp.tid())
    i = free_row(fixed_mask, free_map, v)
    if i < 0:
        return
    m = mass[v]
    p = positions[v]
    rhs[0, i] = rhs[0, i] + m * p[0]
    rhs[1, i] = rhs[1, i] + m * p[1]
    rhs[2, i] = rhs[2, i] + m * p[2]
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
) -> None:
    # Write the reduced solution back to the unpinned vertices only. Pinned ones are left holding
    # whatever they already have, which is their original position -- they never move.
    v = wp.int32(wp.tid())
    i = free_row(fixed_mask, free_map, v)
    if i < 0:
        return
    out_positions[v] = wp.vec3d(sol_x[i], sol_y[i], sol_z[i])


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
    volume_current: wp.array[wp.float64], state: wp.array[wp.float64]
) -> None:
    """
    Form the volume constraint's scale and centre once on the device, for a whole pass.

    In place: ``state`` is ``[volume, centroid integrals x 4, scale, centre x 3]``. The first five
    are written before the smoothing starts -- the signed volume of the input in ``float64``
    (``measures.volume``'s kernel) and the first four of ``measures.moments``' integrals of its
    ``float32`` copy -- and this kernel writes the last four from them and the current volume. The
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
        ratio = state[0] / current
        if ratio > wp.float64(0.0):
            scale = wp.pow(ratio, wp.float64(1.0) / wp.float64(3.0))
    state[5] = scale
    total = state[1]
    center = wp.vec3d(wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    if total != wp.float64(0.0):
        center = to_vec3d(
            wp.vec3(
                wp.float32(state[2] / total),
                wp.float32(state[3] / total),
                wp.float32(state[4] / total),
            )
        )
    for k in range(3):
        state[6 + k] = center[k]


@wp.func
def rescaled_to_volume(state: wp.array[wp.float64], position: wp.vec3d) -> wp.vec3d:
    # ``position`` rescaled by the scale and about the centre ``volume_rescale_parameters`` wrote;
    # a zero scale is the skip.
    scale = state[5]
    if scale == wp.float64(0.0):
        return position
    return rescale_about_center(position, wp.vec3d(state[6], state[7], state[8]), scale)


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


@wp.kernel
def diffuse_vec3_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[wp.vec3d],
    coeff: wp.float64,
    out_next: wp.array[wp.vec3d],
) -> None:
    # Explicit diffusion step v' = v + coeff * (L.v - v); coeff = +lambda (shrink) or -nu
    # (inflate). ``wp.lerp`` extrapolates for coeff outside [0, 1], which the inflating step of
    # ``filter_taubin`` relies on.
    i = wp.int32(wp.tid())
    lv = operator_row(offsets, columns, values, positions, i)
    out_next[i] = wp.lerp(positions[i], lv, coeff)


@wp.kernel
def neighborhood_average_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[wp.vec3d],
    out_next: wp.array[wp.vec3d],
) -> None:
    # Closed 1-ring average: new_v = (v + deg * L.v) / (deg + 1), where L is the neighbors-only
    # averaging operator and deg is the CSR row length (the vertex degree). deg = 0 -> new_v = v.
    i = wp.int32(wp.tid())
    lv = operator_row(offsets, columns, values, positions, i)
    deg = wp.float64(offsets[i + 1] - offsets[i])
    out_next[i] = (positions[i] + deg * lv) / (deg + wp.float64(1.0))


@wp.kernel
def mark_spikes(
    angle_sums: wp.array[wp.float32],
    min_defect: wp.float32,
    out_spikes: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
) -> None:
    # A spike is a vertex whose angle *defect* ``2 pi - angle_sum`` exceeds ``min_defect``
    # (``vertices.vertex_defects``' quantity, ``array.greater``'s test), counted into ``out_count``.
    # In place: the sum is re-zeroed once read, so the next pass's scatter needs no fill.
    v = wp.int32(wp.tid())
    spike = greater(angle_defect(angle_sums[v]), min_defect)
    angle_sums[v] = wp.float32(0.0)
    out_spikes[v] = spike
    if spike:
        wp.atomic_add(out_count, 0, 1)


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
    out_count: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The spike rows of ``laplacian.laplacian(symmetric=True)``'s pattern, as
    # ``row * base + column`` keys: each corner on a spike contributes its face's other two
    # corners, a degenerate face's repeated vertex its self-edge. Appended through ``out_count``'s
    # cursor into a radix sort's double-width buffers (identity payload) while they have room; past
    # it only the count grows, and the caller re-runs at the size it read.
    f = wp.int32(wp.tid())
    capacity = out_order.shape[0] // 2
    for k in range(3):
        v = faces[3 * f + k]
        if spikes[v]:
            slot = wp.atomic_add(out_count, 0, 2)
            if slot + 2 <= capacity:
                row = wp.uint64(v) * base
                out_keys[slot] = row + wp.uint64(faces[3 * f + (k + 1) % 3])
                out_keys[slot + 1] = row + wp.uint64(faces[3 * f + (k + 2) % 3])
                out_order[slot] = slot
                out_order[slot + 1] = slot + 1


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


@wp.kernel
def humphrey_residual_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[wp.vec3d],
    original: wp.array[wp.vec3d],
    alpha: wp.float64,
    out_lv: wp.array[wp.vec3d],
    out_b: wp.array[wp.vec3d],
) -> None:
    # b = L.v - (alpha * original + (1 - alpha) * q), the Humphrey correction term. ``out_lv`` is
    # still written because the update pass below reads it; only ``L.b`` disappears.
    i = wp.int32(wp.tid())
    lv = operator_row(offsets, columns, values, positions, i)
    out_lv[i] = lv
    out_b[i] = lv - wp.lerp(positions[i], original[i], alpha)


@wp.kernel
def humphrey_update_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lv: wp.array[wp.vec3d],
    b: wp.array[wp.vec3d],
    beta: wp.float64,
    out_next: wp.array[wp.vec3d],
) -> None:
    # v' = L.v - (beta * b + (1 - beta) * L.b), with L.b formed here rather than in a buffer.
    i = wp.int32(wp.tid())
    lb = operator_row(offsets, columns, values, b, i)
    out_next[i] = lv[i] - wp.lerp(lb, b[i], beta)


@wp.func
def mut_dif_adil(normal: wp.vec3, position: wp.vec3d, lv: wp.vec3d) -> wp.float64:
    # adil = 1 / max(1e-12, |N . (V - L.V)|), the reciprocal normal-residual magnitude per vertex.
    # One spelling for the pass that sums it and the step that recomputes it from the same three
    # values, so the step's coefficient is bit-identical to the one the mean was taken over.
    residual = wp.abs(wp.dot(to_vec3d(normal), position - lv))
    return wp.float64(1.0) / wp.max(wp.float64(1e-12), residual)


MUT_DIF_ROWS_PER_BLOCK = wp.constant(256)


@wp.kernel
def mut_dif_adil_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    positions: wp.array[wp.vec3d],
    normals: wp.array[wp.vec3],
    out_lv: wp.array[wp.vec3d],
    out_adil_sum: wp.array[wp.float64],
) -> None:
    # Applies the operator row and folds each vertex's ``mut_dif_adil`` straight into the mean's
    # sum: ``MUT_DIF_ROWS_PER_BLOCK`` rows per block, lane-strided by ``wp.block_dim()`` (so the
    # CPU device's single lane covers the block's rows), one ``float64`` atomic per block. The
    # coefficient is not stored: ``mut_dif_step_scaled`` recomputes it at its own vertex from the
    # ``out_lv`` row written here, which costs it a normal load against an ``(n,)`` buffer's write
    # and read and the separate reduction launch. A narrower fold than the reduce module's because
    # the kernel also writes per row (CLAUDE.md section 13.2).
    chunk, lane = wp.tid()
    offset, remaining = block_chunk(positions.shape[0], chunk, MUT_DIF_ROWS_PER_BLOCK)
    if remaining <= 0:
        return
    total = wp.float64(0.0)
    for r in range(lane, remaining, wp.block_dim()):
        i = offset + r
        lv = operator_row(offsets, columns, values, positions, i)
        out_lv[i] = lv
        total = total + mut_dif_adil(normals[i], positions[i], lv)
    commit_block_total(lane, total, out_adil_sum, 0)


@wp.func
def add_scaled_normal(v_prev: wp.vec3d, normal: wp.vec3, scale: wp.float64) -> wp.vec3d:
    # v' = v + scale * N; reused for the eps finite-difference probe and the volume correction.
    return v_prev + scale * to_vec3d(normal)


@wp.kernel
def mut_dif_step_scaled(
    positions: wp.array[wp.vec3d],
    lv: wp.array[wp.vec3d],
    adil_sum: wp.array[wp.float64],
    inv_n: wp.float64,
    lamb: wp.float64,
    normals: wp.array[wp.vec3],
    probe_scale: wp.float64,
    out_next: wp.array[wp.vec3d],
    out_probe: wp.array[wp.vec3d],
) -> None:
    # ``mut_dif_step`` with the mean coefficient read from a device scalar (adil_sum[0] * inv_n),
    # so the smoothing loop never synchronises with the host. A real kernel rather than wp.map:
    # the length-1 ``adil_sum`` is a uniform argument, which wp.map cannot broadcast.
    #
    # A non-empty ``out_probe`` (the volume constraint's first pass) also receives the stepped
    # vertex offset by ``probe_scale`` along its normal: the finite-difference probe
    # ``mut_dif_volume_correct`` calibrates its slope against, formed from the value this thread
    # has just computed rather than read back by a map of its own.
    i = wp.int32(wp.tid())
    mean_adil = adil_sum[0] * inv_n
    position = positions[i]
    row = lv[i]
    stepped = mut_dif_step(position, row, mut_dif_adil(normals[i], position, row), mean_adil, lamb)
    out_next[i] = stepped
    if out_probe.shape[0] > 0:
        out_probe[i] = add_scaled_normal(stepped, normals[i], probe_scale)


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
    eps: wp.float64,
    calibrate: wp.int32,
    slope: wp.array[wp.float64],
    out_positions: wp.array[wp.vec3d],
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
        s = mut_dif_volume_slope(volume_current, volume_probe, eps)
        if v == 0:
            slope[0] = s
    else:
        s = slope[0]
    offset = s * (volume_initial[0] - volume_current[0])
    out_positions[v] = add_scaled_normal(out_positions[v], normals[v], offset)


@wp.func
def extract_components(v: wp.vec3d) -> tuple[wp.float64, wp.float64, wp.float64]:
    return v[0], v[1], v[2]


@wp.func
def seed_and_mass_weight_components(
    v: wp.vec3d, mass: wp.float64
) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64, wp.float64, wp.float64]:
    # ``extract_components`` twice over, once as is and once weighted by the lumped mass: the
    # implicit-fairing pass seeds its solve with the positions and solves against ``M V``, and both
    # are per-vertex, so one map writes the three seed rows and the three right-hand-side rows.
    return v[0], v[1], v[2], mass * v[0], mass * v[1], mass * v[2]


@wp.func
def combine_components(x: wp.float64, y: wp.float64, z: wp.float64) -> wp.vec3d:
    return wp.vec3d(x, y, z)


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
def operator_row_abs_sums(
    offsets: wp.array[wp.int32], values: wp.array[wp.float32], out_sums: wp.array[wp.float64]
) -> None:
    # ``sum_j |L_ij|`` per row, the infinity norm the fixed-point iteration below contracts by. An
    # empty row reads 1: ``operator_row`` applies it as the identity.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    total = wp.float64(0.0)
    for k in range(start, end):
        total += wp.abs(wp.float64(values[k]))
    out_sums[i] = wp.where(end == start, wp.float64(1.0), total)


@wp.kernel
def implicit_laplacian_step(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lamb: wp.float64,
    rhs: wp.array[wp.vec3d],
    x: wp.array[wp.vec3d],
    out_x: wp.array[wp.vec3d],
) -> None:
    # One step of the fixed-point iteration for the backward-Euler system
    # ``((1 + lamb) I - lamb L) x = rhs``: ``x' = (rhs + lamb L x) / (1 + lamb)``. It contracts by
    # ``lamb ||L||_inf / (1 + lamb)`` whatever ``L``'s symmetry, which conjugate gradient needs.
    # Each thread reads ``rhs`` at its own row only, so ``out_x`` may alias ``rhs`` -- the last
    # step writes the answer over it -- but never ``x``, which the row sum reads at other rows.
    # Shares ``operator_row`` with the explicit step, empty-row convention included.
    i = wp.int32(wp.tid())
    out_x[i] = (rhs[i] + lamb * operator_row(offsets, columns, values, x, i)) / (
        wp.float64(1.0) + lamb
    )


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
    # contaminates its (nonexistent) neighbours.
    i = wp.int32(wp.tid())
    value = field[i]
    start = offsets[i]
    end = offsets[i + 1]
    average = value
    if end > start:
        total = wp.float32(0.0)
        for k in range(start, end):
            total += values[k] * field[columns[k]]
        average = total
    out_next[i] = wp.lerp(value, average, lamb)


@wp.kernel
def renormalize_and_reseed(
    areas: wp.array[wp.float32], accumulated: wp.array[wp.vec3], out_normals: wp.array[wp.vec3]
) -> None:
    # One pass's normalization fused with the *next* pass's seed. The two are adjacent across the
    # loop boundary rather than inside one iteration -- ``accumulate_smoothed_normals`` sits between
    # the seed and the normalization and scatters across faces, so it needs the whole seeded buffer
    # and cannot be folded in -- and both halves are per-face, so a pass loop issues two launches
    # instead of three once the first seed is peeled off the front.
    #
    # ``accumulated`` is in place: this thread reads its own slot and immediately overwrites it with
    # the next pass's seed, so it is both the input and the result.
    f = wp.int32(wp.tid())
    normal = wp.normalize(accumulated[f])
    out_normals[f] = normal
    accumulated[f] = seed_weighted_normal(normal, areas[f])


@wp.kernel
def accumulate_smoothed_normals(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    face_adjacency: wp.array2d[wp.int32],
    threshold_cos: wp.float32,
    out_accumulated: wp.array[wp.vec3],
) -> None:
    # Area-weighted average of a face's normal with those of its edge-neighbours -- but only the
    # neighbours pointing *within* ``threshold_cos`` of it. That gate is the whole point: across a
    # crease the two normals disagree by more than the threshold and simply do not average, so a
    # sharp edge survives an arbitrary number of passes while noise on a flat region diffuses away.
    k = wp.int32(wp.tid())
    f0 = face_adjacency[k, 0]
    f1 = face_adjacency[k, 1]
    if wp.dot(face_normals[f0], face_normals[f1]) <= threshold_cos:
        return
    wp.atomic_add(out_accumulated, f0, face_areas[f1] * face_normals[f1])
    wp.atomic_add(out_accumulated, f1, face_areas[f0] * face_normals[f0])


@wp.func
def seed_weighted_normal(normal: wp.vec3, area: wp.float32) -> wp.vec3:
    # A face's own area-weighted normal: the seed of the accumulator above, so the face always
    # contributes to its own average even when every neighbour is across a crease.
    return area * normal


@wp.kernel
def fit_vertices_to_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    out_delta: wp.array[wp.vec3],
) -> None:
    # One gradient step of the vertex-fitting half of two-step smoothing (Ohtake et al.): each
    # incident face wants its corner to lie in the plane through the face centroid with the
    # *filtered* normal, and the correction is the component of that offset along the normal.
    #
    # Summed per vertex and divided by the incident-face count in ``apply_fit_step_and_reset``,
    # which is the step size that makes the iteration a contraction without a tuning constant. The
    # count is the topology's, so the caller takes it once with ``scatter.count_occurrences``
    # rather than re-accumulating it here every fit iteration -- a sum of ones is exact in any
    # order, so the step is unchanged to the bit.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    normal = face_normals[f]
    centroid = (vertices[i0] + vertices[i1] + vertices[i2]) / 3.0
    for k in range(3):
        v = faces[f * 3 + k]
        wp.atomic_add(out_delta, v, normal * wp.dot(normal, centroid - vertices[v]))


@wp.kernel
def apply_fit_step_and_reset(
    counts: wp.array[wp.int32], out_positions: wp.array[wp.vec3], out_delta: wp.array[wp.vec3]
) -> None:
    # Move each vertex by its mean correction, then zero its accumulator for the next fit
    # iteration. The reset rides here because this is the one thread that reads the slot, and it
    # reads it before overwriting it -- so the fit loop issues two launches per iteration where a
    # separate ``zero_`` in front of each scatter made three. ``out_positions`` and ``out_delta``
    # are both in place.
    i = wp.int32(wp.tid())
    count = counts[i]
    if count > 0:
        out_positions[i] = out_positions[i] + out_delta[i] / wp.float32(count)
    out_delta[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def unsharp_step(
    position: wp.vec3, smoothed: wp.vec3, weight: wp.float32, weight_original: wp.float32
) -> wp.vec3:
    # MeshLab's ``apply_coord_unsharp_mask``: add back a multiple of the high-frequency detail the
    # smoothing pass removed. ``weight_original = 1`` keeps the surface in place and only sharpens.
    return weight_original * position + weight * (position - smoothed)


@wp.func
def step_along_normal(position: wp.vec3, normal: wp.vec3, distance: wp.float32) -> wp.vec3:
    """Move a vertex along its own normal, which is one half of an inflation step."""
    return position + normal * distance


@wp.kernel
def inflate_displace(
    positions: wp.array[wp.vec3],
    normal_sums: wp.array2d[wp.float64],
    distance: wp.float32,
    out_displaced: wp.array[wp.vec3],
    out_displaced_wide: wp.array[wp.vec3d],
) -> None:
    # One inflation step: each vertex moved along its area-weighted normal, normalized as it is read
    # from the scatter's accumulator (``vertices.normalize_accumulated_rows``' arithmetic, so the
    # normals are never written), stored at both precisions -- the ``float32`` copy feeds the
    # centre-of-mass integrals, the exact ``float64`` widening the diffusion and its volume.
    v = wp.int32(wp.tid())
    moved = step_along_normal(positions[v], normalized_accumulated_row(normal_sums, v), distance)
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


@wp.func
def next_row_entry(
    values: wp.array[wp.int32],
    begin: wp.int32,
    end: wp.int32,
    previous_value: wp.int32,
    previous_slot: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # The entry of ``values[begin:end]`` that follows ``(previous_value, previous_slot)`` in
    # ``(value, slot)`` order -- a row visited in ascending value order whatever order it is
    # stored in, a repeated value once per occurrence. Start from ``(-1, -1)``; an O(row) scan, for
    # rows of a few entries.
    best_value = wp.int32(2147483647)
    best_slot = wp.int32(-1)
    for slot in range(begin, end):
        value = values[slot]
        after = value > previous_value or (value == previous_value and slot > previous_slot)
        if after and (value < best_value or (value == best_value and slot < best_slot)):
            best_value = value
            best_slot = slot
    return best_value, best_slot


# ``equal_area_position``'s float32 path is trusted while its system's determinant is at least this
# fraction of the cubed (or, in the tangent plane, squared) trace -- an isotropic 3x3 reads 1/27 --
# and the float64 accumulation is redone below it. Swept at 1e-6 / 1e-4 / 1e-2 on bunny, dragon and
# lucy (one pass, against the float64 accumulation): 1e-6 still let a dragon ring through 3e-6
# off, from 1e-4 the largest difference is the positions' own float32 rounding. Ten passes through
# ``equalize_triangle_areas``: 1.44x bunny, 2.9x dragon, 3.2x lucy (1.8x with ``no_shrinkage``,
# which recomputes normals each pass). With ``no_shrinkage`` the filter flings ~500 dragon vertices
# beyond 0.01 of their start in ten passes on either accumulation, and those chaotic vertices
# differ run to run on the float64 one too; everything else agrees to rounding.
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


@wp.kernel
def equalize_area_step(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    no_shrinkage: wp.bool,
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # One pass of area equalization: step each in-region vertex a fraction ``force`` of the way to
    # its own equal-area minimum, then clamp it back near where it started.
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    if not in_region(region, vertex) or offsets[vertex] == offsets[vertex + 1]:
        out_positions[vertex] = current
        return
    target = equal_area_position(
        positions, faces, offsets, vertex_faces, vertex, normals[vertex], no_shrinkage
    )
    moved = current + (target - current) * force
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


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
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    if not in_region(region, vertex) or begin == end:
        out_push[vertex] = wp.vec3(0.0, 0.0, 0.0)
        return
    total = wp.vec3d()
    for slot in range(begin, end):
        total += to_vec3d(positions[columns[slot]])
    mean = total / wp.float64(end - begin)
    average = to_vec3(mean)
    out_push[vertex] = (average - positions[vertex]) * force


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
    current = positions[vertex]
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    if not in_region(region, vertex) or begin == end:
        out_positions[vertex] = current
        return
    total = wp.vec3()
    for slot in range(begin, end):
        neighbor = columns[slot]
        if in_region(region, neighbor):
            total += push[neighbor]
    moved = current + push[vertex] - total / wp.float32(end - begin)
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


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


@wp.kernel
def relax_approx_step(
    positions: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
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
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    begin = neighbor_offsets[vertex]
    end = neighbor_offsets[vertex + 1]  # terminated (n + 1) CSR row bounds from ``geodesic_ball``
    if not in_region(region, vertex) or end - begin < 6:
        out_positions[vertex] = current
        return

    centroid, axis_u, axis_v, axis_w, radius = _neighborhood_frame(
        positions, neighbor_indices, begin, end
    )
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
            local = positions[neighbor_indices[slot]] - centroid
            u = wp.float64(wp.dot(local, axis_u)) * inv_radius
            v = wp.float64(wp.dot(local, axis_v)) * inv_radius
            row = vec6d(u * u, u * v, v * v, u, v, wp.float64(1.0))
            normal_matrix += wp.outer(row, row)
            normal_rhs += row * (wp.float64(wp.dot(local, axis_w)) * inv_radius)
        coefficients, ok = solve_normal_equations(normal_matrix, normal_rhs)
        if ok:
            # Evaluated in the same units the fit was solved in, then carried back: ``w`` was
            # divided by the radius alongside ``u`` and ``v``, so the height is multiplied by it.
            u = wp.float64(wp.dot(offset, axis_u)) * inv_radius
            v = wp.float64(wp.dot(offset, axis_v)) * inv_radius
            height = wp.float32(
                wp.float64(radius)
                * (
                    coefficients[0] * u * u
                    + coefficients[1] * u * v
                    + coefficients[2] * v * v
                    + coefficients[3] * u
                    + coefficients[4] * v
                    + coefficients[5]
                )
            )

    target = current + axis_w * (height - wp.dot(offset, axis_w))
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
