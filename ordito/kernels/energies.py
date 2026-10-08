import warp as wp

from ordito.kernels.array import OverloadTable, to_vec3d
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.predicates import (
    angle_defect,
    corner_cosines_from_l2,
    squared_edge_lengths,
    triangle_double_area,
)
from ordito.kernels.scatter import add_corner_triple, append_to_pair
from ordito.kernels.triangles import face_vertices_vec3d

wp.set_module_options({"enable_backward": False})


@wp.func
def reciprocal_or_zero(value: wp.Float) -> wp.Float:
    # ``igl::invert_diag`` semantics: zero entries stay zero instead of becoming infinities, so a
    # killed degree of freedom (boundary vertex, empty edge row) simply contributes nothing.
    if value > type(value)(0.0):
        return type(value)(1.0) / value
    return type(value)(0.0)


@wp.func
def squared_deviation(value: wp.Float, target: wp.Float) -> wp.Float:
    # ``(x - L0)^2``, the per-edge term of the edge-length regularizer. A ``@wp.func`` rather than
    # an inline kernel body so ``energies.edge_length_loss`` can reach it through ``wp.map``.
    difference = value - target
    return difference * difference


@wp.func
def reciprocal_scaled_or_zero(value: wp.Float, numerator: wp.Float) -> wp.Float:
    # ``numerator / value``, with the ``reciprocal_or_zero`` convention above for a non-positive
    # denominator: a vertex whose lumped area is zero contributes nothing rather than an infinity.
    # Called by ``laplacian_smoothing_norms`` and by ``scaled_angle_defect`` below.
    if value > type(value)(0.0):
        return numerator / value
    return type(value)(0.0)


@wp.func
def scaled_angle_defect(angle_sum: wp.float64) -> wp.float64:
    # ``angle_defect / angle_sum``, zeroed for a non-positive angle sum --
    # igl::cr_vector_curvature_correction's kappa scaling, formed by ``curved_hessian_triplets`` as
    # it reads each vertex's angle sum. ``curved_hessian_energy`` zeroes the boundary's sums first,
    # which this maps to the zero kappa the boundary takes.
    return reciprocal_scaled_or_zero(angle_sum, angle_defect(angle_sum))


@wp.func
def one_minus_cosine(angle: wp.Float) -> wp.Float:
    # ``1 - cos(theta)``, the per-pair term of the normal-consistency regularizer: 0 for a flat
    # pair, 1 at a right angle, 2 for a fold back on itself.
    return type(angle)(1.0) - wp.cos(angle)


# ``laplacian_smoothing_norms``' warp-uniform method selector, one per
# ``energies.laplacian_smoothing_loss`` method.
SMOOTHING_UNIFORM = wp.constant(0)
SMOOTHING_COT = wp.constant(1)
SMOOTHING_COTCURV = wp.constant(2)


@wp.kernel(enable_backward=False)
def laplacian_smoothing_norms(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    vertices: wp.array[wp.vec3],
    mass: wp.array[wp.float32],
    method: wp.int32,
    out_norms: wp.array[wp.float32],
) -> None:
    # ``|| row_scale * (L v)_i + self_scale * v_i ||`` for one CSR row, with the two scales of
    # ``energies.laplacian_smoothing_loss``'s ``method`` formed in the row's own thread:
    #
    # - ``SMOOTHING_UNIFORM``: the umbrella residual ``(A v)_i - v_i`` (``1`` and ``-1``).
    # - ``SMOOTHING_COT``: the cotangent-weighted neighbour average ``(L v)_i / s_i``, ``s_i`` the
    #   off-diagonal row sum. A cotangent Laplacian carries ``-s_i`` on its diagonal, so the scale
    #   is the negated diagonal, picked up during the same row walk. Where ``s_i`` is not positive
    #   (an obtuse ring whose weights cancel, or an isolated vertex) the averaging is undefined and
    #   the residual falls back to ``-v_i``, which is what the reference's ``norm_w = 0`` branch
    #   amounts to.
    # - ``SMOOTHING_COTCURV``: the stiffness residual over six times the lumped area ``mass[i]``,
    #   with ``reciprocal_scaled_or_zero``'s convention that a zero-area vertex contributes nothing.
    #
    # ``mass`` is read only by the last; the others pass a null array.
    #
    # **The diagonal is read during the row walk deliberately, and this is the fourth spelling of
    # "get the diagonal" in the tree.** The other three are ``linalg._multigrid_levels`` (through
    # ``wps.bsr_get_diag``), ``reconstruction.poisson_level_setup`` and
    # ``algorithms/conjugate_gradient.scaled_diagonal_apply``. Routing this one through
    # ``wps.bsr_get_diag`` was measured and declined at several times the cost of a separate scale
    # launch, flat in the system size, and here the walk is the residual's own, so the diagonal
    # costs one compare per entry. The values agree **exactly**. The walk assumes the diagonal is
    # *present* in the pattern, and that is not load-bearing: an absent diagonal leaves
    # ``diagonal`` at zero and takes the same undefined-averaging fallback ``bsr_get_diag``'s own
    # zero would.
    i = wp.int32(wp.tid())
    accumulator = wp.vec3(0.0, 0.0, 0.0)
    diagonal = wp.float32(0.0)
    for slot in range(offsets[i], offsets[i + 1]):
        column = columns[slot]
        accumulator += values[slot] * vertices[column]
        if column == i:
            diagonal = values[slot]
    row_scale = wp.float32(1.0)
    self_scale = wp.float32(-1.0)
    if method == SMOOTHING_COT:
        row_sum = -diagonal
        row_scale = wp.float32(0.0)
        if row_sum > 0.0:
            row_scale = 1.0 / row_sum
            self_scale = wp.float32(0.0)
    elif method == SMOOTHING_COTCURV:
        row_scale = reciprocal_scaled_or_zero(mass[i], wp.float32(1.0 / 6.0))
        self_scale = wp.float32(0.0)
    out_norms[i] = wp.length(row_scale * accumulator + self_scale * vertices[i])


@wp.kernel
def sandwich_row_counts(
    a_offsets: wp.array[wp.int32],
    b_offsets: wp.array[wp.int32],
    inv_mass: wp.array[wp.Float],
    out_counts: wp.array[wp.int32],
) -> None:
    # Triplets emitted by row ``t`` of the product ``A diag(inv_mass) B``: the full outer product
    # of the two CSR rows, or nothing when the diagonal weight is zero.
    t = wp.int32(wp.tid())
    if inv_mass[t] > type(inv_mass[0])(0.0):
        out_counts[t] = (a_offsets[t + 1] - a_offsets[t]) * (b_offsets[t + 1] - b_offsets[t])
    else:
        out_counts[t] = 0


@wp.kernel
def sandwich_row_triplets(
    a_offsets: wp.array[wp.int32],
    a_columns: wp.array[wp.int32],
    a_values: wp.array[wp.Float],
    b_offsets: wp.array[wp.int32],
    b_columns: wp.array[wp.int32],
    b_values: wp.array[wp.Float],
    inv_mass: wp.array[wp.Float],
    segment_offsets: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # ``(A diag(d) B)_ij = sum_t A_ti d_t B_tj``; with both operands symmetric ``A_ti`` is row
    # ``t``'s entry at column ``i``, so each row of the diagonal sandwich is the scaled outer
    # product of the two matching CSR rows, assembling the product without ``bsr_mm``. (The
    # ``bsr_mm`` nondeterminism this originally avoided was a triplet-capacity defect of this
    # package, not a Warp bug — see ``k_harmonic`` in ``ordito/energies.py``.)
    t = wp.int32(wp.tid())
    weight = inv_mass[t]
    # The negation of ``sandwich_row_counts``'s own gate, not an independent ``<=`` -- a NaN
    # ``inv_mass[t]`` fails both ``>`` and ``<=``, so the two spellings disagree on it and the
    # count kernel (which reserves 0 triplets for a killed row) would silently under-allocate
    # against this kernel writing a full row's worth into the next row's reserved segment.
    if not (weight > type(inv_mass[0])(0.0)):
        return
    cursor = segment_offsets[t]
    for a in range(a_offsets[t], a_offsets[t + 1]):
        row = a_columns[a]
        left = a_values[a] * weight
        for b in range(b_offsets[t], b_offsets[t + 1]):
            out_rows[cursor] = row
            out_cols[cursor] = b_columns[b]
            out_vals[cursor] = left * b_values[b]
            cursor += 1


@wp.func
def triangle_geometry_f64(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64]:
    """
    Squared edge lengths and twice the area of face ``f``, promoted to ``float64``.

    The shared geometry preamble of
    [`hessian_face_terms`][ordito.kernels.energies.hessian_face_terms],
    [`internal_angles_and_sums`][ordito.kernels.energies.internal_angles_and_sums] and
    [`curved_hessian_triplets`][ordito.kernels.energies.curved_hessian_triplets] -- all three load
    a face's vertices, its three squared edge lengths and its double area before doing their own,
    unrelated per-corner computation with them. ``squared_edge_lengths`` and
    ``triangle_double_area`` have no data dependency on each other, so factoring their call order
    into one place changes neither result; only the caller-specific math after this stays apart.
    """
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    l2_0, l2_1, l2_2 = squared_edge_lengths(v0, v1, v2)
    dbl_area = triangle_double_area(v0, v1, v2)
    return l2_0, l2_1, l2_2, dbl_area


@wp.kernel
def hessian_face_terms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_gradients: wp.array[wp.vec3d],
    out_areas: wp.array[wp.float64],
    out_mass: wp.array[wp.float64],
) -> None:
    # ``hessian_energy``'s two per-face passes in one launch. The corner gradients and the Voronoi
    # mass lumping are independent quantities over the same faces -- neither reads the other -- so
    # nothing is shared but the face index and the vertex loads, and running them in one thread
    # removes a launch and lets the two share those loads.
    f = wp.int32(wp.tid())

    # Gradient of each corner's hat function inside its face, ``n x e_c / (2 A)`` with ``e_c`` the
    # CCW edge opposite corner ``c``. A degenerate face gets zero gradients (it contributes nothing
    # to the energy) rather than a division by its zero area.
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    normal = wp.cross(v1 - v0, v2 - v0)
    dbl_area = wp.length(normal)
    out_areas[f] = wp.float64(0.5) * dbl_area
    zero = wp.float64(0.0)
    g0 = wp.vec3d(zero, zero, zero)
    g1 = wp.vec3d(zero, zero, zero)
    g2 = wp.vec3d(zero, zero, zero)
    if dbl_area > zero:
        unit = normal / dbl_area
        g0 = wp.cross(unit, v2 - v1) / dbl_area
        g1 = wp.cross(unit, v0 - v2) / dbl_area
        g2 = wp.cross(unit, v1 - v0) / dbl_area
    out_gradients[f * 3 + 0] = g0
    out_gradients[f * 3 + 1] = g1
    out_gradients[f * 3 + 2] = g2

    # The ``igl::massmatrix`` MASSMATRIX_TYPE_VORONOI lumping: true Voronoi quad areas on
    # non-obtuse triangles, the 1/2 : 1/4 : 1/4 split on obtuse ones (the obtuse corner gets the
    # half). A degenerate face contributes nothing (igl would emit NaN).
    # ``dbl_area`` above is ``wp.length(wp.cross(v1 - v0, v2 - v0))``, which is exactly what
    # ``triangle_double_area`` computes, so the mass half reuses it rather than re-deriving the
    # face's geometry -- that shared preamble is the second thing the fusion removes, after the
    # launch.
    if dbl_area <= zero:
        return
    l2_0, l2_1, l2_2 = squared_edge_lengths(v0, v1, v2)
    cos0, cos1, cos2 = corner_cosines_from_l2(l2_0, l2_1, l2_2)
    bary0 = cos0 * wp.sqrt(l2_0)
    bary1 = cos1 * wp.sqrt(l2_1)
    bary2 = cos2 * wp.sqrt(l2_2)
    total = bary0 + bary1 + bary2
    half_dbl = wp.float64(0.5) * dbl_area
    partial0 = bary0 / total * half_dbl
    partial1 = bary1 / total * half_dbl
    partial2 = bary2 / total * half_dbl
    quad0 = wp.float64(0.5) * (partial1 + partial2)
    quad1 = wp.float64(0.5) * (partial2 + partial0)
    quad2 = wp.float64(0.5) * (partial0 + partial1)
    # At most one angle can be obtuse, so the three rewrites cannot both fire.
    if cos0 < wp.float64(0.0):
        quad0 = wp.float64(0.25) * dbl_area
        quad1 = wp.float64(0.125) * dbl_area
        quad2 = wp.float64(0.125) * dbl_area
    if cos1 < wp.float64(0.0):
        quad0 = wp.float64(0.125) * dbl_area
        quad1 = wp.float64(0.25) * dbl_area
        quad2 = wp.float64(0.125) * dbl_area
    if cos2 < wp.float64(0.0):
        quad0 = wp.float64(0.125) * dbl_area
        quad1 = wp.float64(0.125) * dbl_area
        quad2 = wp.float64(0.25) * dbl_area
    add_corner_triple(out_mass, faces, f, quad0, quad1, quad2)


@wp.kernel
def zero_at_indices(indices: wp.array[wp.int32], out_values: wp.array[wp.Float]) -> None:
    i = wp.int32(wp.tid())
    out_values[indices[i]] = type(out_values[0])(0.0)


@wp.kernel
def hessian_energy_counts(
    vf_offsets: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    out_inverse_mass: wp.array[wp.float64],
    out_counts: wp.array[wp.int32],
) -> None:
    # Vertex ``k`` couples every ordered pair of its incident faces, 3 x 3 corners each; a killed
    # degree of freedom (boundary vertex, zeroed mass) emits nothing. The mass inversion rides in
    # the same launch: the gate reads the inverse it has just written, which is the value
    # ``hessian_energy_triplets`` gates on.
    k = wp.int32(wp.tid())
    inverse = reciprocal_or_zero(mass[k])
    out_inverse_mass[k] = inverse
    if inverse > wp.float64(0.0):
        degree = vf_offsets[k + 1] - vf_offsets[k]
        out_counts[k] = 9 * degree * degree
    else:
        out_counts[k] = 0


@wp.func
def face_corner_of_vertex(faces: wp.array[wp.int32], face: wp.int32, vertex: wp.int32) -> wp.int32:
    if faces[face * 3 + 0] == vertex:
        return wp.int32(0)
    if faces[face * 3 + 1] == vertex:
        return wp.int32(1)
    return wp.int32(2)


@wp.kernel
def hessian_energy_triplets(
    faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    gradients: wp.array[wp.vec3d],
    areas: wp.array[wp.float64],
    inv_mass: wp.array[wp.float64],
    segment_offsets: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # ``Q = H^T Minv H`` contracted analytically over the 9 component pairs of the stacked
    # Hessian: ``Q_ij = sum_k Minv_k sum_{f,g ni k} A_f A_g (g_fk . g_gk)(g_fi . g_gj)`` with
    # ``g_fc`` the hat-function gradient of corner ``c`` in face ``f``. One thread per vertex
    # ``k``; the per-thread work is quadratic in the vertex's valence.
    k = wp.int32(wp.tid())
    weight = inv_mass[k]
    # The negation of ``hessian_energy_counts``'s own gate -- see ``sandwich_row_triplets`` for why
    # this must not be an independent ``<=``.
    if not (weight > wp.float64(0.0)):
        return
    start = vf_offsets[k]
    end = vf_offsets[k + 1]
    cursor = segment_offsets[k]
    for a in range(start, end):
        f = vertex_faces[a]
        corner_f = face_corner_of_vertex(faces, f, k)
        gradient_fk = gradients[f * 3 + corner_f]
        left = weight * areas[f]
        for b in range(start, end):
            g = vertex_faces[b]
            corner_g = face_corner_of_vertex(faces, g, k)
            pair = left * areas[g] * wp.dot(gradient_fk, gradients[g * 3 + corner_g])
            for ci in range(3):
                row = faces[f * 3 + ci]
                gradient_fi = gradients[f * 3 + ci]
                for cj in range(3):
                    out_rows[cursor] = row
                    out_cols[cursor] = faces[g * 3 + cj]
                    out_vals[cursor] = type(out_vals[0])(
                        pair * wp.dot(gradient_fi, gradients[g * 3 + cj])
                    )
                    cursor += 1


@wp.kernel
def zero_at_boundary_edges(
    unique_edges: wp.array2d[wp.int32],
    halfedge_counts: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # Zero ``out_values`` at both endpoints of every unique edge with one halfedge -- every
    # boundary vertex, read off the halfedge counts ``internal_angles_and_sums`` leaves rather than
    # a second sort of the halfedges (``energies._zero_at_boundary``'s route).
    #
    # Riding this on ``cr_gradient_rows``' grid (same ``dim``, independent; a boundary edge is one
    # whose second halfedge slot is ``-1``) was measured and declined: ``curved_hessian_energy``
    # 1.00-1.01x on ``sphere_small`` / ``saddle`` / ``hemisphere`` / ``sphere_med``.
    e = wp.int32(wp.tid())
    if halfedge_counts[e] == 1:
        out_values[unique_edges[e, 0]] = wp.float64(0.0)
        out_values[unique_edges[e, 1]] = wp.float64(0.0)


@wp.kernel
def internal_angles_and_sums(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    edge_cursor: wp.array[wp.int32],
    out_angles: wp.array2d[wp.float64],
    out_angle_sums: wp.array[wp.float64],
    out_edge_halfedges: wp.array2d[wp.int32],
) -> None:
    # Interior angle at each corner (law of cosines, float64) plus the per-vertex angle sum the
    # curvature correction normalizes by. ``wp.acos`` clamps its argument, so a sliver face yields
    # 0 / pi rather than NaN.
    #
    # ``triangles.angles`` is the same quantity in float32 from normalized edge vectors, and stays a
    # separate kernel: it emits no angle sums, takes its third angle as ``PI - a0 - a1``, and zeroes
    # all three angles of a degenerate face instead of letting the acos clamp report 0 / pi.
    #
    # The face's three halfedges are also recorded against their unique edges (``inverse``), up to
    # two per edge in arbitrary order, before any exit: the topology half of
    # ``curved_hessian_energy``'s setup, riding the face pass it already pays for. The cursor counts
    # every halfedge, so that caller validates edge-manifoldness from it afterwards (a count above
    # two); ``scatter.append_to_pair``'s two-slot bound keeps a third halfedge from writing out of
    # ``out_edge_halfedges``' row width before the check raises.
    f = wp.int32(wp.tid())
    for k in range(3):
        h = 3 * f + k
        append_to_pair(inverse[h], h, edge_cursor, out_edge_halfedges)
    l2_0, l2_1, l2_2, dbl_area = triangle_geometry_f64(vertices, faces, f)
    zero = wp.float64(0.0)
    if dbl_area <= zero:
        # A coincident-vertex edge (not merely a thin sliver) drives ``corner_cosines_from_l2``'s
        # law-of-cosines division to a genuine 0/0 for the two corners touching it -- ``wp.acos``'s
        # clamp rescues a finite-but-out-of-[-1,1] cosine, not a NaN, and ``add_corner_triple``
        # would then atomically accumulate that NaN into ``out_angle_sums`` for both of the
        # degenerate edge's vertices, poisoning the Gaussian-curvature correction at every *other*,
        # perfectly valid face sharing one of them. A degenerate face contributes nothing instead,
        # the same convention ``hessian_face_terms`` already uses.
        out_angles[f, 0] = zero
        out_angles[f, 1] = zero
        out_angles[f, 2] = zero
        return
    cos0, cos1, cos2 = corner_cosines_from_l2(l2_0, l2_1, l2_2)
    theta0 = wp.acos(cos0)
    theta1 = wp.acos(cos1)
    theta2 = wp.acos(cos2)
    out_angles[f, 0] = theta0
    out_angles[f, 1] = theta1
    out_angles[f, 2] = theta2
    add_corner_triple(out_angle_sums, faces, f, theta0, theta1, theta2)


@wp.kernel
def cr_gradient_rows(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_halfedges: wp.array2d[wp.int32],
    out_vertex_slots: wp.array2d[wp.int32],
    out_par: wp.array2d[wp.float64],
    out_perp: wp.array2d[wp.float64],
) -> None:
    # One row pair of ``igl::scalar_to_cr_vector_gradient`` per unique edge: the parallel and
    # perpendicular components of the CR gradient operator, supported on at most 4 vertices (the
    # edge's endpoints in slots 0/1, each incident face's apex in slot 2/3, ``-1`` when absent).
    # The edge's positive orientation is min-vertex-first, matching ``unique_edges``' sorted rows;
    # the assembled energy is invariant to that gauge choice.
    u = wp.int32(wp.tid())
    a = unique_edges[u, 0]
    out_vertex_slots[u, 0] = a
    out_vertex_slots[u, 1] = unique_edges[u, 1]
    for t in range(2):
        h = edge_halfedges[u, t]
        if h < 0:
            continue
        f = h // 3
        s = h % 3
        i = faces[f * 3 + s]
        j = faces[f * 3 + (s + 1) % 3]
        k = faces[f * 3 + (s + 2) % 3]
        vi = to_vec3d(vertices[i])
        vj = to_vec3d(vertices[j])
        vk = to_vec3d(vertices[k])
        eij = wp.length_sq(vi - vj)
        ejk = wp.length_sq(vj - vk)
        eki = wp.length_sq(vk - vi)
        dbl_area = triangle_double_area(vi, vj, vk)
        sqrt_eij = wp.sqrt(eij)
        if dbl_area <= wp.float64(0.0) or sqrt_eij <= wp.float64(0.0):
            continue
        orientation = wp.float64(-1.0)
        slot_i = 1
        slot_j = 0
        if i == a:
            orientation = wp.float64(1.0)
            slot_i = 0
            slot_j = 1
        six_sqrt = wp.float64(6.0) * sqrt_eij
        twelve_sqrt = wp.float64(12.0) * sqrt_eij
        out_par[u, slot_i] = out_par[u, slot_i] - orientation * dbl_area / six_sqrt
        out_perp[u, slot_i] = out_perp[u, slot_i] - orientation * (eij + ejk - eki) / twelve_sqrt
        out_par[u, slot_j] = out_par[u, slot_j] + orientation * dbl_area / six_sqrt
        out_perp[u, slot_j] = out_perp[u, slot_j] - orientation * (eij - ejk + eki) / twelve_sqrt
        out_vertex_slots[u, 2 + t] = k
        out_perp[u, 2 + t] = orientation * sqrt_eij / wp.float64(6.0)


@wp.func
def select3(value0: wp.Scalar, value1: wp.Scalar, value2: wp.Scalar, index: wp.int32) -> wp.Scalar:
    # Index into three loose values. Generic over the scalar type, so the int32 edge ids and the
    # float64 matrix entries of the curved-Hessian assembly share one definition.
    if index == 0:
        return value0
    if index == 1:
        return value1
    return value2


@wp.func
def curved_pair_terms(
    eij: wp.float64,
    ejk: wp.float64,
    eki: wp.float64,
    dbl_area: wp.float64,
    o2: wp.float64,
    ki: wp.float64,
    kj: wp.float64,
    kk: wp.float64,
) -> tuple[wp.float64, wp.float64, wp.float64]:
    # Per edge-slot terms of ``igl::cr_vector_laplacian + cr_vector_curvature_correction``: the
    # diagonal weight, the parallel-parallel coupling to the previous slot's edge, and the
    # parallel-perpendicular one. ``eij``/``ejk``/``eki`` are squared lengths.
    lens = wp.sqrt(eij * eki)
    base = eij - ejk + eki
    two = wp.float64(2.0)
    diag = two * eij / dbl_area + (ki + kj + kk)
    cos_div = base / (two * lens)
    sin_div = wp.sqrt(
        wp.max(wp.float64(0.0), wp.float64(1.0) - base * base / (wp.float64(4.0) * eij * eki))
    )
    kcomb = ki - kj - kk
    off_pp = o2 * base * base / (two * lens * dbl_area) - o2 * kcomb * cos_div
    off_pq = -o2 * base / lens + o2 * kcomb * sin_div
    return diag, off_pp, off_pq


@wp.func
def halfedge_orientation(faces: wp.array[wp.int32], halfedge: wp.int32) -> wp.float64:
    # +1 when the halfedge runs from the smaller vertex index to the larger, i.e. agrees with the
    # min-first row ``edges_unique`` stores for its edge.
    origin, destination = halfedge_endpoints(faces, halfedge)
    if origin < destination:
        return wp.float64(1.0)
    return wp.float64(-1.0)


@wp.kernel
def curved_hessian_triplets(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    angles: wp.array2d[wp.float64],
    angle_sums: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    edge_vertex_slots: wp.array2d[wp.int32],
    par: wp.array2d[wp.float64],
    perp: wp.array2d[wp.float64],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # Per face, the sandwich ``D^T Mi (L + K) Mi D`` restricted to this face's 6x6 block of
    # ``L + K`` (both operators couple edges within a face only): 9 ordered edge-slot pairs, each a
    # 2x2 block over the parallel/perpendicular components, contracted against the two edges' CR
    # gradient rows (up to 4 vertices each) -> a fixed 144 triplets per face, zero-padded.
    f = wp.int32(wp.tid())
    zero = wp.float64(0.0)
    base_out = f * 144
    l2_0, l2_1, l2_2, dbl_area = triangle_geometry_f64(vertices, faces, f)
    if dbl_area <= zero:
        # Padding must be a hole, not a value: -1 is out of range for ``bsr_from_triplets``'s
        # ``(n_vertices, n_vertices)`` build and is silently dropped, where row/col 0 would collide
        # with every other degenerate face (and every absent edge-vertex slot below) on that one
        # real matrix entry and serialize the assembly's duplicate accumulation there.
        for empty in range(144):
            out_rows[base_out + empty] = -1
            out_cols[base_out + empty] = -1
            out_vals[base_out + empty] = type(out_vals[0])(0.0)
        return
    # l2_0/l2_1/l2_2: column e opposite corner e (the igl intrinsic convention).
    # Curvature ingredients per corner c: scaledKappa(F(f,c)) * theta(f,c), the scaled kappa formed
    # from the vertex's angle sum as it is read (the wrapper zeroes the boundary's sums, which
    # ``scaled_angle_defect`` maps to the zero kappa a boundary vertex takes).
    kv0 = scaled_angle_defect(angle_sums[faces[f * 3 + 0]]) * angles[f, 0]
    kv1 = scaled_angle_defect(angle_sums[faces[f * 3 + 1]]) * angles[f, 1]
    kv2 = scaled_angle_defect(angle_sums[faces[f * 3 + 2]]) * angles[f, 2]
    # igl edge slot e is ordito halfedge (e + 1) % 3; o2[e] = oE(f,e) * oE(f,(e+2)%3).
    oh0 = halfedge_orientation(faces, f * 3 + 0)
    oh1 = halfedge_orientation(faces, f * 3 + 1)
    oh2 = halfedge_orientation(faces, f * 3 + 2)
    eid0 = inverse[f * 3 + 1]
    eid1 = inverse[f * 3 + 2]
    eid2 = inverse[f * 3 + 0]
    # The three edges' inverse lumped masses, each formed once from the mass diagonal.
    im0 = reciprocal_or_zero(mass[eid0])
    im1 = reciprocal_or_zero(mass[eid1])
    im2 = reciprocal_or_zero(mass[eid2])
    diag0, pp0, pq0 = curved_pair_terms(l2_0, l2_1, l2_2, dbl_area, oh0 * oh1, kv1, kv2, kv0)
    diag1, pp1, pq1 = curved_pair_terms(l2_1, l2_2, l2_0, dbl_area, oh1 * oh2, kv2, kv0, kv1)
    diag2, pp2, pq2 = curved_pair_terms(l2_2, l2_0, l2_1, dbl_area, oh2 * oh0, kv0, kv1, kv2)
    for alpha in range(3):
        edge_a = select3(eid0, eid1, eid2, alpha)
        mi_a = select3(im0, im1, im2, alpha)
        for beta in range(3):
            edge_b = select3(eid0, eid1, eid2, beta)
            mimi = mi_a * select3(im0, im1, im2, beta)
            b_pp = zero
            b_pq = zero
            b_qp = zero
            if alpha == beta:
                b_pp = select3(diag0, diag1, diag2, alpha)
            elif beta == (alpha + 2) % 3:
                b_pp = select3(pp0, pp1, pp2, alpha)
                b_pq = select3(pq0, pq1, pq2, alpha)
                b_qp = -b_pq
            else:
                b_pp = select3(pp0, pp1, pp2, beta)
                b_qp = select3(pq0, pq1, pq2, beta)
                b_pq = -b_qp
            pair_out = base_out + (alpha * 3 + beta) * 16
            for u_slot in range(4):
                row = edge_vertex_slots[edge_a, u_slot]
                par_u = par[edge_a, u_slot]
                perp_u = perp[edge_a, u_slot]
                for v_slot in range(4):
                    col = edge_vertex_slots[edge_b, v_slot]
                    position = pair_out + u_slot * 4 + v_slot
                    value = zero
                    # An absent slot (a boundary edge's missing second incident face) pads with an
                    # out-of-range row/col rather than 0 -- see the degenerate-face branch above.
                    out_row = -1
                    out_col = -1
                    if row >= 0 and col >= 0:
                        par_v = par[edge_b, v_slot]
                        perp_v = perp[edge_b, v_slot]
                        value = mimi * (
                            b_pp * (par_u * par_v + perp_u * perp_v)
                            + b_pq * par_u * perp_v
                            + b_qp * perp_u * par_v
                        )
                        out_row = row
                        out_col = col
                    out_rows[position] = out_row
                    out_cols[position] = out_col
                    out_vals[position] = type(out_vals[0])(value)


@wp.kernel
def crouzeix_raviart_cotmatrix_triplets(
    inverse: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # 12 triplets per face: for each corner, the two edges meeting there couple with minus four
    # times the corner's half-cotangent, and each gains the same amount on its diagonal
    # (``igl::crouzeix_raviart_cotmatrix``'s triangle table). Edge ids come from
    # ``edges_unique``'s inverse; ordito halfedge slot ``s`` spans corners ``s -> s+1``, so the
    # edge opposite corner ``c`` is halfedge ``(c + 1) % 3``.
    f = wp.int32(wp.tid())
    for c in range(3):
        edge_1 = inverse[f * 3 + (c + 1) % 3]
        edge_2 = inverse[f * 3 + (c + 2) % 3]
        weight = type(out_vals[0])(4.0) * type(out_vals[0])(cot_entries[f, (c + 2) % 3])
        base = f * 12 + c * 4
        out_rows[base + 0] = edge_1
        out_cols[base + 0] = edge_2
        out_vals[base + 0] = -weight
        out_rows[base + 1] = edge_2
        out_cols[base + 1] = edge_1
        out_vals[base + 1] = -weight
        out_rows[base + 2] = edge_1
        out_cols[base + 2] = edge_1
        out_vals[base + 2] = weight
        out_rows[base + 3] = edge_2
        out_cols[base + 3] = edge_2
        out_vals[base + 3] = weight


@wp.kernel
def crouzeix_raviart_mass_diag(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_mass: wp.array[wp.Float],
) -> None:
    # Each face donates a third of its area to each of its three edges — both
    # ``igl::crouzeix_raviart_massmatrix``'s diagonal and (per parallel/perpendicular component)
    # ``igl::cr_vector_mass``'s.
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    third = type(out_mass[0])(triangle_double_area(v0, v1, v2) / wp.float64(6.0))
    add_corner_triple(out_mass, inverse, f, third, third, third)


@wp.func
def write_vector_area_triplets(
    boundary_edges: wp.array2d[wp.int32],
    n_vertices: wp.int32,
    scale: wp.float64,
    b: wp.int32,
    base: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # ``igl::vector_area_matrix``: per oriented boundary edge ``(i, j)`` emit the four
    # cross-quadrant triplets ``(i+n, j, -q)``, ``(j, i+n, -q)``, ``(i, j+n, +q)``, ``(j+n, i, +q)``
    # with ``q = 0.25 * scale``, at slots ``base .. base + 3``. ``scale = 1`` builds ``A`` itself
    # (``vector_area_triplets``); ``scale = -2`` builds the ``-2A`` term of the LSCM Hessian
    # (``lscm_hessian_triplets``).
    i = boundary_edges[b, 0]
    j = boundary_edges[b, 1]
    q = wp.float64(0.25) * scale
    out_rows[base] = i + n_vertices
    out_cols[base] = j
    out_vals[base] = -q
    out_rows[base + 1] = j
    out_cols[base + 1] = i + n_vertices
    out_vals[base + 1] = -q
    out_rows[base + 2] = i
    out_cols[base + 2] = j + n_vertices
    out_vals[base + 2] = q
    out_rows[base + 3] = j + n_vertices
    out_cols[base + 3] = i
    out_vals[base + 3] = q


@wp.kernel
def lscm_hessian_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    n_vertices: wp.int32,
    boundary_edges: wp.array2d[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # The LSCM Hessian ``-repdiag(L, 2) - 2 A`` as triplets, in one launch over
    # ``max(n_vertices, n_boundary_edges)`` threads writing disjoint slots of one buffer:
    #
    # - ``-repdiag(L, 2)``, the block-diagonal ``[[-L, 0], [0, -L]]`` (2n x 2n): thread ``t`` below
    #   ``n_vertices`` takes CSR row ``t`` of ``L``, and each entry ``e`` emits both diagonal-block
    #   copies into slots ``2*e`` (upper block) and ``2*e + 1`` (lower block, shifted by
    #   ``n_vertices``);
    # - ``-2 A``: thread ``t`` below the boundary-edge count writes edge ``t``'s four vector-area
    #   triplets after them, at ``2 * offsets[n_vertices] + 4 t``.
    t = wp.int32(wp.tid())
    if t < n_vertices:
        for e in range(offsets[t], offsets[t + 1]):
            j = columns[e]
            v = -values[e]
            out_rows[2 * e] = t
            out_cols[2 * e] = j
            out_vals[2 * e] = v
            out_rows[2 * e + 1] = t + n_vertices
            out_cols[2 * e + 1] = j + n_vertices
            out_vals[2 * e + 1] = v
    if t < boundary_edges.shape[0]:
        base = 2 * offsets[n_vertices] + 4 * t
        write_vector_area_triplets(
            boundary_edges, n_vertices, wp.float64(-2.0), t, base, out_rows, out_cols, out_vals
        )


@wp.kernel
def vector_area_triplets(
    boundary_edges: wp.array2d[wp.int32],
    n_vertices: wp.int32,
    scale: wp.float64,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # ``write_vector_area_triplets`` for every oriented boundary edge, slot base ``4 * b``.
    b = wp.int32(wp.tid())
    write_vector_area_triplets(
        boundary_edges, n_vertices, scale, b, 4 * b, out_rows, out_cols, out_vals
    )


# Concrete overloads, registered at import -- rationale in ``ordito/kernels/reduce.py``. Measured
# over the suite: 8 overloads created across **9** module loads.
#
# Every generic argument here is the assembled matrix's value precision, which ``ordito.energies``
# exposes as the same public ``dtype`` keyword ``ordito.laplacian`` does. The float64 inputs the
# curvature and Hessian kernels read (angles, kappa, inverse masses) are fixed at float64 by their
# producers and are not part of the template.
_MATRIX_DTYPES = (wp.float32, wp.float64)


# The concrete handles keyed by the caller's matrix dtype -- see
# [`OverloadTable`][ordito.kernels.array.OverloadTable].
# ``CROUZEIX_RAVIART_COTMATRIX_TRIPLETS`` keys on the pair ``(entry dtype, matrix dtype)`` for the
# reason its registration already gave: the two templates are independent.
ZERO_AT_INDICES: OverloadTable
CROUZEIX_RAVIART_MASS_DIAG: OverloadTable
CROUZEIX_RAVIART_COTMATRIX_TRIPLETS: OverloadTable
SANDWICH_ROW_COUNTS: OverloadTable
SANDWICH_ROW_TRIPLETS: OverloadTable
HESSIAN_ENERGY_TRIPLETS: OverloadTable
CURVED_HESSIAN_TRIPLETS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global ZERO_AT_INDICES, CROUZEIX_RAVIART_MASS_DIAG, CROUZEIX_RAVIART_COTMATRIX_TRIPLETS
    global SANDWICH_ROW_COUNTS, SANDWICH_ROW_TRIPLETS
    global HESSIAN_ENERGY_TRIPLETS, CURVED_HESSIAN_TRIPLETS
    i32 = wp.array[wp.int32]
    # ``_zero_at_boundary``'s only caller, ``float64`` in and out (``mass``/``kappa`` are fixed at
    # that precision by their own producers) -- one dtype, not the full ``_MATRIX_DTYPES`` cross
    # product this kernel's ``wp.Float`` genericity would otherwise admit.
    ZERO_AT_INDICES = OverloadTable(zero_at_indices, {wp.float64: [i32, wp.array[wp.float64]]})
    CROUZEIX_RAVIART_MASS_DIAG = OverloadTable(
        crouzeix_raviart_mass_diag,
        {d: [wp.array[wp.vec3], i32, i32, wp.array[d]] for d in _MATRIX_DTYPES},
    )
    # ``cot_entries`` and the matrix precision are *independent* templates:
    # ``crouzeix_raviart_cotmatrix`` takes ``cot_entries`` as ``odt.Array2dFloat`` beside a
    # separate ``dtype`` keyword, and the kernel casts the entries to the matrix precision, so
    # this is a genuine 2x2 rather than a diagonal. Registering only the diagonal made the first
    # float64-entries/float32-matrix launch recompile this whole module, taking over a minute and
    # returning the right answer -- the silent cost of a lazily instantiated overload.
    CROUZEIX_RAVIART_COTMATRIX_TRIPLETS = OverloadTable(
        crouzeix_raviart_cotmatrix_triplets,
        {
            (entry_dtype, dtype): [i32, wp.array2d[entry_dtype], i32, i32, wp.array[dtype]]
            for dtype in _MATRIX_DTYPES
            for entry_dtype in _MATRIX_DTYPES
        },
    )
    SANDWICH_ROW_COUNTS = OverloadTable(
        sandwich_row_counts, {d: [i32, i32, wp.array[d], i32] for d in _MATRIX_DTYPES}
    )
    SANDWICH_ROW_TRIPLETS = OverloadTable(
        sandwich_row_triplets,
        {
            d: [
                i32,
                i32,
                wp.array[d],
                i32,
                i32,
                wp.array[d],
                wp.array[d],
                i32,
                i32,
                i32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    HESSIAN_ENERGY_TRIPLETS = OverloadTable(
        hessian_energy_triplets,
        {
            d: [
                i32,
                i32,
                i32,
                wp.array[wp.vec3d],
                wp.array[wp.float64],
                wp.array[wp.float64],
                i32,
                i32,
                i32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    CURVED_HESSIAN_TRIPLETS = OverloadTable(
        curved_hessian_triplets,
        {
            d: [
                wp.array[wp.vec3],
                i32,
                i32,
                wp.array2d[wp.float64],
                wp.array[wp.float64],
                wp.array[wp.float64],
                wp.array2d[wp.int32],
                wp.array2d[wp.float64],
                wp.array2d[wp.float64],
                i32,
                i32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )


_register_overloads()
