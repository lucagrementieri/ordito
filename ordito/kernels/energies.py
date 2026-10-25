import warp as wp

from ordito.kernels.array import OverloadTable, store_coo_key, to_vec3d
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.laplacian import cotmatrix_row, store_half_cotangents
from ordito.kernels.predicates import (
    angle_defect,
    corner_cosines_from_l2,
    squared_edge_lengths,
    triangle_double_area,
)
from ordito.kernels.reduce import term_sum_1d
from ordito.kernels.scatter import add_corner_triple, add_or_store_corner_triple, append_to_pair
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
def reciprocal_scaled_or_zero(value: wp.Float, numerator: wp.Float) -> wp.Float:
    # ``numerator / value``, with the ``reciprocal_or_zero`` convention above for a non-positive
    # denominator: a vertex whose lumped area is zero contributes nothing rather than an infinity.
    # Called by ``smoothing_norm`` and by ``scaled_angle_defect`` below.
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


# ``smoothing_norm``'s warp-uniform method selector, one per
# ``energies.laplacian_smoothing_loss`` method.
SMOOTHING_UNIFORM = wp.constant(0)
SMOOTHING_COT = wp.constant(1)
SMOOTHING_COTCURV = wp.constant(2)


@wp.struct
class EdgeDeviationTerms:
    """``edge_length_loss``'s edge count, edge lengths and resting length ``L0``."""

    n: wp.int32
    lengths: wp.array[wp.float32]
    target: wp.float32


@wp.func
def edge_deviation_term(local: wp.int32, offset: wp.int32, terms: EdgeDeviationTerms) -> wp.float32:
    # ``(x - L0)^2`` for edge ``offset + local``, the per-edge term of the edge-length regularizer.
    difference = terms.lengths[offset + local] - terms.target
    return difference * difference


@wp.struct
class DihedralTerms:
    """``normal_consistency_loss``'s pair count and dihedral angles, one per adjacent face pair."""

    n: wp.int32
    angles: wp.array[wp.float32]


@wp.func
def dihedral_term(local: wp.int32, offset: wp.int32, terms: DihedralTerms) -> wp.float32:
    # ``1 - cos(theta)`` for pair ``offset + local``, the per-pair term of the normal-consistency
    # regularizer: 0 for a flat pair, 1 at a right angle, 2 for a fold back on itself.
    return wp.float32(1.0) - wp.cos(terms.angles[offset + local])


@wp.func
def smoothing_norm(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    vertices: wp.array[wp.vec3],
    area_sums: wp.array[wp.float64],
    method: wp.int32,
    i: wp.int32,
) -> wp.float32:
    # ``|| row_scale * (L v)_i + self_scale * v_i ||`` for CSR row ``i``, with ``method``'s two
    # scales (``smoothing_norm_term`` explains them): the row walk and tail both of its
    # kernels share, which differ only by who writes the row.
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
        mass = wp.float32(area_sums[i] / wp.float64(3.0))
        row_scale = reciprocal_scaled_or_zero(mass, wp.float32(1.0 / 6.0))
        self_scale = wp.float32(0.0)
    return wp.length(row_scale * accumulator + self_scale * vertices[i])


@wp.struct
class SmoothingNormTerms:
    """``laplacian_smoothing_loss``' row count, stored Laplacian rows, positions and method."""

    n: wp.int32
    offsets: wp.array[wp.int32]
    columns: wp.array[wp.int32]
    values: wp.array[wp.float32]
    vertices: wp.array[wp.vec3]
    area_sums: wp.array[wp.float64]
    method: wp.int32


@wp.func
def smoothing_norm_term(local: wp.int32, offset: wp.int32, terms: SmoothingNormTerms) -> wp.float32:
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
    # - ``SMOOTHING_COTCURV``: the stiffness residual over six times the lumped area ``mass_i``,
    #   with ``reciprocal_scaled_or_zero``'s convention that a zero-area vertex contributes nothing.
    #   ``mass_i`` is ``laplacian.mass_matrix_entries``' ``float32`` entry, formed here from the
    #   vertex's exact ``area_sums`` (``kernels/scatter.scatter_face_areas_exact``) by the one
    #   division and rounding ``kernels/scatter.scale_face_sums`` makes.
    #
    # ``area_sums`` is read only by the last; the others pass a null array.
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
    #
    # Summed where it is formed (``SMOOTHING_NORM_SUM``), so no per-row norm is stored.
    return smoothing_norm(
        terms.offsets,
        terms.columns,
        terms.values,
        terms.vertices,
        terms.area_sums,
        terms.method,
        offset + local,
    )


@wp.kernel(enable_backward=False)
def laplacian_smoothing_cot_norms(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    vertices: wp.array[wp.vec3],
    area_sums: wp.array[wp.float64],
    method: wp.int32,
    out_values: wp.array[wp.float32],
    out_norms: wp.array[wp.float32],
) -> None:
    # ``smoothing_norm_term`` for the two cotangent methods, with the ``float32`` cotangent
    # Laplacian's row written by the thread that reads it back (``kernels/laplacian.cotmatrix_row``
    # over its mesh operator pattern), so the row's values are ``cotmatrix``'s and the residual is
    # the stored-matrix kernel's, for one launch. Forming each entry again in the residual walk
    # instead of storing it cost 1.4-1.5x the two kernels' device time on scrambled meshes.
    # Folding the norms' mean into this pass as a ``term_sum_1d`` (as the uniform method does) was
    # measured and declined: 1.14x this kernel and the sum's device time at 655 k scrambled
    # vertices and 7x at 10 k, where every lane walks the last chunk's rows serially.
    i = wp.int32(wp.tid())
    cotmatrix_row(i, offsets, columns, run_start, keys, count, order, cot_entries, out_values)
    out_norms[i] = smoothing_norm(offsets, columns, out_values, vertices, area_sums, method, i)


# One ``kernels/reduce.TermSum`` per regularizer: the fused map and fixed-order (or atomic) block
# sum that ``energies._mean_of_terms`` hands to ``reduce.sum_of_terms``.
EDGE_DEVIATION_SUM = term_sum_1d(edge_deviation_term, EdgeDeviationTerms, "edge_deviation_sum")
DIHEDRAL_SUM = term_sum_1d(dihedral_term, DihedralTerms, "dihedral_sum")
SMOOTHING_NORM_SUM = term_sum_1d(smoothing_norm_term, SmoothingNormTerms, "smoothing_norm_sum")


@wp.kernel
def sandwich_row_counts(
    a_offsets: wp.array[wp.int32],
    b_offsets: wp.array[wp.int32],
    weights: wp.array[wp.Float],
    invert: wp.int32,
    out_counts: wp.array[wp.int32],
    out_inv_mass: wp.array[wp.Float],
) -> None:
    # Triplets emitted by row ``t`` of the product ``A diag(inv_mass) B``: the full outer product
    # of the two CSR rows, or nothing when the diagonal weight is zero. With ``invert`` set,
    # ``weights`` is the mass and the row forms its own ``reciprocal_or_zero`` weight, storing it
    # in ``out_inv_mass`` for the emission pass and every later sandwich; otherwise ``weights`` is
    # already ``inv_mass`` and ``out_inv_mass`` is not written.
    t = wp.int32(wp.tid())
    weight = weights[t]
    if invert != 0:
        weight = reciprocal_or_zero(weight)
        out_inv_mass[t] = weight
    if weight > type(weight)(0.0):
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
    sign: wp.Float,
    segment_offsets: wp.array[wp.int32],
    n_rows: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # ``(A diag(d) B)_ij = sum_t A_ti d_t B_tj``; with both operands symmetric ``A_ti`` is row
    # ``t``'s entry at column ``i``, so each row of the diagonal sandwich is the scaled outer
    # product of the two matching CSR rows, assembling the product without ``bsr_mm``. (The
    # ``bsr_mm`` nondeterminism this originally avoided was a triplet-capacity defect of this
    # package, not a Warp bug — see ``k_harmonic`` in ``ordito/energies.py``.) ``sign`` (``1`` or
    # ``-1``) multiplies each ``B`` entry exactly, so ``k_harmonic`` passes ``L`` where its product
    # wants ``-L`` and keeps no negated copy.
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
            store_coo_key(cursor, row, b_columns[b], n_rows, n_rows, out_keys, out_order)
            out_vals[cursor] = left * (sign * b_values[b])
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
    out_corner_mass: wp.array2d[wp.float64],
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
    # launch. The corner shares go to exactly one of two outputs, whichever the caller sized: added
    # atomically into ``out_mass``, or written per corner into ``out_corner_mass`` for the caller
    # to sum in a fixed order when ``ordito.config`` asks for reproducible results.
    corners = out_corner_mass.shape[0] > 0
    if corners:
        out_corner_mass[f, 0] = zero
        out_corner_mass[f, 1] = zero
        out_corner_mass[f, 2] = zero
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
    add_or_store_corner_triple(out_mass, out_corner_mass, faces, f, quad0, quad1, quad2)


@wp.func
def corner_multiplicity(faces: wp.array[wp.int32], face: wp.int32, vertex: wp.int32) -> wp.int32:
    # How many of ``face``'s corners are ``vertex``: how many times a vertex-face row lists it.
    r = wp.int32(0)
    for k in range(3):
        if faces[3 * face + k] == vertex:
            r += 1
    return r


@wp.func
def row_edge_weight(
    faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    vertex: wp.int32,
    lo: wp.int32,
    hi: wp.int32,
) -> wp.int32:
    # Six times the number of halfedges spanning the undirected edge ``(lo, hi)``, counted over
    # ``vertex``'s vertex-face row, which holds every face with a halfedge at ``vertex``. A face
    # the row lists once per corner at ``vertex`` weighs ``6 / multiplicity`` a listing, so each
    # face counts once in whole numbers.
    weight = wp.int32(0)
    for t in range(vf_offsets[vertex], vf_offsets[vertex + 1]):
        g = vf_indices[t]
        matches = wp.int32(0)
        for q in range(3):
            c = faces[3 * g + q]
            d = faces[3 * g + (q + 1) % 3]
            if wp.min(c, d) == lo and wp.max(c, d) == hi:
                matches += 1
        if matches > 0:
            weight += matches * (6 // corner_multiplicity(faces, g, vertex))
    return weight


@wp.func
def row_vertex_on_boundary(
    faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    vertex: wp.int32,
) -> wp.bool:
    # Whether a halfedge at ``vertex`` spans its undirected edge alone -- the boundary code of
    # ``halfedge.halfedge_mates``, the rule ``kernels/boundary.zero_at_boundary_vertices`` zeroes
    # by -- read off ``vertex``'s own vertex-face row instead of a mesh-wide pairing: every
    # halfedge on an edge at ``vertex`` belongs to a face in that row. Quadratic in the row's
    # length, as the Hessian rows it gates are.
    for s in range(vf_offsets[vertex], vf_offsets[vertex + 1]):
        f = vf_indices[s]
        for k in range(3):
            a = faces[3 * f + k]
            b = faces[3 * f + (k + 1) % 3]
            if a == vertex or b == vertex:
                weight = row_edge_weight(
                    faces, vf_offsets, vf_indices, vertex, wp.min(a, b), wp.max(a, b)
                )
                if weight == 6:
                    return True
    return False


@wp.kernel
def hessian_energy_counts(
    faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vf_indices: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    out_inverse_mass: wp.array[wp.float64],
    out_counts: wp.array[wp.int32],
) -> None:
    # Vertex ``k`` couples every ordered pair of its incident faces, 3 x 3 corners each; a killed
    # degree of freedom (boundary vertex, zeroed mass) emits nothing. The mass is defined only at
    # interior vertices, so a boundary vertex (``row_vertex_on_boundary``) inverts as a zero mass.
    # The mass inversion rides in the same launch: the gate reads the inverse it has just written,
    # which is the value ``hessian_energy_triplets`` gates on.
    k = wp.int32(wp.tid())
    inverse = wp.float64(0.0)
    if not row_vertex_on_boundary(faces, vf_offsets, vf_indices, k):
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
    n_vertices: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
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
                    store_coo_key(
                        cursor, row, faces[g * 3 + cj], n_vertices, n_vertices, out_keys, out_order
                    )
                    out_vals[cursor] = type(out_vals[0])(
                        pair * wp.dot(gradient_fi, gradients[g * 3 + cj])
                    )
                    cursor += 1


@wp.kernel
def zero_at_boundary_edges(
    unique_edges: wp.array2d[wp.int32],
    halfedge_counts: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
    out_nonmanifold: wp.array[wp.int32],
) -> None:
    # Zero ``out_values`` at both endpoints of every unique edge with one halfedge -- every
    # boundary vertex, read off the halfedge counts ``internal_angles_and_sums`` leaves rather
    # than a pass over the halfedge mates. An edge
    # with three or more halfedges sets ``out_nonmanifold[0]``, the edge-manifold validation
    # ``curved_hessian_energy`` raises on, read off the same counts in the same pass.
    #
    # Riding this on ``cr_gradient_rows``' grid (same ``dim``, independent; a boundary edge is one
    # whose second halfedge slot is ``-1``) was measured and declined: ``curved_hessian_energy``
    # 1.00-1.01x on ``sphere_small`` / ``saddle`` / ``hemisphere`` / ``sphere_med``.
    e = wp.int32(wp.tid())
    count = halfedge_counts[e]
    if count == 1:
        out_values[unique_edges[e, 0]] = wp.float64(0.0)
        out_values[unique_edges[e, 1]] = wp.float64(0.0)
    elif count > 2:
        out_nonmanifold[0] = 1


@wp.func
def cr_face_mass_third(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> wp.float64:
    # The third of face ``f``'s area it gives each of its edges' lumped Crouzeix-Raviart mass: the
    # one per-face term of ``crouzeix_raviart_mass_diag`` and of ``internal_angles_and_sums``' mass,
    # so ``crouzeix_raviart_massmatrix`` and ``curved_hessian_energy`` share one mass.
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    return triangle_double_area(v0, v1, v2) / wp.float64(6.0)


@wp.kernel
def internal_angles_and_sums(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    edge_cursor: wp.array[wp.int32],
    out_angles: wp.array2d[wp.float64],
    out_angle_sums: wp.array[wp.float64],
    out_edge_halfedges: wp.array2d[wp.int32],
    out_edge_mass: wp.array[wp.float64],
) -> None:
    # Interior angle at each corner (law of cosines, float64), plus the per-vertex angle sum the
    # curvature correction normalizes by, added atomically into ``out_angle_sums`` when the caller
    # sized it; an empty one leaves the sum to the caller, which forms it from ``out_angles`` in a
    # fixed order when ``ordito.config`` asks for reproducible results. ``wp.acos`` clamps its
    # argument, so a sliver face yields 0 / pi rather than NaN.
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
    # ``out_edge_halfedges``' row width before the check raises. The lumped Crouzeix-Raviart edge
    # mass rides the same pass (``cr_face_mass_third``, the scatter of
    # ``crouzeix_raviart_mass_diag``) into a zeroed ``out_edge_mass``.
    f = wp.int32(wp.tid())
    for k in range(3):
        h = 3 * f + k
        append_to_pair(inverse[h], h, edge_cursor, out_edge_halfedges)
    third = cr_face_mass_third(vertices, faces, f)
    add_corner_triple(out_edge_mass, inverse, f, third, third, third)
    l2_0, l2_1, l2_2, dbl_area = triangle_geometry_f64(vertices, faces, f)
    zero = wp.float64(0.0)
    if dbl_area <= zero:
        # A coincident-vertex edge (not merely a thin sliver) drives ``corner_cosines_from_l2``'s
        # law-of-cosines division to a genuine 0/0 for the two corners touching it -- ``wp.acos``'s
        # clamp rescues a finite-but-out-of-[-1,1] cosine, not a NaN, and the per-vertex angle sum
        # would then accumulate that NaN for both of the degenerate edge's vertices, poisoning the
        # Gaussian-curvature correction at every *other*, perfectly valid face sharing one of them.
        # A degenerate face contributes nothing instead, the same convention
        # ``hessian_face_terms`` already uses.
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
    if out_angle_sums.shape[0] > 0:
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
    # the assembled energy is invariant to that gauge choice. The thread clears its own row first,
    # so the three tables need no fill.
    u = wp.int32(wp.tid())
    a = unique_edges[u, 0]
    out_vertex_slots[u, 0] = a
    out_vertex_slots[u, 1] = unique_edges[u, 1]
    out_vertex_slots[u, 2] = -1
    out_vertex_slots[u, 3] = -1
    for slot in range(4):
        out_par[u, slot] = wp.float64(0.0)
        out_perp[u, slot] = wp.float64(0.0)
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
    n_vertices: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
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
            store_coo_key(base_out + empty, -1, -1, n_vertices, n_vertices, out_keys, out_order)
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
                    store_coo_key(
                        position, out_row, out_col, n_vertices, n_vertices, out_keys, out_order
                    )
                    out_vals[position] = type(out_vals[0])(value)


# The "no further column" sentinel of ``crouzeix_raviart_next_column``.
CR_NO_COLUMN = wp.int32(2147483647)


@wp.kernel
def crouzeix_raviart_face_terms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    write_cot: wp.int32,
    out_face_counts: wp.array[wp.int32],
    out_edge_faces: wp.array[wp.int32],
    out_nonmanifold: wp.array[wp.int32],
    out_cot: wp.array2d[wp.float32],
) -> None:
    # One pass over the faces that both tests edge-manifoldness and records each edge's (at most
    # two) incident faces: each corner ``k`` claims a slot of its edge, slots ``2 e`` and
    # ``2 e + 1`` hold the faces, and a third claim flags the mesh. A degenerate face naming one
    # edge twice claims two slots with the same face. With ``write_cot`` set the same pass writes
    # the face's ``float32`` half-cotangents (``laplacian.cotmatrix_entries``' row), which
    # ``crouzeix_raviart_entry`` casts to the matrix precision as it reads them.
    f = wp.int32(wp.tid())
    for c in range(3):
        k = 3 * f + c
        e = inverse[k]
        slot = wp.atomic_add(out_face_counts, e, 1)
        if slot < 2:
            out_edge_faces[2 * e + slot] = f
        else:
            out_nonmanifold[0] = 1
    if write_cot != 0:
        store_half_cotangents(vertices, faces, f, out_cot)


@wp.func
def crouzeix_raviart_row_faces(
    edge_faces: wp.array[wp.int32], e: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Edge ``e``'s distinct incident faces in ascending order (the order the triplet build summed
    # them in), an absent second one as ``-1``; the slots' own order is the atomics' and varies.
    a = edge_faces[2 * e]
    b = edge_faces[2 * e + 1]
    if b < 0 or b == a:
        return a, -1
    return wp.min(a, b), wp.max(a, b)


@wp.func
def crouzeix_raviart_next_column(
    inverse: wp.array[wp.int32], f0: wp.int32, f1: wp.int32, e: wp.int32, previous: wp.int32
) -> wp.int32:
    # The smallest column of row ``e`` above ``previous``, or ``CR_NO_COLUMN``. Row ``e``'s columns
    # are exactly ``e`` (the diagonal) and the edges of its incident faces: a face's corners pair
    # every two of its edge slots, so each other edge couples with ``e``.
    best = wp.where(e > previous, e, CR_NO_COLUMN)
    for k in range(2):
        f = wp.where(k == 0, f0, f1)
        if f >= 0:
            for j in range(3):
                col = inverse[f * 3 + j]
                if col > previous and col < best:
                    best = col
    return best


@wp.func
def crouzeix_raviart_entry(
    inverse: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    f0: wp.int32,
    f1: wp.int32,
    e: wp.int32,
    col: wp.int32,
    zero: wp.Float,
) -> wp.Float:
    # Entry ``(e, col)``: at each corner the two edges meeting there couple with minus four times
    # the corner's half-cotangent, and each gains the same amount on its diagonal
    # (``igl::crouzeix_raviart_cotmatrix``'s triangle table). Contributions are summed in triplet
    # order -- ascending face, corner, then ``(e1, e2)``, ``(e2, e1)``, ``(e1, e1)``, ``(e2, e2)``
    # -- starting from the first, the summation a ``csr_from_triplets`` build of those triplets
    # performs. Ordito halfedge slot ``s`` spans corners ``s -> s+1``, so the edge opposite corner
    # ``c`` is halfedge ``(c + 1) % 3``.
    acc = zero
    started = wp.int32(0)
    for k in range(2):
        f = wp.where(k == 0, f0, f1)
        if f >= 0:
            for c in range(3):
                edge_1 = inverse[f * 3 + (c + 1) % 3]
                edge_2 = inverse[f * 3 + (c + 2) % 3]
                weight = type(zero)(4.0) * type(zero)(cot_entries[f, (c + 2) % 3])
                if edge_1 == e and edge_2 == col:
                    acc = wp.where(started != 0, acc - weight, -weight)
                    started = 1
                if edge_2 == e and edge_1 == col:
                    acc = wp.where(started != 0, acc - weight, -weight)
                    started = 1
                if edge_1 == e and edge_1 == col:
                    acc = wp.where(started != 0, acc + weight, weight)
                    started = 1
                if edge_2 == e and edge_2 == col:
                    acc = wp.where(started != 0, acc + weight, weight)
                    started = 1
    return acc


@wp.kernel
def crouzeix_raviart_row_counts(
    inverse: wp.array[wp.int32], edge_faces: wp.array[wp.int32], out_offsets: wp.array[wp.int32]
) -> None:
    # Row ``e``'s column count into ``out_offsets[e + 1]``; an in-place scan makes the offsets.
    # ``crouzeix_raviart_csr`` walks the same columns with the same helper.
    e = wp.int32(wp.tid())
    f0, f1 = crouzeix_raviart_row_faces(edge_faces, e)
    count = wp.int32(0)
    col = crouzeix_raviart_next_column(inverse, f0, f1, e, -1)
    while col != CR_NO_COLUMN:
        count += 1
        col = crouzeix_raviart_next_column(inverse, f0, f1, e, col)
    out_offsets[e + 1] = count


@wp.kernel
def crouzeix_raviart_csr(
    inverse: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    edge_faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.Float],
) -> None:
    # Row ``e`` of the Crouzeix-Raviart stiffness matrix written in place, columns ascending: the
    # pattern follows from the edge-face incidence, so no triplet is sorted.
    e = wp.int32(wp.tid())
    f0, f1 = crouzeix_raviart_row_faces(edge_faces, e)
    slot = offsets[e]
    zero = type(out_values[0])(0.0)
    col = crouzeix_raviart_next_column(inverse, f0, f1, e, -1)
    while col != CR_NO_COLUMN:
        out_columns[slot] = col
        out_values[slot] = crouzeix_raviart_entry(inverse, cot_entries, f0, f1, e, col, zero)
        slot += 1
        col = crouzeix_raviart_next_column(inverse, f0, f1, e, col)


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
    third = type(out_mass[0])(cr_face_mass_third(vertices, faces, f))
    add_corner_triple(out_mass, inverse, f, third, third, third)


@wp.func
def lscm_boundary_side(
    faces: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    n_keys: wp.int32,
    order: wp.array[wp.int32],
    r: wp.int32,
    c: wp.int32,
    e: wp.int32,
) -> wp.int32:
    # Which side of a boundary edge vertex ``r`` is, for the ``cotmatrix`` pattern entry ``e`` at
    # ``(r, c)``: ``1`` when the edge is a boundary edge running ``r -> c`` in its face's winding,
    # ``-1`` when it runs ``c -> r``, ``0`` when it is not a boundary edge (or ``e`` is the
    # diagonal). An entry's run of equal sorted keys holds one corner per halfedge on the edge in
    # either build of the pattern, so a run of one is ``halfedge_mates``' lone halfedge -- the
    # boundary edge ``oriented_boundary_edges`` emits -- and its corner ``3 * f + k`` names the
    # face whose winding runs from corner ``k + 1`` to ``k + 2``.
    if c == r:
        return 0
    p = run_start[e]
    if p + 1 < n_keys:
        if keys[p + 1] == keys[p]:
            return 0
    corner = order[p]
    tail = faces[(corner // 3) * 3 + (corner % 3 + 1) % 3]
    return wp.where(tail == r, 1, -1)


@wp.kernel
def lscm_laplacian_rows_and_counts(
    faces: wp.array[wp.int32],
    laplacian_offsets: wp.array[wp.int32],
    laplacian_columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    n_keys: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    out_laplacian_values: wp.array[wp.float64],
    out_counts: wp.array[wp.int32],
) -> None:
    # One thread per vertex over ``L``'s pattern: row ``r`` of the ``float64`` cotangent Laplacian
    # (``kernels/laplacian.cotmatrix_row``, as ``cotmatrix_rows`` writes it), and the vertex's
    # boundary edge count into slot ``r + 1`` (thread 0 also writes the leading zero), whose
    # inclusive scan places every row's ``-2 A`` entries in ``lscm_hessian_csr``.
    r = wp.int32(wp.tid())
    cotmatrix_row(
        r,
        laplacian_offsets,
        laplacian_columns,
        run_start,
        keys,
        n_keys,
        order,
        cot_entries,
        out_laplacian_values,
    )
    if r == 0:
        out_counts[0] = 0
    count = wp.int32(0)
    for e in range(laplacian_offsets[r], laplacian_offsets[r + 1]):
        side = lscm_boundary_side(faces, run_start, keys, n_keys, order, r, laplacian_columns[e], e)
        if side != 0:
            count += 1
    out_counts[r + 1] = count


@wp.kernel
def lscm_hessian_csr(
    faces: wp.array[wp.int32],
    laplacian_offsets: wp.array[wp.int32],
    laplacian_columns: wp.array[wp.int32],
    laplacian_values: wp.array[wp.float64],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    n_keys: wp.int32,
    order: wp.array[wp.int32],
    area_offsets: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # The LSCM Hessian ``-repdiag(L, 2) + (-2 A)`` written straight into CSR, one thread per row
    # of the ``2n`` rows, with ``A`` (``igl::vector_area_matrix``) read off the pattern of ``L``:
    # an oriented boundary edge ``(i, j)`` puts ``-2 A``'s ``-1/2`` at ``(i, j + n)`` and
    # ``(j + n, i)`` and ``+1/2`` at ``(j, i + n)`` and ``(i + n, j)``, so an upper row ``r``
    # holds ``-1/2`` at column ``c + n`` for each boundary neighbour ``c`` it leaves
    # (``lscm_boundary_side`` 1) and ``+1/2`` for each it enters, and the lower row ``r + n`` the
    # negation at column ``c``. ``vector_area_triplets`` writes the same entries as triplets for
    # ``vector_area_matrix``. The two terms never share an entry: ``-2 A`` holds only
    # cross-quadrant entries, so an upper row ``r`` is ``-L``'s row ``r`` (columns below ``n``)
    # followed by ``-2 A``'s (columns from ``n``), and a lower row ``r + n`` is ``-2 A``'s row
    # (columns below ``n``) followed by ``-L``'s row ``r`` shifted by ``n``; both walk ``L``'s
    # sorted row, so each row stays sorted. ``area_offsets`` is the scan of
    # ``lscm_laplacian_rows_and_counts``: a row's start is ``L.offsets[r] + area_offsets[r]``
    # above and ``L.offsets[n] + area_offsets[n] + L.offsets[r - n] + area_offsets[r - n]`` below.
    # Thread 0 also writes the leading zero.
    r = wp.int32(wp.tid())
    vertex = r
    shift = wp.int32(0)
    base = wp.int32(0)
    sign = wp.float64(-0.5)
    if r >= n_vertices:
        vertex = r - n_vertices
        shift = n_vertices
        base = laplacian_offsets[n_vertices] + area_offsets[n_vertices]
        sign = wp.float64(0.5)
    lap_begin = laplacian_offsets[vertex]
    lap_end = laplacian_offsets[vertex + 1]
    if r == 0:
        out_offsets[0] = 0
    out_offsets[r + 1] = base + lap_end + area_offsets[vertex + 1]
    slot = base + lap_begin + area_offsets[vertex]
    if r >= n_vertices:
        for e in range(lap_begin, lap_end):
            c = laplacian_columns[e]
            side = lscm_boundary_side(faces, run_start, keys, n_keys, order, vertex, c, e)
            if side != 0:
                out_columns[slot] = c
                out_values[slot] = wp.where(side > 0, sign, -sign)
                slot += 1
    for e in range(lap_begin, lap_end):
        out_columns[slot] = laplacian_columns[e] + shift
        out_values[slot] = -laplacian_values[e]
        slot += 1
    if r < n_vertices:
        for e in range(lap_begin, lap_end):
            c = laplacian_columns[e]
            side = lscm_boundary_side(faces, run_start, keys, n_keys, order, vertex, c, e)
            if side != 0:
                out_columns[slot] = c + n_vertices
                out_values[slot] = wp.where(side > 0, sign, -sign)
                slot += 1


@wp.kernel
def vector_area_triplets(
    boundary_edges: wp.array2d[wp.int32],
    n_vertices: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # ``igl::vector_area_matrix``: per oriented boundary edge ``(i, j)`` the four cross-quadrant
    # triplets ``(i+n, j, -1/4)``, ``(j, i+n, -1/4)``, ``(i, j+n, +1/4)``, ``(j+n, i, +1/4)`` at
    # slots ``4 * b .. 4 * b + 3``. ``lscm_hessian_csr`` writes ``-2`` times the same entries
    # straight into its rows.
    b = wp.int32(wp.tid())
    i = boundary_edges[b, 0]
    j = boundary_edges[b, 1]
    q = wp.float64(0.25)
    base = 4 * b
    size = 2 * n_vertices
    store_coo_key(base, i + n_vertices, j, size, size, out_keys, out_order)
    out_vals[base] = -q
    store_coo_key(base + 1, j, i + n_vertices, size, size, out_keys, out_order)
    out_vals[base + 1] = -q
    store_coo_key(base + 2, i, j + n_vertices, size, size, out_keys, out_order)
    out_vals[base + 2] = q
    store_coo_key(base + 3, j + n_vertices, i, size, size, out_keys, out_order)
    out_vals[base + 3] = q


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
# ``CROUZEIX_RAVIART_CSR`` keys on the pair ``(entry dtype, matrix dtype)`` for the
# reason its registration already gave: the two templates are independent.
CROUZEIX_RAVIART_MASS_DIAG: OverloadTable
CROUZEIX_RAVIART_CSR: OverloadTable
SANDWICH_ROW_COUNTS: OverloadTable
SANDWICH_ROW_TRIPLETS: OverloadTable
HESSIAN_ENERGY_TRIPLETS: OverloadTable
CURVED_HESSIAN_TRIPLETS: OverloadTable
LSCM_LAPLACIAN_ROWS_AND_COUNTS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global CROUZEIX_RAVIART_MASS_DIAG, CROUZEIX_RAVIART_CSR
    global SANDWICH_ROW_COUNTS, SANDWICH_ROW_TRIPLETS
    global HESSIAN_ENERGY_TRIPLETS, CURVED_HESSIAN_TRIPLETS, LSCM_LAPLACIAN_ROWS_AND_COUNTS
    i32 = wp.array[wp.int32]
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
    CROUZEIX_RAVIART_CSR = OverloadTable(
        crouzeix_raviart_csr,
        {
            (entry_dtype, dtype): [i32, wp.array2d[entry_dtype], i32, i32, i32, wp.array[dtype]]
            for dtype in _MATRIX_DTYPES
            for entry_dtype in _MATRIX_DTYPES
        },
    )
    SANDWICH_ROW_COUNTS = OverloadTable(
        sandwich_row_counts,
        {d: [i32, i32, wp.array[d], wp.int32, i32, wp.array[d]] for d in _MATRIX_DTYPES},
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
                d,
                i32,
                wp.int32,
                wp.array[wp.uint64],
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
                wp.int32,
                wp.array[wp.uint64],
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
                wp.int32,
                wp.array[wp.uint64],
                i32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    # Keyed on the half-cotangents' dtype alone: the Laplacian is always ``float64`` here.
    LSCM_LAPLACIAN_ROWS_AND_COUNTS = OverloadTable(
        lscm_laplacian_rows_and_counts,
        {
            d: [
                i32,
                i32,
                i32,
                i32,
                wp.array[wp.uint64],
                wp.int32,
                i32,
                wp.array2d[d],
                wp.array[wp.float64],
                i32,
            ]
            for d in _MATRIX_DTYPES
        },
    )


_register_overloads()
