import warp as wp

from ordito import constants as twc
from ordito.kernels.array import cross2, to_vec2d
from ordito.kernels.laplacian import cotmatrix_offdiagonal
from ordito.kernels.linalg import free_row
from ordito.kernels.predicates import squared_edge_lengths
from ordito.kernels.triangles import corner_triple, face_vertices, row_triple

wp.set_module_options({"enable_backward": False})

# Below this (float64) squared edge length the isometric rest-triangle flattening is treated as
# degenerate: its rest edges are zeroed so the ARAP local step contributes nothing for that face.
EPSILON_ARAP_EDGE_SQ = wp.constant(wp.float64(1.0e-20))


@wp.func
def face_is_flipped(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], fi: wp.int32
) -> wp.bool:
    # Whether face ``fi`` has negative signed area in the plane. Shared by the mask kernel and the
    # flag kernel, which differ only in the dtype they store it as.
    v0, v1, v2 = face_vertices(vertices, faces, fi)
    e0 = v1 - v0
    e1 = v2 - v0
    # 2D signed area * 2 == det of libigl's homogeneous 3x3 matrix
    return cross2(e0, e1) < 0.0


@wp.kernel
def face_flipped_mask(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    fi = wp.int32(wp.tid())
    out_mask[fi] = face_is_flipped(vertices, faces, fi)


@wp.kernel
def face_flipped_flags(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # ``face_flipped_mask`` as the ``int32`` flag a caller scans in place to compact the flipped
    # face indices, with no bool mask in between.
    fi = wp.int32(wp.tid())
    out_flags[fi] = wp.where(face_is_flipped(vertices, faces, fi), wp.int32(1), wp.int32(0))


@wp.kernel
def scatter_fixed_uv(
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    out_fixed_mask: wp.array[wp.bool],
    out_fixed_values: wp.array2d[wp.float64],
) -> None:
    # Scatter the prescribed boundary positions into a ``(2, n_vertices)`` buffer (row 0 = u,
    # row 1 = v) so the system-assembly kernel can look up ``bc[c, j]`` by right-hand-side column
    # ``c`` and original vertex index ``j``. float64 to match the float64 conjugate-gradient path.
    #
    # The fixed-vertex mask is marked in the same pass -- the ``array.indices_to_mask`` of the same
    # indices, whose out-of-range rule (skip negative and ``>= n``) is the one applied here -- as
    # ``scatter_pinned_stacked`` does for LSCM's stacked layout. The skip is memory safety only: a
    # skipped index, or a repeated one (whose writes race), marks fewer vertices than the indices
    # list, and the wrapper rejects the call from the mask's free count, which it reads anyway.
    b = wp.int32(wp.tid())
    i = boundary_indices[b]
    if i < wp.int32(0) or i >= out_fixed_values.shape[1]:
        return
    uv = boundary_uv[b]
    out_fixed_mask[i] = wp.bool(True)
    out_fixed_values[0, i] = wp.float64(uv[0])
    out_fixed_values[1, i] = wp.float64(uv[1])


@wp.func
def fixed_uv(fixed_values: wp.array2d[wp.float64], i: wp.int32) -> wp.vec2:
    # The prescribed position of fixed vertex ``i``, narrowed from the ``(2, n_vertices)`` buffer.
    return wp.vec2(wp.float32(fixed_values[0, i]), wp.float32(fixed_values[1, i]))


@wp.kernel
def negated_cotmatrix_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    out_values: wp.array[wp.float64],
) -> None:
    # The ``k == 1`` harmonic system ``-L`` over a mesh operator pattern, one thread per row:
    # ``kernels/laplacian.cotmatrix_rows``' ``float64`` row with each entry negated as it is
    # written -- every off-diagonal ``-cotmatrix_offdiagonal`` and the diagonal the off-diagonal
    # sum itself, added in the same column order -- so ``L`` is never stored. Negation is exact, so
    # the entries are ``energies.k_harmonic(L, k=1)``'s bit for bit.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    total = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(start, end):
        if columns[k] == i:
            slot = k
            continue
        value = cotmatrix_offdiagonal(
            k, run_start, keys, count, order, cot_entries, wp.float64(0.0)
        )
        out_values[k] = -value
        total += value
    if slot >= 0:
        out_values[slot] = total


@wp.kernel
def free_mass_roots(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    out_roots: wp.array[wp.float64],
) -> None:
    # ``sqrt(M_ii)`` at each free vertex's compact index: the diagonal ``D`` for which the k = 2
    # operator ``L M^-1 L`` is the ``L D^-2 L`` a squared-Laplacian preconditioner inverts. A
    # vertex no face refers to has zero mass and an empty row; 1 keeps ``D`` positive there.
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    out_roots[ri] = wp.where(mass[i] > wp.float64(0.0), wp.sqrt(mass[i]), wp.float64(1.0))


@wp.func
def solved_uv(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    i: wp.int32,
) -> wp.vec2:
    # Vertex ``i``'s UV in a reduced solution: its prescribed position when fixed, else the solved
    # value at its compact index, narrowed to ``float32``. ``scatter_solution`` writes it; ARAP's
    # local step reads it in place of a scattered UV field.
    if fixed_mask[i]:
        return fixed_uv(fixed_values, i)
    ri = free_map[i]
    return wp.vec2(wp.float32(sol[0, ri]), wp.float32(sol[1, ri]))


@wp.kernel
def scatter_solution(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Reassemble the full ``(n_vertices,)`` UV field: fixed vertices keep their prescribed position
    # (``fixed_values`` is ``(2, n_vertices)``), free vertices read the solved value at their
    # compact index (``sol`` is ``(2, n_free)``). Handles the all-fixed case (``n_free == 0``): the
    # free branch is then never taken, so the empty ``sol`` is never indexed.
    i = wp.int32(wp.tid())
    out_uv[i] = solved_uv(fixed_mask, free_map, sol, fixed_values, i)


@wp.kernel
def boundary_edge_lengths(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], out_len: wp.array[wp.float32]
) -> None:
    # Segment length between consecutive boundary vertices; ``out_len[0] = 0`` seeds the arc-length
    # prefix sum (matches ``igl::map_vertices_to_circle``).
    i = wp.int32(wp.tid())
    if i == 0:
        out_len[0] = wp.float32(0.0)
    else:
        out_len[i] = wp.length(vertices[boundary[i]] - vertices[boundary[i - 1]])


@wp.kernel
def circle_positions(
    boundary: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    cumulative_length: wp.array[wp.float32],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Arc-length parametrization onto the unit circle: ``frac = len[i] * 2pi / total`` with the
    # total perimeter closing over the wrap edge ``bnd[0] -> bnd[n-1]``.
    i = wp.int32(wp.tid())
    n = cumulative_length.shape[0]
    wrap = wp.length(vertices[boundary[0]] - vertices[boundary[n - 1]])
    total = cumulative_length[n - 1] + wrap
    if total <= 0.0:
        # A single-vertex loop, or one whose vertices are all coincident, has zero perimeter --
        # every ``cumulative_length`` entry is then 0 too, so this is the only guard needed to
        # avoid the 0/0 that would otherwise silently write NaN into every position.
        total = wp.float32(1.0)
    frac = cumulative_length[i] * twc.TWO_PI / total
    out_uv[i] = wp.vec2(wp.cos(frac), wp.sin(frac))


@wp.kernel
def scatter_pinned_stacked(
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    n_vertices: wp.int32,
    out_fixed_mask: wp.array[wp.bool],
    out_fixed_values: wp.array2d[wp.float64],
) -> None:
    # Fused mask + value scatter for LSCM's stacked ``[u; v]`` DOFs: pin ``i`` fixes DOF ``i`` (u)
    # and ``i + n`` (v). ``out_fixed_values`` is ``(1, 2n)`` (single right-hand-side column). An
    # out-of-range pin is skipped and a repeated one marks its vertex once, which the wrapper
    # rejects from the free count, as for ``scatter_fixed_uv``.
    b = wp.int32(wp.tid())
    i = pinned_indices[b]
    if i < wp.int32(0) or i >= n_vertices:
        return
    uv = pinned_uv[b]
    out_fixed_mask[i] = True
    out_fixed_mask[i + n_vertices] = True
    out_fixed_values[0, i] = wp.float64(uv[0])
    out_fixed_values[0, i + n_vertices] = wp.float64(uv[1])


@wp.kernel
def scatter_solution_stacked(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array[wp.float64],
    fixed_values: wp.array[wp.float64],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Unstack the solved ``[u; v]`` DOF vector into ``(n, 2)`` UV. DOF ``i`` holds u, DOF ``i + n``
    # holds v; each is either a pinned value (``fixed_values``) or a solved free value (``sol`` at
    # the compact free index). Locals are initialized before the branch per Warp's branch-scope
    # rule.
    i = wp.int32(wp.tid())
    n = out_uv.shape[0]
    u = wp.float64(0.0)
    v = wp.float64(0.0)
    if fixed_mask[i]:
        u = fixed_values[i]
    else:
        u = sol[free_map[i]]
    if fixed_mask[i + n]:
        v = fixed_values[i + n]
    else:
        v = sol[free_map[i + n]]
    out_uv[i] = wp.vec2(wp.float32(u), wp.float32(v))


@wp.func
def arap_rest_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float64],
    f: wp.int32,
    out_rest_edges: wp.array2d[wp.vec2d],
) -> None:
    # Isometrically flatten each triangle into the plane (igl project_isometrically_to_plane) and
    # store its three weight-folded rest edges ``c_e * p_e`` in igl edge order e0:(1,2), e1:(2,0),
    # e2:(0,1). Run once (``arap_setup``): folding the half-cotangent weight ``c_e`` in here removes
    # the cotangent lookup from the per-iteration local step. Computed in float64 (squared edge
    # lengths promoted from the float32 vertex precision) for a deterministic operator.
    v0, v1, v2 = face_vertices(vertices, faces, f)
    l2_0f, l2_1f, l2_2f = squared_edge_lengths(v0, v1, v2)
    # l0 = |v1 - v2|, l1 = |v2 - v0|, l2 = |v0 - v1| (igl edge_lengths column order).
    l2_0 = wp.float64(l2_0f)
    l2_1 = wp.float64(l2_1f)
    l2_2 = wp.float64(l2_2f)
    c0, c1, c2 = row_triple(cot_entries, f)
    if l2_2 < EPSILON_ARAP_EDGE_SQ:
        # Degenerate base edge |v0 - v1|: emit zero rest edges (local step contributes nothing).
        out_rest_edges[f, 0] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        out_rest_edges[f, 1] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        out_rest_edges[f, 2] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        return
    # Flattened rest positions: P0 = (0, 0), P1 = (|v0 - v1|, 0), P2 from the law of cosines.
    base = wp.sqrt(l2_2)
    p2x = (-l2_0 + l2_1 + l2_2) / (wp.float64(2.0) * base)
    p2y_sq = wp.max(l2_1 - p2x * p2x, wp.float64(0.0))
    p1 = wp.vec2d(base, wp.float64(0.0))
    p2 = wp.vec2d(p2x, wp.sqrt(p2y_sq))
    # p_e = P_source - P_dest per igl edge, folded with c_e: e0 -> P1 - P2, e1 -> P2 (P0 = 0),
    # e2 -> -P1 (P0 - P1).
    out_rest_edges[f, 0] = c0 * (p1 - p2)
    out_rest_edges[f, 1] = c1 * p2
    out_rest_edges[f, 2] = -c2 * p1


@wp.func
def arap_edge_endpoints(corners: wp.vec3i, e: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The (source, dest) vertices of a face's ARAP edge ``e`` in igl edge order e0:(1,2), e1:(2,0),
    # e2:(0,1). One spelling for the local step's scatter and the endpoint buffer ``arap_setup``
    # writes for the deterministic sum, whose slot ``6 * f + 2 * e`` (source) and ``+ 1`` (dest)
    # must name the vertex the scatter's term goes to.
    return corners[(e + 1) % 3], corners[(e + 2) % 3]


@wp.func
def scatter_arap_edge(
    rotation: wp.mat22d,
    w: wp.vec2d,
    corners: wp.vec3i,
    f: wp.int32,
    e: wp.int32,
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
    out_edge_terms: wp.array2d[wp.vec2d],
) -> None:
    # Rotate the weight-folded rest edge ``e`` of face ``f`` and scatter the ARAP right-hand side
    # ``+r`` to the edge source and ``-r`` to the edge dest. The two components go to separate 1D
    # buffers so the atomic adds stay 1D (row-view outputs of the (2, n_vertices) rotation RHS).
    #
    # Under ``ordito.config``'s deterministic mode ``out_edge_terms`` is a non-empty
    # ``(n_faces, 6)`` table and the two terms are stored in it instead, at columns ``2 * e`` and
    # ``2 * e + 1``, for ``scatter.sum_sorted_corner_values`` to add per vertex over the stable sort
    # of the endpoint buffer: in ascending face, edge and source-before-dest order, the order the
    # CPU's in-order atomics add them in (the branch is warp-uniform).
    r = rotation * w
    if out_edge_terms.shape[0] > 0:
        out_edge_terms[f, 2 * e] = r
        out_edge_terms[f, 2 * e + 1] = -r
    else:
        source, dest = arap_edge_endpoints(corners, e)
        wp.atomic_add(out_rhs_x, source, r[0])
        wp.atomic_add(out_rhs_y, source, r[1])
        wp.atomic_add(out_rhs_x, dest, -r[0])
        wp.atomic_add(out_rhs_y, dest, -r[1])


@wp.kernel
def arap_local_step(
    faces: wp.array[wp.int32],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    rest_edges: wp.array2d[wp.vec2d],
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
    out_edge_terms: wp.array2d[wp.vec2d],
) -> None:
    # Per face (dim = n_faces, once per iteration): fit the closest 2D rotation to the current UV
    # and scatter the rotation right-hand side, fused so no per-face rotation buffer is needed.
    # The current UV is read straight off the reduced solution (``solved_uv``: the value
    # ``scatter_solution`` stores), so the loop scatters the full field only once, after its last
    # step. ``out_rhs_x`` / ``out_rhs_y`` are the two rows of the (2, n_vertices) rotation RHS and
    # must be zero at launch (``arap_interior_rhs`` clears them as it reads them), unless
    # ``out_edge_terms`` is non-empty and takes the terms instead (``scatter_arap_edge``).
    # Degenerate faces have zero rest edges, so S = 0, atan2(0, 0) = 0, and the identity rotation
    # scatters nothing.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    uv0 = solved_uv(fixed_mask, free_map, sol, fixed_values, i0)
    uv1 = solved_uv(fixed_mask, free_map, sol, fixed_values, i1)
    uv2 = solved_uv(fixed_mask, free_map, sol, fixed_values, i2)
    # UV edges in igl order e0:(1,2), e1:(2,0), e2:(0,1), promoted to float64.
    u0 = to_vec2d(uv1) - to_vec2d(uv2)
    u1 = to_vec2d(uv2) - to_vec2d(uv0)
    u2 = to_vec2d(uv0) - to_vec2d(uv1)
    w0, w1, w2 = row_triple(rest_edges, f)
    # Covariance S = sum_e u_e (outer) w_e (rest edges already weight-folded with c_e).
    s = wp.outer(u0, w0) + wp.outer(u1, w1) + wp.outer(u2, w2)
    # Closest proper rotation (reflections forbidden), fit_rotations_planar closed form.
    theta = wp.atan2(s[1, 0] - s[0, 1], s[0, 0] + s[1, 1])
    cos_t = wp.cos(theta)
    sin_t = wp.sin(theta)
    rotation = wp.mat22d(cos_t, -sin_t, sin_t, cos_t)
    corners = wp.vec3i(i0, i1, i2)
    scatter_arap_edge(rotation, w0, corners, f, 0, out_rhs_x, out_rhs_y, out_edge_terms)
    scatter_arap_edge(rotation, w1, corners, f, 1, out_rhs_x, out_rhs_y, out_edge_terms)
    scatter_arap_edge(rotation, w2, corners, f, 2, out_rhs_x, out_rhs_y, out_edge_terms)


@wp.kernel
def arap_interior_rhs(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    rhs_const: wp.array2d[wp.float64],
    rhs_rot_x: wp.array[wp.float64],
    rhs_rot_y: wp.array[wp.float64],
    rhs_rot_sorted: wp.array[wp.vec2d],
    out_b: wp.array2d[wp.float64],
) -> None:
    # Assemble the interior right-hand side of the global step (dim = n_vertices). Interior rows get
    # the constant term ``rhs_const`` (the ``-(-L)_ub bc`` boundary contribution, already at the
    # compact free index) plus the scattered rotation RHS at the original vertex index; boundary
    # rows carry no unknown and are skipped. ``rhs_const`` / ``out_b`` are (2, n_interior) (rows
    # u, v); ``rhs_rot_*`` are (n_vertices,).
    #
    # ``rhs_rot_*`` are cleared here, every row, as they are read: the next iteration's local
    # step scatters into them, and each slot is read only by its own thread, after the scatter
    # that filled it, so the loop issues no memset.
    #
    # Under ``ordito.config``'s deterministic mode the rotation RHS is instead the non-empty
    # ``rhs_rot_sorted``, the ``(n_vertices,)`` sums ``scatter.sum_sorted_corner_values`` writes
    # whole every iteration, so nothing is cleared (the branch is warp-uniform).
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if rhs_rot_sorted.shape[0] > 0:
        if ri >= 0:
            rot = rhs_rot_sorted[i]
            out_b[0, ri] = rhs_const[0, ri] + rot[0]
            out_b[1, ri] = rhs_const[1, ri] + rot[1]
        return
    if ri >= 0:
        out_b[0, ri] = rhs_const[0, ri] + rhs_rot_x[i]
        out_b[1, ri] = rhs_const[1, ri] + rhs_rot_y[i]
    rhs_rot_x[i] = wp.float64(0.0)
    rhs_rot_y[i] = wp.float64(0.0)


@wp.func
def gather_interior_uv(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    i: wp.int32,
    out_sol: wp.array2d[wp.float64],
) -> None:
    # Seed the conjugate-gradient warm start with the current interior UV at vertex ``i``:
    # interior vertex ``i`` writes its UV into the two rows (u, v) of ``out_sol`` at the compact
    # free index. ``out_sol`` is (2, n_interior), and every row is written, because ``free_map`` is
    # a bijection of the free vertices onto it. The first local step reads the working UV field
    # off this seed (``solved_uv``), which gives a free vertex ``uv[i]`` itself: a ``float32``
    # widened to ``float64`` and narrowed again is exact.
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    p = uv[i]
    out_sol[0, ri] = wp.float64(p[0])
    out_sol[1, ri] = wp.float64(p[1])


@wp.kernel
def arap_setup(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    out_rest_edges: wp.array2d[wp.vec2d],
    out_sol: wp.array2d[wp.float64],
    out_edge_endpoints: wp.array[wp.int32],
    out_edge_slots: wp.array[wp.int32],
) -> None:
    # The ARAP loop's two one-time inputs in one launch over ``max(n_faces, n_vertices)`` threads:
    # the rest edges per face (``arap_rest_edges``) and the warm start per vertex
    # (``gather_interior_uv``). They write disjoint buffers from the input alone.
    #
    # Under ``ordito.config``'s deterministic mode ``out_edge_endpoints`` / ``out_edge_slots`` are
    # the non-empty ``(12 * n_faces,)`` buffers of a radix sort, and their leading halves receive
    # each face's six edge endpoints in ``scatter_arap_edge``'s table order and the identity
    # payload: the stable sort of those orders the per-vertex sum of the local step's terms.
    t = wp.int32(wp.tid())
    if t < faces.shape[0] // 3:
        arap_rest_edges(vertices, faces, cot_entries, t, out_rest_edges)
        if out_edge_endpoints.shape[0] > 0:
            i0, i1, i2 = corner_triple(faces, t)
            corners = wp.vec3i(i0, i1, i2)
            for e in range(3):
                source, dest = arap_edge_endpoints(corners, e)
                slot = 6 * t + 2 * e
                out_edge_endpoints[slot] = source
                out_edge_endpoints[slot + 1] = dest
                out_edge_slots[slot] = slot
                out_edge_slots[slot + 1] = slot + 1
    if t < vertices.shape[0]:
        gather_interior_uv(fixed_mask, free_map, uv, t, out_sol)
