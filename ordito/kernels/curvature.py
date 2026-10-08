import warp as wp

from ordito.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from ordito.kernels.adjacency import unshared_projection, unshared_vertex
from ordito.kernels.linalg import solve_normal_equations
from ordito.kernels.neighbors import in_ball
from ordito.kernels.predicates import angle_defect, segment_aabb, unit_tangent, vector_angle
from ordito.kernels.tangent_space import any_perpendicular
from ordito.kernels.triangles import face_normals_and_area

wp.set_module_options({"enable_backward": False})

# Custom fixed-size float64 types for the 5x5 quadric-fit normal equations: the rest of the
# kernel runs in float32, but the least-squares solve is done in float64 for conditioning.
vec5d = wp.types.vector(length=5, dtype=wp.float64)
mat55d = wp.types.matrix(shape=(5, 5), dtype=wp.float64)

# ---------------------------------------------------------------------------
# principal_curvature helpers
# ---------------------------------------------------------------------------


@wp.func
def _build_reference_frame(
    vertex: wp.vec3, normal: wp.vec3, first_neighbor: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """Return (t1, t2) orthonormal tangent frame with t1 pointing toward first_neighbor."""
    # Same construction as ``tangent_space.vertex_tangent_frames``: the reference direction is the
    # neighbour projected into the tangent plane, falling back to an arbitrary perpendicular when
    # that projection vanishes (the neighbour sits on the normal line through the vertex).
    t1 = any_perpendicular(normal)
    tangential, length = unit_tangent(first_neighbor - vertex, normal, TOLERANCE_ZERO_CONSTANT)
    if length > TOLERANCE_ZERO_CONSTANT:
        t1 = tangential
    # The ``normalize`` is *not* redundant, though the arithmetic says it is: ``normal`` is unit
    # and ``t1`` is orthogonal to it, so the cross product is unit in exact arithmetic and 1 +- an
    # ULP or so in float32. Dropping it was measured and reverted -- that residual is enough to flip
    # which of two nearly equal principal curvatures is the larger, which moves ``PV1`` visibly at
    # exactly the vertices where the pair is close. With it, the whole family is bit-identical on
    # the CPU device across every fixture, both ``frame_independent`` modes and both fit radii.
    return t1, wp.normalize(wp.cross(normal, t1))


@wp.func
def _eigvec_2x2(
    m00: wp.float32, m01: wp.float32, m10: wp.float32, m11: wp.float32, lam: wp.float32
) -> wp.vec2:
    """
    Return the unit eigenvector of a 2x2 matrix for the eigenvalue ``lam``.

    The matrix is ``[[m00, m01], [m10, m11]]`` and the result is expressed in the (t1, t2)
    tangent-frame basis. ``m - lam*I`` is singular, so its two rows are proportional and each
    gives the eigenvector as its own perpendicular: row 0 gives ``[m01, lam - m00]`` and row 1
    gives ``[lam - m11, m10]``. Both forms are valid for any 2x2 (symmetric or the non-symmetric
    Weingarten map), but either row can vanish on its own, so the longer of the two is taken.

    Picking by *length* rather than against an absolute threshold is what makes a diagonal matrix
    come out right. There, ``m01 = 0`` and the eigenvalue equals one of the diagonal entries, so
    for that eigenvalue row 0 is the zero row up to rounding -- a few ULP of ``m00``, far above any
    fixed epsilon -- and normalizing it returns an arbitrarily-signed ``(0, +-1)`` instead of the
    correct ``(1, 0)``. Reading row 1 instead recovers it.

    ``(1, 0)`` is returned when both rows vanish, i.e. ``m == lam*I`` (an umbilic point, where
    every direction is a principal direction and there is nothing to find). That test is taken
    *relative* to the matrix's own magnitude, because these entries are curvatures and so scale as
    one over the mesh's: an absolute epsilon would call a large mesh's whole shape operator
    umbilic. Which arbitrary direction comes back does not matter, because the caller derives the
    second principal direction from this one rather than solving for it again.
    """
    row0 = wp.vec2(m01, lam - m00)
    row1 = wp.vec2(lam - m11, m10)
    v = wp.where(wp.length_sq(row0) >= wp.length_sq(row1), row0, row1)
    magnitude = wp.max(wp.abs(m00) + wp.abs(m01), wp.abs(m10) + wp.abs(m11))
    floor = TOLERANCE_ZERO_CONSTANT * magnitude
    if wp.length_sq(v) <= floor * floor:
        return wp.vec2(1.0, 0.0)
    return wp.normalize(v)


@wp.func
def _principal_curvatures_from_monge(
    first_form: wp.vec3, second_form: wp.vec3, frame_independent: wp.bool
) -> tuple[wp.float32, wp.float32, wp.vec2]:
    """
    Extract principal curvatures and directions from the fundamental forms (Monge patch).

    Takes the first fundamental form ``first_form = (E, F, G)`` and second fundamental form
    ``second_form = (L, M, N)`` and builds the shape operator (Weingarten map) as a 2x2 matrix whose
    (real) eigenvalues are the principal curvatures. The two formulations share ``m00``, ``m10`` and
    ``m11`` and differ only in the upper-right entry:

        m = [[L*G - M*F,  m01      ],
             [M*E - L*F,  N*E - M*F]] / (E*G - F*F)

    * ``frame_independent=True`` (textbook): ``m01 = (M*G - N*F) / (E*G - F*F)`` -- the true
      generalized eigenvalue problem ``II*v = lam*I*v``. The eigenvalues are surface invariants and
      do not depend on the chosen tangent frame.
    * ``frame_independent=False``: ``m01 = M*E - L*F`` (reuses the lower-left term), reproducing
      ``igl::principal_curvature``'s ``finalEigenStuff`` verbatim. libigl forces this symmetry,
      which keeps the trace (mean curvature) exact but alters the determinant (eigenvalue spread)
      and makes the result depend on the reference frame.

    Returns (lam0, lam1, ev0) where lam0 <= lam1 are eigenvalues of ``m``. The caller negates them
    (libigl's ``c_val = -c_val``). ``ev0`` is ``lam0``'s unit eigenvector as a ``wp.vec2`` in the
    (t1, t2) tangent-frame 2D basis.

    **Only the first eigenvector is returned, deliberately.** Principal directions are orthogonal
    -- the shape operator is self-adjoint with respect to the first fundamental form -- so the
    second is the first rotated a quarter turn in the tangent plane, and the caller gets it from a
    cross product with the vertex normal. Solving for it separately instead costs the guarantee:
    the two solves see the same numerically degenerate matrix at an umbilic point and can land on
    the *same* direction, which is the one answer that is wrong however arbitrary the choice is
    allowed to be. On a sphere, umbilic everywhere, that returns exactly parallel principal
    directions at a handful of vertices. The non-symmetric ``frame_independent=True`` branch is
    worse, because its eigenvectors are orthogonal under the first fundamental form rather than in
    the frame's own coordinates: on a torus it puts half the pairs visibly off perpendicular at
    eigenvalue gaps where nothing is degenerate at all.
    """
    e_ff = first_form[0]
    f_ff = first_form[1]
    g_ff = first_form[2]
    l_ff = second_form[0]
    m_ff = second_form[1]
    n_ff = second_form[2]

    inv_denom = wp.float32(1.0) / (e_ff * g_ff - f_ff * f_ff)
    m00 = (l_ff * g_ff - m_ff * f_ff) * inv_denom
    m10 = (m_ff * e_ff - l_ff * f_ff) * inv_denom
    m11 = (n_ff * e_ff - m_ff * f_ff) * inv_denom
    # Upper-right entry: the textbook Weingarten map uses (M*G - N*F); libigl forces symmetry by
    # reusing the lower-left (M*E - L*F), which is what makes its eigenvalues frame-dependent.
    if frame_independent:
        m01 = (m_ff * g_ff - n_ff * f_ff) * inv_denom
    else:
        m01 = m10

    # Eigenvalues of [[m00, m01], [m10, m11]] (ascending). The discriminant argument is
    # mathematically >= 0 for the Weingarten map; wp.max guards numerical dips near umbilics.
    half_trace = (m00 + m11) * wp.float32(0.5)
    half_diff = (m00 - m11) * wp.float32(0.5)
    disc = wp.sqrt(wp.max(half_diff * half_diff + m01 * m10, wp.float32(0.0)))
    lam0 = half_trace - disc
    lam1 = half_trace + disc

    # Eigenvector of (m - lam0*I)v = 0. lam1's is the caller's cross product -- see the docstring.
    return lam0, lam1, _eigvec_2x2(m00, m01, m10, m11, lam0)


# ---------------------------------------------------------------------------
# principal curvature kernel
# ---------------------------------------------------------------------------


@wp.func
def _write_no_curvature(
    i: wp.int32,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # What ``fit_principal_curvature`` writes at a vertex it declines to fit -- too few neighbours,
    # a neighbourhood of coincident points, or a singular normal matrix. One name for it because
    # the three exits must stay the same answer: a caller reads a zero ``PV1`` as "no fit here",
    # and a branch that wrote only part of the four would leave the rest holding another vertex's.
    out_pd1[i] = wp.vec3()
    out_pd2[i] = wp.vec3()
    out_pv1[i] = wp.float32(0.0)
    out_pv2[i] = wp.float32(0.0)


@wp.kernel
def fit_principal_curvature(
    vertices: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    reference_neighbors: wp.array[wp.int32],
    frame_independent: wp.bool,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    """
    Fit a quadric surface in a local tangent frame per vertex.

    Extract principal curvature directions and magnitudes. Matches igl::principal_curvature.
    """
    i = wp.int32(wp.tid())

    start = offsets[i]
    end = offsets[i + 1]  # ``geodesic_ball`` returns the terminated (n + 1) CSR row bounds
    n_nbr = end - start

    # The ball includes the centre vertex itself, which the fit loop below skips, so a determined
    # 5-parameter quadric fit needs 6 entries here. Anything short of that leaves the normal
    # equations rank-deficient and ``solve_normal_equations`` would report it -- this only saves
    # running a solve whose answer is already known. Remaining degeneracies are caught there.
    if n_nbr < 6:
        _write_no_curvature(i, out_pd1, out_pd2, out_pv1, out_pv2)
        return

    vertex = vertices[i]
    normal = wp.normalize(vertex_normals[i])

    # Build the tangent frame from the lowest-indexed mesh-adjacency neighbor, matching libigl's
    # computeReferenceFrame (adjacency_list[i][0]). When frame_independent is False, libigl extracts
    # curvature from a *symmetrized* shape operator whose eigenvalues depend on the chosen frame, so
    # the principal values only agree with igl::principal_curvature when this exact reference
    # direction is used. When frame_independent is True the eigenvalues are surface invariants, so
    # the exact frame is irrelevant (any orthonormal tangent basis yields the same result).
    ref = reference_neighbors[i]
    t1, t2 = _build_reference_frame(vertex, normal, vertices[ref])

    # Count neighbors passing projection-plane filter, including self (self always passes,
    # dot=1).
    # Matches libigl's applyProjOnPlane which includes vv[self] because dot(n_i, n_i) = 1 > 0.
    # The neighbour normal is not normalized here or in the fit loop below: only the *sign* of the
    # dot product is read, and scaling by a positive length cannot change it.
    #
    # The same sweep takes the neighbourhood's radius, which the fit below divides its local
    # coordinates by. That is not a tolerance choice, it is what makes the fit scale-free: the
    # design row is ``[u^2, u v, v^2, u, v]``, so at mesh scale ``h`` the normal matrix's diagonal
    # spans ``h^8`` to ``h^2`` and ``solve_normal_equations``' singularity test -- absolute, and
    # necessarily so, since it cannot see the caller's units -- starts rejecting well-conditioned
    # fits outright. Without the division a small enough mesh returns zero curvature everywhere,
    # silently.
    n_valid = wp.int32(0)
    ring_radius = wp.float32(0.0)
    for k in range(n_nbr):
        j = neighbor_indices[start + k]
        if j == i:
            n_valid = n_valid + 1  # self always passes
            continue
        ring_radius = wp.max(ring_radius, wp.length(vertices[j] - vertex))
        if wp.dot(vertex_normals[j], normal) > wp.float32(0.0):
            n_valid = n_valid + 1

    # Every neighbour coincides with the centre: no frame, no fit, and the normal equations would
    # be exactly singular. Same answer this kernel gives for a solve that reports singular.
    if ring_radius <= wp.float32(0.0):
        _write_no_curvature(i, out_pd1, out_pd2, out_pv1, out_pv2)
        return
    inv_radius = wp.float64(1.0) / wp.float64(ring_radius)

    # Mirror libigl: only apply the filter if it leaves at least 6 neighbours. libigl also requires
    # the filtered set to be strictly smaller than the full one, which is not repeated here because
    # it cannot change the answer -- when every neighbour passes, filtering removes nothing.
    use_filter = n_valid >= 6

    # Least-squares quadric fit: accumulate the normal equations AᵀA x = Aᵀb.
    ata = mat55d()
    atb = vec5d()

    for k in range(n_nbr):
        j = neighbor_indices[start + k]
        if j == i:
            continue  # self contributes (0,0,0) — skip to avoid frame degeneration
        if use_filter and wp.dot(vertex_normals[j], normal) <= wp.float32(0.0):
            continue

        # Local coordinates in units of the ring radius, per the comment above the sweep that
        # measured it.
        diff = vertices[j] - vertex
        u = wp.float64(wp.dot(diff, t1)) * inv_radius
        v_c = wp.float64(wp.dot(diff, t2)) * inv_radius
        w = wp.float64(wp.dot(diff, normal)) * inv_radius

        # row of A: [u², uv, v², u, v]
        r = vec5d(u * u, u * v_c, v_c * v_c, u, v_c)
        ata += wp.outer(r, r)
        atb += r * w

    solution, ok = solve_normal_equations(ata, atb)
    if not ok:
        _write_no_curvature(i, out_pd1, out_pd2, out_pv1, out_pv2)
        return

    # Cast the float64 solution back to float32 for the rest of the kernel, undoing the ring-radius
    # scaling as it goes. With ``u' = u / R`` and ``w' = w / R``, matching ``w = a u^2 + ... + e v``
    # against ``w' = a' u'^2 + ... + e' v'`` gives the quadratic coefficients a factor ``1 / R`` and
    # leaves the linear ones alone -- so the Monge form below is in the caller's own units.
    a = wp.float32(solution[0] * inv_radius)
    b = wp.float32(solution[1] * inv_radius)
    c = wp.float32(solution[2] * inv_radius)
    d = wp.float32(solution[3])
    e = wp.float32(solution[4])

    # First fundamental form coefficients. Its determinant needs no degeneracy guard: it expands
    # to (1 + d*d)(1 + e*e) - (d*e)^2 = 1 + d*d + e*e, which is >= 1 for every finite fit, so
    # ``_principal_curvatures_from_monge`` can divide by it unconditionally.
    E_ff = wp.float32(1.0) + d * d  # noqa: N806
    F_ff = d * e  # noqa: N806
    G_ff = wp.float32(1.0) + e * e  # noqa: N806

    # Normal z-component in local frame
    nz = wp.float32(1.0) / wp.sqrt(d * d + e * e + wp.float32(1.0))

    # Second fundamental form
    L_ff = wp.float32(2.0) * a * nz  # noqa: N806
    M_ff = b * nz  # noqa: N806
    N_ff = wp.float32(2.0) * c * nz  # noqa: N806

    first_form = wp.vec3(E_ff, F_ff, G_ff)
    second_form = wp.vec3(L_ff, M_ff, N_ff)
    lam0, lam1, ev0 = _principal_curvatures_from_monge(first_form, second_form, frame_independent)

    # Negate: the Monge patch height function curves downward for convex surfaces,
    # giving negative eigenvalues; convention is positive curvature for convex.
    k0 = -lam0
    k1 = -lam1

    # Reconstruct the global direction from the local eigenvector, and take the second as its
    # quarter turn in the tangent plane. (t1, t2, normal) is orthonormal, so the cross product is
    # already unit; it is also what guarantees the pair is a *frame* rather than two independently
    # solved vectors that can coincide -- see ``_principal_curvatures_from_monge``.
    dir0 = wp.normalize(t1 * ev0[0] + t2 * ev0[1])
    dir1 = wp.cross(normal, dir0)

    # Assign so that PV1 >= PV2
    if k0 >= k1:
        out_pd1[i] = dir0
        out_pd2[i] = dir1
        out_pv1[i] = k0
        out_pv2[i] = k1
    else:
        out_pd1[i] = dir1
        out_pd2[i] = dir0
        out_pv1[i] = k1
        out_pv2[i] = k0


@wp.kernel
def ball_angle_defect_sum(
    vertices: wp.array[wp.vec3],
    angle_sum: wp.array[wp.float64],
    points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    out_curvature: wp.array[wp.float32],
) -> None:
    # The Cohen-Steiner/Morvan Gaussian measure of the ball around one query: the angle defect of
    # every vertex inside it, formed from its incident-angle sum as it is read and summed during the
    # hash-grid walk. The walk and predicate are ``kernels.neighbors.ball_collect``'s hash-grid
    # branch, so a query folds exactly the vertices a ball-query CSR would list, in its order --
    # with no defect pass, count pass, scan, readback or neighbour-sized buffers in between, and one
    # sequential sum per query where a scatter over that CSR added the same terms by atomics. The
    # defect is ``predicates.angle_defect``, the rule ``vertices.vertex_defects`` applies.
    #
    # In ``float64``, angle sums and ball total alike: a defect is a full turn less an angle sum
    # close to it, so in ``float32`` each carries an absolute error of a few ulp of 2 pi, and a
    # ball of hundreds of vertices added those up -- 7.4e-5 off trimesh at 375 vertices.
    q = wp.int32(wp.tid())
    p = points[q]
    total = wp.float64(0.0)
    v = wp.int32(0)
    query = wp.hash_grid_query(grid_id, p, radius)
    while wp.hash_grid_query_next(query, v):
        if in_ball(vertices[v] - p, radius):
            total = total + angle_defect(angle_sum[v])
    out_curvature[q] = wp.float32(total)


@wp.func
def line_ball_intersection_segment(
    start_point: wp.vec3, end_point: wp.vec3, center: wp.vec3, radius: wp.float32
) -> wp.float32:
    segment = end_point - start_point
    oc = start_point - center
    r = radius
    ldotl = wp.length_sq(segment)
    ldotoc = wp.dot(segment, oc)
    ocdotoc = wp.length_sq(oc)
    discrim = ldotoc * ldotoc - ldotl * (ocdotoc - r * r)

    if discrim <= wp.float32(0.0):
        return wp.float32(0.0)

    sqrt_discrim = wp.sqrt(discrim)
    d1 = (-ldotoc - sqrt_discrim) / ldotl
    d2 = (-ldotoc + sqrt_discrim) / ldotl

    d1 = wp.clamp(d1, wp.float32(0.0), wp.float32(1.0))
    d2 = wp.clamp(d2, wp.float32(0.0), wp.float32(1.0))

    return (d2 - d1) * wp.length(segment)


@wp.kernel
def face_pair_dihedrals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_signed_angles: wp.array[wp.float32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    # Everything the mean-curvature measure needs per adjacent face pair, in one pass: the dihedral
    # angle signed by convexity, and the shared edge's AABB for the segment BVH. Each is also a
    # public per-pair table of its own, one launch apiece; every one of them is read here only at
    # its own pair, so taking them together writes no intermediate table.
    #
    # Each quantity is the public one exactly, spelled with the same helpers: the angle is
    # ``vector_angle`` of the two face normals (``adjacency.face_adjacency_angles``); convexity is
    # ``unshared_projection`` against ``TOLERANCE_MERGE`` (``adjacency.face_adjacency_convex``);
    # the bounds are ``segment_aabb`` (``edges.edge_aabb_bounds``).
    #
    # Folding the sign in is exact: ``ball_mean_curvature`` multiplies ``length * angle *
    # sign``, and scaling by ``-1`` commutes with rounding, so the product is bit-identical.
    #
    # The two face normals are derived here rather than read from a ``(n_faces,)`` table: with
    # ``face_normals_and_area`` -- the body of ``triangles.face_normals_and_areas`` -- they are that
    # table's entries bit for bit, and the wrapper then launches no normal pass and allocates no
    # normal or (unused) area table.
    k = wp.int32(wp.tid())
    normal_a, _area_a = face_normals_and_area(vertices, faces, face_adjacency[k, 0])
    normal_b, _area_b = face_normals_and_area(vertices, faces, face_adjacency[k, 1])
    angle = vector_angle(normal_a, normal_b)

    e0 = face_adjacency_edges[k, 0]
    e1 = face_adjacency_edges[k, 1]
    base = face_adjacency[k, 1] * 3
    other = unshared_vertex(faces[base], faces[base + 1], faces[base + 2], e0, e1)
    convex = unshared_projection(vertices, normal_a, e0, other) < TOLERANCE_MERGE_CONSTANT
    out_signed_angles[k] = wp.where(convex, angle, -angle)

    lower, upper = segment_aabb(vertices[e0], vertices[e1])
    out_lower[k] = lower
    out_upper[k] = upper


@wp.kernel
def ball_mean_curvature(
    queries: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    face_adjacency_edges: wp.array2d[wp.int32],
    signed_angles: wp.array[wp.float32],
    bvh_id: wp.uint64,
    radius: wp.float32,
    out_mean_curvature: wp.array[wp.float32],
) -> None:
    # The Cohen-Steiner/Morvan mean-curvature measure of the ball around one query: half the signed
    # dihedral angle of every adjacent face pair, weighted by the length of their shared edge inside
    # the ball, summed during the BVH walk over the edges' bounds. One sequential sum per query,
    # with no candidate list, count pass, scan, readback or per-candidate binary search and atomic.
    #
    # The ball query, not the circumscribed cube: re-timed after Warp 1.18 slowed the sphere walk
    # on point-like bounds, the cube walk (whose extra edges add exact zeros here) is still the
    # slower ``discrete_mean_curvature`` by 1.2-1.4x across the benchmark spheres and radii.
    q = wp.int32(wp.tid())
    center = queries[q]
    total = wp.float32(0.0)
    k = wp.int32(0)
    query = wp.bvh_query_sphere(bvh_id, center, radius)
    while wp.bvh_query_next(query, k):
        start_point = vertices[face_adjacency_edges[k, 0]]
        end_point = vertices[face_adjacency_edges[k, 1]]
        length = line_ball_intersection_segment(start_point, end_point, center, radius)
        # Positive across a convex edge, negative across a concave one; see ``face_pair_dihedrals``.
        total = total + length * signed_angles[k] * wp.float32(0.5)
    out_mean_curvature[q] = total
