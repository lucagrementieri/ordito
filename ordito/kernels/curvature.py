import warp as wp

from ordito.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from ordito.kernels.adjacency import unshared_projection, unshared_vertex
from ordito.kernels.algorithms.bfs import ball_row_bounds
from ordito.kernels.array import declare_map_signatures, map_probe
from ordito.kernels.halfedge import halfedge_destination, halfedge_prev
from ordito.kernels.linalg import solve_normal_equations
from ordito.kernels.neighbors import in_ball
from ordito.kernels.normals import gathered_angle_weighted_normal
from ordito.kernels.predicates import (
    angle_defect,
    any_perpendicular,
    reference_tangent,
    segment_aabb,
    vector_angle,
)
from ordito.kernels.scatter import peer_summed_add_corner_triple
from ordito.kernels.triangles import (
    face_corner_angles,
    face_normals_and_area,
    face_vertices_vec3d,
    local_corner,
)

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
    # The reference direction is the neighbour projected into the tangent plane, falling back to an
    # arbitrary perpendicular when that projection vanishes (the neighbour sits on the normal line
    # through the vertex).
    t1, _length = reference_tangent(first_neighbor - vertex, normal, TOLERANCE_ZERO_CONSTANT)
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
def monge_principal_curvature(
    vertices: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    start: wp.int32,
    end: wp.int32,
    i: wp.int32,
    reference: wp.int32,
    frame_independent: wp.bool,
) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.float32]:
    """
    ``(PD1, PD2, PV1, PV2)`` of point ``i``, a quadric fitted over ``neighbor_indices[start:end]``.

    The fit of ``igl::principal_curvature``: a Monge patch through the point in the tangent frame
    of its normal, the first axis toward point ``reference`` (any perpendicular when that projects
    to nothing, ``reference == i`` included). Entries equal to ``i`` are the point itself and
    negative entries are padding; neither enters the fit. A declined fit -- too few neighbours, a
    neighbourhood of coincident points, or a singular normal matrix -- returns all zeros, one
    answer for every exit, because a caller reads a zero ``PV1`` as "no fit here".
    """
    zero = wp.vec3()
    vertex = vertices[i]
    normal = wp.normalize(vertex_normals[i])

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
    present = wp.int32(0)
    n_valid = wp.int32(0)
    ring_radius = wp.float32(0.0)
    for k in range(start, end):
        j = neighbor_indices[k]
        if j < 0:
            continue
        present = present + 1
        if j == i:
            n_valid = n_valid + 1  # self always passes
            continue
        ring_radius = wp.max(ring_radius, wp.length(vertices[j] - vertex))
        if wp.dot(vertex_normals[j], normal) > wp.float32(0.0):
            n_valid = n_valid + 1

    # The neighbourhood includes the centre point itself, which the fit loop below skips, so a
    # determined 5-parameter quadric fit needs 6 entries. Anything short of that leaves the normal
    # equations rank-deficient and ``solve_normal_equations`` would report it -- this only saves
    # running a solve whose answer is already known. Every neighbour coinciding with the centre
    # leaves no frame and an exactly singular system: the same answer.
    if present < 6 or ring_radius <= wp.float32(0.0):
        return zero, zero, wp.float32(0.0), wp.float32(0.0)
    inv_radius = wp.float64(1.0) / wp.float64(ring_radius)

    # Build the tangent frame from the reference neighbour (the lowest-indexed mesh-adjacency
    # neighbor on a mesh, matching libigl's computeReferenceFrame, adjacency_list[i][0]). When
    # frame_independent is False, libigl extracts curvature from a *symmetrized* shape operator
    # whose eigenvalues depend on the chosen frame, so the principal values only agree with
    # igl::principal_curvature when this exact reference direction is used. When frame_independent
    # is True the eigenvalues are surface invariants, so the exact frame is irrelevant (any
    # orthonormal tangent basis yields the same result).
    t1, t2 = _build_reference_frame(vertex, normal, vertices[reference])

    # Mirror libigl: only apply the filter if it leaves at least 6 neighbours. libigl also requires
    # the filtered set to be strictly smaller than the full one, which is not repeated here because
    # it cannot change the answer -- when every neighbour passes, filtering removes nothing.
    use_filter = n_valid >= 6

    # Least-squares quadric fit: accumulate the normal equations AᵀA x = Aᵀb.
    ata = mat55d()
    atb = vec5d()

    for k in range(start, end):
        j = neighbor_indices[k]
        if j < 0 or j == i:
            continue  # padding, or self, which contributes (0,0,0) and would degenerate the frame
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
        return zero, zero, wp.float32(0.0), wp.float32(0.0)

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
        return dir0, dir1, k0, k1
    return dir1, dir0, k1, k0


# The two kernels below differ only in where a point's neighbourhood lies: a geodesic ball in
# either of ``ball_row_bounds``' layouts on a mesh, a fixed-width ``-1``-padded row on a cloud.


@wp.kernel
def fit_principal_curvature(
    vertices: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    row_stride: wp.int32,
    chunk_start: wp.int32,
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
    # The neighborhoods arrive in either of ``ball_row_bounds``' layouts; thread ``t`` fits vertex
    # ``chunk_start + t``.
    t = wp.int32(wp.tid())
    i = chunk_start + t
    start, end = ball_row_bounds(offsets, row_stride, i, t)
    pd1, pd2, pv1, pv2 = monge_principal_curvature(
        vertices,
        vertex_normals,
        neighbor_indices,
        start,
        end,
        i,
        reference_neighbors[i],
        frame_independent,
    )
    out_pd1[i] = pd1
    out_pd2[i] = pd2
    out_pv1[i] = pv1
    out_pv2[i] = pv2


@wp.kernel
def fit_point_principal_curvature(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    row_stride: wp.int32,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # One point a thread over its ``row_stride``-wide row of the flattened neighbour table. A
    # cloud has no adjacency to pick a reference neighbour from, so the frame takes the point
    # itself, i.e. any perpendicular: the fit is always the frame-independent one.
    i = wp.int32(wp.tid())
    start = i * row_stride
    pd1, pd2, pv1, pv2 = monge_principal_curvature(
        points, normals, neighbor_indices, start, start + row_stride, i, i, True
    )
    out_pd1[i] = pd1
    out_pd2[i] = pd2
    out_pv1[i] = pv1
    out_pv2[i] = pv2


# The 1-ring estimators below load a face's corners in ``float64``
# (``triangles.face_vertices_vec3d``): their cotangents and dihedrals lose every digit of a needle
# in ``float32``.


@wp.func
def corner_at(p0: wp.vec3d, p1: wp.vec3d, p2: wp.vec3d, i: wp.int32) -> wp.vec3d:
    # Corner ``i`` (taken mod 3) of a face's three positions, for a walk that rotates the corners.
    j = i % 3
    if j == 0:
        return p0
    if j == 1:
        return p1
    return p2


@wp.func
def unsigned_angle_d(u: wp.vec3d, w: wp.vec3d) -> wp.float64:
    # ``vcg::Angle`` / ``vtkMath::AngleBetweenVectors`` as the 1-ring estimators read it: the
    # clamped arccosine of the normalized dot product, and ``1`` where an edge has no length (vcg
    # returns ``-1`` there and every caller takes its absolute value).
    w2 = wp.length(u) * wp.length(w)
    if w2 == wp.float64(0.0):
        return wp.float64(1.0)
    return wp.acos(wp.clamp(wp.dot(u, w) / w2, wp.float64(-1.0), wp.float64(1.0)))


@wp.func
def extreme_curvatures(mean: wp.float64, gaussian: wp.float64) -> tuple[wp.float64, wp.float64]:
    # The principal curvatures a mean and a Gaussian curvature imply, ``H +- sqrt(H^2 - K)``, and
    # ``H`` twice where ``H^2 < K`` (no real pair): VTK's rule and Meyer et al.'s clamp alike.
    spread = mean * mean - gaussian
    first = mean
    second = mean
    if spread >= wp.float64(0.0):
        first = mean + wp.sqrt(spread)
        second = mean - wp.sqrt(spread)
    return first, second


@wp.func
def signed_dihedral_d(cross_f: wp.vec3d, cross_g: wp.vec3d, direction: wp.vec3d) -> wp.float64:
    # The bending angle at an edge shared by faces ``f`` and ``g``, from their unnormalized
    # normals: ``atan2((n_f x n_g) . e, n_f . n_g)``, ``e`` the edge's unit direction as ``f``
    # walks it, positive where the surface bends away from the normals (a convex edge of an
    # outward-wound solid). A face with no normal or an edge with no length reads zero. The rule
    # of VTK's mean curvature and of the normal-cycle tensor alike.
    normal_f = wp.vec3d()
    normal_g = wp.vec3d()
    if wp.length(cross_f) > wp.float64(0.0):
        normal_f = wp.normalize(cross_f)
    if wp.length(cross_g) > wp.float64(0.0):
        normal_g = wp.normalize(cross_g)
    length = wp.length(direction)
    cs = wp.dot(normal_f, normal_g)
    sn = wp.float64(0.0)
    if length > wp.float64(0.0):
        sn = wp.dot(wp.cross(normal_f, normal_g), direction / length)
    theta = wp.float64(0.0)
    if sn != wp.float64(0.0) or cs != wp.float64(0.0):
        theta = wp.atan2(sn, cs)
    return theta


@wp.func
def write_directionless_principal_curvature(
    v: wp.int32,
    first: wp.float64,
    second: wp.float64,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # The answer of a 1-ring method that forms no directions (VTK's, Meyer et al.'s): its extreme
    # curvatures rounded once, and zero ``PD1`` and ``PD2``.
    out_pd1[v] = wp.vec3()
    out_pd2[v] = wp.vec3()
    out_pv1[v] = wp.float32(first)
    out_pv2[v] = wp.float32(second)


@wp.kernel
def vtk_face_terms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    out_face_terms: wp.array2d[wp.float64],
) -> None:
    # Everything ``vtk_principal_curvatures`` reads of face ``f`` besides its corner angles: column
    # 0 its area, column ``1 + l`` the mean-curvature term of its halfedge ``l`` when ``f`` is the
    # lower face of that edge (zero otherwise, never read). The term is
    # ``3 |e| theta / (A_f + A_g)``, ``theta`` the dihedral signed by ``(n_f x n_g) . e`` with
    # ``e`` walked as ``f`` walks it. Each area was formed at each of the face's three corners and
    # each edge's term at both its ends: tabulated, bit-identical, kernel 0.43-0.64x on CUDA
    # (2.5 k to 870 k faces) and 0.76x on CPU, for 32 bytes a face.
    f = wp.int32(wp.tid())
    p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
    cross_f = wp.cross(p1 - p0, p2 - p0)
    area_f = wp.float64(0.5) * wp.length(cross_f)
    out_face_terms[f, 0] = area_f
    for local in range(3):
        mate = mates[3 * f + local]
        term = wp.float64(0.0)
        if mate >= 0 and mate // 3 > f:
            q0, q1, q2 = face_vertices_vec3d(vertices, faces, mate // 3)
            cross_g = wp.cross(q1 - q0, q2 - q0)
            direction = corner_at(p0, p1, p2, local + 1) - corner_at(p0, p1, p2, local)
            term = wp.length(direction) * signed_dihedral_d(cross_f, cross_g, direction)
            areas = area_f + wp.float64(0.5) * wp.length(cross_g)
            if areas != wp.float64(0.0):
                term = term / areas * wp.float64(3.0)
        out_face_terms[f, 1 + local] = term


@wp.kernel
def vtk_principal_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    face_terms: wp.array2d[wp.float64],
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # ``vtkCurvatures``' extreme curvatures, ``H +- sqrt(H^2 - K)`` (``H`` alone where that is
    # negative), over its 1-ring ``K`` and ``H``, gathered per vertex over its face row:
    # - ``K``: three times the angle defect over the summed incident face areas (no rim term);
    # - ``H``: every edge shared by exactly two faces adds ``3 |e| theta / (A_f + A_g)`` to both
    #   ends (``vtk_face_terms``), and a vertex takes half the mean over those edges. Each edge at
    #   the vertex is met from both its faces; it is counted from the lower one, as VTK counts it.
    # VTK forms no directions, so ``PD1`` and ``PD2`` are written zero here, sparing their fill.
    v = wp.int32(wp.tid())
    angle_sum = wp.float64(0.0)
    area_sum = wp.float64(0.0)
    mean_sum = wp.float64(0.0)
    edge_count = wp.int32(0)
    for slot in range(vf_offsets[v], vf_offsets[v + 1]):
        f = vertex_faces[slot]
        k = local_corner(faces, f, v)
        p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
        here = corner_at(p0, p1, p2, k)
        after = corner_at(p0, p1, p2, k + 1)
        before = corner_at(p0, p1, p2, k + 2)
        angle_sum += unsigned_angle_d(after - here, before - here)
        area_sum += face_terms[f, 0]
        # The two edges at ``v`` in this face: ``k -> k + 1`` and ``k + 2 -> k``.
        for side in range(2):
            local = wp.where(side == 0, k, (k + 2) % 3)
            mate = mates[3 * f + local]
            if mate >= 0 and mate // 3 > f:
                mean_sum += face_terms[f, 1 + local]
                edge_count += 1
    gaussian = wp.float64(0.0)
    if area_sum > wp.float64(0.0):
        gaussian = wp.float64(3.0) * (wp.float64(2.0) * wp.float64(wp.pi) - angle_sum) / area_sum
    mean = wp.float64(0.0)
    if edge_count > 0:
        mean = wp.float64(0.5) * mean_sum / wp.float64(edge_count)
    first, second = extreme_curvatures(mean, gaussian)
    write_directionless_principal_curvature(v, first, second, out_pd1, out_pd2, out_pv1, out_pv2)


@wp.func
def meyer_third_angle(a0: wp.float64, a1: wp.float64) -> wp.float64:
    # A face's third angle as vcg forms it, ``pi - a0 - a1``, rather than its own arccosine.
    return wp.float64(wp.pi) - (a0 + a1)


@wp.kernel
def meyer_face_terms(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_face_terms: wp.array2d[wp.float64]
) -> None:
    # The arccosines and tangents of ``meyer_corner_terms``, once a face: its angles ``a0`` and
    # ``a1`` (the third is ``meyer_third_angle``) and the tangents of all three. Each face is
    # walked by its three corners' vertices, and each walk formed these itself: tabulated, the
    # walk keeps only its corner's own divisions. Bit-identical to forming them in the walk;
    # kernel 0.50x on CUDA and CPU at 2.5 k to 870 k faces, for 40 bytes a face.
    f = wp.int32(wp.tid())
    p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
    a0 = unsigned_angle_d(p1 - p0, p2 - p0)
    a1 = unsigned_angle_d(p0 - p1, p2 - p1)
    out_face_terms[f, 0] = a0
    out_face_terms[f, 1] = a1
    out_face_terms[f, 2] = wp.tan(a0)
    out_face_terms[f, 3] = wp.tan(a1)
    out_face_terms[f, 4] = wp.tan(meyer_third_angle(a0, a1))


@wp.func
def meyer_corner_terms(
    p0: wp.vec3d,
    p1: wp.vec3d,
    p2: wp.vec3d,
    k: wp.int32,
    face_terms: wp.array2d[wp.float64],
    f: wp.int32,
) -> tuple[wp.float64, wp.vec3d, wp.float64, wp.bool]:
    # One face's share at its corner ``k`` of Meyer et al.'s operators, as vcg's
    # ``UpdateCurvature::MeanAndGaussian`` forms it: the mixed area (a non-obtuse face's
    # circumcentric part ``(|e_prev|^2 cot a_next + |e_next|^2 cot a_prev) / 8``; an obtuse face's
    # quarter of its area at the obtuse corner and an eighth at the others), the cotangent
    # Laplacian's term ``(e_prev cot a_next - e_next cot a_prev) / 4``, the corner's angle, and
    # whether the face is degenerate (an angle of exactly zero), which keeps its area but no other
    # term. The angles and their tangents are face ``f``'s row of ``meyer_face_terms``; ``e_prev``
    # runs from the previous corner to ``k``, ``e_next`` from ``k`` to the next.
    a0 = face_terms[f, 0]
    a1 = face_terms[f, 1]
    a2 = meyer_third_angle(a0, a1)
    here = corner_at(p0, p1, p2, k)
    after = corner_at(p0, p1, p2, k + 1)
    before = corner_at(p0, p1, p2, k + 2)
    angles = wp.vec3d(a0, a1, a2)
    a_here = angles[k]
    tan_next = face_terms[f, 2 + (k + 1) % 3]
    tan_prev = face_terms[f, 2 + (k + 2) % 3]
    e_prev = here - before
    e_next = after - here
    half = wp.float64(wp.pi) * wp.float64(0.5)
    double_area = wp.length(wp.cross(p1 - p0, p2 - p0))
    area = double_area / wp.float64(8.0)
    if a0 < half and a1 < half and a2 < half:
        area = (wp.dot(e_prev, e_prev) / tan_next + wp.dot(e_next, e_next) / tan_prev) / wp.float64(
            8.0
        )
    elif a_here >= half:
        area = double_area / wp.float64(4.0)
    degenerate = a0 == wp.float64(0.0) or a1 == wp.float64(0.0) or a2 == wp.float64(0.0)
    laplacian = wp.vec3d()
    if not degenerate:
        laplacian = (e_prev / tan_next - e_next / tan_prev) / wp.float64(4.0)
    return area, laplacian, a_here, degenerate


@wp.func
def meyer_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    face_terms: wp.array2d[wp.float64],
    v: wp.int32,
) -> tuple[wp.float64, wp.float64]:
    # The pointwise curvatures of Meyer et al. as vcg forms them, gathered per vertex over its face
    # row in ``float64``, each face's angle terms read off ``meyer_face_terms``:
    # ``K = (2 pi - sum of angles - rim angle) / A`` and ``H = +-|sum of Laplacian terms| / A``,
    # ``A`` the mixed area, the sign that of the terms' projection on the area-weighted vertex
    # normal (``-`` where it is not positive). The rim angle is vcg's: at a vertex with one
    # outgoing and one incoming rim edge, the angle between them (vcg takes it from a
    # non-degenerate face's rim edge). A vertex whose mixed area is at most ``DBL_EPSILON`` reads
    # zero for both.
    area = wp.float64(0.0)
    laplacian = wp.vec3d()
    gaussian = wp.float64(2.0) * wp.float64(wp.pi)
    normal = wp.vec3d()
    rim_out = wp.int32(-1)
    rim_in = wp.int32(-1)
    rim_outs = wp.int32(0)
    for slot in range(vf_offsets[v], vf_offsets[v + 1]):
        f = vertex_faces[slot]
        k = local_corner(faces, f, v)
        p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
        corner_area, corner_laplacian, angle, degenerate = meyer_corner_terms(
            p0, p1, p2, k, face_terms, f
        )
        area += corner_area
        normal += wp.cross(p1 - p0, p2 - p0)
        if not degenerate:
            laplacian += corner_laplacian
            gaussian -= angle
            h = 3 * f + k
            if mates[h] == -1:
                rim_out = halfedge_destination(faces, h)
                rim_outs += 1
            if mates[halfedge_prev(h)] == -1:
                rim_in = faces[halfedge_prev(h)]
    if rim_outs == 1 and rim_in >= 0:
        here = wp.vec3d(vertices[v])
        gaussian -= unsigned_angle_d(
            wp.vec3d(vertices[rim_out]) - here, wp.vec3d(vertices[rim_in]) - here
        )
    mean = wp.float64(0.0)
    if area > wp.float64(2.220446049250313e-16):
        sign = wp.where(
            wp.dot(laplacian, normal) > wp.float64(0.0), wp.float64(1.0), wp.float64(-1.0)
        )
        mean = sign * wp.length(laplacian) / area
        gaussian = gaussian / area
    else:
        gaussian = wp.float64(0.0)
    return mean, gaussian


@wp.kernel
def meyer_vertex_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    face_terms: wp.array2d[wp.float64],
    out_mean: wp.array[wp.float32],
    out_gaussian: wp.array[wp.float32],
) -> None:
    # ``curvature.vertex_curvatures``: ``meyer_curvatures`` as they are.
    # ``meyer_principal_curvatures`` is the same per-vertex walk writing the principal pair instead.
    v = wp.int32(wp.tid())
    mean, gaussian = meyer_curvatures(
        vertices, faces, vertex_faces, vf_offsets, mates, face_terms, v
    )
    out_mean[v] = wp.float32(mean)
    out_gaussian[v] = wp.float32(gaussian)


@wp.kernel
def meyer_principal_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    face_terms: wp.array2d[wp.float64],
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # ``principal_curvature(method="meyer")``: the extremes of ``meyer_curvatures``, rounded once,
    # and zero directions (Meyer et al. form none).
    v = wp.int32(wp.tid())
    mean, gaussian = meyer_curvatures(
        vertices, faces, vertex_faces, vf_offsets, mates, face_terms, v
    )
    first, second = extreme_curvatures(mean, gaussian)
    write_directionless_principal_curvature(v, first, second, out_pd1, out_pd2, out_pv1, out_pv2)


@wp.func
def write_no_principal_curvature(
    i: wp.int32,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # The answer a 1-ring tensor or an MLS kernel writes where it has none -- a ring of no area, no
    # interior edge, an MLS surface undefined at the point: zero directions and zero values, all
    # four written, so a caller reads a zero ``PD1`` as "nothing here".
    out_pd1[i] = wp.vec3()
    out_pd2[i] = wp.vec3()
    out_pv1[i] = wp.float32(0.0)
    out_pv2[i] = wp.float32(0.0)


@wp.func
def ring_angle_weighted_normal(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vertex_normals: wp.array[wp.vec3],
    gather: wp.int32,
    v: wp.int32,
) -> wp.vec3:
    # The angle-weighted vertex normal the 1-ring tensors orient by: ``vertex_normals[v]`` as
    # ``normals.vertex_normals(weighting="angle")`` scattered it, or, where ``gather`` is set,
    # ``normals.gathered_angle_weighted_normal`` over the face row the tensor walks (equal up to
    # summation order), sparing the accumulator, its fill, the scatter and the normalization. CUDA
    # gathers (Taubin 0.76-0.90x wall, normal cycles 0.90-1.00x, scrambled dragon included); the CPU
    # reads the scattered table, since the gather forms each face's angles at all three of its
    # corners (1.2-1.6x and 1.13-1.16x there).
    if gather != 0:
        return gathered_angle_weighted_normal(vertices, faces, vf_offsets, vertex_faces, v)
    return vertex_normals[v]


@wp.kernel
def taubin_principal_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vertex_normals: wp.array[wp.vec3],
    gather_normals: wp.int32,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # ``principal_curvature(method="taubin")``: Taubin's 1-ring tensor
    # ``M = sum_j w_j k_j T_j T_j^T`` over the edges at ``v``, gathered per vertex over its face
    # row. ``k_j = 2 n . (v - v_j) / |v - v_j|^2`` is the edge's normal curvature, ``T_j`` its
    # unit projection on the tangent plane, and ``w_j`` the area of the faces sharing the edge
    # over twice the ring's: each face adds its area to both its edges at ``v``, so a face pass
    # forms the edge weights without the edge list. ``M``'s eigenvectors in the tangent plane are
    # the principal directions and its eigenvalues ``m1 >= m2`` give ``k1 = 3 m1 - m2`` and
    # ``k2 = 3 m2 - m1``. An edge along the normal has no tangent and is skipped; a ring of no
    # area writes the zero answer. ``n`` is ``ring_angle_weighted_normal``.
    v = wp.int32(wp.tid())
    normal = wp.normalize(
        ring_angle_weighted_normal(
            vertices, faces, vertex_faces, vf_offsets, vertex_normals, gather_normals, v
        )
    )
    here = vertices[v]
    tensor = wp.mat33()
    ring_area = wp.float32(0.0)
    for slot in range(vf_offsets[v], vf_offsets[v + 1]):
        f = vertex_faces[slot]
        _, area = face_normals_and_area(vertices, faces, f)
        ring_area += area
        k = local_corner(faces, f, v)
        for side in range(1, 3):
            edge = here - vertices[faces[3 * f + (k + side) % 3]]
            length_sq = wp.dot(edge, edge)
            tangential = edge - wp.dot(edge, normal) * normal
            tangent_length = wp.length(tangential)
            if length_sq > wp.float32(0.0) and tangent_length > wp.float32(0.0):
                unit = tangential / tangent_length
                curvature = wp.float32(2.0) * wp.dot(normal, edge) / length_sq
                tensor += (area * curvature) * wp.outer(unit, unit)
    if ring_area <= wp.float32(0.0):
        write_no_principal_curvature(v, out_pd1, out_pd2, out_pv1, out_pv2)
        return
    tensor = tensor * (wp.float32(0.5) / ring_area)
    t1 = any_perpendicular(normal)
    t2 = wp.cross(normal, t1)
    m00 = wp.dot(t1, tensor * t1)
    m01 = wp.dot(t1, tensor * t2)
    m11 = wp.dot(t2, tensor * t2)
    half_trace = (m00 + m11) * wp.float32(0.5)
    half_diff = (m00 - m11) * wp.float32(0.5)
    spread = wp.sqrt(half_diff * half_diff + m01 * m01)
    larger = half_trace + spread
    smaller = half_trace - spread
    first = _eigvec_2x2(m00, m01, m01, m11, larger)
    pd1 = wp.normalize(t1 * first[0] + t2 * first[1])
    out_pd1[v] = pd1
    out_pd2[v] = wp.cross(normal, pd1)
    out_pv1[v] = wp.float32(3.0) * larger - smaller
    out_pv2[v] = wp.float32(3.0) * smaller - larger


@wp.kernel
def normal_cycle_principal_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    vertex_normals: wp.array[wp.vec3],
    gather_normals: wp.int32,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # ``principal_curvature(method="normal_cycles")``: Cohen-Steiner and Morvan's tensor over the
    # vertex's barycentric cell, ``T = sum_e beta_e |e| e e^T / (2 A)``, gathered per vertex over
    # its face row in ``float64``. Each interior edge at ``v`` is the halfedge leaving ``v`` in
    # exactly one face, so it is met once; ``beta_e`` is its signed dihedral
    # (``signed_dihedral_d``), half of it lies in the cell, and ``A`` is a third of the ring's area.
    # A rim edge bends nothing and adds nothing. vcg's ``PrincipalDirectionsNormalCycle`` sums the
    # same ``beta_e |e| e e^T`` (its ``beta_e`` an arcsine, equal below a right-angle fold) and
    # divides by nothing. Of ``T``'s three eigenvectors the one nearest the vertex normal is the
    # normal; the other two carry the principal curvatures crosswise: the larger eigenvalue is the
    # curvature *across* its eigenvector (a cylinder's ridge edges run along the axis), so ``PD1``
    # is the smaller eigenvalue's eigenvector.
    v = wp.int32(wp.tid())
    tensor = wp.mat33d()
    ring_area = wp.float64(0.0)
    edges = wp.int32(0)
    for slot in range(vf_offsets[v], vf_offsets[v + 1]):
        f = vertex_faces[slot]
        k = local_corner(faces, f, v)
        p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
        cross_f = wp.cross(p1 - p0, p2 - p0)
        ring_area += wp.float64(0.5) * wp.length(cross_f)
        mate = mates[3 * f + k]
        if mate >= 0:
            q0, q1, q2 = face_vertices_vec3d(vertices, faces, mate // 3)
            direction = corner_at(p0, p1, p2, k + 1) - corner_at(p0, p1, p2, k)
            length = wp.length(direction)
            if length > wp.float64(0.0):
                beta = signed_dihedral_d(cross_f, wp.cross(q1 - q0, q2 - q0), direction)
                unit = direction / length
                tensor += (beta * length) * wp.outer(unit, unit)
                edges += 1
    if edges == 0 or ring_area <= wp.float64(0.0):
        write_no_principal_curvature(v, out_pd1, out_pd2, out_pv1, out_pv2)
        return
    tensor = tensor * (wp.float64(1.5) / ring_area)
    vectors, values = wp.eig3(tensor)
    vertex_normal = ring_angle_weighted_normal(
        vertices, faces, vertex_faces, vf_offsets, vertex_normals, gather_normals, v
    )
    normal = wp.vec3d(vertex_normal)
    across = wp.int32(0)
    best = wp.float64(-1.0)
    for i in range(3):
        alignment = wp.abs(wp.dot(wp.vec3d(vectors[0, i], vectors[1, i], vectors[2, i]), normal))
        if alignment > best:
            best = alignment
            across = i
    a = (across + 1) % 3
    b = (across + 2) % 3
    larger = wp.max(values[a], values[b])
    smaller = wp.min(values[a], values[b])
    small = wp.where(values[a] < values[b], a, b)
    pd1 = wp.vec3(wp.vec3d(vectors[0, small], vectors[1, small], vectors[2, small]))
    axis = wp.vec3(wp.vec3d(vectors[0, across], vectors[1, across], vectors[2, across]))
    if wp.dot(axis, vertex_normal) < wp.float32(0.0):
        axis = -axis
    out_pd1[v] = pd1
    out_pd2[v] = wp.cross(axis, pd1)
    out_pv1[v] = wp.float32(larger)
    out_pv2[v] = wp.float32(smaller)


# ---------------------------------------------------------------------------
# vertex_defects: a full turn less the corner angles incident to each vertex
# ---------------------------------------------------------------------------


@wp.func
def angle_defect_rounded(angle_sum: wp.float64) -> wp.float32:
    """``angle_defect`` of a ``float64`` angle sum, formed in ``float64`` and rounded once."""
    return wp.float32(angle_defect(angle_sum))


@wp.kernel
def scatter_corner_angles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_angle_sums: wp.array[wp.float32]
) -> None:
    # Each vertex's sum of incident corner angles, the angles formed in the thread
    # (``triangles.face_corner_angles``, the ``angles`` kernel's body) and added corner by corner in
    # ``scatter.scatter_sum_scalar``'s order, so no ``(n_faces, 3)`` angle table is written. The
    # defect's angle sums from positions, for a caller holding no angle table
    # (``smoothing.filter_spikes``, which tests the defect against its threshold).
    f = wp.int32(wp.tid())
    a0, a1, a2 = face_corner_angles(vertices, faces, f)
    peer_summed_add_corner_triple(out_angle_sums, faces, f, a0, a1, a2)


# ---------------------------------------------------------------------------
# discrete_gaussian_curvature: the defects summed over a ball
# ---------------------------------------------------------------------------


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
    # defect is ``predicates.angle_defect``, the rule ``curvature.vertex_defects`` applies.
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


# ---------------------------------------------------------------------------
# mls_principal_curvature: the MLS surfaces of Guennebaud and Gross (APSS) and of Oztireli,
# Guennebaud and Gross (RIMLS), as MeshLab's ``filter_mls`` (Gael Guennebaud's ``GaelMls``) forms
# them, all in ``float64``. A point's neighbourhood is every sample whose own ball covers it (a
# sample's ball radius is ``filter_scale`` times its spacing), weighted by
# ``(1 - d^2 / r^2)^4``; the BVH holds those balls' boxes, so one point query walks exactly the
# candidates.
# ---------------------------------------------------------------------------

# How many steps a projection onto the MLS surface takes at most (MeshLab's default), and the step
# below which it stops, as a fraction of the point's reach. MeshLab's stopping distance is its
# ``ProjectionAccuracy`` times a point spacing it never sets (zero in a fresh process, so it runs
# every step), and the curvature is sensitive to where the projection lands: stopping at 1e-4 of
# the spacing (its documented intent) moved ``sphere_irregular``'s fields by 8e-06. At 1e-10 the
# answer is MeshLab's to ``float32`` rounding and most points stop after a few steps (15 steps
# cost 2.5x three on a 2 562-point sphere).
MLS_PROJECTION_ITERATIONS = 15
MLS_PROJECTION_TOLERANCE = wp.float64(1e-10)
# APSS's spherical parameter: 1 fits pure spheres (MeshLab's default).
APSS_SPHERICAL_PARAMETER = wp.float64(1.0)
# RIMLS's normal sharpness ``sigma_n`` (MeshLab's default) and its refitting rounds.
RIMLS_SIGMA_N = wp.float64(0.75)
RIMLS_MAX_REFITS = 3
RIMLS_REFIT_THRESHOLD = wp.float64(1e-3)
# vcg's APSS status: a sphere, a plane, or a sphere too flat to place.
APSS_SPHERE = 0
APSS_PLANE = 1
APSS_UNDETERMINED = 2


@wp.kernel
def mls_sample_balls(
    points: wp.array[wp.vec3],
    neighbor_indices: wp.array2d[wp.int32],
    filter_scale: wp.float64,
    out_radii: wp.array[wp.float64],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    # Each sample's ball: ``filter_scale`` times its spacing ``2 sqrt(d_k^2 / k)``, ``d_k`` the
    # distance to the farthest of its ``k`` nearest samples (itself included), as vcg's
    # ``computeVertexRadius`` sets it; the distance re-formed in ``float64``. The box is widened by
    # a few ulp so the ``float32`` box test never drops a sample the ``float64`` ball test keeps.
    i = wp.int32(wp.tid())
    here = wp.vec3d(points[i])
    count = wp.int32(0)
    farthest = wp.float64(0.0)
    for slot in range(neighbor_indices.shape[1]):
        j = neighbor_indices[i, slot]
        if j >= 0:
            count += 1
            farthest = wp.max(farthest, wp.length_sq(wp.vec3d(points[j]) - here))
    radius = wp.float64(0.0)
    if count > 0:
        radius = filter_scale * (wp.float64(2.0) * wp.sqrt(farthest / wp.float64(count)))
    out_radii[i] = radius
    reach = wp.float32(radius) * wp.float32(1.0001) + wp.float32(1e-30)
    margin = wp.vec3(reach, reach, reach)
    out_lower[i] = points[i] - margin
    out_upper[i] = points[i] + margin


@wp.func
def mls_weight(
    x: wp.vec3d, sample: wp.vec3d, radius: wp.float64
) -> tuple[wp.bool, wp.float64, wp.float64, wp.float64]:
    # Whether ``sample``'s ball covers ``x``, and its weight ``(1 - d^2 s)^4`` with ``s = 1 / r^2``
    # and the weight's first and second derivatives in ``d^2`` scaled as vcg caches them: the
    # weight's gradient is ``(x - sample) * first`` and the ``(j, k)`` Hessian entry
    # ``(x - sample)_j (x - sample)_k second + [j == k] first``.
    d2 = wp.length_sq(x - sample)
    inv = wp.float64(1.0) / radius
    s = inv * inv
    aux = wp.max(wp.float64(1.0) - d2 * s, wp.float64(0.0))
    square = aux * aux
    weight = square * square
    first = wp.float64(-2.0) * s * (wp.float64(4.0) * aux * square)
    second = wp.float64(4.0) * s * s * (wp.float64(12.0) * square)
    return d2 < radius * radius, weight, first, second


@wp.func
def apss_moments(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radii: wp.array[wp.float64],
    bvh_id: wp.uint64,
    x: wp.vec3d,
) -> tuple[wp.int32, wp.int32, wp.vec3d, wp.vec3d, wp.float64, wp.float64, wp.float64]:
    # The weighted moments APSS fits at ``x``: how many samples' balls cover it (and one of them),
    # ``sum w p``, ``sum w n``, ``sum w n . p``, ``sum w p . p`` and ``sum w``.
    count = wp.int32(0)
    first = wp.int32(-1)
    sum_p = wp.vec3d()
    sum_n = wp.vec3d()
    sum_pn = wp.float64(0.0)
    sum_pp = wp.float64(0.0)
    sum_w = wp.float64(0.0)
    probe = wp.vec3(x)
    query = wp.bvh_query_aabb(bvh_id, probe, probe)
    i = wp.int32(0)
    while wp.bvh_query_next(query, i):
        p = wp.vec3d(points[i])
        inside, w, _first, _second = mls_weight(x, p, radii[i])
        if inside:
            n = wp.vec3d(normals[i])
            count += 1
            first = i
            sum_p += p * w
            sum_n += n * w
            sum_pn += w * wp.dot(n, p)
            sum_pp += w * wp.dot(p, p)
            sum_w += w
    return count, first, sum_p, sum_n, sum_pn, sum_pp, sum_w


@wp.func
def apss_sphere(
    sum_p: wp.vec3d, sum_n: wp.vec3d, sum_pn: wp.float64, sum_pp: wp.float64, sum_w: wp.float64
) -> tuple[wp.vec3d, wp.float64, wp.float64, wp.int32, wp.vec3d, wp.float64]:
    # vcg's ``APSS::fit`` from two or more samples' moments: the algebraic sphere
    # ``u0 + u13 . x + u4 |x|^2`` and its status, normalized as vcg normalizes a plane or a sphere
    # too flat to place, and the centre and radius of a placed one.
    inv = wp.float64(1.0) / sum_w
    quad = (
        APSS_SPHERICAL_PARAMETER
        * wp.float64(0.5)
        * (sum_pn - inv * wp.dot(sum_p, sum_n))
        / (sum_pp - inv * wp.dot(sum_p, sum_p))
    )
    linear = (sum_n - sum_p * (wp.float64(2.0) * quad)) * inv
    constant = -inv * (wp.dot(linear, sum_p) + sum_pp * quad)
    status = APSS_UNDETERMINED
    center = wp.vec3d()
    radius = wp.float64(0.0)
    if wp.abs(quad) > wp.float64(1e-7):
        status = APSS_SPHERE
        b = wp.float64(1.0) / quad
        center = linear * (wp.float64(-0.5) * b)
        radius = wp.sqrt(wp.dot(center, center) - b * constant)
    elif quad == wp.float64(0.0):
        status = APSS_PLANE
        s = wp.float64(1.0) / wp.length(linear)
        linear = linear * s
        constant = constant * s
    else:
        f = wp.float64(1.0) / wp.sqrt(wp.dot(linear, linear) - wp.float64(4.0) * constant * quad)
        constant = constant * f
        linear = linear * f
        quad = quad * f
    return linear, constant, quad, status, center, radius


@wp.func
def apss_project(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radii: wp.array[wp.float64],
    bvh_id: wp.uint64,
    x: wp.vec3d,
    tolerance: wp.float64,
) -> wp.vec3d:
    # vcg's ``APSS::project``: refit at the moving point and move ``x`` (the start, every round)
    # onto that sphere or plane, or by three Newton steps onto an undetermined one, until a round
    # moves the point by at most ``tolerance``. A point no sample covers stays where it is.
    position = x
    for _ in range(MLS_PROJECTION_ITERATIONS):
        count, first, sum_p, sum_n, sum_pn, sum_pp, sum_w = apss_moments(
            points, normals, radii, bvh_id, position
        )
        if count == 0:
            return x
        previous = position
        if count == 1:
            n = wp.vec3d(normals[first])
            position = x - n * (wp.dot(x, n) - wp.dot(wp.vec3d(points[first]), n))
        else:
            linear, constant, quad, status, center, radius = apss_sphere(
                sum_p, sum_n, sum_pn, sum_pp, sum_w
            )
            if status == APSS_SPHERE:
                position = center + wp.normalize(x - center) * radius
            elif status == APSS_PLANE:
                position = x - linear * (wp.dot(x, linear) + constant)
            else:
                direction = linear + x * (wp.float64(2.0) * quad)
                inverse_length = wp.float64(1.0) / wp.length(direction)
                direction = direction * inverse_length
                value = constant + wp.dot(linear, x) + quad * wp.dot(x, x)
                step = x + direction * (-value * wp.min(inverse_length, wp.float64(1.0)))
                for _newton in range(2):
                    inverse_length = wp.float64(1.0) / wp.length(
                        linear + step * (wp.float64(2.0) * quad)
                    )
                    value = constant + wp.dot(linear, step) + quad * wp.dot(step, step)
                    step = step + direction * (-value * wp.min(inverse_length, wp.float64(1.0)))
                position = step
        if wp.length_sq(previous - position) <= tolerance * tolerance:
            break
    return position


@wp.func
def implicit_principal_curvatures(
    gradient: wp.vec3d, hessian: wp.mat33d
) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.float32]:
    # The principal pair of the level set through a point of an implicit function, from its
    # gradient and Hessian there: the shape operator is the Hessian projected on the tangent plane
    # over the gradient's length, a symmetric 2x2 in any tangent frame, and its eigenpairs are the
    # curvatures (positive where the level set bends away from the gradient) and their directions.
    # vcg's ``implicits::WeingartenMap`` forms the same operator as ``(I - n n^T) H / |g|``; its
    # trace and determinant give the same pair, which it orders by magnitude.
    length = wp.length(gradient)
    normal = gradient / length
    t1 = any_perpendicular(normal)
    t2 = wp.cross(normal, t1)
    symmetric = (hessian + wp.transpose(hessian)) * wp.float64(0.5)
    m00 = wp.dot(t1, symmetric * t1) / length
    m01 = wp.dot(t1, symmetric * t2) / length
    m11 = wp.dot(t2, symmetric * t2) / length
    half_trace = (m00 + m11) * wp.float64(0.5)
    half_diff = (m00 - m11) * wp.float64(0.5)
    spread = wp.sqrt(half_diff * half_diff + m01 * m01)
    larger = half_trace + spread
    first = _eigvec_2x2(
        wp.float32(m00), wp.float32(m01), wp.float32(m01), wp.float32(m11), wp.float32(larger)
    )
    pd1 = wp.normalize(t1 * wp.float64(first[0]) + t2 * wp.float64(first[1]))
    pd2 = wp.cross(normal, pd1)
    return wp.vec3(pd1), wp.vec3(pd2), wp.float32(larger), wp.float32(half_trace - spread)


@wp.kernel
def apss_principal_curvature(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radii: wp.array[wp.float64],
    bvh_id: wp.uint64,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # ``mls_principal_curvature(method="apss")``: project the point onto the APSS surface, refit
    # there, and take vcg's ``mlsGradient`` and ``mlsHessian`` -- the derivatives of the fitted
    # sphere's value at the point with the fit's own dependence on the point carried through every
    # moment -- to ``implicit_principal_curvatures``. The gradient walk gathers each moment's three
    # partial derivatives, and each of three Hessian walks the second derivatives of one row.
    i = wp.int32(wp.tid())
    start = wp.vec3d(points[i])
    x = apss_project(points, normals, radii, bvh_id, start, MLS_PROJECTION_TOLERANCE * radii[i])
    count, _first, sum_p, sum_n, sum_pn, sum_pp, sum_w = apss_moments(
        points, normals, radii, bvh_id, x
    )
    if count < 2:
        write_no_principal_curvature(i, out_pd1, out_pd2, out_pv1, out_pv2)
        return
    linear, constant, quad, _status, _center, _radius = apss_sphere(
        sum_p, sum_n, sum_pn, sum_pp, sum_w
    )
    inv = wp.float64(1.0) / sum_w
    nume = sum_pn - inv * wp.dot(sum_p, sum_n)
    deno = sum_pp - inv * wp.dot(sum_p, sum_p)

    # First derivatives of the moments, one row (or entry) per axis ``k``.
    d_p = wp.mat33d()
    d_n = wp.mat33d()
    d_w = wp.vec3d()
    d_pn = wp.vec3d()
    d_pp = wp.vec3d()
    probe = wp.vec3(x)
    query = wp.bvh_query_aabb(bvh_id, probe, probe)
    j = wp.int32(0)
    while wp.bvh_query_next(query, j):
        p = wp.vec3d(points[j])
        inside, _weight, first, _second = mls_weight(x, p, radii[j])
        if inside:
            n = wp.vec3d(normals[j])
            dw = (x - p) * first
            d_w += dw
            d_p += wp.outer(dw, p)
            d_n += wp.outer(dw, n)
            d_pn += dw * wp.dot(n, p)
            d_pp += dw * wp.dot(p, p)

    # The fit's first derivatives and the gradient, as ``APSS::mlsGradient`` forms them.
    d_nume = wp.vec3d()
    d_deno = wp.vec3d()
    d_quad = wp.vec3d()
    d_linear = wp.mat33d()
    d_constant = wp.vec3d()
    gradient = wp.vec3d()
    for k in range(3):
        dp = wp.vec3d(d_p[k, 0], d_p[k, 1], d_p[k, 2])
        dn = wp.vec3d(d_n[k, 0], d_n[k, 1], d_n[k, 2])
        nk = d_pn[k] - inv * inv * (
            sum_w * (wp.dot(dp, sum_n) + wp.dot(sum_p, dn)) - d_w[k] * wp.dot(sum_p, sum_n)
        )
        dk = d_pp[k] - inv * inv * (
            wp.float64(2.0) * sum_w * wp.dot(dp, sum_p) - d_w[k] * wp.dot(sum_p, sum_p)
        )
        qk = APSS_SPHERICAL_PARAMETER * wp.float64(0.5) * (deno * nk - dk * nume) / (deno * deno)
        lk = ((dn - (dp * quad + sum_p * qk) * wp.float64(2.0)) - linear * d_w[k]) * inv
        ck = -inv * (
            wp.dot(lk, sum_p)
            + qk * sum_pp
            + wp.dot(linear, dp)
            + quad * d_pp[k]
            + d_w[k] * constant
        )
        d_nume[k] = nk
        d_deno[k] = dk
        d_quad[k] = qk
        d_constant[k] = ck
        for c in range(3):
            d_linear[k, c] = lk[c]
        gradient[k] = (
            ck + wp.dot(lk, x) + qk * wp.dot(x, x) + linear[k] + wp.float64(2.0) * x[k] * quad
        )
    if wp.length(gradient) <= wp.float64(1e-8):
        write_no_principal_curvature(i, out_pd1, out_pd2, out_pv1, out_pv2)
        return

    # Second derivatives, one row ``r`` of the Hessian a walk, as ``APSS::mlsHessian`` forms them
    # (its ``j``; ``k`` the column).
    hessian = wp.mat33d()
    deno2 = deno * deno
    inv4 = inv * inv * inv * inv
    for r in range(3):
        d2_p = wp.mat33d()
        d2_n = wp.mat33d()
        d2_w = wp.vec3d()
        d2_pn = wp.vec3d()
        d2_pp = wp.vec3d()
        query = wp.bvh_query_aabb(bvh_id, probe, probe)
        while wp.bvh_query_next(query, j):
            p = wp.vec3d(points[j])
            inside, _weight, first, second = mls_weight(x, p, radii[j])
            if inside:
                n = wp.vec3d(normals[j])
                d2w = (x - p) * ((x[r] - p[r]) * second)
                d2w[r] = d2w[r] + first
                d2_w += d2w
                d2_p += wp.outer(d2w, p)
                d2_n += wp.outer(d2w, n)
                d2_pn += d2w * wp.dot(n, p)
                d2_pp += d2w * wp.dot(p, p)
        dp_r = wp.vec3d(d_p[r, 0], d_p[r, 1], d_p[r, 2])
        dn_r = wp.vec3d(d_n[r, 0], d_n[r, 1], d_n[r, 2])
        dl_r = wp.vec3d(d_linear[r, 0], d_linear[r, 1], d_linear[r, 2])
        for k in range(3):
            dp = wp.vec3d(d_p[k, 0], d_p[k, 1], d_p[k, 2])
            dn = wp.vec3d(d_n[k, 0], d_n[k, 1], d_n[k, 2])
            dl = wp.vec3d(d_linear[k, 0], d_linear[k, 1], d_linear[k, 2])
            d2p = wp.vec3d(d2_p[k, 0], d2_p[k, 1], d2_p[k, 2])
            d2n = wp.vec3d(d2_n[k, 0], d2_n[k, 1], d2_n[k, 2])
            d2_nume = d2_pn[k] - inv4 * (
                wp.float64(-2.0)
                * sum_w
                * d_w[r]
                * (sum_w * (wp.dot(dp, sum_n) + wp.dot(sum_p, dn)) - d_w[k] * wp.dot(sum_p, sum_n))
                + sum_w
                * sum_w
                * (
                    d_w[r] * (wp.dot(dp, sum_n) + wp.dot(sum_p, dn))
                    + sum_w
                    * (
                        wp.dot(d2p, sum_n)
                        + wp.dot(sum_p, d2n)
                        + wp.dot(dp_r, dn)
                        + wp.dot(dp, dn_r)
                    )
                    - d2_w[k] * wp.dot(sum_p, sum_n)
                    - d_w[k] * (wp.dot(dp_r, sum_n) + wp.dot(sum_p, dn_r))
                )
            )
            d2_deno = d2_pp[k] - inv4 * (
                wp.float64(-2.0)
                * sum_w
                * d_w[r]
                * (wp.float64(2.0) * sum_w * wp.dot(dp, sum_p) - d_w[k] * wp.dot(sum_p, sum_p))
                + sum_w
                * sum_w
                * (
                    wp.float64(2.0) * d_w[r] * wp.dot(dp, sum_p)
                    + wp.float64(2.0) * sum_w * (wp.dot(dp_r, dp) + wp.dot(d2p, sum_p))
                    - d2_w[k] * wp.dot(sum_p, sum_p)
                    - d_w[k] * (wp.float64(2.0) * wp.dot(dp_r, sum_p))
                )
            )
            d2_quad = (
                APSS_SPHERICAL_PARAMETER
                * wp.float64(0.5)
                * (
                    deno2
                    * (
                        d2_nume * deno
                        + d_deno[r] * d_nume[k]
                        - d2_deno * nume
                        - d_deno[k] * d_nume[r]
                    )
                    - wp.float64(2.0) * deno * d_deno[r] * (deno * d_nume[k] - d_deno[k] * nume)
                )
                / (deno2 * deno2)
            )
            d2_linear = (
                -dl * d_w[r]
                + (
                    d2n
                    - (dp * d_quad[r] + d2p * quad + sum_p * d2_quad + dp_r * d_quad[k])
                    * wp.float64(2.0)
                )
                - linear * d2_w[k]
                - dl_r * d_w[k]
            ) * inv
            d2_constant = (
                -d_constant[k] * d_w[r]
                - (
                    wp.dot(dl, dp_r)
                    + wp.dot(d2_linear, sum_p)
                    + d2_quad * sum_pp
                    + d_quad[k] * d_pp[r]
                    + wp.dot(linear, d2p)
                    + wp.dot(dl_r, dp)
                    + d_pp[k] * d_quad[r]
                    + d2_pp[k] * quad
                    + d2_w[k] * constant
                    + d_w[k] * d_constant[r]
                )
            ) * inv
            diagonal = wp.where(r == k, wp.float64(2.0) * quad, wp.float64(0.0))
            hessian[r, k] = (
                dl[r]
                + wp.float64(2.0) * d_quad[k] * x[r]
                + d2_constant
                + wp.dot(d2_linear, x)
                + d2_quad * wp.dot(x, x)
                + dl_r[k]
                + diagonal
                + wp.float64(2.0) * x[k] * d_quad[r]
            )
    pd1, pd2, pv1, pv2 = implicit_principal_curvatures(gradient, hessian)
    out_pd1[i] = pd1
    out_pd2[i] = pd2
    out_pv1[i] = pv1
    out_pv2[i] = pv2


@wp.func
def rimls_potential(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radii: wp.array[wp.float64],
    bvh_id: wp.uint64,
    x: wp.vec3d,
) -> tuple[wp.bool, wp.float64, wp.vec3d, wp.vec3d, wp.float64, wp.vec3d, wp.bool]:
    # vcg's ``RIMLS::computePotentialAndGradient``: the robust implicit MLS value and gradient at
    # ``x``, refitted while the gradient moves (at least once, at most ``RIMLS_MAX_REFITS`` times),
    # each refit down-weighting samples whose normal differs from the previous gradient by
    # ``exp(-|n - g|^2 / sigma_n^2)``. Returns whether any sample weighs, the value, the gradient,
    # ``sum grad w`` and ``sum w`` of the last round, and the gradient that round weighed by
    # (``refitted`` false when the last round was the unweighted first).
    inverse_sigma2 = wp.float64(1.0) / (RIMLS_SIGMA_N * RIMLS_SIGMA_N)
    gradient = wp.vec3d()
    previous = wp.vec3d()
    potential = wp.float64(0.0)
    sum_gw = wp.vec3d()
    sum_w = wp.float64(0.0)
    refitted = wp.bool(False)
    probe = wp.vec3(x)
    for refit_round in range(RIMLS_MAX_REFITS):
        previous = gradient
        refitted = refit_round > 0
        sum_gw = wp.vec3d()
        sum_gwf = wp.vec3d()
        sum_n = wp.vec3d()
        potential = wp.float64(0.0)
        sum_w = wp.float64(0.0)
        query = wp.bvh_query_aabb(bvh_id, probe, probe)
        i = wp.int32(0)
        while wp.bvh_query_next(query, i):
            p = wp.vec3d(points[i])
            inside, w, first, _second = mls_weight(x, p, radii[i])
            if inside:
                n = wp.vec3d(normals[i])
                f = wp.dot(x - p, n)
                refit = wp.float64(1.0)
                if refitted:
                    refit = wp.exp(-wp.length_sq(n - previous) * inverse_sigma2)
                gw = (x - p) * (first * refit)
                sum_gw += gw
                sum_gwf += gw * f
                sum_n += n * (w * refit)
                potential += w * refit * f
                sum_w += w * refit
        if sum_w == wp.float64(0.0):
            return False, potential, gradient, sum_gw, sum_w, previous, refitted
        potential = potential / sum_w
        gradient = (-sum_gw * potential + sum_gwf + sum_n) * (wp.float64(1.0) / sum_w)
        if wp.length_sq(gradient - previous) <= RIMLS_REFIT_THRESHOLD:
            break
    return True, potential, gradient, sum_gw, sum_w, previous, refitted


@wp.kernel
def rimls_principal_curvature(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radii: wp.array[wp.float64],
    bvh_id: wp.uint64,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
) -> None:
    # ``mls_principal_curvature(method="rimls")``: vcg's ``RIMLS::project`` (step along the
    # normalized gradient by the value until the value is within tolerance), the value and gradient
    # refitted at the projection, and ``RIMLS::mlsHessian`` -- the gradient's derivative with the
    # last round's refitting weights held fixed -- to ``implicit_principal_curvatures``. vcg leaves
    # the refitting weight off the weight's own second-derivative term on the diagonal; it is
    # applied to every term here, so the Hessian is the derivative of the gradient it pairs with.
    i = wp.int32(wp.tid())
    start = wp.vec3d(points[i])
    tolerance = MLS_PROJECTION_TOLERANCE * radii[i]
    x = start
    for _ in range(MLS_PROJECTION_ITERATIONS):
        ok, potential, gradient, _sum_gw, _sum_w, _previous, _refitted = rimls_potential(
            points, normals, radii, bvh_id, x
        )
        if not ok:
            x = start
            break
        x = x - wp.normalize(gradient) * potential
        if wp.abs(potential) <= tolerance:
            break
    ok, potential, gradient, sum_gw, sum_w, previous, refitted = rimls_potential(
        points, normals, radii, bvh_id, x
    )
    if not ok or wp.length(gradient) <= wp.float64(1e-8):
        write_no_principal_curvature(i, out_pd1, out_pd2, out_pv1, out_pv2)
        return
    inverse_sigma2 = wp.float64(1.0) / (RIMLS_SIGMA_N * RIMLS_SIGMA_N)
    # Column ``k`` of each: ``sum d(grad w)/dx_k``, ``sum n (grad w)_k``, ``sum grad w n_k`` and
    # ``sum d(grad w)/dx_k f``.
    d_gw = wp.mat33d()
    n_gw = wp.mat33d()
    gw_n = wp.mat33d()
    d_gwf = wp.mat33d()
    probe = wp.vec3(x)
    query = wp.bvh_query_aabb(bvh_id, probe, probe)
    j = wp.int32(0)
    while wp.bvh_query_next(query, j):
        p = wp.vec3d(points[j])
        inside, _weight, first, second = mls_weight(x, p, radii[j])
        if inside:
            n = wp.vec3d(normals[j])
            refit = wp.float64(1.0)
            if refitted:
                refit = wp.exp(-wp.length_sq(n - previous) * inverse_sigma2)
            diff = x - p
            f = wp.dot(diff, n)
            gw = diff * (first * refit)
            # ``d(grad w)/dx_k = (diff diff_k second + e_k first) refit``, all k at once.
            d_grad = wp.outer(diff, diff) * (second * refit) + wp.identity(
                n=3, dtype=wp.float64
            ) * (first * refit)
            d_gw += d_grad
            n_gw += wp.outer(n, gw)
            gw_n += wp.outer(gw, n)
            d_gwf += d_grad * f
    hessian = (
        d_gw * (-potential)
        + n_gw
        + gw_n
        + d_gwf
        - wp.outer(sum_gw, gradient)
        - wp.outer(gradient, sum_gw)
    ) * (wp.float64(1.0) / sum_w)
    pd1, pd2, pv1, pv2 = implicit_principal_curvatures(gradient, hessian)
    out_pd1[i] = pd1
    out_pd2[i] = pd2
    out_pv1[i] = pv1
    out_pv2[i] = pv2


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense = map_probe
    declare_map_signatures([(angle_defect_rounded, (dense(wp.float64),), wp.float32)])


_declare_map_kernels()
