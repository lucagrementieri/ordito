"""
Discrete curvature: principal directions and magnitudes, Gaussian and mean.

Three quantities computed three different ways, because on a mesh "curvature" is not one thing:

- [`principal_curvature`][triwarp.curvature.principal_curvature] fits a quadric to a geodesic-ball
  neighbourhood of each vertex and reads the shape operator off it, giving both principal
  directions and magnitudes. This is the extrinsic, neighbourhood-scale answer, and the only one
  with a ``radius`` to tune (``igl::principal_curvature``).
- [`discrete_gaussian_curvature`][triwarp.curvature.discrete_gaussian_curvature] is the angle
  defect: purely *intrinsic*, exact rather than fitted, and computable from the angles alone.
- [`discrete_mean_curvature`][triwarp.curvature.discrete_mean_curvature] sums dihedral angle times
  edge length over the one ring, which is the integrated mean curvature rather than a pointwise one.

The neighbourhood the first one fits over is a *geodesic* ball
([`geodesic_ball`][triwarp.neighbors.geodesic_ball]), not a Euclidean one: a spatial query would
pull in vertices across a fold of the surface and corrupt the fit.
"""

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_same_device
from triwarp.kernels import curvature as kernel_curvature
from triwarp.kernels import scatter as kernel_scatter
from triwarp.vertices import vertex_defects
from triwarp.vertices import vertex_normals as _vertex_normals


def principal_curvature(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: int = 5,
    *,
    frame_independent: bool = True,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Principal curvature directions and magnitudes per vertex via quadric fitting.

    For each vertex a quadric surface is fitted to a sphere-neighborhood of vertices in the
    local tangent frame. The principal curvatures and directions are extracted from the
    eigendecomposition of the resulting shape operator (Weingarten map), built over a
    sphere-search neighborhood of radius ``radius * avg_edge_length``.

    ``(PD1, PD2)`` is an **orthonormal pair** at every vertex where the fit succeeded, and it is
    so by construction rather than by luck: principal directions are orthogonal, so only the first
    is solved for and the second is its quarter turn about the vertex normal. Both are unit and
    both lie in the tangent plane the fit used.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions as ``wp.vec3``.
    faces
        Length-``3 * n_faces`` flat triangle index buffer as ``wp.int32``.
    radius
        Neighborhood size multiplier applied to the average edge length. Larger values
        collect more neighbors and produce smoother curvature estimates.
    frame_independent
        When ``True`` (default), the principal curvatures are the eigenvalues of the true
        (textbook) Weingarten map ``II*v = lam*I*v``; these are surface invariants and do not
        depend on the chosen tangent frame. When ``False``, the symmetrized shape operator of
        ``igl::principal_curvature`` is reproduced verbatim (frame-dependent), matching
        ``igl.principal_curvature`` exactly. The two agree closely on well-sampled smooth
        surfaces; they differ only in the off-diagonal of the shape operator.
    face_normals
        Optional length-``n_faces`` unit face normals and matching areas from
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]; recomputed together
        when either is ``None``. [`Trimesh.face_normals`][triwarp.mesh.Trimesh.face_normals] and
        [`Trimesh.face_areas`][triwarp.mesh.Trimesh.face_areas] cache the pair.
    face_areas
        See ``face_normals``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]
        ``(PD1, PD2, PV1, PV2)`` where ``PV1 >= PV2`` at every vertex. Vertices for which
        the quadric fit failed (a neighbourhood holding fewer than 5 vertices besides the centre,
        or a degenerate system) have zero directions and zero curvature values.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``face_normals`` and ``face_areas`` are not all on one device.

    Notes
    -----
    A failed fit is signalled by that all-zero output and nothing else -- there is no separate
    validity mask. The two states it conflates are a failed fit and a genuinely flat vertex, which
    is why the test is worth stating: ``PD1`` is zero only where the fit did not produce a frame.

    The fit is **ill conditioned at a near-flat vertex**, and that is worth knowing before reading
    a small curvature as a measurement: a perturbation of the vertex normals far below their own
    precision moves the returned value there by a large *relative* amount, while barely moving it
    anywhere the surface actually curves. Compare a small ``PV`` against the mesh's own curvature
    scale rather than against zero.
    """
    require_same_device(
        vertices=vertices, faces=faces, face_normals=face_normals, face_areas=face_areas
    )
    device = vertices.device
    n_vertices = vertices.size

    # Area-weighted vertex normals. The pair is used only when both halves were supplied, which is
    # the documented contract; otherwise ``vertex_normals`` derives both inside its own scatter and
    # no per-face table is written.
    if face_normals is None or face_areas is None:
        face_normals = face_areas = None
    vertex_normals = _vertex_normals(
        vertices, faces, face_normals=face_normals, face_weights=face_areas
    )

    # ``mean_edge_length`` is the per-face average, matching libigl's
    # ``CurvatureCalculator::getAverageEdge`` -- the one ``igl::principal_curvature`` uses to set
    # ``scaledRadius``. Not ``mean_unique_edge_length``: switching to it takes the mean deviation
    # from ``igl.principal_curvature`` on an open half-torus from 0.0056 to 0.0117 and drops the
    # within-5% fraction from 99.3% to 98.5%.
    avg_edge = tw.edges.mean_edge_length(vertices, faces)
    scaled_radius = float(radius) * avg_edge

    # Collect vertex neighborhoods as geodesic balls (libigl getSphere) on device. A Euclidean ball
    # would pull in vertices across surface folds and corrupt the quadric fit; see
    # tw.neighbors.geodesic_ball.
    neighbor_indices, offsets, reference_neighbors = tw.neighbors.geodesic_ball(
        vertices, faces, scaled_radius
    )

    # Fit quadric and extract principal curvature per vertex. ``wp.empty``: every exit of the
    # kernel -- the three declines included, through ``_write_no_curvature`` -- writes all four.
    pd1 = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    pd2 = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    pv1 = wp.empty(n_vertices, dtype=wp.float32, device=device)
    pv2 = wp.empty(n_vertices, dtype=wp.float32, device=device)

    wp.launch(
        kernel_curvature.fit_principal_curvature,
        dim=n_vertices,
        inputs=[
            vertices,
            vertex_normals,
            neighbor_indices,
            offsets,
            reference_neighbors,
            frame_independent,
            pd1,
            pd2,
            pv1,
            pv2,
        ],
        device=device,
    )

    return pd1, pd2, pv1, pv2


def discrete_gaussian_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: twt.Array2dFloat32,
    radius: float,
) -> wp.array[wp.float32]:
    """
    Return the discrete Gaussian curvature measure of a sphere centered at each query point.

    As detailed in Cohen-Steiner and Morvan, "Restricted Delaunay triangulations and
    normal cycle". This is the sum of vertex defects at all vertices within the radius
    for each point.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space as ``wp.vec3``.
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_angles
        ``(n_faces, 3)`` interior angles per face (from
        [`face_angles`][triwarp.triangles.face_angles]).
    radius
        Sphere radius; may be zero when ``vertices`` are the query points.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` discrete Gaussian curvature measure on ``points.device``.

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices``, ``faces`` and ``face_angles`` are not all on one device.

    See Also
    --------
    [`vertex_defects`][triwarp.vertices.vertex_defects]
        The pointwise angle defect this integrates. Same quantity, no scale: at ``radius = 0`` with
        the vertices as query points the two agree, and it is the ball measure rather than the
        pointwise defect that converges under refinement.
    [`discrete_mean_curvature`][triwarp.curvature.discrete_mean_curvature]
        The mean-curvature measure over the same ball.
    """
    require_same_device(points=points, vertices=vertices, faces=faces, face_angles=face_angles)
    nearest_indices, _, nearest_offsets = tw.neighbors.query_ball_with_offsets(
        vertices, points, radius
    )
    defects = vertex_defects(vertices.size, faces, face_angles)
    gauss_curvature = wp.zeros(points.size, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_scatter.SCATTER_OFFSET_SUM[defects.dtype],
        dim=nearest_indices.size,
        inputs=[defects, nearest_indices, nearest_offsets, gauss_curvature],
        device=points.device,
    )
    return gauss_curvature


def discrete_mean_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: float,
    *,
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.float32]:
    """
    Return the discrete mean curvature measure of a sphere centered at each query point.

    As detailed in Cohen-Steiner and Morvan, "Restricted Delaunay triangulations and
    normal cycle". This is the sum of edge angles contained in the sphere for each point.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space as ``wp.vec3``.
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    radius
        Sphere radius which should typically be greater than zero.
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs. Must be supplied
        together with ``face_adjacency`` or omitted with it.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` discrete mean curvature measure on ``points.device``.

    Raises
    ------
    ValueError
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided.
    RuntimeError
        If ``points``, ``vertices``, ``faces``, ``face_adjacency`` and ``face_adjacency_edges`` are
        not all on one device.

    See Also
    --------
    [`discrete_gaussian_curvature`][triwarp.curvature.discrete_gaussian_curvature]
    [`trimesh.curvature.discrete_mean_curvature_measure`][]
    """
    require_same_device(
        points=points,
        vertices=vertices,
        faces=faces,
        face_adjacency=face_adjacency,
        face_adjacency_edges=face_adjacency_edges,
    )
    device = points.device
    n_points = points.size
    # The pairing check first, so a half-supplied pair raises whatever the mesh looks like; the
    # resolve after the two guards, so an empty input does not allocate tables nothing reads.
    tw.adjacency.require_paired_adjacency(face_adjacency, face_adjacency_edges)
    if n_points == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    n_faces = faces.size // 3
    if n_faces == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.adjacency.face_adjacency(
            faces, return_edges=True, n_vertices=vertices.size
        )
    assert face_adjacency_edges is not None

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    # Per adjacent pair: the dihedral angle signed by convexity, and the shared edge's bounds for
    # the BVH -- one launch over the pairs, reading one set of face normals.
    signed_angles = wp.empty(m, dtype=wp.float32, device=device)
    edge_lower = wp.empty(m, dtype=wp.vec3, device=device)
    edge_upper = wp.empty(m, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_curvature.face_pair_dihedrals,
        dim=m,
        inputs=[
            vertices,
            faces,
            face_adjacency,
            face_adjacency_edges,
            signed_angles,
            edge_lower,
            edge_upper,
        ],
        device=device,
    )

    bvh = tw.neighbors.bvh_from_bounds(edge_lower, edge_upper)
    # The predicate the narrow phase below applies is the *ball* -- it keeps the edge length inside
    # the ball of ``radius`` -- so the broad phase is the ball query and not the enclosing cube:
    # ``wp.bvh_query_sphere``'s traversal is cheaper than ``wp.bvh_query_aabb``'s on the identical
    # BVH, and the cube would also admit candidates outside the ball that the narrow phase would
    # then have to reject.
    candidate_edges, offsets = tw.neighbors.query_bvh_ball(bvh, points, radius)

    mean_curvature = wp.zeros(n_points, dtype=wp.float32, device=device)
    n_candidates = candidate_edges.size
    if n_candidates > 0:
        wp.launch(
            kernel_curvature.accumulate_mean_curvature,
            dim=n_candidates,
            inputs=[
                points,
                vertices,
                face_adjacency_edges,
                signed_angles,
                candidate_edges,
                offsets,
                wp.float32(radius),
                mean_curvature,
            ],
            device=device,
        )

    return mean_curvature
