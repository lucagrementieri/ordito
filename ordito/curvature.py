"""
Discrete curvature: principal directions and magnitudes, Gaussian and mean.

Several quantities computed several ways, because on a mesh "curvature" is not one thing:

- [`principal_curvature`][ordito.curvature.principal_curvature] fits a quadric to a geodesic-ball
  neighbourhood of each vertex and reads the shape operator off it, giving both principal
  directions and magnitudes. This is the extrinsic, neighbourhood-scale answer, and the only one
  with a ``radius`` to tune (``igl::principal_curvature``). Its ``method="pyvista"`` and
  ``method="meyer"`` take the 1-ring extremes of VTK's and of Meyer et al.'s curvatures instead.
- [`vertex_curvatures`][ordito.curvature.vertex_curvatures] is the pointwise mean and Gaussian
  curvature over one ring (Meyer et al.), per unit mixed area.
- [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature] is the angle
  defect: purely *intrinsic*, exact rather than fitted, and computable from the angles alone.
- [`discrete_mean_curvature`][ordito.curvature.discrete_mean_curvature] sums dihedral angle times
  edge length over the one ring, which is the integrated mean curvature rather than a pointwise one.

The neighbourhood the first one fits over is a *geodesic* ball
([`geodesic_ball`][ordito.neighbors.geodesic_ball]), not a Euclidean one: a spatial query would
pull in vertices across a fold of the surface and corrupt the fit.
"""

from typing import Literal

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.kernels import curvature as kernel_curvature
from ordito.vertices import vertex_normals as _vertex_normals


def principal_curvature(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: int = 5,
    *,
    frame_independent: bool = True,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
    mean_edge_length: float | None = None,
    method: Literal["quadric", "pyvista", "meyer"] = "quadric",
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
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
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
        ``(n_faces,)`` unit face normals and matching areas from
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]; recomputed together
        when either is ``None``. [`Trimesh.face_normals`][ordito.mesh.Trimesh.face_normals] and
        [`Trimesh.face_areas`][ordito.mesh.Trimesh.face_areas] cache the pair.
    face_areas
        ``(n_faces,)`` face areas; see ``face_normals``.
    mean_edge_length
        The mesh's [`mean_edge_length`][ordito.edges.mean_edge_length] when the caller holds it
        ([`Trimesh.mean_edge_length`][ordito.mesh.Trimesh.mean_edge_length]), which scales
        ``radius``; passing it spares the reduction and readback that measure it. Trusted, not
        checked.
    method
        ``"quadric"`` (the default) is the fit above. ``"pyvista"`` is VTK's ``vtkCurvatures``
        (``pyvista.PolyData.curvature("maximum")`` / ``("minimum")``): no fit, but
        ``H +- sqrt(H^2 - K)`` over its 1-ring Gaussian curvature ``K`` (three times the angle
        defect over the incident faces' area) and mean curvature ``H`` (half the mean, over the
        vertex's interior edges, of ``3 |e| theta / (A_f + A_g)``, ``theta`` the signed dihedral),
        ``H`` alone where ``H^2 < K``. ``"meyer"`` is the same extremes over Meyer et al.'s
        pointwise curvatures, [`vertex_curvatures`][ordito.curvature.vertex_curvatures]. Neither
        1-ring method forms directions, so ``PD1`` and ``PD2`` are zero, and ``radius``,
        ``frame_independent`` and the face tables are unread.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]
        ``(n_vertices,)`` arrays ``(PD1, PD2, PV1, PV2)`` where ``PV1 >= PV2`` at every vertex.
        Vertices for which the quadric fit failed (a neighbourhood holding fewer than 5 vertices
        besides the centre, or a degenerate system) have zero directions and zero curvature
        values.

    Raises
    ------
    ValueError
        If ``method`` is not ``"quadric"``, ``"pyvista"`` or ``"meyer"``.
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

    Examples
    --------
    Both principal curvatures of a sphere of radius 2 are ``1 / 2``, and the two directions are
    orthogonal:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3, radius=2.0)
    pd1, pd2, pv1, pv2 = od.curvature.principal_curvature(vertices, faces, radius=2)
    print(pv1.shape)  # (642,)
    print(f"{np.median(pv1.numpy()):.2f} {np.median(pv2.numpy()):.2f}")  # 0.51 0.51
    print(f"{np.abs((pd1.numpy() * pd2.numpy()).sum(axis=1)).max():.4f}")  # 0.0000
    ```
    """
    require_same_device(
        vertices=vertices, faces=faces, face_normals=face_normals, face_areas=face_areas
    )
    if method not in ("quadric", "pyvista", "meyer"):
        raise ValueError(f"method must be 'quadric', 'pyvista' or 'meyer', got {method!r}")
    device = vertices.device
    n_vertices = vertices.size
    if method != "quadric":
        return _one_ring_principal_curvatures(
            vertices,
            faces,
            kernel_curvature.vtk_principal_curvatures
            if method == "pyvista"
            else kernel_curvature.meyer_principal_curvatures,
        )

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
    avg_edge = (
        od.edges.mean_edge_length(vertices, faces) if mean_edge_length is None else mean_edge_length
    )
    scaled_radius = float(radius) * avg_edge

    # Collect vertex neighborhoods as geodesic balls (libigl getSphere) on device. A Euclidean ball
    # would pull in vertices across surface folds and corrupt the quadric fit; see
    # od.neighbors.geodesic_ball. The fit reads each chunk of the walk's queue rows as the walk
    # leaves them -- the same rows ``geodesic_ball`` packs, entry for entry -- so no neighbor list
    # over the whole mesh is held. ``min_count`` is ``geodesic_ball``'s default of 6.
    walk = od.neighbors._GeodesicBallWalk(  # pyright: ignore[reportPrivateUsage]
        vertices, faces, scaled_radius, 6
    )

    # Fit quadric and extract principal curvature per vertex. ``wp.empty``: every exit of the
    # kernel -- the three declines included, through ``_write_no_curvature`` -- writes all four.
    pd1 = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    pd2 = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    pv1 = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    pv2 = _launch.empty(n_vertices, dtype=wp.float32, device=device)

    for start, m in walk.chunks():
        _launch.launch(
            kernel_curvature.fit_principal_curvature,
            dim=m,
            inputs=[
                vertices,
                vertex_normals,
                walk.rows_flat,
                walk.counts,
                walk.row_stride,
                wp.int32(start),
                walk.reference,
                frame_independent,
                pd1,
                pd2,
                pv1,
                pv2,
            ],
            device=device,
        )
    walk.warn_clipped()

    return pd1, pd2, pv1, pv2


def _one_ring_principal_curvatures(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], kernel: odt.Kernel
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]:
    """Zero directions and ``kernel``'s 1-ring extremes: ``method="pyvista"`` or ``"meyer"``."""
    device = vertices.device
    n_vertices = vertices.size
    vertex_faces, vf_offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    mates = od.halfedge.halfedge_mates(faces, vertex_faces=(vertex_faces, vf_offsets))
    pv1 = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    pv2 = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    if n_vertices > 0:
        _launch.launch(
            kernel,
            dim=n_vertices,
            inputs=[vertices, faces, vertex_faces, vf_offsets, mates],
            outputs=[pv1, pv2],
            device=device,
        )
    directions = _launch.zeros(n_vertices, dtype=wp.vec3, device=device)
    return directions, _launch.zeros(n_vertices, dtype=wp.vec3, device=device), pv1, pv2


def vertex_curvatures(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Pointwise mean and Gaussian curvature per vertex (Meyer et al.).

    The discrete operators of Meyer, Desbrun, Schroder and Barr, "Discrete Differential-Geometry
    Operators for Triangulated 2-Manifolds" (2002), per unit area: the Gaussian curvature is the
    angle defect over the vertex's mixed (Voronoi) area, and the mean curvature is the length of
    the cotangent Laplacian of the positions over twice that area, signed positive where it points
    along the area-weighted vertex normal. This is MeshLab's
    ``compute_scalar_by_discrete_curvature_per_vertex`` (vcg's ``MeanAndGaussian``), including its
    rim rule: a vertex on one rim also subtracts the angle between its two rim edges, so a straight
    flat rim reads zero.

    Where [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature] and
    [`discrete_mean_curvature`][ordito.curvature.discrete_mean_curvature] integrate over a ball,
    this reads one ring: its values follow the mesh's resolution and are noisy on irregular
    meshes, and a vertex whose ring area is tiny (a needle's tip) amplifies rounding.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    mean : wp.array[wp.float32]
        ``(n_vertices,)`` mean curvature; zero where the mixed area vanishes.
    gaussian : wp.array[wp.float32]
        ``(n_vertices,)`` Gaussian curvature; zero where the mixed area vanishes.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    See Also
    --------
    [`principal_curvature`][ordito.curvature.principal_curvature]
        Its ``method="meyer"`` is the principal pair these imply, ``H +- sqrt(H^2 - K)``.
    [`ordito.vertices.vertex_defects`][ordito.vertices.vertex_defects]
        The angle defect alone, not divided by an area.

    Examples
    --------
    On a sphere of radius 2 both curvatures are close to their exact values, ``1 / 2`` and
    ``1 / 4``:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=4, radius=2.0)
    mean, gaussian = od.curvature.vertex_curvatures(vertices, faces)
    print(f"{np.median(mean.numpy()):.3f} {np.median(gaussian.numpy()):.3f}")  # 0.500 0.250
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size
    mean = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    gaussian = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    if n_vertices == 0:
        return mean, gaussian
    vertex_faces, vf_offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    mates = od.halfedge.halfedge_mates(faces, vertex_faces=(vertex_faces, vf_offsets))
    _launch.launch(
        kernel_curvature.meyer_vertex_curvatures,
        dim=n_vertices,
        inputs=[vertices, faces, vertex_faces, vf_offsets, mates],
        outputs=[mean, gaussian],
        device=device,
    )
    return mean, gaussian


def discrete_gaussian_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: odt.Array2dFloat32,
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
        ``(n,)`` query positions in space.
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    face_angles
        ``(n_faces, 3)`` interior angles per face (from
        [`face_angles`][ordito.triangles.face_angles]).
    radius
        Sphere radius; may be zero when ``vertices`` are the query points.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` discrete Gaussian curvature measure on ``points.device``.

    Raises
    ------
    ValueError
        If ``face_angles`` does not have one row per triangle in ``faces``.
    RuntimeError
        If ``points``, ``vertices``, ``faces`` and ``face_angles`` are not all on one device.

    See Also
    --------
    [`vertex_defects`][ordito.vertices.vertex_defects]
        The pointwise angle defect this integrates. Same quantity, no scale: at ``radius = 0`` with
        the vertices as query points the two agree, and it is the ball measure rather than the
        pointwise defect that converges under refinement.
    [`discrete_mean_curvature`][ordito.curvature.discrete_mean_curvature]
        The mean-curvature measure over the same ball.

    Examples
    --------
    A ball containing a whole closed sphere measures its total Gaussian curvature, ``4 * pi``
    (Gauss-Bonnet):

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    angles = od.triangles.face_angles(vertices, faces)
    points = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    gauss = od.curvature.discrete_gaussian_curvature(points, vertices, faces, angles, 2.0)
    print(f"{gauss.numpy()[0]:.4f}")  # 12.5663
    ```
    """
    require_same_device(points=points, vertices=vertices, faces=faces, face_angles=face_angles)
    device = points.device
    n_faces = faces.size // 3
    if face_angles.shape[0] != n_faces:
        raise ValueError(
            f"face_angles must have one row per triangle, got {face_angles.shape[0]} rows for "
            f"{n_faces} faces"
        )
    if points.size == 0 or vertices.size == 0:
        return _launch.zeros(points.size, dtype=wp.float32, device=device)
    # Each vertex's incident-angle sum, in float64; the ball walk forms its defect as it reads it.
    angle_sum = _launch.zeros(vertices.size, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_curvature.vertex_angle_sums,
        dim=n_faces,
        inputs=[face_angles, faces],
        outputs=[angle_sum],
        device=device,
    )
    grid = od.neighbors.hashgrid_from_points(vertices, float(radius))
    gauss_curvature = _launch.empty(points.size, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_curvature.ball_angle_defect_sum,
        dim=points.size,
        inputs=[vertices, angle_sum, points, grid.id, wp.float32(radius), gauss_curvature],
        device=device,
    )
    return gauss_curvature


def discrete_mean_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: float,
    *,
    face_adjacency: odt.Array2dInt32 | None = None,
    face_adjacency_edges: odt.Array2dInt32 | None = None,
) -> wp.array[wp.float32]:
    """
    Return the discrete mean curvature measure of a sphere centered at each query point.

    As detailed in Cohen-Steiner and Morvan, "Restricted Delaunay triangulations and
    normal cycle". This is the sum of edge angles contained in the sphere for each point.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space.
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    radius
        Sphere radius which should typically be greater than zero.
    face_adjacency
        ``(m, 2)`` face index pairs from
        [`face_adjacency`][ordito.adjacency.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        ``(m, 2)`` sorted shared vertex pairs, or ``None``. Must be supplied
        together with ``face_adjacency`` or omitted with it.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` discrete mean curvature measure on ``points.device``.

    Raises
    ------
    ValueError
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided.
    RuntimeError
        If ``points``, ``vertices``, ``faces``, ``face_adjacency`` and ``face_adjacency_edges`` are
        not all on one device.

    See Also
    --------
    [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature]
    [`trimesh.curvature.discrete_mean_curvature_measure`][]

    Examples
    --------
    A ball containing the whole unit sphere measures close to ``4 * pi``; a ball that reaches no
    edge measures zero:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    points = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    mean = od.curvature.discrete_mean_curvature(points, vertices, faces, 2.0)
    print(f"{mean.numpy()[0]:.3f}")  # 12.487
    mean = od.curvature.discrete_mean_curvature(points, vertices, faces, 0.5)
    print(f"{mean.numpy()[0]:.3f}")  # 0.000
    ```
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
    od.adjacency.require_paired_adjacency(face_adjacency, face_adjacency_edges)
    if n_points == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.zeros(n_points, dtype=wp.float32, device=device)

    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = od.adjacency.face_adjacency(
            faces, return_edges=True, n_vertices=vertices.size
        )
    assert face_adjacency_edges is not None

    m = int(face_adjacency.shape[0])
    if m == 0:
        return _launch.zeros(n_points, dtype=wp.float32, device=device)

    # Per adjacent pair: the dihedral angle signed by convexity, and the shared edge's bounds for
    # the BVH -- one launch over the pairs, reading one set of face normals.
    signed_angles = _launch.empty(m, dtype=wp.float32, device=device)
    edge_lower, edge_upper = _launch.carve(wp.vec3, [m, m], device)
    _launch.launch(
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

    bvh = od.neighbors.bvh_from_bounds(edge_lower, edge_upper)
    # The predicate the narrow phase below applies is the *ball* -- it keeps the edge length inside
    # the ball of ``radius`` -- so the broad phase is the ball query and not the enclosing cube:
    # ``wp.bvh_query_sphere``'s traversal is cheaper than ``wp.bvh_query_aabb``'s on the identical
    # BVH, and the cube would also admit candidates outside the ball that the narrow phase would
    # then have to reject.
    mean_curvature = _launch.empty(n_points, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_curvature.ball_mean_curvature,
        dim=n_points,
        inputs=[
            points,
            vertices,
            face_adjacency_edges,
            signed_angles,
            bvh.id,
            wp.float32(radius),
            mean_curvature,
        ],
        device=device,
    )
    return mean_curvature
