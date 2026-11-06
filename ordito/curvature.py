"""
Discrete curvature: principal directions and magnitudes, Gaussian and mean.

Several quantities computed several ways, because on a mesh "curvature" is not one thing:

- [`principal_curvature`][ordito.curvature.principal_curvature] fits a quadric to a geodesic-ball
  neighbourhood of each vertex and reads the shape operator off it, giving both principal
  directions and magnitudes. This is the extrinsic, neighbourhood-scale answer, and the only one
  with a ``radius`` to tune (``igl::principal_curvature``). Its ``method="pyvista"`` and
  ``method="meyer"`` take the 1-ring extremes of VTK's and of Meyer et al.'s curvatures instead,
  and ``method="taubin"`` and ``method="normal_cycles"`` 1-ring tensors that carry directions.
- [`point_principal_curvature`][ordito.curvature.point_principal_curvature] is the same quadric
  fit on a point cloud with normals, over each point's nearest neighbours.
- [`vertex_curvatures`][ordito.curvature.vertex_curvatures] is the pointwise mean and Gaussian
  curvature over one ring (Meyer et al.), per unit mixed area.
- [`vertex_defects`][ordito.curvature.vertex_defects] is the angle defect, ``2 * pi`` minus the
  incident angle sum: the Gaussian curvature concentrated at each vertex, purely *intrinsic*, exact
  rather than fitted, and computable from the angles alone.
- [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature] sums those defects
  over a ball around each query point, the measure that converges under refinement.
- [`discrete_mean_curvature`][ordito.curvature.discrete_mean_curvature] sums dihedral angle times
  edge length over the one ring, which is the integrated mean curvature rather than a pointwise one.
- [`mls_principal_curvature`][ordito.curvature.mls_principal_curvature] is the principal pair of
  the smooth moving-least-squares surface a point cloud with normals defines (APSS or RIMLS), at
  each point's projection onto it.

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
from ordito.config import is_deterministic
from ordito.kernels import curvature as kernel_curvature
from ordito.kernels import predicates as kernel_predicates
from ordito.kernels import scatter as kernel_scatter
from ordito.normals import vertex_normals as _vertex_normals


def principal_curvature(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: int = 5,
    *,
    frame_independent: bool = True,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
    mean_edge_length: float | None = None,
    method: Literal["quadric", "pyvista", "meyer", "taubin", "normal_cycles"] = "quadric",
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
        of those two forms directions, so ``PD1`` and ``PD2`` are zero. ``"taubin"`` is Taubin's
        1-ring tensor: each edge's normal curvature ``2 n . (v - v_j) / |v - v_j|^2`` along its
        tangent projection, weighted by the area of the faces sharing it, with ``n`` the
        angle-weighted vertex normal; the tensor's eigenvectors are the principal directions and
        its eigenvalues ``m1 >= m2`` give ``3 m1 - m2`` and ``3 m2 - m1``. ``"normal_cycles"`` is
        Cohen-Steiner and Morvan's normal-cycle tensor over the vertex's barycentric cell: each
        interior edge at the vertex adds its signed dihedral angle times half its length times
        its direction's outer product, over a third of the ring's area. Its eigenvector nearest
        the vertex normal is dropped and the other two give the principal curvatures, each
        belonging to the direction across its eigenvector. All four 1-ring methods leave
        ``radius``, ``frame_independent`` and the face tables unread.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]
        ``(n_vertices,)`` arrays ``(PD1, PD2, PV1, PV2)`` where ``PV1 >= PV2`` at every vertex.
        Vertices for which the quadric fit failed (a neighbourhood holding fewer than 5 vertices
        besides the centre, or a degenerate system) have zero directions and zero curvature
        values, as do a ``"taubin"`` vertex whose ring has no area and a ``"normal_cycles"``
        vertex with no interior edge.

    Raises
    ------
    ValueError
        If ``method`` is not ``"quadric"``, ``"pyvista"``, ``"meyer"``, ``"taubin"`` or
        ``"normal_cycles"``.
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
    if method not in ("quadric", "pyvista", "meyer", "taubin", "normal_cycles"):
        raise ValueError(
            "method must be 'quadric', 'pyvista', 'meyer', 'taubin' or 'normal_cycles', "
            f"got {method!r}"
        )
    device = vertices.device
    n_vertices = vertices.size
    if method != "quadric":
        return _one_ring_principal_curvatures(vertices, faces, method)

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
    walk = od.neighbors.GeodesicBallWalk(vertices, faces, scaled_radius, 6)
    # ``kernels/algorithms/bfs.ball_row_bounds`` reads the rows flat at their stride.
    rows_flat = walk.rows.flatten()
    row_stride = wp.int32(walk.rows.shape[1])

    # Fit quadric and extract principal curvature per vertex. ``wp.empty``: every exit of the
    # kernel -- the three declines of ``monge_principal_curvature`` included -- writes all four.
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
                rows_flat,
                walk.counts,
                row_stride,
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

    return pd1, pd2, pv1, pv2


def _one_ring_principal_curvatures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    method: Literal["pyvista", "meyer", "taubin", "normal_cycles"],
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]:
    """``principal_curvature``'s 1-ring methods: one thread per vertex over its face row."""
    device = vertices.device
    n_vertices = vertices.size
    vertex_faces, vf_offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    # ``wp.empty``: every kernel below writes all four at every vertex, the zero directions of
    # ``"pyvista"`` and ``"meyer"`` and the zero answer of the tensors included.
    pd1 = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    pd2 = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    pv1 = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    pv2 = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    if method in ("pyvista", "meyer"):
        mates = od.halfedge.halfedge_mates(faces, vertex_faces=(vertex_faces, vf_offsets))
        if n_vertices > 0:
            if method == "pyvista":
                kernel = kernel_curvature.vtk_principal_curvatures
                face_terms = _launch.empty((faces.size // 3, 4), dtype=wp.float64, device=device)
                _launch.launch(
                    kernel_curvature.vtk_face_terms,
                    dim=faces.size // 3,
                    inputs=[vertices, faces, mates],
                    outputs=[face_terms],
                    device=device,
                )
            else:
                kernel = kernel_curvature.meyer_principal_curvatures
                face_terms = _meyer_face_terms(vertices, faces)
            inputs = [vertices, faces, vertex_faces, vf_offsets, mates, face_terms]
            _launch.launch(
                kernel, dim=n_vertices, inputs=inputs, outputs=[pd1, pd2, pv1, pv2], device=device
            )
        return pd1, pd2, pv1, pv2
    normals = _vertex_normals(vertices, faces, weighting="angle")
    if method == "taubin":
        kernel = kernel_curvature.taubin_principal_curvatures
        inputs = [vertices, faces, vertex_faces, vf_offsets, normals]
    else:
        kernel = kernel_curvature.normal_cycle_principal_curvatures
        mates = od.halfedge.halfedge_mates(faces, vertex_faces=(vertex_faces, vf_offsets))
        inputs = [vertices, faces, vertex_faces, vf_offsets, mates, normals]
    if n_vertices > 0:
        _launch.launch(
            kernel, dim=n_vertices, inputs=inputs, outputs=[pd1, pd2, pv1, pv2], device=device
        )
    return pd1, pd2, pv1, pv2


def point_principal_curvature(
    points: wp.array[wp.vec3], normals: wp.array[wp.vec3], neighbor_idx: odt.Array2dInt32
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Principal curvature directions and magnitudes per point of a cloud, by a Monge-patch fit.

    The point-cloud form of [`principal_curvature`][ordito.curvature.principal_curvature]: the
    same quadric ``w = a u^2 + b u v + c v^2 + d u + e v`` is fitted through each point in the
    tangent frame of its normal, over the neighbours its row of ``neighbor_idx`` lists, and the
    principal curvatures are the eigenvalues of the shape operator read off the fit (the
    frame-independent Weingarten map). Neighbours whose normal faces away from the point's are
    left out of the fit when at least six remain, so the far wall of a thin part does not bend
    it.

    ``(PD1, PD2)`` is an orthonormal pair in the plane perpendicular to the point's normal
    wherever the fit succeeded.

    Parameters
    ----------
    points
        ``(n,)`` point positions.
    normals
        ``(n,)`` point normals, oriented consistently between neighbours; need not be unit. They
        fix the tangent frame and the sign of the curvature: positive where the surface bends
        away from the normal, as on a sphere with outward normals.
        [`estimate_normals`][ordito.points.estimate_normals] estimates them.
    neighbor_idx
        ``(n, k)`` neighbour indices per point, as returned by
        [`query_nearest`][ordito.neighbors.query_nearest] (unused slots ``-1``, a self-query
        listing the point itself, which the fit skips). Pass ``max_radius`` there to bound the
        neighbourhood's radius as well as its count. The fit needs at least five neighbours
        besides the point itself, and is better determined with several times that.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]
        ``(n,)`` arrays ``(PD1, PD2, PV1, PV2)`` with ``PV1 >= PV2`` at every point. A point
        whose fit failed (fewer than five neighbours besides itself, neighbours all coincident
        with it, or a degenerate system) has zero directions and zero curvature values.

    Raises
    ------
    ValueError
        If ``normals`` or ``neighbor_idx`` does not have one entry or row per point.
    RuntimeError
        If ``points``, ``normals`` and ``neighbor_idx`` are not all on one device.

    See Also
    --------
    [`principal_curvature`][ordito.curvature.principal_curvature]

    Examples
    --------
    Both principal curvatures of points on a sphere of radius 2 are ``1 / 2``:

    ```python
    import numpy as np

    import ordito as od

    points = od.sample.sample_fibonacci_sphere(2000)
    points = od.transform.transform_points(points, od.transform.scale_matrix(2.0))
    normals = od.sample.sample_fibonacci_sphere(2000)
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, 16)
    pd1, pd2, pv1, pv2 = od.curvature.point_principal_curvature(points, normals, neighbor_idx)
    print(pv1.shape)  # (2000,)
    print(f"{np.median(pv1.numpy()):.2f} {np.median(pv2.numpy()):.2f}")  # 0.50 0.50
    ```
    """
    require_same_device(points=points, normals=normals, neighbor_idx=neighbor_idx)
    odt.ensure_ndim(neighbor_idx, 2, dtype=wp.int32)
    device = points.device
    n = points.size
    # One thread per point reads its own normal and row, so a short table is an out-of-bounds
    # read, not a short answer.
    if normals.size != n or int(neighbor_idx.shape[0]) != n:
        raise ValueError(
            f"normals and neighbor_idx must have one entry per point, got {normals.size} and "
            f"{neighbor_idx.shape} for {n} points"
        )
    # ``wp.empty``: every exit of ``monge_principal_curvature`` returns all four.
    pd1 = _launch.empty(n, dtype=wp.vec3, device=device)
    pd2 = _launch.empty(n, dtype=wp.vec3, device=device)
    pv1 = _launch.empty(n, dtype=wp.float32, device=device)
    pv2 = _launch.empty(n, dtype=wp.float32, device=device)
    if n > 0:
        _launch.launch(
            kernel_curvature.fit_point_principal_curvature,
            dim=n,
            inputs=[points, normals, neighbor_idx.flatten(), wp.int32(neighbor_idx.shape[1])],
            outputs=[pd1, pd2, pv1, pv2],
            device=device,
        )
    return pd1, pd2, pv1, pv2


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
    [`vertex_defects`][ordito.curvature.vertex_defects]
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
    face_terms = _meyer_face_terms(vertices, faces)
    _launch.launch(
        kernel_curvature.meyer_vertex_curvatures,
        dim=n_vertices,
        inputs=[vertices, faces, vertex_faces, vf_offsets, mates, face_terms],
        outputs=[mean, gaussian],
        device=device,
    )
    return mean, gaussian


def _meyer_face_terms(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.float64, Literal[2]]:
    """``(n_faces, 5)`` angle terms of Meyer et al.'s walk, formed once a face, not a corner."""
    n_faces = faces.size // 3
    face_terms = _launch.empty((n_faces, 5), dtype=wp.float64, device=vertices.device)
    if n_faces > 0:
        _launch.launch(
            kernel_curvature.meyer_face_terms,
            dim=n_faces,
            inputs=[vertices, faces],
            outputs=[face_terms],
            device=vertices.device,
        )
    return face_terms


def vertex_defects(
    n_vertices: int, faces: wp.array[wp.int32], face_angles: odt.Array2dFloat32
) -> wp.array[wp.float32]:
    """
    Discrete angle defect per vertex: ``2π`` minus the sum of incident corner angles.

    For each vertex, interior angles from every triangle corner that references that vertex
    are accumulated on ``faces.device``, then subtracted from a full turn. Under
    [`is_deterministic`][ordito.config.is_deterministic] the sum is formed exactly in ``float64``
    and rounded once, so it is the same on every run; otherwise it is accumulated in ``float32`` in
    the atomics' arrival order.
    This is the standard piecewise-linear angle defect (related to discrete Gaussian
    curvature via the Gauss—Bonnet viewpoint on triangle meshes).

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        ``(3 * n_faces,)`` triangle indices, interpreted as ``(n_faces, 3)`` via
        ``reshape((-1, 3))`` (row-major flat layout is fine).
    face_angles
        ``(n_faces, 3)`` interior angles at the three corners of each triangle, with rows aligned
        with ``faces``.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_vertices,)`` device array ``2π - Σ angles`` at each vertex. Vertices not
        referenced by any face have defect ``2π`` (empty angle sum).

    Raises
    ------
    ValueError
        If ``face_angles`` does not have one row per triangle in ``faces``.
    RuntimeError
        If ``faces`` and ``face_angles`` are not all on one device.

    See Also
    --------
    [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature]
        The same quantity at a *scale*: the Cohen-Steiner/Morvan ball measure sums these defects
        over a ball of given radius, where this is the pointwise value at one vertex. The ball
        measure is what converges under refinement; the pointwise defect does not.
    [`face_angles`][ordito.triangles.face_angles]
        The angles this sums.

    Examples
    --------
    Each cube corner meets three right angles, so its defect is ``pi / 2``, and the defects sum to
    ``4 * pi`` on any closed genus-0 surface (Gauss-Bonnet):

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    angles = od.triangles.face_angles(vertices, faces)
    defects = od.curvature.vertex_defects(vertices.shape[0], faces, angles)
    print(f"{defects.numpy()[0]:.4f}")  # 1.5708
    print(f"{defects.numpy().sum():.4f}")  # 12.5664
    ```
    """
    require_same_device(faces=faces, face_angles=face_angles)
    n_faces = faces.size // 3
    _require_face_angle_rows(n_faces, face_angles)
    if not is_deterministic():
        angle_sum = _launch.zeros(n_vertices, dtype=wp.float32, device=faces.device)
        _launch.launch(
            kernel_scatter.SCATTER_SUM_SCALAR[face_angles.dtype],
            dim=n_faces,
            inputs=[face_angles, faces, angle_sum],
            device=faces.device,
        )
        # In place over the accumulator: ``angle_sum`` is scratch, and the operator spelling
        # ``TWO_PI - angle_sum`` would run the same wp.map into a second allocation.
        _launch.map(kernel_predicates.angle_defect, angle_sum, out=angle_sum)
        return angle_sum
    # A ``float32`` angle is exact in ``float64``, and so is the sum, so the atomics' commit order
    # changes no bit: reproducible for the price of the ``float32`` output's own allocation.
    exact_sum = _launch.zeros(n_vertices, dtype=wp.float64, device=faces.device)
    _launch.launch(
        kernel_scatter.scatter_sum_exact,
        dim=n_faces,
        inputs=[face_angles, faces, exact_sum],
        device=faces.device,
    )
    defects = _launch.empty(n_vertices, dtype=wp.float32, device=faces.device)
    _launch.map(kernel_curvature.angle_defect_rounded, exact_sum, out=defects)
    return defects


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
    [`vertex_defects`][ordito.curvature.vertex_defects]
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
    _require_face_angle_rows(n_faces, face_angles)
    if points.size == 0 or vertices.size == 0:
        return _launch.zeros(points.size, dtype=wp.float32, device=device)
    # Each vertex's incident-angle sum, in float64; the ball walk forms its defect as it reads it.
    angle_sum = _launch.zeros(vertices.size, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_scatter.scatter_sum_exact,
        dim=n_faces,
        inputs=[face_angles, faces, angle_sum],
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


def _require_face_angle_rows(n_faces: int, face_angles: odt.Array2dFloat32) -> None:
    """
    Check that a corner-angle table has one row per triangle.

    The angle scatters index ``face_angles`` by the face id they read ``faces`` with, so a short
    table is read out of range: an illegal access on the CUDA device and a heap read on the host,
    never an exception.

    Parameters
    ----------
    n_faces
        Triangle count the table must match, derived from the caller's ``faces`` buffer.
    face_angles
        ``(n_faces, 3)`` interior angles per face.

    Raises
    ------
    ValueError
        If ``face_angles`` does not have ``n_faces`` rows.
    """
    rows = int(face_angles.shape[0])
    if rows != n_faces:
        raise ValueError(
            f"face_angles must have one row per triangle, got {rows} rows for {n_faces} faces"
        )


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
    signed_angles, edge_lower, edge_upper = _launch.carve_typed(
        ((wp.float32, m), (wp.vec3, m), (wp.vec3, m)), device
    )
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


def mls_principal_curvature(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    method: Literal["apss", "rimls"] = "apss",
    filter_scale: float = 2.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Principal curvature directions and magnitudes of a moving-least-squares surface per point.

    The point cloud with normals defines a smooth implicit surface, and each point is projected
    onto it and given that surface's curvature there: the shape operator of the implicit's level
    set, from its gradient and Hessian, whose eigenpairs are the principal curvatures and their
    directions. Each sample influences the points within ``filter_scale`` times its spacing (twice
    the root-mean-square distance to its 16 nearest samples), with weight ``(1 - d^2 / r^2)^4``.

    ``method`` picks the surface, both as MeshLab's ``filter_mls`` builds them:

    - ``"apss"``: algebraic point set surfaces (Guennebaud and Gross), a sphere fitted to the
      weighted samples and normals around each location. The gradient and Hessian are the fit's
      exact derivatives, its own dependence on the location included.
    - ``"rimls"``: robust implicit MLS (Oztireli, Guennebaud and Gross), a weighted average of the
      samples' tangent planes refitted up to three times with samples whose normal disagrees with
      the current gradient down-weighted, which keeps sharp features sharp. Its Hessian holds
      the last round's refitting weights fixed.

    ``(PD1, PD2)`` is an orthonormal pair in the surface's tangent plane at every point where it is
    defined.

    Parameters
    ----------
    points
        ``(n,)`` point positions.
    normals
        ``(n,)`` unit point normals, oriented consistently; they define the surface, and the
        curvature is positive where it bends away from them, as on a sphere with outward normals.
        [`estimate_normals`][ordito.points.estimate_normals] estimates them.
    method
        ``"apss"`` (default) or ``"rimls"``, above.
    filter_scale
        Each sample's reach as a multiple of its spacing: larger is smoother.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]
        ``(n,)`` arrays ``(PD1, PD2, PV1, PV2)`` with ``PV1 >= PV2`` at every point. A point where
        the surface is undefined (fewer than two samples reach its projection, or the gradient
        vanishes there) has zero directions and zero curvature values.

    Raises
    ------
    ValueError
        If ``method`` is not ``"apss"`` or ``"rimls"``, ``filter_scale`` is not positive, or
        ``normals`` does not have one entry per point.
    RuntimeError
        If ``points`` and ``normals`` are not on one device.

    See Also
    --------
    [`principal_curvature`][ordito.curvature.principal_curvature]

    Notes
    -----
    MeshLab's APSS colorize filter (``compute_curvature_and_color_apss_per_vertex``) computes the
    same surface, projection and derivatives; its ``"Mean"`` output is the sum ``PV1 + PV2``, and
    it orders its ``K1`` and ``K2`` by magnitude.

    Examples
    --------
    Both principal curvatures of points on a sphere of radius 2 are ``1 / 2``:

    ```python
    import numpy as np

    import ordito as od

    normals = od.sample.sample_fibonacci_sphere(2000)
    points = od.transform.transform_points(normals, od.transform.scale_matrix(2.0))
    pd1, pd2, pv1, pv2 = od.curvature.mls_principal_curvature(points, normals)
    print(pv1.shape)  # (2000,)
    print(f"{np.median(pv1.numpy()):.2f} {np.median(pv2.numpy()):.2f}")  # 0.50 0.50
    ```
    """
    require_same_device(points=points, normals=normals)
    if method not in ("apss", "rimls"):
        raise ValueError(f"method must be 'apss' or 'rimls', got {method!r}")
    if not filter_scale > 0.0:
        raise ValueError(f"filter_scale must be positive, got {filter_scale}")
    device = points.device
    n = points.size
    # One thread per point reads its own normal, so a short table is an out-of-bounds read.
    if normals.size != n:
        raise ValueError(f"normals must have one entry per point, got {normals.size} for {n}")
    if n < 2:
        # No surface: it needs two samples reaching a point.
        return (
            _launch.zeros(n, dtype=wp.vec3, device=device),
            _launch.zeros(n, dtype=wp.vec3, device=device),
            _launch.zeros(n, dtype=wp.float32, device=device),
            _launch.zeros(n, dtype=wp.float32, device=device),
        )
    # ``wp.empty``: both kernels write all four at every point.
    pd1 = _launch.empty(n, dtype=wp.vec3, device=device)
    pd2 = _launch.empty(n, dtype=wp.vec3, device=device)
    pv1 = _launch.empty(n, dtype=wp.float32, device=device)
    pv2 = _launch.empty(n, dtype=wp.float32, device=device)
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, min(16, n))
    radii = _launch.empty(n, dtype=wp.float64, device=device)
    lower = _launch.empty(n, dtype=wp.vec3, device=device)
    upper = _launch.empty(n, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_curvature.mls_sample_balls,
        dim=n,
        inputs=[points, odt.as_array2d(neighbor_idx, wp.int32), wp.float64(filter_scale)],
        outputs=[radii, lower, upper],
        device=device,
    )
    bvh = od.neighbors.bvh_from_bounds(lower, upper)
    _launch.launch(
        kernel_curvature.apss_principal_curvature
        if method == "apss"
        else kernel_curvature.rimls_principal_curvature,
        dim=n,
        inputs=[points, normals, radii, bvh.id],
        outputs=[pd1, pd2, pv1, pv2],
        device=device,
    )
    return pd1, pd2, pv1, pv2
