"""
Quantities carried by one triangle at a time: its normal, area, angles, shape and barycentre.

[`face_normals_and_areas`][ordito.triangles.face_normals_and_areas],
[`face_angles`][ordito.triangles.face_angles],
[`face_centroids`][ordito.triangles.face_centroids] and
[`face_quality`][ordito.triangles.face_quality] are the per-face fields the rest of the package
reduces over; [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask] flags the faces
those quantities are meaningless on.
[`barycentric_to_points`][ordito.triangles.barycentric_to_points],
[`points_to_barycentric`][ordito.triangles.points_to_barycentric] and
[`closest_point`][ordito.triangles.closest_point] work one query point against one triangle, row
by row, without a BVH -- for a query against the whole surface see
[`ordito.proximity`][ordito.proximity].
"""

from typing import Literal

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.kernels import triangles as kernel_triangles


def face_normals_and_areas(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """
    Compute the unit normal and area of each triangle.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    normals : wp.array[wp.vec3]
        ``(n_faces,)`` unit face normals on ``vertices.device``. Zero when a face is degenerate.
    areas : wp.array[wp.float32]
        ``(n_faces,)`` triangle areas on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`trimesh.Trimesh.face_normals`][]
    [`trimesh.Trimesh.area_faces`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    f = faces.size // 3
    out_normal = _launch.empty(f, dtype=wp.vec3, device=vertices.device)
    out_area = _launch.empty(f, dtype=wp.float32, device=vertices.device)
    _launch.launch(
        kernel_triangles.face_normals_and_areas,
        dim=f,
        inputs=[vertices, faces, out_normal, out_area],
        device=vertices.device,
    )
    return out_normal, out_area


def face_angles(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> odt.Array2dFloat32:
    """
    Interior angle at each triangle vertex, in radians.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    odt.Array2dFloat32
        ``(n_faces, 3)`` interior angles on ``vertices.device``, aligned with the corners
        ``(i0, i1, i2)`` of each face.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`trimesh.triangles.angles`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    f = faces.size // 3
    out_angle = odt.empty_2d((f, 3), wp.float32, device=vertices.device)
    _launch.launch(
        kernel_triangles.angles, dim=f, inputs=[vertices, faces, out_angle], device=vertices.device
    )
    return odt.as_array2d(out_angle, wp.float32)


_QUALITY_METRICS: dict[str, wp.int32] = {
    "aspect_ratio": kernel_triangles.QUALITY_ASPECT_RATIO,
    "radius_ratio": kernel_triangles.QUALITY_RADIUS_RATIO,
    "area_max_side": kernel_triangles.QUALITY_AREA_MAX_SIDE,
    "mean_ratio": kernel_triangles.QUALITY_MEAN_RATIO,
    "area": kernel_triangles.QUALITY_AREA,
}


CornerNormalWeighting = Literal["angle", "area"]
"""Per-corner weight selected by [`corner_normals`][ordito.triangles.corner_normals]."""


def corner_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    crease_edges: odt.Array2dInt32 | None = None,
    *,
    weighting: CornerNormalWeighting = "angle",
    twins: wp.array[wp.int32] | None = None,
    n_vertices: int | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> odt.Array2dVec3:
    """
    Per-corner normals: one normal for each ``(face, corner)``, averaged over its smooth group.

    The third weighting ordito lacked. A *face* normal is flat, a *vertex* normal is smooth
    everywhere, and neither can render a crease: a cube needs three different normals at each of its
    corners, one per incident face group. This gives each corner the angle-weighted average of the
    faces reachable from its own by rotating about its vertex **without crossing a crease edge**,
    which is what an OBJ export or a hard-edge renderer needs.

    Two exact identities fall out of that definition and are worth knowing, because they are also
    how it is tested:

    * with **no** creases, every corner of a vertex gets that vertex's normal under the matching
      weighting -- [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="angle"``
      or [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="area"``;
    * with **every** edge a crease, every corner gets its own face's normal, whatever the weighting.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    crease_edges
        ``(k, 2)`` vertex pairs to treat as hard edges, in either order per row --
        typically [`crease_edges`][ordito.seams.crease_edges]. ``None`` means no creases, which is
        the fully smooth case above and costs one comparison per rotation step rather than a branch.
    weighting
        Per-corner weight for the average, named as in
        [`ordito.vertices`][ordito.vertices]' four vertex-normal functions: ``"angle"`` is the
        corner angle (Thuerrner & Wuethrich) and ``"area"`` is the face's own area, constant across
        its three corners. Both are offered because the two reference implementations disagree --
        the one that computes per-corner normals directly is the **area** one.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins], or
        ``None`` to compute them from ``faces``.
    n_vertices
        Total vertex count. When ``None`` it is taken from ``vertices``, which is an upper bound on
        what [`array.index_bound`][ordito.array.index_bound] would report and serves the same
        purpose here -- it is only the radix the halfedge keys are packed against -- without the
        host readback that inferring it costs.
    face_normals
        ``(n_faces,)`` unit face normals and matching areas from
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]; recomputed together
        when either is ``None``. [`Trimesh.face_normals`][ordito.mesh.Trimesh.face_normals] and
        [`Trimesh.face_areas`][ordito.mesh.Trimesh.face_areas] cache the pair.
    face_areas
        ``(n_faces,)`` face areas; see ``face_normals``.

    Returns
    -------
    odt.Array2dVec3
        ``(n_faces, 3)`` unit normals on ``faces.device``; entry ``(f, k)`` belongs to corner
        ``k`` of face ``f``. A corner whose weighted sum cancels to zero comes back as the zero
        vector, matching [`weighted_vertex_normals`][ordito.vertices.weighted_vertex_normals].

    Raises
    ------
    TypeError
        If ``crease_edges`` is given and is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``crease_edges`` is given and does not have two columns, or ``weighting`` is not one of
        the two names above. Also propagated from
        [`halfedge_twins`][ordito.halfedge.halfedge_twins] when the mesh is **not edge-manifold**:
        rotating about a vertex is what this computes, and an edge with three faces has no rotation.
        Pass ``twins`` yourself if you have already resolved that.
    RuntimeError
        If ``vertices``, ``faces``, ``crease_edges``, ``twins``, ``face_normals`` and
        ``face_areas`` are not all on one device.

    Examples
    --------
    ```python
    creases = od.seams.crease_edges(v, f, angle=0.5)
    normals = od.triangles.corner_normals(v, f, creases)
    assert normals.shape == (f.shape[0] // 3, 3)
    ```

    See Also
    --------
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
        The flat weighting: one normal per face.
    [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="angle"``
        The smooth weighting this reduces to when there are no creases.
    [`crease_edges`][ordito.seams.crease_edges]
        Produces the hard-edge set.
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        crease_edges=crease_edges,
        twins=twins,
        face_normals=face_normals,
        face_areas=face_areas,
    )
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return odt.as_array2d(odt.empty_2d((0, 3), wp.vec3, device=device), wp.vec3)
    if n_vertices is None:
        # ``vertices.shape[0]`` rather than ``index_bound(faces)``: this count is only the radix
        # the edge keys below are packed against, so any bound above the largest index is equally
        # correct, and a mesh with trailing unreferenced vertices simply packs against a slightly
        # wider one. Inferring the tight bound is a device reduction plus a host readback, which is
        # a fifth of the whole call.
        n_vertices = vertices.size
    if twins is None:
        twins = od.halfedge.halfedge_twins(faces, n_vertices=n_vertices)

    crease_keys = _launch.empty(0, dtype=wp.uint64, device=device)
    if crease_edges is not None:
        odt.ensure_edge_pairs(crease_edges, "crease_edges")
        if int(crease_edges.shape[0]) > 0:
            crease_keys = od.grouping.sorted_undirected_edge_keys(crease_edges, n_vertices)

    if face_normals is None or face_areas is None:
        face_normals, face_areas = face_normals_and_areas(vertices, faces)
    if weighting == "angle":
        # ``(n_faces, 3)`` reshaped flat is already corner-indexed: row ``f`` column ``k`` is
        # corner ``3f + k``, the same index the kernel's thread carries.
        corner_weights = face_angles(vertices, faces).reshape(3 * n_faces)
    elif weighting == "area":
        # The face's own area, indexed by ``halfedge // 3`` inside the kernel. Broadcasting it to
        # the three corners here instead cost an ``arange_repeat`` launch and a gather, for a
        # division the kernel already performs.
        corner_weights = face_areas
    else:
        raise ValueError(f"weighting must be 'angle' or 'area', got {weighting!r}")

    normals = odt.empty_2d((n_faces, 3), wp.vec3, device=device)
    _launch.launch(
        kernel_triangles.corner_normals,
        dim=3 * n_faces,
        inputs=[
            vertices,
            faces,
            twins,
            face_normals,
            corner_weights,
            weighting == "angle",
            crease_keys,
            wp.uint64(n_vertices),
            normals,
        ],
        device=device,
    )
    return odt.as_array2d(normals, wp.vec3)


FaceQualityMetric = Literal["aspect_ratio", "radius_ratio", "area_max_side", "mean_ratio", "area"]
"""Shape-quality measure selected by [`face_quality`][ordito.triangles.face_quality]."""


def face_quality(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: FaceQualityMetric = "aspect_ratio",
) -> wp.array[wp.float32]:
    """
    Per-face shape-quality measure.

    This is the quantity the library already uses internally to decide whether a triangle is worth
    keeping — [`isotropic_remesh`][ordito.remesh.isotropic_remesh] and
    [`flip_to_delaunay`][ordito.remesh.flip_to_delaunay] gate on ``aspect_ratio``, and the
    hole-filling cost functions weight candidate triangles by it — exposed so a caller can inspect
    or threshold it directly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    metric
        Which measure to compute. All but ``"area"`` are invariant to a uniform scaling of the
        mesh, and all but ``"aspect_ratio"`` are *larger is better*:

        - ``"aspect_ratio"`` (default) — circumradius over twice the inradius. ``1`` for an
          equilateral triangle and unbounded above, so **smaller is better**; a degenerate
          triangle reads ``+inf``. This is the measure the remeshing gates use.
        - ``"radius_ratio"`` — inradius over circumradius, rescaled so an equilateral triangle
          reads ``1``; ``0`` when degenerate. MeshLab's ``inradius/circumradius``.
        - ``"area_max_side"`` — twice the area over the longest side squared, ``sqrt(3)/2`` at
          best. MeshLab's ``area/max side`` (scale-invariant despite the name).
        - ``"mean_ratio"`` — ``4 sqrt(3) A / (a^2 + b^2 + c^2)``, ``1`` at best. MeshLab's
          ``Mean ratio``.
        - ``"area"`` — the plain triangle area, for parity with MeshLab's ``Area``; identical to
          the second return of
          [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas].

    Returns
    -------
    wp.array[wp.float32]
        ``(n_faces,)`` quality values on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the listed names.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
    [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask]
    [`ordito.remesh.isotropic_remesh`][ordito.remesh.isotropic_remesh]
    """
    require_same_device(vertices=vertices, faces=faces)
    if metric not in _QUALITY_METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {sorted(_QUALITY_METRICS)}")
    f = faces.size // 3
    out_quality = _launch.empty(f, dtype=wp.float32, device=vertices.device)
    _launch.launch(
        kernel_triangles.face_quality,
        dim=f,
        inputs=[vertices, faces, _QUALITY_METRICS[metric], out_quality],
        device=vertices.device,
    )
    return out_quality


def face_centroids(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    """
    Barycentre of every face: the mean of its three corners.

    Not to be confused with [`surface_centroid`][ordito.measures.surface_centroid], which is the
    *mesh's* single area-weighted centre. This is one point per triangle and no weighting is
    involved.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_faces,)`` face barycentres on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`centroid`][ordito.measures.surface_centroid]
    [`moments`][ordito.measures.moments]
    ``igl.barycenter``
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    out_centroids = _launch.empty(n_faces, dtype=wp.vec3, device=vertices.device)
    if n_faces == 0:
        return out_centroids
    _launch.launch(
        kernel_triangles.face_centroids,
        dim=n_faces,
        inputs=[vertices, faces, out_centroids],
        device=vertices.device,
    )
    return out_centroids


def face_signed_volumes(
    vertices: wp.array[wp.vec3] | wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    apex: wp.vec3 | wp.vec3d | None = None,
) -> wp.array[wp.float32] | wp.array[wp.float64]:
    """
    Signed volume of the tetrahedron ``(apex, v0, v1, v2)`` for each face.

    ``dot(v0 - apex, cross(v1 - apex, v2 - apex)) / 6``, so the sign follows the face's winding:
    positive where the face turns its front to the apex. Summed over a closed, consistently wound
    surface this is the enclosed volume ([`volume`][ordito.measures.volume]) and is independent of
    ``apex``; per face it is not, which is why the argument exists —
    [`sample_volume`][ordito.sample.sample_volume] fans from the surface centroid so that every
    tetrahedron is positive on a star-shaped mesh, and uses the array as a cumulative distribution.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` positions, ``wp.vec3`` or ``wp.vec3d``. The output dtype follows: pass
        ``vec3d`` where the sum's low digits matter, as
        [`filter_laplacian`][ordito.smoothing.filter_laplacian]'s volume constraint does.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    apex
        Common apex of every tetrahedron, in ``vertices``' dtype. ``None`` (the default) means the
        world origin, which is what every whole-mesh volume wants.

    Returns
    -------
    wp.array[wp.float32] | wp.array[wp.float64]
        ``(n_faces,)`` signed volumes on ``vertices.device``, in ``vertices``' scalar type. Empty
        when ``faces`` is empty.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`volume`][ordito.measures.volume]
        The sum of these over the whole mesh.
    [`is_volume`][ordito.validation.is_volume]
        Whether that sum is a meaningful volume at all.
    [`face_centroids`][ordito.triangles.face_centroids]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    device = vertices.device
    volumes: wp.array[wp.float32] | wp.array[wp.float64] = (
        _launch.empty(n_faces, dtype=wp.float64, device=device)
        if vertices.dtype is wp.vec3d
        else _launch.empty(n_faces, dtype=wp.float32, device=device)
    )
    if n_faces == 0:
        return volumes
    if apex is None:
        apex = wp.vec3d() if vertices.dtype is wp.vec3d else wp.vec3()
    _launch.launch(
        kernel_triangles.FACE_SIGNED_VOLUMES[vertices.dtype],
        dim=n_faces,
        inputs=[vertices, faces, apex, volumes],
        device=device,
    )
    return volumes


def face_nondegenerate_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    """
    Flag triangles with non-zero area.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` mask on ``vertices.device``; ``True`` where the triangle area is non-zero.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`trimesh.triangles.nondegenerate`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    f = faces.size // 3
    out_nondegenerate = _launch.empty(f, dtype=wp.bool, device=vertices.device)
    _launch.launch(
        kernel_triangles.face_nondegenerate_mask,
        dim=f,
        inputs=[vertices, faces, out_nondegenerate],
        device=vertices.device,
    )
    return out_nondegenerate


def barycentric_to_points(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], barycentric: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Convert barycentric coordinates to Cartesian points, one triangle per row.

    Operates on the "triangle soup" convention shared with `trimesh.triangles`: row ``i``
    of ``barycentric`` is evaluated against triangle ``i`` of ``faces`` (not every triangle
    against every coordinate).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    barycentric
        ``(n_faces,)`` barycentric coordinates ``(u, v, w)``, aligned one-to-one with the
        triangles in ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_faces,)`` Cartesian points on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``barycentric`` are not all on one device.

    See Also
    --------
    [`points_to_barycentric`][ordito.triangles.points_to_barycentric]
        The inverse: recovers barycentric coordinates from Cartesian points.
    [`trimesh.triangles.barycentric_to_points`][]
    """
    require_same_device(vertices=vertices, faces=faces, barycentric=barycentric)
    f = faces.size // 3
    out_points = _launch.empty(f, dtype=wp.vec3, device=vertices.device)
    _launch.launch(
        kernel_triangles.barycentric_to_points,
        dim=f,
        inputs=[vertices, faces, barycentric, out_points],
        device=vertices.device,
    )
    return out_points


def points_to_barycentric(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], points: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Convert Cartesian points to barycentric coordinates, one triangle per row.

    Operates on the "triangle soup" convention shared with `trimesh.triangles`: row ``i`` of
    ``points`` is projected onto triangle ``i`` of ``faces`` (not every triangle against every
    point).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    points
        ``(n_faces,)`` query positions, aligned one-to-one with the triangles in ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_faces,)`` barycentric coordinates ``(u, v, w)`` on ``vertices.device``. A triangle with
        exactly zero area has no barycentric frame; it is answered along the triangle's longest
        edge, which is what such a "triangle" actually is.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``points`` are not all on one device.

    See Also
    --------
    [`barycentric_to_points`][ordito.triangles.barycentric_to_points]
        The inverse: recovers Cartesian points from barycentric coordinates.
    [`trimesh.triangles.points_to_barycentric`][]

    Notes
    -----
    [`trimesh.triangles.points_to_barycentric`][] offers a ``method`` choice between a
    cross-product ratio and Cramer's rule on the Gram system, and this function deliberately does
    not: the two are algebraically equal, and the Cramer form is less accurate at every triangle
    shape and scale measured and more accurate at none, returning wholly wrong *finite* values on
    the slivers an ordinary decimated mesh carries. The two also cost the same, both being
    launch-bound, so there was nothing to trade. The measured table is in
    ``ordito.kernels.triangles.point_barycentric``.
    """
    require_same_device(vertices=vertices, faces=faces, points=points)
    f = faces.size // 3
    out_barycentric = _launch.empty(f, dtype=wp.vec3, device=vertices.device)
    _launch.launch(
        kernel_triangles.points_to_barycentric,
        dim=f,
        inputs=[vertices, faces, points, out_barycentric],
        device=vertices.device,
    )
    return out_barycentric


def closest_point(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], points: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Closest point on each triangle to a query point, one triangle per row.

    Operates on the "triangle soup" convention shared with `trimesh.triangles`: row ``i`` of
    ``points`` is projected onto triangle ``i`` of ``faces`` (not every triangle against every
    point). For nearest-triangle queries over an entire mesh, see
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] instead.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    points
        ``(n_faces,)`` query positions, aligned one-to-one with the triangles in ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_faces,)`` closest points on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``points`` are not all on one device.

    See Also
    --------
    [`trimesh.triangles.closest_point`][]
    """
    require_same_device(vertices=vertices, faces=faces, points=points)
    f = faces.size // 3
    out_closest = _launch.empty(f, dtype=wp.vec3, device=vertices.device)
    _launch.launch(
        kernel_triangles.closest_point,
        dim=f,
        inputs=[vertices, faces, points, out_closest],
        device=vertices.device,
    )
    return out_closest
