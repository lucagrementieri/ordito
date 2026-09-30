"""
Move a scalar or vector field between the mesh's elements: faces, vertices and edges.

Averaging is the only honest transfer between element types on a mesh, and the direction determines
what it costs: face-to-vertex
([`average_onto_vertices`][triwarp.interpolation.average_onto_vertices]) and edge-to-vertex
([`average_from_edges_onto_vertices`][triwarp.interpolation.average_from_edges_onto_vertices]) are
gathers over the incident elements, while vertex-to-face
([`average_onto_faces`][triwarp.interpolation.average_onto_faces]) is the mean of exactly three
corners and needs no adjacency at all.

These are unweighted means. For an area- or angle-weighted transfer of *normals* specifically, see
[`triwarp.vertices`][triwarp.vertices]; for a UV-space resampling, see
[`triwarp.texture`][triwarp.texture].

[`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices] moves a field between two
*different* meshes instead of between one mesh's element types, by closest-point projection onto the
source surface — the operation you need after a remesh or a decimation has changed the vertex set
under a field. [`interpolate_from_points`][triwarp.interpolation.interpolate_from_points] drops the
mesh requirement entirely: it interpolates a field known on a scattered *cloud*, which is the only
function here whose source has no connectivity to average over, and so the only one that needs a
kernel width rather than an incidence structure.
"""

from __future__ import annotations

from typing import TypeVar

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_same_device
from triwarp.kernels import array as kernel_array
from triwarp.kernels import interpolation as kernel_interpolation
from triwarp.kernels import scatter as kernel_scatter

DType = TypeVar("DType")
"""
Element type of a transferred field: ``wp.float32``, ``wp.vec2`` or ``wp.vec3``.

The set is bounded by the weights, which are ``float32``: Warp requires both operands of a product
to share a scalar type, so a field whose scalar is not ``float32`` -- a ``float64`` one, say --
cannot be scaled by them. A dtype outside the set raises ``KeyError`` naming the kernel rather than
silently recompiling the module (``triwarp.kernels.array.OverloadTable`` -- kernel modules
are not part of the rendered reference, so this one is not a link).
"""


def average_onto_faces(
    faces: wp.array[wp.int32], vertex_values: wp.array[wp.float32]
) -> wp.array[wp.float32]:
    """
    Move a scalar field defined on vertices to faces by averaging.

    Each face value is the mean of ``vertex_values`` at its three corners.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    vertex_values
        Length-``n_vertices`` scalar field defined on vertices.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_faces`` scalar field defined on faces. Empty when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``vertex_values`` are not all on one device.
    """
    require_same_device(faces=faces, vertex_values=vertex_values)
    n_faces = faces.size // 3
    device = vertex_values.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_face_values = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_interpolation.average_onto_faces,
        dim=n_faces,
        inputs=[faces, vertex_values, out_face_values],
        device=device,
    )
    return out_face_values


def average_onto_vertices(
    n_vertices: int, faces: wp.array[wp.int32], face_values: wp.array[wp.float32]
) -> wp.array[wp.float32]:
    """
    Move a scalar field defined on faces to vertices by averaging.

    Each vertex value is the mean of ``face_values`` over incident triangle corners.
    Vertices referenced by no face divide by a zero valence and are ``nan``, matching
    the unguarded division in ``igl::average_onto_vertices``.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    face_values
        Length-``n_faces`` scalar field defined on faces.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_vertices`` scalar field defined on vertices.

    Raises
    ------
    ValueError
        If ``face_values`` does not have one entry per face.
    RuntimeError
        If ``faces`` and ``face_values`` are not all on one device.

    See Also
    --------
    [`average_from_edges_onto_vertices`][triwarp.interpolation.average_from_edges_onto_vertices]
        The same scatter-sum-then-divide over edges instead of faces. The two disagree on a
        **zero-valence** vertex on purpose, each mirroring its own igl function: this one divides
        unguarded and yields ``nan``, that one guards and yields ``0``.
    """
    require_same_device(faces=faces, face_values=face_values)
    device = faces.device
    n_faces = faces.size // 3
    if face_values.size != n_faces:
        raise ValueError(
            f"face_values must have one entry per face, got {face_values.size} for {n_faces} faces"
        )

    out_sum = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    out_valence = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_scatter.scatter_face_values_sum_and_valence,
        dim=n_faces,
        inputs=[faces, face_values, out_sum, out_valence],
        device=device,
    )
    wp.map(wp.div, out_sum, out_valence, out=out_sum)
    return out_sum


def average_from_edges_onto_vertices(
    n_vertices: int,
    faces: wp.array[wp.int32],
    edges: twt.Array2dInt32,
    edges_orientation: twt.Array2dInt32,
    edge_values: wp.array[wp.float32],
) -> wp.array[wp.float32]:
    """
    Move a scalar field defined on edges to vertices by averaging.

    (``igl::average_from_edges_onto_vertices``).

    For each face half-edge with non-negative orientation, ``edge_values`` at the
    corresponding unique edge is accumulated onto both of its endpoint vertices; each
    vertex value is then the mean over its incident (positively oriented) half-edges.
    Vertices with zero valence are left at ``0``, matching the guarded division in
    ``igl::average_from_edges_onto_vertices``.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges
        Shape ``(n_faces, 3)`` mapping from each face half-edge to a unique edge index,
        as produced by ``igl::orient_halfedges``.
    edges_orientation
        Shape ``(n_faces, 3)`` half-edge orientation relative to its unique edge
        (``igl::orient_halfedges``); half-edges with a negative value are skipped.
    edge_values
        Length-``n_unique_edges`` scalar field defined on unique edges.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_vertices`` scalar field defined on vertices.

    Raises
    ------
    ValueError
        If ``edges`` or ``edges_orientation`` is not ``(n_faces, 3)``.
    RuntimeError
        If ``faces``, ``edges``, ``edges_orientation`` and ``edge_values`` are not all on one
        device.

    See Also
    --------
    [`average_onto_vertices`][triwarp.interpolation.average_onto_vertices]
        The same scatter-sum-then-divide over faces instead of edges. The two disagree on a
        **zero-valence** vertex on purpose, each mirroring its own igl function: this one guards
        the division and yields ``0``, that one divides unguarded and yields ``nan``.
    """
    require_same_device(
        faces=faces, edges=edges, edges_orientation=edges_orientation, edge_values=edge_values
    )
    device = faces.device
    n_faces = faces.size // 3
    # Both tables are indexed at ``[f, j]`` for every face and corner, so a table built for another
    # mesh is an out-of-bounds read rather than a wrong answer -- on the CPU device, where a Warp
    # array is host heap, a silent one.
    for name, table in (("edges", edges), ("edges_orientation", edges_orientation)):
        if (int(table.shape[0]), int(table.shape[1])) != (n_faces, 3):
            raise ValueError(
                f"{name} must be ({n_faces}, 3), one row per face, got {tuple(table.shape)}"
            )

    out_sum = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    out_valence = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_scatter.scatter_edges_sum_and_valence,
        dim=n_faces,
        inputs=[faces, edges, edges_orientation, edge_values, out_sum, out_valence],
        device=device,
    )
    wp.map(kernel_array.divide_if_positive, out_sum, out_valence, out=out_sum)
    return out_sum


def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[DType],
    target_vertices: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
) -> tuple[wp.array[DType], wp.array[wp.float32]]:
    """
    Resample a source mesh's per-vertex field onto another mesh's vertices.

    For each target vertex, the closest point on the *source surface* is found and the source field
    is interpolated there barycentrically. This is the transfer to use when two meshes describe the
    same shape at different resolutions — after a remesh, a decimation or a reconstruction — and a
    field computed on one of them has to follow. It is MeshLab's ``transfer_attributes_per_vertex``
    with ``vertexsampling=False``.

    Because the closest point is taken on the surface rather than at the nearest *vertex*, the
    result is continuous in the target positions and exact for a field that is already linear on the
    source triangles — which is what makes it safe to chain (a transfer onto a refinement of the
    same mesh reproduces the field, not a blur of it).

    Parameters
    ----------
    source_vertices
        ``(n_source,)`` source mesh vertex positions.
    source_faces
        Length-``3 * n_source_faces`` ``wp.int32`` source triangle index buffer.
    source_values
        Length-``n_source`` field on the source vertices. ``wp.float32`` for a scalar, ``wp.vec2``
        for a UV pair, ``wp.vec3`` for a normal or a colour; see
        [`DType`][triwarp.interpolation.DType] for why those three and nothing else.
    target_vertices
        ``(n_target,)`` positions to sample at — usually another mesh's vertices, but any point set
        will do.
    max_dist
        Maximum search radius per target vertex. Targets with no source face within it keep the
        zero-filled output and report ``inf`` in the returned distance, which is how a caller
        detects them. When ``None``, derived from the box enclosing both meshes, so every target
        hits.

    Returns
    -------
    values : wp.array[DType]
        Length-``n_target`` transferred field on ``target_vertices.device``. Zero-filled wherever
        the closest-point query missed.
    distance : wp.array[wp.float32]
        Length-``n_target`` distance from each target vertex to the source surface — the transfer's
        own confidence measure, and ``inf`` for a miss.

    Raises
    ------
    ValueError
        If ``source_values`` does not have one entry per source vertex.
    RuntimeError
        If ``source_vertices``, ``source_faces``, ``source_values`` and ``target_vertices`` are not
        all on one device.

    See Also
    --------
    [`average_onto_vertices`][triwarp.interpolation.average_onto_vertices]
    [`triwarp.proximity.closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    [`triwarp.triangles.barycentric_to_points`][triwarp.triangles.barycentric_to_points]
    """
    require_same_device(
        source_vertices=source_vertices,
        source_faces=source_faces,
        source_values=source_values,
        target_vertices=target_vertices,
    )
    device = target_vertices.device
    n_target = target_vertices.size
    n_source = source_vertices.size
    if source_values.size != n_source:
        raise ValueError(
            f"source_values must have one entry per source vertex, got "
            f"{source_values.size} for {n_source} vertices"
        )

    out_values = wp.zeros(n_target, dtype=source_values.dtype, device=device)
    if n_target == 0 or source_faces.size == 0:
        return out_values, wp.full(n_target, float("inf"), dtype=wp.float32, device=device)

    closest, distance, face_id = tw.proximity.closest_point_on_mesh(
        source_vertices, source_faces, target_vertices, max_dist=max_dist
    )
    wp.launch(
        kernel_interpolation.TRANSFER_ONTO_VERTICES[source_values.dtype],
        dim=n_target,
        inputs=[
            source_vertices,
            source_faces,
            source_values,
            closest,
            face_id,
            out_values,
            distance,
        ],
        device=device,
    )
    return out_values, distance


def transfer_through_operator(
    values: wp.array[DType], operator: wps.BsrMatrix[wp.float32]
) -> wp.array[DType]:
    """
    Carry a per-vertex field through a topology edit, using the edit's own interpolation operator.

    The companion of [`triwarp.remesh.subdivide_loop`][triwarp.remesh.subdivide_loop]'s
    ``return_operator``: that pass moves *and* creates vertices, so an output vertex is an affine
    combination of several input ones and no index map can express the correspondence. The operator
    can, and applying it to a field is the same product that produced the new positions:
    ``transfer_through_operator(vertices, P)`` reproduces the subdivided vertex buffer, which is
    what makes this exact rather than a resampling.

    Prefer this over
    [`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices] whenever the operator
    is available: that one projects onto the source surface and is therefore lossy by construction,
    where this one applies the very weights the edit used.

    Parameters
    ----------
    values
        Length-``n_source`` field on the input vertices. Any **float32-based** Warp dtype closed
        under scaling and addition: ``wp.float32`` for a scalar, ``wp.vec2`` for a UV, ``wp.vec3``
        for a normal or a colour. Not ``wp.float64`` -- see Notes.
    operator
        ``(n_out, n_source)`` ``float32`` interpolation matrix, as returned by
        ``subdivide_loop(..., return_operator=True)``.

    Returns
    -------
    wp.array[DType]
        Length-``n_out`` transferred field on ``values.device``, of the same dtype as ``values``.

    Raises
    ------
    ValueError
        If the operator's column count does not match ``values``.
    RuntimeError
        If ``values`` and ``operator`` are not on one device.

    Examples
    --------
    ```python
    field = tw.vertices.vertex_normals(v, f)  # per-vertex field
    fine_v, fine_f, prolongation = tw.remesh.subdivide_loop(v, f, return_operator=True)
    fine_field = tw.interpolation.transfer_through_operator(field, prolongation)
    ```

    Notes
    -----
    An output row the operator leaves empty gets the dtype's zero rather than uninitialized memory,
    so a partial operator is safe to apply.

    ``warp.sparse.bsr_mv`` would serve for a ``float32`` field and nothing else -- its vector dtype
    has to match the matrix's 1x1 block -- so this goes through a small CSR kernel instead, which is
    what makes the ``wp.vec2`` and ``wp.vec3`` cases (a UV, a colour) reachable at all.

    The field's scalar must be ``float32`` because the operator's weights are: Warp requires both
    operands of a product to share a scalar type, so a ``wp.float64`` field does not compile against
    a float32 operator. Cast it with ``wp.utils.array_cast`` if that is what you hold. The
    restriction is honest rather than incidental -- these operators are assembled from float32
    vertex data, so carrying a float64 field through one would advertise precision the weights do
    not have.

    See Also
    --------
    [`triwarp.remesh.subdivide_loop`][triwarp.remesh.subdivide_loop]
        The one edit in this package that publishes such an operator.
    [`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices]
        The lossy alternative, for when the two meshes are related only by geometry.
    [`triwarp.array.gather`][triwarp.array.gather]
        What to use instead where the edit *can* report an index map -- ``split_edges`` and
        ``subdivide_to_size`` both do, and a gather through their ``index`` is the whole transfer.
    """
    require_same_device(values=values, operator=operator)
    device = values.device
    n_source = values.size
    n_out = int(operator.nrow)
    if int(operator.ncol) != n_source:
        raise ValueError(f"operator has {operator.ncol} columns but values has {n_source} entries")

    out_values = wp.zeros(n_out, dtype=values.dtype, device=device)
    if n_out == 0 or n_source == 0:
        return out_values

    wp.launch(
        kernel_interpolation.APPLY_TRANSFER_OPERATOR[values.dtype],
        dim=n_out,
        inputs=[operator.offsets, operator.columns, operator.values, values, out_values],
        device=device,
    )
    return out_values


def interpolate_from_points(
    source_points: wp.array[wp.vec3],
    source_values: wp.array[DType],
    query_points: wp.array[wp.vec3],
    radius: float,
    *,
    k: int | None = None,
    sharpness: float = 2.0,
    null_value: float = 0.0,
) -> wp.array[DType]:
    """
    Interpolate a field known on a scattered point cloud at arbitrary query points.

    A Gaussian-weighted mean of each query's neighbours, weight
    ``exp(-(sharpness * d / radius) ** 2)`` — the one transfer in this module that needs no mesh on
    either side, since the source is a bare cloud. Use it to bring a measured or simulated field
    onto a mesh's vertices, onto [`grid_points`][triwarp.voxels.grid_points], or onto any other
    sample set; use
    [`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices] instead when the source
    *is* a mesh, since projecting onto its surface is exact for a piecewise-linear field where this
    is a smoothing, and
    [`sample_grid_trilinear`][triwarp.voxels.sample_grid_trilinear] when the source is a dense
    lattice, where trilinear weights are exact rather than a kernel estimate.

    Parameters
    ----------
    source_points
        ``(n_source,)`` positions the field is known at.
    source_values
        Length-``n_source`` field on those points. ``wp.float32`` for a scalar, ``wp.vec2`` for a
        UV pair, ``wp.vec3`` for a vector; see [`DType`][triwarp.interpolation.DType] for why those
        three and nothing else.
    query_points
        ``(n_query,)`` positions to interpolate at.
    radius
        Length scale of the kernel, and — unless ``k`` is given — the footprint: only sources within
        it contribute. **Required**, and absolute: it is in the coordinates' own units, so a cloud
        rescaled by 100 needs a radius rescaled by 100. Derive it from the data rather than guessing
        (e.g. [`mean_edge_length`][triwarp.edges.mean_edge_length] or
        [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]).
    k
        When given, use the ``k`` nearest sources of each query as the footprint instead of every
        source within ``radius``, at any distance. ``radius`` still sets the kernel's length scale.
    sharpness
        Falloff: a larger value reduces the influence of distant sources. At ``sharpness``,
        a source at ``radius`` weighs ``exp(-sharpness ** 2)``.
    null_value
        Value written where a query has no neighbour at all (or where every weight underflowed to
        zero). For a ``wp.vec3`` field it fills every component.

    Returns
    -------
    wp.array[DType]
        Length-``n_query`` interpolated field on ``query_points.device``, with ``source_values``'
        dtype.

    Raises
    ------
    ValueError
        If ``source_values`` does not have one entry per source point, if ``radius`` is not
        positive, or if ``k`` is given and is not positive.
    RuntimeError
        If ``source_points``, ``source_values`` and ``query_points`` are not all on one device.

    Notes
    -----
    A query that coincides with a source takes that source's value exactly rather than blending its
    neighbours, so the interpolant reproduces the data at the data. VTK's kernels do the same.

    This is ``vtkPointInterpolator`` with a ``vtkGaussianKernel``, which pyvista exposes as
    ``DataSet.interpolate``. Two conventions of the reference are deliberately not copied: it
    clamps ``sharpness`` up to ``1.0`` (so its own ``0.5`` behaves as ``1.0``), and it offers
    ``mask_points`` / ``closest_point`` fallbacks for a query with no neighbour. The mask is
    ``counts == 0`` from [`query_ball_count`][triwarp.neighbors.query_ball_count] and the
    closest-point fallback is this function at ``k=1``, so neither needs a mode of its own.

    See Also
    --------
    [`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices]
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    """
    require_same_device(
        source_points=source_points, source_values=source_values, query_points=query_points
    )
    device = query_points.device
    n_source = source_points.size
    n_query = query_points.size
    if source_values.size != n_source:
        raise ValueError(
            f"source_values must have one entry per source point, got "
            f"{source_values.size} for {n_source} points"
        )
    if not float(radius) > 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if k is not None and int(k) <= 0:
        raise ValueError(f"k must be positive when given, got {k}")

    out_values = wp.full(n_query, null_value, dtype=source_values.dtype, device=device)
    if n_query == 0 or n_source == 0:
        return out_values

    inverse_scale = wp.float32(float(sharpness) / float(radius))
    if k is None:
        # The mean is reduced during the grid walk, so no neighbour list is ever materialized: the
        # same hash grid and predicate ``query_ball_with_offsets`` uses, visiting the neighbours in
        # the order its list would hold them.
        grid = tw.neighbors.hashgrid_from_points(source_points, float(radius))
        wp.launch(
            kernel_interpolation.INTERPOLATE_FROM_POINTS_IN_BALL[source_values.dtype],
            dim=n_query,
            inputs=[
                source_points,
                source_values,
                query_points,
                grid.id,
                wp.float32(radius),
                inverse_scale,
                out_values,
            ],
            device=device,
        )
        return out_values

    # The unfilled slots of a row carry index -1, which the kernel skips.
    row_indices, row_distances = tw.neighbors.query_nearest(source_points, query_points, int(k))
    n_slots = n_query * int(k)
    wp.launch(
        kernel_interpolation.INTERPOLATE_FROM_POINTS_NEAREST[source_values.dtype],
        dim=n_query,
        inputs=[
            source_values,
            row_indices.reshape((n_slots,)),
            row_distances.reshape((n_slots,)),
            wp.int32(k),
            inverse_scale,
            out_values,
        ],
        device=device,
    )
    return out_values
