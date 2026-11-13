"""
Move a scalar or vector field between the mesh's elements: faces, vertices and edges.

Averaging is the only honest transfer between element types on a mesh, and the direction determines
what it costs: face-to-vertex
([`average_onto_vertices`][ordito.interpolation.average_onto_vertices]) and edge-to-vertex
([`average_from_edges_onto_vertices`][ordito.interpolation.average_from_edges_onto_vertices]) are
gathers over the incident elements, while vertex-to-face
([`average_onto_faces`][ordito.interpolation.average_onto_faces]) is the mean of exactly three
corners and needs no adjacency at all.

These are unweighted means. For an area- or angle-weighted transfer of *normals* specifically, see
[`ordito.normals`][ordito.normals]; for a UV-space resampling, see
[`ordito.texture`][ordito.texture].

[`Provenance`][ordito.interpolation.Provenance] moves fields across a topology edit: every function
that rebuilds a mesh's connectivity (a subdivision, a decimation, a selection, a cut, a repair)
reports one with ``return_provenance=True``, and it carries any per-vertex or per-face attribute --
a colour, a UV, a label -- onto the edit's output exactly, with the weights the edit itself used.
[`transfer_onto_vertices`][ordito.interpolation.transfer_onto_vertices] moves a field between two
*different* meshes instead, by closest-point projection onto the source surface — the operation
for meshes related only by geometry, such as a reconstruction or a remesh that reprojects its
vertices. [`interpolate_barycentric`][ordito.interpolation.interpolate_barycentric] reads a field at
points given by a face and barycentric coordinates, the form every surface sample and closest-point
query reports. [`interpolate_from_points`][ordito.interpolation.interpolate_from_points] drops the
mesh requirement entirely: it interpolates a field known on a scattered *cloud*, which is the only
function here whose source has no connectivity to average over, and so the only one that needs a
kernel width rather than an incidence structure.
"""

from __future__ import annotations

import math
from typing import NamedTuple, TypeVar, cast

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._common import (
    ball_grid_cell_width,
    require_nonempty_mesh,
    require_same_device,
    scan_count,
    search_order,
    tiled_ball_walk,
)
from ordito.config import is_deterministic
from ordito.kernels import array as kernel_array
from ordito.kernels import interpolation as kernel_interpolation
from ordito.kernels import neighbors as kernel_neighbors
from ordito.kernels import scatter as kernel_scatter

kernel_array.declare_block_dims(
    kernel_interpolation.interpolate_from_points_in_ball_tiled,
    (kernel_neighbors.TILED_BALL_WALK_BLOCK_DIM,),
)

DType = TypeVar("DType")
"""
Element type of a transferred field: ``wp.float32``, ``wp.vec2`` or ``wp.vec3``.

The set is bounded by the weights, which are ``float32``: Warp requires both operands of a product
to share a scalar type, so a field whose scalar is not ``float32`` -- a ``float64`` one, say --
cannot be scaled by them. A dtype outside the set raises ``KeyError`` naming the kernel rather than
silently recompiling the module (``ordito.kernels.array.OverloadTable`` -- kernel modules
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
        ``(3 * n_faces,)`` face index buffer.
    vertex_values
        ``(n_vertices,)`` scalar field defined on vertices.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_faces,)`` scalar field defined on faces. Empty when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``faces`` and ``vertex_values`` are not all on one device.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([0, 1, 2, 1, 3, 2], dtype=wp.int32)
    values = wp.array([0.0, 3.0, 6.0, 9.0], dtype=wp.float32)
    print(od.interpolation.average_onto_faces(faces, values).numpy())  # [3. 6.]
    ```
    """
    require_same_device(faces=faces, vertex_values=vertex_values)
    n_faces = faces.size // 3
    device = vertex_values.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    out_face_values = _launch.empty(n_faces, dtype=wp.float32, device=device)
    _launch.launch(
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
    the unguarded division in ``igl::average_onto_vertices``. Under
    [`is_deterministic`][ordito.config.is_deterministic] each sum is formed exactly and rounded
    once, so the result repeats bit for bit.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        ``(3 * n_faces,)`` face index buffer.
    face_values
        ``(n_faces,)`` scalar field defined on faces.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_vertices,)`` scalar field defined on vertices.

    Raises
    ------
    ValueError
        If ``face_values`` does not have one entry per face.
    RuntimeError
        If ``faces`` and ``face_values`` are not all on one device.

    See Also
    --------
    [`average_from_edges_onto_vertices`][ordito.interpolation.average_from_edges_onto_vertices]
        The same scatter-sum-then-divide over edges instead of faces. The two disagree on a
        **zero-valence** vertex on purpose, each mirroring its own igl function: this one divides
        unguarded and yields ``nan``, that one guards and yields ``0``.

    Examples
    --------
    Vertices 1 and 2 are shared by both faces and take the mean of their values; vertex 4 is in no
    face and is ``nan``:

    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([0, 1, 2, 1, 3, 2], dtype=wp.int32)
    face_values = wp.array([2.0, 4.0], dtype=wp.float32)
    values = od.interpolation.average_onto_vertices(5, faces, face_values)
    print(values.numpy())  # [ 2.  3.  3.  4. nan]
    ```
    """
    require_same_device(faces=faces, face_values=face_values)
    device = faces.device
    n_faces = faces.size // 3
    if face_values.size != n_faces:
        raise ValueError(
            f"face_values must have one entry per face, got {face_values.size} for {n_faces} faces"
        )

    return _vertex_mean(
        n_vertices,
        n_faces,
        kernel_scatter.SCATTER_FACE_VALUES_SUM_AND_VALENCE,
        [faces, face_values],
        wp.div,
        kernel_interpolation.mean_rounded,
        device,
    )


def average_from_edges_onto_vertices(
    n_vertices: int,
    faces: wp.array[wp.int32],
    edges: odt.Array2dInt32,
    edges_orientation: odt.Array2dInt32,
    edge_values: wp.array[wp.float32],
) -> wp.array[wp.float32]:
    """
    Move a scalar field defined on edges to vertices by averaging.

    (``igl::average_from_edges_onto_vertices``).

    For each face half-edge with non-negative orientation, ``edge_values`` at the
    corresponding unique edge is accumulated onto both of its endpoint vertices; each
    vertex value is then the mean over its incident (positively oriented) half-edges.
    Vertices with zero valence are left at ``0``, matching the guarded division in
    ``igl::average_from_edges_onto_vertices``. Under
    [`is_deterministic`][ordito.config.is_deterministic] each sum is formed exactly and rounded
    once, so the result repeats bit for bit.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        ``(3 * n_faces,)`` face index buffer.
    edges
        ``(n_faces, 3)`` mapping from each face half-edge to a unique edge index,
        as produced by ``igl::orient_halfedges``.
    edges_orientation
        ``(n_faces, 3)`` half-edge orientation relative to its unique edge
        (``igl::orient_halfedges``); half-edges with a negative value are skipped.
    edge_values
        ``(n_unique_edges,)`` scalar field defined on unique edges.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_vertices,)`` scalar field defined on vertices.

    Raises
    ------
    ValueError
        If ``edges`` or ``edges_orientation`` is not ``(n_faces, 3)``.
    RuntimeError
        If ``faces``, ``edges``, ``edges_orientation`` and ``edge_values`` are not all on one
        device.

    See Also
    --------
    [`average_onto_vertices`][ordito.interpolation.average_onto_vertices]
        The same scatter-sum-then-divide over faces instead of edges. The two disagree on a
        **zero-valence** vertex on purpose, each mirroring its own igl function: this one guards
        the division and yields ``0``, that one divides unguarded and yields ``nan``.

    Examples
    --------
    One triangle whose edge ``j`` is opposite corner ``j``: each corner averages the two edges it
    touches, and the unreferenced vertex 3 stays ``0``:

    ```python
    import warp as wp

    import ordito as od

    faces = wp.array([0, 1, 2], dtype=wp.int32)
    edges = wp.array([[0, 1, 2]], dtype=wp.int32)
    orientation = wp.array([[1, 1, 1]], dtype=wp.int32)
    edge_values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
    values = od.interpolation.average_from_edges_onto_vertices(
        4, faces, edges, orientation, edge_values
    )
    print(values.numpy())  # [2.5 2.  1.5 0. ]
    ```
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

    return _vertex_mean(
        n_vertices,
        n_faces,
        kernel_scatter.SCATTER_EDGES_SUM_AND_VALENCE,
        [faces, edges, edges_orientation, edge_values],
        kernel_array.divide_if_positive,
        kernel_interpolation.mean_rounded_if_counted,
        device,
    )


def _vertex_mean(
    n_vertices: int,
    n_faces: int,
    scatter: kernel_array.OverloadTable,
    inputs: list[object],
    divide: object,
    divide_rounded: object,
    device: wp.DeviceLike,
) -> wp.array[wp.float32]:
    """
    Scatter a per-vertex sum and valence over the faces, then divide: both averaging scatters.

    With [`is_deterministic`][ordito.config.is_deterministic] off the sum is ``float32`` in arrival
    order and the division runs in place over it; with it on the ``float32`` terms sum exactly in
    ``float64``, the same bits whatever order the atomics commit in, and are rounded once into the
    output.
    """
    valence = _launch.zeros(n_vertices, dtype=wp.float32, device=device)
    if not is_deterministic():
        out_sum = _launch.zeros(n_vertices, dtype=wp.float32, device=device)
        _launch.launch(
            scatter[wp.float32], dim=n_faces, inputs=[*inputs, out_sum, valence], device=device
        )
        _launch.map(divide, out_sum, valence, out=out_sum)
        return out_sum
    exact_sum = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        scatter[wp.float64], dim=n_faces, inputs=[*inputs, exact_sum, valence], device=device
    )
    out_mean = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    _launch.map(divide_rounded, exact_sum, valence, out=out_mean)
    return out_mean


def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[DType],
    target_vertices: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    mesh: wp.Mesh | None = None,
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
        ``(3 * n_source_faces,)`` source triangle index buffer.
    source_values
        ``(n_source,)`` field on the source vertices. ``wp.float32`` for a scalar, ``wp.vec2``
        for a UV pair, ``wp.vec3`` for a normal or a colour; see
        [`DType`][ordito.interpolation.DType] for why those three and nothing else.
    target_vertices
        ``(n_target,)`` positions to sample at — usually another mesh's vertices, but any point set
        will do.
    max_dist
        Maximum search radius per target vertex. Targets with no source face within it keep the
        zero-filled output and report ``inf`` in the returned distance, which is how a caller
        detects them. When ``None``, unbounded, so every target hits.
    mesh
        A ``wp.Mesh`` already built over ``source_vertices`` and ``source_faces``, to spare the BVH
        build this otherwise pays on every call. Not checked against them: a mesh over other
        geometry silently answers for that geometry.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is what to pass.

    Returns
    -------
    values : wp.array[DType]
        ``(n_target,)`` transferred field on ``target_vertices.device``. Zero-filled wherever
        the closest-point query missed.
    distance : wp.array[wp.float32]
        ``(n_target,)`` distance from each target vertex to the source surface — the transfer's
        own confidence measure, and ``inf`` for a miss.

    Raises
    ------
    ValueError
        If ``source_values`` does not have one entry per source vertex.
    RuntimeError
        If ``source_vertices``, ``source_faces``, ``source_values``, ``target_vertices`` and
        ``mesh`` are not all on one device.

    See Also
    --------
    [`average_onto_vertices`][ordito.interpolation.average_onto_vertices]
    [`ordito.proximity.closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
    [`ordito.triangles.barycentric_to_points`][ordito.triangles.barycentric_to_points]

    Examples
    --------
    A field linear on the source triangles transfers exactly onto a finer grid over the same square:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    source_vertices, source_faces = od.creation.grid(count=(3, 3))
    target_vertices, _ = od.creation.grid(count=(9, 9))
    x = wp.array(source_vertices.numpy()[:, 0], dtype=wp.float32)
    values, distance = od.interpolation.transfer_onto_vertices(
        source_vertices, source_faces, x, target_vertices
    )
    print(values.shape)  # (81,)
    print(np.allclose(values.numpy(), target_vertices.numpy()[:, 0]))  # True
    print(distance.numpy().max())  # 0.0
    ```
    """
    require_same_device(
        source_vertices=source_vertices,
        source_faces=source_faces,
        source_values=source_values,
        target_vertices=target_vertices,
        mesh=mesh,
    )
    device = target_vertices.device
    n_target = target_vertices.size
    n_source = source_vertices.size
    if source_values.size != n_source:
        raise ValueError(
            f"source_values must have one entry per source vertex, got "
            f"{source_values.size} for {n_source} vertices"
        )

    out_values = _launch.zeros(n_target, dtype=source_values.dtype, device=device)
    if n_target == 0 or source_faces.size == 0:
        return out_values, _launch.full(n_target, float("inf"), dtype=wp.float32, device=device)

    # The order is chosen before the mesh is built, so the read it may take waits on no build.
    order = search_order(
        target_vertices, sampled_on_cuda=kernel_interpolation.TRANSFER_SAMPLED_ORDER_ON_CUDA
    )
    if mesh is None:
        require_nonempty_mesh(source_faces, "transfer_onto_vertices")
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
        mesh = wp.Mesh(points=source_vertices, indices=source_faces)
    distance = _launch.empty(n_target, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_interpolation.TRANSFER_ONTO_VERTICES[source_values.dtype],
        dim=n_target,
        inputs=[
            source_vertices,
            source_faces,
            source_values,
            mesh.id,
            target_vertices,
            wp.float32(math.inf if max_dist is None else max_dist),
            order,
            out_values,
            distance,
        ],
        device=device,
    )
    return out_values, distance


def interpolate_barycentric(
    faces: wp.array[wp.int32],
    face_index: wp.array[wp.int32],
    barycentric: wp.array[wp.vec3],
    vertex_values: wp.array[DType],
) -> wp.array[DType]:
    """
    Per-vertex field values at points given by a face and barycentric coordinates.

    The attribute half of every query that lands on a surface:
    [`sample_surface`][ordito.sample.sample_surface] with ``return_barycentric=True``,
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] and the ray casts all report
    a face and barycentric coordinates per point, and this reads any per-vertex field there -- a
    colour, a UV, a normal, a scalar. Point ``i`` reads
    ``u * values[a] + v * values[b] + w * values[c]`` with ``(a, b, c)`` the corners of face
    ``face_index[i]`` and ``(u, v, w) = barycentric[i]``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    face_index
        ``(n_points,)`` face of each point. A negative entry marks a point on no face (a missed
        query) and reads zero.
    barycentric
        ``(n_points,)`` barycentric coordinates of each point in its face, in corner order.
    vertex_values
        ``(n_vertices,)`` field on the vertices: ``wp.float32``, ``wp.vec2``, ``wp.vec3`` or
        ``wp.vec4``; see [`DType`][ordito.interpolation.DType].

    Returns
    -------
    wp.array[DType]
        ``(n_points,)`` field values on ``vertex_values.device``, of ``vertex_values``' dtype.

    Raises
    ------
    ValueError
        If ``face_index`` and ``barycentric`` differ in length.
    RuntimeError
        If ``faces``, ``face_index``, ``barycentric`` and ``vertex_values`` are not all on one
        device.

    See Also
    --------
    [`ordito.triangles.points_to_barycentric`][ordito.triangles.points_to_barycentric]
        The coordinates of known points in known faces.
    [`transfer_onto_vertices`][ordito.interpolation.transfer_onto_vertices]
        The same reading at the closest surface point of arbitrary positions.

    Examples
    --------
    The colour of surface samples, read from per-vertex colours:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    colors = wp.array(np.abs(vertices.numpy()), dtype=wp.vec3)
    points, face_index, barycentric = od.sample.sample_surface(
        vertices, faces, 100, seed=0, return_barycentric=True
    )
    sampled = od.interpolation.interpolate_barycentric(faces, face_index, barycentric, colors)
    print(sampled.shape)  # (100,)
    positions = od.interpolation.interpolate_barycentric(faces, face_index, barycentric, vertices)
    print(np.allclose(positions.numpy(), points.numpy(), atol=1e-6))  # True
    ```
    """
    require_same_device(
        faces=faces, face_index=face_index, barycentric=barycentric, vertex_values=vertex_values
    )
    n_points = face_index.size
    if barycentric.size != n_points:
        raise ValueError(
            f"face_index and barycentric must have one entry per point, got {n_points} and "
            f"{barycentric.size}"
        )
    device = vertex_values.device
    kernel = kernel_interpolation.INTERPOLATE_BARYCENTRIC[vertex_values.dtype]
    out_values = _launch.empty(n_points, dtype=vertex_values.dtype, device=device)
    if n_points > 0:
        _launch.launch(
            kernel,
            dim=n_points,
            inputs=[faces, face_index, barycentric, vertex_values, out_values],
            device=device,
        )
    return out_values


def transfer_through_operator(
    values: wp.array[DType], operator: odt.BsrMatrix[wp.float32]
) -> wp.array[DType]:
    """
    Carry a per-vertex field through a topology edit, using the edit's own interpolation operator.

    The float half of [`Provenance.vertex_values`][ordito.interpolation.Provenance.vertex_values]:
    an edit that moves *and* creates vertices, such as
    [`ordito.remesh.subdivide_loop`][ordito.remesh.subdivide_loop], makes an output vertex an affine
    combination of several input ones, which no index map can express. Its provenance's
    ``vertex_operator`` can, and applying it to a field is the same product that produced the new
    positions: ``transfer_through_operator(vertices, P)`` reproduces the subdivided vertex buffer,
    which is what makes this exact rather than a resampling.

    Prefer this over
    [`transfer_onto_vertices`][ordito.interpolation.transfer_onto_vertices] whenever the operator
    is available: that one projects onto the source surface and is therefore lossy by construction,
    where this one applies the very weights the edit used.

    Parameters
    ----------
    values
        ``(n_source,)`` field on the input vertices. Any **float32-based** Warp dtype closed
        under scaling and addition: ``wp.float32`` for a scalar, ``wp.vec2`` for a UV, ``wp.vec3``
        for a normal or a colour, ``wp.vec4`` for an RGBA colour. Not ``wp.float64`` -- see Notes.
    operator
        ``(n_out, n_source)`` interpolation matrix, such as a
        [`Provenance`][ordito.interpolation.Provenance]'s ``vertex_operator``.

    Returns
    -------
    wp.array[DType]
        ``(n_out,)`` transferred field on ``values.device``, of the same dtype as ``values``.

    Raises
    ------
    ValueError
        If the operator's column count does not match ``values``.
    RuntimeError
        If ``values`` and ``operator`` are not on one device.

    Notes
    -----
    An output row the operator leaves empty gets the dtype's zero rather than uninitialized memory,
    so a partial operator is safe to apply.

    ``warp.sparse.bsr_mv`` would serve for a ``float32`` field and nothing else -- its vector dtype
    has to match the matrix's 1x1 block -- so this goes through a small CSR kernel instead, which is
    what makes the vector cases (a UV, a colour) reachable at all.

    The field's scalar must be ``float32`` because the operator's weights are: Warp requires both
    operands of a product to share a scalar type, so a ``wp.float64`` field does not compile against
    a float32 operator. Cast it with [`astype`][ordito.array.astype] if that is what you hold. The
    restriction is honest rather than incidental -- these operators are assembled from float32
    vertex data, so carrying a float64 field through one would advertise precision the weights do
    not have.

    See Also
    --------
    [`Provenance`][ordito.interpolation.Provenance]
        Which carries the operator of every topology edit in this package, and picks this
        transfer or a gather by the field's dtype.
    [`transfer_onto_vertices`][ordito.interpolation.transfer_onto_vertices]
        The lossy alternative, for when the two meshes are related only by geometry.

    Examples
    --------
    Carrying the vertex positions through Loop subdivision's operator reproduces the subdivided
    vertices:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=1)
    fine_vertices, fine_faces, provenance = od.remesh.subdivide_loop(
        vertices, faces, return_provenance=True
    )
    prolongation = provenance.vertex_operator
    assert prolongation is not None
    print(prolongation.shape)  # (162, 42)
    moved = od.interpolation.transfer_through_operator(vertices, prolongation)
    print(np.allclose(moved.numpy(), fine_vertices.numpy(), atol=1e-6))  # True
    ```
    """
    require_same_device(values=values, operator=operator)
    device = values.device
    n_source = values.size
    n_out = int(operator.nrow)
    if int(operator.ncol) != n_source:
        raise ValueError(f"operator has {operator.ncol} columns but values has {n_source} entries")

    if n_out == 0 or n_source == 0:
        return _launch.zeros(n_out, dtype=values.dtype, device=device)

    # Every row is written, an empty one with the additive identity, so no initial value.
    out_values = _launch.empty(n_out, dtype=values.dtype, device=device)
    _launch.launch(
        kernel_interpolation.APPLY_TRANSFER_OPERATOR[values.dtype],
        dim=n_out,
        inputs=[operator.offsets, operator.columns, operator.values, values, out_values],
        device=device,
    )
    return out_values


def transfer_through_index(values: wp.array[DType], index: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Carry a field through a topology edit's index map: ``values[index]``, zero where it is ``-1``.

    The copy half of [`Provenance`][ordito.interpolation.Provenance]: a face field follows an edit
    through its ``face_index``, and a field that cannot be averaged -- a label, a mask, a
    ``float64`` field -- through its ``vertex_index``. An output element with no source (an index
    of ``-1``, such as a face a hole fill created) reads the dtype's zero, where
    [`ordito.array.gather`][ordito.array.gather] would read out of bounds.

    Parameters
    ----------
    values
        ``(n,)`` field to carry: ``wp.bool``, ``wp.uint8``, ``wp.int32``, ``wp.int64``,
        ``wp.float32``, ``wp.float64``, or a ``wp.vec2`` / ``vec3`` / ``vec4`` / ``vec2d`` /
        ``vec3d`` / ``vec2i`` / ``vec3i`` or four-byte colour vector.
    index
        ``(m,)`` entry of ``values`` each output element copies, ``-1`` for none.

    Returns
    -------
    wp.array[DType]
        ``(m,)`` carried field on ``values.device``, of ``values``' dtype.

    Raises
    ------
    TypeError
        If ``values``' dtype is not one listed above.
    RuntimeError
        If ``values`` and ``index`` are not on one device.

    See Also
    --------
    [`transfer_through_operator`][ordito.interpolation.transfer_through_operator]
        The interpolating half, for a float field.
    [`operator_representatives`][ordito.interpolation.operator_representatives]
        The index map a label follows through an edit known only by its operator.

    Examples
    --------
    Face labels follow a submesh, through its face map:

    ```python
    import warp as wp

    import ordito as od

    labels = wp.array([10, 11, 12, 13], dtype=wp.int32)
    index = wp.array([3, -1, 0], dtype=wp.int32)
    print(od.interpolation.transfer_through_index(labels, index).numpy().tolist())  # [13, 0, 10]
    ```
    """
    require_same_device(values=values, index=index)
    if values.dtype not in kernel_interpolation.ATTRIBUTE_DTYPES:
        raise TypeError(f"cannot carry a field of dtype {values.dtype} through an index map")
    out_values = _launch.empty(index.size, dtype=values.dtype, device=values.device)
    if index.size > 0:
        _launch.launch(
            kernel_interpolation.GATHER_OR_ZERO[values.dtype],
            dim=index.size,
            inputs=[values, index, values.dtype(), out_values],
            device=values.device,
        )
    return cast("wp.array[DType]", out_values)


def operator_from_index(index: wp.array[wp.int32], n_columns: int) -> odt.BsrMatrix[wp.float32]:
    """
    Sparse operator of an index map: row ``i`` holds a single ``1`` at column ``index[i]``.

    The operator form of a copy -- a selection's, a repair's, a cut's ``vertex_index`` -- so it
    chains with an interpolating edit through
    [`compose_operators`][ordito.interpolation.compose_operators], and carrying a float field
    through it with
    [`transfer_through_operator`][ordito.interpolation.transfer_through_operator] is the gather. A
    ``-1`` entry is an empty row, which reads zero.

    Parameters
    ----------
    index
        ``(n_rows,)`` column of each row, ``-1`` for an empty row.
    n_columns
        Column count, above every entry of ``index``.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_rows, n_columns)`` ``float32`` matrix on ``index.device``.

    See Also
    --------
    [`operator_representatives`][ordito.interpolation.operator_representatives]
        The way back, from an operator to the index map of its largest weights.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    operator = od.interpolation.operator_from_index(wp.array([2, -1, 0], dtype=wp.int32), 3)
    print(operator.shape)  # (3, 3)
    values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
    moved = od.interpolation.transfer_through_operator(values, operator)
    print(moved.numpy().tolist())  # [3.0, 0.0, 1.0]
    ```
    """
    device = index.device
    n = index.size
    offsets = _launch.zeros(n + 1, dtype=wp.int32, device=device)
    # One entry a row at most, so the buffers are sized by the rows and nothing is read back.
    columns = _launch.empty(n, dtype=wp.int32, device=device)
    weights = _launch.empty(n, dtype=wp.float32, device=device)
    if n > 0:
        _launch.launch(
            kernel_interpolation.count_gather_rows, dim=n, inputs=[index, offsets], device=device
        )
        _launch.array_scan(offsets, out_array=offsets, inclusive=True)
        _launch.launch(
            kernel_interpolation.write_gather_rows,
            dim=n,
            inputs=[index, offsets],
            outputs=[columns, weights],
            device=device,
        )
    return od.array.bsr_from_csr(n, n_columns, offsets, columns, weights)


def operator_representatives(operator: odt.BsrMatrix[wp.float32]) -> wp.array[wp.int32]:
    """
    Each row's column of largest weight: where a field that cannot be averaged is copied from.

    An interpolating edit forms an output vertex from several inputs; an integer label cannot be
    averaged over them, so it is copied from the one that weighs most -- the lower-numbered on a
    tie, so a midpoint takes its lower endpoint's label and a cluster its lowest member's. This is
    how [`Provenance`][ordito.interpolation.Provenance]'s ``vertex_index`` is derived from its
    operator, so the two never disagree.

    Parameters
    ----------
    operator
        ``(n_rows, n_columns)`` ``float32`` interpolation matrix.

    Returns
    -------
    wp.array[wp.int32]
        ``(n_rows,)`` column of each row's largest weight, ``-1`` for an empty row, on the
        operator's device.

    See Also
    --------
    [`transfer_through_index`][ordito.interpolation.transfer_through_index]
        Which carries a label through the result.

    Examples
    --------
    Loop subdivision's operator, read for the vertex each output vertex takes its labels from:

    ```python
    import ordito as od

    vertices, faces = od.creation.icosahedron()
    _, _, provenance = od.remesh.subdivide_loop(vertices, faces, return_provenance=True)
    assert provenance.vertex_operator is not None
    sources = od.interpolation.operator_representatives(provenance.vertex_operator)
    print(sources.numpy()[:12].tolist() == list(range(12)))  # True
    ```
    """
    n_rows = int(operator.nrow)
    device = operator.offsets.device
    representative = _launch.empty(n_rows, dtype=wp.int32, device=device)
    if n_rows > 0:
        _launch.launch(
            kernel_interpolation.row_representatives,
            dim=n_rows,
            inputs=[operator.offsets, operator.columns, operator.values],
            outputs=[representative],
            device=device,
        )
    return representative


def compose_operators(
    first: odt.BsrMatrix[wp.float32], second: odt.BsrMatrix[wp.float32]
) -> odt.BsrMatrix[wp.float32]:
    """
    Operator of two topology edits applied in turn: ``second @ first``.

    ``first`` carries a field from a mesh to a second one and ``second`` from that to a third, so
    their product carries it from the first mesh to the third in one transfer, and equals the two
    transfers in turn up to ``float32`` rounding. Each row's entries are expanded through the other
    operator and duplicate columns summed, so a vertex that is a midpoint of midpoints weighs the
    corners of the face it lies in.

    Parameters
    ----------
    first
        ``(n_middle, n_columns)`` operator of the earlier edit.
    second
        ``(n_rows, n_middle)`` operator of the later edit.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_rows, n_columns)`` ``float32`` product on the operators' device.

    Raises
    ------
    ValueError
        If ``second``'s column count is not ``first``'s row count.
    RuntimeError
        If the two operators are not on one device.

    See Also
    --------
    [`Provenance.then`][ordito.interpolation.Provenance.then]
        The same chaining for a whole provenance.
    [`operator_from_index`][ordito.interpolation.operator_from_index]
        An index map as an operator, to chain a copy with an interpolation.

    Examples
    --------
    Two midpoint subdivisions of a box, as one operator from its 8 corners:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    once, once_faces, first = od.remesh.subdivide(vertices, faces, return_provenance=True)
    twice, _, second = od.remesh.subdivide(once, once_faces, return_provenance=True)
    assert first.vertex_operator is not None and second.vertex_operator is not None
    both = od.interpolation.compose_operators(first.vertex_operator, second.vertex_operator)
    print(both.shape)  # (98, 8)
    moved = od.interpolation.transfer_through_operator(vertices, both)
    print(np.allclose(moved.numpy(), twice.numpy()))  # True
    ```
    """
    require_same_device(first=first.offsets, second=second.offsets)
    if int(second.ncol) != int(first.nrow):
        raise ValueError(
            f"second has {second.ncol} columns but first has {first.nrow} rows; the operators "
            "do not chain"
        )
    device = first.offsets.device
    n_out = int(second.nrow)
    offsets = _launch.zeros(n_out + 1, dtype=wp.int32, device=device)
    if n_out > 0:
        _launch.launch(
            kernel_interpolation.count_composed_rows,
            dim=n_out,
            inputs=[first.offsets, second.offsets, second.columns, offsets],
            device=device,
        )
    # The product's triplet count sizes its buffers: one tail readback.
    n_triplets = scan_count(_launch.view(offsets, 1, n_out + 1))
    rows, columns, weights = od.array.triplet_buffers(n_triplets, wp.float32, device)
    if n_out > 0:
        _launch.launch(
            kernel_interpolation.write_composed_rows,
            dim=n_out,
            inputs=[
                first.offsets,
                first.columns,
                first.values,
                second.offsets,
                second.columns,
                second.values,
                offsets,
            ],
            outputs=[rows, columns, weights],
            device=device,
        )
    return od.array.csr_from_triplets(n_out, int(first.ncol), rows, columns, weights)


class Provenance(NamedTuple):
    """
    Where each element of a topology edit's output came from, and how a field follows it.

    Every function that rebuilds a mesh's connectivity -- a subdivision, a decimation, a
    selection, a cut, a repair -- takes ``return_provenance=True`` and then returns one of these
    after the new vertices and faces. It is how a per-vertex or per-face attribute (a colour, a UV,
    a label, a normal, a scalar) survives the edit:
    ``provenance.vertex_values(colors)`` and ``provenance.face_values(labels)`` are the attribute
    on the output mesh.

    An output vertex has a **representative** input vertex, ``vertex_index``, which is where a
    field that cannot be averaged -- an integer label, a mask -- is copied from. Where the edit
    creates vertices *between* input vertices (a midpoint, a crossing, a collapse) it also reports
    ``vertex_operator``, the sparse matrix of weights the new positions were formed with, and a
    float field is carried by the same weights, so carrying the positions themselves reproduces
    the output vertices. An output face always has a single source face, ``face_index``.

    Two edits chain with [`then`][ordito.interpolation.Provenance.then], so a pipeline of edits
    still carries its fields from the first input to the last output.

    It is a plain record of arrays, and every method is one of this module's functions on its
    fields, which work as well on arrays a caller holds without one:
    [`transfer_through_operator`][ordito.interpolation.transfer_through_operator] and
    [`transfer_through_index`][ordito.interpolation.transfer_through_index] carry fields,
    [`compose_operators`][ordito.interpolation.compose_operators] chains operators,
    [`operator_from_index`][ordito.interpolation.operator_from_index] turns a copy into an operator,
    and [`operator_representatives`][ordito.interpolation.operator_representatives] gives the
    ``vertex_index`` of an operator.

    Examples
    --------
    Colours carried through a subdivision: the original vertices keep theirs and every midpoint
    gets the mean of its edge's two:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    colors = wp.array(np.abs(vertices.numpy()) * 2.0, dtype=wp.vec3)
    new_vertices, new_faces, provenance = od.remesh.subdivide(
        vertices, faces, return_provenance=True
    )
    new_colors = provenance.vertex_values(colors)
    print(new_colors.shape)  # (26,)
    moved = provenance.vertex_values(vertices)
    print(np.allclose(moved.numpy(), new_vertices.numpy()))  # True
    face_ids = od.array.arange(12, device=faces.device)
    print(provenance.face_values(face_ids).numpy()[:8].tolist())  # [0, 0, 0, 0, 1, 1, 1, 1]
    ```
    """

    vertex_index: wp.array[wp.int32]
    """``(n_out_vertices,)`` representative input vertex of each output vertex, ``-1`` for one
    with no source: ``operator_representatives(vertex_operator)`` when there is an operator."""
    face_index: wp.array[wp.int32]
    """``(n_out_faces,)`` input face each output face came from, ``-1`` for one with no source
    (a patch that fills a hole)."""
    vertex_operator: odt.BsrMatrix[wp.float32] | None
    """``(n_out_vertices, n_vertices)`` interpolation weights, each row summing to one, or ``None``
    when every output vertex is a copy of its representative."""
    n_vertices: int
    """Input vertex count, the operator's column count."""
    n_faces: int
    """Input face count."""

    @classmethod
    def identity(cls, n_vertices: int, n_faces: int, device: wp.DeviceLike = None) -> Provenance:
        """
        Provenance of an edit that changed no vertex or face, every element its own source.

        Parameters
        ----------
        n_vertices
            Vertex count.
        n_faces
            Face count.
        device
            Device of the index maps.

        Returns
        -------
        Provenance
            The identity provenance.

        Examples
        --------
        ```python
        import ordito as od

        provenance = od.interpolation.Provenance.identity(4, 2)
        print(provenance.vertex_index.numpy().tolist())  # [0, 1, 2, 3]
        ```
        """
        return cls(
            od.array.arange(n_vertices, device=device),
            od.array.arange(n_faces, device=device),
            None,
            n_vertices,
            n_faces,
        )

    @classmethod
    def from_rows(
        cls,
        sources: odt.Array2dInt32,
        weights: odt.Array2dFloat32,
        face_index: wp.array[wp.int32],
        n_vertices: int,
        n_faces: int,
    ) -> Provenance:
        """
        Provenance whose output vertices are each a short weighted sum of input vertices.

        The constructor for an edit of one's own. Row ``r`` of ``sources`` and ``weights`` lists
        the input vertices output vertex ``r`` was formed from and their weights: one column copies
        a vertex, two interpolate along an edge, three inside a face. A source repeated in a row is
        merged, a ``-1`` source is padding and a zero weight is dropped. The representative is the
        source of largest weight, the smaller index on a tie, so a midpoint's labels come from its
        lower-numbered endpoint.

        Parameters
        ----------
        sources
            ``(n_out_vertices, k)`` input vertex indices, ``-1`` for an unused slot.
        weights
            ``(n_out_vertices, k)`` weight of each slot.
        face_index
            ``(n_out_faces,)`` input face of each output face, ``-1`` for none.
        n_vertices
            Input vertex count.
        n_faces
            Input face count.

        Returns
        -------
        Provenance
            The provenance, with ``vertex_operator`` holding the rows on ``sources.device``.

        Raises
        ------
        ValueError
            If ``sources`` and ``weights`` differ in shape.
        RuntimeError
            If ``sources``, ``weights`` and ``face_index`` are not all on one device.

        Examples
        --------
        A triangle's three vertices and its centroid:

        ```python
        import warp as wp

        import ordito as od

        sources = wp.array([[0, -1, -1], [1, -1, -1], [2, -1, -1], [0, 1, 2]], dtype=wp.int32)
        third = 1.0 / 3.0
        weights = wp.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [third, third, third]],
            dtype=wp.float32,
        )
        provenance = od.interpolation.Provenance.from_rows(
            sources, weights, wp.array([0, 0, 0], dtype=wp.int32), 3, 1
        )
        values = wp.array([0.0, 3.0, 6.0], dtype=wp.float32)
        print(provenance.vertex_values(values).numpy().tolist())  # [0.0, 3.0, 6.0, 3.0]
        ```
        """
        require_same_device(sources=sources, weights=weights, face_index=face_index)
        if tuple(sources.shape) != tuple(weights.shape):
            raise ValueError(
                f"sources and weights must have one shape, got {tuple(sources.shape)} and "
                f"{tuple(weights.shape)}"
            )
        device = sources.device
        n_out = int(sources.shape[0])
        offsets = _launch.zeros(n_out + 1, dtype=wp.int32, device=device)
        if n_out == 0:
            operator = od.array.bsr_from_csr(
                0,
                n_vertices,
                offsets,
                _launch.empty(0, dtype=wp.int32, device=device),
                _launch.empty(0, dtype=wp.float32, device=device),
                nnz=0,
            )
            return cls(
                operator_representatives(operator), face_index, operator, n_vertices, n_faces
            )
        _launch.launch(
            kernel_interpolation.count_provenance_rows,
            dim=n_out,
            inputs=[sources, weights, offsets],
            device=device,
        )
        # The entry count sizes the column and weight buffers: one tail readback.
        n_entries = scan_count(_launch.view(offsets, 1, n_out + 1))
        columns = _launch.empty(n_entries, dtype=wp.int32, device=device)
        values = _launch.empty(n_entries, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_interpolation.write_provenance_rows,
            dim=n_out,
            inputs=[sources, weights, offsets],
            outputs=[columns, values],
            device=device,
        )
        operator = od.array.bsr_from_csr(n_out, n_vertices, offsets, columns, values, nnz=n_entries)
        return cls(operator_representatives(operator), face_index, operator, n_vertices, n_faces)

    @classmethod
    def from_edge_points(
        cls,
        n_vertices: int,
        edges: odt.Array2dInt32,
        face_index: wp.array[wp.int32],
        n_faces: int,
        *,
        t: wp.array[wp.float32] | None = None,
    ) -> Provenance:
        """
        Provenance of an edit that keeps every input vertex and appends points on edges.

        The layout of a refinement: the output's first ``n_vertices`` vertices are the input's, in
        order, and output vertex ``n_vertices + e`` lies on the edge ``edges[e]`` at
        ``(1 - t[e]) * a + t[e] * b``.

        Parameters
        ----------
        n_vertices
            Input vertex count.
        edges
            ``(n_new, 2)`` endpoints ``(a, b)`` of the edge each appended vertex lies on.
        face_index
            ``(n_out_faces,)`` input face of each output face, ``-1`` for none.
        n_faces
            Input face count.
        t
            ``(n_new,)`` position of each appended vertex along its edge, from ``a`` at ``0`` to
            ``b`` at ``1``. ``None`` puts every one at its edge's midpoint.

        Returns
        -------
        Provenance
            The provenance, on ``edges.device``.

        Raises
        ------
        ValueError
            If ``t`` does not have one entry per edge.
        RuntimeError
            If ``edges``, ``face_index`` and ``t`` are not all on one device.

        Examples
        --------
        A segment's two endpoints and a point a quarter of the way along it:

        ```python
        import warp as wp

        import ordito as od

        edges = wp.array([[0, 1]], dtype=wp.int32)
        provenance = od.interpolation.Provenance.from_edge_points(
            2,
            edges,
            wp.empty(0, dtype=wp.int32),
            0,
            t=wp.array([0.25], dtype=wp.float32),
        )
        values = wp.array([0.0, 4.0], dtype=wp.float32)
        print(provenance.vertex_values(values).numpy().tolist())  # [0.0, 4.0, 1.0]
        ```
        """
        require_same_device(edges=edges, face_index=face_index, t=t)
        device = edges.device
        n_new = int(edges.shape[0])
        if t is not None and t.size != n_new:
            raise ValueError(f"t must have one entry per edge, got {t.size} for {n_new} edges")
        n_out = n_vertices + n_new
        sources = odt.empty_2d((n_out, 2), wp.int32, device=device)
        weights = odt.empty_2d((n_out, 2), wp.float32, device=device)
        if n_out > 0:
            _launch.launch(
                kernel_interpolation.edge_point_rows,
                dim=n_out,
                inputs=[
                    n_vertices,
                    edges,
                    _launch.empty(0, dtype=wp.float32, device=device) if t is None else t,
                ],
                outputs=[sources, weights],
                device=device,
            )
        return cls.from_rows(sources, weights, face_index, n_vertices, n_faces)

    @classmethod
    def from_clusters(
        cls,
        cluster: wp.array[wp.int32],
        n_out_vertices: int,
        face_index: wp.array[wp.int32],
        n_faces: int,
    ) -> Provenance:
        """
        Provenance whose output vertices each merge a cluster of input vertices.

        The constructor for a collapse or a weld: ``cluster`` maps every input vertex to the
        output vertex it became, many to one. A float field is averaged over each cluster, with
        equal weights, and the representative is the cluster's lowest-numbered member.

        Parameters
        ----------
        cluster
            ``(n_vertices,)`` output vertex of each input vertex, ``-1`` for one that survives in
            none.
        n_out_vertices
            Output vertex count. An output vertex no input maps to has no source.
        face_index
            ``(n_out_faces,)`` input face of each output face, ``-1`` for none.
        n_faces
            Input face count.

        Returns
        -------
        Provenance
            The provenance, on ``cluster.device``.

        Raises
        ------
        RuntimeError
            If ``cluster`` and ``face_index`` are not on one device.

        Examples
        --------
        Four input vertices welded into two:

        ```python
        import warp as wp

        import ordito as od

        cluster = wp.array([0, 1, 0, 1], dtype=wp.int32)
        provenance = od.interpolation.Provenance.from_clusters(
            cluster, 2, wp.empty(0, dtype=wp.int32), 0
        )
        print(provenance.vertex_index.numpy().tolist())  # [0, 1]
        values = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
        print(provenance.vertex_values(values).numpy().tolist())  # [2.0, 3.0]
        ```
        """
        require_same_device(cluster=cluster, face_index=face_index)
        device = cluster.device
        n_in = cluster.size
        offsets = _launch.zeros(n_out_vertices + 1, dtype=wp.int32, device=device)
        weights = _launch.empty(n_in, dtype=wp.float32, device=device)
        if n_in == 0:
            columns = _launch.empty(0, dtype=wp.int32, device=device)
        else:
            _launch.launch(
                kernel_interpolation.count_cluster_members,
                dim=n_in,
                inputs=[cluster, offsets],
                device=device,
            )
            _launch.array_scan(offsets, out_array=offsets, inclusive=True)
            # A stable sort by cluster lists each cluster's members in ascending order, which is
            # the operator's columns as they stand; the unclustered inputs sort last and are past
            # the last row's end.
            keys, order = _launch.sort_buffers(n_in, wp.int32, device)
            _launch.launch(
                kernel_interpolation.cluster_keys_and_identity,
                dim=n_in,
                inputs=[cluster, wp.int32(n_out_vertices), keys, order],
                device=device,
            )
            _launch.radix_sort_pairs(
                keys, order, n_in, end_bit=_launch.key_bits(n_out_vertices + 1)
            )
            columns = _launch.view(order, 0, n_in)
            _launch.launch(
                kernel_interpolation.write_cluster_rows,
                dim=n_in,
                inputs=[keys, offsets, wp.int32(n_out_vertices)],
                outputs=[weights],
                device=device,
            )
        operator = od.array.bsr_from_csr(n_out_vertices, n_in, offsets, columns, weights)
        # Equal weights in ascending columns: the representative is each cluster's lowest member.
        return cls(operator_representatives(operator), face_index, operator, n_in, n_faces)

    def vertex_values(self, values: wp.array[DType]) -> wp.array[DType]:
        """
        Carry a per-vertex field of the edit's input onto its output vertices.

        A float field (``wp.float32``, ``wp.vec2``, ``wp.vec3``, ``wp.vec4``) is interpolated
        through ``vertex_operator`` when the edit has one, so a midpoint gets the mean of its
        edge's values and carrying the input positions reproduces the output's. Any other field --
        an integer label, a mask, a ``float64`` field -- and every field of an edit without an
        operator is copied from each output vertex's representative. An output vertex with no
        source reads zero.

        Parameters
        ----------
        values
            ``(n_vertices,)`` field on the input vertices.

        Returns
        -------
        wp.array[DType]
            ``(n_out_vertices,)`` field on the output vertices, same dtype and device.

        Raises
        ------
        ValueError
            If ``values`` does not have one entry per input vertex.
        TypeError
            If ``values``' dtype is not one a provenance carries.
        RuntimeError
            If ``values`` is not on the provenance's device.

        Examples
        --------
        A label is copied and a scalar averaged across a subdivision:

        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.box()
        _, _, provenance = od.remesh.subdivide(vertices, faces, return_provenance=True)
        labels = od.array.arange(8, device=faces.device)
        print(provenance.vertex_values(labels).numpy()[:8].tolist())  # [0, 1, 2, 3, 4, 5, 6, 7]
        heights = wp.array(vertices.numpy()[:, 2], dtype=wp.float32)
        print(provenance.vertex_values(heights).shape)  # (26,)
        ```
        """
        require_same_device(values=values, vertex_index=self.vertex_index)
        if values.size != self.n_vertices:
            raise ValueError(
                f"values must have one entry per input vertex, got {values.size} for "
                f"{self.n_vertices} vertices"
            )
        if self.vertex_operator is not None and values.dtype in kernel_interpolation.FIELD_DTYPES:
            return transfer_through_operator(values, self.vertex_operator)
        return transfer_through_index(values, self.vertex_index)

    def face_values(self, values: wp.array[DType]) -> wp.array[DType]:
        """
        Copy a per-face field of the edit's input onto each output face from its source.

        Parameters
        ----------
        values
            ``(n_faces,)`` field on the input faces, of any dtype
            [`vertex_values`][ordito.interpolation.Provenance.vertex_values] carries.

        Returns
        -------
        wp.array[DType]
            ``(n_out_faces,)`` field on the output faces, zero on a face with no source.

        Raises
        ------
        ValueError
            If ``values`` does not have one entry per input face.
        TypeError
            If ``values``' dtype is not one a provenance carries.
        RuntimeError
            If ``values`` is not on the provenance's device.

        Examples
        --------
        Face labels follow a submesh:

        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.box()
        keep = wp.array([2, 5, 7], dtype=wp.int32)
        _, _, provenance = od.selection.submesh_from_face_indices(
            vertices, faces, keep, return_provenance=True
        )
        labels = wp.array([10 * f for f in range(12)], dtype=wp.int32)
        print(provenance.face_values(labels).numpy().tolist())  # [20, 50, 70]
        ```
        """
        require_same_device(values=values, face_index=self.face_index)
        if values.size != self.n_faces:
            raise ValueError(
                f"values must have one entry per input face, got {values.size} for "
                f"{self.n_faces} faces"
            )
        return transfer_through_index(values, self.face_index)

    def then(self, after: Provenance) -> Provenance:
        """
        Compose this edit with ``after``, into one provenance from this input to that output.

        Fields carried through the result equal fields carried through both edits in turn, so a
        pipeline (a repair, then a decimation, then a cut) keeps one provenance from first input
        to last output. The operators multiply; with no operator on either side the index maps
        compose, and otherwise the representatives are the product's.

        Parameters
        ----------
        after
            Provenance of an edit applied to this edit's output.

        Returns
        -------
        Provenance
            The composed provenance.

        Raises
        ------
        ValueError
            If ``after``'s input is not this edit's output.
        RuntimeError
            If the two are not on one device.

        Examples
        --------
        A subdivision twice is one provenance from the box to the twice-refined mesh:

        ```python
        import numpy as np

        import ordito as od

        vertices, faces = od.creation.box()
        once, once_faces, first = od.remesh.subdivide(vertices, faces, return_provenance=True)
        twice, _, second = od.remesh.subdivide(once, once_faces, return_provenance=True)
        both = first.then(second)
        print(both.vertex_operator.shape)  # (98, 8)
        print(np.allclose(both.vertex_values(vertices).numpy(), twice.numpy()))  # True
        ```
        """
        require_same_device(vertex_index=self.vertex_index, after=after.vertex_index)
        if after.n_vertices != self.vertex_index.size or after.n_faces != self.face_index.size:
            raise ValueError(
                f"after's input ({after.n_vertices} vertices, {after.n_faces} faces) is not this "
                f"edit's output ({self.vertex_index.size} vertices, {self.face_index.size} faces)"
            )
        face_index = od.array.remap_indices(after.face_index, self.face_index)
        if self.vertex_operator is None and after.vertex_operator is None:
            vertex_index = od.array.remap_indices(after.vertex_index, self.vertex_index)
            return Provenance(vertex_index, face_index, None, self.n_vertices, self.n_faces)
        first = self.vertex_operator
        if first is None:
            first = operator_from_index(self.vertex_index, self.n_vertices)
        second = after.vertex_operator
        if second is None:
            second = operator_from_index(after.vertex_index, after.n_vertices)
        operator = compose_operators(first, second)
        representative = operator_representatives(operator)
        return Provenance(representative, face_index, operator, self.n_vertices, self.n_faces)


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
    onto a mesh's vertices, onto [`grid_points`][ordito.voxels.grid_points], or onto any other
    sample set; use
    [`transfer_onto_vertices`][ordito.interpolation.transfer_onto_vertices] instead when the source
    *is* a mesh, since projecting onto its surface is exact for a piecewise-linear field where this
    is a smoothing, and
    [`sample_grid_trilinear`][ordito.voxels.sample_grid_trilinear] when the source is a dense
    lattice, where trilinear weights are exact rather than a kernel estimate.

    Parameters
    ----------
    source_points
        ``(n_source,)`` positions the field is known at.
    source_values
        ``(n_source,)`` field on those points. ``wp.float32`` for a scalar, ``wp.vec2`` for a
        UV pair, ``wp.vec3`` for a vector; see [`DType`][ordito.interpolation.DType] for why those
        three and nothing else.
    query_points
        ``(n_query,)`` positions to interpolate at.
    radius
        Length scale of the kernel, and — unless ``k`` is given — the footprint: only sources within
        it contribute. **Required**, and absolute: it is in the coordinates' own units, so a cloud
        rescaled by 100 needs a radius rescaled by 100. Derive it from the data rather than guessing
        (e.g. [`mean_edge_length`][ordito.edges.mean_edge_length] or
        [`knn_initial_radius`][ordito.neighbors.knn_initial_radius]).
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
        ``(n_query,)`` interpolated field on ``query_points.device``, with ``source_values``'
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
    ``counts == 0`` from [`query_ball_count`][ordito.neighbors.query_ball_count] and the
    closest-point fallback is this function at ``k=1``, so neither needs a mode of its own.

    See Also
    --------
    [`transfer_onto_vertices`][ordito.interpolation.transfer_onto_vertices]
    [`query_ball_with_offsets`][ordito.neighbors.query_ball_with_offsets]
    [`query_nearest`][ordito.neighbors.query_nearest]

    Examples
    --------
    A query on a source takes its value, one in between blends the two, one out of reach gets
    ``null_value``; with ``k=1`` every query takes its nearest source's value:

    ```python
    import warp as wp

    import ordito as od

    sources = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    values = wp.array([0.0, 10.0], dtype=wp.float32)
    queries = wp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=wp.vec3)
    blended = od.interpolation.interpolate_from_points(
        sources, values, queries, radius=1.0, null_value=-1.0
    )
    print(blended.numpy().round(2))  # [ 0.   3.1 -1. ]
    nearest = od.interpolation.interpolate_from_points(sources, values, queries, radius=1.0, k=1)
    print(nearest.numpy().round(2))  # [ 0.  0. 10.]
    ```
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

    out_values = _launch.full(n_query, null_value, dtype=source_values.dtype, device=device)
    if n_query == 0 or n_source == 0:
        return out_values

    inverse_scale = wp.float32(float(sharpness) / float(radius))
    if k is None:
        # The mean is reduced during the grid walk, so no neighbour list is ever materialized: the
        # predicate ``query_ball_with_offsets`` applies. Few queries on CUDA walk a block a query
        # over cells of the radius; otherwise one thread a query over cells sized for the count.
        few = (
            not wp.get_device(device).is_cpu
            and n_query < kernel_interpolation.INTERPOLATE_TILED_MAX_QUERIES
        )
        cell = (
            float(radius)
            if few
            else ball_grid_cell_width(
                float(radius),
                n_query,
                device,
                kernel_neighbors.BALL_GRID_WIDE_CELL_QUERIES_CUDA,
                kernel_neighbors.BALL_GRID_WIDE_CELL_QUERIES_CPU,
            )
        )
        grid = od.neighbors.hashgrid_from_points(source_points, cell)
        walk = tiled_ball_walk(
            grid, n_query, device, kernel_interpolation.INTERPOLATE_TILED_MAX_QUERIES
        )
        if walk is not None:
            _launch.launch_tiled(
                kernel_interpolation.INTERPOLATE_FROM_POINTS_IN_BALL_TILED[source_values.dtype],
                dim=[n_query],
                inputs=[
                    source_points,
                    source_values,
                    query_points,
                    grid.id,
                    wp.float32(radius),
                    inverse_scale,
                    *walk,
                ],
                outputs=[out_values],
                device=device,
                block_dim=kernel_neighbors.TILED_BALL_WALK_BLOCK_DIM,
            )
            return out_values
        _launch.launch(
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
    row_indices, row_distances = od.neighbors.query_nearest(source_points, query_points, int(k))
    n_slots = n_query * int(k)
    _launch.launch(
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
