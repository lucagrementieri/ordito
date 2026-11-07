"""
Straightest geodesics: walk a fixed distance across a mesh in a fixed direction.

A *straightest* geodesic is what you get by walking forward and, at every edge, unfolding the two
incident triangles into a common plane and continuing in a straight line. It is the surface analogue
of "go that way for this far", which makes it the tool for exponential maps, streamline tracing
and
extending a direction field along a surface — and unlike a shortest path it is fixed by an initial
condition rather than by two endpoints.

Both entry points are batched over many rays: one thread walks one ray, and the traced polylines
come back packed into one buffer with CSR offsets, the same shape
[`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] uses.

[`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex] is the **exponential map** of the
surface, and its inverse is [`log_map`][ordito.heat.log_map]: one takes a tangent direction
and a distance to a point, the other takes a point back to the direction and distance that reach it.
The two live apart because they are different methods -- this one unfolds triangles combinatorially,
that one solves a vector-heat system -- and geodesic *distance* by the heat method is a third,
[`heat_geodesic`][ordito.heat.heat_geodesic].

!!! note "Cone points"
    A path that runs exactly into a vertex has no unique straightest continuation — the angle around
    a vertex is not ``2 * pi``, so "straight through" is ambiguous. Such a crossing is resolved by
    unfolding across the edge the walk reached the vertex along: the correct limit for a path
    passing arbitrarily close to the vertex, but not the split-the-angle convention. Paths through
    high-curvature vertices therefore drift from geometry-central's by about the angle defect.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import math
import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

import numpy as np
import warp as wp

# ``@wp.struct`` rebinds the decorated name to a ``Struct`` value, so a helper passing one around
# names the instance's base instead (as ``holes`` does).
from warp._src.codegen import StructInstance

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._common import read_scalar, read_values, require_same_device, run_device_loop
from ordito.halfedge import halfedge_twins, vertex_one_rings
from ordito.kernels import array as kernel_array
from ordito.kernels import geodesic_walk as kernel_geodesic_walk
from ordito.kernels.algorithms import flip_geodesics as kernel_flip_geodesics
from ordito.mesh import Trimesh, mesh_arguments

_DEFAULT_MAX_STEPS = 4096

GeodesicPathMethod = Literal["heat", "potpourri3d"]
"""The ways [`geodesic_path`][ordito.geodesic_walk.geodesic_path] finds a path."""


def trace_from_vertex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    start_vertices: wp.array[wp.int32],
    directions: wp.array[wp.vec3],
    twins: wp.array[wp.int32] | None = None,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
    frames: tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Trace a straightest geodesic from each of a batch of vertices.

    Each ray starts at ``vertices[start_vertices[r]]`` and walks in ``directions[r]``, for an arc
    length equal to that direction's length *after projection into the vertex's tangent plane* — a
    unit direction traces at most a unit distance, and a direction along the normal traces nothing.
    That is ``potpourri3d.GeodesicTracer``'s convention too.

    The direction is read as a polar angle in the vertex's tangent space, the fan with its corner
    angles rescaled to a full turn (half a turn at a boundary vertex) -- the normalized angle of
    Polthier and Schmies. Its zero is placed by fitting the whole fan: each incident edge, projected
    into the tangent plane, is rotated back by its normalized angle, and the sum of those vectors is
    the direction of angle zero. So the start depends neither on how the mesh is numbered nor on
    the tangent frame, and it is defined at cone and saddle vertices alike; at a vertex whose fan is
    flat, a direction along an edge leaves along it. This is ``potpourri3d.GeodesicTracer``'s
    convention. At a boundary vertex a direction outside the fan points off the surface and traces
    nothing.

    A walk that runs into a vertex leaves it as a straightest geodesic, with half the vertex's total
    angle on either side. It stops early when it reaches the mesh boundary (or a boundary vertex) or
    exceeds ``max_steps`` edge crossings.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    start_vertices
        ``(n_rays,)`` start vertex per ray.
    directions
        ``(n_rays,)`` initial directions; the length of each sets how far its ray is traced.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins], or
        ``None`` to compute them from ``faces``.
    rings
        ``(3 * n_faces,)``, ``(n_vertices + 1,)`` and ``(n_vertices,)`` precomputed
        [`vertex_one_rings`][ordito.halfedge.vertex_one_rings], or ``None``, used to find each
        ray's starting face.
    frames
        ``(n_vertices,)`` triple of precomputed
        [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames], or ``None``. Their
        normal defines each vertex's tangent plane, which sets the trace length and the start; the
        in-plane axes do not matter.
    max_steps
        Maximum edge crossings per ray.

    Returns
    -------
    points : wp.array[wp.vec3]
        ``(n_points,)`` traced points, packed ray after ray, on ``vertices.device``;
        ``n_points == offsets[-1]``.
    offsets : wp.array[wp.int32]
        ``(n_rays + 1,)`` offsets; ray ``r`` owns ``points[offsets[r] : offsets[r + 1]]``,
        beginning at its start vertex. A ray always contributes at least one point.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``start_vertices``, ``directions``, ``twins``, ``rings`` and
        ``frames`` are not all on one device.

    See Also
    --------
    [`trace_from_face`][ordito.geodesic_walk.trace_from_face]
    [`split`][ordito.array.split]
    [`heat_geodesic`][ordito.heat.heat_geodesic]

    Examples
    --------
    On a flat grid a straightest geodesic is a straight line, as long as its direction:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    center = wp.array([60], dtype=wp.int32)  # the vertex at the origin
    directions = wp.array([wp.vec3(0.3, 0.2, 0.0)], dtype=wp.vec3)
    points, offsets = od.geodesic_walk.trace_from_vertex(vertices, faces, center, directions)
    print(offsets.numpy().tolist())  # [0, 6]
    print(points.numpy()[-1].round(3))  # [0.3 0.2 0. ]
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        start_vertices=start_vertices,
        directions=directions,
        twins=twins,
        rings=rings,
        frames=frames,
    )
    device = vertices.device
    n_rays = start_vertices.size
    n_vertices = vertices.size
    if n_rays == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.zeros(
            max(n_rays + 1, 1), dtype=wp.int32, device=device
        )
    # No short-circuit on ``faces.shape[0] == 0``: unlike ``trace_from_face``, a start *vertex* is
    # still a valid reference into ``vertices`` with no faces at all (every vertex is isolated), and
    # the general path below already produces the documented one-point-per-ray answer for that case
    # -- ``vertex_one_rings`` gives every vertex an empty ring, and the kernel's
    # ``start_direction_at_vertex`` then reports "isolated vertex, nowhere to go" (``f == -1``),
    # which ``trace_from_vertices`` turns into exactly one written point, the start vertex itself.
    # A short-circuit here returned an empty polyline per ray instead, contradicting this function's
    # own "a ray always contributes at least one point" guarantee.

    if rings is None:
        # The rings and the twins they are walked over come back together, for one readback.
        ring_halfedges, ring_offsets, is_boundary, twins = vertex_one_rings(
            faces, twins=twins, n_vertices=n_vertices, return_twins=True
        )
        rings = (ring_halfedges, ring_offsets, is_boundary)
    elif twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    ring_halfedges, ring_offsets, is_boundary = rings
    if frames is None:
        frames = od.tangent_space.vertex_tangent_frames(vertices, faces, rings=rings)
    basis_x, basis_y, normals = frames

    inputs = [
        vertices,
        faces,
        twins,
        ring_offsets,
        ring_halfedges,
        is_boundary,
        basis_x,
        basis_y,
        normals,
        start_vertices,
        directions,
        wp.int32(max_steps),
        *_length_epsilon(vertices, faces),
    ]
    return _trace(
        kernel_geodesic_walk.trace_from_vertices, inputs, n_rays, max_steps, faces.size // 3, device
    )


def trace_from_face(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    start_faces: wp.array[wp.int32],
    start_barycentric: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    twins: wp.array[wp.int32] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Trace a straightest geodesic from each of a batch of barycentric points inside faces.

    As [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex], but each ray
    starts at an interior point of a known face, so no wedge search is needed. This is the form for
    streamlines of a face-based vector field.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    start_faces
        ``(n_rays,)`` starting face per ray.
    start_barycentric
        ``(n_rays,)`` barycentric coordinates of each start point within its face.
    directions
        ``(n_rays,)`` initial directions; the length of each sets how far its ray is traced.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins], or
        ``None`` to compute them from ``faces``.
    max_steps
        Maximum edge crossings per ray.

    Returns
    -------
    points : wp.array[wp.vec3]
        ``(n_points,)`` traced points, packed ray after ray, on ``vertices.device``;
        ``n_points == offsets[-1]``.
    offsets : wp.array[wp.int32]
        ``(n_rays + 1,)`` CSR bounds into ``points``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``start_faces``, ``start_barycentric``, ``directions`` and
        ``twins`` are not all on one device.

    See Also
    --------
    [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex]
    [`barycentric_to_points`][ordito.triangles.barycentric_to_points]

    Examples
    --------
    A ray from the centroid of face 0 of a flat grid, walking 0.5 along X:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    start_faces = wp.array([0], dtype=wp.int32)
    centroid = wp.array([wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)], dtype=wp.vec3)
    directions = wp.array([wp.vec3(0.5, 0.0, 0.0)], dtype=wp.vec3)
    points, offsets = od.geodesic_walk.trace_from_face(
        vertices, faces, start_faces, centroid, directions
    )
    print(f"{od.polyline.polyline_length(points):.4f}")  # 0.5000
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        start_faces=start_faces,
        start_barycentric=start_barycentric,
        directions=directions,
        twins=twins,
    )
    device = vertices.device
    n_rays = start_faces.size
    if n_rays == 0 or faces.size == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.zeros(
            max(n_rays + 1, 1), dtype=wp.int32, device=device
        )

    if twins is None:
        twins = halfedge_twins(faces, n_vertices=vertices.size)

    inputs = [
        vertices,
        faces,
        twins,
        start_faces,
        start_barycentric,
        directions,
        wp.int32(max_steps),
        *_length_epsilon(vertices, faces),
    ]
    return _trace(
        kernel_geodesic_walk.trace_from_faces, inputs, n_rays, max_steps, faces.size // 3, device
    )


def descend_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    starts: wp.array[wp.int32],
    *,
    stop_value: float = 0.0,
    twins: wp.array[wp.int32] | None = None,
    vertex_faces: tuple[wp.array[wp.int32], wp.array[wp.int32]] | None = None,
    gradients: wp.array[wp.vec3d] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Follow a per-vertex scalar field downhill from each of a batch of start vertices.

    The **field-descent** counterpart of this module's two straightest-geodesic tracers: where those
    are fixed by an initial direction, this one is fixed by a *field*, and it goes wherever that
    field decreases fastest. Fed a geodesic distance field it traces the geodesic back to its
    source, which is what [`geodesic_path`][ordito.geodesic_walk.geodesic_path] is; fed any other
    scalar it traces that scalar's flow lines.

    Three cases, and the field value at each written point strictly decreases in all of them --
    which is what makes the walk terminate rather than orbit:

    1. **Inside a face** the piecewise-linear interpolant's gradient is constant, so the path is a
       straight segment to the exit edge.
    2. **Along an edge**, when the face across it has a descent that points back: the walk slides to
       the edge's lower-valued endpoint.
    3. **At a vertex**, where the field has no single gradient: each incident face is asked whether
       its own descent direction points into the fan wedge, and the steepest admissible one wins.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    values
        ``(n_vertices,)`` field to descend. ``float64`` because the fields this
        serves are exponentially decaying -- see
        [`ordito.laplacian.face_gradients`][ordito.laplacian.face_gradients], which computes the
        per-face gradient this walks along.
    starts
        ``(n_paths,)`` vertices to descend from, one path each.
    stop_value
        Field value at which a path stops. The default ``0.0`` is what a distance field's source
        sits at.
    twins
        ``(3 * n_faces,)`` precomputed
        [`ordito.halfedge.halfedge_twins`][ordito.halfedge.halfedge_twins], or ``None`` to compute
        them from ``faces``.
    vertex_faces
        ``(3 * n_faces,)`` and ``(n_vertices + 1,)`` precomputed
        [`ordito.adjacency.vertex_face_adjacency`][ordito.adjacency.vertex_face_adjacency] pair,
        or ``None``, which case 3 needs. ``(vertex_faces, offsets)``, values first, as that function
        returns it and as every packed pair in the package is spelled -- passing it the other way
        round reads offsets as face indices and raises nothing, since both are ``wp.int32``.
    gradients
        ``(n_faces,)`` precomputed per-face gradient of ``values``, or ``None``. Pass it when
        descending the same field from several batches.
    max_steps
        Cap on steps per path. A path that hits it is returned truncated rather than reported.

    Returns
    -------
    points, offsets
        ``(n_points,)`` and ``(n_paths + 1,)``: ``points`` holds every path's polyline end to end
        and ``offsets`` is the CSR bound, the same packing
        [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex] returns and
        [`split`][ordito.array.split] slices.

    Raises
    ------
    ValueError
        If ``values`` does not have one entry per vertex, or ``gradients`` is given and does not
        have one entry per face.
    RuntimeError
        If ``vertices``, ``faces``, ``values``, ``starts``, ``twins``, ``vertex_faces`` and
        ``gradients`` are not all on one device.

    Notes
    -----
    A path can stop before reaching ``stop_value``, and the caller can tell: its last point is not
    within tolerance of a vertex whose value is at the stop. That happens at a **local minimum** of
    the field, on a flat face where the gradient vanishes, at the mesh **boundary**, and when
    ``max_steps`` runs out. None of those is an error -- a field with several minima has several
    basins, and this walks the one it starts in.

    See Also
    --------
    [`geodesic_path`][ordito.geodesic_walk.geodesic_path]
        The distance-field case, which is what this is usually reached for.
    [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex]
        The direction-driven walk, for a *straightest* geodesic rather than a shortest one.
    [`ordito.laplacian.face_gradients`][ordito.laplacian.face_gradients]

    Examples
    --------
    Descending the distance to the grid's corner vertex 0 runs straight back to it, from the far
    corner along the diagonal and from vertex 10 along the edge:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    positions = vertices.numpy()
    values = wp.array(np.linalg.norm(positions - positions[0], axis=1), dtype=wp.float64)
    starts = wp.array([120, 10], dtype=wp.int32)
    points, offsets = od.geodesic_walk.descend_field(vertices, faces, values, starts)
    paths = od.array.split(points, offsets)
    print([f"{od.polyline.polyline_length(path):.4f}" for path in paths])  # ['1.4142', '1.0000']
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        values=values,
        starts=starts,
        twins=twins,
        vertex_faces=vertex_faces,
        gradients=gradients,
    )
    n_vertices = vertices.size
    if values.size != n_vertices:
        raise ValueError(
            f"values must have one entry per vertex, got {values.size} for {n_vertices}"
        )
    n_faces = faces.size // 3
    if gradients is not None and gradients.size != n_faces:
        # Unlike ``values``, a mismatched ``gradients`` was an out-of-bounds read with no
        # exception -- ``descent_walk``/``descend_at_vertex`` index it as ``gradients[f]`` for
        # every face index up to ``n_faces - 1`` (silent host-heap corruption on CPU, garbage or a
        # crash on CUDA), the exact hazard a caller-supplied precomputed cache exists to avoid.
        raise ValueError(
            f"gradients must have one entry per face, got {gradients.size} for {n_faces}"
        )
    points, offsets, _ = _descend(
        vertices, faces, values, starts, stop_value, twins, vertex_faces, gradients, max_steps
    )
    return points, offsets


@overload
def geodesic_path(
    mesh: Trimesh,
    source: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    /,
    *,
    method: GeodesicPathMethod = "heat",
    t: float | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def geodesic_path(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    source: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    /,
    *,
    method: GeodesicPathMethod = "heat",
    t: float | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
def geodesic_path(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    source: wp.array[wp.int32] | None = None,
    targets: wp.array[wp.int32] | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    method: GeodesicPathMethod = "heat",
    t: float | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Trace a path across the surface from each target vertex back to a source set.

    The **point-to-point** geodesic, as against this module's straightest walks and
    [`heat_geodesic`][ordito.heat.heat_geodesic]'s distance *field*. ``method`` picks how:

    - ``"heat"`` (the default): one heat solve gives the distance to the source everywhere, and
      descending it from a target follows that geodesic back. Every target shares the one solve,
      so a thousand paths to one source cost one system and a thousand independent walks -- which
      is why the signature is one source and many targets rather than a list of pairs. The path is
      approximate (see Notes).
    - ``"potpourri3d"``: ``potpourri3d``'s ``EdgeFlipGeodesicSolver.find_geodesic_path`` (Sharp
      and Crane's edge-flip geodesics, from geometry-central). Each path starts as the shortest
      mesh-edge path from the nearest source and is straightened by flipping the edges of an
      intrinsic triangulation until it is a *locally shortest* path: exact for the polyhedral
      surface, within the homotopy class of the edge path it started from. The answer is the
      reference's point for point.

    Takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh], which keeps the heat
    method's operators and factorizations for the next call (several sources traced on one mesh),
    or as its ``vertices`` and ``faces``.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    source
        ``(k,)`` source vertices. Several make the paths run to whichever is nearest,
        since the field is the distance to the *set*.
    targets
        ``(n_paths,)`` vertices to trace from.
    method
        ``"heat"`` or ``"potpourri3d"``, the two ways above.
    t
        Heat diffusion time, forwarded to
        [`heat_geodesic`][ordito.heat.heat_geodesic]. ``None`` uses its default. Read by
        ``"heat"`` only.
    max_steps
        Cap on steps per path. Read by ``"heat"`` only.

    Returns
    -------
    points, offsets
        ``(n_points,)`` and ``(n_paths + 1,)`` packed polylines and their CSR bounds, each running
        **from its target to the source**. Slice with [`split`][ordito.array.split] and measure
        with [`ordito.polyline.polyline_length`][ordito.polyline.polyline_length].

    Raises
    ------
    ValueError
        If ``method`` is not one of the two; with ``"potpourri3d"``, if an entry of ``source`` or
        ``targets`` is not a vertex index in ``[0, n_vertices)`` or a vertex's faces do not form a
        single fan. ``"potpourri3d"`` also takes
        [`halfedge_twins`][ordito.halfedge.halfedge_twins]' refusals: an edge of three or more
        faces, or two faces wound against each other.
    RuntimeError
        If ``vertices``, ``faces``, ``source`` and ``targets`` are not all on one device.

    Warns
    -----
    UserWarning
        With ``"heat"``, when paths stopped short of the source on the heat distance and were
        finished along mesh edges (see Notes); the message counts the paths so finished. With
        ``"potpourri3d"``, when paths stopped shortening at the step bound (a numerically stuck
        flip sequence), counting them.

    Notes
    -----
    **The path is as accurate as the field it descends, and no more.** The heat method's distance is
    first-order, so this is an *approximate* geodesic: measured against ``potpourri3d``'s edge-flip
    geodesics -- which are exact -- the length comes out a few per cent long, and the excess is the
    field's error rather than the walk's. It is never *shorter* than the true geodesic, which is the
    invariant worth testing against.

    **Every path reaches the source.** A path ends when it reaches a source vertex, whatever the
    field reads there: with several sources the heat distance is zero on their *mean*, not on each.
    The heat distance can have a spurious local minimum -- on a mesh of mostly obtuse triangles a
    vertex can read well below its neighbours -- and a descent can also run into the mesh boundary.
    A path stopped either way is finished along the shortest mesh-edge path from where it stopped
    to the nearest source, and a warning counts how many were. That tail follows edges, so it is
    longer than the surface geodesic it stands in for; it is never shorter. Only ``max_steps``
    running out leaves a path truncated, and a target in a component no source reaches keeps the
    path the descent left, with no tail and no count in the warning.
    [`descend_field`][ordito.geodesic_walk.descend_field] keeps the plain behaviour: a descent of
    any other field stops where that field does.

    **``"potpourri3d"`` is exact, but local.** Its path is a geodesic of the polyhedral surface:
    straight across every face and edge, and through a vertex only where the angle on both sides
    is at least ``pi``. It is the shortest path near the one it started from, not necessarily the
    shortest overall: on a surface with handles or holes it keeps the side of each one the
    starting edge path took, so it can be longer than the shortest path there. With several
    sources it runs from the source nearest along mesh edges. A target that is
    a source, or that no source reaches, gets a path of the one point (``potpourri3d`` raises on
    both). The mesh's tables, the edge paths and the traces of the result run on the mesh's
    device; the flips of each path depend on the ones before them, so they run path after path in
    one thread on the CPU device, whatever the mesh's device.

    See Also
    --------
    [`descend_field`][ordito.geodesic_walk.descend_field]
        The walk itself, for descending any other scalar field.
    [`ordito.heat.heat_geodesic`][ordito.heat.heat_geodesic]
        The field, when the distance is wanted and not the path.

    Examples
    --------
    Pole to pole on a unit sphere, whose true geodesic is half a great circle, of length ``pi``.
    The edge-flip path is the exact geodesic of the polyhedron, a little shorter than the heat
    method's descent:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = vertices.numpy()[:, 2]
    source = wp.array([np.argmax(heights)], dtype=wp.int32)
    targets = wp.array([np.argmin(heights)], dtype=wp.int32)
    points, offsets = od.geodesic_walk.geodesic_path(vertices, faces, source, targets)
    print(f"{od.polyline.polyline_length(points):.3f}")  # 3.137
    points, offsets = od.geodesic_walk.geodesic_path(
        vertices, faces, source, targets, method="potpourri3d"
    )
    print(f"{od.polyline.polyline_length(points):.3f}")  # 3.133
    ```
    """
    bound, owned, arguments = mesh_arguments(
        "geodesic_path", mesh, vertices, faces, (source, targets), 2
    )
    source = cast("wp.array[wp.int32]", arguments[0])
    targets = cast("wp.array[wp.int32]", arguments[1])
    require_same_device(vertices=bound.vertices, faces=bound.faces, source=source, targets=targets)
    if method not in ("heat", "potpourri3d"):
        raise ValueError(f"method must be 'heat' or 'potpourri3d', got {method!r}")
    if method == "potpourri3d":
        points, offsets, n_unconverged = _flip_geodesic_paths(bound, source, targets)
        if n_unconverged > 0:
            warnings.warn(
                f"geodesic_path: {n_unconverged} of {targets.size} paths stopped shortening at "
                "the step bound and are returned as they stood.",
                stacklevel=2,
            )
        return points, offsets
    # The ``vertices, faces`` form hands them on, so ``heat_geodesic`` builds and drops the mesh
    # (and logs a factorization it drops); a `Trimesh` keeps its heat solver for the next source.
    distance = (
        od.heat.heat_geodesic(bound, source, t=t)
        if owned
        else od.heat.heat_geodesic(bound.vertices, bound.faces, source, t)
    )
    # A path arrives on a source vertex, not at a field value: the heat distance is shifted to a
    # zero *mean* over the sources, so with several of them each reads a little above or below
    # zero and no one threshold separates the sources from their neighbours.
    points, offsets, n_completed = _descend(
        bound.vertices,
        bound.faces,
        distance,
        targets,
        -math.inf,
        bound.halfedge_twins,
        bound.vertex_face_adjacency,
        None,
        max_steps,
        complete_to=(bound, source),
    )
    if n_completed > 0:
        warnings.warn(
            f"geodesic_path: {n_completed} of {targets.size} paths stopped short of the source "
            "on the heat distance (a spurious local minimum, or the mesh boundary) and were "
            "finished along the shortest mesh-edge path from where they stopped.",
            stacklevel=2,
        )
    return points, offsets


def _descend(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    starts: wp.array[wp.int32],
    stop_value: float,
    twins: wp.array[wp.int32] | None,
    vertex_faces: tuple[wp.array[wp.int32], wp.array[wp.int32]] | None,
    gradients: wp.array[wp.vec3d] | None,
    max_steps: int,
    *,
    complete_to: tuple[Trimesh, wp.array[wp.int32]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], int]:
    """
    ``descend_field``'s walk; with ``complete_to``, short paths are finished along mesh edges.

    Returns the packed paths and how many were finished that way. ``complete_to`` is the mesh and
    its source vertices: a path then arrives on any source (whatever ``values`` reads there), and
    one that stops short of them -- at a local minimum of ``values``, on a flat face, or at the
    mesh boundary -- follows the shortest-edge-path tree to the nearest source. That tree is only
    built when a path needs it: the plain walk's counting pass counts the short ones first (one
    4-byte read), and a path in a component no source reaches stays as it stopped and is not
    counted.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_paths = starts.size
    if n_paths == 0:
        points, offsets = _launch.empty_packed(wp.vec3, device)
        return points, offsets, 0

    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    if vertex_faces is None:
        vertex_faces = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    if gradients is None:
        gradients = od.laplacian.face_gradients(vertices, faces, values)
    incident_faces, face_offsets = vertex_faces

    lanes = _walk_lanes(n_paths, device)
    stop_mask = (
        od.array.indices_to_mask(complete_to[1], n_vertices)
        if complete_to is not None
        else _launch.empty(0, dtype=wp.bool, device=device)
    )
    inputs = [
        vertices,
        faces,
        twins,
        face_offsets,
        incident_faces,
        values,
        gradients,
        starts,
        wp.float64(stop_value),
        stop_mask,
        wp.int32(max_steps),
        *_length_epsilon(vertices, faces),
        wp.int32(lanes),
    ]
    kernel = kernel_geodesic_walk.descent_paths
    no_graph = _launch.empty(0, dtype=wp.float32, device=device)
    no_offsets = _launch.empty(0, dtype=wp.int32, device=device)
    # A descent writes its start, at most two points a step (a crossing and the stop it lands
    # on), and with ``complete_to`` a point a step of the completion and the corner it starts from.
    staged_points, staged_next = _staged_points(n_paths, 3 * max_steps + 3, faces.size // 3, device)
    # The counting passes write nothing to ``out_points``; the pool's own buffer stands in.
    no_points = staged_points
    # The counts are written straight into the offsets' tail and scanned there in place (the
    # writing pass leaves them alone). With ``complete_to``, the same zeroed allocation also holds
    # one flag per path that stopped short, their number, how many the completion finished, and
    # the counts' total.
    if complete_to is None:
        offsets = _launch.zeros(n_paths + 1, dtype=wp.int32, device=device)
        short_paths = _launch.empty(0, dtype=wp.int32, device=device)
    else:
        offsets, short_paths = _launch.carve(
            wp.int32, (n_paths + 1, n_paths + 3), device, zeroed=True
        )
    counts = _launch.view(offsets, 1, n_paths + 1)
    _launch.launch(
        kernel,
        dim=n_paths * lanes,
        inputs=[*inputs, no_graph, short_paths, staged_points, staged_next, no_offsets],
        outputs=[counts, no_points],
        device=device,
    )
    graph = no_graph
    n_completed = 0
    if complete_to is None:
        _launch.array_scan(counts, out_array=counts, inclusive=True)
        # Host readback: only the device knows the walk's total length, and it sizes the point
        # buffer.
        total = int(read_scalar(offsets, n_paths))
    else:
        # Host readback, one copy: whether any path needs the edge-graph completion decides
        # whether its shortest-path relaxation runs at all, and the counts' total sizes the point
        # buffer.
        n_short, _, total = (int(v) for v in read_values(short_paths, n_paths, 3))
        if n_short > 0:
            graph = _edge_graph_distance(*complete_to)
            _launch.launch(
                kernel,
                dim=n_paths * lanes,
                inputs=[*inputs, graph, short_paths, staged_points, staged_next, no_offsets],
                outputs=[counts, no_points],
                device=device,
            )
            # Host readback, one copy: the warning's count of paths the completion actually
            # finished, and the total it changed.
            n_completed, total = (int(v) for v in read_values(short_paths, n_paths + 1, 2))
        # The total is known, so the offsets are the counts' scan behind a leading zero alone.
        _launch.array_scan(counts, out_array=counts, inclusive=True)
    points = _launch.empty(total, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel,
        dim=n_paths * lanes,
        inputs=[*inputs, graph, short_paths, staged_points, staged_next, offsets],
        outputs=[counts, points],
        device=device,
    )
    return points, offsets, n_completed


def _edge_graph_distance(mesh: Trimesh, sources: wp.array[wp.int32]) -> wp.array[wp.float32]:
    """``(n_vertices,)`` shortest mesh-edge path length to the nearest source, or unreached."""
    device = mesh.device
    n_vertices = mesh.n_vertices
    # The mesh's cached edges: a `Trimesh` the caller keeps pays for them once.
    adjacency = od.graph.edges_to_csr(n_vertices, mesh.edges_unique, mesh.edges_unique_length)
    seed = wp.full(
        n_vertices, float(kernel_geodesic_walk.GRAPH_UNREACHED), dtype=wp.float32, device=device
    )
    _launch.launch(
        kernel_geodesic_walk.seed_graph_sources,
        dim=sources.size,
        inputs=[sources],
        outputs=[seed],
        device=device,
    )
    return od.graph.shortest_path_envelope(adjacency, seed)


# ``geodesic_path(method="potpourri3d")``'s pools for the shortening: the wedge queue's entries
# beyond two an edge, the segment slots beyond one an edge, the output entries a path (a geodesic
# through no vertex is one). Each regrows and the shortening reruns when one runs out; the flip
# log restores every table instead.
_FLIP_HEAP_SLACK = 1 << 14
_FLIP_SEGMENT_SLACK = 64
_FLIP_ENTRIES_PER_PATH = 8
_FLIP_TOUCHED_SLACK = 1 << 12
# ``traceIntrinsicHalfedgeAlongInput``'s step bound, in faces crossed per face.
_FLIP_TRACE_STEPS_PER_FACE = 10


def _flip_geodesic_paths(
    mesh: Trimesh, source: wp.array[wp.int32], targets: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], int]:
    """
    ``geodesic_path(method="potpourri3d")``: geometry-central's edge-flip geodesics.

    ``kernels/algorithms/flip_geodesics``: the mesh tables, the shortest-edge tree and the traces on
    the mesh's device, one thread an element; the shortening, whose flip order is the algorithm,
    in one thread on the CPU device. Returns the packed paths and how many paths the step bound
    cut off.
    """
    device = mesh.device
    n_paths = targets.size
    if n_paths == 0:
        points, offsets = _launch.empty_packed(wp.vec3, device)
        return points, offsets, 0
    n_vertices = mesh.n_vertices
    input_mesh, flags, n_edges = _flip_input_mesh(mesh)
    _launch.launch(
        kernel_flip_geodesics.check_targets,
        dim=n_paths,
        inputs=[targets, wp.int32(n_vertices)],
        outputs=[flags],
        device=device,
    )
    incoming = _flip_shortest_edge_tree(input_mesh, n_vertices, 2 * n_edges, source, targets, flags)
    # Host readback: whether the mesh or an index was refused decides whether the call raises,
    # before any path is shortened.
    flag_values = [int(v) for v in read_values(flags, 0, kernel_flip_geodesics.FLAG_SIZE)]
    nonmanifold = flag_values[kernel_flip_geodesics.FLAG_NONMANIFOLD_VERTEX]
    invalid = flag_values[kernel_flip_geodesics.FLAG_INVALID_INDEX]
    if nonmanifold:
        raise ValueError(
            "geodesic_path(method='potpourri3d') requires a manifold mesh: a vertex's faces do not "
            "form a single fan."
        )
    if invalid:
        raise ValueError(
            f"geodesic_path: every entry of source and targets must be a vertex index in "
            f"[0, {n_vertices})"
        )

    entries, n_entries, n_unconverged = _flip_shorten_on_host(
        input_mesh, n_edges, incoming, targets
    )
    path_offsets, entry_path, entry_kind, entry_tail, entry_tip, entry_vector = (
        entry if entry.device == device else _launch.clone(entry, device=device)
        for entry in entries
    )

    limits = _launch.empty(n_entries, dtype=wp.int32, device=device)
    point_offsets = _launch.zeros(n_entries + 1, dtype=wp.int32, device=device)
    max_iterations = wp.int32(_FLIP_TRACE_STEPS_PER_FACE * (mesh.faces.size // 3))
    traced = [
        input_mesh,
        path_offsets,
        entry_path,
        entry_kind,
        entry_tail,
        entry_tip,
        entry_vector,
        max_iterations,
    ]
    _launch.launch(
        kernel_flip_geodesics.count_traced_points,
        dim=n_entries,
        inputs=[*traced, _launch.empty(0, dtype=wp.vec3, device=device)],
        outputs=[limits, _launch.view(point_offsets, 1, n_entries + 1)],
        device=device,
    )
    _launch.array_scan(point_offsets, out_array=point_offsets, inclusive=True)
    # Host readback: the point total sizes the polyline buffer.
    total = int(read_scalar(point_offsets, n_entries))
    points = _launch.empty(total, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_flip_geodesics.write_traced_points,
        dim=n_entries,
        inputs=[*traced, limits, point_offsets],
        outputs=[points],
        device=device,
    )
    offsets = cast("wp.array[wp.int32]", od.array.gather(point_offsets, path_offsets))
    return points, offsets, n_unconverged


def _flip_input_mesh(mesh: Trimesh) -> tuple[StructInstance, wp.array[wp.int32], int]:
    """Build the mesh in ``ManifoldSurfaceMesh`` numbering with its geometry; flags, edge count."""
    device = mesh.device
    faces = mesh.faces
    twins = mesh.halfedge_twins
    n_vertices = mesh.n_vertices
    n_corners = faces.size
    n_faces = n_corners // 3
    first_offsets = _launch.zeros(n_corners + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_flip_geodesics.flag_first_corners,
        dim=n_corners,
        inputs=[twins],
        outputs=[_launch.view(first_offsets, 1, n_corners + 1)],
        device=device,
    )
    _launch.array_scan(first_offsets, out_array=first_offsets, inclusive=True)
    # Host readback: the edge count sizes every halfedge and edge table.
    n_edges = int(read_scalar(first_offsets, n_corners))
    n_halfedges = 2 * n_edges

    input_mesh = kernel_flip_geodesics.InputMesh()
    input_mesh.position = od.array.astype(mesh.vertices, wp.vec3d)
    input_mesh.next, input_mesh.vertex, input_mesh.face, input_mesh.face_halfedge = _launch.carve(
        wp.int32, (n_halfedges, n_halfedges, n_halfedges, n_faces), device
    )
    (input_mesh.vertex_halfedge, input_mesh.boundary_out, last_corner, boundary_corner) = (
        _launch.carve(wp.int32, (n_vertices,) * 4, device)
    )
    _launch.fill_(input_mesh.boundary_out, -1)
    _launch.fill_(last_corner, -1)
    _launch.fill_(boundary_corner, -1)
    degree, flags = _launch.carve(
        wp.int32, (n_vertices, kernel_flip_geodesics.FLAG_SIZE), device, zeroed=True
    )
    (
        input_mesh.length,
        input_mesh.angle_sum,
        input_mesh.intrinsic_angle_sum,
        input_mesh.signpost,
        corner_angle,
        intrinsic_corner_angle,
    ) = _launch.carve(
        wp.float64, (n_edges, n_vertices, n_vertices, n_halfedges, n_halfedges, n_halfedges), device
    )
    input_mesh.vector_in_face, input_mesh.vector_in_vertex = _launch.carve(
        wp.vec2d, (n_halfedges, n_halfedges), device
    )
    _launch.launch(
        kernel_flip_geodesics.number_corners,
        dim=n_corners,
        inputs=[faces, twins, first_offsets],
        outputs=[input_mesh, last_corner, degree, boundary_corner, flags],
        device=device,
    )
    _launch.launch(
        kernel_flip_geodesics.face_geometry,
        dim=n_faces,
        inputs=[],
        outputs=[corner_angle, intrinsic_corner_angle, input_mesh],
        device=device,
    )
    _launch.launch(
        kernel_flip_geodesics.vertex_geometry,
        dim=n_vertices,
        inputs=[
            twins,
            first_offsets,
            last_corner,
            degree,
            boundary_corner,
            corner_angle,
            intrinsic_corner_angle,
        ],
        outputs=[input_mesh, flags],
        device=device,
    )
    return input_mesh, flags, n_edges


def _flip_shortest_edge_tree(
    input_mesh: StructInstance,
    n_vertices: int,
    n_halfedges: int,
    source: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    flags: wp.array[wp.int32],
) -> wp.array[wp.int32]:
    """Each vertex's halfedge in geometry-central's shortest-edge-path tree from the sources."""
    device = wp.get_device(flags.device)
    distance = _launch.full(n_vertices, math.inf, dtype=wp.float64, device=device)
    source_mask = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
    if source.size > 0:
        _launch.launch(
            kernel_flip_geodesics.seed_sources,
            dim=source.size,
            inputs=[source],
            outputs=[distance, source_mask, flags],
            device=device,
        )
    if device.is_cuda:
        # The relaxation passes as one conditional graph; a pass that moves nothing ends it.
        state = _launch.array([0, 1, 0], dtype=wp.int32, device=device)
        n_vertices_i32 = wp.int32(n_vertices)

        def relaxation_pass() -> None:
            _launch.launch(
                kernel_flip_geodesics.relax_distances,
                dim=n_vertices,
                inputs=[input_mesh, distance],
                outputs=[state],
                device=device,
            )
            _launch.launch(
                kernel_array.loop_advance, dim=1, inputs=[n_vertices_i32, state], device=device
            )

        run_device_loop(
            device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), relaxation_pass
        )
    else:
        # The CPU device runs a launch's threads one after another, so the relaxation's passes
        # would each sweep every vertex: the lazy Dijkstra builds the same tree in one pass.
        incoming = _launch.empty(n_vertices, dtype=wp.int32, device=device)
        state, wanted = _launch.carve(wp.int32, (1, n_vertices), device)
        _launch.launch(
            kernel_flip_geodesics.shortest_edge_tree_serial,
            dim=1,
            inputs=[
                input_mesh,
                source_mask,
                targets,
                _launch.empty(n_halfedges, dtype=wp.float64, device=device),
                _launch.empty(n_halfedges, dtype=wp.int32, device=device),
                wanted,
                state,
            ],
            outputs=[incoming],
            device=device,
        )
        return incoming
    incoming = _launch.empty(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_flip_geodesics.settle_incoming,
        dim=n_vertices,
        inputs=[input_mesh, distance, source_mask],
        outputs=[incoming],
        device=device,
    )
    return incoming


def _flip_shorten_on_host(
    input_mesh: StructInstance,
    n_edges: int,
    incoming: wp.array[wp.int32],
    targets: wp.array[wp.int32],
) -> tuple[tuple[wp.array[Any], ...], int, int]:
    """
    Shorten every path on the CPU device, each from the input triangulation.

    Returns the path offsets and the entries (path, kind, tail, tip, trace vector), on the CPU
    device from one range and on the targets' device from several, the entry count and how many
    paths the step bound cut off. Each path starts from the input triangulation
    (``kernels/algorithms/flip_geodesics.shorten_paths`` rewinds its flips), so the paths split
    into ranges, each shortened by one launch with a triangulation and pools of its own, built and
    launched on host threads (a CPU launch releases the interpreter), and their entries joined in
    path order. A range whose pools ran out runs again, grown, from a fresh copy of the input
    triangulation.
    """
    host = wp.get_device("cpu")
    n_paths = targets.size

    def on_host(array: wp.array[Any]) -> wp.array[Any]:
        return array if array.device == host else _launch.clone(array, device=host)

    host_mesh = kernel_flip_geodesics.InputMesh()
    for name in (
        "next",
        "vertex",
        "face",
        "boundary_out",
        "length",
        "intrinsic_angle_sum",
        "signpost",
    ):
        setattr(host_mesh, name, on_host(getattr(input_mesh, name)))
    incoming_host = on_host(incoming)
    targets_host = on_host(targets)
    n_groups = 1
    if n_paths >= _FLIP_SPLIT_FROM:
        n_groups = min(_FLIP_GROUPS, os.cpu_count() or 1, n_paths // _FLIP_GROUP_PATHS)
    starts = [n_paths * g // n_groups for g in range(n_groups + 1)]
    capacities = [
        [2 * n_edges + _FLIP_HEAP_SLACK, n_edges + _FLIP_SEGMENT_SLACK, 0] for _ in range(n_groups)
    ]
    results: list[tuple[np.ndarray, ...] | None] = [None] * n_groups
    unconverged = [0] * n_groups
    single: tuple[wp.array[Any], ...] | None = None
    pending = list(range(n_groups))

    def record(g: int) -> tuple[Any, tuple[wp.array[Any], ...]]:
        return _flip_group_run(
            host_mesh,
            (host_mesh.next, host_mesh.vertex, host_mesh.length, host_mesh.signpost),
            n_edges,
            incoming_host,
            _launch.view(targets_host, starts[g], starts[g + 1]),
            capacities[g],
        )

    def record_and_launch(g: int) -> tuple[Any, tuple[wp.array[Any], ...]]:
        run = record(g)
        run[0].launch()
        return run

    with concurrent.futures.ThreadPoolExecutor(n_groups) as pool:
        while pending:
            # Each range copies its triangulation and allocates its pools on its own thread too
            # (Warp's native copies and allocations release the interpreter), the first on this
            # one, which loads the kernel's module before any other thread records a launch.
            first = record(pending[0])
            others = [pool.submit(record_and_launch, g) for g in pending[1:]]
            first[0].launch()
            runs = [first, *(future.result() for future in others)]
            still = []
            for g, (_, outputs) in zip(pending, runs, strict=True):
                # Host reads, one a range: its entry count, and whether a pool ran out.
                status = outputs[-1].numpy()
                n_entries, heap_overflow, segment_overflow, n_unconverged = (
                    int(v) for v in status[: kernel_flip_geodesics.STATUS_SIZE]
                )
                heap, segment, entry = capacities[g]
                grow = bool(heap_overflow or segment_overflow)
                capacities[g] = [
                    heap * 4 if heap_overflow else heap,
                    segment * 4 if segment_overflow else segment,
                    max(entry, n_entries) if not grow and n_entries > entry else entry,
                ]
                if grow or n_entries > entry:
                    still.append(g)
                    continue
                unconverged[g] = n_unconverged
                if n_groups == 1:
                    # One range is the whole answer, already in place.
                    single = (outputs[0], *(_launch.view(a, 0, n_entries) for a in outputs[1:-1]))
                    break
                offsets, path, kind, tail, tip, vector = (array.numpy() for array in outputs[:-1])
                results[g] = (
                    offsets,
                    path[:n_entries] + starts[g],
                    kind[:n_entries],
                    tail[:n_entries],
                    tip[:n_entries],
                    vector[:n_entries],
                )
            pending = still
    if single is not None:
        return single, single[1].size, unconverged[0]
    done = [result for result in results if result is not None]
    bases = np.cumsum([0] + [len(result[1]) for result in done])
    offsets_np = np.concatenate(
        [[0]] + [result[0][1:] + base for result, base in zip(done, bases[:-1], strict=True)]
    ).astype(np.int32)
    n_entries = int(bases[-1])
    n_unconverged = sum(unconverged)
    # Joined on the host and uploaded to the targets' device once: the offsets and the four
    # integer columns in one buffer, the trace vectors in another.
    columns = np.concatenate(
        [offsets_np, *(np.concatenate([result[k] for result in done]) for k in range(1, 5))]
    )
    integers = wp.array(columns.astype(np.int32), dtype=wp.int32, device=targets.device)
    bounds = [0, n_paths + 1, *(n_paths + 1 + k * n_entries for k in range(1, 5))]
    entries = (
        *(_launch.view(integers, a, b) for a, b in itertools.pairwise(bounds)),
        wp.array(
            np.concatenate([result[5] for result in done]).reshape(-1, 2),
            dtype=wp.vec2d,
            device=targets.device,
        ),
    )
    return entries, n_entries, n_unconverged


# The ranges of paths the shortening splits into: at most this many, of at least this many paths,
# from this many paths on. A range's own triangulation and pools cost about what a few paths do:
# two ranges of 8 took 16 paths 1.05x, of 12 paths 24 0.85x where ranges of 8 took them 0.74x
# (an icosphere of six subdivisions; 64 paths 0.47x, 512 0.15x).
_FLIP_GROUPS = 16
_FLIP_GROUP_PATHS = 8
_FLIP_SPLIT_FROM = 24


def _flip_group_run(
    host_mesh: StructInstance,
    input_triangulation: tuple[wp.array[Any], ...],
    n_edges: int,
    incoming: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    capacities: list[int],
) -> tuple[Any, tuple[wp.array[Any], ...]]:
    """
    Record one range's ``shorten_paths`` launch on the CPU device; return it and its outputs.

    The range's triangulation is a fresh copy of the input's (``input_triangulation``: its
    ``next``, ``vertex``, ``length`` and ``signpost`` tables), and its pools are sized by
    ``capacities`` (heap, segment and entry slots; an entry capacity of ``0`` starts at
    ``_FLIP_ENTRIES_PER_PATH`` a path). The outputs end with the status words.
    """
    host = wp.get_device("cpu")
    n_halfedges = 2 * n_edges
    heap_capacity, segment_capacity, entry_capacity = capacities
    entry_capacity = max(entry_capacity, _FLIP_ENTRIES_PER_PATH * targets.size)
    capacities[2] = entry_capacity
    tri = kernel_flip_geodesics.Intrinsic()
    tri.next, tri.vertex, tri.length, tri.signpost = (
        _launch.clone(table, device=host) for table in input_triangulation
    )
    tri.original = _launch.full(n_edges, 1, dtype=wp.int32, device=host)
    net = kernel_flip_geodesics.Network()
    (
        net.segment_halfedge,
        net.segment_prev,
        net.segment_next,
        net.segment_id,
        net.stack_prev,
        net.stack_next,
        net.free,
    ) = _launch.carve(wp.int32, (segment_capacity,) * 7, host)
    net.heap_angle = _launch.empty(heap_capacity, dtype=wp.float64, device=host)
    (
        net.heap_turn,
        net.heap_id,
        net.heap_slot,
        net.edge_front,
        net.edge_back,
        net.touched,
        net.scratch,
        net.state,
    ) = _launch.carve(
        wp.int32,
        (
            heap_capacity,
            heap_capacity,
            heap_capacity,
            n_edges,
            n_edges,
            5 * n_edges + _FLIP_TOUCHED_SLACK,
            n_halfedges,
            kernel_flip_geodesics.STATE_SIZE,
        ),
        host,
    )
    path_offsets = _launch.empty(targets.size + 1, dtype=wp.int32, device=host)
    entry_path, entry_kind, entry_tail, entry_tip = _launch.carve(
        wp.int32, (entry_capacity,) * 4, host
    )
    entry_vector = _launch.empty(entry_capacity, dtype=wp.vec2d, device=host)
    status = _launch.zeros(kernel_flip_geodesics.STATUS_SIZE, dtype=wp.int32, device=host)
    outputs = (path_offsets, entry_path, entry_kind, entry_tail, entry_tip, entry_vector, status)
    command = wp.launch(
        kernel_flip_geodesics.shorten_paths,
        dim=1,
        inputs=[host_mesh, tri, net, incoming, targets],
        outputs=list(outputs),
        device=host,
        record_cmd=True,
    )
    return command, outputs


def shorten_loop(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: Sequence[wp.array[wp.int32]],
    *,
    max_iter: int = 100,
    tolerance: float = 0.0,
    twins: wp.array[wp.int32] | None = None,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
) -> tuple[list[wp.array[wp.int32]], int]:
    """
    Shorten closed edge loops within their homotopy class, keeping them on mesh edges.

    Takes exactly what [`homology_generators`][ordito.homology.homology_generators] returns -- a
    list of vertex-index cycles -- and returns cycles of the same kind, shorter. The loops a
    tree-cotree construction produces are as long and as jagged as the spanning trees that built
    them, which is fine for a *basis* and useless as a curve; this makes them short enough to look
    at, cut along, or measure.

    Each sweep rewrites the loop *locally*. Around one of its vertices ``b``, with neighbours ``a``
    and ``c`` on the loop, the sub-path ``a -> b -> c`` is replaced by whichever way round ``b``'s
    link is shorter, when either beats going through ``b``. Both alternatives lie in the star of
    ``b``, which is a disk, so the replacement cannot change the loop's homotopy class -- and since
    it is only ever accepted when it is strictly shorter, the total length falls monotonically. A
    loop that doubles back on itself has the spur contracted away by the same rule.

    !!! note "Shorter, not geodesic"
        The result stays **on the edge graph**, so it is a local minimum over edge paths rather than
        the shortest curve on the surface -- reaching that means letting the loop cross face
        interiors, which needs an intrinsic triangulation it can flip. The gap to the true geodesic
        is small, and is widest where the mesh is a regular grid whose rows are not geodesics,
        because no one-ring move can step the loop off a row without lengthening the edge path
        first -- so the result is a true local minimum over edge paths, and the remaining gap is the
        edge graph's, not the sweep's.

        Sweeps ratchet the loop one position at a time, so the count needed to converge grows with
        the loop's own length, which is what ``max_iter`` has to cover.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    loops
        Closed vertex-index cycles, each a rank-1 array whose consecutive entries share an
        edge, as do its last and first. Loops shorter than three vertices are returned unchanged.
    max_iter
        Cap on the number of sweeps. Two sweeps of opposite parity are needed to give every
        position a turn, so an odd cap leaves one parity class one visit short. Sweeping stops as
        soon as two consecutive sweeps accept nothing, which is what makes the default generous
        rather than expensive.
    tolerance
        Absolute length a replacement must save to be accepted. The default ``0.0`` accepts any
        strict improvement, which is what makes the result independent of the sweep count; raise it
        to stop the last few sweeps chasing float32 noise on a fine mesh.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins], or
        ``None`` to compute them from ``faces``.
    rings
        ``(3 * n_faces,)``, ``(n_vertices + 1,)`` and ``(n_vertices,)`` precomputed
        [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] as
        ``(ring_halfedges, offsets, is_boundary)``, or ``None``. Depends on the connectivity alone,
        so one CSR serves every fan walk over the same mesh --
        [`Trimesh.vertex_one_rings`][ordito.mesh.Trimesh.vertex_one_rings] has it cached, and
        passing it skips the vertex-manifold check and the host readback that check costs.

    Returns
    -------
    loops : list[wp.array[wp.int32]]
        One shortened cycle per input loop, in the same order, on ``faces.device``.
    sweeps : int
        How many sweeps ran. Below ``max_iter`` this means the loops stopped changing, so the
        answer is locally minimal; equal to it, the cap bound the result.

    Raises
    ------
    TypeError
        If any loop is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices``, ``faces``, ``loops``, ``twins`` and ``rings`` are not all on one device.

    See Also
    --------
    [`shorten_loop_with_offsets`][ordito.geodesic_walk.shorten_loop_with_offsets]
        The same, on loops packed into one buffer with their offsets.
    [`homology_generators`][ordito.homology.homology_generators]
        Produces the loops this shortens.
    [`polyline_length`][ordito.polyline.polyline_length]
        Measures the result, after gathering the positions.
    [`geodesic_path`][ordito.geodesic_walk.geodesic_path]
        The open, endpoint-to-endpoint problem, solved by descending a heat field instead.

    Examples
    --------
    A loop around a torus with a bump in it slides onto the shortest nearby ring of edges:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.torus(2.0, 0.5, major_sections=16, minor_sections=8)
    bumpy = [0, 1, 2, 3, 19, 35, 36, 37, 21, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    (loop,), sweeps = od.geodesic_walk.shorten_loop(
        vertices, faces, [wp.array(bumpy, dtype=wp.int32)]
    )
    ring = vertices.numpy()[loop.numpy()]
    length = np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1).sum()
    print(loop.size, f"{length:.3f}")  # 16 12.486
    ```
    """
    require_same_device(vertices=vertices, faces=faces, loops=loops, twins=twins, rings=rings)
    loops = list(loops)
    for loop in loops:
        odt.ensure_ndim(loop, 1, dtype=wp.int32)
    if not loops or faces.size == 0 or max_iter <= 0:
        return loops, 0
    # ``copy=False``: the packed form never writes into its input -- each sweep writes a freshly
    # sized buffer -- so the first pack can alias the caller's loops.
    packed, loop_offsets = od.array.pack_1d_arrays(loops, copy=False)
    packed, loop_offsets, sweeps = shorten_loop_with_offsets(
        vertices,
        faces,
        packed,
        loop_offsets,
        max_iter=max_iter,
        tolerance=tolerance,
        twins=twins,
        rings=rings,
    )
    return od.array.split(packed, loop_offsets), sweeps


def shorten_loop_with_offsets(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    *,
    max_iter: int = 100,
    tolerance: float = 0.0,
    twins: wp.array[wp.int32] | None = None,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int]:
    """
    Shorten packed closed edge loops within their homotopy class, keeping them on mesh edges.

    The packed form of [`shorten_loop`][ordito.geodesic_walk.shorten_loop], which is this plus a
    pack of its input and a split of its output: the loops arrive and leave as one buffer and its
    offsets, which is what
    [`homology_generators_with_offsets`][ordito.homology.homology_generators_with_offsets]
    returns. The sweeps and their stopping rule are ``shorten_loop``'s.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    loops
        ``(n_positions,)`` every loop's vertex-index cycle, loop after loop,
        ``n_positions == loop_offsets[-1]``. It is read and never written.
    loop_offsets
        ``(n_loops + 1,)`` total-terminated offsets: loop ``i`` is ``loops[loop_offsets[i] :
        loop_offsets[i + 1]]``.
    max_iter
        Cap on the number of sweeps, as in [`shorten_loop`][ordito.geodesic_walk.shorten_loop].
    tolerance
        Absolute length a replacement must save to be accepted.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins], or
        ``None`` to compute them from ``faces``.
    rings
        ``(3 * n_faces,)``, ``(n_vertices + 1,)`` and ``(n_vertices,)`` precomputed
        [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] as
        ``(ring_halfedges, offsets, is_boundary)``, or ``None``.

    Returns
    -------
    loops : wp.array[wp.int32]
        ``(m,)`` shortened cycles, in the input's order, on ``faces.device``, ``m`` the returned
        ``loop_offsets[-1]``. The input itself when no sweep ran.
    loop_offsets : wp.array[wp.int32]
        ``(n_loops + 1,)`` total-terminated offsets of the shortened cycles.
    sweeps : int
        How many sweeps ran; equal to ``max_iter``, the cap bound the result.

    Raises
    ------
    TypeError
        If ``loops`` or ``loop_offsets`` is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices``, ``faces``, ``loops``, ``loop_offsets``, ``twins`` and ``rings`` are not
        all on one device.

    See Also
    --------
    [`shorten_loop`][ordito.geodesic_walk.shorten_loop]
        The same, on one array per loop.
    [`homology_generators_with_offsets`][ordito.homology.homology_generators_with_offsets]
        Produces the loops this shortens, in this form.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.torus(2.0, 0.5, major_sections=16, minor_sections=8)
    bumpy = [0, 1, 2, 3, 19, 35, 36, 37, 21, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    loops = wp.array(bumpy, dtype=wp.int32)
    loop_offsets = wp.array([0, len(bumpy)], dtype=wp.int32)
    loops, loop_offsets, sweeps = od.geodesic_walk.shorten_loop_with_offsets(
        vertices, faces, loops, loop_offsets
    )
    print(loop_offsets.numpy().tolist())  # [0, 16]
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        loops=loops,
        loop_offsets=loop_offsets,
        twins=twins,
        rings=rings,
    )
    device = faces.device
    odt.ensure_ndim(loops, 1, dtype=wp.int32)
    odt.ensure_ndim(loop_offsets, 1, dtype=wp.int32)
    n_loops = loop_offsets.size - 1
    if n_loops <= 0 or faces.size == 0 or max_iter <= 0 or loops.size == 0:
        return loops, loop_offsets, 0

    # ``twins`` is read only to walk the rings, which build their own when it is ``None``.
    ring_halfedges, ring_offsets, is_boundary = (
        rings
        if rings is not None
        else vertex_one_rings(faces, twins=twins, n_vertices=vertices.size)
    )

    # The loop state lives on the device (``kernels/geodesic_walk.SHORTEN_*``) and every sweep
    # reads and writes fixed buffers sized with headroom, so a group of sweeps is recorded once
    # and replayed with one read of the state per group. A sweep whose rewrite would outgrow the
    # buffers is undone and flagged; the buffers then grow and the loop resumes from that sweep.
    state = _launch.array([0, 0, 0, 0, 0, loops.size, 0, max_iter], dtype=wp.int32, device=device)
    capacity = max(
        loops.size + max(_SHORTEN_LOOP_SLACK, int(_SHORTEN_LOOP_GROWTH * loops.size)), n_loops + 1
    )
    current, current_offsets, length = loops, loop_offsets, loops.size
    while True:
        sweep = _LoopSweep(
            vertices,
            faces,
            ring_offsets,
            ring_halfedges,
            is_boundary,
            tolerance,
            state,
            capacity,
            n_loops,
        )
        sweep.load(current, current_offsets, length)
        values = sweep.run()
        if values[kernel_geodesic_walk.SHORTEN_OVERFLOW] == 0:
            break
        # The undone sweep's rewritten length is the counts' total; grow past it, keep the loops
        # it started from, and clear the stop, overflow and accepted flags to run it again.
        needed = int(read_scalar(sweep.positions, capacity))
        capacity = 2 * max(needed, capacity)
        current, current_offsets = sweep.loops, sweep.loop_offsets
        length = values[kernel_geodesic_walk.SHORTEN_LENGTH]
        _launch.zero_(
            state[kernel_geodesic_walk.SHORTEN_DONE : kernel_geodesic_walk.SHORTEN_LENGTH]
        )

    sweeps = values[kernel_geodesic_walk.SHORTEN_SWEEPS]
    if values[kernel_geodesic_walk.SHORTEN_ANY] == 0:
        return loops, loop_offsets, sweeps
    return (
        odt.as_dense(sweep.loops[: values[kernel_geodesic_walk.SHORTEN_LENGTH]]),
        sweep.loop_offsets,
        sweeps,
    )


# Headroom of ``shorten_loop_with_offsets``' fixed buffers over the input's positions: a sweep
# replaces a vertex by a way round its link, which can lengthen a loop, and one that would not fit
# is undone and run again in buffers twice the size. The shortening usually shrinks the loops, so
# a sweep needing the headroom is rare and an overflow rarer.
_SHORTEN_LOOP_GROWTH = 0.5
_SHORTEN_LOOP_SLACK = 1024

# Sweeps recorded into one replayed graph on CUDA, after the first sweep is issued directly (it
# decides whether there is a loop to record at all). A sweep is nine launches; a group that runs
# past the loop's stop is guarded no-op launches, so the group trades those against one state read
# per sweep.
_SHORTEN_LOOP_GRAPH_SWEEPS = 4


class _LoopSweep:
    """The fixed buffers of one ``shorten_loop_with_offsets`` run, and the sweep over them."""

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        ring_offsets: wp.array[wp.int32],
        ring_halfedges: wp.array[wp.int32],
        is_boundary: wp.array[wp.bool],
        tolerance: float,
        state: wp.array[wp.int32],
        capacity: int,
        n_loops: int,
    ) -> None:
        device = faces.device
        self._device = wp.get_device(device)
        self._mesh = (vertices, faces, ring_offsets, ring_halfedges, is_boundary)
        self._tolerance = tolerance
        self._state = state
        self._capacity = capacity
        # The current loops, which the call returns, and the rewrite the compaction reads back
        # into them, carved with the per-slot arc tables.
        self.loops = _launch.empty(capacity, dtype=wp.int32, device=device)
        self.loop_offsets = _launch.empty(n_loops + 1, dtype=wp.int32, device=device)
        self._rewritten, self._rewritten_offsets, self._arc_slot, self._arc_step = _launch.carve(
            wp.int32, (capacity, n_loops + 1, capacity, capacity), device
        )
        # Each is the per-slot counts behind a leading zero, scanned in place into exclusive
        # offsets with the total at entry ``capacity``.
        self.positions, self._kept = _launch.carve(
            wp.int32, (capacity + 1, capacity + 1), device, zeroed=True
        )
        self._counts = _launch.view(self.positions, 1, capacity + 1)
        self._kept_counts = _launch.view(self._kept, 1, capacity + 1)

    def load(
        self, loops: wp.array[wp.int32], loop_offsets: wp.array[wp.int32], length: int
    ) -> None:
        """Copy the loops a run starts from into the buffers."""
        _launch.copy(self.loops, loops, count=length)
        _launch.copy(self.loop_offsets, loop_offsets)

    def run(self) -> list[int]:
        """Sweep until the device-side state stops the loop; return the state's words."""
        n_state = kernel_geodesic_walk.SHORTEN_STATE_SLOTS
        if self._device.is_cpu:
            # The CPU runs a launch's threads one after another, so the whole loop runs as one
            # thread's sweeps in that same order (``kernels/geodesic_walk.serial_shorten_sweeps``).
            self._issue_serial()
            return [int(v) for v in read_values(self._state, 0, n_state)]
        self._issue()
        values = [int(v) for v in read_values(self._state, 0, n_state)]
        if values[kernel_geodesic_walk.SHORTEN_DONE] != 0:
            return values
        if not (self._device.is_cuda and wp.is_conditional_graph_supported()):
            while values[kernel_geodesic_walk.SHORTEN_DONE] == 0:
                self._issue()
                values = [int(v) for v in read_values(self._state, 0, n_state)]
            return values
        with wp.ScopedCapture(self._device) as capture:
            for _ in range(_SHORTEN_LOOP_GRAPH_SWEEPS):
                self._issue()
        graph = capture.graph
        assert graph is not None
        while values[kernel_geodesic_walk.SHORTEN_DONE] == 0:
            wp.capture_launch(graph)
            values = [int(v) for v in read_values(self._state, 0, n_state)]
        return values

    def _issue_serial(self) -> None:
        """Every sweep until the state stops the loop, in one single-thread launch."""
        vertices, faces, ring_offsets, ring_halfedges, is_boundary = self._mesh
        _launch.launch(
            kernel_geodesic_walk.serial_shorten_sweeps,
            dim=1,
            inputs=[
                vertices,
                faces,
                ring_offsets,
                ring_halfedges,
                is_boundary,
                self._tolerance,
                self.loops,
                self.loop_offsets,
                self._rewritten,
                self._rewritten_offsets,
                self.positions,
                self._counts,
                self._kept,
                self._kept_counts,
                self._arc_slot,
                self._arc_step,
                self._state,
            ],
            device=self._device,
        )

    def _issue(self) -> None:
        """One sweep: counts, rewrite, compaction and the closing state update."""
        vertices, faces, ring_offsets, ring_halfedges, is_boundary = self._mesh
        device = self._device
        state = self._state
        capacity = self._capacity
        _launch.launch(
            kernel_geodesic_walk.shorten_loop_counts,
            dim=capacity,
            inputs=[
                vertices,
                faces,
                ring_offsets,
                ring_halfedges,
                is_boundary,
                self.loops,
                self.loop_offsets,
                self._tolerance,
                state,
                self._counts,
                self._arc_slot,
                self._arc_step,
            ],
            device=device,
        )
        _launch.array_scan(self._counts, out_array=self._counts, inclusive=True)
        _launch.launch(
            kernel_geodesic_walk.shorten_loop_write,
            dim=capacity,
            inputs=[
                faces,
                ring_offsets,
                ring_halfedges,
                self.loops,
                self.loop_offsets,
                self._arc_slot,
                self._arc_step,
                self.positions,
                state,
                self._rewritten,
                self._rewritten_offsets,
            ],
            device=device,
        )
        # Drop positions repeating their predecessor, which a contracted spur leaves behind.
        _launch.launch(
            kernel_geodesic_walk.distinct_from_predecessor,
            dim=capacity,
            inputs=[
                self._rewritten,
                self._rewritten_offsets,
                self.positions,
                state,
                self._kept_counts,
            ],
            device=device,
        )
        _launch.array_scan(self._kept_counts, out_array=self._kept_counts, inclusive=True)
        _launch.launch(
            kernel_geodesic_walk.compact_kept,
            dim=capacity,
            inputs=[
                self._rewritten,
                self._rewritten_offsets,
                self._kept,
                self.positions,
                state,
                self.loops,
                self.loop_offsets,
            ],
            device=device,
        )
        _launch.launch(
            kernel_geodesic_walk.shorten_loop_advance,
            dim=1,
            inputs=[self.positions, self._kept, state],
            device=device,
        )


def _length_epsilon(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> list[object]:
    """
    Return the two arguments the walk kernels form their crossing-distance floor from.

    The walk needs *some* scale-aware floor: a ray starting at a vertex stands on two of its face's
    edges, and one starting on an edge stands on that edge, so a crossing at distance ~0 has to be
    rejected or the walk rotates on the spot. It is ``1e-6`` of
    [`mean_edge_length`][ordito.edges.mean_edge_length], which keeps the behaviour invariant under
    a global rescale of the mesh. The per-face edge length total stays on the device and each
    kernel divides it itself (``kernels/geodesic_walk.mean_length_epsilon``), so the floor costs no
    readback: the total is [`edge_length_sum`][ordito.edges.edge_length_sum], the one
    ``mean_edge_length`` reads back, and the floor is the same ``float32`` the host expression
    ``1e-6 * mean_edge_length(...)`` rounds to.
    """
    return [od.edges.edge_length_sum(vertices, faces), faces.size // 3 * 3]


def _trace(
    kernel: odt.Kernel,
    inputs: Sequence[object],
    n_rays: int,
    max_steps: int,
    n_faces: int,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Run a tracing kernel twice: once to count each ray's points, once to write them.

    The walk's length is not known in advance, so counting first and sizing the output exactly
    beats reserving ``max_steps`` points per ray, which at the default cap would be tens of
    megabytes for a few thousand rays. The counting pass stages the points it counts
    (``_staged_points``), so the writing pass copies a ray rather than walking it again.
    """
    # A ray writes its start, then at most one point a step.
    max_points = max_steps + 1
    # The counts are written straight into the offsets' tail and scanned there in place; the
    # writing pass leaves them alone.
    offsets = _launch.zeros(n_rays + 1, dtype=wp.int32, device=device)
    counts = _launch.view(offsets, 1, n_rays + 1)
    no_offsets = _launch.empty(0, dtype=wp.int32, device=device)
    lanes = _walk_lanes(n_rays, device)
    staged_points, staged_next = _staged_points(n_rays, max_points, n_faces, device)
    inputs = [*inputs, wp.int32(lanes), staged_points, staged_next]
    # The counting pass writes nothing to ``out_points``; the pool's own buffer stands in.
    _launch.launch(
        kernel,
        dim=n_rays * lanes,
        inputs=[*inputs, no_offsets, counts, staged_points],
        device=device,
    )
    _launch.array_scan(counts, out_array=counts, inclusive=True)

    # Host readback: only the device knows the walk's total length, and it sizes the point buffer.
    total = int(read_scalar(offsets, n_rays))
    points = _launch.empty(total, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel, dim=n_rays * lanes, inputs=[*inputs, offsets, counts, points], device=device
    )
    return points, offsets


def _staged_points(
    n_walks: int, max_points: int, n_faces: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the pool a walk kernel's counting pass stages its points in, and its chunk links.

    The writing pass then copies each walk from it instead of walking it again, and walks again
    only one the pool did not hold (``kernels/geodesic_walk.copy_staged``). The walks share the
    pool, a long one taking what a short one left, so it is sized for each at about a walk across
    the mesh (``isqrt(n_faces)`` points, at most its ``max_points`` bound), up to
    ``_STAGED_POINTS`` in all, in one allocation. The links' last slot, the chunk counter, starts
    past the walks' own first chunks.
    """
    chunk = int(kernel_geodesic_walk.STAGED_CHUNK)
    per_walk = min(max_points, math.isqrt(n_faces) + chunk)
    n_chunks = min(n_walks * -(-per_walk // chunk), _STAGED_POINTS // chunk)
    staged_next, staged_points = _launch.carve_typed(
        ((wp.int32, n_chunks + 1), (wp.vec3, n_chunks * chunk)), device
    )
    _launch.fill_(_launch.view(staged_next, n_chunks, n_chunks + 1), min(n_walks, n_chunks))
    return staged_points, staged_next


# The most points a walk call stages (12 MiB), however many walks: what it does not hold is
# walked twice, as every walk was before the pool. Its own size is the call's peak memory it adds
# (``descend_field``: 4 096 paths on a sphere of 82 k faces, 12.7 -> 25 MiB).
_STAGED_POINTS = 1 << 20


def _walk_lanes(n_walks: int, device: wp.DeviceLike) -> int:
    """
    Threads a walk for the tracing and descent kernels: up to a warp a walk on CUDA, one on the CPU.

    A walk is a serial state machine whose steps branch, so walks sharing a warp wait on each
    other's branches and on the longest of them; few walks each get a warp of their own
    ([`walk_lanes`][ordito._launch.walk_lanes]). The CPU runs a launch's threads one after
    another, where the idle lanes would only cost.
    """
    return _launch.walk_lanes(n_walks) if wp.get_device(device).is_cuda else 1
