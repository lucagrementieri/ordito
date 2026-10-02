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

from collections.abc import Sequence

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.halfedge import halfedge_twins, vertex_one_rings
from ordito.kernels import array as kernel_array
from ordito.kernels import geodesic_walk as kernel_geodesic_walk

_DEFAULT_MAX_STEPS = 4096


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

    Which incident face the ray starts in is decided in the vertex's *flattened* tangent space
    ([`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles]): rescaling the
    incident corner angles to a full turn makes the fan a disk, so every tangent direction lands in
    exactly one wedge, including directions a naive per-face projection would place outside all of
    them. At a boundary vertex the fan spans only half a disk, and a direction outside it — pointing
    off the surface — traces nothing.

    A walk stops early when it reaches the mesh boundary or exceeds ``max_steps`` edge crossings.

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
        [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames], or ``None``. They
        define each vertex's tangent plane, which sets both the trace length and the starting wedge.
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

    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    if rings is None:
        rings = vertex_one_rings(faces, twins=twins, n_vertices=n_vertices)
    ring_halfedges, ring_offsets, is_boundary = rings
    if frames is None:
        frames = od.tangent_space.vertex_tangent_frames(vertices, faces, rings=rings)
    basis_x, basis_y, normals = frames

    inputs = [
        vertices,
        faces,
        twins,
        od.triangles.face_angles(vertices, faces),
        ring_offsets,
        ring_halfedges,
        is_boundary,
        basis_x,
        basis_y,
        normals,
        start_vertices,
        directions,
        wp.int32(max_steps),
        wp.float32(_length_epsilon(vertices, faces)),
    ]
    return _trace(kernel_geodesic_walk.trace_from_vertices, inputs, n_rays, device)


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
        wp.float32(_length_epsilon(vertices, faces)),
    ]
    return _trace(kernel_geodesic_walk.trace_from_faces, inputs, n_rays, device)


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
    device = vertices.device
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
    n_paths = starts.size
    if n_paths == 0:
        return _launch.empty_packed(wp.vec3, device)

    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    if vertex_faces is None:
        vertex_faces = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    if gradients is None:
        gradients = od.laplacian.face_gradients(vertices, faces, values)
    incident_faces, face_offsets = vertex_faces

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
        wp.int32(max_steps),
        wp.float32(_length_epsilon(vertices, faces)),
    ]
    return _trace(kernel_geodesic_walk.descent_paths, inputs, n_paths, device)


def geodesic_path(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    source: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    *,
    t: float | None = None,
    operators: od.heat.HeatOperators | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Trace a path across the surface from each target vertex back to a source set.

    The **point-to-point** geodesic, as against this module's straightest walks and
    [`heat_geodesic`][ordito.heat.heat_geodesic]'s distance *field*: one heat solve gives
    the distance to the source everywhere, and descending it from a target follows that geodesic
    back. Every target shares the one solve, so a thousand paths to one source cost one system and a
    thousand independent walks -- which is why the signature is one source and many targets rather
    than a list of pairs.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    source
        ``(k,)`` source vertices. Several make the paths run to whichever is nearest,
        since the field is the distance to the *set*.
    targets
        ``(n_paths,)`` vertices to trace from.
    t
        Heat diffusion time, forwarded to
        [`heat_geodesic`][ordito.heat.heat_geodesic]. ``None`` uses its default.
    operators
        Prebuilt [`HeatOperators`][ordito.heat.HeatOperators] for this mesh, to spare the
        factorization when several sources are traced on one mesh.
    max_steps
        Cap on steps per path.

    Returns
    -------
    points, offsets
        ``(n_points,)`` and ``(n_paths + 1,)`` packed polylines and their CSR bounds, each running
        **from its target to the source**. Slice with [`split`][ordito.array.split] and measure
        with [`ordito.polyline.polyline_length`][ordito.polyline.polyline_length].

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``source`` and ``targets`` are not all on one device.

    Examples
    --------
    ```python
    source = od.array.arange(1, device=v.device)
    targets = od.array.arange(int(v.shape[0]), device=v.device)
    points, offsets = od.geodesic_walk.geodesic_path(v, f, source, targets)
    ```

    Notes
    -----
    **The path is as accurate as the field it descends, and no more.** The heat method's distance is
    first-order, so this is an *approximate* geodesic: measured against ``potpourri3d``'s edge-flip
    geodesics -- which are exact -- the length comes out a few per cent long, and the excess is the
    field's error rather than the walk's. It is never *shorter* than the true geodesic, which is the
    invariant worth testing against.

    A path that cannot reach the source stops early rather than failing; see
    [`descend_field`][ordito.geodesic_walk.descend_field] for the four ways that happens.

    See Also
    --------
    [`descend_field`][ordito.geodesic_walk.descend_field]
        The walk itself, for descending any other scalar field.
    [`ordito.heat.heat_geodesic`][ordito.heat.heat_geodesic]
        The field, when the distance is wanted and not the path.
    """
    require_same_device(
        vertices=vertices, faces=faces, source=source, targets=targets, operators=operators
    )
    distance = od.heat.heat_geodesic(vertices, faces, source, t, operators)
    return descend_field(vertices, faces, distance, targets, stop_value=0.0, max_steps=max_steps)


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
    if n_loops <= 0 or faces.size == 0 or max_iter <= 0:
        return loops, loop_offsets, 0

    n_vertices = vertices.size
    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    ring_halfedges, ring_offsets, is_boundary = (
        rings if rings is not None else vertex_one_rings(faces, twins=twins, n_vertices=n_vertices)
    )

    # Each sweep reads ``packed`` and writes a freshly sized buffer, so the caller's loops are never
    # written; a sweep that accepts nothing leaves them alone. The offsets are total-terminated,
    # which is what every kernel below reads ``loop_offsets[l + 1]`` against.
    packed = loops

    sweeps = 0
    # Requires TWO CONSECUTIVE sweeps to accept nothing, not "this sweep is odd and accepted
    # nothing" -- a rewrite can shift which loop positions land on even/odd parity, so a single
    # unchanged sweep says nothing about whether the *other* parity would still find something.
    consecutive_unchanged = 0
    position_loop = positions = counts = changed = arc_slot = arc_step = None
    for sweep in range(max_iter):
        n_positions = packed.size
        if n_positions == 0:
            break
        if consecutive_unchanged == 0:
            # An unchanged sweep leaves `packed` and `loop_offsets` as they were, so the owner
            # labels and the per-position scratch carry over; rebuild them only after a rewrite.
            position_loop = _launch.empty(n_positions, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_array.segment_owner_labels,
                dim=n_loops,
                inputs=[loop_offsets, position_loop],
                device=device,
            )
            # One zeroed buffer: a leading zero, the per-position counts, and the accepted-anything
            # flag after them. Scanning the counts in place makes the head the exclusive offsets
            # and entry ``n`` the total, so the total and the flag come back in one read. The flag
            # stays zero across unchanged sweeps -- that is what unchanged means -- so it is never
            # re-zeroed, and the leading zero is outside the scan.
            positions = _launch.zeros(n_positions + 2, dtype=wp.int32, device=device)
            counts = odt.as_dense(positions[1 : n_positions + 1])
            changed = odt.as_dense(positions[n_positions + 1 :])
            arc_slot = _launch.empty(n_positions, dtype=wp.int32, device=device)
            arc_step = _launch.empty(n_positions, dtype=wp.int32, device=device)
        assert position_loop is not None
        assert positions is not None
        assert counts is not None
        assert changed is not None
        assert arc_slot is not None
        assert arc_step is not None
        _launch.launch(
            kernel_geodesic_walk.shorten_loop_counts,
            dim=n_positions,
            inputs=[
                vertices,
                faces,
                ring_offsets,
                ring_halfedges,
                is_boundary,
                packed,
                position_loop,
                loop_offsets,
                sweep % 2,
                tolerance,
                counts,
                arc_slot,
                arc_step,
                changed,
            ],
            device=device,
        )
        sweeps = sweep + 1
        _launch.array_scan(counts, out_array=counts, inclusive=True)
        # One readback per sweep, and the only way to stop early: whether any replacement was
        # accepted is a device-side fact, and the alternative -- always running `max_iter` sweeps --
        # costs a full pass over every loop for each one that would have been skipped. The same
        # read carries the rewritten length that sizes the rewrite.
        total, n_changed = read_values(positions, n_positions, 2)
        if int(n_changed) == 0:
            consecutive_unchanged += 1
            if consecutive_unchanged >= 2:
                break  # both parities have now had a turn with nothing to do
            continue
        consecutive_unchanged = 0
        packed, loop_offsets = _rewrite_loops(
            faces,
            ring_offsets,
            ring_halfedges,
            packed,
            positions,
            int(total),
            arc_slot,
            arc_step,
            loop_offsets,
        )
        packed, loop_offsets = _compact_repeats(packed, loop_offsets, n_loops)

    return packed, loop_offsets, sweeps


def _rewrite_loops(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    packed: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    total: int,
    arc_slot: wp.array[wp.int32],
    arc_step: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Scatter each position's replacement into a freshly sized buffer, and remap the offsets.

    ``positions`` is the sweep's counts scanned in place behind a leading zero (its first
    ``len(packed) + 1`` entries are the exclusive offsets, entry ``len(packed)`` the ``total``).
    """
    device = packed.device
    rewritten = _launch.empty(max(total, 1), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_geodesic_walk.shorten_loop_write,
        dim=packed.size,
        inputs=[
            faces,
            ring_offsets,
            ring_halfedges,
            packed,
            arc_slot,
            arc_step,
            positions,
            rewritten,
        ],
        device=device,
    )
    # Each loop's old offset mapped through the position remap: a Python-scope gather.
    offsets = _launch.empty(loop_offsets.size, dtype=wp.int32, device=device)
    _launch.copy(offsets, positions[loop_offsets])
    return odt.as_dense(rewritten[:total]), offsets


def _compact_repeats(
    packed: wp.array[wp.int32], loop_offsets: wp.array[wp.int32], n_loops: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Drop positions repeating their cyclic predecessor, which a contracted spur leaves behind.

    Shares its resize-then-scatter-then-remap tail with
    [`_rewrite_loops`][ordito.geodesic_walk._rewrite_loops] (both scan counts in place behind a
    zeroed head, then allocate/launch/slice, then gather the loop offsets through the remap), and
    the two are kept separate rather than merged: this function has a legitimate optimization
    ``_rewrite_loops`` does not need -- when ``total == n_positions`` (nothing was dropped) it
    returns the original buffers unchanged instead of allocating and launching a no-op scatter. A
    shared helper would either drop that optimization or need an early-exit signal threaded back
    through it, which is more machinery than the handful of duplicated lines are worth
    (CLAUDE.md section 2.4: "merge on identity of meaning, not identity of tokens" -- two different
    amounts of control flow around one similarly-shaped call is not one function).
    """
    device = packed.device
    n_positions = packed.size
    if n_positions == 0:
        return packed, loop_offsets
    position_loop = _launch.empty(n_positions, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_array.segment_owner_labels,
        dim=n_loops,
        inputs=[loop_offsets, position_loop],
        device=device,
    )
    # The 0/1 counts are written behind a zeroed head and scanned in place, so the buffer is the
    # exclusive offsets with the total at its end: ``counts_to_offsets`` without the counts buffer.
    positions = _launch.zeros(n_positions + 1, dtype=wp.int32, device=device)
    counts = odt.as_dense(positions[1:])
    _launch.launch(
        kernel_geodesic_walk.distinct_from_predecessor,
        dim=n_positions,
        inputs=[packed, position_loop, loop_offsets, counts],
        device=device,
    )
    _launch.array_scan(counts, out_array=counts, inclusive=True)
    # Sizes the output: the one host readback of the compaction.
    total = int(read_scalar(positions))
    if total == n_positions:
        return packed, loop_offsets
    kept = _launch.empty(max(total, 1), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_geodesic_walk.compact_kept,
        dim=n_positions,
        inputs=[packed, positions, kept],
        device=device,
    )
    offsets = _launch.empty(loop_offsets.size, dtype=wp.int32, device=device)
    _launch.copy(offsets, positions[loop_offsets])
    return odt.as_dense(kept[:total]), offsets


def _length_epsilon(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Smallest crossing distance the walk will accept, as a fraction of the mean edge length.

    The walk needs *some* scale-aware floor: a ray starting at a vertex stands on two of its face's
    edges, and one starting on an edge stands on that edge, so a crossing at distance ~0 has to be
    rejected or the walk rotates on the spot. Tying it to the mean edge length keeps the behaviour
    invariant under a global rescale of the mesh.
    """
    return 1e-6 * od.edges.mean_edge_length(vertices, faces)


def _trace(
    kernel: odt.Kernel, inputs: Sequence[object], n_rays: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Run a tracing kernel twice: once to count each ray's points, once to write them.

    The walk is cheap arithmetic and its length is not known in advance, so counting first and
    sizing the output exactly beats reserving ``max_steps`` points per ray, which at the default cap
    would be tens of megabytes for a few thousand rays.
    """
    counts = _launch.empty(n_rays, dtype=wp.int32, device=device)
    no_offsets = _launch.empty(0, dtype=wp.int32, device=device)
    no_points = _launch.empty(0, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel, dim=n_rays, inputs=[*inputs, no_offsets, counts, no_points], device=device
    )

    # Host readback: only the device knows the walk's total length, and it sizes the point buffer.
    offsets, total = od.array.counts_to_offsets(counts)
    points = _launch.empty(total, dtype=wp.vec3, device=device)
    _launch.launch(kernel, dim=n_rays, inputs=[*inputs, offsets, counts, points], device=device)
    return points, offsets
