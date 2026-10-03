"""
Mesh boundary edges and vertices.

A mesh edge lies on the boundary when it appears exactly once among all triangle edges.
Boundary detection radix-sorts every halfedge's undirected edge key with the halfedge's index as
the payload (the analog of ``trimesh.grouping.group_rows(require_count=1)``): a key occurring
exactly once is a boundary edge, and its halfedge names the face corner it came from.

Every loop-shaped entry point comes in two forms, and the pairing is the module's one convention
worth stating up front: a **list** form returning or taking one ``wp.array`` per loop
([`boundary_loops`][ordito.boundary.boundary_loops],
[`loop_perimeters`][ordito.boundary.loop_perimeters],
[`loop_directed_areas`][ordito.boundary.loop_directed_areas]) and a packed form over one
buffer plus total-terminated per-loop offsets
([`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets],
[`loop_perimeters_from_offsets`][ordito.boundary.loop_perimeters_from_offsets],
[`loop_directed_areas_from_offsets`][ordito.boundary.loop_directed_areas_from_offsets]). They
compute the same answer; the packed form is the one whose cost is independent of the loop *count*,
and it is what [`ordito.holes`][ordito.holes] carries its rims in from end to end. The list forms
pack and delegate, so there is one segmented launch behind each measure rather than one per form.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.constants import INDEX_RADIX_PAIR, INT32_MAX
from ordito.kernels import adjacency as kernel_adjacency
from ordito.kernels import array as kernel_array
from ordito.kernels import boundary as kernel_boundary
from ordito.kernels import graph as kernel_graph
from ordito.kernels import halfedge as kernel_halfedge
from ordito.kernels import scatter as kernel_scatter


def boundary_edges(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> odt.Array2dInt32:
    """
    Undirected boundary edges (each row min-first), without preserving orientation.

    An edge belongs to the boundary when it appears exactly once among the sorted edges of
    all triangles.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the device and the length are read. Every face
        index must be below ``n_vertices``.
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    odt.Array2dInt32
        ``(n_boundary, 2)`` sorted boundary edges on ``faces.device``. Empty ``(0, 2)`` when the
        mesh has no boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    return _boundary_edges_impl(faces, vertices.size, oriented=False)


def oriented_boundary_edges(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> odt.Array2dInt32:
    """
    Directed boundary edges, preserving the orientation from the face winding.

    Boundary edges are detected on the sorted edges (appearing exactly once), but the
    directed ``(i, j)`` pairs are returned.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the device and the length are read. Every face
        index must be below ``n_vertices``.
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    odt.Array2dInt32
        ``(n_boundary, 2)`` directed boundary edges on ``faces.device``. Empty
        ``(0, 2)`` when the mesh has no boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    return _boundary_edges_impl(faces, vertices.size, oriented=True)


def _boundary_edges_impl(
    faces: wp.array[wp.int32], n_vertices: int, *, oriented: bool
) -> odt.Array2dInt32:
    """
    Shared body of [`boundary_edges`][ordito.boundary.boundary_edges] and its oriented form.

    The vertex count is the key radix, so the sort orders only the bits a key can occupy.
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return odt.empty_2d((0, 2), wp.int32, device=faces.device)
    return _BoundaryHalfedges(faces, n_vertices or None).edges(sort_pair=not oriented)


def boundary_loops(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, copy: bool = False
) -> list[wp.array[wp.int32]]:
    """
    Ordered vertex-index loops along each mesh boundary.

    Each boundary loop is a simple cycle in the directed-boundary-edge successor graph (on a
    manifold boundary every boundary vertex has exactly one outgoing boundary edge). Loops are
    grouped via connected-component labeling, then each vertex's ordinal position within its
    loop is ranked by following the successor chain from itself to its loop's canonical start
    (the smallest vertex index in the loop) — entirely GPU-parallel, mirroring
    ``igl::boundary_loop`` (its first overload).

    All loops are found in one batched pass
    ([`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets]); this is
    [`split`][ordito.array.split] over its packed result.

    !!! note "The returned arrays are views"
        Each loop slices the single packed buffer ``boundary_loops_with_offsets`` produced, which
        costs no device memory and no launches. Two consequences: holding on to a single loop keeps
        the *whole* buffer alive, and writing into one loop writes into the shared allocation. Pass
        ``copy=True`` for independent buffers.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used, as the successor-array
        size and the edge-key radix, so every face index must be below it.
    faces
        ``(3 * n_faces,)`` face index buffer.
    copy
        Return independent buffers instead of views into the packed result.

    Returns
    -------
    list[wp.array[wp.int32]]
        One array per boundary loop, each holding the ordered vertex indices around that loop
        (not repeating the start vertex), on ``faces.device``. Empty list when the mesh has no
        boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    Notes
    -----
    A vertex where the rim meets itself (a non-manifold **pinch** point, as deleting two faces that
    share only a corner leaves) is walked by halfedge sector rather than by vertex: the loops are
    the boundary of the surface with that vertex split once per fan, so every boundary edge
    appears exactly once and every consecutive pair is a real boundary edge -- but a loop can pass
    through a pinch vertex more than once, where two holes touch there. That needs an
    edge-manifold mesh; on one that is not, the loops are bounded but unspecified.

    Orientability is *not* assumed. On a non-orientable surface the winding cannot orient the
    boundary globally -- at the seam two boundary edges leave the same vertex -- so the loops are
    recovered from the undirected edges instead, and their direction is then an arbitrary but
    reproducible choice rather than the face winding's. Only the direction is arbitrary; the loops
    themselves are exact. Costs one extra host readback to detect the case.

    See Also
    --------
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets]
        The packed form this splits: the same loops in one buffer plus offsets.
    [`longest_boundary_loop`][ordito.boundary.longest_boundary_loop]
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges]
    ``igl.boundary_loop_all``
    """
    require_same_device(vertices=vertices, faces=faces)
    flat_loops, offsets = boundary_loops_with_offsets(vertices, faces)
    return od.array.split(flat_loops, offsets, copy=copy)


def boundary_loops_with_offsets(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every boundary loop at once, packed into one buffer plus per-loop offsets.

    Same loops, same order, same contents as
    [`boundary_loops`][ordito.boundary.boundary_loops] — but with no per-loop Python and no
    per-loop allocation, which is the only form whose cost is independent of the loop *count*.
    Prefer it when a mesh has many small holes or when the loops feed straight into another
    batched kernel, as [`ordito.holes`][ordito.holes] does. The loops come out as
    [`successor_cycles`][ordito.graph.successor_cycles] orders a successor graph's cycles --
    each from its smallest vertex, the loops in ascending order of it -- ranked by pointer jumping
    specialised to a graph that is all cycles by construction.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used, as the successor-array
        size and the edge-key radix, so every face index must be below it.
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    flat_loops
        ``(n_loop_vertices,)`` concatenated ordered vertex indices of every loop, on
        ``faces.device``.
    offsets
        ``(n_loops + 1,)`` total-terminated offsets: loop ``i`` occupies
        ``flat_loops[offsets[i] : offsets[i + 1]]``, and ``offsets == [0]`` when the mesh has no
        boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    See Also
    --------
    [`boundary_loops`][ordito.boundary.boundary_loops]
        The list form: [`split`][ordito.array.split] over this result.
    [`longest_boundary_loop`][ordito.boundary.longest_boundary_loop]
    [`successor_cycles`][ordito.graph.successor_cycles]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty_packed(wp.int32, device)

    n_vertices = vertices.size
    # One boundary detection for every edge view below; ``boundary_edges`` /
    # ``oriented_boundary_edges`` / ``boundary_vertex_indices`` would each redo the key sort.
    boundary = _BoundaryHalfedges(faces, n_vertices)
    has_seam, has_pinch = _boundary_defects(boundary, n_vertices)
    if boundary.count() == 0:
        return _launch.empty_packed(wp.int32, device)
    if has_pinch:
        return _pinched_boundary_cycles(boundary)
    if has_seam:
        return _unoriented_boundary_cycles(boundary.edges(sort_pair=True), n_vertices)
    # With neither defect no boundary vertex has two boundary edges out or three incident, so the
    # directed rows are cycles over the boundary vertices -- plus, on a mesh that is not
    # edge-manifold, chains running into a vertex with no edge out, which the ranking drops.
    tails, next_node = boundary.successors(n_vertices)
    return _closed_successor_cycles(tails, next_node, kernel_boundary.CYCLE_NODES)


def _boundary_defects(boundary: _BoundaryHalfedges, n_vertices: int) -> tuple[bool, bool]:
    """
    Count the boundary edges and detect whether their directed rows have a seam or a pinch.

    Returns whether the rows have an orientation seam and whether they have a pinch; the count is
    left in ``boundary`` for
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] to read.

    They have neither on an orientable surface with a manifold boundary, which is what lets
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] rank them as the
    cycles of a successor graph over the vertices. The two defects break that differently and each
    has its own walk:

    - **A non-orientable seam.** The winding cannot be made consistent globally, so at the seam two
      boundary edges leave the same vertex and ``succ[tail] = head`` drops one silently. On the
      Moebius fixture, exactly one vertex of 78 has out-degree 2, and the walk that follows returns
      78 entries over only 40 distinct vertices. The boundary is still 2-regular, so
      [`_unoriented_boundary_cycles`][ordito.boundary._unoriented_boundary_cycles] recovers it
      exactly.
    - **A pinch point**, where two loops meet at one vertex, which then has four boundary
      incidences -- and, on a consistently wound surface, two out-edges as well, so a pinch raises
      *both* flags. No walk over vertices exists at all, but one over halfedges does:
      [`_pinched_boundary_cycles`][ordito.boundary._pinched_boundary_cycles].

    Conflating them is easy: an icosphere with every seventh face removed has pinch points but no
    orientability problem, and a gate reading only out-degree would send it down the undirected
    walk, which has nothing to offer a vertex of degree four.

    No pass and no readback of its own: the degree census rides in the launch that flags the
    boundary runs (``kernels/boundary.mark_boundary_runs``), and its two bits come back in the one
    readback that sizes the boundary. Neither flag needs a pass over the *vertices*: only a
    boundary vertex ever has a non-zero degree, and the thread that pushes one past its threshold
    learns so from the value its own ``wp.atomic_add`` returns.
    """
    boundary.count(census=n_vertices)
    return boundary.defects


def _pinched_boundary_cycles(
    boundary: _BoundaryHalfedges,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Boundary cycles of a rim that meets itself at a vertex: walk the boundary *halfedges* instead.

    At a pinch the vertex successor table has two writers and no answer, which is why the vertex
    walk came back with slots left at ``0``. A boundary halfedge has exactly one successor -- the
    next boundary halfedge around its tip within its own sector
    (``kernels/halfedge.next_boundary_halfedge``) -- so the halfedges form a true successor graph,
    and each cycle is read out as the origin vertex of each halfedge, which keeps the face winding's
    direction.

    So the cycles are the boundary of the surface with each pinch vertex split once per fan: every
    boundary edge appears exactly once, and a cycle can pass through a pinch vertex twice where two
    holes touch there (one fan hands the walk from the first rim to the second, the other hands it
    back). Only this branch builds a twin table (unvalidated: a mesh this function accepts need not
    be edge-manifold, and on one that is not, the walk is still
    bounded and in range, just not meaningful). There a rotation can dead-end on a halfedge that is
    not a boundary one, and the chain it leaves is dropped rather than ranked (see
    ``kernels/boundary.closed_cycle_windows``), so every returned pair is still a boundary edge.
    """
    faces = boundary.faces
    # ``halfedge_twins(faces, n_vertices=n_vertices, validate=False)``, read off the boundary
    # detection's own sort: the same keys against the same radix, sorted stably, so the pairing
    # kernel sees the identical sorted list and the mesh-sized key build and sort are not repeated.
    n = boundary.n
    twins = _launch.full(n, -1, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_halfedge.pair_sorted_halfedges,
        dim=n,
        inputs=[
            faces,
            odt.as_dense(boundary.keys[:n]),
            boundary.order,
            twins,
            _launch.zeros(2, dtype=wp.int32, device=faces.device),
        ],
        device=faces.device,
    )
    tails, next_node = boundary.successors(n, twins=twins)
    return _closed_successor_cycles(tails, next_node, kernel_boundary.CYCLE_HALFEDGES, faces)


def _unoriented_boundary_cycles(
    boundary_edges: odt.Array2dInt32, n_vertices: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Boundary cycles of a mesh whose winding cannot orient them: walk the undirected edges instead.

    A manifold boundary is 2-regular whether or not the surface is orientable, so the cycles exist
    even where a *consistent direction* for them does not. The walk runs on **darts**: dart ``2 * v
    + s`` means "at vertex ``v``, arrived from neighbour slot ``s``", and its successor leaves by
    the other slot. That is a genuine successor graph by construction -- every dart has exactly one
    out-edge -- so it ranks like the oriented walk, at twice the node count.

    Each undirected cycle therefore comes back **twice**, once per direction, over disjoint dart
    sets. The two mirrors share their lowest vertex ``v`` but start at darts ``2v`` and ``2v + 1``,
    so keeping the even-starting one picks exactly one per pair. With the neighbour slots sorted
    ascending, that direction is "leave the lowest vertex toward its larger neighbour" -- an
    arbitrary but *reproducible* choice, which is the honest answer when no winding defines one.

    On a Moebius band this returns the single cycle the surface actually has, with every consecutive
    pair a real boundary edge. Two references get it wrong in different ways and neither is worth
    matching: ``igl.boundary_loop_all`` cuts that cycle into open chains, each with one consecutive
    pair that is *not* a boundary edge, and a half-edge hole ring walks the band's *double* cover
    and reads twice the length.

    The mirror filter costs nothing of its own: a cycle's start is its smallest dart, so the
    ranking counts and places only the cycles whose start is even, and writes each dart out as its
    vertex.
    """
    device = boundary_edges.device
    # Allocated with the sentinel rather than filled after: a buffer whose initial value matters
    # is created holding it, so there is no window in which it holds garbage and no second
    # statement to keep in step with the first.
    neighbors = odt.as_array2d(
        _launch.full((n_vertices, 2), -1, dtype=wp.int32, device=device), wp.int32
    )
    slot_count = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boundary.scatter_boundary_neighbors,
        dim=int(boundary_edges.shape[0]),
        inputs=[boundary_edges, slot_count, neighbors],
        device=device,
    )

    # Each edge end is one dart, and every successor is a dart or, where a rim ends on a vertex with
    # one boundary edge, ``-1``: ``next_node`` is read only at darts and needs no fill.
    n_edges = int(boundary_edges.shape[0])
    tails = _launch.empty(2 * n_edges, dtype=wp.int32, device=device)
    next_node = _launch.empty(2 * n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boundary.dart_successors,
        dim=(n_edges, 2),
        inputs=[boundary_edges, neighbors, tails, next_node],
        device=device,
    )
    return _closed_successor_cycles(tails, next_node, kernel_boundary.CYCLE_DARTS)


def _closed_successor_cycles(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    mode: int,
    faces: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every cycle of a successor graph whose nodes are ``tails``, packed as ``successor_cycles`` does.

    The boundary walks' ranking: [`successor_cycles`][ordito.graph.successor_cycles]' answer --
    each cycle from its smallest node in edge direction, the cycles in ascending order of it --
    for a graph that is all cycles by construction, where each node is the tail of exactly one
    edge. ``next_node`` is the node-sized successor table, ``-1`` wherever a successor can fail to
    be a tail (the chains a mesh that is not edge-manifold leaves, which are dropped). ``mode``
    (``kernels/boundary.CYCLE_*``) says what each ranked node is written out as; ``faces`` is read
    by the halfedge walk.

    Because the nodes and their count are known and no two tails share a successor, the general
    function's node-list compaction, chain pass and component labelling all drop out: pointer
    jumping finds each node's cycle start (its smallest node) and its distance to it together,
    several pointers a launch (``kernels/graph.pointer_jump_schedule``), and one inclusive scan
    over the cycle starts places every cycle. One readback, of the cycle and node counts.
    """
    device = tails.device
    m = tails.size
    node_count = next_node.size
    hops, rounds = kernel_graph.pointer_jump_schedule(m)
    windows = _launch.empty(node_count, dtype=wp.vec3i, device=device)
    # Per cycle start, ``(1, length)``, accumulated by the last round and scanned in place.
    cycle_counts = _launch.zeros(node_count, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_boundary.closed_cycle_windows,
        dim=m,
        inputs=[tails, next_node, hops, mode, windows, cycle_counts if rounds == 1 else None],
        device=device,
    )
    if rounds > 1:
        spare = _launch.empty(node_count, dtype=wp.vec3i, device=device)
        window = hops
        for r in range(1, rounds):
            _launch.launch(
                kernel_boundary.closed_cycle_jump,
                dim=m,
                inputs=[
                    tails,
                    windows,
                    window,
                    hops,
                    mode,
                    spare,
                    cycle_counts if r == rounds - 1 else None,
                ],
                device=device,
            )
            windows, spare = spare, windows
            window *= hops
    _launch.array_scan(cycle_counts, out_array=cycle_counts, inclusive=True)
    # Sizes both outputs: the cycle count and the ranked node count, in one read.
    total = read_scalar(cycle_counts)
    n_cycles, n_ranked = int(total[0]), int(total[1])
    flat = _launch.empty(n_ranked, dtype=wp.int32, device=device)
    if n_cycles == 0:
        return flat, _launch.zeros(1, dtype=wp.int32, device=device)
    # Every entry, the leading zero and the total included, is written by the cycle it bounds.
    offsets = _launch.empty(n_cycles + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boundary.scatter_closed_cycles,
        dim=m,
        inputs=[tails, windows, cycle_counts, mode, faces, flat, offsets],
        device=device,
    )
    return flat, offsets


def loop_perimeters(
    vertices: wp.array[wp.vec3], loops: Sequence[wp.array[wp.int32]]
) -> wp.array[wp.float32]:
    """
    Perimeter of every closed loop, in one launch.

    Takes what [`boundary_loops`][ordito.boundary.boundary_loops] returns -- a list of vertex-index
    cycles whose last entry joins back to the first -- and measures them all together, so the cost
    is one launch and one packing pass rather than one call per loop. Equivalent to
    [`polyline_length`][ordito.polyline.polyline_length] with ``closed=True`` on each loop's
    gathered positions, and the segmented form is why the fill's ``preserve_largest_hole`` can rank
    every rim for the price of one readback.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    loops
        Closed vertex-index cycles. An empty sequence gives an empty result; a loop of fewer than
        two entries measures ``0``.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_loops,)`` one perimeter per loop, in the order given, on ``vertices.device``.

    Raises
    ------
    TypeError
        If any loop is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices`` and ``loops`` are not all on one device.

    See Also
    --------
    [`loop_directed_areas`][ordito.boundary.loop_directed_areas]
        The vector measure of the same loops: norm is the spanned area, direction is its normal.
    [`boundary_loops`][ordito.boundary.boundary_loops]
        Produces the loops.
    [`polyline_length`][ordito.polyline.polyline_length]
        The single-loop form, over positions rather than indices.
    """
    require_same_device(vertices=vertices, loops=loops)
    packed = _pack_loop_segments(vertices, loops)
    if packed is None:
        return _launch.empty(0, dtype=wp.float32, device=vertices.device)
    flat_loops, loop_id, offsets = packed
    return _launch_loop_measure(
        kernel_boundary.loop_perimeters, wp.float32, vertices, flat_loops, loop_id, offsets
    )


def loop_perimeters_from_offsets(
    vertices: wp.array[wp.vec3],
    flat_loops: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    *,
    loop_id: wp.array[wp.int32] | None = None,
    validate: bool = True,
) -> wp.array[wp.float32]:
    """
    Perimeter of every loop, taking the loops in the packed form rather than as a list.

    Same measure as [`loop_perimeters`][ordito.boundary.loop_perimeters] and the same launch; the
    two arguments after ``vertices`` are exactly what
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] returns, in that
    order. Reach for this whenever the loops are already packed, which is the form
    [`ordito.holes`][ordito.holes] carries them in throughout: the list form would have to be
    split back out and repacked to be measured, and the split is a per-loop Python object.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    flat_loops
        ``(n_loop_vertices,)`` concatenated ordered vertex indices of every loop.
    offsets
        ``(n_loops + 1,)`` total-terminated offsets: loop ``i`` is
        ``flat_loops[offsets[i] : offsets[i + 1]]``.
    loop_id
        ``(n_loop_vertices,)`` optional precomputed label saying which loop each packed position
        belongs to. Built here when ``None``, which costs an allocation and a launch -- **roughly
        doubling the call**, since the measure itself is one launch. Pass it when the caller
        already holds it, as [`ordito.holes`][ordito.holes] does.
    validate
        When ``True`` (default), check that ``offsets`` is non-decreasing from ``0`` to
        ``flat_loops``' length before launching -- a real cost (a host readback of ``offsets``),
        paid because a hand-built or stale pair otherwise drives the kernel's per-loop index past
        ``flat_loops``'s end with no exception raised. Pass ``False`` only when the pair is known
        correct by construction, as [`ordito.holes`][ordito.holes]'s internal packing already is.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_loops,)`` one perimeter per loop, in the packed order, on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``flat_loops``, ``offsets`` and ``loop_id`` are not all on one device.
    ValueError
        If ``validate`` and the packed pair is malformed -- see ``validate`` above.

    See Also
    --------
    [`loop_perimeters`][ordito.boundary.loop_perimeters]
        The list form, which packs and then launches the same kernel.
    [`loop_directed_areas_from_offsets`][ordito.boundary.loop_directed_areas_from_offsets]
        The vector measure of the same packed loops.
    """
    require_same_device(vertices=vertices, flat_loops=flat_loops, offsets=offsets, loop_id=loop_id)
    if validate:
        _validate_packed_loops(flat_loops, offsets, "loop_perimeters_from_offsets")
    return _launch_loop_measure(
        kernel_boundary.loop_perimeters,
        wp.float32,
        vertices,
        flat_loops,
        _loop_owner_labels(flat_loops, offsets) if loop_id is None else loop_id,
        offsets,
    )


def loop_directed_areas(
    vertices: wp.array[wp.vec3], loops: Sequence[wp.array[wp.int32]]
) -> wp.array[wp.vec3]:
    """
    Directed area vector of every loop, in one launch.

    Half the sum of ``p_i x p_{i+1}`` around each cycle. Its **norm** is the area of the planar
    polygon the loop spans and its **direction** is that polygon's normal, oriented by the loop's
    own winding -- so it answers "how big is this hole" and "which way does it face" at once, and
    the sign flips if the loop is reversed. A vector rather than a scalar for that reason: the area
    alone loses the orientation, which is what tells an outer boundary from an inner one.

    Origin-independent, because the cross products of a closed ring cancel any shift of the origin,
    so no centroid pass is needed. Exact for a planar loop; for a non-planar one it is the area of
    the loop's projection onto the plane normal to the result, which is the standard convention.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    loops
        Closed vertex-index cycles, as [`boundary_loops`][ordito.boundary.boundary_loops] returns.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_loops,)`` one directed area per loop, in the order given, on ``vertices.device``.

    Raises
    ------
    TypeError
        If any loop is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices`` and ``loops`` are not all on one device.

    See Also
    --------
    [`loop_perimeters`][ordito.boundary.loop_perimeters]
        The scalar measure of the same loops.
    [`polyline_normal`][ordito.polyline.polyline_normal]
        The single-loop direction, normalized and over positions.
    """
    require_same_device(vertices=vertices, loops=loops)
    packed = _pack_loop_segments(vertices, loops)
    if packed is None:
        return _launch.empty(0, dtype=wp.vec3, device=vertices.device)
    flat_loops, loop_id, offsets = packed
    return _launch_loop_measure(
        kernel_boundary.loop_directed_areas, wp.vec3, vertices, flat_loops, loop_id, offsets
    )


def loop_directed_areas_from_offsets(
    vertices: wp.array[wp.vec3],
    flat_loops: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    *,
    loop_id: wp.array[wp.int32] | None = None,
    validate: bool = True,
) -> wp.array[wp.vec3]:
    """
    Directed area vector of every loop, taking the loops in the packed form rather than as a list.

    The packed counterpart of
    [`loop_directed_areas`][ordito.boundary.loop_directed_areas], exactly as
    [`loop_perimeters_from_offsets`][ordito.boundary.loop_perimeters_from_offsets] is of
    [`loop_perimeters`][ordito.boundary.loop_perimeters]; the arguments after ``vertices`` are what
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] returns.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    flat_loops
        ``(n_loop_vertices,)`` concatenated ordered vertex indices of every loop.
    offsets
        ``(n_loops + 1,)`` total-terminated offsets: loop ``i`` is
        ``flat_loops[offsets[i] : offsets[i + 1]]``.
    loop_id
        ``(n_loop_vertices,)`` optional precomputed label saying which loop each packed position
        belongs to. Built here when ``None``, which costs an allocation and a launch -- **roughly
        doubling the call**, since the measure itself is one launch. Pass it when the caller
        already holds it, as [`ordito.holes`][ordito.holes] does.
    validate
        When ``True`` (default), check that ``offsets`` is non-decreasing from ``0`` to
        ``flat_loops``' length before launching -- see
        [`loop_perimeters_from_offsets`][ordito.boundary.loop_perimeters_from_offsets]'s docstring
        for the cost and the reasoning.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_loops,)`` one directed area per loop, in the packed order, on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``flat_loops``, ``offsets`` and ``loop_id`` are not all on one device.
    ValueError
        If ``validate`` and the packed pair is malformed -- see ``validate`` above.

    See Also
    --------
    [`loop_directed_areas`][ordito.boundary.loop_directed_areas]
        The list form, which packs and then launches the same kernel.
    [`loop_perimeters_from_offsets`][ordito.boundary.loop_perimeters_from_offsets]
        The scalar measure of the same packed loops.
    """
    require_same_device(vertices=vertices, flat_loops=flat_loops, offsets=offsets, loop_id=loop_id)
    if validate:
        _validate_packed_loops(flat_loops, offsets, "loop_directed_areas_from_offsets")
    return _launch_loop_measure(
        kernel_boundary.loop_directed_areas,
        wp.vec3,
        vertices,
        flat_loops,
        _loop_owner_labels(flat_loops, offsets) if loop_id is None else loop_id,
        offsets,
    )


def _validate_packed_loops(
    flat_loops: wp.array[wp.int32], offsets: wp.array[wp.int32], name: str
) -> None:
    """
    Raise if ``offsets`` does not describe a set of segments that tiles ``flat_loops``.

    Shared by [`loop_perimeters_from_offsets`][ordito.boundary.loop_perimeters_from_offsets] and
    [`loop_directed_areas_from_offsets`][ordito.boundary.loop_directed_areas_from_offsets] (its
    last caller), whose ``validate=True`` default calls this before launching. A malformed pair
    otherwise drives ``_launch_loop_measure``'s kernel to read ``flat_loops`` past its own end -- an
    out-of-bounds read with no exception, not merely a wrong answer -- so this reads ``offsets``
    back to the host and checks the one invariant that matters before that can happen.
    """
    total = flat_loops.size
    if int(offsets.ndim) != 1 or offsets.size == 0:
        raise ValueError(f"{name}: offsets must be a non-empty rank-1 total-terminated array")
    offsets_np = offsets.numpy().astype(np.int64)
    if (
        int(offsets_np[0]) != 0
        or int(offsets_np[-1]) != total
        or bool((offsets_np[1:] < offsets_np[:-1]).any())
    ):
        raise ValueError(
            f"{name}: offsets must run non-decreasing from 0 to the length of flat_loops "
            f"({total}); got {offsets_np.tolist()}"
        )


def _launch_loop_measure(
    kernel: odt.Kernel,
    dtype: type,
    vertices: wp.array[wp.vec3],
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
) -> odt.ArrayNd:
    """
    One segmented launch over every packed loop at once, shared by both measures and both forms.

    The two kernels differ only in what they accumulate -- an arc length into a ``float32`` or a
    cross-product sum into a ``vec3`` -- and take the identical four inputs, so the launch is
    written once here rather than four times above. ``wp.zeros`` rather than ``wp.empty``: both
    kernels accumulate into their output with an atomic add.
    """
    out = _launch.zeros(offsets.size - 1, dtype=dtype, device=vertices.device)
    _launch.launch(
        kernel,
        dim=flat_loops.size,
        inputs=[flat_loops, loop_id, offsets, vertices, out],
        device=vertices.device,
    )
    return out


def _loop_owner_labels(
    flat_loops: wp.array[wp.int32], offsets: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Which loop each packed position belongs to, built on the device from ``offsets`` alone.

    The list form builds the same labels on the host with ``numpy.repeat``, because it already has
    the sizes there; the packed form does not, and reading them back to reuse that path would cost
    a synchronization this saves. So the two forms build ``loop_id`` differently on purpose, and
    each is the cheaper one for the inputs it has.
    """
    n_loops = offsets.size - 1
    device = flat_loops.device
    loop_id = _launch.empty(flat_loops.size, dtype=wp.int32, device=device)
    if n_loops <= 0:
        return loop_id
    _launch.launch(
        kernel_array.segment_owner_labels, dim=n_loops, inputs=[offsets, loop_id], device=device
    )
    return loop_id


def _pack_loop_segments(
    vertices: wp.array[wp.vec3], loops: Sequence[wp.array[wp.int32]]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]] | None:
    """
    Pack a list of cycles into the three arrays a segmented per-loop kernel needs.

    ``loop_id`` inverts ``offsets`` so a ``dim=total`` launch finds its own loop without a search,
    which is what lets both measures above run in one launch over every loop at once. ``None`` when
    there is nothing to measure.
    """
    # The NumPy here is host-side *metadata* -- one integer per loop, read off each loop's own
    # ``shape`` -- so it is the sanctioned kind and not a readback: nothing crosses the bus except
    # the uploads at the end, and there is no device buffer to reduce.
    #
    # The sizes are already on the host, so ``numpy.repeat`` builds ``loop_id`` here where the
    # packed form, which has only device offsets, launches ``kernels/array.py``'s
    # ``segment_owner_labels``. This is also why the packed form takes ``loop_id`` as a keyword:
    # the two forms build it from different inputs and each is the cheaper one for what it holds.
    device = vertices.device
    loops = list(loops)
    for loop in loops:
        odt.ensure_ndim(loop, 1, dtype=wp.int32)
    if not loops or all(loop.size == 0 for loop in loops):
        return None
    # ``copy=False``: nothing below writes into ``flat_loops``, and a caller's loops usually
    # come straight from ``boundary_loops``, which already sliced them out of one packed
    # buffer -- so the pack is free instead of one ``wp.copy`` per rim.
    flat_loops, offsets = od.array.pack_1d_arrays(loops, copy=False)
    sizes_np = np.array([loop.size for loop in loops], dtype=np.int32)
    loop_id = _launch.array(
        np.repeat(np.arange(len(loops), dtype=np.int32), sizes_np), dtype=wp.int32, device=device
    )
    return flat_loops, loop_id, offsets


def longest_boundary_loop(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Ordered vertex-index loop along the longest mesh boundary.

    The name carries the *longest*, because the plural
    [`boundary_loops`][ordito.boundary.boundary_loops] returns every one and two public names
    differing by a single character are a defect even when both are correct.

    Parameters
    ----------
    vertices, faces
        Forwarded to [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets].

    Returns
    -------
    wp.array[wp.int32]
        ``(m,)`` ordered vertex indices around the longest boundary loop, ``m`` its length, on
        ``faces.device``. An independent buffer, not a view into the packed result. Empty when the
        mesh has no boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    See Also
    --------
    [`boundary_loops`][ordito.boundary.boundary_loops]
        Every loop, not only the longest.
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets]
    ``igl.boundary_loop``
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    flat_loops, offsets = boundary_loops_with_offsets(vertices, faces)
    n_loops = offsets.size - 1
    if n_loops == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)
    # The sizes are the offsets' differences on the device, so the winner is an argmax there
    # rather than a Python scan: unpacking the loops first would read the offsets back, build one
    # array view per loop, and then recover the lengths from those views -- a cost linear in the
    # rim count for an answer that is one loop. ``-1`` is below every packed key, so the reduction
    # needs no separate seeding pass.
    best = _launch.full(1, -1, dtype=wp.int64, device=device)
    _launch.launch(
        kernel_boundary.longest_loop_key, dim=n_loops, inputs=[offsets, best], device=device
    )
    # One readback, because the key carries the winner's start in its low half as well as its
    # length in its high half -- see the kernel.
    key = int(read_scalar(best, 0))
    start = INT32_MAX - (key & 0xFFFFFFFF)
    size = key >> 32
    # Only the winner is materialized: the rest of the packed buffer is never copied.
    return _launch.clone(odt.as_dense(flat_loops[start : start + size]))


def boundary_vertex_indices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Sorted unique vertex indices lying on the mesh boundary.

    A vertex belongs to the boundary when it belongs to at least one boundary edge.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (every boundary vertex index is
        below it).
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    wp.array[wp.int32]
        ``(n_boundary_vertices,)`` sorted unique boundary vertex indices on ``faces.device``.
        Empty when the mesh has no boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    if faces.size // 3 == 0:
        return _launch.empty(0, dtype=wp.int32, device=faces.device)
    # The endpoints are vertex indices below ``len(vertices)``, so a per-vertex flag array and its
    # scan give the sorted unique set directly -- no edge list and no ``unique_1d``.
    n_vertices = vertices.size
    return _BoundaryHalfedges(faces, n_vertices).vertex_indices(n_vertices)


def boundary_vertices(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    """
    Coordinates of the vertices lying on the mesh boundary.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_boundary_vertices,)`` boundary vertex positions on ``vertices.device``,
        ordered by ascending vertex index. Empty when the mesh has no boundary.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    indices = boundary_vertex_indices(vertices, faces)
    return od.array.gather(vertices, indices)


def ears(faces: wp.array[wp.int32]) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Find ear faces (triangles with exactly two boundary edges).

    For each ear face, ``ear_opp`` is the local index of the non-boundary edge, where local edge
    ``i`` is ``(faces[f, i], faces[f, (i + 1) % 3])``.

    !!! note "``igl::ears`` numbers the edge differently"

        The same quantity in ``igl.ears`` is indexed *opposite-vertex* style -- edge ``i`` is the
        one facing vertex ``i``, ``(faces[f, (i + 1) % 3], faces[f, (i + 2) % 3])`` -- because it
        reads its mask from ``igl::on_boundary``, whose columns are documented that way. The two
        agree on *which* faces are ears and differ on the index by a cyclic shift:
        ``ordito_opp == (igl_opp + 1) % 3``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` face index buffer.

    Returns
    -------
    ear : wp.array[wp.int32]
        ``(n_ears,)`` face indices of ear triangles on ``faces.device``, ascending. Empty when no
        ears exist.
    ear_opp : wp.array[wp.int32]
        ``(n_ears,)`` local edge index of the interior edge for each ear face, same length as
        ``ear``.

    See Also
    --------
    ``igl.ears``
    """
    n_faces = faces.size // 3
    device = faces.device
    empty = _launch.empty(0, dtype=wp.int32, device=device)
    if n_faces == 0:
        return empty, empty

    # The mask is read straight off the sorted keys: no scan and no readback.
    edge_boundary = _BoundaryHalfedges(faces).halfedge_mask()

    # Flag, scan in place, and emit at the scan's steps: ascending face order, one readback.
    inclusive = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boundary.mark_ears, dim=n_faces, inputs=[edge_boundary, inclusive], device=device
    )
    _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # Sizes the output: the one host readback.
    n_ears = int(read_scalar(inclusive))
    if n_ears == 0:
        return empty, empty
    ear = _launch.empty(n_ears, dtype=wp.int32, device=device)
    ear_opp = _launch.empty(n_ears, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boundary.emit_ears,
        dim=n_faces,
        inputs=[edge_boundary, inclusive, ear, ear_opp],
        device=device,
    )
    return ear, ear_opp


class _BoundaryHalfedges:
    """
    The boundary halfedges of a mesh: one radix sort of every halfedge's undirected edge key.

    A boundary edge is a key occurring exactly once, and the sort's payload is its halfedge index,
    which is row ``h`` of [`faces_to_edges`][ordito.edges.faces_to_edges] -- so every view the
    module needs (the undirected rows, the directed rows, the halfedge indices, the vertices, a
    per-halfedge mask) is read off ``faces`` and the sorted keys without an edge table. The keys
    pack against [`constants.INDEX_RADIX_PAIR`][ordito.constants.INDEX_RADIX_PAIR], which orders
    them exactly as the vertex count would and needs no bound -- or, where the caller already
    relies on every index being below ``n_vertices``, against that count, so the sort orders only
    the bits a key can occupy. Either radix gives the same key order.
    """

    def __init__(self, faces: wp.array[wp.int32], n_vertices: int | None = None) -> None:
        device = faces.device
        n = faces.size // 3 * 3
        self.faces = faces
        self.n = n
        self.keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
        self.order = _launch.empty(2 * n, dtype=wp.int32, device=device)
        radix = n_vertices if n_vertices else INDEX_RADIX_PAIR
        base = wp.uint64(radix)
        # The sort's double-width buffers are kept whole: every reader here passes ``n``, so the
        # trimmed views ``adjacency.sorted_face_edge_keys`` hands back would be pure host cost.
        _launch.launch(
            kernel_adjacency.face_edge_keys_and_order,
            dim=n // 3,
            inputs=[faces, base, self.keys, self.order],
            device=device,
        )
        _launch.radix_sort_pairs(
            self.keys,
            self.order,
            count=n,
            end_bit=min(64, max(1, (radix * radix - 1).bit_length())),
        )
        self._inclusive: wp.array[wp.int32] | None = None
        self._count = 0
        self.defects = (False, False)

    def count(self, census: int | None = None) -> int:
        """
        Return the boundary edge count; the first call scans and reads the total back.

        ``census`` is ``boundary_loops_with_offsets``'s vertex count. Given on the first call, the
        degree census runs in the flagging launch and its seam and pinch bits come back with the
        total, into ``defects``.
        """
        if self._inclusive is None:
            device = self.faces.device
            n = self.n
            degrees: odt.Array2dInt32 | None = None
            if census is None:
                flags = _launch.empty(n, dtype=wp.int32, device=device)
            else:
                n_vertices = census
                # One zeroed buffer: the scanned flags, the census' two defect bits beyond them,
                # and the ``(n_vertices, 2)`` degree table after those -- one allocation, and the
                # total and both bits adjacent for the single readback.
                flags = _launch.zeros(n + 2 + 2 * n_vertices, dtype=wp.int32, device=device)
                degrees = odt.as_array2d(
                    odt.as_dense(flags[n + 2 :]).reshape((n_vertices, 2)), wp.int32
                )
            _launch.launch(
                kernel_boundary.mark_boundary_runs,
                dim=n,
                inputs=[self.keys, self.order, wp.int32(n), self.faces, degrees, flags],
                device=device,
            )
            inclusive = flags if census is None else odt.as_dense(flags[:n])
            _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
            self._inclusive = inclusive
            # Sizes every output below: the one host readback of the boundary detection, which
            # carries the census' two bits alongside the total when there is one.
            if census is None:
                self._count = int(read_scalar(inclusive))
            else:
                total, seam, pinch = read_values(flags, n - 1, 3)
                self._count = total
                self.defects = (bool(seam), bool(pinch))
        return self._count

    def edges(self, *, sort_pair: bool) -> odt.Array2dInt32:
        """Emit the boundary edge rows in ascending key order (ascending pairs if ``sort_pair``)."""
        device = self.faces.device
        k = self.count()
        out_edges = odt.empty_2d((k, 2), wp.int32, device=device)
        if k > 0:
            _launch.launch(
                kernel_boundary.emit_boundary_edges,
                dim=self.n,
                inputs=[self._inclusive, self.order, self.faces, sort_pair, out_edges],
                device=device,
            )
        return out_edges

    def successors(
        self, node_count: int, *, twins: wp.array[wp.int32] | None = None
    ) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        """
        Return the boundary as a successor graph: its nodes, and the ``node_count``-sized table.

        The nodes are the directed rows' tail vertices, or with ``twins`` the boundary halfedges;
        see
        ``kernels/boundary.emit_boundary_successors``. The table is ``-1`` off the nodes, which is
        where a chain on a mesh that is not edge-manifold ends.
        """
        device = self.faces.device
        k = self.count()
        tails = _launch.empty(k, dtype=wp.int32, device=device)
        # ``-1`` off the nodes: the ranking reads it to tell where a chain ends.
        next_node = _launch.full(node_count, -1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_boundary.emit_boundary_successors,
            dim=self.n,
            inputs=[self._inclusive, self.order, self.faces, twins, tails, next_node],
            device=device,
        )
        return tails, next_node

    def vertex_indices(self, n_vertices: int) -> wp.array[wp.int32]:
        """Sorted unique boundary vertex indices, from a scan of per-vertex flags."""
        device = self.faces.device
        flags = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
        if n_vertices == 0:
            return flags
        _launch.launch(
            kernel_boundary.mark_boundary_vertices,
            dim=self.n,
            inputs=[self.keys, self.order, wp.int32(self.n), self.faces, flags],
            device=device,
        )
        _launch.array_scan(flags, out_array=flags, inclusive=True)
        # Sizes the output: the one host readback of this path.
        n_out = int(read_scalar(flags))
        out = _launch.empty(n_out, dtype=wp.int32, device=device)
        if n_out > 0:
            _launch.launch(
                kernel_scatter.scatter_index_where_scanned,
                dim=n_vertices,
                inputs=[flags, out],
                device=device,
            )
        return out

    def halfedge_mask(self) -> wp.array[wp.bool]:
        """Per halfedge, whether its edge is a boundary edge; no scan and no readback."""
        device = self.faces.device
        mask = _launch.empty(self.n, dtype=wp.bool, device=device)
        _launch.launch(
            kernel_boundary.boundary_halfedge_mask,
            dim=self.n,
            inputs=[self.keys, self.order, wp.int32(self.n), mask],
            device=device,
        )
        return mask
