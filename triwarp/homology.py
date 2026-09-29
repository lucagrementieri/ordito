"""
Homology generators: the loops that make a surface's topology, found by tree-cotree.

On a closed genus-``g`` surface there are ``2 * g`` independent loops that cannot be contracted to a
point — two per handle, going *around* it and *through* it. They are what a cutting tool needs (cut
along them and the surface opens into a disk), and what tells a parametrizer where its seams have to
go.

The construction is Erickson & Whittlesey's tree-cotree decomposition, and it is a counting
argument as much as an algorithm: take a spanning tree of the vertex graph, then a spanning tree of
the *dual* graph that avoids the primal tree's edges, and every remaining edge closes exactly one
independent loop. Euler's formula fixes how many are left: ``E - (V - 1) - (F - 1) = 2 - chi``,
which is ``2 * g``.

Unlike the rest of this package there is no reference implementation to compare against —
potpourri3d does not expose geometry-central's homology code — so the tests check structural
invariants instead: the loop count against
[`euler_characteristic`][triwarp.measures.euler_characteristic], and that every loop is a closed
walk along real mesh edges that visits no vertex twice.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_same_device, run_device_loop
from triwarp.constants import INT32_MAX, TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import homology as kernel_homology
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels import scatter as kernel_scatter


def homology_generators(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, copy: bool = False
) -> list[wp.array[wp.int32]]:
    """
    Non-contractible loops of the surface, as ordered vertex-index cycles.

    Returns ``2 * g`` loops for a closed genus-``g`` surface and none for a sphere. Each loop is a
    closed walk along mesh edges: consecutive entries share an edge, and so do the last and first
    (the start vertex is not repeated, matching
    [`boundary_loops`][triwarp.boundary.boundary_loops]).

    The loops are a *basis*, not canonical: any generating set is as valid as any other, and this
    one falls out of the spanning trees the construction happens to build. They are also as long
    and as jagged as those trees, which is what
    [`shorten_loop`][triwarp.geodesic_walk.shorten_loop] is for. The basis is **reproducible**, and
    the same on every device: the primal tree resolves competing claims by lowest vertex index and
    the dual forest by lowest edge index, so neither depends on the order the threads ran in.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only the count is used.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be a closed, connected,
        edge-manifold surface: the counting argument this rests on assumes it.
    copy
        Return independent buffers instead of views into one packed allocation.

    Returns
    -------
    list[wp.array[wp.int32]]
        One ``wp.int32`` array of vertex indices per generator, on ``faces.device``. Empty for a
        sphere, and for a mesh with no faces.

    Raises
    ------
    ValueError
        If the mesh has a boundary, since a surface with boundary has a different homology basis
        (every boundary loop contributes one, and the tree-cotree count no longer applies), or if
        its referenced vertices are not all in one connected component, since the count is then a
        sum over components that a single pair of spanning trees does not produce.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`homology_generators_with_offsets`][triwarp.homology.homology_generators_with_offsets]
        The same loops, packed into one buffer with their offsets.
    [`shorten_loop`][triwarp.geodesic_walk.shorten_loop]
        Shortens these loops within their homotopy class, keeping them on mesh edges.
    [`euler_characteristic`][triwarp.measures.euler_characteristic]
        Fixes how many loops there are: ``2 * g == 2 - chi``.
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    loops, offsets = homology_generators_with_offsets(vertices, faces)
    return tw.array.split(loops, offsets, copy=copy)


def homology_generators_with_offsets(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Non-contractible loops of the surface, packed: every loop's vertices, then their offsets.

    The packed form of [`homology_generators`][triwarp.homology.homology_generators]: the same
    basis in the same order, concatenated into one buffer with no per-loop Python object.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only the count is used.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be a closed, connected,
        edge-manifold surface: the counting argument this rests on assumes it.

    Returns
    -------
    loops : wp.array[wp.int32]
        Every generator's vertex-index cycle, loop after loop, on ``faces.device``.
    offsets : wp.array[wp.int32]
        ``(2 * g + 1,)`` total-terminated offsets: loop ``i`` is ``loops[offsets[i] :
        offsets[i + 1]]``, and ``[0]`` for a sphere or a mesh with no faces.

    Raises
    ------
    ValueError
        If the mesh has a boundary, or its referenced vertices are not all in one connected
        component (see [`homology_generators`][triwarp.homology.homology_generators]).
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`homology_generators`][triwarp.homology.homology_generators]
        The same loops, one array each.
    [`shorten_loop_with_offsets`][triwarp.geodesic_walk.shorten_loop_with_offsets]
        Shortens them in the packed form.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_vertices == 0 or n_faces == 0:
        return _no_loops(device)

    # One grouping of the edge rows answers everything: ``inverse`` maps each face corner to its
    # unique edge, and one scatter over it fills both the per-edge face count and the two incident
    # faces. That is the whole dual graph, indexed by unique edge — which is why nothing here needs
    # to locate a shared edge's row afterwards.
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    n_edges = int(unique_edges.shape[0])
    if n_edges == 0:
        # A face carries three edges, so no edges means no faces: vacuously closed, nothing to span
        # and nothing left over.
        return _no_loops(device)
    edge_face_count = wp.zeros(n_edges, dtype=wp.int32, device=device)
    edge_faces = twt.empty_2d((n_edges, 2), wp.int32, device=device)
    wp.launch(
        kernel_scatter.scatter_edge_incidence,
        dim=int(inverse.shape[0]),
        inputs=[inverse, edge_face_count, edge_faces],
        device=device,
    )

    # The vertex graph the primal tree spans. Unsorted rows, which the level loop cannot see: it
    # resolves competing claims by lowest vertex index, not by column position.
    neighbors, offsets = tw.graph.edges_to_neighbor_lists(n_vertices, unique_edges, validate=False)
    parents, distances = _primal_spanning_tree(offsets, neighbors, unique_edges)

    # Both preconditions are counting questions, and both are answered on the device into one
    # three-slot buffer that is read back once: a guard that raises does not need to raise early,
    # and a separate readback apiece would serialise the pipeline twice more. The interior-edge
    # count rides in the candidate mask below, which already tests ``edge_face_count == 2``, so
    # neither guard costs a pass of its own.
    counts = wp.zeros(kernel_homology.COUNT_SIZE, dtype=wp.int32, device=device)
    wp.launch_tiled(
        kernel_homology.count_reached_and_referenced,
        dim=kernel_reduce.blocks_1d(n_vertices),
        inputs=[offsets, distances, counts],
        block_dim=TILE_1D,
        device=device,
    )

    # A generator is an edge in neither tree. ``candidate`` starts as the edges the cotree is
    # allowed to cross — interior, and not already claimed by the primal tree — and the forest pass
    # removes the ones it took, so the leftovers need no predicate of their own. It is issued
    # *before* the readback so both guards read one buffer; on the raising path that is one wasted
    # launch, and on every other path it is one fewer.
    candidate = wp.empty(n_edges, dtype=wp.bool, device=device)
    wp.launch_tiled(
        kernel_homology.dual_candidate_mask,
        # One *tile* per block, not one reduce-module chunk: this kernel writes a mask entry per
        # edge as well as folding the count, so its per-edge dimension has to stay in the grid.
        dim=(n_edges + TILE_1D - 1) // TILE_1D,
        inputs=[unique_edges, edge_face_count, parents, candidate, counts],
        block_dim=TILE_1D,
        device=device,
    )

    n_interior, n_reached, n_referenced = (int(value) for value in counts.numpy())
    if n_interior * 2 != 3 * n_faces:
        raise ValueError(
            "homology_generators requires a closed surface: this mesh has "
            f"{3 * n_faces - 2 * n_interior} boundary edge(s)."
        )
    if n_reached != n_referenced:
        raise ValueError(
            "homology_generators requires a connected surface: the traversal reached "
            f"{n_reached} of {n_referenced} referenced vertices."
        )

    _remove_dual_spanning_forest(candidate, edge_faces, n_faces)
    generator_edge_ids = tw.array.flatnonzero(candidate)
    if int(generator_edge_ids.shape[0]) == 0:
        return _no_loops(device)
    return _trace_generator_loops(generator_edge_ids, unique_edges, parents, distances)


def _primal_spanning_tree(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], unique_edges: twt.Array2dInt32
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Rooted spanning tree of the vertex graph as ``(parents, distances)``.

    A breadth-first level loop, pushed from the frontier, two launches per level and no host
    synchronization at all. ``parents`` holds
    [`NO_PARENT`][triwarp.kernels.homology.NO_PARENT] at the root and at every unreferenced vertex,
    which the tree never reaches; ``distances`` holds the depth, or ``-1`` for the same vertices.

    **The root is the lowest-indexed *referenced* vertex, read on the device rather than handed in**
    -- one host readback the call does not take. Rooting at vertex 0 unconditionally is the bug
    that avoids: on a mesh whose vertex 0 carries no edges the "tree" is a single isolated node,
    every primal edge becomes a generator, and the count comes back enormous rather than
    wrong-looking. [`edges_unique`][triwarp.edges.edges_unique] returns its rows lexicographically
    sorted, so the first entry of the flattened buffer *is* that vertex -- referenced by
    construction, and deterministic -- which is what lets the seed be
    [`scatter_index`][triwarp.kernels.scatter.scatter_index] at ``dim=1`` rather than a kernel of
    its own.

    The depth is kept although a spanning tree does not need it: it is what lets
    [`_trace_generator_loops`][triwarp.homology._trace_generator_loops] size a loop without walking
    it, and the level loop writes it either way.
    """
    device = unique_edges.device
    n_vertices = int(offsets.shape[0]) - 1
    parents = wp.full(n_vertices, INT32_MAX, dtype=wp.int32, device=device)
    distances = wp.full(n_vertices, -1, dtype=wp.int32, device=device)
    # Level 1 is the first to claim; the condition starts true because ``wp.capture_while`` reads
    # it before the first round; nothing claimed yet. Allocated holding those values rather than
    # zeroed and then assigned -- the zeroing is thrown away and the assign is a second upload.
    state = wp.array([1, 1, 0], dtype=wp.int32, device=device)
    wp.launch(
        kernel_scatter.scatter_index, dim=1, inputs=[unique_edges[0], distances], device=device
    )

    # A level claims at least one vertex or the loop is over, so ``n_vertices + 1`` bounds it --
    # cheap insurance against a captured loop hanging the device rather than answering wrongly.
    max_levels = wp.int32(n_vertices + 1)

    def level() -> None:
        wp.launch(
            kernel_homology.bfs_push_level,
            dim=n_vertices,
            inputs=[offsets, columns, state, parents, distances],
            device=device,
        )
        wp.launch(kernel_array.loop_advance, dim=1, inputs=[max_levels, state], device=device)

    run_device_loop(device, state[kernel_array.LOOP_CONDITION_VIEW], level)
    return parents, distances


def _remove_dual_spanning_forest(
    candidate: wp.array[wp.bool], edge_faces: twt.Array2dInt32, n_faces: int
) -> None:
    """
    Clear from ``candidate``, in place, a spanning forest of the dual graph over its edges.

    What is left set is exactly the generator edges: interior, in neither the primal tree nor the
    dual forest.

    Boruvka: each round hands every component its lowest-indexed incident candidate edge, accepts
    those edges and unions the components, so the component count at least halves per round and the
    loop finishes in ``O(log n_faces)`` of them. Four launches per round and no host
    synchronization; the round cap in the state word only bounds a loop the halving argument
    already bounds, since a captured loop that runs away hangs the device.

    A **forest** is all the decomposition needs — the cotree is read as a set, and the loop tracing
    walks the *primal* parents — so the dual side has no root and no parent pointers, and its shape
    is free to choose. Breadth-first is the expensive shape here: the dual graph restricted to
    non-primal-tree edges is already nearly a tree, so its diameter is enormous.
    """
    device = candidate.device
    n_candidates = int(candidate.shape[0])
    if n_candidates == 0 or n_faces == 0:
        return
    labels = tw.array.arange(n_faces, device=device)
    roots = wp.empty(n_faces, dtype=wp.int32, device=device)
    proposal = wp.empty(n_faces, dtype=wp.int32, device=device)
    # The condition starts true because ``wp.capture_while`` reads it before the first round;
    # allocated holding that seed rather than zeroed and then assigned.
    state = wp.array([0, 1, 0], dtype=wp.int32, device=device)
    # The component count at least halves per round, so ``bit_length`` -- ``ceil(log2)`` plus one --
    # caps a loop the halving argument already bounds; it can only be reached by a logic error.
    max_rounds = wp.int32(max(1, n_faces.bit_length()) + 1)

    def round_of_boruvka() -> None:
        wp.launch(
            kernel_homology.forest_round_setup,
            dim=n_faces,
            inputs=[labels, roots, proposal],
            device=device,
        )
        wp.launch(
            kernel_homology.forest_propose,
            dim=n_candidates,
            inputs=[candidate, edge_faces, roots, proposal],
            device=device,
        )
        wp.launch(
            kernel_homology.forest_link,
            dim=n_candidates,
            inputs=[candidate, edge_faces, roots, proposal, labels, state],
            device=device,
        )
        wp.launch(kernel_array.loop_advance, dim=1, inputs=[max_rounds, state], device=device)

    run_device_loop(device, state[kernel_array.LOOP_CONDITION_VIEW], round_of_boruvka)


def _trace_generator_loops(
    generator_edge_ids: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32,
    parents: wp.array[wp.int32],
    distances: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Close each generator edge into a loop through the primal tree, on the device.

    Two launches and a scan: one thread per generator finds the apex (the lowest common ancestor of
    the edge's endpoints) and the loop's length from the two depths, the lengths scan into offsets,
    and a second thread per generator fills its slice from both ends. The grid is ``2 * g`` wide and
    so tiny, but so is the work — the alternative is one Python walk per generator over a
    ``parents`` array read back in full.
    """
    device = unique_edges.device
    n_generators = int(generator_edge_ids.shape[0])
    apex = wp.empty(n_generators, dtype=wp.int32, device=device)
    lengths = wp.empty(n_generators, dtype=wp.int32, device=device)
    wp.launch(
        kernel_homology.generator_loop_lengths,
        dim=n_generators,
        inputs=[generator_edge_ids, unique_edges, parents, distances, apex, lengths],
        device=device,
    )
    # The total-terminated form: ``write_generator_loops`` reads ``offsets[g + 1]`` as its slice's
    # end, and the total is the packed length, so one call answers both.
    offsets, total = tw.array.counts_to_offsets(lengths)
    loops = wp.empty(total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_homology.write_generator_loops,
        dim=n_generators,
        inputs=[generator_edge_ids, unique_edges, parents, apex, offsets, loops],
        device=device,
    )
    return loops, offsets


def _no_loops(device: wp.context.Device) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """Return the empty packed basis: no loop vertices and ``[0]`` offsets."""
    return wp.empty(0, dtype=wp.int32, device=device), wp.zeros(1, dtype=wp.int32, device=device)
