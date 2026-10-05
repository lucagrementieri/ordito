"""
Kernels for [`ordito.homology`][ordito.homology]: one fused tree-cotree pass.

The module builds, in order, a rooted spanning tree of the vertex graph, a spanning **forest** of
the dual graph over the edges that tree left alone, and then traces one loop per leftover edge.
Every stage here is shaped by the same two facts about this call: it is **launch-bound**, and its
only consumer reads ``parents`` and ``distances`` and nothing else.

**The primal tree is a two-kernel level loop, not an order-exact traversal.** A breadth-first
search that has to reproduce ``scipy.sparse.csgraph.breadth_first_order`` spends most of a level
compacting its claims into a FIFO in ``(rank, ascending node id)`` order -- a tiled block scan plus
a serial advance, more than a third of the level's cost against the claim/commit pair that actually
builds the tree. Nothing downstream of a spanning tree reads a discovery order, so the scan is gone
and the level is ``bfs_push_level`` plus a one-thread
[`loop_advance`][ordito.kernels.array.loop_advance].

**What replaces the order as the tie-break is ``wp.atomic_min`` on the parent**, and it buys two
things beyond the launch. The tree stays deterministic -- ``parents[w]`` is the lowest-indexed
frontier vertex adjacent to ``w``, whatever order the threads ran in -- and the *adjacency* no
longer has to be column-sorted for that to hold, which is what lets the wrapper reach for
[`edges_to_neighbor_lists`][ordito.graph.edges_to_neighbor_lists] -- a counting pass and a cursor
scatter -- instead of routing 2 * n_edges triplets through ``warp.sparse.bsr_from_triplets``' radix
sort.

``bfs_push_level`` races on ``out_distances`` deliberately: two frontier vertices claiming the same
neighbour both store the same ``level``, so the write is idempotent, and the ``seen == level`` arm
of the visit test is what keeps the second claimer from skipping the ``wp.atomic_min`` it still
has to take part in. Dropping that arm loses the minimum and makes the tree thread-order dependent.

**The dual side is a Boruvka forest, and a forest is all that is needed**: ``homology_generators``
reads the cotree as a *set* -- the generators are the edges in neither tree -- and the loop tracing
walks the **primal** parents. So the dual side has no root and no parent pointers, and its shape is
free to choose. Breadth-first is the one shape that is expensive here: the dual graph restricted to
non-primal-tree edges is *already nearly a tree*, so its diameter is enormous -- on the benchmark
meshes a dual traversal runs several times the primal's level count. Boruvka needs ``O(log F)``
rounds instead. Each round gives every component its minimum-index incident candidate edge, accepts
those edges and unions the components; the union-find core is
[`find_representative`][ordito.kernels.algorithms.connected_components.find_representative] and
[`ecl_hook_edge`][ordito.kernels.algorithms.connected_components.ecl_hook_edge], imported rather
than re-derived, so the forest and ``graph.connected_component_labels`` share one implementation of
the pointer-jumping find and of the CAS hook.

**Why the accepted set is a forest.** Edge *indices* are distinct, so "minimum incident index" is a
strict total order on the candidate edges, and the classical Boruvka argument applies: a cycle among
the accepted edges would need a strictly decreasing cycle of minima. The round reads its components
from an immutable ``roots`` snapshot taken before any union, which is what makes that argument exact
here -- deciding against a ``label`` array that other threads are mutating would let one component
accept two edges in a round and could close a cycle.

**Both loops carry their own condition so the wrapper can capture them**, and both do it through
the package's shared round-loop slot table rather than one of their own -- ``array.LOOP_ROUND`` /
``LOOP_CONDITION`` / ``LOOP_PROGRESS``, closed by ``array.loop_advance`` and driven by
``_device.run_device_loop``. A per-round host readback drains the pipeline the round's launches
just filled, and there were about thirty of them across the two loops; the wrapper now synchronizes
three times in the whole call. Each loop passes its own round cap: the merge count and the claim
flag are each a sound termination argument on their own, but a captured loop that fails one hangs
the device rather than returning a wrong answer, so the cap is cheap insurance rather than a
schedule.

**The tracing is two kernels and a scan rather than a Python loop per generator.** Each
generator edge ``(a, b)`` closes into a loop through the tree as ``a -> lca(a, b) -> b``, and its
length is ``depth(a) + depth(b) - 2 * depth(lca) + 1`` -- so ``generator_loop_counts`` finds the
apex, the wrapper scans the counts into ranks and offsets, and ``write_generator_loops`` fills
each loop's slice from both ends at once. ``distances`` is the depth array the level loop already
wrote, which is why dropping the *order* from the traversal and keeping the *distances* was the
right half to cut: the tracer needs the depths to size a loop without walking it.
"""

import warp as wp

from ordito.constants import INT32_MAX, TILE_1D
from ordito.kernels.algorithms.connected_components import ecl_hook_edge, find_representative
from ordito.kernels.array import LOOP_PROGRESS, LOOP_ROUND
from ordito.kernels.reduce import block_chunk, block_chunk_1d, commit_block_sum, commit_block_total

wp.set_module_options({"enable_backward": False})

# One past the largest edge index any proposal can hold, so ``wp.atomic_min`` starts empty. The
# candidate count is bounded by the unique-edge count, which is well inside int32.
FOREST_NO_PROPOSAL = wp.constant(wp.int32(INT32_MAX))

# ``parents`` is seeded with this and narrowed by ``wp.atomic_min``, so "no parent yet" has to be
# larger than every vertex index rather than the ``-1`` a sequential traversal would write. It
# never collides with a real index, which is what lets ``dual_candidate_mask`` test parenthood
# without a special case for the root.
NO_PARENT = wp.constant(wp.int32(INT32_MAX))

# Slots of the one buffer that carries every host-visible count out of the decomposition, read back
# once. Slot 0 is filled over the edges and slots 1-2 over the vertices, by two different kernels,
# which is why they share a buffer rather than each owning one: the readback is the cost, not the
# four bytes.
COUNT_INTERIOR_EDGES = 0
COUNT_REACHED = 1
COUNT_REFERENCED = 2
COUNT_SIZE = 3


@wp.kernel
def bfs_push_level(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_parents: wp.array[wp.int32],
    out_distances: wp.array[wp.int32],
) -> None:
    # One breadth-first level, pushed from the frontier. Launched at ``dim=n_vertices`` and
    # early-exiting off ``out_distances`` rather than reading a compacted frontier: the compaction
    # is a tiled scan per level and the early exit is one coalesced load, and the launch is the same
    # width either way.
    #
    # ``state`` is the shared round-loop word of ``kernels/array.py``: this reads the level to claim
    # from ``LOOP_ROUND`` and raises ``LOOP_PROGRESS``, which the closing ``array.loop_advance``
    # turns into the next round's condition. A plain store, not an atomic -- one address, one value.
    #
    # The visit test admits ``seen == level`` as well as an unvisited vertex, and that arm is
    # load-bearing rather than defensive. Two frontier vertices may reach the same neighbour in one
    # level; whichever gets there first stores ``level``, and without this arm the other would read
    # a visited vertex and skip the ``wp.atomic_min`` -- leaving ``parents`` dependent on which
    # thread ran first. Both stores write the same value, so the race on ``out_distances`` itself is
    # idempotent.
    v = wp.int32(wp.tid())
    level = state[LOOP_ROUND]
    if out_distances[v] != level - 1:
        return
    for k in range(offsets[v], offsets[v + 1]):
        w = columns[k]
        seen = out_distances[w]
        if seen < 0 or seen == level:
            out_distances[w] = level
            wp.atomic_min(out_parents, w, v)
            state[LOOP_PROGRESS] = 1


@wp.func
def count_reached_and_referenced(
    offsets: wp.array[wp.int32],
    distances: wp.array[wp.int32],
    chunk: wp.int32,
    lane: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    # The connectivity guard's two numbers, folded per block into the same buffer the interior-edge
    # count went to. A vertex is referenced exactly when its adjacency row is non-empty and reached
    # exactly when the level loop gave it a depth, so the two agree if and only if the referenced
    # vertices form one component. Unreferenced vertices are deliberately not counted: they carry no
    # edges, so they cannot change the generator count.
    offset, n_rows = block_chunk_1d(distances.shape[0], chunk)
    if n_rows <= 0:
        return

    reached = wp.int32(0)
    referenced = wp.int32(0)
    for k in range(lane, n_rows, wp.block_dim()):
        v = offset + k
        if distances[v] >= 0:
            reached = reached + 1
        if offsets[v + 1] > offsets[v]:
            referenced = referenced + 1

    # Block-collective, so it runs outside the ``lane == 0`` guard.
    # ``COUNT_REFERENCED`` is the slot after ``COUNT_REACHED``.
    commit_block_sum(lane, wp.vec2i(reached, referenced), out_counts, COUNT_REACHED)


@wp.func
def dual_candidate_mask(
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    chunk: wp.int32,
    lane: wp.int32,
    out_candidate: wp.array[wp.bool],
    out_counts: wp.array[wp.int32],
) -> None:
    # Whether the cotree may cross this dual edge: an interior edge (exactly two incident faces)
    # that the primal tree did not already claim. Non-manifold edges count higher than 2 and are
    # excluded, which is what the exactly-2 row grouping behind ``adjacency.face_adjacency`` does
    # with them too -- and it is also what keeps the Boruvka round away from ``edge_faces``' second
    # column on a boundary edge, which ``scatter_edge_incidence`` leaves unwritten.
    #
    # The primal-tree test and the candidate test were two kernels and an intermediate mask. An
    # undirected edge belongs to a rooted spanning tree exactly when one endpoint is the other's
    # parent, and ``NO_PARENT`` matches neither endpoint, so the root needs no special case.
    #
    # **The closed-surface guard's interior-edge count rides along**, because ``edge_face_count[e]
    # == 2`` is already this kernel's own predicate -- so the count is the fold of a value the
    # thread computed anyway, and the guard needs no pass of its own.
    #
    # A block fold rather than a conditional ``wp.atomic_add`` per edge, because on a closed mesh
    # *every* edge is interior -- so the conditional atomic is the unconditional one and it
    # serializes the whole launch on one address.
    #
    # **One tile per block, not ``ITEMS_PER_BLOCK_1D``**, and that is the whole difference between
    # this being free and being a regression: the reduce module's fold width gives each lane a dozen
    # or so elements, which is right for a kernel whose *only* output is the reduction, and wrong
    # here because this one also writes a mask entry per edge -- the wide fold collapses the grid to
    # under one block per SM. Section 2.3's occupancy rule: a kernel that already has a per-element
    # dimension must not collapse it into ``block_dim`` lanes. One atomic per tile is still one per
    # block, which is the shape section 13.2 asks for.
    offset, n_rows = block_chunk(unique_edges.shape[0], chunk, TILE_1D)
    if n_rows <= 0:
        return

    interior = wp.int32(0)
    for k in range(lane, n_rows, wp.block_dim()):
        e = offset + k
        a = unique_edges[e, 0]
        b = unique_edges[e, 1]
        is_interior = edge_face_count[e] == 2
        in_primal_tree = parents[b] == a or parents[a] == b
        out_candidate[e] = is_interior and not in_primal_tree
        if is_interior:
            interior = interior + 1

    # Block-collective, so it runs outside the ``lane == 0`` guard.
    commit_block_total(lane, interior, out_counts, COUNT_INTERIOR_EDGES)


@wp.kernel
def guards_and_dual_candidates(
    offsets: wp.array[wp.int32],
    distances: wp.array[wp.int32],
    vertex_blocks: wp.int32,
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    out_candidate: wp.array[wp.bool],
    out_counts: wp.array[wp.int32],
) -> None:
    # ``count_reached_and_referenced`` and ``dual_candidate_mask`` in one grid: the first
    # ``vertex_blocks`` blocks take the vertex fold at its reduce-module chunk width, the rest one
    # edge tile each -- the two widths each body needs (see ``dual_candidate_mask`` for why its own
    # must stay narrow). The branch is per block, so every lane of a block takes the same body and
    # its block-collective fold. They fold into different slots of ``out_counts``.
    chunk, lane = wp.tid()
    if chunk < vertex_blocks:
        count_reached_and_referenced(offsets, distances, chunk, lane, out_counts)
    else:
        dual_candidate_mask(
            unique_edges,
            edge_face_count,
            parents,
            chunk - vertex_blocks,
            lane,
            out_candidate,
            out_counts,
        )


@wp.kernel
def forest_round_setup(
    labels: wp.array[wp.int32], out_roots: wp.array[wp.int32], out_proposal: wp.array[wp.int32]
) -> None:
    # Open a Boruvka round: freeze this round's component of every node and empty the proposal
    # slots. ``find_representative`` path-compresses ``labels`` as it goes, so the snapshot doubles
    # as the round's flatten pass.
    #
    # The snapshot line alone is
    # [`ecl_flatten`][ordito.kernels.algorithms.connected_components.ecl_flatten] verbatim; what
    # this adds is the proposal fill, which used to be a host-side ``fill_`` call per round and is a
    # store here because the kernel already runs at ``dim=n_faces`` and touches every element of
    # ``out_proposal`` anyway. The round's merge flag is *not* reset here -- ``array.loop_advance``
    # clears it when it closes the round, which is the only point at which the round's claim has
    # already been tested.
    f = wp.int32(wp.tid())
    out_roots[f] = find_representative(labels, f)
    out_proposal[f] = FOREST_NO_PROPOSAL


@wp.func
def boruvka_cross_edge(
    candidate: wp.array[wp.bool],
    edge_faces: wp.array2d[wp.int32],
    roots: wp.array[wp.int32],
    e: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # The two components edge ``e`` joins this Boruvka round, or ``(-1, -1)`` when it joins none --
    # it is not a candidate, or both its faces already sit in one component. The sentinel form
    # rather than an early return, because a ``@wp.func`` cannot return for its caller; this is the
    # convention ``remesh._resolve_flip_quad_guarded`` already uses in this tree.
    #
    # Reading ``roots`` rather than ``labels`` is the whole point and is why the two round kernels
    # must open identically: ``roots`` is the round's frozen snapshot, so both the proposal and the
    # acceptance decide against the same partition and the round's outcome does not depend on
    # thread order. See this module's docstring.
    if not candidate[e]:
        return wp.int32(-1), wp.int32(-1)
    root_a = roots[edge_faces[e, 0]]
    root_b = roots[edge_faces[e, 1]]
    if root_a == root_b:
        return wp.int32(-1), wp.int32(-1)
    return root_a, root_b


@wp.kernel
def forest_propose(
    candidate: wp.array[wp.bool],
    edge_faces: wp.array2d[wp.int32],
    roots: wp.array[wp.int32],
    out_proposal: wp.array[wp.int32],
) -> None:
    # Offer each candidate edge to the components on both sides; the lowest edge index wins.
    e = wp.int32(wp.tid())
    root_a, root_b = boruvka_cross_edge(candidate, edge_faces, roots, e)
    if root_a < 0:
        return
    wp.atomic_min(out_proposal, root_a, e)
    wp.atomic_min(out_proposal, root_b, e)


@wp.kernel
def forest_link(
    candidate: wp.array[wp.bool],
    edge_faces: wp.array2d[wp.int32],
    roots: wp.array[wp.int32],
    proposal: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Accept the edge either side chose and union the two components. Both endpoints may name the
    # same edge, and one edge may be the choice of two components at once; either way the union runs
    # once per accepted edge and the second attempt finds the roots already equal.
    #
    # An accepted edge leaves ``candidate`` rather than being marked in a mask of its own, because
    # what the caller wants is the candidates *not* in the forest. Clearing it changes no later
    # round: the thread reads only its own entry, and from the next round on the edge's two faces
    # share a root, so ``boruvka_cross_edge`` rejects it whether or not the bit is still set.
    #
    # ``LOOP_PROGRESS`` is raised by a plain store rather than counted with ``wp.atomic_add``:
    # nothing reads the merge *count*, only whether the round merged anything, and a round that
    # merged nothing means every component is spanned -- which is the loop's real exit.
    #
    # The union is ``ecl_hook_edge`` verbatim rather than a second copy of its CAS retry: it takes a
    # representative and a node, so handing it the two roots this round snapshotted costs one
    # already-compressed find. Its termination argument (parents monotonically non-increasing, so
    # ``max(root_a, root_b)`` strictly decreases) therefore applies unchanged.
    e = wp.int32(wp.tid())
    root_a, root_b = boruvka_cross_edge(candidate, edge_faces, roots, e)
    if root_a < 0:
        return
    if proposal[root_a] != e and proposal[root_b] != e:
        return
    candidate[e] = False
    out_state[LOOP_PROGRESS] = 1
    ecl_hook_edge(labels, root_a, root_b)


@wp.func
def tree_apex(
    parents: wp.array[wp.int32], distances: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.int32:
    # Lowest common ancestor of two vertices in the rooted primal tree, by lifting the deeper one to
    # the other's depth and then climbing in step. Both walks are ``O(depth)`` dependent loads on
    # one thread, which is affordable here only because there are ``2 * g`` of them and not ``V``.
    x = a
    y = b
    while distances[x] > distances[y]:
        x = parents[x]
    while distances[y] > distances[x]:
        y = parents[y]
    while x != y:
        x = parents[x]
        y = parents[y]
    return x


@wp.kernel
def generator_loop_counts(
    candidate: wp.array[wp.bool],
    unique_edges: wp.array2d[wp.int32],
    parents: wp.array[wp.int32],
    distances: wp.array[wp.int32],
    out_apex: wp.array[wp.int32],
    out_counts: wp.array[wp.vec2i],
) -> None:
    # Size each generator's loop without writing it, straight off the edge-space generator mask:
    # ``(1, length)`` at a generator edge and ``(0, 0)`` elsewhere, so one inclusive scan of the
    # pairs gives every generator its rank and its offset together, and its tail is the generator
    # count and the packed length in one read -- no ``flatnonzero`` of the mask and no second scan.
    # The length is pure depth arithmetic given the apex -- the two legs are
    # ``depth(a) - depth(apex)`` and ``depth(b) - depth(apex)`` edges long and the apex is counted
    # once -- which is the whole reason the level loop keeps ``distances`` after dropping its
    # discovery order.
    e = wp.int32(wp.tid())
    if not candidate[e]:
        out_counts[e] = wp.vec2i(0, 0)
        return
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    apex = tree_apex(parents, distances, a, b)
    out_apex[e] = apex
    out_counts[e] = wp.vec2i(1, distances[a] + distances[b] - 2 * distances[apex] + 1)


@wp.kernel
def write_generator_loops(
    candidate: wp.array[wp.bool],
    unique_edges: wp.array2d[wp.int32],
    parents: wp.array[wp.int32],
    apex: wp.array[wp.int32],
    counts: wp.array[wp.vec2i],
    out_offsets: wp.array[wp.int32],
    out_loops: wp.array[wp.int32],
) -> None:
    # Fill one generator's slice from both ends: the ``a`` leg forward from the slice's start up to
    # and including the apex, the ``b`` leg backward from its last slot down to the apex's child.
    # The two cursors meet exactly because ``generator_loop_counts`` sized the slice from the same
    # two depths, so neither bound needs re-deriving and the apex is written once.
    #
    # ``counts`` is that kernel's pairs scanned inclusively in place, so the entry below a
    # generator is ``(its rank, its offset)``. Each generator writes its own offset and its end into
    # the total-terminated ``out_offsets``; the end is the next generator's offset, so every
    # interior entry is written twice with the same value.
    #
    # The resulting order is the closed walk ``a -> apex -> b``, with the generator edge ``(b, a)``
    # closing it implicitly -- the start vertex is not repeated, matching ``boundary_loops``.
    e = wp.int32(wp.tid())
    if not candidate[e]:
        return
    top = apex[e]
    exclusive = wp.vec2i(0, 0)
    if e > 0:
        exclusive = counts[e - 1]
    inclusive = counts[e]
    out_offsets[exclusive[0]] = exclusive[1]
    out_offsets[inclusive[0]] = inclusive[1]

    slot = exclusive[1]
    node = unique_edges[e, 0]
    while node != top:
        out_loops[slot] = node
        slot = slot + 1
        node = parents[node]
    out_loops[slot] = top

    back = inclusive[1] - 1
    node = unique_edges[e, 1]
    while node != top:
        out_loops[back] = node
        back = back - 1
        node = parents[node]
