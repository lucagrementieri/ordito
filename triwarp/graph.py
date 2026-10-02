"""Generic graph algorithms on sparse adjacency matrices and edge lists (mesh-agnostic)."""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, require_same_device, run_device_loop
from triwarp.array import arange
from triwarp.kernels import array as kernel_array
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels.algorithms import connected_components as kernel_connected_components


def edges_to_csr(
    node_count: int, edges: twt.Array2dInt32, weights: wp.array[wp.float32] | None = None
) -> twt.BsrMatrix[wp.float32]:
    """
    Undirected adjacency as a 1x1-block ``warp.sparse.BsrMatrix`` (CSR form).

    Each undirected edge ``(a, b)`` contributes directed entries ``(a, b)`` and ``(b, a)``.

    Parameters
    ----------
    node_count
        Number of vertices ``0 .. node_count - 1``.
    edges
        ``(m, 2)`` edge rows on the target device.
    weights
        ``(m,)`` edge weights, one per undirected edge, written into both of its directed
        entries. Defaults to unit weights, which is what the unweighted traversals want; the
        weighted relaxation in
        [`shortest_path_envelope`][triwarp.graph.shortest_path_envelope] measures paths in whatever
        this carries — mesh edge lengths from
        [`edges_unique_length`][triwarp.edges.edges_unique_length] for a geometric distance.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(node_count, node_count)`` square adjacency. Duplicate directed pairs from repeated input
        edges are merged (values **summed**), so a repeated weighted edge doubles its weight —
        deduplicate with [`edges_unique`][triwarp.edges.edges_unique] first when that matters.

    Raises
    ------
    ValueError
        If ``weights`` is given and does not have one entry per edge row.
    RuntimeError
        If ``edges`` and ``weights`` are not all on one device.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`shortest_path_envelope`][triwarp.graph.shortest_path_envelope]
    """
    require_same_device(edges=edges, weights=weights)
    device = edges.device
    m = int(edges.shape[0])
    if weights is not None and weights.size != m:
        raise ValueError(f"weights must have one entry per edge, got {weights.size} for {m} edges")

    n_entries = 2 * m
    rows = _launch.empty(n_entries, dtype=wp.int32, device=device)
    cols = _launch.empty(n_entries, dtype=wp.int32, device=device)
    # One launch either way: the weighted form emits the structure and the two duplicated values
    # together, and the unweighted form's values are a fill rather than a launch at all.
    if weights is None:
        data = _launch.ones(n_entries, dtype=wp.float32, device=device)
        if m > 0:
            _launch.launch(
                kernel_graph.edges_to_adjacency, dim=m, inputs=[edges, rows, cols], device=device
            )
    else:
        data = _launch.empty(n_entries, dtype=wp.float32, device=device)
        if m > 0:
            _launch.launch(
                kernel_graph.edges_to_adjacency_weighted,
                dim=m,
                inputs=[edges, weights, rows, cols, data],
                device=device,
            )
    return tw.array.csr_from_triplets(node_count, node_count, rows, cols, data)


def edges_to_neighbor_lists(
    node_count: int, edges: twt.Array2dInt32, *, validate: bool = True, sort_rows: bool = False
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Per-node neighbour lists of an undirected edge list, packed as ``(neighbors, offsets)``.

    The same adjacency [`edges_to_csr`][triwarp.graph.edges_to_csr] carries, as two plain
    ``wp.int32`` buffers instead of a ``warp.sparse.BsrMatrix``: node ``v``'s neighbours are
    ``neighbors[offsets[v] : offsets[v + 1]]``. Built by counting degrees and filling through a
    per-node cursor, so there is no sort and no value array — reach for it when a traversal reads
    the structure alone, and for [`edges_to_csr`][triwarp.graph.edges_to_csr] when the answer needs
    weights, sorted rows or sparse linear algebra.

    Parameters
    ----------
    node_count
        Number of nodes ``0 .. node_count - 1``, i.e. the number of CSR rows.
    edges
        ``(m, 2)`` edge rows on the target device. Each row ``(a, b)`` contributes
        ``b`` to ``a``'s list and ``a`` to ``b``'s; a repeated row appears twice in both, so
        deduplicate with [`edges_unique`][triwarp.edges.edges_unique] first when that matters.
    validate
        When ``False``, skip the range check on ``edges`` and its host readback. See the warning
        below. Follows the same convention as
        [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges].
    sort_rows
        When ``True``, sort each row ascending in one extra launch, which makes the result
        **exactly** [`edges_to_csr`][triwarp.graph.edges_to_csr]'s structure and lifts the
        reproducibility warning below. One thread sorts one row, so the launch waits for the
        widest one and the cost grows faster than linearly in the largest degree: this is for a
        bounded-degree graph such as a mesh vertex graph, and a row of a few hundred neighbours
        already loses to [`edges_to_csr`][triwarp.graph.edges_to_csr] outright.

    Returns
    -------
    neighbors : wp.array[wp.int32]
        ``(2 * m,)`` node indices grouped by node. Ascending within a row when ``sort_rows``
        is ``True``; otherwise the order is **not specified and not reproducible** — see the
        warning below.
    offsets : wp.array[wp.int32]
        ``(node_count + 1,)`` row offsets on ``edges.device``.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``, or
        ``node_count`` is negative.

    Warning
    -------
    !!! warning "``validate=False`` trades a guard for a synchronization"
        The range check reduces the whole ``(m, 2)`` edge buffer, so it costs a device
        synchronization on a path that otherwise has none. Pass ``validate=False`` **only** when
        the caller produced ``edges`` itself and knows the bound holds. With an out-of-range index
        the unchecked path writes out of bounds rather than raising: the degree count and the fill
        both index a ``node_count``-element buffer by the raw endpoint. On a CUDA device that lands
        in device memory; on the **CPU** device a Warp array is host heap, so it overwrites glibc's
        allocator metadata and aborts the process later, somewhere unrelated. This is a sharper
        edge than [`edges_to_csr`][triwarp.graph.edges_to_csr] has, where an out-of-range triplet
        is dropped silently by ``warp.sparse``.

    Warning
    -------
    !!! warning "At the default ``sort_rows=False`` the row order is not reproducible between runs"
        Each row's slots are handed out with ``wp.atomic_add``, so the order within a row is
        thread-arrival order: it is neither sorted nor stable, and two calls on the same input
        return different permutations. The row *contents* are exact and ``offsets`` is identical
        every time; only the order inside a row moves.

        A consumer is safe when it reduces over the whole row (a minimum, a sum, a relaxation) or
        breaks its ties by node or edge index. It is **not** safe when it emits in traversal
        order, or when it feeds an ill-conditioned fit that a permutation can perturb. Both of
        those want ``sort_rows=True``, which pins the order for one extra launch;
        [`edges_to_csr`][triwarp.graph.edges_to_csr] is the answer instead when a row can hold
        hundreds of neighbours, or when the caller wants weights or sparse linear algebra.

    Notes
    -----
    **The fill is a counting sort with no sort in it, and that is the whole reason this exists
    beside [`edges_to_csr`][triwarp.graph.edges_to_csr].** A triplet build
    radix-sorts ``2 * m`` triplets and carries a ``float32`` value array a structure-only traversal
    never reads; counting and filling does neither, and even with ``sort_rows=True`` -- whose
    per-row sort is over one node's neighbours rather than the whole edge list -- the result is the
    same structure for a fraction of the work on a bounded-degree graph.

    See Also
    --------
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    [`triwarp.adjacency.vertex_face_adjacency`][triwarp.adjacency.vertex_face_adjacency]
        The same counting-sort fill for the vertex-to-face relation.
    ``igl.adjacency_list``
    """
    node_count = _validate_edge_list(edges, node_count, validate=validate)
    device = edges.device
    m = int(edges.shape[0])

    offsets = _launch.zeros(node_count + 1, dtype=wp.int32, device=device)
    neighbors = _launch.empty(2 * m, dtype=wp.int32, device=device)
    if m == 0 or node_count == 0:
        return neighbors, offsets

    # An ``(m, 2)`` edge buffer flattened *is* the entry -> node map, so one histogram over it is
    # every node's degree. ``count_occurrences``' own docstring names this as what it is for.
    degree = _launch.zeros(node_count, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_scatter.count_occurrences, dim=2 * m, inputs=[edges.flatten(), degree], device=device
    )
    # Deliberately NOT tw.array.counts_to_offsets: that helper always reads the total back, and
    # this function never needs it (it is 2 * m, known on the host). The same decline
    # ``adjacency.vertex_face_adjacency`` and ``halfedge.vertex_one_rings`` already write down.
    _launch.array_scan(degree, out_array=offsets[1:], inclusive=True)
    # ``degree`` has done its job and becomes the fill's write cursor, counting each row's free
    # slots down to zero -- which saves an allocation and a fill, and is why the scatter takes both
    # it and ``offsets``.
    _launch.launch(
        kernel_graph.scatter_neighbor_lists,
        dim=m,
        inputs=[edges, offsets, degree, neighbors],
        device=device,
    )
    if sort_rows:
        _launch.launch(
            kernel_array.sort_segments, dim=node_count, inputs=[offsets, neighbors], device=device
        )
    return neighbors, offsets


def connected_component_labels(adjacency: twt.BsrMatrix[wp.Scalar]) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from a sparse adjacency matrix.

    Uses ECL-CC (init, single-pass CAS hooking with in-kernel retry, intermediate
    pointer jumping) on the CSR structure of ``adjacency`` (1x1 BSR blocks): one
    init launch, one hook launch, one flatten launch — no host-side convergence
    loop. Each label is the smallest node id in its component (hooks always point
    the larger root at the smaller one); values are not necessarily contiguous in
    ``0 .. k-1`` (compare partitions, not raw ids).

    Parameters
    ----------
    adjacency
        ``(n, n)`` square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form, ``n``
        the node count. Each nonzero ``(i, j)`` denotes an edge between nodes ``i`` and ``j``; for
        undirected graphs both ``(i, j)`` and ``(j, i)`` should be present.

    Returns
    -------
    wp.array[wp.int32]
        ``(adjacency.nrow,)`` component labels on ``adjacency.device``. Isolated nodes (empty rows)
        receive distinct labels. When ``nnz == 0``, ``labels[i] == i``.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square or does not use 1x1 blocks.

    See Also
    --------
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    """
    node_count, offsets, indices = _validate_square_csr(adjacency)

    device = adjacency.device
    if node_count <= 1:
        return _launch.zeros(node_count, dtype=wp.int32, device=device)
    if adjacency.nnz == 0:
        return arange(node_count, device=device)

    labels = _launch.empty(node_count, dtype=wp.int32, device=device)
    parents = _launch.empty(node_count, dtype=wp.int32, device=device)

    _launch.launch(
        kernel_connected_components.ecl_init_parent,
        dim=node_count,
        inputs=[offsets, indices, parents],
        device=device,
    )
    _launch.launch(
        kernel_connected_components.ecl_hook,
        dim=node_count,
        inputs=[offsets, indices, parents],
        device=device,
    )
    _launch.launch(
        kernel_connected_components.ecl_flatten,
        dim=node_count,
        inputs=[parents, labels],
        device=device,
    )
    return labels


def connected_component_labels_from_edges(
    edges: twt.Array2dInt32, node_count: int | None = None, *, validate: bool = True
) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from an undirected edge list.

    The same ECL-CC labelling as
    [`connected_component_labels`][triwarp.graph.connected_component_labels], hooked straight off
    the edge list rather than off a sparse adjacency matrix, so no matrix is built. Each label is
    the smallest node id in its component, exactly as there.

    Parameters
    ----------
    edges
        ``(m, 2)`` edge list. Each row ``(a, b)`` connects nodes ``a``
        and ``b`` (undirected; order does not matter).
    node_count
        Number of nodes ``0 .. node_count - 1``. When ``None``, inferred as
        ``max(edges) + 1`` if ``m > 0``, else ``0``.
    validate
        When ``False``, skip the range check on ``edges`` and its host readback. Requires
        ``node_count``; see the warning below. Follows the same convention as
        [`group_int_rows`][triwarp.grouping.group_int_rows].

    Returns
    -------
    wp.array[wp.int32]
        ``(node_count,)`` component labels on ``edges.device``.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``,
        or ``node_count`` is negative.

    Warning
    -------
    !!! warning "``validate=False`` trades a guard for a synchronization"
        The range check copies the whole ``(m, 2)`` edge buffer to the host, so it costs a device
        synchronization on a path that otherwise has none. Pass ``validate=False`` **only** when
        the caller produced ``edges`` itself and knows the bound holds — an adjacency list from
        [`face_adjacency`][triwarp.adjacency.face_adjacency], say. With an out-of-range index the
        unchecked path reads **and writes** out of bounds rather than raising: the scatter kernels
        downstream index a ``node_count``-element buffer by the raw endpoint. On a CUDA device that
        lands in device memory; on the **CPU** device a Warp array is host heap, so it overwrites
        glibc's allocator metadata and aborts the process later, somewhere unrelated.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`connected_component_parity_from_edges`][triwarp.graph.connected_component_parity_from_edges]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    [`trimesh.graph.connected_component_labels`][]
    """
    node_count = _validate_edge_list(edges, node_count, validate=validate)

    # With no edges every node is its own component, which ``arange`` gives directly -- the
    # hook and flatten below would reach the same answer the long way.
    if int(edges.shape[0]) == 0:
        return arange(node_count, device=edges.device)

    # A pre-hook and a hook, one thread per edge each, then the flatten. No adjacency matrix is
    # built: the unions are per edge either way, so a CSR would only be walked back into the edges.
    device = edges.device
    parents = arange(node_count, device=device)
    _launch.launch(
        kernel_connected_components.ecl_init_parent_edges,
        dim=int(edges.shape[0]),
        inputs=[edges, parents],
        device=device,
    )
    if node_count >= kernel_connected_components.ECL_COMPRESS_FROM:
        _launch.launch(
            kernel_connected_components.ecl_compress,
            dim=node_count,
            inputs=[parents],
            device=device,
        )
    _launch.launch(
        kernel_connected_components.ecl_hook_edges,
        dim=int(edges.shape[0]),
        inputs=[edges, parents],
        device=device,
    )
    labels = _launch.empty(node_count, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_connected_components.ecl_flatten,
        dim=node_count,
        inputs=[parents, labels],
        device=device,
    )
    return labels


def connected_component_parity_from_edges(
    edges: twt.Array2dInt32, signs: wp.array[wp.int32], node_count: int, *, validate: bool = True
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Component labels plus a Z2 potential satisfying the per-edge parity constraints.

    Each edge ``(a, b)`` carries a sign in ``{0, 1}`` demanding ``parity[a] ^ parity[b] == sign``.
    On a component where those constraints are consistent this determines ``parity`` uniquely once
    the component representative is pinned to ``0``, and this function returns exactly that — the
    discrete analog of a potential function, and the reason it can replace an iterative flood fill.

    Extends ECL-CC by packing ``(parent, parity-to-parent)`` into a single ``int32`` word, so the
    union-find still hooks with one ``wp.atomic_cas``: **three launches and no host
    synchronization** whatever the graph's diameter, where propagating the bits edge by edge costs
    one launch per graph level.

    Parameters
    ----------
    edges
        ``(m, 2)`` undirected edge list; each row ``(a, b)`` constrains ``a`` and
        ``b``. Endpoints must lie in ``[0, node_count)``. Self-loops are ignored.
    signs
        ``(m,)`` parity constraint per edge, ``0`` (equal) or ``1`` (opposite).
        Any other value is rejected under ``validate``.
    node_count
        Number of nodes ``0 .. node_count - 1``.
    validate
        When ``False``, skip the range check on ``edges`` and the device synchronization it
        costs. Follows the same convention as
        [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
        — see the warning there.

    Returns
    -------
    labels : wp.array[wp.int32]
        ``(node_count,)`` labels, the smallest node id in each component, as in
        [`connected_component_labels`][triwarp.graph.connected_component_labels]. Isolated nodes
        label themselves.
    parity : wp.array[wp.int32]
        ``(node_count,)`` ``0`` / ``1`` bits, ``0`` at every component representative.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, ``signs`` is not length ``m``, ``node_count`` is
        negative, or (with ``validate``) an endpoint is out of range or a sign is not ``0`` or
        ``1``.
    RuntimeError
        If ``edges`` and ``signs`` are not all on one device.

    Warning
    -------
    !!! warning "``validate=False`` trades a guard for a synchronization"
        With an out-of-range endpoint the unchecked path reads **and writes** out of bounds
        rather than raising: ``ecl_hook_parity`` indexes a ``node_count``-element buffer by the
        raw endpoint. On a CUDA device that lands in device memory; on the **CPU** device a Warp
        array is host heap, so it overwrites glibc's allocator metadata and aborts the process
        later, somewhere unrelated. An out-of-range **sign** is bounded rather than unsafe — the
        hook keeps only its low bit — but the answer is then computed for a different constraint
        than the one passed. Pass ``validate=False`` only when the caller produced ``edges`` and
        ``signs`` itself and knows both bounds hold.

    Notes
    -----
    A component whose constraints are **contradictory** (an odd-signed cycle — a Möbius band, in
    the orientation application) admits no potential at all. No error is raised: the constraints
    along whichever spanning tree the union-find happened to build are satisfied and the remaining
    edges are left violated, so a caller that needs to know must re-test the edges against the
    returned ``parity``. This is the same best-effort contract a flood fill gives.

    See Also
    --------
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_orientation_bits`][triwarp.validation.face_orientation_bits]
    """
    require_same_device(edges=edges, signs=signs)
    twt.ensure_edge_pairs(edges, "edges")
    m = int(edges.shape[0])
    if signs.size != m:
        raise ValueError(f"signs must have length {m} to match edges, got {signs.size}")
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    if validate and m > 0:
        # One 8-byte read of both bounds, not a copy of the whole edge buffer -- see
        # ``_validate_edge_list``, whose range check this mirrors for a signed edge list.
        lowest, highest = tw.reduce.minmax(edges)
        if lowest < 0 or highest >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), got min={lowest} max={highest}"
            )
        # ``signs`` needs the same range check and for the same reason: the hook packs the parity
        # bit into the low bit of a word whose upper bits are a node id, so a sign outside {0, 1}
        # is not a tolerable approximation of one -- it addresses a different node.
        lowest_sign, highest_sign = tw.reduce.minmax(signs)
        if lowest_sign < 0 or highest_sign > 1:
            raise ValueError(f"signs must be 0 or 1, got min={lowest_sign} max={highest_sign}")

    device = edges.device
    labels = _launch.empty(node_count, dtype=wp.int32, device=device)
    parity = _launch.zeros(node_count, dtype=wp.int32, device=device)
    if node_count == 0:
        return labels, parity

    words = _launch.empty(node_count, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_connected_components.ecl_init_parent_parity,
        dim=node_count,
        inputs=[words],
        device=device,
    )
    if m > 0:
        _launch.launch(
            kernel_connected_components.ecl_hook_parity,
            dim=m,
            inputs=[edges, signs, words],
            device=device,
        )
    _launch.launch(
        kernel_connected_components.ecl_flatten_parity,
        dim=node_count,
        inputs=[words, labels, parity],
        device=device,
    )
    return labels, parity


def successor_cycles(
    edges: twt.Array2dInt32, node_count: int, *, validate: bool = True
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every cycle of a successor graph in traversal order, packed flat plus per-cycle offsets.

    The edges of a **successor graph** — one in which each node has at most one outgoing edge —
    decompose into node-disjoint cycles and chains. This orders every node along its cycle,
    following the edge direction from the cycle's smallest node index, with no per-cycle Python
    and no per-cycle allocation. The ranking is pointer-jumping (Wyllie's list ranking), so the
    work is ``O(k log L)`` over ``k`` cycle nodes with longest cycle ``L`` rather than the
    quadratic per-node successor walk. A mesh boundary's oriented edges are the motivating input
    (see [`boundary_loops_with_offsets`][triwarp.boundary.boundary_loops_with_offsets]).

    Parameters
    ----------
    edges
        ``(m, 2)`` directed edges; row ``(a, b)`` makes ``b`` the successor of
        ``a``. At most one out-edge per node. Nodes appearing in no edge belong to no cycle and
        do not appear in the result.
    node_count
        Number of nodes ``0 .. node_count - 1``.
    validate
        When ``False``, skip the range check on ``edges`` and the device synchronization it
        costs; forwarded to [`connected_component_labels_from_edges`]
        [triwarp.graph.connected_component_labels_from_edges] — see the warning there.

    Returns
    -------
    flat_cycles : wp.array[wp.int32]
        ``(n_cycle_nodes,)`` concatenated ordered node indices of every cycle, on
        ``edges.device``, ``n_cycle_nodes`` the total cycle length. Each cycle starts at its
        smallest node index and follows the edge direction.
    offsets : wp.array[wp.int32]
        ``(n_cycles + 1,)`` total-terminated offsets: cycle ``i`` occupies
        ``flat_cycles[offsets[i] : offsets[i + 1]]``; ``[0]`` when there is no cycle.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, ``node_count`` is negative, or (with ``validate``) an
        endpoint is out of range.

    Notes
    -----
    On malformed input — a node with several in-edges, so two chains merge — the cycle ranks can
    collide. Colliding nodes overwrite one slot and leave another at ``0``, which is a valid node
    index, so the result stays in-range rather than returning uninitialized garbage; it is the
    caller's job to pass a true successor graph if exact cycles are required.

    See Also
    --------
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`boundary_loops_with_offsets`][triwarp.boundary.boundary_loops_with_offsets]
    """
    # The range check must run BEFORE ``scatter_successor`` below, not be deferred to the
    # ``connected_component_labels_from_edges`` call: that kernel indexes ``next_node`` by the raw
    # edge endpoints, so an out-of-range endpoint writes past a ``node_count``-element buffer. On
    # the CPU device that is a host-heap overwrite, silent at the point of the write and surfacing
    # later as a glibc abort somewhere unrelated.
    node_count = _validate_edge_list(edges, node_count, validate=validate)

    device = edges.device
    m = int(edges.shape[0])
    if m == 0 or node_count == 0:
        return _launch.empty(0, dtype=wp.int32, device=device), _launch.zeros(
            1, dtype=wp.int32, device=device
        )

    # The successor table and the distinct-endpoint mask come out of one pass over the edges. The
    # distinct endpoints, ascending, are then the mask's ``flatnonzero`` rather than ``unique_1d``
    # over the flattened pairs: both return the sorted distinct values, and the range check above
    # has already guaranteed every endpoint indexes the mask, but this one is a zeroed buffer and a
    # scan where that one is a hash table, a compaction, a radix sort and two host readbacks.
    next_node = _launch.full(node_count, -1, dtype=wp.int32, device=device)
    node_mask = _launch.zeros(node_count, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_graph.scatter_successor, dim=m, inputs=[edges, next_node, node_mask], device=device
    )

    # Already range-checked above, so the downstream call skips the second reduction and host sync.
    labels = connected_component_labels_from_edges(edges, node_count=node_count, validate=False)

    cycle_nodes = tw.array.flatnonzero(node_mask)
    n_nodes = cycle_nodes.size

    label_min = _launch.full(node_count, node_count, dtype=wp.int32, device=device)
    label_count = _launch.zeros(node_count, dtype=wp.int32, device=device)
    is_chain = _launch.zeros(node_count, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_graph.scatter_cycle_min_and_count,
        dim=n_nodes,
        inputs=[cycle_nodes, next_node, labels, label_min, label_count, is_chain],
        device=device,
    )

    # A successor graph decomposes into node-disjoint cycles *and* chains (see the docstring). A
    # chain component contains a node with no outgoing edge (``next_node[v] < 0``), which
    # ``init_rank_arrays`` below treats as a fixed point identical to a genuine cycle's cut at
    # ``label_min`` -- two fixed points in one component collide onto rank slot 0 and fabricate a
    # bogus "cycle" out of whatever the collision leaves there, including node ids that never
    # appeared in the input. Excluding a chain's nodes here keeps that collision scoped to the
    # malformed input the Notes above already describe (an in-degree collision), rather than
    # firing on an ordinary chain.
    _launch.launch(
        kernel_graph.chain_node_mask,
        dim=n_nodes,
        inputs=[cycle_nodes, labels, is_chain, node_mask],
        device=device,
    )
    cycle_nodes = tw.array.flatnonzero(node_mask)
    n_nodes = cycle_nodes.size
    if n_nodes == 0:
        return _launch.empty(0, dtype=wp.int32, device=device), _launch.zeros(
            1, dtype=wp.int32, device=device
        )

    # Pointer-jumping list ranking (Wyllie): O(log L) rounds of pointer jumping replace the
    # per-node successor walk, whose total work was quadratic in the cycle length. Each round
    # chases several pointers (``pointer_jump_schedule``), so the round count is a logarithm to a
    # base above two.
    successor = _launch.empty(node_count, dtype=wp.int32, device=device)
    steps = _launch.empty(node_count, dtype=wp.int32, device=device)
    successor_next = _launch.empty(node_count, dtype=wp.int32, device=device)
    steps_next = _launch.empty(node_count, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_graph.init_rank_arrays,
        dim=n_nodes,
        inputs=[cycle_nodes, next_node, labels, label_min, successor, steps],
        device=device,
    )
    hops, rounds = kernel_graph.pointer_jump_schedule(n_nodes)
    for _ in range(rounds):
        _launch.launch(
            kernel_graph.jump_rank,
            dim=n_nodes,
            inputs=[cycle_nodes, successor, steps, hops, successor_next, steps_next],
            device=device,
        )
        successor, successor_next = successor_next, successor
        steps, steps_next = steps_next, steps

    position = _launch.empty(n_nodes, dtype=wp.int32, device=device)
    label_mask = _launch.zeros(node_count, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_graph.finalize_rank_positions,
        dim=n_nodes,
        inputs=[cycle_nodes, labels, label_count, steps, position, label_mask],
        device=device,
    )

    # The cycles are grouped by label, and a label is a node index, so the mask's compact ranks
    # number them in ascending label order -- the order ``unique_1d`` over the labels gave.
    label_ranks, n_cycles = tw.array.mask_to_compact_ranks(label_mask)
    # The sizes land in the tail of the ``n_cycles + 1`` offsets buffer and are scanned there in
    # place, behind the leading zero, so no separate sizes buffer is allocated.
    offsets = _launch.zeros(n_cycles + 1, dtype=wp.int32, device=device)
    sizes = twt.as_dense(offsets[1:])
    _launch.launch(
        kernel_graph.compact_cycle_sizes,
        dim=node_count,
        inputs=[label_mask, label_ranks, label_count, sizes],
        device=device,
    )
    _launch.array_scan(sizes, out_array=sizes, inclusive=True)

    # Zero-initialised (not wp.empty): colliding ranks on malformed input (see Notes) can leave
    # slots unwritten by scatter_cycle_slot, and zero is a valid node index.
    flat_cycles = _launch.zeros(n_nodes, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_graph.scatter_cycle_slot,
        dim=n_nodes,
        inputs=[cycle_nodes, labels, label_ranks, position, offsets, flat_cycles],
        device=device,
    )

    return flat_cycles, offsets


def shortest_path_envelope(
    adjacency: twt.BsrMatrix[wp.float32], values: wp.array[wp.float32], max_iterations: int = 0
) -> wp.array[wp.float32]:
    """

    Lower every node's value onto the shortest-path envelope ``min_u (values[u] + d(u, v))``.

    Two readings of one relaxation, and both are worth knowing because they are the same call:

    * **A shortest-path distance — Dijkstra's answer.** Seed ``values`` with ``0`` on the source
      nodes and a number larger than any reachable distance elsewhere, and the result is the
      weighted multi-source shortest-path distance to the nearest source, ``d`` being the sum of
      ``adjacency``'s values along the path, agreeing with
      [`scipy.sparse.csgraph.dijkstra`][] up to float32 precision.
    * **A Lipschitz cap.** Applied to an arbitrary field it enforces
      ``values[i] <= values[j] + w(i, j)`` on every edge by lowering values only, so every local
      minimum of the input survives untouched and only peaks that rise too steeply out of them are
      shaved down. That is MeshLab's ``apply_scalar_saturation_per_vertex`` (VCG
      ``UpdateQuality::VertexSaturate``) and the standard way to make a raw scalar usable as a
      **sizing field**: an adaptive remesher fed an ungraded target-length field produces a band of
      bad triangles where the field jumps, and this is the projection that removes the jump while
      respecting the field's small values.

    MeshLab's ``gradientthr`` is not a parameter here because it is a property of the *graph*: its
    cap is ``|p_i - p_j| / gradientthr``, so dividing the edge lengths by it when building
    ``adjacency`` reproduces it exactly, and the same weights then serve any other slope.


    Parameters
    ----------
    adjacency
        ``(node_count, node_count)`` square undirected adjacency in 1x1-block
        ``warp.sparse.BsrMatrix`` form whose **values are the edge weights**, as
        [`edges_to_csr`][triwarp.graph.edges_to_csr] builds with its ``weights`` argument.
        Negative weights are not admissible: the iteration would not terminate at the envelope.
    values
        ``(node_count,)`` initial labels. Not modified.
    max_iterations
        Cap on relaxation passes. Each pass propagates one edge further, so the number needed is the
        graph diameter of the region that violates the bound. ``0`` (the default) means
        ``node_count``, which can never be exceeded.

    Returns
    -------
    wp.array[wp.float32]
        ``(node_count,)`` envelope on ``values.device``.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square with 1x1 blocks, if ``max_iterations`` is negative, if
        ``values`` is not length ``node_count``, or if any weight in ``adjacency`` is negative.
    RuntimeError
        If ``adjacency`` and ``values`` are not on the same device.

    Examples
    --------
    Geodesic-ish distance from vertex 0 along the mesh's edges — the mesh recipe for both readings,
    since the weights are what make the envelope geometric:

    ```python
    n_vertices = int(v.shape[0])
    edges = tw.edges.edges_unique(f, n_vertices=n_vertices, validate=False)[0]
    lengths = tw.edges.edges_unique_length(v, f, edges)
    adjacency = tw.graph.edges_to_csr(n_vertices, edges, lengths)
    seed = wp.full(n_vertices, 1.0e6, dtype=wp.float32, device=v.device)
    wp.copy(seed[:1], wp.zeros(1, dtype=wp.float32, device=v.device))
    print(float(tw.reduce.max(tw.graph.shortest_path_envelope(adjacency, seed))))
    ```

    Notes
    -----
    **The answer is Dijkstra's; the method is Bellman-Ford.** A priority queue is inherently serial
    -- it processes one node per pop -- so this relaxes *every* node against its neighbours in
    parallel and repeats until nothing improves. Shortest-path distances are unique, so the two
    agree on the result; what differs is the cost model -- ``O(diameter)`` launches over the whole
    CSR here against ``O(E log V)`` sequential work there. The name is the result, per this
    package's naming rule, not the algorithm.

    One relaxation kernel per pass, so a pass is a pure function of the previous labels and the
    answer does not depend on thread interleaving. The pass count is data-dependent; on CUDA the
    whole pass loop runs as one device-side conditional graph (``wp.capture_while``), so the
    convergence check costs no host readback at all rather than the one-per-pass a naive early exit
    would need. The CPU backend, which has no conditional-graph capture, still checks with a plain
    readback per pass.

    For distance *across* a surface rather than along its edges -- shorter, and what "geodesic"
    usually means -- use [`heat_geodesic`][triwarp.heat.heat_geodesic]. The edge-graph distance is
    an upper bound on it.

    See Also
    --------
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    [`triwarp.remesh.isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    [`scipy.sparse.csgraph.dijkstra`][]
    """
    require_same_device(adjacency=adjacency, values=values)
    node_count, offsets, columns = _validate_square_csr(adjacency)
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")
    if values.size != node_count:
        raise ValueError(
            f"values must have one entry per node, got {values.size} for {node_count} nodes"
        )

    device = wp.get_device(values.device)
    labels = _launch.clone(values)
    if node_count == 0:
        return labels

    weights = adjacency.values
    if weights.size > 0 and float(tw.reduce.min(weights)) < 0.0:
        raise ValueError("adjacency weights must be non-negative for the envelope to converge")

    max_pass_count = max_iterations or node_count
    relaxed = _launch.empty(node_count, dtype=wp.float32, device=device)
    # The shared round-loop state word (``kernels/array.py``): the relaxation pass raises
    # ``LOOP_PROGRESS`` and ``array.loop_advance`` publishes the condition against the pass cap.
    # Allocated inside each branch because the two seed it differently -- the host loop tests
    # ``LOOP_PROGRESS`` itself and needs no condition, while the captured loop must start with a
    # non-zero one. Allocating zeroed above and re-seeding below would be an allocation plus a
    # whole second upload of the same twelve bytes.
    if not device.is_cuda:
        state = _launch.zeros(kernel_array.LOOP_ADVANCE_STATE_SIZE, dtype=wp.int32, device=device)
        # No conditional-graph capture on the CPU backend, so the pass loop runs on the host;
        # the plain per-pass loop below, with its one 4-byte readback per pass, is already the
        # cheapest thing a CPU launch can do here.
        progress = twt.as_dense(state[kernel_array.LOOP_PROGRESS_VIEW])
        for _ in range(max_pass_count):
            _launch.zero_(progress)
            _launch.launch(
                kernel_graph.shortest_path_envelope_pass,
                dim=node_count,
                inputs=[offsets, columns, weights, labels, relaxed, state],
                device=device,
            )
            labels, relaxed = relaxed, labels
            if int(read_scalar(progress, 0)) == 0:
                break
        return labels

    # CUDA: the whole pass loop runs on-device via ``wp.capture_while``, so the
    # only host sync in the common case is none at all -- each pass's convergence check and
    # iteration cap are folded into ``array.loop_advance``, the package's shared round-closing
    # kernel, which runs after the relax kernel and the label copy below.
    #
    # Buffer *swapping* (the CPU path's ``labels, relaxed = relaxed, labels``) cannot be captured:
    # a conditional graph replays the exact pointers its body recorded the one time it was traced,
    # so a Python-level rebind between iterations has no effect on the device-side loop -- every
    # replayed pass would keep reading and writing the same two buffers in the same direction.
    # ``wp.copy`` moves this pass's answer into ``labels`` in place instead, which is itself just a
    # device memcpy and captures fine (no allocation, no host sync, unlike ``wp.utils.array_scan``).
    #
    # Unrolling two passes per round into each other's buffer would remove that copy and is **not
    # portable** -- a ``wp.capture_while`` body is not guaranteed to replay as an indivisible unit
    # across devices, so a loop whose result buffer depends on it cannot rely on it. The numbers and
    # the ping-pong corollary are at ``kernels/graph.shortest_path_envelope_pass``.
    # ``wp.capture_while`` reads the condition before the first round, so it starts non-zero.
    state = _launch.array([0, 1, 0], dtype=wp.int32, device=device)
    max_pass_count_i32 = wp.int32(max_pass_count)

    def envelope_pass_body() -> None:
        # One pass, then a copy of its result back over ``labels``. **Two passes per round, written
        # into each other's buffer, would remove the copy** -- it is a whole device pass over the
        # node array and a real fraction of this call -- and it is **not portable**: the two devices
        # then disagree. Measured with the unrolled body, the CPU device relaxed strictly fewer
        # nodes per round than CUDA at every cap, because the recorded body did not replay as two
        # passes per round there. A ``wp.capture_while`` body is not guaranteed to execute as an
        # indivisible unit across devices, so a loop whose *result
        # buffer* depends on the body running whole cannot rely on it. A Python-level ping-pong
        # cannot help either: the body is recorded once and replayed, so rebinding the names would
        # only take effect at record time.
        _launch.launch(
            kernel_graph.shortest_path_envelope_pass,
            dim=node_count,
            inputs=[offsets, columns, weights, labels, relaxed, state],
            device=device,
        )
        _launch.copy(labels, relaxed)
        _launch.launch(
            kernel_array.loop_advance, dim=1, inputs=[max_pass_count_i32, state], device=device
        )

    run_device_loop(
        device, twt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), envelope_pass_body
    )
    return labels


# --- private helpers ---------------------------------------------------------------------


def _validate_square_csr(
    adjacency: twt.BsrMatrix[wp.Scalar],
) -> tuple[int, wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Check a CSR adjacency is square with scalar blocks, and unpack what the traversals need.

    The shared entry check of every function here that takes a prebuilt adjacency. Warp's stub
    omits ``BsrMatrix.nrow`` / ``.ncol`` / ``.offsets`` / ``.columns``, all of which exist at
    runtime; ``reportAttributeAccessIssue`` is off package-wide for that reason (see
    ``pyproject.toml``), so the reads below need no per-site marker.
    """
    node_count = adjacency.nrow
    ncol = adjacency.ncol
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")
    offsets = adjacency.offsets
    columns = adjacency.columns
    return int(node_count), offsets, columns


def _validate_edge_list(edges: twt.Array2dInt32, node_count: int | None, *, validate: bool) -> int:
    """
    Check an ``(m, 2)`` edge list and resolve its node count.

    The shared entry check of every function here that takes an edge list instead of a prebuilt
    adjacency. When ``node_count`` is ``None`` it is inferred from the edges; when it is supplied
    it is checked for sign and, under ``validate``, the edge indices are checked to fall inside it.

    ``validate=False`` skips only the range check, which is one
    [`minmax`][triwarp.reduce.minmax] plus a host synchronization -- pass it from a caller that
    forwards the same edges to another checked entry point, so the reduction is not paid twice.
    """
    twt.ensure_edge_pairs(edges, "edges")

    if node_count is None:
        return int(tw.array.index_bound(edges))
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    if validate and int(edges.shape[0]) > 0:
        # One 8-byte read of both bounds, not a copy of the whole edge buffer.
        lowest, highest = tw.reduce.minmax(edges)
        if lowest < 0 or highest >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), got min={lowest} max={highest}"
            )
    return node_count
