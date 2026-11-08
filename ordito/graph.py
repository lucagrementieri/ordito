"""Generic graph algorithms on sparse adjacency matrices and edge lists (mesh-agnostic)."""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._common import read_scalar, read_values, require_same_device, run_device_loop
from ordito.array import arange
from ordito.constants import INT32_MAX, TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import graph as kernel_graph
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels.algorithms import connected_components as kernel_connected_components


def edges_to_csr(
    node_count: int, edges: odt.Array2dInt32, weights: wp.array[wp.float32] | None = None
) -> odt.BsrMatrix[wp.float32]:
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
        [`shortest_path_envelope`][ordito.graph.shortest_path_envelope] measures paths in whatever
        this carries — mesh edge lengths from
        [`edges_unique_length`][ordito.edges.edges_unique_length] for a geometric distance.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(node_count, node_count)`` square adjacency. Duplicate directed pairs from repeated input
        edges are merged (values **summed**), so a repeated weighted edge doubles its weight —
        deduplicate with [`edges_unique`][ordito.edges.edges_unique] first when that matters.

    Raises
    ------
    ValueError
        If ``weights`` is given and does not have one entry per edge row.
    RuntimeError
        If ``edges`` and ``weights`` are not all on one device.

    See Also
    --------
    [`connected_component_labels`][ordito.graph.connected_component_labels]
    [`shortest_path_envelope`][ordito.graph.shortest_path_envelope]

    Examples
    --------
    Three edges become six directed entries; node 5 has no edge and an empty row:

    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [3, 4]], dtype=wp.int32)
    adjacency = od.graph.edges_to_csr(6, edges)
    print(adjacency.nrow, adjacency.nnz_sync())  # 6 6
    print(adjacency.offsets.numpy().tolist())  # [0, 1, 3, 4, 5, 6, 6]
    ```
    """
    require_same_device(edges=edges, weights=weights)
    device = edges.device
    m = int(edges.shape[0])
    if weights is not None and weights.size != m:
        raise ValueError(f"weights must have one entry per edge, got {weights.size} for {m} edges")

    n_entries = 2 * m
    # The edge kernel writes each directed entry's key straight into the sort buffers, and both of
    # its values, so no ``rows`` / ``cols`` triplet table is written only to be packed and the unit
    # weights take no fill: one launch, weighted or not.
    keys, order = od.array.csr_key_buffers(n_entries, device)
    data = _launch.empty(n_entries, dtype=wp.float32, device=device)
    if m > 0:
        _launch.launch(
            kernel_graph.edges_to_adjacency_keys,
            dim=m,
            inputs=[edges, weights, wp.int32(node_count)],
            outputs=[keys, order, data],
            device=device,
        )
    return od.array.csr_from_coo_keys(node_count, node_count, keys, order, data)


def edges_to_neighbor_lists(
    node_count: int, edges: odt.Array2dInt32, *, validate: bool = True, sort_rows: bool = False
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Per-node neighbour lists of an undirected edge list, packed as ``(neighbors, offsets)``.

    The same adjacency [`edges_to_csr`][ordito.graph.edges_to_csr] carries, as two plain
    ``wp.int32`` buffers instead of a ``warp.sparse.BsrMatrix``: node ``v``'s neighbours are
    ``neighbors[offsets[v] : offsets[v + 1]]``. Built by counting degrees and filling through a
    per-node cursor, so there is no sort and no value array — reach for it when a traversal reads
    the structure alone, and for [`edges_to_csr`][ordito.graph.edges_to_csr] when the answer needs
    weights, sorted rows or sparse linear algebra.

    Parameters
    ----------
    node_count
        Number of nodes ``0 .. node_count - 1``, i.e. the number of CSR rows.
    edges
        ``(m, 2)`` edge rows on the target device. Each row ``(a, b)`` contributes
        ``b`` to ``a``'s list and ``a`` to ``b``'s; a repeated row appears twice in both, so
        deduplicate with [`edges_unique`][ordito.edges.edges_unique] first when that matters.
    validate
        When ``False``, skip the range check on ``edges`` and its host readback. See the warning
        below. Follows the same convention as
        [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges].
    sort_rows
        When ``True``, sort each row ascending in one extra launch, which makes the result
        **exactly** [`edges_to_csr`][ordito.graph.edges_to_csr]'s structure and lifts the
        reproducibility warning below. One thread sorts one row, so the launch waits for the
        widest one and the cost grows faster than linearly in the largest degree: this is for a
        bounded-degree graph such as a mesh vertex graph, and a row of a few hundred neighbours
        already loses to [`edges_to_csr`][ordito.graph.edges_to_csr] outright.

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
        The range check reads a flag the degree count raises back to the host, so it costs a
        device synchronization on a path that otherwise has none. Pass ``validate=False``
        **only** when the caller produced ``edges`` itself and knows the bound holds. With an
        out-of-range index the unchecked path writes out of bounds rather than raising: the
        degree count and the fill both index a ``node_count``-element buffer by the raw endpoint.
        On a CUDA device that lands in device memory; on the **CPU** device a Warp array is host
        heap, so it overwrites glibc's allocator metadata and aborts the process later, somewhere
        unrelated. This is a sharper edge than [`edges_to_csr`][ordito.graph.edges_to_csr] has,
        where an out-of-range triplet is dropped silently by ``warp.sparse``.

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
        [`edges_to_csr`][ordito.graph.edges_to_csr] is the answer instead when a row can hold
        hundreds of neighbours, or when the caller wants weights or sparse linear algebra.

    Notes
    -----
    **The fill is a counting sort with no sort in it, and that is the whole reason this exists
    beside [`edges_to_csr`][ordito.graph.edges_to_csr].** A triplet build
    radix-sorts ``2 * m`` triplets and carries a ``float32`` value array a structure-only traversal
    never reads; counting and filling does neither, and even with ``sort_rows=True`` -- whose
    per-row sort is over one node's neighbours rather than the whole edge list -- the result is the
    same structure for a fraction of the work on a bounded-degree graph.

    See Also
    --------
    [`edges_to_csr`][ordito.graph.edges_to_csr]
    [`edges_to_neighbor_lists_from_counts`][ordito.graph.edges_to_neighbor_lists_from_counts]
        The same lists from node degrees the caller already holds.
    [`ordito.adjacency.vertex_face_adjacency`][ordito.adjacency.vertex_face_adjacency]
        The same counting-sort fill for the vertex-to-face relation.
    ``igl.adjacency_list``

    Examples
    --------
    Node ``v``'s neighbours are ``neighbors[offsets[v] : offsets[v + 1]]``:

    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [3, 4]], dtype=wp.int32)
    neighbors, offsets = od.graph.edges_to_neighbor_lists(6, edges, sort_rows=True)
    print(neighbors.numpy().tolist())  # [1, 0, 2, 1, 4, 3]
    print(offsets.numpy().tolist())  # [0, 1, 3, 4, 5, 6, 6]
    ```
    """
    # The range check rides in the degree count below rather than in a reduction of its own.
    node_count = _validate_edge_list(edges, node_count)
    device = edges.device
    m = int(edges.shape[0])
    if m == 0 or node_count == 0:
        if validate and m > 0:
            _raise_edge_range(edges, node_count)
        return (
            _launch.empty(2 * m, dtype=wp.int32, device=device),
            _launch.zeros(node_count + 1, dtype=wp.int32, device=device),
        )

    # The degrees are counted in the offsets' own buffer, behind two zeros, and scanned there in
    # place, so the fill's atomics advance each row's start to its end and the buffer ends as the
    # offsets (``kernels/scatter.claim_row_slot``). The rows share its allocation (a CSR pair is
    # held and dropped together); only the counts are zeroed, as the rows are written whole.
    buffer, neighbors = _launch.carve(wp.int32, (node_count + 2, 2 * m), device)
    _launch.zero_(buffer)
    if validate:
        # The count skips an out-of-range row and flags it in entry 1, which must read zero for the
        # fill anyway, so the check is this launch's and one 4-byte read, and the fill below never
        # sees such a row.
        _launch.launch(
            kernel_graph.count_edge_endpoints_checked,
            dim=m,
            inputs=[edges],
            outputs=[_launch.view(buffer, 1, node_count + 2)],
            device=device,
        )
        # Decides whether to raise before the fill indexes by the raw endpoints.
        if read_scalar(buffer, 1) != 0:
            _raise_edge_range(edges, node_count)
    else:
        # An ``(m, 2)`` edge buffer flattened *is* the entry -> node map, so one histogram over it
        # is every node's degree. ``count_occurrences``' own docstring names this as what it is for.
        _launch.launch(
            kernel_scatter.count_occurrences,
            dim=2 * m,
            inputs=[edges.flatten()],
            outputs=[_launch.view(buffer, 2, node_count + 2)],
            device=device,
        )
    counts = _launch.view(buffer, 2, node_count + 2)
    _launch.array_scan(counts, out_array=counts, inclusive=True)
    return _fill_neighbor_lists(edges, node_count, buffer, neighbors, sort_rows)


def edges_to_neighbor_lists_from_counts(
    edges: odt.Array2dInt32, counts: wp.array[wp.int32], *, sort_rows: bool = False
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    [`edges_to_neighbor_lists`][ordito.graph.edges_to_neighbor_lists] from degrees already counted.

    The second half of that function, for a caller whose own kernel counted every node's degree in
    a pass it makes anyway: the degree histogram is the one launch this skips. ``counts`` is
    trusted, not checked: it must be exactly each node's number of entries in ``edges``.

    Parameters
    ----------
    edges
        ``(m, 2)`` edge rows, every endpoint in ``[0, n_nodes)``. Unchecked, with the
        consequences [`edges_to_neighbor_lists`][ordito.graph.edges_to_neighbor_lists] describes
        for ``validate=False``.
    counts
        ``(n_nodes,)`` ``int32`` number of times each node appears in ``edges``, on
        ``edges.device``. Not modified.
    sort_rows
        As in [`edges_to_neighbor_lists`][ordito.graph.edges_to_neighbor_lists]: sort each row
        ascending in one extra launch, which makes the row order reproducible.

    Returns
    -------
    neighbors : wp.array[wp.int32]
        ``(2 * m,)`` node indices grouped by node, ordered as in
        [`edges_to_neighbor_lists`][ordito.graph.edges_to_neighbor_lists].
    offsets : wp.array[wp.int32]
        ``(n_nodes + 1,)`` row offsets on ``edges.device``.

    Raises
    ------
    RuntimeError
        If ``edges`` and ``counts`` are not on one device.

    See Also
    --------
    [`edges_to_neighbor_lists`][ordito.graph.edges_to_neighbor_lists]
    [`ordito.adjacency.vertex_face_adjacency_from_counts`][ordito.adjacency.vertex_face_adjacency_from_counts]
        The same split for the vertex-to-face relation.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [3, 4]], dtype=wp.int32)
    counts = wp.array([1, 2, 1, 1, 1, 0], dtype=wp.int32)
    neighbors, offsets = od.graph.edges_to_neighbor_lists_from_counts(edges, counts, sort_rows=True)
    print(neighbors.numpy().tolist())  # [1, 0, 2, 1, 4, 3]
    print(offsets.numpy().tolist())  # [0, 1, 3, 4, 5, 6, 6]
    ```
    """
    require_same_device(edges=edges, counts=counts)
    device = edges.device
    m = int(edges.shape[0])
    node_count = counts.size
    if m == 0 or node_count == 0:
        return (
            _launch.empty(2 * m, dtype=wp.int32, device=device),
            _launch.zeros(node_count + 1, dtype=wp.int32, device=device),
        )
    # The offsets' buffer as ``edges_to_neighbor_lists`` leaves it before its fill, one entry
    # shorter: the exclusive scan of ``counts`` behind entry 0 puts each row's start where the
    # fill's claims read it, row 0's zero included, and the fill writes entry 0 itself, so nothing
    # is zeroed. Deliberately NOT od.array.counts_to_offsets: that helper always reads the total
    # back, and this function never needs it (it is 2 * m, known on the host). The same decline
    # ``adjacency.vertex_face_adjacency`` and ``halfedge.vertex_one_rings`` already write down.
    buffer, neighbors = _launch.carve(wp.int32, (node_count + 1, 2 * m), device)
    _launch.array_scan(counts, out_array=_launch.view(buffer, 1, node_count + 1), inclusive=False)
    return _fill_neighbor_lists(edges, node_count, buffer, neighbors, sort_rows)


def _fill_neighbor_lists(
    edges: odt.Array2dInt32,
    node_count: int,
    buffer: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    sort_rows: bool,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Fill the neighbour lists from each row's start in ``buffer[v + 1]``; return them packed.

    The fill's claims (``kernels/scatter.claim_row_slot``) advance each start to its row's end, and
    the fill writes ``buffer[0]``, so the first ``node_count + 1`` entries end as the offsets.
    """
    device = edges.device
    _launch.launch(
        kernel_graph.scatter_neighbor_lists,
        dim=int(edges.shape[0]),
        inputs=[edges],
        outputs=[buffer, neighbors],
        device=device,
    )
    offsets = _launch.view(buffer, 0, node_count + 1)
    if sort_rows:
        _launch.launch(
            kernel_array.sort_segments, dim=node_count, inputs=[offsets, neighbors], device=device
        )
    return neighbors, offsets


def connected_component_labels(adjacency: odt.BsrMatrix[wp.Scalar]) -> wp.array[wp.int32]:
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
        receive distinct labels, so a matrix with no entries gives ``labels[i] == i``.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square or does not use 1x1 blocks.

    See Also
    --------
    [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
    [`edges_to_csr`][ordito.graph.edges_to_csr]
    [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels]

    Examples
    --------
    Each label is the smallest node of its component; the isolated node 5 labels itself:

    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [3, 4]], dtype=wp.int32)
    adjacency = od.graph.edges_to_csr(6, edges)
    labels = od.graph.connected_component_labels(adjacency)
    print(labels.numpy().tolist())  # [0, 0, 0, 3, 3, 5]
    ```
    """
    node_count, offsets, indices = _validate_square_csr(adjacency)

    device = adjacency.device
    if node_count <= 1:
        return _launch.zeros(node_count, dtype=wp.int32, device=device)
    # No short-circuit on an empty matrix: ``adjacency.nnz`` is a capacity rather than the entry
    # count, and the general path below already labels every node of an empty row by itself.

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
    edges: odt.Array2dInt32, node_count: int | None = None, *, validate: bool = True
) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from an undirected edge list.

    The same ECL-CC labelling as
    [`connected_component_labels`][ordito.graph.connected_component_labels], hooked straight off
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
        [`group_int_rows`][ordito.grouping.group_int_rows].

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
        The range check reads a flag the pre-hook raises back to the host, so it costs a device
        synchronization on a path that otherwise has none. Pass ``validate=False`` **only** when
        the caller produced ``edges`` itself and knows the bound holds — an adjacency list from
        [`face_adjacency`][ordito.adjacency.face_adjacency], say. With an out-of-range index the
        unchecked path reads **and writes** out of bounds rather than raising: the scatter kernels
        downstream index a ``node_count``-element buffer by the raw endpoint. On a CUDA device that
        lands in device memory; on the **CPU** device a Warp array is host heap, so it overwrites
        glibc's allocator metadata and aborts the process later, somewhere unrelated.

    See Also
    --------
    [`connected_component_labels`][ordito.graph.connected_component_labels]
    [`connected_component_parity_from_edges`][ordito.graph.connected_component_parity_from_edges]
    [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels]
    [`trimesh.graph.connected_component_labels`][]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [3, 4]], dtype=wp.int32)
    labels = od.graph.connected_component_labels_from_edges(edges, node_count=6)
    print(labels.numpy().tolist())  # [0, 0, 0, 3, 3, 5]
    ```
    """
    # The range check rides in the pre-hook below rather than in a reduction of its own.
    validate = validate and node_count is not None
    node_count = _validate_edge_list(edges, node_count)

    # With no edges every node is its own component, which ``arange`` gives directly -- the
    # hook and flatten below would reach the same answer the long way.
    if int(edges.shape[0]) == 0:
        return arange(node_count, device=edges.device)

    # A pre-hook and a hook, one thread per edge each, then the flatten. No adjacency matrix is
    # built: the unions are per edge either way, so a CSR would only be walked back into the edges.
    device = edges.device
    if validate:
        # One identity entry past the forest is the range flag: the checked pre-hook skips an
        # out-of-range row and overwrites the entry, so the check is that launch's and one 4-byte
        # read, made before the hook indexes by the raw endpoints.
        parents_and_flag = arange(node_count + 1, device=device)
        _launch.launch(
            kernel_graph.ecl_init_parent_edges_checked,
            dim=int(edges.shape[0]),
            inputs=[edges, parents_and_flag],
            device=device,
        )
        # Decides whether to raise. Declined, a checked hook skipping the same rows and this read
        # made after the flatten: 1.03-1.35x from ``sphere_small`` to scrambled ``dragon`` on
        # CUDA (2026-11-08). Read here, the host waits on the short pre-hook alone and the hook
        # and flatten run on while the caller issues its next work; read last, it waits on all.
        if read_scalar(parents_and_flag, node_count) != node_count:
            _raise_edge_range(edges, node_count)
        parents = _launch.view(parents_and_flag, 0, node_count)
    else:
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
    edges: odt.Array2dInt32, signs: wp.array[wp.int32], node_count: int, *, validate: bool = True
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
        [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
        — see the warning there.

    Returns
    -------
    labels : wp.array[wp.int32]
        ``(node_count,)`` labels, the smallest node id in each component, as in
        [`connected_component_labels`][ordito.graph.connected_component_labels]. Isolated nodes
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
    [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
    [`face_orientation_bits`][ordito.validation.face_orientation_bits]

    Examples
    --------
    Edges ``0-1`` and ``1-2`` demand opposite bits, ``2-0`` equal bits, which is consistent:

    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [2, 0], [3, 4]], dtype=wp.int32)
    signs = wp.array([1, 1, 0, 1], dtype=wp.int32)
    labels, parity = od.graph.connected_component_parity_from_edges(edges, signs, 5)
    print(labels.numpy().tolist())  # [0, 0, 0, 3, 3]
    print(parity.numpy().tolist())  # [0, 1, 0, 0, 1]
    ```
    """
    require_same_device(edges=edges, signs=signs)
    odt.ensure_edge_pairs(edges, "edges")
    m = int(edges.shape[0])
    if signs.size != m:
        raise ValueError(f"signs must have length {m} to match edges, got {signs.size}")
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    # Both range checks -- endpoints and signs -- ride in the hook: it skips a row that fails
    # either and sets a flag word past the forest, read back once after the flatten. ``signs``
    # needs the check for the same reason ``edges`` does: the hook packs the parity bit into the
    # low bit of a word whose upper bits are a node id, so a sign outside {0, 1} is not a tolerable
    # approximation of one -- it addresses a different node.
    checked = validate and m > 0
    if checked and node_count == 0:
        _raise_signed_edge_range(edges, signs, node_count)

    device = edges.device
    # ``ecl_flatten_parity`` writes both for every node, so neither needs an initial value.
    # Declined, both carved from one allocation (``_launch.carve``): 1.02-1.04x on CUDA from
    # ``sphere_small`` to ``dragon``, 0.99x on ``lucy`` (2026-11-08).
    labels = _launch.empty(node_count, dtype=wp.int32, device=device)
    parity = _launch.empty(node_count, dtype=wp.int32, device=device)
    if node_count == 0:
        return labels, parity

    if checked:
        words = _launch.empty(node_count + 1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_graph.ecl_init_parent_parity_checked,
            dim=node_count + 1,
            inputs=[words],
            device=device,
        )
        _launch.launch(
            kernel_graph.ecl_hook_parity_checked, dim=m, inputs=[edges, signs, words], device=device
        )
    else:
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
    # The flag decides whether to raise; the forest stayed in range either way.
    if checked and read_scalar(words, node_count) != 0:
        _raise_signed_edge_range(edges, signs, node_count)
    return labels, parity


def _raise_signed_edge_range(
    edges: odt.Array2dInt32, signs: wp.array[wp.int32], node_count: int
) -> None:
    """
    Raise the error for a signed edge list ``ecl_hook_parity_checked`` flagged, naming its bounds.

    One launch folds both ranges (``kernels/graph.signed_edge_bounds``) and one read brings them
    back, on the error path only.
    """
    m = int(edges.shape[0])
    bounds = _launch.full(4, INT32_MAX, dtype=wp.int32, device=edges.device)
    _launch.launch_tiled(
        kernel_graph.signed_edge_bounds,
        dim=[kernel_reduce.blocks_1d(m)],
        inputs=[edges, signs, bounds],
        block_dim=TILE_1D,
        device=edges.device,
    )
    # The bounds name the offending range in the message.
    lowest, highest_flip, lowest_sign, highest_sign_flip = read_values(bounds, 0, 4)
    highest, highest_sign = ~highest_flip, ~highest_sign_flip
    if lowest < 0 or highest >= node_count:
        raise ValueError(
            f"edge indices must lie in [0, {node_count}), got min={lowest} max={highest}"
        )
    raise ValueError(f"signs must be 0 or 1, got min={lowest_sign} max={highest_sign}")


def successor_cycles(
    edges: odt.Array2dInt32, node_count: int, *, validate: bool = True
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every cycle of a successor graph in traversal order, packed flat plus per-cycle offsets.

    The edges of a **successor graph** — one in which each node has at most one outgoing edge —
    decompose into node-disjoint cycles and chains. This orders every node along its cycle,
    following the edge direction from the cycle's smallest node index, with no per-cycle Python
    and no per-cycle allocation. The ranking is pointer jumping (Wyllie's list ranking), so the
    work is ``O(m log m)`` over ``m`` edges rather than the quadratic per-node successor walk. A
    mesh boundary's oriented edges are the motivating input (see
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets]).

    Parameters
    ----------
    edges
        ``(m, 2)`` directed edges; row ``(a, b)`` makes ``b`` the successor of
        ``a``. At most one out-edge per node. Nodes appearing in no edge belong to no cycle and
        do not appear in the result.
    node_count
        Number of nodes ``0 .. node_count - 1``.
    validate
        When ``False``, do not raise on an endpoint outside ``[0, node_count)``: the edge is
        dropped and the cycles of the others are returned. The check costs no launch or
        synchronization of its own either way.

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
    caller's job to pass a true successor graph if exact cycles are required. A node with several
    out-edges keeps one of them.

    See Also
    --------
    [`successor_cycles_from_table`][ordito.graph.successor_cycles_from_table]
        The same cycles from a successor table and its tails.
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets]

    Examples
    --------
    Two cycles, ``1 -> 2 -> 3 -> 1`` and ``4 -> 5 -> 4``; the chain ``6 -> 7`` is no cycle:

    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[2, 3], [1, 2], [5, 4], [3, 1], [4, 5], [6, 7]], dtype=wp.int32)
    flat_cycles, offsets = od.graph.successor_cycles(edges, 8)
    print(flat_cycles.numpy().tolist())  # [1, 2, 3, 4, 5]
    print(offsets.numpy().tolist())  # [0, 3, 5]
    ```
    """
    # The range check is ``scatter_successor_tails``' own: that kernel indexes ``next_node`` by the
    # raw edge endpoints, so it skips a row with an endpoint out of range rather than writing past
    # a ``node_count``-element buffer (on the CPU device a host-heap overwrite, surfacing later as a
    # glibc abort somewhere unrelated), and flags it for the readback of the ranking's totals.
    node_count = _validate_edge_list(edges, node_count)

    device = edges.device
    m = int(edges.shape[0])
    if m == 0 or node_count == 0:
        if validate and m > 0:
            _raise_edge_range(edges, node_count)
        return _launch.empty_packed(wp.int32, device)

    # The successor table and its tails in one pass over the edges, the tails in node space so the
    # ranking walks the nodes in ascending order; a node with several out-edges (malformed input)
    # is listed once, so the ranking places every node at most once.
    # Both node-sized tables are halves of one ``-1``-filled buffer: one allocation and one fill.
    table = _launch.full(2 * node_count, -1, dtype=wp.int32, device=device)
    next_node = _launch.view(table, 0, node_count)
    tails = _launch.view(table, node_count, 2 * node_count)
    # The ranking's cycle counts with one zeroed entry past them, where the scatter flags an edge
    # it skipped for an endpoint out of range; the flag comes back with the counts' totals.
    cycle_counts = _launch.zeros(node_count + 1, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_graph.scatter_successor_tails,
        dim=m,
        inputs=[edges, next_node, tails, _launch.view(cycle_counts, node_count, node_count + 1)],
        device=device,
    )
    # Zero-initialised: colliding ranks on malformed input (see Notes) can leave slots unwritten,
    # and zero is a valid node index.
    # A cycle has at most ``m`` nodes, which bounds the pointer-jumping rounds.
    return _ranked_successor_cycles(
        tails,
        next_node,
        kernel_graph.CYCLE_NODES,
        None,
        min(m, node_count),
        zero_unwritten=True,
        cycle_counts=cycle_counts,
        on_flag=(lambda: _raise_edge_range(edges, node_count)) if validate else None,
    )


def successor_cycles_from_table(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    *,
    paired: bool = False,
    node_values: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every cycle of a successor graph given as a successor table, packed flat plus offsets.

    [`successor_cycles`][ordito.graph.successor_cycles]' answer — each cycle from its smallest
    node in edge direction, the cycles in ascending order of it — for a caller that already holds
    the successor table and the list of nodes with an edge out (its *tails*), so no edge list is
    scattered. Chains, walks that reach a node with no successor, are dropped. The boundary walks
    of [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] rank their
    vertices, halfedges and darts through it.

    Parameters
    ----------
    tails
        ``(m,)`` distinct nodes that have a successor, in any order. An entry of ``-1`` is
        skipped.
    next_node
        ``(node_count,)`` successor of every node, ``-1`` where a node has none. Every successor
        must be ``-1`` or below ``node_count``, and no two tails may share a successor.
    paired
        Nodes ``2 * v`` and ``2 * v + 1`` are the two directions through ``v``, so every cycle
        has a mirror over the partner nodes: keep the one of each pair whose start is even, and
        write node ``2 * v + s`` out as ``v``.
    node_values
        ``(node_count,)`` value written for each ranked node in place of the node (after the
        ``paired`` halving), as the origin vertex of a halfedge. ``None`` writes the nodes.

    Returns
    -------
    flat_cycles : wp.array[wp.int32]
        ``(n_cycle_nodes,)`` concatenated ordered nodes (or their values) of every cycle, on
        ``tails.device``.
    offsets : wp.array[wp.int32]
        ``(n_cycles + 1,)`` total-terminated offsets: cycle ``i`` occupies
        ``flat_cycles[offsets[i] : offsets[i + 1]]``; ``[0]`` when there is no cycle.

    Raises
    ------
    RuntimeError
        If ``tails``, ``next_node`` and ``node_values`` are not all on one device.

    See Also
    --------
    [`successor_cycles`][ordito.graph.successor_cycles]
        The same cycles from an edge list.

    Examples
    --------
    The cycles ``0 -> 2 -> 1 -> 0`` and ``3 -> 4 -> 3``; node ``5`` has no successor:

    ```python
    import warp as wp

    import ordito as od

    tails = wp.array([4, 0, 1, 2, 3], dtype=wp.int32)
    next_node = wp.array([2, 0, 1, 4, 3, -1], dtype=wp.int32)
    flat_cycles, offsets = od.graph.successor_cycles_from_table(tails, next_node)
    print(flat_cycles.numpy().tolist())  # [0, 2, 1, 3, 4]
    print(offsets.numpy().tolist())  # [0, 3, 5]
    ```
    """
    require_same_device(tails=tails, next_node=next_node, node_values=node_values)
    if tails.size == 0 or next_node.size == 0:
        return _launch.empty_packed(wp.int32, tails.device)
    mode = kernel_graph.CYCLE_PAIRED if paired else kernel_graph.CYCLE_NODES
    return _ranked_successor_cycles(
        tails, next_node, mode, node_values, tails.size, zero_unwritten=False
    )


def _ranked_successor_cycles(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    mode: int,
    node_values: wp.array[wp.int32] | None,
    max_length: int,
    *,
    zero_unwritten: bool,
    cycle_counts: wp.array[wp.vec2i] | None = None,
    on_flag: Callable[[], None] | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Rank the cycles through ``tails``, the one implementation behind both successor-cycle forms.

    Pointer jumping finds each node's cycle start (its smallest node) and its distance to it
    together, several pointers a launch (``kernels/graph.pointer_jump_schedule``), and one
    inclusive scan over the cycle starts places every cycle. One readback, of the cycle and node
    counts. ``mode`` (``kernels/graph.CYCLE_*``) says which cycles are kept and what each ranked
    node is written out as; ``max_length`` bounds the longest cycle, which sets the round count;
    ``zero_unwritten`` zeroes the packed buffer for input on which ranks can collide.
    ``cycle_counts`` is the zeroed ``(node_count + 1,)`` count buffer when the caller holds it,
    its last entry a flag the caller's scatter raised; ``on_flag`` is called when that flag is set,
    read in the same readback as the totals.
    """
    device = tails.device
    m = tails.size
    node_count = next_node.size
    if (
        mode == kernel_graph.CYCLE_NODES
        and cycle_counts is None
        and node_count >= _COMPACT_SUCCESSORS_FROM
        and node_count >= _COMPACT_SUCCESSORS_RATIO * m
    ):
        tails, next_node, node_values = _compact_successor_table(tails, next_node, node_values)
        node_count = next_node.size
    hops, rounds = kernel_graph.pointer_jump_schedule(max_length)
    # The ranking's windows and, past one round, their ping-pong partner, and per cycle start
    # ``(1, length)``, accumulated by the last round and scanned in place: one allocation.
    # The CPU device runs every round in one thread (``closed_cycle_windows_serial``), the second
    # half of the pair its walk scratch.
    serial = not wp.get_device(device).is_cuda
    pair_size = node_count if rounds == 1 and not serial else 2 * node_count
    flagged = cycle_counts
    if cycle_counts is None:
        pair, cycle_counts = _launch.carve_typed(
            ((wp.vec3i, pair_size), (wp.vec2i, node_count)), device
        )
        _launch.zero_(cycle_counts)
    else:
        pair = _launch.empty(pair_size, dtype=wp.vec3i, device=device)
        cycle_counts = _launch.view(cycle_counts, 0, node_count)
    windows = _launch.view(pair, 0, node_count)
    if serial:
        _launch.launch(
            kernel_graph.closed_cycle_windows_serial,
            dim=1,
            inputs=[
                tails,
                next_node,
                mode,
                windows,
                _launch.view(pair, node_count, 2 * node_count),
            ],
            outputs=[cycle_counts],
            device=device,
        )
    else:
        windows = _closed_cycle_rounds(
            tails, next_node, mode, pair, cycle_counts, node_count, hops, rounds
        )
        _launch.array_scan(cycle_counts, out_array=cycle_counts, inclusive=True)
    # Sizes both outputs: the cycle count and the ranked node count, in one read, with the
    # caller's flag beside them when it has one.
    if flagged is None:
        total = read_scalar(cycle_counts)
    else:
        total, flag = read_values(flagged, node_count - 1, 2)
        if on_flag is not None and int(flag[0]) != 0:
            on_flag()
    n_cycles, n_ranked = int(total[0]), int(total[1])
    if n_cycles == 0:
        return _launch.empty_packed(wp.int32, device)
    # Both outputs in one allocation, the offsets after the nodes. Every offset, the leading zero
    # and the total included, is written by the cycle it bounds.
    allocate = _launch.zeros if zero_unwritten else _launch.empty
    packed = allocate(n_ranked + n_cycles + 1, dtype=wp.int32, device=device)
    flat = _launch.view(packed, 0, n_ranked)
    offsets = _launch.view(packed, n_ranked, n_ranked + n_cycles + 1)
    _launch.launch(
        kernel_graph.scatter_closed_cycles,
        dim=m,
        inputs=[tails, windows, cycle_counts, mode, node_values, flat, offsets],
        device=device,
    )
    return flat, offsets


# A successor table this long, and this many times its listed nodes, is ranked over those nodes
# alone (``_compact_successor_table``): the ranking's windows and cycle counts are ``32`` bytes a
# table node, and a boundary walk lists a few thousand of a mesh's millions of halfedges. Below
# the length the four launches the compaction adds outweigh what it saves.
_COMPACT_SUCCESSORS_FROM = 1 << 22
_COMPACT_SUCCESSORS_RATIO = 4


def _compact_successor_table(
    tails: wp.array[wp.int32], next_node: wp.array[wp.int32], node_values: wp.array[wp.int32] | None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Restate a successor table over its listed nodes, numbered by rank, with the values to write.

    Returns the renumbered ``tails``, the ``(m,)`` table over them and each rank's output value
    (``node_values`` of its node, or the node), so ``CYCLE_NODES`` ranking of the result writes
    what ranking the original would, in the same order: a rank is monotone in its node
    (``kernels/graph.compact_successor_table``).
    """
    device = tails.device
    m = tails.size
    node_count = next_node.size
    n_words = (node_count + 31) // 32
    bits, word_ranks = _launch.carve_typed(
        ((wp.uint32, n_words), (wp.int32, n_words)), device, zeroed=True
    )
    _launch.launch(
        kernel_graph.mark_tail_bits,
        dim=m,
        inputs=[tails],
        outputs=[bits, word_ranks],
        device=device,
    )
    # The set bits' word counts, scanned in place into ``bit_rank``'s table.
    _launch.array_scan(word_ranks, word_ranks, inclusive=True)
    compact_tails, compact_next, values = _launch.carve(wp.int32, [m, m, m], device)
    _launch.launch(
        kernel_graph.compact_successor_table,
        dim=m,
        inputs=[tails, next_node, bits, word_ranks, node_values],
        outputs=[compact_tails, compact_next, values],
        device=device,
    )
    return compact_tails, compact_next, values


def _closed_cycle_rounds(
    tails: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    mode: int,
    pair: wp.array[wp.vec3i],
    cycle_counts: wp.array[wp.vec2i],
    node_count: int,
    hops: int,
    rounds: int,
) -> wp.array[wp.vec3i]:
    """
    Run ``_ranked_successor_cycles``' pointer-jumping rounds, returning the finished windows.

    ``pair`` holds the windows and, past one round, their ping-pong partner; the last round
    accumulates ``cycle_counts``.
    """
    device = tails.device
    m = tails.size
    windows = _launch.view(pair, 0, node_count)
    _launch.launch(
        kernel_graph.closed_cycle_windows,
        dim=m,
        inputs=[tails, next_node, hops, mode, windows, cycle_counts if rounds == 1 else None],
        device=device,
    )
    if rounds > 1:
        spare = _launch.view(pair, node_count, 2 * node_count)
        window = hops
        for r in range(1, rounds):
            _launch.launch(
                kernel_graph.closed_cycle_jump,
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
    return windows


def shortest_path_envelope(
    adjacency: odt.BsrMatrix[wp.float32],
    values: wp.array[wp.float32],
    max_iterations: int = 0,
    *,
    order: wp.array[wp.int32] | None = None,
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
        [`edges_to_csr`][ordito.graph.edges_to_csr] builds with its ``weights`` argument.
        Negative weights are not admissible: the iteration would not terminate at the envelope.
    values
        ``(node_count,)`` initial labels. Not modified.
    max_iterations
        Cap on relaxation passes. Each pass propagates one edge further, so the number needed is the
        graph diameter of the region that violates the bound. ``0`` (the default) means
        ``node_count``, which can never be exceeded.
    order
        ``(node_count,)`` every node once, in an order whose consecutive nodes lie close together
        in the graph, such as the vertices sorted along a space-filling curve. Trusted, not
        checked: a node it misses is never relaxed. On a CUDA device without ``max_iterations``,
        the passes after the first then relax patches of consecutive nodes several times over, so
        a label crosses a patch in a pass. That pays on a field many passes deep, a distance from
        a few sources; on one a few passes settle, the extra rounds can cost more than they save.
        Elsewhere it is unused. The answer is identical either way.

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

    Notes
    -----
    **The answer is Dijkstra's; the method is Bellman-Ford.** A priority queue is inherently serial
    -- it processes one node per pop -- so this relaxes *every* node against its neighbours in
    parallel and repeats until nothing improves. Shortest-path distances are unique, so the two
    agree on the result; what differs is the cost model -- ``O(diameter)`` launches over the whole
    CSR here against ``O(E log V)`` sequential work there. The name is the result, per this
    package's naming rule, not the algorithm.

    The answer does not depend on thread interleaving. Uncapped, each pass relaxes the labels in
    place, reading whatever its neighbours hold; every value read is an upper bound the relaxation
    reaches anyway, so the labels fall to the same fixpoint bit for bit, in no more passes. With
    ``max_iterations`` each pass is a pure function of the previous labels, so a capped answer is
    exactly that many passes. The pass count is data-dependent; on CUDA the passes after the first
    run as one device-side conditional graph (``wp.capture_while``), so the convergence check costs
    one host readback, after the first pass, rather than one a pass. The CPU device runs every pass
    in one launch.

    For distance *across* a surface rather than along its edges -- shorter, and what "geodesic"
    usually means -- use [`heat_geodesic`][ordito.heat.heat_geodesic]. The edge-graph distance is
    an upper bound on it.

    See Also
    --------
    [`edges_to_csr`][ordito.graph.edges_to_csr]
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`ordito.remesh.isotropic_remesh`][ordito.remesh.isotropic_remesh]
    [`scipy.sparse.csgraph.dijkstra`][]

    Examples
    --------
    Shortest-path distance from one corner of a unit grid, along its edges weighted by length. The
    grid's diagonals run corner to corner, so the far corner is at distance ``sqrt(2)``:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    n_vertices = vertices.shape[0]
    edges, _ = od.edges.edges_unique(faces)
    lengths = od.edges.edges_unique_length(vertices, faces, edges)
    adjacency = od.graph.edges_to_csr(n_vertices, edges, lengths)
    seed = wp.full(n_vertices, 1.0e6, dtype=wp.float32)  # "unreached"
    seed[:1].fill_(0.0)  # the source: vertex 0 at distance 0
    distance = od.graph.shortest_path_envelope(adjacency, seed)
    print(f"{distance.numpy()[-1]:.4f}")  # 1.4142
    ```

    As a Lipschitz cap on a path graph with unit weights, a step of 5 is lowered to a slope of 1:

    ```python
    import warp as wp

    import ordito as od

    edges = wp.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=wp.int32)
    adjacency = od.graph.edges_to_csr(5, edges)
    values = wp.array([0.0, 5.0, 5.0, 5.0, 5.0], dtype=wp.float32)
    capped = od.graph.shortest_path_envelope(adjacency, values)
    print(capped.numpy().tolist())  # [0.0, 1.0, 2.0, 3.0, 4.0]
    ```
    """
    require_same_device(adjacency=adjacency, values=values, order=order)
    node_count, offsets, columns = _validate_square_csr(adjacency)
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")
    if values.size != node_count:
        raise ValueError(
            f"values must have one entry per node, got {values.size} for {node_count} nodes"
        )
    if order is not None and order.size != node_count:
        raise ValueError(
            f"order must list every node once, got {order.size} entries for {node_count} nodes"
        )

    device = wp.get_device(values.device)
    labels = _launch.clone(values)
    if node_count == 0:
        return labels

    weights = adjacency.values
    max_pass_count = max_iterations or node_count
    # Uncapped, the passes relax ``labels`` in place: the same fixpoint bit for bit, in no more
    # passes and with no copy a pass (``kernels/graph.shortest_path_envelope_pass``). A cap keeps
    # the Jacobi passes, whose second buffer starts as the labels, so a capped answer is exactly
    # that many passes on either device.
    relaxed = labels if max_iterations == 0 else _launch.clone(values)
    # The shared round-loop word (``kernels/array.py``) and the envelope's negative-weight flag past
    # it. ``wp.capture_while`` reads the condition before the first round, so it starts non-zero.
    state = _launch.array([0, 1, 0, 0], dtype=wp.int32, device=device)
    if not device.is_cuda:
        # The CPU device runs a launch's threads one after another, so a launch and a progress
        # readback a pass buy nothing: one single-thread launch, uncapped a few sweeps and then a
        # Dijkstra over what still moves (``kernels/graph.shortest_path_envelope_dijkstra``),
        # capped every pass, bit for bit the per-pass loop's labels
        # (``kernels/graph.shortest_path_envelope_serial``).
        if relaxed is labels:
            heap, position = _launch.carve(wp.int32, (node_count, node_count), device)
            _launch.launch(
                kernel_graph.shortest_path_envelope_dijkstra,
                dim=1,
                inputs=[
                    offsets,
                    columns,
                    weights,
                    kernel_graph.ENVELOPE_CPU_SWEEPS,
                    labels,
                    heap,
                    position,
                    state,
                ],
                device=device,
            )
        else:
            _launch.launch(
                kernel_graph.shortest_path_envelope_serial,
                dim=1,
                inputs=[offsets, columns, weights, max_pass_count, labels, relaxed, state],
                device=device,
            )
        # Decides whether to raise.
        if read_scalar(state, int(kernel_graph.ENVELOPE_NEGATIVE_WEIGHT)) != 0:
            _raise_negative_weights()
        return labels
    max_pass_count_i32 = wp.int32(max_pass_count)
    # With an ``order``, the captured passes relax patches of it a block, several rounds a pass
    # (``kernels/graph.shortest_path_envelope_patch_pass``); the first pass stays a plain pull, so a
    # field it all but settles pays for no patch rounds.
    patches = [False]

    def envelope_pass_body() -> None:
        # One pass, then (Jacobi) a copy of its result back over ``labels``, then the shared
        # round-closing kernel, which turns the pass's progress and the cap into the condition.
        # **Two Jacobi passes per round, written into each other's buffer, would remove the copy**
        # and are **not portable**: measured with that body, the CPU device relaxed strictly fewer
        # nodes per round than CUDA at every cap, because the recorded body did not replay as two
        # passes per round there. A ``wp.capture_while`` body is not guaranteed to execute as an
        # indivisible unit across devices, so a loop whose *result buffer* depends on the body
        # running whole cannot rely on it. A Python-level ping-pong cannot help either: the body is
        # recorded once and replayed, so rebinding the names would only take effect at record time.
        if patches[0]:
            _launch.launch_tiled(
                kernel_graph.shortest_path_envelope_patch_pass,
                dim=[-(-node_count // _ENVELOPE_PATCH_LANES)],
                inputs=[offsets, columns, weights, order, kernel_graph.ENVELOPE_PATCH_ROUNDS],
                outputs=[labels, state],
                block_dim=_ENVELOPE_PATCH_LANES,
                device=device,
            )
        else:
            _launch.launch(
                kernel_graph.shortest_path_envelope_pass,
                dim=node_count,
                inputs=[offsets, columns, weights, labels, relaxed, state],
                device=device,
            )
        if relaxed is not labels:
            _launch.copy(labels, relaxed)
        _launch.launch(
            kernel_array.loop_advance, dim=1, inputs=[max_pass_count_i32, state], device=device
        )

    # The first pass runs uncaptured and its read is the call's one readback: it carries the
    # negative-weight flag (a negative weight would keep the labels falling until the cap) and the
    # loop's condition, so a field that one pass leaves unmoved records no graph at all.
    envelope_pass_body()
    _, condition, _, negative = read_values(state, 0, kernel_graph.ENVELOPE_STATE_SIZE)
    if negative != 0:
        _raise_negative_weights()
    # The rest of the passes run on-device via ``wp.capture_while``, so their convergence checks
    # and the cap cost no host readback.
    patches[0] = order is not None and relaxed is labels
    if condition != 0:
        run_device_loop(
            device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), envelope_pass_body
        )
    return labels


# Lanes of a ``shortest_path_envelope_patch_pass`` block, a patch: ``_launch.launch``'s width,
# which a tiled launch of the module must share or build it again.
_ENVELOPE_PATCH_LANES = 256


def _raise_negative_weights() -> None:
    """Raise the error for a negative weight the envelope's first pass flagged."""
    raise ValueError("adjacency weights must be non-negative for the envelope to converge")


# --- private helpers ---------------------------------------------------------------------


def _validate_square_csr(
    adjacency: odt.BsrMatrix[wp.Scalar],
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


def _validate_edge_list(edges: odt.Array2dInt32, node_count: int | None) -> int:
    """
    Check an ``(m, 2)`` edge list's shape and resolve its node count.

    The shared entry check of every function here that takes an edge list instead of a prebuilt
    adjacency. When ``node_count`` is ``None`` it is inferred from the edges; when it is supplied
    it is checked for sign. The range of the edge indices is checked by each caller's first kernel
    that indexes by them, which skips an out-of-range row and flags it
    (``_raise_edge_range`` names the bounds), so no reduction of the edges runs on the path that
    does not raise.
    """
    odt.ensure_edge_pairs(edges, "edges")

    if node_count is None:
        return int(od.array.index_bound(edges))
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    return node_count


def _raise_edge_range(edges: odt.Array2dInt32, node_count: int) -> None:
    """
    Raise the range error for an edge list a checked kernel flagged, naming its bounds.

    Only the error path reduces the edges: the flag a checked kernel raised says that some
    endpoint lies outside ``[0, node_count)``, and this measures which.
    """
    # The bounds name the offending range in the message.
    lowest, highest = od.reduce.minmax(edges)
    raise ValueError(f"edge indices must lie in [0, {node_count}), got min={lowest} max={highest}")
