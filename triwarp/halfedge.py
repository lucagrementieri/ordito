"""
Implicit halfedge connectivity: edge twins and counter-clockwise vertex one-rings.

There is no halfedge *structure* here, only an indexing convention over the flat face buffer every
other module already uses. Halfedge ``h = 3 * f + k`` runs from ``faces[3f + k]`` to
``faces[3f + (k + 1) % 3]``, so ``next`` and ``prev`` are index arithmetic and the halfedges of a
face are consecutive. Exactly two arrays are needed to navigate a mesh:
[`halfedge_twins`][triwarp.halfedge.halfedge_twins] to cross an edge, and
[`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] to rotate around a vertex.

This is the ordering the tangent-space machinery in [`triwarp.tangent_space`][triwarp.tangent_space]
is built on: a rotational order of the outgoing halfedges at a vertex is what turns per-corner
angles into a polar coordinate system on the tangent plane.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

import triwarp as tw
from triwarp import _launch
from triwarp._device import read_scalar, read_values, require_same_device
from triwarp.constants import INT32_MAX
from triwarp.kernels import halfedge as kernel_halfedge


def halfedge_twins(
    faces: wp.array[wp.int32], n_vertices: int | None = None, *, validate: bool = True
) -> wp.array[wp.int32]:
    """
    Opposite halfedge of every halfedge, or ``-1`` on a boundary.

    Halfedges ``h = 3 * f + k`` and ``twins[h]`` traverse the same undirected edge in opposite
    directions, so ``twins[twins[h]] == h`` wherever ``twins[h] != -1``. The halfedges left at
    ``-1`` are exactly the mesh boundary, in the orientation
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges] reports.

    The *opposite directions* half of that is a precondition on the mesh and not merely a property
    of the output: it holds only where the two faces meeting at an edge are wound consistently, so
    an inconsistently wound or non-orientable mesh is rejected rather than paired up. Every
    consumer of this table -- the counter-clockwise rotation
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] walks above all -- reads a twin as "the
    same edge, seen from the other side", and there is no other side to see when both halfedges
    face the same way.

    Undirected edges are matched by packing each sorted endpoint pair into one key
    ([`hash_indices_rows`][triwarp.grouping.hash_indices_rows]), radix-sorting the keys with the
    halfedge index as payload, and pairing up the runs of equal keys — the same mechanism behind
    [`edges_unique`][triwarp.edges.edges_unique] and
    [`face_adjacency`][triwarp.adjacency.face_adjacency], without materializing the unique edge
    list.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    n_vertices
        Total vertex count. It is only ever the key radix here, so passing it is an optimization
        rather than a requirement: when ``None`` the keys pack against
        [`constants.INDEX_RADIX_PAIR`][triwarp.constants.INDEX_RADIX_PAIR], which bounds every
        ``int32`` index without a reduction and orders the keys the same way.
    validate
        When ``True`` (the default), reject the two meshes below. ``False`` skips the check and
        the synchronization it costs, for a caller that knows ``faces`` is edge-manifold and
        consistently wound -- one that built it from a buffer already validated here by an
        operation that preserves both; on a mesh that is neither the table is garbage.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` twin halfedge indices on ``faces.device``; ``-1`` for boundary
        halfedges.

    Raises
    ------
    ValueError
        If an undirected edge carries three or more halfedges (an edge-non-manifold mesh, where
        "the" opposite halfedge is not defined), or if both halfedges of an edge traverse it in the
        *same* direction, which is what an inconsistently wound or non-orientable mesh looks like
        from here and which leaves the "opposite directions" guarantee above with nothing to mean.
        Detecting either needs one 8-byte readback, so under ``validate`` this function
        synchronizes once.

    See Also
    --------
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]
    """
    defect_counts = _launch.zeros(2, dtype=wp.int32, device=faces.device)
    twins = _pair_halfedges(faces, n_vertices, defect_counts)
    if validate and twins.size > 0:
        _raise_twin_defects(read_values(defect_counts, 0, 2))
    return twins


def require_matching_twins(faces: wp.array[wp.int32], twins: wp.array[wp.int32] | None) -> None:
    """
    Raise unless a precomputed twin table really is one for ``faces``.

    The contract behind every ``twins=`` keyword in the package, checked in both of its halves. The
    table is indexed *by halfedge* (``h = 3 * f + k``), so it is meaningful only for the face buffer
    it was built from: a table cached from a smaller mesh is not merely stale -- it is short, and
    the kernels that walk it index past its end, which on the CPU device reads the host heap rather
    than raising. And each entry must be the *opposite* halfedge, since that is what every consumer
    reads it as -- the counter-clockwise rotation
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] walks, the transport angle
    [`triwarp.tangent_space`][triwarp.tangent_space] pairs across an edge, and the dual edge
    [`triwarp.selection`][triwarp.selection] floods through. Public because those callers live in
    three modules and must all reject the same table the same way; only the first reaches the
    rotation, so a check placed in that walk would leave the other two unguarded.

    What it checks is every entry that *is* present; what it cannot check is an entry that is
    absent. A ``-1`` claims "this halfedge has no twin", and deciding whether that is true means
    finding out whether another halfedge spans the same edge -- which is the sort
    [`halfedge_twins`][triwarp.halfedge.halfedge_twins] does and the work ``twins=`` exists to skip,
    so demanding it here would make the keyword pointless. A table of nothing but ``-1`` therefore
    passes; it is not silent, because a fabricated boundary shortens the fan and
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] then raises on the ring it could not
    complete.

    The length half is free. The structural half costs one launch over the halfedges and one
    readback, and is only ever paid when a table was actually supplied -- the path a caller takes to
    skip an edge build, a hash, a radix sort, a launch and a readback, so verifying the shortcut
    stays a fraction of what taking it saved. A table this package produced can never fail it, since
    ``halfedge_twins`` establishes the property by construction.

    The device half of the same contract is the caller's own ``require_same_device`` call, which
    covers every argument it received rather than this pair alone.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    twins
        ``(3 * n_faces,)`` candidate [`halfedge_twins`][triwarp.halfedge.halfedge_twins] table, or
        ``None``.

    Raises
    ------
    ValueError
        If ``twins`` is given and its length is not ``3 * n_faces``, or if any of its entries is
        not the opposite halfedge of its own index. ``-1`` is a boundary halfedge and is always
        accepted.

    See Also
    --------
    [`halfedge_twins`][triwarp.halfedge.halfedge_twins]
        Produces the table.
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings]
    """
    if twins is None:
        return
    n_halfedges = faces.size // 3 * 3
    if twins.size != n_halfedges:
        raise ValueError(
            f"twins must have one entry per halfedge, got {twins.size} for {n_halfedges} "
            f"halfedges ({n_halfedges // 3} faces)"
        )
    if n_halfedges == 0:
        return
    device = faces.device
    mispaired = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_halfedge.count_mispaired_twins,
        dim=n_halfedges,
        inputs=[faces, twins, mispaired],
        device=device,
    )
    # Unavoidable: the count only exists on the device, and the whole point is to raise on it
    # before a consumer walks the table.
    n_mispaired = int(read_scalar(mispaired, 0))
    if n_mispaired > 0:
        raise ValueError(
            f"twins must hold the opposite halfedge of each halfedge of faces: {n_mispaired} "
            f"entry/entries do not. A twin must run back along the same edge (so "
            f"twins[twins[h]] == h and the two endpoints swap), or be -1 on a boundary. Build the "
            f"table with halfedge_twins for the same face buffer."
        )


def vertex_one_rings(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32] | None = None,
    n_vertices: int | None = None,
    *,
    validate: bool = True,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Outgoing halfedges of every vertex in counter-clockwise order, as a CSR buffer.

    Returned values first and offsets second, the package's packed-buffer convention -- see
    [`array.pack_1d_arrays`][triwarp.array.pack_1d_arrays], which states it.

    Vertex ``v`` owns ``ring_halfedges[offsets[v] : offsets[v + 1]]``, each entry a halfedge leaving
    ``v``; the ring size is therefore the number of incident *faces*, one less than the number of
    adjacent vertices at a boundary vertex. Consecutive entries are consecutive around ``v`` in the
    direction the face orientation calls counter-clockwise, because the rotation ``h ->
    twins[prev(h)]`` steps from edge ``v -> a`` to edge ``v -> b`` inside the CCW-oriented face
    ``(v, a, b)``.

    A boundary vertex starts its ring at its outgoing boundary halfedge (the clockwise-most edge of
    its fan) so the whole fan is enumerated; interior vertices start at their lowest-indexed
    outgoing halfedge, which makes the rotational *order* canonical but its starting point
    arbitrary. Isolated vertices get an empty row.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins]. When
        ``None`` it is computed here.
    n_vertices
        Total vertex count (the number of CSR rows). When ``None`` it is inferred with
        [`array.index_bound`][triwarp.array.index_bound], which costs a host readback.
    validate
        When ``True`` (the default), reject a pinched vertex, and forward the same choice to
        [`halfedge_twins`][triwarp.halfedge.halfedge_twins] when ``twins`` is computed here.
        ``False`` skips both checks and their synchronizations, for a caller that knows ``faces``
        is manifold and consistently wound.

    Returns
    -------
    ring_halfedges : wp.array[wp.int32]
        ``(3 * n_faces,)`` outgoing halfedges, grouped and ordered per vertex.
    offsets : wp.array[wp.int32]
        ``(n_vertices + 1,)`` CSR row starts; ``offsets[-1] == 3 * n_faces``.
    is_boundary : wp.array[wp.bool]
        ``(n_vertices,)`` flags; ``True`` where the vertex is incident to a boundary edge.

    Raises
    ------
    ValueError
        If ``twins`` is given and is not a twin table for ``faces``
        ([`require_matching_twins`][triwarp.halfedge.require_matching_twins] states what that
        means), or if a vertex's rotation closes before its whole fan is covered — a pinched,
        vertex-non-manifold vertex where two fans meet at a single index, under ``validate``.
        Detecting the latter needs one 4-byte readback, so under ``validate`` this function
        synchronizes once.
    RuntimeError
        If ``faces`` and ``twins`` are not all on one device.

    See Also
    --------
    [`halfedge_twins`][triwarp.halfedge.halfedge_twins]
    [`require_matching_twins`][triwarp.halfedge.require_matching_twins]
    [`halfedge_tangent_angles`][triwarp.tangent_space.halfedge_tangent_angles]
    """
    require_same_device(faces=faces, twins=twins)
    require_matching_twins(faces, twins)
    device = faces.device
    n_halfedges = faces.size // 3 * 3

    if n_vertices is None:
        n_vertices = tw.array.index_bound(faces)
    # Slots 0-1 are the twin table's two rejections when this call derives it, slot 2 the pinch
    # count below: one buffer, so a validated call reads all three back once rather than once for
    # the twins and again for the rings. The twin rejections are still raised first.
    defect_counts = _launch.zeros(3, dtype=wp.int32, device=device)
    check_twins = twins is None and validate
    if twins is None:
        # The pairing writes only the first two slots, so it takes the buffer whole.
        twins = _pair_halfedges(faces, n_vertices, defect_counts)

    offsets = _launch.zeros(n_vertices + 1, dtype=wp.int32, device=device)
    ring_halfedges = _launch.full(n_halfedges, -1, dtype=wp.int32, device=device)
    if n_halfedges == 0 or n_vertices == 0:
        if check_twins and n_halfedges > 0:
            _raise_twin_defects(read_values(defect_counts, 0, 2))
        return ring_halfedges, offsets, _launch.zeros(n_vertices, dtype=wp.bool, device=device)

    # One pass over the halfedges sizes the CSR and picks every vertex's two start candidates (row
    # 0 over all outgoing halfedges, row 1 over the boundary ones): every face contributes exactly
    # one outgoing halfedge per corner, so a vertex's ring size is how often it appears in the
    # flat face buffer -- no walk needed. The degrees are counted straight into ``offsets[1:]``,
    # already zeroed, and scanned there in place.
    counts = offsets[1:]
    candidate_starts = _launch.full((2, n_vertices), INT32_MAX, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_halfedge.ring_degrees_and_starts,
        dim=n_halfedges,
        inputs=[faces, twins, counts, candidate_starts],
        device=device,
    )
    # The inclusive scan in place leaves the leading zero, giving the usual CSR bounds.
    # Deliberately NOT tw.array.counts_to_offsets: that helper always reads the total back, and
    # this function never needs it (it is n_halfedges, known on the host). Converting for symmetry
    # would add a device synchronization where there is currently none.
    _launch.array_scan(counts, out_array=counts, inclusive=True)

    # The walk resolves each vertex's start and boundary flag from the candidates itself, and
    # writes the flag for every vertex, so ``is_boundary`` needs no initial value.
    is_boundary = _launch.empty(n_vertices, dtype=wp.bool, device=device)
    incomplete = defect_counts[2:3]
    _launch.launch(
        kernel_halfedge.write_one_rings,
        dim=n_vertices,
        inputs=[candidate_starts, twins, offsets, ring_halfedges, is_boundary, incomplete],
        device=device,
    )
    if not validate:
        return ring_halfedges, offsets, is_boundary
    counts_np = read_values(defect_counts, 0, 3)
    if check_twins:
        _raise_twin_defects(counts_np)
    n_incomplete = int(counts_np[2])
    if n_incomplete > 0:
        raise ValueError(
            f"vertex_one_rings requires a vertex-manifold mesh: {n_incomplete} vertex/vertices "
            f"have more than one fan of faces (a pinch point)."
        )
    return ring_halfedges, offsets, is_boundary


def _pair_halfedges(
    faces: wp.array[wp.int32], n_vertices: int | None, defect_counts: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Build the twin table, counting its two rejections into ``defect_counts`` without reading them.

    ``defect_counts`` is a caller-owned, zeroed two-slot buffer, so a caller that validates more
    than the twins (``vertex_one_rings``) can pack its own counter beside them and read all of them
    back at once.
    """
    device = faces.device
    n_halfedges = faces.size // 3 * 3
    twins = _launch.full(n_halfedges, -1, dtype=wp.int32, device=device)
    if n_halfedges == 0:
        return twins

    # Edge keys are built from face indices, so they are non-negative and below the vertex count by
    # construction: the range check would only add a readback. And ``n_vertices`` is the packing
    # radix and nothing else -- no buffer here is sized by it -- so when the caller does not supply
    # one the pair radix serves instead of inferring the tight bound, which would be a device
    # reduction plus a host readback for a fifth of this call. The keys are written straight from
    # the faces (``adjacency.face_edge_keys``), the same keys ``faces_to_edges(sorted=True)`` plus
    # ``hash_indices_rows`` produce without the ``(3F, 2)`` rows between them.
    sorted_keys, order = tw.adjacency.sorted_face_edge_keys(faces, n_vertices=n_vertices)

    # Slot 0 counts edge-non-manifold edges, slot 1 edges whose two halfedges run the same way;
    # one buffer so the two rejections cost one readback between them rather than two.
    _launch.launch(
        kernel_halfedge.pair_sorted_halfedges,
        dim=n_halfedges,
        inputs=[faces, sorted_keys, order, twins, defect_counts],
        device=device,
    )
    return twins


def _raise_twin_defects(defect_counts: Sequence[int]) -> None:
    """Raise ``halfedge_twins``' two rejections from its read-back defect counts."""
    n_nonmanifold, n_misoriented = (int(count) for count in defect_counts[:2])
    if n_nonmanifold > 0:
        raise ValueError(
            f"halfedge_twins requires an edge-manifold mesh: {n_nonmanifold} edge(s) are shared by "
            f"three or more faces."
        )
    if n_misoriented > 0:
        raise ValueError(
            f"halfedge_twins requires a consistently wound mesh: {n_misoriented} edge(s) are "
            f"traversed in the same direction by both of their halfedges, so those two halfedges "
            f"are not opposites of each other. Run make_winding_consistent first; a non-orientable "
            f"surface has no consistent winding and no halfedge twin table at all."
        )
