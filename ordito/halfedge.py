"""
Implicit halfedge connectivity: edge twins and counter-clockwise vertex one-rings.

There is no halfedge *structure* here, only an indexing convention over the flat face buffer every
other module already uses. Halfedge ``h = 3 * f + k`` runs from ``faces[3f + k]`` to
``faces[3f + (k + 1) % 3]``, so ``next`` and ``prev`` are index arithmetic and the halfedges of a
face are consecutive. Exactly two arrays are needed to navigate a mesh:
[`halfedge_twins`][ordito.halfedge.halfedge_twins] to cross an edge, and
[`vertex_one_rings`][ordito.halfedge.vertex_one_rings] to rotate around a vertex.

This is the ordering the tangent-space machinery in [`ordito.tangent_space`][ordito.tangent_space]
is built on: a rotational order of the outgoing halfedges at a vertex is what turns per-corner
angles into a polar coordinate system on the tangent plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.constants import INDEX_RADIX_PAIR, INT32_MAX
from ordito.kernels import halfedge as kernel_halfedge
from ordito.kernels import scatter as kernel_scatter

# Whether a halfedge pairing with a known vertex count goes through per-vertex edge buckets on the
# CPU device too, as it does on CUDA; otherwise the CPU sorts the edge keys. Both build the same
# twins, mates and defect counts; the device split is cost alone (kernels/halfedge.py).
_BUCKETED_PAIRING_ON_CPU = False

# Halfedge count from which ``halfedge_mates`` pairs through the edge buckets rather than the key
# sort. Below it the bucket build's extra launches outweigh the sort it replaces
# (kernels/halfedge.py). The twin table takes the buckets at every size.
_BUCKETED_MATES_FROM_HALFEDGES = 1 << 19


def halfedge_twins(
    faces: wp.array[wp.int32], n_vertices: int | None = None, *, validate: bool = True
) -> wp.array[wp.int32]:
    """
    Opposite halfedge of every halfedge, or ``-1`` on a boundary.

    Halfedges ``h = 3 * f + k`` and ``twins[h]`` traverse the same undirected edge in opposite
    directions, so ``twins[twins[h]] == h`` wherever ``twins[h] != -1``. The halfedges left at
    ``-1`` are exactly the mesh boundary, in the orientation
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges] reports.

    The *opposite directions* half of that is a precondition on the mesh and not merely a property
    of the output: it holds only where the two faces meeting at an edge are wound consistently, so
    an inconsistently wound or non-orientable mesh is rejected rather than paired up. Every
    consumer of this table -- the counter-clockwise rotation
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] walks above all -- reads a twin as "the
    same edge, seen from the other side", and there is no other side to see when both halfedges
    face the same way.

    On a CUDA device with ``n_vertices`` given, undirected edges are matched by a counting sort of
    the halfedges into per-vertex buckets, each edge under its lower-degree endpoint so that a
    high-valence vertex keeps a small bucket, and each halfedge scans its bucket for the others.
    Otherwise each sorted endpoint pair is packed into one key
    ([`hash_indices_rows`][ordito.grouping.hash_indices_rows]), the keys are radix-sorted with the
    halfedge index as payload, and the runs of equal keys are paired up -- the same mechanism
    behind [`edges_unique`][ordito.edges.edges_unique] and
    [`face_adjacency`][ordito.adjacency.face_adjacency], without materializing the unique edge
    list. Both give the same table and reject the same meshes.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    n_vertices
        Total vertex count, which must exceed every index in ``faces``: it sizes the per-vertex
        buckets, or bounds the sort keys. Passing it is an optimization rather than a requirement:
        when ``None`` the keys pack against
        [`constants.INDEX_RADIX_PAIR`][ordito.constants.INDEX_RADIX_PAIR], which bounds every
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
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]
    [`face_adjacency`][ordito.adjacency.face_adjacency]
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges]
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
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] walks, the transport angle
    [`ordito.tangent_space`][ordito.tangent_space] pairs across an edge, and the dual edge
    [`ordito.selection`][ordito.selection] floods through. Public because those callers live in
    three modules and must all reject the same table the same way; only the first reaches the
    rotation, so a check placed in that walk would leave the other two unguarded.

    What it checks is every entry that *is* present; what it cannot check is an entry that is
    absent. A ``-1`` claims "this halfedge has no twin", and deciding whether that is true means
    finding out whether another halfedge spans the same edge -- which is the sort
    [`halfedge_twins`][ordito.halfedge.halfedge_twins] does and the work ``twins=`` exists to skip,
    so demanding it here would make the keyword pointless. A table of nothing but ``-1`` therefore
    passes; it is not silent, because a fabricated boundary shortens the fan and
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] then raises on the ring it could not
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
        ``(3 * n_faces,)`` candidate [`halfedge_twins`][ordito.halfedge.halfedge_twins] table, or
        ``None``.

    Raises
    ------
    ValueError
        If ``twins`` is given and its length is not ``3 * n_faces``, or if any of its entries is
        not the opposite halfedge of its own index. ``-1`` is a boundary halfedge and is always
        accepted.

    See Also
    --------
    [`halfedge_twins`][ordito.halfedge.halfedge_twins]
        Produces the table.
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]
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
    [`array.pack_1d_arrays`][ordito.array.pack_1d_arrays], which states it.

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
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins]. When
        ``None`` it is computed here.
    n_vertices
        Total vertex count (the number of CSR rows). When ``None`` it is inferred with
        [`array.index_bound`][ordito.array.index_bound], which costs a host readback.
    validate
        When ``True`` (the default), reject a pinched vertex, and forward the same choice to
        [`halfedge_twins`][ordito.halfedge.halfedge_twins] when ``twins`` is computed here.
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
        ([`require_matching_twins`][ordito.halfedge.require_matching_twins] states what that
        means), or if a vertex's rotation closes before its whole fan is covered — a pinched,
        vertex-non-manifold vertex where two fans meet at a single index, under ``validate``.
        Detecting the latter needs one 4-byte readback, so under ``validate`` this function
        synchronizes once.
    RuntimeError
        If ``faces`` and ``twins`` are not all on one device.

    See Also
    --------
    [`halfedge_twins`][ordito.halfedge.halfedge_twins]
    [`require_matching_twins`][ordito.halfedge.require_matching_twins]
    [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles]
    """
    require_same_device(faces=faces, twins=twins)
    require_matching_twins(faces, twins)
    device = faces.device
    n_halfedges = faces.size // 3 * 3

    if n_vertices is None:
        n_vertices = od.array.index_bound(faces)
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
    # Deliberately NOT od.array.counts_to_offsets: that helper always reads the total back, and
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


@overload
def halfedge_mates(
    faces: wp.array[wp.int32],
    n_vertices: int | None = None,
    *,
    return_key_order: Literal[False] = False,
) -> wp.array[wp.int32]: ...


@overload
def halfedge_mates(
    faces: wp.array[wp.int32], n_vertices: int | None = None, *, return_key_order: Literal[True]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32] | None]: ...


def halfedge_mates(
    faces: wp.array[wp.int32], n_vertices: int | None = None, *, return_key_order: bool = False
) -> wp.array[wp.int32] | tuple[wp.array[wp.int32], wp.array[wp.int32] | None]:
    """
    Every halfedge's partner on its undirected edge, whatever its direction, or how many share it.

    The unvalidated pairing beneath [`halfedge_twins`][ordito.halfedge.halfedge_twins], for
    consumers that classify edges rather than walk them. Halfedge ``h = 3 * f + k`` gets:

    - the other halfedge (``>= 0``) when exactly two halfedges span its edge, whichever way each
      runs, so a pair wound the same way is a pair here where ``halfedge_twins`` rejects it;
    - ``-1`` when it spans its edge alone (a boundary edge);
    - ``-2 - lowest`` when three or more do, ``lowest`` being the smallest halfedge on that edge,
      so a consumer can act on each such edge once.

    So ``mates[mates[h]] == h`` wherever ``mates[h] >= 0``, and a pair is acted on once by its
    lower halfedge (``h < mates[h]``). Nothing is checked and nothing is read back.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    n_vertices
        Total vertex count, which must exceed every index in ``faces``. When it is given, a CUDA
        device pairs a large mesh's halfedges through per-vertex buckets; otherwise, and on the
        CPU, through a sort of the edge keys, which is correct for any non-negative indices
        without it. Both give the same table.
    return_key_order
        Also return the sort's permutation when the pairing sorted the edge keys, for a caller
        that wants some of the halfedges in key order
        ([`key_ordered_halfedges`][ordito.halfedge.key_ordered_halfedges]).

    Returns
    -------
    mates : wp.array[wp.int32]
        ``(3 * n_faces,)`` mate codes on ``faces.device``.
    key_order : wp.array[wp.int32] | None
        Only with ``return_key_order``: ``(3 * n_faces,)`` halfedge indices in ascending
        undirected key order, ties by index -- the order of
        [`sorted_face_edge_keys`][ordito.adjacency.sorted_face_edge_keys] -- or ``None`` when the
        pairing bucketed instead of sorting.

    See Also
    --------
    [`halfedge_twins`][ordito.halfedge.halfedge_twins]
    [`key_ordered_halfedges`][ordito.halfedge.key_ordered_halfedges]
    [`sorted_face_edge_keys`][ordito.adjacency.sorted_face_edge_keys]
    """
    device = faces.device
    n_halfedges = faces.size // 3 * 3
    mates = _launch.empty(n_halfedges, dtype=wp.int32, device=device)
    key_order: wp.array[wp.int32] | None = None
    if n_halfedges == 0:
        key_order = mates
    elif (
        n_vertices is not None
        and n_halfedges >= _BUCKETED_MATES_FROM_HALFEDGES
        and _buckets_on(faces)
    ):
        degrees, ends, buckets = _edge_buckets(faces, n_vertices)
        _launch.launch(
            kernel_halfedge.bucketed_halfedge_mates,
            dim=n_halfedges,
            inputs=[faces, degrees, ends, buckets, mates],
            device=device,
        )
    else:
        sorted_keys, key_order = od.adjacency.sorted_face_edge_keys(faces, n_vertices=n_vertices)
        _launch.launch(
            kernel_halfedge.sorted_halfedge_mates,
            dim=n_halfedges,
            inputs=[sorted_keys, key_order, mates],
            device=device,
        )
    if return_key_order:
        return mates, key_order
    return mates


def key_ordered_halfedges(
    faces: wp.array[wp.int32],
    inclusive: odt.ArrayNdInt32,
    count: int,
    *,
    key_order: wp.array[wp.int32] | None = None,
    n_vertices: int | None = None,
) -> wp.array[wp.int32]:
    """
    Return the flagged halfedges, class by class, each class in ascending undirected key order.

    The key is the one [`edges_unique`][ordito.edges.edges_unique] orders its rows by, so a
    selection of one halfedge per edge comes back in that function's row order. Halfedges sharing
    a key keep ascending index order. Given the sort's ``key_order``
    ([`halfedge_mates`][ordito.halfedge.halfedge_mates]' second return), the flags are read in
    that order and nothing is sorted; without it only the flagged halfedges are sorted, which is
    cheap when they are few.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    inclusive
        ``(n_classes * 3 * n_faces,)`` inclusive scan of 0/1 flags, row ``c`` of the flattened
        ``(n_classes, 3 * n_faces)`` table flagging the halfedges of class ``c``: entry ``i`` of a
        row flags halfedge ``key_order[i]`` when ``key_order`` is given, halfedge ``i`` otherwise.
        Read, not modified.
    count
        The number of flagged entries, ``inclusive[-1]``, which the caller has already read back
        to size its output.
    key_order
        ``(3 * n_faces,)`` halfedges in ascending key order, ties by index, or ``None``.
    n_vertices
        Total vertex count, which must exceed every index in ``faces``; the sort then orders only
        the bits a key can occupy. Required to sort more than one class.

    Returns
    -------
    wp.array[wp.int32]
        ``(count,)`` halfedge indices on ``faces.device``.

    Raises
    ------
    ValueError
        If ``inclusive`` is not a whole number of halfedge rows, or several classes are to be
        sorted and ``n_vertices`` is not given.

    See Also
    --------
    [`halfedge_mates`][ordito.halfedge.halfedge_mates]
    [`sorted_face_edge_keys`][ordito.adjacency.sorted_face_edge_keys]
    """
    device = faces.device
    n = faces.size // 3 * 3
    if (n == 0 and inclusive.size > 0) or (n > 0 and inclusive.size % n != 0):
        raise ValueError(
            f"inclusive must hold a whole number of {n}-halfedge rows, got {inclusive.size}"
        )
    n_classes = inclusive.size // n if n > 0 else 0
    if key_order is not None:
        halfedges = _launch.empty(count, dtype=wp.int32, device=device)
        if count > 0:
            _launch.launch(
                kernel_halfedge.flagged_key_ordered_halfedges,
                dim=inclusive.size,
                inputs=[inclusive, key_order, wp.int32(n), halfedges],
                device=device,
            )
        return halfedges
    if n_vertices is None:
        if n_classes > 1:
            raise ValueError("n_vertices is required to sort more than one class of halfedges")
        n_vertices = INDEX_RADIX_PAIR
    keys = _launch.empty(2 * count, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * count, dtype=wp.int32, device=device)
    if count == 0:
        return order
    _launch.launch(
        kernel_halfedge.flagged_halfedge_keys,
        dim=inclusive.size,
        inputs=[faces, inclusive, wp.int32(n), wp.uint64(n_vertices), keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(
        keys,
        order,
        count=count,
        end_bit=min(64, max(1, (n_classes * n_vertices * n_vertices - 1).bit_length())),
    )
    return odt.as_dense(order[:count])


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
    if n_halfedges > 0 and n_vertices is not None and _buckets_on(faces):
        return _pair_bucketed_halfedges(faces, n_vertices, defect_counts)
    twins = _launch.empty(n_halfedges, dtype=wp.int32, device=device)
    if n_halfedges == 0:
        return twins
    # The sort path's mates, then the twin rule over them: slot 0 counts edge-non-manifold edges,
    # slot 1 edges whose two halfedges run the same way, one buffer so the two rejections cost one
    # readback between them rather than two.
    _launch.launch(
        kernel_halfedge.twins_from_mates,
        dim=n_halfedges,
        inputs=[faces, halfedge_mates(faces, n_vertices), twins, defect_counts],
        device=device,
    )
    return twins


def _pair_bucketed_halfedges(
    faces: wp.array[wp.int32], n_vertices: int, defect_counts: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Build the twin table from per-vertex edge buckets, with the sort path's rule and defect counts.

    A counting sort of the halfedges by the lower-degree endpoint of their edge (the bucketing of
    ``warp.geometry.tri_tri_adjacency``, keyed so that a hub's bucket stays small), then one thread
    per halfedge scanning its bucket. ``n_vertices`` sizes the buckets, so it must exceed every
    index in ``faces``.
    """
    device = faces.device
    n_halfedges = faces.size // 3 * 3
    degrees, cursors, buckets = _edge_buckets(faces, n_vertices)
    twins = _launch.empty(n_halfedges, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_halfedge.pair_bucketed_halfedges,
        dim=n_halfedges,
        inputs=[faces, degrees, cursors, buckets, twins, defect_counts],
        device=device,
    )
    return twins


def _edge_buckets(
    faces: wp.array[wp.int32], n_vertices: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.vec2i]]:
    """
    Counting-sort the halfedges into per-vertex buckets, each under its edge's lower-degree end.

    Returns the corner count of every vertex (which decides the owner), the bucket ends
    (``ends[v - 1]`` to ``ends[v]`` is bucket ``v``, bucket 0 starting at 0) and the buckets of
    ``(other endpoint, halfedge)``. ``n_vertices`` sizes them, so it must exceed every index.
    """
    device = faces.device
    n_halfedges = faces.size // 3 * 3
    degrees = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_scatter.count_occurrences, dim=n_halfedges, inputs=[faces, degrees], device=device
    )
    # Bucket ``v`` is counted into ``cursors[v + 1]``; the inclusive scan in place turns the counts
    # into bucket ends there, which leaves ``cursors[v]`` holding bucket ``v``'s start for the
    # scatter to advance to its end.
    cursors = _launch.zeros(n_vertices + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_halfedge.count_edge_buckets,
        dim=n_halfedges,
        inputs=[faces, degrees, cursors],
        device=device,
    )
    _launch.array_scan(cursors[1:], out_array=cursors[1:], inclusive=True)
    buckets = _launch.empty(n_halfedges, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_halfedge.scatter_edge_buckets,
        dim=n_halfedges,
        inputs=[faces, degrees, cursors, buckets],
        device=device,
    )
    return degrees, cursors, buckets


def _buckets_on(faces: wp.array[wp.int32]) -> bool:
    """Whether a pairing of ``faces`` with a known vertex count takes the edge buckets (CUDA)."""
    return wp.get_device(faces.device).is_cuda or _BUCKETED_PAIRING_ON_CPU


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
