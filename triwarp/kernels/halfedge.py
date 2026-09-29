"""
Kernels for the halfedge structure (``triwarp.halfedge``).

Every function here is index arithmetic under the ``h = 3 * f + k`` convention: halfedge ``h``
belongs to face ``h // 3``, occupies corner ``h % 3`` of it, and runs
``faces[h] -> faces[3 * f + (k + 1) % 3]``. There is no stored ``next`` or ``prev`` pointer and no
face table to consult, which is why these are ``@wp.func``s over an ``int32`` rather than lookups.

**The decomposition itself is deliberately not wrapped.** ``h // 3`` for the face appears at ~20
sites across the tree and stays spelled that way: it is shorter than a call, the convention is
stated here and restated at the sites that need it (``adjacency.py`` puts it in one clause -- *"the
face of edge ``e`` is just ``e // 3``"*), and a ``halfedge_face(h)`` helper would add a name without
adding information. What *is* worth reaching for is anything with a branch in it --
``halfedge_next`` and ``halfedge_prev`` below -- because those get re-derived by hand instead:
``kernels/repair.py`` carried ``(forward // 3) * 3 + (forward + 1) % 3`` twice, in a file that
already imported from here.
"""

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT


@wp.func
def halfedge_next(h: wp.int32) -> wp.int32:
    # Next halfedge inside the same face, under the ``h = 3 * f + k`` convention of
    # ``triwarp.halfedge``: index arithmetic, no structure to look up.
    return h - h % 3 + (h + 1) % 3


@wp.func
def halfedge_prev(h: wp.int32) -> wp.int32:
    # Halfedge ``3*f + k`` runs ``faces[3f+k] -> faces[3f+(k+1)%3]``, so the face-local cycle is
    # pure index arithmetic: no stored ``prev`` pointer.
    k = h % wp.int32(3)
    if k == wp.int32(0):
        return h + wp.int32(2)
    return h - wp.int32(1)


@wp.func
def next_boundary_halfedge(
    faces: wp.array[wp.int32], twins: wp.array[wp.int32], h: wp.int32
) -> wp.int32:
    # The boundary halfedge following ``h`` around the rim, or -1 if the fan does not close.
    #
    # Rotate around ``h``'s tip through the faces on ``h``'s own side until an outgoing halfedge
    # with no twin turns up. That sector is what makes this well defined where a per-*vertex* table
    # is not: a bowtie rim vertex -- two rim loops pinched at one point, edge-manifold and
    # consistently wound, so nothing upstream rejects it -- has two outgoing boundary halfedges,
    # and this picks the one bounding the same sector as ``h`` rather than whichever won a race.
    # The map is a bijection on boundary halfedges for the same reason, which is what lets
    # ``repair`` invert it by scatter without an atomic and ``boundary`` walk its cycles as loops.
    #
    # The bound is the face count because a fan cannot be longer than the mesh; it is never
    # approached, and it is here so a malformed twin table cannot spin forever.
    g = halfedge_next(h)
    for _ in range(faces.shape[0] // 3):
        if twins[g] < 0:
            return g
        g = halfedge_next(twins[g])
    return -1


@wp.func
def halfedge_destination(faces: wp.array[wp.int32], h: wp.int32) -> wp.int32:
    # Halfedge ``3*f + k`` ends at the next corner of its face; ``faces[h]`` is its origin.
    k = h % wp.int32(3)
    if k == wp.int32(2):
        return faces[h - wp.int32(2)]
    return faces[h + wp.int32(1)]


@wp.func
def halfedge_endpoints(faces: wp.array[wp.int32], h: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Halfedge ``h`` as the directed vertex pair ``(origin, destination)`` it runs along, which
    # puts its own face on the left because a face's corners run counter-clockwise.
    #
    # Named because ``faces[h]`` being the origin is a *convention*, and a caller that reads one
    # endpoint through ``halfedge_destination`` and the other by indexing ``faces`` directly is
    # spelling half of it out by hand.
    return faces[h], halfedge_destination(faces, h)


@wp.kernel
def halfedge_vertex_pairs(
    faces: wp.array[wp.int32],
    halfedges: wp.array[wp.int32],
    indirect: wp.bool,
    halfedge_rows: wp.array[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # Selected halfedges read as directed vertex pairs, which puts each one's own face on the left.
    # Row ``i`` is halfedge ``halfedges[halfedge_rows[i]]`` when ``indirect``; otherwise
    # ``halfedges`` is ``None`` and never read, and ``halfedge_rows`` holds the halfedge indices
    # themselves. Those are the two forms a caller holds them in -- a per-edge halfedge table
    # indexed by selected rows, or a list of halfedges -- served without materialising the gather
    # first. Row ``h`` of ``edges.faces_to_edges`` is halfedge ``h``, so this is also that
    # table's rows at ``halfedge_rows``, without building the table.
    i = wp.int32(wp.tid())
    h = halfedge_rows[i]
    if indirect:
        h = halfedges[h]
    # Bound to locals first: Warp cannot assign a multi-return straight into array elements.
    tail, tip = halfedge_endpoints(faces, h)
    out_edges[i, 0] = tail
    out_edges[i, 1] = tip


# ``sorted_halfedge_run_class`` verdicts for one sorted position of the packed halfedge keys.
HALFEDGE_RUN_NONE = wp.constant(wp.int32(0))  # not a run's first position, or a boundary edge
HALFEDGE_RUN_NON_MANIFOLD = wp.constant(wp.int32(1))  # first of a run of three or more
HALFEDGE_RUN_SAME_DIRECTION = wp.constant(wp.int32(2))  # first of a pair running the same way
HALFEDGE_RUN_TWINS = wp.constant(wp.int32(3))  # first of a pair of opposite halfedges


@wp.func
def sorted_halfedge_run_class(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    i: wp.int32,
) -> wp.int32:
    """
    Classify sorted position ``i`` of the packed halfedge keys by the halfedge-twin rule.

    Only the first position of each run of equal keys answers, so a kernel over every sorted
    position resolves each undirected edge exactly once: a run of one is a boundary edge
    (``HALFEDGE_RUN_NONE``), three or more a non-manifold edge, and a pair is a twin pair unless
    its two halfedges share an origin, i.e. traverse the edge the same way. The one spelling of the
    rule ``pair_sorted_halfedges`` applies while pairing twins and ``repair.degree3_fan_tables``
    applies to validate without building the table.
    """
    key = sorted_keys[i]
    if i > 0 and sorted_keys[i - 1] == key:
        return HALFEDGE_RUN_NONE
    n = sorted_keys.shape[0]
    if i + 1 >= n or sorted_keys[i + 1] != key:
        return HALFEDGE_RUN_NONE
    if i + 2 < n and sorted_keys[i + 2] == key:
        return HALFEDGE_RUN_NON_MANIFOLD
    if faces[order[i]] == faces[order[i + 1]]:
        return HALFEDGE_RUN_SAME_DIRECTION
    return HALFEDGE_RUN_TWINS


@wp.kernel
def pair_sorted_halfedges(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    out_twins: wp.array[wp.int32],
    out_defect_counts: wp.array[wp.int32],
) -> None:
    # One thread per position in the hash-sorted halfedge list; only the first position of each run
    # of equal keys acts, so each undirected edge is resolved exactly once. Run length 1 is a
    # boundary edge (twin stays -1), 2 an interior edge, 3+ a non-manifold edge.
    #
    # ``out_defect_counts`` carries the two rejections in one buffer so the wrapper needs one
    # readback: slot 0 counts edge-non-manifold edges, slot 1 counts edges whose two halfedges run
    # the *same* way.
    #
    # **The direction test is what makes the twin contract true rather than merely plausible.** The
    # sort key is built from the *sorted* endpoint pair, so it identifies the undirected edge and
    # says nothing about which way either halfedge crosses it -- and on a mesh that is not
    # consistently wound, the two halfedges of an edge can both run ``a -> b``. Pairing those still
    # satisfies ``twins[twins[h]] == h``, so nothing downstream notices; what breaks is the
    # *orientation* half of the contract ``halfedge_twins`` documents, and with it the CCW rotation
    # ``h -> twins[prev(h)]`` that ``write_one_rings`` walks. On a closed, edge- and
    # vertex-manifold but *non-orientable* mesh that leaves ``vertex_one_rings`` succeeding while
    # returning ring entries whose halfedge does not originate at the owning vertex, and halfedges
    # appearing in two rings at once -- so ``halfedge_tangent_angles`` sums corner angles belonging
    # to other vertices and races two threads onto one slot, or the walk closes early and the
    # wrapper reports a "pinch point" on a mesh that has none.
    #
    # Two halfedges of one undirected edge run the same way exactly when they share an origin, so
    # the test is one gather and no geometry. Two further inputs it rejects, both correctly: an
    # exactly *duplicated* face, whose three edges each carry two halfedges pointing the same way
    # (a reversed duplicate is a consistently wound degenerate surface and is still accepted); and
    # a face with a repeated vertex, whose self-edge ``a -> a`` has no opposite direction to find.
    # The wrapper's message says "the same direction", which is true of all three.
    i = wp.int32(wp.tid())
    run = sorted_halfedge_run_class(faces, sorted_keys, order, i)
    if run == HALFEDGE_RUN_NON_MANIFOLD:
        wp.atomic_add(out_defect_counts, 0, 1)
    elif run == HALFEDGE_RUN_SAME_DIRECTION:
        wp.atomic_add(out_defect_counts, 1, 1)
    elif run == HALFEDGE_RUN_TWINS:
        h0 = order[i]
        h1 = order[i + 1]
        out_twins[h0] = h1
        out_twins[h1] = h0


@wp.kernel
def count_mispaired_twins(
    faces: wp.array[wp.int32], twins: wp.array[wp.int32], out_mispaired: wp.array[wp.int32]
) -> None:
    # One thread per halfedge, counting the entries of a *caller-supplied* twin table that are not
    # the opposite halfedge. ``halfedge_twins`` cannot produce one -- ``pair_sorted_halfedges``
    # makes the property hold by construction -- so this exists for the ``twins=`` keyword, which
    # three public modules accept and none of them can trust.
    #
    # The three tests together are the whole contract ``halfedge_twins`` documents, and none of
    # them implies the others: ``twins[twins[h]] == h`` alone permits a pair that spans two
    # different edges, and the endpoint test alone permits a third halfedge on the same edge
    # claiming one of them back. A boundary entry (``-1``) is legal and skipped.
    #
    # The range test has to run *before* the gathers it guards: an out-of-range twin would
    # otherwise index the face buffer, which on the CPU device is a host-heap read rather than a
    # fault. (Kernel-scope ``or`` does short-circuit on Warp 1.17 -- ``emit_BoolOp`` guards each
    # operand -- but the separate ``if``s keep the order explicit.)
    #
    # That range test is against ``twins.shape[0]`` and **not** ``faces.shape[0]``, which are not
    # the same number: the wrapper defines the halfedge count as ``faces.shape[0] // 3 * 3``, so a
    # ragged face buffer leaves up to two trailing entries that are not halfedges. A twin of
    # ``faces.shape[0] - 1`` there would pass a check against the face length and then send
    # ``halfedge_destination`` one past the end -- the very read this guard exists to stop.
    #
    # It is cheap, which is what the wrapper claims: a small fraction of the ``halfedge_twins``
    # rebuild it lets the caller skip, flat in the halfedge count because both are launch-bound.
    h = wp.int32(wp.tid())
    twin = twins[h]
    if twin < wp.int32(0):
        return
    if twin >= twins.shape[0]:
        wp.atomic_add(out_mispaired, 0, 1)
        return
    if twins[twin] != h:
        wp.atomic_add(out_mispaired, 0, 1)
        return
    if faces[twin] != halfedge_destination(faces, h):
        wp.atomic_add(out_mispaired, 0, 1)
        return
    if faces[h] != halfedge_destination(faces, twin):
        wp.atomic_add(out_mispaired, 0, 1)


@wp.kernel
def ring_degrees_and_starts(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    out_degrees: wp.array[wp.int32],
    out_starts: wp.array2d[wp.int32],
) -> None:
    # One thread per halfedge, answering both per-vertex questions the ring walk needs before it
    # can start: the ring size (every face contributes exactly one outgoing halfedge per corner, so
    # it is how often the vertex appears in the flat face buffer) and the lowest-indexed outgoing
    # halfedge, tracked twice -- row 0 over all of them, row 1 over the twin-less (boundary) ones
    # alone. A boundary vertex must start its CCW walk at its boundary halfedge -- the
    # clockwise-most edge of its fan -- or the walk covers only part of the fan. ``atomic_min``
    # makes both picks deterministic regardless of thread order.
    #
    # Both halves are a scatter over the same ``faces[h]`` index, so they share one pass rather
    # than a ``scatter.count_occurrences`` launch of their own over the same buffer.
    # ``out_degrees`` arrives zeroed and ``out_starts`` filled with ``INT32_MAX``.
    h = wp.int32(wp.tid())
    origin = faces[h]
    wp.atomic_add(out_degrees, origin, 1)
    wp.atomic_min(out_starts, 0, origin, h)
    if twins[h] == wp.int32(-1):
        wp.atomic_min(out_starts, 1, origin, h)


@wp.kernel
def write_one_rings(
    starts: wp.array2d[wp.int32],
    twins: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_halfedges: wp.array[wp.int32],
    out_is_boundary: wp.array[wp.bool],
    out_incomplete: wp.array[wp.int32],
) -> None:
    # Rotate counter-clockwise through the outgoing halfedges at one vertex: ``h -> twin(prev(h))``
    # steps from edge ``v->a`` to edge ``v->b`` within the CCW-oriented face ``(v, a, b)``. The walk
    # is bounded by the vertex's known ring size, so a pinched (vertex-non-manifold) vertex — where
    # the rotation closes early on one of its fans — is reported instead of silently truncated.
    #
    # ``starts`` is ``ring_degrees_and_starts``' two candidate rows, resolved at this thread's own
    # vertex. Start: prefer the boundary halfedge; ``INT32_MAX`` means "no candidate of this kind".
    # Boundary: a vertex lies on one exactly when an outgoing halfedge has no twin, which is the
    # same comparison -- so ``out_is_boundary`` is written here, for every vertex, isolated ones
    # included, rather than by a pass of its own over the same row.
    v = wp.int32(wp.tid())
    boundary_start = starts[1, v]
    interior_start = starts[0, v]
    on_boundary = boundary_start != INT32_MAX_CONSTANT
    out_is_boundary[v] = on_boundary
    start = wp.int32(-1)
    if on_boundary:
        start = boundary_start
    elif interior_start != INT32_MAX_CONSTANT:
        start = interior_start
    if start == wp.int32(-1):
        return
    begin = offsets[v]
    degree = offsets[v + 1] - begin
    count = wp.int32(0)
    h = start
    while h != wp.int32(-1) and count < degree:
        out_halfedges[begin + count] = h
        count += wp.int32(1)
        nxt = twins[halfedge_prev(h)]
        if nxt == wp.int32(-1) or nxt == start:
            break
        h = nxt
    if count != degree:
        wp.atomic_add(out_incomplete, 0, 1)
