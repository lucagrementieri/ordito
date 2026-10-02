import warp as wp

from triwarp.constants import INDEX_RADIX_PAIR
from triwarp.kernels.array import (
    binary_search_sorted_contains,
    declare_map_signatures,
    map_probe,
    map_probe_single,
    pack_edge_key,
    pack_ranked_key,
    scanned_count,
    update_argmin_pair,
)
from triwarp.kernels.bounds import packed_box_diagonal
from triwarp.kernels.halfedge import (
    HALFEDGE_RUN_NON_MANIFOLD,
    HALFEDGE_RUN_SAME_DIRECTION,
    halfedge_destination,
    halfedge_next,
    next_boundary_halfedge,
    sorted_halfedge_run_class,
)
from triwarp.kernels.predicates import triangle_aspect_ratio, triangle_normal
from triwarp.kernels.triangles import (
    QUALITY_AREA,
    corner_triple,
    face_signed_volume,
    face_vertices,
    triangle_cross,
    triangle_quality,
    write_corner_triple,
)


@wp.kernel
def mark_referenced(faces: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    # A ``1`` flag at every vertex a face names, in the ``int32`` the scan below reads, so no mask
    # has to be converted first. Out-of-range entries -- the ``-1`` sentinels ``repair`` preserves
    # through a remap -- are dropped rather than written off the end (CLAUDE.md section 12.1).
    # Every writer stores ``1``, so racing threads agree.
    index = faces[wp.int32(wp.tid())]
    if index >= 0 and index < out_flags.shape[0]:
        out_flags[index] = 1


@wp.kernel
def compact_referenced(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inclusive: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_inverse: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The old-to-new map, the compacted positions, the new-to-old map and the remapped faces from
    # the in-place scan of the referenced flags, in one pass: ``flatnonzero``, the ``-1`` fill, the
    # index scatter, the position gather and ``array.remap_indices`` would be one launch each.
    # Launched over ``max(n_vertices, len(faces))``. A face entry reads the scan rather than
    # ``out_remap``, which another thread of this launch writes: every vertex a face names was
    # marked, so its new index is its exclusive scan value, which is what ``out_remap`` holds. A
    # negative sentinel passes through unchanged (``array.gather_1d_skip_negative``).
    t = wp.int32(wp.tid())
    if t < faces.shape[0]:
        index = faces[t]
        if index >= 0:
            index = inclusive[index] - 1
        out_faces[t] = index
    if t >= vertices.shape[0]:
        return
    slot, referenced = scanned_count(inclusive, t)
    if referenced == 0:
        out_remap[t] = -1
        return
    out_remap[t] = slot
    out_vertices[slot] = vertices[t]
    out_inverse[slot] = t


@wp.func
def cyclic_match(
    a0: wp.int32, a1: wp.int32, a2: wp.int32, b0: wp.int32, b1: wp.int32, b2: wp.int32
) -> wp.bool:
    """Whether ``(a0, a1, a2)`` is a cyclic rotation of ``(b0, b1, b2)`` (same orientation)."""
    return (
        (a0 == b0 and a1 == b1 and a2 == b2)
        or (a0 == b1 and a1 == b2 and a2 == b0)
        or (a0 == b2 and a1 == b0 and a2 == b1)
    )


@wp.kernel
def scatter_duplicate_face_stats(
    faces: wp.array[wp.int32],
    unique_faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_member_count: wp.array[wp.int32],
    out_signed_count: wp.array[wp.int32],
    out_first_member: wp.array[wp.int32],
    out_first_positive: wp.array[wp.int32],
    out_first_negative: wp.array[wp.int32],
) -> None:
    # Per input face: orientation sign against its group representative (the first-occurring
    # face, gathered via ``inverse``) scattered into per-group stats. ``first_*`` slots hold the
    # smallest face index of each sign class (seeded with the n_faces sentinel).
    f = wp.int32(wp.tid())
    ui = inverse[f]
    consistent = cyclic_match(
        faces[f * 3],
        faces[f * 3 + 1],
        faces[f * 3 + 2],
        unique_faces[ui * 3],
        unique_faces[ui * 3 + 1],
        unique_faces[ui * 3 + 2],
    )
    wp.atomic_add(out_member_count, ui, wp.int32(1))
    wp.atomic_min(out_first_member, ui, f)
    if consistent:
        wp.atomic_add(out_signed_count, ui, wp.int32(1))
        wp.atomic_min(out_first_positive, ui, f)
    else:
        wp.atomic_add(out_signed_count, ui, wp.int32(-1))
        wp.atomic_min(out_first_negative, ui, f)


@wp.kernel
def resolve_duplicate_groups(
    member_count: wp.array[wp.int32],
    signed_count: wp.array[wp.int32],
    first_member: wp.array[wp.int32],
    first_positive: wp.array[wp.int32],
    first_negative: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
    out_keep_mask: wp.array[wp.bool],
    out_error_group: wp.array[wp.int32],
) -> None:
    # Keep-decision per duplicate group (igl::resolve_duplicated_faces): singletons stay; a net
    # +1/-1 orientation keeps the first member of the majority sign; a cancelling group drops;
    # anything else is non-orientable (the smallest offending group index is reported).
    #
    # ``out_keep_mask`` is ``out_keep >= 0``, written by the thread that decided it so the caller
    # can compact the kept groups without a second pass re-reading ``out_keep`` for the sign.
    ui = wp.int32(wp.tid())
    count = signed_count[ui]
    keep = wp.int32(-1)
    if member_count[ui] == 1:
        keep = first_member[ui]
    elif count == 1:
        keep = first_positive[ui]
    elif count == -1:
        keep = first_negative[ui]
    elif count != 0:
        wp.atomic_min(out_error_group, 0, ui)
    out_keep[ui] = keep
    out_keep_mask[ui] = keep >= 0


@wp.kernel
def reduce_largest_group(counts: wp.array[wp.int32], out_best: wp.array[wp.int64]) -> None:
    # The group with the most members, ties going to the lowest group index, reduced into one
    # ``int64`` by ``pack_ranked_key`` so the whole answer is a single ``wp.atomic_max``. Launch
    # over the group domain; ``out_best`` is one element seeded to ``-1``, which every real key
    # exceeds.
    #
    # An empty group contributes nothing, which matters because the group domain here is the *face*
    # domain -- a connected-component label is a representative face index, so most slots are empty.
    group = wp.int32(wp.tid())
    if counts[group] > 0:
        wp.atomic_max(out_best, 0, pack_ranked_key(counts[group], group))


@wp.kernel
def mark_largest_group_mask(
    groups: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    best: wp.array[wp.int64],
    out_mask: wp.array[wp.bool],
) -> None:
    # Flag every member of the group ``reduce_largest_group`` chose. Recomputing the key and testing
    # it against the reduced maximum is what keeps this readback-free: the winning group index is
    # never brought to the host, and the ``-1`` seed marks nothing when there are no groups at all.
    #
    # Not convertible to a ``wp.map`` over a gather, unlike the *threshold* criteria beside it in
    # ``repair.remove_small_components``: those are one table lookup compared against a scalar, so
    # ``wp.map(greater_equal, statistic[labels], threshold)`` is the whole kernel, whereas this
    # needs two gathers and a call to rebuild the key before it can compare.
    f = wp.int32(wp.tid())
    group = groups[f]
    out_mask[f] = pack_ranked_key(counts[group], group) == best[0]


@wp.kernel(enable_backward=False)
def small_triangle_collapse_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    min_dbl_area: wp.float32,
    out_pairs: wp.array2d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    """Emit the shortest edge of each face with double-area below ``min_dbl_area`` (libigl)."""
    # ``out_count`` is how many faces were small, which is all the caller's loop reads: one
    # *conditional* atomic per small face -- contention scales with the (rare) hits, not with the
    # launch -- in place of a per-face flag buffer the caller would reduce and read back.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    dbl_area = wp.length(triangle_cross(vertices, faces, f))
    if dbl_area < min_dbl_area:
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        # Shortest of the three edges (0,1), (1,2), (2,0); collapse its endpoints together.
        best = wp.length_sq(v1 - v0)
        a = i0
        b = i1
        update_argmin_pair(best, a, b, wp.length_sq(v2 - v1), i1, i2)
        update_argmin_pair(best, a, b, wp.length_sq(v0 - v2), i2, i0)
        out_pairs[f, 0] = a
        out_pairs[f, 1] = b
        wp.atomic_add(out_count, 0, 1)
    else:
        out_pairs[f, 0] = i0
        out_pairs[f, 1] = i0


@wp.kernel
def reverse_face_winding(faces: wp.array[wp.int32], out_faces: wp.array[wp.int32]) -> None:
    # np.fliplr on an (n, 3) face block. All three indices are read before any is written, so
    # this is safe to run in place (out_faces is faces).
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    write_corner_triple(out_faces, f, c, b, a)


@wp.func
def write_face_winding(
    faces: wp.array[wp.int32], f: wp.int32, reversed_winding: wp.bool, out_faces: wp.array[wp.int32]
) -> None:
    """Copy face ``f``'s corners, swapping corners 1 and 2 when ``reversed_winding``."""
    # Corner 0 is kept and 1 and 2 swap, which is a *different* reversal from
    # ``triangles.write_corner_triple_reversible``'s ``np.fliplr`` -- see that function's docstring
    # for why the two conventions are deliberately not merged behind one flag. The read and the
    # write still go through the shared triple helpers rather than three hand-written offsets each.
    i0, i1, i2 = corner_triple(faces, f)
    if reversed_winding:
        i1, i2 = i2, i1
    write_corner_triple(out_faces, f, i0, i1, i2)


@wp.kernel
def flip_faces_masked(
    faces: wp.array[wp.int32], flip: wp.array[wp.bool], out_faces: wp.array[wp.int32]
) -> None:
    """Copy ``faces`` to ``out_faces``, reversing winding (swap corners 1,2) where ``flip``."""
    f = wp.int32(wp.tid())
    write_face_winding(faces, f, flip[f], out_faces)


@wp.kernel
def flip_all_faces(faces: wp.array[wp.int32], out_faces: wp.array[wp.int32]) -> None:
    """Copy ``faces`` to ``out_faces`` with every winding reversed (corners 1 and 2 swapped)."""
    # ``flip_faces_masked``' convention with the mask constant, so the whole-mesh flip needs no
    # all-ones flag buffer allocated only to be read back as ``True`` at every face.
    f = wp.int32(wp.tid())
    write_face_winding(faces, f, True, out_faces)


@wp.kernel
def scatter_face_area_by_group(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    groups: wp.array[wp.int32],
    out_group_area: wp.array[wp.float32],
) -> None:
    # Each face's area added into its group's slot. The area is ``triangle_quality``'s, so it is the
    # value ``triangles.face_quality(metric="area")`` would have stored for this face -- computed
    # where it is summed rather than written to a per-face buffer for this pass to read back.
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    wp.atomic_add(out_group_area, groups[f], triangle_quality(v0, v1, v2, QUALITY_AREA))


@wp.kernel
def scatter_face_volume_by_group(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    groups: wp.array[wp.int32],
    out_group_volume: wp.array[wp.float32],
) -> None:
    # Each face's signed tetrahedron volume from the origin added into its group's slot:
    # ``triangles.face_signed_volume``, the value ``triangles.face_signed_volumes`` would have
    # stored for this face, computed where it is summed. ``scatter_face_area_by_group``'s shape.
    f = wp.int32(wp.tid())
    wp.atomic_add(out_group_volume, groups[f], face_signed_volume(vertices, faces, f, wp.vec3()))


@wp.kernel
def group_diameter_flags(
    corners: wp.array[wp.float32],
    groups: wp.array[wp.int32],
    min_diameter: wp.float32,
    out_keep: wp.array[wp.bool],
) -> None:
    # A face is kept when its group's bounding-box diagonal reaches ``min_diameter`` (inclusive):
    # the group's packed box, accumulated by ``scatter.scatter_group_bounds``, decoded by
    # ``bounds.packed_box_diagonal`` as the face reads it, so no per-group diagonal table is written
    # and gathered back.
    f = wp.int32(wp.tid())
    out_keep[f] = packed_box_diagonal(corners, groups[f]) >= min_diameter


@wp.kernel
def flip_faces_by_component_volume(
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    component_volume: wp.array[wp.float32],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``make_volume(multibody=True)``'s orientation pass: a face is reversed when the component it
    # belongs to encloses a negative signed volume, i.e. is wound inward.
    #
    # The component's volume is read through this face's own label rather than gathered into a
    # per-face buffer first: the Python-scope form was an ``indexedarray`` view, a ``wp.map`` to
    # materialise a face-sized flag array, and then this pass to consume it -- two launches and a
    # buffer for a value one indirection away. Measured 1.15x on ``make_volume(multibody=True)``.
    f = wp.int32(wp.tid())
    write_face_winding(faces, f, component_volume[labels[f]] < wp.float32(0.0), out_faces)


@wp.kernel
def halfedge_orientation_slots(
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    out_forward_count: wp.array[wp.int32],
    out_backward_count: wp.array[wp.int32],
    out_forward_corner: wp.array[wp.int32],
    out_backward_corner: wp.array[wp.int32],
) -> None:
    # Per unique edge: how many half-edges traverse it each way, and which corner they belong to.
    #
    # Corner ``c = 3f + j`` is the half-edge ``(faces[c], halfedge_destination(faces, c))`` -- the
    # shared face-local cycle rather than a second spelling of it here. ``unique_edges`` rows are
    # sorted min-first, so a half-edge is "forward" when it runs low index to high. A
    # *consistently oriented manifold* edge has exactly one of each; anything else -- two forward
    # (a flipped neighbour), three or more of either (a non-manifold edge), or one alone (a
    # boundary) -- is not mergeable, so its corner slot is never read and the ``atomic_max`` only
    # keeps the write deterministic.
    c = wp.int32(wp.tid())
    e = edge_of_corner[c]
    if faces[c] < halfedge_destination(faces, c):
        wp.atomic_add(out_forward_count, e, 1)
        wp.atomic_max(out_forward_corner, e, c)
    else:
        wp.atomic_add(out_backward_count, e, 1)
        wp.atomic_max(out_backward_corner, e, c)


@wp.kernel
def corner_merge_links(
    forward_count: wp.array[wp.int32],
    backward_count: wp.array[wp.int32],
    forward_corner: wp.array[wp.int32],
    backward_corner: wp.array[wp.int32],
    out_links: wp.array2d[wp.int32],
) -> None:
    # Two corner-graph links per mergeable edge: one joining the corners at each endpoint.
    #
    # Nodes are ``(face, vertex slot)`` pairs under the same ``3f + k`` numbering, so node
    # ``3f + k`` *is* the corner whose vertex is ``faces[3f + k]``, and a link only ever joins two
    # copies of one original vertex. Connected components of this graph are the vertex copies.
    #
    # A non-mergeable edge emits two self-loops on node 0 rather than nothing, which keeps the
    # output a fixed ``(2 * n_edges, 2)`` buffer with no compaction pass; a self-loop merges
    # nothing.
    e = wp.int32(wp.tid())
    out_links[e * 2 + 0, 0] = 0
    out_links[e * 2 + 0, 1] = 0
    out_links[e * 2 + 1, 0] = 0
    out_links[e * 2 + 1, 1] = 0
    if forward_count[e] != 1 or backward_count[e] != 1:
        return

    forward = forward_corner[e]
    backward = backward_corner[e]
    # The forward half-edge runs (low, high) from its own corner; the backward one runs (high, low),
    # so its *next* slot holds the low endpoint.
    forward_next = halfedge_next(forward)
    backward_next = halfedge_next(backward)
    out_links[e * 2 + 0, 0] = forward  # low endpoint, forward face
    out_links[e * 2 + 0, 1] = backward_next  # low endpoint, backward face
    out_links[e * 2 + 1, 0] = forward_next  # high endpoint, forward face
    out_links[e * 2 + 1, 1] = backward  # high endpoint, backward face


@wp.func
def is_interior_degree3(ring_start: wp.int32, ring_end: wp.int32, on_boundary: wp.bool) -> wp.bool:
    # An interior vertex with exactly three incident faces, read off a one-ring CSR. Its ring size
    # *is* the face count, so a boundary vertex with three faces has four neighbours and is excluded
    # by the flag rather than by the count. ``interior_degree3_rim`` is the same test from the three
    # faces alone.
    return not on_boundary and ring_end - ring_start == 3


@wp.func
def is_interior_degree3_in_region(
    ring_start: wp.int32, ring_end: wp.int32, on_boundary: wp.bool, in_region: wp.bool
) -> wp.bool:
    # ``is_interior_degree3`` restricted to a caller's vertex region, so the restriction rides the
    # same map rather than a second pass over the mask.
    return in_region and is_interior_degree3(ring_start, ring_end, on_boundary)


@wp.func
def wins_degree3_conflict(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    v: wp.int32,
) -> wp.bool:
    # Two adjacent candidates share faces, so only one of them can be removed in a pass. The lowest
    # index wins, which makes the choice deterministic and independent of launch order -- the same
    # rule the Delaunay flip pass uses to resolve competing edges.
    #
    # The ring form of the rule ``emit_degree3_replacement`` decides from its three-face tables,
    # for ``flatten_degree3_vertices``, whose ``rings=`` argument hands it a one-ring CSR. The two
    # are one rule over two representations of the same fan: a candidate's neighbours are its ring
    # destinations here and its rim there, the same three vertices.
    if not candidate[v]:
        return False
    for slot in range(ring_offsets[v], ring_offsets[v + 1]):
        neighbour = halfedge_destination(faces, ring_halfedges[slot])
        if candidate[neighbour] and neighbour < v:
            return False
    return True


@wp.func
def record_degree3_halfedge(
    faces: wp.array[wp.int32],
    h: wp.int32,
    counts: wp.array[wp.int32],
    link_sums: wp.array[wp.int32],
    fans: wp.array2d[wp.int32],
) -> None:
    # One halfedge's share of ``remove_degree3_vertices``' per-vertex tables, in place of a whole
    # one-ring CSR: its origin's corner count (its face count), the origin's first three outgoing
    # halfedges in arrival order, and a telescoping sum over its link.
    #
    # The link sum adds ``x - y`` for the edge ``x -> y`` opposite each outgoing halfedge. Around an
    # interior vertex those edges form a closed cycle, so every link vertex is added once and
    # subtracted once and the sum is exactly zero; around a boundary fan they form one chain from
    # ``x_first`` to ``y_last``, two distinct vertices, so it is not -- modulo 2^32 too, because the
    # telescoped value lies strictly inside ``(-2^31, 2^31)``. Zero is therefore a sound "may be
    # interior" flag, which is all ``emit_degree3_replacement``'s next-pass signal needs from it.
    #
    # The table's row is arrival-ordered, so only its *contents* are meaningful;
    # ``interior_degree3_rim`` canonicalizes it.
    v = faces[h]
    x = halfedge_destination(faces, h)
    y = halfedge_destination(faces, halfedge_next(h))
    slot = wp.atomic_add(counts, v, 1)
    if slot < 3:
        fans[v, slot] = h
    wp.atomic_add(link_sums, v, x - y)


@wp.kernel
def degree3_fan_tables_input(
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_link_sums: wp.array[wp.int32],
    out_fans: wp.array2d[wp.int32],
    out_faces: wp.array[wp.int32],
    out_kept: wp.array[wp.int32],
    out_defects: wp.array[wp.int32],
) -> None:
    # Pass 0 of ``remove_degree3_vertices``, over the input's halfedges: the fan tables, plus the
    # three things only the first pass does. It copies the input into the front of the pass loop's
    # fixed-capacity face buffer (``out_faces``) and sets its kept flags, one store per face from
    # its first halfedge, in place of a copy and a fill; and it counts its position's twin defect
    # on the input's sorted halfedge keys into ``out_defects`` -- the input validation, riding on
    # this launch and on the pass's one readback. ``degree3_fan_tables`` is every later pass.
    h = wp.int32(wp.tid())
    run = sorted_halfedge_run_class(faces, sorted_keys, order, h)
    if run == HALFEDGE_RUN_NON_MANIFOLD:
        wp.atomic_add(out_defects, 0, 1)
    elif run == HALFEDGE_RUN_SAME_DIRECTION:
        wp.atomic_add(out_defects, 1, 1)
    out_faces[h] = faces[h]
    if h % 3 == 0:
        out_kept[h // 3] = 1
    record_degree3_halfedge(faces, h, out_counts, out_link_sums, out_fans)


@wp.kernel
def degree3_fan_tables(
    faces: wp.array[wp.int32],
    kept: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_link_sums: wp.array[wp.int32],
    out_fans: wp.array2d[wp.int32],
) -> None:
    # Every pass of ``remove_degree3_vertices`` after the first, over the fixed-capacity face
    # buffer's used rows: the fan tables of the faces still kept. A replaced fan's faces stay in the
    # buffer with their flag cleared and the replacements are appended after the last used row, so
    # no pass compacts; a cleared row exits here.
    #
    # Its fixed buffer makes every pass recordable, and recording passes 1 onward under
    # ``_device.record_device_loop`` (``array.loop_advance`` closing each round, so no per-pass
    # read) was built and declined: interleaved against the host loop of one small read per pass,
    # it was 0.86x at ``dragon`` (two passes) and 0.87-0.96x at ``happy_buddha`` (three). The
    # recording costs more than the reads it removes at one or two replayed passes, and no
    # benchmark mesh takes more.
    h = wp.int32(wp.tid())
    if kept[h // 3] == 0:
        return
    record_degree3_halfedge(faces, h, out_counts, out_link_sums, out_fans)


@wp.func
def interior_degree3_rim(
    faces: wp.array[wp.int32], counts: wp.array[wp.int32], fans: wp.array2d[wp.int32], v: wp.int32
) -> wp.vec3i:
    # The replacement triangle of ``v`` when it is an interior vertex with exactly three faces, and
    # ``(-1, -1, -1)`` otherwise. Everything is local to the three faces: an interior degree-3
    # vertex's three opposite edges form one directed 3-cycle ``a -> b -> c -> a``, and that cycle
    # in order *is* the triangle over its rim, wound the way the fan was. A boundary fan's opposite
    # edges form an open chain and fail the test, so no boundary flag is needed.
    #
    # The cycle starts at the lowest-indexed of the three halfedges, which is where
    # ``vertex_one_rings`` starts an interior ring -- so ``(a, b, c)`` is the ring's destinations
    # in ring order, and the replacement is the one the ring walk wrote.
    none = wp.vec3i(-1, -1, -1)
    if counts[v] != 3:
        return none
    h0 = wp.min(wp.min(fans[v, 0], fans[v, 1]), fans[v, 2])
    a = halfedge_destination(faces, h0)
    b = halfedge_destination(faces, halfedge_next(h0))
    c = wp.int32(-1)
    for k in range(3):
        h = fans[v, k]
        if halfedge_destination(faces, h) == b:
            c = halfedge_destination(faces, halfedge_next(h))
    if a == v or b == v or c == v or c == a or c == b or a == b:
        return none
    # Each of the three cycle edges must be matched by exactly one halfedge's opposite edge.
    matched = wp.int32(0)
    for k in range(3):
        h = fans[v, k]
        x = halfedge_destination(faces, h)
        y = halfedge_destination(faces, halfedge_next(h))
        if x == a and y == b:
            matched |= 1
        elif x == b and y == c:
            matched |= 2
        elif x == c and y == a:
            matched |= 4
    if matched != 7:
        return none
    return wp.vec3i(a, b, c)


@wp.kernel
def emit_degree3_replacement(
    faces: wp.array[wp.int32],
    fans: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],
    link_sums: wp.array[wp.int32],
    n_input_faces: wp.int32,
    cursor: wp.array[wp.int32],
    out_lost: wp.array[wp.int32],
    out_kept: wp.array[wp.int32],
    out_new_faces: wp.array2d[wp.int32],
    out_next_candidates: wp.array[wp.int32],
) -> None:
    # Selection and replacement in one pass. Two adjacent candidates share faces, so only one of
    # them can be removed in a pass: the lowest index wins, which makes the choice deterministic and
    # independent of launch order -- the same rule the Delaunay flip pass uses to resolve competing
    # edges. Nothing here needs another thread's *selection*, only its candidacy, which is
    # recomputed from the read-only tables, so no selection mask is written or read back.
    #
    # ``cursor`` is the count of replacements appended by every pass so far, so it hands out each
    # replacement's row in ``out_new_faces`` -- the face buffer past the input's ``n_input_faces``
    # rows, which ``faces`` also views -- and read at the end is the number of vertices removed.
    # Rows are appended pass after pass, so the kept rows in buffer order are the order a
    # compaction after every pass would have produced.
    #
    # Each fan face's kept flag is cleared here and each replacement's set, so ``out_kept`` is the
    # kept-face flag the caller scans in place into its ranks once the loop ends.
    #
    # ``out_next_candidates`` counts the rim vertices this pass may turn into candidates. Each
    # selected fan takes two of a rim vertex's faces and gives back one, so a rim vertex's count
    # falls by one per selected fan; a candidate needs it to land on exactly 3 with a closed link.
    # ``out_lost`` tallies the decrements, and the decrement that lands on 3 counts it -- a later
    # one taking it lower is not seen, so the signal is conservative (at worst one extra detection
    # pass that finds nothing) but never misses a candidate. Only zero versus non-zero is read, and
    # non-zero implies this pass selected something, which is why it alone continues the loop.
    v = wp.int32(wp.tid())
    rim = interior_degree3_rim(faces, counts, fans, v)
    if rim[0] < 0:
        return
    for k in range(3):
        neighbour = rim[k]
        if neighbour < v and interior_degree3_rim(faces, counts, fans, neighbour)[0] >= 0:
            return
    slot = wp.atomic_add(cursor, 0, 1)
    out_kept[n_input_faces + slot] = 1
    for k in range(3):
        r = rim[k]
        out_kept[fans[v, k] // 3] = 0
        out_new_faces[slot, k] = r
        lost = wp.atomic_add(out_lost, r, 1)
        if link_sums[r] == 0 and counts[r] - lost == 4:
            wp.atomic_add(out_next_candidates, 0, 1)


@wp.kernel
def mark_kept_face_vertices(
    faces: wp.array[wp.int32], kept: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # ``mark_referenced`` over the kept faces of ``remove_degree3_vertices``' face buffer only.
    h = wp.int32(wp.tid())
    if kept[h // 3] != 0:
        out_flags[faces[h]] = 1


@wp.kernel
def compact_kept_faces_and_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    kept_inclusive: wp.array[wp.int32],
    referenced_inclusive: wp.array[wp.int32],
    n_kept: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``remove_degree3_vertices``' output in one pass: the kept faces in buffer order with their
    # corners renumbered, and the referenced positions compacted -- a face compaction followed by
    # ``remove_unreferenced_vertices``' mark, scan and ``compact_referenced`` would be two passes
    # and a second face buffer. ``kept_inclusive`` and ``referenced_inclusive`` are two views of
    # *one* in-place scan over the kept flags followed by the referenced flags, so the referenced
    # ranks carry the ``n_kept`` total in front of them. Launched over ``max(n_vertices, n_used)``
    # for the ``n_used`` face rows ever written.
    t = wp.int32(wp.tid())
    if t < kept_inclusive.shape[0]:
        row, kept = scanned_count(kept_inclusive, t)
        if kept != 0:
            for k in range(3):
                out_faces[3 * row + k] = referenced_inclusive[faces[3 * t + k]] - n_kept - 1
    if t < vertices.shape[0]:
        # The first referenced entry's predecessor in the one scan is the last kept flag's, whose
        # value is the kept total.
        start = n_kept
        if t > 0:
            start = referenced_inclusive[t - 1]
        if referenced_inclusive[t] != start:
            out_vertices[start - n_kept] = vertices[t]


@wp.kernel
def collect_rim_links_and_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    min_normal_dot: wp.float32,
    max_aspect_ratio: wp.float32,
    out_rim_next: wp.array[wp.int32],
    out_rim_prev: wp.array[wp.int32],
    out_candidate: wp.array[wp.bool],
) -> None:
    # The rim as a linked list over *halfedges*, from those that have no twin.
    #
    # Keyed by halfedge rather than by vertex deliberately. Edge-manifoldness -- the caller's only
    # validation -- does not make the rim a set of simple loops through the vertices: at a bowtie
    # vertex two rim loops pass through one point, so a per-vertex ``next`` / ``prev`` / ``face``
    # triple has two writers and no answer, and the three slots could be won by different halfedges
    # so that the fold-over normal gate tested a face not bordering the candidate triangle at all
    # (measured: the two devices disagreed on one five-vertex bowtie). Per halfedge each slot has
    # exactly one writer, both rim loops keep their own links, and the bordering faces are read off
    # the halfedges themselves rather than from a third table.
    #
    # The notch test below rides in this same pass. It reads ``rim_next`` only at its own
    # halfedge, which this thread has just computed and still holds in ``following`` -- so run
    # apart it cost a launch and a full round trip of the link table through global memory to
    # recover a successor that never had to leave a register. ``out_rim_prev`` is still scattered
    # for the emit pass that follows, which genuinely does need every link written.
    #
    # Output byte-identical, and the wall clock is **flat** within run-to-run noise: a pass is a
    # halfedge-twin build plus a handful of launches around one readback, so the host was already
    # waiting on the device. What this buys is a launch and a round trip of the link table, not a
    # measurable speed-up -- worth having, not worth quoting.
    h = wp.int32(wp.tid())
    if twins[h] >= 0:
        return
    following = next_boundary_halfedge(faces, twins, h)
    out_rim_next[h] = following
    if following >= 0:
        out_rim_prev[following] = h
    # A notch that can be closed by one triangle, indexed by the boundary halfedge *entering* it.
    # That halfedge runs ``previous -> v`` and its successor runs ``v -> following``, with the
    # surface on their left, so a face attached outside the rim must carry the halfedges
    # ``v -> previous`` and ``following -> v`` -- which is the triangle ``(following, v,
    # previous)``, and that winding is what makes the result consistently oriented rather than
    # merely watertight.
    #
    # Two gates, both from the caller: the new triangle's normal must agree with the two rim faces
    # it will border -- which are the two halfedges' own faces, so no lookup can pair it with the
    # wrong one -- and its aspect ratio must be finite enough to be worth adding.
    outgoing = following
    if outgoing < 0:
        return
    previous = faces[h]
    v = halfedge_destination(faces, h)
    following = halfedge_destination(faces, outgoing)
    if previous == following:
        return  # a two-edge rim loop spans no notch
    normal = triangle_normal(vertices[following], vertices[v], vertices[previous])
    if wp.length(normal) == 0.0:
        return
    if wp.dot(normal, face_normals[h // 3]) < min_normal_dot:
        return
    if wp.dot(normal, face_normals[outgoing // 3]) < min_normal_dot:
        return
    ratio = triangle_aspect_ratio(vertices[following], vertices[v], vertices[previous])
    if ratio > max_aspect_ratio:
        return
    out_candidate[h] = True


@wp.kernel
def emit_straighten_faces(
    faces: wp.array[wp.int32],
    rim_next: wp.array[wp.int32],
    rim_prev: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    cursor: wp.array[wp.int32],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    # One new face per accepted notch, and only where neither rim neighbour with a lower index also
    # qualifies -- two adjacent notches share a rim edge, so filling both in one pass would attach
    # two faces to it. Lowest halfedge index wins, which makes the choice deterministic.
    #
    # ``rim_next[h]`` is read unguarded because ``candidate[h]`` is set only where it is a real
    # halfedge; ``rim_prev[h]`` is guarded because a fan that failed to close leaves it unwritten.
    h = wp.int32(wp.tid())
    if not candidate[h]:
        return
    outgoing = rim_next[h]
    incoming = rim_prev[h]
    if candidate[outgoing] and outgoing < h:
        return
    if incoming >= 0 and candidate[incoming] and incoming < h:
        return
    slot = wp.atomic_add(cursor, 0, 1)
    out_new_faces[slot, 0] = halfedge_destination(faces, outgoing)
    out_new_faces[slot, 1] = halfedge_destination(faces, h)
    out_new_faces[slot, 2] = faces[h]


@wp.kernel
def select_and_flatten_degree3(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    out_candidate: wp.array[wp.bool],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Move each selected vertex to the centroid of its three neighbours -- which lies in their
    # plane, so the little tetrahedral bump flattens into it.
    #
    # ``selected`` is an *independent* set of interior valence-3 vertices, and the independence is
    # what makes the centroid the answer rather than an approximation of it: two such vertices can
    # be neighbours (a tetrahedron is four of them, each adjacent to the other three), and moving
    # both at once puts neither in the other's *new* plane. Doing it anyway maps the regular
    # tetrahedron to a mirrored fraction of itself -- every normal flipped and the signed volume
    # negated -- which is why this selects an independent set first rather than moving every
    # candidate.
    #
    # The divisor is a literal 3 because ``selected`` implies interior valence 3; the ring walk is
    # over ``ring_offsets`` all the same, so a stale mask cannot make it read past the ring.
    # The selection, the move it implies and the next pass's candidates, in one pass. They are the
    # same thread's decision about the same vertex, so running them apart cost a launch and a full
    # round trip of a selection mask through global memory to tell the next kernel what this one
    # had just decided. The remaining candidates go to a second buffer (``out_candidate``, the
    # caller ping-pongs the two) because ``wins_degree3_conflict`` reads ``candidate`` at the
    # neighbours. No selection count here: this caller's loop reads the remaining candidates
    # instead, and the positions are written for *every* vertex because the caller ping-pongs two
    # buffers and an unwritten slot would hold an iteration-old value.
    #
    # Measured 1.12x on ``flatten_degree3_vertices`` at 3 413 flattened vertices, byte-identical.
    # The ring walk is done twice in the taken branch -- once to decide, once to average -- and
    # that is still cheaper than writing the mask out and reading it back.
    vertex = wp.int32(wp.tid())
    if not wins_degree3_conflict(faces, ring_offsets, ring_halfedges, candidate, vertex):
        out_candidate[vertex] = candidate[vertex]
        out_positions[vertex] = positions[vertex]
        return
    out_candidate[vertex] = False
    total = wp.vec3()
    for slot in range(ring_offsets[vertex], ring_offsets[vertex + 1]):
        total += positions[halfedge_destination(faces, ring_halfedges[slot])]
    out_positions[vertex] = total / wp.float32(3.0)


@wp.kernel
def sever_barrier_pairs(
    shared: wp.array2d[wp.int32],
    barrier_keys: wp.array[wp.uint64],
    adjacency: wp.array2d[wp.int32],
    out_pairs: wp.array2d[wp.int32],
    out_barrier: wp.array[wp.bool],
) -> None:
    # The face pairs with every pair across a barrier edge turned into a self-loop, so a labelling
    # over them sees the barrier as cut: a self-loop joins nothing, and the pair keeps its row, so
    # no compaction is needed. A pair is across a barrier when its shared edge's
    # ``array.pack_edge_key`` at ``INDEX_RADIX_PAIR`` is in the sorted ``barrier_keys`` -- a
    # few hundred loop edges, searched here rather than hashed and matched in a pass of their own.
    # Written to a new buffer rather than in place, with the mask beside it, so the caller keeps the
    # faces across each barrier edge.
    k = wp.int32(wp.tid())
    key = pack_edge_key(shared[k, 0], shared[k, 1], wp.uint64(INDEX_RADIX_PAIR))
    cut = binary_search_sorted_contains(barrier_keys, key)
    f0 = adjacency[k, 0]
    out_pairs[k, 0] = f0
    out_pairs[k, 1] = wp.where(cut, f0, adjacency[k, 1])
    out_barrier[k] = cut


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not two.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was derived
    and what forks a ``wp.map`` module; only this module's *own* forking ops belong here (the shared
    builtins are declared there).

    Two ops: ``is_interior_degree3``, mapped over ``(ring_offsets[:-1], ring_offsets[1:],
    is_boundary)`` by ``repair.flatten_degree3_vertices``, and ``is_interior_degree3_in_region``,
    the same with the caller's ``region`` mask. The length-1 rows are not speculative -- a ring CSR
    over a one-vertex mesh makes both offset slices length 1, and the broadcast mask is part of the
    cache key, so that call would fork the module on first use.
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (is_interior_degree3, (dense(wp.int32), dense(wp.int32), dense(wp.bool)), wp.bool),
            (is_interior_degree3, (single(wp.int32), single(wp.int32), single(wp.bool)), wp.bool),
            (
                is_interior_degree3_in_region,
                (dense(wp.int32), dense(wp.int32), dense(wp.bool), dense(wp.bool)),
                wp.bool,
            ),
            (
                is_interior_degree3_in_region,
                (single(wp.int32), single(wp.int32), single(wp.bool), single(wp.bool)),
                wp.bool,
            ),
        ]
    )


_declare_map_kernels()
