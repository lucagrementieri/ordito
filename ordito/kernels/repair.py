import warp as wp

from ordito.constants import INDEX_RADIX_PAIR
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_prehook_pair,
    write_corner_component_rank,
)
from ordito.kernels.array import (
    binary_search_index,
    binary_search_sorted_contains,
    declare_map_signatures,
    map_probe,
    map_probe_single,
    pack_edge_key,
    pack_ranked_key,
    scanned_count,
    update_argmin_pair,
)
from ordito.kernels.bounds import packed_box_diagonal
from ordito.kernels.grouping import hash_slot, next_slot
from ordito.kernels.halfedge import (
    HALFEDGE_DEFECT_NONE,
    halfedge_destination,
    halfedge_next,
    mate_twin_defect,
    next_boundary_halfedge,
)
from ordito.kernels.predicates import triangle_aspect_ratio, triangle_normal
from ordito.kernels.selection import face_selected_by_vertices, mark_submesh_face
from ordito.kernels.triangles import (
    QUALITY_AREA,
    corner_triple,
    face_normals_and_area,
    face_signed_volume,
    face_vertices,
    triangle_cross,
    triangle_nondegenerate,
    triangle_quality,
    write_corner_triple,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def mark_referenced(faces: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    # A ``1`` flag at every vertex a face names, in the ``int32`` the scan below reads, so no mask
    # has to be converted first. Out-of-range entries -- the ``-1`` sentinels ``repair`` preserves
    # through a remap -- are dropped rather than written off the end.
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
    # face, gathered via ``inverse``) scattered into per-group stats. Every table arrives zeroed,
    # so a ``first_*`` slot holds the smallest face index of its sign class *reflected* as
    # ``n_faces - f`` (an ``atomic_max``), and zero means no member; ``first_member_index``
    # reads it back.
    f = wp.int32(wp.tid())
    reflected = faces.shape[0] // 3 - f
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
    wp.atomic_max(out_first_member, ui, reflected)
    if consistent:
        wp.atomic_add(out_signed_count, ui, wp.int32(1))
        wp.atomic_max(out_first_positive, ui, reflected)
    else:
        wp.atomic_add(out_signed_count, ui, wp.int32(-1))
        wp.atomic_max(out_first_negative, ui, reflected)


@wp.func
def first_member_index(reflected: wp.array[wp.int32], n_faces: wp.int32, ui: wp.int32) -> wp.int32:
    # The smallest face index ``scatter_duplicate_face_stats`` reflected into slot ``ui``.
    return n_faces - reflected[ui]


@wp.kernel
def resolve_duplicate_groups(
    member_count: wp.array[wp.int32],
    signed_count: wp.array[wp.int32],
    first_member: wp.array[wp.int32],
    first_positive: wp.array[wp.int32],
    first_negative: wp.array[wp.int32],
    n_faces: wp.int32,
    out_keep: wp.array[wp.int32],
    out_keep_flags: wp.array[wp.int32],
    out_error_group: wp.array[wp.int32],
) -> None:
    # Keep-decision per duplicate group (igl::resolve_duplicated_faces): singletons stay; a net
    # +1/-1 orientation keeps the first member of the majority sign; a cancelling group drops;
    # anything else is non-orientable. The smallest offending group index is reported reflected,
    # as ``n_groups - ui`` into a zeroed word, so zero means none.
    #
    # ``out_keep_flags`` is ``out_keep >= 0`` as a 0/1 ``int32``, written by the thread that
    # decided it, so the caller scans it in place to number the kept groups without a second pass
    # re-reading ``out_keep`` for the sign.
    ui = wp.int32(wp.tid())
    count = signed_count[ui]
    keep = wp.int32(-1)
    if member_count[ui] == 1:
        keep = first_member_index(first_member, n_faces, ui)
    elif count == 1:
        keep = first_member_index(first_positive, n_faces, ui)
    elif count == -1:
        keep = first_member_index(first_negative, n_faces, ui)
    elif count != 0:
        wp.atomic_max(out_error_group, 0, member_count.shape[0] - ui)
    out_keep[ui] = keep
    out_keep_flags[ui] = wp.where(keep >= 0, wp.int32(1), wp.int32(0))


@wp.kernel
def compact_kept_duplicate_faces(
    faces: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    keep_ranks: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_kept: wp.array[wp.int32],
) -> None:
    # Each kept group's face, in ascending group order: ``keep_ranks`` is
    # ``resolve_duplicate_groups``' keep flags scanned in place (inclusive), so a kept group's entry
    # less one is its output row.
    ui = wp.int32(wp.tid())
    f = keep[ui]
    if f < 0:
        return
    row = keep_ranks[ui] - 1
    out_kept[row] = f
    for k in range(3):
        out_faces[3 * row + k] = faces[3 * f + k]


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


@wp.func
def mark_face_fate(
    faces: wp.array[wp.int32],
    f: wp.int32,
    keep: wp.bool,
    out_keep: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # Record a face's keep decision for the submesh compaction: the mask entry it reads, and, when
    # kept, the face and its three vertices flagged in the zeroed ``(n_faces + n_vertices,)`` rank
    # buffer, as ``selection.mark_submesh_faces_and_vertices`` does from a mask. Shared by the
    # kernels that decide each face's fate inline (``remap_mark_nondegenerate_faces``,
    # ``mark_kept_components``); they differ only in the decision.
    out_keep[f] = keep
    if keep:
        mark_submesh_face(faces, out_keep.shape[0], f, out_flags)


@wp.kernel
def remap_mark_nondegenerate_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_keep: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # One collapse pass after the merge closure: remap each corner to its class label, test the
    # remapped triangle with ``face_nondegenerate_mask``'s predicate, and mark the survivors for the
    # submesh compaction, which reads ``out_faces`` as its face buffer.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    a = labels[i0]
    b = labels[i1]
    c = labels[i2]
    write_corner_triple(out_faces, f, a, b, c)
    keep = triangle_nondegenerate(vertices[a], vertices[b], vertices[c])
    mark_face_fate(out_faces, f, keep, out_keep, out_flags)


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


# ``remove_small_components``' four criteria, as the selector ``component_kept`` branches on.
COMPONENT_MIN_FACES = wp.int32(0)
COMPONENT_MIN_AREA = wp.int32(1)
COMPONENT_MIN_DIAMETER = wp.int32(2)
COMPONENT_LARGEST = wp.int32(3)


@wp.func
def component_kept(
    criterion: wp.int32,
    group: wp.int32,
    counts: wp.array[wp.int32],
    statistic: wp.array[wp.float32],
    corners: wp.array[wp.float32],
    best: wp.array[wp.int64],
    min_count: wp.int32,
    min_value: wp.float32,
) -> wp.bool:
    # Whether ``group`` survives ``remove_small_components``' criterion, read from the one per-group
    # table that criterion accumulated (the others are never read, so the wrapper passes ``None``;
    # the branch is warp-uniform). Every bound is inclusive.
    #
    # - ``COMPONENT_MIN_FACES``: the group's member count (``scatter.count_occurrences``).
    # - ``COMPONENT_MIN_AREA``: its summed area (``scatter_face_area_by_group``).
    # - ``COMPONENT_MIN_DIAMETER``: its bounding-box diagonal, decoded from the packed box
    #   ``scatter.scatter_group_bounds`` accumulated, so no per-group diagonal table is written.
    # - ``COMPONENT_LARGEST``: the group ``reduce_largest_group`` chose. Recomputing the key and
    #   testing it against the reduced maximum keeps this readback-free: the winning group index is
    #   never brought to the host, and the ``-1`` seed keeps nothing when there are no groups.
    if criterion == COMPONENT_MIN_FACES:
        return counts[group] >= min_count
    if criterion == COMPONENT_MIN_AREA:
        return statistic[group] >= min_value
    if criterion == COMPONENT_MIN_DIAMETER:
        return packed_box_diagonal(corners, group) >= min_value
    return pack_ranked_key(counts[group], group) == best[0]


@wp.kernel
def mark_kept_components(
    faces: wp.array[wp.int32],
    groups: wp.array[wp.int32],
    criterion: wp.int32,
    counts: wp.array[wp.int32],
    statistic: wp.array[wp.float32],
    corners: wp.array[wp.float32],
    best: wp.array[wp.int64],
    min_count: wp.int32,
    min_value: wp.float32,
    out_keep: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # The keep decision and the submesh extraction's marking pass in one launch: each face reads its
    # group's fate (``component_kept``) and, when kept, flags itself and its vertices in the zeroed
    # ``(n_faces + n_vertices,)`` rank buffer exactly as
    # ``selection.mark_submesh_faces_and_vertices`` does from a mask. ``out_keep`` is that mask,
    # which the compaction still reads.
    f = wp.int32(wp.tid())
    keep = component_kept(
        criterion, groups[f], counts, statistic, corners, best, min_count, min_value
    )
    mark_face_fate(faces, f, keep, out_keep, out_flags)


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


@wp.func
def corner_merge_link(
    forward_count: wp.array[wp.int32],
    backward_count: wp.array[wp.int32],
    forward_corner: wp.array[wp.int32],
    backward_corner: wp.array[wp.int32],
    link: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # One link of the corner graph: link ``2e`` joins the corners at edge ``e``'s low endpoint and
    # link ``2e + 1`` those at its high endpoint, one corner from each of the edge's two faces.
    #
    # Nodes are ``(face, vertex slot)`` pairs under the same ``3f + k`` numbering, so node
    # ``3f + k`` *is* the corner whose vertex is ``faces[3f + k]``, and a link only ever joins two
    # copies of one original vertex. Connected components of this graph are the vertex copies.
    #
    # A non-mergeable edge gives the self-loop ``(0, 0)``, which unions nothing.
    e = link // 2
    if forward_count[e] != 1 or backward_count[e] != 1:
        return wp.int32(0), wp.int32(0)
    forward = forward_corner[e]
    backward = backward_corner[e]
    # The forward half-edge runs (low, high) from its own corner; the backward one runs (high, low),
    # so its *next* slot holds the low endpoint.
    if link % 2 == 0:
        return forward, halfedge_next(backward)  # low endpoint, both faces
    return halfedge_next(forward), backward  # high endpoint, both faces


@wp.kernel
def corner_merge_prehook(
    forward_count: wp.array[wp.int32],
    backward_count: wp.array[wp.int32],
    forward_corner: wp.array[wp.int32],
    backward_corner: wp.array[wp.int32],
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_init_parent_edges`` over the corner graph, each link formed by
    # ``corner_merge_link`` in the thread rather than read from a materialised edge list. Runs over
    # an identity ``parents``; changes no root. ``corner_merge_hook`` hooks the same links.
    link = wp.int32(wp.tid())
    a, b = corner_merge_link(forward_count, backward_count, forward_corner, backward_corner, link)
    ecl_prehook_pair(parents, a, b)


@wp.kernel
def corner_merge_hook(
    forward_count: wp.array[wp.int32],
    backward_count: wp.array[wp.int32],
    forward_corner: wp.array[wp.int32],
    backward_corner: wp.array[wp.int32],
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_hook_edges`` over the same links. Every union points the larger
    # root at the smaller, so each component's root is its smallest corner whichever order the
    # unions ran in.
    link = wp.int32(wp.tid())
    a, b = corner_merge_link(forward_count, backward_count, forward_corner, backward_corner, link)
    ecl_hook_pair(parents, a, b)


@wp.kernel
def scatter_split_corners(
    faces: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    roots: wp.array[wp.int32],
    root_ranks: wp.array[wp.int32],
    out_new_faces: wp.array[wp.int32],
    out_source: wp.array[wp.int32],
    out_new_vertices: wp.array[wp.vec3],
) -> None:
    # Number each corner's component and, from its root corner alone, write the component's source
    # vertex and position. ``seams.scatter_corner_values`` is the same ranking for a value the
    # corners share, with no source map and no root test.
    h = wp.int32(wp.tid())
    root, rank = write_corner_component_rank(roots, root_ranks, h, out_new_faces)
    if root == h:
        v = faces[h]
        out_source[rank] = v
        out_new_vertices[rank] = vertices[v]


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
) -> wp.int32:
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
    # The arrival slot: ``0`` for exactly one halfedge of every referenced vertex.
    return slot


@wp.kernel
def degree3_fan_tables_input(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
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
    # its first halfedge, in place of a copy and a fill; and it counts its halfedge's twin defect
    # from the input's halfedge mates into ``out_defects`` -- the input validation, riding on
    # this launch and on the pass's one readback -- with, in its third slot, the input's referenced
    # vertex count: each removal unreferences exactly its centre, so the output's vertex count is
    # this less the removals and the final compaction needs no readback. ``degree3_fan_tables`` is
    # every later pass.
    h = wp.int32(wp.tid())
    _twin, defect = mate_twin_defect(faces, mates[h], h)
    if defect != HALFEDGE_DEFECT_NONE:
        wp.atomic_add(out_defects, defect, 1)
    out_faces[h] = faces[h]
    if h % 3 == 0:
        out_kept[h // 3] = 1
    if record_degree3_halfedge(faces, h, out_counts, out_link_sums, out_fans) == 0:
        wp.atomic_add(out_defects, 2, 1)


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
    # The two rim faces' normals are formed here, from the only faces read, rather than taken from
    # a per-face normal buffer filled for the whole mesh.
    entering_normal, _ = face_normals_and_area(vertices, faces, h // 3)
    if wp.dot(normal, entering_normal) < min_normal_dot:
        return
    outgoing_normal, _ = face_normals_and_area(vertices, faces, outgoing // 3)
    if wp.dot(normal, outgoing_normal) < min_normal_dot:
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


# ``remove_tunnels``' complete search: the shortest non-separating loop through each root.
#
# A loop is non-separating exactly when its mod-2 homology class is non-zero, and that class is
# read off a **cocycle**: an integer per edge whose sum around every face is even, so it sums to
# the same parity around any two homologous loops. For a simple loop ``b`` of a consistently wound
# surface, marking every edge that leaves ``b`` to its left (an edge whose both ends leave ``b`` to
# the left counted twice) is one: a face meets ``b``'s left side in two such edges or none. Its sum
# around a loop ``g`` is the parity of ``g`` crossing ``b``, and the crossing parities with a basis
# of loops determine ``g``'s class, since the intersection form of a closed orientable surface is
# non-degenerate. So every basis loop ``i`` adds its own 64-bit word ``tunnel_loop_word(i)`` to the
# edges it marks, xor being addition mod 2, and a loop's class is non-zero exactly when the xor of
# its edges' words is (but for a 2**-64 chance per class that a non-empty set of words xors to
# zero, which can only hide a loop, never present a separating one as non-separating).

# The ``splitmix64`` finalizer: distinct well-mixed words for consecutive loop indices.
_LOOP_WORD_INCREMENT = wp.uint64(0x9E3779B97F4A7C15)
_LOOP_WORD_MIX_1 = wp.uint64(0xBF58476D1CE4E5B9)
_LOOP_WORD_MIX_2 = wp.uint64(0x94D049BB133111EB)


@wp.func
def tunnel_loop_word(i: wp.int32) -> wp.uint64:
    """Return basis loop ``i``'s cocycle word: ``splitmix64`` of ``i + 1``."""
    z = wp.uint64(i + 1) * _LOOP_WORD_INCREMENT
    z = (z ^ (z >> wp.uint64(30))) * _LOOP_WORD_MIX_1
    z = (z ^ (z >> wp.uint64(27))) * _LOOP_WORD_MIX_2
    return z ^ (z >> wp.uint64(31))


@wp.func
def ring_slot_toward(
    faces: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    start: wp.int32,
    end: wp.int32,
    target: wp.int32,
) -> wp.int32:
    """Ring slot in ``[start, end)`` of the halfedge ending at ``target``, or ``-1``."""
    for slot in range(start, end):
        if halfedge_destination(faces, ring_halfedges[slot]) == target:
            return slot
    return wp.int32(-1)


@wp.kernel
def tunnel_loop_cocycle(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    loops: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    out_cocycle: wp.array[wp.uint64],
) -> None:
    # One thread per loop position ``u``, with loop neighbours ``p`` before and ``n`` after: the
    # halfedges counter-clockwise strictly between ``u -> n`` and ``u -> p`` leave the loop to its
    # left, and each takes the loop's word on both of its halfedges, so a traversal reads the
    # edge's whole class from whichever end it starts. ``out_cocycle`` is zeroed by the caller;
    # loops are simple and at least three long, as the homology generators are.
    j = wp.int32(wp.tid())
    i = binary_search_index(loop_offsets, j) - 1
    start = loop_offsets[i]
    size = loop_offsets[i + 1] - start
    position = j - start
    u = loops[j]
    after = loops[start + (position + 1) % size]
    before = loops[start + (position + size - 1) % size]
    ring_start = ring_offsets[u]
    degree = ring_offsets[u + 1] - ring_start
    toward_after = ring_slot_toward(faces, ring_halfedges, ring_start, ring_start + degree, after)
    toward_before = ring_slot_toward(faces, ring_halfedges, ring_start, ring_start + degree, before)
    if toward_after < 0 or toward_before < 0:
        return
    word = tunnel_loop_word(i)
    slot = ring_start + (toward_after - ring_start + 1) % degree
    while slot != toward_before:
        h = ring_halfedges[slot]
        wp.atomic_xor(out_cocycle, h, word)
        wp.atomic_xor(out_cocycle, twins[h], word)
        slot = ring_start + (slot - ring_start + 1) % degree


@wp.func
def tunnel_slot_find(
    table: wp.array[wp.int32], slot_vertex: wp.array[wp.int32], vertex: wp.int32
) -> wp.int32:
    """Return the root's slot holding ``vertex``, or ``-1``; the table maps vertices to slots."""
    mask = table.shape[0] - 1
    h = hash_slot(vertex, mask)
    stored = table[h]
    while stored >= 0:
        if slot_vertex[stored] == vertex:
            return stored
        h = next_slot(h, mask)
        stored = table[h]
    return wp.int32(-1)


@wp.func
def tunnel_slot_insert(table: wp.array[wp.int32], vertex: wp.int32, slot: wp.int32) -> None:
    """Record ``vertex -> slot``; the vertex must be absent and the table less than full."""
    mask = table.shape[0] - 1
    h = hash_slot(vertex, mask)
    while table[h] >= 0:
        h = next_slot(h, mask)
    table[h] = slot


@wp.func
def tunnel_heap_before(dist: wp.array[wp.float32], a: wp.int32, b: wp.int32) -> wp.bool:
    """Heap order: nearer first, the earlier slot on a tie, so the search is deterministic."""
    return dist[a] < dist[b] or (dist[a] == dist[b] and a < b)


@wp.func
def tunnel_heap_sift_up(
    heap: wp.array[wp.int32], position: wp.array[wp.int32], dist: wp.array[wp.float32], k: wp.int32
) -> None:
    """Move heap entry ``k`` toward the top until its parent comes first."""
    slot = heap[k]
    while k > 0:
        parent = (k - 1) // 2
        above = heap[parent]
        if not tunnel_heap_before(dist, slot, above):
            break
        heap[k] = above
        position[above] = k
        k = parent
    heap[k] = slot
    position[slot] = k


@wp.func
def tunnel_heap_pop(
    heap: wp.array[wp.int32],
    position: wp.array[wp.int32],
    dist: wp.array[wp.float32],
    size: wp.int32,
) -> wp.int32:
    """Remove and return the top slot of a heap of ``size >= 1``, marking it settled (``-1``)."""
    top = heap[0]
    position[top] = wp.int32(-1)
    last = size - 1
    if last == 0:
        return top
    slot = heap[last]
    k = wp.int32(0)
    while True:
        child = 2 * k + 1
        if child >= last:
            break
        if child + 1 < last and tunnel_heap_before(dist, heap[child + 1], heap[child]):
            child += 1
        if not tunnel_heap_before(dist, heap[child], slot):
            break
        heap[k] = heap[child]
        position[heap[k]] = k
        k = child
    heap[k] = slot
    position[slot] = k
    return top


@wp.func
def tunnel_tree_depth(parent: wp.array[wp.int32], slot: wp.int32) -> wp.int32:
    """Hops from ``slot`` up to the root of its search tree."""
    depth = wp.int32(0)
    while parent[slot] >= 0:
        slot = parent[slot]
        depth += 1
    return depth


@wp.func
def tunnel_write_loop(
    parent: wp.array[wp.int32],
    slot_vertex: wp.array[wp.int32],
    s: wp.int32,
    t: wp.int32,
    out_loop: wp.array[wp.int32],
) -> wp.int32:
    """
    Write the simple cycle that edge ``(s, t)`` closes in the search tree; return its length.

    The loop through the root is the two tree paths and the edge; the stretch both paths share
    from the root to their lowest common ancestor is walked twice and cancels mod 2, so the cycle
    written is the rest -- the ancestor, down to ``s``, across to ``t`` and back up -- in the same
    class and no longer.
    """
    depth_s = tunnel_tree_depth(parent, s)
    depth_t = tunnel_tree_depth(parent, t)
    a = s
    b = t
    for _ in range(depth_t, depth_s):
        a = parent[a]
    for _ in range(depth_s, depth_t):
        b = parent[b]
    while a != b:
        a = parent[a]
        b = parent[b]
    ancestor_depth = tunnel_tree_depth(parent, a)
    side_s = depth_s - ancestor_depth
    side_t = depth_t - ancestor_depth
    out_loop[0] = slot_vertex[a]
    walk = s
    for k in range(side_s, 0, -1):
        out_loop[k] = slot_vertex[walk]
        walk = parent[walk]
    walk = t
    for k in range(side_s + 1, side_s + side_t + 1):
        out_loop[k] = slot_vertex[walk]
        walk = parent[walk]
    return side_s + side_t + 1


@wp.kernel
def shortest_nontrivial_loop_per_root(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    cocycle: wp.array[wp.uint64],
    roots: wp.array[wp.int32],
    radius: wp.float32,
    bound: wp.float32,
    table: wp.array2d[wp.int32],
    slot_vertex: wp.array2d[wp.int32],
    slot_dist: wp.array2d[wp.float32],
    slot_parent: wp.array2d[wp.int32],
    slot_class: wp.array2d[wp.uint64],
    heap: wp.array2d[wp.int32],
    heap_position: wp.array2d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # One thread per root, over its own scratch rows (``table`` pre-filled with ``-1``): Dijkstra
    # over edge lengths, truncated at ``radius``, looking for the shortest loop "tree path, edge,
    # tree path" whose class is non-zero (``tunnel_loop_cocycle``; a tree edge closes nothing and
    # reads zero by construction). Erickson and Whittlesey's argument, which needs no unique
    # shortest paths: walking a shortest non-separating closed walk through the root, the first
    # prefix whose closure by a tree path is non-separating ends on an edge whose tree loop is
    # non-separating and exactly as long. So the minimum over roots on a loop is that loop's
    # length, and a loop of length at most ``bound`` stays within ``bound / 2`` of its root, which
    # is ``radius``.
    #
    # An edge is tested when the later of its two ends settles, both distances then being final.
    # A loop through an end at distance ``d`` is at least ``2 d`` long, so once a loop of length
    # ``best`` is found nothing farther than ``best / 2`` is inserted, and the search stops when
    # the settled distance reaches it: a root near a short loop explores only that far.
    #
    # ``out_count`` is the written cycle's vertex count, ``0`` when no loop is non-separating within
    # ``bound``, and ``-1`` when the ball outgrew the rows, which the caller re-runs wider. The
    # cycle is written over the root's ``heap`` row, which the search no longer reads.
    r = wp.int32(wp.tid())
    row_table = table[r]
    row_vertex = slot_vertex[r]
    row_dist = slot_dist[r]
    row_parent = slot_parent[r]
    row_class = slot_class[r]
    row_heap = heap[r]
    row_position = heap_position[r]
    capacity = row_vertex.shape[0]

    root = roots[r]
    row_vertex[0] = root
    row_dist[0] = wp.float32(0.0)
    row_parent[0] = wp.int32(-1)
    row_class[0] = wp.uint64(0)
    tunnel_slot_insert(row_table, root, wp.int32(0))
    row_heap[0] = wp.int32(0)
    row_position[0] = wp.int32(0)
    heap_size = wp.int32(1)
    count = wp.int32(1)
    overflow = wp.bool(False)
    best = wp.float32(wp.inf)
    best_s = wp.int32(-1)
    best_t = wp.int32(-1)
    reach_limit = radius
    while heap_size > 0 and not overflow:
        s = tunnel_heap_pop(row_heap, row_position, row_dist, heap_size)
        heap_size -= 1
        settled = row_dist[s]
        if wp.float32(2.0) * settled >= best:
            break
        u = row_vertex[s]
        position_u = vertices[u]
        for k in range(ring_offsets[u], ring_offsets[u + 1]):
            h = ring_halfedges[k]
            w = halfedge_destination(faces, h)
            edge = wp.length(vertices[w] - position_u)
            t = tunnel_slot_find(row_table, row_vertex, w)
            if t >= 0 and row_position[t] < 0:
                if (row_class[s] ^ row_class[t] ^ cocycle[h]) != wp.uint64(0):
                    length = settled + edge + row_dist[t]
                    if length <= bound and length < best:
                        best = length
                        best_s = s
                        best_t = t
                        reach_limit = wp.min(radius, wp.float32(0.5) * best)
                continue
            reach = settled + edge
            if reach > reach_limit:
                continue
            if t < 0:
                if count == capacity:
                    overflow = wp.bool(True)
                    break
                t = count
                count += 1
                row_vertex[t] = w
                tunnel_slot_insert(row_table, w, t)
                row_dist[t] = reach
                row_parent[t] = s
                row_class[t] = row_class[s] ^ cocycle[h]
                row_heap[heap_size] = t
                tunnel_heap_sift_up(row_heap, row_position, row_dist, heap_size)
                heap_size += 1
            elif reach < row_dist[t]:
                row_dist[t] = reach
                row_parent[t] = s
                row_class[t] = row_class[s] ^ cocycle[h]
                tunnel_heap_sift_up(row_heap, row_position, row_dist, row_position[t])
    if overflow:
        out_count[r] = wp.int32(-1)
        return
    if best_s < 0:
        out_count[r] = wp.int32(0)
        return
    out_count[r] = tunnel_write_loop(row_parent, row_vertex, best_s, best_t, row_heap)


@wp.kernel
def pack_root_loops(
    rows: wp.array2d[wp.int32], offsets: wp.array[wp.int32], out_loops: wp.array[wp.int32]
) -> None:
    # Root ``r``'s cycle, the head of its row, into its slice of the packed buffer.
    r = wp.int32(wp.tid())
    start = offsets[r]
    for k in range(offsets[r + 1] - start):
        out_loops[start + k] = rows[r, k]


@wp.kernel
def flag_any_selected(mask: wp.array[wp.bool], out_state: wp.array[wp.int32]) -> None:
    # ``fix_self_intersections``' "is any face bad" as a raised word rather than a readback of the
    # whole mask: every selected face stores the same ``1`` into slot 0, so the store order is
    # irrelevant and the host reads four bytes.
    f = wp.int32(wp.tid())
    if mask[f]:
        out_state[0] = wp.int32(1)


@wp.kernel
def face_mask_from_vertex_mask_flag_unselected(
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    out_face_mask: wp.array[wp.bool],
    out_state: wp.array[wp.int32],
) -> None:
    # The dilation's last any-corner lookup (``selection.face_mask_from_vertex_mask``), raising
    # slot 1 wherever a face stays unselected, so "the region swallowed the mesh" is that word
    # still reading zero. A flag raised by the faces outside the region rather than a count of the
    # faces inside it: a count would put one atomic per selected face on one address.
    f = wp.int32(wp.tid())
    selected = face_selected_by_vertices(faces, vertex_mask, wp.bool(False), f)
    out_face_mask[f] = selected
    if not selected:
        out_state[1] = wp.int32(1)


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
