import warp as wp

from ordito.kernels.adjacency import write_edge_row, write_face_edge_keys
from ordito.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_prehook_pair,
    find_representative,
)
from ordito.kernels.array import (
    binary_search_index,
    binary_search_sorted_contains,
    masked_at,
    pack_directed_key,
    pack_edge_key,
    scanned_count,
)
from ordito.kernels.grouping import sorted_run_of_length
from ordito.kernels.halfedge import halfedge_endpoints
from ordito.kernels.scatter import mark_corners
from ordito.kernels.triangles import corner_triple

wp.set_module_options({"enable_backward": False})


@wp.kernel
def mark_indexed_face_vertices(
    faces: wp.array[wp.int32], face_indices: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # Flag every vertex a listed face references, in a zeroed per-vertex ``int32`` array the
    # caller scans in place: the vertex indices are bounded by the vertex count, so the scan of a
    # mask is the sorted unique set ``unique_1d`` would hash and sort for, and a listed duplicate
    # face only re-marks the corners it already marked.
    i = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, face_indices[i])
    mark_corners(out_flags, 0, a, b, c, wp.int32(1))


@wp.kernel
def compact_indexed_submesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    inclusive: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_vertex_index: wp.array[wp.int32],
) -> None:
    # Both halves of the index-driven extraction in one launch over ``max(k, n_vertices)``
    # threads, from ``mark_indexed_face_vertices``' flags scanned in place: listed face ``t`` is
    # written at row ``t`` with each corner at its vertex's compact rank, and a flagged vertex
    # ``t`` is written at that rank. ``out_vertex_index`` is optional (``None``).
    t = wp.int32(wp.tid())
    if t < face_indices.shape[0]:
        a, b, c = corner_triple(faces, face_indices[t])
        out_faces[3 * t] = inclusive[a] - 1
        out_faces[3 * t + 1] = inclusive[b] - 1
        out_faces[3 * t + 2] = inclusive[c] - 1
    if t < vertices.shape[0]:
        start, count = scanned_count(inclusive, t)
        if count != 0:
            out_vertices[start] = vertices[t]
            if out_vertex_index.shape[0] > 0:
                out_vertex_index[start] = t


@wp.kernel
def pack_group_vertex_keys(
    faces: wp.array[wp.int32],
    group_face_indices: wp.array[wp.int32],
    group_offsets: wp.array[wp.int32],
    radix: wp.int64,
    out_keys: wp.array[wp.int64],
) -> None:
    # ``group * radix + vertex`` for every corner of every selected face, with ``radix`` the source
    # vertex count. Ascending key order is exactly ``(group, vertex)`` lexicographic order, which is
    # what lets a single global ``unique_1d`` do the work of one dedup per group and still emit each
    # group's vertices ascending by original index.
    #
    # ``group_offsets`` is the CSR start of each group, so the group owning position ``p`` is the
    # last one starting at or before it — a binary search rather than a materialized per-position
    # label array. ``out_keys`` is ``wp.int64`` concretely, not ``wp.Int``: a generic packer cannot
    # construct the promoted value portably.
    corner = wp.int32(wp.tid())
    position = corner // 3
    group = binary_search_index(group_offsets, position) - 1
    face = group_face_indices[position]
    out_keys[corner] = wp.int64(group) * radix + wp.int64(faces[face * 3 + corner % 3])


@wp.kernel
def decode_group_vertex_keys(
    unique_keys: wp.array[wp.int64],
    radix: wp.int64,
    vertices: wp.array[wp.vec3],
    out_slot_groups: wp.array[wp.int32],
    out_group_counts: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
) -> None:
    # One thread per sorted unique ``(group, vertex)`` slot: record the slot's group, count it
    # into that group's histogram (the group's vertex count), and gather the source position it
    # names -- the three things the slot's key is needed for, read off one decode. The vertex id
    # itself is never stored, since the position gather is its only consumer.
    slot = wp.int32(wp.tid())
    key = unique_keys[slot]
    # The inverse of ``pack_group_vertex_keys``: one divmod.
    group = wp.int32(key // radix)
    vertex = wp.int32(key % radix)
    out_slot_groups[slot] = group
    wp.atomic_add(out_group_counts, group, wp.int32(1))
    out_vertices[slot] = vertices[vertex]


@wp.kernel
def local_corner_indices(
    inverse: wp.array[wp.int32],
    slot_groups: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The compacted, zero-based vertex index each selected corner refers to: its slot's rank within
    # its own group. Resolved per corner through ``inverse`` rather than tabulated per slot and
    # gathered, since the corner is the only reader of a slot's rank.
    corner = wp.int32(wp.tid())
    slot = inverse[corner]
    out_faces[corner] = slot - vertex_offsets[slot_groups[slot]]


# The mask-driven submesh extraction shares one ``(n_faces + n_vertices,)`` rank buffer: the kept
# faces' 0/1 flags followed by their vertices' referenced flags, scanned inclusively as one array.
# A kept face's compact slot is then ``ranks[f] - 1`` and a kept vertex's is
# ``ranks[n_faces + v] - ranks[n_faces - 1] - 1``, so one scan orders both halves and the two
# output sizes come out of one small readback. Both orders are ascending by input index, which is
# what the index-driven path's sorted ``unique_1d`` produces.


@wp.kernel
def mark_submesh_faces_and_vertices(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    keep_masked: wp.bool,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``out_flags`` arrives zeroed. A face is kept when its mask entry equals ``keep_masked``, so
    # one kernel serves a selection and its complement.
    f = wp.int32(wp.tid())
    if face_mask[f] != keep_masked:
        return
    n_faces = face_mask.shape[0]
    a, b, c = corner_triple(faces, f)
    out_flags[f] = 1
    mark_corners(out_flags, n_faces, a, b, c, wp.int32(1))


@wp.kernel
def submesh_counts(
    ranks: wp.array[wp.int32], n_faces: wp.int32, out_counts: wp.array[wp.int32]
) -> None:
    # The kept face count and the kept vertex count, side by side for a single readback.
    face_total = ranks[n_faces - 1]
    out_counts[0] = face_total
    out_counts[1] = ranks[ranks.shape[0] - 1] - face_total


@wp.kernel
def compact_submesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    keep_masked: wp.bool,
    ranks: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_vertex_index: wp.array[wp.int32],
) -> None:
    # Both halves of the extraction in one launch over ``max(n_faces, n_vertices)`` threads: thread
    # ``t`` compacts face ``t`` and vertex ``t`` where each exists. The two halves read the same
    # finished scan and write disjoint outputs, so nothing orders them.
    t = wp.int32(wp.tid())
    n_faces = face_mask.shape[0]
    offset = ranks[n_faces - 1] + 1
    if t < n_faces and face_mask[t] == keep_masked:
        # A kept face lands at its rank, its corners at their vertices' ranks.
        a, b, c = corner_triple(faces, t)
        slot = 3 * (ranks[t] - 1)
        out_faces[slot] = ranks[n_faces + a] - offset
        out_faces[slot + 1] = ranks[n_faces + b] - offset
        out_faces[slot + 2] = ranks[n_faces + c] - offset
    if t < vertices.shape[0]:
        # A vertex is kept when its inclusive rank steps up; for vertex 0 the step is taken from the
        # face half's total, which is exactly the vertex half's baseline.
        rank = ranks[n_faces + t]
        if rank != ranks[n_faces + t - 1]:
            slot = rank - offset
            out_vertices[slot] = vertices[t]
            out_vertex_index[slot] = t


@wp.func
def face_selected_by_vertices(
    faces: wp.array[wp.int32], vertex_mask: wp.array[wp.bool], require_all: wp.bool, f: wp.int32
) -> wp.bool:
    # Reduce a per-vertex selection onto its faces: keep a face when all three of its corners are
    # selected (``require_all``), or when at least one is. One warp-uniform branch rather than two
    # kernels, since the two differ by a parameter and not by an algorithm.
    #
    # This replaces the route the wrapper used to take -- ``flatnonzero`` the mask, ``array.isin``
    # the face buffer against the resulting index list, then reduce along the rows -- which
    # rebuilt as a lookup table exactly the membership the mask already *is*, at the cost of two
    # min/max reductions with a host readback each, a sort or a span-sized table, and a second
    # ``flatnonzero``. Reading the mask directly is one O(1) lookup per corner; see
    # ``masked_at`` for why the read is guarded.
    a, b, c = corner_triple(faces, f)
    hit_a = masked_at(vertex_mask, a)
    hit_b = masked_at(vertex_mask, b)
    hit_c = masked_at(vertex_mask, c)
    if require_all:
        return hit_a and hit_b and hit_c
    return hit_a or hit_b or hit_c


@wp.kernel
def face_mask_from_vertex_mask(
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    require_all: wp.bool,
    out_face_mask: wp.array[wp.bool],
) -> None:
    f = wp.int32(wp.tid())
    out_face_mask[f] = face_selected_by_vertices(faces, vertex_mask, require_all, f)


@wp.kernel
def face_flags_from_vertex_mask(
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    require_all: wp.bool,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``face_mask_from_vertex_mask`` as the ``int32`` flag a caller scans in place to compact the
    # selected face indices, with no bool mask in between.
    f = wp.int32(wp.tid())
    out_flags[f] = wp.where(
        face_selected_by_vertices(faces, vertex_mask, require_all, f), wp.int32(1), wp.int32(0)
    )


@wp.kernel
def dilate_vertex_mask(
    faces: wp.array[wp.int32],
    in_mask: wp.array[wp.bool],
    value: wp.bool,
    out_mask: wp.array[wp.bool],
) -> None:
    # One one-ring dilation round, reached through the faces: on a triangle mesh two vertices share
    # an edge exactly when they share a face, so marking every corner of a face with a selected
    # corner adds precisely the edge neighbours, and no unique-edge table has to exist. Measured
    # against the same round over a caller's unique edges, it is level at a hundred thousand faces
    # and 1.2x at tens of millions, so the edge form was retired rather than kept beside it.
    # ``out_mask`` must be pre-seeded with ``in_mask``; re-marking the selected corner is a no-op.
    #
    # ``value`` is the polarity spread: ``True`` dilates, and ``False`` dilates the *complement* --
    # a face with an unselected corner clears all three -- which is ``shrink_vertex_mask``'s erosion
    # without forming the complement on the way in or out.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    if in_mask[a] == value or in_mask[b] == value or in_mask[c] == value:
        mark_corners(out_mask, 0, a, b, c, value)


@wp.kernel
def mark_incident_vertices(
    faces: wp.array[wp.int32], face_mask: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    # Mark every corner of every selected face: the inverse direction of
    # ``face_mask_from_vertex_mask``. Concurrent writes all store ``True``, so the race is benign
    # and no atomic is needed. ``dilate_vertex_mask`` above is this composed with an any-corner
    # ``face_mask_from_vertex_mask`` in one pass, for a caller that does not keep the face mask.
    f = wp.int32(wp.tid())
    if not face_mask[f]:
        return
    a, b, c = corner_triple(faces, f)
    mark_corners(out_mask, 0, a, b, c, wp.bool(True))


@wp.kernel
def mark_region_seam(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    n: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # One thread per position of the radix-sorted halfedge keys, whose payload is each halfedge's
    # index (``adjacency.face_edge_keys_and_order``): 1 where a run of *exactly two* halfedges
    # starts -- an interior edge -- whose two faces sit on opposite sides of the region. That is
    # the seam rule, read off the sort with no unique-edge table and no per-edge counts. The flags
    # are the ``int32`` the caller scans in place.
    i = wp.int32(wp.tid())
    flag = wp.int32(0)
    if sorted_run_of_length(sorted_keys, n, i, 2):
        if face_mask[order[i] // 3] != face_mask[order[i + 1] // 3]:
            flag = wp.int32(1)
    out_flags[i] = flag


@wp.kernel
def emit_region_seam(
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    oriented: wp.bool,
    out_edges: wp.array2d[wp.int32],
) -> None:
    # Where ``mark_region_seam``'s in-place scan steps, write the seam edge at its rank: in
    # ascending key order, the region face's halfedge -- directed when ``oriented``, which puts the
    # region on its left, else ascending.
    i = wp.int32(wp.tid())
    g, count = scanned_count(inclusive, i)
    if count == 0:
        return
    h = order[i]
    if not face_mask[h // 3]:
        h = order[i + 1]
    if oriented:
        a, b = halfedge_endpoints(faces, h)
        out_edges[g, 0] = a
        out_edges[g, 1] = b
    else:
        write_edge_row(faces, h, g, out_edges)


@wp.kernel
def prehook_face_edges(faces: wp.array[wp.int32], parents: wp.array[wp.int32]) -> None:
    # ``ecl_init_parent_edges`` over the three directed edges of each face, formed in the thread:
    # the components are labelled by their smallest vertex whatever the multiplicity and order of
    # the unions, so the faces' own edges serve without an edge table.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    ecl_prehook_pair(parents, a, b)
    ecl_prehook_pair(parents, b, c)
    ecl_prehook_pair(parents, c, a)


@wp.kernel
def hook_face_edges(faces: wp.array[wp.int32], parents: wp.array[wp.int32]) -> None:
    # ``ecl_hook_edges`` over the same face edges, after ``prehook_face_edges`` has finished.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    ecl_hook_pair(parents, a, b)
    ecl_hook_pair(parents, b, c)
    ecl_hook_pair(parents, c, a)


@wp.kernel
def label_flagged_components(
    parents: wp.array[wp.int32],
    flags: wp.array[wp.bool],
    value: wp.bool,
    out_labels: wp.array[wp.int32],
    out_root_flagged: wp.array[wp.bool],
) -> None:
    # ``ecl_flatten`` and a per-component flag in one pass: each node's label is its root -- the
    # smallest node of its component, whatever order the unions ran in -- and a node whose flag
    # equals ``value`` flags its root in the zeroed ``out_root_flagged``. The per-node answer is
    # then a read through the node's own label. ``exclude_fully_selected_components`` flags a
    # component holding an *unselected* vertex; ``faces_left_of_contour`` one holding a seed face.
    # ``out_labels`` must not alias ``parents``: another thread's path halving can write a stale
    # ancestor over a label already stored. ``out_root_flagged`` may alias ``flags`` when ``value``
    # is ``True``, as ``faces_left_of_contour`` passes it -- only a flagged component's root is
    # written, and to ``True``, so a root reading its own written entry flags itself again.
    v = wp.int32(wp.tid())
    root = find_representative(parents, v)
    out_labels[v] = root
    if flags[v] == value:
        out_root_flagged[root] = True


@wp.kernel
def keep_selected_by_component(
    mask: wp.array[wp.bool],
    labels: wp.array[wp.int32],
    keep: wp.array[wp.bool],
    out_mask: wp.array[wp.bool],
) -> None:
    # Stay selected only if the vertex was selected and its component is not fully selected -- has
    # at least one unselected vertex. Both reads are at this thread's own index, so the per-vertex
    # copy of the component flag a Python-scope gather would build never exists.
    v = wp.int32(wp.tid())
    out_mask[v] = mask[v] and keep[labels[v]]


@wp.func
def open_dual_twin(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    contour_keys_sorted: wp.array[wp.uint64],
    base: wp.uint64,
    h: wp.int32,
) -> tuple[wp.bool, wp.int32]:
    # Halfedge ``h`` against a directed contour. First: does it run *along* the contour in its own
    # direction? Its face is then on the left, since a face's corners run counter-clockwise -- so
    # the contour's orientation is the only thing deciding which side is "left". Second: the twin
    # whose face the dual graph joins to ``h``'s, or ``-1`` -- for a boundary halfedge, for the far
    # half of an edge (the lower half answers for it, so each dual edge is taken once), and for an
    # edge on the contour in *either* direction, which is blocked.
    #
    # Everything comes from ``twins`` rather than ``face_adjacency``: one hash-and-group pass
    # instead of that plus a second key sort over every halfedge.
    tail, tip = halfedge_endpoints(faces, h)
    along_contour = binary_search_sorted_contains(
        contour_keys_sorted, pack_directed_key(tail, tip, base)
    )
    twin = twins[h]
    open_twin = wp.int32(-1)
    if twin > h and not along_contour:
        backward = pack_directed_key(tip, tail, base)
        if not binary_search_sorted_contains(contour_keys_sorted, backward):
            open_twin = twin
    return along_contour, open_twin


@wp.kernel
def seed_and_prehook_dual(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    contour_keys_sorted: wp.array[wp.uint64],
    base: wp.uint64,
    parents: wp.array[wp.int32],
    out_seeds: wp.array[wp.bool],
) -> None:
    # The flood fill's first pass over halfedges: seed the faces left of the contour and pre-hook
    # the open dual edges into an identity ``parents``. The union-find is per edge and order-free,
    # so the dual edges are formed in the thread and never listed -- no cursor, no buffer, no
    # readback to trim one. The pre-hook is ``ecl_init_parent_edges``' and keeps the hook cheap on
    # long sequential cycles and hub faces.
    h = wp.int32(wp.tid())
    along_contour, twin = open_dual_twin(faces, twins, contour_keys_sorted, base, h)
    if along_contour:
        out_seeds[h // 3] = True
    if twin >= 0:
        ecl_prehook_pair(parents, h // 3, twin // 3)


@wp.kernel
def hook_dual(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    contour_keys_sorted: wp.array[wp.uint64],
    base: wp.uint64,
    parents: wp.array[wp.int32],
) -> None:
    # The hook over the same open dual edges, after ``seed_and_prehook_dual`` has finished.
    h = wp.int32(wp.tid())
    _along_contour, twin = open_dual_twin(faces, twins, contour_keys_sorted, base, h)
    if twin >= 0:
        ecl_hook_pair(parents, h // 3, twin // 3)


@wp.kernel
def deleted_face_edge_keys(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    kept_ranks: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
) -> None:
    # The undirected edge keys of the *masked* faces only, three per face at the face's compact
    # rank among the masked faces, so the deleted region's edge table costs three keys per deleted
    # face rather than per mesh face. ``kept_ranks`` is the inclusive scan of the *unmasked* faces
    # (the submesh extraction's face half), so a masked face has ``f - kept_ranks[f]`` masked
    # faces before it.
    f = wp.int32(wp.tid())
    if not face_mask[f]:
        return
    write_face_edge_keys(faces, f, 3 * (f - kept_ranks[f]), base, out_keys)


@wp.kernel
def loops_are_input_rims(
    flat_loops: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    to_input: wp.array[wp.int32],
    deleted_edge_keys: wp.array[wp.uint64],
    base: wp.uint64,
    out_kept_counts: wp.array[wp.vec3i],
) -> None:
    # Per loop: was *every* one of its edges already a boundary edge of the input mesh? A loop for
    # which that holds is the input's own rim surfacing in the submesh rather than a rim the
    # deletion opened, which is what ``delete_region_keep_boundary`` filters on.
    #
    # Answered from the *deletion*, not from the input's boundary, so nothing here is mesh-sized:
    # a loop edge the submesh holds in exactly one kept face had ``1 + (deleted faces containing
    # it)`` faces in the input, so it was an input boundary edge exactly when no deleted face
    # contains it -- one search in ``deleted_edge_keys``, the sorted edge table of the deleted
    # faces in input indices (``to_input`` maps a submesh vertex back).
    #
    # That premise -- every consecutive loop pair is a boundary edge of the submesh -- is what
    # ``boundary_loops_with_offsets`` guarantees on *any* input, which is what makes this safe
    # without a check. Its vertex and seam walks run only on a 2-regular rim. Its pinch walk follows
    # ``next_boundary_halfedge``, whose every successor starts where its predecessor ends (twins
    # are true opposites or ``-1``, even unvalidated), and no rotation can enter another boundary
    # halfedge's face (a face is entered through the twin of its incoming halfedge, which a
    # boundary halfedge lacks), so two rotations never merge and the ranking never sees two tails
    # sharing a successor. What a mesh that is not edge-manifold
    # *can* do is make a loop go missing -- a rotation stopping at a three-faced edge dead-ends
    # -- and a missing loop defeats any classifier equally. ``tests/test_boundary.py`` pins the
    # guarantee on pinched and non-manifold rims. A kept-face count per loop edge would make it
    # checked here too, and measured as costly as the mesh-sized table this replaces.
    #
    # One thread per *loop*, walking its own rim, rather than one per rim vertex with a segment
    # label: boundary loops are few and short, which is the regime
    # ``kernels/array.segment_owner_labels`` names as the one where the per-segment shape wins --
    # and it needs no owner array. The
    # early exit is what makes it cheap in the common case, since a loop the deletion opened
    # usually fails on one of its first edges.
    ell = wp.int32(wp.tid())
    start = loop_offsets[ell]
    size = loop_offsets[ell + 1] - start
    is_rim = wp.int32(1)
    for k in range(size):
        a = to_input[flat_loops[start + k]]
        b = to_input[flat_loops[start + (k + 1) % size]]
        if binary_search_sorted_contains(deleted_edge_keys, pack_edge_key(a, b, base)):
            is_rim = wp.int32(0)
            break
    # ``(1, size, start)`` for a loop the deletion opened and ``(0, 0, start)`` for the input's
    # own rim. Read as it is, that is every kept loop's extent in ``flat_loops``, which is all the
    # list form needs to cut its views; scanned in place, the first two components give every kept
    # loop its rank and the end of its run in the packed output (``compact_kept_loops``), and the
    # tail sizes it. The scanned third component is never read.
    kept = 1 - is_rim
    out_kept_counts[ell] = wp.vec3i(kept, kept * size, start)


@wp.kernel
def compact_kept_loops(
    flat_loops: wp.array[wp.int32],
    loop_offsets: wp.array[wp.int32],
    kept_scan: wp.array[wp.vec3i],
    out_loops: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
) -> None:
    # One thread per input loop, copying a kept loop to its scanned place and writing the end of
    # its run as the next offset: the terminated offsets of the kept loops, with ``out_offsets[0]``
    # the zero the caller allocated. Loops are few and short (see ``loops_are_input_rims``).
    ell = wp.int32(wp.tid())
    end = kept_scan[ell]
    before = wp.vec3i(0, 0, 0)
    if ell > 0:
        before = kept_scan[ell - 1]
    if end[0] == before[0]:
        return
    start = loop_offsets[ell]
    for k in range(end[1] - before[1]):
        out_loops[before[1] + k] = flat_loops[start + k]
    out_offsets[end[0]] = end[1]
