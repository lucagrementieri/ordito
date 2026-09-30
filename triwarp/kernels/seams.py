from typing import Any

import warp as wp

from triwarp.kernels.adjacency import write_edge_row
from triwarp.kernels.algorithms.connected_components import (
    ecl_hook_pair,
    ecl_prehook_pair,
    find_representative,
)
from triwarp.kernels.array import (
    OverloadTable,
    binary_search_sorted_contains,
    cross2,
    mark_at,
    pack_edge_key,
    scanned_count,
)
from triwarp.kernels.grouping import sorted_run_of_length
from triwarp.kernels.halfedge import halfedge_next, halfedge_prev
from triwarp.kernels.predicates import vector_angle
from triwarp.kernels.triangles import face_normals_and_area


@wp.kernel
def crease_flags(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    threshold: wp.float32,
    out_flags: wp.array2d[wp.int32],
) -> None:
    # One thread per position of the radix-sorted halfedge keys, whose payload is each halfedge's
    # index (``adjacency.face_edge_keys_and_order``). Row 0 of ``out_flags``: 1 where a run of
    # *exactly two* keys starts -- ``adjacency.face_adjacency``'s row, in the same ascending key
    # order -- and its two faces meet at more than ``threshold``. Row 1, when the table has one:
    # 1 where a run of exactly one key starts, a boundary edge. The caller scans the flattened
    # table in place, so the boundary rows number on from the creases.
    #
    # The angle is ``adjacency.face_adjacency_angles``' exactly: the same unit normals (from
    # ``face_normals_and_area``, which ``triangles.face_normals_and_areas`` writes them with, formed
    # here inline rather than through a per-face buffer), taken smaller face first as that
    # function's rows are, and the same ``vector_angle``, so ``angle > threshold`` is the predicate
    # thresholding its output would be. Strictly greater, so a zero threshold selects every
    # non-coplanar interior edge.
    i = wp.int32(wp.tid())
    crease = wp.int32(0)
    if sorted_run_of_length(sorted_keys, n, i, 2):
        face_a = order[i] // 3
        face_b = order[i + 1] // 3
        normal_a, _area_a = face_normals_and_area(vertices, faces, wp.min(face_a, face_b))
        normal_b, _area_b = face_normals_and_area(vertices, faces, wp.max(face_a, face_b))
        crease = wp.where(vector_angle(normal_a, normal_b) > threshold, wp.int32(1), wp.int32(0))
    out_flags[0, i] = crease
    if out_flags.shape[0] > 1:
        out_flags[1, i] = wp.where(
            sorted_run_of_length(sorted_keys, n, i, 1), wp.int32(1), wp.int32(0)
        )


@wp.kernel
def emit_crease_edges(
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    n: wp.int32,
    faces: wp.array[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One thread per entry of ``crease_flags``' table, flattened and scanned in place: where the
    # scan steps, write the flagged run's first halfedge as its sorted endpoint pair -- the edge
    # row ``face_adjacency(return_edges=True)`` and ``boundary_edges`` both write for it.
    p = wp.int32(wp.tid())
    row, flag = scanned_count(inclusive, p)
    if flag != 0:
        write_edge_row(faces, order[p % n], row, out_edges)


@wp.func
def corner_union(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    marked_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    h: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # One edge of the corner graph per halfedge: the two face corners that meet at one endpoint of
    # an *uncut* interior mesh edge, or ``(h, h)`` -- a self-loop, which unions nothing -- for a
    # boundary or marked edge. Connected components of that graph are exactly the copies each
    # vertex needs: a vertex whose whole fan is uncut stays one vertex, and every marked edge
    # crossing the fan splits it.
    #
    # Each undirected edge joins corners at *both* of its endpoints, and the edge's two halfedges
    # emit one join each. Getting only one endpoint right is silent: the fan around a vertex is
    # then connected by half its edges and every vertex splits into two.
    #
    # ``twin`` runs opposite ``h`` in every table ``halfedge_twins`` builds, which leaves a
    # same-direction pair at ``-1``; but a caller's own ``twins`` may pair the two halfedges of an
    # edge by their undirected endpoints alone, so ``twin`` can also run the same way as ``h`` (an
    # edge-manifold but inconsistently-wound mesh). Both cases have to be handled, or the two
    # corners joined belong to two different vertices.
    #
    # For ``h: u -> v`` with ``twin: v -> u`` (opposite), ``h``'s own corner sits at ``u`` with
    # ``next(twin)``, and ``twin`` answers the ``v`` end the same way. With ``twin: u -> v`` (same
    # direction), the corners at ``u`` are ``h`` and ``twin`` and those at ``v`` are ``next(h)``
    # and ``next(twin)``; the lower-indexed half takes ``u`` and the higher ``v``.
    twin = twins[h]
    if twin < 0:
        return h, h
    if binary_search_sorted_contains(
        marked_keys, pack_edge_key(faces[h], faces[halfedge_next(h)], key_base)
    ):
        return h, h
    if faces[twin] != faces[h]:
        return h, halfedge_next(twin)
    if h < twin:
        return h, twin
    return halfedge_next(h), halfedge_next(twin)


@wp.kernel
def corner_union_prehook(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    marked_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_init_parent_edges`` over the corner graph, with each edge formed
    # from ``corner_union`` in the thread rather than read from a materialised edge list. Runs over
    # an identity ``parents``; changes no root and keeps the hook below cheap (see there).
    h = wp.int32(wp.tid())
    a, b = corner_union(faces, twins, marked_keys, key_base, h)
    ecl_prehook_pair(parents, a, b)


@wp.kernel
def corner_union_hook(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    marked_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    parents: wp.array[wp.int32],
) -> None:
    # ``connected_components.ecl_hook_edges`` over the same corner graph. Every union points the
    # larger root at the smaller, so each component's root is its smallest corner whichever order
    # the unions ran in.
    h = wp.int32(wp.tid())
    a, b = corner_union(faces, twins, marked_keys, key_base, h)
    ecl_hook_pair(parents, a, b)


@wp.kernel
def corner_roots(
    parents: wp.array[wp.int32], out_roots: wp.array[wp.int32], out_is_root: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_flatten`` plus a 0/1 flag on each component's root corner -- its
    # smallest -- so an inclusive scan of the flags numbers the components in ascending root order.
    h = wp.int32(wp.tid())
    root = find_representative(parents, h)
    out_roots[h] = root
    out_is_root[h] = wp.where(root == h, wp.int32(1), wp.int32(0))


@wp.kernel
def scatter_corner_values(
    faces: wp.array[wp.int32],
    roots: wp.array[wp.int32],
    root_ranks: wp.array[wp.int32],
    values: wp.array[Any],
    out_corner_index: wp.array[wp.int32],
    out_values: wp.array[Any],
) -> None:
    # Rank each corner's component and gather its position (or any per-vertex attribute) into it.
    # ``root_ranks`` is ``corner_roots``' flags scanned in place, so a root's entry less one is its
    # component's rank among the roots, in ascending root order. Every corner in a component
    # writes the *same* value, so the race on ``out_values`` is benign by construction and needs no
    # atomics.
    h = wp.int32(wp.tid())
    rank = root_ranks[roots[h]] - 1
    out_corner_index[h] = rank
    out_values[rank] = values[faces[h]]


@wp.func
def texcoords_differ(
    texcoords: wp.array[wp.vec2], first: wp.int32, second: wp.int32, tolerance_sq: wp.float32
) -> wp.bool:
    # Coordinate comparison, MeshLab's predicate. ``tolerance_sq == 0`` makes this exact inequality,
    # which is what VCG's ``TexCoord2f::operator==`` does.
    return wp.length_sq(texcoords[first] - texcoords[second]) > tolerance_sq


# The class ``classify_uv_halfedge`` assigns a halfedge. Only the canonical (lower-index) half of an
# interior edge carries a seam or foldover verdict, so every undirected edge is decided once.
UV_EDGE_NONE = wp.constant(wp.int32(0))
UV_EDGE_SEAM = wp.constant(wp.int32(1))
UV_EDGE_BOUNDARY = wp.constant(wp.int32(2))
UV_EDGE_FOLDOVER = wp.constant(wp.int32(3))


@wp.func
def corner_texcoord(
    face_texcoords: wp.array[wp.int32], has_face_texcoords: wp.bool, h: wp.int32
) -> wp.int32:
    # The texcoord index of corner ``h``. Without a corner-to-texcoord table the texcoords are
    # per-corner, so the index *is* the corner and the table is never read -- which is what lets
    # the caller pass ``None`` for it instead of materialising ``arange(3 * n_faces)``.
    if has_face_texcoords:
        return face_texcoords[h]
    return h


@wp.func
def canonical_edge_halfedges(
    faces: wp.array[wp.int32], h: wp.int32, twin: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # igl's canonical direction: the halfedge running from the smaller vertex index to the larger.
    # A degenerate edge (both endpoints equal) has no such direction; keeping ``h`` forwards is
    # arbitrary but deterministic.
    if faces[h] > faces[halfedge_next(h)]:
        return twin, h
    return h, twin


@wp.func
def classify_uv_halfedge(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    has_face_texcoords: wp.bool,
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    h: wp.int32,
) -> wp.int32:
    # The one seam / boundary / foldover decision, shared by the per-edge compaction and the
    # per-vertex mask so the two cannot disagree about which edges are seams. The pair
    # ``(h, twin)`` is classified by whichever of the two has the smaller index; the other half
    # answers ``UV_EDGE_NONE``.
    twin = twins[h]
    if twin < 0:
        return UV_EDGE_BOUNDARY
    if twin < h:
        return UV_EDGE_NONE
    forwards, backwards = canonical_edge_halfedges(faces, h, twin)

    # ``halfedge_twins`` pairs only opposite halfedges, but a caller's own ``twins`` may pair the
    # two halfedges of an edge by their undirected endpoints alone, so ``backwards`` can also run
    # the same way as ``forwards`` on an inconsistently-wound mesh. When it runs the *other* way,
    # the corner sitting on top of ``forwards``' tail is the one *following* ``backwards``, and vice
    # versa; when it runs the *same* way, that corner is ``backwards`` itself, and the one on top of
    # ``forwards``' head is the one following it. ``faces[backwards] == faces[forwards]`` is exactly
    # the same-direction case, since both then share the same origin vertex.
    tail_forwards = corner_texcoord(face_texcoords, has_face_texcoords, forwards)
    head_forwards = corner_texcoord(face_texcoords, has_face_texcoords, halfedge_next(forwards))
    if faces[backwards] == faces[forwards]:
        tail_backwards = corner_texcoord(face_texcoords, has_face_texcoords, backwards)
        head_backwards = corner_texcoord(
            face_texcoords, has_face_texcoords, halfedge_next(backwards)
        )
    else:
        tail_backwards = corner_texcoord(
            face_texcoords, has_face_texcoords, halfedge_next(backwards)
        )
        head_backwards = corner_texcoord(face_texcoords, has_face_texcoords, backwards)

    if match_uv:
        is_seam = texcoords_differ(
            texcoords, tail_forwards, tail_backwards, tolerance_sq
        ) or texcoords_differ(texcoords, head_forwards, head_backwards, tolerance_sq)
    else:
        is_seam = tail_forwards != tail_backwards or head_forwards != head_backwards
    if is_seam:
        return UV_EDGE_SEAM

    # Matched texcoords, so both triangles agree on where the shared edge lands in UV space. They
    # fold over each other exactly when their two opposite corners land on the *same* side of it.
    # Strictly the same side: a collinear corner is a degenerate UV triangle, not a foldover.
    a = texcoords[tail_forwards]
    b = texcoords[head_forwards]
    c_forwards = texcoords[
        corner_texcoord(face_texcoords, has_face_texcoords, halfedge_prev(forwards))
    ]
    c_backwards = texcoords[
        corner_texcoord(face_texcoords, has_face_texcoords, halfedge_prev(backwards))
    ]
    orientation_forwards = cross2(a - c_forwards, b - c_forwards)
    orientation_backwards = cross2(a - c_backwards, b - c_backwards)
    if (orientation_forwards > 0.0 and orientation_backwards > 0.0) or (
        orientation_forwards < 0.0 and orientation_backwards < 0.0
    ):
        return UV_EDGE_FOLDOVER
    return UV_EDGE_NONE


@wp.kernel
def classify_uv_halfedges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    has_face_texcoords: wp.bool,
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    out_flags: wp.array2d[wp.int32],
) -> None:
    # One thread per halfedge. Row ``k`` of ``out_flags`` is the 0/1 selection of class ``k + 1``
    # (seam, boundary, foldover), laid out so one inclusive scan of the flattened buffer, in place,
    # numbers all three blocks at once and leaves each block's running total in its last column.
    # Every flag is written on every path, so the caller may allocate ``out_flags`` with
    # ``wp.empty``.
    h = wp.int32(wp.tid())
    kind = classify_uv_halfedge(
        faces, twins, face_texcoords, has_face_texcoords, texcoords, match_uv, tolerance_sq, h
    )
    for k in range(3):
        out_flags[k, h] = wp.where(kind == k + 1, wp.int32(1), wp.int32(0))


@wp.func
def scanned_block_row(inclusive: wp.array[wp.int32], n: wp.int32, k: wp.int32, h: wp.int32):
    # Row of entry ``h`` within block ``k`` of a ``(3, n)`` flag table scanned in place as one flat
    # buffer, and that entry's flag. Block ``k``'s positions run on from the totals of the blocks
    # before it, which sit in the flat entry just ahead of the block; subtracting it makes each
    # block zero-based.
    start, flag = scanned_count(inclusive, k * n + h)
    base = wp.int32(0)
    if k > 0:
        base = inclusive[k * n - 1]
    return start - base, flag


@wp.func
def write_canonical_pair_row(
    forwards: wp.int32, backwards: wp.int32, row: wp.int32, out_rows: wp.array2d[wp.int32]
) -> None:
    # A seam or foldover row: the ``(face, corner)`` pair of both canonical halfedges, under the
    # ``h = 3 * f + k`` convention.
    out_rows[row, 0] = forwards // 3
    out_rows[row, 1] = forwards % 3
    out_rows[row, 2] = backwards // 3
    out_rows[row, 3] = backwards % 3


@wp.kernel
def scatter_uv_halfedges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    inclusive: wp.array[wp.int32],
    out_seams: wp.array2d[wp.int32],
    out_boundaries: wp.array2d[wp.int32],
    out_foldovers: wp.array2d[wp.int32],
) -> None:
    # Compact the three classes into their rows in ascending halfedge order. ``inclusive`` is
    # ``classify_uv_halfedges``' flags scanned in place as one flat buffer, so each flag is
    # recovered as a step of the scan. A seam or foldover row is the ``(face, corner)`` pair of
    # both canonical halfedges; a boundary row is this halfedge's own ``(face, corner)``, under the
    # ``h = 3 * f + k`` convention. A halfedge carries at most one class.
    h = wp.int32(wp.tid())
    n = inclusive.shape[0] // 3
    row, is_boundary = scanned_block_row(inclusive, n, 1, h)
    if is_boundary != 0:
        out_boundaries[row, 0] = h // 3
        out_boundaries[row, 1] = h % 3
        return
    seam_row, is_seam = scanned_block_row(inclusive, n, 0, h)
    foldover_row, is_foldover = scanned_block_row(inclusive, n, 2, h)
    if is_seam == 0 and is_foldover == 0:
        return
    forwards, backwards = canonical_edge_halfedges(faces, h, twins[h])
    if is_seam != 0:
        write_canonical_pair_row(forwards, backwards, seam_row, out_seams)
    else:
        write_canonical_pair_row(forwards, backwards, foldover_row, out_foldovers)


@wp.kernel
def mark_uv_seam_vertices(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    has_face_texcoords: wp.bool,
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    include_boundary: wp.bool,
    out_mask: wp.array[wp.bool],
) -> None:
    # The per-vertex seam mask straight from the classification: both endpoints of every seam
    # halfedge (and of every boundary halfedge, when asked). A seam row's forward halfedge and
    # ``h`` span the same vertex pair, so marking ``h``'s own endpoints is the same answer as
    # decoding the row. Out-of-range endpoints are skipped, as ``array.indices_to_mask`` does;
    # every write stores ``True``, so the race between two halfedges on one vertex is benign.
    h = wp.int32(wp.tid())
    kind = classify_uv_halfedge(
        faces, twins, face_texcoords, has_face_texcoords, texcoords, match_uv, tolerance_sq, h
    )
    if kind == UV_EDGE_SEAM or (include_boundary and kind == UV_EDGE_BOUNDARY):
        mark_at(out_mask, faces[h])
        mark_at(out_mask, faces[halfedge_next(h)])


@wp.kernel
def face_corner_edge_vertices(
    faces: wp.array[wp.int32], face_corners: wp.array2d[wp.int32], out_edges: wp.array2d[wp.int32]
) -> None:
    # ``(face, corner)`` provenance back to the vertex pair it names. Columns 0 and 1 are the
    # forward side of a seam/foldover row and the whole of a boundary row, so one kernel serves all
    # three blocks.
    i = wp.int32(wp.tid())
    h = 3 * face_corners[i, 0] + face_corners[i, 1]
    out_edges[i, 0] = faces[h]
    out_edges[i, 1] = faces[halfedge_next(h)]


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5. Two overloads across **2** module loads: the smallest fork in the package.
#
# ``triwarp.seams.cut_along_edges`` scatters the corner *positions* it is splitting, so the value
# dtype is the vertex dtype it was handed -- ``wp.vec3`` or ``wp.vec3d`` -- and nothing else reaches
# this kernel. Both are registered because the wrapper's own dispatch (``SCATTER_CORNER_VALUES[
# vertices.dtype]``) can reach either one.
# The concrete handle keyed by the value dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable].
SCATTER_CORNER_VALUES: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global SCATTER_CORNER_VALUES
    SCATTER_CORNER_VALUES = OverloadTable(
        scatter_corner_values,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.int32],
                wp.array[d],
            ]
            for d in (wp.vec3, wp.vec3d)
        },
    )


_register_overloads()
