from typing import Any

import warp as wp

from ordito.constants import TILE_1D
from ordito.kernels.array import (
    OverloadTable,
    append_slot_if,
    csr_key,
    csr_row_long,
    csr_run_start,
    declare_block_dims,
    device_is_cpu,
    popcount32,
    row_chunk_slots,
    sorted_run_end,
    store_coo_key,
)
from ordito.kernels.halfedge import halfedge_destination
from ordito.kernels.predicates import doublearea_from_lengths, triangle_double_area
from ordito.kernels.reduce import (
    GroupSumPass,
    block_chunk,
    block_chunk_1d,
    block_min,
    block_sum,
    read_group_sum_vector,
    warp_active_lanes,
    warp_ballot,
    warp_lane_rank,
    warp_lanes_sum_float,
    warp_long_rows,
    warp_lowest_row,
    warp_walk,
    warp_walk_total,
)
from ordito.kernels.scatter import (
    ARRIVAL_POOL_UNIFORM,
    arrival_order_add_vector,
    group_sum_vector_term,
    peer_summed_add_corner_triple,
)
from ordito.kernels.triangles import (
    face_normals_and_area,
    face_vertices,
    row_triple,
    store_face_edge_lengths,
)

wp.set_module_options({"enable_backward": False})

# ``cot_entries_from_l2`` and ``cot_entries_from_edge_lengths`` stay here rather than joining
# ``squared_edge_lengths`` / ``doublearea_from_lengths`` in ``kernels/predicates.py``: a cotangent
# weight is not a general triangle quantity, it is this operator's own entry, and the zero-area
# reasoning below is numerical defence *of the Laplacian* that belongs beside the thing it defends.


@wp.func
def cot_entries_from_l2(
    l2_0: wp.Float, l2_1: wp.Float, l2_2: wp.Float, dbl_area: wp.Float
) -> tuple[wp.Float, wp.Float, wp.Float]:
    # A zero-area triangle contributes nothing rather than an infinity. Its angles are 0 or pi, so
    # it has no finite cotangent, and ``doublearea_from_lengths`` deliberately reports 0.0 for one:
    # without this guard that 0 divides straight through to +-inf, and a *single* collapsed face
    # poisons the whole assembled operator -- and every solve against it -- with NaN.
    #
    # The test is against exact zero, not a tolerance. ``dbl_area`` is already clamped
    # non-negative, so this changes results only where they used to be non-finite; a merely
    # sliver triangle still yields its (huge, finite) weight, because that is ill-conditioning
    # rather than a division by zero and the fix for it is mollification -- see
    # ``laplacian.robust_laplacian``.
    denominator = type(dbl_area)(4.0) * dbl_area
    zero = type(dbl_area)(0.0)
    if denominator <= zero:
        return zero, zero, zero
    c0 = (l2_1 + l2_2 - l2_0) / denominator
    c1 = (l2_2 + l2_0 - l2_1) / denominator
    c2 = (l2_0 + l2_1 - l2_2) / denominator
    return c0, c1, c2


@wp.func
def cot_entries_from_edge_lengths(
    l0: wp.Float, l1: wp.Float, l2: wp.Float
) -> tuple[wp.Float, wp.Float, wp.Float]:
    l2_0 = l0 * l0
    l2_1 = l1 * l1
    l2_2 = l2 * l2
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    return cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)


@wp.func
def half_cotangents_of(v0: Any, v1: Any, v2: Any) -> tuple[wp.Float, wp.Float, wp.Float]:
    # The three half-cotangents of the triangle ``(v0, v1, v2)`` at its corners' precision, each
    # ``dot / |cross|`` of the corner's two edges, the cross product taken at the largest angle
    # (``triangle_area_vector``). The zero-area guard is the one ``cot_entries_from_l2`` explains.
    dbl_area = triangle_double_area(v0, v1, v2)
    denominator = type(dbl_area)(2.0) * dbl_area
    zero = type(dbl_area)(0.0)
    if denominator <= zero:
        return zero, zero, zero
    c0 = wp.dot(v1 - v0, v2 - v0) / denominator
    c1 = wp.dot(v2 - v1, v0 - v1) / denominator
    c2 = wp.dot(v0 - v2, v1 - v2) / denominator
    return c0, c1, c2


@wp.func
def face_half_cotangents(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    # Face ``f``'s three half-cotangent weights, column ``e`` for the edge opposite corner ``e``:
    # one row of ``cotmatrix_entries``' table, for a kernel that needs a few faces' weights and not
    # the whole mesh's (``smoothing.band_dirichlet_values``).
    #
    # From positions (``half_cotangents_of``), not by the law of cosines the intrinsic overload has
    # to use (``cot_entries_from_l2``), which cancels the squared lengths against each other and
    # loses ``eps`` times the aspect ratio: on ``saddle_graded``'s aspect-4 900 needles that was
    # 1.4e-4 absolute on weights whose true value is 1e-4, against ``igl.cotmatrix_entries``'
    # ``float64``. The cross product at the largest angle still cancels when that angle nears pi:
    # a weight's relative error is about ``eps / sin(largest angle)``, 1e-5 on
    # ``torus_irregular_holes``' caps. ``face_half_cotangents_float64`` is the same from the
    # exactly widened corners, for the heat method's systems.
    p0, p1, p2 = face_vertices(vertices, faces, f)
    return half_cotangents_of(p0, p1, p2)


@wp.func
def face_half_cotangents_float64(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32
) -> tuple[wp.float64, wp.float64, wp.float64]:
    # ``face_half_cotangents`` formed in ``float64`` from the ``float32`` corners, which widen
    # exactly: only the final rounding is left, whatever the face's shape.
    p0, p1, p2 = face_vertices(vertices, faces, f)
    return half_cotangents_of(wp.vec3d(p0), wp.vec3d(p1), wp.vec3d(p2))


@wp.func
def store_half_cotangents(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32, out_cot: Any
) -> None:
    # Face ``f``'s row of the half-cotangent table, computed in float32 (the vertex precision) and
    # cast to the table's dtype (float32 or float64) at store time: ``cotmatrix_entries``' body,
    # shared by every face pass that writes the table beside something else.
    c0, c1, c2 = face_half_cotangents(vertices, faces, f)
    out_cot[f, 0] = type(out_cot[f, 0])(c0)
    out_cot[f, 1] = type(out_cot[f, 1])(c1)
    out_cot[f, 2] = type(out_cot[f, 2])(c2)


@wp.func
def add_face_area_exact(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: wp.int32,
    out_area_sums: wp.array[wp.float64],
) -> None:
    # Face ``f``'s ``float32`` area (``face_normals_and_area``'s) added exactly into each corner's
    # ``float64`` sum: the barycentric mass before its division by three, as
    # ``scatter.scatter_face_areas_exact`` forms it from the positions.
    _normal, area = face_normals_and_area(vertices, faces, f)
    value = wp.float64(area)
    peer_summed_add_corner_triple(out_area_sums, faces, f, value, value, value)


@wp.kernel
def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cot: wp.array2d[wp.Float]
) -> None:
    # ``out_cot`` is generic: see ``store_half_cotangents``.
    f = wp.int32(wp.tid())
    store_half_cotangents(vertices, faces, f, out_cot)


@wp.kernel
def cotmatrix_entries_and_area_sums(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    write_cot: wp.int32,
    out_cot: wp.array2d[wp.float32],
    out_area_sums: wp.array[wp.float64],
) -> None:
    # The two per-face quantities a cotangent stiffness and its lumped mass are built from, in one
    # pass over the faces: the ``float32`` half-cotangent table (``cotmatrix_entries``' row;
    # skipped when ``write_cot`` is 0) and the area's exact ``float64`` scatter into each corner
    # (``scatter.scatter_face_areas_exact``, the barycentric mass before its division by three).
    # Each value is the standalone kernel's bit for bit, and a ``float64`` matrix casts each
    # ``float32`` half-cotangent as it sums it, so the narrower table changes no entry.
    f = wp.int32(wp.tid())
    if write_cot != 0:
        store_half_cotangents(vertices, faces, f, out_cot)
    add_face_area_exact(vertices, faces, f, out_area_sums)


@wp.func
def store_intrinsic_cotangents(
    edge_lengths: wp.array2d[wp.float32], f: wp.int32, out_cot: Any
) -> None:
    # Face ``f``'s row of the intrinsic half-cotangent table, evaluated at the output's precision:
    # the lengths are ``float32``, but the law of cosines cancels their squares against each
    # other, so a ``float64`` table formed in ``float32`` and cast (as this did until 2026-10-08)
    # carried ``float32``'s cancellation error -- 6.5x the ``1e-5`` tolerance off igl's
    # ``float64`` answer on the same lengths, on ``saddle_graded``. ``cotmatrix_entries_intrinsic``'
    # body, and the half the intrinsic key pass adds to its keys.
    l0, l1, l2 = row_triple(edge_lengths, f)
    c0, c1, c2 = cot_entries_from_edge_lengths(
        type(out_cot[f, 0])(l0), type(out_cot[f, 0])(l1), type(out_cot[f, 0])(l2)
    )
    out_cot[f, 0] = c0
    out_cot[f, 1] = c1
    out_cot[f, 2] = c2


@wp.kernel
def cotmatrix_entries_intrinsic(
    edge_lengths: wp.array2d[wp.float32], out_cot: wp.array2d[wp.Float]
) -> None:
    # ``out_cot`` is generic: see ``store_intrinsic_cotangents``.
    f = wp.int32(wp.tid())
    store_intrinsic_cotangents(edge_lengths, f, out_cot)


# Relative perturbation of one ``float32`` length by its rounding: half an ulp.
LENGTH_ROUNDING = wp.float64(5.9604645e-08)


@wp.func
def half_cotangent_rounding(
    edge_lengths: wp.array2d[wp.float32], f: wp.int32
) -> tuple[wp.vec3d, wp.vec3d]:
    # How far the rounding of face ``f``'s ``float32`` edge lengths can move its three intrinsic
    # half-cotangents, beside their magnitudes: each one's change under a half-ulp relative change
    # of each of the three lengths (in ``float64``, so the finite difference measures the lengths'
    # conditioning and not this arithmetic). A needle's area is ``eps * aspect^2`` sensitive to its
    # sides, so a needle's half-cotangents are where the ratio grows.
    l0, l1, l2 = row_triple(edge_lengths, f)
    d0 = wp.float64(l0)
    d1 = wp.float64(l1)
    d2 = wp.float64(l2)
    c0, c1, c2 = cot_entries_from_edge_lengths(d0, d1, d2)
    step = wp.float64(1.0) + LENGTH_ROUNDING
    a0, a1, a2 = cot_entries_from_edge_lengths(d0 * step, d1, d2)
    b0, b1, b2 = cot_entries_from_edge_lengths(d0, d1 * step, d2)
    e0, e1, e2 = cot_entries_from_edge_lengths(d0, d1, d2 * step)
    bound = wp.vec3d(
        wp.abs(a0 - c0) + wp.abs(b0 - c0) + wp.abs(e0 - c0),
        wp.abs(a1 - c1) + wp.abs(b1 - c1) + wp.abs(e1 - c1),
        wp.abs(a2 - c2) + wp.abs(b2 - c2) + wp.abs(e2 - c2),
    )
    return bound, wp.vec3d(wp.abs(c0), wp.abs(c1), wp.abs(c2))


@wp.func
def half_cotangent_row(
    faces: wp.array[wp.int32], f: wp.int32, k: wp.int32, end: wp.int32
) -> wp.int32:
    # The row half-cotangent ``k`` of face ``f`` adds to, for ``end`` 1 and 2: it weights the edge
    # opposite corner ``k``, whose endpoints are the other two corners.
    return faces[f * 3 + (k + end) % 3]


@wp.func
def add_rounding_terms(
    faces: wp.array[wp.int32], f: wp.int32, bound: wp.vec3d, weight: wp.vec3d, sums: GroupSumPass
):
    # Face ``f``'s six ``(bound, weight)`` terms into their rows' exact group sum (the setting on).
    for k in range(3):
        for end in range(1, 3):
            v = half_cotangent_row(faces, f, k, end)
            group_sum_vector_term(sums, v, wp.vec2d(bound[k], weight[k]))


@wp.func(inline=False)
def add_rounding_terms_out_of_line(
    faces: wp.array[wp.int32], f: wp.int32, bound: wp.vec3d, weight: wp.vec3d, sums: GroupSumPass
):
    # ``add_rounding_terms`` as one call a face, for the CPU device (see the kernel).
    add_rounding_terms(faces, f, bound, weight, sums)


@wp.func
def half_cotangent_sensitivities(
    l0: wp.float32, l1: wp.float32, l2: wp.float32
) -> tuple[wp.vec3, wp.vec3]:
    # A face's ``float32`` half-cotangents ``c_k`` and, beside them, a first-order bound ``S_k``
    # on ``sum_m |l_m dc_k/dl_m|``: with ``c_k = (l_i^2 + l_j^2 - l_k^2) / 4A`` and
    # ``l_m dA/dl_m = l_m^2 cot(theta_m) / 2 = l_m^2 c_m``, each term is at most
    # ``l_m^2 / 2A + |c_k| l_m^2 |c_m| / A``. ``half_cotangent_rounding``'s finite difference
    # under a relative step ``h`` is ``h S_k`` to first order. A zero area returns infinite bounds.
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    c0, c1, c2 = cot_entries_from_edge_lengths(l0, l1, l2)
    q0 = l0 * l0
    q1 = l1 * l1
    q2 = l2 * l2
    area = wp.float32(0.5) * dbl_area
    lead = (q0 + q1 + q2) / (wp.float32(2.0) * area)
    tail = (q0 * wp.abs(c0) + q1 * wp.abs(c1) + q2 * wp.abs(c2)) / area
    c = wp.vec3(c0, c1, c2)
    bound = wp.vec3(lead + wp.abs(c0) * tail, lead + wp.abs(c1) * tail, lead + wp.abs(c2) * tail)
    return c, bound


# How far under the warning limit a ``float32`` sensitivity must put a row term for
# ``flag_rounding_rows`` to leave it unflagged: the margin covers the finite difference's second
# order and the ``float32`` evaluation, both far below it wherever a term passes.
ROUNDING_FILTER_MARGIN = wp.float32(4.0)


@wp.kernel
def flag_rounding_rows(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    limit: wp.float32,
    out_flags: wp.array[wp.int32],
    out_listed: wp.array[wp.int32],
) -> None:
    # The rows ``intrinsic_rounding_rows`` must sum exactly. Face ``f`` adds to the row of its
    # corner ``j`` the bounds and weights of its half-cotangents ``j + 1`` and ``j + 2``, and a
    # row's ratio is a mediant of those pairs: it exceeds no pair's ``(b_{j+1} + b_{j+2}) /
    # (|c_{j+1}| + |c_{j+2}|)``. A pair whose ``float32`` sensitivity (``half_cotangent_
    # sensitivities``) puts it a ``ROUNDING_FILTER_MARGIN`` under ``limit`` cannot raise its row
    # past the limit; any other pair (a NaN or infinite bound included) flags its row. A row
    # nothing flags stays under the limit whatever subset of its terms is summed, so only the faces
    # touching a flagged row need the ``float64`` pass, and those rows get every term they had.
    # Thread 0 also zeroes ``out_listed``'s counter for ``list_rounding_faces`` (when listing).
    f = wp.int32(wp.tid())
    if f == 0 and out_listed.shape[0] != 0:
        out_listed[0] = 0
    l0, l1, l2 = row_triple(edge_lengths, f)
    c, bound = half_cotangent_sensitivities(l0, l1, l2)
    step = wp.float32(LENGTH_ROUNDING)
    for j in range(3):
        a = (j + 1) % 3
        b = (j + 2) % 3
        spread = ROUNDING_FILTER_MARGIN * step * (bound[a] + bound[b])
        if not (spread <= limit * (wp.abs(c[a]) + wp.abs(c[b]))):
            out_flags[faces[f * 3 + j]] = 1


@wp.func
def add_intrinsic_rounding_rows(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    terms: wp.array2d[wp.vec3d],
    sums: GroupSumPass,
    out_rows: wp.array[wp.vec2d],
    f: wp.int32,
) -> None:
    # Each half-cotangent's rounding movement, summed into the rows of the edge it weights, beside
    # the row's sum of ``|weight|``: ``(error, scale)`` rows of a ``_launch.group_sums`` buffer,
    # so with ``ordito.config``'s deterministic setting on the rows, the largest ratio and the
    # warning repeat bit for bit on CUDA. Bound by its ``float64`` arithmetic, not by the commits:
    # a first pass marking the rows that can exceed the limit (a row's ratio is a mediant of its
    # terms') measured slower, since the faces it keeps evaluate the bound twice; the ``float32``
    # filter that replaced it (``flag_rounding_rows``) evaluates it once.
    #
    # With the setting on, the first pass keeps each face's terms in ``terms`` (``(n_faces, 2)``:
    # bounds, then weights) for the second to read; off, ``terms`` is ``None`` and unwritten.
    # Against the arrival-order call's device time at 1.3 M faces the on mode is 1.5x, where
    # evaluating the ``float64`` bound in both passes measured 2.0x; ``float32`` rows (one word a
    # component) measured 1.3x and were declined, the rows being ``float64``.
    #
    # ``scatter.scatter_add_vector`` spelled out, its mode test taken once a face rather than per
    # term: the off arm commits in arrival order, the on arm adds the six terms (the same rows, so
    # the arms differ only by the commit), out of line on the CPU device, where the CUDA arm folds
    # away. On the CPU the six inlined on-mode terms slowed the off mode (1.07-1.08x a separate
    # off-mode kernel of two ``float64`` atomics, which this body replaced) and six out-of-line
    # calls a face slowed the on mode (1.26-1.40x, their spilled registers); one call a face is
    # neither. New/old, the old being that off kernel and an on kernel of per-term
    # ``scatter_add_vector``: off CPU 0.87x, CUDA host 0.82x at 5 k faces and device 0.89-1.00x;
    # on CPU 0.95-0.98x, CUDA 1.00x; the bits unchanged on the CPU in both modes and on CUDA on.
    #
    # Declined (2026-11-09): this pass's first phase taken by the intrinsic key pass
    # (``cotmatrix_entries_intrinsic_and_operator_keys``, below the undirected threshold), one
    # launch fewer and byte-identical, device time 0.92x on ``bunny``; but ``robust_laplacian``
    # new/old CUDA 0.97-0.99x on ``sphere_small``, 1.00-1.02x on ``bunny_decimated`` and
    # 1.04-1.10x on ``bunny`` (all six minima slower), the call rising where the device fell.
    bound = wp.vec3d()
    weight = wp.vec3d()
    if sums.phase == 0:
        bound, weight = half_cotangent_rounding(edge_lengths, f)
        if sums.words.shape[0] != 0:
            terms[f, 0] = bound
            terms[f, 1] = weight
    else:
        bound = terms[f, 0]
        weight = terms[f, 1]
    if sums.words.shape[0] == 0:
        for k in range(3):
            for end in range(1, 3):
                v = half_cotangent_row(faces, f, k, end)
                arrival_order_add_vector(
                    out_rows, v, wp.vec2d(bound[k], weight[k]), ARRIVAL_POOL_UNIFORM
                )
    elif device_is_cpu() != 0:
        add_rounding_terms_out_of_line(faces, f, bound, weight, sums)
    else:
        add_rounding_terms(faces, f, bound, weight, sums)


@wp.func
def touches_flagged_row(
    faces: wp.array[wp.int32], flags: wp.array[wp.int32], f: wp.int32
) -> wp.int32:
    # Non-zero when face ``f`` adds to a row ``flag_rounding_rows`` flagged.
    return flags[faces[f * 3]] + flags[faces[f * 3 + 1]] + flags[faces[f * 3 + 2]]


@wp.kernel
def intrinsic_rounding_rows(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    flags: wp.array[wp.int32],
    terms: wp.array2d[wp.vec3d],
    sums: GroupSumPass,
    out_rows: wp.array[wp.vec2d],
) -> None:
    # One thread a face: ``add_intrinsic_rounding_rows``. With ``flags`` (``flag_rounding_rows``;
    # empty to take every face) a face touching no flagged row returns first, in both phases: the
    # CPU's form, whose threads run in turn. On CUDA the faces kept are scattered over most warps,
    # which then issue the ``float64`` work all the same (``lucy`` 1.06x, ``dragon`` 0.94x device
    # time with 16-23 % of the faces kept), so CUDA walks a list instead
    # (``intrinsic_rounding_rows_listed``).
    f = wp.int32(wp.tid())
    if flags.shape[0] != 0:
        if touches_flagged_row(faces, flags, f) == 0:
            return
    add_intrinsic_rounding_rows(faces, edge_lengths, terms, sums, out_rows, f)


@wp.kernel
def list_rounding_faces(
    faces: wp.array[wp.int32], flags: wp.array[wp.int32], out_listed: wp.array[wp.int32]
) -> None:
    # The faces touching a flagged row, appended from ``out_listed[1]`` with their count in
    # ``out_listed[0]`` (zeroed by ``flag_rounding_rows``), one atomic a warp. Their order is the
    # arrival order, which neither commit of ``add_intrinsic_rounding_rows`` reads: CUDA's
    # arrival-order atomics take any order, the exact group sum is the same in all.
    f = wp.int32(wp.tid())
    slot = append_slot_if(out_listed, 0, touches_flagged_row(faces, flags, f))
    if slot >= 0:
        out_listed[slot + 1] = f


@wp.kernel
def intrinsic_rounding_rows_listed(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    listed: wp.array[wp.int32],
    width: wp.int32,
    terms: wp.array2d[wp.vec3d],
    sums: GroupSumPass,
    out_rows: wp.array[wp.vec2d],
) -> None:
    # ``intrinsic_rounding_rows`` over ``list_rounding_faces``' list, ``width`` threads striding
    # it: the kept faces fill whole warps, and the count stays on the device.
    t = wp.int32(wp.tid())
    for k in range(t, listed[0], width):
        add_intrinsic_rounding_rows(faces, edge_lengths, terms, sums, out_rows, listed[k + 1])


@wp.func
def commit_row_ratio(row: wp.vec2d, limit: wp.float64, out_max: wp.array[wp.float64]) -> None:
    # One of ``intrinsic_rounding_rows``' rows into the largest ``error / scale`` past ``limit``
    # (``out_max`` stays ``0`` when no row exceeds it): only a row carrying weight past the limit
    # takes the one-address atomic, and a maximum is exact in any order.
    if row[1] > wp.float64(0.0):
        ratio = row[0] / row[1]
        if ratio > limit:
            wp.atomic_max(out_max, 0, ratio)


@wp.func
def edge_weight(
    a: wp.int32, b: wp.int32, vertices: wp.array[wp.vec3], equal_weight: wp.int32
) -> wp.float32:
    return point_edge_weight(vertices[a], vertices[b], equal_weight)


@wp.func
def point_edge_weight(pa: wp.vec3, pb: wp.vec3, equal_weight: wp.int32) -> wp.float32:
    # The umbrella weight of an edge from its endpoints' positions: 1, or the inverse length.
    # ``edge_weight`` reads them from a table; ``kernels/smoothing.umbrella_diffuse_pass`` narrows
    # them from the positions it diffuses.
    if equal_weight != 0:
        return wp.float32(1.0)
    return wp.float32(1.0) / (wp.length(pa - pb) + wp.float32(1.0e-12))


@wp.func
def edge_entry(
    edges: wp.array2d[wp.int32], e: wp.int32, vertices: wp.array[wp.vec3], equal_weight: wp.int32
) -> tuple[wp.int32, wp.int32, wp.float32]:
    # Edge ``e``'s endpoints and weight, which the triplet and the key kernels below both emit;
    # they differ only in whether an entry is written as ``(row, col)`` or as its sort key.
    a = edges[e, 0]
    b = edges[e, 1]
    return a, b, edge_weight(a, b, vertices, equal_weight)


@wp.kernel
def laplacian_triplets(
    edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    symmetric: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # One triplet per edge row, ``(a, b)``, matching trimesh's ``mesh.edges`` adjacency over the
    # directed triangle edges; with ``symmetric`` (warp-uniform) both directed pairs ``(a, b)`` and
    # ``(b, a)`` of each unique undirected edge, in slots ``2 e`` and ``2 e + 1``, so the adjacency
    # is symmetric, matching trimesh's ``vertex_neighbors``. Duplicate multiplicity cancels under
    # row-normalization. ``out_vals`` is generic: the float32 edge weight is cast to the requested
    # output dtype.
    e = wp.int32(wp.tid())
    a, b, weight = edge_entry(edges, e, vertices, equal_weight)
    w = out_vals.dtype(weight)
    slot = e * (symmetric + 1)
    out_rows[slot] = a
    out_cols[slot] = b
    out_vals[slot] = w
    if symmetric != 0:
        out_rows[slot + 1] = b
        out_cols[slot + 1] = a
        out_vals[slot + 1] = w


@wp.kernel
def laplacian_keys(
    edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    symmetric: wp.int32,
    n_vertices: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # ``laplacian_triplets``' entries as the sort keys ``array.csr_from_coo_keys`` reads, in the
    # same slots, for ``laplacian.laplacian``, which assembles them and never returns the triplets.
    e = wp.int32(wp.tid())
    a, b, weight = edge_entry(edges, e, vertices, equal_weight)
    w = out_vals.dtype(weight)
    slot = e * (symmetric + 1)
    store_coo_key(slot, a, b, n_vertices, n_vertices, out_keys, out_order)
    out_vals[slot] = w
    if symmetric != 0:
        store_coo_key(slot + 1, b, a, n_vertices, n_vertices, out_keys, out_order)
        out_vals[slot + 1] = w


@wp.kernel
def row_normalize(offsets: wp.array[wp.int32], out_values: wp.array[wp.Float]) -> None:
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    # ``out_values[0]`` is always valid: the launcher only runs this kernel when nnz > 0. It is
    # read solely to source the generic scalar type for the accumulator / zero literals.
    total = type(out_values[0])(0.0)
    for k in range(start, end):
        total += out_values[k]
    if total > type(out_values[0])(0.0):
        for k in range(start, end):
            out_values[k] = out_values[k] / total


# One row of the row-stochastic averaging operator, as a ``@wp.func`` so a consumer can apply it
# and use the result in the same thread instead of round-tripping an intermediate buffer through
# global memory and a second launch. Every smoothing filter that iterates ``L`` does exactly that
# (see ``ordito/smoothing.py``), so the row apply is the shared run rather than the kernel.
#
# Note the precision: the float32 weight is promoted to float64 because the accumulator is a
# ``wp.vec3d``. ``kernels/smoothing.diffuse_scalar_pass`` walks the same row on a float32 scalar
# field and accumulates in float32, so it cannot call this -- ``float64 * float32`` does not parse,
# and the float64-field form that would let one generic serve both was measured as a loss. That
# comment carries the reasoning.


@wp.func
def operator_row_entries(
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[Any],
    start: wp.int32,
    end: wp.int32,
    step: wp.int32,
) -> wp.vec3d:
    # ``operator_row``'s sum over a row's entries ``start``, ``start + step``, ... below ``end``:
    # a thread's whole row at step 1, or a block's lanes over a long row at the block width.
    acc = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(start, end, step):
        w = wp.float64(values[k])
        acc += w * wp.vec3d(field[columns[k]])
    return acc


@wp.func
def operator_row_warp(
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[Any],
    start: wp.int32,
    end: wp.int32,
    lanes: wp.int32,
) -> wp.vec3d:
    # ``operator_row``'s sum over entries ``start`` to ``end`` by the warp's lanes in ``lanes``
    # (``reduce.warp_active_lanes``), every one of which must call it: each strides the entries
    # from its rank, and the partial sums are shuffled into the lowest lane, which alone holds the
    # row's sum (``reduce.warp_lane_leads``). Four interleaved sums a lane cost the hub's walk
    # 1.34x (2026-11-09, ``fan_hub``: 845 -> 1 134 us a pass); one walk a lane it stays.
    part = operator_row_entries(
        columns, values, field, start + warp_lane_rank(lanes), end, popcount32(wp.uint32(lanes))
    )
    return wp.vec3d(
        warp_lanes_sum_float(lanes, part[0]),
        warp_lanes_sum_float(lanes, part[1]),
        warp_lanes_sum_float(lanes, part[2]),
    )


# Operators of more rows than this walk ``operator_row`` four entries a step on CUDA. A mesh row
# is about six entries, so the chunk's clamped duplicate loads cost an L2-resident operator more
# than its overlap buys: ungated, the explicit pass's device time read 1.08-1.11x on
# ``sphere_small`` and 1.04-1.19x on ``sphere_med`` (bunny par, dragon 0.97x), which the implicit
# filter's replayed steps exposed as 1.07-1.12x calls (2026-11-11).
OPERATOR_ROW_CHUNK_FROM = 1 << 18


@wp.func
def operator_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[Any],
    i: wp.int32,
) -> wp.vec3d:
    # Row ``i`` of the operator applied to a ``wp.vec3d`` field -- or a ``wp.vec3`` one, widened
    # exactly as each entry is read, so a caller holding ``float32`` positions gets the same sum as
    # from their ``float64`` copy without materializing it.
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        # Isolated vertex (empty row): the averaging operator acts as the identity so the
        # vertex does not drift toward the origin.
        return wp.vec3d(field[i])
    if device_is_cpu() != 0 or offsets.shape[0] <= OPERATOR_ROW_CHUNK_FROM:
        return operator_row_entries(columns, values, field, start, end, 1)
    # Four entries a step on CUDA past ``OPERATOR_ROW_CHUNK_FROM`` rows (``array.row_chunk_slots``,
    # as ``linalg.csr_row_dot``): the column, value and gathered position loads of a step are in
    # flight together, where an entry a step waited on each gather before the next column's load.
    # The terms still add in row order, so the sum is ``operator_row_entries``' bit for bit; the
    # CPU keeps that loop. Calls new/old (2026-11-11): lucy explicit filters 0.93-0.96x, implicit
    # 0.91x; dragon 0.95-0.98x; smaller meshes at par.
    acc = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(start, end, 4):
        slots = row_chunk_slots(k, end)
        p0 = wp.vec3d(field[columns[slots[0]]])
        p1 = wp.vec3d(field[columns[slots[1]]])
        p2 = wp.vec3d(field[columns[slots[2]]])
        p3 = wp.vec3d(field[columns[slots[3]]])
        w0 = wp.float64(values[slots[0]])
        w1 = wp.float64(values[slots[1]])
        w2 = wp.float64(values[slots[2]])
        w3 = wp.float64(values[slots[3]])
        acc += w0 * p0
        if k + 1 < end:
            acc += w1 * p1
        if k + 2 < end:
            acc += w2 * p2
        if k + 3 < end:
            acc += w3 * p3
    return acc


@wp.func
def store_operator_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    diagonal: wp.int32,
    t: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The sparsity of every vertex operator on a triangle mesh, as ``kernels/array.csr_key`` keys,
    # for the *directed* pattern build (small meshes): for each face and each corner ``e``, the
    # edge opposite it both ways -- slots ``6 * f + 2 * e`` and ``+ 1`` hold ``(i, j)`` and
    # ``(j, i)``, where ``i`` is corner ``e + 1`` and ``j`` corner ``e + 2`` -- and one diagonal
    # slot per vertex in the tail ``[6 * n_faces, 6 * n_faces + n_vertices)``. ``diagonal`` chooses
    # what the tail holds: nothing (0: the pattern has no diagonal slots), the diagonal of every
    # vertex a face references (1), or of every vertex (2). For 1 the caller fills the tail with the
    # sentinel first, so a vertex no face references keeps an empty row; every face writes the same
    # key into a shared vertex's slot, so the race is benign. For 2 the launch covers
    # ``max(n_faces, n_vertices)`` threads and thread ``v`` writes vertex ``v``'s key. Both
    # directions carry the corner slot ``3 * f + e`` as payload -- the convention
    # ``mesh_edge_keys`` shares, so one value kernel serves both builds; it reads the direction off
    # the row. A diagonal slot's payload is never read, so it is not written. A degenerate face's
    # self-edge keeps its keys and lands on the diagonal entry; each value kernel decides what that
    # means. This is thread ``t``'s share of the build.
    n_faces = faces.shape[0] // 3
    tail = 6 * n_faces
    if diagonal == 2 and t < n_vertices:
        out_keys[tail + t] = csr_key(t, t, n_vertices, n_vertices)
    if t >= n_faces:
        return
    for e in range(3):
        i = faces[t * 3 + (e + 1) % 3]
        j = faces[t * 3 + (e + 2) % 3]
        out_keys[t * 6 + e * 2] = csr_key(i, j, n_vertices, n_vertices)
        out_keys[t * 6 + e * 2 + 1] = csr_key(j, i, n_vertices, n_vertices)
        out_order[t * 6 + e * 2] = t * 3 + e
        out_order[t * 6 + e * 2 + 1] = t * 3 + e
        # Range-checked at the write: an index past the vertex count must not
        # become a store past the tail.
        vertex = faces[t * 3 + e]
        if diagonal == 1 and vertex >= 0 and vertex < n_vertices:
            out_keys[tail + vertex] = csr_key(vertex, vertex, n_vertices, n_vertices)


@wp.kernel
def mesh_operator_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    diagonal: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The directed pattern build's keys (``store_operator_keys``).
    t = wp.int32(wp.tid())
    store_operator_keys(faces, n_vertices, diagonal, t, out_keys, out_order)


@wp.kernel
def cotmatrix_entries_and_operator_keys(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    diagonal: wp.int32,
    out_cot: wp.array2d[wp.Float],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # ``mesh_operator_keys`` and ``cotmatrix_entries`` in one pass, for an operator built from the
    # faces alone (``laplacian._cotmatrix_entries_and_pattern``): the two kernels differ only in
    # that this one also stores face ``t``'s half-cotangents.
    t = wp.int32(wp.tid())
    store_operator_keys(faces, n_vertices, diagonal, t, out_keys, out_order)
    if t < faces.shape[0] // 3:
        store_half_cotangents(vertices, faces, t, out_cot)


@wp.kernel
def cotmatrix_entries_intrinsic_and_operator_keys(
    edge_lengths: wp.array2d[wp.float32],
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    diagonal: wp.int32,
    out_cot: wp.array2d[wp.Float],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # ``cotmatrix_entries_and_operator_keys`` from edge lengths (``laplacian.robust_laplacian``):
    # the two kernels differ only in where face ``t``'s half-cotangents come from, positions
    # (``store_half_cotangents``) or lengths (``store_intrinsic_cotangents``).
    t = wp.int32(wp.tid())
    store_operator_keys(faces, n_vertices, diagonal, t, out_keys, out_order)
    if t < faces.shape[0] // 3:
        store_intrinsic_cotangents(edge_lengths, t, out_cot)


@wp.kernel
def mesh_halfedge_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # One key per halfedge ``faces[h] -> faces[next(h)]``, the directed adjacency trimesh's
    # ``edges_to_coo(mesh.edges)`` builds -- a self-edge included, as an entry on the diagonal.
    # The payload is the halfedge index.
    h = wp.int32(wp.tid())
    out_keys[h] = csr_key(faces[h], halfedge_destination(faces, h), n_vertices, n_vertices)
    out_order[h] = h


# The *undirected* pattern build (large meshes) sorts one key per corner instead of two plus one per
# vertex: a run of equal ``(min, max)`` keys is one unique edge, and in that order every row's
# upper half -- the edges whose smaller endpoint it is -- is contiguous and sorted by column. The
# lower half is the transpose, laid out by a second, stable sort of the unique edges by their larger
# endpoint over 32-bit keys. Each row is ``[lower | diagonal | upper]``, and an entry's value comes
# from its edge's run in the first sort, so both builds hand the value kernels the same thing.


@wp.kernel
def mesh_edge_keys(
    faces: wp.array[wp.int32],
    n_vertices: wp.int32,
    mark: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_referenced: wp.array[wp.int32],
) -> None:
    # Corner ``e``'s opposite edge as an undirected ``(min, max)`` key with the corner slot
    # ``3 * f + e`` as payload. A degenerate face's self-edge never gets a key; instead ``mark``
    # chooses which vertices ``out_referenced`` flags for a diagonal slot: every in-range corner
    # (1), the endpoint of a self-edge (2) -- the entry the directed build keeps for it -- or every
    # vertex (3), for which the launch covers ``max(n_faces, n_vertices)`` threads and thread ``v``
    # flags vertex ``v``.
    t = wp.int32(wp.tid())
    if mark == 3 and t < n_vertices:
        out_referenced[t] = 1
    if t >= faces.shape[0] // 3:
        return
    for e in range(3):
        i = faces[t * 3 + (e + 1) % 3]
        j = faces[t * 3 + (e + 2) % 3]
        key = csr_key(wp.min(i, j), wp.max(i, j), n_vertices, n_vertices)
        if i == j:
            key = wp.uint64(n_vertices) * wp.uint64(n_vertices)
            if mark == 2 and i >= 0 and i < n_vertices:
                out_referenced[i] = 1
        out_keys[t * 3 + e] = key
        out_order[t * 3 + e] = t * 3 + e
        vertex = faces[t * 3 + e]
        if mark == 1 and vertex >= 0 and vertex < n_vertices:
            out_referenced[vertex] = 1


@wp.kernel
def mesh_edge_runs(
    keys: wp.array[wp.uint64],
    sentinel: wp.uint64,
    n_vertices: wp.int32,
    out_flags: wp.array[wp.int32],
    out_upper: wp.array[wp.int32],
    out_lower: wp.array[wp.int32],
    out_second_keys: wp.array[wp.int32],
    out_second_order: wp.array[wp.int32],
) -> None:
    # Over the sorted corner keys: each run start is one unique edge ``(lo, hi)``. Flag it (the
    # flags' scan ranks the edges), count it into ``lo``'s upper half and ``hi``'s lower half, and
    # key it by ``hi`` for the second sort, whose payload is its position here.
    i = wp.int32(wp.tid())
    flag = 0
    second = n_vertices
    if csr_run_start(keys, i, sentinel):
        lo = wp.int32(keys[i] // wp.uint64(n_vertices))
        hi = wp.int32(keys[i] % wp.uint64(n_vertices))
        wp.atomic_add(out_upper, lo, 1)
        wp.atomic_add(out_lower, hi, 1)
        flag = 1
        second = hi
    out_flags[i] = flag
    out_second_keys[i] = second
    out_second_order[i] = i


@wp.func
def mesh_row_start(before: wp.array2d[wp.int32], r: wp.int32) -> wp.int32:
    # Row ``r``'s first entry in the undirected build's ``[lower | diagonal | upper]`` rows.
    # ``before`` is the exclusive scan of the ``(3, n_vertices)`` tallies (upper-half count,
    # lower-half count, diagonal flag) taken as one flat array, so its second and third rows carry
    # the whole upper total and the upper-plus-lower total ahead of them; ``before[1, 0]`` and
    # ``before[2, 0]`` are exactly those totals, and removing them leaves each part's own prefix.
    return before[0, r] + (before[1, r] - before[1, 0]) + (before[2, r] - before[2, 0])


@wp.func
def mesh_place_upper(
    keys: wp.array[wp.uint64],
    inclusive: wp.array[wp.int32],
    sentinel: wp.uint64,
    n_vertices: wp.int32,
    tallies: wp.array2d[wp.int32],
    before: wp.array2d[wp.int32],
    i: wp.int32,
    out_columns: wp.array[wp.int32],
    out_run_start: wp.array[wp.int32],
) -> None:
    # Edge ``(lo, hi)`` of sorted key ``i``, in row ``lo``: after the lower half and the diagonal,
    # at its rank among the edges whose smaller endpoint is ``lo`` -- its global rank less the edges
    # of earlier rows.
    if not csr_run_start(keys, i, sentinel):
        return
    lo = wp.int32(keys[i] // wp.uint64(n_vertices))
    rank = inclusive[i] - 1 - before[0, lo]
    slot = mesh_row_start(before, lo) + tallies[1, lo] + tallies[2, lo] + rank
    out_columns[slot] = wp.int32(keys[i] % wp.uint64(n_vertices))
    out_run_start[slot] = i


@wp.kernel
def mesh_place_columns(
    keys: wp.array[wp.uint64],
    inclusive: wp.array[wp.int32],
    sentinel: wp.uint64,
    second_keys: wp.array[wp.int32],
    second_order: wp.array[wp.int32],
    count: wp.int32,
    n_vertices: wp.int32,
    tallies: wp.array2d[wp.int32],
    before: wp.array2d[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_run_start: wp.array[wp.int32],
) -> None:
    # Launched over ``max(count, n_vertices)``; every write lands in a slot no other thread writes.
    # As a vertex: its row's offset (the last also closes the total) and its diagonal, which has no
    # contributors. As a position of the first sort: its upper-half entry (``mesh_place_upper``).
    # As a sorted position of the second sort: edge ``(lo, hi)`` in row ``hi``, at its rank among
    # the edges whose larger endpoint is ``hi`` -- stable, so in ascending ``lo``.
    #
    # Past L2 this is ~3.5x a streaming pass over the same bytes (``lucy``: 5.1 against 1.5 ms;
    # ``dragon`` 0.036 ms, in L2), and not for the lower half's ``keys[i]`` gather (0.3 ms of it).
    # Declined (2026-11-09): the lower half as a launch of its own, 2.0 + 1.9 ms of device time
    # against 5.1, byte-identical, but calls 1.02-1.04x from ``dragon`` to a 1.3 M-face icosphere
    # (the launch) and at par on ``lucy``, whose wall the mempool's trim after the call hides.
    t = wp.int32(wp.tid())
    if t < n_vertices:
        start = mesh_row_start(before, t)
        out_offsets[t] = start
        if t == n_vertices - 1:
            out_offsets[n_vertices] = start + tallies[0, t] + tallies[1, t] + tallies[2, t]
        if tallies[2, t] != 0:
            slot = start + tallies[1, t]
            out_columns[slot] = t
            out_run_start[slot] = 0
    if t < count:
        mesh_place_upper(
            keys, inclusive, sentinel, n_vertices, tallies, before, t, out_columns, out_run_start
        )
        hi = second_keys[t]
        if hi < n_vertices:
            i = second_order[t]
            slot = mesh_row_start(before, hi) + t - (before[1, hi] - before[1, 0])
            out_columns[slot] = wp.int32(keys[i] // wp.uint64(n_vertices))
            out_run_start[slot] = i


@wp.func
def cotmatrix_offdiagonal(
    k: wp.int32,
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: Any,
    zero: wp.Float,
) -> wp.Float:
    # Off-diagonal entry ``k`` of the cotangent Laplacian over a mesh operator pattern: the sum of
    # its contributing half-cotangents -- the corner slots ``order[p]`` over its run of equal
    # sorted ``keys`` from ``run_start[k]``, in face order since the sort is stable -- each cast
    # once to ``zero``'s precision, whatever the entries' own. The row walk of ``cotmatrix_rows``
    # and of ``kernels/heat.heat_system_rows``, which forms the heat system from the same entries.
    value = zero
    for p in range(run_start[k], sorted_run_end(keys, run_start[k], count)):
        q = order[p]
        value += type(zero)(cot_entries[q // 3, q % 3])
    return value


# A thread-a-row operator builder takes a long row (``array.csr_row_long``, a fan's hub) as a
# warp, as the explicit smoothing passes do (``kernels/smoothing.warp_long_operator_row``): one
# ballot names the warp's long rows (``reduce.warp_long_rows``), each lane whose row is short writes
# it in its own thread as before, and the warp walks each long row in turn (``reduce.warp_walk``),
# a row sum (a diagonal, a normalization) taken over the lanes in a fixed tree and broadcast to
# every lane (``reduce.warp_walk_total``). One row function serves both, its thread form
# ``lanes == 0``. Every short row is its thread's, to the bit; on the CPU device the ballot is empty
# and every row is. A long row's sum takes the lanes' tree, not the row's order. The kernel reaches
# the dispatch converged (its first statement), and no thread reads a row's values back after it
# (a lane's write is not visible to another lane without a warp barrier).


@wp.func
def signed_cotmatrix_row_lanes(
    lanes: wp.int32,
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: Any,
    negate: wp.int32,
    out_values: Any,
) -> None:
    # Row ``i`` of the cotangent Laplacian over a mesh operator pattern
    # (``laplacian._mesh_operator_pattern``), written into ``out_values``, or with ``negate`` row
    # ``i`` of ``-L``, each entry negated as it is written (exact, so the same bits with the sign
    # flipped). An off-diagonal is ``cotmatrix_offdiagonal`` and the diagonal is minus the row's
    # off-diagonal sum, in column order, so every row sums to zero up to that one sum's rounding. A
    # degenerate face's self-edge never reaches a row: its four contributions ``w + w - w - w``
    # cancelled exactly in the 12-triplet form too. The half-cotangents may be float32 or float64
    # whatever the matrix precision; each is cast once. ``cotmatrix_row`` is the unnegated row;
    # ``kernels/parametrization.negated_cotmatrix_rows`` the negated one. Walked by the lanes of
    # ``lanes`` (``reduce.warp_walk``), or by one thread at ``lanes == 0``.
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    total = type(out_values[start])(0.0)
    slot = wp.int32(-1)
    first, stride = warp_walk(lanes)
    for k in range(start + first, end, stride):
        if columns[k] == i:
            slot = k
            continue
        value = cotmatrix_offdiagonal(
            k, run_start, keys, count, order, cot_entries, type(out_values[k])(0.0)
        )
        out_values[k] = wp.where(negate != 0, -value, value)
        total += value
    total = type(total)(warp_walk_total(lanes, wp.float64(total)))
    if slot >= 0:
        out_values[slot] = wp.where(negate != 0, total, -total)


@wp.func
def signed_cotmatrix_row(
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: Any,
    negate: wp.int32,
    out_values: Any,
) -> None:
    # ``signed_cotmatrix_row_lanes`` by the one thread.
    signed_cotmatrix_row_lanes(
        0, i, offsets, columns, run_start, keys, count, order, cot_entries, negate, out_values
    )


@wp.func
def signed_cotmatrix_rows(
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: Any,
    negate: wp.int32,
    out_values: Any,
) -> None:
    # Row ``i`` by its thread and the warp's long rows by the warp: the body of the kernels that
    # write the cotangent Laplacian a row a thread.
    lanes = warp_active_lanes()
    long_rows = warp_long_rows(offsets, i, lanes)
    if long_rows == 0 or not csr_row_long(offsets, i):
        signed_cotmatrix_row(
            i, offsets, columns, run_start, keys, count, order, cot_entries, negate, out_values
        )
    while long_rows != 0:
        row = warp_lowest_row(lanes, long_rows, i)
        signed_cotmatrix_row_lanes(
            lanes,
            row,
            offsets,
            columns,
            run_start,
            keys,
            count,
            order,
            cot_entries,
            negate,
            out_values,
        )
        long_rows = long_rows & (long_rows - 1)


@wp.func
def cotmatrix_row(
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: Any,
    out_values: Any,
) -> None:
    # Row ``i`` of the cotangent Laplacian over a mesh operator pattern: ``signed_cotmatrix_row``
    # unnegated. The body of ``cotmatrix_rows``, and of the kernels that form the Laplacian's row
    # in the thread consuming it (``kernels/energies``).
    signed_cotmatrix_row(
        i, offsets, columns, run_start, keys, count, order, cot_entries, 0, out_values
    )


@wp.kernel
def cotmatrix_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    out_values: wp.array[wp.Float],
) -> None:
    # One thread per row of a mesh operator pattern: ``cotmatrix_row``, a long row by the warp
    # (``signed_cotmatrix_rows``).
    i = wp.int32(wp.tid())
    signed_cotmatrix_rows(
        i, offsets, columns, run_start, keys, count, order, cot_entries, 0, out_values
    )


@wp.kernel
def cotmatrix_rows_and_rounding_ratio(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    rounding: wp.array[wp.vec2d],
    rounding_sums: GroupSumPass,
    limit: wp.float64,
    out_values: wp.array[wp.Float],
    out_max: wp.array[wp.float64],
) -> None:
    # ``cotmatrix_rows`` for ``laplacian.robust_laplacian``, which also commits row ``i``'s
    # rounding ratio (``commit_row_ratio``): the two passes ran over the same vertices. The rows
    # are read through the consumer-fused finish (``read_group_sum_vector``), so with the setting
    # on no finish launch or result buffer is needed either.
    i = wp.int32(wp.tid())
    signed_cotmatrix_rows(
        i, offsets, columns, run_start, keys, count, order, cot_entries, 0, out_values
    )
    commit_row_ratio(read_group_sum_vector(rounding, rounding_sums, i), limit, out_max)


@wp.func
def umbrella_entry(
    w: wp.Float,
    i: wp.int32,
    c: wp.int32,
    k: wp.int32,
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    symmetric: wp.int32,
) -> wp.Float:
    # Entry ``k`` (row ``i``, column ``c``) of the umbrella operator before its row is normalized,
    # from its edge weight ``w``: see ``laplacian_rows``. Shared with
    # ``kernels/smoothing.umbrella_diffuse_pass``, which forms the row where it applies it.
    value = type(w)(0.0)
    if symmetric != 0:
        value = w
        if c == i:
            value += w
    else:
        for _p in range(run_start[k], sorted_run_end(keys, run_start[k], count)):
            value += w
    return value


@wp.func
def umbrella_row(
    lanes: wp.int32,
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    symmetric: wp.int32,
    out_values: Any,
) -> None:
    # Row ``i`` of ``laplacian_rows``, walked by the lanes of ``lanes`` (``reduce.warp_walk``), or
    # by one thread at ``lanes == 0``.
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    total = type(out_values[start])(0.0)
    first, stride = warp_walk(lanes)
    for k in range(start + first, end, stride):
        c = columns[k]
        w = type(out_values[k])(edge_weight(i, c, vertices, equal_weight))
        value = umbrella_entry(w, i, c, k, run_start, keys, count, symmetric)
        out_values[k] = value
        total += value
    total = type(total)(warp_walk_total(lanes, wp.float64(total)))
    if total > type(total)(0.0):
        for k in range(start + first, end, stride):
            out_values[k] = out_values[k] / total


@wp.kernel
def laplacian_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    symmetric: wp.int32,
    out_values: wp.array[wp.Float],
) -> None:
    # The row-normalized umbrella operator over a mesh pattern with no diagonal slots. An entry's
    # weight is ``edge_weight`` of its row and column (cast once). Directed (``symmetric == 0``,
    # trimesh's ``mesh.edges``): once per halfedge in its run, so a duplicated halfedge counts
    # twice. Symmetric (trimesh's ``vertex_neighbors``): once per unique edge whatever its run --
    # but twice for a degenerate face's self-edge, whose unique ``(a, a)`` edge emits both of its
    # directions onto one entry. The row is then divided by its sum, in column order, if positive.
    # A long row is the warp's (``signed_cotmatrix_row_lanes``' comment), its sum the lanes' tree.
    i = wp.int32(wp.tid())
    lanes = warp_active_lanes()
    long_rows = warp_long_rows(offsets, i, lanes)
    if long_rows == 0 or not csr_row_long(offsets, i):
        umbrella_row(
            0,
            i,
            offsets,
            columns,
            run_start,
            keys,
            count,
            vertices,
            equal_weight,
            symmetric,
            out_values,
        )
    while long_rows != 0:
        row = warp_lowest_row(lanes, long_rows, i)
        umbrella_row(
            lanes,
            row,
            offsets,
            columns,
            run_start,
            keys,
            count,
            vertices,
            equal_weight,
            symmetric,
            out_values,
        )
        long_rows = long_rows & (long_rows - 1)


@wp.func
def pattern_row_on_rim_lanes(
    lanes: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    i: wp.int32,
) -> wp.bool:
    # Whether vertex ``i`` lies on the mesh's rim, read off a ``cotmatrix`` pattern: one of its
    # off-diagonal entries' runs holds a single key, i.e. that edge lies on one face. Walked by the
    # lanes of ``lanes`` (``reduce.warp_walk``), each lane answering for the whole row, or by one
    # thread at ``lanes == 0``.
    rim = wp.bool(False)
    first, stride = warp_walk(lanes)
    for k in range(offsets[i] + first, offsets[i + 1], stride):
        if columns[k] != i:
            if sorted_run_end(keys, run_start[k], count) - run_start[k] == 1:
                rim = True
    if lanes != 0:
        rim = warp_ballot(lanes, wp.where(rim, 1, 0)) != 0
    return rim


@wp.func
def pattern_row_on_rim(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    i: wp.int32,
) -> wp.bool:
    # ``pattern_row_on_rim_lanes`` by the one thread. Shared by the face-weighted umbrella and the
    # heat method's boundary-pinned system (``kernels/heat``).
    return pattern_row_on_rim_lanes(0, offsets, columns, run_start, keys, count, i)


@wp.func
def face_weighted_row(
    lanes: wp.int32,
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    out_values: Any,
) -> None:
    # Row ``i`` of ``face_weighted_laplacian_rows``, walked by the lanes of ``lanes``
    # (``reduce.warp_walk``), or by one thread at ``lanes == 0``.
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    rim = pattern_row_on_rim_lanes(lanes, offsets, columns, run_start, keys, count, i)
    total = type(out_values[start])(0.0)
    first, stride = warp_walk(lanes)
    for k in range(start + first, end, stride):
        value = type(out_values[k])(0.0)
        if columns[k] == i:
            value = wp.where(rim, type(value)(2.0), type(value)(1.0))
        else:
            faces_on_edge = sorted_run_end(keys, run_start[k], count) - run_start[k]
            if not rim:
                value = type(value)(faces_on_edge)
            elif faces_on_edge == 1:
                value = type(value)(1.0)
        out_values[k] = value
        total += value
    total = type(total)(warp_walk_total(lanes, wp.float64(total)))
    for k in range(start + first, end, stride):
        out_values[k] = out_values[k] / total


@wp.kernel
def face_weighted_laplacian_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    out_values: wp.array[wp.Float],
) -> None:
    # The face-weighted umbrella over the ``cotmatrix`` pattern (every edge both ways plus every
    # referenced vertex's diagonal): an edge's run holds one key per face on it, so its length is
    # the weight a face-by-face accumulation gives that neighbour, and the vertex counts itself
    # once. A rim vertex -- one of whose edges lies on a single face -- averages along the rim
    # only: its rim neighbours once each and itself twice, its interior edges explicit zeros. The
    # row is divided by its sum, in column order. ``laplacian_rows`` is the plain 1-ring mean over
    # its own pattern; the two differ in the weights and the rim rule. A long row is the warp's
    # (``signed_cotmatrix_row_lanes``' comment), its sum the lanes' tree.
    i = wp.int32(wp.tid())
    lanes = warp_active_lanes()
    long_rows = warp_long_rows(offsets, i, lanes)
    if long_rows == 0 or not csr_row_long(offsets, i):
        face_weighted_row(0, i, offsets, columns, run_start, keys, count, out_values)
    while long_rows != 0:
        row = warp_lowest_row(lanes, long_rows, i)
        face_weighted_row(lanes, row, offsets, columns, run_start, keys, count, out_values)
        long_rows = long_rows & (long_rows - 1)


@wp.func
def graph_laplacian_row(
    lanes: wp.int32,
    i: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_values: Any,
) -> None:
    # Row ``i`` of ``graph_laplacian_rows``, walked by the lanes of ``lanes``
    # (``reduce.warp_walk``), or by one thread at ``lanes == 0``.
    start = offsets[i]
    end = offsets[i + 1]
    first, stride = warp_walk(lanes)
    for k in range(start + first, end, stride):
        if columns[k] == i:
            out_values[k] = -type(out_values[k])(end - start - 1)
        else:
            out_values[k] = type(out_values[k])(1.0)


@wp.kernel
def graph_laplacian_rows(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], out_values: wp.array[wp.Float]
) -> None:
    # ``A - diag(deg)`` over a mesh pattern with every vertex's diagonal: one per neighbour, the
    # diagonal minus their count (``igl::adjacency_matrix``'s unit weights). A degenerate face's
    # self-edge is not a neighbour. A long row is the warp's (``signed_cotmatrix_row_lanes``'
    # comment): a row's columns are distinct, so its diagonal is minus its length less one, and
    # whichever lane meets the diagonal writes it with no sum.
    i = wp.int32(wp.tid())
    lanes = warp_active_lanes()
    long_rows = warp_long_rows(offsets, i, lanes)
    if long_rows == 0 or not csr_row_long(offsets, i):
        graph_laplacian_row(0, i, offsets, columns, out_values)
    while long_rows != 0:
        row = warp_lowest_row(lanes, long_rows, i)
        graph_laplacian_row(lanes, row, offsets, columns, out_values)
        long_rows = long_rows & (long_rows - 1)


@wp.func
def connection_offdiagonal(
    i: wp.int32,
    k: wp.int32,
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    cot_entries: Any,
    transport_rotations: wp.array[wp.vec2d],
    weight: wp.float64,
) -> tuple[wp.mat22d, wp.float64]:
    # Off-diagonal entry ``k`` of row ``i`` of the connection Laplacian, and ``weight`` advanced by
    # its weights -- the row's running diagonal, added one contribution at a time in the row's
    # order. Corner ``e``'s opposite halfedge ``h`` runs ``i -> j``; it contributes
    # ``-w R(-rho_h)`` to row ``i`` and ``-w R(rho_h)`` to row ``j``, summed over the contributing
    # faces in face order, ``R(rho_h)`` read as ``(cos, sin)`` from ``transport_rotations``
    # (``predicates.polar_unit`` of the transport angles), formed once a halfedge across the
    # device. Each halfedge serves two rows, and a hub row that formed its own ``float64`` sines
    # and cosines did so on its one block's SM, at a 64th of the ``float32`` rate (``fan_hub``'s
    # two hubs share a block: the table and the rows 0.33x the rows alone, dragon 0.76x). The
    # entries are byte-identical: ``sin(-x)`` is ``-sin(x)`` and ``cos(-x)`` is ``cos(x)`` bit
    # for bit. The row walk of ``connection_laplacian_rows`` and of
    # ``kernels/heat.vector_heat_system_rows``, which forms the vector heat system from the same
    # entries.
    value = wp.mat22d()
    for p in range(run_start[k], sorted_run_end(keys, run_start[k], count)):
        q = order[p]
        h = (q // 3) * 3 + (q % 3 + 1) % 3
        w = wp.float64(cot_entries[q // 3, q % 3])
        rotation = transport_rotations[h]
        c = rotation[0]
        sine = wp.where(faces[h] == i, -rotation[1], rotation[1])
        value += -w * wp.mat22d(c, -sine, sine, c)
        weight += w
    return value, weight


@wp.func
def connection_laplacian_row_block(
    i: wp.int32,
    lane: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    cot_entries: Any,
    transport_rotations: wp.array[wp.vec2d],
    out_values: wp.array[wp.mat22d],
) -> None:
    # Long row ``i`` of ``connection_laplacian_rows`` by the whole block (every lane must call it),
    # lanes striding its entries and the diagonal weight summed over them in the block's fixed
    # tree; the kernel writes its short rows inline (see there).
    weight = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        if columns[k] == i:
            slot = k
            continue
        value, weight = connection_offdiagonal(
            i, k, run_start, keys, count, order, faces, cot_entries, transport_rotations, weight
        )
        out_values[k] = value
    weight = block_sum(weight)
    if slot >= 0:
        out_values[slot] = weight * wp.mat22d(1.0, 0.0, 0.0, 1.0)


# Rows a block of ``connection_laplacian_rows`` holds, a row a lane on CUDA (the occupancy of a
# thread-a-row launch); on the CPU device the block's one lane walks them all.
CONNECTION_ROWS_PER_BLOCK = 256


@wp.kernel
def connection_laplacian_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    transport_rotations: wp.array[wp.vec2d],
    out_values: wp.array[wp.mat22d],
) -> None:
    # The vector Laplacian over a mesh operator pattern, a row a lane: each off-diagonal weight
    # becomes a rotation (``connection_offdiagonal``), since the two endpoints of an edge measure
    # tangent directions from different reference directions and a difference between them is
    # only meaningful after transporting one into the other's frame. The diagonal is
    # ``sum w * I`` over the row's entries in column order. Positive semi-definite (positive
    # diagonal), unlike ``cotmatrix``'s igl sign convention, because every consumer here feeds it
    # straight to a conjugate-gradient solve. A degenerate face's self-edge never reaches a row.
    # Launched tiled over ``CONNECTION_ROWS_PER_BLOCK`` rows a block: a long row
    # (``array.csr_row_long``, a fan's hub) is skipped by its lane and walked by the whole block
    # after (``connection_laplacian_row_block``), as ``kernels/heat.vector_heat_system_rows`` does,
    # its sum the block's tree. A warp a hub row held ``fan_hub`` at 14.7 ms and eight warps
    # about eight times faster; what then held the block was the rotations' ``float64`` sines and
    # cosines on its one SM, which the rotation table now forms across the device.
    # The short row stays written out here rather than as a row
    # function with a lane stride of one: called through a ``@wp.func`` the same body ran
    # 1.25-1.4x device time (2026-11-10, dragon 0.57 -> 0.72 ms, bunny 0.057 -> 0.080 ms), at the
    # same register count; inline it is at par. What the two copies share is the entry
    # (``connection_offdiagonal``); they differ in the walk and the sum.
    blk, lane = wp.tid()
    base, rows = block_chunk(offsets.shape[0] - 1, blk, CONNECTION_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    # The CPU device's one lane walks every row in turn anyway, so it defers none.
    defer = device_is_cpu() == 0
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = base + r
        if defer and csr_row_long(offsets, i):
            found = 1
            continue
        start = offsets[i]
        end = offsets[i + 1]
        weight = wp.float64(0.0)
        slot = wp.int32(-1)
        for k in range(start, end):
            if columns[k] == i:
                slot = k
                continue
            value, weight = connection_offdiagonal(
                i, k, run_start, keys, count, order, faces, cot_entries, transport_rotations, weight
            )
            out_values[k] = value
        if slot >= 0:
            out_values[slot] = weight * wp.mat22d(1.0, 0.0, 0.0, 1.0)
    if block_sum(found) == 0:
        return
    for r in range(rows):
        i = base + r
        if csr_row_long(offsets, i):
            connection_laplacian_row_block(
                i,
                lane,
                offsets,
                columns,
                run_start,
                keys,
                count,
                order,
                faces,
                cot_entries,
                transport_rotations,
                out_values,
            )


@wp.kernel
def face_edge_lengths_and_margin(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_lengths: wp.array2d[wp.float32],
    out_bounds: wp.array[wp.float32],
) -> None:
    # ``edges.face_edge_lengths``' table (``store_face_edge_lengths``), and the smallest
    # triangle-inequality margin ``a + b - c`` over every face and side into
    # ``out_bounds[1]`` (seeded ``+inf``): the two passes ran over the same faces, the margin
    # reading the lengths at its own row. ``laplacian.mollify_intrinsic`` turns the margin into the
    # constant ``delta = max(epsilon - margin, 0)`` that would have to be added to all three edges
    # of the worst face. Adding ``delta`` to all three grows each margin ``a + b - c`` by exactly
    # ``delta`` -- the two short sides gain ``2 * delta`` and the long side gives ``delta`` of it
    # back -- so the shortfall is the constant, undivided. Sharp & Crane (2020) eq. 3 states the
    # same rule, and an independent port of it (kentechx/HoleFillingPy, MIT) computes the identical
    # quantity a different way -- ``max(2 * max(L) - sum(L)) + delta``, which is
    # ``epsilon - min_f margin_f`` rearranged -- also undivided. ``epsilon - x`` rounds monotonely
    # in ``x``, so ``epsilon`` minus the smallest margin is bit for bit the largest per-face
    # shortfall, and the margin, needing no ``epsilon``, is folded beside the mean edge length.
    #
    # **``TOLERANCE_MOLLIFY = 1e-5`` sits between two arms, both of which were swept.** Below a few
    # ULP of the mean edge length the added constant does not survive the float32 store and the
    # mollification silently does nothing, which is the failure the whole function exists to
    # prevent -- so the left arm is a *storage* floor rather than a numerical-quality one, and it
    # moves with the mesh's length distribution. On the right, the perturbation this introduces into
    # rows no degenerate face touches grows linearly with epsilon and reaches the tolerance the igl
    # ``intrinsic_delaunay_cotmatrix`` parity test runs at around 1e-4. ``1e-5`` is more than a
    # decade clear of both, and a decade more conservative than that port's own 1e-4, which sits on
    # the right arm. A clean mesh yields ``delta == 0`` at every epsilon, so none of this is paid
    # where nothing is degenerate.
    #
    # Lane-strided by ``wp.block_dim()`` over the block's ``ITEMS_PER_BLOCK_1D`` faces; a minimum
    # is exact in any order, so one ``atomic_min`` per block commits it.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(faces.shape[0] // 3, block)
    low = wp.float32(wp.inf)
    for k in range(lane, count, wp.block_dim()):
        a, b, c = store_face_edge_lengths(vertices, faces, offset + k, out_lengths)
        low = wp.min(low, wp.min(wp.min(a + b - c, b + c - a), c + a - b))
    low = block_min(low)
    if lane == 0 and count > 0:
        wp.atomic_min(out_bounds, 1, low)


@wp.func
def add_constant(length: wp.float32, delta: wp.float32) -> wp.float32:
    return length + delta


@wp.func
def diagonal_pattern_row(
    i: wp.int32, out_offsets: wp.array[wp.int32], out_columns: wp.array[wp.int32]
) -> None:
    # Row ``i`` of a diagonal matrix's compressed rows: one entry, at column ``i``. Row 0 also
    # writes the offsets' leading zero, so the ``n + 1`` offsets need no fill.
    out_columns[i] = i
    out_offsets[i + 1] = i + 1
    if i == 0:
        out_offsets[0] = 0


@wp.kernel
def mass_diagonal_rows(
    sums: wp.array[wp.float64],
    count: wp.float64,
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.Float],
) -> None:
    # ``laplacian.mass_matrix`` as compressed rows: each vertex's exact area sum over ``count``
    # (three), one rounding, in the matrix precision, at the vertex's diagonal. ``out_values`` may
    # be ``sums`` itself. ``diagonal_pattern`` is the same rows with the values already formed.
    i = wp.int32(wp.tid())
    out_values[i] = type(out_values[i])(sums[i] / count)
    diagonal_pattern_row(i, out_offsets, out_columns)


@wp.kernel
def diagonal_pattern(out_offsets: wp.array[wp.int32], out_columns: wp.array[wp.int32]) -> None:
    # The compressed rows of a diagonal matrix whose values are stored already, one row a thread.
    i = wp.int32(wp.tid())
    diagonal_pattern_row(i, out_offsets, out_columns)


# Concrete overloads, registered at import -- see the long-form rationale in
# ``ordito/kernels/reduce.py``. In short: these kernels are
# generic, Warp instantiates an overload on the first launch at each new dtype, and a module's hash
# covers the instantiated set -- so a lazily-created overload rebuilds the whole module.
#
# ``ordito.laplacian`` exposes the precision as a public ``dtype`` keyword documented as "may be
# float32 or float64", so both are reachable for every kernel here.
_MATRIX_DTYPES = (wp.float32, wp.float64)


# The concrete handles keyed by the caller's dtype -- see
# [`OverloadTable`][ordito.kernels.array.OverloadTable]; worth about a tenth of
# ``laplacian.cotmatrix`` / ``laplacian.laplacian``, from removing one generic launch each.
# ``COTMATRIX_ROWS`` keys on the pair ``(entry dtype, matrix dtype)`` because those two
# templates are independent, exactly as the registration already was.
COTMATRIX_ENTRIES: OverloadTable
COTMATRIX_ENTRIES_AND_OPERATOR_KEYS: OverloadTable
COTMATRIX_ENTRIES_INTRINSIC: OverloadTable
COTMATRIX_ENTRIES_INTRINSIC_AND_OPERATOR_KEYS: OverloadTable
ROW_NORMALIZE: OverloadTable
LAPLACIAN_TRIPLETS: OverloadTable
LAPLACIAN_KEYS: OverloadTable
COTMATRIX_ROWS: OverloadTable
COTMATRIX_ROWS_AND_ROUNDING_RATIO: OverloadTable
LAPLACIAN_ROWS: OverloadTable
GRAPH_LAPLACIAN_ROWS: OverloadTable
FACE_WEIGHTED_LAPLACIAN_ROWS: OverloadTable
CONNECTION_LAPLACIAN_ROWS: OverloadTable
MASS_DIAGONAL_ROWS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global COTMATRIX_ENTRIES, COTMATRIX_ENTRIES_AND_OPERATOR_KEYS, COTMATRIX_ENTRIES_INTRINSIC
    global COTMATRIX_ENTRIES_INTRINSIC_AND_OPERATOR_KEYS
    global ROW_NORMALIZE
    global LAPLACIAN_TRIPLETS, LAPLACIAN_KEYS
    global COTMATRIX_ROWS, COTMATRIX_ROWS_AND_ROUNDING_RATIO, CONNECTION_LAPLACIAN_ROWS
    global LAPLACIAN_ROWS, GRAPH_LAPLACIAN_ROWS
    global FACE_WEIGHTED_LAPLACIAN_ROWS
    global MASS_DIAGONAL_ROWS
    COTMATRIX_ENTRIES = OverloadTable(
        cotmatrix_entries,
        {d: [wp.array[wp.vec3], wp.array[wp.int32], wp.array2d[d]] for d in _MATRIX_DTYPES},
    )
    COTMATRIX_ENTRIES_AND_OPERATOR_KEYS = OverloadTable(
        cotmatrix_entries_and_operator_keys,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.int32,
                wp.int32,
                wp.array2d[d],
                wp.array[wp.uint64],
                wp.array[wp.int32],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    COTMATRIX_ENTRIES_INTRINSIC = OverloadTable(
        cotmatrix_entries_intrinsic,
        {d: [wp.array2d[wp.float32], wp.array2d[d]] for d in _MATRIX_DTYPES},
    )
    COTMATRIX_ENTRIES_INTRINSIC_AND_OPERATOR_KEYS = OverloadTable(
        cotmatrix_entries_intrinsic_and_operator_keys,
        {
            d: [
                wp.array2d[wp.float32],
                wp.array[wp.int32],
                wp.int32,
                wp.int32,
                wp.array2d[d],
                wp.array[wp.uint64],
                wp.array[wp.int32],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    ROW_NORMALIZE = OverloadTable(
        row_normalize, {d: [wp.array[wp.int32], wp.array[d]] for d in _MATRIX_DTYPES}
    )
    LAPLACIAN_TRIPLETS = OverloadTable(
        laplacian_triplets,
        {
            d: [
                wp.array2d[wp.int32],
                wp.array[wp.vec3],
                wp.int32,
                wp.int32,
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    LAPLACIAN_KEYS = OverloadTable(
        laplacian_keys,
        {
            d: [
                wp.array2d[wp.int32],
                wp.array[wp.vec3],
                wp.int32,
                wp.int32,
                wp.int32,
                wp.array[wp.uint64],
                wp.array[wp.int32],
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    # ``cot_entries`` and the matrix precision are *independent* templates: cotmatrix's
    # docstring says the entries "may be float32 or float64 regardless of dtype: the assembly
    # kernel casts them to the matrix precision", so this is a genuine 2x2, not a diagonal.
    COTMATRIX_ROWS = OverloadTable(
        cotmatrix_rows,
        {
            (entry_dtype, dtype): [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.int32],
                wp.array2d[entry_dtype],
                wp.array[dtype],
            ]
            for dtype in _MATRIX_DTYPES
            for entry_dtype in _MATRIX_DTYPES
        },
    )
    COTMATRIX_ROWS_AND_ROUNDING_RATIO = OverloadTable(
        cotmatrix_rows_and_rounding_ratio,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.int32],
                wp.array2d[d],
                wp.array[wp.vec2d],
                GroupSumPass,
                wp.float64,
                wp.array[d],
                wp.array[wp.float64],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    # The connection Laplacian's values are always ``wp.mat22d``; only its cotangent entries
    # follow the caller, who may pass their own in place of the float64 default.
    LAPLACIAN_ROWS = OverloadTable(
        laplacian_rows,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.vec3],
                wp.int32,
                wp.int32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    GRAPH_LAPLACIAN_ROWS = OverloadTable(
        graph_laplacian_rows,
        {d: [wp.array[wp.int32], wp.array[wp.int32], wp.array[d]] for d in _MATRIX_DTYPES},
    )
    FACE_WEIGHTED_LAPLACIAN_ROWS = OverloadTable(
        face_weighted_laplacian_rows,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    CONNECTION_LAPLACIAN_ROWS = OverloadTable(
        connection_laplacian_rows,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.uint64],
                wp.int32,
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array2d[d],
                wp.array[wp.vec2d],
                wp.array[wp.mat22d],
            ]
            for d in _MATRIX_DTYPES
        },
    )
    MASS_DIAGONAL_ROWS = OverloadTable(
        mass_diagonal_rows,
        {
            d: [
                wp.array[wp.float64],
                wp.float64,
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
            ]
            for d in _MATRIX_DTYPES
        },
    )


_register_overloads()


def _declare_block_dims() -> None:
    """Declare the CUDA launch widths of this module's kernels besides the default (precompile)."""
    declare_block_dims(__name__, (TILE_1D,))


_declare_block_dims()
