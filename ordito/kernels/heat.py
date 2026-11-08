"""
Kernels for the three heat-diffusion solvers of [`ordito.heat`][ordito.heat].

Everything runs in ``float64``: far from the source the diffused field decays exponentially and
would underflow ``float32``, destroying the gradient direction and collapsing the far field. The
per-face half-cotangent weights are reused from ``ordito.laplacian`` (they are ``O(1)`` and
numerically safe in ``float32``); only the assembled operators, the diffused field and the linear
solves need double precision.

Three sections, in the order their wrappers appear: the scalar heat method's diffusion and gradient
normalization, the signed method's curve seeding and level-set constraints, and the vector method's
transport, extension and log map. They share `face_unit_gradient`, `tangent_to_world` and the
``float64`` convention, which is why they are one module rather than three.
"""

import math

import warp as wp

from ordito.constants import INT32_MAX_CONSTANT, TOLERANCE_ZERO_CONSTANT
from ordito.kernels.array import (
    LOOP_CONDITION,
    LOOP_PROGRESS,
    LOOP_ROUND,
    OverloadTable,
    advance_round_loop,
    append_slot,
    binary_search_index,
    cross2,
    csr_key,
    divide_if_positive,
    next_up,
    sorted_run_end,
    to_vec2,
    to_vec2d,
    to_vec3,
    to_vec3d,
    unique_edge_row,
)
from ordito.kernels.halfedge import halfedge_endpoints, halfedge_next, halfedge_prev, mate_halfedge
from ordito.kernels.laplacian import (
    connection_offdiagonal,
    cotmatrix_offdiagonal,
    face_half_cotangents,
    face_half_cotangents_float64,
    pattern_row_on_rim,
)
from ordito.kernels.linalg import (
    csr_row_long,
    shift_factor,
    shifted_block_entry,
    shifted_scalar_entry,
)
from ordito.kernels.predicates import (
    stable_length,
    stable_normalize,
    unit_tangent,
    vector_angle,
    world_to_tangent,
)
from ordito.kernels.reduce import (
    GroupSumPass,
    block_barrier,
    block_chunk,
    block_chunk_1d,
    block_max,
    block_sum,
    store_sum_and_count,
)
from ordito.kernels.scatter import (
    ARRIVAL_POOL_UNIFORM,
    add_corner_triple,
    add_or_store_corner_triple,
    scatter_add_vector,
)
from ordito.kernels.triangles import (
    corner_triple,
    face_normals_and_area,
    face_unit_gradient,
    face_vertices,
    face_vertices_vec3d,
    face_voronoi_areas,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def heat_face_terms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    write_cot: wp.int32,
    write_cot_float64: wp.int32,
    voronoi_mass: wp.int32,
    out_cot: wp.array2d[wp.float32],
    out_cot_float64: wp.array2d[wp.float64],
    out_normals: wp.array[wp.vec3],
    out_areas: wp.array[wp.float32],
    out_area_sums: wp.array[wp.float64],
) -> None:
    # Every per-face quantity the heat operators read, in one pass: the ``float32`` half-cotangent
    # table (``laplacian.cotmatrix_entries``' row, skipped when ``write_cot`` is 0 because the
    # caller has its own), and with ``write_cot_float64`` the same formed in ``float64``
    # (``laplacian.face_half_cotangents_float64``) for the vector system's rows; the unit normal
    # and area (``triangles.face_normals_and_areas``'); and the area's exact
    # ``float64`` scatter into each corner (``scatter.scatter_face_areas_exact``, the lumped mass
    # before its division by three). Each ``float32`` value is the standalone kernel's bit for bit.
    # With ``voronoi_mass`` set the corners take their Voronoi shares instead
    # (``triangles.face_voronoi_areas``), the lumped mass itself with no division to follow.
    #
    # ``float64`` because a cap's ``float32`` weight is off by ``eps / sin(largest angle)``, and the
    # vector heat system's rows cancel such weights against each other: on
    # ``torus_irregular_holes`` ``log_map(method="potpourri3d")`` was 7.9x the Class A bound from
    # potpourri3d's ``float64`` weights, and 0.07x with these.
    f = wp.int32(wp.tid())
    if write_cot != 0:
        c0, c1, c2 = face_half_cotangents(vertices, faces, f)
        out_cot[f, 0] = c0
        out_cot[f, 1] = c1
        out_cot[f, 2] = c2
    if write_cot_float64 != 0:
        d0, d1, d2 = face_half_cotangents_float64(vertices, faces, f)
        out_cot_float64[f, 0] = d0
        out_cot_float64[f, 1] = d1
        out_cot_float64[f, 2] = d2
    normal, area = face_normals_and_area(vertices, faces, f)
    out_normals[f] = normal
    out_areas[f] = area
    if voronoi_mass != 0:
        a0, a1, a2 = face_voronoi_areas(vertices, faces, f)
        add_corner_triple(out_area_sums, faces, f, wp.float64(a0), wp.float64(a1), wp.float64(a2))
    else:
        value = wp.float64(area)
        add_corner_triple(out_area_sums, faces, f, value, value, value)


@wp.kernel
def half_cotangents_float64(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cot: wp.array2d[wp.float64]
) -> None:
    # ``heat_face_terms``' ``float64`` table alone, for a vector system built over a scalar bundle
    # that did not form one.
    f = wp.int32(wp.tid())
    c0, c1, c2 = face_half_cotangents_float64(vertices, faces, f)
    out_cot[f, 0] = c0
    out_cot[f, 1] = c1
    out_cot[f, 2] = c2


@wp.func
def upper_edge_sums(
    row: wp.int32,
    start: wp.int32,
    end: wp.int32,
    step: wp.int32,
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    vertices: wp.array[wp.vec3],
    face_edges: wp.int32,
) -> wp.vec2d:
    # ``upper_edge_length_sum_and_count``'s sum and count over row ``row``'s entries ``start``,
    # ``start + step``, ... below ``end``.
    sums = wp.vec2d()
    for entry in range(start, end, step):
        column = columns[entry]
        if column > row:
            weight = wp.float64(1.0)
            if face_edges != 0:
                weight = wp.float64(sorted_run_end(keys, run_start[entry], count))
                weight -= wp.float64(run_start[entry])
            sums += wp.vec2d(
                weight * wp.float64(wp.length(vertices[column] - vertices[row])), weight
            )
    return sums


@wp.kernel
def upper_edge_length_sum_and_count(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    vertices: wp.array[wp.vec3],
    face_edges: wp.int32,
    stride: wp.int32,
    out_sum_and_count: wp.array[wp.float64],
) -> None:
    # Sum and count of the edge lengths over an operator's *strict upper triangle*, which for the
    # heat method's Laplacians -- one entry per edge, nothing pruned -- is
    # exactly the mesh's unique edge set. The timestep ``h ** 2`` needs only their mean, and the
    # operator already exists, so the edge set comes free where ``edges_unique`` would re-sort
    # every edge of the mesh to recover it. Reads the sparsity only, so the scalar and the
    # vector solver get the identical number from their two different operators.
    #
    # With ``face_edges`` set each edge counts once per face holding it -- its run of sorted keys
    # in the ``cotmatrix`` pattern -- which is the mean over every face's three edges, MeshLab's
    # heat-method timestep, where the unique mean is igl's and geometry-central's.
    #
    # ``reduce``'s mask-shaped block fold over *rows*: lanes stride the block's rows by
    # ``wp.block_dim()`` (right on the CPU device too), and one tile sum per quantity is stored in
    # the block's row (``store_sum_and_count``), added in a fixed order afterwards. A long row
    # (``linalg.csr_row_long``, a fan's hub) is left to the whole block after, its lanes striding
    # the entries into their own sums, so a block without one pays one reduction.
    i, t = wp.tid()
    n = offsets.shape[0] - 1
    base, remaining = block_chunk_1d(n, i)
    if remaining <= 0:
        return
    sums = wp.vec2d()
    found = wp.int32(0)
    for k in range(t, remaining, wp.block_dim()):
        row = base + k
        if csr_row_long(offsets, row):
            found = 1
        else:
            sums += upper_edge_sums(
                row, offsets[row], offsets[row + 1], 1, columns, run_start, keys, count,
                vertices, face_edges,
            )  # fmt: skip
    if block_sum(found) != 0:
        for k in range(remaining):
            row = base + k
            if csr_row_long(offsets, row):
                sums += upper_edge_sums(
                    row, offsets[row] + t, offsets[row + 1], wp.block_dim(), columns, run_start,
                    keys, count, vertices, face_edges,
                )  # fmt: skip
    store_sum_and_count(i, t, sums[0], sums[1], out_sum_and_count, stride)


# Rows a block of the heat systems' row kernels holds, a row a lane on CUDA (the occupancy of a
# thread-a-row launch); on the CPU device the block's one lane walks them all.
HEAT_ROWS_PER_BLOCK = 256


@wp.func
def vector_heat_row(
    i: wp.int32,
    start: wp.int32,
    end: wp.int32,
    lane: wp.int32,
    lanes: wp.int32,
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    transport_angles: wp.array[wp.float64],
    m: wp.float64,
    factor: wp.float64,
    out_values: wp.array[wp.mat22d],
) -> tuple[wp.float64, wp.int32]:
    # This lane's share of row ``i`` (pattern entries ``start`` to ``end``, not empty) of the vector
    # heat system ``M + factor L_connection``: entries ``start + lane``, every ``lanes``-th,
    # ``laplacian.connection_laplacian_rows``' (``connection_offdiagonal``) shifted as they are
    # formed (``linalg.shifted_block_entry``), ``m`` the vertex's scalar lumped mass, which both of
    # its tangent unknowns take. Returns the share's diagonal weight and the diagonal's slot if the
    # share holds it (``-1`` otherwise), for ``vector_heat_diagonal``.
    weight = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(start + lane, end, lanes):
        if columns[k] == i:
            slot = k
            continue
        value, weight = connection_offdiagonal(
            i, k, run_start, keys, count, order, faces, cot_entries, transport_angles, weight
        )
        out_values[k] = shifted_block_entry(value, factor, False, m)
    return weight, slot


@wp.func
def vector_heat_diagonal(
    i: wp.int32,
    slot: wp.int32,
    weight: wp.float64,
    columns: wp.array[wp.int32],
    m: wp.float64,
    factor: wp.float64,
    out_values: wp.array[wp.mat22d],
) -> None:
    # The vector heat row's diagonal from its whole weight (``vector_heat_row``), at ``slot`` when
    # the row has one. The flag is read off the pattern for the reason ``heat_system_rows`` gives.
    if slot >= 0:
        diagonal = weight * wp.mat22d(1.0, 0.0, 0.0, 1.0)
        out_values[slot] = shifted_block_entry(diagonal, factor, columns[slot] == i, m)


@wp.func
def scalar_heat_row(
    i: wp.int32,
    start: wp.int32,
    end: wp.int32,
    lane: wp.int32,
    lanes: wp.int32,
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    m: wp.float64,
    factor: wp.float64,
    out_laplacian: wp.array[wp.float64],
    out_heat: wp.array[wp.float64],
    out_poisson: wp.array[wp.float64],
) -> tuple[wp.float64, wp.int32]:
    # This lane's share of row ``i`` of ``heat_system_rows``' three scalar operators: its
    # off-diagonal entries (``cotmatrix_offdiagonal``), their sum and the diagonal's slot if the
    # share holds it, for ``scalar_heat_diagonal``.
    total = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(start + lane, end, lanes):
        if columns[k] == i:
            slot = k
            continue
        value = cotmatrix_offdiagonal(
            k, run_start, keys, count, order, cot_entries, wp.float64(0.0)
        )
        out_laplacian[k] = value
        total += value
        out_heat[k] = shifted_scalar_entry(value, factor, False, m)
        out_poisson[k] = -value
    return total, slot


@wp.func
def scalar_heat_diagonal(
    i: wp.int32,
    slot: wp.int32,
    total: wp.float64,
    columns: wp.array[wp.int32],
    m: wp.float64,
    factor: wp.float64,
    out_laplacian: wp.array[wp.float64],
    out_heat: wp.array[wp.float64],
    out_poisson: wp.array[wp.float64],
) -> None:
    # The scalar heat row's diagonal entries from its off-diagonal sum (``scalar_heat_row``).
    if slot >= 0:
        diagonal = -total
        out_laplacian[slot] = diagonal
        # The flag is read off the pattern rather than written as the constant it is here, so the
        # entry is formed exactly as ``shifted_system_values`` forms it: a constant flag lets the
        # CUDA compiler fuse the mass's addition into the product's rounding, which moves the last
        # bit of the diagonal.
        out_heat[slot] = shifted_scalar_entry(diagonal, factor, columns[slot] == i, m)
        out_poisson[slot] = -diagonal


@wp.kernel
def heat_system_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    scale: wp.float64,
    edge_sum_and_count: wp.array[wp.float64],
    from_edges: wp.int32,
    area_sums: wp.array[wp.float64],
    mass_divisor: wp.float64,
    faces: wp.array[wp.int32],
    transport_angles: wp.array[wp.float64],
    vector_cot: wp.array2d[wp.float64],
    out_laplacian: wp.array[wp.float64],
    out_heat: wp.array[wp.float64],
    out_poisson: wp.array[wp.float64],
    out_mass: wp.array[wp.float64],
    out_vector: wp.array[wp.mat22d],
) -> None:
    # The rows of the scalar heat method's three operators over a mesh operator pattern: the
    # ``float64`` cotangent Laplacian ``L`` (``laplacian.cotmatrix_rows``' entries,
    # ``cotmatrix_offdiagonal``), the heat system ``M + factor L`` (``linalg.shift_factor`` and
    # ``shifted_scalar_entry``, ``factor = -t``) and the Poisson operator ``-L``
    # (``linalg.shifted_system_values``' with ``negate`` set). The three are formed from the one
    # walk, so ``L`` is not read back between them. ``out_mass`` is the lumped mass,
    # ``heat_face_terms``' exact ``area_sums`` over ``mass_divisor`` -- three for the barycentric
    # lumping (``scatter.scale_face_sums``), one for the Voronoi shares, already the mass -- and
    # may be ``area_sums`` itself: each row's lane reads its own vertex's sum before it writes its
    # mass, and a long row's block reads the mass written.
    #
    # A non-empty ``out_vector`` also takes the row of the vector heat system at ``t`` (``-scale``)
    # over the same pattern, ``vector_heat_system_rows``' (``vector_heat_row``), from the
    # ``float64`` table ``vector_cot`` (``heat_face_terms``), for a vector assembly that builds its
    # scalar bundle alongside: the two kernels differ only in that this one has the mass in hand
    # rather than reading it back.
    #
    # Launch tiled over ``HEAT_ROWS_PER_BLOCK`` rows a block. A lane forms its row's entries in the
    # row's order; a long row (``linalg.csr_row_long``, a fan's hub) is left to the whole block
    # after, its lanes striding the entries and the block summing the diagonal in a fixed tree, so
    # a block without one pays one reduction.
    blk, lane = wp.tid()
    base, rows = block_chunk(offsets.shape[0] - 1, blk, HEAT_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    factor = shift_factor(scale, edge_sum_and_count, from_edges)
    vector_factor = shift_factor(-scale, edge_sum_and_count, from_edges)
    vector = out_vector.shape[0] > 0
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = base + r
        m = area_sums[i] / mass_divisor
        out_mass[i] = m
        start = offsets[i]
        end = offsets[i + 1]
        if end == start:
            continue
        if csr_row_long(offsets, i):
            found = 1
            continue
        if vector:
            weight, vector_slot = vector_heat_row(
                i, start, end, 0, 1, columns, run_start, keys, count, order, faces, vector_cot,
                transport_angles, m, vector_factor, out_vector,
            )  # fmt: skip
            vector_heat_diagonal(i, vector_slot, weight, columns, m, vector_factor, out_vector)
        total, slot = scalar_heat_row(
            i, start, end, 0, 1, columns, run_start, keys, count, order, cot_entries, m, factor,
            out_laplacian, out_heat, out_poisson,
        )  # fmt: skip
        scalar_heat_diagonal(
            i, slot, total, columns, m, factor, out_laplacian, out_heat, out_poisson
        )
    if block_sum(found) == 0:
        return
    lanes = wp.block_dim()
    for r in range(rows):
        i = base + r
        if csr_row_long(offsets, i):
            start = offsets[i]
            end = offsets[i + 1]
            m = out_mass[i]
            if vector:
                weight, vector_slot = vector_heat_row(
                    i, start, end, lane, lanes, columns, run_start, keys, count, order, faces,
                    vector_cot, transport_angles, m, vector_factor, out_vector,
                )  # fmt: skip
                weight = block_sum(weight)
                vector_slot = block_max(vector_slot)
                if lane == 0:
                    vector_heat_diagonal(
                        i, vector_slot, weight, columns, m, vector_factor, out_vector
                    )
            total, slot = scalar_heat_row(
                i, start, end, lane, lanes, columns, run_start, keys, count, order, cot_entries,
                m, factor, out_laplacian, out_heat, out_poisson,
            )  # fmt: skip
            total = block_sum(total)
            slot = block_max(slot)
            if lane == 0:
                scalar_heat_diagonal(
                    i, slot, total, columns, m, factor, out_laplacian, out_heat, out_poisson
                )


@wp.kernel
def rim_vertices(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    out_rim: wp.array[wp.int32],
    out_n_rim: wp.array[wp.int32],
) -> None:
    # 1 where a vertex lies on the rim (``laplacian.pattern_row_on_rim``), and their count in
    # ``out_n_rim[0]`` (pre-zeroed), for the heat system pinned to zero there.
    i = wp.int32(wp.tid())
    rim = pattern_row_on_rim(offsets, columns, run_start, keys, count, i)
    out_rim[i] = wp.where(rim, 1, 0)
    if rim:
        append_slot(out_n_rim, 0)


@wp.kernel
def pinned_heat_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    rim: wp.array[wp.int32],
    heat_values: wp.array[wp.float64],
    out_values: wp.array[wp.float64],
) -> None:
    # The heat system ``M - t L`` with the solution pinned to zero on the rim, over its own
    # pattern: a rim row becomes the identity's and an interior row drops its rim columns. Still
    # symmetric positive definite, and solved against a right-hand side zero on the rim it is the
    # interior system's solution padded with the rim's zeros -- ``igl::heat_geodesics``' Dirichlet
    # solve.
    i = wp.int32(wp.tid())
    for k in range(offsets[i], offsets[i + 1]):
        c = columns[k]
        value = heat_values[k]
        if rim[i] != 0:
            value = wp.where(c == i, wp.float64(1.0), wp.float64(0.0))
        elif rim[c] != 0:
            value = wp.float64(0.0)
        out_values[k] = value


@wp.func
def off_rim(value: wp.float64, rim: wp.int32) -> wp.float64:
    """``value``, or zero on a rim vertex: the pinned heat solve's right-hand side."""
    return wp.where(rim != 0, wp.float64(0.0), value)


@wp.func
def average_heat(neumann: wp.float64, pinned: wp.float64) -> wp.float64:
    """Average the Neumann and the rim-pinned heat, ``igl::heat_geodesics``' boundary rule."""
    return wp.float64(0.5) * (neumann + pinned)


@wp.kernel
def vector_heat_system_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    transport_angles: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    scale: wp.float64,
    edge_sum_and_count: wp.array[wp.float64],
    from_edges: wp.int32,
    out_values: wp.array[wp.mat22d],
) -> None:
    # The rows of the vector heat system ``M + factor L_connection`` over a mesh operator pattern
    # (``vector_heat_row``, ``linalg.shift_factor`` with ``factor = t``), so the connection
    # Laplacian itself is never stored, beside a scalar bundle built earlier: ``mass`` is its
    # lumped mass. ``heat_system_rows`` writes the same rows when it builds that bundle alongside,
    # and is launched the same way: tiled over ``HEAT_ROWS_PER_BLOCK`` rows a block, a long row
    # the block's.
    blk, lane = wp.tid()
    base, rows = block_chunk(offsets.shape[0] - 1, blk, HEAT_ROWS_PER_BLOCK)
    if rows <= 0:
        return
    factor = shift_factor(scale, edge_sum_and_count, from_edges)
    found = wp.int32(0)
    for r in range(lane, rows, wp.block_dim()):
        i = base + r
        start = offsets[i]
        end = offsets[i + 1]
        if end == start:
            continue
        if csr_row_long(offsets, i):
            found = 1
            continue
        weight, slot = vector_heat_row(
            i, start, end, 0, 1, columns, run_start, keys, count, order, faces, cot_entries,
            transport_angles, mass[i], factor, out_values,
        )  # fmt: skip
        vector_heat_diagonal(i, slot, weight, columns, mass[i], factor, out_values)
    if block_sum(found) == 0:
        return
    lanes = wp.block_dim()
    for r in range(rows):
        i = base + r
        if csr_row_long(offsets, i):
            weight, slot = vector_heat_row(
                i, offsets[i], offsets[i + 1], lane, lanes, columns, run_start, keys, count,
                order, faces, cot_entries, transport_angles, mass[i], factor, out_values,
            )  # fmt: skip
            weight = block_sum(weight)
            slot = block_max(slot)
            if lane == 0:
                vector_heat_diagonal(i, slot, weight, columns, mass[i], factor, out_values)


@wp.func
def tangent_to_world(tangent: wp.vec2, basis_x: wp.vec3, basis_y: wp.vec3) -> wp.vec3:
    # A tangent vector's world-space direction, in the vertex's own (orthonormal) frame. The
    # inverse of [`predicates.world_to_tangent`][ordito.kernels.predicates.world_to_tangent], and
    # here rather than beside it because ``ordito.heat`` publishes this half as a ``wp.map`` op.
    #
    # Above the section headers because two of the three solvers reach it: the signed method
    # expands each corner's tangent field to 3D before averaging, and the vector method's public
    # wrapper maps it over a whole transported field.
    return tangent[0] * basis_x + tangent[1] * basis_y


# --------------------------------------------------------------------------------------
# Scalar heat method: geodesic distance (Crane et al. 2013)
# --------------------------------------------------------------------------------------


@wp.func
def flag_out_of_range(
    v: wp.int32, n_vertices: wp.int32, out_invalid: wp.array[wp.int32]
) -> wp.bool:
    # Whether vertex index ``v`` falls outside ``[0, n_vertices)``, raising ``out_invalid[0]`` when
    # it does. The guard every seeding kernel below runs before it writes at a caller's index: the
    # write is skipped, so a bad index cannot reach memory, and the wrapper reads the one flag back
    # to raise. Concurrent threads only ever store ``1``, so the race between them is benign.
    if v < 0 or v >= n_vertices:
        out_invalid[0] = 1
        return True
    return False


@wp.kernel
def seed_source_indicator(
    sources: wp.array[wp.int32], out_u0: wp.array[wp.float64], out_invalid: wp.array[wp.int32]
) -> None:
    # Set the initial heat to 1 at each source vertex (out_u0 pre-zeroed by the caller); an
    # out-of-range source writes nothing and raises ``out_invalid[0]``.
    t = wp.int32(wp.tid())
    v = sources[t]
    if flag_out_of_range(v, out_u0.shape[0], out_invalid):
        return
    out_u0[v] = wp.float64(1.0)


@wp.func
def accumulate_face_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    f: wp.int32,
    x: wp.vec3d,
    out_div: wp.array[wp.float64],
    out_corner_div: wp.array2d[wp.float64],
) -> None:
    # Cotangent integrated divergence of one face's vector ``x``, accumulated onto its three
    # vertices (or stored per corner, ``scatter.add_or_store_corner_triple``). cot_entries[f, k] =
    # 1/2 cot(angle at corner k); each vertex gets contributions from the two edges of the
    # triangle incident to it, weighted by the cotangent opposite those edges.
    #
    # A ``@wp.func`` taking ``x`` as a value rather than a kernel reading it out of an array,
    # because the per-face field is always written by the immediately preceding launch and read
    # only at the producing thread's own face -- so every caller below forms it in a register
    # instead of round-tripping an ``(n_faces,)`` ``wp.vec3d`` buffer through global memory.
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    c0 = wp.float64(cot_entries[f, 0])
    c1 = wp.float64(cot_entries[f, 1])
    c2 = wp.float64(cot_entries[f, 2])

    d0 = c2 * wp.dot(v1 - v0, x) + c1 * wp.dot(v2 - v0, x)
    d1 = c0 * wp.dot(v2 - v1, x) + c2 * wp.dot(v0 - v1, x)
    d2 = c1 * wp.dot(v0 - v2, x) + c0 * wp.dot(v1 - v2, x)

    add_or_store_corner_triple(out_div, out_corner_div, faces, f, d0, d1, d2)


@wp.kernel
def unit_gradient_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    cot_entries: wp.array2d[wp.float32],
    out_div: wp.array[wp.float64],
    out_corner_div: wp.array2d[wp.float64],
) -> None:
    # The field's unit gradient direction, integrated by the same thread that forms it. For a
    # distance field the direction is radial, *away* from the source -- the opposite of the
    # ``X = -grad(u)/|grad(u)|`` the heat method integrates back into a distance. That is what
    # ``heat_geodesic`` wants all the same: its Poisson right-hand side is ``-div(X)``, and since
    # the divergence is linear in the field and a sign flip is exact in every product and sum
    # below, integrating the opposite direction accumulates that negation directly.
    #
    # The gradient is deliberately *not* merged into ``triangles.face_gradients`` behind a
    # ``normalize`` flag: the arithmetic is already shared -- ``face_unit_gradient`` is
    # ``normalize(face_gradient(...))`` -- so a flag would put a mode argument on that path which
    # only this module's callers would ever set (speculative generality: one caller per
    # mode). The two also return different quantities: a gradient carries the field's rate of
    # change, this carries only a direction.
    #
    # The terms go to ``out_div`` atomically (arrival order), or per corner into a non-empty
    # ``out_corner_div`` for ``scatter.sum_sorted_corner_values`` to add in ascending corner order
    # under ``ordito.config``'s deterministic mode, as in ``vertex_field_divergence``.
    f = wp.int32(wp.tid())
    x = face_unit_gradient(vertices, faces, normals, areas, values, f)
    accumulate_face_divergence(vertices, faces, cot_entries, f, x, out_div, out_corner_div)


@wp.kernel
def add_to_diagonal(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    shift: wp.float64,
    out_values: wp.array[wp.float64],
) -> None:
    # ``values + shift * I`` over a CSR pattern that stores every row's diagonal, one thread per
    # row: geometry-central's Poisson system ``L + 1e-6 I``, which
    # ``heat_geodesic(method="potpourri3d")`` solves, formed over the Poisson operator's own
    # pattern so it shares the heat system's analysis.
    i = wp.int32(wp.tid())
    for k in range(offsets[i], offsets[i + 1]):
        value = values[k]
        if columns[k] == i:
            value += shift
        out_values[k] = value


# --------------------------------------------------------------------------------------
# The heat method on geometry-central's intrinsic tufted cover
#
# ``heat_geodesic(method="potpourri3d", use_robust=True)``: what
# ``geometrycentral::surface::HeatMethodDistanceSolver`` builds with ``useRobustLaplacian``, and
# so ``potpourri3d.MeshHeatMethodDistanceSolver``'s default. Every face is doubled into a front
# copy and a reversed back copy over the same vertices (Sharp & Crane 2020's tufted cover), each
# edge glued front to front or front to back as its two faces wind it, and a rim edge glued to its
# own face's back copy, so the cover is a closed surface. Its lengths are mollified
# (``TOLERANCE_MOLLIFY`` of the mean edge length) and flipped to intrinsic Delaunay, and every
# operator -- mass, Laplacian, gradient, divergence, the timestep's mean edge length -- is formed
# on the cover from its lengths alone, in ``float64``.
#
# A cover halfedge is a slot ``h = 3 * f + e``: the edge of cover face ``f`` opposite its corner
# ``e``, running from corner ``e + 1`` to corner ``e + 2``; ``lengths[f, e]`` is its length and
# ``twin[h]`` the slot on the other side. Nothing here looks a halfedge up by its vertex labels,
# because flipping a rim edge's quad -- a face and its own back copy -- makes an edge from the apex
# to itself, and a face with a repeated vertex, which geometry-central allows
# (``SurfaceMesh::flip(e, false)``) and which then contributes nothing to the Laplacian. That is
# what separates these flips from ``kernels/remesh.intrinsic_delaunay_candidates``', which find a
# quad's corners by label, decline a flip onto one vertex, test the angle sum in ``float32`` and
# carry ``float32`` lengths: the two differ in the triangulation they reach.
# --------------------------------------------------------------------------------------


@wp.func
def cover_back_corner(e: wp.int32) -> wp.int32:
    # The back copy of a face ``(c0, c1, c2)`` is ``(c0, c2, c1)``: its edge opposite corner
    # ``cover_back_corner(e)`` is the front copy's edge opposite corner ``e``, run the other way.
    return wp.where(e == 0, wp.int32(0), 3 - e)


@wp.func
def cover_slot_of_halfedge(halfedge: wp.int32) -> wp.int32:
    # The cover slot of a mesh halfedge ``3 * f + k`` (corner ``k`` to corner ``k + 1``, the
    # ``halfedge.halfedge_mates`` convention): the front copy's edge opposite corner ``k + 2``.
    return halfedge_prev(halfedge)


@wp.kernel
def tufted_cover(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    out_cover_faces: wp.array[wp.int32],
    out_lengths: wp.array2d[wp.float64],
    out_twin: wp.array[wp.int32],
    out_nonmanifold: wp.array[wp.int32],
) -> None:
    # One thread per mesh halfedge slot (face ``f``, the edge opposite corner ``e``): its corner
    # in both copies, its length in ``float64`` from the widened positions (geometry-central's
    # ``edgeLengths``, formed from the same ``double`` coordinates), and the twins of its front
    # and back cover slots. The front copy of face ``f`` is cover face ``f``, its back copy cover
    # face ``n_faces + f``.
    #
    # The gluing is ``buildIntrinsicTuftedCover``'s for an edge of one or two faces: two faces
    # winding the edge opposite ways glue front to front and back to back, two winding it the
    # same way (a non-orientable pair) front to back, and a rim edge glues its front slot to its
    # own back slot. An edge of three or more faces (``mates`` codes ``-2 - lowest``) raises
    # ``out_nonmanifold[0]`` and leaves its twins to ``glue_nonmanifold_cover_edges``.
    h = wp.int32(wp.tid())
    n_faces = faces.shape[0] // 3
    f = h // 3
    e = h % 3
    back = 3 * (n_faces + f) + cover_back_corner(e)
    out_cover_faces[h] = faces[h]
    out_cover_faces[3 * (n_faces + f) + cover_back_corner(e)] = faces[h]
    tail = faces[halfedge_next(h)]
    tip = faces[halfedge_prev(h)]
    length = wp.length(wp.vec3d(vertices[tip]) - wp.vec3d(vertices[tail]))
    out_lengths[f, e] = length
    out_lengths[n_faces + f, cover_back_corner(e)] = length
    mate = mates[halfedge_next(h)]
    if mate <= -2:
        out_nonmanifold[0] = 1
        return
    if mate < 0:
        out_twin[h] = back
        out_twin[back] = h
        return
    front_mate = cover_slot_of_halfedge(mate)
    g = front_mate // 3
    back_mate = 3 * (n_faces + g) + cover_back_corner(front_mate % 3)
    if faces[mate] == tail:
        out_twin[h] = back_mate
        out_twin[back] = front_mate
    else:
        out_twin[h] = front_mate
        out_twin[back] = back_mate


@wp.kernel
def nonmanifold_cover_keys(
    mates: wp.array[wp.int32],
    sentinel: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The sort keys that line up each edge of three or more faces' halfedges in index order:
    # ``(lowest << 32) | h`` from the ``-2 - lowest`` mate code, ``sentinel`` (past every edge's)
    # for any other halfedge; the identity payload beside them
    # (``array.sort_and_argsort_in_place``).
    h = wp.int32(wp.tid())
    mate = mates[h]
    key = sentinel
    if mate <= -2:
        key = (wp.uint64(mate_halfedge(mate)) << wp.uint64(32)) | wp.uint64(h)
    out_keys[h] = key
    out_order[h] = h


@wp.func
def same_cover_edge(keys: wp.array[wp.uint64], p: wp.int32, q: wp.int32) -> wp.bool:
    # Whether sorted positions ``p`` and ``q`` hold halfedges of one edge
    # (``nonmanifold_cover_keys``).
    return (keys[p] >> wp.uint64(32)) == (keys[q] >> wp.uint64(32))


@wp.func
def cover_copy_of_halfedge(
    faces: wp.array[wp.int32], halfedge: wp.int32, forward: wp.bool
) -> wp.int32:
    # The cover slot of mesh halfedge ``halfedge``'s copy that runs its edge in the direction
    # ``forward`` names -- the halfedge's own direction (the front copy) or against it (the back).
    n_faces = faces.shape[0] // 3
    front = cover_slot_of_halfedge(halfedge)
    if forward:
        return front
    return 3 * (n_faces + halfedge // 3) + cover_back_corner(front % 3)


@wp.kernel
def glue_nonmanifold_cover_edges(
    faces: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    sentinel: wp.uint64,
    out_twin: wp.array[wp.int32],
) -> None:
    # ``buildIntrinsicTuftedCover``'s chain around an edge of three or more faces, one sorted
    # position per thread. geometry-central lists an edge's halfedges from its lowest one ``h0``
    # along ``sibling``, each halfedge's predecessor in index order (the lowest's is the highest),
    # and glues each face's copy running the edge against ``h0`` to the next face's copy running
    # it along ``h0``: halfedge ``h``'s backward copy to its sibling's forward copy. Every slot is
    # written once, by the halfedge whose copy it is or by the one whose sibling owns it. For one
    # or two faces the rule is ``tufted_cover``'s gluing.
    p = wp.int32(wp.tid())
    if keys[p] == sentinel:
        return
    start = p
    while start > 0 and same_cover_edge(keys, start - 1, p):
        start -= 1
    sibling = p - 1
    if p == start:
        sibling = p
        while sibling + 1 < keys.shape[0] and same_cover_edge(keys, sibling + 1, p):
            sibling += 1
    h = order[p]
    s = order[sibling]
    tail = faces[order[start]]
    backward = cover_copy_of_halfedge(faces, h, faces[h] != tail)
    forward = cover_copy_of_halfedge(faces, s, faces[s] == tail)
    out_twin[backward] = forward
    out_twin[forward] = backward


@wp.kernel
def length_sum_and_violation(
    lengths: wp.array2d[wp.float64],
    stride: wp.int32,
    out_sum_and_count: wp.array[wp.float64],
    out_violation: wp.array[wp.float64],
) -> None:
    # The sum and count of a length table's entries (``store_sum_and_count``, a fixed-order fold)
    # and its largest triangle-inequality violation ``c - a - b`` over every face and side, into
    # ``out_violation[0]`` (seeded ``-inf``; a maximum is exact in any order, so one
    # ``atomic_max`` per block commits it). Over a cover face table every cover edge has two
    # slots, so the mean entry is the mean cover edge: geometry-central's ``meanEdgeLength``, the
    # mollification's scale and the timestep's.
    block, lane = wp.tid()
    base, remaining = block_chunk_1d(lengths.shape[0], block)
    total = wp.float64(0.0)
    count = wp.float64(0.0)
    worst = wp.float64(-wp.inf)
    for k in range(lane, remaining, wp.block_dim()):
        row = base + k
        a = lengths[row, 0]
        b = lengths[row, 1]
        c = lengths[row, 2]
        total += a + b + c
        count += wp.float64(3.0)
        worst = wp.max(worst, wp.max(a - b - c, wp.max(b - c - a, c - a - b)))
    store_sum_and_count(block, lane, total, count, out_sum_and_count, stride)
    worst = block_max(worst)
    if lane == 0 and remaining > 0:
        wp.atomic_max(out_violation, 0, worst)


@wp.kernel
def mollify_cover_lengths(
    sum_and_count: wp.array[wp.float64],
    violation: wp.array[wp.float64],
    epsilon: wp.float64,
    lengths: wp.array[wp.float64],
) -> None:
    # ``mollifyIntrinsic(mesh, lengths, epsilon)``: every length grows by the smallest constant
    # that leaves each triangle inequality a margin of ``epsilon`` times the mean length, ``0``
    # where every one already has it -- read from ``length_sum_and_violation``'s sums on the
    # device. Over the flattened table, in place.
    i = wp.int32(wp.tid())
    delta = epsilon * sum_and_count[0] / sum_and_count[1]
    lengths[i] = lengths[i] + wp.max(wp.float64(0.0), violation[0] + delta)


@wp.func
def heron_area_float64(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:
    # ``geometrycentral::triangleArea``: Heron's formula, a negative radicand read as zero.
    s = (a + b + c) / wp.float64(2.0)
    return wp.sqrt(wp.max(wp.float64(0.0), s * (s - a) * (s - b) * (s - c)))


@wp.func
def cover_half_cotangent(
    lengths: wp.array2d[wp.float64], f: wp.int32, e: wp.int32, area: wp.float64
) -> wp.float64:
    # Half the cotangent of cover face ``f``'s angle at corner ``e`` from its lengths alone,
    # ``(b**2 + c**2 - a**2) / (8 A)``: geometry-central's ``halfedgeCotanWeights`` (and the flip
    # test's ``halfedgeCotanWeight``) for the slot opposite that corner.
    a = lengths[f, e]
    b = lengths[f, (e + 1) % 3]
    c = lengths[f, (e + 2) % 3]
    return (b * b + c * c - a * a) / (wp.float64(8.0) * area)


@wp.func
def cover_slot_half_cotangent(lengths: wp.array2d[wp.float64], h: wp.int32) -> wp.float64:
    # ``cover_half_cotangent`` of slot ``h``, the area formed for it.
    f = h // 3
    area = heron_area_float64(lengths[f, 0], lengths[f, 1], lengths[f, 2])
    return cover_half_cotangent(lengths, f, h % 3, area)


@wp.func
def unfolded_diagonal(
    ab: wp.float64, bc: wp.float64, ca: wp.float64, bd: wp.float64, dc: wp.float64
) -> wp.float64:
    # The other diagonal ``|ad|`` of the quad of triangles ``abc`` and ``dcb`` unfolded about
    # their shared edge ``bc``: ``b`` at the origin, ``c`` on the x axis, ``a`` above and ``d``
    # below (``layoutTriangleVertex``'s construction, as geometry-central's Euclidean flip
    # measures it).
    ax = (ab * ab + bc * bc - ca * ca) / (wp.float64(2.0) * bc)
    ay = wp.sqrt(wp.max(wp.float64(0.0), ab * ab - ax * ax))
    dx = (bd * bd + bc * bc - dc * dc) / (wp.float64(2.0) * bc)
    dy = -wp.sqrt(wp.max(wp.float64(0.0), bd * bd - dx * dx))
    return wp.length(wp.vec2d(ax - dx, ay - dy))


@wp.kernel
def cover_flip_candidates(
    lengths: wp.array2d[wp.float64],
    twin: wp.array[wp.int32],
    delaunay_epsilon: wp.float64,
    out_flip: wp.array[wp.int32],
    out_new_length: wp.array[wp.float64],
    out_face_claim: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_flipped_face: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # One cover slot per thread: whether its edge fails ``flipToDelaunay``'s Euclidean test -- its
    # cotangent weight, the two half-cotangents facing it, below ``-delaunay_epsilon`` -- the
    # length of the diagonal it would flip to, and a claim on both of its faces for the
    # independent set ``cover_flip_commit`` takes (``atomic_min`` of the slot, so the lowest
    # candidate on a face owns it and the lowest candidate overall always flips). The edge is
    # tested once, by its lower slot. A flip never changes another edge's length, and the
    # intrinsic Delaunay triangulation does not depend on the order the edges are flipped in
    # (random serial orders reached potpourri3d's field to 1e-11 on the test fixtures), so rounds
    # of independent sets reach geometry-central's serial queue.
    #
    # Also clears what the commit writes for this slot and its face, and the round's flip count,
    # before the commit reads them.
    h = wp.int32(wp.tid())
    out_flip[h] = 0
    out_remap[h] = -1
    if h % 3 == 0:
        out_flipped_face[h // 3] = 0
    if h == 0:
        out_count[0] = 0
    t = twin[h]
    if t <= h:
        return
    f0 = h // 3
    f1 = t // 3
    if f0 == f1:
        return  # two edges of one face glued together (``SurfaceMesh::flip``'s degree-1 vertex)
    weight = cover_slot_half_cotangent(lengths, h) + cover_slot_half_cotangent(lengths, t)
    if not (weight < -delaunay_epsilon):
        return
    e0 = h % 3
    e1 = t % 3
    # Face ``f0`` is ``(a, b, c)`` from corner ``e0`` and its slot ``h`` runs ``b -> c``; face
    # ``f1`` is ``(d, c, b)`` from corner ``e1``.
    new_length = unfolded_diagonal(
        lengths[f0, (e0 + 2) % 3],
        lengths[f0, e0],
        lengths[f0, (e0 + 1) % 3],
        lengths[f1, (e1 + 1) % 3],
        lengths[f1, (e1 + 2) % 3],
    )
    if not wp.isfinite(new_length):
        return
    out_flip[h] = 1
    out_new_length[h] = new_length
    # The lowest candidate claims each face, stored as ``INT32_MAX - h`` under ``atomic_max`` so a
    # claim table arriving zeroed means "unclaimed" and is cut from a zeroed allocation unfilled.
    wp.atomic_max(out_face_claim, f0, INT32_MAX_CONSTANT - h)
    wp.atomic_max(out_face_claim, f1, INT32_MAX_CONSTANT - h)


@wp.func
def move_cover_slot(
    twin: wp.array[wp.int32],
    old: wp.int32,
    new: wp.int32,
    out_pending: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
):
    # Record that the halfedge in slot ``old`` now sits in slot ``new``: its twin (still an old
    # slot, which ``cover_flip_fixup`` remaps) and the move itself.
    out_pending[new] = twin[old]
    out_remap[old] = new


@wp.kernel
def cover_flip_commit(
    flip: wp.array[wp.int32],
    new_length: wp.array[wp.float64],
    face_claim: wp.array[wp.int32],
    twin: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    lengths: wp.array2d[wp.float64],
    out_pending: wp.array[wp.int32],
    out_remap: wp.array[wp.int32],
    out_flipped_face: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # Flip slot ``h``'s edge when it owns both of its faces. ``(a, b, c)`` and ``(d, c, b)``
    # become ``(a, b, d)`` and ``(d, c, a)`` in place, written from corner 0, and every slot that
    # moved records its old twin and its move for ``cover_flip_fixup``; the new diagonal's two
    # slots are each other's twins already (``-2 - slot`` marks a final twin). Only this thread
    # touches the two faces, and ``twin`` is read only, so the commits of one round never race.
    h = wp.int32(wp.tid())
    if flip[h] == 0:
        return
    t = twin[h]
    f0 = h // 3
    f1 = t // 3
    claim = INT32_MAX_CONSTANT - h
    if face_claim[f0] != claim or face_claim[f1] != claim:
        return
    e0 = h % 3
    e1 = t % 3
    a = faces[h]
    b = faces[halfedge_next(h)]
    c = faces[halfedge_prev(h)]
    d = faces[t]
    ca = lengths[f0, (e0 + 1) % 3]
    ab = lengths[f0, (e0 + 2) % 3]
    bd = lengths[f1, (e1 + 1) % 3]
    dc = lengths[f1, (e1 + 2) % 3]
    ad = new_length[h]
    # Old slots: ``c -> a`` opposite ``b`` and ``a -> b`` opposite ``c`` in ``f0``; ``b -> d``
    # opposite ``c`` and ``d -> c`` opposite ``b`` in ``f1``.
    move_cover_slot(twin, halfedge_next(t), 3 * f0, out_pending, out_remap)
    move_cover_slot(twin, halfedge_prev(h), 3 * f0 + 2, out_pending, out_remap)
    move_cover_slot(twin, halfedge_next(h), 3 * f1, out_pending, out_remap)
    move_cover_slot(twin, halfedge_prev(t), 3 * f1 + 2, out_pending, out_remap)
    out_pending[3 * f0 + 1] = -2 - (3 * f1 + 1)
    out_pending[3 * f1 + 1] = -2 - (3 * f0 + 1)
    faces[3 * f0] = a
    faces[3 * f0 + 1] = b
    faces[3 * f0 + 2] = d
    faces[3 * f1] = d
    faces[3 * f1 + 1] = c
    faces[3 * f1 + 2] = a
    lengths[f0, 0] = bd
    lengths[f0, 1] = ad
    lengths[f0, 2] = ab
    lengths[f1, 0] = ca
    lengths[f1, 1] = ad
    lengths[f1, 2] = dc
    out_flipped_face[f0] = 1
    out_flipped_face[f1] = 1
    append_slot(out_count, 0)


@wp.kernel
def cover_flip_fixup(
    twin: wp.array[wp.int32],
    pending: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    flipped_face: wp.array[wp.int32],
    out_twin: wp.array[wp.int32],
    out_face_claim: wp.array[wp.int32],
) -> None:
    # Every slot's twin after the round, into the other twin buffer: a slot of a flipped face
    # takes the twin its halfedge carried in (``pending``), any other keeps its own, and either is
    # moved to where that halfedge now sits (``remap``); a new diagonal's twin is final. Also
    # re-arms the face claims for the next round.
    h = wp.int32(wp.tid())
    f = h // 3
    other = twin[h]
    if flipped_face[f] != 0:
        other = pending[h]
    if other <= -2:
        other = -2 - other
    elif remap[other] >= 0:
        other = remap[other]
    out_twin[h] = other
    if h % 3 == 0:
        out_face_claim[f] = wp.int32(0)


@wp.kernel
def cover_face_terms(
    faces: wp.array[wp.int32],
    lengths: wp.array2d[wp.float64],
    out_cot: wp.array2d[wp.float64],
    out_area_sums: wp.array[wp.float64],
    out_corner_areas: wp.array2d[wp.float64],
) -> None:
    # ``heat_face_terms`` on the cover, from its lengths: each face's three half-cotangents into
    # ``out_cot`` (``heat_system_rows``' table) and its area to each corner for the lumped mass
    # (``vertexLumpedMassMatrix``, a third of each face's area per corner once the rows divide;
    # a face's repeated vertex takes two thirds), atomically or, under the deterministic mode,
    # stored per corner (``add_or_store_corner_triple``).
    f = wp.int32(wp.tid())
    area = heron_area_float64(lengths[f, 0], lengths[f, 1], lengths[f, 2])
    out_cot[f, 0] = cover_half_cotangent(lengths, f, 0, area)
    out_cot[f, 1] = cover_half_cotangent(lengths, f, 1, area)
    out_cot[f, 2] = cover_half_cotangent(lengths, f, 2, area)
    add_or_store_corner_triple(out_area_sums, out_corner_areas, faces, f, area, area, area)


@wp.kernel
def cover_unit_gradient_divergence(
    faces: wp.array[wp.int32],
    lengths: wp.array2d[wp.float64],
    cot: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    out_div: wp.array[wp.float64],
    out_corner_div: wp.array2d[wp.float64],
) -> None:
    # ``unit_gradient_divergence`` on a cover face, from its lengths: the face laid out in the
    # plane (``halfedgeVectorsInFace``), the gradient's direction there (geometry-central's
    # ``gradUDir``, the edges' perpendiculars weighted by the field, normalized as
    # ``stable_normalize`` does -- where its squared length underflows, geometry-central's
    # ``normalizeCutoff`` loses the direction and this keeps it), and each slot's term
    # ``cot * dot(edge, x)`` added to its tail and taken from its tip. Positive divergence, the
    # right-hand side of ``L phi = div x`` with geometry-central's positive-semi-definite ``L``
    # (the Poisson operator here); the sign is the field's direction toward the sources, which
    # the shift that follows orients either way.
    f = wp.int32(wp.tid())
    l0 = lengths[f, 0]
    l1 = lengths[f, 1]
    l2 = lengths[f, 2]
    p0 = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    p1 = wp.vec2d(l2, wp.float64(0.0))
    x2 = (l2 * l2 + l1 * l1 - l0 * l0) / (wp.float64(2.0) * l2)
    p2 = wp.vec2d(x2, wp.sqrt(wp.max(wp.float64(0.0), l1 * l1 - x2 * x2)))
    e0 = p2 - p1
    e1 = p0 - p2
    e2 = p1 - p0
    u0, u1, u2 = face_vertices(values, faces, f)
    g = wp.vec2d(-e0[1], e0[0]) * u0 + wp.vec2d(-e1[1], e1[0]) * u1 + wp.vec2d(-e2[1], e2[0]) * u2
    x = stable_normalize(g)
    t0 = cot[f, 0] * wp.dot(e0, x)
    t1 = cot[f, 1] * wp.dot(e1, x)
    t2 = cot[f, 2] * wp.dot(e2, x)
    add_or_store_corner_triple(out_div, out_corner_div, faces, f, t2 - t1, t0 - t2, t1 - t0)


# --------------------------------------------------------------------------------------
# Fast marching: the label-correcting front, relaxed in parallel rounds
# --------------------------------------------------------------------------------------


@wp.kernel
def seed_front_sources(
    sources: wp.array[wp.int32], out_labels: wp.array[wp.float32], out_invalid: wp.array[wp.int32]
) -> None:
    # Distance zero at each source (``out_labels`` filled with ``+inf``, "unreached", by the
    # caller); an out-of-range source writes nothing and raises ``out_invalid[0]``.
    t = wp.int32(wp.tid())
    v = sources[t]
    if flag_out_of_range(v, out_labels.shape[0], out_invalid):
        return
    out_labels[v] = wp.float32(0.0)


@wp.func
def front_past(origin: wp.float32, step: wp.float32) -> wp.float32:
    # ``origin + step``, kept strictly above ``origin``: a label a vertex hands on is never its own,
    # even across a zero-length edge or a step below ``origin``'s rounding.
    reached = origin + step
    if reached <= origin:
        reached = next_up(origin)
    return reached


@wp.func
def front_edge_step(pa: wp.vec3, pc: wp.vec3) -> wp.float32:
    # The edge's length as the front prices it: the norm of ``pc - pa`` in ``float32``.
    return wp.sqrt(wp.length_sq(pc - pa))


@wp.func
def front_triangle_step(b: wp.vec3, c: wp.vec3, vb: wp.float32) -> wp.float32:
    # The planar front's advance from corner ``a`` (the origin, label ``va``) to corner ``c`` of a
    # triangle whose third corner ``b`` is ``vb >= 0`` further from the sources, or ``-1`` (the
    # sentinel) where that front does not reach ``c`` through the triangle's interior. ``b`` and
    # ``c`` are relative to ``a``.
    #
    # The front is a plane wave whose unit direction ``n`` makes ``cos^2 = vb^2 / |b|^2`` with
    # ``b``; it reaches ``c`` at ``|c| cos(angle(n, c))`` past ``a``, with the angle composed from
    # ``b0c`` and ``(n, b)`` (``cos(x - y)``). Every rejection tests the squared cosines, never an
    # angle: ``n`` must leave ``a`` between ``b`` and ``c`` (not past the side ``ac``), its gradient
    # must fit in the edge ``ab`` (``|b| > vb``), and the wave must cross ``bc`` (not run past ``b``
    # along that side).
    dot_bc = wp.dot(b, c)
    if dot_bc <= wp.float32(0.0):
        return wp.float32(-1.0)
    b_sq = wp.length_sq(b)
    vb_sq = vb * vb
    if b_sq <= vb_sq:
        return wp.float32(-1.0)
    cos_n_sq = vb_sq / b_sq
    c_sq = wp.length_sq(c)
    bc_sq = b_sq * c_sq
    if bc_sq <= wp.float32(0.0):
        return wp.float32(-1.0)
    cos_b0c_sq = dot_bc * dot_bc / bc_sq
    if cos_b0c_sq <= cos_n_sq:
        return wp.float32(-1.0)
    a = c - b
    dot_ba = wp.dot(b, a)
    if dot_ba >= wp.float32(0.0):
        if dot_ba * dot_ba >= cos_n_sq * b_sq * wp.length_sq(a):
            return wp.float32(-1.0)
    cos_b0c_sq = wp.min(cos_b0c_sq, wp.float32(1.0))
    return wp.sqrt(c_sq) * (
        wp.sqrt(cos_b0c_sq * cos_n_sq)
        + wp.sqrt((wp.float32(1.0) - cos_b0c_sq) * (wp.float32(1.0) - cos_n_sq))
    )


@wp.func
def front_corner_label(
    pa: wp.vec3, pb: wp.vec3, pc: wp.vec3, va: wp.float32, vb: wp.float32, vc: wp.float32
) -> wp.float32:
    # The smallest label the front offers corner ``c`` of one face from its other corners ``a`` and
    # ``b`` (in the face's order after ``c``), never above ``vc``: along each edge, and across the
    # face from the nearer of the two once both are reached. On a tie ``a`` is the origin.
    best = vc
    if va < wp.inf:
        best = wp.min(best, front_past(va, front_edge_step(pa, pc)))
    if vb < wp.inf:
        best = wp.min(best, front_past(vb, front_edge_step(pb, pc)))
        if va < wp.inf:
            if vb < va:
                step = front_triangle_step(pa - pb, pc - pb, va - vb)
                if step >= wp.float32(0.0):
                    best = wp.min(best, front_past(vb, step))
            else:
                step = front_triangle_step(pb - pa, pc - pa, vb - va)
                if step >= wp.float32(0.0):
                    best = wp.min(best, front_past(va, step))
    return best


@wp.func
def lower_front_label(
    labels: wp.array[wp.float32],
    v: wp.int32,
    offer: wp.float32,
    out_labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # Commit an offer below the round's label of ``v``: an ``atomic_min``, exact and independent of
    # arrival order, and the round's progress flag (a plain store; every writer stores ``1``).
    # ``out_labels[v]`` is first lowered to the round's own label, which it may still be above.
    label = labels[v]
    if label < out_labels[v]:
        wp.atomic_min(out_labels, v, label)
    if offer < label:
        wp.atomic_min(out_labels, v, offer)
        out_state[LOOP_PROGRESS] = 1


@wp.func
def fast_marching_face(
    f: wp.int32,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    labels: wp.array[wp.float32],
    out_labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # One thread of ``fast_marching_pass``: face ``f``'s offers to its three corners.
    i0, i1, i2 = corner_triple(faces, f)
    p0 = vertices[i0]
    p1 = vertices[i1]
    p2 = vertices[i2]
    v0 = labels[i0]
    v1 = labels[i1]
    v2 = labels[i2]
    lower_front_label(labels, i0, front_corner_label(p1, p2, p0, v1, v2, v0), out_labels, out_state)
    lower_front_label(labels, i1, front_corner_label(p2, p0, p1, v2, v0, v1), out_labels, out_state)
    lower_front_label(labels, i2, front_corner_label(p0, p1, p2, v0, v1, v2), out_labels, out_state)


@wp.kernel
def fast_marching_pass(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    labels: wp.array[wp.float32],
    out_labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # One face's offers to its three corners from the round's ``labels``, lowered into
    # ``out_labels``, which holds the round before ``labels`` on entry (the wrapper alternates
    # the two buffers): labels only fall, so lowering each corner to ``labels`` first
    # (``lower_front_label``) makes the result ``labels`` lowered by the offers, a vertex on no
    # face never changing. Every offer reads the previous round only, so a round is a pure
    # function of ``labels``, the same on both devices and under any interleaving.
    #
    # The serial label-correcting front pops the nearest vertex and offers its neighbours the
    # same two updates, the triangle one only where the edge one did not lower the neighbour;
    # run uncapped, its answer is this fixpoint (the triangle offer it skips is never below the
    # edge offer it made instead). A serial front that caps the times a vertex is re-opened
    # answers differently, by the order its queue happened to pop in.
    #
    # Declined (2026-10-30): choosing the direction here by the round's parity, so a captured
    # body is one pass rather than two: 0.97-1.07x the copy-back it replaced on CUDA
    # (``icosphere(4)`` to ``(7)``), where two passes a body ran 0.75-0.82x.
    fast_marching_face(wp.int32(wp.tid()), vertices, faces, labels, out_labels, out_state)


@wp.kernel
def fast_marching_rounds_serial(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_rounds: wp.int32,
    labels: wp.array[wp.float32],
    relaxed: wp.array[wp.float32],
    state: wp.array[wp.int32],
) -> None:
    # dim=1, the CPU device's round loop: each round's two ``fast_marching_pass`` launches and its
    # ``loop_advance``, as one thread's loops in the order a device running each launch's threads
    # one after another runs them, until the state's condition stops the loop -- so ``labels``
    # (the answer) and ``relaxed`` come out as the launched loop leaves them.
    n_faces = faces.shape[0] // 3
    while state[LOOP_CONDITION] != 0:
        for f in range(n_faces):
            fast_marching_face(f, vertices, faces, labels, relaxed, state)
        for f in range(n_faces):
            fast_marching_face(f, vertices, faces, relaxed, labels, state)
        advance_round_loop(max_rounds, state)


# The unfolding front (``method="meshlab"``) runs in ``float64``, as the reference does: its tests
# are acos/Heron compositions whose ``float32`` forms drift a few 1e-6 of the range. Each vertex
# also carries the source whose front reached it, since an offer unfolds across a face only when its
# two corners came from the same source; with one source every reached vertex has it, and no labels
# are kept.

# The reference's relative tolerance on its triangle-inequality tests, the ``float`` literal
# ``.0001f`` widened to ``double``.
UNFOLDING_TOLERANCE = wp.float64(9.999999747378752e-05)
# How far above a target's distance a ``float32`` unfolded offer must lie to be dropped unrefined,
# relative to the distances and edges it is formed from (``unfolded_distance_exceeds``).
UNFOLDING_FILTER_MARGIN = wp.float64(1e-3)


@wp.func
def heron_half_cosine(numerator: wp.Float, denominator: wp.Float) -> wp.Float:
    # ``min(1, sqrt(numerator / denominator))`` with C's ``std::min`` on a NaN: a negative or
    # undefined ratio yields ``1``. The cosine of half the virtual triangle's angle, which the
    # reference turns into that angle as ``2 acos`` of it.
    ratio = numerator / denominator
    q = type(ratio)(1.0)
    if ratio >= type(ratio)(0.0) and ratio < type(ratio)(1.0):
        q = wp.sqrt(ratio)
    return q


@wp.func
def virtual_triangle_ratios(
    d_origin: wp.Float, d_partner: wp.Float, ec_w1: wp.Float
) -> tuple[wp.Float, wp.Float]:
    # The virtual triangle's ``a = s / ec_w1`` and ``b = a s``, ``s`` its semi-perimeter, from
    # which both half-angles' Heron ratios ``(b - a d_partner) / d_origin`` and
    # ``(b - a d_origin) / d_partner`` are formed: the preamble ``unfolded_distance`` and
    # ``vcg_unfolded_distance`` share, in the reference's operation order.
    s = (d_origin + d_partner + ec_w1) / type(ec_w1)(2.0)
    a = s / ec_w1
    return a, a * s


@wp.func
def unfolded_distance(
    ew_c: wp.Float,
    ew_w1: wp.Float,
    ec_w1: wp.Float,
    cos_alpha: wp.Float,
    cos_beta: wp.Float,
    d_partner: wp.Float,
    d_origin: wp.Float,
) -> wp.Float:
    # The distance ``target`` is offered across its face from ``origin`` once ``partner`` (the
    # face's third corner) is reached too: the virtual source both distances place on the far side
    # of the edge ``origin partner``, unfolded into the face's plane, and the straight line from it.
    # ``ew_c``, ``ew_w1`` and ``ec_w1`` are the edges target-origin, target-partner and
    # origin-partner, ``cos_alpha`` and ``cos_beta`` the face's interior angle cosines at origin and
    # partner. Where the line would leave the face past ``origin`` (its angle there plus the virtual
    # triangle's exceeds pi) the offer is the edge from ``origin``; past ``partner``, the edge from
    # ``partner``. An angle whose cosine leaves ``[-1, 1]`` (NaN on a zero-length edge) is NaN in
    # the reference, which fails its test and, when it is the angle at ``origin`` the final line
    # needs, drops the offer: ``+inf`` here.
    #
    # The reference forms each angle with ``acos`` and the line with ``cos`` and ``sin`` of
    # ``pi`` minus their sum; here every angle stays a cosine, since in ``float64`` the
    # transcendentals were most of the pass's time. Both angles lie in ``[0, pi]``, so
    # ``alpha + 2 acos(q) > pi`` is ``cos(alpha) < 1 - 2 q^2`` (cosine falls on ``[0, pi]``), and
    # the line needs only the sum's cosine and sine, composed from ``cos(2 acos(q)) = 2 q^2 - 1``
    # and ``sin = 2 q sqrt(1 - q^2)``. The two forms agree to rounding, and a rounding flip of
    # either test cannot move the offer: at the boundary the line passes through that corner, so
    # both branches give its edge.
    one = type(ew_c)(1.0)
    two = type(ew_c)(2.0)
    alpha_defined = cos_alpha >= -one and cos_alpha <= one
    a, b = virtual_triangle_ratios(d_origin, d_partner, ec_w1)
    q_alpha = heron_half_cosine(b - a * d_partner, d_origin)
    if alpha_defined and cos_alpha < one - two * q_alpha * q_alpha:
        return d_origin + ew_c

    q_beta = heron_half_cosine(b - a * d_origin, d_partner)
    if cos_beta >= -one and cos_beta <= one:
        if cos_beta < one - two * q_beta * q_beta:
            return d_partner + ew_w1
    if not alpha_defined:
        return type(ew_c)(wp.inf)
    # cos and sin of ``theta = pi - (alpha + alpha_)``.
    sin_alpha = wp.sqrt(one - cos_alpha * cos_alpha)
    cos_alpha_ = two * q_alpha * q_alpha - one
    sin_alpha_ = two * q_alpha * wp.sqrt(one - q_alpha * q_alpha)
    cos_theta = sin_alpha * sin_alpha_ - cos_alpha * cos_alpha_
    sin_theta = sin_alpha * cos_alpha_ + cos_alpha * sin_alpha_
    delta = cos_theta * ew_c
    h = sin_theta * ew_c
    return wp.sqrt(h * h + (d_origin + delta) * (d_origin + delta))


@wp.func
def unfolded_distance_exceeds(
    ew_c: wp.float64,
    ew_w1: wp.float64,
    ec_w1: wp.float64,
    cos_alpha: wp.float64,
    cos_beta: wp.float64,
    d_partner: wp.float64,
    d_origin: wp.float64,
    d_target: wp.float64,
) -> wp.bool:
    # Whether the unfolded offer lies clearly above ``d_target``, judged in ``float32``: such an
    # offer neither lowers the target (that takes an offer below it) nor labels it (an offer
    # equal to it), so it can be dropped without its ``float64`` value. ``float32`` is rounded
    # once per input and the composition loses at most a few hundred ulps where the half-angle's
    # sine is ill-conditioned, far inside the margin; a non-finite ``float32`` value proves
    # nothing (a cosine rounded past 1 there may be valid in ``float64``).
    v = unfolded_distance(
        wp.float32(ew_c),
        wp.float32(ew_w1),
        wp.float32(ec_w1),
        wp.float32(cos_alpha),
        wp.float32(cos_beta),
        wp.float32(d_partner),
        wp.float32(d_origin),
    )
    if not wp.isfinite(v):
        return False
    margin = UNFOLDING_FILTER_MARGIN * (d_target + ew_c + ew_w1 + ec_w1)
    return wp.float64(v) > d_target + margin


@wp.func
def vcg_norm(d: wp.vec3d) -> wp.float64:
    # ``vcg::Point3::Norm``: the square root of ``x x + y y + z z``, summed in that order.
    return wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


@wp.func
def vcg_scaled_direction(d: wp.vec3d, length: wp.float64) -> wp.vec3d:
    # ``d.Normalize() * length``, as ``GeoDistance`` forms its edge vectors: each component divided
    # by the norm (a zero vector left as it is), then multiplied by ``length``.
    n = vcg_norm(d)
    if n > wp.float64(0.0):
        d = wp.vec3d(d[0] / n, d[1] / n, d[2] / n)
    return wp.vec3d(d[0] * length, d[1] * length, d[2] * length)


@wp.func
def vcg_dot(a: wp.vec3d, b: wp.vec3d) -> wp.float64:
    # ``vcg::Point3``'s dot product, summed in component order.
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.func
def vcg_acos(x: wp.float64) -> wp.float64:
    # C's ``acos``: NaN outside ``[-1, 1]``, where Warp's clamps (and propagates NaN as NaN).
    if x >= wp.float64(-1.0) and x <= wp.float64(1.0):
        return wp.acos(x)
    return wp.float64(wp.nan)


@wp.func
def vcg_half_angle(numerator: wp.float64, denominator: wp.float64) -> wp.float64:
    # ``2 acos(std::min(1.0, sqrt(numerator / denominator)))``: ``std::min`` returns ``1`` for a
    # NaN square root, which ``heron_half_cosine`` reproduces (it takes the root only of a ratio in
    # ``[0, 1)``, whose root is below ``1``, so the two guards pick the same value).
    return wp.float64(2.0) * wp.acos(heron_half_cosine(numerator, denominator))


@wp.func
def vcg_corner_angle(
    p: wp.vec3d, q: wp.vec3d, r: wp.vec3d, l_pq: wp.float64, l_pr: wp.float64
) -> wp.float64:
    # A face's interior angle at ``p`` as ``GeoDistance`` forms it, for ``alpha`` at the origin
    # and ``beta`` at the partner: the edges from ``p`` normalized and rescaled to their lengths
    # ``l_pq`` and ``l_pr`` (vcg's ``Distance``), then ``acos`` of their dot product over the
    # lengths' product. Both are symmetric in ``q`` and ``r`` bit for bit, and ``beta``'s
    # ``-w1_c`` is the rescaled edge from the partner exactly, so the angle depends on the corner
    # alone and is formed once per face.
    w_q = vcg_scaled_direction(q - p, l_pq)
    w_r = vcg_scaled_direction(r - p, l_pr)
    return vcg_acos(vcg_dot(w_q, w_r) / (l_pq * l_pr))


@wp.func
def vcg_unfolded_distance(
    alpha: wp.float64,
    beta: wp.float64,
    ew_c: wp.float64,
    ew_w1: wp.float64,
    ec_w1: wp.float64,
    d_partner: wp.float64,
    d_origin: wp.float64,
) -> wp.float64:
    # ``vcg::tri::Geodesic::GeoDistance`` operation for operation, from the face's angles at the
    # origin and the partner (``vcg_corner_angle``; edges as ``unfolded_distance`` names them), so
    # that it rounds as the reference does: which of two near-equal offers lowers a vertex decides
    # its label, and with it the unfolding downstream. ``unfolded_distance`` is the same line in
    # cosines, cheaper and equal to rounding; it remains the ``float32`` filter's.
    a, b = virtual_triangle_ratios(d_origin, d_partner, ec_w1)
    alpha_ = vcg_half_angle(b - a * d_partner, d_origin)
    if alpha + alpha_ > wp.float64(math.pi):
        return d_origin + ew_c
    beta_ = vcg_half_angle(b - a * d_origin, d_partner)
    if beta + beta_ > wp.float64(math.pi):
        return d_partner + ew_w1
    theta = wp.float64(math.pi) - alpha - alpha_
    delta = wp.cos(theta) * ew_c
    h = wp.sin(theta) * ew_c
    return wp.sqrt(h * h + (d_origin + delta) * (d_origin + delta))


@wp.func
def unfolding_offer(
    ew_c: wp.float64,
    ew_w1: wp.float64,
    inter: wp.float64,
    cos_alpha: wp.float64,
    cos_beta: wp.float64,
    alpha: wp.float64,
    beta: wp.float64,
    d_origin: wp.float64,
    d_partner: wp.float64,
    d_target: wp.float64,
    same_source: wp.bool,
) -> wp.float64:
    # What a reached ``origin`` offers ``target`` through one face whose third corner is
    # ``partner`` (edges and cosines as in ``unfolded_distance``; ``inter`` is origin-partner):
    # the unfolded line where ``partner`` is reached from the same source and the three distances
    # ``d_origin``, ``d_partner`` and ``inter`` form a triangle by more than the relative
    # tolerance, else the edge ``origin target``. An unreached partner (``+inf``) always fails the
    # triangle test. These tests choose between two different offers, so they read the edge
    # length and the distances exactly as the reference does.
    tol = (inter + d_origin + d_partner) * UNFOLDING_TOLERANCE
    if (
        not same_source
        or inter + d_origin < d_partner + tol
        or inter + d_partner < d_origin + tol
        or d_origin + d_partner < inter + tol
    ):
        return d_origin + ew_c
    if unfolded_distance_exceeds(
        ew_c, ew_w1, inter, cos_alpha, cos_beta, d_partner, d_origin, d_target
    ):
        return wp.float64(wp.inf)
    return vcg_unfolded_distance(alpha, beta, ew_c, ew_w1, inter, d_partner, d_origin)


@wp.func
def unfolding_corner_offer(
    l_ca: wp.float64,
    l_cb: wp.float64,
    l_ab: wp.float64,
    cos_a: wp.float64,
    cos_b: wp.float64,
    da: wp.float64,
    db: wp.float64,
    dc: wp.float64,
    alpha: wp.float64,
    beta: wp.float64,
    sa: wp.int32,
    sb: wp.int32,
) -> wp.float64:
    # The offer corner ``c`` of one face takes from its corner ``a`` as the origin, with ``b`` as
    # the partner (``+inf`` while ``a`` is unreached). ``l_ca``, ``l_cb`` and ``l_ab`` are the
    # face's edges, ``cos_a`` and ``cos_b`` its interior angle cosines, ``alpha`` and ``beta`` the
    # same angles as the reference forms them (``vcg_corner_angle``).
    if da < wp.inf:
        return unfolding_offer(l_ca, l_cb, l_ab, cos_a, cos_b, alpha, beta, da, db, dc, sa == sb)
    return wp.float64(wp.inf)


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def unfolding_face_geometry(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_lengths: wp.array[wp.vec3d],
    out_cosines: wp.array[wp.vec3d],
    out_angles: wp.array[wp.vec3d],
) -> None:
    # Each face's ``float64`` geometry, formed once before the rounds since the positions never
    # move: ``out_lengths[f][k]``, the edge opposite corner ``k``, and ``out_cosines[f][k]``, the
    # interior angle cosine at corner ``k`` (NaN beside a zero-length edge), and
    # ``out_angles[f][k]`` that angle as the reference forms it, which every offer through the
    # face reads. The lengths are vcg's ``Distance`` bit for bit (no contraction):
    # the offers' tests compare them with distances to a relative tolerance.
    f = wp.int32(wp.tid())
    p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
    e0 = p2 - p1
    e1 = p0 - p2
    e2 = p1 - p0
    lengths = wp.vec3d(vcg_norm(e0), vcg_norm(e1), vcg_norm(e2))
    out_lengths[f] = lengths
    out_cosines[f] = wp.vec3d(
        -wp.dot(e2, e1) / (lengths[2] * lengths[1]),
        -wp.dot(e0, e2) / (lengths[0] * lengths[2]),
        -wp.dot(e1, e0) / (lengths[1] * lengths[0]),
    )
    out_angles[f] = wp.vec3d(
        vcg_corner_angle(p0, p1, p2, lengths[2], lengths[1]),
        vcg_corner_angle(p1, p2, p0, lengths[0], lengths[2]),
        vcg_corner_angle(p2, p0, p1, lengths[1], lengths[0]),
    )


# The ordered front: vcg's serial queue, a step at a time. The queue pops its nearest pending
# vertex (one lowered since it last popped) and offers each face around it, in the order its
# vertex-face lists hold them, reading every partner's distance as it stands at that moment: a
# partner not yet popped holds a tentative distance, so the answer depends on the pop order, and
# the fixpoint of the offers alone does not fix it (rounds relaxing every face against the last
# round's distances settle elsewhere, up to 1e-4 of the range on scan meshes).
#
# A step pops every pending vertex that nothing pending in its 2-ring precedes (smaller distance,
# then smaller index) and whose distance lies within ``UNFOLDING_WINDOW_EDGES`` mean edges of the
# window floor. Two such pops are at least three rings apart, so neither writes what the other
# reads, and each sees exactly what the queue would show it. The floor is the smallest distance of
# such a 2-ring minimum at the previous step: it keeps a front arriving from far away, nearer the
# sources, from being overtaken, which the 2-ring test alone cannot see, while a vertex held behind
# a cycle (below) does not hold back the rest of the mesh. A vertex the queue has popped
# ``UNFOLDING_RELAX_POPS`` times is caught in a cycle of faces lowering each other below their own
# distances, which the serial queue walks one pop at a time (half a million pops of one vertex on
# ``happy_buddha``); it pops whenever no ordinary pending vertex in its 2-ring precedes it, reading
# the step's starting distances, so a cycle settles in steps instead of pops, at the limit the
# queue creeps to. On dragon, happy_buddha and bunny, from one source or three, no vertex is more
# than 1e-7 of the range off MeshLab (the ``float32`` rounding of the result).
#
# Each step is two launches over double-buffered lists (parity ``p``), a block of lanes per listed
# vertex: ``ordered_pop_pass`` selects and pops ``front[p]``, lowering into ``out_relaxed`` and
# listing every vertex it lowered or kept into ``touched[p]``; ``ordered_settle_pass`` copies the
# lowered distances back into ``labels`` and lists the vertices still pending into
# ``front[1 - p]``. Several sources add ``ordered_relabel_pass`` between the two.
#
# The offers round as the reference's do: the step kernels and the face geometry compile without
# floating-point contraction (``fuse_fp``), the face lengths are vcg's ``Distance`` and the
# unfolded line its ``GeoDistance`` operation for operation (``vcg_unfolded_distance``, its angles
# formed once per face). Where two sources' offers to a vertex tie to the last bits, the rounding
# decides which labels it, and the label decides the unfolding downstream: with contraction two
# ``torus_irregular`` vertices were 2.5e-4 of the range off MeshLab from three sources, and
# without it but with the line in cosines one dragon vertex 1.3e-5.
#
# Declined layouts (CUDA): the step loop as one launch of resident blocks with a grid barrier
# between phases, 1.12-1.44x on dragon and 0.89-0.97x on meshes of 500 to 36 k vertices at the
# best of 2-8 blocks per SM and 16 or 32 lanes, with no setting best on all of them: a size gate
# for a few per cent, and a spin barrier that hangs wherever the blocks are not all resident
# (``compute-sanitizer``, a debugger); one block for a small mesh, 46x on bunny (one SM's
# ``float64`` units then run every pop); 16 or 64 lanes a pop, or 256 to 4 096 blocks, within 3 %
# on bunny and dragon.

# Pops after which a vertex is taken as cycling (``ordered_pop_selected``).
UNFOLDING_RELAX_POPS = 8
# Width of the pop window, in mean edge lengths.
UNFOLDING_WINDOW_EDGES = 2.0
# ``counts`` slots of the ordered front's lists, plus the parity: the pending vertices a step pops
# from, and the vertices it lowered or kept.
ORDERED_FRONT = 0
ORDERED_TOUCHED = 2
# A source label no cycling pop has bid (``ordered_relabel_pass``).
ORDERED_NO_BID = 2147483647


@wp.kernel
def seed_ordered_front(
    sources: wp.array[wp.int32],
    out_labels: wp.array[wp.float64],
    out_relaxed: wp.array[wp.float64],
    out_source_of: wp.array[wp.int32],
    out_next_source_of: wp.array[wp.int32],
    out_listed: wp.array[wp.int32],
    out_front: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_invalid: wp.array[wp.int32],
) -> None:
    # Distance zero at each source and its own index as its label (when labels are kept), listed
    # once in ``front[0]`` however often ``sources`` names it. An out-of-range source writes
    # nothing and raises ``out_invalid[0]``.
    t = wp.int32(wp.tid())
    v = sources[t]
    if flag_out_of_range(v, out_labels.shape[0], out_invalid):
        return
    out_labels[v] = wp.float64(0.0)
    out_relaxed[v] = wp.float64(0.0)
    if out_source_of.shape[0] != 0:
        out_source_of[v] = v
        out_next_source_of[v] = v
    if wp.atomic_exch(out_listed, v, -2) != -2:
        out_front[append_slot(out_counts, ORDERED_FRONT)] = v


@wp.kernel
def ordered_corner_records(
    corner_order: wp.array[wp.int32], faces: wp.array[wp.int32], out_records: wp.array[wp.vec4i]
) -> None:
    # For each corner in sorted order, ``(face, its index in the face, the next corner's vertex,
    # the one after)``: everything a pop reads to reach a face's corners, in one load.
    #
    # The wrapper counts and scans each vertex's corners for their offsets though this sort
    # already orders them. Searching each vertex's first sorted corner in this launch on CUDA
    # (``adjacency.emit_sorted_vertex_faces``' form, the CPU keeping the count) saved a launch,
    # a memset and a scan, byte-identical, and timed 0.99-1.02x at 1 k-330 k faces: the call is
    # its front loop.
    e = wp.int32(wp.tid())
    c = corner_order[e]
    f = c // 3
    z = c - 3 * f
    out_records[e] = wp.vec4i(f, z, faces[halfedge_next(c)], faces[halfedge_prev(c)])


@wp.func
def ordered_source(source_of: wp.array[wp.int32], v: wp.int32) -> wp.int32:
    # The source label of ``v``; ``0`` for every vertex when labels are not kept (one source).
    if source_of.shape[0] == 0:
        return wp.int32(0)
    return source_of[v]


# What a pending vertex finds below it in its 2-ring (``ordered_precedence``): nothing, only cycling
# vertices, or an ordinary one. Ordered so the block's maximum is what any lane found.
ORDERED_LOWEST = 0
ORDERED_BELOW_CYCLING = 1
ORDERED_BELOW_ORDINARY = 2


@wp.func
def ordered_precedence(
    u: wp.int32,
    v: wp.int32,
    labels: wp.array[wp.float64],
    popped: wp.array[wp.float64],
    pop_count: wp.array[wp.int32],
) -> wp.int32:
    # Whether ``u`` is pending and precedes ``v`` in the queue's order (smaller distance, then
    # smaller index), and if so whether it is cycling.
    if not (labels[u] < popped[u]):
        return ORDERED_LOWEST
    if not (labels[u] < labels[v] or (labels[u] == labels[v] and u < v)):
        return ORDERED_LOWEST
    if pop_count[u] >= UNFOLDING_RELAX_POPS:
        return ORDERED_BELOW_CYCLING
    return ORDERED_BELOW_ORDINARY


@wp.func
def ordered_pop_selected(
    v: wp.int32,
    lane: wp.int32,
    window_bound: wp.float64,
    neighbors: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    labels: wp.array[wp.float64],
    popped: wp.array[wp.float64],
    pop_count: wp.array[wp.int32],
) -> tuple[wp.bool, wp.bool]:
    # Whether pending ``v`` pops this step, and whether it is an ordinary vertex nothing in its
    # 2-ring precedes (whose distance bounds the next step's window). An ordinary ``v`` pops when
    # it is such a vertex and its distance lies inside the window; a cycling one when no ordinary
    # vertex in its 2-ring precedes it. The block's lanes split the 1-ring, each scanning its
    # neighbours' rings; every lane returns the same answer.
    cycling = pop_count[v] >= UNFOLDING_RELAX_POPS
    stop = ORDERED_BELOW_CYCLING
    if cycling:
        stop = ORDERED_BELOW_ORDINARY
    below = ORDERED_LOWEST
    for e in range(neighbor_offsets[v] + lane, neighbor_offsets[v + 1], wp.block_dim()):
        u = neighbors[e]
        below = wp.max(below, ordered_precedence(u, v, labels, popped, pop_count))
        for e2 in range(neighbor_offsets[u], neighbor_offsets[u + 1]):
            if below >= stop:
                break
            w = neighbors[e2]
            if w != v:
                below = wp.max(below, ordered_precedence(w, v, labels, popped, pop_count))
        if below >= stop:
            break
    below = block_max(below)
    if cycling:
        return below != ORDERED_BELOW_ORDINARY, False
    lowest = below == ORDERED_LOWEST
    return lowest and labels[v] < window_bound, lowest


@wp.func
def list_touched(
    v: wp.int32,
    stamp: wp.int32,
    parity: wp.int32,
    out_listed: wp.array[wp.int32],
    out_touched: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # List ``v`` in ``touched[parity]``, once a step (its stamp swapped to the step's).
    if wp.atomic_exch(out_listed, v, stamp) != stamp:
        slot = append_slot(out_counts, ORDERED_TOUCHED + parity)
        out_touched[parity * out_listed.shape[0] + slot] = v


@wp.func
def ordered_offer_corners(
    v: wp.int32, j: wp.int32, corner_records: wp.array[wp.vec4i], corner_offsets: wp.array[wp.int32]
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32, wp.int32]:
    # The ``j``-th offer a pop of ``v`` makes, in vcg's visit order: through ``v``'s corners newest
    # face first, each to the corner after ``v`` with the one after that as the partner, then the
    # other way round. Returns the face, ``v``'s corner in it, the target's corner, and the target
    # and partner vertices, all read off one corner record (``ordered_corner_records``).
    record = corner_records[corner_offsets[v + 1] - 1 - j // 2]
    z = record[1]
    k = j % 2
    return record[0], z, (z + 1 + k) % 3, record[2 + k], record[3 - k]


@wp.func
def ordered_offer(
    f: wp.int32,
    z: wp.int32,
    target: wp.int32,
    lengths: wp.array[wp.vec3d],
    cosines: wp.array[wp.vec3d],
    angles: wp.array[wp.vec3d],
    d_v: wp.float64,
    s_v: wp.int32,
    d_partner: wp.float64,
    s_partner: wp.int32,
    d_target: wp.float64,
) -> wp.float64:
    # The offer through face ``f`` from ``v`` (its corner ``z``) at distance ``d_v`` and label
    # ``s_v`` to the corner ``target``, the face's third corner the partner (``unfolding_offer``).
    partner = 3 - z - target
    face_lengths = lengths[f]
    face_cosines = cosines[f]
    face_angles = angles[f]
    return unfolding_corner_offer(
        face_lengths[partner],
        face_lengths[z],
        face_lengths[target],
        face_cosines[z],
        face_cosines[partner],
        d_v,
        d_partner,
        d_target,
        face_angles[z],
        face_angles[partner],
        s_v,
        s_partner,
    )


@wp.func
def ordered_pop_block(
    block: wp.int32,
    lane: wp.int32,
    blocks: wp.int32,
    parity: wp.int32,
    round_index: wp.int32,
    lengths: wp.array[wp.vec3d],
    cosines: wp.array[wp.vec3d],
    angles: wp.array[wp.vec3d],
    corner_records: wp.array[wp.vec4i],
    corner_offsets: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    window_scale: wp.array[wp.float64],
    labels: wp.array[wp.float64],
    source_of: wp.array[wp.int32],
    popped: wp.array[wp.float64],
    pop_count: wp.array[wp.int32],
    front: wp.array[wp.int32],
    out_relaxed: wp.array[wp.float64],
    out_next_source_of: wp.array[wp.int32],
    out_popped_step: wp.array[wp.int32],
    out_listed: wp.array[wp.int32],
    out_touched: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_window: wp.array[wp.float64],
    out_scratch_targets: wp.array[wp.int32],
    out_scratch_offers: wp.array[wp.float64],
) -> None:
    # One step's selection and pops over ``front[parity]``, block ``block``'s share
    # (``ordered_pop_pass``). Selection reads only what the settle pass writes, so every block sees
    # the step's starting state.
    #
    # An ordinary pop owns its 1-ring for the step, so it reads and writes ``out_relaxed`` and
    # ``out_next_source_of`` there in place. Its offers are spread over the lanes, each computed
    # from the partner's distance as it stood at the chunk's start; the serial pop reads a partner
    # an earlier offer of the same pop lowered, so each lane then takes the smallest of the
    # chunk's earlier offers to its partner (through the block's scratch row) and recomputes,
    # until no lane's partner moves (two or three passes: a pop's offers chain at most a few
    # deep). One lane walks the offers one at a time and needs no second pass. A cycling pop
    # reads the starting ``labels`` and lowers by ``atomic_min``, its labels left to
    # ``ordered_relabel_pass``.
    lanes = wp.block_dim()
    n_vertices = labels.shape[0]
    stamp = 2 * round_index + parity
    window_bound = out_window[parity] + window_scale[0]
    if block == 0 and lane == 0:
        out_counts[ORDERED_TOUCHED + 1 - parity] = 0
    row = block * lanes
    for item in range(block, out_counts[ORDERED_FRONT + parity], blocks):
        v = front[parity * n_vertices + item]
        if lane == 0:
            list_touched(v, stamp, parity, out_listed, out_touched, out_counts)
        selected, lowest = ordered_pop_selected(
            v, lane, window_bound, neighbors, neighbor_offsets, labels, popped, pop_count
        )
        if lowest and lane == 0 and labels[v] < out_window[1 - parity]:
            wp.atomic_min(out_window, 1 - parity, labels[v])
        if not selected:
            continue
        if lane == 0:
            out_popped_step[v] = stamp
        d_v = labels[v]
        s_v = ordered_source(source_of, v)
        cycling = pop_count[v] >= UNFOLDING_RELAX_POPS
        n_offers = 2 * (corner_offsets[v + 1] - corner_offsets[v])
        for base in range(0, n_offers, lanes):
            j = base + lane
            active = j < n_offers
            f = wp.int32(0)
            z = wp.int32(0)
            target = wp.int32(1)
            pw = wp.int32(-1)
            pw1 = wp.int32(0)
            if active:
                f, z, target, pw, pw1 = ordered_offer_corners(v, j, corner_records, corner_offsets)
            offer = wp.float64(wp.inf)
            if cycling:
                if active:
                    offer = ordered_offer(
                        f,
                        z,
                        target,
                        lengths,
                        cosines,
                        angles,
                        d_v,
                        s_v,
                        labels[pw1],
                        ordered_source(source_of, pw1),
                        labels[pw],
                    )
                    if offer < labels[pw]:
                        wp.atomic_min(out_relaxed, pw, offer)
                        list_touched(pw, stamp, parity, out_listed, out_touched, out_counts)
                continue
            d_start = wp.float64(wp.inf)
            s_start = wp.int32(0)
            d_target = wp.float64(wp.inf)
            if active:
                d_start = out_relaxed[pw1]
                s_start = ordered_source(out_next_source_of, pw1)
                d_target = out_relaxed[pw]
            d_partner = d_start
            s_partner = s_start
            while True:
                if active:
                    offer = ordered_offer(
                        f,
                        z,
                        target,
                        lengths,
                        cosines,
                        angles,
                        d_v,
                        s_v,
                        d_partner,
                        s_partner,
                        d_target,
                    )
                if lanes == 1:
                    break
                out_scratch_targets[row + lane] = pw
                out_scratch_offers[row + lane] = offer
                block_barrier()
                d_seen = d_start
                s_seen = s_start
                for i in range(lane):
                    if out_scratch_targets[row + i] == pw1 and out_scratch_offers[row + i] < d_seen:
                        d_seen = out_scratch_offers[row + i]
                        s_seen = s_v
                moved = wp.int32(0)
                if active and (d_seen != d_partner or s_seen != s_partner):
                    moved = 1
                if block_max(moved) == 0:
                    break
                d_partner = d_seen
                s_partner = s_seen
            if active and offer < d_target:
                wp.atomic_min(out_relaxed, pw, offer)
                if out_next_source_of.shape[0] != 0:
                    out_next_source_of[pw] = s_v
                list_touched(pw, stamp, parity, out_listed, out_touched, out_counts)
            if lanes > 1:
                block_barrier()


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def ordered_pop_pass(
    parity: wp.int32,
    lengths: wp.array[wp.vec3d],
    cosines: wp.array[wp.vec3d],
    angles: wp.array[wp.vec3d],
    corner_records: wp.array[wp.vec4i],
    corner_offsets: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    window_scale: wp.array[wp.float64],
    labels: wp.array[wp.float64],
    source_of: wp.array[wp.int32],
    popped: wp.array[wp.float64],
    pop_count: wp.array[wp.int32],
    front: wp.array[wp.int32],
    state: wp.array[wp.int32],
    blocks: wp.int32,
    out_relaxed: wp.array[wp.float64],
    out_next_source_of: wp.array[wp.int32],
    out_popped_step: wp.array[wp.int32],
    out_listed: wp.array[wp.int32],
    out_touched: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_window: wp.array[wp.float64],
    out_scratch_targets: wp.array[wp.int32],
    out_scratch_offers: wp.array[wp.float64],
) -> None:
    # One step's selection and pops, a block per listed vertex: ``ordered_pop_block``.
    block, lane = wp.tid()
    ordered_pop_block(
        block,
        lane,
        blocks,
        parity,
        state[LOOP_ROUND],
        lengths,
        cosines,
        angles,
        corner_records,
        corner_offsets,
        neighbors,
        neighbor_offsets,
        window_scale,
        labels,
        source_of,
        popped,
        pop_count,
        front,
        out_relaxed,
        out_next_source_of,
        out_popped_step,
        out_listed,
        out_touched,
        out_counts,
        out_window,
        out_scratch_targets,
        out_scratch_offers,
    )


@wp.func
def ordered_relabel_block(
    block: wp.int32,
    lane: wp.int32,
    blocks: wp.int32,
    parity: wp.int32,
    round_index: wp.int32,
    lengths: wp.array[wp.vec3d],
    cosines: wp.array[wp.vec3d],
    angles: wp.array[wp.vec3d],
    corner_records: wp.array[wp.vec4i],
    corner_offsets: wp.array[wp.int32],
    labels: wp.array[wp.float64],
    source_of: wp.array[wp.int32],
    pop_count: wp.array[wp.int32],
    popped_step: wp.array[wp.int32],
    front: wp.array[wp.int32],
    relaxed: wp.array[wp.float64],
    counts: wp.array[wp.int32],
    out_bids: wp.array[wp.int32],
) -> None:
    # With several sources, after ``ordered_pop_pass``: each cycling pop recomputes its offers from
    # the same starting state and bids its label where one set its target's new distance (the
    # smallest bid wins a tie). Block ``block``'s share (``ordered_relabel_pass``).
    n_vertices = labels.shape[0]
    stamp = 2 * round_index + parity
    for item in range(block, counts[ORDERED_FRONT + parity], blocks):
        v = front[parity * n_vertices + item]
        if popped_step[v] != stamp or pop_count[v] < UNFOLDING_RELAX_POPS:
            continue
        d_v = labels[v]
        s_v = ordered_source(source_of, v)
        n_offers = 2 * (corner_offsets[v + 1] - corner_offsets[v])
        for j in range(lane, n_offers, wp.block_dim()):
            f, z, target, pw, pw1 = ordered_offer_corners(v, j, corner_records, corner_offsets)
            offer = ordered_offer(
                f,
                z,
                target,
                lengths,
                cosines,
                angles,
                d_v,
                s_v,
                labels[pw1],
                ordered_source(source_of, pw1),
                labels[pw],
            )
            if offer < labels[pw] and offer == relaxed[pw]:
                wp.atomic_min(out_bids, pw, s_v)


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def ordered_relabel_pass(
    parity: wp.int32,
    lengths: wp.array[wp.vec3d],
    cosines: wp.array[wp.vec3d],
    angles: wp.array[wp.vec3d],
    corner_records: wp.array[wp.vec4i],
    corner_offsets: wp.array[wp.int32],
    labels: wp.array[wp.float64],
    source_of: wp.array[wp.int32],
    pop_count: wp.array[wp.int32],
    popped_step: wp.array[wp.int32],
    front: wp.array[wp.int32],
    relaxed: wp.array[wp.float64],
    counts: wp.array[wp.int32],
    state: wp.array[wp.int32],
    blocks: wp.int32,
    out_bids: wp.array[wp.int32],
) -> None:
    # The cycling pops' labels: ``ordered_relabel_block``.
    block, lane = wp.tid()
    ordered_relabel_block(
        block,
        lane,
        blocks,
        parity,
        state[LOOP_ROUND],
        lengths,
        cosines,
        angles,
        corner_records,
        corner_offsets,
        labels,
        source_of,
        pop_count,
        popped_step,
        front,
        relaxed,
        counts,
        out_bids,
    )


@wp.func
def ordered_settle_block(
    block: wp.int32,
    lane: wp.int32,
    blocks: wp.int32,
    parity: wp.int32,
    round_index: wp.int32,
    touched: wp.array[wp.int32],
    relaxed: wp.array[wp.float64],
    next_source_of: wp.array[wp.int32],
    popped_step: wp.array[wp.int32],
    out_labels: wp.array[wp.float64],
    out_source_of: wp.array[wp.int32],
    out_bids: wp.array[wp.int32],
    out_popped: wp.array[wp.float64],
    out_pop_count: wp.array[wp.int32],
    out_front: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_window: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # After a step's pops, over ``touched[parity]``: a vertex that popped records the distance it
    # popped at, every vertex takes its lowered distance and label, and one still pending (lowered
    # since it last popped) is listed in ``front[1 - parity]``; the smallest such distance of an
    # ordinary vertex, folded over the block first, sets the next step's window. Block
    # ``block``'s share (``ordered_settle_pass``).
    lanes = wp.block_dim()
    n_vertices = out_labels.shape[0]
    stamp = 2 * round_index + parity
    if block == 0 and lane == 0:
        out_counts[ORDERED_FRONT + parity] = 0
        out_window[parity] = wp.float64(wp.inf)
    for item in range(block * lanes + lane, out_counts[ORDERED_TOUCHED + parity], blocks * lanes):
        v = touched[parity * n_vertices + item]
        if popped_step[v] == stamp:
            out_popped[v] = out_labels[v]
            out_pop_count[v] = out_pop_count[v] + 1
        out_labels[v] = relaxed[v]
        if out_source_of.shape[0] != 0:
            out_source_of[v] = next_source_of[v]
            if out_bids[v] != ORDERED_NO_BID:
                out_source_of[v] = out_bids[v]
                out_bids[v] = ORDERED_NO_BID
        if out_labels[v] < out_popped[v]:
            slot = append_slot(out_counts, ORDERED_FRONT + 1 - parity)
            out_front[(1 - parity) * n_vertices + slot] = v
            out_state[LOOP_PROGRESS] = 1


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def ordered_settle_pass(
    parity: wp.int32,
    touched: wp.array[wp.int32],
    relaxed: wp.array[wp.float64],
    next_source_of: wp.array[wp.int32],
    popped_step: wp.array[wp.int32],
    state: wp.array[wp.int32],
    blocks: wp.int32,
    out_labels: wp.array[wp.float64],
    out_source_of: wp.array[wp.int32],
    out_bids: wp.array[wp.int32],
    out_popped: wp.array[wp.float64],
    out_pop_count: wp.array[wp.int32],
    out_front: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_window: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # A step's settle: ``ordered_settle_block``.
    block, lane = wp.tid()
    ordered_settle_block(
        block,
        lane,
        blocks,
        parity,
        state[LOOP_ROUND],
        touched,
        relaxed,
        next_source_of,
        popped_step,
        out_labels,
        out_source_of,
        out_bids,
        out_popped,
        out_pop_count,
        out_front,
        out_counts,
        out_window,
        out_state,
    )


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def ordered_front_serial(
    lengths: wp.array[wp.vec3d],
    cosines: wp.array[wp.vec3d],
    angles: wp.array[wp.vec3d],
    corner_records: wp.array[wp.vec4i],
    corner_offsets: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    window_scale: wp.array[wp.float64],
    max_rounds: wp.int32,
    labels: wp.array[wp.float64],
    source_of: wp.array[wp.int32],
    relaxed: wp.array[wp.float64],
    next_source_of: wp.array[wp.int32],
    bids: wp.array[wp.int32],
    popped: wp.array[wp.float64],
    pop_count: wp.array[wp.int32],
    popped_step: wp.array[wp.int32],
    listed: wp.array[wp.int32],
    front: wp.array[wp.int32],
    touched: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    window: wp.array[wp.float64],
    scratch_targets: wp.array[wp.int32],
    scratch_offers: wp.array[wp.float64],
    state: wp.array[wp.int32],
) -> None:
    # One block of one lane, the CPU device's step loop: each step's ``ordered_pop_pass``, its
    # ``ordered_relabel_pass`` when labels are kept and ``ordered_settle_pass``, two steps a
    # round, then ``loop_advance``, as one lane running each launch in launch order, so every
    # buffer comes out as the launched loop leaves it.
    _block, lane = wp.tid()
    while state[LOOP_CONDITION] != 0:
        for parity in range(2):
            ordered_pop_block(
                0,
                lane,
                1,
                parity,
                state[LOOP_ROUND],
                lengths,
                cosines,
                angles,
                corner_records,
                corner_offsets,
                neighbors,
                neighbor_offsets,
                window_scale,
                labels,
                source_of,
                popped,
                pop_count,
                front,
                relaxed,
                next_source_of,
                popped_step,
                listed,
                touched,
                counts,
                window,
                scratch_targets,
                scratch_offers,
            )
            if source_of.shape[0] != 0:
                ordered_relabel_block(
                    0,
                    lane,
                    1,
                    parity,
                    state[LOOP_ROUND],
                    lengths,
                    cosines,
                    angles,
                    corner_records,
                    corner_offsets,
                    labels,
                    source_of,
                    pop_count,
                    popped_step,
                    front,
                    relaxed,
                    counts,
                    bids,
                )
            ordered_settle_block(
                0,
                lane,
                1,
                parity,
                state[LOOP_ROUND],
                touched,
                relaxed,
                next_source_of,
                popped_step,
                labels,
                source_of,
                bids,
                popped,
                pop_count,
                front,
                counts,
                window,
                state,
            )
        advance_round_loop(max_rounds, state)


# --------------------------------------------------------------------------------------
# Signed heat method: signed distance to oriented curves (Feng & Crane 2024)
# --------------------------------------------------------------------------------------


@wp.kernel
def source_and_global_sums(
    phi: wp.array[wp.float64],
    sources: wp.array[wp.int32],
    stride: wp.int32,
    out_sums: wp.array[wp.float64],
) -> None:
    # ``(sum of phi over the sources, sum of phi over every vertex)`` as one row per block of
    # ``out_sums``, which the last block adds in a fixed order (``store_sum_and_count``), for
    # ``shift_and_orient``: both means the heat method's normalization needs, in one launch. The
    # ``reduce`` block fold over whichever of the two ranges the launch covers; a block past the end
    # of the other contributes zero to it, so a launch sized to the sources alone gives a correct
    # first sum -- all ``shift_and_orient`` reads when it does not orient. ``sources`` may repeat a
    # vertex, which is then counted as often as it appears, as a gathered mean counts it.
    i, t = wp.tid()
    base, remaining = block_chunk_1d(phi.shape[0], i)
    source_base, source_remaining = block_chunk_1d(sources.shape[0], i)
    at_sources = wp.float64(0.0)
    everywhere = wp.float64(0.0)
    for k in range(t, source_remaining, wp.block_dim()):
        at_sources += phi[sources[source_base + k]]
    for k in range(t, remaining, wp.block_dim()):
        everywhere += phi[base + k]
    store_sum_and_count(i, t, at_sources, everywhere, out_sums, stride)


@wp.kernel
def shift_and_orient(
    sums: wp.array[wp.float64], n_sources: wp.int32, orient: wp.int32, phi: wp.array[wp.float64]
) -> None:
    # Shift ``phi`` so its mean over the sources is zero and, with ``orient`` set, orient it
    # positive -- the ``igl::heat_geodesics_solve`` convention, which makes a single source's
    # distance exactly zero -- from ``source_and_global_sums``' two sums, read on the device so the
    # host never waits for them. The shifted field's mean is the global mean less the offset, so
    # its sign needs no second pass. ``phi`` is updated in place. The signed method shifts its
    # field onto its curve the same way and must not orient it: its sign is the answer.
    v = wp.int32(wp.tid())
    offset = sums[0] / wp.float64(n_sources)
    shifted = phi[v] - offset
    if orient != 0 and sums[1] / wp.float64(phi.shape[0]) - offset < wp.float64(0.0):
        shifted = offset - phi[v]
    phi[v] = shifted


@wp.func
def curve_segment_end(
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32],
    single_curve: wp.int32,
    closed: wp.int32,
    k: wp.int32,
) -> wp.int32:
    # The entry of the packed ``curve_vertices`` where the segment starting at entry ``k`` ends:
    # the next entry of its curve, or -- the last entry of a ``closed`` curve of two or more --
    # the curve's first; ``-1`` when ``k`` starts no segment. The curve is ``curve_offsets``' CSR
    # row holding ``k`` (the whole buffer with ``single_curve`` set, when ``curve_offsets`` may be
    # ``None``); an entry in no row starts nothing. Shared by both signed heat discretizations'
    # source kernels and the edge one's shift onto the curve, so all three walk one segment list.
    begin = wp.int32(0)
    end = curve_vertices.shape[0]
    if single_curve == 0:
        curve = binary_search_index(curve_offsets, k) - 1
        if curve < 0 or curve >= curve_offsets.shape[0] - 1:
            return wp.int32(-1)
        begin = curve_offsets[curve]
        # Clamped to the buffer, as slicing it by a row past its end would be.
        end = wp.min(curve_offsets[curve + 1], end)
    if k + 1 < end:
        return k + 1
    if closed != 0 and end - begin >= 2:
        return begin
    return wp.int32(-1)


@wp.kernel
def splat_curve_normals(
    vertices: wp.array[wp.vec3],
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32],
    single_curve: wp.int32,
    closed: wp.int32,
    normals: wp.array[wp.vec3],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    out_field: wp.array[wp.vec2d],
    out_fixed: wp.array[wp.bool],
    out_invalid: wp.array[wp.int32],
) -> None:
    # The signed heat method's source term: each curve segment contributes its own *normal* -- the
    # tangent direction perpendicular to it -- to the two vertices it connects, weighted by half the
    # segment's length. Diffusing normals rather than an indicator is what makes the result signed:
    # the field arrives at a point already knowing which side of the curve it is on.
    #
    # One thread per entry ``k`` of the packed ``curve_vertices``, owning the segment that starts
    # there: to the next entry of its curve, or -- the last entry of a ``closed`` curve of two or
    # more -- back to the curve's first. The curve is ``curve_offsets``' CSR row holding ``k`` (the
    # whole buffer with ``single_curve`` set, when ``curve_offsets`` may be ``None``); an entry in
    # no row starts nothing. So the segment list is never built, and the offsets are never read
    # back to build it; on the CPU device the threads run in the order that list had.
    #
    # Every entry is range-checked by its own thread before anything else (raising
    # ``out_invalid[0]``), and a segment's far end again before it is read, so an out-of-range
    # curve vertex reads and writes nothing.
    #
    # A non-empty ``out_fixed`` (pre-zeroed) also gets each curve vertex marked, the pinned set of
    # the ``"zero_set"`` Poisson solve (``array.indices_to_mask``' membership mask), from the
    # thread that range-checked it.
    k = wp.int32(wp.tid())
    n_vertices = out_field.shape[0]
    if flag_out_of_range(curve_vertices[k], n_vertices, out_invalid):
        return
    if out_fixed.shape[0] > 0:
        out_fixed[curve_vertices[k]] = True
    far = curve_segment_end(curve_vertices, curve_offsets, single_curve, closed, k)
    if far < 0:
        return
    a = curve_vertices[k]
    b = curve_vertices[far]
    if flag_out_of_range(b, n_vertices, out_invalid):
        return
    edge = vertices[b] - vertices[a]
    length = wp.length(edge)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return
    direction = edge / length
    weight = wp.float64(0.5 * length)

    for k in range(2):
        v = a
        if k == 1:
            v = b
        normal = normals[v]
        # The segment direction as this vertex sees it, then rotated a quarter turn in the tangent
        # plane. ``cross(normal, direction)`` is the left normal, which is the orientation
        # geometry-central signs with: the region a counter-clockwise curve encloses comes out
        # positive.
        tangential, tangential_length = unit_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)
        if tangential_length <= TOLERANCE_ZERO_CONSTANT:
            continue
        curve_normal = wp.cross(normal, tangential)
        wp.atomic_add(
            out_field, v, weight * to_vec2d(world_to_tangent(curve_normal, basis_x[v], basis_y[v]))
        )


@wp.kernel
def vertex_field_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    field: wp.array[wp.vec2d],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    cot_entries: wp.array2d[wp.float32],
    out_div: wp.array[wp.float64],
    out_corner_div: wp.array2d[wp.float64],
) -> None:
    # The signed heat method's stage 3. Average the three corners' tangent vectors into one
    # per-face vector, in world space, and integrate it by the same thread so it stays in a
    # register. Each corner's 2D components mean nothing outside its own frame, so they have to be
    # expanded to 3D *before* averaging.
    #
    # ``field`` is the raw diffused field; each corner is normalized here (``stable_normalize``,
    # without underflow, zero only where the field is exactly zero), which is what makes the
    # *absolute* ``TOLERANCE_ZERO_CONSTANT`` correct below, where a raw diffused field carries the
    # mesh's scale and no absolute floor applies to it: the sum of three unit vectors carries no
    # coordinate scale, so the test only asks whether the three corners cancelled. Normalizing per
    # corner rather than in a pass of its own costs three normalizations a face instead of one a
    # vertex, against a map launch and an ``(n_vertices,)`` buffer.
    #
    # Accumulates the **negated** divergence, the Poisson right-hand side for the ``-L`` operator,
    # by integrating ``-X``: as in ``unit_gradient_divergence`` the divergence is linear in the
    # field and a sign flip is exact, so this is the negated sum without a pass to negate it.
    f = wp.int32(wp.tid())
    total = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        v = faces[f * 3 + k]
        total += tangent_to_world(to_vec2(stable_normalize(field[v])), basis_x[v], basis_y[v])
    tangential, _length = unit_tangent(total, normals[f], TOLERANCE_ZERO_CONSTANT)
    x = -to_vec3d(tangential)
    accumulate_face_divergence(vertices, faces, cot_entries, f, x, out_div, out_corner_div)


@wp.kernel
def gather_free_solution(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array2d[wp.float64],
    out_field: wp.array[wp.float64],
) -> None:
    # Expand the reduced solution back over every vertex; the pinned ones keep the value they were
    # pinned to, which for a zero level set is zero.
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        out_field[i] = wp.float64(0.0)
        return
    out_field[i] = solution[0, free_map[i]]


# --------------------------------------------------------------------------------------
# Signed heat method on edges: the Crouzeix-Raviart discretization of geometry-central
# --------------------------------------------------------------------------------------
#
# The same three stages as above, with the diffused field one complex number per unique edge
# instead of one per vertex. Edge ``e``'s value is measured in its own frame: the real axis along
# ``edges[e, 0] -> edges[e, 1]``, the imaginary axis that direction turned a quarter in a face
# holding the edge (by the face's winding). A frame is a gauge, so the direction chosen for each
# edge cancels: a source, a system entry and a sampled vector all change sign with it.
#
# The system is ``M + t L`` over the edges, ``M`` a third of the areas of the faces holding an
# edge and ``L`` the Crouzeix-Raviart connection Laplacian: each face corner couples the two edges
# meeting there with weight ``2 cot`` of the corner (the half-cotangent weight of the edge opposite,
# four times over) and the rotation between their frames, ``(-cos, sin)`` of the corner angle. It
# is Hermitian, and stored as the real ``2 x 2`` blocks ``[[a, -b], [b, a]]`` of each complex
# ``a + ib``, which is symmetric positive definite.


@wp.func
def crouzeix_raviart_slot(
    face_edges: wp.array2d[wp.int32], slot: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32, wp.int32]:
    # The term ``slot`` of the edge system: twelve per face, four per corner ``c`` -- whose two
    # edges ``i`` (from corner ``c + 1`` to ``c + 2``) and ``j`` (from ``c + 2`` to ``c``) meet at
    # corner ``c + 2`` -- as ``(face, corner, part, i, j)``; ``part`` 0 to 3 is the entry
    # ``(i, i)``, ``(j, j)``, ``(i, j)`` or ``(j, i)``. Shared by the key pass and the row pass, so
    # the pattern and the values index one term table.
    f = slot // 12
    local = slot - 12 * f
    corner = local // 4
    part = local - 4 * corner
    i = face_edges[f, (corner + 1) % 3]
    j = face_edges[f, (corner + 2) % 3]
    return f, corner, part, i, j


@wp.func
def crouzeix_raviart_entry(part: wp.int32, i: wp.int32, j: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The ``(row, column)`` a term of ``crouzeix_raviart_slot`` lands on.
    row = wp.where(part == 1 or part == 3, j, i)
    col = wp.where(part == 1 or part == 2, j, i)
    return row, col


@wp.func
def crouzeix_raviart_term(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    f: wp.int32,
    corner: wp.int32,
    part: wp.int32,
    i: wp.int32,
    j: wp.int32,
) -> tuple[wp.float64, wp.vec2d]:
    # One term of the edge system as ``(mass, stiffness)``: the mass a diagonal term adds (a sixth
    # of the face's area: each edge of a face is a diagonal term at two of its corners, so the face
    # gives it a third) and the stiffness as a complex number ``(re, im)``. Formed in ``float64``
    # from the positions; a face of exactly zero area gives nothing.
    h = 3 * f + corner
    tail, tip = halfedge_endpoints(faces, h)
    apex = faces[halfedge_prev(h)]
    p_apex = to_vec3d(vertices[apex])
    to_tail = to_vec3d(vertices[tail]) - p_apex
    to_tip = to_vec3d(vertices[tip]) - p_apex
    double_area = wp.length(wp.cross(to_tail, to_tip))
    if double_area == wp.float64(0.0):
        return wp.float64(0.0), wp.vec2d(0.0, 0.0)
    cosine = wp.dot(to_tail, to_tip)
    weight = wp.float64(2.0) * cosine / double_area
    if part < 2:
        return double_area / wp.float64(12.0), wp.vec2d(weight, wp.float64(0.0))
    # Edge ``i`` runs tip -> apex in this face and edge ``j`` apex -> tail; ``sign`` is whether the
    # two frames' directions agree with those runs alike.
    sign = wp.where(edges[i, 0] == tip, wp.float64(1.0), wp.float64(-1.0))
    if edges[j, 0] != apex:
        sign = -sign
    lengths = wp.length(to_tail) * wp.length(to_tip)
    rotation = wp.vec2d(-cosine / lengths, double_area / lengths)
    if part == 3:
        rotation = wp.vec2d(rotation[0], -rotation[1])
    return wp.float64(0.0), -weight * sign * rotation


@wp.kernel
def crouzeix_raviart_keys(
    face_edges: wp.array2d[wp.int32],
    n_edges: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The edge system's sparsity, as one ``csr_key`` per term of ``crouzeix_raviart_slot`` with the
    # term's slot as its payload: one thread per face, twelve terms.
    f = wp.int32(wp.tid())
    for local in range(12):
        slot = 12 * f + local
        _face, _corner, part, i, j = crouzeix_raviart_slot(face_edges, slot)
        row, col = crouzeix_raviart_entry(part, i, j)
        out_keys[slot] = csr_key(row, col, n_edges, n_edges)
        out_order[slot] = slot


@wp.kernel
def crouzeix_raviart_system_rows(
    offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    order: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_edges: wp.array2d[wp.int32],
    edges: wp.array2d[wp.int32],
    scale: wp.float64,
    edge_sum_and_count: wp.array[wp.float64],
    from_edges: wp.int32,
    out_values: wp.array[wp.mat22d],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    # Edge ``e``'s row of ``M + t L``, each entry the sum of its terms in the sorted order
    # ``array.csr_from_keys`` left them (the producer's slot order), so it repeats bit for bit.
    # ``t`` is ``shift_factor``'s: ``scale`` itself, or with ``from_edges`` set ``scale`` times the
    # squared mean unique-edge length, formed on the device. Also the edge's midpoint, the position
    # the system's factorization orders its rows by.
    e = wp.int32(wp.tid())
    factor = shift_factor(scale, edge_sum_and_count, from_edges)
    for p in range(offsets[e], offsets[e + 1]):
        mass = wp.float64(0.0)
        stiffness = wp.vec2d(0.0, 0.0)
        for q in range(starts[p], starts[p + 1]):
            f, corner, part, i, j = crouzeix_raviart_slot(face_edges, order[q])
            term_mass, term = crouzeix_raviart_term(vertices, faces, edges, f, corner, part, i, j)
            mass += term_mass
            stiffness += term
        real = mass + factor * stiffness[0]
        imaginary = factor * stiffness[1]
        out_values[p] = wp.mat22d(real, -imaginary, imaginary, real)
    out_midpoints[e] = wp.float32(0.5) * (vertices[edges[e, 0]] + vertices[edges[e, 1]])


@wp.kernel
def splat_curve_edges(
    vertices: wp.array[wp.vec3],
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32],
    single_curve: wp.int32,
    closed: wp.int32,
    edges: wp.array2d[wp.int32],
    sums: GroupSumPass,
    out_source: wp.array[wp.vec2d],
    out_fixed: wp.array[wp.bool],
    out_lengths: wp.array[wp.float64],
    out_invalid: wp.array[wp.int32],
) -> None:
    # The edge discretization's source term: a segment lying on a mesh edge adds its length along
    # the imaginary axis of that edge's frame -- the edge's normal, ``+i`` where the segment runs
    # along the frame's direction -- and a segment on no edge adds nothing. One thread per entry
    # of the packed ``curve_vertices`` (``curve_segment_end``), range-checked first as in
    # ``splat_curve_normals``, which also marks the zero set into a non-empty ``out_fixed``. A
    # non-empty ``out_lengths`` (pre-zeroed) gets each source segment's length at its first entry,
    # the weights of the shift onto the curve (``curve_midpoint_sums``). The terms go through
    # ``scatter.scatter_add_vector``, so an edge two segments share sums in a fixed order under
    # ``ordito.config``'s deterministic mode.
    k = wp.int32(wp.tid())
    n_vertices = vertices.shape[0]
    if flag_out_of_range(curve_vertices[k], n_vertices, out_invalid):
        return
    if out_fixed.shape[0] > 0:
        out_fixed[curve_vertices[k]] = True
    far = curve_segment_end(curve_vertices, curve_offsets, single_curve, closed, k)
    if far < 0:
        return
    a = curve_vertices[k]
    b = curve_vertices[far]
    if flag_out_of_range(b, n_vertices, out_invalid) or a == b:
        return
    e = unique_edge_row(edges, edges.shape[0], a, b, wp.uint64(n_vertices))
    if e < 0:
        return
    length = wp.length(to_vec3d(vertices[b]) - to_vec3d(vertices[a]))
    if out_lengths.shape[0] > 0:
        out_lengths[k] = length
    along = wp.where(edges[e, 0] == a, length, -length)
    scatter_add_vector(out_source, sums, e, wp.vec2d(wp.float64(0.0), along), ARRIVAL_POOL_UNIFORM)


# The magnitude below which a diffused edge value is taken to have run into ``float64`` underflow
# (``edge_field_divergence``'s flag). At geometry-central's ``t = (h / 2) ** 2`` the heat falls
# about 0.7 decades an edge hop, so on a fine mesh it leaves the representable range before it
# has crossed the surface, and every value past that front is zero or unconverged noise (unit
# icosphere of 327 680 faces: 0.88 rad of error, potpourri3d's own solve 1.29). The front always
# passes through this band on its way down -- the broken field held 391 edges in it -- while every
# field that held its accuracy kept its smallest value above 1e-137, so the bar sits far from
# both. Exact zeros are not flagged: a component the curve never touches is zero by right.
EDGE_FIELD_UNDERFLOW = wp.float64(1e-280)


@wp.kernel
def edge_field_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_edges: wp.array2d[wp.int32],
    edges: wp.array2d[wp.int32],
    field: wp.array[wp.vec2d],
    cot_entries: wp.array2d[wp.float32],
    out_div: wp.array[wp.float64],
    out_corner_div: wp.array2d[wp.float64],
    out_underflow: wp.array[wp.int32],
) -> None:
    # The edge discretization's stage 3, the counterpart of ``vertex_field_divergence``: face
    # ``f``'s vector is the sum of its three edges' values expanded from their frames into world
    # space (the face's own quarter turn for the imaginary axis), normalized (zero only where it is
    # exactly zero), and integrated by the same thread. Accumulates the negated divergence by
    # integrating ``-X``, as ``vertex_field_divergence`` does. Sets ``out_underflow[0]`` to 1 when
    # an edge it reads holds a non-zero value below ``EDGE_FIELD_UNDERFLOW`` (every writer stores
    # the same word, so no atomic).
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    normal = stable_normalize(wp.cross(v1 - v0, v2 - v0))
    total = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(3):
        e = face_edges[f, k]
        tangent = stable_normalize(
            to_vec3d(vertices[edges[e, 1]]) - to_vec3d(vertices[edges[e, 0]])
        )
        value = field[e]
        size = wp.max(wp.abs(value[0]), wp.abs(value[1]))
        if size > wp.float64(0.0) and size < EDGE_FIELD_UNDERFLOW:
            out_underflow[0] = 1
        total += value[0] * tangent + value[1] * wp.cross(normal, tangent)
    x = -stable_normalize(total)
    accumulate_face_divergence(vertices, faces, cot_entries, f, x, out_div, out_corner_div)


@wp.kernel
def curve_midpoint_sums(
    phi: wp.array[wp.float64],
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32],
    single_curve: wp.int32,
    closed: wp.int32,
    lengths: wp.array[wp.float64],
    stride: wp.int32,
    out_sums: wp.array[wp.float64],
) -> None:
    # ``(integral of phi along the source segments, their total length)`` as one row per block of
    # ``out_sums`` (``store_sum_and_count``), the midpoint rule on each segment -- exact for the
    # piecewise-linear field -- weighted by the lengths ``splat_curve_edges`` stored (zero for an
    # entry that starts no source segment).
    i, t = wp.tid()
    base, remaining = block_chunk_1d(curve_vertices.shape[0], i)
    integral = wp.float64(0.0)
    total = wp.float64(0.0)
    for k in range(t, remaining, wp.block_dim()):
        entry = base + k
        length = lengths[entry]
        if length > wp.float64(0.0):
            far = curve_segment_end(curve_vertices, curve_offsets, single_curve, closed, entry)
            middle = phi[curve_vertices[entry]] + phi[curve_vertices[far]]
            integral += wp.float64(0.5) * length * middle
            total += length
    store_sum_and_count(i, t, integral, total, out_sums, stride)


@wp.kernel
def shift_by_mean(sums: wp.array[wp.float64], phi: wp.array[wp.float64]) -> None:
    # Shift ``phi`` in place by ``curve_midpoint_sums``' mean along the curve, read on the device;
    # no shift for a curve with no source segment.
    v = wp.int32(wp.tid())
    phi[v] = phi[v] - divide_if_positive(sums[0], sums[1])


# --------------------------------------------------------------------------------------
# Vector heat method: transport, scalar extension and the log map (Sharp et al. 2019)
# --------------------------------------------------------------------------------------


@wp.kernel
def seed_source_scalars(
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_indicator: wp.array[wp.float64],
    out_weighted: wp.array[wp.float64],
    out_range: wp.array[wp.float64],
    out_invalid: wp.array[wp.int32],
) -> None:
    # Scalar extension needs two right-hand sides: where the sources are, and what they carry.
    # ``out_range`` (seeded ``+inf``) also collects the source values' range for
    # ``extend_from_ratio``'s overshoot check, stored as ``(min, -max)`` so both ends are one
    # ``atomic_min`` from one ``wp.full``. An out-of-range source raises ``out_invalid[0]``.
    s = wp.int32(wp.tid())
    v = sources[s]
    if flag_out_of_range(v, out_indicator.shape[0], out_invalid):
        return
    wp.atomic_add(out_indicator, v, wp.float64(1.0))
    wp.atomic_add(out_weighted, v, values[s])
    wp.atomic_min(out_range, 0, values[s])
    wp.atomic_min(out_range, 1, -values[s])


@wp.kernel
def seed_transport_sources(
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    out_field: wp.array[wp.vec2d],
    out_indicator: wp.array[wp.float64],
    out_magnitude: wp.array[wp.float64],
    out_invalid: wp.array[wp.int32],
) -> None:
    # ``transport_tangent_vectors``' three right-hand sides from one pass over the sources: the
    # widened vector, where the sources are, and the length each carries -- the vector heat seed and
    # ``seed_source_scalars``' pair, which a caller of both used to write in four launches. An
    # out-of-range source raises ``out_invalid[0]``.
    s = wp.int32(wp.tid())
    v = sources[s]
    if flag_out_of_range(v, out_field.shape[0], out_invalid):
        return
    vector = to_vec2d(vectors[s])
    wp.atomic_add(out_field, v, vector)
    wp.atomic_add(out_indicator, v, wp.float64(1.0))
    wp.atomic_add(out_magnitude, v, wp.length(vector))


@wp.kernel
def seed_log_map_source(
    source: wp.int32,
    out_sources: wp.array[wp.int32],
    out_field: wp.array[wp.vec2d],
    out_indicator: wp.array[wp.float64],
) -> None:
    # ``log_map``'s two right-hand sides, the source's reference direction and its heat indicator,
    # and the one-entry source list the distance's shift reads.
    out_sources[0] = source
    out_field[source] = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
    out_indicator[source] = wp.float64(1.0)


@wp.func
def divide_nonzero(numerator: wp.float64, denominator: wp.float64) -> wp.float64:
    # ``extend_scalar``'s ratio of the diffused values to the diffused indicator, zero only where
    # the indicator is exactly zero -- a component no source reaches. Both fields are converged per
    # vertex (the settle rule at ``heat._HEAT_CHECK_ROUNDS``), so a small indicator is a real one;
    # and it can be *negative*, as the heat system is not an M-matrix where obtuse triangles give
    # positive off-diagonal entries, where the ratio is still the extension (geometry-central
    # divides the same way).
    #
    # Not ``array.divide_if_positive``: that one's fallback is the unchanged numerator.
    if denominator == wp.float64(0.0):
        return wp.float64(0.0)
    return numerator / denominator


@wp.kernel
def extend_from_ratio(
    diffused_indicator: wp.array[wp.float64],
    diffused_weighted: wp.array[wp.float64],
    out_extended: wp.array[wp.float64],
    out_range: wp.array[wp.float64],
) -> None:
    # ``extend_scalar``'s quotient, and how far it leaves the sources' range: with every heat
    # weight non-negative the extension is a convex combination of the source values, and an
    # overshoot past ``[min, max]`` is the sign that obtuse triangles made the system lose that.
    # The overshoot is stored negated in ``out_range[2]`` (seeded ``+inf``, one ``atomic_min``
    # like the range ``seed_source_scalars`` wrote in slots 0 and 1); a vertex no source reaches
    # (indicator exactly zero) has no value and is not measured.
    i = wp.int32(wp.tid())
    indicator = diffused_indicator[i]
    extended = divide_nonzero(diffused_weighted[i], indicator)
    out_extended[i] = extended
    if indicator != wp.float64(0.0):
        excess = wp.max(extended - (-out_range[1]), out_range[0] - extended)
        wp.atomic_min(out_range, 2, -wp.max(excess, wp.float64(0.0)))


@wp.func
def scale_to_magnitude(direction: wp.vec2d, magnitude: wp.float64) -> wp.vec2d:
    # The vector heat method splits a transported vector into a direction (from the vector
    # diffusion) and a magnitude (from a scalar extension): short-time vector diffusion smears
    # magnitudes but preserves directions well. The direction is normalized without underflow
    # (``stable_normalize``) and is zero only where the diffused field is exactly zero: it is
    # converged per vertex (the settle rule at ``heat._HEAT_CHECK_ROUNDS``), so its far field is a
    # direction however small -- ``1e-100`` of the maximum a few hundred rings out -- and a floor
    # relative to the field's maximum, as this once took, discarded exactly that.
    return magnitude * stable_normalize(direction)


@wp.func
def narrow_direction(v: wp.vec2d) -> wp.vec2:
    # A diffused tangent field's direction, normalized in ``float64`` and only then narrowed to its
    # storage precision: narrowing first would underflow the far field, whose raw values are far
    # below ``float32``'s range. Zero where the field is exactly zero.
    return to_vec2(stable_normalize(v))


@wp.func
def transported_and_resolved(
    direction: wp.vec2d,
    diffused_magnitude: wp.float64,
    diffused_indicator: wp.float64,
    resolved_fraction: wp.float64,
) -> tuple[wp.vec2, wp.bool]:
    # The whole tail of ``transport_tangent_vectors`` in one pass: extend the source magnitudes
    # (``extend_scalar``'s ``divide_nonzero`` of the two diffused scalars), rescale the diffused
    # direction to it, narrow it to the field's storage precision, and report whether this
    # vertex's direction can be told from round-off.
    #
    # Resolution is asked **locally**: the diffused vector's length against the diffused
    # *magnitudes* at the same vertex (the same heat, applied to ``|v|``). The two are equal where
    # the copies arriving from the sources agree -- a vector's length is never more than the heat
    # of its magnitude -- and the ratio falls to round-off where they cancel, the cut locus. Against
    # the field's global maximum instead, as this once asked, every far vertex read as unresolved.
    magnitude = divide_nonzero(diffused_magnitude, diffused_indicator)
    return (
        to_vec2(scale_to_magnitude(direction, magnitude)),
        stable_length(direction) > resolved_fraction * diffused_magnitude,
    )


@wp.kernel
def scatter_unit_gradient_to_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    out_vertex_field: wp.array[wp.vec3],
    out_corner_field: wp.array2d[wp.vec3],
) -> None:
    # The log map's radial direction -- the unit gradient of the distance field, pointing away from
    # the source -- formed and scattered onto vertices by the same thread. The scatter is a plain
    # area-weighted add: the weights are the same for all three corners, so no normalization is
    # needed before projecting. Added atomically, or stored per corner into a non-empty
    # ``out_corner_field`` for ``scatter.sum_sorted_corner_values`` to add in ascending corner
    # order under ``ordito.config``'s deterministic mode: at the cut locus the faces' directions
    # cancel, and the atomics' arrival order then chose the angle there.
    f = wp.int32(wp.tid())
    x = face_unit_gradient(vertices, faces, normals, areas, values, f)
    area = areas[f]
    value = to_vec3(x) * area
    add_or_store_corner_triple(out_vertex_field, out_corner_field, faces, f, value, value, value)


@wp.func
def world_to_tangent_unit(value: wp.vec3, basis_x: wp.vec3, basis_y: wp.vec3) -> wp.vec2:
    # Express a 3D vertex field in each vertex's tangent basis, normalized. Only the direction
    # survives, which is all the log map's angle needs; zero only for an exactly zero projection.
    return stable_normalize(world_to_tangent(value, basis_x, basis_y))


@wp.kernel
def log_map_from_angles(
    vertex_gradient: wp.array[wp.vec3],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    transported_raw: wp.array[wp.vec2d],
    distance: wp.array[wp.float64],
    out_log: wp.array[wp.vec2],
) -> None:
    # Polar coordinates of each vertex as seen from the source.
    #
    # ``transported`` is the source's reference direction parallel-transported to this vertex, and
    # ``radial`` points away from the source here. The angle between them is preserved by transport
    # along the connecting geodesic, so it *is* the angle at which that geodesic leaves the
    # source -- which with the distance gives the vertex's position in the source's tangent plane.
    #
    # Both directions are formed here from the raw fields, each read only at ``v``: the radial one
    # from the scattered distance gradient (``world_to_tangent_unit``) and the reference one from
    # the diffused field (``narrow_direction``), so neither is a map and a buffer of its own.
    v = wp.int32(wp.tid())
    reference = narrow_direction(transported_raw[v])
    outward = world_to_tangent_unit(vertex_gradient[v], basis_x[v], basis_y[v])
    r = wp.float32(distance[v])
    # Both are unit length or exactly zero: ``narrow_direction`` and ``world_to_tangent_unit``
    # normalized them before narrowing, so the test is only "was it zeroed".
    if (
        wp.length(reference) <= TOLERANCE_ZERO_CONSTANT
        or wp.length(outward) <= TOLERANCE_ZERO_CONSTANT
    ):
        # On the cut locus the transported directions arriving from either side cancel and there is
        # no angle to report -- the log map genuinely has none there. Keep the radius and use angle
        # zero, so the magnitude still means what it should.
        out_log[v] = wp.vec2(r, 0.0)
        return
    angle = wp.atan2(cross2(reference, outward), wp.dot(reference, outward))
    out_log[v] = wp.vec2(r * wp.cos(angle), r * wp.sin(angle))


# The vector heat method's own log map (Sharp, Soliman and Crane 2019, section 8 and appendix A),
# ``log_map(method="potpourri3d")``: the radial direction is a second vector diffusion, of an
# outward-pointing ball at the source, rather than the distance gradient ``log_map_from_angles``
# reads, and the radius is that radial field integrated by one Poisson solve. Tangent vectors are
# complex numbers in each vertex's polar coordinates (``tangent_space.halfedge_tangent_angles``);
# a halfedge ``i -> j`` carries one from ``i``'s frame into ``j``'s by its transport angle.

HALF_PI_D = wp.float64(0.5 * math.pi)
PI_D = wp.float64(math.pi)


@wp.func
def polar_unit(angle: wp.float64) -> wp.vec2d:
    # The unit tangent vector at polar ``angle``, as the complex number ``exp(i angle)``.
    return wp.vec2d(wp.cos(angle), wp.sin(angle))


@wp.func
def complex_product(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    # Two tangent vectors multiplied as complex numbers: ``b`` rotated and scaled by ``a``.
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def reversed_halfedge_angle(
    tangent_angles: wp.array[wp.float64], transport_angles: wp.array[wp.float64], h: wp.int32
) -> wp.float64:
    # The polar angle of halfedge ``h``'s reverse ``j -> i`` in ``j``'s frame, read off ``h``
    # alone: transport along ``h`` turns ``h``'s own direction into the reverse's opposite. Exists
    # for a boundary edge too, whose reverse is no face's halfedge.
    return tangent_angles[h] + transport_angles[h] - PI_D


@wp.kernel
def seed_log_map_outward_ball(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    tangent_angles: wp.array[wp.float64],
    transport_angles: wp.array[wp.float64],
    source: wp.int32,
    out_radial: wp.array[wp.vec2d],
    out_horizontal: wp.array[wp.vec2d],
) -> None:
    # The two right-hand sides of the vector heat log map, by one thread: the source's reference
    # direction (``out_horizontal``) and the outward ball (``out_radial``), the integrated radial
    # unit field of a small disk around the source in closed form (appendix A of the vector heat
    # method). Each face at the source, with corners ``(a, b, c)`` counter-clockwise from the
    # source ``a``, adds one term at each corner, from its angles at ``a`` and ``b`` and its
    # heights; the terms at ``b`` and ``c`` point along their edges back to the source, the one at
    # ``a`` along ``a -> b``. A degenerate face has no heights and adds nothing. One thread walks
    # the source's ring in its fixed order, so the sums repeat bit for bit with no atomics.
    out_horizontal[source] = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
    pa = to_vec3d(vertices[source])
    for j in range(ring_offsets[source], ring_offsets[source + 1]):
        ab = ring_halfedges[j]
        f = ab // wp.int32(3)
        k = ab - wp.int32(3) * f
        bc = wp.int32(3) * f + (k + wp.int32(1)) % wp.int32(3)
        ca = wp.int32(3) * f + (k + wp.int32(2)) % wp.int32(3)
        b = faces[bc]
        c = faces[ca]
        pb = to_vec3d(vertices[b])
        pc = to_vec3d(vertices[c])
        twice_area = wp.length(wp.cross(pb - pa, pc - pa))
        if twice_area == wp.float64(0.0):
            continue
        theta = vector_angle(pb - pa, pc - pa)
        gamma = vector_angle(pa - pb, pc - pb)
        sine = wp.sin(theta)
        swing = theta * wp.cos(theta) - sine

        height_b = twice_area / wp.length(pc - pa)
        toward_b = wp.vec2d(-theta * sine, swing) / (wp.float64(2.0) * height_b)
        out_radial[b] = out_radial[b] + complex_product(
            toward_b, polar_unit(reversed_halfedge_angle(tangent_angles, transport_angles, ab))
        )

        height_c = twice_area / wp.length(pb - pa)
        toward_c = wp.vec2d(-theta * sine, -swing) / (wp.float64(2.0) * height_c)
        out_radial[c] = out_radial[c] + complex_product(toward_c, polar_unit(tangent_angles[ca]))

        height_a = twice_area / wp.length(pc - pb)
        alpha = HALF_PI_D - gamma
        at_source = wp.vec2d(
            -(theta * wp.cos(alpha) + wp.cos(alpha - theta) * sine) / (wp.float64(2.0) * height_a),
            -(wp.cos(alpha) - wp.cos(alpha - wp.float64(2.0) * theta) + wp.float64(2.0) * theta
              * wp.sin(alpha)) / (wp.float64(4.0) * height_a),
        )  # fmt: skip
        out_radial[source] = out_radial[source] + complex_product(
            at_source, polar_unit(tangent_angles[ab])
        )


@wp.func
def unit_radial(radial_raw: wp.array[wp.vec2d], v: wp.int32, source: wp.int32) -> wp.vec2d:
    # The diffused outward ball's direction at ``v``, and zero at the source, where it has none.
    if v == source:
        return wp.vec2d()
    return stable_normalize(radial_raw[v])


@wp.func
def halfedge_radial_flux(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    tangent_angles: wp.array[wp.float64],
    transport_angles: wp.array[wp.float64],
    radial_raw: wp.array[wp.vec2d],
    source: wp.int32,
    half_cotangent: wp.float64,
    h: wp.int32,
    next_h: wp.int32,
) -> wp.float64:
    # One face's share of the radial field's flux along its halfedge ``h = i -> j``: the field
    # averaged over the edge in ``i``'s frame (``j``'s carried back across the edge) dotted with the
    # edge vector, times the face's half-cotangent weight of the edge. The divergence at ``i`` loses
    # it and at ``j`` gains it: the reverse halfedge's flux is its negation, so both directions of
    # an edge, the boundary's included, come from the face that holds it.
    i = faces[h]
    j = faces[next_h]
    radial_i = unit_radial(radial_raw, i, source)
    radial_j = unit_radial(radial_raw, j, source)
    back = complex_product(radial_j, polar_unit(-transport_angles[h]))
    length = wp.length(to_vec3d(vertices[j]) - to_vec3d(vertices[i]))
    edge = length * polar_unit(tangent_angles[h])
    return half_cotangent * wp.dot(wp.float64(0.5) * (radial_i + back), edge)


@wp.kernel
def radial_field_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    tangent_angles: wp.array[wp.float64],
    transport_angles: wp.array[wp.float64],
    radial_raw: wp.array[wp.vec2d],
    source: wp.int32,
    cot_entries: wp.array2d[wp.float32],
    out_div: wp.array[wp.float64],
    out_corner_div: wp.array2d[wp.float64],
) -> None:
    # The negated cotangent divergence of the unit radial field, ``-sum_j w_ij <X_ij, e_ij>`` at
    # each vertex ``i``, which the Poisson system ``-L phi = rhs`` integrates into the distance.
    # Per face, each edge weighted by the half-cotangent of the opposite corner, so the two faces
    # of an interior edge add up to its whole cotangent weight. Added atomically, or stored per
    # corner for ``scatter.sum_sorted_corner_values`` under the deterministic mode, as
    # ``unit_gradient_divergence``.
    f = wp.int32(wp.tid())
    base = wp.int32(3) * f
    flux_01 = halfedge_radial_flux(
        vertices, faces, tangent_angles, transport_angles, radial_raw, source,
        wp.float64(cot_entries[f, 2]), base, base + 1,
    )  # fmt: skip
    flux_12 = halfedge_radial_flux(
        vertices, faces, tangent_angles, transport_angles, radial_raw, source,
        wp.float64(cot_entries[f, 0]), base + 1, base + 2,
    )  # fmt: skip
    flux_20 = halfedge_radial_flux(
        vertices, faces, tangent_angles, transport_angles, radial_raw, source,
        wp.float64(cot_entries[f, 1]), base + 2, base,
    )  # fmt: skip
    add_or_store_corner_triple(
        out_div, out_corner_div, faces, f, flux_20 - flux_01, flux_01 - flux_12, flux_12 - flux_20
    )


@wp.kernel
def log_map_from_vector_fields(
    radial_raw: wp.array[wp.vec2d],
    horizontal_raw: wp.array[wp.vec2d],
    distance: wp.array[wp.float64],
    source: wp.int32,
    out_log: wp.array[wp.vec2],
) -> None:
    # The radial direction measured from the transported reference, as the quotient of the two unit
    # fields (``radial * conj(horizontal)``), scaled by the distance shifted to zero at the source.
    # Zero where either field is exactly zero: at the source, and where no diffusion reached.
    v = wp.int32(wp.tid())
    radial = unit_radial(radial_raw, v, source)
    horizontal = stable_normalize(horizontal_raw[v])
    conjugate = wp.vec2d(horizontal[0], -horizontal[1])
    out_log[v] = to_vec2((distance[v] - distance[source]) * complex_product(radial, conjugate))


# The concrete handles of the two generic row kernels, keyed by the half-cotangent table's dtype
# -- see [`OverloadTable`][ordito.kernels.array.OverloadTable]. The scalar system reads the
# ``float32`` table ``heat_face_terms`` writes or a caller's own in either precision; the vector
# system reads that same ``float32`` table, or ``laplacian.cotmatrix_entries``' ``float64`` one
# beside a bundle that holds another.
HEAT_SYSTEM_ROWS: OverloadTable
VECTOR_HEAT_SYSTEM_ROWS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global HEAT_SYSTEM_ROWS, VECTOR_HEAT_SYSTEM_ROWS
    i32 = wp.array[wp.int32]
    f64 = wp.array[wp.float64]
    pattern = [i32, i32, i32, wp.array[wp.uint64], wp.int32, i32]
    HEAT_SYSTEM_ROWS = OverloadTable(
        heat_system_rows,
        {
            d: [
                *pattern,
                wp.array2d[d],
                wp.float64,
                f64,
                wp.int32,
                f64,
                wp.float64,
                i32,
                f64,
                wp.array2d[wp.float64],
                f64,
                f64,
                f64,
                f64,
                wp.array[wp.mat22d],
            ]
            for d in (wp.float32, wp.float64)
        },
    )
    VECTOR_HEAT_SYSTEM_ROWS = OverloadTable(
        vector_heat_system_rows,
        {
            d: [
                *pattern,
                i32,
                wp.array2d[d],
                f64,
                f64,
                wp.float64,
                f64,
                wp.int32,
                wp.array[wp.mat22d],
            ]
            for d in (wp.float32, wp.float64)
        },
    )


_register_overloads()
