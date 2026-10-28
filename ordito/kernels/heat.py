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

from ordito.constants import TOLERANCE_ZERO_CONSTANT
from ordito.kernels.array import (
    OverloadTable,
    binary_search_index,
    cross2,
    csr_key,
    divide_if_positive,
    sorted_run_end,
    to_vec2,
    to_vec2d,
    to_vec3,
    to_vec3d,
    unique_edge_row,
)
from ordito.kernels.laplacian import (
    connection_offdiagonal,
    cotmatrix_offdiagonal,
    face_half_cotangents,
    pattern_row_on_rim,
)
from ordito.kernels.linalg import shift_factor, shifted_block_entry, shifted_scalar_entry
from ordito.kernels.predicates import (
    stable_length,
    stable_normalize,
    unit_tangent,
    vector_angle,
    world_to_tangent,
)
from ordito.kernels.reduce import GroupSumPass, block_chunk_1d, store_sum_and_count
from ordito.kernels.scatter import add_corner_triple, add_or_store_corner_triple, scatter_add_vector
from ordito.kernels.triangles import (
    face_normals_and_area,
    face_unit_gradient,
    face_vertices_vec3d,
    face_voronoi_areas,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def heat_face_terms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    write_cot: wp.int32,
    voronoi_mass: wp.int32,
    out_cot: wp.array2d[wp.float32],
    out_normals: wp.array[wp.vec3],
    out_areas: wp.array[wp.float32],
    out_area_sums: wp.array[wp.float64],
) -> None:
    # Every per-face quantity the heat operators read, in one pass: the ``float32`` half-cotangent
    # table (``laplacian.cotmatrix_entries``' row, skipped when ``write_cot`` is 0 because the
    # caller has its own), the unit normal and area (``triangles.face_normals_and_areas``'), and
    # the area's exact ``float64`` scatter into each corner (``scatter.scatter_face_areas_exact``,
    # the lumped mass before its division by three). Each value is the standalone kernel's bit
    # for bit. With ``voronoi_mass`` set the corners take their Voronoi shares instead
    # (``triangles.face_voronoi_areas``), the lumped mass itself with no division to follow.
    f = wp.int32(wp.tid())
    if write_cot != 0:
        c0, c1, c2 = face_half_cotangents(vertices, faces, f)
        out_cot[f, 0] = c0
        out_cot[f, 1] = c1
        out_cot[f, 2] = c2
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
    # the block's row (``store_sum_and_count``), added in a fixed order afterwards.
    i, t = wp.tid()
    n = offsets.shape[0] - 1
    base, remaining = block_chunk_1d(n, i)
    if remaining <= 0:
        return
    total = wp.float64(0.0)
    n_edges = wp.float64(0.0)
    for k in range(t, remaining, wp.block_dim()):
        row = base + k
        for entry in range(offsets[row], offsets[row + 1]):
            column = columns[entry]
            if column > row:
                weight = wp.float64(1.0)
                if face_edges != 0:
                    weight = wp.float64(sorted_run_end(keys, run_start[entry], count))
                    weight -= wp.float64(run_start[entry])
                total += weight * wp.float64(wp.length(vertices[column] - vertices[row]))
                n_edges += weight
    store_sum_and_count(i, t, total, n_edges, out_sum_and_count, stride)


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
    out_laplacian: wp.array[wp.float64],
    out_heat: wp.array[wp.float64],
    out_poisson: wp.array[wp.float64],
    out_mass: wp.array[wp.float64],
) -> None:
    # One row of the scalar heat method's three operators over a mesh operator pattern, one thread
    # per vertex: the ``float64`` cotangent Laplacian ``L`` (``laplacian.cotmatrix_rows``' entries,
    # ``cotmatrix_offdiagonal``), the heat system ``M + factor L`` (``linalg.shift_factor`` and
    # ``shifted_scalar_entry``, ``factor = -t``) and the Poisson operator ``-L``
    # (``linalg.shifted_system_values``' with ``negate`` set). The three are formed from the one
    # walk, so ``L`` is not read back between them. ``out_mass`` is the lumped mass,
    # ``heat_face_terms``' exact ``area_sums`` over ``mass_divisor`` -- three for the barycentric
    # lumping (``scatter.scale_face_sums``), one for the Voronoi shares, already the mass -- and
    # may be ``area_sums`` itself: each thread reads its own vertex's sum before it writes its mass.
    i = wp.int32(wp.tid())
    m = area_sums[i] / mass_divisor
    out_mass[i] = m
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    factor = shift_factor(scale, edge_sum_and_count, from_edges)
    total = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(start, end):
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
        wp.atomic_add(out_n_rim, 0, 1)


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
    transport_angles: wp.array[wp.float32],
    mass: wp.array[wp.float64],
    scale: wp.float64,
    edge_sum_and_count: wp.array[wp.float64],
    from_edges: wp.int32,
    out_values: wp.array[wp.mat22d],
) -> None:
    # One row of the vector heat system ``M + factor L_connection`` over a mesh operator pattern:
    # ``laplacian.connection_laplacian_rows``' entries (``connection_offdiagonal``, the diagonal
    # ``sum w * I``) shifted as they are formed (``linalg.shift_factor`` and
    # ``shifted_block_entry``, ``factor = t``), so the connection Laplacian itself is never stored.
    # ``mass`` is the scalar lumped mass, which both of a vertex's tangent unknowns take.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        return
    factor = shift_factor(scale, edge_sum_and_count, from_edges)
    m = mass[i]
    weight = wp.float64(0.0)
    slot = wp.int32(-1)
    for k in range(start, end):
        if columns[k] == i:
            slot = k
            continue
        value, weight = connection_offdiagonal(
            i, k, run_start, keys, count, order, faces, cot_entries, transport_angles, weight
        )
        out_values[k] = shifted_block_entry(value, factor, False, m)
    if slot >= 0:
        # The flag read off the pattern for the reason ``heat_system_rows`` gives.
        diagonal = weight * wp.mat22d(1.0, 0.0, 0.0, 1.0)
        out_values[slot] = shifted_block_entry(diagonal, factor, columns[slot] == i, m)


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
    tail = faces[3 * f + corner]
    tip = faces[3 * f + (corner + 1) % 3]
    apex = faces[3 * f + (corner + 2) % 3]
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
    scatter_add_vector(out_source, sums, e, wp.vec2d(wp.float64(0.0), along))


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
) -> None:
    # The edge discretization's stage 3, the counterpart of ``vertex_field_divergence``: face
    # ``f``'s vector is the sum of its three edges' values expanded from their frames into world
    # space (the face's own quarter turn for the imaginary axis), normalized (zero only where it is
    # exactly zero), and integrated by the same thread. Accumulates the negated divergence by
    # integrating ``-X``, as ``vertex_field_divergence`` does.
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
    tangent_angles: wp.array[wp.float32], transport_angles: wp.array[wp.float32], h: wp.int32
) -> wp.float64:
    # The polar angle of halfedge ``h``'s reverse ``j -> i`` in ``j``'s frame, read off ``h``
    # alone: transport along ``h`` turns ``h``'s own direction into the reverse's opposite. Exists
    # for a boundary edge too, whose reverse is no face's halfedge.
    return wp.float64(tangent_angles[h]) + wp.float64(transport_angles[h]) - PI_D


@wp.kernel
def seed_log_map_outward_ball(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    tangent_angles: wp.array[wp.float32],
    transport_angles: wp.array[wp.float32],
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
        out_radial[c] = out_radial[c] + complex_product(
            toward_c, polar_unit(wp.float64(tangent_angles[ca]))
        )

        height_a = twice_area / wp.length(pc - pb)
        alpha = HALF_PI_D - gamma
        at_source = wp.vec2d(
            -(theta * wp.cos(alpha) + wp.cos(alpha - theta) * sine) / (wp.float64(2.0) * height_a),
            -(wp.cos(alpha) - wp.cos(alpha - wp.float64(2.0) * theta) + wp.float64(2.0) * theta
              * wp.sin(alpha)) / (wp.float64(4.0) * height_a),
        )  # fmt: skip
        out_radial[source] = out_radial[source] + complex_product(
            at_source, polar_unit(wp.float64(tangent_angles[ab]))
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
    tangent_angles: wp.array[wp.float32],
    transport_angles: wp.array[wp.float32],
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
    back = complex_product(radial_j, polar_unit(-wp.float64(transport_angles[h])))
    length = wp.length(to_vec3d(vertices[j]) - to_vec3d(vertices[i]))
    edge = length * polar_unit(wp.float64(tangent_angles[h]))
    return half_cotangent * wp.dot(wp.float64(0.5) * (radial_i + back), edge)


@wp.kernel
def radial_field_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    tangent_angles: wp.array[wp.float32],
    transport_angles: wp.array[wp.float32],
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
                f64,
                f64,
                f64,
                f64,
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
                wp.array[wp.float32],
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
