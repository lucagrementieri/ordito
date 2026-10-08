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

import warp as wp

from ordito.constants import TOLERANCE_ZERO_CONSTANT
from ordito.kernels.array import binary_search_index, cross2, to_vec2, to_vec2d, to_vec3, to_vec3d
from ordito.kernels.predicates import (
    stable_length,
    stable_normalize,
    unit_tangent,
    world_to_tangent,
)
from ordito.kernels.reduce import block_chunk_1d, commit_sum_and_count
from ordito.kernels.scatter import add_corner_triple
from ordito.kernels.triangles import face_unit_gradient, face_vertices_vec3d

wp.set_module_options({"enable_backward": False})


@wp.kernel
def upper_edge_length_sum_and_count(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_sum_and_count: wp.array[wp.float64],
) -> None:
    # Sum and count of the edge lengths over an operator's *strict upper triangle*, which for the
    # heat method's Laplacians -- one entry per edge, nothing pruned -- is
    # exactly the mesh's unique edge set. The timestep ``h ** 2`` needs only their mean, and the
    # operator already exists, so the edge set comes free where ``edges_unique`` would re-sort
    # every edge of the mesh to recover it. Reads the sparsity only, so the scalar and the
    # vector solver get the identical number from their two different operators.
    #
    # ``reduce``'s mask-shaped block fold over *rows*: lanes stride the block's rows by
    # ``wp.block_dim()`` (right on the CPU device too), and one tile sum per quantity commits.
    i, t = wp.tid()
    n = offsets.shape[0] - 1
    base, remaining = block_chunk_1d(n, i)
    if remaining <= 0:
        return
    total = wp.float64(0.0)
    count = wp.float64(0.0)
    for k in range(t, remaining, wp.block_dim()):
        row = base + k
        for entry in range(offsets[row], offsets[row + 1]):
            column = columns[entry]
            if column > row:
                total += wp.float64(wp.length(vertices[column] - vertices[row]))
                count += wp.float64(1.0)
    commit_sum_and_count(t, total, count, out_sum_and_count)


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


@wp.kernel
def seed_source_indicator(sources: wp.array[wp.int32], out_u0: wp.array[wp.float64]) -> None:
    # Set the initial heat to 1 at each source vertex (out_u0 pre-zeroed by the caller).
    t = wp.int32(wp.tid())
    out_u0[sources[t]] = wp.float64(1.0)


@wp.func
def accumulate_face_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    f: wp.int32,
    x: wp.vec3d,
    out_div: wp.array[wp.float64],
) -> None:
    # Cotangent integrated divergence of one face's vector ``x``, accumulated onto its three
    # vertices. cot_entries[f, k] = 1/2 cot(angle at corner k); each vertex gets contributions from
    # the two edges of the triangle incident to it, weighted by the cotangent opposite those edges.
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

    add_corner_triple(out_div, faces, f, d0, d1, d2)


@wp.kernel
def unit_gradient_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    cot_entries: wp.array2d[wp.float32],
    out_div: wp.array[wp.float64],
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
    # only this module's callers would ever set (§4.2, speculative generality: one caller per
    # mode). The two also return different quantities: a gradient carries the field's rate of
    # change, this carries only a direction.
    f = wp.int32(wp.tid())
    x = face_unit_gradient(vertices, faces, normals, areas, values, f)
    accumulate_face_divergence(vertices, faces, cot_entries, f, x, out_div)


# --------------------------------------------------------------------------------------
# Signed heat method: signed distance to oriented curves (Feng & Crane 2024)
# --------------------------------------------------------------------------------------


@wp.kernel
def source_and_global_sums(
    phi: wp.array[wp.float64], sources: wp.array[wp.int32], out_sums: wp.array[wp.float64]
) -> None:
    # ``(sum of phi over the sources, sum of phi over every vertex)`` into ``out_sums`` (zeroed by
    # the caller), for ``shift_and_orient``: both means the heat method's normalization needs, in
    # one launch. The ``reduce`` block fold over whichever of the two ranges the launch covers; a
    # block past the end of the other contributes zero to it, so a launch sized to the sources
    # alone gives a correct first sum -- all ``shift_and_orient`` reads when it does not orient.
    # ``sources`` may repeat a vertex, which is then counted as often as it appears, as a gathered
    # mean counts it.
    i, t = wp.tid()
    base, remaining = block_chunk_1d(phi.shape[0], i)
    source_base, source_remaining = block_chunk_1d(sources.shape[0], i)
    at_sources = wp.float64(0.0)
    everywhere = wp.float64(0.0)
    for k in range(t, source_remaining, wp.block_dim()):
        at_sources += phi[sources[source_base + k]]
    for k in range(t, remaining, wp.block_dim()):
        everywhere += phi[base + k]
    commit_sum_and_count(t, at_sources, everywhere, out_sums)


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
    k = wp.int32(wp.tid())
    begin = wp.int32(0)
    end = curve_vertices.shape[0]
    if single_curve == 0:
        curve = binary_search_index(curve_offsets, k) - 1
        if curve < 0 or curve >= curve_offsets.shape[0] - 1:
            return
        begin = curve_offsets[curve]
        # Clamped to the buffer, as slicing it by a row past its end would be.
        end = wp.min(curve_offsets[curve + 1], end)
    a = curve_vertices[k]
    b = a
    if k + 1 < end:
        b = curve_vertices[k + 1]
    elif closed != 0 and end - begin >= 2:
        b = curve_vertices[begin]
    else:
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
    accumulate_face_divergence(vertices, faces, cot_entries, f, x, out_div)


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
# Vector heat method: transport, scalar extension and the log map (Sharp et al. 2019)
# --------------------------------------------------------------------------------------


@wp.kernel
def seed_source_scalars(
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_indicator: wp.array[wp.float64],
    out_weighted: wp.array[wp.float64],
    out_range: wp.array[wp.float64],
) -> None:
    # Scalar extension needs two right-hand sides: where the sources are, and what they carry.
    # ``out_range`` (seeded ``+inf``) also collects the source values' range for
    # ``extend_from_ratio``'s overshoot check, stored as ``(min, -max)`` so both ends are one
    # ``atomic_min`` from one ``wp.full``.
    s = wp.int32(wp.tid())
    v = sources[s]
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
) -> None:
    # ``transport_tangent_vectors``' three right-hand sides from one pass over the sources: the
    # widened vector, where the sources are, and the length each carries -- the vector heat seed and
    # ``seed_source_scalars``' pair, which a caller of both used to write in four launches.
    s = wp.int32(wp.tid())
    v = sources[s]
    vector = to_vec2d(vectors[s])
    wp.atomic_add(out_field, v, vector)
    wp.atomic_add(out_indicator, v, wp.float64(1.0))
    wp.atomic_add(out_magnitude, v, wp.length(vector))


@wp.kernel
def seed_log_map_source(
    source: wp.int32, out_field: wp.array[wp.vec2d], out_indicator: wp.array[wp.float64]
) -> None:
    # ``log_map``'s two right-hand sides: the source's reference direction and its heat indicator.
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
) -> None:
    # The log map's radial direction -- the unit gradient of the distance field, pointing away from
    # the source -- formed and scattered onto vertices by the same thread. The scatter is a plain
    # area-weighted add: the weights are the same for all three corners, so no normalization is
    # needed before projecting.
    f = wp.int32(wp.tid())
    x = face_unit_gradient(vertices, faces, normals, areas, values, f)
    area = areas[f]
    value = to_vec3(x) * area
    add_corner_triple(out_vertex_field, faces, f, value, value, value)


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
