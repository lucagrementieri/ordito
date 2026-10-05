from typing import Any

import warp as wp

from ordito.kernels.array import OverloadTable, trilinear_cell, trilinear_corner, trilinear_weight
from ordito.kernels.neighbors import in_ball
from ordito.kernels.triangles import face_vertices, point_barycentric

wp.set_module_options({"enable_backward": False})


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    x0, x1, x2 = face_vertices(vertex_values, faces, f)
    out_face_values[f] = (x0 + x1 + x2) / wp.float32(3.0)


@wp.kernel
def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[Any],
    closest: wp.array[wp.vec3],
    face_id: wp.array[wp.int32],
    out_values: wp.array[Any],
    out_distance: wp.array[wp.float32],
) -> None:
    # Barycentric resample of a source per-vertex field at each target vertex's closest point.
    #
    # ``face_id < 0`` is the ``max_dist`` miss case. The value slot keeps whatever the wrapper
    # pre-filled (so a caller who narrowed the search still gets a deterministic buffer), and the
    # distance is promoted to ``inf``: ``wp.mesh_query_point_no_sign`` reports ``max_dist`` there,
    # which is indistinguishable from a genuine hit at exactly that range.
    i = wp.int32(wp.tid())
    f = face_id[i]
    if f < 0:
        out_distance[i] = wp.float32(wp.INF)
        return
    #
    # ``point_barycentric``'s conditioning is load-bearing here, not incidental: a source face thin
    # enough for the Gram-determinant form to cancel to zero in float32 would write ``nan`` into the
    # field while ``out_distance[i]`` stayed finite, so the confidence measure this function returns
    # would report a healthy hit on a poisoned value. Slivers are ordinary in a decimated or
    # reconstructed source mesh, which is exactly the input this transfer exists for -- that is the
    # defect that got the Gram form removed from the package outright.
    v0, v1, v2 = face_vertices(source_vertices, source_faces, f)
    bary = point_barycentric(v0, v1, v2, closest[i])
    a0, a1, a2 = face_vertices(source_values, source_faces, f)
    out_values[i] = a0 * bary[0] + a1 * bary[1] + a2 * bary[2]


@wp.kernel
def apply_transfer_operator(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    source_values: wp.array[Any],
    out_values: wp.array[Any],
) -> None:
    # One CSR row per output element: the field's value there is the weighted sum of the source
    # values the row references. Written rather than delegated to ``warp.sparse.bsr_mv``, which
    # requires the vector's dtype to match the matrix's 1x1 ``float32`` block, so it can move a
    # scalar field and nothing else -- where an attribute is as often a ``wp.vec3`` colour or a
    # ``wp.vec2`` UV, and those are exactly what a topology edit drops today.
    #
    # The accumulator is seeded as ``source_values[0] - source_values[0]``, which is how a generic
    # ``Any`` kernel spells "the additive identity of this dtype": there is no ``dtype(0)``
    # constructor to reach for, and ``x * wp.float32(0.0)`` would not parse for a dtype whose scalar
    # is not float32 (Warp requires both operands' scalars to match). It also gives an empty row --
    # an output element no source reaches -- the right answer rather than uninitialized memory.
    #
    # The weights being ``float32`` is what bounds the dtype set: the product below needs the
    # field's own scalar to be float32 too, so ``wp.float32`` / ``wp.vec2`` / ``wp.vec3`` work and a
    # ``float64`` field does not. That is not a gap -- these operators are assembled from float32
    # vertex data, so a float64 field carried through one would advertise precision the weights do
    # not have.
    row = wp.int32(wp.tid())
    total = source_values[0] - source_values[0]
    for k in range(offsets[row], offsets[row + 1]):
        total = total + source_values[columns[k]] * weights[k]
    out_values[row] = total


@wp.func
def gaussian_kernel_weight(distance: wp.float32, inverse_scale: wp.float32) -> wp.float32:
    # ``exp(-(sharpness * d / radius) ** 2)``, VTK's ``vtkGaussianKernel``, with the caller having
    # folded ``sharpness / radius`` into one reciprocal length. An unused neighbour slot arrives at
    # distance ``inf`` and weighs exactly 0.
    scaled = distance * inverse_scale
    return wp.exp(-scaled * scaled)


@wp.func
def accumulate_gaussian_neighbor(
    source_values: wp.array[Any],
    index: wp.int32,
    distance: wp.float32,
    inverse_scale: wp.float32,
    accumulated: Any,
    total: wp.float32,
    coincident: wp.int32,
) -> tuple[Any, wp.float32, wp.int32]:
    # Fold one neighbour into a query's Gaussian-weighted mean. Both interpolation kernels call it,
    # the one reading a neighbour CSR and the one walking the hash grid itself, so the two agree
    # to the bit whenever they visit the same neighbours in the same order.
    if distance <= wp.float32(0.0):
        coincident = index
    weight = gaussian_kernel_weight(distance, inverse_scale)
    return accumulated + source_values[index] * weight, total + weight, coincident


@wp.func
def write_gaussian_mean(
    source_values: wp.array[Any],
    q: wp.int32,
    accumulated: Any,
    total: wp.float32,
    coincident: wp.int32,
    out_values: wp.array[Any],
) -> None:
    # VTK returns the source's own value where a query sits on a data point, rather than blending
    # its neighbours in; the interpolant is then exact at the data. A query with no neighbours, or
    # whose every weight underflowed, is left at the null value the wrapper pre-filled -- so the
    # miss case needs no per-dtype null argument here.
    if coincident >= 0:
        out_values[q] = source_values[coincident]
    elif total > wp.float32(0.0):
        out_values[q] = accumulated / total


@wp.kernel
def interpolate_from_points_nearest(
    source_values: wp.array[Any],
    neighbor_indices: wp.array[wp.int32],
    neighbor_distances: wp.array[wp.float32],
    k: wp.int32,
    inverse_scale: wp.float32,
    out_values: wp.array[Any],
) -> None:
    # Gaussian-weighted mean of one query's ``k`` nearest sources, over the flattened ``(m, k)``
    # rows ``query_nearest`` returns; a slot no neighbour was found for carries index -1 and is
    # skipped. The radius footprint walks the grid in ``interpolate_from_points_in_ball`` instead,
    # which differs from this only in where the neighbours come from.
    q = wp.int32(wp.tid())
    # A zero of the field's own dtype, which a generic kernel cannot spell any other way.
    accumulated = source_values[0] * wp.float32(0.0)
    total = wp.float32(0.0)
    coincident = wp.int32(-1)
    for slot in range(q * k, (q + 1) * k):
        index = neighbor_indices[slot]
        if index < 0:
            continue
        accumulated, total, coincident = accumulate_gaussian_neighbor(
            source_values,
            index,
            neighbor_distances[slot],
            inverse_scale,
            accumulated,
            total,
            coincident,
        )
    write_gaussian_mean(source_values, q, accumulated, total, coincident, out_values)


@wp.kernel
def interpolate_from_points_in_ball(
    source_points: wp.array[wp.vec3],
    source_values: wp.array[Any],
    query_points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    inverse_scale: wp.float32,
    out_values: wp.array[Any],
) -> None:
    # The Gaussian-weighted mean over every source within ``radius``, reduced during the hash-grid
    # walk instead of over a ball-query CSR. The walk, the predicate (``in_ball``) and the distance
    # are ``kernels.neighbors.ball_collect``'s hash-grid branch, so a query folds exactly the
    # neighbours, in exactly the order, that CSR would have listed -- with no count pass, scan,
    # readback or two neighbour-sized buffers in between. What still differs is the body: the
    # collect writes each neighbour out, this folds it in.
    q = wp.int32(wp.tid())
    p = query_points[q]
    accumulated = source_values[0] * wp.float32(0.0)
    total = wp.float32(0.0)
    coincident = wp.int32(-1)
    j = wp.int32(0)
    query = wp.hash_grid_query(grid_id, p, radius)
    while wp.hash_grid_query_next(query, j):
        offset = source_points[j] - p
        if in_ball(offset, radius):
            accumulated, total, coincident = accumulate_gaussian_neighbor(
                source_values, j, wp.length(offset), inverse_scale, accumulated, total, coincident
            )
    write_gaussian_mean(source_values, q, accumulated, total, coincident, out_values)


@wp.kernel(enable_backward=False)
def sample_grid_trilinear(
    field: wp.array3d[Any],
    lower: wp.vec3,
    inverse_spacing: wp.vec3,
    points: wp.array[wp.vec3],
    out_values: wp.array[Any],
) -> None:
    # Read a dense lattice at one arbitrary position, as the trilinearly weighted sum of the eight
    # corners around it -- the transpose of ``kernels/scatter.splat_grid_trilinear``. The wrapper's
    # ``Notes`` records which round trips through the pair are exact; averaging makes the two
    # adjoint rather than inverse.
    #
    # Summed rather than nested as ``wp.lerp``. ``kernels/reconstruction.poisson_sample_grid`` is
    # the lerp spelling over a flat ``res**3`` float32 buffer; the two are the same function and
    # differ only in float32 summation order, and they are deliberately kept apart because that
    # one's arithmetic is what the Poisson iso-value was measured against. All three share
    # ``trilinear_cell``.
    #
    # A position outside the lattice reads the nearest cell's stencil rather than a null value,
    # because ``trilinear_cell`` clamps: the field is extended by its boundary cells, which is what
    # a signed-distance or density lattice wants and is the convention ``poisson_sample_grid``
    # already had.
    s = wp.int32(wp.tid())
    shape = wp.vec3i(field.shape[0], field.shape[1], field.shape[2])
    base, next_corner, fractions = trilinear_cell(
        wp.cw_mul(points[s] - lower, inverse_spacing), shape
    )
    accumulator = out_values.dtype(0.0)
    for offset_x in range(2):
        for offset_y in range(2):
            for offset_z in range(2):
                corner = trilinear_corner(base, next_corner, wp.vec3i(offset_x, offset_y, offset_z))
                accumulator += (
                    trilinear_weight(fractions, offset_x, offset_y, offset_z)
                    * field[corner[0], corner[1], corner[2]]
                )
    out_values[s] = accumulator


# Concrete overloads, registered at import -- rationale in ``ordito/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5.
#
# Field dtypes the three transfer kernels admit, and the whole set: every one of them scales the
# field by a ``float32`` weight and sums, and Warp requires both operands of that product to share a
# scalar type, so the admissible fields are exactly the ``float32``-scalar ones this package
# transfers -- a scalar, a ``wp.vec2`` UV, a ``wp.vec3`` position / normal / colour. The three
# tables register the same set because a caller who can move a UV through
# ``transfer_through_operator`` must be able to move it through the closest-point transfer too; the
# operator path is documented as the preferred one *when an operator exists*, which makes the other
# two its fallback rather than a narrower API. A ``float64`` field is deliberately absent -- these
# operators are assembled from ``float32`` vertex data, so carrying one would advertise precision
# the weights do not have.
_FIELD_DTYPES = (wp.float32, wp.vec2, wp.vec3)

# The concrete handles keyed by the caller's value dtype -- see
# [`OverloadTable`][ordito.kernels.array.OverloadTable].
APPLY_TRANSFER_OPERATOR: OverloadTable
SAMPLE_GRID_TRILINEAR: OverloadTable
TRANSFER_ONTO_VERTICES: OverloadTable
INTERPOLATE_FROM_POINTS_NEAREST: OverloadTable
INTERPOLATE_FROM_POINTS_IN_BALL: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global APPLY_TRANSFER_OPERATOR, SAMPLE_GRID_TRILINEAR
    global TRANSFER_ONTO_VERTICES, INTERPOLATE_FROM_POINTS_NEAREST, INTERPOLATE_FROM_POINTS_IN_BALL
    APPLY_TRANSFER_OPERATOR = OverloadTable(
        apply_transfer_operator,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.float32],
                wp.array[d],
                wp.array[d],
            ]
            for d in _FIELD_DTYPES
        },
    )
    SAMPLE_GRID_TRILINEAR = OverloadTable(
        sample_grid_trilinear,
        {
            d: [wp.array3d[d], wp.vec3, wp.vec3, wp.array[wp.vec3], wp.array[d]]
            for d in (wp.float32, wp.vec3)
        },
    )
    TRANSFER_ONTO_VERTICES = OverloadTable(
        transfer_onto_vertices,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.float32],
            ]
            for d in _FIELD_DTYPES
        },
    )
    INTERPOLATE_FROM_POINTS_NEAREST = OverloadTable(
        interpolate_from_points_nearest,
        {
            d: [
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.float32],
                wp.int32,
                wp.float32,
                wp.array[d],
            ]
            for d in _FIELD_DTYPES
        },
    )
    INTERPOLATE_FROM_POINTS_IN_BALL = OverloadTable(
        interpolate_from_points_in_ball,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[d],
                wp.array[wp.vec3],
                wp.uint64,
                wp.float32,
                wp.float32,
                wp.array[d],
            ]
            for d in _FIELD_DTYPES
        },
    )


_register_overloads()
