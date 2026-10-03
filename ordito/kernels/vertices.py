import warp as wp

from ordito.kernels.array import OverloadTable
from ordito.kernels.scatter import atomic_add_vec3
from ordito.kernels.triangles import (
    corner_triple,
    face_corner_angles,
    face_normals_and_area,
    triangle_cross,
)


@wp.func
def max_corner_inverse_edge_length_sq(
    vertices: wp.array[wp.vec3], i0: wp.int32, i1: wp.int32, i2: wp.int32
) -> wp.float32:
    """
    Per-corner MWSELR factor ``1 / (||e1||^2 * ||e2||^2)``.

    **The zero test is against zero and not against a tolerance**, for the same reason
    [`face_normals_and_area`][ordito.kernels.triangles.face_normals_and_area] states at length:
    this denominator scales as ``h^4``, so an absolute floor of ``TOLERANCE_ZERO_CONSTANT``
    (which this used to carry) rejects *every* corner of a small enough mesh and hands back a
    weight of zero for all of them -- so ``vertex_normals(weighting="mwselr")`` returned an
    all-zero table on a cleanly scaled sphere while ``"area"`` and ``"angle"`` returned correct
    unit normals from the same buffers, and the zero-row contract those callers document covers an
    unreferenced vertex or a cancelling fan, not a clean sphere. ``1 / denom`` is well conditioned
    for any positive ``denom``, so the tolerance was never protecting the division; at
    ``denom == 0`` the corner is exactly degenerate and both branches want zero.

    The remaining range is ``float32``'s, not this test's, and the single test covers both ends:
    ``denom`` underflows to zero below a very short edge, which this branch reads as degenerate,
    and overflows to infinity above a very long one, where ``1 / inf`` is the same zero the other
    branch returns. Position storage is ``float32``, so the small end sits below the scale at which
    the corner coordinates carry any digits at all.
    """
    e1 = vertices[i1] - vertices[i0]
    e2 = vertices[i2] - vertices[i0]
    denom = wp.length_sq(e1) * wp.length_sq(e2)
    if denom > wp.float32(0.0):
        return wp.float32(1.0) / denom
    return wp.float32(0.0)


@wp.func
def normalized_accumulated_row(sums: wp.array2d[wp.Float], i: wp.int32) -> wp.vec3:
    """
    Row ``i`` of an ``(n, 3)`` normal accumulator, unit-normalized and narrowed to ``wp.vec3``.

    The zero row comes back as the zero vector. Shared by ``normalize_accumulated_rows`` and by
    the consumers that normalize on read (``smoothing.inflate_displace``) instead of materializing
    the normals.
    """
    x = sums[i, 0]
    y = sums[i, 1]
    z = sums[i, 2]
    length = wp.sqrt(x * x + y * y + z * z)
    if length > sums.dtype(0.0):
        x = x / length
        y = y / length
        z = z / length
    return wp.vec3(wp.float32(x), wp.float32(y), wp.float32(z))


@wp.kernel
def normalize_accumulated_rows(sums: wp.array2d[wp.Float], out_normals: wp.array[wp.vec3]) -> None:
    """
    Unit-normalize each row of an ``(n, 3)`` accumulator into a ``wp.vec3``.

    The tail of ``vertices._accumulate_and_normalize``, and the reason its accumulator can be
    ``float64`` while its answer is ``float32``: it narrows and normalizes in one pass, where the
    ``float32`` accumulator it replaced could reach the same answer with a zero-copy
    ``wp.utils.array_cast`` reinterpretation plus a ``wp.map(wp.normalize, ...)``. One launch
    instead of two, so the wider accumulator costs nothing here.

    Generic over the accumulator's precision, and paired with
    ``kernels.scatter.scatter_sum_vec`` / ``scatter_weighted_sum_vec``, which are generic over the
    same thing -- so the pair takes a second precision as a registration row rather than as a
    second copy of either kernel. The three components are carried as scalars rather than assembled
    into a vector because there is no rank-3 vector type to name generically in kernel scope, and
    the length is the same arithmetic either way.

    A zero row -- an unreferenced vertex, or a fan whose contributions cancel -- comes back as the
    zero vector, which is ``wp.normalize``'s own answer for one (its ``kEps`` is 0) and is the
    contract the callers document.
    """
    i = wp.int32(wp.tid())
    out_normals[i] = normalized_accumulated_row(sums, i)


@wp.func
def add_to_face_corners(
    out_sums: wp.array2d[wp.Float],
    faces: wp.array[wp.int32],
    f: wp.int32,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
) -> None:
    # Accumulate one vector per corner of face ``f`` into the rows of its three vertices, corner 0
    # first -- the order ``kernels.scatter.scatter_sum_vec`` / ``scatter_weighted_sum_vec`` add in,
    # which is what keeps the fused kernels below byte-identical to the scatter they replace on the
    # CPU device, where the atomics serialize.
    base = f * 3
    atomic_add_vec3(out_sums, faces[base], v0)
    atomic_add_vec3(out_sums, faces[base + 1], v1)
    atomic_add_vec3(out_sums, faces[base + 2], v2)


@wp.kernel
def scatter_area_weighted_normals(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_sums: wp.array2d[wp.Float]
) -> None:
    # ``vertex_normals(weighting="area")`` with nothing precomputed: the face normal scaled by its
    # area, scattered onto the corners, with no ``(n_faces,)`` normal, area or product buffer in
    # between -- the composition ``face_normals_and_areas``, ``wp.map(wp.mul)``, ``scatter_sum_vec``
    # in one launch, since each of those reads only its own face.
    f = wp.int32(wp.tid())
    normal, area = face_normals_and_area(vertices, faces, f)
    value = normal * area
    add_to_face_corners(out_sums, faces, f, value, value, value)


@wp.kernel
def scatter_scaled_normals(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    faces: wp.array[wp.int32],
    out_sums: wp.array2d[wp.Float],
) -> None:
    # The same scatter when the caller supplied either table: the product is formed here rather
    # than by a ``wp.map(wp.mul)`` into a buffer only this kernel reads.
    f = wp.int32(wp.tid())
    value = face_normals[f] * face_areas[f]
    add_to_face_corners(out_sums, faces, f, value, value, value)


@wp.kernel
def scatter_corner_angles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_angle_sums: wp.array[wp.float32]
) -> None:
    # Each vertex's sum of incident corner angles, the angles formed in the thread
    # (``triangles.face_corner_angles``, the ``angles`` kernel's body) and added corner by corner in
    # ``scatter.scatter_sum_scalar``'s order, so no ``(n_faces, 3)`` angle table is written.
    f = wp.int32(wp.tid())
    a0, a1, a2 = face_corner_angles(vertices, faces, f)
    wp.atomic_add(out_angle_sums, faces[3 * f + 0], a0)
    wp.atomic_add(out_angle_sums, faces[3 * f + 1], a1)
    wp.atomic_add(out_angle_sums, faces[3 * f + 2], a2)


@wp.kernel
def scatter_angle_weighted_normals(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_sums: wp.array2d[wp.Float]
) -> None:
    # ``vertex_normals(weighting="angle")`` with nothing precomputed: the unit face normal times
    # each corner's interior angle -- the composition ``face_normals_and_areas``,
    # ``triangles.angles``, ``scatter_weighted_sum_vec`` in one launch and no per-face table. The
    # corner angles are ``triangles.face_corner_angles``, the ``angles`` kernel's own body, so the
    # two cannot drift.
    f = wp.int32(wp.tid())
    normal, _area = face_normals_and_area(vertices, faces, f)
    a0, a1, a2 = face_corner_angles(vertices, faces, f)
    add_to_face_corners(out_sums, faces, f, normal * a0, normal * a1, normal * a2)


@wp.func
def add_max_corner_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: wp.int32,
    normal: wp.vec3,
    scale: wp.float32,
    out_sums: wp.array2d[wp.Float],
) -> None:
    # ``vertex_normals(weighting="mwselr")``'s per-corner scatter: the face normal times the MWSELR
    # corner weight times a caller-supplied scale, added corner 0 first. The scale is the face's
    # cross product magnitude when the normal is *unit* (the sine the weight divides out is then
    # not already in the normal) and 1 when the normal is the raw cross. Forming ``scale * weight``
    # in float32 before the product is what the ``(n_faces, 3)`` weight table this replaced stored,
    # so the sum is byte-identical to ``scatter_weighted_sum_vec`` over that table.
    a, b, c = corner_triple(faces, f)
    w0 = scale * max_corner_inverse_edge_length_sq(vertices, a, b, c)
    w1 = scale * max_corner_inverse_edge_length_sq(vertices, b, c, a)
    w2 = scale * max_corner_inverse_edge_length_sq(vertices, c, a, b)
    add_to_face_corners(out_sums, faces, f, normal * w0, normal * w1, normal * w2)


@wp.kernel
def scatter_max_weighted_normals(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_sums: wp.array2d[wp.Float]
) -> None:
    # Nothing supplied: the raw cross product is the normal, so no scale is needed and neither a
    # normal nor a weight table is written.
    f = wp.int32(wp.tid())
    add_max_corner_normals(
        vertices, faces, f, triangle_cross(vertices, faces, f), wp.float32(1.0), out_sums
    )


@wp.kernel
def scatter_max_weighted_unit_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    out_sums: wp.array2d[wp.Float],
) -> None:
    # Caller-supplied unit normals; differs from ``scatter_max_weighted_normals`` only in taking
    # the normal from the table and restoring the cross-product magnitude as the scale.
    f = wp.int32(wp.tid())
    add_max_corner_normals(
        vertices, faces, f, face_normals[f], wp.length(triangle_cross(vertices, faces, f)), out_sums
    )


# The accumulator precision ``vertices._accumulate_and_normalize`` allocates, and the only one
# registered: float64, so that the order the scatter's atomics pick cannot reach the answer
# (``kernels.scatter.atomic_add_vec3`` carries the measurement). Mirrors
# ``kernels/scatter._VECTOR_ACCUMULATOR_DTYPES``, which must list the same set -- the two kernels
# are launched back to back on one buffer.
_ACCUMULATOR_DTYPES = (wp.float64,)

NORMALIZE_ACCUMULATED_ROWS: OverloadTable
SCATTER_AREA_WEIGHTED_NORMALS: OverloadTable
SCATTER_SCALED_NORMALS: OverloadTable
SCATTER_ANGLE_WEIGHTED_NORMALS: OverloadTable
SCATTER_MAX_WEIGHTED_NORMALS: OverloadTable
SCATTER_MAX_WEIGHTED_UNIT_NORMALS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global NORMALIZE_ACCUMULATED_ROWS, SCATTER_AREA_WEIGHTED_NORMALS
    global SCATTER_SCALED_NORMALS, SCATTER_ANGLE_WEIGHTED_NORMALS
    global SCATTER_MAX_WEIGHTED_NORMALS, SCATTER_MAX_WEIGHTED_UNIT_NORMALS
    NORMALIZE_ACCUMULATED_ROWS = OverloadTable(
        normalize_accumulated_rows,
        {d: [wp.array2d[d], wp.array[wp.vec3]] for d in _ACCUMULATOR_DTYPES},
    )
    geometry = [wp.array[wp.vec3], wp.array[wp.int32]]
    SCATTER_AREA_WEIGHTED_NORMALS = OverloadTable(
        scatter_area_weighted_normals, {d: [*geometry, wp.array2d[d]] for d in _ACCUMULATOR_DTYPES}
    )
    SCATTER_ANGLE_WEIGHTED_NORMALS = OverloadTable(
        scatter_angle_weighted_normals, {d: [*geometry, wp.array2d[d]] for d in _ACCUMULATOR_DTYPES}
    )
    SCATTER_MAX_WEIGHTED_NORMALS = OverloadTable(
        scatter_max_weighted_normals, {d: [*geometry, wp.array2d[d]] for d in _ACCUMULATOR_DTYPES}
    )
    SCATTER_MAX_WEIGHTED_UNIT_NORMALS = OverloadTable(
        scatter_max_weighted_unit_normals,
        {d: [*geometry, wp.array[wp.vec3], wp.array2d[d]] for d in _ACCUMULATOR_DTYPES},
    )
    SCATTER_SCALED_NORMALS = OverloadTable(
        scatter_scaled_normals,
        {
            d: [wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32], wp.array2d[d]]
            for d in _ACCUMULATOR_DTYPES
        },
    )


_register_overloads()
