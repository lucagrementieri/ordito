import warp as wp

from ordito.kernels.array import update_argmax_lowest_index
from ordito.kernels.predicates import barycentric_2d
from ordito.kernels.triangles import corner_triple, face_vertices

wp.set_module_options({"enable_backward": False})

# NaN payload for vertices whose UV is non-finite (never sampled) and out-of-bounds reads.
NAN_F32 = wp.constant(wp.float32(float("nan")))
# Inclusive coverage tolerance so shared triangle edges are not dropped (avoids seam gaps).
COVERAGE_EPS = wp.constant(wp.float32(1e-6))


@wp.func
def _sanitize_uv(uv: wp.vec2) -> wp.vec2:
    """Replace a non-finite UV with the origin so the rasterizer never sees NaN/Inf."""
    if not wp.isfinite(uv[0]) or not wp.isfinite(uv[1]):
        return wp.vec2(0.0, 0.0)
    return uv


@wp.func
def _uv_to_pixel(uv: wp.vec2, resolution: wp.int32) -> wp.vec2:
    """
    Map a UV in ``[0, 1]`` to its pixel-center coordinate ``(x=col, y=row)``.

    Inverse of the sampling convention in the ``sample_*`` kernels: ``col = u * W - 0.5``
    and ``row = (1 - v) * H - 0.5`` (row 0 corresponds to ``v = 1``).
    """
    size = wp.float32(resolution)
    return wp.vec2(uv[0] * size - 0.5, (1.0 - uv[1]) * size - 0.5)


@wp.func
def _covered(bary: wp.vec3) -> wp.bool:
    """Return whether a pixel center lies inside the triangle (edge-inclusive)."""
    return bary[0] >= -COVERAGE_EPS and bary[1] >= -COVERAGE_EPS and bary[2] >= -COVERAGE_EPS


@wp.func
def _face_pixels(
    uv: wp.array[wp.vec2], faces: wp.array[wp.int32], f: wp.int32, resolution: wp.int32
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
    """Pixel-center coordinates of the three (sanitized) corner UVs of face ``f``."""
    uv0, uv1, uv2 = face_vertices(uv, faces, f)
    q0 = _uv_to_pixel(_sanitize_uv(uv0), resolution)
    q1 = _uv_to_pixel(_sanitize_uv(uv1), resolution)
    q2 = _uv_to_pixel(_sanitize_uv(uv2), resolution)
    return q0, q1, q2


@wp.func
def _pixel_bounds(
    q0: wp.vec2, q1: wp.vec2, q2: wp.vec2, resolution: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    """Clamped raster bounds ``(row_lo, row_hi, col_lo, col_hi)`` of the triangle's bbox."""
    # wp.min / wp.max on vectors are element-wise, so these are the bbox corners in UV space.
    lower = wp.min(q0, wp.min(q1, q2))
    upper = wp.max(q0, wp.max(q1, q2))

    col_lo = wp.clamp(wp.int32(wp.floor(lower[0])), wp.int32(0), resolution - 1)
    col_hi = wp.clamp(wp.int32(wp.ceil(upper[0])), wp.int32(0), resolution - 1)
    row_lo = wp.clamp(wp.int32(wp.floor(lower[1])), wp.int32(0), resolution - 1)
    row_hi = wp.clamp(wp.int32(wp.ceil(upper[1])), wp.int32(0), resolution - 1)
    return row_lo, row_hi, col_lo, col_hi


@wp.func
def _face_pixel_window(
    uv: wp.array[wp.vec2], faces: wp.array[wp.int32], f: wp.int32, resolution: wp.int32
) -> tuple[wp.vec2, wp.vec2, wp.vec2, wp.int32, wp.int32, wp.int32, wp.int32]:
    """Compute the face's three pixel-space corners and the raster window to scan for them."""
    # The two always travel together -- all three rasterization kernels below open with this pair
    # and then loop over exactly that window -- and pairing them here is what keeps the owner pass
    # and the two write passes scanning the *same* pixels, which is what makes ``owner`` a valid
    # per-pixel filter for both.
    q0, q1, q2 = _face_pixels(uv, faces, f, resolution)
    row_lo, row_hi, col_lo, col_hi = _pixel_bounds(q0, q1, q2, resolution)
    return q0, q1, q2, row_lo, row_hi, col_lo, col_hi


@wp.kernel
def rasterize_owner(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    resolution: wp.int32,
    out_owner: wp.array2d[wp.int32],
) -> None:
    """Resolve per-pixel triangle ownership: the lowest face index covering each pixel wins."""
    f = wp.int32(wp.tid())
    q0, q1, q2, row_lo, row_hi, col_lo, col_hi = _face_pixel_window(uv, faces, f, resolution)

    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            bary = barycentric_2d(q0, q1, q2, wp.vec2(wp.float32(col), wp.float32(row)))
            if _covered(bary):
                wp.atomic_min(out_owner, row, col, f)


@wp.kernel
def rasterize_scatter(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    attribute: wp.array2d[wp.float32],
    n_channels: wp.int32,
    owner: wp.array2d[wp.int32],
    out_image: wp.array3d[wp.float32],
) -> None:
    """Write the barycentric-interpolated attribute at every pixel this face owns."""
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    resolution = out_image.shape[0]
    q0, q1, q2, row_lo, row_hi, col_lo, col_hi = _face_pixel_window(uv, faces, f, resolution)

    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            if owner[row, col] != f:
                continue
            bary = barycentric_2d(q0, q1, q2, wp.vec2(wp.float32(col), wp.float32(row)))
            for k in range(n_channels):
                out_image[row, col, k] = (
                    bary[0] * attribute[i0, k]
                    + bary[1] * attribute[i1, k]
                    + bary[2] * attribute[i2, k]
                )


@wp.kernel(enable_backward=False)
def rasterize_labels(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    owner: wp.array2d[wp.int32],
    out_labels: wp.array2d[wp.int32],
) -> None:
    """
    Write the argmax-of-barycentric-weight label at every pixel this face owns.

    Equivalent to one-hot encoding the vertex labels, barycentrically interpolating, and taking
    the per-pixel argmax: label weights of shared vertices sum, and ties resolve to the lowest
    label value (matching ``numpy.argmax`` over one-hot columns).
    """
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    resolution = out_labels.shape[0]
    q0, q1, q2, row_lo, row_hi, col_lo, col_hi = _face_pixel_window(uv, faces, f, resolution)

    l0 = labels[i0]
    l1 = labels[i1]
    l2 = labels[i2]

    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            if owner[row, col] != f:
                continue
            bary = barycentric_2d(q0, q1, q2, wp.vec2(wp.float32(col), wp.float32(row)))
            # Per-class weight = sum of barycentric weights of vertices sharing that class.
            s0 = bary[0] + wp.where(l1 == l0, bary[1], 0.0) + wp.where(l2 == l0, bary[2], 0.0)
            s1 = bary[1] + wp.where(l0 == l1, bary[0], 0.0) + wp.where(l2 == l1, bary[2], 0.0)
            s2 = bary[2] + wp.where(l0 == l2, bary[0], 0.0) + wp.where(l1 == l2, bary[1], 0.0)
            best_label = l0
            best_weight = s0
            update_argmax_lowest_index(best_weight, best_label, s1, l1)
            update_argmax_lowest_index(best_weight, best_label, s2, l2)
            out_labels[row, col] = best_label


@wp.func
def _uv_out_of_range(uv: wp.vec2) -> wp.bool:
    """Return whether a *finite* UV lies outside ``[0, 1]`` (a non-finite one is never flagged)."""
    u = uv[0]
    w = uv[1]
    if wp.isfinite(u) and wp.isfinite(w):
        return u < 0.0 or u > 1.0 or w < 0.0 or w > 1.0
    return False


@wp.kernel
def check_uv_range(uv: wp.array[wp.vec2], out_flag: wp.array[wp.int32]) -> None:
    """Set ``out_flag[0] = 1`` if any finite UV lies outside ``[0, 1]``."""
    v = wp.int32(wp.tid())
    if _uv_out_of_range(uv[v]):
        wp.atomic_max(out_flag, 0, wp.int32(1))


@wp.kernel
def check_uv_range_and_labels(
    uv: wp.array[wp.vec2], labels: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    """
    Set ``out_flags[0] = 1`` for an out-of-range finite UV and ``out_flags[1] = 1`` for a label < 0.

    ``check_uv_range`` plus the discrete rasterizer's own label test, over the same vertices: the
    two validations share a launch and, more to the point, one two-slot readback, where running
    them apart cost a device reduction and a readback each.
    """
    v = wp.int32(wp.tid())
    if _uv_out_of_range(uv[v]):
        wp.atomic_max(out_flags, 0, wp.int32(1))
    if labels[v] < 0:
        wp.atomic_max(out_flags, 1, wp.int32(1))


SAMPLE_NEAREST = wp.constant(wp.int32(0))
SAMPLE_BILINEAR = wp.constant(wp.int32(1))


@wp.func
def _texel_coordinate(
    u: wp.float32, w: wp.float32, height: wp.int32, width: wp.int32
) -> tuple[wp.float32, wp.float32]:
    """Continuous ``(row, col)`` of a UV: ``row = (1 - v) * H - 0.5``, ``col = u * W - 0.5``."""
    return (1.0 - w) * wp.float32(height) - 0.5, u * wp.float32(width) - 0.5


@wp.func
def _nearest_texel(
    row: wp.float32, col: wp.float32, height: wp.int32, width: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """Return the edge-clamped texel nearest a continuous ``(row, col)``."""
    r = wp.clamp(wp.int32(wp.floor(row + 0.5)), wp.int32(0), height - 1)
    c = wp.clamp(wp.int32(wp.floor(col + 0.5)), wp.int32(0), width - 1)
    return r, c


@wp.kernel
def sample_texture(
    uv: wp.array[wp.vec2],
    image: wp.array3d[wp.float32],
    n_channels: wp.int32,
    mode: wp.int32,
    out_values: wp.array2d[wp.float32],
) -> None:
    """Sample the texture at each vertex UV (edge-clamped), nearest or bilinear per ``mode``."""
    v = wp.int32(wp.tid())
    u = uv[v][0]
    w = uv[v][1]
    height = image.shape[0]
    width = image.shape[1]
    if not wp.isfinite(u) or not wp.isfinite(w):
        for k in range(n_channels):
            out_values[v, k] = NAN_F32
        return
    row, col = _texel_coordinate(u, w, height, width)
    if mode == SAMPLE_NEAREST:
        r, c = _nearest_texel(row, col, height, width)
        for k in range(n_channels):
            out_values[v, k] = image[r, c, k]
    else:
        r0 = wp.int32(wp.floor(row))
        c0 = wp.int32(wp.floor(col))
        fr = row - wp.float32(r0)
        fc = col - wp.float32(c0)
        r0c = wp.clamp(r0, wp.int32(0), height - 1)
        r1c = wp.clamp(r0 + 1, wp.int32(0), height - 1)
        c0c = wp.clamp(c0, wp.int32(0), width - 1)
        c1c = wp.clamp(c0 + 1, wp.int32(0), width - 1)
        for k in range(n_channels):
            top = wp.lerp(image[r0c, c0c, k], image[r0c, c1c, k], fc)
            bottom = wp.lerp(image[r1c, c0c, k], image[r1c, c1c, k], fc)
            out_values[v, k] = wp.lerp(top, bottom, fr)


@wp.kernel
def sample_class_image(
    uv: wp.array[wp.vec2], class_image: wp.array2d[wp.int32], out_labels: wp.array[wp.int32]
) -> None:
    """
    Read an ``int32`` class image nearest-neighbour at each vertex UV; ``-1`` for an Inf / NaN UV.

    ``sample_texture``'s ``SAMPLE_NEAREST`` branch over integer storage, sharing its texel helpers,
    so the labels are read as they are rather than through a ``float32`` copy of the whole image
    and a rounding pass back -- which also keeps a label above ``2**24`` exact.
    """
    v = wp.int32(wp.tid())
    u = uv[v][0]
    w = uv[v][1]
    if not wp.isfinite(u) or not wp.isfinite(w):
        out_labels[v] = wp.int32(-1)
        return
    height = class_image.shape[0]
    width = class_image.shape[1]
    row, col = _texel_coordinate(u, w, height, width)
    r, c = _nearest_texel(row, col, height, width)
    out_labels[v] = class_image[r, c]
