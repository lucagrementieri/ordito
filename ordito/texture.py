"""
Bake per-vertex attributes into UV-space textures and sample them back.

The functions here operate on an *existing* UV map (from a disk parameterization or a mesh
file, e.g. loaded via [`load_mesh`][ordito.io.load_mesh]): they rasterize per-vertex attributes
into a square texture and sample a texture back to per-vertex values, and rasterize which face
covers each pixel of any window of UV space. They are a
Warp software-rasterizer replacement for an OpenGL/moderngl pipeline, keeping all data on the
input Warp device.

Computing a UV map from mesh connectivity (parameterization) is a separate concern and is not
part of this module.

See [`rasterize_attribute`][ordito.texture.rasterize_attribute],
[`rasterize_discrete_attribute`][ordito.texture.rasterize_discrete_attribute],
[`rasterize_face_indices`][ordito.texture.rasterize_face_indices],
[`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv], and
[`remap_discrete_attribute_from_uv`][ordito.texture.remap_discrete_attribute_from_uv].

A [`Material`][ordito.texture.Material] is what a textured mesh carries: a base-colour image and
the factor that multiplies it, read at the mesh's UVs by
[`material_colors`][ordito.texture.material_colors].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._common import read_scalar, read_values, require_same_device
from ordito.constants import INT32_MAX
from ordito.kernels import texture as kernel_texture

# Owner sentinel: larger than any face index, so the ``wp.atomic_min`` in ``rasterize_owner`` lets
# the lowest-index covering face win each pixel. It is seeded here and nowhere else -- the kernels
# never name it, testing ``owner[row, col] != f`` instead, which an uncovered pixel fails for every
# face -- so this allocation is the whole contract and a caller reading the returned owner map back
# compares against this value.
_OWNER_SENTINEL = INT32_MAX


def rasterize_attribute(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    attribute: odt.ArrayNdFloat32,
    resolution: int,
    *,
    subpixel_bits: int | None = None,
) -> odt.Array3dFloat32:
    """
    Rasterize a per-vertex attribute into a square UV-space texture.

    Each triangle is rasterized in UV space and its per-vertex attribute is barycentrically
    interpolated at every covered pixel center. Pixels not covered by any triangle stay ``0``.
    Where triangles overlap (degenerate for a valid parameterization), the lowest-index face wins.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates in ``[0, 1]``. Non-finite rows (e.g.
        unreferenced vertices left ``NaN`` by a disk parameterization) are ignored; by
        construction no face indexes them.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    attribute
        ``(n_vertices, n_channels)`` per-vertex attribute to interpolate.
    resolution
        Output image size in pixels (square).
    subpixel_bits
        When given, every vertex is first snapped to the nearest ``1 / 2 ** subpixel_bits`` of a
        pixel, as a GPU rasterizer snaps it before testing coverage and interpolating; ``8``
        reproduces OpenGL on current NVIDIA drivers. ``None`` (the default) rasterizes the UVs as
        given.

    Returns
    -------
    Array3dFloat32
        ``(resolution, resolution, n_channels)`` image on ``uv.device``. Row ``0`` corresponds to
        ``v = 1`` (vertical flip), matching the sampling convention of
        [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv].

    Raises
    ------
    TypeError
        If ``attribute`` is not a rank-2 ``wp.float32`` array.
    ValueError
        If ``resolution`` is not positive, if ``uv`` and ``attribute`` disagree on their row
        count, if any finite UV lies outside ``[0, 1]``, or if ``subpixel_bits`` is negative.
    RuntimeError
        If ``uv``, ``faces`` and ``attribute`` are not all on one device.

    See Also
    --------
    [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv]
    [`rasterize_discrete_attribute`][ordito.texture.rasterize_discrete_attribute]

    Examples
    --------
    Bake the ``u`` coordinate of a unit-square grid into an 8 x 8 texture; the top-right
    pixel centre sits at ``u = 7.5 / 8``:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5), center=False)
    uv = wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)
    attribute = wp.array(vertices.numpy()[:, :1], dtype=wp.float32)
    image = od.texture.rasterize_attribute(uv, faces, attribute, 8)
    print(image.shape)  # (8, 8, 1)
    print(f"{image.numpy()[0, 7, 0]:.4f}")  # 0.9375
    ```
    """
    require_same_device(uv=uv, faces=faces, attribute=attribute)
    odt.ensure_ndim(attribute, 2, dtype=wp.float32)
    n_channels = int(attribute.shape[1])
    _check_rasterize_inputs(uv, int(attribute.shape[0]), resolution)
    uv = _snapped_uv(uv, resolution, subpixel_bits)

    device = uv.device
    owner = _rasterize_owner(uv, faces, resolution, _launch.empty(0, dtype=wp.int32, device=device))
    # Every pixel is written by the resolve pass, covered or not.
    image = _launch.empty((resolution, resolution, n_channels), dtype=wp.float32, device=device)
    _launch.launch(
        kernel_texture.rasterize_pixels,
        dim=(resolution, resolution),
        inputs=[uv, faces, attribute, n_channels, owner, image],
        device=device,
    )
    return odt.as_array3d(image, wp.float32)


def rasterize_discrete_attribute(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    attribute: wp.array[wp.int32],
    resolution: int,
    *,
    subpixel_bits: int | None = None,
) -> odt.Array2dInt32:
    """
    Rasterize a per-vertex discrete label into a square UV-space class image.

    At each covered pixel the label with the greatest barycentric weight is emitted (weights of
    vertices sharing a label sum; ties resolve to the lowest label value). This is equivalent to
    one-hot encoding the labels, interpolating, and taking the per-pixel ``argmax``. Pixels not
    covered by any triangle are set to ``-1``.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates in ``[0, 1]``. Non-finite rows are
        ignored.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    attribute
        ``(n_vertices,)`` per-vertex labels, all ``>= 0``.
    resolution
        Output image size in pixels (square).
    subpixel_bits
        As in [`rasterize_attribute`][ordito.texture.rasterize_attribute]: snap every vertex to the
        nearest ``1 / 2 ** subpixel_bits`` of a pixel first.

    Returns
    -------
    Array2dInt32
        ``(resolution, resolution)`` class image with values in ``[-1, n_classes - 1]`` (``-1``
        marks uncovered pixels), on ``uv.device``.

    Raises
    ------
    ValueError
        If any entry of ``attribute`` is negative, if ``resolution`` is not positive, if ``uv``
        and ``attribute`` disagree on their row count, if any finite UV lies outside ``[0, 1]``,
        or if ``subpixel_bits`` is negative.
    RuntimeError
        If ``uv``, ``faces`` and ``attribute`` are not all on one device.

    See Also
    --------
    [`remap_discrete_attribute_from_uv`][ordito.texture.remap_discrete_attribute_from_uv]
    [`rasterize_attribute`][ordito.texture.rasterize_attribute]

    Examples
    --------
    Label the grid vertices right of ``x = 0.5`` ``1`` and the rest ``0``; a pixel takes the label
    with the greater interpolated weight:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5), center=False)
    uv = wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)
    labels = wp.array((vertices.numpy()[:, 0] > 0.5).astype("int32"), dtype=wp.int32)
    class_image = od.texture.rasterize_discrete_attribute(uv, faces, labels, 8)
    print(class_image.shape)  # (8, 8)
    print(class_image.numpy()[0].tolist())  # [0, 0, 0, 0, 0, 1, 1, 1]
    ```
    """
    require_same_device(uv=uv, faces=faces, attribute=attribute)
    odt.ensure_ndim(attribute, 1, dtype=wp.int32)
    n_vertices = attribute.size
    _check_rasterize_inputs(uv, n_vertices, resolution)
    uv = _snapped_uv(uv, resolution, subpixel_bits)

    device = uv.device
    owner = _rasterize_owner(uv, faces, resolution, attribute)
    # Every pixel is written by the resolve pass, covered or not.
    labels_image = _launch.empty((resolution, resolution), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_texture.rasterize_label_pixels,
        dim=(resolution, resolution),
        inputs=[uv, faces, attribute, owner, labels_image],
        device=device,
    )
    return odt.as_array2d(labels_image, wp.int32)


def rasterize_face_indices(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    origin: Sequence[float],
    pixel_size: float,
    shape: tuple[int, int],
    *,
    face_priority: wp.array[wp.float32] | None = None,
) -> odt.Array2dInt32:
    """
    Index of the face covering every pixel of a window of UV space: a visibility buffer.

    The window spans ``origin`` to ``origin + (columns, rows) * pixel_size``, with row ``0`` at its
    top (vertical flip), as in [`rasterize_attribute`][ordito.texture.rasterize_attribute]: pixel
    ``(i, j)`` is the UV point ``origin + (j + 0.5, rows - i - 0.5) * pixel_size``. A
    face covers a pixel when its UV triangle holds the pixel's centre, edges included, so faces
    sharing an edge leave no gap between them. Where several faces cover a pixel, the one of
    lowest ``face_priority`` wins, and between equal priorities (or with none given) the one of
    lowest index.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates, in any units: not limited to ``[0, 1]``. A
        face with a non-finite corner covers nothing.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    origin
        UV of the window's lower-left corner, ``(u, v)``.
    pixel_size
        Side of a pixel in UV units.
    shape
        ``(rows, columns)`` of the window.
    face_priority
        ``(n_faces,)`` priority of every face, the lowest winning where faces overlap; ``-0.0``
        ties with ``0.0`` and ``NaN`` loses to every number. ``None`` lets the face index decide.

    Returns
    -------
    Array2dInt32
        ``(rows, columns)`` index of the face covering every pixel, ``-1`` where none does, row
        ``0`` at the window's top, on ``uv.device``.

    Raises
    ------
    TypeError
        If ``face_priority`` is not a rank-1 ``wp.float32`` array.
    ValueError
        If ``shape`` holds no pixel, ``pixel_size`` is not positive and finite, ``origin`` is not
        two finite numbers, or ``face_priority`` does not hold one value a face.
    RuntimeError
        If ``uv``, ``faces`` and ``face_priority`` are not all on one device.

    See Also
    --------
    [`rasterize_attribute`][ordito.texture.rasterize_attribute]

    Examples
    --------
    A window of 4 x 4 pixels over the right half of a unit-square grid and beyond: its two
    right-hand columns lie past ``u = 1``, where no face is.

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(3, 3), center=False)
    uv = wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)
    face_image = od.texture.rasterize_face_indices(uv, faces, (0.5, 0.0), 0.25, (4, 4))
    print(face_image.shape)  # (4, 4)
    print(int((face_image.numpy() < 0).sum()))  # 8
    ```
    """
    require_same_device(uv=uv, faces=faces, face_priority=face_priority)
    rows, columns = (int(count) for count in shape)
    if rows <= 0 or columns <= 0:
        raise ValueError(f"shape must hold pixels, got {tuple(shape)}")
    if not (math.isfinite(pixel_size) and pixel_size > 0.0):
        raise ValueError(f"pixel_size must be positive and finite, got {pixel_size}")
    origin_u, origin_v = (float(coordinate) for coordinate in origin)
    if not (math.isfinite(origin_u) and math.isfinite(origin_v)):
        raise ValueError(f"origin must be finite, got {tuple(origin)}")
    device = uv.device
    n_faces = faces.size // 3
    if face_priority is None:
        priority = _launch.empty(0, dtype=wp.float32, device=device)
    else:
        odt.ensure_ndim(face_priority, 1, dtype=wp.float32)
        if face_priority.size != n_faces:
            raise ValueError(
                f"face_priority must hold one value a face: {face_priority.size} for {n_faces}"
            )
        priority = face_priority
    if n_faces == 0:
        return odt.as_array2d(
            _launch.full((rows, columns), -1, dtype=wp.int32, device=device), wp.int32
        )

    n_pixels = rows * columns
    keys = _launch.full(n_pixels, kernel_texture.NO_FACE_KEY, dtype=wp.uint64, device=device)
    keys_2d = keys.reshape((rows, columns))
    # Every vertex in the window's pixel frame, formed in float64 once rather than by every thread
    # of a face's split.
    pixels = _launch.empty(uv.size, dtype=wp.vec2, device=device)
    _launch.map(
        kernel_texture.window_pixel,
        uv,
        wp.vec2d(origin_u, origin_v),
        wp.float64(1.0 / pixel_size),
        wp.float64(rows),
        out=pixels,
    )
    slices = _owner_slices(n_faces, math.sqrt(n_pixels), device)
    if slices == 1:
        _launch.launch(
            kernel_texture.rasterize_face_keys,
            dim=n_faces,
            inputs=[pixels, faces, priority, keys_2d],
            device=device,
        )
    else:
        _launch.launch(
            kernel_texture.rasterize_face_keys_sliced,
            dim=n_faces * slices,
            inputs=[pixels, faces, priority, slices, keys_2d],
            device=device,
        )
    # Every pixel is written by the map, covered or not.
    face_image = _launch.empty(n_pixels, dtype=wp.int32, device=device)
    _launch.map(kernel_texture.face_of_key, keys, out=face_image)
    return odt.as_array2d(face_image.reshape((rows, columns)), wp.int32)


def _snapped_uv(
    uv: wp.array[wp.vec2], resolution: int, subpixel_bits: int | None
) -> wp.array[wp.vec2]:
    """``uv`` itself, or a copy snapped to ``2 ** subpixel_bits`` positions a pixel."""
    if subpixel_bits is None:
        return uv
    if subpixel_bits < 0:
        raise ValueError(f"subpixel_bits must be non-negative, got {subpixel_bits}")
    snapped = _launch.empty(uv.size, dtype=wp.vec2, device=uv.device)
    if uv.size > 0:
        scale = wp.float32(resolution * 2.0**subpixel_bits)
        _launch.map(kernel_texture.snap_uv, uv, scale, out=snapped)
    return snapped


def _check_rasterize_inputs(uv: wp.array[wp.vec2], n_vertices: int, resolution: int) -> None:
    """
    Validate the host-side arguments both rasterizers share: resolution and row count.

    The device-side checks (UV range, labels) run in the owner pass, ``_rasterize_owner``.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive or ``uv`` and the attribute disagree on their row count.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive")
    if uv.size != n_vertices:
        raise ValueError(f"uv and attribute row count mismatch: {uv.size} vs {n_vertices}")


def _rasterize_owner(
    uv: wp.array[wp.vec2], faces: wp.array[wp.int32], resolution: int, labels: wp.array[wp.int32]
) -> odt.Array2dInt32:
    """
    Resolve which face owns each pixel -- the lowest-index triangle covering it -- and validate.

    Both rasterizers run this first so their per-pixel pass writes exactly one face's contribution
    per pixel. Overlap is degenerate for a valid parameterization; where it happens the lowest face
    index wins, which ``wp.atomic_min`` against the sentinel gives for free. The same launch tests
    the vertices (``kernels/texture.rasterize_owner``): one readback of both flags, after it.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    resolution
        Output image size in pixels.
    labels
        The discrete rasterizer's ``(n_vertices,)`` labels, checked non-negative; empty for the
        continuous one.

    Returns
    -------
    odt.Array2dInt32
        ``(resolution, resolution)`` owning face index, ``_OWNER_SENTINEL`` where uncovered.

    Raises
    ------
    ValueError
        If any finite UV lies outside ``[0, 1]``, or any of ``labels`` is negative.
    """
    device = uv.device
    # The owner image and, behind it, the owner pass's two flags: one allocation and one fill,
    # the flags raised by lowering them below the sentinel.
    n_pixels = resolution * resolution
    owner_and_flags = _launch.full(n_pixels + 2, _OWNER_SENTINEL, dtype=wp.int32, device=device)
    owner = _launch.view(owner_and_flags, 0, n_pixels).reshape((resolution, resolution))
    n_faces = faces.size // 3
    flags = _launch.view(owner_and_flags, n_pixels, n_pixels + 2)
    slices = _owner_slices(n_faces, resolution, device)
    if slices == 1:
        n_threads = max(n_faces, uv.size)
        if n_threads == 0:
            return odt.as_array2d(owner, wp.int32)
        _launch.launch(
            kernel_texture.rasterize_owner,
            dim=n_threads,
            inputs=[uv, faces, labels, resolution, owner, flags],
            device=device,
        )
    else:
        _launch.launch(
            kernel_texture.rasterize_owner_sliced,
            dim=max(n_faces * slices, uv.size),
            inputs=[uv, faces, labels, resolution, slices, owner, flags],
            device=device,
        )
    # One readback of both flags: nothing can raise without bringing them to the host.
    uv_flag, label_flag = (int(flag) for flag in read_values(owner_and_flags, n_pixels, 2))
    if uv_flag != _OWNER_SENTINEL:
        raise ValueError("UV coordinates must be in the range [0, 1]")
    if label_flag != _OWNER_SENTINEL:
        raise ValueError("Attribute values must be greater than or equal to 0")
    return odt.as_array2d(owner, wp.int32)


def _owner_slices(n_faces: int, pixels_across: float, device: wp.DeviceLike) -> int:
    """
    Threads the owner pass deals each face's raster window to: a power of two, ``1`` on the CPU.

    A face of an atlas ``pixels_across`` pixels on a side covers about ``pixels_across ** 2 /
    n_faces`` pixels, so its window is about ``pixels_across / sqrt(n_faces)`` pixels wide;
    ``kernels/texture.OWNER_SLICE_WIDTHS`` such widths a face, capped at
    ``2 ** kernels/texture.OWNER_MAX_SLICES_LOG2``. The CPU runs a launch's threads one after
    another, where a split only adds work.
    """
    if n_faces == 0 or wp.get_device(device).is_cpu:
        return 1
    target = kernel_texture.OWNER_SLICE_WIDTHS * pixels_across / math.sqrt(n_faces)
    if target < 1.0:
        return 1
    return 1 << min(round(math.log2(target)), kernel_texture.OWNER_MAX_SLICES_LOG2)


def remap_attribute_from_uv(
    uv: wp.array[wp.vec2],
    image: odt.Array2dFloat32 | odt.Array3dFloat32,
    *,
    order: Literal[0, 1] = 1,
) -> odt.Array2dFloat32:
    """
    Sample a UV-space texture back to per-vertex values.

    Inverse of [`rasterize_attribute`][ordito.texture.rasterize_attribute].
    Sampling uses the same pixel-center convention as
    [`rasterize_attribute`][ordito.texture.rasterize_attribute] (``col = u * W - 0.5``,
    ``row = (1 - v) * H - 0.5``) and clamps to the nearest edge pixel, so vertices on the
    coverage boundary read the boundary value instead of blending toward ``0``. Vertices whose
    UV is non-finite are not sampled; their output row is ``NaN`` on every channel.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates in ``[0, 1]``. Non-finite rows yield
        ``NaN`` output rows.
    image
        ``(H, W)`` or ``(H, W, C)`` ``float32`` UV-space texture to sample.
    order
        Interpolation order: ``1`` for bilinear (default), ``0`` for nearest-neighbor.

    Returns
    -------
    Array2dFloat32
        ``(n_vertices, C)`` per-vertex values on ``uv.device`` (``C == 1`` for a 2D input
        image). Non-finite-UV rows are ``NaN``.

    Raises
    ------
    TypeError
        If ``image`` is neither rank 2 nor rank 3.
    ValueError
        If ``order`` is not ``0`` or ``1``, or if any finite UV lies outside ``[0, 1]``.
    RuntimeError
        If ``uv`` and ``image`` are not all on one device.

    See Also
    --------
    [`rasterize_attribute`][ordito.texture.rasterize_attribute]

    Examples
    --------
    A texture baked from a per-vertex value samples back to nearly the same value:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5), center=False)
    uv = wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)
    attribute = wp.array(vertices.numpy()[:, :1], dtype=wp.float32)
    image = od.texture.rasterize_attribute(uv, faces, attribute, 64)
    values = od.texture.remap_attribute_from_uv(uv, image)
    print(values.shape)  # (25, 1)
    print(abs(values.numpy() - attribute.numpy()).max() < 0.01)  # True
    ```
    """
    require_same_device(uv=uv, image=image)
    if order not in (0, 1):
        raise ValueError(f"order must be 0 (nearest) or 1 (bilinear), got {order!r}")
    if int(image.ndim) == 2:
        height, width = int(image.shape[0]), int(image.shape[1])
        n_channels = 1
        image3d = image.reshape((height, width, 1))
    elif int(image.ndim) == 3:
        n_channels = int(image.shape[2])
        image3d = image
    else:
        raise TypeError(f"image must be (H, W) or (H, W, C), got ndim={image.ndim}")
    odt.ensure_ndim(image3d, 3, dtype=wp.float32)

    device = uv.device
    n_vertices = uv.size
    out_values = odt.empty_2d((n_vertices, n_channels), wp.float32, device=device)
    if n_vertices > 0:
        # ``order`` is rejected above rather than defaulted here: an ``if order == 1 else``
        # sampled nearest-neighbour for every off-menu value, so a caller whose interpolation
        # order arrived through a variable got a plausible image and no error.
        mode = kernel_texture.SAMPLE_BILINEAR if order == 1 else kernel_texture.SAMPLE_NEAREST
        flag = _launch.zeros(1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_texture.sample_texture,
            dim=n_vertices,
            inputs=[uv, image3d, n_channels, mode, out_values, flag],
            device=device,
        )
        _raise_on_uv_flag(flag)
    return odt.as_array2d(out_values, wp.float32)


def remap_discrete_attribute_from_uv(
    uv: wp.array[wp.vec2], class_image: odt.Array2dInt32
) -> odt.Array1dInt32:
    """
    Sample a UV-space class image back to per-vertex labels.

    Inverse of [`rasterize_discrete_attribute`][ordito.texture.rasterize_discrete_attribute].
    Nearest-neighbor sampling is used so labels are never blended, including the ``-1`` uncovered
    sentinel. Vertices whose UV is non-finite cannot be sampled and map to the same ``-1``
    sentinel.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates in ``[0, 1]``. Non-finite rows yield
        ``-1``.
    class_image
        ``(H, W)`` class image (values e.g. in ``[-1, n_classes - 1]``).

    Returns
    -------
    Array1dInt32
        ``(n_vertices,)`` per-vertex labels on ``uv.device``. Non-finite-UV rows
        are ``-1``.

    Raises
    ------
    ValueError
        If any finite UV lies outside ``[0, 1]``.
    RuntimeError
        If ``uv`` and ``class_image`` are not all on one device.

    See Also
    --------
    [`rasterize_discrete_attribute`][ordito.texture.rasterize_discrete_attribute]
    [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv]

    Examples
    --------
    A class image baked from per-vertex labels samples back to the same labels:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5), center=False)
    uv = wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)
    labels = wp.array((vertices.numpy()[:, 0] > 0.5).astype("int32"), dtype=wp.int32)
    class_image = od.texture.rasterize_discrete_attribute(uv, faces, labels, 64)
    remapped = od.texture.remap_discrete_attribute_from_uv(uv, class_image)
    print((remapped.numpy() == labels.numpy()).all())  # True
    ```
    """
    require_same_device(uv=uv, class_image=class_image)
    odt.ensure_ndim(class_image, 2, dtype=wp.int32)

    device = uv.device
    n_vertices = uv.size
    if n_vertices == 0:
        return odt.empty_1d(0, wp.int32, device=device)
    # The labels and, behind them, the range flag: one allocation, the flag's slot zeroed.
    labels_and_flag = _launch.empty(n_vertices + 1, dtype=wp.int32, device=device)
    out_labels = _launch.view(labels_and_flag, 0, n_vertices)
    flag = _launch.view(labels_and_flag, n_vertices, n_vertices + 1)
    _launch.zero_(flag)
    _launch.launch(
        kernel_texture.sample_class_image,
        dim=n_vertices,
        inputs=[uv, class_image, out_labels, flag],
        device=device,
    )
    _raise_on_uv_flag(flag)
    return out_labels


def _raise_on_uv_flag(flag: wp.array[wp.int32]) -> None:
    """
    Raise if the sampling pass flagged a finite UV outside ``[0, 1]`` (non-finite UVs never are).

    The sampling kernels test every UV as they read it and set the one-slot flag, so the check
    costs no launch of its own; the samples written meanwhile are discarded with the raise. The
    readback is unavoidable: nothing can raise without bringing the flag back to the host. Kept
    for the samplers although it is a fixed share of a launch-bound call, because an out-of-range
    UV is not rejected downstream but silently *clamped* to the nearest edge pixel: dropping the
    guard trades a raised public-API error for a plausible wrong answer -- the same trade
    ``_common.require_same_device`` is argued from. If a caller ever samples in a loop, the shape
    to add is a ``validate=False`` keyword **with** an in-repo caller passing it, not an
    unconditional removal.

    Raises
    ------
    ValueError
        If the flag is set.
    """
    if int(read_scalar(flag, 0)) != 0:
        raise ValueError("UV coordinates must be in the range [0, 1]")


@dataclass(frozen=True, eq=False)
class Material:
    """
    A surface's base colour: an RGBA texture image and a factor multiplying it.

    What a textured [`Trimesh`][ordito.mesh.Trimesh] carries beside its UVs (glTF's base colour,
    an OBJ material's ``map_Kd`` and ``Kd``). The colour of a point is the image sampled at its UV,
    channel by channel times ``base_color``; with no image it is ``base_color`` alone. Colours are
    in ``[0, 1]`` in the image's own encoding (sRGB for an image read from a file): nothing is
    converted.

    Two meshes share a material only by holding the same object: equality is identity.

    Parameters
    ----------
    image
        ``(height, width, 4)`` ``float32`` RGBA texture, row 0 the top of the image (``v = 1``),
        or ``None`` for a uniform colour.
    base_color
        RGBA factor multiplying the image. Defaults to opaque white, which leaves the image as it
        is.
    name
        Name the material is written under.

    Raises
    ------
    TypeError
        If ``image`` is not a rank-3 ``wp.float32`` array.
    ValueError
        If ``image`` does not have four channels or has no texel.

    See Also
    --------
    [`material_colors`][ordito.texture.material_colors]
    [`ordito.mesh.Trimesh.material`][]
    [`trimesh.visual.material.PBRMaterial`][]

    Examples
    --------
    A two-texel texture, red over blue, read at the centre of each texel:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    texels = np.array([[[1.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    material = od.texture.Material(image=wp.array(texels, dtype=wp.float32))
    uv = wp.array([[0.5, 0.75], [0.5, 0.25]], dtype=wp.vec2)
    print(od.texture.material_colors(material, uv).numpy().tolist())
    # [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    ```
    """

    image: odt.Array3dFloat32 | None = None
    base_color: wp.vec4 = field(default_factory=lambda: wp.vec4(1.0, 1.0, 1.0, 1.0))
    name: str = "material"

    def __post_init__(self) -> None:
        """Validate the image's rank, dtype and channels."""
        if self.image is None:
            return
        odt.ensure_ndim(self.image, 3, dtype=wp.float32)
        height, width, channels = (int(extent) for extent in self.image.shape)
        if channels != 4:
            raise ValueError(f"a material image must have 4 (RGBA) channels, got {channels}")
        if height == 0 or width == 0:
            raise ValueError(f"a material image must have a texel, got shape {(height, width, 4)}")


def material_colors(material: Material, uv: wp.array[wp.vec2]) -> wp.array[wp.vec4]:
    """
    Return a material's RGBA colour at each UV.

    The image is sampled bilinearly at pixel centres (``col = u * W - 0.5``,
    ``row = (1 - v) * H - 0.5``, the convention of
    [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv]) and tiles: a UV outside
    ``[0, 1]`` reads the texture repeated, as glTF's and OBJ's default sampler does. Each colour is
    the sample times the material's ``base_color``; a material with no image gives
    ``base_color`` everywhere.

    Parameters
    ----------
    material
        Material to read.
    uv
        ``(n,)`` texture coordinates. An Inf or NaN UV gives a NaN colour.

    Returns
    -------
    wp.array[wp.vec4]
        ``(n,)`` RGBA colours.

    Raises
    ------
    RuntimeError
        If ``uv`` and the material's image are not on one device.

    See Also
    --------
    [`Material`][ordito.texture.Material]
    [`ordito.mesh.Trimesh.to_color`][]
    [`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv]
        The clamped, any-channel sampler of a baked texture.

    Notes
    -----
    trimesh's ``TextureVisuals.to_color`` reads the nearest pixel under ``col = u * (W - 1)`` and
    leaves the factor out when there is an image; the two agree wherever the texture is constant
    over a few texels around the UV.

    Examples
    --------
    A uniform material is its factor at every UV:

    ```python
    import warp as wp

    import ordito as od

    material = od.texture.Material(base_color=wp.vec4(0.5, 0.25, 1.0, 1.0))
    uv = wp.array([[0.1, 0.2], [3.5, -1.0]], dtype=wp.vec2)
    print(od.texture.material_colors(material, uv).numpy().tolist())
    # [[0.5, 0.25, 1.0, 1.0], [0.5, 0.25, 1.0, 1.0]]
    ```
    """
    device = uv.device
    if material.image is None:
        return _launch.full(uv.size, material.base_color, dtype=wp.vec4, device=device)
    require_same_device(uv=uv, image=material.image)
    out_colors = _launch.empty(uv.size, dtype=wp.vec4, device=device)
    if uv.size > 0:
        _launch.launch(
            kernel_texture.material_colors,
            dim=uv.size,
            inputs=[uv, material.image, material.base_color, out_colors],
            device=device,
        )
    return out_colors
