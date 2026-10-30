"""
Bake per-vertex attributes into UV-space textures and sample them back.

The four functions here operate on an *existing* UV map (from a disk parameterization or a
mesh file, e.g. loaded via [`load_mesh`][ordito.io.load_mesh]): they rasterize per-vertex
attributes into a square texture and sample a texture back to per-vertex values. They are a
Warp software-rasterizer replacement for an OpenGL/moderngl pipeline, keeping all data on the
input Warp device.

Computing a UV map from mesh connectivity (parameterization) is a separate concern and is not
part of this module.

See [`rasterize_attribute`][ordito.texture.rasterize_attribute],
[`rasterize_discrete_attribute`][ordito.texture.rasterize_discrete_attribute],
[`remap_attribute_from_uv`][ordito.texture.remap_attribute_from_uv], and
[`remap_discrete_attribute_from_uv`][ordito.texture.remap_discrete_attribute_from_uv].
"""

from __future__ import annotations

from typing import Literal

import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
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
    n_threads = max(faces.size // 3, uv.size)
    if n_threads == 0:
        return odt.as_array2d(owner, wp.int32)
    _launch.launch(
        kernel_texture.rasterize_owner,
        dim=n_threads,
        inputs=[
            uv,
            faces,
            labels,
            resolution,
            owner,
            _launch.view(owner_and_flags, n_pixels, n_pixels + 2),
        ],
        device=device,
    )
    # One readback of both flags: nothing can raise without bringing them to the host.
    uv_flag, label_flag = (int(flag) for flag in read_values(owner_and_flags, n_pixels, 2))
    if uv_flag != _OWNER_SENTINEL:
        raise ValueError("UV coordinates must be in the range [0, 1]")
    if label_flag != _OWNER_SENTINEL:
        raise ValueError("Attribute values must be greater than or equal to 0")
    return odt.as_array2d(owner, wp.int32)


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
    out_labels = odt.empty_1d(n_vertices, wp.int32, device=device)
    if n_vertices > 0:
        flag = _launch.zeros(1, dtype=wp.int32, device=device)
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
    ``_device.require_same_device`` is argued from. If a caller ever samples in a loop, the shape
    to add is a ``validate=False`` keyword **with** an in-repo caller passing it, not an
    unconditional removal.

    Raises
    ------
    ValueError
        If the flag is set.
    """
    if int(read_scalar(flag, 0)) != 0:
        raise ValueError("UV coordinates must be in the range [0, 1]")
