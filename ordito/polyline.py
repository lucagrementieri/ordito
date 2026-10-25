"""
Open and closed 3D polyline operations.

**The ``polyline_`` prefix.** Where a name carries the module's own token, the token comes
**first** -- ``polyline_length``, ``polyline_open``, ``polyline_resample``,
``polyline_triangulate`` -- so the whole family sorts and completes together.

A function that does *not* need the token does not gain one:
[`is_closed`][ordito.polyline.is_closed] and
[`cumulative_arc_length`][ordito.polyline.cumulative_arc_length] have an unambiguous subject
already, and [`triangulate_polygon`][ordito.polyline.triangulate_polygon] takes a 2D polygon
rather than a polyline, so the token there would be wrong rather than redundant.
[`polyline_point_distance`][ordito.polyline.polyline_point_distance] keeps both nouns because the
prefix alone would lose "from what"; it is spelled to match
[`points.point_plane_distance`][ordito.points.point_plane_distance].

**The ``closed=`` convention.** Ten of these functions -- length, centroid, radius, angles,
point-to-curve distance, and the five resampling operations -- take a keyword-only
``closed: bool = False``. Passing ``closed=True`` appends the closing edge back to the first point
if it is absent ([`polyline_close`][ordito.polyline.polyline_close]) and then does the open
computation on that, so the seam is treated like any other segment. Three of the ten additionally
drop the duplicated seam point on the way out, so their result is a clean cyclic ring with one entry
per *original* point rather than one extra:
[`polyline_resample`][ordito.polyline.polyline_resample],
[`polyline_angles`][ordito.polyline.polyline_angles] and
[`polyline_smooth_upsample`][ordito.polyline.polyline_smooth_upsample].

The keyword is not always needed. [`polyline_angles`][ordito.polyline.polyline_angles] and
[`polyline_smooth_upsample`][ordito.polyline.polyline_smooth_upsample] already detect an
*explicitly* closed input -- one whose last point equals its first -- with
[`is_closed`][ordito.polyline.is_closed]; ``closed=True`` is for the common case of a loop stored
without that duplicate, which is the form
[`boundary_loops`][ordito.boundary.boundary_loops] returns.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device, run_device_loop
from ordito.config import is_deterministic
from ordito.constants import TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import graph as kernel_graph
from ordito.kernels import polyline as kernel_polyline
from ordito.kernels import reduce as kernel_reduce

# Point count from which [`polyline_downsample`][ordito.polyline.polyline_downsample] stops
# walking its greedy selection serially and pointer-doubles it instead -- **on the CUDA device
# only**. The doubling costs a logarithmic number of launches, which pays off once the polyline
# is long enough that a serial walk's linear cost exceeds it; the two masks are verified
# **byte-identical** at every size.
#
# On the CPU device the doubling loses at every size, because it does ``n log n`` work where the
# serial form does ``n``, on a backend that runs a launch grid as one serial loop -- there is no GPU
# win being paid for, so the branch takes the device too.
_DOWNSAMPLE_DOUBLING_FROM = 2048


# ``polyline_radius``'s folded reductions: the kernel's selector and the result slot's seed.
_RADIUS_REDUCTIONS: dict[str, tuple[wp.int32, float]] = {
    "min": (kernel_polyline.RADIUS_MIN, math.inf),
    "max": (kernel_polyline.RADIUS_MAX, -math.inf),
    "mean": (kernel_polyline.RADIUS_SUM, 0.0),
}


def is_closed(polyline: wp.array[wp.vec3]) -> bool:
    """
    Whether a polyline is closed (its first and last points coincide).

    The endpoint comparison runs on-device — one thread, one flag, one four-byte readback — so no
    array is copied to the host. It applies the same tolerance predicate
    [`allclose`][ordito.array.allclose] does, through the same kernel-side function.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.

    Returns
    -------
    bool
        ``True`` when the polyline has at least two points and its first and last points are
        equal within the default ``allclose`` tolerance; ``False`` otherwise.

    See Also
    --------
    [`polyline_open`][ordito.polyline.polyline_open]
    [`polyline_close`][ordito.polyline.polyline_close]

    Examples
    --------
    Four corners of a unit square are open until the first point is repeated at the end:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.is_closed(square))  # False
    print(od.polyline.is_closed(od.polyline.polyline_close(square)))  # True
    ```
    """
    n = polyline.size
    if n < 2:
        return False
    return bool(int(read_scalar(_endpoints_coincide_flag(polyline), 0)) != 0)


def _endpoints_coincide_flag(polyline: wp.array[wp.vec3]) -> wp.array[wp.int32]:
    """
    One-element device flag, ``1`` when the first and last points coincide.

    [`is_closed`][ordito.polyline.is_closed]'s device half. The kernels that take the closure
    decision on the device instead evaluate the same predicate themselves, so none of them pays
    this launch. One launch rather than ``allclose`` over two one-element slices: that spelling is
    a ``wp.map`` into a mask plus a whole reduction over it, to compare six floats. Needs at least
    two points.
    """
    flag = _launch.empty(1, dtype=wp.int32, device=polyline.device)
    _launch.launch(
        kernel_polyline.endpoints_coincide,
        dim=1,
        inputs=[polyline],
        outputs=[flag],
        device=polyline.device,
    )
    return flag


def polyline_open(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Open a polyline by dropping the last point when it duplicates the first.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` or ``(n - 1,)`` polyline: the input unchanged when it has fewer than two points or
        is already open; otherwise a view without the duplicated closing point.

    See Also
    --------
    [`polyline_close`][ordito.polyline.polyline_close]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    closed = od.polyline.polyline_close(square)
    print(closed.shape, od.polyline.polyline_open(closed).shape)  # (5,) (4,)
    ```
    """
    n = polyline.size
    if not is_closed(polyline):
        return polyline
    return odt.as_dense(polyline[0 : n - 1])


def polyline_close(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Close a polyline by appending the first point when it is not already the last.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` or ``(n + 1,)`` polyline: the input unchanged when it has fewer than two points or
        is already closed; otherwise an array with the first point appended.

    See Also
    --------
    [`polyline_open`][ordito.polyline.polyline_open]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    closed = od.polyline.polyline_close(square)
    print(closed.shape)  # (5,)
    print(closed.numpy()[-1])  # [0. 0. 0.]
    ```
    """
    n = polyline.size
    if n < 2 or is_closed(polyline):
        return polyline
    return _append_first_point(polyline)


def _append_first_point(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Return the polyline with its first point appended: two copies into one allocation.

    What [`array.concatenate`][ordito.array.concatenate] of the polyline and its first point
    returns, without the segment table that function builds for an arbitrary list.
    """
    n = polyline.size
    closed = _launch.empty(n + 1, dtype=wp.vec3, device=polyline.device)
    _launch.copy(closed, polyline, count=n)
    _launch.copy(closed, polyline, dest_offset=n, count=1)
    return closed


def polyline_length(polyline: wp.array[wp.vec3], *, closed: bool = False) -> float:
    """
    Total arc length of a polyline (sum of segment lengths).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]), so the closing
        edge counts toward the total.

    Returns
    -------
    float
        The summed segment length, ``0.0`` for fewer than two points.

    See Also
    --------
    [`cumulative_arc_length`][ordito.polyline.cumulative_arc_length]
        The same lengths, unreduced.

    Examples
    --------
    The open square has three unit sides; closing it adds the fourth:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.polyline_length(square))  # 3.0
    print(od.polyline.polyline_length(square, closed=True))  # 4.0
    ```
    """
    device = polyline.device
    n_points = polyline.size
    # ``closed`` adds the segment from the last point back to the first, which the kernel reaches
    # by wrapping its index -- no ``polyline_close`` copy of the whole buffer for one segment.
    n_segments = n_points if closed else n_points - 1
    if n_segments < 1:
        return 0.0
    # Summed where the segment lengths are computed, rather than through an ``(n - 1,)`` scratch
    # buffer and a separate reduction over it -- one launch, one allocation and one readback
    # instead of two of each. See ``kernels/polyline.polyline_total_length``, including why the
    # answer's last bits move.
    blocks = kernel_reduce.blocks_1d(n_segments)
    total, stride = _launch.block_partials(blocks, 1, wp.float32, device)
    _launch.launch_tiled(
        kernel_polyline.polyline_total_length,
        dim=blocks,
        inputs=[polyline, wp.int32(n_segments), stride],
        outputs=[total],
        block_dim=TILE_1D,
        device=device,
    )
    return float(read_scalar(total, 0))


def polyline_centroid(polyline: wp.array[wp.vec3], *, closed: bool = False) -> wp.vec3:
    """
    Segment-length-weighted centroid of a polyline.

    Each segment contributes its midpoint weighted by its length, so the result is invariant to
    how densely the polyline is sampled (unlike the plain mean of the vertices).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]), so the closing
        segment contributes its midpoint and length like any other.

    Returns
    -------
    wp.vec3
        The weighted centroid on the host.

    Raises
    ------
    ValueError
        If the polyline has fewer than two points.

    See Also
    --------
    [`polyline_normal`][ordito.polyline.polyline_normal]
    [`polyline_radius`][ordito.polyline.polyline_radius]
        Both default their plane to this centroid.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.polyline_centroid(square, closed=True))  # [0.5, 0.5, 0.0]
    ```
    """
    n_points = polyline.size
    # ``closed`` is the wrap-around segment, which the kernel reaches by index rather than by a
    # ``polyline_close`` copy of the whole buffer -- as in ``polyline_length``.
    n_segments = n_points if closed else n_points - 1
    if n_segments < 1:
        raise ValueError("polyline_centroid requires at least two points")
    # One launch and one readback for all four sums: ``polyline_radius``'s frame pass with no
    # Newell sum, whose first four slots are the weighted midpoints and the total length.
    sums = _accumulate_frame(polyline, n_segments, 0, with_normal=False)
    sums_np = np.asarray(read_values(sums, 0, 4), dtype=np.float32)
    # A NumPy float32 quotient: the same IEEE division per component, without Warp's Python-scope
    # dispatch of a ``wp.vec3`` operator.
    return wp.vec3(*(sums_np[:3] / sums_np[3]))


def polyline_normal(polyline: wp.array[wp.vec3]) -> wp.vec3:
    """
    Average unit normal of a 3D polyline via Newell's method.

    The polyline is treated as a closed loop (its closing edge is added if absent), then the
    cross products of consecutive vertices, ``cross(V_i, V_{i + 1})``, are summed and normalized.
    This is Newell's method: the summed cross products of consecutive position vectors around a
    closed loop equal the area-weighted, translation-invariant plane normal. The open-chain
    variant (omitting the closing edge) is origin-dependent and not computed.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices. The closing edge is added if absent.

    Returns
    -------
    wp.vec3
        The unit normal on the host.

    Raises
    ------
    ValueError
        If the polyline has fewer than three points.

    Examples
    --------
    A counter-clockwise loop in the ``xy`` plane has the ``+z`` normal:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.polyline_normal(square))  # [0.0, 0.0, 1.0]
    ```
    """
    n = polyline.size
    # A non-degenerate loop normal needs three distinct vertices, and a polyline whose last vertex
    # duplicates its first has only ``n - 1`` of them.
    if n < 3:
        raise ValueError("polyline_normal requires at least three points")
    # Whether the loop is already closed is decided on the device: the kernel evaluates the closure
    # predicate itself and sums over the closing edge by wrapping its index, so nothing is copied
    # to close the loop and nothing is read back to decide whether to. Only a three-point input
    # needs the answer on the host, to tell a triangle from a closed two-point loop.
    if n == 3 and is_closed(polyline):
        raise ValueError("polyline_normal requires at least three points")
    # ``polyline_radius``'s frame pass with no segments: only its Newell sum, in its slots.
    frame = _accumulate_frame(polyline, 0, 0, with_normal=True)
    # Normalized on the host, in float32 as ``wp.normalize`` does, rather than by a ``wp.map``
    # launch before the readback: the three components cross either way.
    normal_slot = int(kernel_polyline.RADIUS_FRAME_NORMAL)
    x, y, z = np.asarray(read_values(frame, normal_slot, 3), dtype=np.float32)
    length = np.sqrt(x * x + y * y + z * z)
    if length > 0.0:
        return wp.vec3(x / length, y / length, z / length)
    return wp.vec3(0.0, 0.0, 0.0)


def polyline_point_distance(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3], *, closed: bool = False
) -> wp.array[wp.float32]:
    """
    Minimum distance from each query point to the nearest segment of a polyline.

    Parameters
    ----------
    points
        ``(n,)`` query points.
    polyline
        ``(m,)`` polyline vertices.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]), so the closing
        edge is a candidate segment like any other.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` minimum distances on ``points.device``. Every entry is ``inf`` when
        ``polyline`` is empty, there being no segment to measure against -- the same
        ``inf``-on-miss convention
        [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] uses.

    Raises
    ------
    RuntimeError
        If ``points`` and ``polyline`` are not all on one device.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    points = wp.array([[0.5, 0.5, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3)
    distances = od.polyline.polyline_point_distance(points, square, closed=True)
    print(distances.numpy())  # [0.5 1. ]
    ```
    """
    require_same_device(points=points, polyline=polyline)
    device = points.device
    n_points = points.size
    m = polyline.size
    if m == 0:
        return _launch.full(n_points, float("inf"), dtype=wp.float32, device=device)
    if n_points == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)
    if m == 1:
        out_distances = _launch.empty(n_points, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_polyline.distance_to_first_point,
            dim=n_points,
            inputs=[points, polyline, out_distances],
            device=device,
        )
        return out_distances
    # ``closed`` adds the closing segment inside the kernel, which decides per thread whether the
    # input already repeats its first point -- no ``polyline_close`` readback or copy. On CUDA the
    # segments are split into slices when the queries alone would leave the device mostly idle,
    # each slice folding its minimum into the query's slot; a minimum is order-free, so the answer
    # is the same. The CPU device runs a launch grid as one loop, where slicing buys nothing.
    n_open = m - 1
    n_slices = 1
    if wp.get_device(device).is_cuda:
        n_slices = min(
            -(-kernel_polyline.POINT_DISTANCE_THREADS // n_points),
            -(-n_open // kernel_polyline.POINT_DISTANCE_MIN_SLICE),
        )
    if n_slices <= 1:
        out_distances = _launch.empty(n_points, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_polyline.distance_to_segments,
            dim=n_points,
            inputs=[points, polyline, wp.int32(closed), out_distances],
            device=device,
        )
        return out_distances
    out_distances = _launch.full(n_points, math.inf, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_polyline.distance_to_segment_slices,
        dim=(n_slices, n_points),
        inputs=[points, polyline, wp.int32(closed), wp.int32(-(-n_open // n_slices))],
        outputs=[out_distances],
        device=device,
    )
    return out_distances


def polyline_upsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Upsample a polyline to an approximately uniform step size.

    Each segment is split into ``max(floor(length / step_size), 1)`` equal pieces. The final
    endpoint of the polyline is not emitted (matching the source implementation).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    step_size
        Target spacing between consecutive output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]), and the
        duplicated closing point is not emitted, so the result is a cyclic ring.

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` upsampled polyline of ``m`` points. The input is returned unchanged for fewer than
        two points.

    See Also
    --------
    [`polyline_smooth_upsample`][ordito.polyline.polyline_smooth_upsample]
        The same subdivision with the new points placed on a fitted arc instead of the chord.
    [`polyline_downsample`][ordito.polyline.polyline_downsample]
    [`polyline_resample`][ordito.polyline.polyline_resample]

    Examples
    --------
    Each unit side of the closed square is split into four pieces:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    upsampled = od.polyline.polyline_upsample(square, 0.25, closed=True)
    print(upsampled.shape)  # (16,)
    print(upsampled.numpy()[1])  # [0.25 0.   0.  ]
    ```
    """
    return _upsample(polyline, step_size, closed, kernel_polyline.upsample_gather, [])


def polyline_smooth_upsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Upsample a polyline to an approximately uniform step size, following local curvature.

    Like [`polyline_upsample`][ordito.polyline.polyline_upsample], each segment is split into
    ``max(floor(length / step_size), 1)`` pieces and the final endpoint is not emitted. Unlike it,
    the inserted points are placed on a circular arc fitted to the segment's endpoint tangents
    (estimated from the two bracketing neighbour vertices) rather than on the straight chord, so a
    coarsely sampled curve is refined smoothly. Curvature-aware placement is usually stated for the
    edge midpoint alone; here it is generalised to every interpolation parameter.

    The first and last segments of an open polyline have no bracketing neighbour and are subdivided
    linearly; collinear neighbours likewise reduce to the straight chord. Original vertices are
    preserved exactly, since each segment's first sample coincides with its start vertex.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    step_size
        Target spacing between consecutive output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]). Every segment
        including the seam is then treated as interior, so neighbour tangents wrap cyclically and
        the whole loop is smoothed rather than only its middle.

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` curvature-aware upsampled polyline of ``m`` points. The input is returned unchanged
        for fewer than two points.

    See Also
    --------
    [`polyline_upsample`][ordito.polyline.polyline_upsample]
        The straight-chord version, which is what this reduces to on the end segments.

    Examples
    --------
    The inserted points bulge outward from the square's sides, following the corners' turn:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    straight = od.polyline.polyline_upsample(square, 0.25, closed=True)
    rounded = od.polyline.polyline_smooth_upsample(square, 0.25, closed=True)
    print(rounded.shape)  # (16,)
    print(f"{straight.numpy()[2, 1]:.3f} {rounded.numpy()[2, 1]:.3f}")  # 0.000 -0.207
    ```
    """
    # An *explicitly* closed input (last point already equal to the first) is smoothed as a loop
    # too, per the module docstring -- so a caller does not have to pass ``closed=True`` for a ring
    # ``boundary_loops`` already returned. The gather kernel decides that per thread, with the
    # predicate ``is_closed`` applies, so neither case costs a closure readback.
    gather = kernel_polyline.smooth_upsample_gather
    return _upsample(polyline, step_size, closed, gather, [wp.int32(closed)])


def _upsample(
    polyline: wp.array[wp.vec3],
    step_size: float,
    closed: bool,
    gather_kernel: odt.Kernel,
    extra_inputs: list[wp.int32 | wp.float32],
) -> wp.array[wp.vec3]:
    """
    Split every segment into ``max(floor(length / step_size), 1)`` pieces and gather the samples.

    The shared body of [`polyline_upsample`][ordito.polyline.polyline_upsample] and
    [`polyline_smooth_upsample`][ordito.polyline.polyline_smooth_upsample], which differ only in
    the gather kernel that places each sample -- on the chord or on a fitted arc -- and in the
    extra arguments that kernel takes. ``closed`` counts ``n`` segments rather than ``n - 1``: the
    last is the closing one, reached by wrapping the index, or a segment with no samples when the
    input already repeats its first point (``segment_step_counts``) -- so the host never asks.
    """
    device = polyline.device
    n_points = polyline.size
    if n_points < 2:
        return polyline
    n_segments = n_points if closed else n_points - 1

    steps = _launch.empty(n_segments, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.segment_step_counts,
        dim=n_segments,
        inputs=[polyline, wp.float32(step_size), wp.int32(closed), steps],
        device=device,
    )
    offsets, total = od.array.counts_to_offsets(steps)

    out_points = _launch.empty(total, dtype=wp.vec3, device=device)
    _launch.launch(
        gather_kernel,
        dim=total,
        inputs=[polyline, offsets, steps, *extra_inputs, out_points],
        device=device,
    )
    return out_points


def cumulative_arc_length(polyline: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Cumulative arc length from the first vertex to each vertex of an open polyline.

    The underlying arc-length parametrization behind
    [`polyline_downsample`][ordito.polyline.polyline_downsample] and
    [`polyline_resample`][ordito.polyline.polyline_resample]; exposed directly for callers
    doing custom resampling along the polyline.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` cumulative arc lengths on ``polyline.device``. Entry ``0`` is ``0.0``; entry ``i``
        is the summed length of segments ``0..i-1``.

    See Also
    --------
    [`polyline_length`][ordito.polyline.polyline_length]
    [`polyline_downsample`][ordito.polyline.polyline_downsample]
    [`polyline_resample`][ordito.polyline.polyline_resample]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.cumulative_arc_length(square).numpy())  # [0. 1. 2. 3.]
    ```
    """
    n_points = polyline.size
    if n_points < 2:
        # No segment to sum: a single vertex is at arc length 0, an empty polyline has no entry.
        return _launch.zeros(n_points, dtype=wp.float32, device=polyline.device)
    return _arc_length_table(polyline, closed=False)


def polyline_downsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Downsample a polyline to a minimum arc-length spacing between kept points.

    Greedily keeps the first point, then each subsequent point at least ``step_size`` of arc
    length beyond the previously kept point.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    step_size
        Minimum arc-length distance between kept points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]), so the closing
        edge counts toward the spacing.

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` downsampled polyline of ``m`` points. The input is returned unchanged for fewer
        than two points.

    See Also
    --------
    [`polyline_simplify`][ordito.polyline.polyline_simplify]
        Drops points by *shape* error rather than by spacing.
    [`polyline_upsample`][ordito.polyline.polyline_upsample]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    line = wp.array([[0.1 * i, 0.0, 0.0] for i in range(11)], dtype=wp.vec3)
    kept = od.polyline.polyline_downsample(line, 0.25)
    print(kept.shape)  # (4,)
    print(kept.numpy()[:, 0].round(2))  # [0.  0.3 0.6 0.9]
    ```
    """
    device = polyline.device
    if polyline.size < 2:
        return polyline

    # ``closed`` walks an ``n + 1``-entry table whose last entry is the closing segment's end, or,
    # when the input already repeats its first point, a stand-in the walks stop short of
    # (``kernels/polyline.seam_repeats_first``) -- the closure is decided on the device.
    n_table = polyline.size + 1 if closed else polyline.size
    if wp.get_device(device).is_cuda and n_table >= _DOWNSAMPLE_DOUBLING_FROM:
        cumulative = _arc_length_table(polyline, closed=closed)
        keep = _launch.zeros(n_table, dtype=wp.int32, device=device)
        _greedy_downsample_doubling(cumulative, step_size, polyline, closed, keep)
        out_points, _ = _gather_kept(keep, polyline, return_indices=False)
        return out_points
    # The serial walk forms the arc-length table as it goes (no table is passed), and writes the
    # kept points' indices compacted behind their count, so the gather reads them as they are.
    indices = _launch.empty(n_table + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.greedy_downsample_indices,
        dim=1,
        inputs=[None, wp.float32(step_size), polyline, wp.int32(closed), indices],
        device=device,
    )
    # Sizes the output: the one host readback.
    n_kept = int(read_scalar(indices, 0))
    out_points = _launch.empty(n_kept, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_polyline.gather_walked_points,
        dim=n_kept,
        inputs=[polyline, indices, out_points],
        device=device,
    )
    return out_points


def _greedy_downsample_doubling(
    cumulative: wp.array[wp.float32],
    step_size: float,
    polyline: wp.array[wp.vec3] | None,
    closed: bool,
    out_keep: wp.array[wp.int32],
) -> None:
    """
    Mark the greedy walk's kept points by pointer-doubling its step function.

    The kept set is the orbit of point 0 under "the next point at least ``step_size`` further
    along", so building that step function for every point at once
    ([`greedy_successors`][ordito.kernels.polyline.greedy_successors]) turns an ``n``-step walk
    into a logarithmic number of rounds of jumping several pointers along it. The answer is the
    serial walk's, exactly and not approximately: the successor search evaluates the same float32
    comparison the walk does, so the two masks agree bit for bit -- verified over 27 shapes
    including exact ties and heavily clustered spacing. ``polyline`` and ``closed`` are the closure
    the walk stops short of (``greedy_successors``); an open table needs no polyline.
    """
    device = cumulative.device
    n = cumulative.size
    successor = _launch.empty(n, dtype=wp.int32, device=device)
    # Also marks the first point kept, which the walk always does; the caller zeroed the rest.
    _launch.launch(
        kernel_polyline.greedy_successors,
        dim=n,
        inputs=[cumulative, wp.float32(step_size), polyline, wp.int32(closed), successor, out_keep],
        device=device,
    )
    squared = _launch.empty(n, dtype=wp.int32, device=device)
    # Several pointers a round (``kernels/graph.pointer_jump_schedule``): the orbit is at most
    # ``n`` long and a round multiplies the covered prefix of it by ``hops``.
    hops, rounds = kernel_graph.pointer_jump_schedule(n + 1)
    for _ in range(rounds):
        _launch.launch(
            kernel_polyline.double_greedy_orbit,
            dim=n,
            inputs=[successor, out_keep, hops, squared, out_keep],
            device=device,
        )
        successor, squared = squared, successor


def polyline_simplify(
    polyline: wp.array[wp.vec3], tol: float, *, closed: bool = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """

    Simplify a polyline with the Ramer-Douglas-Peucker algorithm.

    Drops interior vertices whose perpendicular distance to the chord spanning a kept sub-range is
    at most ``tol``; the first and last vertices are always retained. The recursion is evaluated
    **level-synchronously** rather than depth-first -- one round of every point in parallel per
    level of the split tree, so the cost is the tree's *depth* (about ``log2(n)`` on a mesh
    boundary loop) rather than one thread's walk of the whole tree. The accepted set is identical
    either way, since breadth-first and depth-first evaluation of the same recursion accept the
    same points.


    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    tol
        Maximum Euclidean distance allowed between a dropped vertex and the retained chord.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]). ``indices``
        then refer to the *closed* polyline, so the shared start/end vertex is preserved at both
        ends.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(m,)`` and ``(m,)`` arrays ``(simplified, indices)`` on ``polyline.device``: the retained
        vertices and the sorted indices into the input such that
        ``polyline[indices] == simplified``. An empty input yields two empty arrays; a single point
        is returned unchanged with ``indices == [0]``.

    Notes
    -----
    The round loop runs **on device** with no readback. Up to a few thousand points on CUDA, and at
    every size on the CPU device, it runs as a single block whose lanes share the points, with a
    block barrier between the steps of a round; a longer polyline on CUDA drives the four launches
    with ``wp.capture_while`` exactly as
    [`polyline_triangulate`][ordito.polyline.polyline_triangulate]'s ear rounds are driven. Both
    forms run the same per-point steps and accept the same points. The one host synchronisation in
    the call is the compaction that follows the loop, which reads back the kept count in order to
    size ``indices``; ``closed=True`` reaches its closing point by wrapping the index, so it copies
    nothing and reads back nothing more.

    **The depth is bounded by the accepted count, not by ``n``, and that is why there is no round
    cap and no serial fallback here.** Every root-to-leaf path of the split tree accepts one point
    per level, so ``rounds <= kept + 1`` -- a deep tree is precisely an input that accepts most of
    its points, which is also the input the recursion does the most work on. A spiral's depth tracks
    its *turn count* rather than ``n``, while a power curve, a geometric staircase and a decaying
    sawtooth are all shallower than a boundary loop.

    See Also
    --------
    [`polyline_downsample`][ordito.polyline.polyline_downsample]
        Drops points by *spacing* rather than by shape error.

    Examples
    --------
    The near-collinear point ``1`` is within ``tol`` of the chord and is dropped; the peak is kept:

    ```python
    import warp as wp

    import ordito as od

    polyline = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.01, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=wp.vec3,
    )
    simplified, indices = od.polyline.polyline_simplify(polyline, 0.1)
    print(indices.numpy().tolist())  # [0, 2, 3, 4]
    ```
    """
    device = polyline.device
    n = polyline.size
    if n == 0:
        return polyline, _launch.empty(0, dtype=wp.int32, device=device)
    # ``closed`` runs over ``n + 1`` entries, the last being the first point again, reached by
    # wrapping the index rather than through a ``polyline_close`` copy; when the input already
    # repeats its first point that entry is a dead slot, decided on the device
    # (``kernels/polyline.rdp_loop_length``). A polyline of fewer than two points is its own loop.
    wrap_open = closed and n >= 2
    n_entries = n + 1 if wrap_open else n
    squared_tolerance = wp.float32(tol * tol)
    if n_entries > 2 and (
        not wp.get_device(device).is_cuda or n_entries <= kernel_polyline.RDP_ONE_BLOCK_MAX
    ):
        # The whole round loop as one block (``kernels/polyline.rdp_simplify_block``): at this size
        # a round is too little work to fill the device, and the launch form's cost is the
        # conditional graph it records on every call. The block also compacts what it kept: the
        # indices into the span table's first row, the points into a buffer as long as the input,
        # and the count after them, so the results are prefixes of the two.
        spans = odt.empty_2d((4, n_entries), wp.int32, device=device)
        span_values = odt.empty_2d((2, n_entries), wp.float32, device=device)
        points = _launch.empty(n_entries, dtype=wp.vec3, device=device)
        _launch.launch_tiled(
            kernel_polyline.rdp_simplify_block,
            dim=[1],
            inputs=[polyline, wp.int32(wrap_open), squared_tolerance],
            outputs=[spans, span_values, points],
            block_dim=kernel_polyline.RDP_BLOCK_DIM,
            device=device,
        )
        # Sizes the output: the one host readback.
        n_kept = int(read_scalar(odt.as_dense(spans[1]), 0))
        return (odt.as_dense(points[:n_kept]), odt.as_dense(odt.as_dense(spans[0])[:n_kept]))
    span_lo = _launch.empty(n_entries, dtype=wp.int32, device=device)
    span_hi = _launch.empty(n_entries, dtype=wp.int32, device=device)
    keep = _launch.empty(n_entries, dtype=wp.int32, device=device)
    # state = [levels run, loop condition], seeded by ``rdp_seed_spans`` and then written on device
    # so the round loop needs no readback -- ``polyline_triangulate``'s ear rounds are driven the
    # same way.
    state = _launch.empty(kernel_array.LOOP_STATE_SIZE, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.rdp_seed_spans,
        dim=n_entries,
        inputs=[polyline, wp.int32(wrap_open), span_lo, span_hi, keep, state],
        device=device,
    )
    if n_entries > 2:  # fewer than three entries have no interior to drop
        span_max = _launch.empty(n_entries, dtype=wp.float32, device=device)
        span_argmax = _launch.empty(n_entries, dtype=wp.int32, device=device)
        squared_distances = _launch.empty(n_entries, dtype=wp.float32, device=device)

        def split_round() -> None:
            _launch.launch(
                kernel_polyline.rdp_begin_round,
                dim=n_entries,
                inputs=[state, span_max, span_argmax],
                device=device,
            )
            _launch.launch(
                kernel_polyline.rdp_span_max,
                dim=n_entries,
                inputs=[polyline, span_lo, span_hi, squared_distances, span_max],
                device=device,
            )
            _launch.launch(
                kernel_polyline.rdp_span_argmax,
                dim=n_entries,
                inputs=[span_lo, squared_distances, span_max, span_argmax],
                device=device,
            )
            _launch.launch(
                kernel_polyline.rdp_split_spans,
                dim=n_entries,
                inputs=[squared_tolerance, span_max, span_argmax, span_lo, span_hi, state, keep],
                device=device,
            )

        condition = odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW])
        run_device_loop(device, condition, split_round)

    simplified, indices = _gather_kept(keep, polyline, return_indices=True)
    assert indices is not None
    return simplified, indices


def _gather_kept(
    flags: wp.array[wp.int32], polyline: wp.array[wp.vec3], *, return_indices: bool
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32] | None]:
    """
    Compact the entries whose ``0`` / ``1`` ``flags`` are set: ``(points, indices or None)``.

    The flags are scanned in place: the tail is the output size (the one readback) and the scan's
    steps are where each kept entry lands, so one launch writes the points -- and, with
    ``return_indices``, their indices -- directly. Entry ``n`` of a ``closed=True`` table is the
    first point again (``kernels/polyline.gather_kept_points``).
    """
    device = polyline.device
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    n_kept = int(read_scalar(flags))
    out_points = _launch.empty(n_kept, dtype=wp.vec3, device=device)
    out_indices = _launch.empty(n_kept, dtype=wp.int32, device=device) if return_indices else None
    _launch.launch(
        kernel_polyline.gather_kept_points,
        dim=flags.size,
        inputs=[flags, polyline, out_points, out_indices],
        device=device,
    )
    return out_points, out_indices


def polyline_resample(
    polyline: wp.array[wp.vec3], num_points: int, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Resample a polyline to a fixed number of points evenly spaced by arc length.

    Points are sampled at ``num_points`` arc lengths evenly spanning ``[0, total_length]`` and
    linearly interpolated between the bracketing vertices (matching ``numpy.interp`` semantics).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    num_points
        Number of output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]). The result has
        ``num_points`` *distinct* points: the duplicated seam is dropped.

    Returns
    -------
    wp.array[wp.vec3]
        ``(num_points,)`` resampled polyline. An empty input is returned unchanged; a single
        point is repeated ``num_points`` times.

    See Also
    --------
    [`polyline_upsample`][ordito.polyline.polyline_upsample]
        Targets a step size instead of a point count.

    Examples
    --------
    Seven points evenly spaced along the open square's arc length of 3:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    resampled = od.polyline.polyline_resample(square, 7)
    print(resampled.shape)  # (7,)
    print(resampled.numpy()[1])  # [0.5 0.  0. ]
    ```
    """
    device = polyline.device
    n = polyline.size
    if n == 0:
        return polyline
    out_points = _launch.empty(num_points, dtype=wp.vec3, device=device)
    if n == 1:
        _launch.launch(
            kernel_polyline.broadcast_first_point,
            dim=num_points,
            inputs=[polyline, out_points],
            device=device,
        )
        return out_points

    # ``closed`` samples ``num_points + 1`` arc lengths of the loop and computes all but the last,
    # the duplicated seam, so the result has ``num_points`` *distinct* points and is a clean cyclic
    # ring. The closure is decided on the device (``kernels/polyline.resample_interp``).
    cumulative = _arc_length_table(polyline, closed=closed)
    _launch.launch(
        kernel_polyline.resample_interp,
        dim=num_points,
        inputs=[
            polyline,
            cumulative,
            wp.int32(num_points + 1 if closed else num_points),
            wp.int32(closed),
            out_points,
        ],
        device=device,
    )
    return out_points


def _arc_length_table(polyline: wp.array[wp.vec3], *, closed: bool) -> wp.array[wp.float32]:
    """
    Cumulative arc length of a polyline of at least two points, ``closed`` over ``n`` segments.

    [`cumulative_arc_length`][ordito.polyline.cumulative_arc_length]'s body, and with ``closed``
    the table of the loop without a ``polyline_close`` copy: entry ``n`` is the closing segment's
    end, or -- when the input already repeats its first point -- a zero-length stand-in its
    consumers skip (``kernels/polyline.arc_segment_lengths``). The kernel writes the leading zero
    and each segment's length one slot along, and the inclusive scan of everything after the zero
    runs in place, the one-allocation idiom of ``array.counts_to_offsets`` in float32.
    """
    device = polyline.device
    n_points = polyline.size
    n_segments = n_points if closed else n_points - 1
    cumulative = _launch.empty(n_segments + 1, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_polyline.arc_segment_lengths,
        dim=n_segments,
        inputs=[polyline, wp.int32(closed), cumulative],
        device=device,
    )
    lengths = cumulative[1:]
    _launch.array_scan(lengths, out_array=lengths, inclusive=True)
    return cumulative


def polyline_radius(
    polyline: wp.array[wp.vec3],
    reduction: Literal["min", "max", "mean", "median"] = "min",
    center: wp.vec3 | None = None,
    normal: wp.vec3 | None = None,
    *,
    closed: bool = False,
) -> float:
    """
    Radius of a polyline projected onto the plane through ``center`` with the given ``normal``.

    Each segment is projected onto the plane and its closest point to ``center`` is found; the
    per-segment distances to ``center`` are then reduced. With ``reduction="min"`` this is the
    inner radius (nearest point, not vertex, on the polyline).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    reduction
        Reduction over the per-segment radial distances: ``"min"``, ``"max"``, ``"mean"``, or
        ``"median"``. Defaults to ``"min"``.
    center
        Plane origin. Defaults to [`polyline_centroid`][ordito.polyline.polyline_centroid].
    normal
        Plane normal (need not be unit). Defaults to
        [`polyline_normal`][ordito.polyline.polyline_normal].

        These two are the circle's frame, not a cutting plane, so they deliberately keep their own
        names and their own order rather than the ``(plane_normal, plane_origin)`` convention every
        plane argument in the package follows -- both are keyword-defaulted here, and calling them
        a plane would misdescribe the geometry.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]), so the closing
        segment contributes a radial distance like any other, and the default ``center`` /
        ``normal`` are the closed polyline's.

    Returns
    -------
    float
        The reduced radius.

    Raises
    ------
    ValueError
        If ``reduction`` is not one of the supported values, if the polyline has fewer than two
        points, or if it has fewer than three and either ``center`` or ``normal`` is left at its
        default (both defaults need a plane derived from the polyline, which a single segment does
        not determine).

    See Also
    --------
    [`polyline_centroid`][ordito.polyline.polyline_centroid]
    [`polyline_normal`][ordito.polyline.polyline_normal]
        The two defaults for the plane.
    [`median`][ordito.reduce.median]

    Examples
    --------
    Each side of a 4 x 2 rectangle contributes its point nearest the centre, so the smallest
    radius reaches the long sides and the largest the short ones:

    ```python
    import warp as wp

    import ordito as od

    rectangle = wp.array(
        [[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.polyline_radius(rectangle, "min", closed=True))  # 1.0
    print(od.polyline.polyline_radius(rectangle, "max", closed=True))  # 2.0
    ```
    """
    if reduction not in ("min", "max", "mean", "median"):
        # Before the closure test, which is device work and a readback: a rejected argument
        # should not cost a launch first.
        raise ValueError(f"unsupported reduction {reduction!r}")
    device = polyline.device
    n = polyline.size
    if n < 2:
        raise ValueError("polyline_radius requires at least two points")
    # ``closed`` adds the segment back to the first point when the input does not already end
    # there. The kernels reach it by wrapping the index rather than through a ``polyline_close``
    # copy, and decide whether it is there themselves (``closing_segment_flag``), so the host asks
    # only where the answer decides something on the host: an error below, for fewer than four
    # points, or the size of the distances the median sorts -- a stand-in segment would move it.
    n_segments = n - 1
    if closed and (n <= 3 or reduction == "median") and not is_closed(polyline):
        n_segments = n

    if (center is None or normal is None) and n_segments < 2:
        # ``polyline_centroid`` has no such floor, but ``polyline_normal`` needs three distinct
        # points to fit a plane, so a 2-point input can only reach past here with both supplied
        # explicitly -- raising ``polyline_radius``'s own message rather than deferring to
        # ``polyline_normal``'s, whose "three points" precondition this function does not itself
        # document anywhere else.
        raise ValueError(
            "polyline_radius requires at least three points when 'center' or 'normal' is not "
            "supplied explicitly"
        )
    # The default centre and normal are ``polyline_centroid``'s and ``polyline_normal``'s, built by
    # one accumulation pass and consumed where the distances are, so neither crosses to the host.
    # Only a loop of three entries needs the closure first: ``polyline_normal`` rejects a closed
    # one, and a ``closed=True`` loop ends at its first point by construction.
    if normal is None and n_segments == 2 and (closed or is_closed(polyline)):
        raise ValueError("polyline_normal requires at least three points")
    # The midpoint sums only for a defaulted centre, the Newell sum only for a defaulted normal.
    n_open = n - 1 if center is None else 0
    wrap_open = int(closed) if center is None else 0
    blocks = kernel_reduce.blocks_1d(n)
    if reduction != "median" and blocks == 1:
        # A polyline one block covers: the frame and the reduction in that one block
        # (``kernels/polyline.radius_reduce_block``), the same sums as the two launches below.
        result = _launch.empty(2, dtype=wp.float32, device=device)
        _launch.launch_tiled(
            kernel_polyline.radius_reduce_block,
            dim=1,
            inputs=[
                polyline,
                wp.int32(n_open),
                wp.int32(wrap_open),
                wp.vec3() if center is None else center,
                wp.vec3() if normal is None else normal,
                wp.int32(1 if center is None else 0),
                wp.int32(1 if normal is None else 0),
                wp.int32(closed),
                _RADIUS_REDUCTIONS[reduction][0],
                result,
            ],
            block_dim=TILE_1D,
            device=device,
        )
        return _radius_result(result, reduction)
    frame = None
    if center is None or normal is None:
        frame = _accumulate_frame(polyline, n_open, wrap_open, with_normal=normal is None)
    plane = [
        polyline,
        frame,
        wp.vec3() if center is None else center,
        wp.vec3() if normal is None else normal,
        wp.int32(1 if center is None else 0),
        wp.int32(1 if normal is None else 0),
    ]
    if reduction == "median":
        distances = odt.empty_1d(n_segments, wp.float32, device=device)
        _launch.launch(
            kernel_polyline.radius_distances,
            dim=n_segments,
            inputs=plane,
            outputs=[distances],
            device=device,
        )
        return od.reduce.median(distances)
    # ``min`` / ``max`` / ``mean`` are folded where the distances are computed
    # (``kernels/polyline.radius_reduce``): no distance buffer, no reduction launch over it, and one
    # readback of the value and, for the mean, the segment count the kernel decided.
    kind, identity = _RADIUS_REDUCTIONS[reduction]
    stride = 0
    if reduction == "mean":
        # The sum and the segment count as fixed-order partials, so the mean repeats on CUDA.
        result, stride = _launch.block_partials(blocks, 2, wp.float32, device)
    else:
        result = _launch.full(1, identity, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_polyline.radius_reduce,
        dim=blocks,
        inputs=[*plane, wp.int32(closed), kind, stride, result],
        block_dim=TILE_1D,
        device=device,
    )
    return _radius_result(result, reduction)


def _radius_result(result: wp.array[wp.float32], reduction: str) -> float:
    """Read ``polyline_radius``'s folded answer: the value, or the mean from its sum and count."""
    if reduction != "mean":
        return float(read_scalar(result, 0))
    total, count = read_values(result, 0, 2)
    return float(total) / float(count)


def _accumulate_frame(
    polyline: wp.array[wp.vec3], n_segments: int, wrap_open: int, *, with_normal: bool
) -> wp.array[wp.float32]:
    """
    Launch ``kernels/polyline.accumulate_radius_frame`` into a fresh ``RADIUS_FRAME_SIZE`` buffer.

    The shared frame pass of [`polyline_centroid`][ordito.polyline.polyline_centroid],
    [`polyline_normal`][ordito.polyline.polyline_normal] and
    [`polyline_radius`][ordito.polyline.polyline_radius]: the length-weighted midpoint sums over
    ``n_segments`` segments plus the closing one ``wrap_open`` asks for, and, with
    ``with_normal``, Newell's sum over the loop.
    """
    device = polyline.device
    blocks = kernel_reduce.blocks_1d(polyline.size)
    # Fixed-order partials, so the frame (and every answer read off it) repeats on CUDA.
    frame, stride = _launch.block_partials(
        blocks, kernel_polyline.RADIUS_FRAME_SIZE, wp.float32, device
    )
    _launch.launch_tiled(
        kernel_polyline.accumulate_radius_frame,
        dim=blocks,
        inputs=[
            polyline,
            wp.int32(n_segments),
            wp.int32(wrap_open),
            wp.int32(1 if with_normal else 0),
            stride,
            frame,
        ],
        block_dim=TILE_1D,
        device=device,
    )
    return frame


def polyline_angles(polyline: wp.array[wp.vec3], *, closed: bool = False) -> wp.array[wp.float32]:
    """
    Angles between consecutive segments at each vertex of a polyline.

    Returns one angle per point. For an open polyline the two endpoints get an angle of ``0``;
    for a closed polyline (first point equal to last) the turning angles wrap around.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][ordito.polyline.polyline_close]). The result
        still has one angle per *original* point, and the turning angles wrap around.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` angles in radians on ``polyline.device``. All zeros for fewer than two points.

    See Also
    --------
    [`is_closed`][ordito.polyline.is_closed]
        What decides the wrap-around when ``closed`` is left ``False``.

    Examples
    --------
    An open polyline's endpoints get ``0``; closing the square turns every corner by ``pi / 2``:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3
    )
    print(od.polyline.polyline_angles(square).numpy().round(4))  # [0.     1.5708 1.5708 0.    ]
    angles = od.polyline.polyline_angles(square, closed=True)
    print(angles.numpy().round(4))  # [1.5708 1.5708 1.5708 1.5708]
    ```
    """
    device = polyline.device
    n = polyline.size
    if n < 2:
        return _launch.zeros(n, dtype=wp.float32, device=device)

    # One launch writes every angle in its final slot, deciding the closure on the device: no
    # readback decides the wrap-around, and ``closed=True`` reaches the closing segment by wrapping
    # the index rather than through a ``polyline_close`` copy. See
    # ``kernels/polyline.vertex_turning_angles``.
    angles = _launch.empty(n, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_polyline.vertex_turning_angles,
        dim=n,
        inputs=[polyline, wp.int32(1 if closed else 0)],
        outputs=[angles],
        device=device,
    )
    return angles


def polyline_triangulate(polyline: wp.array[wp.vec3]) -> odt.Array2dInt32:
    """
    Triangulate the simple planar polygon bounded by a closed 3D polyline (ear clipping).

    The polyline is treated as the boundary of a simple polygon, which is filled with triangles
    whose vertices are the polyline vertices themselves -- no new (Steiner) points are introduced. A
    simple ``n``-gon yields ``n - 2`` non-overlapping triangles that cover the polygon. The loop is
    first projected onto its best-fit plane (via
    [`polyline_normal`][ordito.polyline.polyline_normal]) so any planar loop works, not only ones
    lying in the ``xy`` plane.

    The implementation is a GPU-parallel port of ``ear_clipping.cpp`` from libigl: a strictly
    convex polygon uses a single fan, any other clips a maximal independent set of ears per round
    until the polygon is exhausted. Returned faces are consistently wound counter-clockwise with
    respect to the loop's turning direction; the exact set of triangles may differ from a sequential
    ear clip, but every triangulation of a simple polygon has ``n - 2`` faces. The result is the
    same on every run.

    An ear whose tip is collinear with its neighbours to within the points' ``float32`` rounding is
    clipped only in a round with no other ear, so zero-area slivers appear only where the polygon
    leaves no choice (a vertex on a zero-length side, or within rounding of a non-adjacent side).

    The cost is set by the **round count**, and the round count by how many ears the
    independent-set rule can retire at once. Competing ears are ranked by a bijective hash of their
    ring index rather than by the index itself, which is what keeps that logarithmic: under the raw
    index an alternating star lets the ear at ``i - 2`` suppress the ear at ``i`` for every ``i``,
    so one ear is clipped per round and the loop runs its full ``n``-round cap. A long run of
    near-collinear points is the exception: it can only be retired from its two ends, a vertex or
    so a round.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices. A duplicated closing point is dropped, with the
        predicate [`polyline_open`][ordito.polyline.polyline_open] applies.

    Returns
    -------
    odt.Array2dInt32
        ``(m, 3)`` triangle vertex indices into ``polyline`` on ``polyline.device``. ``m`` is
        ``n - 2`` for a simple polygon; a degenerate or self-intersecting loop may yield fewer
        (a partial triangulation). Empty ``(0, 3)`` for fewer than three points.

    Notes
    -----
    The whole clip runs **on device**. A short ring is clipped by a single block that runs every
    round itself; a long one by a round loop driven by ``wp.capture_while`` over a device-side
    condition, whose ear tests look up the vertices near each candidate ear in a uniform grid
    rather than walking the whole ring, so a round costs about linear work. Two readbacks are left
    in the whole function, both structural: one carrying the ring length (whether the last point
    repeats the first), the loop's orientation, its reflex count and its bounding box, which
    together decide every launch dimension, the grid and the convex fan fast path; and the face
    count, which sizes the returned slice. The plane frame is accumulated and consumed on
    the device and never crosses to the host.

    Every path produces the same triangulation up to row order. Faces are appended through an atomic
    counter, so on CUDA their row order changes from call to call, unless
    [`is_deterministic`][ordito.config.is_deterministic]: then the rows are listed by ascending
    middle index (the tip of the ear each one clipped).

    See Also
    --------
    [`polyline_normal`][ordito.polyline.polyline_normal]
    [`polyline_close`][ordito.polyline.polyline_close]

    Examples
    --------
    An L-shaped hexagon is filled with ``6 - 2 = 4`` triangles over its own vertices:

    ```python
    import warp as wp

    import ordito as od

    l_shape = wp.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=wp.vec3
    )
    faces = od.polyline.polyline_triangulate(l_shape)
    print(faces.shape)  # (4, 3)
    ```
    """
    device = polyline.device
    n = polyline.size
    if n < 3:
        return odt.empty_2d((0, 3), wp.int32, device=device)

    # The plane frame is built and consumed entirely on device: one single-block accumulation, which
    # also decides whether the last point repeats the first, then the turning pass, which projects
    # every point into the frame it stored as it reads it.
    sums = _launch.zeros(kernel_polyline.RING_SUMS_SIZE, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_polyline.accumulate_loop_frame,
        dim=(1,),
        inputs=[polyline, sums],
        block_dim=kernel_polyline.LOOP_FRAME_BLOCK_DIM,
        device=device,
    )
    points2d = _launch.empty(n, dtype=wp.vec2, device=device)
    _launch.launch_tiled(
        kernel_polyline.project_turning_angle,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[polyline, sums, points2d],
        block_dim=TILE_1D,
        device=device,
    )
    _, faces, _ = _triangulate_ring(points2d, sums, write_fan=True)
    assert faces is not None
    return faces


def polyline_triangulate_from_offsets(
    polylines: wp.array[wp.vec3], offsets: wp.array[wp.int32]
) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]:
    """
    Triangulate several packed closed 3D polylines at once (ear clipping), adding no new points.

    Loop ``r`` is ``polylines[offsets[r] : offsets[r + 1]]``. Each loop gets the triangles
    [`polyline_triangulate`][ordito.polyline.polyline_triangulate] gives it alone (up to row
    order), but the loops are triangulated together rather than one call after another: the short
    ones one block per loop in a single launch, the long ones side by side in one round loop.

    Parameters
    ----------
    polylines
        ``(n,)`` points of every loop, packed loop after loop.
    offsets
        ``(n_loops + 1,)`` loop offsets into ``polylines``, total-terminated.

    Returns
    -------
    faces : odt.Array2dInt32
        ``(m, 3)`` triangle vertex indices into ``polylines`` (not into each loop), loop after
        loop. A simple ``k``-point loop contributes ``k - 2``; a degenerate or self-intersecting
        one may contribute fewer, and a loop of fewer than three points none. Each loop's rows are
        in the order [`polyline_triangulate`][ordito.polyline.polyline_triangulate] describes.
    face_offsets : wp.array[wp.int32]
        ``(n_loops + 1,)`` offsets of each loop's rows in ``faces``, total-terminated.

    Raises
    ------
    RuntimeError
        If ``polylines`` and ``offsets`` are on different devices.

    Notes
    -----
    Two small readbacks: the total rows the loops may write, and the rows they did.

    See Also
    --------
    [`polyline_triangulate`][ordito.polyline.polyline_triangulate]

    Examples
    --------
    A square and a triangle packed into one buffer give two triangles and one:

    ```python
    import warp as wp

    import ordito as od

    polylines = wp.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]], dtype=wp.vec3
    )
    offsets = wp.array([0, 4, 7], dtype=wp.int32)
    faces, face_offsets = od.polyline.polyline_triangulate_from_offsets(polylines, offsets)
    print(faces.shape)  # (3, 3)
    print(face_offsets.numpy().tolist())  # [0, 2, 3]
    print(sorted(faces.numpy()[2].tolist()))  # [4, 5, 6]
    ```
    """
    require_same_device(polylines=polylines, offsets=offsets)
    device = polylines.device
    n_loops = offsets.size - 1
    if n_loops <= 0:
        return (
            odt.empty_2d((0, 3), wp.int32, device=device),
            _launch.zeros(1, dtype=wp.int32, device=device),
        )
    # Each loop's reserved face rows, scanned into bases, and the number of loops too long for the
    # block launch, read back together: the reservation sizes the shared face buffer, and a long
    # loop adds the packed round loop. One zeroed buffer holds the bases, the long-loop count after
    # them and then the face counts. The frame pass writes the reservation beside each loop's frame.
    buffer = _launch.zeros(2 * n_loops + 3, dtype=wp.int32, device=device)
    bases = _launch.view(buffer, 0, n_loops + 2)
    bases_scan = _launch.view(buffer, 0, n_loops + 1)
    face_offsets = _launch.view(buffer, n_loops + 2, 2 * n_loops + 3)
    sums = _launch.zeros(n_loops * kernel_polyline.RING_SUMS_SIZE, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_polyline.accumulate_loop_frames,
        dim=(n_loops,),
        inputs=[polylines, offsets, wp.int32(kernel_polyline.EAR_ONE_BLOCK_MAX), sums, bases],
        block_dim=kernel_polyline.LOOP_FRAME_BLOCK_DIM,
        device=device,
    )
    _launch.array_scan(bases_scan, out_array=bases_scan, inclusive=True)
    n_reserved, n_long = (int(value) for value in read_values(bases, n_loops, 2))

    if n_reserved == 0:
        return odt.empty_2d((0, 3), wp.int32, device=device), face_offsets
    faces, ends = _triangulate_rings(
        polylines, offsets, bases, sums, n_reserved, face_offsets, n_long
    )
    _launch.array_scan(face_offsets, out_array=face_offsets, inclusive=True)
    # Sizes the output: the faces the loops actually wrote.
    n_rows = int(read_scalar(face_offsets))
    out_faces = odt.empty_2d((n_rows, 3), wp.int32, device=device)
    if n_rows > 0:
        _launch.launch(
            kernel_polyline.compact_ring_faces,
            dim=n_reserved,
            inputs=[faces, bases_scan, ends, face_offsets, out_faces],
            device=device,
        )
        if is_deterministic() and n_rows > 1:
            # Each loop's tips lie in its own range of ``polylines``, so one sort of every row by
            # tip keeps the loops in order and lists each loop's rows by ascending tip.
            return _faces_in_tip_order(out_faces, polylines.size), face_offsets
    return odt.as_array2d(out_faces, wp.int32), face_offsets


def _triangulate_rings(
    polylines: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    bases: wp.array[wp.int32],
    sums: wp.array[wp.float32],
    n_reserved: int,
    face_counts: wp.array[wp.int32],
    n_long: int,
) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]:
    """
    Triangulate every packed loop into reserved rows: ``(reserved faces, end rows)``.

    Loop ``r`` writes its faces from row ``bases[r]`` of the ``(max(n_reserved, 1), 3)`` buffer and
    ``ends[r]`` is the row after its last; loops shorter than three points write none, and the
    zeroed ``(n_loops + 1,)`` ``face_counts`` gets each loop's face count in slot ``r + 1``. One
    block per loop projects it and clips it when it has at most ``EAR_ONE_BLOCK_MAX`` points; the
    ``n_long`` longer ones are triangulated together by ``_triangulate_long_rings``. ``sums``
    holds every loop's frame (``kernels/polyline.accumulate_loop_frames``).
    """
    device = polylines.device
    n_loops = offsets.size - 1
    n_points = polylines.size
    points2d = _launch.empty(n_points, dtype=wp.vec2, device=device)
    faces = odt.empty_2d((max(n_reserved, 1), 3), wp.int32, device=device)
    rows = kernel_polyline.RING_SCRATCH_ROWS if n_long > 0 else 5
    ring = odt.empty_2d((rows, n_points), wp.int32, device=device)
    ends = _launch.empty(n_loops, dtype=wp.int32, device=device)
    _launch.launch_tiled(
        kernel_polyline.triangulate_rings,
        dim=(n_loops,),
        inputs=[
            polylines,
            offsets,
            bases,
            wp.int32(kernel_polyline.EAR_ONE_BLOCK_MAX),
            sums,
            points2d,
            faces,
            ring,
            ends,
            face_counts,
        ],
        block_dim=kernel_polyline.EAR_BLOCK_DIM,
        device=device,
    )
    if n_long > 0:
        _triangulate_long_rings(
            polylines, offsets, bases, sums, points2d, ring, faces, ends, face_counts
        )
    return odt.as_array2d(faces, wp.int32), ends


def _triangulate_long_rings(
    polylines: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    bases: wp.array[wp.int32],
    sums: wp.array[wp.float32],
    points2d: wp.array[wp.vec2],
    ring: odt.Array2dInt32,
    faces: odt.Array2dInt32,
    ends: wp.array[wp.int32],
    face_counts: wp.array[wp.int32],
) -> None:
    """
    Triangulate every loop ``triangulate_rings`` left, those past ``EAR_ONE_BLOCK_MAX`` points.

    ``_triangulate_ring``'s grid and round loop over every such loop side by side, in one
    captured loop (``kernels/polyline.ring_compute_ears``), each loop with its own grid, state
    row and face counter, all seeded on the device (``seed_long_rings``, which also fans a convex
    loop), so nothing is read back. The loops' grids share one cell table sized by their bound
    (``RING_CELLS_PER_POINT``), so it needs no loop's size either.
    """
    device = points2d.device
    n_points = points2d.size
    n_loops = offsets.size - 1
    rounds = _launch.zeros(
        kernel_polyline.RINGS_HEADER + n_loops * kernel_polyline.RING_STATE_SIZE,
        dtype=wp.int32,
        device=device,
    )
    grid = _launch.empty(n_loops * kernel_polyline.RING_GRID_SIZE, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_polyline.seed_long_rings,
        dim=(n_loops,),
        inputs=[
            polylines,
            offsets,
            bases,
            wp.int32(kernel_polyline.EAR_ONE_BLOCK_MAX),
            sums,
            points2d,
            faces,
            ends,
            face_counts,
            rounds,
            grid,
        ],
        block_dim=kernel_polyline.LONG_RING_BLOCK_DIM,
        device=device,
    )
    n_cells = (
        kernel_polyline.RING_CELLS_PER_POINT * n_points
        + kernel_polyline.RING_CELLS_PER_LOOP * n_loops
    )
    cell_keys = _launch.empty(2 * n_points, dtype=wp.int32, device=device)
    cell_points = _launch.empty(2 * n_points, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.ring_ear_grid_cells,
        dim=n_points,
        inputs=[points2d, offsets, rounds, grid, wp.int32(n_cells), ring],
        outputs=[cell_keys, cell_points],
        device=device,
    )
    _launch.radix_sort_pairs(cell_keys, cell_points, count=n_points, end_bit=n_cells.bit_length())
    cell_starts = _launch.empty(n_cells + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.init_ring_ear_clips,
        dim=n_cells + 1,
        inputs=[_launch.view(cell_keys, 0, n_points), offsets, rounds, ring],
        outputs=[cell_starts],
        device=device,
    )

    def clip_round() -> None:
        _launch.launch(
            kernel_polyline.ring_compute_ears,
            dim=n_points,
            inputs=[points2d, offsets, cell_starts, cell_points, grid, ring, rounds],
            device=device,
        )
        _launch.launch(
            kernel_polyline.ring_select_independent,
            dim=n_points,
            inputs=[offsets, rounds, ring],
            device=device,
        )
        _launch.launch(
            kernel_polyline.ring_clip_selected,
            dim=n_points,
            inputs=[rounds, ring],
            outputs=[faces],
            device=device,
        )
        _launch.launch(
            kernel_polyline.ring_ear_loop_continue,
            dim=n_loops,
            inputs=[rounds],
            outputs=[ends, face_counts],
            device=device,
        )

    condition = odt.as_dense(rounds[kernel_array.LOOP_CONDITION_VIEW])
    run_device_loop(device, condition, clip_round)


def triangulate_polygon(polygon: wp.array[wp.vec2]) -> tuple[wp.array[wp.vec2], wp.array[wp.int32]]:
    """
    Triangulate a simple 2D polygon by ear clipping, adding no new vertices.

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped.

    Returns
    -------
    ring : wp.array[wp.vec2]
        ``(n,)`` or ``(n - 1,)`` input ring with any repeated closing point removed; the vertices
        ``faces`` indexes.
    faces : wp.array[wp.int32]
        ``(3 * (n - 2),)`` flat triangle index buffer into ``ring``.

    Notes
    -----
    Differs from [`trimesh.creation.triangulate_polygon`][] in three ways, all of which follow from
    replacing the CPU polygon libraries (``mapbox_earcut`` / ``manifold3d`` / ``triangle``) with
    ordito's own GPU ear clipper, [`polyline_triangulate`][ordito.polyline.polyline_triangulate]:

    - The input is a ``wp.vec2`` ring, not a ``shapely.geometry.Polygon``, and **interior rings
      (holes) are not supported**.
    - No Steiner points are ever inserted, which is trimesh's ``force_vertices=True`` contract
      rather than its default.
    - There is no ``engine`` selection, and ``faces`` is flat rather than ``(m, 3)``, matching the
      face layout used throughout ordito.

    The ring must be a simple (non self-intersecting) polygon. A degenerate ring yields a partial
    triangulation with fewer than ``n - 2`` triangles rather than raising. The faces come in the
    order [`polyline_triangulate`][ordito.polyline.polyline_triangulate] describes: on CUDA it
    repeats from call to call only with [`is_deterministic`][ordito.config.is_deterministic].

    See Also
    --------
    [`polyline_triangulate`][ordito.polyline.polyline_triangulate]
    [`extrude_polygon`][ordito.creation.extrude_polygon]
    [`trimesh.creation.triangulate_polygon`][]

    Examples
    --------
    The repeated closing point of a square ring is dropped, and the square is split in two:

    ```python
    import warp as wp

    import ordito as od

    polygon = wp.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=wp.vec2)
    ring, faces = od.polyline.triangulate_polygon(polygon)
    print(ring.shape, faces.shape)  # (4,) (6,)
    ```
    """
    odt.ensure_ndim(polygon, 1, dtype=wp.vec2)
    ring, faces, _ = _triangulate_polygon(polygon, write_fan=True)
    assert faces is not None
    return ring, faces


def _triangulate_polygon(
    polygon: wp.array[wp.vec2], *, write_fan: bool
) -> tuple[wp.array[wp.vec2], wp.array[wp.int32] | None, bool]:
    """
    ``triangulate_polygon``, also returning whether the ring runs clockwise: ``(ring, faces, cw)``.

    Every face is wound like the ring (an ear is convex in the ring's own sense), so ``cw`` is the
    sign of their total signed area unless the faces enclose no area beyond rounding;
    [`extrude_polygon`][ordito.creation.extrude_polygon] reads it instead of summing the area.
    Without ``write_fan``, ``faces`` is ``None`` for a convex ring, whose triangulation is the fan
    ``(0, k + 1, k + 2)`` (``kernels/polyline.fan_face``) the caller then forms where it reads it,
    and no fan launch runs.
    """
    device = polygon.device
    n = polygon.size
    if n < 3:
        return polygon, _launch.empty(0, dtype=wp.int32, device=device), False

    sums = _launch.zeros(kernel_polyline.RING_SUMS_SIZE, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_polyline.accumulate_turning_angle,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[polygon, sums],
        block_dim=TILE_1D,
        device=device,
    )
    n_ring, faces, clockwise = _triangulate_ring(polygon, sums, write_fan=write_fan)
    # The ring is the input minus any repeated closing point, i.e. a prefix of it.
    ring = polygon[:n_ring].contiguous()
    return ring, None if faces is None else faces.reshape((-1,)), clockwise


def _triangulate_ring(
    points2d: wp.array[wp.vec2], sums: wp.array[wp.float32], *, write_fan: bool
) -> tuple[int, odt.Array2dInt32 | None, bool]:
    """
    Ear-clip a 2D ring whose turning pass has run: ``(ring length, (m, 3) faces, clockwise)``.

    ``points2d`` holds the ring plus, possibly, a repeated closing point; the caller's turning
    pass (``kernels/polyline.accumulate_turning_angle`` or ``project_turning_angle``) has written
    the closing flag, the turning angle, the reflex counts and the ring's box into ``sums``, and
    the one readback of ``sums`` carries them together. ``points2d`` is never written: a clockwise
    ring is read mirrored by the ear tests. A convex ring's faces are ``None`` unless
    ``write_fan`` (see ``_triangulate_polygon``).
    """
    device = points2d.device
    n = points2d.size
    # The only readback before the convex fast path returns: ring length, turning angle and the
    # reflex count of the oriented loop, all decided on device.
    # The eight slots are adjacent, ``RING_TURNING`` through the ring's box: one offset read.
    turning, reflex, reflex_mirrored, closing, *extent = read_values(
        sums, int(kernel_polyline.RING_TURNING), 8
    )
    n_ring = n - int(closing)
    # A clockwise ring is mirrored before the ear tests, and its reflex count is the mirror's.
    clockwise = bool(turning < 0.0)
    if n_ring < 3:
        return n_ring, odt.empty_2d((0, 3), wp.int32, device=device), clockwise

    if (reflex_mirrored if clockwise else reflex) == 0.0:
        if not write_fan:
            return n_ring, None, clockwise
        # The fan's faces are index triples, so a convex ring needs no orientation fix-up at all.
        out_faces = odt.empty_2d((n_ring - 2, 3), wp.int32, device=device)
        _launch.launch(
            kernel_polyline.fan_triangulate, dim=n_ring - 2, inputs=[out_faces], device=device
        )
        return n_ring, odt.as_array2d(out_faces, wp.int32), clockwise
    out_faces = odt.empty_2d((n_ring - 2, 3), wp.int32, device=device)

    # The ear tests read a clockwise ring mirrored (``kernels/polyline.ring_point``), so it is never
    # written mirrored, and a caller's ring needs no copy.
    mirror = wp.int32(1 if clockwise else 0)

    # One block runs every round (see ``ear_clip_block``): no graph to record, one launch, but an
    # O(ring) containment walk per corner, so only short rings take it (``EAR_ONE_BLOCK_MAX``).
    if n_ring <= kernel_polyline.EAR_ONE_BLOCK_MAX:
        ring = odt.empty_2d((5, n_ring), wp.int32, device=device)
        count_wp = _launch.empty(1, dtype=wp.int32, device=device)
        _launch.launch_tiled(
            kernel_polyline.ear_clip_block,
            dim=1,
            inputs=[points2d, mirror, out_faces, ring, count_wp],
            block_dim=kernel_polyline.EAR_BLOCK_DIM,
            device=device,
        )
        count = int(read_scalar(count_wp, 0))
        return n_ring, _clipped_faces(out_faces, count, n_ring), clockwise

    # The ear grid (see ``kernels/polyline.ear_grade_grid``): square cells, about
    # ``EAR_GRID_CELLS_PER_POINT`` per ring point over the ring's box, read back with the turning
    # angle. A degenerate box (a ring on a line) still gets cells along its long side.
    above_x, above_y, below_x, below_y = (float(e) for e in extent)
    width = max(above_x + below_x, 0.0)
    height = max(above_y + below_y, 0.0)
    target = kernel_polyline.EAR_GRID_CELLS_PER_POINT * n_ring
    cell = max(math.sqrt(width * height / target), max(width, height) / target)
    if not cell > 0.0 or not math.isfinite(cell):
        cell = 1.0
    dims = (
        min(max(1, math.ceil(width / cell)), target),
        min(max(1, math.ceil(height / cell)), target),
    )
    n_cells = dims[0] * dims[1]
    shift = wp.vec2(below_x, below_y)
    inv_cell = wp.float32(1.0 / cell)
    dims_wp = wp.vec2i(wp.int32(dims[0]), wp.int32(dims[1]))
    cell_keys = _launch.empty(2 * n_ring, dtype=wp.int32, device=device)
    cell_points = _launch.empty(2 * n_ring, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.ear_grid_cells,
        dim=n_ring,
        inputs=[points2d, shift, inv_cell, dims_wp],
        outputs=[cell_keys, cell_points],
        device=device,
    )
    _launch.radix_sort_pairs(
        cell_keys, cell_points, count=n_ring, end_bit=max(1, (n_cells - 1).bit_length())
    )
    cell_starts = _launch.empty(n_cells + 1, dtype=wp.int32, device=device)
    left = _launch.empty(n_ring, dtype=wp.int32, device=device)
    right = _launch.empty(n_ring, dtype=wp.int32, device=device)
    active = _launch.empty(n_ring, dtype=wp.int32, device=device)
    is_ear = _launch.empty(n_ring, dtype=wp.int32, device=device)
    selected = _launch.empty(n_ring, dtype=wp.int32, device=device)
    # [rounds run, loop condition, face count, some ear good], seeded by ``init_ear_clip`` with
    # the grid's cell starts and the ring's links.
    state = _launch.empty(kernel_polyline.EAR_STATE_SIZE, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.init_ear_clip,
        dim=n_ring,
        inputs=[cell_keys, wp.int32(n_ring), left, right, active],
        outputs=[cell_starts, state],
        device=device,
    )

    def clip_round() -> None:
        _launch.launch(
            kernel_polyline.compute_ears,
            dim=n_ring,
            inputs=[
                points2d,
                left,
                right,
                active,
                mirror,
                cell_starts,
                cell_points,
                shift,
                inv_cell,
                dims_wp,
                state,
                is_ear,
            ],
            device=device,
        )
        _launch.launch(
            kernel_polyline.select_independent,
            dim=n_ring,
            inputs=[is_ear, left, right, state, selected],
            device=device,
        )
        _launch.launch(
            kernel_polyline.clip_selected,
            dim=n_ring,
            inputs=[selected, left, right, active, wp.int32(n_ring - 2), wp.int32(n_ring)],
            outputs=[out_faces, state],
            device=device,
        )

    condition = odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW])
    run_device_loop(device, condition, clip_round)

    count = int(read_scalar(state, int(kernel_polyline.EAR_COUNT)))
    return n_ring, _clipped_faces(out_faces, count, n_ring), clockwise


def _clipped_faces(faces: odt.Array2dInt32, count: int, n_tips: int) -> odt.Array2dInt32:
    """
    Return the first ``count`` rows of an ear clip's faces, by ascending tip when deterministic.

    The clip appends its faces through an atomic counter, so on CUDA their rows come in arrival
    order; with [`is_deterministic`][ordito.config.is_deterministic] they are sorted by their middle
    index (the ear's tip, below ``n_tips`` and distinct per face), the order a fan already has.
    """
    clipped = odt.as_array2d(odt.as_dense(faces[0:count]), wp.int32)
    if not is_deterministic() or count < 2:
        return clipped
    return _faces_in_tip_order(clipped, n_tips)


def _faces_in_tip_order(faces: odt.Array2dInt32, n_tips: int) -> odt.Array2dInt32:
    """
    Return ``faces`` with its rows sorted by their middle index, distinct per row below ``n_tips``.

    One radix sort of the tips over the bits ``n_tips`` needs, then a gather of the rows.
    """
    device = faces.device
    n_faces = faces.shape[0]
    keys = _launch.empty(2 * n_faces, dtype=wp.int32, device=device)
    rows = _launch.empty(2 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_polyline.face_tip_keys,
        dim=n_faces,
        inputs=[faces],
        outputs=[keys, rows],
        device=device,
    )
    _launch.radix_sort_pairs(keys, rows, count=n_faces, end_bit=max(1, (n_tips - 1).bit_length()))
    ordered = odt.empty_2d((n_faces, 3), wp.int32, device=device)
    _launch.copy(ordered, faces[odt.as_dense(rows[:n_faces])])
    return odt.as_array2d(ordered, wp.int32)
