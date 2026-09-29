"""
Open and closed 3D polyline operations.

**The ``polyline_`` prefix.** Where a name carries the module's own token, the token comes
**first** -- ``polyline_length``, ``polyline_open``, ``polyline_resample``,
``polyline_triangulate`` -- so the whole family sorts and completes together.

A function that does *not* need the token does not gain one:
[`is_closed`][triwarp.polyline.is_closed] and
[`cumulative_arc_length`][triwarp.polyline.cumulative_arc_length] have an unambiguous subject
already, and [`triangulate_polygon`][triwarp.polyline.triangulate_polygon] takes a 2D polygon
rather than a polyline, so the token there would be wrong rather than redundant.
[`polyline_point_distance`][triwarp.polyline.polyline_point_distance] keeps both nouns because the
prefix alone would lose "from what"; it is spelled to match
[`points.point_plane_distance`][triwarp.points.point_plane_distance].

**The ``closed=`` convention.** Ten of these functions -- length, centroid, radius, angles,
point-to-curve distance, and the five resampling operations -- take a keyword-only
``closed: bool = False``. Passing ``closed=True`` appends the closing edge back to the first point
if it is absent ([`polyline_close`][triwarp.polyline.polyline_close]) and then does the open
computation on that, so the seam is treated like any other segment. Three of the ten additionally
drop the duplicated seam point on the way out, so their result is a clean cyclic ring with one entry
per *original* point rather than one extra:
[`polyline_resample`][triwarp.polyline.polyline_resample],
[`polyline_angles`][triwarp.polyline.polyline_angles] and
[`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample].

The keyword is not always needed. [`polyline_angles`][triwarp.polyline.polyline_angles] and
[`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample] already detect an
*explicitly* closed input -- one whose last point equals its first -- with
[`is_closed`][triwarp.polyline.is_closed]; ``closed=True`` is for the common case of a loop stored
without that duplicate, which is the form
[`boundary_loops`][triwarp.boundary.boundary_loops] returns.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, read_values, require_same_device, run_device_loop
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import polyline as kernel_polyline
from triwarp.kernels import reduce as kernel_reduce

# Point count from which [`polyline_downsample`][triwarp.polyline.polyline_downsample] stops
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
    [`allclose`][triwarp.array.allclose] does, through the same kernel-side function.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    bool
        ``True`` when the polyline has at least two points and its first and last points are
        equal within the default ``allclose`` tolerance; ``False`` otherwise.

    See Also
    --------
    [`polyline_open`][triwarp.polyline.polyline_open]
    [`polyline_close`][triwarp.polyline.polyline_close]
    """
    n = int(polyline.shape[0])
    if n < 2:
        return False
    return bool(int(read_scalar(_endpoints_coincide_flag(polyline), 0)) != 0)


def _endpoints_coincide_flag(polyline: wp.array[wp.vec3]) -> wp.array[wp.int32]:
    """
    One-element device flag, ``1`` when the first and last points coincide.

    [`is_closed`][triwarp.polyline.is_closed]'s device half. The kernels that take the closure
    decision on the device instead evaluate the same predicate themselves, so none of them pays
    this launch. One launch rather than ``allclose`` over two one-element slices: that spelling is
    a ``wp.map`` into a mask plus a whole reduction over it, to compare six floats. Needs at least
    two points.
    """
    flag = wp.empty(1, dtype=wp.int32, device=polyline.device)
    wp.launch(
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
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.vec3]
        The input unchanged when it has fewer than two points or is already open;
        otherwise a length ``n - 1`` view without the duplicated closing point.

    See Also
    --------
    [`polyline_close`][triwarp.polyline.polyline_close]
    """
    n = int(polyline.shape[0])
    if not is_closed(polyline):
        return polyline
    return twt.as_dense(polyline[0 : n - 1])


def polyline_close(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Close a polyline by appending the first point when it is not already the last.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.vec3]
        The input unchanged when it has fewer than two points or is already closed;
        otherwise a length ``n + 1`` array with the first point appended.

    See Also
    --------
    [`polyline_open`][triwarp.polyline.polyline_open]
    """
    n = int(polyline.shape[0])
    if n < 2 or is_closed(polyline):
        return polyline
    return _append_first_point(polyline)


def _append_first_point(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Return the polyline with its first point appended: two copies into one allocation.

    What [`array.concatenate`][triwarp.array.concatenate] of the polyline and its first point
    returns, without the segment table that function builds for an arbitrary list.
    """
    n = int(polyline.shape[0])
    closed = wp.empty(n + 1, dtype=wp.vec3, device=polyline.device)
    wp.copy(closed, polyline, count=n)
    wp.copy(closed, polyline, dest_offset=n, count=1)
    return closed


def polyline_length(polyline: wp.array[wp.vec3], *, closed: bool = False) -> float:
    """
    Total arc length of a polyline (sum of segment lengths).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        edge counts toward the total.

    Returns
    -------
    float
        The summed segment length, ``0.0`` for fewer than two points.

    See Also
    --------
    [`cumulative_arc_length`][triwarp.polyline.cumulative_arc_length]
        The same lengths, unreduced.
    """
    device = polyline.device
    n_points = int(polyline.shape[0])
    # ``closed`` adds the segment from the last point back to the first, which the kernel reaches
    # by wrapping its index -- no ``polyline_close`` copy of the whole buffer for one segment.
    n_segments = n_points if closed else n_points - 1
    if n_segments < 1:
        return 0.0
    # Summed where the segment lengths are computed, rather than through an ``(n - 1,)`` scratch
    # buffer and a separate reduction over it -- one launch, one allocation and one readback
    # instead of two of each. See ``kernels/polyline.polyline_total_length``, including why the
    # answer's last bits move.
    total = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch_tiled(
        kernel_polyline.polyline_total_length,
        dim=kernel_reduce.blocks_1d(n_segments),
        inputs=[polyline, wp.int32(n_segments)],
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
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
    [`polyline_normal`][triwarp.polyline.polyline_normal]
    [`polyline_radius`][triwarp.polyline.polyline_radius]
        Both default their plane to this centroid.
    """
    n_points = int(polyline.shape[0])
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
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.

    Returns
    -------
    wp.vec3
        The unit normal on the host.

    Raises
    ------
    ValueError
        If the polyline has fewer than three points.
    """
    n = int(polyline.shape[0])
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
        ``(n,)`` query points as ``wp.vec3``.
    polyline
        ``(m,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        edge is a candidate segment like any other.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` minimum distances on ``points.device``. Every entry is ``inf`` when
        ``polyline`` is empty, there being no segment to measure against -- the same
        ``inf``-on-miss convention
        [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] uses.

    Raises
    ------
    RuntimeError
        If ``points`` and ``polyline`` are not all on one device.
    """
    require_same_device(points=points, polyline=polyline)
    device = points.device
    n_points = int(points.shape[0])
    m = int(polyline.shape[0])
    if m == 0:
        return wp.full(n_points, float("inf"), dtype=wp.float32, device=device)
    if n_points == 0:
        return wp.empty(0, dtype=wp.float32, device=device)
    if m == 1:
        out_distances = wp.empty(n_points, dtype=wp.float32, device=device)
        wp.launch(
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
        out_distances = wp.empty(n_points, dtype=wp.float32, device=device)
        wp.launch(
            kernel_polyline.distance_to_segments,
            dim=n_points,
            inputs=[points, polyline, wp.int32(closed), out_distances],
            device=device,
        )
        return out_distances
    out_distances = wp.full(n_points, math.inf, dtype=wp.float32, device=device)
    wp.launch(
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Target spacing between consecutive output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), and the
        duplicated closing point is not emitted, so the result is a cyclic ring.

    Returns
    -------
    wp.array[wp.vec3]
        The upsampled polyline. The input is returned unchanged for fewer than two points.

    See Also
    --------
    [`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample]
        The same subdivision with the new points placed on a fitted arc instead of the chord.
    [`polyline_downsample`][triwarp.polyline.polyline_downsample]
    [`polyline_resample`][triwarp.polyline.polyline_resample]
    """
    return _upsample(polyline, step_size, closed, kernel_polyline.upsample_gather, [])


def polyline_smooth_upsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Upsample a polyline to an approximately uniform step size, following local curvature.

    Like [`polyline_upsample`][triwarp.polyline.polyline_upsample], each segment is split into
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Target spacing between consecutive output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). Every segment
        including the seam is then treated as interior, so neighbour tangents wrap cyclically and
        the whole loop is smoothed rather than only its middle.

    Returns
    -------
    wp.array[wp.vec3]
        The curvature-aware upsampled polyline. The input is returned unchanged for fewer than
        two points.

    See Also
    --------
    [`polyline_upsample`][triwarp.polyline.polyline_upsample]
        The straight-chord version, which is what this reduces to on the end segments.
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
    gather_kernel: wp.Kernel,
    extra_inputs: list[wp.int32 | wp.float32],
) -> wp.array[wp.vec3]:
    """
    Split every segment into ``max(floor(length / step_size), 1)`` pieces and gather the samples.

    The shared body of [`polyline_upsample`][triwarp.polyline.polyline_upsample] and
    [`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample], which differ only in
    the gather kernel that places each sample -- on the chord or on a fitted arc -- and in the
    extra arguments that kernel takes. ``closed`` counts ``n`` segments rather than ``n - 1``: the
    last is the closing one, reached by wrapping the index, or a segment with no samples when the
    input already repeats its first point (``segment_step_counts``) -- so the host never asks.
    """
    device = polyline.device
    n_points = int(polyline.shape[0])
    if n_points < 2:
        return polyline
    n_segments = n_points if closed else n_points - 1

    steps = wp.empty(n_segments, dtype=wp.int32, device=device)
    wp.launch(
        kernel_polyline.segment_step_counts,
        dim=n_segments,
        inputs=[polyline, wp.float32(step_size), wp.int32(closed), steps],
        device=device,
    )
    offsets, total = tw.array.counts_to_offsets(steps)

    out_points = wp.empty(total, dtype=wp.vec3, device=device)
    wp.launch(
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
    [`polyline_downsample`][triwarp.polyline.polyline_downsample] and
    [`polyline_resample`][triwarp.polyline.polyline_resample]; exposed directly for callers
    doing custom resampling along the polyline.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` array on ``polyline.device``. Entry ``0`` is ``0.0``; entry ``i`` is the
        summed length of segments ``0..i-1``.

    See Also
    --------
    [`polyline_length`][triwarp.polyline.polyline_length]
    [`polyline_downsample`][triwarp.polyline.polyline_downsample]
    [`polyline_resample`][triwarp.polyline.polyline_resample]
    """
    n_points = int(polyline.shape[0])
    if n_points < 2:
        # No segment to sum: a single vertex is at arc length 0, an empty polyline has no entry.
        return wp.zeros(n_points, dtype=wp.float32, device=polyline.device)
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Minimum arc-length distance between kept points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        edge counts toward the spacing.

    Returns
    -------
    wp.array[wp.vec3]
        The downsampled polyline. The input is returned unchanged for fewer than two points.

    See Also
    --------
    [`polyline_simplify`][triwarp.polyline.polyline_simplify]
        Drops points by *shape* error rather than by spacing.
    [`polyline_upsample`][triwarp.polyline.polyline_upsample]
    """
    device = polyline.device
    if int(polyline.shape[0]) < 2:
        return polyline

    # ``closed`` walks an ``n + 1``-entry table whose last entry is the closing segment's end, or,
    # when the input already repeats its first point, a stand-in the walks stop short of
    # (``kernels/polyline.seam_repeats_first``) -- the closure is decided on the device.
    cumulative = _arc_length_table(polyline, closed=closed)
    n_table = int(cumulative.shape[0])
    keep = wp.zeros(n_table, dtype=wp.int32, device=device)
    if wp.get_device(device).is_cuda and n_table >= _DOWNSAMPLE_DOUBLING_FROM:
        _greedy_downsample_doubling(cumulative, step_size, polyline, closed, keep)
    else:
        wp.launch(
            kernel_polyline.greedy_downsample_mask,
            dim=1,
            inputs=[cumulative, wp.float32(step_size), polyline, wp.int32(closed), keep],
            device=device,
        )
    out_points, _ = _gather_kept(keep, polyline, return_indices=False)
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
    ([`greedy_successors`][triwarp.kernels.polyline.greedy_successors]) turns an ``n``-step walk
    into a logarithmic number of rounds of jumping several pointers along it. The answer is the
    serial walk's, exactly and not approximately: the successor search evaluates the same float32
    comparison the walk does, so the two masks agree bit for bit -- verified over 27 shapes
    including exact ties and heavily clustered spacing. ``polyline`` and ``closed`` are the closure
    the walk stops short of (``greedy_successors``); an open table needs no polyline.
    """
    device = cumulative.device
    n = int(cumulative.shape[0])
    successor = wp.empty(n, dtype=wp.int32, device=device)
    # Also marks the first point kept, which the walk always does; the caller zeroed the rest.
    wp.launch(
        kernel_polyline.greedy_successors,
        dim=n,
        inputs=[cumulative, wp.float32(step_size), polyline, wp.int32(closed), successor, out_keep],
        device=device,
    )
    squared = wp.empty(n, dtype=wp.int32, device=device)
    # Several pointers a round (``kernels/graph.pointer_jump_schedule``): the orbit is at most
    # ``n`` long and a round multiplies the covered prefix of it by ``hops``.
    hops, rounds = kernel_graph.pointer_jump_schedule(n + 1)
    for _ in range(rounds):
        wp.launch(
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    tol
        Maximum Euclidean distance allowed between a dropped vertex and the retained chord.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). ``indices``
        then refer to the *closed* polyline, so the shared start/end vertex is preserved at both
        ends.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(simplified, indices)`` on ``polyline.device``: the ``(m,)`` retained vertices and the
        ``(m,)`` sorted indices into the input such that ``polyline[indices] == simplified``. An
        empty input yields two empty arrays; a single point is returned unchanged with
        ``indices == [0]``.

    Notes
    -----
    The round loop runs **on device** with no readback. Up to a few thousand points on CUDA, and at
    every size on the CPU device, it runs as a single block whose lanes share the points, with a
    block barrier between the steps of a round; a longer polyline on CUDA drives the four launches
    with ``wp.capture_while`` exactly as
    [`polyline_triangulate`][triwarp.polyline.polyline_triangulate]'s ear rounds are driven. Both
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
    [`polyline_downsample`][triwarp.polyline.polyline_downsample]
        Drops points by *spacing* rather than by shape error.
    """
    device = polyline.device
    n = int(polyline.shape[0])
    if n == 0:
        return polyline, wp.empty(0, dtype=wp.int32, device=device)
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
        # conditional graph it records on every call.
        spans = twt.empty_2d((4, n_entries), wp.int32, device=device)
        span_values = twt.empty_2d((2, n_entries), wp.float32, device=device)
        wp.launch_tiled(
            kernel_polyline.rdp_simplify_block,
            dim=[1],
            inputs=[polyline, wp.int32(wrap_open), squared_tolerance],
            outputs=[spans, span_values],
            block_dim=kernel_polyline.RDP_BLOCK_DIM,
            device=device,
        )
        simplified, indices = _gather_kept(twt.as_dense(spans[3]), polyline, return_indices=True)
        assert indices is not None
        return simplified, indices
    span_lo = wp.empty(n_entries, dtype=wp.int32, device=device)
    span_hi = wp.empty(n_entries, dtype=wp.int32, device=device)
    keep = wp.empty(n_entries, dtype=wp.int32, device=device)
    # state = [levels run, loop condition], seeded by ``rdp_seed_spans`` and then written on device
    # so the round loop needs no readback -- ``polyline_triangulate``'s ear rounds are driven the
    # same way.
    state = wp.empty(kernel_array.LOOP_STATE_SIZE, dtype=wp.int32, device=device)
    wp.launch(
        kernel_polyline.rdp_seed_spans,
        dim=n_entries,
        inputs=[polyline, wp.int32(wrap_open), span_lo, span_hi, keep, state],
        device=device,
    )
    if n_entries > 2:  # fewer than three entries have no interior to drop
        span_max = wp.empty(n_entries, dtype=wp.float32, device=device)
        span_argmax = wp.empty(n_entries, dtype=wp.int32, device=device)
        squared_distances = wp.empty(n_entries, dtype=wp.float32, device=device)

        def split_round() -> None:
            wp.launch(
                kernel_polyline.rdp_begin_round,
                dim=n_entries,
                inputs=[state, span_max, span_argmax],
                device=device,
            )
            wp.launch(
                kernel_polyline.rdp_span_max,
                dim=n_entries,
                inputs=[polyline, span_lo, span_hi, squared_distances, span_max],
                device=device,
            )
            wp.launch(
                kernel_polyline.rdp_span_argmax,
                dim=n_entries,
                inputs=[span_lo, squared_distances, span_max, span_argmax],
                device=device,
            )
            wp.launch(
                kernel_polyline.rdp_split_spans,
                dim=n_entries,
                inputs=[squared_tolerance, span_max, span_argmax, span_lo, span_hi, state, keep],
                device=device,
            )

        condition = state[kernel_array.LOOP_CONDITION_VIEW]
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
    wp.utils.array_scan(flags, out_array=flags, inclusive=True)
    n_kept = int(read_scalar(flags))
    out_points = wp.empty(n_kept, dtype=wp.vec3, device=device)
    out_indices = wp.empty(n_kept, dtype=wp.int32, device=device) if return_indices else None
    wp.launch(
        kernel_polyline.gather_kept_points,
        dim=int(flags.shape[0]),
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    num_points
        Number of output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). The result has
        ``num_points`` *distinct* points: the duplicated seam is dropped.

    Returns
    -------
    wp.array[wp.vec3]
        Length ``num_points`` resampled polyline. An empty input is returned unchanged; a single
        point is repeated ``num_points`` times.

    See Also
    --------
    [`polyline_upsample`][triwarp.polyline.polyline_upsample]
        Targets a step size instead of a point count.
    """
    device = polyline.device
    n = int(polyline.shape[0])
    if n == 0:
        return polyline
    out_points = wp.empty(num_points, dtype=wp.vec3, device=device)
    if n == 1:
        wp.launch(
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
    wp.launch(
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

    [`cumulative_arc_length`][triwarp.polyline.cumulative_arc_length]'s body, and with ``closed``
    the table of the loop without a ``polyline_close`` copy: entry ``n`` is the closing segment's
    end, or -- when the input already repeats its first point -- a zero-length stand-in its
    consumers skip (``kernels/polyline.arc_segment_lengths``). The kernel writes the leading zero
    and each segment's length one slot along, and the inclusive scan of everything after the zero
    runs in place, the one-allocation idiom of ``array.counts_to_offsets`` in float32.
    """
    device = polyline.device
    n_points = int(polyline.shape[0])
    n_segments = n_points if closed else n_points - 1
    cumulative = wp.empty(n_segments + 1, dtype=wp.float32, device=device)
    wp.launch(
        kernel_polyline.arc_segment_lengths,
        dim=n_segments,
        inputs=[polyline, wp.int32(closed), cumulative],
        device=device,
    )
    lengths = cumulative[1:]
    wp.utils.array_scan(lengths, out_array=lengths, inclusive=True)
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    reduction
        Reduction over the per-segment radial distances: ``"min"``, ``"max"``, ``"mean"``, or
        ``"median"``. Defaults to ``"min"``.
    center
        Plane origin. Defaults to [`polyline_centroid`][triwarp.polyline.polyline_centroid].
    normal
        Plane normal (need not be unit). Defaults to
        [`polyline_normal`][triwarp.polyline.polyline_normal].

        These two are the circle's frame, not a cutting plane, so they deliberately keep their own
        names and their own order rather than the ``(plane_normal, plane_origin)`` convention every
        plane argument in the package follows -- both are keyword-defaulted here, and calling them
        a plane would misdescribe the geometry.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
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
    [`polyline_centroid`][triwarp.polyline.polyline_centroid]
    [`polyline_normal`][triwarp.polyline.polyline_normal]
        The two defaults for the plane.
    [`median`][triwarp.reduce.median]
    """
    if reduction not in ("min", "max", "mean", "median"):
        # Before the closure test, which is device work and a readback: a rejected argument
        # should not cost a launch first.
        raise ValueError(f"unsupported reduction {reduction!r}")
    device = polyline.device
    n = int(polyline.shape[0])
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
    frame = None
    if center is None or normal is None:
        # The midpoint sums only for a defaulted centre, the Newell sum only for a defaulted normal.
        n_open = n - 1 if center is None else 0
        wrap_open = int(closed) if center is None else 0
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
        distances = twt.empty_1d(n_segments, wp.float32, device=device)
        wp.launch(
            kernel_polyline.radius_distances,
            dim=n_segments,
            inputs=plane,
            outputs=[distances],
            device=device,
        )
        return tw.reduce.median(distances)
    # ``min`` / ``max`` / ``mean`` are folded where the distances are computed
    # (``kernels/polyline.radius_reduce``): no distance buffer, no reduction launch over it, and one
    # readback of the value and, for the mean, the segment count the kernel decided.
    kind, identity = _RADIUS_REDUCTIONS[reduction]
    result = wp.full(kernel_polyline.RADIUS_RESULT_SIZE, identity, dtype=wp.float32, device=device)
    wp.launch_tiled(
        kernel_polyline.radius_reduce,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[*plane, wp.int32(closed), kind, result],
        block_dim=TILE_1D,
        device=device,
    )
    if reduction != "mean":
        return float(read_scalar(result, 0))
    total, count = read_values(result, 0, 2)
    return float(total) / float(count)


def _accumulate_frame(
    polyline: wp.array[wp.vec3], n_segments: int, wrap_open: int, *, with_normal: bool
) -> wp.array[wp.float32]:
    """
    Launch ``kernels/polyline.accumulate_radius_frame`` into a fresh ``RADIUS_FRAME_SIZE`` buffer.

    The shared frame pass of [`polyline_centroid`][triwarp.polyline.polyline_centroid],
    [`polyline_normal`][triwarp.polyline.polyline_normal] and
    [`polyline_radius`][triwarp.polyline.polyline_radius]: the length-weighted midpoint sums over
    ``n_segments`` segments plus the closing one ``wrap_open`` asks for, and, with
    ``with_normal``, Newell's sum over the loop.
    """
    device = polyline.device
    frame = wp.zeros(kernel_polyline.RADIUS_FRAME_SIZE, dtype=wp.float32, device=device)
    wp.launch_tiled(
        kernel_polyline.accumulate_radius_frame,
        dim=kernel_reduce.blocks_1d(int(polyline.shape[0])),
        inputs=[
            polyline,
            wp.int32(n_segments),
            wp.int32(wrap_open),
            wp.int32(1 if with_normal else 0),
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
        ``(n,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). The result
        still has one angle per *original* point, and the turning angles wrap around.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` angles in radians on ``polyline.device``. All zeros for fewer than two points.

    See Also
    --------
    [`is_closed`][triwarp.polyline.is_closed]
        What decides the wrap-around when ``closed`` is left ``False``.
    """
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 2:
        return wp.zeros(n, dtype=wp.float32, device=device)

    # One launch writes every angle in its final slot, deciding the closure on the device: no
    # readback decides the wrap-around, and ``closed=True`` reaches the closing segment by wrapping
    # the index rather than through a ``polyline_close`` copy. See
    # ``kernels/polyline.vertex_turning_angles``.
    angles = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_polyline.vertex_turning_angles,
        dim=n,
        inputs=[polyline, wp.int32(1 if closed else 0)],
        outputs=[angles],
        device=device,
    )
    return angles


def polyline_triangulate(polyline: wp.array[wp.vec3]) -> twt.Array2dInt32:
    """
    Triangulate the simple planar polygon bounded by a closed 3D polyline (ear clipping).

    The polyline is treated as the boundary of a simple polygon, which is filled with triangles
    whose vertices are the polyline vertices themselves -- no new (Steiner) points are introduced. A
    simple ``n``-gon yields ``n - 2`` non-overlapping triangles that cover the polygon. The loop is
    first projected onto its best-fit plane (via
    [`polyline_normal`][triwarp.polyline.polyline_normal]) so any planar loop works, not only ones
    lying in the ``xy`` plane.

    The implementation is a GPU-parallel port of ``ear_clipping.cpp`` from libigl: convex polygons
    use a single fan, while non-convex polygons clip a maximal independent set of ears per round
    until the polygon is exhausted. Returned faces are consistently wound counter-clockwise with
    respect to the loop's turning direction; the exact set of triangles may differ from a sequential
    ear clip, but every triangulation of a simple polygon has ``n - 2`` faces.

    The cost is set by the **round count**, and the round count by how many ears the
    independent-set rule can retire at once. Competing ears are ranked by a bijective hash of their
    ring index rather than by the index itself, which is what keeps that logarithmic: under the raw
    index an alternating star lets the ear at ``i - 2`` suppress the ear at ``i`` for every ``i``,
    so one ear is clipped per round and the loop runs its full ``n``-round cap.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. A duplicated closing point is dropped, with the
        predicate [`polyline_open`][triwarp.polyline.polyline_open] applies.

    Returns
    -------
    twt.Array2dInt32
        ``(m, 3)`` triangle vertex indices into ``polyline`` on ``polyline.device``. ``m`` is
        ``n - 2`` for a simple polygon; a degenerate or self-intersecting loop may yield fewer
        (a partial triangulation). Empty ``(0, 3)`` for fewer than three points.

    Notes
    -----
    The whole clip runs **on device**. A short ring is clipped by a single block that runs every
    round itself; a long one, whose rounds fill the device, by a round loop driven by
    ``wp.capture_while`` over a device-side condition. On the CPU device the single-block form is
    used at every length. Two readbacks are left in the whole function, both structural: one
    carrying the ring length (whether the last point repeats the first), the loop's orientation and
    its reflex count, which together decide every launch dimension and the convex fan fast path; and
    the face count, which sizes the returned slice. The plane frame is accumulated and consumed on
    the device and never crosses to the host.

    Every path produces the same triangulation up to row order, and that row order is not stable on
    CUDA -- faces are appended through an atomic counter.

    See Also
    --------
    [`polyline_normal`][triwarp.polyline.polyline_normal]
    [`polyline_close`][triwarp.polyline.polyline_close]
    """
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 3:
        return twt.empty_2d((0, 3), wp.int32, device=device)

    # The plane frame is built and consumed entirely on device: one accumulation pass, which also
    # decides whether the last point repeats the first, then a projection whose threads each derive
    # the frame from the accumulated sums.
    sums = wp.zeros(kernel_polyline.RING_SUMS_SIZE, dtype=wp.float32, device=device)
    wp.launch_tiled(
        kernel_polyline.accumulate_loop_frame,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[polyline, sums],
        block_dim=TILE_1D,
        device=device,
    )
    points2d = wp.empty(n, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_polyline.project_polyline_to_plane,
        dim=n,
        inputs=[polyline, sums, points2d],
        device=device,
    )
    _, faces = _triangulate_ring(points2d, sums, detect_closing=False)
    return faces


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
        The input ring with any repeated closing point removed; the vertices ``faces`` indexes.
    faces : wp.array[wp.int32]
        Length-``3 * (n - 2)`` flat triangle index buffer into ``ring``.

    Notes
    -----
    Differs from [`trimesh.creation.triangulate_polygon`][] in three ways, all of which follow from
    replacing the CPU polygon libraries (``mapbox_earcut`` / ``manifold3d`` / ``triangle``) with
    triwarp's own GPU ear clipper, [`polyline_triangulate`][triwarp.polyline.polyline_triangulate]:

    - The input is a ``wp.vec2`` ring, not a ``shapely.geometry.Polygon``, and **interior rings
      (holes) are not supported**.
    - No Steiner points are ever inserted, which is trimesh's ``force_vertices=True`` contract
      rather than its default.
    - There is no ``engine`` selection, and ``faces`` is flat rather than ``(m, 3)``, matching the
      face layout used throughout triwarp.

    The ring must be a simple (non self-intersecting) polygon. A degenerate ring yields a partial
    triangulation with fewer than ``n - 2`` triangles rather than raising.

    See Also
    --------
    [`polyline_triangulate`][triwarp.polyline.polyline_triangulate]
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`trimesh.creation.triangulate_polygon`][]
    """
    twt.ensure_ndim(polygon, 1, dtype=wp.vec2)
    device = polygon.device
    n = int(polygon.shape[0])
    if n < 3:
        return polygon, wp.empty(0, dtype=wp.int32, device=device)

    sums = wp.zeros(kernel_polyline.RING_SUMS_SIZE, dtype=wp.float32, device=device)
    n_ring, faces = _triangulate_ring(polygon, sums, detect_closing=True)
    # The ring is the input minus any repeated closing point, i.e. a prefix of it.
    return polygon[:n_ring].contiguous(), faces.reshape((-1,))


def _triangulate_ring(
    points2d: wp.array[wp.vec2], sums: wp.array[wp.float32], *, detect_closing: bool
) -> tuple[int, twt.Array2dInt32]:
    """
    Ear-clip a 2D ring, from the turning angle onwards: ``(ring length, (m, 3) faces)``.

    ``points2d`` holds the ring plus, possibly, a repeated closing point. With
    ``detect_closing=False`` the caller's prologue has already written the closing flag into
    ``sums``; otherwise the turning-angle pass decides it. Either way the one readback of ``sums``
    carries the ring length, the orientation and the reflex count together. ``points2d`` is only
    mutated -- mirrored in place for a clockwise ring -- when ``detect_closing=False``, i.e. when it
    is a buffer of the caller's own; a caller's input ring is cloned first.
    """
    device = points2d.device
    n = int(points2d.shape[0])
    wp.launch_tiled(
        kernel_polyline.accumulate_turning_angle,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[points2d, wp.int32(1 if detect_closing else 0), sums],
        block_dim=TILE_1D,
        device=device,
    )
    # The only readback before the convex fast path returns: ring length, turning angle and the
    # reflex count of the oriented loop, all decided on device.
    # The four slots are adjacent, ``RING_TURNING`` through ``RING_CLOSING``: one offset read.
    turning, reflex, reflex_mirrored, closing = read_values(
        sums, int(kernel_polyline.RING_TURNING), 4
    )
    n_ring = n - int(closing)
    if n_ring < 3:
        return n_ring, twt.empty_2d((0, 3), wp.int32, device=device)

    out_faces = twt.empty_2d((n_ring - 2, 3), wp.int32, device=device)
    # A clockwise ring is mirrored before the ear tests, and its reflex count is the mirror's.
    clockwise = turning < 0.0
    if (reflex_mirrored if clockwise else reflex) == 0.0:
        # The fan's faces are index triples, so a convex ring needs no orientation fix-up at all.
        wp.launch(
            kernel_polyline.fan_triangulate, dim=n_ring - 2, inputs=[out_faces], device=device
        )
        return n_ring, twt.as_array2d(out_faces, wp.int32)

    if clockwise:
        if detect_closing:
            points2d = wp.clone(points2d)
        wp.launch(kernel_polyline.orient_ccw, dim=n_ring, inputs=[points2d], device=device)

    # One block runs every round (see ``ear_clip_block``): no graph to record, one launch. On the
    # CPU device a launch grid is a serial loop either way, so the block form does the same walk
    # without a launch and a readback per round, and it is taken at every size.
    if n_ring <= kernel_polyline.EAR_ONE_BLOCK_MAX or not wp.get_device(device).is_cuda:
        ring = twt.empty_2d((5, n_ring), wp.int32, device=device)
        count_wp = wp.empty(1, dtype=wp.int32, device=device)
        wp.launch_tiled(
            kernel_polyline.ear_clip_block,
            dim=1,
            inputs=[points2d, out_faces, ring, count_wp],
            block_dim=kernel_polyline.EAR_BLOCK_DIM,
            device=device,
        )
        count = int(read_scalar(count_wp, 0))
        return n_ring, twt.as_array2d(out_faces[0:count], wp.int32)

    left = wp.empty(n_ring, dtype=wp.int32, device=device)
    right = wp.empty(n_ring, dtype=wp.int32, device=device)
    active = wp.empty(n_ring, dtype=wp.int32, device=device)
    is_ear = wp.empty(n_ring, dtype=wp.int32, device=device)
    selected = wp.empty(n_ring, dtype=wp.int32, device=device)
    # [rounds run, loop condition, face count], seeded by ``init_ring``.
    state = wp.empty(kernel_polyline.EAR_STATE_SIZE, dtype=wp.int32, device=device)
    wp.launch(
        kernel_polyline.init_ring, dim=n_ring, inputs=[left, right, active, state], device=device
    )

    def clip_round() -> None:
        wp.launch(
            kernel_polyline.compute_ears,
            dim=n_ring,
            inputs=[points2d, left, right, active, is_ear],
            device=device,
        )
        wp.launch(
            kernel_polyline.select_independent,
            dim=n_ring,
            inputs=[is_ear, left, right, selected],
            device=device,
        )
        wp.launch(
            kernel_polyline.clip_selected,
            dim=n_ring,
            inputs=[selected, left, right, active, out_faces, state],
            device=device,
        )
        wp.launch(
            kernel_polyline.ear_loop_continue,
            dim=1,
            inputs=[wp.int32(n_ring - 2), wp.int32(n_ring), state],
            device=device,
        )

    condition = state[kernel_array.LOOP_CONDITION_VIEW]
    run_device_loop(device, condition, clip_round)

    count = int(read_scalar(state, int(kernel_polyline.EAR_COUNT)))
    return n_ring, twt.as_array2d(out_faces[0:count], wp.int32)
