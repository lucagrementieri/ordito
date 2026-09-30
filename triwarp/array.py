"""NumPy-style structural and elementwise ops on Warp arrays (ranges, gather, sort, masks)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, require_same_device
from triwarp.constants import ALLCLOSE_ATOL, ALLCLOSE_RTOL, TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels import scatter as kernel_scatter

DType = TypeVar("DType")


# Use a direct-index membership table when the value *span* (max - min + 1, over both inputs) is at
# most this multiple of |test_elements|.
_ISIN_MASK_SIZE_FACTOR = 8

# Row width up to which [`sort_rows`][triwarp.array.sort_rows] uses a per-row insertion sort instead
# of a segmented radix sort. Comfortably above every in-library row width (edges 2, corners 3).
SORT_ROWS_INSERTION_MAX_COLS = 8


def arange(
    start: int, stop: int | None = None, step: int = 1, *, device: wp.DeviceLike
) -> wp.array[wp.int32]:
    """
    Evenly spaced integers over the half-open interval ``[start, stop)`` (``numpy.arange``).

    Called with a varying number of positional arguments, exactly as [`numpy.arange`][]:
    ``arange(stop, device=...)`` runs from ``0``, ``arange(start, stop, device=...)`` from
    ``start``, and ``arange(start, stop, step, device=...)`` spaces the values by ``step``. A
    ``step`` may be negative, in which case the interval runs downwards.

    Parameters
    ----------
    start
        Start of the interval, included. Read as ``stop`` when ``stop`` is omitted, in which case
        the interval starts at ``0``.
    stop
        End of the interval, excluded.
    step
        Spacing between consecutive values, so ``out[i + 1] - out[i] == step``. Must be non-zero.
    device
        Warp device for the result. Keyword-only, unlike the first three, so the positional
        arguments stay [`numpy.arange`][]'s; required, because nothing in this package allocates
        onto Warp's ambient current device.

    Returns
    -------
    wp.array[wp.int32]
        1-D array of ``max(0, ceil((stop - start) / step))`` values on ``device``.

    Raises
    ------
    ValueError
        If ``step`` is zero, or the first or last value does not fit in ``int32``.

    Notes
    -----
    There is no ``dtype`` argument, which is the one place this departs from [`numpy.arange`][]'s
    signature. Every index buffer in this package is ``int32``, so a second width has no caller,
    and registering the kernel overloads for widths nothing reaches would cost compile time on
    every rebuild. The range check above is what stands in for choosing a wider dtype: a range
    that does not fit raises rather than silently wrapping.

    See Also
    --------
    [`arange_repeat`][triwarp.array.arange_repeat]
        The same range with each value repeated a fixed number of times.
    [`numpy.arange`][]
    """
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise ValueError("step must be non-zero")
    # ``ceil((stop - start) / step)`` in integers, correct for either sign of ``step``: Python's
    # ``//`` floors, so negating both sides of the division ceils.
    n = max(0, -((start - stop) // step))
    if n > 0:
        _check_int32_fits(start, "start")
        _check_int32_fits(start + (n - 1) * step, "stop")
    out = _launch.empty(n, dtype=wp.int32, device=device)
    if n == 0:
        return out
    if start == 0 and step == 1:
        _launch.launch(kernel_array.ARANGE[wp.int32], dim=n, inputs=[out], device=device)
    else:
        _launch.launch(
            kernel_array.ARANGE_AFFINE[wp.int32],
            dim=n,
            inputs=[wp.int32(start), wp.int32(step), out],
            device=device,
        )
    return out


def arange_repeat(count: int, repeats: int, device: wp.DeviceLike) -> wp.array[wp.int32]:
    """
    Fill ``out[i] = i // repeats`` (``numpy.repeat`` of an index range).

    Parameters
    ----------
    count
        Number of elements; must be non-negative.
    repeats
        How many consecutive entries share an index; must be positive.
    device
        Warp device for the result.

    Returns
    -------
    wp.array[wp.int32]
        Length-``count`` array on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is negative, ``repeats`` is not positive, or the largest value does not fit
        in ``int32``.

    See Also
    --------
    [`arange`][triwarp.array.arange]
        The range this repeats, and whose ``Notes`` explain why neither takes a ``dtype``.
    [`numpy.repeat`][]
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if count > 0:
        _check_int32_fits((count - 1) // repeats, "count // repeats")
    out = _launch.empty(count, dtype=wp.int32, device=device)
    if count > 0:
        _launch.launch(
            kernel_array.ARANGE_REPEAT[wp.int32],
            dim=count,
            inputs=[wp.int32(repeats), out],
            device=device,
        )
    return out


def sort_pair_indices(n: int, fill_value: int, device: wp.DeviceLike) -> wp.array[wp.int32]:
    """
    Fill ``[0, 1, ..., n-1, fill_value, ..., fill_value]`` (length ``2 * n``).

    The payload buffer ``warp.utils.radix_sort_pairs`` wants: the first half seeded with the
    identity permutation, the second half (its scratch) filled with a padding value.

    Parameters
    ----------
    n
        Number of real entries; the result has length ``2 * n``.
    fill_value
        Padding written into the upper half.
    device
        Warp device for the result.

    Returns
    -------
    wp.array[wp.int32]
        Length-``2 * n`` array on ``device``.

    Raises
    ------
    ValueError
        If ``n`` is negative, or ``n - 1`` / ``fill_value`` does not fit in ``int32``.

    See Also
    --------
    [`sort_and_argsort`][triwarp.array.sort_and_argsort]
    [`arange`][triwarp.array.arange]
        Whose ``Notes`` explain why neither takes a ``dtype``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > 0:
        _check_int32_fits(n - 1, "n")
    _check_int32_fits(fill_value, "fill_value")
    out = _launch.empty(2 * n, dtype=wp.int32, device=device)
    if n > 0:
        _launch.launch(
            kernel_array.SORT_PAIR_INDICES[wp.int32],
            dim=2 * n,
            inputs=[wp.int32(n), wp.int32(fill_value), out],
            device=device,
        )
    return out


def pack_1d_arrays(
    arrays: Sequence[wp.array[wp.Scalar]], *, copy: bool = True
) -> tuple[wp.array[wp.Scalar], wp.array[wp.int32]]:
    """
    Concatenate several 1-D ``warp.array`` instances into one buffer plus per-segment offsets.

    Segment ``i`` occupies ``flat[offsets[i] : offsets[i + 1]]``. This is the usual packed
    representation for variable-length per-item lists on the device (no nested arrays).

    !!! note "The packed pair is spelled values first, then total-terminated offsets"
        This function is where the package's convention is stated, because it is the primitive the
        others are built on: **a packed buffer and its offsets are returned, and accepted, values
        first, and the offsets are the ``n + 1`` total-terminated CSR form** (``scipy``'s
        ``indptr``): item ``i`` is ``values[offsets[i] : offsets[i + 1]]``, ``offsets[0] == 0``,
        ``offsets[n] == values.shape[0]``, and a size is the difference of two neighbouring
        entries -- there is no separate sizes array. An empty packed result has ``offsets == [0]``.
        Every public function that hands back such a pair --
        [`boundary_loops_with_offsets`][triwarp.boundary.boundary_loops_with_offsets],
        [`successor_cycles`][triwarp.graph.successor_cycles],
        [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets],
        [`geodesic_ball`][triwarp.neighbors.geodesic_ball],
        [`vertex_face_adjacency`][triwarp.adjacency.vertex_face_adjacency],
        [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] and the
        [`triwarp.geodesic_walk`][triwarp.geodesic_walk] tracers among them -- and every function
        that *takes* one ([`split`][triwarp.array.split],
        [`trace_polylines`][triwarp.geodesic_walk.trace_polylines],
        [`submeshes_from_face_groups`][triwarp.selection.submeshes_from_face_groups]) uses the same
        order and form. Both halves are ``wp.int32`` in the common case, so a transposed unpack
        type-checks, runs, and indexes garbage; there is nothing but the convention to lean on.
        Where a third array rides along it is a *per-item* one and goes last, as in
        ``(ring_halfedges, offsets, is_boundary)``.

    Parameters
    ----------
    arrays
        Non-empty sequence of 1-D arrays sharing the same ``dtype`` and ``device``.
    copy
        Keep ``False`` for a read-only result. ``arrays`` that are already consecutive non-empty
        views of one buffer -- what [`split`][triwarp.array.split] returns with ``copy=False`` --
        are then handed back as that buffer's span instead of being copied into a new one, so the
        ``split`` round trip costs nothing at all. The default copies, so writing into ``flat`` is
        safe; with ``copy=False`` such a write reaches the segments.

    Returns
    -------
    flat
        1-D array of length ``sum(a.size for a in arrays)``, same ``dtype`` and ``device`` as
        the inputs.
    offsets
        Length ``len(arrays) + 1`` total-terminated offsets, ``dtype`` ``wp.int32``, same
        ``device`` as the inputs: ``offsets[0] == 0`` and ``offsets[-1] == flat.size``.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, any input is not rank-1, or their ``dtype`` differs.
    RuntimeError
        If the segments are not all on one device.

    See Also
    --------
    [`split`][triwarp.array.split]
        The inverse: recovers the per-segment arrays from ``(flat, offsets)``.
    [`concatenate`][triwarp.array.concatenate]
    """
    flat, offsets = _pack_segments(arrays, caller="pack_1d_arrays", copy=copy)
    return flat, _launch.array([*offsets, flat.size], dtype=wp.int32, device=flat.device)


def concatenate(arrays: Sequence[wp.array[DType]], *, copy: bool = True) -> wp.array[DType]:
    """
    Concatenate 1-D ``warp.array`` instances in order (``numpy.concatenate``).

    Parameters
    ----------
    arrays
        Non-empty sequence of rank-1 arrays sharing the same ``dtype`` and ``device``.
        Empty segments are allowed.
    copy
        Keep ``False`` for a read-only result, which then costs nothing when the segments already
        tile one buffer -- see [`pack_1d_arrays`][triwarp.array.pack_1d_arrays].

    Returns
    -------
    wp.array
        Contiguous 1-D array of length ``sum(a.size for a in arrays)`` on the input
        device. When ``arrays`` has a single element, that array is returned without
        copying whatever ``copy`` says, since there is nothing to concatenate it with.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, any input is not rank-1, or their ``dtype`` differs.
    RuntimeError
        If the segments are not all on one device.

    See Also
    --------
    [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]
    [`concatenate`][triwarp.combine.concatenate]
        The mesh-level operation of the same name, which joins ``(vertices, faces)`` pairs and
        reindexes as it goes. Both names are required: this one mirrors
        [`numpy.concatenate`][], that one [`trimesh.util.concatenate`][].
    [`numpy.concatenate`][]
    """
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty")
    if len(arrays) == 1:
        arr = arrays[0]
        if int(arr.ndim) != 1:
            raise ValueError(f"concatenate requires rank-1 arrays, got ndim={arr.ndim}")
        return arr
    return _pack_segments(arrays, caller="concatenate", copy=copy)[0]


def split(
    array: wp.array[DType],
    offsets: wp.array[wp.int32] | Sequence[int],
    *,
    copy: bool = False,
    row_size: int = 1,
) -> list[wp.array[DType]]:
    """
    Break a packed 1-D array into its per-segment arrays (``numpy.split``).

    The inverse of [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]: segment ``i`` is
    ``array[offsets[i] : offsets[i + 1]]``. One host readback of ``offsets`` (none when they are
    passed as a host sequence), then zero copies by default -- the segments are views into
    ``array``, which they keep alive.

    Parameters
    ----------
    array
        Rank-1 array to split, any ``dtype``.
    offsets
        Length-``n_segments + 1`` total-terminated ``wp.int32`` offsets: ``offsets[0] == 0``,
        non-decreasing, and ``offsets[-1]`` equal to ``array``'s row count -- exactly what
        [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] and
        [`counts_to_offsets`][triwarp.array.counts_to_offsets] return. ``[0]`` describes no
        segments. A host sequence of ints with the same contents is accepted too, for a caller
        that already holds the bounds.
    copy
        When ``True``, return independent copies, each in its own allocation, instead of views.
    row_size
        Number of elements each offset counts. With ``row_size=3`` the offsets index the rows of a
        flat ``(3 * n_faces,)`` face buffer, and segment ``i`` is
        ``array[3 * offsets[i] : 3 * offsets[i + 1]]`` -- the form
        [`split_with_offsets`][triwarp.combine.split_with_offsets] returns its face offsets in.

    Returns
    -------
    list[wp.array]
        One array per segment, on ``array.device``, in segment order. Empty list when
        ``offsets == [0]``.

    Raises
    ------
    ValueError
        If ``array`` or ``offsets`` is not rank-1, ``row_size`` is not positive or does not divide
        ``array``'s length, ``offsets`` is empty, or ``offsets`` is not a non-decreasing sequence
        from ``0`` to ``array``'s row count -- which is what an unterminated, length-``n_segments``
        offsets array fails.
    RuntimeError
        If ``array`` and ``offsets`` are not all on one device.

    See Also
    --------
    [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]
        The inverse: packs per-segment arrays into one buffer plus these offsets.
    [`split`][triwarp.combine.split]
        The mesh-level operation of the same name, which separates a mesh into connected
        components. Both names are required: this one mirrors [`numpy.split`][], that one
        ``trimesh.Trimesh.split``.
    [`numpy.split`][]
    """
    require_same_device(array=array, offsets=offsets)
    if int(array.ndim) != 1:
        raise ValueError(f"split requires a rank-1 array, got ndim={array.ndim}")
    on_device = isinstance(offsets, wp.array)
    offsets_ndim = int(offsets.ndim) if on_device else np.ndim(offsets)
    if offsets_ndim != 1:
        raise ValueError(f"split requires rank-1 offsets, got ndim={offsets_ndim}")
    n = array.size
    if row_size < 1 or n % row_size != 0:
        raise ValueError(
            f"row_size must be positive and divide the array length {n}, got {row_size}"
        )
    n_offsets = offsets.size if on_device else len(offsets)
    if n_offsets == 0:
        raise ValueError("split requires total-terminated offsets of length n_segments + 1, got []")
    if n_offsets == 1 and n == 0:
        # No segments over an empty array: the only valid entry is the ``0`` both ends must be.
        return []

    # The one readback, unless the caller already holds the bounds on the host. ``.numpy()`` of a
    # CPU array is a view, so the bounds are built in a fresh buffer rather than in place.
    bounds_np = np.array(offsets.numpy() if on_device else offsets, dtype=np.int64)
    n_rows = n // row_size
    if bounds_np[0] != 0 or bounds_np[-1] != n_rows or bool((bounds_np[1:] < bounds_np[:-1]).any()):
        raise ValueError(
            f"offsets must be total-terminated: start at 0, be non-decreasing and end at the row "
            f"count {n_rows}, got {bounds_np.tolist()}"
        )
    if bounds_np.size == 1:
        return []
    if row_size != 1:
        bounds_np *= row_size
    bounds = bounds_np.tolist()
    return _copy_segments(array, bounds) if copy else _segment_views(array, bounds)


def _segment_views(array: wp.array[DType], bounds: Sequence[int]) -> list[wp.array[DType]]:
    """
    Return the views ``array[a:b]`` for every consecutive pair of ``bounds``.

    A Warp slice is built in interpreted Python (``__getitem__`` into ``__init__``), and for a
    contiguous rank-1 base its whole state is a plain ``__dict__`` of which only ``ptr``, ``shape``,
    ``size`` and ``capacity`` depend on the bounds. So one real slice serves as a template and every
    segment is ``object.__new__`` plus a copy of its dictionary with those four patched: the same
    object a slice would be, ``_ref`` included (it is what keeps ``array`` alive), for a fraction of
    the construction cost. ``test_segment_views_match_real_slices`` compares the two dictionaries
    key by key, so a Warp release that adds or renames per-view state fails there rather than here.

    Anything the template cannot describe takes real slices: a ``requires_grad`` base (a slice
    also carries a ``grad`` view of its own), a non-contiguous or empty base, or a subclass. A
    zero-length segment at the very end is stamped too, although Warp rejects the slice
    ``array[n:n]``: its state is exactly that of an interior empty slice, one past the last element.
    """
    n = array.size
    if type(array) is not wp.array or array.requires_grad or not array.is_contiguous or n == 0:
        return [
            twt.as_dense(array[begin:end])
            if end > begin
            else _launch.empty(0, dtype=array.dtype, device=array.device)
            for begin, end in itertools.pairwise(bounds)
        ]
    template = array[0:1].__dict__
    stride = int(array.strides[0])
    base_ptr = int(array.ptr)
    new = object.__new__
    views: list[wp.array[DType]] = []
    for begin, end in itertools.pairwise(bounds):
        size = end - begin
        state = template.copy()
        state["ptr"] = base_ptr + begin * stride
        state["shape"] = (size,)
        state["size"] = size
        state["capacity"] = size * stride
        view = new(wp.array)
        view.__dict__ = state
        views.append(view)
    return views


def _copy_segments(array: wp.array[DType], bounds: Sequence[int]) -> list[wp.array[DType]]:
    """
    Return an independent copy of every segment ``array[a:b]`` of consecutive ``bounds``.

    Each segment gets its own allocation rather than a view of one shared buffer, because a shared
    allocation would mean holding one segment pins the whole buffer alive -- the opposite of what
    ``split(copy=True)`` promises. The allocations cannot be batched, but the copies can: from
    ``PACK_SEGMENTS_KERNEL_FROM`` segments one launch over a descriptor table of the destinations
    fills them all (``kernels.array.unpack_segment_words``, the packing kernel run backwards), in
    place of one ``wp.clone`` -- an allocation and a copy -- per segment. Anything the word copy
    cannot address clones per segment: a ``requires_grad`` base (a clone carries its own
    gradient), a non-contiguous one, or a dtype whose itemsize is not a multiple of four.
    """
    n_segments = len(bounds) - 1
    words_per_element = _words_per_element(array.dtype)
    if (
        n_segments < PACK_SEGMENTS_KERNEL_FROM
        or words_per_element == 0
        or type(array) is not wp.array
        or array.requires_grad
        or not array.is_contiguous
    ):
        return [_launch.clone(segment) for segment in _segment_views(array, bounds)]
    device = array.device
    sizes = [end - begin for begin, end in itertools.pairwise(bounds)]
    segments = [
        _launch.empty(size, dtype=array.dtype, device=device, pinned=array.pinned) for size in sizes
    ]
    table = _word_segment_table(
        [segment.ptr or 0 for segment in segments], sizes, bounds[:-1], words_per_element, device
    )
    if table is not None:
        descriptor, width = table
        _launch.launch(
            kernel_array.unpack_segment_words,
            dim=(n_segments, width),
            inputs=[_as_words(array, words_per_element), wp.int32(width)],
            outputs=[descriptor],
            device=device,
        )
    return segments


def _pack_segments(
    arrays: Sequence[wp.array[DType]], *, caller: str, copy: bool = True
) -> tuple[wp.array[DType], list[int]]:
    """
    Validate rank-1 segments and copy them into one contiguous buffer.

    The shared body of [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] and
    [`concatenate`][triwarp.array.concatenate] -- its only two callers -- which differ only in
    whether the caller wants the segment offsets back as a device array.

    **The offsets come back as a host list, and only ``pack_1d_arrays`` converts.** They are
    computed on the host in the first place, from each segment's ``shape``, and ``concatenate``
    discards the offsets entirely, taking ``[0]`` of this return. Converting here would make one
    of the two callers pay for a buffer it never reads; the conversion therefore sits in the
    caller that wants it, which is also the only place the *convention* for the pair (values
    first) is stated.

    With ``copy=False`` the result may be one of the inputs' own storage rather than a fresh
    buffer -- see [`_tiled_span`][triwarp.array._tiled_span] for when, and for why the choice
    cannot be made here.
    """
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty")

    # The device check belongs here rather than in each public caller because *this* is where the
    # answer's device is chosen -- ``arrays[0]``'s, arbitrarily. Without it a mixed sequence does
    # not fail: ``wp.copy`` transfers across devices happily, so the result is a silently migrated
    # buffer on whichever segment happened to be first, which then decides the device of every
    # launch built on it downstream. Both callers name the parameter ``arrays``, so the labels in
    # the message (``arrays[3]``) point at the segment a caller can identify.
    require_same_device(arrays=arrays)
    dtype = arrays[0].dtype
    device = arrays[0].device
    sizes = []
    for i, arr in enumerate(arrays):
        if int(arr.ndim) != 1:
            raise ValueError(f"{caller} requires rank-1 arrays, got ndim={arr.ndim} at index {i}")
        if arr.dtype != dtype:
            raise ValueError(
                f"all arrays must have the same dtype, got {dtype} and {arr.dtype} at index {i}"
            )
        sizes.append(arr.size)

    offsets = list(itertools.accumulate(sizes[:-1], initial=0))
    total = offsets[-1] + sizes[-1]
    if not copy:
        already_packed = _tiled_span(arrays, sizes, total)
        if already_packed is not None:
            return already_packed, offsets

    flat = _launch.empty(total, dtype=dtype, device=device)
    if not _pack_in_one_launch(arrays, sizes, offsets, flat):
        # One ``wp.copy`` per segment: the fallback, and the right answer below the threshold above
        # (a copy is cheaper than a launch, so a handful of segments never earns the descriptor).
        # Graph capture cannot amortize this loop either, since the segment
        # pointers change on every call, so a recorded graph could never be replayed against new
        # segments.
        for arr, offset, n in zip(arrays, offsets, sizes, strict=True):
            if n > 0:
                _launch.copy(flat, arr, dest_offset=offset, count=n)
    return flat, offsets


# Segment count from which ``_pack_segments`` builds a descriptor table and copies in one launch
# instead of one ``wp.copy`` per segment. The two costs are a straight line against a flat one --
# the loop is a few microseconds of host time per segment whatever it holds, the single launch is
# flat in the segment count *and* in the total size -- so the whole choice is where they cross, and
# above it the launch wins by orders of magnitude. The lines cross around half this value; the
# threshold is set at the first swept point on the winning side rather than the interpolated one.
PACK_SEGMENTS_KERNEL_FROM = 32

# Widest per-segment launch dimension ``_pack_segments`` will ask for. The kernel strides by this,
# so it only bounds the *grid*, never the work: without a cap a 256-way split whose first piece
# holds nearly all of a 2.6 M-element buffer would launch 670 M threads of which 99.6 % exit at
# once.
_PACK_SEGMENTS_MAX_WIDTH = 4096


def _pack_in_one_launch(
    arrays: Sequence[wp.array[DType]],
    sizes: Sequence[int],
    offsets: Sequence[int],
    flat: wp.array[DType],
) -> bool:
    """
    Copy every segment into ``flat`` with a single launch, or return ``False`` if that cannot apply.

    Warp has no array-of-arrays type, but a ``@wp.struct`` may carry a ``wp.array`` field and a
    ``wp.array`` of *that* is a descriptor table a kernel can index — see
    ``kernels.array.pack_segment_words`` for the measurement and for why one kernel serves every
    dtype rather than a table of them. The descriptor is built as a single NumPy structured array
    through the struct's own ``numpy_dtype()`` and uploaded once, which matters: constructing the
    struct instances one at a time in Python is almost the whole cost of a per-object build, and
    would give most of the win straight back.

    Returns ``False`` — leaving ``flat`` untouched for the caller's copy loop — when there are too
    few segments to pay for the descriptor, or when the dtype's itemsize is not a multiple of four
    and so cannot be addressed as whole words.
    """
    n_segments = len(arrays)
    if n_segments < PACK_SEGMENTS_KERNEL_FROM:
        return False
    words_per_element = _words_per_element(flat.dtype)
    if words_per_element == 0:
        return False
    table = _word_segment_table(
        [arr.ptr or 0 for arr in arrays], sizes, offsets, words_per_element, flat.device
    )
    if table is None:
        return False
    descriptor, width = table
    _launch.launch(
        kernel_array.pack_segment_words,
        dim=(n_segments, width),
        inputs=[descriptor, wp.int32(width)],
        outputs=[_as_words(flat, words_per_element)],
        device=flat.device,
    )
    return True


def _words_per_element(dtype: type) -> int:
    """Return how many 4-byte words one element of ``dtype`` spans; ``0`` if not a whole number."""
    words, remainder = divmod(int(wp.types.type_size_in_bytes(dtype)), 4)
    return 0 if remainder else words


def _word_segment_table(
    pointers: Sequence[int],
    sizes: Sequence[int],
    offsets: Sequence[int],
    words_per_element: int,
    device: wp.DeviceLike,
) -> tuple[wp.array[Any], int] | None:
    """
    Build the ``WordSegment`` descriptor table of a segmented copy, and its launch width.

    Segment ``i`` is ``sizes[i]`` elements at address ``pointers[i]``, and sits at element
    ``offsets[i]`` of the packed buffer. The shared half of the two directions of the segmented
    copy, [`_pack_in_one_launch`][triwarp.array._pack_in_one_launch] and
    [`_copy_segments`][triwarp.array._copy_segments]. ``None`` when every segment is empty, since
    there is then nothing to launch.
    """
    word_counts = np.asarray(sizes, dtype=np.int64) * words_per_element
    longest = int(word_counts.max())
    if longest == 0:
        return None
    # ``Struct.numpy_dtype()`` is unannotated and builds a plain ``dict``, where numpy's
    # ``zeros`` wants the ``_DTypeDict`` TypedDict; ``np.dtype`` is the documented way to
    # turn that mapping into a real structured dtype.
    record_dtype = np.dtype(cast("npt.DTypeLike", kernel_array.WordSegment.numpy_dtype()))
    record_np = cast("npt.NDArray[np.void]", np.zeros(len(sizes), dtype=record_dtype))
    record_np["data"]["data"] = np.asarray(pointers, dtype=np.uint64)
    record_np["data"]["shape"][:, 0] = word_counts
    record_np["data"]["strides"][:, 0] = 4
    record_np["data"]["ndim"] = 1
    record_np["offset"] = np.asarray(offsets, dtype=np.int64) * words_per_element
    record_np["count"] = word_counts
    descriptor = _launch.array(record_np, dtype=kernel_array.WordSegment, device=device, copy=True)
    return descriptor, min(longest, _PACK_SEGMENTS_MAX_WIDTH)


def _as_words(arr: wp.array[DType], words_per_element: int) -> wp.array[wp.int32]:
    """
    Return a zero-copy ``wp.int32`` handle on a contiguous ``arr``, typed as the words it holds.

    The same reinterpretation ``wp.array.view`` performs, written out because ``view`` refuses a
    dtype of a different size and every dtype wider than four bytes needs exactly that.
    """
    return wp.array(
        ptr=arr.ptr, dtype=wp.int32, shape=arr.size * words_per_element, device=arr.device
    )


def _tiled_span(
    arrays: Sequence[wp.array[DType]], sizes: Sequence[int], total: int
) -> wp.array[DType] | None:
    """
    Return the span these segments already occupy, when they are consecutive views of one buffer.

    ``split`` and [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] are documented inverses, and the
    round trip is common: [`boundary_loops`][triwarp.boundary.boundary_loops] slices one packed
    buffer into per-loop views and every batched consumer of those loops packs them straight back.
    Copying there would rebuild a buffer that already exists, one ``wp.copy`` per segment to move
    data that never needs to move.

    It runs only for a caller that asked (``copy=False``), because a packer that *sometimes* aliases
    is a trap: ``combine.concatenate`` adds each piece's vertex offset into the packed face buffer
    **in place**, so a view handed to it rewrites the caller's own faces. Which callers write is not
    inferable from here, so the choice stays theirs.

    The gate is *identity* on the base rather than adjacency of the pointers. Two separately
    allocated buffers can land adjacent in Warp's memory pool by luck, and an adjacency test would
    then alias or copy depending on the allocator. Requiring a common base restricts the fast path
    to callers already holding aliases of one allocation -- exactly the ``split`` round trip.

    ``_ref`` is Warp's own back-reference from a slice to the array it keeps alive (Warp 1.17); it
    is read through ``getattr`` and every conclusion drawn from it is re-checked against the public
    ``ptr`` / ``shape`` / ``strides`` / ``dtype`` / ``device``, so a release that drops the
    attribute loses the fast path rather than the correctness. ``None`` when the segments are not
    one buffer's, which is the ordinary case.
    """
    if any(n <= 0 for n in sizes):
        return None  # a zero-length segment has no address to chain through
    base = _view_base(arrays[0])
    if int(base.ndim) != 1 or not base.is_contiguous or base.dtype != arrays[0].dtype:
        return None
    stride = int(base.strides[0])
    cursor = int(arrays[0].ptr)
    for arr, n in zip(arrays, sizes, strict=True):
        # A direct view of ``base`` -- every segment ``split`` returns -- answers in one lookup.
        if (
            (getattr(arr, "_ref", None) is not base and _view_base(arr) is not base)
            or not arr.is_contiguous
            or int(arr.strides[0]) != stride
            or int(arr.ptr) != cursor
        ):
            return None
        cursor += n * stride
    start, remainder = divmod(int(arrays[0].ptr) - int(base.ptr), stride)
    if remainder or start < 0 or start + total > base.size:
        return None
    return twt.as_dense(base[start : start + total])


def _view_base(arr: wp.array[DType]) -> wp.array[DType]:
    """Resolve a slice view to the allocation it reads, or return an owning array unchanged."""
    while (parent := getattr(arr, "_ref", None)) is not None:
        arr = parent
    return arr


def allclose(
    a: wp.array[wp.Float] | wp.array[wp.vec3],
    b: wp.array[wp.Float] | wp.array[wp.vec3],
    *,
    rtol: float = ALLCLOSE_RTOL,
    atol: float = ALLCLOSE_ATOL,
) -> bool:
    """
    Test whether two arrays are element-wise equal within a tolerance (``numpy.allclose``).

    Reduces ``|a - b| <= atol + rtol * |b|`` (element-wise, and component-wise for ``wp.vec3``)
    to a single Python ``bool`` on-device, without copying either array to the host. Matches the
    asymmetric ``numpy.allclose`` / ``torch.allclose`` tolerance convention.

    Parameters
    ----------
    a
        Length-``n`` array of any float dtype (``float16`` / ``float32`` / ``float64``) or of
        ``wp.vec3``, on the target device.
    b
        Array of the same length and dtype as ``a``.
    rtol
        Relative tolerance. Defaults to ``1e-05``. Converted to ``a``'s precision.
    atol
        Absolute tolerance. Defaults to ``1e-08``. Converted to ``a``'s precision, so a
        ``float16`` comparison cannot resolve a tolerance below its own epsilon.

    Returns
    -------
    bool
        ``True`` when every element (every component, for ``wp.vec3``) is within tolerance.
        ``True`` for empty inputs, following the ``numpy.allclose`` convention.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or dtypes.
    RuntimeError
        If ``a`` and ``b`` are not all on one device.
    """
    require_same_device(a=a, b=b)
    if a.dtype != b.dtype:
        raise ValueError(f"allclose requires matching dtypes, got {a.dtype} and {b.dtype}")
    n = a.size
    if n != b.size:
        raise ValueError(f"allclose requires equal lengths, got {n} and {b.size}")
    if n == 0:
        return True

    # The predicate folds into its own reduction: one launch and one four-byte accumulator, rather
    # than a ``wp.map`` into an ``(n,)`` mask and a whole ``reduce.all`` over it. See
    # ``kernels/reduce.ALLCLOSE_1D_TILED``, including why it is a table of concrete kernels.
    tolerance = wp.float32 if a.dtype == wp.vec3 else a.dtype
    flag = _launch.ones(1, dtype=wp.int32, device=a.device)
    _launch.launch_tiled(
        kernel_reduce.ALLCLOSE_1D_TILED[a.dtype],
        dim=kernel_reduce.blocks_1d(n),
        inputs=[a, b, tolerance(rtol), tolerance(atol)],
        outputs=[flag],
        block_dim=TILE_1D,
        device=a.device,
    )
    return bool(int(read_scalar(flag, 0)) != 0)


def sort_and_argsort(
    keys: wp.array[wp.Scalar], *, fill_value: int = -1
) -> tuple[wp.array[wp.Scalar], wp.array[wp.int32]]:
    """
    Ascending sort of ``keys`` together with the permutation that produced it.

    Both halves of ``numpy.sort`` and ``numpy.argsort`` at once: a radix sort produces the ordered
    keys as a side effect of computing the order, so returning only one of the two would throw work
    away. ``sorted_keys[i] == keys[order[i]]``.

    Wraps ``warp.utils.radix_sort_pairs``, which needs double-width scratch for both the keys and
    the payload; this allocates that scratch, seeds the payload with ``0..n-1`` and hands back
    length-``n`` views of the sorted prefixes.

    Parameters
    ----------
    keys
        Length-``n`` sort keys (any radix-sortable scalar dtype; see
        [`sortable_dtype`][triwarp.typing.sortable_dtype] for which those are).
    fill_value
        Padding written into the upper half of the payload buffer, where the sort's scratch lives.
        Only matters to callers that read past ``n``.

    Returns
    -------
    sorted_keys : wp.array
        Length-``n`` view of the ascending keys.
    order : wp.array[wp.int32]
        Length-``n`` view of the original index of each sorted key.

    Notes
    -----
    Both results are **views** into the scratch buffers, kept alive by the returned arrays. Clone
    them if they must outlive the caller's frame alongside another sort.

    The sort is **stable**: equal keys keep their input order, so ``order`` is ascending within
    each run of duplicate keys. ``warp.utils.radix_sort_pairs`` documents this ("the sort is
    stable and operates in linear time"), it is inherent to its LSD radix passes, and it is
    verified on both devices against ``numpy.argsort(kind="stable")``. Callers may rely on it --
    [`split_with_offsets`][triwarp.combine.split_with_offsets] does, to keep faces ascending within
    each component.

    See Also
    --------
    [`sort_rows`][triwarp.array.sort_rows]
    [`sort_pair_indices`][triwarp.array.sort_pair_indices]
    [`sortable_dtype`][triwarp.typing.sortable_dtype]
    """
    device = keys.device
    n = keys.size
    if n == 0:
        return keys, _launch.empty(0, dtype=wp.int32, device=device)
    keys_buffer = _launch.empty(2 * n, dtype=keys.dtype, device=device)
    _launch.copy(keys_buffer, keys, count=n)
    order_buffer = sort_pair_indices(n, fill_value, device)
    _launch.radix_sort_pairs(keys_buffer, order_buffer, count=n)
    return twt.as_dense(keys_buffer[:n]), twt.as_dense(order_buffer[:n])


def sort_rows(data: twt.Array2dInt32 | twt.Array2dFloat32) -> None:
    """
    Sort each row of a 2D array independently, in place, ascending.

    Each row is treated as its own radix-sort segment, so rows are reordered internally
    but their relative row order is unaffected.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` device array sorted in place, row by row.

    Raises
    ------
    TypeError
        If ``data`` is not ``int32`` or ``float32``.

    Notes
    -----
    Rows no wider than ``SORT_ROWS_INSERTION_MAX_COLS`` are sorted by a per-row insertion sort (one
    thread per row); wider rows fall back to a segmented radix sort. The narrow path is not a
    micro-optimization: ``segmented_sort_pairs`` pays a fixed cost per *segment*, which for rows
    this narrow dominates the comparison work by orders of magnitude.

    The dtype restriction comes from that fallback -- ``warp.utils.segmented_sort_pairs`` takes
    ``int32`` or ``float32`` keys and nothing else -- and it is checked up front rather than left
    to whichever path a row width happens to select, so the accepted dtypes do not depend on ``w``.
    """
    # The narrow path's margin, for whoever considers deleting it: a fixed per-segment cost in
    # ``segmented_sort_pairs`` dominates the comparison work for rows this narrow, where a plain
    # compare-and-swap needs none of it.
    #
    # The dtype guard is here rather than at the wide branch because without it the two paths
    # disagree: the insertion kernel is generic, so without it a ``float64`` table sorts silently
    # at ``w <= SORT_ROWS_INSERTION_MAX_COLS`` and raises ``RuntimeError: Unsupported data type:
    # float64`` from inside Warp one column later. Support that turns on the row width is worse
    # than no support.
    if data.dtype not in (wp.int32, wp.float32):
        raise TypeError(f"sort_rows requires an int32 or float32 array, got {data.dtype}")
    n = data.size
    n_rows, n_cols = int(data.shape[0]), int(data.shape[1])
    if n_rows == 0 or n_cols < 2:
        return
    if n_cols <= SORT_ROWS_INSERTION_MAX_COLS:
        _launch.launch(
            kernel_array.SORT_ROWS_INSERTION[data.dtype],
            dim=n_rows,
            inputs=[data],
            device=data.device,
        )
        return

    data_buffer = _launch.empty(n * 2, dtype=data.dtype, device=data.device)
    _launch.copy(data_buffer, data, count=n)
    indices_buffer = sort_pair_indices(n, -1, data.device)
    segment_start_indices = arange(0, (n // n_cols + 1) * n_cols, n_cols, device=data.device)
    wp.utils.segmented_sort_pairs(
        data_buffer, indices_buffer, n, segment_start_indices=segment_start_indices
    )
    _launch.copy(data, data_buffer, count=n)


def triplet_buffers(
    n_triplets: int, dtype: type, device: wp.DeviceLike
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], twt.ArrayNd]:
    """
    Uninitialized ``(rows, cols, values)`` COO buffers for one ``csr_from_triplets`` build.

    Parameters
    ----------
    n_triplets
        Length of each of the three buffers: the number of ``(row, col, value)`` entries the
        writing kernel will emit, counting duplicates, since
        [`csr_from_triplets`][triwarp.array.csr_from_triplets]
        sums entries that land on the same position.
    dtype
        Element type of the value buffer. A scalar (``wp.float32`` / ``wp.float64``) for a
        1x1-block matrix, or a matrix type (``wp.mat22d``) for a block matrix.
    device
        Warp device for all three buffers.

    Returns
    -------
    rows, cols, values
        Three length-``n_triplets`` arrays on ``device``. The index buffers are ``wp.int32``;
        ``values`` takes ``dtype``. All three are **uninitialized** -- the caller's kernel is
        expected to write every entry.

    Notes
    -----
    ``wp.empty`` rather than ``wp.zeros`` deliberately: a triplet writer fills all three buffers,
    so zeroing them first would be three wasted launches. A kernel that emits *fewer* than
    ``n_triplets`` entries must therefore write an explicit structural zero (typically a
    self-entry) or point the slot outside the matrix, where the build drops it; every operator
    build in this package keeps numerical zeros -- see the note on
    [`index_sparse`][triwarp.array.index_sparse], the one caller that prunes.

    See Also
    --------
    [`index_sparse`][triwarp.array.index_sparse]
    """
    rows = _launch.empty(n_triplets, dtype=wp.int32, device=device)
    cols = _launch.empty(n_triplets, dtype=wp.int32, device=device)
    values = _launch.empty(n_triplets, dtype=dtype, device=device)
    return rows, cols, values


# ``BsrMatrix[Any]``, not ``BsrMatrix[wp.float32]``: ``dtype`` is a runtime argument and the
# seven callers pass ``wp.float64`` and ``wp.mat22d`` as well as a variable, so a concrete
# parameter here is simply wrong for most of them. No overload can narrow a runtime ``type``.
def empty_square_bsr(n_rows: int, dtype: type, device: wp.DeviceLike) -> wps.BsrMatrix[Any]:
    """
    Zero-nnz ``(n_rows, n_rows)`` operator, the ``n_faces == 0`` return several assemblers share.

    A square BSR matrix with no faces to assemble from is still a valid, correctly-shaped
    operator -- just an empty one -- so [`laplacian.cotmatrix`][triwarp.laplacian.cotmatrix],
    [`laplacian.connection_laplacian`][triwarp.laplacian.connection_laplacian],
    [`laplacian.graph_laplacian`][triwarp.laplacian.graph_laplacian] and
    [`energies`][triwarp.energies]'s assembly wrappers all construct this exact matrix for their
    empty-mesh early return.

    Parameters
    ----------
    n_rows
        Row and column count of the square matrix.
    dtype
        Element type of the (empty) value buffer -- a scalar for a 1x1-block matrix, or a matrix
        type (e.g. ``wp.mat22d``) for a block matrix.
    device
        Warp device for the matrix.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_rows, n_rows)`` matrix with zero stored entries, on ``device``.

    See Also
    --------
    [`bsr_from_csr`][triwarp.array.bsr_from_csr]
    """
    return bsr_from_csr(
        n_rows,
        n_rows,
        _launch.zeros(n_rows + 1, dtype=wp.int32, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        _launch.empty(0, dtype=dtype, device=device),
        nnz=0,
    )


def bsr_from_csr(
    n_rows: int,
    n_cols: int,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    *,
    nnz: int | None = None,
) -> wps.BsrMatrix[Any]:
    """
    Wrap compressed-row arrays that already hold a matrix's storage as a ``BsrMatrix``, no copy.

    ``values``' dtype is the block type: a scalar for a 1x1-block (CSR) matrix, or a matrix type
    such as ``wp.mat22d``. Row ``i``'s entries are ``columns[offsets[i] : offsets[i + 1]]``, and
    every ``warp.sparse`` operation expects them sorted by column with no duplicates.

    Parameters
    ----------
    n_rows, n_cols
        Block row and column counts.
    offsets
        Length ``n_rows + 1``, total-terminated (``offsets[-1]`` is the stored entry count).
    columns, values
        One element per stored entry, at least ``offsets[-1]`` long. A longer buffer is a
        capacity, as ``warp.sparse`` itself allocates.
    nnz
        The stored entry count, when the host knows it. ``None`` records the buffers' length as an
        upper bound and starts the asynchronous copy of ``offsets[-1]`` that
        ``BsrMatrix.nnz_sync()`` then completes -- the state ``warp.sparse.bsr_from_triplets``
        leaves, with no host synchronization here.

    Returns
    -------
    warp.sparse.BsrMatrix
        An ``(n_rows, n_cols)`` compact matrix holding exactly these arrays.

    Raises
    ------
    ValueError
        If ``offsets`` is not ``n_rows + 1`` long, or ``columns`` and ``values`` differ in length.

    See Also
    --------
    [`csr_from_keys`][triwarp.array.csr_from_keys]
    [`csr_from_triplets`][triwarp.array.csr_from_triplets]
    """
    if offsets.size != n_rows + 1:
        raise ValueError(f"offsets must have n_rows + 1 = {n_rows + 1} entries, got {offsets.size}")
    if columns.size != int(values.shape[0]):
        raise ValueError(
            f"columns and values must have one element per entry, got {columns.size} and "
            f"{values.shape[0]}"
        )
    matrix = wps.bsr_matrix_t(values.dtype)()
    matrix.nrow = n_rows
    matrix.ncol = n_cols
    matrix.offsets = offsets
    matrix.columns = columns
    matrix.values = values
    matrix.row_counts = None
    if nnz is None:
        matrix.notify_nnz_changed(nnz_capacity=columns.size)
    else:
        matrix.notify_nnz_changed(nnz=nnz)
    return matrix


def csr_key_buffers(
    count: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Uninitialized ``(keys, order)`` buffers for one [`csr_from_keys`][triwarp.array.csr_from_keys].

    Both are ``2 * count`` long because ``warp.utils.radix_sort_pairs`` sorts in place with its
    upper halves as scratch; a producer writes entry ``i``'s key into ``keys[i]`` and its payload
    (usually ``i`` itself, or anything that locates the entry's value) into ``order[i]``, for
    ``i < count``, and never touches the upper halves.

    Parameters
    ----------
    count
        The number of keys the producer writes, counting duplicates.
    device
        Warp device for both buffers.

    Returns
    -------
    keys, order
        ``wp.uint64`` and ``wp.int32`` arrays of length ``2 * count``.

    See Also
    --------
    [`csr_from_keys`][triwarp.array.csr_from_keys]
    """
    keys = _launch.empty(2 * count, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * count, dtype=wp.int32, device=device)
    return keys, order


def csr_from_keys(
    keys: wp.array[wp.uint64], order: wp.array[wp.int32], count: int, n_rows: int, n_cols: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Sort row-major entry keys and return the CSR pattern they spell, with each entry's positions.

    A producer writes ``count`` keys ``row * n_cols + col`` -- ``kernels/array.csr_key``, which
    maps an index outside the matrix to the sentinel ``n_rows * n_cols`` -- with a payload each,
    into [`csr_key_buffers`][triwarp.array.csr_key_buffers]. This sorts them in place (stably, and
    only over the bits ``n_rows * n_cols`` needs), keeps each run of equal in-range keys as one
    entry and drops the sentinels. Entry ``e``'s contributors are the payloads
    ``order[starts[e] : starts[e + 1]]``, in the order the producer wrote them, so a value kernel
    can form the entry's value deterministically. Nothing is read back to the host.

    Parameters
    ----------
    keys, order
        The ``2 * count`` sort buffers, holding the keys and payloads in their first halves. Both
        are sorted in place, and ``order``'s upper half is overwritten.
    count
        The number of keys written.
    n_rows, n_cols
        The matrix shape the keys encode.

    Returns
    -------
    offsets
        ``n_rows + 1`` row bounds, total-terminated.
    columns
        Sorted column index of each entry, per row; ``count`` long, of which the first
        ``offsets[-1]`` are written.
    starts
        ``count + 1`` long: entry ``e``'s first sorted position is ``starts[e]`` and
        ``starts[offsets[-1]]`` closes the last.

    See Also
    --------
    [`csr_from_triplets`][triwarp.array.csr_from_triplets]
    [`bsr_from_csr`][triwarp.array.bsr_from_csr]
    """
    device = keys.device
    offsets = _launch.zeros(n_rows + 1, dtype=wp.int32, device=device)
    columns = _launch.empty(count, dtype=wp.int32, device=device)
    if count == 0:
        return offsets, columns, _launch.zeros(1, dtype=wp.int32, device=device)
    starts = _launch.empty(count + 1, dtype=wp.int32, device=device)
    sentinel = n_rows * n_cols
    _launch.radix_sort_pairs(keys, order, count=count, end_bit=max(1, sentinel.bit_length()))
    # The run flags and their scan live in the payload buffer's upper half, which is sort scratch.
    inclusive = order[count:]
    _launch.launch(
        kernel_array.csr_run_flags,
        dim=count,
        inputs=[keys, wp.uint64(sentinel), inclusive],
        device=device,
    )
    _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
    _launch.launch(
        kernel_array.csr_from_runs,
        dim=count,
        inputs=[
            keys,
            inclusive,
            wp.uint64(sentinel),
            wp.int32(n_rows),
            wp.uint64(n_cols),
            offsets,
            columns,
            starts,
        ],
        device=device,
    )
    return offsets, columns, starts


def csr_from_triplets(
    n_rows: int,
    n_cols: int,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[Any],
    *,
    prune_numerical_zeros: bool = False,
) -> wps.BsrMatrix[Any]:
    """
    Assemble a sparse matrix from coordinate triplets, summing duplicates.

    The replacement for ``warp.sparse.bsr_from_triplets`` over 1x1 or matrix blocks: a triplet
    whose row or column is outside the matrix is dropped, duplicates are summed in the triplets'
    own order, and nothing is read back. It sorts a key over only the bits the shape needs and
    allocates a fraction of ``bsr_from_triplets``' scratch. Assemblers whose sparsity follows from
    the mesh write their keys directly with [`csr_from_keys`][triwarp.array.csr_from_keys] and
    never materialize the triplets at all.

    Parameters
    ----------
    n_rows, n_cols
        Matrix shape in blocks.
    rows, cols
        ``wp.int32`` block coordinates, one per triplet.
    values
        One block per triplet; its dtype is the matrix's block type (``wp.float32``,
        ``wp.float64``, ``wp.int32`` or ``wp.mat22d``).
    prune_numerical_zeros
        Drop every entry whose *summed* value is exactly zero -- including one whose triplets
        cancel. ``bsr_from_triplets`` instead drops zero-valued triplets before summing and keeps
        an entry that sums to zero, so the two differ only in stored zeros.

    Returns
    -------
    warp.sparse.BsrMatrix
        The assembled ``(n_rows, n_cols)`` matrix. Its ``nnz`` is the triplet count as an upper
        bound until ``nnz_sync()``, exactly as after ``bsr_from_triplets``.

    Raises
    ------
    ValueError
        If ``rows``, ``cols`` and ``values`` differ in length.
    RuntimeError
        If they are not all on one device.
    KeyError
        If ``values``' dtype is not one of the four supported block types.

    See Also
    --------
    [`csr_from_keys`][triwarp.array.csr_from_keys]
    [`triplet_buffers`][triwarp.array.triplet_buffers]
    """
    require_same_device(rows=rows, cols=cols, values=values)
    count = rows.size
    if cols.size != count or int(values.shape[0]) != count:
        raise ValueError(
            f"rows, cols and values must have equal length, got {count}, {cols.size} and "
            f"{values.shape[0]}"
        )
    device = rows.device
    offsets = _launch.zeros(n_rows + 1, dtype=wp.int32, device=device)
    columns = _launch.empty(count, dtype=wp.int32, device=device)
    summed = _launch.empty(count, dtype=values.dtype, device=device)
    if count == 0:
        return bsr_from_csr(n_rows, n_cols, offsets, columns, summed)
    keys, order = csr_key_buffers(count, device)
    if prune_numerical_zeros:
        # Zero-valued triplets are never sorted: they cannot change a sum, so this only saves work.
        _launch.launch(
            kernel_array.COO_KEYS_NONZERO[values.dtype],
            dim=count,
            inputs=[
                rows,
                cols,
                values,
                wp.int32(n_rows),
                wp.int32(n_cols),
                values.dtype(),
                keys,
                order,
            ],
            device=device,
        )
    else:
        _launch.launch(
            kernel_array.coo_keys,
            dim=count,
            inputs=[rows, cols, wp.int32(n_rows), wp.int32(n_cols), keys, order],
            device=device,
        )
    sentinel = n_rows * n_cols
    _launch.radix_sort_pairs(keys, order, count=count, end_bit=max(1, sentinel.bit_length()))
    # One pass flags the entries (pruning on each run's sum), one scan numbers the kept ones, and
    # one pass places them, summing each run where it lands; the flags and their scan live in the
    # payload buffer's upper half, which is sort scratch.
    flags = order[count:]
    if prune_numerical_zeros:
        _launch.launch(
            kernel_array.CSR_NONZERO_RUN_FLAGS[values.dtype],
            dim=count,
            inputs=[keys, order, values, wp.uint64(sentinel), values.dtype(), flags],
            device=device,
        )
    else:
        _launch.launch(
            kernel_array.csr_run_flags,
            dim=count,
            inputs=[keys, wp.uint64(sentinel), flags],
            device=device,
        )
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    _launch.launch(
        kernel_array.CSR_FROM_FLAGGED_RUNS[values.dtype],
        dim=count,
        inputs=[
            keys,
            order,
            values,
            flags,
            wp.uint64(sentinel),
            wp.int32(n_rows),
            wp.uint64(n_cols),
            offsets,
            columns,
            summed,
        ],
        device=device,
    )
    return bsr_from_csr(n_rows, n_cols, offsets, columns, summed)


def index_sparse(
    n_rows: int,
    indices: twt.Array2dInt32,
    data: wp.array[wp.Scalar] | None = None,
    dtype: type[wp.Scalar] | None = None,
    *,
    prune_numerical_zeros: bool = True,
) -> wps.BsrMatrix[wp.Scalar]:
    """
    Build a sparse row/column incidence matrix from flat index columns.

    This mirrors ``trimesh.geometry.index_sparse``, but returns a ``warp.sparse.BsrMatrix``
    in 1x1 BSR (CSR) form instead of ``scipy.sparse.coo_matrix``.

    Parameters
    ----------
    n_rows
        Number of matrix rows (e.g. vertex count). Matrix shape is ``(n_rows, len(indices))``.
    indices
        Integer array of shape ``(m, d)`` — typically ``mesh.faces`` with three vertex indices
        per face.
    data
        Optional 1-D array of length ``m * d``. If omitted, ``wp.ones`` is used; see ``dtype``.
    dtype
        Scalar type for ``wp.ones`` when ``data`` is ``None`` (defaults to ``wp.float32`` if
        ``dtype`` is ``None``). When ``data`` and ``dtype`` are provided, the values of the
        matrix are cast to ``dtype``.
    prune_numerical_zeros
        Forwarded to [`csr_from_triplets`][triwarp.array.csr_from_triplets].

    Returns
    -------
    warp.sparse.BsrMatrix
        Sparse matrix with shape ``(n_rows, len(indices))`` and 1x1 blocks.

    Raises
    ------
    ValueError
        If ``data`` is given and its size differs from ``indices.size``.
    RuntimeError
        If ``indices`` and ``data`` are not all on one device.
    """
    require_same_device(indices=indices, data=data)
    prune_numerical_zeros = prune_numerical_zeros and data is not None
    if data is None:
        data = _launch.ones(
            indices.size, dtype=dtype if dtype is not None else wp.float32, device=indices.device
        )
    else:
        if data.size != indices.size:
            raise ValueError(
                f"data must have the same size as indices, got {data.size} and {indices.size}"
            )
        if dtype is not None and data.dtype != dtype:
            data = astype(data, dtype)

    n_cols, n_repeats = indices.shape
    cols = arange_repeat(n_cols * n_repeats, n_repeats, indices.device)
    return csr_from_triplets(
        n_rows,
        indices.shape[0],
        indices.flatten(),
        cols,
        data,
        prune_numerical_zeros=prune_numerical_zeros,
    )


def isin(
    elements: twt.ArrayNd, test_elements: wp.array[wp.Int], *, max_index: int | None = None
) -> wp.array[wp.bool]:
    """

    Test whether each element appears in ``test_elements`` (``numpy.isin`` for integers).

    Works for every Warp integer dtype -- ``int8`` through ``int64``, ``uint8`` through ``uint64``
    -- and for negative values. Both arrays must share one dtype.

    Two strategies, chosen by the value **span** ``max - min + 1`` taken over both inputs together.
    When the span is modest relative to ``len(test_elements)``, membership is a boolean lookup table
    indexed by ``value - min`` (fast for dense mesh indices). Otherwise ``test_elements`` is sorted
    and each query is a binary search, which keeps memory bounded when the values are sparse in
    their dtype.


    Parameters
    ----------
    elements
        Integer array of **any rank** on the target device. Membership is a per-element predicate,
        so the array is flattened, tested, and the result reshaped back; nothing in the two
        strategies looks at the shape.
    test_elements
        1D array of values to test membership against, of the same dtype as ``elements``.
    max_index
        Optional exclusive upper bound on the values of **both** arrays, which must then be
        non-negative -- the same escape hatch, and the same shape of it, as
        [`hash_indices_rows`][triwarp.grouping.hash_indices_rows]' ``max_index``. Supplying it
        pins the strategy to the direct-index table and skips the two ``triwarp.reduce.minmax``
        reductions that would otherwise infer the span, and with them two host readbacks that
        serialise the device pipeline. Must be positive. Pass it wherever the bound is structural,
        as it is for any buffer of mesh vertex indices.

    Returns
    -------
    wp.array[wp.bool]
        Boolean array with the same shape as ``elements``. All ``False`` when either
        input is empty.

    Raises
    ------
    TypeError
        If either array is not an integer dtype, or the two dtypes differ.
    ValueError
        If ``max_index`` is not positive.

    Warnings
    --------
    A ``max_index`` smaller than the true maximum is a wrong answer, not a raise: a value at or
    above it reads as absent, however far above it lies and on either side of the comparison.
    Unlike [`hash_indices_rows`][triwarp.grouping.hash_indices_rows]' bound it is at least
    memory-safe, because the table lookup range-guards each slot.

    See Also
    --------
    [`indices_to_mask`][triwarp.array.indices_to_mask]
    [`sortable_dtype`][triwarp.typing.sortable_dtype]
    [`numpy.isin`][]

    Notes
    -----
    Dtypes narrower than four bytes are widened to ``int32`` / ``uint32`` (the
    [`sortable_dtype`][triwarp.typing.sortable_dtype] rule) before either strategy runs: Warp's
    radix sort does not accept them, and neither does the tiled min/max reduction the span needs.

    There is no ``assume_unique``, and its absence is deliberate rather than an omission:
    [`numpy.isin`][] gains from one because the sort path behind it calls [`numpy.unique`][] on both
    arrays first, and neither strategy here dedups anything -- the table is an idempotent scatter
    and the binary search is over the sorted keys as given, so duplicates on either side are already
    free. What the two strategies *do* pay for is inferring the value span, which is why the
    guarantee this takes is a bound rather than a uniqueness claim.

    Without ``max_index``: two host readbacks, one min/max reduction per input, which is what
    selects the strategy and anchors the table.
    """
    device = elements.device
    dtype = elements.dtype
    if dtype != test_elements.dtype:
        raise TypeError(
            f"isin requires one dtype for both arrays, got {dtype} and {test_elements.dtype}"
        )
    if not wp.types.type_is_int(dtype) or dtype == wp.bool:
        raise TypeError(f"isin requires an integer dtype, got {dtype}")

    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")

    k = test_elements.size
    if k == 0 or int(elements.size) == 0:
        return _launch.zeros(elements.shape, dtype=wp.bool, device=device)

    is_flat = int(elements.ndim) == 1
    elements_flat = elements if is_flat else elements.flatten()
    # Widen sub-32-bit dtypes once, up front: neither ``reduce.minmax`` nor the radix sort accepts
    # them, and a widened span cannot overflow the type it is measured in (int8's span reaches 256).
    if wp.types.type_size_in_bytes(dtype) < 4:
        wide = twt.sortable_dtype(dtype)
        elements_flat = astype(elements_flat, wide)
        test_elements = astype(test_elements, wide)

    if max_index is not None:
        # A supplied bound is the whole point of the keyword: it is what the two reductions below
        # would have inferred, so it also settles the strategy -- a caller who knows the values are
        # dense mesh indices is describing exactly the table's best case, and honouring a
        # ``_ISIN_MASK_SIZE_FACTOR`` test here would spend the readback to reach the same branch.
        return _reshaped(_isin_lookup_mask(elements_flat, test_elements, max_index, 0), elements)

    # One ``(min, max)`` accumulator over both inputs: the tiled minmax kernel folds into what the
    # buffer already holds, so launching it once per input yields the joint bounds for a single
    # readback, where two ``reduce.minmax`` calls allocate and read back once each.
    reduce_dtype = elements_flat.dtype
    bounds = _launch.array(
        [twt.dtype_max(reduce_dtype), twt.dtype_min(reduce_dtype)],
        dtype=reduce_dtype,
        device=device,
    )
    for values in (elements_flat, test_elements):
        _launch.launch_tiled(
            kernel_reduce.MINMAX1D_TILED[reduce_dtype],
            dim=[kernel_reduce.blocks_1d(values.size)],
            inputs=[values, bounds],
            block_dim=TILE_1D,
            device=device,
        )
    offset, high = (int(bound) for bound in bounds.numpy())
    span = high - offset + 1
    if span <= _ISIN_MASK_SIZE_FACTOR * k:
        out_flat = _isin_lookup_mask(elements_flat, test_elements, span, offset)
    else:
        out_flat = _isin_lookup_sorted(elements_flat, test_elements)

    return _reshaped(out_flat, elements)


def _reshaped(out_flat: wp.array[wp.bool], elements: twt.ArrayNd) -> wp.array[wp.bool]:
    """Restore ``elements``' shape on a per-element answer computed over its flattened view."""
    return out_flat if int(elements.ndim) == 1 else out_flat.reshape(elements.shape)


def _isin_lookup_mask(
    elements_flat: wp.array[wp.Scalar], test_elements: wp.array[wp.Scalar], span: int, offset: int
) -> wp.array[wp.bool]:
    # The table is anchored at ``offset`` (the global minimum over both inputs) rather than at zero,
    # so it holds negative values and stays span-sized instead of max-sized. Anchoring at zero
    # instead is a *silent wrong answer* for negative input: ``mark_membership_mask`` drops the
    # negative test values as out of range, so they read back as absent. A caller-supplied
    # ``max_index`` anchors at zero precisely because it asserts the values are non-negative.
    #
    # Each side is one guarded launch that shifts and then marks or reads -- see
    # ``kernels/array.py::isin_mark_table`` / ``isin_lookup_mask`` for what that buys and why the
    # guard is required.
    #
    # ``last`` is the largest value the table holds, and both sides are shifted through it so that
    # ``shifted_index`` range-tests before it narrows to int32. Computing it here in Python
    # integers is exact: on the inferred path it is the maximum over both inputs, and on the
    # ``max_index`` path it is ``max_index - 1``, so it is representable in ``dtype`` by
    # construction where ``offset + span`` need not be.
    device = elements_flat.device
    dtype = elements_flat.dtype
    anchor = dtype(offset)
    last = dtype(offset + span - 1)
    membership_wp = _launch.zeros(span, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_array.ISIN_MARK_TABLE[dtype],
        dim=test_elements.size,
        inputs=[test_elements, anchor, last, membership_wp],
        device=device,
    )
    out_mask = _launch.empty(elements_flat.size, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_array.ISIN_LOOKUP_MASK[dtype],
        dim=elements_flat.size,
        inputs=[elements_flat, anchor, last, membership_wp, out_mask],
        device=device,
    )
    return out_mask


def _isin_lookup_sorted(
    elements_flat: wp.array[wp.Scalar], test_elements: wp.array[wp.Scalar]
) -> wp.array[wp.bool]:
    device = elements_flat.device
    sorted_test_wp = _sorted_copy(test_elements)
    out_wp = _launch.empty(elements_flat.shape, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_array.ISIN_LOOKUP_SORTED[elements_flat.dtype],
        dim=elements_flat.size,
        inputs=[elements_flat, sorted_test_wp, out_wp],
        device=device,
    )
    return out_wp


def _sorted_copy(values: wp.array[DType]) -> wp.array[DType]:
    """
    Ascending-sorted copy of a 1D scalar array.

    ``sort_and_argsort`` returns a *view* into its own scratch and this outlives the caller's
    frame, so the keys are cloned. The order payload is discarded, which is why the padding value
    it seeds does not matter here.
    """
    if values.size <= 1:
        return values
    return _launch.clone(sort_and_argsort(values)[0])


def flatnonzero(values: wp.array[wp.bool] | wp.array[wp.Scalar]) -> wp.array[wp.int32]:
    """
    Return the indices of the non-zero entries of a 1D array (``numpy.flatnonzero``).

    Takes a boolean mask, which is the common case, or any scalar array — every non-zero value
    selects its index, exactly as ``numpy.flatnonzero`` does, so ``-2`` and ``3`` both count and
    only ``0`` does not.

    Parameters
    ----------
    values
        Length-``n`` ``wp.bool`` mask, or a ``wp.int32`` / float / other scalar array, on the
        target device.

    Returns
    -------
    wp.array[wp.int32]
        Selected indices on ``values.device``, ascending. Empty when nothing is non-zero.

    Raises
    ------
    ValueError
        If ``values`` is not rank-1.

    See Also
    --------
    [`indices_to_mask`][triwarp.array.indices_to_mask]
    [`mask_to_compact_ranks`][triwarp.array.mask_to_compact_ranks]
    [`numpy.flatnonzero`][]
    """
    if int(values.ndim) != 1:
        raise ValueError(f"flatnonzero requires a 1D array, got ndim={values.ndim}")

    device = values.device
    n = values.size
    if n == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    # One buffer for the flags and their scan: the flags are written into it and scanned in place,
    # which ``wp.utils.array_scan`` supports on both devices -- the host scan is a sequential loop
    # that reads each element before overwriting it, and CUB's device scan accepts aliased input
    # and output. The scatter then recovers each flag as the step between neighbouring scan values.
    inclusive = _launch.empty(n, dtype=wp.int32, device=device)
    if values.dtype == wp.bool:
        # A mask needs its own kernel: ``wp.Scalar`` does not instantiate for ``wp.bool``, so
        # ``nonzero_flag`` cannot serve one. For every other dtype the plain cast that kernel
        # replaces would copy the *values*, and the scan below would then sum them instead of
        # counting them.
        _launch.launch(kernel_array.bool_flags, dim=n, inputs=[values, inclusive], device=device)
    else:
        _launch.map(kernel_array.nonzero_flag, values, out=inclusive)

    # Inclusive scan: the total is its last element, so one 4-byte tail read sizes the output
    # (the scatter kernel derives each exclusive position as inclusive[i] - 1).
    _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
    n_out = int(read_scalar(inclusive))

    if n_out == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    out_indices = _launch.empty(n_out, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_scatter.scatter_index_where_scanned,
        dim=n,
        inputs=[inclusive, out_indices],
        device=device,
    )
    return out_indices


def gather(
    src: wp.array[DType] | twt.ArrayNd, indices: wp.array[wp.int32]
) -> wp.array[DType] | twt.ArrayNd:
    """
    Dense copy of ``src`` gathered along its first axis by ``indices`` (``numpy.take``).

    Warp's ``src[indices]`` fancy indexing yields a ``warp.indexedarray`` view; this
    materializes a contiguous ``warp.array`` (performing the copy) so callers get a real
    array supporting ``.reshape`` and a stable return type. Works for rank-1 sources
    (``src[indices]``) and rank-2 row gather (``src[indices, :]``), with any scalar or vector
    ``dtype``.

    Only the first axis is indexable, which is what every caller in this package needs. To take a
    *column*, materialize it with ``wp.clone(src[:, k])`` — a column view is strided, and Warp's
    fancy indexing silently ignores the stride of an index array.

    Parameters
    ----------
    src
        Rank-1 or rank-2 ``wp.array`` on the target device.
    indices
        1D ``wp.int32`` array of indices into the first axis of ``src``. **Must be contiguous**;
        see the warning above.

    Returns
    -------
    wp.array
        Contiguous gathered copy on ``src.device`` with shape
        ``(len(indices), *src.shape[1:])`` and the same ``dtype`` as ``src``. Empty along the
        first axis when ``indices`` is empty.

    Raises
    ------
    RuntimeError
        If ``src`` and ``indices`` are not all on one device.

    See Also
    --------
    [`index_sparse`][triwarp.array.index_sparse]
    [`remap_indices`][triwarp.array.remap_indices]
        The sentinel-preserving variant for index buffers that may carry ``-1`` entries.
    """
    require_same_device(src=src, indices=indices)
    k = indices.size
    out_shape = (k, *(int(dim) for dim in src.shape[1:]))
    out = _launch.empty(out_shape, dtype=src.dtype, device=src.device)
    if k > 0:
        _launch.copy(out, src[indices])
    return out


def astype(values: twt.ArrayNd, dtype: type) -> twt.ArrayNd:
    """
    Element-wise dtype conversion, shape and rank preserved (``numpy.ndarray.astype``).

    The Python-scope counterpart of ``wp.cast``, which exists only inside a kernel. Allocates a
    buffer of ``values``' shape on ``values``' device and fills it with the conversion
    ``warp.utils.array_cast`` performs -- the pair this replaces at twenty-odd call sites -- through
    a concrete kernel for the conversions the package reaches most, and through
    ``warp.utils.array_cast`` itself for any other.

    Parameters
    ----------
    values
        Rank-1 or rank-2 Warp array of any scalar dtype ``array_cast`` accepts.
    dtype
        Target scalar dtype.

    Returns
    -------
    wp.array
        A new array of ``values``' shape on ``values``' device, with element type ``dtype``.

    Raises
    ------
    ValueError
        If ``values`` is rank-2 and not contiguous, since the rank-2 path flattens.

    Notes
    -----
    ``warp.utils.array_cast``'s kernel is ``dest[i] = dest.dtype(src[i])``, which is a *scalar*
    conversion -- handed a rank-2 array it fails to compile, because ``src[i]`` is a row. So rank-2
    input is cast through paired ``flatten()`` views here rather than at each call site, which is
    what several of them were doing by hand.

    The output keeps ``values``' shape, so this is not a reinterpretation that changes rank: an
    ``(n, 3)`` ``float32`` read as ``(n,)`` ``wp.vec3`` is a different operation and still calls
    ``wp.utils.array_cast`` directly, as [`hash_rows`][triwarp.grouping.hash_rows] and
    [`mean_vertex_normals`][triwarp.vertices.mean_vertex_normals] do.

    See Also
    --------
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
        Reinterpret the *bits* rather than convert the value.
    """
    out = _launch.empty(values.shape, dtype=dtype, device=values.device)
    source, target = values, out
    if int(values.ndim) != 1:
        if not values.is_contiguous:
            raise ValueError("astype requires a contiguous array for rank-2 input")
        source, target = values.flatten(), out.flatten()
    # The common conversions launch a concrete kernel; see ``kernels/array.ASTYPE`` for the census
    # behind the table and why any other pair is left to ``wp.utils.array_cast``.
    kernel = kernel_array.ASTYPE.get((values.dtype, dtype))
    n = source.size
    if kernel is None:
        wp.utils.array_cast(source, target)
    elif n > 0:
        _launch.launch(kernel, dim=n, inputs=[source, target], device=values.device)
    return out


def index_bound(indices: twt.IntArray, *, require_non_negative: bool = False) -> int:
    """
    Exclusive upper bound on an index buffer's values, as ``max(indices) + 1``.

    For a face or edge buffer this is the vertex count, and it follows libigl's
    ``F.maxCoeff() + 1`` convention: one past the largest referenced index, so a mesh with trailing
    unreferenced vertices reports fewer than it has. Accepts any ``wp.int32`` index buffer of any
    shape -- a length-``3 * n_faces`` flat triangle buffer, an ``(n, 2)`` edge array -- and reads
    only the 4-byte maximum back to the host.

    Named for the index buffer rather than for vertices because that is all it sees: it is
    index arithmetic, and nothing about it is geometric. Named a *bound* rather than a size
    because that is what a consumer wants it for -- to allocate a table the indices address, or to
    hand [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] its radix -- and because a mesh
    with trailing unreferenced vertices has more of them than this reports.

    Parameters
    ----------
    indices
        A ``wp.int32`` index buffer of any shape (e.g. a flat ``faces`` array or a ``(n, 2)``
        edge array).
    require_non_negative
        Raise if any index is negative. Free: the minimum comes out of the same reduction, the
        same buffer and the same host readback as the maximum, so asking for both costs what
        asking for one does. It exists because the pairing it replaces is not free -- a caller
        that takes this bound and then hands it to
        [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] with ``validate=True`` reduces
        the same array a second time to re-check a bound derived from it, where only the negative
        half of that check can ever fire. Pass this instead and the packing's ``validate=False``.

    Returns
    -------
    int
        ``max(indices) + 1``, or ``0`` when ``indices`` is empty.

    Raises
    ------
    ValueError
        If ``require_non_negative`` is set and any index is negative.
    """
    if int(indices.size) == 0:
        return 0
    # Device-side tiled max: only the 4-byte result crosses to the host, not the whole buffer.
    if not require_non_negative:
        return int(tw.reduce.max(indices)) + 1
    low, high = tw.reduce.minmax(indices)
    if low < 0:
        raise ValueError(f"indices must be non-negative, got a minimum of {int(low)}")
    return int(high) + 1


def indices_to_mask(
    indices: wp.array[wp.int32], n: int, *, device: wp.DeviceLike = None
) -> wp.array[wp.bool]:
    """
    Boolean membership mask of length ``n`` marking each value in ``indices`` as ``True``.

    Wraps the ``mark_membership_mask`` scatter kernel: every ``indices[i]`` sets
    ``out_mask[indices[i]] = True``. The inverse of [`flatnonzero`][triwarp.array.flatnonzero].

    Parameters
    ----------
    indices
        1D ``wp.int32`` array of values in ``[0, n)`` to mark. May be empty.
    n
        Length of the returned mask.
    device
        Target Warp device. Defaults to ``indices.device``.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask, all ``False`` except at positions named by ``indices``.

    See Also
    --------
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`isin`][triwarp.array.isin]
    """
    device = device if device is not None else indices.device
    mask = _launch.zeros(n, dtype=wp.bool, device=device)
    k = indices.size
    if k > 0:
        _launch.launch(
            kernel_scatter.mark_membership_mask, dim=k, inputs=[indices, mask], device=device
        )
    return mask


def mask_to_compact_ranks(
    mask: wp.array[wp.bool], *, invert: bool = False
) -> tuple[wp.array[wp.int32], int]:
    """
    Compact index map over the ``True`` entries of a boolean mask, plus their count.

    Not [`flatnonzero`][triwarp.array.flatnonzero], though the two come from the same scan:
    ``flatnonzero`` returns *which* elements are selected (length ``count``, values are positions
    into ``mask``), while this returns *where each element lands* in the compacted numbering
    (length ``n``, values are compact ranks). Renumbering consumers — the free/fixed
    degree-of-freedom partitions in [`triwarp.smoothing`][triwarp.smoothing] and
    [`triwarp.linalg`][triwarp.linalg] — need this full-length scatter-side form, which
    ``flatnonzero`` cannot provide without an extra pass.

    Parameters
    ----------
    mask
        Length-``n`` ``wp.bool`` array.
    invert
        When ``True``, map the ``False`` entries instead. Useful for a free/fixed degree-of-freedom
        partition, where the mask marks the *constrained* entries and the compact map is wanted over
        the unconstrained complement (see [`free_partition`][triwarp.linalg.free_partition]).

    Returns
    -------
    compact_ranks : wp.array[wp.int32]
        Length-``n`` array on ``mask.device``: an exclusive scan of the (optionally inverted) mask,
        so ``index_map[i]`` is the compact 0-based rank of element ``i`` among the selected
        entries at or before it (meaningful only where element ``i`` is itself selected).
    count : int
        Total number of selected entries in ``mask``.
    """
    device = mask.device
    n = mask.size
    if n == 0:
        return _launch.zeros(0, dtype=wp.int32, device=device), 0
    # The flags are written straight into the tail of the ``n + 1`` offsets buffer and scanned
    # there in place, so the separate flag buffer ``counts_to_offsets`` would take them from is
    # never allocated.
    buffer = _launch.zeros(n + 1, dtype=wp.int32, device=device)
    flags = twt.as_dense(buffer[1:])
    if invert:
        _launch.map(kernel_array.complement_flag, mask, out=flags)
    else:
        _launch.launch(kernel_array.bool_flags, dim=n, inputs=[mask, flags], device=device)
    return twt.as_dense(buffer[:n]), _scan_into_tail(buffer, flags, flags)


def counts_to_offsets(counts: wp.array[wp.int32]) -> tuple[wp.array[wp.int32], int]:
    """
    Total-terminated exclusive prefix sum of ``counts``, plus their total.

    The CSR-building step that turns per-element counts into row starts. Done in **one** scan pass
    and one 4-byte host read: the scan runs *inclusive* into the tail of an ``n + 1`` buffer whose
    leading zero is already in place, which makes the first ``n`` entries the exclusive sum and the
    last entry the total. The obvious spelling -- one exclusive scan for the offsets and a second
    inclusive scan (or a ``reduce.sum``) for the total -- costs a second full pass over ``counts``,
    and reading the total as ``inclusive.numpy()[-1]`` copies the whole array to the host to look at
    one element of it.

    Parameters
    ----------
    counts
        Length-``n`` ``wp.int32`` per-element counts.

    Returns
    -------
    offsets : wp.array[wp.int32]
        Length-``n + 1`` total-terminated offsets (the package's one convention, stated at
        [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]): element ``i`` owns
        ``[offsets[i], offsets[i + 1])`` and ``offsets[n] == total``. ``[0]`` when ``counts`` is
        empty.
    total : int
        Sum of ``counts``, read back so a caller sizing an output needs no second device access.

    Notes
    -----
    **This is for callers that want ``total``**, which it reads back unconditionally, and a host
    readback serialises the device pipeline. A caller that only needs the offsets and already knows
    its buffer size should keep the open-coded ``wp.zeros(n + 1)`` plus a scan into ``[1:]``, as
    ``halfedge.vertex_one_rings`` and ``adjacency.vertex_face_adjacency`` do: both size their
    payload from ``3 * n_faces`` and would gain a synchronization they currently do not have.

    See Also
    --------
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`mask_to_compact_ranks`][triwarp.array.mask_to_compact_ranks]
    """
    # The unconditional readback of ``total`` is a host synchronization -- which is why the Notes
    # section advises open-coding the scan when only the offsets are wanted.
    n = counts.size
    device = counts.device
    if n == 0:
        return _launch.zeros(1, dtype=wp.int32, device=device), 0
    # The leading zero from ``wp.zeros`` is the first exclusive offset; the inclusive scan fills the
    # rest, so ``buffer[n]`` is the total and ``buffer[:n]`` the exclusive offsets.
    buffer = _launch.zeros(n + 1, dtype=wp.int32, device=device)
    return buffer, _scan_into_tail(buffer, counts, buffer[1:])


def _scan_into_tail(
    buffer: wp.array[wp.int32], counts: wp.array[wp.int32], tail: wp.array[wp.int32]
) -> int:
    """
    Scan ``counts`` inclusively into ``tail`` (``buffer[1:]``) and read the total off the end.

    The shared tail of [`counts_to_offsets`][triwarp.array.counts_to_offsets] and
    [`mask_to_compact_ranks`][triwarp.array.mask_to_compact_ranks]. ``counts`` may *be* ``tail``:
    ``wp.utils.array_scan`` scans in place on both devices (see
    [`flatnonzero`][triwarp.array.flatnonzero]), which is what lets the mask form skip its own flag
    buffer.
    """
    _launch.array_scan(counts, out_array=tail, inclusive=True)
    return int(read_scalar(buffer))


def remap_indices(indices: wp.array[wp.int32], remap: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Remap an index buffer through a lookup table, passing negative (sentinel) entries through.

    Not a plain [`gather`][triwarp.array.gather]: the ``-1`` slots that mark padded or removed
    entries in an index buffer must survive the remap unchanged, where a gather would read out of
    bounds on them. [`triwarp.repair`][triwarp.repair] relies on this — its functions preserve
    ``-1`` face sentinels through vertex renumbering (see ``remove_unreferenced_vertices``).

    Parameters
    ----------
    indices
        1D ``wp.int32`` array of indices into ``remap`` (e.g. a flat face buffer). Negative
        entries are passed through unchanged.
    remap
        1D ``wp.int32`` lookup table (e.g. old-to-new vertex index map).

    Returns
    -------
    wp.array[wp.int32]
        Length ``len(indices)`` array on ``indices.device`` with ``out[i] = remap[indices[i]]``
        for non-negative ``indices[i]``, and ``out[i] = indices[i]`` otherwise.

    Raises
    ------
    RuntimeError
        If ``indices`` and ``remap`` are not all on one device.

    See Also
    --------
    [`gather`][triwarp.array.gather]
        The sentinel-free form: a dense first-axis gather for index buffers known to be in range.
    """
    require_same_device(indices=indices, remap=remap)
    n = indices.size
    device = indices.device
    if n == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)
    out = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_array.gather_1d_skip_negative, dim=n, inputs=[indices, remap, out], device=device
    )
    return out


def trim_to_count(
    counter: wp.array[wp.int32], *buffers: wp.array[DType]
) -> tuple[int, list[wp.array[DType]]]:
    """
    Trim atomic-append output buffers to the number of elements actually written.

    A kernel that emits an unpredictable number of results cannot size its output ahead of
    time. The usual pattern is to over-allocate the output buffers to a safe upper bound and
    have each thread claim its slots with ``wp.atomic_add`` on a shared length-1 ``counter``,
    writing into ``buffer[slot]``. After the launch, only the first ``n_out`` slots hold valid
    data and the tail is uninitialized, but ``n_out`` is known only on the device.

    This finalizes that pattern: it reads the counter back to the host once, then copies the
    valid prefix ``buffer[:n_out]`` of each over-allocated buffer into a freshly allocated,
    exact-size array. Pass every buffer filled by the same counter in one call so they are all
    trimmed to a consistent length.

    Parameters
    ----------
    counter
        Length-1 ``wp.int32`` array holding the final atomic-append count on the target device.
    *buffers
        Over-allocated output buffers to trim, all indexed along their first axis by the same
        counter. Any rank and ``dtype``; trailing dimensions are preserved.

    Returns
    -------
    n_out : int
        The counter value: the number of valid leading elements in each buffer.
    trimmed : list[wp.array]
        One contiguous ``(n_out, *buffer.shape[1:])`` copy per input buffer, in order, each on
        its buffer's device.
    """
    n_out = int(read_scalar(counter, 0))
    trimmed = []
    for buffer in buffers:
        out_shape = (n_out, *(int(dim) for dim in buffer.shape[1:]))
        out = _launch.empty(out_shape, dtype=buffer.dtype, device=buffer.device)
        if n_out > 0:
            _launch.copy(out, buffer[:n_out])
        trimmed.append(out)
    return n_out, trimmed


def bitcast_to_int(
    data: wp.array[wp.Scalar], count: int | None = None
) -> wp.array[wp.int32] | wp.array[wp.int64]:
    """
    Reinterpret an array's underlying bits as a same-width signed integer dtype.

    32-bit-or-narrower dtypes (``wp.int32``, ``wp.uint32``, ``wp.float32``, and narrower) are
    reinterpreted as ``wp.int32``; wider dtypes (``wp.int64``, ``wp.uint64``, ``wp.float64``) as
    ``wp.int64``. Narrower-than-32-bit floating point values are first upcast to ``wp.float32``
    (a numeric cast, not a bit reinterpretation) so every dtype narrower than 32 bits shares one
    ``wp.int32`` key space. Used by the hashing/uniqueness machinery
    ([`unique_1d`][triwarp.grouping.unique_1d]) to give arbitrary scalar dtypes a common sortable,
    hashable integer key.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` of any scalar dtype.
    count
        Output length. Defaults to ``data.shape[0]``. When greater than the input length, the
        tail is left uninitialized (over-allocation for in-place radix-sort scratch).

    Returns
    -------
    wp.array[wp.int32] | wp.array[wp.int64]
        Bit-reinterpreted (or, for sub-32-bit floats, upcast-then-reinterpreted) copy of length
        ``count`` on ``data.device``.

    See Also
    --------
    [`bitcast_from_int`][triwarp.array.bitcast_from_int]
    """
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n = data.size
    # ``is None``, not ``or``: zero is a legitimate length and ``or`` would read it as "not passed"
    # and hand back the whole buffer -- an ``n``-element array of stale bits where the caller asked
    # for an empty one. The signature already spells the distinction; only this line lost it.
    count = n if count is None else count
    copy_count = min(n, count)
    target = wp.int64 if n_bits > 32 else wp.int32

    # ``count=0`` reaches ``wp.copy`` and ``wp.utils.array_cast`` as *"copy the whole source"* --
    # a documented back-compatibility rule in Warp 1.17 (``if count == 0: count = src.size``), so
    # the zero that means "nothing" and the zero that means "everything" are the same argument.
    # Into a length-0 destination that is not even a clean refusal: it raises ``TypeError:
    # unsupported operand type(s) for +: 'NoneType' and 'int'`` from inside the copy. Returning the
    # empty allocation here is both the right answer and the only way to state it.
    if copy_count == 0:
        return _launch.empty(count, dtype=target, device=data.device)

    if n_bits > 32:
        reinterpreted = _launch.empty(count, dtype=wp.int64, device=data.device)
        _launch.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    reinterpreted = _launch.empty(count, dtype=wp.int32, device=data.device)
    if wp.types.type_is_float(data.dtype):
        src = data
        if n_bits < 32:
            src = _launch.empty(copy_count, dtype=wp.float32, device=data.device)
            wp.utils.array_cast(data, src, count=copy_count)
        _launch.copy(reinterpreted, src, count=copy_count)
    else:
        wp.utils.array_cast(data, reinterpreted, count=copy_count)
    return reinterpreted


def bitcast_from_int(
    data: wp.array[wp.int32] | wp.array[wp.int64], dtype: type[wp.Scalar], count: int | None = None
) -> wp.array[wp.Scalar]:
    """
    Inverse of [`bitcast_to_int`][triwarp.array.bitcast_to_int]: recover the original dtype.

    Reinterprets (same-width) or upcasts-then-reinterprets (narrower target) the bits produced
    by ``bitcast_to_int`` back into ``dtype``.

    Parameters
    ----------
    data
        Rank-1 ``wp.array[wp.int32]`` or ``wp.array[wp.int64]``, typically the output of
        [`bitcast_to_int`][triwarp.array.bitcast_to_int].
    dtype
        Target scalar dtype to reinterpret ``data`` as.
    count
        Output length. Defaults to ``data.shape[0]``. When greater than the input length, the
        tail is left uninitialized.

    Returns
    -------
    wp.array[wp.Scalar]
        Array of dtype ``dtype`` and length ``count`` on ``data.device``.

    See Also
    --------
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
    """
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n_target_bits = wp.types.type_size_in_bytes(dtype) * 8
    n = data.size
    # ``is None`` rather than ``or``, and the zero-length short circuit, both for the reasons
    # recorded at ``bitcast_to_int``: Warp reads ``count=0`` as "copy everything".
    count = n if count is None else count
    copy_count = min(n, count)
    if copy_count == 0:
        return _launch.empty(count, dtype=dtype, device=data.device)

    if n_bits == n_target_bits:
        reinterpreted = _launch.empty(count, dtype=dtype, device=data.device)
        _launch.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    if wp.types.type_is_float(dtype) and n_bits > n_target_bits:
        wide_dtype = getattr(wp, f"float{n_bits}")
        wide = _launch.empty(count, dtype=wide_dtype, device=data.device)
        _launch.copy(wide, data, count=copy_count)
        reinterpreted_casted = _launch.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(wide, reinterpreted_casted, count=copy_count)
    else:
        reinterpreted_casted = _launch.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(data, reinterpreted_casted, count=copy_count)
    return reinterpreted_casted


# ---------------------------------------------------------------------------
# Private cross-cutting helpers (used by a single external caller today; promote to public
# only with demonstrated cross-module demand).
# ---------------------------------------------------------------------------


def _check_int32_fits(value: int, name: str) -> None:
    # The range guard for the three index-buffer builders, which are ``int32`` by signature. It is
    # what stands in for a ``dtype`` argument on them: a range too wide for the buffer raises here
    # instead of wrapping silently in the ``wp.int32(...)`` constructor a few lines on.
    vmin = twt.dtype_min(wp.int32)
    vmax = twt.dtype_max(wp.int32)
    if value < vmin or value > vmax:
        raise ValueError(f"{name}={value} is out of range for int32 [{vmin}, {vmax}]")
