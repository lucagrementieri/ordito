"""Global reductions on Warp arrays."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, cast, overload

import numpy as np
import warp as wp

import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, require_same_device
from triwarp.array import _sorted_copy, astype  # pyright: ignore[reportPrivateUsage]
from triwarp.constants import TILE_1D, TILE_2D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import reduce as kernel_reduce


# The global overloads take the rank-agnostic ``ArrayNd*`` aliases so a caller holding this
# package's usual ``wp.array[dtype]`` spelling resolves: ``NDim`` is invariant, so pinning the
# rank here rejects it (triwarp.typing, "accept wide, return narrow"). The integer arm spans
# ``kernel_reduce._GLOBAL_DTYPES``, which is wider than int32. The ``axis`` overload stays on
# ``Array2dScalar`` deliberately -- an axis reduction is rank-2 only and ``_AXIS_DTYPES`` is only
# the three narrow dtypes, so the rank and the dtype are both real constraints there.
@overload
def min(array: twt.ArrayNdInt, *, axis: None = ...) -> int: ...
@overload
def min(array: twt.ArrayNdFloat, *, axis: None = ...) -> float: ...
@overload
def min(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
def min(
    array: twt.ArrayNdScalar, *, axis: Literal[0, 1] | None = None
) -> float | int | twt.Array1dScalar:
    """
    Minimum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min`` + ``wp.atomic_min``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` scalar array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; ``(n,)`` (``axis=1``) or ``(m,)`` (``axis=0``) array
        otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 input, or ``axis`` is not ``0``, ``1``, or ``None`` for a rank-2 input.
    """
    return cast(float | int | twt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["min"]))


@overload
def max(array: twt.ArrayNdInt, *, axis: None = ...) -> int: ...
@overload
def max(array: twt.ArrayNdFloat, *, axis: None = ...) -> float: ...
@overload
def max(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
def max(
    array: twt.ArrayNdScalar, *, axis: Literal[0, 1] | None = None
) -> float | int | twt.Array1dScalar:
    """
    Maximum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_max`` + ``wp.atomic_max``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` scalar array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; ``(n,)`` (``axis=1``) or ``(m,)`` (``axis=0``) array
        otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 input, or ``axis`` is not ``0``, ``1``, or ``None`` for a rank-2 input.
    """
    return cast(float | int | twt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["max"]))


@overload
def minmax(array: twt.ArrayNdInt, *, axis: None = ...) -> tuple[int, int]: ...
@overload
def minmax(array: twt.ArrayNdFloat, *, axis: None = ...) -> tuple[float, float]: ...
@overload
def minmax(array: wp.array[wp.vec3], *, axis: None = ...) -> tuple[wp.vec3, wp.vec3]: ...
@overload
def minmax(
    array: twt.Array2dScalar, *, axis: Literal[0, 1]
) -> tuple[twt.Array1dScalar, twt.Array1dScalar]: ...
def minmax(
    array: twt.ArrayNdScalar | wp.array[wp.vec3], *, axis: Literal[0, 1] | None = None
) -> (
    tuple[float, float]
    | tuple[int, int]
    | tuple[wp.vec3, wp.vec3]
    | tuple[twt.Array1dScalar, twt.Array1dScalar]
):
    """
    Minimum and maximum of ``array``.

    With ``axis=None`` (default), reduces every element to two Python scalars using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min``/``tile_max`` +
    ``wp.atomic_min``/``atomic_max``). Either direction alone would be one launch, one
    buffer and one readback too, so asking for both costs nothing extra.

    A rank-1 ``wp.vec3`` array reduces component-wise to a ``(wp.vec3, wp.vec3)`` corner
    pair — one chunked kernel into a single six-slot buffer (the upper corner negated so
    one ``inf`` fill seeds both ends) and one readback, which is what keeps
    [`aabb`][triwarp.bounds.aabb] host-latency-bound and nothing more.

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    pair of 1D ``wp.array`` buffers (min, max) of the same dtype. The two are rows 0 and 1 of
    one contiguous ``(2, k)`` buffer, so the max starts exactly ``k`` elements after the min
    and a caller that needs both on the host can read the pair back as one copy.

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` scalar array, or a ``(n,)`` ``wp.vec3`` array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for per-axis 1D
        results (rank-2 scalar input only).

    Returns
    -------
    tuple[float, float] | tuple[int, int] | tuple[wp.vec3, wp.vec3] | tuple[wp.array, wp.array]
        ``(min, max)`` as Python scalars — or ``wp.vec3`` corners for a ``wp.vec3`` input —
        when ``axis=None``; pair of ``(n,)`` (``axis=1``) or ``(m,)`` (``axis=0``) arrays
        otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 or ``wp.vec3`` input, or ``axis`` is not ``0``, ``1``, or ``None`` for a
        rank-2 input.

    See Also
    --------
    [`aabb`][triwarp.bounds.aabb]
        The mesh-facing spelling of the ``wp.vec3`` reduction, with the empty-input
        ``(+inf, -inf)`` convention instead of a raise.
    """
    if array.dtype == wp.vec3:
        if axis is not None:
            raise ValueError("minmax requires axis=None for a wp.vec3 array.")
        return _launch_global_vec3_minmax(cast("wp.array[wp.vec3]", array))
    return cast(
        tuple[float, float] | tuple[int, int] | tuple[twt.Array1dScalar, twt.Array1dScalar],
        _reduce_scalar(cast(twt.ScalarArray, array), axis, _SCALAR_REDUCE["minmax"]),
    )


@overload
def any(array: wp.array[wp.bool], *, axis: None = None) -> bool: ...
@overload
def any(array: wp.array[wp.bool], *, axis: Literal[0, 1]) -> wp.array[wp.bool]: ...
def any(array: wp.array[wp.bool], *, axis: Literal[0, 1] | None = None) -> wp.array[wp.bool] | bool:
    """
    Reduce a boolean array with logical OR.

    For rank-1 input, or rank-2 input with ``axis=None``, reduce all elements
    to one Python ``bool``. For rank-2 input with an explicit axis, reduce along
    that axis to a 1D ``wp.bool`` array.

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` array. Must be non-empty.
    axis
        ``0``, ``1``, or ``None``. For rank-2 input, ``None`` reduces all elements
        to a single scalar. Ignored for rank-1 input.

    Returns
    -------
    bool | wp.array[wp.bool]
        Python ``bool`` for rank-1 input or rank-2 with ``axis=None``.
        ``(array.shape[1],)`` when ``axis=0``, else ``(array.shape[0],)``,
        for rank-2 input with an explicit axis.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``0``, ``1``, or ``None`` for rank-2 input.
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["any"])


@overload
def sum(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def sum(array: twt.ArrayNdInt, *, axis: None = ...) -> int: ...
@overload
def sum(array: twt.ArrayNdFloat, *, axis: None = ...) -> float: ...
@overload
def sum(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: None = ...) -> int: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: Literal[0, 1]) -> twt.Array1dInt32: ...
def sum(
    array: twt.ArrayNdScalar | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | int | twt.Array1dScalar | twt.Array1dInt32 | wp.vec3:
    """
    Sum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_sum`` + ``wp.atomic_add``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    For ``wp.bool`` input, counts ``True`` values and returns ``int`` (global) or
    ``wp.int32`` (per-axis).

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` scalar or ``wp.bool`` array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; ``(n,)`` (``axis=1``) or ``(m,)`` (``axis=0``) array
        otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 input, or ``axis`` is not ``0``, ``1``, or ``None`` for a rank-2 input.
    """
    if array.dtype == wp.vec3:
        if axis is not None:
            raise ValueError("sum over a vec3 array supports only axis=None.")
        if array.ndim != 1:
            raise ValueError("sum over a vec3 array requires a 1D array.")
        n = int(array.shape[0])
        if n == 0:
            raise ValueError("sum requires a non-empty array.")
        total = _sum_in_fixed_order(kernel_reduce.sum_vec3_1d_partials, [array], n, wp.vec3)
        return cast(wp.vec3, total.list()[0])
    if array.dtype == wp.bool:
        mask = cast(wp.array[wp.bool], array)
        if axis is None:
            # Counted straight off the mask bytes; see ``_launch_global_bool_tiled`` for why the
            # ``int32`` widening is worth avoiding here, and why the ``axis`` branch below keeps
            # it.
            flat = mask.flatten() if mask.ndim == 2 else mask
            if flat.size == 0:
                raise ValueError("sum requires a non-empty array.")
            total = _launch.zeros(1, dtype=wp.int32, device=flat.device)
            _launch.launch_tiled(
                kernel_reduce.sum_bool_1d_tiled,
                dim=[kernel_reduce.blocks_1d(flat.size)],
                inputs=[flat, total],
                block_dim=TILE_1D,
                device=flat.device,
            )
            return int(read_scalar(total, 0))
        result = _reduce_scalar(
            cast(twt.ScalarArray, astype(mask, wp.int32)), axis, _SCALAR_REDUCE["sum"]
        )
        return cast(twt.Array1dInt32, result)
    return cast(
        float | int | twt.Array1dScalar,
        _reduce_scalar(cast(twt.ScalarArray, array), axis, _SCALAR_REDUCE["sum"]),
    )


@overload
def mean(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def mean(array: twt.ArrayNdScalar | wp.array[wp.bool], *, axis: None = ...) -> float: ...
@overload
def mean(
    array: twt.Array2dScalar | wp.array[wp.bool], *, axis: Literal[0, 1]
) -> twt.Array1dFloat32: ...
def mean(
    array: twt.ArrayNdScalar | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | twt.Array1dFloat32 | wp.vec3:
    """
    Arithmetic mean of ``array``.

    Delegates the reduction to the tiled [`sum`][triwarp.reduce.sum], then divides by the
    element count. With ``axis=None`` (default), returns one Python ``float``. With
    ``axis=0`` or ``axis=1`` on a rank-2 input, returns a 1D ``wp.float32`` array.

    The result is always floating point regardless of input dtype. For ``wp.bool``
    input, the mean is the fraction of ``True`` values. For a 1D ``wp.vec3`` input
    the mean is the component-wise average ``wp.vec3`` (``axis=None`` only).

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` scalar or ``wp.bool`` array, or a ``(n,)`` ``wp.vec3`` array. Must
        be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | wp.vec3 | wp.array
        Global ``float`` (scalar/``wp.bool`` input) or ``wp.vec3`` (``wp.vec3``
        input) when ``axis=None``; ``(n,)`` (``axis=1``) or ``(m,)`` (``axis=0``) ``wp.float32``
        array otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 (or ``wp.vec3``) input, or ``axis`` is not ``0``, ``1``, or ``None`` for a
        rank-2 input.
    """
    # ``axis`` is narrowed before ``sum`` is called: each of its overloads pins ``axis`` to one
    # side, so the union cannot resolve against any of them. The axis path below casts to
    # ``Array2dScalar`` unconditionally, which is a no-op at runtime -- a ``wp.vec3`` array falls
    # through it into ``sum``, which raises "supports only axis=None". That rule belongs to ``sum``
    # alone; restating it here would be a second copy of one decision.
    if axis is None:
        if array.dtype == wp.vec3:
            vectors = cast("wp.array[wp.vec3]", array)
            return sum(vectors) / float(array.size)
        total = sum(cast(twt.ArrayNdScalar, array))
        return float(total) / array.size
    sums = sum(cast(twt.Array2dScalar, array), axis=axis)
    out = astype(sums, wp.float32)
    _launch.map(wp.div, out, wp.float32(array.shape[axis]), out=out)
    return cast(twt.Array1dFloat32, out)


@overload
def weighted_sum(values: twt.ArrayNdFloat32, weights: twt.ArrayNdFloat32) -> float: ...
@overload
def weighted_sum(values: wp.array[wp.vec3], weights: twt.ArrayNdFloat32) -> wp.vec3: ...
def weighted_sum(
    values: twt.ArrayNdFloat32 | wp.array[wp.vec3], weights: twt.ArrayNdFloat32
) -> float | wp.vec3:
    """
    Weighted sum ``sum_i weights[i] * values[i]``.

    Parameters
    ----------
    values
        ``(n,)`` ``wp.float32`` or ``wp.vec3`` array. Must be non-empty.
    weights
        ``(n,)`` weights, the same length as ``values``.

    Returns
    -------
    float | wp.vec3
        Scalar weighted sum on the host (``float`` for scalar ``values``,
        ``wp.vec3`` for ``wp.vec3`` ``values``).

    Raises
    ------
    ValueError
        If either array is not rank-1, either is empty, or their lengths differ.
    RuntimeError
        If ``values`` and ``weights`` are not all on one device.
    """
    require_same_device(values=values, weights=weights)
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("weighted_sum requires rank-1 values and weights arrays.")
    n_values = int(values.shape[0])
    n_weights = int(weights.shape[0])
    if n_values == 0 or n_weights == 0:
        raise ValueError("weighted_sum requires non-empty arrays.")
    if n_values != n_weights:
        raise ValueError("weighted_sum requires values and weights of equal length.")

    if values.dtype == wp.vec3:
        total = _sum_in_fixed_order(
            kernel_reduce.weighted_sum_vec3_1d_partials, [values, weights], n_values, wp.vec3
        )
        return cast(wp.vec3, total.list()[0])
    total = _sum_in_fixed_order(
        kernel_reduce.weighted_sum1d_partials, [values, weights], n_values, wp.float32
    )
    return float(read_scalar(total, 0))


@overload
def all(array: wp.array[wp.bool], *, axis: None = None) -> bool: ...
@overload
def all(array: wp.array[wp.bool], *, axis: Literal[0, 1]) -> wp.array[wp.bool]: ...
def all(array: wp.array[wp.bool], *, axis: Literal[0, 1] | None = None) -> wp.array[wp.bool] | bool:
    """
    Reduce a boolean array with logical AND.

    For rank-1 input, or rank-2 input with ``axis=None``, reduce all elements
    to one Python ``bool``. For rank-2 input with an explicit axis, reduce along
    that axis to a 1D ``wp.bool`` array.

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` array. Must be non-empty.
    axis
        ``0``, ``1``, or ``None``. For rank-2 input, ``None`` reduces all elements
        to a single scalar. Ignored for rank-1 input.

    Returns
    -------
    bool | wp.array[wp.bool]
        Python ``bool`` for rank-1 input or rank-2 with ``axis=None``.
        ``(array.shape[1],)`` when ``axis=0``, else ``(array.shape[0],)``,
        for rank-2 input with an explicit axis.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``0``, ``1``, or ``None`` for rank-2 input.
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["all"])


def median(array: twt.ArrayNdScalar) -> float:
    """
    Median of a 1D scalar array (``numpy.median``).

    Sorts a copy of the values on-device with ``warp.utils.radix_sort_pairs`` and reads the
    middle element (odd length) or averages the two middle elements (even length). The result
    is always a Python ``float``, regardless of input dtype.

    Parameters
    ----------
    array
        ``(n,)`` array of any 32- or 64-bit scalar dtype accepted by
        ``warp.utils.radix_sort_pairs`` (``wp.int32``, ``wp.uint32``, ``wp.float32``,
        ``wp.int64``, ``wp.uint64``, ``wp.float64``). Must be non-empty.

    Returns
    -------
    float
        The median value on the host.

    Raises
    ------
    ValueError
        If ``array`` is empty or not rank-1.
    """
    if array.ndim != 1:
        raise ValueError("median requires a 1D array.")
    n = array.size
    if n == 0:
        raise ValueError("median requires a non-empty array.")

    sorted_values = _sorted_copy(array)
    if n % 2 == 1:
        return float(read_scalar(sorted_values, n // 2))
    middle = sorted_values[n // 2 - 1 : n // 2 + 1].numpy()
    return (float(middle[0]) + float(middle[1])) / 2.0


# Every kernel slot below is a ``{dtype: wp.Kernel}`` table rather than one kernel, because
# ``kernels/reduce.py`` bakes the dtype into each factory instantiation -- see the note above
# ``MIN1D_TILED`` there for why (a ``wp.Scalar`` template made every launch pay Warp's host-side
# overload resolution). The bool specs are the exception: their masks are cast to ``wp.int32``
# before any kernel sees them, so one kernel each.
class _ScalarReduceSpec(NamedTuple):
    name: str
    axis_rows_tiled: kernel_array.KernelTable
    axis_cols_tiled: kernel_array.KernelTable
    axis_rows_serial: kernel_array.KernelTable
    axis_cols_serial: kernel_array.KernelTable
    tiled_1d: kernel_array.KernelTable
    tiled_2d: kernel_array.KernelTable
    init_global: Callable[[type], int | float]
    global_output_slots: int
    dual_axis: bool


class _BoolReduceSpec(NamedTuple):
    name: str
    bool_1d: twt.Kernel
    axis_rows_tiled: twt.Kernel
    axis_cols_tiled: twt.Kernel
    axis_rows_serial: twt.Kernel
    axis_cols_serial: twt.Kernel
    tiled_1d: twt.Kernel
    init_global: int


_SCALAR_REDUCE: dict[str, _ScalarReduceSpec] = {
    "min": _ScalarReduceSpec(
        name="min",
        axis_rows_tiled=kernel_reduce.MIN_2D_ROWS_TILED,
        axis_cols_tiled=kernel_reduce.MIN_2D_COLS_TILED,
        axis_rows_serial=kernel_reduce.MIN_2D_ROWS_SERIAL,
        axis_cols_serial=kernel_reduce.MIN_2D_COLS_SERIAL,
        tiled_1d=kernel_reduce.MIN1D_TILED,
        tiled_2d=kernel_reduce.MIN2D_TILED,
        init_global=twt.dtype_max,
        global_output_slots=1,
        dual_axis=False,
    ),
    "max": _ScalarReduceSpec(
        name="max",
        axis_rows_tiled=kernel_reduce.MAX_2D_ROWS_TILED,
        axis_cols_tiled=kernel_reduce.MAX_2D_COLS_TILED,
        axis_rows_serial=kernel_reduce.MAX_2D_ROWS_SERIAL,
        axis_cols_serial=kernel_reduce.MAX_2D_COLS_SERIAL,
        tiled_1d=kernel_reduce.MAX1D_TILED,
        tiled_2d=kernel_reduce.MAX2D_TILED,
        init_global=twt.dtype_min,
        global_output_slots=1,
        dual_axis=False,
    ),
    "minmax": _ScalarReduceSpec(
        name="minmax",
        axis_rows_tiled=kernel_reduce.MINMAX_2D_ROWS_TILED,
        axis_cols_tiled=kernel_reduce.MINMAX_2D_COLS_TILED,
        axis_rows_serial=kernel_reduce.MINMAX_2D_ROWS_SERIAL,
        axis_cols_serial=kernel_reduce.MINMAX_2D_COLS_SERIAL,
        tiled_1d=kernel_reduce.MINMAX1D_TILED,
        tiled_2d=kernel_reduce.MINMAX2D_TILED,
        init_global=twt.dtype_max,
        global_output_slots=2,
        dual_axis=True,
    ),
    "sum": _ScalarReduceSpec(
        name="sum",
        axis_rows_tiled=kernel_reduce.SUM_2D_ROWS_TILED,
        axis_cols_tiled=kernel_reduce.SUM_2D_COLS_TILED,
        axis_rows_serial=kernel_reduce.SUM_2D_ROWS_SERIAL,
        axis_cols_serial=kernel_reduce.SUM_2D_COLS_SERIAL,
        tiled_1d=kernel_reduce.SUM1D_TILED,
        tiled_2d=kernel_reduce.SUM2D_TILED,
        init_global=twt.dtype_zero,
        global_output_slots=1,
        dual_axis=False,
    ),
}

_BOOL_REDUCE: dict[str, _BoolReduceSpec] = {
    "any": _BoolReduceSpec(
        name="any",
        bool_1d=kernel_reduce.any_bool_1d_tiled,
        axis_rows_tiled=kernel_reduce.any_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.any_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.any_2d_rows_serial,
        axis_cols_serial=kernel_reduce.any_2d_cols_serial,
        tiled_1d=kernel_reduce.any_1d_tiled,
        init_global=0,
    ),
    "all": _BoolReduceSpec(
        name="all",
        bool_1d=kernel_reduce.all_bool_1d_tiled,
        axis_rows_tiled=kernel_reduce.all_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.all_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.all_2d_rows_serial,
        axis_cols_serial=kernel_reduce.all_2d_cols_serial,
        tiled_1d=kernel_reduce.all_1d_tiled,
        init_global=1,
    ),
}


# --- private helpers ---------------------------------------------------------------------
#
# The launch/dispatch layer every public reduction above shares, kept together because that is what
# makes the six of them one mechanism rather than six. ``_launch_global_vec3_minmax`` is the only
# member with a single caller (``minmax``) and it stays here for that reason: it is the ``wp.vec3``
# sibling of ``_launch_global_scalar_tiled`` below, and moving it 20 functions up would put half of
# one dispatch table in the middle of the public surface.


def _sum_in_fixed_order(
    first: twt.Kernel, inputs: list[twt.ArrayNd], n: int, dtype: type
) -> wp.array[Any]:
    """
    Sum a float reduction in a fixed order, into a ``(1,)`` device array holding the total.

    ``first`` is a ``partials=True`` kernel storing each block's sum in its own slot; while more
    than one partial remains, ``SUM1D_PARTIALS`` (or its ``wp.vec3`` sibling) folds them a block at
    a time. Every stage is a fixed tree, where one ``atomic_add`` per block would add the blocks in
    arrival order and move the total's last bits between runs on CUDA. One launch up to
    ``ITEMS_PER_BLOCK_1D`` elements, one more per factor of it after; integer sums keep the atomic.
    ``inputs`` holds the arrays ``first`` takes before its output.
    """
    device = inputs[0].device
    fold = (
        kernel_reduce.sum_vec3_1d_partials
        if dtype == wp.vec3
        else kernel_reduce.SUM1D_PARTIALS[dtype]
    )
    kernel = first
    blocks = kernel_reduce.blocks_1d(n)
    while True:
        partials = _launch.empty(blocks, dtype=dtype, device=device)
        _launch.launch_tiled(
            kernel, dim=[blocks], inputs=[*inputs, partials], block_dim=TILE_1D, device=device
        )
        if blocks == 1:
            return partials
        kernel, inputs = fold, [partials]
        blocks = kernel_reduce.blocks_1d(blocks)


def _launch_global_vec3_minmax(array: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """Component-wise corner pair of a ``wp.vec3`` array: one launch, one buffer, one readback."""
    if int(array.ndim) != 1:
        raise ValueError("minmax requires a rank-1 wp.vec3 array.")
    n = array.size
    if n == 0:
        raise ValueError("minmax requires a non-empty array.")
    corners = _launch.full(6, math.inf, dtype=wp.float32, device=array.device)
    _launch.launch(
        kernel_reduce.minmax_vec3_chunked,
        dim=kernel_reduce.chunks_1d(n),
        inputs=[array, corners],
        device=array.device,
    )
    corners_np = corners.numpy()
    # Slots 3..5 hold the *negated* upper corner; see the kernel.
    return wp.vec3(*corners_np[:3]), wp.vec3(*(-corners_np[3:]))


def _reduce_scalar(
    array: twt.ArrayNdScalar, axis: Literal[0, 1] | None, spec: _ScalarReduceSpec
) -> (
    float
    | int
    | twt.Array1dScalar
    | tuple[twt.Array1dScalar, twt.Array1dScalar]
    | tuple[float, float]
    | tuple[int, int]
):
    _validate_scalar_array(array, spec, axis)

    if array.ndim == 2 and axis is not None:
        return _launch_axis_scalar(cast("twt.Array2dScalar", array), axis, spec)

    return _launch_global_scalar_tiled(array, spec)


def _launch_global_scalar_tiled(
    array: twt.ArrayNdScalar, spec: _ScalarReduceSpec
) -> float | int | tuple[float, float] | tuple[int, int]:
    flat = _flattened_for_global(array)
    if flat is not None and spec.name == "sum" and array.dtype in (wp.float32, wp.float64):
        total = _sum_in_fixed_order(
            kernel_reduce.SUM1D_PARTIALS[array.dtype], [flat], flat.size, array.dtype
        )
        return float(read_scalar(total, 0))
    if spec.global_output_slots == 2:
        out = _launch.array(
            [twt.dtype_max(array.dtype), twt.dtype_min(array.dtype)],
            dtype=array.dtype,
            device=array.device,
        )
    else:
        out = _launch.full(1, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)

    if flat is not None:
        _launch.launch_tiled(
            spec.tiled_1d[array.dtype],
            dim=[kernel_reduce.blocks_1d(flat.size)],
            inputs=[flat, out],
            block_dim=TILE_1D,
            device=array.device,
        )
    else:
        n, m = array.shape
        n_tiles = (n + TILE_2D - 1) // TILE_2D
        m_tiles = (m + TILE_2D - 1) // TILE_2D
        _launch.launch_tiled(
            spec.tiled_2d[array.dtype],
            dim=[n_tiles, m_tiles],
            inputs=[array, out],
            block_dim=TILE_2D * TILE_2D,
            device=array.device,
        )

    # ``.item()``, not ``read_scalar`` -- ``out`` is already a single-slot array, so there is no
    # whole-array copy to avoid, and unlike ``read_scalar`` (which hands back a numpy scalar; see
    # its own docstring on wrapping with ``int()``/``float()``), ``.item()`` converts to the
    # correct native Python type for whichever of ``int``/``float`` ``array.dtype`` is.
    out_np = out.numpy()
    if spec.global_output_slots == 2:
        return cast(tuple[int, int] | tuple[float, float], (out_np[0].item(), out_np[1].item()))
    return cast(float | int, out_np.item())


def _flattened_for_global(array: twt.ArrayNdScalar) -> twt.Array1dScalar | None:
    """
    Return a 1-D view when the 1-D kernel is the better way to reduce ``array``, else ``None``.

    Rank-1 input is already the 1-D case. **Any contiguous** rank-2 input flattens too, regardless
    of its trailing extent: flattening costs nothing on a contiguous buffer (it is a reshape, not a
    copy), and the 1-D kernel carries the ``TILES_PER_BLOCK_1D`` fold that the rank-2 tiled kernel
    does not — one atomic per ``ITEMS_PER_BLOCK_1D`` (1024) elements against the rank-2 kernel's one
    atomic per ``TILE_2D**2`` (64), which serializes badly on a wide table (a ``(1_000_000, 16)``
    array puts 250 000 blocks on one accumulator address through the unfolded rank-2 path). This
    also fixes the narrow case ``_launch_axis_scalar`` documents on its own axis: a table narrower
    than ``TILE_2D`` clips every rank-2 tile, so its ``wp.tile_load`` branch never runs and all
    ``TILE_2D**2`` lanes redundantly walk the same short block. Flattening is a wash only at the
    small end, where too few blocks are produced to fill the device either way — invisible next to
    the fixed host cost every scalar-returning reduction pays regardless. The bool reductions
    (``reduce.any``/``reduce.all``) already flatten unconditionally for exactly this reason; this
    makes the scalar family consistent with that precedent rather than a narrower special case of
    it.

    ``wp.array.flatten()`` raises on a non-contiguous array rather than copying, so a strided view
    (a column slice, a transpose) keeps the rank-2 kernel — correctness first, and such a view is
    not the common case; ``kernels/reduce.py``'s ``_reduce_2d_tiled``/``_minmax_2d_tiled`` exist
    for it.

    Parameters
    ----------
    array
        Rank-1 or rank-2 scalar array being reduced with ``axis=None``.

    Returns
    -------
    wp.array | None
        A 1-D view to reduce with the 1-D kernel, or ``None`` to use the rank-2 kernel.
    """
    if array.ndim == 1:
        return cast(twt.Array1dScalar, array)
    if array.is_contiguous:
        return cast(twt.Array1dScalar, array.flatten())
    return None


def _launch_axis_scalar(
    array: twt.Array2dScalar, axis: Literal[0, 1], spec: _ScalarReduceSpec
) -> twt.Array1dScalar | tuple[twt.Array1dScalar, twt.Array1dScalar]:
    n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
    n_out, reduced = (n_rows, n_cols) if axis == 1 else (n_cols, n_rows)

    if reduced < TILE_1D:
        # The tiled kernels never take their tile_load branch below TILE_1D: every lane of every
        # block redundantly runs the serial remainder, a TILE_1D-fold read amplification. One plain
        # thread per output, direct write, no init fill, avoids that.
        serial = (spec.axis_rows_serial if axis == 1 else spec.axis_cols_serial)[array.dtype]
        if spec.dual_axis:
            # One ``(2, n_out)`` buffer rather than two: every slot is written, and the pair
            # stays adjacent so a caller reading both can do it in one copy.
            packed = twt.empty_2d((2, n_out), array.dtype, device=array.device)
            out_min, out_max = packed[0], packed[1]
            _launch.launch(serial, dim=n_out, inputs=[array, out_min, out_max], device=array.device)
            return cast(twt.Array1dScalar, out_min), cast(twt.Array1dScalar, out_max)
        out = twt.empty_1d(n_out, array.dtype, device=array.device)
        _launch.launch(serial, dim=n_out, inputs=[array, out], device=array.device)
        return out

    tiled = (spec.axis_rows_tiled if axis == 1 else spec.axis_cols_tiled)[array.dtype]
    n_tiles = (reduced + TILE_1D - 1) // TILE_1D
    if spec.dual_axis:
        # Both seeds in one allocation, uploaded together: row 0 accumulates the minimum from
        # the dtype's largest value, row 1 the maximum from its smallest.
        seeds = np.empty((2, n_out), dtype=wp.dtype_to_numpy(array.dtype))
        seeds[0] = twt.dtype_max(array.dtype)
        seeds[1] = twt.dtype_min(array.dtype)
        packed = _launch.array(seeds, dtype=array.dtype, device=array.device)
        out_min, out_max = packed[0], packed[1]
        _launch.launch_tiled(
            tiled,
            dim=[n_out, n_tiles],
            inputs=[array, out_min, out_max],
            block_dim=TILE_1D,
            device=array.device,
        )
        return cast(twt.Array1dScalar, out_min), cast(twt.Array1dScalar, out_max)

    out = _launch.full(n_out, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)
    _launch.launch_tiled(
        tiled, dim=[n_out, n_tiles], inputs=[array, out], block_dim=TILE_1D, device=array.device
    )
    return cast(twt.Array1dScalar, out)


def _validate_scalar_array(
    array: twt.ArrayNdScalar, spec: _ScalarReduceSpec, axis: Literal[0, 1] | None
) -> None:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")
    if array.ndim == 1 and axis is not None:
        raise ValueError(f"{spec.name} requires axis=None for a 1D array.")
    if array.ndim not in (1, 2):
        raise ValueError(f"{spec.name} requires a 1D or 2D array.")
    if array.ndim == 2 and axis is not None and axis not in (0, 1):
        raise ValueError(f"{spec.name} requires axis to be 0, 1, or None for a 2D array.")


def _reduce_bool(
    array: wp.array[wp.bool], axis: Literal[0, 1] | None, spec: _BoolReduceSpec
) -> wp.array[wp.bool] | bool:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")

    if array.ndim == 1:
        return _launch_global_bool_tiled(array, spec)

    if array.ndim == 2:
        n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
        if axis is None:
            return _launch_global_bool_tiled(array.flatten(), spec)
        if axis not in (0, 1):
            raise ValueError(f"{spec.name} requires axis to be 0, 1, or None for a 2D array.")
        # The *axis* reductions still widen the mask first. They return a per-row or per-column
        # array rather than one scalar, so their answer is an ``int32`` buffer that has to be cast
        # back to ``bool`` on the way out anyway, and the kernels behind them are the shared
        # ``_reduce_2d_axis_*`` factories -- giving those a bool-input twin would double a family
        # of four to save one of the two casts. The whole-array path above, which is every in-repo
        # caller, reads the mask directly.
        mask_i32 = astype(array, wp.int32)
        n_out, reduced = (n_rows, n_cols) if axis == 1 else (n_cols, n_rows)
        if reduced < TILE_1D:
            # Same rule as _launch_axis_scalar: below TILE_1D the tiled form only amplifies reads.
            serial = spec.axis_rows_serial if axis == 1 else spec.axis_cols_serial
            out_i32 = _launch.empty(n_out, dtype=wp.int32, device=array.device)
            _launch.launch(serial, dim=n_out, inputs=[mask_i32, out_i32], device=array.device)
        else:
            tiled = spec.axis_rows_tiled if axis == 1 else spec.axis_cols_tiled
            n_tiles = (reduced + TILE_1D - 1) // TILE_1D
            out_i32 = _launch.full(n_out, spec.init_global, dtype=wp.int32, device=array.device)
            _launch.launch_tiled(
                tiled,
                dim=[n_out, n_tiles],
                inputs=[mask_i32, out_i32],
                block_dim=TILE_1D,
                device=array.device,
            )
        out = astype(out_i32, wp.bool)
        return out

    raise ValueError(f"{spec.name} requires a 1D or 2D array.")


def _launch_global_bool_tiled(mask: wp.array[wp.bool], spec: _BoolReduceSpec) -> bool:
    # The mask is read as ``wp.bool`` rather than widened to ``int32`` first. ``wp.Scalar`` does
    # not instantiate for ``wp.bool`` (CLAUDE.md section 12.4), so the shared scalar factories
    # cannot serve one, and the ``array.astype`` alternative is an allocation of ``4n`` bytes, a
    # launch, a full read of ``n`` and a full write of ``4n``, after which the reduction reads
    # ``4n`` rather than ``n``. See ``kernels.reduce._reduce_bool_1d_tiled`` for the concrete bool
    # kernel and why its block shape differs from the tile-load family's.
    out = _launch.full(1, spec.init_global, dtype=wp.int32, device=mask.device)
    n = mask.size
    _launch.launch_tiled(
        spec.bool_1d,
        dim=[kernel_reduce.blocks_1d(n)],
        inputs=[mask, out],
        block_dim=TILE_1D,
        device=mask.device,
    )
    return bool(int(read_scalar(out, 0)) != 0)
