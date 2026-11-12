"""Global reductions on Warp arrays."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, TypeVar, cast, overload

import numpy as np
import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._common import read_scalar, read_values, require_same_device
from ordito.array import _sorted_copy, astype  # pyright: ignore[reportPrivateUsage]
from ordito.config import is_deterministic
from ordito.constants import TILE_1D, TILE_2D, UINT64_MAX
from ordito.kernels import array as kernel_array
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import scatter as kernel_scatter

DType = TypeVar("DType")
"""Element type of a per-group reduction: a scalar, or a ``wp.float32`` / ``wp.float64`` vector."""


# The global overloads take the rank-agnostic ``ArrayNd*`` aliases so a caller holding this
# package's usual ``wp.array[dtype]`` spelling resolves: ``NDim`` is invariant, so pinning the
# rank here rejects it (ordito.typing, "accept wide, return narrow"). The integer arm spans
# ``kernel_reduce._GLOBAL_DTYPES``, which is wider than int32. The ``axis`` overload stays on
# ``Array2dScalar`` deliberately -- an axis reduction is rank-2 only and ``_AXIS_DTYPES`` is only
# the three narrow dtypes, so the rank and the dtype are both real constraints there.
@overload
def min(array: odt.ArrayNdInt, *, axis: None = ...) -> int: ...
@overload
def min(array: odt.ArrayNdFloat, *, axis: None = ...) -> float: ...
@overload
def min(array: odt.Array2dScalar, *, axis: Literal[0, 1]) -> odt.Array1dScalar: ...
def min(
    array: odt.ArrayNdScalar, *, axis: Literal[0, 1] | None = None
) -> float | int | odt.Array1dScalar:
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

    Examples
    --------
    Like [`numpy.min`][]: a Python scalar for the whole array, an array along an axis:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([[3, 1], [4, 1], [5, 9]], dtype=wp.int32)
    print(od.reduce.min(values))  # 1
    print(od.reduce.min(values, axis=0).numpy().tolist())  # [3, 1]
    print(od.reduce.min(values, axis=1).numpy().tolist())  # [1, 1, 5]
    ```
    """
    return cast(float | int | odt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["min"]))


@overload
def max(array: odt.ArrayNdInt, *, axis: None = ...) -> int: ...
@overload
def max(array: odt.ArrayNdFloat, *, axis: None = ...) -> float: ...
@overload
def max(array: odt.Array2dScalar, *, axis: Literal[0, 1]) -> odt.Array1dScalar: ...
def max(
    array: odt.ArrayNdScalar, *, axis: Literal[0, 1] | None = None
) -> float | int | odt.Array1dScalar:
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

    Examples
    --------
    Like [`numpy.max`][]: a Python scalar for the whole array, an array along an axis:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([[3.0, 1.0], [4.0, 1.0], [5.0, 9.0]], dtype=wp.float32)
    print(od.reduce.max(values))  # 9.0
    print(od.reduce.max(values, axis=0).numpy().tolist())  # [5.0, 9.0]
    print(od.reduce.max(values, axis=1).numpy().tolist())  # [3.0, 4.0, 9.0]
    ```
    """
    return cast(float | int | odt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["max"]))


@overload
def minmax(array: odt.ArrayNdInt, *, axis: None = ...) -> tuple[int, int]: ...
@overload
def minmax(array: odt.ArrayNdFloat, *, axis: None = ...) -> tuple[float, float]: ...
@overload
def minmax(array: wp.array[wp.vec3], *, axis: None = ...) -> tuple[wp.vec3, wp.vec3]: ...
@overload
def minmax(
    array: odt.Array2dScalar, *, axis: Literal[0, 1]
) -> tuple[odt.Array1dScalar, odt.Array1dScalar]: ...
def minmax(
    array: odt.ArrayNdScalar | wp.array[wp.vec3], *, axis: Literal[0, 1] | None = None
) -> (
    tuple[float, float]
    | tuple[int, int]
    | tuple[wp.vec3, wp.vec3]
    | tuple[odt.Array1dScalar, odt.Array1dScalar]
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
    [`aabb`][ordito.bounds.aabb] host-latency-bound and nothing more.

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
    [`aabb`][ordito.bounds.aabb]
        The mesh-facing spelling of the ``wp.vec3`` reduction, with the empty-input
        ``(+inf, -inf)`` convention instead of a raise.

    Examples
    --------
    Both extremes from one pass; a ``wp.vec3`` array gives its bounding-box corners:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([3, -1, 4, 1, 5], dtype=wp.int32)
    print(od.reduce.minmax(values))  # (-1, 5)
    points = wp.array([[0.0, 1.0, 2.0], [3.0, -1.0, 5.0]], dtype=wp.vec3)
    low, high = od.reduce.minmax(points)
    print(low, high)  # [0.0, -1.0, 2.0] [3.0, 1.0, 5.0]
    ```
    """
    if array.dtype == wp.vec3:
        if axis is not None:
            raise ValueError("minmax requires axis=None for a wp.vec3 array.")
        return _launch_global_vec3_minmax(cast("wp.array[wp.vec3]", array))
    return cast(
        tuple[float, float] | tuple[int, int] | tuple[odt.Array1dScalar, odt.Array1dScalar],
        _reduce_scalar(cast(odt.ScalarArray, array), axis, _SCALAR_REDUCE["minmax"]),
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

    Examples
    --------
    Like [`numpy.any`][]:

    ```python
    import warp as wp

    import ordito as od

    mask = wp.array([[False, False], [False, True]], dtype=wp.bool)
    print(od.reduce.any(mask))  # True
    print(od.reduce.any(mask, axis=1).numpy().tolist())  # [False, True]
    ```
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["any"])


@overload
def sum(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def sum(array: wp.array[wp.vec4], *, axis: None = ...) -> wp.vec4: ...
@overload
def sum(array: odt.ArrayNdInt, *, axis: None = ...) -> int: ...
@overload
def sum(array: odt.ArrayNdFloat, *, axis: None = ...) -> float: ...
@overload
def sum(array: odt.Array2dScalar, *, axis: Literal[0, 1]) -> odt.Array1dScalar: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: None = ...) -> int: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: Literal[0, 1]) -> odt.Array1dInt32: ...
def sum(
    array: odt.ArrayNdScalar | wp.array[wp.bool] | wp.array[wp.vec3] | wp.array[wp.vec4],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | int | odt.Array1dScalar | odt.Array1dInt32 | wp.vec3 | wp.vec4:
    """
    Sum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_sum`` + ``wp.atomic_add``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    For ``wp.bool`` input, counts ``True`` values and returns ``int`` (global) or
    ``wp.int32`` (per-axis). A rank-1 ``wp.vec3`` or ``wp.vec4`` array sums componentwise to one
    vector of its dtype (``axis=None`` only).

    Parameters
    ----------
    array
        ``(n,)`` or ``(n, m)`` scalar or ``wp.bool`` array, or ``(n,)`` ``wp.vec3`` / ``wp.vec4``
        array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array | wp.vec3 | wp.vec4
        Global scalar (vector, for a vector array) when ``axis=None``; ``(n,)`` (``axis=1``) or
        ``(m,)`` (``axis=0``) array otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 input, or ``axis`` is not ``0``, ``1``, or ``None`` for a rank-2 input.

    Examples
    --------
    Like [`numpy.sum`][]; a mask sums to its ``True`` count and a ``wp.vec3`` array to a vector:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([[1, 2], [3, 4], [5, 6]], dtype=wp.int32)
    print(od.reduce.sum(values))  # 21
    print(od.reduce.sum(values, axis=0).numpy().tolist())  # [9, 12]
    print(od.reduce.sum(wp.array([True, False, True], dtype=wp.bool)))  # 2
    points = wp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5]], dtype=wp.vec3)
    print(od.reduce.sum(points))  # [1.0, 2.0, 0.5]
    ```
    """
    if array.dtype in kernel_reduce.VECTOR_SUMS:
        name = array.dtype.__name__
        if axis is not None:
            raise ValueError(f"sum over a {name} array supports only axis=None.")
        if array.ndim != 1:
            raise ValueError(f"sum over a {name} array requires a 1D array.")
        n = int(array.shape[0])
        if n == 0:
            raise ValueError("sum requires a non-empty array.")
        dtype = array.dtype
        total = _sum_of_terms_buffer(
            kernel_reduce.VECTOR_SUMS[dtype], [array], n, device=array.device
        )
        return dtype(*read_scalar(total, 0))
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
            cast(odt.ScalarArray, astype(mask, wp.int32)), axis, _SCALAR_REDUCE["sum"]
        )
        return cast(odt.Array1dInt32, result)
    return cast(
        float | int | odt.Array1dScalar,
        _reduce_scalar(cast(odt.ScalarArray, array), axis, _SCALAR_REDUCE["sum"]),
    )


@overload
def mean(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def mean(array: odt.ArrayNdScalar | wp.array[wp.bool], *, axis: None = ...) -> float: ...
@overload
def mean(
    array: odt.Array2dScalar | wp.array[wp.bool], *, axis: Literal[0, 1]
) -> odt.Array1dFloat32: ...
def mean(
    array: odt.ArrayNdScalar | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | odt.Array1dFloat32 | wp.vec3:
    """
    Arithmetic mean of ``array``.

    Delegates the reduction to the tiled [`sum`][ordito.reduce.sum], then divides by the
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

    Examples
    --------
    Like [`numpy.mean`][]; the mean of a mask is the fraction of ``True`` entries:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([[1, 2], [3, 5]], dtype=wp.int32)
    print(od.reduce.mean(values))  # 2.75
    print(od.reduce.mean(values, axis=1).numpy().tolist())  # [1.5, 4.0]
    print(od.reduce.mean(wp.array([True, False, True, True], dtype=wp.bool)))  # 0.75
    ```
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
        total = sum(cast(odt.ArrayNdScalar, array))
        return float(total) / array.size
    sums = sum(cast(odt.Array2dScalar, array), axis=axis)
    # ``sum`` returns a fresh array, so a ``float32`` one is divided in place rather than copied.
    out = sums if sums.dtype == wp.float32 else astype(sums, wp.float32)
    _launch.map(wp.div, out, wp.float32(array.shape[axis]), out=out)
    return cast(odt.Array1dFloat32, out)


@overload
def weighted_sum(values: odt.ArrayNdFloat32, weights: odt.ArrayNdFloat32) -> float: ...
@overload
def weighted_sum(values: wp.array[wp.vec3], weights: odt.ArrayNdFloat32) -> wp.vec3: ...
def weighted_sum(
    values: odt.ArrayNdFloat32 | wp.array[wp.vec3], weights: odt.ArrayNdFloat32
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

    Examples
    --------
    ``numpy.dot(weights, values)``, for scalars or ``wp.vec3`` rows:

    ```python
    import warp as wp

    import ordito as od

    weights = wp.array([0.5, 0.25, 0.25], dtype=wp.float32)
    values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
    print(od.reduce.weighted_sum(values, weights))  # 1.75
    points = wp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=wp.vec3)
    print(od.reduce.weighted_sum(points, weights))  # [1.0, 1.0, 0.0]
    ```
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
        total = _sum_of_terms_buffer(kernel_reduce.WEIGHTED_SUM_VEC3, [values, weights], n_values)
        return wp.vec3(*read_scalar(total, 0))
    total = _sum_of_terms_buffer(kernel_reduce.WEIGHTED_SUM, [values, weights], n_values)
    return float(read_scalar(total, 0))


def sum_of_terms(
    kernels: kernel_reduce.TermSum,
    inputs: Sequence[object],
    n: int,
    *,
    out: wp.array[Any] | None = None,
    device: wp.DeviceLike = None,
) -> wp.array[Any]:
    """
    Total of ``n`` terms a block-sum kernel pair forms, left on the device.

    The device-side form of [`sum`][ordito.reduce.sum]: the same blocks and determinism rule, but
    the kernel is any ``ordito.kernels.reduce.TermSum`` -- ``sum``'s own over an array
    (``kernel_reduce.ARRAY_SUM``), the weighted sum's (``kernel_reduce.WEIGHTED_SUM``), or a fused
    term built by ``ordito.kernels.reduce.term_sum_1d`` that never stores its terms -- and the total
    stays in a one-slot device array instead of being read back. The terms are ``float32``, or the
    ``dtype`` the kernel was built with (a ``wp.vec2`` of two sums formed in one pass). Under
    [`is_deterministic`][ordito.config.is_deterministic] every block stores its sum and the last
    block to finish adds them in a fixed tree (a second launch adds them past a few million
    terms); otherwise every block adds its sum into the total, in one launch.

    Parameters
    ----------
    kernels
        The first-stage kernel pair.
    inputs
        What the kernels take before their output, in order.
    n
        Number of terms; sets the block count. Must be positive.
    out
        ``(1,)`` buffer to receive the total, such as one slot of a larger buffer that is read
        back once. It must hold zero: the sum may be added into it. ``None`` allocates one.
    device
        Device to launch on. ``None`` takes ``out``'s, else that of ``inputs[0]``, which must then
        be an array (a ``@wp.struct`` input carries no device).

    Returns
    -------
    wp.array
        ``(1,)`` the total: ``out`` when given.

    See Also
    --------
    [`sum`][ordito.reduce.sum]
        The same total of an array, read back to the host.

    Examples
    --------
    The weighted sum, kept on the device:

    ```python
    import warp as wp

    import ordito as od
    from ordito.kernels import reduce as kernel_reduce

    values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
    total = od.reduce.sum_of_terms(kernel_reduce.WEIGHTED_SUM, [values, values], values.size)
    print(total.numpy().tolist())  # [14.0]
    ```
    """
    total = _sum_of_terms_buffer(kernels, inputs, n, out=out, device=device)
    return total if total.size == 1 else _launch.view(total, 0, 1)


def scatter_sum(
    values: wp.array[DType], indices: wp.array[wp.int32], n_groups: int | None = None
) -> wp.array[DType]:
    """
    Per-group sums: ``out[g]`` is the sum of the ``values[i]`` whose ``indices[i] == g``.

    ``numpy.bincount(indices, weights=values, minlength=n_groups)`` for scalars, and
    ``numpy.add.at(out, indices, values)`` into zeros for vectors, componentwise. Under
    [`is_deterministic`][ordito.config.is_deterministic] every group is summed exactly in fixed
    point and rounded once, so the result repeats bit for bit from call to call and is the same on
    the CPU and the CUDA device; otherwise the values are added in arrival order, which on the CUDA
    device can move a group's last bits from one call to the next. A group holding a non-finite
    value sums to NaN under the setting.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.float32``, ``wp.float64``, or a ``wp.vec2`` / ``wp.vec3`` /
        ``wp.vec4`` vector of either precision.
    indices
        ``(n,)`` ``wp.int32`` group of each value, in ``[0, n_groups)``.
    n_groups
        Number of groups. ``None`` takes ``max(indices) + 1`` (0 for no values).

    Returns
    -------
    wp.array
        ``(n_groups,)`` sums, of ``values``' dtype; a group no index names sums to zero.

    Raises
    ------
    ValueError
        If ``values`` or ``indices`` is not rank-1, their lengths differ, ``n_groups`` is
        negative, or an index lies outside ``[0, n_groups)``.
    TypeError
        If ``values`` has another dtype, or ``indices`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``indices`` are not on one device.

    See Also
    --------
    [`segment_sum`][ordito.reduce.segment_sum]
        The same sums over contiguous segments given by offsets.

    Examples
    --------
    Like [`numpy.bincount`][] with weights:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
    groups = wp.array([0, 2, 0, 2], dtype=wp.int32)
    print(od.reduce.scatter_sum(values, groups).numpy().tolist())  # [4.0, 0.0, 6.0]
    print(od.reduce.scatter_sum(values, groups, 4).numpy().tolist())  # [4.0, 0.0, 6.0, 0.0]
    ```
    """
    require_same_device(values=values, indices=indices)
    return _scatter_sum(values, indices, n_groups, "scatter_sum")


def segment_sum(values: wp.array[DType], offsets: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Per-segment sums of ``values`` over the contiguous segments ``offsets`` delimits.

    Segment ``s`` is ``values[offsets[s]:offsets[s + 1]]``, so ``offsets`` is the
    ``n_segments + 1`` total-terminated form (``offsets[0] == 0``, non-decreasing, ending at
    ``n``); an empty segment sums to zero, where ``numpy.add.reduceat`` would repeat a value.
    Obeys [`is_deterministic`][ordito.config.is_deterministic] as
    [`scatter_sum`][ordito.reduce.scatter_sum] does.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.float32``, ``wp.float64``, or a ``wp.vec2`` / ``wp.vec3`` /
        ``wp.vec4`` vector of either precision.
    offsets
        ``(n_segments + 1,)`` ``wp.int32`` segment starts, closed by ``n``.

    Returns
    -------
    wp.array
        ``(n_segments,)`` sums, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``offsets`` is not rank-1, ``offsets`` is empty, or its last entry is not
        ``n``.
    TypeError
        If ``values`` has another dtype, or ``offsets`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``offsets`` are not on one device.

    See Also
    --------
    [`scatter_sum`][ordito.reduce.scatter_sum]
        The same sums for groups in any order.

    Examples
    --------
    Three segments of a packed buffer, the middle one empty:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float64)
    offsets = wp.array([0, 3, 3, 4], dtype=wp.int32)
    print(od.reduce.segment_sum(values, offsets).numpy().tolist())  # [6.0, 0.0, 4.0]
    ```
    """
    require_same_device(values=values, offsets=offsets)
    return _segment_sum(values, offsets, "segment_sum")


def scatter_mean(
    values: wp.array[DType], indices: wp.array[wp.int32], n_groups: int | None = None
) -> wp.array[DType]:
    """
    Per-group means: ``out[g]`` is the mean of the ``values[i]`` whose ``indices[i] == g``.

    [`scatter_sum`][ordito.reduce.scatter_sum] divided by each group's count, componentwise for
    vectors, so it obeys [`is_deterministic`][ordito.config.is_deterministic] as the sum does. A
    group no index names is ``0 / 0``, a NaN, as [`numpy.mean`][] of an empty slice.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.float32``, ``wp.float64``, or a ``wp.vec2`` / ``wp.vec3`` /
        ``wp.vec4`` vector of either precision.
    indices
        ``(n,)`` ``wp.int32`` group of each value, in ``[0, n_groups)``.
    n_groups
        Number of groups. ``None`` takes ``max(indices) + 1`` (0 for no values).

    Returns
    -------
    wp.array
        ``(n_groups,)`` means, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``indices`` is not rank-1, their lengths differ, ``n_groups`` is
        negative, or an index lies outside ``[0, n_groups)``.
    TypeError
        If ``values`` has another dtype, or ``indices`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``indices`` are not on one device.

    See Also
    --------
    [`segment_mean`][ordito.reduce.segment_mean]
        The same means over contiguous segments given by offsets.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    values = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
    groups = wp.array([0, 2, 0, 2], dtype=wp.int32)
    print(od.reduce.scatter_mean(values, groups).numpy().tolist())  # [2.0, nan, 3.0]
    ```
    """
    require_same_device(values=values, indices=indices)
    sums = _scatter_sum(values, indices, n_groups, "scatter_mean")
    counts = _launch.zeros(sums.size, dtype=wp.int32, device=values.device)
    if values.size > 0:
        _launch.launch(
            kernel_scatter.count_occurrences,
            dim=values.size,
            inputs=[indices, counts],
            device=values.device,
        )
    return _divided_by_counts(sums, counts)


def segment_mean(values: wp.array[DType], offsets: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Per-segment means of ``values`` over the contiguous segments ``offsets`` delimits.

    [`segment_sum`][ordito.reduce.segment_sum] divided by each segment's length, componentwise
    for vectors, so it obeys [`is_deterministic`][ordito.config.is_deterministic] as the sum does.
    An empty segment is ``0 / 0``, a NaN, as [`numpy.mean`][] of an empty slice.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.float32``, ``wp.float64``, or a ``wp.vec2`` / ``wp.vec3`` /
        ``wp.vec4`` vector of either precision.
    offsets
        ``(n_segments + 1,)`` ``wp.int32`` segment starts, closed by ``n``.

    Returns
    -------
    wp.array
        ``(n_segments,)`` means, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``offsets`` is not rank-1, ``offsets`` is empty, or its last entry is not
        ``n``.
    TypeError
        If ``values`` has another dtype, or ``offsets`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``offsets`` are not on one device.

    See Also
    --------
    [`scatter_mean`][ordito.reduce.scatter_mean]
        The same means for groups in any order.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    values = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float64)
    offsets = wp.array([0, 3, 3, 4], dtype=wp.int32)
    print(od.reduce.segment_mean(values, offsets).numpy().tolist())  # [2.0, nan, 4.0]
    ```
    """
    require_same_device(values=values, offsets=offsets)
    sums = _segment_sum(values, offsets, "segment_mean")
    n_segments = sums.size
    if n_segments > 0:
        # Each segment's length read off its two offsets inside the division.
        mean_of_segment = (
            kernel_scatter.scalar_mean_of_segment
            if sums.dtype in (wp.float32, wp.float64)
            else kernel_scatter.vector_mean_of_segment
        )
        ends = _launch.view(offsets, 1, n_segments + 1)
        starts = _launch.view(offsets, 0, n_segments)
        wp.map(mean_of_segment, sums, ends, starts, out=sums)
    return sums


def scatter_min(
    values: wp.array[DType], indices: wp.array[wp.int32], n_groups: int | None = None
) -> wp.array[DType]:
    """
    Per-group minima: ``out[g]`` is the least of the ``values[i]`` whose ``indices[i] == g``.

    ``numpy.minimum.at(out, indices, values)`` into ``out`` filled with the dtype's largest value,
    componentwise for vectors: a group no index names holds ``inf`` (the largest integer for an
    integer dtype). A minimum does not depend on the order its values arrive in, so the result
    repeats bit for bit from call to call whatever the
    [`is_deterministic`][ordito.config.is_deterministic] setting. ``values`` must hold no NaN.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.int32``, ``wp.int64``, ``wp.float32``, ``wp.float64``, or a
        ``wp.vec2`` / ``wp.vec3`` / ``wp.vec4`` vector of either float precision.
    indices
        ``(n,)`` ``wp.int32`` group of each value, in ``[0, n_groups)``.
    n_groups
        Number of groups. ``None`` takes ``max(indices) + 1`` (0 for no values).

    Returns
    -------
    wp.array
        ``(n_groups,)`` minima, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``indices`` is not rank-1, their lengths differ, ``n_groups`` is
        negative, or an index lies outside ``[0, n_groups)``.
    TypeError
        If ``values`` has another dtype, or ``indices`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``indices`` are not on one device.

    See Also
    --------
    [`scatter_max`][ordito.reduce.scatter_max]
        The per-group maxima.
    [`segment_min`][ordito.reduce.segment_min]
        The same minima over contiguous segments given by offsets.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    values = wp.array([3, 1, 4, 1, 5], dtype=wp.int32)
    groups = wp.array([0, 1, 0, 1, 3], dtype=wp.int32)
    print(od.reduce.scatter_min(values, groups).numpy().tolist())  # [3, 1, 2147483647, 5]
    ```
    """
    require_same_device(values=values, indices=indices)
    return _scatter_extrema(
        values, indices, n_groups, int(kernel_scatter.EXTREMUM_MIN), "scatter_min"
    )


def scatter_max(
    values: wp.array[DType], indices: wp.array[wp.int32], n_groups: int | None = None
) -> wp.array[DType]:
    """
    Per-group maxima: ``out[g]`` is the greatest of the ``values[i]`` whose ``indices[i] == g``.

    ``numpy.maximum.at(out, indices, values)`` into ``out`` filled with the dtype's least value,
    componentwise for vectors: a group no index names holds ``-inf`` (the least integer for an
    integer dtype). Repeats bit for bit as [`scatter_min`][ordito.reduce.scatter_min] does.
    ``values`` must hold no NaN.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.int32``, ``wp.int64``, ``wp.float32``, ``wp.float64``, or a
        ``wp.vec2`` / ``wp.vec3`` / ``wp.vec4`` vector of either float precision.
    indices
        ``(n,)`` ``wp.int32`` group of each value, in ``[0, n_groups)``.
    n_groups
        Number of groups. ``None`` takes ``max(indices) + 1`` (0 for no values).

    Returns
    -------
    wp.array
        ``(n_groups,)`` maxima, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``indices`` is not rank-1, their lengths differ, ``n_groups`` is
        negative, or an index lies outside ``[0, n_groups)``.
    TypeError
        If ``values`` has another dtype, or ``indices`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``indices`` are not on one device.

    See Also
    --------
    [`scatter_min`][ordito.reduce.scatter_min]
        The per-group minima.
    [`segment_max`][ordito.reduce.segment_max]
        The same maxima over contiguous segments given by offsets.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    values = wp.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=wp.float32)
    groups = wp.array([0, 1, 0, 1, 3], dtype=wp.int32)
    print(od.reduce.scatter_max(values, groups).numpy().tolist())  # [4.0, 1.0, -inf, 5.0]
    ```
    """
    require_same_device(values=values, indices=indices)
    return _scatter_extrema(
        values, indices, n_groups, int(kernel_scatter.EXTREMUM_MAX), "scatter_max"
    )


def segment_min(values: wp.array[DType], offsets: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Per-segment minima of ``values`` over the contiguous segments ``offsets`` delimits.

    ``numpy.minimum.reduceat`` over the segments, componentwise for vectors, except that an empty
    segment holds ``inf`` (the largest integer for an integer dtype) where ``reduceat`` would
    repeat a value. Repeats bit for bit as [`scatter_min`][ordito.reduce.scatter_min] does.
    ``values`` must hold no NaN.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.int32``, ``wp.int64``, ``wp.float32``, ``wp.float64``, or a
        ``wp.vec2`` / ``wp.vec3`` / ``wp.vec4`` vector of either float precision.
    offsets
        ``(n_segments + 1,)`` ``wp.int32`` segment starts, closed by ``n``.

    Returns
    -------
    wp.array
        ``(n_segments,)`` minima, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``offsets`` is not rank-1, ``offsets`` is empty, or its last entry is not
        ``n``.
    TypeError
        If ``values`` has another dtype, or ``offsets`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``offsets`` are not on one device.

    See Also
    --------
    [`segment_max`][ordito.reduce.segment_max]
        The per-segment maxima.
    [`scatter_min`][ordito.reduce.scatter_min]
        The same minima for groups in any order.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    values = wp.array([3.0, 1.0, 4.0, 2.0], dtype=wp.float64)
    offsets = wp.array([0, 3, 3, 4], dtype=wp.int32)
    print(od.reduce.segment_min(values, offsets).numpy().tolist())  # [1.0, inf, 2.0]
    ```
    """
    require_same_device(values=values, offsets=offsets)
    return _segment_extrema(values, offsets, int(kernel_scatter.EXTREMUM_MIN), "segment_min")


def segment_max(values: wp.array[DType], offsets: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Per-segment maxima of ``values`` over the contiguous segments ``offsets`` delimits.

    ``numpy.maximum.reduceat`` over the segments, componentwise for vectors, except that an empty
    segment holds ``-inf`` (the least integer for an integer dtype) where ``reduceat`` would
    repeat a value. Repeats bit for bit as [`scatter_min`][ordito.reduce.scatter_min] does.
    ``values`` must hold no NaN.

    Parameters
    ----------
    values
        ``(n,)`` values: ``wp.int32``, ``wp.int64``, ``wp.float32``, ``wp.float64``, or a
        ``wp.vec2`` / ``wp.vec3`` / ``wp.vec4`` vector of either float precision.
    offsets
        ``(n_segments + 1,)`` ``wp.int32`` segment starts, closed by ``n``.

    Returns
    -------
    wp.array
        ``(n_segments,)`` maxima, of ``values``' dtype.

    Raises
    ------
    ValueError
        If ``values`` or ``offsets`` is not rank-1, ``offsets`` is empty, or its last entry is not
        ``n``.
    TypeError
        If ``values`` has another dtype, or ``offsets`` is not ``wp.int32``.
    RuntimeError
        If ``values`` and ``offsets`` are not on one device.

    See Also
    --------
    [`segment_min`][ordito.reduce.segment_min]
        The per-segment minima.
    [`scatter_max`][ordito.reduce.scatter_max]
        The same maxima for groups in any order.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    values = wp.array([3, 1, 4, 2], dtype=wp.int64)
    offsets = wp.array([0, 3, 4], dtype=wp.int32)
    print(od.reduce.segment_max(values, offsets).numpy().tolist())  # [4, 2]
    ```
    """
    require_same_device(values=values, offsets=offsets)
    return _segment_extrema(values, offsets, int(kernel_scatter.EXTREMUM_MAX), "segment_max")


def _scatter_sum(
    values: wp.array[DType], indices: wp.array[wp.int32], n_groups: int | None, name: str
) -> wp.array[DType]:
    """[`scatter_sum`][ordito.reduce.scatter_sum] for the public function ``name``."""
    kernels = _group_sum_kernels(values, name)
    n_groups = _scatter_group_count(values, indices, n_groups, name)
    pooling = kernel_scatter.scatter_sum_pooling(values.size, n_groups)
    return _scatter_groups(kernels[0], values, indices, n_groups, pooling)


def _segment_sum(
    values: wp.array[DType], offsets: wp.array[wp.int32], name: str
) -> wp.array[DType]:
    """[`segment_sum`][ordito.reduce.segment_sum] for the public function ``name``."""
    kernels = _group_sum_kernels(values, name)
    _check_segment_offsets(values, offsets, name)
    return _scatter_groups(kernels[1], values, offsets, offsets.size - 1, None)


def _divided_by_counts(sums: wp.array[DType], counts: wp.array[wp.int32]) -> wp.array[DType]:
    """Return ``sums`` divided in place by ``counts``, componentwise for a vector dtype."""
    if sums.size > 0:
        mean_of_sum = (
            kernel_scatter.scalar_mean_of_sum
            if sums.dtype in (wp.float32, wp.float64)
            else kernel_scatter.vector_mean_of_sum
        )
        wp.map(mean_of_sum, sums, counts, out=sums)
    return sums


def _scatter_extrema(
    values: wp.array[DType],
    indices: wp.array[wp.int32],
    n_groups: int | None,
    maximum: int,
    name: str,
) -> wp.array[DType]:
    """Per-group extrema (``maximum``: ``kernels/scatter.EXTREMUM_*``) for the public ``name``."""
    _validate_extremum_values(values, name)
    n_groups = _scatter_group_count(values, indices, n_groups, name)
    extrema = _extremum_identities(values, n_groups, maximum)
    if values.size > 0:
        _launch.launch(
            kernel_scatter.SCATTER_EXTREMA[values.dtype],
            dim=values.size,
            inputs=[values, indices, maximum, extrema],
            device=values.device,
        )
    return extrema


def _segment_extrema(
    values: wp.array[DType], offsets: wp.array[wp.int32], maximum: int, name: str
) -> wp.array[DType]:
    """Per-segment extrema (``maximum``: ``kernels/scatter.EXTREMUM_*``) for the public ``name``."""
    _validate_extremum_values(values, name)
    _check_segment_offsets(values, offsets, name)
    extrema = _extremum_identities(values, offsets.size - 1, maximum)
    if values.size > 0:
        _launch.launch(
            kernel_scatter.SEGMENT_EXTREMA[values.dtype],
            dim=values.size,
            inputs=[values, offsets, maximum, extrema],
            device=values.device,
        )
    return extrema


def _validate_extremum_values(values: wp.array[Any], name: str) -> None:
    """Raise unless ``values`` is a rank-1 array of a dtype the extremum kernels take."""
    if values.ndim != 1:
        raise ValueError(f"{name} requires rank-1 values.")
    if values.dtype not in kernel_scatter.EXTREMUM_DTYPES:
        raise TypeError(
            f"{name} takes wp.int32, wp.int64, wp.float32, wp.float64 or a vec2/vec3/vec4 of "
            f"either float, not {values.dtype.__name__}."
        )


def _extremum_identities(values: wp.array[DType], n_groups: int, maximum: int) -> wp.array[DType]:
    """Return ``(n_groups,)`` copies of the extremum's identity: ``inf`` for a float min."""
    dtype = values.dtype
    scalar = getattr(dtype, "_wp_scalar_type_", dtype)
    if maximum == int(kernel_scatter.EXTREMUM_MAX):
        identity = odt.dtype_min(scalar)
    else:
        identity = odt.dtype_max(scalar)
    length = getattr(dtype, "_length_", 0)
    fill = dtype(*([identity] * length)) if length else identity
    return _launch.full(n_groups, fill, dtype=dtype, device=values.device)


def _group_sum_kernels(values: wp.array[Any], name: str) -> tuple[odt.Kernel, odt.Kernel]:
    """Return the scatter and segment kernels for ``values``' dtype, after validating it."""
    if values.ndim != 1:
        raise ValueError(f"{name} requires rank-1 values.")
    dtype = values.dtype
    if dtype in kernel_scatter.SCATTER_SUM_SCALARS:
        return kernel_scatter.SCATTER_SUM_SCALARS[dtype], kernel_scatter.SEGMENT_SUM_SCALARS[dtype]
    if dtype in kernel_scatter.SCATTER_SUM_VECTORS:
        return kernel_scatter.SCATTER_SUM_VECTORS[dtype], kernel_scatter.SEGMENT_SUM_VECTORS[dtype]
    raise TypeError(
        f"{name} sums wp.float32, wp.float64 or a vec2/vec3/vec4 of either, not {dtype.__name__}."
    )


def _scatter_group_count(
    values: wp.array[Any], indices: wp.array[wp.int32], n_groups: int | None, name: str
) -> int:
    """Return the group count of a scatter, after validating ``indices`` against ``values``."""
    if indices.ndim != 1:
        raise ValueError(f"{name} requires rank-1 indices.")
    if indices.dtype != wp.int32:
        raise TypeError(f"{name} requires wp.int32 indices, got {indices.dtype.__name__}.")
    n = values.size
    if indices.size != n:
        raise ValueError(f"{name} requires one index per value: {indices.size} for {n}.")
    if n_groups is not None and n_groups < 0:
        raise ValueError(f"{name} requires n_groups >= 0, got {n_groups}.")
    if n > 0:
        # One read of the index range: it sizes the result when ``n_groups`` is not given, and an
        # index outside it would scatter out of bounds.
        lowest, highest = minmax(indices)
        if n_groups is None:
            n_groups = highest + 1
        if lowest < 0 or highest >= n_groups:
            raise ValueError(f"{name} indices span [{lowest}, {highest}], outside [0, {n_groups}).")
    return n_groups or 0


def _check_segment_offsets(values: wp.array[Any], offsets: wp.array[wp.int32], name: str) -> None:
    """Raise unless ``offsets`` is a CSR packing of ``values``."""
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError(f"{name} requires non-empty rank-1 offsets.")
    if offsets.dtype != wp.int32:
        raise TypeError(f"{name} requires wp.int32 offsets, got {offsets.dtype.__name__}.")
    n = values.size
    # The closing offset, read back: a packing that does not end at ``n`` would assign the tail
    # of ``values`` to the wrong segment.
    total = int(read_scalar(offsets, -1))
    if total != n:
        raise ValueError(f"{name} offsets end at {total}, not at the {n} values.")


def _scatter_groups(
    kernel: odt.Kernel,
    values: wp.array[DType],
    groups: wp.array[wp.int32],
    n_groups: int,
    pooling: int | None,
) -> wp.array[DType]:
    """
    Each value into its group through ``kernel``, per the deterministic setting.

    ``pooling`` is the scatter kernel's lane pooling (``kernels/scatter.scatter_sum_pooling``);
    ``None`` for the segment kernel, which takes none.
    """
    device = values.device
    sums = _launch.group_sums(n_groups, values.dtype, values.size or 1, device)
    if values.size > 0 and n_groups > 0:
        inputs = [values, groups] if pooling is None else [values, groups, pooling]
        _launch.launch_group_sums(kernel, values.size, inputs, sums, device=device)
    return cast("wp.array[DType]", _launch.finish_group_sums(sums))


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

    Examples
    --------
    Like [`numpy.all`][]:

    ```python
    import warp as wp

    import ordito as od

    mask = wp.array([[True, True], [False, True]], dtype=wp.bool)
    print(od.reduce.all(mask))  # False
    print(od.reduce.all(mask, axis=0).numpy().tolist())  # [False, True]
    ```
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["all"])


def median(array: odt.ArrayNdScalar) -> float:
    """
    Median of a 1D scalar array (``numpy.median``).

    Finds the middle order statistic on the device, by a radix select or a radix sort of a copy,
    and reads the middle element (odd length) or averages the two middle elements (even length).
    The result is always a Python ``float``, regardless of input dtype.

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

    Examples
    --------
    Mirrors [`numpy.median`][]: an even length averages the two middle values:

    ```python
    import warp as wp

    import ordito as od

    print(od.reduce.median(wp.array([3, 1, 4, 1, 5], dtype=wp.int32)))  # 3.0
    print(od.reduce.median(wp.array([3, 1, 4, 1], dtype=wp.int32)))  # 2.0
    ```

    See Also
    --------
    [`quantile`][ordito.reduce.quantile]
    """
    return _sorted_quantile(array, 0.5, "median")


def quantile(array: odt.ArrayNdScalar, q: float) -> float:
    """
    Quantile ``q`` of a 1D scalar array (``numpy.quantile`` with ``method="linear"``).

    Finds the two order statistics around the fractional rank ``q * (n - 1)`` on the device (a
    radix select, or a radix sort of a copy) and interpolates linearly between them, as NumPy's
    default method does, so ``q = 0`` is the minimum, ``q = 1`` the maximum and ``q = 0.5`` the
    [`median`][ordito.reduce.median]. The result is always a Python ``float``.

    Parameters
    ----------
    array
        ``(n,)`` array of any 32- or 64-bit scalar dtype accepted by
        ``warp.utils.radix_sort_pairs`` (``wp.int32``, ``wp.uint32``, ``wp.float32``,
        ``wp.int64``, ``wp.uint64``, ``wp.float64``). Must be non-empty.
    q
        Quantile to compute, in ``[0, 1]``.

    Returns
    -------
    float
        The quantile on the host.

    Raises
    ------
    ValueError
        If ``array`` is empty or not rank-1, or ``q`` lies outside ``[0, 1]``.

    See Also
    --------
    [`median`][ordito.reduce.median]

    Examples
    --------
    Mirrors [`numpy.quantile`][]: a quantile between two order statistics interpolates them:

    ```python
    import warp as wp

    import ordito as od

    values = wp.array([10.0, 0.0, 30.0, 20.0, 40.0], dtype=wp.float32)
    print(od.reduce.quantile(values, 0.25))  # 10.0
    print(od.reduce.quantile(values, 0.9))  # 36.0
    ```
    """
    return _sorted_quantile(array, q, "quantile")


def _sorted_quantile(array: odt.ArrayNdScalar, q: float, name: str) -> float:
    """
    [`quantile`][ordito.reduce.quantile]'s body, its error messages naming the public caller.

    The two order statistics around the fractional rank come back in one readback, adjacent in the
    sorted copy or side by side in the select's state. Interpolates as NumPy's ``_lerp`` does, from
    the nearer end, so the answer is exact at both ends of the interval and ``q = 0.5`` of an even
    count is the mean of the middle pair, as [`median`][ordito.reduce.median] promises.
    """
    if array.ndim != 1:
        raise ValueError(f"{name} requires a 1D array.")
    n = array.size
    if n == 0:
        raise ValueError(f"{name} requires a non-empty array.")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile q must lie in [0, 1], got {q}.")

    rank = q * (n - 1)
    # ``min`` is this module's reduction, hence the clamp spelled out: ``q = 1`` lands on ``n - 1``.
    lower = math.floor(rank)
    if lower >= n - 1:
        lower = n - 1
    fraction = rank - lower
    if _selects_order_statistics(array):
        low, high = _selected_order_statistics(array, lower, fraction != 0.0)
    else:
        sorted_values = _sorted_copy(array)
        if fraction == 0.0:
            # The rank is an order statistic: one value, and no interpolation to round.
            return float(read_scalar(sorted_values, lower))
        # The two order statistics around a fractional rank, read at once.
        pair = read_values(sorted_values, lower, 2)
        low, high = float(pair[0]), float(pair[1])
    if fraction == 0.0:
        return low
    if fraction < 0.5:
        return low + (high - low) * fraction
    return high - (high - low) * (1.0 - fraction)


# The sizes at which ``quantile`` / ``median`` take ``kernels/reduce``'s radix select over a sort
# on CUDA (2026-11-11, ``quantile(0.9)`` new/old): 32-bit keys 0.79-0.93x from 8 192 to 262 144
# values, par at 524 288, 1.4-1.6x from 2 M (the top digits' few hot bins serialize even pooled)
# and 1.35-1.6x at 4 096, where CUB sorts in one tile; ``int64`` 0.87-0.90x from 4 096 to 262 144
# and 1.05-1.17x from 524 288; ``float64`` 0.52-0.83x at every size from 4 096 to 4 M, its sort
# being twice the 32-bit one's. The CPU device sorts, cheaper there.
_SELECT_FROM_32 = 8192
_SELECT_FROM_64 = 4096
_SELECT_UP_TO = 1 << 18


def _selects_order_statistics(values: odt.ArrayNdScalar) -> bool:
    """Whether ``quantile`` selects ``values``' order statistics rather than sorting them."""
    if not wp.get_device(values.device).is_cuda:
        return False
    n = values.size
    if values.dtype is wp.float64:
        return n >= _SELECT_FROM_64
    if wp.types.type_size_in_bytes(values.dtype) == 8:
        return _SELECT_FROM_64 <= n <= _SELECT_UP_TO
    return _SELECT_FROM_32 <= n <= _SELECT_UP_TO


def _selected_order_statistics(
    values: odt.ArrayNdScalar, rank: int, pair: bool
) -> tuple[float, float]:
    """
    Return the ``rank``-th smallest of ``values`` and, when ``pair``, the next one, on CUDA.

    A radix select (``kernels/reduce.RADIX_SELECT_PASS``) rather than a sort: three digit passes of
    a 32-bit dtype's order-preserving keys, six of a 64-bit one. The next order statistic is the
    selected key again while its run of equals continues past the rank, else the smallest key
    above it (``RADIX_SELECT_SUCCESSOR``, one pass more). Taken where
    [`_selects_order_statistics`][ordito.reduce._selects_order_statistics] says so. Without
    ``pair`` the second value repeats the first.
    """
    device = values.device
    dtype = values.dtype
    n = values.size
    passes = (
        kernel_reduce.RADIX_SELECT_PASSES_32
        if wp.types.type_size_in_bytes(dtype) == 4
        else kernel_reduce.RADIX_SELECT_PASSES_64
    )
    # The state, the digit histogram and the passes' ticket in one zeroed allocation: each pass's
    # last block re-zeroes the bins it read and the ticket, and the first pass takes the rank.
    bins = kernel_reduce.RADIX_SELECT_BINS
    state_bytes = kernel_reduce.SELECT_STATE_SIZE * 8
    scratch = _launch.zeros(
        kernel_reduce.SELECT_STATE_SIZE + (bins + 2) // 2, dtype=wp.uint64, device=device
    )
    state = _launch.view(scratch, 0, kernel_reduce.SELECT_STATE_SIZE)
    hist = _launch.alias(scratch, wp.int32, state_bytes, bins)
    ticket = _launch.alias(scratch, wp.int32, state_bytes + 4 * bins, 1)
    blocks = -(-n // kernel_reduce.RADIX_SELECT_ITEMS_PER_BLOCK)
    select_pass = kernel_reduce.RADIX_SELECT_PASS[dtype]
    seed_rank = rank
    for shift, width in passes:
        _launch.launch_tiled(
            select_pass,
            dim=[blocks],
            inputs=[values, shift, width, seed_rank, hist, ticket, state],
            block_dim=TILE_1D,
            device=device,
        )
        seed_rank = -1
    if pair:
        _launch.launch_tiled(
            kernel_reduce.RADIX_SELECT_SUCCESSOR[dtype],
            dim=[blocks],
            inputs=[values, state],
            block_dim=TILE_1D,
            device=device,
        )
    # Host readback: the selected key, its rank within its run of equals, the run's length and the
    # smallest key above it (complemented) -- the answer.
    key, within, run, successor = (int(value) for value in read_values(state, 0, 4))
    next_key = key if not pair or within + 1 < run else UINT64_MAX - successor
    return _ordered_key_value(key, dtype), _ordered_key_value(next_key, dtype)


def _ordered_key_value(key: int, dtype: object) -> float:
    """Return the value an order-preserving key of ``kernels/reduce``'s radix select stands for."""
    if dtype is wp.uint32 or dtype is wp.uint64:
        return float(key)
    if dtype is wp.int32:
        return float(key - (1 << 31))
    if dtype is wp.int64:
        return float(key - (1 << 63))
    width = 32 if dtype is wp.float32 else 64
    sign = 1 << (width - 1)
    bits = key ^ sign if key & sign else ~key & ((1 << width) - 1)
    host = np.array([bits], dtype=np.uint32 if width == 32 else np.uint64)
    return float(host.view(np.float32 if width == 32 else np.float64)[0])


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
    bool_1d: odt.Kernel
    axis_rows_tiled: odt.Kernel
    axis_cols_tiled: odt.Kernel
    axis_rows_serial: odt.Kernel
    axis_cols_serial: odt.Kernel
    tiled_1d: odt.Kernel
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
        init_global=odt.dtype_max,
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
        init_global=odt.dtype_min,
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
        init_global=odt.dtype_max,
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
        init_global=odt.dtype_zero,
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


def _sum_of_terms_buffer(
    kernels: kernel_reduce.TermSum,
    inputs: Sequence[object],
    n: int,
    *,
    out: wp.array[Any] | None = None,
    device: wp.DeviceLike = None,
) -> wp.array[Any]:
    """
    [`sum_of_terms`][ordito.reduce.sum_of_terms]'s body: a buffer whose slot 0 holds the total.

    ``out`` when given, else the commit's own buffer, which a reproducible fold leaves longer than
    one slot: a caller reading slot 0 back skips the view that
    [`sum_of_terms`][ordito.reduce.sum_of_terms] cuts.
    """
    if device is None:
        device = out.device if out is not None else cast(wp.array[Any], inputs[0]).device
    blocks = kernels.blocks(n)
    if blocks > kernel_reduce.FOLDED_ROWS_UP_TO and is_deterministic():
        # More rows than one last block should fold: stored, then added by a launch of their own.
        rows = _launch.empty(blocks, dtype=kernels.dtype, device=device)
        _launch.launch_tiled(
            kernels.partials, dim=[blocks], inputs=[*inputs, rows], block_dim=TILE_1D, device=device
        )
        inputs = [rows]
        kernels = kernel_reduce.FOLD_ROWS[kernels.dtype]
        blocks = kernel_reduce.blocks_1d(blocks)
    if blocks == 1:
        # One block stores its sum: no fold, and no zeroed slot to add into.
        if out is None:
            out = _launch.empty(1, dtype=kernels.dtype, device=device)
        _launch.launch_tiled(
            kernels.partials, dim=[1], inputs=[*inputs, out], block_dim=TILE_1D, device=device
        )
        return out
    # The buffer picks the commit: a one-slot buffer takes each block's ``atomic_add`` (the setting
    # off), a ``block_partials`` one a stored row a block and the last block's fixed-order fold.
    if is_deterministic():
        partials, _ = _launch.block_partials(blocks, 1, kernels.dtype, device)
        _launch.launch_tiled(
            kernels.kernel,
            dim=[blocks],
            inputs=[*inputs, partials],
            block_dim=TILE_1D,
            device=device,
        )
        if out is None:
            return partials
        _launch.copy(out, partials, count=1)
        return out
    if out is None:
        out = _launch.zeros(1, dtype=kernels.dtype, device=device)
    _launch.launch_tiled(
        kernels.kernel, dim=[blocks], inputs=[*inputs, out], block_dim=TILE_1D, device=device
    )
    return out


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
    # One readback of the six slots; 3..5 hold the *negated* upper corner (see the kernel).
    lower_and_negated_upper = read_values(corners, 0, 6)
    return wp.vec3(*lower_and_negated_upper[:3]), wp.vec3(
        *(-value for value in lower_and_negated_upper[3:])
    )


def _reduce_scalar(
    array: odt.ArrayNdScalar, axis: Literal[0, 1] | None, spec: _ScalarReduceSpec
) -> (
    float
    | int
    | odt.Array1dScalar
    | tuple[odt.Array1dScalar, odt.Array1dScalar]
    | tuple[float, float]
    | tuple[int, int]
):
    _validate_scalar_array(array, spec, axis)

    if array.ndim == 2 and axis is not None:
        return _launch_axis_scalar(cast("odt.Array2dScalar", array), axis, spec)

    return _launch_global_scalar_tiled(array, spec)


def _launch_global_scalar_tiled(
    array: odt.ArrayNdScalar, spec: _ScalarReduceSpec
) -> float | int | tuple[float, float] | tuple[int, int]:
    flat = _flattened_for_global(array)
    if spec.name == "sum" and array.dtype in (wp.float32, wp.float64) and is_deterministic():
        if flat is None:
            # A strided view has only the rank-2 kernel's arrival-order commit, so the
            # reproducible sum takes a contiguous copy and the fixed-order 1-D sum.
            flat = cast(odt.Array1dScalar, _launch.clone(array).flatten())
        total = _sum_of_terms_buffer(kernel_reduce.ARRAY_SUMS[array.dtype], [flat], flat.size)
        return float(read_scalar(total, 0))
    if spec.global_output_slots == 2:
        # Both seeds in one native fill of their packed bytes: 1.4x cheaper on CUDA than uploading
        # them from a host list, which was itself 2x cheaper than a ``full`` plus a ``fill_`` of
        # the second slot.
        out = _launch.pattern(
            [odt.dtype_max(array.dtype), odt.dtype_min(array.dtype)],
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

    # ``read_values``, not ``out.numpy()``: one offset copy into a cached host scratch rather than a
    # fresh host array (about half the cost), and its ``tolist`` hands back the native Python
    # ``int`` / ``float`` for whichever ``array.dtype`` is.
    values = read_values(out, 0, spec.global_output_slots)
    if spec.global_output_slots == 2:
        return cast(tuple[int, int] | tuple[float, float], (values[0], values[1]))
    return cast(float | int, values[0])


def _flattened_for_global(array: odt.ArrayNdScalar) -> odt.Array1dScalar | None:
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
        return cast(odt.Array1dScalar, array)
    if array.is_contiguous:
        return cast(odt.Array1dScalar, array.flatten())
    return None


def _launch_axis_scalar(
    array: odt.Array2dScalar, axis: Literal[0, 1], spec: _ScalarReduceSpec
) -> odt.Array1dScalar | tuple[odt.Array1dScalar, odt.Array1dScalar]:
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
            packed = odt.empty_2d((2, n_out), array.dtype, device=array.device)
            out_min, out_max = packed[0], packed[1]
            _launch.launch(serial, dim=n_out, inputs=[array, out_min, out_max], device=array.device)
            return cast(odt.Array1dScalar, out_min), cast(odt.Array1dScalar, out_max)
        out = odt.empty_1d(n_out, array.dtype, device=array.device)
        _launch.launch(serial, dim=n_out, inputs=[array, out], device=array.device)
        return out

    tiled = (spec.axis_rows_tiled if axis == 1 else spec.axis_cols_tiled)[array.dtype]
    n_tiles = (reduced + TILE_1D - 1) // TILE_1D
    if spec.dual_axis:
        # Both seeds in one allocation, uploaded together: row 0 accumulates the minimum from
        # the dtype's largest value, row 1 the maximum from its smallest.
        seeds = np.empty((2, n_out), dtype=wp.dtype_to_numpy(array.dtype))
        seeds[0] = odt.dtype_max(array.dtype)
        seeds[1] = odt.dtype_min(array.dtype)
        packed = _launch.array(seeds, dtype=array.dtype, device=array.device)
        out_min, out_max = packed[0], packed[1]
        _launch.launch_tiled(
            tiled,
            dim=[n_out, n_tiles],
            inputs=[array, out_min, out_max],
            block_dim=TILE_1D,
            device=array.device,
        )
        return cast(odt.Array1dScalar, out_min), cast(odt.Array1dScalar, out_max)

    if (
        n_tiles > 1
        and spec.name == "sum"
        and array.dtype in (wp.float32, wp.float64)
        and is_deterministic()
    ):
        # One partial per tile, added in a fixed order, so a float sum repeats bit for bit on CUDA.
        # A single tile per slot commits one atomic onto the zero seed, which is already exact.
        partials_kernel = (
            kernel_reduce.SUM_2D_ROWS_PARTIALS if axis == 1 else kernel_reduce.SUM_2D_COLS_PARTIALS
        )[array.dtype]
        partials = _launch.zeros((n_tiles + 2) * n_out, dtype=array.dtype, device=array.device)
        _launch.launch_tiled(
            partials_kernel,
            dim=[n_out, n_tiles],
            inputs=[array, partials],
            block_dim=TILE_1D,
            device=array.device,
        )
        return cast(odt.Array1dScalar, _launch.view(partials, 0, n_out))
    out = _launch.full(n_out, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)
    _launch.launch_tiled(
        tiled, dim=[n_out, n_tiles], inputs=[array, out], block_dim=TILE_1D, device=array.device
    )
    return cast(odt.Array1dScalar, out)


def _validate_scalar_array(
    array: odt.ArrayNdScalar, spec: _ScalarReduceSpec, axis: Literal[0, 1] | None
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
    # not instantiate for ``wp.bool``, so the shared scalar factories
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
