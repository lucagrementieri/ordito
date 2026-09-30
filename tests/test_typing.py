from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from tests.conversions import warp_empty


def test_ensure_ndim_rejects_1d(device: str) -> None:
    arr = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    with pytest.raises(TypeError, match="expected 2D array"):
        twt.ensure_ndim(arr, 2)


def test_as_array2d_accepts_2d(device: str) -> None:
    arr = warp_empty((2, 3), wp.int32, device)
    out = twt.as_array2d(arr, wp.int32)
    assert out.ndim == 2
    assert out.dtype == wp.int32


_INT_WP_TO_NUMPY: tuple[tuple[type, type[np.integer]], ...] = (
    (wp.int8, np.int8),
    (wp.uint8, np.uint8),
    (wp.int16, np.int16),
    (wp.uint16, np.uint16),
    (wp.int32, np.int32),
    (wp.uint32, np.uint32),
    (wp.int64, np.int64),
    (wp.uint64, np.uint64),
)

_FLOAT_DTYPES: tuple[type, ...] = (wp.float16, wp.float32, wp.float64)


def test_dtype_max() -> None:
    for wp_dt, np_ic in _INT_WP_TO_NUMPY:
        np_dtype = np.dtype(np_ic)
        assert twt.dtype_max(wp_dt) == np.iinfo(np_dtype).max
    for wp_dt in _FLOAT_DTYPES:
        assert np.isposinf(twt.dtype_max(wp_dt))


def test_dtype_min() -> None:
    for wp_dt, np_ic in _INT_WP_TO_NUMPY:
        np_dtype = np.dtype(np_ic)
        assert twt.dtype_min(wp_dt) == np.iinfo(np_dtype).min
    for wp_dt in _FLOAT_DTYPES:
        assert np.isneginf(twt.dtype_min(wp_dt))


_ALLOCATOR_DTYPES = [wp.int32, wp.float32, wp.float64, wp.bool, wp.vec3, wp.mat33]


@pytest.mark.parametrize("dtype_wp", _ALLOCATOR_DTYPES)
@pytest.mark.parametrize("shape", [(5,), (0,), (3, 4), (0, 2), (2, 3, 4)])
def test_the_empty_family_is_one_allocator_at_three_ranks(
    device: str, shape: tuple[int, ...], dtype_wp: type
) -> None:
    """
    Not a library comparison: no reference exposes a rank-fixing allocator.

    The three entry points differ only in the rank they fix, so every dtype must work at every
    rank -- they used to admit three different dtype sets (1d int32/float32/float64, 2d plus
    ``vec3``, 3d only float32/bool), which was an artifact of hand-written overload tables rather
    than a real restriction. Parametrizing one test over the cross product is what keeps them from
    drifting apart again. Excludes: nothing about the *contents*, which are deliberately
    uninitialized.
    """
    allocators: dict[int, Callable[..., wp.array[object, Any]]] = {
        1: twt.empty_1d,
        2: twt.empty_2d,
        3: twt.empty_3d,
    }
    allocate = allocators[len(shape)]
    arr = allocate(shape[0] if len(shape) == 1 else shape, dtype_wp, device=device)
    assert arr.shape == shape
    assert arr.ndim == len(shape)
    assert arr.dtype == dtype_wp
    assert str(arr.device) == device


def test_the_empty_family_rejects_a_shape_of_the_wrong_rank(device: str) -> None:
    """The rank each entry point fixes is checked against ``shape``, not merely annotated."""
    with pytest.raises(ValueError, match="2D shape must have length 2"):
        twt.empty_2d((2, 3, 4), wp.int32, device=device)  # pyright: ignore[reportArgumentType]  # the wrong rank under test
    with pytest.raises(ValueError, match="3D shape must have length 3"):
        twt.empty_3d((2, 3), wp.int32, device=device)  # pyright: ignore[reportArgumentType]  # the wrong rank under test


def test_dtype_zero_splits_int_and_float_like_python() -> None:
    """Integer types give a Python ``int`` and float ones a ``float``, not merely ``== 0``."""
    for dtype_wp, _np_dtype in _INT_WP_TO_NUMPY:
        zero = twt.dtype_zero(dtype_wp)
        assert zero == 0
        assert isinstance(zero, int)
        assert not isinstance(zero, bool)
    for dtype_wp in _FLOAT_DTYPES:
        zero = twt.dtype_zero(dtype_wp)
        assert zero == 0.0
        assert isinstance(zero, float)


@pytest.mark.parametrize("dtype_wp", [wp.float32, wp.bool])
def test_empty_3d_shape(device: str, dtype_wp: type) -> None:
    """The rank-3 allocator over both dtypes its overloads admit, on the caller's device."""
    arr = twt.empty_3d((2, 3, 4), dtype_wp, device=device)
    assert arr.shape == (2, 3, 4)
    assert arr.ndim == 3
    assert arr.dtype == dtype_wp
    assert str(arr.device) == device


_SCALAR_DTYPES: tuple[type, ...] = (
    wp.int8,
    wp.uint8,
    wp.int16,
    wp.uint16,
    wp.int32,
    wp.uint32,
    wp.int64,
    wp.uint64,
    wp.float16,
    wp.float32,
    wp.float64,
)


def test_sortable_dtype_is_exactly_what_warp_can_radix_sort(device: str) -> None:
    """
    The widening table's *reason*, asserted rather than described: it names Warp's accepted set.

    ``sortable_dtype`` exists because ``warp.utils.radix_sort_pairs`` refuses sub-32-bit keys, and
    its docstring records having re-probed that on 1.16. This runs the probe instead of citing it:
    every dtype the table maps to itself must sort, every dtype it widens must *not*, and the
    widened target must sort. So a Warp release that grows the accepted set fails here -- which is
    the only way anyone would notice that the table had become unnecessarily lossy.

    Measured on Warp 1.17, both devices: ``int32`` / ``uint32`` / ``int64`` / ``uint64`` /
    ``float32`` / ``float64`` are accepted, and ``int8`` / ``uint8`` / ``int16`` / ``uint16`` /
    ``float16`` raise ``Unsupported keys and values data types``.
    """

    def sorts(dtype: type) -> bool:
        keys_wp = wp.zeros(8, dtype=dtype, device=device)
        values_wp = wp.zeros(8, dtype=wp.int32, device=device)
        try:
            wp.utils.radix_sort_pairs(keys_wp, values_wp, 4)
        except RuntimeError:
            return False
        return True

    accepted = {dtype for dtype in _SCALAR_DTYPES if sorts(dtype)}
    assert accepted == {wp.int32, wp.uint32, wp.int64, wp.uint64, wp.float32, wp.float64}

    for dtype in _SCALAR_DTYPES:
        target = twt.sortable_dtype(dtype)
        assert target in accepted, f"{dtype.__name__} widened to an unsortable {target.__name__}"
        # A fixed point exactly on the accepted set: nothing sortable is widened, nothing else is
        # left alone.
        assert (target is dtype) == (dtype in accepted)
        # Same kind and signedness, never narrower -- the order has to survive the widening.
        assert wp.types.type_is_float(target) == wp.types.type_is_float(dtype)
        assert target.__name__.startswith("u") == dtype.__name__.startswith("u")
        assert wp.types.type_size_in_bytes(target) >= wp.types.type_size_in_bytes(dtype)


@pytest.mark.parametrize("dtype_wp", [wp.int8, wp.uint16, wp.float16, wp.uint64])
def test_sortable_dtype_preserves_the_order_of_the_original_values(
    device: str, dtype_wp: type
) -> None:
    """
    Class A: casting to the widened dtype and sorting there orders the values as numpy does.

    Widening is only useful if it is order-preserving, and the two ways to get that wrong are the
    two the docstring names: a float's sign bit makes negatives descend under a bit-order sort, and
    a ``uint64`` with its top bit set reads as a negative ``int64``. Both are covered --
    ``float16`` carries negatives and ``uint64`` carries values above ``2**63``.

    The cast-then-sort shape is the caller's, not the helper's: ``sort_and_argsort`` requires an
    already-sortable dtype and says so, and this mirrors what ``grouping.group`` and
    ``grouping.unique_1d`` do with the answer.
    """
    sort_dtype = twt.sortable_dtype(dtype_wp)
    if dtype_wp is wp.uint64:
        values_np = np.array([2**63 + 5, 1, 2**64 - 1, 0, 2**63], dtype=np.uint64)
    elif dtype_wp is wp.float16:
        values_np = np.array([-2.5, 0.0, 1.5, -7.0, 3.0], dtype=np.float16)
    elif dtype_wp is wp.int8:
        values_np = np.array([-128, 0, 127, -1, 63], dtype=np.int8)
    else:
        values_np = np.array([65535, 0, 1, 32768, 7], dtype=np.uint16)
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)

    widened_wp = warp_empty(values_np.size, sort_dtype, device)
    if sort_dtype is dtype_wp:
        wp.copy(widened_wp, values_wp)
    else:
        wp.utils.array_cast(values_wp, widened_wp)

    # One array of a union dtype, where the sort takes a union of arrays: dtype is invariant.
    sorted_wp, order_wp = tw.array.sort_and_argsort(cast("twt.ArrayNdScalar", widened_wp))

    # Compare against the *original* dtype's numpy order: the widening must not have changed it.
    assert np.array_equal(order_wp.numpy(), np.argsort(values_np, kind="stable"))
    assert np.array_equal(sorted_wp.numpy(), np.sort(values_np).astype(sorted_wp.numpy().dtype))
