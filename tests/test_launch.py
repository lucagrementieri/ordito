"""
Tests for ``triwarp._launch``: the cached launcher, stamped allocation and native utility paths.

Not a library comparison: every fast path here reimplements a Warp call, so Warp's own call is the
oracle and each test is triwarp against Warp on the same inputs, byte for byte. The CPU device
exercises the fallbacks, which must be Warp itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
import pytest
import warp as wp

from tests.conversions import warp_empty
from triwarp import _launch


class _CopyKwargs(TypedDict, total=False):
    """The optional ``wp.copy`` keywords the copy test varies."""

    dest_offset: int
    src_offset: int
    count: int


@wp.kernel  # pyright: ignore[reportUntypedFunctionDecorator]  # wp.kernel has no return annotation
def _mixed_arguments(
    a: wp.array[wp.float32],
    grid: wp.array2d[wp.float32],
    vectors: wp.array[wp.vec3],
    scale: wp.float32,
    shift: wp.int32,
    big: wp.uint64,
    offset: wp.vec3,
    frame: wp.mat33,
    flag: wp.bool,
    out: wp.array[wp.float32],
    out_vectors: wp.array[wp.vec3],
) -> None:
    # Kernel scope: Warp's stubs type tid() and element access for Python scope.
    i = wp.int32(wp.tid())  # pyright: ignore[reportArgumentType]
    value = a[i] * scale + wp.float32(shift) + grid[i, 1] + wp.float32(big % wp.uint64(7))  # pyright: ignore[reportArgumentType, reportIndexIssue]
    if flag:
        value += 1.0
    out[i] = value  # pyright: ignore[reportIndexIssue]
    out_vectors[i] = frame * (vectors[i] + offset)  # pyright: ignore[reportIndexIssue]


@wp.kernel  # pyright: ignore[reportUntypedFunctionDecorator]  # wp.kernel has no return annotation
def _tile_index(out: wp.array2d[wp.int32]) -> None:
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]  # Warp's stub types tid() loosely
    out[i, j] = i * 100 + j  # pyright: ignore[reportIndexIssue]


@wp.func
def _scaled_point(point: wp.vec3, matrix: wp.mat44) -> wp.vec3:
    return wp.transform_point(matrix, point)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _inputs(device: str, n: int = 257) -> dict[str, wp.array[object]]:
    rng = np.random.default_rng(3)
    return {
        "a": wp.array(rng.random(n, dtype=np.float32), device=device),
        "grid": wp.array(rng.random((n, 3), dtype=np.float32), device=device),
        "vectors": wp.array(rng.random((n, 3), dtype=np.float32), dtype=wp.vec3, device=device),
    }


def _launch_both(
    kernel_launch: Callable[..., object], device: str, args: Sequence[object], n: int
) -> tuple[np.ndarray, np.ndarray]:
    out = wp.zeros(n, dtype=wp.float32, device=device)
    out_vectors = wp.zeros(n, dtype=wp.vec3, device=device)
    kernel_launch(_mixed_arguments, dim=n, inputs=[*args, out, out_vectors], device=device)
    return out.numpy(), out_vectors.numpy()


def test_launch_matches_wp_launch_on_every_argument_kind(device: str) -> None:
    """Arrays of rank 1 and 2, vector arrays, scalars, a 64-bit id, a vector, a matrix, a bool."""
    n = 257
    data = _inputs(device, n)
    frame = wp.mat33(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 2.0)
    args = [data["a"], data["grid"], data["vectors"], 1.5, 3, 2**40 + 5, wp.vec3(1, 2, 3), frame,
            True]  # fmt: skip
    expected = _launch_both(wp.launch, device, args, n)
    for _ in range(3):  # first call fills the cache, later ones take the packed block
        got = _launch_both(_launch.launch, device, args, n)
        assert np.array_equal(got[0], expected[0])
        assert np.array_equal(got[1], expected[1])


def test_launch_falls_back_for_values_the_block_cannot_pack(device: str) -> None:
    """A Warp scalar, a tuple for a vector and an out-of-range int all reach Warp's marshalling."""
    n = 64
    data = _inputs(device, n)
    frame = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    args = [data["a"], data["grid"], data["vectors"], wp.float32(0.5), 7, 11, (0.5, 0.0, 1.0),
            frame, False]  # fmt: skip
    expected = _launch_both(wp.launch, device, args, n)
    for _ in range(2):
        got = _launch_both(_launch.launch, device, args, n)
        assert np.array_equal(got[0], expected[0])
        assert np.array_equal(got[1], expected[1])


@pytest.mark.parametrize("dim", [(5, 7), (5,), (2, 3, 4)])
def test_launch_extent_matches_wp_launch(device: str, dim: tuple[int, ...]) -> None:
    """A shorter ``dim`` pads with 1, a longer one folds into the kernel's last index, like Warp."""
    rows = int(np.prod(dim[:1]))
    cols = int(np.prod(dim[1:])) if len(dim) > 1 else 1
    expected = wp.full((rows, cols), -1, dtype=wp.int32, device=device)
    got = wp.full((rows, cols), -1, dtype=wp.int32, device=device)
    wp.launch(_tile_index, dim=dim, inputs=[expected], device=device)
    for _ in range(2):
        _launch.launch(_tile_index, dim=dim, inputs=[got], device=device)
    assert np.array_equal(got.numpy(), expected.numpy())


def test_map_with_a_matrix_input_keeps_the_cached_path(device: str) -> None:
    """``wp.map``'s kernel carries its own ``vec_t`` / ``mat_t`` classes; they must still match."""
    points = _inputs(device)["vectors"]
    matrix = wp.mat44(*[float(k % 5) for k in range(16)])
    expected = wp.zeros_like(points)
    wp.map(_scaled_point, points, matrix, out=expected)
    got = wp.zeros_like(points)
    for _ in range(3):
        _launch.map(_scaled_point, points, matrix, out=got)
    assert np.array_equal(got.numpy(), expected.numpy())


def test_launch_inside_a_graph_capture_replays(device: str) -> None:
    """A launch recorded into a CUDA graph through the cached path replays like ``wp.launch``'s."""
    if not wp.get_device(device).is_cuda:
        pytest.skip("graph capture is CUDA-only")
    n = 128
    data = _inputs(device, n)
    frame = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    args = [data["a"], data["grid"], data["vectors"], 2.0, 1, 3, wp.vec3(0, 0, 1), frame, False]
    expected = _launch_both(wp.launch, device, args, n)
    _launch_both(_launch.launch, device, args, n)  # cache the entry outside the capture
    out = wp.zeros(n, dtype=wp.float32, device=device)
    out_vectors = wp.zeros(n, dtype=wp.vec3, device=device)
    with wp.ScopedCapture(device=device) as capture:
        _launch.launch(_mixed_arguments, dim=n, inputs=[*args, out, out_vectors], device=device)
    out.zero_()
    assert capture.graph is not None
    wp.capture_launch(capture.graph)
    assert np.array_equal(out.numpy(), expected[0])
    assert np.array_equal(out_vectors.numpy(), expected[1])


@pytest.mark.parametrize("shape", [7, (4, 3), (2, 3, 5), 0])
@pytest.mark.parametrize("dtype", [wp.float32, wp.int64, wp.vec3, wp.mat22d])
def test_allocations_match_warp(device: str, shape: int | tuple[int, ...], dtype: type) -> None:
    """Stamped arrays carry Warp's shape, strides, size, capacity and element type."""
    reference = wp.zeros(shape, dtype=dtype, device=device)
    for allocate in (_launch.empty, _launch.zeros):
        arr = allocate(shape, dtype=dtype, device=device)
        assert arr.shape == reference.shape
        assert arr.strides == reference.strides
        assert arr.size == reference.size
        assert arr.capacity == reference.capacity
        assert arr.dtype is reference.dtype
        assert arr.device == reference.device
    assert np.array_equal(_launch.zeros(shape, dtype=dtype, device=device).numpy(),
                          reference.numpy())  # fmt: skip
    filled = _launch.full(shape, 3, dtype=wp.int32, device=device)
    assert np.array_equal(filled.numpy(), np.full(reference.shape, 3, dtype=np.int32))


def test_stamped_arrays_free_and_reuse_memory(device: str) -> None:
    """Allocating and dropping many stamped arrays neither leaks nor corrupts the pool."""
    total = 0
    for k in range(200):
        arr = _launch.full(1000 + k, float(k), dtype=wp.float32, device=device)
        total += int(arr.numpy()[-1])
        del arr
    assert total == sum(range(200))


def test_clone_and_likes_match_warp(device: str) -> None:
    """``clone`` / ``empty_like`` / ``zeros_like`` keep the source's layout."""
    src = _inputs(device)["grid"]
    clone = _launch.clone(src)
    assert np.array_equal(clone.numpy(), src.numpy())
    assert clone.ptr != src.ptr
    assert _launch.empty_like(src).shape == src.shape
    assert not _launch.zeros_like(src).numpy().any()


@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_array_scan_matches_warp(device: str, inclusive: bool, dtype: type) -> None:
    """The native scan call gives Warp's scan, inclusive and exclusive."""
    values = wp.array(np.random.default_rng(1).integers(0, 9, 1000).astype(dtype), device=device)
    expected = wp.empty_like(values)
    wp.utils.array_scan(values, expected, inclusive=inclusive)
    got = wp.empty_like(values)
    _launch.array_scan(values, got, inclusive=inclusive)
    assert np.array_equal(got.numpy(), expected.numpy())


@pytest.mark.parametrize("end_bit", [None, 12])
def test_radix_sort_pairs_matches_warp(device: str, end_bit: int | None) -> None:
    """The native sort call gives Warp's stable sort, over the full key or a bit prefix."""
    n = 1000
    keys_np = np.random.default_rng(2).integers(0, 4096, n).astype(np.int32)
    results = []
    for sort in (wp.utils.radix_sort_pairs, _launch.radix_sort_pairs):
        keys = wp.array(np.concatenate([keys_np, keys_np]), device=device)
        values = wp.array(np.arange(2 * n, dtype=np.int32), device=device)
        sort(keys, values, n, end_bit=end_bit)
        results.append((keys.numpy()[:n], values.numpy()[:n]))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])


def test_copy_matches_warp(device: str) -> None:
    """Whole, offset and counted copies; ``count=0`` copies everything, as in Warp."""
    src = wp.array(np.arange(100, dtype=np.int32), device=device)
    cases: tuple[_CopyKwargs, ...] = (
        {},
        {"dest_offset": 3, "src_offset": 10, "count": 20},
        {"count": 0},
    )
    for kwargs in cases:
        expected = wp.full(120, -1, dtype=wp.int32, device=device)
        got = wp.full(120, -1, dtype=wp.int32, device=device)
        wp.copy(expected, src, **kwargs)
        _launch.copy(got, src, **kwargs)
        assert np.array_equal(got.numpy(), expected.numpy())


def test_copy_out_of_bounds_raises_like_warp(device: str) -> None:
    """A copy past the destination's end is Warp's error, not a silent overrun."""
    src = wp.zeros(10, dtype=wp.int32, device=device)
    dest = wp.zeros(5, dtype=wp.int32, device=device)
    with pytest.raises(RuntimeError):
        _launch.copy(dest, src)


@pytest.mark.parametrize("value", [0, 2.5, -0.0, 7])
def test_fill_and_zero_match_warp(device: str, value: float) -> None:
    """``fill_`` / ``zero_`` write Warp's bytes, ``-0.0`` included (it is not a zero memset)."""
    expected = warp_empty(50, wp.float32, device)
    got = warp_empty(50, wp.float32, device)
    expected.fill_(value)
    _launch.fill_(got, value)
    assert np.array_equal(got.numpy().view(np.uint32), expected.numpy().view(np.uint32))
    _launch.zero_(got)
    assert not got.numpy().any()
    view = wp.array(np.arange(20, dtype=np.int32), device=device)[::2]
    _launch.fill_(view, 5)  # non-contiguous: Warp's generic fill
    assert (view.numpy() == 5).all()


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        pytest.param(np.arange(24, dtype=np.int32), wp.int32, id="int32"),
        pytest.param(np.arange(18, dtype=np.float64).reshape(6, 3), wp.vec3, id="float64-to-vec3"),
        pytest.param(np.arange(1000), wp.int32, id="int64-to-int32"),
        pytest.param(np.arange(12, dtype=np.int32).reshape(4, 3), wp.int32, id="rank-2"),
        pytest.param([1, 1, 0], wp.int32, id="int-list"),
        pytest.param([wp.vec3(1.0, 2.0, 3.0)], wp.vec3, id="vec3-list"),
        pytest.param([wp.mat44(*range(16))], wp.mat44, id="mat44-list"),
        pytest.param(np.arange(12, dtype=np.float32), wp.vec3, id="flat-to-vec3"),
        pytest.param(np.arange(20, dtype=np.int32)[::2], wp.int32, id="strided"),
        pytest.param(np.zeros(0, dtype=np.int32), wp.int32, id="empty"),
    ],
)
def test_array_upload_matches_wp_array(
    device: str, data: list[object] | np.ndarray, dtype: type
) -> None:
    """Shape, strides, dtype and contents equal ``wp.array``'s, the reshape-lenient cases too."""
    expected = wp.array(data, dtype=dtype, device=device)
    got = _launch.array(data, dtype=dtype, device=device)
    assert got.shape == expected.shape
    assert got.strides == expected.strides
    assert got.dtype is expected.dtype
    assert np.array_equal(got.numpy(), expected.numpy())


def test_array_upload_rejects_a_scalar_like_wp_array(device: str) -> None:
    """A scalar is not array data; the error is Warp's."""
    with pytest.raises(RuntimeError):
        wp.array(3, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]  # deliberately a scalar
    with pytest.raises(RuntimeError):
        _launch.array(3, dtype=wp.int32, device=device)
