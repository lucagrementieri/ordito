"""Regression tests for ``triwarp.geometry`` against ``trimesh.geometry`` (CPU reference)."""

from __future__ import annotations

import gc
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytest
import pytorch3d.ops as p3d_ops
import scipy.sparse
import torch
import trimesh as tm
import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from tests.conversions import points_to_warp, warp_empty
from triwarp import _device

DType = TypeVar("DType")


@pytest.mark.parametrize(
    "args",
    [(8,), (0,), (3, 11), (0, 18, 3), (2, 20, 6), (11, 3, -2), (5, 5), (5, 2), (-4, 4), (-4, 4, 3)],
)
def test_arange_matches_numpy(device: str, args: tuple[int, ...]) -> None:
    """
    Class A: the merged range reproduces [`numpy.arange`][] over every positional arity.

    The cases are chosen so each one exercises something the single-argument form cannot: the
    two- and three-argument overloads, a negative ``step`` (a descending interval), an empty
    result from ``start == stop`` and from ``start > stop`` with a positive step, and negative
    values, which the ``i * step`` kernel would get wrong if it dropped ``start``.
    """
    out_wp = tw.array.arange(*args, device=device)
    assert np.array_equal(out_wp.numpy(), np.arange(*args, dtype=np.int32))


def test_arange_rejects_a_zero_step(device: str) -> None:
    with pytest.raises(ValueError, match="step must be non-zero"):
        tw.array.arange(0, 10, 0, device=device)


def test_sort_pair_indices(device: str) -> None:
    n, fill = 5, -1
    out_wp = tw.array.sort_pair_indices(n, fill, device)
    expected_np = np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1], dtype=np.int32)
    assert np.array_equal(out_wp.numpy(), expected_np)


def test_sort_pair_indices_zero(device: str) -> None:
    out_wp = tw.array.sort_pair_indices(0, -1, device)
    assert out_wp.shape == (0,)


def test_arange_repeat(device: str) -> None:
    count, repeats = 9, 3
    out_wp = tw.array.arange_repeat(count, repeats, device)
    expected_np = np.repeat(np.arange(count // repeats, dtype=np.int32), repeats)
    assert np.array_equal(out_wp.numpy(), expected_np)


def test_arange_repeat_zero_count(device: str) -> None:
    out_wp = tw.array.arange_repeat(0, 4, device)
    assert out_wp.shape == (0,)


def test_index_builders_are_int32_and_guard_the_range(device: str) -> None:
    """
    Not a library comparison: the dtype contract of the three index-buffer builders.

    They are ``int32`` by signature -- the ``dtype=`` keyword they used to advertise accepted only
    ``wp.int32`` and raised a ``KeyError`` from the kernel table for every other value, so it was
    removed rather than widened. The dtype assert is what makes that a stated contract instead of
    an accident of the default, and it is a *membership* check of the kind a value comparison
    cannot make: a builder that silently returned ``int64`` would still compare equal to numpy.

    The range guard is the other half. With no ``dtype`` to widen to, a range too large for the
    buffer must raise rather than wrap, and each of the three reaches that check by a different
    route -- the interval's endpoints, the repeated index, and the padding value.
    """
    assert tw.array.arange(4, device=device).dtype == wp.int32
    assert tw.array.arange_repeat(9, 3, device).dtype == wp.int32
    assert tw.array.sort_pair_indices(5, -1, device).dtype == wp.int32

    # Each raises before it allocates, so these cost nothing despite the sizes named.
    with pytest.raises(ValueError, match=r"start=.* out of range for int32"):
        tw.array.arange(2**31, 2**31 + 2, device=device)
    with pytest.raises(ValueError, match=r"out of range for int32"):
        tw.array.arange_repeat(2**31 + 1, 1, device)
    with pytest.raises(ValueError, match=r"fill_value=.* out of range for int32"):
        tw.array.sort_pair_indices(1, 2**31, device)


def test_concatenate(device: str) -> None:
    parts = [
        wp.array([0, 3], dtype=wp.int32, device=device),
        wp.array([], dtype=wp.int32, device=device),
        wp.array([7, 12], dtype=wp.int32, device=device),
    ]
    out_wp = tw.array.concatenate(parts)
    assert np.array_equal(out_wp.numpy(), np.array([0, 3, 7, 12], dtype=np.int32))


def test_concatenate_single_returns_input(device: str) -> None:
    arr_wp = wp.array([1, 2], dtype=wp.int32, device=device)
    out_wp = tw.array.concatenate([arr_wp])
    assert out_wp is arr_wp


def test_concatenate_empty_segments(device: str) -> None:
    out_wp = tw.array.concatenate([warp_empty(0, wp.int32, device)])
    assert out_wp.shape == (0,)


def test_split_roundtrips_pack_1d_arrays(device: str) -> None:
    """``split(*pack_1d_arrays(arrays))`` recovers every input segment (the inverse pair)."""
    rng = np.random.default_rng(3)
    parts_np = [rng.integers(0, 100, size=size).astype(np.int32) for size in (4, 1, 0, 7)]
    parts_wp = [wp.array(part, dtype=wp.int32, device=device) for part in parts_np]

    flat_wp, offsets_wp = tw.array.pack_1d_arrays(parts_wp)
    segments_wp = tw.array.split(flat_wp, offsets_wp)

    assert len(segments_wp) == len(parts_np)
    for segment_wp, part_np in zip(segments_wp, parts_np, strict=True):
        assert np.array_equal(segment_wp.numpy(), part_np)


@pytest.mark.parity("concatenate_arrays", "numpy")
@pytest.mark.parity("pack_1d_arrays", "numpy")
@pytest.mark.parity("split_array", "numpy")
@pytest.mark.parametrize("copy", [False, True])
def test_the_packing_family_matches_numpy(device: str, copy: bool) -> None:
    """
    Class A on all three: ``np.concatenate``, a ``cumsum`` of the lengths, and ``np.split``.

    The three primitives are one round trip, so they are asserted as one -- and against the NumPy
    calls the benchmark rows are timed against rather than against a literal, which is what makes
    this a comparison instead of a restatement. The offsets are host metadata on both sides, so
    ``pack_1d_arrays``' second return has an exact NumPy counterpart and not merely a compatible
    shape.

    Both ``copy`` modes are run because ``np.split`` has the same two and names them the same way:
    its result is views, and a copy is one ``np.copy`` per piece. The values are identical either
    way, which is the point -- what differs is storage, and that is asserted in
    ``test_split_views_share_storage_and_copies_do_not`` instead.

    A **zero-length segment** is in the middle of the input on purpose. It is the case an offsets
    scheme gets wrong (two consecutive offsets that are equal), and the one where ``np.split`` and
    triwarp could plausibly disagree about which piece is empty.
    """
    rng = np.random.default_rng(3)
    parts_np = [
        np.ascontiguousarray(rng.integers(0, 100, size=size).astype(np.int32))
        for size in (4, 1, 0, 7)
    ]
    parts_wp = [wp.array(part, dtype=wp.int32, device=device) for part in parts_np]

    flat_np = np.concatenate(parts_np)
    offsets_np = np.cumsum([0] + [part.size for part in parts_np])
    assert np.array_equal(tw.array.concatenate(parts_wp).numpy(), flat_np)

    packed_wp, packed_offsets_wp = tw.array.pack_1d_arrays(parts_wp)
    assert np.array_equal(packed_wp.numpy(), flat_np)
    assert np.array_equal(packed_offsets_wp.numpy(), offsets_np.astype(np.int32))

    split_np = np.split(flat_np, offsets_np[1:-1])
    if copy:
        split_np = [np.copy(part) for part in split_np]
    segments_wp = tw.array.split(packed_wp, packed_offsets_wp, copy=copy)
    assert len(segments_wp) == len(split_np)
    for segment_wp, part_np in zip(segments_wp, split_np, strict=True):
        assert np.array_equal(segment_wp.numpy(), part_np)


@pytest.mark.parity("concatenate_arrays", "pytorch3d")
def test_the_packing_family_matches_pytorch3d(device: str) -> None:
    """
    Class B: pytorch3d's ``packed_to_padded`` is triwarp's ``(flat, offsets)`` densified.

    The transform is the padding itself: pytorch3d's packed form is triwarp's flat buffer verbatim
    and its padded form is a dense ``(n_segments, max_size)`` block zero-filled past each segment's
    length -- so ``split`` produces exactly the rows of that block, and ``padded_to_packed`` undoes
    it byte-for-byte.

    The convention worth pinning is the second argument: pytorch3d's ``first_idxs`` are **starting
    indices, not counts**, which makes them ``pack_1d_arrays``' ``offsets`` without its trailing
    total -- measured ``[0, 3, 8]`` for segment lengths ``(3, 5, 2)`` from both sides, against
    triwarp's ``[0, 3, 8, 10]``. That trailing total is what ``padded_to_packed`` wants as
    ``total_size``; handing it a mismatched size does not raise, it **corrupts the heap** (the
    C++ kernels bounds-check nothing), which is why the sizes below are read off the buffers.
    """
    rng = np.random.default_rng(9)
    segments_np = [rng.normal(size=length).astype(np.float32) for length in (3, 5, 2)]
    flat_wp, offsets_wp = tw.array.pack_1d_arrays(
        [wp.array(segment_np, dtype=wp.float32, device=device) for segment_np in segments_np]
    )
    flat_np, offsets_np = flat_wp.numpy(), offsets_wp.numpy()
    first_p3d = cast(
        "torch.LongTensor", torch.as_tensor(offsets_np[:-1].astype(np.int64), device=device)
    )

    assert np.array_equal(offsets_np, np.array([0, 3, 8, 10], dtype=np.int32))
    padded_p3d = p3d_ops.packed_to_padded(
        torch.as_tensor(flat_np, device=device), first_p3d, max(len(s) for s in segments_np)
    )
    repacked_p3d = p3d_ops.padded_to_packed(padded_p3d, first_p3d, flat_np.size)

    assert padded_p3d.shape == (len(segments_np), 5)
    assert np.array_equal(repacked_p3d.cpu().numpy(), flat_np)
    parts_wp = tw.array.split(flat_wp, offsets_wp)
    padded_np = padded_p3d.cpu().numpy()
    for index, part_wp in enumerate(parts_wp):
        length = part_wp.size
        assert np.array_equal(padded_np[index, :length], part_wp.numpy())
        assert np.array_equal(padded_np[index, length:], np.zeros(5 - length, dtype=np.float32))


def test_pack_1d_arrays_reuses_what_split_produced(device: str) -> None:
    """
    Triwarp against triwarp: ``pack_1d_arrays(split(flat), copy=False)`` hands ``flat`` back.

    The oracle for the *values* is the ``copy=True`` default, which every other test in this file
    exercises; what only this can check is that the free path is taken at all and that it is taken
    **only** where the segments really tile one buffer -- an adjacency test would also fire on two
    separate allocations the memory pool happened to place end to end, and would then silently turn
    a caller's copy into an alias.
    """
    flat_wp = wp.array(np.arange(12, dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array([0, 3, 8, 12], dtype=np.int32), dtype=wp.int32, device=device)
    segments_wp = tw.array.split(flat_wp, offsets_wp)

    packed_wp, packed_offsets_wp = tw.array.pack_1d_arrays(segments_wp, copy=False)
    assert packed_wp.ptr == flat_wp.ptr
    assert np.array_equal(packed_wp.numpy(), flat_wp.numpy())
    assert np.array_equal(packed_offsets_wp.numpy(), offsets_wp.numpy())
    assert tw.array.concatenate(segments_wp, copy=False).ptr == flat_wp.ptr
    # The default still copies, so a caller that writes into the result is unaffected.
    assert tw.array.pack_1d_arrays(segments_wp)[0].ptr != flat_wp.ptr

    for label, sequence in (
        ("out of order", segments_wp[::-1]),
        ("a gap between them", [segments_wp[0], segments_wp[2]]),
        ("separate allocations", [wp.clone(segment) for segment in segments_wp]),
    ):
        rebuilt_wp, _ = tw.array.pack_1d_arrays(sequence, copy=False)
        assert rebuilt_wp.ptr != sequence[0].ptr, label
        assert np.array_equal(
            rebuilt_wp.numpy(), np.concatenate([piece.numpy() for piece in sequence])
        ), label


def test_pack_1d_arrays_copy_false_aliases_a_boundary_loop_pack(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Triwarp against triwarp: the round trip the flag exists for, on a real mesh.

    ``boundary_loops`` slices one packed buffer into per-rim views and both loop measures pack them
    straight back, which on ``dragon``'s 407 rims was 407 ``warp.copy`` calls rebuilding a buffer
    that already existed. The oracle for the values is the copying default; what this adds is that
    the free path fires on loops nobody constructed by hand, which the synthetic case above cannot
    show. That the measures themselves are unaffected is
    [`test_loop_measures_agree_with_the_single_loop_forms`]'s job.
    """
    _mesh_tm, mesh_wp = half_torus
    loops_wp = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    assert len(loops_wp) > 1

    packed_wp, offsets_wp = tw.array.pack_1d_arrays(loops_wp, copy=False)
    assert packed_wp.ptr == loops_wp[0].ptr
    assert np.array_equal(packed_wp.numpy(), tw.array.pack_1d_arrays(loops_wp)[0].numpy())
    assert np.array_equal(
        packed_wp.numpy(), np.concatenate([loop_wp.numpy() for loop_wp in loops_wp])
    )
    assert offsets_wp.numpy()[0] == 0
    assert offsets_wp.numpy()[-1] == packed_wp.size


def test_split_views_share_storage_and_copies_do_not(device: str) -> None:
    """Default segments alias the packed buffer; ``copy=True`` detaches them."""
    flat_wp = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array([0, 2, 6], dtype=np.int32), dtype=wp.int32, device=device)

    views = tw.array.split(flat_wp, offsets_wp)
    copies = tw.array.split(flat_wp, offsets_wp, copy=True)
    flat_wp.fill_(9)

    assert np.array_equal(views[0].numpy(), np.array([9, 9], dtype=np.int32))
    assert np.array_equal(copies[0].numpy(), np.array([0, 1], dtype=np.int32))
    assert np.array_equal(copies[1].numpy(), np.array([2, 3, 4, 5], dtype=np.int32))


def test_split_rejects_bad_offsets(device: str) -> None:
    """``[0, 2]`` over four elements is the unterminated form, and is rejected like the rest."""
    flat_wp = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    for bad in ([1, 2, 4], [0, 3, 2, 4], [0, 5], [0, 2], [0]):
        offsets_wp = wp.array(np.array(bad, dtype=np.int32), dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="offsets must be total-terminated"):
            tw.array.split(flat_wp, offsets_wp)
    with pytest.raises(ValueError, match="total-terminated"):
        tw.array.split(flat_wp, warp_empty(0, wp.int32, device))


def test_split_empty_offsets(device: str) -> None:
    """``[0]`` describes no segments, over an empty buffer."""
    empty_wp = warp_empty(0, wp.int32, device)
    assert tw.array.split(empty_wp, wp.zeros(1, dtype=wp.int32, device=device)) == []


@pytest.mark.parametrize("copy", [False, True])
def test_split_accepts_host_offsets(device: str, copy: bool) -> None:
    """
    Triwarp against triwarp: host bounds split exactly as the same bounds on the device do.

    The device offsets carry the ``numpy.split`` oracle through the tests above; the host form
    exists for a caller that already holds the bounds, and must validate the same way.
    """
    flat_wp = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    bounds = [0, 2, 2, 6]
    offsets_wp = wp.array(np.array(bounds, dtype=np.int32), dtype=wp.int32, device=device)

    segments_host = tw.array.split(flat_wp, bounds, copy=copy)
    segments_device = tw.array.split(flat_wp, offsets_wp, copy=copy)

    assert [s.numpy().tolist() for s in segments_host] == [
        s.numpy().tolist() for s in segments_device
    ]
    assert [s.numpy().tolist() for s in segments_host] == [[0, 1], [], [2, 3, 4, 5]]
    with pytest.raises(ValueError, match="offsets must be total-terminated"):
        tw.array.split(flat_wp, [0, 2])


@pytest.mark.parametrize("copy", [False, True])
def test_split_trailing_empty_segment(device: str, copy: bool) -> None:
    """A segment that is empty *at the end* of the buffer splits (Warp rejects ``arr[n:n]``)."""
    flat_wp = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array([0, 4, 4, 4], dtype=np.int32), dtype=wp.int32, device=device)

    segments_wp = tw.array.split(flat_wp, offsets_wp, copy=copy)

    assert [segment.size for segment in segments_wp] == [4, 0, 0]
    assert np.array_equal(segments_wp[0].numpy(), np.arange(4, dtype=np.int32))


# Every dtype ``split`` is reached with in the package: index loops and neighbour lists
# (``int32``), neighbour distances (``float32``), mesh components and traced polylines
# (``vec3``), plus ``float64`` and ``vec2`` for the widths in between.
_SPLIT_LAYOUTS: dict[type, tuple[type, int]] = {
    wp.int32: (np.int32, 1),
    wp.float32: (np.float32, 1),
    wp.float64: (np.float64, 1),
    wp.vec2: (np.float32, 2),
    wp.vec3: (np.float32, 3),
}
_SPLIT_DTYPES = list(_SPLIT_LAYOUTS)


def _warp_array_of(
    dtype: type[DType], n: int, device: str, *, requires_grad: bool = False
) -> wp.array[DType]:
    """Build a length-``n`` array of ``dtype`` whose every scalar component is distinct."""
    scalar_np, width = _SPLIT_LAYOUTS[dtype]
    values_np = np.arange(n * width, dtype=scalar_np)
    shaped_np = values_np.reshape(n, width) if width > 1 else values_np
    return wp.array(shaped_np, dtype=dtype, device=device, requires_grad=requires_grad)


@pytest.mark.parametrize("dtype", _SPLIT_DTYPES, ids=lambda d: d.__name__)
def test_segment_views_match_real_slices(device: str, dtype: type) -> None:
    """
    Triwarp against Warp: every view ``split`` builds carries exactly a real slice's state.

    ``split`` stamps its views from one template slice's ``__dict__`` rather than slicing per
    segment, so the claim is that the two objects are indistinguishable. It is checked key by key
    -- the same key set, and equal values everywhere except ``_ref``, which must be the base array
    itself -- so a Warp release that adds, drops or renames per-view state fails here. The bounds
    include an interior empty segment, which Warp can slice, and a trailing one, which it cannot
    (``arr[n:n]`` raises): that one is compared against an interior empty slice moved one past the
    last element.
    """
    n = 37
    base_wp = _warp_array_of(dtype, n, device)
    starts = [0, 5, 5, 6, 20, 37]
    bounds = [*starts, n]
    offsets_wp = wp.array(np.array(bounds, dtype=np.int32), dtype=wp.int32, device=device)

    views_wp = tw.array.split(base_wp, offsets_wp)

    assert len(views_wp) == len(starts)
    stride = int(cast("int", base_wp.strides[0]))
    for view_wp, begin, end in zip(views_wp, bounds[:-1], bounds[1:], strict=True):
        if end > begin or end < n:
            slice_state = dict(base_wp[begin:end].__dict__)
        else:
            slice_state = dict(base_wp[0:0].__dict__)
            slice_state["ptr"] = int(cast("int", base_wp.ptr)) + begin * stride
        view_state = view_wp.__dict__
        assert type(view_wp) is type(base_wp)
        assert sorted(view_state) == sorted(slice_state)
        assert view_state["_ref"] is base_wp
        assert slice_state["_ref"] is base_wp
        for key, value in slice_state.items():
            if key != "_ref":
                assert view_state[key] == value, key
        assert np.array_equal(view_wp.numpy(), base_wp.numpy()[begin:end])


@pytest.mark.parametrize("dtype", _SPLIT_DTYPES, ids=lambda d: d.__name__)
def test_segment_views_keep_their_base_alive_and_write_through(device: str, dtype: type) -> None:
    """
    Triwarp against NumPy: stamped views outlive their base and a kernel writing one writes it.

    The base is dropped and the pool churned before the views are read, so a view that did not hold
    its base would read recycled memory. ``wp.copy`` into a view and ``wp.clone`` of the trailing
    empty one exercise the pointer every stamped view is launched with.
    """
    n = 29
    base_wp = _warp_array_of(dtype, n, device)
    expected_np = base_wp.numpy().copy()
    starts = [0, 7, 7, 18, 29]
    offsets_wp = wp.array(np.array([*starts, n], dtype=np.int32), dtype=wp.int32, device=device)
    views_wp = tw.array.split(base_wp, offsets_wp)

    wp.copy(views_wp[2], views_wp[3], count=7)  # elements 18..24 over elements 7..13
    wp.synchronize_device(device)
    del base_wp
    gc.collect()
    _churn = [wp.zeros(n * 4, dtype=dtype, device=device) for _ in range(8)]

    trailing_wp = wp.clone(views_wp[-1])
    assert trailing_wp.size == 0
    bounds = [*starts, n]
    expected_np[7:14] = expected_np[18:25]
    for view_wp, begin, end in zip(views_wp, bounds[:-1], bounds[1:], strict=True):
        assert np.array_equal(view_wp.numpy(), expected_np[begin:end])
    del _churn


@pytest.mark.parametrize("dtype", [*_SPLIT_DTYPES, wp.uint8], ids=lambda d: d.__name__)
def test_split_copies_past_the_one_launch_threshold(device: str, dtype: type) -> None:
    """
    Triwarp against NumPy: ``copy=True`` over enough segments for the one-launch copy is exact.

    Forty segments -- past ``PACK_SEGMENTS_KERNEL_FROM`` -- with an interior and a trailing empty
    one, each checked against the NumPy slice and for owning its storage: no copy may point into
    the base, and rewriting the base afterwards must leave every copy unchanged. ``uint8``, whose
    elements are not whole 4-byte words, is the per-segment clone fallback at the same count.
    """
    assert tw.array.PACK_SEGMENTS_KERNEL_FROM <= 40
    layouts = {**_SPLIT_LAYOUTS, wp.uint8: (np.uint8, 1)}
    scalar_np, width = layouts[dtype]
    n = 211
    values_np = (np.arange(n * width) % 251).astype(scalar_np)
    shaped_np = values_np.reshape(n, width) if width > 1 else values_np
    base_wp = wp.array(shaped_np, dtype=dtype, device=device)
    starts = np.linspace(0, n - 11, 38).astype(np.int32).tolist()
    starts = [*starts[:5], starts[5], *starts[5:], n]  # an interior and a trailing empty segment
    assert len(starts) == 40
    offsets_wp = wp.array(np.array([*starts, n], dtype=np.int32), dtype=wp.int32, device=device)

    copies_wp = tw.array.split(base_wp, offsets_wp, copy=True)
    base_begin = int(cast("int", base_wp.ptr))
    base_end = base_begin + n * int(cast("int", base_wp.strides[0]))
    base_wp.zero_()

    bounds = [*starts, n]
    assert len(copies_wp) == len(starts)
    for copy_wp, begin, end in zip(copies_wp, bounds[:-1], bounds[1:], strict=True):
        assert copy_wp.dtype == dtype
        assert copy_wp.size == end - begin
        assert copy_wp._ref is None  # pyright: ignore[reportPrivateUsage]
        if end > begin:
            assert not base_begin <= int(cast("int", copy_wp.ptr)) < base_end
        assert np.array_equal(copy_wp.numpy(), shaped_np[begin:end])


def test_segment_views_fall_back_for_grad_and_strided_bases(device: str) -> None:
    """
    Triwarp against NumPy: the bases a template cannot describe still split correctly.

    A ``requires_grad`` base yields non-empty segments that carry their own ``grad`` view, as a
    real slice does, and a strided base (every other element) yields views reading the right
    elements.
    """
    starts = [0, 3, 3, 8]
    offsets_wp = wp.array(np.array([*starts, 10], dtype=np.int32), dtype=wp.int32, device=device)
    grad_wp = _warp_array_of(wp.float32, 10, device, requires_grad=True)
    grad_views_wp = tw.array.split(grad_wp, offsets_wp)
    assert all(
        view.requires_grad and view.grad is not None for view in grad_views_wp if view.size > 0
    )
    first_grad_wp, base_grad_wp = grad_views_wp[0].grad, grad_wp.grad
    assert first_grad_wp is not None
    assert base_grad_wp is not None
    assert int(cast("int", first_grad_wp.ptr)) == int(cast("int", base_grad_wp.ptr))

    wide_wp = _warp_array_of(wp.int32, 20, device)
    strided_wp = twt.as_dense(wide_wp[::2])
    assert not strided_wp.is_contiguous
    strided_views_wp = tw.array.split(strided_wp, offsets_wp)
    strided_np = wide_wp.numpy()[::2]
    for view_wp, begin, end in zip(strided_views_wp, starts, [*starts[1:], 10], strict=True):
        assert np.array_equal(view_wp.numpy(), strided_np[begin:end])


def test_split_row_size_counts_rows(device: str) -> None:
    """Class A: ``row_size=3`` splits a flat face buffer at ``3 * offsets``, as numpy does."""
    flat_np = np.arange(3 * 7, dtype=np.int32)
    flat_wp = wp.array(flat_np, dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array([0, 2, 2, 5, 7], dtype=np.int32), dtype=wp.int32, device=device)

    segments_wp = tw.array.split(flat_wp, offsets_wp, row_size=3)

    segments_np = np.split(flat_np, [6, 6, 15])
    assert len(segments_wp) == len(segments_np)
    for segment_wp, segment_np in zip(segments_wp, segments_np, strict=True):
        assert np.array_equal(segment_wp.numpy(), segment_np)
    with pytest.raises(ValueError, match="row_size"):
        tw.array.split(flat_wp, offsets_wp, row_size=2)
    with pytest.raises(ValueError, match="offsets must be total-terminated"):
        tw.array.split(
            flat_wp,
            wp.array(np.array([0, 8], dtype=np.int32), dtype=wp.int32, device=device),
            row_size=3,
        )


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"),
    [(wp.float32, np.float32), (wp.float64, np.float64), (wp.vec3, np.float32)],
    ids=["float32", "float64", "vec3"],
)
def test_allclose_matches_numpy(device: str, dtype_wp: type, dtype_np: type) -> None:
    """The tolerances must instantiate at the input's precision, not always at ``float32``."""
    rng = np.random.default_rng(95)
    shape = (6, 3) if dtype_wp is wp.vec3 else (18,)
    a_np = rng.standard_normal(shape).astype(dtype_np)
    a_wp = wp.array(a_np, dtype=dtype_wp, device=device)
    close_wp = wp.array(a_np + dtype_np(1e-9), dtype=dtype_wp, device=device)
    far_wp = wp.array(a_np + dtype_np(1e-2), dtype=dtype_wp, device=device)
    assert tw.array.allclose(a_wp, close_wp) == bool(np.allclose(a_np, a_np + dtype_np(1e-9)))
    assert tw.array.allclose(a_wp, far_wp) == bool(np.allclose(a_np, a_np + dtype_np(1e-2)))


def test_allclose_empty_is_true(device: str) -> None:
    empty_wp = warp_empty(0, wp.vec3, device)
    assert tw.array.allclose(empty_wp, empty_wp) is True


def test_allclose_rejects_mismatched_dtypes(device: str) -> None:
    a_wp = wp.zeros(3, dtype=wp.float32, device=device)
    b_wp = wp.zeros(3, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="matching dtypes"):
        tw.array.allclose(a_wp, b_wp)


def test_allclose_rejects_mismatched_devices() -> None:
    """
    Triwarp against triwarp: a public two-array function must reject a cross-device call.

    Not a library comparison: no reference library shares Warp's device model. This pins the
    public-boundary contract every function wired to ``_device.require_same_device`` shares --
    ``allclose`` stands in for the family. See
    ``test_require_same_device_flags_a_mismatch_and_ignores_none`` below for the shared helper
    itself.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to construct a mismatch")
    a_wp = wp.zeros(3, dtype=wp.float32, device="cpu")
    b_wp = wp.zeros(3, dtype=wp.float32, device="cuda:0")
    with pytest.raises(RuntimeError, match="one device"):
        tw.array.allclose(a_wp, b_wp)


@pytest.mark.parity("sort_and_argsort", "numpy")
def test_sort_and_argsort_distinct_keys(device: str) -> None:
    """Class A: with distinct keys the permutation is unique, so it must equal ``numpy.argsort``."""
    rng = np.random.default_rng(31)
    keys_np = rng.permutation(64).astype(np.int32)
    keys_wp = wp.array(keys_np, dtype=wp.int32, device=device)
    sorted_keys_wp, order_wp = tw.array.sort_and_argsort(keys_wp)
    assert np.array_equal(sorted_keys_wp.numpy(), np.sort(keys_np))
    assert np.array_equal(order_wp.numpy(), np.argsort(keys_np).astype(np.int32))


def test_sort_and_argsort_duplicate_keys(device: str) -> None:
    """
    With ties the radix sort is not documented as stable, so only the permutation property holds.

    ``sorted_keys[i] == keys[order[i]]`` and ``order`` is a permutation of ``0..n-1`` -- that is
    everything the contract promises, and it is what distinguishes a correct sort from one that
    dropped or duplicated an entry.
    """
    rng = np.random.default_rng(32)
    keys_np = rng.integers(0, 8, size=64).astype(np.int32)
    keys_wp = wp.array(keys_np, dtype=wp.int32, device=device)
    sorted_keys_wp, order_wp = tw.array.sort_and_argsort(keys_wp)
    assert np.array_equal(sorted_keys_wp.numpy(), np.sort(keys_np))
    assert np.array_equal(np.sort(order_wp.numpy()), np.arange(64, dtype=np.int32))
    assert np.array_equal(keys_np[order_wp.numpy()], sorted_keys_wp.numpy())


def test_sort_and_argsort_empty(device: str) -> None:
    keys_wp = warp_empty(0, wp.int32, device)
    sorted_keys_wp, order_wp = tw.array.sort_and_argsort(keys_wp)
    assert sorted_keys_wp.shape == (0,)
    assert order_wp.shape == (0,)


def test_pack_1d_wp_arrays(device: str):
    parts = [
        wp.array([1, 2, 3], dtype=wp.int32, device=device),
        wp.array([], dtype=wp.int32, device=device),
        wp.array([4], dtype=wp.int32, device=device),
    ]
    flat, offsets = tw.array.pack_1d_arrays(parts)
    assert flat.device == device
    assert offsets.device == device
    assert np.array_equal(flat.numpy(), np.array([1, 2, 3, 4], dtype=np.int32))
    assert np.array_equal(offsets.numpy(), np.array([0, 3, 3, 4], dtype=np.int32))


def test_pack_1d_wp_arrays_vec3(device: str):
    parts = [
        wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device),
        wp.array([wp.vec3(2.0, 2.0, 2.0)], dtype=wp.vec3, device=device),
    ]
    flat, offsets = tw.array.pack_1d_arrays(parts)
    expected_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    flat_wp_np = flat.numpy().reshape(-1, 3)
    assert np.allclose(flat_wp_np, expected_np, rtol=1e-5, atol=1e-5)
    assert np.array_equal(offsets.numpy(), np.array([0, 2, 3], dtype=np.int32))


def test_pack_1d_wp_arrays_dtype_mismatch(device: str):
    parts = [
        wp.array([1], dtype=wp.int32, device=device),
        wp.array([2.0], dtype=wp.float32, device=device),
    ]
    with pytest.raises(ValueError, match="same dtype"):
        tw.array.pack_1d_arrays(parts)


def test_sort_rows(device: str):
    rng = np.random.default_rng(42)
    data = rng.random(size=(32, 4), dtype=np.float32)
    sorted_data_np = np.sort(data, axis=1)

    data_wp = twt.as_array2d(wp.array(data, dtype=wp.float32, device=device), wp.float32)
    tw.array.sort_rows(data_wp)
    assert np.array_equal(data_wp.numpy(), sorted_data_np)


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64, wp.mat22d])
def test_triplet_buffers(dtype: type, device: str) -> None:
    """Three same-length buffers on the requested device, int32 indices and ``dtype`` values."""
    n_triplets = 12
    rows_wp, cols_wp, values_wp = tw.array.triplet_buffers(n_triplets, dtype, device)

    for buffer_wp in (rows_wp, cols_wp, values_wp):
        assert buffer_wp.shape == (n_triplets,)
        assert wp.get_device(str(buffer_wp.device)) == wp.get_device(device)
    assert rows_wp.dtype == wp.int32
    assert cols_wp.dtype == wp.int32
    assert values_wp.dtype == dtype


def test_triplet_buffers_feed_a_sparse_build(device: str) -> None:
    """The buffers round-trip through ``bsr_from_triplets``: a 3x3 identity written by hand."""
    rows_wp, cols_wp, values_wp = tw.array.triplet_buffers(3, wp.float64, device)
    wp.copy(rows_wp, wp.array([0, 1, 2], dtype=wp.int32, device=device))
    wp.copy(cols_wp, wp.array([0, 1, 2], dtype=wp.int32, device=device))
    wp.copy(values_wp, wp.array([1.0, 1.0, 1.0], dtype=wp.float64, device=device))

    # Warp's stub types the index arrays as ``Array[int]``, which no Warp dtype satisfies.
    matrix_wp = wps.bsr_from_triplets(3, 3, rows_wp, cols_wp, values_wp)  # pyright: ignore[reportArgumentType]
    assert twt.has_blocks(matrix_wp, wp.float64)
    assert np.array_equal(matrix_wp.offsets.numpy()[:4], np.arange(4))
    assert np.allclose(matrix_wp.values.numpy().reshape(-1), np.ones(3))


def test_triplet_buffers_zero_length(device: str) -> None:
    """A zero-entry build is a real case (an empty mesh), so the buffers must allocate empty."""
    rows_wp, cols_wp, values_wp = tw.array.triplet_buffers(0, wp.float32, device)
    for buffer_wp in (rows_wp, cols_wp, values_wp):
        assert buffer_wp.shape == (0,)


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64, wp.int32])
def test_csr_from_triplets_matches_warp_sparse(dtype: type, device: str) -> None:
    """
    ``csr_from_triplets`` assembles what ``warp.sparse.bsr_from_triplets`` assembles.

    Not a library comparison: Warp's own builder is the oracle for the replacement. The triplets
    carry duplicates, indices outside the matrix on both axes, and empty rows. The pattern must be
    identical and the values equal to rounding -- bit-identical on CUDA, where both sum in triplet
    order; Warp's CPU path sums in another order.
    """
    rng = np.random.default_rng(3)
    n_rows, n_cols, m = 40, 30, 500
    rows_np = rng.integers(-2, n_rows + 2, m).astype(np.int32)
    cols_np = rng.integers(-2, n_cols + 2, m).astype(np.int32)
    values_np = (rng.standard_normal(m) * 10).astype(wp.dtype_to_numpy(dtype))
    rows_np[rows_np == 7] = 8  # an empty row
    rows_wp = wp.array(rows_np, dtype=wp.int32, device=device)
    cols_wp = wp.array(cols_np, dtype=wp.int32, device=device)
    values_wp = wp.array(values_np, dtype=dtype, device=device)
    warp_wp = wps.bsr_from_triplets(
        n_rows, n_cols, rows_wp, cols_wp, values_wp, prune_numerical_zeros=False
    )
    ours_wp = tw.array.csr_from_triplets(n_rows, n_cols, rows_wp, cols_wp, values_wp)
    assert isinstance(warp_wp, twt.BsrMatrix)
    n_entries = warp_wp.nnz_sync()
    assert ours_wp.nnz_sync() == n_entries
    assert np.array_equal(ours_wp.offsets.numpy(), warp_wp.offsets.numpy())
    assert np.array_equal(ours_wp.columns.numpy()[:n_entries], warp_wp.columns.numpy()[:n_entries])
    assert np.allclose(
        ours_wp.values.numpy()[:n_entries], warp_wp.values.numpy()[:n_entries], rtol=1e-6, atol=1e-5
    )


def test_csr_from_triplets_prunes_assembled_zeros(device: str) -> None:
    """
    Pruning drops every entry whose *sum* is zero, where ``bsr_from_triplets`` keeps it.

    Not a library comparison: Warp's builder is the oracle, and the two are meant to differ in one
    way. Warp drops zero-valued triplets before summing, so an entry whose triplets cancel stays,
    holding zero; ``csr_from_triplets`` drops it. Every entry Warp keeps that ours drops must
    therefore hold zero there, and every entry ours keeps must match Warp's.
    """
    # (0, 0): 1 + 2; (1, 1): 3 - 3 cancels; (2, 0): a lone zero triplet; (2, 2): 4.
    rows_wp = wp.array([0, 0, 1, 1, 2, 2], dtype=wp.int32, device=device)
    cols_wp = wp.array([0, 0, 1, 1, 0, 2], dtype=wp.int32, device=device)
    values_wp = wp.array([1.0, 2.0, 3.0, -3.0, 0.0, 4.0], dtype=wp.float32, device=device)
    ours_wp = tw.array.csr_from_triplets(
        3, 3, rows_wp, cols_wp, values_wp, prune_numerical_zeros=True
    )
    warp_wp = wps.bsr_from_triplets(3, 3, rows_wp, cols_wp, values_wp, prune_numerical_zeros=True)
    assert twt.has_blocks(warp_wp, wp.float32)
    n_ours, n_warp = ours_wp.nnz_sync(), warp_wp.nnz_sync()
    assert np.array_equal(ours_wp.offsets.numpy(), [0, 1, 1, 2])
    assert np.array_equal(ours_wp.columns.numpy()[:n_ours], [0, 2])
    assert np.array_equal(ours_wp.values.numpy()[:n_ours], [3.0, 4.0])

    def entries(matrix: twt.BsrMatrix[wp.float32], n: int) -> dict[tuple[int, int], float]:
        offsets = matrix.offsets.numpy()
        rows = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
        return dict(
            zip(
                zip(rows.tolist(), matrix.columns.numpy()[:n].tolist(), strict=True),
                matrix.values.numpy()[:n].tolist(),
                strict=True,
            )
        )

    ours, warp = entries(ours_wp, n_ours), entries(warp_wp, n_warp)
    assert all(warp.get(key) == value for key, value in ours.items())
    dropped = [value for key, value in warp.items() if key not in ours]
    assert dropped == [0.0]  # (1, 1): Warp keeps the cancelled entry


@pytest.mark.parametrize("data", [None, np.arange(1, 13, dtype=np.int32)])
def test_index_sparse(data: npt.NDArray[np.int32] | None, device: str):
    n_rows = 4
    indices = np.array([[0, 1, 2], [0, 3, 1], [1, 2, 3], [0, 2, 3]])

    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = twt.as_array2d(wp.array(indices, dtype=wp.int32, device=device), wp.int32)
    data_wp = wp.array(data, dtype=wp.int32, device=device) if data is not None else None

    result_wp = tw.array.index_sparse(n_rows, indices_wp, data_wp)
    assert result_wp.values.dtype == (data_wp.dtype if data_wp is not None else wp.float32)
    assert np.array_equal(result_wp.offsets.numpy(), result_np.indptr)
    assert np.array_equal(result_wp.values.numpy(), result_np.data)


def test_index_sparse_repeated_indices(device: str):
    n_rows = 4
    indices = np.array([[0, 1, 0], [3, 3, 1], [1, 2, 3]])
    data = np.ones(9, dtype=np.int32)
    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = twt.as_array2d(wp.array(indices, dtype=wp.int32, device=device), wp.int32)
    data_wp = wp.array(data, dtype=wp.int32, device=device)

    result_wp = tw.array.index_sparse(n_rows, indices_wp, data_wp, dtype=wp.float64)
    assert result_wp.values.dtype == wp.float64
    result_csr = scipy.sparse.csr_matrix(
        (result_wp.values.numpy(), result_wp.columns.numpy(), result_wp.offsets.numpy()),
        shape=result_wp.shape,
    )
    assert np.array_equal(result_csr.todense(), result_np.todense())


def test_index_sparse_uses_the_input_device(device: str) -> None:
    """
    The implicit ``wp.ones`` lands on ``indices``' device, not Warp's current one.

    Needs two devices to say anything, so it skips without CUDA. Before the ``device=`` was
    supplied, ``bsr_from_triplets`` rejected the mixed set with "Rows and columns must reside on
    the destination matrix device, got cuda:0, cuda:0 and cpu" -- and every other test in this file
    passed, because each one runs with its arrays' device already current.
    """
    if not wp.get_device(device).is_cuda:
        pytest.skip("needs a second device to distinguish 'input' from 'current'")
    indices_wp = twt.as_array2d(
        wp.array(np.array([[0, 1, 2], [1, 2, 3]]), dtype=wp.int32, device=device), wp.int32
    )
    with wp.ScopedDevice("cpu"):
        matrix = tw.array.index_sparse(4, indices_wp)
    assert str(matrix.values.device) == str(indices_wp.device)


def test_isin_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    elements_np = rng.integers(0, 20, size=50, dtype=np.int32)
    test_np = rng.choice(20, size=8, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_2d(device: str) -> None:
    rng = np.random.default_rng(7)
    elements_np = rng.integers(0, 15, size=(12, 3), dtype=np.int32)
    test_np = rng.choice(15, size=5, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_empty_test(device: str) -> None:
    elements_np = np.array([0, 1, 2, 3], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = warp_empty(0, wp.int32, device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_ref_np = np.zeros_like(elements_np, dtype=bool)
    assert np.array_equal(mask_wp.numpy(), mask_ref_np)


@pytest.mark.parametrize("shape", [(4, 3, 2), (2, 2, 3, 5)], ids=["rank3", "rank4"])
def test_isin_higher_rank(device: str, shape: tuple[int, ...]) -> None:
    """Membership is a per-element predicate, so any rank round-trips through flatten/reshape."""
    rng = np.random.default_rng(13)
    elements_np = rng.integers(0, 15, size=shape, dtype=np.int32)
    test_np = rng.choice(15, size=5, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    assert mask_wp.shape == shape
    assert np.array_equal(mask_wp.numpy(), np.isin(elements_np, test_np))


def test_isin_sparse_large_indices(device: str) -> None:
    """Forces sort + binary-search path (max index >> len(test_elements))."""
    elements_np = np.array([1, 1_000_000, 2, 999_999, 3], dtype=np.int32)
    test_np = np.array([1, 2, 3], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_negative_values(device: str) -> None:
    """
    Regression: the membership table is anchored at the global minimum, not at zero.

    Anchored at zero this returned a *silent wrong answer* -- ``mark_membership_mask`` drops a
    negative test value as out of range, so ``-5`` read back as absent while ``3`` was found.
    """
    elements_np = np.array([-5, 3, -1, 4, -5], dtype=np.int32)
    test_np = np.array([3, -5], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.array.isin(elements_wp, test_wp).numpy(), np.isin(elements_np, test_np)
    )


_ISIN_DTYPES = [
    (wp.int8, np.int8),
    (wp.uint8, np.uint8),
    (wp.int16, np.int16),
    (wp.uint16, np.uint16),
    (wp.int32, np.int32),
    (wp.uint32, np.uint32),
    (wp.int64, np.int64),
    (wp.uint64, np.uint64),
]


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), _ISIN_DTYPES, ids=lambda d: getattr(d, "__name__", "")
)
def test_isin_integer_dtypes(device: str, dtype_wp: type, dtype_np: type) -> None:
    """Every Warp integer dtype, on the membership-table branch. Sub-32-bit ones are widened."""
    info = np.iinfo(dtype_np)
    low, high = max(int(info.min), -60), min(int(info.max), 60)
    rng = np.random.default_rng(19)
    elements_np = rng.integers(low, high + 1, size=97).astype(dtype_np)
    test_np = rng.integers(low, high + 1, size=11).astype(dtype_np)

    elements_wp = wp.array(elements_np, dtype=dtype_wp, device=device)
    test_wp = wp.array(test_np, dtype=dtype_wp, device=device)
    assert np.array_equal(
        tw.array.isin(elements_wp, test_wp).numpy(), np.isin(elements_np, test_np)
    )


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), _ISIN_DTYPES, ids=lambda d: getattr(d, "__name__", "")
)
def test_isin_integer_dtypes_sparse(device: str, dtype_wp: type, dtype_np: type) -> None:
    """The same dtypes on the sort + binary-search branch, where the span dwarfs the test set."""
    span = min(int(np.iinfo(dtype_np).max), 2**40)
    test_np = np.array([0, span // 3, span // 2, span], dtype=dtype_np)
    elements_np = np.array([0, 1, span // 2, span], dtype=dtype_np)

    elements_wp = wp.array(elements_np, dtype=dtype_wp, device=device)
    test_wp = wp.array(test_np, dtype=dtype_wp, device=device)
    assert np.array_equal(
        tw.array.isin(elements_wp, test_wp).numpy(), np.isin(elements_np, test_np)
    )


def test_isin_rejects_mismatched_and_non_integer_dtypes(device: str) -> None:
    """Both arrays must share one integer dtype: the kernels are instantiated per dtype."""
    elements_wp = wp.array(np.arange(4, dtype=np.int64), dtype=wp.int64, device=device)
    with pytest.raises(TypeError, match="one dtype"):
        tw.array.isin(elements_wp, wp.array(np.array([1], np.int32), dtype=wp.int32, device=device))
    floats_wp = wp.array(np.zeros(3, np.float32), dtype=wp.float32, device=device)
    with pytest.raises(TypeError, match="integer dtype"):
        tw.array.isin(floats_wp, floats_wp)


@pytest.mark.parity("flatnonzero", "numpy")
def test_flatnonzero(device: str) -> None:
    """Class A: the same indices ``np.flatnonzero`` returns, for the same mask."""
    rng = np.random.default_rng(11)
    mask_np = rng.choice([False, True], size=64, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    indices_wp = tw.array.flatnonzero(mask_wp)
    indices_ref_np = np.flatnonzero(mask_np).astype(np.int32)
    assert np.array_equal(indices_wp.numpy(), indices_ref_np)


def test_flatnonzero_empty(device: str) -> None:
    mask_wp = wp.array(np.zeros(8, dtype=bool), dtype=wp.bool, device=device)
    indices_wp = tw.array.flatnonzero(mask_wp)
    assert indices_wp.shape == (0,)


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"),
    [(wp.int32, np.int32), (wp.int8, np.int8), (wp.float32, np.float32)],
    ids=["int32", "int8", "float32"],
)
def test_flatnonzero_nonboolean(device: str, dtype_wp: type, dtype_np: type) -> None:
    """
    Any non-zero value selects its index, matching ``numpy.flatnonzero``.

    Values above one and negative values both count -- the flag pass must map to 0/1 rather than
    cast, or the prefix sum would total the values instead of counting them.
    """
    values_np = np.array([0, 3, 0, -2, 1, 0, 7, -1], dtype=dtype_np)
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    indices_wp = tw.array.flatnonzero(values_wp)
    assert np.array_equal(indices_wp.numpy(), np.flatnonzero(values_np).astype(np.int32))


def test_flatnonzero_rejects_rank2(device: str) -> None:
    values_wp = wp.array(np.zeros((3, 4), dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="1D"):
        tw.array.flatnonzero(values_wp)


@pytest.mark.parametrize("first", [False, True])
def test_flatnonzero_matches_numpy_across_scan_blocks(device: str, first: bool) -> None:
    """
    Class A: ``flatnonzero`` equals ``numpy.flatnonzero`` on a mask wide enough to scan in blocks.

    The flags are scanned in place and each selection is recovered as a step between neighbouring
    scan values, so the two things this pins are the in-place scan across many CUDA scan tiles and
    the first element, which has no left neighbour -- hence both values of ``mask[0]``. The
    mask-to-ranks form shares the in-place scan and is checked against the same mask.
    """
    rng = np.random.default_rng(29)
    mask_np = rng.random(1_000_003) < 0.4
    mask_np[0] = first
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.array_equal(tw.array.flatnonzero(mask_wp).numpy(), np.flatnonzero(mask_np))
    ranks_wp, count = tw.array.mask_to_compact_ranks(mask_wp)
    assert count == int(mask_np.sum())
    assert np.array_equal(ranks_wp.numpy(), np.cumsum(mask_np) - mask_np)


@pytest.mark.parametrize("n", [16, 257], ids=["small", "large"])
def test_flatnonzero_indices_to_mask_round_trip(device: str, n: int) -> None:
    """
    The two are inverses: each recovers the other's input.

    ``flatnonzero`` returns ascending unique indices, so the mask direction round-trips exactly
    while the index direction round-trips up to ``numpy.unique`` (duplicate indices mark the same
    slot once).
    """
    rng = np.random.default_rng(23)
    mask_np = rng.random(n) < 0.3
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.array_equal(
        tw.array.indices_to_mask(tw.array.flatnonzero(mask_wp), n).numpy(), mask_np
    )

    indices_np = rng.choice(n, size=n // 4, replace=True).astype(np.int32)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.array.flatnonzero(tw.array.indices_to_mask(indices_wp, n)).numpy(), np.unique(indices_np)
    )


@pytest.mark.parametrize("invert", [False, True])
def test_mask_to_compact_ranks_is_the_exclusive_scan_of_the_mask(device: str, invert: bool) -> None:
    """
    Class A: the map is ``cumsum(mask) - mask`` and the count is the mask's population.

    It is the scatter-side counterpart of [`test_flatnonzero`] -- ``flatnonzero`` says *which*
    elements are selected and this says *where each one lands* -- so the round trip between them is
    asserted too: indexing the map by the selected positions has to give ``0 .. count - 1``.
    """
    mask_np = np.array([True, False, True, True, False, False, True], dtype=bool)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    selected_np = ~mask_np if invert else mask_np

    index_map_wp, count = tw.array.mask_to_compact_ranks(mask_wp, invert=invert)

    assert count == int(selected_np.sum())
    assert np.array_equal(index_map_wp.numpy(), np.cumsum(selected_np) - selected_np)
    positions_np = np.flatnonzero(selected_np)
    assert np.array_equal(index_map_wp.numpy()[positions_np], np.arange(count, dtype=np.int32))


def test_mask_to_compact_ranks_empty(device: str) -> None:
    """An empty mask maps to an empty array and a zero count, without launching a scan."""
    index_map_wp, count = tw.array.mask_to_compact_ranks(warp_empty(0, wp.bool, device))
    assert count == 0
    assert index_map_wp.shape == (0,)


def test_counts_to_offsets(device: str) -> None:
    """Class A: the total-terminated exclusive prefix sum is ``numpy.cumsum`` behind a zero."""
    rng = np.random.default_rng(41)
    counts_np = rng.integers(0, 7, size=32).astype(np.int32)
    counts_wp = wp.array(counts_np, dtype=wp.int32, device=device)
    offsets_wp, total = tw.array.counts_to_offsets(counts_wp)

    assert total == int(counts_np.sum())
    assert np.array_equal(
        offsets_wp.numpy(), np.concatenate([[0], np.cumsum(counts_np)]).astype(np.int32)
    )


def test_counts_to_offsets_empty(device: str) -> None:
    """No counts scan to the one-element ``[0]``, the empty terminated form."""
    counts_wp = warp_empty(0, wp.int32, device)
    offsets_wp, total = tw.array.counts_to_offsets(counts_wp)
    assert total == 0
    assert np.array_equal(offsets_wp.numpy(), np.zeros(1, dtype=np.int32))


def test_remap_indices_passes_negative_sentinels_through(device: str) -> None:
    """
    ``-1`` entries survive the remap unchanged; everything else reads the table.

    This is the behavior that separates ``remap_indices`` from a plain ``gather`` — a gather
    would read out of bounds on the sentinel — and what ``repair``'s sentinel-preserving face
    remaps rely on.
    """
    indices_wp = wp.array(
        np.array([2, -1, 0, 1, -1], dtype=np.int32), dtype=wp.int32, device=device
    )
    remap_wp = wp.array(np.array([10, 11, 12], dtype=np.int32), dtype=wp.int32, device=device)
    remapped_wp = tw.array.remap_indices(indices_wp, remap_wp)
    assert np.array_equal(remapped_wp.numpy(), np.array([12, -1, 10, 11, -1], dtype=np.int32))


def test_remap_indices_empty(device: str) -> None:
    indices_wp = warp_empty(0, wp.int32, device)
    remap_wp = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.array.remap_indices(indices_wp, remap_wp).shape == (0,)


def test_indices_to_mask_empty(device: str) -> None:
    indices_wp = warp_empty(0, wp.int32, device)
    mask_wp = tw.array.indices_to_mask(indices_wp, 5)
    assert np.array_equal(mask_wp.numpy(), np.zeros(5, dtype=bool))


@pytest.mark.parity("gather", "numpy")
def test_gather_1d(device: str) -> None:
    """Class A: the same values ``src[indices]`` returns, which is NumPy's whole answer here."""
    rng = np.random.default_rng(5)
    values_np = rng.integers(0, 1000, size=32, dtype=np.int32)
    indices_np = rng.integers(0, 32, size=10, dtype=np.int32)

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(values_wp, indices_wp)

    assert np.array_equal(gathered_wp.numpy(), values_np[indices_np])


def test_gather_2d_rows(device: str) -> None:
    rng = np.random.default_rng(6)
    rows_np = rng.integers(0, 1000, size=(20, 2), dtype=np.int32)
    indices_np = rng.integers(0, 20, size=7, dtype=np.int32)

    rows_wp = wp.array(rows_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(rows_wp, indices_wp)

    assert gathered_wp.shape == (7, 2)
    assert np.array_equal(gathered_wp.numpy(), rows_np[indices_np])


def test_gather_vec3(device: str) -> None:
    rng = np.random.default_rng(7)
    points_np = rng.standard_normal((16, 3)).astype(np.float32)
    indices_np = rng.integers(0, 16, size=5, dtype=np.int32)

    points_wp = points_to_warp(points_np, device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(points_wp, indices_wp)

    assert np.array_equal(gathered_wp.numpy(), points_np[indices_np])


def test_gather_empty_indices(device: str) -> None:
    rows_wp = wp.array(np.zeros((4, 2), dtype=np.int32), dtype=wp.int32, device=device)
    indices_wp = warp_empty(0, wp.int32, device)
    gathered_wp = tw.array.gather(rows_wp, indices_wp)
    assert gathered_wp.shape == (0, 2)


bitcast_test_data = (
    wp.array([-128, -127, -1, 0, 1, 127], dtype=wp.int8),
    wp.array([255, 0, 254, 1], dtype=wp.uint8),
    wp.array([-32768, -32767, 10, 0, -1, 32767, 32765], dtype=wp.int16),
    wp.array([65535, 32, 65534, 0, 1], dtype=wp.uint16),
    wp.array([-2147483648, -2147483647, 10, 0, -1, 2147483647, 2147483645], dtype=wp.int32),
    wp.array([4294967295, 32, 4294967294, 0, 1], dtype=wp.uint32),
    wp.array(
        [
            -9223372036854775808,
            -9223372036854775807,
            10,
            0,
            -1,
            9223372036854775806,
            9223372036854775807,
        ],
        dtype=wp.int64,
    ),
    wp.array([18446744073709551615, 65535, 18446744073709551614, 131070, 1], dtype=wp.uint64),
    wp.array(
        np.asarray(
            [
                0x0000,  # +0
                0x8000,  # -0
                0x7F80,  # +inf
                0xFF80,  # -inf
                0x7FFF,  # quiet NaN (preserved through float32 widen/narrow on CUDA)
                0x7F7F,  # largest finite
                0xFF7F,  # smallest (most negative) finite
                0x0080,  # smallest subnormal
                0x8100,  # negative subnormal
            ],
            dtype=np.uint16,
        ),
        dtype=wp.bfloat16,
    ),
    wp.array(
        [
            np.finfo(np.float16).min,
            np.finfo(np.float16).max,
            -np.finfo(np.float16).max,
            np.finfo(np.float16).smallest_subnormal,
            -np.finfo(np.float16).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float16,
    ),
    wp.array(
        [
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.finfo(np.float32).smallest_subnormal,
            -np.finfo(np.float32).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float32,
    ),
    wp.array(
        [
            np.finfo(np.float64).min,
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            np.finfo(np.float64).smallest_subnormal,
            -np.finfo(np.float64).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float64,
    ),
)


def test_astype_matches_numpy(device: str) -> None:
    """
    Class A: ``astype`` equals ``numpy.ndarray.astype`` for every conversion the package uses.

    Covers the rank-2 case as well, because the helper allocates from ``values.shape`` rather than
    from ``shape[0]`` -- a rank-1-only implementation would silently truncate a table.
    """
    rng = np.random.default_rng(3)
    values_np = (rng.standard_normal(64) * 100.0).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    assert np.array_equal(tw.array.astype(values_wp, wp.int32).numpy(), values_np.astype(np.int32))
    assert np.allclose(tw.array.astype(values_wp, wp.float64).numpy(), values_np.astype(np.float64))

    flags_np = rng.integers(0, 2, 64).astype(np.int32)
    flags_wp = wp.array(flags_np, dtype=wp.int32, device=device)
    assert np.array_equal(tw.array.astype(flags_wp, wp.bool).numpy(), flags_np.astype(bool))
    bool_wp = wp.array(flags_np.astype(bool), dtype=wp.bool, device=device)
    assert np.array_equal(tw.array.astype(bool_wp, wp.int32).numpy(), flags_np)

    counts_wp = wp.array(flags_np * 7, dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.array.astype(counts_wp, wp.float32).numpy(), (flags_np * 7).astype(np.float32)
    )

    rows_np = rng.integers(0, 50, (16, 3)).astype(np.int32)
    rows_wp = wp.array(rows_np, dtype=wp.int32, device=device)
    rows_out = tw.array.astype(rows_wp, wp.float32)
    assert rows_out.shape == (16, 3)
    assert np.allclose(rows_out.numpy(), rows_np.astype(np.float32))


def test_astype_uses_the_input_device(device: str) -> None:
    """
    The output lands on ``values``' device, not Warp's current one.

    Needs two devices to say anything, so it skips without CUDA: the site this replaced in
    ``array.index_sparse`` allocated with no ``device=`` at all and was correct only because the
    current device happened to match.
    """
    if not wp.get_device(device).is_cuda:
        pytest.skip("needs a second device to distinguish 'input' from 'current'")
    values_wp = wp.array(np.arange(8, dtype=np.float32), dtype=wp.float32, device=device)
    with wp.ScopedDevice("cpu"):
        out = tw.array.astype(values_wp, wp.int32)
    assert str(out.device) == str(values_wp.device)


@pytest.mark.parametrize(
    "data", bitcast_test_data, ids=[a.dtype.__name__ for a in bitcast_test_data]
)
def test_bitcast_int_reciprocity(device: str, data: wp.array[wp.Scalar]):
    as_int = tw.array.bitcast_to_int(cast("wp.array[wp.Scalar]", data.to(device)))
    recovered = tw.array.bitcast_from_int(as_int, data.dtype)
    assert np.array_equal(data.numpy(), recovered.numpy(), equal_nan=True)


@pytest.mark.parametrize(
    "data", bitcast_test_data, ids=[a.dtype.__name__ for a in bitcast_test_data]
)
def test_bitcast_honours_an_explicit_zero_count(device: str, data: wp.array[wp.Scalar]):
    """
    Not a library comparison: ``count=0`` is a length, not an omitted argument.

    Both functions resolved the argument with ``count = count or n``, so a caller asking for an
    empty result got the whole buffer -- ``n`` elements of *stale* bits, since the tail past
    ``copy_count`` is deliberately left uninitialized for radix-sort scratch. The signature already
    spelled the distinction as ``int | None``; only the resolution lost it.

    Parametrized over the same dtype table as the reciprocity test above, because the two functions
    branch on the input's width and the zero path has to hold on every branch. ``count=1`` is the
    control: it separates "honours a zero" from "returns empty for any small count".
    """
    data_wp = cast("wp.array[wp.Scalar]", data.to(device))
    as_int = tw.array.bitcast_to_int(data_wp)

    assert tw.array.bitcast_to_int(data_wp, count=0).shape == (0,)
    assert tw.array.bitcast_from_int(as_int, data.dtype, count=0).shape == (0,)
    assert tw.array.bitcast_to_int(data_wp, count=1).shape == (1,)
    assert tw.array.bitcast_from_int(as_int, data.dtype, count=1).shape == (1,)


def test_trim_to_count_keeps_the_written_prefix_of_every_buffer(device: str) -> None:
    """
    Class A: the atomic-append pattern this finalizes, checked across dtype and rank together.

    Both buffers are indexed by the same counter, so the contract is that they come back the *same*
    length -- trimming them in separate calls is what this function exists to prevent. Trailing
    dimensions are preserved, which is why a ``wp.vec3`` buffer is in the call.
    """
    n_written = 3
    counter_wp = wp.array([n_written], dtype=wp.int32, device=device)
    scalars_np = np.arange(10, dtype=np.int32)
    vectors_np = np.arange(30, dtype=np.float32).reshape(10, 3)

    n_out, (scalars_wp, vectors_wp) = tw.array.trim_to_count(
        counter_wp,
        wp.array(scalars_np, dtype=wp.int32, device=device),
        points_to_warp(vectors_np, device),
    )

    assert n_out == n_written
    assert np.array_equal(scalars_wp.numpy(), scalars_np[:n_written])
    assert np.array_equal(vectors_wp.numpy(), vectors_np[:n_written])
    # A fresh allocation, not a view: writing the source tail must not reach the trimmed copy.
    assert scalars_wp.ptr != counter_wp.ptr


def test_trim_to_count_zero(device: str) -> None:
    """A counter of zero gives empty buffers rather than a zero-length copy of the whole tail."""
    n_out, (trimmed_wp,) = tw.array.trim_to_count(
        wp.zeros(1, dtype=wp.int32, device=device),
        wp.array(np.arange(10, dtype=np.int32), dtype=wp.int32, device=device),
    )
    assert n_out == 0
    assert trimmed_wp.shape == (0,)


def test_isin_max_index_matches_the_inferred_span(device: str) -> None:
    """
    Triwarp against triwarp: the supplied bound reaches the answer the two reductions infer.

    ``max_index`` skips the ``minmax`` pair that would otherwise select the strategy and anchor the
    table, so the inferred path is the oracle for the supplied one -- and [`numpy.isin`][] is the
    oracle for the inferred path in the tests above it.

    The last case is the one that matters: values **at or above** the bound. They are outside what
    the keyword promises, so the answer is documented as wrong rather than raised -- but the
    element-side lookup range-guards its slot, so it must be wrong by reading ``False`` and not by
    reading past the end of the table. Without the guard this case is an out-of-bounds gather,
    which on the CPU device is host-heap corruption (CLAUDE.md section 12.1) rather than a failure
    anything here could catch.
    """
    rng = np.random.default_rng(11)
    elements_np = rng.integers(0, 64, size=500).astype(np.int32)
    test_np = rng.integers(0, 64, size=40).astype(np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)

    inferred_np = tw.array.isin(elements_wp, test_wp).numpy()
    assert inferred_np.any()
    assert not inferred_np.all()
    assert np.array_equal(tw.array.isin(elements_wp, test_wp, max_index=64).numpy(), inferred_np)

    # Rank-2 input, so the reshape path is covered by the supplied branch too.
    rows_wp = wp.array(elements_np.reshape(-1, 5), dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.array.isin(rows_wp, test_wp, max_index=64).numpy(), inferred_np.reshape(-1, 5)
    )

    # A bound the values exceed: everything at or above it reads absent, nothing reads out of range.
    truncated_np = tw.array.isin(elements_wp, test_wp, max_index=32).numpy()
    assert np.array_equal(truncated_np, inferred_np & (elements_np < 32))
    assert truncated_np.any()

    with pytest.raises(ValueError, match="max_index must be positive"):
        tw.array.isin(elements_wp, test_wp, max_index=0)


@pytest.mark.parametrize("wide_side", ["elements", "test_elements"])
def test_isin_max_index_rejects_a_value_that_would_wrap_int32(device: str, wide_side: str) -> None:
    """
    Class A against [`numpy.isin`][]: a 64-bit value far above ``max_index`` reads absent.

    The test above this one bounds ``int32`` values, where the offending value is representable as
    a slot however far outside the table it lies -- so it is the case that *cannot* fail, and it
    was the only one covered. What makes the bound load-bearing is a value whose distance from the
    table anchor does not fit in the ``int32`` slot type: ``2**32 + 5`` is congruent to ``5``, so a
    lookup that narrows before it range-tests places it at slot 5 and reports the membership of a
    value the array does not contain.

    Both sides are parametrized because the shift runs twice with the same helper -- once over
    ``test_elements`` to fill the table and once over ``elements`` to read it -- and each direction
    fabricates a different wrong answer: a wide *element* borrows the flag of its congruent value,
    a wide *test value* plants a flag for a value nobody asked about.

    The inferred path is the control. It never reaches this: it anchors the table at the true
    minimum and only takes the table strategy when the span is small, which is the precondition
    ``max_index`` exists to skip.
    """
    near_np = np.array([5, 7], dtype=np.int64)
    wide_np = np.array([2**32 + 5, 7], dtype=np.int64)
    elements_np, test_np = (wide_np, near_np) if wide_side == "elements" else (near_np, wide_np)
    elements_wp = wp.array(elements_np, dtype=wp.int64, device=device)
    test_wp = wp.array(test_np, dtype=wp.int64, device=device)

    expected_np = np.isin(elements_np, test_np)
    assert not expected_np.all()  # The comparison is not vacuous: exactly one element matches.
    assert np.array_equal(tw.array.isin(elements_wp, test_wp, max_index=100).numpy(), expected_np)
    assert np.array_equal(tw.array.isin(elements_wp, test_wp).numpy(), expected_np)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("index_bound", "trimesh")
def test_index_bound_matches_the_index_maximum(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the count inferred from the face buffer, against the numpy formula.

    ``Trimesh`` has no uncached equivalent -- its vertex count comes from the array it was built
    with -- so the benchmark's "trimesh" row is the stand-in formula ``int(faces.max()) + 1``, and
    that is the reference here. The value is checked against the fixture's actual vertex count too,
    which is the part that would catch an off-by-one that the formula shares.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces

    assert tw.array.index_bound(mesh_wp.indices) == int(faces_np.max()) + 1
    assert tw.array.index_bound(mesh_wp.indices) == len(mesh_tm.vertices)


def test_index_bound_can_check_non_negativity_in_the_same_reduction(device: str) -> None:
    """
    Triwarp against triwarp: ``require_non_negative`` adds a guard, not a different bound.

    Not a library comparison: numpy's ``arr.max() + 1`` is the oracle for the bound itself and is
    checked by ``test_index_bound_matches_the_index_maximum``. What this pins is the pairing the
    hashing callers rely on -- taking the bound and the negative check out of one reduction, so
    they can hand ``grouping.hash_indices_rows`` ``validate=False`` without losing the half of its
    check that a bound derived from the same array cannot provide.
    """
    indices_np = np.array([[0, 4], [2, 7], [7, 3]], dtype=np.int32)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)

    assert tw.array.index_bound(indices_wp) == 8
    assert tw.array.index_bound(indices_wp, require_non_negative=True) == 8

    negative_wp = wp.array(
        np.array([[0, 4], [-2, 7], [7, 3]], dtype=np.int32), dtype=wp.int32, device=device
    )
    # The plain form is a max and cannot see it; the guarded form is the only one that raises.
    assert tw.array.index_bound(negative_wp) == 8
    with pytest.raises(ValueError, match="non-negative"):
        _ = tw.array.index_bound(negative_wp, require_non_negative=True)

    # An empty buffer has no indices to be out of range, and both forms agree on that.
    empty_wp = warp_empty((0, 2), wp.int32, device)
    assert tw.array.index_bound(empty_wp) == 0
    assert tw.array.index_bound(empty_wp, require_non_negative=True) == 0


def test_read_scalar_returns_a_detached_row_for_a_vector_dtype(device: str) -> None:
    """
    Triwarp against triwarp: two reads of one vector array must not alias each other.

    Not a library comparison: ``_device.read_scalar`` is a private readback helper with no
    counterpart in any reference. The invariant is that the value survives the *next* read, which
    is what a shared per-dtype scratch buffer threatens: ``.numpy()`` on a ``wp.array[wp.vec3]``
    yields a view, so without a copy the first read would take the second's value. That defect is
    silent -- every scalar dtype is unaffected, because ``numpy`` hands those back as scalars --
    and it was found by ``creation.sweep_polygon``, which reads a path's two endpoints back to back
    and decided every open path was closed.
    """
    path_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [9.0, 8.0, 7.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )

    first = _device.read_scalar(path_wp, 0)
    last = _device.read_scalar(path_wp, 2)

    assert np.array_equal(first, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(last, np.array([9.0, 8.0, 7.0], dtype=np.float32))
    # The scalar path is the one that never needed the copy; assert it still reads correctly.
    counts_wp = wp.array(np.array([4, 5, 6], dtype=np.int32), dtype=wp.int32, device=device)
    assert int(_device.read_scalar(counts_wp, 0)) == 4
    assert int(_device.read_scalar(counts_wp)) == 6


def test_packers_reject_a_cross_device_sequence() -> None:
    """
    Triwarp against triwarp: a mixed-device sequence must be rejected, not silently migrated.

    Not a library comparison: no reference library shares Warp's device model. The failure this
    prevents is not a crash -- unlike ``combine.concatenate``, which launches a kernel and so
    segfaults on a mismatch, these two only copy, and ``wp.copy`` transfers across devices happily.
    Measured before the guard: ``concatenate([cpu, cuda])`` returned the right *values* on ``cpu``,
    silently choosing the answer's device from whichever segment happened to be first and, with it,
    the device of every launch built on the result downstream.

    Both entry points are covered because the guard lives in the private body they share, so a
    future caller that stopped delegating would keep passing a test that checked only one.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to construct a mismatch")
    cpu_wp = wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device="cpu")
    cuda_wp = wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device="cuda:0")
    for pack in (tw.array.concatenate, tw.array.pack_1d_arrays):
        with pytest.raises(RuntimeError, match=r"'arrays\[0\]' is on .+ while 'arrays\[1\]' is on"):
            pack([cpu_wp, cuda_wp])
        pack([cpu_wp, cpu_wp])  # a same-device sequence still goes through


def test_require_same_device_flags_a_mismatch_and_ignores_none(device: str) -> None:
    """
    Not a library comparison: ``_device.require_same_device`` has no reference-library equivalent.

    No other library shares Warp's launch-time device hazard, so there is nothing to compare its
    verdict against. Pins its three contracts directly instead: a ``None`` argument is skipped
    rather than compared, a ``list``/``tuple`` argument is unpacked element-wise with an
    ``f"{name}[{i}]"`` label, and same-device arguments are a silent no-op.
    """
    a_wp = wp.zeros(3, dtype=wp.float32, device=device)
    b_wp = wp.zeros(3, dtype=wp.float32, device=device)

    _device.require_same_device(a=a_wp, b=b_wp, unset=None)  # no raise
    _device.require_same_device(loops=[a_wp, b_wp, None])  # no raise

    if not wp.is_cuda_available():
        pytest.skip("needs both devices to construct a mismatch")
    c_wp = wp.zeros(3, dtype=wp.float32, device="cuda:0" if device == "cpu" else "cpu")
    with pytest.raises(RuntimeError, match=r"'a' is on .+ while 'c' is on"):
        _device.require_same_device(a=a_wp, c=c_wp)
    with pytest.raises(RuntimeError, match=r"'loops\[0\]' is on .+ while 'loops\[1\]' is on"):
        _device.require_same_device(loops=[a_wp, c_wp])


@pytest.mark.parametrize("sizes", [(3, 0, 5, 1), (4,), (0, 0, 2)])
def test_segment_owner_labels_matches_repeat(device: str, sizes: tuple[int, ...]):
    """
    Not a library comparison: pins the owner labels to ``numpy.repeat`` of the segment indices.

    ``segment_owner_labels`` takes the total-terminated offsets; the answer must be the owner array
    ``numpy.repeat`` builds, including empty segments and an empty trailing segment.
    """
    from triwarp.kernels import array as kernel_array

    sizes_np = np.asarray(sizes, dtype=np.int32)
    total = int(sizes_np.sum())
    offsets_np = np.concatenate([[0], np.cumsum(sizes_np)]).astype(np.int32)
    offsets_wp = wp.array(offsets_np, dtype=wp.int32, device=device)
    owner_wp = wp.full(total, -1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_array.segment_owner_labels,
        dim=len(sizes),
        inputs=[offsets_wp, owner_wp],
        device=device,
    )
    assert np.array_equal(owner_wp.numpy(), np.repeat(np.arange(len(sizes)), sizes_np))
