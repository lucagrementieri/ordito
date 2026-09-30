"""
Benchmarks for ``triwarp.array``: the packing, sorting and compaction primitives.

Every group here sweeps **segment or selection count**, not mesh size, because that is the axis
these primitives actually respond to. ``pack_1d_arrays`` and ``concatenate`` issue one ``wp.copy``
per input segment on top of a fixed amount of host-side Python, so holding the total element count
fixed and moving only the number of segments separates the two costs: flat means the copies
dominate and the Python loops are noise, rising means the per-segment overhead is the thing to
attack. ``flatnonzero`` sweeps selectivity for the same reason -- the scan is oblivious to it and
only the scatter's output size moves.

**That question has been answered and the axis was the right one: rising, steeply, and the
per-segment overhead is not attackable.** The packing family's cost is a per-*segment* host constant
-- a few microseconds a ``wp.copy``, more for a ``wp.clone``, less for a ``wp.array`` view -- while
the same total moved in one call is flat in its size, so the rows are flat over a four-orders-of-
magnitude range of bytes and linear in the segment count. The break-even against NumPy is a segment
size of **about 100 kB**, and it does not move with the count. ``_SEGMENT_COUNTS`` brackets it:
``few`` on ``dragon`` is megabytes a segment and wins by an order of magnitude, ``many`` on
``bunny_decimated`` is under a kilobyte a segment and loses by two. Both are the same statement
about segment size, which is why the loss half of that bracket is not a work item -- see
``test_concatenate`` and ``test_split`` for the sweeps and for the levers (graph capture, a shared
output buffer, raw view construction) that were measured and declined.

``sort_and_argsort`` and ``gather`` are the two primitives on the hot path of nearly every other
module (``grouping.group``, ``adjacency.face_adjacency``, every submesh extraction), so they are
timed on mesh-derived buffers at whatever size the suite is running.

References
----------
**NumPy is the reference, on the same grounds ``benchmarks/test_reduce.py`` argues at length.**
The host/device asymmetry is *the question* rather than a reason not to ask it: a device primitive
that hands back a Python value pays a launch and a readback NumPy never pays, so NumPy should win at
small sizes and each row answers where the crossover sits.

``test_reduce``'s answer transfers: **the split is by return type, not by size.** The groups here
that hand back a device array (``concatenate_arrays``, ``gather``, ``sort_and_argsort``) have no
host synchronisation to pay and should track bandwidth;
``flatnonzero`` and ``split_array`` end in a 4-byte tail readback and an ``offsets`` transfer
respectively, so they carry the flat host cost that sets the crossover. ``pack_1d_arrays`` is the
pair to read ``concatenate_arrays`` against: the offsets are host metadata on both sides.

trimesh, igl, open3d and pymeshlab stay unregistered here and it is not a judgement -- they operate
a level above and expose nothing comparable. ``index_bound``'s "trimesh" row is likewise a
NumPy stand-in (``int(faces.max()) + 1``) rather than a library call, because trimesh has no
*uncached* equivalent: its vertex count comes from the array it was built with. That row predates
this section and is the group that first established the crossover is inside the registry's size
range.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import pytorch3d.ops as p3d_ops
import torch
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase

# Segment counts at a fixed total length: few large pieces against many small ones. The element
# count copied is identical, so only the per-segment cost moves.
_SEGMENT_COUNTS = [4, 256]

# Fraction of the mask that is True. The scan pass is identical either way; only the compaction
# output changes size.
_SELECTIVITIES = [0.5, 0.01]

_segments_cache: dict[tuple[str, str, int], list[wp.array[wp.int32]]] = {}
_keys_cache: dict[tuple[str, str], wp.array[wp.int32]] = {}
_mask_cache: dict[tuple[str, str, float], wp.array[wp.bool]] = {}
_gather_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = {}


def _segments(bench_case: BenchCase, n_segments: int) -> list[wp.array[wp.int32]]:
    """Split the flat face buffer into ``n_segments`` contiguous 1-D pieces, same total length."""
    key = (bench_case.mesh_name, str(bench_case.device), n_segments)
    if key not in _segments_cache:
        faces_np = bench_case.faces_np.reshape(-1).astype(np.int32)
        pieces = np.array_split(faces_np, n_segments)
        _segments_cache[key] = [
            wp.array(np.ascontiguousarray(piece), dtype=wp.int32, device=bench_case.device)
            for piece in pieces
        ]
    return _segments_cache[key]


def _segments_np(bench_case: BenchCase, n_segments: int) -> list[np.ndarray]:
    """Split the flat face buffer as ``_segments`` does, on the host: a numpy row has no device."""
    faces_np = bench_case.faces_np.reshape(-1).astype(np.int32)
    return [np.ascontiguousarray(piece) for piece in np.array_split(faces_np, n_segments)]


def _keys_np(bench_case: BenchCase) -> np.ndarray:
    """Build the same shuffled key buffer as ``_keys``, on the host."""
    return np.random.default_rng(0).permutation(bench_case.faces_np.size).astype(np.int32)


def _mask_np(bench_case: BenchCase, selectivity: float) -> np.ndarray:
    """Build the same mask as ``_mask``, on the host."""
    return np.random.default_rng(1).random(bench_case.faces_np.size) < selectivity


def _gather_inputs_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Build the same ``(vertices, indices)`` pair as ``_gather_inputs``, on the host."""
    n = bench_case.n_vertices
    indices_np = np.random.default_rng(2).integers(0, n, size=max(1, n // 2)).astype(np.int32)
    return np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32), indices_np


def _keys(bench_case: BenchCase) -> wp.array[wp.int32]:
    """Build a shuffled int32 key buffer, one key per face index."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _keys_cache:
        rng = np.random.default_rng(0)
        keys_np = rng.permutation(bench_case.faces_np.size).astype(np.int32)
        _keys_cache[key] = wp.array(keys_np, dtype=wp.int32, device=bench_case.device)
    return _keys_cache[key]


def _mask(bench_case: BenchCase, selectivity: float) -> wp.array[wp.bool]:
    key = (bench_case.mesh_name, str(bench_case.device), selectivity)
    if key not in _mask_cache:
        rng = np.random.default_rng(1)
        mask_np = rng.random(bench_case.faces_np.size) < selectivity
        _mask_cache[key] = wp.array(mask_np, dtype=wp.bool, device=bench_case.device)
    return _mask_cache[key]


def _gather_inputs(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """``(vertices, indices)`` for a half-size vertex gather."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _gather_cache:
        rng = np.random.default_rng(2)
        n = bench_case.n_vertices
        indices_np = rng.integers(0, n, size=max(1, n // 2)).astype(np.int32)
        _gather_cache[key] = (
            bench_case.vertices_wp,
            wp.array(indices_np, dtype=wp.int32, device=bench_case.device),
        )
    return _gather_cache[key]


@pytest.mark.benchmark(group="concatenate_arrays")
@pytest.mark.benchlibs("triwarp", "numpy", "pytorch3d")
@pytest.mark.parametrize("n_segments", _SEGMENT_COUNTS, ids=["few", "many"])
def test_concatenate(bench_case: BenchCase, n_segments: int) -> None:
    """
    One buffer from many, at two segment counts with the total element count held fixed.

    **triwarp charges per segment below the kernel threshold, NumPy charges per byte, and that
    predicts every row here.** Under ``array.PACK_SEGMENTS_KERNEL_FROM`` the pack is one ``wp.copy``
    per segment at a few microseconds each, so its cost is **flat over four orders of magnitude of
    bytes** and the data movement is a percent of the row. NumPy's column is a host ``memcpy`` and
    rises with the bytes, so the break-even is a **segment size of about 100 kB and it does not move
    with the segment count**: ``few`` on ``dragon`` is megabytes a segment and triwarp wins by an
    order of magnitude, ``many`` on ``bunny_decimated`` is under a kilobyte a segment.

    Above that threshold the loop is replaced by **one segmented-copy launch** over a ``@wp.struct``
    descriptor table, which is flat in both the segment count and the total size -- so the two
    points of this axis straddle the dispatch as well as the NumPy crossover.

    **Graph capture is declined and this row is a floor for it**: recording the copy loop and
    replaying it once is a loss at every count, because a pack's pointers change per call so the
    recording is never reused. Replaying an *existing* graph is the several-fold win that makes
    capture look attractive, and it is unreachable here.

    **pytorch3d**'s ``padded_to_packed`` is the same concatenation from the other direction: it
    reads a dense ``(n_segments, max_size)`` block and writes the flat buffer in one device kernel.
    It is not free of a cost model of its own -- the padded block is ``n_segments * max_size``
    elements whatever the true lengths, so at ``many`` on an uneven split it moves more memory than
    there is data -- and the block is built outside the timed callable because it is the input.
    ``tests/test_array.py::test_the_packing_family_matches_pytorch3d`` pins the two forms to byte
    equality.
    """
    if bench_case.kind == "pytorch3d":
        segments_np = _segments_np(bench_case, n_segments)
        total = sum(int(segment_np.size) for segment_np in segments_np)
        widest = max(int(segment_np.size) for segment_np in segments_np)
        first_p3d = torch.as_tensor(
            np.cumsum([0, *(int(s.size) for s in segments_np[:-1])], dtype=np.int64),
            device=bench_case.torch_device,
        )
        padded_np = np.zeros((len(segments_np), widest), dtype=np.float32)
        for row, segment_np in enumerate(segments_np):
            padded_np[row, : segment_np.size] = segment_np
        padded_p3d = torch.as_tensor(padded_np, device=bench_case.torch_device)
        flat_p3d = bench_case.run(
            lambda: p3d_ops.padded_to_packed(padded_p3d, cast("torch.LongTensor", first_p3d), total)
        )
        assert flat_p3d.shape[0] == total
        return
    if bench_case.kind == "numpy":
        segments_np = _segments_np(bench_case, n_segments)
        flat_np = bench_case.run(lambda: np.concatenate(segments_np))
        assert flat_np.size == bench_case.faces_np.size
        return
    segments = _segments(bench_case, n_segments)
    flat = bench_case.run(lambda: tw.array.concatenate(segments))
    assert int(flat.shape[0]) == bench_case.faces_np.size


@pytest.mark.benchmark(group="pack_1d_arrays")
@pytest.mark.benchlibs("triwarp", "numpy")
@pytest.mark.parametrize("n_segments", _SEGMENT_COUNTS, ids=["few", "many"])
def test_pack_1d_arrays(bench_case: BenchCase, n_segments: int) -> None:
    """
    The same packing plus the per-segment offsets, on the same axis as ``concatenate_arrays``.

    Read the two groups together: the offsets are the only difference, so a gap between them at
    ``many`` is the host-side accumulate and the ``wp.array(list)`` transfer, not the copies.

    NumPy's counterpart is ``concatenate`` plus a ``cumsum`` of the lengths, which is the honest
    comparison: the offsets are host metadata on both sides, so this pair isolates the *transfer* of
    them from their computation.

    The gap turns out to be a constant and **flat in everything** -- the segment count, the segment
    size and the mesh -- because it is one ``warp.array(list)`` upload of at most a few hundred
    ``int32``. So this group is ``concatenate_arrays`` plus a constant, and its crossover is that
    group's: read the cost model there, not here. At ``many`` the offsets are a rounding error on
    the row; at ``few`` they are nearly half of it, which is the honest reading of a row that is
    four ``warp.copy`` calls in total.
    """
    if bench_case.kind == "numpy":
        segments_np = _segments_np(bench_case, n_segments)

        def pack_np() -> tuple[np.ndarray, np.ndarray]:
            return (
                np.concatenate(segments_np),
                np.cumsum([0] + [piece.size for piece in segments_np]),
            )

        flat_np, offsets_np = bench_case.run(pack_np)
        assert flat_np.size == bench_case.faces_np.size
        assert offsets_np.size == n_segments + 1
        return
    segments = _segments(bench_case, n_segments)
    flat, offsets = bench_case.run(lambda: tw.array.pack_1d_arrays(segments))
    assert int(flat.shape[0]) == bench_case.faces_np.size
    assert int(offsets.shape[0]) == n_segments + 1


@pytest.mark.benchmark(group="split_array")
@pytest.mark.benchlibs("triwarp", "numpy")
@pytest.mark.parametrize("n_segments", _SEGMENT_COUNTS, ids=["few", "many"])
@pytest.mark.parametrize("copy", [False, True], ids=["views", "copies"])
def test_split(bench_case: BenchCase, n_segments: int, copy: bool) -> None:
    """
    The inverse of ``pack_1d_arrays``: one offsets readback, then views or per-segment copies.

    The ``views`` rows price the readback plus Python-side ``warp.array`` construction; the gap to
    ``copies`` at ``many`` is one allocation per segment, which ``copy=True``'s contract makes the
    return value itself (the copies are then filled by one launch).

    NumPy's ``split`` has the same two modes and the same names for them -- its result is views, and
    a copy is one ``np.copy`` per piece -- so the ``views`` / ``copies`` pair reads across both
    libraries and the ratio between the pairs is the readback triwarp cannot avoid.

    **Both modes are per-segment constants, and the readback is not one of them** -- it is a couple
    of percent of the ``views`` row. The views are stamped from one template slice's state rather
    than sliced one at a time, which is several times cheaper per segment than Warp's own slice;
    ``copies`` is that plus one allocation per piece. The ``copies`` rows invert on the group's own
    axis, winning at ``few``.

    **One lever is declined.** Allocating **one** buffer, filling it with a single ``warp.copy`` and
    returning disjoint views of *that* is several times faster -- but it is then no longer the
    operation NumPy's column performs (``np.copy`` per piece is an independent allocation each), so
    it would win the row by doing less, and it would silently drop half of what ``copy=True`` is
    documented to promise: a segment could no longer be held without keeping the whole buffer
    alive.
    """
    if bench_case.kind == "numpy":
        segments_np = _segments_np(bench_case, n_segments)
        flat_np = np.concatenate(segments_np)
        offsets_np = np.cumsum([0] + [piece.size for piece in segments_np[:-1]])

        def split_np() -> list[np.ndarray]:
            parts = np.split(flat_np, offsets_np[1:])
            return [np.copy(part) for part in parts] if copy else parts

        assert len(bench_case.run(split_np)) == n_segments
        return
    flat, offsets = tw.array.pack_1d_arrays(_segments(bench_case, n_segments))
    parts = bench_case.run(lambda: tw.array.split(flat, offsets, copy=copy))
    assert len(parts) == n_segments


@pytest.mark.benchmark(group="sort_and_argsort")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_sort_and_argsort(bench_case: BenchCase) -> None:
    """
    Radix sort plus its permutation: the primitive under ``group`` and every dedup here.

    NumPy needs **two** calls for what one radix pass returns -- ``argsort`` for the permutation and
    a gather to apply it -- which is the shape of the comparison rather than a handicap: a
    least-significant-digit radix sort carries its payload through the same passes, so the pair
    prices "sorted keys and their order" against "the order, then use it".
    """
    if bench_case.kind == "numpy":
        keys_np = _keys_np(bench_case)

        def sort_and_argsort_np() -> tuple[np.ndarray, np.ndarray]:
            order_np = np.argsort(keys_np, kind="stable")
            return keys_np[order_np], order_np

        sorted_np, order_np = bench_case.run(sort_and_argsort_np)
        assert sorted_np.size == keys_np.size
        assert order_np.size == keys_np.size
        return
    keys = _keys(bench_case)
    sorted_keys, order = bench_case.run(lambda: tw.array.sort_and_argsort(keys))
    assert int(sorted_keys.shape[0]) == keys.size
    assert int(order.shape[0]) == keys.size


@pytest.mark.benchmark(group="flatnonzero")
@pytest.mark.benchlibs("triwarp", "numpy")
@pytest.mark.parametrize("selectivity", _SELECTIVITIES, ids=["half", "sparse"])
def test_flatnonzero(bench_case: BenchCase, selectivity: float) -> None:
    """
    Mask compaction at two selectivities.

    The flag pass and the scan are the same work either way, and only the scatter's output shrinks,
    so these two ids should sit close together. They also pin the cost of the single 4-byte tail
    readback that sizes the output -- the one host synchronisation this primitive cannot avoid,
    and the whole of what ``np.flatnonzero`` does not pay.
    """
    if bench_case.kind == "numpy":
        mask_np = _mask_np(bench_case, selectivity)
        assert bench_case.run(lambda: np.flatnonzero(mask_np)).size > 0
        return
    mask = _mask(bench_case, selectivity)
    indices = bench_case.run(lambda: tw.array.flatnonzero(mask))
    assert int(indices.shape[0]) > 0


@pytest.mark.benchmark(group="gather")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_gather(bench_case: BenchCase) -> None:
    """
    Dense materialization of a fancy-index view: one ``wp.copy`` out of an ``indexedarray``.

    ``src[indices]`` is NumPy's whole answer and it is already dense, so this is the group where the
    two libraries are closest in shape and the ratio is nearly pure memory bandwidth -- the one row
    here to read as a hardware comparison rather than as an API one.
    """
    if bench_case.kind == "numpy":
        src_np, indices_np = _gather_inputs_np(bench_case)
        assert bench_case.run(lambda: src_np[indices_np]).shape[0] == indices_np.size
        return
    src, indices = _gather_inputs(bench_case)
    out = bench_case.run(lambda: tw.array.gather(src, indices))
    assert int(out.shape[0]) == int(indices.shape[0])


@pytest.mark.benchmark(group="index_bound")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_index_bound(bench_case: BenchCase) -> None:
    """
    A max-reduce over ``3F`` indices; the scan sweep is here for ``lucy``'s 84M of them.

    Below roughly ``10 ** 3`` indices this row reports the wrapper floor and nothing else
    (see ``test_creation::test_box``), so the small end of the axis loses to ``faces.max()`` by up
    to two orders of magnitude while the largest mesh wins by as much. Read the whole axis, not one
    point: the
    crossover, not either endpoint, is what this group establishes.
    """
    if bench_case.kind == "triwarp":
        faces = cast(twt.Array1dInt32, bench_case.faces_wp)
        result = bench_case.run(lambda: tw.array.index_bound(faces))
        assert result == bench_case.n_vertices
    else:  # numpy reference: what trimesh-style code does on host arrays
        faces_np = bench_case.faces_np
        result = bench_case.run(lambda: int(faces_np.max()) + 1)
        assert result == bench_case.n_vertices
