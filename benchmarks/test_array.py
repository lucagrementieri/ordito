"""
Benchmarks for ``ordito.array``: the packing, sorting and compaction primitives.

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
import scipy.sparse
import torch
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, BenchLibrary

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
_triplet_cache: dict[
    tuple[str, str], tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.float32]]
] = {}
_key_cache: dict[tuple[str, str], tuple[wp.array[wp.uint64], wp.array[wp.int32]]] = {}

# Element count for the mesh-free allocation rows: a mid-sized operator's triplet count. They
# allocate uninitialized storage, so the count moves nothing and one point is the whole axis.
_ALLOCATION_COUNT = 1 << 20


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


def _face_rows(bench_case: BenchCase) -> odt.Array2dInt32:
    """Return the face index buffer viewed as ``(n_faces, 3)`` rows."""
    return odt.as_array2d(bench_case.faces_wp.reshape((-1, 3)), wp.int32)


def _half_edges_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)`` of every directed face edge, ``3 * n_faces`` each: a mesh's COO pattern."""
    faces_np = bench_case.faces_np.astype(np.int32)
    rows_np = np.ascontiguousarray(faces_np.reshape(-1))
    cols_np = np.ascontiguousarray(np.roll(faces_np, -1, axis=1).reshape(-1))
    return rows_np, cols_np


def _half_edge_triplets(
    bench_case: BenchCase,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.float32]]:
    """Return the directed-edge pattern as device ``(rows, cols, values)`` unit triplets."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _triplet_cache:
        rows_np, cols_np = _half_edges_np(bench_case)
        device = bench_case.device
        _triplet_cache[key] = (
            wp.array(rows_np, dtype=wp.int32, device=device),
            wp.array(cols_np, dtype=wp.int32, device=device),
            wp.ones(rows_np.size, dtype=wp.float32, device=device),
        )
    return _triplet_cache[key]


def _half_edge_keys(bench_case: BenchCase) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Return the directed-edge keys and payloads a producer writes into a key sort's buffers.

    Both buffers are sorted in place by the functions that read them, so a row copies these into a
    fresh pair in its ``setup`` rather than handing them over.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _key_cache:
        rows_np, cols_np = _half_edges_np(bench_case)
        keys_np = rows_np.astype(np.uint64) * np.uint64(bench_case.n_vertices) + cols_np
        device = bench_case.device
        _key_cache[key] = (
            wp.array(keys_np, dtype=wp.uint64, device=device),
            wp.array(np.arange(keys_np.size, dtype=np.int32), dtype=wp.int32, device=device),
        )
    return _key_cache[key]


def _filled_key_buffers(bench_case: BenchCase) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """Return a fresh ``csr_key_buffers`` pair holding ``_half_edge_keys`` in its leading halves."""
    keys, order = _half_edge_keys(bench_case)
    keys_buffer, order_buffer = od.array.csr_key_buffers(keys.size, bench_case.device)
    wp.copy(keys_buffer, keys, count=keys.size)
    wp.copy(order_buffer, order, count=order.size)
    wp.synchronize_device(bench_case.device)
    return keys_buffer, order_buffer


def _flags_np(n: int) -> np.ndarray:
    """``(n,)`` ``int32`` ``0`` / ``1`` flags, about half set: the in-place compactions' input."""
    return (np.random.default_rng(3).random(n) < 0.5).astype(np.int32)


def _fresh_copy(source: wp.array[wp.int32], device: str | None) -> wp.array[wp.int32]:
    """Return a synchronized device copy of flags a row's call overwrites in place."""
    copy = wp.clone(source, device=device)
    wp.synchronize_device(device)
    return copy


@pytest.mark.benchmark(group="arange")
@pytest.mark.benchlibs("ordito", "numpy")
def test_arange(bench_case: BenchCase) -> None:
    """
    ``0 .. 3 * n_faces - 1`` as ``int32``: one closed-form write per element and one allocation.

    The scan sweep, since the length is the only thing that moves the work. ``numpy.arange`` is
    the reference ``tests/test_array.py`` compares against, so this row prices an allocation plus
    one launch against a host fill, and the crossover sits where the bandwidth outruns the launch.
    """
    n = bench_case.faces_np.size
    if bench_case.kind == "numpy":
        assert bench_case.run(lambda: np.arange(n, dtype=np.int32)).size == n
        return
    device = bench_case.device
    out = bench_case.run(lambda: od.array.arange(n, device=device))
    assert out.shape == (n,)


@pytest.mark.benchmark(group="arange_repeat")
@pytest.mark.benchlibs("ordito")
def test_arange_repeat(bench_case: BenchCase) -> None:
    """
    ``i // 3`` over ``3 * n_faces`` slots, each face's id at its three corners.

    The same one-write-per-element shape as ``arange`` at the same length, so the two rows read
    side by side; ordito-only, as the test checks it against a NumPy expression and no library
    function.
    """
    n = bench_case.faces_np.size
    device = bench_case.device
    out = bench_case.run(lambda: od.array.arange_repeat(n, 3, device))
    assert out.shape == (n,)


@pytest.mark.benchmark(group="sort_pair_indices")
@pytest.mark.benchlibs("ordito")
def test_sort_pair_indices(bench_case: BenchCase) -> None:
    """
    The identity payload of a radix sort's double buffer, ``2 * 3 * n_faces`` slots.

    At ``arange``'s length, so the pair prices the fill value's second half; ordito-only, as no
    library builds a sort's scratch layout.
    """
    n = bench_case.faces_np.size
    device = bench_case.device
    out = bench_case.run(lambda: od.array.sort_pair_indices(n, -1, device))
    assert out.shape == (2 * n,)


@pytest.mark.benchmark(group="concatenate_arrays")
@pytest.mark.benchlibs("ordito", "numpy", "pytorch3d")
@pytest.mark.parametrize("n_segments", _SEGMENT_COUNTS, ids=["few", "many"])
def test_concatenate(bench_case: BenchCase, n_segments: int) -> None:
    """
    One buffer from many, at two segment counts with the total element count held fixed.

    **ordito charges per segment below the kernel threshold, NumPy charges per byte, and that
    predicts every row here.** Under ``array.PACK_SEGMENTS_KERNEL_FROM`` the pack is one ``wp.copy``
    per segment at a few microseconds each, so its cost is **flat over four orders of magnitude of
    bytes** and the data movement is a percent of the row. NumPy's column is a host ``memcpy`` and
    rises with the bytes, so the break-even is a **segment size of about 100 kB and it does not move
    with the segment count**: ``few`` on ``dragon`` is megabytes a segment and ordito wins by an
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
    flat = bench_case.run(lambda: od.array.concatenate(segments))
    assert int(flat.shape[0]) == bench_case.faces_np.size


@pytest.mark.benchmark(group="pack_1d_arrays")
@pytest.mark.benchlibs("ordito", "numpy")
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
    flat, offsets = bench_case.run(lambda: od.array.pack_1d_arrays(segments))
    assert int(flat.shape[0]) == bench_case.faces_np.size
    assert int(offsets.shape[0]) == n_segments + 1


@pytest.mark.benchmark(group="split_array")
@pytest.mark.benchlibs("ordito", "numpy")
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
    libraries and the ratio between the pairs is the readback ordito cannot avoid.

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
    flat, offsets = od.array.pack_1d_arrays(_segments(bench_case, n_segments))
    parts = bench_case.run(lambda: od.array.split(flat, offsets, copy=copy))
    assert len(parts) == n_segments


@pytest.mark.benchmark(group="allclose")
@pytest.mark.benchlibs("ordito", "numpy")
def test_allclose(bench_case: BenchCase) -> None:
    """
    Element-wise closeness of the vertex table and a copy of it, read back as one ``bool``.

    The two tables are equal, so no element short-circuits either side and the whole of both is
    read. ``numpy.allclose`` is the reference ``tests/test_array.py`` compares against; like every
    reduction returning a Python value, ordito's row carries a launch and a readback NumPy never
    pays, so the scan sweep shows where the bandwidth repays them.
    """
    if bench_case.kind == "numpy":
        a_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        b_np = a_np.copy()
        assert bench_case.run(lambda: bool(np.allclose(a_np, b_np)))
        return
    a = bench_case.vertices_wp
    b = wp.clone(a, device=bench_case.device)
    assert bench_case.run(lambda: od.array.allclose(a, b))


@pytest.mark.benchmark(group="sort_and_argsort")
@pytest.mark.benchlibs("ordito", "numpy")
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
    sorted_keys, order = bench_case.run(lambda: od.array.sort_and_argsort(keys))
    assert int(sorted_keys.shape[0]) == keys.size
    assert int(order.shape[0]) == keys.size


@pytest.mark.benchmark(group="sort_and_argsort_in_place")
@pytest.mark.benchlibs("ordito")
def test_sort_and_argsort_in_place(bench_case: BenchCase) -> None:
    """
    The same radix sort over the same keys, handed in the sort's own double-width buffers.

    At ``sort_and_argsort``'s operating point, so the pair prices the key copy and the payload
    fill the buffer form skips. The buffers are filled in ``setup``, outside the timed region,
    because the sort overwrites them. ordito-only: ``tests/test_array.py`` pins it to
    ``sort_and_argsort``, which carries the NumPy oracle.
    """
    keys = _keys(bench_case)
    n = keys.size
    device = bench_case.device

    def fill() -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        keys_buffer = od.array.sort_pair_indices(n, 0, device)  # upper half is scratch
        wp.copy(keys_buffer, keys, count=n)
        order_buffer = od.array.sort_pair_indices(n, 0, device)
        wp.synchronize_device(device)
        return keys_buffer, order_buffer

    sorted_keys, order = bench_case.run(
        lambda buffers: od.array.sort_and_argsort_in_place(buffers[0], buffers[1]), setup=fill
    )
    assert sorted_keys.shape == (n,)
    assert order.shape == (n,)


@pytest.mark.benchmark(group="sort_rows")
@pytest.mark.benchlibs("ordito", "numpy")
def test_sort_rows(bench_case: BenchCase) -> None:
    """
    Each face's three corner indices sorted ascending, in place: the in-register insertion sort.

    Three columns is the width every edge and face dedup sorts at, and the scan sweep moves only
    the row count. ``numpy.sort(axis=1)`` is the reference ``tests/test_array.py`` compares
    against; it returns a copy, so ordito's input is refreshed in ``setup`` instead, outside the
    timed region, since its sort overwrites it.
    """
    if bench_case.kind == "numpy":
        faces_np = bench_case.faces_np.astype(np.int32)
        sorted_np = bench_case.run(lambda: np.sort(faces_np, axis=1))
        assert sorted_np.shape == faces_np.shape
        return
    faces = bench_case.faces_wp
    device = bench_case.device

    def fill() -> odt.Array2dInt32:
        fresh = odt.as_array2d(wp.clone(faces, device=device).reshape((-1, 3)), wp.int32)
        wp.synchronize_device(device)
        return fresh

    bench_case.run(od.array.sort_rows, setup=fill)


@pytest.mark.benchmark(group="triplet_buffers")
@pytest.mark.benchlibs("ordito")
def test_triplet_buffers(bench_lib: BenchLibrary) -> None:
    """
    Three uninitialized COO buffers: allocation only, so its cost is the allocator's per call.

    Mesh-free and at one size, because nothing is written and the count does not move the work.
    ordito-only: an allocation helper has no library counterpart.
    """
    device = bench_lib.device
    rows, cols, values = bench_lib.run(
        lambda: od.array.triplet_buffers(_ALLOCATION_COUNT, wp.float32, device)
    )
    assert rows.shape == cols.shape == values.shape == (_ALLOCATION_COUNT,)


@pytest.mark.benchmark(group="empty_square_bsr")
@pytest.mark.benchlibs("ordito")
def test_empty_square_bsr(bench_lib: BenchLibrary) -> None:
    """
    The zero-entry square operator an assembler returns for a face-free mesh.

    Mesh-free: it allocates the ``n_rows + 1`` zero offsets and wraps them, so the row count is
    the only input and it moves one memset. ordito-only.
    """
    device = bench_lib.device
    matrix = bench_lib.run(lambda: od.array.empty_square_bsr(_ALLOCATION_COUNT, wp.float32, device))
    assert matrix.nrow == _ALLOCATION_COUNT


@pytest.mark.benchmark(group="bsr_from_csr")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("ordito")
def test_bsr_from_csr(bench_case: BenchCase) -> None:
    """
    Wrapping compressed-row storage a producer already wrote as a ``BsrMatrix``, with no copy.

    Host work and a fixed amount of it, so one small feature mesh is the whole axis: the storage
    is the directed-edge operator of ``csr_from_triplets`` below, built outside the timed region.
    ordito-only: no library wraps Warp storage.
    """
    rows, cols, values = _half_edge_triplets(bench_case)
    n = bench_case.n_vertices
    built = od.array.csr_from_triplets(n, n, rows, cols, values)
    offsets, columns, entries = built.offsets, built.columns, built.values
    matrix = bench_case.run(lambda: od.array.bsr_from_csr(n, n, offsets, columns, entries))
    assert (matrix.nrow, matrix.ncol) == (n, n)


@pytest.mark.benchmark(group="bsr_with_values")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito")
def test_bsr_with_values(bench_case: BenchCase) -> None:
    """A matrix over another's pattern with new values: host bookkeeping only, no launch."""
    operator = od.laplacian.cotmatrix(bench_case.vertices_wp, bench_case.faces_wp, dtype=wp.float64)
    values = wp.zeros_like(operator.values)
    matrix = bench_case.run(lambda: od.array.bsr_with_values(operator, values))
    assert matrix.offsets is operator.offsets


@pytest.mark.benchmark(group="block_diag")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "sphere_large")
@pytest.mark.benchlibs("ordito", "scipy")
def test_block_diag(bench_case: BenchCase) -> None:
    """
    Three copies of the heat system stacked block-diagonally, across the scale sweep.

    ``block_diag`` keeps its result per set of operators, so the ordito row passes a fresh view of
    the first operator (``bsr_with_values`` over its own arrays, built outside the timed region) to
    time the build, not the cache hit. SciPy's ``block_diag`` of the same three CSR matrices is the
    test oracle and the reference.
    """
    if bench_case.kind == "scipy":
        # The reference's input is the same operator, assembled by ordito and read back.
        device = wp.get_preferred_device()
        system = od.heat.heat_operators(
            wp.array(bench_case.vertices_np, dtype=wp.vec3, device=device),
            wp.array(bench_case.faces_np.ravel(), dtype=wp.int32, device=device),
        )[0]
        n = bench_case.n_vertices
        offsets_np = system.offsets.numpy()[: n + 1]
        count = int(offsets_np[-1])
        system_np = scipy.sparse.csr_matrix(
            (system.values.numpy()[:count], system.columns.numpy()[:count], offsets_np),
            shape=(n, n),
        )
        stacked_np = bench_case.run(
            lambda: scipy.sparse.block_diag((system_np, system_np, system_np), format="csr")
        )
        assert stacked_np.shape[0] == 3 * n
        return
    system = od.heat.heat_operators(bench_case.vertices_wp, bench_case.faces_wp)[0]
    stacked = bench_case.run(
        lambda first: od.array.block_diag((first, system, system)),
        setup=lambda: od.array.bsr_with_values(system, system.values),
    )
    assert stacked.nrow == 3 * bench_case.n_vertices


@pytest.mark.benchmark(group="scalar_expansion")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "sphere_large")
@pytest.mark.benchlibs("ordito")
def test_scalar_expansion(bench_case: BenchCase) -> None:
    """
    A ``wp.mat22d`` operator over the cotangent pattern expanded to its scalar CSR, per call.

    The storage is kept per operator, so a timed call is the steady state a solver pays: one
    launch rewriting the expansion. ordito-only: no library takes Warp's block storage.
    """
    operator = od.laplacian.cotmatrix(bench_case.vertices_wp, bench_case.faces_wp, dtype=wp.float64)
    n = bench_case.n_vertices
    blocks = wp.full(operator.values.size, wp.mat22d(2.0, 1.0, 1.0, 3.0), device=operator.device)
    matrix = od.array.bsr_from_csr(n, n, operator.offsets, operator.columns, blocks)
    scalar = bench_case.run(lambda: od.array.scalar_expansion(matrix))
    assert scalar.nrow == 2 * n


@pytest.mark.benchmark(group="csr_key_buffers")
@pytest.mark.benchlibs("ordito")
def test_csr_key_buffers(bench_lib: BenchLibrary) -> None:
    """
    The two uninitialized double-width buffers a key producer writes into: allocation only.

    Mesh-free and at one size, as for ``triplet_buffers``. ordito-only.
    """
    device = bench_lib.device
    keys, order = bench_lib.run(lambda: od.array.csr_key_buffers(_ALLOCATION_COUNT, device))
    assert keys.shape == order.shape == (2 * _ALLOCATION_COUNT,)


@pytest.mark.benchmark(group="csr_from_keys")
@pytest.mark.benchlibs("ordito", "scipy")
def test_csr_from_keys(bench_case: BenchCase) -> None:
    """
    The CSR pattern of a mesh's ``3 * n_faces`` directed edges, sorted from their written keys.

    The scan sweep, since the key count is the work. The keys are refilled in ``setup``, outside
    the timed region, because the sort overwrites them. SciPy's ``coo_matrix(...).tocsr()`` is the
    reference ``tests/test_array.py`` compares the pattern against; it sums the duplicate entries'
    values where ordito returns each entry's sorted positions, so the scipy row does one more
    pass over the values and none over the keys.
    """
    n = bench_case.n_vertices
    if bench_case.kind == "scipy":
        rows_np, cols_np = _half_edges_np(bench_case)
        ones_np = np.ones(rows_np.size, dtype=np.float32)
        csr_sp = bench_case.run(
            lambda: scipy.sparse.coo_matrix((ones_np, (rows_np, cols_np)), shape=(n, n)).tocsr()
        )
        assert csr_sp.indptr.size == n + 1
        return
    count = bench_case.faces_np.size
    offsets, _columns, starts = bench_case.run(
        lambda buffers: od.array.csr_from_keys(buffers[0], buffers[1], count, n, n),
        setup=lambda: _filled_key_buffers(bench_case),
    )
    assert offsets.shape == (n + 1,)
    assert starts.shape == (count + 1,)


@pytest.mark.benchmark(group="csr_from_triplets")
@pytest.mark.benchlibs("ordito")
def test_csr_from_triplets(bench_case: BenchCase) -> None:
    """
    The directed-edge operator assembled from ``3 * n_faces`` unit triplets, duplicates summed.

    The scan sweep, as for ``csr_from_keys``, which this wraps: the pair prices the key packing and
    the value pass. ordito-only, since its test's oracle is ``warp.sparse``'s own builder.
    """
    rows, cols, values = _half_edge_triplets(bench_case)
    n = bench_case.n_vertices
    matrix = bench_case.run(lambda: od.array.csr_from_triplets(n, n, rows, cols, values))
    assert (matrix.nrow, matrix.ncol) == (n, n)


@pytest.mark.benchmark(group="csr_from_coo_keys")
@pytest.mark.benchlibs("ordito")
def test_csr_from_coo_keys(bench_case: BenchCase) -> None:
    """
    The same operator from keys a producer wrote itself: ``csr_from_triplets`` less the packing.

    At ``csr_from_triplets``' operating point, so the two rows read side by side. The keys are
    refilled in ``setup`` because the sort overwrites them. ordito-only.
    """
    _rows, _cols, values = _half_edge_triplets(bench_case)
    n = bench_case.n_vertices
    matrix = bench_case.run(
        lambda buffers: od.array.csr_from_coo_keys(n, n, buffers[0], buffers[1], values),
        setup=lambda: _filled_key_buffers(bench_case),
    )
    assert (matrix.nrow, matrix.ncol) == (n, n)


@pytest.mark.benchmark(group="csr_transpose")
@pytest.mark.benchlibs("ordito")
def test_csr_transpose(bench_case: BenchCase) -> None:
    """
    The transpose of the directed-edge operator, with no host readback.

    The scan sweep: the entry count is the work. ordito-only -- its test's oracles are a dense
    NumPy transpose of a small matrix and Warp's own ``bsr_transposed``, neither a timed reference.
    """
    rows, cols, values = _half_edge_triplets(bench_case)
    n = bench_case.n_vertices
    matrix = od.array.csr_from_triplets(n, n, rows, cols, values)
    transposed = bench_case.run(lambda: od.array.csr_transpose(matrix))
    assert (transposed.nrow, transposed.ncol) == (n, n)


@pytest.mark.benchmark(group="index_sparse")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_index_sparse(bench_case: BenchCase) -> None:
    """
    The vertex-by-face incidence matrix of the face table, as CSR.

    The scan sweep: ``3 * n_faces`` entries. ``trimesh.geometry.index_sparse`` is the reference
    ``tests/test_array.py`` compares against; it returns COO, so its row includes the ``tocsr``
    that test reads it through, which makes both sides hand back the same compressed form.
    """
    n = bench_case.n_vertices
    if bench_case.kind == "trimesh":
        faces_np = bench_case.faces_np
        incidence_tm = bench_case.run(lambda: tm.geometry.index_sparse(n, faces_np).tocsr())
        assert incidence_tm.shape == (n, bench_case.n_faces)
        return
    rows = _face_rows(bench_case)
    matrix = bench_case.run(lambda: od.array.index_sparse(n, rows))
    assert (matrix.nrow, matrix.ncol) == (n, bench_case.n_faces)


@pytest.mark.benchmark(group="isin")
@pytest.mark.benchlibs("ordito", "numpy")
def test_isin(bench_case: BenchCase) -> None:
    """
    Membership of every face corner in a quarter of the vertex ids, as a mask over the corners.

    The scan sweep, the corner count being the work; the test set grows with it, so the dense
    membership table's span is the vertex count throughout. ``numpy.isin`` is the reference
    ``tests/test_array.py`` compares against.
    """
    rng = np.random.default_rng(4)
    test_np = rng.choice(bench_case.n_vertices, size=max(1, bench_case.n_vertices // 4))
    test_np = test_np.astype(np.int32)
    if bench_case.kind == "numpy":
        elements_np = bench_case.faces_np.reshape(-1)
        mask_np = bench_case.run(lambda: np.isin(elements_np, test_np))
        assert mask_np.size == elements_np.size
        return
    elements = bench_case.faces_wp
    test_elements = wp.array(test_np, dtype=wp.int32, device=bench_case.device)
    mask = bench_case.run(lambda: od.array.isin(elements, test_elements))
    assert mask.shape == (elements.size,)


@pytest.mark.benchmark(group="flatnonzero")
@pytest.mark.benchlibs("ordito", "numpy")
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
    indices = bench_case.run(lambda: od.array.flatnonzero(mask))
    assert int(indices.shape[0]) > 0


@pytest.mark.benchmark(group="flatnonzero_in_place")
@pytest.mark.benchlibs("ordito", "numpy")
def test_flatnonzero_in_place(bench_case: BenchCase) -> None:
    """
    The positions of the ones in a half-set ``0`` / ``1`` buffer over the corners, scanning it.

    ``flatnonzero``'s ``half`` point with the flags already ``int32``, so the pair prices the
    flag pass and the scan buffer the in-place form skips. The flags are refreshed in ``setup``,
    outside the timed region, because the scan overwrites them. ``numpy.flatnonzero`` is the
    reference ``tests/test_array.py`` compares against.
    """
    flags_np = _flags_np(bench_case.faces_np.size)
    if bench_case.kind == "numpy":
        indices_np = bench_case.run(lambda: np.flatnonzero(flags_np))
        assert indices_np.size > 0
        return
    device = bench_case.device
    flags = wp.array(flags_np, dtype=wp.int32, device=device)
    indices = bench_case.run(
        od.array.flatnonzero_in_place, setup=lambda: _fresh_copy(flags, device)
    )
    assert 0 < indices.size < flags.size


@pytest.mark.benchmark(group="compress_in_place")
@pytest.mark.benchlibs("ordito", "numpy")
def test_compress_in_place(bench_case: BenchCase) -> None:
    """
    The vertex positions at the ones of a half-set flag buffer, scanning the flags in place.

    The scan sweep over the vertices; the same scan as ``flatnonzero_in_place`` with a ``wp.vec3``
    gather in place of the index write. The flags are refreshed in ``setup``. ``numpy.compress``
    is the reference ``tests/test_array.py`` compares against.
    """
    flags_np = _flags_np(bench_case.n_vertices)
    if bench_case.kind == "numpy":
        values_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        selected_np = flags_np.astype(bool)
        kept_np = bench_case.run(lambda: np.compress(selected_np, values_np, axis=0))
        assert 0 < kept_np.shape[0] < values_np.shape[0]
        return
    device = bench_case.device
    flags = wp.array(flags_np, dtype=wp.int32, device=device)
    values = bench_case.vertices_wp
    kept = bench_case.run(
        lambda fresh: od.array.compress_in_place(fresh, values),
        setup=lambda: _fresh_copy(flags, device),
    )
    assert 0 < kept.size < values.size


@pytest.mark.benchmark(group="gather")
@pytest.mark.benchlibs("ordito", "numpy")
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
    out = bench_case.run(lambda: od.array.gather(src, indices))
    assert int(out.shape[0]) == int(indices.shape[0])


@pytest.mark.benchmark(group="astype")
@pytest.mark.benchlibs("ordito", "numpy")
def test_astype(bench_case: BenchCase) -> None:
    """
    The ``int32`` face buffer converted to ``float32``: one allocation and one element-wise pass.

    The scan sweep, the element count being the work. ``numpy.ndarray.astype`` is the reference
    ``tests/test_array.py`` compares against, so the ratio is bandwidth against a launch.
    """
    if bench_case.kind == "numpy":
        faces_np = bench_case.faces_np.astype(np.int32).reshape(-1)
        converted_np = bench_case.run(lambda: faces_np.astype(np.float32))
        assert converted_np.size == faces_np.size
        return
    faces = bench_case.faces_wp
    converted = bench_case.run(lambda: od.array.astype(faces, wp.float32))
    assert converted.shape == faces.shape


@pytest.mark.benchmark(group="copyto")
@pytest.mark.benchlibs("ordito", "numpy")
def test_copyto(bench_case: BenchCase) -> None:
    """
    The same conversion written into a buffer the caller holds: ``astype`` less the allocation.

    At ``astype``'s operating point, so the pair prices the allocation. ``numpy.copyto`` is the
    reference ``tests/test_array.py`` compares against, with ``casting="unsafe"`` for the
    ``int32`` to ``float32`` step that ordito's conversion takes as ``astype`` does.
    """
    if bench_case.kind == "numpy":
        faces_np = bench_case.faces_np.astype(np.int32).reshape(-1)
        dst_np = np.empty(faces_np.size, dtype=np.float32)
        bench_case.run(lambda: np.copyto(dst_np, faces_np, casting="unsafe"))
        assert np.isfinite(dst_np).all()
        return
    faces = bench_case.faces_wp
    dst = odt.empty_1d(faces.size, wp.float32, device=bench_case.device)
    bench_case.run(lambda: od.array.copyto(dst, faces))
    assert np.isfinite(dst.numpy()).all()


@pytest.mark.benchmark(group="index_bound")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_index_bound(bench_case: BenchCase) -> None:
    """
    A max-reduce over ``3F`` indices; the scan sweep is here for ``lucy``'s 84M of them.

    Below roughly ``10 ** 3`` indices this row reports the wrapper floor and nothing else
    (see ``test_creation::test_box``), so the small end of the axis loses to ``faces.max()`` by up
    to two orders of magnitude while the largest mesh wins by as much. Read the whole axis, not one
    point: the
    crossover, not either endpoint, is what this group establishes.
    """
    if bench_case.kind == "ordito":
        faces = cast(odt.Array1dInt32, bench_case.faces_wp)
        result = bench_case.run(lambda: od.array.index_bound(faces))
        assert result == bench_case.n_vertices
    else:  # numpy reference: what trimesh-style code does on host arrays
        faces_np = bench_case.faces_np
        result = bench_case.run(lambda: int(faces_np.max()) + 1)
        assert result == bench_case.n_vertices


@pytest.mark.benchmark(group="indices_to_mask")
@pytest.mark.benchlibs("ordito")
def test_indices_to_mask(bench_case: BenchCase) -> None:
    """
    A vertex mask over ``gather``'s half-size random index list: a zeroed allocation, a scatter.

    The scan sweep, at ``gather``'s indices. ordito-only: its test is a round trip through
    ``flatnonzero``, not a library comparison.
    """
    _src, indices = _gather_inputs(bench_case)
    n = bench_case.n_vertices
    mask = bench_case.run(lambda: od.array.indices_to_mask(indices, n))
    assert mask.shape == (n,)


@pytest.mark.benchmark(group="mask_to_compact_ranks")
@pytest.mark.benchlibs("ordito", "numpy")
def test_mask_to_compact_ranks(bench_case: BenchCase) -> None:
    """
    Each selected corner's rank among the selected, and their count, over ``flatnonzero``'s mask.

    The ``half`` point of ``flatnonzero``, so the pair prices the scatter of indices against the
    exclusive scan alone; the count is one tail readback. The NumPy row is the
    ``cumsum(mask) - mask`` and population ``tests/test_array.py`` compares against.
    """
    if bench_case.kind == "numpy":
        mask_np = _mask_np(bench_case, 0.5)

        def ranks_np() -> tuple[np.ndarray, int]:
            selected_np = mask_np.astype(np.int32)
            return np.cumsum(selected_np) - selected_np, int(selected_np.sum())

        index_map_np, count_np = bench_case.run(ranks_np)
        assert index_map_np.size == mask_np.size
        assert count_np > 0
        return
    mask = _mask(bench_case, 0.5)
    index_map, count = bench_case.run(lambda: od.array.mask_to_compact_ranks(mask))
    assert index_map.shape == (mask.size,)
    assert count > 0


@pytest.mark.benchmark(group="counts_to_offsets")
@pytest.mark.benchlibs("ordito", "numpy")
def test_counts_to_offsets(bench_case: BenchCase) -> None:
    """
    The total-terminated offsets of one random ``0 .. 6`` count per face, and their total.

    The scan sweep over the faces: one in-place scan of the ``n_faces + 1`` buffer and one tail
    readback. The NumPy row is the ``cumsum`` behind a zero ``tests/test_array.py`` compares
    against.
    """
    counts_np = np.random.default_rng(5).integers(0, 7, bench_case.n_faces).astype(np.int32)
    if bench_case.kind == "numpy":

        def offsets_np() -> tuple[np.ndarray, int]:
            out_np = np.concatenate([[0], np.cumsum(counts_np)]).astype(np.int32)
            return out_np, int(out_np[-1])

        out_np, total_np = bench_case.run(offsets_np)
        assert out_np.size == counts_np.size + 1
        assert total_np > 0
        return
    counts = wp.array(counts_np, dtype=wp.int32, device=bench_case.device)
    offsets, total = bench_case.run(lambda: od.array.counts_to_offsets(counts))
    assert offsets.shape == (counts.size + 1,)
    assert total > 0


@pytest.mark.benchmark(group="remap_indices")
@pytest.mark.benchlibs("ordito")
def test_remap_indices(bench_case: BenchCase) -> None:
    """
    The face buffer renumbered through a random vertex permutation: one read-through-table pass.

    The scan sweep, the corner count being the work. ordito-only: what separates it from
    ``gather`` is the sentinel pass-through, which no library function has.
    """
    permutation_np = np.random.default_rng(6).permutation(bench_case.n_vertices).astype(np.int32)
    remap = wp.array(permutation_np, dtype=wp.int32, device=bench_case.device)
    faces = bench_case.faces_wp
    remapped = bench_case.run(lambda: od.array.remap_indices(faces, remap))
    assert remapped.shape == faces.shape


@pytest.mark.benchmark(group="trim_to_count")
@pytest.mark.benchlibs("ordito")
def test_trim_to_count(bench_case: BenchCase) -> None:
    """
    Two ``3 * n_faces`` atomic-append buffers cut to the counter's ``n_faces`` written slots.

    One counter readback and one copy per buffer, of a prefix that grows along the scan sweep.
    ordito-only: the test's oracle is a NumPy slice of its own input, not a library call.
    """
    faces, keys = bench_case.faces_wp, _keys(bench_case)
    n_written = bench_case.n_faces
    counter = wp.array([n_written], dtype=wp.int32, device=bench_case.device)
    count, (faces_kept, keys_kept) = bench_case.run(
        lambda: od.array.trim_to_count(counter, faces, keys)
    )
    assert count == n_written
    assert faces_kept.shape == keys_kept.shape == (n_written,)


def _float_corners(bench_case: BenchCase) -> wp.array[wp.float32]:
    """Return the face buffer as ``float32`` values, the input both bit-cast rows reinterpret."""
    return od.array.astype(bench_case.faces_wp, wp.float32)


@pytest.mark.benchmark(group="bitcast_to_int")
@pytest.mark.benchlibs("ordito")
def test_bitcast_to_int(bench_case: BenchCase) -> None:
    """
    ``3 * n_faces`` ``float32`` values reinterpreted as ``int32``: one copy, no arithmetic.

    The scan sweep, the element count being the work. ordito-only: its test is a round trip with
    ``bitcast_from_int``, the next row, timed at the same point.
    """
    values = _float_corners(bench_case)
    as_int = bench_case.run(lambda: od.array.bitcast_to_int(values))
    assert as_int.shape == values.shape


@pytest.mark.benchmark(group="bitcast_from_int")
@pytest.mark.benchlibs("ordito")
def test_bitcast_from_int(bench_case: BenchCase) -> None:
    """The inverse reinterpretation, at ``bitcast_to_int``'s operating point. ordito-only."""
    as_int = od.array.bitcast_to_int(_float_corners(bench_case))
    values = bench_case.run(lambda: od.array.bitcast_from_int(as_int, wp.float32))
    assert values.shape == as_int.shape
