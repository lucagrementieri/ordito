"""
Private helpers shared across wrapper modules.

What several wrappers need and no one public module owns: the argument guards
([`require_same_device`][ordito._common.require_same_device], which every public function
taking two or more device-bearing arguments calls first, and the mesh and index checks), host
readbacks ([`read_scalar`][ordito._common.read_scalar],
[`read_values`][ordito._common.read_values]), the index-buffer extrema behind the index checks,
scan counts and offset buffers, device loops, and per-device tuning choices. Kernel launching
lives in ``_launch``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito.constants import ITEMS_PER_SLICE_CPU, ITEMS_PER_SLICE_CUDA, TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import reduce as kernel_reduce

T = TypeVar("T")


def prefers_tiled_reduction(device: wp.DeviceLike) -> bool:
    """
    Whether ``device`` should run the ``wp.tile``-based variant of a global reduction.

    Reductions that land in a single accumulator have two implementations in this package, and the
    choice is forced rather than stylistic. ``wp.launch_tiled`` runs exactly **one** lane per block
    on the Warp CPU device -- ``wp.tid()``'s lane index is always 0 -- so a tile built out of
    *per-lane* values, ``wp.tile(x)``, holds a single element there and any reduction over it
    silently returns one element's worth of answer.

    The distinction matters, because it is *only* the lane-constructed tile that breaks.
    ``wp.tile_load`` reads its whole tile out of an array and is lane-independent, so it totals the
    same value on both devices -- which is why every factory in ``kernels/reduce.py`` may be tiled
    unconditionally while ``kernels/measures.py`` and ``kernels/metrics.py``, which build their
    tiles from a per-thread contribution, must branch here.

    **This is a known platform limitation, not a bug awaiting a report.** ``wp.launch`` documents
    ``block_dim`` as "always 1 for cpu devices" and ``launch_tiled`` forces it, so ``wp.tile(x)``
    correctly forms a one-element tile there. Upstream tracks closing the gap in NVIDIA/warp#1480
    (*CPU/GPU parity for all tile code*, which names ``wp.tile(lane_value)`` followed by reductions
    or scans as an affected pattern) and NVIDIA/warp#1638 (*Add efficient CPU block execution with
    fibers*). Warp 1.18 shipped the latter as the experimental, off-by-default
    ``wp.config.enable_cpu_blocks``, under which a CPU block runs every lane; this function then
    answers ``True`` on the CPU too, so the tiled variant runs there. That is correct but costs a
    fixed price per block far above the sliced variant's, so it is what the test suite's
    ``--cpu-blocks`` pass uses to run the CUDA code path on the deterministic CPU device, not a
    production setting. The branch becomes removable only once CPU blocks are cheap by default.

    The portable form instead gives each thread a strided slice and one atomic, which is correct on
    both devices but gives up the block shuffle-reduce, costing real CUDA time once the input
    exceeds the launch overhead. Below roughly 100k elements both forms sit at the launch floor.

    So: tiles on CUDA, slices on CPU. Reductions with *many* accumulators do not need this -- one
    per query already fills the device, and there the portable form is the faster one on CUDA too,
    so those have a single implementation.

    Parameters
    ----------
    device
        Warp device (or device string) the reduction will run on.

    Returns
    -------
    bool
        ``True`` for a CUDA device, ``False`` for CPU unless ``wp.config.enable_cpu_blocks`` is set.

    See Also
    --------
    [`items_per_slice`][ordito._common.items_per_slice]
    """
    return wp.get_device(device).is_cuda or wp.config.enable_cpu_blocks


def ball_grid_cell_width(
    radius: float, n_queries: int, device: wp.DeviceLike, wide_below_cuda: int, wide_below_cpu: int
) -> float:
    """
    Cell width for a hash grid that ball queries of ``radius`` walk: the radius or twice it.

    At twice the radius a query's box is one cell wide and meets 8 cells rather than up to 27,
    which shortens the walk when queries are few, while larger cells hold more candidates to reject
    when queries are many. Twice the radius below ``wide_below_cuda`` queries on a CUDA device and
    below ``wide_below_cpu`` on the CPU; each caller passes the thresholds measured for its own
    walk. The answer of a ball query does not depend on the cell width, only its cost and the
    order in which a walk meets the points.
    """
    wide_below = wide_below_cpu if wp.get_device(device).is_cpu else wide_below_cuda
    return 2.0 * radius if n_queries < wide_below else radius


def tiled_ball_walk(
    grid: object, n_queries: int, device: wp.DeviceLike, max_queries: int
) -> tuple[float, float, int] | None:
    """
    Return the cell geometry for a ball walk a block a query over ``grid``, or ``None``.

    ``None`` means the one-thread walk. A few-query walk on a CUDA device gives each lane of a
    block a share of the query's hash-grid cells (``kernels/neighbors.ball_cell_box``). It needs
    the grid's cell width and resolution, which
    [`hashgrid_from_points`][ordito.neighbors.hashgrid_from_points] records on the grids it
    builds; a grid built otherwise takes the one-thread walk. Returns ``(cell_width,
    cell_width_inv, resolution)``, ``cell_width_inv`` rounded as Warp rounds it, so the lanes'
    cells are exactly the grid query's.
    """
    if n_queries >= max_queries or wp.get_device(device).is_cpu:
        return None
    cell_width = getattr(grid, "cell_width", None)
    resolution = getattr(grid, "resolution", None)
    if cell_width is None or resolution is None or not cell_width > 0.0:
        return None
    width = np.float32(cell_width)
    return float(width), float(np.float32(1.0) / width), int(resolution)


def run_device_loop(
    device: wp.DeviceLike, condition: wp.array[wp.int32, Any], body: Callable[[], None]
) -> None:
    """
    Run ``body`` until the device-side ``condition`` word reads zero, without a host readback.

    A loop whose body writes its own continuation flag runs on CUDA as one captured conditional
    graph, so the per-round readback it would otherwise take -- which drains the pipeline that
    round's launches just filled -- disappears entirely. Graph capture needs a CUDA stream, so the
    CPU device falls back to ``wp.capture_while``'s direct execution, which is the same loop with
    one four-byte read per round.

    ``wp.capture_while`` evaluates ``condition`` **before** each round, including the first, so it
    must be seeded non-zero or the loop runs zero rounds. The shared slot table and the
    [`loop_advance`][ordito.kernels.array.loop_advance] kernel that writes it are in
    ``ordito/kernels/array.py``.

    Parameters
    ----------
    device
        Warp device (or device string) the loop's launches run on.
    condition
        Length-1 ``wp.int32`` view of the state word's condition slot, on ``device``.
    body
        Issues one round's launches. It is *recorded once* on CUDA and replayed, so it must not
        rebind buffers between rounds -- a Python-level ping-pong has no effect on the replayed
        loop, and a body whose result depends on running as an indivisible unit is not portable
        (see [`shortest_path_envelope`][ordito.graph.shortest_path_envelope]).

    See Also
    --------
    [`prefers_tiled_reduction`][ordito._common.prefers_tiled_reduction]
    """
    record_device_loop(device, condition, body)()


def record_device_loop(
    device: wp.DeviceLike, condition: wp.array[wp.int32, Any], body: Callable[[], None]
) -> Callable[[], None]:
    """
    Record [`run_device_loop`][ordito._common.run_device_loop]'s loop now; return what runs it.

    The same two branches, split between recording and launching, so a caller can record while
    the device is still busy with work already issued, then read that work's outcome and skip the
    launch. Both ICP loops do this with their first round: recording first overlaps it, so the read
    that decides whether the loop runs at all waits on nothing, where reading first leaves the
    device idle for the recording (2-8 % on every gated call), and not reading at all records and
    launches a loop of zero rounds after a weightless first round. On a device that cannot record,
    the returned call is ``wp.capture_while``'s direct execution.
    """
    resolved = wp.get_device(device)
    # ``wp.is_conditional_graph_supported`` is a *machine* query, so it answers ``True`` on a box
    # with a GPU even when this loop's arrays are on the CPU device; the ``is_cuda`` test is what
    # actually decides. A caller already inside a capture must not call this at all -- it nests by
    # calling ``wp.capture_while`` directly, as ``remesh._quadric_collapse_rounds`` does.
    if resolved.is_cuda and wp.is_conditional_graph_supported():
        with wp.ScopedCapture(resolved) as capture:
            wp.capture_while(cast("wp.array[int]", condition), body)
        graph = capture.graph
        assert graph is not None
        return lambda: wp.capture_launch(graph)
    return lambda: wp.capture_while(cast("wp.array[int]", condition), body)


def items_per_slice(device: wp.DeviceLike) -> int:
    """
    Elements per thread for the lane-free strided-slice reductions on ``device``.

    The optimum splits by device by more than a tolerance in both directions, so the value is
    chosen here rather than read from a single module constant -- see the measurements next to
    [`ITEMS_PER_SLICE_CUDA`][ordito.constants.ITEMS_PER_SLICE_CUDA].

    Parameters
    ----------
    device
        Warp device (or device string) the reduction will run on.

    Returns
    -------
    int
        Slice length: ``ITEMS_PER_SLICE_CUDA`` on CUDA, ``ITEMS_PER_SLICE_CPU`` on CPU.

    See Also
    --------
    [`prefers_tiled_reduction`][ordito._common.prefers_tiled_reduction]
    """
    return ITEMS_PER_SLICE_CUDA if wp.get_device(device).is_cuda else ITEMS_PER_SLICE_CPU


def slice_count(count: int, device: wp.DeviceLike) -> int:
    """
    Thread count for a lane-free strided-slice reduction over ``count`` elements.

    The launch dimension that goes with [`items_per_slice`][ordito._common.items_per_slice]: one
    thread per strided slice, and at least one thread so an empty input still launches a well-formed
    grid. Callers should not divide by ``items_per_slice`` themselves -- that spelling is what this
    exists to hold in one place.

    Parameters
    ----------
    count
        Number of elements to reduce.
    device
        Warp device (or device string) the reduction will run on.

    Returns
    -------
    int
        ``ceil(count / items_per_slice(device))``, floored at 1.

    See Also
    --------
    [`items_per_slice`][ordito._common.items_per_slice]
    """
    per_slice = items_per_slice(device)
    return max(1, (count + per_slice - 1) // per_slice)


def require_nonempty_mesh(faces: wp.array[wp.int32], name: str) -> None:
    """
    Raise before constructing a ``warp.Mesh`` with zero triangles.

    A ``warp.Mesh`` built with an empty ``indices`` array does not raise, but silently
    corrupts CUDA driver/allocator state: the constructor itself "succeeds", but a later, unrelated
    CUDA allocation anywhere else in the process then fails and cascades into "illegal memory
    access" errors. Every ``wp.Mesh(...)`` call site in this package must call this first instead
    of letting the native constructor run on an empty face buffer.

    Parameters
    ----------
    faces
        Flat ``wp.int32`` triangle index buffer about to be passed to ``warp.Mesh``.
    name
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        If ``faces`` is empty (zero triangles).
    """
    if faces.size == 0:
        raise ValueError(
            f"{name} cannot build a warp.Mesh with zero triangles: this silently corrupts CUDA "
            "state through Warp 1.19 (see the Warp issue tracker for wp.Mesh + empty BVH)."
        )


def require_valid_faces(faces: wp.array[wp.int32], n_vertices: int, name: str) -> None:
    """
    Raise if any index in ``faces`` falls outside ``[0, n_vertices)``.

    An out-of-range face index is not a wrong-answer bug; it is an out-of-bounds read at every
    downstream kernel that indexes ``vertices[faces[...]]`` -- silent glibc heap corruption on the
    CPU device, and a silently wrong or crashing answer on CUDA, with no Python exception either way
    (the same memory-safety class as an unvalidated device mismatch, which is why
    [`require_same_device`][ordito._common.require_same_device] exists). Nothing downstream of a
    well-formed mesh checks this -- several reference libraries this package tests against don't
    either -- so it is a precondition, not a runtime-checked invariant.

    Unlike [`require_nonempty_mesh`][ordito._common.require_nonempty_mesh], this is **not** free:
    it costs a device reduction and a host readback, because it has to read the actual index
    values, not just a shape. That is deliberate: it is meant for the small number of public entry
    points that are the real trust boundary for a mesh's connectivity -- a
    freshly loaded file ([`io.load_mesh`][ordito.io.load_mesh],
    [`io.mesh_from_numpy`][ordito.io.mesh_from_numpy]) or a freshly repaired one
    ([`repair.make_solid`][ordito.repair.make_solid]) -- not every downstream helper, which is
    expected to trust the connectivity it was handed the same way every other per-face kernel
    wrapper in this package already does.

    Parameters
    ----------
    faces
        Flat ``wp.int32`` triangle (or other) index buffer to check.
    n_vertices
        Exclusive upper bound every index in ``faces`` must stay under.
    name
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        If ``faces`` is non-empty and any of its indices is negative or ``>= n_vertices``.
    """
    if faces.size == 0:
        return
    require_indices_below(faces, n_vertices, f"{name}: faces must reference vertex indices in")


def require_same_device(**named: object) -> None:
    """
    Raise if two or more of the given device-bearing arguments disagree on device.

    Warp does not catch this for you. Since Warp 1.14, ``wp.launch`` accepts a cross-device
    argument list without complaint -- the default ``wp.config.launch_array_access_mode`` is
    ``RELAXED``, which passes pointers straight through -- and the two ways such a call then fails
    are both invisible to a Python ``except`` clause: a CUDA launch reading CPU arrays computes the
    *right* answer and corrupts the host heap later, once those arrays are freed while the kernel
    is still running asynchronously; a CPU launch reading CUDA arrays segfaults immediately, with no
    Python exception at all. Neither is something a caller can catch or debug from the traceback it
    gets, so every public function that accepts more than one device-bearing argument calls this
    first, before any of them reaches a kernel.

    Every argument may be ``None`` -- a caller should pass every device-bearing parameter it
    received unconditionally, including an ``X | None = None`` precomputed-cache argument, rather
    than filtering beforehand. A ``list`` or ``tuple`` argument (a sequence of loops, rings, ...) is
    unpacked element-wise, each labelled ``f"{name}[{i}]"`` in the message, rather than compared as
    one opaque object.

    Parameters
    ----------
    **named
        Every device-bearing argument the caller received (arrays, meshes, BVHs, hash grids,
        volumes, or a list/tuple of any of those), keyed by its own parameter name.

    Raises
    ------
    RuntimeError
        If two of the given arguments report a different ``.device``.
    """
    # Scan unlabelled first: the labels are only needed for the message, and building one per
    # element of a long list of loops or rings costs more than the comparison itself.
    first: list[object] = []
    for value in named.values():
        if _first_device_mismatch(value, first):
            _raise_device_mismatch(named)


def _first_device_mismatch(value: object, first: list[object]) -> bool:
    """Whether ``value`` (or an element of it) disagrees with ``first[0]``, seeding it if empty."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        items: Iterable[Any] = value
        head: Any = value[0]
        if all(map(isinstance, value, itertools.repeat((list, tuple)))):
            # A sequence of pairs (``combine.concatenate``'s meshes) flattens one level into the
            # same single pass: a call per pair cost 3x the whole comparison at 4 096 pairs.
            items = itertools.chain.from_iterable(value)
            head = head[0] if head else None
        try:
            # The common case, a flat sequence of arrays, compares without a call per element;
            # ``None`` or a deeper nesting has no ``.device`` and takes the element-wise walk.
            device = head.device
            if not first:
                first.append(device)
            reference = first[0]
            return any(item.device is not reference and item.device != reference for item in items)
        except AttributeError:
            return any(_first_device_mismatch(item, first) for item in value)
    device = getattr(value, "device", None)
    if device is None:
        return False
    if not first:
        first.append(device)
        return False
    return device is not first[0] and device != first[0]


def _raise_device_mismatch(named: dict[str, object]) -> None:
    """Raise naming the first mismatched pair, if the labelled walk confirms the fast scan's."""
    seen: list[tuple[str, object]] = []
    for name, value in named.items():
        seen.extend(_named_devices(name, value))
    if not seen:
        return
    first_name, first_device = seen[0]
    for name, device in seen[1:]:
        if device != first_device:
            raise RuntimeError(
                f"ordito requires every argument to run on one device, but '{first_name}' is on "
                f"{first_device} while '{name}' is on {device}. Move one onto the other's device "
                "(e.g. wp.clone(array, device=...) for a wp.array) before calling this function."
            )


def _named_devices(name: str, value: object) -> list[tuple[str, object]]:
    """Flatten ``value`` into ``(label, device)`` pairs, descending into a list/tuple."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        found: list[tuple[str, object]] = []
        for i, item in enumerate(value):
            found.extend(_named_devices(f"{name}[{i}]", item))
        return found
    device = getattr(value, "device", None)
    return [(name, device)] if device is not None else []


# One scratch buffer per dtype for ``read_scalar`` below, allocated on first use and reused for the
# life of the process, with the NumPy view of it the reads index (built once: a ``.numpy()`` per
# read is a fifth of the read). A single element each, so the whole table is a few dozen bytes.
_SCALAR_SCRATCH: dict[type, tuple[wp.array[Any], np.ndarray[Any, Any]]] = {}


def read_scalar(arr: odt.ArrayNd, index: int = -1) -> Any:
    """
    One element of ``arr``, read back to the host as a Python scalar.

    The spelling matters more than it looks. ``int(arr[n - 1 :].numpy()[0])`` -- a natural first
    attempt -- builds a one-element Warp view, then a *fresh* host array for it, then synchronizes.
    Copying into a scratch buffer allocated once avoids that intermediate array; on a host array,
    where ``.numpy()`` is already a zero-copy view of the whole buffer, indexing that view directly
    is cheaper than slicing first. So the device branch is not a portability concession: each
    side's fast path is the other's slow one.

    !!! warning "The scratch must not be pinned"
        A pinned host destination makes ``cudaMemcpyAsync`` genuinely asynchronous, and Warp issues
        the copy without an event or a synchronization, so the read can race the producing kernel
        and silently return the *previous* round's value instead. Pageable memory is documented to
        return only once a device-to-host copy has completed.

    Parameters
    ----------
    arr
        Warp array to read from. Any scalar dtype; one scratch buffer per dtype is cached.
    index
        Element to read, negative from the end as in Python. Defaults to the last element, which
        is what an inclusive scan's total lives in.

    Returns
    -------
    Any
        The element, as the value ``numpy`` gives for that dtype: a Python scalar for the scalar
        dtypes (``int`` for the integer ones, ``float`` for the floating ones), and a **copy** of
        the row for a vector or matrix dtype. Callers wrap it in ``int(...)`` / ``float(...)``
        where a definite type is wanted.

    Notes
    -----
    Not reentrant: the scratch is shared, so two concurrent readbacks of the same dtype from
    different threads would clobber each other. Nothing in this package reads back off-thread.

    **The copy is load-bearing for the non-scalar dtypes, and its absence is silent.** Indexing a
    ``wp.array[wp.vec3]``'s ``.numpy()`` yields a *view*, so without it two sequential reads of the
    same dtype would both alias the one cached scratch row and the first would take the second's
    value -- ``creation.sweep_polygon`` reads a path's two endpoints back to back and would decide
    every open path was closed. On the host branch the view is onto the caller's own buffer, where
    a caller writing through it would corrupt the array. Scalar dtypes are unaffected (``numpy``
    hands back a scalar, which is already a copy), which is exactly why this hides.
    """
    n = int(arr.shape[0])
    slot = index if index >= 0 else n + index
    device = arr.device
    if device is None or not device.is_cuda:
        return _detached(arr.numpy()[slot])
    cached = _SCALAR_SCRATCH.get(arr.dtype)
    if cached is None:
        buffer = _launch.empty(1, dtype=arr.dtype, device="cpu")
        cached = _SCALAR_SCRATCH[arr.dtype] = (buffer, buffer.numpy())
    scratch, scratch_np = cached
    # An offset copy rather than a copy of ``arr[slot : slot + 1]``: the one-element slice is a
    # whole ``wp.array`` construction to name four bytes ``wp.copy`` can address directly. The
    # offset path needs a contiguous source -- a strided one would be staged whole across devices
    # -- so a strided view keeps the slice, which is contiguous by construction.
    if arr.ndim == 1 and arr.is_contiguous:
        _launch.copy_to_host(scratch, arr, slot, 1)
    else:
        _launch.copy(scratch, arr[slot : slot + 1])
    return _detached(scratch_np[0])


def read_values(arr: wp.array[Any], start: int, count: int) -> list[Any]:
    """
    ``count`` consecutive elements of ``arr`` from ``start``, read back in one copy.

    The several-value counterpart of [`read_scalar`][ordito._common.read_scalar], for a function
    that writes a few small device values side by side and reads them once: a cached pageable
    scratch per ``(dtype, count)`` and an offset copy, rather than a slice view plus ``.numpy()``'s
    fresh host array, which costs over twice as much. Same pageable-scratch rule and the same
    non-reentrancy as ``read_scalar``. ``count`` must be positive -- ``wp.copy`` reads
    ``count=0`` as "the whole source".

    Parameters
    ----------
    arr
        Contiguous rank-1 Warp array of a scalar or vector dtype; a vector element is read
        back as a list.
    start
        First element to read.
    count
        Number of elements to read.

    Returns
    -------
    list[Any]
        The elements as Python scalars.
    """
    if count <= 0:
        return []
    device = arr.device
    if device is None or not device.is_cuda:
        return arr.numpy()[start : start + count].tolist()
    key = (arr.dtype, count)
    cached = _VALUES_SCRATCH.get(key)
    if cached is None:
        buffer = _launch.empty(count, dtype=arr.dtype, device="cpu")
        cached = _VALUES_SCRATCH[key] = (buffer, buffer.numpy())
    scratch, scratch_np = cached
    _launch.copy_to_host(scratch, arr, start, count)
    return scratch_np.tolist()


def read_array(arr: wp.array[Any]) -> np.ndarray:
    """
    ``arr.numpy()``, read back through a fresh pageable host buffer.

    A CUDA array's ``.numpy()`` allocates its host array and copies through ``wp.copy``'s checks;
    ``_launch.empty`` and ``_launch.copy_to_host`` do the same work for a contiguous rank-1 array
    in about 7 us less (14 against 21 us at 513 ``int32``, 44 against 55 at 100 k). Above a
    handful of values it is also the cheaper of this and
    [`read_values`][ordito._common.read_values], whose list costs a microsecond per few dozen
    elements. Anything else, and a host array (where ``.numpy()`` is a view), takes ``.numpy()``.

    Parameters
    ----------
    arr
        Warp array to read back.

    Returns
    -------
    np.ndarray
        What ``arr.numpy()`` returns: a fresh host copy for a CUDA array, of ``arr``'s shape (a
        vector dtype adds its trailing axis).
    """
    device = arr.device
    if device is None or not device.is_cuda or arr.ndim != 1 or not arr.is_contiguous:
        return arr.numpy()
    host = _launch.empty(arr.size, dtype=arr.dtype, device="cpu")
    if arr.size > 0:
        _launch.copy_to_host(host, arr, 0, arr.size)
    return host.numpy()


# ``read_values``' scratch per ``(dtype, count)`` and its NumPy view, as ``_SCALAR_SCRATCH``.
_VALUES_SCRATCH: dict[tuple[type, int], tuple[wp.array[Any], np.ndarray[Any, Any]]] = {}


def index_extrema(indices: odt.ArrayNdInt) -> tuple[int, int]:
    """
    ``(min(0, minimum), max(0, maximum))`` of a non-empty index buffer, in one readback.

    Exact wherever an index check acts -- a negative minimum, a non-negative maximum -- so a caller
    testing ``low < 0`` and ``high >= bound``, or taking ``high + 1`` as a bound once a negative
    is rejected, reads it as it would ``reduce.minmax``'s pair; a message quoting the range
    re-measures with ``reduce.minmax`` on its error path. A contiguous ``wp.int32`` buffer of any
    rank takes one launch of ``kernels/reduce.index_extrema`` into a zeroed two-slot buffer, with
    neither ``minmax``'s seeded buffer and fill nor its dispatch; any other dtype or a strided
    view takes ``reduce.minmax``, clamped the same way.

    Parameters
    ----------
    indices
        Non-empty index buffer of any rank.

    Returns
    -------
    tuple[int, int]
        ``(min(0, minimum), max(0, maximum))``.
    """
    if indices.dtype is not wp.int32 or not indices.is_contiguous:
        low, high = od.reduce.minmax(cast(odt.ScalarArray, indices))
        return min(int(low), 0), max(int(high), 0)
    flat = indices if indices.ndim == 1 else indices.flatten()
    n = int(flat.size)
    device = flat.device
    out = _launch.zeros(2, dtype=wp.int32, device=device)
    _launch.launch_tiled(
        kernel_reduce.index_extrema,
        dim=[kernel_reduce.blocks_1d(n)],
        inputs=[flat, n, 0, out],
        block_dim=TILE_1D,
        device=device,
    )
    # The pair is the answer.
    low, high = read_values(out, 0, 2)
    return int(low), int(high)


def checked_index_bound(
    low: int, high: int, name: str, bound: int | None = None, bound_name: str = ""
) -> int:
    """
    Return the exclusive bound of an index buffer from its extrema, raising if one is out of range.

    The decision every inferred radix or bound check shares, from the pair
    [`index_extrema`][ordito._common.index_extrema] returns (or one a caller reduced into its own
    buffer): a negative ``low`` raises, then a given ``bound`` is checked against ``high`` and
    returned; without one the bound is ``high + 1``.

    Parameters
    ----------
    low
        The buffer's minimum, or ``min(0, minimum)``.
    high
        The buffer's maximum, or ``max(0, maximum)``.
    name
        What the indices are, as the error message names them (``"faces"``).
    bound
        Exclusive bound every index must stay under, or ``None`` to infer it.
    bound_name
        What ``bound`` is, as the error message names it (``"n_vertices"``).

    Returns
    -------
    int
        ``bound`` when given, else ``high + 1``.

    Raises
    ------
    ValueError
        If ``low`` is negative, or ``bound`` is given and ``high`` reaches it.
    """
    if low < 0:
        raise ValueError(f"{name} must be non-negative, got a minimum of {low}")
    if bound is None:
        return high + 1
    if high >= bound:
        raise ValueError(f"{name} must be less than {bound_name} {bound}, got a maximum of {high}")
    return bound


def require_indices_below(
    indices: odt.ArrayNdInt, bound: int, prefix: str, suffix: str = ""
) -> None:
    """
    Raise if any entry of the non-empty ``indices`` falls outside ``[0, bound)``.

    One [`index_extrema`][ordito._common.index_extrema] on the path that does not raise; the
    message quotes the exact range, so the error path re-measures it with ``reduce.minmax``:
    ``f"{prefix} [0, {bound}){suffix}, got a range of [{low}, {high}]"``.

    Parameters
    ----------
    indices
        Non-empty index buffer of any rank.
    bound
        Exclusive upper bound every entry must stay under.
    prefix
        Message text before the interval.
    suffix
        Message text after the interval.

    Raises
    ------
    ValueError
        If any entry is negative or ``>= bound``.
    """
    low, high = index_extrema(indices)
    if low < 0 or high >= bound:
        # The range the message quotes.
        low, high = od.reduce.minmax(cast(odt.ScalarArray, indices))
        raise ValueError(f"{prefix} [0, {bound}){suffix}, got a range of [{low}, {high}]")


def _detached(value: T) -> T:
    """Copy ``value`` when it is a view, so a vector or matrix element outlives the next read."""
    return value.copy() if isinstance(value, np.ndarray) else value


def scan_count(flags: wp.array[Any]) -> int:
    """
    Scan integer ``flags`` inclusively in place and return their total: one tail readback.

    An inclusive scan of ``0`` / ``1`` flags (or counts) ends at their sum, so the one 4-byte read
    of [`read_scalar`][ordito._common.read_scalar] sizes the outputs where a reduction would read
    the whole array. Applied to the ``tail`` of [`offsets_buffer`][ordito._common.offsets_buffer]
    it turns the counts written there into total-terminated offsets and returns the total.
    """
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    return int(read_scalar(flags))


def scan_count_checked(offsets: wp.array[wp.int32], what: str) -> int:
    """
    Scan the counts in ``offsets[1:-1]`` in place and return their total, refusing a wrapped scan.

    ``offsets`` is ``(n + 2,)`` and zeroed but for the counts: a leading zero, the ``n`` counts,
    and a trailing slot that ``kernels/array.flag_wrapped_scan`` sets when the ``int32`` scan
    wrapped, read back with the total in one copy. A total past ``2**32`` wraps all the way round
    to a small positive count, so its sign alone does not tell. ``offsets[: n + 1]`` are then the
    total-terminated offsets. Use it, not [`scan_count`][ordito._common.scan_count], wherever the
    counts are per-row products that can pass ``2**31`` and the total sizes a buffer.

    Raises
    ------
    ValueError
        If the counts sum past ``2**31 - 1``; the message names ``what``.
    """
    n = offsets.size - 2
    counts = _launch.view(offsets, 1, n + 1)
    _launch.array_scan(counts, out_array=counts, inclusive=True)
    _launch.launch(
        kernel_array.flag_wrapped_scan,
        dim=n,
        inputs=[counts, _launch.view(offsets, n + 1, n + 2)],
        device=offsets.device,
    )
    # Host readback: only the device knows the scan total, and the caller sizes buffers from it.
    total, wrapped = read_values(offsets, n, 2)
    if wrapped:
        raise ValueError(
            f"{what} has more entries than int32 indices address; process a coarser mesh or "
            "smaller pieces"
        )
    return int(total)


def offsets_buffer(n: int, device: wp.DeviceLike) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Allocate zeroed ``(n + 1,)`` offsets and return them with their tail view ``offsets[1:]``.

    A producer writes its ``n`` counts into the tail; scanning the tail in place
    ([`scan_count`][ordito._common.scan_count]) then leaves the total-terminated offsets, the
    leading zero being the first one, with no separate count buffer. ``n`` must be positive.
    """
    offsets = _launch.zeros(n + 1, dtype=wp.int32, device=device)
    return offsets, _launch.view(offsets, 1, n + 1)


def staging_pool(
    n_walks: int, points_per_walk: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Allocate the pool a serial walk's counting pass stages its points in, and its chunk links.

    The walk-once form of a counting pass and a writing pass over the same serial walks: the
    counting pass stages each walk's points (``kernels/array.stage_point``) and the writing pass
    copies them (``copy_staged``), walking again only a walk the pool ran out on. The walks share
    the pool, so ``points_per_walk`` sizes the total (``n_walks`` times it, at most
    ``STAGED_POINTS``), in one allocation; the links' last slot, the chunk counter, starts past the
    walks' own first chunks. Returns ``(staged_points, staged_next)`` as the kernels take them.
    """
    chunk = int(kernel_array.STAGED_CHUNK)
    n_chunks = min(n_walks * -(-points_per_walk // chunk), STAGED_POINTS // chunk)
    staged_next, points = _launch.carve_typed(
        ((wp.int32, n_chunks + 1), (wp.vec3, n_chunks * chunk)), device
    )
    _launch.fill_(_launch.view(staged_next, n_chunks, n_chunks + 1), min(n_walks, n_chunks))
    return points, staged_next


# The most points one call stages (12 MiB), however many walks: what it does not hold is walked
# twice, as every walk was before the pool. Its own size is the peak memory it adds
# (``geodesic_walk.descend_field``: 4 096 paths on a sphere of 82 k faces, 12.7 -> 25 MiB).
STAGED_POINTS = 1 << 20


def hilbert_point_order(points: wp.array[wp.vec3]) -> wp.array[wp.int32]:
    """
    Return ``(n,)`` the points' indices in Hilbert order over their bounding box.

    A launch over the points in this order puts a compact patch in each block: a block-local
    iteration then carries a value across the patch (``geodesic_walk``'s shortest-edge tree). One
    box reduction, one key pass and a 30-bit radix sort, with no readback.
    """
    device = points.device
    n = points.size
    corners = _launch.full(6, math.inf, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_reduce.minmax_vec3_chunked,
        dim=kernel_reduce.chunks_1d(n),
        inputs=[points, corners],
        device=device,
    )
    keys, order = _launch.carve(wp.int32, (2 * n, 2 * n), device)
    _launch.launch(
        kernel_array.point_hilbert_keys, dim=n, inputs=[points, corners, keys, order], device=device
    )
    _launch.radix_sort_pairs(keys, order, count=n, end_bit=30)
    return _launch.view(order, 0, n)
