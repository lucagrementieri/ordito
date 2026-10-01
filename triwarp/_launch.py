"""
Private kernel launcher: ``wp.launch`` / ``wp.launch_tiled`` with the per-call resolution cached.

``wp.launch`` re-derives, on every call, everything that is fixed for a kernel on a device: it
re-resolves the device, re-checks that the module is loaded and current, re-reads the kernel's
hooks, and marshals every argument through ``pack_arg``'s general type dispatch. For the short
launches that make up most of this package's host time that resolution *is* the launch, so
[`launch`][triwarp._launch.launch] resolves it once per ``(kernel, CUDA context, block_dim)`` and
keeps one packer per parameter, then calls the native launch directly.

It is ``wp.launch`` exactly wherever it can be: anything the cache does not cover -- the CPU device,
a generic kernel, an explicit stream, an adjoint or recorded launch, an active tape, an API
(APIC) capture, deterministic or clustered kernels, ``verify_cuda`` / ``print_launches``, an
argument that is not already the parameter's exact type -- goes to ``wp.launch`` unchanged, and
so does every launch whose module has been modified since the cache entry was made. A launch
recorded into a CUDA graph capture keeps the cached path and registers its module executable with
the graph, as ``wp.launch`` does. An argument a
packer does not recognise is marshalled by Warp's own ``pack_arg``, so conversions and error
messages are Warp's.
"""

# This module drives Warp's launch machinery directly (``_launch_bounds_classes``, a stream's
# ``_stream``, the runtime's ``_apic_capture``, ``_raise_cuda_launch_error``), which is its purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import ctypes
import struct
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import numpy as np
import warp as wp
import warp._src.codegen as _codegen
import warp._src.context as _ctx
import warp._src.types as _types

if TYPE_CHECKING:
    import triwarp.typing as twt

_RELAXED = wp.config.LaunchArrayAccessMode.RELAXED
_Device = _ctx.Device
_bounds_classes = _types._launch_bounds_classes
_pack_arg = _ctx.pack_arg
_types_equal = _types.types_equal
_c_void_p_arrays: dict[int, Any] = {}
_SCALAR_PY = (int, float, bool)
_HALF = (_types.float16, _types.bfloat16)
_ARRAY = wp.array
DType = TypeVar("DType")
# Any Warp array type, kept whole: a union of array types in is the same union out.
ArrayT = TypeVar("ArrayT", bound="wp.array[Any, Any]")


class _Entry:
    """One kernel's launch state on one CUDA context and block size."""

    __slots__ = ("exec_", "hashers", "hooks", "kernel_dim", "packers", "pool", "tid_limit")

    def __init__(self, kernel: Any, exec_: Any, hashers: dict[Any, Any], hooks: Any) -> None:
        self.exec_ = exec_
        self.hashers = hashers
        self.hooks = hooks
        self.kernel_dim = kernel.adj.kernel_dim
        self.tid_limit = kernel.adj.scalar_tid_extent_limit_candidate
        self.packers = tuple(_packer(kernel, arg) for arg in kernel.adj.args)
        block = _Block.build(kernel, self.kernel_dim)
        # A list, so a launch takes the block with an atomic ``pop``: a second thread launching the
        # same kernel while the first is inside the native call (ctypes releases the GIL) finds the
        # pool empty and marshals per argument instead of overwriting a block being read.
        self.pool = [] if block is None else [block]


# -- the parameter block ---------------------------------------------------------------------------
#
# ``cuLaunchKernel`` takes an array of pointers, one per kernel parameter, and copies the values
# they point at before it returns, so the storage can be reused by the next launch. A block is one
# buffer laid out as the kernel's parameters (the launch bounds, then every argument at 8-byte
# alignment) plus a pointer table built once into it; a launch fills the whole buffer with one
# ``struct.pack_into``. That replaces a ctypes struct per argument -- for a freshly allocated array,
# the ``array_t`` Warp builds and caches on it -- and the pointer table built per call.

_ARRAY_FORMAT = "QQ4i4iHH4x"  # ``array_t``: data, grad, shape[4], strides[4], ndim, flags
_ARRAY_NULL = (0,) * 12
_SCALAR_CHARS = {
    ctypes.c_int8: "b",
    ctypes.c_uint8: "B",
    ctypes.c_int16: "h",
    ctypes.c_uint16: "H",
    ctypes.c_int32: "i",
    ctypes.c_uint32: "I",
    ctypes.c_int64: "q",
    ctypes.c_uint64: "Q",
    ctypes.c_float: "f",
    ctypes.c_double: "d",
    ctypes.c_bool: "?",
}
_KIND_ARRAY, _KIND_SCALAR, _KIND_VALUE = 0, 1, 2


class _Block:
    """A kernel's parameters as one buffer, filled per launch by one ``struct.pack_into``."""

    __slots__ = ("buffer", "kinds", "pack_into", "params")

    def __init__(self, layout: struct.Struct, kinds: tuple[Any, ...], offsets: list[int]) -> None:
        self.pack_into = layout.pack_into
        self.kinds = kinds
        self.buffer = ctypes.create_string_buffer(max(layout.size, 8))
        base = ctypes.addressof(self.buffer)
        self.params = (ctypes.c_void_p * len(offsets))(*[base + o for o in offsets])

    @staticmethod
    def build(kernel: Any, kernel_dim: int) -> _Block | None:
        """Return the block for ``kernel``, or ``None`` if a parameter type has no packed form."""
        bounds_size = ctypes.sizeof(_bounds_classes[kernel_dim])
        fmt = ["=", f"{kernel_dim}i", "4x" if kernel_dim % 2 else "", "QQ"]
        offsets = [0]
        kinds = []
        offset = bounds_size
        for arg in kernel.adj.args:
            arg_type = arg.type
            pad = -offset % 8
            if pad:
                fmt.append(f"{pad}x")
                offset += pad
            offsets.append(offset)
            if _types.is_array(arg_type) and _types.concrete_array_type(arg_type) is _ARRAY:
                fmt.append(_ARRAY_FORMAT)
                kinds.append((_KIND_ARRAY, arg_type.ndim, {arg_type.dtype}, arg_type.dtype))
                offset += 56
            elif isinstance(arg_type, type) and issubclass(arg_type, ctypes.Array):
                size = ctypes.sizeof(arg_type)
                fmt.append(f"{size}s")
                kinds.append((_KIND_VALUE, 0, {arg_type}, arg_type))
                offset += size
            elif (
                isinstance(arg_type, type)
                and arg_type not in _HALF
                and getattr(arg_type, "_type_", None) in _SCALAR_CHARS
            ):
                char = _SCALAR_CHARS[arg_type._type_]
                fmt.append(char)
                kinds.append((_KIND_SCALAR, 0, None, arg_type))
                offset += struct.calcsize("=" + char)
            else:
                return None
        layout = struct.Struct("".join(fmt))
        if layout.size != offset or bounds_size != struct.calcsize("".join(fmt[:4])):
            return None
        return _Block(layout, tuple(kinds), offsets)

    def fill(
        self,
        extent: tuple[Any, ...],
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        device: Any,
        strict: bool,
    ) -> bool:
        """Pack ``extent`` and the arguments; ``False`` if an argument has no packed form."""
        shape, size, coord_mult = extent
        values = [*shape, size, coord_mult]
        append = values.append
        extend = values.extend
        kinds = self.kinds
        i = 0
        for group in (inputs, outputs):
            for value in group:
                kind, ndim, accepted, target = kinds[i]
                i += 1
                if kind == _KIND_ARRAY:
                    if value is None:
                        extend(_ARRAY_NULL)
                        continue
                    if type(value) is not _ARRAY or value.ndim != ndim or value._grad is not None:
                        return False
                    dtype = value.dtype
                    if dtype not in accepted and not _accept(accepted, dtype, target):
                        return False
                    if strict and value.device is not device:
                        return False
                    ptr = value.ptr or 0
                    sh = value.shape
                    st = value.strides
                    if ndim == 1:
                        extend((ptr, 0, sh[0], 0, 0, 0, st[0], 0, 0, 0, 1, 0))
                    elif ndim == 2:
                        extend((ptr, 0, sh[0], sh[1], 0, 0, st[0], st[1], 0, 0, 2, 0))
                    elif ndim == 3:
                        extend((ptr, 0, *sh, 0, *st, 0, 3, 0))
                    else:
                        extend((ptr, 0, *sh, *st, 4, 0))
                elif kind == _KIND_SCALAR:
                    value_type = type(value)
                    if value_type in _SCALAR_PY:
                        append(value)
                    elif value_type is target:
                        append(value.value)
                    else:
                        return False
                else:
                    value_type = type(value)
                    if value_type not in accepted and not _accept(accepted, value_type, target):
                        return False
                    append(bytes(value))
        try:
            self.pack_into(self.buffer, 0, *values)
        except struct.error:
            # A value outside its C type's range (ctypes would wrap it) or of the wrong kind: the
            # per-argument path marshals it exactly as Warp does.
            return False
        return True


def _packer(kernel: Any, arg: Any) -> Any:
    """Return a function packing one argument value for ``arg``, or ``None`` on a miss."""
    arg_type = arg.type
    label = arg.label

    def generic(value: Any, device: Any) -> Any:
        return _pack_arg(kernel, arg_type, label, value, device)

    if _types.is_array(arg_type):
        cls = _types.concrete_array_type(arg_type)
        dtype = arg_type.dtype
        ndim = arg_type.ndim
        null = arg_type.__ctype__()

        # A kernel generated for a generic signature (``wp.map``, an overload) carries its own
        # ``vec_t`` / ``mat_t`` classes, equal to ``wp.vec3`` / ``wp.mat44`` but not identical,
        # so an element type is accepted by identity and, once ``types_equal`` agrees, remembered.
        dtypes = {dtype}

        def pack_array(value: Any, device: Any) -> Any:
            if type(value) is cls and value.ndim == ndim:
                value_dtype = value.dtype
                if value_dtype in dtypes or _accept(dtypes, value_dtype, dtype):
                    return value.__ctype__()
            if value is None:
                return null
            return generic(value, device)

        return pack_array, True

    if isinstance(arg_type, _codegen.Struct):

        def pack_struct(value: Any, _device: Any) -> Any:
            return value.__ctype__()

        return pack_struct, False

    if isinstance(arg_type, type) and issubclass(arg_type, ctypes.Array):
        value_types = {arg_type}

        def pack_value(value: Any, device: Any) -> Any:
            value_type = type(value)
            if value_type in value_types or _accept(value_types, value_type, arg_type):
                return value
            return generic(value, device)

        return pack_value, False

    if isinstance(arg_type, type) and hasattr(arg_type, "_type_") and arg_type not in _HALF:
        ctor = arg_type._type_

        def pack_scalar(value: Any, device: Any) -> Any:
            if type(value) in _SCALAR_PY:
                return ctor(value)
            if type(value) is arg_type:
                return ctor(value.value)
            return generic(value, device)

        return pack_scalar, False

    return generic, False


def _accept(accepted: set[Any], candidate: Any, target: Any) -> bool:
    """Whether ``candidate`` is Warp-equal to ``target``; remember it in ``accepted`` if so."""
    if _types_equal(candidate, target):
        accepted.add(candidate)
        return True
    return False


def _entry(kernel: Any, device: Any, block_dim: int) -> _Entry | Literal[False] | None:
    """
    Return ``kernel``'s cache entry on ``device``.

    ``None`` if absent or stale, ``False`` if the kernel is not eligible for the cached path.
    """
    cache = kernel.__dict__.get("_tw_launch")
    if cache is None:
        return None
    key = (device.context, block_dim)
    entry = cache.get(key)
    if not entry:
        return entry
    module = kernel.module
    if (
        module.hashers is not entry.hashers
        or module.execs.get(key) is not entry.exec_
        or module.has_unresolved_static_expressions
    ):
        return None
    return entry


def _remember(kernel: Any, device: Any, block_dim: int) -> None:
    """After a ``wp.launch`` of ``kernel``, cache its launch state if it is eligible."""
    module = kernel.module
    key = (device.context, block_dim)
    exec_ = module.execs.get(key)
    if exec_ is None or block_dim not in module.hashers or module.has_unresolved_static_expressions:
        return
    cache = kernel.__dict__.setdefault("_tw_launch", {})
    hooks = exec_.get_kernel_hooks(kernel)
    det = hooks.det_launch_meta
    if (
        hooks.forward is None
        or hooks.cluster_dim > 1
        or (det is not None and det.needs_deterministic)
        or not kernel.grid_stride
    ):
        cache[key] = False
        return
    cache[key] = _Entry(kernel, exec_, module.hashers, hooks)


def launch(
    kernel: twt.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence[Any] = (),
    outputs: Sequence[Any] = (),
    adj_inputs: Sequence[Any] = (),
    adj_outputs: Sequence[Any] = (),
    device: Any = None,  # a ``wp.DeviceLike``; ``Any`` because it is rebound to a ``Device``
    stream: Any = None,
    adjoint: bool = False,
    record_tape: bool = True,
    record_cmd: bool = False,
    max_blocks: int = 0,
    block_dim: int = 256,
) -> Any:
    """Launch ``kernel`` exactly as ``wp.launch`` would, with its resolution cached."""
    runtime = _ctx.runtime
    kernel_any: Any = kernel
    if type(device) is not _Device:
        # ``_ctx.runtime`` is ``None`` until ``wp.init()``; Warp annotates it as always set.
        device = None if runtime is None else runtime.get_device(device)  # pyright: ignore[reportUnnecessaryComparison]
    if (
        device is None
        or not device.is_cuda
        or stream is not None
        or adjoint
        or record_cmd
        or adj_inputs
        or adj_outputs
        or kernel_any.is_generic
        or runtime.tape is not None
        or runtime._apic_capture is not None
        or wp.config.verify_cuda
        or wp.config.print_launches
    ):
        return _fallback(
            kernel,
            dim,
            inputs,
            outputs,
            adj_inputs,
            adj_outputs,
            device,
            stream,
            adjoint,
            record_tape,
            record_cmd,
            max_blocks,
            block_dim,
        )
    if block_dim <= 0:
        block_dim = 256
    entry = _entry(kernel, device, block_dim)
    if not entry:
        return _slow(kernel, dim, inputs, outputs, device, max_blocks, block_dim, entry)

    extent = _extent(dim, entry)
    if extent is None:
        return _slow(kernel, dim, inputs, outputs, device, max_blocks, block_dim, entry)

    packers = entry.packers
    n_args = len(inputs) + len(outputs)
    if n_args != len(packers):
        return _fallback(
            kernel,
            dim,
            inputs,
            outputs,
            (),
            (),
            device,
            None,
            False,
            record_tape,
            False,
            max_blocks,
            block_dim,
        )
    strict = wp.config.launch_array_access_mode is not _RELAXED
    hooks = entry.hooks
    current: Any = device._stream  # always set on a CUDA device; Warp types it optional for the CPU
    if runtime.captures:
        _retain_for_capture(runtime, device, entry)
    pool = entry.pool
    if pool:
        try:
            block = pool.pop()
        except IndexError:
            block = None
        if block is not None:
            try:
                if block.fill(extent, inputs, outputs, device, strict):
                    if runtime.core.wp_cuda_launch_kernel(
                        device.context,
                        hooks.forward,
                        extent[1],
                        max_blocks,
                        block_dim,
                        1,
                        1,
                        hooks.forward_smem_bytes,
                        block.params,
                        current.cuda_stream,
                        None,
                    ):
                        _ctx._raise_cuda_launch_error(kernel_any, device, hooks, False)
                    return None
            finally:
                pool.append(block)

    shape, size, coord_mult = extent
    bounds = _bounds_classes[len(shape)](shape)
    if coord_mult != 1:
        bounds.coord_mult = coord_mult
        bounds.size = size
    params = [bounds]
    i = 0
    for values in (inputs, outputs):
        for value in values:
            pack, is_array = packers[i]
            if (
                strict
                and is_array
                and value is not None
                and getattr(value, "device", device) != device
            ):
                return _fallback(
                    kernel,
                    dim,
                    inputs,
                    outputs,
                    (),
                    (),
                    device,
                    None,
                    False,
                    record_tape,
                    False,
                    max_blocks,
                    block_dim,
                )
            params.append(pack(value, device))
            i += 1

    n_params = len(params)
    array_type = _c_void_p_arrays.get(n_params)
    if array_type is None:
        array_type = _c_void_p_arrays[n_params] = ctypes.c_void_p * n_params
    kernel_params = array_type(*[ctypes.addressof(p) for p in params])
    if runtime.core.wp_cuda_launch_kernel(
        device.context,
        hooks.forward,
        bounds.size,
        max_blocks,
        block_dim,
        1,
        1,
        hooks.forward_smem_bytes,
        kernel_params,
        current.cuda_stream,
        None,
    ):
        _ctx._raise_cuda_launch_error(kernel_any, device, hooks, False)
    return None


def _retain_for_capture(runtime: Any, device: Any, entry: _Entry) -> None:
    """
    Register the kernel's module executable with a graph recording ``device``'s stream.

    A CUDA graph capture records the launch itself (and copies its parameters into the graph node),
    so the only thing ``wp.launch`` adds is keeping the executable alive until the graph is
    released; this is that step.
    """
    stream = device._stream.cuda_stream
    if runtime.core.wp_cuda_stream_is_capturing(stream):
        graph = runtime.captures.get(runtime.core.wp_cuda_stream_get_capture_id(stream))
        if graph is not None:
            graph._retain_module_exec(entry.exec_)


def _extent(dim: Any, entry: _Entry) -> tuple[tuple[int, ...], int, int] | None:
    """
    Return ``dim`` as ``(shape, size, coord_mult)``, as ``_build_launch_bounds_from_tuple`` sets it.

    ``shape`` has the kernel's own rank: a shorter ``dim`` is padded with 1, a longer one folds its
    trailing extents into ``coord_mult``. ``None`` for anything that is not a tuple of positive
    ints within the scalar ``wp.tid()`` limit.
    """
    kernel_dim = entry.kernel_dim
    if type(dim) is int:
        if dim <= 0 or dim > entry.tid_limit:
            return None
        if kernel_dim == 1:
            return (dim,), dim, 1
        return (dim,) + (1,) * (kernel_dim - 1), dim, 1
    if type(dim) is not tuple:
        dim = tuple(dim)
    n = len(dim)
    if n == 0 or n > 4:
        return None
    size = 1
    for extent in dim:
        if type(extent) is not int or extent <= 0:
            return None
        size *= extent
    if dim[0] > entry.tid_limit:
        return None
    if n == kernel_dim:
        return dim, size, 1
    if n < kernel_dim:
        return dim + (1,) * (kernel_dim - n), size, 1
    coord_mult = 1
    for extent in dim[kernel_dim:]:
        coord_mult *= extent
    return dim[:kernel_dim], size, coord_mult


def _fallback(
    kernel: twt.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    adj_inputs: Sequence[Any],
    adj_outputs: Sequence[Any],
    device: wp.DeviceLike,
    stream: wp.Stream | None,
    adjoint: bool,
    record_tape: bool,
    record_cmd: bool,
    max_blocks: int,
    block_dim: int,
) -> Any:
    """``wp.launch`` itself, for every launch the cache does not cover."""
    return wp.launch(
        kernel,
        dim,
        inputs=inputs,
        outputs=outputs,
        adj_inputs=adj_inputs,
        adj_outputs=adj_outputs,
        device=device,
        stream=stream,
        adjoint=adjoint,
        record_tape=record_tape,
        record_cmd=record_cmd,
        max_blocks=max_blocks,
        block_dim=block_dim,
    )


def _slow(
    kernel: twt.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    device: wp.Device,
    max_blocks: int,
    block_dim: int,
    entry: _Entry | Literal[False] | None,
) -> None:
    """Launch through ``wp.launch`` and, if the kernel had no cache entry, make one."""
    wp.launch(kernel, dim, inputs=inputs, outputs=outputs, device=device, max_blocks=max_blocks,
              block_dim=block_dim)  # fmt: skip
    if entry is None:
        _remember(kernel, device, block_dim)


def launch_tiled(
    kernel: twt.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence[Any] = (),
    outputs: Sequence[Any] = (),
    device: wp.DeviceLike = None,
    block_dim: int = 256,
    **kwargs: Any,
) -> Any:
    """``wp.launch_tiled`` through [`launch`][triwarp._launch.launch]."""
    if type(device) is not _Device:
        device = wp.get_device(device)
    if device.is_cpu or kwargs:
        return wp.launch_tiled(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device,
                               block_dim=block_dim, **kwargs)  # fmt: skip
    dim = (dim,) if isinstance(dim, int) else tuple(dim)
    return launch(kernel, (*dim, block_dim), inputs, outputs, device=device, block_dim=block_dim)


_map_kernels: dict[tuple[Any, ...], tuple[twt.Kernel, tuple[set[Any], ...]]] = {}
_is_array = _types.is_array


def map(  # noqa: A001 - mirrors ``wp.map``
    func: Any, *inputs: Any, out: Any = None, block_dim: int = 256, device: wp.DeviceLike = None
) -> Any:
    """
    ``wp.map`` with its kernel lookup cached, launched through [`launch`][triwarp._launch.launch].

    The cache key is ``wp.map``'s own -- per input the array class, dtype, rank and broadcast mask,
    or the scalar's type -- plus the output classes. Anything else (no output, an input that is
    neither an array nor a plain scalar or vector, a gradient-tracked input) is ``wp.map`` itself.
    """
    raw_out: Any = (
        out  # ``out`` is narrowed by the ``isinstance`` below; ``wp.map`` wants it as given
    )
    key = [func]  # the object, not id(): a cached id could be recycled by a new function
    shape = None
    broadcast = False
    for value in inputs:
        if _is_array(value):
            value_shape = value.shape
            if shape is None:
                shape = value_shape
                if device is None:
                    device = value.device
            elif value_shape != shape:
                broadcast = True
            if getattr(value, "requires_grad", False):
                return wp.map(func, *inputs, out=raw_out, block_dim=block_dim, device=device)
            key.append((type(value), value.dtype, value.ndim, tuple(d == 1 for d in value_shape)))
        else:
            value_type = type(value)
            if value_type not in _SCALAR_PY and not (
                hasattr(value_type, "_wp_scalar_type_") or value_type in _types.scalar_types
            ):
                return wp.map(func, *inputs, out=raw_out, block_dim=block_dim, device=device)
            key.append(value_type)
    if out is None or shape is None or broadcast:
        return wp.map(func, *inputs, out=raw_out, block_dim=block_dim, device=device)
    outputs = out if isinstance(out, (list, tuple)) else (out,)
    for o in outputs:
        if not _is_array(o) or o.shape != shape:
            return wp.map(func, *inputs, out=raw_out, block_dim=block_dim, device=device)
        key.append((type(o), o.dtype))
    key = tuple(key)
    cached = _map_kernels.get(key)
    if cached is None:
        kernel: Any = wp.map(
            func, *inputs, out=raw_out, return_kernel=True, block_dim=block_dim, device=device
        )
        out_dtypes = tuple({arg.type.dtype} for arg in kernel.adj.args[len(inputs) :])
        if len(out_dtypes) != len(outputs):
            return wp.map(func, *inputs, out=raw_out, block_dim=block_dim, device=device)
        cached = _map_kernels[key] = (kernel, out_dtypes)
    kernel, out_dtypes = cached
    for o, dtypes in zip(outputs, out_dtypes, strict=True):
        if o.dtype not in dtypes and not _accept(dtypes, o.dtype, next(iter(dtypes))):
            return wp.map(func, *inputs, out=raw_out, block_dim=block_dim, device=device)
    launch(kernel, shape, inputs, outputs, device=device, block_dim=block_dim)
    return out


# -- Warp's utility wrappers ---------------------------------------------------------------------
#
# ``wp.utils.array_scan``, ``wp.utils.radix_sort_pairs``, ``wp.copy`` and ``array.fill_`` /
# ``zero_`` are each one native call behind several microseconds of validation, dtype dispatch and
# API-capture bookkeeping. For the case every triwarp call site is in -- contiguous CUDA arrays on
# one device, no API capture active -- the functions below make the same native call directly;
# anything else, including every input Warp would reject, is handed to Warp unchanged, so errors
# and the edge cases stay Warp's.

_SCAN_NATIVE = {
    _types.int32: "wp_array_scan_int_device",
    _types.int64: "wp_array_scan_int64_device",
    _types.float32: "wp_array_scan_float_device",
    _types.float64: "wp_array_scan_double_device",
}
_SORT_NATIVE = {
    _types.int32: ("wp_radix_sort_pairs_int_device", 32),
    _types.uint32: ("wp_radix_sort_pairs_uint_device", 32),
    _types.float32: ("wp_radix_sort_pairs_float_device", 32),
    _types.int64: ("wp_radix_sort_pairs_int64_device", 64),
    _types.uint64: ("wp_radix_sort_pairs_uint64_device", 64),
    _types.float64: ("wp_radix_sort_pairs_double_device", 64),
}
_scan_dispatch: dict[Any, tuple[Any, int, int] | None] = {}


def _plain_cuda(runtime: Any, *arrays: Any) -> bool:
    """Whether every array is a contiguous ``wp.array`` on one CUDA device, outside API capture."""
    if runtime is None or runtime._apic_capture is not None:
        return False
    device = arrays[0].device
    for arr in arrays:
        if type(arr) is not _ARRAY or not arr.is_contiguous or arr.device is not device:
            return False
    return device.is_cuda


def _scan_native(dtype: Any) -> tuple[Any, int, int] | None:
    """``(native function, components, element bytes)`` scanning ``dtype``, or ``None``."""
    if dtype not in _scan_dispatch:
        native = None
        if _types.type_is_scalar(dtype) or _types.type_is_vector(dtype):
            name = _SCAN_NATIVE.get(_types.type_scalar_type(dtype))
            if name is not None:
                native = (
                    getattr(_ctx.runtime.core, name),
                    _types.type_size(dtype),
                    _type_size(dtype),
                )
        _scan_dispatch[dtype] = native
    return _scan_dispatch[dtype]


def array_scan(in_array: Any, out_array: Any, inclusive: bool = True) -> None:
    """``wp.utils.array_scan``, calling the native scan directly for contiguous CUDA arrays."""
    runtime = _ctx.runtime
    size = in_array.size
    dtype = in_array.dtype
    if (
        size
        and out_array.size == size
        and out_array.dtype is dtype
        and _plain_cuda(runtime, in_array, out_array)
    ):
        native = _scan_native(dtype)
        if native is not None:
            function, components, itemsize = native
            function(in_array.ptr, out_array.ptr, size, itemsize, itemsize, components, inclusive)
            return
    wp.utils.array_scan(in_array, out_array, inclusive=inclusive)


def radix_sort_pairs(
    keys: Any, values: Any, count: int, begin_bit: int = 0, end_bit: int | None = None
) -> None:
    """``wp.utils.radix_sort_pairs``, calling the native sort directly for contiguous CUDA keys."""
    runtime = _ctx.runtime
    native = _SORT_NATIVE.get(keys.dtype)
    if (
        native is not None
        and type(count) is int
        and count > 0
        and keys.size >= 2 * count
        and values.size >= 2 * count
        and _type_size(values.dtype) in (4, 8)
        and _plain_cuda(runtime, keys, values)
    ):
        name, width = native
        if end_bit is None:
            end_bit = width
        if type(begin_bit) is int and type(end_bit) is int and 0 <= begin_bit <= end_bit <= width:
            if begin_bit != end_bit:
                getattr(runtime.core, name)(
                    keys.ptr, values.ptr, count, begin_bit, end_bit, _type_size(values.dtype)
                )
            return
    wp.utils.radix_sort_pairs(keys, values, count, begin_bit=begin_bit, end_bit=end_bit)


def copy(dest: Any, src: Any, dest_offset: int = 0, src_offset: int = 0, count: int = 0) -> None:
    """``wp.copy``, calling the native device-to-device copy directly for one CUDA device."""
    runtime = _ctx.runtime
    if (
        type(dest_offset) is int
        and type(src_offset) is int
        and type(count) is int
        and dest_offset >= 0
        and src_offset >= 0
        and count >= 0
        and _plain_cuda(runtime, dest, src)
    ):
        itemsize = _type_size(src.dtype)
        if _type_size(dest.dtype) == itemsize:
            if count == 0:
                count = src.size
            if src_offset + count <= src.size and dest_offset + count <= dest.size:
                if count == 0:
                    return
                device = dest.device
                if not runtime.core.wp_memcpy_d2d(
                    device.context,
                    dest.ptr + dest_offset * itemsize,
                    src.ptr + src_offset * itemsize,
                    count * itemsize,
                    device._stream.cuda_stream,
                ):
                    raise RuntimeError(f"Warp copy error: {runtime.get_error_string()}")
                return
    wp.copy(dest, src, dest_offset=dest_offset, src_offset=src_offset, count=count)


def fill_(arr: Any, value: Any) -> None:
    """``arr.fill_(value)``, as one native memset or memtile for a contiguous CUDA scalar array."""
    runtime = _ctx.runtime
    if arr.size and _plain_cuda(runtime, arr):
        dtype = arr.dtype
        ctor: Any = getattr(dtype, "_type_", None)
        if ctor in _SCALAR_CHARS and dtype not in _HALF and not issubclass(dtype, ctypes.Array):
            device = arr.device
            if _is_zero(value):
                runtime.core.wp_memset_device(
                    device.context, arr.ptr, 0, arr.size * _type_size(dtype), _WP_CURRENT_STREAM
                )
                arr._is_read = False
                return
            value_type = type(value)
            if value_type in _SCALAR_PY or value_type is dtype:
                cvalue = ctor(value if value_type in _SCALAR_PY else value.value)
                runtime.core.wp_memtile_device(
                    device.context, arr.ptr, ctypes.byref(cvalue), ctypes.sizeof(cvalue), arr.size
                )
                arr._is_read = False
                return
    arr.fill_(value)


def zero_(arr: Any) -> None:
    """``arr.zero_()``, as one native memset for a contiguous CUDA array."""
    runtime = _ctx.runtime
    if arr.size and _plain_cuda(runtime, arr):
        device = arr.device
        runtime.core.wp_memset_device(
            device.context, arr.ptr, 0, arr.size * _type_size(arr.dtype), _WP_CURRENT_STREAM
        )
        arr._is_read = False
        return
    arr.zero_()


# -- allocation ------------------------------------------------------------------------------------
#
# ``wp.empty`` builds an array through ``wp.array.__init__`` -> ``_init_new``: a dozen attribute
# writes, a shape canonicalisation, a stride loop, an allocator lookup and an APIC probe. For a
# CUDA allocation outside any capture every one of those is fixed by ``(dtype, rank, context)``
# except the shape, the strides, the size and the pointer, so ``empty`` stamps a cached template of
# a real array's attributes and patches those four -- the same trick ``array.split`` uses for views.
# ``__del__`` reads only ``device``, ``ptr``, ``deleter``, ``_allocator`` and ``capacity``, all of
# which are set exactly as ``_init_new`` sets them.

_new_array = object.__new__
_alloc_templates: dict[tuple[Any, ...], tuple[dict[str, Any], Any, int, bool]] = {}
_type_size = _types.type_size_in_bytes
_PY_DTYPES = {int: _types.int32, float: _types.float32, bool: _types.bool}
_WP_CURRENT_STREAM = ctypes.c_void_p(0xFFFFFFFFFFFFFFFF)


# Warp's two CUDA allocators call natives that make the context current themselves
# (``wp_alloc_device_async`` / ``_default`` and the matching frees open a native ``ContextGuard``),
# so the Python-side push and pop Warp wraps around each allocation and each free are two redundant
# native calls. An array stamped from these allocators allocates without them and carries a
# subclass of its allocator that says so to ``array.__del__``; every other Warp reader of
# ``_allocator`` tests it with ``isinstance``, which the subclass satisfies.
_unguarded_classes: dict[type[Any], type[Any]] = {}


def _unguarded(allocator: Any) -> Any | None:
    """``allocator`` as an instance of a subclass freeing without the Python context guard."""
    cls = type(allocator)
    if cls not in (_ctx.CudaMempoolAllocator, _ctx.CudaDefaultAllocator):
        return None
    sub = _unguarded_classes.get(cls)
    if sub is None:
        sub = _unguarded_classes[cls] = type(
            f"_Unguarded{cls.__name__}", (cls,), {"deallocate_requires_context_guard": False}
        )
    shim = object.__new__(sub)
    shim.__dict__.update(allocator.__dict__)
    return shim


def _template(dtype: Any, ndim: int, device: Any) -> tuple[dict[str, Any], Any, int, bool] | None:
    key = (dtype, ndim, device.context)
    template = _alloc_templates.get(key)
    # A template records the allocator it was made with; ``wp.set_mempool_enabled`` or a custom
    # allocator installed since then must be honoured, so a changed allocator rebuilds it.
    if template is None or template[1] is not device.get_allocator():
        probe = wp.empty((1,) * ndim, dtype=dtype, device=device)
        state = dict(probe.__dict__)
        if state.get("_apic_capture_origin") is not None or state["_requires_grad"]:
            return None
        for name in ("ptr", "shape", "strides", "size", "capacity"):
            state.pop(name)
        allocator = probe._allocator
        shim = _unguarded(allocator)
        if shim is not None:
            state["_allocator"] = shim
            state["deleter"] = shim.deallocate
        template = (state, allocator, _type_size(dtype), shim is None)
        _alloc_templates[key] = template
        del probe
    return template


@overload
def empty(
    shape: Any, dtype: type[DType], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]: ...
@overload
def empty(
    shape: Any = 0, *, device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[wp.float32, Any]: ...
def empty(
    shape: Any = 0, dtype: Any = float, device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[Any]:
    """``wp.empty``, stamped from a cached template for a CUDA allocation outside capture."""
    if type(device) is not _Device:
        # ``_ctx.runtime`` is ``None`` until ``wp.init()``; Warp annotates it as always set.
        device = None if _ctx.runtime is None else _ctx.runtime.get_device(device)  # pyright: ignore[reportUnnecessaryComparison]
    runtime = _ctx.runtime
    if kwargs and device is not None and device.is_cuda and kwargs.keys() == {"pinned"}:
        kwargs = {}  # ``pinned`` is a host-memory property; Warp ignores it on a CUDA device
    if (
        kwargs
        or device is None
        or not device.is_cuda
        or runtime.captures
        or runtime._apic_capture is not None
    ):
        return wp.empty(shape, dtype=dtype, device=device, **kwargs)
    dtype = _PY_DTYPES.get(dtype, dtype)
    if type(shape) is int:
        if shape < 0:
            return wp.empty(shape, dtype=dtype, device=device)
        shape = (shape,)
    else:
        shape = tuple(shape)
        for extent in shape:
            if type(extent) is not int or extent < 0:
                return wp.empty(shape, dtype=dtype, device=device)
        if not 0 < len(shape) <= 4:
            return wp.empty(shape, dtype=dtype, device=device)
    ndim = len(shape)
    template = _template(dtype, ndim, device)
    if template is None:
        return wp.empty(shape, dtype=dtype, device=device)
    state, allocator, itemsize, guarded = template
    if ndim == 1:
        size = shape[0]
        strides = (itemsize,)
    else:
        strides = [itemsize] * ndim
        size = shape[-1]
        for i in range(ndim - 1, 0, -1):
            strides[i - 1] = strides[i] * shape[i]
            size *= shape[i - 1]
        strides = tuple(strides)
    capacity = size * itemsize
    if capacity > 0:
        if guarded:
            guard = device.context_guard
            guard.__enter__()
            try:
                ptr = allocator.allocate(capacity)
            finally:
                guard.__exit__(None, None, None)
        else:
            ptr = allocator.allocate(capacity)
    else:
        ptr = None
    arr = _new_array(_ARRAY)
    d = arr.__dict__
    d.update(state)
    d["ptr"] = ptr
    d["shape"] = shape
    d["strides"] = strides
    d["size"] = size
    d["capacity"] = capacity
    return arr


@overload
def zeros(
    shape: Any, dtype: type[DType], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]: ...
@overload
def zeros(
    shape: Any = 0, *, device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[wp.float32, Any]: ...
def zeros(
    shape: Any = 0, dtype: Any = float, device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[Any]:
    """``wp.zeros`` through [`empty`][triwarp._launch.empty] plus one device memset."""
    if kwargs:
        return wp.zeros(shape, dtype=dtype, device=device, **kwargs)
    arr = empty(shape, dtype=dtype, device=device)
    arr_device: Any = arr.device
    if arr.capacity and arr_device.is_cuda and type(arr) is _ARRAY and not _ctx.runtime.captures:
        _ctx.runtime.core.wp_memset_device(
            arr_device.context, arr.ptr, 0, arr.capacity, _WP_CURRENT_STREAM
        )
    elif arr.capacity:
        arr.zero_()
    return arr


def _is_zero(value: Any) -> bool:
    """Whether ``value`` is a plain scalar zero whose bytes are all zero (``-0.0`` is not)."""
    return (
        type(value) in _SCALAR_PY
        and value == 0
        and not (type(value) is float and str(value)[0] == "-")
    )


@overload
def full(
    shape: Any, value: Any, dtype: type[DType], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]: ...
@overload
def full(
    shape: Any = None,
    value: Any = 0,
    dtype: None = None,
    device: wp.DeviceLike = None,
    **kwargs: Any,
) -> wp.array[Any]: ...
def full(
    shape: Any = None,
    value: Any = 0,
    dtype: Any = None,
    device: wp.DeviceLike = None,
    **kwargs: Any,
) -> wp.array[Any]:
    """
    Return ``wp.full`` through [`empty`][triwarp._launch.empty].

    ``wp.full`` itself when ``dtype`` is left to be inferred from ``value``.
    """
    if kwargs or dtype is None or shape is None:
        return wp.full(shape, value, dtype=dtype, device=device, **kwargs)
    if _is_zero(value):
        return zeros(shape, dtype=dtype, device=device)
    arr = empty(shape, dtype=dtype, device=device)
    arr.fill_(value)
    return arr


@overload
def ones(
    shape: Any, dtype: type[DType], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]: ...
@overload
def ones(
    shape: Any = None, *, device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[wp.float32, Any]: ...
def ones(
    shape: Any = None, dtype: Any = float, device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[Any]:
    """``wp.ones`` through [`full`][triwarp._launch.full]."""
    if kwargs or shape is None:
        return wp.ones(shape, dtype=dtype, device=device, **kwargs)
    return full(shape, 1, dtype=_PY_DTYPES.get(dtype, dtype), device=device)


def empty_packed(
    dtype: type[DType], device: wp.DeviceLike
) -> tuple[wp.array[DType, Any], wp.array[wp.int32, Any]]:
    """Return the packed ``(values, offsets)`` pair of no items: no values, offsets ``[0]``."""
    return empty(0, dtype=dtype, device=device), zeros(1, dtype=wp.int32, device=device)


def _like_fast(src: Any, device: Any, kwargs: dict[str, Any]) -> bool:
    """Whether a ``*_like`` / ``clone`` of ``src`` can take the stamped path."""
    return (
        not kwargs
        and type(src) is _ARRAY
        and not src.requires_grad
        and (device is None or device is src.device)
    )


def empty_like(
    src: wp.array[DType, Any], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]:
    """``wp.empty_like`` through [`empty`][triwarp._launch.empty]."""
    if not _like_fast(src, device, kwargs):
        return wp.empty_like(src, device=device, **kwargs)
    return empty(src.shape, dtype=src.dtype, device=src.device)


def zeros_like(
    src: wp.array[DType, Any], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]:
    """``wp.zeros_like`` through [`zeros`][triwarp._launch.zeros]."""
    if not _like_fast(src, device, kwargs):
        return wp.zeros_like(src, device=device, **kwargs)
    return zeros(src.shape, dtype=src.dtype, device=src.device)


def clone(src: ArrayT, device: wp.DeviceLike = None, **kwargs: Any) -> ArrayT:
    """``wp.clone`` of a contiguous array through [`empty`][triwarp._launch.empty]."""
    if not _like_fast(src, device, kwargs) or not src.is_contiguous:
        return cast("ArrayT", wp.clone(src, device=device, **kwargs))
    dst = empty(src.shape, dtype=src.dtype, device=src.device)
    copy(dst, src)
    return cast("ArrayT", dst)  # ``src``'s own dtype and shape


# -- host upload ----------------------------------------------------------------------------------
#
# ``wp.array(data, dtype=..., device=cuda)`` converts ``data`` with ``np.asarray(data, dtype=<the
# dtype's NumPy scalar type>)``, reconciles the shape with a vector or matrix element type,
# allocates through ``_init_new``, wraps the host buffer in a CPU array and copies it with
# ``wp.copy``. For a C-contiguous result whose trailing axes are exactly the element type's shape,
# that is a stamped ``empty`` plus one native host-to-device copy; ``array`` does that and hands
# every other case -- Warp's reshape leniency, native / struct element types, a CPU device, any
# keyword -- to ``wp.array``.

_upload_dtypes: dict[Any, tuple[Any, tuple[int, ...]] | None] = {}


def _upload_layout(dtype: Any) -> tuple[Any, tuple[int, ...]] | None:
    """``(NumPy scalar dtype, element shape)`` for uploading into ``dtype``, or ``None``."""
    if dtype not in _upload_dtypes:
        layout = None
        scalar = getattr(dtype, "_wp_scalar_type_", dtype)
        element_shape = tuple(getattr(dtype, "_shape_", ()))
        np_dtype = _types.warp_type_to_np_dtype.get(scalar)
        if np_dtype is not None and scalar not in _HALF[1:] and not _types.is_native_type(dtype):
            layout = (np.dtype(np_dtype), element_shape)
        _upload_dtypes[dtype] = layout
    return _upload_dtypes[dtype]


@overload
def array(
    data: Any, dtype: type[DType], device: Any = None, **kwargs: Any
) -> wp.array[DType, Any]: ...
@overload
def array(
    data: Any, dtype: _codegen.Struct, device: Any = None, **kwargs: Any
) -> wp.array[_codegen.StructInstance, Any]: ...
def array(
    data: Any, dtype: type[DType] | _codegen.Struct, device: Any = None, **kwargs: Any
) -> wp.array[DType, Any] | wp.array[_codegen.StructInstance, Any]:
    """``wp.array(data, dtype=dtype, device=device)``, uploaded directly for the common case."""
    if type(device) is not _Device:
        # ``_ctx.runtime`` is ``None`` until ``wp.init()``; Warp annotates it as always set.
        device = None if _ctx.runtime is None else _ctx.runtime.get_device(device)  # pyright: ignore[reportUnnecessaryComparison]
    layout = None if kwargs or device is None or not device.is_cuda else _upload_layout(dtype)
    if (
        layout is not None
        and len(layout[1]) > 1
        and type(data) is list
        and data
        and isinstance(data[0], ctypes.Array)
    ):
        # A list of Warp matrices: ``np.asarray`` walks each one row by row, which costs more than
        # everything this path saves (a list of vectors converts quickly and stays).
        layout = None
    if layout is not None:
        np_dtype, element_shape = layout
        try:
            host = np.asarray(data, dtype=np_dtype)
        except (TypeError, ValueError):
            host = None
        # A 0-d result is a scalar, which ``wp.array`` rejects; it keeps that error.
        if host is not None and host.ndim and host.flags.c_contiguous:
            k = len(element_shape)
            shape = host.shape
            if k and (host.ndim > k and shape[-k:] == element_shape and shape[-1] != 1):
                shape = shape[:-k]
            elif k:
                shape = None
            if shape is not None:
                # A struct dtype has no upload layout, so this branch only sees a plain dtype.
                out = empty(shape, dtype=cast("type[DType]", dtype), device=device)
                if type(out) is _ARRAY and out.capacity:
                    if out.capacity != host.nbytes:
                        return wp.array(data, dtype=dtype, device=device)
                    runtime = _ctx.runtime
                    target: Any = device  # a CUDA ``Device``: it has a context and a stream
                    if not runtime.core.wp_memcpy_h2d(
                        target.context,
                        out.ptr,
                        host.ctypes.data,
                        host.nbytes,
                        target._stream.cuda_stream,
                    ):
                        raise RuntimeError(f"Warp copy error: {runtime.get_error_string()}")
                    return out
                if type(out) is _ARRAY:
                    return out
    return wp.array(data, dtype=dtype, device=device, **kwargs)
