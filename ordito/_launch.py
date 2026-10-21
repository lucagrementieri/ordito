"""
Private kernel launcher: ``wp.launch`` / ``wp.launch_tiled`` with the per-call resolution cached.

``wp.launch`` re-derives, on every call, everything that is fixed for a kernel on a device: it
re-resolves the device, re-checks that the module is loaded and current, re-reads the kernel's
hooks, and marshals every argument through ``pack_arg``'s general type dispatch. For the short
launches that make up most of this package's host time that resolution *is* the launch, so
[`launch`][ordito._launch.launch] resolves it once per ``(kernel, CUDA context, block_dim)`` and
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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, cast, overload

import numpy as np
import warp as wp
import warp._src.codegen as _codegen
import warp._src.context as _ctx
import warp._src.types as _types

from ordito.config import is_deterministic

if TYPE_CHECKING:
    import ordito.typing as odt

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

    __slots__ = (
        "exec_",
        "hashers",
        "hooks",
        "kernel_dim",
        "key",
        "module",
        "n_args",
        "packers",
        "pool",
        "tid_limit",
    )

    def __init__(
        self,
        kernel: Any,
        exec_: Any,
        hashers: dict[Any, Any],
        hooks: Any,
        context: Any,
        block_dim: int,
    ) -> None:
        self.exec_ = exec_
        self.hashers = hashers
        self.hooks = hooks
        self.module = kernel.module
        self.key = (context, block_dim)
        self.n_args = len(kernel.adj.args)
        self.kernel_dim = kernel.adj.kernel_dim
        self.tid_limit = kernel.adj.tid_extent_limit_candidate
        self.packers = tuple(_packer(kernel, arg) for arg in kernel.adj.args)
        block = _Block.build(kernel, self.kernel_dim)
        if block is not None:
            block.compile(kernel, hooks, context, block_dim)
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
_KIND_ARRAY, _KIND_SCALAR, _KIND_VALUE, _KIND_STRUCT = 0, 1, 2, 3


class _Block:
    """A kernel's parameters as one buffer, filled per launch by one ``struct.pack_into``."""

    __slots__ = ("buffer", "kinds", "pack_into", "params", "run")

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
            elif isinstance(arg_type, _codegen.Struct) and ctypes.alignment(arg_type.ctype) <= 8:
                # A ``@wp.struct`` is its ctypes image, the bytes Warp's own launch hands over.
                size = ctypes.sizeof(arg_type.ctype)
                fmt.append(f"{size}s")
                kinds.append((_KIND_STRUCT, 0, None, arg_type))
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

    def compile(self, kernel: Any, hooks: Any, context: Any, block_dim: int) -> None:
        """
        Generate ``run``: pack the arguments and launch, specialised to this kernel's parameters.

        The generated function is straight-line code -- one type test per argument, the values
        handed to ``pack_into`` as locals -- closing over everything fixed for the entry (the
        native launch, context, function handle, shared memory and block size), so a launch is one
        Python call, one ``pack_into`` and the native call. It returns ``False`` without launching
        when an argument has no packed form: not exactly the parameter's array class and rank, a
        gradient-carrying array, a dtype or value type that is not Warp-equal to the parameter's,
        an instance of another struct, an array on another device under a strict access mode, or
        a value outside its C type. A struct's own array members are not checked, as in Warp's
        launch.
        """
        kernel_dim = kernel.adj.kernel_dim
        env: dict[str, Any] = {
            "_ARRAY": _ARRAY,
            "_SCALAR_PY": _SCALAR_PY,
            "_accept": _accept,
            "_error": struct.error,
            "pack_into": self.pack_into,
            "buffer": self.buffer,
            "params": self.params,
            "launch_native": _ctx.runtime.core.wp_cuda_launch_kernel,
            "raise_error": _ctx._raise_cuda_launch_error,
            "kernel": kernel,
            "hooks": hooks,
            "context": context,
            "forward": hooks.forward,
            "smem": hooks.forward_smem_bytes,
        }
        extents = [f"e{d}" for d in range(kernel_dim)]
        body = [
            "def run(extent, inputs, outputs, device, strict, max_blocks, stream):",
            "    shape, size, cm = extent",
            f"    {', '.join(extents)}, = shape",
            "    args = (*inputs, *outputs) if outputs else inputs",
        ]
        values = [*extents, "size", "cm"]
        for j, (kind, ndim, accepted, target) in enumerate(self.kinds):
            env[f"A{j}"] = accepted
            env[f"D{j}"] = target
            body.append(f"    v = args[{j}]")
            if kind == _KIND_ARRAY:
                shape_names = [f"h{j}_{d}" for d in range(ndim)]
                stride_names = [f"t{j}_{d}" for d in range(ndim)]
                zeros = " = ".join([f"p{j}", f"n{j}", *shape_names, *stride_names])
                body += [
                    "    if v is None:",
                    f"        {zeros} = 0",
                    "    else:",
                    f"        if type(v) is not _ARRAY or v.ndim != {ndim} or v._grad is not None:",
                    "            return False",
                    "        d = v.dtype",
                    f"        if d not in A{j} and not _accept(A{j}, d, D{j}):",
                    "            return False",
                    "        if strict and v.device is not device:",
                    "            return False",
                    f"        p{j} = v.ptr or 0",
                    f"        {', '.join(shape_names)}, = v.shape",
                    f"        {', '.join(stride_names)}, = v.strides",
                    f"        n{j} = {ndim}",
                ]
                pad = ["0"] * (4 - ndim)
                values += [f"p{j}", "0", *shape_names, *pad, *stride_names, *pad, f"n{j}", "0"]
            elif kind == _KIND_SCALAR:
                body += [
                    "    t = type(v)",
                    "    if t is not int and t is not float and t is not bool:",
                    f"        if t is not D{j}:",
                    "            return False",
                    "        v = v.value",
                    f"    x{j} = v",
                ]
                values.append(f"x{j}")
            elif kind == _KIND_VALUE:
                body += [
                    "    t = type(v)",
                    f"    if t not in A{j} and not _accept(A{j}, t, D{j}):",
                    "        return False",
                    f"    x{j} = bytes(v)",
                ]
                values.append(f"x{j}")
        body += [
            "    try:",
            f"        pack_into(buffer, 0, {', '.join(values)})",
            "    except _error:",
            "        return False",
            "    if launch_native(context, forward, size, max_blocks, "
            f"{block_dim}, 1, 1, smem, params, stream, None):",
            "        raise_error(kernel, device, hooks, False)",
            "    return True",
        ]
        exec("\n".join(body), env)
        self.run = env["run"]


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
    cache = kernel.__dict__.get("_od_launch")
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
    cache = kernel.__dict__.setdefault("_od_launch", {})
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
    cache[key] = _Entry(kernel, exec_, module.hashers, hooks, device.context, block_dim)


def launch(
    kernel: odt.Kernel,
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
    kernel_any: Any = kernel
    # The hot path: the same kernel launched again on the device and block size of its last cached
    # launch, outside any capture or tape -- most launches in a wrapper's loop. Every condition the
    # general path tests below is tested here, cheapest first; any miss falls through to it.
    hot = kernel_any.__dict__.get("_od_hot")
    if (
        hot is not None
        and hot[0] is device
        and hot[1] == block_dim
        and stream is None
        and not adjoint
        and not record_cmd
        and not adj_inputs
        and not adj_outputs
    ):
        entry = hot[2]
        runtime = _ctx.runtime
        module = entry.module
        config = wp.config
        if type(dim) is int:
            extent = (
                ((dim,), dim, 1) if 0 < dim <= entry.tid_limit and entry.kernel_dim == 1 else None
            )
        else:
            extent = _extent(dim, entry)
        if (
            extent is not None
            and len(inputs) + len(outputs) == entry.n_args
            and runtime.tape is None
            and runtime._apic_capture is None
            and not runtime.captures
            and not config.verify_cuda
            and not config.print_launches
            and module.hashers is entry.hashers
            and module.execs.get(entry.key) is entry.exec_
            and not module.has_unresolved_static_expressions
        ):
            pool = entry.pool
            if pool:
                try:
                    block = pool.pop()
                except IndexError:
                    block = None
                if block is not None:
                    try:
                        if block.run(
                            extent,
                            inputs,
                            outputs,
                            device,
                            config.launch_array_access_mode is not _RELAXED,
                            max_blocks,
                            device._stream.cuda_stream,
                        ):
                            return None
                    finally:
                        pool.append(block)
    runtime = _ctx.runtime
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
    kernel_any.__dict__["_od_hot"] = (device, block_dim, entry)

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
                if block.run(
                    extent, inputs, outputs, device, strict, max_blocks, current.cuda_stream
                ):
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
    ints within the ``wp.tid()`` extent limit.
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
    limit = entry.tid_limit
    for extent in dim:
        # Warp bounds every ``wp.tid()`` axis, not only the leading one; an oversized extent
        # goes to ``wp.launch``, whose exact check decides whether it raises.
        if type(extent) is not int or extent <= 0 or extent > limit:
            return None
        size *= extent
    if n == kernel_dim:
        return dim, size, 1
    if n < kernel_dim:
        return dim + (1,) * (kernel_dim - n), size, 1
    coord_mult = 1
    for extent in dim[kernel_dim:]:
        coord_mult *= extent
    return dim[:kernel_dim], size, coord_mult


def _fallback(
    kernel: odt.Kernel,
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
    kernel: odt.Kernel,
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


# The widest block a debug build can launch (see ``launch_tiled``).
DEBUG_MAX_BLOCK_DIM = 256


def launch_tiled(
    kernel: odt.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence[Any] = (),
    outputs: Sequence[Any] = (),
    device: wp.DeviceLike = None,
    block_dim: int = 256,
    **kwargs: Any,
) -> Any:
    """``wp.launch_tiled`` through [`launch`][ordito._launch.launch]."""
    if type(device) is not _Device:
        device = wp.get_device(device)
    if block_dim > DEBUG_MAX_BLOCK_DIM and wp.config.mode == "debug":
        # A ``--device-debug`` build runs at -O0 and spends up to the 255-register cap per thread,
        # so a 512- or 1 024-lane block exceeds the 64 K-register file and the launch fails with
        # CUDA error 701 (the one-block ear-clip, RDP and CG kernels: 39-56 registers released,
        # 163-255 in debug). 256 lanes always fit 255 registers. Every kernel launched wider
        # strides by ``wp.block_dim()`` (it already runs at width 1 on the CPU device), so the
        # narrower block is the same walk; only float reduction order moves.
        block_dim = DEBUG_MAX_BLOCK_DIM
    if device.is_cpu or kwargs:
        return wp.launch_tiled(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device,
                               block_dim=block_dim, **kwargs)  # fmt: skip
    dim = (dim,) if isinstance(dim, int) else tuple(dim)
    return launch(kernel, (*dim, block_dim), inputs, outputs, device=device, block_dim=block_dim)


_map_kernels: dict[tuple[Any, ...], tuple[odt.Kernel, tuple[set[Any], ...]]] = {}
_is_array = _types.is_array


def map(  # noqa: A001 - mirrors ``wp.map``
    func: Any, *inputs: Any, out: Any = None, block_dim: int = 256, device: wp.DeviceLike = None
) -> Any:
    """
    ``wp.map`` with its kernel lookup cached, launched through [`launch`][ordito._launch.launch].

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
# API-capture bookkeeping. For the case every ordito call site is in -- contiguous CUDA arrays on
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
    """
    ``wp.utils.array_scan``, calling the native scan directly for contiguous CUDA arrays.

    While [`ordito.config.is_deterministic`][ordito.config.is_deterministic] holds (read at the
    call), an inclusive ``float32`` scan on CUDA takes the fixed-order tile scan of
    ``kernels/array.py`` (``float_scan_chunks``) instead of CUB's, whose decoupled look-back sums
    in arrival order and so moves the last bits of the result from call to call. It costs two
    launches and a small allocation over CUB's one call once the input spans more than one
    ``FLOAT_SCAN_CHUNK`` (and is cheaper below that). Every other scan is exact (integers) or
    already sequential (the CPU device), and ignores the setting.
    """
    runtime = _ctx.runtime
    size = in_array.size
    dtype = in_array.dtype
    if (
        size
        and out_array.size == size
        and out_array.dtype is dtype
        and _plain_cuda(runtime, in_array, out_array)
    ):
        if inclusive and dtype is _types.float32 and is_deterministic():
            _float_scan_fixed_order(in_array, out_array, size)
            return
        native = _scan_native(dtype)
        if native is not None:
            function, components, itemsize = native
            # The CUDA scan returns ``False`` when it fails (an out-of-memory scratch allocation
            # used to launch the scan on a null buffer and leave ``out_array`` partly unwritten);
            # ``wp.utils.array_scan`` raises on it, and so does this path.
            if not function(
                in_array.ptr, out_array.ptr, size, itemsize, itemsize, components, inclusive
            ):
                raise RuntimeError(runtime.get_error_string())
            return
    wp.utils.array_scan(in_array, out_array, inclusive=inclusive)


_kernel_array: Any = None


def block_partials(n_blocks: int, width: int, dtype: Any, device: Any) -> tuple[Any, int]:
    """
    Allocate the zeroed buffer of a block commit; return it and the ``stride`` its kernel takes.

    ``kernels/reduce.store_block_sum`` / ``store_block_total`` store each of the ``n_blocks``
    blocks' ``width`` sums in a row of it, and the last block to finish adds the rows in a fixed
    order into the first ``width`` slots, where the caller reads them: ``width`` result slots, a
    ticket counter and ``n_blocks`` rows. While
    [`ordito.config.is_deterministic`][ordito.config.is_deterministic] is off (read at the call)
    the buffer is the ``width`` result slots alone, which the blocks ``atomic_add`` into in
    arrival order; a caller that reads more than the first ``width`` slots tells the two apart by
    the buffer's size.
    """
    if not is_deterministic():
        return zeros(width, dtype=dtype, device=device), width
    return zeros((n_blocks + 1) * width + 1, dtype=dtype, device=device), width


_kernel_reduce: Any = None
_off_pass: Any = None
# Above this many terms per slot no headroom keeps two words' sums inside ``int64``.
_MAX_GROUP_SUM_HEADROOM = 31


class GroupSums(NamedTuple):
    """
    The buffers of one per-group sum (``kernels/reduce.py``'s exact group sums), from `group_sums`.

    ``out`` is the ``(n_groups,)`` result in the caller's dtype: with
    [`ordito.config.is_deterministic`][ordito.config.is_deterministic] off it is the zeroed float
    accumulator the scatter adds into in arrival order, and on it is written by
    `finish_group_sums`. ``launch_pass`` is the ``kernels/reduce.GroupSumPass`` the scattering
    kernel takes, launched once per entry of ``phases``: ``(0,)`` when off, ``(1,)`` when a bound
    makes the field pass unnecessary, else ``(0, 1)`` (pass 0 raises the group fields, pass 1 adds
    the words). ``words`` is the fields-and-words buffer, ``None`` when off.
    """

    out: Any
    launch_pass: Any
    phases: tuple[int, ...]
    words: Any

    def add_only(self) -> GroupSums:
        """Return these sums with pass 1 alone: a loop's rounds once its fields bound every term."""
        return self._replace(phases=self.phases[-1:])


def group_sums(
    n_groups: int,
    dtype: Any,
    max_terms: int,
    device: Any,
    *,
    bound: float | None = None,
    shared_scale: bool = False,
) -> GroupSums:
    """
    Allocate a per-group sum of ``dtype`` over ``n_groups`` groups, per the deterministic setting.

    Off: the zeroed ``(n_groups,)`` accumulator alone, the arrival-order scatter's own allocation,
    and a pass whose empty buffer makes ``kernels/scatter.scatter_add`` an ``atomic_add``. On: the
    result and one zeroed ``int64`` buffer holding the group fields and the fixed-point words, one
    word per component for a ``float32`` component and two for a ``float64`` one. ``max_terms``
    bounds the terms any one component of a group receives (the total term count is always safe)
    and sets the headroom: the smaller, the finer the words.

    ``bound`` promises that no term's magnitude exceeds it: one field shared by every group is set
    from it and pass 0 is skipped (a term that breaks the promise makes every group NaN).
    ``shared_scale`` keeps pass 0 but finds one field for every group: no gathered field read in
    either pass, at the cost of setting every group's precision from the largest term overall.
    """
    global _kernel_reduce, _off_pass
    kernel_reduce = _kernel_reduce
    if kernel_reduce is None:
        # Imported on first use, so that ``_launch`` stays kernel-free at import.
        from ordito.kernels import reduce as kernel_reduce

        _kernel_reduce = kernel_reduce
    if not is_deterministic():
        if _off_pass is None:
            _off_pass = kernel_reduce.GroupSumPass()
        return GroupSums(zeros(n_groups, dtype=dtype, device=device), _off_pass, (0,), None)
    two_words, size = kernel_reduce.GROUP_SUM_LAYOUTS[dtype]
    n_slots = n_groups * size
    headroom = max(max_terms - 1, 0).bit_length()
    if headroom > _MAX_GROUP_SUM_HEADROOM:
        raise ValueError(f"a group sum takes at most 2**31 terms per slot, not {max_terms}.")
    base_field = 0
    if bound is not None:
        (bits,) = struct.unpack("<q", struct.pack("<d", abs(float(bound))))
        base_field = (bits >> 52) & 2047
    n_fields = 1 if shared_scale or bound is not None else n_groups
    words = zeros(n_fields + (2 if two_words else 1) * n_slots, dtype=_types.int64, device=device)
    launch_pass = kernel_reduce.GroupSumPass()
    launch_pass.words = words
    launch_pass.n_fields = n_fields
    launch_pass.n_slots = n_slots
    launch_pass.base_field = base_field
    launch_pass.headroom = headroom
    phases = (1,) if bound is not None else (0, 1)
    return GroupSums(empty(n_groups, dtype=dtype, device=device), launch_pass, phases, words)


def launch_group_sums(
    kernel: Any,
    dim: Any,
    inputs: Sequence[Any],
    sums: GroupSums,
    outputs: Sequence[Any] = (),
    device: Any = None,
) -> None:
    """
    Launch a scattering kernel once per pass of ``sums``.

    The kernel takes ``inputs``, then the ``GroupSumPass``, then ``sums.out``, then ``outputs``
    (``kernels/scatter.scatter_add``'s caller convention), so with the setting off this is the one
    launch the arrival-order scatter issued. The pass is set before each launch, which copies it
    (a graph capture records each launch's copy).
    """
    launch_pass = sums.launch_pass
    for phase in sums.phases:
        if sums.words is not None:
            launch_pass.phase = phase
        launch(
            kernel,
            dim=dim,
            inputs=[*inputs, launch_pass],
            outputs=[sums.out, *outputs],
            device=device,
        )


def finish_group_sums(sums: GroupSums, *, clear: bool = False) -> Any:
    """
    Write the sums into ``sums.out`` (setting on; nothing to do off) and return it.

    ``clear`` leaves the words zeroed for a loop's next round over the same ``sums``.
    """
    if sums.words is not None:
        out = sums.out
        launch(
            _kernel_reduce.FINISH_GROUP_SUMS[out.dtype],
            dim=out.shape[0],
            inputs=[sums.launch_pass, 1 if clear else 0],
            outputs=[out],
            device=out.device,
        )
    return sums.out


def _float_scan_fixed_order(in_array: Any, out_array: Any, size: int) -> None:
    """Scan a contiguous CUDA ``float32`` array inclusively, the same bits on every call."""
    global _kernel_array
    kernel_array = _kernel_array
    if kernel_array is None:
        # Imported on first use, so that ``_launch`` stays kernel-free at import (``ordito`` is
        # lazy).
        from ordito.kernels import array as kernel_array

        _kernel_array = kernel_array
    chunk = int(kernel_array.FLOAT_SCAN_CHUNK)
    n_chunks = (size + chunk - 1) // chunk
    device = in_array.device
    block_dim = kernel_array.FLOAT_SCAN_BLOCK_DIM
    if n_chunks == 1:
        launch_tiled(kernel_array.float_scan_single_chunk, dim=1, inputs=[in_array, out_array],
                     block_dim=block_dim, device=device)  # fmt: skip
        return
    totals = empty(n_chunks, dtype=_types.float32, device=device)
    launch_tiled(kernel_array.float_scan_chunk_totals, dim=n_chunks, inputs=[in_array, totals],
                 block_dim=block_dim, device=device)  # fmt: skip
    launch_tiled(kernel_array.float_scan_chunks, dim=n_chunks, inputs=[in_array, totals, out_array],
                 block_dim=block_dim, device=device)  # fmt: skip


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


def copy_to_host(dest: Any, src: Any, src_offset: int, count: int) -> None:
    """
    ``wp.copy(dest, src, src_offset=src_offset, count=count)`` into a pageable host buffer.

    The one native device-to-host copy for a contiguous CUDA ``src`` and a contiguous CPU ``dest``
    of the same element size, outside any capture, on ``src``'s current stream -- what
    ``wp.copy`` issues there, without its validation. A pageable destination makes the copy return
    only once it has landed, which is what a readback relies on. Anything else, including every
    input Warp would reject, goes to ``wp.copy``.
    """
    runtime = _ctx.runtime
    if (
        runtime is not None  # pyright: ignore[reportUnnecessaryComparison]
        and runtime._apic_capture is None
        and not runtime.captures
        and type(src) is _ARRAY
        and type(dest) is _ARRAY
        and src.is_contiguous
        and dest.is_contiguous
        and type(src_offset) is int
        and type(count) is int
        and 0 <= src_offset
        and 0 < count
        and src_offset + count <= src.size
        and count <= dest.size
        and not dest.pinned
    ):
        # Both are allocated arrays here (a size of at least ``count > 0``), so neither has a
        # ``None`` device or pointer, which Warp's annotations allow.
        device: Any = src.device
        dest_device: Any = dest.device
        src_ptr: Any = src.ptr
        itemsize = _type_size(src.dtype)
        if device.is_cuda and dest_device.is_cpu and _type_size(dest.dtype) == itemsize:
            if not runtime.core.wp_memcpy_d2h(
                device.context,
                dest.ptr,
                src_ptr + src_offset * itemsize,
                count * itemsize,
                device._stream.cuda_stream,
            ):
                raise RuntimeError(f"Warp copy error: {runtime.get_error_string()}")
            return
    wp.copy(dest, src, src_offset=src_offset, count=count)


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


def pattern(values: Sequence[Any], dtype: Any, device: Any) -> Any:
    """
    ``array(values, dtype, device)`` for a few host scalars: one native fill, no upload.

    On a CUDA device outside capture the scalars are written by one ``wp_memtile_device`` of their
    packed bytes (passed to the fill kernel by value, so nothing is staged from host memory) into a
    stamped [`empty`][ordito._launch.empty] buffer; anything else -- the CPU device, a vector or
    half dtype -- is [`array`][ordito._launch.array].
    """
    if type(device) is not _Device:
        # ``_ctx.runtime`` is ``None`` until ``wp.init()``; Warp annotates it as always set.
        device = None if _ctx.runtime is None else _ctx.runtime.get_device(device)  # pyright: ignore[reportUnnecessaryComparison]
    runtime = _ctx.runtime
    ctor: Any = getattr(dtype, "_type_", None)
    if (
        device is not None
        and device.is_cuda
        and values
        and ctor in _SCALAR_CHARS
        and dtype not in _HALF
        and not issubclass(dtype, ctypes.Array)
        and not runtime.captures
        and runtime._apic_capture is None
    ):
        arr = empty(len(values), dtype=dtype, device=device)
        if type(arr) is _ARRAY:
            packed = (ctor * len(values))(*values)
            runtime.core.wp_memtile_device(
                device.context, arr.ptr, ctypes.byref(packed), ctypes.sizeof(packed), 1
            )
            return arr
    return array(list(values), dtype=dtype, device=device)


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
    """``wp.zeros`` through [`empty`][ordito._launch.empty] plus one device memset."""
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
    Return ``wp.full`` through [`empty`][ordito._launch.empty].

    ``wp.full`` itself when ``dtype`` is left to be inferred from ``value``.
    """
    if kwargs or dtype is None or shape is None:
        return wp.full(shape, value, dtype=dtype, device=device, **kwargs)
    if _is_zero(value):
        return zeros(shape, dtype=dtype, device=device)
    arr = empty(shape, dtype=dtype, device=device)
    fill_(arr, value)
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
    """``wp.ones`` through [`full`][ordito._launch.full]."""
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
    """``wp.empty_like`` through [`empty`][ordito._launch.empty]."""
    if not _like_fast(src, device, kwargs):
        return wp.empty_like(src, device=device, **kwargs)
    return empty(src.shape, dtype=src.dtype, device=src.device)


def zeros_like(
    src: wp.array[DType, Any], device: wp.DeviceLike = None, **kwargs: Any
) -> wp.array[DType, Any]:
    """``wp.zeros_like`` through [`zeros`][ordito._launch.zeros]."""
    if not _like_fast(src, device, kwargs):
        return wp.zeros_like(src, device=device, **kwargs)
    return zeros(src.shape, dtype=src.dtype, device=src.device)


def clone(src: ArrayT, device: wp.DeviceLike = None, **kwargs: Any) -> ArrayT:
    """``wp.clone`` of a contiguous array through [`empty`][ordito._launch.empty]."""
    if not _like_fast(src, device, kwargs) or not src.is_contiguous:
        return cast("ArrayT", wp.clone(src, device=device, **kwargs))
    dst = empty(src.shape, dtype=src.dtype, device=src.device)
    copy(dst, src)
    return cast("ArrayT", dst)  # ``src``'s own dtype and shape


def view(src: ArrayT, start: int, stop: int) -> ArrayT:
    """
    ``src[start:stop]`` of a contiguous 1-D array, stamped without ``wp.array.__getitem__``.

    The view Warp's slicing builds -- same state, ``src`` held as its owner -- at a fraction of the
    cost, for a caller cutting many per-level views out of one buffer. Anything else (another rank,
    a gradient, a bound outside ``[0, src.size]``) goes through ``__getitem__``.
    """
    s: Any = src
    if (
        type(s) is not _ARRAY
        or s.ndim != 1
        or s._grad is not None
        or s._requires_grad
        or not 0 <= start <= stop <= s.shape[0]
        or start == s.shape[0]
    ):
        return cast("ArrayT", s[start:stop])
    strides: Any = s.strides
    itemsize = int(strides[0])
    count = stop - start
    arr = _new_array(_ARRAY)
    arr.__dict__.update(
        _array_interface=None,
        _grad=None,
        _is_read=False,
        _memory_kind=None,
        _ref=s,
        _requires_grad=False,
        _retain_grad=False,
        _storage_base_ptr=None,
        _storage_nbytes=None,
        capacity=count * itemsize,
        ctype=None,
        deleter=None,
        device=s.device,
        dtype=s.dtype,
        is_contiguous=True,
        is_transposed=False,
        ndim=1,
        pinned=s.pinned,
        ptr=None if s.ptr is None else s.ptr + start * itemsize,
        shape=(count,),
        size=count,
        strides=(itemsize,),
    )
    return cast("ArrayT", arr)


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
