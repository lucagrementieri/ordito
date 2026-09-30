"""
Private kernel launcher: ``wp.launch`` / ``wp.launch_tiled`` with the per-call resolution cached.

``wp.launch`` re-derives, on every call, everything that is fixed for a kernel on a device: it
re-resolves the device, re-checks that the module is loaded and current, re-reads the kernel's
hooks, and marshals every argument through ``pack_arg``'s general type dispatch. For the short
launches that make up most of this package's host time that resolution *is* the launch, so
[`launch`][triwarp._launch.launch] resolves it once per ``(kernel, CUDA context, block_dim)`` and
keeps one packer per parameter, then calls the native launch directly.

It is ``wp.launch`` exactly wherever it can be: anything the cache does not cover -- the CPU device,
a generic kernel, an explicit stream, an adjoint or recorded launch, an active tape, graph capture
or APIC capture, deterministic or clustered kernels, ``verify_cuda`` / ``print_launches``, an
argument that is not already the parameter's exact type -- goes to ``wp.launch`` unchanged, and
so does every launch whose module has been modified since the cache entry was made. An argument a
packer does not recognise is marshalled by Warp's own ``pack_arg``, so conversions and error
messages are Warp's.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import warp as wp
import warp._src.codegen as _codegen
import warp._src.context as _ctx
import warp._src.types as _types

_RELAXED = wp.config.LaunchArrayAccessMode.RELAXED
_Device = _ctx.Device
_bounds_classes = _types._launch_bounds_classes
_pack_arg = _ctx.pack_arg
_c_void_p_arrays: dict[int, Any] = {}
_SCALAR_PY = (int, float, bool)
_HALF = (_types.float16, _types.bfloat16)


class _Entry:
    """One kernel's launch state on one CUDA context and block size."""

    __slots__ = ("exec_", "hashers", "hooks", "kernel_dim", "packers", "tid_limit")

    def __init__(self, kernel: wp.Kernel, exec_: Any, hashers: dict, hooks: Any) -> None:
        self.exec_ = exec_
        self.hashers = hashers
        self.hooks = hooks
        self.kernel_dim = kernel.adj.kernel_dim
        self.tid_limit = kernel.adj.scalar_tid_extent_limit_candidate
        self.packers = tuple(_packer(kernel, arg) for arg in kernel.adj.args)


def _packer(kernel: wp.Kernel, arg: Any) -> Any:
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

        def pack_array(value: Any, device: Any) -> Any:
            if type(value) is cls and value.dtype is dtype and value.ndim == ndim:
                return value.__ctype__()
            if value is None:
                return null
            return generic(value, device)

        return pack_array, True

    if isinstance(arg_type, _codegen.Struct):

        def pack_struct(value: Any, device: Any) -> Any:
            return value.__ctype__()

        return pack_struct, False

    if isinstance(arg_type, type) and issubclass(arg_type, ctypes.Array):

        def pack_value(value: Any, device: Any) -> Any:
            if type(value) is arg_type:
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


def _entry(kernel: wp.Kernel, device: Any, block_dim: int) -> _Entry | bool | None:
    """
    Return ``kernel``'s cache entry on ``device``: ``None`` if absent or stale, ``False`` if
    the kernel is not eligible for the cached path.
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


def _remember(kernel: wp.Kernel, device: Any, block_dim: int) -> None:
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
    kernel: wp.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence = (),
    outputs: Sequence = (),
    adj_inputs: Sequence = (),
    adj_outputs: Sequence = (),
    device: wp.DeviceLike = None,
    stream: Any = None,
    adjoint: bool = False,
    record_tape: bool = True,
    record_cmd: bool = False,
    max_blocks: int = 0,
    block_dim: int = 256,
) -> Any:
    """Launch ``kernel`` exactly as ``wp.launch`` would, with its resolution cached."""
    runtime = _ctx.runtime
    if type(device) is not _Device:
        device = None if runtime is None else runtime.get_device(device)
    if (
        device is None
        or not device.is_cuda
        or stream is not None
        or adjoint
        or record_cmd
        or adj_inputs
        or adj_outputs
        or kernel.is_generic
        or runtime.tape is not None
        or runtime.captures
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

    bounds = _bounds(dim, entry)
    if bounds is None:
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
    hooks = entry.hooks
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
        device._stream.cuda_stream,
        None,
    ):
        _ctx._raise_cuda_launch_error(kernel, device, hooks, False)
    return None


def _bounds(dim: Any, entry: _Entry) -> Any:
    """
    ``dim``'s launch bounds, as ``_build_launch_bounds_from_tuple`` builds them, or ``None``
    for anything that is not a tuple of positive ints within the scalar ``wp.tid()`` limit.
    """
    kernel_dim = entry.kernel_dim
    if type(dim) is int:
        if dim <= 0 or dim > entry.tid_limit:
            return None
        if kernel_dim == 1:
            return _bounds_classes[1]((dim,))
        return _bounds_classes[kernel_dim]((dim,) + (1,) * (kernel_dim - 1))
    if type(dim) is not tuple:
        dim = tuple(dim)
    n = len(dim)
    if n == 0 or n > 4:
        return None
    for extent in dim:
        if type(extent) is not int or extent <= 0:
            return None
    if dim[0] > entry.tid_limit:
        return None
    if n == kernel_dim:
        return _bounds_classes[n](dim)
    if n < kernel_dim:
        return _bounds_classes[kernel_dim](dim + (1,) * (kernel_dim - n))
    bounds = _bounds_classes[kernel_dim](dim[:kernel_dim])
    coord_mult = 1
    for extent in dim[kernel_dim:]:
        coord_mult *= extent
    bounds.coord_mult = coord_mult
    bounds.size *= coord_mult
    return bounds


def _fallback(kernel, dim, inputs, outputs, adj_inputs, adj_outputs, device, stream, adjoint,
              record_tape, record_cmd, max_blocks, block_dim):  # fmt: skip
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


def _slow(kernel, dim, inputs, outputs, device, max_blocks, block_dim, entry):  # fmt: skip
    """Launch through ``wp.launch`` and, if the kernel had no cache entry, make one."""
    wp.launch(kernel, dim, inputs=inputs, outputs=outputs, device=device, max_blocks=max_blocks,
              block_dim=block_dim)  # fmt: skip
    if entry is None:
        _remember(kernel, device, block_dim)


def launch_tiled(
    kernel: wp.Kernel,
    dim: int | Sequence[int],
    inputs: Sequence = (),
    outputs: Sequence = (),
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
    dim = (dim,) if type(dim) is int else tuple(dim)
    return launch(kernel, (*dim, block_dim), inputs, outputs, device=device, block_dim=block_dim)


_map_kernels: dict[tuple, tuple[wp.Kernel, tuple]] = {}
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
    key = [id(func)]
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
            if value.requires_grad:
                return wp.map(func, *inputs, out=out, block_dim=block_dim, device=device)
            key.append((type(value), value.dtype, value.ndim, tuple(d == 1 for d in value_shape)))
        else:
            value_type = type(value)
            if value_type not in _SCALAR_PY and not (
                (isinstance(value_type, type) and hasattr(value_type, "_wp_scalar_type_"))
                or value_type in _types.scalar_types
            ):
                return wp.map(func, *inputs, out=out, block_dim=block_dim, device=device)
            key.append(value_type)
    if out is None or shape is None or broadcast:
        return wp.map(func, *inputs, out=out, block_dim=block_dim, device=device)
    outputs = out if isinstance(out, (list, tuple)) else (out,)
    for o in outputs:
        if not _is_array(o) or o.shape != shape:
            return wp.map(func, *inputs, out=out, block_dim=block_dim, device=device)
        key.append((type(o), o.dtype))
    key = tuple(key)
    cached = _map_kernels.get(key)
    if cached is None:
        kernel = wp.map(
            func, *inputs, out=out, return_kernel=True, block_dim=block_dim, device=device
        )
        out_dtypes = tuple(arg.type.dtype for arg in kernel.adj.args[len(inputs) :])
        if len(out_dtypes) != len(outputs):
            return wp.map(func, *inputs, out=out, block_dim=block_dim, device=device)
        cached = _map_kernels[key] = (kernel, out_dtypes)
    kernel, out_dtypes = cached
    for o, dtype in zip(outputs, out_dtypes, strict=True):
        if o.dtype is not dtype:
            return wp.map(func, *inputs, out=out, block_dim=block_dim, device=device)
    launch(kernel, shape, inputs, outputs, device=device, block_dim=block_dim)
    return out
