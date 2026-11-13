"""
Ahead-of-time kernel builds: fill Warp's kernel cache with every kernel ordito launches.

Warp compiles a module of kernels the first time one of them launches, and keeps the result in its
kernel cache, keyed by the module's source and options, the launch's ``block_dim`` and the GPU
architecture. A process starting from an empty cache pays those builds inside its first calls,
seconds to minutes in all. [`precompile`][ordito.precompile.precompile] pays them ahead of time, in
parallel and without running any workload, so a container image or a CI job can ship a warm cache:

```bash
python -m ordito.precompile --arch 86,89 --cache-dir /opt/warp-cache --jobs 16
```

and the processes that use it run with ``WARP_CACHE_PATH=/opt/warp-cache``.

What is built is what [`kernel_variants`][ordito.precompile.kernel_variants] lists: every module of
ordito's kernels at every launch width ordito uses, the kernels ordito creates on first use, and
the Warp modules (sparse, geometry) importing ordito creates. The cache is built against the
installed Warp and ordito and holds only what they would compile, so an upgrade of either needs a
new build. A kernel that only a caller's input creates (a dtype conversion ordito itself never
makes) still compiles on first use.

Targets:

- ``arch`` names CUDA compute capabilities (``86`` for ``sm_86``). Compiling needs no GPU, so a
  build machine without one can serve any architecture. A cache entry serves its exact
  architecture only.
- Without ``arch``, the build is for the GPUs of the machine it runs on, in the format Warp would
  load on them; with none, for the CPU alone.
- ``cpu`` (on by default; ``--no-cpu`` turns it off) adds the CPU device, which a GPU process also
  loads for host-side work. Warp compiles CPU code for the instruction set of the machine it runs
  on and keys the cache entry by it, so the objects serve only machines of that CPU type; on
  another, those modules compile on first use as they would without the cache.

From the command line, ``python -m ordito.precompile --help`` lists the options; ``--arch`` takes a
comma-separated list and may repeat.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import importlib
import logging
import multiprocessing
import os
import pkgutil
import time
from collections.abc import Sequence
from typing import Any

import warp as wp

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PrecompileReport:
    """
    What a [`precompile`][ordito.precompile.precompile] call built, found built and failed on.

    Attributes
    ----------
    targets
        The targets built for: ``"sm_<arch>"`` (``"sm_<arch>.ptx"`` for PTX) per CUDA
        architecture, and ``"cpu"``.
    cache_dir
        The kernel cache directory written, Warp's version included.
    built
        Number of module builds this call compiled.
    cached
        Number of module builds already in the cache, left as they were.
    failed
        ``(module, target, error)`` for each build that raised.

    Examples
    --------
    ```python
    import ordito as od

    report = od.precompile.precompile()
    print(report.targets[-1])  # cpu
    print(report.failed)  # ()
    ```
    """

    targets: tuple[str, ...]
    cache_dir: str
    built: int
    cached: int
    failed: tuple[tuple[str, str, str], ...]


def precompile(
    arch: int | Sequence[int] | None = None,
    *,
    cpu: bool = True,
    cache_dir: str | os.PathLike[str] | None = None,
    jobs: int | None = None,
) -> PrecompileReport:
    """
    Compile every kernel ordito launches into Warp's kernel cache, without running any of them.

    Each worker process imports ordito's kernels and compiles its share of
    [`kernel_variants`][ordito.precompile.kernel_variants] for every target; a build already in
    the cache is left as it is, so a second call compiles nothing. Progress and the targets chosen
    are logged on the ``ordito.precompile`` logger at ``INFO``.

    Parameters
    ----------
    arch
        CUDA compute capabilities to build for (``86`` for ``sm_86``); no GPU is needed.
        ``None`` builds for the GPUs of this machine, or for none when it has none.
    cpu
        Also build for the CPU device, for this machine's instruction set. A process on a GPU
        loads some CPU modules too, so leave it on unless no process using the cache runs on this
        CPU type.
    cache_dir
        Kernel cache directory, the value the processes using it set as ``WARP_CACHE_PATH``
        (Warp adds its version below it). ``None`` is the cache Warp uses by default.
    jobs
        Number of worker processes. ``None`` is one per CPU core.

    Returns
    -------
    PrecompileReport
        The targets, the cache directory and the number of builds compiled, found and failed.

    Raises
    ------
    ValueError
        If ``arch`` is ``None``, ``cpu`` is ``False`` and this machine has no GPU (there would be
        nothing to build), if an ``arch`` is below ``50``, or if ``jobs`` is below ``1``.

    See Also
    --------
    kernel_variants : The modules and launch widths built.

    Examples
    --------
    ```python
    import ordito as od

    report = od.precompile.precompile()
    print(report.built + report.cached > 100)  # True
    ```
    """
    if jobs is not None and jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    targets = _targets(arch, cpu)
    workers = jobs if jobs is not None else os.cpu_count() or 1
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    root = os.fspath(cache_dir) if cache_dir is not None else None
    with concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=context, initializer=_start_worker, initargs=(root,)
    ) as pool:
        resolved_cache, variants, sizes = pool.submit(_worker_variants).result()
        _LOGGER.info(
            "precompile: %d modules, %d launch widths, for %s into %s with %d workers",
            len(variants),
            sum(len(widths) for widths in variants.values()),
            ", ".join(target.label for target in targets),
            resolved_cache,
            workers,
        )
        # Largest modules first: one module is one build, and the largest takes a minute, so it
        # sets the wall time unless it starts at once.
        builds = [
            (name, target, block_dim)
            for name in sorted(variants, key=lambda name: -sizes[name])
            for target in targets
            for block_dim in (variants[name] if target.arch is not None else (1,))
        ]
        futures = {
            pool.submit(_build, name, target.arch, target.use_ptx, block_dim): (name, target)
            for name, target, block_dim in builds
        }
        built = cached = 0
        failed: list[tuple[str, str, str]] = []
        step = max(len(futures) // 10, 1)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            name, target = futures[future]
            compiled, error = future.result()
            if error is not None:
                failed.append((name, target.label, error))
                _LOGGER.warning("precompile: %s for %s failed: %s", name, target.label, error)
            elif compiled:
                built += 1
            else:
                cached += 1
            if done % step == 0 or done == len(futures):
                _LOGGER.info("precompile: %d/%d builds done", done, len(futures))
    _LOGGER.info(
        "precompile: %d built, %d already cached, %d failed in %.0f s",
        built,
        cached,
        len(failed),
        time.perf_counter() - started,
    )
    return PrecompileReport(
        targets=tuple(target.label for target in targets),
        cache_dir=resolved_cache,
        built=built,
        cached=cached,
        failed=tuple(sorted(failed)),
    )


@dataclasses.dataclass(frozen=True)
class _Target:
    """One build target: a CUDA architecture (and whether as PTX), or the CPU (``arch=None``)."""

    arch: int | None
    use_ptx: bool | None = None

    @property
    def label(self) -> str:
        if self.arch is None:
            return "cpu"
        return f"sm_{self.arch}.ptx" if self.use_ptx else f"sm_{self.arch}"


def _targets(arch: int | Sequence[int] | None, cpu: bool) -> list[_Target]:
    """Resolve the ``arch`` / ``cpu`` request into build targets, logging what was inferred."""
    targets: list[_Target] = []
    if arch is not None:
        arches = sorted({int(arch)} if isinstance(arch, int) else {int(value) for value in arch})
        low = [value for value in arches if value < 50]
        if low:
            listed = ", ".join(map(str, low))
            raise ValueError(f"arch takes compute capabilities such as 86 for sm_86, got {listed}")
        targets = [_Target(value) for value in arches]
    else:
        # Compile for each GPU exactly as Warp would on first launch: its architecture, or the
        # PTX target Warp falls back to when its compiler cannot emit that architecture's binary.
        seen: set[tuple[int, bool]] = set()
        for ordinal in range(wp.get_cuda_device_count()):
            device = wp.get_cuda_device(ordinal)
            compile_arch = device.get_cuda_compile_arch()
            assert compile_arch is not None
            use_ptx = device.get_cuda_output_format() == "ptx"
            if (compile_arch, use_ptx) not in seen:
                seen.add((compile_arch, use_ptx))
                targets.append(_Target(compile_arch, use_ptx))
                _LOGGER.info(
                    "precompile: no arch given: building for %s (%s, %s)",
                    targets[-1].label,
                    device.alias,
                    device.name,
                )
        if not targets:
            if not cpu:
                raise ValueError(
                    "no CUDA device found and no arch given: pass the compute capabilities to "
                    "build for (--arch 86,89), or cpu=True (drop --no-cpu) for the CPU alone"
                )
            _LOGGER.warning(
                "precompile: no arch given and no CUDA device found: building no CUDA kernels, "
                "CPU only (pass --arch to build for GPUs)"
            )
    if cpu:
        _LOGGER.info(
            "precompile: CPU objects target this machine's instruction set (-march=native); "
            "run them on the same CPU type"
        )
        targets.append(_Target(None))
    return targets


def kernel_variants() -> dict[str, tuple[int, ...]]:
    """
    List every Warp module ordito launches, with the CUDA launch widths it is built at.

    Imports every ordito module and creates the kernels ordito otherwise makes on first use. On
    the CPU device Warp loads every module at width 1 whatever the launch asks, so a CPU build is
    one per module.

    Returns
    -------
    dict of str to tuple of int
        Warp module name to the ascending ``block_dim``s a CUDA launch loads it at, its default
        included. The Warp modules importing ordito creates (sparse, geometry) are listed too.

    See Also
    --------
    precompile : Compile them.

    Examples
    --------
    ```python
    import ordito as od

    variants = od.precompile.kernel_variants()
    print(256 in variants["ordito.kernels.reduce"])  # True
    ```
    """
    import ordito
    import ordito.kernels
    from ordito.kernels import array as kernel_array

    # The wrappers too: a wrapper declares the launch widths it owns.
    for info in pkgutil.walk_packages(ordito.__path__, "ordito."):
        importlib.import_module(info.name)
    for build in kernel_array.LAZY_KERNEL_BUILDS:
        build()
    variants: dict[str, tuple[int, ...]] = {}
    for name, module in _warp_modules().items():
        if name in ordito.kernels.WARP_MODULES_BEFORE_ORDITO or not _compilable(module):
            continue
        widths = {int(module.options["block_dim"])}
        widths |= kernel_array.LAUNCH_BLOCK_DIMS.get(name, set())
        variants[name] = tuple(sorted(widths))
    return variants


def _warp_modules() -> dict[str, Any]:
    """Warp's registry of every module defined in this process, by name."""
    import warp._src.context as warp_context

    return dict(warp_context.user_modules)


def _compilable(module: Any) -> bool:
    """Whether a module has a kernel to compile: a concrete one, or a generic one with overloads."""
    return any(not kernel.is_generic or kernel.overloads for kernel in module.kernels.values())


# --- worker processes -----------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _start_worker(cache_dir: str | None) -> None:
    """Initialise Warp quietly on the requested cache, and create every module to build."""
    # Warp logs its start-up banner and a warning per module compiled for the CPU with
    # ``-march=native``; the parent logs what is built instead.
    wp.config.log_level = wp.LOG_ERROR
    if cache_dir is not None:
        wp.config.kernel_cache_dir = cache_dir
    wp.init()
    _WORKER["variants"] = kernel_variants()


def _worker_variants() -> tuple[str, dict[str, tuple[int, ...]], dict[str, int]]:
    """Return the resolved cache directory, the variants, and each module's kernel count."""
    modules = _warp_modules()
    variants: dict[str, tuple[int, ...]] = _WORKER["variants"]
    sizes = {name: len(modules[name].kernels) for name in variants}
    return str(wp.config.kernel_cache_dir), variants, sizes


def _build(
    name: str, arch: int | None, use_ptx: bool | None, block_dim: int
) -> tuple[bool, str | None]:
    """
    Compile one module at one width for one target: ``(compiled, error)``.

    ``compiled`` is ``False`` when the cache already held the build. ``wp.compile_aot_module``
    builds at the module's default width, and the width is part of the hash a launch looks up, so
    the option is set for the build and restored after.
    """
    import warp._src.context as warp_context

    module = _warp_modules()[name]
    default = module.options["block_dim"]
    wp.set_module_options({"block_dim": block_dim}, module)
    try:
        device = wp.get_device("cpu") if arch is None else None
        artifact = warp_context._get_module_artifact_path(  # pyright: ignore[reportPrivateUsage]
            module, device, arch, None, use_ptx
        )
        if artifact.exists():
            return False, None
        if arch is None:
            wp.compile_aot_module(module, device="cpu")
        else:
            wp.compile_aot_module(module, arch=arch, use_ptx=use_ptx)
        return True, None
    except Exception as error:  # noqa: BLE001 - one failed build is reported, the rest go on
        return False, f"{type(error).__name__}: {error}"
    finally:
        wp.set_module_options({"block_dim": default}, module)


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m ordito.precompile``: parse the options, build, and exit non-zero on a failure."""
    parser = argparse.ArgumentParser(
        prog="python -m ordito.precompile",
        description="Compile every kernel ordito launches into Warp's kernel cache.",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=None,
        help="CUDA compute capabilities to build for, e.g. 86,89 (repeatable); "
        "default: this machine's GPUs",
    )
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also build for the CPU device, for this machine's CPU type (default: on)",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="kernel cache directory (WARP_CACHE_PATH at run time)"
    )
    parser.add_argument("--jobs", type=int, default=None, help="worker processes (default: cores)")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Warp's start-up banner; the targets and the cache directory are logged instead.
    wp.config.log_level = wp.LOG_WARNING
    arch = None
    if options.arch is not None:
        try:
            arch = [int(value) for item in options.arch for value in item.split(",") if value]
        except ValueError:
            parser.error(f"--arch takes integers such as 86,89, got {options.arch}")
    try:
        report = precompile(arch, cpu=options.cpu, cache_dir=options.cache_dir, jobs=options.jobs)
    except ValueError as error:
        parser.error(str(error))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
