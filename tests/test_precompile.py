"""
Tests for ``ordito.precompile``: what it lists, how it picks targets, and what a launch loads.

Not a library comparison anywhere in this file: the kernel cache is Warp's own, and what a launch
loads from it is the oracle. Whether the list covers every module a real run loads is checked by
the census (``benchmarks/census.py``), which runs every benchmarked function.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest
import warp as wp

import ordito.kernels
import ordito.precompile as od
from ordito import linalg
from ordito.kernels.algorithms import conjugate_gradient as kernel_cg

# A kernel module with a declared launch width (``TILE_1D``) besides its default.
_MODULE = "ordito.kernels.measures"


def test_precompile_without_a_target_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GPU, no ``arch`` and ``cpu`` off there is nothing to build, and it says so."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 0)
    with pytest.raises(ValueError, match=r"no CUDA device found and no arch given.*--no-cpu"):
        od.precompile(cpu=False)


@pytest.mark.parametrize(
    ("arch", "jobs", "message"),
    [
        pytest.param([86, 35], None, "compute capabilities such as 86", id="arch-below-50"),
        pytest.param(86, 0, "jobs must be at least 1", id="no-jobs"),
    ],
)
def test_precompile_rejects_invalid_arguments(
    arch: int | list[int], jobs: int, message: str
) -> None:
    """An ``arch`` that is not a compute capability, and fewer than one worker, raise up front."""
    with pytest.raises(ValueError, match=message):
        od.precompile(arch, jobs=jobs)


def test_targets_follow_the_request(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Build ``arch`` as given, sorted and deduplicated, and without it and a GPU, the CPU alone.

    The log says that no CUDA kernel will be built.
    """
    targets = od._targets([120, 86, 120], cpu=True)  # pyright: ignore[reportPrivateUsage]
    assert [target.label for target in targets] == ["sm_86", "sm_120", "cpu"]
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 0)
    with caplog.at_level(logging.INFO, logger="ordito.precompile"):
        targets = od._targets(None, cpu=True)  # pyright: ignore[reportPrivateUsage]
    assert [target.label for target in targets] == ["cpu"]
    assert "building no CUDA kernels, CPU only" in caplog.text


def test_targets_default_to_this_machines_gpus(caplog: pytest.LogCaptureFixture) -> None:
    """Without ``arch``, each GPU is a target at the architecture Warp compiles for, logged."""
    if wp.get_cuda_device_count() == 0:
        pytest.skip("needs a CUDA device")
    with caplog.at_level(logging.INFO, logger="ordito.precompile"):
        targets = od._targets(None, cpu=False)  # pyright: ignore[reportPrivateUsage]
    compile_arch = wp.get_cuda_device(0).get_cuda_compile_arch()
    assert compile_arch in {target.arch for target in targets}
    assert "no arch given: building for sm_" in caplog.text


@pytest.mark.parametrize(
    ("argv", "cpu"),
    [pytest.param([], True, id="default"), pytest.param(["--no-cpu"], False, id="no-cpu")],
)
def test_command_line_builds_the_cpu_unless_told_not_to(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], cpu: bool
) -> None:
    """The CPU is a default target of the command line; ``--no-cpu`` turns it off."""
    requests: list[bool] = []

    def record(_arch: object, *, cpu: bool, **_options: object) -> od.PrecompileReport:
        requests.append(cpu)
        return od.PrecompileReport(targets=(), cache_dir="", built=0, cached=0, failed=())

    monkeypatch.setattr(od, "precompile", record)
    assert od._main(argv) == 0  # pyright: ignore[reportPrivateUsage]
    assert requests == [cpu]


def test_command_line_rejects_a_non_integer_arch() -> None:
    """``--arch`` takes integers; anything else is a usage error (exit status 2)."""
    with pytest.raises(SystemExit) as raised:
        od._main(["--arch", "sm_86"])  # pyright: ignore[reportPrivateUsage]
    assert raised.value.code == 2


def test_kernel_variants_lists_every_kernel_module_and_declared_width() -> None:
    """
    List every kernel module at its default width, plus each declared width.

    A kernel module's own declaration and a wrapper's (``cg_one_block``'s wide block) both count.
    """
    variants = od.kernel_variants()
    from warp._src.context import user_modules

    for info in pkgutil.walk_packages(ordito.kernels.__path__, "ordito.kernels."):
        importlib.import_module(info.name)
    # Read Warp's registry rather than calling ``wp.get_module``, which registers an empty module
    # for a package name (``ordito.kernels.algorithms``) and fails the backward-pass convention.
    for module in list(user_modules.values()):
        if module.name.startswith("ordito.kernels") and module.kernels:
            assert module.options["block_dim"] in variants[module.name], module.name
    assert variants[_MODULE] == (64, 256)
    wide = linalg._CG_ONE_BLOCK_WIDE  # pyright: ignore[reportPrivateUsage]
    assert wide in variants[kernel_cg.__name__]
    # The kernels made on first use exist after the listing: a cast no import builds.
    assert any(name.startswith("cast_float64_float32_") for name in variants)


def test_build_is_the_one_a_launch_loads(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Build a module that a fresh process then loads from the cache instead of compiling it.

    On CUDA the build is at a non-default width, which the process loads at; a second build finds
    the first.
    """
    resolved = wp.get_device(device)
    arch = resolved.get_cuda_compile_arch() if resolved.is_cuda else None
    use_ptx = resolved.get_cuda_output_format() == "ptx" if resolved.is_cuda else None
    width = 64 if resolved.is_cuda else 1
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / wp.config.version))
    importlib.import_module(_MODULE)
    build = od._build  # pyright: ignore[reportPrivateUsage]
    assert build(_MODULE, arch, use_ptx, width) == (True, None)
    assert build(_MODULE, arch, use_ptx, width) == (False, None)
    probe = (
        "import warp as wp\n"
        "wp.init()\n"
        f"import {_MODULE}\n"
        f"wp.get_module({_MODULE!r}).load(wp.get_device({device!r}), {width})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
        env={**os.environ, "WARP_CACHE_PATH": str(tmp_path)},
    )
    loads = [line for line in result.stdout.splitlines() if "load on device" in line]
    assert len(loads) == 1, result.stdout
    assert "(cached)" in loads[0], loads[0]
