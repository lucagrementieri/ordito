"""
Time both ordito targets, as two processes, so the CPU rows are not inflated.

``uv run python benchmarks/devices.py`` runs a CUDA pass and then a ``ordito-cpu`` pass, and the
CPU pass is spawned with ``CUDA_VISIBLE_DEVICES=""``. The variable is the whole point: **Warp's CPU
work costs more once CUDA has been initialised in the process**, and the charge behaves like a
per-launch one, so the factor scales with launch count rather than with work:

- a few big kernels -- ``edges_unique`` barely notices, a few tens of percent;
- an iterative solver pays more than an order of magnitude, and so does a whole test module built
  on one.

Unchanged by ``warp.config.launch_array_access_mode``, so it is CUDA presence and not the
launch guard. Either way a
``ordito-cpu`` row taken in a CUDA-initialised process is not a slow number, it is a **wrong** one,
and it reads as ordito losing to CPU references it actually beats. ``benchmarks/conftest.py`` warns
when that configuration is selected; this script is the way to avoid it.

``CUDA_VISIBLE_DEVICES`` has to be set before the process starts, so this can only be a second
process -- not a fixture and not a context manager. Same reason ``tests/devices.py`` exists.

Why the CPU pass is narrowed to ``-k ordito-cpu``
-------------------------------------------------
The CPU *reference* libraries (trimesh / igl / open3d / scipy / potpourri3d / pymeshlab / pyvista /
meshlib) are included in **every** pass by ``_selected_libraries``, so a naive two-pass split would
time them twice and write duplicate rows. Benchmark ids carry the library
(``test_faces_to_edges[bunny-ordito-cpu]``), so ``-k ordito-cpu`` keeps the second pass to exactly
the rows that need the CUDA-free process. The references stay in the CUDA pass, where they are
unaffected -- they never touch Warp.

Usage
-----
``uv run python benchmarks/devices.py``
    CUDA pass (``ordito-cuda`` + every reference), then a CUDA-hidden ``ordito-cpu`` pass.
``uv run python benchmarks/devices.py -- --size=small -k edges``
    Everything after ``--`` is forwarded to both pytest invocations.

Pass ``--benchmark-json=<path>`` and the two passes write ``<path>.cuda.json`` and
``<path>.cpu.json``: one file per process, because pytest-benchmark writes its JSON at session end
and the second pass would otherwise overwrite the first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _split_json_flag(extra: list[str], suffix: str) -> list[str]:
    """Give ``--benchmark-json`` a per-pass filename so the second pass cannot clobber the first."""
    out = []
    for argument in extra:
        if argument.startswith("--benchmark-json="):
            out.append(f"{argument}.{suffix}.json")
        else:
            out.append(argument)
    return out


def _and_into_k(extra: list[str], clause: str) -> list[str]:
    """
    Conjoin ``clause`` into an existing ``-k`` expression, or add one if there is none.

    Appending a second ``-k`` does **not** work: pytest keeps only the last one, so a forwarded
    ``-k edges_unique`` would be silently replaced by ``-k ordito-cpu`` and the pass would run
    every CPU row instead of the requested selection. Handles both ``-k expr`` and ``-k=expr``.
    """
    out: list[str] = []
    found = False
    index = 0
    while index < len(extra):
        argument = extra[index]
        if argument == "-k" and index + 1 < len(extra):
            out += ["-k", f"({extra[index + 1]}) and {clause}"]
            found = True
            index += 2
            continue
        if argument.startswith("-k="):
            out.append(f"-k=({argument[3:]}) and {clause}")
            found = True
            index += 1
            continue
        out.append(argument)
        index += 1
    if not found:
        out += ["-k", clause]
    return out


def _run(label: str, device: str, hide_cuda: bool, extra: list[str]) -> tuple[str, int, float]:
    """Run one benchmark pass and return ``(label, returncode, seconds)``."""
    env = dict(os.environ)
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
        # The references are timed in the CUDA pass; this pass exists only for the Warp CPU rows.
        extra = _and_into_k(extra, "ordito-cpu")
    command = [sys.executable, "-m", "pytest", "benchmarks/", f"--device={device}", *extra]
    print(f"\n=== {label}: {' '.join(command)}", flush=True)
    if hide_cuda:
        print('    with CUDA_VISIBLE_DEVICES=""', flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=False)
    return label, completed.returncode, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Run both ordito target passes and summarize them."""
    parser = argparse.ArgumentParser(
        prog="python benchmarks/devices.py",
        description="Time both ordito targets as two processes, the CPU one with CUDA hidden.",
    )
    parser.add_argument(
        "pytest_args", nargs="*", help="extra arguments forwarded to both pytest passes"
    )
    args = parser.parse_args(argv)

    results = [
        _run("cuda pass", "cuda", False, _split_json_flag(args.pytest_args, "cuda")),
        _run("cpu pass", "cpu", True, _split_json_flag(args.pytest_args, "cpu")),
    ]

    print("\n=== summary")
    for label, returncode, seconds in results:
        print(
            f"    {label:<10} {'ok' if returncode == 0 else f'FAILED ({returncode})'}  "
            f"{seconds:7.1f} s"
        )
    return max(returncode for _, returncode, _ in results)


if __name__ == "__main__":
    raise SystemExit(main())
