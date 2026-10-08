"""
Run the suite on both devices, as two processes, because one process cannot do it cheaply.

``uv run python -m tests.devices`` runs a CUDA pass and then a CPU pass, and the CPU pass is spawned
with ``CUDA_VISIBLE_DEVICES=""``. That environment variable is the whole point of the runner:
**Warp's CPU work is ~36x slower once CUDA has been initialised in the process**, so a single
``pytest --device=both`` pays that on every CPU parametrization. Measured on one
``heat_signed_distance`` call -- same mesh, same code, only ``CUDA_VISIBLE_DEVICES`` differing --
and the figure is unchanged by ``warp.config.launch_array_access_mode`` at ``RELAXED`` or
``CHECKED``, so it is CUDA presence rather than the launch guard. Whole-suite
consequence: roughly 4x, one process against two.

Both-device coverage earns the second process. It is what caught the ``warp.fem`` ambient-device
leak in the former ``warp.fem`` Poisson backend -- broken for CPU input on any box with a GPU,
and invisible to both a CUDA-only run (devices matched) and a ``CUDA_VISIBLE_DEVICES=""`` run
(``warp.fem`` then defaults to CPU) -- and the module-scope ``wp.array`` in ``test_grouping``.

Usage
-----
``uv run python -m tests.devices``
    CUDA pass, then a CUDA-hidden CPU pass with the ``slow_cpu`` tests skipped and the
    ``device_agnostic`` source scans deselected (the CUDA pass ran them). The everyday both-device
    check.
``uv run python -m tests.devices --slow-cpu``
    Same, but the CPU pass runs ``--device=both``, which in a CUDA-hidden process means "all of CPU,
    skip nothing" -- the ``slow_cpu`` Poisson tests included.
``uv run python -m tests.devices --cpu-blocks``
    Adds a third pass: the CPU pass again with ``--cpu-blocks``, so every ``launch_tiled`` kernel
    runs all its lanes on the CPU device and the tiled reductions take their CUDA code path there.
    That turns the deterministic CPU device into an oracle for the CUDA-only tile code, which the
    ordinary CPU pass, running one lane per block, never executes. Opt-in: a CPU block costs a fixed
    price per launch far above the one-lane form.
``uv run python -m tests.devices -- -x -k grouping``
    Everything after ``--`` is forwarded to both pytest invocations.

Exits non-zero if either pass does, and reports both outcomes rather than stopping at the first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _run(label: str, device: str, hide_cuda: bool, extra: list[str]) -> tuple[str, int, float]:
    """Run one pytest pass and return ``(label, returncode, seconds)``."""
    env = dict(os.environ)
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", "pytest", "tests/", f"--device={device}", *extra]
    print(f"\n=== {label}: {' '.join(command)}", flush=True)
    if hide_cuda:
        print('    with CUDA_VISIBLE_DEVICES=""', flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=False)
    return label, completed.returncode, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Run both device passes and summarize them."""
    parser = argparse.ArgumentParser(prog="python -m tests.devices", description=__doc__)
    parser.add_argument(
        "--slow-cpu",
        action="store_true",
        help="run the slow_cpu tests in the CPU pass too (uses --device=both there)",
    )
    parser.add_argument(
        "--cpu-blocks",
        action="store_true",
        help="add a CPU pass with --cpu-blocks (every lane of a CPU block runs)",
    )
    parser.add_argument(
        "pytest_args", nargs="*", help="extra arguments forwarded to both pytest passes"
    )
    args = parser.parse_args(argv)

    # The static scans of the source tree (``device_agnostic``) answer the same in every process,
    # and they were a fifth of the CPU pass's wall clock; only the CUDA pass runs them.
    cpu_extra = ["--skip-device-agnostic", *args.pytest_args]
    results = [
        _run("cuda pass", "cuda", hide_cuda=False, extra=args.pytest_args),
        # ``both`` in a CUDA-hidden process resolves to ["cpu"] and disables the slow_cpu skip, so
        # it is how this runner says "all of CPU" without needing a second flag in conftest.
        _run("cpu pass", "both" if args.slow_cpu else "cpu", hide_cuda=True, extra=cpu_extra),
    ]
    if args.cpu_blocks:
        cpu_blocks = ["--cpu-blocks", *cpu_extra]
        results.append(_run("cpu-blocks pass", "cpu", hide_cuda=True, extra=cpu_blocks))

    print("\n=== summary")
    for label, returncode, seconds in results:
        print(
            f"    {label:<15} {'ok' if returncode == 0 else f'FAILED ({returncode})'}  "
            f"{seconds:7.1f} s"
        )
    print(f"    {'total':<15}    {sum(seconds for _, _, seconds in results):10.1f} s")
    return max(returncode for _, returncode, _ in results)


if __name__ == "__main__":
    raise SystemExit(main())
