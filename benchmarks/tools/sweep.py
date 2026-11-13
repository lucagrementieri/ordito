"""
Run every benchmark row on every mesh and library, each under a time limit, and merge the results.

A default run caps the CPU references at ``dragon`` and keeps the hand-written ``skip_larger_than``
caps; this sweep lifts both (``--no-size-caps``, every size named) to show how wide the gaps get.
Some reference calls then take hours, so each row runs under a wall-clock limit (a slow row is
already cut to the rounds that fit the harness's ``ROUND_BUDGET``). A full run is two sweeps, one
a process kind, merged:

    uv run python -m benchmarks.tools.sweep run <cuda_dir> -k "ordito-cuda or pytorch3d-cuda"
    uv run python -m benchmarks.tools.sweep run <cpu_dir> --device cpu
    uv run python -m benchmarks.tools.sweep merge <merged_dir> <cuda_dir> <cpu_dir>
    uv run python -m benchmarks.tools.render_docs <merged_dir>

``--device cpu`` runs with ``CUDA_VISIBLE_DEVICES=""`` (an ``ordito-cpu`` row timed in a
CUDA-initialised process is inflated, ``tools/devices.py``), so it times ``ordito-cpu`` and every
CPU reference; the CUDA sweep then keeps the two device libraries with ``-k``. ``--done
<json_dir>`` skips the rows a previous run's ``--benchmark-json`` files already time.

``run`` collects every row under the uncapped flags, drops the ones already in ``--done``, writes
the rest to ``<sweep_dir>/pending.txt`` and runs them module by module through
``sweep_plugin.py``, which saves each row as it finishes. A row that runs past ``--timeout`` is
recorded and the module restarts after it; so does a failed row, and a process that dies (a
segfault, the out-of-memory killer) is charged to the row it was running. A row whose cell timed
out or ran out of memory on a smaller mesh is recorded as predicted and not run
(``sweep_plugin.py``). Rerun the same command to resume: recorded rows are not run again.

**Memory.** Each pytest process runs in a ``systemd-run --user --scope`` with ``MemoryMax`` set to
``--memory-max`` and no swap, so a row that outgrows it is killed by the kernel and recorded as
out of memory, and the machine never swaps. A row is also killed when the host's available memory
drops below ``--min-available``. Memory other processes hold must not decide a row: before each
module chunk the sweep waits until other processes hold at most ``--max-foreign-host`` GiB of the
host and (CUDA) ``--max-foreign-gpu`` GiB of the GPU, and a row that fails (out of memory, a
crash, an error) while they held more than either is forgotten and run again, up to ``RETRIES``
times.

``merge`` writes the sweeps' rows, over a copy of ``--done``'s JSON files when given, one JSON file
a module, and ``incomplete.jsonl``: one line per row that left no time, ``{"module", "group",
"params", "status", "detail", "limit", "predicted"}``, with ``status`` one of ``timeout``,
``oom``, ``crash`` or ``error`` (``results.classify``). ``render_docs.py`` draws those rows.

Run it on an idle machine, like any timing: nothing else should share the GPU or the cores.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks import results

_REPO = Path(__file__).resolve().parent.parent.parent
UNCAPPED = ("--bench-all-libs", "--no-size-caps", "--size", "small,medium,large,extralarge,huge")
# How many times a row that failed under another process's memory is run again.
RETRIES = 3
# Seconds between two looks at the machine while waiting for it to free memory.
_WAIT_POLL = 30.0
# Seconds between two samples of other processes' memory while a chunk runs.
_SAMPLE_EVERY = 5.0


@dataclass
class Settings:
    """What every pytest process of one sweep runs with."""

    device: str = "cuda"
    select: str | None = None
    memory_max: float = 0.0
    min_available: float = 0.0
    max_foreign_gpu: float = 4.0
    max_foreign_host: float = 24.0

    def flags(self) -> list[str]:
        """Return the pytest flags of the sweep's rows."""
        selection = ["-k", self.select] if self.select else []
        return [f"--device={self.device}", *UNCAPPED, *selection]

    def env(self, **extra: str) -> dict[str, str]:
        """Return the environment of the sweep's pytest processes."""
        env = dict(os.environ, **extra)
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env


@dataclass
class Pressure:
    """The most memory other processes held while one pytest process ran, in GiB."""

    foreign_gpu: float = 0.0
    foreign_host: float = 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, or an empty list when it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pytest(
    args: list[str], settings: Settings, env: dict[str, str], log: Path | None = None
) -> tuple[int, Pressure]:
    """
    Run pytest from the repository root, returning its exit code and the memory others held.

    The process runs under a memory cap when ``settings.memory_max`` is set. While it runs, the
    host's available memory is polled; below ``settings.min_available`` GiB the whole process group
    is killed with ``SIGKILL`` (exit ``-9``, which ``results.classify`` reads as out of memory)
    before the machine starts swapping and stops responding.
    """
    command = ["uv", "run", "--no-sync", "python", "-m", "pytest", *args, "-p", "no:cacheprovider"]
    if settings.memory_max > 0.0:
        cap = f"{int(settings.memory_max * 1024)}M"
        scope = ["systemd-run", "--user", "--scope", "--quiet", "--collect"]
        command = [*scope, "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0", *command]
    pressure = Pressure()
    with open(log, "a") if log is not None else open(os.devnull, "w") as out:
        process = subprocess.Popen(
            command,
            cwd=_REPO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        sampled = 0.0
        try:
            while True:
                try:
                    return process.wait(timeout=0.1), pressure
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() - sampled > _SAMPLE_EVERY:
                    sampled = time.monotonic()
                    own = _group_pids(process.pid)
                    gpu = _foreign_gpu_gib(own) if settings.device == "cuda" else 0.0
                    pressure.foreign_gpu = max(pressure.foreign_gpu, gpu)
                    host = _used_gib() - _rss_gib(own)
                    pressure.foreign_host = max(pressure.foreign_host, host)
                available = _available_gib()
                if settings.min_available > 0.0 and available < settings.min_available:
                    own = _group_pids(process.pid)
                    host = _used_gib() - _rss_gib(own)
                    pressure.foreign_host = max(pressure.foreign_host, host)
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return -signal.SIGKILL, pressure
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            raise


def _meminfo(field: str) -> float:
    """Return one ``/proc/meminfo`` field in GiB."""
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) / 2**20
    return float("inf")


def _available_gib() -> float:
    """Return the host's ``MemAvailable`` in GiB."""
    return _meminfo("MemAvailable")


def _used_gib() -> float:
    """Return the host memory in use, ``MemTotal - MemAvailable``, in GiB."""
    return _meminfo("MemTotal") - _meminfo("MemAvailable")


def _group_pids(pgid: int) -> set[int]:
    """Return the processes of one process group."""
    pids: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may hold spaces and parentheses; the fields after it do not.
        fields = stat[stat.rindex(")") + 2 :].split()
        if int(fields[2]) == pgid:
            pids.add(int(entry.name))
    return pids


def _rss_gib(pids: set[int]) -> float:
    """Return the resident memory of some processes in GiB."""
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/statm") as handle:
                total += int(handle.read().split()[1])
        except OSError:
            continue
    return total * os.sysconf("SC_PAGE_SIZE") / 2**30


def _foreign_gpu_gib(own: set[int]) -> float:
    """Return the GPU memory processes outside ``own`` hold, in GiB (0 when it cannot be read)."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for line in listing.splitlines():
        pid, _, used = line.partition(",")
        if pid.strip().isdigit() and int(pid) not in own and used.strip().isdigit():
            total += int(used) / 1024
    return total


def _wait_for_memory(sweep: Path, settings: Settings) -> None:
    """Wait until other processes hold no more memory than the sweep allows them."""
    waited = False
    while True:
        # Nothing of the sweep's own runs between chunks: every byte in use is someone else's.
        gpu = _foreign_gpu_gib(set()) if settings.device == "cuda" else 0.0
        host = _used_gib()
        if host <= settings.max_foreign_host and gpu <= settings.max_foreign_gpu:
            if waited:
                _log(sweep, f"memory free again: foreign host {host:.1f} GiB, GPU {gpu:.1f} GiB")
            return
        if not waited:
            _log(
                sweep,
                f"waiting for memory: foreign host {host:.1f} GiB (limit "
                f"{settings.max_foreign_host:g}), GPU {gpu:.1f} GiB (limit "
                f"{settings.max_foreign_gpu:g})",
            )
            waited = True
        time.sleep(_WAIT_POLL)


def collect(out: Path, settings: Settings) -> list[dict[str, Any]]:
    """Collect every row under the sweep's flags."""
    env = settings.env(SWEEP_COLLECT=str(out))
    args = ["benchmarks", "--collect-only", "-q", "-p", "benchmarks.sweep_plugin"]
    code, _pressure = _pytest([*args, *settings.flags()], Settings(device=settings.device), env)
    if code != 0:
        raise SystemExit(f"collection failed with exit code {code}")
    return _read_jsonl(out)


def plan(sweep: Path, done: Path | None, settings: Settings) -> list[str]:
    """Write ``pending.txt``: every row under the sweep's flags that ``done`` has no time for."""
    finished: set[str] = set()
    if done is not None:
        for path in done.glob("*.json"):
            finished |= {row["fullname"] for row in json.loads(path.read_text())["benchmarks"]}
    items = collect(sweep / "collected.jsonl", settings)
    pending = [
        item["nodeid"]
        for item in items
        if item["nodeid"] not in finished and "test_meshes.py" not in item["nodeid"]
    ]
    (sweep / "pending.txt").write_text("\n".join(pending) + "\n")
    return pending


def run(sweep: Path, done: Path | None, timeout: float, settings: Settings) -> None:
    """Run the pending rows module by module, restarting after a timeout, a failure or a crash."""
    sweep.mkdir(parents=True, exist_ok=True)
    pending_file = sweep / "pending.txt"
    pending = (
        [line for line in pending_file.read_text().splitlines() if line.strip()]
        if pending_file.exists()
        else plan(sweep, done, settings)
    )
    by_module: dict[str, list[str]] = collections.defaultdict(list)
    for nodeid in pending:
        by_module[nodeid.split("::")[0]].append(nodeid)
    env = settings.env(SWEEP_DIR=str(sweep), SWEEP_TIMEOUT=str(timeout))
    for module, nodeids in by_module.items():
        while True:
            recorded = {o["nodeid"] for o in _read_jsonl(sweep / "outcomes.jsonl")}
            remaining = [nodeid for nodeid in nodeids if nodeid not in recorded]
            if not remaining:
                break
            _wait_for_memory(sweep, settings)
            _log(sweep, f"{module}: running {len(remaining)} rows")
            listing = sweep / "args.txt"
            listing.write_text("\n".join(remaining) + "\n")
            before = len(_read_jsonl(sweep / "outcomes.jsonl"))
            code, pressure = _pytest(
                [f"@{listing}", "-q", "-p", "benchmarks.sweep_plugin", *settings.flags()],
                settings,
                env,
                sweep / "pytest.log",
            )
            _log(
                sweep,
                f"{module}: exit {code} (foreign memory at most: host {pressure.foreign_host:.1f}"
                f" GiB, GPU {pressure.foreign_gpu:.1f} GiB)",
            )
            if code == -signal.SIGKILL:
                _log(sweep, f"{module}: killed below {settings.min_available:g} GiB available")
            now = {o["nodeid"] for o in _read_jsonl(sweep / "outcomes.jsonl")}
            if len(now) == len(recorded) or (code == -signal.SIGKILL and set(remaining) - now):
                # Nothing recorded, or the watchdog killed it: the process died under a row (a
                # segfault, an out-of-memory kill) or before any. Charge the row it started
                # last, so the loop moves on.
                started = [s["nodeid"] for s in _read_jsonl(sweep / "started.jsonl")]
                fresh = [n for n in remaining if n not in now]
                victim = started[-1] if started and started[-1] in fresh else fresh[0]
                with open(sweep / "outcomes.jsonl", "a") as handle:
                    record = {"nodeid": victim, "outcome": "crashed", "exit": code}
                    handle.write(json.dumps(record) + "\n")
                _log(sweep, f"{module}: charged a crash to {victim}")
            busy = (
                pressure.foreign_gpu > settings.max_foreign_gpu
                or pressure.foreign_host > settings.max_foreign_host
            )
            if busy:
                _forget_failures(sweep, before)
    (sweep / "DONE").write_text(time.strftime("%c") + "\n")


def _forget_failures(sweep: Path, start: int) -> None:
    """
    Drop the failures recorded from line ``start`` of ``outcomes.jsonl`` on, so they run again.

    Called when other processes held enough memory to cause them. A timeout stays (memory others
    hold does not slow a row down without swap), and so does every row that passed. A row is run
    again at most ``RETRIES`` times.
    """
    path = sweep / "outcomes.jsonl"
    outcomes = _read_jsonl(path)
    retried = collections.Counter(r["nodeid"] for r in _read_jsonl(sweep / "retries.jsonl"))
    keep = outcomes[:start]
    for outcome in outcomes[start:]:
        status = results.classify(outcome)
        nodeid = outcome["nodeid"]
        if status is None or status[0] == "timeout" or retried[nodeid] >= RETRIES:
            keep.append(outcome)
            continue
        retried[nodeid] += 1
        with open(sweep / "retries.jsonl", "a") as handle:
            handle.write(json.dumps({"nodeid": nodeid, "dropped": outcome}) + "\n")
        _log(sweep, f"{nodeid}: {status[0]} under other processes' memory; will run again")
    path.write_text("".join(json.dumps(o) + "\n" for o in keep))


def _log(sweep: Path, message: str) -> None:
    """Append a timestamped line to the sweep's own log."""
    with open(sweep / "sweep.log", "a") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def merge(out: Path, sweeps: list[Path], done: Path | None) -> None:
    """Write ``done``'s JSON files, if any, plus the sweeps' rows, and ``incomplete.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for sweep in sweeps:
        for row in _read_jsonl(sweep / "rows.jsonl"):
            rows[results.module_of(row["fullname"])].append(row)
    machine = _machine_info()
    modules = {path.stem for path in done.glob("test_*.json")} if done is not None else set()
    for module in sorted(modules | set(rows)):
        data: dict[str, Any]
        if module in modules:
            assert done is not None
            data = json.loads((done / f"{module}.json").read_text())
        else:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            data = {"machine_info": machine, "datetime": stamp, "benchmarks": []}
        known = {row["fullname"] for row in data["benchmarks"]}
        for row in rows.pop(module, []):
            if row["fullname"] not in known:
                data["benchmarks"].append(row)
                known.add(row["fullname"])
        (out / f"{module}.json").write_text(json.dumps(data))

    # The last outcome of a row wins: a resumed sweep may have retried it.
    final: dict[str, dict[str, Any]] = {}
    started: dict[str, dict[str, Any]] = {}
    total = 0
    for sweep in sweeps:
        final |= {o["nodeid"]: o for o in _read_jsonl(sweep / "outcomes.jsonl")}
        started |= {s["nodeid"]: s for s in _read_jsonl(sweep / "started.jsonl")}
        total += len(_read_jsonl(sweep / "rows.jsonl"))
    incomplete = {nodeid: status for nodeid, o in final.items() if (status := results.classify(o))}
    # A process that died before its first row charges the crash to a row that never started, so
    # nothing names that row's cell.
    for nodeid in sorted(set(incomplete) - set(started)):
        print(f"{nodeid}: {incomplete.pop(nodeid)[0]} before it started; left out", file=sys.stderr)
    with open(out / "incomplete.jsonl", "w") as handle:
        for nodeid, (status, detail) in incomplete.items():
            outcome = final[nodeid]
            record = {
                "module": results.module_of(nodeid),
                "group": started[nodeid]["group"],
                "params": started[nodeid]["params"],
                "status": status,
                "detail": detail,
                "limit": outcome.get("seconds") if status == "timeout" else None,
                "predicted": outcome["outcome"] == "predicted",
            }
            handle.write(json.dumps(record, default=str) + "\n")
    counts = collections.Counter(status for status, _detail in incomplete.values())
    print(f"merged {total} rows; {dict(counts)}")


def _machine_info() -> dict[str, Any]:
    """Return pytest-benchmark's description of this machine, for a module no run recorded."""
    from pytest_benchmark.plugin import pytest_benchmark_generate_machine_info

    return pytest_benchmark_generate_machine_info()


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``run`` and ``merge``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="time the rows a previous run did not")
    run_parser.add_argument("sweep_dir", type=Path)
    run_parser.add_argument("--done", type=Path, help="directory of --benchmark-json files")
    run_parser.add_argument("--timeout", type=float, default=600.0, help="seconds per row")
    run_parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cpu: hide CUDA and time ordito-cpu and the CPU references",
    )
    run_parser.add_argument("-k", dest="select", help="pytest -k expression narrowing the rows")
    run_parser.add_argument(
        "--memory-max",
        type=float,
        default=0.0,
        help="GiB a pytest process may hold (a cgroup cap, no swap); 0 for none",
    )
    run_parser.add_argument(
        "--min-available",
        type=float,
        default=8.0,
        help="GiB of available host memory below which the running row is killed as out of memory",
    )
    run_parser.add_argument(
        "--max-foreign-gpu",
        type=float,
        default=4.0,
        help="GiB of GPU memory other processes may hold before rows wait (and failures rerun)",
    )
    run_parser.add_argument(
        "--max-foreign-host",
        type=float,
        default=24.0,
        help="GiB of host memory other processes may hold before failures rerun",
    )
    merge_parser = commands.add_parser("merge", help="merge sweeps into one directory")
    merge_parser.add_argument("out_dir", type=Path)
    merge_parser.add_argument("sweep_dirs", type=Path, nargs="+")
    merge_parser.add_argument("--done", type=Path, help="directory of --benchmark-json files")
    args = parser.parse_args(argv)
    if args.command == "run":
        settings = Settings(
            device=args.device,
            select=args.select,
            memory_max=args.memory_max,
            min_available=args.min_available,
            max_foreign_gpu=args.max_foreign_gpu,
            max_foreign_host=args.max_foreign_host,
        )
        run(args.sweep_dir, args.done, args.timeout, settings)
    else:
        merge(args.out_dir, args.sweep_dirs, args.done)


if __name__ == "__main__":
    main()
