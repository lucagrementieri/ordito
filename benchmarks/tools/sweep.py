"""
Run every benchmark row on every mesh and library, each under a time limit, and merge the results.

A default run caps the CPU references at ``dragon`` and keeps the hand-written ``skip_larger_than``
caps; this sweep lifts both (``--no-size-caps``, every size named) to show how wide the gaps get.
Some reference calls then take hours, so each row runs under a wall-clock limit (a slow row is
already cut to the rounds that fit the harness's ``ROUND_BUDGET``). It times only the rows a
previous run did not:

    uv run python -m benchmarks.tools.sweep run <sweep_dir> --done <json_dir>
    uv run python -m benchmarks.tools.sweep merge <json_dir> <sweep_dir> <merged_dir>
    uv run python -m benchmarks.tools.render_docs <merged_dir>

``run`` collects every row under the uncapped flags, drops the ones already in ``--done``'s
``--benchmark-json`` files, writes the rest to ``<sweep_dir>/pending.txt`` and runs them module by
module through ``sweep_plugin.py``, which saves each row as it finishes. A row that runs past
``--timeout`` is recorded and the module restarts after it; so does a failed row, and a process that
dies (a segfault, the kernel's out-of-memory killer) is charged to the row it was running. A row
whose cell timed out or ran out of memory on a smaller mesh is recorded as predicted and not run
(``sweep_plugin.py``). Rerun the same command to resume: recorded rows are not run again.

``merge`` appends the sweep's rows to a copy of ``--done``'s JSON files and writes
``incomplete.jsonl``: one line per row that left no time, ``{"module", "group", "params",
"status", "detail", "limit", "predicted"}``, with ``status`` one of ``timeout``, ``oom``,
``crash`` or ``error`` (``results.classify``). ``render_docs.py`` draws those rows.

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
from pathlib import Path
from typing import Any

from benchmarks import results

_REPO = Path(__file__).resolve().parent.parent.parent
FLAGS = (
    "--device=cuda",
    "--bench-all-libs",
    "--no-size-caps",
    "--size",
    "small,medium,large,extralarge,huge",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, or an empty list when it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pytest(
    args: list[str], env: dict[str, str], log: Path | None = None, min_available: float = 0.0
) -> int:
    """
    Run pytest from the repository root, returning its exit code.

    While it runs, the host's available memory is polled; below ``min_available`` GiB the whole
    process group is killed with ``SIGKILL`` (exit ``-9``, which ``results.classify`` reads as out
    of memory) before the machine starts swapping and stops responding.
    """
    with open(log, "a") if log is not None else open(os.devnull, "w") as out:
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", "pytest", *args, "-p", "no:cacheprovider"],
            cwd=_REPO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    return process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                if min_available > 0.0 and _available_gib() < min_available:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return -signal.SIGKILL
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            raise


def _available_gib() -> float:
    """Return the host's ``MemAvailable`` in GiB."""
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("inf")


def collect(out: Path) -> list[dict[str, Any]]:
    """Collect every row under the sweep's flags."""
    env = dict(os.environ, SWEEP_COLLECT=str(out))
    code = _pytest(
        ["benchmarks", "--collect-only", "-q", "-p", "benchmarks.sweep_plugin", *FLAGS], env
    )
    if code != 0:
        raise SystemExit(f"collection failed with exit code {code}")
    return _read_jsonl(out)


def plan(sweep: Path, done: Path | None) -> list[str]:
    """Write ``pending.txt``: every row under the sweep's flags that ``done`` has no time for."""
    finished: set[str] = set()
    if done is not None:
        for path in done.glob("*.json"):
            finished |= {row["fullname"] for row in json.loads(path.read_text())["benchmarks"]}
    items = collect(sweep / "collected.jsonl")
    pending = [
        item["nodeid"]
        for item in items
        if item["nodeid"] not in finished and "test_meshes.py" not in item["nodeid"]
    ]
    (sweep / "pending.txt").write_text("\n".join(pending) + "\n")
    return pending


def run(sweep: Path, done: Path | None, timeout: float, min_available: float) -> None:
    """Run the pending rows module by module, restarting after a timeout, a failure or a crash."""
    sweep.mkdir(parents=True, exist_ok=True)
    pending_file = sweep / "pending.txt"
    pending = (
        [line for line in pending_file.read_text().splitlines() if line.strip()]
        if pending_file.exists()
        else plan(sweep, done)
    )
    by_module: dict[str, list[str]] = collections.defaultdict(list)
    for nodeid in pending:
        by_module[nodeid.split("::")[0]].append(nodeid)
    env = dict(os.environ, SWEEP_DIR=str(sweep), SWEEP_TIMEOUT=str(timeout))
    for module, nodeids in by_module.items():
        while True:
            recorded = {o["nodeid"] for o in _read_jsonl(sweep / "outcomes.jsonl")}
            remaining = [nodeid for nodeid in nodeids if nodeid not in recorded]
            if not remaining:
                break
            _log(sweep, f"{module}: running {len(remaining)} rows")
            listing = sweep / "args.txt"
            listing.write_text("\n".join(remaining) + "\n")
            code = _pytest(
                [f"@{listing}", "-q", "-p", "benchmarks.sweep_plugin", *FLAGS],
                env,
                sweep / "pytest.log",
                min_available,
            )
            _log(sweep, f"{module}: exit {code}")
            if code == -signal.SIGKILL:
                _log(sweep, f"{module}: killed below {min_available:g} GiB of available memory")
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
    (sweep / "DONE").write_text(time.strftime("%c") + "\n")


def _log(sweep: Path, message: str) -> None:
    """Append a timestamped line to the sweep's own log."""
    with open(sweep / "sweep.log", "a") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def merge(done: Path, sweep: Path, out: Path) -> None:
    """Write ``done``'s JSON files plus the sweep's rows to ``out``, and ``incomplete.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(sweep / "rows.jsonl"):
        rows[results.module_of(row["fullname"])].append(row)
    for path in sorted(done.glob("test_*.json")):
        data = json.loads(path.read_text())
        known = {row["fullname"] for row in data["benchmarks"]}
        data["benchmarks"] += [
            row for row in rows.pop(path.stem, []) if row["fullname"] not in known
        ]
        (out / path.name).write_text(json.dumps(data))
    if rows:
        print(f"no --done file for {sorted(rows)}; their rows are left out", file=sys.stderr)

    # The last outcome of a row wins: a resumed sweep may have retried it.
    final = {o["nodeid"]: o for o in _read_jsonl(sweep / "outcomes.jsonl")}
    started = {s["nodeid"]: s for s in _read_jsonl(sweep / "started.jsonl")}
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
    print(f"merged {sum(1 for _ in _read_jsonl(sweep / 'rows.jsonl'))} rows; {dict(counts)}")


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
        "--min-available",
        type=float,
        default=8.0,
        help="GiB of available host memory below which the running row is killed as out of memory",
    )
    merge_parser = commands.add_parser("merge", help="merge a sweep into a run's JSON files")
    merge_parser.add_argument("done", type=Path, help="directory of --benchmark-json files")
    merge_parser.add_argument("sweep_dir", type=Path)
    merge_parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.sweep_dir, args.done, args.timeout, args.min_available)
    else:
        merge(args.done, args.sweep_dir, args.out_dir)


if __name__ == "__main__":
    main()
