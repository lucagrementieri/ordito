"""
Run every benchmark row on every mesh and library, each under a time limit, and merge the results.

A default run caps the CPU references at ``dragon`` and keeps the hand-written ``skip_larger_than``
caps; this sweep lifts both (``--no-size-caps``, every size named) to show how wide the gaps get.
Some reference calls then take hours, so each row runs under a wall-clock limit, and a slow row is
cut to the rounds that fit ``--round-budget``. It times only the rows a previous run did not:

    uv run python benchmarks/sweep.py run <sweep_dir> --done <json_dir>
    uv run python benchmarks/sweep.py merge <json_dir> <sweep_dir> <merged_dir>
    uv run python benchmarks/render_docs.py <merged_dir>

``run`` collects every row under the uncapped flags, drops the ones already in ``--done``'s
``--benchmark-json`` files, writes the rest to ``<sweep_dir>/pending.txt`` and runs them module by
module through ``sweep_plugin.py``, which saves each row as it finishes. A row that runs past
``--timeout`` is recorded and the module restarts after it; so does a failed row, and a process that
dies (a segfault, the kernel's out-of-memory killer) is charged to the row it was running. Rerun
the same command to resume: recorded rows are not run again.

``merge`` appends the sweep's rows to a copy of ``--done``'s JSON files and writes
``incomplete.jsonl``: one line per row that left no time, ``{"module", "group", "params",
"status", "detail", "limit"}``, with ``status`` one of ``timeout``, ``oom``, ``crash`` or
``error``. ``render_docs.py`` draws those rows.

Run it on an idle machine, like any timing: nothing else should share the GPU or the cores.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
FLAGS = (
    "--device=cuda",
    "--bench-all-libs",
    "--no-size-caps",
    "--size",
    "small,medium,large,extralarge,huge",
)
# A process killed by a signal reports 128 + the signal through ``uv run``, or minus the signal.
_SIGNALS = {6: "aborted", 9: "killed", 11: "segfault"}
_MEMORY_ERRORS = ("MemoryError", "out of memory", "cudaErrorMemoryAllocation", "bad_alloc")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, or an empty list when it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pytest(args: list[str], env: dict[str, str], log: Path | None = None) -> int:
    """Run pytest from the repository root, returning its exit code."""
    with open(log, "a") if log is not None else open(os.devnull, "w") as out:
        return subprocess.run(
            ["uv", "run", "python", "-m", "pytest", *args, "-p", "no:cacheprovider"],
            cwd=_REPO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def collect(nodeids: list[str] | None, out: Path) -> list[dict[str, Any]]:
    """Collect rows under the sweep's flags, all of them or the given ids, with their parameters."""
    env = dict(os.environ, SWEEP_COLLECT=str(out))
    args = ["benchmarks"]
    if nodeids is not None:
        listing = out.with_suffix(".args")
        listing.write_text("\n".join(nodeids) + "\n")
        args = [f"@{listing}"]
    code = _pytest([*args, "--collect-only", "-q", "-p", "benchmarks.sweep_plugin", *FLAGS], env)
    if code != 0:
        raise SystemExit(f"collection failed with exit code {code}")
    return _read_jsonl(out)


def plan(sweep: Path, done: Path | None) -> list[str]:
    """Write ``pending.txt``: every row under the sweep's flags that ``done`` has no time for."""
    finished: set[str] = set()
    if done is not None:
        for path in done.glob("*.json"):
            finished |= {row["fullname"] for row in json.loads(path.read_text())["benchmarks"]}
    items = collect(None, sweep / "collected.jsonl")
    pending = [
        item["nodeid"]
        for item in items
        if item["nodeid"] not in finished and "test_meshes.py" not in item["nodeid"]
    ]
    (sweep / "pending.txt").write_text("\n".join(pending) + "\n")
    return pending


def run(sweep: Path, done: Path | None, timeout: float, round_budget: float) -> None:
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
                [
                    f"@{listing}",
                    "-q",
                    "-p",
                    "benchmarks.sweep_plugin",
                    *FLAGS,
                    "--round-budget",
                    str(round_budget),
                ],
                env,
                sweep / "pytest.log",
            )
            _log(sweep, f"{module}: exit {code}")
            if len({o["nodeid"] for o in _read_jsonl(sweep / "outcomes.jsonl")}) == len(recorded):
                # Nothing recorded: the process died under a row (a segfault, an out-of-memory
                # kill) or before any. Charge the row it started last, so the loop moves on.
                started = [s["nodeid"] for s in _read_jsonl(sweep / "started.jsonl")]
                victim = started[-1] if started and started[-1] in remaining else remaining[0]
                with open(sweep / "outcomes.jsonl", "a") as handle:
                    record = {"nodeid": victim, "outcome": "crashed", "exit": code}
                    handle.write(json.dumps(record) + "\n")
                _log(sweep, f"{module}: charged a crash to {victim}")
    (sweep / "DONE").write_text(time.strftime("%c") + "\n")


def _log(sweep: Path, message: str) -> None:
    """Append a timestamped line to the sweep's own log."""
    with open(sweep / "sweep.log", "a") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def classify(outcome: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(status, detail)`` for a row that left no time, or ``None`` for one that did."""
    kind = outcome["outcome"]
    if kind == "timeout":
        return "timeout", f"stopped at {outcome['seconds']:g} s"
    if kind == "crashed":
        code = int(outcome.get("exit", 0))
        signal = code - 128 if code > 128 else -code if code < 0 else 0
        if signal == 9:
            return "oom", "killed, most likely by the kernel's out-of-memory killer"
        return "crash", _SIGNALS.get(signal, f"exit code {code}")
    if kind == "failed":
        error = str(outcome.get("error", ""))
        if any(marker in error for marker in _MEMORY_ERRORS):
            return "oom", "ran out of memory"
        lines = [line for line in error.splitlines() if line.strip()]
        return "error", lines[-1][:200] if lines else "failed"
    return None


def merge(done: Path, sweep: Path, out: Path) -> None:
    """Write ``done``'s JSON files plus the sweep's rows to ``out``, and ``incomplete.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(sweep / "rows.jsonl"):
        rows[Path(row["fullname"].split("::")[0]).stem].append(row)
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
    incomplete = {nodeid: status for nodeid, o in final.items() if (status := classify(o))}
    items = {
        item["nodeid"]: item
        for item in (collect(sorted(incomplete), out / "collected.jsonl") if incomplete else [])
    }
    with open(out / "incomplete.jsonl", "w") as handle:
        for nodeid, (status, detail) in incomplete.items():
            item = items[nodeid]
            record = {
                "module": Path(nodeid.split("::")[0]).stem,
                "group": item["group"],
                "params": item["params"],
                "status": status,
                "detail": detail,
                "limit": final[nodeid].get("seconds") if status == "timeout" else None,
            }
            handle.write(json.dumps(record, default=str) + "\n")
    for scratch in ("collected.jsonl", "collected.args"):
        (out / scratch).unlink(missing_ok=True)
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
        "--round-budget", type=float, default=60.0, help="seconds of timed rounds per row"
    )
    merge_parser = commands.add_parser("merge", help="merge a sweep into a run's JSON files")
    merge_parser.add_argument("done", type=Path, help="directory of --benchmark-json files")
    merge_parser.add_argument("sweep_dir", type=Path)
    merge_parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.sweep_dir, args.done, args.timeout, args.round_budget)
    else:
        merge(args.done, args.sweep_dir, args.out_dir)


if __name__ == "__main__":
    main()
