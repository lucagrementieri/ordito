"""
Read benchmark results: one definition of a cell, shared by the harness and every tool.

A **cell** is one input of one function: ``(module, group, mesh_name, rest)``, where ``rest`` is
every parametrize value except ``library`` and ``mesh_name``, as sorted ``(name, str(value))``
pairs. Each library timed on the cell is one entry of it. The harness (``conftest.py``'s
generated tables), the sweep and the tools under ``tools/`` all key cells through
[`split_params`][], so they agree on what a cell is.

Two sources feed a directory of results: the ``--benchmark-json`` files of a run (one per test
module), and the ``incomplete.jsonl`` that ``tools/sweep.py merge`` writes beside them for the rows
that left no time (see [`load_incomplete`][]).
"""

from __future__ import annotations

import collections
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

ORDITO_IDS = frozenset({"ordito-cpu", "ordito-cuda"})
# The library every ratio is taken against.
SUBJECT = "ordito-cuda"

Rest = tuple[tuple[str, str], ...]
# ``(module, group, mesh_name, rest)``.
CellKey = tuple[str, str, "str | None", Rest]
Cells = dict[CellKey, dict[str, float]]

# Markers of an out-of-memory failure in a row's error text.
_MEMORY_ERRORS = ("MemoryError", "out of memory", "cudaErrorMemoryAllocation", "bad_alloc")
# A process killed by a signal reports 128 + the signal through ``uv run``, or minus the signal.
_SIGNALS = {6: "aborted", 9: "killed", 11: "segfault"}


def split_params(params: Mapping[str, object] | None) -> tuple[str | None, str | None, Rest]:
    """Split a row's parameters into ``(library, mesh_name, rest)``."""
    rest = dict(params or {})
    library = rest.pop("library", None)
    mesh_name = rest.pop("mesh_name", None)
    return (
        None if library is None else str(library),
        None if mesh_name is None else str(mesh_name),
        tuple(sorted((name, str(value)) for name, value in rest.items())),
    )


def module_of(nodeid: str) -> str:
    """Return the test module of a node id or ``fullname``: ``test_edges``."""
    return Path(nodeid.split("::")[0]).stem


def is_reference(library: str) -> bool:
    """Whether a library id is a reference rather than ordito on some device."""
    return library not in ORDITO_IDS


class Suspect(NamedTuple):
    """One row whose median sits well above its own minimum (``tools/aggregate.py``)."""

    ratio: float
    rounds: int
    minimum: float
    median: float
    name: str


@dataclass
class Results:
    """
    The measured cells of a directory of ``--benchmark-json`` files.

    ``cells`` maps each cell to ``{library: median_seconds}``; ``modules`` counts the rows read per
    module; ``suspects`` lists every row whose median exceeds its minimum, unfiltered.
    """

    cells: Cells = field(default_factory=lambda: collections.defaultdict(dict))
    modules: dict[str, int] = field(default_factory=dict)
    suspects: list[Suspect] = field(default_factory=list)


def load(json_dir: Path | str) -> Results:
    """Read every ``*.json`` of ``json_dir``; a file that does not parse is reported and skipped."""
    results = Results()
    for path in sorted(Path(json_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as error:  # a module still being written, or a crash
            print(f"  ! {path.stem}: {error}", file=sys.stderr)
            continue
        rows = data["benchmarks"]
        results.modules[path.stem] = len(rows)
        for row in rows:
            library, mesh_name, rest = split_params(row["params"])
            if library is None:
                continue
            stats = row["stats"]
            results.cells[(path.stem, row["group"], mesh_name, rest)][library] = stats["median"]
            if stats["min"] > 0 and stats["median"] > stats["min"]:
                results.suspects.append(
                    Suspect(
                        stats["median"] / stats["min"],
                        stats["rounds"],
                        stats["min"],
                        stats["median"],
                        row["name"],
                    )
                )
    return results


@dataclass(frozen=True)
class Incomplete:
    """
    A row that left no time: stopped at the limit, out of memory, crashed or failed.

    ``predicted`` marks a row the sweep did not run because the same cell on a smaller mesh
    already settled it (``sweep_plugin.py``); ``detail`` then names that evidence.
    """

    status: str
    detail: str
    limit: float | None
    predicted: bool


def load_incomplete(json_dir: Path | str) -> dict[tuple[CellKey, str], Incomplete]:
    """Read ``incomplete.jsonl``, keyed by ``(cell, library)``; empty when there is none."""
    path = Path(json_dir) / "incomplete.jsonl"
    out: dict[tuple[CellKey, str], Incomplete] = {}
    lines = path.read_text().splitlines() if path.exists() else []
    for line in filter(str.strip, lines):
        entry = json.loads(line)
        library, mesh_name, rest = split_params(entry["params"])
        assert library is not None
        key: CellKey = (entry["module"], entry["group"], mesh_name, rest)
        out[(key, library)] = Incomplete(
            entry["status"], entry["detail"], entry["limit"], bool(entry.get("predicted"))
        )
    return out


def classify(outcome: Mapping[str, Any]) -> tuple[str, str] | None:
    """
    Return ``(status, detail)`` for a sweep outcome that left no time, else ``None``.

    ``status`` is ``timeout``, ``oom``, ``crash`` or ``error``. A process the kernel killed counts
    as out of memory, an abort or a segfault as a crash.
    """
    kind = outcome["outcome"]
    if kind == "timeout":
        return "timeout", f"stopped at {outcome['seconds']:g} s"
    if kind == "predicted":
        return str(outcome["status"]), str(outcome["detail"])
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
    if kind == "nostats":
        return "error", "passed without recording a time"
    return None


def format_duration(seconds: float) -> str:
    """Format one duration in whichever unit reads most naturally for its own magnitude."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"
