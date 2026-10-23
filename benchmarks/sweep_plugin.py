"""
pytest plugin for ``tools/sweep.py``: save each row as it finishes, stop a row at its limit.

Loaded with ``-p benchmarks.sweep_plugin`` by the sweep, never by a plain benchmark run. Two modes,
chosen by environment variables the sweep sets:

``SWEEP_DIR``
    Append each row's start, with its group and parameters, to ``started.jsonl``; its
    pytest-benchmark stats to ``rows.jsonl``; and its outcome (``passed``, ``skipped``,
    ``failed``, ``nostats``, ``timeout``, ``predicted``) to ``outcomes.jsonl``, flushed line by
    line, so a killed process loses nothing. A row still running after ``SWEEP_TIMEOUT`` seconds,
    setup included, is recorded as ``timeout`` and the process exits at once: a long call into a
    compiled library cannot be interrupted any other way. A failed row also ends the process, after
    its teardown, so a broken CUDA context cannot fail the rows after it. The sweep restarts the
    module with what is left.
``SWEEP_COLLECT``
    Write each collected item's node id, one JSON object a line, and nothing else.

**Predicted rows.** A row on a scan mesh whose cell (module, group, other parameters, library)
already timed out or ran out of memory on a smaller scan mesh in this sweep is recorded as the same
failure, predicted, and not run: a larger input of the same kind does not take less time or less
memory. A crash or an error predicts nothing, since it can come from the input's shape rather than
its size.

Extrapolating a measured median by face count was tried and declined: replayed over the
2026-10-23 sweep it would have skipped measured rows. Linearly from ``happy_buddha``, pyvista's
grazing ``mesh_collision_pairs`` on ``lucy`` was predicted at 99 min a call and took 56 s, since
its cost follows the contact count, not the faces; with an exponent fitted between the two largest
smaller meshes, two more (open3d's ``oriented_bounding_box``, 3 s; pyvista's ``mesh_with_mesh``,
136 s).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from benchmarks import results
from benchmarks.meshes import MESH_ORDER, MESHES_BY_NAME

_DIR = os.environ.get("SWEEP_DIR")
_TIMEOUT = float(os.environ.get("SWEEP_TIMEOUT", "600"))
_COLLECT = os.environ.get("SWEEP_COLLECT")
_LOCK = threading.Lock()
_FAILED = pytest.StashKey[bool]()
_PREDICTED = pytest.StashKey[str]()

# ``(module, group, rest, library)``: a cell's row, without its mesh.
_Series = tuple[str, str, results.Rest, str]
# Per series and scan mesh, the failures that predict the larger meshes.
_FAILURES: dict[_Series, dict[str, str]] = {}


def _append(name: str, record: dict[str, Any]) -> None:
    """Append one JSON line to a file of the sweep directory, flushed to disk."""
    assert _DIR is not None
    with _LOCK, open(os.path.join(_DIR, name), "a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(name: str) -> list[dict[str, Any]]:
    """Read a JSON-lines file of the sweep directory, or nothing."""
    assert _DIR is not None
    path = Path(_DIR) / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _identity(item: pytest.Item) -> tuple[str | None, dict[str, Any]]:
    """Return an item's benchmark group and parameters."""
    marker = item.get_closest_marker("benchmark")
    callspec = getattr(item, "callspec", None)
    group = marker.kwargs.get("group") if marker is not None else None
    return group, dict(callspec.params) if callspec is not None else {}


def _series(module: str, group: object, params: dict[str, Any]) -> tuple[_Series, str] | None:
    """Return a row's series and scan mesh, or ``None`` off the scan ladder."""
    library, mesh_name, rest = results.split_params(params)
    if library is None or group is None or mesh_name not in MESHES_BY_NAME:
        return None
    return (module, str(group), rest, library), mesh_name


def _note_failure(series: _Series, mesh_name: str, status: str) -> None:
    if status in ("timeout", "oom"):
        _FAILURES.setdefault(series, {})[mesh_name] = status


def pytest_configure() -> None:
    """Read the failures this sweep already recorded."""
    if _DIR is None:
        return
    identity = {s["nodeid"]: s for s in _read("started.jsonl") if "params" in s}
    for outcome in _read("outcomes.jsonl"):
        start = identity.get(outcome["nodeid"])
        status = results.classify(outcome)
        if start is None or status is None:
            continue
        found = _series(results.module_of(outcome["nodeid"]), start["group"], start["params"])
        if found is not None:
            _note_failure(*found, status[0])


def predict(series: _Series, mesh_name: str) -> tuple[str, str] | None:
    """Return ``(status, detail)`` for a row the smaller meshes already settle, else ``None``."""
    smaller = MESH_ORDER[: MESH_ORDER.index(mesh_name)]
    failures = _FAILURES.get(series, {})
    for name in reversed(smaller):
        if name in failures:
            what = "timed out" if failures[name] == "timeout" else "ran out of memory"
            return failures[name], f"predicted: {what} on {name}"
    return None


def pytest_collection_finish(session: pytest.Session) -> None:
    """In collect mode, write every item's node id."""
    if _COLLECT is None:
        return
    with open(_COLLECT, "w") as handle:
        for item in session.items:
            handle.write(json.dumps({"nodeid": item.nodeid}) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None]:
    """Record the row's start, predict it or arm its time limit, and exit after a failure."""
    del nextitem
    if _DIR is None:
        yield
        return
    group, params = _identity(item)
    _append(
        "started.jsonl", {"nodeid": item.nodeid, "t": time.time(), "group": group, "params": params}
    )
    found = _series(results.module_of(item.nodeid), group, params)
    prediction = predict(*found) if found is not None else None
    if found is not None and prediction is not None:
        status, detail = prediction
        _note_failure(*found, status)
        item.stash[_PREDICTED] = detail
        record = {"nodeid": item.nodeid, "outcome": "predicted", "status": status}
        _append("outcomes.jsonl", {**record, "detail": detail, "seconds": _TIMEOUT})
        yield
        return

    def expire() -> None:
        _append(
            "outcomes.jsonl", {"nodeid": item.nodeid, "outcome": "timeout", "seconds": _TIMEOUT}
        )
        os._exit(3)

    timer = threading.Timer(_TIMEOUT, expire)
    timer.daemon = True
    timer.start()
    item.stash[_FAILED] = False
    yield
    timer.cancel()
    if item.stash[_FAILED]:
        os._exit(4)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a predicted row before its fixtures build anything."""
    if _PREDICTED in item.stash:
        pytest.skip(item.stash[_PREDICTED])


def _skip_reason(report: pytest.TestReport) -> str:
    """Return a skip's reason, or ``""``."""
    return str(report.longrepr[-1]) if isinstance(report.longrepr, tuple) else ""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any]:
    """Record each row's outcome, and its stats when it passed."""
    del call
    outcome = yield
    if _DIR is None or _PREDICTED in item.stash:
        return
    report: pytest.TestReport = outcome.get_result()
    if report.skipped and report.when in ("setup", "call"):
        _append(
            "outcomes.jsonl",
            {"nodeid": item.nodeid, "outcome": "skipped", "reason": _skip_reason(report)},
        )
    elif report.failed and report.when in ("setup", "call"):
        item.stash[_FAILED] = True
        _append(
            "outcomes.jsonl",
            {
                "nodeid": item.nodeid,
                "outcome": "failed",
                "when": report.when,
                "error": str(report.longrepr)[-1500:],
            },
        )
    elif report.when == "call":
        fixture = getattr(item, "funcargs", {}).get("benchmark")
        stats = getattr(fixture, "stats", None)
        if stats is None:
            _append("outcomes.jsonl", {"nodeid": item.nodeid, "outcome": "nostats"})
            return
        _append("rows.jsonl", stats.as_dict(include_data=True))
        _append(
            "outcomes.jsonl",
            {"nodeid": item.nodeid, "outcome": "passed", "seconds": report.duration},
        )
