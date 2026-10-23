"""
pytest plugin for ``benchmarks/sweep.py``: save each row as it finishes, stop a row at its limit.

Loaded with ``-p benchmarks.sweep_plugin`` by the sweep, never by a plain benchmark run. Two modes,
chosen by environment variables the sweep sets:

``SWEEP_DIR``
    Append each row's pytest-benchmark stats to ``rows.jsonl`` and its outcome (``passed``,
    ``skipped``, ``failed``, ``timeout``) to ``outcomes.jsonl``, flushed line by line, so a killed
    process loses nothing. A row still running after ``SWEEP_TIMEOUT`` seconds, setup included, is
    recorded as ``timeout`` and the process exits at once: a long call into a compiled library
    cannot be interrupted any other way. A failed row also ends the process, after its teardown, so
    a broken CUDA context cannot fail the rows after it. The sweep restarts the module with what is
    left.
``SWEEP_COLLECT``
    Write each collected item's benchmark group and parameters to that file, one JSON object a
    line, and nothing else. ``sweep.py merge`` uses it to name the cells of rows that left no
    stats behind.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Generator
from typing import Any

import pytest

_DIR = os.environ.get("SWEEP_DIR")
_TIMEOUT = float(os.environ.get("SWEEP_TIMEOUT", "600"))
_COLLECT = os.environ.get("SWEEP_COLLECT")
_LOCK = threading.Lock()
_FAILED = pytest.StashKey[bool]()


def _append(name: str, record: dict[str, Any]) -> None:
    """Append one JSON line to a file of the sweep directory, flushed to disk."""
    assert _DIR is not None
    with _LOCK, open(os.path.join(_DIR, name), "a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pytest_collection_finish(session: pytest.Session) -> None:
    """In collect mode, write every item's group and parameters."""
    if _COLLECT is None:
        return
    with open(_COLLECT, "w") as handle:
        for item in session.items:
            marker = item.get_closest_marker("benchmark")
            callspec = getattr(item, "callspec", None)
            record = {
                "nodeid": item.nodeid,
                "group": marker.kwargs.get("group") if marker is not None else None,
                "params": dict(callspec.params) if callspec is not None else {},
            }
            handle.write(json.dumps(record, default=str) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None]:
    """Record the row's start, arm its time limit, and end the process after a failure."""
    del nextitem
    if _DIR is None:
        yield
        return
    _append("started.jsonl", {"nodeid": item.nodeid, "t": time.time()})

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
    if _DIR is None:
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
