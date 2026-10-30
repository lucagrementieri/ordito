"""
Host-traffic census of the ordito benchmark rows, and the count-budget gate built on it.

Load with ``-p benchmarks.census``. Every ordito row's ``BenchLibrary.run`` is replaced by two
warm calls (the first compiles) and one counted call, and pytest-benchmark is bypassed. What is
counted is what a call costs on the host besides its kernels' device time: launches, allocations,
memsets, copies by direction, readbacks, synchronizations, CUB scans and radix sorts. For a fixed
input and code path these numbers are the same on every run, unlike a timing, which is what makes
them a gate.

The counts are taken at Warp's native entry points (``warp._src.context.runtime.core.wp_*``, and
``warp._src.context.invoke`` for a CPU kernel), below both ``wp.launch`` and ``ordito._launch``'s
cached fast path, so a launch is a launch whichever route issued it. On the CPU device, where there
is no device-to-host copy, a readback is a ``.numpy()`` / ``.list()`` call made from ``ordito``
code. Two devices therefore keep two budgets.

Modes:

* default: compare every counted row with the budget file and fail the session on any count that
  rose, on a row the budget does not know, and (with ``--census-strict``) on a budget row that no
  longer exists. A count that fell is printed so the budget can be tightened in the same change.
* ``--census-update``: rewrite the budget from this run, keeping each row's ``exempt`` reason. The
  run's rows are merged into the budget; with ``--census-strict`` too, the run is the whole budget
  and a row it did not count is dropped.
* ``--census-report=PATH``: also write every row's counts and sequence as JSON lines: each launch
  (kernel, dim, the ``ordito`` line that issued it) and each other counted operation (``<alloc>``,
  ``<readback>``, ``<d2h>``, ...) with its ``ordito`` line -- an exploration aid, not part of the
  gate.

A budget row may carry ``"exempt": "<reason>"`` when its counts depend on convergence (an iterative
solver's round count), which keeps it out of the comparison while still listing it.

Runs under ``pytest-xdist`` (``-n 4 --dist loadfile``, as CI does): every worker counts its own rows
and hands them to the controller in ``workeroutput``, and only the controller compares, reports or
rewrites the budget. ``loadfile`` keeps each file whole and in order inside one process, the same
history a serial run gives it; the counts must not depend on that history anyway (a gate passing
under both is the check that they do not).

The plugin patches Warp's private native table. A Warp upgrade that renames one of the patched
symbols fails the session at startup rather than silently counting zero.
"""

from __future__ import annotations

import collections
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest
import warp as wp

_NATIVE: dict[str, str] = {
    "wp_cuda_launch_kernel": "launch",
    "wp_cpu_launch_kernel": "launch",
    "wp_memcpy_d2d": "d2d",
    "wp_memcpy_h2d": "h2d",
    "wp_memcpy_d2h": "d2h",
    "wp_memcpy_h2h": "h2h",
    "wp_memset_device": "memset",
    "wp_memset_host": "memset",
    "wp_memtile_device": "memset",
    "wp_memtile_host": "memset",
    "wp_alloc_device": "alloc",
    "wp_alloc_device_async": "alloc",
    "wp_alloc_device_default": "alloc",
    "wp_alloc_host": "alloc",
    "wp_alloc_pinned": "alloc",
    "wp_cuda_context_synchronize": "sync",
    "wp_cuda_stream_synchronize": "sync",
    "wp_cuda_event_synchronize": "sync",
    "wp_cuda_graph_launch": "graph_launch",
}
for _suffix in ("device", "host"):
    for _dtype in ("int", "float", "double", "int64"):
        _NATIVE[f"wp_array_scan_{_dtype}_{_suffix}"] = "scan"
    for _dtype in ("int", "uint", "int64", "uint64", "float", "double"):
        _NATIVE[f"wp_radix_sort_pairs_{_dtype}_{_suffix}"] = "sort"

# The counts the gate compares. ``h2h`` and ``alloc`` on the CPU device include Warp's own host
# staging, which is still host work a call pays, so they are gated too.
GATED = (
    "launch",
    "alloc",
    "memset",
    "d2d",
    "h2d",
    "d2h",
    "h2h",
    "readback",
    "sync",
    "scan",
    "sort",
)

# The ``ordito`` package's own directory. A path test on ``/ordito/`` alone would also match every
# file of a checkout named ``ordito`` (CI's ``.../ordito/ordito/.venv``), Warp's internals included.
# Found without importing it: ``pytest_configure`` must patch Warp before ``ordito`` loads.
_SPEC = importlib.util.find_spec("ordito")
assert _SPEC is not None
assert _SPEC.origin is not None
_PACKAGE = os.path.dirname(_SPEC.origin) + os.sep

_STATE: dict[str, Any] = {"armed": False, "depth": 0, "cpu_row": False}
_COUNTS: collections.Counter[str] = collections.Counter()
_SEQUENCE: list[tuple[str, str, str]] = []
_ROWS: dict[str, dict[str, int]] = {}
_REPORT: list[dict[str, Any]] = []


def _ordito_site(depth: int = 2) -> str:
    """Return the first ``ordito`` line outside the launch and device helpers on the stack."""
    frame = inspect.currentframe()
    for _ in range(depth):
        frame = frame.f_back if frame is not None else None
    while frame is not None:
        name = frame.f_code.co_filename
        if name.startswith(_PACKAGE) and not name.endswith(("/_launch.py", "/_device.py")):
            return f"{name.removeprefix(_PACKAGE)}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


def _count_native(core: Any, name: str, key: str) -> None:
    original = getattr(core, name)

    def counted(*args: Any) -> Any:
        if _STATE["armed"]:
            _COUNTS[key] += 1
            if key != "launch":
                _SEQUENCE.append((f"<{key}>", "-", _ordito_site()))
        return original(*args)

    setattr(core, name, counted)


def _count_readbacks() -> None:
    """On a CPU row a readback is a host view of the result, taken from ``ordito`` code."""
    for method in ("numpy", "list"):
        original = getattr(wp.array, method)

        def counted(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            # Only on a CPU row: on CUDA a readback is the ``d2h`` copy, and the host view of the
            # pinned scratch ``_device.read_scalar`` copies into would count it twice.
            if (
                _STATE["armed"]
                and _STATE["cpu_row"]
                and self.device is not None
                and self.device.is_cpu
            ):
                frame = inspect.currentframe()
                caller = frame.f_back if frame is not None else None
                if caller is not None and caller.f_code.co_filename.startswith(_PACKAGE):
                    _COUNTS["readback"] += 1
                    _SEQUENCE.append(("<readback>", "-", _ordito_site()))
            return _original(self, *args, **kwargs)

        setattr(wp.array, method, counted)


def _record_launches(owner: Any, name: str) -> None:
    original = getattr(owner, name)

    def recorded(*args: Any, **kwargs: Any) -> Any:
        if _STATE["armed"] and _STATE["depth"] == 0:
            kernel = args[0] if args else kwargs.get("kernel")
            dim = args[1] if len(args) > 1 else kwargs.get("dim", "-")
            _SEQUENCE.append((str(getattr(kernel, "key", kernel)), str(dim), _ordito_site()))
        _STATE["depth"] += 1
        try:
            return original(*args, **kwargs)
        finally:
            _STATE["depth"] -= 1

    setattr(owner, name, recorded)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the census options."""
    group = parser.getgroup("ordito-census")
    group.addoption("--census-budget", default=None, help="budget JSON (default: per device)")
    group.addoption("--census-update", action="store_true", help="rewrite the budget")
    group.addoption("--census-strict", action="store_true", help="fail on stale budget rows")
    group.addoption("--census-report", default=None, help="write counts and sequences here")


def pytest_configure() -> None:
    """Patch Warp's native table before ``ordito`` caches any entry point from it."""
    wp.init()
    import warp._src.context as context

    core = context.runtime.core
    missing = [name for name in _NATIVE if not hasattr(core, name)]
    if missing:
        raise pytest.UsageError(f"census: Warp no longer exports {missing}; update _NATIVE")
    for name, key in _NATIVE.items():
        _count_native(core, name, key)
    # A CPU kernel is called through its forward hook from these two module functions, never
    # through ``wp_cpu_launch_kernel`` (which only a recorded CPU graph uses).
    for name in ("invoke", "invoke_cpu_blocks"):
        if not hasattr(context, name):
            raise pytest.UsageError(f"census: warp._src.context.{name} is gone; update the plugin")
        _count_native(context, name, "launch")
    _count_readbacks()
    from ordito import (
        _launch,
    )  # after the native patch, so its cached entry points are the counted ones

    for owner, name in ((_launch, "launch"), (_launch, "launch_tiled"), (wp, "launch")):
        _record_launches(owner, name)
    _record_launches(wp, "launch_tiled")


# The ``meshes.AXES`` that grow one kind of input, smallest first. Their larger meshes take the
# smallest one's branches and only run more rounds of them (CG iterations, Cholesky panels,
# pointer-doubling steps), and a regression adds host work per round, which the smallest shows
# too; so the census runs a test on the smallest mesh of each such axis it has, and keeps a larger
# one only where ``censuskeep`` names it because the test branches on size there. The feature axes
# (``loops``, ``quality``, ``components``, ...) are untouched: their meshes are different inputs,
# except a row ``censusskip`` names, whose counts equal a cheaper row's by construction.
_SIZE_AXES = ("scale", "polyline", "tangle", "loops_dense")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect a size axis's larger meshes no ``censuskeep`` keeps, and ``censusskip``'s rows."""
    from benchmarks.meshes import AXES

    ladders = [AXES[name] for name in _SIZE_AXES]
    meshes: collections.defaultdict[tuple[object, ...], set[str]] = collections.defaultdict(set)
    for item in items:
        key = _row_key(item)
        if key is not None:
            meshes[key[0]].add(key[1])
    kept: set[tuple[object, str]] = set()
    for item in items:
        marker = item.get_closest_marker("censuskeep")
        key = _row_key(item)
        if marker is None or key is None:
            continue
        name = marker.args[0] if len(marker.args) == 1 else None
        if name is None or not marker.kwargs.get("reason"):
            raise pytest.UsageError(f"{item.nodeid}: censuskeep takes one mesh and a reason=")
        smaller = [ladder[: ladder.index(name)] for ladder in ladders if name in ladder]
        if name not in meshes[key[0]] or not any(set(s) & meshes[key[0]] for s in smaller):
            # The staleness half: a keep that deselects nothing is a stale one.
            raise pytest.UsageError(
                f"{item.nodeid}: censuskeep({name!r}) names no larger size-axis mesh of this test"
            )
        kept.add((key[0][0], name))
    selected, deselected = [], []
    skips_used: set[str] = set()
    for item in items:
        key = _row_key(item)
        drop = False
        if key is not None and (key[0][0], key[1]) not in kept:
            for ladder in ladders:
                if key[1] in ladder and set(ladder[: ladder.index(key[1])]) & meshes[key[0]]:
                    drop = True
        if key is not None and _census_skips(item, key[1]):
            drop = True
            skips_used.add(str(key[0][0]))
        (deselected if drop else selected).append(item)
    for item in items:
        # The staleness half: a skip that matches no row of its test is a stale one.
        if item.get_closest_marker("censusskip") is not None and (key := _row_key(item)):
            if str(key[0][0]) not in skips_used:
                raise pytest.UsageError(f"{item.nodeid}: censusskip matches no row of this test")
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _census_skips(item: pytest.Item, mesh: str) -> bool:
    """Return whether a ``censusskip`` marker names this row's mesh and its other parameters."""
    params = item.callspec.params  # pyright: ignore[reportAttributeAccessIssue]
    for marker in item.iter_markers("censusskip"):
        reason = marker.kwargs.get("reason")
        if len(marker.args) != 1 or not reason:
            raise pytest.UsageError(f"{item.nodeid}: censusskip takes one mesh and a reason=")
        match = {name: value for name, value in marker.kwargs.items() if name != "reason"}
        if marker.args[0] == mesh and all(params.get(k) == v for k, v in match.items()):
            return True
    return False


def _row_key(item: pytest.Item) -> tuple[tuple[object, ...], str] | None:
    """Return ``((test, other parameters), mesh)`` for a row with a mesh, else ``None``."""
    callspec = getattr(item, "callspec", None)
    if callspec is None or "mesh_name" not in callspec.params:
        return None
    test = (str(item.path), getattr(item, "originalname", item.name))
    others = tuple(sorted((k, repr(v)) for k, v in callspec.params.items() if k != "mesh_name"))
    return (test, *others), str(callspec.params["mesh_name"])


def pytest_collection_finish(session: pytest.Session) -> None:
    """Replace ``BenchLibrary.run`` once the benchmark conftest is loaded."""
    for plugin in session.config.pluginmanager.get_plugins():
        if hasattr(plugin, "BenchLibrary") and hasattr(plugin, "BenchCase"):
            _install(plugin)
            return


def _install(conftest: Any) -> None:
    original_run = conftest.BenchLibrary.run

    def run(self: Any, fn: Any, *, rounds: int = 10, setup: Any = None) -> Any:
        if self.kind != "ordito":
            return original_run(self, fn, rounds=rounds, setup=setup)
        device = self.device

        def call() -> Any:
            return fn(setup()) if setup is not None else fn()

        # ``linalg._cached_solver`` pools a state for an operator whose shape an earlier solve in
        # the process has already seen, and a pooled state pays a ``refresh`` launch per call. Left
        # alone, an earlier row of the same shape decided the count (``solve_spd_settled`` on
        # ``saddle``: 650 launches after ``test_heat.py``, 649 alone), so every row starts with the
        # shapes forgotten, as in a fresh process.
        _forget_solver_shapes()
        # Two warm calls, not one: the first compiles, and some state only reaches its steady form
        # on a later call (``linalg._cached_solver`` gives the first operator of a shape its own
        # state and pools from the second, which records; the third replays). With one warm call
        # the counted call recorded or replayed depending on whether an earlier row had already
        # seen the shape, so the count followed the session's test order.
        for _ in range(2):
            result = call()
        wp.synchronize_device(device)
        arguments = (setup(),) if setup is not None else ()
        wp.synchronize_device(device)
        # Warp pools some host buffers (a ``BsrMatrix``'s pinned ``nnz`` transfer buffer) and hands
        # them back from a ``weakref.finalize``, so an object caught in a reference cycle returns
        # its buffer whenever the cyclic collector happens to run. Collect before arming and keep
        # the collector off during the counted call, or the allocation count follows GC timing.
        gc.collect()
        gc.disable()
        _COUNTS.clear()
        _SEQUENCE.clear()
        _STATE["cpu_row"] = wp.get_device(device).is_cpu
        _STATE["armed"] = True
        try:
            result = fn(*arguments)
        finally:
            _STATE["armed"] = False
            gc.enable()
        wp.synchronize_device(device)
        node = self._benchmark.fullname
        _ROWS[node] = {key: _COUNTS[key] for key in GATED if _COUNTS[key]}
        _REPORT.append({"node": node, "counts": dict(_COUNTS), "sequence": list(_SEQUENCE)})
        return result

    conftest.BenchLibrary.run = run


def _forget_solver_shapes() -> None:
    """Empty ``linalg``'s shape-keyed solver pool, the one cache that outlives its row."""
    from ordito import linalg

    linalg._SOLVER_POOL.clear()  # pyright: ignore[reportPrivateUsage]
    linalg._SHAPES_SEEN.clear()  # pyright: ignore[reportPrivateUsage]


def _budget_path(config: pytest.Config) -> Path:
    explicit = config.getoption("--census-budget")
    if explicit:
        return Path(explicit)
    device = str(config.getoption("--device"))
    name = "cpu" if device == "cpu" or not wp.is_cuda_available() else "cuda"
    return Path(__file__).with_name(f"census_budget_{name}.json")


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    """Under xdist, merge a finished worker's rows into the controller's."""
    del error  # a crashed worker fails the session on its own; its rows are simply missing
    output = getattr(node, "workeroutput", {})
    if "census" in output:
        rows, report = json.loads(output["census"])
        _ROWS.update(rows)
        _REPORT.extend(report)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Compare against, or rewrite, the budget (a worker only hands its rows over)."""
    config = session.config
    workeroutput = getattr(config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput["census"] = json.dumps([_ROWS, _REPORT])
        return
    if not _ROWS:
        return
    report = config.getoption("--census-report")
    if report:
        with open(report, "w") as handle:
            for row in _REPORT:
                handle.write(json.dumps(row) + "\n")
    path = _budget_path(config)
    budget: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    if config.getoption("--census-update"):
        strict = config.getoption("--census-strict")
        merged = {node: entry for node, entry in budget.items() if not strict or node in _ROWS}
        for node, counts in _ROWS.items():
            entry: dict[str, Any] = {"counts": counts}
            if "exempt" in budget.get(node, {}):
                entry["exempt"] = budget[node]["exempt"]
            merged[node] = entry
        path.write_text(json.dumps(dict(sorted(merged.items())), indent=1) + "\n")
        print(f"\ncensus: wrote {len(_ROWS)} rows to {path}")
        return
    rose, fell, unknown = [], [], []
    for node, counts in _ROWS.items():
        if node not in budget:
            unknown.append(node)
            continue
        entry = budget[node]
        if "exempt" in entry:
            continue
        allowed = entry["counts"]
        for key in GATED:
            now, then = counts.get(key, 0), allowed.get(key, 0)
            if now > then:
                rose.append(f"{node}: {key} {then} -> {now}")
            elif now < then:
                fell.append(f"{node}: {key} {then} -> {now}")
    stale = sorted(set(budget) - set(_ROWS)) if config.getoption("--census-strict") else []
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    write = terminal.write_line if terminal is not None else print
    for title, lines in (
        ("counts rose", rose),
        ("rows missing from the budget (run with --census-update)", unknown),
        ("stale budget rows", stale),
        ("counts fell -- tighten the budget with --census-update", fell),
    ):
        if lines:
            write(f"census: {title}:")
            for line in lines:
                write(f"  {line}")
    if rose or unknown or stale:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
