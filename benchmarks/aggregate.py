"""
Aggregate per-module ``pytest-benchmark`` JSONs into a loss table (and feed `plot_comparison.py`).

Cell = ``(module, group, mesh_name, every other param except library)``. ordito-cuda's median
against the *minimum* median over the non-ordito libraries in the cell.

The median is the comparison statistic, but it is not self-certifying: pytest-benchmark's
``rounds`` drops to 3 for the slowest groups, and at n=3 the median *is* the middle sample, so a
single one-off (a Warp module load, a scheduler hiccup) lands on it and cannot be averaged out.
One round's only apparent regression was exactly that: the *minimum* had not moved at all while two
of three samples carried a one-off cost. A whole bisect was planned against it.

So ``--suspect`` prints every cell whose median exceeds its own minimum by more than a factor,
and it should be read *before* the loss table. On the ordito side it is a cheap guard rather than
a common problem: over a whole round's ordito cells only a couple exceed the threshold.

It flags *reference* cells too, and those tilt the table the other way -- an inflated reference
median makes ordito look better than it is, and the worst offenders are the references whose cost
is dominated by Python object churn. Before reading a *win* against one of those as real, check the
reference's own min.

This module is the single source of truth for that grouping logic, and both consumers
share it: this file's own CLI keeps the plain-text loss table
(``uv run python -m benchmarks.aggregate <json_dir>``), and ``benchmarks/plot_comparison.py``
imports ``load()`` directly rather than re-parsing JSON.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import NamedTuple

ORDITO_IDS = frozenset({"ordito-cpu", "ordito-cuda"})

# (module, group, mesh_name, rest) -- `rest` is every other parametrize value for this exact case,
# as a sorted tuple of (name, str(value)) pairs so it hashes and orders consistently.
CellKey = tuple[str, str, "str | None", tuple[tuple[str, str], ...]]


class Suspect(NamedTuple):
    """One benchmark row whose median sits well above its own minimum -- read `report_suspects`."""

    ratio: float
    rounds: int
    minimum: float
    median: float
    name: str


def load(json_dir: str) -> tuple[dict[CellKey, dict[str, float]], dict[str, int], list[Suspect]]:
    """
    Parse every ``*.json`` under `json_dir` into `(cells, modules, suspects)`.

    Parameters
    ----------
    json_dir
        Directory of ``--benchmark-json`` output files, one per test module (the shape
        `uv run pytest benchmarks/ --benchmark-json=...` produces when pointed at a directory of
        per-module runs, or a single combined file's directory).

    Returns
    -------
    cells
        `key -> {library_id: median_seconds}`, one entry per `(module, group, mesh, rest)` cell.
    modules
        `module_name -> row_count`, for a quick sanity readout of what was actually parsed.
    suspects
        Every row's `(median / min, rounds, min, median, name)`, unfiltered -- see
        `report_suspects` for the threshold this is meant to be read through.
    """
    cells: dict[CellKey, dict[str, float]] = collections.defaultdict(dict)
    modules: dict[str, int] = {}
    suspects: list[Suspect] = []
    for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        mod = os.path.basename(path)[:-5]
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:  # a module still being written, or a crash
            print(f"  ! {mod}: {exc}", file=sys.stderr)
            continue
        rows = data["benchmarks"]
        modules[mod] = len(rows)
        for b in rows:
            params = dict(b["params"] or {})
            library = params.pop("library", None)
            if library is None:
                continue
            mesh_name = params.pop("mesh_name", None)
            rest = tuple(sorted((k, str(v)) for k, v in params.items()))
            key: CellKey = (mod, b["group"], mesh_name, rest)
            stats = b["stats"]
            cells[key][library] = stats["median"]
            if stats["min"] > 0 and stats["median"] / stats["min"] > 1.0:
                suspects.append(
                    Suspect(
                        stats["median"] / stats["min"],
                        stats["rounds"],
                        stats["min"],
                        stats["median"],
                        b["name"],
                    )
                )
    return cells, modules, suspects


def cell_label(key: CellKey) -> str:
    """Render a `CellKey` as the `group[mesh param=value ...]` string used in every report."""
    _, group, mesh_name, rest = key
    params = " ".join(f"{k}={v}" for k, v in rest)
    inner = " ".join(part for part in (mesh_name, params) if part)
    return f"{group}[{inner}]" if inner else group


def report_suspects(suspects: list[Suspect], factor: float = 1.5) -> None:
    """Print cells whose median is inflated over their own minimum -- read this first."""
    bad = sorted((s for s in suspects if s.ratio > factor), reverse=True)
    print(f"\n=== cells whose median exceeds their min by more than {factor}x ===")
    if not bad:
        print("  none -- every median sits within that factor of its own floor")
        return
    for s in bad:
        print(
            f"  med/min={s.ratio:5.2f}  rounds={s.rounds:3d}  min={s.minimum * 1e3:9.3f} ms  "
            f"med={s.median * 1e3:9.3f} ms  {s.name}"
        )
    print(
        f"  {len(bad)} cell(s). At rounds=3 the median IS the middle sample, so one hiccup owns "
        "it;\n  check the min against a previous run before calling any of these a regression."
    )


def compare(
    cells: dict[CellKey, dict[str, float]], exclude: tuple[str, ...] = ()
) -> list[tuple[CellKey, float, str, float]]:
    """Yield `(key, ordito_ms, best_reference_lib, best_reference_ms)` for each comparable cell."""
    out = []
    for key, libs in cells.items():
        ordito_seconds = libs.get("ordito-cuda")
        if ordito_seconds is None:
            continue
        refs = {k: v for k, v in libs.items() if k not in ORDITO_IDS and k not in exclude}
        if not refs:
            continue
        best_lib = min(refs, key=refs.__getitem__)
        out.append((key, ordito_seconds * 1e3, best_lib, refs[best_lib] * 1e3))
    return out


def report(
    cells: dict[CellKey, dict[str, float]], exclude: tuple[str, ...], label: str, top: int
) -> tuple[list[tuple[CellKey, float, str, float]], list[tuple[CellKey, float, str, float]], float]:
    """Print a loss table for `label` and return `(rows, losses, total_gap_ms)`."""
    rows = compare(cells, exclude)
    losses = [(k, t, bl, b) for k, t, bl, b in rows if t > b]
    gap = sum(t - b for _, t, _, b in losses)
    wins = len(rows) - len(losses)
    print(f"\n=== {label} ===")
    print(f"comparisons {len(rows)}   wins {wins}   losses {len(losses)}   gap {gap:.1f} ms")
    losses.sort(key=lambda r: r[1] - r[3], reverse=True)
    print(f"\n{'gap ms':>9} {'ratio':>7} {'ordito':>10} {'best':>10}  cell")
    for key, t, bl, b in losses[:top]:
        print(f"{t - b:9.2f} {t / b:6.2f}x {t:10.3f} {b:10.3f}  {cell_label(key)} vs {bl}")
    return rows, losses, gap


def by_group(
    losses: list[tuple[CellKey, float, str, float]], top: int
) -> dict[tuple[str, str], list[float | int]]:
    """Aggregate `losses` (from `report`) by `(module, group)` and print the worst offenders."""
    agg: dict[tuple[str, str], list[float | int]] = collections.defaultdict(lambda: [0.0, 0, 0.0])
    for (mod, group, _, _), t, _, b in losses:
        a = agg[(mod, group)]
        a[0] += t - b
        a[1] += 1
        a[2] = max(a[2], t / b)
    print(f"\n{'gap ms':>9} {'rows':>5} {'worst':>7}  group")
    for (mod, group), (g, n, r) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{g:9.2f} {n:5d} {r:6.2f}x  {group}  ({mod})")
    return agg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_dir")
    ap.add_argument("--top", type=int, default=45)
    ap.add_argument(
        "--suspect",
        type=float,
        default=1.5,
        help="flag cells whose median exceeds their own min by more than this factor",
    )
    args = ap.parse_args(argv)

    cells, modules, suspects = load(args.json_dir)
    print(f"{len(modules)} modules, {sum(modules.values())} rows, {len(cells)} cells")
    report_suspects(suspects, args.suspect)
    all_libs = collections.Counter(lib for v in cells.values() for lib in v)
    print("libraries:", dict(all_libs.most_common()))

    _rows, losses, _gap = report(cells, (), "all references", args.top)
    by_group(losses, args.top)
    report(cells, ("meshlib",), "meshlib excluded", 25)


if __name__ == "__main__":
    main()
