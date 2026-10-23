"""
Print a directory of ``--benchmark-json`` files as a loss table: where ordito loses, by how much.

    uv run python -m benchmarks.tools.aggregate <json_dir>

Each cell (``results.CellKey``) compares ordito-cuda's median with the *minimum* median over the
reference libraries timed on it.

**Read the suspect report first.** The median is the comparison statistic, but it does not certify
itself: a slow group runs as few as 3 rounds, and at 3 the median *is* the middle sample, so one
one-off (a Warp module load, a scheduler hiccup) lands on it. ``--suspect`` prints every row whose
median exceeds its own minimum by more than a factor; check such a row's minimum against a previous
run before calling it a regression. It flags reference rows too, and those tilt the table the other
way: an inflated reference median makes ordito look better than it is, worst for the references
whose cost is Python object churn.
"""

from __future__ import annotations

import argparse
import collections

from benchmarks import results
from benchmarks.results import CellKey, Cells, Suspect


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


def compare(cells: Cells, exclude: tuple[str, ...] = ()) -> list[tuple[CellKey, float, str, float]]:
    """Yield `(key, ordito_ms, best_reference_lib, best_reference_ms)` for each comparable cell."""
    out = []
    for key, libs in cells.items():
        ordito_seconds = libs.get("ordito-cuda")
        if ordito_seconds is None:
            continue
        refs = {k: v for k, v in libs.items() if results.is_reference(k) and k not in exclude}
        if not refs:
            continue
        best_lib = min(refs, key=refs.__getitem__)
        out.append((key, ordito_seconds * 1e3, best_lib, refs[best_lib] * 1e3))
    return out


def report(
    cells: Cells, exclude: tuple[str, ...], label: str, top: int
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

    loaded = results.load(args.json_dir)
    cells, modules, suspects = loaded.cells, loaded.modules, loaded.suspects
    print(f"{len(modules)} modules, {sum(modules.values())} rows, {len(cells)} cells")
    report_suspects(suspects, args.suspect)
    all_libs = collections.Counter(lib for v in cells.values() for lib in v)
    print("libraries:", dict(all_libs.most_common()))

    _rows, losses, _gap = report(cells, (), "all references", args.top)
    by_group(losses, args.top)
    report(cells, ("meshlib",), "meshlib excluded", 25)


if __name__ == "__main__":
    main()
