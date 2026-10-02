"""
Regenerate ``benchmarks/_known_slow_libraries.json`` from a full benchmark round's JSON output.

The full suite spends most of its wall clock on reference libraries, not on ordito: measured, the
overwhelming majority of pytest-benchmark's own timed regions across the suite is reference-library
rows rather than ordito's own. Most of that buys nothing -- a library that is already orders of
magnitude slower than ordito on one cell does not become more informative by being timed again on
the next mesh size. ``benchmarks/conftest.py``'s
``bench_case`` / ``bench_lib`` fixtures skip a reference-library row outright when this table says
so, the same way ``skip_larger_than`` skips one by hand at a handful of call sites -- except this
table is generated rather than written by hand, because 668 sites is too many to place
individually.

**The policy, and why it is shaped this way.** A reference library is skipped for one exact
``(group, mesh_name, rest, library)`` cell when, in the source round's data:

1. it is **not** one of the two fastest entries overall for that cell (i.e. it is not ordito
   itself and not the single fastest reference library) -- so every benchmarked cell always keeps
   at least one reference comparison, and
2. its own median exceeded ``--ratio`` (default 2.0) times ordito-cuda's median.

This is deliberately **per-library, not per-cell**: a cell with three references where only the
third is a decisive loss keeps the other two. A stronger, per-*cell* policy -- if even the single
fastest reference already loses to ordito by a wide margin, skip every reference for that cell
outright -- was measured to save a comparable amount, precisely because it also catches the
single-reference cells this per-library policy cannot touch, but it is not the default here: it was
still being evaluated, not adopted, when this table was cut. Re-run that analysis before raising
``--ratio`` past this one.

**Staleness.** This table is a snapshot, not a live measurement -- a future optimization could
flip a skipped cell back into a genuinely close race, and this script would not know until someone
reruns it. Regenerate after any full-suite round that ships a real win (a changed cotmatrix
assembly, a new BVH adoption, ...), the same discipline as re-probing a tuning constant after a
Warp upgrade (CLAUDE.md section 9). ``pytest --bench-all-libs`` ignores this table for one run
without needing to delete it -- use that before regenerating, so the new table is cut from a run
that saw every library.

Usage
-----
``uv run python benchmarks/regenerate_known_slow_libraries.py plans/benchmark-round-11-data/json``
    Rewrite ``benchmarks/_known_slow_libraries.json`` from a fresh round's ``--benchmark-json``
    output.
``uv run python benchmarks/regenerate_known_slow_libraries.py <json_dir> --ratio 2.0 --min-rank 3``
    Override the policy (defaults shown).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from typing import Any

ORDITO = {"ordito-cuda", "ordito-cpu"}
OUT_PATH = os.path.join(os.path.dirname(__file__), "_known_slow_libraries.json")

# ``(group, mesh_name, rest)``, the cell key every benchmark JSON row maps to.
_Cell = tuple[str, str | None, tuple[tuple[str, str], ...]]


def _load_cells(json_dir: str) -> dict[_Cell, dict[str, float]]:
    cells: dict[_Cell, dict[str, float]] = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue  # a module still being written, or an empty/corrupt file
        for b in data.get("benchmarks", []):
            params = dict(b["params"] or {})
            library = params.pop("library", None)
            if library is None:
                continue
            mesh_name = params.pop("mesh_name", None)
            rest = tuple(sorted((k, str(v)) for k, v in params.items()))
            cells[(b["group"], mesh_name, rest)][library] = b["stats"]["median"]
    return cells


def _known_slow_entries(
    cells: dict[_Cell, dict[str, float]], ratio: float, min_rank: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (group, mesh_name, rest), libs in cells.items():
        ordito_median = libs.get("ordito-cuda") or libs.get("ordito-cpu")
        refs = {lib: median for lib, median in libs.items() if lib not in ORDITO}
        if ordito_median is None or not refs:
            continue
        ranked = sorted([("ordito", ordito_median), *refs.items()], key=lambda kv: kv[1])
        rank = {lib: position + 1 for position, (lib, _) in enumerate(ranked)}
        for library, median in refs.items():
            if rank[library] >= min_rank and median > ratio * ordito_median:
                records.append(
                    {
                        "group": group,
                        "mesh_name": mesh_name,
                        "rest": list(rest),
                        "library": library,
                        "ratio": round(median / ordito_median, 1),
                    }
                )
    records.sort(key=lambda r: (r["group"], r["mesh_name"] or "", r["library"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_dir", help="a round's plans/benchmark-round-N-data/json directory")
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--min-rank", type=int, default=3)
    args = parser.parse_args()

    cells = _load_cells(args.json_dir)
    entries = _known_slow_entries(cells, args.ratio, args.min_rank)
    out = {"policy": {"ratio": args.ratio, "min_rank": args.min_rank}, "entries": entries}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"{len(entries)} entries written to {OUT_PATH}")


if __name__ == "__main__":
    main()
