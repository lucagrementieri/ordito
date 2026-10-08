"""
Regenerate ``benchmarks/_reference_free_cells.json`` from a full benchmark round's JSON output.

A **reference-free cell** is a ``(group, mesh_name, rest)`` cell in which no reference library
produced a measurement -- every reference was capped off it (``skip_larger_than``, the CPU-bound
cap on the largest meshes) or never had a branch for it -- so the only rows are ordito's own. The
``lucy`` rows of the visibility family are the motivating case: multi-second calls timed ten
rounds each against nothing.

``benchmarks/conftest.py``'s ``bench_case`` / ``bench_lib`` fixtures run a **ordito** row of such
a cell at ``REFERENCE_FREE_ROUNDS`` (3) rounds instead of the default. The row still measures
exactly what it measured -- only the sample count drops -- and its ``min`` stays comparable across
rounds; at three rounds the median is the middle sample, which is what ``aggregate.py --suspect``
exists to flag. No loss-table cell is affected, because a
reference-free cell is not in the loss table.

**Why this cannot be decided at run time.** References are skipped inside each test body, at run
time, and a reference row may run after the ordito row of its cell, so the ordito row has no way
to know whether its cell will end up with a comparison. The table answers it from the previous
round, the same way ``_known_slow_libraries.json`` does. Its policy never skips a cell's fastest
reference, so a cell with no reference row in a default run really had none.

**Staleness.** A cell absent from the table -- a new mesh, a new parameter value, a reference that
gained a branch -- runs the default rounds, so a stale table can only leave a row at ten rounds,
never cut a compared row to three. Regenerate after a full round that adds meshes or references.

Usage
-----
``uv run python benchmarks/regenerate_reference_free_cells.py plans/benchmark-round-27-data/json``
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

ORDITO = {"ordito-cuda", "ordito-cpu"}
OUT_PATH = os.path.join(os.path.dirname(__file__), "_reference_free_cells.json")

# ``(group, mesh_name, rest)``, the cell key every benchmark JSON row maps to.
_Cell = tuple[str, str | None, tuple[tuple[str, str], ...]]


def _load_cells(json_dir: str) -> dict[_Cell, set[str]]:
    cells: dict[_Cell, set[str]] = collections.defaultdict(set)
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
            cells[(b["group"], mesh_name, rest)].add(library)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_dir", help="a round's plans/benchmark-round-N-data/json directory")
    args = parser.parse_args()

    cells = _load_cells(args.json_dir)
    entries = [
        {"group": group, "mesh_name": mesh_name, "rest": list(rest)}
        for (group, mesh_name, rest), libraries in cells.items()
        if libraries and libraries <= ORDITO
    ]
    entries.sort(key=lambda r: (r["group"], r["mesh_name"] or "", r["rest"]))
    with open(OUT_PATH, "w") as f:
        json.dump({"entries": entries}, f, indent=1)
        f.write("\n")
    print(f"{len(entries)} reference-free cells of {len(cells)} written to {OUT_PATH}")


if __name__ == "__main__":
    main()
