"""
Regenerate the harness's two generated tables from a full benchmark round.

    uv run pytest benchmarks/ --bench-all-libs ...  # one --benchmark-json per module, in results/
    uv run python -m benchmarks.tools.tables results/

Both tables key cells as ``(group, mesh_name, rest)`` (``results.split_params``) and are read by
``conftest.py``; a missing or unreadable table changes nothing but speed.

``_known_slow_libraries.json``
    The reference rows a default run skips. A reference library is skipped on one cell when it is
    neither ordito nor among the ``--min-rank - 1`` fastest entries of that cell (so every cell
    keeps its fastest reference), and its median exceeds ``--ratio`` times ordito's. A library
    already far slower than ordito says nothing more on the next mesh, and reference rows are most
    of a run's wall clock. The policy is per library, not per cell: a cell with three references
    where only the third is a decisive loss keeps the other two.
``_reference_free_cells.json``
    The cells no reference measures, whose ordito rows a default run times at
    ``REFERENCE_FREE_ROUNDS`` rounds. References skip inside the test body, so the ordito row of a
    cell cannot know at run time whether its cell gets a comparison; the previous round answers. A
    cell missing from the table runs the default rounds, so a stale table can only leave a row at
    full rounds, never cut a compared one.

Cut both from a ``--bench-all-libs`` round, which ignores the known-slow table: a round run under
it would never see the rows it skips. Regenerate after a round that changes a ratio materially (a
new algorithm, a Warp upgrade) or adds meshes, parameters or references.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from benchmarks import results

_HERE = Path(__file__).resolve().parent.parent
KNOWN_SLOW_PATH = _HERE / "_known_slow_libraries.json"
REFERENCE_FREE_PATH = _HERE / "_reference_free_cells.json"

# ``(group, mesh_name, rest)``: a cell as the harness sees it, without the module.
_Cell = tuple[str, str | None, results.Rest]


def harness_cells(cells: results.Cells) -> dict[_Cell, dict[str, float]]:
    """Drop the module from every cell key, the way the harness keys its tables."""
    out: dict[_Cell, dict[str, float]] = collections.defaultdict(dict)
    for (_module, group, mesh_name, rest), libraries in cells.items():
        out[(group, mesh_name, rest)].update(libraries)
    return out


def known_slow_entries(
    cells: dict[_Cell, dict[str, float]], ratio: float, min_rank: int
) -> list[dict[str, Any]]:
    """List every ``(cell, library)`` the known-slow policy skips."""
    records: list[dict[str, Any]] = []
    for (group, mesh_name, rest), libraries in cells.items():
        ordito = libraries.get("ordito-cuda") or libraries.get("ordito-cpu")
        references = {lib: t for lib, t in libraries.items() if results.is_reference(lib)}
        if ordito is None or not references:
            continue
        ranked = sorted([("ordito", ordito), *references.items()], key=lambda item: item[1])
        rank = {lib: position + 1 for position, (lib, _seconds) in enumerate(ranked)}
        records += [
            {
                "group": group,
                "mesh_name": mesh_name,
                "rest": list(rest),
                "library": library,
                "ratio": round(median / ordito, 1),
            }
            for library, median in references.items()
            if rank[library] >= min_rank and median > ratio * ordito
        ]
    records.sort(key=lambda r: (r["group"], r["mesh_name"] or "", r["library"]))
    return records


def reference_free_entries(cells: dict[_Cell, dict[str, float]]) -> list[dict[str, Any]]:
    """List every cell timed only by ordito."""
    entries = [
        {"group": group, "mesh_name": mesh_name, "rest": list(rest)}
        for (group, mesh_name, rest), libraries in cells.items()
        if libraries and not any(map(results.is_reference, libraries))
    ]
    entries.sort(key=lambda r: (r["group"], r["mesh_name"] or "", r["rest"]))
    return entries


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Rewrite both tables from a round's ``--benchmark-json`` directory."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("json_dir", type=Path, help="directory of --benchmark-json files")
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--min-rank", type=int, default=3)
    args = parser.parse_args(argv)

    cells = harness_cells(results.load(args.json_dir).cells)
    slow = known_slow_entries(cells, args.ratio, args.min_rank)
    _write(
        KNOWN_SLOW_PATH,
        {"policy": {"ratio": args.ratio, "min_rank": args.min_rank}, "entries": slow},
    )
    free = reference_free_entries(cells)
    _write(REFERENCE_FREE_PATH, {"entries": free})
    print(f"{len(slow)} known-slow rows, {len(free)} reference-free cells of {len(cells)}")


if __name__ == "__main__":
    main()
