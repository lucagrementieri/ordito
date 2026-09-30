"""Regression tests for `benchmarks.aggregate`, the shared loss-table/chart data loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import benchmarks.aggregate as agg


class _Stats(TypedDict):
    median: float
    min: float
    rounds: int


class _Row(TypedDict):
    group: str
    name: str
    params: dict[str, object]
    stats: _Stats


def _write_benchmark_json(path: Path, rows: list[_Row]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}))


def _row(
    group: str, library: str, median: float, mesh_name: str | None = None, **params: object
) -> _Row:
    full_params: dict[str, object] = {
        "library": library,
        **({"mesh_name": mesh_name} if mesh_name else {}),
        **params,
    }
    return {
        "group": group,
        "name": f"test_{group}[{library}]",
        "params": full_params,
        "stats": {"median": median, "min": median, "rounds": 10},
    }


def test_load_groups_by_module_group_mesh_and_rest(tmp_path: Path) -> None:
    """Two rows differing only in `library` land in the same cell; a third `rest` value does not."""
    _write_benchmark_json(
        tmp_path / "test_edges.json",
        [
            _row("faces_to_edges", "triwarp-cuda", 0.001, mesh_name="bunny"),
            _row("faces_to_edges", "trimesh", 0.010, mesh_name="bunny"),
            _row("faces_to_edges", "trimesh", 0.020, mesh_name="dragon"),
        ],
    )

    cells, modules, _suspects = agg.load(str(tmp_path))

    assert modules == {"test_edges": 3}
    bunny_key = ("test_edges", "faces_to_edges", "bunny", ())
    dragon_key = ("test_edges", "faces_to_edges", "dragon", ())
    assert cells[bunny_key] == {"triwarp-cuda": 0.001, "trimesh": 0.010}
    assert cells[dragon_key] == {"trimesh": 0.020}


def test_load_keys_distinct_rest_params_into_separate_cells(tmp_path: Path) -> None:
    """A third parametrize value (beyond mesh_name/library) is part of the cell key."""
    _write_benchmark_json(
        tmp_path / "test_heat.json",
        [
            _row("heat_geodesic", "triwarp-cuda", 0.005, mesh_name="sphere", setup="full"),
            _row("heat_geodesic", "triwarp-cuda", 0.001, mesh_name="sphere", setup="amortized"),
        ],
    )

    cells, _modules, _suspects = agg.load(str(tmp_path))

    assert len(cells) == 2
    full_key = ("test_heat", "heat_geodesic", "sphere", (("setup", "full"),))
    amortized_key = ("test_heat", "heat_geodesic", "sphere", (("setup", "amortized"),))
    assert cells[full_key] == {"triwarp-cuda": 0.005}
    assert cells[amortized_key] == {"triwarp-cuda": 0.001}


def test_load_skips_a_row_with_no_library(tmp_path: Path) -> None:
    """A row with no `library` param (a stray/malformed one) is dropped, not crashed on."""
    row = _row("faces_to_edges", "triwarp-cuda", 0.001)
    del row["params"]["library"]
    _write_benchmark_json(tmp_path / "test_edges.json", [row])

    cells, modules, _suspects = agg.load(str(tmp_path))

    assert modules == {"test_edges": 1}
    assert cells == {}


def test_load_tolerates_an_unreadable_json_file(tmp_path: Path) -> None:
    """A module whose JSON failed to write (a crashed run) is reported and skipped, not fatal."""
    (tmp_path / "test_broken.json").write_text("not json")
    _write_benchmark_json(
        tmp_path / "test_edges.json", [_row("faces_to_edges", "triwarp-cuda", 0.001)]
    )

    _cells, modules, _suspects = agg.load(str(tmp_path))

    assert "test_broken" not in modules
    assert modules == {"test_edges": 1}


def test_load_flags_a_median_well_above_its_own_minimum_as_suspect(tmp_path: Path) -> None:
    """A row whose median/min ratio is large is reported as a `Suspect`, per `report_suspects`."""
    row = _row("marching_triangles", "triwarp-cuda", 9.0)
    row["stats"]["min"] = 3.0
    _write_benchmark_json(tmp_path / "test_levelset.json", [row])

    _cells, _modules, suspects = agg.load(str(tmp_path))

    assert len(suspects) == 1
    assert suspects[0].ratio == 3.0
    assert suspects[0].minimum == 3.0
    assert suspects[0].median == 9.0


def test_compare_picks_the_fastest_reference_and_ignores_excluded_libraries() -> None:
    cells: dict[agg.CellKey, dict[str, float]] = {
        ("test_x", "op", "mesh", ()): {
            "triwarp-cuda": 0.010,
            "triwarp-cpu": 0.050,
            "slow_ref": 0.100,
            "fast_ref": 0.020,
        }
    }

    rows = agg.compare(cells)

    assert len(rows) == 1
    key, triwarp_ms, best_lib, best_ms = rows[0]
    assert key == ("test_x", "op", "mesh", ())
    assert triwarp_ms == 10.0
    assert best_lib == "fast_ref"
    assert best_ms == 20.0

    excluded = agg.compare(cells, exclude=("fast_ref",))
    assert excluded[0][2] == "slow_ref"


def test_compare_skips_a_cell_with_no_triwarp_cuda_row_or_no_reference() -> None:
    cells: dict[agg.CellKey, dict[str, float]] = {
        ("test_x", "op", "a", ()): {"trimesh": 0.010},  # no triwarp-cuda
        ("test_x", "op", "b", ()): {"triwarp-cuda": 0.010},  # no reference
        ("test_x", "op", "c", ()): {"triwarp-cuda": 0.010, "trimesh": 0.020},  # comparable
    }

    rows = agg.compare(cells)

    assert [key for key, *_ in rows] == [("test_x", "op", "c", ())]


def test_cell_label_renders_group_mesh_and_params() -> None:
    assert agg.cell_label(("mod", "heat_geodesic", "sphere_small", (("setup", "full"),))) == (
        "heat_geodesic[sphere_small setup=full]"
    )
    assert agg.cell_label(("mod", "faces_to_edges", "bunny", ())) == "faces_to_edges[bunny]"
    assert agg.cell_label(("mod", "box", None, ())) == "box"
