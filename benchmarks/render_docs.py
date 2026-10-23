"""
Render the benchmark documentation from a directory of ``--benchmark-json`` files.

    uv run python benchmarks/render_docs.py <json_dir>

Writes, from the cells `aggregate.load` reads:

- ``docs/benchmarks/<module>.md``, generated whole, for every module with at least one function a
  reference library also computes: a summary chart of the module, then one chart per such function
  and a collapsed table of every input it was measured on. A function no reference computes gets no
  chart; the page lists those in one table at the end.
- The ``Benchmarks`` entries of ``docs/SUMMARY.md``, nesting the module pages under the same
  shelves as the API reference.
- The generated blocks of ``docs/benchmarks/index.md``, each between ``<!-- bench:NAME -->`` and
  ``<!-- /bench:NAME -->``. The prose around them is hand-written and left alone; a marker names a
  block kind (``provenance``, ``summary``, ``modules``) or one highlighted function
  (``cell test_proximity/signed_distance_on_mesh``).

**Every chart is a ratio to ordito.** ordito (CUDA) is the bar at 1x, in the brand accent; each
library's bar, in muted gray, is its time divided by ordito's, so a bar longer than ordito's is a
library that took longer. Bars are sorted fastest first, ordito wherever its time puts it. The axis
is logarithmic, the usual scale for ratios, so 10x slower and 10x faster sit at the same distance
from 1x. A function's chart shows the input with the most libraries measured on it (then the
largest); the table below it has the rest.

**Rows that left no time.** ``sweep.py merge`` writes ``incomplete.jsonl`` beside the JSON files:
one ``{"module", "group", "params", "status", "detail", "limit"}`` object per row it stopped at its
time limit (``timeout``), or that crashed, ran out of memory or failed (``crash``, ``oom``,
``error``). A timeout is drawn at the limit, paler, labelled ``> 10 min``: the library took at
least that long, and the averages count it at the limit, so a library with such runs is at least
as slow as its bar says. A row with no time at all gets no bar: it is listed under the timed bars
with a red cross and what happened, and the averages leave it out.

**The summary charts.** A library's bar is the geometric mean, over the functions both implement, of
its per-function ratio, itself the geometric mean over the inputs both were measured on. Averaging
per function first keeps a function timed on six meshes from counting six times. The main page's
per-module chart uses, on each input, the fastest library measured on it.

Charts are inline SVG: a few kilobytes each, recoloured by the site's light/dark switch through CSS
classes (``docs/stylesheets/extra.css``), with a native hover tooltip per bar. The accent and gray
are ``plot_comparison.py``'s validated pair; every bar carries its value as a direct label.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import aggregate
from assets.logos import registry
from meshes import ALL_MESHES_BY_NAME

SUBJECT = "ordito-cuda"
_REPO = Path(__file__).resolve().parent.parent
_DOCS = _REPO / "docs"
_PAGES = _DOCS / "benchmarks"
_MAIN_PAGE = _PAGES / "index.md"
_LOGO_DIR = _DOCS / "assets" / "logos"
# Relative to the markdown sources in ``docs/benchmarks/``; the site build rewrites it per page.
_LOGO_HREF = "../assets/logos/"
_NAV = _DOCS / "SUMMARY.md"

# SVG geometry, in viewBox units; the stylesheet caps the chart at this width, so units are pixels.
_WIDTH = 640
_LABEL_WIDTH = 196
_VALUE_WIDTH = 132
_ROW = 20
_BAR = 10
_TOP = 4
_AXIS = 18
# Every logo is fitted, at its own aspect ratio, into one square box this size and centred in it.
_LOGO_WIDTH = 18
_LOGO_HEIGHT = 18
_CROSS = 4  # half the width of a failed row's cross

# Appended to a summary caption: what a run stopped at the time limit contributes.
_LIMIT_NOTE = " A run stopped at the time limit counts as the limit."

_MARKER = re.compile(
    r"(<!-- bench:(?P<name>[^>]+?) -->\n)(?P<body>.*?)(<!-- /bench:(?P=name) -->)", re.S
)


@dataclass(frozen=True)
class Bar:
    """One row of a chart: a label, its ratio, its library (for colour and logo) and its texts."""

    label: str
    value: float
    library: str | None
    tooltip: str
    text: str
    timed_out: bool = False
    failed: bool = False


# Rows that left no time, filled by ``load``: ``(cell, library)`` stopped at the time limit, and
# ``(cell, library) -> (status, detail)`` for the ones that crashed, ran out of memory or failed.
_TIMED_OUT: set[tuple[aggregate.CellKey, str]] = set()
_FAILED: dict[tuple[aggregate.CellKey, str], tuple[str, str]] = {}


def load(json_dir: Path) -> dict[aggregate.CellKey, dict[str, float]]:
    """Read the cells, adding each timed-out row at its limit and noting each failed one."""
    cells, _modules, _suspects = aggregate.load(str(json_dir))
    incomplete = json_dir / "incomplete.jsonl"
    lines = incomplete.read_text().splitlines() if incomplete.exists() else []
    for line in filter(str.strip, lines):
        entry = json.loads(line)
        params = dict(entry["params"])
        library = params.pop("library")
        mesh_name = params.pop("mesh_name", None)
        rest = tuple(sorted((k, str(v)) for k, v in params.items()))
        key: aggregate.CellKey = (entry["module"], entry["group"], mesh_name, rest)
        if library in cells.get(key, {}):
            continue  # measured after all, by another run
        if entry["status"] == "timeout":
            cells[key][library] = float(entry["limit"])
            _TIMED_OUT.add((key, library))
        else:
            _FAILED[(key, library)] = (entry["status"], entry["detail"])
    return cells


def failures(key: aggregate.CellKey) -> list[tuple[str, str]]:
    """Return ``(library, what happened)`` for each library of a cell that left no time."""
    return sorted(
        (library, describe_failure(*status))
        for (failed_key, library), status in _FAILED.items()
        if failed_key == key
    )


def describe_failure(status: str, detail: str) -> str:
    """Say in a few words what happened to a row that left no time."""
    if status == "oom":
        return "out of memory"
    if status == "crash":
        return f"crashed ({detail})"
    return "failed"


def timed_out(key: aggregate.CellKey, library: str) -> bool:
    """Whether a cell's library was stopped at the time limit instead of measured."""
    return (key, library) in _TIMED_OUT


def format_limit(seconds: float) -> str:
    """Format a time limit, ``> 10 min``."""
    return f"> {seconds / 60:g} min" if seconds >= 60 else f"> {format_duration(seconds)}"


def format_duration(seconds: float) -> str:
    """Format one duration in whichever unit reads most naturally for its own magnitude."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def format_ratio(ratio: float) -> str:
    """Format a ratio as ``12x`` / ``3.4x`` / ``0.62x``."""
    if ratio >= 10:
        return f"{ratio:.0f}x"
    if ratio >= 1:
        return f"{ratio:.1f}x".replace(".0x", "x")
    return f"{ratio:.2f}x"


def svg_chart(bars: list[Bar], *, label: str, subject_fill: bool = False) -> str:
    """
    Render ratio bars as one inline SVG, on a log axis, every bar starting at its left edge.

    A bar is ordito's (accent) when its library is ordito; every other bar is a reference's
    (gray), unless ``subject_fill`` paints them all in the accent (a chart of ordito's own
    speedups, such as the per-module one).
    """
    values = [bar.value for bar in bars if not bar.failed]
    # One decade of room below the smallest value, so the shortest bar is never a sliver.
    e_lo = math.floor(math.log10(min(values))) - (1 if min(values) <= 1 else 0)
    e_hi = max(math.ceil(math.log10(max(values))), e_lo + 1)
    left, right = _LABEL_WIDTH, _WIDTH - _VALUE_WIDTH

    def scale(value: float) -> float:
        return left + (math.log10(value) - e_lo) / (e_hi - e_lo) * (right - left)

    axis_y = _TOP + len(bars) * _ROW
    out = [
        f'<svg class="bench-chart" viewBox="0 0 {_WIDTH} {axis_y + _AXIS}" role="img" '
        f'aria-label="{html.escape(label)}">'
    ]
    for exponent in range(e_lo, e_hi + 1):
        x = scale(10.0**exponent)
        cls = "bench-baseline" if exponent == 0 else "bench-grid"
        out.append(f'<line class="{cls}" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{axis_y}"/>')
        out.append(
            f'<text class="bench-tick" x="{x:.1f}" y="{axis_y + 13}" text-anchor="middle">'
            f"{10.0**exponent:g}x</text>"
        )
    logos = [registry.clear_logo_path(bar.library) if bar.library else None for bar in bars]
    # Reserve the logo column whenever one row has a logo, so every label ends at the same x.
    text_x = left - 6 - (_LOGO_WIDTH + 6 if any(logos) else 0)
    for i, (bar, logo) in enumerate(zip(bars, logos, strict=True)):
        y = _TOP + i * _ROW + (_ROW - _BAR) / 2
        mid = y + _BAR / 2
        subject = bar.library == SUBJECT
        cls = "bench-subject" if subject or subject_fill else "bench-reference"
        if bar.timed_out:
            cls += " bench-timeout"
        out.append(f"<g><title>{html.escape(bar.tooltip)}</title>")
        if bar.failed:
            # No time, so no bar: a red cross where the bar would start, and what happened.
            cx = left + _CROSS + 2
            out.append(
                f'<path class="bench-failed" d="M{cx - _CROSS},{mid - _CROSS:.1f}'
                f"L{cx + _CROSS},{mid + _CROSS:.1f}M{cx + _CROSS},{mid - _CROSS:.1f}"
                f'L{cx - _CROSS},{mid + _CROSS:.1f}"/>'
            )
            end = cx + _CROSS + 1
        else:
            end = scale(bar.value)
            out.append(
                f'<rect class="{cls}" x="{left}" y="{y:.1f}" width="{max(end - left, 1):.1f}" '
                f'height="{_BAR}"/>'
            )
        if logo is not None:
            width, height = _logo_size(logo)
            x = left - 6 - (_LOGO_WIDTH + width) / 2
            out.append(
                f'<image class="bench-logo bench-logo-{logo.stem}" href="{_LOGO_HREF}{logo.name}" '
                f'x="{x:.1f}" y="{mid - height / 2:.1f}" width="{width:.1f}" '
                f'height="{height:.1f}"/>'
            )
        weight = ' font-weight="600"' if subject else ""
        out.append(
            f'<text class="bench-label" x="{text_x}" y="{mid + 4:.1f}" text-anchor="end"{weight}>'
            f"{html.escape(bar.label)}</text>"
        )
        out.append(
            f'<text class="bench-value" x="{end + 5:.1f}" y="{mid + 4:.1f}">'
            f"{html.escape(bar.text)}</text></g>"
        )
    out.append("</svg>")
    return "".join(out)


def _logo_size(path: Path) -> tuple[float, float]:
    """Return a logo's drawn size, fitted into the logo box at its aspect ratio (PNG header)."""
    with open(path, "rb") as handle:
        width, height = struct.unpack(">II", handle.read(24)[16:24])
    scale = min(_LOGO_WIDTH / width, _LOGO_HEIGHT / height)
    return width * scale, height * scale


# --- data -----------------------------------------------------------------------------------------

Cells = dict[aggregate.CellKey, dict[str, float]]
GroupKey = tuple[str, str]


def is_reference(library: str) -> bool:
    """Whether a library id is a reference rather than ordito on some device."""
    return not registry.is_subject(library)


def compared(cells: Cells, key: aggregate.CellKey) -> bool:
    """Whether a cell has ordito-cuda and a reference timed, or tried and failed, beside it."""
    row = cells[key]
    return SUBJECT in row and (any(map(is_reference, row)) or bool(failures(key)))


def by_group(cells: Cells) -> dict[GroupKey, list[aggregate.CellKey]]:
    """Group cell keys by ``(module, group)``, in the order the JSON lists them."""
    groups: dict[GroupKey, list[aggregate.CellKey]] = collections.defaultdict(list)
    for key in cells:
        groups[(key[0], key[1])].append(key)
    return groups


def _size(key: aggregate.CellKey) -> int:
    """Return a cell's mesh size in faces, or 0 for a mesh-free cell."""
    spec = ALL_MESHES_BY_NAME.get(key[2] or "")
    return int(spec["n_faces"]) if spec is not None else 0


def representative(cells: Cells, keys: list[aggregate.CellKey]) -> aggregate.CellKey | None:
    """Pick the group's widest comparison: most references, then the largest mesh."""
    candidates = [key for key in keys if compared(cells, key)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda key: (
            sum(map(is_reference, cells[key])) + len(failures(key)),
            _size(key),
            -keys.index(key),
        ),
    )


def cell_caption(key: aggregate.CellKey) -> str:
    """Name a cell's input: the mesh with its face count, then the other parameters."""
    parts: list[str] = []
    spec = ALL_MESHES_BY_NAME.get(key[2] or "")
    if key[2] is not None:
        parts.append(f"{key[2]} ({int(spec['n_faces']):,} faces)" if spec else key[2])
    parts.extend(f"{name}={value}" for name, value in key[3])
    return " · ".join(parts).replace(",", "\N{THIN SPACE}")


def cell_name(key: aggregate.CellKey) -> str:
    """Name a cell compactly for a table row."""
    parts = [key[2]] if key[2] is not None else []
    parts.extend(f"{name}={value}" for name, value in key[3])
    return " ".join(parts) or "—"


def geomean(values: list[float]) -> float:
    """Geometric mean of positive values."""
    return math.exp(sum(map(math.log, values)) / len(values))


def library_ratios(cells: Cells, groups: list[list[aggregate.CellKey]]) -> dict[str, list[float]]:
    """Per reference library, its per-function geometric-mean time ratio to ordito-cuda."""
    out: dict[str, list[float]] = collections.defaultdict(list)
    for keys in groups:
        ratios: dict[str, list[float]] = collections.defaultdict(list)
        for key in keys:
            row = cells[key]
            if SUBJECT not in row:
                continue
            for library, seconds in row.items():
                if is_reference(library):
                    ratios[library].append(seconds / row[SUBJECT])
        for library, values in ratios.items():
            out[library].append(geomean(values))
    return out


def fastest_ratios(cells: Cells, keys: list[aggregate.CellKey]) -> list[float]:
    """Per input of one function, the fastest reference's time over ordito-cuda's."""
    return [
        min(t for lib, t in cells[key].items() if is_reference(lib)) / cells[key][SUBJECT]
        for key in keys
        if SUBJECT in cells[key] and any(map(is_reference, cells[key]))
    ]


def api_shelves() -> list[tuple[str, list[str]]]:
    """Read the API reference's shelves, ``(title, modules)``, in nav order."""
    shelves: list[tuple[str, list[str]]] = []
    in_api = False
    for line in _NAV.read_text().splitlines():
        if line.startswith("* "):
            in_api = "(reference.md)" in line
        elif in_api and re.match(r"^ {4}\* [^\[]", line):
            shelves.append((line.strip()[2:], []))
        elif in_api and shelves and (match := re.search(r"\(api/ordito/(\w+)\.md\)", line)):
            shelves[-1][1].append(match.group(1))
    return shelves


# --- blocks ---------------------------------------------------------------------------------------


def ratio_bars(per_library: dict[str, list[float]], noun: str) -> list[Bar]:
    """Turn per-library ratio lists into ordito's 1x bar and one gray bar per library."""
    rows = sorted((geomean(v), lib, v) for lib, v in per_library.items())
    bars = [Bar("ordito (CUDA)", 1.0, SUBJECT, "ordito (CUDA): the reference, 1x", "1x")]
    for ratio, lib, values in rows:
        name = registry.display_label(lib)
        slower = sum(x > 1 for x in values)
        bars.append(
            Bar(
                name,
                ratio,
                lib,
                f"{name}: {format_ratio(ratio)} ordito's time, geometric mean over {len(values)} "
                f"{noun}; slower than ordito in {slower}",
                f"{format_ratio(ratio)}  ({len(values)})",
            )
        )
    return sorted(bars, key=lambda bar: bar.value)


def ratio_table(per_library: dict[str, list[float]], noun: str) -> list[str]:
    """Tabulate a summary chart's values, one row per library."""
    rows = sorted((geomean(v), lib, v) for lib, v in per_library.items())
    lines = [
        f"| Library | Time relative to ordito | {noun.capitalize()} compared | ordito faster in |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {registry.display_label(lib)} | {format_ratio(g)} | {len(v)} | "
        f"{sum(x > 1 for x in v)} |"
        for g, lib, v in rows
    ]
    return lines


def summary_block(cells: Cells) -> str:
    """Render the main page's per-library chart over every function, with its table."""
    per = library_ratios(cells, list(by_group(cells).values()))
    chart = svg_chart(ratio_bars(per, "functions"), label="Time of each library relative to ordito")
    return _figure(
        chart,
        "Time relative to ordito (CUDA), geometric mean over every function both implement; in "
        "parentheses, the number of functions. Log scale." + (_LIMIT_NOTE if _TIMED_OUT else ""),
        ratio_table(per, "functions"),
    )


def modules_block(cells: Cells, pages: set[str]) -> str:
    """Render the main page's per-module chart against the fastest reference, with its table."""
    per: dict[str, list[float]] = collections.defaultdict(list)
    for (module, _group), keys in by_group(cells).items():
        ratios = fastest_ratios(cells, keys)
        if ratios and module.removeprefix("test_") in pages:
            per[module.removeprefix("test_")].append(geomean(ratios))
    rows = sorted(((geomean(v), m, v) for m, v in per.items()), reverse=True)
    bars = [
        Bar(
            m,
            g,
            None,
            f"{m}: ordito {format_ratio(g)} faster than the fastest library, geometric mean over "
            f"{len(v)} functions; faster in {sum(x > 1 for x in v)}",
            f"{format_ratio(g)}  ({len(v)})",
        )
        for g, m, v in rows
    ]
    chart = svg_chart(bars, label="ordito's speedup per module", subject_fill=True)
    table = ["| Module | ordito's speedup | Functions | ordito faster in |", "|---|---|---|---|"]
    table += [
        f"| [`{m}`]({m}.md) | {format_ratio(g)} | {len(v)} | {sum(x > 1 for x in v)} |"
        for g, m, v in rows
    ]
    return _figure(
        chart,
        "ordito (CUDA)'s speedup over the fastest library measured on each input, geometric mean "
        "per module; in parentheses, the number of functions. Log scale.",
        table,
    )


def cell_figure(cells: Cells, key: aggregate.CellKey, *, link: str | None = None) -> str:
    """Render one input of one function: every library's time relative to ordito's, sorted."""
    row = cells[key]
    base = row[SUBJECT]
    entries = sorted((t, lib) for lib, t in row.items() if lib == SUBJECT or is_reference(lib))
    bars: list[Bar] = []
    table = ["| Library | Median | Relative to ordito |", "|---|---|---|"]
    for seconds, lib in entries:
        name = registry.display_label(lib)
        ratio = seconds / base
        if timed_out(key, lib):
            time_text, ratio_text = format_limit(seconds), f"> {format_ratio(ratio)}"
            tooltip = f"{name}: stopped at the {format_limit(seconds)[2:]} limit"
        else:
            time_text, ratio_text = format_duration(seconds), format_ratio(ratio)
            tooltip = f"{name}: {time_text}, {ratio_text} ordito's time"
        bars.append(
            Bar(name, ratio, lib, tooltip, f"{ratio_text} · {time_text}", timed_out(key, lib))
        )
        table.append(f"| {name} | {time_text} | {ratio_text} |")
    for lib, what in failures(key):
        name = registry.display_label(lib)
        bars.append(Bar(name, 1.0, lib, f"{name}: {what}", what, failed=True))
        table.append(f"| {name} | {what} | |")
    where = cell_caption(key)
    chart = svg_chart(bars, label=f"{key[1]} on {where}" if where else key[1])
    caption = " · ".join(part for part in (where, "time relative to ordito, log scale") if part)
    if any(bar.timed_out for bar in bars):
        caption += " · a paler bar was stopped at the time limit"
    if any(bar.failed for bar in bars):
        caption += " · a red cross is a run that left no time"
    if link is not None:
        caption += f" · {link}"
    return _figure(chart, caption, table)


def _figure(chart: str, caption: str, table: list[str]) -> str:
    """Wrap a chart with its caption and, when given, a collapsed table of its values."""
    lines = [
        '<figure class="bench-figure">',
        chart,
        f"<figcaption>{caption}</figcaption>",
        "</figure>",
    ]
    if table:
        lines += ["", '??? note "Values"', "", *(f"    {line}" for line in table)]
    return "\n".join(lines)


def group_table(cells: Cells, keys: list[aggregate.CellKey]) -> list[str]:
    """Every input of one function: one row per input, one column per library."""
    failed = {key: dict(failures(key)) for key in keys}
    libraries = sorted(
        {lib for key in keys for lib in cells[key] if lib == SUBJECT or is_reference(lib)}
        | {lib for what in failed.values() for lib in what},
        key=lambda lib: (lib != SUBJECT, registry.display_label(lib).lower()),
    )
    header = "| Input | " + " | ".join(registry.display_label(lib) for lib in libraries) + " |"
    lines = [header, "|---" * (len(libraries) + 1) + "|"]
    for key in sorted(keys, key=lambda key: (_size(key), keys.index(key))):
        row = cells[key]
        values = [
            failed[key].get(lib, "")
            if lib not in row
            else format_limit(row[lib])
            if timed_out(key, lib)
            else format_duration(row[lib])
            for lib in libraries
        ]
        lines.append(f"| {cell_name(key)} | " + " | ".join(values) + " |")
    return lines


def module_page(
    cells: Cells, module: str, groups: list[tuple[GroupKey, list[aggregate.CellKey]]], machine: str
) -> str:
    """Render one module's page: its summary chart, one chart per compared function, the rest."""
    with_reference = [(g, keys) for g, keys in groups if representative(cells, keys) is not None]
    alone = [(g, keys) for g, keys in groups if representative(cells, keys) is None]
    per = library_ratios(cells, [keys for _g, keys in with_reference])
    out = [
        "---",
        "search:",
        "  exclude: true",
        "---",
        "",
        "<!-- Generated by benchmarks/render_docs.py; edit that script, not this file. -->",
        "",
        f"# {module}",
        "",
        f"Benchmarks of [`ordito.{module}`](../api/ordito/{module}.md) against every library that "
        "computes the same thing. Each chart shows the time of every library relative to ordito "
        "on CUDA (1x, in green): a longer bar took longer. The table under each chart lists every "
        "input the function was measured on. See [Benchmarks](index.md) for how to read and "
        "reproduce these numbers.",
        "",
        f"*{machine}*",
        "",
        "## Summary",
        "",
        _figure(
            svg_chart(ratio_bars(per, "functions"), label=f"{module}: time relative to ordito"),
            f"Time relative to ordito (CUDA), geometric mean over the {module} functions both "
            "implement; in parentheses, the number of functions. Log scale."
            + (_LIMIT_NOTE if _TIMED_OUT else ""),
            ratio_table(per, "functions"),
        ),
        "",
    ]
    for (_module, group), keys in with_reference:
        key = representative(cells, keys)
        assert key is not None
        out += [f"## `{group}`", "", cell_figure(cells, key), ""]
        out += ['??? note "All inputs"', "", *(f"    {line}" for line in group_table(cells, keys))]
        out += [""]
    if alone:
        out += [
            "## Without a reference",
            "",
            "No other library computes these, so they have no comparison. ordito (CUDA)'s median "
            "time on the largest input it was measured on:",
            "",
            "| Function | Input | ordito (CUDA) |",
            "|---|---|---|",
        ]
        for (_module, group), keys in alone:
            timed = [key for key in keys if SUBJECT in cells[key]]
            if timed:
                key = max(timed, key=lambda key: (_size(key), keys.index(key)))
                out.append(
                    f"| `{group}` | {cell_name(key)} | {format_duration(cells[key][SUBJECT])} |"
                )
        out.append("")
    return "\n".join(out)


def write_nav(pages: set[str]) -> None:
    """Rewrite the Benchmarks entries of ``docs/SUMMARY.md`` with one link per module page."""
    lines = _NAV.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("* [Benchmarks]"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("* "))
    nav = ["* [Benchmarks](benchmarks/index.md)"]
    for title, modules in api_shelves():
        listed = [m for m in modules if m in pages]
        if listed:
            nav.append(f"    * {title}")
            nav += [f"        * [{m}](benchmarks/{m}.md)" for m in listed]
    _NAV.write_text("\n".join([*lines[:start], *nav, *lines[end:]]) + "\n")


def splice(page: str, cells: Cells, machine: str, pages: set[str]) -> str:
    """Replace every ``<!-- bench:NAME -->`` block of the main page with a freshly rendered one."""
    groups = by_group(cells)

    def render(match: re.Match[str]) -> str:
        name = match.group("name")
        if name == "provenance":
            body = f"*{machine}*"
        elif name == "summary":
            body = summary_block(cells)
        elif name == "modules":
            body = modules_block(cells, pages)
        elif name.startswith("cell "):
            module, group = name.removeprefix("cell ").split("/")
            keys = groups.get((module, group))
            key = representative(cells, keys) if keys else None
            if key is None:
                raise SystemExit(f"{name}: no such function compared with a reference in the data")
            short = module.removeprefix("test_")
            body = cell_figure(cells, key, link=f"[all {short} benchmarks]({short}.md)")
        else:
            raise SystemExit(f"unknown generated block {name!r}")
        return f"{match.group(1)}{body}\n{match.group(4)}"

    return _MARKER.sub(render, page)


def machine_line(json_dir: Path, gpu: str | None) -> str:
    """Name the machine the numbers come from; no date or commit, which would only age."""
    for path in sorted(json_dir.glob("*.json")):
        machine = json.loads(path.read_text()).get("machine_info", {})
        cpu = re.sub(r"\((R|TM)\)", "", machine.get("cpu", {}).get("brand_raw", "not recorded"))
        return f"Measured on GPU: {gpu or machine.get('gpu', 'not recorded')} · CPU: {cpu}"
    return ""


def main(argv: list[str] | None = None) -> None:
    """Render the module pages, the nav entries and the main page's generated blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_dir", type=Path, help="directory of --benchmark-json files")
    parser.add_argument("--gpu", help="GPU name, for a run that did not record one")
    args = parser.parse_args(argv)
    cells = load(args.json_dir)
    machine = machine_line(args.json_dir, args.gpu)

    _LOGO_DIR.mkdir(parents=True, exist_ok=True)
    for png in registry.CLEAR_DIR.glob("*.png"):
        shutil.copyfile(png, _LOGO_DIR / png.name)

    by_module: dict[str, list[tuple[GroupKey, list[aggregate.CellKey]]]] = collections.defaultdict(
        list
    )
    for group, keys in by_group(cells).items():
        by_module[group[0].removeprefix("test_")].append((group, keys))
    known = {m for _title, modules in api_shelves() for m in modules}
    pages = {
        module
        for module, groups in by_module.items()
        if module in known and any(representative(cells, keys) for _g, keys in groups)
    }
    _PAGES.mkdir(parents=True, exist_ok=True)
    for stale in _PAGES.glob("*.md"):
        if stale.name != "index.md" and stale.stem not in pages:
            stale.unlink()
    for module in sorted(pages):
        (_PAGES / f"{module}.md").write_text(
            module_page(cells, module, by_module[module], machine) + "\n"
        )
    write_nav(pages)
    _MAIN_PAGE.write_text(splice(_MAIN_PAGE.read_text(), cells, machine, pages))
    print(f"wrote {len(pages)} module pages, the nav and {_MAIN_PAGE.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
