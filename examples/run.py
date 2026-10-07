"""
Run the examples, render their images and write the gallery pages.

Usage::

    python -m examples.run                 # every example, images and pages
    python -m examples.run --only T1,G1    # a subset (pages are still written for every example)
    python -m examples.run --pages-only    # rewrite the pages from the code, keep the images

Each example module under ``examples/<topic>/`` defines ``META`` ([`Meta`][examples._meta.Meta]),
``run(device) -> dict`` whose body between the ``# --8<-- [start:code]`` / ``[end:code]`` markers is
the code shown on the page, and ``figure(result) -> Figure``. Anything the code prints is shown on
the page under the code.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import pkgutil
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import warp as wp

import examples
from examples._meta import Meta
from examples._render import render_figure

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMAGES = DOCS / "assets" / "examples"
PAGES = DOCS / "examples"
OUTPUTS = Path(__file__).resolve().parent / "_outputs"
"""Captured stdout of each example, kept so ``--pages-only`` can rebuild a page."""

_START = "# --8<-- [start:code]"
_END = "# --8<-- [end:code]"


@dataclass
class Topic:
    """One gallery page: a sub-package of ``examples`` and its example modules."""

    module: ModuleType
    examples: list[ModuleType]

    @property
    def slug(self) -> str:
        """The page file stem."""
        return self.module.__name__.rsplit(".", 1)[1].replace("_", "-")

    @property
    def title(self) -> str:
        """The page title."""
        return str(self.module.TITLE)


def _natural_key(module: ModuleType) -> tuple[str, int]:
    meta: Meta = module.META
    match = re.fullmatch(r"([A-Z]+)(\d+)", meta.id)
    return (match.group(1), int(match.group(2))) if match else (meta.id, 0)


def discover() -> list[Topic]:
    """Return every topic page in gallery order, each with its examples in id order."""
    topics = []
    for info in pkgutil.iter_modules(examples.__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        package = importlib.import_module(f"examples.{info.name}")
        members = [
            importlib.import_module(f"{package.__name__}.{sub.name}")
            for sub in pkgutil.iter_modules(package.__path__)
            if not sub.name.startswith("_")
        ]
        topics.append(Topic(package, sorted(members, key=_natural_key)))
    return sorted(topics, key=lambda t: int(t.module.ORDER))


def code_of(module: ModuleType) -> str:
    """Return the dedented code between an example's ``start:code`` and ``end:code`` markers."""
    source = Path(module.__file__ or "").read_text()
    start = source.index(_START) + len(_START)
    end = source.index(_END)
    body = source[start:end].strip("\n")
    return textwrap.dedent(body).rstrip() + "\n"


def run_example(module: ModuleType, device: str) -> None:
    """Run one example, render its image and keep its printed output."""
    meta: Meta = module.META
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result: dict[str, Any] = module.run(device)
    figure = module.figure(result)
    render_figure(
        figure, IMAGES / f"{meta.anchor}.webp", thumb=IMAGES / f"{meta.anchor}-thumb.webp"
    )
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / f"{meta.anchor}.txt").write_text(stdout.getvalue())


def _credits(meta: Meta) -> str:
    if not meta.credits:
        return ""
    links = " · ".join(f"[{label}]({url})" for label, url in meta.credits)
    return f"Modelled on: {links}\n{{: .ordito-credits }}\n"


def page_text(topic: Topic) -> str:
    """Return the Markdown of ``docs/examples/<topic>.md``."""
    doc = textwrap.dedent(str(topic.module.INTRO)).strip()
    lines = [f"# {topic.title}\n", f"{doc}\n" if doc else ""]
    for module in topic.examples:
        meta: Meta = module.META
        lines.append(f"## {meta.title} {{#{meta.anchor}}}\n")
        lines.append(f"{textwrap.dedent(meta.summary).strip()}\n")
        lines.append(f"```python\n{code_of(module)}```\n")
        output_file = OUTPUTS / f"{meta.anchor}.txt"
        output = output_file.read_text().rstrip() if output_file.exists() else ""
        if output:
            lines.append(f'```text title="Output"\n{output}\n```\n')
        lines.append(f"![{meta.title}](../assets/examples/{meta.anchor}.webp)\n")
        if meta.notes:
            lines.append(f"{textwrap.dedent(meta.notes).strip()}\n")
        lines.append(_credits(meta))
    return "\n".join(line for line in lines if line)


def index_text(topics: list[Topic]) -> str:
    """Return the Markdown of ``docs/examples/index.md``: a thumbnail grid per topic."""
    lines = ["# Examples\n", textwrap.dedent(examples.GALLERY_INTRO).strip() + "\n"]
    for topic in topics:
        lines.append(f"## [{topic.title}]({topic.slug}.md)\n")
        lines.append('<div class="grid cards ordito-gallery" markdown>\n')
        for module in topic.examples:
            meta: Meta = module.META
            lines.append(
                f"-   [![{meta.title}](../assets/examples/{meta.anchor}-thumb.webp)"
                f"]({topic.slug}.md#{meta.anchor})\n\n"
                f"    [{meta.title}]({topic.slug}.md#{meta.anchor})\n"
            )
        lines.append("</div>\n")
    return "\n".join(lines)


def write_pages(topics: list[Topic]) -> None:
    """Write every topic page and the gallery index under ``docs/examples``."""
    PAGES.mkdir(parents=True, exist_ok=True)
    for topic in topics:
        (PAGES / f"{topic.slug}.md").write_text(page_text(topic))
    (PAGES / "index.md").write_text(index_text(topics))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated example ids to run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pages-only", action="store_true")
    args = parser.parse_args(argv)
    wp.config.log_level = wp.LOG_WARNING  # keep module-load lines out of the captured output
    topics = discover()
    wanted = {i.strip().upper() for i in args.only.split(",") if i.strip()}
    failures = []
    if not args.pages_only:
        for topic in topics:
            for module in topic.examples:
                meta: Meta = module.META
                if wanted and meta.id.upper() not in wanted:
                    continue
                start = time.perf_counter()
                try:
                    run_example(module, args.device)
                except Exception:  # noqa: BLE001 -- report every failing example, then exit 1
                    failures.append(meta.id)
                    traceback.print_exc()
                    print(f"FAILED {meta.id}", file=sys.stderr)
                    continue
                print(f"{meta.id:>5}  {time.perf_counter() - start:6.1f} s  {meta.title}")
    write_pages(topics)
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
