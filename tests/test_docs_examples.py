"""
The guide pages' examples run, page by page, on both devices.

The fenced ``python`` blocks in docstrings are executed by ``tests/test_api_conventions.py``
(check 12); the hand-written pages under ``docs/`` and the README had no such gate, and two of
their examples called functions that did not exist or names that were never imported. Each page
is run the way a reader would type it: its blocks in order, in one namespace, with later blocks
seeing what earlier ones defined. Blocks nested in a list item are dedented first.

Not a library comparison: this checks that the examples run and what their own asserts say, not
the values they print (those are approximate in the prose, and several vary in their last digits
between CUDA runs).
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import warp as wp

_ROOT = Path(__file__).resolve().parent.parent
_PAGES = sorted(
    path.relative_to(_ROOT).as_posix()
    for path in [_ROOT / "README.md", *(_ROOT / "docs").rglob("*.md")]
    if "```python" in path.read_text()
)
_BLOCK = re.compile(r"^([ \t]*)```python\n(.*?)^\1```", re.MULTILINE | re.DOTALL)


def _blocks(page: str) -> list[str]:
    text = (_ROOT / page).read_text()
    return [textwrap.dedent(match.group(2)) for match in _BLOCK.finditer(text)]


def test_every_page_with_examples_is_collected() -> None:
    """Not a library comparison: the parametrization below finds the pages that have examples."""
    assert "README.md" in _PAGES
    assert "docs/cookbook/align-two-scans.md" in _PAGES
    assert all(_blocks(page) for page in _PAGES)


@pytest.mark.parametrize("page", _PAGES)
def test_page_examples_run(device: str, page: str) -> None:
    """Not a library comparison: every block of ``page`` runs, in order, on ``device``."""
    namespace: dict[str, object] = {"__name__": "__docs_example__"}
    with wp.ScopedDevice(device):
        for block in _blocks(page):
            exec(compile(block, page, "exec"), namespace)
