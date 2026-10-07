"""
The example gallery stays in step with its code.

`examples.run` generates the pages under ``docs/examples`` from each example module, so a page
can go stale only when someone edits an example's code and does not regenerate. These tests catch
that, and catch an example module that breaks the gallery's contract. Running the examples
themselves needs the Stanford scans (``python -m examples.fetch_data``) and renders images, so it
is left to ``python -m examples.run``. Set ``ORDITO_RUN_EXAMPLES=1`` to execute every example's
code here too, without rendering.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import ModuleType

import pytest

from examples import run as gallery
from examples._meta import Meta

TOPICS = gallery.discover()
MODULES = [module for topic in TOPICS for module in topic.examples]
IDS = [module.META.id for module in MODULES]


def test_example_ids_are_unique() -> None:
    """Not a library comparison: every example id names exactly one anchor and image."""
    assert len(IDS) == len(set(IDS))
    assert all(re.fullmatch(r"[A-Z]+\d+", i) for i in IDS)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_example_module_contract(module: ModuleType) -> None:
    """Not a library comparison: each module carries META, run, figure and the code markers."""
    assert isinstance(module.META, Meta)
    assert callable(module.run)
    assert callable(module.figure)
    code = gallery.code_of(module)
    assert "import ordito as od" in code
    assert module.META.summary.strip()


@pytest.mark.parametrize("topic", TOPICS, ids=[t.slug for t in TOPICS])
def test_pages_are_up_to_date(topic: gallery.Topic) -> None:
    """Not a library comparison: the committed page equals the one the code generates."""
    page = gallery.PAGES / f"{topic.slug}.md"
    assert page.read_text() == gallery.page_text(topic), (
        f"{page} is stale: run `python -m examples.run --pages-only`"
    )


def test_index_is_up_to_date() -> None:
    """Not a library comparison: the committed gallery index equals the generated one."""
    assert (gallery.PAGES / "index.md").read_text() == gallery.index_text(TOPICS)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_every_example_has_an_image(module: ModuleType) -> None:
    """Not a library comparison: each example's image and thumbnail are committed."""
    anchor = module.META.anchor
    assert Path(gallery.IMAGES / f"{anchor}.webp").exists()
    assert Path(gallery.IMAGES / f"{anchor}-thumb.webp").exists()


@pytest.mark.skipif(
    os.environ.get("ORDITO_RUN_EXAMPLES") != "1", reason="set ORDITO_RUN_EXAMPLES=1 to run them"
)
@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_example_runs(module: ModuleType) -> None:
    """Not a library comparison: the example's code runs and returns its figure's data."""
    result = module.run("cuda:0")
    assert isinstance(result, dict)
    assert module.figure(result) is not None
