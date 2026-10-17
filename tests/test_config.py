"""
Tests for ``ordito.config``: the process default, the scoped override and their isolation.

Not a library comparison anywhere in this file: the setting is ordito's own. That the leaf points
honour it is pinned in ``tests/test_launch.py``, and that each family's fast path agrees with the
default in the ``arrival_order`` tests of the module that owns it.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from collections.abc import Iterator

import pytest

import ordito as od


@pytest.fixture(autouse=True)
def _library_default() -> Iterator[None]:
    """
    Start from the library default (off), not the suite's (on), and put the suite's back after.

    The tests below are written against the default a caller gets; that it really is off is
    checked in a fresh interpreter by ``test_the_library_default_is_off``.
    """
    od.config.set_deterministic(False)
    yield
    od.config.set_deterministic(True)


def test_the_library_default_is_off() -> None:
    """Not a library comparison: a fresh ``import ordito`` reads the setting as off."""
    probe = "import ordito as od; print(od.config.is_deterministic())"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "False"


def test_set_deterministic_sets_the_process_default() -> None:
    """Not a library comparison: ``set_deterministic`` flips the process default both ways."""
    assert od.config.is_deterministic() is False
    od.config.set_deterministic(True)
    assert od.config.is_deterministic() is True
    od.config.set_deterministic(False)
    assert od.config.is_deterministic() is False


def test_deterministic_nests_and_restores() -> None:
    """Not a library comparison: each block overrides the one around it and restores it on exit."""
    with od.config.deterministic(True):
        assert od.config.is_deterministic() is True
        with od.config.deterministic(False):
            assert od.config.is_deterministic() is False
            with od.config.deterministic(True):
                assert od.config.is_deterministic() is True
            assert od.config.is_deterministic() is False
        assert od.config.is_deterministic() is True
    assert od.config.is_deterministic() is False


def test_deterministic_defaults_to_on_and_wins_over_the_process_default() -> None:
    """
    Not a library comparison: an override holds while the default changes under it.

    The bare ``deterministic()`` switches determinism on under an off default, and a default
    changed inside the block takes effect only once the block exits.
    """
    od.config.set_deterministic(False)
    with od.config.deterministic():
        assert od.config.is_deterministic() is True
        od.config.set_deterministic(False)
        assert od.config.is_deterministic() is True
    assert od.config.is_deterministic() is False
    with od.config.deterministic(False):
        od.config.set_deterministic(True)
        assert od.config.is_deterministic() is False
    assert od.config.is_deterministic() is True


def test_deterministic_restores_on_exception() -> None:
    """Not a library comparison: a block left by an exception restores the previous value."""
    message = "inside the block"
    seen: list[bool] = []

    def fail_inside() -> None:
        with od.config.deterministic(True):
            seen.append(od.config.is_deterministic())
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        fail_inside()
    assert seen == [True]
    assert od.config.is_deterministic() is False


def test_an_override_does_not_leak_into_another_thread() -> None:
    """
    Not a library comparison: a thread sees the process default, not another thread's override.

    The worker starts while the main thread is inside an on block and must still read off; its own
    on block must in turn leave the main thread's view alone.
    """
    seen: dict[str, bool] = {}
    entered, release = threading.Event(), threading.Event()

    def worker() -> None:
        seen["start"] = od.config.is_deterministic()
        with od.config.deterministic(True):
            seen["inside"] = od.config.is_deterministic()
            entered.set()
            release.wait(timeout=10.0)
        seen["after"] = od.config.is_deterministic()

    with od.config.deterministic(True):
        thread = threading.Thread(target=worker)
        thread.start()
        assert entered.wait(timeout=10.0)
        assert od.config.is_deterministic() is True
    assert od.config.is_deterministic() is False
    release.set()
    thread.join(timeout=10.0)
    assert seen == {"start": False, "inside": True, "after": False}


def test_a_new_thread_sees_the_process_default() -> None:
    """Not a library comparison: ``set_deterministic`` reaches a thread with no override."""
    seen: list[bool] = []
    od.config.set_deterministic(True)
    thread = threading.Thread(target=lambda: seen.append(od.config.is_deterministic()))
    thread.start()
    thread.join(timeout=10.0)
    assert seen == [True]


def test_an_asyncio_task_inherits_its_creators_override() -> None:
    """Not a library comparison: a task copies the context it was created in, override included."""

    async def read() -> bool:
        return od.config.is_deterministic()

    async def main() -> tuple[bool, bool]:
        with od.config.deterministic(True):
            inside = await asyncio.create_task(read())
        outside = await asyncio.create_task(read())
        return inside, outside

    assert asyncio.run(main()) == (True, False)


@pytest.mark.parametrize("name", ["set_deterministic", "is_deterministic", "deterministic"])
def test_top_level_names_are_the_config_functions(name: str) -> None:
    """Not a library comparison: ``od.<name>`` is ``od.config.<name>``, the same object."""
    assert getattr(od, name) is getattr(od.config, name)
