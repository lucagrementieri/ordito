"""
Root conftest: the one flag both suites read.

``--device`` is registered here rather than in ``tests/conftest.py`` and ``benchmarks/conftest.py``
because argparse rejects a duplicate option string, so registering it in both made
``pytest tests benchmarks`` die with ``conflicting option string: --device`` before collecting
anything. Each suite interprets it on its own, but ``auto`` means the same thing in both --
**one** device, cuda if available -- for two different reasons that happen to agree:

- ``benchmarks/`` -- a ``ordito-cpu`` row timed beside a ``ordito-cuda`` one is a fight the CPU
  cannot win (CLAUDE.md section 13), so it is off unless asked for.
- ``tests/`` -- correctness *is* worth checking on both devices, but not inside one process:
  **Warp's CPU work is ~36x slower once CUDA has been initialised** (50.57 s against 1.40 s on one
  ``heat_signed_distance`` call, unchanged by ``launch_array_access_mode``, so it is CUDA presence
  and not section 8's launch guard). Both-device coverage is therefore bought as *two* processes by
  ``uv run python -m tests.devices``, which hides CUDA from the CPU pass -- never by
  ``--device=both`` on a GPU box, which measured 717 s against the runner's ~193 s.

``both`` means "everything" in both suites, which is the property worth keeping aligned: it is the
escape hatch a reader reaches for without checking which suite they are in.
"""

from __future__ import annotations

import pytest

DEVICE_CHOICES = ("auto", "cpu", "cuda", "both")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--device`` once for the whole repository."""
    parser.getgroup("ordito").addoption(
        "--device",
        action="store",
        default="auto",
        choices=list(DEVICE_CHOICES),
        help="device selection. tests/: auto=both devices minus slow_cpu, cpu, cuda, both="
        "everything. benchmarks/: ordito target(s) to time, auto=cuda if available else cpu.",
    )
