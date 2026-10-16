"""
Runtime configuration: whether floating-point reductions and scans repeat bit for bit.

On CUDA, a floating-point sum whose blocks commit in arrival order, or a ``float32`` scan whose
blocks combine in arrival order, can change its last bits from one call to the next. With
determinism on (the default) every such sum in ordito adds its per-block partials in one fixed
order and every ``float32`` scan takes a fixed-order tile scan, so repeated calls on the same input
return identical bits, and a seeded sampler draws the same points. Off, the arrival-order forms are
used, which saves a few microseconds per reduction and matters only on short calls.

What the setting governs:

- floating-point block reductions (sums, means, moments, fits, the reductions inside the energies,
  metrics, measures, registration and solver loops) and ``float32`` scans (the CDF behind every
  sampler, arc lengths, resampling) on the CUDA device. The CPU device runs its blocks in turn and
  is deterministic either way.

What it does not govern:

- floating-point atomic scatters, such as the face-to-vertex accumulation of vertex normals or
  areas, which stay in arrival order on CUDA whatever the setting;
- integer reductions and scans, which are exact in any order.

The setting is read when a reduction launches. Two consequences follow:

- a CUDA graph keeps the mode it was recorded with: replaying it after a change of setting replays
  the old choice;
- state a [`Trimesh`][ordito.mesh.Trimesh] caches (heat operators, solvers) keeps the mode it was
  built in until it is rebuilt.

[`set_deterministic`][ordito.config.set_deterministic] sets the process-wide default, which every
thread and task sees. The [`deterministic`][ordito.config.deterministic] context manager overrides
it for the current thread or ``asyncio`` task only, and restores the previous value on exit; an
override wins over the default for as long as it is active, even if the default changes meanwhile.

The three names are also reachable from the package itself, as ``od.set_deterministic``,
``od.is_deterministic`` and ``od.deterministic``; they are the same objects as the ones here.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# The process-wide default, and the scoped override. The override lives in a ``ContextVar`` so a
# ``with deterministic(...)`` block changes only its own thread or ``asyncio`` task: a new thread
# starts from an empty context and sees the process default, a new task copies its creator's
# context and so inherits an active override. ``None`` means "no override, read the default".
_process_default = True
_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "ordito_deterministic", default=None
)


def set_deterministic(enabled: bool) -> None:
    """
    Set the process-wide default for reproducible floating-point reductions and scans.

    Every thread and task without an active [`deterministic`][ordito.config.deterministic]
    override sees the new value from its next reduction on. An active override is not changed.

    Parameters
    ----------
    enabled
        ``True`` (the default at import) adds per-block partial sums in one fixed order and takes
        the fixed-order ``float32`` scan, so repeated calls return identical bits on CUDA.
        ``False`` takes the faster arrival-order forms.

    See Also
    --------
    deterministic : Override the setting within a ``with`` block.
    is_deterministic : Read the setting in force.

    Examples
    --------
    ```python
    import ordito as od

    od.config.set_deterministic(False)
    print(od.config.is_deterministic())  # False
    od.config.set_deterministic(True)
    print(od.config.is_deterministic())  # True
    ```
    """
    global _process_default
    _process_default = bool(enabled)


def is_deterministic() -> bool:
    """
    Whether floating-point reductions and scans are reproducible in the current context.

    Returns
    -------
    bool
        The innermost active [`deterministic`][ordito.config.deterministic] override of this
        thread or task, or the process-wide default set by
        [`set_deterministic`][ordito.config.set_deterministic] when none is active.

    Examples
    --------
    ```python
    import ordito as od

    print(od.config.is_deterministic())  # True
    ```
    """
    override = _override.get()
    return _process_default if override is None else override


@contextlib.contextmanager
def deterministic(enabled: bool = True) -> Generator[None, None, None]:
    """
    Override the determinism setting within a ``with`` block.

    The override applies to the current thread or ``asyncio`` task only (tasks created inside the
    block inherit it), nests, and restores the previous value on exit, also when the block raises.

    Parameters
    ----------
    enabled
        ``True`` for reproducible floating-point reductions and ``float32`` scans, ``False`` for
        the faster arrival-order forms (see [`ordito.config`][ordito.config]).

    Yields
    ------
    None
        Control to the body of the ``with`` block.

    See Also
    --------
    set_deterministic : Set the process-wide default.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    with od.config.deterministic(False):
        print(od.config.is_deterministic())  # False
        points, _ = od.sample.sample_surface(vertices, faces, 100, seed=0)
    print(od.config.is_deterministic())  # True
    print(points.shape)  # (100,)
    ```
    """
    token = _override.set(bool(enabled))
    try:
        yield
    finally:
        _override.reset(token)
