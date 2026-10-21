"""
Runtime configuration: whether results repeat bit for bit, and come back in the same order.

On CUDA, a floating-point sum whose blocks commit in arrival order, or a ``float32`` scan whose
blocks combine in arrival order, can change its last bits from one call to the next, and an output
filled through atomic slots can list its entries in a different order. With determinism on every
such sum in ordito adds its per-block partials in one fixed order, every ``float32`` scan takes a
fixed-order tile scan and those outputs come back in a fixed order, so repeated calls on the same
input return identical arrays, and a seeded sampler draws the same points. Off (the default), the
arrival-order forms are used, which saves a few microseconds per reduction and a sort per ordered
output.

What the setting governs, on the CUDA device (the CPU device runs its blocks in turn and is
deterministic either way):

- floating-point block reductions (sums, means, moments, fits, the reductions inside the energies,
  metrics, measures, registration and solver loops) and ``float32`` scans (the CDF behind every
  sampler, arc lengths, resampling);
- the lumped masses of [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries] and
  [`mass_matrix`][ordito.laplacian.mass_matrix] given ``float64`` face areas (summed in ascending
  face order when on; ``float32`` or computed areas sum exactly and repeat either way);
- the per-vertex sums of [`hessian_energy`][ordito.energies.hessian_energy],
  [`curved_hessian_energy`][ordito.energies.curved_hessian_energy] and
  [`vertex_defects`][ordito.vertices.vertex_defects];
- the rotation right-hand side of every local step of
  [`arap`][ordito.parametrization.arap] (each vertex's edge terms added in ascending face order
  when on);
- the order within each row of
  [`vertex_face_adjacency`][ordito.adjacency.vertex_face_adjacency] (ascending when on);
- the row order of the ear-clipped faces of
  [`polyline_triangulate`][ordito.polyline.polyline_triangulate],
  [`polyline_triangulate_from_offsets`][ordito.polyline.polyline_triangulate_from_offsets] and
  [`triangulate_polygon`][ordito.polyline.triangulate_polygon] (by ascending middle index when on);
- the per-loop sums of [`loop_perimeters`][ordito.boundary.loop_perimeters],
  [`loop_directed_areas`][ordito.boundary.loop_directed_areas] and their packed forms, and the
  hole planes of [`fill_min_weight`][ordito.holes.fill_min_weight] and the fills built on it;
- the flip mask of [`face_flip_mask`][ordito.validation.face_flip_mask] and the bits of
  [`face_orientation_bits`][ordito.validation.face_orientation_bits] on a non-orientable mesh,
  and so the seam [`make_winding_consistent`][ordito.repair.make_winding_consistent] leaves;
- the per-component signed volumes of [`make_volume`][ordito.repair.make_volume] and
  [`make_normals_outward`][ordito.repair.make_normals_outward] with ``multibody=True``, and the
  per-component areas of [`remove_small_components`][ordito.repair.remove_small_components] with
  ``min_area`` (summed exactly when on, so on the CPU device too a near-tie can resolve otherwise
  than with it off), and so which components are flipped or kept;
- the positions of [`subdivide_loop`][ordito.remesh.subdivide_loop], whose 1-ring sums are
  added in a fixed order;
- the values of the differentiable Chamfer losses
  [`chamfer_points_to_points_loss`][ordito.metrics.chamfer_points_to_points_loss],
  [`chamfer_points_to_mesh_loss`][ordito.metrics.chamfer_points_to_mesh_loss] and
  [`chamfer_mesh_to_mesh_loss`][ordito.metrics.chamfer_mesh_to_mesh_loss], and the gradients a
  ``wp.Tape`` takes through them (each point's or vertex's gradient summed in ascending query
  order when on);
- the order of the hits [`intersects_location`][ordito.ray.intersects_location] returns
  (ascending ray order when on).
- the lattice data of [`screened_poisson`][ordito.reconstruction.screened_poisson], and so its
  surface: the splat of the normals (each node's terms added in ascending sample order when on)
  and the adaptive backend's restriction of it down its levels.

What it does not govern:

- the other floating-point atomic scatters, which stay in arrival order on CUDA whatever the
  setting (the vertex areas and normals sum exactly, and so repeat anyway);
- integer reductions and scans, which are exact in any order.

The setting is read when a reduction launches. Two consequences follow:

- a CUDA graph keeps the mode it was recorded with: replaying it after a change of setting replays
  the old choice;
- state a [`Trimesh`][ordito.mesh.Trimesh] caches (heat operators, solvers) keeps the mode it was
  built in until it is rebuilt. The one exception is
  [`Trimesh.vertex_face_adjacency`][ordito.mesh.Trimesh.vertex_face_adjacency]: a value cached
  with the setting off is rebuilt, with ascending rows, the first time it is read with the setting
  on.

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
_process_default = False
_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "ordito_deterministic", default=None
)


def set_deterministic(enabled: bool) -> None:
    """
    Set the process-wide default for reproducible results (see [`ordito.config`][ordito.config]).

    Every thread and task without an active [`deterministic`][ordito.config.deterministic]
    override sees the new value from its next reduction on. An active override is not changed.

    Parameters
    ----------
    enabled
        ``True`` adds per-block partial sums in one fixed order, takes the fixed-order
        ``float32`` scan and fixes the order of the outputs the setting governs, so repeated calls
        return identical arrays on CUDA. ``False`` (the default at import) takes the faster
        arrival-order forms.

    See Also
    --------
    deterministic : Override the setting within a ``with`` block.
    is_deterministic : Read the setting in force.

    Examples
    --------
    ```python
    import ordito as od

    od.config.set_deterministic(True)
    print(od.config.is_deterministic())  # True
    od.config.set_deterministic(False)
    print(od.config.is_deterministic())  # False
    ```
    """
    global _process_default
    _process_default = bool(enabled)


def is_deterministic() -> bool:
    """
    Whether results are reproducible in the current context (see [`ordito.config`][ordito.config]).

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

    with od.config.deterministic(True):
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
        ``True`` for reproducible results, ``False`` for the faster arrival-order forms (see
        [`ordito.config`][ordito.config]).

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
        with od.config.deterministic(True):
            print(od.config.is_deterministic())  # True
            points, _ = od.sample.sample_surface(vertices, faces, 100, seed=0)
        print(od.config.is_deterministic())  # False
    print(points.shape)  # (100,)
    ```
    """
    token = _override.set(bool(enabled))
    try:
        yield
    finally:
        _override.reset(token)
