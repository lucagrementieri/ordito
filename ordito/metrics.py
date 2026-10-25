"""
Chamfer and Hausdorff distances between point clouds and meshes.

All metrics are fully GPU-resident: they compose the nearest-neighbor and
point-to-surface primitives in [`proximity`][ordito.proximity] with the tiled reductions
in [`reduce`][ordito.reduce], and never move per-element data to the host.

Chamfer distances reduce plain Euclidean distances by default, the usual definition of the
metric. ``squared=True`` squares each distance before reducing instead, which is the ``pytorch3d``
convention (``pytorch3d.loss.chamfer_distance``) and the one to pass when comparing against it.
Hausdorff distances follow libigl's ``igl::hausdorff`` and reduce the
(already Euclidean) per-element distances with a maximum, so no squaring or final square root is
needed.

One place the convention does *not* carry over, because the two libraries answer different
questions: ``pytorch3d.loss.point_mesh_face_distance`` sums point-to-triangle with
**face**-to-point, where
[`chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh]'s backward direction is
mesh-*vertex* to nearest query. The forward halves agree, but the whole scalars differ by
construction and are not comparable.

Two families of geometry are supported and can be mixed:

- ``points_to_points`` -- both sides are point clouds
  (nearest neighbor via [`query_nearest`][ordito.neighbors.query_nearest]).
- ``points_to_mesh`` -- a point cloud versus a triangle mesh
  (point-to-surface via [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
  one way, mesh-vertex nearest neighbor the other way).
- ``mesh_to_mesh`` -- both sides are triangle meshes (vertex-to-surface both directions).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple, cast, overload

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import prefers_tiled_reduction, read_values, require_same_device, slice_count
from ordito.config import is_deterministic
from ordito.constants import TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import metrics as kernel_metrics
from ordito.kernels import proximity as kernel_proximity
from ordito.kernels import reduce as kernel_reduce

_PointReduction = Literal["mean", "sum", "max"]
# Differentiable Chamfer losses support only additive point reductions ("max" has no
# useful gradient here, and "None" is a per-point array rather than a scalar loss).
_DiffReduction = Literal["mean", "sum"]

# The backward cloud search of a pair takes a seeded hash grid when the searched cloud is on a CUDA
# device and has at least ``_GRID_BACKWARD_MIN_POINTS`` points, the forward half's largest distance
# is at most ``_GRID_BACKWARD_CEILING`` of its density radius and at least
# ``_GRID_BACKWARD_MIN_HITS`` of the queries are some point's forward answer; the seed is held no
# lower than ``_GRID_BACKWARD_FLOOR`` of the radius -- see ``_backward_nearest``.
_GRID_BACKWARD_MIN_POINTS = 262_144
_GRID_BACKWARD_CEILING = 0.5
_GRID_BACKWARD_MIN_HITS = 0.75
_GRID_BACKWARD_FLOOR = 0.1
# ``pair_search_scale``'s seed: ``inf`` under the seven ``atomic_min`` slots, zero under the count.
_SCALE_SEED = np.array([math.inf] * 7 + [0.0], dtype=np.float32)

# Per-point squared-distance result when ``point_reduction is None``.
_UnreducedChamfer = odt.Array1dFloat32 | tuple[odt.Array1dFloat32, odt.Array1dFloat32]


class _Distances(NamedTuple):
    """
    Forward/backward per-element distances from one of the three geometry dispatches.

    ``backward`` is ``None`` in the single-directional case. ``forward_max`` is the forward array's
    maximum when the dispatch already read it back to choose the backward search, so a ``"max"``
    reduction or a Hausdorff distance reuses that value instead of reducing the same array again;
    ``None`` when nothing has read it.
    """

    forward: odt.ArrayNdFloat32
    backward: odt.ArrayNdFloat32 | None
    forward_max: float | None


class _NnTerm(NamedTuple):
    """One point-to-point Chamfer loss term: each ``x[i]`` against ``y[nearest[i]]``."""

    x: wp.array[wp.vec3]
    y: wp.array[wp.vec3]
    nearest: odt.ArrayNdInt32
    scale: float


class _SurfaceTerm(NamedTuple):
    """One point-to-surface Chamfer loss term: each point against triangle ``face_id[i]``."""

    points: wp.array[wp.vec3]
    vertices: wp.array[wp.vec3]
    faces: wp.array[wp.int32]
    face_id: odt.ArrayNdInt32
    scale: float


_ChamferTerm = _NnTerm | _SurfaceTerm


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------


@overload
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: _PointReduction = ...,
    single_directional: bool = ...,
    squared: bool = ...,
) -> float: ...
@overload
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: None,
    single_directional: Literal[True],
    squared: bool = ...,
) -> odt.Array1dFloat32: ...
@overload
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: None,
    single_directional: Literal[False] = ...,
    squared: bool = ...,
) -> tuple[odt.Array1dFloat32, odt.Array1dFloat32]: ...
@overload
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: None,
    single_directional: bool,
    squared: bool = ...,
) -> _UnreducedChamfer: ...
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: _PointReduction | None = "mean",
    single_directional: bool = False,
    squared: bool = False,
) -> float | _UnreducedChamfer:
    """
    Chamfer distance between two point clouds.

    For each point in ``x`` the Euclidean distance to its nearest neighbor in ``y`` is
    accumulated (and symmetrically for ``y`` into ``x`` unless ``single_directional``).
    ``pytorch3d.loss.chamfer_distance`` is this call with ``squared=True``; its
    ``single_directional`` keyword maps onto this one exactly.

    Parameters
    ----------
    x
        ``(n,)`` query point cloud.
    y
        ``(m,)`` target point cloud.
    point_reduction
        How to reduce the per-point distances of each direction:
        ``"mean"`` (default), ``"sum"``, ``"max"``, or ``None`` to return the
        per-point distances unreduced. ``"max"`` yields a directed
        Hausdorff-like value.
    single_directional
        If ``True``, only the ``x -> y`` term is computed.
    squared
        Square each distance before reducing, as ``pytorch3d`` does. Off by default: the result
        is then in squared units, and its ``"mean"`` is not a mean distance, which is easy to
        misread.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        The reduced Chamfer distance as a Python ``float`` when ``point_reduction``
        is not ``None``. Otherwise the per-point distances: a single
        ``(n,)`` array when ``single_directional``, else an ``((n,), (m,))`` tuple.
        Empty inputs yield ``0.0`` (reduced) or empty arrays.

    Raises
    ------
    RuntimeError
        If ``x`` and ``y`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"``, ``"sum"``, ``"max"`` or ``None``.

    See Also
    --------
    [`chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh]
    [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh]
    [`query_nearest`][ordito.neighbors.query_nearest]

    Examples
    --------
    The ``x -> y`` distances are ``[0, 1]`` and the ``y -> x`` ones ``[0, 3]``; the symmetric value
    is the sum of the two means:

    ```python
    import warp as wp

    import ordito as od

    x = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    y = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=wp.vec3)
    print(od.metrics.chamfer_points_to_points(x, y))  # 2.0
    print(od.metrics.chamfer_points_to_points(x, y, single_directional=True))  # 0.5
    print(od.metrics.chamfer_points_to_points(x, y, squared=True))  # 5.0
    forward, backward = od.metrics.chamfer_points_to_points(x, y, point_reduction=None)
    print(forward.numpy().tolist(), backward.numpy().tolist())  # [0.0, 1.0] [0.0, 3.0]
    ```
    """
    require_same_device(x=x, y=y)
    _validate_point_reduction(point_reduction)
    distances = _distances_points_to_points(x, y, single_directional)
    if distances is None:
        return _empty_chamfer(point_reduction, single_directional, x.device)
    return _chamfer(distances, point_reduction, single_directional, squared)


@overload
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction = ...,
    single_directional: bool = ...,
    squared: bool = ...,
    mesh: wp.Mesh | None = ...,
) -> float: ...
@overload
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: Literal[True],
    squared: bool = ...,
    mesh: wp.Mesh | None = ...,
) -> odt.Array1dFloat32: ...
@overload
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: Literal[False] = ...,
    squared: bool = ...,
    mesh: wp.Mesh | None = ...,
) -> tuple[odt.Array1dFloat32, odt.Array1dFloat32]: ...
@overload
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: bool,
    squared: bool = ...,
    mesh: wp.Mesh | None = ...,
) -> _UnreducedChamfer: ...
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction | None = "mean",
    single_directional: bool = False,
    squared: bool = False,
    mesh: wp.Mesh | None = None,
) -> float | _UnreducedChamfer:
    """
    Chamfer distance between a point cloud and a triangle mesh.

    The forward term is the distance from each point to the mesh surface
    (exact, via [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]);
    the backward term is the distance from each mesh vertex to its nearest
    point in the cloud.

    Parameters
    ----------
    points
        ``(n,)`` query point cloud.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index array.
    point_reduction
        ``"mean"`` (default), ``"sum"``, ``"max"``, or ``None``. See
        [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points].
    single_directional
        If ``True``, only the ``points -> mesh surface`` term is computed.
    squared
        Square each distance before reducing, as ``pytorch3d`` does. Off by default: the result
        is then in squared units, and its ``"mean"`` is not a mean distance, which is easy to
        misread.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the BVH build each
        call otherwise pays (worth it when one mesh is measured repeatedly). Not checked against
        ``vertices`` / ``faces``: the answer is for the geometry ``mesh`` holds.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is the cached one to pass.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        Same layout as [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points].
        The forward array is ``(n,)`` and the backward array ``(n_vertices,)``, except when
        ``points``, ``vertices`` or ``faces`` is empty -- a mesh with no faces counts as empty here
        even when ``vertices`` is not, since there is no surface for the forward term to measure
        against -- in which case both arrays are length 0, matching
        [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points]'s own convention.

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices``, ``faces`` and ``mesh`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"``, ``"sum"``, ``"max"`` or ``None``.

    See Also
    --------
    [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points]
    [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh]
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]

    Examples
    --------
    A point ``1.0`` outside a unit cube and one at its centre, ``0.5`` from every face. The backward
    term adds the mean distance from the cube's corners to the cloud, ``sqrt(3) / 2``:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    points = wp.array([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3)
    forward = od.metrics.chamfer_points_to_mesh(points, vertices, faces, single_directional=True)
    print(forward)  # 0.75
    print(f"{od.metrics.chamfer_points_to_mesh(points, vertices, faces):.4f}")  # 1.6160
    ```
    """
    require_same_device(points=points, vertices=vertices, faces=faces, mesh=mesh)
    _validate_point_reduction(point_reduction)
    distances = _distances_points_to_mesh(points, vertices, faces, single_directional, mesh)
    if distances is None:
        return _empty_chamfer(point_reduction, single_directional, points.device)
    return _chamfer(distances, point_reduction, single_directional, squared)


@overload
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction = ...,
    single_directional: bool = ...,
    squared: bool = ...,
    mesh_a: wp.Mesh | None = ...,
    mesh_b: wp.Mesh | None = ...,
) -> float: ...
@overload
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: Literal[True],
    squared: bool = ...,
    mesh_a: wp.Mesh | None = ...,
    mesh_b: wp.Mesh | None = ...,
) -> odt.Array1dFloat32: ...
@overload
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: Literal[False] = ...,
    squared: bool = ...,
    mesh_a: wp.Mesh | None = ...,
    mesh_b: wp.Mesh | None = ...,
) -> tuple[odt.Array1dFloat32, odt.Array1dFloat32]: ...
@overload
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: bool,
    squared: bool = ...,
    mesh_a: wp.Mesh | None = ...,
    mesh_b: wp.Mesh | None = ...,
) -> _UnreducedChamfer: ...
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction | None = "mean",
    single_directional: bool = False,
    squared: bool = False,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> float | _UnreducedChamfer:
    """
    Chamfer distance between two triangle meshes (vertex-to-surface).

    The forward term is the distance from each vertex of mesh ``A`` to the
    surface of mesh ``B``; the backward term is the reverse. This mirrors the
    vertex-based approach of libigl while accumulating (mean/sum) instead of maximizing.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` vertices and ``(3 * n_faces_a,)`` flat faces of mesh ``A``.
    vertices_b, faces_b
        ``(n_vertices_b,)`` vertices and ``(3 * n_faces_b,)`` flat faces of mesh ``B``.
    point_reduction
        ``"mean"`` (default), ``"sum"``, ``"max"``, or ``None``. See
        [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points].
    single_directional
        If ``True``, only the ``A -> surface(B)`` term is computed.
    squared
        Square each distance before reducing, as ``pytorch3d`` does. Off by default: the result
        is then in squared units, and its ``"mean"`` is not a mean distance, which is easy to
        misread.
    mesh_a, mesh_b
        ``wp.Mesh`` objects already built over mesh ``A`` and mesh ``B``, to spare the BVH build
        each call otherwise pays per mesh queried (worth it when a mesh is measured repeatedly).
        Not checked against the arrays: the answer is for the geometry each holds.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is the cached one to pass.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        Same layout as [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points].
        The forward array is ``(n_vertices_a,)`` and the backward array ``(n_vertices_b,)``, except
        when any of ``vertices_a``, ``faces_a``, ``vertices_b`` or ``faces_b`` is empty -- a mesh
        with no faces counts as empty here even when its vertices are not -- in which case both
        arrays are length 0.

    Raises
    ------
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b``, ``faces_b``, ``mesh_a`` and ``mesh_b``
        are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"``, ``"sum"``, ``"max"`` or ``None``.

    See Also
    --------
    [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points]
    [`chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh]
    [`hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh]

    Examples
    --------
    A unit cube inside a cube of side 2: each inner corner is ``0.5`` from the outer surface, each
    outer corner ``sqrt(3) / 2`` from the inner one:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(extents=(2.0, 2.0, 2.0))
    forward = od.metrics.chamfer_mesh_to_mesh(
        vertices_a, faces_a, vertices_b, faces_b, single_directional=True
    )
    print(forward)  # 0.5
    symmetric = od.metrics.chamfer_mesh_to_mesh(vertices_a, faces_a, vertices_b, faces_b)
    print(f"{symmetric:.4f}")  # 1.3660
    ```
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    _validate_point_reduction(point_reduction)
    distances = _distances_mesh_to_mesh(
        vertices_a, faces_a, vertices_b, faces_b, single_directional, mesh_a, mesh_b
    )
    if distances is None:
        return _empty_chamfer(point_reduction, single_directional, vertices_a.device)
    return _chamfer(distances, point_reduction, single_directional, squared)


# ---------------------------------------------------------------------------
# Differentiable Chamfer losses
# ---------------------------------------------------------------------------
#
# The Chamfer functions above return a Python ``float`` (a host scalar) and are not
# differentiable. The ``*_loss`` variants below instead return a length-1 ``wp.float32``
# device array carrying the Chamfer loss (squared under ``squared``), so gradients
# can be back-propagated with a caller-owned ``wp.Tape``.
#
# Autodiff strategy (mirrors pytorch3d): the nearest-neighbor / closest-face assignment
# is a non-differentiable ``argmin`` and is computed *outside* the tape; the assignment
# is then held constant while the per-element distance is recomputed by a
# differentiable kernel (see [`ordito.kernels.metrics`]). Gradients flow to the point
# positions (and, for the surface terms, the mesh ``vertices``).
#
# Usage::
#
#     x = wp.array(..., dtype=wp.vec3, requires_grad=True)
#     y = wp.array(..., dtype=wp.vec3, requires_grad=True)
#     tape = wp.Tape()
#     loss = od.metrics.chamfer_points_to_points_loss(x, y, tape=tape)
#     tape.backward(loss=loss)
#     grad_x = x.grad  # dloss/dx
#
# Passing ``tape=None`` still returns the loss value but records nothing (no gradient).


def chamfer_points_to_points_loss(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    tape: wp.Tape | None = None,
    point_reduction: _DiffReduction = "mean",
    single_directional: bool = False,
    squared: bool = False,
) -> wp.array[wp.float32]:
    """
    Differentiable Chamfer loss between two point clouds.

    Squared-distance analogue of
    [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points] that returns
    a length-1 ``wp.float32`` device array instead of a host ``float``, so the loss can be
    back-propagated to ``x`` and ``y`` through a caller-owned ``wp.Tape``.

    The nearest-neighbor assignment (via
    [`query_nearest`][ordito.neighbors.query_nearest]) is computed
    outside ``tape`` and held constant during the backward pass, matching pytorch3d's
    ``chamfer_distance`` gradient.

    Parameters
    ----------
    x
        ``(n,)`` query point cloud. Set ``requires_grad=True`` for gradients.
    y
        ``(m,)`` target point cloud. Set ``requires_grad=True`` for gradients.
    tape
        Caller-owned ``wp.Tape`` into which the differentiable kernels are recorded. When
        ``None`` the loss is still computed but no operations are taped (no gradient).
    point_reduction
        ``"mean"`` (default) or ``"sum"``; reduces the per-point distances of each
        direction. ``"max"`` and ``None`` are not supported (see
        [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points]).
    single_directional
        If ``True``, only the ``x -> y`` term is accumulated.
    squared
        Square each distance before reducing, as ``pytorch3d`` does. Off by default: the result
        is then in squared units, and its ``"mean"`` is not a mean distance, which is easy to
        misread.

    Returns
    -------
    wp.array[wp.float32]
        ``(1,)`` device array holding the Chamfer loss (``requires_grad=True``). Empty
        inputs yield a length-1 zero array.

    Raises
    ------
    RuntimeError
        If ``x`` and ``y`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"`` or ``"sum"``.

    See Also
    --------
    [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points]
    [`chamfer_points_to_mesh_loss`][ordito.metrics.chamfer_points_to_mesh_loss]
    [`chamfer_mesh_to_mesh_loss`][ordito.metrics.chamfer_mesh_to_mesh_loss]

    Examples
    --------
    The point ``(1, 0, 0)`` is 1 from its nearest neighbour at the origin, so the mean loss is
    ``0.5`` and its gradient pushes it towards the origin with weight ``1 / 2``:

    ```python
    import warp as wp

    import ordito as od

    x = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True)
    y = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]], dtype=wp.vec3)
    tape = wp.Tape()
    loss = od.metrics.chamfer_points_to_points_loss(x, y, tape=tape, single_directional=True)
    tape.backward(loss=loss)
    print(loss.numpy())  # [0.5]
    print(x.grad.numpy()[1])  # [0.5 0.  0. ]
    ```
    """
    require_same_device(x=x, y=y)
    _validate_diff_reduction(point_reduction)
    device = x.device
    n = x.size
    m = y.size
    loss = _zero_loss(device)
    if n == 0 or m == 0:
        return loss

    # Non-differentiable nearest-neighbor indices (computed outside the tape). The backward search
    # is chosen from the forward one's own distances -- see `_backward_nearest`.
    nearest_xy, distance_xy = _nearest(y, x)
    terms: list[_ChamferTerm] = [_NnTerm(x, y, nearest_xy, _reduction_scale(point_reduction, n))]
    if not single_directional:
        nearest_yx = _backward_nearest(x, y, distance_xy, nearest_xy)[0][0]
        terms.append(_NnTerm(y, x, nearest_yx, _reduction_scale(point_reduction, m)))

    _accumulate_chamfer_terms(tape, terms, loss, squared=squared)
    return loss


def chamfer_points_to_mesh_loss(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    tape: wp.Tape | None = None,
    point_reduction: _DiffReduction = "mean",
    single_directional: bool = False,
    squared: bool = False,
    mesh: wp.Mesh | None = None,
) -> wp.array[wp.float32]:
    """
    Differentiable Chamfer loss between a point cloud and a triangle mesh.

    Squared-distance analogue of
    [`chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh]. The forward term
    is the exact point-to-surface distance (closest triangle held constant during
    backprop); the backward term is the distance from each mesh vertex to its
    nearest point in the cloud. Gradients flow to ``points`` **and** ``vertices``.

    Parameters
    ----------
    points
        ``(n,)`` query point cloud.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index array.
    tape
        Caller-owned ``wp.Tape`` for the backward pass, or ``None`` to skip taping.
    point_reduction
        ``"mean"`` (default) or ``"sum"``.
    single_directional
        If ``True``, only the ``points -> mesh surface`` term is accumulated.
    squared
        Square each distance before reducing, as ``pytorch3d`` does. Off by default: the result
        is then in squared units, and its ``"mean"`` is not a mean distance, which is easy to
        misread.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the BVH build each
        call otherwise pays (worth it when one mesh is measured repeatedly). Not checked against
        ``vertices`` / ``faces``: the answer is for the geometry ``mesh`` holds.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is the cached one to pass.

    Returns
    -------
    wp.array[wp.float32]
        ``(1,)`` device array holding the Chamfer loss (``requires_grad=True``).

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices``, ``faces`` and ``mesh`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"`` or ``"sum"``.

    See Also
    --------
    [`chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh]
    [`chamfer_points_to_points_loss`][ordito.metrics.chamfer_points_to_points_loss]
    [`chamfer_mesh_to_mesh_loss`][ordito.metrics.chamfer_mesh_to_mesh_loss]

    Examples
    --------
    A point ``1.0`` outside the ``x = 0.5`` face of a unit cube; the gradient of its distance points
    away from the face:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    points = wp.array([[1.5, 0.1, 0.2]], dtype=wp.vec3, requires_grad=True)
    tape = wp.Tape()
    loss = od.metrics.chamfer_points_to_mesh_loss(
        points, vertices, faces, tape=tape, single_directional=True
    )
    tape.backward(loss=loss)
    print(loss.numpy())  # [1.]
    print(points.grad.numpy()[0])  # [1. 0. 0.]
    ```
    """
    require_same_device(points=points, vertices=vertices, faces=faces, mesh=mesh)
    _validate_diff_reduction(point_reduction)
    device = points.device
    n = points.size
    v = vertices.size
    n_faces = faces.size // 3
    loss = _zero_loss(device)
    if n == 0 or v == 0 or n_faces == 0:
        return loss

    # Non-differentiable closest-face and nearest-neighbor assignment (outside the tape). The
    # backward search is chosen from the forward one's own distances -- see `_backward_nearest`.
    distance_forward, face_id = _closest_on_mesh(points, vertices, faces, mesh)
    terms: list[_ChamferTerm] = [
        _SurfaceTerm(points, vertices, faces, face_id, _reduction_scale(point_reduction, n))
    ]
    if not single_directional:
        nearest_vp = _backward_nearest(points, vertices, distance_forward, face_id, faces)[0][0]
        terms.append(_NnTerm(vertices, points, nearest_vp, _reduction_scale(point_reduction, v)))

    _accumulate_chamfer_terms(tape, terms, loss, squared=squared)
    return loss


def chamfer_mesh_to_mesh_loss(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    tape: wp.Tape | None = None,
    point_reduction: _DiffReduction = "mean",
    single_directional: bool = False,
    squared: bool = False,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> wp.array[wp.float32]:
    """
    Differentiable Chamfer loss between two triangle meshes (vertex-to-surface).

    Squared-distance analogue of
    [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh]. The forward term is
    the distance from each vertex of mesh ``A`` to the surface of mesh ``B`` (with
    the closest triangle held constant during backprop); the backward term is the reverse.
    Gradients flow to both meshes' vertices.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` vertices and ``(3 * n_faces_a,)`` flat faces of mesh ``A``.
    vertices_b, faces_b
        ``(n_vertices_b,)`` vertices and ``(3 * n_faces_b,)`` flat faces of mesh ``B``.
    tape
        Caller-owned ``wp.Tape`` for the backward pass, or ``None`` to skip taping.
    point_reduction
        ``"mean"`` (default) or ``"sum"``.
    single_directional
        If ``True``, only the ``A -> surface(B)`` term is accumulated.
    squared
        Square each distance before reducing, as ``pytorch3d`` does. Off by default: the result
        is then in squared units, and its ``"mean"`` is not a mean distance, which is easy to
        misread.
    mesh_a, mesh_b
        ``wp.Mesh`` objects already built over mesh ``A`` and mesh ``B``, to spare the BVH build
        each call otherwise pays per mesh queried (worth it when a mesh is measured repeatedly).
        Not checked against the arrays: the answer is for the geometry each holds.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is the cached one to pass.

    Returns
    -------
    wp.array[wp.float32]
        ``(1,)`` device array holding the Chamfer loss (``requires_grad=True``).

    Raises
    ------
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b``, ``faces_b``, ``mesh_a`` and ``mesh_b``
        are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"`` or ``"sum"``.

    See Also
    --------
    [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh]
    [`chamfer_points_to_points_loss`][ordito.metrics.chamfer_points_to_points_loss]
    [`chamfer_points_to_mesh_loss`][ordito.metrics.chamfer_points_to_mesh_loss]

    Examples
    --------
    A unit cube under a slab whose bottom face is the plane ``z = 1``: the cube's top corners are
    ``0.5`` from it and the bottom ones ``1.5``, a mean of 1, and every corner is pulled up with
    weight ``1 / 8``:

    ```python
    import warp as wp

    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(bounds=[[-2.0, -2.0, 1.0], [2.0, 2.0, 2.0]])
    vertices_a.requires_grad = True
    tape = wp.Tape()
    loss = od.metrics.chamfer_mesh_to_mesh_loss(
        vertices_a, faces_a, vertices_b, faces_b, tape=tape, single_directional=True
    )
    tape.backward(loss=loss)
    print(loss.numpy())  # [1.]
    print(set(vertices_a.grad.numpy()[:, 2].tolist()))  # {-0.125}
    ```
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    _validate_diff_reduction(point_reduction)
    device = vertices_a.device
    va = vertices_a.size
    vb = vertices_b.size
    n_faces_a = faces_a.size // 3
    n_faces_b = faces_b.size // 3
    loss = _zero_loss(device)
    if va == 0 or vb == 0 or n_faces_a == 0 or n_faces_b == 0:
        return loss

    # Non-differentiable closest-face assignment for each direction (outside the tape).
    _distances, face_ids = _closest_both_ways(
        vertices_a, faces_a, vertices_b, faces_b, single_directional, mesh_a, mesh_b
    )
    face_id_ab = odt.as_dense(face_ids[:va])
    terms: list[_ChamferTerm] = [
        _SurfaceTerm(
            vertices_a, vertices_b, faces_b, face_id_ab, _reduction_scale(point_reduction, va)
        )
    ]
    if not single_directional:
        face_id_ba = odt.as_dense(face_ids[va:])
        terms.append(
            _SurfaceTerm(
                vertices_b, vertices_a, faces_a, face_id_ba, _reduction_scale(point_reduction, vb)
            )
        )

    _accumulate_chamfer_terms(tape, terms, loss, squared=squared)
    return loss


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------


def hausdorff_points_to_points(
    x: wp.array[wp.vec3], y: wp.array[wp.vec3], *, single_directional: bool = False
) -> float:
    """
    Hausdorff distance between two point clouds.

    The directed distance ``d(x, y) = max_i min_j ||x_i - y_j||`` is computed (and
    symmetrized with ``d(y, x)`` unless ``single_directional``). Equivalent to
    symmetrizing [`scipy.spatial.distance.directed_hausdorff`][].

    Parameters
    ----------
    x
        ``(n,)`` query point cloud.
    y
        ``(m,)`` target point cloud.
    single_directional
        If ``True``, return only the directed distance ``d(x, y)``.

    Returns
    -------
    float
        The (symmetric or directed) Hausdorff distance. ``0.0`` for empty inputs.

    Raises
    ------
    RuntimeError
        If ``x`` and ``y`` are not all on one device.

    See Also
    --------
    [`hausdorff_points_to_mesh`][ordito.metrics.hausdorff_points_to_mesh]
    [`hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh]
    [`scipy.spatial.distance.directed_hausdorff`][]

    Examples
    --------
    Every point of ``x`` is within 1 of ``y``, but ``(3, 0, 0)`` is 2 from ``x``:

    ```python
    import warp as wp

    import ordito as od

    x = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    y = wp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.metrics.hausdorff_points_to_points(x, y))  # 2.0
    print(od.metrics.hausdorff_points_to_points(x, y, single_directional=True))  # 1.0
    ```
    """
    require_same_device(x=x, y=y)
    distances = _distances_points_to_points(x, y, single_directional)
    if distances is None:
        return 0.0
    return _hausdorff(distances, single_directional)


def hausdorff_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    single_directional: bool = False,
    mesh: wp.Mesh | None = None,
) -> float:
    """
    Hausdorff distance between a point cloud and a triangle mesh.

    The forward distance is the maximum over points of the distance to the mesh
    surface; the backward distance is the maximum over mesh vertices of the distance
    to the nearest point in the cloud.

    Parameters
    ----------
    points
        ``(n,)`` query point cloud.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index array.
    single_directional
        If ``True``, return only the directed ``points -> mesh surface`` distance.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the BVH build each
        call otherwise pays (worth it when one mesh is measured repeatedly). Not checked against
        ``vertices`` / ``faces``: the answer is for the geometry ``mesh`` holds.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is the cached one to pass.

    Returns
    -------
    float
        The (symmetric or directed) Hausdorff distance. ``0.0`` for empty inputs.

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices``, ``faces`` and ``mesh`` are not all on one device.

    See Also
    --------
    [`hausdorff_points_to_points`][ordito.metrics.hausdorff_points_to_points]
    [`hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh]

    Examples
    --------
    A point ``1.0`` outside a unit cube and one at its centre. The farthest cube corner from the
    cloud is ``sqrt(3) / 2`` from the centre point, so the symmetric value is still 1:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    points = wp.array([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3)
    directed = od.metrics.hausdorff_points_to_mesh(points, vertices, faces, single_directional=True)
    print(directed)  # 1.0
    print(f"{od.metrics.hausdorff_points_to_mesh(points, vertices, faces):.4f}")  # 1.0000
    ```
    """
    require_same_device(points=points, vertices=vertices, faces=faces, mesh=mesh)
    distances = _distances_points_to_mesh(points, vertices, faces, single_directional, mesh)
    if distances is None:
        return 0.0
    return _hausdorff(distances, single_directional)


def hausdorff_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    single_directional: bool = False,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> float:
    """
    Hausdorff distance between two triangle meshes.

    Direct port of libigl's ``igl::hausdorff`` (vertex-based overload): the distance
    from every vertex of one mesh to the surface of the other is computed in both
    directions and the overall maximum is returned. This equals
    ``sqrt(max(d_ba, d_ab))`` in libigl, where ``d_*`` are squared distances.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` vertices and ``(3 * n_faces_a,)`` flat faces of mesh ``A``.
    vertices_b, faces_b
        ``(n_vertices_b,)`` vertices and ``(3 * n_faces_b,)`` flat faces of mesh ``B``.
    single_directional
        If ``True``, return only the directed ``A -> surface(B)`` distance.
    mesh_a, mesh_b
        ``wp.Mesh`` objects already built over mesh ``A`` and mesh ``B``, to spare the BVH build
        each call otherwise pays per mesh queried (worth it when a mesh is measured repeatedly).
        Not checked against the arrays: the answer is for the geometry each holds.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is the cached one to pass.

    Returns
    -------
    float
        The (symmetric or directed) Hausdorff distance. ``0.0`` for empty inputs.

    Raises
    ------
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b``, ``faces_b``, ``mesh_a`` and ``mesh_b``
        are not all on one device.

    See Also
    --------
    [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh]
    [`hausdorff_points_to_mesh`][ordito.metrics.hausdorff_points_to_mesh]

    Examples
    --------
    A unit cube inside a cube of side 2: the inner corners are ``0.5`` from the outer surface, the
    outer corners ``sqrt(3) / 2`` from the inner one:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(extents=(2.0, 2.0, 2.0))
    directed = od.metrics.hausdorff_mesh_to_mesh(
        vertices_a, faces_a, vertices_b, faces_b, single_directional=True
    )
    print(directed)  # 0.5
    symmetric = od.metrics.hausdorff_mesh_to_mesh(vertices_a, faces_a, vertices_b, faces_b)
    print(f"{symmetric:.4f}")  # 0.8660
    ```
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    distances = _distances_mesh_to_mesh(
        vertices_a, faces_a, vertices_b, faces_b, single_directional, mesh_a, mesh_b
    )
    if distances is None:
        return 0.0
    return _hausdorff(distances, single_directional)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
#
# Every helper below is shared by two or three of the public entry points above -- the
# validation guards and reductions by all three ``chamfer_*`` functions, the three geometry
# dispatches by a ``chamfer_*`` and a ``hausdorff_*`` pair each, and the launch/tape machinery
# by the ``*_loss`` family -- so they are grouped in one trailing block rather than placed after
# any single caller.


def _validate_point_reduction(point_reduction: _PointReduction | None) -> None:
    if point_reduction is not None and point_reduction not in ("mean", "sum", "max"):
        raise ValueError('point_reduction must be one of "mean", "sum", "max" or None')


def _chamfer(
    distances: _Distances,
    point_reduction: _PointReduction | None,
    single_directional: bool,
    squared: bool,
) -> float | _UnreducedChamfer:
    """
    Combine forward/backward Euclidean distances into a Chamfer value.

    Under ``squared`` the distances are squared element-wise before reduction (the ``pytorch3d``
    convention), and a reduced value never materializes the squared array: ``"sum"`` and
    ``"mean"`` square each term inside [`_totals_of_both`][ordito.metrics._totals_of_both], and
    ``"max"`` squares the largest distance -- in ``float32``, which is exact rather than
    approximate: rounding is monotone, so the largest rounded square is the rounded square of the
    largest distance. Otherwise they are reduced as they are.
    """
    backward = cast(wp.array[wp.float32], distances.backward)
    if point_reduction is None:
        if not squared:
            forward = cast(odt.Array1dFloat32, distances.forward)
            return forward if single_directional else (forward, cast(odt.Array1dFloat32, backward))
        sq_forward = _square(distances.forward)
        if single_directional:
            return sq_forward
        return sq_forward, _square(backward)

    if point_reduction != "max":
        total_forward, total_backward = _totals_of_both(
            distances.forward, None if single_directional else backward, squared
        )
        if point_reduction == "mean":
            total_forward /= distances.forward.size
            if not single_directional:
                total_backward /= backward.size
        return total_forward + total_backward

    if not single_directional and distances.forward_max is None:
        largest = _max_of_both(distances.forward, backward)
    else:
        largest = distances.forward_max
        if largest is None:
            largest = od.reduce.max(cast(odt.Array1dFloat32, distances.forward))
        if not single_directional:
            largest = max(largest, od.reduce.max(cast(odt.Array1dFloat32, backward)))
    return float(np.float32(largest) * np.float32(largest)) if squared else largest


def _square(distances: wp.array[wp.float32]) -> odt.Array1dFloat32:
    n = distances.size
    squared = odt.empty_1d(n, wp.float32, device=distances.device)
    if n > 0:
        _launch.map(kernel_array.square_scalar, distances, out=squared)
    return squared


def _hausdorff(distances: _Distances, single_directional: bool) -> float:
    """
    Directed (or symmetric) Hausdorff distance from Euclidean distances.

    ``max(d)`` over Euclidean per-element distances equals ``sqrt(max(d**2))``,
    so this matches libigl's ``sqrt(max(dba, dab))`` without squaring.
    """
    directed_forward = distances.forward_max
    if directed_forward is None and not single_directional:
        return _max_of_both(distances.forward, cast(wp.array[wp.float32], distances.backward))
    if directed_forward is None:
        directed_forward = od.reduce.max(cast(odt.Array1dFloat32, distances.forward))
    if single_directional:
        return directed_forward
    directed_backward = od.reduce.max(cast(odt.Array1dFloat32, distances.backward))
    return max(directed_forward, directed_backward)


def _max_of_both(forward: wp.array[wp.float32], backward: wp.array[wp.float32]) -> float:
    """
    Largest of two non-empty distance arrays in one readback, for the symmetric ``"max"`` forms.

    Both arrays fold into one seeded slot in one launch (an ``atomic_max`` per block), so the
    symmetric maximum is the same float as the larger of two [`reduce.max`][ordito.reduce.max]
    calls, at one launch and one sync rather than two of each.
    """
    device = forward.device
    out = _launch.full(1, -math.inf, dtype=wp.float32, device=device)
    forward_blocks = kernel_reduce.blocks_1d(forward.size)
    _launch.launch_tiled(
        kernel_metrics.max_of_pair,
        dim=[forward_blocks + kernel_reduce.blocks_1d(backward.size)],
        inputs=[forward, backward, forward_blocks],
        outputs=[out],
        block_dim=TILE_1D,
        device=device,
    )
    return float(out.numpy()[0])  # the reduced value is the function's answer


def _totals_of_both(
    forward: wp.array[wp.float32], backward: wp.array[wp.float32] | None, squared: bool
) -> tuple[float, float]:
    """
    Totals of non-empty distance arrays (of their squares under ``squared``) in one readback.

    The ``"sum"`` / ``"mean"`` counterpart of [`_max_of_both`][ordito.metrics._max_of_both]: both
    arrays fold in one launch into the two result slots of one block-partials buffer, in a fixed
    order under [`is_deterministic`][ordito.config.is_deterministic], so both come back in one
    copy. ``backward=None`` (a directed Chamfer value) totals ``forward`` alone, with ``0.0``
    for the second total.
    """
    device = forward.device
    forward_blocks = kernel_reduce.blocks_1d(forward.size)
    blocks = forward_blocks
    if backward is not None:
        blocks += kernel_reduce.blocks_1d(backward.size)
    totals, stride = _launch.block_partials(blocks, 2, wp.float32, device)
    _launch.launch_tiled(
        kernel_metrics.totals_of_pair,
        dim=[blocks],
        inputs=[
            forward,
            forward if backward is None else backward,
            forward_blocks,
            int(squared),
            stride,
        ],
        outputs=[totals],
        block_dim=TILE_1D,
        device=device,
    )
    # The two totals are the function's answer.
    total_forward, total_backward = read_values(totals, 0, 2)
    return float(total_forward), float(total_backward)


def _empty_chamfer(
    point_reduction: _PointReduction | None, single_directional: bool, device: wp.DeviceLike
) -> float | _UnreducedChamfer:
    if point_reduction is not None:
        return 0.0
    if single_directional:
        return odt.empty_1d(0, wp.float32, device=device)
    return (odt.empty_1d(0, wp.float32, device=device), odt.empty_1d(0, wp.float32, device=device))


# The three geometry dispatches below are each shared by a ``chamfer_*`` and a ``hausdorff_*``
# entry point, which differ only in how they reduce the pair and what they return for degenerate
# input. ``None`` means degenerate, leaving that choice to the caller.


def _distances_points_to_points(
    x: wp.array[wp.vec3], y: wp.array[wp.vec3], single_directional: bool
) -> _Distances | None:
    """Nearest-neighbour distances between two clouds, both directions unless directed."""
    if x.size == 0 or y.size == 0:
        return None
    nearest_forward, distance_forward = _nearest(y, x)
    d_forward = distance_forward
    if single_directional:
        return _Distances(d_forward, None, None)
    (_, d_backward), forward_max = _backward_nearest(x, y, d_forward, nearest_forward)
    return _Distances(d_forward, d_backward, forward_max)


def _distances_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    single_directional: bool,
    mesh: wp.Mesh | None,
) -> _Distances | None:
    """Point-to-surface distances forward, cloud nearest-neighbour distances back."""
    if points.size == 0 or vertices.size == 0 or faces.size == 0:
        return None
    d_forward, face_id = _closest_on_mesh(points, vertices, faces, mesh)
    if single_directional:
        return _Distances(d_forward, None, None)
    # The forward half is point-to-*surface*, a lower bound on the point-to-vertex answer the
    # backward search wants -- still the right scale to choose it by.
    (_, d_backward), forward_max = _backward_nearest(points, vertices, d_forward, face_id, faces)
    return _Distances(d_forward, d_backward, forward_max)


def _nearest(
    points: wp.array[wp.vec3], queries: wp.array[wp.vec3]
) -> tuple[odt.Array1dInt32, odt.Array1dFloat32]:
    """
    Every query's nearest point: the ``k = 1`` search each direction of a cloud pair runs.

    Under ``backend="bvh"`` [`query_nearest`][ordito.neighbors.query_nearest] answers ``k = 1``
    with a nearest-point descent of a point tree, whose cost does not depend on how far the two
    clouds sit apart and which needs no search radius -- so neither the cloud's
    bounding box and density estimate nor the forward answer's largest distance (the hash grid's
    seed for the backward half) is reduced and read back. The distances are the hash grid's to the
    bit. It is the faster of the two from coincident clouds to clouds several spacings apart, with
    one exception: a coincident pair of clouds of several hundred thousand points, where building
    the tree costs more than the grid's walk.
    """
    return od.neighbors.query_nearest(points, queries, k=1, backend="bvh")


def _backward_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    forward_distances: odt.ArrayNdFloat32,
    forward_assignment: odt.ArrayNdInt32,
    faces: wp.array[wp.int32] | None = None,
) -> tuple[tuple[odt.Array1dInt32, odt.Array1dFloat32], float | None]:
    """
    Run the backward half's ``k = 1`` search, choosing the search from the forward half's answer.

    ``forward_distances`` and ``forward_assignment`` are the forward half's answer at each point
    of ``points``: how far it lies from the other side, and which query (or, with ``faces``, which
    triangle of queries) it found. When the largest distance is small against ``points``' own
    spacing *and* nearly every query is some point's answer, the pair is coincident or nearly so,
    and a hash grid seeded at that scale answers the backward rows from its first ring of cells --
    without the tree build the closest-point descent needs, which on a large cloud is most of
    that search's cost. Otherwise the descent (see [`_nearest`][ordito.metrics._nearest]) is
    kept: a grid walk grows with the distance to the answer and the descent's does not. The
    second test is what catches a pair whose forward answer is small only because ``points``
    covers part of ``queries``; the rows off the covered part are far. Both searches are exact,
    so the distances are the same either way; which of two points at *exactly* the same distance
    is reported is not specified by either search, and may differ between them.

    Only a cloud of at least ``_GRID_BACKWARD_MIN_POINTS`` on a CUDA device is considered. The
    choice costs a readback, which also holds the backward search's host-side issue back until
    the forward one has finished, and below that size the descent's build is cheap enough that
    this costs more than the grid saves; above it the readback is still paid, as a few percent,
    by a pair that is not coincident. On the CPU the grid is not the faster search even on a
    coincident pair.

    The seed is the largest forward distance held no lower than a fraction of
    [`knn_initial_radius`][ordito.neighbors.knn_initial_radius]: under ``"hashgrid"`` it is also
    the cell width, and a cell much wider than the answer makes every row scan more of the cloud.
    A row the grid cannot certify within its widest walk is still answered by the descent, inside
    [`query_nearest`][ordito.neighbors.query_nearest], so a wrong choice costs time and never the
    result.

    Returns the search's ``(indices, distances)`` and the largest forward distance when the choice
    read it back (a ``"max"`` reduction or a Hausdorff distance reuses it), else ``None``.
    """
    n = points.size
    device = points.device
    if not wp.get_device(device).is_cuda or n < _GRID_BACKWARD_MIN_POINTS:
        return _nearest(points, queries), None
    m = queries.size
    scale = _launch.array(_SCALE_SEED, dtype=wp.float32, device=device)
    hits = _launch.zeros(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_metrics.pair_search_scale,
        dim=kernel_reduce.chunks_1d(n),
        inputs=[points, forward_distances, forward_assignment, faces, int(faces is not None), hits],
        outputs=[scale],
        device=device,
    )
    # The one readback the choice costs: the box the grid needs, the forward maximum and the hit
    # count, together.
    values = scale.numpy()
    # ``0.0 - x`` rather than ``-x``: a zero maximum is stored as ``-0.0``, and a distance reads
    # ``0.0``.
    forward_max = 0.0 - float(values[6])
    # Slots 3..5 hold the *negated* upper corner, decoded as ``reduce.minmax`` decodes it.
    bounds = (wp.vec3(*values[:3]), wp.vec3(*(-values[3:6])))
    spacing = od.neighbors.knn_initial_radius(points, 1, bounds=bounds)
    coincident = (
        math.isfinite(forward_max)
        and forward_max <= _GRID_BACKWARD_CEILING * spacing
        and float(values[7]) >= _GRID_BACKWARD_MIN_HITS * m
    )
    if coincident:
        seed = max(forward_max, _GRID_BACKWARD_FLOOR * spacing)
        found = od.neighbors.query_nearest(points, queries, k=1, initial_radius=seed, bounds=bounds)
        return found, forward_max
    return _nearest(points, queries), forward_max


def _distances_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    single_directional: bool,
    mesh_a: wp.Mesh | None,
    mesh_b: wp.Mesh | None,
) -> _Distances | None:
    """
    Each mesh's vertices to the other's surface. Vertex-sampled, not a true surface metric.

    Both directions are one launch ([`_closest_both_ways`][ordito.metrics._closest_both_ways]);
    the distances are views of its one buffer.
    """
    if vertices_a.size == 0 or vertices_b.size == 0 or faces_a.size == 0 or faces_b.size == 0:
        return None
    distances, _faces = _closest_both_ways(
        vertices_a, faces_a, vertices_b, faces_b, single_directional, mesh_a, mesh_b
    )
    n_a = vertices_a.size
    d_forward = odt.as_dense(distances[:n_a])
    d_backward = None if single_directional else odt.as_dense(distances[n_a:])
    return _Distances(d_forward, d_backward, None)


def _closest_both_ways(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    single_directional: bool,
    mesh_a: wp.Mesh | None,
    mesh_b: wp.Mesh | None,
) -> tuple[odt.Array1dFloat32, odt.Array1dInt32]:
    """
    Each mesh's vertices to the other's surface: closest distances and faces, in one launch.

    ``(n_a + n_b,)`` rows, ``vertices_a`` against B first and then ``vertices_b`` against A (only
    the first ``n_a`` under ``single_directional``), each exactly what
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] returns for it. A mesh not
    supplied is built over its non-empty buffers; it aliases them and is discarded, so it needs no
    copy.
    """
    device = vertices_a.device
    if mesh_b is None:
        mesh_b = wp.Mesh(points=vertices_b, indices=faces_b)
    n_rows = vertices_a.size
    # A directed call launches over ``vertices_a`` alone, so the second pair is never read.
    second_mesh, second_points = mesh_b, vertices_b
    if not single_directional:
        if mesh_a is None:
            mesh_a = wp.Mesh(points=vertices_a, indices=faces_a)
        second_mesh, second_points = mesh_a, vertices_b
        n_rows += vertices_b.size
    distances = odt.empty_1d(n_rows, wp.float32, device=device)
    faces = odt.empty_1d(n_rows, wp.int32, device=device)
    _launch.launch(
        kernel_proximity.closest_faces_both_ways,
        dim=n_rows,
        inputs=[mesh_b.id, vertices_a, second_mesh.id, second_points],
        outputs=[distances, faces],
        device=device,
    )
    return distances, faces


def _closest_on_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    mesh: wp.Mesh | None,
) -> tuple[odt.Array1dFloat32, odt.Array1dInt32]:
    """``closest_point_on_mesh``'s distances and faces for ``points``, without the points."""
    # The directed form of the pair: ``points`` stand in for mesh A's vertices, whose faces it
    # never reads.
    return _closest_both_ways(points, faces, vertices, faces, True, None, mesh)


def _validate_diff_reduction(point_reduction: _DiffReduction) -> None:
    if point_reduction not in ("mean", "sum"):
        raise ValueError(
            'Differentiable Chamfer losses support point_reduction "mean" or "sum" only. '
            'Use the non-differentiable chamfer_* functions for "max"/None.'
        )


def _reduction_scale(point_reduction: _DiffReduction, count: int) -> float:
    return (1.0 / count) if point_reduction == "mean" else 1.0


def _zero_loss(device: wp.DeviceLike) -> wp.array[wp.float32]:
    # Rank ``int``, not ``Literal[1]``: the loss is handed to ``wp.Tape.backward``, which takes a
    # plain ``wp.array``, and rank is invariant.
    return _launch.zeros(1, dtype=wp.float32, device=device, requires_grad=True)


def _accumulate_chamfer_terms(
    tape: wp.Tape | None, terms: list[_ChamferTerm], loss: wp.array[wp.float32], *, squared: bool
) -> None:
    """
    Accumulate every term into ``loss``, recording the launches on ``tape`` when there is one.

    Each of the three ``chamfer_*_loss`` entry points hands over one term per direction it
    accumulates (one when ``single_directional``, two otherwise). Recording is opt-in because a
    ``wp.Tape`` context is only wanted when the caller intends to backpropagate; passing
    ``tape=None`` still runs every term, just untaped.

    On CUDA each term commits one atomic per block, and its adjoint scatters a gathered operand's
    gradient atomically; under ``ordito.config``'s deterministic setting
    [`_ordered_chamfer_terms`][ordito.metrics._ordered_chamfer_terms] takes the fixed-order form of
    both instead. The CPU device runs the sliced kernels' threads in turn, so they repeat either
    way and keep their form.
    """
    if is_deterministic() and prefers_tiled_reduction(loss.device):
        record = _ordered_chamfer_terms(terms, loss, squared)
    else:

        def record() -> None:
            for term in terms:
                if isinstance(term, _NnTerm):
                    _launch_nn_term(*term, loss, squared=squared)
                else:
                    _launch_surface_term(*term, loss, squared=squared)

    if tape is not None:
        with tape:
            record()
    else:
        record()


def _launch_nn_term(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: odt.ArrayNdInt32,
    scale: float,
    loss: wp.array[wp.float32],
    *,
    squared: bool,
) -> None:
    """Accumulate the point-to-point Chamfer term for ``x`` into ``loss``."""
    _launch_reduction_pair(
        kernel_metrics.chamfer_nn_term_tiled,
        kernel_metrics.chamfer_nn_term_sliced,
        x.size,
        x.device,
        [x, y, nearest, wp.float32(scale), wp.int32(squared)],
        loss,
    )


def _launch_surface_term(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: odt.ArrayNdInt32,
    scale: float,
    loss: wp.array[wp.float32],
    *,
    squared: bool,
) -> None:
    """Accumulate the point-to-surface Chamfer term for ``points`` into ``loss``."""
    _launch_reduction_pair(
        kernel_metrics.chamfer_surface_term_tiled,
        kernel_metrics.chamfer_surface_term_sliced,
        points.size,
        points.device,
        [points, vertices, faces, face_id, wp.float32(scale), wp.int32(squared)],
        loss,
    )


def _launch_reduction_pair(
    kernel_tiled: odt.Kernel,
    kernel_sliced: odt.Kernel,
    n: int,
    device: wp.DeviceLike,
    common_inputs: list[object],
    loss: wp.array[wp.float32],
) -> None:
    """
    Dispatch one of a matched ``*_tiled`` / ``*_sliced`` kernel pair over ``n`` items into ``loss``.

    The block-reducing ``*_tiled`` kernel runs on CUDA, the portable strided-slice one on CPU, where
    ``wp.launch_tiled`` runs a single lane per block (see
    [`prefers_tiled_reduction`][ordito._device.prefers_tiled_reduction]). ``common_inputs`` is
    every argument both kernels share, in the order they declare it, before the sliced kernel's
    extra ``slices`` count and both kernels' trailing ``loss``.

    Shared by [`_launch_nn_term`][ordito.metrics._launch_nn_term] and
    [`_launch_surface_term`][ordito.metrics._launch_surface_term] (its last caller), whose only
    difference is which kernel pair and argument list they carry.
    """
    if prefers_tiled_reduction(device):
        _launch.launch_tiled(
            kernel_tiled,
            dim=[(n + TILE_1D - 1) // TILE_1D],
            inputs=[*common_inputs, loss],
            block_dim=TILE_1D,
            device=device,
        )
    else:
        slices = slice_count(n, device)
        _launch.launch(
            kernel_sliced,
            dim=slices,
            inputs=[*common_inputs, wp.int32(slices), loss],
            device=device,
        )


def _ordered_chamfer_terms(
    terms: list[_ChamferTerm], loss: wp.array[wp.float32], squared: bool
) -> Callable[[], None]:
    """
    Prepare the fixed-order form of ``terms`` and return the closure that writes ``loss``.

    Everything the adjoint never needs -- the stable sorts that turn each gathered index into
    per-source runs, and the corner keys they sort -- runs here, outside any tape. The closure runs
    the differentiable launches: the pulls that copy each gathered operand out per query
    ([`gather_by_sorted_key`][ordito.kernels.metrics.gather_by_sorted_key]), the per-query terms
    into one buffer at each term's offset, and the two fixed-order passes of
    [`sum_in_blocks`][ordito.kernels.metrics.sum_in_blocks] from that buffer into ``loss``. Each
    adjoint is then a gather or a one-writer write, so the loss and every gradient repeat bit for
    bit; the order of the sum differs from the arrival-order kernels', so the bits differ from
    theirs.
    """
    device = loss.device
    counts = [term.x.size if isinstance(term, _NnTerm) else term.points.size for term in terms]
    total = sum(counts)
    values = _launch.empty(total, dtype=wp.float32, device=device, requires_grad=True)
    launches: list[Callable[[], None]] = []
    offset = 0
    for term, count in zip(terms, counts, strict=True):
        if isinstance(term, _NnTerm):
            sorted_keys, order = od.array.sort_and_argsort(
                term.nearest, end_bit=max(1, (term.y.size - 1).bit_length())
            )
            nearest, pull = _pull_gather(term.y, sorted_keys, order, count)
            launches += [
                pull,
                _launcher(
                    kernel_metrics.chamfer_nn_terms,
                    count,
                    [term.x, nearest, wp.float32(term.scale), wp.int32(squared), offset],
                    values,
                ),
            ]
        else:
            corners, pull = _pull_gather(
                term.vertices,
                *_corner_runs(term.faces, term.face_id, term.vertices.size),
                3 * count,
            )
            launches += [
                pull,
                _launcher(
                    kernel_metrics.chamfer_surface_terms,
                    count,
                    [
                        term.points,
                        corners,
                        term.face_id,
                        wp.float32(term.scale),
                        wp.int32(squared),
                        offset,
                    ],
                    values,
                ),
            ]
        offset += count
    blocks = kernel_reduce.blocks_1d(total)
    partials = _launch.empty(blocks, dtype=wp.float32, device=device, requires_grad=True)

    def record() -> None:
        for launch in launches:
            launch()
        for source, width, out in (
            (values, int(kernel_reduce.ITEMS_PER_BLOCK_1D), partials),
            (partials, blocks, loss),
        ):
            _launch.launch_tiled(
                kernel_metrics.sum_in_blocks,
                dim=[out.size],
                inputs=[source, width, out],
                block_dim=TILE_1D,
                device=device,
            )

    return record


def _corner_runs(
    faces: wp.array[wp.int32], face_id: odt.ArrayNdInt32, n_vertices: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Sort the vertex under each corner ``3 i + k`` of query ``i``'s face, returning keys and order.

    A query with no face keys its corners ``n_vertices``, past every run a pull reads.
    """
    device = faces.device
    n = face_id.size
    keys = _launch.empty(6 * n, dtype=wp.int32, device=device)
    order = _launch.empty(6 * n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_metrics.surface_corner_keys,
        dim=n,
        inputs=[faces, face_id, n_vertices],
        outputs=[keys, order],
        device=device,
    )
    return od.array.sort_and_argsort_in_place(keys, order, end_bit=n_vertices.bit_length())


def _pull_gather(
    values: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.int32],
    order: wp.array[wp.int32],
    count: int,
) -> tuple[wp.array[wp.vec3], Callable[[], None]]:
    """Allocate ``values`` gathered through a sorted index; return it and the launch filling it."""
    gathered = _launch.empty(count, dtype=wp.vec3, device=values.device, requires_grad=True)
    return gathered, _launcher(
        kernel_metrics.gather_by_sorted_key, values.size, [values, sorted_keys, order], gathered
    )


def _launcher(
    kernel: odt.Kernel,
    dim: int,
    inputs: list[object],
    out: wp.array[wp.float32] | wp.array[wp.vec3],
) -> Callable[[], None]:
    """Return a closure launching ``kernel`` over ``dim`` threads with ``inputs`` then ``out``."""

    def launch() -> None:
        _launch.launch(kernel, dim=dim, inputs=inputs, outputs=[out], device=out.device)

    return launch
