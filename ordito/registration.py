"""Point cloud registration: Procrustes analysis and ICP."""

from __future__ import annotations

import math
from typing import Literal, TypedDict, TypeVar, cast, overload

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import (
    read_scalar,
    record_device_loop,
    require_nonempty_mesh,
    require_same_device,
)
from ordito.constants import TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import registration as kernel_registration
from ordito.kernels import transform as kernel_transform

DType = TypeVar("DType")


# ``return_cost`` is keyword-only, and that is what lets the overload set be unambiguous. The
# ``Literal[True]`` overload must come first because ``True`` is the runtime default and overload
# resolution picks the first match -- listing ``Literal[False]`` first would type the bare
# ``procrustes(a, b)`` call as returning the matrix alone, when it in fact returns the three-tuple.
# While ``return_cost`` was positional it also had to carry a default in the ``False`` overload
# (a non-defaulted parameter cannot follow defaulted ones), which made a no-argument call match
# both overloads with incompatible return types. Keyword-only removes the default, and with it
# the overlap.
@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    *,
    return_cost: Literal[True] = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]: ...
@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    *,
    return_cost: Literal[False],
) -> wp.array[wp.mat44]: ...
@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    *,
    return_cost: bool = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float] | wp.array[wp.mat44]: ...
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    *,
    return_cost: bool = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float] | wp.array[wp.mat44]:
    """
    Find the optimal transform mapping point cloud *a* onto *b*.

    Warp port of [`trimesh.registration.procrustes`][]. Accepts and returns
    GPU-resident arrays; no NumPy in the hot path.

    Parameters
    ----------
    a, b:
        ``(n,)`` corresponding point clouds.
    weights:
        ``(n,)`` per-point weights.
        Uniform weights are assumed when ``None``.
    reflection:
        Allow reflections in the rotation component.
    translation:
        Allow translation.
    scale:
        Allow uniform scaling.
    return_cost:
        Keyword-only. When ``True`` (the default) also return the transformed points and cost.

    Returns
    -------
    matrix:
        ``(1,)`` array encoding the transform.
    transformed:
        ``(n,)`` image of *a* under the transform.
        Only returned when ``return_cost=True``.
    cost:
        Weighted mean of squared distances between *transformed* and *b*.
        Only returned when ``return_cost=True``.

    Raises
    ------
    RuntimeError
        If ``a``, ``b`` and ``weights`` are not all on one device.
    ValueError
        If ``b`` (or a non-empty ``weights``) does not have the same length as ``a``, or a
        non-empty ``weights`` sums to zero.

    Notes
    -----
    Latency-bound at every size that matters: the fit is two kernel launches (four with
    ``return_cost``), one allocation for the packed moment accumulator and one readback, so the
    cost is nearly flat in ``n``. [`icp`][ordito.registration.icp] runs the same fit inside a
    device-side loop, with no readback per iteration.

    **The returned matrix is column-vector, which is the transpose of pytorch3d's.**
    ``pytorch3d.ops.corresponding_points_alignment`` solves the row-vector form ``s X R + T = Y``,
    so its ``R`` is this matrix's linear block divided by the scale and *transposed*.
    """
    require_same_device(a=a, b=b, weights=weights)
    n = a.size
    device = a.device
    if b.size != n:
        raise ValueError(f"a and b must have the same length, got {n} and {b.size}")
    # A zero-length weights array is the kernels' own "uniform weights" sentinel (see
    # ``_zero_length``), not a caller mistake -- only a non-empty mismatch is a real error.
    if weights is not None and weights.size not in (0, n):
        raise ValueError(f"weights must have the same length as a, got {weights.size} and {n}")

    if n == 0:
        matrix = _identity_mat44(device)
        if not return_cost:
            return matrix
        return matrix, _launch.clone(a), 0.0

    workspace: _ProcrustesWorkspace = {
        "acc": _launch.zeros(
            kernel_registration.PROCRUSTES_ACC_SIZE, dtype=wp.float32, device=device
        ),
        "matrix": _launch.empty(1, dtype=wp.mat44, device=device),
        "transformed": _launch.empty(n, dtype=wp.vec3, device=device) if return_cost else None,
        "uniform_weights": _zero_length(wp.float32, device),
    }
    weighted = weights is not None and weights.size == n
    result = _procrustes_into(
        a, b, weights, reflection, translation, scale, return_cost, workspace, weighted
    )

    # An all-zero non-empty ``weights`` divides by zero inside the kernel (the accumulated weight
    # sum is the denominator of every centroid and, with ``scale=True``, of the singular values
    # too), silently returning a matrix of NaN with no exception. The fit *above* already
    # accumulates that sum, so the check reads it back rather than running a reduction over
    # ``weights`` first -- the fit it would have skipped is two launches, against a launch, an
    # allocation and a host sync for the reduction. Its NaN output is discarded by the raise.
    if not return_cost:
        matrix = cast(wp.array[wp.mat44], result)
        if weighted and float(read_scalar(workspace["acc"], kernel_registration.ACC_W_SUM)) == 0.0:
            raise ValueError("weights sum to zero: no point carries any weight")
        return matrix
    matrix, transformed, cost, weight_sum = cast(
        tuple[wp.array[wp.mat44], wp.array[wp.vec3], float, float], result
    )
    if weighted and weight_sum == 0.0:
        raise ValueError("weights sum to zero: no point carries any weight")
    return matrix, transformed, cost


class _ProcrustesWorkspace(TypedDict):
    """Buffers a [`procrustes`][ordito.registration.procrustes] fit writes into."""

    acc: wp.array[wp.float32]
    matrix: wp.array[wp.mat44]
    transformed: wp.array[wp.vec3] | None
    uniform_weights: wp.array[wp.float32]


def _procrustes_into(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None,
    reflection: bool,
    translation: bool,
    scale: bool,
    return_cost: bool,
    workspace: _ProcrustesWorkspace,
    need_weight_sum: bool = False,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float, float] | wp.array[wp.mat44]:
    """Run one Procrustes fit into caller-owned buffers. See ``procrustes`` for the semantics."""
    n = a.size
    device = a.device
    acc = workspace["acc"]
    out_matrix = workspace["matrix"]
    if weights is None:
        weights = workspace["uniform_weights"]

    _launch.launch_tiled(
        kernel_registration.accumulate_procrustes_moments,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[a, b, weights, translation, acc],
        block_dim=TILE_1D,
        device=device,
    )
    if not return_cost:
        _launch.launch(
            kernel_registration.build_procrustes_matrix,
            dim=1,
            inputs=[a, b, acc, reflection, translation, scale, out_matrix],
            device=device,
        )
        return out_matrix

    out_transformed = workspace["transformed"]
    assert out_transformed is not None
    # One launch, not three: the fit, the moved points and the residual reduced against them.
    _launch.launch_tiled(
        kernel_registration.fit_transform_and_accumulate_cost,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[a, b, weights, acc, reflection, translation, scale, out_matrix, out_transformed],
        block_dim=TILE_1D,
        device=device,
    )
    # One readback for the whole accumulator; ICP's convergence test needs the cost on the host,
    # and an extra device-side pass would cost more than the readback. ``ACC_W_SUM`` rides along in
    # the same transfer: ``icp``'s "every correspondence was rejected" guard needs it, and reading
    # it here rather than reducing the weight array separately is one host sync per iteration
    # instead of two -- but only when ``need_weight_sum`` says a caller wants it.
    cost_slot = int(kernel_registration.ACC_COST)
    if not need_weight_sum:
        return out_matrix, out_transformed, float(read_scalar(acc, cost_slot)), 0.0
    # Both scalars in *one* transfer when the caller wants both. Two ``read_scalar`` calls measured
    # worse than this even though the second read rides on a drained pipeline, and one
    # ``read_scalar`` is better than this when only the cost is wanted, because ``.numpy()``
    # allocates a host array where ``read_scalar`` reuses a cached scratch. Hence the flag rather
    # than one spelling for both callers.
    acc_np = acc.numpy()
    return (
        out_matrix,
        out_transformed,
        float(acc_np[cost_slot]),
        float(acc_np[int(kernel_registration.ACC_W_SUM)]),
    )


_ROBUST_KINDS: dict[str, int] = {"none": 0, "huber": 1, "tukey": 2}


def icp(
    a: wp.array[wp.vec3],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32] | None = None,
    *,
    initial: wp.array[wp.mat44] | wp.mat44 | None = None,
    max_iterations: int = 20,
    threshold: float = 1e-5,
    max_distance: float | None = None,
    reflection: bool = False,
    translation: bool = True,
    scale: bool = False,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]:
    """
    Point-to-point iterative closest point registration.

    Aligns the source cloud *a* onto a target mesh or point cloud by repeatedly
    matching each source point to its nearest target counterpart and solving the
    resulting correspondence with [`procrustes`][ordito.registration.procrustes].
    Following the ``pytorch3d`` formulation, every iteration re-aligns the
    *original* *a* to the current correspondences, so the returned transform is a
    single map from *a* to the target (no incremental-composition drift). Only a
    rough initial alignment is required for a good result.

    Warp port combining [`trimesh.registration.icp`][] and
    ``pytorch3d.ops.iterative_closest_point``. Correspondence search runs on the
    GPU via [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
    (mesh target) or [`query_nearest`][ordito.neighbors.query_nearest]
    (point-cloud target); the alignment is the GPU tiled SVD of
    [`procrustes`][ordito.registration.procrustes].

    Its ``R`` is the **transpose** of the linear block returned here, because it solves the
    row-vector form ``s X R + T = Y``; the *converged transform* is what is comparable, not the
    iteration count, since ``relative_rmse_thr`` is its own stopping rule.

    Parameters
    ----------
    a
        ``(n,)`` source point cloud.
    target_vertices
        ``(m,)`` target vertex positions. Used as the
        target point cloud when ``target_faces`` is ``None``.
    target_faces
        ``(3 * n_faces,)`` flat triangle index buffer of the target. When given, the internally
        built ``wp.Mesh`` aliases ``target_vertices`` and ``target_faces`` rather than copying them;
        do not mutate them for the duration of the call. When given (and non-empty),
        correspondences are the closest points on the triangle surface; otherwise
        the target is the point cloud ``target_vertices``.
    initial
        ``(1,)`` ``wp.mat44`` array or a scalar ``wp.mat44``: the initial transform seeding the
        first correspondence search. Identity when ``None``.
    max_iterations
        Maximum number of ICP iterations.
    threshold
        Stop when the cost decreases by less than this between iterations.
    max_distance
        Reject correspondences farther than this (weight ``0``, excluded from the
        fit). No rejection when ``None``.
    reflection, translation, scale
        Forwarded to [`procrustes`][ordito.registration.procrustes]. Defaults are
        rigid (`reflection=False, scale=False`); set ``scale=True`` for a
        similarity transform.

    Returns
    -------
    matrix
        ``(1,)`` transform mapping *a* onto the target.
    transformed
        ``(n,)`` image of *a* under *matrix*.
    cost
        Weighted mean squared correspondence distance at the final iteration.

    Raises
    ------
    RuntimeError
        If ``a``, ``target_vertices``, ``target_faces`` and ``initial`` are not all on one device.

    See Also
    --------
    [`procrustes`][ordito.registration.procrustes]
    [`icp_point_to_plane`][ordito.registration.icp_point_to_plane]
    [`trimesh.registration.icp`][]
    """
    require_same_device(
        a=a, target_vertices=target_vertices, target_faces=target_faces, initial=initial
    )
    device = a.device
    n = a.size

    if n == 0 or target_vertices.size == 0:
        return _identity_mat44(device), _launch.clone(a), math.inf

    # ``total`` is the transform kept so far, and ``_resolve_initial``'s own copy, so the loop may
    # rewrite it in place. The source is never moved inside the loop: each correspondence pass
    # moves its own point by ``total`` in registers, and ``transformed`` is written again after the
    # loop by the transform it kept. The seed image is the answer of a call that runs no
    # iteration.
    total = _resolve_initial(initial, device)
    transformed = _launch.empty(n, dtype=wp.vec3, device=device)
    if max_iterations <= 0:
        _apply_transform(a, total, transformed)
        return total, transformed, math.inf
    mesh, query_max, target_index = _resolve_icp_target(target_vertices, target_faces, "icp")
    search_mesh = mesh if mesh is not None else target_index
    assert search_mesh is not None

    # Every buffer is allocated once and rewritten in place by every round. The fit lands in
    # ``fitted`` and is copied into ``total`` only once ``point_to_point_round`` has seen that it
    # carried weight; ``state`` is the shared round / condition word; ``acc`` is the fit's moments
    # plus the kept fit's cost, which is also the "previous cost" the convergence test reads.
    closest = _launch.empty(n, dtype=wp.vec3, device=device)
    weights = (
        _launch.empty(n, dtype=wp.float32, device=device)
        if max_distance is not None
        else _zero_length(wp.float32, device)
    )
    acc = _launch.zeros(kernel_registration.ICP_POINT_ACC_SIZE, dtype=wp.float32, device=device)
    fitted = _launch.empty(1, dtype=wp.mat44, device=device)
    state = _launch.zeros(kernel_array.LOOP_STATE_SIZE, dtype=wp.int32, device=device)

    blocks = kernel_reduce.blocks_1d(n)
    search_inputs = [
        search_mesh.id,
        target_vertices,
        a,
        total,
        mesh is None,
        wp.float32(query_max),
        wp.float32(max_distance if max_distance is not None else 0.0),
        closest,
        weights,
    ]
    moment_inputs = [a, closest, weights, translation, acc]
    fit_inputs = [
        a,
        closest,
        weights,
        acc,
        reflection,
        translation,
        scale,
        fitted,
        _zero_length(wp.vec3, device),
    ]
    round_inputs = [wp.float64(threshold), wp.int32(max_iterations), fitted]
    round_outputs = [acc, total, state]

    def iterate() -> None:
        _launch.launch(
            kernel_registration.point_to_point_correspondence_pass,
            dim=n,
            inputs=search_inputs,
            device=device,
        )
        _launch.launch_tiled(
            kernel_registration.accumulate_procrustes_moments,
            dim=blocks,
            inputs=moment_inputs,
            block_dim=TILE_1D,
            device=device,
        )
        _launch.launch_tiled(
            kernel_registration.fit_transform_and_accumulate_cost,
            dim=blocks,
            inputs=fit_inputs,
            block_dim=TILE_1D,
            device=device,
        )
        _launch.launch(
            kernel_registration.point_to_point_round,
            dim=1,
            inputs=round_inputs,
            outputs=round_outputs,
            device=device,
        )

    # Iteration 0 is issued from the host and seeds the condition; iterations 1.. are one recorded
    # body replayed on the device, which a one-iteration call has no need to record. The recording
    # is taken while round 0 runs and hides behind it. Round 0 can stop the loop only by being
    # weightless (its test compares against an ``inf`` previous cost), which only a distance gate
    # can make it -- uniform weights sum to ``n`` -- so only a gated call reads the round counter,
    # which a weightless round does not advance, before launching what it recorded. The read comes
    # after the recording, so it finds round 0 finished and waits on nothing.
    iterate()
    if max_iterations > 1:
        replay = record_device_loop(
            device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), iterate
        )
        if max_distance is not None and int(read_scalar(state, int(kernel_array.LOOP_ROUND))) == 0:
            # Weightless at the seed: ``total`` is the seed, and ``transformed`` its image.
            _apply_transform(a, total, transformed)
            return total, transformed, math.inf
        replay()
    _apply_transform(a, total, transformed)
    # The kept fit's cost, ``inf`` if every fit was weightless: a pinned call's one readback.
    return total, transformed, float(read_scalar(acc, int(kernel_registration.ACC_KEPT_COST)))


# Values are zero-length arrays of the key's dtype; ``_zero_length`` restores the element type.
_ZERO_LENGTH: dict[tuple[str, type], object] = {}


def _zero_length(dtype: type[DType], device: wp.DeviceLike) -> wp.array[DType]:
    """
    Return a zero-length array of ``dtype``, shared per device: the kernels' "not given" sentinel.

    A zero-length ``weights`` means "every weight is 1" (``sample_weight``), so the common
    weightless call needs neither a ``wp.full(n, 1.0)`` allocation nor its fill, and a zero-length
    ``out_transformed`` asks ``fit_transform_and_accumulate_cost`` for the cost alone. The buffer
    carries no data, so one instance per device and dtype serves every caller.
    """
    key = (str(device), dtype)
    if key not in _ZERO_LENGTH:
        _ZERO_LENGTH[key] = _launch.empty(0, dtype=dtype, device=device)
    return cast("wp.array[DType]", _ZERO_LENGTH[key])


def icp_point_to_plane(
    a: wp.array[wp.vec3],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32] | None = None,
    *,
    target_normals: wp.array[wp.vec3] | None = None,
    initial: wp.array[wp.mat44] | wp.mat44 | None = None,
    max_iterations: int = 20,
    threshold: float = 1e-6,
    max_distance: float | None = None,
    robust_kernel: Literal["none", "huber", "tukey"] = "none",
    robust_scale: float | None = None,
    damping: float = 1e-6,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]:
    """
    Point-to-plane iterative closest point registration with optional robust loss.

    Minimizes the squared distance from each transformed source point to the
    *tangent plane* of its target correspondence, which converges faster than the
    point-to-point metric on smooth surfaces. Each iteration linearizes the
    residual, assembles the ``6x6`` Gauss-Newton system on the GPU, solves it, and
    composes the incremental rigid step into the running transform.

    Warp port of the ``libigl`` ``iterative_closest_point`` (rigid, point-to-plane)
    and Open3D's robust ``TransformationEstimationPointToPlane``. An optional
    Huber or Tukey M-estimator down-weights outlier correspondences.

    Parameters
    ----------
    a
        ``(n,)`` source point cloud.
    target_vertices
        ``(m,)`` target vertex positions.
    target_faces
        ``(3 * n_faces,)`` flat triangle index buffer of the target. When given, target normals are
        the closest triangles' face normals; otherwise the target is the point
        cloud ``target_vertices`` and ``target_normals`` is required. When given, the internally
        built ``wp.Mesh`` aliases ``target_vertices`` and ``target_faces`` rather than copying
        them; do not mutate them for the duration of the call.
    target_normals
        ``(m,)`` per-vertex normals for a point-cloud target. They need not be unit
        length: each is normalized where it is read, by the fit and by the robust scale alike (a
        zero normal contributes nothing). Required (and only used) when ``target_faces`` is
        ``None``. Estimate them with
        [`estimate_normals`][ordito.points.estimate_normals] if absent.
    initial
        ``(1,)`` ``wp.mat44`` array or scalar ``wp.mat44``: the initial transform.
        Identity when ``None``.
    max_iterations
        Maximum number of ICP iterations.
    threshold
        Stop after an iteration whose ``cost`` (see Returns) is less than ``threshold`` below the
        previous iteration's; a rise stops it too. The first iteration is never tested, and
        ``-inf`` runs all ``max_iterations``. For every kernel the tested quantity falls as the
        fit improves, including Tukey's, whose weights redescend.
    max_distance
        Reject correspondences farther than this. No rejection when ``None``.
    robust_kernel
        M-estimator applied to residuals: ``"none"``, ``"huber"``, or ``"tukey"``.
    robust_scale
        Tuning constant of the robust kernel (Huber ``k`` / Tukey ``c``). When
        ``None`` and a robust kernel is active, derived from the median absolute
        deviation of the first iteration's residuals.
    damping
        Relative Levenberg damping added to the normal-equations diagonal (scaled
        by its mean magnitude) for conditioning on planar/degenerate targets.

    Returns
    -------
    matrix
        ``(1,)`` rigid transform mapping *a* onto the target.
    transformed
        ``(n,)`` image of *a* under *matrix*.
    cost
        The robust objective summed over the final iteration's in-range correspondences, with
        ``r`` the point-to-plane residual: ``sum r^2`` for ``"none"``; ``sum w(r) r^2`` for
        ``"huber"``, i.e. ``r^2`` within ``k`` and ``k |r|`` beyond it; and for ``"tukey"`` twice
        the biweight loss, ``c^2 / 3 * (1 - (1 - (r / c)^2)^3)`` within ``c`` and ``c^2 / 3``
        beyond it, so a correspondence the kernel rejects still counts at the saturated value. It
        is the objective of the returned ``matrix`` itself, over correspondences found at that
        pose -- one more correspondence pass after the last step, so it is also defined with
        ``max_iterations=0``, where it scores the initial transform. ``math.inf`` when no
        correspondence at that pose carries weight.

    Raises
    ------
    ValueError
        If ``robust_kernel`` is not one of the three names, or if the target is a point cloud
        (``target_faces is None``) and ``target_normals`` is not provided, or does not have the
        same length as ``target_vertices``.
    RuntimeError
        If ``a``, ``target_vertices``, ``target_faces``, ``target_normals`` and ``initial`` are not
        all on one device.

    See Also
    --------
    [`icp`][ordito.registration.icp]
    [`estimate_normals`][ordito.points.estimate_normals]
    [`normals_at_closest_faces`][ordito.proximity.normals_at_closest_faces]
    """
    require_same_device(
        a=a,
        target_vertices=target_vertices,
        target_faces=target_faces,
        target_normals=target_normals,
        initial=initial,
    )
    device = a.device
    n = a.size
    is_mesh = _is_mesh_target(target_faces)
    if robust_kernel not in _ROBUST_KINDS:
        raise ValueError(
            f"robust_kernel must be one of {list(_ROBUST_KINDS)}, got {robust_kernel!r}"
        )
    kind = _ROBUST_KINDS[robust_kernel]

    if not is_mesh and target_normals is None:
        raise ValueError(
            "icp_point_to_plane requires target_normals for a point-cloud target "
            "(target_faces is None)."
        )
    if not is_mesh and target_normals is not None and target_normals.size != target_vertices.size:
        # ``target_normals`` feeds ``accumulate_point_to_plane`` below, indexed by a nearest-vertex
        # id that ranges over ``target_vertices``; a shorter buffer is an out-of-bounds read on
        # both devices (CLAUDE.md §12.1), not merely a wrong answer.
        raise ValueError(
            "target_normals must have the same length as target_vertices, got "
            f"{target_normals.size} and {target_vertices.size}."
        )

    if n == 0 or target_vertices.size == 0:
        return _identity_mat44(device), _launch.clone(a), math.inf

    # ``current`` is the moving source. Every search writes it, the first from ``a`` and the seed.
    initial_matrix = _resolve_initial(initial, device)
    current = _launch.empty(n, dtype=wp.vec3, device=device)

    mesh, query_max, target_index = _resolve_icp_target(
        target_vertices, target_faces, "icp_point_to_plane"
    )
    face_normals: wp.array[wp.vec3] | None = None
    if is_mesh:
        assert target_faces is not None
        face_normals, _ = od.triangles.face_normals_and_areas(target_vertices, target_faces)

    max_d = max_distance if max_distance is not None else math.inf
    scale_value = robust_scale

    # All buffers are allocated once and the loop rewrites them in place: the step is applied to
    # ``current`` by the next correspondence search, the solve composes into ``total`` and zeroes
    # the moments it has just read. ``total`` is ``_resolve_initial``'s own copy, so composing into
    # it never touches a caller's array. ``acc`` is the normal equations, the two scalars and the
    # previous kept cost in one buffer (``kernel_registration.ICP_ACC_SIZE``); ``state`` the shared
    # round and condition slots.
    closest = _launch.empty(n, dtype=wp.vec3, device=device)
    distance = _launch.empty(n, dtype=wp.float32, device=device)
    triangle_id = _launch.empty(n, dtype=wp.int32, device=device)
    acc = _launch.zeros(kernel_registration.ICP_ACC_SIZE, dtype=wp.float32, device=device)
    step = _launch.empty(1, dtype=wp.mat44, device=device)
    total = initial_matrix
    state = _launch.zeros(kernel_array.LOOP_STATE_SIZE, dtype=wp.int32, device=device)

    # The correspondence step is one launch: the step applied to each point, and its match on the
    # target -- the closest point on a mesh, or the nearest vertex of a cloud through the hoisted
    # collapsed-triangle mesh (``kernel_registration.icp_match``). The target normal is read by
    # the accumulation at the match's index, from the face normals of a mesh or the per-vertex
    # normals of a cloud, so an iteration is three launches: search, accumulate, and the ``dim=1``
    # round. The first search moves ``a`` by the seed, bit for bit what ``_apply_transform``
    # writes.
    #
    # Measure this kind of change with ``threshold=-inf``, or the reading is fiction: with the
    # convergence break live the two arms stop at *different* iterations. The plateau is also not
    # bit-reproducible on CUDA -- the point-to-plane normal equations are accumulated with
    # ``float32`` atomics, so two runs of the identical build disagree in the last digits once the
    # cost stops moving.
    normal_table = face_normals if mesh is not None else target_normals
    assert normal_table is not None
    search_mesh = mesh if mesh is not None else target_index
    assert search_mesh is not None
    search_target = [search_mesh.id, target_vertices]
    search_options = [mesh is None, wp.float32(query_max)]
    search_outputs = [current, closest, distance, triangle_id]

    def search(stepped: bool) -> None:
        source, source_step = (current, step) if stepped else (a, total)
        _launch.launch(
            kernel_registration.point_to_plane_correspondence_pass,
            dim=n,
            inputs=[*search_target, source, source_step, *search_options],
            outputs=search_outputs,
            device=device,
        )

    def accumulate() -> None:
        _launch.launch_tiled(
            kernel_registration.accumulate_point_to_plane,
            dim=kernel_reduce.blocks_1d(n),
            inputs=[
                current,
                closest,
                normal_table,
                distance,
                triangle_id,
                wp.float32(max_d),
                wp.int32(kind),
                wp.float32(scale_value if scale_value is not None else 0.0),
                acc,
            ],
            block_dim=TILE_1D,
            device=device,
        )

    def resolve_scale() -> None:
        # It reads the residuals back, so it runs before any recorded round.
        nonlocal scale_value
        if kind == 0 or scale_value is not None:
            return
        scale_value = _robust_scale_from_residuals(
            current, closest, normal_table, distance, triangle_id, max_d, kind
        )

    round_inputs = [wp.float32(damping), wp.float64(threshold), wp.int32(max_iterations)]
    round_outputs = [acc, total, step, state]

    def close_round() -> None:
        _launch.launch(
            kernel_registration.point_to_plane_round,
            dim=1,
            inputs=round_inputs,
            outputs=round_outputs,
            device=device,
        )

    def iterate() -> None:
        search(True)
        accumulate()
        close_round()

    # --- iteration 0, issued from the host: it may need the robust scale, a host reduction ---
    # The weight sum it tests is over the correspondences that survive the ``max_distance`` gate,
    # so it is zero when that gate rejected every one -- without the test, every accumulator stays
    # zero, the damped 6x6 solve returns a zero step, and the loop reports ``cost=0.0``,
    # indistinguishable from a perfect fit, instead of stopping with the last real cost (or
    # ``math.inf`` if nothing ever matched). It is equally zero when a Tukey kernel drove every
    # in-range correspondence's own robust weight to exactly zero (``|residual| >= scale``,
    # reachable through an explicit ``robust_scale`` too tight for the residual distribution, or an
    # earlier iteration's step overshooting far past what a first-iteration-derived scale
    # anticipated): reproduced with ``robust_kernel="tukey", robust_scale=1e-9``, which otherwise
    # returned the identity transform with ``cost=0.0`` on a cloud still offset by (1.0, 0.5, -0.3)
    # from its target.
    stepped = False
    if max_iterations > 0:
        search(False)
        resolve_scale()
        accumulate()
        close_round()
        # --- iterations 1.., one recorded body replayed on the device with no readback ---
        # ``close_round`` seeded the condition; the body is a fixed three-launch sequence over
        # buffers it rewrites in place, so it records once per call. Recording costs about what
        # issuing a round does even when the condition is already clear, so a one-iteration call,
        # which the host knows cannot run a second, skips it.
        #
        # A first round that was weightless -- the only way round 0 can stop the loop -- makes
        # both the recording and the closing pass below pure cost, and only a distance gate or
        # Tukey's redescending weights can zero every weight (``"none"`` and Huber weights are
        # positive). So those two configurations read the round counter, which a weightless round
        # does not advance, and return the seed's answer; the others never pay the read.
        replay = None
        if max_iterations > 1:
            replay = record_device_loop(
                device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), iterate
            )
        if (max_distance is not None or kind == _ROBUST_KINDS["tukey"]) and int(
            read_scalar(state, int(kernel_array.LOOP_ROUND))
        ) == 0:
            return total, current, math.inf
        if replay is not None:
            replay()
        stepped = True

    # ``cost`` is the objective of the transform returned, not of the pose the last step was solved
    # from: one more correspondence pass at the returned pose -- which is also where the last step
    # is applied -- and one more accumulation over it, into the accumulators the last round zeroed.
    # With no iteration run it scores the seed. After a weightless stop the step is the identity
    # (``point_to_plane_round``), so this pass finds the same correspondences, sums zero weight
    # again and reports ``inf`` -- the stop costs no readback of its own.
    search(stepped)
    resolve_scale()
    accumulate()
    scalars = acc.numpy()
    if float(scalars[kernel_registration.ICP_WEIGHT_SUM]) <= 0.0:
        return total, current, math.inf
    return total, current, float(scalars[kernel_registration.ICP_COST])


def _robust_scale_from_residuals(
    current: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: float,
    kind: int,
) -> float:
    """
    Robust scale (Huber/Tukey) from the MAD of the current point-to-plane residuals.

    ``normals`` is the target's normal table -- per face or per vertex -- read at each
    correspondence's index.

    Both medians are taken on the device, over the sorted prefix of in-range residuals
    (``kernel_registration.robust_residual_keys``), so the whole estimate is one readback of the
    in-range count and ``1.4826 * MAD``. A zero MAD falls back to the standard deviation, on the
    host path the medians used to take too, since its means must sum the in-range residuals in
    their original order.
    """
    device = current.device
    n = current.size
    # Radix-sort buffers: keys and a payload the sort needs and nobody reads, both ``2n`` long.
    keys = _launch.empty(2 * n, dtype=wp.float32, device=device)
    payload = _launch.empty(2 * n, dtype=wp.int32, device=device)
    stats = _launch.empty(kernel_registration.MAD_STATS_SIZE, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_registration.robust_residual_keys,
        dim=n,
        inputs=[current, closest, normals, triangle_id, distance, wp.float32(max_distance)],
        outputs=[keys],
        device=device,
    )
    _launch.radix_sort_pairs(keys, payload, n)
    _launch.launch(
        kernel_registration.robust_residual_center,
        dim=1,
        inputs=[keys, wp.int32(n)],
        outputs=[stats],
        device=device,
    )
    _launch.launch(
        kernel_registration.robust_deviation_keys,
        dim=n,
        inputs=[stats],
        outputs=[keys],
        device=device,
    )
    _launch.radix_sort_pairs(keys, payload, n)
    _launch.launch(
        kernel_registration.robust_sigma, dim=1, inputs=[keys], outputs=[stats], device=device
    )
    # The estimate's one readback: the in-range count and the scale.
    stats_np = stats.numpy()
    if int(stats_np[kernel_registration.MAD_STATS_COUNT]) == 0:
        return 0.0
    sigma = float(stats_np[kernel_registration.MAD_STATS_VALUE])
    if sigma <= 0.0:
        sigma = _residual_standard_deviation(
            current, closest, normals, distance, triangle_id, max_distance
        )
    if sigma <= 0.0:
        return 0.0
    # 95% asymptotic efficiency tuning constants.
    return 1.345 * sigma if kind == 1 else 4.685 * sigma


def _residual_standard_deviation(
    current: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: float,
) -> float:
    """
    ``sqrt(mean((r - mean(r))^2))`` over the in-range residuals, in their original order.

    ``normals`` is the target's normal table, read at each correspondence's index.
    """
    device = current.device
    # ``(sum, count, squared deviation)`` of the in-range residuals, in two folds that form the
    # residuals where they are summed: no residual table, compaction or intermediate readback.
    sums = _launch.zeros(3, dtype=wp.float64, device=device)
    for centered in (0, 1):
        _launch.launch_tiled(
            kernel_registration.residual_moment,
            dim=[kernel_reduce.blocks_1d(current.size)],
            inputs=[
                current,
                closest,
                normals,
                triangle_id,
                distance,
                wp.float32(max_distance),
                centered,
            ],
            outputs=[sums],
            block_dim=TILE_1D,
            device=device,
        )
    _, count, squares = sums.numpy()
    return math.sqrt(squares / count)


# The identity as host data a ``(1,)`` ``wp.mat44`` upload takes as it is.
_IDENTITY_MAT44_NP = np.eye(4, dtype=np.float32).reshape(1, 4, 4)


def _identity_mat44(device: wp.DeviceLike) -> wp.array[wp.mat44]:
    """Return a ``(1,)`` array holding the 4x4 identity transform."""
    return _launch.array(_IDENTITY_MAT44_NP, dtype=wp.mat44, device=device)


def _apply_transform(
    points: wp.array[wp.vec3], matrix: wp.array[wp.mat44], out_points: wp.array[wp.vec3]
) -> None:
    """Write ``matrix[0]`` applied to every point of ``points`` into ``out_points``."""
    _launch.launch(
        kernel_transform.apply_transform_mat44,
        dim=points.size,
        inputs=[points, matrix, out_points],
        device=points.device,
    )


def _resolve_initial(
    initial: wp.array[wp.mat44] | wp.mat44 | None, device: wp.DeviceLike
) -> wp.array[wp.mat44]:
    """Normalize ``initial`` to a ``(1,)`` ``wp.mat44`` device array."""
    if initial is None:
        return _identity_mat44(device)
    if isinstance(initial, wp.array):
        return _launch.clone(initial)
    # A ``wp.mat44`` is a ctypes array, so NumPy views its 16 floats in place (its stub does not
    # declare the buffer protocol, hence the cast); a one-element list of it would make NumPy walk
    # the matrix row by row.
    host = np.frombuffer(cast("bytes", initial), dtype=np.float32).reshape(1, 4, 4)
    return _launch.array(host, dtype=wp.mat44, device=device)


def _resolve_icp_target(
    target_vertices: wp.array[wp.vec3], target_faces: wp.array[wp.int32] | None, caller: str
) -> tuple[wp.Mesh | None, float, wp.Mesh | None]:
    """
    Build the loop-invariant search state for whichever target kind was supplied.

    A mesh target gets a ``wp.Mesh`` and a query radius; a point-cloud target gets the
    collapsed-triangle mesh of [`_target_index`][ordito.registration._target_index].
    Exactly one of the two is non-``None``.

    Parameters
    ----------
    target_vertices
        ``(m,)`` target positions.
    target_faces
        Flat triangle index buffer, or ``None`` / empty for a point-cloud target.
    caller
        Calling function's name, for ``require_nonempty_mesh``'s error message.

    Returns
    -------
    tuple[wp.Mesh | None, float, wp.Mesh | None]
        ``(mesh, query_max, target_index)``.
    """
    if not _is_mesh_target(target_faces):
        return None, 0.0, _target_index(target_vertices)

    assert target_faces is not None
    require_nonempty_mesh(target_faces, caller)
    # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
    mesh = wp.Mesh(points=target_vertices, indices=target_faces)
    # An unbounded closest-point query: the correspondence gate is ``max_distance``, applied to
    # the distance the query returns, so the query itself needs no radius.
    return mesh, math.inf, None


def _target_index(target_vertices: wp.array[wp.vec3]) -> wp.Mesh:
    """
    Precompute the nearest-vertex search structure for a point-cloud target.

    Only the *source* moves between ICP iterations, so the target's tree is loop-invariant, and
    hoisting it turns each iteration's correspondence step into a single launch with no host
    synchronisation at all. It is a closest-point mesh rather than a ``wp.Bvh``: the source sits
    off the target until the fit converges, and a radius-deepening walk pays for that distance
    where the mesh descent does not -- and it needs neither the bounding box nor the density
    estimate the walk seeds from.
    """
    return od.neighbors.mesh_from_points(target_vertices)


def _is_mesh_target(target_faces: wp.array[wp.int32] | None) -> bool:
    return target_faces is not None and target_faces.size // 3 > 0
