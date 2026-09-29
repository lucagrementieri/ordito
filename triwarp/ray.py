"""Ray and point-in-mesh queries on ``wp.Mesh``."""

from __future__ import annotations

import math

import warp as wp

import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.bounds import aabb
from triwarp.constants import TOLERANCE_PLANAR
from triwarp.kernels import proximity as kernel_proximity
from triwarp.kernels import ray as kernel_ray


def intersects_location(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return world-space locations where rays hit the mesh surface (first hit per ray).

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. Returns only rays that hit within ``max_t`` as sparse ``(m,)`` arrays,
    carrying the same set of hits as the dense output of
    [`intersects_first`][triwarp.ray.intersects_first].

    The returned arrays are prefix views of buffers sized for every ray, so they keep that
    storage alive for as long as they are held.

    **Row order is unspecified.** Each hitting ray claims its output slot from an atomic
    counter, so rows arrive in the order the rays completed rather than in ray order, and two
    runs over the same input may order them differently. Sort by ``index_ray`` (or read
    [`intersects_first`][triwarp.ray.intersects_first] instead) when a stable order matters.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.

    Returns
    -------
    locations
        ``(m,)`` intersection points.
    index_ray
        ``(m,)`` index of the ray that produced each hit.
    index_tri
        ``(m,)`` face indices for each hit.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.shape[0]
    device = ray_origins.device
    if n == 0:
        empty_int = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int

    if max_t is None:
        max_t = math.inf

    index_ray = wp.empty(n, dtype=wp.int32, device=device)
    index_tri = wp.empty(n, dtype=wp.int32, device=device)
    locations = wp.empty(n, dtype=wp.vec3, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_ray.first_hit_append,
        dim=n,
        inputs=[
            wp.uint64(mesh.id),
            ray_origins,
            ray_directions,
            wp.float32(max_t),
            index_ray,
            index_tri,
            locations,
            counter,
        ],
        device=device,
    )
    # Prefix views of the per-ray buffers rather than exact-size copies: three allocations and
    # three copies fewer, at the price of the returned arrays keeping ``n``-long storage alive.
    n_hits = int(read_scalar(counter, 0))
    if n_hits == 0:
        empty_int = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int
    return (
        twt.as_dense(locations[:n_hits]),
        twt.as_dense(index_ray[:n_hits]),
        twt.as_dense(index_tri[:n_hits]),
    )


def intersects_first(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> wp.array[wp.int32]:
    """
    Find the index of the first triangle each ray hits.

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. The search along each ray is unbounded unless ``max_t`` is given.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.

    Returns
    -------
    wp.array[wp.int32]
        ``(n,)`` face indices; ``-1`` when a ray misses within ``max_t``.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_triangle_index = wp.empty(n, dtype=wp.int32, device=ray_origins.device)
    # ``first_hit`` returns ``(face, location)`` and ``wp.map`` wants one ``out=`` per returned
    # value, so the location is written and dropped. Kept rather than given a face-only
    # ``@wp.func`` of its own: that would generate a second ``map_*`` module to save one
    # allocation and an (n,) vec3 store, well under a percent of a call this size.
    locations_scratch = wp.empty(n, dtype=wp.vec3, device=ray_origins.device)
    wp.map(
        kernel_ray.first_hit,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        out=[out_triangle_index, locations_scratch],
    )
    return out_triangle_index


def intersects_any(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> wp.array[wp.bool]:
    """
    Check whether each ray hits the mesh surface.

    Uses ``wp.mesh_query_ray_anyhit`` on the mesh BVH. Ray directions are unitized
    before querying. The search along each ray is unbounded unless ``max_t`` is given.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` hit flags; ``True`` when a ray hits within ``max_t``.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.bool, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_hit = wp.empty(n, dtype=wp.bool, device=ray_origins.device)
    wp.map(
        kernel_ray.any_hit,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        out=out_hit,
    )
    return out_hit


def longest_ray(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
    planar_tol: float = TOLERANCE_PLANAR,
) -> wp.array[wp.float32]:
    """
    Find the length of the longest unobstructed ray segment along each direction.

    Uses iterative ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are
    unitized before querying. For each ray, returns the distance to the first mesh
    intersection strictly beyond ``planar_tol`` (to ignore degenerate on-surface
    hits), or ``inf`` when no such intersection exists within ``max_t``.

    Equivalent to [`trimesh.proximity.longest_ray`][].

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.
    planar_tol
        Ignore intersections closer than this distance from the ray origin.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` unobstructed ray lengths; ``inf`` when a ray misses within ``max_t``.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_distances = wp.empty(n, dtype=wp.float32, device=ray_origins.device)
    wp.map(
        kernel_ray.longest_ray_distance,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        wp.float32(planar_tol),
        out=out_distances,
    )
    return out_distances


def _validate_ray_inputs(ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3]) -> None:
    if ray_origins.shape != ray_directions.shape:
        raise ValueError(
            "ray_origins and ray_directions must have the same shape: "
            f"{tuple(ray_origins.shape)} vs {tuple(ray_directions.shape)}"
        )


def contains_points(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, n_sample: int = 5, perturbation_scale: float = 0.1
) -> wp.array[wp.bool]:
    """
    Test whether query points lie inside a closed mesh (ray parity sign).

    Uses ``wp.mesh_query_point_sign_parity`` on the mesh BVH, sharing the same
    parity path as
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]. Points
    outside the mesh axis-aligned bounding box are rejected without a parity test.

    Triwarp mesh queries use **Warp's SDF sign convention** (outside positive,
    inside negative). A point is classified as inside when its signed distance
    would be negative; behavior on the on-surface tolerance band is undefined.
    Boolean results still match [`trimesh.Trimesh.contains`][].

    !!! note "Non-watertight meshes"

        Ray parity has no principled answer on an open or holed surface. For those, use the
        generalized winding-number sign instead:

        ```python
        from triwarp.kernels import array as kernel_array

        signed = tw.proximity.signed_distance_on_mesh(v, f, pts, sign_mode="winding")
        inside = wp.empty(signed.shape, dtype=wp.bool, device=signed.device)
        wp.map(kernel_array.less, signed, wp.float32(0.0), out=inside)
        ```

        That mode is deliberately **not** offered here. It needs a ``wp.Mesh`` built with
        ``support_winding_number=True``, and Warp neither records that flag on the mesh object nor
        errors when it is missing — it silently falls back to ray parity. Since this function takes
        a caller-supplied ``wp.Mesh``, triwarp cannot verify the flag, so the option would be able
        to quietly return the parity answer.
        [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] builds its own mesh
        and therefore can guarantee it.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(n,)`` query positions as ``wp.vec3``.
    n_sample
        Number of perturbed rays for parity voting (higher is more robust).
    perturbation_scale
        Uniform perturbation scale applied to the base ray direction.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` flags; ``True`` when the point is classified as inside the mesh.

    Raises
    ------
    RuntimeError
        If ``mesh`` and ``points`` are not all on one device.

    See Also
    --------
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
    """
    require_same_device(mesh=mesh, points=points)
    n = points.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.bool, device=points.device)
    # One reduction, not two: ``enclosing_diagonal(mesh.points)`` would recompute exactly these
    # corners, and the parity kernel needs both them and the diagonal, so compute the AABB once
    # and derive the diagonal from it directly.
    mesh_min, mesh_max = aabb(mesh.points)
    # ``math.dist`` rather than ``float(wp.length(upper - lower))``: a Warp operator and a
    # Warp builtin at Python scope each route through builtin dispatch, several times dearer. It
    # computes in float64 where ``wp.length`` is float32, i.e. the correctly-rounded answer for
    # float32 corners. Section 13.1.
    max_dist = math.dist(mesh_min, mesh_max)
    out_contains = wp.empty(n, dtype=wp.bool, device=points.device)
    wp.launch(
        kernel_proximity.contains_points_sign_parity,
        dim=n,
        inputs=[
            mesh.id,
            points,
            wp.float32(max_dist),
            wp.int32(n_sample),
            wp.float32(perturbation_scale),
            mesh_min,
            mesh_max,
            out_contains,
        ],
        device=points.device,
    )
    return out_contains
