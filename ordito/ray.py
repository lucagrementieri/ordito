"""Ray and point-in-mesh queries on ``wp.Mesh``."""

from __future__ import annotations

import math

import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.config import is_deterministic
from ordito.constants import TOLERANCE_PLANAR
from ordito.kernels import proximity as kernel_proximity
from ordito.kernels import ray as kernel_ray


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
    [`intersects_first`][ordito.ray.intersects_first].

    With [`is_deterministic`][ordito.config.is_deterministic] off (the default) the returned
    arrays are prefix views of buffers sized for every ray, so they keep that storage alive for
    as long as they are held, and **row order is unspecified**: each hitting ray claims its
    output slot from an atomic counter, so rows arrive in the order the rays completed rather than
    in ray order, and two runs over the same input may order them differently. With it on, the
    rows come back in ascending ray order, from a scan of the hit flags.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
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

    Examples
    --------
    Two rays along ``+x`` towards a unit cube: the first enters through the ``x = -0.5`` face, the
    second passes above the cube and is left out of the result:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[-2.0, 0.1, 0.2], [-2.0, 3.0, 0.0]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    locations, index_ray, index_tri = od.ray.intersects_location(mesh, origins, directions)
    print(index_ray.numpy().tolist())  # [0]
    print(locations.numpy()[0].round(3))  # [-0.5  0.1  0.2]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    device = ray_origins.device
    if n == 0:
        empty_int = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int

    if max_t is None:
        max_t = math.inf

    if is_deterministic():
        return _intersects_location_ordered(mesh, ray_origins, ray_directions, max_t)
    index_ray = _launch.empty(n, dtype=wp.int32, device=device)
    index_tri = _launch.empty(n, dtype=wp.int32, device=device)
    locations = _launch.empty(n, dtype=wp.vec3, device=device)
    counter = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
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
        empty_int = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int
    return (
        odt.as_dense(locations[:n_hits]),
        odt.as_dense(index_ray[:n_hits]),
        odt.as_dense(index_tri[:n_hits]),
    )


def _intersects_location_ordered(
    mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3], max_t: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    [`intersects_location`][ordito.ray.intersects_location]'s hits in ascending ray order.

    The deterministic setting's form: every ray's hit is stored at its own index, the hit flags
    are scanned in place and the hits compacted at their rank, so the order does not follow which
    ray's thread finished first.
    """
    device = ray_origins.device
    n = ray_origins.size
    faces = _launch.empty(n, dtype=wp.int32, device=device)
    hit_locations = _launch.empty(n, dtype=wp.vec3, device=device)
    inclusive = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_ray.first_hit_flags,
        dim=n,
        inputs=[wp.uint64(mesh.id), ray_origins, ray_directions, wp.float32(max_t)],
        outputs=[faces, hit_locations, inclusive],
        device=device,
    )
    _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # The hit count sizes the outputs.
    n_hits = int(read_scalar(inclusive))
    locations = _launch.empty(n_hits, dtype=wp.vec3, device=device)
    index_ray = _launch.empty(n_hits, dtype=wp.int32, device=device)
    index_tri = _launch.empty(n_hits, dtype=wp.int32, device=device)
    if n_hits > 0:
        _launch.launch(
            kernel_ray.compact_scanned_hits,
            dim=n,
            inputs=[faces, hit_locations, inclusive],
            outputs=[index_ray, index_tri, locations],
            device=device,
        )
    return locations, index_ray, index_tri


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
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
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

    Examples
    --------
    The same two rays as in [`intersects_location`][ordito.ray.intersects_location], with one row
    per ray and ``-1`` for the miss:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[-2.0, 0.1, 0.2], [-2.0, 3.0, 0.0]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.intersects_first(mesh, origins, directions).numpy().tolist())  # [0, -1]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    if n == 0:
        return _launch.empty(0, dtype=wp.int32, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_triangle_index = _launch.empty(n, dtype=wp.int32, device=ray_origins.device)
    # ``first_hit`` returns ``(face, location)`` and ``wp.map`` wants one ``out=`` per returned
    # value, so the location is written and dropped. Kept rather than given a face-only
    # ``@wp.func`` of its own: that would generate a second ``map_*`` module to save one
    # allocation and an (n,) vec3 store, well under a percent of a call this size.
    locations_scratch = _launch.empty(n, dtype=wp.vec3, device=ray_origins.device)
    _launch.map(
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
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
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

    Examples
    --------
    A ray towards a unit cube, one passing above it and one pointing away. The cube is ``1.5``
    away along the first ray, so ``max_t=1.0`` stops short of it:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[-2.0, 0.1, 0.2], [-2.0, 3.0, 0.0], [-2.0, 0.1, 0.2]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.intersects_any(mesh, origins, directions).numpy().tolist())  # [True, False, False]
    hits = od.ray.intersects_any(mesh, origins, directions, max_t=1.0)
    print(hits.numpy().tolist())  # [False, False, False]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    if n == 0:
        return _launch.empty(0, dtype=wp.bool, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_hit = _launch.empty(n, dtype=wp.bool, device=ray_origins.device)
    _launch.map(
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
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
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

    Examples
    --------
    From the centre of a unit cube the free path along ``+x`` ends on the face at ``x = 0.5``; from
    outside, pointing away, it never ends:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[0.0, 0.1, 0.2], [2.0, 0.1, 0.2]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.longest_ray(mesh, origins, directions).numpy().tolist())  # [0.5, inf]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    if n == 0:
        return _launch.empty(0, dtype=wp.float32, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_distances = _launch.empty(n, dtype=wp.float32, device=ray_origins.device)
    _launch.map(
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
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]. Points
    outside the mesh axis-aligned bounding box are rejected without a parity test.

    Ordito mesh queries use **Warp's SDF sign convention** (outside positive,
    inside negative). A point is classified as inside when its signed distance
    would be negative; behavior on the on-surface tolerance band is undefined.
    Boolean results still match [`trimesh.Trimesh.contains`][].

    !!! note "Non-watertight meshes"

        Ray parity has no principled answer on an open or holed surface. For those, use the
        generalized winding-number sign of
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] instead
        (``sign_mode="winding"``, negative inside; see the second example below).

        That mode needs a ``wp.Mesh`` built with ``support_winding_number=True``; without the flag
        Warp's winding query silently falls back to ray parity.
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] builds such a mesh
        itself, or accepts one passed as ``mesh=`` after checking the flag.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(n,)`` query positions.
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
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    mesh = wp.Mesh(points=vertices, indices=faces)
    points = wp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.contains_points(mesh, points).numpy().tolist())  # [True, True, False]
    ```

    On an open surface, here a sphere with a hole around its ``-z`` pole, the winding-number sign
    still separates the inside from the outside:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=2.5)
    print(od.validation.is_watertight(vertices, faces))  # False
    points = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=wp.vec3)
    signed = od.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode="winding")
    print((signed.numpy() < 0.0).tolist())  # [True, False]
    ```
    """
    require_same_device(mesh=mesh, points=points)
    n = points.size
    if n == 0:
        return _launch.empty(0, dtype=wp.bool, device=points.device)
    # The kernel reads the mesh's box off its BVH root for the box rejection and for the query
    # cap, its diagonal, so the call reads nothing back.
    out_contains = _launch.empty(n, dtype=wp.bool, device=points.device)
    _launch.launch(
        kernel_proximity.contains_points_sign_parity,
        dim=n,
        inputs=[mesh.id, points, wp.int32(n_sample), wp.float32(perturbation_scale), out_contains],
        device=points.device,
    )
    return out_contains
