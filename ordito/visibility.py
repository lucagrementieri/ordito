"""
How far the surface is from a point.

Outward until nothing is hit ([`ambient_occlusion`][ordito.visibility.ambient_occlusion],
[`volumetric_obscurance`][ordito.visibility.volumetric_obscurance]); inward until it is hit again
([`shape_diameter`][ordito.visibility.shape_diameter],
[`thickness`][ordito.visibility.thickness]); or in every direction at once, which is the radius at
which the surface first appears
([`max_tangent_sphere`][ordito.visibility.max_tangent_sphere]).

All five take ``(mesh: wp.Mesh, points, *, normals=None, ...)`` -- a prebuilt BVH and a set of
positions to measure at -- which is what separates this module from
[`ordito.proximity`][ordito.proximity], where the queries take raw ``(vertices, faces)`` and ask
*where* the surface is rather than how far away.

The two outward fields integrate the same bundle of rays over the outward hemisphere at each point,
weighted by Lambert's cosine law, and differ only in what a blocked ray costs: ambient occlusion
charges a hit its full weight however far away it is, obscurance discounts it by
``exp(-tau * distance)`` so that only nearby geometry darkens a point. Ambient occlusion is the
``tau -> 0`` limit of obscurance, and asks of each ray only whether it is blocked, not how far
away. Neither is normalized against a scene, so both are comparable across meshes and resolutions,
and on a **convex** closed surface no ray can return and every point reads exactly ``0`` -- the
cheapest available sanity check on a result.

The three inward measures differ in how much evidence they take.
[`thickness`][ordito.visibility.thickness] is a dispatcher over the other two: one inward ray, or
one tangent sphere. [`shape_diameter`][ordito.visibility.shape_diameter] fires a whole cone and
takes an outlier-trimmed mean, which is what makes it stable enough for skeleton extraction and part
segmentation. [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere] uses no rays at all: it
shrinks a sphere until nothing but the surface touches it, so it answers the question for a *volume*
rather than along a direction, and it is the reason this module's name is a loose fit for one of its
five members.
"""

from __future__ import annotations

import math
from typing import Literal

import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.constants import TOLERANCE_PLANAR
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import visibility as kernel_visibility
from ordito.proximity import ITEMS_PER_QUERY_SLICE, normals_at_closest_faces

# Ray-origin offset along the normal, as a fraction of the query AABB diagonal. Without it every
# ray would hit the surface it started on; the value is small enough not to shadow a real occluder
# and large enough to clear float32 error on the starting triangle. The inward bundles offset
# *below* the surface by the same fraction, for the same reason.
_SURFACE_OFFSET = 1e-4

# Point count from which ``ambient_occlusion`` traces a CUDA cloud in the point-major layout (a
# block of Morton-consecutive points, a warp per ray direction; see ``kernels/visibility.py``)
# rather than one block per point. Set by sweeping random vertex subsets of a 14 M-point scan at 64
# rays: break-even at ~2-4 M points (0.96x at 1 M, 1.02x at 4 M, 1.16x at 14 M, sort included).
_OCCLUSION_POINT_MAJOR_FROM = 4_000_000

_WEIGHT_MODES: dict[str, wp.int32] = {
    "cosine": kernel_visibility.WEIGHT_COSINE,
    "uniform": kernel_visibility.WEIGHT_UNIFORM,
}

# A lookup rather than a chain of comparisons, so an unrecognised name fails loudly in one place
# instead of falling through to a branch. Same shape as ``_WEIGHT_MODES`` above and
# ``triangles._QUALITY_METRICS``.
_THICKNESS_METHODS = frozenset({"max_sphere", "ray"})

RayWeight = Literal["cosine", "uniform"]
"""Weighting of a ray; see [`ambient_occlusion`][ordito.visibility.ambient_occlusion]."""


def ambient_occlusion(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    normals: wp.array[wp.vec3] | None = None,
    n_rays: int = 64,
    weight: RayWeight = "cosine",
    max_t: float | None = None,
) -> wp.array[wp.float32]:
    """
    Fraction of the outward hemisphere at each point that is blocked by the mesh itself.

    A Fibonacci hemisphere lattice is rotated into each point's tangent frame and every direction is
    traced against the mesh; the result is the weighted share of directions that hit something. This
    is MeshLab's ``compute_scalar_ambient_occlusion`` and libigl's ``igl::ambient_occlusion``, which
    differ from each other exactly in the ``weight`` argument below.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``). The occluder *and* the surface being shaded.
        Every ray is traced through this BVH, so its quality sets the cost and not the values:
        [`Trimesh.traced_mesh`][ordito.mesh.Trimesh.traced_mesh] traces a large bundle markedly
        faster than [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] for a costlier build, and
        [`Trimesh.mesh_for_rays`][ordito.mesh.Trimesh.mesh_for_rays] picks between the two.
    points
        ``(m,)`` positions to shade, normally the mesh's own vertices.
    normals
        ``(m,)`` outward unit normals, defining which hemisphere is "outward" at each point. When
        ``None`` they are taken from the closest face of ``mesh``
        ([`normals_at_closest_faces`][ordito.proximity.normals_at_closest_faces]), which is right
        for points on the surface and meaningless for points far off it — pass them explicitly in
        that case. For a smooth result on the mesh's own vertices, pass
        [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="area"`` instead:
        face normals make the field piecewise constant across each vertex's ring.
    n_rays
        Directions per point. Error falls as ``1 / sqrt(n_rays)``; MeshLab's default is ``64``,
        which is this one. Cost is exactly linear in it.
    weight
        How a direction is weighted in the integral:

        - ``"cosine"`` (default) — by ``dot(direction, normal)``, which is the physically correct
          weighting for ambient irradiance and MeshLab's. Grazing directions barely matter.
        - ``"uniform"`` — every direction counts once, which is libigl's convention and reads as a
          solid-angle fraction rather than an irradiance one.
    max_t
        Maximum ray length. Anything beyond it does not occlude. When ``None``, the diagonal of the
        AABB enclosing the mesh and the query points, so nothing in the mesh is missed.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` occlusion in ``[0, 1]`` on ``points.device``: ``0`` where the hemisphere is fully
        open and ``1`` where it is fully blocked. **Exactly ``0`` everywhere on a convex closed
        mesh**, which is the cheapest available sanity check on a result.

    Raises
    ------
    ValueError
        If ``n_rays < 1``, ``weight`` is not one of the two names, or ``normals`` has a different
        length from ``points``.
    RuntimeError
        If ``mesh``, ``points`` and ``normals`` are not all on one device.

    See Also
    --------
    [`volumetric_obscurance`][ordito.visibility.volumetric_obscurance]
    [`ordito.visibility.shape_diameter`][ordito.visibility.shape_diameter]
    [`ordito.sample.sample_fibonacci_hemisphere`][ordito.sample.sample_fibonacci_hemisphere]

    Notes
    -----
    MeshLab reports the *complement* of this (higher is more exposed) and leaves it unnormalized —
    its number is roughly ``(1 - occlusion) * n_rays / 4`` for a cosine-weighted bundle over
    ``n_rays`` whole-sphere directions. Its direction set is also its own, so the two agree in
    distribution and in ranking rather than value by value.

    Examples
    --------
    A convex mesh never occludes its own surface, while a point at the centre of a sphere sees
    nothing but the sphere:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    mesh = wp.Mesh(points=vertices, indices=faces)
    occlusion = od.visibility.ambient_occlusion(mesh, vertices)
    print(occlusion.shape, float(occlusion.numpy().max()))  # (642,) 0.0

    center = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    up = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3)
    print(od.visibility.ambient_occlusion(mesh, center, normals=up).numpy())  # [1.]
    ```
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    return _occlusion_bundle(mesh, points, normals, n_rays, weight, 0.0, max_t, "ambient_occlusion")


def volumetric_obscurance(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    normals: wp.array[wp.vec3] | None = None,
    n_rays: int = 64,
    tau: float = 0.1,
    weight: RayWeight = "cosine",
    max_t: float | None = None,
) -> wp.array[wp.float32]:
    """
    Distance-attenuated ambient occlusion: an occluder at range ``t`` counts ``exp(-tau * t)``.

    Iones et al.'s obscurance, and MeshLab's ``compute_scalar_by_volumetric_obscurance``. It exists
    because binary [`ambient_occlusion`][ordito.visibility.ambient_occlusion] treats a wall across
    the room like a crevice wall a millimetre away, which darkens the interior of any closed room
    uniformly and hides exactly the small-scale detail the field is usually wanted for. Attenuating
    by distance keeps the response local.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
        Every ray is traced through this BVH, so its quality sets the cost and not the values:
        [`Trimesh.traced_mesh`][ordito.mesh.Trimesh.traced_mesh] traces a large bundle markedly
        faster than [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] for a costlier build, and
        [`Trimesh.mesh_for_rays`][ordito.mesh.Trimesh.mesh_for_rays] picks between the two.
    points
        ``(m,)`` positions to shade.
    normals
        ``(m,)`` outward unit normals; see
        [`ambient_occlusion`][ordito.visibility.ambient_occlusion] for the default.
    n_rays
        Directions per point.
    tau
        Attenuation rate, in inverse length units of the mesh — so it is **not** scale-invariant,
        and a mesh scaled by ``k`` wants ``tau / k`` for the same result. MeshLab's default is
        ``0.1``, which suits a mesh of extent order 1. As ``tau -> 0`` this becomes
        [`ambient_occlusion`][ordito.visibility.ambient_occlusion]; as ``tau -> inf`` everything
        reads ``0``. Must be positive.
    weight
        Ray weighting, as in [`ambient_occlusion`][ordito.visibility.ambient_occlusion].
    max_t
        Maximum ray length.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` obscurance-weighted occlusion in ``[0, 1]`` on ``points.device``, ``0`` where
        nothing nearby blocks. Exactly ``0`` on a convex closed mesh.

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``n_rays < 1``, ``weight`` is unknown, or ``normals`` is the wrong length.
    RuntimeError
        If ``mesh``, ``points`` and ``normals`` are not all on one device.

    See Also
    --------
    [`ambient_occlusion`][ordito.visibility.ambient_occlusion]

    Examples
    --------
    From the centre of a unit sphere every ray is blocked at distance 1, so the obscurance falls
    from about ``exp(-0.1)`` to ``exp(-1)`` as ``tau`` grows:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    mesh = wp.Mesh(points=vertices, indices=faces)
    center = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    up = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3)
    near = od.visibility.volumetric_obscurance(mesh, center, normals=up, tau=0.1)
    far = od.visibility.volumetric_obscurance(mesh, center, normals=up, tau=1.0)
    print(f"{near.numpy()[0]:.3f} {far.numpy()[0]:.3f}")  # 0.905 0.369
    ```
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}; use ambient_occlusion for the limit")
    return _occlusion_bundle(
        mesh, points, normals, n_rays, weight, tau, max_t, "volumetric_obscurance"
    )


def _occlusion_bundle(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None,
    n_rays: int,
    weight: RayWeight,
    tau: float,
    max_t: float | None,
    name: str,
) -> wp.array[wp.float32]:
    """Shared hemisphere-bundle trace behind both public functions; ``tau <= 0`` means binary."""
    if n_rays < 1:
        raise ValueError(f"{name} requires n_rays >= 1, got {n_rays}")
    if weight not in _WEIGHT_MODES:
        raise ValueError(f"weight must be 'cosine' or 'uniform', got {weight!r}")

    device = points.device
    m = points.size
    # Resolved before the empty-input early return: a mismatched ``normals`` length is a caller
    # bug independent of how many points there are.
    normals = _resolve_normals(mesh, points, normals, name)
    # `wp.empty`, not `wp.zeros`: either bundle kernel writes `out_occlusion[i]` unconditionally
    # for every block, so nothing ever reads the zero-fill.
    out_occlusion = _launch.empty(m, dtype=wp.float32, device=device)
    if m == 0:
        return out_occlusion

    # The hemisphere lattice (``sample_fibonacci_hemisphere``'s, a band of height 1).
    directions, *limits = _bundle_inputs(mesh, points, max_t, n_rays, 1.0)
    # One block per point, lanes over the bundle -- see ``kernel_visibility.BUNDLE_BLOCK`` for why
    # this wins over a thread per point, and why the width is 64.
    # Binary occlusion asks only whether each ray is blocked (an any-hit trace); obscurance needs
    # the nearest hit's distance.
    trace = [mesh.id, points, normals, directions, _WEIGHT_MODES[weight], *limits, out_occlusion]
    if tau > 0.0:
        kernel, inputs = kernel_visibility.obscurance, [*trace[:4], wp.float32(tau), *trace[4:]]
    elif m >= _OCCLUSION_POINT_MAJOR_FROM and wp.get_device(device).is_cuda:
        _launch.launch_tiled(
            kernel_visibility.occlusion_point_major,
            dim=(-(-m // kernel_visibility.POINT_MAJOR_POINTS),),
            inputs=[*trace[:3], _morton_order(points), *trace[3:]],
            block_dim=kernel_visibility.POINT_MAJOR_BLOCK,
            device=device,
        )
        return out_occlusion
    else:
        kernel, inputs = kernel_visibility.occlusion, trace
    _launch.launch_tiled(
        kernel, dim=(m,), inputs=inputs, block_dim=kernel_visibility.BUNDLE_BLOCK, device=device
    )
    return out_occlusion


def _morton_order(points: wp.array[wp.vec3]) -> wp.array[wp.int32]:
    """
    ``(m,)`` permutation of ``points`` into Morton (Z-order) order over their own box.

    The order the point-major occlusion kernel walks a large cloud in, so a block's points are
    spatial neighbours. Its first ``m`` entries are the permutation; the buffer is the radix sort's
    double width.
    """
    device = points.device
    m = points.size
    corners = _launch.full(6, math.inf, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_reduce.minmax_vec3_chunked,
        dim=kernel_reduce.chunks_1d(m),
        inputs=[points, corners],
        device=device,
    )
    keys = _launch.empty(2 * m, dtype=wp.int32, device=device)
    order = _launch.empty(2 * m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_visibility.morton_point_order,
        dim=m,
        inputs=[points, corners],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, count=m, end_bit=30)
    return order


def shape_diameter(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    normals: wp.array[wp.vec3] | None = None,
    n_rays: int = 64,
    cone_angle: float = math.pi / 3.0,
    trim: float = 1.0,
    max_t: float | None = None,
) -> wp.array[wp.float32]:
    """
    Shape diameter function: local thickness of the volume, from an inward cone of rays.

    Shapira et al.'s SDF and MeshLab's ``compute_scalar_by_shape_diameter_function_per_vertex``. A
    cone of ``n_rays`` rays is fired *into* the volume about ``-normal``, each is traced to the far
    side, and the result is the cosine-weighted mean of those distances **after discarding the
    outliers** — the rays that escaped through a nearby opening or crossed the entire model, which
    would otherwise dominate the average near any concavity.

    This is the many-ray generalization of
    [`thickness(method="ray")`][ordito.visibility.thickness]: at ``n_rays=1`` with a vanishing
    ``cone_angle`` the bundle collapses to the inward normal and the two agree to float32. The extra
    rays are what make it stable — a single ray through a thin sliver of geometry reads a thickness
    the neighbourhood does not have — and the reason it is the quantity used for skeleton extraction
    and part segmentation rather than the one-ray version.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``). Should be closed: on an open surface the rays
        that find nothing on the far side are simply absent from the mean.
        Every ray is traced through this BVH, so its quality sets the cost and not the values:
        [`Trimesh.traced_mesh`][ordito.mesh.Trimesh.traced_mesh] traces a large bundle markedly
        faster than [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] for a costlier build, and
        [`Trimesh.mesh_for_rays`][ordito.mesh.Trimesh.mesh_for_rays] picks between the two.
    points
        ``(m,)`` surface positions to measure at, normally the mesh's own vertices.
    normals
        ``(m,)`` **outward** unit normals; the cone opens along ``-normals``. When ``None`` they are
        taken from the closest face of ``mesh``. Pass
        [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="area"`` for a smooth
        field over a mesh's own vertices.
    n_rays
        Rays per point. MeshLab's default is ``64``, which is this one. Note that the single ray of
        ``n_rays=1`` is the *centroid* of the cone's Fibonacci lattice rather than its axis, so it
        only coincides with the inward normal as ``cone_angle`` goes to zero.
    cone_angle
        Half-angle of the cone in **radians**, measured from the inward normal. The default
        ``pi / 3`` (60 degrees) is Shapira's 120-degree cone. Must be in ``(0, pi / 2]`` — beyond
        that the cone reaches around to the outside of the surface and the distances stop meaning
        thickness.
    trim
        Keep only rays whose distance is within ``trim`` standard deviations of the mean before
        averaging. ``1.0`` (the default) is Shapira's rule; a large value keeps everything and turns
        this into a plain weighted mean. Must be non-negative.
    max_t
        Maximum ray length. When ``None``, the diagonal of the AABB enclosing the mesh and the query
        points.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` diameters in the mesh's own length units on ``points.device``. ``inf`` at a point
        where no ray in the cone found the far side at all.

    Raises
    ------
    ValueError
        If ``n_rays < 1``, ``cone_angle`` is outside ``(0, pi / 2]``, ``trim < 0``, or ``normals``
        has a different length from ``points``.
    RuntimeError
        If ``mesh``, ``points`` and ``normals`` are not all on one device.

    See Also
    --------
    [`thickness`][ordito.visibility.thickness]
    [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere]
    [`ordito.visibility.ambient_occlusion`][ordito.visibility.ambient_occlusion]
    [`ordito.sample.sample_fibonacci_cone`][ordito.sample.sample_fibonacci_cone]

    Notes
    -----
    MeshLab's ``cone_amplitude`` parameter is a **no-op** in the 2025.07 build — its output is
    byte-identical at ``90`` and ``120`` degrees — and its trimming rule is not the one documented
    in the paper, so its values differ from these by a roughly constant factor on a given mesh.
    Compare against it by rank rather than by value; the exactly-checkable statements are the
    reduction to [`thickness`][ordito.visibility.thickness] at ``n_rays=1`` and the sphere. There a
    ray at angle ``theta`` from the inward normal crosses a chord of ``2 R cos(theta)``, so the
    result tends to the diameter ``2 R`` only as ``cone_angle`` goes to zero; the default cone
    reads the weighted, trimmed mean of those chords, about ``1.5 R``.

    Examples
    --------
    A narrow cone through a unit sphere measures its diameter; the default wide cone averages
    shorter chords:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    mesh = wp.Mesh(points=vertices, indices=faces)
    normals = od.vertices.vertex_normals(vertices, faces)
    diameter = od.visibility.shape_diameter(mesh, vertices, normals=normals, cone_angle=0.1)
    print(f"{diameter.numpy().min():.2f} {diameter.numpy().max():.2f}")  # 1.99 1.99
    diameter = od.visibility.shape_diameter(mesh, vertices, normals=normals)
    print(f"{diameter.numpy().mean():.2f}")  # 1.51
    ```
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    if n_rays < 1:
        raise ValueError(f"shape_diameter requires n_rays >= 1, got {n_rays}")
    if not 0.0 < cone_angle <= math.pi / 2.0:
        raise ValueError(f"cone_angle must be in (0, pi / 2] radians, got {cone_angle}")
    if trim < 0.0:
        raise ValueError(f"trim must be non-negative, got {trim}")

    device = points.device
    m = points.size
    # See `_occlusion_bundle`'s identical comment: resolved before the empty-input check so a
    # mismatched `normals` length is caught even when there is nothing to measure.
    normals = _resolve_normals(mesh, points, normals, "shape_diameter")
    if m == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    # ``sample_fibonacci_cone``'s lattice.
    directions, *limits = _bundle_inputs(mesh, points, max_t, n_rays, 1.0 - math.cos(cone_angle))
    # Distances are kept so the trimming pass can revisit them against a mean the first pass had not
    # finished computing; re-tracing instead would double the only expensive part of the kernel.
    scratch = odt.empty_2d((m, n_rays), wp.float32, device=device)
    out_diameter = _launch.empty(m, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_visibility.shape_diameter,
        dim=(m,),
        inputs=[
            mesh.id,
            points,
            normals,
            directions,
            *limits,
            wp.float32(trim),
            scratch,
            out_diameter,
        ],
        block_dim=kernel_visibility.BUNDLE_BLOCK,
        device=device,
    )
    return out_diameter


def thickness(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    exterior: bool = False,
    normals: wp.array[wp.vec3] | None = None,
    method: Literal["max_sphere", "ray"] = "max_sphere",
) -> wp.array[wp.float32]:
    """
    Local thickness of the volume at each point, by one inward ray or one tangent sphere.

    A dispatcher over the module's other two inward measures, and the cheapest of the three: it
    takes a single piece of evidence per point where
    [`shape_diameter`][ordito.visibility.shape_diameter] fires a whole cone and trims the outliers.
    ``method="max_sphere"`` returns twice the radius of
    [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere], which answers the question for a
    *volume* rather than along a direction; ``method="ray"`` returns
    [`longest_ray`][ordito.ray.longest_ray] along ``-normals`` (or ``+normals`` with
    ``exterior=True``), which is one ray and therefore reads whatever thin sliver of geometry it
    happens to cross.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(m,)`` surface positions to measure at.
    exterior
        When ``True`` measure outward (the reach) instead of inward (the thickness).
    normals
        ``(m,)`` **outward** unit normals. When ``None`` they are taken from the closest face of
        ``mesh``; see [`ambient_occlusion`][ordito.visibility.ambient_occlusion].
    method
        ``"max_sphere"`` (default) or ``"ray"``; see the summary for the difference.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` thickness values in the mesh's own length units on ``points.device``. ``inf``
        where the measure is unbounded (no far side was found).

    Raises
    ------
    ValueError
        If ``method`` is neither ``"max_sphere"`` nor ``"ray"``, or if ``normals`` has a different
        length from ``points``.
    RuntimeError
        If ``mesh``, ``points`` and ``normals`` are not all on one device.

    See Also
    --------
    [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere]
    [`shape_diameter`][ordito.visibility.shape_diameter]
        The stable many-ray generalization of ``method="ray"``.
    [`longest_ray`][ordito.ray.longest_ray]

    Examples
    --------
    At the centre of the top face of a ``1 x 2 x 4`` box, the largest inscribed sphere is bounded
    by the narrow side, while the ray along the normal crosses the full height:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box(extents=(1.0, 2.0, 4.0))
    mesh = wp.Mesh(points=vertices, indices=faces)
    points = wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3)
    normals = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3)
    sphere = od.visibility.thickness(mesh, points, normals=normals)
    ray = od.visibility.thickness(mesh, points, normals=normals, method="ray")
    print(f"{sphere.numpy()[0]:.3f} {ray.numpy()[0]:.3f}")  # 1.000 4.000
    ```
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    if method not in _THICKNESS_METHODS:
        raise ValueError(f"method must be one of {sorted(_THICKNESS_METHODS)}, got {method!r}")

    if method == "max_sphere":
        # ``max_tangent_sphere``'s defaults; the diameter is stored as the sphere's radius is
        # formed, so it takes no pass of its own.
        normals = _tangent_sphere_normals(mesh, points, normals, "max_tangent_sphere")
        if points.size == 0:
            return _launch.empty(0, dtype=wp.float32, device=points.device)
        return _tangent_spheres(mesh, points, normals, not exterior, 1e-6, 100, 2.0)[1]

    # No ray length: an unbounded ``longest_ray`` finds the same first hit as one bounded by the
    # box enclosing the mesh and the points, and a miss is ``inf`` either way. The sign flip is
    # folded into the ray kernel.
    normals = _resolve_normals(mesh, points, normals, "thickness")
    out_thickness = _launch.empty(points.size, dtype=wp.float32, device=points.device)
    if points.size > 0:
        _launch.launch(
            kernel_visibility.longest_ray_along_normals,
            dim=points.size,
            inputs=[
                wp.uint64(mesh.id),
                points,
                normals,
                not exterior,
                wp.float32(math.inf),
                wp.float32(TOLERANCE_PLANAR),
                out_thickness,
            ],
            device=points.device,
        )
    return out_thickness


def max_tangent_sphere(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    inwards: bool = True,
    normals: wp.array[wp.vec3] | None = None,
    threshold: float = 1e-6,
    max_iter: int = 100,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """
    Find the center and radius of the sphere tangent to the mesh at each point.

    Implements the shrinking-sphere algorithm (Inui et al. 2016): iteratively
    finds the largest sphere tangent to the mesh at ``points`` with no
    non-tangential intersections.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` surface points.
    inwards
        If ``True``, sphere grows inward (into the mesh interior). If ``False``,
        grows outward.
    normals
        ``(m,)`` unit surface normals at ``points``. If ``None``, computed from
        the closest triangle. A caller-supplied array is normalized defensively (the default
        never needs it), since a non-unit normal would silently scale the reported radius.
    threshold
        Convergence threshold as a fraction of the **mesh's own** bounding-box diagonal, not the
        (possibly larger) box enclosing the query points too -- a query far outside the mesh
        should not loosen how tightly the sphere converges.
    max_iter
        Maximum number of shrink iterations.

    Returns
    -------
    centers
        ``(m,)`` sphere center positions.
    radii
        ``(m,)`` sphere radii. ``inf`` when the sphere is unbounded.

    Raises
    ------
    ValueError
        If ``normals`` has a different length from ``points``.
    RuntimeError
        If ``mesh``, ``points`` and ``normals`` are not all on one device.

    Examples
    --------
    The sphere touching the top face of a ``1 x 2 x 4`` box from inside has the box's half-width as
    its radius; outwards nothing stops it:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box(extents=(1.0, 2.0, 4.0))
    mesh = wp.Mesh(points=vertices, indices=faces)
    points = wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3)
    normals = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3)
    centers, radii = od.visibility.max_tangent_sphere(mesh, points, normals=normals)
    print(f"{radii.numpy()[0]:.3f}", centers.numpy()[0].round(3))  # 0.500 [0.  0.  1.5]
    _, radii = od.visibility.max_tangent_sphere(mesh, points, normals=normals, inwards=False)
    print(radii.numpy())  # [inf]
    ```
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    device = points.device
    m = points.size
    # Resolved (and, for a caller-supplied array, normalized) before the empty-input early return
    # and before the AABB reductions below -- a mismatched or non-unit ``normals`` is a property of
    # ``normals`` alone, and there is no reason to pay for two reductions first only to reject it.
    normals = _tangent_sphere_normals(mesh, points, normals, "max_tangent_sphere")
    if m == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
        )
    return _tangent_spheres(mesh, points, normals, inwards, threshold, max_iter, 1.0)


def _tangent_sphere_normals(
    mesh: wp.Mesh, points: wp.array[wp.vec3], normals: wp.array[wp.vec3] | None, name: str
) -> wp.array[wp.vec3] | None:
    """
    [`_resolve_normals`][ordito.visibility._resolve_normals] for the tangent-sphere passes.

    ``None`` when the closest faces' normals are wanted from a mesh with faces: those
    [`_tangent_spheres`][ordito.visibility._tangent_spheres] forms in its first launch.
    """
    if normals is None and mesh.indices.size > 0:
        return None
    return _resolve_normals(mesh, points, normals, name)


def _tangent_spheres(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None,
    inwards: bool,
    threshold: float,
    max_iter: int,
    radius_scale: float,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """
    [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere] on resolved, non-empty input.

    Radii are stored multiplied by ``radius_scale`` (``2`` for
    [`thickness`][ordito.visibility.thickness], exact), so a diameter needs no second pass.
    ``normals=None`` takes each point's closest-face normal, formed in the launch that reduces the
    boxes.
    """
    device = points.device
    m = points.size
    # ``max_t`` is the diagonal of the box enclosing the mesh *and* the queries, while the
    # convergence threshold is a fraction of the mesh's own diagonal. Both boxes reduce into one
    # twelve-slot corner buffer (``minmax_vec3_chunked``'s packing, the mesh's first and the
    # queries' second) in one launch, and the kernels form both scalars from it in the thread
    # (``tangent_sphere_limits``), so neither is read back.
    #
    # ``max_t`` is part of the answer, not only a bound: the shrink step's closest-point query is
    # capped at it, and a centre farther than that from the mesh misses and stops the sphere where
    # an unbounded query would keep shrinking it, so the two converge to different spheres.
    corners = _launch.full(12, math.inf, dtype=wp.float32, device=device)
    computed_normals = None
    if normals is None:
        computed_normals = _launch.empty(m, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_visibility.tangent_sphere_setup,
        dim=kernel_reduce.chunks_1d(mesh.points.size)
        + kernel_reduce.chunks_1d(m)
        + (0 if computed_normals is None else m),
        inputs=[wp.uint64(mesh.id), mesh.points, mesh.indices, points],
        outputs=[corners, computed_normals],
        device=device,
    )
    if computed_normals is not None:
        normals = computed_normals
    assert normals is not None
    # Every ray-bundle measure in this module normalizes defensively per-thread (see
    # `hemisphere_frame`); this path shrinks a sphere along the ray direction directly with no such
    # guard, so a caller-supplied non-unit normal would silently scale `sphere_center`'s radius away
    # from what the shrink step's own convergence test expects. The seed kernel normalizes it
    # (``ray_direction``, with the sign for ``inwards``, exact), casts the longest ray along it
    # (``ray.longest_ray``'s walk at its default ``planar_tol``) and seeds the sphere, writing the
    # direction for the passes below.
    n_verts = mesh.points.size
    n_support = _launch.zeros(1, dtype=wp.int32, device=device)
    ray_dirs = _launch.empty(m, dtype=wp.vec3, device=device)
    radii = _launch.empty(m, dtype=wp.float32, device=device)
    centers = _launch.empty(m, dtype=wp.vec3, device=device)
    not_converged = _launch.empty(m, dtype=wp.bool, device=device)
    support_indices = _launch.empty(m, dtype=wp.int32, device=device)
    limits = [corners, wp.float64(threshold)]
    _launch.launch(
        kernel_visibility.seed_tangent_spheres,
        dim=m,
        inputs=[
            wp.uint64(mesh.id),
            points,
            normals,
            wp.float32(-1.0 if inwards else 1.0),
            *limits,
            wp.float32(TOLERANCE_PLANAR),
        ],
        outputs=[ray_dirs, radii, centers, not_converged, support_indices, n_support],
        device=device,
    )
    # Escaped rays (typically exterior/reach queries) need the support point of the vertex cloud
    # in the ray direction. The seed lists them -- interior queries usually leave the subset empty
    # -- then one grid-stride packed-argmax pass runs over the vertices for just that subset
    # instead of a serial all-vertices loop per query thread.
    # Host readback: the deferred count sizes the support passes.
    k = int(read_scalar(n_support, 0))
    if k > 0:
        support_indices = odt.as_dense(support_indices[:k])
        n_vert_slices = max(1, (n_verts + ITEMS_PER_QUERY_SLICE - 1) // ITEMS_PER_QUERY_SLICE)
        packed_support = _launch.zeros(k, dtype=wp.uint64, device=device)
        kernel, dim = kernel_visibility.SUPPORT_ARGMAX_SLICED.launch_shape(k, n_vert_slices)
        _launch.launch(
            kernel,
            dim=dim,
            inputs=[
                mesh.points,
                wp.int32(n_verts),
                wp.int32(n_vert_slices),
                ray_dirs,
                support_indices,
                packed_support,
            ],
            device=device,
        )
        _launch.launch(
            kernel_visibility.seed_support_spheres,
            dim=k,
            inputs=[mesh.points, points, ray_dirs, support_indices, packed_support],
            outputs=[radii, centers, not_converged],
            device=device,
        )
    # The spheres shrink independently of each other, so each runs its whole iteration in its own
    # thread: no launch or convergence readback per round.
    _launch.launch(
        kernel_visibility.shrink_tangent_spheres,
        dim=m,
        inputs=[
            wp.uint64(mesh.id),
            points,
            ray_dirs,
            *limits,
            wp.int32(max_iter),
            wp.float32(radius_scale),
            not_converged,
            radii,
            centers,
        ],
        device=device,
    )
    return centers, radii


def _resolve_normals(
    mesh: wp.Mesh, points: wp.array[wp.vec3], normals: wp.array[wp.vec3] | None, name: str
) -> wp.array[wp.vec3]:
    """
    Per-point normals: ``normals`` itself when given (length-checked), else the closest face's.

    Shared by every measure in this module, so the validation and defaulting are written once.

    Raises
    ------
    ValueError
        If ``normals`` has a different length from ``points``.
    """
    m = points.size
    if normals is None:
        return normals_at_closest_faces(mesh, points)
    if normals.size != m:
        raise ValueError(
            f"{name}: normals must have one entry per point, got {normals.size} for {m} points"
        )
    return normals


def _bundle_inputs(
    mesh: wp.Mesh, points: wp.array[wp.vec3], max_t: float | None, n_rays: int, z_span: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.float32, wp.float64]:
    """
    Form the ray-bundle kernels' ``(directions, corners, caller_max_t, offset_scale)`` arguments.

    ``directions`` is the ``n_rays``-point Fibonacci lattice over a band of height ``z_span``, the
    [`ordito.sample`][ordito.sample] generators' own. ``corners`` is the box enclosing the mesh
    and the non-empty ``points`` (one packed box); both are formed in one launch. The kernels form
    the box's diagonal -- the default ray length and the scale of the ray-origin offset
    (``_SURFACE_OFFSET`` of it) -- in the thread, so no search radius is read back.
    ``caller_max_t`` is NaN when ``max_t`` is ``None``.
    """
    device = points.device
    corners = _launch.full(6, math.inf, dtype=wp.float32, device=device)
    directions = _launch.empty(n_rays, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_visibility.bundle_setup,
        dim=kernel_reduce.chunks_1d(mesh.points.size)
        + kernel_reduce.chunks_1d(points.size)
        + n_rays,
        inputs=[mesh.points, points, wp.float32(z_span)],
        outputs=[corners, directions],
        device=device,
    )
    caller_max_t = math.nan if max_t is None else max_t
    return directions, corners, wp.float32(caller_max_t), wp.float64(_SURFACE_OFFSET)
