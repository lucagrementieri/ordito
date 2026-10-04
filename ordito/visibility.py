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

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.bounds import enclosing_diagonal
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
    # bug independent of how many points there are, and `_resolve_normals_and_radius` handles
    # `m == 0` on its own (an empty `points` measures the mesh's own box alone).
    normals, diagonal = _resolve_normals_and_radius(mesh, points, normals, name)
    # `wp.empty`, not `wp.zeros`: either bundle kernel writes `out_occlusion[i]` unconditionally
    # for every block, so nothing ever reads the zero-fill.
    out_occlusion = _launch.empty(m, dtype=wp.float32, device=device)
    if m == 0:
        return out_occlusion

    directions = od.sample.sample_fibonacci_hemisphere(n_rays, device=device)
    # One block per point, lanes over the bundle -- see ``kernel_visibility.BUNDLE_BLOCK`` for why
    # this wins over a thread per point, and why the width is 64.
    # Binary occlusion asks only whether each ray is blocked (an any-hit trace); obscurance needs
    # the nearest hit's distance.
    trace = [
        mesh.id,
        points,
        normals,
        directions,
        _WEIGHT_MODES[weight],
        wp.float32(max_t if max_t is not None else diagonal),
        wp.float32(_SURFACE_OFFSET * max(diagonal, 1e-12)),
        out_occlusion,
    ]
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
    lower, upper = od.bounds.aabb(points)
    extent = [float(upper[k]) - float(lower[k]) for k in range(3)]
    inv_extent = wp.vec3(*(1023.0 / e if e > 0.0 else 0.0 for e in extent))
    keys = _launch.empty(2 * m, dtype=wp.int32, device=device)
    order = _launch.empty(2 * m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_visibility.morton_point_order,
        dim=m,
        inputs=[points, lower, inv_extent],
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
    reduction to [`thickness`][ordito.visibility.thickness] at ``n_rays=1`` and the analytic
    ``2 R`` on a sphere.
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
    normals, diagonal = _resolve_normals_and_radius(mesh, points, normals, "shape_diameter")
    if m == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    directions = od.sample.sample_fibonacci_cone(n_rays, cone_angle, device=device)
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
            wp.float32(max_t if max_t is not None else diagonal),
            wp.float32(_SURFACE_OFFSET * max(diagonal, 1e-12)),
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
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    if method not in _THICKNESS_METHODS:
        raise ValueError(f"method must be one of {sorted(_THICKNESS_METHODS)}, got {method!r}")

    if method == "max_sphere":
        _centers, radii = max_tangent_sphere(mesh, points, inwards=not exterior, normals=normals)
        _launch.map(wp.mul, radii, wp.float32(2.0), out=radii)
        return radii

    # No ray length: an unbounded ``longest_ray`` finds the same first hit as one bounded by the
    # box enclosing the mesh and the points, and a miss is ``inf`` either way.
    normals = _resolve_normals(mesh, points, normals, "thickness")
    ray_dirs = normals
    if not exterior:
        ray_dirs = _launch.empty(points.size, dtype=wp.vec3, device=points.device)
        _launch.map(wp.neg, normals, out=ray_dirs)
    return od.ray.longest_ray(mesh, points, ray_dirs)


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
    """
    require_same_device(mesh=mesh, points=points, normals=normals)
    device = points.device
    m = points.size
    # Resolved (and, for a caller-supplied array, normalized) before the empty-input early return
    # and before the AABB reductions below -- a mismatched or non-unit ``normals`` is a property of
    # ``normals`` alone, and there is no reason to pay for two reductions first only to reject it.
    normals = _resolve_normals(mesh, points, normals, "max_tangent_sphere")
    if m == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
        )
    # Every ray-bundle measure in this module normalizes defensively per-thread (see
    # `hemisphere_frame`); this path shrinks a sphere along `normals` directly with no such guard,
    # so a caller-supplied non-unit normal silently scales `sphere_center`'s radius away from what
    # `step_sphere_shrink`'s own convergence test expects. One `wp.map` fixes it for the whole
    # iterative loop; `normals_at_closest_faces`'s own output is already unit, so this is a no-op
    # there, but cheap enough not to special-case.
    # The sign for ``inwards`` rides in the same map (``ray_direction``), which is exact.
    ray_dirs = _launch.empty(m, dtype=wp.vec3, device=device)
    _launch.map(
        kernel_visibility.ray_direction, normals, wp.float32(-1.0 if inwards else 1.0), out=ray_dirs
    )

    # ``max_t`` needs the box enclosing the mesh *and* the queries, while the convergence threshold
    # is a fraction of the mesh's own diagonal. Both boxes reduce into one twelve-slot corner
    # buffer (``minmax_vec3_chunked``'s packing, the mesh's first and the queries' second) in one
    # launch, read back once; the union is then the componentwise extreme of the two, exact in any
    # order.
    #
    # ``max_t`` is part of the answer, not only a bound: the shrink step's closest-point query is
    # capped at it, and a centre farther than that from the mesh misses and stops the sphere where
    # an unbounded query would keep shrinking it, so the two converge to different spheres.
    corners = _launch.full(12, math.inf, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_reduce.minmax_vec3_pair_chunked,
        dim=kernel_reduce.chunks_1d(mesh.points.size) + kernel_reduce.chunks_1d(m),
        inputs=[mesh.points, points, wp.int32(1)],
        outputs=[corners],
        device=device,
    )
    # Slots 3..5 and 9..11 hold the *negated* upper corners. ``math.dist`` on plain floats rather
    # than ``wp.length`` of a Warp vector difference: a Warp operator or builtin at Python scope
    # routes through builtin dispatch, several times dearer. It computes in float64 where
    # ``wp.length`` is float32, i.e. the correctly-rounded answer for float32 corners.
    c = read_values(corners, 0, 12)
    mesh_lower, mesh_upper = c[0:3], [-x for x in c[3:6]]
    union_lower = [min(a, b) for a, b in zip(mesh_lower, c[6:9], strict=True)]
    union_upper = [max(a, -b) for a, b in zip(mesh_upper, c[9:12], strict=True)]
    max_t = math.dist(union_lower, union_upper)
    mesh_diagonal = math.dist(mesh_lower, mesh_upper)

    distances = od.ray.longest_ray(mesh, points, ray_dirs, max_t=max_t)

    n_verts = mesh.points.size
    radii = _launch.empty(m, dtype=wp.float32, device=device)
    not_converged = _launch.empty(m, dtype=wp.bool, device=device)
    needs_support = _launch.empty(m, dtype=wp.bool, device=device)
    centers = _launch.empty(m, dtype=wp.vec3, device=device)
    _launch.map(
        kernel_visibility.init_sphere_radii_finite,
        distances,
        points,
        ray_dirs,
        out=[radii, not_converged, needs_support, centers],
    )
    # Escaped rays (typically exterior/reach queries) need the support point of the vertex
    # cloud in the ray direction. Compact them first — interior queries usually leave the
    # subset empty — then run one grid-stride packed-argmax pass over the vertices for just
    # that subset instead of a serial all-vertices loop per query thread.
    support_indices = od.array.flatnonzero(needs_support)
    k = support_indices.size
    if k > 0:
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
            kernel_visibility.init_sphere_radii_support,
            dim=k,
            inputs=[
                mesh.points,
                points,
                ray_dirs,
                support_indices,
                packed_support,
                radii,
                not_converged,
                centers,
            ],
            device=device,
        )

    convergence_threshold = wp.float32(threshold * mesh_diagonal)

    # All per-iteration buffers are preallocated once and ping-ponged (the step kernel writes
    # every lane, passing converged state through). The convergence count is checked every
    # iteration on purpose: an extra iteration runs a BVH closest-point query for every point still
    # shrinking, far more expensive than the 8-byte readback the check costs.
    new_radii = _launch.empty(m, dtype=wp.float32, device=device)
    new_centers = _launch.empty(m, dtype=wp.vec3, device=device)
    new_nc = _launch.empty(m, dtype=wp.bool, device=device)
    # Slot ``r`` is the not-converged count after round ``r - 1``, accumulated by the step kernel
    # itself, so each round's test is one 4-byte read with no reduction launch in front of it.
    # Slot 0 is the seed's count, the one reduction the loop still needs.
    n_not_converged = _launch.zeros(max_iter + 1, dtype=wp.int32, device=device)
    n_active = od.reduce.sum(not_converged)

    for round_index in range(max_iter):
        if round_index > 0:
            n_active = int(read_scalar(n_not_converged, round_index))
        if n_active == 0:
            break

        _launch.launch(
            kernel_visibility.step_sphere_shrink,
            dim=m,
            inputs=[
                mesh.id,
                points,
                ray_dirs,
                centers,
                radii,
                wp.float32(max_t),
                convergence_threshold,
                not_converged,
                wp.int32(round_index + 1),
                new_radii,
                new_centers,
                new_nc,
                n_not_converged,
            ],
            device=device,
        )
        radii, new_radii = new_radii, radii
        centers, new_centers = new_centers, centers
        not_converged, new_nc = new_nc, not_converged

    return centers, radii


def _resolve_normals(
    mesh: wp.Mesh, points: wp.array[wp.vec3], normals: wp.array[wp.vec3] | None, name: str
) -> wp.array[wp.vec3]:
    """
    Per-point normals: ``normals`` itself when given (length-checked), else the closest face's.

    Split out of [`_resolve_normals_and_radius`][ordito.visibility._resolve_normals_and_radius]
    so [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere] and ``thickness(method="ray")``
    can share the validation and defaulting without also paying for the union-box diagonal: the
    first derives its own from a reduction it performs anyway, and the second needs none.

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


def _resolve_normals_and_radius(
    mesh: wp.Mesh, points: wp.array[wp.vec3], normals: wp.array[wp.vec3] | None, name: str
) -> tuple[wp.array[wp.vec3], float]:
    """
    Per-point normals and the search radius every measure in this module needs.

    ``normals`` defaults to the closest face's normal, which is right for points on the surface and
    meaningless off it; the radius is the diagonal of the box enclosing both the mesh and the
    queries, so no ray or sphere is cut short.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(m,)`` positions being measured at.
    normals
        ``(m,)`` outward unit normals, or ``None`` to take them from the closest face.
    name
        Calling function's name, used in the error message.

    Returns
    -------
    tuple[wp.array[wp.vec3], float]
        ``(normals, diagonal)``.

    Raises
    ------
    ValueError
        If ``normals`` has a different length from ``points``.
    """
    return _resolve_normals(mesh, points, normals, name), enclosing_diagonal(mesh.points, points)
