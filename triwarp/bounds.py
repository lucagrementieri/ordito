"""
Bounding boxes: building them, and selecting or cropping with them.

Two halves. The **constructors** -- [`aabb`][triwarp.bounds.aabb],
[`aabb_union`][triwarp.bounds.aabb_union], [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal]
and [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] -- reduce a cloud to a box. The
**queries** -- [`points_in_aabb`][triwarp.bounds.points_in_aabb],
[`points_in_obb`][triwarp.bounds.points_in_obb] and the
[`crop_points`][triwarp.bounds.crop_points] / [`crop_mesh`][triwarp.bounds.crop_mesh] pair -- run
the other way, testing points against a box someone already has. The two halves compose directly:
every constructor's return is a query's argument, in that layout, which is why an oriented box is a
``(rotation, min_bound, max_bound)`` triple here rather than an object.

[`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal] is the one entry point here that
exists for another module's sake rather than its own: it is the default search radius every
mesh query in the package derives from its input's extent.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, require_same_device
from triwarp.kernels import array as kernel_array
from triwarp.kernels import bounds as kernel_bounds
from triwarp.kernels import predicates as kernel_predicates
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels import scatter as kernel_scatter

# Points reduced per thread by the per-candidate extent reduction in
# [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]. Long, and one value for both
# devices, because the *candidate* dimension already fills the device -- this is the per-query
# regime [`ITEMS_PER_SLICE_CUDA`][triwarp.constants.ITEMS_PER_SLICE_CUDA]'s note describes, not the
# global-reduction one.
ITEMS_PER_CANDIDATE_SLICE = 256

# Cloud size at which [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] first runs
# [`convex_superset_mask`][triwarp.points.convex_superset_mask] and searches only the survivors.
# The mask keeps every convex-hull vertex and the extent reduction is decided by hull vertices.
CONVEX_PREFILTER_MIN_POINTS = 100_000

# Placeholder frame for the axis-aligned arm of ``kernels/bounds.box_flags``, which never reads it.
_UNUSED_ROTATION = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def aabb(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Axis-aligned bounding box of ``points`` (component-wise min / max).

    The reduction is [`minmax`][triwarp.reduce.minmax]'s ``wp.vec3`` path: one chunked kernel
    writes both corners into a single six-element buffer, read back once. This wrapper contributes
    only the empty-input ``(+inf, -inf)`` convention, where ``minmax`` raises.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(min_bound, max_bound)`` with ``min_bound[i] ≤ p[i] ≤ max_bound[i]`` for every
        point ``p`` and axis ``i``. If ``n == 0``, ``min_bound`` is ``(+inf, …)`` and
        ``max_bound`` is ``(-inf, …)``.

    See Also
    --------
    [`aabb_union`][triwarp.bounds.aabb_union]
    [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal]
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The other direction: which points a box already in hand contains.
    [`triwarp.reduce.minmax`][triwarp.reduce.minmax]
    """
    if points.size == 0:
        return (wp.vec3(math.inf, math.inf, math.inf), wp.vec3(-math.inf, -math.inf, -math.inf))
    return tw.reduce.minmax(points)


def aabb_union(
    a_min: wp.vec3, a_max: wp.vec3, b_min: wp.vec3, b_max: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """
    Smallest axis-aligned bounding box enclosing two given boxes.

    Parameters
    ----------
    a_min, a_max
        Min/max corners of the first box.
    b_min, b_max
        Min/max corners of the second box.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(min_bound, max_bound)`` of the box enclosing both inputs.

    See Also
    --------
    [`aabb`][triwarp.bounds.aabb]
    [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal]
    """
    # Componentwise ``min`` / ``max`` in plain Python rather than ``wp.min`` / ``wp.max``: this
    # function does no device work at all, so Warp's Python-scope builtin dispatch was its entire
    # cost, and the plain spelling is several times cheaper and byte-identical. A ``wp.vec3``
    # indexes to a native ``float``, which is what makes the plain builtins applicable.
    return (
        wp.vec3(min(a_min[0], b_min[0]), min(a_min[1], b_min[1]), min(a_min[2], b_min[2])),
        wp.vec3(max(a_max[0], b_max[0]), max(a_max[1], b_max[1]), max(a_max[2], b_max[2])),
    )


def enclosing_diagonal(points: wp.array[wp.vec3], other: wp.array[wp.vec3] | None = None) -> float:
    """
    Diagonal of the axis-aligned box enclosing one or two point sets.

    The default search radius for every mesh query in the package: no closest point, ray hit or
    tangent sphere can be further away than the diagonal of a box containing both the surface and
    the query points, so it is the smallest bound that is guaranteed not to cut an answer off. Pass
    ``other`` whenever the queries may lie outside the mesh's own box, which is the usual case --
    with ``points`` alone a query far outside would get a radius too short to reach the surface.

    Parameters
    ----------
    points
        ``(n,)`` positions as ``wp.vec3``, typically a mesh's vertices.
    other
        Optional second ``(m,)`` set, typically the query points. ``None`` or empty measures
        ``points`` alone.

    Returns
    -------
    float
        ``|max_bound - min_bound|`` of the union box. ``inf`` when ``points`` is empty, since an
        empty box has infinite negative extent -- callers guard on the point count first.

    Raises
    ------
    RuntimeError
        If ``points`` and ``other`` are not all on one device.

    See Also
    --------
    [`aabb`][triwarp.bounds.aabb]
    [`aabb_union`][triwarp.bounds.aabb_union]
    """
    require_same_device(points=points, other=other)
    # Both clouds reduce into **one** corner buffer, so the union costs one readback rather than
    # two boxes and an ``aabb_union``: ``minmax_vec3_chunked`` accumulates with ``wp.atomic_min``
    # into a buffer the caller seeds, so a second launch over a second cloud continues the same
    # reduction. Reducing a concatenation instead is slower -- the union buffer is an allocation
    # and a copy of both clouds, against one launch that is flat in its ``dim``.
    device = points.device
    corners = _launch.full(6, math.inf, dtype=wp.float32, device=device)
    for cloud in (points, other):
        if cloud is None or cloud.size == 0:
            continue
        _launch.launch(
            kernel_reduce.minmax_vec3_chunked,
            dim=kernel_reduce.chunks_1d(cloud.size),
            inputs=[cloud, corners],
            device=device,
        )
    # Slots 3..5 hold the *negated* upper corner; see the kernel. An all-empty input leaves the
    # seeded ``inf`` in place, so ``upper - lower`` is ``-inf`` componentwise and the length is
    # ``inf`` -- the documented empty-input answer, reached without a branch.
    #
    # Taken in NumPy, on the buffer the readback already produced, rather than through two
    # ``wp.vec3`` constructions and ``wp.length``: every Warp operator and builtin at *Python*
    # scope routes through Warp's builtin dispatch (``inspect.signature().bind()`` per operand),
    # an order of magnitude dearer than this. Section 13.1 has the cost model.
    corners_np = corners.numpy()
    return float(np.linalg.norm(-corners_np[3:] - corners_np[:3]))


def points_in_aabb(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.int32]:
    """
    Return the indices of the points inside an axis-aligned box, boundary included.

    The bounded selection primitive, and the one that needs no spatial index at all: every point is
    tested independently against six planes, so the cost is one pass over the cloud whatever the
    box holds. A tree would only help if the *box* were the thing being searched for -- see
    [`triwarp.neighbors.query_bvh_box`][triwarp.neighbors.query_bvh_box] for that direction.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    min_bound, max_bound
        Opposite corners of the box, as [`aabb`][triwarp.bounds.aabb] returns them. An empty box
        (any ``min_bound[i] > max_bound[i]``) selects nothing.

    Returns
    -------
    wp.array[wp.int32]
        Ascending indices into ``points`` on ``points.device``. Empty when nothing is inside.

    Examples
    --------
    ```python
    inside = tw.bounds.points_in_aabb(pts, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0))
    ```

    Notes
    -----
    Containment is **inclusive**: a point exactly on a face, edge or corner of the box is selected.
    A point with a ``nan`` coordinate is *not* selected, and neither is one at an infinity. Both
    conventions differ from [`half_space_mask`][triwarp.points.half_space_mask], whose test is
    strict.

    See Also
    --------
    [`points_in_aabb_mask`][triwarp.bounds.points_in_aabb_mask]
        The same selection as a boolean mask, for a caller that wants to combine it.
    [`points_in_obb`][triwarp.bounds.points_in_obb]
        The oriented counterpart.
    [`crop_points`][triwarp.bounds.crop_points]
        This plus the gather, when the points themselves are wanted.
    [`aabb`][triwarp.bounds.aabb]
    """
    return _box_indices(points, min_bound, max_bound, None)


def points_in_aabb_mask(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.bool]:
    """
    Flag the points inside an axis-aligned box, boundary included.

    The mask form of [`points_in_aabb`][triwarp.bounds.points_in_aabb], for a caller that wants to
    intersect the selection with another predicate before compacting it -- combining masks costs
    one ``wp.map`` where combining index lists costs a set operation.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    min_bound, max_bound
        Opposite corners of the box, as [`aabb`][triwarp.bounds.aabb] returns them.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``. ``True`` marks a point to **keep**, the same sense
        as [`triwarp.points.half_space_mask`][triwarp.points.half_space_mask].

    See Also
    --------
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The index form, and where the boundary and ``nan`` conventions are documented.
    [`points_in_obb_mask`][triwarp.bounds.points_in_obb_mask]
    [`triwarp.array.flatnonzero`][triwarp.array.flatnonzero]
    """
    out_mask = _launch.empty(points.size, dtype=wp.bool, device=points.device)
    if points.size == 0:
        return out_mask

    _launch.map(kernel_predicates.is_in_aabb, points, min_bound, max_bound, out=out_mask)
    return out_mask


def points_in_obb(
    points: wp.array[wp.vec3], rotation: wp.mat33, min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.int32]:
    """
    Return the indices of the points inside an oriented box, boundary included.

    Takes the box exactly as [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]
    returns it, so the two compose without a transpose or a corner rebuild.

    Parameters
    ----------
    points
        ``(n,)`` positions in world space as ``wp.vec3``.
    rotation
        World-to-box frame as ``wp.mat33``, i.e. its **rows** are the box axes in world
        coordinates. A point's box coordinates are ``rotation * p``.
    min_bound, max_bound
        Extent of the box along its own axes, in box coordinates.

    Returns
    -------
    wp.array[wp.int32]
        Ascending indices into ``points`` on ``points.device``. Empty when nothing is inside.

    Examples
    --------
    ```python
    rotation, lower, upper = tw.bounds.oriented_bounding_box(v, 256)
    inside = tw.bounds.points_in_obb(v, rotation, lower, upper)
    ```

    Notes
    -----
    One rotated point per thread, then the identical six comparisons
    [`points_in_aabb`][triwarp.bounds.points_in_aabb] makes -- the two share one predicate, so the
    inclusive boundary and the ``nan`` exclusion documented there hold here too, and the two
    functions cannot drift apart on either. Passing the identity as ``rotation`` reproduces the
    axis-aligned answer exactly.

    Being a proper rotation, the frame's inverse is its transpose, so nothing here inverts a
    matrix; a general affine box would, and is not what this takes.

    See Also
    --------
    [`points_in_obb_mask`][triwarp.bounds.points_in_obb_mask]
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The axis-aligned counterpart, and the shared conventions.
    [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]
        Where the ``(rotation, min_bound, max_bound)`` triple comes from.
    """
    return _box_indices(points, min_bound, max_bound, rotation)


def points_in_obb_mask(
    points: wp.array[wp.vec3], rotation: wp.mat33, min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.bool]:
    """
    Flag the points inside an oriented box, boundary included.

    The mask form of [`points_in_obb`][triwarp.bounds.points_in_obb]; see
    [`points_in_aabb_mask`][triwarp.bounds.points_in_aabb_mask] for why the mask form exists.

    Parameters
    ----------
    points
        ``(n,)`` positions in world space as ``wp.vec3``.
    rotation
        World-to-box frame as ``wp.mat33``, rows being the box axes
        ([`points_in_obb`][triwarp.bounds.points_in_obb] documents the convention).
    min_bound, max_bound
        Extent of the box along its own axes, in box coordinates.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``. ``True`` marks a point to **keep**.

    See Also
    --------
    [`points_in_obb`][triwarp.bounds.points_in_obb]
        The index form, and where the conventions are documented.
    [`points_in_aabb_mask`][triwarp.bounds.points_in_aabb_mask]
    """
    out_mask = _launch.empty(points.size, dtype=wp.bool, device=points.device)
    if points.size == 0:
        return out_mask

    _launch.map(kernel_predicates.is_in_obb, points, rotation, min_bound, max_bound, out=out_mask)
    return out_mask


def crop_points(
    points: wp.array[wp.vec3],
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    *,
    rotation: wp.mat33 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the points inside a box, returning the survivors and where they came from.

    Parameters
    ----------
    points
        ``(n,)`` positions in world space as ``wp.vec3``.
    min_bound, max_bound
        Opposite corners of the box: in world coordinates when ``rotation`` is ``None``, in box
        coordinates otherwise.
    rotation
        World-to-box frame as ``wp.mat33``, rows being the box axes. ``None`` -- the default --
        crops to the **axis-aligned** box, which is the same test with one transform fewer rather
        than a different rule; pass the frame
        [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] returns to crop to an
        oriented one.

    Returns
    -------
    kept : wp.array[wp.vec3]
        The points inside the box, in ascending index order, on ``points.device``.
    indices : wp.array[wp.int32]
        Their indices in ``points``, so a caller can crop a parallel per-point attribute with
        [`triwarp.array.gather`][triwarp.array.gather] instead of re-running the test. This is the
        second return rather than an option because the compaction has already computed it.

    Examples
    --------
    ```python
    kept, indices = tw.bounds.crop_points(pts, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0))
    ```

    See Also
    --------
    [`crop_mesh`][triwarp.bounds.crop_mesh]
        The mesh counterpart, which must also decide what to do with a straddling face.
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The selection alone, when the points are not needed.
    [`triwarp.array.gather`][triwarp.array.gather]
    """
    device = points.device
    inclusive, n_kept = _scanned_box_flags(points, min_bound, max_bound, rotation)
    kept = _launch.empty(n_kept, dtype=wp.vec3, device=device)
    indices = _launch.empty(n_kept, dtype=wp.int32, device=device)
    if n_kept > 0:
        _launch.launch(
            kernel_bounds.compact_scanned_points,
            dim=points.size,
            inputs=[inclusive, points, kept, indices],
            device=device,
        )
    return kept, indices


def crop_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    *,
    rotation: wp.mat33 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the faces whose three vertices all lie inside a box, reindexed from zero.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    min_bound, max_bound
        Opposite corners of the box: in world coordinates when ``rotation`` is ``None``, in box
        coordinates otherwise.
    rotation
        World-to-box frame as ``wp.mat33``, rows being the box axes; ``None`` crops to the
        axis-aligned box. Same argument as
        [`crop_points`][triwarp.bounds.crop_points]'s.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``, carrying only the vertices
        the kept faces reference.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    ```python
    rotation, lower, upper = tw.bounds.oriented_bounding_box(v, 256)
    sub_v, sub_f = tw.bounds.crop_mesh(v, f, lower, upper, rotation=rotation)
    ```

    Notes
    -----
    A face straddling the box boundary is **dropped**, not clipped: the result is a subset of the
    input's triangles, so the crop never introduces a vertex the input did not have and the cut
    edge is ragged at the triangle scale. That is what makes the operation a selection rather than
    a boolean -- for a flush cut, intersect the mesh with the box as a solid instead.

    To keep the straddling faces instead, take the mask and pass it on directly, which is the
    ``face_mode="any"`` rule:

    ```python
    mask = tw.bounds.points_in_aabb_mask(v, wp.vec3(-1.0, -1.0, 0.0), wp.vec3(1.0, 1.0, 3.0))
    sub_v, sub_f = tw.selection.submesh_from_vertex_mask(v, f, mask, face_mode="any")
    ```

    See Also
    --------
    [`crop_points`][triwarp.bounds.crop_points]
    [`triwarp.selection.submesh_from_vertex_mask`][triwarp.selection.submesh_from_vertex_mask]
        The face selection this delegates to, where ``face_mode`` is exposed.
    """
    require_same_device(vertices=vertices, faces=faces)
    vertex_mask = _box_mask(vertices, min_bound, max_bound, rotation)
    return tw.selection.submesh_from_vertex_mask(vertices, faces, vertex_mask, face_mode="all")


def _box_mask(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3, rotation: wp.mat33 | None
) -> wp.array[wp.bool]:
    """Dispatch ``crop_mesh`` onto the axis-aligned or the oriented mask."""
    if rotation is None:
        return points_in_aabb_mask(points, min_bound, max_bound)
    return points_in_obb_mask(points, rotation, min_bound, max_bound)


def _box_indices(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3, rotation: wp.mat33 | None
) -> wp.array[wp.int32]:
    """Ascending indices of the points inside the box: the index forms' shared body."""
    inclusive, n_inside = _scanned_box_flags(points, min_bound, max_bound, rotation)
    out_indices = _launch.empty(n_inside, dtype=wp.int32, device=points.device)
    if n_inside > 0:
        _launch.launch(
            kernel_scatter.scatter_index_where_scanned,
            dim=points.size,
            inputs=[inclusive, out_indices],
            device=points.device,
        )
    return out_indices


def _scanned_box_flags(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3, rotation: wp.mat33 | None
) -> tuple[wp.array[wp.int32], int]:
    """
    Inclusive scan of the inside-the-box flags, scanned in place, and the count inside.

    The flags are written straight as ``int32`` and the one tail read sizes the caller's output,
    so neither a ``wp.bool`` mask nor its widening copy is ever allocated.
    """
    device = points.device
    n = points.size
    inclusive = _launch.empty(n, dtype=wp.int32, device=device)
    if n == 0:
        return inclusive, 0
    _launch.launch(
        kernel_bounds.box_flags,
        dim=n,
        inputs=[
            points,
            _UNUSED_ROTATION if rotation is None else rotation,
            min_bound,
            max_bound,
            0 if rotation is None else 1,
            inclusive,
        ],
        device=device,
    )
    _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # One 4-byte tail read: the count sizes every output.
    return inclusive, int(read_scalar(inclusive))


def oriented_bounding_box(
    points: wp.array[wp.vec3],
    rotations: int = 4096,
    objective: Literal["volume", "surface_area", "diagonal"] = "volume",
    refine_iterations: int = 8,
) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
    """

    Smallest bounding box over sampled orientations, sharpened by local refinement, and its frame.

    The box is searched, not solved, in two phases. The **global phase** scores ``rotations``
    candidate orientations from the **Super-Fibonacci spiral** [Alexa 2022], a low-discrepancy
    sampling of ``SO(3)``, with the identity appended as the last candidate, scored in parallel.
    The **refinement phase** then takes the best-scoring candidates from up to four mutually distant
    basins and runs ``refine_iterations`` trust-region rounds around each: every round scores a
    low-discrepancy ball of perturbed frames whose angular radius starts at the global grid's
    covering radius and shrinks by a factor of 0.4 per round, keeping each chain's best. Refinement
    is monotone (each
    chain re-scores its own base), so the answer is never worse than the global phase's and never
    worse than [`aabb`][triwarp.bounds.aabb]; with ``refine_iterations=0`` and ``rotations=1`` it
    reproduces the axis-aligned box exactly.

    Cost is ``rotations * len(points)`` point transforms for the global phase plus
    ``refine_iterations * 512 * len(points)`` for refinement — but past
    ``CONVEX_PREFILTER_MIN_POINTS`` a
    [`convex_superset_mask`][triwarp.points.convex_superset_mask] prefilter first drops every
    point that provably cannot touch the box, so both phases run on the few hull-candidate
    survivors and the cost stops growing with the cloud (the box is identical: the mask keeps
    every hull vertex and the extents are order-independent reductions over them).


    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    rotations
        Number of global candidates to score, ``>= 1``. The refinement default makes raising this
        past a few thousand pointless: the global phase only needs to land *inside* the optimum's
        basin, and refinement does the rest.
    objective
        Which box quantity to minimize: ``"volume"``, ``"surface_area"``, or ``"diagonal"`` (the
        squared diagonal length, which has the same minimizer as the length).
    refine_iterations
        Trust-region rounds after the global phase, ``>= 0``. ``0`` disables refinement and returns
        the pure sampled answer.

    Returns
    -------
    rotation : wp.mat33
        World-to-box frame, i.e. its **rows** are the box axes in world coordinates. A world point
        ``p`` has box coordinates ``rotation * p``, and a box corner maps back with
        ``wp.transpose(rotation) * corner``. ``rotation`` is a proper rotation (orthonormal,
        determinant ``+1``).
    min_bound, max_bound
        Extent of ``points`` along the box axes, in box coordinates -- so the box is
        ``{transpose(rotation) * q : min_bound <= q <= max_bound}`` and its side lengths are
        ``max_bound - min_bound``. For ``n == 0`` these are ``(+inf, …)`` and ``(-inf, …)`` and
        ``rotation`` is the identity, matching
        [`aabb`][triwarp.bounds.aabb].

    Raises
    ------
    ValueError
        If ``rotations < 1``, ``refine_iterations < 0``, or ``objective`` is not one of the three
        named above.

    Notes
    -----
    The whole search runs on the device: candidate frames, extents, the objective, the argmin that
    picks the chains, the perturbed frames each round composes, and the per-chain selection. One row
    of the winning chain comes back at the end, and that is the only host traffic.

    **The result is a converged local minimum, not a certified global one.** Certifying the true
    minimum-volume box requires the exact-arithmetic search over the convex hull's face and edge
    events (O'Rourke's rotating-calipers family), and triwarp deliberately has no exact convex hull.
    What the sampling-plus-refinement search gives up is the *certificate*, not the volume: the
    global phase lands in the optimum's basin whenever the grid's covering radius resolves it,
    refinement converges within the basin, and the refined default ties or beats hull-based
    reference methods on volume across a range of test shapes. A pathological cloud whose optimum
    basin is narrower than the covering radius of ``rotations`` samples can still hide its box from
    this search; raise ``rotations`` if the input is a near-symmetric polyhedron far from any
    sampled orientation.

    Neither hull-based reference is an oracle for the minimum either: on a stretched icosahedron the
    refined search returns **less** volume than both, because the hull-face-flush restriction misses
    optima whose box touches only edges and vertices.

    See Also
    --------
    [`aabb`][triwarp.bounds.aabb]
    [`points_in_obb`][triwarp.bounds.points_in_obb]
        Takes this triple as returned, to select the points a box contains.
    [`trimesh.bounds.oriented_bounds`][]
    ``igl.oriented_bounding_box``
    """
    if rotations < 1:
        raise ValueError(f"rotations must be >= 1, got {rotations}")
    if refine_iterations < 0:
        raise ValueError(f"refine_iterations must be >= 0, got {refine_iterations}")
    if objective not in ("volume", "surface_area", "diagonal"):
        raise ValueError(
            f'objective must be "volume", "surface_area" or "diagonal", got {objective!r}'
        )

    n = points.size
    if n == 0:
        return (
            wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            wp.vec3(math.inf, math.inf, math.inf),
            wp.vec3(-math.inf, -math.inf, -math.inf),
        )

    if n >= CONVEX_PREFILTER_MIN_POINTS:
        # Identical box, decided at the threshold above: only hull vertices can touch an
        # enclosing box, the mask keeps all of them, and min/max extents do not care about the
        # discarded interior points.
        points = _compact_masked_points(points, tw.points.convex_superset_mask(points))
        n = points.size

    device = points.device
    # The axes kernel also seeds each candidate's six extent slots, so ``corners`` is uninitialised
    # until then and every slot is written before the extents kernel reads it.
    axes = _launch.empty(rotations, dtype=wp.mat33, device=device)
    corners = _launch.empty(6 * rotations, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_bounds.oriented_box_candidate_axes,
        dim=rotations,
        inputs=[rotations, axes, corners],
        device=device,
    )

    n_slices = max(1, (n + ITEMS_PER_CANDIDATE_SLICE - 1) // ITEMS_PER_CANDIDATE_SLICE)
    _score_extents_into(points, axes, n_slices, corners)

    loss = _launch.empty(rotations, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_bounds.oriented_box_losses,
        dim=rotations,
        inputs=[corners, _BOX_OBJECTIVES[objective], loss],
        device=device,
    )
    # One chain when there is nothing to refine: the seeding walk's first pick is the global
    # argmin, so it stops after a single round and the row it writes is already the answer.
    n_chains = 1 if refine_iterations == 0 else _REFINE_CHAINS
    chains = _launch.empty(n_chains, dtype=wp.mat33, device=device)
    chain_state = twt.as_array2d(
        _launch.empty((n_chains, kernel_bounds.BOX_STATE_COLUMNS), dtype=wp.float32, device=device),
        wp.float32,
    )
    _launch.launch_tiled(
        kernel_bounds.oriented_box_seed_chains,
        dim=(1,),
        inputs=[
            loss,
            axes,
            corners,
            wp.int32(min(_REFINE_WINDOW, rotations)),
            wp.int32(n_chains),
            chains,
            chain_state,
        ],
        block_dim=_SEED_BLOCK_DIM,
        device=device,
    )

    if refine_iterations > 0:
        _refine_box(points, rotations, objective, refine_iterations, n_slices, chains, chain_state)
    return _best_chain(chain_state)


# Refinement geometry: up to four chains cover distinct basins of the sampled landscape (a
# near-symmetric shape has several near-tied optima), and 128 frames per chain per round keeps a
# full refinement at the cost of one extra 512-candidate global pass per round. Four is also the
# ceiling: ``oriented_box_seed_chains`` carries its picks in a ``wp.vec4i``.
_REFINE_CHAINS = 4
_REFINE_CANDIDATES = 128

# Leading candidates the seeding walk considers. Adjacent spiral candidates score adjacently, so
# the best few are usually one basin sampled several times and a window this wide is what gives the
# spread test distinct basins to find; beyond it the losses are too far above the winner's to be
# worth a chain.
_REFINE_WINDOW = 32

# The seeding walk is a single block, so this is its whole width -- and the walk is dominated by
# the scan rather than by its per-round ``block_argmin`` pair, so it wants a wide one: the sweep is
# flat from here up, with the chains identical at every width. 256 rather than the 128 that ties
# with it because it is Warp's default, so the module is not loaded a second time for a second
# ``block_dim`` (section 2.5).
_SEED_BLOCK_DIM = 256
_BOX_OBJECTIVES: dict[str, wp.int32] = {
    "volume": kernel_bounds.BOX_OBJECTIVE_VOLUME,
    "surface_area": kernel_bounds.BOX_OBJECTIVE_SURFACE_AREA,
    "diagonal": kernel_bounds.BOX_OBJECTIVE_DIAGONAL,
}


def _compact_masked_points(points: wp.array[wp.vec3], mask: wp.array[wp.bool]) -> wp.array[wp.vec3]:
    """
    Compact the points ``mask`` keeps, in index order.

    ``gather(points, flatnonzero(mask))`` without the index list or the gathered copy between them.
    """
    device = points.device
    n = points.size
    inclusive = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(kernel_array.bool_flags, dim=n, inputs=[mask, inclusive], device=device)
    _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
    # The kept count sizes the output.
    n_kept = int(read_scalar(inclusive))
    kept = _launch.empty(n_kept, dtype=wp.vec3, device=device)
    if n_kept > 0:
        _launch.launch(
            kernel_bounds.compact_scanned_points,
            dim=n,
            inputs=[inclusive, points, kept, None],
            device=device,
        )
    return kept


def _refine_box(
    points: wp.array[wp.vec3],
    rotations: int,
    objective: str,
    refine_iterations: int,
    n_slices: int,
    chains: wp.array[wp.mat33],
    chain_state: twt.Array2dFloat32,
) -> None:
    """
    Trust-region refinement of the seeded chains: shrink a low-discrepancy ball per round.

    The chains live on the device throughout. Each round generates a ball of perturbed frames
    around every chain, scores them with the same extent kernel as the global phase, and keeps each
    chain's best in place; the last delta of every ball is the identity, which re-scores the base
    and is what makes a chain monotone. Nothing is read back until the caller unpacks the winning
    row.
    """
    device = points.device
    n_chains = chains.size
    total = n_chains * _REFINE_CANDIDATES
    axes = _launch.empty(total, dtype=wp.mat33, device=device)
    corners = _launch.empty(6 * total, dtype=wp.float32, device=device)
    objective_code = _BOX_OBJECTIVES[objective]

    # Start at the covering radius of the global grid: the sampled winner is at most about this
    # far from its basin's optimum, and each round shrinks the radius.
    sigma = 2.0 * (math.pi**2 / max(rotations, 2)) ** (1.0 / 3.0)
    for _ in range(refine_iterations):
        _launch.launch(
            kernel_bounds.oriented_box_refine_axes,
            dim=total,
            inputs=[
                chains,
                wp.float64(sigma / math.pi),
                wp.int32(_REFINE_CANDIDATES),
                axes,
                corners,
            ],
            device=device,
        )
        _score_extents_into(points, axes, n_slices, corners)
        _launch.launch(
            kernel_bounds.oriented_box_select_chains,
            dim=n_chains,
            inputs=[
                corners,
                axes,
                wp.int32(_REFINE_CANDIDATES),
                objective_code,
                chains,
                chain_state,
            ],
            device=device,
        )
        # 0.4 rather than 0.5: a flat-flush optimum is a *kink*, so the volume error is linear in
        # the final angular resolution, and the 127-frame ball resolves ~sigma/5 per round, so
        # shrinking by 0.4 never outruns what a round can see.
        sigma *= 0.4


def _score_extents_into(
    points: wp.array[wp.vec3], axes: twt.ArrayNd, n_slices: int, corners: wp.array[wp.float32]
) -> None:
    """Fill ``corners`` with the cloud's extent in every candidate frame, six packed slots each."""
    # The kernel only ever ``atomic_min``s into ``corners``, so it must arrive holding ``+inf``:
    # both axes kernels seed a candidate's slots as they write its frame, which is what lets the
    # refinement reuse one buffer across its rounds.
    _launch.launch(
        kernel_bounds.oriented_box_extents,
        dim=(int(axes.shape[0]), n_slices),
        inputs=[points, axes, n_slices, corners],
        device=points.device,
    )


def _best_chain(chain_state: twt.Array2dFloat32) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
    """
    Unpack the best chain's row -- the search's one host readback.

    Seeding writes every chain's loss as ``+inf`` and refinement replaces it, so with no refinement
    this picks the single seeded row and with refinement the chain that descended furthest. The
    column layout is ``kernels/bounds.write_chain_state``'s.
    """
    state_np = chain_state.numpy()
    row = state_np[int(state_np[:, 0].argmin())]
    return (wp.mat33(*row[7:16]), wp.vec3(*row[1:4]), wp.vec3(*row[4:7]))
