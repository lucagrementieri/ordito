"""
Point-set acceleration structures (BVH/HashGrid) and raw neighbor queries.

**The accelerator is a keyword, not a function name.** There are two questions here -- "everything
within ``r``" and "the ``k`` nearest" -- and each is one function:
[`query_ball`][triwarp.neighbors.query_ball] (with
[`query_ball_count`][triwarp.neighbors.query_ball_count] and
[`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets] for the count alone and the
flat CSR form) and [`query_nearest`][triwarp.neighbors.query_nearest]. Which broad phase runs is a
[`QueryBackend`][triwarp.neighbors.QueryBackend] keyword -- ``"hashgrid"`` or ``"bvh"`` -- or is
inferred from a prebuilt structure passed as ``accelerator``. Both backends are **exact and return
the same answer**; the choice is a cost one, and ``query_nearest``'s docstring carries guidance on
which to pick. The kernel side is one warp-uniform kernel branching on an ``ACCEL_*`` selector.

The BVH-only queries keep the structure in their names, because naming it is informative rather
than redundant there: these index arbitrary **bounds** rather than points, which a hash grid
cannot, and they are broad phase only -- every bound the query region reaches is returned, with no
narrow-phase filter on the primitive's own geometry.
[`query_bvh_ball`][triwarp.neighbors.query_bvh_ball] is the ball and
[`query_bvh_box`][triwarp.neighbors.query_bvh_box] the per-query box; both return the same flat
``(indices, offsets)`` pair, so neither carries a ``_with_offsets`` suffix -- unlike
``query_ball_with_offsets``, which needs one to separate it from ``query_ball``'s dense form.
**Prefer the ball whenever the predicate is a ball**: it is not merely tighter, its traversal is
substantially cheaper per candidate than the box one (see ``query_bvh_ball``).

Also home to [`geodesic_ball`][triwarp.neighbors.geodesic_ball], the surface-aware counterpart to
the spatial ball queries here: it returns the same CSR ``(indices, offsets)`` shape but walks the
mesh edge graph, so it excludes vertices that are close in space yet across a fold of the surface.

[`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance] is the one derived
quantity rather than a raw query: the per-point distance to the closest other point, which is what a
cloud's scale is normally estimated from.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Literal, cast, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, read_values, require_same_device
from triwarp.constants import INT64_MAX, TILE_1D
from triwarp.kernels import neighbors as kernel_neighbors
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels.algorithms import bfs as kernel_bfs

# An axis counts towards a point cloud's effective dimension when its extent is at least this
# fraction of the largest one. Below that the cloud is flat (or collinear) along that axis and the
# volume-based density estimate in [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
# would invert a (near-)zero volume.
_FLAT_AXIS_FRACTION = 1e-6

# Cost of one hash-grid cell probe, expressed in linear-scan point tests. Sets where
# [`query_nearest`][triwarp.neighbors.query_nearest] stops widening its cell walk
# under ``backend="hashgrid"``
# and scans exactly instead; see ``_knn_widest_grid_radius``.
_CELL_PROBE_POINTS = 600

# Smallest cloud a ``k > 1`` hash-grid search defers its unfinished self-query rows to the BVH at.
# Below it the scan those rows would otherwise take is short enough that the tree build, the
# second launch and the readback deciding them cost more than they save.
_KNN_DEFER_MIN_POINTS = 8192

# Default hash-grid resolution: ``wp.HashGrid`` folds cell coordinates modulo its bins, so a cloud
# occupying more cells than the table has bins aliases unrelated cells onto one bucket and every
# probe walks their points too. ``_resolve_grid_bins`` sizes the table so ``bins ** 3`` is at least
# ``_GRID_BINS_PER_POINT`` bins a point -- occupied cells never exceed the point count, and at a
# ball radius of a couple of spacings they are about a third of it -- and never below
# ``_GRID_BINS_MIN``, which every cloud up to about a million points keeps. The cost of a table
# too small grows with the cloud; past a few bins per occupied cell it is flat, while a build
# clears both ``int32`` per-bin tables every time, so ``_GRID_BINS_MAX`` caps them at ~450 MB.
_GRID_BINS_MIN = 128
_GRID_BINS_MAX = 384
_GRID_BINS_PER_POINT = 2
_GRID_BINS_STEP = 32


def bvh_from_points(points: wp.array[wp.vec3], leaf_size: int = 4) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over ``points`` for radius queries.

    Each leaf stores the same geometry as ``points`` (degenerate bounds via a clone),
    matching the broad-phase pattern used by
    [`query_ball`][triwarp.neighbors.query_ball] and
    [`query_nearest`][triwarp.neighbors.query_nearest].

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to ``warp.Bvh``.

    Returns
    -------
    warp.Bvh
        BVH suited for the ``query_ball*`` and ``query_nearest`` family; pass it as their
        ``accelerator``, which selects ``backend="bvh"`` by its type.

    See Also
    --------
    [`query_ball`][triwarp.neighbors.query_ball]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    """
    return wp.Bvh(points, points, leaf_size=leaf_size)


def mesh_from_points(points: wp.array[wp.vec3]) -> wp.Mesh:
    """
    Build a ``warp.Mesh`` over ``points`` whose triangles each collapse onto one point.

    Its closest-face query is then a nearest-point query: the closest point of a triangle whose
    three corners coincide is that corner, so ``wp.mesh_query_point_no_sign`` returns the nearest
    point's index as its face. This is the tree
    [`query_nearest`][triwarp.neighbors.query_nearest] answers ``k = 1`` with under
    ``backend="bvh"``, and its cost follows the tree's depth rather than how far a query sits from
    the cloud.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``, ``n >= 1``. The mesh aliases them rather than copying,
        so do not mutate them while it is in use.

    Returns
    -------
    warp.Mesh
        Mesh of ``n`` degenerate triangles, triangle ``i`` on ``points[i]``.

    Raises
    ------
    ValueError
        If ``points`` is empty: a ``warp.Mesh`` with no triangles corrupts the CUDA allocator.

    See Also
    --------
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    """
    n = points.size
    if n == 0:
        raise ValueError("mesh_from_points needs at least one point")
    corners = _launch.empty(3 * n, dtype=wp.int32, device=points.device)
    _launch.launch(
        kernel_neighbors.point_triangle_indices, dim=3 * n, inputs=[corners], device=points.device
    )
    # ``lbvh``: the other constructors build this tree several times slower, for no faster query.
    # One triangle per leaf: a leaf test here is Warp's full closest-point-on-triangle routine on a
    # triangle that is one point, so every extra primitive in a leaf is wasted arithmetic: one per
    # leaf is the fastest query, for a build no slower and the same distances.
    return wp.Mesh(points=points, indices=corners, bvh_constructor="lbvh", bvh_leaf_size=1)


def hashgrid_from_points(
    points: wp.array[wp.vec3], radius: float, grid_bins: int | None = None
) -> wp.HashGrid:
    """
    Build a 3D hash grid over ``points`` for radius queries.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    radius
        Cell size passed to ``warp.HashGrid.build`` and used by
        the ``query_ball*`` and ``query_nearest`` kernels under ``backend="hashgrid"``.
    grid_bins
        Resolution of the hash grid along each axis. ``None`` (the default) sizes it from the point
        count: 128 up to about a million points, then growing with the cube root of the count so
        the table keeps at least two bins a point, up to 384. The grid wraps cell coordinates
        modulo this resolution, so a cloud spanning more cells than it has bins shares buckets
        between distant cells; the answer of every query is unaffected, only its cost.

    Returns
    -------
    warp.HashGrid
        Hash grid suited for the ``query_ball*`` and ``query_nearest`` family,
        and related kernels. The cell width is recorded on the returned object as
        ``cell_width``, which
        [`query_nearest`][triwarp.neighbors.query_nearest] reads back to size its
        search (``warp.HashGrid`` itself does not keep it).

    See Also
    --------
    [`query_ball_count`][triwarp.neighbors.query_ball_count]
    [`query_ball`][triwarp.neighbors.query_ball]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    """
    n = points.size
    bins = _resolve_grid_bins(grid_bins, n)
    grid = wp.HashGrid(bins, bins, bins, device=points.device)
    grid.reserve(n)
    grid.build(points, radius)
    grid.cell_width = float(radius)
    return grid


def bvh_from_bounds(
    lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], leaf_size: int = 4
) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over axis-aligned bounds.

    Each primitive ``i`` is represented by ``lower[i]`` and ``upper[i]`` corner
    positions, suitable for
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball] and
    [`query_bvh_box`][triwarp.neighbors.query_bvh_box]
    broad-phase intersection tests.

    Parameters
    ----------
    lower
        ``(n,)`` minimum corner of each bound as ``wp.vec3``.
    upper
        ``(n,)`` maximum corner of each bound as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to ``warp.Bvh``.

    Returns
    -------
    warp.Bvh
        BVH suited for AABB intersection queries.

    Raises
    ------
    RuntimeError
        If ``lower`` and ``upper`` are not all on one device.
    """
    require_same_device(lower=lower, upper=upper)
    return wp.Bvh(lower, upper, leaf_size=leaf_size)


def query_bvh_ball(
    bvh: wp.Bvh, queries: wp.array[wp.vec3], radius: float
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Low-level BVH ball query: primitive indices in one flat buffer plus offsets.

    For each query center ``q``, tests whether every primitive bound in ``bvh`` comes within
    ``radius`` of ``q`` -- the exact bound-to-point distance, so the returned set is the ball's and
    not its enclosing cube's. Like
    [`query_bvh_box`][triwarp.neighbors.query_bvh_box] and unlike
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets], there is no
    narrow-phase filter on the primitive's own geometry: every bound the ball reaches is returned.

    Parameters
    ----------
    bvh
        Pre-built BVH from [`bvh_from_bounds`][triwarp.neighbors.bvh_from_bounds]
        or [`bvh_from_points`][triwarp.neighbors.bvh_from_points].
    queries
        ``(m, 3)`` query centers stored as ``wp.vec3``.
    radius
        Ball radius about each query center.

    Returns
    -------
    candidate_indices_flat, offsets
        Exactly the packing
        [`query_bvh_box`][triwarp.neighbors.query_bvh_box] returns: ``offsets`` is the
        length-``m + 1`` total-terminated prefix sum of per-query hit counts, and query ``k`` owns
        ``candidate_indices_flat[offsets[k] : offsets[k + 1]]``.

    Raises
    ------
    RuntimeError
        If ``bvh`` and ``queries`` are not all on one device.

    Notes
    -----
    Prefer this over the cube query wherever the caller's predicate is a ball, and not only for the
    candidates it does not return: the ball traversal is *substantially cheaper per candidate* than
    the box one, enough that the cube query loses even when handed the inscribed cube and therefore
    strictly fewer candidates. The margin grows with the radius, which is where a cube broad phase
    hurts most.

    See Also
    --------
    [`query_bvh_box`][triwarp.neighbors.query_bvh_box]
        The same packing for an axis-aligned **box**, given per query. A box is a different
        predicate and not a looser ball -- it admits corners the ball excludes -- so reach for it
        when the neighbourhood really is axis-aligned, and for this when it is a ball.
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]
        The same ball, over a **point** cloud and with a narrow-phase distance filter.
    """
    require_same_device(bvh=bvh, queries=queries)
    device = queries.device
    m = queries.size

    if m == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.zeros(1, dtype=wp.int32, device=device),
        )

    # The counts land behind the leading zero of the ``m + 1`` offsets buffer and are scanned there
    # in place, so no separate count buffer is allocated.
    offsets = _launch.zeros(m + 1, dtype=wp.int32, device=device)
    hit_counts = twt.as_dense(offsets[1:])
    # ``wp.uint64(bvh.id)`` explicitly: unlike ``wp.launch``, ``wp.map`` infers a bare Python int
    # scalar's dtype as ``wp.int32`` rather than matching the mapped @wp.func's declared parameter
    # type, and a mismatched dtype is a codegen-time TypeError, not a silent truncation.
    _launch.map(
        kernel_neighbors.ball_count_in_bounds,
        wp.uint64(bvh.id),
        queries,
        wp.float32(radius),
        out=hit_counts,
    )

    _launch.array_scan(hit_counts, out_array=hit_counts, inclusive=True)
    total_hits = int(read_scalar(offsets))
    if total_hits == 0:
        return _launch.empty(0, dtype=wp.int32, device=device), offsets

    candidate_indices_flat = _launch.empty(total_hits, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_neighbors.query_bvh_ball_neighbors,
        dim=m,
        inputs=[queries, bvh.id, wp.float32(radius), offsets, candidate_indices_flat],
        device=device,
    )

    return candidate_indices_flat, offsets


def query_bvh_box(
    bvh: wp.Bvh, query_lower: wp.array[wp.vec3], query_upper: wp.array[wp.vec3]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Primitives overlapping one axis-aligned box **per query**, in the same CSR packing.

    The box counterpart of the ball queries, and the per-query generalization of
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball]: each query
    carries its own ``(lower, upper)`` corners rather than sharing one cube size. On a BVH built by
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points], whose leaf bounds are degenerate, a hit
    means the point is **inside** the box, so the answer is exact and no narrow phase is needed; on
    one built by [`bvh_from_bounds`][triwarp.neighbors.bvh_from_bounds] it is box-versus-box
    overlap, like every other broad phase here.

    Parameters
    ----------
    bvh
        Pre-built BVH over points or bounds.
    query_lower, query_upper
        ``(m,)`` ``wp.vec3`` corners of the query boxes, one pair per query.

    Returns
    -------
    candidate_indices_flat, offsets
        Query ``k`` owns ``candidate_indices_flat[offsets[k] : offsets[k + 1]]``, with ``offsets``
        the length-``m + 1`` total-terminated prefix sum of per-query hit counts.

    Raises
    ------
    ValueError
        If ``query_lower`` and ``query_upper`` do not have the same length.
    RuntimeError
        If ``bvh``, ``query_lower`` and ``query_upper`` are not all on one device.

    Notes
    -----
    The test is **inclusive** on every face: a point exactly on a box face (lower or upper) is
    inside it. A box with any ``upper < lower`` component matches nothing, and that is not checked
    -- the check would cost a host readback per call to reject a caller error whose answer is
    already empty.

    There is no list-returning sibling, so the name carries no ``_with_offsets`` suffix: this
    query's consumers are kernels that recover the owning query from ``offsets``, and a Python list
    of per-query arrays would add ``O(m)`` host slicing to a query whose whole point is that it is
    batched.

    See Also
    --------
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball]
        The same packing over the same bounds for a **ball**, and the one to prefer wherever the
        predicate is a ball rather than an axis-aligned region -- not only because the box admits
        corners the ball excludes, but because the box *traversal* is itself several times more
        expensive per candidate returned: the cost is dominated by the node test itself, not by
        how many candidates come back.
    [`triwarp.points.half_space_mask`][triwarp.points.half_space_mask]
        The unbounded counterpart: selection by one plane rather than by a box.
    """
    require_same_device(bvh=bvh, query_lower=query_lower, query_upper=query_upper)
    device = query_lower.device
    m = query_lower.size
    if query_upper.size != m:
        raise ValueError("query_lower and query_upper must have the same length")

    if m == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.zeros(1, dtype=wp.int32, device=device),
        )

    offsets = _launch.zeros(m + 1, dtype=wp.int32, device=device)
    hit_counts = twt.as_dense(offsets[1:])
    # ``wp.uint64(bvh.id)`` explicitly -- see the same cast in ``query_bvh_ball``
    # above.
    _launch.map(
        kernel_neighbors.aabb_count_in_bounds,
        wp.uint64(bvh.id),
        query_lower,
        query_upper,
        out=hit_counts,
    )

    _launch.array_scan(hit_counts, out_array=hit_counts, inclusive=True)
    total_hits = int(read_scalar(offsets))
    if total_hits == 0:
        return _launch.empty(0, dtype=wp.int32, device=device), offsets

    candidate_indices_flat = _launch.empty(total_hits, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_neighbors.query_bvh_box_neighbors,
        dim=m,
        inputs=[query_lower, query_upper, bvh.id, offsets, candidate_indices_flat],
        device=device,
    )

    return candidate_indices_flat, offsets


# Which broad phase a query runs. Public and named because it appears in four public signatures;
# both values are exact and return the same answer, so this is a cost choice -- see
# ``query_nearest``'s docstring for guidance.
QueryBackend = Literal["hashgrid", "bvh"]


@overload
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    grid_bins: int | None = ...,
    leaf_size: int = ...,
    return_sorted: bool = ...,
    copy: bool = ...,
) -> tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]: ...
@overload
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    grid_bins: int | None = ...,
    leaf_size: int = ...,
    return_sorted: bool = ...,
    copy: bool = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    grid_bins: int | None = None,
    leaf_size: int = 4,
    return_sorted: bool = False,
    copy: bool = False,
) -> (
    tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]
    | tuple[wp.array[wp.int32], wp.array[wp.float32]]
):
    """
    Find all data points within distance ``r`` of each query center (per-query arrays).

    Same exact search as [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``;
    only the spatial index differs, and which index is a keyword rather than a function name (see
    ``backend``). High-level wrapper around
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets].

    Unlike SciPy's object array of lists, multi-query results are two Python lists of length ``m``,
    each element a rank-1 ``wp.array`` for that query. A single ``wp.vec3`` query returns one
    ``(indices, distances)`` pair directly (not wrapped in lists). This is
    [`split`][triwarp.array.split] over
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]'s packed result; call
    that instead for one flat buffer plus offsets on device.

    !!! note "The returned arrays are views"
        Each query's arrays slice the two flat buffers the packed query produced, which costs no
        device memory and no launches. Holding on to one query's arrays keeps the *whole* buffers
        alive, and writing into one writes into the shared allocation. Pass ``copy=True`` for
        independent buffers.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``
        (treated as one query).
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    accelerator
        A prebuilt ``warp.HashGrid`` or ``warp.Bvh`` over ``points``, to reuse across queries. It
        selects the backend by its own type, so ``backend`` is redundant when this is given and
        raises if it names the other one.
    backend
        Which broad phase to use when ``accelerator`` is ``None``: ``"hashgrid"`` (the default)
        enumerates the cells overlapping the query cube, ``"bvh"`` descends an AABB tree. The
        narrow phase and the answer are identical -- this is a cost choice, not a semantic one.
    grid_bins
        Grid resolution when building a hash grid; ``None`` sizes it from the point count, as in
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]. Ignored under
        ``backend="bvh"`` and whenever ``accelerator`` is given.
    leaf_size
        Maximum primitives per leaf when building a BVH. Ignored under ``backend="hashgrid"``,
        whenever ``accelerator`` is given, and at ``k == 1``, where the tree is a
        [`mesh_from_points`][triwarp.neighbors.mesh_from_points].
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance. If ``False``,
        order follows the broad phase's traversal (undefined ordering).
    copy
        Return independent buffers instead of views into the packed result.

    Returns
    -------
    neighbor_indices, neighbor_distances
        If ``queries`` has ``m`` rows: ``list[wp.array[wp.int32]]`` and
        ``list[wp.array[wp.float32]]``, each of length ``m``. Element ``k`` lists neighbors
        of ``queries[k]`` (indices into ``points`` and distances ``‖points[i] - q‖₂``).

        If ``queries`` is a single ``wp.vec3``: two rank-1 arrays (possibly length 0), not lists.

        Empty ``points`` yields empty neighbor arrays and per-query empty slices; duplicate
        neighbors are not produced.

    Raises
    ------
    ValueError
        If ``backend`` is neither name, or contradicts the type of ``accelerator``.
    RuntimeError
        If ``points``, ``queries`` and ``accelerator`` are not all on one device.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries; here
    sorting only occurs when ``return_sorted=True``, and sorts by distance, not by index. Ball
    boundaries use ``float32`` arithmetic; extremely tight radii near representable limits may
    disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]
        The flat CSR form, without cloning a segment per query.
    [`query_ball_count`][triwarp.neighbors.query_ball_count]
        The counts alone, when the neighbors themselves are not wanted.
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    require_same_device(points=points, queries=queries, accelerator=accelerator)
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = _launch.array([queries], dtype=wp.vec3, device=device)

    neighbor_indices_flat, neighbor_distances_flat, offsets = query_ball_with_offsets(
        points,
        queries,
        r,
        accelerator=accelerator,
        backend=backend,
        grid_bins=grid_bins,
        leaf_size=leaf_size,
        return_sorted=return_sorted,
    )
    neighbor_indices = tw.array.split(neighbor_indices_flat, offsets, copy=copy)
    neighbor_distances = tw.array.split(neighbor_distances_flat, offsets, copy=copy)

    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return neighbor_indices, neighbor_distances


def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    grid_bins: int | None = None,
    leaf_size: int = 4,
) -> wp.array[wp.int32]:
    """
    Count neighbors of each query within Euclidean distance ``r``.

    For each query center ``q``, returns how many entries ``p`` in ``points`` satisfy
    ``‖p - q‖₂ ≤ r``. This matches [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2``,
    ``eps=0``, and ``return_length=True`` (exact search; only the spatial index differs).

    The narrow phase keeps points with ``float32`` Euclidean distance at most ``r`` whichever
    broad phase ran, so the count does not depend on ``backend``.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        ``(m, 3)`` query centers stored as ``wp.vec3``.
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    accelerator, backend, grid_bins, leaf_size
        As in [`query_ball`][triwarp.neighbors.query_ball].

    Returns
    -------
    wp.array[wp.int32]
        Length-``m`` device array whose ``k``-th element is the neighbor count for
        ``queries[k]``. If ``n == 0``, returns zeros.

    Raises
    ------
    ValueError
        If ``backend`` is neither name, or contradicts the type of ``accelerator``.
    RuntimeError
        If ``points``, ``queries`` and ``accelerator`` are not all on one device.

    See Also
    --------
    [`query_ball`][triwarp.neighbors.query_ball]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    require_same_device(points=points, queries=queries, accelerator=accelerator)
    kind, accelerator = _resolve_accelerator(accelerator, backend)
    device = points.device
    n = points.size
    m = queries.size
    if n == 0:
        return _launch.zeros(m, dtype=wp.int32, device=device)

    neighbor_counts = _launch.empty(m, dtype=wp.int32, device=device)
    if kind == "hashgrid":
        if accelerator is None:
            accelerator = hashgrid_from_points(points, r, grid_bins)
        accel_selector = kernel_neighbors.ACCEL_HASHGRID
    else:
        if accelerator is None:
            accelerator = bvh_from_points(points, leaf_size)
        accel_selector = kernel_neighbors.ACCEL_BVH

    _launch.launch(
        kernel_neighbors.query_ball_count,
        dim=m,
        inputs=[points, queries, accel_selector, accelerator.id, wp.float32(r), neighbor_counts],
        device=device,
    )
    return neighbor_counts


def query_ball_with_offsets(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    grid_bins: int | None = None,
    leaf_size: int = 4,
    return_sorted: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Low-level ball query: neighbors in one concatenated pair plus per-query offsets.

    Same geometry as [`query_ball`][triwarp.neighbors.query_ball] (broad phase out to ``r``,
    ``float32`` test ``‖points[i] - q‖₂ ≤ r``). Semantics match
    [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``.

    Prefer [`query_ball`][triwarp.neighbors.query_ball] for a Python list of one array per query;
    use this when you want a single flat buffer on device (e.g. fused downstream kernels) and
    CSR-style boundaries without cloning each segment.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``
        (treated as one query).
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    accelerator, backend, grid_bins, leaf_size
        As in [`query_ball`][triwarp.neighbors.query_ball].
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.

    Returns
    -------
    neighbor_indices_flat, neighbor_distances_flat, offsets
        Three rank-1 arrays. Let ``m = queries.shape[0]`` after any ``wp.vec3`` wrap.

        ``offsets`` has length ``m + 1`` and is the total-terminated prefix sum of per-query
        neighbor counts: query ``k`` owns ``neighbor_indices_flat[offsets[k] : offsets[k + 1]]``,
        and ``offsets[m]`` is ``neighbor_indices_flat.shape[0]`` (the total neighbor count).

        ``neighbor_indices_flat`` and ``neighbor_distances_flat`` have that total length
        and list point indices and distances ``‖points[i] - q‖₂`` in parallel. Empty
        ``points`` still returns zero ``offsets``; empty neighbor sets yield
        length-0 flat arrays and zero ``offsets``.

    Raises
    ------
    ValueError
        If ``backend`` is neither name, or contradicts the type of ``accelerator``.
    RuntimeError
        If ``points``, ``queries`` and ``accelerator`` are not all on one device.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries; here
    sorting only occurs when ``return_sorted=True``, and sorts by distance, not by index. Ball
    boundaries use ``float32`` arithmetic; extremely tight radii near representable limits may
    disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    [`query_ball`][triwarp.neighbors.query_ball]
    [`query_ball_count`][triwarp.neighbors.query_ball_count]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    require_same_device(points=points, queries=queries, accelerator=accelerator)
    kind, resolved = _resolve_accelerator(accelerator, backend)
    if kind == "hashgrid":
        return _ball_with_offsets(
            points,
            queries,
            r,
            build_accelerator=lambda: (
                resolved if resolved is not None else hashgrid_from_points(points, r, grid_bins)
            ),
            accel=kernel_neighbors.ACCEL_HASHGRID,
            return_sorted=return_sorted,
        )
    return _ball_with_offsets(
        points,
        queries,
        r,
        build_accelerator=lambda: (
            resolved if resolved is not None else bvh_from_points(points, leaf_size)
        ),
        accel=kernel_neighbors.ACCEL_BVH,
        return_sorted=return_sorted,
    )


def _ball_with_offsets(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    build_accelerator: Callable[[], wp.HashGrid | wp.Bvh],
    accel: wp.int32,
    return_sorted: bool,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Count, scan, gather and optionally sort one ball query -- the body both accelerators share.

    The two backends of [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]
    differ only in which structure they build and which traversal the shared counting and
    neighbour kernels take (``accel``, one of ``kernels.neighbors.ACCEL_*``); everything else --
    the empty guards, the CSR scan, the ``2x`` sort scratch and the trailing compaction -- is
    identical. The accelerator is built lazily so an empty query never pays for one. The counts
    are [`query_ball_count`][triwarp.neighbors.query_ball_count]'s launch, written behind the
    leading zero of the offsets buffer so they are scanned in place.
    """
    device = points.device

    if isinstance(queries, wp.vec3):
        queries = _launch.array([queries], dtype=wp.vec3, device=device)
    m = queries.size

    if points.size == 0 or m == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
            _launch.zeros(m + 1, dtype=wp.int32, device=device),
        )

    accelerator = build_accelerator()
    # The total-terminated offsets are also the segment-bounds array ``segmented_sort_pairs``
    # wants below.
    offsets = _launch.zeros(m + 1, dtype=wp.int32, device=device)
    neighbor_counts = twt.as_dense(offsets[1:])
    _launch.launch(
        kernel_neighbors.query_ball_count,
        dim=m,
        inputs=[points, queries, accel, accelerator.id, wp.float32(r), neighbor_counts],
        device=device,
    )
    _launch.array_scan(neighbor_counts, out_array=neighbor_counts, inclusive=True)
    total_neighbors = int(read_scalar(offsets))
    if total_neighbors == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
            offsets,
        )

    flat_len = total_neighbors * (2 if return_sorted else 1)
    neighbor_indices_flat = _launch.empty(flat_len, dtype=wp.int32, device=device)
    neighbor_distances_flat = _launch.empty(flat_len, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_neighbors.query_ball_neighbors,
        dim=m,
        inputs=[
            points,
            queries,
            accel,
            accelerator.id,
            wp.float32(r),
            offsets,
            neighbor_indices_flat,
            neighbor_distances_flat,
        ],
        device=device,
    )

    if not return_sorted:
        # Sized exactly, so the buffers are the answer as they stand.
        return neighbor_indices_flat, neighbor_distances_flat, offsets
    wp.utils.segmented_sort_pairs(
        neighbor_distances_flat, neighbor_indices_flat, total_neighbors, offsets
    )
    # The sort needs a second half of scratch; copying the sorted half out lets it go.
    return (
        _launch.clone(neighbor_indices_flat[:total_neighbors]),
        _launch.clone(neighbor_distances_flat[:total_neighbors]),
        offsets,
    )


def knn_initial_radius(
    points: wp.array[wp.vec3], k: int, *, bounds: tuple[wp.vec3, wp.vec3] | None = None
) -> float:
    """
    Radius at which a ``k``-nearest search is expected to succeed on the first try.

    Inverts a uniform-density model of ``points``: the smallest ball expected to hold ``k`` of
    ``n`` points is the one whose volume is ``k / n`` of the bounding box's. This is the default
    ``initial_radius`` of [`query_nearest`][triwarp.neighbors.query_nearest], which deepens from it
    until each row certifies itself, so the value affects **speed only** and never the result.

    A cloud that is flat or collinear has a (near-)zero box volume, which the 3-D formula would
    invert into a useless radius, so the effective dimension ``d`` is taken to be the number of
    axes whose extent is non-negligible against the largest, and the matching formula is used:
    ``d = 3`` gives ``(3 f V / 4pi)^(1/3)``, ``d = 2`` gives ``(f A / pi)^(1/2)`` and ``d = 1``
    gives ``f L / 2``, with fill fraction ``f = k / n``.

    Parameters
    ----------
    points
        ``(n,)`` data points as ``wp.vec3``.
    k
        Number of neighbors the search will ask for; must be ``>= 1``.
    bounds
        Optional ``(min_bound, max_bound)`` from
        [`aabb`][triwarp.bounds.aabb]. Pass it to reuse a reduction you already ran;
        otherwise it is computed here (one device reduction plus one readback). Trusted, not
        validated — but only the estimate depends on it, so a wrong box costs speed and not
        correctness, unlike the identically named argument of
        [`query_nearest`][triwarp.neighbors.query_nearest].

    Returns
    -------
    float
        The estimated radius, or ``math.inf`` when no finite estimate is meaningful — ``n == 0``,
        ``k >= n`` (every point is a neighbor) or a degenerate box (all points coincident). The
        query kernels read ``math.inf`` as "one complete scan", which is the right answer in each
        of those cases.

    See Also
    --------
    [`query_nearest`][triwarp.neighbors.query_nearest]
    [`aabb`][triwarp.bounds.aabb]
    """
    n = points.size
    if n == 0 or k >= n:
        return math.inf

    if bounds is None:
        bounds = tw.bounds.aabb(points)
    min_bound, max_bound = bounds
    extents = sorted((float(max_bound[axis] - min_bound[axis]) for axis in range(3)), reverse=True)
    if extents[0] <= 0.0:
        return math.inf

    fill = k / n
    dimension = sum(extent > _FLAT_AXIS_FRACTION * extents[0] for extent in extents)
    if dimension == 3:
        volume = extents[0] * extents[1] * extents[2]
        return (3.0 * fill * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    if dimension == 2:
        return math.sqrt(fill * extents[0] * extents[1] / math.pi)
    return 0.5 * fill * extents[0]


@overload
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    k: int,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    max_radius: float = ...,
    grid_bins: int | None = ...,
    leaf_size: int = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array1dInt32, twt.Array1dFloat32]: ...
@overload
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: Literal[1] = 1,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    max_radius: float = ...,
    grid_bins: int | None = ...,
    leaf_size: int = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
    # Rank 1, not 2: a single neighbour per query collapses the trailing axis away, whichever form
    # `queries` took. `scipy.spatial.KDTree.query` does the same at `k=1`.
) -> tuple[twt.Array1dInt32, twt.Array1dFloat32]: ...
@overload
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    max_radius: float = ...,
    grid_bins: int | None = ...,
    leaf_size: int = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    max_radius: float = math.inf,
    grid_bins: int | None = None,
    leaf_size: int = 4,
    initial_radius: float | None = None,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    For each query center, find the ``k`` nearest data points.

    Iterative deepening: each query grows a search radius from ``initial_radius`` until its ``k``-th
    neighbour is closer than the radius, which certifies the row -- so the answer is exact and does
    not depend on ``backend``. The candidate row is register-resident for ``k <= 64``.

    !!! note "Which backend to use"
        ``"hashgrid"`` (the default) is the faster broad phase when the query scale matches the
        cloud's density, because ``initial_radius`` sets the cell width: the walk visits
        ``(2 ceil(r / cell) + 1) ** 3`` cells, so a radius **f** times too large costs ``f ** 3``.
        A row the walk cannot certify within a few cells is finished another way. At ``k == 1``
        that is a closest-point query over a tree of the cloud, whose cost does not grow with how
        far the query is from it. Above it, a query that sits on a point of the cloud -- a
        self-query, in a part of the cloud sparser than the cell was sized for -- goes to the
        BVH's radius search, so a cloud whose density is not uniform costs its sparse rows a tree
        search rather than a scan; any other row falls back to an exact linear scan of the cloud.

        ``"bvh"`` has no cell width to get wrong, so it is the one to reach for when the query scale
        is unknown and the queries do not sit on the cloud. Its own cost grows with ``k`` faster
        than the grid's. At ``k == 1`` with no ``accelerator`` it is not a radius search at all but
        a closest-point query over [`mesh_from_points`][triwarp.neighbors.mesh_from_points], whose
        cost does not grow with the queries' distance from the cloud.

        **For queries off the cloud at moderate ``k``, the grid's cost is non-monotonic in cloud
        size**: a row whose true ``k``-th distance runs past the widest walk takes the linear scan,
        and how much of the query set falls in that tail moves with ``n``. The tree's cost stays
        monotonic over the same range.

        Both are exact; this is a cost choice only. Reuse the structure across calls by passing it
        as ``accelerator`` when several queries share one cloud.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``.
    k
        Number of neighbours per query, ``>= 1``.
    accelerator
        A prebuilt ``warp.HashGrid`` or ``warp.Bvh`` over ``points``, to reuse across queries. It
        selects the backend by its own type, so ``backend`` is redundant when this is given and
        raises if it names the other one. A hash grid passed here keeps its own ``cell_width``
        rather than one derived from ``initial_radius``.
    backend
        ``"hashgrid"`` (the default) or ``"bvh"``, when ``accelerator`` is ``None``. See the note
        above.
    max_radius
        Stop deepening past this distance and leave the remaining slots unfilled (index ``-1``,
        distance ``inf``). Defaults to unbounded.
    grid_bins
        Grid resolution when building a hash grid; ``None`` sizes it from the point count, as in
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]. Ignored under
        ``backend="bvh"`` and whenever ``accelerator`` is given.
    leaf_size
        Maximum primitives per leaf when building a BVH. Ignored under ``backend="hashgrid"`` and
        whenever ``accelerator`` is given.
    initial_radius
        First search radius. Defaults to
        [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius], which estimates it from the
        *cloud's* density -- so pass it explicitly when the queries are at a different scale, since
        under ``"hashgrid"`` this also sets the cell width.
    bounds
        ``(min_bound, max_bound)`` of ``points``, to skip
        [`triwarp.bounds.aabb`][triwarp.bounds.aabb] and its readback. **Trusted, not validated, and
        not only a shortcut**: it also fixes the per-query radius at which a scan is provably
        complete, so a box that does not contain ``points`` can end the search early and leave
        slots unfilled (``-1`` and ``inf``) rather than merely running slower. Pass the real
        bounding box, or leave it ``None``. Neither this nor ``initial_radius`` is read by the
        ``k == 1`` BVH-backend query without an ``accelerator``, which has no radius to deepen.

    Returns
    -------
    neighbor_indices, neighbor_distances
        ``(m, k)`` ``wp.int32`` indices into ``points`` and ``(m, k)`` ``wp.float32`` distances,
        each row sorted by increasing distance. A slot no neighbour was found for holds ``-1`` and
        ``inf``.

        The trailing axis collapses when there is only one of it, as
        [`scipy.spatial.KDTree.query`][] does: at ``k == 1`` the two are rank-1 of length ``m``, and
        for a single ``wp.vec3`` query they are rank-1 of length ``k``. The rank depends only on
        ``k`` and the form of ``queries``, never on whether an answer was found — an empty
        ``points`` or an empty ``queries`` returns the same rank a populated one would.

    Raises
    ------
    ValueError
        If ``k < 1``, ``max_radius < 0``, ``initial_radius < 0``, or ``backend`` is neither name or
        contradicts the type of ``accelerator``.
    RuntimeError
        If ``points``, ``queries`` and ``accelerator`` are not all on one device.

    See Also
    --------
    [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
        The default first radius, and the one parameter worth passing by hand.
    [`query_ball`][triwarp.neighbors.query_ball]
        A fixed radius rather than a fixed count.
    [`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance]
        The ``k=2`` self-query, which is the cloud's spacing.
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query`][]
    """
    require_same_device(points=points, queries=queries, accelerator=accelerator)
    kind, resolved = _resolve_accelerator(accelerator, backend)
    _validate_nearest(k, max_radius, initial_radius)

    device = points.device
    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = _launch.array([queries], dtype=wp.vec3, device=device)

    m = queries.size
    n = points.size

    # Both degenerate inputs answer "every slot unfilled" over `(m, k)`, which is empty of its own
    # accord when there are no queries -- and both must go through the same shaping as the general
    # path, or `k == 1` comes back rank-2 here and rank-1 everywhere else.
    if m == 0 or n == 0:
        return _empty_nearest(m, k, single_query, device)

    # ``wp.empty``, not ``wp.full``: every row is written in full by the kernel, so pre-filling
    # here would be two wasted launches.
    neighbor_indices = twt.empty_2d((m, k), wp.int32, device=device)
    neighbor_distances = twt.empty_2d((m, k), wp.float32, device=device)
    if kind == "bvh" and k == 1 and resolved is None:
        # One nearest point is a closest-point query over the collapsed-triangle mesh: its descent
        # needs no radius to deepen, so neither the bounding box nor the density estimate (two
        # reductions and their readbacks) is computed, and a query far off the cloud costs what an
        # on-surface one does. The distances are the radius search's to the bit. A caller's
        # ``wp.Bvh`` cannot take this path, since the query needs a ``wp.Mesh``.
        mesh = mesh_from_points(points)
        _launch.launch(
            kernel_neighbors.query_nearest_via_mesh,
            dim=m,
            inputs=[mesh.id, points, queries, wp.float32(max_radius)],
            outputs=[neighbor_indices, neighbor_distances],
            device=device,
        )
        return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)

    if bounds is None:
        bounds = tw.bounds.aabb(points)
    min_bound, max_bound = bounds
    if initial_radius is None:
        initial_radius = knn_initial_radius(points, k, bounds=bounds)
    search = [points, queries]
    shared = [wp.int32(k), wp.float32(max_radius), wp.float32(initial_radius)]
    outputs = [neighbor_indices, neighbor_distances]
    if kind == "bvh":
        bvh = resolved if resolved is not None else bvh_from_points(points, leaf_size)
        # The BVH follows an unbounded radius, so it needs no linear-scan cutover argument.
        _launch.launch(
            kernel_neighbors.bvh_nearest_kernel(k),
            dim=m,
            inputs=[*search, bvh.id, *shared, min_bound, max_bound, wp.int32(0), *outputs],
            device=device,
        )
        return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)

    if resolved is None:
        bins = _resolve_grid_bins(grid_bins, n)
        cell_size = _knn_cell_size(initial_radius, min_bound, max_bound, bins)
        grid = hashgrid_from_points(points, cell_size, bins)
    else:
        grid = resolved
        cell_size = float(getattr(grid, "cell_width", initial_radius))
    # At ``k == 1`` a row the cell walk cannot certify is *deferred* rather than finished by a
    # linear scan over the whole cloud: that scan is what a query far from the cloud costs the
    # grid, and a closest-point descent answers the same row in time that does not grow with the
    # distance. So the walk stays the fast path for queries on or near the cloud, and only the rows
    # it gives up on pay for the tree. The walk gives up at the same radius it hands over to the
    # scan at: bringing that in shortens a displaced row's walk, but it also defers the odd outlier
    # of an on-surface query set, and one deferred row pays for the whole tree.
    #
    # Above ``k == 1`` a row is deferred only when the query sits on a point of the cloud -- a
    # self-query -- in a part sparser than the cell width was sized for: the sparse end of a cloud
    # whose density is not uniform. Those rows go to the BVH's radius search, which stays small
    # there; without it each one would scan every point. A query off the cloud keeps the scan,
    # which is the cheaper finish for it (``kernel_neighbors.DEFER_NEAR``). A cloud smaller than
    # ``_KNN_DEFER_MIN_POINTS`` and the global-row kernel past the largest register bucket never
    # defer above ``k == 1``.
    defer = k == 1 or (k <= kernel_neighbors.KNN_ROW_BUCKETS[-1] and n >= _KNN_DEFER_MIN_POINTS)
    # The deferral counter exists only when rows may be deferred; otherwise the kernel never
    # touches it and takes a null descriptor.
    deferred = _launch.zeros(1, dtype=wp.int32, device=device) if defer else None
    _launch.launch(
        kernel_neighbors.hashgrid_nearest_kernel(k),
        dim=m,
        inputs=[
            *search,
            grid.id,
            *shared,
            wp.float32(_knn_widest_grid_radius(cell_size, n)),
            min_bound,
            max_bound,
            wp.int32(1 if defer else 0),
            deferred,
            *outputs,
        ],
        device=device,
    )
    if deferred is not None:
        _finish_deferred_nearest(
            points,
            queries,
            k,
            max_radius,
            initial_radius,
            (min_bound, max_bound),
            leaf_size,
            deferred,
            neighbor_indices,
            neighbor_distances,
        )
    return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)


def _finish_deferred_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    max_radius: float,
    initial_radius: float,
    bounds: tuple[wp.vec3, wp.vec3],
    leaf_size: int,
    deferred: wp.array[wp.int32],
    neighbor_indices: twt.Array2dInt32,
    neighbor_distances: twt.Array2dFloat32,
) -> None:
    """
    Answer the rows the grid deferred, by a tree over the cloud.

    One four-byte readback decides whether there are any, since the tree is a build over the whole
    cloud: a query set the grid certified in full pays only that read. At ``k = 1`` the tree is a
    ``wp.Mesh`` whose triangles each collapse onto one point, so its closest-face query is a
    nearest-point query with the mesh BVH's pruned descent; above it, the BVH's radius search
    finishes each deferred row and returns at once on the others.
    """
    # The one sync deferral adds, and it decides whether a whole-cloud build is needed.
    if int(read_scalar(deferred, 0)) == 0:
        return
    device = points.device
    m = queries.size
    if k == 1:
        mesh = mesh_from_points(points)
        _launch.launch(
            kernel_neighbors.nearest_point_via_mesh,
            dim=m,
            inputs=[mesh.id, points, queries, wp.float32(max_radius)],
            outputs=[neighbor_indices, neighbor_distances],
            device=device,
        )
        return
    bvh = bvh_from_points(points, leaf_size)
    _launch.launch(
        kernel_neighbors.bvh_nearest_kernel(k),
        dim=m,
        inputs=[
            points,
            queries,
            bvh.id,
            wp.int32(k),
            wp.float32(max_radius),
            wp.float32(initial_radius),
            bounds[0],
            bounds[1],
            wp.int32(1),
            neighbor_indices,
            neighbor_distances,
        ],
        device=device,
    )


def _resolve_grid_bins(grid_bins: int | None, n: int) -> int:
    """
    Resolve the hash-grid resolution: ``grid_bins`` itself, or the point-count default at ``None``.

    The default is the smallest multiple of ``_GRID_BINS_STEP`` whose cube holds
    ``_GRID_BINS_PER_POINT`` bins a point, clamped to ``[_GRID_BINS_MIN, _GRID_BINS_MAX]``.
    """
    if grid_bins is not None:
        return int(grid_bins)
    side = math.ceil((_GRID_BINS_PER_POINT * n) ** (1.0 / 3.0) / _GRID_BINS_STEP) * _GRID_BINS_STEP
    return min(max(side, _GRID_BINS_MIN), _GRID_BINS_MAX)


def _knn_cell_size(
    initial_radius: float, min_bound: wp.vec3, max_bound: wp.vec3, grid_bins: int
) -> float:
    """
    Hash-grid cell width for a k-NN search starting at ``initial_radius``.

    **This width, not the deepening ladder's starting radius, is what actually governs the cost of a
    hash-grid k-NN search.** Too narrow a cell and a query abandons the grid for an exact linear
    scan of the whole cloud; too wide and every cell probe scans far more points than it needs to.
    The true optimum depends on how far the query points sit from the data cloud, which the cloud's
    own density says nothing about -- so this uses a simple, safe default, bounded between one grid
    period and the cloud's full extent. The failure mode of a bad width is a slower query, never a
    wrong one: every row still certifies itself.

    **An automatic query-aware width was built, measured and declined -- on its cost, not its
    accuracy. Do not re-propose it without making the probe cheaper.** The missing term really is
    the query displacement, and a cheap probe recovers it: brute-force the nearest distance for a
    small query subsample against a small subsample of the cloud, subtract *that subsample's* own
    expected spacing (``knn_initial_radius(subsample, 1)`` -- without this correction the estimate
    inherits the subsample's sparsity and over-widens severalfold at the near end), and add
    ``knn_initial_radius(points, k)``. It is sound: near-neutral where nothing was wrong, several
    times faster through the middle band of displacements, and better than ``backend="bvh"`` there
    too. Two things sink it as a *default*. The probe is itself a few slices, copies, a launch and a
    reduction, which on its own doubles the on-surface call that is the benchmarked operating point;
    and at a large displacement a wide cell is a loss, because by then abandoning the grid for the
    linear scan is genuinely the right algorithm. A gate for either would have to decide *without*
    the probe, which is the thing that cannot be done cheaply.

    The lever that remains is the public one: pass ``initial_radius=`` when the caller knows the
    scale, which is what ``metrics.chamfer_points_to_points`` already does by seeding its backward
    search from the forward half's answer.

    A block-cooperative walk (one warp per query) is separately refuted here. The grid walk itself
    cannot be split across lanes at all -- ``wp.HashGrid`` exposes no per-cell entry point -- and
    the linear-scan fallback, which can, is not load-imbalanced in the regime that costs: once the
    displacement is large enough to trigger it, nearly every row takes it and the launch is
    uniformly expensive rather than held up by stragglers.
    """
    extent = max(float(max_bound[axis] - min_bound[axis]) for axis in range(3))
    if extent <= 0.0:
        # Every point is at the same position, so any positive width buckets them together.
        return 1.0
    # Lower bound ``extent / grid_bins`` keeps the cloud inside one period of the spatial hash;
    # upper bound ``extent`` keeps a huge ``initial_radius`` (``k >= n`` gives ``inf``) finite.
    return min(max(initial_radius, extent / grid_bins), extent)


def _knn_widest_grid_radius(cell_size: float, n: int) -> float:
    """
    Radius past which an exact linear scan beats widening the hash-grid walk.

    ``wp.hash_grid_query`` visits ``(2 ceil(r / cell) + 1) ** 3`` cells, and each visit is a hash
    plus two dependent, uncoalesced global loads. The linear scan it falls back to is the opposite:
    every thread in a warp reads the *same* ``points[j]``, so it streams out of L2 as a broadcast.
    One cell probe costs on the order of ``_CELL_PROBE_POINTS`` point tests, which makes the
    break-even span grow as ``n ** (1/3)`` -- a small cloud needs only one cell of slack, a much
    larger one several. A fixed span gets one of those two badly wrong.
    """
    span = 0.5 * ((n / _CELL_PROBE_POINTS) ** (1.0 / 3.0) - 1.0)
    # Never below one cell: the 3x3x3 walk is what the grid was built for and is cheap at any n.
    return max(span, 1.0) * cell_size


def _validate_nearest(k: int, max_radius: float, initial_radius: float | None) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")
    if initial_radius is not None and initial_radius < 0:
        raise ValueError("initial_radius must be >= 0")


def _empty_nearest(
    m: int, k: int, single_query: bool, device: wp.DeviceLike
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    Build the result rows for a query that can find nothing: every slot unfilled.

    Serves both degenerate inputs -- no data points, and no queries at all, where ``m`` is zero and
    the two buffers come out empty. Shapes through
    [`_shape_nearest`][triwarp.neighbors._shape_nearest] rather than returning rank-2 directly, so
    that the rank a caller sees does not depend on whether the answer happened to be empty.
    """
    neighbor_indices = _launch.full((m, k), -1, dtype=wp.int32, device=device)
    neighbor_distances = _launch.full((m, k), math.inf, dtype=wp.float32, device=device)
    return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)


def _shape_nearest(
    neighbor_indices: wp.array[wp.int32],
    neighbor_distances: wp.array[wp.float32],
    k: int,
    single_query: bool,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """Collapse the ``(m, k)`` result to the rank the caller's ``queries`` / ``k`` imply."""
    if k == 1:
        return (
            cast(twt.Array1dInt32, neighbor_indices.reshape(-1)),
            cast(twt.Array1dFloat32, neighbor_distances.reshape(-1)),
        )
    if single_query:
        return (
            cast(twt.Array1dInt32, neighbor_indices[0]),
            cast(twt.Array1dFloat32, neighbor_distances[0]),
        )
    return twt.as_array2d(neighbor_indices, wp.int32), twt.as_array2d(
        neighbor_distances, wp.float32
    )


def _resolve_accelerator(
    accelerator: wp.HashGrid | wp.Bvh | None, backend: QueryBackend | None
) -> tuple[QueryBackend, wp.HashGrid | wp.Bvh | None]:
    """
    Settle ``backend`` against ``accelerator``, the one new failure mode the merged API has.

    ``backend`` is ``None`` rather than ``"hashgrid"`` in the signatures so that "not passed" is
    distinguishable from "passed as hashgrid": a caller who hands over a ``wp.Bvh`` and nothing else
    must not be told it contradicts a default they never wrote. The *effective* default is still
    ``"hashgrid"``, and it applies only when no accelerator is given.

    A prebuilt accelerator already knows what it is, so it wins the dispatch -- but a ``backend``
    that names the other one is a mistake in the call rather than a preference to be silently
    dropped, and it raises.
    """
    if accelerator is None:
        if backend is None:
            return "hashgrid", None
        if backend not in ("hashgrid", "bvh"):
            raise ValueError(f'backend must be "hashgrid" or "bvh", got {backend!r}')
        return backend, None

    inferred: QueryBackend = "hashgrid" if isinstance(accelerator, wp.HashGrid) else "bvh"
    if backend is not None and backend != inferred:
        raise ValueError(
            f"backend={backend!r} contradicts the accelerator passed, which is a "
            f"{type(accelerator).__name__} ({inferred!r}). Pass one or the other."
        )
    return inferred, accelerator


def query_weighted_nearest(
    points: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    queries: wp.array[wp.vec3],
    *,
    max_weight: float | None = None,
    accelerator: wp.Bvh | None = None,
    leaf_size: int = 4,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]:
    """
    Nearest site under the weighted distance ``|p - q| - w(p)``.

    Each site carries a radius, and the winner is the one whose *surface* is closest rather than
    whose centre is -- the additively weighted (Apollonius) nearest-neighbour query, which is what
    picks the influencing site when the sites have different scales: a sphere set, a level-of-detail
    cluster, a set of samples with per-sample confidence. Plain
    [`query_nearest`][triwarp.neighbors.query_nearest] is the ``w = 0`` case, and the two
    genuinely differ -- on a random 40-site cloud, 1 query in 6 had a different winner.

    Parameters
    ----------
    points
        ``(n,)`` site positions as ``wp.vec3``.
    weights
        ``(n,)`` per-site weights, subtracted from the distance. Larger wins ties of distance; may
        be negative, which pushes a site away.
    queries
        ``(m,)`` query positions as ``wp.vec3``.
    max_weight
        An **upper bound** on ``weights``, which is what makes the search prunable. ``None`` reduces
        ``weights`` on the device and reads the maximum back (one readback), so pass it when the
        bound is already known -- a radius cap, or a previous call's reduction.
    accelerator
        A prebuilt ``wp.Bvh`` over ``points``, to reuse across queries. Spelled the way the four
        [`query_ball`][triwarp.neighbors.query_ball] /
        [`query_nearest`][triwarp.neighbors.query_nearest] entry points spell it, but with no
        ``backend`` beside it: this query has only the BVH broad phase, so there is nothing to
        select between.
    leaf_size
        Maximum primitives per BVH leaf when one is built here.

    Returns
    -------
    index, weighted_distance
        ``(m,)`` winning site per query and its ``|p - q| - w(p)``, which is **negative** wherever a
        query lies inside a site's radius. A query with no site at all (an empty cloud) reports
        ``-1`` and ``inf``.

    Raises
    ------
    ValueError
        If ``weights`` does not have one entry per point.
    RuntimeError
        If ``points``, ``weights``, ``queries`` and ``accelerator`` are not all on one device.

    Notes
    -----
    ``max_weight`` is trusted, not checked: a bound *smaller* than some weight can silently prune
    the true winner, and verifying it would cost the very reduction the parameter exists to avoid.
    The default is therefore the safe one.

    Exactness is the same argument the k-NN queries use, with the weight folded in: a site outside
    the cube of half-extent ``r`` is farther than ``r``, so it scores worse than ``r - max_weight``,
    and a best score at or below that bound cannot be beaten. Unlike a plain nearest query this
    means the search radius must exceed the answer's distance *by the weight range*, so a wide
    weight distribution costs more scans than a narrow one.

    See Also
    --------
    [`query_nearest`][triwarp.neighbors.query_nearest]
        The unweighted query, and the ``k > 1`` form. This one answers ``k = 1`` only, because no
        caller has needed more.
    """
    require_same_device(points=points, weights=weights, queries=queries, accelerator=accelerator)
    device = points.device
    n = points.size
    m = queries.size
    if weights.size != n:
        raise ValueError("weights must have one entry per point")

    if m == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
        )
    if n == 0:
        return (
            _launch.full(m, -1, dtype=wp.int32, device=device),
            _launch.full(m, float("inf"), dtype=wp.float32, device=device),
        )

    if accelerator is None:
        accelerator = bvh_from_points(points, leaf_size=leaf_size)
    if max_weight is None:
        max_weight = float(tw.reduce.max(weights))
    min_bound, max_bound = tw.bounds.aabb(points)
    # First radius: the mean spacing's own estimate plus the weight bound, since a query cannot be
    # certified below it however close its winner is.
    initial_radius = knn_initial_radius(points, 1, bounds=(min_bound, max_bound)) + max(
        max_weight, 0.0
    )

    out_indices = _launch.empty(m, dtype=wp.int32, device=device)
    out_distances = _launch.empty(m, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_neighbors.query_weighted_nearest_neighbors,
        dim=m,
        inputs=[
            points,
            weights,
            queries,
            accelerator.id,
            wp.float32(max_weight),
            wp.float32(initial_radius),
            min_bound,
            max_bound,
            out_indices,
            out_distances,
        ],
        device=device,
    )
    return out_indices, out_distances


def nearest_neighbor_distance(points: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Distance from each point to the closest *other* point of the same cloud.

    The standard scale estimate for an unstructured cloud, and what a sampling-driven parameter is
    normally derived from: the mean of this array is the cloud's mean spacing, which is how
    [`ball_pivoting`][triwarp.reconstruction.ball_pivoting] guesses its ball radius and how
    [`screened_poisson`][triwarp.reconstruction.screened_poisson] caps its octree depth. A
    self-query for two neighbours, keeping the second — the first is the point itself, at distance
    zero.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` distances on ``points.device``. Zero where two points coincide exactly.

    Notes
    -----
    A cloud of fewer than two points has no answer, and this reports ``inf`` for every entry —
    the value [`query_nearest`][triwarp.neighbors.query_nearest] already uses for a slot it
    could not fill. Open3D's ``compute_nearest_neighbor_distance`` reports ``0.0`` in that one
    case; on any cloud of two or more points the two agree, because every point then has a nearest
    neighbour. The distances are ``float32`` (``wp.length``), so they can differ from a
    ``float64`` reference in the last digits.

    See Also
    --------
    [`query_nearest`][triwarp.neighbors.query_nearest]
        The underlying query; call it directly to reuse a BVH across several queries of one cloud.
    [`triwarp.points.farthest_point_sample`][triwarp.points.farthest_point_sample]
    """
    device = points.device
    n = points.size
    if n < 2:
        return _launch.full(n, math.inf, dtype=wp.float32, device=device)

    _indices, distances = query_nearest(points, points, k=2)
    # Column 1 of the ``(n, 2)`` table, which is a *strided* view -- so it is copied into a dense
    # buffer rather than returned, both because callers expect a plain ``wp.array`` and because a
    # strided array is the shape that silently corrupts a downstream Python-scope gather.
    nearest = _launch.empty(n, dtype=wp.float32, device=device)
    _launch.copy(nearest, distances[:, 1])
    return nearest


def closest_pair(points: wp.array[wp.vec3]) -> tuple[int, int, float]:
    """
    Find the two closest points of a cloud, and the distance between them.

    The global minimum of
    [`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance], with the partner
    recovered -- so it is the same ``k=2`` self-query, reduced instead of returned.
    Answers "does this cloud contain a near-duplicate, and where" in one call, which is the question
    a tolerance for
    [`triwarp.points.point_duplicate_mask`][triwarp.points.point_duplicate_mask] is normally chosen
    from.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``, ``n >= 2``.

    Returns
    -------
    index_a, index_b, distance
        The two point indices — ``index_a < n``, ``index_b`` its nearest neighbour — and their
        Euclidean distance, all as Python scalars.

    Raises
    ------
    ValueError
        If ``points`` holds fewer than two points, which have no pair.

    Notes
    -----
    Ties are broken toward the **lowest** ``index_a``: the reduction is a ``min`` over one ``int64``
    per point holding the distance in the high half and the index in the low, so a shorter distance
    wins and an equal distance defers to the smaller index. Exact duplicates therefore report the
    first duplicated point at distance ``0.0``.

    One host readback, of two elements: the reduction's result and ``index_b`` from the query
    table. Everything up to it stays on the device, so this does not move the ``(n, 2)`` table
    across the bus.

    See Also
    --------
    [`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance]
        The per-point form, when every distance is wanted rather than the smallest.
    [`triwarp.points.point_duplicate_mask`][triwarp.points.point_duplicate_mask]
        Exact coincidence rather than proximity, and a mask rather than one pair.
    """
    n = points.size
    if n < 2:
        raise ValueError("closest_pair needs at least two points")

    device = points.device
    indices, distances = query_nearest(points, points, k=2)
    # Column 1 of the table, as strided views -- column 0 is each point itself. The two kernels only
    # index them, so neither needs a dense copy.
    result = _launch.full(2, INT64_MAX, dtype=wp.int64, device=device)
    _launch.launch_tiled(
        kernel_neighbors.nearest_key_argmin,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[distances[:, 1], result],
        block_dim=TILE_1D,
        device=device,
    )
    _launch.launch(
        kernel_neighbors.nearest_key_partner,
        dim=1,
        inputs=[result, indices[:, 1], result],
        device=device,
    )
    key, index_b = (int(value) for value in result.numpy())
    # The key's high half is the distance's own float32 bits, so it decodes on the host for free.
    distance = float(np.array([key >> 32], dtype=np.uint32).view(np.float32)[0])
    return key & 0xFFFFFFFF, index_b, distance


def geodesic_ball(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], radius: float, min_count: int = 6
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Per-vertex geodesic-ball neighborhoods, and the reference neighbor that frames each one.

    For each vertex this is a breadth-first traversal of the mesh edge graph, enqueueing a neighbor
    only when it lies within Euclidean ``radius`` of the center — i.e. the connected component of
    the center within the radius ball. This is a *geodesic* ball rather than a pure Euclidean one,
    so it excludes vertices that are spatially close but lie across a fold of the surface (e.g. the
    opposite wall of a torus tube), which a Euclidean hash-grid query would wrongly include and
    which corrupts the quadric fit. When fewer than ``min_count`` vertices are reachable, the
    nearest out-of-ball vertices are appended (libigl's ``extra_candidates`` path).

    Also returns the per-vertex reference neighbor used to build the tangent frame: the
    lowest-indexed edge neighbor, matching libigl's ``adjacency_list[i][0]``. libigl's symmetrized
    shape operator is frame-dependent, so reproducing its principal values (
    [`principal_curvature`][triwarp.curvature.principal_curvature] with ``frame_independent=False``)
    requires this exact frame; the default frame-independent computation does not depend on it.
    Isolated vertices reference themselves.

    The traversal runs entirely on device. Vertex adjacency is built as a CSR graph via
    [`edges_unique`][triwarp.edges.edges_unique] +
    [`edges_to_neighbor_lists`][triwarp.graph.edges_to_neighbor_lists] with sorted rows, then a
    single-pass BFS collects each ball into its per-source queue row (the queue prefix *is* the
    result) and a scan + gather compacts the rows into the CSR neighbor buffer. Each source uses
    fixed-capacity scratch of ``_PER_SOURCE_MAX_NEIGHBORS`` neighbors
    (``triwarp.kernels.algorithms.bfs``, currently 512);
    if a vertex collects more than that the surplus is dropped and a warning is emitted.

    !!! note

        Distances and tie-breaking are computed in ``float32`` (set-equivalent to the libigl
        reference; borderline ties between equidistant neighbors may resolve differently but leave
        the order-independent quadric fit unchanged).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions as ``wp.vec3``.
    faces
        Length-``3 * n_faces`` flat triangle index buffer as ``wp.int32``.
    radius
        Geodesic-ball radius in world units.
    min_count
        Minimum neighbors per vertex; the nearest out-of-ball vertices backfill any shortfall.

    Returns
    -------
    tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
        ``(neighbor_indices, offsets, reference_neighbors)``. ``offsets`` is the
        length-``n_vertices + 1`` exclusive prefix sum of per-vertex neighbor counts (CSR row
        bounds); vertex ``i`` owns ``neighbor_indices[offsets[i] : offsets[i + 1]]`` and
        ``offsets[n_vertices]`` is the total. ``reference_neighbors`` has length ``n_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n = vertices.size
    if n == 0:
        # A single zero rather than an empty buffer: the CSR row-bounds form is ``n + 1`` long.
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.zeros(1, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
        )

    # ``sort_rows=True`` is load-bearing, not tidiness. The ball is a geometric predicate and is
    # order-independent, but the traversal below emits its queue in *visit* order, so a permuted
    # adjacency row permutes the returned row -- and ``curvature.principal_curvature``, which
    # consumes this, accumulates its quadric's normal equations along that row, so a permutation
    # moves its answer at the near-flat vertices where the fit is ill conditioned. Sorted rows make
    # this builder's output identical to ``graph.edges_to_csr``'s, for a fraction of its cost; a
    # mesh vertex's valence is far below the degree where the per-row sort stops paying.
    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n, validate=False)
    adj_columns, adj_offsets = tw.graph.edges_to_neighbor_lists(
        n, unique_edges, validate=False, sort_rows=True
    )

    reference_neighbors = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_neighbors.geodesic_ball_reference_neighbors,
        dim=n,
        inputs=[adj_offsets, adj_columns, reference_neighbors],
        device=device,
    )

    # Per-source scratch lives in shared global-memory pools sized for one chunk of sources
    # (queue rows, an open-addressing visited row pre-filled with -1 per launch, and a small
    # nearest-fallback pool) instead of kilobytes of per-thread local arrays.
    chunk = min(n, 1 << 15)
    queue_pool = _launch.empty(
        (chunk, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS), dtype=wp.int32, device=device
    )
    visited_pool = _launch.full(
        (chunk, kernel_bfs._VISITED_HASH_CAPACITY), -1, dtype=wp.int32, device=device
    )
    ext_dist_pool = twt.empty_2d((chunk, kernel_bfs._EXTRAS_CAPACITY), wp.float32, device=device)
    ext_idx_pool = twt.empty_2d((chunk, kernel_bfs._EXTRAS_CAPACITY), wp.int32, device=device)

    counts = _launch.empty(n, dtype=wp.int32, device=device)
    # CSR row bounds in the length-``n + 1`` form. The leading zero from ``wp.zeros`` is the first
    # exclusive offset and the inclusive scan fills the rest, so ``offsets[n]`` holds the total --
    # ``array.counts_to_offsets``' convention, open-coded here because that helper reads the total
    # back to the host and no caller here wants it. The terminator belongs at this producer rather
    # than in each consumer's kernel: both of them read ``offsets[i + 1]`` as the row end.
    single_chunk = n <= chunk
    # With one chunk the global scan *is* the chunk's: its prefix is the exclusive offsets and its
    # terminator the total, so no chunk-local scan or total buffer is needed. The overflow counter
    # sits right after the total -- in the offsets' own buffer on one chunk, beside the chunk total
    # on several -- so the last chunk reads both in one copy.
    if single_chunk:
        totals = _launch.zeros(n + 2, dtype=wp.int32, device=device)
        total_at = n
        offsets = twt.as_dense(totals[: n + 1])
        local_offsets = twt.as_dense(offsets[:n])
    else:
        totals = _launch.zeros(2, dtype=wp.int32, device=device)
        total_at = 0
        offsets = _launch.zeros(n + 1, dtype=wp.int32, device=device)
        local_offsets = _launch.empty(chunk, dtype=wp.int32, device=device)
    overflow = twt.as_dense(totals[total_at + 1 : total_at + 2])
    chunk_flats: list[wp.array[wp.int32]] = []
    n_overflow = 0
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        if start > 0:
            _launch.fill_(visited_pool, -1)
        _launch.launch(
            kernel_neighbors.query_geodesic_ball_collect,
            dim=m,
            inputs=[
                vertices,
                adj_offsets,
                adj_columns,
                wp.float32(radius),
                wp.int32(min_count),
                wp.int32(start),
                queue_pool,
                visited_pool,
                ext_dist_pool,
                ext_idx_pool,
                counts,
                overflow,
            ],
            device=device,
        )
        # Gather this chunk's queue rows before the next chunk reuses the pools: chunk-local
        # exclusive scan of counts, one 4-byte readback for the chunk total, then a coalesced
        # 2D copy into the chunk's flat buffer.
        if single_chunk:
            _launch.array_scan(counts, out_array=offsets[1:], inclusive=True)
        else:
            _launch.array_scan(
                counts[start : start + m], out_array=local_offsets[:m], inclusive=False
            )
            _launch.map(
                wp.add, local_offsets[m - 1 : m], counts[start + m - 1 : start + m], out=totals[:1]
            )
        if start + m == n:
            chunk_total, n_overflow = read_values(totals, total_at, 2)
        else:
            chunk_total = int(read_scalar(totals, 0))
        flat_chunk = _launch.empty(chunk_total, dtype=wp.int32, device=device)
        if chunk_total > 0:
            _launch.launch(
                kernel_neighbors.gather_queue_rows,
                dim=(m, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS),
                inputs=[queue_pool, counts, local_offsets, wp.int32(start), flat_chunk],
                device=device,
            )
        chunk_flats.append(flat_chunk)

    if n_overflow > 0:
        warnings.warn(
            f"geodesic_ball: {n_overflow} neighborhood capacity breaches "
            f"(fixed cap {kernel_bfs._PER_SOURCE_MAX_NEIGHBORS}); surplus neighbors dropped.",
            stacklevel=2,
        )

    if single_chunk:
        # Single chunk (n <= chunk): the chunk buffer already is the global CSR neighbor buffer.
        return chunk_flats[0], offsets, reference_neighbors

    _launch.array_scan(counts, out_array=offsets[1:], inclusive=True)

    # Chunk order equals ascending source order, so concatenation lines up with the global scan.
    # Same per-segment ``wp.copy`` loop either way -- that is the packing floor -- one call for it.
    return tw.array.concatenate(chunk_flats), offsets, reference_neighbors
