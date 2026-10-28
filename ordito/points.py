"""
Unstructured point-cloud geometry: fitting, second moments and normal estimation.

No connectivity here -- everything takes a bare ``(n,)`` array of positions. Four groups:

- **Whole-cloud fits.** [`centroid`][ordito.points.centroid],
  [`covariance`][ordito.points.covariance] and
  [`centered_covariance`][ordito.points.centered_covariance] give the second moments;
  [`fit_line`][ordito.points.fit_line] and [`fit_plane`][ordito.points.fit_plane] read the
  dominant and weakest eigenvector off them. [`gram_matrix`][ordito.points.gram_matrix] is the
  uncentered form, for callers that want to center differently.
- **Per-point.** [`estimate_normals`][ordito.points.estimate_normals] fits a plane to each point's
  k-nearest neighbourhood, which is how an unoriented cloud acquires normals before reconstruction.
  [`outlier_probability`][ordito.points.outlier_probability] and
  [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask] score the same neighbourhood
  for isolation, which is how a scanned cloud loses its stragglers *before* the normals are fitted.
  [`plane_basis`][ordito.points.plane_basis] and [`radial_sort`][ordito.points.radial_sort] then
  let a caller work in the tangent plane it defines.
- **Cleanup and subsampling.** Four masks and one selector, all returning indices or
  ``wp.array[wp.bool]`` rather than a copied cloud, so a caller pays for the gather only if it wants
  one: [`point_finite_mask`][ordito.points.point_finite_mask] and
  [`point_duplicate_mask`][ordito.points.point_duplicate_mask] are the two exact predicates,
  [`radius_outlier_mask`][ordito.points.radius_outlier_mask] and
  [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask] the two density ones (an
  absolute floor and a cloud-relative threshold), and
  [`farthest_point_sample`][ordito.points.farthest_point_sample] picks an exact count spread over
  the cloud's support. Run the exact predicates first: they are cheap and they remove the inputs the
  others are ill-defined on.
- **Convex hull.** [`convex_hull`][ordito.points.convex_hull] is the exact hull as outward-wound
  triangles over the cloud, decided by exact orientation signs, so thin, flat and degenerate
  clouds get the true hull rather than one within a tolerance. Two approximations answer only
  *which* points are hull vertices, each in a fixed number of parallel launches and each
  one-sided -- but in *opposite* directions:

    - [`convex_subset_mask`][ordito.points.convex_subset_mask] (and its point-returning form
      [`convex_subset`][ordito.points.convex_subset]) accumulates *positive* certificates: a
      direction a point is extremal along proves it is on the hull. Cheap and precise, but a hull
      vertex with no sampled certificate is dropped, so the result can be **smaller** than the
      hull-vertex set.
    - [`convex_superset_mask`][ordito.points.convex_superset_mask] accumulates *negative*
      certificates: a tetrahedron of hull points strictly containing a point proves that point is
      interior. It keeps everything it cannot rule out, so the result is **never smaller** than the
      hull-vertex set.

Normals from [`estimate_normals`][ordito.points.estimate_normals] are *unoriented* -- a plane fit
cannot pick a side. See [`ordito.repair`][ordito.repair] for orientation propagation.

One member here takes no cloud at all: [`vector_angle`][ordito.points.vector_angle] is the unsigned
angle between two ``wp.vec3`` arrays, geometry rather than array structure, so
[`ordito.array`][ordito.array] is no better a home.
"""

from typing import Literal, NamedTuple, cast

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_values, require_same_device, run_device_loop, slice_count
from ordito.config import is_deterministic
from ordito.constants import TILE_1D, TOLERANCE_ZERO
from ordito.kernels import array as kernel_array
from ordito.kernels import grouping as kernel_grouping
from ordito.kernels import points as kernel_points
from ordito.kernels import predicates as kernel_predicates
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels.algorithms import convex_hull as kernel_hull

# Relative slack, against the support extent along a direction, for recognizing a point as that
# direction's support point. It cannot be zero: the reducing kernel and the marking kernel compute
# the same dot product in different code, so FMA contraction can leave them a ULP apart and an
# exact-equality test would match nothing and fall back to an arbitrary shell vertex. It must also
# stay small -- a wide slack lets the lowest-index *near*-support point win, pulling shell vertices
# inward and costing selectivity. This is a float32 epsilon question, not a tuning knob, so it is
# not exposed.
SUPPORT_TIE_SLACK = wp.constant(wp.float32(1e-6))

# Point count from which the support sweeps after ``hull_support_extremes`` walk only the slices
# whose extremes reach a threshold (``_support_extremes``), instead of testing every point. Below
# it the slice tables and the per-thread walk cost more than the one-thread-per-point grid:
# 0.70-0.91x at 36 k points, 1.1-1.9x at 0.44-0.54 M and 2.4-3.0x at 14 M.
SUPPORT_SLICE_FILTER_FROM = 1 << 17

# Minimum normalized determinant (against the edge-length product) for a shell tetrahedron to be
# used. Rejection is free -- neighbouring, well-shaped tetrahedra cover the same region -- while a
# sliver's face normals are ill-conditioned cross products of nearly parallel edges, and that error
# is the one that can cost the superset guarantee. Chosen well above ``TOLERANCE_PLANAR`` for that
# reason, and paired with ``convex_superset_mask``'s ``margin`` default -- dropping this to 1e-9
# makes every margin in the useful range unsafe.
TETRAHEDRON_FLATNESS = wp.constant(wp.float32(1e-3))

# Point count from which ``convex_hull`` runs ``convex_superset_mask`` (at this subdivision level)
# before growing: one pass that certifies most of a very large cloud interior, cheaper than the
# first growth passes that would drop it. new/old: lucy (14 M) 0.72x, a 4 M ball 0.90x; at 1 M it
# lost on four of five clouds (0.95-1.12x), and on the scans below it (dragon, happy_buddha)
# 1.03-1.10x. Before growth interleaved its flip rounds the filter paid from 0.4 M points, by
# halving Flip-Flop's rounds on scans; interleaving removed most of those rounds itself.
# Subdivision 1 measured 0.84x / 0.92x on the two, and 2 lost on both.
_HULL_PREFILTER_FROM = 1 << 22

# V-criterion flip rounds per growth round, each followed by a pass moving the points through it.
# One round left the surface's fans to be straightened by Flip-Flop one step a round: on a drum
# (two identical cocircular rings) every proposal won its locks, but ~130 waves advanced one flip a
# round, 517 rounds at 16 k points a ring. Two rounds keep the surface near convex while it grows:
# drums 4.25 / 10.8 / 37.6 -> 2.9 / 3.5 / 4.2 ms at 1 k / 4 k / 16 k a ring, dragon 0.81x, thin
# disk 0.92x; small clouds 1.13-1.19x, a 1 M sphere surface 1.17x. Three and four lost everywhere,
# and the passes support at most two: a round's first pass hands its list to the second through the
# one ``COUNT_FOLLOW`` slot.
_GROWTH_FLIP_ROUNDS = 2
_HULL_PREFILTER_SUBDIVISIONS = 0

# A bound on Flip-Flop's rounds that no input reaches: the flips terminate on their own (each raises
# the volume or lowers the degree of a non-extreme vertex), and the cap only turns a defect into the
# reflex-edge error instead of a hang.
_FLIP_ROUND_CAP = 1 << 20

# Bytes in one ``wp.mat33``: where a fit's ticket sits in its two-matrix atomic buffer.
_MAT33_BYTES = 9 * 4

# Blocks above which an arrival-order fit (``fit_line`` / ``fit_plane`` / ``principal_axes`` with
# ``ordito.config`` determinism off) runs its SVD in a launch of its own rather than in the last
# block of the scatter launch. The election is one integer ticket a block on one counter, which
# measured equal device time to 2 048 blocks (2 M points) and 1.11x the separate launch's at 14 k
# (``lucy``); the reproducible commit elects its last block anyway and always fuses.
_FIT_TICKET_MAX_BLOCKS = 1 << 11


def point_plane_distance(
    points: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3 | None = None
) -> wp.array[wp.float32]:
    """
    Signed perpendicular distance of each point to a plane.

    The sign follows ``plane_normal``: positive on the side it points to, negative on the other,
    zero on the plane. An unsigned distance is ``wp.map(wp.abs, ...)`` over the result.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space.
    plane_normal
        Plane normal vector as ``wp.vec3``; need not be unit length, as the
        distance is normalized by its magnitude.
    plane_origin
        Point on the plane as ``wp.vec3``. When ``None``, the origin
        ``(0, 0, 0)`` is used.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` signed distances from each point to the plane on ``points.device``.

    Examples
    --------
    The normal need not be unit length; the sign says which side a point is on:

    ```python
    import warp as wp

    import ordito as od

    points = wp.array([[0.0, 0.0, 3.0], [1.0, 1.0, -0.5]], dtype=wp.vec3)
    normal, origin = wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, 1.0)
    distances = od.points.point_plane_distance(points, normal, origin)
    print(distances.numpy().tolist())  # [2.0, -1.5]
    ```
    """
    if plane_origin is None:
        plane_origin = wp.vec3(0.0, 0.0, 0.0)
    n = points.size
    out_distances = _launch.empty(n, dtype=wp.float32, device=points.device)
    _launch.map(
        kernel_points.point_plane_distance, points, plane_normal, plane_origin, out=out_distances
    )
    return out_distances


def half_space_mask(
    points: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3 | None = None
) -> wp.array[wp.bool]:
    """
    Flag the points strictly on the normal's side of a plane.

    The unbounded selection primitive: one plane cuts space in two and this is the half the normal
    points into. Composes into a convex region by intersecting several masks, which is what makes it
    worth having next to the box query — a slab, a wedge or a frustum is a handful of these, and
    none of them needs a spatial index because every point is tested independently.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.
    plane_normal
        Plane normal as ``wp.vec3``, pointing into the selected half. Need not be unit length: only
        the sign of the projection is read.
    plane_origin
        Point on the plane as ``wp.vec3``. When ``None``, the origin ``(0, 0, 0)`` is used.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` mask on ``points.device``. ``True`` marks a point to **keep**, the same sense as
        [`point_finite_mask`][ordito.points.point_finite_mask].

    Notes
    -----
    The test is **strict**, so a point exactly on the plane is excluded and the two masks for
    opposite normals are disjoint rather than overlapping -- with the plane ``z = 1``, a point at
    ``z = 1`` is in neither half -- and it makes the pair of masks a partition of the points off
    the plane.

    See Also
    --------
    [`point_plane_distance`][ordito.points.point_plane_distance]
        The signed distance this thresholds, when the magnitude is wanted too.
    [`ordito.neighbors.query_bvh_box`][ordito.neighbors.query_bvh_box]
        The bounded counterpart: selection by a box, through a BVH.
    [`ordito.array.flatnonzero`][ordito.array.flatnonzero]

    Examples
    --------
    Keep the vertices of a sphere above the plane ``z = 0``:

    ```python
    import warp as wp

    import ordito as od

    vertices, _ = od.creation.icosphere(subdivisions=2)
    upper = od.points.half_space_mask(vertices, wp.vec3(0.0, 0.0, 1.0))
    kept = od.array.gather(vertices, od.array.flatnonzero(upper))
    print(vertices.shape, kept.shape)  # (162,) (73,)
    ```
    """
    if plane_origin is None:
        plane_origin = wp.vec3(0.0, 0.0, 0.0)
    n = points.size
    out_mask = _launch.empty(n, dtype=wp.bool, device=points.device)
    if n == 0:
        return out_mask

    _launch.map(kernel_points.is_in_half_space, points, plane_normal, plane_origin, out=out_mask)
    return out_mask


def centroid(points: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Mean position ``sum(points) / n`` as a ``(1,)`` device ``wp.vec3`` array.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.

    Returns
    -------
    wp.array[wp.vec3]
        ``(1,)`` device array holding the centroid on ``points.device``.
        All-zeros when ``points`` is empty.

    Examples
    --------
    ```python
    import ordito as od

    vertices, _ = od.creation.box(bounds=[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    center = od.points.centroid(vertices)
    print(center.shape)  # (1,)
    print(center.numpy()[0])  # [1. 2. 3.]
    ```
    """
    return _point_sum(points, divisor=float(max(points.size, 1)))


def gram_matrix(points: wp.array[wp.vec3]) -> wp.array[wp.mat33]:
    """
    Uncentered Gram (scatter) matrix ``G = sum_k outer(x_k, x_k)``.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.

    Returns
    -------
    wp.array[wp.mat33]
        ``(1,)`` device array holding the ``3x3`` Gram matrix on ``points.device``. All-zeros when
        ``points`` is empty.

    Examples
    --------
    The diagonal holds the sums of squared coordinates, about the origin:

    ```python
    import warp as wp

    import ordito as od

    points = wp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3)
    gram = od.points.gram_matrix(points)
    print(gram.numpy()[0].diagonal())  # [1. 4. 0.]
    ```
    """
    # The uncentred Gram matrix is the scatter matrix around a zero center, which the kernel
    # reads off a null center array.
    return _scatter_matrix(points, None, 1.0)


def fit_line(points: wp.array[wp.vec3]) -> wp.vec3:
    """
    Major axis of a point set, weighted across all three singular directions.

    The result is ``normalize(S @ V)`` over the SVD of the **uncentered** point matrix -- a sum of
    all three right singular vectors weighted by their singular values. It is **not** the first
    principal axis, and the two part company as soon as the cloud is neither strongly elongated nor
    centred on the origin: they agree on a needle and diverge visibly on a merely anisotropic cloud,
    more so once it is offset from the origin. For the principal frame, use
    [`principal_axes`][ordito.points.principal_axes].

    Parameters
    ----------
    points
        ``(n,)`` positions in space.

    Returns
    -------
    wp.vec3
        Unit vector along the weighted major axis. The result is
        direction-only: its sign is governed by the SVD convention and is
        not meaningful.

    Notes
    -----
    This is the quantity [`trimesh.points.major_axis`][] computes. Because the weighted sum mixes
    all three singular vectors, its value depends on each one's sign, which no SVD fixes -- so the
    two implementations agree to ``1.000`` where one singular value dominates and to ``0.992`` on a
    moderate cloud.

    See Also
    --------
    [`principal_axes`][ordito.points.principal_axes]
        The first principal axis, and the full frame, of the *centred* cloud.
    [`fit_plane`][ordito.points.fit_plane]
        The complementary weakest direction.
    [`trimesh.points.major_axis`][]

    Examples
    --------
    On points along one line the axis is that line's direction, up to sign:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    direction = np.array([1.0, 2.0, 2.0]) / 3.0
    points = wp.array(np.outer(np.arange(1.0, 6.0), direction), dtype=wp.vec3)
    axis = od.points.fit_line(points)
    print(np.round(np.abs(np.array(axis)), 3))  # [0.333 0.667 0.667]
    ```
    """
    n = points.size
    if n == 0:
        return wp.vec3(0.0, 0.0, 0.0)

    # One launch: the (uncentred) Gram matrix G = sum_j outer(x_j, x_j), whose last block takes
    # its SVD and writes the axis.
    out_axis = _scatter_fit(points, None, 1.0, kernel_points.SCATTER_FIT_LINE, 1)
    return cast(wp.vec3, out_axis.list()[0])


def centered_covariance(
    points: wp.array[wp.vec3], center: wp.array[wp.vec3] | None = None
) -> wp.array[wp.mat33]:
    """
    Centered scatter matrix ``C = sum_k outer(x_k - mu, x_k - mu)`` (no ``1/n``).

    Centering happens inside the outer-product loop (rather than via the
    ``sum(x x^T) - n mu mu^T`` identity) to avoid float32 catastrophic
    cancellation.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.
    center
        ``(1,)`` precomputed centroid device array. When ``None`` it is computed on-device as
        ``sum(points) / n``.

    Returns
    -------
    wp.array[wp.mat33]
        ``(1,)`` device array holding the centered ``3x3`` scatter matrix on
        ``points.device``. All-zeros when ``points`` is empty.

    Raises
    ------
    RuntimeError
        If ``points`` and ``center`` are not all on one device.

    Examples
    --------
    Unlike [`covariance`][ordito.points.covariance], the sum is not divided by ``n - 1``:

    ```python
    import warp as wp

    import ordito as od

    points = wp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=wp.vec3)
    scatter = od.points.centered_covariance(points)
    print(scatter.numpy()[0][0, 0])  # 8.0
    ```
    """
    require_same_device(points=points, center=center)
    if center is not None:
        return _scatter_matrix(points, center, 1.0)
    return _scatter_matrix(points, _point_sum(points), float(points.size))


def fit_plane(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Fit a plane to a point set using SVD.

    The plane origin is the centroid of the points and the normal is the
    singular vector with the smallest singular value of the centered
    covariance matrix.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(normal, centroid)`` where ``normal`` is the plane's unit normal and ``centroid`` is a
        point on it. The normal is sign-ambiguous: its orientation is governed by the SVD
        convention. Orientation first, position last, matching
        [`principal_axes`][ordito.points.principal_axes] and
        [`bounds.oriented_bounding_box`][ordito.bounds.oriented_bounding_box] -- and so the pair
        splats directly into the ``(plane_normal, plane_origin)`` argument order every plane entry
        point takes. [`trimesh.points.plane_fit`][] returns the two the other way round.

    See Also
    --------
    [`principal_axes`][ordito.points.principal_axes]
        All three axes at once; this normal is its third row.
    [`plane_basis`][ordito.points.plane_basis]
        Two in-plane axes completing this normal into a frame.

    Examples
    --------
    A flat grid in ``z = 0`` fits the plane with normal ``+z`` or ``-z``:

    ```python
    import ordito as od

    vertices, _ = od.creation.grid(count=(8, 8))
    normal, origin = od.points.fit_plane(vertices)
    print(f"{abs(normal[2]):.3f}")  # 1.000
    ```
    """
    n = points.size
    if n == 0:
        # Both zero, so the order is readability rather than behaviour: normal, then centroid.
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    # Pass 1: the point sum on-device; each consumer divides it by ``n`` itself, bit for bit the
    # centroid ``centroid`` would return, so no division launch runs between the passes.
    point_sum = _point_sum(points)

    # Pass 2: covariance matrix of the centred points, whose last block takes its SVD and writes
    # the normal and the centroid. Centring before the outer products (rather than via the
    # sum(x x^T) - n c c^T identity) avoids float32 catastrophic cancellation. Both results land
    # in one buffer and cross to the host in one readback: they are two rows of three floats, so
    # what a second readback would buy is nothing and what it costs is another synchronization.
    out_plane = _scatter_fit(points, point_sum, float(n), kernel_points.SCATTER_FIT_PLANE, 2)
    normal, centroid_out = out_plane.numpy()
    return (wp.vec3(*normal), wp.vec3(*centroid_out))


def plane_basis(normal: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """
    Right-handed orthonormal basis ``(u, v)`` spanning the plane with the given ``normal``.

    Parameters
    ----------
    normal
        Plane normal as ``wp.vec3``; need not be unit length.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(u, v)`` unit vectors perpendicular to each other and to ``normal``, such that
        ``(u, v, normalize(normal))`` is right-handed.

    Examples
    --------
    The two axes complete the normal into a right-handed frame:

    ```python
    import warp as wp

    import ordito as od

    u, v = od.points.plane_basis(wp.vec3(0.0, 0.0, 2.0))
    print(f"{wp.dot(u, v):.3f}")  # 0.000
    print(wp.cross(u, v))  # [0.0, 0.0, 1.0]
    ```
    """
    # These four Warp builtins are Python-scope, so each pays builtin dispatch and the function --
    # which does no device work -- is entirely that. Both alternatives were measured and declined:
    # NumPy is a wash, because ``np.cross`` is itself slower than ``wp.cross``; hand-written
    # components are an order of magnitude cheaper but fork the one tangent-frame rule into a second
    # spelling beside ``kernels/predicates.plane_basis``, which is the duplicated-decision-rule
    # hazard this package treats as a correctness risk.
    # ``odt.normalize`` / ``odt.cross`` are those same builtins, re-exported over the concrete
    # vector types so a vector held in a variable resolves against them.
    unit_normal = odt.normalize(normal)
    axis = wp.vec3(1.0, 0.0, 0.0)
    normal_x, _, _ = unit_normal
    if abs(normal_x) > 0.9:
        axis = wp.vec3(0.0, 1.0, 0.0)
    u = odt.normalize(odt.cross(axis, unit_normal))
    v = odt.cross(unit_normal, u)
    return u, v


def covariance(points: wp.array[wp.vec3], ddof: int = 1) -> wp.array[wp.mat33]:
    """
    Sample covariance matrix ``(1 / (n - ddof)) sum_k outer(x_k - mu, x_k - mu)``.

    Matches ``numpy.cov(points.T, ddof=ddof)`` for the default ``ddof=1``.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.
    ddof
        Delta degrees of freedom; the divisor is ``n - ddof``. Defaults to ``1``.

    Returns
    -------
    wp.array[wp.mat33]
        ``(1,)`` device array holding the ``3x3`` covariance matrix on ``points.device``.

    Raises
    ------
    ValueError
        If ``n - ddof <= 0``.

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    points = wp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.points.covariance(points).numpy()[0][0, 0])  # 4.0
    print(f"{od.points.covariance(points, ddof=0).numpy()[0][0, 0]:.4f}")  # 2.6667
    ```
    """
    n = points.size
    if n - ddof <= 0:
        raise ValueError(f"covariance requires n > ddof, got n={n}, ddof={ddof}")
    return _scatter_matrix(points, _point_sum(points), float(n), divisor=float(n - ddof))


def principal_axes(points: wp.array[wp.vec3]) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
    """
    Principal frame of a point cloud: its three axes, spreads and centroid.

    The eigenvectors of the centred covariance, ordered widest first, as the **rows** of a proper
    rotation -- so ``rotation * (p - centroid)`` are the coordinates of ``p`` in the principal
    frame, and ``wp.transpose(rotation)`` maps back. This is the fit
    [`fit_line`][ordito.points.fit_line] and [`fit_plane`][ordito.points.fit_plane] each read one
    direction of: the first row is the true major axis and the third is the plane normal.

    Parameters
    ----------
    points
        ``(n,)`` positions in space.

    Returns
    -------
    rotation : wp.mat33
        Principal axes as rows, in descending order of spread. A proper rotation: orthonormal with
        determinant ``+1``, which fixes the third row's sign, so only the first two are ambiguous.
    eigenvalues : wp.vec3
        The three spreads, descending. These are the eigenvalues of the *scatter* matrix
        [`centered_covariance`][ordito.points.centered_covariance] returns, so they carry no
        ``1 / n``; divide by ``n - 1`` for the sample variances along each axis.
    centroid : wp.vec3
        Mean position, the origin of the frame.

    Notes
    -----
    Matches ``pyvista.principal_axes`` up to the sign of the first two axes, which no
    eigendecomposition fixes.

    See Also
    --------
    [`fit_line`][ordito.points.fit_line]
        A different estimator, and the reason this exists; see its docstring.
    [`fit_plane`][ordito.points.fit_plane]
    [`covariance`][ordito.points.covariance]
    [`ordito.bounds.oriented_bounding_box`][ordito.bounds.oriented_bounding_box]
        A searched box rather than a covariance fit, and tighter for it.

    Examples
    --------
    The corners of a ``4 x 2 x 1`` box spread most along ``x``:

    ```python
    import numpy as np

    import ordito as od

    vertices, _ = od.creation.box(extents=(4.0, 2.0, 1.0))
    rotation, eigenvalues, centroid = od.points.principal_axes(vertices)
    print(np.abs(np.array(rotation[0])).round(3))  # [1. 0. 0.]
    print(np.array(eigenvalues).round(3))  # [32.  8.  2.]
    ```
    """
    n = points.size
    if n == 0:
        return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), wp.vec3(), wp.vec3()

    point_sum = _point_sum(points)
    # The scatter matrix, whose last block takes its SVD. All three results in one buffer, read
    # back once: fifteen floats is less than a single readback's fixed cost, so three of them
    # bought three synchronizations and nothing else.
    frame = _scatter_fit(
        points, point_sum, float(n), kernel_points.SCATTER_PRINCIPAL_AXES, 5
    ).numpy()
    return (wp.mat33(*frame[:3].ravel()), wp.vec3(*frame[3]), wp.vec3(*frame[4]))


def estimate_normals(
    points: wp.array[wp.vec3],
    neighbor_idx: odt.Array2dInt32,
    *,
    orient_reference: wp.vec3 | None = None,
    camera_location: wp.vec3 | None = None,
    orientation: Literal["neighbors", "centroid"] = "neighbors",
) -> wp.array[wp.vec3]:
    """
    Estimate per-point normals by PCA over each point's neighbourhood.

    Each normal is the eigenvector of the smallest eigenvalue of the local
    covariance matrix accumulated over the point's neighbours — the standard
    choice, which Open3D's ``FastEigen3x3`` also makes, so the result matches the
    references up to sign. The neighbourhood is supplied by
    the caller as ``neighbor_idx``: build it with
    [`query_nearest`][ordito.neighbors.query_nearest] using a plain
    ``k`` for a k-nearest (KNN) neighbourhood, or with ``max_radius`` set for a
    radius-bounded (hybrid) neighbourhood — mirroring the two neighbour modes of
    Open3D's ``estimate_normals(max_nn, radius)``.

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device.
    neighbor_idx
        ``(n, k)`` table of neighbour indices per point, as returned by
        [`query_nearest`][ordito.neighbors.query_nearest] (unused slots
        marked ``-1``). A self-query table includes each point itself once, which
        is counted normally.
    orient_reference
        When given, normals are flipped to align with this fixed direction
        (``dot(normal, orient_reference) >= 0``), matching Open3D's
        ``orient_normals_to_align_with_direction``. Mutually exclusive with
        ``camera_location``.
    camera_location
        When given, normals are flipped to point toward this location
        (``dot(normal, camera_location - point) >= 0``), matching Open3D's
        ``orient_normals_towards_camera_location`` and MeshLab's ``compute_normal_for_point_clouds``
        flipped toward ``viewpos``. Mutually exclusive with ``orient_reference``.
    orientation
        The sign rule when neither ``orient_reference`` nor ``camera_location`` is given.
        ``"neighbors"`` (default) is the local SHOT rule of Tombari et al., pytorch3d's
        ``estimate_pointcloud_normals`` default (``disambiguate_directions``): a normal is kept
        when at least half of its neighbours' offsets from the point project strictly positively
        onto it and flipped otherwise, so it points to the side its neighbourhood bulges toward
        (inward on a convex surface). Neighbouring normals agree, but nothing makes the whole
        cloud face one way. ``"centroid"`` orients each normal outward from the whole-cloud
        centroid, a global rule valid for star-shaped clouds, which is what a surface
        reconstruction from the normals needs ([`screened_poisson`]
        [ordito.reconstruction.screened_poisson]). Ignored when either argument above is given.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` unit normals on ``points.device``, signed by ``orientation`` when neither
        orientation argument is given (Open3D leaves the sign arbitrary instead). Points with
        fewer than two valid neighbours (or a degenerate neighbourhood) receive a fallback
        ``(0, 0, 1)`` normal.

    Raises
    ------
    ValueError
        If both ``orient_reference`` and ``camera_location`` are given, if ``orientation`` is not
        ``"neighbors"`` or ``"centroid"``, or if ``neighbor_idx`` does not have one row per point.
    RuntimeError
        If ``points`` and ``neighbor_idx`` are not all on one device.

    See Also
    --------
    [`ordito.points.fit_plane`][ordito.points.fit_plane]
    [`ordito.reconstruction.triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud]

    Examples
    --------
    On a sphere's vertices the default sign points every normal inward, toward the side the
    neighbours bulge to; ``orientation="centroid"`` points them outward along the positions:

    ```python
    import ordito as od

    vertices, _ = od.creation.icosphere(subdivisions=2)
    neighbor_idx, _ = od.neighbors.query_nearest(vertices, vertices, 8)
    normals = od.points.estimate_normals(vertices, neighbor_idx)
    print(normals.shape)  # (162,)
    print(bool((normals.numpy() * vertices.numpy()).sum(axis=1).max() < -0.99))  # True
    outward = od.points.estimate_normals(vertices, neighbor_idx, orientation="centroid")
    print(bool((outward.numpy() * vertices.numpy()).sum(axis=1).min() > 0.99))  # True
    ```
    """
    require_same_device(points=points, neighbor_idx=neighbor_idx)
    if orient_reference is not None and camera_location is not None:
        raise ValueError("pass at most one of orient_reference and camera_location")
    if orientation not in ("neighbors", "centroid"):
        raise ValueError(f"orientation must be 'neighbors' or 'centroid', got {orientation!r}")

    odt.ensure_ndim(neighbor_idx, 2, dtype=wp.int32)

    device = points.device
    n = points.size
    # The launch is one thread per *point* and each reads its own row, so a table with fewer rows
    # than the cloud is an out-of-bounds read rather than a short answer -- and on the CPU device a
    # Warp array is host heap, so that is heap corruption with no exception in release mode
    # The sibling ``outlier_probability`` checks the same pair of shapes;
    # this one only checked the rank.
    if int(neighbor_idx.shape[0]) != n:
        raise ValueError(
            f"neighbor_idx must have one row per point, got {neighbor_idx.shape} for {n} points"
        )
    if n == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device)

    if camera_location is not None:
        orient_mode = kernel_points.ORIENT_CAMERA
        reference = camera_location
    elif orient_reference is not None:
        orient_mode = kernel_points.ORIENT_DIRECTION
        reference = orient_reference
    elif orientation == "neighbors":
        orient_mode = kernel_points.ORIENT_NEIGHBORS
        reference = wp.vec3(0.0, 0.0, 0.0)
    else:
        orient_mode = kernel_points.ORIENT_CENTROID
        reference = wp.vec3(0.0, 0.0, 0.0)

    # Only the centroid orientation reads the cloud's centroid, as its raw sum divided in the
    # kernel; the other two modes pass a null array and skip the reduction.
    centroid_mode = orient_mode == kernel_points.ORIENT_CENTROID
    point_sum = _point_sum(points) if centroid_mode else None
    out_normals = _launch.empty(n, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_points.estimate_point_normals,
        dim=n,
        inputs=[points, neighbor_idx, point_sum, wp.float32(n), orient_mode, reference],
        outputs=[out_normals],
        device=device,
    )
    return out_normals


def _point_sum(points: wp.array[wp.vec3], divisor: float = 1.0) -> wp.array[wp.vec3]:
    """``sum(points) / divisor`` as a ``(1,)`` device array, zero when ``points`` is empty."""
    device = points.device
    n = points.size
    if n == 0:
        return _launch.zeros(1, dtype=wp.vec3, device=device)
    # Block sums added in a fixed order, so the sum repeats bit for bit on CUDA.
    blocks = kernel_reduce.blocks_1d(n)
    out, stride = _launch.block_partials(blocks, 1, wp.vec3, device)
    _launch.launch_tiled(
        kernel_points.point_sum,
        dim=[blocks],
        inputs=[points, wp.float32(divisor), stride, out],
        block_dim=TILE_1D,
        device=device,
    )
    return _launch.view(out, 0, 1)


def _scatter_matrix(
    points: wp.array[wp.vec3],
    center: wp.array[wp.vec3] | None,
    center_divisor: float,
    divisor: float = 1.0,
) -> wp.array[wp.mat33]:
    """
    ``sum_k outer(x_k - c, x_k - c)`` with ``c = center[0] / center_divisor`` as a ``(1,)`` array.

    ``center`` is either a centroid (``center_divisor = 1.0``, which divides exactly) or the raw
    point sum with ``n``; ``None`` is the zero center, i.e. the uncentred Gram matrix. The result
    is divided by ``divisor``, as in ``_point_sum``.
    """
    device = points.device
    n = points.size
    if n == 0:
        return _launch.zeros(1, dtype=wp.mat33, device=device)
    blocks = kernel_reduce.blocks_1d(n)
    out, stride = _launch.block_partials(blocks, 1, wp.mat33, device)
    _launch.launch_tiled(
        kernel_points.centered_covariance,
        dim=[blocks],
        inputs=[
            points,
            center,
            wp.float32(center_divisor),
            wp.float32(divisor),
            stride,
            kernel_points.SCATTER_ONLY,
            blocks,
            None,
            out,
            None,
        ],
        block_dim=TILE_1D,
        device=device,
    )
    return _launch.view(out, 0, 1)


def _scatter_fit(
    points: wp.array[wp.vec3],
    center: wp.array[wp.vec3] | None,
    center_divisor: float,
    finish: wp.int32,
    rows: int,
) -> wp.array[wp.vec3]:
    """
    Run fit ``finish`` on ``_scatter_matrix(points, center, center_divisor)``; ``(rows,)`` result.

    One launch: the block that completes the matrix runs the fit on it
    (``kernel_points.write_scatter_fit``). ``points`` must be non-empty.
    """
    device = points.device
    blocks = kernel_reduce.blocks_1d(points.size)
    out_frame = _launch.empty(rows, dtype=wp.vec3, device=device)
    if not is_deterministic() and blocks > _FIT_TICKET_MAX_BLOCKS:
        # The arrival-order commit with a launch of its own for the fit: past this many blocks one
        # ticket a block on a single counter costs more device time than the launch.
        scatter = _scatter_matrix(points, center, center_divisor)
        _launch.launch(
            kernel_points.scatter_fit,
            dim=1,
            inputs=[finish, scatter, center, wp.float32(center_divisor), out_frame],
            device=device,
        )
        return out_frame
    ticket = None
    if is_deterministic():
        out, stride = _launch.block_partials(blocks, 1, wp.mat33, device)
    else:
        # The atomic commit's total and, in the zeroed matrix after it, the integer ticket that
        # elects the block to run the fit: one allocation, read as ``int32`` through a view.
        out, stride = _launch.zeros(2, dtype=wp.mat33, device=device), 1
        ticket = wp.array(
            ptr=cast(int, out.ptr) + _MAT33_BYTES, dtype=wp.int32, shape=1, device=device
        )
    _launch.launch_tiled(
        kernel_points.centered_covariance,
        dim=[blocks],
        inputs=[
            points,
            center,
            wp.float32(center_divisor),
            wp.float32(1.0),
            stride,
            finish,
            blocks,
            ticket,
            out,
            out_frame,
        ],
        block_dim=TILE_1D,
        device=device,
    )
    return out_frame


def outlier_probability(
    neighbor_idx: odt.Array2dInt32, neighbor_distance: odt.Array2dFloat32, *, scale: float = 3.0
) -> wp.array[wp.float32]:
    """
    Local Outlier Probability (LoOP) of each point, in ``[0, 1]``.

    A point is an outlier to the extent that its own neighbourhood is *stretched* relative to the
    neighbourhoods of the points in it. Following Kriegel et al., each point gets a "standard
    distance" ``sigma`` (the RMS distance to its neighbours), a local factor
    ``plof = sigma / mean(sigma over the neighbourhood) - 1``, and a probability
    ``max(0, erf(plof / (nplof * sqrt(2))))`` where ``nplof = scale * sqrt(mean(plof^2))`` over the
    whole cloud. Because the normalization is cloud-wide, the score is comparable across points but
    **not** across clouds — it is a rank, calibrated so that a threshold near ``0.8`` selects the
    tail.

    This is the measure behind MeshLab's ``compute_selection_point_cloud_outliers``, whose own
    ``propthreshold`` default is ``0.8``; threshold the result there to reproduce its selection
    (see Examples).

    Parameters
    ----------
    neighbor_idx
        ``(n, k)`` neighbour table, as returned by
        [`query_nearest`][ordito.neighbors.query_nearest] (unused slots marked ``-1``).
        A self-query table includes each point itself, which is counted normally — the same
        convention MeshLab's k-d tree query uses.
    neighbor_distance
        ``(n, k)`` distances aligned with ``neighbor_idx``; unused slots are ``inf``.
    scale
        The LoOP normalization factor ``lambda``. Larger values make the score more conservative
        (fewer points near ``1``). MeshLab fixes it at ``3``, which is the default here.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` probabilities on ``neighbor_idx.device``. A point whose neighbour row is empty
        scores ``0``. All-zeros when the cloud has no spread at all.

    Raises
    ------
    ValueError
        If ``scale <= 0``, or the two tables disagree in shape.
    RuntimeError
        If ``neighbor_idx`` and ``neighbor_distance`` are not all on one device.

    See Also
    --------
    [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask]
    [`estimate_normals`][ordito.points.estimate_normals]
    [`ordito.neighbors.query_nearest`][ordito.neighbors.query_nearest]

    Examples
    --------
    A point far from a sphere's vertices scores near ``1``; threshold at MeshLab's ``0.8``:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    sphere, _ = od.creation.icosphere(subdivisions=2)
    cloud = wp.array(np.vstack([sphere.numpy(), [[3.0, 0.0, 0.0]]]), dtype=wp.vec3)
    neighbor_idx, neighbor_distance = od.neighbors.query_nearest(cloud, cloud, 8)
    probability = od.points.outlier_probability(neighbor_idx, neighbor_distance)
    print(np.flatnonzero(probability.numpy() > 0.8).tolist())  # [162]
    ```
    """
    require_same_device(neighbor_idx=neighbor_idx, neighbor_distance=neighbor_distance)
    odt.ensure_ndim(neighbor_idx, 2, dtype=wp.int32)
    odt.ensure_ndim(neighbor_distance, 2, dtype=wp.float32)
    if neighbor_idx.shape != neighbor_distance.shape:
        raise ValueError(
            "neighbor_idx and neighbor_distance must have the same shape, got "
            f"{neighbor_idx.shape} and {neighbor_distance.shape}"
        )
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    device = neighbor_idx.device
    n = int(neighbor_idx.shape[0])
    if n == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    _mean, standard_distance, _count, _totals = _neighbor_distance_moments(
        neighbor_distance, mean_and_count=False, rms=True
    )
    # nplof = scale * sqrt(E[plof^2]) over the whole cloud: the sum of squares is folded by the
    # pass that writes the factors and left on the device -- the probability kernel reads it and
    # forms the normalizer itself.
    plof = _launch.empty(n, dtype=wp.float32, device=device)
    blocks = (n + TILE_1D - 1) // TILE_1D
    plof_sum_squares, stride = _launch.block_partials(blocks, 1, wp.float32, device)
    _launch.launch_tiled(
        kernel_points.local_outlier_factors,
        dim=[blocks],
        inputs=[standard_distance, neighbor_idx, stride],
        outputs=[plof, plof_sum_squares],
        block_dim=TILE_1D,
        device=device,
    )
    out_probability = _launch.empty(n, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_points.outlier_probabilities,
        dim=n,
        inputs=[plof, plof_sum_squares, wp.float64(scale), wp.float32(n)],
        outputs=[out_probability],
        device=device,
    )
    return out_probability


def statistical_outlier_mask(
    neighbor_distance: odt.Array2dFloat32, *, std_ratio: float = 2.0
) -> wp.array[wp.bool]:
    """
    Flag points whose mean neighbour distance exceeds ``mean + std_ratio * std`` over the cloud.

    Open3D's ``remove_statistical_outlier`` criterion, and the cheaper cousin of
    [`outlier_probability`][ordito.points.outlier_probability]: one global threshold on a single
    per-point statistic rather than a neighbourhood-relative one. It is the right choice for a
    cloud of roughly uniform density and the wrong one for a cloud that is dense in places and
    sparse in others, where the sparse region is entirely above a global threshold.

    Parameters
    ----------
    neighbor_distance
        ``(n, k)`` distances from
        [`query_nearest`][ordito.neighbors.query_nearest]; unused slots are ``inf`` and
        are excluded from the per-point mean. A self-query table contributes one zero distance per
        row, exactly as Open3D's ``SearchKNN`` does, so pass the same ``k`` Open3D gets as
        ``nb_neighbors``.
    std_ratio
        Multiplier on the cloud-wide standard deviation of the per-point means. Lower is stricter.
        Open3D's own examples use ``2.0``, the default here.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` mask on ``neighbor_distance.device``; ``True`` marks an **outlier** (the
        complement of Open3D's *keep* mask). A point with an empty or fully coincident
        neighbourhood is marked as an outlier, matching Open3D's ``avg > 0`` guard.

    Raises
    ------
    ValueError
        If ``neighbor_distance`` is not a rank-2 float32 array.

    See Also
    --------
    [`outlier_probability`][ordito.points.outlier_probability]
    [`ordito.neighbors.query_nearest`][ordito.neighbors.query_nearest]

    Examples
    --------
    The one point far from a sphere's vertices is flagged:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    sphere, _ = od.creation.icosphere(subdivisions=2)
    cloud = wp.array(np.vstack([sphere.numpy(), [[3.0, 0.0, 0.0]]]), dtype=wp.vec3)
    _, neighbor_distance = od.neighbors.query_nearest(cloud, cloud, 8)
    outliers = od.points.statistical_outlier_mask(neighbor_distance)
    print(od.array.flatnonzero(outliers).numpy().tolist())  # [162]
    ```
    """
    odt.ensure_ndim(neighbor_distance, 2, dtype=wp.float32)

    device = neighbor_distance.device
    n = int(neighbor_distance.shape[0])
    # `wp.empty`, not `wp.zeros`: the mask launch below writes every slot of `out_mask`.
    out_mask = _launch.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    # Cloud mean and (ddof=1) deviation over the *counted* rows only, exactly as Open3D divides by
    # its ``valid_distances``. Empty rows contribute zero to every sum, so plain reductions work.
    # The three sums -- counted rows, distance total, squared deviation -- stay on the device: the
    # moments pass folds the first two, the deviation pass reads the mean from them and the mask
    # launch the threshold from all three, so nothing is read back. Each is summed in a fixed
    # order, so the mask repeats on CUDA.
    mean_distance, _rms, count, totals = _neighbor_distance_moments(
        neighbor_distance, mean_and_count=True, rms=False, totals=True
    )
    assert totals is not None
    blocks = kernel_reduce.blocks_1d(n)
    deviation, stride = _launch.block_partials(blocks, 1, wp.float64, device)
    _launch.launch_tiled(
        kernel_points.accumulate_counted_deviation,
        dim=[blocks],
        inputs=[count, mean_distance, totals, stride],
        outputs=[deviation],
        block_dim=TILE_1D,
        device=device,
    )
    _launch.launch(
        kernel_points.statistical_outlier_from_totals,
        dim=n,
        inputs=[mean_distance, count, totals, deviation, wp.float64(std_ratio)],
        outputs=[out_mask],
        device=device,
    )
    return out_mask


def _neighbor_distance_moments(
    neighbor_distance: odt.Array2dFloat32, *, mean_and_count: bool, rms: bool, totals: bool = False
) -> tuple[
    wp.array[wp.float32] | None,
    wp.array[wp.float32] | None,
    wp.array[wp.int32] | None,
    wp.array[wp.float64] | None,
]:
    """
    Per-row ``(mean, rms, count)`` of a neighbour-distance table, ignoring ``inf`` slots.

    A moment a caller does not ask for comes back as ``None`` and is never written. With
    ``totals``, a fourth array holds ``(rows with a neighbour, sum of their means)`` in its first
    two slots.
    """
    device = neighbor_distance.device
    n = int(neighbor_distance.shape[0])
    out_mean = _launch.empty(n, dtype=wp.float32, device=device) if mean_and_count else None
    out_rms = _launch.empty(n, dtype=wp.float32, device=device) if rms else None
    out_count = _launch.empty(n, dtype=wp.int32, device=device) if mean_and_count else None
    rows_per_block = kernel_points.MOMENT_ROWS_PER_BLOCK
    blocks = (n + rows_per_block - 1) // rows_per_block
    out_totals, stride = None, 0
    if totals:
        out_totals, stride = _launch.block_partials(blocks, 2, wp.float64, device)
    _launch.launch_tiled(
        kernel_points.neighbor_distance_moments,
        dim=[blocks],
        inputs=[neighbor_distance, stride],
        outputs=[out_mean, out_rms, out_count, out_totals],
        block_dim=TILE_1D,
        device=device,
    )
    return out_mean, out_rms, out_count, out_totals


def radius_outlier_mask(
    points: wp.array[wp.vec3], radius: float, min_neighbors: int, *, grid: wp.HashGrid | None = None
) -> wp.array[wp.bool]:
    """
    Flag points with at most ``min_neighbors`` other points within ``radius``.

    Open3D's ``remove_radius_outlier`` criterion, and the local counterpart of
    [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask]: an *absolute* density
    floor rather than a cloud-wide threshold on a per-point statistic. That makes it the right
    choice where the density is known in advance (a scanner's nominal spacing) and the wrong one
    where it varies across the cloud, which is the trade the two run in opposite directions.

    Parameters
    ----------
    points
        ``(n,)`` point positions.
    radius
        Search radius, inclusive at exactly ``radius``. Must be ``> 0``.
    min_neighbors
        A point survives when it has **strictly more** than this many neighbours, counting
        *itself* — Open3D's ``nb_points`` rule exactly, so the same number gives the same answer.
        Must be ``>= 1``.
    grid
        Optional pre-built hash grid over ``points``, from
        [`hashgrid_from_points`][ordito.neighbors.hashgrid_from_points]. Pass it to hoist the
        build out of a sweep over several radii.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` mask on ``points.device``; ``True`` marks an **outlier** (the complement of
        Open3D's *keep* mask), matching
        [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask].

    Raises
    ------
    ValueError
        If ``radius <= 0`` or ``min_neighbors < 1``.
    RuntimeError
        If ``points`` and ``grid`` are not all on one device.

    Notes
    -----
    The counts are exact, and the rule is the reference's -- but Open3D's own
    ``remove_radius_outlier`` shares one ``KDTreeFlann`` across an OpenMP loop and its radius search
    is not thread-safe under that sharing, so it returns a *different answer run to run*, differing
    by a point or two between repetitions of the same cloud. Querying the same tree serially
    reproduces this function exactly. Expect a comparison against the filter to disagree on a
    handful of borderline points, and do not read that as a difference in the criterion.

    See Also
    --------
    [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask]
    [`outlier_probability`][ordito.points.outlier_probability]
    [`ordito.neighbors.query_ball_count`][ordito.neighbors.query_ball_count]

    Examples
    --------
    The one point with no neighbour within ``0.5`` is flagged:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    sphere, _ = od.creation.icosphere(subdivisions=2)
    cloud = wp.array(np.vstack([sphere.numpy(), [[3.0, 0.0, 0.0]]]), dtype=wp.vec3)
    outliers = od.points.radius_outlier_mask(cloud, 0.5, 3)
    print(od.array.flatnonzero(outliers).numpy().tolist())  # [162]
    ```
    """
    require_same_device(points=points, grid=grid)
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")

    device = points.device
    n = points.size
    if n == 0:
        return _launch.zeros(0, dtype=wp.bool, device=device)

    if grid is None:
        grid = od.neighbors.hashgrid_from_points(points, radius)
    # A self-query counts the point itself once, at distance 0 -- which is what makes
    # ``min_neighbors`` comparable with Open3D's ``nb_points`` without an off-by-one correction.
    out_mask = _launch.empty(n, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_points.radius_outlier_flags,
        dim=n,
        inputs=[points, grid.id, wp.float32(radius), wp.int32(min_neighbors), out_mask],
        device=device,
    )
    return out_mask


def point_finite_mask(points: wp.array[wp.vec3]) -> wp.array[wp.bool]:
    """
    Flag points whose three coordinates are all finite.

    Open3D's ``remove_non_finite_points`` predicate. A single ``NaN`` or infinity poisons every
    reduction the point enters — a bounding box, a covariance, a BVH build — so this is the first
    pass over a cloud read off a scanner, before any of the fits in this module.

    Parameters
    ----------
    points
        ``(n,)`` point positions.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` mask on ``points.device``. ``True`` marks a point to **keep**, which is the
        opposite sense from the outlier and duplicate masks in this module — the name is the tell:
        this one is named for what it selects.

    See Also
    --------
    [`point_duplicate_mask`][ordito.points.point_duplicate_mask]
        The other exact-predicate cleanup pass; run this one first, since a ``NaN`` position is
        never equal to itself under Open3D's rule.
    [`ordito.array.flatnonzero`][ordito.array.flatnonzero]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    points = wp.array(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [1.0, float("inf"), 0.0], [1.0, 1.0, 1.0]],
        dtype=wp.vec3,
    )
    finite = od.points.point_finite_mask(points)
    print(finite.numpy().tolist())  # [True, False, False, True]
    cleaned = od.array.gather(points, od.array.flatnonzero(finite))
    print(cleaned.shape)  # (2,)
    ```
    """
    device = points.device
    n = points.size
    out_mask = _launch.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    _launch.map(kernel_points.is_finite_point, points, out=out_mask)
    return out_mask


def point_duplicate_mask(points: wp.array[wp.vec3]) -> wp.array[wp.bool]:
    """
    Flag every point that repeats an **exactly** equal position seen earlier in the cloud.

    Open3D's ``remove_duplicated_points`` rule: bit-for-bit coordinate equality, keeping the
    *first* occurrence of each distinct position. This is deliberately not
    [`ordito.grouping.unique_rows`][ordito.grouping.unique_rows], whose ``wp.vec3`` key buckets
    each coordinate to about ``2.4e-4`` relative and so merges positions that are merely close;
    reach for that one when a *tolerance* is what you want, and for this one when equality is.

    One hashed pass does it: each point probes an open-addressing table of point indices and
    compares a candidate's three coordinate bit patterns against its own, so a hash collision is
    never read as a duplicate, and the table keeps each position's smallest index.

    Parameters
    ----------
    points
        ``(n,)`` point positions.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` mask on ``points.device``; ``True`` marks a **duplicate**, i.e. every
        occurrence but the first of each distinct position. Gathering by the complement keeps one
        representative of each in first-occurrence order, as Open3D does.

    Notes
    -----
    ``-0.0`` and ``+0.0`` merge, because IEEE-754 equality holds between them and the reference
    compares with ``==``; the packing folds the two bit patterns together to reproduce that.
    ``NaN`` is the one case that does **not** match the reference: ``NaN != NaN`` makes every
    ``NaN`` row its own class for Open3D, where a bit pattern is a bit pattern here and identical
    ``NaN`` rows merge. Run [`point_finite_mask`][ordito.points.point_finite_mask] first and the
    question does not arise. Positions are compared at ``float32``, so two points that differ only
    below ``float32`` resolution are one position here and two for a ``float64`` reference.

    See Also
    --------
    [`point_finite_mask`][ordito.points.point_finite_mask]
    [`ordito.grouping.unique_rows`][ordito.grouping.unique_rows]
        The tolerance-bucketed sibling, and the right choice when exact equality is too strict.
    [`ordito.grouping.first_occurrence_indices`][ordito.grouping.first_occurrence_indices]

    Examples
    --------
    Every occurrence but the first of a repeated position is flagged:

    ```python
    import warp as wp

    import ordito as od

    points = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=wp.vec3,
    )
    duplicates = od.points.point_duplicate_mask(points)
    print(duplicates.numpy().tolist())  # [False, False, True, True, True]
    print(points.shape[0] - od.reduce.sum(duplicates))  # 2
    ```
    """
    device = points.device
    n = points.size
    out_mask = _launch.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    # One open-addressing table of point indices, at least twice ``n`` slots, a power of two;
    # the first pass leaves each class slot holding its smallest index and each point its slot.
    slot_mask = kernel_grouping.hash_table_mask(n)
    first = _launch.full(slot_mask + 1, -1, dtype=wp.int32, device=device)
    slot = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_points.point_duplicate_first,
        dim=n,
        inputs=[points, slot_mask],
        outputs=[first, slot],
        device=device,
    )
    _launch.launch(
        kernel_points.point_duplicate_from_first,
        dim=n,
        inputs=[first, slot],
        outputs=[out_mask],
        device=device,
    )
    return out_mask


def farthest_point_sample(
    points: wp.array[wp.vec3], count: int, *, start: int = 0
) -> wp.array[wp.int32]:
    """
    Greedily pick ``count`` points, each as far as possible from those already picked.

    Open3D's ``farthest_point_down_sample``: the classic maximin subsample, which spreads its
    output over the cloud's *support* rather than its density — so unlike
    [`ordito.voxels.voxel_down_sample`][ordito.voxels.voxel_down_sample] it returns an exact
    count, and unlike a random subsample it cannot leave a hole. The greedy choice makes it
    inherently sequential in ``count``, and the whole sweep runs as **one persistent block** on the
    device: each round folds the newest sample into every point's running distance and takes the
    global arg-max as a block reduction, so nothing is launched or read back per round.

    Parameters
    ----------
    points
        ``(n,)`` point positions.
    count
        Number of points to select; must satisfy ``0 <= count <= n``. Zero returns an empty array,
        as Open3D's ``num_samples=0`` does.
    start
        Index of the first sample, Open3D's ``start_index``. Defaults to ``0``, which is its
        default too.

    Returns
    -------
    wp.array[wp.int32]
        ``(count,)`` indices into ``points``, on ``points.device``, in **selection order**:
        entry 0 is ``start`` and each later entry is the point farthest from every earlier one.
        Open3D returns the selected *points* instead, and its ``SelectByIndex`` emits them in
        ascending index order, so the sequence here is strictly more information than the reference
        gives back — the sets are the same.

    Raises
    ------
    ValueError
        If ``count`` is not in ``[0, n]``, or ``start`` is not a valid index into ``points``.

    Notes
    -----
    Ties are broken towards the lower index, matching the reference's strict ``>`` arg-max, and that
    is what makes the answer reproducible: the per-iteration arg-max is a single ``wp.atomic_max``
    over a key that packs the squared distance against the complemented index, so the winner does
    not depend on the order the atomics land in. Distances are compared in ``float32`` where Open3D
    uses ``float64``, so a cloud with two points at nearly equal distance from the chosen set can
    diverge from that reference at the point where the tie is resolved — and then, being greedy, for
    the rest of the sequence.

    See Also
    --------
    [`ordito.sample.sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise]
        The same "well-spread points" goal from a *mesh*, by dart throwing rather than greedily.
    [`ordito.voxels.voxel_down_sample`][ordito.voxels.voxel_down_sample]
    [`ordito.neighbors.nearest_neighbor_distance`][ordito.neighbors.nearest_neighbor_distance]

    Examples
    --------
    From a box corner, the farthest point is the opposite corner:

    ```python
    import ordito as od

    vertices, _ = od.creation.box()
    indices = od.points.farthest_point_sample(vertices, 2)
    print(indices.numpy().tolist())  # [0, 7]
    print(vertices.numpy()[indices.numpy()].tolist())  # [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    ```
    """
    device = points.device
    n = points.size
    if not 0 <= count <= n:
        raise ValueError(f"count must be in [0, {n}], got {count}")

    out_selected = _launch.empty(count, dtype=wp.int32, device=device)
    if count == 0:
        return out_selected
    if not 0 <= start < n:
        raise ValueError(f"start must be in [0, {n}), got {start}")

    # Scratch for the running squared distance to the chosen set; the kernel initializes it.
    min_distance_sq = _launch.empty(n, dtype=wp.float32, device=device)
    if n >= kernel_points.FARTHEST_BLOCK_LARGE_FROM:
        block_dim = kernel_points.FARTHEST_BLOCK_LARGE
    elif n >= kernel_points.FARTHEST_BLOCK_MID_FROM:
        block_dim = kernel_points.FARTHEST_BLOCK_MID
    else:
        block_dim = kernel_points.FARTHEST_BLOCK_SMALL
    # The whole greedy sweep runs as one persistent block; see the kernel for why.
    _launch.launch_tiled(
        kernel_points.farthest_point_sample_block,
        dim=(1,),
        inputs=[points, wp.int32(start), wp.int32(count), min_distance_sq, out_selected],
        block_dim=block_dim,
        device=device,
    )
    return out_selected


def convex_hull(points: wp.array[wp.vec3]) -> wp.array[wp.int32]:
    """
    Exact convex hull of a point cloud, as outward-wound triangles over the input points.

    Every decision is an exact orientation sign, so the result is the true convex hull of the
    ``float32`` points as given -- however thin, flat or degenerate the cloud -- not one within a
    tolerance: every point lies inside or on it, and every vertex is an extreme point of the cloud.
    The triangles index ``points`` directly (the [`scipy.spatial.ConvexHull`][] ``simplices``
    layout); [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices] turns
    them into a compact mesh.

    The hull is grown from a tetrahedron, splitting every face at its farthest point in parallel,
    flipping edges as it grows and dropping the points each round encloses, then made convex by
    parallel edge flips; see Notes.

    Parameters
    ----------
    points
        ``(n_points,)`` finite point positions, ``n_points >= 4``, not all coplanar.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer into ``points`` on ``points.device``, each
        triangle wound counter-clockwise seen from outside (its normal ``(b - a) x (c - a)`` points
        out). A hull with ``n_vertices`` vertices has ``2 * n_vertices - 4`` triangles.

    Raises
    ------
    ValueError
        If ``points`` holds fewer than 4 points, or no 4 of them span a tetrahedron (all
        coplanar, collinear or coincident, to within what ``float32`` coordinates can resolve).

    Notes
    -----
    The algorithm is ffHull (Gao, Cao, Tan and Huang, *Flip-Flop: Convex Hull Construction via
    Star-Shaped Polyhedron in 3D*, I3D 2013). Each growth round costs one pass over the points still
    outside the surface, and interior points leave the computation as soon as a round encloses them,
    so its cost follows the points near the hull rather than the cloud's size.

    **Degenerate input.** A facet with more than three vertices (coplanar points, as on a box) is
    triangulated arbitrarily, and of several coincident points one is kept as the vertex. Points
    lying exactly on a hull facet or edge are not vertices.

    See Also
    --------
    [`convex_superset_mask`][ordito.points.convex_superset_mask]
    [`convex_subset_mask`][ordito.points.convex_subset_mask]
    [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices]
    [`scipy.spatial.ConvexHull`][]

    Examples
    --------
    A box's 8 corners span its hull; a point inside is not a vertex:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, _ = od.creation.box()
    cloud = wp.array(np.vstack([vertices.numpy(), [[0.1, 0.2, -0.1]]]), dtype=wp.vec3)
    faces = od.points.convex_hull(cloud)
    print(faces.shape[0] // 3, np.unique(faces.numpy()).shape[0])  # 12 8
    ```
    """
    n_points = points.size
    if n_points < 4:
        raise ValueError(f"convex_hull needs at least 4 points; got {n_points}.")
    if n_points < _HULL_PREFILTER_FROM:
        return _convex_hull_faces(points, None)
    keep = od.array.flatnonzero(convex_superset_mask(points, _HULL_PREFILTER_SUBDIVISIONS))
    return _convex_hull_faces(cast("wp.array[wp.vec3]", od.array.gather(points, keep)), keep)


def _convex_hull_faces(
    points: wp.array[wp.vec3], remap: wp.array[wp.int32] | None
) -> wp.array[wp.int32]:
    """
    Return the hull's faces over ``points``, mapped through ``remap`` into the caller's points.

    Grows the star-shaped surface, flips it convex, then checks the result. Two outcomes rebuild
    from a subset of the surface's vertices, and both leave the hull unchanged because every point
    they drop is a convex combination of the others:

    - **Flip-Flop stalled** with reflex edges left. That takes exactly degenerate input (collinear
      runs of points on a hull edge) where removing a non-extreme vertex needs a flip through a
      collinear triangle. Every reflex edge then waits on a labelled vertex, and every label is
      certified non-extreme, so the rebuild drops the labelled vertices -- at least one -- and the
      loop terminates.
    - **A vertex lies on a hull facet or edge** (fewer than three strictly convex edges meet it):
      not a hull vertex. The rebuild keeps the extreme vertices alone, where neither outcome can
      recur -- an extreme point meets three strictly convex edges in any triangulation of the hull,
      and with no three of them collinear every reflex edge is flippable.
    """
    device = points.device
    # Empty: no remap, and no vertex flags from ``flag_extreme_vertices`` until a rebuild wants any.
    empty = _launch.empty(0, dtype=wp.int32, device=device)
    extreme_only = False
    while True:
        surface = _grow_star_surface(points)
        _flip_to_convex(points, surface)
        tri, adj, n_faces = surface.tri, surface.adj, surface.n_faces

        n_points = points.size
        convex, check = surface.convex, surface.check
        _launch.launch(
            kernel_hull.count_strict_edges,
            dim=n_faces,
            inputs=[points, tri, adj, convex, check],
            device=device,
        )
        face_flags = _launch.view(surface.face_scratch, 0, n_faces)
        _launch.launch(
            kernel_hull.flag_extreme_vertices,
            dim=n_faces,
            inputs=[tri, convex, check, empty, face_flags],
            device=device,
        )
        # Readback: the face count sizes the output, and the two checks decide whether it is final.
        reflex, non_extreme, live_faces = read_values(check, 0, kernel_hull.CHECK_SIZE)
        keep = None
        if reflex != 0:
            unlabelled = _launch.zeros(n_points, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_hull.flag_unlabelled_vertices,
                dim=n_faces,
                inputs=[tri, surface.label, unlabelled],
                device=device,
            )
            keep = od.array.flatnonzero_in_place(unlabelled)
            # A closed surface of ``live_faces`` triangles has ``live_faces / 2 + 2`` vertices.
            if extreme_only or keep.size == live_faces // 2 + 2:
                raise RuntimeError(f"convex_hull: {reflex} reflex edges survived the flips.")
        elif non_extreme != 0 and not extreme_only:
            # The same pass again, now writing the vertex flags (the check word is not read again).
            vertex_flags = _launch.zeros(n_points, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_hull.flag_extreme_vertices,
                dim=n_faces,
                inputs=[tri, convex, check, vertex_flags, face_flags],
                device=device,
            )
            keep = od.array.flatnonzero_in_place(vertex_flags)
            extreme_only = True
        if keep is None:
            break
        points = cast("wp.array[wp.vec3]", od.array.gather(points, keep))
        remap = keep if remap is None else cast("wp.array[wp.int32]", od.array.gather(remap, keep))

    _launch.array_scan(face_flags, face_flags, inclusive=True)
    out_faces = _launch.empty(3 * int(live_faces), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_hull.write_live_faces,
        dim=n_faces,
        inputs=[tri, face_flags, empty if remap is None else remap, out_faces],
        device=device,
    )
    return out_faces


class _HullSurface(NamedTuple):
    """The surface the hull is grown and flipped on, with Flip-Flop's per-face round state."""

    tri: wp.array[wp.vec3i]
    adj: wp.array[wp.vec3i]
    kernel_point: wp.array[wp.vec3d]
    n_faces: int
    dirty: wp.array[wp.int32]
    changed: wp.array[wp.int32]
    lock: wp.array[wp.int32]
    group: wp.array[wp.vec3i]
    proposal: wp.array[wp.vec3i]
    # Zeroed tables and words for after the growth: Flip-Flop's vertex labels (still all zero, as
    # the growth's flip rounds label nothing) and loop state word, the strictly convex edges at
    # each vertex and the closing check word.
    label: wp.array[wp.int32]
    state: wp.array[wp.int32]
    convex: wp.array[wp.int32]
    check: wp.array[wp.int32]
    # A per-face ``int32`` table the growth no longer reads, for the output's live-face flags.
    face_scratch: wp.array[wp.int32]


def _grow_star_surface(points: wp.array[wp.vec3]) -> _HullSurface:
    """
    Grow a star-shaped surface over every point that can be a hull vertex.

    The returned surface holds ``n_faces`` live rows. Every round splits, runs one flip round of
    the V-criterion alone, and moves the points through both: flipping as the surface grows keeps
    it close to the hull of its vertices, so a pass drops the points under its dents instead of
    inserting them as vertices Flip-Flop must later remove. One readback a round: the splits it
    made and the points still outside, which size the next round's launches.

    Raises
    ------
    ValueError
        If the cloud spans no tetrahedron.
    """
    device = points.device
    n_points = points.size
    # A closed surface over ``n_points`` vertices has at most ``2 * n_points - 4`` faces, so that
    # bounds every regrowth. The start and the 4x step keep a small cloud to one or two: every
    # regrowth reallocates the per-face tables.
    max_faces = 2 * n_points
    capacity = min(max_faces, 1 << 13)
    tables = _face_tables(capacity, device)
    # Every face of the tetrahedron starts dirty; a regrowth's split marks the faces it makes.
    _launch.fill_(tables.dirty, 1)
    keys = tables.keys
    _launch.fill_(keys, -1)
    stride = slice_count(n_points, device)
    for stage in range(4):
        _launch.launch(
            kernel_hull.pick_tetrahedron_corner,
            dim=stride,
            inputs=[points, stage, stride, keys],
            device=device,
        )
    tri, adj, best, flags, next_flags, dirty = (
        tables.tri,
        tables.adj,
        tables.best,
        tables.flags,
        tables.next_flags,
        tables.dirty,
    )
    kernel_point = _launch.empty(1, dtype=wp.vec3d, device=device)
    # One zeroed buffer holds both per-point tables and every small word, cut into views.
    zeroed = _launch.zeros(2 * n_points + kernel_hull.WORDS_SIZE, dtype=wp.int32, device=device)
    # No labels while growing: the flip rounds read an all-zero table and a constant round word.
    no_labels = _launch.view(zeroed, 0, n_points)
    convex = _launch.view(zeroed, n_points, 2 * n_points)
    words = 2 * n_points
    counters = _launch.view(zeroed, words, words + kernel_hull.WORD_GROWTH_STATE)
    growth_state = _launch.view(
        zeroed, words + kernel_hull.WORD_GROWTH_STATE, words + kernel_hull.WORD_FLIP_STATE
    )
    flip_state = _launch.view(
        zeroed, words + kernel_hull.WORD_FLIP_STATE, words + kernel_hull.WORD_CORNERS
    )
    corners = _launch.view(zeroed, words + kernel_hull.WORD_CORNERS, words + kernel_hull.WORD_CHECK)
    check = _launch.view(zeroed, words + kernel_hull.WORD_CHECK, words + kernel_hull.WORDS_SIZE)
    _launch.launch(
        kernel_hull.start_tetrahedron,
        dim=1,
        inputs=[points, keys, tri, adj, best, counters, corners, kernel_point, flip_state],
        device=device,
    )
    lists = _launch.empty(2 * n_points, dtype=wp.vec2i, device=device)
    live = _launch.view(lists, 0, n_points)
    next_live = _launch.view(lists, n_points, 2 * n_points)
    _launch.launch(
        kernel_hull.seed_points,
        dim=n_points,
        inputs=[points, corners, kernel_point, tri, counters, best, flags, live],
        device=device,
    )
    n_faces = 4
    n_live = n_points
    while True:
        # A round splits at most one face per live point, and every split adds two faces.
        needed = n_faces + 2 * min(n_faces, n_live)
        if needed > capacity:
            capacity = min(max(needed, 4 * capacity), max_faces)
            grown = _face_tables(capacity, device)
            # The surface's rows and this round's flags carry over; the rest is per round.
            for old, new in ((tri, grown.tri), (adj, grown.adj), (best, grown.best)):
                _launch.copy(new, old, count=n_faces)
            _launch.copy(grown.flags, flags, count=n_faces)
            _launch.copy(grown.dirty, dirty, count=n_faces)
            tables = grown
            tri, adj, best, flags, next_flags, dirty = (
                tables.tri,
                tables.adj,
                tables.best,
                tables.flags,
                tables.next_flags,
                tables.dirty,
            )
        _launch.array_scan(
            _launch.view(flags, 0, n_faces),
            _launch.view(tables.offsets, 0, n_faces),
            inclusive=True,
        )
        _launch.launch(
            kernel_hull.split_faces,
            dim=n_faces,
            inputs=[
                n_faces,
                flags,
                tables.offsets,
                tri,
                adj,
                best,
                tables.child,
                counters,
                dirty,
                next_flags,
                tables.split_corners,
            ],
            device=device,
        )
        # Everything up to this round's readback is launched over upper bounds -- the faces the
        # splits can make and the points the last round kept -- with the kernels skipping dead
        # slots and reading the real list lengths on the device, so the host issues the round
        # ahead of the GPU instead of waiting on it after the split.
        surface = _HullSurface(
            tri,
            adj,
            kernel_point,
            needed,
            dirty,
            tables.changed,
            tables.lock,
            tables.group,
            tables.proposal,
            no_labels,
            flip_state,
            convex,
            check,
            tables.offsets,
        )
        flags, next_flags = next_flags, flags
        for follow in range(_GROWTH_FLIP_ROUNDS):
            _flip_round(points, surface, no_labels, growth_state, 0)
            _launch.launch(
                kernel_hull.assign_points,
                dim=n_live,
                inputs=[
                    points,
                    kernel_point,
                    tri,
                    tables.child,
                    tables.split_corners,
                    tables.changed,
                    tables.group,
                    live,
                    counters,
                    best,
                    flags,
                    int(follow != 0),
                    int(follow == _GROWTH_FLIP_ROUNDS - 1),
                    next_live,
                ],
                device=device,
            )
            live, next_live = next_live, live
        # Readback: the round's splits size the face arrays, the survivors the next round.
        n_live, splits, _, status = read_values(counters, 0, 4)
        if status != int(kernel_hull.STATUS_OK):
            raise ValueError(
                "convex_hull: the points span no tetrahedron (all coplanar, collinear or "
                "coincident to float32 precision)."
            )
        n_faces += 2 * int(splits)
        if n_live == 0:
            return surface._replace(n_faces=n_faces)


class _FaceTables(NamedTuple):
    """The growth's per-face tables of one capacity, cut out of one buffer per dtype."""

    tri: wp.array[wp.vec3i]
    adj: wp.array[wp.vec3i]
    group: wp.array[wp.vec3i]
    proposal: wp.array[wp.vec3i]
    best: wp.array[wp.int64]
    keys: wp.array[wp.int64]
    flags: wp.array[wp.int32]
    next_flags: wp.array[wp.int32]
    changed: wp.array[wp.int32]
    dirty: wp.array[wp.int32]
    lock: wp.array[wp.int32]
    child: wp.array[wp.int32]
    offsets: wp.array[wp.int32]
    split_corners: wp.array[wp.vec4i]


def _face_tables(capacity: int, device: wp.DeviceLike) -> _FaceTables:
    """
    Allocate the per-face tables for ``capacity`` faces: dead faces, zero flags, free locks.

    Unused slots hold dead faces, so a flip round launched over an upper bound skips them. The
    ``int32`` tables zeroed together sit first in their buffer; ``dirty``, ``child`` and
    ``offsets`` start unset, and ``keys`` is four spare ``int64`` slots for the start.
    """
    rows = _launch.empty(4 * capacity, dtype=wp.vec3i, device=device)
    wide = _launch.empty(capacity + 4, dtype=wp.int64, device=device)
    ints = _launch.empty(7 * capacity, dtype=wp.int32, device=device)
    r = [_launch.view(rows, k * capacity, (k + 1) * capacity) for k in range(4)]
    i = [_launch.view(ints, k * capacity, (k + 1) * capacity) for k in range(7)]
    tables = _FaceTables(
        r[0],
        r[1],
        r[2],
        r[3],
        _launch.view(wide, 0, capacity),
        _launch.view(wide, capacity, capacity + 4),
        i[0],
        i[1],
        i[2],
        i[3],
        i[4],
        i[5],
        i[6],
        _launch.empty(capacity, dtype=wp.vec4i, device=device),
    )
    _launch.fill_(tables.tri, kernel_hull.DEAD_FACE)
    _launch.fill_(_launch.view(ints, 0, 3 * capacity), 0)
    _launch.fill_(tables.lock, int(kernel_hull.LOCK_FREE))
    return tables


def _flip_round(
    points: wp.array[wp.vec3],
    surface: _HullSurface,
    label: wp.array[wp.int32],
    state: wp.array[wp.int32],
    labels: int,
) -> None:
    """One Flip-Flop round's three launches; ``labels`` zero flips by the V-criterion alone."""
    device = points.device
    n_faces = surface.n_faces
    _launch.launch(
        kernel_hull.propose_flips,
        dim=n_faces,
        inputs=[
            points,
            surface.kernel_point,
            surface.tri,
            surface.adj,
            label,
            state,
            surface.dirty,
            surface.changed,
            surface.lock,
            labels,
            surface.proposal,
        ],
        device=device,
    )
    _launch.launch(
        kernel_hull.apply_flips,
        dim=n_faces,
        inputs=[
            surface.proposal,
            surface.lock,
            surface.tri,
            surface.adj,
            state,
            surface.changed,
            surface.group,
        ],
        device=device,
    )
    _launch.launch(
        kernel_hull.repair_flip_adjacency,
        dim=n_faces,
        inputs=[
            surface.changed,
            surface.group,
            label,
            state,
            surface.tri,
            surface.adj,
            surface.lock,
            surface.dirty,
        ],
        device=device,
    )


def _flip_to_convex(points: wp.array[wp.vec3], surface: _HullSurface) -> None:
    """
    Flip-Flop the grown surface into the convex hull of its vertices, in place.

    ``surface.label[v]`` ends non-zero for a vertex certified non-extreme (the round that labelled
    it).

    The face count never grows (a 3-1 flip leaves dead slots), so every round launches the same
    three kernels over the same buffers and the loop runs as one recorded device loop on CUDA.
    Every face starts dirty: the growth rounds decided their edges without the D-criterion.
    """
    device = points.device
    _launch.fill_(_launch.view(surface.dirty, 0, surface.n_faces), 1)
    # ``wp.capture_while`` reads the condition before the first round; ``start_tetrahedron`` seeded
    # it non-zero.
    state = surface.state
    max_rounds = wp.int32(_FLIP_ROUND_CAP)

    def flip_round() -> None:
        _flip_round(points, surface, surface.label, state, 1)
        _launch.launch(kernel_array.loop_advance, dim=1, inputs=[max_rounds, state], device=device)

    run_device_loop(device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), flip_round)


def convex_subset_mask(
    points: wp.array[wp.vec3], n_directions: int = 128, tolerance: float = 1e-6
) -> wp.array[wp.bool]:
    """
    Approximate the convex-hull vertices of a point cloud as a boolean mask.

    For any direction ``n``, the point maximizing ``⟨n, p⟩`` is a vertex of the convex hull, and the
    point minimizing it is the hull vertex farthest along ``-n``. A single dot-product sweep
    therefore yields the two hull vertices supporting ``+n`` and ``-n``. Directions are drawn on the
    positive-``z`` hemisphere with the deterministic Fibonacci spiral
    ([`sample_fibonacci_hemisphere`][ordito.sample.sample_fibonacci_hemisphere]); because the
    hemisphere and its reflection tile the full sphere, taking both the max and min per direction
    covers all ``2 * n_directions`` antipodal orientations at half the cost of sampling the full
    sphere.

    The extrema are computed by one thread per ``(direction, point slice)``, each reducing a strided
    slice of [`ITEMS_PER_SLICE_CUDA`][ordito.constants.ITEMS_PER_SLICE_CUDA] points
    (``ITEMS_PER_SLICE_CPU`` on the CPU device) and committing one atomic, then a second pass marks
    the maximizers and minimizers. At most roughly ``2 * n_directions`` points (plus ties) can be
    marked.

    !!! warning "The result is an inner approximation, not a superset"

        Marking a point requires a *certificate* -- a sampled direction it is extremal along -- so a
        hull vertex with no such direction is **dropped**. The mask is therefore a subset of the
        hull boundary, never a conservative superset of the hull vertices; see Notes for the exact
        guarantee and how to raise recall. When losing a hull vertex is unacceptable, use
        [`convex_superset_mask`][ordito.points.convex_superset_mask], which errs the other way by
        construction.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    n_directions
        Number of Fibonacci hemisphere directions. Each covers two antipodal
        orientations, so the effective coverage is ``2 * n_directions``. Larger
        values recover more of the hull vertices.
    tolerance
        Relative slack on the support test; a point is marked when its dot product
        with a direction is within ``tolerance * (max - min)`` of that direction's
        maximum or minimum, where ``max - min`` is the cloud's support extent along
        the direction. Scaling the slack with the extent captures coplanar ties and
        float32 roundoff at any coordinate scale.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_points,)`` mask on ``points.device``; ``True`` for points selected as approximate hull
        vertices. Empty when there are no points.

    Notes
    -----
    **What is guaranteed.** Every marked point lies on the convex-hull *boundary*, to within the
    ``tolerance`` slack. That is the invariant that holds for every input. The stronger statement --
    every marked point is a hull *vertex* -- holds only when no support direction ties, which is the
    generic case for a cloud in general position but fails on structured data: on a grid over each
    face of a cube the mask also returns the face-edge midpoints that tie with a corner along a face
    normal, at any ``n_directions``. Raising ``tolerance`` widens that effect deliberately.

    **What is not guaranteed.** A hull vertex is recovered only when a sampled direction falls
    inside its *normal cone*, so recall degrades with the flatness of the hull around a vertex, not
    with the point count. On 500 standard-normal points (31 hull vertices) the recovered fraction
    measures 0.61 at ``n_directions=32``, 0.87 at 256 and 1.00 at 16384; the vertices missed at 256
    have normal cones spanning a few parts in ten thousand of the sphere or less. Increasing
    ``n_directions`` is the only knob that raises recall -- ``tolerance`` trades precision for it
    and is not a substitute. When *all* hull vertices are required, use the exact
    [`convex_hull`][ordito.points.convex_hull]: no setting of these parameters makes this function
    conservative.

    See Also
    --------
    [`convex_subset`][ordito.points.convex_subset]
    [`convex_superset_mask`][ordito.points.convex_superset_mask]
    [`sample_fibonacci_hemisphere`][ordito.sample.sample_fibonacci_hemisphere]
    [`flatnonzero`][ordito.array.flatnonzero]
    [`scipy.spatial.ConvexHull`][]

    Examples
    --------
    Of a cube's corners and points sampled inside it, only the corners are marked:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    inside = od.sample.sample_volume(vertices, faces, 200, seed=0)
    cloud = wp.array(np.vstack([vertices.numpy(), inside.numpy()]), dtype=wp.vec3)
    mask = od.points.convex_subset_mask(cloud)
    print(od.array.flatnonzero(mask).numpy().tolist())  # [0, 1, 2, 3, 4, 5, 6, 7]
    ```
    """
    device = points.device
    n_points = points.size
    if n_points == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    out_mask = _launch.zeros(n_points, dtype=wp.bool, device=device)
    _mark_hull_support(points, n_directions, tolerance, kernel_points.mark_hull_support, out_mask)
    return out_mask


def convex_subset(
    points: wp.array[wp.vec3], n_directions: int = 128, tolerance: float = 1e-6
) -> wp.array[wp.vec3]:
    """
    Approximate the convex-hull vertices of a point cloud as a point subset.

    Convenience wrapper around
    [`convex_subset_mask`][ordito.points.convex_subset_mask] that returns the
    selected points directly, gathered from ``points`` in ascending index order. The
    accuracy is the mask's: the returned subset lies on the hull boundary but can omit
    hull vertices, and is not a conservative superset of them.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    n_directions
        Number of Fibonacci hemisphere directions (see
        [`convex_subset_mask`][ordito.points.convex_subset_mask]).
    tolerance
        Relative slack on the support test (see
        [`convex_subset_mask`][ordito.points.convex_subset_mask]).

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` subset of ``points`` selected as approximate hull vertices, ``m <= n_points``, on
        ``points.device``. Empty when there are no points.

    See Also
    --------
    [`convex_subset_mask`][ordito.points.convex_subset_mask]
    [`gather`][ordito.array.gather]
    [`scipy.spatial.ConvexHull`][]

    Examples
    --------
    Of a cube's corners and points sampled inside it, the corners are returned:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    inside = od.sample.sample_volume(vertices, faces, 200, seed=0)
    cloud = wp.array(np.vstack([vertices.numpy(), inside.numpy()]), dtype=wp.vec3)
    hull_points = od.points.convex_subset(cloud)
    print(hull_points.shape)  # (8,)
    ```
    """
    device = points.device
    n_points = points.size
    if n_points == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device)
    # ``convex_subset_mask`` as ``int32`` flags, scanned in place; the compaction then writes the
    # selected points themselves, so neither a mask nor an index array is built.
    flags = _launch.zeros(n_points, dtype=wp.int32, device=device)
    _mark_hull_support(points, n_directions, tolerance, kernel_points.flag_hull_support, flags)
    return od.array.compress_in_place(flags, points)


def _mark_hull_support(
    points: wp.array[wp.vec3],
    n_directions: int,
    tolerance: float,
    kernel: odt.Kernel,
    out_marks: wp.array[wp.bool] | wp.array[wp.int32],
) -> None:
    """
    Mark every point extremal along a sampled direction into a zeroed ``out_marks``.

    The support sweep [`convex_subset_mask`][ordito.points.convex_subset_mask] and
    [`convex_subset`][ordito.points.convex_subset] share: ``kernel`` is
    ``kernel_points.mark_hull_support`` for a ``bool`` mask or ``flag_hull_support`` for the
    ``int32`` flags a scan compacts.
    """
    device = points.device
    n_dir = int(n_directions)
    directions = od.sample.sample_fibonacci_hemisphere(n_dir, device=device)
    best_max, best_neg_min, stride, slice_max, slice_min = _support_extremes(points, directions)
    _launch.launch(
        kernel,
        dim=(n_dir, stride),
        inputs=[
            points,
            directions,
            best_max,
            best_neg_min,
            wp.float32(tolerance),
            stride,
            slice_max,
            slice_min,
            out_marks,
        ],
        device=device,
    )


def convex_superset_mask(
    points: wp.array[wp.vec3], subdivisions: int = 2, margin: float = 1e-5
) -> wp.array[wp.bool]:
    """
    Conservatively discard interior points, keeping a superset of the convex-hull vertices.

    Unlike [`convex_subset_mask`][ordito.points.convex_subset_mask], which keeps only points it can
    *prove* are on the hull, this keeps every point it cannot prove is *interior*. The certificate
    is a tetrahedron: the support point of the cloud along each direction of an
    [`icosphere`][ordito.creation.icosphere] is a hull vertex, so for every triangle ``(a, b, c)``
    of the icosphere the tetrahedron ``(centroid, s_a, s_b, s_c)`` has all four corners in the hull
    and therefore lies inside it. Any point strictly inside such a tetrahedron is strictly inside
    the hull and cannot be a hull vertex; every other point is kept.

    Because a hull vertex lies on the hull *boundary*, it is in no tetrahedron's strict interior, so
    **no hull vertex is ever discarded** -- for any input, any ``subdivisions``, and any ``margin``.
    The parameters trade only how many interior points survive. That makes this a prefilter: run it,
    then hand the survivors to an exact hull ([`scipy.spatial.ConvexHull`][] for instance, whose
    superlinear work then runs on a small fraction of the cloud).
    [`convex_hull`][ordito.points.convex_hull] needs no prefilter: its own growth rounds drop
    interior points as they are enclosed.

    Cost is one support sweep (``n_points`` times the ``10 * 4 ** subdivisions + 2`` icosphere
    directions) plus one pass testing each point against the ``20 * 4 ** subdivisions``
    tetrahedra, whose precomputed face planes every thread reads in lockstep.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    subdivisions
        Icosphere refinement level for the direction set, which also fixes the tetrahedron count.
        Higher values wrap the hull more tightly and so discard more interior points, at a
        proportionally higher cost; the guarantee is unaffected.
    margin
        Distance, as a fraction of the shell radius, by which a point must clear all four faces of a
        tetrahedron to count as strictly inside it. Raising it keeps *more* points, so it can only
        weaken the filter, never the guarantee. The default is set by float32 arithmetic rather than
        by taste: a point lying *exactly* on a face must not be read as inside. See Notes.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_points,)`` mask on ``points.device``; ``True`` for points that may be hull
        vertices, which includes every actual hull vertex. Empty when there are no points.

    Notes
    -----
    Selectivity depends strongly on the cloud's shape: a near-spherical cloud keeps a small
    fraction of its points (a triangulated inner shell hugs it tightly), while a flat-faced cloud
    keeps far more, because the shell cannot hug a plane and the survivors form a thin slab under
    each face.

    The guarantee is exact in real arithmetic; in float32 it rests on ``margin`` covering the error
    in the plane evaluation. Values of ``margin`` below ``1e-6`` can discard a true hull vertex --
    always a point lying *essentially exactly* on a tetrahedron face, where the computed distance
    straddles zero -- while ``1e-6`` and above are clean; the ``1e-5`` default sits well above that
    boundary, at a small cost in selectivity. ``TETRAHEDRON_FLATNESS`` is the other half of the
    same protection: it discards sliver tetrahedra whose face normals are too ill-conditioned to
    trust, and without it no margin in this range is safe.

    Degenerate input is handled by the same conservative logic rather than by a special case. A
    coplanar or collinear cloud makes every tetrahedron flat; flat tetrahedra are rejected as
    ill-conditioned, so nothing is certified interior and every point is kept, which is a valid
    (if useless) superset. Fewer than four points returns an all-``True`` mask directly.

    See Also
    --------
    [`convex_subset_mask`][ordito.points.convex_subset_mask]
    [`icosphere`][ordito.creation.icosphere]
    [`flatnonzero`][ordito.array.flatnonzero]
    [`scipy.spatial.ConvexHull`][]

    Examples
    --------
    Every vertex of a sphere is kept, and a point well inside it is discarded:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, _ = od.creation.icosphere(subdivisions=2)
    cloud = wp.array(np.vstack([vertices.numpy(), [[0.2, 0.1, 0.3]]]), dtype=wp.vec3)
    mask = od.points.convex_superset_mask(cloud).numpy()
    print(int(mask.sum()), bool(mask[-1]))  # 162 False
    ```
    """
    device = points.device
    n_points = points.size
    if n_points == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)
    if n_points < 4:
        # No tetrahedron exists, so nothing can be certified interior.
        return _launch.full(n_points, value=True, dtype=wp.bool, device=device)

    directions, shell_faces = od.creation.icosphere(subdivisions=int(subdivisions), device=device)
    n_dir = directions.size
    n_tetra = shell_faces.size // 3
    best_max, best_neg_min, stride, slice_max, _slice_min = _support_extremes(points, directions)

    # Seeded with the last index rather than a sentinel: a direction that somehow marks nothing
    # then yields a real point, which keeps the gather in range and the tetrahedra valid.
    support = _launch.full(n_dir, value=n_points - 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_points.support_indices,
        dim=(n_dir, stride),
        inputs=[
            points,
            directions,
            best_max,
            best_neg_min,
            SUPPORT_TIE_SLACK,
            stride,
            slice_max,
            support,
        ],
        device=device,
    )

    # A view, not a copy: ``support`` is a dense index array, and both kernels read the view.
    shell_vertices = points[support]
    # The shell's centroid and radius, packed in one ``wp.vec4``.
    shell = _launch.empty(1, dtype=wp.vec4, device=device)
    _launch.launch(kernel_points.shell_bounds, dim=1, inputs=[shell_vertices, shell], device=device)

    # `wp.empty`: a flat tetrahedron writes no planes, and `mark_hull_superset` reads a row
    # only where `valid` is set.
    planes = _launch.empty((n_tetra, 4), dtype=wp.vec4, device=device)
    valid = _launch.empty(n_tetra, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_points.tetrahedron_planes,
        dim=n_tetra,
        inputs=[shell_vertices, shell_faces, shell, TETRAHEDRON_FLATNESS, planes, valid],
        device=device,
    )

    out_mask = _launch.empty(n_points, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_points.mark_hull_superset,
        dim=n_points,
        inputs=[points, planes, valid, shell, wp.float32(margin), out_mask],
        device=device,
    )
    return out_mask


def _support_extremes(
    points: wp.array[wp.vec3], directions: wp.array[wp.vec3]
) -> tuple[
    wp.array[wp.float32],
    wp.array[wp.float32],
    int,
    odt.Array2dFloat32 | None,
    odt.Array2dFloat32 | None,
]:
    """
    Per-direction maximum and minimum of the support function over ``points``, and per slice.

    Returns ``(best_max, best_neg_min, stride, slice_max, slice_min)``, the minimum negated so both
    extremes start from ``-inf`` in one filled buffer: the threshold sweeps after this
    one launch ``(n_directions, stride)`` threads, thread ``(k, j)`` walking points ``j, j +
    stride, ...``. From ``SUPPORT_SLICE_FILTER_FROM`` points ``stride`` is the slice count and
    ``slice_max`` / ``slice_min``, each ``(n_directions, stride)``, bound every slice's own
    extremes, which let the sweeps walk only the slices that can hold a point at a threshold.
    Below it, ``stride`` is the point count and both tables are ``None``: the sweeps test every
    point in parallel, which a cloud that small does not leave enough slices to beat. Each thread
    reduces a strided slice of the cloud, so the launch is sized by
    [`items_per_slice`][ordito._device.items_per_slice] points per thread rather than by the point
    count -- enough parallelism to fill the device while keeping the number of atomics into the
    ``n_directions`` accumulator slots low. The same per-device slice length also backs three other
    strided reductions: ``visibility``'s support arg-max, ``proximity``'s winding-number sum and
    ``bounds``' oriented-box extents.
    """
    device = points.device
    n_points = points.size
    n_dir = directions.size
    n_slices = slice_count(n_points, device)

    best = _launch.full(2 * n_dir, value=-float("inf"), dtype=wp.float32, device=device)
    best_max = _launch.view(best, 0, n_dir)
    best_neg_min = _launch.view(best, n_dir, 2 * n_dir)
    stride = n_points
    slice_max = slice_min = None
    if n_points >= SUPPORT_SLICE_FILTER_FROM:
        stride = n_slices
        slice_max = odt.empty_2d((n_dir, n_slices), wp.float32, device=device)
        slice_min = odt.empty_2d((n_dir, n_slices), wp.float32, device=device)
    kernel, dim = kernel_points.HULL_SUPPORT_EXTREMES.launch_shape(n_dir, n_slices)
    _launch.launch(
        kernel,
        dim=dim,
        inputs=[points, directions, n_slices, best_max, best_neg_min, slice_max, slice_min],
        device=device,
    )
    return best_max, best_neg_min, stride, slice_max, slice_min


def vector_angle(a: wp.array[wp.vec3], b: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Unsigned angle in radians between pairs of vectors.

    For each index ``i``, computes ``atan2(norm(cross(a[i], b[i])), dot(a[i], b[i]))``, which lies
    in ``[0, pi]``. Matches [`trimesh.geometry.vector_angle`][] on stacked pairs, and agrees with
    its ``arccos(dot)`` to round-off on unit input while staying accurate for nearly parallel or
    nearly antiparallel pairs, where ``arccos`` loses most of its digits. Being scale-free, it also
    accepts unnormalized vectors, and pairs where either vector is zero read ``0.0``.

    Parameters
    ----------
    a
        ``(n,)`` vectors on the target device.
    b
        ``(n,)`` vectors on the same device as ``a``.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` unsigned angles in radians on ``a.device``. Empty when ``n == 0``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths.
    RuntimeError
        If ``a`` and ``b`` are not all on one device.

    See Also
    --------
    [`trimesh.geometry.vector_angle`][]

    Examples
    --------
    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    a = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3)
    b = wp.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=wp.vec3)
    angles = od.points.vector_angle(a, b)
    print(np.round(np.degrees(angles.numpy()), 3).tolist())  # [0.0, 90.0, 180.0]
    ```
    """
    require_same_device(a=a, b=b)
    device = a.device
    n = a.size
    if n != b.size:
        raise ValueError(f"a and b must have the same length, got {n} and {b.size}")

    if n == 0:
        return _launch.empty(0, dtype=wp.float32, device=device)

    out_angles = _launch.empty(n, dtype=wp.float32, device=device)
    _launch.map(kernel_predicates.vector_angle, a, b, out=out_angles)
    return out_angles


def radial_sort(
    points: wp.array[wp.vec3], origin: wp.vec3, normal: wp.vec3, start: wp.vec3 | None = None
) -> wp.array[wp.vec3]:
    """
    Sort points radially (by angle) around an axis and return them reordered.

    Points are projected onto two axes perpendicular to ``normal`` and ordered by
    the angle ``atan2`` of the projection, in **descending** order (matching
    [`trimesh.points.radial_sort`][]).

    Parameters
    ----------
    points
        ``(n,)`` positions in space.
    origin
        Point to sort around as ``wp.vec3``.
    normal
        Axis to sort around as ``wp.vec3``.
    start
        Optional ``wp.vec3`` specifying the start position in counter-clockwise
        order when viewed along ``normal``. Must not be parallel with ``normal``.
        When ``None``, an arbitrary perpendicular axis is used.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n,)`` input points reordered by descending angle, on ``points.device``. Empty when
        ``points`` is empty.

    Raises
    ------
    ValueError
        If ``start`` is provided and is (near-)parallel with ``normal``.

    Examples
    --------
    Four points around the ``z`` axis come back in angular order:

    ```python
    import warp as wp

    import ordito as od

    points = wp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=wp.vec3
    )
    origin, normal = wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0)
    ordered = od.points.radial_sort(points, origin, normal)
    print(ordered.numpy()[:, :2].tolist())  # [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    ```
    """
    device = points.device
    n = points.size
    if n == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device)

    # Build two axes perpendicular to each other and the normal, onto which the
    # points are projected to recover an angle. Done on the host since the axes
    # are a single O(1) setup shared by every point.
    if start is None:
        # Cross with whichever of x/y `normal` is *less* aligned with, so the cross product is
        # never near-degenerate (the same construction as `kernels.tangent_space.any_perpendicular`
        # -- but that one returns a specific unit chirality this needs to preserve exactly at
        # `normal = +z`, so it's spelled out here rather than called).
        #
        # The formula this replaces, `wp.vec3(normal[0], normal[2], -normal[1])`, is a faithful
        # port of ``trimesh.points.radial_sort``'s own axis0, and both share the same bug: for any
        # `normal` with a nonzero x-component, `dot(normal, axis0) == normal.x**2 != 0`, so axis0 is
        # not actually perpendicular to `normal` and carries a leftover component along it into
        # every point's angle -- exactly zero only at `normal = (0, *, *)`, which is why the
        # existing regression tests (fixed at `normal = (0, 0, 1)`) never caught it. This
        # construction is perpendicular to `normal` for every `normal`, and reduces to the replaced
        # formula's exact axis0/axis1 at `normal = (0, 0, 1)`: `abs(normal[0]) > abs(normal[1])` is
        # then `0 > 0`, False, so `helper = (1, 0, 0)` and
        # `cross((0, 0, 1), (1, 0, 0)) == (0, 1, 0)`, matching it bit for bit.
        helper = wp.vec3(1.0, 0.0, 0.0)
        normal_x, normal_y, _ = normal
        if abs(normal_x) > abs(normal_y):
            helper = wp.vec3(0.0, 1.0, 0.0)
        # Python-scope builtin dispatch, once per call against a per-point device sort; NumPy is no
        # cheaper here, ``np.cross`` being slower than ``wp.cross``. Same decline as ``plane_basis``
        # above, and for the same chirality reason.
        axis0 = odt.cross(normal, helper)
        axis1 = odt.cross(normal, axis0)
    else:
        unit_normal = odt.normalize(normal)
        unit_start = odt.normalize(start)
        if abs(1.0 - abs(odt.dot(unit_normal, unit_start))) < TOLERANCE_ZERO:
            raise ValueError("start must not be parallel with normal")
        axis0 = odt.cross(unit_start, unit_normal)
        axis1 = odt.cross(axis0, unit_normal)

    # Ascending radix sort of the negated angles yields the descending-angle order. The keys and
    # the identity payload go straight into the sort's double-width buffers.
    keys_buffer = _launch.empty(2 * n, dtype=wp.float32, device=device)
    order_buffer = _launch.empty(2 * n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_points.radial_sort_keys_and_identity,
        dim=n,
        inputs=[points, origin, axis0, axis1, keys_buffer, order_buffer],
        device=device,
    )
    _sorted_keys, order = od.array.sort_and_argsort_in_place(keys_buffer, order_buffer)
    return od.array.gather(points, order)
