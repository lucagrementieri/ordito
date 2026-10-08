"""
Generative sampling: new points on a mesh surface, inside its volume, or over a sphere.

Everything here *creates* points -- uniformly over the faces
([`sample_surface`][ordito.sample.sample_surface]), spaced at least ``radius`` apart
([`sample_surface_poisson_disk`][ordito.sample.sample_surface_poisson_disk],
[`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise]), inside a watertight solid
([`sample_volume`][ordito.sample.sample_volume]), or over a sphere, hemisphere or cone as a
low-discrepancy Fibonacci lattice (the ``sample_fibonacci_*`` family, which take a count and no
mesh at all).

*Subsampling* an existing cloud is the other question and lives with the data structures that
answer it: [`points.farthest_point_sample`][ordito.points.farthest_point_sample] for an exact
count, and [`voxels.voxel_down_sample`][ordito.voxels.voxel_down_sample] for one representative
per occupied cell. Neither belongs here -- each is a reduction of points a caller already has.

The one function here that reads like a subsampler and is not is
[`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise]: it draws its own dense pool
from the surface and thins *that*, so its input is a mesh and its output is new points, never a
subset of anything the caller passed.

See Also
--------
[`points.farthest_point_sample`][ordito.points.farthest_point_sample]
    Subsample an existing cloud to an exact count.
[`voxels.voxel_down_sample`][ordito.voxels.voxel_down_sample]
    Subsample an existing cloud to one point per occupied voxel.
"""

from __future__ import annotations

import math
import secrets
from typing import cast

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.array import flatnonzero, gather
from ordito.kernels import grouping as kernel_grouping
from ordito.kernels import sample as kernel_sample
from ordito.kernels.algorithms import blue_noise as kernel_blue_noise
from ordito.neighbors import query_ball_with_offsets
from ordito.triangles import face_normals_and_areas


def sample_fibonacci_sphere(count: int, device: wp.DeviceLike = None) -> wp.array[wp.vec3]:
    """
    Generate near-uniform unit vectors on the sphere via the Fibonacci spiral.

    Successive points are placed at multiples of the golden angle while their
    height ``z`` descends uniformly through ``(-1, 1)``, producing the Fibonacci
    lattice — a deterministic, low-discrepancy covering of the sphere that is far
    more even than independent random sampling for the same ``count``.

    Parameters
    ----------
    count
        Number of directions to generate.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    wp.array[wp.vec3]
        ``(count,)`` unit vectors on the sphere. Empty when ``count`` is 0.

    See Also
    --------
    [`sample_fibonacci_hemisphere`][ordito.sample.sample_fibonacci_hemisphere]
    """
    return _fibonacci_lattice(count, 2.0, device)


def sample_fibonacci_hemisphere(count: int, device: wp.DeviceLike = None) -> wp.array[wp.vec3]:
    """
    Generate near-uniform unit vectors on the positive-``z`` hemisphere.

    Same Fibonacci-spiral construction as
    [`sample_fibonacci_sphere`][ordito.sample.sample_fibonacci_sphere] but with
    ``z`` descending uniformly through ``(0, 1)``, so every direction has a
    positive ``z`` component. Because the hemisphere and its reflection tile the
    full sphere, pairing each direction ``n`` with its antipode ``-n`` (for
    example via a min/max reduction) covers all orientations with half the
    directions.

    Parameters
    ----------
    count
        Number of directions to generate.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    wp.array[wp.vec3]
        ``(count,)`` unit vectors on the positive-``z`` hemisphere. Empty when
        ``count`` is 0.

    See Also
    --------
    [`sample_fibonacci_sphere`][ordito.sample.sample_fibonacci_sphere]
    """
    return _fibonacci_lattice(count, 1.0, device)


def sample_fibonacci_cone(
    count: int, half_angle: float, device: wp.DeviceLike = None
) -> wp.array[wp.vec3]:
    """
    Generate near-uniform unit vectors inside a cone around ``+z``.

    The same Fibonacci-spiral construction as
    [`sample_fibonacci_sphere`][ordito.sample.sample_fibonacci_sphere], with ``z`` descending
    uniformly through ``(cos(half_angle), 1)`` instead of the whole range — which is the *correct*
    restriction, because a uniform ``z`` is a uniform solid angle (Archimedes) whether the band is
    the full sphere or a cap. Rejection-sampling a sphere lattice down to the cone would not stay
    low-discrepancy; this does.

    Parameters
    ----------
    count
        Number of directions to generate.
    half_angle
        Half-angle of the cone in **radians**, measured from ``+z``. ``pi / 2`` reproduces
        [`sample_fibonacci_hemisphere`][ordito.sample.sample_fibonacci_hemisphere] and ``pi``
        reproduces [`sample_fibonacci_sphere`][ordito.sample.sample_fibonacci_sphere]. Must be in
        ``(0, pi]``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    wp.array[wp.vec3]
        ``(count,)`` unit vectors inside the cone. Empty when ``count`` is 0.

    Raises
    ------
    ValueError
        If ``half_angle`` is outside ``(0, pi]``.

    See Also
    --------
    [`sample_fibonacci_hemisphere`][ordito.sample.sample_fibonacci_hemisphere]
    [`ordito.visibility.shape_diameter`][ordito.visibility.shape_diameter]
    """
    if not 0.0 < half_angle <= math.pi:
        raise ValueError(f"half_angle must be in (0, pi] radians, got {half_angle}")
    return _fibonacci_lattice(count, 1.0 - math.cos(half_angle), device)


def _fibonacci_lattice(count: int, z_span: float, device: wp.DeviceLike) -> wp.array[wp.vec3]:
    """
    Generate the Fibonacci lattice over a spherical band of height ``z_span``.

    The three public generators differ only in this number, because a uniform ``z`` is a uniform
    solid angle (Archimedes) whatever band it covers: ``2`` is the whole sphere, ``1`` the
    hemisphere and ``1 - cos(half_angle)`` a cone, and the first two are exactly what the third
    reduces to at ``half_angle`` of ``pi`` and ``pi / 2``.

    Parameters
    ----------
    count
        Number of directions to generate.
    z_span
        Height of the band in ``z``, in ``(0, 2]``.
    device
        Warp device for the result.

    Returns
    -------
    wp.array[wp.vec3]
        ``(count,)`` unit vectors. Empty when ``count`` is 0.
    """
    if count <= 0:
        return _launch.empty(0, dtype=wp.vec3, device=device)
    out_directions = _launch.empty(count, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_sample.fibonacci_lattice,
        dim=count,
        inputs=[count, wp.float32(z_span), out_directions],
        device=device,
    )
    return out_directions


def sample_surface(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    count: int,
    face_weight: wp.array[wp.float32] | None = None,
    seed: int | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample points uniformly on a triangle mesh surface (area-weighted faces).

    Uses ``face_normals_and_areas`` for default triangle weights. Builds a CDF and
    draws triangle indices by inverse-transform sampling it, then uniform points with
    ``wp.sample_triangle`` (same scheme as [`trimesh.sample.sample_surface`][]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle indices, ``(i0, i1, i2)`` per face.
    count
        Number of samples.
    face_weight
        ``(n_faces,)`` per-face weights. If ``None``,
        triangle areas from ``face_normals_and_areas`` are used.
    seed
        RNG seed for ``wp.rand_init``. If ``None``, a random seed is chosen.

    Returns
    -------
    samples
        ``(count,)`` sampled positions on the mesh surface.
    face_index
        ``(count,)`` triangle index for each sample.

    Raises
    ------
    ValueError
        If ``face_weight`` is given and its length is not the triangle count, if the mesh has no
        faces and ``count > 0``, or if the total face weight is not positive.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_weight`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces, face_weight=face_weight)
    n_faces = faces.size // 3
    # Validated before the ``count == 0`` short-circuit below, so a bad ``face_weight`` still
    # raises even when nothing would otherwise be sampled.
    if face_weight is not None and face_weight.size != n_faces:
        raise ValueError(
            f"face_weight length must match number of triangles (expected {n_faces}, "
            f"got {face_weight.size})"
        )
    if count == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=vertices.device),
            _launch.empty(0, dtype=wp.int32, device=vertices.device),
        )
    if n_faces == 0:
        # Without this, an empty ``weights``/``cdf`` reaches ``read_scalar``'s tail read below and
        # raises an unrelated ``IndexError`` (CPU) or a Warp slicing error (CUDA) instead.
        raise ValueError("mesh has no faces; cannot sample its surface")

    if face_weight is None:
        _, weights = face_normals_and_areas(vertices, faces)
    else:
        weights = face_weight

    # ``array_scan`` is inclusive by default, so the total is the scan's last element -- a 4-byte
    # tail read instead of a whole second reduction over the weights. Same trick as
    # ``array.flatnonzero`` and ``array.counts_to_offsets``.
    cdf = _launch.empty(n_faces, dtype=wp.float32, device=vertices.device)
    _launch.array_scan(weights, out_array=cdf)
    total = float(read_scalar(cdf))
    if total <= 0.0:
        raise ValueError("total face weight must be positive")

    out_points = _launch.empty(count, dtype=wp.vec3, device=vertices.device)
    out_face_indices = _launch.empty(count, dtype=wp.int32, device=vertices.device)
    _launch.launch(
        kernel_sample.sample_surface,
        dim=count,
        inputs=[
            vertices,
            faces,
            cdf,
            wp.float32(total),
            resolve_seed(seed),
            out_points,
            out_face_indices,
        ],
        device=vertices.device,
    )
    return out_points, out_face_indices


def sample_surface_poisson_disk(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    count: int,
    init_factor: float = 5.0,
    face_weight: wp.array[wp.float32] | None = None,
    seed: int | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample points on a triangle mesh surface with Poisson disk distribution.

    Uses Weighted Sample Elimination (Öztireli & Gross 2012): generates
    ``init_factor * count`` uniform surface samples, then iteratively removes
    the most "crowded" points in parallel rounds until ``count`` remain.
    Each round deletes, simultaneously, every alive point that is heavier than all of its alive
    neighbours and at least as heavy as the alive point ranked ``excess``-th by weight (``excess``
    being how many deletions remain). Maximality is decided on ``(weight, -index)``, so no two of
    them are within ``r_max`` of each other, and the weight floor keeps the deletions in the global
    weight order that sequential elimination follows: the rounds keep the samples the sequential
    algorithm would keep.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle indices, ``(i0, i1, i2)`` per face.
    count
        Number of output samples.
    init_factor
        Over-sampling factor; initial pool has ``init_factor * count`` points.
        Must satisfy ``init_factor >= 1``.
    face_weight
        ``(n_faces,)`` per-face weights passed to the initial uniform sampling, or ``None``.
    seed
        RNG seed for the initial uniform sampling. If ``None``, a random seed
        is chosen.

    Returns
    -------
    samples
        ``(count,)`` sampled positions on the mesh surface.
    face_index
        ``(count,)`` triangle index for each sample.

    Raises
    ------
    ValueError
        If ``init_factor < 1`` or if the mesh has no faces.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_weight`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces, face_weight=face_weight)
    device = vertices.device

    # Validated before the ``count == 0`` short-circuit below, matching ``sample_surface``.
    if init_factor < 1.0:
        raise ValueError(f"init_factor must be >= 1, got {init_factor}")

    if count == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
        )

    init_count = max(math.ceil(init_factor * count), count)

    # 1. Initial uniform surface samples
    init_points, init_face_indices = sample_surface(
        vertices, faces, init_count, face_weight=face_weight, seed=seed
    )

    # 2. Surface area (for radius computation)
    _, areas = face_normals_and_areas(vertices, faces)
    surface_area = float(cast("float", wp.utils.array_sum(areas)))

    # 3. Poisson disk radii (Öztireli & Gross 2012 constants)
    #
    # Plain Python floats, not ``wp.float32``: these are *host* arithmetic, and a Warp scalar's
    # operators route through Warp's Python-scope builtin dispatch, hundreds of times a plain
    # float's (see ``kernels/array.py``'s slot views and section 13.1). The wrapping
    # bought nothing even numerically, since ``wp.float32(x)`` only stores ``x`` and rounds when it
    # is marshalled into a launch, which ``wp.launch`` does for a plain float anyway.
    alpha = 8.0
    beta = 0.65
    gamma = 1.5
    ratio = float(count) / float(init_count)
    r_max = 2.0 * math.sqrt((surface_area / count) / (2.0 * math.sqrt(3.0)))
    r_min = r_max * beta * (1.0 - ratio**gamma)

    # 4. Neighbor lists (GPU, computed once for the full initial pool)
    nbr_idx, nbr_dists, offsets = query_ball_with_offsets(init_points, init_points, r_max)

    # 5. Initial per-point weights (parallel)
    alive = _launch.ones(init_count, dtype=wp.int32, device=device)
    weights = _launch.zeros(init_count, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_sample.compute_poisson_weights,
        dim=init_count,
        inputs=[nbr_idx, nbr_dists, offsets, alive, r_max, r_min, alpha, weights],
        device=device,
    )

    # 6. Parallel round-based elimination
    alive_count = init_count
    is_max = _launch.zeros(init_count, dtype=wp.int32, device=device)
    # The round's maxima count, accumulated by the flagging pass itself and re-zeroed once read.
    max_count = _launch.zeros(1, dtype=wp.int32, device=device)
    negated = _launch.empty(init_count, dtype=wp.float32, device=device)

    while alive_count > count:
        # Only a point at least as heavy as the ``excess``-th heaviest alive one may go this round,
        # so the deletions follow the global weight order the elimination is defined by; the
        # floor stays on the device as a one-element view of the sorted keys.
        _launch.map(kernel_sample.alive_negated_weight, weights, alive, out=negated)
        sorted_negated, _order = od.array.sort_and_argsort(negated)
        excess = alive_count - count
        weight_floor = sorted_negated[excess - 1 : excess]
        _launch.launch(
            kernel_sample.find_local_maxima,
            dim=init_count,
            inputs=[weights, alive, nbr_idx, offsets, weight_floor, is_max, max_count],
            device=device,
        )

        # The one readback per round: every branch below depends on the count.
        n_max = int(read_scalar(max_count, 0))
        _launch.zero_(max_count)
        if n_max == 0:
            # Nothing is flagged only when no alive point has an alive neighbour inside ``r_max``
            # -- otherwise the heaviest alive point with a neighbour is flagged, the test being a
            # strict ``>``. Every remaining point then carries weight 0, so they are equally good
            # and the round may delete any ``excess`` of them; ranking all of the alive ones keeps
            # the loop making progress, which the maxima alone no longer guarantee.
            deleted_mask = _top_maxima_by_weight(alive, weights, excess)
            n_max = excess
        elif n_max <= excess:
            deleted_mask = is_max
        else:
            deleted_mask = _top_maxima_by_weight(is_max, weights, excess)
            n_max = excess

        _launch.launch(
            kernel_sample.apply_deletions,
            dim=init_count,
            inputs=[deleted_mask, nbr_idx, nbr_dists, offsets, r_max, r_min, alpha, alive, weights],
            device=device,
        )
        alive_count -= n_max

    # 7. Compaction: the 0/1 alive flags are scanned in place (the loop is done with them) and one
    # launch writes the survivors and their faces.
    return _emit_kept_samples(alive, init_points, init_face_indices)


def _top_maxima_by_weight(
    candidates: wp.array[wp.int32], weights: wp.array[wp.float32], excess: int
) -> wp.array[wp.int32]:
    """
    Mark the ``excess`` heaviest flagged points, so a round deletes exactly enough.

    Two rounds need it and neither is an ordinary one: the last, where the local maxima outnumber
    what is left to delete, and a round in which *no* point is a local maximum because every alive
    point is isolated -- there ``candidates`` is the alive set rather than the maxima. Every other
    round deletes all of its maxima and never calls this.

    Sorting the flagged weights on the device is what keeps the loop free of readbacks: the host
    alternative reads the flags and ``weights`` back in full and picks the top ``excess`` with
    ``numpy.argsort``, moving ``2 * init_count`` elements across the bus where the rest of the loop
    moves none.

    Ties order differently from ``numpy.argsort``'s quicksort -- ``radix_sort_pairs`` is stable --
    and exact ties are common rather than rare: ``_poisson_edge_weight`` clamps any distance below
    ``r_min`` up to it, so a point whose neighbours are all closer than that carries exactly its
    neighbour count times one constant. Which of two equally-crowded points is dropped is not a
    property the algorithm defines, and a stable order at least makes the choice reproducible.

    Parameters
    ----------
    candidates
        ``(init_count,)`` ``0``/``1`` flags marking the points this round may delete -- its
        local weight maxima, or the whole alive set when there are none.
    weights
        ``(init_count,)`` crowding weights.
    excess
        How many of the flagged points to delete.

    Returns
    -------
    wp.array[wp.int32]
        ``(init_count,)`` ``0``/``1`` deletion flags with exactly ``excess`` ones.
    """
    n_pool = candidates.size
    flagged = flatnonzero(candidates)
    # Ascending on the negated weight is descending on the weight, and ``sort_and_argsort`` is the
    # package's one radix-sort spelling.
    descending = _launch.empty(flagged.size, dtype=wp.float32, device=candidates.device)
    _launch.map(wp.neg, gather(weights, flagged), out=descending)
    _sorted, order = od.array.sort_and_argsort(descending)
    # No clone: ``order`` need not outlive this frame (no further sort call reuses its scratch),
    # and ``gather`` only requires a contiguous index array, which a prefix slice already is.
    chosen = gather(flagged, odt.as_dense(order[:excess]))
    return od.array.astype(
        od.array.indices_to_mask(chosen, n_pool, device=candidates.device), wp.int32
    )


def sample_surface_blue_noise(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], radius: float, seed: int | None = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample points on a triangle mesh surface with a blue-noise (Poisson-disk) distribution.

    Draws a dense uniform surface pool — ``30x`` the expected output, the oversampling factor
    ``igl::blue_noise`` uses — and reduces it to a **maximal** subset in which no two points are
    within ``radius`` of each other, by randomized-priority parallel dart throwing. The result has
    the distribution of sequential dart throwing over a uniformly random order of the pool; see
    ``kernels/algorithms/blue_noise.py`` for why the parallelism costs nothing in distribution, and
    for the measured spacing and coverage against MeshLab and Open3D.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle indices, ``(i0, i1, i2)`` per face.
    radius
        Minimum Poisson disk radius (Euclidean distance in 3D). Enforced exactly: the closest pair
        in the output is never below it.
    seed
        RNG seed for the initial uniform sampling and the sampling order. If ``None``, a random seed
        is chosen. With a seed the output is reproducible — every round of the loop is a
        deterministic function of its input state.

    Returns
    -------
    samples
        ``(m,)`` sampled positions on the mesh surface. Count ``m`` is determined
        implicitly by ``radius`` and mesh area.
    face_index
        ``(m,)`` triangle index for each sample.

    Raises
    ------
    ValueError
        If ``radius <= 0``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    The count is **derived**, never requested: it is what a maximal ``radius``-packing of the
    surface comes to, so it lands near — not at — the hexagonal-packing estimate the radius is
    usually chosen from. Ask for a *count* with
    [`sample_surface_poisson_disk`][ordito.sample.sample_surface_poisson_disk] instead.

    A mesh with no faces returns two empty arrays rather than raising.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3

    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")

    if n_faces == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
        )

    _, areas = face_normals_and_areas(vertices, faces)
    surface_area = float(cast("float", wp.utils.array_sum(areas)))
    expected = surface_area * (math.pi * math.sqrt(3.0) / 6.0) / (math.pi * radius * radius / 4.0)
    nx = max(1, int(30.0 * expected))

    # The pool draw and the dart-throw priority draw both key off the same point index (``tid``
    # here, ``i`` in ``random_priorities``) at the same pool size, so a shared seed would give
    # ``wp.rand_init(seed, i)`` the identical initial state in both kernels -- ``sample_cdf``'s
    # first draw and ``random_priorities``'s only draw are then the exact same ``rand_pcg`` step,
    # making each point's dart-throw priority an exact, monotonic function of the very draw that
    # picked its face. A distinct, seed-derived salt for the priority draw removes that
    # correlation while staying a deterministic function of the caller's own seed.
    pool_seed = resolve_seed(seed)
    priority_seed = (pool_seed ^ 0x2545F491) & 0x7FFFFFFF
    init_points, init_face_indices = sample_surface(vertices, faces, nx, seed=pool_seed)
    return _dart_throw_blue_noise(init_points, init_face_indices, radius, priority_seed)


def _dart_throw_blue_noise(
    pool_points: wp.array[wp.vec3], pool_faces: wp.array[wp.int32], radius: float, seed: int
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Maximal Poisson-disk subset of ``pool_points`` by randomized-priority parallel dart throwing.

    See ``kernels/algorithms/blue_noise.py`` for why the result has the same distribution as one
    pass of the serial algorithm. The loop is a handful of rounds over a shrinking work list;
    the one host readback per round is the survivor count, which is also the termination test.

    **The round count is stable, and it is not where the cost is.** Measured over five mesh shapes
    and a fifty-fold range of pool sizes it is always a handful of rounds, growing logarithmically
    with the pool as randomized-priority maximal-independent-set theory predicts, and the
    pool-to-output ratio barely moves. What a round actually costs, from ``wp.timing_begin``
    (nothing here graph-captures, so the split is trustworthy): roughly half to two thirds device,
    and the overwhelming majority of the device half is two kernels -- ``dart_select_minima`` and
    ``dart_cover_neighbors``, the shell scans themselves. Those two are where any further win has to
    come from; the loop structure around them is already near its
    launch floor.
    """
    device = pool_points.device
    n_pool = pool_points.size
    empty = (
        _launch.empty(0, dtype=wp.vec3, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
    )
    if n_pool == 0:
        return empty

    # Background grid at cell size ``radius``, so a 3x3x3 neighbourhood covers the disk exactly.
    bbox_min, bbox_max = od.bounds.aabb(pool_points)
    inv_cell_size = np.float32(1.0 / radius)
    # The grid's side, one past the largest cell coordinate: ``kernels/algorithms/blue_noise.
    # grid_coord`` is monotone in the point, so that coordinate is the bounding box's upper
    # corner's, evaluated here in the kernel's float32 arithmetic.
    span = np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32)
    grid_w = int((span * inv_cell_size).astype(np.int32).max()) + 1

    # Bucket the pool by cell: one radix sort of the cell keys gives both the per-cell membership
    # lists and, through the run starts of the sorted keys, the distinct cells and the
    # sentinel-terminated bounds that index them. The keys and the identity payload are written
    # straight into the sort's double-width buffers, and the sort orders only the bits a key below
    # ``grid_w ** 3`` can set. The run marks and their scan live in the payload's upper half, free
    # once the sort has run; the scan doubles as every sorted point's cell index.
    keys = _launch.empty(2 * n_pool, dtype=wp.int64, device=device)
    order = _launch.empty(2 * n_pool, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_blue_noise.pool_cell_keys_and_order,
        dim=n_pool,
        inputs=[pool_points, bbox_min, wp.float32(float(inv_cell_size)), wp.int32(grid_w)],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(
        keys, order, count=n_pool, end_bit=max(1, (grid_w**3 - 1).bit_length())
    )
    sorted_keys = odt.as_dense(keys[:n_pool])
    bucket = odt.as_dense(order[:n_pool])
    cell_ranks = odt.as_dense(order[n_pool:])
    _launch.launch(
        kernel_grouping.MARK_SORTED_RUN_STARTS[wp.int64],
        dim=n_pool,
        inputs=[sorted_keys, cell_ranks],
        device=device,
    )
    _launch.array_scan(cell_ranks, cell_ranks, inclusive=True)
    # The distinct cell count sizes the cell tables.
    n_cells = int(read_scalar(order, 2 * n_pool - 1))
    unique_keys = _launch.empty(n_cells, dtype=wp.int64, device=device)
    cell_offsets = _launch.empty(n_cells + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_grouping.EMIT_SORTED_UNIQUE[wp.int64],
        dim=n_pool,
        inputs=[sorted_keys, bucket, cell_ranks, unique_keys, None, cell_offsets, None],
        device=device,
    )

    # **The whole dart loop runs in cell-sorted index space from here on.** ``bucket`` is already
    # the cell-order permutation, so permuting the payload through it once makes each cell's
    # members the contiguous run its offsets name -- after which the two sweeps that dominate this
    # call read their neighbours at stride 1 instead of scattering into the unsorted pool for every
    # candidate of every one of ``DART_SHELL_CELLS`` cells. ``bucket`` itself survives only as the
    # original-index tie-break (see ``dart_select_minima``) and as the map back at the end. The
    # per-point cell table is born sorted, read off the sorted keys, and the points are permuted
    # in the setup launch below.
    cell_neighbors = odt.empty_2d(
        (n_cells, kernel_blue_noise.DART_SHELL_CELLS), wp.int32, device=device
    )
    _launch.launch(
        kernel_blue_noise.dart_cell_neighbors,
        dim=(n_cells, kernel_blue_noise.DART_SHELL_CELLS),
        inputs=[unique_keys, wp.int32(grid_w), cell_neighbors],
        device=device,
    )

    # The sorted points, the per-point cell table, the priorities, the state, the first round's
    # work list and its per-cell priority summary, one launch. The priority is keyed on the
    # *original* pool index even though it is stored in sorted space: the priority a point holds is
    # what decides the packing, so it has to stay the same function of the seed and the point rather
    # than of where the sort happened to put it.
    sorted_points = _launch.empty(n_pool, dtype=wp.vec3, device=device)
    sorted_cell = _launch.empty(n_pool, dtype=wp.int32, device=device)
    priority = _launch.empty(n_pool, dtype=wp.uint32, device=device)
    state = _launch.empty(n_pool, dtype=wp.int32, device=device)
    # Work-list buffers sized for their final use once: the first round's list is the whole pool and
    # every later one is a prefix of it, so nothing here is reallocated per round.
    alive = _launch.empty(n_pool, dtype=wp.int32, device=device)
    # Per-cell summaries that let each round's two sweeps skip a shell cell whole; see the kernel
    # module for what each one summarises and why the accepted set is unchanged. Both are refilled
    # per round rather than accumulated, so a cell stops pruning the moment it stops being empty;
    # every later round's priority summary is folded into the compaction that produces its work
    # list (see ``dart_compact_alive``).
    cell_min_priority = _launch.full(
        n_cells, kernel_blue_noise.DART_NO_PRIORITY, dtype=wp.uint32, device=device
    )
    cell_accepted = _launch.empty(n_cells, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_blue_noise.dart_point_setup,
        dim=n_pool,
        inputs=[wp.int32(seed), pool_points, cell_ranks, bucket],
        outputs=[sorted_points, sorted_cell, priority, state, alive, cell_min_priority],
        device=device,
    )
    next_alive = _launch.empty(n_pool, dtype=wp.int32, device=device)
    survivor_flag = _launch.empty(n_pool, dtype=wp.int32, device=device)
    positions = _launch.empty(n_pool, dtype=wp.int32, device=device)
    alive_count = n_pool
    rr = wp.float32(radius * radius)

    while alive_count > 0:
        view = alive[:alive_count]
        _launch.fill_(cell_accepted, False)
        # The survivor flags are written by the covering sweep itself, as each thread's last word
        # on its point's state this round.
        alive_flags = survivor_flag[:alive_count]
        _launch.launch(
            kernel_blue_noise.dart_select_minima,
            dim=alive_count,
            inputs=[
                sorted_points,
                priority,
                sorted_cell,
                cell_neighbors,
                bucket,
                cell_offsets,
                view,
                cell_min_priority,
                rr,
                state,
                cell_accepted,
            ],
            device=device,
        )
        _launch.launch(
            kernel_blue_noise.dart_cover_neighbors,
            dim=alive_count,
            inputs=[
                sorted_points,
                sorted_cell,
                cell_neighbors,
                cell_offsets,
                view,
                cell_accepted,
                rr,
                state,
                alive_flags,
            ],
            device=device,
        )
        # Survivors of this round, compacted in place. The scan is **inclusive**, so its last
        # entry is the survivor count outright and one 4-byte read serves both the next launch
        # dimension and the loop's exit test; ``dart_compact_alive`` writes at ``positions[t] - 1``
        # to match. The exclusive form needed a second read for the last element's own flag, and a
        # readback is the most expensive thing in a round -- two of them are a substantial share of
        # the whole call at the small end, where the rounds are cheapest and most numerous relative
        # to the work. This is the shape ``array.flatnonzero`` already uses.
        #
        # The two windows are viewed once per round. ``alive_count`` shrinks every round so they
        # are not loop-invariant, but taking each twice and three times inside one round was five
        # ``wp.array.__getitem__`` calls where two do; and ``read_scalar`` takes its own one-element
        # slice internally, so handing it the index rather than a pre-sliced view drops a sixth.
        alive_positions = positions[:alive_count]
        _launch.array_scan(alive_flags, out_array=alive_positions, inclusive=True)
        total = int(read_scalar(positions, alive_count - 1))
        if total > 0:
            _launch.fill_(cell_min_priority, kernel_blue_noise.DART_NO_PRIORITY)
            _launch.launch(
                kernel_blue_noise.dart_compact_alive,
                dim=alive_count,
                inputs=[
                    view,
                    state,
                    alive_positions,
                    priority,
                    sorted_cell,
                    next_alive[:total],
                    cell_min_priority,
                ],
                device=device,
            )
            alive, next_alive = next_alive, alive
        alive_count = total

    # ``state`` is indexed by sorted position; scattering the accepted flags through ``bucket``
    # permutes them back to pool order, which is what keeps the returned points in the pool's own
    # order rather than the cell sort's. The flags are scanned in place (the survivor positions'
    # buffer is free now) and one launch compacts the points and their faces together.
    _launch.launch(
        kernel_blue_noise.dart_accepted_pool_flags,
        dim=n_pool,
        inputs=[state, bucket, positions],
        device=device,
    )
    return _emit_kept_samples(positions, pool_points, pool_faces)


def _emit_kept_samples(
    flags: wp.array[wp.int32], points: wp.array[wp.vec3], face_indices: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the samples whose ``flags`` entry is 1, and their faces, in input order.

    ``flags`` holds 0/1 keep flags and is scanned in place, so the caller must be done with it.
    One scan, one readback for the kept count and one compaction launch writing both outputs.
    """
    device = points.device
    n = points.size
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    # The kept count sizes the outputs.
    n_kept = int(read_scalar(flags, n - 1))
    kept_points = _launch.empty(n_kept, dtype=wp.vec3, device=device)
    kept_faces = _launch.empty(n_kept, dtype=wp.int32, device=device)
    if n_kept > 0:
        _launch.launch(
            kernel_sample.emit_kept_samples,
            dim=n,
            inputs=[flags, points, face_indices],
            outputs=[kept_points, kept_faces],
            device=device,
        )
    return kept_points, kept_faces


def sample_volume(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], count: int, seed: int | None = None
) -> wp.array[wp.vec3]:
    """
    Sample points uniformly inside a watertight triangle mesh volume.

    Fans tetrahedra from the mesh's area-weighted surface centroid, builds a CDF
    from signed tetrahedron volumes, and draws uniform points inside each selected one via
    the order-statistics barycentric method (zero rejection for meshes that are
    star-shaped with respect to their centroid).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle indices, ``(i0, i1, i2)`` per face.
    count
        Number of samples.
    seed
        RNG seed for ``wp.rand_init``. If ``None``, a random seed is chosen.

    Returns
    -------
    samples
        ``(count,)`` positions inside the mesh volume.

    Raises
    ------
    ValueError
        If the mesh is not watertight (open boundary edges detected).
    ValueError
        If the mesh has zero total volume, some signed tetrahedron volumes are negative after
        fanning from the centroid (the mesh is not star-shaped with respect to its own centroid,
        e.g. a torus), or the mesh has no faces and ``count > 0``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3

    # These two validations run regardless of ``count`` -- unlike the "no faces" guard below,
    # which only matters once something is actually being sampled -- so a caller cannot skip a
    # documented raise on a malformed mesh by asking for zero points.
    if not od.validation.is_edge_manifold(
        faces, allow_boundary_edges=False, n_vertices=vertices.size
    ):
        raise ValueError(
            "mesh is not watertight; tetrahedral decomposition requires a closed surface"
        )

    center = od.measures.surface_centroid(vertices, faces)

    signed_vols = od.triangles.face_signed_volumes(vertices, faces, center)

    # One device reduction, not two: the star-shaped test genuinely needs a ``min``, but the total
    # is the inclusive scan's last element and comes for free with the CDF this builds anyway.
    # ``n_faces > 0`` guards the empty mesh, where "no negative volumes" holds vacuously and
    # ``reduce.min`` would otherwise raise its own unrelated "requires a non-empty array" error.
    if n_faces > 0 and od.reduce.min(signed_vols) < 0.0:
        raise ValueError(
            "mesh is not star-shaped with respect to its centroid (e.g. a torus); "
            "tetrahedral decomposition cannot sample it without rejection"
        )

    if count == 0:
        return _launch.empty(0, dtype=wp.vec3, device=vertices.device)

    if n_faces == 0:
        # Without this, an empty ``signed_vols``/``cdf`` reaches ``read_scalar``'s tail read below
        # and raises an unrelated ``IndexError`` (CPU) or a Warp slicing error (CUDA) instead.
        raise ValueError("mesh has no faces; cannot sample its volume")

    cdf = _launch.empty(n_faces, dtype=wp.float32, device=vertices.device)
    _launch.array_scan(signed_vols, out_array=cdf)
    total_vol = float(read_scalar(cdf))
    if total_vol == 0.0:
        raise ValueError("mesh has zero volume")

    out_points = _launch.empty(count, dtype=wp.vec3, device=vertices.device)
    _launch.launch(
        kernel_sample.sample_volume_tetrahedra,
        dim=count,
        inputs=[
            vertices,
            faces,
            center,
            cdf,
            wp.float32(total_vol),
            resolve_seed(seed),
            out_points,
        ],
        device=vertices.device,
    )
    return out_points


def resolve_seed(seed: int | None) -> int:
    """
    Concrete non-negative ``int32``-range RNG seed, drawn at random when none was given.

    Every generator in the package takes ``seed: int | None`` and means the same thing by it, so
    the draw lives here rather than at each entry point. Public because
    [`random_soup`][ordito.creation.random_soup] needs the identical convention from another
    module.

    Parameters
    ----------
    seed
        User-provided seed, or ``None`` to draw a cryptographically random one.

    Returns
    -------
    int
        ``seed`` unchanged when provided, otherwise a random value in ``[0, 2**31)``.

    See Also
    --------
    [`sample_surface`][ordito.sample.sample_surface]
    [`random_soup`][ordito.creation.random_soup]
    [`voxels.resolve_voxel_grid`][ordito.voxels.resolve_voxel_grid]

    Notes
    -----
    !!! note "The ``resolve_*`` pattern"
        Both of these -- this and [`sample.resolve_seed`][ordito.sample.resolve_seed] /
        [`voxels.resolve_voxel_grid`][ordito.voxels.resolve_voxel_grid] -- turn an optional
        argument into the concrete value the wrapper would have derived, so a caller who wants two
        functions to share the derived thing can resolve it once and pass it to both. Each default
        is domain knowledge, so they cannot share a module. The optional face-adjacency pair
        deliberately has **no** such resolver: deriving it is one
        [`face_adjacency`][ordito.adjacency.face_adjacency] call with ``return_edges=True``, so
        only the *pairing rule* is worth sharing and
        [`adjacency.require_paired_adjacency`][ordito.adjacency.require_paired_adjacency] is that
        rule on its own.
    """
    if seed is None:
        return secrets.randbelow(2**31)
    return int(seed)
