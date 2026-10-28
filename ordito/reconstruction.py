"""
Building a triangle mesh whose triangulation is the method's own, not any input's.

- **From an oriented point cloud.**
  [`screened_poisson`][ordito.reconstruction.screened_poisson] fits an implicit indicator function
  and contours it, so it closes gaps and returns a watertight surface that need not pass through
  any input point; [`ball_pivoting`][ordito.reconstruction.ball_pivoting] instead *interpolates*
  the points, rolling a ball over them, and leaves a hole wherever the ball falls through.
  [`triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud] picks between them.
- **From a mesh.** [`resample_uniform`][ordito.reconstruction.resample_uniform] samples an
  existing mesh's signed distance field on a uniform grid and re-extracts it, which is the way to
  get a clean, uniformly-sampled surface out of a self-intersecting or badly-triangulated one. It
  is shelved with the reconstructors rather than with the mesh edits because it shares their
  defining property: the output's triangulation comes from the grid, not from the input, so no
  vertex, edge or face survives the call.
- **In the plane.** [`delaunay_triangulation`][ordito.reconstruction.delaunay_triangulation]
  triangulates a 2D point set.

Extracting a level set of a field you already hold is
[`levelset.marching_cubes`][ordito.levelset.marching_cubes], not a reconstruction: it lives with
its consumers in [`ordito.levelset`][ordito.levelset], and ``screened_poisson`` and
``resample_uniform`` both end in it.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, cast

import numpy as np
import warp as wp
from warp.geometry import delaunay_edge_flip

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import (
    prefers_tiled_reduction,
    read_scalar,
    read_values,
    require_same_device,
    run_device_loop,
)
from ordito.config import is_deterministic
from ordito.constants import TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import levelset as kernel_levelset
from ordito.kernels import reconstruction as kernel_reconstruction
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import remesh as kernel_remesh
from ordito.kernels import repair as kernel_repair
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels.algorithms import ball_pivoting as kernel_bpa
from ordito.kernels.algorithms import conjugate_gradient as kernel_cg
from ordito.kernels.algorithms import mesh_builder as kernel_mesh_builder
from ordito.kernels.algorithms import poisson_band as kernel_poisson_band
from ordito.kernels.algorithms import poisson_octree as kernel_octree

# Points from which the CUDA flip phase starts with Warp's own ``delaunay_edge_flip`` and lets the
# float64 flip loop below finish from its output. Warp's pass alone is about twice as fast as the
# loop from 0.5 M points but not exact (at 1 M random points it left two edges that fail the exact
# in-circle test); the loop that follows repairs them, so the answer is the loop's own, face for
# face. Whole call: 0.90x at 0.1 M points, 1.04x at 0.5 M, 1.08x at 1 M, 1.10x at 2 M (the
# single-threaded seed is most of it).
_NATIVE_DELAUNAY_FLIP_FROM = 1 << 19


def delaunay_triangulation(points: wp.array[wp.vec2], max_iter: int = 1000) -> wp.array[wp.int32]:
    """
    Delaunay triangulation of a 2D point set.

    Ports ``igl::delaunay_triangulation``: a sequential lexicographic incremental triangulation
    seeds the mesh, then interior edges are flipped until every one satisfies the empty-circumcircle
    (Delaunay) criterion. The flip loop reuses the parallel edge-flip machinery of
    [`flip_to_delaunay`][ordito.remesh.flip_to_delaunay] with a float64 in-circle predicate; ties
    (cocircular points) are left unflipped, which guarantees termination.

    Parameters
    ----------
    points
        ``(n,)`` 2D point positions on the target device.
    max_iter
        Maximum number of parallel flip passes.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` flat buffer of CCW triangles over the input point indices, on
        ``points.device``. Empty when the points are collinear.

    Raises
    ------
    ValueError
        If fewer than 3 points are given, or if a degenerate input (duplicate or collinear points)
        drives the seed past the ``2 * n`` triangle bound a triangulation obeys.

    See Also
    --------
    [`flip_to_delaunay`][ordito.remesh.flip_to_delaunay]
    [`triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud]

    Notes
    -----
    The seed triangulation is an inherently sequential hull sweep, so it runs single-threaded on the
    **CPU** device and only the flips run on ``points.device`` -- a sequential sweep is genuinely
    faster on the CPU than as a single CUDA thread. A float64 in-circle determinant is used rather
    than exact predicates, so near-cocircular inputs may resolve either ambiguous diagonal.

    Examples
    --------
    The corners of a square and its centre triangulate into four triangles, each using the centre:

    ```python
    import warp as wp

    import ordito as od

    points = wp.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=wp.vec2
    )
    faces = od.reconstruction.delaunay_triangulation(points)
    print(faces.shape[0] // 3)  # 4
    print(bool((faces.numpy().reshape(-1, 3) == 4).any(axis=1).all()))  # True
    ```
    """
    device = points.device
    n = points.size
    if n < 3:
        raise ValueError(f"delaunay_triangulation requires at least 3 points, got {n}")

    faces = _lexicographic_triangulation(points)
    if faces.size == 0:
        return faces

    def launch(
        adjacency: odt.Array2dInt32,
        adjacency_edges: odt.Array2dInt32,
        unshared: odt.Array2dInt32,
        sorted_keys: wp.array[wp.uint64],
        edge_set: wp.array[wp.uint64],
        edge_set_mask: wp.int32,
        key_base: wp.uint64,
        out_flip: wp.array[wp.bool],
        out_quad: odt.Array2dInt32,
    ) -> None:
        _launch.launch(
            kernel_remesh.incircle_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                points,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                edge_set,
                edge_set_mask,
                key_base,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    if wp.get_device(device).is_cuda and n >= _NATIVE_DELAUNAY_FLIP_FROM:
        # Warp annotates the triangles ``wp.array2d``, a static helper no runtime array is typed as.
        delaunay_edge_flip(points, cast("wp.array2d[wp.int32]", faces.reshape((-1, 3))))
    od.remesh._flip_interior_edges(faces, n, launch, max_iter)  # pyright: ignore[reportPrivateUsage]
    return faces


def _lexicographic_triangulation(points: wp.array[wp.vec2]) -> wp.array[wp.int32]:
    """
    Sequential lexicographic incremental triangulation (``igl::lexicographic_triangulation``).

    Returns the ``(3 * n_faces,)`` flat buffer of CCW triangles over the input point indices on
    ``points.device``, empty when the points are collinear. The output is not yet Delaunay — the
    caller flips it to Delaunay on device. On the CPU device it is a prefix view of the sweep's
    own buffer.

    The lex sort is a radix sort on the points' device; the sweep itself runs in
    [`lexicographic_triangulation`][ordito.kernels.reconstruction.lexicographic_triangulation]
    on the CPU device, single-threaded; see that kernel for why.
    """
    device = cast("wp.Device", points.device)
    n = points.size
    # The lex order is a stable radix sort of one packed sortable key per point, on the points'
    # own device, which is ``numpy.lexsort((y, x))`` exactly.
    keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.lexicographic_keys, dim=n, inputs=[points, keys, order], device=device
    )
    _launch.radix_sort_pairs(keys, order, count=n)
    # The sweep's inputs, on the CPU device it runs on (a no-op view when that is already theirs):
    # a device-to-host ``wp.copy`` into a fresh CPU array, with no NumPy round trip in between.
    if device.is_cuda:
        points_cpu = _launch.empty(n, dtype=wp.vec2, device="cpu")
        order_cpu = _launch.empty(n, dtype=wp.int32, device="cpu")
        _launch.copy(points_cpu, points)
        _launch.copy(order_cpu, order, count=n)
    else:
        points_cpu = points
        order_cpu = odt.as_dense(order[:n])

    # A triangulation of n points has 2n - 2 - h <= 2n - 5 triangles; 2n is the guard capacity.
    max_faces = 2 * n
    hull = _launch.empty(2 * (n + 1), dtype=wp.int32, device="cpu")
    faces_cpu = _launch.empty(3 * max_faces, dtype=wp.int32, device="cpu")
    counts = _launch.zeros(2, dtype=wp.int32, device="cpu")

    _launch.launch(
        kernel_reconstruction.lexicographic_triangulation,
        dim=1,
        inputs=[points_cpu, order_cpu, wp.int32(max_faces), hull, faces_cpu, counts],
        device="cpu",
    )

    written, wanted = (int(value) for value in counts.numpy())
    if wanted > written:
        raise ValueError(
            f"delaunay_triangulation's seed wanted {wanted} triangles for {n} points but a "
            f"triangulation admits at most {max_faces}; the input is degenerate (duplicate or "
            "collinear points)."
        )
    if written == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)
    if not device.is_cuda:
        return _launch.view(faces_cpu, 0, 3 * written)
    faces = _launch.empty(3 * written, dtype=wp.int32, device=device)
    _launch.copy(faces, faces_cpu, count=3 * written)
    return faces


def triangulate_point_cloud(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None = None,
    *,
    num_neighbours: int = 0,
    radius: float = 0.0,
    crit_angle: float = math.pi / 2.0,
    boundary_angle: float = 0.9 * math.pi,
    max_neighbours: int = 32,
    crit_hole_length: float = -1.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Reconstruct a triangle mesh from an (optionally oriented) point cloud.

    Local-triangulation reconstruction, on the GPU. Each point independently builds a local
    triangle fan over its nearest neighbours in the tangent plane, greedily
    optimised toward a Delaunay-like fan (``build_local_triangulations``). A fan whose search may
    have missed a neighbour is built again from a wider one: a fan with a border (an angular gap
    wider than ``boundary_angle``) searches twice its first radius, and any other fan as far as its
    triangles' circumcircles reach, up to that double. Triangles that recur in two or three of
    these local triangulations are kept and assembled into a triangle-soup mesh (vertices are the
    input points). Non-manifold and degenerate faces are dropped and small boundary holes are
    filled, each hole measured on its own where two touch at a vertex.

    Supplied ``normals`` are **trusted**: each face is wound to agree with them, so an
    inward-facing cloud gives an inward-facing mesh. Leave them ``None`` to estimate them by PCA
    ([`estimate_normals`][ordito.points.estimate_normals]); their signs are then ignored, and each
    connected component of the assembled mesh is wound consistently and faces away from the
    cloud's bounding-box centre at its point farthest from that centre. A closed, orientable surface
    therefore faces outward whatever its shape; an open one, which has no outside, faces the side
    that convention picks.

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device.
    normals
        ``(n,)`` oriented unit normals. When ``None``, unoriented normals are estimated and the
        result's orientation is derived from the mesh.
    num_neighbours
        Number of nearest neighbours used per point when ``radius <= 0``. Defaults to
        ``max_neighbours`` when neither ``num_neighbours`` nor ``radius`` is set.
    radius
        When positive, only neighbours within this distance are used (still capped at
        ``max_neighbours`` nearest). Mutually exclusive with ``num_neighbours``.
    crit_angle
        Maximum dihedral angle (radians) tolerated in a fan triangle before an edge is flipped.
    boundary_angle
        A point is a fan boundary when its neighbour ring has an angular gap wider than this.
    max_neighbours
        Per-point neighbour cap. Must not exceed
        ``kernels.reconstruction.MAX_NEIGHBOURS`` (the compile-time scratch size).
    crit_hole_length
        Boundary loops with perimeter at most this value are filled. When negative, defaults to
        ``0.1 *`` the point-cloud bounding-box diagonal.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(m,)`` referenced input points, compacted from index zero (unreferenced points dropped),
        on ``points.device``, ``m <= n``.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer into ``vertices``.

    Raises
    ------
    ValueError
        If both ``num_neighbours`` and ``radius`` are positive, or ``max_neighbours`` exceeds the
        compile-time cap.
    RuntimeError
        If ``points`` and ``normals`` are not all on one device.

    Notes
    -----
    The first fan of each point is bounded by ``max_neighbours`` and a rebuilt one by
    ``max(2 * k, 100)`` neighbours (``k`` the first fan's count), and the radius grows once, to at
    most twice the first: very sparse or highly non-uniform clouds may still leave extra boundary
    holes. The doubled search of a border fan also triangulates across a gap in the cloud narrower
    than about that radius, so a hole that small closes whatever ``crit_hole_length`` says.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
    [`ordito.repair.remove_degenerate_faces`][]

    Examples
    --------
    The vertices of a sphere, with their positions as normals, reconnect into one closed surface:

    ```python
    import numpy as np

    import ordito as od

    points, _ = od.creation.icosphere(subdivisions=3)
    vertices, faces = od.reconstruction.triangulate_point_cloud(points, points)
    print(od.validation.is_watertight(vertices, faces))  # True
    labels = od.adjacency.face_connected_component_labels(faces).numpy()
    print(np.unique(labels).shape[0])  # 1
    ```
    """
    require_same_device(points=points, normals=normals)
    if num_neighbours > 0 and radius > 0.0:
        raise ValueError("pass at most one of num_neighbours and radius")
    if max_neighbours > kernel_reconstruction.MAX_NEIGHBOURS:
        raise ValueError(
            "max_neighbours must be <= "
            f"{kernel_reconstruction.MAX_NEIGHBOURS}, got {max_neighbours}"
        )

    device = points.device
    n = points.size
    if n == 0:
        # Cloned, like every other exit from this module: the returned vertices are the function's
        # own buffer on the populated path (`_clean_reconstruction`), so handing
        # back the caller's array here would make the degenerate input the one case where mutating
        # the result mutates the input.
        return _launch.clone(points), _launch.empty(0, dtype=wp.int32, device=device)

    k = num_neighbours if num_neighbours > 0 else max_neighbours
    k = min(k, max_neighbours)

    # One box serves the neighbour search and the default hole length, which would otherwise each
    # reduce the cloud and read the corners back.
    bounds = od.bounds.aabb(points)
    if crit_hole_length < 0.0:
        crit_hole_length = 0.1 * od.bounds.box_diagonal(*bounds)
    # Dense (n, k+1) nearest-neighbour table; slot 0 is the point itself and is skipped in-kernel.
    neighbor_idx, neighbor_dist = od.neighbors.query_nearest(points, points, k=k + 1, bounds=bounds)

    oriented = normals is not None
    if normals is None:
        normals = od.points.estimate_normals(points, neighbor_idx, orientation="centroid")

    # Per-point local fan triangulation: point ``v``'s fan fills ``out_tris[v, :count]``. The fans,
    # their counts and the grown points' flags in one allocation.
    widths = [3 * n * k, n, n]
    tri_rows, fan_counts, grown_flags = _launch.carve(wp.int32, widths, device)
    out_tris = tri_rows.reshape((n, k, 3))
    normals_mode = (
        kernel_reconstruction.NORMALS_ORIENTED
        if oriented
        else kernel_reconstruction.NORMALS_UNORIENTED
    )
    grown_radii = _launch.empty(n, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_reconstruction.build_local_triangulations,
        dim=n,
        inputs=[
            points,
            normals,
            neighbor_idx,
            neighbor_dist,
            wp.float32(radius),
            wp.float32(crit_angle),
            wp.float32(boundary_angle),
            normals_mode,
        ],
        outputs=[out_tris, fan_counts, grown_radii, grown_flags],
        device=device,
    )
    _rebuild_grown_fans(
        points,
        normals,
        grown_radii,
        grown_flags,
        k,
        bounds,
        crit_angle,
        boundary_angle,
        normals_mode,
        out_tris,
        fan_counts,
    )
    # Scanned in place: each fan's packed offset is its predecessor's inclusive total, and the last
    # entry sizes the candidate buffers.
    _launch.array_scan(fan_counts, out_array=fan_counts, inclusive=True)
    # Readback: the candidate count sizes the sort buffers.
    n_candidates = int(read_scalar(fan_counts))
    if n_candidates == 0:
        return _launch.clone(points), _launch.empty(0, dtype=wp.int32, device=device)

    if oriented:
        return _assemble_trusted_fans(points, out_tris, fan_counts, n_candidates, crit_hole_length)
    faces = _repeated_oriented_triangles(points, out_tris, fan_counts, n_candidates)
    if faces.size == 0:
        return _launch.clone(points), faces
    # Every face carries a distinct vertex set and none is degenerate, so the duplicate-resolution
    # and degenerate-face stages have nothing to do.
    # Trusted normals keep their winding (``orient=False``): ``make_normals_outward`` would rewind a
    # whole inward-normal cloud's mesh outward, silently breaking that contract. Estimated normals
    # have no sign to keep, so the mesh is oriented instead (each fan was wound by its own center's
    # normal, and the representative faces disagree wherever those signs did): away from the box
    # centre at each component's farthest corner.
    return _clean_reconstruction(
        points,
        faces,
        crit_hole_length,
        away_from=_box_center(bounds),
        deduplicate=False,
        nondegenerate=True,
    )


def _rebuild_grown_fans(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    grown_radii: wp.array[wp.float32],
    grown_flags: wp.array[wp.int32],
    k: int,
    bounds: tuple[wp.vec3, wp.vec3],
    crit_angle: float,
    boundary_angle: float,
    normals_mode: wp.int32,
    out_tris: odt.ArrayNdInt32,
    fan_counts: wp.array[wp.int32],
) -> None:
    """
    Rebuild every fan whose radius grew, from the neighbours inside its grown radius.

    The automatic radius increase: the first fans used ``k`` neighbours, and each one that
    has a border, or whose triangles' circumcircles reach past that search, is triangulated again
    from up to ``max(2 * k, GROWN_NEIGHBOURS_MIN)`` nearest neighbours inside the radius
    ``build_local_triangulations`` gave it, which also flagged it in ``grown_flags`` (consumed).
    Its triangles replace its first fan's in its own slots.
    """
    grown_points = od.array.flatnonzero_in_place(grown_flags)
    n_grown = grown_points.size
    if n_grown == 0:
        return
    k_grown = min(max(2 * k, kernel_reconstruction.GROWN_NEIGHBOURS_MIN), points.size - 1)
    queries = od.array.gather(points, grown_points)
    # Each fan keeps only the neighbours inside its own grown radius, so the search stops at the
    # largest of them (one float above, whatever the walk's boundary test): the rows agree inside
    # every fan's radius and the slots past it stay unfilled, which the fan skips. The few grown
    # fans' walks otherwise run on to all ``k_grown`` neighbours, about 3x the query's time on
    # 8 192-point crown clouds (fewer than 80 neighbours inside any grown radius).
    largest_radius = od.reduce.max(od.array.gather(grown_radii, grown_points))
    # Few grown fans get a warp each, so their serial optimizations do not wait on one another in
    # a shared warp. Measured on the rebuild kernel (CUDA): 182 fans (a crown cloud) 6.9 ms at one
    # lane, 1.35 at 32; 12 762 fans 5.6 at one, 3.7 at 8; 99 778 fans 7.4 at one, 9.5 at two.
    lanes = _launch.walk_lanes(n_grown)
    neighbor_idx, neighbor_dist = od.neighbors.query_nearest(
        points,
        queries,
        k=k_grown + 1,
        bounds=bounds,
        max_radius=float(np.nextafter(np.float32(largest_radius), np.float32(np.inf))),
    )
    _launch.launch(
        kernel_reconstruction.rebuild_grown_local_triangulations,
        dim=n_grown * lanes,
        inputs=[
            points,
            normals,
            grown_points,
            grown_radii,
            neighbor_idx,
            neighbor_dist,
            wp.float32(crit_angle),
            wp.float32(boundary_angle),
            normals_mode,
            wp.int32(lanes),
        ],
        outputs=[out_tris, fan_counts],
        device=points.device,
    )


def _box_center(bounds: tuple[wp.vec3, wp.vec3]) -> wp.vec3:
    """Centre of a box, from its ``float32`` corners."""
    lower = np.array(bounds[0], dtype=np.float32)
    upper = np.array(bounds[1], dtype=np.float32)
    return wp.vec3(*(np.float32(0.5) * (lower + upper)))


def _assemble_trusted_fans(
    points: wp.array[wp.vec3],
    tris: odt.ArrayNdInt32,
    inclusive_counts: wp.array[wp.int32],
    n_candidates: int,
    crit_hole_length: float,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Assemble the fans of a cloud with trusted normals into a mesh.

    The candidates are the triangles two or three fans agree on in one winding, sorted
    ([`_oriented_candidate_rows`][ordito.reconstruction._oriented_candidate_rows]). They are added
    one at a time in that order, the first of any two that conflict kept; where every candidate
    around a vertex forms one fan the order changes nothing, so only the rows near a conflict go
    through the serial builder
    ([`_contested_rows_kept`][ordito.reconstruction._contested_rows_kept]). The small holes left
    are filled by the gap at a pinch, with the circumscribed-circle metric, a hole no
    triangulation closes left open.
    """
    device = points.device
    rows, n_rep3 = _oriented_candidate_rows(points, tris, inclusive_counts, n_candidates)
    if rows.shape[0] == 0:
        return _launch.clone(points), _launch.empty(0, dtype=wp.int32, device=device)
    faces = _contested_rows_kept(points, rows, n_rep3)
    vertices, faces, _ = od.repair.remove_unreferenced_vertices(points, faces)
    if crit_hole_length != 0.0 and faces.size > 0:
        faces = od.holes._fill_small(  # pyright: ignore[reportPrivateUsage]
            vertices,
            faces,
            crit_hole_length,
            n_vertices=vertices.size,
            metric="circumscribed",
            pinches="gap",
            all_or_nothing=True,
        )
    return vertices, faces


def _oriented_candidate_rows(
    points: wp.array[wp.vec3],
    tris: odt.ArrayNdInt32,
    inclusive_counts: wp.array[wp.int32],
    n_candidates: int,
) -> tuple[odt.Array2dInt32, int]:
    """
    Return the candidate triangles as rows in the order the builder adds them, and the ``t3`` count.

    A triangle three fans agree on in one winding (``t3``), else two (``t2``); a degenerate one is
    kept, as the builder rejects only a repeated vertex. Each row is stored lowest vertex
    first and the rows ascend lexicographically, the ``t3`` rows first, by two stable sorts.
    """
    device = tris.device
    n = points.size
    keys, slots = _sorted_candidate_keys(tris, inclusive_counts, n_candidates, n)
    fan_rows = tris.reshape((-1, 3))
    # The flags and the codes in the payload's upper half and a buffer of their own.
    flags = _launch.view(slots, n_candidates, 2 * n_candidates)
    codes = _launch.empty(n_candidates, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.classify_oriented_repeats,
        dim=n_candidates,
        inputs=[keys, slots, fan_rows, wp.int32(n_candidates)],
        outputs=[flags, codes],
        device=device,
    )
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    rep3 = _launch.zeros(1, dtype=wp.int32, device=device)
    # Readback: the candidate count sizes the rows, and its sort.
    m = int(read_scalar(flags))
    rows = odt.empty_2d((m, 3), wp.int32, device=device)
    if m == 0:
        return rows, 0
    low = _launch.empty(2 * m, dtype=wp.uint64, device=device)
    high = _launch.empty(2 * m, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.emit_oriented_repeats,
        dim=n_candidates,
        inputs=[flags, codes, slots, fan_rows, wp.uint64(n)],
        outputs=[rows, low, high, order, rep3],
        device=device,
    )
    # Least significant key first: the stable second sort keeps the first one's order on ties.
    _launch.radix_sort_pairs(low, order, count=m, end_bit=max(1, (n * n - 1).bit_length()))
    high_sorted = _launch.empty(2 * m, dtype=wp.uint64, device=device)
    _launch.copy(high_sorted, od.array.gather(_launch.view(high, 0, m), _launch.view(order, 0, m)))
    _launch.radix_sort_pairs(high_sorted, order, count=m, end_bit=max(1, (2 * n - 1).bit_length()))
    ordered = od.array.gather(rows, _launch.view(order, 0, m))
    # Readback: the class boundary is the builder's first-call region, read on the host.
    return odt.as_array2d(ordered, wp.int32), int(read_scalar(rep3, index=0))


# How far from a conflict the builder runs: the rows that touch a vertex within this many rings
# of one. One ring makes every vertex a deleted face touches keep its whole star in the builder,
# and each delete-and-add round can reach one ring further.
_CONTESTED_RINGS = 1 + kernel_mesh_builder.COMPLICATING_ROUNDS


def _contested_rows_kept(
    points: wp.array[wp.vec3], rows: odt.Array2dInt32, n_rep3: int
) -> wp.array[wp.int32]:
    """
    Return the candidate rows the builder keeps, flat, in their order.

    A vertex is *clean* when its rows share no directed edge and chain into one fan: the builder
    adds such rows whatever its order. Every other vertex seeds a contested region, and only the
    rows within ``_CONTESTED_RINGS`` of a seed run through the serial builder, in their global
    order (``kernels/algorithms/mesh_builder``). Its decisions near the seeds read only rings that
    are complete in the region, and the hole walk of its delete rounds continues past them along
    the clean vertices' single hole edges.

    The builder is one thread. It runs on the CPU device whatever the input's: a CUDA thread
    chasing the rings measured about 16x slower, and the contested rows are a few hundred to a
    few thousand.
    """
    device = rows.device
    m = rows.shape[0]
    n = points.size
    keys = _launch.empty(6 * m, dtype=wp.uint64, device=device)
    edge_rows = _launch.empty(6 * m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.directed_edge_keys,
        dim=(m, 3),
        inputs=[rows, wp.uint64(n)],
        outputs=[keys, edge_rows],
        device=device,
    )
    _launch.radix_sort_pairs(keys, edge_rows, count=3 * m, end_bit=max(1, (n * n - 1).bit_length()))
    conflict = _launch.zeros(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.mark_shared_directed_edges,
        dim=3 * m,
        inputs=[keys, edge_rows, wp.int32(3 * m)],
        outputs=[conflict],
        device=device,
    )
    flat_rows = cast(wp.array[wp.int32], rows.flatten())
    vertex_rows, offsets = od.adjacency.vertex_face_adjacency(flat_rows, n_vertices=n)
    distance = _launch.empty(n, dtype=wp.int32, device=device)
    hole_next = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.classify_candidate_stars,
        dim=n,
        inputs=[rows, vertex_rows, offsets, conflict],
        outputs=[distance, hole_next],
        device=device,
    )
    # One ring past the region, so its outermost vertices know their stars are not complete.
    for ring in range(1, _CONTESTED_RINGS + 2):
        _launch.launch(
            kernel_reconstruction.grow_seed_rings,
            dim=m,
            inputs=[rows, wp.int32(ring), distance],
            device=device,
        )
    row_flags = _launch.empty(m, dtype=wp.int32, device=device)
    vertex_flags = _launch.zeros(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.mark_contested_rows,
        dim=m,
        inputs=[rows, distance, wp.int32(_CONTESTED_RINGS)],
        outputs=[row_flags, vertex_flags],
        device=device,
    )
    contested = od.array.flatnonzero_in_place(row_flags)
    if contested.size == 0:
        return flat_rows
    used = od.array.flatnonzero_in_place(vertex_flags)
    global_to_local = _launch.full(n, -1, dtype=wp.int32, device=device)
    complete = _launch.empty(used.size, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.localize_contested_vertices,
        dim=used.size,
        inputs=[used, distance, wp.int32(_CONTESTED_RINGS)],
        outputs=[global_to_local, complete],
        device=device,
    )
    local_rows = odt.empty_2d((contested.size, 3), wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.localize_rows,
        dim=contested.size,
        inputs=[contested, rows, global_to_local],
        outputs=[local_rows],
        device=device,
    )
    # Readback: the region's rows below the class boundary are the builder's first call; the
    # contested rows ascend, so it is one search of them on the host.
    contested_np = contested.numpy()
    n_rep3_local = int(np.searchsorted(contested_np, n_rep3))
    state = _run_mesh_builder(
        od.array.gather(points, used),
        local_rows,
        n_rep3_local,
        complete,
        used,
        global_to_local,
        hole_next,
    )
    keep = _launch.ones(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.apply_builder_state,
        dim=contested.size,
        inputs=[contested, _launch.clone(state, device=device), kernel_mesh_builder.FACE_ADDED],
        outputs=[keep],
        device=device,
    )
    kept = od.array.flatnonzero_in_place(keep)
    return cast(wp.array[wp.int32], od.array.gather(rows, kept).flatten())


def _run_mesh_builder(
    points: wp.array[wp.vec3],
    rows: odt.Array2dInt32,
    n_rep3: int,
    complete: wp.array[wp.int32],
    local_to_global: wp.array[wp.int32],
    global_to_local: wp.array[wp.int32],
    hole_next: wp.array[wp.int32],
) -> wp.array[wp.int32]:
    """Run ``mesh_builder.build_candidate_mesh`` on the CPU device; return each row's state."""
    host = "cpu"
    n_rows = rows.shape[0]
    capacity = 6 * n_rows
    topology = kernel_mesh_builder.HalfEdges()
    records = _launch.empty(4 * capacity, dtype=wp.int32, device=host)
    topology.next = _launch.view(records, 0, capacity)
    topology.prev = _launch.view(records, capacity, 2 * capacity)
    topology.org = _launch.view(records, 2 * capacity, 3 * capacity)
    topology.left = _launch.view(records, 3 * capacity, 4 * capacity)
    topology.edge_per_vertex = _launch.empty(points.size, dtype=wp.int32, device=host)
    topology.edge_per_face = _launch.empty(n_rows, dtype=wp.int32, device=host)
    topology.count = _launch.empty(1, dtype=wp.int32, device=host)
    state = _launch.empty(n_rows, dtype=wp.int32, device=host)
    _launch.launch(
        kernel_mesh_builder.build_candidate_mesh,
        dim=1,
        inputs=[
            _launch.clone(points, device=host),
            _launch.clone(rows, device=host),
            wp.int32(n_rep3),
            _launch.clone(complete, device=host),
            _launch.clone(local_to_global, device=host),
            _launch.clone(global_to_local, device=host),
            _launch.clone(hole_next, device=host),
            topology,
            _launch.empty(n_rows, dtype=wp.int32, device=host),
            _launch.empty(n_rows, dtype=wp.int32, device=host),
            _launch.empty(capacity // 2, dtype=wp.int32, device=host),
        ],
        outputs=[state],
        device=host,
    )
    return state


def _repeated_oriented_triangles(
    points: wp.array[wp.vec3],
    tris: odt.ArrayNdInt32,
    inclusive_counts: wp.array[wp.int32],
    n_candidates: int,
) -> wp.array[wp.int32]:
    """
    Keep one oriented representative per candidate triangle two or three fans agree on.

    The candidates are the ``(n, k)`` fan slots below each point's count, which
    ``inclusive_counts`` holds as an inclusive scan. Each is keyed by its vertex set -- the three
    indices sorted and packed in radix ``n``, the point count -- and the keys are sorted stably
    with their slot as payload. A run of two or three equal keys is a confirmed triangle, and its
    first -- lowest -- slot supplies the winding: this is the trusted-normal case, where the
    orientation is taken from the fans rather than propagated.

    The faces come out in ascending key order, which is the order the duplicate-resolution stage
    of [`_clean_reconstruction`][ordito.reconstruction._clean_reconstruction] would emit them in:
    no two carry the same vertex set, so that stage keeps every one and needs no pass. A degenerate
    representative is dropped here, as that function's degenerate-face stage would drop it.
    """
    device = tris.device
    n = points.size
    keys, slots = _sorted_candidate_keys(tris, inclusive_counts, n_candidates, n)
    rows = tris.reshape((-1, 3))
    # The run flags live in the payload's upper half, the sort's scratch and free once it ran.
    flags = _launch.view(slots, n_candidates, 2 * n_candidates)
    _launch.launch(
        kernel_reconstruction.repeated_triangle_flags,
        dim=n_candidates,
        inputs=[points, keys, slots, rows, wp.int32(n_candidates), flags],
        device=device,
    )
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    # Readback: the confirmed-triangle count sizes the face buffer.
    n_faces = int(read_scalar(flags))
    faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    if n_faces > 0:
        _launch.launch(
            kernel_reconstruction.emit_repeated_triangles,
            dim=n_candidates,
            inputs=[flags, slots, rows, faces],
            device=device,
        )
    return faces


def _sorted_candidate_keys(
    tris: odt.ArrayNdInt32, inclusive_counts: wp.array[wp.int32], n_candidates: int, n: int
) -> tuple[wp.array[wp.uint64], wp.array[wp.int32]]:
    """
    Sort the fan slots by their triangle's unoriented key, the slot as payload.

    Both candidate selections group the fans' triangles by vertex set this way: the plain one of
    the unoriented clouds and the oriented classes of the trusted ones. The buffers are
    double-width, the radix sort's scratch in their upper halves, which the callers reuse.
    """
    device = tris.device
    keys = _launch.empty(2 * n_candidates, dtype=wp.uint64, device=device)
    slots = _launch.empty(2 * n_candidates, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.candidate_triangle_keys,
        dim=(int(tris.shape[0]), int(tris.shape[1])),
        inputs=[tris, inclusive_counts, wp.uint64(n), keys, slots],
        device=device,
    )
    # Three indices below ``n`` pack below ``n ** 3``, so the sort orders only those low bits.
    _launch.radix_sort_pairs(
        keys, slots, count=n_candidates, end_bit=min(64, max(1, (n**3 - 1).bit_length()))
    )
    return keys, slots


def screened_poisson(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    *,
    depth: int = 8,
    full_depth: int = 5,
    scale: float = 1.1,
    point_weight: float | None = None,
    solver_iterations: int | None = None,
    solver_tolerance: float = 1e-5,
    confidence: bool = False,
    method: Literal["dense", "adaptive", "octree"] = "dense",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Screened-Poisson surface reconstruction from an oriented point cloud (Kazhdan PoissonRecon).

    GPU-native reimplementation of the screened-Poisson filter (PyMeshLab's
    ``generate_surface_reconstruction_screened_poisson``, ``open3d``'s
    ``create_from_point_cloud_poisson``). A dense node-centered grid over a padded bounding cube is
    used instead of PoissonRecon's octree: the oriented normals are trilinearly splatted into a
    smoothed vector field ``V`` (with a density weight ``W``), the indicator function ``x`` is
    recovered by solving the screened-Poisson system ``(L_N + point_weight * W) x = -div V``
    matrix-free with a multigrid-preconditioned conjugate gradient, and the iso-surface is
    extracted with [`levelset.marching_cubes`][ordito.levelset.marching_cubes] at the iso-value
    given by the average of ``x`` over the input points. A **cascadic** coarse-to-fine schedule
    solves the system from ``full_depth`` up to ``depth``, prolonging each level's solution as the
    next level's initial guess (mirroring PoissonRecon's multigrid hierarchy); ``method`` describes
    the adaptive backend's variant of the schedule, and ``method="octree"`` instead reproduces
    PoissonRecon's own system and solver, the surface ``open3d`` returns.

    ``normals`` are **required** and must be globally consistently oriented (all pointing outward or
    all inward): the reconstruction encodes the surface orientation in the sign of the vector field,
    and [`estimate_normals`][ordito.points.estimate_normals] performs no global orientation.

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device (``n >= 3``).
    normals
        ``(n,)`` oriented unit normals (their magnitude is used as a confidence weight when
        ``confidence`` is set).
    depth
        Finest grid depth: the finest grid has ``2**depth + 1`` nodes per axis. Memory grows as the
        cube of this; ``depth=8`` (a ``257**3`` grid) is a safe default, and a ``depth`` above
        ``10`` raises. This is the single biggest lever on both cost and quality — see Notes before
        lowering it.
    full_depth
        Coarsest depth of the cascade (``3 <= full_depth <= depth``). The dense backend solves the
        system at every depth from ``full_depth`` to ``depth``; the adaptive one solves its dense
        base at ``max(full_depth, depth - 2)`` and a band level at every depth above it.
    scale
        Ratio between the reconstruction cube's side and the point cloud's largest bounding-box
        extent (``>= 1``, so the cube always contains the cloud); ``1.1`` pads the cloud by 10 %.
    point_weight
        Screening weight ``alpha`` tying the iso-surface to the input samples. ``0`` recovers the
        unscreened Poisson reconstruction (a tiny epsilon is still added to keep the operator SPD).
        ``None`` is ``4`` for ``"dense"`` and ``"adaptive"``, and ``2`` for ``"octree"``, which
        multiplies it by the samples' mean area weight as PoissonRecon does.
    solver_iterations
        Maximum conjugate-gradient iterations per dense level (every level of the dense backend,
        the adaptive backend's dense base). The default leaves room: the multigrid-preconditioned
        solve reaches the default tolerance in a dozen or so at any depth. The adaptive backend's
        band levels instead run a fixed, smaller number of Jacobi-preconditioned rounds, capped by
        this value: each only corrects its coarser level's solution near the surface. ``None`` is
        ``100``, and for ``"octree"`` ``8``: the Gauss-Seidel sweeps of each depth.
    solver_tolerance
        Relative residual tolerance for the conjugate-gradient solves (not read by ``"octree"``).
        A dense level that stops at ``solver_iterations`` above it warns; an adaptive band level
        stops at it or at its fixed round count and never warns. The system is solved in
        ``float32``, whose rounding bounds the relative residual either backend can actually reach
        at a few ``1e-6`` -- a tolerance below that is never met and only spends the whole
        iteration budget.
    confidence
        When ``True``, weight each sample's splat by ``|normals[i]|`` (treating the normal magnitude
        as a per-sample confidence), matching PoissonRecon's ``confidence`` flag. Not available
        with ``"octree"``, which, like ``open3d``, uses unit normals.
    method
        Solver backend. ``"dense"`` (default) solves on the whole node lattice described above.
        ``"adaptive"`` solves the same lattice's system stored only near the surface: below
        ``max(full_depth, depth - 2)`` the lattice is dense, and every finer level keeps only the
        ``8**3``-node bricks around the samples and around the coarser level's surface, solving for
        that band with the lattice away from it held at the coarser solution (every coarser level's
        data is the finest level's, restricted, so that solution is a valid boundary value). Its
        surface agrees with ``dense`` to a small fraction of a cell, its cost follows the surface
        rather than the volume, and it runs at ``depth=10``, whose dense lattice does not fit a
        32 GB device. With ``depth == full_depth`` there is no band and it is the dense solve.
        ``"octree"`` solves PoissonRecon's system the way ``open3d``'s
        ``create_from_point_cloud_poisson`` does, at the same parameters: hierarchical trilinear
        elements of every depth, active only near the samples above ``full_depth``, a normal field
        splatted at each sample's estimated depth, screening scaled by the samples' mean area, and
        one cascadic pass of ``solver_iterations`` Gauss-Seidel sweeps per depth -- not a converged
        solve, and the surface follows that. Its mesh is ``open3d``'s, run on one thread: the same
        vertices and triangles, stitched across changes of depth as ``open3d`` stitches them,
        except where ``open3d``'s minimal-area split of a polygon is decided by rounding (a few
        percent of the triangles; the surface is the same). Those triangles are returned as is:
        not filtered for degeneracy nor reoriented. ``depth`` is at most 9.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` iso-surface vertices on ``points.device``.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer. With ``"dense"`` and ``"adaptive"`` it is
        free of degenerate triangles in
        [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask]'s sense and oriented
        outward ([`make_normals_outward`][ordito.repair.make_normals_outward]); with ``"octree"``
        it is ``open3d``'s winding. Every extraction keeps every vertex a small fraction of a
        lattice edge off the lattice nodes, which bounds every marching-cubes triangle's altitudes
        away from zero without changing the surface's topology, so the guarantee costs no hole; a
        lattice so fine that its spacing nears the merge tolerance falls back to dropping the faces
        ([`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]).

    Raises
    ------
    ValueError
        If ``points`` has fewer than 3 points, ``normals`` does not have one entry per point, the
        depth/scale parameters are out of range, ``method`` is not ``"dense"``, ``"adaptive"`` or
        ``"octree"``, or with ``"octree"`` ``depth`` is above 9, ``confidence`` is set or no normal
        is nonzero.
    RuntimeError
        If ``points`` and ``normals`` are not all on one device.

    Warns
    -----
    UserWarning
        When a dense level's conjugate-gradient solve (any level of the dense backend, the adaptive
        backend's dense base) stops at ``solver_iterations`` with its residual still above
        ``solver_tolerance``: the surface is then extracted from an unconverged field. The
        adaptive backend's band levels are not checked (see ``solver_iterations``).
    UserWarning
        With ``method="adaptive"``, when the extraction meets crossing cells its narrow band cannot
        hold (as a degenerate, for instance coplanar, cloud does): those cells are dropped and the
        surface has holes.

    See Also
    --------
    [`triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud]
    [`ball_pivoting`][ordito.reconstruction.ball_pivoting]
    [`estimate_normals`][ordito.points.estimate_normals]

    Notes
    -----
    The whole system is solved in index space (grid spacing ``1``); the world scale is reapplied by
    marching cubes via the cube bounds, so the extracted level set is invariant to that scale. The
    per-node density normalization ``V /= max(W, eps)`` is cruder than PoissonRecon's kernel-density
    estimate and can bulge on strongly anisotropic clouds. The screening weight ``point_weight`` is
    not area-normalized as in PoissonRecon, so its numeric scale differs; ``4`` is a reasonable
    default for unit-scale clouds.

    Under [`is_deterministic`][ordito.config.is_deterministic] the splat and the adaptive
    backend's restriction add their terms in one fixed order, so repeated calls return identical
    buffers on CUDA; otherwise they add in arrival order and the surface can move in its last bits
    between calls.

    **Choosing ``depth``.** The useful depth follows the cloud's *sampling density*, not its size,
    and past that point extra depth costs time without buying fidelity -- once the grid outruns the
    samples, accuracy can get worse rather than better. The default of ``8`` matches PoissonRecon's
    own; for a known cloud density it is worth lowering. The ``adaptive`` backend makes a finer
    lattice affordable -- its cost follows the surface -- but not more faithful than the sampling
    supports.

    **``point_weight=0`` is ill-conditioned.** Screening is what conditions the operator; at ``0``
    only the ``1e-4`` floor above keeps it SPD, so the conjugate gradient stops on a solution whose
    level set is genuinely uncertain and can move between runs on the same input. So ``0`` is for
    comparing *against* a screened reconstruction, not for producing one -- and do not pin a count
    taken from it.

    Examples
    --------
    A closed surface from the oriented vertices of a sphere, on a coarse lattice:

    ```python
    import numpy as np

    import ordito as od

    points, _ = od.creation.icosphere(subdivisions=3)
    vertices, faces = od.reconstruction.screened_poisson(points, points, depth=5, full_depth=4)
    print(od.validation.is_watertight(vertices, faces))  # True
    labels = od.adjacency.face_connected_component_labels(faces).numpy()
    print(np.unique(labels).shape[0])  # 1
    ```
    """
    require_same_device(points=points, normals=normals)
    if not (3 <= full_depth <= depth <= 10):
        raise ValueError(
            "screened_poisson requires 3 <= full_depth <= depth <= 10, got "
            f"full_depth={full_depth}, depth={depth}."
        )
    if scale < 1.0:
        # A cube smaller than the cloud's own bounding box leaves points outside
        # ``[cube_lower, cube_upper]``; the grid samplers clamp their cell index into range rather
        # than raising, so those points would silently splat onto the boundary node instead of
        # being rejected or the cube being grown.
        raise ValueError(f"screened_poisson requires scale >= 1, got {scale}.")
    if method not in ("dense", "adaptive", "octree"):
        raise ValueError(
            f"screened_poisson method must be 'dense', 'adaptive' or 'octree', got {method!r}."
        )
    if method == "octree" and depth > _OCTREE_MAX_DEPTH:
        raise ValueError(
            f"screened_poisson with method='octree' requires depth <= {_OCTREE_MAX_DEPTH}, got "
            f"{depth}."
        )
    if method == "octree" and confidence:
        raise ValueError("screened_poisson with method='octree' does not take confidence=True.")
    if point_weight is None:
        point_weight = 2.0 if method == "octree" else 4.0
    if solver_iterations is None:
        solver_iterations = 8 if method == "octree" else 100

    device = points.device
    n = points.size
    if n < 3:
        raise ValueError(f"screened_poisson requires at least 3 points, got {n}.")
    if normals.size != n:
        # Every splat and sampling kernel reads ``normals[i]`` for each point ``i``.
        raise ValueError(
            f"screened_poisson requires one normal per point, got {normals.size} normals for "
            f"{n} points."
        )

    bounds = od.bounds.aabb(points)
    cube_lower, cube_upper, cube_size = _poisson_cube(points, scale, bounds=bounds)
    # Effective screening weight: a floor keeps the operator SPD even at point_weight == 0.
    screen = max(float(point_weight), 1e-4)

    if method == "octree":
        return _screened_poisson_octree(
            points,
            normals,
            bounds,
            scale=scale,
            depth=depth,
            full_depth=full_depth,
            point_weight=float(point_weight),
            iterations=solver_iterations,
        )
    if method == "adaptive" and full_depth < depth:
        vertices, faces = _screened_poisson_band(
            points,
            normals,
            cube_lower,
            cube_size,
            depth=depth,
            full_depth=full_depth,
            screen=screen,
            solver_iterations=solver_iterations,
            solver_tolerance=solver_tolerance,
            confidence=confidence,
        )
    else:
        solution, res = _poisson_dense_solve(
            points,
            normals,
            cube_lower,
            cube_size,
            depth,
            full_depth,
            screen,
            confidence,
            solver_iterations,
            solver_tolerance,
            device,
        )
        iso = _field_iso_value(
            solution, res, cube_lower, float(res - 1) / cube_size, points, normals, confidence
        )
        vertices, faces = od.levelset.marching_cubes(
            odt.as_array3d(solution.reshape((res, res, res)), wp.float32),
            iso,
            bounds=(cube_lower, cube_upper),
            edge_margin=kernel_levelset.lattice_edge_margin(cube_size / float(res - 1)),
        )

    # Both extractions keep every vertex ``kernel_levelset.lattice_edge_margin`` of an edge off the
    # lattice nodes, which is what makes the surface free of degenerate faces: without it a vertex
    # lands within rounding of a node wherever the level set grazes one, and the slivers around it
    # fall under ``face_nondegenerate_mask``'s absolute tolerance -- dozens a call at depth 9, and
    # removing
    # them opened the closed surface, since their neighbours do not share their vertices. This
    # drop is then a guard for a lattice so fine its spacing nears that tolerance. It is
    # deliberately *not* the full ``_clean_reconstruction`` tail the other three reconstructions
    # use: welding coincident vertices manufactures non-manifold edges.
    vertices, faces = od.repair.remove_degenerate_faces(vertices, faces)
    if faces.size > 0:
        faces = od.repair.make_normals_outward(vertices, faces)
    return vertices, faces


# ``method="octree"`` keeps every depth's node lattice dense: depth 9 is a few GB of lattices,
# depth 10 eight times that.
_OCTREE_MAX_DEPTH = 9


def _screened_poisson_octree(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    bounds: tuple[wp.vec3, wp.vec3],
    *,
    scale: float,
    depth: int,
    full_depth: int,
    point_weight: float,
    iterations: int,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Back [`screened_poisson`][ordito.reconstruction.screened_poisson]'s ``method="octree"``.

    ``kernels/algorithms/poisson_octree`` has the system and the schedule: merge the points per
    finest cell, estimate their density and depth, splat the normals, mark the active nodes, form
    every depth's right-hand side, run the cascade, take the iso-value at the samples and extract
    each depth's leaf cells.
    """
    device = points.device
    n = points.size
    kernel_depth = depth - 2
    cell_offsets_host = [sum(8**e for e in range(d)) for d in range(depth + 2)]
    flag_offsets_host = [sum((2**e + 2) ** 3 for e in range(d)) for d in range(depth + 1)]
    node_offsets_host = [sum((2**e + 1) ** 3 for e in range(d)) for d in range(depth + 2)]
    edge_offsets_host = [3 * offset for offset in node_offsets_host]
    # The four offset tables in one upload.
    tables = [cell_offsets_host, flag_offsets_host, node_offsets_host, edge_offsets_host]
    packed = _launch.array(np.concatenate(tables).astype(np.int32), wp.int32, device=device)
    starts = np.cumsum([0] + [len(table) for table in tables]).tolist()
    cell_offsets, flag_offsets, node_offsets, edge_offsets = (
        _launch.view(packed, starts[t], starts[t + 1]) for t in range(4)
    )

    unit_transform = _octree_unit_transform(bounds, scale)
    keys = _launch.empty(n, dtype=wp.int32, device=device)
    positions, unit_normals = _launch.carve(wp.vec3, [n, n], device)
    _launch.launch(
        kernel_octree.octree_point_keys,
        dim=n,
        inputs=[points, normals, *unit_transform, depth],
        outputs=[keys, positions, unit_normals],
        device=device,
    )
    sentinel = 8**depth
    sample_keys, inverse, sample_counts = od.grouping.unique_1d(
        keys, return_inverse=True, return_counts=True, max_value=sentinel
    )
    n_groups = sample_keys.size
    # The last group holds the points with no usable normal, when there are any.
    n_samples = n_groups
    n_merged = n
    if int(read_scalar(sample_keys)) == sentinel:
        n_samples -= 1
        # The dropped points' count, read only when there are any: the iso-value's divisor.
        n_merged -= int(read_scalar(sample_counts))
    if n_samples == 0:
        raise ValueError("screened_poisson requires at least one point with a nonzero normal.")
    sample_positions = _octree_group_sums(positions, inverse, n_groups)
    sample_normals = _octree_group_sums(unit_normals, inverse, n_groups)
    slots = _launch.full(8**depth, -1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_octree.octree_sample_slots,
        dim=n_samples,
        inputs=[sample_keys],
        outputs=[slots],
        device=device,
    )
    sample_args = [slots, sample_counts, sample_positions]

    # Every coarser cell's point count and position sum, finest first.
    counts = _launch.empty(cell_offsets_host[depth], dtype=wp.float32, device=device)
    position_sums = _launch.empty(cell_offsets_host[depth], dtype=wp.vec3, device=device)
    for d in reversed(range(depth)):
        _launch.launch(
            kernel_octree.octree_child_aggregates,
            dim=8**d,
            inputs=[d, depth, *sample_args, cell_offsets, counts, position_sums],
            device=device,
        )

    # The density, at depths 0 ..= kernel_depth and the aliased kernel_depth + 1 (every term is one
    # child's: at most 6**3 of a depth and as many aliased from the one below).
    density_sums = _launch.group_sums(
        cell_offsets_host[kernel_depth + 2], wp.float32, 2 * 6**3, device
    )
    _launch.launch_group_sums(
        kernel_octree.octree_density_splat,
        dim=cell_offsets_host[kernel_depth + 2] - cell_offsets_host[1],
        inputs=[counts, position_sums, cell_offsets, kernel_depth],
        sums=density_sums,
        device=device,
    )
    density = _launch.finish_group_sums(density_sums)

    # The depth estimates and both weight terms in one allocation; the area terms' slots take the
    # iso-value's terms once the weights are summed.
    sample_depths, area_terms, count_terms = _launch.carve(wp.float32, [n_samples] * 3, device)
    _launch.launch(
        kernel_octree.octree_sample_weights,
        dim=n_samples,
        inputs=[
            sample_counts,
            sample_positions,
            sample_normals,
            density,
            cell_offsets,
            kernel_depth,
        ],
        outputs=[sample_depths, area_terms, count_terms],
        device=device,
    )
    # The mean area weight scales the screening, so the host needs it: both sums in one read.
    weight_sums = _launch.zeros(2, dtype=wp.float32, device=device)
    for slot, terms in enumerate((area_terms, count_terms)):
        od.reduce.sum_of_terms(
            kernel_reduce.ARRAY_SUM,
            [terms],
            n_samples,
            out=_launch.view(weight_sums, slot, slot + 1),
            device=device,
        )
    area_sum, count_sum = read_values(weight_sums, 0, 2)
    alpha = point_weight * area_sum / count_sum
    # The density is read only by the weights: released before the normal splat allocates.
    del density_sums, density

    normal_sums = _launch.group_sums(cell_offsets_host[depth + 1], wp.vec3, 2 * n_samples, device)
    _launch.launch_group_sums(
        kernel_octree.octree_normal_splat,
        dim=n_samples,
        inputs=[
            sample_keys,
            sample_counts,
            sample_positions,
            sample_normals,
            sample_depths,
            cell_offsets,
            depth,
        ],
        sums=normal_sums,
        device=device,
    )
    normal_field = _launch.finish_group_sums(normal_sums)

    flags = _launch.empty(flag_offsets_host[depth], dtype=wp.int32, device=device)
    for d in reversed(range(depth)):
        _launch.launch(
            kernel_octree.octree_clip_flags,
            dim=(2**d + 2) ** 3,
            inputs=[d, depth, full_depth, normal_field, cell_offsets, flag_offsets, flags],
            device=device,
        )
    for d in range(depth - 1, full_depth, -1):
        _launch.launch(
            kernel_octree.octree_dilate_flags,
            dim=(2 ** (d - 1) + 2) ** 3,
            inputs=[d, flag_offsets, flags],
            device=device,
        )

    system = [depth, full_depth, wp.float32(alpha)]
    screen_args = [flags, flag_offsets, counts, position_sums, cell_offsets, *sample_args]
    constraints, base = _octree_constraints(
        normal_field,
        cell_offsets,
        cell_offsets_host,
        node_offsets,
        node_offsets_host,
        depth,
        system,
        screen_args,
    )
    # The normal field is read only by the flags and the constraints: released before the solve.
    del normal_sums, normal_field
    # Every screened cell's weight and corner hat values, formed once for every sweep that reads
    # them. A coarser cell with samples holds at least one, so its depth has at most
    # ``n_samples`` such cells.
    n_coarse = sum(min(8**d, n_samples) for d in range(depth))
    screen = _launch.empty(
        (n_samples + n_coarse) * kernel_octree.SCREEN_RECORD, dtype=wp.float32, device=device
    )
    screen_index = _launch.empty(cell_offsets_host[depth], dtype=wp.int32, device=device)
    screen_count = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_octree.octree_screen_records,
        dim=cell_offsets_host[depth + 1],
        inputs=[*system, n_samples, *screen_args],
        outputs=[screen_index, screen_count, screen],
        device=device,
    )
    sweep_args = [flags, flag_offsets, cell_offsets, slots, screen_index, screen]
    prolongeds, fields = _octree_cascade(
        constraints, base, node_offsets_host, depth, full_depth, iterations, alpha, sweep_args
    )
    # The solve's inputs are dead: released before the extraction allocates, which is where the
    # call's memory would otherwise peak.
    del sweep_args, screen_args, sample_args, constraints, slots, counts, position_sums
    del screen, screen_index, screen_count

    res = 2**depth + 1
    iso_terms = area_terms
    _launch.launch(
        kernel_octree.octree_iso_terms,
        dim=n_samples,
        inputs=[
            sample_counts,
            sample_positions,
            _launch.view(fields, node_offsets_host[depth], node_offsets_host[depth + 1]),
            res,
        ],
        outputs=[iso_terms],
        device=device,
    )
    # The extraction takes the iso-value as a host scalar.
    iso = od.reduce.sum(iso_terms) / n_merged
    unit_vertices, faces = _octree_extract(
        prolongeds,
        fields,
        node_offsets,
        node_offsets_host,
        edge_offsets,
        edge_offsets_host,
        iso,
        depth,
        full_depth,
        flags,
        flag_offsets,
        cell_offsets,
        cell_offsets_host,
    )
    # The unit cube back to the cloud's frame, as the upstream's inverse transform.
    inverse, translation = unit_transform
    scale = 1.0 / float(inverse)
    shift = [-float(c) * scale for c in odt.vec3_floats(translation)]
    to_world = wp.mat44(
        scale,
        0.0,
        0.0,
        shift[0],
        0.0,
        scale,
        0.0,
        shift[1],
        0.0,
        0.0,
        scale,
        shift[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return od.transform.transform_points(unit_vertices, to_world, out=unit_vertices), faces


def _octree_group_sums(
    values: wp.array[wp.vec3], groups: wp.array[wp.int32], n_groups: int
) -> wp.array[wp.vec3]:
    """
    [`scatter_sum`][ordito.reduce.scatter_sum] of per-point vectors into their merged samples.

    Without its validation: the groups are ``unique_1d``'s inverse, so the index range needs no
    read.
    """
    device = values.device
    sums = _launch.group_sums(n_groups, wp.vec3, values.size, device)
    _launch.launch_group_sums(
        kernel_scatter.SCATTER_SUM_VECTORS[wp.vec3],
        values.size,
        [values, groups],
        sums,
        device=device,
    )
    return _launch.finish_group_sums(sums)


def _octree_unit_transform(
    bounds: tuple[wp.vec3, wp.vec3], scale: float
) -> tuple[wp.float32, wp.vec3]:
    """
    ``(1 / side, -corner / side)``, the unit-cube map in the upstream's ``float32`` arithmetic.

    The cube's side and corner rounded the way PoissonRecon forms them (centre and extent of the
    ``float32`` bounding box, ``1 / side`` taken in ``float64`` and rounded). The clouds this
    backend is matched against often sit on a lattice whose planes the octree's cell faces land on,
    where a last-bit difference moves a point into the next cell. ``bounds`` is the cloud's
    [`aabb`][ordito.bounds.aabb].
    """
    lo, hi = bounds
    lower = np.asarray(odt.vec3_floats(lo), dtype=np.float32)
    upper = np.asarray(odt.vec3_floats(hi), dtype=np.float32)
    center = (upper + lower) / np.float32(2.0)
    side = np.float32((upper - lower).max()) * np.float32(scale)
    corner = center - side / np.float32(2.0)
    inverse = np.float32(1.0 / float(side))
    return wp.float32(float(inverse)), wp.vec3(*(-corner * inverse).tolist())


def _octree_constraints(
    normal_field: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    cell_offsets_host: list[int],
    node_offsets: wp.array[wp.int32],
    node_offsets_host: list[int],
    depth: int,
    system: Sequence[object],
    screen_args: Sequence[object],
) -> tuple[wp.array[wp.float32], list[float]]:
    """
    Every depth's right-hand side: the divergence of the multi-depth normal field, then screening.

    Own and finer data first, restricted down depth by depth; coarser data enters only through the
    parent depth's coefficients carried up (``octree_upsample_coarse_data``), after the
    restriction so it is not restricted twice. Returns the right-hand sides packed at the node
    offsets, and depth 0's eight entries followed by its one cell's point count and position sum.
    """
    device = normal_field.device
    n_nodes = node_offsets_host[depth + 1]
    constraints = _launch.empty(n_nodes, dtype=wp.float32, device=device)
    # Every depth's 1-D hat-against-B-spline integrals, which both divergences read.
    n_integrals = 8 * (2 ** (depth + 1) - 1 + depth + 1)
    integrals = _launch.empty(n_integrals, dtype=wp.vec2d, device=device)
    _launch.launch(
        kernel_octree.octree_axis_integrals,
        dim=n_integrals,
        inputs=[depth],
        outputs=[integrals],
        device=device,
    )
    _launch.launch(
        kernel_octree.octree_own_constraints,
        dim=n_nodes,
        inputs=[depth, normal_field, cell_offsets, node_offsets, integrals],
        outputs=[constraints],
        device=device,
    )
    for d in reversed(range(depth)):
        res_c = 2**d + 1
        _launch.launch(
            kernel_octree.octree_restrict_add,
            dim=(res_c, res_c, res_c),
            inputs=[
                _launch.view(constraints, node_offsets_host[d + 1], node_offsets_host[d + 2]),
                2 ** (d + 1) + 1,
                res_c,
            ],
            outputs=[_launch.view(constraints, node_offsets_host[d], node_offsets_host[d + 1])],
            device=device,
        )
    accumulated = _launch.empty(cell_offsets_host[depth], dtype=wp.vec3, device=device)
    for d in range(depth):
        _launch.launch(
            kernel_octree.octree_upsample_coarse_data,
            dim=8**d,
            inputs=[d, normal_field, cell_offsets],
            outputs=[accumulated],
            device=device,
        )
    base = _launch.empty(12, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_octree.octree_coarse_screening_constraints,
        dim=n_nodes,
        inputs=[*system, accumulated, *screen_args, node_offsets, integrals],
        outputs=[constraints, base],
        device=device,
    )
    # The host solves depth 0.
    return constraints, read_values(base, 0, 12)


def _octree_cascade(
    constraints: wp.array[wp.float32],
    base: list[float],
    node_offsets: list[int],
    depth: int,
    full_depth: int,
    iterations: int,
    alpha: float,
    sweep_args: Sequence[object],
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Run the cascade: depth 0 on the host, then ``iterations`` wavefront Gauss-Seidel sweeps a depth.

    ``constraints`` holds every depth's right-hand side at ``node_offsets``, and ``base`` depth 0's
    eight entries, its cell's point count and position sum. ``sweep_args`` are the sweeps' tables:
    the activity flags, the cell offsets, the sample slots and the screening records. Returns
    every depth's prolonged coarser field and its own field, packed at the same offsets (a field
    minus its prolonged field is the depth's correction, which the extraction's slopes read).
    """
    device = constraints.device
    fields = _launch.empty(node_offsets[-1], dtype=wp.float32, device=device)
    prolongeds = _launch.zeros(node_offsets[-1], dtype=wp.float32, device=device)

    def level(buffer: wp.array[wp.float32], d: int) -> wp.array[wp.float32]:
        return _launch.view(buffer, node_offsets[d], node_offsets[d + 1])

    # Depth 0's eight unknowns: its right-hand side and its one cell's screening sample.
    b0 = np.asarray(base[:8], dtype=np.float64)
    count0 = base[8]
    centroid0 = np.asarray(base[9:12], dtype=np.float64) / max(count0, 1.0)
    x0 = _octree_base_solve(b0, alpha * count0, centroid0)
    _launch.copy(level(fields, 0), _launch.array(x0.astype(np.float32), wp.float32, device=device))
    serial = not wp.get_device(device).is_cuda
    for d in range(1, depth + 1):
        res = 2**d + 1
        field = level(fields, d)
        _launch.launch(
            kernel_octree.octree_prolong,
            dim=(res, res, res),
            inputs=[level(fields, d - 1), 2 ** (d - 1) + 1, res],
            outputs=[level(prolongeds, d), field],
            device=device,
        )
        b = level(constraints, d)
        if serial:
            # The CPU runs a launch's threads in order anyway: every wave in one launch.
            _launch.launch(
                kernel_octree.octree_gauss_seidel_depth,
                dim=1,
                inputs=[d, depth, full_depth, iterations, b, *sweep_args],
                outputs=[field],
                device=device,
            )
            continue
        half = (res + 1) // 2
        for wave in range(res + 2 * (iterations - 1)):
            first = max(0, (wave - res + 2) // 2)
            last = min(iterations - 1, wave // 2)
            if first > last:
                continue
            for colour in range(4):
                _launch.launch(
                    kernel_octree.octree_gauss_seidel_wave,
                    dim=(last - first + 1, half, half),
                    inputs=[d, depth, full_depth, wave, first, colour, b, *sweep_args],
                    outputs=[field],
                    device=device,
                )
    return prolongeds, fields


def _octree_base_solve(b: np.ndarray, screen: float, centroid: np.ndarray) -> np.ndarray:
    """
    Depth 0's correction: the upstream conjugate gradient on its eight unknowns.

    The hat stiffness of the unit cube's eight corners plus ``screen`` times the outer product of
    their values at ``centroid``; at most eight rounds to a relative residual of ``1e-3``, and no
    round when ``|b|**2`` is already under ``1e-6`` (the upstream absolute exit).
    """
    mass = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])
    stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])
    operator = (
        np.kron(np.kron(stiffness, mass), mass)
        + np.kron(np.kron(mass, stiffness), mass)
        + np.kron(np.kron(mass, mass), stiffness)
    )
    hats = [np.array([1.0 - c, c]) for c in centroid]
    phi = np.kron(np.kron(hats[0], hats[1]), hats[2])
    operator = operator + screen * np.outer(phi, phi)
    eps = 1e-6
    x = np.zeros(8)
    r = b.copy()
    d = r.copy()
    delta = float(r @ r)
    if delta <= eps:
        return x
    delta0 = delta
    for _ in range(8):
        if delta <= eps * delta0:
            break
        q = operator @ d
        dq = float(d @ q)
        if dq == 0.0:
            break
        step = delta / dq
        x += step * d
        r -= step * q
        old, delta = delta, float(r @ r)
        d = r + (delta / old) * d
    return x


def _octree_extract(
    prolongeds: wp.array[wp.float32],
    fields: wp.array[wp.float32],
    node_offsets: wp.array[wp.int32],
    node_offsets_host: list[int],
    edge_offsets: wp.array[wp.int32],
    edge_offsets_host: list[int],
    iso: float,
    depth: int,
    full_depth: int,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    cell_offsets_host: list[int],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    PoissonRecon's extraction over the leaf cells: unit-cube vertices and triangles.

    ``kernels/algorithms/poisson_octree``'s extraction note has the rules. Per depth, finest
    first, the corner sources and the edges' vertices; then the borrowed faces sorted into their
    coarse faces' order, every leaf's segments, its loops and their triangles.
    """
    device = fields.device
    first = max(full_depth, 1)
    table = _octree_extraction_table(wp.get_device(device).alias)
    iso32 = wp.float32(iso)

    # A depth or -1 a node: ``int8`` holds it.
    sources = _launch.empty(node_offsets_host[-1], dtype=wp.int8, device=device)
    # The vertex each edge holds, named by the edge that created it.
    n_edges = edge_offsets_host[-1]
    owners = _launch.full(n_edges, -1, dtype=wp.int32, device=device)
    # One bit an edge: whether it creates a vertex.
    created = _launch.zeros((n_edges + 31) // 32, dtype=wp.uint32, device=device)
    activity = [flags, flag_offsets]
    for d in range(depth, first - 1, -1):
        res = 2**d + 1
        _launch.launch(
            kernel_octree.corner_sources,
            dim=(res, res, res),
            inputs=[d, depth, full_depth, *activity, node_offsets],
            outputs=[sources],
            device=device,
        )
        _launch.launch(
            kernel_octree.classify_edges,
            dim=(3, res, res, res),
            inputs=[
                d,
                depth,
                full_depth,
                iso32,
                fields,
                sources,
                node_offsets,
                *activity,
                edge_offsets,
            ],
            outputs=[owners, created],
            device=device,
        )
    # The vertices' numbers: the created edges' ranks.
    ranks = _bit_ranks(created)
    # One read sizes the vertices.
    n_vertices = int(read_scalar(ranks))
    positions = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    # Each vertex's pair across a refined edge, by vertex number (-1 for none).
    partners = _launch.full(max(n_vertices, 1), -1, dtype=wp.int32, device=device)
    if depth > first:
        _launch.launch(
            kernel_octree.pair_edge_vertices,
            dim=edge_offsets_host[depth] - edge_offsets_host[first],
            inputs=[first, depth, full_depth, *activity, edge_offsets, owners, created, ranks],
            outputs=[partners],
            device=device,
        )
    for d in range(first, depth + 1):
        res = 2**d + 1
        margin = kernel_levelset.lattice_edge_margin(1.0 / 2**d)
        _launch.launch(
            kernel_octree.place_edge_vertices,
            dim=(3, res, res, res),
            inputs=[
                d,
                iso32,
                wp.float32(margin),
                fields,
                prolongeds,
                sources,
                node_offsets,
                edge_offsets,
                created,
                ranks,
            ],
            outputs=[positions],
            device=device,
        )

    # The leaves that can have segments, in cell order: the per-cell passes below run over these
    # alone, and every count and slot they scan follows the same order.
    n_cells = cell_offsets_host[depth + 1] - cell_offsets_host[first]
    common = [first, depth, full_depth, iso32, fields, sources, node_offsets, *activity]
    band_bits = _launch.zeros((n_cells + 31) // 32, dtype=wp.uint32, device=device)
    _launch.launch(
        kernel_octree.extraction_band,
        dim=n_cells,
        inputs=[*common, cell_offsets],
        outputs=[band_bits],
        device=device,
    )
    band_ranks = _bit_ranks(band_bits)
    # One read sizes the band.
    n_band = int(read_scalar(band_ranks))
    if n_band == 0:
        return positions, _launch.empty(0, dtype=wp.int32, device=device)
    band = _launch.empty(n_band, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_octree.band_cells,
        dim=band_bits.size,
        inputs=[band_bits, band_ranks],
        outputs=[band],
        device=device,
    )

    n_faces = 6 * n_band
    face_args = [*common, cell_offsets, table, owners, edge_offsets]
    face_offsets = _launch.empty(n_faces, dtype=wp.int32, device=device)
    # The counting pass: no key buffer (a null array reads as empty in the kernel).
    _launch.launch(
        kernel_octree.borrowed_faces,
        dim=n_faces,
        inputs=[*face_args, band, face_offsets],
        outputs=[None, None],
        device=device,
    )
    _launch.array_scan(face_offsets, face_offsets, inclusive=True)
    # One read sizes the borrowed faces' sort.
    n_borrowed = int(read_scalar(face_offsets))
    keys = _launch.empty(max(2 * n_borrowed, 2), dtype=wp.uint64, device=device)
    borrowed = _launch.empty(max(2 * n_borrowed, 2), dtype=wp.int32, device=device)
    if n_borrowed > 0:
        _launch.launch(
            kernel_octree.borrowed_faces,
            dim=n_faces,
            inputs=[*face_args, band, face_offsets],
            outputs=[keys, borrowed],
            device=device,
        )
        _launch.radix_sort_pairs(keys, borrowed, n_borrowed)
    keys = _launch.view(keys, 0, n_borrowed)

    segment_offsets = _launch.empty(n_band, dtype=wp.int32, device=device)
    segment_args = [*face_args, keys, borrowed, band, segment_offsets]
    _launch.launch(
        kernel_octree.cell_segments, dim=n_band, inputs=segment_args, outputs=[None], device=device
    )
    _launch.array_scan(segment_offsets, segment_offsets, inclusive=True)
    # One read sizes the segments.
    n_segments = int(read_scalar(segment_offsets))
    segments = _launch.empty(max(n_segments, 1), dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_octree.cell_segments,
        dim=n_band,
        inputs=segment_args,
        outputs=[segments],
        device=device,
    )
    loops = _launch.empty(max(2 * n_segments, 1), dtype=wp.int32, device=device)
    sizes = _launch.zeros(max(n_segments, 1), dtype=wp.int32, device=device)
    loop_counts = _launch.empty(n_band, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_octree.cell_loops,
        dim=n_band,
        inputs=[segment_offsets, created, ranks, partners, segments],
        outputs=[loops, sizes, loop_counts],
        device=device,
    )
    _launch.array_scan(loop_counts, loop_counts, inclusive=True)
    # One read sizes the faces and the minimal-area search's scratch.
    totals = read_scalar(loop_counts)
    n_triangles, n_scratch = int(totals[0]), int(totals[1])
    scratch = _launch.empty(max(n_scratch, 1), dtype=wp.float32, device=device)
    choices = _launch.empty(max(n_scratch, 1), dtype=wp.int32, device=device)
    faces = _launch.empty(3 * n_triangles, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_octree.triangulate_loops,
        dim=n_band,
        inputs=[
            segment_offsets,
            loop_counts,
            created,
            ranks,
            positions,
            loops,
            sizes,
            scratch,
            choices,
        ],
        outputs=[faces],
        device=device,
    )
    return positions, faces


def _bit_ranks(words: wp.array[wp.uint32]) -> wp.array[wp.int32]:
    """Return the inclusive scan of a bitmask's word bit counts, for ``kernel_array.bit_rank``."""
    ranks = _launch.empty(words.size, dtype=wp.int32, device=words.device)
    _launch.map(kernel_array.popcount32, words, out=ranks)
    _launch.array_scan(ranks, ranks, inclusive=True)
    return ranks


@functools.cache
def _octree_extraction_table(device: str) -> wp.array[wp.int32]:
    """``kernels/algorithms/poisson_octree.EXTRACTION_TABLE`` on ``device``, uploaded once."""
    return _launch.array(kernel_octree.EXTRACTION_TABLE, dtype=wp.int32, device=device)


def _poisson_cube(
    points: wp.array[wp.vec3], scale: float, *, bounds: tuple[wp.vec3, wp.vec3] | None = None
) -> tuple[wp.vec3, wp.vec3, float]:
    """
    Return the padded cubic reconstruction domain (lower, upper, side) around the cloud AABB.

    ``bounds`` is the cloud's [`aabb`][ordito.bounds.aabb] when the caller holds it.
    """
    lo, hi = od.bounds.aabb(points) if bounds is None else bounds
    center = 0.5 * (lo + hi)
    max_extent = max(b - a for a, b in zip(odt.vec3_floats(lo), odt.vec3_floats(hi), strict=True))
    if max_extent <= 0.0:
        raise ValueError("screened_poisson requires points with a non-degenerate bounding box.")
    cube_size = scale * max_extent
    half = wp.vec3(0.5 * cube_size, 0.5 * cube_size, 0.5 * cube_size)
    return center - half, center + half, cube_size


def _field_iso_value(
    field: wp.array[wp.float32],
    res: int,
    cube_lower: wp.vec3,
    inv_cell: float,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    confidence: bool,
) -> float:
    """``_poisson_iso_value`` of the ``res**3`` node ``field`` at the samples."""
    device = points.device
    terms = kernel_reconstruction.FieldSampleTerms()
    terms.n = points.size
    terms.field = field
    terms.res = res
    terms.cube_lower = cube_lower
    terms.inv_cell = inv_cell

    def sample() -> wp.array[wp.float32]:
        sampled = _launch.empty(points.size, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_reconstruction.sample_field_trilinear,
            dim=points.size,
            inputs=[field, res, cube_lower, wp.float32(inv_cell), points],
            outputs=[sampled],
            device=device,
        )
        return sampled

    terms.points = points
    return _poisson_iso_value(
        kernel_reconstruction.FIELD_SAMPLE_SUM, terms, sample, normals, confidence
    )


def _poisson_iso_value(
    sum_kernels: kernel_reduce.TermSum,
    terms: object,
    sample: Callable[[], wp.array[wp.float32]],
    normals: wp.array[wp.vec3],
    confidence: bool,
) -> float:
    """
    Iso-value for the extraction: the mean of the solution at the input samples.

    PoissonRecon's ``GetIsoValue``. With ``confidence`` set, the mean is weighted by
    ``|normals[i]|`` (the same per-sample confidence the splat/quadrature weights use) instead of
    uniform; the two weightings are deliberately kept distinct rather than unified. Shared by both
    backends, whose only difference is how the solution is sampled: ``sum_kernels`` sums the
    samples over ``terms`` without storing them -- ``reduce.mean`` of the sampled values, bit for
    bit -- and ``sample`` stores them, for the weighted mean.

    Both branches reduce on the device rather than reading the fields back.
    """
    n = normals.size
    if not confidence:
        total = od.reduce.sum_of_terms(sum_kernels, [terms], n, device=normals.device)
        # The mean's one readback: the extraction takes the iso-value as a host scalar.
        return float(read_scalar(total, 0)) / n
    sampled = sample()
    lengths = odt.empty_1d(normals.size, wp.float32, device=normals.device)
    _launch.map(wp.length, normals, out=lengths)
    total_weight = od.reduce.sum(lengths)
    if total_weight <= 0.0:
        return 0.0
    return od.reduce.weighted_sum(sampled, lengths) / total_weight


def _poisson_dense_solve(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    cube_size: float,
    depth: int,
    full_depth: int,
    screen: float,
    confidence: bool,
    solver_iterations: int,
    solver_tolerance: float,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.float32], int]:
    """Run the cascadic dense solve, returning the finest solution buffer and its resolution."""
    prev_solution: wp.array[wp.float32] | None = None
    prev_res = 0
    # Every level's iteration count, read back once after the whole cascade rather than after each
    # level: a read between levels would stall the host until that level's solve drained, where
    # without it the next level's splat, setup and graph recording are issued while it runs.
    levels = range(full_depth, depth + 1)
    counts = _launch.empty(len(levels), dtype=wp.int32, device=device)
    level_results: list[tuple[int, wp.array[wp.float64]]] = []
    for k, level in enumerate(levels):
        res = (1 << level) + 1
        inv_cell = float(res - 1) / cube_size
        n_nodes = res * res * res
        if prev_solution is None:
            initial = _launch.zeros(n_nodes, dtype=wp.float32, device=device)
        else:
            initial = _launch.empty(n_nodes, dtype=wp.float32, device=device)
            _launch.launch(
                kernel_reconstruction.prolong_grid,
                dim=(res, res, res),
                inputs=[prev_solution, prev_res, res, initial],
                device=device,
            )
        residual_tolerance = _poisson_solve_level(
            points,
            normals,
            cube_lower,
            inv_cell,
            res,
            screen,
            confidence,
            initial,
            solver_iterations,
            solver_tolerance,
            odt.as_dense(counts[k : k + 1]),
            device,
        )
        level_results.append((res, residual_tolerance))
        prev_solution = initial
        prev_res = res

    assert prev_solution is not None
    # One read of the counts, and one more only for a level that used its whole budget: a level
    # that stops at ``solver_iterations`` above its tolerance says so, as ``linalg.solve_spd`` does.
    for iterations, (res, residual_tolerance) in zip(counts.numpy(), level_results, strict=True):
        _warn_if_unconverged(res, int(iterations), residual_tolerance, solver_iterations)
    return prev_solution, prev_res


def _warn_if_unconverged(
    res: int, iterations: int, residual_tolerance: wp.array[wp.float64], solver_iterations: int
) -> None:
    """
    Warn when a ``res**3`` level's solve used its whole budget and ended above its tolerance.

    ``residual_tolerance`` is ``_solve_screened_poisson``'s device pair (squared threshold, squared
    residual), read only for a level that hit the cap -- as ``linalg.solve_spd`` reports.
    """
    if iterations < solver_iterations:
        return
    tolerance_sq, residual_sq = residual_tolerance.numpy()
    residual, tolerance = math.sqrt(float(residual_sq)), math.sqrt(float(tolerance_sq))
    if residual > tolerance:
        warnings.warn(
            f"screened_poisson: the {res}^3 level's conjugate gradient hit its "
            f"{solver_iterations}-iteration cap with residual norm {residual:.3e} against "
            f"tolerance {tolerance:.3e}; raise solver_iterations or solver_tolerance.",
            stacklevel=4,
        )


def _poisson_solve_level(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    inv_cell: float,
    res: int,
    screen: float,
    confidence: bool,
    initial: wp.array[wp.float32],
    solver_iterations: int,
    solver_tolerance: float,
    iterations: wp.array[wp.int32],
    device: wp.DeviceLike,
) -> wp.array[wp.float64]:
    """
    Solve one cascade level ``(L_N + screen * W) x = -div V`` matrix-free from ``initial``.

    Splats the oriented normals into the vector field ``V`` and density weight ``W`` on a
    ``res**3`` node grid, normalizes ``V``, builds the negative-divergence right-hand side, and runs
    ``_solve_screened_poisson``'s multigrid-preconditioned conjugate gradient with the screened
    Laplacian applied matrix-free. ``initial`` is the warm start and is solved in place;
    ``iterations`` receives the round count, and the returned device pair is the squared threshold
    and squared residual (``_solve_screened_poisson``) -- nothing is read back here.
    """
    n_nodes = res * res * res
    # The splat's four accumulators in one zeroed allocation.
    splat = _launch.zeros(4 * n_nodes, dtype=wp.float32, device=device)
    vx, vy, vz, weights = (odt.as_dense(splat[k * n_nodes : (k + 1) * n_nodes]) for k in range(4))
    if is_deterministic() and wp.get_device(device).is_cuda:
        _splat_ordered(
            kernel_reconstruction.splat_corner_keys,
            [points, cube_lower, wp.float32(inv_cell), res],
            points,
            normals,
            cube_lower,
            inv_cell,
            res,
            confidence,
            n_nodes,
            [vx, vy, vz, weights],
        )
    else:
        _launch.launch(
            kernel_reconstruction.splat_normals,
            dim=points.size,
            inputs=[
                points,
                normals,
                cube_lower,
                wp.float32(inv_cell),
                res,
                wp.int32(1 if confidence else 0),
                vx,
                vy,
                vz,
                weights,
            ],
            device=device,
        )
    rhs, smoother = _launch.carve(wp.float32, [n_nodes] * 2, device)
    _launch.launch(
        kernel_reconstruction.poisson_level_setup,
        dim=(res, res, res),
        inputs=[vx, vy, vz, weights, wp.float32(screen), res, wp.float32(_MG_OMEGA)],
        outputs=[rhs, smoother],
        device=device,
    )

    return _solve_screened_poisson(
        weights,
        screen,
        res,
        rhs,
        smoother,
        initial,
        solver_iterations,
        solver_tolerance,
        iterations,
        device,
    )


def _solve_screened_poisson(
    weights: wp.array[wp.float32],
    screen: float,
    res: int,
    rhs: wp.array[wp.float32],
    smoother: wp.array[wp.float32],
    solution: wp.array[wp.float32],
    maxiter: int,
    tol: float,
    iterations: wp.array[wp.int32],
    device: wp.DeviceLike,
) -> wp.array[wp.float64]:
    """
    Multigrid-preconditioned conjugate gradient on the grid's screened Laplacian, from ``solution``.

    The Chronopoulos-Gear round ``linalg``'s own solver runs -- two launches, the mat-vec with its
    three dots (``poisson_cg_matvec_dots``) and the shared update ``cg_update`` -- with the
    stencil applied matrix-free, since a CSR of it would be seven entries a node. The vectors stay
    ``float32``: at ``2 ** 8 + 1`` nodes a side a round is bound by the bytes it moves, and doubling
    them would double it; the dots' fold across blocks and ``alpha`` and ``beta`` are ``float64``.
    The stopping rule is ``warp.optim.linear.cg``'s with ``atol = 0`` -- the relative residual
    ``tol`` or ``maxiter`` rounds -- tested on device, so the loop is one recorded graph and nothing
    is read back: the round count lands in ``iterations`` and the squared threshold and residual in
    the returned length-2 array, for the caller to test once every level has been issued.
    """
    n = res * res * res
    tile = int(kernel_cg.CG_TILE)
    span, blocks, fold = kernel_cg.cg_layout(n, od.linalg.CG_FOLD_MAX_BLOCKS)
    stride = blocks * span
    # The V-cycle's vectors and coarse operators ride in the solve's ``float32`` allocation.
    work = _cg_workspace(stride, blocks, _PoissonMultigrid.widths(res), 0, device)
    r, u, w, p, s = work.vectors
    partials, atol_sq, dots, coefficients = (
        work.partials,
        work.atol_sq,
        work.dots,
        work.coefficients,
    )
    gamma_old, alpha_old, gamma_new, alpha_new = work.recurrence
    state = work.state
    screen_f = wp.float32(screen)
    multigrid = _PoissonMultigrid(weights, screen, res, smoother, work.extra, device)
    # The V-cycle's first pre-smoothing sweep is pointwise, so the launches that produce the
    # residual write it too -- this one and every round's ``cg_update``, whose Jacobi apply with
    # ``smoother`` as the scaling is that sweep exactly.
    start = multigrid.start
    _launch.launch_tiled(
        kernel_reconstruction.poisson_cg_initial,
        dim=(blocks,),
        inputs=[n, span, res, weights, screen_f, rhs, solution, smoother],
        outputs=[r, u, p, s, start, partials],
        block_dim=tile,
        device=device,
    )
    _launch.launch_tiled(
        kernel_cg.cg_seed,
        dim=(1,),
        inputs=[wp.float64(tol * tol), wp.float64(0.0), blocks, partials],
        outputs=[atol_sq, gamma_new, alpha_new, iterations, state],
        block_dim=tile,
        device=device,
    )
    multigrid.apply(r, u)

    def round_() -> None:
        _launch.launch_tiled(
            kernel_reconstruction.poisson_cg_matvec_dots,
            dim=(blocks,),
            inputs=[n, span, res, weights, screen_f, r, u, gamma_new, alpha_new],
            outputs=[w, partials, gamma_old, alpha_old, state],
            block_dim=tile,
            device=device,
        )
        if not fold:
            _launch.launch_tiled(
                kernel_cg.cg_coefficients,
                dim=(1,),
                inputs=[1, maxiter, blocks, partials, gamma_old, alpha_old, atol_sq, state],
                outputs=[coefficients, gamma_new, alpha_new, dots, iterations],
                block_dim=tile,
                device=device,
            )
        _launch.launch_tiled(
            kernel_cg.CG_UPDATE[wp.float32],
            # One tile a block: a pure stream unless it folds; see ``linalg._BatchedCg``.
            dim=(1, stride // tile),
            inputs=[
                stride,
                tile,
                n,
                1,
                maxiter,
                1 if fold else 0,
                blocks,
                # The "Jacobi apply" at ``smoother`` is the V-cycle's zero-start sweep, into its
                # start vector; the V-cycle writes ``u`` after the update.
                1,
                partials,
                coefficients,
                gamma_old,
                alpha_old,
                atol_sq,
                smoother,
                w,
                p,
                s,
                r,
                u,
                state,
            ],
            outputs=[solution, start, gamma_new, alpha_new, dots, iterations],
            block_dim=tile,
            device=device,
        )
        multigrid.apply(r, u)

    run_device_loop(device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), round_)
    return work.residual_pair


# Damped-Jacobi sweeps before and after each coarse-grid correction, the damping, the sweeps on the
# coarsest level and its size. Swept on ``bunny`` at depth 8 to ``tol = 1e-5``: one sweep a side at
# ``6/7`` (the classical optimum for the 7-point stencil in 3-D) was fastest at every level; two
# sweeps, ``2/3``, or eight coarsest sweeps all cost 5-15 % more for no fewer iterations. Coarsening
# to three nodes a side makes the coarsest level 27 unknowns, which four sweeps all but solve.
_MG_SWEEPS = 1
_MG_OMEGA = 6.0 / 7.0
# At least two: the coarsest level's first sweep is the zero-start one its restriction writes.
_MG_COARSE_SWEEPS = 4
_MG_COARSEST = 3


class _GridLevel(NamedTuple):
    """
    One level of ``_PoissonMultigrid``: its grid, operator and working vectors.

    ``b`` and ``x`` are ``None`` on the finest level, whose right-hand side and result are the
    caller's vectors.
    """

    res: int
    lap: wp.float32
    weights: wp.array[wp.float32]
    smoother: wp.array[wp.float32]
    b: wp.array[wp.float32] | None
    x: wp.array[wp.float32] | None
    start: wp.array[wp.float32]
    tmp: wp.array[wp.float32]
    residual: wp.array[wp.float32]


class _PoissonMultigrid:
    """
    Geometric V-cycle on the dense grid's nested node grids, as a conjugate-gradient preconditioner.

    Levels ``res, (res - 1) / 2 + 1, ...`` down to ``_MG_COARSEST`` nodes a side, level ``l``'s
    operator ``2 ** l * L_l + screen * W_l`` (see the V-cycle note in ``kernels/reconstruction``),
    ``_MG_SWEEPS`` damped-Jacobi sweeps at ``_MG_OMEGA`` before and after each coarse-grid
    correction and ``_MG_COARSE_SWEEPS`` on the coarsest level. Every level is ``float32`` and
    matrix-free. It keeps the iteration count near a dozen whatever the resolution, where Jacobi's
    grows with it -- to the point that a 100-iteration cap left the finer levels two orders of
    magnitude short of their tolerance.

    Every level's cycle starts from a zero guess, where the first sweep ``omega D^-1 b`` reads no
    neighbour, so it is not a launch of its own: the launch that writes a level's right-hand side
    writes that sweep into the level's ``start`` too (``poisson_mg_restrict`` on the coarse levels;
    on the finest, the caller -- the conjugate gradient's initial-residual and update kernels --
    into ``start`` before each ``apply``). A cycle is then four launches a level.
    """

    def __init__(
        self,
        weights: wp.array[wp.float32],
        screen: float,
        res: int,
        smoother: wp.array[wp.float32],
        storage: list[wp.array[wp.float32]],
        device: wp.DeviceLike,
    ) -> None:
        """Build the hierarchy below ``res`` in ``storage``, the views ``widths(res)`` lists."""
        self._device = device
        self._screen = wp.float32(screen)
        self._levels: list[_GridLevel] = []
        views = iter(storage)
        lap = 1.0
        w, smooth = weights, smoother
        finest = True
        while True:
            owned = 3 if finest else 5
            vectors = [next(views) for _ in range(owned)]
            b, x = (None, None) if finest else (vectors.pop(), vectors.pop())
            self._levels.append(_GridLevel(res, wp.float32(lap), w, smooth, b, x, *vectors))
            finest = False
            if res <= _MG_COARSEST:
                break
            res_c = (res - 1) // 2 + 1
            lap *= 2.0
            w_c = next(views)
            smooth_c = next(views)
            _launch.launch(
                kernel_reconstruction.poisson_mg_coarsen,
                dim=(res_c, res_c, res_c),
                inputs=[w, res, res_c, wp.float32(lap), self._screen, wp.float32(_MG_OMEGA)],
                outputs=[w_c, smooth_c],
                device=device,
            )
            res, w, smooth = res_c, w_c, smooth_c

    @staticmethod
    def widths(res: int) -> list[int]:
        """
        Return the widths of the vectors below a ``res`` grid, in the order ``__init__`` takes them.

        Every vector is one buffer's view, so the whole hierarchy is one allocation (the caller's).
        """
        widths = [res**3] * 3
        while res > _MG_COARSEST:
            res = (res - 1) // 2 + 1
            widths += [res**3] * 2
            widths += [res**3] * 5
        return widths

    @property
    def start(self) -> wp.array[wp.float32]:
        """The finest level's zero-start sweep ``omega D^-1 b``, which ``apply``'s caller writes."""
        return self._levels[0].start

    def _smooth(
        self,
        level: _GridLevel,
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        out: wp.array[wp.float32],
    ) -> None:
        """One damped-Jacobi sweep at ``level`` from ``x`` into ``out``."""
        res = level.res
        _launch.launch(
            kernel_reconstruction.poisson_mg_smooth,
            dim=(res, res, res),
            inputs=[res, level.lap, level.weights, self._screen, level.smoother, b, x],
            outputs=[out],
            device=self._device,
        )

    def _sweeps(
        self,
        level: _GridLevel,
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        count: int,
        out: wp.array[wp.float32],
    ) -> None:
        """``count >= 1`` sweeps from ``x``, alternating with ``level.tmp``, the last in ``out``."""
        cur = x
        for sweep in range(count):
            target = out if (count - 1 - sweep) % 2 == 0 else level.tmp
            self._smooth(level, b, cur, target)
            cur = target

    def _cycle(self, index: int, b: wp.array[wp.float32], out: wp.array[wp.float32]) -> None:
        """``out = V_index(b)`` from a zero guess, whose first sweep is already in ``start``."""
        level = self._levels[index]
        res = level.res
        if index == len(self._levels) - 1:
            self._sweeps(level, b, level.start, _MG_COARSE_SWEEPS - 1, out)
            return
        # The pre-smoothing's remaining sweeps, ping-ponging between ``tmp`` and ``out`` (neither is
        # read before the post-smoothing writes it).
        cur = level.start
        for sweep in range(1, _MG_SWEEPS):
            nxt = level.tmp if sweep % 2 == 1 else out
            self._smooth(level, b, cur, nxt)
            cur = nxt
        _launch.launch(
            kernel_reconstruction.poisson_mg_residual,
            dim=(res, res, res),
            inputs=[res, level.lap, level.weights, self._screen, b, cur],
            outputs=[level.residual],
            device=self._device,
        )
        coarse = self._levels[index + 1]
        assert coarse.b is not None
        assert coarse.x is not None
        res_c = coarse.res
        _launch.launch(
            kernel_reconstruction.poisson_mg_restrict,
            dim=(res_c, res_c, res_c),
            inputs=[level.residual, res, res_c, coarse.smoother],
            outputs=[coarse.b, coarse.start],
            device=self._device,
        )
        self._cycle(index + 1, coarse.b, coarse.x)
        # Corrected into whichever of ``tmp`` / ``out`` lets the post-smoothing's alternation end in
        # ``out``; in place when ``cur`` is already that vector, which the kernel allows.
        corrected = level.tmp if _MG_SWEEPS % 2 == 1 else out
        _launch.launch(
            kernel_reconstruction.poisson_mg_prolong_add,
            dim=(res, res, res),
            inputs=[coarse.x, res_c, res, cur],
            outputs=[corrected],
            device=self._device,
        )
        self._sweeps(level, b, corrected, _MG_SWEEPS, out)

    def apply(self, source: wp.array[wp.float32], destination: wp.array[wp.float32]) -> None:
        """``destination = M^-1 source`` by one V-cycle from zero; ``start`` must hold its sweep."""
        self._cycle(0, source, destination)


# Levels below ``depth`` solved on the dense base lattice by the band backend, and the conjugate-
# gradient rounds a band level runs. A band level starts from the prolonged coarse solution and
# solves only the correction the finer lattice adds near the surface, which is local, so a fixed
# handful of rounds settles it. The finest level is the most expensive and the least in need of
# rounds, since every coarser level has already placed the surface. The cascade alone leaves each
# band at its coarse level's rediscretized operator -- the gap to the dense backend's surface that
# no band width or round count closes -- so one FAS cycle follows it: restrict
# each band's composite residual with its solution injected where it covers the level below,
# solve the dense base's tau-corrected system (a few rounds from that start), then correct and
# re-solve each band (``kernels/algorithms/poisson_band`` has the scheme and its measurements).
_BAND_DENSE_LEVELS_BELOW = 2
_BAND_ROUNDS = 30
_BAND_ROUNDS_FINEST = 8
_BAND_FAS_ROUNDS = 30
_BAND_FAS_ROUNDS_FINEST = 8
_BAND_FAS_BASE_ROUNDS = 4


class _BandLevel(NamedTuple):
    """
    One narrow-band lattice level: which ``8**3``-node bricks of the ``res**3`` grid are stored.

    ``brick_map`` maps a brick to its slot or ``-1``; ``brick_coords`` is each slot's brick; a band
    node's storage index is ``slot * 512 + local`` (``kernels/algorithms/poisson_band``).
    """

    res: int
    nbk: int
    brick_map: wp.array[wp.int32]
    brick_coords: wp.array[wp.vec3i]
    n_nodes: int


def _screened_poisson_band(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    cube_size: float,
    *,
    depth: int,
    full_depth: int,
    screen: float,
    solver_iterations: int,
    solver_tolerance: float,
    confidence: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Solve the screened-Poisson system on a narrow band of the dense lattice and extract it.

    Backs [`screened_poisson`][ordito.reconstruction.screened_poisson]'s ``method="adaptive"``.
    The finest level's splat is the only data; the levels between a dense base lattice and the
    finest are stored only near the surface, in ``8**3``-node bricks, and each solves for its band
    with every out-of-band neighbour held at the prolonged coarser solution. A coarse level's data
    is the ``P^T`` restriction of the finest one's and its operator the dense multigrid's coarse
    operator, so a coarse solution approximates the fine one and its prolongation is a valid
    boundary value -- which an independently splatted coarse level is not: the converged depth-8
    and depth-9 dense solutions differ by a factor that varies across the domain, since screening is
    not normalized per level.

    Two passes: the band of the bricks the samples touch carries all the data, so the dense base is
    solved from its restriction; then the bricks the base's iso-surface crosses are added -- where
    the surface closes a hole the samples leave open -- and the band levels are solved coarse to
    fine, then one FAS cycle makes each coarser level's solution the finer one's
    where the finer band covers it (``kernels/algorithms/poisson_band``). The surface is extracted
    from the composite field (band values, prolonged coarse ones off the band) over the finest band
    grown by a brick, with ``levelset.marching_cubes``' numbering and edge margin. Two counts are
    kept on the device and read with the extraction's totals: coarse nodes a restriction or a ghost
    found absent (the nesting makes it zero) and crossing cells the extraction band could not own
    (each a hole); either being nonzero warns.
    """
    device = points.device
    n = points.size
    base = max(full_depth, depth - _BAND_DENSE_LEVELS_BELOW)
    res_f = (1 << depth) + 1
    nbk_f = (res_f + 7) // 8
    res_b = (1 << base) + 1
    missing = _launch.zeros(1, dtype=wp.int32, device=device)

    # Pass 1: the sample bands, the finest data and its restriction to the dense base.
    sample_flags = _launch.zeros(nbk_f**3, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_poisson_band.mark_sample_bricks,
        dim=n,
        inputs=[points, cube_lower, wp.float32((res_f - 1) / cube_size), res_f, nbk_f],
        outputs=[sample_flags],
        device=device,
    )
    final_flags = _launch.clone(sample_flags)
    sample_levels = _band_hierarchy_flags(sample_flags, depth, base, device)
    sample_built, _ = _compact_bands(list(sample_levels.values()), device)
    sample_bands = dict(zip(sample_levels, sample_built, strict=True))
    data = _band_data(
        sample_bands, points, normals, cube_lower, cube_size, confidence, missing, device
    )
    base_b, base_w = _launch.carve(wp.float32, [res_b**3] * 2, device, zeroed=True)
    _restrict_band(sample_bands[base + 1], *data[base + 1], None, res_b, base_b, base_w, missing)

    lap_b = float(1 << (depth - base))
    rhs, smoother = _launch.carve(wp.float32, [res_b**3] * 2, device)
    _launch.launch(
        kernel_poisson_band.dense_base_setup,
        dim=(res_b, res_b, res_b),
        inputs=[
            base_b,
            base_w,
            res_b,
            wp.float32(1.0 / lap_b),
            wp.float32(screen / lap_b),
            wp.float32(_MG_OMEGA),
        ],
        outputs=[rhs, smoother],
        device=device,
    )
    base_x = _launch.zeros(res_b**3, dtype=wp.float32, device=device)
    base_rounds = _launch.empty(1, dtype=wp.int32, device=device)
    base_residual = _solve_screened_poisson(
        base_w,
        screen / lap_b,
        res_b,
        rhs,
        smoother,
        base_x,
        solver_iterations,
        solver_tolerance,
        base_rounds,
        device,
    )

    # Pass 2: add the bricks the base surface crosses, and move the data onto the final bands.
    base_iso = _field_iso_value(
        base_x, res_b, cube_lower, (res_b - 1) / cube_size, points, normals, confidence
    )
    _launch.launch(
        kernel_poisson_band.mark_crossing_bricks,
        dim=(res_b - 1, res_b - 1, res_b - 1),
        inputs=[base_x, res_b, wp.float32(base_iso), (res_f - 1) // (res_b - 1), res_f, nbk_f],
        outputs=[final_flags],
        device=device,
    )
    extract_flags = _launch.empty_like(final_flags)
    _launch.launch(
        kernel_poisson_band.dilate_bricks,
        dim=(nbk_f, nbk_f, nbk_f),
        inputs=[final_flags, nbk_f],
        outputs=[extract_flags],
        device=device,
    )
    final_levels = _band_hierarchy_flags(final_flags, depth, base, device)
    # Every final band's brick count, the extraction band's and the base solve's round count in
    # one read, which waits for the base solve that the crossing bricks already depend on.
    built, (base_iterations,) = _compact_bands(
        [*final_levels.values(), (extract_flags, res_f)], device, tail=base_rounds
    )
    *final_built, extract = built
    bands = dict(zip(final_levels, final_built, strict=True))
    _warn_if_unconverged(res_b, base_iterations, base_residual, solver_iterations)

    # The band levels, coarse to fine, each from the prolonged level below it; then the FAS cycle.
    states: dict[int, _BandState] = {}
    for level in range(base + 1, depth + 1):
        band = bands[level]
        source = sample_bands[level]
        # The data, weights and solution in one allocation; the solution's zeros are overwritten by
        # its first start.
        b, w, x = _launch.carve(wp.float32, [band.n_nodes] * 3, device, zeroed=True)
        _launch.launch(
            kernel_poisson_band.embed_band_data,
            dim=source.n_nodes,
            inputs=[source.brick_coords, source.res, *data[level], band.brick_map, band.nbk],
            outputs=[b, w],
            device=device,
        )
        states[level] = _BandState(band, float(1 << (depth - level)), b, w, b, x, None)
        rounds = _BAND_ROUNDS_FINEST if level == depth else _BAND_ROUNDS
        _band_level_solve(
            states[level],
            states.get(level - 1),
            base_x,
            res_b,
            screen,
            min(rounds, solver_iterations),
            solver_tolerance,
            missing,
            device,
        )
    base_previous = base_x
    # The FAS cycle. Down: each level's composite residual restricted, with its solution injected
    # where it covers the level below, into that level's tau-corrected right-hand side.
    for level in range(depth, base, -1):
        fine, coarse = states[level], states.get(level - 1)
        residual = _launch.empty(fine.band.n_nodes, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_poisson_band.band_residual,
            dim=fine.band.n_nodes,
            inputs=[
                *_band_level_args(fine, screen),
                *_band_coarse_args(coarse, base_x, res_b),
                fine.rhs,
                fine.x,
            ],
            outputs=[residual],
            device=device,
        )
        if coarse is None:
            base_previous = _launch.empty(res_b**3, dtype=wp.float32, device=device)
            coarse_args = [None, None, 0, res_b, wp.float32(lap_b), base_w, wp.float32(screen)]
            coarse_x, coarse_data, previous, out_rhs, scale = (
                base_x,
                base_b,
                base_previous,
                rhs,
                1.0 / lap_b,
            )
        else:
            if coarse.previous is None:
                tau_rhs, previous = _launch.carve(wp.float32, [coarse.band.n_nodes] * 2, device)
                states[level - 1] = coarse = coarse._replace(rhs=tau_rhs, previous=previous)
            assert coarse.previous is not None
            coarse_args = [
                coarse.band.brick_coords,
                coarse.band.brick_map,
                coarse.band.nbk,
                coarse.band.res,
                wp.float32(coarse.lap),
                coarse.weights,
                wp.float32(screen),
            ]
            coarse_x, coarse_data, previous, out_rhs, scale = (
                coarse.x,
                coarse.data,
                coarse.previous,
                coarse.rhs,
                1.0,
            )
        fine_args = [fine.band.brick_map, fine.band.nbk, fine.band.res]
        _launch.launch(
            kernel_poisson_band.fas_inject,
            dim=previous.size,
            inputs=[coarse_args[0], coarse_args[3], *fine_args, fine.x, coarse_x],
            outputs=[previous],
            device=device,
        )
        _launch.launch(
            kernel_poisson_band.fas_rhs,
            dim=previous.size,
            inputs=[*coarse_args, coarse_data, previous, *fine_args, residual, wp.float32(scale)],
            outputs=[out_rhs],
            device=device,
        )
    _solve_screened_poisson(
        base_w,
        screen / lap_b,
        res_b,
        rhs,
        smoother,
        base_x,
        min(_BAND_FAS_BASE_ROUNDS, solver_iterations),
        solver_tolerance,
        _launch.empty(1, dtype=wp.int32, device=device),
        device,
    )
    # Up: each band level corrected by its coarse level's change and solved again.
    for level in range(base + 1, depth + 1):
        rounds = _BAND_FAS_ROUNDS_FINEST if level == depth else _BAND_FAS_ROUNDS
        coarse = states.get(level - 1)
        _band_level_solve(
            states[level],
            coarse,
            base_x,
            res_b,
            screen,
            min(rounds, solver_iterations),
            solver_tolerance,
            missing,
            device,
            previous=base_previous if coarse is None else coarse.previous,
        )

    fine_x = states[depth].x
    coarse_state = states.get(depth - 1)
    coarse_map, coarse_x, coarse_nbk, coarse_res = (
        (None, base_x, 0, res_b)
        if coarse_state is None
        else (
            coarse_state.band.brick_map,
            coarse_state.x,
            coarse_state.band.nbk,
            coarse_state.band.res,
        )
    )
    finest = bands[depth]
    terms = kernel_poisson_band.BandSampleTerms()
    terms.n = n
    terms.points = points
    terms.lower = cube_lower
    terms.inv_cell = (res_f - 1) / cube_size
    terms.res = res_f
    terms.brick_map = finest.brick_map
    terms.nbk = finest.nbk
    terms.x = fine_x

    def sample_band() -> wp.array[wp.float32]:
        sampled = _launch.empty(n, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_poisson_band.band_sample,
            dim=n,
            inputs=[
                points,
                cube_lower,
                wp.float32((res_f - 1) / cube_size),
                res_f,
                finest.brick_map,
                finest.nbk,
                fine_x,
            ],
            outputs=[sampled],
            device=device,
        )
        return sampled

    iso = _poisson_iso_value(
        kernel_poisson_band.BAND_SAMPLE_SUM, terms, sample_band, normals, confidence
    )
    composite = [
        finest.brick_map,
        fine_x,
        coarse_map,
        coarse_x,
        coarse_nbk,
        coarse_res,
        res_f,
        wp.float32(iso),
    ]
    table = _marching_cubes_table(wp.get_device(device).alias)
    # The per-node counts, then one slot the counts kernel accumulates ``(missing, skipped)`` into
    # (zeroed for it), so the scan's totals and the diagnostics come back in one read.
    extract_nodes = extract.n_nodes
    counts_and_diagnostics = _launch.zeros(extract_nodes + 1, dtype=wp.vec2i, device=device)
    counts = odt.as_dense(counts_and_diagnostics[:extract_nodes])
    _launch.launch(
        kernel_poisson_band.band_marching_cubes_counts,
        dim=extract_nodes,
        inputs=[extract.brick_coords, extract.brick_map, extract.nbk, *composite, table, missing],
        outputs=[counts, odt.as_dense(counts_and_diagnostics[extract_nodes:])],
        device=device,
    )
    _launch.array_scan(counts, counts, inclusive=True)
    # The totals size the outputs, so they are read; the diagnostics ride in the same copy.
    totals, (n_missing, n_skipped) = read_values(counts_and_diagnostics, extract_nodes - 1, 2)
    if n_missing > 0 or n_skipped > 0:
        warnings.warn(
            f"screened_poisson: the adaptive extraction dropped {n_skipped} crossing cell(s) its "
            f"band could not hold and {n_missing} coarse node(s) absent from a band, so the "
            "surface has holes.",
            stacklevel=3,
        )
    vertices = _launch.empty(int(totals[0]), dtype=wp.vec3, device=device)
    faces = _launch.empty(3 * int(totals[1]), dtype=wp.int32, device=device)
    if vertices.size > 0:
        spacing = cube_size / float(res_f - 1)
        _launch.launch(
            kernel_poisson_band.band_marching_cubes_emit,
            dim=extract.n_nodes,
            inputs=[
                extract.brick_coords,
                extract.brick_map,
                extract.nbk,
                *composite,
                cube_lower,
                wp.vec3(spacing, spacing, spacing),
                wp.float32(kernel_levelset.lattice_edge_margin(cube_size / float(res_f - 1))),
                table,
                counts,
            ],
            outputs=[vertices, faces],
            device=device,
        )
    return vertices, faces


class _BandState(NamedTuple):
    """
    One band level's system and solution through the cascade and the FAS cycle.

    ``data`` is the level's restricted data, ``rhs`` the right-hand side it solves before the ghost
    fold (``data`` itself until an FAS cycle tau-corrects it), ``x`` its solution and ``previous``
    the injected start of its last FAS coarse solve (``None`` on the finest level).
    """

    band: _BandLevel
    lap: float
    data: wp.array[wp.float32]
    weights: wp.array[wp.float32]
    rhs: wp.array[wp.float32]
    x: wp.array[wp.float32]
    previous: wp.array[wp.float32] | None


def _band_level_args(state: _BandState, screen: float) -> list[object]:
    """Return the ``(brick_coords, brick_map, nbk, res, lap, weights, screen)`` a band row reads."""
    band = state.band
    return [
        band.brick_coords,
        band.brick_map,
        band.nbk,
        band.res,
        wp.float32(state.lap),
        state.weights,
        wp.float32(screen),
    ]


def _band_coarse_args(
    coarse: _BandState | None, base_x: wp.array[wp.float32], res_b: int
) -> list[object]:
    """``(coarse_map, coarse, nbk_c, res_c)`` of the level below a band level, band or base."""
    if coarse is None:
        return [None, base_x, 0, res_b]
    return [coarse.band.brick_map, coarse.x, coarse.band.nbk, coarse.band.res]


def _band_level_solve(
    state: _BandState,
    coarse: _BandState | None,
    base_x: wp.array[wp.float32],
    res_b: int,
    screen: float,
    rounds: int,
    tol: float,
    missing: wp.array[wp.int32],
    device: wp.DeviceLike,
    *,
    previous: wp.array[wp.float32] | None = None,
) -> None:
    """
    Fold one band level's ghosts onto its right-hand side, start it and solve it in place.

    The start is the prolonged coarse solution, or with ``previous`` the level's own solution
    corrected by the prolonged change of the coarse one since ``previous``.
    """
    band = state.band
    rhs, inv_diag = _launch.carve(wp.float32, [band.n_nodes] * 2, device)
    coarse_map, coarse_x, nbk_c, res_c = _band_coarse_args(coarse, base_x, res_b)
    _launch.launch(
        kernel_poisson_band.band_level_setup,
        dim=band.n_nodes,
        inputs=[
            *_band_level_args(state, screen),
            coarse_map,
            coarse_x,
            previous,
            nbk_c,
            res_c,
            state.rhs,
            state.x,
        ],
        outputs=[rhs, inv_diag, missing],
        device=device,
    )
    _solve_band_level(
        band, state.lap, state.weights, screen, rhs, state.x, inv_diag, rounds, tol, device
    )


def _band_hierarchy_flags(
    flags: wp.array[wp.int32], depth: int, base: int, device: wp.DeviceLike
) -> dict[int, tuple[wp.array[wp.int32], int]]:
    """
    Every band level's brick flags and resolution, from ``depth`` down to ``base + 1``.

    The finest level's flags are ``flags``; each coarser level holds the bricks covering its finer
    level's, grown by one coarse brick, so every ghost a finer level reads and every node its data
    restricts into is stored. Every set is built before ``_compact_bands`` scans any in place.
    """
    level_flags = {depth: (flags, (1 << depth) + 1)}
    for level in range(depth - 1, base, -1):
        nbk_f = ((1 << (level + 1)) + 8) // 8
        nbk_c = ((1 << level) + 8) // 8
        # The projection's zeroed flags and the dilation's (written whole), in one allocation.
        projected, grown = _launch.carve(wp.int32, [nbk_c**3] * 2, device, zeroed=True)
        _launch.launch(
            kernel_poisson_band.project_bricks,
            dim=(nbk_f, nbk_f, nbk_f),
            inputs=[level_flags[level + 1][0], nbk_f, nbk_c],
            outputs=[projected],
            device=device,
        )
        _launch.launch(
            kernel_poisson_band.dilate_bricks,
            dim=(nbk_c, nbk_c, nbk_c),
            inputs=[projected, nbk_c],
            outputs=[grown],
            device=device,
        )
        level_flags[level] = (grown, (1 << level) + 1)
    return level_flags


def _compact_bands(
    flag_sets: list[tuple[wp.array[wp.int32], int]],
    device: wp.DeviceLike,
    *,
    tail: wp.array[wp.int32] | None = None,
) -> tuple[list[_BandLevel], list[int]]:
    """
    Compact each ``(flags, res)`` brick set into a ``_BandLevel``, scanning ``flags`` in place.

    The brick counts size the bands' storage, so they are read -- every set's in one copy, with
    ``tail[0]`` (a device count the caller wants at the same point) appended to the read and
    returned as the second item.
    """
    n_sets = len(flag_sets)
    totals = _launch.empty(n_sets + (0 if tail is None else 1), dtype=wp.int32, device=device)
    for k, (flags, _res) in enumerate(flag_sets):
        _launch.array_scan(flags, flags, inclusive=True)
        _launch.copy(totals, flags, dest_offset=k, src_offset=flags.size - 1, count=1)
    if tail is not None:
        _launch.copy(totals, tail, dest_offset=n_sets, count=1)
    values = [int(value) for value in read_values(totals, 0, totals.size)]
    # Every set's brick map in one allocation, and every set's brick coordinates in another.
    nbks = [(res + 7) // 8 for _flags, res in flag_sets]
    maps = _launch.carve(wp.int32, [nbk**3 for nbk in nbks], device)
    coords = _launch.carve(wp.vec3i, [max(n, 1) for n in values[:n_sets]], device)
    levels: list[_BandLevel] = []
    for (flags, res), n_bricks, nbk, brick_map, brick_coords in zip(
        flag_sets, values[:n_sets], nbks, maps, coords, strict=True
    ):
        _launch.launch(
            kernel_poisson_band.compact_bricks,
            dim=nbk**3,
            inputs=[flags, nbk],
            outputs=[brick_map, brick_coords],
            device=device,
        )
        levels.append(
            _BandLevel(
                res, nbk, brick_map, brick_coords, n_bricks * int(kernel_poisson_band.BRICK_NODES)
            )
        )
    return levels, values[n_sets:]


def _band_data(
    bands: dict[int, _BandLevel],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    cube_size: float,
    confidence: bool,
    missing: wp.array[wp.int32],
    device: wp.DeviceLike,
) -> dict[int, tuple[wp.array[wp.float32], wp.array[wp.float32]]]:
    """Every band level's ``(b, W)``: the finest splat's right-hand side, restricted down."""
    depth = max(bands)
    finest = bands[depth]
    n_nodes = finest.n_nodes
    splat = _launch.zeros(4 * n_nodes, dtype=wp.float32, device=device)
    vx, vy, vz, weights = (odt.as_dense(splat[k * n_nodes : (k + 1) * n_nodes]) for k in range(4))
    inv_cell = (finest.res - 1) / cube_size
    if is_deterministic() and wp.get_device(device).is_cuda:
        _splat_ordered(
            kernel_poisson_band.band_splat_keys,
            [
                points,
                cube_lower,
                wp.float32(inv_cell),
                finest.res,
                finest.brick_map,
                finest.nbk,
                n_nodes,
            ],
            points,
            normals,
            cube_lower,
            inv_cell,
            finest.res,
            confidence,
            n_nodes,
            [vx, vy, vz, weights],
        )
    else:
        _launch.launch(
            kernel_poisson_band.band_splat,
            dim=points.size,
            inputs=[
                points,
                normals,
                cube_lower,
                wp.float32(inv_cell),
                finest.res,
                wp.int32(1 if confidence else 0),
                finest.brick_map,
                finest.nbk,
            ],
            outputs=[vx, vy, vz, weights],
            device=device,
        )
    b = _launch.empty(n_nodes, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_poisson_band.band_rhs,
        dim=n_nodes,
        inputs=[finest.brick_coords, finest.brick_map, finest.nbk, finest.res, vx, vy, vz, weights],
        outputs=[b],
        device=device,
    )
    data = {depth: (b, weights)}
    for level in range(depth - 1, min(bands) - 1, -1):
        coarse = bands[level]
        coarse_b, coarse_w = _launch.carve(wp.float32, [coarse.n_nodes] * 2, device, zeroed=True)
        _restrict_band(
            bands[level + 1], *data[level + 1], coarse, coarse.res, coarse_b, coarse_w, missing
        )
        data[level] = (coarse_b, coarse_w)
    return data


def _splat_ordered(
    key_kernel: odt.Kernel,
    key_inputs: list[object],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: float,
    res: int,
    confidence: bool,
    n_nodes: int,
    outputs: list[wp.array[wp.float32]],
) -> None:
    """
    Splat the normals into the zeroed ``outputs`` (``V`` and ``W``) in one fixed order.

    The deterministic setting's form on CUDA (the CPU device's atomic splat adds in sample order
    on every call). ``key_kernel`` writes each sample's eight node keys (the dense grid's or a
    band's index, the sentinel ``n_nodes`` for none) with the term ``8 s + corner`` as payload; a
    stable sort lists each node's terms in ascending sample order, and
    ``kernels/reconstruction.sum_splat_chunks`` and ``sum_splat_runs`` add each node's run in
    fixed-size chunks of it, so a coarse level's long runs do not serialise.
    """
    device = points.device
    count = 8 * points.size
    keys = _launch.empty(2 * count, dtype=wp.int32, device=device)
    order = _launch.empty(2 * count, dtype=wp.int32, device=device)
    _launch.launch(
        key_kernel, dim=points.size, inputs=key_inputs, outputs=[keys, order], device=device
    )
    _launch.radix_sort_pairs(keys, order, count, end_bit=n_nodes.bit_length())
    # Chunks of 16 to 128 terms, about ``2**15`` of them: long chunks shorten the second pass's
    # walk over a coarse level's long runs, short ones keep a small cloud's first pass parallel.
    chunk = min(128, max(16, 1 << (count >> 15).bit_length()))
    n_chunks = -(-count // chunk)
    pieces = _launch.empty(2 * n_chunks, dtype=wp.vec4, device=device)
    heads, tails = odt.as_dense(pieces[:n_chunks]), odt.as_dense(pieces[n_chunks:])
    _launch.launch(
        kernel_reconstruction.sum_splat_chunks,
        dim=n_chunks,
        inputs=[
            keys,
            order,
            count,
            points,
            normals,
            lower,
            wp.float32(inv_cell),
            res,
            wp.int32(1 if confidence else 0),
            chunk,
        ],
        outputs=[heads, tails, *outputs],
        device=device,
    )
    _launch.launch(
        kernel_reconstruction.sum_splat_runs,
        dim=n_chunks,
        inputs=[keys, count, chunk, heads, tails],
        outputs=outputs,
        device=device,
    )


def _restrict_band(
    fine: _BandLevel,
    b: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    coarse: _BandLevel | None,
    res_c: int,
    out_b: wp.array[wp.float32],
    out_weights: wp.array[wp.float32],
    missing: wp.array[wp.int32],
) -> None:
    """
    ``P^T`` of ``fine``'s data into the band ``coarse``, or the dense ``res_c**3`` grid.

    With the deterministic setting on each target gathers its fine terms in a fixed order
    (``kernels/algorithms/poisson_band.restrict_band_ordered``) instead of the atomic scatter.
    """
    if is_deterministic():
        n_targets = res_c**3 if coarse is None else coarse.n_nodes
        _launch.launch(
            kernel_poisson_band.restrict_band_ordered,
            dim=max(fine.n_nodes, n_targets),
            inputs=[
                fine.brick_coords,
                fine.brick_map,
                fine.nbk,
                fine.res,
                b,
                weights,
                None if coarse is None else coarse.brick_coords,
                None if coarse is None else coarse.brick_map,
                0 if coarse is None else coarse.nbk,
                res_c,
                n_targets,
            ],
            outputs=[out_b, out_weights, missing],
            device=b.device,
        )
        return
    _launch.launch(
        kernel_poisson_band.restrict_band,
        dim=fine.n_nodes,
        inputs=[
            fine.brick_coords,
            fine.res,
            b,
            weights,
            None if coarse is None else coarse.brick_map,
            0 if coarse is None else coarse.nbk,
            res_c,
        ],
        outputs=[out_b, out_weights, missing],
        device=b.device,
    )


def _solve_band_level(
    band: _BandLevel,
    lap: float,
    weights: wp.array[wp.float32],
    screen: float,
    rhs: wp.array[wp.float32],
    solution: wp.array[wp.float32],
    inv_diag: wp.array[wp.float32],
    maxiter: int,
    tol: float,
    device: wp.DeviceLike,
) -> None:
    """
    Jacobi-preconditioned conjugate gradient on one band level, from ``solution`` in place.

    ``_solve_screened_poisson``'s Chronopoulos-Gear round with the band's row in place of the dense
    stencil and Jacobi in place of the V-cycle: the shared ``cg_seed`` / ``cg_coefficients`` /
    ``cg_update`` kernels, one recorded device loop, nothing read back.
    """
    n = band.n_nodes
    tile = int(kernel_cg.CG_TILE)
    span, blocks, fold = kernel_cg.cg_layout(n, od.linalg.CG_FOLD_MAX_BLOCKS)
    stride = blocks * span
    work = _cg_workspace(stride, blocks, [], 1, device)
    r, u, w, p, s = work.vectors
    partials, atol_sq, dots, coefficients = (
        work.partials,
        work.atol_sq,
        work.dots,
        work.coefficients,
    )
    gamma_old, alpha_old, gamma_new, alpha_new = work.recurrence
    state = work.state
    iterations = work.counts
    assert iterations is not None
    level = [band.brick_coords, band.brick_map, band.nbk, band.res, wp.float32(lap), weights]
    screen_f = wp.float32(screen)
    _launch.launch_tiled(
        kernel_poisson_band.band_cg_initial,
        dim=(blocks,),
        inputs=[n, span, *level, screen_f, rhs, solution, inv_diag],
        outputs=[r, u, p, s, partials],
        block_dim=tile,
        device=device,
    )
    _launch.launch_tiled(
        kernel_cg.cg_seed,
        dim=(1,),
        inputs=[wp.float64(tol * tol), wp.float64(0.0), blocks, partials],
        outputs=[atol_sq, gamma_new, alpha_new, iterations, state],
        block_dim=tile,
        device=device,
    )

    def round_() -> None:
        _launch.launch_tiled(
            kernel_poisson_band.band_cg_matvec_dots,
            dim=(blocks,),
            inputs=[n, span, *level, screen_f, r, u, gamma_new, alpha_new],
            outputs=[w, partials, gamma_old, alpha_old, state],
            block_dim=tile,
            device=device,
        )
        if not fold:
            _launch.launch_tiled(
                kernel_cg.cg_coefficients,
                dim=(1,),
                inputs=[1, maxiter, blocks, partials, gamma_old, alpha_old, atol_sq, state],
                outputs=[coefficients, gamma_new, alpha_new, dots, iterations],
                block_dim=tile,
                device=device,
            )
        _launch.launch_tiled(
            kernel_cg.CG_UPDATE[wp.float32],
            dim=(1, stride // tile),
            inputs=[
                stride,
                tile,
                n,
                1,
                maxiter,
                1 if fold else 0,
                blocks,
                1,
                partials,
                coefficients,
                gamma_old,
                alpha_old,
                atol_sq,
                inv_diag,
                w,
                p,
                s,
                r,
                u,
                state,
            ],
            outputs=[solution, u, gamma_new, alpha_new, dots, iterations],
            block_dim=tile,
            device=device,
        )

    run_device_loop(device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), round_)


def _cg_workspace(
    stride: int, blocks: int, extra: list[int], counts: int, device: wp.DeviceLike
) -> _CgWorkspace:
    """
    One conjugate-gradient solve's buffers, one allocation per dtype.

    The five ``float32`` vectors of ``stride`` entries, then a view per ``extra`` width (a
    preconditioner's vectors); the ``(3, 1, blocks)`` dot partials and the ten per-solve scalars
    in ``float64``; the loop state and ``counts`` more ``int32`` slots. Every entry is written
    before it is read, by the solve's initial launch or the seed.
    """
    widths = [stride] * 5 + extra
    vectors = _launch.carve(wp.float32, widths, device)
    reals = _launch.empty(3 * blocks + 10, dtype=wp.float64, device=device)
    scalars = _launch.view(reals, 3 * blocks, 3 * blocks + 10)
    ints = _launch.empty(kernel_array.LOOP_STATE_SIZE + counts, dtype=wp.int32, device=device)
    return _CgWorkspace(
        vectors[:5],
        vectors[5:],
        _launch.view(reals, 0, 3 * blocks).reshape((3, 1, blocks)),
        _launch.view(scalars, 0, 1),
        odt.as_array2d(_launch.view(scalars, 1, 3).reshape((2, 1)), wp.float64),
        odt.as_array2d(_launch.view(scalars, 3, 6).reshape((3, 1)), wp.float64),
        [_launch.view(scalars, row, row + 1) for row in range(6, 10)],
        _launch.view(scalars, 0, 2),
        _launch.view(ints, 0, kernel_array.LOOP_STATE_SIZE),
        _launch.view(ints, kernel_array.LOOP_STATE_SIZE, ints.size) if counts > 0 else None,
    )


class _CgWorkspace(NamedTuple):
    """``_cg_workspace``'s views."""

    vectors: list[wp.array[wp.float32]]
    """``r, u, w, p, s``."""
    extra: list[wp.array[wp.float32]]
    """One view per requested width."""
    partials: wp.array[wp.float64]
    """``(3, 1, blocks)`` per-block dot partials."""
    atol_sq: wp.array[wp.float64]
    """``(1,)`` squared threshold."""
    dots: odt.Array2dFloat64
    """``(2, 1)`` ``(r.r, gamma)``."""
    coefficients: odt.Array2dFloat64
    """``(3, 1)`` step coefficients."""
    recurrence: list[wp.array[wp.float64]]
    """``gamma_old, alpha_old, gamma_new, alpha_new``, each ``(1,)``."""
    residual_pair: wp.array[wp.float64]
    """``(2,)`` squared threshold and squared residual, the pair the caller tests."""
    state: wp.array[wp.int32]
    """``(kernel_array.LOOP_STATE_SIZE,)`` device loop state."""
    counts: wp.array[wp.int32] | None
    """The requested ``int32`` slots, ``None`` for none."""


@functools.cache
def _marching_cubes_table(device: str) -> wp.array[wp.int32]:
    """``kernels/levelset.MARCHING_CUBES_TABLE`` on the device ``device`` names, uploaded once."""
    return _launch.array(kernel_levelset.MARCHING_CUBES_TABLE, dtype=wp.int32, device=device)


def resample_uniform(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    voxel_size: float | None = None,
    offset: float = 0.0,
    sign_mode: Literal["parity", "winding"] = "winding",
    method: Literal["padded", "igl"] = "padded",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Rebuild a mesh by sampling its signed distance field on a uniform grid and re-extracting it.

    MeshLab's ``generate_resampled_uniform_mesh``. Three uses, all the same call:

    - **Regularize.** The output triangulation comes from the grid, not from the input, so every
      pathology of the input topology — self-intersections, non-manifold edges, duplicated or
      inverted faces, a soup — is simply not carried over. This is the bluntest repair there is and
      the one that always works.
    - **Offset.** A positive ``offset`` extracts the level set *outside* the surface (a dilation)
      and a negative one inside (an erosion): how shells, clearances and tool paths are built.
    - **Shrink-wrap.** A coarse ``voxel_size`` with ``offset=0`` gives a watertight envelope of a
      complicated input.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    voxel_size
        Grid spacing. Defaults to ``1 %`` of the bounding-box diagonal, which is a ~100-cell grid
        across the mesh. **Cost is cubic in the reciprocal**, so halving it is eight times the field
        evaluation; and no feature thinner than a voxel survives.
    offset
        Level set to extract, in world units. ``0`` (the default) reproduces the surface.
    sign_mode
        How the sign of the distance field is decided, forwarded to
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]. The default here is
        ``"winding"`` rather than that function's ``"parity"``: resampling is usually applied to a
        mesh that is *broken*, and ray parity has no principled answer through a hole, while the
        generalized winding number degrades gracefully.
    method
        The lattice. ``"padded"`` (the default) pads the bounding box by three voxels plus
        ``offset`` and keeps every vertex off the lattice nodes, then cleans the extraction (see
        ``Notes``). ``"igl"`` is ``igl.offset_surface``'s: ``round(longest_extent / voxel_size)``
        nodes along the box's longest axis, one of them padding at each end, the box grown by
        ``offset``, the lattice centred on it; the extraction is returned as marched, as igl does.
        Its spacing is therefore near ``voxel_size`` rather than equal to it.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` resampled vertex positions on ``vertices.device``.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat triangle index buffer over the ``n_out_faces`` output triangles.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, if ``method`` is not ``"padded"`` or ``"igl"``, or if
        under ``"igl"`` it leaves fewer than four nodes along the longest axis.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`marching_cubes`][ordito.levelset.marching_cubes]
    [`ordito.proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]
    [`ordito.remesh.cluster_decimate`][ordito.remesh.cluster_decimate]
    [`ordito.voxels.voxelize_mesh`][ordito.voxels.voxelize_mesh]

    Notes
    -----
    The grid is padded by three voxels beyond the bounding box *plus* ``offset``, so a dilated level
    set is never clipped by the lattice boundary and the extracted surface is always closed. That
    padding is why the memory cost is a little above ``(extent / voxel_size) ** 3``.

    Examples
    --------
    Re-extract a unit cube on a ``0.07`` grid, and a shell offset ``0.1`` outside it:

    ```python
    import numpy as np

    import ordito as od

    cube, cube_faces = od.creation.box()
    vertices, faces = od.reconstruction.resample_uniform(cube, cube_faces, voxel_size=0.07)
    print(od.validation.is_watertight(vertices, faces))  # True
    labels = od.adjacency.face_connected_component_labels(faces).numpy()
    print(np.unique(labels).shape[0])  # 1
    shell, _ = od.reconstruction.resample_uniform(cube, cube_faces, voxel_size=0.07, offset=0.1)
    print(f"{shell.numpy()[:, 0].max():.1f}")  # 0.6
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    lower, upper = od.bounds.aabb(vertices)
    # ``math.dist`` rather than ``float(wp.length(upper - lower))``: a Warp operator and a
    # Warp builtin at Python scope each route through builtin dispatch, several times dearer. It
    # computes in float64 where ``wp.length`` is float32, i.e. the correctly-rounded answer for
    # float32 corners.
    diagonal = math.dist(lower, upper)
    if voxel_size is None:
        voxel_size = 0.01 * diagonal
    if voxel_size <= 0.0:
        raise ValueError(f"resample_uniform requires voxel_size > 0, got {voxel_size}")
    if method not in ("padded", "igl"):
        raise ValueError(f"method must be 'padded' or 'igl', got {method!r}")
    if method == "igl":
        shape, bounds = _igl_offset_lattice(lower, upper, offset, voxel_size)
        return od.levelset.signed_distance_level_set(
            vertices, faces, offset, shape, bounds=bounds, sign_mode=sign_mode
        )

    # Three voxels of slack beyond the box and beyond the offset, so a dilated level set closes
    # inside the lattice instead of being cut off by it.
    pad = 3.0 * voxel_size + max(offset, 0.0)
    grid_lower = wp.vec3(*(component - pad for component in odt.vec3_floats(lower)))
    grid_upper = wp.vec3(*(component + pad for component in odt.vec3_floats(upper)))
    # The ``float32`` corners just built, as floats: the extents below are taken from the rounded
    # lattice the kernels see, not from the host values before rounding.
    grid_lower_f, grid_upper_f = odt.vec3_floats(grid_lower), odt.vec3_floats(grid_upper)
    # Carry the extents as plain ints and build the ``wp.vec3i`` once, where the launch needs it:
    # a ``wp.vec3i`` has to be unpacked with ``int(...)`` at every use anyway.
    n_x, n_y, n_z = (
        max(2, math.ceil((grid_upper_f[axis] - grid_lower_f[axis]) / voxel_size) + 1)
        for axis in range(3)
    )
    out_vertices, out_faces = od.levelset.signed_distance_level_set(
        vertices,
        faces,
        offset,
        (n_x, n_y, n_z),
        bounds=(grid_lower, grid_upper),
        sign_mode=sign_mode,
        edge_margin=kernel_levelset.lattice_edge_margin(voxel_size),
    )
    # The margin keeps every vertex off the lattice nodes. Without it a node landing on the level
    # set (a unit cube on a 0.05 grid puts whole planes of nodes on its faces) leaves slivers the
    # cleanup's degenerate-face filter deletes, and deleting them opens the surface, since their
    # neighbours do not share their vertices. Hole filling is off (``crit_hole_length=0``): a level
    # set of a signed field is already closed, so any hole here would be a symptom worth surfacing
    # rather than patching over.
    return _clean_reconstruction(out_vertices, out_faces, 0.0)


def _igl_offset_lattice(
    lower: wp.vec3, upper: wp.vec3, offset: float, voxel_size: float
) -> tuple[tuple[int, int, int], tuple[wp.vec3, wp.vec3]]:
    """
    ``igl::voxel_grid(V, offset, s, 1)``'s node counts and corner nodes, with ``s`` from a spacing.

    ``s = round(longest_extent / voxel_size)`` nodes along the longest axis of the vertex box,
    which is ``igl.offset_surface``'s ``s`` for that spacing. The box grows by ``offset``; its
    longest axis takes ``s - 2`` nodes and every other ``ceil`` of its share, one node of padding
    is added at each end, and the lattice, scaled by the axis that needs the widest spacing, is
    centred on the box. Host arithmetic in ``float64``, as igl forms it.
    """
    low = [component - offset for component in odt.vec3_floats(lower)]
    high = [component + offset for component in odt.vec3_floats(upper)]
    extents = [high[axis] - low[axis] for axis in range(3)]
    longest = max(range(3), key=lambda axis: (extents[axis], -axis))
    nodes = round(
        max(odt.vec3_floats(upper)[a] - odt.vec3_floats(lower)[a] for a in range(3)) / voxel_size
    )
    if nodes < 4:
        raise ValueError(
            f"method='igl' needs at least 4 nodes along the longest axis, got {nodes}; "
            "pass a smaller voxel_size"
        )
    inner = nodes - 2
    side = [
        inner if axis == longest else math.ceil(inner * extents[axis] / extents[longest])
        for axis in range(3)
    ]
    side = [count + 2 for count in side]
    stretch = [extents[axis] / (1.0 - 2.0 / (side[axis] - 1.0)) for axis in range(3)]
    widest = max(range(3), key=lambda axis: (stretch[axis], -axis))
    span = [stretch[widest] * (side[axis] - 1.0) / (side[widest] - 1.0) for axis in range(3)]
    centre = [0.5 * (low[axis] + high[axis]) for axis in range(3)]
    grid_lower = wp.vec3(*(centre[axis] - 0.5 * span[axis] for axis in range(3)))
    grid_upper = wp.vec3(*(centre[axis] + 0.5 * span[axis] for axis in range(3)))
    return (side[0], side[1], side[2]), (grid_lower, grid_upper)


def ball_pivoting(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None = None,
    *,
    radius: float = 0.0,
    clustering: float = 0.2,
    crease_angle: float = math.pi / 2.0,
    max_waves: int = 0,
    crit_hole_length: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """

    Ball-pivoting surface reconstruction from an oriented point cloud (interpolating).

    GPU-native, wave-parallel port of the ball-pivoting algorithm (PyMeshLab's
    ``generate_surface_reconstruction_ball_pivoting``, ``open3d``'s
    ``create_from_point_cloud_ball_pivoting``, vcglib BPA). A ball of the given ``radius`` is rolled
    over the oriented points; wherever it rests on three points without containing another point it
    forms a triangle, and it then pivots around each boundary edge of the growing front to add the
    next triangle. The output mesh **interpolates** the input points (its vertices are a subset of
    ``points``), unlike the implicit
    [`screened_poisson`][ordito.reconstruction.screened_poisson].

    The advancing front is a **persistent** device-side structure — an edge hash table plus a
    compacted list of its boundary edges, mutated in place as triangles commit. Each wave pivots
    every live front edge independently through a hash-grid ball query and commits a conflict-free
    independent set via a two-phase vertex claim. Manifoldness is protected by an interior-edge
    guard (an edge already shared by two triangles never gains a third). Every triangle is wound so
    its normal agrees with the point normals: a seed triangle by its vertices' normals, every later
    one consistently with the triangle it pivots from. The result is passed through the standard
    cleanup tail (deduplicate, drop degenerate / non-manifold faces, optionally fill small holes),
    which keeps that winding.


    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device (``n >= 3``).
    normals
        ``(n,)`` oriented unit normals. When ``None``, they are estimated by PCA oriented
        outward from the centroid ([`estimate_normals`][ordito.points.estimate_normals]; valid for
        star-shaped clouds only).
    radius
        Ball radius. When ``<= 0``, it is auto-guessed as ``1.5 x`` the mean nearest-neighbour
        spacing, summed in a fixed order so that the guess repeats from call to call.
    clustering
        Candidate points closer than ``clustering * radius`` to an edge endpoint are rejected
        (vcglib clustering, as a fraction of ``radius``).
    crease_angle
        Maximum dihedral angle (radians) between a new triangle and the one it pivots from; larger
        folds are rejected. Use ``>= pi`` to disable the crease guard.
    max_waves
        Maximum number of parallel waves, where each wave either pivots the current front or seeds
        orphan triangles. ``0`` picks a generous safety cap from the point count.
    crit_hole_length
        Boundary loops with perimeter at most this value are filled at the end. ``0`` disables hole
        filling; a negative value defaults to ``0.1 x`` the bounding-box diagonal.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(m,)`` referenced input points, compacted from index zero, on ``points.device``,
        ``m <= n``.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer, wound consistently with ``normals`` (outward
        when they point outward).

    Raises
    ------
    ValueError
        If fewer than 3 points are given.
    RuntimeError
        If ``points`` and ``normals`` are not all on one device.

    See Also
    --------
    [`screened_poisson`][ordito.reconstruction.screened_poisson]
    [`triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud]
    [`estimate_normals`][ordito.points.estimate_normals]

    Notes
    -----
    A single radius is used (the classic multi-radius schedule is a documented follow-up), and the
    triangle budget grows on demand rather than raising.

    The triangulation is **reproducible at an explicit ``radius``**: repeated runs on one device and
    build return the same set of triangles, because a wave resolves competing proposals by a key
    packed from their own vertices and the wave counter rather than by the order they reached an
    atomic counter -- device state the loop advances deterministically, so the ordering varies
    between waves (which keeps a front edge from being starved by a globally fixed key) without
    varying between runs; the default ``radius <= 0`` guesses its radius with a fixed-order sum, so
    it is covered too. One caveat on how far that reaches: the face buffer's *row order* is not
    pinned -- ``commit_triangles`` appends with a ``wp.atomic_add`` -- so compare reconstructions as
    a set of wound triangles, not buffer-to-buffer.

    Results are not comparable across devices, where floating-point contraction differs.

    A point closer to a ball's sphere than single precision can resolve is placed in double
    precision, and a ball resting on four or more points at once -- the corners of every cell of a
    regular grid -- is resolved by a fixed rule on the points' indices rather than by rounding. So
    four points sampled co-circular, exactly or nearly, get exactly one triangulation, consistently
    wound, on every device.

    The result is **interpolating and edge-manifold**, and on a densely, uniformly sampled closed
    surface it is watertight in practice: a subdivided icosphere reconstructs to exactly ``2 v - 4``
    faces with no boundary edge at all. Watertightness is not *guaranteed* -- the wave-parallel
    front has no per-vertex fan structure to align colliding sheets with, so pathological sampling
    can still leave a seam. Regions sampled too sparsely for the ball to rest are left as holes by
    construction (widen ``radius`` or set ``crit_hole_length``); for an implicit surface that is
    watertight by construction use [`screened_poisson`][ordito.reconstruction.screened_poisson].

    Examples
    --------
    A ball a little wider than the spacing of a sphere's vertices rolls over all of them:

    ```python
    import numpy as np

    import ordito as od

    points, _ = od.creation.icosphere(subdivisions=3)
    vertices, faces = od.reconstruction.ball_pivoting(points, points, radius=0.15)
    print(od.validation.is_watertight(vertices, faces))  # True
    labels = od.adjacency.face_connected_component_labels(faces).numpy()
    print(np.unique(labels).shape[0])  # 1
    ```
    """
    require_same_device(points=points, normals=normals)
    device = points.device
    n = points.size
    if n < 3:
        return _launch.clone(points), _launch.empty(0, dtype=wp.int32, device=device)

    # Nearest-neighbour table drives both the radius auto-guess and (if needed) normal estimation.
    neighbor_idx, neighbor_dist = od.neighbors.query_nearest(points, points, k=7)
    if radius <= 0.0:
        # The whole table, not columns ``1:``: the self-distance in slot 0 is exactly zero and the
        # positive-finite filter drops it, so flattening costs nothing and keeps the reduction on
        # the device -- a column slice would be strided and could not be flattened at all.
        mean_spacing = _mean_positive_finite(neighbor_dist.flatten())
        radius = 1.5 * (mean_spacing if mean_spacing is not None else 1.0)
    if normals is None:
        normals = od.points.estimate_normals(points, neighbor_idx, orientation="centroid")

    # Cell width equal to the ball radius, not to the ``2 * radius`` neighbourhood the pivot
    # searches: the inner empty-ball test is by far the most frequent query, and a cell twice its
    # radius makes it enumerate roughly 8x the points it needs. Going finer than this loses more to
    # cell-probe overhead than it saves in point tests.
    grid = od.neighbors.hashgrid_from_points(points, radius)
    # The pivot search walks this cooperatively (one warp an edge); the empty-ball test inside
    # it stays on the hash grid, which is the better structure for a *serial* per-lane query.
    bvh = od.neighbors.bvh_from_points(points)
    crease_cos = math.cos(crease_angle) if crease_angle < math.pi else -1.0

    state = _BpaState(points, normals, grid, bvh, radius, clustering, crease_cos, 4 * n + 16)
    count = _bpa_run(state, max_waves if max_waves > 0 else 16 * n)
    # A prefix view, not a copy: nothing writes the face buffer after the run, and the cleanup
    # below returns faces of its own, so the view never reaches the caller.
    faces = (
        odt.as_dense(state.all_faces[: count * 3])
        if count > 0
        else _launch.empty(0, dtype=wp.int32, device=device)
    )
    # Orientation is not re-derived (``orient=False``): every triangle is already wound from the
    # normals, and ``make_normals_outward`` would rewind an inward-normal cloud's mesh outward.
    return _clean_reconstruction(points, faces, crit_hole_length, orient=False)


class _BpaState:
    """
    Every buffer a ball-pivoting run touches, allocated once and mutated in place.

    The wave loop does no allocation and no host synchronisation, which is what keeps it cheap.
    """

    def __init__(
        self,
        points: wp.array[wp.vec3],
        normals: wp.array[wp.vec3],
        grid: wp.HashGrid,
        bvh: wp.Bvh,
        radius: float,
        clustering: float,
        crease_cos: float,
        max_faces: int,
    ) -> None:
        """Allocate the persistent state for a cloud of ``points`` and a triangle budget."""
        self.points = points
        self.normals = normals
        self.grid = grid
        self.bvh = bvh
        self.radius = wp.float32(radius)
        self.clustering = wp.float32(clustering)
        self.crease_cos = wp.float32(crease_cos)
        self.device = points.device
        self.n = points.size
        self.key_base = wp.uint64(self.n)

        # One zeroed allocation holds the counters and, after them, the node bitset of the point
        # BVH's leaves too large for ``pivot_front_edges``' cooperative walk
        # (``kernel_bpa.mark_overfull_point_leaves``); a binary tree over ``n`` points has fewer
        # than ``2 * n`` nodes. ``counters`` is a prefix view, so its readbacks stay small.
        #
        # The bitset is filled only where a block runs more than one lane
        # (``prefers_tiled_reduction``: CUDA, or the CPU under ``wp.config.enable_cpu_blocks``).
        # On the plain cpu device the cooperative walk is a serial query with no result buffer to
        # overrun, so the bitset stays zero there and every pivot search takes the builtin.
        n_nodes = 2 * self.n
        state_words = _launch.zeros(
            kernel_bpa.BPA_COUNTERS + (n_nodes + 31) // 32, dtype=wp.int32, device=self.device
        )
        self.counters = odt.as_dense(state_words[: kernel_bpa.BPA_COUNTERS])
        self.overfull = odt.as_dense(state_words[kernel_bpa.BPA_COUNTERS :])
        if prefers_tiled_reduction(self.device):
            _launch.launch(
                kernel_bpa.mark_overfull_point_leaves,
                dim=n_nodes,
                inputs=[bvh.id, self.overfull],
                device=self.device,
            )
        _launch.fill_(self.counters[kernel_bpa.CNT_SEEDING : kernel_bpa.CNT_SEEDING + 1], 1)
        # ``seed_failed`` persists across every seeding wave for the run's whole lifetime -- see
        # ``kernel_bpa.seed_triangles`` for why a point that once exhausted its candidates without
        # seeding can never succeed later (its candidate set only shrinks), which is what makes
        # never resetting this safe.
        self.point_used, self.seed_failed = _launch.carve(
            wp.bool, [self.n] * 2, self.device, zeroed=True
        )
        # The orphan each point last deferred to, or whether it can ever seed; see the same kernel.
        self.seed_blocker = _launch.full(
            self.n, int(kernel_bpa.SEED_UNKNOWN), dtype=wp.int32, device=self.device
        )
        self.boundary_degree = _launch.zeros(self.n, dtype=wp.int32, device=self.device)
        # ``uint64`` because the per-wave vertex claim is a ``wp.atomic_min`` over a *packed vertex
        # pair* rather than over a proposal index — see ``kernel_bpa.proposal_key``, which is what
        # makes a run reproducible. Doubling one n-sized buffer is the whole memory cost.
        self.owner = _launch.empty(self.n, dtype=wp.uint64, device=self.device)
        self._allocate_budget(max_faces)

    def _allocate_budget(self, max_faces: int) -> None:
        """(Re)size everything that scales with the triangle budget."""
        self.max_faces = max_faces
        # A triangle soup of ``max_faces`` faces has at most ``3 * max_faces`` distinct edges, and
        # the insert probe does not terminate on a full table, so the hash is sized for a load
        # factor of at most 1/2. The front and the proposal list share the same bound.
        capacity = 1 << (6 * max_faces - 1).bit_length()
        front_capacity = 3 * max_faces + self.n
        self.edge_mask = wp.int32(capacity - 1)
        self.edge_key = _launch.zeros(capacity, dtype=wp.uint64, device=self.device)
        # The zeroed ``int32`` tables in one allocation, and the unfilled ones in another.
        self.edge_count, self.edge_state = _launch.carve(
            wp.int32, [capacity] * 2, self.device, zeroed=True
        )
        unfilled = [max_faces * 3] + [capacity] * 4 + [front_capacity] * 5
        (
            self.all_faces,
            self.edge_src,
            self.edge_tgt,
            self.edge_opp,
            self.edge_cand,
            self.front_in,
            self.front_out,
            self.tri_a,
            self.tri_b,
            self.tri_c,
        ) = _launch.carve(wp.int32, unfilled, self.device)
        self._bind_edge_table()

        # Grid-stride bounds: fixed launch dimensions, which a captured graph requires. Sized from
        # the cloud, not from the budget — the live front peaks well below the point count, and a
        # fixed 64k-wide launch spent most of a small mesh's wave scheduling no-op threads.
        self.front_capacity = front_capacity
        self.front_grid = min(front_capacity, max(_BPA_MIN_GRID, self.n))
        self.claim_grid = self.front_grid

    def _bind_edge_table(self) -> None:
        """Rebuild the ``BpaEdgeTable`` view of the edge arrays: once here, never per launch."""
        # Nine of the wave kernels' arguments live in here, so binding these once here rather than
        # per launch avoids re-marshalling them across the whole wave loop. Call this after
        # anything that *reallocates* an edge array, which is only ``grow``.
        table = kernel_bpa.BpaEdgeTable()
        table.key = self.edge_key
        table.count = self.edge_count
        table.src = self.edge_src
        table.tgt = self.edge_tgt
        table.opp = self.edge_opp
        table.state = self.edge_state
        table.cand = self.edge_cand
        table.mask = self.edge_mask
        table.key_base = self.key_base
        self.edge_table = table

    def grow(self) -> None:
        """
        Double the triangle budget in place, preserving the run in progress.

        Slot indices move when the edge table is rehashed, so the front list is rebuilt from the
        new table rather than remapped.
        """
        old = (
            self.edge_key,
            self.edge_count,
            self.edge_src,
            self.edge_tgt,
            self.edge_opp,
            self.edge_state,
            self.edge_cand,
        )
        old_capacity = self.edge_key.size
        old_faces = self.all_faces
        old_count = min(int(read_scalar(self.counters, kernel_bpa.CNT_FACE)), self.max_faces)

        self._allocate_budget(2 * self.max_faces)
        if old_count > 0:
            # The guard is the API's, not the algorithm's: ``wp.copy`` reads ``count=0`` as "copy
            # the entire source" for backwards compatibility, so a budget grown before the first
            # triangle commits would copy the whole stale buffer instead of nothing. In bounds
            # (the destination is twice the size) and unreachable today, since the initial budget
            # is ``4 n + 16`` and a wave proposes at most one triangle per front edge -- but the
            # count is derived from device state, which is exactly the shape that rule is about.
            _launch.copy(self.all_faces, old_faces, count=old_count * 3)
        _launch.launch(
            kernel_bpa.rehash_edges,
            dim=old_capacity,
            inputs=[
                *old,
                self.edge_mask,
                self.edge_key,
                self.edge_count,
                self.edge_src,
                self.edge_tgt,
                self.edge_opp,
                self.edge_state,
                self.edge_cand,
            ],
            device=self.device,
        )
        # The over-increment from the losing threads of the overflowing wave never wrote a face.
        counters = self.counters.numpy()
        counters[kernel_bpa.CNT_FACE] = old_count
        counters[kernel_bpa.CNT_NEXT_FRONT] = 0
        counters[kernel_bpa.CNT_GROW] = 0
        self.counters.assign(counters)
        _launch.launch(
            kernel_bpa.collect_front_from_table,
            dim=self.edge_key.size,
            inputs=[wp.int32(self.front_capacity), self.counters, self.edge_table],
            outputs=[self.front_in],
            device=self.device,
        )
        self._adopt_next_front()

    def compact(self) -> None:
        """Drop closed and retired edges from the front list."""
        _launch.zero_(self.counters[kernel_bpa.CNT_NEXT_FRONT : kernel_bpa.CNT_NEXT_FRONT + 1])
        _launch.launch(
            kernel_bpa.compact_front,
            dim=self.front_grid,
            inputs=[
                self.front_in,
                self.front_grid,
                wp.int32(self.front_capacity),
                self.counters,
                self.edge_table,
                self.front_out,
            ],
            device=self.device,
        )
        self.front_in, self.front_out = self.front_out, self.front_in
        self._adopt_next_front()

    def _adopt_next_front(self) -> None:
        _launch.copy(
            self.counters[kernel_bpa.CNT_FRONT : kernel_bpa.CNT_FRONT + 1],
            self.counters[kernel_bpa.CNT_NEXT_FRONT : kernel_bpa.CNT_NEXT_FRONT + 1],
        )


def _mean_positive_finite(values: odt.ArrayNdFloat32) -> float | None:
    """
    Mean of the strictly positive finite entries of ``values``, or ``None`` if there are none.

    The spacing estimator both auto-guessing call sites in this module share: a neighbour-distance
    table carries a zero per self-match and an ``inf`` per unfilled slot, and neither belongs in a
    mean spacing. Reduces on the device rather than reading the whole table back, since the buffer
    scales with the cloud, and folds the sum and the count in one pass into one buffer, so the
    answer costs one launch and one readback.
    """
    device = values.device
    n = values.size
    if n == 0:
        return None
    blocks = kernel_reduce.blocks_1d(n)
    sum_and_count, stride = _launch.block_partials(blocks, 2, wp.float64, device)
    _launch.launch_tiled(
        kernel_reconstruction.positive_finite_sum_and_count,
        dim=[blocks],
        inputs=[values, stride, sum_and_count],
        block_dim=TILE_1D,
        device=device,
    )
    total, count = (float(x) for x in read_values(sum_and_count, 0, 2))
    if count == 0.0:
        return None
    return total / count


# Floor on the launch width of the grid-strided wave kernels, for clouds too small to fill the
# device on their own.
_BPA_MIN_GRID = 1 << 12


# Waves queued between host synchronisations. The wave loop is device-driven — ``end_wave`` keeps
# the seeding flag, the progress test and the continue flag in ``counters`` — so the host only ever
# needs to look in order to *stop*, and it can queue a batch and let the device run ahead. A wave
# that runs after the flag clears costs five no-op launches, which is far less than a sync.
#
# ``wp.capture_while`` is not used here: its conditional-graph per-iteration overhead is larger
# than the sync it would replace, because a batch already amortises the sync over eight waves.
_BPA_WAVES_PER_BATCH = 8

# Host round-trips beyond the batching are only taken to grow the budget (doubling, so
# logarithmically many) or to compact a front that ``pivot_front_edges`` left sparse (which needs
# the front to have grown 4x against its live count, so also rare).
_BPA_MAX_BATCHES = 1 << 16


def _bpa_run(state: _BpaState, max_waves: int) -> int:
    """
    Drive the wave loop until the front and the orphan set are both exhausted; return the faces.

    The face count comes off the last batch's counter readback.

    The host is woken once per batch, and only acts when the device asks it to — to grow the
    triangle budget, to compact a sparse front, or to stop.

    The invariant every batch boundary restores is that **``front_in`` holds the live front**: a
    wave reads ``front_in`` and writes ``front_out``, so one host swap per wave hands the result to
    the next one. The batch is queued blind, though, and the device may stop partway through it —
    every wave kernel opens with ``if counters[CNT_CONTINUE] == 0: return`` — so the trailing waves
    of a batch write nothing while the loop below swaps for them regardless. An odd number of those
    leaves the pair exchanged, and ``compact()`` then reads the buffer the *previous* wave wrote
    (in the first batch, one that ``wp.empty`` never wrote at all). ``CNT_WAVE`` counts only the
    waves that ran, and the batch's counter readback is already paid for, so the correction is a
    parity test on a number the host is holding.
    """
    _launch.fill_(state.counters[kernel_bpa.CNT_CONTINUE : kernel_bpa.CNT_CONTINUE + 1], 1)
    waves_run = 0
    # A batch that follows one the device kept going through starts from the reset its last
    # ``end_wave`` did, so only the first batch and one resuming a loop the host stopped open with
    # ``begin_wave``.
    resumed = True
    for _ in range(_BPA_MAX_BATCHES):
        for wave in range(_BPA_WAVES_PER_BATCH):
            _bpa_wave(state, max_waves, begin=resumed and wave == 0)
            state.front_in, state.front_out = state.front_out, state.front_in
        resumed = False
        counters = state.counters.numpy()
        no_op_waves = _BPA_WAVES_PER_BATCH - (int(counters[kernel_bpa.CNT_WAVE]) - waves_run)
        waves_run = int(counters[kernel_bpa.CNT_WAVE])
        if no_op_waves % 2:
            state.front_in, state.front_out = state.front_out, state.front_in
        if counters[kernel_bpa.CNT_CONTINUE]:
            continue
        if counters[kernel_bpa.CNT_DONE] or counters[kernel_bpa.CNT_WAVE] >= max_waves:
            return int(counters[kernel_bpa.CNT_FACE])
        if counters[kernel_bpa.CNT_GROW]:
            state.grow()
        else:
            state.compact()
        _launch.fill_(state.counters[kernel_bpa.CNT_CONTINUE : kernel_bpa.CNT_CONTINUE + 1], 1)
        resumed = True
    # Out of batches: the counters have moved since the last read.
    return int(read_scalar(state.counters, kernel_bpa.CNT_FACE))


def _bpa_wave(state: _BpaState, max_waves: int, *, begin: bool) -> None:
    """
    Queue one wave: seed or pivot, claim, commit, advance. No allocations, no readbacks.

    ``begin`` opens a batch with the counter reset; otherwise the previous wave's ``end_wave``
    has done it already.
    """
    device = state.device
    if begin:
        _launch.launch(kernel_bpa.begin_wave, dim=1, inputs=[state.counters], device=device)
    _launch.launch(
        kernel_bpa.seed_triangles,
        dim=state.n,
        inputs=[
            state.points,
            state.normals,
            state.point_used,
            state.seed_failed,
            state.seed_blocker,
            state.grid.id,
            state.radius,
            state.clustering,
            wp.int32(state.front_capacity),
            state.counters,
            state.owner,
            state.tri_a,
            state.tri_b,
            state.tri_c,
        ],
        device=device,
    )
    _launch.launch_tiled(
        kernel_bpa.pivot_front_edges,
        dim=[state.front_grid],
        block_dim=kernel_bpa.BPA_PIVOT_BLOCK,
        inputs=[
            state.points,
            state.normals,
            state.grid.id,
            state.bvh.id,
            state.overfull,
            state.radius,
            state.clustering,
            state.crease_cos,
            state.point_used,
            state.boundary_degree,
            state.front_in,
            state.front_grid,
            wp.int32(state.front_capacity),
            state.counters,
            state.edge_table,
            state.owner,
            state.front_out,
            state.tri_a,
            state.tri_b,
            state.tri_c,
        ],
        device=device,
    )
    _launch.launch(
        kernel_bpa.claim_triangle_vertices,
        dim=state.claim_grid,
        inputs=[
            state.tri_a,
            state.tri_b,
            state.tri_c,
            state.claim_grid,
            wp.int32(state.front_capacity),
            state.counters,
            state.owner,
        ],
        device=device,
    )
    _launch.launch(
        kernel_bpa.commit_triangles,
        dim=state.claim_grid,
        inputs=[
            state.tri_a,
            state.tri_b,
            state.tri_c,
            state.owner,
            wp.int32(state.max_faces),
            state.claim_grid,
            state.point_used,
            state.boundary_degree,
            wp.int32(state.front_capacity),
            state.counters,
            state.edge_table,
            state.front_out,
            state.all_faces,
        ],
        device=device,
    )
    _launch.launch(
        kernel_bpa.end_wave, dim=1, inputs=[wp.int32(max_waves), state.counters], device=device
    )


def _clean_reconstruction(
    points: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    crit_hole_length: float,
    *,
    orient: bool = True,
    away_from: wp.vec3 | None = None,
    deduplicate: bool = True,
    nondegenerate: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Shared reconstruction cleanup: dedup, drop degenerate/non-manifold, orient, fill small holes.

    Duplicate faces go first, then degenerate ones (which also compacts the vertex set), then faces
    on non-manifold edges so the result is edge-manifold — required before boundary extraction,
    since both hole filling and ``boundary_loops`` assume a manifold boundary. ``crit_hole_length``
    follows the public convention: ``0`` skips hole filling, a negative value means ``0.1 x`` the
    point-cloud bounding-box diagonal. ``orient=False`` keeps the incoming winding for callers
    that already have a trusted orientation. ``away_from`` replaces the outward orientation with
    ``_face_away_from``'s, which also decides open components. ``deduplicate=False`` skips the
    duplicate-resolution stage for a caller whose faces already carry pairwise distinct vertex
    sets *in ascending unoriented-key order* -- the order that stage emits -- so the result is
    unchanged. ``nondegenerate=True`` skips the degenerate-face test for a caller whose faces hold
    no degenerate face.
    """
    if faces.size == 0:
        return _launch.clone(points), faces

    if deduplicate:
        faces, _ = od.repair.resolve_duplicated_faces(faces, n_vertices=points.size)
    if nondegenerate:
        # The combined filter's answer: with no degenerate face its first stage only compacts the
        # vertices, which commutes with the manifold filter. That compacts them only when it drops
        # a face, so the referenced vertices are compacted here otherwise.
        vertices, faces = od.repair.remove_non_manifold_faces(points, faces)
        if vertices is points:
            vertices, faces, _ = od.repair.remove_unreferenced_vertices(points, faces)
    else:
        # One vertex compaction for both filters, rather than one after each: the faces here are
        # this package's own, so their indices are in range, which the combined filter requires.
        vertices, faces = od.repair.remove_degenerate_and_non_manifold_faces(points, faces)

    if faces.size > 0:
        mates: wp.array[wp.int32] | None = None
        if away_from is not None:
            faces, mates = _face_away_from(vertices, faces, away_from)
        elif orient:
            faces = od.repair.make_normals_outward(vertices, faces)
        if crit_hole_length != 0.0:
            hole_length = crit_hole_length
            if hole_length < 0.0:
                hole_length = 0.1 * od.bounds.enclosing_diagonal(points)
            # The faces are this function's own, so a mesh with nothing to fill keeps them rather
            # than copying them, and their indices are below ``vertices.size`` (compacted above).
            faces = od.holes._fill_small(  # pyright: ignore[reportPrivateUsage]
                vertices, faces, hole_length, mates=mates, n_vertices=vertices.size
            )
    return vertices, faces


def _face_away_from(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], center: wp.vec3
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Wind each component consistently, facing away from ``center`` at its farthest corner.

    The orientation of a cloud without normals: each component's corner farthest from the box
    centre, the lowest vertex on a tie, keeps a normal pointing away from the centre, and the rest
    of the component follows it. Unlike an outward
    test by signed volume it decides an open component too; on a closed one it agrees, since the
    surface lies inside the sphere through its farthest point. Nonempty ``faces``. Returns the
    oriented faces and their halfedge mates (a ``-2`` code need not name the lowest halfedge),
    moved with each flipped face, for the hole fill to read rather than pair again.
    """
    n_faces = faces.size // 3
    device = faces.device
    # ``make_winding_consistent`` with the wound faces' halfedge mates, which the labelling reads
    # rather than pairing the halfedges again.
    faces, mates = od.repair._wound_with_mates(faces, vertices.size)  # pyright: ignore[reportPrivateUsage]
    labels = od.adjacency.face_connected_component_labels(
        faces, n_vertices=vertices.size, mates=mates
    )
    farthest = _launch.full(n_faces, -1, dtype=wp.int64, device=device)
    _launch.launch(
        kernel_reconstruction.farthest_component_corners,
        dim=n_faces,
        inputs=[vertices, faces, labels, center],
        outputs=[farthest],
        device=device,
    )
    # A component can hold every face, so ``n_faces`` terms bound any one sum.
    sums = _launch.group_sums(n_faces, wp.float32, n_faces, device, fused=True)
    _launch.launch_group_sums(
        kernel_reconstruction.scatter_farthest_corner_normal,
        dim=n_faces,
        inputs=[vertices, faces, labels, center, farthest],
        sums=sums,
        device=device,
    )
    out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    out_mates = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.flip_faces_and_mates_by_component_sign,
        dim=n_faces,
        inputs=[faces, mates, labels, sums.out, sums.launch_pass],
        outputs=[out_faces, out_mates],
        device=device,
    )
    return out_faces, out_mates
