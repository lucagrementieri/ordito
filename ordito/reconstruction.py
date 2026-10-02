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

import math
import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device, run_device_loop
from ordito.constants import TILE_1D
from ordito.kernels import array as kernel_array
from ordito.kernels import reconstruction as kernel_reconstruction
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import remesh as kernel_remesh
from ordito.kernels.algorithms import ball_pivoting as kernel_bpa
from ordito.kernels.algorithms import conjugate_gradient as kernel_cg

if TYPE_CHECKING:
    # Type-checking only: the adaptive-backend helpers import ``warp.fem`` lazily (inside the
    # functions) so ``import ordito`` never pays its tens-of-seconds first-call codegen unused.
    import numpy.typing as npt
    import warp.fem as fem


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
    """
    device = points.device
    n = points.size
    if n < 3:
        raise ValueError(f"delaunay_triangulation requires at least 3 points, got {n}")

    faces_np = _lexicographic_triangulation(points)
    if faces_np.shape[0] == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    faces = _launch.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device)

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

    od.remesh._flip_interior_edges(faces, n, launch, max_iter)  # pyright: ignore[reportPrivateUsage]
    return faces


def _lexicographic_triangulation(points: wp.array[wp.vec2]) -> np.ndarray:
    """
    Sequential lexicographic incremental triangulation (``igl::lexicographic_triangulation``).

    Returns an ``(n_faces, 3)`` ``int32`` array of CCW triangles over the input point indices,
    or an empty ``(0, 3)`` array when the points are collinear. The output is not yet Delaunay —
    the caller flips it to Delaunay on device.

    The sweep itself runs in
    [`lexicographic_triangulation`][ordito.kernels.reconstruction.lexicographic_triangulation]
    on the CPU device, single-threaded; see that kernel for why. Only the lex sort stays in
    NumPy, where it is one vectorised call.
    """
    # See ``boundary._unoriented_boundary_cycles``: ``wp.array.numpy()`` has no return annotation
    # and pyright infers an empty shape tuple for it. The cast names the ``(n, 2)`` float64 buffer
    # the ``.astype`` produces.
    points_np = cast("npt.NDArray[np.float64]", points.numpy().astype(np.float64))
    n = points_np.shape[0]
    order_np = np.lexsort((points_np[:, 1], points_np[:, 0])).astype(np.int32)

    # A triangulation of n points has 2n - 2 - h <= 2n - 5 triangles; 2n is the guard capacity.
    max_faces = 2 * n
    points_cpu = _launch.array(np.ascontiguousarray(points_np), dtype=wp.vec2d, device="cpu")
    order_cpu = _launch.array(order_np, dtype=wp.int32, device="cpu")
    boundary = _launch.empty(n + 1, dtype=wp.int32, device="cpu")
    boundary_next = _launch.empty(n + 1, dtype=wp.int32, device="cpu")
    orientations = _launch.empty(n, dtype=wp.float64, device="cpu")
    faces_cpu = _launch.empty(3 * max_faces, dtype=wp.int32, device="cpu")
    counts = _launch.zeros(2, dtype=wp.int32, device="cpu")

    _launch.launch(
        kernel_reconstruction.lexicographic_triangulation,
        dim=1,
        inputs=[
            points_cpu,
            order_cpu,
            wp.int32(max_faces),
            boundary,
            boundary_next,
            orientations,
            faces_cpu,
            counts,
        ],
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
        return np.empty((0, 3), dtype=np.int32)
    return faces_cpu.numpy()[: 3 * written].reshape(-1, 3)


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
    optimised toward a Delaunay-like fan (``build_local_triangulations``); triangles that recur in
    two or three of these local triangulations are kept and assembled into a triangle-soup mesh
    (vertices are the input points). Non-manifold and degenerate faces are dropped and small
    boundary holes are filled.

    Orientation relies on **trusted normals** rather than on a sequential propagation pass, which
    is what keeps the assembly parallel: supply ``normals`` for arbitrary geometry, or leave them
    ``None`` to estimate them by PCA oriented outward from the cloud centroid (valid for
    star-shaped clouds only — see [`estimate_normals`][ordito.points.estimate_normals]).

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device.
    normals
        ``(n,)`` oriented unit normals. When ``None``, normals are estimated.
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
    The per-point fan is bounded by ``max_neighbours`` and the search radius is never grown
    automatically, so very sparse or highly non-uniform clouds may leave extra boundary holes.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
    [`ordito.repair.remove_degenerate_faces`][]
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
        crit_hole_length = 0.1 * _box_diagonal(bounds)
    # Dense (n, k+1) nearest-neighbour table; slot 0 is the point itself and is skipped in-kernel.
    neighbor_idx, neighbor_dist = od.neighbors.query_nearest(points, points, k=k + 1, bounds=bounds)

    if normals is None:
        normals = od.points.estimate_normals(points, neighbor_idx)

    # Per-point local fan triangulation: point ``v``'s fan fills ``out_tris[v, :count]``.
    out_tris = _launch.empty((n, k, 3), dtype=wp.int32, device=device)
    fan_counts = _launch.empty(n, dtype=wp.int32, device=device)
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
            out_tris,
            fan_counts,
        ],
        device=device,
    )
    # Scanned in place: each fan's packed offset is its predecessor's inclusive total, and the last
    # entry sizes the candidate buffers.
    _launch.array_scan(fan_counts, out_array=fan_counts, inclusive=True)
    # Readback: the candidate count sizes the sort buffers.
    n_candidates = int(read_scalar(fan_counts))
    if n_candidates == 0:
        return _launch.clone(points), _launch.empty(0, dtype=wp.int32, device=device)

    faces = _repeated_oriented_triangles(out_tris, fan_counts, n_candidates, n)
    if faces.size == 0:
        return _launch.clone(points), faces
    # Every face carries a distinct vertex set, so the duplicate-resolution stage has nothing to do.
    # Orientation is not re-derived (``orient=False``): the fans are wound from the trusted normals,
    # and ``make_normals_outward`` would rewind a whole inward-normal cloud's mesh outward, silently
    # breaking that contract.
    return _clean_reconstruction(points, faces, crit_hole_length, orient=False, deduplicate=False)


def _box_diagonal(bounds: tuple[wp.vec3, wp.vec3]) -> float:
    """
    Length of a box's diagonal, bit for bit what ``bounds.enclosing_diagonal`` returns.

    Both read the same ``float32`` corners and take the norm of their ``float32`` difference.
    """
    lower = np.array(bounds[0], dtype=np.float32)
    upper = np.array(bounds[1], dtype=np.float32)
    return float(np.linalg.norm(upper - lower))


def _repeated_oriented_triangles(
    tris: odt.ArrayNdInt32, inclusive_counts: wp.array[wp.int32], n_candidates: int, n: int
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
    no two carry the same vertex set, so that stage keeps every one and needs no pass.
    """
    device = tris.device
    # Double-width, as ``warp.utils.radix_sort_pairs`` wants: the upper halves are its scratch.
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

    flags = _launch.empty(n_candidates, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_reconstruction.repeated_triangle_flags,
        dim=n_candidates,
        inputs=[keys, wp.int32(n_candidates), flags],
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
            inputs=[flags, slots, tris.reshape((-1, 3)), faces],
            device=device,
        )
    return faces


def screened_poisson(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    *,
    depth: int = 8,
    full_depth: int = 5,
    scale: float = 1.1,
    point_weight: float = 4.0,
    solver_iterations: int = 100,
    solver_tolerance: float = 1e-5,
    confidence: bool = False,
    method: Literal["dense", "adaptive"] = "dense",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Screened-Poisson surface reconstruction from an oriented point cloud (Kazhdan PoissonRecon).

    GPU-native reimplementation of the screened-Poisson filter (PyMeshLab's
    ``generate_surface_reconstruction_screened_poisson``, ``open3d``'s
    ``create_from_point_cloud_poisson``).
    A dense node-centered grid over a padded bounding cube is used instead of PoissonRecon's octree:
    the oriented normals are trilinearly splatted into a smoothed vector field ``V`` (with a density
    weight ``W``), the indicator function ``x`` is recovered by solving the screened-Poisson system
    ``(L_N + point_weight * W) x = -div V`` matrix-free with a Jacobi-preconditioned conjugate
    gradient, and the iso-surface is extracted with ``warp.MarchingCubes`` at the iso-value given by
    the average of ``x`` over the input points. A **cascadic** coarse-to-fine schedule solves the
    system from ``full_depth`` up to ``depth``, prolonging each level's solution as the next level's
    initial guess (mirroring PoissonRecon's multigrid hierarchy).

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
        cube of this; ``depth=8`` (a ``257**3`` grid) is a safe default and ``depth`` is capped at
        ``10``. This is the single biggest lever on both cost and quality — see Notes before
        lowering it.
    full_depth
        Coarsest depth of the cascade (``3 <= full_depth <= depth``). The system is solved at every
        depth from ``full_depth`` to ``depth``.
    scale
        Ratio between the reconstruction cube's side and the point cloud's largest bounding-box
        extent (``>= 1``, so the cube always contains the cloud); ``1.1`` pads the cloud by 10 %.
    point_weight
        Screening weight ``alpha`` tying the iso-surface to the input samples. ``0`` recovers the
        unscreened Poisson reconstruction (a tiny epsilon is still added to keep the operator SPD).
    solver_iterations
        Maximum conjugate-gradient iterations per cascade level. The default leaves room: the dense
        grid's multigrid-preconditioned solve reaches the default tolerance in a dozen or so at any
        depth, and the adaptive backend's Jacobi-preconditioned one in under a hundred.
    solver_tolerance
        Relative residual tolerance for the conjugate-gradient solve. A level that stops at
        ``solver_iterations`` above it warns. The system is solved in ``float32``, whose rounding
        bounds the relative residual either backend can actually reach at a few ``1e-6`` -- a
        tolerance below that is never met and only spends the whole iteration budget.
    confidence
        When ``True``, weight each sample's splat by ``|normals[i]|`` (treating the normal magnitude
        as a per-sample confidence), matching PoissonRecon's ``confidence`` flag.
    method
        Solver backend. ``"dense"`` (default) uses the dense node-centered grid described above.
        ``"adaptive"`` uses a ``warp.fem`` adaptive Nanogrid refined only near the samples with a
        variational (finite-element) assembly: fewer degrees of freedom for the same finest
        resolution, so it reaches higher ``depth`` on the same memory budget. The two backends agree
        up to discretization. A point-source weak form rings when the cells are much finer than the
        sampling, so the adaptive backend caps the finest near-surface cell at roughly the mean
        sample spacing (PoissonRecon-style); a ``depth`` above what the sampling supports then only
        refines the extraction lattice. It lazily imports ``warp.fem`` (a one-time codegen cost of
        tens of seconds on first use).

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` iso-surface vertices on ``points.device``.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer, free of zero-area triangles
        ([`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]) and oriented outward
        ([`make_normals_outward`][ordito.repair.make_normals_outward]).

    Raises
    ------
    ValueError
        If ``points`` has fewer than 3 points, or the depth/scale parameters are out of range.
    RuntimeError
        If ``points`` and ``normals`` are not all on one device.

    Warns
    -----
    UserWarning
        When a level's conjugate-gradient solve stops at ``solver_iterations`` with its residual
        still above ``solver_tolerance``: the surface is then extracted from an unconverged field.

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

    **Choosing ``depth``.** The useful depth follows the cloud's *sampling density*, not its size,
    and past that point extra depth costs time without buying fidelity -- once the grid outruns the
    samples, accuracy can get worse rather than better. The default of ``8`` matches PoissonRecon's
    own; for a known cloud density it is worth lowering. The ``adaptive`` backend is much flatter in
    depth (it already caps near-surface refinement at the sample spacing) and is the better choice
    when the depth wanted for the extraction lattice exceeds what the sampling supports.

    **``point_weight=0`` is ill-conditioned, and its output is not reproducible.** Screening is what
    conditions the operator; at ``0`` only the ``1e-4`` floor above keeps it SPD, so the conjugate
    gradient stops on a solution whose level set is genuinely uncertain, and the face count (and
    how many faces come out zero-area before the cleanup described under ``Returns``) varies
    between runs on the same input; the surface is also not watertight. The default
    ``point_weight=4`` is stable across runs and watertight. So ``0`` is for comparing *against* a
    screened reconstruction, not for producing one -- and do not pin a count taken from it.
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
    if method not in ("dense", "adaptive"):
        raise ValueError(f"screened_poisson method must be 'dense' or 'adaptive', got {method!r}.")

    device = points.device
    n = points.size
    if n < 3:
        raise ValueError(f"screened_poisson requires at least 3 points, got {n}.")

    cube_lower, cube_upper, cube_size = _poisson_cube(points, scale)
    # Effective screening weight: a floor keeps the operator SPD even at point_weight == 0.
    screen = max(float(point_weight), 1e-4)

    if method == "adaptive":
        vertices, faces = _screened_poisson_adaptive(
            points,
            normals,
            cube_lower,
            cube_upper,
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
        inv_cell = float(res - 1) / cube_size
        sampled = _launch.empty(n, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_reconstruction.sample_field_trilinear,
            dim=n,
            inputs=[solution, res, cube_lower, inv_cell, points, sampled],
            device=device,
        )
        iso = _poisson_iso_value(sampled, normals, confidence)
        vertices, faces = od.levelset.marching_cubes(
            odt.as_array3d(solution.reshape((res, res, res)), wp.float32),
            iso,
            bounds=(cube_lower, cube_upper),
        )

    # Drop zero-area triangles before orienting. Marching cubes emits one wherever the level set
    # grazes a lattice node, and such a face has no normal for ``make_normals_outward`` to orient
    # and hands the caller a NaN out of any closest-point query -- trimesh's ``closest_point``
    # divides by the squared length of the zero-length edge. This is deliberately *not* the full
    # ``_clean_reconstruction`` tail the other three reconstructions use: welding the coincident
    # vertices as well preserves the boundary-edge count but manufactures non-manifold edges, and
    # dedup would change the default path's output. As written it is **byte-identical** on a
    # well-screened reconstruction -- the default ``point_weight`` emits no degenerate face at all.
    vertices, faces = od.repair.remove_degenerate_faces(vertices, faces)
    if faces.size > 0:
        faces = od.repair.make_normals_outward(vertices, faces)
    return vertices, faces


def _poisson_cube(points: wp.array[wp.vec3], scale: float) -> tuple[wp.vec3, wp.vec3, float]:
    """Return the padded cubic reconstruction domain (lower, upper, side) around the cloud AABB."""
    lo, hi = od.bounds.aabb(points)
    center = 0.5 * (lo + hi)
    max_extent = max(b - a for a, b in zip(odt.vec3_floats(lo), odt.vec3_floats(hi), strict=True))
    if max_extent <= 0.0:
        raise ValueError("screened_poisson requires points with a non-degenerate bounding box.")
    cube_size = scale * max_extent
    half = wp.vec3(0.5 * cube_size, 0.5 * cube_size, 0.5 * cube_size)
    return center - half, center + half, cube_size


def _poisson_iso_value(
    sampled: wp.array[wp.float32], normals: wp.array[wp.vec3], confidence: bool
) -> float:
    """
    Iso-value for the extraction: the mean of the solution at the input samples.

    PoissonRecon's ``GetIsoValue``. With ``confidence`` set, the mean is weighted by
    ``|normals[i]|`` (the same per-sample confidence the splat/quadrature weights use) instead of
    uniform; the two weightings are deliberately kept distinct rather than unified. Shared by both
    backends, whose only difference is how ``sampled`` was produced.

    Both branches reduce on the device rather than reading the fields back.
    """
    if not confidence:
        return od.reduce.mean(sampled)
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
        if int(iterations) < solver_iterations:
            continue
        tolerance_sq, residual_sq = residual_tolerance.numpy()
        residual, tolerance = math.sqrt(float(residual_sq)), math.sqrt(float(tolerance_sq))
        if residual > tolerance:
            warnings.warn(
                f"screened_poisson: the {res}^3 level's conjugate gradient hit its "
                f"{solver_iterations}-iteration cap with residual norm {residual:.3e} against "
                f"tolerance {tolerance:.3e}; raise solver_iterations or solver_tolerance.",
                stacklevel=2,
            )
    return prev_solution, prev_res


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
    a Jacobi-preconditioned conjugate gradient with the screened Laplacian applied as a
    ``warp.optim.linear.LinearOperator``. ``initial`` is the warm start and is solved in place;
    ``iterations`` receives the round count, and the returned device pair is the squared threshold
    and squared residual (``_solve_screened_poisson``) -- nothing is read back here.
    """
    n_nodes = res * res * res
    # The splat's four accumulators in one zeroed allocation.
    splat = _launch.zeros(4 * n_nodes, dtype=wp.float32, device=device)
    vx, vy, vz, weights = (odt.as_dense(splat[k * n_nodes : (k + 1) * n_nodes]) for k in range(4))
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
    rhs = _launch.empty(n_nodes, dtype=wp.float32, device=device)
    smoother = _launch.empty(n_nodes, dtype=wp.float32, device=device)
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
    # The five vectors in one allocation; every entry is written by ``poisson_cg_initial``.
    vectors = _launch.empty(5 * stride, dtype=wp.float32, device=device)
    r, u, w, p, s = (odt.as_dense(vectors[k * stride : (k + 1) * stride]) for k in range(5))
    # Every entry a fold reads, ``[0, blocks)``, is written by the partials launch before it.
    partials = _launch.empty((3, 1, blocks), dtype=wp.float64, device=device)
    # The per-solve scalars in one allocation, the squared threshold and the dots' ``(r.r, gamma)``
    # leading so the pair the caller tests is contiguous, then the coefficients and the
    # recurrence's four scalars.
    scalars = _launch.empty((10, 1), dtype=wp.float64, device=device)
    atol_sq = odt.as_dense(scalars[0, 0:1])
    dots = odt.as_array2d(odt.as_dense(scalars[1:3]), wp.float64)
    coefficients = odt.as_array2d(odt.as_dense(scalars[3:6]), wp.float64)
    gamma_old, alpha_old, gamma_new, alpha_new = (
        odt.as_dense(scalars[row, 0:1]) for row in range(6, 10)
    )
    state = _launch.empty(kernel_array.LOOP_STATE_SIZE, dtype=wp.int32, device=device)
    screen_f = wp.float32(screen)
    multigrid = _PoissonMultigrid(weights, screen, res, smoother, device)
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
    return odt.as_dense(scalars.flatten()[0:2])


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
        device: wp.DeviceLike,
    ) -> None:
        self._device = device
        self._screen = wp.float32(screen)
        self._levels: list[_GridLevel] = []
        lap = 1.0
        w, smooth = weights, smoother
        finest = True
        while True:
            n = res**3
            owned = 3 if finest else 5
            vectors = [_launch.empty(n, dtype=wp.float32, device=device) for _ in range(owned)]
            b, x = (None, None) if finest else (vectors.pop(), vectors.pop())
            self._levels.append(_GridLevel(res, wp.float32(lap), w, smooth, b, x, *vectors))
            finest = False
            if res <= _MG_COARSEST:
                break
            res_c = (res - 1) // 2 + 1
            lap *= 2.0
            w_c = _launch.empty(res_c**3, dtype=wp.float32, device=device)
            smooth_c = _launch.empty(res_c**3, dtype=wp.float32, device=device)
            _launch.launch(
                kernel_reconstruction.poisson_mg_coarsen,
                dim=(res_c, res_c, res_c),
                inputs=[w, res, res_c, wp.float32(lap), self._screen, wp.float32(_MG_OMEGA)],
                outputs=[w_c, smooth_c],
                device=device,
            )
            res, w, smooth = res_c, w_c, smooth_c

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


def _screened_poisson_adaptive(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    cube_upper: wp.vec3,
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
    Solve the screened-Poisson system on a ``warp.fem`` adaptive Nanogrid and extract its surface.

    Backs the ``method="adaptive"`` path of
    [`screened_poisson`][ordito.reconstruction.screened_poisson]. ``warp.fem`` and its integrand
    module are imported lazily here (not at module scope) so ``import ordito`` never pays the
    tens-of-seconds warp.fem first-call codegen unless the adaptive backend is used.

    The solve is variational instead of the dense backend's finite-difference stencil: a dense
    coarse grid at ``2**full_depth`` is refined toward the samples with an ``fem.ImplicitField``
    oracle that never carves voxels (full-cube coverage), then the weak Laplacian
    ``integral(grad u . grad v)``,
    an exact point-measure screening term, and a point-source right-hand side are assembled over the
    cells and a ``PicQuadrature`` at the samples and solved with a diagonal-preconditioned conjugate
    gradient. Everything runs in index space (finest spacing ``1``) so the screening-vs-gradient
    balance matches the dense backend's index-space calibration. The (un-oriented) iso-surface
    ``(vertices, faces)`` is returned; the caller orients it outward.
    """
    # Deferred: importing ``warp.fem`` adds meaningfully to ``import ordito``, and the kernel
    # module below imports it at module scope, so both stay behind the one adaptive-Poisson path.
    # The deferral is only real because ``kernels/curvature.py`` and ``kernels/smoothing.py`` take
    # their two QR helpers from ``warp._src.fem.linalg``; while they used the public path this saved
    # nothing at all, since ``import ordito`` loaded the whole fem package anyway.
    import warp.fem as fem

    from ordito.kernels.algorithms import poisson_fem as kernel_poisson_fem

    device = points.device
    # ``warp.fem`` launches its internal kernels on Warp's *ambient* device, not on the device of
    # the arrays it is handed, so on a box with a CUDA device every ``fem`` call below would land
    # on ``cuda:0`` while these buffers sit on the host -- a genuine cross-device launch that a
    # strict launch-device check rejects (``PicQuadrature``'s ``finalize_cell_particle_data`` is the
    # first to fire). ordito's own allocations already carry ``device=``; this is the one place
    # where a dependency picks the device for us, so the whole fem section runs under a scope.
    with wp.ScopedDevice(device):
        n = points.size

        res_fine = 1 << depth
        scale_to_index = float(res_fine) / cube_size

        # Index-space sample positions, unit normals, and per-sample quadrature measures.
        positions = _launch.empty(n, dtype=wp.vec3, device=device)
        _launch.map(
            kernel_poisson_fem.world_to_index,
            points,
            cube_lower,
            wp.float32(scale_to_index),
            out=positions,
        )
        unit_normals = _launch.empty(n, dtype=wp.vec3, device=device)
        # These maps are independent and the same width, so they would merge into one
        # multi-output call -- declined because it saves one launch on a function whose body is a
        # finite-element Poisson solve, which is orders of magnitude more work.
        _launch.map(wp.normalize, normals, out=unit_normals)
        if confidence:
            measures = _launch.empty(n, dtype=wp.float32, device=device)
            _launch.map(wp.length, normals, out=measures)
        else:
            measures = _launch.full(n, wp.float32(1.0), device=device)

        # Match the finest near-surface cell size to the sample spacing (PoissonRecon-style): a
        # point-source weak form rings if cells are much finer than the sampling, so cap the
        # effective octree depth at ~two cells per mean nearest-neighbour distance (never below
        # full_depth, never above the requested depth). ``depth`` beyond this only refines the
        # extraction lattice, which merely samples the already-smooth field more densely.
        # The whole ``k = 2`` self-query table, as ``ball_pivoting`` reduces its own: slot 0 is the
        # self-match at exactly zero, which the positive-finite filter drops, so the mean is the
        # nearest-neighbour spacing with no copy of column 1 first.
        mean_spacing = (
            _mean_positive_finite(od.neighbors.query_nearest(points, points, k=2)[1].flatten())
            if n >= 2
            else None
        )
        spacing = mean_spacing if mean_spacing is not None else cube_size / float(res_fine)
        grid_depth = int(np.floor(np.log2(max(2.0 * cube_size / spacing, 1.0))))
        grid_depth = max(full_depth, min(depth, grid_depth))

        res_coarse = 1 << full_depth
        level_count = grid_depth - full_depth + 1
        coarse_voxel = float(1 << (depth - full_depth))
        fine_voxel = float(1 << (depth - grid_depth))
        spacing_idx = spacing * scale_to_index
        band_r = max(2.0 * fine_voxel, 1.5 * spacing_idx)
        falloff = max(coarse_voxel, 2.0 * spacing_idx)

        # Coarse dense base grid covering the whole cube in index space, then refine to the samples.
        ijk = np.stack(np.meshgrid(*(np.arange(res_coarse),) * 3, indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        # Translate by half a voxel so the voxel-centered grid spans exactly ``[0, 2**depth]`` per
        # axis. Without it the domain is ``[-coarse_voxel/2, ...]`` and the outer lattice shell
        # falls outside; failed lookups leave zeros marching cubes reads as a spurious surface.
        coarse_grid = wp.Volume.allocate_by_voxels(
            _launch.array(ijk.astype(np.int32), dtype=wp.vec3i, device=device),
            voxel_size=coarse_voxel,
            translation=(0.5 * coarse_voxel, 0.5 * coarse_voxel, 0.5 * coarse_voxel),
            device=device,
        )
        hashgrid = od.neighbors.hashgrid_from_points(positions, band_r + falloff)
        refinement = fem.ImplicitField(
            domain=fem.Cells(fem.Nanogrid(coarse_grid)),
            func=cast("wp.Function", kernel_poisson_fem.refinement_oracle),
            values={
                "grid": hashgrid.id,
                "pts": positions,
                "r": wp.float32(band_r),
                "falloff": wp.float32(falloff),
            },
        )
        geometry = fem.adaptive_nanogrid_from_field(
            coarse_grid, level_count, refinement_field=refinement, grading="face"
        )

        # Weak-form assembly: stiffness + screening = source.
        space = fem.make_polynomial_space(geometry, degree=1, dtype=float)
        domain = fem.Cells(geometry)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)
        quadrature = fem.PicQuadrature(domain, positions, measures)

        # ``kernels/`` is not type-checked, so its ``@fem.integrand`` forms reach here untyped; a
        # bilinear form integrates to a ``float32`` matrix, a linear one to a ``float32`` array.
        matrix = cast(
            "odt.BsrMatrix[wp.float32]",
            fem.integrate(
                cast("fem.Integrand", kernel_poisson_fem.diffusion_form),
                fields={"u": trial, "v": test},
            ),
        )
        screening = cast(
            "odt.BsrMatrix[wp.float32]",
            fem.integrate(
                cast("fem.Integrand", kernel_poisson_fem.screening_form),
                quadrature=quadrature,
                fields={"u": trial, "v": test},
                values={"screen": wp.float32(screen)},
            ),
        )
        matrix = odt.bsr_axpy(screening, matrix)  # ``matrix += screening``, in place
        rhs = cast(
            "wp.array[wp.float32]",
            fem.integrate(
                cast("fem.Integrand", kernel_poisson_fem.source_form),
                quadrature=quadrature,
                fields={"v": test},
                values={"normals": unit_normals},
                output_dtype=float,
            ),
        )

        solution = _launch.zeros_like(rhs)
        # ``linalg``'s own conjugate gradient on the ``float32`` system as assembled, where Warp's
        # records a new conditional graph per call. The default cadence, so that a solve which runs
        # out of iterations above its tolerance warns, on either device.
        od.linalg.solve_spd(
            matrix,
            rhs,
            solution,
            tol=solver_tolerance,
            maxiter=solver_iterations,
            name="screened_poisson",
        )
        field = space.make_field()
        field.dof_values = solution

        sampled = _launch.zeros(n, dtype=wp.float32, device=device)
        fem.interpolate(
            cast("fem.Integrand", kernel_poisson_fem.sample_field),
            at=domain,
            dim=n,
            fields={"u": field},
            values={"positions": positions, "out_values": sampled},
        )
        iso = _poisson_iso_value(sampled, normals, confidence)

        return _extract_poisson_surface_fem(
            field, domain, iso, cube_lower, cube_upper, depth, res_fine, device
        )


def _extract_poisson_surface_fem(
    field: fem.DiscreteField,
    domain: fem.GeometryDomain,
    iso: float,
    cube_lower: wp.vec3,
    cube_upper: wp.vec3,
    depth: int,
    res_fine: int,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample the adaptive ``field`` onto a dense lattice and marching-cube the ``iso`` surface.

    The lattice resolution is capped at ``2**min(depth, 9) + 1``: ``warp.MarchingCubes`` needs about
    nine times the field bytes in scratch, so a full-cube call overflows above depth 9. A
    slab-chunked pass would lift that cap, but ``warp.MarchingCubes`` is crack-free only within a
    single grid -- its per-cell face triangulation is not consistent across independent invocations,
    so welding independent slabs leaves non-manifold seams -- and depth 9-10 already oversamples the
    spacing-capped solve, so the simple capped extraction is used.
    """
    # Deferred: importing ``warp.fem`` adds to ``import ordito``'s cost, and the kernel module
    # below imports it at module scope, so both stay behind the one adaptive-Poisson path.
    import warp.fem as fem

    from ordito.kernels.algorithms import poisson_fem as kernel_poisson_fem

    res = (1 << min(depth, 9)) + 1
    step_index = float(res_fine) / float(res - 1)
    positions = _launch.empty(res * res * res, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_poisson_fem.lattice_positions,
        dim=(res, res, res),
        inputs=[wp.float32(step_index), res, wp.float32(float(res_fine) - 1e-3), positions],
        device=device,
    )
    values = _launch.zeros(res * res * res, dtype=wp.float32, device=device)
    fem.interpolate(
        cast("fem.Integrand", kernel_poisson_fem.sample_field),
        at=domain,
        dim=res * res * res,
        fields={"u": field},
        values={"positions": positions, "out_values": values},
    )
    return od.levelset.marching_cubes(
        odt.as_array3d(values.reshape((res, res, res)), wp.float32),
        iso,
        bounds=(cube_lower, cube_upper),
    )


def resample_uniform(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    voxel_size: float | None = None,
    offset: float = 0.0,
    sign_mode: Literal["parity", "winding"] = "winding",
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

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` resampled vertex positions on ``vertices.device``.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat triangle index buffer over the ``n_out_faces`` output triangles.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive.
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
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    lower, upper = od.bounds.aabb(vertices)
    # ``math.dist`` rather than ``float(wp.length(upper - lower))``: a Warp operator and a
    # Warp builtin at Python scope each route through builtin dispatch, several times dearer. It
    # computes in float64 where ``wp.length`` is float32, i.e. the correctly-rounded answer for
    # float32 corners. Section 13.1.
    diagonal = math.dist(lower, upper)
    if voxel_size is None:
        voxel_size = 0.01 * diagonal
    if voxel_size <= 0.0:
        raise ValueError(f"resample_uniform requires voxel_size > 0, got {voxel_size}")

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
    resolution = wp.vec3i(wp.int32(n_x), wp.int32(n_y), wp.int32(n_z))
    spacing = wp.vec3(
        *(
            (grid_upper_f[axis] - grid_lower_f[axis]) / float(extent - 1)
            for axis, extent in enumerate((n_x, n_y, n_z))
        )
    )

    n_points = n_x * n_y * n_z
    points = _launch.empty(n_points, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_reconstruction.lattice_points,
        dim=(n_x, n_y, n_z),
        inputs=[resolution, grid_lower, spacing, points],
        device=device,
    )
    field = od.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode=sign_mode)
    out_vertices, out_faces = od.levelset.marching_cubes(
        odt.as_array3d(field.reshape((n_x, n_y, n_z)), wp.float32),
        iso=offset,
        bounds=(grid_lower, grid_upper),
    )
    # Warp's extractor emits a vertex per crossing per cell, so coincident duplicates are normal
    # rather than exceptional; welding them is what makes the result a closed surface. Hole filling
    # is off (``crit_hole_length=0``): a level set of a signed field is already closed, so any hole
    # here would be a symptom worth surfacing rather than patching over.
    return _clean_reconstruction(out_vertices, out_faces, 0.0)


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
    guard (an edge already shared by two triangles never gains a third). The result is passed
    through the standard cleanup tail (deduplicate, drop degenerate / non-manifold faces, orient
    outward, optionally fill small holes).


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
        spacing — a device reduction, so the guess is not bit-reproducible between calls and neither
        is the triangulation built from it (see ``Notes``).
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
        ``(3 * n_faces,)`` flat triangle index buffer, oriented outward.

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
    varying between runs. Three caveats on how far that reaches:

    * the default ``radius <= 0`` is **not** covered. The auto-guess averages the nearest-neighbour
      spacings with a device reduction whose summation order is not fixed, so two calls on one cloud
      can guess radii a few ULP apart -- enough to tip a borderline pivot and change the triangle
      set. Pass ``radius`` explicitly wherever the result has to be reproducible;
    * the face buffer's *row order* is not pinned -- ``commit_triangles`` appends with a
      ``wp.atomic_add`` -- so compare reconstructions as a set of triangles, not buffer-to-buffer;
    * neither is the *winding* of a component the cleanup tail cannot orient by volume, since
      [`make_winding_consistent`][ordito.repair.make_winding_consistent] seeds each connected
      component from an arbitrary face. Compare with the row entries sorted, or orient both sides
      first.

    Results are not comparable across devices, where floating-point contraction differs.

    The result is **interpolating and edge-manifold**, and on a densely, uniformly sampled closed
    surface it is watertight in practice: a subdivided icosphere reconstructs to exactly ``2 v - 4``
    faces with no boundary edge at all. Watertightness is not *guaranteed* -- the wave-parallel
    front has no per-vertex fan structure to align colliding sheets with, so pathological sampling
    can still leave a seam. Regions sampled too sparsely for the ball to rest are left as holes by
    construction (widen ``radius`` or set ``crit_hole_length``); for an implicit surface that is
    watertight by construction use [`screened_poisson`][ordito.reconstruction.screened_poisson].
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
        normals = od.points.estimate_normals(points, neighbor_idx)

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
    _bpa_run(state, max_waves if max_waves > 0 else 16 * n)

    count = int(read_scalar(state.counters, kernel_bpa.CNT_FACE))
    faces = _launch.clone(odt.as_dense(state.all_faces[: count * 3]))
    return _clean_reconstruction(points, faces, crit_hole_length)


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

        self.counters = _launch.zeros(kernel_bpa.BPA_COUNTERS, dtype=wp.int32, device=self.device)
        _launch.fill_(self.counters[kernel_bpa.CNT_SEEDING : kernel_bpa.CNT_SEEDING + 1], 1)
        self.point_used = _launch.zeros(self.n, dtype=wp.bool, device=self.device)
        # Persists across every seeding wave for the run's whole lifetime -- see
        # ``kernel_bpa.seed_triangles`` for why a point that once exhausted its candidates without
        # seeding can never succeed later (its candidate set only shrinks), which is what makes
        # never resetting this safe.
        self.seed_failed = _launch.zeros(self.n, dtype=wp.bool, device=self.device)
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
        self.all_faces = _launch.empty(max_faces * 3, dtype=wp.int32, device=self.device)
        # A triangle soup of ``max_faces`` faces has at most ``3 * max_faces`` distinct edges, and
        # the insert probe does not terminate on a full table, so the hash is sized for a load
        # factor of at most 1/2. The front and the proposal list share the same bound.
        capacity = 1 << (6 * max_faces - 1).bit_length()
        self.edge_mask = wp.int32(capacity - 1)
        self.edge_key = _launch.zeros(capacity, dtype=wp.uint64, device=self.device)
        self.edge_count = _launch.zeros(capacity, dtype=wp.int32, device=self.device)
        self.edge_src = _launch.empty(capacity, dtype=wp.int32, device=self.device)
        self.edge_tgt = _launch.empty(capacity, dtype=wp.int32, device=self.device)
        self.edge_opp = _launch.empty(capacity, dtype=wp.int32, device=self.device)
        self.edge_state = _launch.zeros(capacity, dtype=wp.int32, device=self.device)
        self.edge_cand = _launch.empty(capacity, dtype=wp.int32, device=self.device)
        self._bind_edge_table()

        front_capacity = 3 * max_faces + self.n
        self.front_in = _launch.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.front_out = _launch.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.tri_a = _launch.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.tri_b = _launch.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.tri_c = _launch.empty(front_capacity, dtype=wp.int32, device=self.device)
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
    sum_and_count = _launch.zeros(2, dtype=wp.float64, device=device)
    _launch.launch_tiled(
        kernel_reconstruction.positive_finite_sum_and_count,
        dim=[kernel_reduce.blocks_1d(n)],
        inputs=[values, sum_and_count],
        block_dim=TILE_1D,
        device=device,
    )
    total, count = (float(x) for x in sum_and_count.numpy())
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


def _bpa_run(state: _BpaState, max_waves: int) -> None:
    """
    Drive the wave loop until the front and the orphan set are both exhausted.

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
    for _ in range(_BPA_MAX_BATCHES):
        for wave in range(_BPA_WAVES_PER_BATCH):
            _bpa_wave(state, max_waves, begin=wave == 0)
            state.front_in, state.front_out = state.front_out, state.front_in
        counters = state.counters.numpy()
        no_op_waves = _BPA_WAVES_PER_BATCH - (int(counters[kernel_bpa.CNT_WAVE]) - waves_run)
        waves_run = int(counters[kernel_bpa.CNT_WAVE])
        if no_op_waves % 2:
            state.front_in, state.front_out = state.front_out, state.front_in
        if counters[kernel_bpa.CNT_CONTINUE]:
            continue
        if counters[kernel_bpa.CNT_DONE] or counters[kernel_bpa.CNT_WAVE] >= max_waves:
            return
        if counters[kernel_bpa.CNT_GROW]:
            state.grow()
        else:
            state.compact()
        _launch.fill_(state.counters[kernel_bpa.CNT_CONTINUE : kernel_bpa.CNT_CONTINUE + 1], 1)


def _bpa_wave(state: _BpaState, max_waves: int, *, begin: bool) -> None:
    """
    Queue one wave: seed or pivot, claim, commit, advance. No allocations, no readbacks.

    ``begin`` opens a batch with the counter reset; within a batch the previous wave's
    ``end_wave`` has done it already.
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
    deduplicate: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Shared reconstruction cleanup: dedup, drop degenerate/non-manifold, orient, fill small holes.

    Duplicate faces go first, then degenerate ones (which also compacts the vertex set), then faces
    on non-manifold edges so the result is edge-manifold — required before boundary extraction,
    since both hole filling and ``boundary_loops`` assume a manifold boundary. ``crit_hole_length``
    follows the public convention: ``0`` skips hole filling, a negative value means ``0.1 x`` the
    point-cloud bounding-box diagonal. ``orient=False`` keeps the incoming winding for callers
    that already have a trusted orientation. ``deduplicate=False`` skips the duplicate-resolution
    stage for a caller whose faces already carry pairwise distinct vertex sets *in ascending
    unoriented-key order* -- the order that stage emits -- so the result is unchanged.
    """
    if faces.size == 0:
        return _launch.clone(points), faces

    if deduplicate:
        faces, _ = od.repair.resolve_duplicated_faces(faces)
    # One vertex compaction for both filters, rather than one after each: the faces here are this
    # package's own, so their indices are in range, which the combined filter requires.
    vertices, faces = od.repair.remove_degenerate_and_non_manifold_faces(points, faces)

    if faces.size > 0:
        if orient:
            faces = od.repair.make_normals_outward(vertices, faces)
        if crit_hole_length != 0.0:
            hole_length = crit_hole_length
            if hole_length < 0.0:
                hole_length = 0.1 * od.bounds.enclosing_diagonal(points)
            faces = od.holes.fill_small(vertices, faces, hole_length)
    return vertices, faces
