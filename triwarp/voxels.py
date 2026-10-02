"""
Sparse voxel grids: voxelize a mesh or a cloud, test membership, reshape, and mesh the result.

The grid is a ``warp.Volume`` -- a NanoVDB *index* grid -- rather than a triwarp value type, and
every function here either produces one or consumes one. Four properties of that grid shape the rest
of the module:

- **The builder is the dedup.** ``Volume.allocate_by_voxels`` collapses repeated cells as it builds,
  so no function here ever calls [`triwarp.grouping.unique_rows`][triwarp.grouping.unique_rows]
  first.
- **The volume's linear index is the row index of its cell array.** ``wp.volume_lookup_index``
  returns exactly the row [`cells`][triwarp.voxels.cells] puts that voxel in, so a per-voxel payload
  is a plain ``wp.array`` of length ``n_voxels`` and membership costs one ``O(1)`` probe.
- **NanoVDB centres voxels on integers**, so the volume's translation is ``origin + 0.5 * s`` where
  ``origin`` is the world position of the lower corner of cell ``(0, 0, 0)``. Under that shift cell
  ``c`` covers world ``[origin + c * s, origin + (c + 1) * s)`` exactly, which is Open3D's cell and
  the anchor [`triwarp.remesh.cluster_decimate`][triwarp.remesh.cluster_decimate] also uses.
  [`grid_transform`][triwarp.voxels.grid_transform] undoes the shift so callers never see it.
- **The cell order is leaf-major, not lexicographic**: ``get_voxels`` walks NanoVDB's 8-cubed leaves
  and the cells within each. It is deterministic and identical on CPU and CUDA, but it is *not* a
  C-order ``reshape``, so [`cells`][triwarp.voxels.cells] carries an ``order`` flag for callers who
  need the ascending-key order [`triwarp.grouping.unique_rows`][triwarp.grouping.unique_rows]
  produces.

Because the grid *is* a ``warp.Volume``, a caller can sample it (``wp.volume_sample_f``), probe it
(``wp.volume_lookup_index``) inside their own kernels, save it with ``grid.save_to_nvdb(path)``, and
hand it straight to ``warp.fem``'s nanogrid geometries. There is deliberately no ``to_volume`` pair.

**Three functions here are about a *dense corner lattice* rather than the sparse grid**, and share
nothing with it but the ``(lower, upper)`` ``bounds`` convention
[`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] takes:
[`grid_points`][triwarp.voxels.grid_points] produces the sample positions,
[`splat_onto_grid`][triwarp.voxels.splat_onto_grid] accumulates a scattered field onto them and
[`sample_grid_trilinear`][triwarp.voxels.sample_grid_trilinear] reads one back. The last two are
transposes of each other and share the world-to-lattice map, so they are kept together rather than
split across modules where the two halves could drift a half-cell apart.
[`interpolate_from_points`][triwarp.interpolation.interpolate_from_points] carries the
cross-reference from the other side.

!!! note "Recipes this module does not wrap"
    - **Implicit CSG**: [`grid_points`][triwarp.voxels.grid_points] →
      [`triwarp.proximity.signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] per
      solid → ``wp.map(wp.min, ...)`` for a union →
      [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes]. Wrapping it would put a
      four-module pipeline behind one name and duplicate ``marching_cubes``'s own arguments; the
      *voxel* answer to the same question is [`union`][triwarp.voxels.union] and its two siblings.
    - **A full dense box**: ``from_dense(wp.full(shape, True), s, o)``, which is Open3D's
      ``VoxelGrid.create_dense``. Two calls with no gotcha between them, and a wrapper would
      allocate the identical dense lattice.

See Also
--------
[`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes]
[`triwarp.remesh.cluster_decimate`][triwarp.remesh.cluster_decimate]
"""

from __future__ import annotations

import math
from typing import Literal, TypeVar, cast, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, require_same_device
from triwarp.kernels import grouping as kernel_grouping
from triwarp.kernels import interpolation as kernel_interpolation
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import voxels as kernel_voxels

DType = TypeVar("DType")

# Neighbourhood stencils, in the order ``scipy.ndimage.generate_binary_structure`` induces:
# 6 = face-adjacent (rank 1), 18 = face + edge (rank 2), 26 = the full 3-cubed shell (rank 3).
_CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}

# The six axis directions, in the order the cube-face table below expects: -x, +x, -y, +y, -z, +z.
_FACE_NEIGHBORS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=np.int32
)

# The four corners of each cube face, as local corner indices ``c = 4 * dx + 2 * dy + dz``, wound so
# that the quad's normal points along the corresponding row of ``_FACE_NEIGHBORS`` -- i.e. away from
# the voxel, which is what makes [`to_boxes`][triwarp.voxels.to_boxes] outward-facing by
# construction rather than by a sign convention applied afterwards.
_FACE_CORNERS = np.array(
    [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]],
    dtype=np.int32,
)

_STENCIL_CACHE: dict[tuple[str, int, bool], twt.Array2dInt32] = {}


def voxelize_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    voxel_size: float | None = None,
    *,
    origin: wp.vec3 | None = None,
    mode: Literal["surface", "solid"] = "surface",
    max_candidates: int = 1 << 28,
) -> wp.Volume:
    """
    Occupancy grid of a triangle mesh, by an exact triangle-box overlap test per candidate cell.

    A cell is occupied when the closed triangle actually meets the closed cell, decided by the
    13-axis separating-axis test (three box normals, the triangle plane, nine edge cross-products).
    No sampling, no subdivision heuristic, no tolerance: the accept set is the same one Open3D's
    ``CreateFromTriangleMesh`` produces, and it is what makes ``mode="solid"`` sound.

    The work is flattened into one thread per **(triangle, candidate cell)** pair rather than one
    per triangle, because per-triangle window sizes span orders of magnitude on any real mesh and a
    thread-per-triangle launch is load-imbalanced by that same factor. Each triangle's window is its
    exact inclusive AABB span, whose sizes are prefix-summed into the work-item offsets.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    voxel_size
        Cell width. Defaults to ``1 %`` of the bounding-box diagonal, a ~100-cell grid across the
        mesh. **Cost is cubic in the reciprocal.**
    origin
        World position of the lower corner of cell ``(0, 0, 0)``. Defaults to
        ``aabb(vertices).min - 0.5 * voxel_size``, Open3D's half-voxel pad.
    mode
        ``"surface"`` (default) keeps only the cells the surface passes through. ``"solid"`` fills
        the enclosed interior afterwards with [`fill_cavities`][triwarp.voxels.fill_cavities], which
        is correct for a closed input: an exact tri-box voxelization of a closed surface is
        **6-connected sealed**, so no face-adjacent path leaves the interior and the fill needs no
        containment query at all. On an open surface it fills whatever the shell happens to enclose.
    max_candidates
        Guard on the total number of (triangle, cell) pairs, which grows cubically as
        ``voxel_size`` shrinks. Exceeding it raises rather than attempting the allocation.

    Returns
    -------
    wp.Volume
        A NanoVDB index grid on ``vertices.device`` with one active voxel per occupied cell. Empty
        (zero active voxels) when the mesh has no faces.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, if ``mode`` is not one of the two names, if
        ``max_candidates`` is not positive, or if the candidate count exceeds ``max_candidates``.
        At ``mode="solid"``, also whatever [`fill_cavities`][triwarp.voxels.fill_cavities] raises:
        ``max_candidates`` bounds the (triangle, cell) pairs, which is a *surface* count, and the
        fill densifies the occupied *box*, so a large thin mesh can clear this guard and meet that
        one. The fill runs at its own default budget.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`voxelize_points`][triwarp.voxels.voxelize_points]
    [`fill_cavities`][triwarp.voxels.fill_cavities]
    [`triwarp.reconstruction.resample_uniform`][triwarp.reconstruction.resample_uniform]

    Notes
    -----
    Open3D runs the same test in ``float64`` and this one runs in ``float32``, so a triangle exactly
    tangent to a cell plane can be resolved differently by the two. Nothing else differs: Open3D's
    ``round((max - min) / voxel_size) + 2`` candidate window is a superset of the exact AABB window
    used here, and a cell outside a triangle's AABB cannot overlap the triangle.

    trimesh's ``voxelize_subdivide`` is a *different* algorithm -- subdivide until every edge is
    under half a pitch, then round each vertex to a cell -- and its result neither contains nor is
    contained in this one.
    """
    require_same_device(vertices=vertices, faces=faces)
    if mode not in ("surface", "solid"):
        raise ValueError(f"mode must be 'surface' or 'solid', got {mode!r}")
    if max_candidates <= 0:
        raise ValueError(f"max_candidates must be positive, got {max_candidates}")

    device = vertices.device
    n_faces = faces.size // 3
    voxel_size, origin = resolve_voxel_grid(vertices, voxel_size, origin, caller="voxelize_mesh")
    if n_faces == 0:
        return _empty_grid(voxel_size, origin, device)

    counts = _launch.empty(n_faces, dtype=wp.int32, device=device)
    counts_f32 = _launch.empty(n_faces, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_voxels.count_triangle_candidates,
        dim=n_faces,
        inputs=[vertices, faces, origin, wp.float32(1.0 / voxel_size), counts, counts_f32],
        device=device,
    )
    # The budget check runs on the float32 copy: the exact count can overflow ``int32`` for an
    # absurd voxel_size, and the whole point of the guard is to catch exactly that case before the
    # prefix sum is asked to represent it.
    approximate_total = tw.reduce.sum(counts_f32)
    if approximate_total > float(max_candidates):
        raise ValueError(
            f"voxelize_mesh would test {approximate_total:.3g} (triangle, cell) pairs, above "
            f"max_candidates={max_candidates}: voxel_size={voxel_size:.6g} is too small for this "
            "mesh (the count grows as its reciprocal cubed)"
        )
    offsets, total = tw.array.counts_to_offsets(counts)
    if total == 0:
        return _empty_grid(voxel_size, origin, device)

    candidate_cells = twt.empty_2d((total, 3), wp.int32, device=device)
    accepted = _launch.empty(total, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_voxels.test_triangle_candidates,
        dim=total,
        inputs=[
            vertices,
            faces,
            offsets,
            origin,
            wp.float32(voxel_size),
            wp.float32(1.0 / voxel_size),
            candidate_cells,
            accepted,
        ],
        device=device,
    )
    # ``point_mask`` lets the builder skip the rejected candidates in place, so no compaction pass
    # and no second cell buffer.
    grid = wp.Volume.allocate_by_voxels(
        cast("wp.array[wp.int32]", candidate_cells),
        voxel_size=voxel_size,
        translation=_translation(origin, voxel_size),
        point_mask=accepted,
        device=device,
    )
    if mode == "solid":
        return fill_cavities(grid)
    return grid


def voxelize_points(
    points: wp.array[wp.vec3], voxel_size: float | None = None, *, origin: wp.vec3 | None = None
) -> wp.Volume:
    """
    Occupancy grid of a point cloud: one active voxel per cell that contains at least one point.

    Open3D's ``VoxelGrid.create_from_point_cloud``. There is no dedup step to write — the builder
    collapses repeated cells itself.

    Parameters
    ----------
    points
        ``(n_points,)`` positions.
    voxel_size
        Cell width. Defaults to ``1 %`` of the cloud's bounding-box diagonal.
    origin
        World position of the lower corner of cell ``(0, 0, 0)``. Defaults to
        ``aabb(points).min - 0.5 * voxel_size``.

    Returns
    -------
    wp.Volume
        A NanoVDB index grid on ``points.device``. Empty when ``points`` is empty.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive.

    See Also
    --------
    [`voxelize_mesh`][triwarp.voxels.voxelize_mesh]
    [`voxel_down_sample`][triwarp.voxels.voxel_down_sample]
    [`cell_indices`][triwarp.voxels.cell_indices]
    """
    device = points.device
    voxel_size, origin = resolve_voxel_grid(points, voxel_size, origin, caller="voxelize_points")
    if points.size == 0:
        return _empty_grid(voxel_size, origin, device)
    return from_cells(cell_indices(points, voxel_size, origin=origin), voxel_size, origin)


@overload
def voxel_down_sample(
    points: wp.array[wp.vec3],
    voxel_size: float | None = None,
    *,
    origin: wp.vec3 | None = None,
    pooling: Literal["mean", "min", "max", "sum"] = "mean",
    return_inverse: Literal[False] = False,
) -> wp.array[wp.vec3]: ...
@overload
def voxel_down_sample(
    points: wp.array[wp.vec3],
    voxel_size: float | None = None,
    *,
    origin: wp.vec3 | None = None,
    pooling: Literal["mean", "min", "max", "sum"] = "mean",
    return_inverse: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
def voxel_down_sample(
    points: wp.array[wp.vec3],
    voxel_size: float | None = None,
    *,
    origin: wp.vec3 | None = None,
    pooling: Literal["mean", "min", "max", "sum"] = "mean",
    return_inverse: bool = False,
) -> wp.array[wp.vec3] | tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Reduce a cloud to one point per occupied voxel.

    Open3D's ``PointCloud.voxel_down_sample``: the standard uniform decimation for a scan, and the
    cheapest way to put two clouds on a common sampling density before registration. The default
    ``"mean"`` is Open3D's ``AccumulatedPoint`` average and does **not** renormalize, so the same
    call pools per-point normals or colours through
    [`pool_by_voxel`][triwarp.voxels.pool_by_voxel].

    Parameters
    ----------
    points
        ``(n_points,)`` positions.
    voxel_size
        Cell width. Defaults to ``1 %`` of the cloud's bounding-box diagonal.
    origin
        World position of the lower corner of cell ``(0, 0, 0)``. Defaults to
        ``aabb(points).min - 0.5 * voxel_size``.
    pooling
        How each voxel reduces the points inside it: ``"mean"`` (default), ``"sum"``, ``"min"`` or
        ``"max"`` (both component-wise).
    return_inverse
        Also return, for each input point, the row of the output it was pooled into.

    Returns
    -------
    points : wp.array[wp.vec3]
        ``(n_voxels,)`` pooled positions, in the grid's own cell order (see
        [`cells`][triwarp.voxels.cells]).
    inverse : wp.array[wp.int32], optional
        ``(n_points,)`` inverse map, present when ``return_inverse=True``. ``inverse[i]`` is the
        output row of input point ``i``.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, or ``pooling`` is not one of the four names.

    See Also
    --------
    [`pool_by_voxel`][triwarp.voxels.pool_by_voxel]
    [`voxelize_points`][triwarp.voxels.voxelize_points]
    [`triwarp.remesh.cluster_decimate`][triwarp.remesh.cluster_decimate]

    Notes
    -----
    ``"mean"`` and ``"sum"`` are bitwise reproducible run to run: they sort the points by voxel and
    reduce each segment in index order rather than accumulating with float atomics, whose ordering
    varies between launches. That costs one radix sort over the point count. ``"min"`` and ``"max"``
    use atomics directly, since those are order-independent for floats.
    """
    device = points.device
    n_points = points.size
    # A derived origin sits below every point, so the cells are non-negative and the bound the
    # box already gives says whether they fit one root tile; a caller's origin says nothing.
    if origin is None:
        voxel_size, origin, cell_bound = resolve_voxel_grid(
            points, voxel_size, caller="voxel_down_sample", return_cell_bound=True
        )
    else:
        voxel_size, origin = resolve_voxel_grid(
            points, voxel_size, origin, caller="voxel_down_sample"
        )
        cell_bound = 0
    if pooling not in ("mean", "min", "max", "sum"):
        raise ValueError(f"pooling must be 'mean', 'sum', 'min' or 'max', got {pooling!r}")
    if n_points == 0:
        pooled = _launch.empty(0, dtype=wp.vec3, device=device)
        return (
            (pooled, _launch.empty(0, dtype=wp.int32, device=device)) if return_inverse else pooled
        )
    table = _VoxelTable(points, voxel_size, origin, one_root_tile=0 < cell_bound <= 4096)
    n_voxels = table.n_voxels
    slots = _launch.empty(n_points, dtype=wp.int32, device=device) if return_inverse else None
    if pooling in ("min", "max"):
        # One sentinel bucket past the last voxel collects the points that probe outside the set.
        counts = _launch.zeros(n_voxels + 1, dtype=wp.int32, device=device)
        largest = pooling == "max"
        limit = -math.inf if largest else math.inf
        pooled = _launch.full(n_voxels, wp.vec3(limit, limit, limit), dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_voxels.pool_extremum_table,
            dim=n_points,
            inputs=[points, *table.probe_inputs(), largest],
            outputs=[slots, counts, pooled],
            device=device,
        )
        _launch.launch(
            kernel_voxels.zero_empty_voxels,
            dim=n_voxels,
            inputs=[counts],
            outputs=[pooled],
            device=device,
        )
    else:
        buckets, order = table.buckets, table.order
        _launch.launch(
            kernel_voxels.bucket_table_slots,
            dim=n_points,
            inputs=[points, *table.probe_inputs(), wp.int32(n_voxels)],
            outputs=[slots, buckets, order],
            device=device,
        )
        pooled = _segment_pool(points, buckets, order, n_voxels, average=pooling == "mean")
    if slots is None:
        return pooled
    return pooled, slots


class _VoxelTable:
    """
    [`voxel_down_sample`][triwarp.voxels.voxel_down_sample]'s voxel set, without a ``wp.Volume``.

    The rows [`voxelize_points`][triwarp.voxels.voxelize_points]' grid would have, in its order, and
    the table its pooling probe resolves a point's row through. One open-addressing table of point
    indices keyed on the cell each point inserts into (``voxel_cell_indices``' arithmetic); the
    distinct cells are sorted into ``Volume.get_voxels()``'s order -- NanoVDB's tree order, one
    36-bit in-tile key, plus a stable second pass over the root-tile key when the cells may span
    more than one tile -- and each claimed slot is rewritten to its row. The pooling kernels then
    probe it with the volume's own index transform, reproduced bit for bit, so every row, bucket
    and pooled value is the one the volume path produces.

    The mean/sum pooling's sort buffers are allocated here, because the table borrows them as
    scratch before the pooling launch writes them: the appended slots live in the upper half of
    ``buckets`` and the grid-order payload in ``order``.
    """

    def __init__(
        self, points: wp.array[wp.vec3], voxel_size: float, origin: wp.vec3, *, one_root_tile: bool
    ) -> None:
        device = points.device
        n_points = points.size
        inverse_size = wp.float32(1.0 / voxel_size)
        # A cloud with no repeated cell fills the table at most half full.
        self.mask = kernel_grouping.hash_table_mask(n_points)
        # One slot past the table is the append cursor, so the ``-1`` fill seeds it too and it
        # counts from ``-1``; the hash never reaches it (``mask`` addresses the table alone).
        self.table = _launch.full(self.mask + 2, -1, dtype=wp.int32, device=device)
        self.buckets = _launch.empty(2 * n_points, dtype=wp.int32, device=device)
        self.order = _launch.empty(2 * n_points, dtype=wp.int32, device=device)
        unique = self.buckets[n_points:]
        _launch.launch(
            kernel_voxels.insert_point_cells,
            dim=n_points,
            inputs=[points, origin, inverse_size, self.mask, self.table],
            outputs=[unique],
            device=device,
        )
        # One readback, unavoidable: the voxel count sizes the output and every per-voxel buffer.
        n_voxels = int(read_scalar(self.table)) + 1
        self.n_voxels = n_voxels
        keys = _launch.empty(2 * n_voxels, dtype=wp.uint64, device=device)
        rank = self.order
        _launch.launch(
            kernel_voxels.unique_cell_keys,
            dim=n_voxels,
            inputs=[points, origin, inverse_size, self.table, unique],
            outputs=[keys, rank],
            device=device,
        )
        _launch.radix_sort_pairs(keys, rank, count=n_voxels, end_bit=36)
        if not one_root_tile:
            # Stable, so the in-tile order survives within each root tile.
            _launch.launch(
                kernel_voxels.unique_cell_root_keys,
                dim=n_voxels,
                inputs=[points, origin, inverse_size, self.table, unique, rank],
                outputs=[keys],
                device=device,
            )
            _launch.radix_sort_pairs(keys, rank, count=n_voxels, end_bit=60)
        self.row_cells = _launch.empty(n_voxels, dtype=wp.vec3i, device=device)
        _launch.launch(
            kernel_voxels.assign_voxel_rows,
            dim=n_voxels,
            inputs=[points, origin, inverse_size, unique, rank, self.table],
            outputs=[self.row_cells],
            device=device,
        )
        # The volume's own transform, as ``allocate_by_voxels`` stores it: a float32 translation
        # half a cell above ``origin`` and the float32 inverse of the float32 cell width.
        self.translation = wp.vec3(*_translation(origin, voxel_size))
        self.probe_inverse = float(np.float32(1.0 / float(np.float32(voxel_size))))

    def probe_inputs(self) -> list[object]:
        """Return the pooling kernels' probe arguments, in ``table_point_slot``'s order."""
        return [self.translation, self.probe_inverse, 0.0, self.mask, self.table, self.row_cells]


def pool_by_voxel(
    grid: wp.Volume,
    points: wp.array[wp.vec3],
    values: wp.array[wp.vec3],
    *,
    pooling: Literal["mean", "min", "max", "sum"] = "mean",
) -> wp.array[wp.vec3]:
    """
    Reduce a per-point ``wp.vec3`` payload onto the voxels of ``grid``.

    The general form of [`voxel_down_sample`][triwarp.voxels.voxel_down_sample]: positions,
    normals, colours or velocities, pooled onto whichever grid the caller already has. It takes the
    grid rather than a precomputed inverse map because the grid *is* the inverse map — one
    ``O(1)`` probe per point recovers its output row.

    Parameters
    ----------
    grid
        Index grid whose voxels are the output rows.
    points
        ``(n_points,)`` positions locating each payload value. Points falling outside ``grid`` are
        ignored.
    values
        ``(n_points,)`` payload, one per entry of ``points``.
    pooling
        ``"mean"`` (default), ``"sum"``, ``"min"`` or ``"max"`` (both component-wise).

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_voxels,)`` pooled values, indexed by the grid's own voxel numbering. Voxels no point
        landed in are zero, for every ``pooling``.

    Raises
    ------
    ValueError
        If ``pooling`` is not one of the four names, or ``points`` and ``values`` differ in length.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.
    RuntimeError
        If ``grid``, ``points`` and ``values`` are not all on one device.

    See Also
    --------
    [`voxel_down_sample`][triwarp.voxels.voxel_down_sample]
    [`cell_centers`][triwarp.voxels.cell_centers]
    """
    require_same_device(grid=grid, points=points, values=values)
    return _pool_by_voxel(grid, points, values, pooling, return_slots=False)[0]


def _pool_by_voxel(
    grid: wp.Volume,
    points: wp.array[wp.vec3],
    values: wp.array[wp.vec3],
    pooling: Literal["mean", "min", "max", "sum"],
    *,
    return_slots: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32] | None]:
    """
    [`pool_by_voxel`][triwarp.voxels.pool_by_voxel], also returning each point's voxel row.

    The rows are the probe the pooling runs on anyway, and
    [`voxel_down_sample`][triwarp.voxels.voxel_down_sample]'s inverse is exactly them. ``None``
    when nothing was probed (an empty grid or cloud) or when ``return_slots`` is ``False``.
    """
    if pooling not in ("mean", "min", "max", "sum"):
        raise ValueError(f"pooling must be 'mean', 'sum', 'min' or 'max', got {pooling!r}")
    if points.size != values.size:
        raise ValueError(
            f"points and values must be the same length, got {points.size} and {values.size}"
        )
    _require_index_grid(grid)
    device = points.device
    n_points = points.size
    n_voxels = _voxel_count(grid)
    if n_voxels == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), None
    if n_points == 0:
        return _launch.zeros(n_voxels, dtype=wp.vec3, device=device), None

    # Only a caller asking for the inverse gets the slots written; neither branch reads them back.
    slots = _launch.empty(n_points, dtype=wp.int32, device=device) if return_slots else None
    if pooling in ("min", "max"):
        # One sentinel bucket past the last voxel collects the points that fall outside the grid.
        counts = _launch.zeros(n_voxels + 1, dtype=wp.int32, device=device)
        largest = pooling == "max"
        limit = -math.inf if largest else math.inf
        # The atomics reduce from +-inf in the probe launch itself; the voxels no point reached are
        # reset to zero afterwards, once ``counts`` is final.
        pooled = _launch.full(n_voxels, wp.vec3(limit, limit, limit), dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_voxels.pool_extremum_points,
            dim=n_points,
            inputs=[grid.id, points, values, wp.int32(n_voxels), largest],
            outputs=[slots, counts, pooled],
            device=device,
        )
        _launch.launch(
            kernel_voxels.zero_empty_voxels,
            dim=n_voxels,
            inputs=[counts],
            outputs=[pooled],
            device=device,
        )
        return pooled, slots

    # The launch seeds the sort's two double buffers, keys and identity payload.
    buckets = _launch.empty(2 * n_points, dtype=wp.int32, device=device)
    order = _launch.empty(2 * n_points, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_voxels.bucket_point_slots,
        dim=n_points,
        inputs=[grid.id, points, wp.int32(n_voxels)],
        outputs=[slots, buckets, order],
        device=device,
    )
    return _segment_pool(values, buckets, order, n_voxels, average=pooling == "mean"), slots


def _segment_pool(
    values: wp.array[wp.vec3],
    buckets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    n_voxels: int,
    *,
    average: bool,
) -> wp.array[wp.vec3]:
    """
    Sort the bucketed points by voxel and reduce each voxel's segment in index order.

    The mean/sum tail shared by [`pool_by_voxel`][triwarp.voxels.pool_by_voxel] and
    [`voxel_down_sample`][triwarp.voxels.voxel_down_sample]: ``buckets`` / ``order`` are the two
    ``2 * n_points`` sort buffers their bucketing launch seeded, a point outside the grid in the
    sentinel bucket ``n_voxels``. The segments are read off the sorted buckets, so no histogram.
    """
    n_points = values.size
    # Stable, so each voxel's segment lists its points in index order.
    # Every bucket is at most ``n_voxels`` (the sentinel), so only those low bits are sorted.
    _launch.radix_sort_pairs(
        buckets, order, count=n_points, end_bit=max(1, int(n_voxels).bit_length())
    )
    pooled = _launch.empty(n_voxels, dtype=wp.vec3, device=values.device)
    _launch.launch(
        kernel_voxels.segment_reduce_vec3,
        dim=n_voxels,
        inputs=[order, values, buckets[:n_points], average],
        outputs=[pooled],
        device=values.device,
    )
    return pooled


def cells(grid: wp.Volume, *, order: Literal["grid", "sorted"] = "grid") -> twt.Array2dInt32:
    """
    Integer cell coordinates of every active voxel, one row per voxel.

    The array-facing half of the grid, and the only place its row order is observable: everything
    else here returns an opaque ``warp.Volume``.

    Parameters
    ----------
    grid
        Index grid to read.
    order
        Row order of the result:

        - ``"grid"`` (default) — the volume's own leaf-major order, free because it is what the
          volume already holds. Deterministic, and identical on CPU and CUDA.
        - ``"sorted"`` — ascending in the packed cell key, which costs one radix sort over the voxel
          count. Column 0 is the key's *least* significant digit, so this is lexicographic with
          ``x`` as the last tiebreak — chosen that way because it is then bit-identical to
          [`triwarp.grouping.unique_rows`][triwarp.grouping.unique_rows]'s order, which is what
          callers comparing against the pre-volume code path need.

    Returns
    -------
    Array2dInt32
        ``(n_voxels, 3)`` cell coordinates on the grid's device. Coordinates may be negative.

    Raises
    ------
    ValueError
        If ``order`` is not one of the two names.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`from_cells`][triwarp.voxels.from_cells]
    [`cell_centers`][triwarp.voxels.cell_centers]

    Notes
    -----
    The length comes from ``Volume.get_active_stats``, never from ``get_voxel_count``: on a
    *rebuildable* grid the latter reports the reserved **capacity**, and the trailing rows of
    ``get_voxels`` are padding at ``[0, 0, 0]`` that would read as real voxels.
    """
    if order not in ("grid", "sorted"):
        raise ValueError(f"order must be 'grid' or 'sorted', got {order!r}")
    _require_index_grid(grid)
    n_voxels = _voxel_count(grid)
    if n_voxels == 0:
        return twt.empty_2d((0, 3), wp.int32, device=grid.device)
    rows = twt.as_array2d(twt.as_dense(grid.get_voxels()[:n_voxels]), wp.int32)
    if order == "grid":
        return rows

    # The corner pair stays on the device: ``pack_cell_keys`` derives the per-axis shift and the
    # radix from these two three-element buffers itself, so this whole stage is readback-free.
    lower, upper = tw.reduce.minmax(rows, axis=0)
    keys = _launch.empty(n_voxels, dtype=wp.uint64, device=grid.device)
    _launch.launch(
        kernel_voxels.pack_cell_keys,
        dim=n_voxels,
        inputs=[rows, lower, upper, keys],
        device=grid.device,
    )
    _sorted_keys, permutation = tw.array.sort_and_argsort(keys)
    return twt.as_array2d(tw.array.gather(rows, permutation), wp.int32)


def from_cells(cells: twt.Array2dInt32, voxel_size: float, origin: wp.vec3) -> wp.Volume:
    """
    Build an index grid from integer cell coordinates.

    The inverse of [`cells`][triwarp.voxels.cells], and the constructor every other producer here
    ends in. Repeated rows are collapsed by the builder, so the input need not be unique.

    Parameters
    ----------
    cells
        ``(n_cells, 3)`` cell coordinates. Negative coordinates are fine.
    voxel_size
        Cell width.
    origin
        World position of the lower corner of cell ``(0, 0, 0)``.

    Returns
    -------
    wp.Volume
        A NanoVDB index grid on ``cells.device``, empty when ``cells`` is empty.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, or ``cells`` does not have three columns.
    TypeError
        If ``cells`` is not a rank-2 ``wp.int32`` array.

    See Also
    --------
    [`cells`][triwarp.voxels.cells]
    [`from_dense`][triwarp.voxels.from_dense]
    """
    twt.ensure_ndim(cells, 2, dtype=wp.int32)
    if int(cells.shape[1]) != 3:
        raise ValueError(f"cells must have three columns, got {int(cells.shape[1])}")
    if voxel_size <= 0.0:
        raise ValueError(f"from_cells requires voxel_size > 0, got {voxel_size}")
    if int(cells.shape[0]) == 0:
        return _empty_grid(voxel_size, origin, cells.device)
    # The builder reads the buffer as if contiguous, so a strided view has to be densified first.
    rows = cells if cells.is_contiguous else _launch.clone(cells)
    return wp.Volume.allocate_by_voxels(
        cast("wp.array[wp.int32]", rows),
        voxel_size=voxel_size,
        translation=_translation(origin, voxel_size),
        device=cells.device,
    )


def grid_transform(grid: wp.Volume) -> tuple[float, wp.vec3]:
    """
    Cell width and world origin of an index grid.

    Undoes the half-voxel shift NanoVDB's integer-centred voxels need, so ``origin`` here is the
    world position of the lower corner of cell ``(0, 0, 0)`` — the same anchor
    [`from_cells`][triwarp.voxels.from_cells] takes, not the volume's own translation.

    Parameters
    ----------
    grid
        Index grid to read.

    Returns
    -------
    voxel_size : float
        Cell width.
    origin : wp.vec3
        World position of the lower corner of cell ``(0, 0, 0)``.

    Raises
    ------
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`from_cells`][triwarp.voxels.from_cells]
    [`cell_centers`][triwarp.voxels.cell_centers]
    """
    _require_index_grid(grid)
    voxel_size = float(grid.get_voxel_size()[0])
    translation = grid.get_grid_info().translation
    origin = wp.vec3(*(float(translation[axis]) - 0.5 * voxel_size for axis in range(3)))
    return voxel_size, origin


@overload
def resolve_voxel_grid(
    points: wp.array[wp.vec3],
    voxel_size: float | None = None,
    origin: wp.vec3 | None = None,
    *,
    caller: str = "resolve_voxel_grid",
    return_cell_bound: Literal[False] = False,
) -> tuple[float, wp.vec3]: ...
@overload
def resolve_voxel_grid(
    points: wp.array[wp.vec3],
    voxel_size: float | None = None,
    origin: wp.vec3 | None = None,
    *,
    caller: str = "resolve_voxel_grid",
    return_cell_bound: Literal[True],
) -> tuple[float, wp.vec3, int]: ...
def resolve_voxel_grid(
    points: wp.array[wp.vec3],
    voxel_size: float | None = None,
    origin: wp.vec3 | None = None,
    *,
    caller: str = "resolve_voxel_grid",
    return_cell_bound: bool = False,
) -> tuple[float, wp.vec3] | tuple[float, wp.vec3, int]:
    """
    Voxel size and grid origin for a point set, filling in either default from its bounding box.

    The package's one definition of what an unspecified voxel grid means: a cell of ``1 %`` of the
    bounding-box diagonal, anchored half a cell below the box. That anchor is Open3D's, and keeping
    it in one place is what lets ``voxelize_points``,
    [`voxel_down_sample`][triwarp.voxels.voxel_down_sample] and
    [`cluster_decimate`][triwarp.remesh.cluster_decimate] agree cell for cell.

    Public because [`cluster_decimate`][triwarp.remesh.cluster_decimate] lives in another module and
    needs the same convention; it is pure host arithmetic over one
    [`aabb`][triwarp.bounds.aabb].

    Parameters
    ----------
    points
        ``(n,)`` positions the grid must cover. A set with no extent -- empty, a single point, or
        any cloud whose points coincide -- yields a unit diagonal, since there is no scale to read
        off it and one point belongs in one cell at any width.
    voxel_size
        Edge length of a cell. ``None`` takes ``1 %`` of the bounding-box diagonal.
    origin
        Lower corner of cell ``(0, 0, 0)``. ``None`` places it half a cell below the box.
    caller
        Name used in the error message, so a caller's own name appears rather than this one.
    return_cell_bound
        Also return an exclusive upper bound on every coordinate
        [`cell_indices`][triwarp.voxels.cell_indices] gives a point of ``points`` on this grid, read
        off the bounding box this already takes. It is the ``max_index`` a cell-row hash
        ([`hash_indices_rows`][triwarp.grouping.hash_indices_rows]) would otherwise infer with a
        reduction and a host readback. With a *derived* origin the coordinates are also
        non-negative, so the two together make that hash's ``validate=False`` sound; a caller's
        own ``origin`` above the box gives negative coordinates, which the bound says nothing
        about. When both ``voxel_size`` and ``origin`` are given the box is taken for this alone.

    Returns
    -------
    tuple[float, wp.vec3] or tuple[float, wp.vec3, int]
        ``(voxel_size, origin)``, both resolved, followed by the cell bound when
        ``return_cell_bound`` is ``True``. The bound may exceed the largest coordinate by up to
        two, which covers the float32 rounding of the cell kernel; any larger radix packs cell rows
        in the same order.

    Raises
    ------
    ValueError
        If ``voxel_size`` is passed and is not positive. A derived one never trips this: the
        zero-extent fallback above is what keeps the guard about the caller's argument.

    See Also
    --------
    [`voxelize_points`][triwarp.voxels.voxelize_points]
    [`cell_indices`][triwarp.voxels.cell_indices]
    [`cluster_decimate`][triwarp.remesh.cluster_decimate]
    [`sample.resolve_seed`][triwarp.sample.resolve_seed]

    Notes
    -----
    !!! note "The ``resolve_*`` pattern"
        Both of these -- this and [`sample.resolve_seed`][triwarp.sample.resolve_seed] /
        [`voxels.resolve_voxel_grid`][triwarp.voxels.resolve_voxel_grid] -- turn an optional
        argument into the concrete value the wrapper would have derived, so a caller who wants two
        functions to share the derived thing can resolve it once and pass it to both. Each default
        is domain knowledge, so they cannot share a module. The optional face-adjacency pair
        deliberately has **no** such resolver: deriving it is one
        [`face_adjacency`][triwarp.adjacency.face_adjacency] call with ``return_edges=True``, so
        only the *pairing rule* is worth sharing and
        [`adjacency.require_paired_adjacency`][triwarp.adjacency.require_paired_adjacency] is that
        rule on its own.
    """
    upper = wp.vec3(0.0, 0.0, 0.0)
    if voxel_size is None or origin is None or return_cell_bound:
        if points.size == 0:
            lower = wp.vec3(0.0, 0.0, 0.0)
            diagonal = 1.0
        else:
            lower, upper = tw.bounds.aabb(points)
            # ``math.dist`` rather than ``float(wp.length(upper - lower))``: a Warp operator and a
            # Warp builtin at Python scope each route through builtin dispatch, several times
            # dearer. It computes in float64 where ``wp.length`` is float32, i.e. the
            # correctly-rounded answer for float32 corners. Section 13.1.
            diagonal = math.dist(lower, upper)
            # A single point -- or any cloud whose points are all coincident -- has a zero-extent
            # box and so carries no scale to derive a cell width from. Take the same unit diagonal
            # the empty cloud takes: one point belongs in one cell whatever the width, so there is
            # a right answer here and it does not depend on the number chosen. Without this the
            # derived width is ``0.01 * 0.0`` and the guard below rejects it, naming a
            # ``voxel_size`` the caller never passed.
            if diagonal == 0.0:
                diagonal = 1.0
        if voxel_size is None:
            voxel_size = 0.01 * diagonal
        if origin is None:
            # Half a cell of slack below the box: Open3D's anchor, and the one
            # ``remesh.cluster_decimate`` uses, so all three agree cell for cell.
            origin = wp.vec3(*(float(lower[axis]) - 0.5 * voxel_size for axis in range(3)))
    if voxel_size <= 0.0:
        raise ValueError(f"{caller} requires voxel_size > 0, got {voxel_size}")
    if not return_cell_bound:
        return voxel_size, origin
    # The farthest cell along an axis is the one the box's upper corner falls in. The kernel
    # floors a float32 product, which can land one cell past the float64 quotient here, so the
    # bound carries one cell of slack on top of the exclusive ``+ 1``; an empty set has no cells
    # and takes the smallest radix a hash accepts.
    inverse = float(np.float32(1.0 / voxel_size))
    extent = max(
        math.floor((float(upper[axis]) - float(origin[axis])) * inverse) for axis in range(3)
    )
    return voxel_size, origin, max(extent, 0) + 2


def cell_indices(
    points: wp.array[wp.vec3], voxel_size: float, *, origin: wp.vec3
) -> twt.Array2dInt32:
    """
    Cell each point falls in, one row per point and no deduplication.

    trimesh's ``points_to_indices`` and Open3D's ``VoxelGrid.get_voxel``. It takes the transform
    explicitly rather than a grid because its job is to be used *before* a grid exists — it is what
    [`voxelize_points`][triwarp.voxels.voxelize_points] feeds to the builder.

    Parameters
    ----------
    points
        ``(n_points,)`` positions.
    voxel_size
        Cell width.
    origin
        World position of the lower corner of cell ``(0, 0, 0)``.

    Returns
    -------
    Array2dInt32
        ``(n_points, 3)`` cell coordinates on ``points.device``, negative below ``origin``.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive.

    See Also
    --------
    [`occupancy_at_points`][triwarp.voxels.occupancy_at_points]
    [`voxelize_points`][triwarp.voxels.voxelize_points]
    """
    if voxel_size <= 0.0:
        raise ValueError(f"cell_indices requires voxel_size > 0, got {voxel_size}")
    n_points = points.size
    out_cells = twt.empty_2d((n_points, 3), wp.int32, device=points.device)
    if n_points == 0:
        return out_cells
    _launch.launch(
        kernel_voxels.voxel_cell_indices,
        dim=n_points,
        inputs=[points, origin, wp.float32(1.0 / voxel_size), out_cells],
        device=points.device,
    )
    return out_cells


def cell_centers(grid: wp.Volume) -> wp.array[wp.vec3]:
    """
    World position of the centre of every active voxel.

    Parameters
    ----------
    grid
        Index grid to read.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_voxels,)`` centres, in the grid's own voxel order.

    Raises
    ------
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`cells`][triwarp.voxels.cells]
    [`to_boxes`][triwarp.voxels.to_boxes]
    """
    _require_index_grid(grid)
    voxels = cells(grid)
    n_voxels = int(voxels.shape[0])
    centers = _launch.empty(n_voxels, dtype=wp.vec3, device=grid.device)
    if n_voxels == 0:
        return centers
    _launch.launch(
        kernel_voxels.cell_center_positions,
        dim=n_voxels,
        inputs=[grid.id, voxels, centers],
        device=grid.device,
    )
    return centers


def occupancy_at_points(grid: wp.Volume, points: wp.array[wp.vec3]) -> wp.array[wp.bool]:
    """
    Occupancy mask of the voxel each query point falls in.

    Open3D's ``VoxelGrid.check_if_included`` and trimesh's ``VoxelGrid.is_filled``, which are one
    hash lookup per query on one core; here it is one ``O(1)`` probe per query in a single kernel,
    with no host work.

    Parameters
    ----------
    grid
        Index grid to test against.
    points
        ``(n_points,)`` query positions.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_points,)`` mask on ``points.device``.

    Raises
    ------
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.
    RuntimeError
        If ``grid`` and ``points`` are not all on one device.

    See Also
    --------
    [`occupancy_at_cells`][triwarp.voxels.occupancy_at_cells]
    [`cell_indices`][triwarp.voxels.cell_indices]
    """
    require_same_device(grid=grid, points=points)
    _require_index_grid(grid)
    n_points = points.size
    mask = _launch.empty(n_points, dtype=wp.bool, device=points.device)
    if n_points == 0:
        return mask
    _launch.launch(
        kernel_voxels.point_occupancy,
        dim=n_points,
        inputs=[grid.id, points, True, mask],
        device=points.device,
    )
    return mask


def occupancy_at_cells(grid: wp.Volume, cells: twt.Array2dInt32) -> wp.array[wp.bool]:
    """
    Occupancy mask of a list of integer cells.

    The index-space form of [`occupancy_at_points`][triwarp.voxels.occupancy_at_points], and the
    primitive the set algebra is built on: [`intersection`][triwarp.voxels.intersection] keeps the
    voxels this mask accepts and [`difference`][triwarp.voxels.difference] keeps the rest.

    Parameters
    ----------
    grid
        Index grid to test against.
    cells
        ``(n_cells, 3)`` cell coordinates.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_cells,)`` mask on ``cells.device``.

    Raises
    ------
    ValueError
        If ``cells`` does not have three columns.
    TypeError
        If ``cells`` is not a rank-2 ``wp.int32`` array, or ``grid`` is not a NanoVDB index grid
        with isotropic voxels.
    RuntimeError
        If ``grid`` and ``cells`` are not all on one device.

    See Also
    --------
    [`occupancy_at_points`][triwarp.voxels.occupancy_at_points]
    [`cells`][triwarp.voxels.cells]
    """
    require_same_device(grid=grid, cells=cells)
    _require_index_grid(grid)
    twt.ensure_ndim(cells, 2, dtype=wp.int32)
    if int(cells.shape[1]) != 3:
        raise ValueError(f"cells must have three columns, got {int(cells.shape[1])}")
    n_cells = int(cells.shape[0])
    mask = _launch.empty(n_cells, dtype=wp.bool, device=cells.device)
    if n_cells == 0:
        return mask
    _launch.launch(
        kernel_voxels.cell_occupancy,
        dim=n_cells,
        inputs=[grid.id, cells, True, mask],
        device=cells.device,
    )
    return mask


def splat_onto_grid(
    points: wp.array[wp.vec3],
    values: wp.array[DType],
    shape: tuple[int, int, int],
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
    min_weight: float = 1e-4,
) -> tuple[wp.array[DType, Literal[3]], twt.Array3dFloat32]:
    """
    Accumulate a scattered field onto a dense lattice with trilinear weights, and average it.

    pytorch3d's ``add_points_features_to_volume_densities_features``, and the scatter half of the
    lattice pair: each point contributes to the **eight** lattice corners around it, weighted by
    the trilinear fractions, and the same eight weights accumulate into a *density* lattice whose
    entry is the number of points that corner saw. The field is then divided by that density, so
    the result is a weighted **mean** rather than a sum and is independent of how many points
    happened to land in a cell.

    This is a **corner lattice** addressed by ``bounds``, the same convention
    [`grid_points`][triwarp.voxels.grid_points] and
    [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] take -- not a
    ``wp.Volume`` and not the voxel-centre addressing the rest of this module uses. That is
    deliberate: the field it produces is meant to be marched, sampled or reshaped, and those three
    all speak ``bounds``.

    Parameters
    ----------
    points
        ``(n_points,)`` positions.
    values
        ``(n_points,)`` field on those points, ``wp.float32`` or ``wp.vec3``.
    shape
        ``(nx, ny, nz)`` lattice size, each at least 1.
    bounds
        ``(lower, upper)`` world corners the lattice spans, so corner ``(0, 0, 0)`` sits at
        ``lower`` and ``(nx-1, ny-1, nz-1)`` at ``upper``. ``None`` (the default) is index space:
        the positions are read as lattice coordinates directly.
    min_weight
        Density floor: the division is by ``max(density, min_weight)``, which is what keeps a
        lattice larger than its point cloud finite instead of full of amplified noise. A corner
        nothing reached comes out zero regardless, since its numerator is zero too.

    Returns
    -------
    field : wp.array[DType, Literal[3]]
        ``(nx, ny, nz)`` averaged field on ``points.device``, zero at every corner no point
        reached.
    density : Array3dFloat32
        ``(nx, ny, nz)`` accumulated trilinear weight per corner -- the point count each corner
        saw. Returned rather than discarded because it *is* the occupancy: a corner at zero was
        reached by nothing, and thresholding it is how a caller separates the reconstructed region
        from the empty one.

    Raises
    ------
    ValueError
        If ``shape`` is not three positive integers, if ``points`` and ``values`` differ in length,
        or if ``min_weight`` is not positive.
    RuntimeError
        If ``points`` and ``values`` are not all on one device.

    See Also
    --------
    [`sample_grid_trilinear`][triwarp.voxels.sample_grid_trilinear]
        The inverse: read a dense lattice back at arbitrary positions, with the same eight weights.
    [`grid_points`][triwarp.voxels.grid_points]
        The positions of the lattice this writes into.
    [`pool_by_voxel`][triwarp.voxels.pool_by_voxel]
        The nearest-corner counterpart over a sparse ``wp.Volume``: one cell per point rather than
        eight, and a ``min`` / ``max`` / ``sum`` choice this has no analogue for.
    [`interpolate_from_points`][triwarp.interpolation.interpolate_from_points]
        The gather-side alternative when the target is an arbitrary sample set rather than a
        lattice, and the source weighting should be Gaussian rather than trilinear.

    Notes
    -----
    Departs from pytorch3d in two ways, both forced. It takes ``bounds`` where pytorch3d takes a
    normalized ``[-1, 1]`` local space plus an ``align_corners`` switch, because every dense
    lattice in triwarp is addressed by ``bounds``; and it *returns* the pair where pytorch3d
    accumulates into caller-supplied ``volume_densities`` / ``volume_features`` in place. The
    ``min_weight`` floor is pytorch3d's, default and all.

    **This averages, so it is not the inverse of a sample except on the lattice itself.** A few
    properties hold exactly and are worth knowing before reading a round trip as a check:

    - ``density.sum()`` is the point count exactly -- the eight weights per point sum to 1.
    - Points *on* the lattice corners give ``density == 1`` everywhere and an exact
      splat-then-sample round trip, because each point's stencil degenerates to its own corner.
    - A **constant** field comes back as that constant everywhere some point was reached. Off the
      lattice the round trip is a smoothing rather than the identity, and the residual comes from
      the corners no point reached: a query whose stencil touches one averages a zero in.
    """
    require_same_device(points=points, values=values)
    if len(shape) != 3 or min(int(n) for n in shape) < 1:
        raise ValueError(f"shape must be three positive integers, got {shape!r}")
    if points.size != values.size:
        raise ValueError(
            f"points and values must have the same length, got {points.size} and {values.size}"
        )
    if min_weight <= 0.0:
        raise ValueError(f"min_weight must be positive, got {min_weight}")
    dims = (int(shape[0]), int(shape[1]), int(shape[2]))
    device = points.device
    field = cast(
        "wp.array[DType, Literal[3]]", _launch.zeros(dims, dtype=values.dtype, device=device)
    )
    density = twt.as_array3d(_launch.zeros(dims, dtype=wp.float32, device=device), wp.float32)
    if points.size == 0:
        return field, density
    lower, inverse_spacing = _lattice_transform(dims, bounds)
    _launch.launch(
        kernel_scatter.SPLAT_GRID_TRILINEAR[values.dtype],
        dim=points.size,
        inputs=[points, values, lower, inverse_spacing, field, density],
        device=device,
    )
    _launch.launch(
        kernel_scatter.DIVIDE_BY_DENSITY[field.dtype],
        dim=dims,
        inputs=[density, wp.float32(min_weight), field],
        device=device,
    )
    return field, density


def sample_grid_trilinear(
    field: wp.array[DType, Literal[3]],
    points: wp.array[wp.vec3],
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
) -> wp.array[DType]:
    """
    Read a dense lattice at arbitrary positions by trilinear interpolation.

    The gather half of the lattice pair, and the exact transpose of
    [`splat_onto_grid`][triwarp.voxels.splat_onto_grid]: the value at a position is the sum of the
    eight surrounding corners weighted by the same trilinear fractions the splat distributes with.
    Use it to read a signed-distance field, a density or a reconstructed vector field back at query
    points; use [`interpolate_from_points`][triwarp.interpolation.interpolate_from_points] instead
    when the source is a scattered cloud rather than a lattice.

    Parameters
    ----------
    field
        ``(nx, ny, nz)`` lattice, ``wp.float32`` or ``wp.vec3``.
    points
        ``(n_points,)`` positions to sample at.
    bounds
        ``(lower, upper)`` world corners the lattice spans, the same convention
        [`grid_points`][triwarp.voxels.grid_points] and
        [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] take. ``None`` (the
        default) is index space.

    Returns
    -------
    wp.array[DType]
        ``(n_points,)`` sampled field on ``points.device``, with ``field``'s dtype.

    Raises
    ------
    ValueError
        If ``field`` is not rank 3.
    RuntimeError
        If ``field`` and ``points`` are not all on one device.

    See Also
    --------
    [`splat_onto_grid`][triwarp.voxels.splat_onto_grid]
        The scatter this transposes: accumulate a scattered field onto the lattice this reads. Its
        ``Notes`` records exactly which round trips through the pair are exact and which are a
        smoothing -- averaging makes the pair adjoint, not inverse.
    [`interpolate_from_points`][triwarp.interpolation.interpolate_from_points]
        The same question with a scattered source and a Gaussian kernel.
    [`occupancy_at_points`][triwarp.voxels.occupancy_at_points]
        The nearest-cell boolean over a sparse ``wp.Volume``, where this is a smooth read of a
        dense one.

    Notes
    -----
    A position outside the lattice reads the nearest boundary cell's stencil rather than a null
    value -- the field is extended by its boundary, which is what a distance or density lattice
    wants and is the convention the Poisson sampler in
    [`triwarp.reconstruction`][triwarp.reconstruction] already had. Clamp or mask the query set
    yourself if an outside position should be an error.
    """
    require_same_device(field=field, points=points)
    if field.ndim != 3:
        raise ValueError(f"field must be a rank-3 lattice, got ndim {field.ndim}")
    dims = (int(field.shape[0]), int(field.shape[1]), int(field.shape[2]))
    values = _launch.empty(points.size, dtype=field.dtype, device=points.device)
    if points.size == 0:
        return values
    lower, inverse_spacing = _lattice_transform(dims, bounds)
    _launch.launch(
        kernel_interpolation.SAMPLE_GRID_TRILINEAR[field.dtype],
        dim=points.size,
        inputs=[field, lower, inverse_spacing, points, values],
        device=points.device,
    )
    return values


def _lattice_transform(
    dims: tuple[int, int, int], bounds: tuple[wp.vec3, wp.vec3] | None
) -> tuple[wp.vec3, wp.vec3]:
    """
    World-to-lattice map as ``(lower, inverse_spacing)``, the pair both grid kernels take.

    Shared by [`splat_onto_grid`][triwarp.voxels.splat_onto_grid] and
    [`sample_grid_trilinear`][triwarp.voxels.sample_grid_trilinear] so the two cannot
    disagree about the convention: they are transposes of each other, and a half-cell disagreement
    between them would move every sampled value by up to one cell. ``bounds=None``
    is index space, i.e. the identity map, matching
    [`grid_points`][triwarp.voxels.grid_points]'s own default.

    A degenerate axis (one sample) has no spacing to invert and is mapped to 0, which puts every
    position on that axis's single slice. That is only half of what makes a single-slice lattice
    safe -- it fixes the *coordinate*, and ``kernels/array.trilinear_cell`` is what keeps the
    stencil's far corner from stepping off the end of that axis.
    """
    if bounds is None:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0)
    lower, upper = bounds
    return wp.vec3(*(float(lower[axis]) for axis in range(3))), wp.vec3(
        *(
            float(dims[axis] - 1) / (float(upper[axis]) - float(lower[axis]))
            if dims[axis] > 1 and float(upper[axis]) != float(lower[axis])
            else 0.0
            for axis in range(3)
        )
    )


def grid_points(
    shape: tuple[int, int, int],
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
    device: wp.DeviceLike = None,
) -> wp.array[wp.vec3]:
    """
    Regular lattice of sample positions spanning a box, flattened in C order.

    ``igl.grid``, and the producer side of the implicit-surface round trip: it takes the same
    ``(lower, upper)`` corner-mapping tuple
    [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] takes, so a
    per-point scalar answer ``.reshape(shape)``s straight back into a field that function accepts.
    This is a **corner lattice**, not a voxel grid: it has nothing to do with the rest of the module
    except that both speak the same ``bounds`` convention.

    Parameters
    ----------
    shape
        ``(nx, ny, nz)`` number of samples per axis, each at least 1.
    bounds
        ``(lower, upper)`` world corners the lattice spans, so sample ``(0, 0, 0)`` sits at
        ``lower`` and sample ``(nx-1, ny-1, nz-1)`` at ``upper``. ``None`` (the default) gives index
        space: the coordinates are the lattice indices, matching
        [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes]'s own
        default.
    device
        Target Warp device.

    Returns
    -------
    wp.array[wp.vec3]
        ``(nx * ny * nz,)`` positions, ``z`` fastest.

    Raises
    ------
    ValueError
        If ``shape`` is not three positive integers.

    See Also
    --------
    [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes]
    [`to_field`][triwarp.voxels.to_field]

    Notes
    -----
    ``igl.voxel_grid`` produces this same lattice up to its own transform: ``s`` cells along the
    longest side, ``pad_count`` cells beyond the box, then an isotropic rescale re-centred on the
    box centre.
    """
    if len(shape) != 3 or min(int(n) for n in shape) < 1:
        raise ValueError(f"shape must be three positive integers, got {shape!r}")
    dims = (int(shape[0]), int(shape[1]), int(shape[2]))
    lower, step = _lattice_step(dims, bounds)
    lattice = _launch.empty(dims, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_voxels.lattice_points, dim=dims, inputs=[lower, step, lattice], device=device
    )
    return lattice.reshape((dims[0] * dims[1] * dims[2],))


def union(a: wp.Volume, b: wp.Volume) -> wp.Volume:
    """
    Cells occupied in either grid: ``a | b``.

    trimesh's ``ops.boolean_sparse`` under ``numpy.logical_or``, and the cheapest of the three set
    operations here -- the two cell arrays are concatenated and the builder collapses the overlap,
    so there is no membership test and no unique pass to write.

    Parameters
    ----------
    a
        Index grid.
    b
        Index grid on the same lattice and device as ``a``.

    Returns
    -------
    wp.Volume
        A new index grid on ``a``'s device.

    Raises
    ------
    ValueError
        If the two grids differ in cell width or in origin.
    TypeError
        If either argument is not a NanoVDB index grid with isotropic voxels.
    RuntimeError
        If ``a`` and ``b`` are not all on one device.

    See Also
    --------
    [`intersection`][triwarp.voxels.intersection]
    [`difference`][triwarp.voxels.difference]
    [`from_cells`][triwarp.voxels.from_cells]

    Notes
    -----
    A cell coordinate means nothing without a lattice, so the three set operations require one:
    two grids share it only when they were built with the same ``voxel_size`` and ``origin``.
    [`resolve_voxel_grid`][triwarp.voxels.resolve_voxel_grid] is what a caller resolves once and
    passes to both builds; [`revoxelize`][triwarp.voxels.revoxelize] moves an existing grid onto
    another lattice.
    """
    require_same_device(a=a, b=b)
    voxel_size, origin = _require_same_lattice(a, b, caller="union")
    rows_a = cells(a)
    rows_b = cells(b)
    n_a = int(rows_a.shape[0])
    n_b = int(rows_b.shape[0])
    both = twt.empty_2d((n_a + n_b, 3), wp.int32, device=rows_a.device)
    if n_a > 0:
        _launch.copy(both[:n_a], rows_a)
    if n_b > 0:
        _launch.copy(both[n_a:], rows_b)
    return from_cells(both, voxel_size, origin)


def intersection(a: wp.Volume, b: wp.Volume) -> wp.Volume:
    """
    Cells occupied in both grids: ``a & b``.

    trimesh's ``ops.boolean_sparse`` under ``numpy.logical_and``. One ``O(1)`` probe of ``b`` per
    voxel of ``a`` -- no dense lattice is materialized, so the cost is set by the *smaller*
    argument when it is passed first.

    Parameters
    ----------
    a
        Index grid; its voxels are the ones tested, so pass the smaller set here.
    b
        Index grid on the same lattice and device as ``a``.

    Returns
    -------
    wp.Volume
        A new index grid on ``a``'s device, empty when the two sets are disjoint.

    Raises
    ------
    ValueError
        If the two grids differ in cell width or in origin.
    TypeError
        If either argument is not a NanoVDB index grid with isotropic voxels.
    RuntimeError
        If ``a`` and ``b`` are not all on one device.

    See Also
    --------
    [`union`][triwarp.voxels.union]
    [`difference`][triwarp.voxels.difference]
    [`occupancy_at_cells`][triwarp.voxels.occupancy_at_cells]
    """
    require_same_device(a=a, b=b)
    voxel_size, origin = _require_same_lattice(a, b, caller="intersection")
    return _select_cells(a, b, voxel_size, origin, present=True)


def difference(a: wp.Volume, b: wp.Volume) -> wp.Volume:
    """
    Cells occupied in ``a`` and not in ``b``: ``a - b``.

    trimesh's ``ops.boolean_sparse`` under ``numpy.logical_and`` of ``a`` with the complement of
    ``b``, and [`intersection`][triwarp.voxels.intersection]'s pass with the membership mask
    negated. Asymmetric: ``difference(a, b)`` and ``difference(b, a)`` are different sets.

    Parameters
    ----------
    a
        Index grid to keep voxels from.
    b
        Index grid on the same lattice and device as ``a``, whose voxels are removed.

    Returns
    -------
    wp.Volume
        A new index grid on ``a``'s device, empty when ``a`` is contained in ``b``.

    Raises
    ------
    ValueError
        If the two grids differ in cell width or in origin.
    TypeError
        If either argument is not a NanoVDB index grid with isotropic voxels.
    RuntimeError
        If ``a`` and ``b`` are not all on one device.

    See Also
    --------
    [`union`][triwarp.voxels.union]
    [`intersection`][triwarp.voxels.intersection]
    [`surface_voxels`][triwarp.voxels.surface_voxels]
        The shell, which is this operation against the grid's own erosion.
    """
    require_same_device(a=a, b=b)
    voxel_size, origin = _require_same_lattice(a, b, caller="difference")
    return _select_cells(a, b, voxel_size, origin, present=False)


def revoxelize(
    grid: wp.Volume, voxel_size: float, *, origin: wp.vec3 | None = None, max_cells: int = 1 << 28
) -> wp.Volume:
    """
    Resample the occupied set onto a lattice of a different cell width.

    trimesh's ``VoxelGrid.revoxelized``, which asks the same question the same way: a cell of the
    new lattice is occupied when **its centre** falls in an occupied cell of the old one. That rule
    is what makes the operation exact at an unchanged ``voxel_size`` and safe when refining, where
    voxelizing the old cell *centres* would instead leave holes between them.

    Parameters
    ----------
    grid
        Index grid to resample.
    voxel_size
        Cell width of the result. Smaller refines, larger coarsens.
    origin
        World position of the lower corner of the new cell ``(0, 0, 0)``. Defaults to ``grid``'s
        own origin, which is what makes a resample at an unchanged ``voxel_size`` land on the
        *identical* lattice -- same cells, same numbering -- so the result composes with
        [`union`][triwarp.voxels.union] and its siblings.
    max_cells
        Budget for the sampling lattice, which covers the occupied box and therefore grows as the
        cube of ``1 / voxel_size``. Exceeding it raises rather than allocating.

    Returns
    -------
    wp.Volume
        A new index grid on ``grid``'s device.

    Raises
    ------
    ValueError
        If ``voxel_size`` or ``max_cells`` is not positive, or the new lattice would exceed
        ``max_cells`` cells.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`voxelize_points`][triwarp.voxels.voxelize_points]
    [`occupancy_at_points`][triwarp.voxels.occupancy_at_points]
    [`union`][triwarp.voxels.union]
        Set algebra, which needs both grids on one lattice -- this is how one gets there.

    Notes
    -----
    Coarsening by centre sampling drops an old voxel whose cell contains no new centre, so it is
    not the same as "any overlap": at ``2 * voxel_size`` a lone voxel survives only if a new centre
    lands inside it. Take [`dilate`][triwarp.voxels.dilate] first when the result must contain the
    input.

    Only the occupied box is sampled, not the whole box between ``origin`` and the set, so a grid
    whose voxels sit far from its own origin costs no more than a tight one. That is what the
    ``origin_cell`` half of [`from_dense`][triwarp.voxels.from_dense] is for.

    An **even integer** coarsening is the one ill-conditioned case: it places every new centre
    exactly on a face of the old lattice, at ``origin + (2c + 1) * old_size``, so which of the two
    neighbouring cells answers is decided by float32 rounding. Pass an ``origin`` displaced by a
    fraction of a cell where that matters; odd and fractional factors sample strictly inside an old
    cell and are exact.
    """
    old_size, old_origin = grid_transform(grid)
    if voxel_size <= 0.0:
        raise ValueError(f"revoxelize requires voxel_size > 0, got {voxel_size}")
    if max_cells <= 0:
        raise ValueError(f"max_cells must be positive, got {max_cells}")
    if origin is None:
        origin = old_origin
    if _voxel_count(grid) == 0:
        return _empty_grid(voxel_size, origin, grid.device)

    # The occupied box in world coordinates, then the half-open cell range of the new lattice that
    # covers it. Anchoring the *cells* rather than the origin is what keeps the sampling tight for
    # a grid whose voxels sit far from its own origin.
    lower_cell, extent = _cell_bounds(grid)
    lower_world = [float(old_origin[axis]) + lower_cell[axis] * old_size for axis in range(3)]
    upper_world = [lower_world[axis] + extent[axis] * old_size for axis in range(3)]
    base_cell = tuple(
        int(np.floor((lower_world[axis] - float(origin[axis])) / voxel_size)) for axis in range(3)
    )
    shape = tuple(
        max(
            1,
            int(np.ceil((upper_world[axis] - float(origin[axis])) / voxel_size)) - base_cell[axis],
        )
        for axis in range(3)
    )
    total = shape[0] * shape[1] * shape[2]
    if total > max_cells:
        raise ValueError(
            f"revoxelize would sample {total} cells, above max_cells={max_cells}: "
            f"voxel_size={voxel_size:.6g} is too small for this grid (the count grows as its "
            "reciprocal cubed)"
        )
    # The lattice is the new cell *centres* -- the sample positions trimesh's ``is_filled`` tests.
    # Each is probed against the old grid and written as a build candidate in the same pass, so
    # neither the centre lattice nor its occupancy is stored.
    half = 0.5 * voxel_size
    dims = (int(shape[0]), int(shape[1]), int(shape[2]))
    lower, step = _lattice_step(
        dims,
        (
            wp.vec3(
                *(float(origin[axis]) + base_cell[axis] * voxel_size + half for axis in range(3))
            ),
            wp.vec3(
                *(
                    float(origin[axis]) + (base_cell[axis] + shape[axis]) * voxel_size - half
                    for axis in range(3)
                )
            ),
        ),
    )
    candidates = twt.empty_2d((total, 3), wp.int32, device=grid.device)
    mask = _launch.empty(total, dtype=wp.int32, device=grid.device)
    dims_i = wp.vec3i(wp.int32(dims[0]), wp.int32(dims[1]), wp.int32(dims[2]))
    base_i = wp.vec3i(*base_cell)
    _launch.launch(
        kernel_voxels.resampled_cell_candidates,
        dim=dims,
        inputs=[grid.id, lower, step, dims_i, base_i, candidates, mask],
        device=grid.device,
    )
    return _allocate_masked(candidates, mask, voxel_size, origin)


def fill_cavities(grid: wp.Volume, *, max_cells: int = 1 << 28) -> wp.Volume:
    """
    Fill every enclosed cavity: an empty cell is kept empty only if it reaches the outside.

    trimesh's ``morphology.fill_holes`` (``scipy.ndimage.binary_fill_holes``) and the default of
    ``VoxelGrid.fill``. The one operation here the sparse grid cannot answer by itself, because it
    is a statement about the *empty* complement, so this densifies over the occupied bounding box
    plus one cell of padding and labels the empty cells' 6-connected components.

    Parameters
    ----------
    grid
        Index grid to fill.
    max_cells
        Budget for the dense box this labels over, which spans the occupied cells and is therefore
        set by how far apart the voxels are rather than by how many there are. Exceeding it raises
        rather than allocating. Five lattices are built over the box, so this bounds the footprint.

    Returns
    -------
    wp.Volume
        A new index grid containing ``grid``'s voxels plus every enclosed empty cell.

    Raises
    ------
    ValueError
        If ``max_cells`` is not positive, or the occupied box exceeds ``max_cells`` cells.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`fill_orthographic`][triwarp.voxels.fill_orthographic]
        The blunter fill, with the same budget for the same reason.
    [`voxelize_mesh`][triwarp.voxels.voxelize_mesh]

    Notes
    -----
    The 6-neighbour stencil is *implicit*: the union-find hooks three backward probes per cell
    straight off the dense occupancy, so no edge list is ever built. An explicit one would cost
    ``3 * n_empty`` rows, which is why the connected-component pass is written against the stencil
    rather than against
    [`triwarp.graph.connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges].
    """
    if max_cells <= 0:
        raise ValueError(f"max_cells must be positive, got {max_cells}")
    voxel_size, origin = grid_transform(grid)
    device = grid.device
    if _voxel_count(grid) == 0:
        return _empty_grid(voxel_size, origin, device)

    occupancy, origin_cell = _to_dense_padded(
        grid, pad=1, max_cells=max_cells, caller="fill_cavities"
    )
    dims = (int(occupancy.shape[0]), int(occupancy.shape[1]), int(occupancy.shape[2]))
    n_nodes = dims[0] * dims[1] * dims[2]

    parents = _launch.empty(n_nodes, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_voxels.flood_init_parent, dim=dims, inputs=[occupancy, parents], device=device
    )
    _launch.launch(kernel_voxels.flood_hook, dim=dims, inputs=[occupancy, parents], device=device)
    labels = _launch.empty(n_nodes, dtype=wp.int32, device=device)
    outside = _launch.zeros(n_nodes, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_voxels.flatten_and_mark_outside,
        dim=dims,
        inputs=[occupancy, parents],
        outputs=[labels, outside],
        device=device,
    )
    # The filled set goes straight to the builder as candidates plus a keep mask -- the pair
    # ``from_dense`` would derive from a filled lattice -- so that lattice is never stored.
    candidates = twt.empty_2d((n_nodes, 3), wp.int32, device=device)
    mask = _launch.empty(n_nodes, dtype=wp.int32, device=device)
    origin_i = wp.vec3i(
        wp.int32(origin_cell[0]), wp.int32(origin_cell[1]), wp.int32(origin_cell[2])
    )
    _launch.launch(
        kernel_voxels.enclosed_cell_candidates,
        dim=dims,
        inputs=[occupancy, labels, outside, origin_i, candidates, mask],
        device=device,
    )
    return _allocate_masked(candidates, mask, voxel_size, origin)


def fill_orthographic(grid: wp.Volume, *, max_cells: int = 1 << 28) -> wp.Volume:
    """
    Fill the intersection of the three axis-aligned solid shadows of the voxel set.

    trimesh's ``ops.fill_orthographic``: along every line parallel to an axis, fill from the first
    occupied cell to the last, then keep only the cells all three axes agree on. Cheaper and blunter
    than [`fill_cavities`][triwarp.voxels.fill_cavities] — it fills concavities that are open in
    only one or two directions, and it cannot fill a cavity whose enclosing shell is not convex
    along all three axes.

    Parameters
    ----------
    grid
        Index grid to fill.
    max_cells
        Budget for the dense box the three axis sweeps run over, which spans the occupied cells and
        is therefore set by how far apart the voxels are rather than by how many there are.
        Exceeding it raises rather than allocating.

    Returns
    -------
    wp.Volume
        A new index grid containing ``grid``'s voxels plus the cells all three axis fills agree on.

    Raises
    ------
    ValueError
        If ``max_cells`` is not positive, or the occupied box exceeds ``max_cells`` cells.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`fill_cavities`][triwarp.voxels.fill_cavities]
        The exact fill, with the same budget for the same reason.
    """
    if max_cells <= 0:
        raise ValueError(f"max_cells must be positive, got {max_cells}")
    voxel_size, origin = grid_transform(grid)
    device = grid.device
    if _voxel_count(grid) == 0:
        return _empty_grid(voxel_size, origin, device)

    occupancy, origin_cell = _to_dense_padded(
        grid, pad=0, max_cells=max_cells, caller="fill_orthographic"
    )
    dims = (int(occupancy.shape[0]), int(occupancy.shape[1]), int(occupancy.shape[2]))
    # The first axis writes every cell of ``filled`` and the other two intersect into it in place.
    filled = twt.empty_3d(dims, wp.bool, device=device)
    for axis in range(3):
        other = [dims[a] for a in range(3) if a != axis]
        _launch.launch(
            kernel_voxels.fill_axis_span,
            dim=(other[0], other[1]),
            inputs=[occupancy, wp.int32(axis), wp.int32(dims[axis]), axis > 0, filled],
            device=device,
        )
    return from_dense(filled, voxel_size, origin, origin_cell=origin_cell)


def dilate(
    grid: wp.Volume, *, connectivity: Literal[6, 18, 26] = 6, iterations: int = 1
) -> wp.Volume:
    """
    Grow the voxel set by one shell of neighbours per iteration.

    trimesh's ``morphology.binary_dilation`` (``scipy.ndimage.binary_dilation``). Every voxel writes
    its whole neighbourhood as candidate cells and the builder dedups them, so there is no unique
    pass — but the candidate buffer is ``(connectivity + 1) * n_voxels`` cells, which grows quickly
    at ``connectivity=26``. That growth is why 6 is the default.

    Parameters
    ----------
    grid
        Index grid to dilate.
    connectivity
        Neighbourhood: ``6`` face-adjacent (the default, and
        ``scipy.ndimage.generate_binary_structure(3, 1)``), ``18`` face and edge, ``26`` the full
        3-cubed shell.
    iterations
        Number of dilation passes. Each is a separate grid build; a widened stencil is deliberately
        not used, since a ``(2r+1)``-cubed structure is strictly more candidate work.

    Returns
    -------
    wp.Volume
        A new index grid on ``grid``'s device.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 6, 18 or 26, or ``iterations`` is negative.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`erode`][triwarp.voxels.erode]
    [`closing`][triwarp.voxels.closing]
    [`surface_voxels`][triwarp.voxels.surface_voxels]

    Notes
    -----
    Binary closing is ``erode(dilate(g))`` and opening ``dilate(erode(g))``; both are wrapped, as
    [`closing`][triwarp.voxels.closing] and [`opening`][triwarp.voxels.opening], because the order
    is the whole operation and each name is the other one's mistake.
    """
    voxel_size, origin = grid_transform(grid)
    _check_iterations(connectivity, iterations)
    device = grid.device
    for _ in range(iterations):
        voxels = cells(grid)
        n_voxels = int(voxels.shape[0])
        if n_voxels == 0:
            return _empty_grid(voxel_size, origin, device)
        stencil = _stencil(connectivity, device, include_self=True)
        n_offsets = int(stencil.shape[0])
        candidates = twt.empty_2d((n_voxels * n_offsets, 3), wp.int32, device=device)
        _launch.launch(
            kernel_voxels.neighborhood_candidates,
            dim=(n_voxels, n_offsets),
            inputs=[voxels, stencil, candidates],
            device=device,
        )
        grid = from_cells(candidates, voxel_size, origin)
    return grid


def erode(
    grid: wp.Volume, *, connectivity: Literal[6, 18, 26] = 6, iterations: int = 1
) -> wp.Volume:
    """
    Shrink the voxel set to the voxels whose whole neighbourhood is occupied.

    trimesh's ``morphology.binary_erosion``. One thread per voxel and ``connectivity`` ``O(1)``
    probes, with no allocation and no sort: neighbours usually live in the same 8-cubed NanoVDB
    leaf, so the probes are cache-local.

    Parameters
    ----------
    grid
        Index grid to erode.
    connectivity
        Neighbourhood: ``6`` (the default), ``18`` or ``26``.
    iterations
        Number of erosion passes.

    Returns
    -------
    wp.Volume
        A new index grid on ``grid``'s device, possibly empty.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 6, 18 or 26, or ``iterations`` is negative.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`dilate`][triwarp.voxels.dilate]
    [`opening`][triwarp.voxels.opening]
    [`surface_voxels`][triwarp.voxels.surface_voxels]

    Notes
    -----
    Cells outside the grid are empty by definition, so a voxel on the boundary of the occupied
    region always erodes away — the same convention ``scipy.ndimage.binary_erosion`` takes with
    ``border_value=0``.
    """
    voxel_size, origin = grid_transform(grid)
    _check_iterations(connectivity, iterations)
    for _ in range(iterations):
        interior = _interior_flags(grid, connectivity)
        if interior is None:
            return _empty_grid(voxel_size, origin, grid.device)
        grid = _grid_from_flagged_cells(grid, interior)
    return grid


def closing(
    grid: wp.Volume, *, connectivity: Literal[6, 18, 26] = 6, iterations: int = 1
) -> wp.Volume:
    """
    Dilation followed by erosion: bridge gaps and cracks thinner than the structuring element.

    trimesh's ``morphology.binary_closing`` (``scipy.ndimage.binary_closing``), and exactly
    ``erode(dilate(grid))`` at the same settings. It is a name rather than a recipe because the
    order is the whole operation and the wrong one is the *other* function here.

    Parameters
    ----------
    grid
        Index grid to close.
    connectivity
        Neighbourhood of the structuring element: ``6`` (the default), ``18`` or ``26``.
    iterations
        Cells of reach: the dilation runs ``iterations`` times and the erosion undoes exactly as
        many, so the set's extent is unchanged and only gaps up to that width close.

    Returns
    -------
    wp.Volume
        A new index grid on ``grid``'s device, containing ``grid``.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 6, 18 or 26, or ``iterations`` is negative.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`opening`][triwarp.voxels.opening]
    [`dilate`][triwarp.voxels.dilate]
    [`erode`][triwarp.voxels.erode]
    [`fill_cavities`][triwarp.voxels.fill_cavities]
        The unbounded form: it closes an enclosed void of any size, where this one closes a gap of
        at most ``iterations`` cells.

    Notes
    -----
    Nothing is clipped here, which is the one way this differs from the dense reference: the
    intermediate dilation grows onto whatever cells it needs, where ``scipy.ndimage``'s runs inside
    the array it was handed and erodes a shell off every face of it unless the caller padded first.
    """
    return erode(
        dilate(grid, connectivity=connectivity, iterations=iterations),
        connectivity=connectivity,
        iterations=iterations,
    )


def opening(
    grid: wp.Volume, *, connectivity: Literal[6, 18, 26] = 6, iterations: int = 1
) -> wp.Volume:
    """
    Erosion followed by dilation: drop specks and necks thinner than the structuring element.

    ``scipy.ndimage.binary_opening``, and exactly ``dilate(erode(grid))`` at the same settings --
    the dual of [`closing`][triwarp.voxels.closing], and the one that removes rather than adds.

    Parameters
    ----------
    grid
        Index grid to open.
    connectivity
        Neighbourhood of the structuring element: ``6`` (the default), ``18`` or ``26``.
    iterations
        Cells of reach: a feature that survives ``iterations`` erosions is restored, and one that
        does not is gone.

    Returns
    -------
    wp.Volume
        A new index grid on ``grid``'s device, contained in ``grid``.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 6, 18 or 26, or ``iterations`` is negative.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`closing`][triwarp.voxels.closing]
    [`erode`][triwarp.voxels.erode]
    [`dilate`][triwarp.voxels.dilate]

    Notes
    -----
    A component narrower than the element vanishes entirely rather than shrinking, which is what
    makes this the speck filter of a noisy voxelization; the surviving components keep their
    extent but lose their corners.
    """
    return dilate(
        erode(grid, connectivity=connectivity, iterations=iterations),
        connectivity=connectivity,
        iterations=iterations,
    )


def surface_voxels(grid: wp.Volume, *, connectivity: Literal[6, 18, 26] = 6) -> wp.Volume:
    """
    Keep the occupied voxels that touch an empty one: the boundary shell of the set.

    trimesh's ``VoxelGrid.surface``, i.e. ``grid`` minus [`erode`][triwarp.voxels.erode]``(grid)``,
    computed with the same probes and one launch.

    Parameters
    ----------
    grid
        Index grid to read.
    connectivity
        Neighbourhood deciding adjacency: ``6`` (the default), ``18`` or ``26``.

    Returns
    -------
    wp.Volume
        A new index grid holding only the boundary voxels.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 6, 18 or 26.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`erode`][triwarp.voxels.erode]
    [`to_boxes`][triwarp.voxels.to_boxes]

    Notes
    -----
    A voxel on the edge of the grid's occupied region counts as surface, since everything outside
    the set is empty by definition.
    """
    voxel_size, origin = grid_transform(grid)
    # ``interior=False`` asks the neighbourhood kernel for the complement it already computes,
    # rather than launching a second pass to negate the flags it just wrote.
    boundary = _interior_flags(grid, connectivity, interior=False)
    if boundary is None:
        return _empty_grid(voxel_size, origin, grid.device)
    return _grid_from_flagged_cells(grid, boundary)


def to_dense(
    grid: wp.Volume,
    *,
    origin_cell: tuple[int, int, int] | None = None,
    shape: tuple[int, int, int] | None = None,
) -> tuple[twt.Array3dBool, tuple[int, int, int]]:
    """
    Occupancy of a cell box as a dense boolean lattice.

    trimesh's ``sparse_to_matrix``, plus the cell offset that makes it invertible: unlike trimesh's
    grids, cell coordinates here can be negative, so the lattice's own origin has to be returned
    alongside it.

    Parameters
    ----------
    grid
        Index grid to read.
    origin_cell
        Cell that becomes ``occupancy[0, 0, 0]``. Defaults to the component-wise minimum of the
        occupied cells, i.e. the tight bounding box.
    shape
        ``(nx, ny, nz)`` extent of the lattice in cells. Defaults to the tight bounding box of the
        occupied cells, measured from ``origin_cell``.

    Returns
    -------
    occupancy : Array3dBool
        ``(nx, ny, nz)`` occupancy on ``grid``'s device.
    origin_cell : tuple[int, int, int]
        The cell at ``occupancy[0, 0, 0]``, to hand back to
        [`from_dense`][triwarp.voxels.from_dense].

    Raises
    ------
    ValueError
        If ``shape`` is not three positive integers.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`from_dense`][triwarp.voxels.from_dense]
    [`to_field`][triwarp.voxels.to_field]
    """
    _require_index_grid(grid)
    if origin_cell is None or shape is None:
        bounding_lower, bounding_shape = _cell_bounds(grid)
        if origin_cell is None:
            origin_cell = bounding_lower
        if shape is None:
            shape = (
                bounding_shape[0] + bounding_lower[0] - origin_cell[0],
                bounding_shape[1] + bounding_lower[1] - origin_cell[1],
                bounding_shape[2] + bounding_lower[2] - origin_cell[2],
            )
    dims = (int(shape[0]), int(shape[1]), int(shape[2]))
    if len(dims) != 3 or min(dims) < 1:
        raise ValueError(f"shape must be three positive integers, got {shape!r}")
    occupancy = twt.empty_3d(dims, wp.bool, device=grid.device)
    _launch.launch(
        kernel_voxels.dense_occupancy,
        dim=dims,
        inputs=[grid.id, wp.vec3i(*(int(c) for c in origin_cell)), occupancy],
        device=grid.device,
    )
    return occupancy, (int(origin_cell[0]), int(origin_cell[1]), int(origin_cell[2]))


def from_dense(
    occupancy: twt.Array3dBool,
    voxel_size: float,
    origin: wp.vec3,
    *,
    origin_cell: tuple[int, int, int] = (0, 0, 0),
) -> wp.Volume:
    """
    Build an index grid from a dense boolean occupancy lattice.

    trimesh's ``DenseEncoding``, and the inverse of [`to_dense`][triwarp.voxels.to_dense]: pass back
    the ``origin_cell`` that function returned and the round trip is exact, negative cells included.

    Parameters
    ----------
    occupancy
        ``(nx, ny, nz)`` lattice; ``True`` marks an occupied cell.
    voxel_size
        Cell width.
    origin
        World position of the lower corner of cell ``(0, 0, 0)``.
    origin_cell
        Cell coordinate of ``occupancy[0, 0, 0]``.

    Returns
    -------
    wp.Volume
        A NanoVDB index grid on ``occupancy``'s device, empty when nothing is occupied.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive.
    TypeError
        If ``occupancy`` is not a rank-3 ``wp.bool`` array.

    See Also
    --------
    [`to_dense`][triwarp.voxels.to_dense]
    [`from_cells`][triwarp.voxels.from_cells]
    """
    occupancy = twt.as_array3d(occupancy, wp.bool)
    if voxel_size <= 0.0:
        raise ValueError(f"from_dense requires voxel_size > 0, got {voxel_size}")
    device = occupancy.device
    dims = (int(occupancy.shape[0]), int(occupancy.shape[1]), int(occupancy.shape[2]))
    n_cells = dims[0] * dims[1] * dims[2]
    if n_cells == 0:
        return _empty_grid(voxel_size, origin, device)

    candidates = twt.empty_2d((n_cells, 3), wp.int32, device=device)
    mask = _launch.empty(n_cells, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_voxels.occupied_cells,
        dim=dims,
        inputs=[occupancy, wp.vec3i(*(int(c) for c in origin_cell)), candidates, mask],
        device=device,
    )
    return _allocate_masked(candidates, mask, voxel_size, origin)


def to_field(
    grid: wp.Volume, *, pad: int = 1
) -> tuple[twt.Array3dFloat32, tuple[wp.vec3, wp.vec3]]:
    """
    Occupancy as a padded ``float32`` lattice on the cell centres, with the box it spans.

    The bridge to [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes]:
    ``marching_cubes(*to_field(grid), 0.5)`` reproduces trimesh's ``ops.matrix_to_marching_cubes``,
    ``VoxelGrid.marching_cubes`` and ``ops.points_to_marching_cubes``, which are all the same recipe
    (pad by one, threshold at ``0.5``). The padding is what closes the surface: without it a voxel
    on the lattice boundary has no zero outside it to cross.

    Parameters
    ----------
    grid
        Index grid to read.
    pad
        Empty cells of margin added on every side.

    Returns
    -------
    field : Array3dFloat32
        ``(nx, ny, nz)`` lattice, ``1.0`` inside an occupied voxel and ``0.0`` outside.
    bounds : tuple[wp.vec3, wp.vec3]
        ``(lower, upper)`` world corners the lattice spans, i.e. the centres of its first and last
        cells — exactly the tuple
        [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] takes.

    Raises
    ------
    ValueError
        If ``pad`` is negative.
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes]
    [`to_dense`][triwarp.voxels.to_dense]
    [`to_boxes`][triwarp.voxels.to_boxes]

    Notes
    -----
    Because the samples sit on cell *centres*, the extracted surface runs half a voxel inside the
    occupied cells' outer faces — the same half-voxel offset trimesh's marching-cubes path has.
    """
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")
    voxel_size, origin = grid_transform(grid)
    lower_cell, extent = _cell_bounds(grid)
    base = (lower_cell[0] - pad, lower_cell[1] - pad, lower_cell[2] - pad)
    dims = (extent[0] + 2 * pad, extent[1] + 2 * pad, extent[2] + 2 * pad)
    field = twt.empty_3d(dims, wp.float32, device=grid.device)
    _launch.launch(
        kernel_voxels.dense_field,
        dim=dims,
        inputs=[grid.id, wp.vec3i(*(int(c) for c in base)), field],
        device=grid.device,
    )
    origin_f = twt.vec3_floats(origin)
    lower = wp.vec3(*(origin_f[axis] + (base[axis] + 0.5) * voxel_size for axis in range(3)))
    upper = wp.vec3(
        *(origin_f[axis] + (base[axis] + dims[axis] - 0.5) * voxel_size for axis in range(3))
    )
    return twt.as_array3d(field, wp.float32), (lower, upper)


def to_boxes(
    grid: wp.Volume, *, cull_internal: bool = True
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Triangulate the voxel set as a mesh of axis-aligned cubes.

    trimesh's ``ops.multibox`` / ``VoxelGrid.as_boxes``, and the cube rendering Open3D draws a
    ``VoxelGrid`` as. Corners are shared rather than duplicated — the deduplicated corner lattice
    comes from [`voxel_corners`][triwarp.voxels.voxel_corners] — so no welding pass runs afterwards.

    Parameters
    ----------
    grid
        Index grid to mesh.
    cull_internal
        Drop the faces between two occupied voxels (the default), leaving only the visible shell.
        ``False`` emits all six faces of every voxel, which is what ``multibox`` builds.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_corners,)`` corner positions on ``grid``'s device, every one of them referenced by a
        face.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer, wound so that normals point away from the
        voxel they belong to.

    Raises
    ------
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`voxel_corners`][triwarp.voxels.voxel_corners]
    [`to_field`][triwarp.voxels.to_field]
    [`surface_voxels`][triwarp.voxels.surface_voxels]

    Notes
    -----
    With ``cull_internal=True`` on a solid set the result is closed and manifold; with ``False`` it
    is not, since interior faces are duplicated back to back.

    The corner lattice is shared rather than duplicated, so culling the interior faces leaves the
    interior *corners* unreferenced -- they belong to the lattice but to no surviving face. Those
    are compacted away before returning, which is a change worth knowing about if a caller was
    relying on an index into the full lattice: it was not, since the lattice itself is
    [`voxel_corners`][triwarp.voxels.voxel_corners] and that is where an index into it comes from.
    """
    _require_index_grid(grid)
    device = grid.device
    corner_cells, cell_corners = voxel_corners(grid)
    n_corners = int(corner_cells.shape[0])
    n_voxels = int(cell_corners.shape[0])
    vertices = _launch.empty(n_corners, dtype=wp.vec3, device=device)
    if n_voxels == 0:
        return vertices, _launch.empty(0, dtype=wp.int32, device=device)
    voxels = cells(grid)
    neighbors = _face_neighbors(device)
    counts = _launch.empty(n_voxels, dtype=wp.int32, device=device)
    # The corner positions ride the face-count launch.
    _launch.launch(
        kernel_voxels.count_box_faces,
        dim=max(n_corners, n_voxels),
        inputs=[grid.id, voxels, neighbors, cull_internal, corner_cells],
        outputs=[counts, vertices],
        device=device,
    )
    offsets, n_quads = tw.array.counts_to_offsets(counts)
    faces = _launch.empty(n_quads * 6, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_voxels.emit_box_faces,
        dim=n_voxels,
        inputs=[
            grid.id,
            voxels,
            cell_corners,
            neighbors,
            _face_corner_table(device),
            offsets,
            cull_internal,
            faces,
        ],
        device=device,
    )
    if not cull_internal:
        # Every corner of every voxel is referenced by six faces, so there is nothing to compact
        # and the buffers are returned exactly as the emit kernels wrote them.
        return vertices, faces
    compacted_vertices, compacted_faces, _ = tw.repair.remove_unreferenced_vertices(vertices, faces)
    return compacted_vertices, compacted_faces


def voxel_corners(grid: wp.Volume) -> tuple[twt.Array2dInt32, twt.Array2dInt32]:
    """
    Build the deduplicated corner lattice of the voxel set, plus each voxel's eight corners.

    ``igl.unique_sparse_voxel_corners``. The lattice is not computed here: ``warp.fem``'s sparse
    nanogrid geometry derives its own vertex grid from the cell grid, and that grid *is* the
    deduplicated corner set — so this is eight ``O(1)`` probes per voxel and no ``8 * n_voxels``
    candidate buffer, no unique pass, and no hand-built dual grid.

    Corner ``(i, j, k)`` is the **lower** corner of cell ``(i, j, k)``, at world position
    ``origin + (i, j, k) * voxel_size``.

    Parameters
    ----------
    grid
        Index grid to read.

    Returns
    -------
    corners : Array2dInt32
        ``(n_corners, 3)`` corner coordinates, in the corner grid's own leaf-major order.
    cell_corners : Array2dInt32
        ``(n_voxels, 8)`` indices into ``corners``, one row per voxel of
        [`cells`][triwarp.voxels.cells], ordered by binary counting on ``(dx, dy, dz)`` with ``dx``
        the most significant bit.

    Raises
    ------
    TypeError
        If ``grid`` is not a NanoVDB index grid with isotropic voxels.

    See Also
    --------
    [`to_boxes`][triwarp.voxels.to_boxes]
    [`cells`][triwarp.voxels.cells]

    Notes
    -----
    igl numbers a cell's corners in ``yxz`` binary-counting order, a fixed permutation of the order
    used here.

    ``warp.fem`` is imported inside this function, not at module scope, since nothing else in this
    module needs it.
    """
    _require_index_grid(grid)
    device = grid.device
    voxels = cells(grid)
    n_voxels = int(voxels.shape[0])
    if n_voxels == 0:
        return twt.empty_2d((0, 3), wp.int32, device=device), twt.empty_2d(
            (0, 8), wp.int32, device=device
        )

    # Deferred: importing ``warp.fem`` here, not at module scope, since only this path needs it.
    import warp.fem as fem

    corner_grid = fem.Nanogrid(grid).vertex_grid
    n_corners = int(corner_grid.get_active_stats().voxel_count)
    corner_cells = twt.as_array2d(twt.as_dense(corner_grid.get_voxels()[:n_corners]), wp.int32)
    cell_corners = twt.empty_2d((n_voxels, 8), wp.int32, device=device)
    _launch.launch(
        kernel_voxels.cell_corner_indices,
        dim=n_voxels,
        inputs=[corner_grid.id, voxels, cell_corners],
        device=device,
    )
    return corner_cells, cell_corners


def _require_index_grid(grid: wp.Volume) -> None:
    """
    Validate that ``grid`` is a NanoVDB index grid with isotropic voxels.

    Separate from [`grid_transform`][triwarp.voxels.grid_transform], which reads the transform
    *through* it, because eight entry points here want the guard and not the pair -- and a bare
    ``grid_transform(grid)`` on a line of its own would read as a discarded computation rather than
    a check, which is how a type guard gets deleted as dead code.
    """
    if not grid.is_index:
        raise TypeError(
            "voxels functions need a NanoVDB *index* grid, whose linear indices address a "
            "per-voxel payload; build one with triwarp.voxels.from_cells"
        )
    sizes = grid.get_voxel_size()
    if not (sizes[0] == sizes[1] == sizes[2]):
        raise TypeError(f"voxels functions need isotropic voxels, got voxel_size={tuple(sizes)}")


def _require_same_lattice(a: wp.Volume, b: wp.Volume, *, caller: str) -> tuple[float, wp.vec3]:
    """Shared cell width and origin of two grids, or a ``ValueError`` naming both transforms."""
    size_a, origin_a = grid_transform(a)
    size_b, origin_b = grid_transform(b)
    tolerance = 1e-4 * size_a
    offset = max(abs(float(origin_a[axis]) - float(origin_b[axis])) for axis in range(3))
    if abs(size_a - size_b) > tolerance or offset > tolerance:
        raise ValueError(
            f"{caller} needs both grids on one lattice, got voxel_size {size_a:.6g} and "
            f"{size_b:.6g} at origins {tuple(float(x) for x in origin_a)} and "
            f"{tuple(float(x) for x in origin_b)}; revoxelize one onto the other's lattice first"
        )
    return size_a, origin_a


def _select_cells(
    a: wp.Volume, b: wp.Volume, voxel_size: float, origin: wp.vec3, *, present: bool
) -> wp.Volume:
    """Voxels of ``a`` whose membership in ``b`` is ``present``: intersection, or difference."""
    rows = cells(a)
    n_cells = int(rows.shape[0])
    if n_cells == 0:
        return _empty_grid(voxel_size, origin, rows.device)
    # ``present`` rides into the kernel rather than inverting its answer afterwards: membership
    # and its complement are the same probe, so the difference costs a selector instead of a
    # second launch over a mask the first one just wrote.
    #
    # The flags go to the builder as its ``point_mask`` over the unfiltered rows, which is what the
    # kept rows would have been compacted for: no scan, count readback or gathered copy.
    flags = _launch.empty(n_cells, dtype=wp.int32, device=rows.device)
    _launch.launch(
        kernel_voxels.cell_occupancy_flags,
        dim=n_cells,
        inputs=[b.id, rows, present, flags],
        device=rows.device,
    )
    return _allocate_masked(rows, flags, voxel_size, origin)


def _voxel_count(grid: wp.Volume) -> int:
    """Count the *active* voxels: ``get_voxel_count`` reports capacity instead (see `cells`)."""
    return int(grid.get_active_stats().voxel_count)


def _translation(origin: wp.vec3, voxel_size: float) -> tuple[float, float, float]:
    """NanoVDB centres voxels on integers, so the volume's translation is half a cell above."""
    half = 0.5 * voxel_size
    return (float(origin[0]) + half, float(origin[1]) + half, float(origin[2]) + half)


def _lattice_step(
    dims: tuple[int, int, int], bounds: tuple[wp.vec3, wp.vec3] | None
) -> tuple[wp.vec3, wp.vec3]:
    """
    Node 0 and the node spacing of a ``dims`` lattice spanning ``bounds`` inclusively.

    The ``(lower, step)`` pair ``kernels/array.lattice_position`` takes; shared by
    [`grid_points`][triwarp.voxels.grid_points] and [`revoxelize`][triwarp.voxels.revoxelize],
    whose resampling lattice is the same construction without the stored positions. A single-sample
    axis has no spacing and steps by 0. ``bounds=None`` is index space.
    """
    if bounds is None:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0)
    lower, upper = bounds
    lower_f, upper_f = twt.vec3_floats(lower), twt.vec3_floats(upper)
    step = wp.vec3(
        *(
            (upper_f[axis] - lower_f[axis]) / float(dims[axis] - 1) if dims[axis] > 1 else 0.0
            for axis in range(3)
        )
    )
    return lower, step


def _allocate_masked(
    candidates: twt.Array2dInt32, mask: wp.array[wp.int32], voxel_size: float, origin: wp.vec3
) -> wp.Volume:
    """
    Build an index grid from the candidate cells whose ``mask`` flag is non-zero.

    ``point_mask`` lets the builder skip the rejected rows in place, so a producer that already
    knows which of its rows to keep never compacts them. ``candidates`` must be contiguous and
    non-empty; an all-zero ``mask`` builds a legal empty grid.
    """
    return wp.Volume.allocate_by_voxels(
        cast("wp.array[wp.int32]", candidates),
        voxel_size=voxel_size,
        translation=_translation(origin, voxel_size),
        point_mask=mask,
        device=candidates.device,
    )


def _empty_grid(voxel_size: float, origin: wp.vec3, device: wp.DeviceLike) -> wp.Volume:
    """
    Build an index grid with the given transform and no active voxels.

    ``allocate_by_voxels`` raises ``Failed to create volume`` on a zero-length point set, but a
    one-point set that ``point_mask`` rejects builds a legal empty topology.
    """
    return wp.Volume.allocate_by_voxels(
        _launch.zeros((1, 3), dtype=wp.int32, device=device),
        voxel_size=voxel_size,
        translation=_translation(origin, voxel_size),
        point_mask=_launch.zeros(1, dtype=wp.int32, device=device),
        device=device,
    )


# Two members of this trailing block have a single caller -- ``_face_neighbors`` and
# ``_face_corner_table`` (both ``to_boxes``) -- and both stay here rather than moving up to their
# caller. They are one half of a pair whose other half *is* cross-cutting -- the constant tables
# ``_stencil`` above them builds from -- and splitting a pair across the file to save a backward
# jump is the worse trade.


def _cell_bounds(grid: wp.Volume) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Tight cell bounding box of the occupied voxels, as ``(lower_cell, extent)``."""
    voxels = cells(grid)
    if int(voxels.shape[0]) == 0:
        return (0, 0, 0), (1, 1, 1)
    # One readback of six integers: the dense lattice has to be sized on the host.
    lower_wp, upper_wp = tw.reduce.minmax(voxels, axis=0)
    lower = lower_wp.numpy()
    upper = upper_wp.numpy()
    return (
        (int(lower[0]), int(lower[1]), int(lower[2])),
        (int(upper[0] - lower[0]) + 1, int(upper[1] - lower[1]) + 1, int(upper[2] - lower[2]) + 1),
    )


def _to_dense_padded(
    grid: wp.Volume, *, pad: int, max_cells: int, caller: str
) -> tuple[twt.Array3dBool, tuple[int, int, int]]:
    """
    Dense occupancy of the tight cell box grown by ``pad`` empty cells on every side.

    ``max_cells`` bounds the box before it is allocated, the way
    [`revoxelize`][triwarp.voxels.revoxelize] bounds its sampling lattice. The guard belongs here
    rather than at the two callers because the quantity it bounds is this box, and because a
    caller's own argument does not bound it: the box is the *bounding box* of the cell set and has
    no relation to the voxel count, so a two-voxel grid whose cells sit far apart densifies to
    whatever their separation cubed happens to be. Each caller then allocates several more lattices
    over the same box, so this count is the one that decides the footprint.
    """
    lower_cell, extent = _cell_bounds(grid)
    base = (lower_cell[0] - pad, lower_cell[1] - pad, lower_cell[2] - pad)
    shape = (extent[0] + 2 * pad, extent[1] + 2 * pad, extent[2] + 2 * pad)
    total = shape[0] * shape[1] * shape[2]
    if total > max_cells:
        raise ValueError(
            f"{caller} would densify {total} cells, above max_cells={max_cells}: the occupied box "
            f"is {shape[0]}x{shape[1]}x{shape[2]} cells, which is set by how far apart the voxels "
            "are and not by how many there are"
        )
    return to_dense(grid, origin_cell=base, shape=shape)


def _check_iterations(connectivity: int, iterations: int) -> None:
    """Shared argument validation for the two morphology passes."""
    if connectivity not in _CONNECTIVITY_RANK:
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity!r}")
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")


def _interior_flags(
    grid: wp.Volume, connectivity: int, *, interior: bool = True
) -> wp.array[wp.int32] | None:
    """
    Per-voxel 1 / 0 flag: is the whole neighbourhood occupied? ``None`` for an empty grid.

    ``interior=False`` returns the complement -- the surface set -- which the kernel writes
    directly, so the two callers differ by a selector rather than by a pass.
    """
    if connectivity not in _CONNECTIVITY_RANK:
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity!r}")
    voxels = cells(grid)
    n_voxels = int(voxels.shape[0])
    if n_voxels == 0:
        return None
    flags = _launch.empty(n_voxels, dtype=wp.int32, device=grid.device)
    _launch.launch(
        kernel_voxels.neighborhood_complete,
        dim=n_voxels,
        inputs=[
            grid.id,
            voxels,
            _stencil(connectivity, grid.device, include_self=False),
            interior,
            flags,
        ],
        device=grid.device,
    )
    return flags


def _grid_from_flagged_cells(grid: wp.Volume, flags: wp.array[wp.int32]) -> wp.Volume:
    """Rebuild ``grid`` keeping only the voxels whose flag is non-zero."""
    voxel_size, origin = grid_transform(grid)
    return _allocate_masked(cells(grid), flags, voxel_size, origin)


def _stencil(connectivity: int, device: wp.DeviceLike, *, include_self: bool) -> twt.Array2dInt32:
    """Return the ``connectivity`` neighbour offsets, cached per device (6 to 27 rows)."""
    key = (str(device), connectivity, include_self)
    cached = _STENCIL_CACHE.get(key)
    if cached is not None:
        return cached
    rank = _CONNECTIVITY_RANK[connectivity]
    offsets = [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (abs(i) + abs(j) + abs(k) <= rank) and (include_self or (i, j, k) != (0, 0, 0))
    ]
    stencil = twt.as_array2d(
        _launch.array(np.array(offsets, dtype=np.int32), dtype=wp.int32, device=device), wp.int32
    )
    _STENCIL_CACHE[key] = stencil
    return stencil


def _face_neighbors(device: wp.DeviceLike) -> twt.Array2dInt32:
    """Return the six axis directions, in the order ``_FACE_CORNERS`` is written against."""
    key = (str(device), 0, False)
    cached = _STENCIL_CACHE.get(key)
    if cached is None:
        cached = twt.as_array2d(
            _launch.array(_FACE_NEIGHBORS, dtype=wp.int32, device=device), wp.int32
        )
        _STENCIL_CACHE[key] = cached
    return cached


def _face_corner_table(device: wp.DeviceLike) -> twt.Array2dInt32:
    """Return the four local corner indices of each cube face, wound outward."""
    key = (str(device), 1, False)
    cached = _STENCIL_CACHE.get(key)
    if cached is None:
        cached = twt.as_array2d(
            _launch.array(_FACE_CORNERS, dtype=wp.int32, device=device), wp.int32
        )
        _STENCIL_CACHE[key] = cached
    return cached
